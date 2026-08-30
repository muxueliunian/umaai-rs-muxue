"""从按 rollout 序号对齐的原始分数生成 policy/value 标签。

默认 policy 标签是配对 Bayesian bootstrap 下“每个候选为最优”的概率。
同一次 bootstrap 对所有候选使用同一组 rollout 权重，因此保留 CRN 的配对降噪；
精确并列的候选平分该次抽样的胜者质量。

value 标签使用 leave-one-rollout-out cross-fitting：第 k 个随机世界只负责估值，
动作由其余 R-1 个随机世界选择。遍历全部 k 后仍得到 R 个互相对齐的终局分，
用它们计算均值、样本标准差和排名加权均值，避免把选择噪声写成乐观 value。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

INPUT_DIM = 754
POLICY_DIM = 234


@dataclass(frozen=True)
class LabelConfig:
    """可复现的标签生成配置。"""

    bootstrap_draws: int = 512
    bootstrap_seed: int = 20260830
    radical_factor: float = 1.4
    policy_temperature: float = 1.0
    tie_atol: float = 1e-5

    def validate(self) -> None:
        """校验配置，拒绝会产生空标签或反向偏好的值。"""

        if self.bootstrap_draws < 16:
            raise ValueError("bootstrap_draws 至少为 16")
        if self.radical_factor < 0.0:
            raise ValueError("radical_factor 不得为负")
        if self.policy_temperature <= 0.0:
            raise ValueError("policy_temperature 必须为正")
        if self.tie_atol < 0.0:
            raise ValueError("tie_atol 不得为负")


def make_bayesian_weights(rollouts: int, draws: int, seed: int) -> np.ndarray:
    """生成 Bayesian bootstrap 权重，列和恒为 1。

    指数分布归一化等价于 Dirichlet(1, ..., 1)。同一矩阵会用于所有样本；
    rollout 序号在不同根局面间可交换，复用矩阵不改变每条标签的边缘分布，
    还能避免为每条样本生成数十万随机数。
    """

    if rollouts < 2:
        raise ValueError("至少需要 2 个 rollout")
    rng = np.random.default_rng(seed)
    weights = rng.standard_exponential((rollouts, draws), dtype=np.float32)
    weights /= weights.sum(axis=0, keepdims=True)
    return weights


def bayesian_best_probabilities(
    scores: np.ndarray,
    weights: np.ndarray,
    valid: np.ndarray | None = None,
    tie_atol: float = 1e-5,
    temperature: float = 1.0,
) -> np.ndarray:
    """计算候选在配对 Bayesian bootstrap 中成为最优的概率。

    ``scores`` 为 ``[candidate, rollout]``，``weights`` 为
    ``[rollout, draw]``。失败槽位通过 ``valid`` 排除并按候选重新归一化；
    这仍共享同一组基础权重，但失败过多的数据应在采集侧优先剔除。
    """

    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("scores 必须是非空二维数组")
    if weights.shape[0] != values.shape[1]:
        raise ValueError(f"weights rollout 维 {weights.shape[0]} != scores {values.shape[1]}")
    if not np.isfinite(values).all():
        raise ValueError("scores 含非有限值")

    if valid is None:
        boot_mean = values @ weights
    else:
        mask = np.asarray(valid, dtype=np.float32)
        if mask.shape != values.shape:
            raise ValueError("valid 与 scores 形状不一致")
        denom = mask @ weights
        if np.any(denom <= 0.0):
            raise ValueError("某候选在某次 bootstrap 中没有有效 rollout")
        boot_mean = (values * mask) @ weights / denom

    best = boot_mean.max(axis=0, keepdims=True)
    winners = np.isclose(boot_mean, best, rtol=0.0, atol=tie_atol)
    shares = winners / winners.sum(axis=0, keepdims=True)
    probs = shares.mean(axis=1, dtype=np.float64)

    if temperature != 1.0:
        positive = probs > 0.0
        probs[positive] = np.power(probs[positive], 1.0 / temperature)
        probs /= probs.sum()
    return probs.astype(np.float32)


def candidate_probs_to_policy(
    candidate_probs: np.ndarray,
    candidate_slots: np.ndarray,
    policy_dim: int = POLICY_DIM,
) -> np.ndarray:
    """把候选概率投影到冻结的 policy 格位。

    普通候选占一格；地区组合占三格，因此把该组合的概率各分三分之一。
    这等价于对组合分布求地区边缘后再除以 3，使最终标签和仍为 1。
    """

    probs = np.asarray(candidate_probs, dtype=np.float64)
    slots = np.asarray(candidate_slots)
    if probs.ndim != 1 or slots.shape != (probs.size, 3):
        raise ValueError("candidate_probs/candidate_slots 形状不匹配")
    if np.any(probs < 0.0) or not np.isfinite(probs).all():
        raise ValueError("candidate_probs 必须是有限的非负数")
    if not np.isclose(probs.sum(), 1.0, atol=1e-5):
        raise ValueError(f"candidate_probs 和应为 1，实得 {probs.sum()}")

    target = np.zeros(policy_dim, dtype=np.float64)
    for prob, row in zip(probs, slots, strict=True):
        occupied = row[row >= 0]
        if occupied.size not in (1, 3):
            raise ValueError(f"候选必须占 1 或 3 格，实得 {row.tolist()}")
        if np.any(occupied >= policy_dim) or np.unique(occupied).size != occupied.size:
            raise ValueError(f"候选格位越界或重复: {row.tolist()}")
        target[occupied] += prob / occupied.size
    if not np.isclose(target.sum(), 1.0, atol=1e-5):
        raise RuntimeError(f"policy target 和不是 1: {target.sum()}")
    return target.astype(np.float32)


def leave_one_out_outcomes(
    scores: np.ndarray,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """用其余 rollout 选候选，在留出的同列随机世界上估值。

    返回 ``(outcomes, selected_candidate)``。存在失败槽时，只允许选择在当前
    留出列有效的候选；若某列所有候选都失败则丢弃该列。
    """

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("scores 必须是 [非空候选, 至少2个rollout]")
    if valid is None:
        mask = np.ones(values.shape, dtype=bool)
    else:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("valid 与 scores 形状不一致")

    masked = np.where(mask, values, 0.0)
    totals = masked.sum(axis=1, keepdims=True)
    counts = mask.sum(axis=1, keepdims=True)
    train_counts = counts - mask
    train_sums = totals - masked
    with np.errstate(divide="ignore", invalid="ignore"):
        utility = np.where(train_counts > 0, train_sums / train_counts, -np.inf)
    utility = np.where(mask, utility, -np.inf)
    usable = np.isfinite(utility).any(axis=0)
    if not usable.any():
        raise ValueError("没有可用于 cross-fit 估值的 rollout 列")
    selected = np.argmax(utility[:, usable], axis=0)
    columns = np.flatnonzero(usable)
    outcomes = values[selected, columns]
    return outcomes.astype(np.float32), selected.astype(np.int32)


def weighted_mean(values: np.ndarray, radical_factor: float) -> float:
    """复现 Rust ``ActionResult::weighted_mean`` 的分组中点排名积分。"""

    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("values 必须是非空有限一维数组")
    if radical_factor < 0.0:
        raise ValueError("radical_factor 不得为负")
    unique, counts = np.unique(x, return_counts=True)
    before = np.cumsum(counts, dtype=np.float64) - counts
    quantile = (before + counts / 2.0) / x.size
    rank_weight = np.power(quantile, radical_factor)
    mass = counts * rank_weight
    return float(np.sum(unique * mass) / np.sum(mass))


def crossfit_value_target(
    scores: np.ndarray,
    radical_factor: float,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """构造三路 value 标签及选择稳定率。

    稳定率是 leave-one-out 所选候选的众数占比，用于诊断“动作是否依赖某几个
    rollout”；它不参与 loss。
    """

    outcomes, selected = leave_one_out_outcomes(scores, valid)
    stdev = float(np.std(outcomes, ddof=1)) if outcomes.size > 1 else 0.0
    target = np.asarray(
        [float(np.mean(outcomes)), stdev, weighted_mean(outcomes, radical_factor)],
        dtype=np.float32,
    )
    counts = np.bincount(selected)
    stability = float(counts.max() / selected.size)
    return target, stability


def _required_array(root: Path, name: str, mmap: bool = True) -> np.ndarray:
    """读取必需数组并给出带路径的错误。"""

    path = root / f"{name}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {path}")
    return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)


def _hash_small_arrays(arrays: Iterable[np.ndarray]) -> str:
    """哈希小型索引数组，供 sidecar 对齐校验。"""

    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.asarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    """生成不泄露用户目录的元数据路径。"""

    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def generate_label_sidecar(source: Path, output: Path, config: LabelConfig, overwrite: bool = False) -> dict:
    """为一个 raw `.npy` 目录流式生成可训练标签 sidecar。"""

    config.validate()
    source = source.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    final_names = ["policy_target.npy", "value_target.npy", "policy_entropy.npy", "selector_stability.npy"]
    if not overwrite and any((output / name).exists() for name in final_names):
        raise FileExistsError(f"{output} 已有标签文件；确认后加 --overwrite")

    x = _required_array(source, "x")
    stage = _required_array(source, "stage")
    index = _required_array(source, "index")
    ptr = _required_array(source, "cand_ptr")
    slots = _required_array(source, "cand_slots")
    candidate_n = _required_array(source, "cand_n")
    scores = _required_array(source, "cand_scores")
    valid_path = source / "cand_valid.npy"
    valid = np.load(valid_path, mmap_mode="r", allow_pickle=False) if valid_path.is_file() else None
    legal = _required_array(source, "legal_mask")

    n = len(index)
    if x.shape != (n, INPUT_DIM) or stage.shape != (n,) or legal.shape != (n, POLICY_DIM):
        raise ValueError("x/stage/legal_mask 维度与冻结契约不一致")
    if ptr.shape != (n + 1,) or int(ptr[0]) != 0 or int(ptr[-1]) != len(slots):
        raise ValueError("cand_ptr 不是覆盖全部候选的合法 CSR 偏移")
    if slots.shape != (scores.shape[0], 3):
        raise ValueError("cand_slots 与 cand_scores 候选维不一致")
    if candidate_n.shape != (scores.shape[0],) or np.any(candidate_n < 2):
        raise ValueError("cand_n 形状错误或某候选不足 2 个有效 rollout")
    if valid is not None and valid.shape != scores.shape:
        raise ValueError("cand_valid 与 cand_scores 形状不一致")
    if valid is not None and not np.array_equal(np.sum(valid, axis=1, dtype=np.int64), candidate_n):
        raise ValueError("cand_n 与 cand_valid 有效位计数不一致")

    weights = make_bayesian_weights(scores.shape[1], config.bootstrap_draws, config.bootstrap_seed)
    tmp_paths = {name: output / name.replace(".npy", ".tmp.npy") for name in final_names}
    policy_out = np.lib.format.open_memmap(tmp_paths["policy_target.npy"], mode="w+", dtype=np.float32, shape=(n, POLICY_DIM))
    value_out = np.lib.format.open_memmap(tmp_paths["value_target.npy"], mode="w+", dtype=np.float32, shape=(n, 3))
    entropy_out = np.lib.format.open_memmap(tmp_paths["policy_entropy.npy"], mode="w+", dtype=np.float32, shape=(n,))
    stability_out = np.lib.format.open_memmap(tmp_paths["selector_stability.npy"], mode="w+", dtype=np.float32, shape=(n,))

    started = time.perf_counter()
    for i in range(n):
        begin, end = int(ptr[i]), int(ptr[i + 1])
        sample_scores = np.asarray(scores[begin:end])
        sample_valid = None if valid is None else np.asarray(valid[begin:end], dtype=bool)
        probs = bayesian_best_probabilities(
            sample_scores,
            weights,
            sample_valid,
            tie_atol=config.tie_atol,
            temperature=config.policy_temperature,
        )
        target = candidate_probs_to_policy(probs, np.asarray(slots[begin:end]))
        legal_row = np.asarray(legal[i], dtype=bool)
        if np.any(target[~legal_row] != 0.0):
            raise ValueError(f"样本 {int(index[i])} 的 policy target 泄漏到非法格位")
        policy_out[i] = target
        positive = target > 0.0
        entropy_out[i] = float(-np.sum(target[positive] * np.log(target[positive])))
        value_out[i], stability_out[i] = crossfit_value_target(
            sample_scores,
            config.radical_factor,
            sample_valid,
        )
        if (i + 1) % 1000 == 0 or i + 1 == n:
            elapsed = time.perf_counter() - started
            print(f"标签 {i + 1}/{n}，{(i + 1) / elapsed:.1f} 样本/s", flush=True)

    for array in (policy_out, value_out, entropy_out, stability_out):
        array.flush()
    # Windows 不允许重命名仍被 memmap 句柄占用的文件；for 循环变量会在循环后
    # 继续引用最后一个数组，必须与四个具名引用一起显式释放。
    del array
    del policy_out, value_out, entropy_out, stability_out
    for name in final_names:
        os.replace(tmp_paths[name], output / name)
    np.save(output / "index.npy", np.asarray(index))

    value = np.load(output / "value_target.npy", mmap_mode="r")
    entropy = np.load(output / "policy_entropy.npy", mmap_mode="r")
    stability = np.load(output / "selector_stability.npy", mmap_mode="r")
    stage_summary: dict[str, dict[str, float | int]] = {}
    for code in sorted(int(v) for v in np.unique(stage)):
        mask = np.asarray(stage) == code
        stage_summary[str(code)] = {
            "samples": int(mask.sum()),
            "policy_entropy_mean": float(np.mean(entropy[mask])),
            "selector_stability_mean": float(np.mean(stability[mask])),
        }
    meta = {
        "format_version": 1,
        "source": _display_path(source),
        "samples": n,
        "candidates": int(scores.shape[0]),
        "rollouts": int(scores.shape[1]),
        "index_ptr_sha256": _hash_small_arrays((index, ptr)),
        "config": asdict(config),
        "value_mean": np.mean(value, axis=0, dtype=np.float64).tolist(),
        "value_stdev": np.std(value, axis=0, ddof=1, dtype=np.float64).tolist(),
        "stages": stage_summary,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "labels.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def _parse_args() -> argparse.Namespace:
    """解析命令行。"""

    parser = argparse.ArgumentParser(description="从 raw .npy 目录生成拉面杯训练标签")
    parser.add_argument("--input", type=Path, required=True, help="含 cand_scores.npy 的 raw 目录")
    parser.add_argument("--output", type=Path, required=True, help="标签 sidecar 输出目录")
    parser.add_argument("--bootstrap-draws", type=int, default=512)
    parser.add_argument("--bootstrap-seed", type=int, default=20260830)
    parser.add_argument("--radical-factor", type=float, default=1.4)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args = _parse_args()
    config = LabelConfig(
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
        radical_factor=args.radical_factor,
        policy_temperature=args.policy_temperature,
    )
    generate_label_sidecar(args.input, args.output, config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
