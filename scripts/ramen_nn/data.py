"""目录式 `.npy` 教师数据的 mmap 加载、稳定划分与归一化。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import json
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .model import INPUT_DIM, POLICY_DIM
except ImportError:
    from model import INPUT_DIM, POLICY_DIM

STAGE_COUNT = 5
UINT64_MASK = (1 << 64) - 1


def _load(root: Path, name: str) -> np.ndarray:
    """以只读 mmap 打开一个数组。"""

    path = root / f"{name}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {path}")
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _splitmix64(value: int, seed: int) -> int:
    """稳定的 64 位混合函数，使追加数据不改变旧样本划分。"""

    z = (value + seed + 0x9E3779B97F4A7C15) & UINT64_MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
    return (z ^ (z >> 31)) & UINT64_MASK


def _read_plan_count(data_dir: Path) -> int | None:
    """从导出目录的 meta.json 读采样计划数；旧版导出没有该字段则返回 None。"""

    meta_path = data_dir / "meta.json"
    if not meta_path.is_file():
        return None
    with meta_path.open(encoding="utf-8") as handle:
        value = json.load(handle).get("plan_count")
    if value is None:
        return None
    count = int(value)
    if count <= 0:
        raise ValueError(f"{meta_path}: plan_count 必须为正")
    return count


def _display_path(path: Path) -> str:
    """将工作区内绝对路径缩成相对路径，避免日志携带用户名。"""

    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


@dataclass(frozen=True)
class ValueNormalization:
    """三路 value 的仿射归一化参数。"""

    center: tuple[float, float, float]
    scale: tuple[float, float, float]

    def to_dict(self) -> dict[str, list[float]]:
        """转为 JSON/checkpoint 友好的字典。"""

        return {"center": list(self.center), "scale": list(self.scale)}

    @classmethod
    def from_dict(cls, value: dict) -> "ValueNormalization":
        """从 checkpoint 字典恢复。"""

        return cls(tuple(float(v) for v in value["center"]), tuple(float(v) for v in value["scale"]))

    def normalize_tensor(self, value: torch.Tensor) -> torch.Tensor:
        """在 value 所在设备上执行归一化。"""

        center = value.new_tensor(self.center)
        scale = value.new_tensor(self.scale)
        return (value - center) / scale

    def denormalize_tensor(self, value: torch.Tensor) -> torch.Tensor:
        """在 value 所在设备上执行反归一化。"""

        center = value.new_tensor(self.center)
        scale = value.new_tensor(self.scale)
        return value * scale + center


class NpyShard:
    """一组 reduced 数据与其标签 sidecar。"""

    def __init__(self, data_dir: Path, label_dir: Path) -> None:
        self.data_dir = data_dir.resolve()
        self.label_dir = label_dir.resolve()
        self.x = _load(self.data_dir, "x")
        self.stage = _load(self.data_dir, "stage")
        self.turn = _load(self.data_dir, "turn")
        self.index = _load(self.data_dir, "index")
        self.legal_mask = _load(self.data_dir, "legal_mask")
        self.cand_ptr = _load(self.data_dir, "cand_ptr")
        self.cand_slots = _load(self.data_dir, "cand_slots")
        self.cand_mean = _load(self.data_dir, "cand_mean")
        self.policy_target = _load(self.label_dir, "policy_target")
        self.value_target = _load(self.label_dir, "value_target")
        label_index = _load(self.label_dir, "index")
        self.plan_count = _read_plan_count(self.data_dir)
        # 组合键：由导出器按 (马娘, 卡组) 直接算出，与采样空间无关。缺失说明是加入该列
        # 之前导出的目录，此时按组合切分回落到 `index % plan_count` 的旧口径。
        combo_path = self.data_dir / "combo_key.npy"
        self.combo_key = np.load(combo_path, mmap_mode="r") if combo_path.exists() else None
        # 原始 rollout 列只在 `--raw` 导出的目录里存在，且体积远大于其余数组，
        # 故不在构造时打开：只有按列窗口评估时才 mmap。
        self._cand_scores: np.ndarray | None = None
        self._cand_valid: np.ndarray | None = None
        self._raw_opened = False
        self._validate(label_index)

    def _open_raw(self) -> None:
        """惰性 mmap ``cand_scores``/``cand_valid``；缺失时留空并只报一次。"""

        if self._raw_opened:
            return
        self._raw_opened = True
        scores_path = self.data_dir / "cand_scores.npy"
        if not scores_path.is_file():
            return
        scores = _load(self.data_dir, "cand_scores")
        if scores.ndim != 2 or scores.shape[0] != len(self.cand_mean):
            raise ValueError(f"{self.data_dir}: cand_scores 形状 {scores.shape} 与候选数不符")
        self._cand_scores = scores
        valid_path = self.data_dir / "cand_valid.npy"
        if valid_path.is_file():
            valid = _load(self.data_dir, "cand_valid")
            if valid.shape != scores.shape:
                raise ValueError(f"{self.data_dir}: cand_valid 与 cand_scores 形状不一致")
            self._cand_valid = valid

    @property
    def has_raw_columns(self) -> bool:
        """本分片是否带完整 rollout 列（`--raw` 导出）。"""

        self._open_raw()
        return self._cand_scores is not None

    @property
    def rollout_columns(self) -> int:
        """每个候选的 rollout 列数；无原始列时为 0。"""

        self._open_raw()
        return 0 if self._cand_scores is None else int(self._cand_scores.shape[1])

    def candidate_window_mean(self, local_idx: int, lo: int, hi: int) -> np.ndarray | None:
        """只用 ``[lo, hi)`` 列重算一个样本的候选均值。

        失败槽（``cand_valid`` 为 0）不计入。窗口内某候选一个有效列都没有时返回
        ``None``——该样本在此窗口下无法评估，交由调用方计数并跳过，而不是伪造一个值。
        """

        self._open_raw()
        if self._cand_scores is None:
            raise ValueError(f"{self.data_dir}: 没有 cand_scores，无法按列窗口评估")
        total = int(self._cand_scores.shape[1])
        if not 0 <= lo < hi <= total:
            raise ValueError(f"列窗口 [{lo}, {hi}) 越出 {total} 列")
        begin, end = int(self.cand_ptr[local_idx]), int(self.cand_ptr[local_idx + 1])
        block = np.asarray(self._cand_scores[begin:end, lo:hi], dtype=np.float64)
        if self._cand_valid is None:
            return block.mean(axis=1)
        mask = np.asarray(self._cand_valid[begin:end, lo:hi], dtype=bool)
        counts = mask.sum(axis=1)
        if np.any(counts == 0):
            return None
        return np.where(mask, block, 0.0).sum(axis=1) / counts

    def _validate(self, label_index: np.ndarray) -> None:
        """校验训练时可能静默错位的全部形状和主键。"""

        n = len(self.index)
        if self.x.shape != (n, INPUT_DIM):
            raise ValueError(f"{self.data_dir}: x 形状错误 {self.x.shape}")
        if self.stage.shape != (n,) or self.turn.shape != (n,):
            raise ValueError(f"{self.data_dir}: stage/turn 形状错误")
        if self.combo_key is not None and self.combo_key.shape != (n,):
            raise ValueError(f"{self.data_dir}: combo_key 形状错误 {self.combo_key.shape}")
        if self.legal_mask.shape != (n, POLICY_DIM):
            raise ValueError(f"{self.data_dir}: legal_mask 形状错误 {self.legal_mask.shape}")
        if self.policy_target.shape != (n, POLICY_DIM) or self.value_target.shape != (n, 3):
            raise ValueError(f"{self.label_dir}: 标签形状错误")
        if label_index.shape != self.index.shape or not np.array_equal(label_index, self.index):
            raise ValueError(f"{self.label_dir}: index 与数据目录不一致")
        if self.cand_ptr.shape != (n + 1,) or int(self.cand_ptr[-1]) != len(self.cand_mean):
            raise ValueError(f"{self.data_dir}: cand_ptr 非法")
        if self.cand_slots.shape != (len(self.cand_mean), 3):
            raise ValueError(f"{self.data_dir}: cand_slots 候选维错误")
        if not np.all(np.isin(np.unique(self.stage), np.arange(STAGE_COUNT))):
            raise ValueError(f"{self.data_dir}: 出现未知 stage")
        sums = np.sum(self.policy_target, axis=1)
        if not np.allclose(sums, 1.0, atol=2e-5):
            raise ValueError(f"{self.label_dir}: policy target 行和不是 1")
        if np.any(np.asarray(self.policy_target)[np.asarray(self.legal_mask) == 0] != 0.0):
            raise ValueError(f"{self.label_dir}: policy target 泄漏到非法格位")
        if not np.isfinite(self.value_target).all():
            raise ValueError(f"{self.label_dir}: value target 含非有限值")

    def __len__(self) -> int:
        return len(self.index)


def load_shards(data_dirs: Sequence[Path], label_dirs: Sequence[Path]) -> list[NpyShard]:
    """按位置配对加载多个数据目录，并拒绝跨目录重复样本 id。"""

    if not data_dirs or len(data_dirs) != len(label_dirs):
        raise ValueError("--data 与 --labels 必须非空且一一对应")
    shards = [NpyShard(data, labels) for data, labels in zip(data_dirs, label_dirs, strict=True)]
    ids = np.concatenate([np.asarray(shard.index) for shard in shards])
    if np.unique(ids).size != ids.size:
        raise ValueError("多个数据目录之间存在重复 index")
    return shards


def resolve_plan_count(shards: Sequence[NpyShard]) -> int:
    """取各数据目录一致的采样计划数。

    采样器按 ``index % plan_count`` 轮转分配 (马娘, 卡组) 组合，所以它是按组合
    切分留出集的前提。各目录不一致说明混了不同采样空间的数据，必须报错。
    """

    values = {shard.plan_count for shard in shards}
    if None in values:
        raise ValueError(
            "数据目录缺少 meta.json 的 plan_count 字段；请用当前版本的 "
            "ramen_export_npy 重新导出后再按组合切分"
        )
    if len(values) != 1:
        raise ValueError(f"各数据目录的 plan_count 不一致: {sorted(values)}")
    return int(values.pop())


def stable_split_refs(
    shards: Sequence[NpyShard],
    validation_fraction: float,
    seed: int,
    split_by: str = "combo",
) -> tuple[np.ndarray, np.ndarray]:
    """返回 ``[M,2]`` 的 ``(shard, local_index)`` 训练/验证引用。

    ``split_by`` 决定哈希什么：

    - ``"combo"``（默认）：哈希 (马娘, 卡组) 组合。
      留出的组合完全不参与训练，验证指标才真的在测泛化。
    - ``"sample"``：哈希样本 id。同一套卡组会同时落进训练与验证，
      验证后悔值会系统性偏乐观，只在需要与旧结果对齐时使用。

    按组合切分有两种取键口径，优先用前者：

    - **``combo_key.npy``**：导出器按 (马娘, 卡组) 直接算出的稳定键。它与采样空间无关，
      因此**不同空间采的目录可以合并**，且同一副卡组必定落在切分的同一侧。
    - **``index % plan_count``**（回落）：仅当全部目录都没有 ``combo_key.npy`` 时使用。
      该口径绑死在单一空间上，故此时仍要求各目录的 ``plan_count`` 一致。
    """

    if not 0.01 <= validation_fraction <= 0.5:
        raise ValueError("validation_fraction 必须位于 [0.01, 0.5]")
    if split_by not in ("combo", "sample"):
        raise ValueError("split_by 必须是 combo 或 sample")
    use_combo_key = split_by == "combo" and all(shard.combo_key is not None for shard in shards)
    if split_by == "combo" and not use_combo_key:
        missing = [_display_path(s.data_dir) for s in shards if s.combo_key is None]
        if len(missing) != len(shards):
            raise ValueError(
                "部分目录有 combo_key.npy、部分没有，两种组合口径不可混用："
                f"缺少的目录 {missing}。请用当前版本的 ramen_export_npy 重新导出这些目录"
            )
    plan_count = 0 if use_combo_key or split_by == "sample" else resolve_plan_count(shards)
    threshold = int(validation_fraction * 10_000)
    train: list[tuple[int, int]] = []
    validation: list[tuple[int, int]] = []
    for shard_idx, shard in enumerate(shards):
        for local_idx, sample_id in enumerate(shard.index):
            ref = (shard_idx, local_idx)
            if split_by == "sample":
                key = int(sample_id)
            elif use_combo_key:
                key = int(shard.combo_key[local_idx])
            else:
                key = int(sample_id) % plan_count
            bucket = _splitmix64(key, seed) % 10_000
            (validation if bucket < threshold else train).append(ref)
    if not train or not validation:
        raise ValueError("稳定划分产生空集合；数据量过小或 validation_fraction 不合理")
    return np.asarray(train, dtype=np.int64), np.asarray(validation, dtype=np.int64)


def subsample_train_refs(
    shards: Sequence[NpyShard],
    train_refs: np.ndarray,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    """把训练引用稳定地截到 ``max_samples`` 条，用于数据量曲线实验。

    按 ``splitmix64(sample_id, seed)`` 对每条训练样本排序后取前 ``max_samples`` 条。
    这样做有三个性质：

    - **确定性**：同一 ``seed`` 下结果可复现。
    - **嵌套**：同一 ``seed`` 下 3k 的子集严格包含于 5k、8k、12k，
      曲线上各点的差异只来自数据量本身，不来自换了一批样本。
    - **与真实采集同分布**：按样本而非按 (马娘, 卡组) 组合抽稀——真实补数据时
      新样本也是散落在全部组合上的，按组合抽稀会额外引入组合覆盖度的混淆。

    验证集不受影响，各数据量点的留出指标因此可比。

    ``max_samples >= len(train_refs)`` 时原样返回。
    """

    if max_samples <= 0:
        raise ValueError("max_samples 必须为正")
    if max_samples >= len(train_refs):
        return train_refs
    keys = np.empty(len(train_refs), dtype=np.uint64)
    for i, (shard_idx, local_idx) in enumerate(train_refs):
        sample_id = int(shards[int(shard_idx)].index[int(local_idx)])
        keys[i] = _splitmix64(sample_id, seed)
    order = np.argsort(keys, kind="stable")[:max_samples]
    return train_refs[np.sort(order)]


class RamenDataset(Dataset):
    """只在 ``__getitem__`` 时复制一个样本的 mmap 数据。"""

    def __init__(self, shards: Sequence[NpyShard], refs: np.ndarray) -> None:
        self.shards = list(shards)
        self.refs = np.asarray(refs, dtype=np.int64)
        if self.refs.ndim != 2 or self.refs.shape[1] != 2:
            raise ValueError("refs 必须是 [N,2]")

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        shard_idx, local_idx = (int(v) for v in self.refs[item])
        shard = self.shards[shard_idx]
        return {
            "x": torch.from_numpy(np.array(shard.x[local_idx], dtype=np.float32, copy=True)),
            "legal_mask": torch.from_numpy(np.array(shard.legal_mask[local_idx], dtype=bool, copy=True)),
            "policy_target": torch.from_numpy(np.array(shard.policy_target[local_idx], dtype=np.float32, copy=True)),
            "value_target": torch.from_numpy(np.array(shard.value_target[local_idx], dtype=np.float32, copy=True)),
            "stage": torch.tensor(int(shard.stage[local_idx]), dtype=torch.long),
        }


def fit_value_normalization(shards: Sequence[NpyShard], train_refs: np.ndarray) -> ValueNormalization:
    """仅用训练划分拟合三路 value 的均值和样本标准差。"""

    values = np.empty((len(train_refs), 3), dtype=np.float64)
    for i, (shard_idx, local_idx) in enumerate(train_refs):
        values[i] = shards[int(shard_idx)].value_target[int(local_idx)]
    center = np.mean(values, axis=0)
    scale = np.std(values, axis=0, ddof=1)
    if np.any(scale < 1e-6) or not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise ValueError("value 标签无法得到有效归一化尺度")
    return ValueNormalization(tuple(center.tolist()), tuple(scale.tolist()))


def compute_stage_weights(
    shards: Sequence[NpyShard],
    train_refs: np.ndarray,
    max_weight: float = 4.0,
) -> tuple[torch.Tensor, list[int]]:
    """计算截断的逆平方根频率权重，并按训练分布归一到均值 1。"""

    counts = np.zeros(STAGE_COUNT, dtype=np.int64)
    for shard_idx, local_idx in train_refs:
        counts[int(shards[int(shard_idx)].stage[int(local_idx)])] += 1
    present = counts > 0
    weights = np.zeros(STAGE_COUNT, dtype=np.float64)
    reference = counts[present].max()
    weights[present] = np.minimum(np.sqrt(reference / counts[present]), max_weight)
    weights[present] /= np.sum(weights[present] * counts[present]) / np.sum(counts[present])
    return torch.tensor(weights, dtype=torch.float32), counts.tolist()


def describe_split(shards: Sequence[NpyShard], train_refs: np.ndarray, validation_refs: np.ndarray) -> dict:
    """生成可写入日志/checkpoint 的划分摘要。"""

    def counts(refs: np.ndarray) -> list[int]:
        result = [0] * STAGE_COUNT
        for shard_idx, local_idx in refs:
            result[int(shards[int(shard_idx)].stage[int(local_idx)])] += 1
        return result

    return {
        "total": int(sum(len(shard) for shard in shards)),
        "train": int(len(train_refs)),
        "validation": int(len(validation_refs)),
        "train_stage_counts": counts(train_refs),
        "validation_stage_counts": counts(validation_refs),
        "data_dirs": [_display_path(shard.data_dir) for shard in shards],
        "label_dirs": [_display_path(shard.label_dir) for shard in shards],
    }
