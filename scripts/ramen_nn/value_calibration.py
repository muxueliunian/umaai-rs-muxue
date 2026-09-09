"""拆开 value 头的偏差，判断「尾部失真」是失准、表示受限还是训练不足。

## 被诊断的量不是 V

三路 value 的分量 0 是 leave-one-rollout-out cross-fit 估值（见 `DESIGN.md` §2）：
用其余 511 列选动作、只用第 k 列给该动作估分。所以它是

    W(s) = Q^手写(s, pi_search(s))

**既不是** `V^手写`，**也不是** `V^NN`，更不是反复搜索的 `V^search`。把它当 leaf
估值器用会换掉被评估的后续策略——这是语义差异，不是精度问题。

## 两种分箱缺一不可

- **按真值分箱**（`by_truth`）：显示「高真值被低估、低真值被高估」。这**不等于失准**：
  只要输入不能完全解释标签，即使最优预测器 `Vhat = E[Y|X]` 也必然满足
  `Cov(Y,Vhat)/Var(Y) = Var(Vhat)/Var(Y) <= 1`，按真值看就会收缩。754 维表示遗漏的
  信息同样计入这个「不可解释部分」，故「训练集上也收缩」**不能**单独排除表示问题。
- **按预测值分箱**（`by_prediction`）：检查 `E[Y | Vhat=v] ~= v`。这才是校准的定义，
  偏离它才是真正的失准。

## 标签噪声地板

两种口径都报：

- `label_noise_floor_sd_naive` = `mean(sd)/sqrt(n)`，旧口径，仅供与历史数字对齐
- `label_noise_floor_rmse` = `sqrt(mean(sd^2/n))`，**与 RMSE 可比的那个**

两者都假定 n 次估值独立，而 cross-fit 的 512 项共用 511 列做选择、彼此相关，
相关会让真实标准误**大于**这里算出的地板。所以由此得到的「噪声占残差比例」是
**下界**，不足以据此排除数据因素。要真正核准，得对少量状态用独立 rollout 种子
重复生成标签，直接测标签的重测方差。

用法::

    python value_calibration.py --checkpoint target/xxx/step_020000.pt \
        --data training_data/npy_v6ck --labels training_data/labels_v6 \
        --data training_data/npy_ood_half101 --labels training_data/labels_ood_half101 \
        --split validation
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .data import RamenDataset, ValueNormalization, load_shards, stable_split_refs
    from .model import model_from_checkpoint
except ImportError:
    from data import RamenDataset, ValueNormalization, load_shards, stable_split_refs
    from model import model_from_checkpoint

#: 真值分位分箱的边界（百分位）
QUANTILE_EDGES = (0.0, 5.0, 25.0, 75.0, 95.0, 100.0)


def _choose_device(name: str) -> torch.device:
    """解析 `auto` / 显式设备名。"""

    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Predictions:
    """留出集上的逐样本预测与分组键。

    `keys` 在缺 `combo_key.npy` 的分片上回填 0，只影响 `--combo-keys` 的筛选。
    """

    predicted: np.ndarray
    truth: np.ndarray
    outcome_sd: np.ndarray
    keys: np.ndarray
    stage: np.ndarray
    turn: np.ndarray

    def select(self, mask: np.ndarray) -> "Predictions":
        """按布尔掩码取子集，各数组同步筛选。"""

        return Predictions(
            self.predicted[mask], self.truth[mask], self.outcome_sd[mask],
            self.keys[mask], self.stage[mask], self.turn[mask]
        )


def collect_predictions(
    model, shards, refs, normalization: ValueNormalization, device, batch_size: int
) -> Predictions:
    """跑一遍留出集，返回逐样本预测、真值、结局 sd 与 combo_key / stage / turn。"""

    model.eval()
    dataset = RamenDataset(shards, refs)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False, num_workers=0)
    center = np.asarray(normalization.center, dtype=np.float32)
    scale = np.asarray(normalization.scale, dtype=np.float32)
    predicted: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    outcome_sd: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            output = model(batch["x"].to(device))
            _, _, value_normalized = model.split_output(output)
            value = value_normalized.cpu().numpy() * scale + center
            targets = batch["value_target"].numpy()
            predicted.append(value[:, 0].copy())
            truth.append(targets[:, 0].copy())
            outcome_sd.append(targets[:, 1].copy())
    # combo_key 是 fnv1a64，超出 int64 值域，必须按 uint64 搬运
    keys = np.zeros(len(refs), dtype=np.uint64)
    stage = np.zeros(len(refs), dtype=np.int64)
    turn = np.zeros(len(refs), dtype=np.int64)
    for row, (shard_idx, local_idx) in enumerate(refs):
        shard = shards[int(shard_idx)]
        local = int(local_idx)
        if shard.combo_key is not None:
            keys[row] = np.uint64(shard.combo_key[local])
        stage[row] = int(shard.stage[local])
        turn[row] = int(shard.turn[local])
    return Predictions(
        np.concatenate(predicted),
        np.concatenate(truth),
        np.concatenate(outcome_sd),
        keys,
        stage,
        turn,
    )


def _quantile_bins(by: np.ndarray, predicted: np.ndarray, truth: np.ndarray) -> list[dict]:
    """按 `by` 的分位切箱，逐箱报出预测均值、真值均值与有符号偏差。

    `by` 取真值即经典的「按真值分箱」，取预测值即校准图所需的分箱。两者的
    `signed_bias` 含义不同，故箱内同时给出 `predicted_mean` 与 `truth_mean`，
    让读者自己看是哪一侧在动。
    """

    cuts = np.percentile(by, QUANTILE_EDGES)
    residual = predicted - truth
    bins: list[dict] = []
    for i in range(len(QUANTILE_EDGES) - 1):
        low, high = cuts[i], cuts[i + 1]
        last = i == len(QUANTILE_EDGES) - 2
        mask = (by >= low) & (by <= high) if last else (by >= low) & (by < high)
        if not np.any(mask):
            continue
        count = int(mask.sum())
        bins.append(
            {
                "quantile": f"{QUANTILE_EDGES[i]:g}-{QUANTILE_EDGES[i + 1]:g}%",
                "samples": count,
                "predicted_mean": float(np.mean(predicted[mask])),
                "truth_mean": float(np.mean(truth[mask])),
                "signed_bias": float(np.mean(residual[mask])),
                "signed_bias_se": (
                    float(np.std(residual[mask], ddof=1) / np.sqrt(count)) if count > 1 else 0.0
                ),
                "mae": float(np.mean(np.abs(residual[mask]))),
            }
        )
    return bins


def calibration_report(
    predicted: np.ndarray, truth: np.ndarray, outcome_sd: np.ndarray, rollout_width: int
) -> dict:
    """整体指标 + 按真值分箱与按预测值分箱两张偏差表。

    两张表必须一起看：按真值分箱的收缩在最优预测器上也会出现（见模块文档），
    只有按预测值分箱的偏离才是失准。
    """

    residual = predicted - truth
    width = float(rollout_width)
    # 与 RMSE 可比的地板是 sqrt(E[sd^2/n])；旧口径 mean(sd)/sqrt(n) 只为对齐历史数字
    noise_rmse = float(np.sqrt(np.mean(outcome_sd.astype(np.float64) ** 2 / width)))
    noise_naive = float(np.mean(outcome_sd)) / np.sqrt(width)
    rmse = float(np.sqrt(np.mean(residual**2)))
    overall = {
        "samples": int(residual.size),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": rmse,
        "signed_bias": float(np.mean(residual)),
        "signed_bias_se": float(np.std(residual, ddof=1) / np.sqrt(residual.size)),
        "correlation": float(np.corrcoef(predicted, truth)[0, 1]),
        "truth_sd": float(np.std(truth, ddof=1)),
        "predicted_sd": float(np.std(predicted, ddof=1)),
        "label_noise_floor_rmse": noise_rmse,
        "label_noise_floor_sd_naive": noise_naive,
        # 分子是地板的下界（cross-fit 项相关会抬高真实地板），故这是噪声占比的**下界**
        "noise_share_of_mse_lower_bound": (
            float(min(noise_rmse**2 / rmse**2, 1.0)) if rmse > 0 else 0.0
        ),
        "model_error_sd_upper_bound": float(np.sqrt(max(rmse**2 - noise_rmse**2, 0.0))),
    }
    # 按真值回归：斜率 < 1 是回归稀释，最优预测器上同样出现，不能单独当作失准证据
    overall["slope_pred_on_truth"] = float(np.polyfit(truth, predicted, 1)[0])
    # 按预测值回归：校准良好时应 ~= 1，显著偏离才说明失准
    overall["slope_truth_on_pred"] = float(np.polyfit(predicted, truth, 1)[0])
    # 最优条件均值满足 Var(Vhat)/Var(Y) == slope_pred_on_truth，两者背离即为失准信号
    overall["variance_ratio"] = float(np.var(predicted, ddof=1) / np.var(truth, ddof=1))

    return {
        "overall": overall,
        "by_truth": _quantile_bins(truth, predicted, truth),
        "by_prediction": _quantile_bins(predicted, predicted, truth),
    }


#: `stage` 列的取值到人类可读名（与 `scripts/ramen_nn/eval.py` 一致）
STAGE_NAMES = {0: "RamenSelect", 1: "SpecialSelect", 2: "Train", 3: "SuperRamenSelect", 4: "RegionSelect"}

#: 回合分组的上界（含），用于看「剩余回合」方向上的校准
TURN_EDGES = (24, 48, 78)


def _group_calibration(pred: Predictions, width: float) -> dict:
    """按阶段与回合区间分组，各组给出整体偏差与按预测值分箱的失准幅度。

    分组的意义是：整体校准良好仍可能掩盖某个阶段或某段回合上的系统偏差，
    而 leaf 估值器恰好只在特定深度被调用。
    """

    def summarize(mask: np.ndarray) -> dict | None:
        count = int(mask.sum())
        if count < 50:
            return None
        sub = pred.select(mask)
        residual = sub.predicted - sub.truth
        rmse = float(np.sqrt(np.mean(residual**2)))
        bins = _quantile_bins(sub.predicted, sub.predicted, sub.truth)
        worst = max((abs(b["signed_bias"]) for b in bins), default=0.0)
        return {
            "samples": count,
            "signed_bias": float(np.mean(residual)),
            "signed_bias_se": float(np.std(residual, ddof=1) / np.sqrt(count)),
            "rmse": rmse,
            "slope_truth_on_pred": float(np.polyfit(sub.predicted, sub.truth, 1)[0]),
            "label_noise_floor_rmse": float(np.sqrt(np.mean(sub.outcome_sd.astype(np.float64) ** 2 / width))),
            "worst_bin_abs_bias_by_prediction": float(worst),
        }

    by_stage = {}
    for value, name in STAGE_NAMES.items():
        entry = summarize(pred.stage == value)
        if entry is not None:
            by_stage[name] = entry

    by_turn = {}
    low = 0
    for high in TURN_EDGES:
        entry = summarize((pred.turn >= low) & (pred.turn <= high))
        if entry is not None:
            by_turn[f"turn {low}-{high}"] = entry
        low = high + 1
    return {"by_stage": by_stage, "by_turn": by_turn}


def _parse_args() -> argparse.Namespace:
    """解析命令行。"""

    parser = argparse.ArgumentParser(description="value 头的分位标定诊断")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--labels", type=Path, action="append", required=True)
    parser.add_argument("--split", choices=("train", "validation", "all"), default="validation")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--combo-keys",
        type=Path,
        help="只统计 combo_key 落在该 .npy 里的样本（用于只看分布外组合）",
    )
    return parser.parse_args()


def main() -> None:
    """诊断入口。"""

    args = _parse_args()
    if len(args.data) != len(args.labels):
        raise ValueError("--data 与 --labels 数量必须一致")
    device = _choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    shards = load_shards(args.data, args.labels)
    split = checkpoint.get("split", {})
    train_refs, validation_refs = stable_split_refs(
        shards,
        float(split.get("validation_fraction", 0.1)),
        int(split.get("seed", 20260830)),
        str(split.get("split_by", "sample")),
    )
    refs = {"train": train_refs, "validation": validation_refs}.get(
        args.split, np.concatenate([train_refs, validation_refs], axis=0)
    )
    normalization = ValueNormalization.from_dict(checkpoint["value_normalization"])
    model = model_from_checkpoint(checkpoint, device)
    pred = collect_predictions(model, shards, refs, normalization, device, args.batch_size)
    if args.combo_keys is not None:
        wanted = np.load(args.combo_keys)
        mask = np.isin(pred.keys, wanted)
        if not np.any(mask):
            raise ValueError("--combo-keys 过滤后没有样本")
        pred = pred.select(mask)
    rollout_width = max(shard.rollout_columns for shard in shards)
    report = calibration_report(pred.predicted, pred.truth, pred.outcome_sd, rollout_width)
    report.update(_group_calibration(pred, float(rollout_width)))
    report.update(
        {
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "rollout_width": int(rollout_width),
            "combo_filter": str(args.combo_keys) if args.combo_keys else None,
        }
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
