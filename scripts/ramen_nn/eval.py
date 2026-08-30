"""在稳定留出集上评估纯网络 policy 与三路 value。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .data import NpyShard, RamenDataset, ValueNormalization, load_shards, stable_split_refs
    from .model import RamenNetwork, model_from_checkpoint
except ImportError:
    from data import NpyShard, RamenDataset, ValueNormalization, load_shards, stable_split_refs
    from model import RamenNetwork, model_from_checkpoint

STAGE_NAMES = {
    0: "RamenSelect",
    1: "SpecialSelect",
    2: "Train",
    3: "SuperRamenSelect",
    4: "RegionSelect",
}


@dataclass
class MetricAccumulator:
    """一个阶段的可加评估统计。"""

    samples: int = 0
    top1_hits: int = 0
    regret_sum: float = 0.0
    value_abs_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    invalid_region_combo: int = 0

    def report(self) -> dict[str, float | int | list[float] | None]:
        """转成均值指标；空阶段返回 JSON ``null`` 而不是伪造 0。"""

        if self.samples == 0:
            return {
                "samples": 0,
                "top1_accuracy": None,
                "expected_regret": None,
                "value_mae": None,
                "invalid_region_combo": 0,
            }
        return {
            "samples": self.samples,
            "top1_accuracy": self.top1_hits / self.samples,
            "expected_regret": self.regret_sum / self.samples,
            "value_mae": (self.value_abs_sum / self.samples).tolist(),
            "invalid_region_combo": self.invalid_region_combo,
        }


def select_candidate(logits: np.ndarray, shard: NpyShard, local_idx: int) -> tuple[int, bool]:
    """按 Rust 预期语义把格位 logits 还原成候选下标。

    RegionSelect 严格执行合法地区中的 top-3，再查找对应组合；其他阶段直接取
    单格候选的最大 logit。返回值第二项表示地区 top-3 是否未出现在候选表。
    """

    begin, end = int(shard.cand_ptr[local_idx]), int(shard.cand_ptr[local_idx + 1])
    slots = np.asarray(shard.cand_slots[begin:end])
    stage = int(shard.stage[local_idx])
    if stage != 4:
        candidate_logits = logits[slots[:, 0]]
        return int(np.argmax(candidate_logits)), False

    legal_slots = np.flatnonzero(np.asarray(shard.legal_mask[local_idx], dtype=bool))
    if legal_slots.size < 3:
        raise ValueError(f"样本 {int(shard.index[local_idx])} 的 RegionSelect 合法地区少于 3")
    top = legal_slots[np.argpartition(logits[legal_slots], -3)[-3:]]
    wanted = tuple(sorted(int(v) for v in top))
    for candidate_idx, row in enumerate(slots):
        occupied = tuple(sorted(int(v) for v in row if v >= 0))
        if occupied == wanted:
            return candidate_idx, False

    # 防御性兜底：若未来候选不再是完整组合表，仍给出可评估动作，同时显式计错。
    scores = np.sum(logits[slots], axis=1)
    return int(np.argmax(scores)), True


@torch.inference_mode()
def evaluate_model(
    model: RamenNetwork,
    shards: list[NpyShard],
    refs: np.ndarray,
    normalization: ValueNormalization,
    device: torch.device,
    batch_size: int = 1024,
    workers: int = 0,
) -> dict:
    """计算按阶段 top-1、期望后悔值和三路 value MAE。"""

    model.eval()
    dataset = RamenDataset(shards, refs)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False, num_workers=workers)
    accumulators = {stage: MetricAccumulator() for stage in STAGE_NAMES}
    overall = MetricAccumulator()
    offset = 0
    center = np.asarray(normalization.center, dtype=np.float32)
    scale = np.asarray(normalization.scale, dtype=np.float32)

    for batch in loader:
        x = batch["x"].to(device)
        output = model(x)
        policy, _, value_normalized = model.split_output(output)
        logits_batch = policy.cpu().numpy()
        value_batch = value_normalized.cpu().numpy() * scale + center
        targets = batch["value_target"].numpy()
        stages = batch["stage"].numpy()

        for row in range(len(stages)):
            shard_idx, local_idx = (int(v) for v in refs[offset + row])
            shard = shards[shard_idx]
            begin, end = int(shard.cand_ptr[local_idx]), int(shard.cand_ptr[local_idx + 1])
            means = np.asarray(shard.cand_mean[begin:end], dtype=np.float64)
            selected, invalid = select_candidate(logits_batch[row], shard, local_idx)
            best = float(np.max(means))
            chosen = float(means[selected])
            hit = bool(np.isclose(chosen, best, rtol=0.0, atol=1e-4))
            absolute_error = np.abs(value_batch[row] - targets[row])
            for accumulator in (accumulators[int(stages[row])], overall):
                accumulator.samples += 1
                accumulator.top1_hits += int(hit)
                accumulator.regret_sum += best - chosen
                accumulator.value_abs_sum += absolute_error
                accumulator.invalid_region_combo += int(invalid)
        offset += len(stages)

    return {
        "overall": overall.report(),
        "by_stage": {STAGE_NAMES[stage]: accumulators[stage].report() for stage in STAGE_NAMES},
    }


def _parse_args() -> argparse.Namespace:
    """解析独立评估命令。"""

    parser = argparse.ArgumentParser(description="评估拉面杯 ONNX 前的 PyTorch checkpoint")
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--labels", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "all"), default="validation")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _choose_device(name: str) -> torch.device:
    """解析 auto/cpu/cuda 设备。"""

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    """独立评估入口。"""

    args = _parse_args()
    device = _choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    shards = load_shards(args.data, args.labels)
    split = checkpoint.get("split", {})
    validation_fraction = float(split.get("validation_fraction", 0.1))
    split_seed = int(split.get("seed", 20260830))
    train_refs, validation_refs = stable_split_refs(shards, validation_fraction, split_seed)
    if args.split == "train":
        refs = train_refs
    elif args.split == "validation":
        refs = validation_refs
    else:
        refs = np.concatenate([train_refs, validation_refs], axis=0)
    normalization = ValueNormalization.from_dict(checkpoint["value_normalization"])
    model = model_from_checkpoint(checkpoint, device)
    report = evaluate_model(model, shards, refs, normalization, device, args.batch_size, args.workers)
    report.update({"split": args.split, "checkpoint": str(args.checkpoint), "device": str(device)})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
