"""直接比较两种 rollout 预算下生成的标签差多少（描述性，不做推断）。

用途：回答「标得更精确到底改了什么」。两套标签来自**同一批根、同一批候选**，
只是列宽不同（例如 1024 与 2048），所以可以逐样本对齐比较：

- `policy_target` 的逐样本 KL（以列宽更大的一侧为参照分布）与 top-1 格位不一致率；
- `value_target` 三路的平均绝对差。

对齐靠两侧的 `index.npy` **逐值相同**，不算任何指纹。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def kl(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    """逐行 KL(reference ‖ other)，只在参照侧为正的格位上求和。"""

    mask = reference > 0.0
    ratio = np.where(mask, reference / np.maximum(other, 1e-12), 1.0)
    return np.sum(np.where(mask, reference * np.log(ratio), 0.0), axis=1)


def _parse_args() -> argparse.Namespace:
    """解析命令行。"""

    parser = argparse.ArgumentParser(description="比较两种 rollout 预算下的标签差异")
    parser.add_argument("--reference", type=Path, required=True, help="参照标签根目录（列宽更大的一侧）")
    parser.add_argument("--other", type=Path, required=True, help="对照标签根目录")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args = _parse_args()
    names = sorted(p.name for p in args.reference.iterdir() if (p / "labels.json").is_file())
    total = 0
    kl_sum = 0.0
    kl_max = 0.0
    top1_diff = 0
    value_abs = np.zeros(3, dtype=np.float64)
    rows = []
    for name in names:
        ref_dir, oth_dir = args.reference / name, args.other / name
        ref_idx = np.load(ref_dir / "index.npy", allow_pickle=False)
        oth_idx = np.load(oth_dir / "index.npy", allow_pickle=False)
        if not np.array_equal(ref_idx, oth_idx):
            raise ValueError(f"{name}: 两侧 index 不是逐值相同，无法对齐")
        ref_p = np.load(ref_dir / "policy_target.npy", allow_pickle=False).astype(np.float64)
        oth_p = np.load(oth_dir / "policy_target.npy", allow_pickle=False).astype(np.float64)
        ref_v = np.load(ref_dir / "value_target.npy", allow_pickle=False).astype(np.float64)
        oth_v = np.load(oth_dir / "value_target.npy", allow_pickle=False).astype(np.float64)
        divergence = kl(ref_p, oth_p)
        mismatch = int(np.count_nonzero(np.argmax(ref_p, axis=1) != np.argmax(oth_p, axis=1)))
        value = np.mean(np.abs(ref_v - oth_v), axis=0)
        n = len(ref_idx)
        rows.append({
            "dir": name,
            "samples": n,
            "policy_kl_mean": float(divergence.mean()),
            "top1_mismatch": mismatch,
            "top1_mismatch_rate": mismatch / n,
            "value_mae": value.tolist(),
        })
        total += n
        kl_sum += float(divergence.sum())
        kl_max = max(kl_max, float(divergence.max()))
        top1_diff += mismatch
        value_abs += value * n
        print(f"{name:<42} n={n:>5}  KL {divergence.mean():.5f}  top1 不一致 {mismatch:>4} ({100 * mismatch / n:5.2f}%)")

    summary = {
        "reference": str(args.reference),
        "other": str(args.other),
        "samples": total,
        "policy_kl_mean": kl_sum / total,
        "policy_kl_max": kl_max,
        "top1_mismatch": top1_diff,
        "top1_mismatch_rate": top1_diff / total,
        "value_mae": (value_abs / total).tolist(),
        "by_dir": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n合计 {total} 条：policy KL 均值 {summary['policy_kl_mean']:.5f}（最大 {kl_max:.4f}），"
          f"top-1 不一致 {top1_diff}（{100 * summary['top1_mismatch_rate']:.2f}%）")
    print(f"value MAE 三路 {[round(v, 2) for v in summary['value_mae']]}")
    print(f"已写出 {args.output}")


if __name__ == "__main__":
    main()
