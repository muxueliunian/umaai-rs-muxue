"""按列窗口 `[0, width)` 从 raw `.npy` 目录派生一个较小 rollout 预算的数据目录。

同一批根局面在采集时录了 `search_n = 4096` 条有序 rollout。取前 N 列即可得到
「同样的根、同样的候选、只是 rollout 预算更小」的对照组，无需重新模拟。

❗只截 `cand_scores` 是错的：`cand_n` / `cand_mean` / `cand_stdev` 都是对全部列
求出来的汇总量，必须一并重算，否则标签会拿 4096 列的统计去配 N 列的分数。
`labels.py` 会拦住 `cand_n` 不一致，但拦不住 `cand_mean`。

逐样本对齐用的 `x` / `stage` / `turn` / `index` / `legal_mask` / `cand_ptr` /
`cand_slots` / `combo_key` 与列数无关，原样复制。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

# 与列数无关、直接复制的数组
PASSTHROUGH = ("x", "stage", "turn", "index", "legal_mask", "cand_ptr", "cand_slots", "combo_key")
# 每次处理的候选行数，避免一次性把 f64 中间量摊平到内存
CHUNK = 4096


def _recompute(scores: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在截断后的窗口上重算有效次数、均值与样本标准差（n-1 分母）。

    统计一律用 float64 累加，最后再落回导出器使用的 f32 / i32。
    """

    rows = scores.shape[0]
    n_out = np.empty(rows, dtype=np.int32)
    mean_out = np.empty(rows, dtype=np.float32)
    stdev_out = np.empty(rows, dtype=np.float32)
    for begin in range(0, rows, CHUNK):
        end = min(begin + CHUNK, rows)
        block = np.asarray(scores[begin:end], dtype=np.float64)
        mask = np.asarray(valid[begin:end], dtype=bool)
        block = np.where(mask, block, 0.0)
        cnt = mask.sum(axis=1, dtype=np.int64)
        if np.any(cnt < 2):
            bad = int(np.argmin(cnt)) + begin
            raise ValueError(f"候选 {bad} 在窗口内只有 {int(cnt[bad - begin])} 个有效 rollout，不足 2 个")
        total = block.sum(axis=1)
        mean = total / cnt
        # 方差用「有效位与均值的偏差平方和」，缺失位已被置零故要减去掩码
        dev = np.where(mask, block - mean[:, None], 0.0)
        var = (dev * dev).sum(axis=1) / (cnt - 1)
        n_out[begin:end] = cnt.astype(np.int32)
        mean_out[begin:end] = mean.astype(np.float32)
        stdev_out[begin:end] = np.sqrt(var).astype(np.float32)
    return n_out, mean_out, stdev_out


def truncate(source: Path, output: Path, width: int, overwrite: bool = False) -> dict:
    """派生一个 rollout 宽度为 `width` 的数据目录，返回新的 meta。

    # 错误

    源目录缺数组、窗口超出原宽度、目标目录已有内容而未加 `--overwrite` 时报错。
    """

    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("源目录与目标目录相同")
    scores = np.load(source / "cand_scores.npy", mmap_mode="r", allow_pickle=False)
    valid_path = source / "cand_valid.npy"
    if not valid_path.is_file():
        raise ValueError(f"{source} 没有 cand_valid.npy，无法判定窗口内的失败槽位")
    valid = np.load(valid_path, mmap_mode="r", allow_pickle=False)
    if valid.shape != scores.shape:
        raise ValueError("cand_valid 与 cand_scores 形状不一致")
    total_width = int(scores.shape[1])
    if not 2 <= width <= total_width:
        raise ValueError(f"窗口宽度 {width} 超出原始宽度 {total_width}")
    meta = json.loads((source / "meta.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    if not overwrite and (output / "cand_scores.npy").exists():
        raise FileExistsError(f"{output} 已有数据；确认后加 --overwrite")

    for name in PASSTHROUGH:
        src = source / f"{name}.npy"
        if src.is_file():
            shutil.copyfile(src, output / f"{name}.npy")

    rows = int(scores.shape[0])
    out_scores = np.lib.format.open_memmap(
        output / "cand_scores.npy", mode="w+", dtype=scores.dtype, shape=(rows, width)
    )
    out_valid = np.lib.format.open_memmap(
        output / "cand_valid.npy", mode="w+", dtype=valid.dtype, shape=(rows, width)
    )
    for begin in range(0, rows, CHUNK):
        end = min(begin + CHUNK, rows)
        out_scores[begin:end] = scores[begin:end, :width]
        out_valid[begin:end] = valid[begin:end, :width]
    out_scores.flush()
    out_valid.flush()

    cand_n, cand_mean, cand_stdev = _recompute(out_scores, out_valid)
    np.save(output / "cand_n.npy", cand_n)
    np.save(output / "cand_mean.npy", cand_mean)
    np.save(output / "cand_stdev.npy", cand_stdev)

    # 与源目录 Rust 侧汇总量的差异：源汇总由 f64 累加得到，这里是从落盘的 f32
    # 分数重算，同宽度下也不必逐位相等。差异大到肉眼可见才是问题。
    drift = None
    if width == total_width:
        ref = np.load(source / "cand_mean.npy", allow_pickle=False)
        drift = float(np.max(np.abs(ref.astype(np.float64) - cand_mean.astype(np.float64))))

    meta["stats"] = dict(meta.get("stats", {}))
    meta["stats"]["rollout_width"] = width
    meta["rollout_budget"] = width
    meta["truncation"] = {
        "source": str(source.name),
        "source_width": total_width,
        "window": [0, width],
        "collected_search_n": meta.get("search_n"),
        "cand_mean_max_drift_vs_source": drift,
    }
    (output / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.name), **meta["truncation"], "candidates": rows}, ensure_ascii=False, indent=2))
    return meta


def _parse_args() -> argparse.Namespace:
    """解析命令行。"""

    parser = argparse.ArgumentParser(description="按列窗口派生较小 rollout 预算的 raw .npy 目录")
    parser.add_argument("--source", type=Path, required=True, help="含 cand_scores.npy 的原始 raw 目录")
    parser.add_argument("--output", type=Path, required=True, help="派生目录")
    parser.add_argument("--width", type=int, required=True, help="保留的列数 N，窗口固定为 [0, N)")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args = _parse_args()
    truncate(args.source, args.output, args.width, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
