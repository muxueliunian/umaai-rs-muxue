#!/usr/bin/env python3
"""汇总 pt_favor_rate 扫参结果：每档 CSV → (score 均值, skill_pt 均值, Δ)。

用法: python scripts/summarize_ptfavor_scan.py [--log-dir logs] [--prefix ptfavor-deep]
--prefix 指定目录名前缀（深测目录为 logs/ptfavor-deep-<rate>，用 `--prefix ptfavor-deep`）；
默认前缀 `ptfavor` 即 logs/ptfavor-<rate>/。
基线 = 1.0 档，其余档输出相对基线的 Δscore / Δskill_pt（整局统计算法同 bench_base 汇总）。
"""
import argparse
import csv
from pathlib import Path


def load(out_csv: Path) -> list[dict]:
    with open(out_csv, newline="") as f:
        return list(csv.DictReader(f))


def mean(rows, key: str) -> float:
    return sum(float(r[key]) for r in rows) / len(rows)


def agg(rows) -> dict:
    return {
        "n": len(rows),
        "score": mean(rows, "score"),
        "skill_pt": mean(rows, "skill_pt"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument(
        "--rates",
        default="1.0,2.0,2.5,3.0",
        help="逗号分隔的档位；目录名为 <prefix>-<rate>",
    )
    ap.add_argument(
        "--prefix",
        default="ptfavor",
        help="目录名前缀（深测目录为 logs/ptfavor-deep-<rate>，传 ptfavor-deep）",
    )
    ap.add_argument("--baseline", default="1.0", help="基线档，默认 1.0")
    args = ap.parse_args()
    rates = args.rates.split(",")

    results: dict[str, dict] = {}
    for rate in rates:
        csv_path = Path(args.log_dir) / f"{args.prefix}-{rate}" / "bench_base_results.csv"
        if not csv_path.exists():
            print(f"[skip] 缺失: {csv_path}")
            continue
        rows = load(csv_path)
        results[rate] = agg(rows)
        print(f"[load] {csv_path}  n={len(rows)}")

    if not results:
        raise SystemExit("无任何档位数据")

    base = results[args.baseline]
    print(
        f"\n{'rate':>6} {'n':>4} {'score':>9} {'Δscore':>9} {'skill_pt':>9} {'Δpt':>8}"
    )
    for rate in rates:
        if rate not in results:
            continue
        r = results[rate]
        d_score = r["score"] - base["score"]
        d_pt = r["skill_pt"] - base["skill_pt"]
        mark = "   ← 基线" if rate == args.baseline else ""
        print(
            f"{rate:>6} {r['n']:>4} {r['score']:>9.1f} {d_score:>+9.1f} "
            f"{r['skill_pt']:>9.1f} {d_pt:>+8.1f}{mark}"
        )


if __name__ == "__main__":
    main()