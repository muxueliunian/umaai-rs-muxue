#!/usr/bin/env python3
"""deck CRN 配对对比：两个 bench_base_results.csv 按 seed 逐局配对分析。

用法: python3 scripts/deck_crn_compare.py <基准.csv> <对照.csv> [--cols score skill_pt]

同 seed 逐局 = 同一套随机数流（决策+规则 RNG），配对差只由卡组差异贡献。
输出：均值差、配对 t 检验、胜负局数、RMJ/自选比赛达标均值。
"""
import argparse
import csv
import math
import sys


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def paired_t(diff):
    n = len(diff)
    if n < 2:
        return float("nan"), float("nan"), n
    mean = sum(diff) / n
    var = sum((d - mean) ** 2 for d in diff) / (n - 1)
    se = math.sqrt(var / n)
    t = mean / se if se > 0 else float("nan")
    return mean, t, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_csv")
    ap.add_argument("treat_csv")
    ap.add_argument("--cols", default="score,skill_pt", help="逗号分隔的数值列")
    args = ap.parse_args()

    base = {r["seed"]: r for r in load(args.base_csv)}
    treat = {r["seed"]: r for r in load(args.treat_csv)}
    common = sorted(set(base) & set(treat))
    if not common:
        print("ERROR: 两个 CSV 没有相同 seed");
        sys.exit(1)
    print(f"配对局数: {len(common)}（共 base {len(base)} / treat {len(treat)}）")

    for col in args.cols.split(","):
        col = col.strip()
        diff = [float(treat[s][col]) - float(base[s][col]) for s in common]
        mean, t, n = paired_t(diff)
        wins = sum(1 for d in diff if d > 0)
        base_mean = sum(float(base[s][col]) for s in common) / n
        treat_mean = sum(float(treat[s][col]) for s in common) / n
        print(
            f"[{col}] base={base_mean:.1f} treat={treat_mean:.1f} Δ={mean:+.1f} "
            f"t={t:.2f} 胜/平/负={wins}/{n - wins - sum(1 for d in diff if d == 0)}/{n - wins - sum(1 for d in diff if d == 0) and 0 or (n - wins) - sum(1 for d in diff if d > 0)}"
        )
        # 简洁胜负统计
        neg = sum(1 for d in diff if d < 0)
        print(f"    ↑{wins} / ↓{neg} / ={n - wins - neg} | t={t:.2f}")

    for col in ("rmj_ok", "free_race_ok"):
        if col in base[common[0]]:
            b = sum(int(base[s][col]) for s in common) / len(common)
            t = sum(int(treat[s][col]) for s in common) / len(common)
            print(f"[{col}] base={b:.2f} treat={t:.2f}")


if __name__ == "__main__":
    main()