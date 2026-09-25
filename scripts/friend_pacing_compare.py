#!/usr/bin/env python3
"""友人出行配额（friendcap）配对对比：同 build 同 seed 配对差 + 结构指标。

用法：
    python3 scripts/friend_pacing_compare.py <结果根目录> [base 标签]

结果根目录下每个子目录为一次 bench_base 跑批（含 bench_base_results.csv），
命名约定 `<uma>_<标签>`。base 标签缺省为 `base`。
"""
import csv
import os
import statistics
import sys
from collections import defaultdict


def load(path):
    """读一次跑批结果，返回 {(build, seed): row}。"""
    f = os.path.join(path, "bench_base_results.csv")
    with open(f, encoding="utf-8") as fh:
        return {(r["build"], r["seed"]): r for r in csv.DictReader(fh)}


def paired(base, other):
    """同键配对差列表（缺失键跳过并报告）。"""
    keys = [k for k in base if k in other]
    missing = len(base) - len(keys)
    ds = [int(other[k]["score"]) - int(base[k]["score"]) for k in keys]
    return ds, missing


def fmt(ds):
    m = statistics.mean(ds)
    sd = statistics.stdev(ds) if len(ds) > 1 else 0.0
    se = sd / len(ds) ** 0.5
    t = m / se if se > 0 else float("nan")
    return m, se, 1.96 * se, t


def main():
    root = sys.argv[1]
    base_tag = sys.argv[2] if len(sys.argv) > 2 else "base"
    dirs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    groups = defaultdict(dict)
    for d in dirs:
        uma, _, tag = d.partition("_")
        if not tag or not os.path.exists(os.path.join(root, d, "bench_base_results.csv")):
            continue
        groups[uma][tag] = load(os.path.join(root, d))
    for uma, tags in sorted(groups.items()):
        if base_tag not in tags:
            print(f"[跳过] {uma}: 无 {base_tag} 档")
            continue
        base = tags[base_tag]
        bm = statistics.mean(int(r["score"]) for r in base.values())
        print(f"\n=== uma={uma}  base[{base_tag}] 均分={bm:.0f} n={len(base)} ===")
        for tag in sorted(tags):
            if tag == base_tag:
                continue
            ds, missing = paired(base, tags[tag])
            if not ds:
                print(f"  {tag:>10}: 无配对样本（缺 {missing}）")
                continue
            m, se, ci, t = fmt(ds)
            om = statistics.mean(int(r["score"]) for r in tags[tag].values())
            print(f"  {tag:>10}: Δ={m:+7.1f} ±{ci:.0f}  t={t:+.2f}  均分={om:.0f}  缺配对={missing}")
        print("  --- 结构指标（均值）---")
        cols = [
            "speed", "stamina", "power", "guts", "wisdom", "skill_pt",
            "scenario_pt_y1", "scenario_pt_y2", "scenario_pt_y3",
            "eat_count_y1", "eat_count_y2", "eat_count_y3",
            "friend_turns_y1", "friend_turns_y2", "friend_turns_y3",
        ]
        hdr = "  " + f"{'指标':>16}" + "".join(f"{t:>10}" for t in sorted(tags))
        print(hdr)
        for c in cols:
            line = f"  {c:>16}"
            for t in sorted(tags):
                line += f"{statistics.mean(int(r[c]) for r in tags[t].values()):>10.1f}"
            print(line)


if __name__ == "__main__":
    main()