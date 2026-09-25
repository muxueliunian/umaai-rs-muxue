#!/usr/bin/env python3
"""合宿/满体力「一选休息」审计：解析 bench_base 决策日志 CSV。

输入：一个或多个 bench_base 跑批输出目录（各含 bench_base_decision_<build>_<run>.csv
与 bench_base_results.csv）。MCTS 与手写对照建议各跑一个目录（同名决策文件会互相覆盖）。

用法：
    python3 scripts/analyze_rest_picks.py \
        --dirs logs/rest_audit_mcts logs/rest_audit_hand \
        --builds spd2_sta0,speed_wisdom --runs 10 [--out logs/rest_audit]

判定口径（与代码一致）：
- 合宿回合：`is_xiahesu()` = turn ∈ [36,40) ∪ [60,64)
- 休憩门限：`vital_rest=45`（不吃面回合低于此值强制休息）、`rest_target_vital=55`
  （休息估值目标线：vital≥55 时休息只剩基础值 20）
- "一选休息"：某局合宿回合的 Train 阶段决策（candidates>1，休息必在候选中）
  休息占比 = 100%
- 体力分桶（决策时体力）：vital>55 / 45<vital≤55 / vital≤45

输出：
- 控制台逐 build×局×训练员汇总表 + 疑难清单（合宿全休息 / 高体力仍休息）
- logs/rest_audit_summary.csv（每局一行指标）
- logs/rest_audit_detail.csv（每条 Train 决策明细）
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

# 与 BaseGame::is_xiahesu 同口径：turn ∈ [36,40) ∪ [60,64)
CAMP_INTERVALS = [(36, 40), (60, 64)]


def is_camp(turn: int) -> bool:
    return any(lo <= turn < hi for lo, hi in CAMP_INTERVALS)


def load_decision_rows(directory: str):
    """返回 {build: {run_idx: [rows]}}；rows 为 dict。"""
    out = defaultdict(dict)
    if not os.path.isdir(directory):
        return out
    for fname in sorted(os.listdir(directory)):
        if not fname.startswith("bench_base_decision_") or not fname.endswith(".csv"):
            continue
        # bench_base_decision_<build>_<run>.csv
        tail = fname[len("bench_base_decision_"):-len(".csv")]
        build, _, run = tail.rpartition("_")
        if not run.isdigit():
            continue
        path = os.path.join(directory, fname)
        with open(path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        out[build][int(run)] = rows
    return out


def load_scores(directory: str):
    """{build: {run_idx: score}}：results.csv 行序 = 每 build 的局号序（按 build 分组归位）。"""
    out = defaultdict(dict)
    path = os.path.join(directory, "bench_base_results.csv")
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for i, r in enumerate(rows):
        out[r["build"]][len(out[r["build"]])] = int(r["score"])
    return out


def is_rest(row) -> bool:
    """休息判定：动作文本含"休息"（Train 阶段休息操作 / 合并吃面+休息）。"""
    return "休息" in row.get("action_desc", "")


def rows_for_stage(rows, stage: str = "Train"):
    """筛选指定阶段、candidates>1（有选择空间）的决策行。"""
    return [r for r in rows if r.get("stage") == stage and int(r.get("candidates", 0)) > 1]


def bucket(vital: int) -> str:
    if vital > 55:
        return "vital>55"
    if vital > 45:
        return "45<vital<=55"
    return "vital<=45"


def analyze_run(rows) -> dict:
    """单局指标（只统计 Train 阶段有选择空间的决策）。"""
    train = rows_for_stage(rows)
    camp = [r for r in train if is_camp(int(r["turn"]))]
    camp_rest = [r for r in camp if is_rest(r)]
    by_bucket = defaultdict(lambda: [0, 0])  # bucket -> [n, rest]
    for r in train:
        by_bucket[bucket(int(r["vital"]))][0] += 1
        if is_rest(r):
            by_bucket[bucket(int(r["vital"]))][1] += 1
    return {
        "camp_decisions": len(camp),
        "camp_rest": len(camp_rest),
        "camp_rest_rate": len(camp_rest) / len(camp) if camp else None,
        "camp_all_rest": len(camp) > 0 and len(camp_rest) == len(camp),
        "by_bucket": dict(by_bucket),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="bench_base 输出目录（可多个）")
    ap.add_argument("--builds", required=True, help="build 名，逗号分隔")
    ap.add_argument("--runs", type=int, default=10, help="每 build 局数（默认 10）")
    ap.add_argument("--out", default="logs/rest_audit", help="汇总输出前缀（默认 logs/rest_audit）")
    args = ap.parse_args()

    builds = [b.strip() for b in args.builds.split(",") if b.strip()]
    os.makedirs(args.out, exist_ok=True)

    sum_rows = []
    det_rows = []
    print("===== 合宿/满体力「一选休息」审计 =====")
    for directory in args.dirs:
        tag = os.path.basename(directory.rstrip("/")) or directory
        dec = load_decision_rows(directory)
        scores = load_scores(directory)
        print(f"\n##### 目录 {directory}（标签 {tag}）#####")
        for build in builds:
            run_map = dec.get(build, {})
            score_map = scores.get(build, {})
            if not run_map:
                print(f"[{build}] 无决策日志，跳过")
                continue
            print(f"\n--- {build}（{tag}）：{len(run_map)} 局有日志 ---")
            for run in range(args.runs):
                rows = run_map.get(run)
                if rows is None:
                    continue
                a = analyze_run(rows)
                score = score_map.get(run)
                flag = []
                if a["camp_all_rest"]:
                    flag.append("!!合宿全休息")
                for b, (n, rest) in sorted(a["by_bucket"].items()):
                    if b == "vital>55" and rest > 0:
                        flag.append(f">55休息{rest}/{n}")
                rate = f"{a['camp_rest_rate']*100:.0f}%" if a["camp_rest_rate"] is not None else "-"
                print(
                    f"  局{run:02d} 分={score if score is not None else '?':>6} "
                    f"合宿决策={a['camp_decisions']:>2} 休息={a['camp_rest']:>2} ({rate:>3})"
                    + "".join(f" {b}={rest}/{n}" for b, (n, rest) in sorted(a["by_bucket"].items()))
                    + ("  " + " ".join(flag) if flag else "")
                )
                sum_rows.append({
                    "directory": tag, "build": build, "run": run, "score": score or "",
                    **{k: v for k, v in a.items() if k != "by_bucket"},
                    **{f"{b}_n": n for b, (n, _) in sorted(a["by_bucket"].items())},
                    **{f"{b}_rest": r for b, (_, r) in sorted(a["by_bucket"].items())},
                })
                for r in rows_for_stage(rows):
                    det_rows.append({
                        "directory": tag, "build": build, "run": run,
                        "turn": r["turn"], "stage": r["stage"], "candidates": r["candidates"],
                        "vital": r["vital"], "camp": int(is_camp(int(r["turn"]))),
                        "rest": int(is_rest(r)), "action_desc": r["action_desc"],
                    })

    if not sum_rows:
        print("\n未找到任何决策日志（检查 --dirs / --builds / --runs 口径）")
        sys.exit(1)

    # 汇总 CSV
    sum_keys = list(sum_rows[0].keys())
    sum_path = os.path.join(args.out, "rest_audit_summary.csv")
    with open(sum_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=sum_keys)
        w.writeheader()
        w.writerows(sum_rows)
    det_path = os.path.join(args.out, "rest_audit_detail.csv")
    with open(det_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(det_rows[0].keys()))
        w.writeheader()
        w.writerows(det_rows)
    print(f"\n汇总: {sum_path}\n明细: {det_path}")


if __name__ == "__main__":
    main()