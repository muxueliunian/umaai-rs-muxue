#!/usr/bin/env python3
"""单局日志（umaai 在线记录器产物）的高体力休息审计。

输入：`logs/game{id}/`（decisions.csv + 每个回合的 thisTurn.json 原文）。

与 bench_base 决策日志（`scripts/analyze_rest_picks.py`）不同：
- 列名用 `chosen_desc` 而非 `action_desc`
- 在线记录器不带 `vital` 列；决策时体力必须从对应回合的 thisTurn.json 解析
- 候选列表已按 cand1..cand5 平铺

判定口径与代码一致：
- 合宿回合：turn ∈ [36,40) ∪ [60,64)（与 BaseGame::is_xiahesu 同源）
- 休憩门限：vital_rest=45 / rest_target_vital=55
- "一选休息"：某局合宿回合 Train 决策中（candidates>1），休息在候选里、且被选中
- 体力分桶：决策时 current_vital（按 Train 阶段在 decision_row 之前已经发生计为当前回合快照值）

输出：控制台逐决策点明细 + logs/rest_audit_online_<id>.csv
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

CAMP_INTERVALS = [(36, 40), (60, 64)]


def is_camp(turn: int) -> bool:
    return any(lo <= turn < hi for lo, hi in CAMP_INTERVALS)


def bucket(v: int) -> str:
    if v > 55:
        return "vital>55"
    if v > 45:
        return "45<vital<=55"
    return "vital<=45"


def rest_in_cands(cand_descs):
    """候选列表里是否有 '休息'（决策点把休息列为候选）。"""
    for c in cand_descs:
        if c and "休息" in c:
            return True
    return False


def collect_turn_vitals(directory: str):
    """遍历 game{id}_turn{turn}[_{seq}].json，按 (turn, seq) 返回 vital。

    优先取 'command' 来源且 seq=0 的快照（玩家/AI 命令时点）；缺则退化取该回合任一快照的 vital。
    返回 {(turn, seq): vital} 与 {turn: vital}（best-effort 单值）。
    """
    pattern = re.compile(r"^game\d+_turn(\d+)(?:_(\d+))?\.json$")
    by_turn_seq = {}
    by_turn_best = {}  # 偏好 seq=0，再偏好 source=command
    files = sorted(os.listdir(directory))
    for fname in files:
        m = pattern.match(fname)
        if not m:
            continue
        turn = int(m.group(1))
        seq = int(m.group(2)) if m.group(2) is not None else 0
        path = os.path.join(directory, fname)
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        # 拉面协议顶层 ramen + baseGame；vital 直接在 baseGame.vital
        vital = None
        for path_keys in (
            ("baseGame", "vital"),
            ("vital",),
        ):
            cur = data
            ok = True
            for k in path_keys:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    ok = False
                    break
            if ok and isinstance(cur, (int, float)):
                vital = int(cur)
                break
        if vital is None:
            continue
        by_turn_seq[(turn, seq)] = vital
        # best 优先级：seq=0 胜出；同 seq 维持先到先得
        if turn not in by_turn_best or seq == 0:
            by_turn_best[turn] = (seq, vital)
    # 还原 best 为 turn -> vital
    return by_turn_seq, {t: v for t, (_, v) in by_turn_best.items()}


def analyze(directory: str):
    csv_path = os.path.join(directory, "decisions.csv")
    if not os.path.isfile(csv_path):
        print(f"无 decisions.csv: {csv_path}")
        return
    by_turn_seq, by_turn_best = collect_turn_vitals(directory)
    rows = []
    with open(csv_path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    # 关注 Train 阶段 + candidates>1 的 calc 行
    train_calcs = [
        r for r in rows
        if r.get("stage") == "Train"
        and r.get("outcome") == "calc"
        and r.get("decision_kind") == "train"
        and int(r.get("n_actions", 0)) > 1
    ]
    chosen_rest = [r for r in train_calcs if "休息" in r.get("chosen_desc", "")]

    print(f"=== {directory} ===")
    print(f"Train 决策点（candidates>1）：{len(train_calcs)}")
    print(f"其中选择休息：{len(chosen_rest)}")

    # 按回合+体力桶统计
    by_bucket = defaultdict(lambda: [0, 0])  # [n, rest]
    by_camp = [0, 0]
    high_vital_rest = []
    detail = []

    for r in train_calcs:
        turn = int(r["turn"])
        seq = int(r["seq"])
        cand_descs = [r.get(f"cand{i}_desc", "") or "" for i in range(1, 6)]
        cand_scores = [r.get(f"cand{i}_score", "") or "" for i in range(1, 6)]
        cand_n = [r.get(f"cand{i}_n", "") or "" for i in range(1, 6)]
        vital = by_turn_seq.get((turn, seq), by_turn_best.get(turn))
        is_rest = "休息" in r.get("chosen_desc", "")
        b = bucket(vital) if vital is not None else "vital=?"
        by_bucket[b][0] += 1
        if is_rest:
            by_bucket[b][1] += 1
        if is_camp(turn):
            by_camp[0] += 1
            if is_rest:
                by_camp[1] += 1
        detail.append({
            "turn": turn, "seq": seq, "vital": vital if vital is not None else "",
            "camp": int(is_camp(turn)),
            "rest_in_cands": int(rest_in_cands(cand_descs)),
            "chosen": r.get("chosen_desc", ""),
            "is_rest_chosen": int(is_rest),
            "chosen_score": r.get(f"cand{int(r['chosen_idx'])+1}_score", "") if r.get("chosen_idx") else "",
            "cand1_desc": cand_descs[0],
            "cand2_desc": cand_descs[1],
            "cand3_desc": cand_descs[2],
            "cand4_desc": cand_descs[3],
            "cand5_desc": cand_descs[4],
            "cand1_score": cand_scores[0],
            "cand2_score": cand_scores[1],
            "cand3_score": cand_scores[2],
            "cand4_score": cand_scores[3],
            "cand5_score": cand_scores[4],
            "cand1_n": cand_n[0],
            "cand2_n": cand_n[1],
            "cand3_n": cand_n[2],
            "cand4_n": cand_n[3],
            "cand5_n": cand_n[4],
        })
        if is_rest and vital is not None and vital > 55:
            high_vital_rest.append(detail[-1])

    print("\n[体力桶分布] bucket_n / bucket_rest")
    for b, (n, rest) in sorted(by_bucket.items()):
        print(f"  {b}: {n} 决策点 / {rest} 选休息")

    print(f"\n[合宿回合] Train 决策={by_camp[0]} / 选休息={by_camp[1]}")
    camp_all_rest = by_camp[0] > 0 and by_camp[0] == by_camp[1]
    print(f"合宿全部选休息：{camp_all_rest}")

    print(f"\n[高体力(vital>55)选休息] {len(high_vital_rest)} 次")
    for d in high_vital_rest:
        # 排序候选
        cands = []
        for i in range(1, 6):
            desc = d[f"cand{i}_desc"]
            sc = d[f"cand{i}_score"]
            n = d[f"cand{i}_n"]
            if desc:
                try:
                    cands.append((float(sc), desc, n))
                except (ValueError, TypeError):
                    cands.append((float("-inf"), desc, n))
        cands.sort(reverse=True)
        print(
            f"  t{d['turn']:>2} seq={d['seq']} vital={d['vital']} "
            f"camp={d['camp']} -> {d['chosen']} ({d['chosen_score']})"
        )
        for sc, desc, n in cands:
            mark = " <-- 选中" if desc == d["chosen"] else ""
            print(f"      {sc:>10.2f}  n={n:>5}  {desc}{mark}")

    # 落盘明细
    out = directory.rstrip("/") + "_rest_audit.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        if detail:
            w = csv.DictWriter(fh, fieldnames=list(detail[0].keys()))
            w.writeheader()
            w.writerows(detail)
    print(f"\n明细: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="单局日志目录（如 logs/game421）")
    args = ap.parse_args()
    analyze(args.dir)


if __name__ == "__main__":
    main()
