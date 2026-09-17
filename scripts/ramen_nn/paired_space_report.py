"""两侧 `ramen_space_bench` 逐局日志的配对分析。

输入是两份 `<csv>.journal.csv`（`plan_index` / `run_idx` 定位键 + 逐局单元格）。
配对按 `(plan_index, run_idx)`，并逐值核对两侧的世界 `seed` 相同——种子不同就不是
同一个随机世界，配对差没有意义。

置信区间按**组合聚类**：先在每个计划内对 8 局求配对差的均值，再用这些计划级均值
的样本标准差算标准误。同一副卡组的 8 局共享卡组与马娘，直接按局算标准误会把
组内相关当成独立样本，系统性低估区间。

不筛局：育成失败（`free_race_ok = 0`）照常计入均分，另外单独报失败率。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def read_journal(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    """读一份 journal CSV，按 (计划, 局号) 返回逐局单元格。"""

    rows: dict[tuple[int, int], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["plan_index"]), int(row["run_idx"]))
            if key in rows:
                raise ValueError(f"{path}: 定位键 {key} 重复")
            rows[key] = row
    return rows


def mean(values: list[float]) -> float:
    """算术平均。"""

    return sum(values) / len(values)


def stdev(values: list[float]) -> float:
    """样本标准差（n-1）。"""

    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def summarize(groups: dict[str, list[float]]) -> list[tuple[str, int, float, float, float]]:
    """按组给出 (组名, 计划数, 均值, 标准误, 95% 半宽)。"""

    out = []
    for name in sorted(groups):
        values = groups[name]
        se = stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
        out.append((name, len(values), mean(values), se, 1.96 * se))
    return out


def _parse_args() -> argparse.Namespace:
    """解析命令行。"""

    parser = argparse.ArgumentParser(description="两臂逐局日志的配对分析")
    parser.add_argument("--baseline", type=Path, required=True, help="基线的 journal.csv")
    parser.add_argument("--candidate", type=Path, required=True, help="候选的 journal.csv")
    parser.add_argument("--plans", type=Path, required=True, help="计划清单 JSON（带 plan / fields）")
    parser.add_argument("--card-db", type=Path, default=Path("gamedata/cardDB.json"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args = _parse_args()
    base = read_journal(args.baseline)
    cand = read_journal(args.candidate)
    if set(base) != set(cand):
        raise ValueError(f"两侧局集合不同：基线独有 {len(set(base) - set(cand))}，候选独有 {len(set(cand) - set(base))}")

    with args.card_db.open(encoding="utf-8") as handle:
        types = {int(e["cardId"]): int(e["cardType"]) for e in json.load(handle).values()}
    names = ["速", "耐", "力", "根", "智"]
    plan_meta: dict[int, tuple[int, str]] = {}
    for row in json.loads(args.plans.read_text(encoding="utf-8"))["plans"]:
        counts = [0] * 5
        for card in row["fields"][1:]:
            kind = types[card // 10]
            if kind < 5:
                counts[kind] += 1
        shape = "".join(f"{n}{names[i]}" for i, n in enumerate(counts) if n) + "1友"
        plan_meta[int(row["plan"])] = (int(row["fields"][0]), shape)

    per_plan_diff: dict[int, list[float]] = defaultdict(list)
    base_scores: list[float] = []
    cand_scores: list[float] = []
    base_fail = cand_fail = 0
    seed_mismatch = 0
    for key in sorted(base):
        b, c = base[key], cand[key]
        if b["seed"] != c["seed"]:
            seed_mismatch += 1
        bs, cs = float(b["score"]), float(c["score"])
        base_scores.append(bs)
        cand_scores.append(cs)
        base_fail += b["free_race_ok"] != "1"
        cand_fail += c["free_race_ok"] != "1"
        per_plan_diff[key[0]].append(cs - bs)
    if seed_mismatch:
        raise ValueError(f"{seed_mismatch} 局两侧的世界种子不同，配对不成立")

    plan_means = {p: mean(v) for p, v in per_plan_diff.items()}
    overall = summarize({"全部": list(plan_means.values())})[0]
    by_uma = defaultdict(list)
    by_shape = defaultdict(list)
    for plan, value in plan_means.items():
        uma, shape = plan_meta[plan]
        by_uma[str(uma)].append(value)
        by_shape[shape].append(value)

    report = {
        "games_per_side": len(base),
        "plans": len(plan_means),
        "baseline_mean": mean(base_scores),
        "candidate_mean": mean(cand_scores),
        "paired_diff_mean": overall[2],
        "paired_diff_se_by_combo": overall[3],
        "paired_diff_ci95": [overall[2] - overall[4], overall[2] + overall[4]],
        "plans_positive": sum(1 for v in plan_means.values() if v > 0),
        "plans_negative": sum(1 for v in plan_means.values() if v < 0),
        "baseline_free_race_fail": base_fail,
        "candidate_free_race_fail": cand_fail,
        "by_uma": [dict(zip(("key", "plans", "mean", "se", "half95"), r)) for r in summarize(by_uma)],
        "by_shape": [dict(zip(("key", "plans", "mean", "se", "half95"), r)) for r in summarize(by_shape)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"每侧 {report['games_per_side']} 局 / {report['plans']} 个组合")
    print(f"基线均分 {report['baseline_mean']:.1f}，候选均分 {report['candidate_mean']:.1f}")
    print(f"配对差 {overall[2]:+.1f}  95%CI [{report['paired_diff_ci95'][0]:+.1f}, {report['paired_diff_ci95'][1]:+.1f}]"
          f"（按组合聚类，SE {overall[3]:.1f}）")
    print(f"组合级正/负 {report['plans_positive']}/{report['plans_negative']}")
    print(f"自选比赛未达标：基线 {base_fail} 局，候选 {cand_fail} 局（均计入均分，未剔除）")
    for title, rows in (("按马娘", report["by_uma"]), ("按卡组构成", report["by_shape"])):
        print(f"\n{title}")
        for r in rows:
            print(f"  {r['key']:<12} 组合 {r['plans']:>4}  配对差 {r['mean']:+9.1f}  "
                  f"95%CI [{r['mean'] - r['half95']:+9.1f}, {r['mean'] + r['half95']:+9.1f}]")
    print(f"\n已写出 {args.output}")


if __name__ == "__main__":
    main()
