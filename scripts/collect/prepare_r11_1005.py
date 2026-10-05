"""准备 R11：合入超级拉面修复后的新规则下，gen2_v1 每候选 256、分轮按时截止的采集清单。

只读取真实 Rust 计划 dump 与历史组合清单，不编译、不采集、不计算哈希。
先在工作区根目录用 Release 跑 dump 单测，把完整输出存成 `<dump-dir>/gen2_v1.txt`：
  cargo test --release --lib -p umasim sampler::tests::test_gen2_v1_plan_dump -- --exact --nocapture
然后（--source-root 指向保存历史留出/开发/面板清单的工作区，只读）：
  python scripts/collect/prepare_r11_1005.py --dump-dir target/r11_plan_dumps_1005 \
      --source-root <历史工作区> --history-root <历史工作区>/training_data \
      --history-root <历史工作区>/scripts/collect
输出目录已存在时拒绝；排除来源缺任何一份也拒绝。
"""

import argparse
import json
from pathlib import Path

from prepare_gen2_formal import ROOT, enumerate_plans, prepare, write_json

COLLECT = ROOT / "scripts/collect"
BATCH = "r11_gen2n256_1005"
RECIPE = COLLECT / "gen2_v1_recipe.json"
MODEL = "saved_models/arms/ens_R8A_g123.onnx"
MODEL_ID = "ens_R8A_g123"
SEARCH_N = 256
# 每轮一万根：R9 gen2 四万根配额的四分之一（构成 9:9:13:9，层 general:Y1:Y2:Y3 同 R8/R9）
ROUND_TARGETS = [[1974, 40, 118, 118], [1974, 40, 118, 118],
                 [2850, 54, 173, 173], [1974, 40, 118, 118]]
ROUNDS = 40
SECONDS = 79200  # 22 小时：24 小时云端预算里留出构建、冒烟与打包
RESERVATION = [8000000000, 12000000000]
SMOKE_LOCAL = [12000000000, 12010000000]
SMOKE_CLOUD = [12010000000, 12020000000]

# 排除来源（相对 --source-root）。类别只用于报告；凡在 gen2_v1 内的组合一律不采。
SOURCES = [
    ("闭环留出", "scripts/collect/r8_gen2_0920/holdout.json"),
    ("开发验证", "target/train_r8_0920/dev_validation_combos.json"),
    ("评测面板", "logs/gen1_regression_0914/gen1_plans.json"),
    ("评测面板", "logs/uma113101_recheck_0914/plans_113101.json"),
    ("评测面板", "logs/region_panel_0914/panel_plans.json"),
    ("评测面板", "logs/topdeck40_0919/topdeck40.json"),
    ("评测面板", "logs/deck3_0919/plans_2s2e1w.json"),
    ("评测面板", "logs/deck3_0919/plans_3s1e1w.json"),
    ("评测面板", "logs/deck3_0919/plans_2s1e2w_名将.json"),
    ("评测面板", "logs/deck3_0919/plans_2s1e2w_樱花.json"),
    ("评测面板", "logs/gap_0919/plans40.json"),
    ("评测面板", "logs/gap_0919/plans_pilot.json"),
    ("评测面板", "logs/search_curve_0920/plans9.json"),
    ("评测面板", "logs/search_curve_0920/plans9x64.json"),
    ("评测面板", "logs/region_nn_confirm_0921/plans9x11.json"),
    ("评测面板", "logs/region_nn_pr_pair_0923/plans9.json"),
    ("评测面板", "logs/region_nn_pr_pair_0923/plans_first2.json"),
    ("评测面板", "logs/whole_nn_pair_0923/plans9.json"),
    ("评测面板", "logs/leaf_value_allhistory_0916/run01/sweep/summary.json"),
    *[("评测面板", f"logs/leafh3_0919/refs/plan_{p}.json")
      for p in (154, 412, 865, 1145, 1599, 2006, 2979, 3259, 3423, 4037)],
    ("筛查/冒烟", "logs/region_panel_0914/screen_plans.json"),
    ("筛查/冒烟", "logs/region_panel_0914/smoke_plans.json"),
    *[("筛查/冒烟", f"logs/leaf_value_allhistory_0916/run01/{p}/summary.json")
      for p in ("smoke_timing", "tests/smoke_late", "tests/smoke_wiring", "tests/t2_boundaries",
                "timing/r1b", "timing/r2b", "timing/r3b")],
    # 其他空间的闭环留出：与 gen2_v1 无交集，列出来让报告显式给出 0
    *[("其他空间留出", f"scripts/collect/{b}/holdout.json")
      for b in ("r8_2s1e2w_0920", "r8_newuma_0920", "r8_newcard_0920", "r8_newboth_0920",
                "r10_brian_1004", "r10_admire_1004", "r10_dualwis_1004", "r10_newuma_1004")],
]


def combo_of(entry):
    """把各种历史清单条目归一成 (马娘, 六卡升序)；认不出的条目返回 None。"""
    if isinstance(entry, list) and len(entry) == 7:
        return (int(entry[0]),) + tuple(sorted(int(v) for v in entry[1:]))
    if not isinstance(entry, dict):
        return None
    fields = entry.get("fields")
    if isinstance(fields, list) and len(fields) == 7:
        return (int(fields[0]),) + tuple(sorted(int(v) for v in fields[1:]))
    cards = entry.get("cards", entry.get("deck"))
    if entry.get("uma") is not None and isinstance(cards, list) and len(cards) == 6:
        return (int(entry["uma"]),) + tuple(sorted(int(v) for v in cards))
    return None


def read_combos(path):
    """读取一份清单里全部组合；一条都认不出视为来源格式错误。"""
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = data if isinstance(data, list) else data.get("plans", data.get("combos", []))
    combos = {c for c in map(combo_of, rows) if c}
    if not combos:
        raise ValueError(f"排除来源里没有可识别的组合：{path}")
    return combos


def collect_exclusions(source_root, space_fields, holdout_fields):
    """按完整字段汇总排除组合，并给出每份来源的数量报告。"""
    excluded, report = {}, []
    for category, relative in SOURCES:
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"排除来源缺失：{relative}")
        combos = read_combos(path)
        inside = combos & space_fields
        for combo in inside:
            excluded.setdefault(combo, set()).add(relative)
        report.append(dict(category=category, source=relative, combos=len(combos),
                           in_space=len(inside), outside_holdout=len(inside - holdout_fields)))
    return excluded, report


def check_history(history_roots, ranges):
    """扫描历史采集与清单 manifest 的实际 index 及预留号段，任何相交都拒绝。"""
    checked, max_end = 0, 0
    for root in history_roots:
        if not root.is_dir():
            raise ValueError(f"历史目录不存在：{root}")
        for path in root.rglob("manifest.json"):
            mf = json.loads(path.read_text(encoding="utf-8-sig"))
            spans = []
            if mf.get("work_indices"):
                spans.append((min(mf["work_indices"]), max(mf["work_indices"]) + 1))
            if mf.get("index_reservation"):
                spans.append(tuple(mf["index_reservation"]))
            if "index_start" in mf and "index_end" in mf:
                spans.append((int(mf["index_start"]), int(mf["index_end"])))
            checked += 1
            for start, end in spans:
                max_end = max(max_end, end)
                for lo, hi in ranges:
                    if start < hi and end > lo:
                        raise ValueError(f"号段与历史相交：{path} [{start},{end}) vs [{lo},{hi})")
    return checked, max_end


def main():
    """核对 dump、汇总排除、检查历史号段后冻结分轮清单。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, required=True, help="真实 Rust dump 目录，内含 gen2_v1.txt")
    parser.add_argument("--source-root", type=Path, required=True, help="历史留出/开发/面板清单所在工作区（只读）")
    parser.add_argument("--history-root", type=Path, action="append", required=True,
                        help="历史采集数据或清单目录，可重复；扫描其中全部 manifest.json")
    parser.add_argument("--output-root", type=Path, default=COLLECT)
    args = parser.parse_args()
    output = (args.output_root / BATCH).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("清单输出必须位于当前工作区")
    if output.exists():
        raise FileExistsError(output)
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    if recipe.get("purchase_buffs", []) != []:
        raise ValueError("R11 不购买开局增益")
    plans = enumerate_plans(recipe)
    space_fields = {tuple(p["fields"]) for p in plans}
    holdout = read_combos(args.source_root / SOURCES[0][1])
    excluded, report = collect_exclusions(args.source_root.resolve(), space_fields, holdout)
    # 闭环留出由 prepare 按配方规则重算并写进 holdout.json，不放进 exclusions.json 重复记录
    extra = {f: s for f, s in excluded.items() if f not in holdout}
    # 本机冒烟段在冒烟前已核对空闲、冒烟后被本机冒烟占用，故这里只核对正式段与云端冒烟段
    checked, max_end = check_history([r.resolve() for r in args.history_root], [RESERVATION, SMOKE_CLOUD])
    print(f"历史 manifest 核对 {checked} 份，最大已登记 index 上界 {max_end}；新号段无相交")
    prepare(args.dump_dir / "gen2_v1.txt", output, recipe_path=RECIPE, recipe_id=f"gen2_v1_{BATCH}",
            model=MODEL, model_id=MODEL_ID, search_n=SEARCH_N, layer_targets=ROUND_TARGETS,
            index_start=RESERVATION[0], index_end=RESERVATION[1], seconds=SECONDS,
            friend_gate=True, rounds=ROUNDS, excluded=extra)
    held = {tuple(p["fields"]) for p in json.loads((output / "holdout.json").read_text(encoding="utf-8"))["plans"]}
    if held != holdout:
        raise ValueError("按配方重算的闭环留出与历史 R8 gen2 留出不一致")
    sampled = len(plans) - len(held) - len(extra)
    write_json(output / "exclusion_report.json", dict(
        space=recipe["space"]["version"], plan_count=len(plans), holdout=len(held),
        extra_excluded=len(extra), sampled_plans=sampled, smoke_local=SMOKE_LOCAL, smoke_cloud=SMOKE_CLOUD,
        sources=report))
    print(f"计划 {len(plans)}：闭环留出 {len(held)}，额外排除 {len(extra)}，参与采集 {sampled}")


if __name__ == "__main__":
    main()
