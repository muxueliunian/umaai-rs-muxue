"""准备 R10 无购买增益清单：只读取已有 Rust dump，不编译、不采集。

先在工作区根目录运行 Release dump 测试，将完整输出保存为对应空间名.txt：
  cargo test --release --lib -p umasim sampler::tests::test_gen4_brian_plan_dump -- --exact --nocapture
其余测试名为 test_gen4_admire_plan_dump / test_gen4_dualwis_plan_dump /
test_gen4_newuma_plan_dump / test_gen2_v1_plan_dump。
然后运行：
  python scripts/collect/prepare_r10_1004.py --dump-dir target/r10_plan_dumps_1004
--batches 可分批生成；所有选中输入均通过逐字段比对后才开始写入全新目录。
"""

import argparse
import json
from pathlib import Path

from prepare_gen2_formal import ROOT, enumerate_plans, prepare

COLLECT = ROOT / "scripts/collect"
MODEL = "saved_models/arms/ens_R8A_g123.onnx"
MODEL_ID = "ens_R8A_g123"
SEARCH_N = 1024
SINGLE_TARGETS = [[528, 12, 30, 30], [528, 12, 30, 30],
                  [1056, 24, 60, 60], [528, 12, 30, 30], [2640, 60, 150, 150]]
BATCHES = {
    "brian": dict(version="gen4_brian_v1", recipe="gen4_brian_recipe.json",
                  targets=SINGLE_TARGETS, reservation=[6000000000, 6400000000], seconds=7200),
    "admire": dict(version="gen4_admire_v1", recipe="gen4_admire_recipe.json",
                   targets=SINGLE_TARGETS, reservation=[6400000000, 6800000000], seconds=7200),
    "dualwis": dict(version="gen4_dualwis_v1", recipe="gen4_dualwis_recipe.json",
                    targets=[[2640, 60, 150, 150]], reservation=[6800000000, 6900000000], seconds=3600),
    "newuma": dict(version="gen4_newuma_v1", recipe="gen4_newuma_recipe.json",
                   targets=[[1056, 24, 60, 60], [1056, 24, 60, 60],
                            [2112, 48, 120, 120], [1056, 24, 60, 60], [5280, 120, 300, 300]],
                   reservation=[6900000000, 7400000000], seconds=14400),
    "control": dict(version="gen2_v1", recipe="gen2_v1_recipe.json",
                    targets=[[660, 15, 37, 38] for _ in range(4)],
                    reservation=[7400000000, 7500000000], seconds=3600),
}


def read_recipe(batch):
    """读取空间定义，控制组直接沿用原 gen2 配方与留出规则。"""
    return json.loads((COLLECT / batch["recipe"]).read_text(encoding="utf-8"))


def check_recipes():
    """独立枚举五空间并核对配额、身份互斥与独占号段容量。"""
    all_plans = {}
    seen = set()
    intervals = []
    total = 0
    for name, batch in BATCHES.items():
        recipe = read_recipe(batch)
        plans = enumerate_plans(recipe)
        space = recipe["space"]
        if space["version"] != batch["version"] or len(plans) != space["plan_count"]:
            raise ValueError(f"{name}: 空间版本或枚举总数与配方不一致")
        rows = batch["targets"]
        if len(rows) != len(space["shapes"]) or any(len(row) != 4 or min(row) <= 0 for row in rows):
            raise ValueError(f"{name}: 构成 × 四层配额非法")
        fields = {tuple(plan["fields"]) for plan in plans}
        if len(fields) != len(plans) or seen.intersection(fields):
            raise ValueError(f"{name}: 空间内或跨空间存在重复的完整马卡组合")
        seen.update(fields)
        target = sum(map(sum, rows))
        start, end = batch["reservation"]
        cycle = (start + len(plans) - 1) // len(plans)
        if start >= end or (cycle + target * 2) * len(plans) >= end:
            raise ValueError(f"{name}: 号段不足以容纳双倍候选序号")
        if any(start < other_end and end > other_start for other_start, other_end in intervals):
            raise ValueError(f"{name}: 正式号段相交")
        if name != "control" and (recipe.get("holdout_rule") != "every_tenth_all"
                                  or recipe.get("purchase_buffs") != []):
            raise ValueError(f"{name}: 必须沿用完整组合留出且不添加购买增益")
        intervals.append((start, end))
        total += target
        all_plans[name] = plans
    if total != 30000:
        raise ValueError(f"R10 有效根总目标错误：{total} != 30000")
    return all_plans


def check_dump(path, plans, recipe):
    """核对真实 Rust dump 的计划序号、马娘、卡组及构成原文。"""
    dumped = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("GEN2PLAN "):
            _, index, uma, deck, shape = line.split(maxsplit=4)
            dumped.append((int(index), int(uma), [int(card) for card in deck.split(",")], shape))
    expected = [(plan["plan"], plan["uma"], plan["deck"],
                 recipe["space"]["shapes"][plan["shape"]]["name"]) for plan in plans]
    if dumped != expected:
        raise ValueError(f"{path.name}: Rust dump 与独立枚举字段或顺序不一致")


def prepare_batches(dump_dir, output_root, names):
    """先验完所有选中输入，再复用原生成器冻结清单；既有输出不覆盖。"""
    dump_dir, output_root = dump_dir.resolve(), output_root.resolve()
    if not dump_dir.is_relative_to(ROOT) or not output_root.is_relative_to(ROOT):
        raise ValueError("Rust dump 与清单输出必须位于当前工作区")
    if not names or len(set(names)) != len(names):
        raise ValueError("批次不能为空或重复")
    plans = check_recipes()
    for name in names:
        batch = BATCHES[name]
        check_dump(dump_dir / f"{batch['version']}.txt", plans[name], read_recipe(batch))
        output = output_root / f"r10_{name}_1004"
        if not output.resolve().is_relative_to(ROOT):
            raise ValueError("清单输出经软链接越出工作区")
        if output.exists():
            raise FileExistsError(output)
    for name in names:
        batch = BATCHES[name]
        print(f"R10 {name}: 不施加开局购买增益")
        prepare(dump_dir / f"{batch['version']}.txt", output_root / f"r10_{name}_1004",
                recipe_path=COLLECT / batch["recipe"], recipe_id=f"{batch['version']}_r10_1004",
                model=MODEL, model_id=MODEL_ID, search_n=SEARCH_N, layer_targets=batch["targets"],
                index_start=batch["reservation"][0], index_end=batch["reservation"][1],
                seconds=batch["seconds"], friend_gate=True)


def main():
    """解析平台通用入口参数，默认准备五批三万根正式清单。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, required=True,
                        help="真实 Rust dump 目录，文件名为 <space_version>.txt")
    parser.add_argument("--output-root", type=Path, default=COLLECT)
    parser.add_argument("--batches", choices=list(BATCHES), nargs="+", default=list(BATCHES))
    args = parser.parse_args()
    prepare_batches(args.dump_dir, args.output_root, args.batches)


if __name__ == "__main__":
    main()
