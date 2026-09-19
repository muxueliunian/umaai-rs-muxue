"""从 Rust 计划原文生成无哈希的正式清单；只准备文件，不启动采集。"""

import argparse
import itertools
import json
from math import gcd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# 分层的**固定身份**：层名与两个地区配额（千分之几）。每层采多少根由
# `--layer-targets` 在命令行给出，因为那是随轮次变化的预算决策；
# 而「哪几层、各自定向抓哪一年的地区选择」是配方结构，不随预算变。
# 顺序即 `--layer-targets` 的顺序，不要重排。
LAYER_SPEC = [("general", 0, "0,0"), ("region_y1", 1000, "0,0"),
              ("region_y2", 0, "1000,0"), ("region_y3", 0, "0,1000")]

# 任务**发射顺序**：最贵且最不可替代的层排最前。与 `LAYER_SPEC`（= `--layer-targets`
# 的参数顺序）故意分开，免得改了发射顺序就悄悄改掉同一条命令行的含义。
#
# 存在的理由：驱动顺序执行，撞上截止会在当前任务失败退出、**后面的任务根本不跑**。
# 而 region_y3 每根 120 候选，单根开销约为 general 的 9 倍（按候选-rollout 计占全轮 41%、
# 按冒烟实测耗时计约占三分之一），又是搜索相对手写增益最大的决策点
# （实测 g = W − V^手写 在 Y3 约 1300，普通根只有 ~100）。把它放最后，等于把最贵的
# 证据完全压在截断风险上。general 是可替换的大宗，放最后降级最平滑。
EMIT_ORDER = ["region_y3", "region_y2", "region_y1", "general"]


def write_json(path, value):
    """清单一计划一行、序号每行32项，便于审阅；不生成内容指纹。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    compact = lambda item: json.dumps(item, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            lines = [compact(item) for item in value]
        else:
            lines = [",".join(compact(item) for item in value[i:i + 32]) for i in range(0, len(value), 32)]
        text = "[\n" + ",\n".join(lines) + "\n]\n"
    elif isinstance(value, dict) and isinstance(value.get("plans"), list):
        header = json.dumps({k: v for k, v in value.items() if k != "plans"}, ensure_ascii=False, indent=2)
        text = header.rstrip()[:-1].rstrip() + ',\n  "plans": [\n' + ",\n".join(compact(item) for item in value["plans"]) + "\n  ]\n}\n"
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")


def enumerate_plans(recipe):
    """独立按卡表和角色唯一规则枚举，与 Rust dump 逐字段比较。"""
    db = json.loads((ROOT / "gamedata/cardDB.json").read_text(encoding="utf-8"))
    space = recipe["space"]
    plans = []
    for uma in space["umas"]:
        for shape_id, shape in enumerate(space["shapes"]):
            choices = []
            for kind, count in enumerate(shape["counts"]):
                cards = [c for c in space["cards"] if db[str(c // 10)]["cardType"] == kind
                         and db[str(c // 10)]["charaId"] != uma // 100]
                choices.append(list(itertools.combinations(cards, count)))
            for groups in itertools.product(*choices):
                deck = [c for group in groups for c in group] + [space["friend_card"]]
                charas = [uma // 100] + [db[str(c // 10)]["charaId"] for c in deck]
                if len(set(charas)) != 7:
                    continue
                plans.append(dict(plan=len(plans), uma=uma, deck=deck, shape=shape_id,
                                  fields=[uma] + sorted(deck)))
    return plans


def prepare(dump_path, output, *, recipe_path, recipe_id, model, model_id, search_n, layer_targets,
            index_start, index_end, seconds):
    """冻结计划、留出组合、分层配额及独立世界；只允许写全新目录。

    除 `dump_path` / `output` 外的参数全部来自命令行，**没有隐含默认**：
    每一轮采集的口径都必须在命令行上写出来，才能在 `args.json` 与 shell 历史里留痕。
    0914 那轮的取值见 `scripts/collect/formal2048_0914/manifest.json`，可原样重放。
    """
    if output.exists():
        raise FileExistsError(output)
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    shape_count = len(recipe["space"]["shapes"])
    plans = enumerate_plans(recipe)
    dumped = []
    for line in dump_path.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("GEN2PLAN "):
            _, index, uma, deck, shape = line.split(maxsplit=4)
            dumped.append((int(index), int(uma), [int(v) for v in deck.split(",")], shape))
    expected = [(p["plan"], p["uma"], p["deck"], recipe["space"]["shapes"][p["shape"]]["name"]) for p in plans]
    if dumped != expected or len(plans) != recipe["space"]["plan_count"]:
        raise ValueError("独立枚举与 Rust 计划字段/顺序不一致")

    # gen2_v1：只从已有证据明确未见的两张新速卡/新马娘中选泛化留出，其他新组合不猜历史覆盖。
    # every_tenth_all：整个空间都未见过（如 2速1耐2智 定向补采），每格全部计划参与留出。
    rule = recipe.get("holdout_rule", "unseen_new_cards")
    if rule not in ("unseen_new_cards", "every_tenth_all"):
        raise ValueError(f"未知留出规则 {rule}")
    held = []
    for uma in recipe["space"]["umas"]:
        for shape in range(shape_count):
            unseen = sorted([p for p in plans if p["uma"] == uma and p["shape"] == shape
                             and (rule == "every_tenth_all" or uma == 114101
                                  or any(c in p["deck"] for c in (303124, 303114)))],
                            key=lambda p: p["fields"])
            held.extend(unseen[9::10])
    held_ids = {p["plan"] for p in held}

    # 全区间独占；实际 index 不连续，index % 4288 严格定位原空间计划。
    reserved_start, reserved_end = index_start, index_end
    for path in (ROOT / "training_data").rglob("manifest.json"):
        mf = json.loads(path.read_text(encoding="utf-8-sig"))
        indices = mf.get("work_indices")
        if indices:
            if any(reserved_start <= i < reserved_end for i in indices):
                raise ValueError(f"号段已占用：{path}")
        elif int(mf.get("index_start", 0)) < reserved_end and int(mf.get("index_end", 0)) > reserved_start:
            raise ValueError(f"号段与既有 manifest 相交：{path}")
    cycle = (reserved_start + len(plans) - 1) // len(plans)
    # index = cycle×4288 + plan，而 cycle 每消耗一个候选序号就加一，故号段**跨度**由
    # 候选序号总数决定，与有效根目标不是一回事。放在这里提前算，免得循环跑到最后
    # 才撞上「index 越过独占号段」那条笼统的报错。
    attempts_total = 2 * sum(layer_targets) * len(recipe["space"]["shapes"])
    if (cycle + attempts_total) * len(plans) >= reserved_end:
        need = (cycle + attempts_total + 1) * len(plans)
        raise ValueError(
            f"号段宽度不足：本配方共 {attempts_total} 个候选序号，最大 index 会到约 {need}，"
            f"越过 --index-end {reserved_end}；请把 --index-end 放宽到 {need} 以上")
    jobs = []
    all_indices = []
    layers = [(name, target, y1, y23) for (name, y1, y23), target
              in zip(LAYER_SPEC, layer_targets)]
    layers.sort(key=lambda entry: EMIT_ORDER.index(entry[0]))
    for layer, target, y1, y23 in layers:
        for shape in range(shape_count):
            pools = [[p for p in plans if p["shape"] == shape and p["uma"] == uma
                      and p["plan"] not in held_ids] for uma in recipe["space"]["umas"]]
            pools = [sorted(pool, key=lambda p: p["fields"]) for pool in pools if pool]
            # 同一马娘内等距走遍卡组，马娘间轮转；跨年份轮换起点。
            counts = [0] * len(pools)
            indices = []
            for attempt in range(target * 2):
                cell = attempt % len(pools)
                pool = pools[cell]
                offset = counts[cell]
                # 以互素步长遍历，避免排序前端始终是同几张卡。
                step = len(pool) // 2 + 1
                while gcd(step, len(pool)) != 1:
                    step += 1
                plan = pool[(offset * step + shape * 17 + len(jobs) * 31) % len(pool)]
                counts[cell] += 1
                indices.append(cycle * len(plans) + plan["plan"])
                cycle += 1
            name = f"{layer}_s{shape + 1}"
            jobs.append(dict(name=name, layer=layer, shape=shape, target=target,
                             indices=f"indices/{name}.json", count=len(indices),
                             quota_y1=y1, quota_y2_y3=y23))
            all_indices.extend(indices)
            write_json(output / "indices" / f"{name}.json", indices)
    if len(set(all_indices)) != len(all_indices) or max(all_indices) >= reserved_end:
        raise ValueError("index 重复或越过独占号段")
    if any(i % len(plans) in held_ids for i in all_indices):
        raise ValueError("正式清单含留出组合")
    write_json(output / "plans.json", plans)
    if rule == "every_tenth_all":
        holdout_text = dict(method="uma×shape 内完整字段排序，每十个取第十个；空间内全部计划参与",
                            evidence=recipe["holdout_note"],
                            boundary="留出组合只作闭环验收，不得混进本轮训练")
    else:
        holdout_text = dict(
            method="uma×shape 内完整字段排序，每十个取第十个；仅已确认未见的两张新速卡或 114101",
            evidence="nn_model_registry.md 的 R4 七马娘记录及 logs/newdeck_pair_0913/deck_coverage.txt；后者已报告两速卡在八组 R4 数据均为零",
            boundary="这是已确认未见子集的留出，不声称覆盖所有历史未见组合；不得混进本轮训练")
    write_json(output / "holdout.json", dict(**holdout_text, plans=held))
    target_valid = sum(job["target"] for job in jobs)
    write_json(output / "manifest.json", dict(
        recipe_id=recipe_id, status="frozen", space=recipe["space"],
        model=model, model_id=model_id,
        rollin="nn", search_n=search_n, shard_size=32, inherit=recipe["inherit"],
        epsilon=0.15, seed_base=88241484357425, use_ucb=False, radical_factor_max=1.4,
        target_valid=target_valid, seconds=seconds, jobs=jobs,
        index_reservation=[reserved_start, reserved_end], holdout_count=len(held),
        spare_policy="每层构成双倍候选清单；达到有效目标即停，备用耗尽则失败，不改配方"))
    print(f"计划={len(plans)} 留出={len(held)} 有效目标={target_valid} 清单及备用={len(all_indices)}")
    print(f"号段={reserved_start}..{reserved_end} 实际最大 index={max(all_indices)} "
          f"search_n={search_n} model_id={model_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, default=ROOT / "scripts/collect/gen2_v1_recipe.json",
                        help="空间身份来源；默认 gen2_v1，2速1耐2智 补采用 gen2_2s1e2w_recipe.json")
    parser.add_argument("--recipe-id", required=True, help="写进 manifest 的配方名，例如 gen2_v1_formal1024_0918")
    parser.add_argument("--model", required=True, help="roll-in 模型路径（相对工作区根）")
    parser.add_argument("--model-id", required=True, help="roll-in 模型显式版本名，进 manifest 与 rollin 身份")
    parser.add_argument("--search-n", type=int, required=True, help="每候选 rollout 数")
    parser.add_argument("--layer-targets", required=True,
                        help="四层**每个构成**的有效根目标，逗号分隔，顺序固定为 "
                             "general,region_y1,region_y2,region_y3（0914 那轮是 5000,100,300,300）")
    parser.add_argument("--index-start", type=int, required=True)
    parser.add_argument("--index-end", type=int, required=True)
    parser.add_argument("--seconds", type=int, required=True, help="采集总截止秒数（0914 是 43200 = 12h）")
    args = parser.parse_args()
    targets = [int(v) for v in args.layer_targets.split(",")]
    if len(targets) != len(LAYER_SPEC) or any(t <= 0 for t in targets):
        raise ValueError(f"--layer-targets 需要 {len(LAYER_SPEC)} 个正整数，顺序 "
                         f"{','.join(name for name, _, _ in LAYER_SPEC)}")
    if args.search_n <= 0 or args.index_start >= args.index_end or args.seconds <= 0:
        raise ValueError("--search-n / --seconds 必须为正，且 --index-start < --index-end")
    prepare(args.rust_dump, args.output, recipe_path=args.recipe, recipe_id=args.recipe_id, model=args.model,
            model_id=args.model_id, search_n=args.search_n, layer_targets=targets,
            index_start=args.index_start, index_end=args.index_end, seconds=args.seconds)
