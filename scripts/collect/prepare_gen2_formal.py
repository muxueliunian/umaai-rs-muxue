"""从 Rust 计划原文生成无哈希的正式清单；只准备文件，不启动采集。"""

import argparse
import itertools
import json
from math import gcd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


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


def prepare(dump_path, output):
    """冻结计划、留出组合、分层配额及独立世界；只允许写全新目录。"""
    if output.exists():
        raise FileExistsError(output)
    recipe = json.loads((ROOT / "scripts/collect/gen2_v1_recipe.json").read_text(encoding="utf-8"))
    plans = enumerate_plans(recipe)
    dumped = []
    for line in dump_path.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("GEN2PLAN "):
            _, index, uma, deck, shape = line.split(maxsplit=4)
            dumped.append((int(index), int(uma), [int(v) for v in deck.split(",")], shape))
    expected = [(p["plan"], p["uma"], p["deck"], recipe["space"]["shapes"][p["shape"]]["name"]) for p in plans]
    if dumped != expected or len(plans) != 4288:
        raise ValueError("独立枚举与 Rust 计划字段/顺序不一致")

    # 只从已有证据明确未见的两张新速卡/新马娘中选泛化留出，其他新组合不猜历史覆盖。
    held = []
    for uma in recipe["space"]["umas"]:
        for shape in range(4):
            unseen = sorted([p for p in plans if p["uma"] == uma and p["shape"] == shape
                             and (uma == 114101 or any(c in p["deck"] for c in (303124, 303114)))],
                            key=lambda p: p["fields"])
            held.extend(unseen[9::10])
    held_ids = {p["plan"] for p in held}

    # 全区间独占；实际 index 不连续，index % 4288 严格定位原空间计划。
    reserved_start, reserved_end = 20_000_000, 300_000_000
    for path in (ROOT / "training_data").rglob("manifest.json"):
        mf = json.loads(path.read_text(encoding="utf-8-sig"))
        indices = mf.get("work_indices")
        if indices:
            if any(reserved_start <= i < reserved_end for i in indices):
                raise ValueError(f"号段已占用：{path}")
        elif int(mf.get("index_start", 0)) < reserved_end and int(mf.get("index_end", 0)) > reserved_start:
            raise ValueError(f"号段与既有 manifest 相交：{path}")
    cycle = (reserved_start + len(plans) - 1) // len(plans)
    jobs = []
    all_indices = []
    layers = [("general", 5000, 0, "0,0"), ("region_y1", 100, 1000, "0,0"),
              ("region_y2", 300, 0, "1000,0"), ("region_y3", 300, 0, "0,1000")]
    for layer, target, y1, y23 in layers:
        for shape in range(4):
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
    write_json(output / "holdout.json", dict(
        method="uma×shape 内完整字段排序，每十个取第十个；仅已确认未见的两张新速卡或 114101",
        evidence="nn_model_registry.md 的 R4 七马娘记录及 logs/newdeck_pair_0913/deck_coverage.txt；后者已报告两速卡在八组 R4 数据均为零",
        boundary="这是已确认未见子集的留出，不声称覆盖所有历史未见组合；不得混进本轮训练",
        plans=held))
    write_json(output / "manifest.json", dict(
        recipe_id="gen2_v1_formal2048_0914", status="frozen", space=recipe["space"],
        model="saved_models/arms/ens_R4_g123.onnx", model_id="ens_R4_g123_30k",
        rollin="nn", search_n=2048, shard_size=32, inherit=recipe["inherit"],
        epsilon=0.15, seed_base=88241484357425, use_ucb=False, radical_factor_max=1.4,
        target_valid=22800, seconds=43200, jobs=jobs,
        index_reservation=[reserved_start, reserved_end], holdout_count=len(held),
        spare_policy="每层构成双倍候选清单；达到有效目标即停，备用耗尽则失败，不改配方"))
    print(f"计划={len(plans)} 留出={len(held)} 有效目标=22800 清单及备用={len(all_indices)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.rust_dump, args.output)
