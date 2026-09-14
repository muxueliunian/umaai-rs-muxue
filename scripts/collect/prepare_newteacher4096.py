"""生成新教师清单；旧计划与留出原文不改，运行前仍须核对云端号段登记。"""

import argparse
import copy
import json
from math import gcd
from pathlib import Path

from prepare_gen2_formal import enumerate_plans, write_json
from run_formal_collect import check_plan, read_json

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "scripts/collect/formal2048_0914"


def check_history(roots, start, end):
    """检查历史 manifest 的实际序号或区间；已登记未用号段还须人工核对。"""
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"历史目录不存在：{root}")
        for path in root.rglob("manifest.json"):
            old = read_json(path)
            values = old.get("work_indices")
            if values is not None:
                collision = any(start <= i < end for i in values)
            else:
                bounds = old.get("index_reservation")
                if bounds is None:
                    bounds = [old.get("index_start", 0), old.get("index_end", 0)]
                collision = bounds[0] < end and bounds[1] > start
            if collision:
                raise ValueError(f"号段与历史数据/清单相交：{path}")


def prepare(output, start, end, history_roots):
    """只生成新目录；配额为40000通用+10000地区，保留逐根跨候选CRN。"""
    if output.exists():
        raise FileExistsError(output)
    if not 300000000 <= start < end < 2 ** 63:
        raise ValueError("新正式号段须位于旧预留区间之后，且保持int64安全范围")
    check_history(history_roots, start, end)
    old = read_json(OLD / "manifest.json")
    plans = read_json(OLD / "plans.json")
    holdout = read_json(OLD / "holdout.json")
    if enumerate_plans(read_json(ROOT / "scripts/collect/gen2_v1_recipe.json")) != plans:
        raise ValueError("独立枚举与旧冻结计划字段/顺序不符")
    held = {p["plan"] for p in holdout["plans"]}
    cycle = (start + len(plans) - 1) // len(plans)
    jobs, arrays = [], {}
    layers = [("general", 0, "0,0"), ("region_y1", 1000, "0,0"),
              ("region_y2", 0, "1000,0"), ("region_y3", 0, "0,1000")]
    for layer, y1, y23 in layers:
        for shape in range(4):
            target = {"general": 10000, "region_y1": 357,
                      "region_y2": 1072 if shape < 2 else 1071,
                      "region_y3": 1071 if shape < 2 else 1072}[layer]
            pools = [sorted([p for p in plans if p["uma"] == uma and p["shape"] == shape
                             and p["plan"] not in held], key=lambda p: p["fields"])
                     for uma in old["space"]["umas"]]
            pools = [p for p in pools if p]
            counts, indices = [0] * len(pools), []
            for attempt in range(target * 2):
                cell = attempt % len(pools)
                pool = pools[cell]
                step = len(pool) // 2 + 1
                while gcd(step, len(pool)) != 1:
                    step += 1
                p = pool[(counts[cell] * step + shape * 17 + len(jobs) * 31) % len(pool)]
                counts[cell] += 1
                indices.append(cycle * len(plans) + p["plan"])
                cycle += 1
            name = f"{layer}_s{shape + 1}"
            arrays[f"indices/{name}.json"] = indices
            jobs.append(dict(name=name, layer=layer, shape=shape, target=target,
                             indices=f"indices/{name}.json", count=len(indices),
                             quota_y1=y1, quota_y2_y3=y23))
    values = [i for a in arrays.values() for i in a]
    if max(values) >= end or len(set(values)) != 100000:
        raise ValueError("号段不足或序号重复")
    plan = copy.deepcopy(old)
    plan.update(recipe_id="gen2_newteacher4096_0914", collection_profile="newteacher4096",
                status="prepared_pending_cloud_reservation_check",
                upstream_commit="8a23566b5ded4d782d3ec0483c0b2f62462abc68",
                teacher_config={"ramen_pt_sacrifice_score": 0, "pt_favor_rate": 1.0},
                search_n=4096, target_valid=50000, seconds=None, jobs=jobs,
                index_reservation=[start, end])
    # 所有计算与冲突检查先完成，再写新目录；失败目录保留，禁止覆盖后盲跑。
    for relative, data in arrays.items():
        write_json(output / relative, data)
    for name in ("plans.json", "holdout.json"):
        (output / name).write_bytes((OLD / name).read_bytes())
    write_json(output / "manifest.json", plan)
    check_plan(output, plan)
    print(f"prepared: accepted=50000 attempts=100000 index=[{min(values)},{max(values)}]")
    print("尚须云端历史/预留号段核对；未启动采集。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--history-root", type=Path, action="append", required=True)
    args = parser.parse_args()
    prepare(args.output.resolve(), args.start, args.end, args.history_root)
