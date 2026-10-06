"""分轮采集验收：逐字段核对 raw 导出与冻结清单，报告各轮/层/构成的实际根数。

只读采集产物，不改任何文件（报告写到 --report，默认输出目录下 acceptance.json）。
不计算哈希：样本身份用真实 index 与完整组合字段比较，列宽和缺行用实际数组判断。
任何一项不符都列入 `errors` 并以退出码 1 结束，不静默丢样本。
"""

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np

from run_formal_collect import ROOT, read_json, write_json


def check_export(export, plan, job, work, plans, banned, commit, errors, seen):
    """核对一份 raw 导出；返回实际有效根数与按阶段计数。"""
    tag = job["name"]
    meta = read_json(export / "meta.json")
    if meta.get("git_commit") != commit:
        errors.append(f"{tag}: 导出提交 {meta.get('git_commit')} ≠ 采集提交 {commit}")
    if meta["search_n"] != plan["search_n"] or meta["stats"]["rollout_width"] != plan["search_n"]:
        errors.append(f"{tag}: search_n / 列宽不是 {plan['search_n']}")
    if meta["rollin"] != "nn:asset:" + plan["model_id"] or not meta.get("raw"):
        errors.append(f"{tag}: roll-in 身份或 raw 标记不符")
    index = np.load(export / "index.npy")
    ptr = np.load(export / "cand_ptr.npy")
    n = np.load(export / "cand_n.npy")
    scores = np.load(export / "cand_scores.npy", mmap_mode="r")
    valid = np.load(export / "cand_valid.npy", mmap_mode="r")
    fields = np.load(export / "combo_fields.npy")
    rows = len(index)
    if rows != meta["stats"]["samples"] or len(fields) != rows or len(ptr) != rows + 1:
        errors.append(f"{tag}: 根数组长度不一致（index {rows} / meta {meta['stats']['samples']}）")
    if ptr[0] != 0 or ptr[-1] != len(n) or np.any(np.diff(ptr) <= 0):
        errors.append(f"{tag}: 候选指针缺行或非递增")
    if scores.shape != (len(n), plan["search_n"]) or valid.shape != scores.shape:
        errors.append(f"{tag}: 原始分数形状 {scores.shape} 不是 [候选, {plan['search_n']}]")
    if np.any(n != plan["search_n"]) or not np.all(valid):
        errors.append(f"{tag}: 存在 rollout 数不足或无效列的候选")
    if not np.all(np.isfinite(scores)):
        errors.append(f"{tag}: 原始分数含非有限值")
    work_set = set(work)
    for i, f in zip(index.tolist(), fields.tolist()):
        if i in seen:
            errors.append(f"{tag}: index {i} 重复（另见 {seen[i]}）")
        seen[i] = tag
        if i not in work_set:
            errors.append(f"{tag}: index {i} 不在该任务的冻结清单里")
        expect = plans[i % len(plans)]
        if f != expect["fields"] or expect["shape"] != job["shape"]:
            errors.append(f"{tag}: index {i} 的组合字段或构成与计划原文不符")
        if expect["plan"] in banned:
            errors.append(f"{tag}: index {i} 落在留出/排除组合 {expect['plan']}")
    return rows, Counter(np.load(export / "stage.npy").tolist())


def main():
    """读 run_state 汇总全部已完成任务与截断任务，写验收报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="采集输出目录（含 run_state.json）")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    plan_dir, output = args.plan.resolve(), args.output.resolve()
    plan = read_json(plan_dir / "manifest.json")
    plans = read_json(plan_dir / "plans.json")
    banned = {p["plan"] for p in read_json(plan_dir / "holdout.json")["plans"]}
    if (plan_dir / "exclusions.json").exists():
        banned |= {p["plan"] for p in read_json(plan_dir / "exclusions.json")["plans"]}
    state = read_json(output / "run_state.json")
    commit = state["identity"]["commit"]
    errors, seen, rows = [], {}, []
    jobs = {j["name"]: j for j in plan["jobs"]}
    if state["identity"]["plan"] != plan:
        errors.append("run_state 记录的清单与 --plan 不一致")
    exports = [(name, ROOT / path, False) for name, path in state.get("exports", {}).items()]
    partial = (state.get("truncated") or {}).get("partial")
    if partial and partial.get("export"):
        exports.append((partial["job"], ROOT / partial["export"], True))
    for name in state["completed"]:
        if name not in state.get("exports", {}):
            errors.append(f"{name}: 已完成但没有导出记录")
    stages = Counter()
    for name, export, is_partial in exports:
        job = jobs[name]
        work = state["identity"]["indices"][name]
        count, hist = check_export(export, plan, job, work, plans, banned, commit, errors, seen)
        want = partial["accepted"] if is_partial else job["target"]
        if count != want:
            errors.append(f"{name}: 实际 {count} 根 ≠ 应有 {want}")
        data_mf = read_json(output / "data" / name / "manifest.json")
        if data_mf["accepted"] != count or data_mf["search_n"] != plan["search_n"]:
            errors.append(f"{name}: 采集 manifest 与导出根数/search_n 不一致")
        stages.update(hist)
        rows.append(dict(job=name, round=job.get("round"), layer=job["layer"], shape=job["shape"],
                         samples=count, partial=is_partial, export=str(export.relative_to(ROOT))))
    by = lambda key: {str(k): sum(r["samples"] for r in rows if r[key] == k) for k in sorted({r[key] for r in rows})}
    done = set(state["completed"])
    full_rounds = sorted({j["round"] for j in plan["jobs"] if "round" in j
                          and all(o["name"] in done for o in plan["jobs"] if o.get("round") == j["round"])})
    report = dict(plan=plan["recipe_id"], commit=commit, search_n=plan["search_n"],
                  total_samples=sum(r["samples"] for r in rows), unique_indices=len(seen),
                  full_rounds=len(full_rounds), completed_jobs=len(done), partial=partial,
                  by_layer=by("layer"), by_shape=by("shape"), by_round=by("round"),
                  stage_hist={str(k): v for k, v in sorted(stages.items())},
                  truncated=state.get("truncated"), finished=bool(state.get("finished")),
                  errors=errors, jobs=rows)
    path = args.report or output / "acceptance.json"
    write_json(path, report)
    print(f"验收：{report['total_samples']} 根（{len(seen)} 个不同 index），完整 {len(full_rounds)} 轮，"
          f"已完成任务 {len(done)} 个，错误 {len(errors)} 条 → {path}")
    for e in errors[:20]:
        print("  ✗", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
