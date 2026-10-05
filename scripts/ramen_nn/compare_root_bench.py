"""`ramen_root_bench` 输出的逐字段比较工具（改动前后对拍用）。

# 为什么需要这个工具

改 `RamenNnTrainer` 的决策路径时，「分数看起来一样」不足以说明行为没变。
本工具直接比较实际字段，**不做任何哈希**（哈希只能回答「有没有变」，
答不出「哪个字段、变了多少」，且撞上浮点精度时无法判读）。

与 `compare_bench.py` 的分工：那个比的是**闭环 bench 的整局结果**（按随机世界配对、
管选择集/验收集分离）；本工具比的是**单个固定根内部**的逐 rollout 与逐决策记录。

# 三个子命令

- `raw`：按 `(candidate, j)` 对齐两份 `--raw-csv`，比较 `seed` / `score` / `score_pt`。
  同时报缺失、多出与重复键。
- `decisions`：比较两份 `--decision-csv`。
  ❗**不能按 `seq` 连接**——一旦某一臂少发了请求，`seq` 会整体重新编号。
  故以 `(candidate, j)` 为一条 rollout，组内按 `seq` 排序后**按顺序**逐条比较
  `turn` / `stage` / `n_actions` / `actions` / `chosen` / `features`。
  `--drop-single-candidate` 会先从**参考侧**剔除 `n_actions == 1` 的记录，
  用于「单候选免推理」这类只减少请求、不改变其余决策的改动。
- `cache-eligibility`：统计跨阶段缓存的**资格**比例，即有多少 `SpecialSelect` 请求
  在**同一条 rollout** 内确实存在可复用的前一拍。
  ❗资格判据是「前一条网络请求存在、阶段为 `RamenSelect`、回合相同、输入特征逐项相同」，
  **不能只数剩余 `SpecialSelect` 数量**：前一拍可能已被单候选优化消掉，
  或该 rollout 从中间阶段起步、根本没有可复用输出。

# 用法

```text
python scripts/ramen_nn/compare_root_bench.py raw  base_raw.csv opt_raw.csv
python scripts/ramen_nn/compare_root_bench.py decisions base_dec.csv opt_dec.csv \
    --drop-single-candidate
python scripts/ramen_nn/compare_root_bench.py cache-eligibility opt_dec.csv \
    --baseline base_dec.csv
```

退出码 0 表示一致（`cache-eligibility` 恒为 0，它只统计不判定）。
"""

from __future__ import annotations

import argparse
import csv
import io
from collections import defaultdict
from pathlib import Path

#: 逐决策比较要逐字段核对的列。`features` 按导出的十进制文本逐项比，不用哈希。
DECISION_FIELDS = ["turn", "stage", "n_actions", "actions", "chosen", "features"]


def load_raw(path: Path) -> tuple[dict[tuple[int, int], dict[str, str]], list[tuple[int, int]]]:
    """读一份 raw CSV，返回按 `(candidate, j)` 索引的行与重复键列表。"""
    rows: dict[tuple[int, int], dict[str, str]] = {}
    dups: list[tuple[int, int]] = []
    with io.open(path, encoding="utf-8", newline="") as f:
        for rec in csv.DictReader(f):
            key = (int(rec["candidate"]), int(rec["j"]))
            if key in rows:
                dups.append(key)
            rows[key] = rec
    return rows, dups


def load_decisions(path: Path) -> dict[tuple[int, int], list[dict[str, str]]]:
    """读一份 decision CSV，按 `(candidate, j)` 分组、组内按 `seq` 升序。"""
    groups: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    with io.open(path, encoding="utf-8", newline="") as f:
        for rec in csv.DictReader(f):
            groups[(int(rec["candidate"]), int(rec["j"]))].append(rec)
    for key in groups:
        groups[key].sort(key=lambda r: int(r["seq"]))
    return groups


def cmd_raw(args: argparse.Namespace) -> int:
    """按键对齐比较逐 rollout 终局评分。"""
    ref, ref_dups = load_raw(args.reference)
    new, new_dups = load_raw(args.new)
    print(f"参考   {args.reference}: {len(ref)} 行，重复键 {len(ref_dups)}")
    print(f"新产出 {args.new}: {len(new)} 行，重复键 {len(new_dups)}")
    missing = sorted(set(ref) - set(new))
    extra = sorted(set(new) - set(ref))
    print(f"参考有而新产出缺: {len(missing)}  新产出多出: {len(extra)}")

    ok = not (ref_dups or new_dups or missing or extra)
    for field in ["seed", "score", "score_pt"]:
        diff = 0
        max_abs = 0.0
        for key in sorted(set(ref) & set(new)):
            a, b = ref[key][field], new[key][field]
            if a == b:
                continue
            diff += 1
            if field != "seed":
                max_abs = max(max_abs, abs(float(a) - float(b)))
        if field == "seed":
            print(f"seed     : 不同 {diff} 条")
        else:
            print(f"{field:<9}: 不同 {diff} 条，最大绝对差 {max_abs:.3e}")
        ok = ok and diff == 0
    print("结论: 逐字段完全一致" if ok else "结论: ❗存在差异")
    return 0 if ok else 1


def diff_decision_groups(
    ref: dict[tuple[int, int], list[dict[str, str]]],
    new: dict[tuple[int, int], list[dict[str, str]]],
    drop_single_candidate: bool
) -> tuple[int, dict[str, int], list[str]]:
    """按 rollout 顺序比较两侧决策记录，返回 `(条数不一致的组数, 各字段差异计数, 明细样本)`。

    ❗必须遍历两侧键的**并集**、缺失组按空列表处理，**先过滤参考侧再比长度**。
    若反过来（先按键集合判缺失、后剔除参考侧单候选），会把「整条 rollout 原本全是
    单候选、优化后自然一条记录都不剩」这种**合法**情形误报成缺失。

    判定全部交给长度与字段比较：
    - 参考侧过滤后为空、新产出也缺该组 → 长度 0 == 0，通过；
    - 参考侧过滤后仍有多候选记录而新产出缺该组 → 长度不等，失败。
    """
    diffs: dict[str, int] = defaultdict(int)
    len_mismatch = 0
    samples: list[str] = []
    for key in sorted(set(ref) | set(new)):
        kept = ref.get(key, [])
        if drop_single_candidate:
            kept = [r for r in kept if int(r["n_actions"]) > 1]
        got = new.get(key, [])
        if len(kept) != len(got):
            len_mismatch += 1
            if len(samples) < 5:
                samples.append(f"  rollout {key}: 参考 {len(kept)} 条 vs 新产出 {len(got)} 条")
            continue
        for i, (a, b) in enumerate(zip(kept, got)):
            for field in DECISION_FIELDS:
                if a[field] == b[field]:
                    continue
                diffs[field] += 1
                if len(samples) < 5:
                    va, vb = a[field], b[field]
                    if field == "features":
                        va, vb = va[:60] + "...", vb[:60] + "..."
                    samples.append(f"  rollout {key} 第 {i} 条 {field}: 参考={va} 新产出={vb}")
    return len_mismatch, diffs, samples


def cmd_decisions(args: argparse.Namespace) -> int:
    """按 rollout 顺序比较逐决策记录（不按 `seq` 连接）。"""
    ref, new = load_decisions(args.reference), load_decisions(args.new)
    ref_total = sum(len(v) for v in ref.values())
    new_total = sum(len(v) for v in new.values())
    ref_single = sum(1 for v in ref.values() for r in v if int(r["n_actions"]) == 1)
    new_single = sum(1 for v in new.values() for r in v if int(r["n_actions"]) == 1)
    print(f"参考   {ref_total} 条（其中单候选 {ref_single}）")
    print(f"新产出 {new_total} 条（其中单候选 {new_single}）")
    if args.drop_single_candidate and ref_total:
        print(f"参考剔除单候选后 {ref_total - ref_single}，"
              f"与新产出差 {ref_total - ref_single - new_total}；"
              f"单候选占参考请求 {100.0 * ref_single / ref_total:.2f}%")

    only_ref = sorted(set(ref) - set(new))
    only_new = sorted(set(new) - set(ref))
    for label, lst in (("只在参考", only_ref), ("只在新产出", only_new)):
        if lst:
            print(f"（提示）{label}出现的 rollout {len(lst)} 条，前 5: {lst[:5]}")

    len_mismatch, diffs, samples = diff_decision_groups(ref, new, args.drop_single_candidate)

    print()
    if len_mismatch:
        print(f"❗{len_mismatch} 条 rollout 条数不一致")
    if diffs:
        print("❗字段差异计数：" + ", ".join(f"{k}={v}" for k, v in sorted(diffs.items())))
    for s in samples:
        print(s)
    ok = not (len_mismatch or diffs)
    print("结论: 剩余决策逐条一致" if ok else "结论: ❗存在差异")
    return 0 if ok else 1


def cmd_cache_eligibility(args: argparse.Namespace) -> int:
    """统计 `SpecialSelect` 请求里真正可复用前一拍输出的比例。"""
    groups = load_decisions(args.target)
    total = sum(len(v) for v in groups.values())
    base_total = None
    if args.baseline is not None:
        base_total = sum(len(v) for v in load_decisions(args.baseline).values())

    special = eligible = 0
    no_prev = prev_not_ramen = turn_diff = feat_diff = 0
    for rows in groups.values():
        for i, r in enumerate(rows):
            if r["stage"] != "SpecialSelect":
                continue
            special += 1
            if i == 0:
                no_prev += 1
                continue
            prev = rows[i - 1]
            if prev["stage"] != "RamenSelect":
                prev_not_ramen += 1
            elif prev["turn"] != r["turn"]:
                turn_diff += 1
            elif prev["features"] != r["features"]:
                feat_diff += 1
            else:
                eligible += 1

    print(f"请求总数            {total}")
    if total:
        print(f"其中 SpecialSelect  {special}（占 {100.0 * special / total:.2f}%）")
    print(f"可复用（前一拍在、同回合、特征逐项相同） {eligible}")
    print(f"  不可复用原因：无前一条 {no_prev} / 前一条非 RamenSelect {prev_not_ramen}"
          f" / 回合不同 {turn_diff} / 特征不同 {feat_diff}")
    if total:
        print(f"占本文件请求   {100.0 * eligible / total:.2f}%")
    if base_total:
        print(f"占基线全部请求 {100.0 * eligible / base_total:.2f}%（基线 {base_total} 条）")
    return 0


def main() -> int:
    """解析子命令并分派。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_raw = sub.add_parser("raw", help="比较两份 --raw-csv")
    p_raw.add_argument("reference", type=Path)
    p_raw.add_argument("new", type=Path)
    p_raw.set_defaults(func=cmd_raw)

    p_dec = sub.add_parser("decisions", help="比较两份 --decision-csv")
    p_dec.add_argument("reference", type=Path)
    p_dec.add_argument("new", type=Path)
    p_dec.add_argument("--drop-single-candidate", action="store_true",
                       help="先从参考侧剔除 n_actions == 1 的记录")
    p_dec.set_defaults(func=cmd_decisions)

    p_cache = sub.add_parser("cache-eligibility", help="统计跨阶段缓存的资格比例")
    p_cache.add_argument("target", type=Path)
    p_cache.add_argument("--baseline", type=Path, default=None,
                         help="提供「占基线全部请求」这个分母")
    p_cache.set_defaults(func=cmd_cache_eligibility)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
