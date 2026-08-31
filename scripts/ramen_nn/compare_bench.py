"""闭环 bench 结果的配对比较，并守住「选择集 / 验收集分离」这条纪律。

# 为什么需要这个工具

闭环 CSV 的每一行是一个「随机世界」的对局结果，世界由 `(计划, 局种子)` 唯一确定。
直接把多个 CSV 拼起来算均分与标准误有两个坑，都在 2026-08-31 实际踩过：

1. **重复世界虚增样本量。** 曾用三个相邻基种子（61444/61445/61446）各跑 8 局当作
   12600 局独立样本，实际只有 5248 个唯一世界（3152 个重复 3 次），重复局分数
   逐位相同。naive 标准误因此被低估约 1.5 倍。本工具**强制按世界去重**并报告
   重复情况。根因与正确用法见 `ramen_space_bench.rs` 中 `--seed` / `--run-offset`
   的文档。
2. **在验收集上挑 checkpoint。** 若用同一批对局既选模型又报成绩，报出的分数带
   winner's curse。本工具的 `--selection` 与 `--acceptance` 分开传入，并断言两者
   世界零重叠，从而保证验收对**已选定**的 checkpoint 是无偏的。

# 用法

```text
python scripts/ramen_nn/compare_bench.py \
    --arm 基线=target/sel_ctrl.csv --arm 变体=target/sel_fact.csv \
    --baseline 基线
```

带选择/验收分离时：

```text
python scripts/ramen_nn/compare_bench.py \
    --selection 种子A=target/sel_a.csv --selection 种子B=target/sel_b.csv \
    --acceptance 种子A=target/acc_a.csv --acceptance 种子B=target/acc_b.csv \
    --baseline-csv target/acc_hw.csv
```

此时脚本先在选择集上排名并选出冠军，再**只**报告冠军在验收集上的成绩。
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics as st
import sys
from collections import Counter
from pathlib import Path

# 世界标识：CSV 的 build 列（计划）+ seed 列（该局规则主种子）
WORLD_KEY = ("build", "seed")


def load_worlds(paths: list[Path]) -> tuple[dict[tuple[str, str], float], Counter]:
    """读入若干 CSV，按世界去重，返回 (世界→分数, 重复次数分布)。

    同一世界在不同 CSV 里出现多次时保留第一次；若分数不一致会直接报错，
    因为那意味着这些 CSV 并非同一个策略跑出来的，拼在一起没有意义。

    # 错误

    文件缺列、同一世界分数冲突时报错。
    """

    scores: dict[tuple[str, str], float] = {}
    seen: Counter = Counter()
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                missing = [c for c in (*WORLD_KEY, "score") if c not in row]
                if missing:
                    raise ValueError(f"{path} 缺少列 {missing}")
                key = (row["build"], row["seed"])
                value = float(row["score"])
                seen[key] += 1
                if key in scores:
                    if scores[key] != value:
                        raise ValueError(
                            f"世界 {key} 在 {path} 中分数 {value} 与先前的 {scores[key]} 冲突，"
                            "这些 CSV 不是同一策略的结果"
                        )
                else:
                    scores[key] = value
    return scores, seen


def report_duplicates(name: str, seen: Counter, unique: int) -> None:
    """打印重复世界情况；有重复时明确说明去重后的有效样本量。"""

    total = sum(seen.values())
    if total == unique:
        print(f"  {name:<16} {unique} 个世界，无重复")
        return
    dist = Counter(seen.values())
    detail = "，".join(f"{count} 个出现 {times} 次" for times, count in sorted(dist.items()))
    print(f"  {name:<16} {total} 行 → {unique} 个唯一世界（{detail}）")
    print(f"  {'':<16} ❗有效样本量是 {unique} 而非 {total}，标准误已按去重后计算")


def paired(a: dict, b: dict) -> tuple[int, float, float, float]:
    """配对差统计，返回 (配对数, 均值, 标准误, 胜率)。

    # 错误

    无共同世界时报错——两组跑的不是同一批世界，配对无从谈起。
    """

    keys = sorted(set(a) & set(b))
    if not keys:
        raise ValueError("两组没有共同世界，无法配对")
    diffs = [a[k] - b[k] for k in keys]
    mean = st.mean(diffs)
    stderr = st.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else float("nan")
    win = sum(1 for d in diffs if d > 0) / len(diffs)
    return len(keys), mean, stderr, win


def parse_arm(spec: str) -> tuple[str, list[Path]]:
    """解析 ``名字=路径[,路径...]``。

    # 错误

    缺少 ``=``、或任一路径不存在时报错。
    """

    if "=" not in spec:
        raise ValueError(f"格式应为 名字=路径[,路径...]，实得 {spec!r}")
    name, _, joined = spec.partition("=")
    paths = [Path(p) for p in joined.split(",") if p]
    if not paths:
        raise ValueError(f"{name} 没有给出任何路径")
    for path in paths:
        if not path.exists():
            raise ValueError(f"{path} 不存在")
    return name.strip(), paths


def summarise(title: str, arms: dict[str, dict], baseline: str | None) -> None:
    """打印各 arm 均分，并在给定基准时打印配对差。"""

    print(f"\n{title}")
    print(f"  {'arm':<16} {'世界数':>8} {'均分':>10}")
    for name, scores in arms.items():
        print(f"  {name:<16} {len(scores):>8} {st.mean(list(scores.values())):>10.0f}")
    if baseline is None or baseline not in arms:
        return
    print(f"\n  配对差（vs {baseline}）")
    for name, scores in arms.items():
        if name == baseline:
            continue
        count, mean, stderr, win = paired(scores, arms[baseline])
        t_value = mean / stderr if stderr and not math.isnan(stderr) else float("nan")
        half = 1.96 * stderr
        print(
            f"  {name:<16} n={count:<6} {mean:+9.1f} ± {stderr:5.1f}"
            f"  t={t_value:6.2f}  95%CI [{mean - half:+.0f}, {mean + half:+.0f}]"
            f"  胜率 {100 * win:.1f}%"
        )


def main() -> int:
    """比较入口。

    # 错误

    参数不合法、选择集与验收集世界重叠时返回非零并打印原因。
    """

    parser = argparse.ArgumentParser(description="闭环 bench 配对比较（按世界去重）")
    parser.add_argument("--arm", action="append", default=[], help="名字=CSV[,CSV...]；简单比较模式")
    parser.add_argument("--selection", action="append", default=[], help="名字=CSV[,CSV...]；选择集")
    parser.add_argument("--acceptance", action="append", default=[], help="名字=CSV[,CSV...]；验收集")
    parser.add_argument("--baseline", help="作为配对基准的 arm 名")
    parser.add_argument("--baseline-csv", help="额外的基准 CSV（例如手写策略），并入两个集合比较")
    args = parser.parse_args()

    if args.arm and (args.selection or args.acceptance):
        print("--arm 与 --selection/--acceptance 不能混用", file=sys.stderr)
        return 2
    if not args.arm and not (args.selection and args.acceptance):
        print("需要 --arm，或同时给出 --selection 与 --acceptance", file=sys.stderr)
        return 2

    def build(specs: list[str], label: str) -> dict[str, dict]:
        arms: dict[str, dict] = {}
        print(f"\n{label}")
        for spec in specs:
            name, paths = parse_arm(spec)
            scores, seen = load_worlds(paths)
            report_duplicates(name, seen, len(scores))
            arms[name] = scores
        return arms

    if args.arm:
        arms = build(args.arm, "读入")
        if args.baseline_csv:
            scores, seen = load_worlds([Path(args.baseline_csv)])
            report_duplicates("基准CSV", seen, len(scores))
            arms["基准"] = scores
        summarise("结果", arms, args.baseline or ("基准" if args.baseline_csv else None))
        return 0

    selection = build(args.selection, "读入选择集")
    acceptance = build(args.acceptance, "读入验收集")

    # 分离纪律：两个集合的世界必须零重叠，否则验收带 winner's curse
    sel_worlds: set = set()
    for scores in selection.values():
        sel_worlds |= set(scores)
    acc_worlds: set = set()
    for scores in acceptance.values():
        acc_worlds |= set(scores)
    overlap = sel_worlds & acc_worlds
    if overlap:
        print(
            f"\n❌ 选择集与验收集有 {len(overlap)} 个共同世界，验收不再无偏。"
            "\n   用 ramen_space_bench 的 --run-offset 切出不重叠的局号区间后重跑。",
            file=sys.stderr,
        )
        return 1
    print(f"\n✅ 选择集 {len(sel_worlds)} 个世界与验收集 {len(acc_worlds)} 个世界零重叠")

    missing = sorted(set(selection) - set(acceptance))
    if missing:
        print(f"\n❌ 这些 arm 只有选择集没有验收集: {missing}", file=sys.stderr)
        return 1

    summarise("选择集排名（只用于选，不用于报成绩）", selection, args.baseline)
    champion = max(selection, key=lambda n: st.mean(list(selection[n].values())))
    print(f"\n选择集冠军: {champion}")

    report = {champion: acceptance[champion]}
    if args.baseline_csv:
        scores, seen = load_worlds([Path(args.baseline_csv)])
        report_duplicates("基准CSV", seen, len(scores))
        report["基准"] = scores
    summarise("验收集（仅冠军，无偏）", report, "基准" if args.baseline_csv else None)
    print("\n注：验收集只能报告一次，不得据此回头调参；否则它就变成了新的选择集。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
