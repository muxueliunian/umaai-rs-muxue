"""生成混训用的**开发验证组合清单**（完整马娘/卡组字段）。

两条规则，全局统一，同一个组合只会出现在一侧：

1. **旧八组按 R4 已发生的划分原样保留**。R4 用的是
   ``stable_split_refs(split_by="combo")``，键取各目录已落盘的 ``combo_key.npy``，
   分桶函数是 ``data._splitmix64``。这里只是把那次划分**重放**一遍：不读取任何
   文件内容、不计算任何新指纹，只把已存在的整数键送进仓库原有的划分函数。
   重放结果会与 ``target/arm_R4_seed1/run.json`` 记的 train/val 条数逐值核对。
2. **新批只在旧规则未锁定的组合里补足**。按 (马娘, 卡组构成) 分层，层内按完整
   字段字典序排序后每十个取第十个——与 ``holdout.json`` 的选法同一口径，
   确定性、与标签/分数/模型表现无关，且不计算哈希。

最终清单 = 两部分的并集。``holdout.json`` 的 250 个最终留出组合不在训练语料里，
本脚本另行核对它们既不在旧数据也不在新批中。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .data import _splitmix64
except ImportError:
    from data import _splitmix64

CARD_TYPE_NAMES = ["速", "耐", "力", "根", "智"]
CARD_TYPE_FRIEND = 5


def shape_name(fields: tuple[int, ...], types: dict[int, int]) -> str:
    """按六张卡的实际类型还原构成名，与 `sampler::format_shape_name` 同口径。"""

    counts = [0] * 5
    friends = 0
    for card in fields[1:]:
        kind = types[card // 10]
        if kind == CARD_TYPE_FRIEND:
            friends += 1
        elif 0 <= kind < 5:
            counts[kind] += 1
        else:
            raise ValueError(f"卡 {card} 的类型 {kind} 不受支持")
    if friends != 1 or sum(counts) != 5:
        raise ValueError(f"组合 {fields} 不是 5 张普通卡 + 1 张友人卡")
    return "".join(f"{n}{CARD_TYPE_NAMES[i]}" for i, n in enumerate(counts) if n) + "1友"


def load_fields(path: Path) -> np.ndarray:
    """读一份 `[N,7]` 完整字段数组。"""

    rows = np.load(path, allow_pickle=False)
    if rows.ndim != 2 or rows.shape[1] != 7:
        raise ValueError(f"{path}: 形状 {rows.shape} 不是 [N,7]")
    return rows


def distinct(rows: np.ndarray) -> set[tuple[int, ...]]:
    """取出不同组合。"""

    return {tuple(int(v) for v in r) for r in np.unique(rows, axis=0)}


def replay_r4_split(
    data_dirs: list[Path], backfill: Path, seed: int, fraction: float
) -> tuple[set[tuple[int, ...]], set[tuple[int, ...]], dict[str, int]]:
    """重放 R4 的按组合划分，返回 (验证侧组合, 训练侧组合, 样本条数)。"""

    threshold = int(fraction * 10_000)
    val: set[tuple[int, ...]] = set()
    train: set[tuple[int, ...]] = set()
    counts = {"train": 0, "validation": 0}
    for data_dir in data_dirs:
        keys = np.asarray(np.load(data_dir / "combo_key.npy", mmap_mode="r", allow_pickle=False))
        fields = load_fields(backfill / f"{data_dir.name}.npy")
        if len(keys) != len(fields):
            raise ValueError(f"{data_dir}: combo_key 与补出的字段行数不一致")
        for key, row in zip(keys, fields):
            combo = tuple(int(v) for v in row)
            if _splitmix64(int(key), seed) % 10_000 < threshold:
                val.add(combo)
                counts["validation"] += 1
            else:
                train.add(combo)
                counts["train"] += 1
    crossing = val & train
    if crossing:
        raise ValueError(f"重放出的旧划分有 {len(crossing)} 个组合横跨两侧")
    return val, train, counts


def _parse_args() -> argparse.Namespace:
    """解析命令行。"""

    parser = argparse.ArgumentParser(description="生成开发验证组合清单")
    parser.add_argument("--old-data", type=Path, action="append", default=[])
    parser.add_argument("--new-data", type=Path, action="append", default=[])
    parser.add_argument("--backfill", type=Path, default=Path("training_data/combo_fields_backfill"))
    parser.add_argument("--holdout", type=Path, default=Path("scripts/collect/formal2048_0914/holdout.json"))
    parser.add_argument("--card-db", type=Path, default=Path("gamedata/cardDB.json"))
    parser.add_argument("--split-seed", type=int, default=20260830)
    parser.add_argument("--split-fraction", type=float, default=0.1)
    parser.add_argument("--stride", type=int, default=10, help="新组合分层内每 N 个取第 N 个")
    parser.add_argument("--r4-run", type=Path, default=Path("target/arm_R4_seed1/run.json"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args = _parse_args()
    with args.card_db.open(encoding="utf-8") as handle:
        types = {int(e["cardId"]): int(e["cardType"]) for e in json.load(handle).values()}

    old_val, old_train, old_counts = replay_r4_split(
        args.old_data, args.backfill, args.split_seed, args.split_fraction
    )
    print(f"旧八组重放：train {old_counts['train']} 条 / validation {old_counts['validation']} 条，"
          f"组合 {len(old_train)}+{len(old_val)}")
    if args.r4_run.is_file():
        recorded = json.loads(args.r4_run.read_text(encoding="utf-8"))["split"]
        same = recorded["train"] == old_counts["train"] and recorded["validation"] == old_counts["validation"]
        print(f"  与 {args.r4_run} 的 {recorded['train']}/{recorded['validation']} "
              + ("逐值相同 ✓" if same else "❗不同"))
        if not same:
            raise ValueError("重放没有复现 R4 的划分，拒绝继续")

    new_combos: set[tuple[int, ...]] = set()
    for data_dir in args.new_data:
        new_combos |= distinct(load_fields(data_dir / "combo_fields.npy"))
    print(f"新批不同组合 {len(new_combos)}")

    corpus = new_combos | old_val | old_train
    holdout = {tuple(p["fields"]) for p in json.loads(args.holdout.read_text(encoding="utf-8"))["plans"]}
    leak = corpus & holdout
    print(f"训练语料不同组合 {len(corpus)}；与 250 个最终留出组合的交集 {len(leak)}")
    if leak:
        raise ValueError("最终留出组合出现在训练语料里")

    # 旧规则已经把 gen1 的 525 个组合锁到某一侧；层内补足时把锁定的算进配额，
    # 否则新批的验证占比会被顶高。
    locked_val = new_combos & old_val
    locked_train = new_combos & old_train
    strata: dict[str, list[tuple[int, ...]]] = {}
    for combo in new_combos:
        strata.setdefault(f"{combo[0]}|{shape_name(combo, types)}", []).append(combo)
    picked: set[tuple[int, ...]] = set()
    report = []
    for name in sorted(strata):
        members = sorted(strata[name])
        quota = round(len(members) / args.stride)
        already = [c for c in members if c in locked_val]
        free = [c for c in members if c not in locked_val and c not in locked_train]
        chosen = free[args.stride - 1 :: args.stride][: max(0, quota - len(already))]
        picked |= set(chosen)
        report.append({
            "stratum": name,
            "combos": len(members),
            "quota": quota,
            "locked_val": len(already),
            "locked_train": len(locked_train.intersection(members)),
            "picked": len(chosen),
        })
    validation = sorted(old_val | picked)
    if set(validation) & (old_train - old_val):
        raise ValueError("最终清单与旧训练侧组合重叠")
    in_new = len(new_combos & set(validation))
    print(f"开发验证组合合计 {len(validation)}（旧划分 {len(old_val)}，新批补足 {len(picked)}）")
    print(f"  落在新批 {len(new_combos)} 个组合内的 {in_new}（{100 * in_new / len(new_combos):.1f}%）")
    print(f"  占训练语料 {len(validation)}/{len(corpus)} = {100 * len(validation) / len(corpus):.1f}%")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "method": "旧八组重放 R4 的 combo 划分；新批按 (马娘,构成) 分层、层内字典序每十取十补足",
                "split_seed": args.split_seed,
                "split_fraction": args.split_fraction,
                "stride": args.stride,
                "old_replay_counts": old_counts,
                "corpus_combos": len(corpus),
                "validation_combos": len(validation),
                "strata": report,
                "combos": [list(c) for c in validation],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"已写出 {args.output}")


if __name__ == "__main__":
    main()
