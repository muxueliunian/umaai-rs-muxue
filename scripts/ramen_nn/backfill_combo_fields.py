"""为只写了 `combo_key` 的旧导出目录补出完整的 `(马娘, 6 张卡)` 字段。

旧目录（`export_version = 1`）的组合身份只有 `combo_key.npy`（FNV 指纹）与
`meta.json` 的 `plan_count`。新数据用的是 `combo_fields.npy [N,7]` 完整字段。
两者不能互推：从指纹反猜字段既不可靠也被本轮口径禁止。

本脚本走**实际采样计划**这条来源链，不碰任何指纹：

1. 采样器把 `index` 映射到计划的口径是 `plan_index = index % plan_count`
   （与 `ramen_export_npy` 写 `combo_key` 时同一行代码）。
2. 计划表由 `sampler.rs` 的测试原样打印出来（`GEN1PLAN` / `OODPLAN` 行），
   计划数必须与该目录 `meta.json` 的 `plan_count` 逐值相同。
3. 结果用数据自身的字段复核：`x.npy` 的卡片 token 头 7 位是卡片类型 one-hot，
   逐样本、逐槽位与所分配卡组的实际类型比对；`combo_key` 只用来做**等值**
   一致性检查（同一计划内取值相同、不同计划之间互不相同），不重算指纹。

用法::

    python scripts/ramen_nn/backfill_combo_fields.py \
        --plan-dump target/gen2_train_0914/plan_dump.txt \
        --output training_data/combo_fields_backfill \
        --data training_data/npy_v6ck --data ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# `features.rs`：GLOBAL_DIM 154、CARD_DIM 35、CARD_NUM 6，卡片 token 的头
# CARD_TYPE_NUM=7 位是类型 one-hot（`encode_cards` 的第一个写入）。
GLOBAL_DIM = 154
CARD_DIM = 35
CARD_NUM = 6
CARD_TYPE_NUM = 7


def load_plan_dump(path: Path) -> dict[int, list[tuple[int, list[int]]]]:
    """读取 `cargo test` 打印的计划表，按计划数分组返回。"""

    spaces: dict[str, list[tuple[int, list[int]]]] = {}
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0].endswith("PLAN") and len(parts) >= 4:
            tag = parts[0]
            idx, uma, deck = int(parts[1]), int(parts[2]), [int(v) for v in parts[3].split(",")]
            if len(deck) != CARD_NUM:
                raise ValueError(f"{tag} 第 {idx} 行卡组不是 6 张: {deck}")
            rows = spaces.setdefault(tag, [])
            if len(rows) != idx:
                raise ValueError(f"{tag} 计划序号不连续：期望 {len(rows)}，实得 {idx}")
            rows.append((uma, deck))
        elif parts[0].endswith("PLAN_COUNT"):
            counts[parts[0].removesuffix("_COUNT")] = int(parts[1])
    by_size: dict[int, list[tuple[int, list[int]]]] = {}
    for tag, rows in spaces.items():
        declared = counts.get(tag)
        if declared is None or declared != len(rows):
            raise ValueError(f"{tag}: 打印了 {len(rows)} 个计划，声明 {declared}")
        if len(rows) in by_size:
            raise ValueError(f"两个空间的计划数都是 {len(rows)}，无法按 plan_count 区分")
        by_size[len(rows)] = rows
        print(f"计划表 {tag}: {len(rows)} 个计划")
    return by_size


def card_types(card_db: Path) -> dict[int, int]:
    """从 `cardDB.json` 读 `cardId -> cardType`。"""

    with card_db.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(entry["cardId"]): int(entry["cardType"]) for entry in raw.values()}


def backfill(data_dir: Path, plans: list[tuple[int, list[int]]], types: dict[int, int]) -> np.ndarray:
    """为一个旧目录算出 `[N,7]` 完整字段，并做三项复核。"""

    index = np.load(data_dir / "index.npy", allow_pickle=False)
    plan_count = len(plans)
    plan_index = (index % plan_count).astype(np.int64)

    # 复核一：`combo_key` 在同一计划内取值相同、不同计划之间互不相同。
    # 只比较已落盘的取值，不重新计算任何指纹。
    key_path = data_dir / "combo_key.npy"
    if key_path.is_file():
        keys = np.asarray(np.load(key_path, mmap_mode="r", allow_pickle=False))
        seen: dict[int, int] = {}
        for p, k in zip(plan_index, keys):
            p, k = int(p), int(k)
            if seen.setdefault(p, k) != k:
                raise ValueError(f"{data_dir}: 计划 {p} 的 combo_key 不唯一")
        if len(set(seen.values())) != len(seen):
            raise ValueError(f"{data_dir}: 不同计划共用了同一个 combo_key")
        print(f"  combo_key 分组一致：{len(seen)} 个计划，键两两不同")

    # 复核二：卡片类型 one-hot 与所分配卡组逐样本、逐槽位比对
    x = np.load(data_dir / "x.npy", mmap_mode="r", allow_pickle=False)
    expect = np.asarray([[types[c // 10] for c in deck] for _, deck in plans], dtype=np.int64)
    mismatch = 0
    step = 4096
    for lo in range(0, len(index), step):
        hi = min(lo + step, len(index))
        block = np.asarray(x[lo:hi, GLOBAL_DIM : GLOBAL_DIM + CARD_NUM * CARD_DIM], dtype=np.float32)
        block = block.reshape(hi - lo, CARD_NUM, CARD_DIM)[:, :, :CARD_TYPE_NUM]
        if not np.array_equal(block.sum(axis=2), np.ones((hi - lo, CARD_NUM), dtype=np.float32)):
            raise ValueError(f"{data_dir}: 第 {lo}..{hi} 段的卡片类型不是单点 one-hot")
        mismatch += int(np.count_nonzero(np.argmax(block, axis=2) != expect[plan_index[lo:hi]]))
    if mismatch:
        raise ValueError(f"{data_dir}: 卡片类型与所分配卡组不符的槽位 {mismatch} 个")
    print(f"  卡片类型逐槽位一致：{len(index)} 样本 × {CARD_NUM} 槽位，0 处不符")

    fields = np.empty((len(index), 7), dtype=np.uint64)
    for p, (uma, deck) in enumerate(plans):
        row = np.asarray([uma, *sorted(deck)], dtype=np.uint64)
        fields[plan_index == p] = row
    return fields


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="为旧导出目录补出完整组合字段")
    parser.add_argument("--plan-dump", type=Path, required=True)
    parser.add_argument("--card-db", type=Path, default=Path("gamedata/cardDB.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, action="append", required=True)
    args = parser.parse_args()

    by_size = load_plan_dump(args.plan_dump)
    types = card_types(args.card_db)
    args.output.mkdir(parents=True, exist_ok=True)
    for data_dir in args.data:
        meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
        plan_count = int(meta["plan_count"])
        if plan_count not in by_size:
            raise ValueError(f"{data_dir}: plan_count {plan_count} 没有对应的计划表")
        print(f"{data_dir}（plan_count {plan_count}）")
        fields = backfill(data_dir, by_size[plan_count], types)
        out = args.output / f"{data_dir.name}.npy"
        np.save(out, fields)
        print(f"  已写出 {out}，{len(fields)} 行 × 7 列，{len(np.unique(fields, axis=0))} 个不同组合")


if __name__ == "__main__":
    main()
