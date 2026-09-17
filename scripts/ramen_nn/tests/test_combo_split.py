"""按完整马娘/卡组字段划分留出集的守卫。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import split_refs_by_combos, stable_split_refs  # noqa: E402


class FakeShard:
    """只带 `combo_fields` 的最小分片替身。"""

    def __init__(self, rows: list[list[int]]) -> None:
        self.combo_fields = np.asarray(rows, dtype=np.uint64)
        self.combo_key = None


A = [100603, 302424, 302754, 302894, 302984, 303044, 303054]
B = [100603, 302424, 302754, 302894, 302984, 303004, 303054]
C = [108702, 302424, 302754, 302894, 302984, 303044, 303054]


class ComboSplitTests(unittest.TestCase):
    """留出清单必须切得干净、空集必须报错。"""

    def test_splits_by_fields_and_keeps_a_combo_on_one_side(self) -> None:
        shards = [FakeShard([A, B, A, C]), FakeShard([C, B, A])]
        train, validation = split_refs_by_combos(shards, [B])
        print("train:", train.tolist(), "validation:", validation.tolist())
        self.assertEqual(validation.tolist(), [[0, 1], [1, 1]])
        self.assertEqual(train.tolist(), [[0, 0], [0, 2], [0, 3], [1, 0], [1, 2]])
        train_keys = {tuple(int(v) for v in shards[s].combo_fields[i]) for s, i in train}
        val_keys = {tuple(int(v) for v in shards[s].combo_fields[i]) for s, i in validation}
        self.assertEqual(train_keys & val_keys, set())

    def test_rejects_empty_side_and_duplicate_list(self) -> None:
        shards = [FakeShard([A, A, A])]
        with self.assertRaises(ValueError):
            split_refs_by_combos(shards, [A])  # 训练侧为空
        with self.assertRaises(ValueError):
            split_refs_by_combos(shards, [B])  # 验证侧为空
        with self.assertRaises(ValueError):
            split_refs_by_combos([FakeShard([A, B])], [B, B])  # 清单自身重复
        with self.assertRaises(ValueError):
            split_refs_by_combos([FakeShard([A, B])], [B[:6]])  # 字段宽度不是 7

    def test_missing_fields_shard_is_refused(self) -> None:
        missing = FakeShard([A])
        missing.combo_fields = None
        with self.assertRaises(ValueError):
            split_refs_by_combos([FakeShard([A, B]), missing], [B])

    def test_stable_split_still_refuses_full_field_data(self) -> None:
        """完整字段在场时旧的哈希/索引划分必须继续拒绝，不能静默回落。"""

        with self.assertRaises(ValueError):
            stable_split_refs([FakeShard([A, B, C])], 0.1, 20260830, "combo")


if __name__ == "__main__":
    unittest.main()
