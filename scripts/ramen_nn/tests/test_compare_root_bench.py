"""`compare_root_bench` 的逐决策比较边界测试。

只钉一个边界：剔除参考侧单候选之后，**缺组是否算差异**取决于剔除后还剩不剩记录。
这是实现里最容易写反的地方——若先按键集合判缺失、后剔除，就会把「整条 rollout
原本全是单候选、优化后一条记录都不剩」这种合法情形误报成缺失。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compare_root_bench import diff_decision_groups  # noqa: E402


def rec(seq: int, n_actions: int, turn: int = 5, stage: str = "Train") -> dict[str, str]:
    """构造一条逐决策记录（字段名与 `--decision-csv` 表头一致）。"""
    return {
        "seq": str(seq),
        "turn": str(turn),
        "stage": stage,
        "n_actions": str(n_actions),
        "actions": "a|b" if n_actions > 1 else "a",
        "chosen": "0",
        "features": "1.0e0|2.0e0"
    }


class TestDiffDecisionGroups(unittest.TestCase):
    """缺组的两种情形必须给出相反的判定。"""

    def test_all_filtered_group_may_be_absent(self) -> None:
        """整条 rollout 全是单候选 → 优化侧没有该组是**合法**的，不算差异。"""
        ref = {(0, 0): [rec(0, 1), rec(1, 1)]}
        new: dict[tuple[int, int], list[dict[str, str]]] = {}
        len_mismatch, diffs, samples = diff_decision_groups(ref, new, drop_single_candidate=True)
        print(f"全部被过滤: 条数不一致组数={len_mismatch} 字段差异={dict(diffs)} 样本={samples}")
        self.assertEqual(len_mismatch, 0)
        self.assertEqual(dict(diffs), {})

    def test_remaining_multi_candidate_group_must_not_be_absent(self) -> None:
        """过滤后仍有多候选记录 → 优化侧缺该组必须判为差异。"""
        ref = {(0, 0): [rec(0, 1), rec(1, 3)]}
        new: dict[tuple[int, int], list[dict[str, str]]] = {}
        len_mismatch, diffs, samples = diff_decision_groups(ref, new, drop_single_candidate=True)
        print(f"仍有多候选: 条数不一致组数={len_mismatch} 样本={samples}")
        self.assertEqual(len_mismatch, 1)

    def test_matched_group_compares_fields(self) -> None:
        """条数相同则逐字段比较，字段不同要被计数。"""
        ref = {(0, 0): [rec(0, 1), rec(1, 3)]}
        new = {(0, 0): [rec(0, 3, turn=9)]}
        len_mismatch, diffs, _ = diff_decision_groups(ref, new, drop_single_candidate=True)
        print(f"字段不同: 条数不一致组数={len_mismatch} 字段差异={dict(diffs)}")
        self.assertEqual(len_mismatch, 0)
        self.assertEqual(dict(diffs), {"turn": 1})


if __name__ == "__main__":
    unittest.main()
