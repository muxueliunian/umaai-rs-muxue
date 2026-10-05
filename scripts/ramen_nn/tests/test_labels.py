"""标签配方的独立单元测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labels import (  # noqa: E402
    bayesian_best_probabilities,
    candidate_probs_to_policy,
    crossfit_value_target,
    leave_one_out_outcomes,
    make_bayesian_weights,
    weighted_mean,
)


class LabelTests(unittest.TestCase):
    """CRN、并列、地区边缘与 cross-fit 回归。"""

    def test_crn_paired_difference_is_not_drowned_by_absolute_noise(self) -> None:
        base = np.asarray([0.0, 1000.0, -500.0, 700.0] * 32, dtype=np.float32)
        scores = np.stack([base + 1.0, base], axis=0)
        weights = make_bayesian_weights(scores.shape[1], 256, 7)
        probabilities = bayesian_best_probabilities(scores, weights)
        print("固定配对优势 1 分的最优概率:", probabilities)
        self.assertTrue(np.allclose(probabilities, [1.0, 0.0]))

    def test_exact_equivalent_candidates_split_probability(self) -> None:
        base = np.linspace(40_000.0, 60_000.0, 128, dtype=np.float32)
        scores = np.stack([base, base.copy(), base - 100.0], axis=0)
        weights = make_bayesian_weights(128, 256, 9)
        probabilities = bayesian_best_probabilities(scores, weights)
        print("等价候选概率:", probabilities)
        self.assertTrue(np.allclose(probabilities, [0.5, 0.5, 0.0]))

    def test_equivalent_candidates_split_at_production_scale(self) -> None:
        # 生产规模：约 6.5 万分、1024 列、十余个候选且重复行不相邻。float32 累加的
        # 舍入差远大于 tie_atol，会按 BLAS 分块位置随机打破并列。
        rng = np.random.default_rng(20261005)
        scores = rng.normal(65_000.0, 3_000.0, size=(12, 1024)).astype(np.float32)
        scores[9] = scores[0]
        scores[0] += 200.0
        scores[9] = scores[0]
        weights = make_bayesian_weights(1024, 512, 20260830)
        probabilities = bayesian_best_probabilities(scores, weights)
        print("生产规模等价候选概率:", probabilities[[0, 9]], "其余之和:", probabilities.sum() - probabilities[[0, 9]].sum())
        self.assertAlmostEqual(float(probabilities[0]), float(probabilities[9]), places=6)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=5)

    def test_masked_branch_equivalent_candidates_and_permutation(self) -> None:
        # 生产数据都带 cand_valid：走「按有效位重新归一化」分支。
        rng = np.random.default_rng(20261006)
        scores = rng.normal(65_000.0, 3_000.0, size=(12, 1024)).astype(np.float32)
        scores[0] += 200.0
        scores[9] = scores[0]
        valid = np.ones_like(scores, dtype=bool)
        weights = make_bayesian_weights(1024, 512, 20260830)

        full = bayesian_best_probabilities(scores, weights, valid)
        plain = bayesian_best_probabilities(scores, weights)
        print("全有效 mask 与无 mask:", full[[0, 9]], plain[[0, 9]])
        self.assertTrue(np.allclose(full, plain, atol=1e-6))

        valid[[0, 9], 100:140] = False  # 等价候选带相同的部分失效位
        partial = bayesian_best_probabilities(scores, weights, valid)
        print("相同部分失效 mask 的等价候选:", partial[[0, 9]])
        self.assertAlmostEqual(float(partial[0]), float(partial[9]), places=6)

        order = rng.permutation(12)
        permuted = bayesian_best_probabilities(scores[order], weights, valid[order])
        print("置换候选顺序后概率一致:", np.allclose(permuted, partial[order], atol=1e-6))
        self.assertTrue(np.allclose(permuted, partial[order], atol=1e-6))

    def test_small_real_gap_still_separates_at_production_scale(self) -> None:
        # 6.5 万分尺度下的 0.5 分配对优势仍应被判为严格更优，不能被并列容差吞掉。
        rng = np.random.default_rng(20261007)
        base = rng.normal(65_000.0, 3_000.0, size=1024)
        scores = np.stack([base + 0.5, base]).astype(np.float64)
        weights = make_bayesian_weights(1024, 512, 20260830)
        probabilities = bayesian_best_probabilities(scores, weights)
        print("0.5 分配对优势的最优概率:", probabilities)
        self.assertTrue(np.allclose(probabilities, [1.0, 0.0]))

    def test_region_candidate_distribution_becomes_normalized_marginals(self) -> None:
        probabilities = np.asarray([0.75, 0.25], dtype=np.float32)
        slots = np.asarray([[214, 215, 216], [214, 217, 218]], dtype=np.int32)
        target = candidate_probs_to_policy(probabilities, slots)
        print("地区边缘:", target[214:219], "sum=", target.sum())
        self.assertAlmostEqual(float(target.sum()), 1.0, places=6)
        self.assertTrue(np.allclose(target[214:219], [1 / 3, 0.25, 0.25, 1 / 12, 1 / 12]))

    def test_leave_one_out_removes_outlier_selection_optimism(self) -> None:
        scores = np.asarray([[100.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 10.0]], dtype=np.float32)
        outcomes, selected = leave_one_out_outcomes(scores)
        target, stability = crossfit_value_target(scores, radical_factor=0.0)
        print("LOO selected:", selected, "outcomes:", outcomes, "target:", target)
        self.assertAlmostEqual(float(np.mean(outcomes)), 2.5)
        self.assertAlmostEqual(float(target[0]), 2.5)
        self.assertAlmostEqual(float(target[2]), 2.5)
        self.assertAlmostEqual(stability, 0.75)

    def test_weighted_mean_matches_mean_at_zero_and_favors_upper_tail(self) -> None:
        values = np.asarray([1.0, 1.0, 3.0, 7.0], dtype=np.float32)
        plain = weighted_mean(values, 0.0)
        radical = weighted_mean(values, 1.4)
        print("mean / weighted:", plain, radical)
        self.assertAlmostEqual(plain, 3.0)
        self.assertGreater(radical, plain)


if __name__ == "__main__":
    unittest.main()
