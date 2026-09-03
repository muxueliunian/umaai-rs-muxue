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
