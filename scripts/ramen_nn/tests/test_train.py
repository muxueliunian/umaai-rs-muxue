"""非法格位 mask 与 choice 零权重训练回归。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import CHOICE_DIM, INPUT_DIM, POLICY_DIM, ModelConfig, RamenNetwork  # noqa: E402
from train import make_optimizer, policy_kl_per_sample  # noqa: E402


class TrainTests(unittest.TestCase):
    """训练数值稳定性与冻结占位行。"""

    def test_masked_kl_is_finite_and_has_zero_lower_bound(self) -> None:
        logits = torch.tensor([[2.0, 1.0, 99.0]])
        target = torch.tensor([[1.0, 0.0, 0.0]])
        legal = torch.tensor([[True, True, False]])
        loss = policy_kl_per_sample(logits, target, legal)
        print("masked KL:", loss)
        self.assertTrue(torch.isfinite(loss).all())
        self.assertGreaterEqual(float(loss.item()), 0.0)

    def test_choice_rows_remain_zero_after_optimizer_step(self) -> None:
        model = RamenNetwork(ModelConfig(token_dim=32, heads=4, encoder_blocks=0, mlp_width=64, mlp_blocks=1, dropout=0.0))
        optimizer = make_optimizer(model, 5e-4, 1.25e-4, 2e-5)
        x = torch.randn(3, INPUT_DIM)
        output = model(x)
        loss = output[:, :POLICY_DIM].sum() + output[:, -3:].sum()
        loss.backward()
        optimizer.step()
        choice_weight = model.output.weight[POLICY_DIM : POLICY_DIM + CHOICE_DIM]
        choice_bias = model.output.bias[POLICY_DIM : POLICY_DIM + CHOICE_DIM]
        print("choice max abs:", float(choice_weight.abs().max()), float(choice_bias.abs().max()))
        self.assertTrue(torch.equal(choice_weight, torch.zeros_like(choice_weight)))
        self.assertTrue(torch.equal(choice_bias, torch.zeros_like(choice_bias)))


if __name__ == "__main__":
    unittest.main()
