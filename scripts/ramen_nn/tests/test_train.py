"""非法格位 mask 与 choice 零权重训练回归。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import (  # noqa: E402
    CHOICE_DIM,
    EAT_BASE,
    EAT_END,
    INPUT_DIM,
    POLICY_DIM,
    REGION_NUM,
    TRIPLE_NUM,
    ModelConfig,
    RamenNetwork,
)
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

    def _step_once(self, factorized: bool) -> RamenNetwork:
        """建模型、走一步优化，返回模型。两种输出头共用同一套断言。"""

        model = RamenNetwork(
            ModelConfig(
                token_dim=32,
                heads=4,
                encoder_blocks=0,
                mlp_width=64,
                mlp_blocks=1,
                dropout=0.0,
                factorized_eat_head=factorized,
            )
        )
        optimizer = make_optimizer(model, 5e-4, 1.25e-4, 2e-5, 1e-3)
        # 优化器必须覆盖全部参数：因子化头新增的三层若漏进 param group，
        # 会静默地完全不训练，而 loss 仍然在降
        covered = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertEqual(covered, {id(p) for p in model.parameters()})
        x = torch.randn(3, INPUT_DIM)
        output = model(x)
        loss = output[:, :POLICY_DIM].sum() + output[:, -3:].sum()
        loss.backward()
        optimizer.step()
        return model

    def test_choice_rows_remain_zero_after_optimizer_step(self) -> None:
        for factorized in (False, True):
            with self.subTest(factorized=factorized):
                model = self._step_once(factorized)
                choice_weight = model.output.weight[POLICY_DIM : POLICY_DIM + CHOICE_DIM]
                choice_bias = model.output.bias[POLICY_DIM : POLICY_DIM + CHOICE_DIM]
                print(
                    "factorized=%s choice max abs:" % factorized,
                    float(choice_weight.abs().max()),
                    float(choice_bias.abs().max()),
                )
                self.assertTrue(torch.equal(choice_weight, torch.zeros_like(choice_weight)))
                self.assertTrue(torch.equal(choice_bias, torch.zeros_like(choice_bias)))

    def test_factorized_head_keeps_output_layout(self) -> None:
        """因子化只改 [1,201) 的产生方式，输出宽度与其余格位的来源都不变。"""

        model = RamenNetwork(
            ModelConfig(token_dim=32, heads=4, encoder_blocks=0, mlp_width=64, mlp_blocks=1,
                        dropout=0.0, factorized_eat_head=True)
        )
        model.eval()
        x = torch.randn(4, INPUT_DIM)
        with torch.no_grad():
            output = model(x)
        self.assertEqual(tuple(output.shape), (4, POLICY_DIM + CHOICE_DIM + 3))
        # 零初始化的交互项 => 吃面块必须严格可加：z[r,t]-z[r,0]-z[0,t]+z[0,0] == 0
        with torch.no_grad():
            eat = output[:, EAT_BASE:EAT_END].reshape(-1, REGION_NUM, TRIPLE_NUM)
        residual = (eat - eat[:, :, :1] - eat[:, :1, :] + eat[:, :1, :1]).abs().max()
        print("零初始化时可加性残差:", float(residual))
        self.assertLess(float(residual), 1e-4)


if __name__ == "__main__":
    unittest.main()
