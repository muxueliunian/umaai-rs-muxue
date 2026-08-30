"""模型形状、person mask 与 choice 占位测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import (  # noqa: E402
    CARD_COUNT,
    CARD_DIM,
    CHOICE_DIM,
    GLOBAL_DIM,
    INPUT_DIM,
    OUTPUT_DIM,
    PERSON_DIM,
    PERSON_PRESENT_INDEX,
    POLICY_DIM,
    ModelConfig,
    RamenNetwork,
)


class ModelTests(unittest.TestCase):
    """冻结输入输出契约与掩码语义。"""

    def test_dynamic_batch_shape_and_choice_is_zero(self) -> None:
        model = RamenNetwork(ModelConfig(token_dim=32, heads=4, encoder_blocks=1, mlp_width=64, mlp_blocks=1, dropout=0.0))
        model.eval()
        for batch in (1, 5):
            output = model(torch.randn(batch, INPUT_DIM))
            print("batch", batch, "shape", tuple(output.shape))
            self.assertEqual(tuple(output.shape), (batch, OUTPUT_DIM))
            choice = output[:, POLICY_DIM : POLICY_DIM + CHOICE_DIM]
            self.assertTrue(torch.equal(choice, torch.zeros_like(choice)))

    def test_absent_person_row_is_ignored(self) -> None:
        torch.manual_seed(1)
        model = RamenNetwork(ModelConfig(token_dim=32, heads=4, encoder_blocks=1, mlp_width=64, mlp_blocks=1, dropout=0.0)).eval()
        first = torch.randn(2, INPUT_DIM)
        person_base = GLOBAL_DIM + CARD_COUNT * CARD_DIM
        row = person_base + 12 * PERSON_DIM
        first[:, row : row + PERSON_DIM] = 0.0
        second = first.clone()
        second[:, row : row + PERSON_DIM] = torch.randn(2, PERSON_DIM) * 100.0
        second[:, row + PERSON_PRESENT_INDEX] = 0.0
        with torch.inference_mode():
            a = model(first)
            b = model(second)
        error = float(torch.max(torch.abs(a - b)))
        print("未登场 person 行扰动误差:", error)
        self.assertEqual(error, 0.0)


if __name__ == "__main__":
    unittest.main()
