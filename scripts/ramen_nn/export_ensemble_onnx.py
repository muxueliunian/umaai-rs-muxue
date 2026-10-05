"""把多个 checkpoint 导出成单个集成 ONNX，并与 PyTorch 数值对拍。

集成规则与 `saved_models/c8_top3` 一致：

- **policy / choice**：直接取算术平均 **logit**。Rust 侧不做 softmax，所以平均必须
  发生在 logit 层面；先各自 softmax 再平均会得到另一个分布。
- **value**：各成员按自己的 `value_normalization` 反归一化到原始分数尺度后平均，
  再按**成员 0** 的常数重新归一化。同一数据集训出的成员常数相同，此时该步是恒等变换；
  跨数据集的成员则必须这样做，否则平均的是尺度不同的数。

用法::

    python export_ensemble_onnx.py --checkpoint a.pt --checkpoint b.pt --checkpoint c.pt \
        --output saved_models/ens/model.onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

try:
    from .export_onnx import CONSERVATIVE_OPS, _test_input
    from .model import INPUT_DIM, POLICY_DIM, CHOICE_DIM, model_from_checkpoint
except ImportError:
    from export_onnx import CONSERVATIVE_OPS, _test_input
    from model import INPUT_DIM, POLICY_DIM, CHOICE_DIM, model_from_checkpoint

#: policy 与 choice 合起来的宽度，value 从这里往后
LOGIT_DIM = POLICY_DIM + CHOICE_DIM

#: 集成规则的人类可读描述，写进 sidecar 供后续追溯
ENSEMBLE_RULE = "policy/choice 取算术平均 logit；value 各自反归一化后平均，再按成员 0 的常数重归一化"


class EnsembleNetwork(nn.Module):
    """把若干 `RamenNetwork` 包成一个前向，输出契约与单体完全一致。"""

    def __init__(self, models: list[nn.Module], centers: list[list[float]], scales: list[list[float]]) -> None:
        super().__init__()
        if len(models) < 2:
            raise ValueError("集成至少需要 2 个成员")
        self.members = nn.ModuleList(models)
        # 注册成 buffer，导出时会折叠成常量，不引入新算子
        for i, (center, scale) in enumerate(zip(centers, scales)):
            self.register_buffer(f"center_{i}", torch.tensor(center, dtype=torch.float32))
            self.register_buffer(f"scale_{i}", torch.tensor(scale, dtype=torch.float32))
        # 重归一化写成仿射 `v * a + b` 而不是 `(v - c) / s`：后者会导出出 `Sub`，
        # 而 Rust 侧的算子白名单里没有它。
        base_center = torch.tensor(centers[0], dtype=torch.float32)
        base_scale = torch.tensor(scales[0], dtype=torch.float32)
        self.register_buffer("renorm_scale", 1.0 / base_scale)
        self.register_buffer("renorm_bias", -base_center / base_scale)

    def forward(self, x: Tensor) -> Tensor:
        """从 ``[B,754]`` 产生 ``[B,245]``，与单体同形。"""

        logits = None
        value_sum = None
        for i, member in enumerate(self.members):
            output = member(x)
            head = output[:, :LOGIT_DIM]
            value = output[:, LOGIT_DIM:]
            denormalized = value * getattr(self, f"scale_{i}") + getattr(self, f"center_{i}")
            logits = head if logits is None else logits + head
            value_sum = denormalized if value_sum is None else value_sum + denormalized
        count = float(len(self.members))
        mean_logits = logits / count
        mean_value = value_sum / count
        renormalized = mean_value * self.renorm_scale + self.renorm_bias
        return torch.cat([mean_logits, renormalized], dim=1)


def export_ensemble(checkpoint_paths: list[Path], output_path: Path, opset: int = 13) -> dict:
    """导出集成 ONNX、审计算子集合，并对 batch=1/7 做逐元素对拍。

    # 错误

    成员少于 2 个、算子越出保守白名单，或 PyTorch/ONNX 误差 ≥ 1e-4 时报错。
    """

    try:
        import onnx
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("导出验证需要 onnx 与 onnxruntime；请安装 requirements.txt") from error

    models: list[nn.Module] = []
    centers: list[list[float]] = []
    scales: list[list[float]] = []
    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        models.append(model_from_checkpoint(checkpoint, "cpu").eval())
        normalization = checkpoint["value_normalization"]
        centers.append(list(normalization["center"]))
        scales.append(list(normalization["scale"]))

    model = EnsembleNetwork(models, centers, scales).eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.from_numpy(_test_input(2, 1234))
    torch.onnx.export(
        model,
        example,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        dynamo=False,
    )

    graph = onnx.load(output_path)
    onnx.checker.check_model(graph)
    operators = sorted({node.op_type for node in graph.graph.node})
    unexpected = sorted(set(operators) - CONSERVATIVE_OPS)
    if unexpected:
        raise RuntimeError(f"ONNX 出现未列入保守白名单的算子: {unexpected}")

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    max_error = 0.0
    per_batch: dict[str, float] = {}
    with torch.inference_mode():
        for batch in (1, 7):
            x = _test_input(batch, 10_000 + batch)
            torch_output = model(torch.from_numpy(x)).numpy()
            onnx_output = session.run(["output"], {"input": x})[0]
            error = float(np.max(np.abs(torch_output - onnx_output)))
            per_batch[str(batch)] = error
            max_error = max(max_error, error)
    if max_error >= 1e-4:
        raise RuntimeError(f"PyTorch/ONNX 最大逐元素误差 {max_error:.8g} >= 1e-4")

    report = {
        "ensemble_of": [str(path) for path in checkpoint_paths],
        "members": len(checkpoint_paths),
        "rule": ENSEMBLE_RULE,
        "onnx": str(output_path),
        "opset": opset,
        "operators": operators,
        "dynamic_batch_tested": [1, 7],
        "max_abs_error": max_error,
        "max_abs_error_by_batch": per_batch,
        "input_dim": INPUT_DIM,
        "output_dim": 245,
        "value_normalization": {"center": centers[0], "scale": scales[0]},
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    """解析导出命令。"""

    parser = argparse.ArgumentParser(description="导出并验证拉面杯集成 ONNX")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True, help="成员 checkpoint；可重复")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args = _parse_args()
    report = export_ensemble(args.checkpoint, args.output, args.opset)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
