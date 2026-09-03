"""导出保守算子 ONNX，并用 ONNX Runtime 与 PyTorch 数值对拍。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .model import CARD_COUNT, CARD_DIM, GLOBAL_DIM, INPUT_DIM, PERSON_COUNT, PERSON_DIM, PERSON_PRESENT_INDEX, model_from_checkpoint
except ImportError:
    from model import CARD_COUNT, CARD_DIM, GLOBAL_DIM, INPUT_DIM, PERSON_COUNT, PERSON_DIM, PERSON_PRESENT_INDEX, model_from_checkpoint

CONSERVATIVE_OPS = {
    "Add",
    "Cast",
    "Clip",
    "Concat",
    "Constant",
    "ConstantOfShape",
    "Div",
    "Expand",
    "Gather",
    "Gemm",
    "Identity",
    "MatMul",
    "Mul",
    "ReduceMean",
    "ReduceSum",
    "Relu",
    "Reshape",
    "Shape",
    "Slice",
    "Softmax",
    "Transpose",
    "Unsqueeze",
}


def _test_input(batch: int, seed: int) -> np.ndarray:
    """生成包含真实 0/1 person 掩码形态的确定性测试输入。"""

    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.5, size=(batch, INPUT_DIM)).astype(np.float32)
    person_base = GLOBAL_DIM + CARD_COUNT * CARD_DIM
    for person in range(PERSON_COUNT):
        present = rng.integers(0, 2, size=batch, dtype=np.int32).astype(np.float32)
        row = person_base + person * PERSON_DIM
        x[:, row + PERSON_PRESENT_INDEX] = present
        x[:, row : row + PERSON_DIM] *= present[:, None]
        x[:, row + PERSON_PRESENT_INDEX] = present
    return x


def export_and_verify(checkpoint_path: Path, output_path: Path, opset: int = 13) -> dict:
    """导出 ONNX、审计算子集合，并对 batch=1/7 做逐元素对拍。"""

    try:
        import onnx
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("导出验证需要 onnx 与 onnxruntime；请安装 requirements.txt") from error

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = model_from_checkpoint(checkpoint, "cpu").eval()
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
        "checkpoint": str(checkpoint_path),
        "onnx": str(output_path),
        "opset": opset,
        "operators": operators,
        "dynamic_batch_tested": [1, 7],
        "max_abs_error": max_error,
        "max_abs_error_by_batch": per_batch,
        "input_dim": INPUT_DIM,
        "output_dim": 245,
        "model_config": checkpoint["model_config"],
        "value_normalization": checkpoint["value_normalization"],
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    """解析导出命令。"""

    parser = argparse.ArgumentParser(description="导出并验证拉面杯 ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args = _parse_args()
    report = export_and_verify(args.checkpoint, args.output, args.opset)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
