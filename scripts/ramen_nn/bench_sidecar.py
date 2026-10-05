"""本地性能基准侧车：严格 FP32、固定批量，stdout 仅传二进制结果与控制回执。"""

import argparse
import struct
import sys

import numpy as np
import torch

from model import model_from_checkpoint

#: 控制消息哨兵。数据头 (rows, valid) 的含义不变，rows 恒在 1..=物理批 内，
#: 取不到这个值，于是控制消息与数据消息天然可分，不需要重新解释旧字段。
CTRL_MAGIC = 0xFFFFFFFF
#: 控制操作码：设置物理张量批尺寸
OP_SET_BATCH = 1
#: 协议版本，与 Rust 侧 SIDECAR_PROTO 对应；写进就绪行供两端校验
PROTO = 2


def read_exact(stream, size):
    """读取完整消息；截断消息报错，消息边界 EOF 返回 None。"""
    chunks = bytearray()
    while len(chunks) < size:
        part = stream.read(size - len(chunks))
        if not part:
            if not chunks:
                return None
            raise EOFError("truncated sidecar message")
        chunks.extend(part)
    return chunks


def main():
    """加载冻结成员、预热，再处理 Rust 的双 u32 头、f32 行与换档控制消息。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="把整段集成前向捕获成 CUDA Graph 后重放；默认关闭，与换档互斥",
    )
    args = parser.parse_args()
    if args.batch < 1 or not torch.cuda.is_available():
        raise RuntimeError("positive batch and CUDA required; no CPU fallback")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    checkpoints = [torch.load(p, map_location="cpu", weights_only=False) for p in args.checkpoint]
    models = [model_from_checkpoint(c, "cuda").eval() for c in checkpoints]
    centers = [torch.tensor(c["value_normalization"]["center"], device="cuda") for c in checkpoints]
    scales = [torch.tensor(c["value_normalization"]["scale"], device="cuda") for c in checkpoints]
    # 缓冲区按上限分配一次并常驻：换档只改前向用的行数，不重启进程、不复制模型
    buf = torch.zeros((args.batch, 754), device="cuda", dtype=torch.float32)
    active = args.batch
    warmed = set()

    def infer(rows_n):
        """在前 rows_n 行上做集成前向：平均 policy logits；value 在实际分数空间平均后还原成员零尺度。"""
        view = buf[:rows_n]
        outputs = [model(view) for model in models]
        logits = torch.stack([o[:, :242] for o in outputs]).mean(0)
        values = torch.stack([o[:, 242:] * s + c for o, s, c in zip(outputs, scales, centers)]).mean(0)
        return torch.cat((logits, (values - centers[0]) / scales[0]), dim=1)

    # 开图时物理批固定：图是按某个行数捕的，换档必须重捕，本轮不与自适应批耦合
    graph = None
    static_out = None

    def run(rows_n):
        """跑一次完整集成前向；开图时重放，否则走 eager。

        重放返回的是图的**静态输出缓冲区**，调用方必须在下一次重放前取走内容。
        """
        if graph is None:
            return infer(rows_n)
        if rows_n != active:
            raise ValueError(f"graph captured for batch {active}, got {rows_n}")
        graph.replay()
        return static_out

    def warmup(rows_n):
        """某档位首次使用前预热；否则首个请求会把 kernel 选型算进推理耗时。"""
        if rows_n in warmed:
            return
        buf[:rows_n].zero_()
        for _ in range(3):
            infer(rows_n)
        torch.cuda.synchronize()
        warmed.add(rows_n)

    with torch.inference_mode():
        warmup(args.batch)
        if args.cuda_graph:
            # 捕获前在旁路 stream 上再预热：kernel 选型、cuBLAS 句柄与 workspace
            # 这类一次性初始化不能录进图里，否则每次重放都会重做一遍
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    infer(active)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = infer(active)
            torch.cuda.synchronize()
        print(
            f"[sidecar] ready device=cuda B={args.batch} members={len(models)} tf32=off warmup=3"
            f" proto={PROTO} maxB={args.batch} graph={'on' if args.cuda_graph else 'off'}",
            file=sys.stderr,
            flush=True,
        )
        while True:
            header = read_exact(sys.stdin.buffer, 8)
            if header is None:
                break
            first, second = struct.unpack("<II", header)
            if first == CTRL_MAGIC:
                if second != OP_SET_BATCH:
                    raise ValueError(f"unknown control opcode: {second}")
                # 本轮不把 Graph 与自适应批捆绑：换档要重新捕获，静默沿用旧图会算错行数
                if graph is not None:
                    raise ValueError("batch switching is refused while --cuda-graph is on")
                body = read_exact(sys.stdin.buffer, 4)
                if body is None:
                    raise EOFError("missing control payload")
                (want,) = struct.unpack("<I", body)
                if not 0 < want <= args.batch:
                    raise ValueError(f"invalid physical batch: {want} (max {args.batch})")
                active = want
                warmup(active)
                # 回执让 Rust 侧能确认换档真的生效，而不是单侧改了一个数
                sys.stdout.buffer.write(struct.pack("<I", active))
                sys.stdout.buffer.flush()
                continue
            rows, valid = first, second
            if not 0 < valid <= rows <= active:
                raise ValueError(f"invalid rows/valid: {rows}/{valid} (active batch {active})")
            payload = read_exact(sys.stdin.buffer, rows * 754 * 4)
            if payload is None:
                raise EOFError("missing input payload")
            array = np.frombuffer(payload, dtype="<f4").reshape(rows, 754).copy()
            buf[:active].zero_()
            buf[:rows].copy_(torch.from_numpy(array))
            output = run(active)[:valid].cpu().numpy().astype("<f4", copy=False)
            if not np.isfinite(output).all():
                raise ValueError("nonfinite model output")
            sys.stdout.buffer.write(output.tobytes())
            sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
