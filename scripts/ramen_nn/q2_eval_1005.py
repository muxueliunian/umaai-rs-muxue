"""Q2 评测驱动：导出 ONNX（成员 + 三种子集成）→ 真实输入对拍 → 主面板闭环 → 组合级配对分析。

子命令（工作区根）：
  python scripts/ramen_nn/q2_eval_1005.py export  --step final|7500|15000
  python scripts/ramen_nn/q2_eval_1005.py bench   --step final [--offset 430000] [--models A_ens B_ens ...]
  python scripts/ramen_nn/q2_eval_1005.py analyze --step final [--offset 430000]

口径见 logs/q2_1005/preregistration.md：主统计量 = 组合内 16 世界等权、250 组合等权的配对差，
组合级 t 区间（自由度 249）。程序失败、缺行、重复行、配对世界不一致 ⇒ 整个面板作废（报错退出）。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_ensemble_onnx import EnsembleNetwork, export_ensemble  # noqa: E402
from export_onnx import export_and_verify  # noqa: E402
from model import model_from_checkpoint  # noqa: E402

ARMS = ("A", "B", "C", "D")
SEEDS = (1, 2, 3)
STEP_FILES = {"final": "last.pt", "7500": "step_007500.pt", "15000": "step_015000.pt"}
TRAIN_ROOT = Path("target/q2_1005")
ONNX_ROOT = TRAIN_ROOT / "onnx"
BENCH_ROOT = Path("logs/q2_1005/bench")
WORKTREE = Path("target/wt_q2_53d2d86")
BENCH_EXE = WORKTREE / "target/release/ramen_space_bench.exe"
PLANS_FILE = "scripts/collect/r8_gen2_0920/holdout.json"
BASE_SEED = 61444
RUNS_PER_PLAN = 16
PLAN_COUNT = 250
PARITY_ROWS = 512
PARITY_TOL = 1e-3
# t(0.975, df=249)
T_249 = 1.969537
MASK64 = (1 << 64) - 1
MIX_A = 0xBF58476D1CE4E5B9
MIX_B = 0x94D049BB133111EB


def splitmix64(z: int) -> int:
    """与 crates/umasim/src/rng.rs 的 splitmix64 终混合逐位一致。"""

    z = ((z ^ (z >> 30)) * MIX_A) & MASK64
    z = ((z ^ (z >> 27)) * MIX_B) & MASK64
    return z ^ (z >> 31)


def world_seed(plan: int, run_idx: int) -> int:
    """bench 的逐局世界种子：derive_seed(seed + plan*1000003, [run_idx])。"""

    return splitmix64(((BASE_SEED + plan * 1_000_003) & MASK64) ^ run_idx)


def model_names() -> list[str]:
    """主面板的全部模型名：每臂三成员 + 集成。"""

    return [f"{arm}_s{s}" for arm in ARMS for s in SEEDS] + [f"{arm}_ens" for arm in ARMS]


def onnx_path(name: str, step: str) -> Path:
    """模型名 + 步数 → ONNX 路径。"""

    return ONNX_ROOT / step / f"{name}.onnx"


def parity_rows() -> np.ndarray:
    """从冻结验证集 id 中均匀取 PARITY_ROWS 行真实输入（1024 档数据目录）。"""

    frozen = json.loads(Path("logs/q2_1005/frozen/assets.json").read_text(encoding="utf-8"))
    valid = set(frozen["valid"])
    rows = []
    for data_dir, _ in frozen["pairs_w1024"]:
        index = np.load(Path(data_dir) / "index.npy")
        hit = np.flatnonzero(np.isin(index, list(valid)))
        if hit.size:
            x = np.load(Path(data_dir) / "x.npy", mmap_mode="r")
            rows.append(np.asarray(x[hit[:: max(1, hit.size // 16)][:16]]))
    stacked = np.concatenate(rows).astype(np.float32)
    pick = np.linspace(0, len(stacked) - 1, PARITY_ROWS).astype(int)
    return stacked[pick]


def ort_max_error(onnx_file: Path, reference: torch.nn.Module, x: np.ndarray) -> float:
    """真实输入下 PyTorch 与 onnxruntime 的最大逐元素误差。"""

    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    got = session.run(None, {session.get_inputs()[0].name: x})[0]
    with torch.no_grad():
        want = reference(torch.from_numpy(x)).numpy()
    return float(np.max(np.abs(got - want)))


def export_group(ckpts: list[Path], member_outs: list[Path], ens_out: Path, x: np.ndarray) -> dict:
    """导出一组成员与它们的集成，并用真实输入 ``x`` 对拍；返回逐模型的对拍记录。"""

    report, models, centers, scales = {}, [], [], []
    for ckpt, out in zip(ckpts, member_outs):
        checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
        export_and_verify(ckpt, out)
        model = model_from_checkpoint(checkpoint, "cpu").eval()
        report[out.stem] = {"global_step": int(checkpoint["global_step"]), "real_max_abs_error": ort_max_error(out, model, x)}
        models.append(model)
        centers.append(list(checkpoint["value_normalization"]["center"]))
        scales.append(list(checkpoint["value_normalization"]["scale"]))
    export_ensemble(ckpts, ens_out)
    ensemble = EnsembleNetwork(models, centers, scales).eval()
    report[ens_out.stem] = {"real_max_abs_error": ort_max_error(ens_out, ensemble, x)}
    return report


def cmd_export(step: str) -> None:
    """导出全部成员与集成，并做真实输入对拍；超阈值即报错。"""

    x = parity_rows()
    report = {}
    for arm in ARMS:
        ckpts = [TRAIN_ROOT / f"{arm}_seed{s}" / STEP_FILES[step] for s in SEEDS]
        outs = [onnx_path(f"{arm}_s{s}", step) for s in SEEDS]
        report.update(export_group(ckpts, outs, onnx_path(f"{arm}_ens", step), x))
    bad = {k: v for k, v in report.items() if v["real_max_abs_error"] >= PARITY_TOL}
    path = ONNX_ROOT / step / "parity.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if bad:
        raise SystemExit(f"真实输入对拍超阈值 {PARITY_TOL}：{sorted(bad)}")


def bench_csv(name: str, step: str, offset: int) -> Path:
    """某模型某面板的逐局 CSV 路径。"""

    return BENCH_ROOT / f"o{offset}" / step / f"{name}.csv"


def run_bench(name: str, onnx_file: Path, out: Path, offset: int) -> None:
    """用 ``onnx_file`` 跑一个 250×16 面板写到 ``out``；已完整的跳过，不完整的报错（不续跑、不覆盖）。"""

    meta = out.with_name(out.name + ".meta.json")
    if meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        if m.get("complete") and m.get("games_completed") == PLAN_COUNT * RUNS_PER_PLAN:
            print(f"skip {name}（已完整）")
            return
        raise SystemExit(f"{out} 存在但不完整，需人工处理")
    if out.exists():
        raise SystemExit(f"{out} 已存在但缺 meta，需人工处理")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(BENCH_EXE.resolve()), "--trainer", "nn", "--model", str(onnx_file.resolve()),
        "--space-version", "gen2_v1", "--plans-file", PLANS_FILE,
        "--runs-per-plan", str(RUNS_PER_PLAN), "--run-offset", str(offset), "--seed", str(BASE_SEED),
        "--no-fingerprint", "--csv", str(out.resolve()),
    ]
    print(f"bench {name} …", flush=True)
    env = dict(os.environ, RAYON_NUM_THREADS="8")
    result = subprocess.run(cmd, cwd=WORKTREE, env=env, capture_output=True, text=True, encoding="utf-8")
    (out.with_name(out.name + ".stdout.txt")).write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise SystemExit(f"{name}: bench 退出码 {result.returncode}，面板作废")
    m = json.loads(meta.read_text(encoding="utf-8"))
    if not m.get("complete") or m.get("games_completed") != PLAN_COUNT * RUNS_PER_PLAN or m.get("git_commit", "")[:7] != "53d2d86":
        raise SystemExit(f"{name}: meta 不完整或提交不符，面板作废")
    print(f"  完成 {name}", flush=True)


def cmd_bench(step: str, offset: int, names: list[str]) -> None:
    """逐个模型跑主面板。"""

    for name in names:
        run_bench(name, onnx_path(name, step), bench_csv(name, step, offset), offset)


def load_grid(out: Path, offset: int) -> np.ndarray:
    """读逐局 CSV，按 (计划, 局号) 还原成 [250, 16] 分数矩阵；缺行/重复/多余即报错。"""

    meta = json.loads(out.with_name(out.name + ".meta.json").read_text(encoding="utf-8"))
    plans = meta["plan_selection"]["original_indices"]
    if len(plans) != PLAN_COUNT:
        raise SystemExit(f"{out}: 计划数 {len(plans)} != {PLAN_COUNT}")
    where = {world_seed(p, offset + r): (i, r) for i, p in enumerate(plans) for r in range(RUNS_PER_PLAN)}
    grid = np.full((PLAN_COUNT, RUNS_PER_PLAN), np.nan)
    with out.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = where.get(int(row["seed"]))
            if key is None:
                raise SystemExit(f"{out}: 出现面板外的世界种子 {row['seed']}")
            if not np.isnan(grid[key]):
                raise SystemExit(f"{out}: 世界种子重复 {row['seed']}")
            grid[key] = float(row["score"])
    if np.isnan(grid).any():
        raise SystemExit(f"{out}: 缺 {int(np.isnan(grid).sum())} 局")
    return grid


def load_scores(name: str, step: str, offset: int) -> np.ndarray:
    """按模型名读主面板分数矩阵。"""

    return load_grid(bench_csv(name, step, offset), offset)


def paired(x: np.ndarray, y: np.ndarray) -> dict:
    """组合内等权、组合间等权的配对差与组合级 t 区间。"""

    per_combo = (x - y).mean(axis=1)
    mean = float(per_combo.mean())
    se = float(per_combo.std(ddof=1) / np.sqrt(len(per_combo)))
    return {"diff": mean, "lo": mean - T_249 * se, "hi": mean + T_249 * se, "se": se}


def verdict(r: dict) -> str:
    """按预注册判读表给出实用性结论。"""

    if r["lo"] >= -150 and r["hi"] <= 150:
        practical = "实用等效"
    elif r["lo"] > 150:
        practical = "前者实用占优"
    elif r["hi"] < -150:
        practical = "后者实用占优"
    else:
        practical = "实用差异未定"
    direction = "方向明确" if r["lo"] > 0 or r["hi"] < 0 else "方向不明"
    return f"{practical}；{direction}"


def cmd_analyze(step: str, offset: int) -> None:
    """主比较 + 单成员诊断 + 次要比较，写 JSON 并打印。"""

    scores = {name: load_scores(name, step, offset) for name in model_names()}
    result = {"step": step, "offset": offset, "means": {k: float(v.mean()) for k, v in scores.items()}}
    main = paired(scores["B_ens"], scores["C_ens"])
    result["main_B_minus_C_ens"] = {**main, "verdict": verdict(main)}
    result["members_B_minus_C"] = {f"s{s}": paired(scores[f"B_s{s}"], scores[f"C_s{s}"]) for s in SEEDS}
    secondary = {"B-A": ("B", "A"), "D-C": ("D", "C"), "C-A": ("C", "A"), "D-B": ("D", "B")}
    result["secondary_ens"] = {k: paired(scores[f"{a}_ens"], scores[f"{b}_ens"]) for k, (a, b) in secondary.items()}
    inter = (scores["D_ens"] - scores["C_ens"]) - (scores["B_ens"] - scores["A_ens"])
    result["secondary_ens"]["interaction"] = paired(inter, np.zeros_like(inter))
    path = BENCH_ROOT / f"o{offset}" / step / "analysis.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("export", "bench", "analyze"))
    parser.add_argument("--step", choices=tuple(STEP_FILES), default="final")
    parser.add_argument("--offset", type=int, default=430000)
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()
    if args.command == "export":
        cmd_export(args.step)
    elif args.command == "bench":
        cmd_bench(args.step, args.offset, args.models or model_names())
    else:
        cmd_analyze(args.step, args.offset)


if __name__ == "__main__":
    main()
