"""全量混合臂 M：R8A 训练语料（旧规则，labels64 float64 重打标签）+ R11/b/c/d，基础结构 × 122976 步 × 3 种子。

步数与 R11 第二组 L 档相同（同步数比较 M − L）。冻结清单单独存放，不动第二组的冻结清单。
子命令（工作区根）：
  python scripts/ramen_nn/r11m_1009.py train [--parallel 3] [--dry-run]
  python scripts/ramen_nn/r11m_1009.py all   [--parallel 3]   # 训练 → 导出 → bench → 报告
口径见 logs/r11_1009/preregistration.md 的「追加：全量混合臂」。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as nn_data  # noqa: E402
import q2_eval_1005 as q2e  # noqa: E402
import r11_1008 as r1  # noqa: E402
import r11d_1009 as r2  # noqa: E402

R8A_RUN = Path("target/arm_R8A_seed1/run.json")
LABELS64 = Path("training_data/labels64")
STEPS = 122976
SEEDS = r1.SEEDS
TRAIN_ROOT = Path("target/r11m_1009")
ONNX_ROOT = TRAIN_ROOT / "onnx"
LOG_ROOT = r2.LOG_ROOT
FROZEN = LOG_ROOT / "frozen" / "assets_m.json"


def old_pairs() -> list[tuple[str, str]]:
    """R8A 用过的 (数据目录, labels64 标签目录)，核对 float64 与样本数。"""

    split = json.loads(R8A_RUN.read_text(encoding="utf-8"))["split"]
    out = []
    for d, l in zip(split["data_dirs"], split["label_dirs"]):
        d, l = Path(d.replace("\\", "/")), Path(l.replace("\\", "/"))
        new = LABELS64 / l.parent.name / l.name
        meta = json.loads((new / "labels.json").read_text(encoding="utf-8"))
        if meta.get("policy_accumulator") != "float64":
            raise ValueError(f"{new}: 不是 float64")
        out.append((d.as_posix(), new.as_posix()))
    return out


def combo_fields() -> dict[str, str]:
    """R8A 训练时外挂的完整组合字段（缺 combo_fields.npy 的 10 个旧目录），按 run.json 原样沿用。"""

    split = json.loads(R8A_RUN.read_text(encoding="utf-8"))["split"]
    return {Path(d.replace("\\", "/")).as_posix(): Path(f.replace("\\", "/")).as_posix()
            for d, f in zip(split["data_dirs"], split["combo_fields_sources"]) if f}


def pairs() -> list[tuple[str, str]]:
    """旧语料在前、R11 系列在后。"""

    return old_pairs() + r2.pairs()


def split_ids() -> dict[str, list[int]]:
    """与 train.py 相同的组合划分。"""

    listed = pairs()
    extra = {Path(d): Path(f) for d, f in combo_fields().items()}
    shards = nn_data.load_shards([Path(d) for d, _ in listed], [Path(l) for _, l in listed], extra)
    combos = json.loads(r1.DEV_COMBOS.read_text(encoding="utf-8"))["combos"]
    train, valid = nn_data.split_refs_by_combos(shards, combos)

    def ids(refs: np.ndarray) -> list[int]:
        return sorted(int(shards[int(s)].index[int(i)]) for s, i in refs)

    return {"train": ids(train), "valid": ids(valid)}


def build_cmd(seed: int) -> tuple[list[str], Path]:
    """某种子的训练命令与输出目录。"""

    out = TRAIN_ROOT / f"M_seed{seed}"
    cmd = [sys.executable, "scripts/ramen_nn/train.py", *r1.COMMON, "--max-steps", str(STEPS), *r1.ARCH["B"],
           "--init-seed", str(seed), "--output-dir", out.as_posix()]
    for d, l in pairs():
        cmd += ["--data", d, "--labels", l]
    for d, f in combo_fields().items():
        cmd += ["--combo-fields", d, f]
    return cmd, out


def frozen() -> dict:
    """首次写冻结清单，之后逐字段比对。"""

    ids = split_ids()
    current = {"pairs": [list(p) for p in pairs()], **ids, "split": [len(ids["train"]), len(ids["valid"])],
               "steps": STEPS, "commands": {f"M_seed{s}": build_cmd(s)[0][1:] for s in SEEDS}}
    if FROZEN.exists():
        old = json.loads(FROZEN.read_text(encoding="utf-8"))
        bad = [k for k in current if old.get(k) != current[k]]
        if bad:
            raise SystemExit(f"与冻结清单不一致的字段：{bad}")
    else:
        FROZEN.parent.mkdir(parents=True, exist_ok=True)
        FROZEN.write_text(json.dumps(current, ensure_ascii=False), encoding="utf-8")
    print(f"冻结清单 {FROZEN}：训练 {current['split'][0]} / 验证 {current['split'][1]}，{len(current['pairs'])} 对，步数 {STEPS}", flush=True)
    return current


def cmd_train(parallel: int, dry_run: bool) -> None:
    """3 个种子并行训练；已完成的跳过，残缺目录报错。"""

    fz = frozen()
    if dry_run:
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = LOG_ROOT / "train"
    log_dir.mkdir(parents=True, exist_ok=True)
    queue, blocked = [], []
    for seed in SEEDS:
        cmd, out = build_cmd(seed)
        if out.exists():
            reason = r2.check_done(out, STEPS, fz["split"])
            (blocked.append(f"M_seed{seed}: {reason}") if reason else print(f"skip M_seed{seed}（已完成）"))
            continue
        queue.append((seed, cmd, out))
    if blocked:
        print("❗以下输出目录存在但未完成，需人工处理：", *blocked, sep="\n  ")
        sys.exit(2)
    running, failed, t0 = [], [], time.time()
    while queue or running:
        while queue and len(running) < parallel:
            seed, cmd, out = queue.pop(0)
            log = open(log_dir / f"M_seed{seed}_{stamp}.log", "w", encoding="utf-8")
            argfile = log_dir / f"M_seed{seed}_{stamp}.args.json"
            argfile.write_text(json.dumps(cmd[2:], ensure_ascii=False), encoding="utf-8")
            launch = [sys.executable, r1.__file__, "trainjob", "--argfile", str(argfile)]
            running.append((subprocess.Popen(launch, stdout=log, stderr=subprocess.STDOUT), log, seed, out))
            print(f"{time.time() - t0:6.0f}s 启动 M_seed{seed}", flush=True)
        time.sleep(10)
        for item in list(running):
            proc, log, seed, out = item
            if proc.poll() is None:
                continue
            log.close()
            running.remove(item)
            reason = f"rc={proc.returncode}" if proc.returncode else r2.check_done(out, STEPS, fz["split"])
            if reason:
                failed.append(f"M_seed{seed}: {reason}")
            print(f"{time.time() - t0:6.0f}s 结束 M_seed{seed} {'OK' if not reason else '失败 ' + reason}", flush=True)
    if failed:
        print("R11M_TRAIN_FAILED", *failed, sep="\n  ", flush=True)
        sys.exit(1)
    print("R11M_TRAIN_DONE", flush=True)


def cmd_export() -> None:
    """导出三成员与集成，真实输入（第二组冻结验证集）对拍。"""

    path = ONNX_ROOT / "parity.json"
    if path.exists():
        print("skip export（parity.json 已存在）")
        return
    ONNX_ROOT.mkdir(parents=True, exist_ok=True)
    ckpts = [TRAIN_ROOT / f"M_seed{s}" / "last.pt" for s in SEEDS]
    outs = [ONNX_ROOT / f"M_s{s}.onnx" for s in SEEDS]
    report = q2e.export_group(ckpts, outs, ONNX_ROOT / "M_ens.onnx", r2.parity_rows())
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    bad = sorted(k for k, v in report.items() if v["real_max_abs_error"] >= q2e.PARITY_TOL)
    if bad:
        raise SystemExit(f"真实输入对拍超阈值 {q2e.PARITY_TOL}：{bad}")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def onnx_of(name: str) -> Path:
    """M 系列取本组，其余交给第二组。"""

    return ONNX_ROOT / f"{name}.onnx" if name.startswith("M_") else r2.onnx_of(name)


def run_bench(name: str, offset: int) -> None:
    """同二进制、同面板口径；CSV 与第二组放在同一目录。"""

    saved = (r1.onnx_of, r1.csv_of)
    r1.onnx_of, r1.csv_of = onnx_of, r2.csv_of
    try:
        r1.run_bench(name, offset)
    finally:
        r1.onnx_of, r1.csv_of = saved


def cmd_bench() -> None:
    """460100：M 集成 + 成员；430000：M 集成。"""

    for n in ["M_ens"] + [f"M_s{s}" for s in SEEDS]:
        run_bench(n, r2.CONFIRM_OFFSET)
    run_bench("M_ens", r2.DESC_OFFSET)


def cmd_report() -> None:
    """主比较 M − L（同步数），次要 M − S；标尺 M − R8A、M − B61；成员与友人失败局描述。"""

    names = ["M_ens", "L_ens", "S_ens", "B61_ens", "R8A_ens"]
    result = {}
    for off in (r2.CONFIRM_OFFSET, r2.DESC_OFFSET):
        g = {n: q2e.load_grid(r2.csv_of(n, off), off) for n in names}
        result[f"o{off}"] = {
            "means": {k: float(v.mean()) for k, v in g.items()},
            "main": r2.contrast(g, "M_ens", "L_ens"),
            "secondary": r2.contrast(g, "M_ens", "S_ens"),
            "ruler": [r2.contrast(g, "M_ens", "R8A_ens"), r2.contrast(g, "M_ens", "B61_ens")],
            "friend_fail_M": r2.friend_fails("M_ens", off),
        }
    off = r2.CONFIRM_OFFSET
    gm = {f"{a}_s{s}": q2e.load_grid(r2.csv_of(f"{a}_s{s}", off), off) for a in ("M", "L") for s in SEEDS}
    result["members_o460100"] = {f"M_s{s}-L_s{s}": q2e.paired(gm[f"M_s{s}"], gm[f"L_s{s}"]) for s in SEEDS}
    (LOG_ROOT / "result_m.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "export", "bench", "report", "all"))
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command in ("train", "all"):
        cmd_train(args.parallel, args.dry_run)
        if args.dry_run or args.command == "train":
            return
    if args.command in ("export", "all"):
        cmd_export()
    if args.command in ("bench", "all"):
        cmd_bench()
    if args.command in ("report", "all"):
        cmd_report()
    if args.command == "all":
        print("R11M_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
