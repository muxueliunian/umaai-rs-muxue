"""Q1（训练日程）驱动：在 Q2 的 C 臂数据上，从随机初始化跑完整 cosine 到 7719 / 15189 步。

两个步数正好是 Q2 C 臂中间保存点的实际步数（每轮 249 步的整数倍），因此
「S 臂 vs C 臂同步数保存点」只差学习率日程：数据、标签、结构、batch 顺序（由 init_seed 决定）都相同。

子命令（工作区根）：
  python scripts/ramen_nn/q1_1005.py train   [--parallel 6] [--dry-run]
  python scripts/ramen_nn/q1_1005.py export
  python scripts/ramen_nn/q1_1005.py bench     # 选型面板 430000：S 臂成员 + 集成
  python scripts/ramen_nn/q1_1005.py select    # 写 selection.json（只写一次）
  python scripts/ramen_nn/q1_1005.py confirm   # 确认面板 450000：选中 S 集成 vs C_7719 / C_30000 集成

口径见 logs/q1_1005/preregistration.md。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q2_eval_1005 as q2e  # noqa: E402
import q2_train_1005 as q2t  # noqa: E402

ARMS = {"S7719": 7719, "S15189": 15189}
SEEDS = (1, 2, 3)
TRAIN_ROOT = Path("target/q1_1005")
ONNX_ROOT = TRAIN_ROOT / "onnx"
LOG_ROOT = Path("logs/q1_1005")
BENCH_ROOT = LOG_ROOT / "bench"
SELECT_OFFSET = 430000
CONFIRM_OFFSET = 450000
# 选型时两档集成均分差小于此值则取步数更少的一档
TIE_MARGIN = 50.0
# Q2 C 臂的参照模型：名字 -> (Q2 的 step 键, 预期 global_step)
C_REFS = {"C7719": ("7500", 7719), "C15189": ("15000", 15189), "C30000": ("final", 30000)}


def build_cmd(arm: str, seed: int) -> tuple[list[str], Path]:
    """Q2 的公共参数去掉保存点、换掉总步数，数据固定为 256 档全量。"""

    common = list(q2t.COMMON)
    i = common.index("--checkpoint-steps")
    del common[i : i + 3]
    common[common.index("--max-steps") + 1] = str(ARMS[arm])
    out = TRAIN_ROOT / f"{arm}_seed{seed}"
    cmd = [sys.executable, "scripts/ramen_nn/train.py", *common, "--init-seed", str(seed), "--output-dir", out.as_posix()]
    for data, labels in q2t.pairs(256):
        cmd += ["--data", data, "--labels", labels]
    return cmd, out


def check_done(arm: str, out: Path) -> str | None:
    """完成则返回 None，否则返回原因。"""

    last, run = out / "last.pt", out / "run.json"
    if not last.exists() or not run.exists():
        return "缺 last.pt 或 run.json"
    step = int(torch.load(last, map_location="cpu", weights_only=False)["global_step"])
    if step != ARMS[arm]:
        return f"global_step={step} != {ARMS[arm]}"
    split = json.loads(run.read_text(encoding="utf-8"))["split"]
    if (split["train"], split["validation"]) != (q2t.TRAIN_ROOTS, q2t.VALID_ROOTS):
        return f"划分根数 {split['train']}/{split['validation']} 不符"
    return None


def check_refs() -> None:
    """核对 Q2 C 臂参照 checkpoint 的实际步数与预注册一致。"""

    for _, (step_key, want) in C_REFS.items():
        for s in SEEDS:
            ckpt = q2e.TRAIN_ROOT / f"C_seed{s}" / q2e.STEP_FILES[step_key]
            got = int(torch.load(ckpt, map_location="cpu", weights_only=False)["global_step"])
            if got != want:
                raise SystemExit(f"{ckpt}: global_step={got} != {want}")


def freeze_plan() -> None:
    """数据资产沿用 Q2 冻结清单；训练命令首次写入 frozen/plan.json，之后逐字比对。"""

    q2t.freeze_or_check()
    check_refs()
    plan = {f"{a}_seed{s}": build_cmd(a, s)[0][1:] for a in ARMS for s in SEEDS}
    plan["tie_margin"] = TIE_MARGIN
    plan["offsets"] = {"select": SELECT_OFFSET, "confirm": CONFIRM_OFFSET}
    path = LOG_ROOT / "frozen" / "plan.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != plan:
            raise SystemExit(f"训练计划与冻结的 {path} 不一致")
        print("训练计划与冻结清单一致")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"已冻结训练计划 {path}")


def cmd_train(parallel: int, dry_run: bool) -> None:
    """冻结/核对后按并发池跑 2 臂 × 3 种子；有失败则非零退出。"""

    freeze_plan()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = LOG_ROOT / "train"
    log_dir.mkdir(parents=True, exist_ok=True)
    queue, blocked = [], []
    for seed in SEEDS:
        for arm in ARMS:
            cmd, out = build_cmd(arm, seed)
            if out.exists():
                reason = check_done(arm, out)
                if reason is None:
                    print(f"skip {arm}_seed{seed}（已完成）")
                else:
                    blocked.append(f"{arm}_seed{seed}: {reason}")
                continue
            queue.append((arm, seed, cmd, out))
    if blocked:
        print("❗以下输出目录存在但未完成，需人工处理（重跑请先移走目录）：", *blocked, sep="\n  ")
        sys.exit(2)
    if dry_run:
        for arm, seed, cmd, _ in queue:
            print(f"{arm}_seed{seed}: max-steps {cmd[cmd.index('--max-steps') + 1]}，--data 对数 {cmd.count('--data')}")
        return

    running, failed, t0 = [], [], time.time()
    while queue or running:
        while queue and len(running) < parallel:
            arm, seed, cmd, out = queue.pop(0)
            log = open(log_dir / f"{arm}_seed{seed}_{stamp}.log", "w", encoding="utf-8")
            running.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log, arm, seed, out))
            print(f"{time.time() - t0:6.0f}s 启动 {arm}_seed{seed}", flush=True)
        time.sleep(5)
        for item in list(running):
            proc, log, arm, seed, out = item
            if proc.poll() is None:
                continue
            log.close()
            running.remove(item)
            reason = f"rc={proc.returncode}" if proc.returncode else check_done(arm, out)
            if reason:
                failed.append(f"{arm}_seed{seed}: {reason}")
            print(f"{time.time() - t0:6.0f}s 结束 {arm}_seed{seed} {'OK' if not reason else '失败 ' + reason}", flush=True)
    if failed:
        print("Q1_TRAIN_FAILED", *failed, sep="\n  ", flush=True)
        sys.exit(1)
    print("Q1_TRAIN_DONE", flush=True)


def cmd_export() -> None:
    """导出 S 臂成员与集成，真实输入对拍；超阈值即报错。"""

    x = q2e.parity_rows()
    report = {}
    for arm in ARMS:
        ckpts = [TRAIN_ROOT / f"{arm}_seed{s}" / "last.pt" for s in SEEDS]
        outs = [ONNX_ROOT / f"{arm}_s{s}.onnx" for s in SEEDS]
        report.update(q2e.export_group(ckpts, outs, ONNX_ROOT / f"{arm}_ens.onnx", x))
    (ONNX_ROOT / "parity.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    bad = sorted(k for k, v in report.items() if v["real_max_abs_error"] >= q2e.PARITY_TOL)
    if bad:
        raise SystemExit(f"真实输入对拍超阈值 {q2e.PARITY_TOL}：{bad}")


def s_names() -> list[str]:
    """S 臂全部模型名（集成在前）。"""

    return [f"{a}_ens" for a in ARMS] + [f"{a}_s{s}" for a in ARMS for s in SEEDS]


def s_csv(name: str, offset: int) -> Path:
    """S 臂模型的逐局 CSV 路径。"""

    return BENCH_ROOT / f"o{offset}" / f"{name}.csv"


def c_ref(name: str, member: str = "ens") -> tuple[Path, str]:
    """Q2 C 臂参照模型的 (ONNX, Q2 的 step 键)。"""

    step_key = C_REFS[name][0]
    return q2e.onnx_path(f"C_{member}", step_key), step_key


def cmd_bench() -> None:
    """选型面板 430000：S 臂 2 个集成 + 6 个成员。"""

    for name in s_names():
        q2e.run_bench(name, ONNX_ROOT / f"{name}.onnx", s_csv(name, SELECT_OFFSET), SELECT_OFFSET)


def contrast(x, y) -> dict:
    """配对差 + 判读。"""

    r = q2e.paired(x, y)
    return {**r, "verdict": q2e.verdict(r)}


def cmd_select() -> None:
    """430000 上的描述性比较 + 按预注册规则选型；selection.json 只写一次。"""

    off = SELECT_OFFSET
    s = {n: q2e.load_grid(s_csv(n, off), off) for n in s_names()}
    c = {}
    for ref, (step_key, _) in C_REFS.items():
        c[f"{ref}_ens"] = q2e.load_scores("C_ens", step_key, off)
        for k in SEEDS:
            c[f"{ref}_s{k}"] = q2e.load_scores(f"C_s{k}", step_key, off)
    means = {k: float(v.mean()) for k, v in {**s, **c}.items()}
    result = {"offset": off, "means": means, "contrasts_ens": {
        "S7719-C7719": contrast(s["S7719_ens"], c["C7719_ens"]),
        "S15189-C15189": contrast(s["S15189_ens"], c["C15189_ens"]),
        "S7719-C30000": contrast(s["S7719_ens"], c["C30000_ens"]),
        "S15189-C30000": contrast(s["S15189_ens"], c["C30000_ens"]),
        "S15189-S7719": contrast(s["S15189_ens"], s["S7719_ens"]),
    }, "members_same_seed": {
        f"{arm}-C{arm[1:]}_s{k}": q2e.paired(s[f"{arm}_s{k}"], c[f"C{arm[1:]}_s{k}"]) for arm in ARMS for k in SEEDS
    }}
    short, long = means["S7719_ens"], means["S15189_ens"]
    chosen = "S7719" if short >= long or long - short < TIE_MARGIN else "S15189"
    result["selection"] = {"chosen": chosen, "rule": f"集成均分较高者；差 < {TIE_MARGIN} 取步数更少的 S7719"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    path = LOG_ROOT / "selection.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))["selection"]["chosen"]
        if old != chosen:
            raise SystemExit(f"selection.json 已冻结为 {old}，与本次 {chosen} 不一致")
        print(f"selection.json 已存在（{old}），不覆盖")
        return
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_confirm() -> None:
    """确认面板 450000：选中 S 集成 vs C7719 集成（主），vs C30000 集成（次）。"""

    chosen = json.loads((LOG_ROOT / "selection.json").read_text(encoding="utf-8"))["selection"]["chosen"]
    off = CONFIRM_OFFSET
    sel = f"{chosen}_ens"
    q2e.run_bench(sel, ONNX_ROOT / f"{sel}.onnx", s_csv(sel, off), off)
    grids = {sel: q2e.load_grid(s_csv(sel, off), off)}
    for ref in ("C7719", "C30000"):
        onnx, _ = c_ref(ref)
        out = s_csv(f"{ref}_ens", off)
        q2e.run_bench(f"{ref}_ens", onnx, out, off)
        grids[f"{ref}_ens"] = q2e.load_grid(out, off)
    result = {
        "offset": off, "chosen": chosen, "means": {k: float(v.mean()) for k, v in grids.items()},
        "main": {"name": f"{chosen}-C7719", **contrast(grids[sel], grids["C7719_ens"])},
        "secondary": {
            f"{chosen}-C30000": contrast(grids[sel], grids["C30000_ens"]),
            "C7719-C30000": contrast(grids["C7719_ens"], grids["C30000_ens"]),
        },
    }
    (BENCH_ROOT / f"o{off}" / "confirm.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "export", "bench", "select", "confirm"))
    parser.add_argument("--parallel", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    {"train": lambda: cmd_train(args.parallel, args.dry_run), "export": cmd_export, "bench": cmd_bench,
     "select": cmd_select, "confirm": cmd_confirm}[args.command]()


if __name__ == "__main__":
    main()
