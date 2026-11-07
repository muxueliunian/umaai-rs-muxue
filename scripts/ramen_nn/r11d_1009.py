"""R11 第二组训练驱动：R11 + R11d（新规则、R8A roll-in、每候选 256）上基础结构 {短档, 长档} × 3 种子。

臂（结构同 R11 首组 B，全部显式）：
  S：恰好 82289 步（与首组 B61 同步数，单看多 18.8 万根的作用）；
  L：61 轮（每轮 = ceil(训练根 / 256) 步，冻结时按实际划分算出）。
完整 cosine，只用 last.pt。

子命令（工作区根）：
  python scripts/ramen_nn/r11d_1009.py train   [--parallel 6] [--dry-run]
  python scripts/ramen_nn/r11d_1009.py export
  python scripts/ramen_nn/r11d_1009.py bench                         # 确认面板 460100 + 描述面板 430000 + 成员
  python scripts/ramen_nn/r11d_1009.py report
  python scripts/ramen_nn/r11d_1009.py all     [--parallel 6]        # 以上全部，按序、可续跑

口径见 logs/r11_1009/preregistration.md。评测用 master（f3b742e）编译的 ramen_space_bench。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as nn_data  # noqa: E402
import q2_eval_1005 as q2e  # noqa: E402
import r11_1008 as r1  # noqa: E402

BATCHES = ("r11_gen2n256_1005", "r11b_gen2n256_1006", "r11c_gen2n256_1007", "r11d_gen2n256_1007")
ARMS = ("S", "L")
SEEDS = r1.SEEDS
SHORT_STEPS = 82289
EPOCHS = 61
TRAIN_ROOT = Path("target/r11d_1009")
ONNX_ROOT = TRAIN_ROOT / "onnx"
LOG_ROOT = Path("logs/r11_1009")
BENCH_ROOT = LOG_ROOT / "bench"
FROZEN = LOG_ROOT / "frozen" / "assets.json"
CONFIRM_OFFSET = 460100
DESC_OFFSET = 430000
# 首组的路径函数原件（run_bench 会临时替换 r1 上的同名函数，这里必须用原件以免递归）
R1_ONNX_OF, R1_CSV_OF = r1.onnx_of, r1.csv_of


def pairs() -> list[tuple[str, str]]:
    """按批次与 run_state 导出顺序（+ partial + 首组 EXTRA）列出 (数据目录, 标签目录)。"""

    out: list[tuple[str, str]] = []
    for batch in BATCHES:
        state = json.loads(Path(f"training_data/{batch}/run_state.json").read_text(encoding="utf-8"))
        paths = list(state.get("exports", {}).values())
        part = (state.get("truncated") or {}).get("partial")
        if part and part.get("export"):
            paths.append(part["export"])
        paths += [p for p in r1.EXTRA.get(batch, []) if p not in paths]
        for rel in paths:
            data = Path(rel.replace("\\", "/"))
            labels = Path(f"training_data/labels_{batch}/{data.name}")
            meta = json.loads((labels / "labels.json").read_text(encoding="utf-8"))
            if meta.get("policy_accumulator") != "float64" or meta["rollouts"] != 256:
                raise ValueError(f"{labels}: 标签版本或宽度不符")
            out.append((data.as_posix(), labels.as_posix()))
    return out


def split_ids() -> dict[str, list[int]]:
    """用与 train.py 相同的函数算训练/验证根 id（排序）。"""

    listed = pairs()
    shards = nn_data.load_shards([Path(d) for d, _ in listed], [Path(l) for _, l in listed])
    combos = json.loads(r1.DEV_COMBOS.read_text(encoding="utf-8"))["combos"]
    train, valid = nn_data.split_refs_by_combos(shards, combos)

    def ids(refs: np.ndarray) -> list[int]:
        return sorted(int(shards[int(s)].index[int(i)]) for s, i in refs)

    return {"train": ids(train), "valid": ids(valid)}


def steps_of(arm: str, train_roots: int) -> int:
    """臂 → 训练步数。"""

    return SHORT_STEPS if arm == "S" else EPOCHS * math.ceil(train_roots / 256)


def build_cmd(arm: str, seed: int, steps: int) -> tuple[list[str], Path]:
    """某臂某种子的训练命令与输出目录。"""

    out = TRAIN_ROOT / f"{arm}_seed{seed}"
    cmd = [sys.executable, "scripts/ramen_nn/train.py", *r1.COMMON, "--max-steps", str(steps), *r1.ARCH["B"],
           "--init-seed", str(seed), "--output-dir", out.as_posix()]
    for data, labels in pairs():
        cmd += ["--data", data, "--labels", labels]
    return cmd, out


def frozen() -> dict:
    """首次写冻结清单（数据对、划分 id、步数、训练命令、评测口径），之后逐字段比对并返回。"""

    ids = split_ids()
    n_train, n_valid = len(ids["train"]), len(ids["valid"])
    steps = {a: steps_of(a, n_train) for a in ARMS}
    current = {
        "pairs": [list(p) for p in pairs()], **ids, "split": [n_train, n_valid], "steps": steps,
        "commands": {f"{a}_seed{s}": build_cmd(a, s, steps[a])[0][1:] for a in ARMS for s in SEEDS},
        "eval": {"confirm": CONFIRM_OFFSET, "describe": DESC_OFFSET, "bench_commit": r1.BENCH_COMMIT},
    }
    if FROZEN.exists():
        old = json.loads(FROZEN.read_text(encoding="utf-8"))
        bad = [k for k in current if old.get(k) != current[k]]
        if bad:
            raise SystemExit(f"与冻结清单不一致的字段：{bad}")
        print(f"资产与冻结清单逐字段一致：训练 {n_train} / 验证 {n_valid}，步数 {steps}", flush=True)
    else:
        FROZEN.parent.mkdir(parents=True, exist_ok=True)
        FROZEN.write_text(json.dumps(current, ensure_ascii=False), encoding="utf-8")
        print(f"已冻结 {FROZEN}：训练 {n_train} / 验证 {n_valid}，步数 {steps}", flush=True)
    return current


def check_done(out: Path, steps: int, split: list[int]) -> str | None:
    """完成则返回 None，否则返回原因。"""

    last, run = out / "last.pt", out / "run.json"
    if not last.exists() or not run.exists():
        return "缺 last.pt 或 run.json"
    step = int(torch.load(last, map_location="cpu", weights_only=False)["global_step"])
    if step != steps:
        return f"global_step={step} != {steps}"
    got = json.loads(run.read_text(encoding="utf-8"))["split"]
    if [got["train"], got["validation"]] != split:
        return f"划分根数 {got['train']}/{got['validation']} 不符"
    return None


def cmd_train(parallel: int, dry_run: bool) -> None:
    """按并发池跑训练；长档先启动。"""

    fz = frozen()
    if dry_run:
        for a in ARMS:
            print(f"{a}: max-steps {fz['steps'][a]}，--data 对数 {len(fz['pairs'])}")
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = LOG_ROOT / "train"
    log_dir.mkdir(parents=True, exist_ok=True)
    queue, blocked = [], []
    for arm in ("L", "S"):
        for seed in SEEDS:
            cmd, out = build_cmd(arm, seed, fz["steps"][arm])
            if out.exists():
                reason = check_done(out, fz["steps"][arm], fz["split"])
                (blocked.append(f"{arm}_seed{seed}: {reason}") if reason else print(f"skip {arm}_seed{seed}（已完成）"))
                continue
            queue.append((arm, seed, cmd, out))
    if blocked:
        print("❗以下输出目录存在但未完成，需人工处理（重跑请先移走目录）：", *blocked, sep="\n  ")
        sys.exit(2)
    running, failed, t0 = [], [], time.time()
    while queue or running:
        while queue and len(running) < parallel:
            arm, seed, cmd, out = queue.pop(0)
            log = open(log_dir / f"{arm}_seed{seed}_{stamp}.log", "w", encoding="utf-8")
            argfile = log_dir / f"{arm}_seed{seed}_{stamp}.args.json"
            argfile.write_text(json.dumps(cmd[2:], ensure_ascii=False), encoding="utf-8")
            launch = [sys.executable, r1.__file__, "trainjob", "--argfile", str(argfile)]
            running.append((subprocess.Popen(launch, stdout=log, stderr=subprocess.STDOUT), log, arm, seed, out))
            print(f"{time.time() - t0:6.0f}s 启动 {arm}_seed{seed}", flush=True)
        time.sleep(10)
        for item in list(running):
            proc, log, arm, seed, out = item
            if proc.poll() is None:
                continue
            log.close()
            running.remove(item)
            reason = f"rc={proc.returncode}" if proc.returncode else check_done(out, fz["steps"][arm], fz["split"])
            if reason:
                failed.append(f"{arm}_seed{seed}: {reason}")
            print(f"{time.time() - t0:6.0f}s 结束 {arm}_seed{seed} {'OK' if not reason else '失败 ' + reason}", flush=True)
    if failed:
        print("R11D_TRAIN_FAILED", *failed, sep="\n  ", flush=True)
        sys.exit(1)
    print("R11D_TRAIN_DONE", flush=True)


def parity_rows(count: int = 512) -> np.ndarray:
    """从冻结验证集 id 中均匀取真实输入行。"""

    fz = json.loads(FROZEN.read_text(encoding="utf-8"))
    valid = np.array(fz["valid"])
    rows = []
    for data_dir, _ in fz["pairs"]:
        index = np.load(Path(data_dir) / "index.npy")
        hit = np.flatnonzero(np.isin(index, valid))
        if hit.size:
            x = np.load(Path(data_dir) / "x.npy", mmap_mode="r")
            rows.append(np.asarray(x[hit[:: max(1, hit.size // 4)][:4]]))
    stacked = np.concatenate(rows).astype(np.float32)
    return stacked[np.linspace(0, len(stacked) - 1, count).astype(int)]


def cmd_export() -> None:
    """导出每臂三成员与集成，真实输入对拍；已有 parity.json 则跳过。"""

    path = ONNX_ROOT / "parity.json"
    if path.exists():
        print("skip export（parity.json 已存在）")
        return
    x = parity_rows()
    report = {}
    ONNX_ROOT.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        ckpts = [TRAIN_ROOT / f"{arm}_seed{s}" / "last.pt" for s in SEEDS]
        outs = [ONNX_ROOT / f"{arm}_s{s}.onnx" for s in SEEDS]
        report.update(q2e.export_group(ckpts, outs, ONNX_ROOT / f"{arm}_ens.onnx", x))
    bad = sorted(k for k, v in report.items() if v["real_max_abs_error"] >= q2e.PARITY_TOL)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if bad:
        raise SystemExit(f"真实输入对拍超阈值 {q2e.PARITY_TOL}：{bad}")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def onnx_of(name: str) -> Path:
    """模型名 → ONNX 路径（B61/R8A 取首组与已发布资产）。"""

    if name.startswith("B61") or name.startswith("R8A"):
        return R1_ONNX_OF(name)
    return ONNX_ROOT / f"{name}.onnx"


def csv_of(name: str, offset: int) -> Path:
    """逐局 CSV：430000 上的 B61/R8A 复用首组结果，其余写本组目录。"""

    if offset == DESC_OFFSET and (name.startswith("B61") or name.startswith("R8A")):
        return R1_CSV_OF(name, offset)
    return BENCH_ROOT / f"o{offset}" / f"{name}.csv"


def run_bench(name: str, offset: int) -> None:
    """借首组的 run_bench（同二进制、同面板口径），只换模型与输出路径。"""

    saved = (r1.onnx_of, r1.csv_of)
    r1.onnx_of, r1.csv_of = onnx_of, csv_of
    try:
        r1.run_bench(name, offset)
    finally:
        r1.onnx_of, r1.csv_of = saved


def ens() -> list[str]:
    """集成名。"""

    return ["S_ens", "L_ens", "B61_ens", "R8A_ens"]


def members() -> list[str]:
    """单成员名（只做描述）。"""

    return [f"{a}_s{s}" for a in ("S", "L", "B61") for s in SEEDS]


def cmd_bench() -> None:
    """确认面板 460100（集成 + 成员），描述面板 430000（新集成，旧两条复用）。"""

    for n in ens() + members():
        run_bench(n, CONFIRM_OFFSET)
    for n in ens():
        run_bench(n, DESC_OFFSET)


def friend_fails(name: str, offset: int) -> list[dict]:
    """友人 5 次出行未走完的局（逐年出行次数）。"""

    with csv_of(name, offset).open(encoding="utf-8") as fh:
        return [{"seed": r["seed"], "build": r["build"], "score": int(r["score"]),
                 "out": [int(r[f"friend_out_y{y}"]) for y in (1, 2, 3)],
                 "turns": [int(r[f"friend_turns_y{y}"]) for y in (1, 2, 3)]}
                for r in csv.DictReader(fh) if r["friend_out_all"] != "1"]


def contrast(g: dict, a: str, b: str) -> dict:
    """配对差 + 预注册判读。"""

    r = q2e.paired(g[a], g[b])
    return {"name": f"{a}-{b}", **r, "verdict": q2e.verdict(r)}


def cmd_report() -> None:
    """主比较 S−B61、次要 L−S、标尺 −R8A；成员与友人失败局只做描述。"""

    result = {}
    for off in (CONFIRM_OFFSET, DESC_OFFSET):
        g = {n: q2e.load_grid(csv_of(n, off), off) for n in ens()}
        result[f"o{off}"] = {
            "means": {k: float(v.mean()) for k, v in g.items()},
            "main": contrast(g, "S_ens", "B61_ens"),
            "secondary": contrast(g, "L_ens", "S_ens"),
            "ruler": [contrast(g, "S_ens", "R8A_ens"), contrast(g, "L_ens", "R8A_ens"), contrast(g, "L_ens", "B61_ens")],
            "friend_fail": {n: friend_fails(n, off) for n in ens()},
        }
    gm = {n: q2e.load_grid(csv_of(n, CONFIRM_OFFSET), CONFIRM_OFFSET) for n in members()}
    result["members_o460100"] = {
        "means": {k: float(v.mean()) for k, v in gm.items()},
        "vs_B61_same_seed": {f"{a}_s{s}": q2e.paired(gm[f"{a}_s{s}"], gm[f"B61_s{s}"]) for a in ARMS for s in SEEDS},
    }
    path = LOG_ROOT / "result.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "members_o460100"}, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "export", "bench", "report", "all"))
    parser.add_argument("--parallel", type=int, default=6)
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
        print("R11D_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
