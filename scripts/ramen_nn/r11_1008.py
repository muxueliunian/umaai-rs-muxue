"""R11 首组训练驱动：R11 三批（新规则、R8A roll-in、每候选 256）上 {基础, 加宽} × {31, 61 轮} × 3 种子。

臂（结构全部显式，不走 ModelConfig.for_dataset 的自动档）：
  B31 / B61：token 96、MLP 256（同 R8A）；W31 / W61：token 128、MLP 384，其余结构参数相同。
  31 轮 = 41819 步、61 轮 = 82289 步（训练 345323 根 / batch 256 = 每轮 1349 步），完整 cosine，只用 last.pt。

子命令（工作区根）：
  python scripts/ramen_nn/r11_1008.py train   [--parallel 6] [--dry-run] [--arms ...]
  python scripts/ramen_nn/r11_1008.py export
  python scripts/ramen_nn/r11_1008.py bench   --what ens|members   # 选型面板 430000（含 R8A 重跑）
  python scripts/ramen_nn/r11_1008.py select                       # selection.json 只写一次
  python scripts/ramen_nn/r11_1008.py confirm                      # 确认面板 460000
  python scripts/ramen_nn/r11_1008.py all     [--parallel 6]       # 以上全部，按序、可续跑

口径见 logs/r11_1008/preregistration.md。评测用 master（f3b742e）编译的 ramen_space_bench。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as nn_data  # noqa: E402
import q2_eval_1005 as q2e  # noqa: E402

BATCHES = ("r11_gen2n256_1005", "r11b_gen2n256_1006", "r11c_gen2n256_1007")
# R11 第一批被磁盘守卫停下时单独导出的 partial（不在 run_state 里）
EXTRA = {"r11_gen2n256_1005": ["training_data/r11_gen2n256_1005/npy/r18_region_y3_s2_resource_partial_1791244464059655903"]}
DEV_COMBOS = Path("target/train_r8_0920/dev_validation_combos.json")
SPLIT_SEED = 20260830
TRAIN_ROOTS = 345323
VALID_ROOTS = 35108
STEPS = {"31": 41819, "61": 82289}
ARCH = {
    "B": ["--token-dim", "96", "--heads", "4", "--encoder-blocks", "2", "--mlp-width", "256", "--mlp-blocks", "2",
          "--dropout", "0.08", "--attention-kind", "simple"],
    "W": ["--token-dim", "128", "--heads", "4", "--encoder-blocks", "2", "--mlp-width", "384", "--mlp-blocks", "2",
          "--dropout", "0.08", "--attention-kind", "simple"],
}
# 预注册的偏好顺序：近似平手时取靠前者（先基础、再短档）
ARMS = ("B31", "B61", "W31", "W61")
SEEDS = (1, 2, 3)
COMMON = [
    "--batch-size", "256",
    "--lr", "5e-4", "--head-lr", "5e-4", "--weight-decay", "2e-5",
    "--value-loss-weight", "1.0", "--grad-clip", "1.0",
    "--lr-schedule", "cosine", "--lr-warmup-steps", "300", "--lr-final-factor", "0.02",
    "--no-early-stop",
    "--frozen-constants", "target/dagger_d_seed1/run.json",
    "--validation-combos", str(DEV_COMBOS),
    "--seed", str(SPLIT_SEED), "--split-seed", str(SPLIT_SEED),
]
TRAIN_ROOT = Path("target/r11_1008")
ONNX_ROOT = TRAIN_ROOT / "onnx"
LOG_ROOT = Path("logs/r11_1008")
BENCH_ROOT = LOG_ROOT / "bench"
BENCH_EXE = Path("target/thin_bench/release/ramen_space_bench.exe")
BENCH_COMMIT = "f3b742e"
# meta 记录的是运行时 HEAD；d664729 相对 f3b742e 未改 crates/、gamedata/、Cargo.*（只整理文档与脚本），二进制不变
BENCH_COMMITS = (BENCH_COMMIT, "d664729")
R8A_ONNX = Path("saved_models/arms/ens_R8A_g123.onnx")
R8A_MEMBERS = {f"R8A_s{s}": Path(f"saved_models/arms/R8A_seed{s}.onnx") for s in SEEDS}
SELECT_OFFSET = 430000
CONFIRM_OFFSET = 460000
TIE_MARGIN = 150.0


def pairs() -> list[tuple[str, str]]:
    """按批次与 run_state 导出顺序（+ truncated partial + EXTRA）列出 (数据目录, 标签目录)。"""

    out: list[tuple[str, str]] = []
    for batch in BATCHES:
        state = json.loads(Path(f"training_data/{batch}/run_state.json").read_text(encoding="utf-8"))
        paths = list(state.get("exports", {}).values())
        part = (state.get("truncated") or {}).get("partial")
        if part and part.get("export"):
            paths.append(part["export"])
        paths += [p for p in EXTRA.get(batch, []) if p not in paths]
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
    combos = json.loads(DEV_COMBOS.read_text(encoding="utf-8"))["combos"]
    train, valid = nn_data.split_refs_by_combos(shards, combos)

    def ids(refs: np.ndarray) -> list[int]:
        return sorted(int(shards[int(s)].index[int(i)]) for s, i in refs)

    return {"train": ids(train), "valid": ids(valid)}


def build_cmd(arm: str, seed: int) -> tuple[list[str], Path]:
    """某臂某种子的训练命令与输出目录。"""

    out = TRAIN_ROOT / f"{arm}_seed{seed}"
    cmd = [sys.executable, "scripts/ramen_nn/train.py", *COMMON, "--max-steps", str(STEPS[arm[1:]]), *ARCH[arm[0]],
           "--init-seed", str(seed), "--output-dir", out.as_posix()]
    for data, labels in pairs():
        cmd += ["--data", data, "--labels", labels]
    return cmd, out


def freeze_or_check() -> None:
    """首次写入冻结清单（数据对、划分 id、训练命令、评测口径），之后逐字段比对。"""

    ids = split_ids()
    if (len(ids["train"]), len(ids["valid"])) != (TRAIN_ROOTS, VALID_ROOTS):
        raise SystemExit(f"划分根数 {len(ids['train'])}/{len(ids['valid'])} 与预注册不符")
    current = {
        "pairs": [list(p) for p in pairs()], **ids,
        "commands": {f"{a}_seed{s}": build_cmd(a, s)[0][1:] for a in ARMS for s in SEEDS},
        "eval": {"select": SELECT_OFFSET, "confirm": CONFIRM_OFFSET, "tie_margin": TIE_MARGIN,
                 "preference": list(ARMS), "bench_commit": BENCH_COMMIT},
    }
    path = LOG_ROOT / "frozen" / "assets.json"
    if path.exists():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        bad = [k for k in current if frozen.get(k) != current[k]]
        if bad:
            raise SystemExit(f"与冻结清单不一致的字段：{bad}")
        print("资产与冻结清单逐字段一致", flush=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current, ensure_ascii=False), encoding="utf-8")
        print(f"已冻结资产清单 {path}", flush=True)


def check_done(arm: str, out: Path) -> str | None:
    """完成则返回 None，否则返回原因。"""

    last, run = out / "last.pt", out / "run.json"
    if not last.exists() or not run.exists():
        return "缺 last.pt 或 run.json"
    step = int(torch.load(last, map_location="cpu", weights_only=False)["global_step"])
    if step != STEPS[arm[1:]]:
        return f"global_step={step} != {STEPS[arm[1:]]}"
    split = json.loads(run.read_text(encoding="utf-8"))["split"]
    if (split["train"], split["validation"]) != (TRAIN_ROOTS, VALID_ROOTS):
        return f"划分根数 {split['train']}/{split['validation']} 不符"
    return None


def cmd_train(parallel: int, arms: list[str], dry_run: bool) -> None:
    """按并发池跑训练；队列顺序：基础组（长档先）→ 加宽组（长档先）。"""

    freeze_or_check()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = LOG_ROOT / "train"
    log_dir.mkdir(parents=True, exist_ok=True)
    order = [a for a in ("B61", "B31", "W61", "W31") if a in arms]
    queue, blocked = [], []
    for arm in order:
        for seed in SEEDS:
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
            # 612 对 --data/--labels 超过 Windows 命令行上限：参数写文件，经 trainjob 转交 train.py
            argfile = log_dir / f"{arm}_seed{seed}_{stamp}.args.json"
            argfile.write_text(json.dumps(cmd[2:], ensure_ascii=False), encoding="utf-8")
            launch = [sys.executable, __file__, "trainjob", "--argfile", str(argfile)]
            running.append((subprocess.Popen(launch, stdout=log, stderr=subprocess.STDOUT), log, arm, seed, out))
            print(f"{time.time() - t0:6.0f}s 启动 {arm}_seed{seed}", flush=True)
        time.sleep(10)
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
        print("R11_TRAIN_FAILED", *failed, sep="\n  ", flush=True)
        sys.exit(1)
    print("R11_TRAIN_DONE", flush=True)


def parity_rows(count: int = 512) -> np.ndarray:
    """从冻结验证集 id 中均匀取真实输入行。"""

    frozen = json.loads((LOG_ROOT / "frozen" / "assets.json").read_text(encoding="utf-8"))
    valid = np.array(frozen["valid"])
    rows = []
    for data_dir, _ in frozen["pairs"]:
        index = np.load(Path(data_dir) / "index.npy")
        hit = np.flatnonzero(np.isin(index, valid))
        if hit.size:
            x = np.load(Path(data_dir) / "x.npy", mmap_mode="r")
            rows.append(np.asarray(x[hit[:: max(1, hit.size // 4)][:4]]))
    stacked = np.concatenate(rows).astype(np.float32)
    return stacked[np.linspace(0, len(stacked) - 1, count).astype(int)]


def cmd_export() -> None:
    """导出每臂三成员与集成，真实输入对拍；超阈值即报错。已有 parity.json 则跳过。"""

    path = ONNX_ROOT / "parity.json"
    if path.exists():
        print("skip export（parity.json 已存在）")
        return
    x = parity_rows()
    report = {}
    for arm in ARMS:
        ckpts = [TRAIN_ROOT / f"{arm}_seed{s}" / "last.pt" for s in SEEDS]
        outs = [ONNX_ROOT / f"{arm}_s{s}.onnx" for s in SEEDS]
        ONNX_ROOT.mkdir(parents=True, exist_ok=True)
        report.update(q2e.export_group(ckpts, outs, ONNX_ROOT / f"{arm}_ens.onnx", x))
    bad = sorted(k for k, v in report.items() if v["real_max_abs_error"] >= q2e.PARITY_TOL)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if bad:
        raise SystemExit(f"真实输入对拍超阈值 {q2e.PARITY_TOL}：{bad}")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def onnx_of(name: str) -> Path:
    """模型名 → ONNX 路径（R8A 取已发布资产）。"""

    if name == "R8A_ens":
        return R8A_ONNX
    if name in R8A_MEMBERS:
        return R8A_MEMBERS[name]
    return ONNX_ROOT / f"{name}.onnx"


def csv_of(name: str, offset: int) -> Path:
    """模型在某面板的逐局 CSV。"""

    return BENCH_ROOT / f"o{offset}" / f"{name}.csv"


def run_bench(name: str, offset: int) -> None:
    """跑一个 250×16 面板；已完整的跳过，不完整的报错（不续跑、不覆盖）。"""

    out = csv_of(name, offset)
    meta = out.with_name(out.name + ".meta.json")
    total = q2e.PLAN_COUNT * q2e.RUNS_PER_PLAN
    if meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        if m.get("complete") and m.get("games_completed") == total:
            print(f"skip bench {name}@{offset}（已完整）", flush=True)
            return
        raise SystemExit(f"{out} 存在但不完整，需人工处理")
    if out.exists():
        raise SystemExit(f"{out} 已存在但缺 meta，需人工处理")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(BENCH_EXE.resolve()), "--trainer", "nn", "--model", str(onnx_of(name).resolve()),
        "--space-version", "gen2_v1", "--plans-file", q2e.PLANS_FILE,
        "--runs-per-plan", str(q2e.RUNS_PER_PLAN), "--run-offset", str(offset), "--seed", str(q2e.BASE_SEED),
        "--no-fingerprint", "--csv", str(out.resolve()),
    ]
    t = time.time()
    env = dict(os.environ, RAYON_NUM_THREADS="8")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8")
    out.with_name(out.name + ".stdout.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise SystemExit(f"{name}: bench 退出码 {result.returncode}，面板作废")
    m = json.loads(meta.read_text(encoding="utf-8"))
    if not m.get("complete") or m.get("games_completed") != total or m.get("git_commit", "")[:7] not in BENCH_COMMITS:
        raise SystemExit(f"{name}: meta 不完整或提交不符，面板作废")
    print(f"bench {name}@{offset} 完成 {time.time() - t:.0f}s", flush=True)


def friend_rate(name: str, offset: int) -> float:
    """友人 5 次出行全部完成的局占比。"""

    with csv_of(name, offset).open(encoding="utf-8") as fh:
        vals = [int(r["friend_out_all"]) for r in csv.DictReader(fh)]
    return float(np.mean(vals))


def contrast(x: np.ndarray, y: np.ndarray) -> dict:
    """配对差 + 预注册判读。"""

    r = q2e.paired(x, y)
    return {**r, "verdict": q2e.verdict(r)}


def ens_names() -> list[str]:
    """选型面板的集成（候选 + R8A）。"""

    return [f"{a}_ens" for a in ARMS] + ["R8A_ens"]


def member_names() -> list[str]:
    """选型面板的单成员（候选 + R8A），只做描述。"""

    return [f"{a}_s{s}" for a in ARMS for s in SEEDS] + list(R8A_MEMBERS)


def cmd_bench(what: str) -> None:
    """选型面板：集成或单成员。"""

    for name in ens_names() if what == "ens" else member_names():
        run_bench(name, SELECT_OFFSET)


def choose(means: dict[str, float]) -> str:
    """预注册选型：取最高集成均分；与最高差 < TIE_MARGIN 的候选中按偏好顺序取第一个。"""

    best = max(means[f"{a}_ens"] for a in ARMS)
    return next(a for a in ARMS if best - means[f"{a}_ens"] < TIE_MARGIN)


def cmd_select() -> None:
    """430000 上描述性比较 + 选型；selection.json 只写一次。"""

    off = SELECT_OFFSET
    g = {n: q2e.load_grid(csv_of(n, off), off) for n in ens_names()}
    means = {k: float(v.mean()) for k, v in g.items()}
    chosen = choose(means)
    result = {
        "offset": off, "means": means,
        "friend_complete": {n: friend_rate(n, off) for n in ens_names()},
        "vs_R8A": {a: contrast(g[f"{a}_ens"], g["R8A_ens"]) for a in ARMS},
        "W_minus_B_same_steps": {e: contrast(g[f"W{e}_ens"], g[f"B{e}_ens"]) for e in STEPS},
        "61_minus_31": {w: contrast(g[f"{w}61_ens"], g[f"{w}31_ens"]) for w in "BW"},
        "selection": {"chosen": chosen, "rule": f"集成均分最高者；与最高差 < {TIE_MARGIN} 的按 {list(ARMS)} 顺序取第一个"},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    path = LOG_ROOT / "selection.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))["selection"]["chosen"]
        if old != chosen:
            raise SystemExit(f"selection.json 已冻结为 {old}，与本次 {chosen} 不一致")
        print(f"selection.json 已存在（{old}），不覆盖")
        return
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_confirm() -> None:
    """460000：选中集成 − R8A（主）；选中加宽时加同步数基础臂（次）。"""

    chosen = json.loads((LOG_ROOT / "selection.json").read_text(encoding="utf-8"))["selection"]["chosen"]
    off = CONFIRM_OFFSET
    names = [f"{chosen}_ens", "R8A_ens"] + ([f"B{chosen[1:]}_ens"] if chosen[0] == "W" else [])
    for n in names:
        run_bench(n, off)
    g = {n: q2e.load_grid(csv_of(n, off), off) for n in names}
    result = {
        "offset": off, "chosen": chosen, "means": {k: float(v.mean()) for k, v in g.items()},
        "friend_complete": {n: friend_rate(n, off) for n in names},
        "main": {"name": f"{chosen}-R8A", **contrast(g[f"{chosen}_ens"], g["R8A_ens"])},
    }
    if chosen[0] == "W":
        result["secondary"] = {f"{chosen}-B{chosen[1:]}": contrast(g[f"{chosen}_ens"], g[f"B{chosen[1:]}_ens"])}
    (BENCH_ROOT / f"o{off}" / "confirm.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def cmd_members_report() -> None:
    """单成员描述：每个成员 − 同种子 R8A 成员，以及友人完成率。"""

    off = SELECT_OFFSET
    g = {n: q2e.load_grid(csv_of(n, off), off) for n in member_names()}
    result = {
        "offset": off, "means": {k: float(v.mean()) for k, v in g.items()},
        "friend_complete": {n: friend_rate(n, off) for n in member_names()},
        "member_vs_R8A_same_seed": {f"{a}_s{s}": q2e.paired(g[f"{a}_s{s}"], g[f"R8A_s{s}"]) for a in ARMS for s in SEEDS},
    }
    (BENCH_ROOT / f"o{off}" / "members.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def cmd_trainjob(argfile: Path) -> None:
    """从参数文件读 train.py 参数并在本进程内运行 train.py。"""

    import runpy

    sys.argv = ["train.py", *json.loads(argfile.read_text(encoding="utf-8"))]
    runpy.run_path("scripts/ramen_nn/train.py", run_name="__main__")


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "export", "bench", "select", "confirm", "members", "all", "trainjob"))
    parser.add_argument("--argfile", type=Path)
    parser.add_argument("--parallel", type=int, default=6)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--what", choices=("ens", "members"), default="ens")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "trainjob":
        cmd_trainjob(args.argfile)
        return
    if args.command in ("train", "all"):
        cmd_train(args.parallel, args.arms, args.dry_run)
        if args.dry_run or args.command == "train":
            return
    if args.command in ("export", "all"):
        cmd_export()
    if args.command == "bench":
        cmd_bench(args.what)
    if args.command == "all":
        cmd_bench("ens")
    if args.command in ("select", "all"):
        cmd_select()
    if args.command in ("confirm", "all"):
        cmd_confirm()
    if args.command == "all":
        cmd_bench("members")
    if args.command in ("members", "all"):
        cmd_members_report()
    if args.command == "all":
        print("R11_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
