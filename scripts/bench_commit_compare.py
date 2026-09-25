#!/usr/bin/env python3
"""跨 commit CPU 耗时对比：在 base/head 两版代码上跑同一量具并配对比较。

量具自动选择：
- probe 模式（默认）：两版都有 `perf_probe` 时，对每个固定 Train 根（默认 32/60 回合）
  交替跑 `--rounds` 轮整根搜索，比较每轮耗时与中位数；
- bench 模式（回退）：旧 commit 没有 `perf_probe` 时，改用 `bench_base` 的手写策略
  整局耗时（同 seed 逐局配对，运气噪声被消除），可选 MCTS 小预算整局。

比较纪律（与 perf_profiling.md 一致）：
- 同机、同 gamedata（脚本本身不换数据版本）、同编译配置（默认无 target-cpu 覆盖，
  两版在同一工作环境构建）、同搜索参数；
- 逐轮在两版间交替执行（round 奇偶调换先后），抵消环境漂移；
- 结果只作同机配对对照，不作跨机器承诺。

用法:
  python3 scripts/bench_commit_compare.py --base 04c739c --head current
  python3 scripts/bench_commit_compare.py --base 86de303 --head 04c739c --rounds 3 --search-n 8192
  python3 scripts/bench_commit_compare.py --base 86de303 --head current --bench-mcts --whole-runs 100
  python3 scripts/bench_commit_compare.py --base 91e4b03 --head current --force-bench --bench-runs 300

`--head current` 表示用当前工作树（含未提交改动），其余情况自动创建独立 worktree。
`--force-bench` 在两版都有 perf_probe 时强制走 bench 模式（bench_base 手写整局遍历全部
player_builds、同 seed 配对，同时输出耗时与评分两份配对表；单决策根的 probe 只作诊断）。
产物写到 logs/commit-compare/<base>__<head>_<时间戳>/（manifest.json + rounds.csv + report.txt）。
"""
import argparse
import concurrent.futures
import csv
import datetime
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
TARGET_DIR = REPO / "target" / "commit-compare"
PROBE_SRC = "crates/umasim/src/bin/perf_probe.rs"


def git(*args, cwd=None):
    """在仓库根执行 git，返回 stdout 文本。"""
    return subprocess.run(
        ["git", "-C", str(cwd or REPO), *args],
        check=True,
        capture_output=True,
        text=True
    ).stdout.strip()


def resolve_sha(ref):
    """把 ref 解析成完整 commit sha。"""
    if ref == "current":
        # 当前工作树的基线约定：以 HEAD 的 sha 记录（内容可能含未提交改动）
        return git("rev-parse", "HEAD")
    return git("rev-parse", "--verify", f"{ref}^{{commit}}")


def has_probe_in_commit(ref, sha):
    """该目标是否包含 perf_probe 源文件。

    `current` 目标（当前工作树）直接查工作树文件；其余目标用 git cat-file 在
    commit 树里探测（无需检出）。
    """
    if ref == "current":
        return (REPO / PROBE_SRC).exists()
    rc = subprocess.run(
        ["git", "-C", str(REPO), "cat-file", "-e", f"{sha}:{PROBE_SRC}"],
        capture_output=True
    )
    return rc.returncode == 0


def exe_suffix():
    return ".exe" if os.name == "nt" else ""


class Version:
    """一个待测版本：worktree（或当前工作树）+ 独立构建目录。"""

    def __init__(self, name, ref, sha, worktree_dir, build_dir, is_worktree):
        self.name = name
        self.ref = ref
        self.sha = sha
        self.worktree_dir = worktree_dir
        self.build_dir = build_dir
        # is_worktree=False 表示直接用主工作树（--head current），清理时绝不能删它
        self.is_worktree = is_worktree

    def bin_path(self, bin_name):
        return self.build_dir / "release" / (bin_name + exe_suffix())

    def build(self, bin_name):
        """在独立 CARGO_TARGET_DIR 下构建指定 bin（Release + locked）。"""
        cmd = [
            "cargo", "build", "--release", "--locked",
            "-p", "umasim", "--no-default-features", "--bin", bin_name
        ]
        env = dict(os.environ)
        env["CARGO_TARGET_DIR"] = str(self.build_dir)
        print(f"[build] {self.name} ({bin_name}): {cmd}")
        subprocess.run(cmd, cwd=str(self.worktree_dir), env=env, check=True)


def make_worktree(sha, tag):
    """创建 detached worktree；返回 (worktree_dir, build_dir)。"""
    wt = TARGET_DIR / f"wt-{tag}"
    build = TARGET_DIR / f"build-{tag}"
    if wt.exists():
        git("worktree", "remove", "--force", str(wt))
    git("worktree", "add", "--detach", str(wt), sha)
    wt.mkdir(parents=True, exist_ok=True)
    build.mkdir(parents=True, exist_ok=True)
    return wt, build


def remove_worktree(version):
    """清理 worktree（保留构建产物，方便复跑）。"""
    if version.is_worktree:
        git("worktree", "remove", "--force", str(version.worktree_dir))


def run_probe(version, turn, round_no, args, label):
    """在指定版本上跑一次固定根搜索，返回解析出的 ROOT 记录。"""
    cmd = [
        str(version.bin_path("perf_probe")), "root",
        "--turn", str(turn),
        "--rounds", "1",
        "--seed", str(args.seed),
        "--search-seed", str(args.search_seed),
        "--build", args.build,
        "--label", label,
    ]
    if args.deck:
        cmd += ["--deck", args.deck]
    if args.search_n:
        cmd += ["--search-n", str(args.search_n)]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    except subprocess.CalledProcessError as e:
        sys.exit(f"[probe] {label} turn={turn} 运行失败:\n{e.stderr}")
    for line in out.splitlines():
        if line.startswith("ROOT "):
            rec = parse_kv(line[5:])
            return {
                "version": label,
                "turn": int(rec["turn"]),
                # 探针内部总是 round=1（脚本每次只调一轮）；这里用外层交替轮次标记
                "round": round_no,
                "candidates": int(rec["candidates"]),
                "searched": int(rec["searched"]),
                "best": int(rec["best"]),
                "best_mean": float(rec["best_mean"]),
                # 根局面真实累计评分（calc_score；旧版探针可能没有 → None）
                "root_score": float(rec["root_score"]) if rec.get("root_score") else None,
                "elapsed_ms": float(rec["elapsed_ms"]),
            }
    sys.exit(f"[probe] {label} turn={turn} 输出缺少 ROOT 行:\n{out}")


def parse_kv(s):
    """把 `k=v k=v` 行解析成 dict。"""
    return dict(piece.split("=", 1) for piece in s.split())


def run_whole(version, runs, label, args):
    """跑手写整局耗时批次，返回 WHOLE 记录。"""
    cmd = [
        str(version.bin_path("perf_probe")), "whole",
        "--runs", str(runs),
        "--seed", str(args.seed),
        "--build", args.build,
        "--label", label,
    ]
    if args.deck:
        cmd += ["--deck", args.deck]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    except subprocess.CalledProcessError as e:
        sys.exit(f"[whole] {label} 运行失败:\n{e.stderr}")
    for line in out.splitlines():
        if line.startswith("WHOLE "):
            rec = parse_kv(line[6:])
            return {
                "version": label,
                "runs": int(rec["runs"]),
                "mean_ms": float(rec["mean_ms"]),
                "median_ms": float(rec["median_ms"]),
                "min_ms": float(rec["min_ms"]),
                "max_ms": float(rec["max_ms"]),
                "std_ms": float(rec["std_ms"]),
                "score_mean": float(rec["score_mean"]),
                "score_min": float(rec["score_min"]),
                "score_max": float(rec["score_max"]),
            }
    sys.exit(f"[whole] {label} 输出缺少 WHOLE 行:\n{out}")


def run_bench_base(version, trainer, runs, out_dir, args, extra=None):
    """在版本上跑 bench_base（返回结果 CSV 路径）。"""
    cmd = [
        str(version.bin_path("bench_base")),
        "--trainer", trainer,
        "--runs", str(runs),
        "--seed", str(args.seed),
        "--out", str(out_dir),
    ]
    if extra:
        cmd += extra
    subprocess.run(cmd, cwd=str(version.worktree_dir), check=True)
    return out_dir / "bench_base_results.csv"


def load_bench_csv(path):
    """读取 bench_base_results.csv → [{build, seed, elapsed_ms, ...}]。"""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def median(vals):
    return statistics.median(vals)


def fmt_pct(delta, base):
    if base == 0:
        return "n/a"
    return f"{delta / base * 100:+.1f}%"


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def make_logger(out_dir):
    """逐步执行日志：打印+落盘 logs/commit-compare/<dir>/run.log。

    后台任务跑在独立进程命名空间时无法从外部看进程，run.log 是判断
    「长任务 vs 卡死」的唯一外部信号（每步都即时 flush）。
    """
    log_path = out_dir / "run.log"

    def log(msg):
        line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    return log


def main():
    ap = argparse.ArgumentParser(description="跨 commit CPU 耗时对比（perf_probe / bench_base）")
    ap.add_argument("--base", required=True, help="基线 ref（commit / 分支 / tag）")
    ap.add_argument("--head", default="current", help="对照 ref（默认 current=当前工作树）")
    ap.add_argument("--rounds", type=int, default=3, help="每固定根轮数（默认 3，轮间交替版本）")
    ap.add_argument("--roots", default="32,60", help="固定 Train 根回合，逗号分隔（默认 32,60）")
    ap.add_argument("--search-n", type=int, default=None, help="每候选 rollout 数（probe 模式；默认取各版 game 配置）")
    ap.add_argument("--seed", type=int, default=61444, help="育成推进基础种子（默认 61444）")
    ap.add_argument("--search-seed", type=int, default=61444, help="搜索种子（默认 61444）")
    ap.add_argument("--build", default="speed", help="preset build 名（默认 speed）")
    ap.add_argument("--deck", default=None, help="自定义卡组 idrank 串，覆盖 --build")
    ap.add_argument("--whole-runs", type=int, default=None, help="附加手写整局耗时对比（局数，默认不跑）")
    ap.add_argument("--bench-mcts", action="store_true", help="bench 模式附加 MCTS 小预算整局（search-n 64）")
    ap.add_argument("--bench-mcts-runs", type=int, default=5, help="bench 模式 MCTS 局数（默认 5）")
    ap.add_argument("--force-bench", action="store_true",
                    help="两版都有 perf_probe 时也强制走 bench 模式（bench_base 手写整局遍历全部 player_builds 配对，含耗时+评分）")
    ap.add_argument("--bench-runs", type=int, default=100, help="bench 模式手写整局局数（默认 100；单局约 1ms，可放心加大）")
    ap.add_argument("--build-jobs", type=int, default=1,
                   help="并行构建版本数（默认 1=串行；并行构建可能因共享 ~/.cargo 缓存锁竞争卡死）")
    ap.add_argument("--probe-only", action="store_true", help="base 缺 perf_probe 时报错而非回退 bench 模式")
    ap.add_argument("--keep", action="store_true", help="保留 worktree（默认结束即删除）")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不构建不测量")
    args = ap.parse_args()

    base_sha = resolve_sha(args.base)
    head_sha = resolve_sha(args.head)
    base_short, head_short = base_sha[:7], head_sha[:7]
    probe_ok_base, probe_ok_head = has_probe_in_commit(args.base, base_sha), has_probe_in_commit(args.head, head_sha)
    mode = "bench" if args.force_bench else ("probe" if (probe_ok_base and probe_ok_head) else "bench")
    if args.force_bench and args.probe_only:
        sys.exit("--force-bench 与 --probe-only 互斥，请只保留一个")
    if mode == "bench" and args.probe_only:
        sys.exit(
            f"base={args.base} 或 head={args.head} 的 commit 不含 {PROBE_SRC}（探测: "
            f"base={probe_ok_base} head={probe_ok_head}）；--probe-only 已指定，拒绝回退 bench 模式"
        )

    out_root = REPO / "logs" / "commit-compare"
    out_dir = out_root / f"{base_short}__{head_short}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(out_dir)

    versions = []
    # base 永远用干净 worktree（防止脏工作树污染基线）
    wt_b, build_b = make_worktree(base_sha, f"base-{base_short}")
    versions.append(Version("base", args.base, base_sha, wt_b, build_b, is_worktree=True))
    if args.head == "current":
        versions.append(Version("head", "current", head_sha, REPO, TARGET_DIR / f"build-head-{head_short}",
                                is_worktree=False))
    else:
        wt_h, build_h = make_worktree(head_sha, f"head-{head_short}")
        versions.append(Version("head", args.head, head_sha, wt_h, build_h, is_worktree=True))
    labels = {v.name: f"{v.name}-{v.sha[:7]}" for v in versions}

    print(f"===== 跨 commit 耗时对比 =====")
    print(f"base: {args.base} = {base_sha}")
    print(f"head: {args.head} = {head_sha}" + ("（当前工作树，含未提交改动）" if args.head == "current" else ""))
    print(f"mode: {mode}（probe 需两版都有 perf_probe）")
    print(f"out : {out_dir}")
    # 口径守卫：两 commit 的 gamedata 内容不同 → 模拟数值不同，耗时对比结论受限
    # （git diff --quiet 用退出码表达：0=无差异 1=有差异；不能用 stdout 判定）
    diff_rc = subprocess.run(
        ["git", "-C", str(REPO), "diff", "--quiet", base_sha, head_sha, "--", "gamedata"],
        capture_output=True
    ).returncode
    data_drift = diff_rc == 1
    if data_drift:
        print("⚠ 注意: base/head 的 gamedata 内容不同（数据版本漂移）——模拟数值口径不一致，")
        print("   耗时对比只能反映代码+数据的合计变化；需先对齐 gamedata 再做纯代码对比。")
    if args.search_n is None and mode == "probe":
        print("提示: 未指定 --search-n，各版本使用自己的 game_config.toml 搜索参数"
              "（配置漂移会计入对比；严格可比请显式传 --search-n）。")
    if args.dry_run:
        print("\n[--dry-run] 计划如上，跳过构建与测量。")
        return

    # 1) 构建两版（默认串行：并行 cargo 可能因共享 ~/.cargo 缓存锁竞争卡死）
    bin_name = "perf_probe" if mode == "probe" else "bench_base"
    log(f"[build] 开始构建 {bin_name}（jobs={args.build_jobs}）")
    if args.build_jobs <= 1:
        for v in versions:
            log(f"[build] {v.name} 开始（{v.worktree_dir}）")
            try:
                v.build(bin_name)
            except subprocess.CalledProcessError:
                sys.exit(f"[build] {v.name} 构建失败（{bin_name}）——检查工具链与 Cargo.lock 兼容性")
            log(f"[build] {v.name} 完成")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.build_jobs) as ex:
            futures = {ex.submit(v.build, bin_name): v for v in versions}
            for fut in concurrent.futures.as_completed(futures):
                v = futures[fut]
                try:
                    fut.result()
                except subprocess.CalledProcessError:
                    sys.exit(f"[build] {v.name} 构建失败（{bin_name}）——检查工具链与 Cargo.lock 兼容性")
                log(f"[build] {v.name} 完成")
    log("[build] 全部构建完成")

    manifest = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "machine": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mode": mode,
        "base": {"ref": args.base, "sha": base_sha, "probe_present": probe_ok_base},
        "head": {"ref": args.head, "sha": head_sha, "probe_present": probe_ok_head},
        "params": {
            "rounds": args.rounds, "roots": args.roots, "search_n": args.search_n,
            "seed": args.seed, "search_seed": args.search_seed, "build": args.build,
            "deck": args.deck, "whole_runs": args.whole_runs,
            "bench_mcts": args.bench_mcts, "bench_mcts_runs": args.bench_mcts_runs,
            "force_bench": args.force_bench, "bench_runs": args.bench_runs,
        },
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    report = []
    def emit(s=""):
        print(s)
        report.append(s)

    if mode == "probe":
        log("[probe] 开始固定根交替测量")
        roots = [int(t.strip()) for t in args.roots.split(",") if t.strip()]
        records = []
        for rnd in range(1, args.rounds + 1):
            order = versions if rnd % 2 == 1 else list(reversed(versions))
            for v in order:
                for turn in roots:
                    rec = run_probe(v, turn, rnd, args, labels[v.name])
                    print(
                        f"  [r{rnd}] {rec['version']:>10} turn={turn:>2} "
                        f"searched={rec['searched']:>6} best={rec['best']:>2} "
                        f"elapsed_ms={rec['elapsed_ms']:.3f}"
                    )
                    records.append(rec)
        write_csv(
            out_dir / "rounds.csv",
            ["version", "turn", "round", "candidates", "searched", "best", "best_mean", "root_score", "elapsed_ms"],
            [[r["version"], r["turn"], r["round"], r["candidates"], r["searched"],
              r["best"], f"{r['best_mean']:.3f}",
              f"{r['root_score']:.0f}" if r["root_score"] is not None else "",
              f"{r['elapsed_ms']:.3f}"] for r in records]
        )

        emit("")
        emit("===== 固定根搜索耗时（rounds 中位数）=====")
        for turn in roots:
            by = {}
            for v in versions:
                times = [r["elapsed_ms"] for r in records
                         if r["version"] == labels[v.name] and r["turn"] == turn]
                by[v.name] = times
            if not by["base"] or not by["head"]:
                continue
            b_m, h_m = median(by["base"]), median(by["head"])
            delta = h_m - b_m
            emit(
                f"{'turn':<14} {'base':>10} {'head':>10} {'Δms':>9} {'%':>8}   "
                f"base_rounds={['%.1f' % x for x in by['base']]} head_rounds={['%.1f' % x for x in by['head']]}"
            )
            emit(f"{f'turn {turn}':<14} {b_m:>10.3f} {h_m:>10.3f} {delta:>+9.3f} {fmt_pct(delta, b_m):>8}")
            # 两版评分区别：根局面 calc_score（同种子下只随策略/评分公式变化）
            bs_score = [r["root_score"] for r in records
                        if r["version"] == labels["base"] and r["turn"] == turn and r["root_score"] is not None]
            hs_score = [r["root_score"] for r in records
                        if r["version"] == labels["head"] and r["turn"] == turn and r["root_score"] is not None]
            if bs_score and hs_score:
                b_s, h_s = median(bs_score), median(hs_score)
                note = "（两版到达的根局面评分一致 → 计时差纯来自代码性能）" if b_s == h_s \
                    else "（评分不同 → 到达根局面的策略/打分已变，耗时差含局面漂移分量）"
                emit(f"{'评分':<14} {b_s:>10.0f} {h_s:>10.0f} {h_s - b_s:>+9.0f} {'':>8}  {note}")
            # 搜索工作量一致性检查：同 turn 跨版本 searched 必须一致，否则耗时不可直接对比
            base_searched = {r["round"]: r["searched"] for r in records
                             if r["version"] == labels["base"] and r["turn"] == turn}
            head_searched = {r["round"]: r["searched"] for r in records
                             if r["version"] == labels["head"] and r["turn"] == turn}
            if base_searched != head_searched:
                emit(f"  ⚠ 注意: turn {turn} 两版搜索工作量不一致（base={base_searched} head={head_searched}），"
                     f"耗时差异可能来自局面/策略变化而非代码性能")

        if args.whole_runs and args.whole_runs > 0:
            emit("")
            emit("===== 手写策略整局耗时（whole）=====")
            ws = {v.name: run_whole(v, args.whole_runs, labels[v.name], args) for v in versions}
            emit(f"{'':<6} {'mean_ms':>9} {'median_ms':>9} {'min_ms':>8} {'max_ms':>8}   score_mean")
            for v in versions:
                w = ws[v.name]
                emit(f"{w['version']:<6} {w['mean_ms']:>9.3f} {w['median_ms']:>9.3f} "
                     f"{w['min_ms']:>8.3f} {w['max_ms']:>8.3f}   {w['score_mean']:.1f}")
            d = ws["head"]["mean_ms"] - ws["base"]["mean_ms"]
            emit(f"head-base mean_ms Δ = {d:+.3f} ({fmt_pct(d, ws['base']['mean_ms'])})")
    else:
        log(f"[bench] 模式：开始 bench_base 整局测量（runs={args.bench_runs}，遍历全部 player_builds）")
        # ===== bench 模式：手写整局耗时 + 评分（同 seed 逐局配对，全部 builds）=====
        hw_base = run_bench_base(versions[0], "handwritten", args.bench_runs, out_dir / "bench-base-hw", args)
        hw_head = run_bench_base(versions[1], "handwritten", args.bench_runs, out_dir / "bench-head-hw", args)
        base_rows = load_bench_csv(hw_base)
        head_rows = load_bench_csv(hw_head)
        by_seed = {}
        for r in base_rows:
            by_seed.setdefault(r["build"], {})[r["seed"]] = (float(r["elapsed_ms"]), float(r["score"]))
        pairs = []
        for r in head_rows:
            b = by_seed.get(r["build"], {}).get(r["seed"])
            if b is not None:
                pairs.append((r["build"], r["seed"], b[0], b[1], float(r["elapsed_ms"]), float(r["score"])))
        emit("")
        emit(f"===== 手写整局耗时（bench_base handwritten，runs={args.bench_runs}，同 seed 配对）=====")
        emit(f"{'build':<14} {'base_med':>10} {'head_med':>10} {'Δms':>9} {'%':>8}   n")
        for build_name in sorted({p[0] for p in pairs}):
            sub = [p for p in pairs if p[0] == build_name]
            bs = median(p[2] for p in sub)
            hs = median(p[4] for p in sub)
            delta = hs - bs
            emit(f"{build_name:<14} {bs:>10.3f} {hs:>10.3f} {delta:>+9.3f} {fmt_pct(delta, bs):>8}   {len(sub)}")
        emit("")
        emit(f"===== 手写整局评分（bench_base handwritten，runs={args.bench_runs}，同 seed 配对）=====")
        emit(f"{'build':<14} {'base_med':>10} {'head_med':>10} {'Δ分':>9} {'配对Δ':>9} {'%':>7}   n")
        for build_name in sorted({p[0] for p in pairs}):
            sub = [p for p in pairs if p[0] == build_name]
            bs = median(p[3] for p in sub)
            hs = median(p[5] for p in sub)
            delta = hs - bs
            paired = median(p[5] - p[3] for p in sub)
            emit(f"{build_name:<14} {bs:>10.0f} {hs:>10.0f} {delta:>+9.0f} "
                 f"{paired:>+9.0f} {fmt_pct(paired, bs):>7}   {len(sub)}")
        write_csv(
            out_dir / "bench_paired.csv",
            ["build", "seed", "base_elapsed_ms", "head_elapsed_ms"],
            [[p[0], p[1], f"{p[2]:.3f}", f"{p[4]:.3f}"] for p in pairs]
        )
        write_csv(
            out_dir / "bench_score_paired.csv",
            ["build", "seed", "base_score", "head_score"],
            [[p[0], p[1], f"{p[3]:.0f}", f"{p[5]:.0f}"] for p in pairs]
        )
        if args.bench_mcts:
            mc_base = run_bench_base(
                versions[0], "mcts", args.bench_mcts_runs, out_dir / "bench-base-mcts", args,
                extra=["--search-n", "64", "--search-ucb", "false"]
            )
            mc_head = run_bench_base(
                versions[1], "mcts", args.bench_mcts_runs, out_dir / "bench-head-mcts", args,
                extra=["--search-n", "64", "--search-ucb", "false"]
            )
            emit("")
            emit("===== MCTS 小预算整局耗时（bench_base mcts search-n=64）=====")
            mcb = load_bench_csv(mc_base)
            mch = load_bench_csv(mc_head)
            mby = {}
            for r in mcb:
                mby.setdefault(r["build"], {})[r["seed"]] = float(r["elapsed_ms"])
            for r in mch:
                b = mby.get(r["build"], {}).get(r["seed"])
                if b is not None:
                    emit(f"{r['build']:<14} seed={r['seed']}: {b:>10.3f} -> {float(r['elapsed_ms']):>10.3f} ms")

    with open(out_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"\n报告: {out_dir / 'report.txt'}")

    if not args.keep:
        for v in versions:
            remove_worktree(v)
    else:
        print("--keep: worktree 保留在 target/commit-compare/（可手动删除）")


if __name__ == "__main__":
    main()