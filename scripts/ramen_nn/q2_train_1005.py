"""Q2（标签质量 × 数量）四臂训练驱动：R9 gen2 同口径数据，根数 {25%,100%} × 每候选 rollout {256,1024}。

除数据与 `--max-train-samples` 外照搬 R8A（优化器参数全部显式写出，不依赖默认值）：
显式基础结构、batch 256、cosine、关早停、冻结常量、按 R8 开发组合清单划分。
同一 `--split-seed` 下 25% 子集严格嵌套于全量，且 256/1024 两档样本 id 相同 ⇒ 抽到同一批根。

资产冻结：首次运行把有序数据/标签清单、训练根 id 与 25% 子集 id 写进 `logs/q2_1005/frozen/`，
之后每次运行逐字段比对，不一致直接拒绝。

完成判定：退出码 0、`last.pt` 的 `global_step == 30000`、`run.json` 存在且划分根数符合预期。
半途中断的输出目录不会被当作完成，也不会被自动续跑或覆盖，需人工处理。有任一任务失败则非零退出。

用法（工作区根）：python scripts/ramen_nn/q2_train_1005.py [--parallel 6] [--dry-run] [--arms A B C D] [--seeds 1 2 3]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as nn_data  # noqa: E402

BATCHES = ("r9_gen2_1002", "r9_gen2_1003_extra10k", "r9_gen2_1004_extra20k")
DEV_COMBOS = Path("target/train_r8_0920/dev_validation_combos.json")
SPLIT_SEED = 20260830
MAX_STEPS = 30000
TRAIN_ROOTS = 63531
VALID_ROOTS = 6469
# 按 R8 开发组合清单划分后训练根数的 25%
QUARTER_ROOTS = 15883
ARMS = {  # 臂 -> (rollout 宽度, 训练根上限；None = 全量)
    "A": (256, QUARTER_ROOTS),
    "B": (1024, QUARTER_ROOTS),
    "C": (256, None),
    "D": (1024, None),
}
FROZEN_DIR = Path("logs/q2_1005/frozen")
COMMON = [
    "--batch-size", "256",
    "--lr", "5e-4", "--head-lr", "5e-4", "--weight-decay", "2e-5",
    "--value-loss-weight", "1.0", "--grad-clip", "1.0",
    "--max-steps", str(MAX_STEPS),
    "--lr-schedule", "cosine", "--lr-warmup-steps", "300", "--lr-final-factor", "0.02",
    "--no-early-stop",
    "--frozen-constants", "target/dagger_d_seed1/run.json",
    "--validation-combos", str(DEV_COMBOS),
    "--seed", str(SPLIT_SEED), "--split-seed", str(SPLIT_SEED),
    "--token-dim", "96", "--heads", "4", "--encoder-blocks", "2",
    "--mlp-width", "256", "--mlp-blocks", "2", "--dropout", "0.08",
    "--attention-kind", "simple",
    "--checkpoint-steps", "7500", "15000",
]


def pairs(width: int) -> list[tuple[str, str]]:
    """按固定批次与 run_state 导出顺序列出某宽度的 (数据目录, 标签目录)。"""

    out: list[tuple[str, str]] = []
    for batch in BATCHES:
        state = json.loads(Path(f"training_data/{batch}/run_state.json").read_text(encoding="utf-8"))
        for rel in state["exports"].values():
            job = Path(rel).name
            if width == 1024:
                data, labels = Path(rel), Path(f"training_data/labels_{batch}/{job}")
            else:
                data = Path(f"training_data/q2_r9g2_w{width}/{batch}/{job}")
                labels = Path(f"training_data/labels_q2_r9g2_w{width}/{batch}/{job}")
            meta = json.loads((labels / "labels.json").read_text(encoding="utf-8"))
            if meta.get("policy_accumulator") != "float64" or meta["rollouts"] != width:
                raise ValueError(f"{labels}: 标签版本或宽度不符")
            out.append((data.as_posix(), labels.as_posix()))
    return out


def split_ids(width: int) -> dict[str, list[int]]:
    """用与 train.py 相同的函数算出训练根 id、验证根 id 与 25% 子集 id（均排序）。"""

    listed = pairs(width)
    shards = nn_data.load_shards([Path(d) for d, _ in listed], [Path(l) for _, l in listed])
    combos = json.loads(DEV_COMBOS.read_text(encoding="utf-8"))["combos"]
    train, valid = nn_data.split_refs_by_combos(shards, combos)
    quarter = nn_data.subsample_train_refs(shards, train, QUARTER_ROOTS, SPLIT_SEED)

    def ids(refs: np.ndarray) -> list[int]:
        return sorted(int(shards[int(s)].index[int(i)]) for s, i in refs)

    return {"train": ids(train), "valid": ids(valid), "quarter": ids(quarter)}


def freeze_or_check() -> None:
    """首次写入冻结清单；之后逐字段比对，任何差异都拒绝运行。"""

    # 存成列表：JSON 往返后元组会变成列表，比较前统一类型
    current = {f"pairs_w{w}": [list(p) for p in pairs(w)] for w in (256, 1024)}
    id_sets = {w: split_ids(w) for w in (256, 1024)}
    if id_sets[256] != id_sets[1024]:
        raise ValueError("256 与 1024 两档的划分/子集样本 id 不一致")
    ids = id_sets[1024]
    if (len(ids["train"]), len(ids["valid"]), len(ids["quarter"])) != (TRAIN_ROOTS, VALID_ROOTS, QUARTER_ROOTS):
        raise ValueError("划分根数与预注册不符")
    if not set(ids["quarter"]) <= set(ids["train"]):
        raise ValueError("25% 子集不是训练集的子集")
    current.update(ids)
    path = FROZEN_DIR / "assets.json"
    if path.exists():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        bad = [k for k in current if frozen.get(k) != current[k]]
        if bad:
            raise ValueError(f"与冻结清单不一致的字段：{bad}")
        print("资产与冻结清单逐字段一致")
    else:
        FROZEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current, ensure_ascii=False), encoding="utf-8")
        print(f"已冻结资产清单 {path}")


def build_cmd(arm: str, seed: int) -> tuple[list[str], Path]:
    """返回某臂某种子的训练命令与输出目录。"""

    width, cap = ARMS[arm]
    out = Path(f"target/q2_1005/{arm}_seed{seed}")
    cmd = [sys.executable, "scripts/ramen_nn/train.py", *COMMON, "--init-seed", str(seed), "--output-dir", str(out)]
    if cap is not None:
        cmd += ["--max-train-samples", str(cap)]
    for data, labels in pairs(width):
        cmd += ["--data", data, "--labels", labels]
    return cmd, out


def check_done(arm: str, out: Path) -> str | None:
    """完成则返回 None，否则返回原因。"""

    last, run = out / "last.pt", out / "run.json"
    if not last.exists() or not run.exists():
        return "缺 last.pt 或 run.json"
    step = int(torch.load(last, map_location="cpu", weights_only=False)["global_step"])
    if step != MAX_STEPS:
        return f"global_step={step} != {MAX_STEPS}"
    split = json.loads(run.read_text(encoding="utf-8"))["split"]
    want = QUARTER_ROOTS if ARMS[arm][1] else TRAIN_ROOTS
    if (split["train"], split["validation"]) != (want, VALID_ROOTS):
        return f"划分根数 {split['train']}/{split['validation']} 不符"
    return None


def main() -> None:
    """冻结/核对资产后按并发池跑全部臂 × 种子。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=6)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    freeze_or_check()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs/q2_1005/train")
    log_dir.mkdir(parents=True, exist_ok=True)
    queue, blocked = [], []
    for seed in args.seeds:
        for arm in args.arms:
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
    if args.dry_run:
        for arm, seed, cmd, _ in queue:
            print(f"{arm}_seed{seed}: --data 对数 {cmd.count('--data')}")
        return

    running, failed, t0 = [], [], time.time()
    while queue or running:
        while queue and len(running) < args.parallel:
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
        print("Q2_TRAIN_FAILED", *failed, sep="\n  ", flush=True)
        sys.exit(1)
    print("Q2_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
