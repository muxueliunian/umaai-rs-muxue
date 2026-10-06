"""Q4 探针：冻结 R8A 主干，比较 policy 头重训与候选 Q 解码器（离线）。

预注册见 ``logs/q4_1006/preregistration.md``。三个子命令依次运行：

- ``features``：用三个 R8A 成员对 R9 gen2 七万根算 trunk 特征并缓存，同时缓存候选表；
- ``train --arm {L,A,N} --seed k``：只训解码器，3000 步；
- ``eval``：在验证集上一次性计算全部指标与按组合聚类的配对 bootstrap。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_shards  # noqa: E402
from model import POLICY_DIM, model_from_checkpoint  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs" / "q4_1006"
CACHE = OUT / "cache"
ASSETS = ROOT / "logs" / "q2_1005" / "frozen" / "assets.json"
SEEDS = (1, 2, 3)
ARMS = ("L", "A", "N")
MAX_CAND = 120
REGION_STAGE = 4
STAGE_NAMES = {0: "RamenSelect", 1: "SpecialSelect", 2: "Train", 3: "SuperRamenSelect", 4: "RegionSelect"}

# §4 冻结配方
PLAN = {
    "optimizer": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_roots": 256,
    "steps": 3000,
    "warmup": 100,
    "final_lr_factor": 0.02,
    "q_scale": 1000.0,
    "slot_embed_dim": 64,
    "n_hidden": 256,
    "bootstrap_draws": 10000,
    "bootstrap_seed": 20261006,
    "pair_gap": 100.0,
}


def checkpoint_path(seed: int) -> Path:
    """R8A 成员 checkpoint（与导出 ONNX 的同一个）。"""

    return ROOT / "target" / f"arm_R8A_seed{seed}" / "step_060000.pt"


# ---------------------------------------------------------------- features


def cmd_features(device: torch.device) -> None:
    """缓存候选表与三个成员的 trunk 特征 / 原 policy logits。"""

    assets = json.loads(ASSETS.read_text(encoding="utf-8"))
    pairs = assets["pairs_w1024"]
    shards = load_shards([ROOT / d for d, _ in pairs], [ROOT / lab for _, lab in pairs])
    train_ids = np.asarray(assets["train"], dtype=np.int64)
    valid_ids = np.asarray(assets["valid"], dtype=np.int64)
    if len(train_ids) != 63531 or len(valid_ids) != 6469:
        raise ValueError("冻结划分根数不是 63531 / 6469")

    where: dict[int, tuple[int, int]] = {}
    for s_idx, shard in enumerate(shards):
        for local, rid in enumerate(np.asarray(shard.index)):
            where[int(rid)] = (s_idx, local)
    order = np.concatenate([train_ids, valid_ids])
    n = len(order)
    x = np.zeros((n, 754), dtype=np.float32)
    stage = np.zeros(n, dtype=np.int8)
    legal = np.zeros((n, POLICY_DIM), dtype=bool)
    ptarget = np.zeros((n, POLICY_DIM), dtype=np.float32)
    slots = np.full((n, MAX_CAND, 3), -1, dtype=np.int16)
    cmean = np.zeros((n, MAX_CAND), dtype=np.float64)
    cmask = np.zeros((n, MAX_CAND), dtype=bool)
    combo = np.zeros((n, 7), dtype=np.int64)
    for row, rid in enumerate(order):
        s_idx, local = where[int(rid)]
        sh = shards[s_idx]
        b, e = int(sh.cand_ptr[local]), int(sh.cand_ptr[local + 1])
        c = e - b
        x[row] = sh.x[local]
        stage[row] = sh.stage[local]
        legal[row] = np.asarray(sh.legal_mask[local], dtype=bool)
        ptarget[row] = sh.policy_target[local]
        slots[row, :c] = sh.cand_slots[b:e]
        cmean[row, :c] = sh.cand_mean[b:e]
        cmask[row, :c] = True
        combo[row] = sh.combo_fields[local]
    np.savez(
        CACHE / "roots.npz",
        ids=order,
        n_train=len(train_ids),
        stage=stage,
        legal=legal,
        ptarget=ptarget,
        slots=slots,
        cmean=cmean,
        cmask=cmask,
        combo=combo,
    )
    print(f"候选表已缓存：{n} 根，阶段计数 {np.bincount(stage, minlength=5).tolist()}")

    xt = torch.from_numpy(x)
    for seed in SEEDS:
        ckpt = torch.load(checkpoint_path(seed), map_location="cpu", weights_only=False)
        model = model_from_checkpoint(ckpt, device).eval()
        captured: list[Tensor] = []
        hook = model.output.register_forward_hook(lambda _m, inp, _o: captured.append(inp[0].detach()))
        hidden, logits, max_err = [], [], 0.0
        with torch.inference_mode():
            for begin in range(0, n, 2048):
                out = model(xt[begin : begin + 2048].to(device))
                h = captured.pop()
                # 自检：hook 抓到的 h 过 output 层必须复现模型 policy 输出
                max_err = max(max_err, float((model.output(h)[:, :POLICY_DIM] - out[:, :POLICY_DIM]).abs().max()))
                hidden.append(h.cpu().numpy())
                logits.append(out[:, :POLICY_DIM].cpu().numpy())
        hook.remove()
        if max_err > 1e-4:
            raise RuntimeError(f"seed {seed} 的 hook 特征复现误差 {max_err}")
        head = model.output.state_dict()
        np.savez(
            CACHE / f"member_{seed}.npz",
            h=np.concatenate(hidden),
            p0=np.concatenate(logits),
            head_w=head["weight"][:POLICY_DIM].cpu().numpy(),
            head_b=head["bias"][:POLICY_DIM].cpu().numpy(),
        )
        print(f"成员 {seed}：特征 {n}×{hidden[0].shape[1]}，复现误差 {max_err:.2e}")


# ---------------------------------------------------------------- decoders


class SlotDecoder(nn.Module):
    """L / A 臂：从 h 线性产生每格一个分数。"""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, POLICY_DIM)

    def forward(self, h: Tensor) -> Tensor:
        return self.linear(h)


class CandidateDecoder(nn.Module):
    """N 臂：候选 token = 所占格嵌入之和，与 h 拼接后过两层 MLP 得到 Q。"""

    def __init__(self, width: int, embed: int, hidden: int) -> None:
        super().__init__()
        # 最后一行是填充格（-1 映射到这里），固定为零
        self.slot_embed = nn.Embedding(POLICY_DIM + 1, embed, padding_idx=POLICY_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(width + embed, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: Tensor, slots: Tensor) -> Tensor:
        idx = torch.where(slots >= 0, slots, torch.full_like(slots, POLICY_DIM))
        e = self.slot_embed(idx).sum(dim=2)  # [B,C,E]
        hc = h[:, None, :].expand(-1, e.shape[1], -1)
        return self.mlp(torch.cat([hc, e], dim=2)).squeeze(2)


def slot_sum(scores: Tensor, slots: Tensor) -> Tensor:
    """候选分数 = 所占格分数之和；填充格贡献 0。"""

    idx = torch.clamp_min(slots, 0).reshape(slots.shape[0], -1)
    gathered = torch.gather(scores, 1, idx).reshape(slots.shape)
    return torch.where(slots >= 0, gathered, torch.zeros_like(gathered)).sum(dim=2)


def centered_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """同根候选去中心后的 MSE；根内候选等权、根间等权。"""

    m = mask.float()
    cnt = m.sum(dim=1, keepdim=True)
    pc = pred - (pred * m).sum(dim=1, keepdim=True) / cnt
    tc = target - (target * m).sum(dim=1, keepdim=True) / cnt
    return (((pc - tc) ** 2) * m).sum(dim=1).div(cnt.squeeze(1)).mean()


def policy_kl(logits: Tensor, target: Tensor, legal: Tensor) -> Tensor:
    """与 train.py 相同的合法格 KL(target‖policy)，根间等权。"""

    log_p = functional.log_softmax(logits.masked_fill(~legal, float("-inf")), dim=1)
    log_p = torch.where(legal, log_p, torch.zeros_like(log_p))
    pos = target > 0
    log_t = torch.where(pos, torch.log(torch.clamp_min(target, 1e-30)), torch.zeros_like(target))
    return torch.where(pos, target * (log_t - log_p), torch.zeros_like(target)).sum(dim=1).mean()


def lr_factor(step: int) -> float:
    """warmup + cosine 到 final_lr_factor。"""

    if step < PLAN["warmup"]:
        return (step + 1) / PLAN["warmup"]
    t = (step - PLAN["warmup"]) / max(1, PLAN["steps"] - PLAN["warmup"])
    f = PLAN["final_lr_factor"]
    return f + (1 - f) * 0.5 * (1 + np.cos(np.pi * t))


def build_decoder(arm: str, member: dict, width: int) -> nn.Module:
    """按臂构造解码器；L 从 R8A 原 policy 行出发，A 零初始化。"""

    if arm == "L":
        dec = SlotDecoder(width)
        with torch.no_grad():
            dec.linear.weight.copy_(torch.from_numpy(member["head_w"]))
            dec.linear.bias.copy_(torch.from_numpy(member["head_b"]))
        return dec
    if arm == "A":
        dec = SlotDecoder(width)
        nn.init.zeros_(dec.linear.weight)
        nn.init.zeros_(dec.linear.bias)
        return dec
    return CandidateDecoder(width, PLAN["slot_embed_dim"], PLAN["n_hidden"])


def check_plan() -> None:
    """首次运行写入 plan.json，之后逐字比对。"""

    path = OUT / "frozen" / "plan.json"
    text = json.dumps(PLAN, indent=2, sort_keys=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError("plan.json 与当前配方不一致，拒绝运行")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def cmd_train(arm: str, seed: int, device: torch.device) -> None:
    """训练一个 (臂, 种子) 解码器，只保存第 3000 步。"""

    check_plan()
    out = OUT / "train" / f"{arm}_s{seed}.pt"
    if out.exists():
        raise FileExistsError(f"{out} 已存在；半途重跑需先移走并记偏离")
    roots = np.load(CACHE / "roots.npz")
    member = dict(np.load(CACHE / f"member_{seed}.npz"))
    nt = int(roots["n_train"])
    h = torch.from_numpy(member["h"][:nt]).to(device)
    legal = torch.from_numpy(roots["legal"][:nt]).to(device)
    ptarget = torch.from_numpy(roots["ptarget"][:nt]).to(device)
    slots = torch.from_numpy(roots["slots"][:nt].astype(np.int64)).to(device)
    cmean = torch.from_numpy((roots["cmean"][:nt] / PLAN["q_scale"]).astype(np.float32)).to(device)
    cmask = torch.from_numpy(roots["cmask"][:nt]).to(device)

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    dec = build_decoder(arm, member, h.shape[1]).to(device)
    opt = torch.optim.AdamW(dec.parameters(), lr=PLAN["lr"], weight_decay=PLAN["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    perm, pos, log = torch.randperm(nt, generator=gen), 0, []
    started = time.perf_counter()
    for step in range(PLAN["steps"]):
        if pos + PLAN["batch_roots"] > nt:
            perm, pos = torch.randperm(nt, generator=gen), 0
        idx = perm[pos : pos + PLAN["batch_roots"]].to(device)
        pos += PLAN["batch_roots"]
        if arm == "L":
            loss = policy_kl(dec(h[idx]), ptarget[idx], legal[idx])
        elif arm == "A":
            loss = centered_mse(slot_sum(dec(h[idx]), slots[idx]), cmean[idx], cmask[idx])
        else:
            loss = centered_mse(dec(h[idx], slots[idx]), cmean[idx], cmask[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0 or step == PLAN["steps"] - 1:
            log.append({"step": step + 1, "loss": float(loss)})
            print(f"[{arm} s{seed}] step {step + 1} loss {float(loss):.5f}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arm": arm, "seed": seed, "steps": PLAN["steps"], "state": dec.state_dict(), "log": log}, out)
    print(f"[{arm} s{seed}] 完成，用时 {time.perf_counter() - started:.1f}s")


# ---------------------------------------------------------------- eval


def candidate_scores(arm: str, seed: int, h: Tensor, slots: Tensor, member: dict, device: torch.device) -> tuple:
    """返回 (每格分数或 None, 候选分数)；P0 直接用缓存 logits。"""

    if arm == "P0":
        per_slot = torch.from_numpy(member["p0"]).to(device)
        return per_slot, slot_sum(per_slot, slots)
    ck = torch.load(OUT / "train" / f"{arm}_s{seed}.pt", map_location=device, weights_only=False)
    if ck["steps"] != PLAN["steps"]:
        raise RuntimeError(f"{arm}_s{seed} 不是第 {PLAN['steps']} 步")
    dec = build_decoder(arm, member, h.shape[1]).to(device)
    dec.load_state_dict(ck["state"])
    dec.eval()
    with torch.inference_mode():
        if arm == "N":
            return None, dec(h, slots)
        per_slot = dec(h)
        return per_slot, slot_sum(per_slot, slots)


def pick(per_slot: np.ndarray | None, cand: np.ndarray, slots: np.ndarray, mask: np.ndarray,
         legal: np.ndarray, stage: np.ndarray) -> tuple[np.ndarray, int]:
    """按 §5 选动作；返回候选下标与地区组合查表失败数。"""

    n = len(stage)
    choice = np.zeros(n, dtype=np.int64)
    misses = 0
    for i in range(n):
        valid = mask[i]
        if per_slot is None or stage[i] != REGION_STAGE:
            choice[i] = int(np.argmax(np.where(valid, cand[i], -np.inf)))
            continue
        lslots = np.flatnonzero(legal[i])
        top = lslots[np.argpartition(per_slot[i, lslots], -3)[-3:]]
        wanted = tuple(sorted(int(v) for v in top))
        found = -1
        for c in np.flatnonzero(valid):
            if tuple(sorted(int(v) for v in slots[i, c] if v >= 0)) == wanted:
                found = int(c)
                break
        if found < 0:
            misses += 1
            found = int(np.argmax(np.where(valid, cand[i], -np.inf)))
        choice[i] = found
    return choice, misses


def cluster_ci(diff: np.ndarray, cluster: np.ndarray, rng: np.random.Generator, draws: int) -> dict:
    """按组合聚类的配对 bootstrap；统计量是根等权均值。"""

    uniq, inv = np.unique(cluster, return_inverse=True)
    sums = np.bincount(inv, weights=diff, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    g = len(uniq)
    stats = np.empty(draws)
    for start in range(0, draws, 1000):
        k = min(1000, draws - start)
        pick_idx = rng.integers(0, g, size=(k, g))
        stats[start : start + k] = sums[pick_idx].sum(1) / cnts[pick_idx].sum(1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"mean": float(diff.mean()), "lo": float(lo), "hi": float(hi), "roots": int(len(diff)), "clusters": g}


def pair_accuracy(cand: np.ndarray, cmean: np.ndarray, mask: np.ndarray, gap: float) -> float:
    """同根候选对中 |Δ真实均分| ≥ gap 的对，预测顺序正确的比例。"""

    hit = tot = 0
    for i in range(len(cand)):
        v = np.flatnonzero(mask[i])
        t = cmean[i, v]
        p = cand[i, v]
        dt = t[:, None] - t[None, :]
        dp = p[:, None] - p[None, :]
        sel = dt >= gap
        tot += int(sel.sum())
        hit += int((dp[sel] > 0).sum())
    return hit / tot if tot else float("nan")


def calibration(cand: np.ndarray, cmean: np.ndarray, mask: np.ndarray) -> dict:
    """Q 臂：去中心预测（×q_scale）与去中心真实均分的 MAE 与斜率。"""

    m = mask.astype(np.float64)
    cnt = m.sum(1, keepdims=True)
    p = cand * PLAN["q_scale"]
    pc = (p - (p * m).sum(1, keepdims=True) / cnt)[mask]
    tc = (cmean - (cmean * m).sum(1, keepdims=True) / cnt)[mask]
    slope = float(np.dot(pc, tc) / np.dot(pc, pc))
    return {"mae": float(np.abs(pc - tc).mean()), "slope_true_on_pred": slope}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """无并列修正的 Spearman（平均秩）。"""

    def rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        # 平均并列秩
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=r)
        return sums[inv] / counts[inv]

    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def cmd_eval(device: torch.device) -> None:
    """§6/§7：验证集一次性评测，写 ``logs/q4_1006/result.json``。"""

    check_plan()
    roots = np.load(CACHE / "roots.npz")
    nt = int(roots["n_train"])
    stage = roots["stage"][nt:]
    legal = roots["legal"][nt:]
    slots_np = roots["slots"][nt:].astype(np.int64)
    cmean = roots["cmean"][nt:]
    mask = roots["cmask"][nt:]
    cluster = np.unique(roots["combo"][nt:], axis=0, return_inverse=True)[1].ravel()
    best = np.where(mask, cmean, -np.inf).max(1)
    slots_t = torch.from_numpy(slots_np).to(device)
    rows = np.arange(len(stage))

    members: dict[str, list] = {a: [] for a in ("P0", *ARMS)}
    for seed in SEEDS:
        member = dict(np.load(CACHE / f"member_{seed}.npz"))
        h = torch.from_numpy(member["h"][nt:]).to(device)
        member["p0"] = member["p0"][nt:]
        for arm in members:
            per_slot, cand = candidate_scores(arm, seed, h, slots_t, member, device)
            members[arm].append((None if per_slot is None else per_slot.cpu().numpy(), cand.cpu().numpy()))

    result: dict = {"plan": PLAN, "stage_counts": {STAGE_NAMES[k]: int((stage == k).sum()) for k in STAGE_NAMES}}
    regrets: dict[str, np.ndarray] = {}
    ens_cand: dict[str, np.ndarray] = {}
    for arm, outs in members.items():
        entry: dict = {"members": {}}
        variants = [(f"s{s}", o) for s, o in zip(SEEDS, outs)]
        per_ens = None if outs[0][0] is None else np.mean([o[0] for o in outs], axis=0)
        cand_ens = np.mean([o[1] for o in outs], axis=0)
        ens_cand[arm] = cand_ens
        for name, (per_slot, cand) in variants + [("ens", (per_ens, cand_ens))]:
            choice, misses = pick(per_slot, cand, slots_np, mask, legal, stage)
            regret = best - cmean[rows, choice]
            stats = {
                "regret": float(regret.mean()),
                "top1": float((regret <= 1e-9).mean()),
                "region_combo_misses": misses,
                "by_stage": {
                    STAGE_NAMES[k]: {
                        "regret": float(regret[stage == k].mean()),
                        "top1": float((regret[stage == k] <= 1e-9).mean()),
                    }
                    for k in STAGE_NAMES
                    if (stage == k).any()
                },
            }
            if name == "ens":
                entry.update(stats)
                regrets[arm] = regret
                entry["pair_acc"] = {
                    STAGE_NAMES[k]: pair_accuracy(cand[stage == k], cmean[stage == k], mask[stage == k], PLAN["pair_gap"])
                    for k in STAGE_NAMES
                    if (stage == k).any()
                }
                if arm in ("A", "N"):
                    entry["calibration"] = calibration(cand, cmean, mask)
            else:
                regrets[f"{arm}_{name}"] = regret
                entry["members"][name] = stats
        result[arm] = entry

    # 集成分歧（只描述）：三成员对「集成所选 − 集成次优」Q 差的标准差 vs 实际 regret
    for arm in ("A", "N"):
        ce = np.where(mask, ens_cand[arm], -np.inf)
        top2 = np.argsort(-ce, axis=1)[:, :2]
        gaps = np.array([o[1][rows, top2[:, 0]] - o[1][rows, top2[:, 1]] for o in members[arm]])
        result[arm]["disagreement_spearman"] = spearman(gaps.std(0), regrets[arm])

    rng = np.random.default_rng(PLAN["bootstrap_seed"])
    draws = PLAN["bootstrap_draws"]
    comps = {}
    for a, b in (("A", "L"), ("N", "L"), ("N", "A"), ("L", "P0"), ("A", "P0"), ("N", "P0")):
        d = regrets[a] - regrets[b]
        comps[f"{a}-{b}"] = cluster_ci(d, cluster, rng, draws)
        reg = stage == REGION_STAGE
        comps[f"{a}-{b}@RegionSelect"] = cluster_ci(d[reg], cluster[reg], rng, draws)
        comps[f"{a}-{b}@members"] = [
            float((regrets[f"{a}_s{s}"] - regrets[f"{b}_s{s}"]).mean()) for s in SEEDS
        ]
    result["comparisons"] = comps
    (OUT / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"阶段根数 {result['stage_counts']}")
    for arm in members:
        r = result[arm]
        print(f"{arm:>3} 集成 regret {r['regret']:8.2f}  top1 {r['top1']:.3f}  组合查表失败 {r['region_combo_misses']}  "
              f"成员 {[round(m['regret'], 1) for m in r['members'].values()]}")
        print("     分阶段 " + ", ".join(f"{k} {v['regret']:.1f}" for k, v in r["by_stage"].items()))
    for k, v in comps.items():
        if k.endswith("@members"):
            print(f"  {k:<24} {[round(x, 1) for x in v]}")
        else:
            print(f"  {k:<24} {v['mean']:8.2f} [{v['lo']:8.2f}, {v['hi']:8.2f}]  根 {v['roots']} 组合 {v['clusters']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("features")
    tr = sub.add_parser("train")
    tr.add_argument("--arm", choices=ARMS, required=True)
    tr.add_argument("--seed", type=int, choices=SEEDS, required=True)
    sub.add_parser("eval")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.cmd == "features":
        cmd_features(device)
    elif args.cmd == "train":
        cmd_train(args.arm, args.seed, device)
    else:
        cmd_eval(device)


if __name__ == "__main__":
    main()
