"""GPU 前向执行调度探针：只比较 eager 与 CUDA Graph，不改生产侧车与搜索路径。

本探针独立于 `scripts/ramen_nn/bench_sidecar.py` 运行，只复用 `model_from_checkpoint`
与**逐字照抄的集成表达式**（policy/choice 平均 logits；value 各成员反归一化后平均、
再按成员 0 重归一化）。不做 baddbmm/vmap 重写、不 compile、不开 TF32、不改模型定义。

三个子命令对应三步，必须分进程运行：profiler 的插桩开销不能混进稳态计时。

    python probe.py profile ...   # 第一步：短 profiler，只看 eager
    python probe.py verify ...    # 第二步：Graph 正确性
    python probe.py time ...      # 第三步：稳态交错计时
"""

import argparse
import contextlib
import csv
import io
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch

# 复用训练侧的模型重建入口；不复制模型定义，避免两份结构漂移
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "ramen_nn"))
from model import model_from_checkpoint  # noqa: E402

#: 编码器输入宽度，与 Rust 侧 INPUT_DIM 对应
INPUT_DIM = 754
#: 网络输出宽度：POLICY_DIM(234) + CHOICE_DIM(8) + VALUE_DIM(3)
OUTPUT_DIM = 245
#: logits 段（policy + choice）的结束位置，与侧车的 `o[:, :242]` 一致
LOGIT_END = 242

# ===== policy 格位布局，逐字对应 crates/umasim/src/game/ramen/policy_schema.rs =====
REGION_NUM = 20
TRIPLE_NUM = 10
TRAIN_OP_NUM = 10
SUPER_NUM = 3
EAT_NONE = 0
EAT_BASE = EAT_NONE + 1
TRAIN_BASE = EAT_BASE + REGION_NUM * TRIPLE_NUM
SUPER_BASE = TRAIN_BASE + TRAIN_OP_NUM
REGION_SELECT_BASE = SUPER_BASE + SUPER_NUM
POLICY_DIM = REGION_SELECT_BASE + REGION_NUM

#: 万能风味用法的规范全序，顺序是数据格式的一部分，不得重排
TRIPLES = [
    (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1),
    (0, 2, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0),
]

#: 基础操作 → 段内偏移，对应 policy_schema::train_index
TRAIN_OPS = {
    "Train(Speed)": 0, "Train(Stamina)": 1, "Train(Power)": 2,
    "Train(Guts)": 3, "Train(Wisdom)": 4, "Race": 5,
    "Rest": 6, "NormalOuting": 7, "FriendOuting": 8, "Clinic": 9,
}


# ========================= 模型与集成 =========================

def build_ensemble(paths, device="cuda"):
    """加载冻结成员并返回 (models, centers, scales)，全部 eval 且驻留 device。"""
    checkpoints = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    models = [model_from_checkpoint(c, device).eval() for c in checkpoints]
    centers = [torch.tensor(c["value_normalization"]["center"], device=device) for c in checkpoints]
    scales = [torch.tensor(c["value_normalization"]["scale"], device=device) for c in checkpoints]
    return models, centers, scales


def make_infer(models, centers, scales):
    """返回与 bench_sidecar.infer 表达式完全相同的集成前向闭包。

    集成口径不得改动：policy/choice 段直接平均 logits；value 段按各成员的
    normalization 还原到真实分数空间后平均，再用成员 0 的尺度重新归一化。
    """

    def infer(view):
        outputs = [model(view) for model in models]
        logits = torch.stack([o[:, :LOGIT_END] for o in outputs]).mean(0)
        values = torch.stack([o[:, LOGIT_END:] * s + c for o, s, c in zip(outputs, scales, centers)]).mean(0)
        return torch.cat((logits, (values - centers[0]) / scales[0]), dim=1)

    return infer


def freeze_numerics():
    """FP32、TF32 全关、单线程；与侧车运行期设置一致。"""
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


# ========================= 真实特征与候选 =========================

class Row:
    """一条真实决策记录：754 维输入 + 该决策点的合法候选与生产选中下标。"""

    __slots__ = ("stage", "turn", "actions", "chosen", "feat", "src")

    def __init__(self, stage, turn, actions, chosen, feat, src):
        self.stage = stage
        self.turn = turn
        self.actions = actions
        self.chosen = chosen
        self.feat = feat
        self.src = src


def load_rows(paths, per_stage):
    """从既有决策 CSV 取真实特征，按阶段均衡抽样。

    9 位有效数字的十进制可精确还原 f32，所以解析回来的就是当时实际编码的位模式；
    这里比较的是实际字段，不涉及任何哈希或指纹。
    """
    picked = defaultdict(list)
    for p in paths:
        with io.open(p, newline="", encoding="utf-8") as f:
            for rec in csv.DictReader(f):
                stage = rec["stage"]
                if len(picked[stage]) >= per_stage:
                    continue
                vals = rec["features"].split("|")
                if len(vals) != INPUT_DIM:
                    raise ValueError(f"{p}: 特征列数 {len(vals)} != {INPUT_DIM}")
                feat = np.array([np.float32(x) for x in vals], dtype=np.float32)
                if not np.isfinite(feat).all():
                    raise ValueError(f"{p}: 特征含非有限值")
                picked[stage].append(Row(
                    stage=stage,
                    turn=int(rec["turn"]),
                    actions=rec["actions"].split("|"),
                    chosen=int(rec["chosen"]),
                    feat=feat,
                    src=os.path.basename(p),
                ))
    out = []
    for stage in sorted(picked):
        out.extend(picked[stage])
    if not out:
        raise ValueError("没有取到任何决策行")
    return out


def parse_action(text):
    """把 `ramen/special_targets/operation` 文本解析成三元组。"""
    ramen, targets, op = text.split("/")
    rid = None if ramen == "-" else int(ramen)
    tgt = None if targets == "-" else tuple(int(x) for x in targets.split("+"))
    return rid, tgt, op


def action_slots(stage, text):
    """按生产口径给一个候选算出它占用的 policy 格位。

    返回 ``(kind, slots)``：

    - ``("one", [i])``     单格（Train / SuperRamenSelect / SpecialSelect / 不吃面）
    - ``("sum", [a,b,c])`` 三格求和（RegionSelect）
    - ``("max", [...])``   对合法风味用法取 max（RamenSelect 且选了某碗面）

    ``"max"`` 这一类的**实际**合法用法集合 T 来自 `rules::list_special_targets_for`，
    依赖 `feeling_stock`／`special_feeling`，决策 CSV 里没有记录，无法从本探针复原。
    这里给出的是该地区的 10 个规范三元组，即 T 的**上界**；报告必须写明这一点。
    """
    rid, tgt, op = parse_action(text)
    if stage == "Train":
        if op not in TRAIN_OPS:
            raise ValueError(f"Train 阶段出现未知操作: {op}")
        return "one", [TRAIN_BASE + TRAIN_OPS[op]]
    if stage == "SuperRamenSelect":
        idx = int(op[len("SuperRamenSelect("):-1])
        return "one", [SUPER_BASE + idx]
    if stage == "RegionSelect":
        ids = [int(x) for x in op[len("RegionSelect(["):-2].split(";")]
        return "sum", [REGION_SELECT_BASE + r for r in ids]
    if stage == "SpecialSelect":
        if rid is None or tgt is None:
            raise ValueError(f"SpecialSelect 候选缺少面或用法: {text}")
        return "one", [EAT_BASE + rid * TRIPLE_NUM + TRIPLES.index(tgt)]
    if stage == "RamenSelect":
        if rid is None:
            return "one", [EAT_NONE]
        base = EAT_BASE + rid * TRIPLE_NUM
        return "max", [base + t for t in range(TRIPLE_NUM)]
    raise ValueError(f"不是可映射的决策阶段: {stage}")


def score_actions(stage, actions, policy):
    """给全部候选打分，并返回该行的打分是否**严格**等于生产口径。

    返回 ``(scores, strict)``，分值一律是 ``np.float32``。

    - ``strict is True``：每个候选都是单格或固定三格求和，与
      `ramen_nn_trainer::score_one` 逐项一致，可以直接拿 argmax 作结论。
    - ``strict is False``：该行含「选了某碗面」的 `RamenSelect` 候选。生产口径要对
      `rules::list_special_targets_for` 给出的合法用法集合 T 取 max，而 T 依赖
      `feeling_stock` / `special_feeling`，决策 CSV 未记录、本探针无法复原。这里用
      该地区 10 个规范三元组（T 的上界）打分，**这不是生产口径**，其 argmax 只能
      作参考，不能单独用来判定通过。

    ❗地区三格求和必须用 **f32 逐项累加**且顺序与候选里地区 id 的书写顺序一致：
    生产侧 `score_one` 写的是 ``let mut sum = 0.0f32; for slot in ids { sum += ... }``。
    若改用 Python float（f64）累加再比较，三格和的舍入点与生产不同，
    并列或极窄间隔时可能给出与生产不一样的 argmax——那样「严格口径」就名不副实。
    """
    out = []
    strict = True
    for text in actions:
        kind, slots = action_slots(stage, text)
        if kind == "max":
            strict = False
        if kind == "sum":
            acc = np.float32(0.0)
            for s in slots:
                acc = np.float32(acc + policy[s])
            out.append(acc)
        elif kind == "one":
            out.append(np.float32(policy[slots[0]]))
        else:
            best = np.float32(policy[slots[0]])
            for s in slots[1:]:
                v = np.float32(policy[s])
                if v > best:
                    best = v
            out.append(best)
    return out, strict


def total_order_key(v):
    """把一个 f32 映射成 `f32::total_cmp` 所用的全序整数键。

    Rust 的 `total_cmp` 是**位模式上的全序**，与 IEEE 的 `<` 不同，差别正好落在
    两处本探针可能遇到的边界上：

    - `-0.0 < +0.0`（IEEE 里两者相等，Python 的 `>` 判不出差别）；
    - `-NaN` 小于一切、`+NaN` 大于一切（Python 里任何与 NaN 的比较都是 False）。

    实现照搬 `f32::total_cmp`：位模式当 i32 看，负数再异或 `0x7FFF_FFFF`。
    """
    bits = int(np.float32(v).view(np.uint32))
    signed = bits - 0x1_0000_0000 if bits >= 0x8000_0000 else bits
    return signed ^ ((signed >> 31) & 0x7FFF_FFFF)


def argmax_logit(scores):
    """复刻 `ramen_nn_trainer::argmax_logit`：按 `total_cmp` 取最大，并列取**较小**下标。

    ❗不能写成 `if scores[i] > scores[best]`：生产用的是 `total_cmp`，于是
    `[-0.0, +0.0]` 会选下标 1（`+0.0` 严格更大），而 `>` 认为两者相等、留在下标 0；
    出现 `+NaN` 时生产会选中它，而 `>` 永远不会。这两类值在本轮样本里都没出现过，
    但判据得先与生产一致，才谈得上「零分歧」。
    """
    best = 0
    best_key = total_order_key(scores[0])
    for i in range(1, len(scores)):
        key = total_order_key(scores[i])
        if key > best_key:
            best, best_key = i, key
    return best


def naive_argmax(scores):
    """修正前的朴素比较器，**只在合成回归里用**：用来证明用例真能区分两种实现。"""
    best = 0
    for i in range(1, len(scores)):
        if scores[i] > scores[best]:
            best = i
    return best


def f32_from_bits(bits):
    """按位模式造一个 f32（用来构造 ±NaN 这类没法靠字面量稳定得到的值）。"""
    return np.array([bits], dtype=np.uint32).view(np.float32)[0]


def relevant_slots(stage, actions):
    """该决策点可能被读到的全部格位。

    `RamenSelect` 取上界集合，所以这个集合是生产实际读取格位的**超集**：
    只要它们全部逐位相同，生产选出的动作就必然相同，与未知的 T 无关。
    这是本探针对非严格行的**唯一**判定依据。
    """
    s = set()
    for text in actions:
        _, slots = action_slots(stage, text)
        s.update(slots)
    return sorted(s)


# ========================= CUDA Graph =========================

class GraphRunner:
    """一个档位的 CUDA Graph：静态输入/输出缓冲区常驻，replay 前更新输入。"""

    def __init__(self, infer, batch, device="cuda", warmup=5):
        self.batch = batch
        self.static_in = torch.zeros((batch, INPUT_DIM), device=device, dtype=torch.float32)
        # 捕获前必须在旁路 stream 上预热：首次调用的 kernel 选型、cuBLAS 句柄与
        # workspace 分配都不能落进图里，否则 replay 会重放一次性初始化。
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                infer(self.static_in)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_out = infer(self.static_in)
        torch.cuda.synchronize()
        self.capture_s = time.perf_counter() - t0
        self.out_ptr = self.static_out.data_ptr()

    def replay(self):
        """重放整段前向。返回的是**静态输出缓冲区本身**，调用方必须自行快照。"""
        self.graph.replay()
        return self.static_out


# ========================= 第一步：短 profiler =========================

def probe_activities():
    """报告本环境实际支持的 profiler 活动类型。

    Windows 上 torch 官方 wheel 常常不带 CUPTI，此时 `ProfilerActivity.CUDA`
    会被**静默忽略**，`key_averages()` 里的 CUDA 列是从 CPU 事件推导出来的，
    不是设备实测值，绝不能拿来归因。
    """
    try:
        acts = torch._C._autograd._supported_activities()
    except Exception as e:  # noqa: BLE001 - 只用于报告环境能力
        return None, f"查询失败: {e}"
    names = sorted(str(a).split(".")[-1] for a in acts)
    has_cuda = any("CUDA" in n for n in names)
    return has_cuda, names


def dump_graph_nodes(infer, batch, out_dir):
    """捕获一次 Graph 并导出节点图，得到整段前向的**实际 kernel 节点数**。

    这条路径不经过 CUPTI：节点来自 CUDA 自己记录的图结构，是设备侧事实。
    """
    static_in = torch.zeros((batch, INPUT_DIM), device="cuda", dtype=torch.float32)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(5):
            infer(static_in)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    g.enable_debug_mode()
    with torch.cuda.graph(g):
        infer(static_in)
    torch.cuda.synchronize()
    path = os.path.join(out_dir, f"graph_nodes_B{batch}.dot")
    g.debug_dump(path)
    del g
    torch.cuda.empty_cache()
    if not os.path.exists(path):
        return None, path
    kinds = Counter()
    with io.open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "label=" not in line or "->" in line:
                continue
            lab = line.split("label=", 1)[1].strip().strip('"[]')
            head = lab.split("\\n")[0].strip()
            # 只统计节点：节点标签以 `{类型` 开头，子图/容器行没有这个前缀
            if head.startswith("{"):
                kinds[head.lstrip("{").strip()] += 1
    return kinds, path


def cmd_profile(args):
    """对 eager 路径做短采样，报告算子构成、实际 kernel 节点数与 CPU 提交开销。"""
    freeze_numerics()
    has_cuda_act, act_names = probe_activities()
    print(f"---- profiler 能力 ----")
    print(f"  本环境支持的活动类型: {act_names}")
    if not has_cuda_act:
        print("  ❗CUPTI / ProfilerActivity.CUDA 在本环境不可用：设备侧 kernel 时间线无法采集。")
        print("  ❗因此 key_averages() 的 CUDA 列是由 CPU 事件推导的，本报告不采用它做任何归因。")
        print("  ❗按约束不安装任何包、不改运行环境；改用两条不依赖 CUPTI 的证据：")
        print("     (1) CUDA Graph 节点图 → 整段前向的实际 kernel 节点数；")
        print("     (2) 提交/完成分离计时 → CPU 提交是否已经追不上 GPU。")
    print()

    models, centers, scales = build_ensemble(args.checkpoint)
    infer = make_infer(models, centers, scales)
    rows = load_rows(args.dec_csv, per_stage=args.batch)
    buf = torch.zeros((args.batch, INPUT_DIM), device="cuda", dtype=torch.float32)
    src = np.zeros((args.batch, INPUT_DIM), dtype=np.float32)
    for i in range(args.batch):
        src[i] = rows[i % len(rows)].feat
    buf.copy_(torch.from_numpy(src))

    trace_path = os.path.join(args.out_dir, f"trace_eager_B{args.batch}.json")
    wait, warm, active = 2, 3, args.steps
    with torch.inference_mode():
        for _ in range(10):
            infer(buf)
        torch.cuda.synchronize()
        sched = torch.profiler.schedule(wait=wait, warmup=warm, active=active, repeat=1)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=sched,
            record_shapes=True,
        ) as prof:
            for _ in range(wait + warm + active):
                infer(buf)
                prof.step()
        torch.cuda.synchronize()
        prof.export_chrome_trace(trace_path)

    print(f"[profile] eager B={args.batch}，采样 {active} 步，trace 见 {trace_path}")
    print()
    print("---- torch 自带聚合（按 CPU 自用时间排序，前 15 行）----")
    print("❗只读 CPU 列；CUDA 列在无 CUPTI 的环境下不是设备实测值" if not has_cuda_act else "")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
    print()
    summarize_trace(trace_path, active, has_cuda_act)

    print()
    print("---- 整段前向的实际 kernel 节点数（来自 CUDA Graph 节点图，不经 CUPTI）----")
    with torch.inference_mode():
        kinds, dot_path = dump_graph_nodes(infer, args.batch, args.out_dir)
    if kinds is None:
        print(f"  ❗节点图导出失败，未生成 {dot_path}")
    else:
        total = sum(kinds.values())
        print(f"  节点图 {dot_path}")
        print(f"  一次完整集成前向（3 个成员）共 {total} 个图节点：")
        for k, n in kinds.most_common():
            print(f"    {n:>5} 个  {k[:100]}")
        print(f"  ❗这是 eager 路径录下来的同一批 kernel；Graph 只是把它们打包重放，不合并、不改 kernel。")

    print()
    print("---- 提交 / 完成分离计时（eager，未开 profiler，只作归因不作正式计时）----")
    with torch.inference_mode():
        for _ in range(20):
            infer(buf)
        torch.cuda.synchronize()
        for iters in (20, 100):
            t0 = time.perf_counter()
            for _ in range(iters):
                infer(buf)
            t_submit = (time.perf_counter() - t0) * 1000.0
            torch.cuda.synchronize()
            t_total = (time.perf_counter() - t0) * 1000.0
            print(f"  {iters:>4} 次：CPU 提交完最后一次 {t_submit / iters:7.3f} ms/次 |"
                  f" 等到 GPU 全部做完 {t_total / iters:7.3f} ms/次 |"
                  f" 提交占完成 {100.0 * t_submit / t_total:5.1f}%")
        print("  ❗提交占比接近 100% 说明 CPU 侧派发已经追不上 GPU，队列基本没有积压；")
        print("     接近 0% 则说明 GPU 是限制者。这条只回答「谁在等谁」，不宣称 kernel 级归因。")

    print()
    print("---- 同一口径下 Graph 的提交 / 完成（对照，用来判断谁是节拍器）----")
    with torch.inference_mode():
        gr = GraphRunner(infer, args.batch)
        gr.static_in.copy_(buf)
        for _ in range(20):
            gr.graph.replay()
        torch.cuda.synchronize()
        for iters in (20, 100):
            t0 = time.perf_counter()
            for _ in range(iters):
                gr.graph.replay()
            t_submit = (time.perf_counter() - t0) * 1000.0
            torch.cuda.synchronize()
            t_total = (time.perf_counter() - t0) * 1000.0
            print(f"  {iters:>4} 次：CPU 提交完最后一次 {t_submit / iters:7.3f} ms/次 |"
                  f" 等到 GPU 全部做完 {t_total / iters:7.3f} ms/次 |"
                  f" 提交占完成 {100.0 * t_submit / t_total:5.1f}%")
        del gr
        torch.cuda.empty_cache()


def summarize_trace(trace_path, steps, has_cuda_act=True):
    """直接读 chrome trace，按 trace 里实际存在的事件统计，不做推断。"""
    with io.open(trace_path, encoding="utf-8") as f:
        doc = json.load(f)
    evs = [e for e in doc.get("traceEvents", []) if e.get("ph") == "X"]
    by_cat = Counter()
    dur_by_cat = defaultdict(float)
    for e in evs:
        c = e.get("cat", "?")
        by_cat[c] += 1
        dur_by_cat[c] += e.get("dur", 0) or 0

    kernels = [e for e in evs if e.get("cat") == "kernel"]
    memops = [e for e in evs if e.get("cat") in ("gpu_memcpy", "gpu_memset")]
    rt = [e for e in evs if e.get("cat") in ("cuda_runtime", "cuda_driver")]
    launches = [e for e in rt if e.get("name", "").startswith(("cudaLaunchKernel", "cuLaunchKernel"))]
    syncs = [e for e in rt if "Synchronize" in e.get("name", "")]
    allocs = [e for e in rt if e.get("name", "") in ("cudaMalloc", "cudaFree", "cudaHostAlloc")]
    steps_ev = [e for e in evs if e.get("cat") == "user_annotation" and e.get("name", "").startswith("ProfilerStep")]

    print("---- trace 事件分类（总计 / 每步）----")
    for c, n in by_cat.most_common():
        print(f"  {c:<18} {n:>7} 个 | 每步 {n / steps:8.1f} 个 | 总时长 {dur_by_cat[c] / 1000:9.3f} ms")

    k_total = sum(e.get("dur", 0) for e in kernels)
    m_total = sum(e.get("dur", 0) for e in memops)
    l_total = sum(e.get("dur", 0) for e in launches)
    s_total = sum(e.get("dur", 0) for e in syncs)
    a_total = sum(e.get("dur", 0) for e in allocs)
    step_total = sum(e.get("dur", 0) for e in steps_ev)

    print()
    print("---- 每步关键量（trace 直接可证）----")
    print(f"  ProfilerStep 墙钟   {step_total / max(len(steps_ev), 1) / 1000:8.3f} ms/步（含 profiler 插桩开销，不是稳态耗时）")
    print(f"  CPU 算子事件        {len([e for e in evs if e.get('cat') == 'cpu_op']) / steps:8.1f} 个/步")
    if not kernels and not rt:
        print("  ❗trace 里没有任何 kernel / cuda_runtime 事件：设备时间线未被采集（CUPTI 不可用）。")
        print("  ❗因此下面不报 kernel 时长、launch 次数、同步与分配耗时——没有采到就是没有，不做推测。")
        return
    print(f"  GPU kernel 个数     {len(kernels) / steps:8.1f} 个/步")
    print(f"  GPU kernel 总时长   {k_total / steps / 1000:8.3f} ms/步")
    print(f"  GPU memcpy/memset   {len(memops) / steps:8.1f} 个/步，{m_total / steps / 1000:8.3f} ms/步")
    print(f"  cudaLaunchKernel    {len(launches) / steps:8.1f} 次/步，{l_total / steps / 1000:8.3f} ms/步（CPU 侧提交）")
    print(f"  同步 API            {len(syncs) / steps:8.1f} 次/步，{s_total / steps / 1000:8.3f} ms/步")
    print(f"  分配 API            {len(allocs) / steps:8.1f} 次/步，{a_total / steps / 1000:8.3f} ms/步")
    if step_total:
        print(f"  GPU kernel 占步墙钟 {100.0 * k_total / step_total:8.1f}%（其余是 CPU 提交、间隙与插桩）")

    if kernels:
        print()
        print("---- GPU 时间前 12 的 kernel ----")
        agg = defaultdict(lambda: [0, 0.0])
        for e in kernels:
            a = agg[e.get("name", "?")]
            a[0] += 1
            a[1] += e.get("dur", 0) or 0
        top = sorted(agg.items(), key=lambda kv: -kv[1][1])[:12]
        for name, (n, d) in top:
            share = 100.0 * d / k_total if k_total else 0.0
            print(f"  {d / steps / 1000:7.3f} ms/步 | {share:5.1f}% | {n / steps:6.1f} 次/步 | {name[:88]}")


# ========================= 第二步：Graph 正确性 =========================

def compare(a, b):
    """比较两个 f32 张量：非有限数、最大绝对/相对误差、位不同元素数。"""
    an = np.ascontiguousarray(a.detach().cpu().numpy(), dtype=np.float32)
    bn = np.ascontiguousarray(b.detach().cpu().numpy(), dtype=np.float32)
    bits = int((an.view(np.uint32) != bn.view(np.uint32)).sum())
    # 非有限值单独计数；误差用 nan 安全的方式算，避免 NaN 参与减法刷警告
    with np.errstate(invalid="ignore", over="ignore"):
        diff = np.abs(an.astype(np.float64) - bn.astype(np.float64))
        denom = np.maximum(np.abs(an.astype(np.float64)), np.abs(bn.astype(np.float64)))
        rel = np.where(denom > 0, diff / np.maximum(denom, 1e-300), 0.0)
    diff = np.nan_to_num(diff, nan=np.inf)
    rel = np.nan_to_num(rel, nan=np.inf)
    return {
        "bits": bits,
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "max_rel": float(rel.max()) if rel.size else 0.0,
        "nonfinite": int((~np.isfinite(an)).sum() + (~np.isfinite(bn)).sum()),
        "n": an.size,
    }


def fmt_cmp(tag, c, ok):
    """把一次比较结果排成一行，并带上该项的判定。"""
    return (f"  [{'通过' if ok else '未通过'}] {tag:<30} 位不同 {c['bits']:>8} / {c['n']:<9}"
            f" 最大绝对 {c['max_abs']:.6e} 最大相对 {c['max_rel']:.6e} 非有限 {c['nonfinite']}")


class Verdict:
    """验收累加器：任何一项检查都必须经过它，否则不可能影响退出码。

    这是上一版的漏洞所在——补零尾段的位差、以及「有位差但动作恰好没翻转」
    的情况都只打印不计数，退出码照样是 0。
    """

    def __init__(self):
        self.fails = []
        self.checks = 0

    def note(self, tag, ok, detail=""):
        """记录一项布尔判定。"""
        self.checks += 1
        if not ok:
            self.fails.append(f"{tag}{(' | ' + detail) if detail else ''}")
        return ok

    def equal(self, tag, c):
        """要求两个张量逐位相同且都不含非有限值。"""
        ok = c["bits"] == 0 and c["nonfinite"] == 0
        print(fmt_cmp(tag, c, ok))
        return self.note(tag, ok, f"位不同 {c['bits']} 非有限 {c['nonfinite']}")

    def differ(self, tag, c):
        """要求两个张量确实不同（用于证明输入真的生效）。"""
        ok = c["bits"] != 0 and c["nonfinite"] == 0
        print(fmt_cmp(tag, c, ok))
        return self.note(tag, ok, f"位不同 {c['bits']} 非有限 {c['nonfinite']}")

    @property
    def failed(self):
        return len(self.fails)

    def report(self):
        """打印汇总，返回未通过项数。"""
        print(f"===== 检查项 {self.checks} 项，未通过 {self.failed} 项 =====")
        for f in self.fails:
            print(f"  ❗未通过：{f}")
        return self.failed


def run_verify(infer, rows, batches, inject=None, verbose=True):
    """逐档捕获 Graph 与 eager 对比，返回 [`Verdict`]。

    `inject` 是错误注入钩子：每次取 Graph 快照后调用一次，用来验证本验收器
    真的能拦住错误。正式验收传 None。
    """
    v = Verdict()
    if verbose:
        print(f"真实决策行 {len(rows)} 条，阶段分布：{dict(Counter(r.stage for r in rows))}")
        print(f"来源：{sorted(set(r.src for r in rows))}")
        print()

    def snap(gr):
        """取一份 Graph 输出快照。必须 clone：静态输出缓冲区会被下一次 replay 覆盖。"""
        out = gr.replay().clone()
        return inject(out) if inject is not None else out

    with torch.inference_mode():
        eager_buf = torch.zeros((max(batches), INPUT_DIM), device="cuda", dtype=torch.float32)
        for B in batches:
            print(f"================= 档位 B={B} =================")
            gr = GraphRunner(infer, B)
            print(f"捕获耗时 {gr.capture_s * 1000:.1f} ms；静态输入 {tuple(gr.static_in.shape)}"
                  f" 输出 {tuple(gr.static_out.shape)}")

            # --- 陷阱：graph 输出缓冲区在 replay 之间地址不变，不快照就是在比同一块内存 ---
            gr.static_in.zero_()
            p1 = gr.replay().data_ptr()
            snap_a = snap(gr)
            p2 = gr.static_out.data_ptr()
            v.note("输出缓冲区地址在 replay 间保持不变", p1 == p2 == gr.out_ptr,
                   "地址变了说明下面的 clone 假设不成立")
            print(f"  输出缓冲区地址 replay 前后一致: {p1 == p2 == gr.out_ptr} → 每次 replay 后必须 clone")

            # --- 静态输入确实被读：不更新输入则输出不变，更新后必须变 ---
            snap_b = snap(gr)
            v.equal("不更新输入再 replay", compare(snap_a, snap_b))
            fill_rows(gr.static_in, rows, B, offset=0)
            snap_c = snap(gr)
            v.differ("更新输入后 replay（应当不同）", compare(snap_a, snap_c))

            # --- A→B→A、重复调用 ---
            fill_rows(gr.static_in, rows, B, offset=0)
            a1 = snap(gr)
            a1b = snap(gr)
            fill_rows(gr.static_in, rows, B, offset=B)
            b1 = snap(gr)
            fill_rows(gr.static_in, rows, B, offset=0)
            a2 = snap(gr)
            v.equal("重复调用 A,A", compare(a1, a1b))
            v.equal("A→B→A 的两次 A", compare(a1, a2))
            v.differ("A vs B（应当不同）", compare(a1, b1))

            # --- 尾批补零 / 有效行数变化：有效段与补零尾段都必须进退出码 ---
            for valid in sorted({1, max(1, B // 4), max(1, B // 2), B - 1, B}):
                gr.static_in.zero_()
                fill_rows(gr.static_in, rows, valid, offset=0)
                g_out = snap(gr)
                eager_buf[:B].zero_()
                fill_rows(eager_buf[:B], rows, valid, offset=0)
                e_out = infer(eager_buf[:B])
                v.equal(f"有效 {valid}/{B} 的有效段", compare(e_out[:valid], g_out[:valid]))
                if valid < B:
                    v.equal(f"有效 {valid}/{B} 的补零尾段", compare(e_out[valid:], g_out[valid:]))
                d, u, _, _ = check_actions(rows, valid, e_out, g_out)
                v.note(f"有效 {valid}/{B} 的动作", d == 0 and u == 0, f"分歧 {d} 无法判定 {u}")

            # --- 全批真实输入：完整输出 + 分两类判据的动作检查 ---
            total_bits = 0
            diverge = unverif = strict_n = loose_n = 0
            gaps = defaultdict(list)
            for offset in range(0, len(rows), B):
                n = min(B, len(rows) - offset)
                gr.static_in.zero_()
                fill_rows(gr.static_in, rows, n, offset=offset)
                g_out = snap(gr)
                eager_buf[:B].zero_()
                fill_rows(eager_buf[:B], rows, n, offset=offset)
                e_out = infer(eager_buf[:B])
                c = compare(e_out[:n], g_out[:n])
                total_bits += c["bits"]
                v.note(f"真实输入批 offset={offset} 逐位相同",
                       c["bits"] == 0 and c["nonfinite"] == 0,
                       f"位不同 {c['bits']} 非有限 {c['nonfinite']} 最大绝对 {c['max_abs']:.6e}")
                d, u, sn, ln = check_actions(rows[offset:offset + n], n, e_out, g_out, gaps=gaps)
                diverge += d
                unverif += u
                strict_n += sn
                loose_n += ln
            print(f"  全部 {len(rows)} 条真实输入：累计位不同 {total_bits}")
            print(f"    严格口径行 {strict_n} 条 → 动作分歧 {diverge}")
            print(f"    非严格行（选了面的 RamenSelect）{loose_n} 条 → 无法判定 {unverif}")
            print("      判据是相关格位超集逐位相同；只要成立，未知的 T 取什么都不影响动作")
            v.note(f"B={B} 严格口径动作零分歧", diverge == 0, f"分歧 {diverge} 条")
            v.note(f"B={B} 非严格行全部可判定", unverif == 0, f"无法判定 {unverif} 条")

            if verbose:
                print("  按阶段的 top1-top2 间隔（eager 侧；RamenSelect 为上界口径，仅作余量参考）：")
                for stage in sorted(gaps):
                    s = sorted(gaps[stage])
                    print(f"    {stage:<18} n={len(s):>4} 最小 {s[0]:.6e}"
                          f" 中位 {s[len(s) // 2]:.6e} 最大 {s[-1]:.6e}")
            del gr
            torch.cuda.empty_cache()
            print()
    return v


def cmd_verify(args):
    """正式验收：不注入错误。"""
    models, centers, scales = build_ensemble(args.checkpoint)
    infer = make_infer(models, centers, scales)
    rows = load_rows(args.dec_csv, per_stage=args.per_stage)
    v = run_verify(infer, rows, args.batch)
    rc = v.report()
    print("❗有限样本零分歧不等于全局等价：本检查只覆盖上面列出的输入与档位。")
    return 1 if rc else 0


# ========================= 错误注入自检 =========================

def flip_lsb(col):
    """把一列 f32 的最低位翻一位：位模式变了，数值几乎没变。"""
    raw = col.view(torch.int32) ^ torch.tensor(1, dtype=torch.int32, device=col.device)
    return raw.view(torch.float32)


def injectors():
    """一组注入器；每一个都应当让验收器至少有一项未通过。

    验收器自己也要被验收——否则「日志全绿」只说明它没在报错，不说明它报得出来。
    """

    def one_bit(t):
        """单个元素翻最低位。"""
        out = t.clone()
        out.view(-1)[5] = flip_lsb(out.view(-1)[5:6])[0]
        return out

    def nan_cell(t):
        """注入 NaN。"""
        out = t.clone()
        out.view(-1)[7] = float("nan")
        return out

    def inf_cell(t):
        """注入 Inf。"""
        out = t.clone()
        out.view(-1)[9] = float("inf")
        return out

    def tail_dirt(t):
        """只污染最后一行，专打补零尾段那条上一版漏掉的检查。"""
        out = t.clone()
        out[-1, 0] = out[-1, 0] + 1.0
        return out

    def flip_strict(t):
        """抬高 Train 段的 Rest 格位，制造严格口径下的动作分歧。"""
        out = t.clone()
        out[:, TRAIN_BASE + 6] = out[:, TRAIN_BASE + 6] + 100.0
        return out

    def loose_only(t):
        """只动某个吃面格位一位，改动小到不会翻转上界 argmax。

        上一版会因为「动作恰好没变」而放行；现在必须以「无法判定」拦下。
        """
        out = t.clone()
        col = EAT_BASE + 9 * TRIPLE_NUM + 3
        out[:, col] = flip_lsb(out[:, col])
        return out

    state = {"n": 0}

    def unstable(t):
        """第 2 次及以后的快照漂移，模拟重放不稳定。"""
        state["n"] += 1
        if state["n"] <= 1:
            return t
        out = t.clone()
        out[0, 0] = out[0, 0] + 1e-3
        return out

    return [
        ("单个元素翻最低位", one_bit),
        ("注入 NaN", nan_cell),
        ("注入 Inf", inf_cell),
        ("只污染补零尾段的最后一行", tail_dirt),
        ("抬高 Train/Rest 格位（严格口径动作分歧）", flip_strict),
        ("只改吃面格位一位（动作不变，应判无法判定）", loose_only),
        ("第二次起快照漂移（重放不稳定）", unstable),
    ]


def cmd_selftest(args):
    """逐个注入错误，确认验收器**确实会**未通过并给出非零退出码。"""
    models, centers, scales = build_ensemble(args.checkpoint)
    infer = make_infer(models, centers, scales)
    rows = load_rows(args.dec_csv, per_stage=args.per_stage)
    cases = injectors()
    print(f"错误注入自检：{len(rows)} 条真实输入，档位 {args.batch}，注入 {len(cases)} 种")
    print()

    base = run_verify(infer, rows, args.batch, inject=None, verbose=False)
    ok_base = base.failed == 0
    print(f"--- 基线（不注入）：检查 {base.checks} 项，未通过 {base.failed} 项 ---")
    for f in base.fails[:5]:
        print(f"    ❗{f}")
    print()

    survived = []
    for name, fn in cases:
        v = run_verify(infer, rows, args.batch, inject=fn, verbose=False)
        killed = v.failed > 0
        print(f"[{'拦住' if killed else '漏网'}] {name}：未通过 {v.failed}/{v.checks} 项")
        if v.fails:
            print(f"    首个未通过项：{v.fails[0]}")
        print()
        if not killed:
            survived.append(name)

    print(f"===== 注入自检：{len(cases)} 个错误，拦住 {len(cases) - len(survived)} 个，"
          f"漏网 {len(survived)} 个；基线干净 {ok_base} =====")
    for s in survived:
        print(f"  ❗漏网：{s}")
    return 0 if (ok_base and not survived) else 1


def fill_rows(dst, rows, n, offset=0):
    """把 `rows` 里从 `offset` 起的 n 条真实特征写进目标缓冲区的前 n 行。"""
    src = np.zeros((n, INPUT_DIM), dtype=np.float32)
    for i in range(n):
        src[i] = rows[(offset + i) % len(rows)].feat
    dst[:n].copy_(torch.from_numpy(src))


def check_actions(rows, n, e_out, g_out, gaps=None):
    """比较动作，按「能否严格复现生产口径」分两条判据。

    返回 ``(diverge, unverifiable, strict_rows, loose_rows)``：

    - **严格行**（Train / SuperRamenSelect / SpecialSelect / RegionSelect / 不吃面的
      RamenSelect）：直接比生产 argmax，`diverge` 计数。
    - **非严格行**（选了某碗面的 RamenSelect）：生产口径依赖未记录的合法用法集合 T，
      本探针无法复现，因此**不拿上界 argmax 当判据**。唯一可靠判据是相关格位的超集
      逐位相同——那样无论 T 是什么，生产选出的动作都相同。一旦出现位差，这一行就是
      **无法判定**（`unverifiable`），按未通过处理，而不是「恰好动作一样所以放行」。
    """
    e_pol = e_out[:n, :POLICY_DIM].contiguous().detach().cpu().numpy().astype(np.float32)
    g_pol = g_out[:n, :POLICY_DIM].contiguous().detach().cpu().numpy().astype(np.float32)
    diverge = 0
    unverifiable = 0
    strict_rows = 0
    loose_rows = 0
    for i in range(min(n, len(rows))):
        r = rows[i]
        slots = relevant_slots(r.stage, r.actions)
        same = all(e_pol[i][s].view(np.uint32) == g_pol[i][s].view(np.uint32) for s in slots)
        es, strict = score_actions(r.stage, r.actions, e_pol[i])
        gs, _ = score_actions(r.stage, r.actions, g_pol[i])
        ea, ga = argmax_logit(es), argmax_logit(gs)
        srt = sorted(es, reverse=True)
        gap = (srt[0] - srt[1]) if len(srt) > 1 else float("inf")
        if gaps is not None and len(es) > 1:
            gaps[r.stage].append(gap)
        if strict:
            strict_rows += 1
            if ea != ga:
                diverge += 1
                print(f"  ❗动作分歧（严格口径）turn={r.turn} stage={r.stage}"
                      f" 候选 {len(r.actions)} eager→{ea} graph→{ga} top1-top2={gap:.6e}")
        else:
            loose_rows += 1
            if not same:
                unverifiable += 1
                print(f"  ❗无法判定 turn={r.turn} stage={r.stage} 候选 {len(r.actions)}："
                      f"相关格位出现位差，而该行的生产口径依赖未记录的合法用法集合 T，"
                      f"不能靠上界 argmax（eager→{ea} graph→{ga}）放行")
    return diverge, unverifiable, strict_rows, loose_rows


# ========================= 第三步：稳态计时 =========================

def timed_block(fn, iters):
    """跑 iters 次，返回 (GPU 事件毫秒, 主机墙钟毫秒)。两者都在末尾正确同步。"""
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    ev0.record()
    for _ in range(iters):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1000.0
    return ev0.elapsed_time(ev1), wall


def cmd_time(args):
    """交错运行 eager / Graph，分两种调用口径报告稳态耗时。"""
    freeze_numerics()
    t_load = time.perf_counter()
    models, centers, scales = build_ensemble(args.checkpoint)
    infer = make_infer(models, centers, scales)
    load_s = time.perf_counter() - t_load
    rows = load_rows(args.dec_csv, per_stage=args.per_stage)
    print(f"模型加载 {load_s:.2f} s（单列，不计入稳态）；真实决策行 {len(rows)} 条")
    print(f"迭代 {args.iters} 次/组，{args.reps} 组交错；奇数组 eager 先跑，偶数组 Graph 先跑")
    print()

    results = defaultdict(lambda: defaultdict(list))
    with torch.inference_mode():
        for B in args.batch:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            buf = torch.zeros((B, INPUT_DIM), device="cuda", dtype=torch.float32)
            fill_rows(buf, rows, B, offset=0)
            host = torch.from_numpy(np.ascontiguousarray(
                np.stack([rows[i % len(rows)].feat for i in range(B)]))).pin_memory()

            t_w = time.perf_counter()
            for _ in range(args.warmup):
                infer(buf)
            torch.cuda.synchronize()
            warm_s = time.perf_counter() - t_w
            gr = GraphRunner(infer, B)
            gr.static_in.copy_(buf)
            for _ in range(args.warmup):
                gr.replay()
            torch.cuda.synchronize()
            mem_alloc = torch.cuda.max_memory_allocated() / 2 ** 20
            mem_res = torch.cuda.max_memory_reserved() / 2 ** 20

            # 口径 1：输入已在 GPU，不含 H2D/D2H
            def eager_resident():
                infer(buf)

            def graph_resident():
                gr.graph.replay()

            # 口径 2：含输入更新与输出读出，贴近侧车一次调用做的事
            def eager_full():
                buf.copy_(host, non_blocking=True)
                out = infer(buf)
                out.cpu()

            def graph_full():
                gr.static_in.copy_(host, non_blocking=True)
                gr.graph.replay()
                gr.static_out.cpu()

            arms = {
                ("驻留", "eager"): eager_resident, ("驻留", "graph"): graph_resident,
                ("完整", "eager"): eager_full, ("完整", "graph"): graph_full,
            }
            for rep in range(1, args.reps + 1):
                order = ["eager", "graph"] if rep % 2 else ["graph", "eager"]
                for scen in ("驻留", "完整"):
                    for who in order:
                        gpu_ms, wall_ms = timed_block(arms[(scen, who)], args.iters)
                        results[(B, scen, who)]["gpu"].append(gpu_ms / args.iters)
                        results[(B, scen, who)]["wall"].append(wall_ms / args.iters)

            print(f"---- B={B} ----")
            print(f"  首次成本：eager 预热 {warm_s * 1000:.1f} ms | Graph 捕获 {gr.capture_s * 1000:.1f} ms"
                  f"（捕获前另做 5 次旁路预热）")
            print(f"  显存峰值：allocated {mem_alloc:.1f} MiB | reserved {mem_res:.1f} MiB")
            for scen in ("驻留", "完整"):
                for who in ("eager", "graph"):
                    g = sorted(results[(B, scen, who)]["gpu"])
                    w = sorted(results[(B, scen, who)]["wall"])
                    med = g[len(g) // 2]
                    medw = w[len(w) // 2]
                    print(f"  {scen} {who:<5} GPU 中位 {med:7.3f} ms [{g[0]:7.3f},{g[-1]:7.3f}]"
                          f" | 主机墙钟中位 {medw:7.3f} ms [{w[0]:7.3f},{w[-1]:7.3f}]"
                          f" | {B / (medw / 1000):9.0f} req/s(按墙钟)")
                ge = sorted(results[(B, scen, "eager")]["wall"])
                gg = sorted(results[(B, scen, "graph")]["wall"])
                me, mg = ge[len(ge) // 2], gg[len(gg) // 2]
                # 保守区间：eager 最快 vs graph 最慢，以及反向
                print(f"  {scen} 墙钟中位比 eager/graph = {me / mg:.3f}×"
                      f"；最保守 {ge[0] / gg[-1]:.3f}× ~ 最宽松 {ge[-1] / gg[0]:.3f}×")
            del gr, buf, host
            torch.cuda.empty_cache()
            print()
    return 0


# ========================= 合成回归（不需要 GPU） =========================

def unit_cases():
    """构造用例，返回 `(名称, 是否通过, 说明)`。

    与 `selftest` 互补、不可互相替代：注入自检查的是「验收器对**输出异常**的灵敏度」，
    这里查的是「**打分与比较器本身**是否等于生产口径」。
    后者即使输出完全正常也可能错，错了就会让「零分歧」这个结论失去意义。
    """
    out = []

    # --- 用例 1：f64 求和与 f32 逐项累加给出不同的 argmax ---
    # 1 + 2^-24 在 f32 里正好落在 1.0 与 1+2^-23 的中点，按「就近取偶」回到 1.0，
    # 再加一次仍是 1.0；同样这两个数在 f64 里精确相加得到 1 + 2^-23。
    one = np.float32(1.0)
    eps_half = np.float32(2.0) ** -24
    nxt = np.float32(1.0) + np.float32(2.0) ** -23           # f32 里 1.0 的下一个可表示值
    pol = np.zeros(POLICY_DIM, dtype=np.float32)
    pol[REGION_SELECT_BASE + 0] = one                        # 候选 0 三格：1.0, 2^-24, 2^-24
    pol[REGION_SELECT_BASE + 1] = eps_half
    pol[REGION_SELECT_BASE + 2] = eps_half
    pol[REGION_SELECT_BASE + 3] = nxt                        # 候选 1 三格：1+2^-23, 0, 0
    acts = ["-/-/RegionSelect([0; 1; 2])", "-/-/RegionSelect([3; 4; 5])"]
    scores, strict = score_actions("RegionSelect", acts, pol)
    f32_pick = argmax_logit(scores)
    f64_scores = [sum(float(pol[s]) for s in action_slots("RegionSelect", a)[1]) for a in acts]
    f64_pick = 0 if f64_scores[0] >= f64_scores[1] else 1    # f64 口径下并列取较小下标
    out.append((
        "地区三格：f32 逐项累加与 f64 求和给出不同 argmax",
        strict and f32_pick == 1 and f64_pick == 0,
        f"f32 分值 {float(scores[0]):.9g} / {float(scores[1]):.9g} → 选 {f32_pick}"
        f"；f64 分值 {f64_scores[0]:.17g} / {f64_scores[1]:.17g} → 选 {f64_pick}"
    ))
    out.append((
        "地区三格：分值类型确实是 np.float32",
        all(isinstance(v, np.float32) for v in scores),
        f"实际类型 {[type(v).__name__ for v in scores]}"
    ))

    # --- 用例 2：total_cmp 的边界，逐条与朴素 `>` 对照 ---
    pos_nan = f32_from_bits(0x7FC00000)
    neg_nan = f32_from_bits(0xFFC00000)
    for name, vals, want, want_naive in [
        ("total_cmp：-0.0 < +0.0", [np.float32(-0.0), np.float32(0.0)], 1, 0),
        ("total_cmp：+NaN 大于一切", [np.float32(1.0), pos_nan], 1, 0),
        ("total_cmp：-NaN 小于一切", [neg_nan, np.float32(1.0)], 1, 0),
        ("total_cmp：并列取较小下标", [np.float32(2.5), np.float32(2.5)], 0, 0)
    ]:
        got, nb = argmax_logit(vals), naive_argmax(vals)
        sep = "能区分两种实现" if want != want_naive else "两种实现同解（只作回归锚点）"
        out.append((name, got == want and nb == want_naive,
                    f"total_cmp 选 {got}（期望 {want}）；朴素 `>` 选 {nb}"
                    f"（期望 {want_naive}）——{sep}"))

    return out


def cmd_unit(_args):
    """合成回归：只查打分与比较器口径，不碰 GPU、不读 checkpoint。"""
    print("===== 合成回归（不需要 GPU / checkpoint）=====")
    cases = unit_cases()
    bad = 0
    for name, ok, detail in cases:
        print(f"  [{'通过' if ok else '未通过'}] {name}")
        print(f"      {detail}")
        if not ok:
            bad += 1
    print()
    print(f"===== 合成回归 {len(cases)} 项，未通过 {bad} 项 =====")
    return 1 if bad else 0


# ========================= 入口 =========================

def main():
    """解析子命令并分发；加载/预热/捕获耗时始终单列输出。"""
    ap = argparse.ArgumentParser(description=__doc__)
    # `unit` 子命令不碰 GPU 也不读 checkpoint，故这两项改成解析后按子命令校验
    ap.add_argument("--checkpoint", action="append")
    ap.add_argument("--dec-csv", action="append")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("profile", help="第一步：短 profiler，只看 eager")
    p1.add_argument("--batch", type=int, default=512)
    p1.add_argument("--steps", type=int, default=5)
    p1.add_argument("--out-dir", default="logs/gpu_probe_0915")
    p1.set_defaults(func=cmd_profile)

    p2 = sub.add_parser("verify", help="第二步：Graph 正确性")
    p2.add_argument("--batch", type=int, action="append", default=None)
    p2.add_argument("--per-stage", type=int, default=120)
    p2.set_defaults(func=cmd_verify)

    p2b = sub.add_parser("selftest", help="第二步之前：错误注入，验证验收器拦得住")
    p2b.add_argument("--batch", type=int, action="append", default=None)
    p2b.add_argument("--per-stage", type=int, default=40)
    p2b.set_defaults(func=cmd_selftest)

    p3 = sub.add_parser("time", help="第三步：稳态交错计时")
    p3.add_argument("--batch", type=int, action="append", default=None)
    p3.add_argument("--per-stage", type=int, default=120)
    p3.add_argument("--iters", type=int, default=50)
    p3.add_argument("--reps", type=int, default=5)
    p3.add_argument("--warmup", type=int, default=20)
    p3.set_defaults(func=cmd_time)

    p4 = sub.add_parser("unit", help="合成回归：打分与比较器口径，不需要 GPU")
    p4.set_defaults(func=cmd_unit)

    args = ap.parse_args()
    if args.cmd == "unit":
        sys.exit(args.func(args) or 0)
    for value, flag in ((args.checkpoint, "--checkpoint"), (args.dec_csv, "--dec-csv")):
        if not value:
            ap.error(f"子命令 {args.cmd} 需要 {flag}")
    if getattr(args, "batch", None) is None and args.cmd in ("verify", "selftest", "time"):
        args.batch = [512, 1024]
    if not torch.cuda.is_available():
        raise RuntimeError("本探针要求 CUDA；不做 CPU 回退，也不改运行环境")
    # 先冻结数值设置再打印，避免打印出尚未生效的默认值
    freeze_numerics()
    print(f"[env] python {platform.python_version()} torch {torch.__version__}"
          f" cuda {torch.version.cuda} device {torch.cuda.get_device_name(0)}"
          f" 能力 {torch.cuda.get_device_capability(0)}")
    print(f"[env] tf32 matmul={torch.backends.cuda.matmul.allow_tf32}"
          f" cudnn={torch.backends.cudnn.allow_tf32} 成员 {len(args.checkpoint)}")
    print()
    rc = args.func(args)
    sys.exit(rc or 0)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
