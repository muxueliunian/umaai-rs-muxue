"""拉面杯搜索蒸馏训练入口。

示例：

``python train.py --data training_data/npy_v1 --labels target/ramen_nn_labels_v1``

多个数据目录通过重复 ``--data``/``--labels`` 传入。划分只依赖样本 ``index``，
因此追加新分片或改变目录顺序不会让旧样本在训练集与验证集之间漂移。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.utils.data import DataLoader

try:
    from .data import (
        RamenDataset,
        ValueNormalization,
        compute_stage_weights,
        describe_split,
        fit_value_normalization,
        load_shards,
        stable_split_refs,
        subsample_train_refs,
    )
    from .eval import evaluate_model
    from .model import POLICY_DIM, ModelConfig, RamenNetwork, model_from_checkpoint
except ImportError:
    from data import (
        RamenDataset,
        ValueNormalization,
        compute_stage_weights,
        describe_split,
        fit_value_normalization,
        load_shards,
        stable_split_refs,
        subsample_train_refs,
    )
    from eval import evaluate_model
    from model import POLICY_DIM, ModelConfig, RamenNetwork, model_from_checkpoint

VALUE_COMPONENT_WEIGHTS = (0.2, 0.4, 0.2)
TRAIN_STAGE = 2
TRAIN_ACTION_START = 201
TRAIN_ACTION_COUNT = 10


def seed_everything(seed: int) -> None:
    """设置 Python/NumPy/PyTorch 随机种子。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def policy_kl_per_sample(logits: Tensor, target: Tensor, legal_mask: Tensor) -> Tensor:
    """对合法格位计算 ``KL(target || policy)``，返回 batch 内每条样本。"""

    if torch.any(~torch.any(legal_mask, dim=1)):
        raise ValueError("batch 中存在没有合法 policy 格位的样本")
    masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
    log_policy = functional.log_softmax(masked_logits, dim=1)
    safe_log_policy = torch.where(legal_mask, log_policy, torch.zeros_like(log_policy))
    positive = target > 0.0
    log_target = torch.where(positive, torch.log(torch.clamp_min(target, 1e-30)), torch.zeros_like(target))
    terms = torch.where(positive, target * (log_target - safe_log_policy), torch.zeros_like(target))
    return torch.sum(terms, dim=1)


def value_huber_per_sample(prediction: Tensor, target: Tensor, normalization: ValueNormalization) -> Tensor:
    """计算归一化后三路加权 Huber，返回每条样本。"""

    normalized_target = normalization.normalize_tensor(target)
    component = functional.smooth_l1_loss(prediction, normalized_target, reduction="none")
    weights = prediction.new_tensor(VALUE_COMPONENT_WEIGHTS)
    return torch.sum(component * weights, dim=1)


def compute_train_action_weights(shards, train_refs: np.ndarray, max_weight: float = 4.0) -> tuple[Tensor, list[int]]:
    """按 Train 阶段标签主动作计算截断逆平方根权重，并归一到均值 1。"""

    counts = np.zeros(TRAIN_ACTION_COUNT, dtype=np.int64)
    action_end = TRAIN_ACTION_START + TRAIN_ACTION_COUNT
    for shard_idx, local_idx in train_refs:
        shard = shards[int(shard_idx)]
        index = int(local_idx)
        if int(shard.stage[index]) != TRAIN_STAGE:
            continue
        action = int(np.argmax(shard.policy_target[index, TRAIN_ACTION_START:action_end]))
        counts[action] += 1
    present = counts > 0
    if not np.any(present):
        return torch.ones(TRAIN_ACTION_COUNT, dtype=torch.float32), counts.tolist()
    weights = np.full(TRAIN_ACTION_COUNT, max_weight, dtype=np.float64)
    reference = counts[present].max()
    weights[present] = np.minimum(np.sqrt(reference / counts[present]), max_weight)
    weights /= np.sum(weights[present] * counts[present]) / np.sum(counts[present])
    return torch.tensor(weights, dtype=torch.float32), counts.tolist()


def compute_batch_loss(
    model: RamenNetwork,
    batch: dict[str, Tensor],
    normalization: ValueNormalization,
    stage_weights: Tensor,
    train_action_weights: Tensor | None,
    value_loss_weight: float,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """前向并返回 ``total/policy/value`` 三个标量 loss。"""

    x = batch["x"].to(device, non_blocking=True)
    legal = batch["legal_mask"].to(device, non_blocking=True)
    policy_target = batch["policy_target"].to(device, non_blocking=True)
    value_target = batch["value_target"].to(device, non_blocking=True)
    stage = batch["stage"].to(device, non_blocking=True)
    output = model(x)
    policy, _, value = model.split_output(output)
    policy_samples = policy_kl_per_sample(policy, policy_target, legal)
    sample_weights = stage_weights[stage]
    if train_action_weights is not None:
        action_end = TRAIN_ACTION_START + TRAIN_ACTION_COUNT
        action = torch.argmax(policy_target[:, TRAIN_ACTION_START:action_end], dim=1)
        action_weight = train_action_weights[action]
        sample_weights = sample_weights * torch.where(stage == TRAIN_STAGE, action_weight, 1.0)
    policy_loss = torch.mean(policy_samples * sample_weights)
    value_loss = torch.mean(value_huber_per_sample(value, value_target, normalization))
    total = policy_loss + value_loss * value_loss_weight
    return total, policy_loss, value_loss


def run_epoch(
    model: RamenNetwork,
    loader: DataLoader,
    normalization: ValueNormalization,
    stage_weights: Tensor,
    train_action_weights: Tensor | None,
    value_loss_weight: float,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
    on_step: Callable[[], None] | None = None,
    step_budget: int | None = None,
) -> dict[str, float]:
    """执行一个训练或只读验证 epoch。

    ``on_step`` 在每次 ``optimizer.step()`` 之后调用一次，供权重 EMA、按步 LR 调度之类
    按步推进的设施使用；只读 epoch 不会触发。

    ``step_budget`` 给出本轮最多允许的 optimizer step 数，用完即在 batch 边界收尾。
    有它才能把「训到第 N 步」执行成真正的第 N 步——按轮取整会多训小半轮，两个数据量
    不同的 arm 因此拿到不同的优化预算。只读 epoch 忽略该参数。
    """

    training = optimizer is not None
    model.train(training)
    totals = np.zeros(3, dtype=np.float64)
    samples = 0
    steps = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            if training and step_budget is not None and steps >= step_budget:
                break
            if training:
                optimizer.zero_grad(set_to_none=True)
            total, policy, value = compute_batch_loss(
                model, batch, normalization, stage_weights, train_action_weights, value_loss_weight, device
            )
            if training:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                steps += 1
                if on_step is not None:
                    on_step()
            count = len(batch["stage"])
            totals += np.asarray([total.item(), policy.item(), value.item()]) * count
            samples += count
    means = totals / samples
    return {
        "loss": float(means[0]),
        "policy_kl": float(means[1]),
        "value_huber": float(means[2]),
        "steps": steps,
    }


def make_optimizer(
    model: RamenNetwork,
    lr: float,
    head_lr: float,
    weight_decay: float,
    eat_interaction_weight_decay: float,
) -> torch.optim.Adam:
    """建立 Adam；输出头低学习率且无 weight decay，使 choice 占位行保持为零。

    因子化吃面头的交互项 ``w[r,t]`` 单成一组：它与其它输出头行同用 ``head_lr``，
    但带独立（默认更强的）weight decay，把可加基线之外的自由度压向零，让稀疏的
    联合格只有在数据真的要求时才偏离 ``u_r + v_t``。非因子化时该组为空。
    """

    head, interaction = model.head_parameters()
    excluded = {id(parameter) for parameter in head} | {id(parameter) for parameter in interaction}
    trunk = [parameter for parameter in model.parameters() if id(parameter) not in excluded]
    groups: list[dict] = [
        {"params": trunk, "lr": lr, "weight_decay": weight_decay},
        {"params": head, "lr": head_lr, "weight_decay": 0.0},
    ]
    if interaction:
        groups.append({"params": interaction, "lr": head_lr, "weight_decay": eat_interaction_weight_decay})
    return torch.optim.Adam(groups)


class WeightEma:
    """按 optimizer step 维护的指数滑动平均权重（带偏差校正）。

    ``halflife_steps`` 步之后，旧权重的贡献衰减一半。影子权重从零开始累加，取用时
    除以 ``1 - decay ** t``，因此不残留初始化权重，步数还很少时也是无偏的。
    非浮点张量（若有）不参与平均，取用时直接复制当前值。
    """

    def __init__(self, model: RamenNetwork, halflife_steps: int) -> None:
        if halflife_steps <= 0:
            raise ValueError("EMA 半衰期必须为正")
        self.halflife_steps = int(halflife_steps)
        self.decay = 0.5 ** (1.0 / self.halflife_steps)
        self.steps = 0
        self.shadow = {
            name: torch.zeros_like(tensor, dtype=torch.float32)
            for name, tensor in model.state_dict().items()
            if tensor.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model: RamenNetwork) -> None:
        """吃掉一个 optimizer step 之后的权重。"""

        self.steps += 1
        for name, tensor in model.state_dict().items():
            shadow = self.shadow.get(name)
            if shadow is not None:
                shadow.mul_(self.decay).add_(tensor.detach().to(torch.float32), alpha=1.0 - self.decay)

    def model_state(self, model: RamenNetwork) -> dict:
        """产出可直接喂给 ``model_from_checkpoint`` 的权重字典。"""

        if self.steps == 0:
            raise ValueError("EMA 还没有吃到任何 optimizer step")
        correction = 1.0 - self.decay**self.steps
        state = {}
        for name, tensor in model.state_dict().items():
            shadow = self.shadow.get(name)
            state[name] = tensor.detach().clone() if shadow is None else (shadow / correction).to(tensor.dtype)
        return state


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    """先写临时文件再原子替换 checkpoint。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def make_cosine_schedule(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int, final_factor: float
) -> torch.optim.lr_scheduler.LambdaLR:
    """按 optimizer step 走的线性 warmup + 余弦退火，倍率作用于各组的初始 LR。

    与 ``ReduceLROnPlateau`` 的关键区别是**完全预声明**：LR 曲线只由步数决定，不读
    任何验证指标。留出 regret 同时驱动 scheduler 与 best.pt 时，「换了数据量」会连带
    改变 LR 下降时点，两个 arm 的差异就无法归因；而按轮计数的耐心还会让轮数不同的
    arm 拿到不同的衰减次数。曲线里的 ``warmup_steps`` 与 ``final_factor`` 仍是超参，
    只是预声明后不再随实验调整。

    倍率乘在每个参数组各自的初始 LR 上，故 trunk 与输出头的比例关系保持不变。
    """

    if total_steps <= 0:
        raise ValueError("余弦日程的总步数必须为正")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError(f"warmup 步数 {warmup_steps} 必须落在 [0, {total_steps})")
    if not 0.0 <= final_factor <= 1.0:
        raise ValueError("末端倍率必须落在 [0, 1]")

    def factor(step: int) -> float:
        """第 ``step`` 个 optimizer step（0 基）使用的 LR 倍率。"""

        if step < warmup_steps:
            # 用 step + 1 起步，首步就有非零 LR
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return final_factor + (1.0 - final_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _choose_device(name: str) -> torch.device:
    """解析 auto/cpu/cuda。"""

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _model_config(args: argparse.Namespace, samples: int) -> ModelConfig:
    """以样本量自适应配置为底，再应用显式命令行覆盖。"""

    config = ModelConfig.for_dataset(samples)
    overrides = {
        "token_dim": args.token_dim,
        "heads": args.heads,
        "encoder_blocks": args.encoder_blocks,
        "mlp_width": args.mlp_width,
        "mlp_blocks": args.mlp_blocks,
        "dropout": args.dropout,
        "attention_kind": args.attention_kind,
        "card_slot_embedding": args.card_slot_embedding or None,
        "factorized_eat_head": args.factorized_eat_head or None,
    }
    for name, value in overrides.items():
        if value is not None:
            config = replace(config, **{name: value})
    config.validate()
    return config


def _parse_args() -> argparse.Namespace:
    """解析训练命令。"""

    parser = argparse.ArgumentParser(description="训练拉面杯搜索蒸馏网络")
    parser.add_argument("--data", type=Path, action="append", required=True, help="reduced .npy 目录；可重复")
    parser.add_argument("--labels", type=Path, action="append", required=True, help="对应标签目录；可重复")
    parser.add_argument("--output-dir", type=Path, default=Path("saved_models/ramen"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--head-lr", type=float, default=1.25e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-5)
    parser.add_argument(
        "--eat-interaction-weight-decay",
        type=float,
        default=1e-3,
        help="因子化吃面头交互项 w[r,t] 的 weight decay；仅在 --factorized-eat-head 下生效",
    )
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--train-action-reweight",
        action="store_true",
        help="仅对 Train 阶段按 policy 软标签主动作施加截断逆平方根样本权重（上限 4）",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        help="把训练集稳定截到这么多条（数据量曲线实验用）；验证集不变，各点留出指标可比",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=("plateau", "cosine"),
        default="plateau",
        help="plateau（默认，旧行为）：ReduceLROnPlateau，由每轮的留出 regret 驱动；"
        "cosine：按 optimizer step 的线性 warmup + 余弦退火，完全预声明、不读验证指标。"
        "对照实验必须用 cosine——plateau 让「数据量」与「LR 下降时点」纠缠在一起，"
        "而且它按轮计数，轮数不同的 arm 衰减次数也不同",
    )
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=300,
        help="--lr-schedule cosine 的线性 warmup 步数",
    )
    parser.add_argument(
        "--lr-final-factor",
        type=float,
        default=0.02,
        help="--lr-schedule cosine 末端 LR 相对各参数组初始 LR 的倍率",
    )
    parser.add_argument("--patience", type=int)
    parser.add_argument("--min-regret-improvement", type=float, default=0.5)
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="不因留出后悔值停滞而提前停止，一直训到轮数上限。方差实验要求每个种子跑"
        "同样的步数，否则「只换初始化」的对照里混进了不同长度的训练日程",
    )
    parser.add_argument(
        "--eval-columns",
        type=int,
        nargs=2,
        metavar=("LO", "HI"),
        help="每轮额外算一遍只用 rollout 列 [LO, HI) 的留出指标，记进 metrics.jsonl 的"
        "`evaluation_a`。**只记录，不参与早停与 LR 调度**——默认的全列口径会让被结算的"
        "列参与模型选择，这个字段是用来量化那件事有多严重的",
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        metavar="STEP",
        help="在这些 optimizer step 处额外存 `step_XXXXXX.pt`（落到刚跨过该步数的轮末）。"
        "用于在同一条训练轨迹上取多个点，把「同种子内的抖动」与「种子间的差异」分开",
    )
    parser.add_argument(
        "--ema-halflife-steps",
        type=int,
        nargs="+",
        metavar="STEP",
        help="维护这些半衰期（按 optimizer step）的权重 EMA，在 --checkpoint-steps 处与"
        "训练结束时一并存成 `*_ema<半衰期>.pt`。不影响训练本身",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260830,
        help="总种子；--split-seed / --init-seed 未给出时两者都取它，等价于旧行为",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        help="只决定**样本身份**：train/val 划分与 --max-train-samples 的抽稀。"
        "把它固定住、只变 --init-seed，多次训练就跑在同一份训练集上，"
        "散布只反映初始化与优化路径。数据集与划分是固定资产时，这才是正确的估计目标",
    )
    parser.add_argument(
        "--init-seed",
        type=int,
        help="只决定**优化路径**：参数初始化、dropout、minibatch 顺序。不影响样本身份",
    )
    parser.add_argument(
        "--patience-steps",
        type=int,
        help="按 optimizer step 计的早停耐心（换算成整数轮：patience_steps / 每轮步数）。"
        "给出时覆盖 --patience。按轮计数在数据量变化时等价步数会跟着变，"
        "学习曲线类实验必须用本参数，否则不同数据量点的训练时长口径不一致",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="按 optimizer step 计的训练上限，**精确截到该步**（在 batch 边界收尾）。"
        "给出时覆盖 --epochs。最后一轮通常是半轮，其 train 指标只覆盖实跑的那些 batch",
    )
    parser.add_argument("--token-dim", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--encoder-blocks", type=int)
    parser.add_argument("--mlp-width", type=int)
    parser.add_argument("--mlp-blocks", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--attention-kind", choices=("softmax", "simple"))
    parser.add_argument(
        "--factorized-eat-head",
        action="store_true",
        help="吃面联合格 [1,201) 改由 u_r + v_t + w_(r,t) 产生（地区 20 + 用法 10 + 交互 200）。"
        "联合格监督稀疏（中位每格 17 条）而按地区聚合有 222 条，因子化让地区与用法效应"
        "各自吃到全部数据。输出维度与布局不变，Rust 侧无需改动",
    )
    parser.add_argument(
        "--card-slot-embedding",
        action="store_true",
        help="给卡片 token 加槽位 embedding。默认关闭——卡组顺序在游戏里没有含义，"
        "而训练数据的槽位与卡片类型完全相关，开启会让模型记顺序而非读属性。仅作消融用",
    )
    parser.add_argument(
        "--split-by",
        choices=("combo", "sample"),
        default="combo",
        help="留出集切分粒度。combo 按 (马娘, 卡组) 组合切，留出的组合完全不参与训练；"
        "sample 按样本切，同一套卡组会同时进训练与验证，验证指标偏乐观",
    )
    return parser.parse_args()


def main() -> None:
    """训练主流程。"""

    args = _parse_args()
    if len(args.data) != len(args.labels):
        raise ValueError("--data 与 --labels 数量必须一致")
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("epochs/batch-size 必须为正，workers 不得为负")
    # 两条随机轴分开：样本身份 vs 优化路径。未显式给出时都回落到 --seed，
    # 因此不传新参数的旧命令行为逐位不变。
    split_seed_arg = args.seed if args.split_seed is None else args.split_seed
    init_seed = args.seed if args.init_seed is None else args.init_seed
    seed_everything(init_seed)
    device = _choose_device(args.device)
    shards = load_shards(args.data, args.labels)
    resume_checkpoint = None
    split_fraction = args.validation_fraction
    split_seed = split_seed_arg
    split_by = args.split_by
    if args.resume is not None:
        resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved_split = resume_checkpoint.get("split", {})
        split_fraction = float(saved_split.get("validation_fraction", split_fraction))
        split_seed = int(saved_split.get("seed", split_seed))
        # split_by 加入前的 checkpoint 一律是按样本切的
        split_by = str(saved_split.get("split_by", "sample"))
    train_refs, validation_refs = stable_split_refs(shards, split_fraction, split_seed, split_by)
    full_train_size = int(len(train_refs))
    if args.max_train_samples is not None:
        # 抽稀属于「样本身份」，走 split_seed：固定它就能让不同 init_seed 的多次训练
        # 落在同一份训练子集上，散布只来自优化路径。同一 split_seed 下各数据量点严格嵌套
        train_refs = subsample_train_refs(shards, train_refs, args.max_train_samples, split_seed)
    split_summary = describe_split(shards, train_refs, validation_refs)
    stage_weights, stage_counts = compute_stage_weights(shards, train_refs)
    stage_weights = stage_weights.to(device)
    train_action_weights = None
    train_action_counts = None
    if args.train_action_reweight:
        train_action_weights, train_action_counts = compute_train_action_weights(shards, train_refs)
        train_action_weights = train_action_weights.to(device)

    if resume_checkpoint is not None:
        model = model_from_checkpoint(resume_checkpoint, device)
        normalization = ValueNormalization.from_dict(resume_checkpoint["value_normalization"])
    else:
        config = _model_config(args, len(train_refs))
        model = RamenNetwork(config).to(device)
        normalization = fit_value_normalization(shards, train_refs)

    optimizer = make_optimizer(
        model, args.lr, args.head_lr, args.weight_decay, args.eat_interaction_weight_decay
    )
    # 每轮的 optimizer step 数。DataLoader 不丢尾批，故向上取整。
    steps_per_epoch = max(1, math.ceil(len(train_refs) / min(args.batch_size, max(1, len(train_refs)))))
    patience = args.patience if args.patience is not None else (20 if len(train_refs) < 25_000 else 12)
    epochs = args.epochs
    if args.patience_steps is not None:
        # 按轮计数时，数据量翻倍会让同样的「12 轮」变成两倍的优化步数，
        # 学习曲线上各点的训练时长口径因此不一致。换算成轮数即可对齐：
        # 评估本来就只在每轮末做一次，耐心的分辨率上限就是一轮。
        patience = max(1, round(args.patience_steps / steps_per_epoch))
    # step_cap 是硬上限，epochs 只是「够跑到那么多步」的轮数上界；真正的截断在 batch 边界。
    step_cap = args.max_steps
    if step_cap is not None:
        if step_cap <= 0:
            raise ValueError("--max-steps 必须为正")
        epochs = max(1, math.ceil(step_cap / steps_per_epoch))
    total_steps = step_cap if step_cap is not None else epochs * steps_per_epoch
    print(
        f"每轮 {steps_per_epoch} 步；早停耐心 {patience} 轮（约 {patience * steps_per_epoch} 步）；"
        f"上限 {epochs} 轮 / {total_steps} 步"
        + ("（精确截断）" if step_cap is not None else "")
    )
    # 两种日程互斥：plateau 每轮吃一次 regret，cosine 每步推进一格、不读任何验证指标。
    plateau_scheduler = None
    step_scheduler = None
    if args.lr_schedule == "cosine":
        step_scheduler = make_cosine_schedule(
            optimizer, total_steps, args.lr_warmup_steps, args.lr_final_factor
        )
        print(
            f"LR 日程 cosine：warmup {args.lr_warmup_steps} 步，末端倍率 {args.lr_final_factor}，"
            f"总步数 {total_steps}"
        )
        if not args.no_early_stop:
            print("⚠ cosine 与早停同用：提前停止会让曲线没退火完，对照实验请加 --no-early-stop")
    else:
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(2, patience // 3), min_lr=1e-6
        )
    scheduler = step_scheduler if step_scheduler is not None else plateau_scheduler
    start_epoch = 0
    start_step = 0
    best_regret = float("inf")
    stale_epochs = 0
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        if "scheduler_state" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        # 老 checkpoint 没记步数，只能按整轮回推（那时也确实是整轮训的）
        start_step = int(resume_checkpoint.get("global_step", start_epoch * steps_per_epoch))
        best_regret = float(resume_checkpoint.get("best_regret", best_regret))
        stale_epochs = int(resume_checkpoint.get("stale_epochs", 0))

    # 以下三项都是诊断设施：不改变梯度、不参与早停与 LR 调度，只是多记录/多存盘。
    eval_columns = None
    if args.eval_columns is not None:
        eval_columns = (int(args.eval_columns[0]), int(args.eval_columns[1]))
        available = min(shard.rollout_columns for shard in shards)
        if available == 0:
            raise ValueError("--eval-columns 需要 --raw 导出的数据目录（缺少 cand_scores.npy）")
        if not 0 <= eval_columns[0] < eval_columns[1] <= available:
            raise ValueError(f"--eval-columns {eval_columns} 越出可用的 {available} 列")
    checkpoint_steps = sorted({int(v) for v in (args.checkpoint_steps or [])})
    if any(step <= 0 for step in checkpoint_steps):
        raise ValueError("--checkpoint-steps 必须为正")
    emas = {int(h): WeightEma(model, int(h)) for h in sorted({int(v) for v in (args.ema_halflife_steps or [])})}
    if emas and args.resume is not None:
        # EMA 状态不进 checkpoint，续训会从零重新累积，与一次跑完的轨迹不同。
        raise ValueError("--ema-halflife-steps 目前不支持与 --resume 同用")

    def on_step() -> None:
        """每个 optimizer step 之后：推进全部 EMA，并把按步的 LR 日程推进一格。"""

        for ema in emas.values():
            ema.update(model)
        if step_scheduler is not None:
            step_scheduler.step()

    generator = torch.Generator().manual_seed(init_seed)
    train_dataset = RamenDataset(shards, train_refs)
    validation_dataset = RamenDataset(shards, validation_refs)
    common_loader = {
        "batch_size": min(args.batch_size, len(train_dataset)),
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **common_loader)
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        batch_size=min(args.batch_size, len(validation_dataset)),
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_info = {
        "device": str(device),
        "model_config": model.config.to_dict(),
        "parameters": model.parameter_count(),
        "value_normalization": normalization.to_dict(),
        "value_component_weights": list(VALUE_COMPONENT_WEIGHTS),
        "stage_weights": stage_weights.cpu().tolist(),
        "stage_counts": stage_counts,
        "train_action_weights": None if train_action_weights is None else train_action_weights.cpu().tolist(),
        "train_action_counts": train_action_counts,
        "split": split_summary,
        "seeds": {"seed": args.seed, "split_seed": split_seed, "init_seed": init_seed},
        "schedule": {
            "steps_per_epoch": steps_per_epoch,
            "patience_epochs": patience,
            "patience_steps_arg": args.patience_steps,
            "max_epochs": epochs,
            "max_steps_arg": args.max_steps,
            "total_steps": total_steps,
            "batch_size": args.batch_size,
            "early_stop": not args.no_early_stop,
            "lr_schedule": args.lr_schedule,
            "lr_warmup_steps": args.lr_warmup_steps if args.lr_schedule == "cosine" else None,
            "lr_final_factor": args.lr_final_factor if args.lr_schedule == "cosine" else None,
        },
        "diagnostics": {
            "eval_columns": None if eval_columns is None else list(eval_columns),
            "checkpoint_steps": checkpoint_steps,
            "ema_halflife_steps": sorted(emas),
        },
    }
    (args.output_dir / "run.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_info, ensure_ascii=False, indent=2))

    metrics_path = args.output_dir / "metrics.jsonl"
    completed_steps = start_step
    previous_step = completed_steps
    needs_on_step = bool(emas) or step_scheduler is not None
    checkpoint: dict | None = None
    for epoch in range(start_epoch, epochs):
        started = time.perf_counter()
        budget = None if step_cap is None else step_cap - completed_steps
        if budget is not None and budget <= 0:
            break
        train_metrics = run_epoch(
            model,
            train_loader,
            normalization,
            stage_weights,
            train_action_weights,
            args.value_loss_weight,
            device,
            optimizer,
            args.grad_clip,
            on_step if needs_on_step else None,
            budget,
        )
        completed_steps += int(train_metrics["steps"])
        validation_loss = run_epoch(
            model,
            validation_loader,
            normalization,
            stage_weights,
            train_action_weights,
            args.value_loss_weight,
            device,
            None,
            args.grad_clip,
        )
        evaluation = evaluate_model(
            model, shards, validation_refs, normalization, device, args.batch_size, args.workers
        )
        regret = float(evaluation["overall"]["expected_regret"])
        if plateau_scheduler is not None:
            plateau_scheduler.step(regret)
        improved = regret < best_regret - args.min_regret_improvement
        if improved:
            best_regret = regret
            stale_epochs = 0
        else:
            stale_epochs += 1

        global_step = completed_steps
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "seconds": time.perf_counter() - started,
            "lr": [group["lr"] for group in optimizer.param_groups],
            "train": train_metrics,
            "validation": validation_loss,
            "evaluation": evaluation,
            "best_regret": best_regret,
            "stale_epochs": stale_epochs,
        }
        if eval_columns is not None:
            record["evaluation_a"] = evaluate_model(
                model, shards, validation_refs, normalization, device, args.batch_size, args.workers, eval_columns
            )
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

        checkpoint = {
            "format_version": 1,
            "epoch": epoch,
            "global_step": global_step,
            "model_config": model.config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "value_normalization": normalization.to_dict(),
            "split": {
                "validation_fraction": split_fraction,
                "seed": split_seed,
                "split_by": split_by,
                "full_train_size": full_train_size,
                "max_train_samples": args.max_train_samples,
            },
            "best_regret": best_regret,
            "stale_epochs": stale_epochs,
            "run_info": run_info,
        }
        save_checkpoint(args.output_dir / "last.pt", checkpoint)
        if improved:
            save_checkpoint(args.output_dir / "best.pt", checkpoint)

        # 定步 checkpoint：落到刚跨过该步数的轮末，文件名仍用请求的步数，
        # 使不同数据量/批大小下的同名文件对应同样的优化步数。
        due = [step for step in checkpoint_steps if previous_step < step <= global_step]
        for step in due:
            save_checkpoint(args.output_dir / f"step_{step:06d}.pt", checkpoint)
            for halflife, ema in emas.items():
                save_checkpoint(
                    args.output_dir / f"step_{step:06d}_ema{halflife}.pt",
                    {**checkpoint, "model_state": ema.model_state(model), "ema": {"halflife_steps": halflife, "steps": ema.steps}},
                )
        previous_step = global_step

        if step_cap is not None and completed_steps >= step_cap:
            print(f"已训满 --max-steps {step_cap} 步，停止。")
            break
        if args.no_early_stop:
            continue
        if stale_epochs >= patience:
            print(f"验证集期望后悔值连续 {patience} 轮未改善，提前停止。")
            break

    if checkpoint is not None:
        for halflife, ema in emas.items():
            save_checkpoint(
                args.output_dir / f"last_ema{halflife}.pt",
                {**checkpoint, "model_state": ema.model_state(model), "ema": {"halflife_steps": halflife, "steps": ema.steps}},
            )


if __name__ == "__main__":
    main()
