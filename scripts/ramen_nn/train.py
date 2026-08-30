"""拉面杯搜索蒸馏训练入口。

示例：

``python train.py --data training_data/npy_v1 --labels target/ramen_nn_labels_v1``

多个数据目录通过重复 ``--data``/``--labels`` 传入。划分只依赖样本 ``index``，
因此追加新分片或改变目录顺序不会让旧样本在训练集与验证集之间漂移。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
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
    )
    from eval import evaluate_model
    from model import POLICY_DIM, ModelConfig, RamenNetwork, model_from_checkpoint

VALUE_COMPONENT_WEIGHTS = (0.2, 0.4, 0.2)


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


def compute_batch_loss(
    model: RamenNetwork,
    batch: dict[str, Tensor],
    normalization: ValueNormalization,
    stage_weights: Tensor,
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
    policy_loss = torch.mean(policy_samples * stage_weights[stage])
    value_loss = torch.mean(value_huber_per_sample(value, value_target, normalization))
    total = policy_loss + value_loss * value_loss_weight
    return total, policy_loss, value_loss


def run_epoch(
    model: RamenNetwork,
    loader: DataLoader,
    normalization: ValueNormalization,
    stage_weights: Tensor,
    value_loss_weight: float,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
) -> dict[str, float]:
    """执行一个训练或只读验证 epoch。"""

    training = optimizer is not None
    model.train(training)
    totals = np.zeros(3, dtype=np.float64)
    samples = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            if training:
                optimizer.zero_grad(set_to_none=True)
            total, policy, value = compute_batch_loss(
                model, batch, normalization, stage_weights, value_loss_weight, device
            )
            if training:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            count = len(batch["stage"])
            totals += np.asarray([total.item(), policy.item(), value.item()]) * count
            samples += count
    means = totals / samples
    return {"loss": float(means[0]), "policy_kl": float(means[1]), "value_huber": float(means[2])}


def make_optimizer(model: RamenNetwork, lr: float, head_lr: float, weight_decay: float) -> torch.optim.Adam:
    """建立 Adam；输出头低学习率且无 weight decay，使 choice 占位行保持为零。"""

    head_ids = {id(parameter) for parameter in model.output.parameters()}
    trunk = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    return torch.optim.Adam(
        [
            {"params": trunk, "lr": lr, "weight_decay": weight_decay},
            {"params": list(model.output.parameters()), "lr": head_lr, "weight_decay": 0.0},
        ]
    )


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    """先写临时文件再原子替换 checkpoint。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


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
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--min-regret-improvement", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--token-dim", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--encoder-blocks", type=int)
    parser.add_argument("--mlp-width", type=int)
    parser.add_argument("--mlp-blocks", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--attention-kind", choices=("softmax", "simple"))
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
    seed_everything(args.seed)
    device = _choose_device(args.device)
    shards = load_shards(args.data, args.labels)
    resume_checkpoint = None
    split_fraction = args.validation_fraction
    split_seed = args.seed
    split_by = args.split_by
    if args.resume is not None:
        resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved_split = resume_checkpoint.get("split", {})
        split_fraction = float(saved_split.get("validation_fraction", split_fraction))
        split_seed = int(saved_split.get("seed", split_seed))
        # split_by 加入前的 checkpoint 一律是按样本切的
        split_by = str(saved_split.get("split_by", "sample"))
    train_refs, validation_refs = stable_split_refs(shards, split_fraction, split_seed, split_by)
    split_summary = describe_split(shards, train_refs, validation_refs)
    stage_weights, stage_counts = compute_stage_weights(shards, train_refs)
    stage_weights = stage_weights.to(device)

    if resume_checkpoint is not None:
        model = model_from_checkpoint(resume_checkpoint, device)
        normalization = ValueNormalization.from_dict(resume_checkpoint["value_normalization"])
    else:
        config = _model_config(args, len(train_refs))
        model = RamenNetwork(config).to(device)
        normalization = fit_value_normalization(shards, train_refs)

    optimizer = make_optimizer(model, args.lr, args.head_lr, args.weight_decay)
    patience = args.patience if args.patience is not None else (20 if len(train_refs) < 25_000 else 12)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(2, patience // 3), min_lr=1e-6
    )
    start_epoch = 0
    best_regret = float("inf")
    stale_epochs = 0
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        if "scheduler_state" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_regret = float(resume_checkpoint.get("best_regret", best_regret))
        stale_epochs = int(resume_checkpoint.get("stale_epochs", 0))

    generator = torch.Generator().manual_seed(args.seed)
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
        "split": split_summary,
    }
    (args.output_dir / "run.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_info, ensure_ascii=False, indent=2))

    metrics_path = args.output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        started = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            normalization,
            stage_weights,
            args.value_loss_weight,
            device,
            optimizer,
            args.grad_clip,
        )
        validation_loss = run_epoch(
            model,
            validation_loader,
            normalization,
            stage_weights,
            args.value_loss_weight,
            device,
            None,
            args.grad_clip,
        )
        evaluation = evaluate_model(
            model, shards, validation_refs, normalization, device, args.batch_size, args.workers
        )
        regret = float(evaluation["overall"]["expected_regret"])
        scheduler.step(regret)
        improved = regret < best_regret - args.min_regret_improvement
        if improved:
            best_regret = regret
            stale_epochs = 0
        else:
            stale_epochs += 1

        record = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            "lr": [group["lr"] for group in optimizer.param_groups],
            "train": train_metrics,
            "validation": validation_loss,
            "evaluation": evaluation,
            "best_regret": best_regret,
            "stale_epochs": stale_epochs,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

        checkpoint = {
            "format_version": 1,
            "epoch": epoch,
            "model_config": model.config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "value_normalization": normalization.to_dict(),
            "split": {"validation_fraction": split_fraction, "seed": split_seed, "split_by": split_by},
            "best_regret": best_regret,
            "stale_epochs": stale_epochs,
            "run_info": run_info,
        }
        save_checkpoint(args.output_dir / "last.pt", checkpoint)
        if improved:
            save_checkpoint(args.output_dir / "best.pt", checkpoint)
        if stale_epochs >= patience:
            print(f"验证集期望后悔值连续 {patience} 轮未改善，提前停止。")
            break


if __name__ == "__main__":
    main()
