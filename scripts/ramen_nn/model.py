"""拉面杯结构化 set-attention 模型。

模型只接收冻结的 754 维特征并输出冻结的 245 维向量。注意力手写为基础张量
运算，避免 ``nn.MultiheadAttention`` 在 ONNX 中按 PyTorch 版本展开成不同算子。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

INPUT_DIM = 754
GLOBAL_DIM = 154
CARD_COUNT = 6
CARD_DIM = 35
PERSON_COUNT = 13
PERSON_DIM = 30
PERSON_PRESENT_INDEX = 15

POLICY_DIM = 234
CHOICE_DIM = 8
VALUE_DIM = 3
OUTPUT_DIM = POLICY_DIM + CHOICE_DIM + VALUE_DIM


@dataclass(frozen=True)
class ModelConfig:
    """可序列化的模型结构配置。"""

    token_dim: int = 96
    heads: int = 4
    encoder_blocks: int = 2
    mlp_width: int = 256
    mlp_blocks: int = 2
    dropout: float = 0.08
    attention_kind: str = "simple"
    card_slot_embedding: bool = False

    def validate(self) -> None:
        """检查会影响 reshape 与导出的结构不变量。"""

        if self.token_dim <= 0 or self.token_dim % self.heads != 0:
            raise ValueError("token_dim 必须为正且能被 heads 整除")
        if self.encoder_blocks < 0 or self.mlp_blocks < 0 or self.mlp_width <= 0:
            raise ValueError("block 数不得为负，mlp_width 必须为正")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1)")
        if self.attention_kind not in ("softmax", "simple"):
            raise ValueError("attention_kind 必须是 softmax 或 simple")
        if not isinstance(self.card_slot_embedding, bool):
            raise ValueError("card_slot_embedding 必须是 bool")

    @classmethod
    def for_dataset(cls, samples: int) -> "ModelConfig":
        """按样本量选择保守容量，不把 pilot 的 12000 写死进结构。"""

        if samples < 25_000:
            return cls(token_dim=64, heads=4, encoder_blocks=1, mlp_width=192, mlp_blocks=2, dropout=0.15)
        if samples < 120_000:
            return cls(token_dim=96, heads=4, encoder_blocks=2, mlp_width=256, mlp_blocks=2, dropout=0.08)
        return cls(token_dim=128, heads=4, encoder_blocks=2, mlp_width=384, mlp_blocks=3, dropout=0.05)

    def to_dict(self) -> dict[str, int | float]:
        """转成 checkpoint 可保存的普通字典。"""

        return asdict(self)


class TokenEmbedding(nn.Module):
    """对同类 token 共享参数的两层投影。"""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """投影最后一维，保留前导 batch/token 维。"""

        return self.layers(x)


class ConservativeSelfAttention(nn.Module):
    """仅由保守 ONNX 算子组成的多头 softmax self-attention。"""

    def __init__(self, dim: int, heads: int, sequence_len: int, dropout: float, kind: str) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.sequence_len = sequence_len
        self.scale = self.head_dim**-0.5
        self.kind = kind
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.project = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, token_mask: Tensor) -> Tensor:
        """执行注意力；``token_mask`` 为 ``[B, sequence_len]`` 的 0/1 浮点值。"""

        q = self.query(x).reshape(-1, self.sequence_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.key(x).reshape(-1, self.sequence_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.value(x).reshape(-1, self.sequence_len, self.heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.kind == "softmax":
            # 写成 Mul + Add，避免导出一个没有必要的 Sub 算子。
            key_bias = token_mask * 10_000.0 + -10_000.0
            logits = logits + key_bias[:, None, None, :]
            attention = torch.softmax(logits, dim=-1)
        else:
            # 上游简化注意力的多头版本：无 softmax，以 ReLU 相似度除以有效 token 数。
            attention = torch.relu(logits) * token_mask[:, None, None, :]
            count = torch.sum(token_mask, dim=1, keepdim=True)
            attention = attention / count[:, None, :, None]
        mixed = torch.matmul(attention, v)
        mixed = mixed.transpose(1, 2).reshape(-1, self.sequence_len, self.dim)
        return self.dropout(self.project(mixed))


class AttentionBlock(nn.Module):
    """无归一化、缩放残差的轻量 attention + ReLU FFN 块。"""

    def __init__(self, dim: int, heads: int, sequence_len: int, dropout: float, kind: str) -> None:
        super().__init__()
        self.attention = ConservativeSelfAttention(dim, heads, sequence_len, dropout, kind)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.residual_scale = 0.5
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """缩小残差分支末层初始化，避免无 LayerNorm 时早期激活爆炸。"""

        with torch.no_grad():
            self.attention.project.weight.mul_(0.1)
            last = self.feed_forward[3]
            last.weight.mul_(0.1)
            if last.bias is not None:
                last.bias.zero_()

    def forward(self, x: Tensor, token_mask: Tensor) -> Tensor:
        """执行块并在每个残差后重新清零未登场 person token。"""

        expanded_mask = token_mask[:, :, None]
        x = (x + self.attention(x, token_mask) * self.residual_scale) * expanded_mask
        x = (x + self.feed_forward(x) * self.residual_scale) * expanded_mask
        return x


class ResidualMlpBlock(nn.Module):
    """推理时只含 Gemm/ReLU/Add/Mul 的 ResMLP 块。"""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.Dropout(dropout),
        )
        self.scale = 0.5
        with torch.no_grad():
            last = self.branch[3]
            last.weight.mul_(0.1)
            if last.bias is not None:
                last.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        """执行缩放残差。"""

        return torch.relu(x + self.branch(x) * self.scale)


class RamenNetwork(nn.Module):
    """global/card/person 结构化编码器与冻结的单一 245 维输出头。"""

    sequence_len = 1 + CARD_COUNT + PERSON_COUNT

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.config.validate()
        dim = self.config.token_dim
        self.global_embedding = TokenEmbedding(GLOBAL_DIM, dim)
        self.card_embedding = TokenEmbedding(CARD_DIM, dim)
        self.person_embedding = TokenEmbedding(PERSON_DIM, dim)
        # 卡组槽位在游戏里没有含义，Rust 侧 features.rs 也刻意不编槽位下标
        # （模块头注明 cards 是「置换等变序列」）。而 enumerate_decks 是逐类型
        # 拼接生成卡组的，训练数据里槽位下标与卡片类型完全相关，加了槽位
        # embedding 模型就能走捷径记顺序；换一个卡组顺序（例如 bench_config
        # 的速/智/耐/速/速/友）当场落到分布外。默认关闭，仅保留做消融。
        if self.config.card_slot_embedding:
            self.card_slot_embedding = nn.Parameter(torch.empty(CARD_COUNT, dim))
        else:
            self.register_parameter("card_slot_embedding", None)
        self.token_type_embedding = nn.Parameter(torch.empty(3, dim))
        self.encoder = nn.ModuleList(
            AttentionBlock(dim, self.config.heads, self.sequence_len, self.config.dropout, self.config.attention_kind)
            for _ in range(self.config.encoder_blocks)
        )
        self.trunk_input = nn.Sequential(nn.Linear(dim * 3, self.config.mlp_width), nn.ReLU())
        self.trunk = nn.Sequential(
            *(ResidualMlpBlock(self.config.mlp_width, self.config.dropout) for _ in range(self.config.mlp_blocks))
        )
        self.output = nn.Linear(self.config.mlp_width, OUTPUT_DIM)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化槽位/类型 embedding，并把不训练的 choice 行固定在零起点。"""

        if self.card_slot_embedding is not None:
            nn.init.normal_(self.card_slot_embedding, std=0.02)
        nn.init.normal_(self.token_type_embedding, std=0.02)
        with torch.no_grad():
            self.output.weight[POLICY_DIM : POLICY_DIM + CHOICE_DIM].zero_()
            self.output.bias[POLICY_DIM : POLICY_DIM + CHOICE_DIM].zero_()

    def _split_input(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """按 Rust ``features.rs`` 的冻结布局切分输入。"""

        global_x = x[:, :GLOBAL_DIM]
        card_end = GLOBAL_DIM + CARD_COUNT * CARD_DIM
        cards = x[:, GLOBAL_DIM:card_end].reshape(-1, CARD_COUNT, CARD_DIM)
        persons = x[:, card_end:].reshape(-1, PERSON_COUNT, PERSON_DIM)
        return global_x, cards, persons

    def forward(self, x: Tensor) -> Tensor:
        """从 ``[B,754]`` 产生 ``[B,245]``；batch 维可动态导出。"""

        global_x, cards, persons = self._split_input(x)
        person_mask = persons[:, :, PERSON_PRESENT_INDEX]
        one = person_mask[:, :1] * 0.0 + 1.0
        prefix_mask = torch.cat([one, one, one, one, one, one, one], dim=1)
        token_mask = torch.cat([prefix_mask, person_mask], dim=1)

        global_token = self.global_embedding(global_x)[:, None, :] + self.token_type_embedding[0]
        card_tokens = self.card_embedding(cards) + self.token_type_embedding[1]
        if self.card_slot_embedding is not None:
            card_tokens = card_tokens + self.card_slot_embedding[None, :, :]
        person_tokens = (self.person_embedding(persons) + self.token_type_embedding[2]) * person_mask[:, :, None]
        tokens = torch.cat([global_token, card_tokens, person_tokens], dim=1)
        for block in self.encoder:
            tokens = block(tokens, token_mask)

        global_pool = tokens[:, 0, :]
        card_pool = torch.mean(tokens[:, 1 : 1 + CARD_COUNT, :], dim=1)
        person_values = tokens[:, 1 + CARD_COUNT :, :]
        person_count = torch.sum(person_mask, dim=1, keepdim=True) + 1e-6
        person_pool = torch.sum(person_values, dim=1) / person_count
        hidden = self.trunk_input(torch.cat([global_pool, card_pool, person_pool], dim=1))
        hidden = self.trunk(hidden)
        return self.output(hidden)

    @staticmethod
    def split_output(output: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """切分 policy/choice/value，保持单一 Linear 输出契约。"""

        policy = output[:, :POLICY_DIM]
        choice = output[:, POLICY_DIM : POLICY_DIM + CHOICE_DIM]
        value = output[:, POLICY_DIM + CHOICE_DIM :]
        return policy, choice, value

    def parameter_count(self) -> int:
        """返回可训练参数总数。"""

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def model_from_checkpoint(checkpoint: dict, device: torch.device | str = "cpu") -> RamenNetwork:
    """从训练 checkpoint 重建模型并载入参数。"""

    config_values = dict(checkpoint["model_config"])
    # attention_kind 加入前的 checkpoint 全部由当时唯一的 softmax 实现产生。
    config_values.setdefault("attention_kind", "softmax")
    # card_slot_embedding 加入前的 checkpoint 一律带槽位 embedding
    config_values.setdefault("card_slot_embedding", True)
    config = ModelConfig(**config_values)
    model = RamenNetwork(config)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device)
