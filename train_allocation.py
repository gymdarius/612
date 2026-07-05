from __future__ import annotations

"""训练 Version1 目标分配 Transformer。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.

对外训练/推理接口使用原始实体输入:
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    x_missile, x_aircraft, x_target, mask_m, mask_a, mask_t。
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.

内部流程:
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    1. SituationMLP 预测 S_pred。
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    2. ThreatMLP 预测 ThreatW_pred。
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    3. AllocationCoreTransformer 使用 S_pred + ThreatW_pred 输出目标 logits。
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.

输出没有 UNASSIGNED 类别；有效导弹的目标序号由 argmax(logits) 得到。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
"""

import argparse
import math
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

from dataset import AirCombatDataset
from shape_config import M_MAX, T_MAX
from train_situation import SituationMLP
from train_threat import ThreatMLP


class TransformerBlock(nn.Module):
    """简化 Transformer Encoder block，用于导弹/目标 token 之间的信息交互。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    def __init__(self, d_model: int = 64, num_heads: int = 4, ff_dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, token_mask):
        # token_mask 同时屏蔽 padding 导弹 token 和 padding 目标 token。
        # English note: This comment explains the following implementation detail for offline readability.
        batch_size, token_count, d_model = tokens.shape
        qkv = self.qkv(tokens)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        scores = scores.masked_fill(token_mask[:, None, None, :] == 0, -1.0e4)
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, token_count, d_model)
        tokens = self.norm1(tokens + self.dropout(self.out_proj(context)))
        tokens = self.norm2(tokens + self.dropout(self.ffn(tokens)))
        return tokens * token_mask.unsqueeze(-1)


class AllocationCoreTransformer(nn.Module):
    """
    Core allocation model.

    Input is the same intermediate representation as Version1's business
    allocation step: situation matrix S and target threat weights.
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.missile_proj = nn.Linear(T_MAX * 2, d_model)
        self.target_proj = nn.Linear(M_MAX + 1, d_model)
        self.missile_pos = nn.Parameter(torch.zeros(1, M_MAX, d_model))
        self.target_pos = nn.Parameter(torch.zeros(1, T_MAX, d_model))
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(float(d_model))
        nn.init.normal_(self.missile_pos, std=0.02)
        nn.init.normal_(self.target_pos, std=0.02)

    def forward(self, situation, threat_weights, mask_m, mask_t):
        """从中间特征 S/ThreatW 预测每枚导弹选择各目标的 logits。"""
        # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
        if situation.shape[1] != M_MAX or situation.shape[2] != T_MAX:
            raise ValueError(f"Expected situation shape [B,{M_MAX},{T_MAX}], got {tuple(situation.shape)}.")
        if threat_weights.shape[1] != T_MAX:
            raise ValueError(f"Expected threat_weights shape [B,{T_MAX}], got {tuple(threat_weights.shape)}.")

        # 导弹 token 看本导弹对所有目标的态势分数，以及全局目标威胁权重。
        # English note: This comment explains the following implementation detail for offline readability.
        missile_features = torch.cat([situation, threat_weights.unsqueeze(1).expand(-1, M_MAX, -1)], dim=-1)
        # 目标 token 看所有导弹对该目标的态势分数，以及该目标威胁权重。
        # English note: This comment explains the following implementation detail for offline readability.
        target_features = torch.cat([situation.transpose(1, 2), threat_weights.unsqueeze(-1)], dim=-1)

        missile_tokens = self.missile_proj(missile_features) + self.missile_pos
        target_tokens = self.target_proj(target_features) + self.target_pos
        tokens = torch.cat([missile_tokens, target_tokens], dim=1)
        token_mask = torch.cat([mask_m, mask_t], dim=1)
        tokens = tokens * token_mask.unsqueeze(-1)

        for layer in self.layers:
            tokens = layer(tokens, token_mask)

        missile_tokens = tokens[:, :M_MAX, :]
        target_tokens = tokens[:, M_MAX:, :]
        logits = torch.matmul(self.q_proj(missile_tokens), self.k_proj(target_tokens).transpose(-2, -1)) / self.scale
        return logits.masked_fill(mask_t.unsqueeze(1) == 0, -1.0e9)


class AllocationPipelineTransformer(nn.Module):
    """
    Raw-input allocation model.

    External interface:
        x_missile, x_aircraft, x_target, mask_m, mask_a, mask_t -> allocation logits
    """

    def __init__(
        self,
        situation_model: SituationMLP,
        threat_model: ThreatMLP,
        allocation_core: AllocationCoreTransformer,
    ) -> None:
        super().__init__()
        self.situation_model = situation_model
        self.threat_model = threat_model
        self.allocation_core = allocation_core

    def forward(self, x_missile, x_aircraft, x_target, mask_m, mask_a, mask_t):
        """原始输入接口：服务器训练、ONNX 导出和离线推理都使用这个接口。"""
        # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
        situation, _ = self.situation_model(x_missile, x_target, mask_m, mask_t)
        _, threat_weights, _ = self.threat_model(x_aircraft, x_target, mask_a, mask_t)
        return self.allocation_core(situation, threat_weights, mask_m, mask_t)

    def forward_with_intermediates(self, x_missile, x_aircraft, x_target, mask_m, mask_a, mask_t):
        """ONNX 端到端导出时额外返回 S/Threat/ThreatW，便于离线检查。"""
        # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
        situation, _ = self.situation_model(x_missile, x_target, mask_m, mask_t)
        threat, threat_weights, _ = self.threat_model(x_aircraft, x_target, mask_a, mask_t)
        logits = self.allocation_core(situation, threat_weights, mask_m, mask_t)
        return logits, situation, threat, threat_weights


AllocationTransformer = AllocationCoreTransformer


def build_allocation_pipeline(args: argparse.Namespace, device: torch.device) -> AllocationPipelineTransformer:
    """加载前两个模型并组装完整分配 pipeline。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    situation_model = SituationMLP(hidden_dim=args.situation_hidden_dim).to(device)
    threat_model = ThreatMLP(hidden_dim=args.threat_hidden_dim).to(device)
    allocation_core = AllocationCoreTransformer(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)

    situation_model.load_state_dict(torch.load(args.situation_model, map_location=device))
    threat_model.load_state_dict(torch.load(args.threat_model, map_location=device))

    pipeline = AllocationPipelineTransformer(situation_model, threat_model, allocation_core).to(device)
    set_upstream_trainable(pipeline, not args.freeze_upstream)
    return pipeline


def set_upstream_trainable(model: AllocationPipelineTransformer, trainable: bool) -> None:
    """控制是否联合微调 SituationMLP 和 ThreatMLP。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    for upstream in (model.situation_model, model.threat_model):
        upstream.train(trainable)
        for parameter in upstream.parameters():
            parameter.requires_grad = trainable


def split_dataset(dataset: Dataset, train_ratio: float = 0.9) -> Tuple[Dataset, Dataset | None]:
    if len(dataset) < 2:
        return dataset, None
    train_size = max(1, int(train_ratio * len(dataset)))
    train_size = min(train_size, len(dataset) - 1)
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))


def masked_cross_entropy(logits, target_index, target_mask):
    """只在有效导弹位计算交叉熵。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    raw_loss = F.cross_entropy(logits.transpose(1, 2), target_index, reduction="none")
    return (raw_loss * target_mask).sum() / (target_mask.sum() + 1e-8)


def masked_accuracy(logits, target_index, target_mask):
    """只统计有效导弹位的 top-1 分配一致率。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    pred_index = torch.argmax(logits, dim=-1)
    correct = (pred_index == target_index).float() * target_mask
    return correct.sum() / (target_mask.sum() + 1e-8)


def evaluate(model: AllocationPipelineTransformer, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.to(device) for tensor in batch]
            x_m, x_a, x_t, mask_m, mask_a, mask_t, _, _, _, y_alloc_idx, y_alloc_mask = batch
            logits = model(x_m, x_a, x_t, mask_m, mask_a, mask_t)
            total_loss += masked_cross_entropy(logits, y_alloc_idx, y_alloc_mask).item()
            total_acc += masked_accuracy(logits, y_alloc_idx, y_alloc_mask).item()
    return total_loss / max(len(loader), 1), total_acc / max(len(loader), 1)


def train(args: argparse.Namespace) -> None:
    """训练入口。
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.

    默认 freeze_upstream=True，只训练 AllocationCoreTransformer 并保存其权重。
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    如果需要端到端微调，可传 --no-freeze-upstream。
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    """
    dataset = AirCombatDataset(args.data)
    train_dataset, val_dataset = split_dataset(dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_dataset is not None else None

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_allocation_pipeline(args, device)
    optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)

    print(f"Training AllocationPipelineTransformer on {device}, samples={len(dataset)}")
    print(f"freeze_upstream={args.freeze_upstream}")
    for epoch in range(args.epochs):
        model.train()
        if args.freeze_upstream:
            model.situation_model.eval()
            model.threat_model.eval()

        total_loss = 0.0
        total_acc = 0.0
        for batch in train_loader:
            batch = [tensor.to(device) for tensor in batch]
            x_m, x_a, x_t, mask_m, mask_a, mask_t, _, _, _, y_alloc_idx, y_alloc_mask = batch

            optimizer.zero_grad()
            logits = model(x_m, x_a, x_t, mask_m, mask_a, mask_t)
            loss = masked_cross_entropy(logits, y_alloc_idx, y_alloc_mask)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += masked_accuracy(logits, y_alloc_idx, y_alloc_mask).item()

        train_loss = total_loss / max(len(train_loader), 1)
        train_acc = total_acc / max(len(train_loader), 1)
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, device)
            print(
                f"Epoch [{epoch + 1:03d}/{args.epochs:03d}] "
                f"loss={train_loss:.6f} acc={train_acc:.2%} val_loss={val_loss:.6f} val_acc={val_acc:.2%}"
            )
        else:
            print(f"Epoch [{epoch + 1:03d}/{args.epochs:03d}] loss={train_loss:.6f} acc={train_acc:.2%}")

    torch.save(model.allocation_core.state_dict(), args.output)
    print(f"Saved allocation core model to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Version1 raw-input allocation Transformer.")
    parser.add_argument("--data", type=Path, default=Path("version1_aircombat_data.npz"))
    parser.add_argument("--situation-model", type=Path, default=Path("situation_model.pth"))
    parser.add_argument("--threat-model", type=Path, default=Path("threat_model.pth"))
    parser.add_argument("--output", type=Path, default=Path("allocation_transformer_model.pth"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--situation-hidden-dim", type=int, default=64)
    parser.add_argument("--threat-hidden-dim", type=int, default=64)
    parser.add_argument("--freeze-upstream", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
