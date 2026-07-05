from __future__ import annotations

"""训练 Version1 威胁评估 MLP。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.

模型输入:
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    x_aircraft [B, A_MAX, 6], x_target [B, T_MAX, 6],
    mask_a [B, A_MAX], mask_t [B, T_MAX]。
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
模型输出:
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    Threat_pred [B, A_MAX, T_MAX] 和 ThreatW_pred [B, T_MAX]。
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.

ThreatW_pred 不是单独随意学习的分支，而是由 Threat_pred 经过
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
masked mean + normalize 得到，保持和 Version1 规则算法一致。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
"""

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

from dataset import AirCombatDataset
from shape_config import POSITION_SCALE, VELOCITY_SCALE


class ThreatMLP(nn.Module):
    """友方飞机-目标 pair 级 MLP，拟合 Version1 威胁矩阵。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(13, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_aircraft, x_target, mask_a, mask_t):
        # joint_mask 屏蔽 padding 友方飞机和 padding 目标。
        # English note: This comment explains the following implementation detail for offline readability.
        features = self.build_pair_features(x_aircraft, x_target)
        raw_threat = self.mlp(features).squeeze(-1)
        joint_mask = mask_a.unsqueeze(2) * mask_t.unsqueeze(1)
        threat = raw_threat * joint_mask
        threat_weights = self.compute_target_weights(threat, mask_a, mask_t)
        return threat, threat_weights, joint_mask

    @staticmethod
    def build_pair_features(x_aircraft, x_target):
        """构造友方飞机-目标 pair 特征，形状为 [B, A_MAX, T_MAX, 13]。"""
        # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
        p_a, v_a = x_aircraft[..., :3], x_aircraft[..., 3:]
        p_t, v_t = x_target[..., :3], x_target[..., 3:]

        rel_pos = p_a.unsqueeze(2) - p_t.unsqueeze(1)
        distance = torch.norm(rel_pos, dim=-1, keepdim=True)
        distance_safe = distance + 1e-6

        aircraft_speed = torch.norm(v_a, dim=-1, keepdim=True).unsqueeze(2)
        target_speed = torch.norm(v_t, dim=-1, keepdim=True).unsqueeze(1)
        target_dir = v_t.unsqueeze(1) / (target_speed + 1e-6)
        los_cos = torch.sum(target_dir * rel_pos, dim=-1, keepdim=True) / distance_safe
        los_cos = torch.clamp(los_cos, -1.0, 1.0)

        speed_ratio = target_speed / (aircraft_speed + 1e-6)
        height_delta = p_t[..., 1].unsqueeze(1).unsqueeze(-1) - p_a[..., 1].unsqueeze(2).unsqueeze(-1)

        return torch.cat(
            [
                rel_pos / POSITION_SCALE,
                v_a.unsqueeze(2).expand_as(rel_pos) / VELOCITY_SCALE,
                v_t.unsqueeze(1).expand_as(rel_pos) / VELOCITY_SCALE,
                distance / POSITION_SCALE,
                los_cos,
                speed_ratio,
                height_delta / POSITION_SCALE,
            ],
            dim=-1,
        )

    @staticmethod
    def compute_target_weights(threat, mask_a, mask_t):
        """按 Version1 逻辑从威胁矩阵聚合出目标威胁权重。"""
        # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
        aircraft_count = mask_a.sum(dim=1, keepdim=True)
        raw_value = threat.sum(dim=1) / (aircraft_count + 1e-8)
        raw_value = raw_value * mask_t
        raw_sum = raw_value.sum(dim=-1, keepdim=True)
        valid_target_count = mask_t.sum(dim=-1, keepdim=True)
        uniform = mask_t / (valid_target_count + 1e-8)
        normalized = raw_value / (raw_sum + 1e-8)
        return torch.where(raw_sum > 1e-8, normalized, uniform)


def split_dataset(dataset: Dataset, train_ratio: float = 0.9) -> Tuple[Dataset, Dataset | None]:
    if len(dataset) < 2:
        return dataset, None
    train_size = max(1, int(train_ratio * len(dataset)))
    train_size = min(train_size, len(dataset) - 1)
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))


def masked_mse(prediction, target, mask):
    raw_loss = F.mse_loss(prediction, target, reduction="none")
    return (raw_loss * mask).sum() / (mask.sum() + 1e-8)


def evaluate(model: ThreatMLP, loader: DataLoader, device: torch.device, weight_loss_factor: float) -> Tuple[float, float]:
    model.eval()
    total_pair_loss = 0.0
    total_weight_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.to(device) for tensor in batch]
            _, x_a, x_t, _, mask_a, mask_t, _, y_threat, y_threat_w, *_ = batch
            pred_threat, pred_threat_w, joint_mask = model(x_a, x_t, mask_a, mask_t)
            pair_loss = masked_mse(pred_threat, y_threat, joint_mask)
            weight_loss = masked_mse(pred_threat_w, y_threat_w, mask_t)
            total_pair_loss += pair_loss.item()
            total_weight_loss += weight_loss.item() * weight_loss_factor
    return total_pair_loss / max(len(loader), 1), total_weight_loss / max(len(loader), 1)


def train(args: argparse.Namespace) -> None:
    """训练入口；默认保存 threat_model.pth。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    dataset = AirCombatDataset(args.data)
    train_dataset, val_dataset = split_dataset(dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_dataset is not None else None

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = ThreatMLP(hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Training ThreatMLP on {device}, samples={len(dataset)}")
    for epoch in range(args.epochs):
        model.train()
        total_pair_loss = 0.0
        total_weight_loss = 0.0
        for batch in train_loader:
            batch = [tensor.to(device) for tensor in batch]
            _, x_a, x_t, _, mask_a, mask_t, _, y_threat, y_threat_w, *_ = batch

            optimizer.zero_grad()
            pred_threat, pred_threat_w, joint_mask = model(x_a, x_t, mask_a, mask_t)
            pair_loss = masked_mse(pred_threat, y_threat, joint_mask)
            weight_loss = masked_mse(pred_threat_w, y_threat_w, mask_t)
            loss = pair_loss + args.weight_loss_factor * weight_loss
            loss.backward()
            optimizer.step()

            total_pair_loss += pair_loss.item()
            total_weight_loss += weight_loss.item()

        pair_text = total_pair_loss / max(len(train_loader), 1)
        weight_text = total_weight_loss / max(len(train_loader), 1)
        if val_loader is not None:
            val_pair_loss, val_weight_loss = evaluate(model, val_loader, device, args.weight_loss_factor)
            print(
                f"Epoch [{epoch + 1:03d}/{args.epochs:03d}] "
                f"pair_loss={pair_text:.6f} weight_loss={weight_text:.6f} "
                f"val_pair={val_pair_loss:.6f} val_weight={val_weight_loss:.6f}"
            )
        else:
            print(f"Epoch [{epoch + 1:03d}/{args.epochs:03d}] pair_loss={pair_text:.6f} weight_loss={weight_text:.6f}")

    torch.save(model.state_dict(), args.output)
    print(f"Saved threat model to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Version1 threat MLP.")
    parser.add_argument("--data", type=Path, default=Path("version1_aircombat_data.npz"))
    parser.add_argument("--output", type=Path, default=Path("threat_model.pth"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--weight-loss-factor", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
