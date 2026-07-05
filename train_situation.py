from __future__ import annotations

"""训练 Version1 态势评估 MLP。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.

模型输入:
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    x_missile [B, M_MAX, 6], x_target [B, T_MAX, 6],
    mask_m [B, M_MAX], mask_t [B, T_MAX]。
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
模型输出:
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    S_pred [B, M_MAX, T_MAX]，用于拟合 Python 规则算法的 Y_S。
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
训练损失:
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    只在 mask_m * mask_t 为 1 的真实导弹-目标 pair 上计算 MSE。
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


class SituationMLP(nn.Module):
    """导弹-目标 pair 级 MLP，内部自行构造物理特征。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_missile, x_target, mask_m, mask_t):
        # raw_score 覆盖所有固定位置；padding pair 通过 joint_mask 清零。
        # English note: This comment explains the following implementation detail for offline readability.
        features = self.build_pair_features(x_missile, x_target)
        raw_score = self.mlp(features).squeeze(-1)
        joint_mask = mask_m.unsqueeze(2) * mask_t.unsqueeze(1)
        return raw_score * joint_mask, joint_mask

    @staticmethod
    def build_pair_features(x_missile, x_target):
        """构造导弹-目标 pair 特征，形状为 [B, M_MAX, T_MAX, 10]。"""
        # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
        p_m, v_m = x_missile[..., :3], x_missile[..., 3:]
        p_t, v_t = x_target[..., :3], x_target[..., 3:]

        rel_pos = p_t.unsqueeze(1) - p_m.unsqueeze(2)
        rel_vel = v_t.unsqueeze(1) - v_m.unsqueeze(2)
        distance = torch.norm(rel_pos, dim=-1, keepdim=True)
        distance_safe = distance + 1e-6

        closing_speed = -torch.sum(rel_vel * rel_pos, dim=-1, keepdim=True) / distance_safe
        missile_speed = torch.norm(v_m, dim=-1, keepdim=True).unsqueeze(2)
        boresight_cos = torch.sum(v_m.unsqueeze(2) * rel_pos, dim=-1, keepdim=True) / (
            missile_speed * distance_safe + 1e-6
        )
        boresight_cos = torch.clamp(boresight_cos, -1.0, 1.0)

        return torch.cat(
            [
                rel_pos / POSITION_SCALE,
                rel_vel / VELOCITY_SCALE,
                distance / POSITION_SCALE,
                closing_speed / VELOCITY_SCALE,
                boresight_cos,
                missile_speed.expand_as(distance) / VELOCITY_SCALE,
            ],
            dim=-1,
        )


def split_dataset(dataset: Dataset, train_ratio: float = 0.9) -> Tuple[Dataset, Dataset | None]:
    if len(dataset) < 2:
        return dataset, None
    train_size = max(1, int(train_ratio * len(dataset)))
    train_size = min(train_size, len(dataset) - 1)
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))


def masked_mse(prediction, target, mask):
    """只统计真实实体 pair 的 MSE，padding 区域不参与训练。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    raw_loss = F.mse_loss(prediction, target, reduction="none")
    return (raw_loss * mask).sum() / (mask.sum() + 1e-8)


def evaluate(model: SituationMLP, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.to(device) for tensor in batch]
            x_m, _, x_t, mask_m, _, mask_t, y_s, *_ = batch
            pred_s, joint_mask = model(x_m, x_t, mask_m, mask_t)
            total_loss += masked_mse(pred_s, y_s, joint_mask).item()
    return total_loss / max(len(loader), 1)


def train(args: argparse.Namespace) -> None:
    """训练入口；默认保存 situation_model.pth。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    dataset = AirCombatDataset(args.data)
    train_dataset, val_dataset = split_dataset(dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_dataset is not None else None

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = SituationMLP(hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Training SituationMLP on {device}, samples={len(dataset)}")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = [tensor.to(device) for tensor in batch]
            x_m, _, x_t, mask_m, _, mask_t, y_s, *_ = batch

            optimizer.zero_grad()
            pred_s, joint_mask = model(x_m, x_t, mask_m, mask_t)
            loss = masked_mse(pred_s, y_s, joint_mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / max(len(train_loader), 1)
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device)
            print(f"Epoch [{epoch + 1:03d}/{args.epochs:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        else:
            print(f"Epoch [{epoch + 1:03d}/{args.epochs:03d}] train_loss={train_loss:.6f}")

    torch.save(model.state_dict(), args.output)
    print(f"Saved situation model to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Version1 situation MLP.")
    parser.add_argument("--data", type=Path, default=Path("version1_aircombat_data.npz"))
    parser.add_argument("--output", type=Path, default=Path("situation_model.pth"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
