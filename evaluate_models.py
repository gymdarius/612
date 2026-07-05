from __future__ import annotations

"""评估 Version1 三个训练模型的规则拟合效果。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.

评估分两类：
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
1. 连续值回归指标：态势矩阵、威胁矩阵、目标威胁权重的 MSE/MAE。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
2. 业务一致率：模型输出的 allocation_index 与 Python 规则标签是否一致。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.

allocation_pipeline_top1_acc 是最重要的端到端指标，表示同样原始输入下，
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
模型完整链路和 Python 规则目标分配的有效导弹位一致率。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import AirCombatDataset
from train_allocation import AllocationCoreTransformer, AllocationPipelineTransformer, masked_accuracy
from train_situation import SituationMLP
from train_threat import ThreatMLP


def masked_mean(value, mask):
    """只在 mask 为 1 的位置求平均。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    return (value * mask).sum() / (mask.sum() + 1e-8)


def load_state(model, path: Path, device: torch.device):
    """加载 pth 权重并切到 eval 模式。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def evaluate(args: argparse.Namespace) -> None:
    """评估入口。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset = AirCombatDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    situation_model = load_state(SituationMLP(hidden_dim=args.situation_hidden_dim), args.situation_model, device).to(device)
    threat_model = load_state(ThreatMLP(hidden_dim=args.threat_hidden_dim), args.threat_model, device).to(device)
    allocation_core = load_state(
        AllocationCoreTransformer(
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ff_dim=args.ff_dim,
            dropout=0.0,
        ),
        args.allocation_model,
        device,
    ).to(device)
    allocation_pipeline = AllocationPipelineTransformer(situation_model, threat_model, allocation_core).to(device)
    allocation_pipeline.eval()

    total_s_mse = 0.0
    total_s_mae = 0.0
    total_s_tol = 0.0
    total_t_mse = 0.0
    total_tw_mse = 0.0
    total_alloc_pipeline_acc = 0.0
    total_alloc_core_teacher_acc = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch in loader:
            batch = [tensor.to(device) for tensor in batch]
            x_m, x_a, x_t, mask_m, mask_a, mask_t, y_s, y_threat, y_threat_w, y_alloc_idx, y_alloc_mask = batch

            pred_s, mt_mask = situation_model(x_m, x_t, mask_m, mask_t)
            pred_threat, pred_threat_w, at_mask = threat_model(x_a, x_t, mask_a, mask_t)

            s_error = pred_s - y_s
            t_error = pred_threat - y_threat
            tw_error = pred_threat_w - y_threat_w

            total_s_mse += masked_mean(s_error.square(), mt_mask).item()
            total_s_mae += masked_mean(s_error.abs(), mt_mask).item()
            total_s_tol += masked_mean((s_error.abs() < args.tolerance).float(), mt_mask).item()
            total_t_mse += masked_mean(t_error.square(), at_mask).item()
            total_tw_mse += masked_mean(tw_error.square(), mask_t).item()

            # core_teacher 使用真实中间标签，衡量分配 Transformer 本身是否学到规则。
            # English note: This comment explains the following implementation detail for offline readability.
            core_teacher_logits = allocation_core(y_s, y_threat_w, mask_m, mask_t)
            # pipeline 使用原始输入，衡量完整模型链路在离线推理时的真实表现。
            # English note: This comment explains the following implementation detail for offline readability.
            pipeline_logits = allocation_pipeline(x_m, x_a, x_t, mask_m, mask_a, mask_t)
            total_alloc_core_teacher_acc += masked_accuracy(core_teacher_logits, y_alloc_idx, y_alloc_mask).item()
            total_alloc_pipeline_acc += masked_accuracy(pipeline_logits, y_alloc_idx, y_alloc_mask).item()
            batch_count += 1

    batch_count = max(batch_count, 1)
    print("Version1 model evaluation")
    print(f"situation_mse: {total_s_mse / batch_count:.6f}")
    print(f"situation_mae: {total_s_mae / batch_count:.6f}")
    print(f"situation_abs_error_lt_{args.tolerance}: {total_s_tol / batch_count:.2%}")
    print(f"threat_mse: {total_t_mse / batch_count:.6f}")
    print(f"threat_weight_mse: {total_tw_mse / batch_count:.6f}")
    print(f"allocation_core_teacher_top1_acc: {total_alloc_core_teacher_acc / batch_count:.2%}")
    print(f"allocation_pipeline_top1_acc: {total_alloc_pipeline_acc / batch_count:.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Version1 trained models.")
    parser.add_argument("--data", type=Path, default=Path("version1_aircombat_data.npz"))
    parser.add_argument("--situation-model", type=Path, default=Path("situation_model.pth"))
    parser.add_argument("--threat-model", type=Path, default=Path("threat_model.pth"))
    parser.add_argument("--allocation-model", type=Path, default=Path("allocation_transformer_model.pth"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--situation-hidden-dim", type=int, default=64)
    parser.add_argument("--threat-hidden-dim", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
