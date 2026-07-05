from __future__ import annotations

"""导出 Version1 训练模型到 ONNX。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.

会导出四个文件：
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
- situation_model.onnx: 单独态势模型。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
- threat_model.onnx: 单独威胁模型。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
- allocation_model.onnx: 原始输入到分配结果的分配模型。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
- version1_aircombat_pipeline.onnx: 端到端模型，额外输出中间 S/Threat/ThreatW。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.

ONNX 的实体数量维固定为 shape_config.py 中的 M_MAX/A_MAX/T_MAX，
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
只允许 batch 维动态。离线推理前必须按同样规则 padding 并生成 mask。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from shape_config import A_MAX, FEATURE_DIM, M_MAX, T_MAX
from train_allocation import AllocationCoreTransformer, AllocationPipelineTransformer
from train_situation import SituationMLP
from train_threat import ThreatMLP


class AllocationIndexWrapper(nn.Module):
    """给 allocation 模型补上 argmax 输出，方便 ONNX 直接返回目标序号。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    def __init__(self, allocation_model: AllocationPipelineTransformer) -> None:
        super().__init__()
        self.allocation_model = allocation_model

    def forward(self, x_missile, x_aircraft, x_target, mask_m, mask_a, mask_t):
        logits = self.allocation_model(x_missile, x_aircraft, x_target, mask_m, mask_a, mask_t)
        allocation_index = torch.argmax(logits, dim=-1)
        return logits, allocation_index, mask_m


class Version1AirCombatPipeline(nn.Module):
    """端到端 ONNX 包装器，返回中间结果和最终分配索引。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    def __init__(self, allocation_model: AllocationPipelineTransformer) -> None:
        super().__init__()
        self.allocation_model = allocation_model

    def forward(self, x_missile, x_aircraft, x_target, mask_m, mask_a, mask_t):
        logits, situation, threat, threat_weights = self.allocation_model.forward_with_intermediates(
            x_missile,
            x_aircraft,
            x_target,
            mask_m,
            mask_a,
            mask_t,
        )
        allocation_index = torch.argmax(logits, dim=-1)
        return situation, threat, threat_weights, allocation_index, mask_m


def load_models(args: argparse.Namespace):
    """加载三个 pth 权重并组装 raw-input allocation pipeline。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    situation_model = SituationMLP(hidden_dim=args.situation_hidden_dim)
    threat_model = ThreatMLP(hidden_dim=args.threat_hidden_dim)
    allocation_core = AllocationCoreTransformer(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=0.0,
    )
    situation_model.load_state_dict(torch.load(args.situation_model, map_location="cpu"))
    threat_model.load_state_dict(torch.load(args.threat_model, map_location="cpu"))
    allocation_core.load_state_dict(torch.load(args.allocation_model, map_location="cpu"))

    allocation_pipeline = AllocationPipelineTransformer(situation_model, threat_model, allocation_core)
    for model in (situation_model, threat_model, allocation_core, allocation_pipeline):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
    return situation_model, threat_model, allocation_pipeline


def export_model(model, inputs, output_path: Path, input_names, output_names, opset: int) -> None:
    """通用 ONNX 导出函数；只把 batch 维设为动态。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    dynamic_axes = {name: {0: "batch_size"} for name in input_names}
    dynamic_axes.update({name: {0: "batch_size"} for name in output_names})
    torch.onnx.export(
        model,
        inputs,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print(f"exported {output_path}")


def export_all(args: argparse.Namespace) -> None:
    """导出所有 ONNX 文件。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    situation_model, threat_model, allocation_pipeline = load_models(args)

    x_missile = torch.randn(1, M_MAX, FEATURE_DIM)
    x_aircraft = torch.randn(1, A_MAX, FEATURE_DIM)
    x_target = torch.randn(1, T_MAX, FEATURE_DIM)
    mask_m = torch.ones(1, M_MAX)
    mask_a = torch.ones(1, A_MAX)
    mask_t = torch.ones(1, T_MAX)

    export_model(
        situation_model,
        (x_missile, x_target, mask_m, mask_t),
        args.output_dir / "situation_model.onnx",
        ["X_missile", "X_target", "mask_M", "mask_T"],
        ["S_pred", "joint_mask_MT"],
        args.opset,
    )
    export_model(
        threat_model,
        (x_aircraft, x_target, mask_a, mask_t),
        args.output_dir / "threat_model.onnx",
        ["X_aircraft", "X_target", "mask_A", "mask_T"],
        ["Threat_pred", "ThreatW_pred", "joint_mask_AT"],
        args.opset,
    )
    export_model(
        AllocationIndexWrapper(allocation_pipeline),
        (x_missile, x_aircraft, x_target, mask_m, mask_a, mask_t),
        args.output_dir / "allocation_model.onnx",
        ["X_missile", "X_aircraft", "X_target", "mask_M", "mask_A", "mask_T"],
        ["Alloc_logits", "allocation_index", "allocation_mask"],
        args.opset,
    )
    export_model(
        Version1AirCombatPipeline(allocation_pipeline),
        (x_missile, x_aircraft, x_target, mask_m, mask_a, mask_t),
        args.output_dir / "version1_aircombat_pipeline.onnx",
        ["X_missile", "X_aircraft", "X_target", "mask_M", "mask_A", "mask_T"],
        ["S_pred", "Threat_pred", "ThreatW_pred", "allocation_index", "allocation_mask"],
        args.opset,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Version1 models to ONNX.")
    parser.add_argument("--situation-model", type=Path, default=Path("situation_model.pth"))
    parser.add_argument("--threat-model", type=Path, default=Path("threat_model.pth"))
    parser.add_argument("--allocation-model", type=Path, default=Path("allocation_transformer_model.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--situation-hidden-dim", type=int, default=64)
    parser.add_argument("--threat-hidden-dim", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    export_all(parse_args())
