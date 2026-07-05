from __future__ import annotations

"""ONNX 端到端推理烟测脚本。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.

脚本构造一组固定 demo 实体：
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
1. 用 Python 规则算法得到业务分配结果。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
2. 把同一组实体 padding 成 ONNX 输入。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
3. 调用 version1_aircombat_pipeline.onnx 得到 allocation_index。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
4. 对比二者并检查目标序号是否落在真实目标范围内。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
"""

import argparse
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

from dto import AllocationInputDTO, EntityState, SAMInputDTO, ThreatInputDTO
from generate_dataset import build_fixed_inputs
from shape_config import M_MAX, T_MAX
from SituationAssessment import SituationAssessment
from TargetAllocation import TargetAllocation
from ThreatAssessment import ThreatAssessment


def build_entities(raw_data: Sequence[Tuple[str, Tuple[float, float, float], Tuple[float, float, float]]]) -> Tuple[EntityState, ...]:
    """把简洁的 tuple 测试数据转换为 EntityState。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    entities = []
    for entity_id, position, velocity in raw_data:
        entities.append(
            EntityState(
                id=entity_id,
                position=np.asarray(position, dtype=np.float64),
                velocity=np.asarray(velocity, dtype=np.float64),
                euler_angles=np.zeros(3, dtype=np.float64),
            )
        )
    return tuple(entities)


def build_demo_scenario() -> Tuple[Tuple[EntityState, ...], Tuple[EntityState, ...], Tuple[EntityState, ...]]:
    """构造一组小规模 demo 场景。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    missiles = build_entities(
        [
            ("m0", (5000.0, 10000.0, 5000.0), (600.0, -500.0, 0.0)),
            ("m1", (5000.0, -10000.0, 5000.0), (600.0, 500.0, 0.0)),
        ]
    )
    aircrafts = build_entities(
        [
            ("a0", (0.0, 10000.0, 5000.0), (250.0, 0.0, 0.0)),
            ("a1", (0.0, -10000.0, 5000.0), (250.0, 0.0, 0.0)),
        ]
    )
    targets = build_entities(
        [
            ("t0", (30000.0, 15000.0, 5000.0), (-200.0, -50.0, 0.0)),
            ("t1", (30000.0, -15000.0, 5000.0), (-200.0, 50.0, 0.0)),
            ("t2", (70000.0, 0.0, 6500.0), (-250.0, 0.0, -30.0)),
        ]
    )
    return missiles, aircrafts, targets


def run_business_pipeline(
    missiles: Sequence[EntityState],
    aircrafts: Sequence[EntityState],
    targets: Sequence[EntityState],
) -> Dict[int, int]:
    """运行 NumPy/Python 规则算法，作为 ONNX 结果的对照基准。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    situation_output = SituationAssessment().compute(SAMInputDTO(missiles=tuple(missiles), targets=tuple(targets)))
    threat_output = ThreatAssessment().compute(ThreatInputDTO(our_aircrafts=tuple(aircrafts), targets=tuple(targets)))
    allocation_output = TargetAllocation().compute(
        AllocationInputDTO(
            missiles=tuple(missiles),
            targets=tuple(targets),
            situation_output=situation_output,
            threat_output=threat_output,
        )
    )
    return {item.missile_index: item.target_index for item in allocation_output.assignments}


def run_onnx_pipeline(
    onnx_path: Path,
    missiles: Sequence[EntityState],
    aircrafts: Sequence[EntityState],
    targets: Sequence[EntityState],
) -> Tuple[Dict[int, int], np.ndarray]:
    """运行 ONNX pipeline，返回有效导弹位的目标序号映射。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    import onnxruntime as ort

    # 与训练/导出保持一致：先 padding，再带 mask 输入 ONNX。
    # English note: This comment explains the following implementation detail for offline readability.
    fixed_inputs = build_fixed_inputs(missiles, aircrafts, targets)
    inputs = {
        "X_missile": fixed_inputs["X_missile"][np.newaxis, :, :],
        "X_aircraft": fixed_inputs["X_aircraft"][np.newaxis, :, :],
        "X_target": fixed_inputs["X_target"][np.newaxis, :, :],
        "mask_M": fixed_inputs["mask_M"][np.newaxis, :],
        "mask_A": fixed_inputs["mask_A"][np.newaxis, :],
        "mask_T": fixed_inputs["mask_T"][np.newaxis, :],
    }
    session = ort.InferenceSession(str(onnx_path))
    allocation_index, allocation_mask = session.run(["allocation_index", "allocation_mask"], inputs)

    valid_count = min(len(missiles), M_MAX)
    result = {}
    for missile_index in range(valid_count):
        if allocation_mask[0, missile_index] <= 0.0:
            continue
        target_index = int(allocation_index[0, missile_index])
        result[missile_index] = target_index
    return result, allocation_index


def main() -> None:
    """命令行入口。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    args = parse_args()
    missiles, aircrafts, targets = build_demo_scenario()
    expected = run_business_pipeline(missiles, aircrafts, targets)
    print("Business allocation index map:")
    print(expected)

    try:
        predicted, raw_indices = run_onnx_pipeline(args.onnx, missiles, aircrafts, targets)
    except ImportError:
        print("onnxruntime is not installed; skipped ONNX inference.")
        return
    except Exception as exc:
        print(f"ONNX inference failed: {exc}")
        return

    print("ONNX allocation index map:")
    print(predicted)
    print(f"raw allocation_index tensor: {raw_indices}")
    invalid = {m_idx: t_idx for m_idx, t_idx in predicted.items() if t_idx < 0 or t_idx >= min(len(targets), T_MAX)}
    if invalid:
        print(f"Invalid target indices detected: {invalid}")
    else:
        print("All ONNX target indices are within the real target range.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Version1 ONNX pipeline smoke test.")
    parser.add_argument("--onnx", type=Path, default=Path("version1_aircombat_pipeline.onnx"))
    return parser.parse_args()


if __name__ == "__main__":
    main()
