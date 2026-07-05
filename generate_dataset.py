from __future__ import annotations

"""Version1 训练数据生成脚本。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.

每条样本是一个随机空战场景。脚本先调用 Python 规则算法生成标签，
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
再把不同数量的实体 padding 到固定形状，并生成对应 mask。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.

输出 .npz 中的核心数组：
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
- X_missile/X_aircraft/X_target: 固定形状实体特征 [x,y,z,vx,vy,vz]。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
- mask_M/mask_A/mask_T: 真实实体为 1，padding 位为 0。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
- Y_S/Y_Threat/Y_ThreatW: 态势、威胁、目标威胁权重标签。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
- Y_AllocIndex/Y_AllocMask: 每枚有效导弹对应的目标序号和导弹有效 mask。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
"""

import argparse
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from dto import AllocationInputDTO, EntityState, SAMInputDTO, ThreatInputDTO
from shape_config import A_MAX, FEATURE_DIM, M_MAX, T_MAX
from SituationAssessment import SituationAssessment
from TargetAllocation import TargetAllocation
from ThreatAssessment import ThreatAssessment


def generate_random_entities(
    rng: np.random.Generator,
    count: int,
    prefix: str,
    center_pos: NDArray[np.float64],
) -> Tuple[EntityState, ...]:
    """生成一组随机实体，包含普通分布和密集编队两类场景。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    entities = []
    is_formation = rng.random() < 0.3
    formation_center = center_pos + rng.uniform(-15_000.0, 15_000.0, 3)

    for index in range(count):
        if is_formation:
            position = np.clip(formation_center + rng.uniform(-500.0, 500.0, 3), 0.0, 50_000.0)
        elif rng.random() < 0.5:
            position = rng.uniform(0.0, 50_000.0, 3)
        else:
            position = np.clip(center_pos + rng.uniform(-15_000.0, 15_000.0, 3), 0.0, 50_000.0)

        velocity = rng.uniform(-1_000.0, 1_000.0, 3)
        entities.append(
            EntityState(
                id=f"{prefix}_{index}",
                position=position.astype(np.float64),
                velocity=velocity.astype(np.float64),
                euler_angles=np.zeros(3, dtype=np.float64),
            )
        )
    return tuple(entities)


def pad_entities(
    entities: Sequence[EntityState],
    max_count: int,
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """把变长实体列表补齐到固定长度，并生成实体有效 mask。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    if len(entities) > max_count:
        raise ValueError(f"entity count {len(entities)} exceeds max_count {max_count}.")

    features = np.zeros((max_count, FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((max_count,), dtype=np.float32)
    for index, entity in enumerate(entities):
        features[index] = np.asarray(
            [
                entity.position[0],
                entity.position[1],
                entity.position[2],
                entity.velocity[0],
                entity.velocity[1],
                entity.velocity[2],
            ],
            dtype=np.float32,
        )
        mask[index] = 1.0
    return features, mask


def build_fixed_inputs(
    missiles: Sequence[EntityState],
    aircrafts: Sequence[EntityState],
    targets: Sequence[EntityState],
) -> Dict[str, NDArray[np.float32]]:
    """把三类实体统一转换为模型输入张量和 mask。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    x_missile, mask_m = pad_entities(missiles, M_MAX)
    x_aircraft, mask_a = pad_entities(aircrafts, A_MAX)
    x_target, mask_t = pad_entities(targets, T_MAX)
    return {
        "X_missile": x_missile,
        "X_aircraft": x_aircraft,
        "X_target": x_target,
        "mask_M": mask_m,
        "mask_A": mask_a,
        "mask_T": mask_t,
    }


def generate_one_scenario(rng: np.random.Generator) -> Dict[str, NDArray[np.float32]]:
    """生成一条完整训练样本。
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.

    注意：目标数量从 1 开始随机，保证分配标签一定有合法目标序号。
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    """
    missile_count = int(rng.integers(1, M_MAX + 1))
    aircraft_count = int(rng.integers(1, A_MAX + 1))
    target_count = int(rng.integers(1, T_MAX + 1))
    battle_center = rng.uniform(15_000.0, 35_000.0, 3).astype(np.float64)

    missiles = generate_random_entities(rng, missile_count, "M", battle_center)
    aircrafts = generate_random_entities(rng, aircraft_count, "A", battle_center)
    targets = generate_random_entities(rng, target_count, "T", battle_center)

    # 标签全部来自 Version1 规则算法，神经网络训练目标就是拟合这些输出。
    # English note: This comment explains the following implementation detail for offline readability.
    situation_output = SituationAssessment().compute(SAMInputDTO(missiles=missiles, targets=targets))
    threat_output = ThreatAssessment().compute(ThreatInputDTO(our_aircrafts=aircrafts, targets=targets))
    allocation_output = TargetAllocation().compute(
        AllocationInputDTO(
            missiles=missiles,
            targets=targets,
            situation_output=situation_output,
            threat_output=threat_output,
        )
    )

    scenario = build_fixed_inputs(missiles, aircrafts, targets)
    y_s = np.zeros((M_MAX, T_MAX), dtype=np.float32)
    y_threat = np.zeros((A_MAX, T_MAX), dtype=np.float32)
    y_threat_w = np.zeros((T_MAX,), dtype=np.float32)
    y_alloc_index = np.zeros((M_MAX,), dtype=np.int64)
    y_alloc_mask = np.zeros((M_MAX,), dtype=np.float32)

    y_s[:missile_count, :target_count] = situation_output.situation_matrix.astype(np.float32)
    y_threat[:aircraft_count, :target_count] = threat_output.threat_matrix.astype(np.float32)
    y_threat_w[:target_count] = threat_output.target_threat_weights.astype(np.float32)
    for assignment in allocation_output.assignments:
        if assignment.missile_index >= 0 and assignment.target_index >= 0:
            y_alloc_index[assignment.missile_index] = assignment.target_index
            y_alloc_mask[assignment.missile_index] = 1.0

    scenario.update(
        {
            "Y_S": y_s,
            "Y_Threat": y_threat,
            "Y_ThreatW": y_threat_w,
            "Y_AllocIndex": y_alloc_index,
            "Y_AllocMask": y_alloc_mask,
        }
    )
    return scenario


def generate_dataset(sample_count: int, seed: int | None = None) -> Dict[str, NDArray]:
    """批量生成 sample_count 条场景并 stack 成 batch 维数组。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    rng = np.random.default_rng(seed)
    collections = {
        "X_missile": [],
        "X_aircraft": [],
        "X_target": [],
        "mask_M": [],
        "mask_A": [],
        "mask_T": [],
        "Y_S": [],
        "Y_Threat": [],
        "Y_ThreatW": [],
        "Y_AllocIndex": [],
        "Y_AllocMask": [],
    }
    for index in range(sample_count):
        if (index + 1) % 1000 == 0:
            print(f"generated {index + 1} / {sample_count} samples")
        scenario = generate_one_scenario(rng)
        for key in collections:
            collections[key].append(scenario[key])
    return {key: np.stack(values) for key, values in collections.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Version1 supervised training data.")
    parser.add_argument("--samples", type=int, default=1000, help="Number of random scenarios to generate.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("version1_aircombat_data.npz"),
        help="Output .npz dataset path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be > 0.")
    data = generate_dataset(sample_count=args.samples, seed=args.seed)
    np.savez(args.output, **data)
    print(f"saved dataset to {args.output}")
    for key, value in data.items():
        print(f"{key}: {value.shape}")


if __name__ == "__main__":
    main()
