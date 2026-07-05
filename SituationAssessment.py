from __future__ import annotations

"""Version1 态势评估业务算法。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.

主接口 compute(SAMInputDTO) 返回 DTO 输出；evaluate(...) 保留旧版字典输入
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
兼容。内部使用 NumPy 广播一次性计算所有导弹-目标组合，但评分公式仍是
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
Version1 原业务逻辑：接近速度、距离、前向视场/瞄准轴、导弹能量。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

try:
    from .dto import (
        EntityState,
        SAMInputDTO,
        SAMOutputDTO,
        SAMPairMetricsDTO,
        coerce_entities,
        stack_entity_vectors,
    )
except ImportError:  # pragma: no cover - keeps direct script execution working.
    from dto import (
        EntityState,
        SAMInputDTO,
        SAMOutputDTO,
        SAMPairMetricsDTO,
        coerce_entities,
        stack_entity_vectors,
    )


class SituationAssessment:
    """
    Version1 situation assessment logic with a Version2-style DTO interface.

    The scoring formula is kept from the original Version1 implementation:
    closing speed, distance, boresight, and missile energy are combined into
    an M x N situation matrix in [0, 1].
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        weights = cfg.get("weights", {})
        self.w_closing: float = weights.get("closing_speed", 0.50)
        self.w_distance: float = weights.get("distance", 0.35)
        self.w_boresight: float = weights.get("boresight", 0.10)
        self.w_energy: float = weights.get("energy", 0.05)
        total_weight = self.w_closing + self.w_distance + self.w_boresight + self.w_energy
        if total_weight > 0:
            self.w_closing /= total_weight
            self.w_distance /= total_weight
            self.w_boresight /= total_weight
            self.w_energy /= total_weight

        refs = cfg.get("reference_values", {})
        self.D_ref: float = refs.get("distance_ref", 200000.0)
        self.V_ref: float = refs.get("closing_speed_ref", 1300.0)

        fov_cfg = cfg.get("fov", {})
        self.fov_cos_threshold: float = np.cos(np.radians(fov_cfg.get("half_angle_deg", 60.0)))

        self.epsilon: float = float(cfg.get("epsilon", 1e-8))
        self.export_pair_metrics: bool = bool(cfg.get("export_pair_metrics", True))
        if self.D_ref <= 0.0:
            raise ValueError("distance_ref must be > 0.")
        if self.V_ref <= 0.0:
            raise ValueError("closing_speed_ref must be > 0.")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be > 0.")

    def compute(self, data: SAMInputDTO) -> SAMOutputDTO:
        """计算 M x T 态势矩阵。
        # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.

        输入:
        # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
            data.missiles: M 枚导弹。
            # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
            data.targets: T 个目标/敌机。
            # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
        输出:
        # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
            SAMOutputDTO.situation_matrix，元素范围 [0, 1]。
            # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
        """
        missiles = tuple(data.missiles)
        targets = tuple(data.targets)
        missile_count = len(missiles)
        target_count = len(targets)

        if missile_count == 0 or target_count == 0:
            return SAMOutputDTO(
                situation_matrix=np.zeros((missile_count, target_count), dtype=np.float64),
                pair_metrics=tuple(),
            )

        missile_pos = stack_entity_vectors(missiles, "position")
        target_pos = stack_entity_vectors(targets, "position")
        missile_vel = stack_entity_vectors(missiles, "velocity")
        target_vel = stack_entity_vectors(targets, "velocity")
        missile_speed = np.asarray([entity.speed for entity in missiles], dtype=np.float64)

        # 广播成 (M, T, 3)，避免 Python 双层循环，ONNX/训练标签生成也更容易对齐。
        # English note: This comment explains the following implementation detail for offline readability.
        rel_pos = target_pos[np.newaxis, :, :] - missile_pos[:, np.newaxis, :]
        distance = np.linalg.norm(rel_pos, axis=2)
        distance_safe = np.maximum(distance, self.epsilon)
        zero_distance = distance < self.epsilon

        los = rel_pos / distance_safe[:, :, np.newaxis]
        missile_dir = np.divide(
            missile_vel,
            missile_speed[:, np.newaxis],
            out=np.zeros_like(missile_vel),
            where=missile_speed[:, np.newaxis] > self.epsilon,
        )

        relative_velocity = target_vel[np.newaxis, :, :] - missile_vel[:, np.newaxis, :]
        closing_speed = -np.sum(relative_velocity * los, axis=2)

        # 以下四个分量完全沿用 Version1 原始评分公式。
        # English note: This comment explains the following implementation detail for offline readability.
        closing_score = 1.0 / (1.0 + np.exp(-3.0 * (closing_speed / self.V_ref - 0.3)))
        distance_score = np.exp(-distance / self.D_ref)

        boresight_cos = np.clip(np.sum(missile_dir[:, np.newaxis, :] * los, axis=2), -1.0, 1.0)
        in_fov = boresight_cos >= self.fov_cos_threshold
        boresight_inside = 0.95 + 0.05 * (boresight_cos - self.fov_cos_threshold) / max(
            1.0 - self.fov_cos_threshold,
            self.epsilon,
        )
        boresight_outside = 0.2 * np.clip(
            (boresight_cos + 1.0) / max(self.fov_cos_threshold + 1.0, self.epsilon),
            0.0,
            1.0,
        )
        boresight_score = np.where(in_fov, boresight_inside, boresight_outside)

        energy_score = np.clip(
            missile_speed[:, np.newaxis] / self.V_ref * np.exp(-distance / (2.0 * self.D_ref)),
            0.0,
            1.0,
        )

        # 加权融合后裁剪到 [0, 1]；零距离按原逻辑认为态势评分为 1。
        # English note: This comment explains the following implementation detail for offline readability.
        situation_matrix = np.clip(
            self.w_closing * closing_score
            + self.w_distance * distance_score
            + self.w_boresight * boresight_score
            + self.w_energy * energy_score,
            0.0,
            1.0,
        )
        situation_matrix = np.where(zero_distance, 1.0, situation_matrix)

        pair_metrics = tuple()
        if self.export_pair_metrics:
            pair_metrics = self._build_pair_metrics(
                missiles=missiles,
                targets=targets,
                distance=distance,
                closing_speed=closing_speed,
                boresight_cos=boresight_cos,
                in_fov=in_fov,
                closing_score=closing_score,
                distance_score=distance_score,
                boresight_score=boresight_score,
                energy_score=energy_score,
                situation_matrix=situation_matrix,
            )

        return SAMOutputDTO(situation_matrix=situation_matrix, pair_metrics=pair_metrics)

    def evaluate(
        self,
        missiles: Sequence[EntityState | Dict[str, NDArray[np.float64]]],
        enemies: Sequence[EntityState | Dict[str, NDArray[np.float64]]],
    ) -> NDArray[np.float64]:
        """
        Backward-compatible Version1 API.

        Use compute(SAMInputDTO(...)) for the DTO-based interface.
        """
        data = SAMInputDTO(
            missiles=coerce_entities(missiles, "missile_"),
            targets=coerce_entities(enemies, "target_"),
        )
        return self.compute(data).matrix_copy()

    @staticmethod
    def _build_pair_metrics(
        missiles: Sequence[EntityState],
        targets: Sequence[EntityState],
        distance: NDArray[np.float64],
        closing_speed: NDArray[np.float64],
        boresight_cos: NDArray[np.float64],
        in_fov: NDArray[np.bool_],
        closing_score: NDArray[np.float64],
        distance_score: NDArray[np.float64],
        boresight_score: NDArray[np.float64],
        energy_score: NDArray[np.float64],
        situation_matrix: NDArray[np.float64],
    ) -> Tuple[SAMPairMetricsDTO, ...]:
        metrics = []
        for i, missile in enumerate(missiles):
            for j, target in enumerate(targets):
                metrics.append(
                    SAMPairMetricsDTO(
                        missile_id=missile.id,
                        target_id=target.id,
                        range_m=float(distance[i, j]),
                        closing_speed_mps=float(closing_speed[i, j]),
                        boresight_cos=float(boresight_cos[i, j]),
                        in_fov=bool(in_fov[i, j]),
                        closing_score=float(closing_score[i, j]),
                        distance_score=float(distance_score[i, j]),
                        boresight_score=float(boresight_score[i, j]),
                        energy_score=float(energy_score[i, j]),
                        normalized_score=float(situation_matrix[i, j]),
                    )
                )
        return tuple(metrics)


if __name__ == "__main__":
    missiles: List[Dict[str, NDArray[np.float64]]] = [
        {
            "position": np.array([0.0, 8000.0, 0.0], dtype=np.float64),
            "velocity": np.array([500.0, -50.0, 100.0], dtype=np.float64),
        },
        {
            "position": np.array([5000.0, 9000.0, -2000.0], dtype=np.float64),
            "velocity": np.array([400.0, 20.0, 300.0], dtype=np.float64),
        },
    ]

    enemies: List[Dict[str, NDArray[np.float64]]] = [
        {
            "position": np.array([30000.0, 7000.0, 5000.0], dtype=np.float64),
            "velocity": np.array([-200.0, 10.0, -50.0], dtype=np.float64),
        },
        {
            "position": np.array([20000.0, 7500.0, -3000.0], dtype=np.float64),
            "velocity": np.array([-150.0, -5.0, 80.0], dtype=np.float64),
        },
        {
            "position": np.array([80000.0, 10000.0, 2000.0], dtype=np.float64),
            "velocity": np.array([-250.0, 0.0, 0.0], dtype=np.float64),
        },
    ]

    assessment = SituationAssessment()
    output = assessment.compute(
        SAMInputDTO(
            missiles=coerce_entities(missiles, "missile_"),
            targets=coerce_entities(enemies, "target_"),
        )
    )
    print(np.array2string(output.situation_matrix, precision=4, suppress_small=True))
