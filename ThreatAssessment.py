from __future__ import annotations

"""Version1 威胁评估业务算法。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.

主接口 compute(ThreatInputDTO) 使用友方飞机和目标作为输入，输出 A x T
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
威胁矩阵以及按目标归一化的威胁权重。内部实现改为广播计算，但公式仍保留
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
Version1 原逻辑：LOS 指向、距离、速度比、高度差。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from .dto import (
        EntityState,
        ThreatInputDTO,
        ThreatOutputDTO,
        ThreatPairMetricsDTO,
        coerce_entities,
        stack_entity_vectors,
    )
except ImportError:  # pragma: no cover - keeps direct script execution working.
    from dto import (
        EntityState,
        ThreatInputDTO,
        ThreatOutputDTO,
        ThreatPairMetricsDTO,
        coerce_entities,
        stack_entity_vectors,
    )


class ThreatAssessment:
    """
    Version1 threat assessment logic with a Version2-style DTO interface.

    The original Version1 threat formula is preserved: enemy LOS pointing,
    distance, speed ratio, and height advantage are combined into a K x N
    threat matrix in [0, 1].
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        weights = cfg.get("weights", {})
        self.w_los: float = weights.get("los_angle", 0.40)
        self.w_distance: float = weights.get("distance", 0.30)
        self.w_speed: float = weights.get("speed", 0.15)
        self.w_height: float = weights.get("height", 0.15)
        total_weight = self.w_los + self.w_distance + self.w_speed + self.w_height
        if total_weight > 0:
            self.w_los /= total_weight
            self.w_distance /= total_weight
            self.w_speed /= total_weight
            self.w_height /= total_weight

        refs = cfg.get("reference_values", {})
        self.D_ref: float = refs.get("distance_ref", 100000.0)
        self.D_lethal: float = refs.get("lethal_distance", 50000.0)
        self.V_ref: float = refs.get("speed_ref", 400.0)

        los_cfg = cfg.get("los", {})
        self.los_cos_neutral: float = los_cfg.get("cos_neutral", 0.0)
        self.los_cos_lethal: float = los_cfg.get("cos_lethal", 0.866)

        self.epsilon: float = float(cfg.get("epsilon", 1e-8))
        self.export_pair_metrics: bool = bool(cfg.get("export_pair_metrics", True))
        if self.D_ref <= 0.0:
            raise ValueError("distance_ref must be > 0.")
        if self.D_lethal <= 0.0:
            raise ValueError("lethal_distance must be > 0.")
        if self.V_ref <= 0.0:
            raise ValueError("speed_ref must be > 0.")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be > 0.")

    def compute(self, data: ThreatInputDTO) -> ThreatOutputDTO:
        """计算 A x T 威胁矩阵和目标威胁权重。"""
        # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
        our_aircrafts = tuple(data.our_aircrafts)
        targets = tuple(data.targets)
        aircraft_count = len(our_aircrafts)
        target_count = len(targets)

        if aircraft_count == 0 or target_count == 0:
            empty = np.zeros((aircraft_count, target_count), dtype=np.float64)
            return ThreatOutputDTO(
                threat_matrix=empty,
                target_threat_weights=self._compute_target_threat_weights(empty, target_count),
                pair_metrics=tuple(),
            )

        aircraft_pos = stack_entity_vectors(our_aircrafts, "position")
        target_pos = stack_entity_vectors(targets, "position")
        aircraft_vel = stack_entity_vectors(our_aircrafts, "velocity")
        target_vel = stack_entity_vectors(targets, "velocity")
        aircraft_speed = np.asarray([entity.speed for entity in our_aircrafts], dtype=np.float64)
        target_speed = np.asarray([entity.speed for entity in targets], dtype=np.float64)

        # rel_pos[i, j] 表示目标 j 指向友方飞机 i 的视线向量。
        # English note: This comment explains the following implementation detail for offline readability.
        rel_pos = aircraft_pos[:, np.newaxis, :] - target_pos[np.newaxis, :, :]
        distance = np.linalg.norm(rel_pos, axis=2)
        distance_safe = np.maximum(distance, self.epsilon)
        zero_distance = distance < self.epsilon
        los = rel_pos / distance_safe[:, :, np.newaxis]

        target_dir = np.divide(
            target_vel,
            target_speed[:, np.newaxis],
            out=np.zeros_like(target_vel),
            where=target_speed[:, np.newaxis] > self.epsilon,
        )
        los_cos = np.clip(np.sum(target_dir[np.newaxis, :, :] * los, axis=2), -1.0, 1.0)
        # LOS 分量衡量敌方目标速度方向是否指向我方飞机。
        # English note: This comment explains the following implementation detail for offline readability.
        los_score = self._compute_los_score(los_cos)

        distance_inside = 0.8 + 0.2 * (1.0 - distance / max(self.D_lethal, self.epsilon))
        distance_outside = 0.8 * np.exp(-(distance - self.D_lethal) / self.D_ref)
        distance_score = np.where(distance <= self.D_lethal, distance_inside, distance_outside)

        speed_ratio = target_speed[np.newaxis, :] / np.maximum(aircraft_speed[:, np.newaxis], self.epsilon)
        speed_value = np.where(
            speed_ratio <= 0.6,
            0.1,
            np.where(speed_ratio >= 1.5, 1.0, speed_ratio - 0.5),
        )
        speed_score = np.clip(speed_value * np.exp(-distance / (2.0 * self.D_ref)), 0.0, 1.0)

        height_delta = target_pos[np.newaxis, :, 1] - aircraft_pos[:, np.newaxis, 1]
        height_score = np.where(
            height_delta > 2000.0,
            0.9,
            np.where(
                height_delta > 0.0,
                0.5 + 0.4 * (height_delta / 2000.0),
                np.where(height_delta > -2000.0, 0.5 * (1.0 + height_delta / 2000.0), 0.1),
            ),
        )

        # 四个威胁分量加权融合；零距离按最高威胁处理。
        # English note: This comment explains the following implementation detail for offline readability.
        threat_matrix = np.clip(
            self.w_los * los_score
            + self.w_distance * distance_score
            + self.w_speed * speed_score
            + self.w_height * height_score,
            0.0,
            1.0,
        )
        threat_matrix = np.where(zero_distance, 1.0, threat_matrix)
        target_threat_weights = self._compute_target_threat_weights(threat_matrix, target_count)

        pair_metrics = tuple()
        if self.export_pair_metrics:
            pair_metrics = self._build_pair_metrics(
                our_aircrafts=our_aircrafts,
                targets=targets,
                distance=distance,
                los_cos=los_cos,
                target_speed=target_speed,
                height_delta=height_delta,
                los_score=los_score,
                distance_score=distance_score,
                speed_score=speed_score,
                height_score=height_score,
                threat_matrix=threat_matrix,
            )

        return ThreatOutputDTO(
            threat_matrix=threat_matrix,
            target_threat_weights=target_threat_weights,
            pair_metrics=pair_metrics,
        )

    def evaluate(
        self,
        our_aircrafts: Sequence[EntityState | Dict[str, NDArray[np.float64]]],
        enemies: Sequence[EntityState | Dict[str, NDArray[np.float64]]],
    ) -> NDArray[np.float64]:
        """
        Backward-compatible Version1 API.

        Use compute(ThreatInputDTO(...)) for the DTO-based interface.
        """
        data = ThreatInputDTO(
            our_aircrafts=coerce_entities(our_aircrafts, "aircraft_"),
            targets=coerce_entities(enemies, "target_"),
        )
        return self.compute(data).matrix_copy()

    def _compute_los_score(self, los_cos: NDArray[np.float64]) -> NDArray[np.float64]:
        """按 Version1 分段函数把 LOS cos 转换为威胁分值。"""
        # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
        high_t = (los_cos - self.los_cos_lethal) / max(1.0 - self.los_cos_lethal, self.epsilon)
        mid_t = (los_cos - self.los_cos_neutral) / max(
            self.los_cos_lethal - self.los_cos_neutral,
            self.epsilon,
        )
        low_t = (los_cos + 1.0) / max(self.los_cos_neutral + 1.0, self.epsilon)
        return np.where(
            los_cos >= self.los_cos_lethal,
            0.8 + 0.2 * high_t,
            np.where(los_cos >= self.los_cos_neutral, 0.3 + 0.5 * mid_t, 0.1 * low_t),
        )

    def _compute_target_threat_weights(
        self,
        threat_matrix: NDArray[np.float64],
        target_count: int,
    ) -> NDArray[np.float64]:
        """把 A x T 威胁矩阵聚合成 T 维目标威胁权重。
        # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.

        该权重是目标分配模块使用的全局威胁偏置。没有友方飞机或总威胁为 0
        # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
        时退化为有效目标上的均匀分布。
        # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
        """
        if target_count == 0:
            return np.zeros((0,), dtype=np.float64)
        if threat_matrix.shape[0] == 0:
            return np.full((target_count,), 1.0 / float(target_count), dtype=np.float64)
        raw = np.mean(threat_matrix, axis=0)
        total = float(np.sum(raw))
        if total > self.epsilon:
            return raw / total
        return np.full((target_count,), 1.0 / float(target_count), dtype=np.float64)

    @staticmethod
    def _build_pair_metrics(
        our_aircrafts: Sequence[EntityState],
        targets: Sequence[EntityState],
        distance: NDArray[np.float64],
        los_cos: NDArray[np.float64],
        target_speed: NDArray[np.float64],
        height_delta: NDArray[np.float64],
        los_score: NDArray[np.float64],
        distance_score: NDArray[np.float64],
        speed_score: NDArray[np.float64],
        height_score: NDArray[np.float64],
        threat_matrix: NDArray[np.float64],
    ) -> Tuple[ThreatPairMetricsDTO, ...]:
        metrics = []
        for i, aircraft in enumerate(our_aircrafts):
            for j, target in enumerate(targets):
                metrics.append(
                    ThreatPairMetricsDTO(
                        aircraft_id=aircraft.id,
                        target_id=target.id,
                        range_m=float(distance[i, j]),
                        los_cos=float(los_cos[i, j]),
                        target_speed_mps=float(target_speed[j]),
                        height_delta_m=float(height_delta[i, j]),
                        los_score=float(los_score[i, j]),
                        distance_score=float(distance_score[i, j]),
                        speed_score=float(speed_score[i, j]),
                        height_score=float(height_score[i, j]),
                        normalized_score=float(threat_matrix[i, j]),
                    )
                )
        return tuple(metrics)


if __name__ == "__main__":
    our_aircrafts: List[Dict[str, NDArray[np.float64]]] = [
        {
            "position": np.array([0.0, 8000.0, 0.0], dtype=np.float64),
            "velocity": np.array([250.0, 0.0, 50.0], dtype=np.float64),
        },
        {
            "position": np.array([10000.0, 7000.0, -5000.0], dtype=np.float64),
            "velocity": np.array([200.0, -10.0, 80.0], dtype=np.float64),
        },
    ]

    enemies: List[Dict[str, NDArray[np.float64]]] = [
        {
            "position": np.array([30000.0, 9000.0, 2000.0], dtype=np.float64),
            "velocity": np.array([-300.0, 5.0, -20.0], dtype=np.float64),
        },
        {
            "position": np.array([5000.0, 6000.0, -1000.0], dtype=np.float64),
            "velocity": np.array([-200.0, 0.0, 100.0], dtype=np.float64),
        },
        {
            "position": np.array([80000.0, 8500.0, 8000.0], dtype=np.float64),
            "velocity": np.array([-150.0, 0.0, 0.0], dtype=np.float64),
        },
    ]

    assessment = ThreatAssessment()
    output = assessment.compute(
        ThreatInputDTO(
            our_aircrafts=coerce_entities(our_aircrafts, "aircraft_"),
            targets=coerce_entities(enemies, "target_"),
        )
    )
    print(np.array2string(output.threat_matrix, precision=4, suppress_small=True))
