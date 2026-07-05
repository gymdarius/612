from __future__ import annotations

"""Version1 目标分配业务算法。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.

该文件是规则算法版本，不是训练模型。它把态势矩阵和威胁权重融合后，
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
对每枚有效导弹独立 softmax + argmax，输出从 0 开始的目标索引。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
训练数据中的 Y_AllocIndex 就由这里的 compute(...) 生成。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from .dto import (
        AllocationEntryDTO,
        AllocationInputDTO,
        AllocationOutputDTO,
        EntityState,
        SAMOutputDTO,
        ThreatOutputDTO,
        coerce_entities,
    )
except ImportError:  # pragma: no cover - keeps direct script execution working.
    from dto import (
        AllocationEntryDTO,
        AllocationInputDTO,
        AllocationOutputDTO,
        EntityState,
        SAMOutputDTO,
        ThreatOutputDTO,
        coerce_entities,
    )


class TargetAllocation:
    """
    Version1 target allocation logic with a Version2-style DTO interface.

    The original Version1 allocation strategy is preserved:
    value = w_advantage * situation + w_threat * target_threat_weight,
    followed by row-wise softmax and independent row argmax.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        weights = cfg.get("weights", {})
        self.w_advantage: float = weights.get("advantage", 0.99)
        self.w_threat: float = weights.get("threat", 0.01)

        norm_cfg = cfg.get("normalization", {})
        self.softmax_temperature: float = norm_cfg.get("temperature", 1.0)

        self.epsilon: float = float(cfg.get("epsilon", 1e-8))
        self.unassigned_target_id: str = str(cfg.get("unassigned_target_id", "UNASSIGNED"))
        if self.softmax_temperature <= 0.0:
            raise ValueError("softmax temperature must be > 0.")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be > 0.")

    def compute(self, data: AllocationInputDTO) -> AllocationOutputDTO:
        """DTO 主接口：输入实体和上游输出，返回结构化分配结果。"""
        # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
        missiles = tuple(data.missiles)
        targets = tuple(data.targets)
        missile_count = len(missiles)
        target_count = len(targets)

        situation_matrix = np.asarray(data.situation_output.situation_matrix, dtype=np.float64).copy()
        threat_matrix = np.asarray(data.threat_output.threat_matrix, dtype=np.float64).copy()
        target_threat_weights = np.asarray(data.threat_output.target_threat_weights, dtype=np.float64).reshape(-1)
        self._validate_dto_shapes(
            situation_matrix=situation_matrix,
            threat_matrix=threat_matrix,
            target_threat_weights=target_threat_weights,
            missile_count=missile_count,
            target_count=target_count,
        )

        if target_count == 0:
            empty_matrix = np.zeros((missile_count, 0), dtype=np.float64)
            assignments = tuple(
                AllocationEntryDTO(
                    missile_id=missile.id,
                    target_id=self.unassigned_target_id,
                    missile_index=i,
                    target_index=-1,
                    score=0.0,
                    probability=0.0,
                )
                for i, missile in enumerate(missiles)
            )
            return AllocationOutputDTO(
                assignments=assignments,
                normalized_matrix=empty_matrix,
                value_matrix=empty_matrix,
                target_threat_weights=target_threat_weights,
            )

        # value_matrix 是 Version1 分配决策的核心：态势优势为主，目标威胁权重为辅。
        # English note: This comment explains the following implementation detail for offline readability.
        value_matrix = self._compute_value_matrix(situation_matrix, target_threat_weights)
        normalized_matrix = self._row_softmax(value_matrix)

        if missile_count == 0:
            assignments: Tuple[AllocationEntryDTO, ...] = tuple()
        else:
            col_indices = np.argmax(normalized_matrix, axis=1)
            assignments = tuple(
                AllocationEntryDTO(
                    missile_id=missiles[i].id,
                    target_id=targets[int(col_indices[i])].id,
                    missile_index=i,
                    target_index=int(col_indices[i]),
                    score=float(value_matrix[i, int(col_indices[i])]),
                    probability=float(normalized_matrix[i, int(col_indices[i])]),
                )
                for i in range(missile_count)
            )

        return AllocationOutputDTO(
            assignments=assignments,
            normalized_matrix=normalized_matrix,
            value_matrix=value_matrix,
            target_threat_weights=target_threat_weights,
        )

    def allocate(
        self,
        advantage_matrix: NDArray[np.float64],
        threat_matrix: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], List[Tuple[int, int]]]:
        """
        Backward-compatible Version1 API.

        Use compute(AllocationInputDTO(...)) for the DTO-based interface.
        """
        advantage = np.asarray(advantage_matrix, dtype=np.float64).copy()
        threat = np.asarray(threat_matrix, dtype=np.float64).copy()
        self._validate_legacy_shapes(advantage, threat)

        missile_count, target_count = advantage.shape
        if missile_count == 0 or target_count == 0:
            return np.zeros((missile_count, target_count), dtype=np.float64), []

        threat_weights = self._compute_threat_weights(threat, target_count)
        value = self._compute_value_matrix(advantage, threat_weights)
        normalized = self._row_softmax(value)

        col_indices = np.argmax(normalized, axis=1)
        assignments = [(i, int(col_indices[i])) for i in range(missile_count)]
        return normalized, assignments

    def compute_from_matrices(
        self,
        missiles: Sequence[EntityState | Dict[str, NDArray[np.float64]]],
        targets: Sequence[EntityState | Dict[str, NDArray[np.float64]]],
        situation_matrix: NDArray[np.float64],
        threat_matrix: NDArray[np.float64],
    ) -> AllocationOutputDTO:
        """从矩阵直接构造 DTO 输出，便于离线调试和旧系统接入。"""
        # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
        missile_entities = coerce_entities(missiles, "missile_")
        target_entities = coerce_entities(targets, "target_")
        threat_weights = self._compute_threat_weights(np.asarray(threat_matrix, dtype=np.float64), len(target_entities))
        return self.compute(
            AllocationInputDTO(
                missiles=missile_entities,
                targets=target_entities,
                situation_output=SAMOutputDTO(situation_matrix=situation_matrix, pair_metrics=tuple()),
                threat_output=ThreatOutputDTO(
                    threat_matrix=threat_matrix,
                    target_threat_weights=threat_weights,
                    pair_metrics=tuple(),
                ),
            )
        )

    def _compute_value_matrix(
        self,
        situation_matrix: NDArray[np.float64],
        target_threat_weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """融合态势矩阵和目标威胁权重。"""
        # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
        return self.w_advantage * situation_matrix + self.w_threat * target_threat_weights[np.newaxis, :]

    def _compute_threat_weights(
        self,
        threat_matrix: NDArray[np.float64],
        target_count: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """兼容旧 allocate 接口：从威胁矩阵重新计算目标威胁权重。"""
        # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
        threat = np.asarray(threat_matrix, dtype=np.float64)
        if threat.ndim != 2:
            raise ValueError("threat_matrix must be 2D.")

        count = int(target_count if target_count is not None else threat.shape[1])
        if count == 0:
            return np.zeros((0,), dtype=np.float64)
        if threat.shape[1] != count:
            raise ValueError(f"threat_matrix target dimension mismatch: expected {count}, got {threat.shape[1]}.")
        if threat.shape[0] == 0:
            return np.full((count,), 1.0 / float(count), dtype=np.float64)

        raw = np.mean(threat, axis=0)
        total = float(np.sum(raw))
        if total > self.epsilon:
            return raw / total
        return np.full((count,), 1.0 / float(count), dtype=np.float64)

    def _row_softmax(self, value: NDArray[np.float64]) -> NDArray[np.float64]:
        """按导弹维度逐行 softmax，得到每枚导弹对各目标的概率分布。"""
        # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
        if value.size == 0:
            return np.zeros_like(value, dtype=np.float64)
        v = value / self.softmax_temperature
        v_max = np.max(v, axis=1, keepdims=True)
        exp_v = np.exp(v - v_max)
        return exp_v / np.sum(exp_v, axis=1, keepdims=True)

    @staticmethod
    def _validate_legacy_shapes(
        advantage_matrix: NDArray[np.float64],
        threat_matrix: NDArray[np.float64],
    ) -> None:
        if advantage_matrix.ndim != 2:
            raise ValueError("advantage_matrix must be 2D.")
        if threat_matrix.ndim != 2:
            raise ValueError("threat_matrix must be 2D.")
        if advantage_matrix.shape[1] != threat_matrix.shape[1]:
            raise ValueError(
                "target dimension mismatch: "
                f"advantage_matrix has {advantage_matrix.shape[1]}, threat_matrix has {threat_matrix.shape[1]}."
            )

    @staticmethod
    def _validate_dto_shapes(
        situation_matrix: NDArray[np.float64],
        threat_matrix: NDArray[np.float64],
        target_threat_weights: NDArray[np.float64],
        missile_count: int,
        target_count: int,
    ) -> None:
        if situation_matrix.shape != (missile_count, target_count):
            raise ValueError(
                "situation_matrix shape mismatch: "
                f"expected {(missile_count, target_count)}, got {situation_matrix.shape}."
            )
        if threat_matrix.ndim != 2 or threat_matrix.shape[1] != target_count:
            raise ValueError(
                "threat_matrix shape mismatch: "
                f"expected second dimension {target_count}, got {threat_matrix.shape}."
            )
        if target_threat_weights.shape != (target_count,):
            raise ValueError(
                "target_threat_weights shape mismatch: "
                f"expected {(target_count,)}, got {target_threat_weights.shape}."
            )


if __name__ == "__main__":
    from SituationAssessment import SituationAssessment
    from ThreatAssessment import ThreatAssessment
    from dto import AllocationInputDTO, SAMInputDTO, ThreatInputDTO

    missiles = [
        {
            "id": "missile_left",
            "position": np.array([5000.0, 10000.0, 5000.0], dtype=np.float64),
            "velocity": np.array([600.0, -500.0, 0.0], dtype=np.float64),
        },
        {
            "id": "missile_right",
            "position": np.array([5000.0, -10000.0, 5000.0], dtype=np.float64),
            "velocity": np.array([600.0, 500.0, 0.0], dtype=np.float64),
        },
    ]

    targets = [
        {
            "id": "enemy_left",
            "position": np.array([30000.0, 15000.0, 5000.0], dtype=np.float64),
            "velocity": np.array([-200.0, -50.0, 0.0], dtype=np.float64),
        },
        {
            "id": "enemy_right",
            "position": np.array([30000.0, -15000.0, 5000.0], dtype=np.float64),
            "velocity": np.array([-200.0, 50.0, 0.0], dtype=np.float64),
        },
    ]

    our_aircrafts = [
        {
            "id": "aircraft_left",
            "position": np.array([0.0, 10000.0, 5000.0], dtype=np.float64),
            "velocity": np.array([250.0, 0.0, 0.0], dtype=np.float64),
        },
        {
            "id": "aircraft_right",
            "position": np.array([0.0, -10000.0, 5000.0], dtype=np.float64),
            "velocity": np.array([250.0, 0.0, 0.0], dtype=np.float64),
        },
    ]

    missile_entities = coerce_entities(missiles, "missile_")
    target_entities = coerce_entities(targets, "target_")
    aircraft_entities = coerce_entities(our_aircrafts, "aircraft_")

    situation_output = SituationAssessment().compute(SAMInputDTO(missiles=missile_entities, targets=target_entities))
    threat_output = ThreatAssessment().compute(ThreatInputDTO(our_aircrafts=aircraft_entities, targets=target_entities))
    allocation_output = TargetAllocation().compute(
        AllocationInputDTO(
            missiles=missile_entities,
            targets=target_entities,
            situation_output=situation_output,
            threat_output=threat_output,
        )
    )

    print(np.array2string(allocation_output.normalized_matrix, precision=4, suppress_small=True))
    print(allocation_output.to_dict()["assignments"])
