from __future__ import annotations

"""Version1 业务算法使用的数据契约 DTO。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.

这些类把原始字典输入统一成 EntityState，并给态势、威胁、分配三个模块
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
提供 Version2 风格的结构化输入/输出。旧版 List[dict] 输入仍可通过
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
coerce_entities 兼容，便于离线环境逐步迁移。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


Vector3 = Tuple[float, float, float]


def _as_vector(values: Sequence[float] | NDArray[np.float64], field_name: str) -> NDArray[np.float64]:
    """把输入转换为安全的 3D float64 向量。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.shape != (3,):
        raise ValueError(f"{field_name} must be a 3D vector.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{field_name} contains non-finite values.")
    return vector.copy()


@dataclass(frozen=True)
class EntityState:
    """导弹、友方飞机、目标共用的实体状态。
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.

    position 和 velocity 必须是 3 维向量。speed 如果不传或与 velocity
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    模长不一致，会自动以 velocity 模长为准，保证几何计算一致。
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    """

    id: str
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    speed: Optional[float] = None
    euler_angles: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def __post_init__(self) -> None:
        position = _as_vector(self.position, "position")
        velocity = _as_vector(self.velocity, "velocity")
        euler_angles = _as_vector(self.euler_angles, "euler_angles")

        computed_speed = float(np.linalg.norm(velocity))
        if self.speed is None:
            speed = computed_speed
        else:
            speed = float(self.speed)
            if not np.isfinite(speed) or speed < 0.0:
                raise ValueError("speed must be finite and >= 0.")
            if abs(speed - computed_speed) > max(1e-6, 1e-6 * computed_speed):
                speed = computed_speed

        object.__setattr__(self, "id", str(self.id))
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "velocity", velocity)
        object.__setattr__(self, "speed", speed)
        object.__setattr__(self, "euler_angles", euler_angles)

    @property
    def position_i_m(self) -> Vector3:
        return tuple(float(v) for v in self.position)

    @property
    def euler_321_rad(self) -> Vector3:
        return tuple(float(v) for v in self.euler_angles)

    @property
    def missile_id(self) -> str:
        return self.id

    @property
    def target_id(self) -> str:
        return self.id

    def position_array(self) -> NDArray[np.float64]:
        return self.position.copy()

    def velocity_array(self) -> NDArray[np.float64]:
        return self.velocity.copy()

    def euler_array(self) -> NDArray[np.float64]:
        return self.euler_angles.copy()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "speed": float(self.speed),
            "euler_angles": self.euler_angles.tolist(),
        }


MissileStateDTO = EntityState
TargetStateDTO = EntityState
AircraftStateDTO = EntityState


def entity_from_mapping(item: Mapping[str, Any], default_id: str) -> EntityState:
    """兼容旧字典输入，转换为 EntityState。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    return EntityState(
        id=str(item.get("id", default_id)),
        position=item["position"],
        velocity=item["velocity"],
        speed=item.get("speed"),
        euler_angles=item.get("euler_angles", np.zeros(3, dtype=np.float64)),
    )


def coerce_entities(items: Sequence[EntityState | Mapping[str, Any]], default_prefix: str) -> Tuple[EntityState, ...]:
    """把 EntityState/字典混合列表统一转换为 EntityState 元组。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    entities = []
    for index, item in enumerate(items):
        if isinstance(item, EntityState):
            entities.append(item)
        elif isinstance(item, Mapping):
            entities.append(entity_from_mapping(item, f"{default_prefix}{index}"))
        else:
            raise TypeError(f"{default_prefix} item {index} must be EntityState or mapping.")
    return tuple(entities)


def stack_entity_vectors(entities: Sequence[EntityState], field_name: str) -> NDArray[np.float64]:
    """把实体的 position 或 velocity 堆叠成 (N, 3) 矩阵。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    if not entities:
        return np.zeros((0, 3), dtype=np.float64)
    vectors = np.vstack([getattr(entity, field_name) for entity in entities]).astype(np.float64, copy=True)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError(f"{field_name} must be stackable to shape (N, 3).")
    return vectors


@dataclass(frozen=True)
class SAMInputDTO:
    """态势评估输入：导弹集合和目标集合。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    missiles: Tuple[EntityState, ...]
    targets: Tuple[EntityState, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "missiles", coerce_entities(self.missiles, "missile_"))
        object.__setattr__(self, "targets", coerce_entities(self.targets, "target_"))


@dataclass(frozen=True)
class SAMPairMetricsDTO:
    """单个导弹-目标 pair 的态势评估中间量，便于调试和离线验算。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    missile_id: str
    target_id: str
    range_m: float
    closing_speed_mps: float
    boresight_cos: float
    in_fov: bool
    closing_score: float
    distance_score: float
    boresight_score: float
    energy_score: float
    normalized_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "missile_id": self.missile_id,
            "target_id": self.target_id,
            "range_m": self.range_m,
            "closing_speed_mps": self.closing_speed_mps,
            "boresight_cos": self.boresight_cos,
            "in_fov": self.in_fov,
            "closing_score": self.closing_score,
            "distance_score": self.distance_score,
            "boresight_score": self.boresight_score,
            "energy_score": self.energy_score,
            "normalized_score": self.normalized_score,
        }


@dataclass(frozen=True)
class SAMOutputDTO:
    """态势评估输出：M x T 态势矩阵和可选 pair 级指标。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    situation_matrix: NDArray[np.float64]
    pair_metrics: Tuple[SAMPairMetricsDTO, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "situation_matrix", np.asarray(self.situation_matrix, dtype=np.float64).copy())
        object.__setattr__(self, "pair_metrics", tuple(self.pair_metrics))

    def matrix_copy(self) -> NDArray[np.float64]:
        return self.situation_matrix.copy()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "situation_matrix": self.situation_matrix.tolist(),
            "pair_metrics": [item.to_dict() for item in self.pair_metrics],
        }


@dataclass(frozen=True)
class ThreatInputDTO:
    """威胁评估输入：友方飞机集合和目标集合。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    our_aircrafts: Tuple[EntityState, ...]
    targets: Tuple[EntityState, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "our_aircrafts", coerce_entities(self.our_aircrafts, "aircraft_"))
        object.__setattr__(self, "targets", coerce_entities(self.targets, "target_"))


@dataclass(frozen=True)
class ThreatPairMetricsDTO:
    """单个友方飞机-目标 pair 的威胁评估中间量。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    aircraft_id: str
    target_id: str
    range_m: float
    los_cos: float
    target_speed_mps: float
    height_delta_m: float
    los_score: float
    distance_score: float
    speed_score: float
    height_score: float
    normalized_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aircraft_id": self.aircraft_id,
            "target_id": self.target_id,
            "range_m": self.range_m,
            "los_cos": self.los_cos,
            "target_speed_mps": self.target_speed_mps,
            "height_delta_m": self.height_delta_m,
            "los_score": self.los_score,
            "distance_score": self.distance_score,
            "speed_score": self.speed_score,
            "height_score": self.height_score,
            "normalized_score": self.normalized_score,
        }


@dataclass(frozen=True)
class ThreatOutputDTO:
    """威胁评估输出。
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.

    threat_matrix 是 A x T 威胁矩阵；target_threat_weights 是按目标聚合后
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    归一化的威胁权重，后续目标分配直接使用它。
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    """
    threat_matrix: NDArray[np.float64]
    target_threat_weights: NDArray[np.float64]
    pair_metrics: Tuple[ThreatPairMetricsDTO, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "threat_matrix", np.asarray(self.threat_matrix, dtype=np.float64).copy())
        object.__setattr__(
            self,
            "target_threat_weights",
            np.asarray(self.target_threat_weights, dtype=np.float64).reshape(-1).copy(),
        )
        object.__setattr__(self, "pair_metrics", tuple(self.pair_metrics))

    def matrix_copy(self) -> NDArray[np.float64]:
        return self.threat_matrix.copy()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threat_matrix": self.threat_matrix.tolist(),
            "target_threat_weights": self.target_threat_weights.tolist(),
            "pair_metrics": [item.to_dict() for item in self.pair_metrics],
        }


@dataclass(frozen=True)
class AllocationInputDTO:
    """目标分配输入：实体列表 + 态势输出 + 威胁输出。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    missiles: Tuple[EntityState, ...]
    targets: Tuple[EntityState, ...]
    situation_output: SAMOutputDTO
    threat_output: ThreatOutputDTO

    def __post_init__(self) -> None:
        object.__setattr__(self, "missiles", coerce_entities(self.missiles, "missile_"))
        object.__setattr__(self, "targets", coerce_entities(self.targets, "target_"))


@dataclass(frozen=True)
class AllocationEntryDTO:
    """单枚导弹的分配结果。
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.

    target_index 是从 0 开始的目标序号；训练标签 Y_AllocIndex 使用同样约定。
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    """
    missile_id: str
    target_id: str
    tactical_mode: str = "attack"
    missile_index: int = -1
    target_index: int = -1
    score: float = 0.0
    probability: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "missile_id": self.missile_id,
            "target_id": self.target_id,
            "tactical_mode": self.tactical_mode,
            "missile_index": self.missile_index,
            "target_index": self.target_index,
            "score": self.score,
            "probability": self.probability,
        }


@dataclass(frozen=True)
class AllocationOutputDTO:
    """目标分配输出：分配列表、softmax 矩阵、原始价值矩阵和威胁权重。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    assignments: Tuple[AllocationEntryDTO, ...]
    normalized_matrix: NDArray[np.float64]
    value_matrix: NDArray[np.float64]
    target_threat_weights: NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignments", tuple(self.assignments))
        object.__setattr__(self, "normalized_matrix", np.asarray(self.normalized_matrix, dtype=np.float64).copy())
        object.__setattr__(self, "value_matrix", np.asarray(self.value_matrix, dtype=np.float64).copy())
        object.__setattr__(
            self,
            "target_threat_weights",
            np.asarray(self.target_threat_weights, dtype=np.float64).reshape(-1).copy(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assignments": [item.to_dict() for item in self.assignments],
            "normalized_matrix": self.normalized_matrix.tolist(),
            "value_matrix": self.value_matrix.tolist(),
            "target_threat_weights": self.target_threat_weights.tolist(),
        }
