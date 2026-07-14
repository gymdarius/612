"""专家规则的数据对象与分层配置。"""

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Sequence, Tuple


INVALID_SCORE = -1e9
EPS = 1e-9


@dataclass(frozen=True)
class Entity:
    """三维位置和速度，单位分别为米和米/秒，y轴表示高度。"""

    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    entity_id: Optional[Any] = None

    @property
    def position(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.z

    @property
    def velocity(self) -> Tuple[float, float, float]:
        return self.vx, self.vy, self.vz


@dataclass(frozen=True)
class Geometry:
    """源实体相对目标实体的运动特征。"""

    distance: float
    closing_speed: float
    direction_cosine: float
    direction_angle_rad: float
    t_cpa: float
    d_cpa: float
    height_difference: float


@dataclass
class SystemConfig:
    a_max: int = 8
    f_max: int = 4
    x_max: int = 6
    y_max: int = 8
    force_assignment: bool = True


@dataclass
class StrategyConfig:
    # -1 强攻击，0 中性，1 强防御。
    strategy_bias: float = 0.0
    bias_strength: float = 0.7
    emergency_defense_enabled: bool = True
    emergency_overrides_strategy: bool = True


@dataclass
class ThreatConfig:
    protect_distance_scale: float = 150_000.0
    warning_time: float = 60.0
    closing_speed_reference: float = 500.0
    missile_distance_weight: float = 0.45
    missile_time_weight: float = 0.25
    missile_closing_weight: float = 0.20
    missile_direction_weight: float = 0.10
    emergency_threat: float = 0.80
    emergency_time: float = 20.0


@dataclass
class GeometryConfig:
    engagement_distance_scale: float = 150_000.0
    cpa_distance_scale: float = 15_000.0
    height_scale: float = 20_000.0
    max_cpa_time: float = 180.0
    max_feasible_cpa_distance: float = 20_000.0
    min_forward_cosine: float = -0.20
    time_margin_scale: float = 10.0


@dataclass
class AttackConfig:
    angle_distance_weight: float = 0.76
    closing_weight: float = 0.14
    height_weight: float = 0.10
    direction_weight: float = 0.55
    distance_weight: float = 0.25
    cpa_distance_weight: float = 0.20
    minimum_threat_factor: float = 0.50


@dataclass
class DefenseConfig:
    threat_exponent: float = 1.20
    cpa_distance_weight: float = 0.40
    direction_weight: float = 0.35
    closing_weight: float = 0.25


@dataclass
class FallbackConfig:
    relax_geometry_when_unreachable: bool = True
    prefer_defense_when_scores_equal: bool = True
    score_epsilon: float = 1e-6


@dataclass
class ExpertRuleConfig:
    """专家规则完整配置，按职责分组以便独立调参。"""

    system: SystemConfig = field(default_factory=SystemConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    threat: ThreatConfig = field(default_factory=ThreatConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
    fallback: FallbackConfig = field(default_factory=FallbackConfig)


def to_entity(value: Any, fallback_id: Any = None) -> Entity:
    """接受六维数组或带 ``id/state`` 的字典。"""

    if isinstance(value, Entity):
        return value
    if isinstance(value, Mapping):
        state = value.get("state")
        if state is None:
            state = [value[key] for key in ("x", "y", "z", "vx", "vy", "vz")]
        entity_id = value.get("id", value.get("entity_id", fallback_id))
    else:
        state, entity_id = value, fallback_id

    if (
        not isinstance(state, Sequence)
        or isinstance(state, (str, bytes))
        or len(state) != 6
    ):
        raise ValueError("Each entity must contain [x, y, z, vx, vy, vz].")
    return Entity(*(float(number) for number in state), entity_id=entity_id)


def to_entities(values: Sequence[Any], prefix: str) -> List[Entity]:
    return [to_entity(value, f"{prefix}_{index}") for index, value in enumerate(values)]
