"""专家规则配置及 YAML 加载。"""

from dataclasses import dataclass, field
from math import isclose
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class ComponentWeights:
    """距离、接近速度和方向角三个评分项的权重。"""

    distance: float = 0.4
    speed: float = 0.3
    angle: float = 0.3

    def validate(self, name: str) -> None:
        values = (self.distance, self.speed, self.angle)
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("{} weights must be within [0, 1].".format(name))
        if not isclose(sum(values), 1.0, abs_tol=1e-9):
            raise ValueError("{} weights must sum to 1.0.".format(name))


@dataclass(frozen=True)
class SituationConfig:
    """一类态势评分的权重及归一化尺度。"""

    weights: ComponentWeights = field(default_factory=ComponentWeights)
    distance_scale: float = 100_000.0
    closing_speed_scale: float = 500.0

    def validate(self, name: str) -> None:
        self.weights.validate(name)
        if self.distance_scale <= 0.0:
            raise ValueError("{}.distance_scale must be positive.".format(name))
        if self.closing_speed_scale <= 0.0:
            raise ValueError("{}.closing_speed_scale must be positive.".format(name))


@dataclass(frozen=True)
class CompositeWeights:
    """攻击优势和敌机威胁在综合分数中的占比。"""

    attack: float = 0.6
    threat: float = 0.4

    def validate(self) -> None:
        if not (0.0 <= self.attack <= 1.0 and 0.0 <= self.threat <= 1.0):
            raise ValueError("composite weights must be within [0, 1].")
        if not isclose(self.attack + self.threat, 1.0, abs_tol=1e-9):
            raise ValueError("composite weights must sum to 1.0.")


@dataclass(frozen=True)
class ExpertRuleConfig:
    """目标分配专家规则的全部可配置参数。"""

    threat: SituationConfig = field(default_factory=SituationConfig)
    attack: SituationConfig = field(
        default_factory=lambda: SituationConfig(closing_speed_scale=1_000.0)
    )
    composite: CompositeWeights = field(default_factory=CompositeWeights)
    epsilon: float = 1e-9

    def validate(self) -> None:
        self.threat.validate("threat")
        self.attack.validate("attack")
        self.composite.validate()
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")

    @classmethod
    def from_yaml(cls, path: str) -> "ExpertRuleConfig":
        """从 YAML 文件加载配置，并在使用前严格校验。"""

        with Path(path).open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        if not isinstance(data, dict):
            raise ValueError("The config root must be a mapping.")

        config = cls(
            threat=_situation_from_dict(data.get("threat", {}), 500.0),
            attack=_situation_from_dict(data.get("attack", {}), 1_000.0),
            composite=_composite_from_dict(data.get("composite", {})),
            epsilon=float(data.get("epsilon", 1e-9)),
        )
        config.validate()
        return config


def _situation_from_dict(data: Dict[str, Any], default_speed_scale: float) -> SituationConfig:
    weights = data.get("weights", {})
    return SituationConfig(
        weights=ComponentWeights(
            distance=float(weights.get("distance", 0.4)),
            speed=float(weights.get("speed", 0.3)),
            angle=float(weights.get("angle", 0.3)),
        ),
        distance_scale=float(data.get("distance_scale", 100_000.0)),
        closing_speed_scale=float(
            data.get("closing_speed_scale", default_speed_scale)
        ),
    )


def _composite_from_dict(data: Dict[str, Any]) -> CompositeWeights:
    return CompositeWeights(
        attack=float(data.get("attack", 0.6)),
        threat=float(data.get("threat", 0.4)),
    )
