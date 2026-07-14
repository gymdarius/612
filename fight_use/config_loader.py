"""从JSON加载并校验专家规则配置。"""

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

from models import (
    AttackConfig,
    DefenseConfig,
    ExpertRuleConfig,
    FallbackConfig,
    GeometryConfig,
    StrategyConfig,
    SystemConfig,
    ThreatConfig,
)


T = TypeVar("T")


class ConfigLoader:
    """配置加载器；缺失字段使用默认值，未知字段直接报错。"""

    SECTIONS = {
        "system": SystemConfig,
        "strategy": StrategyConfig,
        "threat": ThreatConfig,
        "geometry": GeometryConfig,
        "attack": AttackConfig,
        "defense": DefenseConfig,
        "fallback": FallbackConfig,
    }

    @classmethod
    def load(cls, path: Any) -> ExpertRuleConfig:
        with Path(path).open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExpertRuleConfig:
        unknown = set(data) - set(cls.SECTIONS)
        if unknown:
            raise ValueError("Unknown config sections: " + ", ".join(sorted(unknown)))

        values = {
            name: cls._section(section_type, data.get(name, {}), name)
            for name, section_type in cls.SECTIONS.items()
        }
        config = ExpertRuleConfig(**values)
        cls._validate(config)
        return config

    @staticmethod
    def _section(section_type: Type[T], data: Dict[str, Any], name: str) -> T:
        if not isinstance(data, dict):
            raise ValueError(f"Config section '{name}' must be an object.")
        allowed = {item.name for item in fields(section_type)}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(
                f"Unknown fields in '{name}': " + ", ".join(sorted(unknown))
            )
        return section_type(**data)

    @classmethod
    def _validate(cls, config: ExpertRuleConfig) -> None:
        if not -1.0 <= config.strategy.strategy_bias <= 1.0:
            raise ValueError("strategy_bias must be between -1 and 1.")
        if config.strategy.bias_strength < 0:
            raise ValueError("bias_strength cannot be negative.")
        if not 0.0 <= config.attack.minimum_threat_factor <= 1.0:
            raise ValueError("minimum_threat_factor must be between 0 and 1.")
        if not 0.0 <= config.threat.emergency_threat <= 1.0:
            raise ValueError("emergency_threat must be between 0 and 1.")
        if config.defense.threat_exponent <= 0:
            raise ValueError("threat_exponent must be greater than zero.")
        if config.fallback.score_epsilon < 0:
            raise ValueError("score_epsilon cannot be negative.")
        if min(
            config.system.a_max,
            config.system.f_max,
            config.system.x_max,
            config.system.y_max,
        ) < 0:
            raise ValueError("Entity maximum counts cannot be negative.")
        if config.geometry.min_forward_cosine < -1 or config.geometry.min_forward_cosine > 1:
            raise ValueError("min_forward_cosine must be between -1 and 1.")

        positive = {
            "protect_distance_scale": config.threat.protect_distance_scale,
            "warning_time": config.threat.warning_time,
            "closing_speed_reference": config.threat.closing_speed_reference,
            "engagement_distance_scale": config.geometry.engagement_distance_scale,
            "cpa_distance_scale": config.geometry.cpa_distance_scale,
            "height_scale": config.geometry.height_scale,
            "max_cpa_time": config.geometry.max_cpa_time,
            "max_feasible_cpa_distance": config.geometry.max_feasible_cpa_distance,
            "time_margin_scale": config.geometry.time_margin_scale,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero.")

        cls._weights(
            "missile threat",
            config.threat.missile_distance_weight,
            config.threat.missile_time_weight,
            config.threat.missile_closing_weight,
            config.threat.missile_direction_weight,
        )
        cls._weights(
            "attack situation",
            config.attack.angle_distance_weight,
            config.attack.closing_weight,
            config.attack.height_weight,
        )
        cls._weights(
            "attack angle-distance",
            config.attack.direction_weight,
            config.attack.distance_weight,
            config.attack.cpa_distance_weight,
        )
        cls._weights(
            "defense geometry",
            config.defense.cpa_distance_weight,
            config.defense.direction_weight,
            config.defense.closing_weight,
        )

    @staticmethod
    def _weights(name: str, *values: float) -> None:
        if any(value < 0 for value in values) or abs(sum(values) - 1.0) > 1e-6:
            raise ValueError(f"{name} weights must be non-negative and sum to 1.")
