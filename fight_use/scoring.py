"""攻击、防御和威胁评分模型。"""

from typing import Dict

from geometry import clamp, norm, safe_exp
from models import EPS, Entity, ExpertRuleConfig, Geometry


class ScoreModel:
    """集中实现所有底层评分公式，策略偏置不在本层处理。"""

    def __init__(self, config: ExpertRuleConfig):
        self.config = config

    @staticmethod
    def direction_score(geometry: Geometry) -> float:
        return clamp((geometry.direction_cosine + 1.0) / 2.0)

    def geometry_components(self, geometry: Geometry) -> Dict[str, float]:
        cfg = self.config.geometry
        attack = self.config.attack
        direction = self.direction_score(geometry)
        distance = safe_exp(-geometry.distance / cfg.engagement_distance_scale)
        cpa_distance = safe_exp(-geometry.d_cpa / cfg.cpa_distance_scale)
        closing = clamp(
            geometry.closing_speed / self.config.threat.closing_speed_reference
        )
        height = safe_exp(-geometry.height_difference / cfg.height_scale)
        angle_distance = (
            attack.direction_weight * direction
            + attack.distance_weight * distance
            + attack.cpa_distance_weight * cpa_distance
        )
        return {
            "direction": direction,
            "distance": distance,
            "cpa_distance": cpa_distance,
            "closing": closing,
            "height": height,
            "angle_distance": angle_distance,
        }

    def attack_geometry(self, geometry: Geometry) -> float:
        parts = self.geometry_components(geometry)
        cfg = self.config.attack
        return clamp(
            cfg.angle_distance_weight * parts["angle_distance"]
            + cfg.closing_weight * parts["closing"]
            + cfg.height_weight * parts["height"]
        )

    def intercept_geometry(self, geometry: Geometry) -> float:
        parts = self.geometry_components(geometry)
        cfg = self.config.defense
        return clamp(
            cfg.cpa_distance_weight * parts["cpa_distance"]
            + cfg.direction_weight * parts["direction"]
            + cfg.closing_weight * parts["closing"]
        )

    def is_feasible(self, geometry: Geometry) -> bool:
        cfg = self.config.geometry
        return (
            geometry.direction_cosine >= cfg.min_forward_cosine
            and geometry.closing_speed > EPS
            and geometry.t_cpa <= cfg.max_cpa_time
            and geometry.d_cpa <= cfg.max_feasible_cpa_distance
        )

    def missile_threat(self, geometry: Geometry) -> float:
        cfg = self.config.threat
        return clamp(
            cfg.missile_distance_weight
            * safe_exp(-geometry.d_cpa / cfg.protect_distance_scale)
            + cfg.missile_time_weight
            * clamp((cfg.warning_time - geometry.t_cpa) / cfg.warning_time)
            + cfg.missile_closing_weight
            * clamp(geometry.closing_speed / cfg.closing_speed_reference)
            + cfg.missile_direction_weight * self.direction_score(geometry)
        )

    def plane_threat(self, enemy_plane: Entity, geometry: Geometry) -> float:
        cfg = self.config
        proximity = safe_exp(-geometry.d_cpa / cfg.threat.protect_distance_scale)
        approach = self.direction_score(geometry)
        speed = clamp(norm(enemy_plane.velocity) / cfg.threat.closing_speed_reference)
        height = safe_exp(-geometry.height_difference / cfg.geometry.height_scale)
        # 敌机威胁沿用项目中的距离、方向、速度和高度组合。
        return clamp(0.45 * proximity + 0.25 * approach + 0.20 * speed + 0.10 * height)
