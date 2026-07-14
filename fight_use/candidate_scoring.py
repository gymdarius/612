"""计算全部友方导弹与全部候选目标之间的攻防评分。"""

from typing import Any, Dict, List, Sequence

from geometry import calculate_geometry, clamp, sigmoid
from models import Entity, ExpertRuleConfig
from scoring import ScoreModel


class CandidateScorer:
    """生成完整的 A×(X+Y) 评分、可达性和解释信息。"""

    def __init__(self, config: ExpertRuleConfig, score_model: ScoreModel):
        self.config = config
        self.score_model = score_model

    def score(
        self,
        friendly_missiles: Sequence[Entity],
        enemy_planes: Sequence[Entity],
        enemy_missiles: Sequence[Entity],
        enemy_plane_threats: Sequence[float],
        enemy_missile_threats: Sequence[Dict[str, Any]],
        x_max: int,
        t_max: int,
    ) -> Dict[str, Any]:
        rows = len(friendly_missiles)
        result = {
            "raw_attack_scores": [[0.0] * x_max for _ in range(rows)],
            "raw_defense_scores": [
                [0.0] * (t_max - x_max) for _ in range(rows)
            ],
            "raw_scores": [[0.0] * t_max for _ in range(rows)],
            "feasible": [[False] * t_max for _ in range(rows)],
            "emergency": [[False] * t_max for _ in range(rows)],
            "reasons": [[None] * t_max for _ in range(rows)],
        }

        for missile_index, missile in enumerate(friendly_missiles):
            self._score_attacks(
                missile_index, missile, enemy_planes, enemy_plane_threats, result
            )
            self._score_defenses(
                missile_index,
                missile,
                enemy_missiles,
                enemy_missile_threats,
                x_max,
                result,
            )
        return result

    def _score_attacks(
        self,
        row: int,
        missile: Entity,
        enemy_planes: Sequence[Entity],
        threats: Sequence[float],
        result: Dict[str, Any],
    ) -> None:
        minimum = self.config.attack.minimum_threat_factor
        for target_index, enemy_plane in enumerate(enemy_planes):
            geometry = calculate_geometry(missile, enemy_plane)
            geometry_score = self.score_model.attack_geometry(geometry)
            score = clamp(geometry_score * (minimum + (1.0 - minimum) * threats[target_index]))

            result["raw_attack_scores"][row][target_index] = score
            result["raw_scores"][row][target_index] = score
            result["feasible"][row][target_index] = self.score_model.is_feasible(geometry)
            result["reasons"][row][target_index] = self._reason(
                threats[target_index], geometry_score, geometry
            )

    def _score_defenses(
        self,
        row: int,
        missile: Entity,
        enemy_missiles: Sequence[Entity],
        threats: Sequence[Dict[str, Any]],
        x_max: int,
        result: Dict[str, Any],
    ) -> None:
        cfg = self.config
        for local_index, enemy_missile in enumerate(enemy_missiles):
            target_index = x_max + local_index
            geometry = calculate_geometry(missile, enemy_missile)
            geometry_score = self.score_model.intercept_geometry(geometry)
            detail = threats[local_index]
            enemy_time = detail["time_to_friend"]

            if enemy_time is None:
                time_advantage = 0.5
                intercept_before_threat = True
            else:
                time_advantage = sigmoid(
                    (enemy_time - geometry.t_cpa) / cfg.geometry.time_margin_scale
                )
                intercept_before_threat = geometry.t_cpa <= enemy_time

            score = clamp(
                detail["threat"] ** cfg.defense.threat_exponent
                * geometry_score
                * time_advantage
            )
            feasible = self.score_model.is_feasible(geometry) and intercept_before_threat
            emergency = (
                cfg.strategy.emergency_defense_enabled
                and detail["threat"] >= cfg.threat.emergency_threat
                and enemy_time is not None
                and enemy_time <= cfg.threat.emergency_time
                and feasible
            )

            result["raw_defense_scores"][row][local_index] = score
            result["raw_scores"][row][target_index] = score
            result["feasible"][row][target_index] = feasible
            result["emergency"][row][target_index] = emergency
            reason = self._reason(detail["threat"], geometry_score, geometry)
            reason.update(
                {
                    "time_advantage": time_advantage,
                    "threatened_friend_index": detail["threatened_friend_index"],
                }
            )
            result["reasons"][row][target_index] = reason

    @staticmethod
    def _reason(enemy_threat: float, geometry_score: float, geometry: Any) -> Dict[str, Any]:
        return {
            "enemy_threat": enemy_threat,
            "geometry_score": geometry_score,
            "time_advantage": None,
            "distance": geometry.distance,
            "closing_speed": geometry.closing_speed,
            "t_cpa": geometry.t_cpa,
            "d_cpa": geometry.d_cpa,
        }
