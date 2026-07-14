"""基于距离、接近速度和方向角的目标分配专家规则。"""

from math import isfinite
from pathlib import Path
from typing import List, Sequence, Tuple

from config import ExpertRuleConfig, SituationConfig
from geometry import RelativeGeometryCalculator, State, angle_score, distance_score, speed_score
from models import AssignmentResult, MissileAssignment, ScoreBreakdown, ThreatResult


class TargetAssignmentExpert:
    """为每枚待发射导弹独立选择综合分数最高的敌机。"""

    def __init__(self, config: ExpertRuleConfig = None) -> None:
        self.config = config or ExpertRuleConfig()
        self.config.validate()
        self.geometry = RelativeGeometryCalculator(self.config.epsilon)

    @classmethod
    def from_yaml(cls, path: str) -> "TargetAssignmentExpert":
        return cls(ExpertRuleConfig.from_yaml(path))

    def assign(
        self,
        friendly_planes: Sequence[Sequence[float]],
        friendly_missiles: Sequence[Sequence[float]],
        enemy_planes: Sequence[Sequence[float]],
    ) -> AssignmentResult:
        """接收三个二维数组，计算评分矩阵并返回每枚导弹的目标索引。"""

        friends = self._validate_states(friendly_planes, "friendly_planes", False)
        missiles = self._validate_states(friendly_missiles, "friendly_missiles", True)
        enemies = self._validate_states(enemy_planes, "enemy_planes", False)

        threat_details, pair_threat_matrix = self._evaluate_threats(friends, enemies)
        threat_scores = [detail.score for detail in threat_details]
        attack_matrix, attack_details = self._evaluate_attacks(missiles, enemies)
        composite_matrix = self._combine_scores(attack_matrix, threat_scores)
        missile_details = self._select_targets(
            composite_matrix, attack_matrix, attack_details, threat_scores
        )
        return AssignmentResult(
            assignments=[detail.target_index for detail in missile_details],
            enemy_threat_scores=threat_scores,
            pair_threat_score_matrix=pair_threat_matrix,
            attack_score_matrix=attack_matrix,
            composite_score_matrix=composite_matrix,
            threat_details=threat_details,
            missile_details=missile_details,
        )

    def _evaluate_threats(
        self, friends: Sequence[State], enemies: Sequence[State]
    ) -> Tuple[List[ThreatResult], List[List[float]]]:
        details: List[ThreatResult] = []
        matrix: List[List[float]] = []
        for enemy_index, enemy in enumerate(enemies):
            breakdowns = [
                self._score_situation(enemy, friend, self.config.threat)
                for friend in friends
            ]
            scores = [item.total for item in breakdowns]
            friend_index = max(range(len(friends)), key=lambda index: scores[index])
            matrix.append(scores)
            details.append(
                ThreatResult(
                    enemy_index,
                    scores[friend_index],
                    friend_index,
                    breakdowns[friend_index],
                )
            )
        return details, matrix

    def _evaluate_attacks(
        self, missiles: Sequence[State], enemies: Sequence[State]
    ) -> Tuple[List[List[float]], List[List[ScoreBreakdown]]]:
        matrix: List[List[float]] = []
        details: List[List[ScoreBreakdown]] = []
        for missile in missiles:
            row = [
                self._score_situation(missile, enemy, self.config.attack)
                for enemy in enemies
            ]
            details.append(row)
            matrix.append([item.total for item in row])
        return matrix, details

    def _score_situation(
        self, source: State, target: State, config: SituationConfig
    ) -> ScoreBreakdown:
        geometry = self.geometry.calculate(source, target)
        distance = distance_score(geometry.distance, config.distance_scale)
        speed = speed_score(geometry.closing_speed, config.closing_speed_scale)
        angle = angle_score(geometry.direction_cosine)
        total = (
            config.weights.distance * distance
            + config.weights.speed * speed
            + config.weights.angle * angle
        )
        return ScoreBreakdown(distance, speed, angle, total)

    def _combine_scores(
        self,
        attack_matrix: Sequence[Sequence[float]],
        threat_scores: Sequence[float],
    ) -> List[List[float]]:
        a = self.config.composite.attack
        b = self.config.composite.threat
        return [
            [
                a * attack_score + b * threat_scores[target_index]
                for target_index, attack_score in enumerate(row)
            ]
            for row in attack_matrix
        ]

    def _select_targets(
        self,
        composite_matrix: Sequence[Sequence[float]],
        attack_matrix: Sequence[Sequence[float]],
        attack_details: Sequence[Sequence[ScoreBreakdown]],
        threat_scores: Sequence[float],
    ) -> List[MissileAssignment]:
        results: List[MissileAssignment] = []
        for missile_index, row in enumerate(composite_matrix):
            # 并列时依次比较威胁、攻击优势，最后稳定选择较小索引。
            target_index = max(
                range(len(row)),
                key=lambda index: (
                    row[index],
                    threat_scores[index],
                    attack_matrix[missile_index][index],
                    -index,
                ),
            )
            sorted_scores = sorted(row, reverse=True)
            margin = (
                sorted_scores[0] - sorted_scores[1]
                if len(sorted_scores) > 1
                else sorted_scores[0]
            )
            selected_score = row[target_index]
            results.append(
                MissileAssignment(
                    missile_index,
                    target_index,
                    attack_matrix[missile_index][target_index],
                    threat_scores[target_index],
                    selected_score,
                    margin,
                    attack_details[missile_index][target_index],
                    selected_score <= self.config.epsilon,
                )
            )
        return results

    @staticmethod
    def _validate_states(
        states: Sequence[Sequence[float]], name: str, allow_empty: bool
    ) -> List[State]:
        if states is None or isinstance(states, (str, bytes)):
            raise ValueError("{} must be a two-dimensional array.".format(name))
        if not allow_empty and len(states) == 0:
            raise ValueError("{} must not be empty.".format(name))

        converted: List[State] = []
        for index, state in enumerate(states):
            if isinstance(state, (str, bytes)) or len(state) != 6:
                raise ValueError(
                    "{}[{}] must contain [x, y, z, vx, vy, vz].".format(name, index)
                )
            values = tuple(float(value) for value in state)
            if not all(isfinite(value) for value in values):
                raise ValueError("{}[{}] contains a non-finite value.".format(name, index))
            converted.append(values)  # type: ignore[arg-type]
        return converted


def create_default_expert() -> TargetAssignmentExpert:
    """使用同目录下的默认 YAML 创建专家规则。"""

    return TargetAssignmentExpert.from_yaml(str(Path(__file__).with_name("config.yaml")))
