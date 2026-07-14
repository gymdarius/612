"""无容量限制下，为每枚友弹独立选择最高分目标。"""

from math import exp
from typing import Any, Dict, List, Sequence

from models import INVALID_SCORE, ExpertRuleConfig


class TargetSelector:
    """应用攻防策略偏置、紧急覆盖和不可达兜底。"""

    def __init__(self, config: ExpertRuleConfig):
        self.config = config
        bias = config.strategy.strategy_bias * config.strategy.bias_strength
        self.attack_multiplier = exp(-bias)
        self.defense_multiplier = exp(bias)

    def select(
        self,
        scores: Dict[str, Any],
        target_mask: Sequence[int],
        x_max: int,
    ) -> Dict[str, Any]:
        final_scores = self._apply_strategy(scores["raw_scores"], target_mask, x_max)
        masked_scores = []
        assignment = []
        decisions = []

        for row, row_scores in enumerate(final_scores):
            valid = [
                bool(target_mask[index]) and scores["feasible"][row][index]
                for index in range(len(target_mask))
            ]
            masked = [
                row_scores[index] if valid[index] else INVALID_SCORE
                for index in range(len(target_mask))
            ]
            selected, fallback, emergency = self._select_row(
                row, row_scores, valid, target_mask, scores, x_max
            )
            masked_scores.append(masked)
            assignment.append(selected)
            decisions.append(
                self._decision(
                    row, selected, x_max, scores, row_scores, fallback, emergency
                )
            )

        return {
            "assignment": assignment,
            "final_scores": final_scores,
            "masked_scores": masked_scores,
            "decisions": decisions,
            "attack_multiplier": self.attack_multiplier,
            "defense_multiplier": self.defense_multiplier,
        }

    def _apply_strategy(
        self, raw_scores: Sequence[Sequence[float]], target_mask: Sequence[int], x_max: int
    ) -> List[List[float]]:
        return [
            [
                score * (self.attack_multiplier if index < x_max else self.defense_multiplier)
                if target_mask[index]
                else 0.0
                for index, score in enumerate(row)
            ]
            for row in raw_scores
        ]

    def _select_row(
        self,
        row: int,
        row_scores: Sequence[float],
        valid: Sequence[bool],
        target_mask: Sequence[int],
        scores: Dict[str, Any],
        x_max: int,
    ):
        strategy = self.config.strategy
        emergency_targets = [
            index
            for index in range(x_max, len(target_mask))
            if target_mask[index] and scores["emergency"][row][index]
        ]
        if strategy.emergency_overrides_strategy and emergency_targets:
            return self._best(emergency_targets, row_scores, x_max), False, True

        valid_targets = [index for index, available in enumerate(valid) if available]
        if valid_targets:
            return self._best(valid_targets, row_scores, x_max), False, False

        if not self.config.system.force_assignment:
            return -1, True, False

        real_targets = [index for index, exists in enumerate(target_mask) if exists]
        return self._best(real_targets, row_scores, x_max), True, False

    def _best(self, candidates: Sequence[int], scores: Sequence[float], x_max: int) -> int:
        epsilon = self.config.fallback.score_epsilon
        best_score = max(scores[index] for index in candidates)
        tied = [index for index in candidates if best_score - scores[index] <= epsilon]
        if self.config.fallback.prefer_defense_when_scores_equal:
            defense = [index for index in tied if index >= x_max]
            if defense:
                return min(defense)
        return min(tied)

    @staticmethod
    def _decision(
        missile_index: int,
        selected: int,
        x_max: int,
        scores: Dict[str, Any],
        final_scores: Sequence[float],
        fallback: bool,
        emergency: bool,
    ) -> Dict[str, Any]:
        if selected < 0:
            return {
                "friendly_missile_index": missile_index,
                "action": None,
                "target_index": -1,
                "fallback": fallback,
            }
        action = "ATTACK" if selected < x_max else "DEFEND"
        return {
            "friendly_missile_index": missile_index,
            "action": action,
            "target_type": "enemy_plane" if action == "ATTACK" else "enemy_missile",
            "target_index": selected,
            "target_local_index": selected if action == "ATTACK" else selected - x_max,
            "raw_score": scores["raw_scores"][missile_index][selected],
            "strategy_adjusted_score": final_scores[selected],
            "emergency_override": emergency,
            "fallback": fallback,
            "reason": scores["reasons"][missile_index][selected],
        }
