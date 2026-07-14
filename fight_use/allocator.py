"""面向对象的专家规则引擎。"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from candidate_scoring import CandidateScorer
from config_loader import ConfigLoader
from models import INVALID_SCORE, ExpertRuleConfig, to_entities
from scoring import ScoreModel
from target_selection import TargetSelector
from threat_evaluation import ThreatEvaluator


class ExpertRuleEngine:
    """为全部无目标友方导弹分配敌机或敌弹目标。"""

    DEFAULT_CONFIG = Path(__file__).with_name("expert_rule_config.json")

    def __init__(
        self,
        config: Optional[ExpertRuleConfig] = None,
        config_path: Optional[Any] = None,
    ):
        if config is not None and config_path is not None:
            raise ValueError("Provide config or config_path, not both.")
        self.config = config or ConfigLoader.load(config_path or self.DEFAULT_CONFIG)
        self.score_model = ScoreModel(self.config)
        self.threat_evaluator = ThreatEvaluator(self.score_model)
        self.candidate_scorer = CandidateScorer(self.config, self.score_model)
        self.target_selector = TargetSelector(self.config)

    def assign(
        self,
        friendly_missiles: Sequence[Any],
        friendly_planes: Sequence[Any],
        enemy_planes: Sequence[Any],
        enemy_missiles: Sequence[Any],
    ) -> Dict[str, Any]:
        """为每枚真实友弹分配一个目标，目标之间没有容量限制。"""

        missiles = to_entities(friendly_missiles, "friendly_missile")
        friends = to_entities(friendly_planes, "friendly_plane")
        planes = to_entities(enemy_planes, "enemy_plane")
        enemy_missiles_e = to_entities(enemy_missiles, "enemy_missile")
        self._validate_counts(missiles, friends, planes, enemy_missiles_e)

        cfg = self.config.system
        t_max = cfg.x_max + cfg.y_max
        target_mask = self._target_mask(len(planes), len(enemy_missiles_e))

        plane_threats = self.threat_evaluator.enemy_planes(planes, friends)
        missile_threats = self.threat_evaluator.enemy_missiles(
            enemy_missiles_e, friends
        )
        scores = self.candidate_scorer.score(
            missiles,
            planes,
            enemy_missiles_e,
            plane_threats,
            missile_threats,
            cfg.x_max,
            t_max,
        )
        selection = self.target_selector.select(scores, target_mask, cfg.x_max)
        return self._build_output(
            missiles,
            friends,
            planes,
            enemy_missiles_e,
            target_mask,
            scores,
            selection,
        )

    def _validate_counts(self, missiles, friends, planes, enemy_missiles) -> None:
        cfg = self.config.system
        counts = {
            "friendly missiles": (len(missiles), cfg.a_max),
            "friendly planes": (len(friends), cfg.f_max),
            "enemy planes": (len(planes), cfg.x_max),
            "enemy missiles": (len(enemy_missiles), cfg.y_max),
        }
        for name, (actual, maximum) in counts.items():
            if actual > maximum:
                raise ValueError(f"Too many {name}: {actual} > configured {maximum}.")
        if not missiles:
            raise ValueError("At least one friendly missile is required.")
        if not planes and not enemy_missiles:
            raise ValueError("At least one real enemy target is required.")

    def _target_mask(self, plane_count: int, missile_count: int) -> List[int]:
        cfg = self.config.system
        mask = [0] * (cfg.x_max + cfg.y_max)
        for index in range(plane_count):
            mask[index] = 1
        for index in range(missile_count):
            mask[cfg.x_max + index] = 1
        return mask

    def _pad_rows(
        self, rows: Sequence[Sequence[float]], fill: float, width: Optional[int] = None
    ) -> List[List[float]]:
        result = [list(row) for row in rows]
        width = width or self.config.system.x_max + self.config.system.y_max
        result.extend(
            [[fill] * width for _ in range(self.config.system.a_max - len(result))]
        )
        return result

    def _build_output(
        self, missiles, friends, planes, enemy_missiles, target_mask, scores, selection
    ) -> Dict[str, Any]:
        cfg = self.config.system
        assignment = selection["assignment"] + [-1] * (
            cfg.a_max - len(selection["assignment"])
        )

        # 为每条决策补充业务 ID；固定编号仍是训练标签和接口主键。
        for decision in selection["decisions"]:
            if decision["target_index"] < 0:
                continue
            local_index = decision["target_local_index"]
            targets = planes if decision["action"] == "ATTACK" else enemy_missiles
            decision["friendly_missile_id"] = missiles[
                decision["friendly_missile_index"]
            ].entity_id
            decision["target_id"] = targets[local_index].entity_id

        return {
            "assignment": assignment,
            "raw_attack_scores": self._pad_rows(
                scores["raw_attack_scores"], 0.0, cfg.x_max
            ),
            "raw_defense_scores": self._pad_rows(
                scores["raw_defense_scores"], 0.0, cfg.y_max
            ),
            "raw_expert_scores": self._pad_rows(scores["raw_scores"], 0.0),
            "final_expert_scores": self._pad_rows(selection["final_scores"], 0.0),
            "masked_expert_scores": self._pad_rows(
                selection["masked_scores"], INVALID_SCORE
            ),
            "friendly_missile_mask": [1] * len(missiles)
            + [0] * (cfg.a_max - len(missiles)),
            "friendly_plane_mask": [1] * len(friends)
            + [0] * (cfg.f_max - len(friends)),
            "enemy_plane_mask": [1] * len(planes) + [0] * (cfg.x_max - len(planes)),
            "enemy_missile_mask": [1] * len(enemy_missiles)
            + [0] * (cfg.y_max - len(enemy_missiles)),
            "target_mask": target_mask,
            "decisions": selection["decisions"],
            "global_total_score": sum(
                item.get("strategy_adjusted_score", 0.0)
                for item in selection["decisions"]
            ),
            "strategy": {
                "strategy_bias": self.config.strategy.strategy_bias,
                "attack_multiplier": selection["attack_multiplier"],
                "defense_multiplier": selection["defense_multiplier"],
            },
            "config": asdict(self.config),
        }


def assign_all_missiles(
    friendly_missiles: Sequence[Any],
    friendly_planes: Sequence[Any],
    enemy_planes: Sequence[Any],
    enemy_missiles: Sequence[Any],
    *,
    config: Optional[ExpertRuleConfig] = None,
    config_path: Optional[Any] = None,
) -> Dict[str, Any]:
    """函数式便捷入口；高频调用时建议复用 ExpertRuleEngine 实例。"""

    return ExpertRuleEngine(config=config, config_path=config_path).assign(
        friendly_missiles, friendly_planes, enemy_planes, enemy_missiles
    )


# 兼容第一版函数名，但语义已变为给全部友弹分配目标。
assign_new_missile = assign_all_missiles
