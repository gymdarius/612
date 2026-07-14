import unittest

from config_loader import ConfigLoader
from expert_rules import ExpertRuleEngine
from target_selection import TargetSelector


def config(**strategy):
    return ConfigLoader.from_dict(
        {
            "system": {"a_max": 4, "f_max": 3, "x_max": 3, "y_max": 3},
            "strategy": strategy,
        }
    )


class ExpertRuleEngineTests(unittest.TestCase):
    def test_all_missiles_receive_assignments(self):
        engine = ExpertRuleEngine(config=config())
        result = engine.assign(
            [[0, 5_000, 0, 400, 0, 0], [1_000, 5_000, 0, 420, 0, 0]],
            [],
            [[40_000, 5_000, 0, -200, 0, 0]],
            [],
        )
        self.assertEqual(result["assignment"][:2], [0, 0])
        self.assertEqual(len(result["decisions"]), 2)
        self.assertNotIn("target_occupancy", result)

    def test_fixed_target_numbering_and_masks(self):
        result = ExpertRuleEngine(config=config()).assign(
            [[0, 8_000, 0, 500, 0, 0]],
            [[0, 8_000, 0, 200, 0, 0]],
            [[60_000, 8_000, 0, -200, 0, 0]],
            [[30_000, 8_000, 0, -500, 0, 0]],
        )
        self.assertEqual(result["target_mask"], [1, 0, 0, 1, 0, 0])
        self.assertIn(result["assignment"][0], (0, 3))
        self.assertEqual(len(result["raw_expert_scores"]), 4)
        self.assertEqual(len(result["raw_expert_scores"][0]), 6)

    def test_only_enemy_plane_means_attack(self):
        result = ExpertRuleEngine(config=config()).assign(
            [[0, 5_000, 0, 400, 0, 0]],
            [],
            [[50_000, 5_000, 0, -200, 0, 0]],
            [],
        )
        self.assertEqual(result["decisions"][0]["action"], "ATTACK")

    def test_only_enemy_missile_means_defend(self):
        result = ExpertRuleEngine(config=config()).assign(
            [[0, 5_000, 0, 500, 0, 0]],
            [[-20_000, 5_000, 0, 200, 0, 0]],
            [],
            [[20_000, 5_000, 0, -500, 0, 0]],
        )
        self.assertEqual(result["assignment"][0], 3)
        self.assertEqual(result["decisions"][0]["action"], "DEFEND")

    def test_unreachable_target_forces_fallback(self):
        result = ExpertRuleEngine(config=config()).assign(
            [[0, 5_000, 0, 400, 0, 0]],
            [],
            [[-100_000, 5_000, 0, -200, 0, 0]],
            [],
        )
        self.assertEqual(result["assignment"][0], 0)
        self.assertTrue(result["decisions"][0]["fallback"])

    def test_strategy_bias_changes_attack_defense_preference(self):
        scores = {
            "raw_scores": [[0.5, 0.0, 0.0, 0.5, 0.0, 0.0]],
            "feasible": [[True, False, False, True, False, False]],
            "emergency": [[False] * 6],
            "reasons": [[{} for _ in range(6)]],
        }
        attack = TargetSelector(config(strategy_bias=-1.0)).select(
            scores, [1, 0, 0, 1, 0, 0], 3
        )
        defense = TargetSelector(config(strategy_bias=1.0)).select(
            scores, [1, 0, 0, 1, 0, 0], 3
        )
        self.assertEqual(attack["assignment"], [0])
        self.assertEqual(defense["assignment"], [3])

    def test_emergency_can_override_attack_bias(self):
        scores = {
            "raw_scores": [[0.9, 0.0, 0.0, 0.2, 0.0, 0.0]],
            "feasible": [[True, False, False, True, False, False]],
            "emergency": [[False, False, False, True, False, False]],
            "reasons": [[{} for _ in range(6)]],
        }
        selector = TargetSelector(
            config(
                strategy_bias=-1.0,
                emergency_defense_enabled=True,
                emergency_overrides_strategy=True,
            )
        )
        result = selector.select(scores, [1, 0, 0, 1, 0, 0], 3)
        self.assertEqual(result["assignment"], [3])
        self.assertTrue(result["decisions"][0]["emergency_override"])

    def test_invalid_bias_is_rejected(self):
        with self.assertRaises(ValueError):
            config(strategy_bias=1.5)

    def test_requires_real_target(self):
        with self.assertRaises(ValueError):
            ExpertRuleEngine(config=config()).assign(
                [[0, 0, 0, 1, 0, 0]], [], [], []
            )


if __name__ == "__main__":
    unittest.main()
