"""目标分配专家规则单元测试。"""

import unittest

from config import ComponentWeights, CompositeWeights
from expert_rule import create_default_expert


class TargetAssignmentExpertTests(unittest.TestCase):
    def test_accepts_three_two_dimensional_arrays(self) -> None:
        result = create_default_expert().assign(
            [[0, 0, 0, 100, 0, 0]],
            [[0, 0, 0, 500, 0, 0]],
            [[50_000, 0, 0, -100, 0, 0]],
        )
        self.assertEqual(result.assignments, [0])

    def test_same_target_has_different_scores_for_missiles(self) -> None:
        result = create_default_expert().assign(
            [[0, 0, 0, 100, 0, 0]],
            [[0, 0, 0, 500, 0, 0], [0, 20_000, 0, 0, 500, 0]],
            [[50_000, 0, 0, -100, 0, 0]],
        )
        self.assertNotEqual(
            result.composite_score_matrix[0][0],
            result.composite_score_matrix[1][0],
        )

    def test_different_missiles_can_select_different_targets(self) -> None:
        result = create_default_expert().assign(
            [[0, 0, 0, 100, 0, 0]],
            [[0, 0, 0, 600, 0, 0], [0, 40_000, 0, 600, 0, 0]],
            [[40_000, 0, 0, -100, 0, 0], [40_000, 40_000, 0, -100, 0, 0]],
        )
        self.assertEqual(result.assignments, [0, 1])

    def test_empty_missile_array_returns_empty_assignment(self) -> None:
        result = create_default_expert().assign(
            [[0, 0, 0, 100, 0, 0]], [], [[50_000, 0, 0, -100, 0, 0]]
        )
        self.assertEqual(result.assignments, [])

    def test_invalid_state_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            create_default_expert().assign(
                [[0, 0, 0, 100, 0]],
                [[0, 0, 0, 500, 0, 0]],
                [[50_000, 0, 0, -100, 0, 0]],
            )

    def test_invalid_weights_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ComponentWeights(0.5, 0.5, 0.5).validate("test")
        with self.assertRaises(ValueError):
            CompositeWeights(0.8, 0.8).validate()


if __name__ == "__main__":
    unittest.main()
