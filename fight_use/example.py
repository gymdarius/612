"""运行示例：python -B example.py"""

from pprint import pprint

from expert_rules import ExpertRuleEngine


def main() -> None:
    engine = ExpertRuleEngine()
    result = engine.assign(
        friendly_missiles=[
            {"id": "friendly_missile_0", "state": [0, 8_000, 0, 450, 0, 0]},
            {"id": "friendly_missile_1", "state": [5_000, 8_100, 2_000, 420, 0, -10]},
        ],
        friendly_planes=[
            {"id": "friendly_plane_0", "state": [-20_000, 8_000, 0, 220, 0, 0]},
        ],
        enemy_planes=[
            {"id": "enemy_plane_0", "state": [80_000, 9_000, 5_000, -220, 0, 0]},
        ],
        enemy_missiles=[
            {"id": "enemy_missile_0", "state": [20_000, 8_000, 0, -500, 0, 0]},
        ],
    )
    pprint(result["decisions"], sort_dicts=False)
    pprint(result["strategy"], sort_dicts=False)


if __name__ == "__main__":
    main()
