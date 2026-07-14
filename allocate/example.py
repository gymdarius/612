"""直接运行：python example.py"""

from pprint import pprint

from expert_rule import create_default_expert


def main() -> None:
    # 三个输入均是二维数组，每一行为 [x, y, z, vx, vy, vz]。
    friendly_planes = [
        [0, 0, 8_000, 250, 0, 0],
        [0, 30_000, 9_000, 230, 0, 0],
    ]
    friendly_missiles = [
        [5_000, 0, 8_000, 900, 0, 0],
        [5_000, 30_000, 9_000, 700, -350, 0],
    ]
    enemy_planes = [
        [70_000, 0, 8_000, -250, 0, 0],
        [45_000, 45_000, 9_000, -180, -180, 0],
    ]

    result = create_default_expert().assign(
        friendly_planes, friendly_missiles, enemy_planes
    )
    print("每枚导弹分配的敌机索引:", result.assignments)
    print("敌机威胁分数:")
    pprint(result.enemy_threat_scores)
    print("攻击优势矩阵（行=导弹，列=敌机）:")
    pprint(result.attack_score_matrix)
    print("综合分数矩阵（行=导弹，列=敌机）:")
    pprint(result.composite_score_matrix)


if __name__ == "__main__":
    main()
