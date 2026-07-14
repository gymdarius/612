"""三维相对运动与 CPA（最近接近点）计算。"""

from math import acos, exp, sqrt
from typing import Sequence, Tuple

from models import EPS, Entity, Geometry


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def subtract(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def add_scaled(
    a: Sequence[float], b: Sequence[float], scale: float
) -> Tuple[float, float, float]:
    return a[0] + b[0] * scale, a[1] + b[1] * scale, a[2] + b[2] * scale


def norm(vector: Sequence[float]) -> float:
    return sqrt(dot(vector, vector))


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def safe_exp(value: float) -> float:
    """限制指数输入，防止极端仿真状态造成浮点溢出。"""

    return exp(max(-60.0, min(60.0, value)))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + safe_exp(-value))


def calculate_geometry(source: Entity, target: Entity) -> Geometry:
    """计算源实体沿当前速度运动时，相对于目标的几何关系。

    CPA 使用直线匀速近似。它适合作为专家启发式，不代表真实制导轨迹。
    ``y`` 分量单独用于高度差，三维距离仍同时包含 x/y/z。
    """

    relative_position = subtract(target.position, source.position)
    relative_velocity = subtract(target.velocity, source.velocity)
    distance = norm(relative_position)
    relative_speed_sq = dot(relative_velocity, relative_velocity)

    if distance <= EPS:
        direction_cosine = 1.0
        closing_speed = 0.0
    else:
        source_speed = norm(source.velocity)
        direction_cosine = (
            dot(source.velocity, relative_position) / (source_speed * distance + EPS)
            if source_speed > EPS
            else 0.0
        )
        # 正值表示距离正在缩短；远离时统一截断为 0。
        closing_speed = max(0.0, -dot(relative_position, relative_velocity) / distance)

    # t_cpa 小于 0 代表最近接近点已在过去，因此截断到当前时刻。
    t_cpa = max(
        0.0,
        -dot(relative_position, relative_velocity) / (relative_speed_sq + EPS),
    )
    d_cpa = norm(add_scaled(relative_position, relative_velocity, t_cpa))
    direction_cosine = clamp(direction_cosine, -1.0, 1.0)

    return Geometry(
        distance=distance,
        closing_speed=closing_speed,
        direction_cosine=direction_cosine,
        direction_angle_rad=acos(direction_cosine),
        t_cpa=t_cpa,
        d_cpa=d_cpa,
        height_difference=abs(target.y - source.y),
    )
