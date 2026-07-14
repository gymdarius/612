"""三维相对几何量及 0～1 归一化函数。"""

from math import sqrt
from typing import Sequence, Tuple

from models import RelativeGeometry

State = Tuple[float, float, float, float, float, float]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def norm(vector: Sequence[float]) -> float:
    return sqrt(dot(vector, vector))


class RelativeGeometryCalculator:
    """统一计算威胁侧和攻击侧需要的相对运动信息。"""

    def __init__(self, epsilon: float = 1e-9) -> None:
        self.epsilon = epsilon

    def calculate(self, source: State, target: State) -> RelativeGeometry:
        """计算源实体相对于目标实体的距离、接近速度和指向余弦。"""

        relative_position = tuple(target[i] - source[i] for i in range(3))
        relative_velocity = tuple(target[i] - source[i] for i in range(3, 6))
        source_velocity = source[3:6]
        distance = norm(relative_position)
        source_speed = norm(source_velocity)

        if distance <= self.epsilon:
            return RelativeGeometry(0.0, 0.0, 1.0)

        closing_speed = max(
            0.0, -dot(relative_position, relative_velocity) / distance
        )
        direction_cosine = (
            -1.0
            if source_speed <= self.epsilon
            else dot(source_velocity, relative_position) / (source_speed * distance)
        )
        return RelativeGeometry(
            distance,
            closing_speed,
            clamp(direction_cosine, -1.0, 1.0),
        )


def distance_score(distance: float, scale: float) -> float:
    """距离越近分数越高；距离达到尺度后分数为零。"""

    return clamp(1.0 - distance / scale)


def speed_score(closing_speed: float, scale: float) -> float:
    """接近速度越大分数越高。"""

    return clamp(closing_speed / scale)


def angle_score(direction_cosine: float) -> float:
    """把方向余弦从 [-1, 1] 映射到 [0, 1]。"""

    return clamp((direction_cosine + 1.0) / 2.0)
