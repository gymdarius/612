"""专家规则使用的内部结果对象。"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class RelativeGeometry:
    """源实体指向目标实体的三项相对几何量。"""

    distance: float
    closing_speed: float
    direction_cosine: float


@dataclass(frozen=True)
class ScoreBreakdown:
    """一个态势总分及其三个可解释分量。"""

    distance: float
    speed: float
    angle: float
    total: float


@dataclass(frozen=True)
class ThreatResult:
    """一架敌机对我方编队的威胁结果。"""

    enemy_index: int
    score: float
    most_threatened_friendly_index: int
    breakdown: ScoreBreakdown


@dataclass(frozen=True)
class MissileAssignment:
    """一枚导弹的最终目标及分数解释。"""

    missile_index: int
    target_index: int
    attack_score: float
    threat_score: float
    composite_score: float
    score_margin: float
    attack_breakdown: ScoreBreakdown
    low_confidence: bool


@dataclass(frozen=True)
class AssignmentResult:
    """完整评分矩阵和每枚导弹的分配结果。"""

    assignments: List[int]
    enemy_threat_scores: List[float]
    pair_threat_score_matrix: List[List[float]]
    attack_score_matrix: List[List[float]]
    composite_score_matrix: List[List[float]]
    threat_details: List[ThreatResult]
    missile_details: List[MissileAssignment]

    def to_dict(self) -> Dict[str, Any]:
        """转换成便于 JSON 序列化和生成训练数据的字典。"""

        return asdict(self)
