"""敌方目标对友方飞机编队的全局威胁评估。"""

from typing import Any, Dict, List, Sequence

from geometry import calculate_geometry
from models import Entity
from scoring import ScoreModel


class ThreatEvaluator:
    def __init__(self, score_model: ScoreModel):
        self.score_model = score_model

    def enemy_planes(
        self, enemy_planes: Sequence[Entity], friendly_planes: Sequence[Entity]
    ) -> List[float]:
        """每架敌机取其对全部友机的最大威胁。"""

        result = []
        for enemy_plane in enemy_planes:
            threats = [
                self.score_model.plane_threat(
                    enemy_plane, calculate_geometry(enemy_plane, friendly_plane)
                )
                for friendly_plane in friendly_planes
            ]
            result.append(max(threats, default=0.0))
        return result

    def enemy_missiles(
        self, enemy_missiles: Sequence[Entity], friendly_planes: Sequence[Entity]
    ) -> List[Dict[str, Any]]:
        """推断每枚敌弹最可能威胁的友机及预计到达时间。"""

        details = []
        for enemy_missile in enemy_missiles:
            candidates = []
            for plane_index, friendly_plane in enumerate(friendly_planes):
                geometry = calculate_geometry(enemy_missile, friendly_plane)
                candidates.append(
                    (self.score_model.missile_threat(geometry), geometry, plane_index)
                )

            if not candidates:
                details.append(self._empty_missile_threat())
                continue

            threat, geometry, plane_index = max(candidates, key=lambda item: item[0])
            details.append(
                {
                    "threat": threat,
                    "threatened_friend_index": plane_index,
                    "time_to_friend": geometry.t_cpa,
                    "distance_to_friend_at_cpa": geometry.d_cpa,
                }
            )
        return details

    @staticmethod
    def _empty_missile_threat() -> Dict[str, Any]:
        return {
            "threat": 0.0,
            "threatened_friend_index": None,
            "time_to_friend": None,
            "distance_to_friend_at_cpa": None,
        }
