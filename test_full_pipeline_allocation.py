import numpy as np

from dto import AllocationInputDTO, EntityState, SAMInputDTO, ThreatInputDTO
from SituationAssessment import SituationAssessment
from TargetAllocation import TargetAllocation
from ThreatAssessment import ThreatAssessment


def test_full_pipeline_allocation():
    """验证同一物理场景下，分配权重能够改变导弹的目标偏好。"""
    # 只修改这一处即可测试不同偏好；攻击优势权重自动保持为 1 - threat_weight。
    threat_weight = 0.45
    advantage_weight = 1.0 - threat_weight

    friendly_aircraft = EntityState(
        id="friendly_aircraft",
        position=np.array([0.0, 10000.0, 0.0]),
        velocity=np.array([250.0, 0.0, 0.0]),
    )
    friendly_missile = EntityState(
        id="friendly_missile",
        position=np.array([0.0, 10000.0, 0.0]),
        velocity=np.array([700.0, 0.0, 0.0]),
    )

    # A 在导弹后方并朝友机飞行：对友机威胁较大，但导弹攻击态势较差。
    enemy_a = EntityState(
        id="enemy_A_high_threat",
        position=np.array([-20000.0, 10000.0, 0.0]),
        velocity=np.array([400.0, 0.0, 0.0]),
    )
    # B 在导弹正前方并远离友机：对友机威胁较小，但导弹攻击态势较好。
    enemy_b = EntityState(
        id="enemy_B_high_advantage",
        position=np.array([30000.0, 10000.0, 0.0]),
        velocity=np.array([300.0, 0.0, 0.0]),
    )
    targets = (enemy_a, enemy_b)

    situation_output = SituationAssessment().compute(
        SAMInputDTO(missiles=(friendly_missile,), targets=targets)
    )
    threat_output = ThreatAssessment().compute(
        ThreatInputDTO(our_aircrafts=(friendly_aircraft,), targets=targets)
    )

    allocation = TargetAllocation(
        {
            "weights": {
                "advantage": advantage_weight,
                "threat": threat_weight,
            }
        }
    )
    allocation_output = allocation.compute(
        AllocationInputDTO(
            missiles=(friendly_missile,),
            targets=targets,
            situation_output=situation_output,
            threat_output=threat_output,
        )
    )

    situation_a = situation_output.situation_matrix[0, 0]
    situation_b = situation_output.situation_matrix[0, 1]
    threat_a = threat_output.target_threat_weights[0]
    threat_b = threat_output.target_threat_weights[1]

    # 先验证物理场景确实形成了预期冲突，再观察权重决定的最终选择。
    assert situation_b > situation_a, "场景无效：导弹攻击 B 的态势优势应高于 A。"
    assert threat_a > threat_b, "场景无效：A 对友方飞机的威胁应高于 B。"
    assert len(allocation_output.assignments) == 1

    selected = allocation_output.assignments[0]
    print("\n=== Full-pipeline target allocation test ===")
    print(f"advantage_weight: {advantage_weight:.2f}")
    print(f"threat_weight:    {threat_weight:.2f}")
    print("situation_matrix:")
    print(np.array2string(situation_output.situation_matrix, precision=4, suppress_small=True))
    print("threat_matrix:")
    print(np.array2string(threat_output.threat_matrix, precision=4, suppress_small=True))
    print("target_threat_weights:")
    print(np.array2string(threat_output.target_threat_weights, precision=4, suppress_small=True))
    print("value_matrix:")
    print(np.array2string(allocation_output.value_matrix, precision=4, suppress_small=True))
    print("normalized_matrix:")
    print(np.array2string(allocation_output.normalized_matrix, precision=4, suppress_small=True))
    print(f"selected target: {selected.target_id} (index={selected.target_index})")


if __name__ == "__main__":
    test_full_pipeline_allocation()