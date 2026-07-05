from __future__ import annotations

"""Version1 迁移回归测试。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.

这个脚本用于离线环境快速确认：
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
1. 广播实现与原始双层循环公式一致。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
2. DTO 新接口和旧接口兼容。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
3. 空输入、shape 校验等边界逻辑没有被破坏。
# English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
"""

import numpy as np

from SituationAssessment import SituationAssessment
from TargetAllocation import TargetAllocation
from ThreatAssessment import ThreatAssessment
from dto import (
    AllocationInputDTO,
    SAMInputDTO,
    SAMOutputDTO,
    ThreatInputDTO,
    ThreatOutputDTO,
    coerce_entities,
)


def _reference_situation(assessment: SituationAssessment, missiles, targets) -> np.ndarray:
    """原始双层循环态势公式，仅用于和广播版做数值对比。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    missile_count = len(missiles)
    target_count = len(targets)
    matrix = np.zeros((missile_count, target_count), dtype=np.float64)

    for i in range(missile_count):
        m_pos = missiles[i]["position"]
        m_vel = missiles[i]["velocity"]
        m_speed = float(np.linalg.norm(m_vel))
        m_dir = m_vel / m_speed if m_speed > 1e-8 else np.zeros(3, dtype=np.float64)

        for j in range(target_count):
            t_pos = targets[j]["position"]
            t_vel = targets[j]["velocity"]

            rel_pos = t_pos - m_pos
            distance = float(np.linalg.norm(rel_pos))
            if distance < 1e-8:
                matrix[i, j] = 1.0
                continue

            los = rel_pos / distance
            closing_speed = float(-np.dot(t_vel - m_vel, los))

            x = closing_speed / assessment.V_ref
            closing_score = float(1.0 / (1.0 + np.exp(-3.0 * (x - 0.3))))
            distance_score = float(np.exp(-distance / assessment.D_ref))

            boresight_cos = float(np.clip(np.dot(m_dir, los), -1.0, 1.0))
            if boresight_cos >= assessment.fov_cos_threshold:
                boresight_score = 0.95 + 0.05 * (boresight_cos - assessment.fov_cos_threshold) / max(
                    1.0 - assessment.fov_cos_threshold,
                    1e-8,
                )
            else:
                boresight_score = 0.2 * np.clip(
                    (boresight_cos + 1.0) / (assessment.fov_cos_threshold + 1.0),
                    0.0,
                    1.0,
                )

            energy_score = float(
                np.clip(
                    m_speed / assessment.V_ref * np.exp(-distance / (2.0 * assessment.D_ref)),
                    0.0,
                    1.0,
                )
            )

            matrix[i, j] = float(
                np.clip(
                    assessment.w_closing * closing_score
                    + assessment.w_distance * distance_score
                    + assessment.w_boresight * boresight_score
                    + assessment.w_energy * energy_score,
                    0.0,
                    1.0,
                )
            )
    return matrix


def _reference_threat(assessment: ThreatAssessment, aircrafts, targets) -> np.ndarray:
    """原始双层循环威胁公式，仅用于和广播版做数值对比。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    aircraft_count = len(aircrafts)
    target_count = len(targets)
    matrix = np.zeros((aircraft_count, target_count), dtype=np.float64)

    for i in range(aircraft_count):
        a_pos = aircrafts[i]["position"]
        a_vel = aircrafts[i]["velocity"]
        a_speed = float(np.linalg.norm(a_vel))
        a_y = float(a_pos[1])

        for j in range(target_count):
            t_pos = targets[j]["position"]
            t_vel = targets[j]["velocity"]
            t_speed = float(np.linalg.norm(t_vel))
            t_dir = t_vel / t_speed if t_speed > 1e-8 else np.zeros(3, dtype=np.float64)
            t_y = float(t_pos[1])

            rel_pos = a_pos - t_pos
            distance = float(np.linalg.norm(rel_pos))
            if distance < 1e-8:
                matrix[i, j] = 1.0
                continue

            los = rel_pos / distance
            los_cos = float(np.clip(np.dot(t_dir, los), -1.0, 1.0))
            if los_cos >= assessment.los_cos_lethal:
                t = (los_cos - assessment.los_cos_lethal) / max(1.0 - assessment.los_cos_lethal, 1e-8)
                los_score = 0.8 + 0.2 * t
            elif los_cos >= assessment.los_cos_neutral:
                t = (los_cos - assessment.los_cos_neutral) / max(
                    assessment.los_cos_lethal - assessment.los_cos_neutral,
                    1e-8,
                )
                los_score = 0.3 + 0.5 * t
            else:
                t = (los_cos + 1.0) / max(assessment.los_cos_neutral + 1.0, 1e-8)
                los_score = 0.1 * t

            if distance <= assessment.D_lethal:
                distance_score = 0.8 + 0.2 * (1.0 - distance / max(assessment.D_lethal, 1e-8))
            else:
                distance_score = 0.8 * np.exp(-(distance - assessment.D_lethal) / assessment.D_ref)

            speed_ratio = t_speed / max(a_speed, 1e-8)
            if speed_ratio <= 0.6:
                speed_value = 0.1
            elif speed_ratio >= 1.5:
                speed_value = 1.0
            else:
                speed_value = speed_ratio - 0.5
            speed_score = float(np.clip(speed_value * np.exp(-distance / (2.0 * assessment.D_ref)), 0.0, 1.0))

            height_delta = t_y - a_y
            if height_delta > 2000.0:
                height_score = 0.9
            elif height_delta > 0.0:
                height_score = 0.5 + 0.4 * (height_delta / 2000.0)
            elif height_delta > -2000.0:
                height_score = 0.5 * (1.0 + height_delta / 2000.0)
            else:
                height_score = 0.1

            matrix[i, j] = float(
                np.clip(
                    assessment.w_los * los_score
                    + assessment.w_distance * distance_score
                    + assessment.w_speed * speed_score
                    + assessment.w_height * height_score,
                    0.0,
                    1.0,
                )
            )
    return matrix


def _sample_data():
    missiles = [
        {
            "id": "m0",
            "position": np.array([5000.0, 10000.0, 5000.0], dtype=np.float64),
            "velocity": np.array([600.0, -500.0, 0.0], dtype=np.float64),
        },
        {
            "id": "m1",
            "position": np.array([5000.0, -10000.0, 5000.0], dtype=np.float64),
            "velocity": np.array([600.0, 500.0, 0.0], dtype=np.float64),
        },
    ]
    targets = [
        {
            "id": "t0",
            "position": np.array([30000.0, 15000.0, 5000.0], dtype=np.float64),
            "velocity": np.array([-200.0, -50.0, 0.0], dtype=np.float64),
        },
        {
            "id": "t1",
            "position": np.array([30000.0, -15000.0, 5000.0], dtype=np.float64),
            "velocity": np.array([-200.0, 50.0, 0.0], dtype=np.float64),
        },
        {
            "id": "t2",
            "position": np.array([85000.0, 500.0, 7000.0], dtype=np.float64),
            "velocity": np.array([-250.0, 10.0, -20.0], dtype=np.float64),
        },
    ]
    aircrafts = [
        {
            "id": "a0",
            "position": np.array([0.0, 10000.0, 5000.0], dtype=np.float64),
            "velocity": np.array([250.0, 0.0, 0.0], dtype=np.float64),
        },
        {
            "id": "a1",
            "position": np.array([0.0, -10000.0, 5000.0], dtype=np.float64),
            "velocity": np.array([250.0, 0.0, 0.0], dtype=np.float64),
        },
    ]
    return missiles, targets, aircrafts


def test_vectorized_formula_matches_reference() -> None:
    """验证广播实现没有改变 Version1 原业务公式。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    missiles, targets, aircrafts = _sample_data()

    situation = SituationAssessment()
    np.testing.assert_allclose(
        situation.evaluate(missiles, targets),
        _reference_situation(situation, missiles, targets),
        rtol=1e-12,
        atol=1e-12,
    )

    threat = ThreatAssessment()
    np.testing.assert_allclose(
        threat.evaluate(aircrafts, targets),
        _reference_threat(threat, aircrafts, targets),
        rtol=1e-12,
        atol=1e-12,
    )


def test_dto_pipeline_and_legacy_allocation_match() -> None:
    """验证 DTO 流程和旧矩阵接口的目标分配结果一致。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    missiles, targets, aircrafts = _sample_data()
    missile_entities = coerce_entities(missiles, "missile_")
    target_entities = coerce_entities(targets, "target_")
    aircraft_entities = coerce_entities(aircrafts, "aircraft_")

    situation_output = SituationAssessment().compute(SAMInputDTO(missiles=missile_entities, targets=target_entities))
    threat_output = ThreatAssessment().compute(ThreatInputDTO(our_aircrafts=aircraft_entities, targets=target_entities))
    allocation = TargetAllocation()
    dto_output = allocation.compute(
        AllocationInputDTO(
            missiles=missile_entities,
            targets=target_entities,
            situation_output=situation_output,
            threat_output=threat_output,
        )
    )
    legacy_matrix, legacy_assignments = allocation.allocate(
        situation_output.situation_matrix,
        threat_output.threat_matrix,
    )

    np.testing.assert_allclose(dto_output.normalized_matrix, legacy_matrix, rtol=1e-12, atol=1e-12)
    assert [(item.missile_index, item.target_index) for item in dto_output.assignments] == legacy_assignments
    assert len(situation_output.pair_metrics) == len(missile_entities) * len(target_entities)
    assert len(threat_output.pair_metrics) == len(aircraft_entities) * len(target_entities)
    assert dto_output.to_dict()["assignments"][0]["missile_id"] == "m0"


def test_empty_inputs_and_shape_validation() -> None:
    """验证空输入和 shape 错误的行为稳定。"""
    # English note: This docstring explains the purpose, inputs, outputs, or constraints of this code block.
    missiles, targets, _ = _sample_data()
    missile_entities = coerce_entities(missiles, "missile_")
    target_entities = coerce_entities(targets, "target_")

    empty_situation = SituationAssessment().compute(SAMInputDTO(missiles=tuple(), targets=target_entities))
    assert empty_situation.situation_matrix.shape == (0, len(target_entities))

    empty_threat = ThreatAssessment().compute(ThreatInputDTO(our_aircrafts=tuple(), targets=target_entities))
    assert empty_threat.threat_matrix.shape == (0, len(target_entities))
    np.testing.assert_allclose(empty_threat.target_threat_weights, np.full((len(target_entities),), 1.0 / 3.0))

    no_target_output = TargetAllocation().compute(
        AllocationInputDTO(
            missiles=missile_entities,
            targets=tuple(),
            situation_output=SAMOutputDTO(situation_matrix=np.zeros((len(missile_entities), 0)), pair_metrics=tuple()),
            threat_output=ThreatOutputDTO(
                threat_matrix=np.zeros((0, 0)),
                target_threat_weights=np.zeros((0,)),
                pair_metrics=tuple(),
            ),
        )
    )
    assert [item.target_id for item in no_target_output.assignments] == ["UNASSIGNED", "UNASSIGNED"]

    try:
        TargetAllocation().compute(
            AllocationInputDTO(
                missiles=missile_entities,
                targets=target_entities,
                situation_output=SAMOutputDTO(situation_matrix=np.zeros((1, 1)), pair_metrics=tuple()),
                threat_output=ThreatOutputDTO(
                    threat_matrix=np.zeros((2, len(target_entities))),
                    target_threat_weights=np.full((len(target_entities),), 1.0 / len(target_entities)),
                    pair_metrics=tuple(),
                ),
            )
        )
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch should raise ValueError")


def run_all() -> None:
    test_vectorized_formula_matches_reference()
    test_dto_pipeline_and_legacy_allocation_match()
    test_empty_inputs_and_shape_validation()


if __name__ == "__main__":
    run_all()
    print("Version1 migration regression tests passed.")
