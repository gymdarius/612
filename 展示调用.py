from __future__ import annotations

"""Version1 rule-based algorithm calling example.

Run from the Version1 directory:

    python example_oop_pipeline.py

This example does not define new business classes. It directly calls the
existing Version1 classes:

    SituationAssessment
    ThreatAssessment
    TargetAllocation
"""

from typing import Sequence

import numpy as np

from dto import AllocationInputDTO, EntityState, SAMInputDTO, ThreatInputDTO
from SituationAssessment import SituationAssessment
from TargetAllocation import TargetAllocation
from ThreatAssessment import ThreatAssessment


def build_entity(entity_id: str, position: Sequence[float], velocity: Sequence[float]) -> EntityState:
    """Create one missile / friendly aircraft / target entity.

    Input format:
        entity_id:
            Entity id, for example "M0", "A0", "T0".

        position:
            3D position in meters, format [x, y, z].

        velocity:
            3D velocity in meters/second, format [vx, vy, vz].

    Output format:
        EntityState:
            Existing Version1 DTO class.
            It stores id, position, velocity, speed and euler_angles.
            speed is automatically computed from velocity if not provided.
    """
    return EntityState(
        id=entity_id,
        position=np.asarray(position, dtype=np.float64),
        velocity=np.asarray(velocity, dtype=np.float64),
        euler_angles=np.zeros(3, dtype=np.float64),
    )


def main() -> None:
    # =========================
    # 0. Initial input
    # =========================
    # missiles:
    #   tuple[EntityState, ...], length M.
    #   In current Version1 dataset configuration, M is usually 1..4.
    #
    # aircrafts:
    #   tuple[EntityState, ...], length A.
    #   In current Version1 dataset configuration, A is usually 1..4.
    #
    # targets:
    #   tuple[EntityState, ...], length T.
    #   In current Version1 dataset configuration, T is usually 1..8.
    #
    # EntityState position format:
    #   [x, y, z], unit: meter.
    #
    # EntityState velocity format:
    #   [vx, vy, vz], unit: meter/second.
    missiles = (
        build_entity("M0", [5000.0, 10000.0, 5000.0], [600.0, -500.0, 0.0]),
        build_entity("M1", [5000.0, -10000.0, 5000.0], [600.0, 500.0, 0.0]),
    )
    aircrafts = (
        build_entity("A0", [0.0, 10000.0, 5000.0], [250.0, 0.0, 0.0]),
        build_entity("A1", [0.0, -10000.0, 5000.0], [250.0, 0.0, 0.0]),
    )
    targets = (
        build_entity("T0", [30000.0, 15000.0, 5000.0], [-200.0, -50.0, 0.0]),
        build_entity("T1", [30000.0, -15000.0, 5000.0], [-200.0, 50.0, 0.0]),
        build_entity("T2", [70000.0, 0.0, 6500.0], [-250.0, 0.0, -30.0]),
    )

    print("Initial input")
    print(f"missiles: {[entity.id for entity in missiles]}")
    print(f"friendly aircrafts: {[entity.id for entity in aircrafts]}")
    print(f"enemy targets: {[entity.id for entity in targets]}")
    print()

    # =========================
    # 1. Situation assessment
    # =========================
    # Existing class:
    #   SituationAssessment
    #
    # Input DTO:
    #   SAMInputDTO(
    #       missiles=tuple[EntityState, ...],  # M missiles
    #       targets=tuple[EntityState, ...],   # T enemy targets
    #   )
    #
    # Output DTO:
    #   situation_output.situation_matrix:
    #       numpy.ndarray, shape [M, T], dtype float64.
    #       situation_matrix[i, j] is the situation score of missile i
    #       against target j. Value range is [0, 1].
    #
    #   situation_output.pair_metrics:
    #       Detailed metrics for every missile-target pair, useful for debugging.
    situation_assessment = SituationAssessment()
    situation_input = SAMInputDTO(missiles=missiles, targets=targets)
    situation_output = situation_assessment.compute(situation_input)

    print("1. Situation assessment")
    print("situation_matrix [M, T]:")
    print(np.array2string(situation_output.situation_matrix, precision=4, suppress_small=True))
    print()

    # =========================
    # 2. Threat assessment
    # =========================
    # Existing class:
    #   ThreatAssessment
    #
    # Input DTO:
    #   ThreatInputDTO(
    #       our_aircrafts=tuple[EntityState, ...],  # A friendly aircraft
    #       targets=tuple[EntityState, ...],        # T enemy targets
    #   )
    #
    # Output DTO:
    #   threat_output.threat_matrix:
    #       numpy.ndarray, shape [A, T], dtype float64.
    #       threat_matrix[i, j] is the threat score of target j
    #       to friendly aircraft i. Value range is [0, 1].
    #
    #   threat_output.target_threat_weights:
    #       numpy.ndarray, shape [T], dtype float64.
    #       This is aggregated from threat_matrix by target and normalized.
    #       Valid target weights sum to 1.
    #
    #   threat_output.pair_metrics:
    #       Detailed metrics for every friendly-aircraft-target pair.
    threat_assessment = ThreatAssessment()
    threat_input = ThreatInputDTO(our_aircrafts=aircrafts, targets=targets)
    threat_output = threat_assessment.compute(threat_input)

    print("2. Threat assessment")
    print("threat_matrix [A, T]:")
    print(np.array2string(threat_output.threat_matrix, precision=4, suppress_small=True))
    print("target_threat_weights [T]:")
    print(np.array2string(threat_output.target_threat_weights, precision=4, suppress_small=True))
    print()

    # =========================
    # 3. Target allocation
    # =========================
    # Existing class:
    #   TargetAllocation
    #
    # Input DTO:
    #   AllocationInputDTO(
    #       missiles=tuple[EntityState, ...],
    #       targets=tuple[EntityState, ...],
    #       situation_output=SAMOutputDTO,
    #       threat_output=ThreatOutputDTO,
    #   )
    #
    # Output DTO:
    #   allocation_output.assignments:
    #       tuple[AllocationEntryDTO, ...].
    #       One assignment per missile. target_index is a 0-based target index.
    #
    #   allocation_output.value_matrix:
    #       numpy.ndarray, shape [M, T].
    #       Raw allocation value matrix before softmax.
    #
    #   allocation_output.normalized_matrix:
    #       numpy.ndarray, shape [M, T].
    #       Row-wise softmax result. Each missile row sums to 1.
    #
    #   allocation_output.target_threat_weights:
    #       numpy.ndarray, shape [T].
    #       The target-level threat weights used during allocation.
    target_allocation = TargetAllocation()
    allocation_input = AllocationInputDTO(
        missiles=missiles,
        targets=targets,
        situation_output=situation_output,
        threat_output=threat_output,
    )
    allocation_output = target_allocation.compute(allocation_input)

    print("3. Target allocation")
    print("value_matrix [M, T]:")
    print(np.array2string(allocation_output.value_matrix, precision=4, suppress_small=True))
    print("normalized_matrix [M, T]:")
    print(np.array2string(allocation_output.normalized_matrix, precision=4, suppress_small=True))
    print("assignments:")
    for item in allocation_output.assignments:
        print(
            f"  missile={item.missile_id} "
            f"-> target={item.target_id} "
            f"(missile_index={item.missile_index}, target_index={item.target_index}, "
            f"score={item.score:.4f}, probability={item.probability:.4f})"
        )


if __name__ == "__main__":
    main()
