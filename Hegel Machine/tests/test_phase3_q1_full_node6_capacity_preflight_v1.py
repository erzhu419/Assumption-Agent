from __future__ import annotations

from unittest.mock import patch

import pytest

from hegel_machine import phase3_q1_full_node6_capacity_preflight_v1 as full
from hegel_machine import phase3_q1_capacity_preflight_v1 as capacity


def _admission() -> full.FullNode6AdmissionV1:
    return full.FullNode6AdmissionV1(
        schema_version=full.ADMISSION_SCHEMA_VERSION,
        action_id=full.ACTION_ID,
        capsule_digest="sha256:" + "1" * 64,
        source_commit="2" * 40,
        predecessor_capsule_digest="sha256:" + "3" * 64,
        predecessor_terminal_receipt_digest="sha256:" + "4" * 64,
        predecessor_artifact_sha256="sha256:" + "5" * 64,
        projection_profile_root="sha256:" + "6" * 64,
        diagnostic_only=True,
        q1_state="NOT_RUN",
        q1_gate_count=0,
        q1_gate_mask=0,
        q1_output_slots=(None,) * 8,
    )


def _row(signature_id: int) -> full.FullNode6PartitionDiagnosticV1:
    return full.FullNode6PartitionDiagnosticV1(
        input_signature_id=signature_id,
        universe_row_count=480 if signature_id == 1 else 85,
        universe_root="sha256:" + ("a" if signature_id == 1 else "b") * 64,
        maximum_ast_depth=3,
        maximum_ast_node_count=6,
        full_v16_structural_limits_applied=True,
        terminal_status=full.TERMINAL_SATURATED,
        resource_guard_id=None,
        resource_guard_name=None,
        traversal_closed=True,
        frozen_leaf_count=810,
        raw_operator_application_count=1000 + signature_id,
        strict_admitted_application_count=1000 + signature_id,
        rewrite_collapse_count=0,
        behavior_class_count=40,
        signature_cohort_count=80,
        continuation_bank_point_count=100,
        visible_frontier_point_count=50,
        maximum_bank_points_per_class=4,
        maximum_frontier_points_per_class=2,
        peak_raw_operator_application_count=1000 + signature_id,
        peak_behavior_class_count=40,
        peak_visible_frontier_point_count=50,
        peak_visible_frontier_points_per_class=2,
        peak_continuation_bank_point_count=100,
        peak_continuation_bank_points_per_class=4,
        peak_work_queue_points=1000,
        peak_saturation_round_count=5,
        vector_cache_entry_count=30,
        vector_cache_hit_count=20,
        vector_cache_miss_count=30,
        depth_barriers=(),
    )


def _guard_row(signature_id: int) -> full.FullNode6PartitionDiagnosticV1:
    saturated = _row(signature_id)
    values = {
        name: getattr(saturated, name)
        for name in saturated.__dataclass_fields__
    }
    values.update(
        terminal_status=full.TERMINAL_GUARD_HIT,
        resource_guard_id=3,
        resource_guard_name="MAXIMUM_BEHAVIOR_CLASSES",
        traversal_closed=False,
    )
    return full.FullNode6PartitionDiagnosticV1(**values)


def test_admission_is_exact_and_closed() -> None:
    assert _admission().q1_output_slots == (None,) * 8
    with pytest.raises(full.FullNode6CapacityPreflightError) as caught:
        full.FullNode6AdmissionV1(
            **{
                **{
                    name: getattr(_admission(), name)
                    for name in _admission().__dataclass_fields__
                },
                "q1_state": "RUNNING",
            }
        )
    assert caught.value.code == "REJECT_FULL_NODE6_ADMISSION"


def test_bounded_source_qualification_matches_frozen_node3_golden() -> None:
    odd, sink = full.bounded_engine_self_test_v1(3)
    assert (
        odd.behavior_class_count,
        odd.visible_frontier_point_count,
        odd.continuation_bank_point_count,
        odd.raw_operator_application_count,
    ) == (40, 59, 110, 1048)
    assert (
        sink.behavior_class_count,
        sink.visible_frontier_point_count,
        sink.continuation_bank_point_count,
        sink.raw_operator_application_count,
    ) == (28, 84, 144, 1101)
    with pytest.raises(full.FullNode6CapacityPreflightError) as caught:
        full.bounded_engine_self_test_v1(6)
    assert caught.value.code == "REJECT_FULL_NODE6_SELF_TEST_SCOPE"


def test_same_run_snapshot_freeze_does_not_require_a_fresh_engine_run() -> None:
    limits = capacity.PreflightLimitsV1(maximum_ast_node_count=3)
    result, state = capacity._execute_q1_partition_capacity_preflight_state_v1(
        1,
        limits=limits,
    )
    with patch.object(
        capacity,
        "_execute_q1_partition_capacity_preflight_state_v1",
        side_effect=AssertionError("fresh quotient rerun"),
    ):
        snapshot = full._snapshot_from_result_and_material_v1(
            result,
            state.immutable_snapshot_material(),
        )
    assert snapshot.behavior_class_count == 40
    assert snapshot.visible_frontier_point_count == 59
    assert snapshot.continuation_bank_point_count == 110
    assert snapshot.raw_operator_application_count == 1048


def test_state_ownership_transfer_matches_frozen_node3_snapshot_exactly() -> None:
    limits = capacity.PreflightLimitsV1(maximum_ast_node_count=3)
    reference_result, reference_state = (
        capacity._execute_q1_partition_capacity_preflight_state_v1(1, limits=limits)
    )
    reference = full._snapshot_from_result_and_material_v1(
        reference_result,
        reference_state.immutable_snapshot_material(),
    )
    transfer_result, transfer_state = (
        capacity._execute_q1_partition_capacity_preflight_state_v1(1, limits=limits)
    )
    observed = full._snapshot_from_result_and_state_transfer_v1(
        transfer_result,
        transfer_state,
    )
    assert observed == reference
    assert transfer_state.classes == {}
    assert transfer_state.vector_cache == {}
    assert transfer_state.barriers == []


def test_full_runner_calls_each_signature_once_without_real_node6_execution() -> None:
    observed: list[int] = []

    def fake(signature_id: int, _limits: object):
        observed.append(signature_id)
        return _row(signature_id), None

    with patch.object(full, "_run_partition_once_v1", side_effect=fake):
        result, snapshots = full.run_full_node6_capacity_preflight_v1(_admission())
    assert observed == [1, 2]
    assert result.terminal_status == full.TERMINAL_SATURATED
    assert result.q1_state == "NOT_RUN"
    assert result.q1_gate_count == 0
    assert result.q1_gate_mask == 0
    assert result.q1_output_slots == (None,) * 8
    assert result.target_truth_accessed is False
    assert result.split_accessed is False
    assert result.role_evaluation_performed is False
    assert snapshots == (None, None)


def test_first_partition_registered_guard_is_a_terminal_prefix_not_failure() -> None:
    observed: list[int] = []

    def fake(signature_id: int, _limits: object):
        observed.append(signature_id)
        return _guard_row(signature_id), None

    with patch.object(full, "_run_partition_once_v1", side_effect=fake):
        result, snapshots = full.run_full_node6_capacity_preflight_v1(_admission())
    assert observed == [1]
    assert tuple(row.input_signature_id for row in result.partitions) == (1,)
    assert result.terminal_status == full.TERMINAL_GUARD_HIT
    assert snapshots == (None,)
