from __future__ import annotations

import ast
from hashlib import sha256
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

preflight = importlib.import_module(
    "hegel_machine.phase3_q1_capacity_preflight_v1"
)


@pytest.fixture(scope="module")
def node3_result():
    limits = preflight.PreflightLimitsV1(maximum_ast_node_count=3)
    return preflight.run_q1_capacity_preflight_v1(limits=limits)


@pytest.fixture(scope="module")
def node4_result():
    limits = preflight.PreflightLimitsV1(maximum_ast_node_count=4)
    return preflight.run_q1_capacity_preflight_v1(limits=limits)


def test_node3_full_leaf_bank_goldens(node3_result) -> None:
    assert preflight.PREFLIGHT_ID == "hegel-phase3a-q1-capacity-preflight-v1"
    assert (
        preflight.PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY
        == "PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY"
    )
    assert (
        node3_result.terminal_status
        == preflight.LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED
    )
    odd, sink = node3_result.partitions
    assert (odd.input_signature_id, sink.input_signature_id) == (1, 2)
    assert (odd.universe_row_count, sink.universe_row_count) == (480, 85)
    assert (odd.frozen_leaf_count, sink.frozen_leaf_count) == (810, 810)
    assert odd.full_v16_structural_limits_applied is False
    assert sink.full_v16_structural_limits_applied is False

    assert (
        odd.raw_operator_application_count,
        odd.behavior_class_count,
        odd.visible_frontier_point_count,
        odd.continuation_bank_point_count,
    ) == (1048, 40, 59, 110)
    assert (
        sink.raw_operator_application_count,
        sink.behavior_class_count,
        sink.visible_frontier_point_count,
        sink.continuation_bank_point_count,
    ) == (1101, 28, 84, 144)


def test_depth_barriers_are_closed_from_prior_complete_bank(node3_result) -> None:
    odd, sink = node3_result.partitions
    assert tuple(row.depth for row in odd.depth_barriers) == (0, 1, 2, 3, 4)
    assert tuple(row.barrier_kind for row in odd.depth_barriers) == (
        "LEAF_SEED",
        "CONSTRUCTION_DEPTH",
        "CONSTRUCTION_DEPTH",
        "CONSTRUCTION_DEPTH",
        "STRUCTURAL_BOUNDARY",
    )
    assert tuple(row.eligible_raw_application_count for row in odd.depth_barriers) == (
        810,
        202,
        36,
        0,
        0,
    )
    assert tuple(row.eligible_raw_application_count for row in sink.depth_barriers) == (
        810,
        249,
        42,
        0,
        0,
    )
    assert sum(row.eligible_raw_application_count for row in odd.depth_barriers) == 1048
    assert sum(row.eligible_raw_application_count for row in sink.depth_barriers) == 1101
    assert odd.depth_barriers[-1].continuation_bank_point_count_after_barrier == 110
    assert sink.depth_barriers[-1].continuation_bank_point_count_after_barrier == 144


def test_latent_bank_not_visible_frontier_drives_expansion(node3_result) -> None:
    odd, sink = node3_result.partitions
    assert odd.continuation_bank_point_count > odd.visible_frontier_point_count
    assert sink.continuation_bank_point_count > sink.visible_frontier_point_count
    assert odd.vector_cache_entry_count == odd.vector_cache_miss_count
    assert sink.vector_cache_entry_count == sink.vector_cache_miss_count
    assert odd.vector_cache_hit_count > 0
    assert sink.vector_cache_hit_count > 0


def test_guard_high_water_fields_cover_every_internal_count_guard(
    node3_result,
) -> None:
    for partition in node3_result.partitions:
        assert (
            partition.peak_raw_operator_application_count
            == partition.raw_operator_application_count
        )
        assert partition.peak_behavior_class_count == partition.behavior_class_count
        assert (
            partition.peak_visible_frontier_point_count
            >= partition.visible_frontier_point_count
        )
        assert (
            partition.peak_visible_frontier_points_per_class
            >= partition.maximum_frontier_points_per_class
        )
        assert (
            partition.peak_continuation_bank_point_count
            == partition.continuation_bank_point_count
        )
        assert (
            partition.peak_continuation_bank_points_per_class
            == partition.maximum_bank_points_per_class
        )
        assert partition.peak_work_queue_points >= partition.vector_cache_entry_count
        assert partition.peak_work_queue_points == 810
        assert partition.peak_saturation_round_count == len(partition.depth_barriers)
        assert partition.peak_saturation_round_count == 5


def test_node4_captures_nonmonotonic_frontier_and_work_queue_peaks(
    node4_result,
) -> None:
    odd, sink = node4_result.partitions
    assert (
        odd.raw_operator_application_count,
        odd.behavior_class_count,
        odd.visible_frontier_point_count,
        odd.continuation_bank_point_count,
    ) == (1_844, 107, 154, 478)
    assert odd.peak_visible_frontier_point_count == 154
    assert odd.peak_visible_frontier_points_per_class == 13
    assert odd.maximum_frontier_points_per_class == 5
    assert (
        odd.peak_visible_frontier_points_per_class
        > odd.maximum_frontier_points_per_class
    )

    assert (
        sink.raw_operator_application_count,
        sink.behavior_class_count,
        sink.visible_frontier_point_count,
        sink.continuation_bank_point_count,
    ) == (2_108, 47, 138, 466)
    assert sink.peak_work_queue_points == 1_023
    assert sink.vector_cache_entry_count == 197
    assert sink.peak_work_queue_points > sink.vector_cache_entry_count


def test_preflight_is_diagnostic_and_keeps_all_downstream_authority_closed(
    node3_result,
) -> None:
    assert node3_result.diagnostic_only is True
    assert node3_result.formal_roots_generated is False
    assert node3_result.formal_roots is None
    assert node3_result.target_truth_accessed is False
    assert node3_result.split_accessed is False
    assert node3_result.role_evaluation_performed is False
    assert node3_result.complete_claim_allowed is False
    for partition in node3_result.partitions:
        assert partition.traversal_closed is True
        assert partition.diagnostic_only is True
        assert partition.formal_roots_generated is False
        assert partition.formal_roots is None
        assert partition.target_truth_accessed is False
        assert partition.split_accessed is False
        assert partition.role_evaluation_performed is False
        assert partition.complete_claim_allowed is False


def test_resource_guard_is_fail_closed_without_formal_output() -> None:
    limits = preflight.PreflightLimitsV1(
        maximum_ast_node_count=3,
        maximum_behavior_classes=1,
    )
    result = preflight.run_q1_partition_capacity_preflight_v1(1, limits=limits)
    assert result.terminal_status == preflight.PREFLIGHT_CAPACITY_GUARD_HIT
    assert (result.resource_guard_id, result.resource_guard_name) == (
        2,
        "BEHAVIOR_CLASSES",
    )
    assert result.behavior_class_count == 1
    assert result.behavior_class_count <= limits.maximum_behavior_classes
    assert result.traversal_closed is False
    assert result.formal_roots_generated is False
    assert result.formal_roots is None
    assert result.complete_claim_allowed is False


def test_rejected_frontier_event_does_not_enter_high_water() -> None:
    limits = preflight.PreflightLimitsV1(
        maximum_ast_node_count=3,
        maximum_visible_frontier_points=1,
    )
    result = preflight.run_q1_partition_capacity_preflight_v1(1, limits=limits)
    assert (result.resource_guard_id, result.resource_guard_name) == (
        3,
        "VISIBLE_FRONTIER_TOTAL",
    )
    assert result.visible_frontier_point_count == 1
    assert result.peak_visible_frontier_point_count == 1
    assert result.raw_operator_application_count == 2
    assert result.peak_raw_operator_application_count == 2


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("maximum_ast_depth", -1),
        ("maximum_ast_depth", 4),
        ("maximum_ast_node_count", 0),
        ("maximum_ast_node_count", 7),
        ("maximum_raw_operator_applications", 0),
        ("maximum_work_queue_points", 0),
    ),
)
def test_preflight_limit_boundary_rejects_out_of_v16_range(field, value) -> None:
    arguments = {field: value}
    with pytest.raises(ValueError):
        preflight.PreflightLimitsV1(**arguments)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("maximum_ast_depth", True),
        ("maximum_ast_node_count", False),
        ("maximum_raw_operator_applications", True),
        ("maximum_behavior_classes", 1.0),
        ("maximum_continuation_bank_points", "2"),
        ("maximum_work_queue_points", False),
        ("maximum_wall_time_seconds", 2.0),
    ),
)
def test_preflight_limit_guards_are_type_exact(field, value) -> None:
    with pytest.raises(ValueError):
        preflight.PreflightLimitsV1(**{field: value})


@pytest.mark.parametrize(
    "value",
    (True, False, 0, 3, 1.0, "1", None),
)
def test_input_signature_is_type_exact_and_closed_to_one_or_two(value) -> None:
    with pytest.raises(preflight.Q1CapacityPreflightError) as caught:
        preflight.run_q1_partition_capacity_preflight_v1(value)
    assert caught.value.code == preflight.REJECT_PREFLIGHT_INPUT_SIGNATURE


def test_guard_registry_and_internal_guard_mapping_are_preregistered() -> None:
    assert preflight.RESOURCE_GUARD_REGISTRY == (
        (1, "RAW_OPERATOR_APPLICATIONS"),
        (2, "BEHAVIOR_CLASSES"),
        (3, "VISIBLE_FRONTIER_TOTAL"),
        (4, "VISIBLE_FRONTIER_PER_CLASS"),
        (5, "CONTINUATION_BANK_TOTAL"),
        (6, "CONTINUATION_BANK_PER_CLASS"),
        (7, "WORK_QUEUE_POINTS"),
        (8, "SATURATION_ROUNDS"),
        (9, "OUTPUT_BYTES"),
        (10, "SCRATCH_BYTES"),
        (11, "RESIDENT_MEMORY"),
        (12, "WALL_TIME"),
    )
    limits = preflight.PreflightLimitsV1(
        maximum_ast_node_count=3,
        maximum_continuation_bank_points=1,
        maximum_continuation_bank_points_per_class=1,
    )
    result = preflight.run_q1_partition_capacity_preflight_v1(1, limits=limits)
    assert result.terminal_status == preflight.PREFLIGHT_CAPACITY_GUARD_HIT
    assert (result.resource_guard_id, result.resource_guard_name) == (
        5,
        "CONTINUATION_BANK_TOTAL",
    )
    assert (
        result.signature_cohort_count
        <= limits.maximum_continuation_bank_points
    )
    assert (
        result.continuation_bank_point_count
        <= limits.maximum_continuation_bank_points
    )


def test_guarded_candidate_event_rolls_back_counters_and_vector_cache() -> None:
    limits = preflight.PreflightLimitsV1(
        maximum_ast_node_count=3,
        maximum_work_queue_points=1,
    )
    result = preflight.run_q1_partition_capacity_preflight_v1(1, limits=limits)
    assert result.terminal_status == preflight.PREFLIGHT_CAPACITY_GUARD_HIT
    assert (result.resource_guard_id, result.resource_guard_name) == (
        7,
        "WORK_QUEUE_POINTS",
    )
    assert result.vector_cache_entry_count == 0
    assert result.vector_cache_miss_count == 0
    assert result.vector_cache_hit_count == 0
    assert result.raw_operator_application_count == 1
    assert result.strict_admitted_application_count == 1
    assert result.rewrite_collapse_count == 0
    assert result.peak_raw_operator_application_count == 1
    assert result.peak_behavior_class_count == result.behavior_class_count
    assert result.peak_work_queue_points == 1
    assert result.peak_saturation_round_count == 0


def test_class_guard_rolls_back_the_entire_first_operator_event() -> None:
    limits = preflight.PreflightLimitsV1(
        maximum_ast_node_count=3,
        maximum_behavior_classes=15,
    )
    result = preflight.run_q1_partition_capacity_preflight_v1(1, limits=limits)
    assert result.terminal_status == preflight.PREFLIGHT_CAPACITY_GUARD_HIT
    assert (result.resource_guard_id, result.resource_guard_name) == (
        2,
        "BEHAVIOR_CLASSES",
    )
    assert result.behavior_class_count == 15
    assert result.raw_operator_application_count == 810
    assert result.strict_admitted_application_count == 810
    assert result.rewrite_collapse_count == 0
    assert result.vector_cache_entry_count == 0
    assert result.vector_cache_hit_count == 0
    assert result.vector_cache_miss_count == 0
    assert len(result.depth_barriers) == 1
    assert result.depth_barriers[0].barrier_kind == "LEAF_SEED"
    assert result.peak_raw_operator_application_count == 810
    assert result.peak_behavior_class_count == 15
    assert result.peak_work_queue_points == 810
    assert result.peak_saturation_round_count == 1


def test_canonical_diagnostic_json_schema_is_stable_and_python_only(
    node3_result,
) -> None:
    encoded = preflight.canonical_capacity_preflight_json_bytes_v1(node3_result)
    assert encoded.endswith(b"\n")
    assert b"dual" not in encoded.lower()
    assert encoded == preflight.canonical_capacity_preflight_json_bytes_v1(
        node3_result
    )
    assert len(encoded) == 8_852
    assert sha256(encoded).hexdigest() == (
        "58a7e160aa16eed84ba1c96e5b067e4509438c5389698a96ad0610130450a4c1"
    )
    value = json.loads(encoded)
    assert set(value) == {
        "active_transition_allowed",
        "closure_semantics_version",
        "complete_claim_allowed",
        "diagnostic_only",
        "dsl_version",
        "formal_roots",
        "formal_roots_generated",
        "m3_formal_roots",
        "outside_certificate_issued",
        "partitions",
        "preflight_id",
        "q1_formal_roots",
        "q1_gate_count",
        "q1_gate_mask",
        "q1_receipt",
        "q1_state",
        "q2_state",
        "role_evaluation_performed",
        "schema_version",
        "split_accessed",
        "target_truth_accessed",
        "terminal_status",
    }
    assert value["preflight_id"] == "hegel-phase3a-q1-capacity-preflight-v1"
    assert value["terminal_status"] == "LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED"
    assert value["q1_state"] == "NOT_RUN"
    assert value["q1_gate_count"] == 0
    assert value["q1_gate_mask"] == 0
    assert value["q1_formal_roots"] is None
    assert value["q1_receipt"] is None
    assert value["q2_state"] == "NOT_RUN"
    assert value["m3_formal_roots"] is None
    assert value["outside_certificate_issued"] is False
    assert value["active_transition_allowed"] is False
    assert [row["input_signature_id"] for row in value["partitions"]] == [1, 2]
    assert all(row["formal_roots"] is None for row in value["partitions"])
    assert all(row["limits"]["maximum_ast_node_count"] == 3 for row in value["partitions"])
    assert all(row["universe_root"].startswith("sha256:") for row in value["partitions"])
    assert all(
        row["peak_raw_operator_application_count"]
        == row["raw_operator_application_count"]
        for row in value["partitions"]
    )
    assert all(row["peak_work_queue_points"] == 810 for row in value["partitions"])


def test_source_import_graph_has_no_target_truth_or_split_module_dependency() -> None:
    path = Path(preflight.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
    forbidden = {
        "phase3_dsl_v1",
        "phase3_m25_rows_v1",
        "phase3_m25_split_v1",
        "phase3_m25_formal_static_basis_v1",
        "phase3_q0_quotient_contract_v1",
    }
    assert imported.isdisjoint(forbidden)


def test_empty_package_bootstrap_loads_no_forbidden_role_or_truth_module() -> None:
    package_path = SRC / "hegel_machine"
    code = f"""
import importlib
import sys
import types
package = types.ModuleType('hegel_machine')
package.__path__ = [{str(package_path)!r}]
package.__package__ = 'hegel_machine'
sys.modules['hegel_machine'] = package
importlib.import_module('hegel_machine.phase3_q1_capacity_preflight_v1')
forbidden = (
    'hegel_machine.phase3_dsl_v1',
    'hegel_machine.phase3_m25_rows_v1',
    'hegel_machine.phase3_m25_split_v1',
    'hegel_machine.phase3_m25_formal_static_basis_v1',
    'hegel_machine.phase3_q0_quotient_contract_v1',
)
loaded = [name for name in forbidden if name in sys.modules]
if loaded:
    raise SystemExit('forbidden modules loaded: ' + ','.join(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
