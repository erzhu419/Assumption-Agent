from __future__ import annotations

from hashlib import sha256
import importlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

preflight = importlib.import_module("hegel_machine.phase3_q1_capacity_preflight_v1")

CONFIG_PATH = ROOT / "config/phase3_q1_capacity_preflight_v1.json"
DOC_PATH = ROOT / "docs/Hegel_Machine_Phase3A_Q1_Capacity_Preflight_Preregistration_v1.md"


def _strict_object(pairs):
    value = {}
    for key, item in pairs:
        assert key not in value
        value[key] = item
    return value


def _config():
    return json.loads(
        CONFIG_PATH.read_bytes(),
        object_pairs_hook=_strict_object,
        parse_constant=lambda value: (_ for _ in ()).throw(AssertionError(value)),
    )


def test_preregistration_authority_and_gate_registry_are_fail_closed() -> None:
    value = _config()
    authority = value["authority"]
    assert value["preflight_status"] == "PREREGISTERED_NOT_RUN"
    assert value["phase_position"] == (
        "Q0.5_TARGET_BLIND_Q1_ADMISSION_PREFLIGHT_Q1_REMAINS_NOT_RUN"
    )
    assert authority["q1_state"] == "NOT_RUN"
    assert authority["q1_gate_count"] == 0
    assert authority["q1_gate_mask"] == 0
    assert authority["q1_formal_roots"] is None
    assert authority["q1_receipt"] is None
    assert authority["q2_state"] == "NOT_RUN"
    assert authority["m3_formal_roots"] is None
    assert authority["outside_certificate_issued"] is False
    assert authority["active_transition_allowed"] is False
    gates = value["planned_q1_gate_registry"]
    assert len(gates) == 20
    assert all(
        type(row) is list
        and len(row) == 2
        and type(row[0]) is int
        and type(row[1]) is str
        for row in gates
    )
    assert [row[0] for row in gates] == list(range(1, 21))
    assert len({row[1] for row in gates}) == 20
    assert gates[9][1] == "Q1_ARCHIVE_WIRE_ROOT_DAG_RESOURCE_PROJECTION_AND_GOLDENS_PASS"
    assert gates[10][1] == "TARGET_BLIND_CAPACITY_PREFLIGHT_AND_RESOURCE_ENVELOPE_FROZEN"
    admission = value["authoritative_preflight_admission"]
    assert admission["full_dual_node6_preflight_allowed_now"] is False
    assert admission["archive_wire_root_dag_and_projection_profile_frozen"] is False
    assert admission["rust_endpoint_source_available"] is False
    assert admission["dual_supervisor_available"] is False
    assert admission["allowed_current_execution"] == (
        "IMPORT_ISOLATED_LOCAL_SUBSET_QUALIFICATION_ONLY"
    )


def test_diagnostic_wire_field_sets_match_python_serializer() -> None:
    value = _config()
    wire = value["diagnostic_wire_contract"]
    result = preflight.run_q1_capacity_preflight_v1(
        limits=preflight.PreflightLimitsV1(maximum_ast_node_count=1)
    )
    diagnostic = preflight.capacity_preflight_diagnostic_object_v1(result)
    partition = diagnostic["partitions"][0]
    barrier = partition["depth_barriers"][0]
    assert wire["engine_schema_version"] == preflight.SCHEMA_VERSION
    assert sorted(wire["engine_top_level_fields"]) == sorted(diagnostic)
    assert sorted(wire["partition_fields"]) == sorted(partition)
    assert sorted(wire["limit_fields"]) == sorted(partition["limits"])
    assert sorted(wire["depth_barrier_fields"]) == sorted(barrier)
    encoded = preflight.canonical_capacity_preflight_json_bytes_v1(result)
    assert encoded.endswith(b"\n")
    assert encoded.count(b"\n") == 1
    assert json.loads(encoded, object_pairs_hook=_strict_object) == diagnostic
    assert wire["formal_cbor_tag_or_root_authority"] is False


def test_budget_projection_does_not_substitute_small_diagnostic_output() -> None:
    value = _config()
    budget = value["formal_budget_derivation"]
    measurement = value["measurement_contract"]
    assert budget[
        "archive_wire_and_resource_projection_must_be_frozen_before_authoritative_preflight"
    ] is True
    assert "high_water" in budget["count_guard_formula"]
    assert budget["actual_preflight_diagnostic_output_bytes_are_not_formal_archive_projection"] is True
    assert budget["actual_preflight_peak_scratch_bytes_are_not_formal_scratch_projection"] is True
    assert "projected_formal_total_output_bytes_per_endpoint" in budget["output_guard_formula"]
    assert "projected_peak_formal_scratch_bytes_per_endpoint" in budget["scratch_guard_formula"]
    assert "projected_formal_archive_payload_bytes" in measurement
    assert "projected_formal_total_output_bytes" in measurement
    assert "projected_host_replay_output_bytes" in measurement
    assert "projected_peak_host_replay_scratch_bytes" in measurement
    assert measurement["wire_or_allocator_change_invalidates_preflight"] is True
    assert measurement["nonmonotonic_frontier_final_value_is_not_a_guard_high_water"] is True
    aggregation = budget["count_aggregation"]
    assert aggregation["raw_operator_application_counts"].startswith(
        "REQUIRE_PYTHON_RUST_EXACT_EQUALITY"
    )
    assert aggregation["work_queue_high_water_counts"].startswith("USE_MAXIMUM")
    host = budget["host_replay_resource_envelope"]
    assert host["algorithm_allocator_and_wire_must_be_frozen_before_authoritative_preflight"] is True
    assert host["authoritative_preflight_must_execute_and_measure_host_replay"] is True
    assert host["resource_failure_without_exact_capacity_infeasibility_adjudication"] == (
        "INCONCLUSIVE_NO_PATH_B_AUTHORITY"
    )
    isolation = value["runtime_isolation_profile"]
    assert isolation["host_replay_container_after_endpoints_exit"] is True
    assert isolation["host_replay_is_third_independent_endpoint"] is False
    assert isolation["host_replay_container_memory_bytes"] == 14 * 1024**3
    assert isolation["host_replay_container_memory_swap_bytes"] == 14 * 1024**3


def test_path_b_and_d_require_new_normative_authority() -> None:
    value = _config()
    path_b = value["path_b_eligibility_contract"]
    path_d = value["path_d_contract"]
    assert path_b["automatic_path_b_transition_allowed"] is False
    assert path_b["ordinary_guard_hit_is_path_b_eligible"] is False
    assert len(path_b["required_conditions"]) == 5
    assert path_d["preflight_can_select_path_d"] is False
    assert path_d["path_b_is_not_mandatory_when_its_capacity_preconditions_do_not_hold"] is True
    assert path_d["transition_requires_new_normative_amendment"] is True


def test_historical_source_y_negative_is_preserved_without_q1_authority() -> None:
    value = _config()["historical_source_y_evidence_preservation"]
    artifact = ROOT.parent / value["artifact_path"]
    assert artifact == ROOT / "artifacts/phase3_shrink6_dual_complete_enumeration_diagnostic_v1.json"
    assert "sha256:" + sha256(artifact.read_bytes()).hexdigest() == value["artifact_sha256"]
    assert value["source_y_execution_source_commit"] == (
        "5217568303d5c7f902682c092750f637c64f080a"
    )
    assert value["evidence_z_repository_commit"] == (
        "ea98157f5d6eb2930ab28dda8f3a6839b343673c"
    )
    assert value["authority"] == "HISTORICAL_SYNTACTIC_CAPACITY_NEGATIVE_ONLY"
    assert value["q1_gate_effect"] == "NONE"
    assert value["cannot_be_reinterpreted_as_quotient_cardinality_or_closure_result"] is True


def test_document_declares_current_blockers_and_exact_route_order() -> None:
    document = DOC_PATH.read_text(encoding="utf-8")
    required = (
        "Q0.5 / Q1-admission preflight",
        "LOCAL_UNISOLATED_NON_FORMAL_PROTOTYPE_GUIDANCE_ONLY",
        "STRUCTURAL_BOUNDARY",
        "Q0.5a archive/resource",
        "A no-argument node-six run must fail closed",
        "Path B is not authorized by an ordinary guard hit",
        "Path B is not a mandatory detour",
        "HISTORICAL_SYNTACTIC_CAPACITY_NEGATIVE_ONLY",
    )
    assert all(text in document for text in required)
