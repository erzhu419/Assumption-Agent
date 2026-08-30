from __future__ import annotations

import json
from pathlib import Path

import pytest

from hegel_machine import phase3_q1_capacity_preflight_v1 as old_capacity
from hegel_machine import phase3_q1_full_node6_capacity_preflight_v1 as full


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config/phase3_q1_full_node6_capacity_preflight_v1.json"
FROZEN_CONFIG_PATH = PROJECT_ROOT / "config/phase3_q1_capacity_preflight_v1.json"
DOC_PATH = (
    PROJECT_ROOT
    / "docs/Hegel_Machine_Phase3A_Q1_Full_Node6_Capacity_Preflight_Successor_v1.md"
)


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_successor_contract_keeps_q1_and_downstream_authority_closed() -> None:
    value = _config()
    assert value["schema_version"] == (
        "hegel-phase3a-q1-full-node6-capacity-preflight-contract/1"
    )
    assert value["action_id"] == full.ACTION_ID
    authority = value["authority"]
    assert authority == {
        "allowed_success_status": full.TERMINAL_SATURATED,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_formal_roots": None,
        "q1_output_slots": [None] * 8,
        "q1_receipt": None,
        "q2_state": "NOT_RUN",
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
        "m3_formal_roots": None,
        "outside_certificate_issued": False,
        "active_transition_allowed": False,
        "adoption_allowed": False,
        "promotion_allowed": False,
        "current_pointer_write_allowed": False,
        "external_scientific_claim_allowed": False,
    }


def test_contract_binds_successful_q05b_predecessor_and_frozen_projection() -> None:
    value = _config()
    predecessor = value["predecessor"]
    assert predecessor["source_commit"] == (
        "33dc9cbb7d753f83650169d6d6b9caa508927764"
    )
    assert predecessor["terminal_status"] == (
        "ATTEMPT2_TARGET_D_V2_QUALIFIED_Q1_NOT_RUN"
    )
    assert predecessor["qualification_count"] == 20
    assert predecessor["qualification_mask"] == 0xFFFFF
    assert predecessor["q1_state"] == "NOT_RUN"
    binding = value["semantic_and_projection_binding"]
    assert binding["full_v16_leaf_manifest_root"] == (
        "sha256:3fefacd3db59294f2b6d44a5d0b813e73af3ec84742a24ab846bbdacae6c1f1b"
    )
    assert binding["semantic_binding_root"] == (
        "sha256:e3b3df3e81b7632c7c713ef5ec84913f990ad8232a25b851f20c46ac7416bfcb"
    )
    assert binding["projection_profile_root"] == (
        "sha256:aa441cdc49ab60324483b9aa44e9fdfc324a6ad49a6bff50af6daa775209816d"
    )
    assert binding["counting_discard_required"] is True
    assert binding["materialize_formal_archive"] is False


def test_measurement_and_pending_boundaries_are_explicit() -> None:
    value = _config()
    resource = value["resource_envelope"]
    assert resource["maximum_concurrently_live_container_memory_bytes"] == (
        14 * 1024**3
    )
    assert resource["host_control_plane_memory_reserve_bytes"] == 2 * 1024**3
    assert resource["minimum_available_memory_bytes"] == 16 * 1024**3
    assert resource[
        "outer_preparation_and_final_publication_reserve_bytes"
    ] == 4 * 1024**3
    for actor in ("python_endpoint", "rust_endpoint"):
        assert resource[actor]["scratch_bytes"] == 48 * 1024**3
        assert resource[actor]["scratch_mount"] == "/scratch"
        assert resource[actor]["output_bytes"] == 32 * 1024**3
        assert resource[actor]["output_mount"] == "/output"
    assert resource["python_runtime_version"] == "3.11.15"
    measurement = value["measurement_policy"]
    assert measurement["maximum_quotient_measurements_per_endpoint_and_signature"] == 1
    assert measurement["per_signature_wall_time_seconds"] == 172800
    assert measurement["ordered_signature_count"] == 2
    assert measurement["bounded_projection_wall_time_seconds"] == 172800
    assert measurement["python_and_rust_run_concurrently"] is False
    assert measurement["python_then_rust_run_strictly_sequentially"] is True
    assert measurement["maximum_concurrently_live_measurement_containers"] == 1
    assert measurement["endpoint_process_deadline_seconds"] == 518400
    assert measurement["endpoint_sequence_process_budget_seconds"] == 1036800
    assert measurement["trusted_host_replay_process_deadline_seconds"] == 172800
    assert measurement["fresh_quotient_rerun_during_projection_forbidden"] is True
    assert measurement["projection_only_application_identity_replay_allowed"] is True
    execution = value["execution_policy"]
    assert execution["endpoint_execution_order"] == [
        "PYTHON_ENDPOINT",
        "RUST_ENDPOINT",
    ]
    assert execution["maximum_concurrently_live_endpoint_containers"] == 1
    assert execution["outer_supervisor_wall_time_seconds"] == 1212000
    pending = value["pending_policy"]
    assert pending["pre_consumption_pending_allowed"] is True
    assert pending["same_run_root_and_capsule_digest_required"] is True
    assert pending["core_measurement_failure_pending_forbidden"] is True
    assert pending["post_consumption_pending_forbidden"] is True


def test_frozen_measurement_and_isolation_profiles_are_replayed_exactly() -> None:
    value = _config()
    frozen = json.loads(FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))
    assert value["measurement_contract"] == frozen["measurement_contract"]
    isolation = dict(value["runtime_isolation_profile"])
    assert isolation.pop("endpoint_user") == "65534:65534"
    assert isolation.pop("network") == "none"
    assert isolation.pop("pull_policy") == "never"
    assert isolation == frozen["runtime_isolation_profile"]


def test_execution_source_closure_is_exact_and_target_blind() -> None:
    closure = _config()["execution_source_closure"]
    python_paths = closure["python_endpoint_relative_paths_in_exact_order"]
    rust_paths = closure["rust_release_build_relative_paths_in_exact_order"]
    host_paths = closure["host_replay_relative_paths_in_exact_order"]
    assert len(python_paths) == closure["python_endpoint_row_count"] == 34
    assert len(rust_paths) == closure["rust_release_build_row_count"] == 21
    assert len(set(python_paths) | set(rust_paths)) == (
        closure["endpoint_execution_union_row_count"]
    ) == 55
    assert host_paths == [
        "src/hegel_machine/phase3_q1_full_node6_host_replay_v1.py"
    ]
    assert closure["host_replay_row_count"] == 1
    assert closure["target_split_role_files_mounted"] is False
    assert "rust/q1_archive_projection_oracle/Cargo.toml" in rust_paths
    assert "rust/q1_archive_projection_oracle/src/lib.rs" in rust_paths


def test_historical_public_api_remains_fail_closed_for_full_node6() -> None:
    with pytest.raises(old_capacity.Q1CapacityPreflightError) as caught:
        old_capacity.run_q1_capacity_preflight_v1()
    assert caught.value.code == old_capacity.REJECT_FULL_NODE6_PREFLIGHT_NOT_AUTHORIZED


def test_successor_document_states_the_only_allowed_success_claim() -> None:
    payload = DOC_PATH.read_text(encoding="utf-8")
    assert "PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY" in payload
    assert "Q1 remains `NOT_RUN`" in payload
    assert "separate later `phase3-q1-start`" in payload
    assert "successor-v3" in payload
    assert "measurement outcome is `UNKNOWN`" in payload
    assert "not a resume or continuation" in payload
    assert (
        "授权 Q1 full-node6 capacity-preflight diagnostic "
        "successor-v3 distinct-outer actual"
    ) in payload
    assert "The gauntlet is implemented" in payload
    assert "not yet been executed, preflighted or authorized" in payload
