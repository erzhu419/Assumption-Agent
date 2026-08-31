from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_SPEC = importlib.util.spec_from_file_location(
    "phase3_q3_bounded_quotient_certificate_tool_test_state",
    ROOT / "tools/phase3_q3_bounded_quotient_certificate_v1.py",
)
assert TOOL_SPEC is not None and TOOL_SPEC.loader is not None
tool = importlib.util.module_from_spec(TOOL_SPEC)
sys.modules[TOOL_SPEC.name] = tool
TOOL_SPEC.loader.exec_module(tool)
state = tool.state


def _loaded() -> tuple[object, object, object, list[dict[str, object]]]:
    contract = state.load_contract(tool.DEFAULT_CONTRACT)
    prereg = state.load_preregistration(Path(contract.value["preregistration_path"]))
    replay = tool._replay_q2(contract, prereg)
    rows = tool._source_closure_rows(contract)
    return contract, prereg, replay, rows


def _capsule() -> tuple[object, object, dict[str, object]]:
    contract, prereg, replay, rows = _loaded()
    inputs = {
        "schema_version": state.CAPSULE_INPUTS_SCHEMA,
        "source_closure_rows": rows,
        "source_closure_root": state.source_closure_root(rows),
        "source_commit": "1" * 40,
        "run_id": "2" * 32,
        "created_at_unix_seconds": 1_788_200_000,
        "preflight_receipt": replay.preflight_receipt,
    }
    capsule = state.build_capsule(contract, prereg.value, inputs)
    return contract, prereg, capsule


def test_sealed_q2_five_input_replay_is_exact_and_zero_action() -> None:
    _, _, replay, _ = _loaded()
    receipt = replay.preflight_receipt

    assert [row["kind"] for row in receipt["selected_q2_inputs"]] == [
        "capsule",
        "authorization",
        "consumption",
        "terminal",
        "artifact",
    ]
    assert receipt["selected_q2_input_total_bytes"] == 22_898
    assert receipt["producer_free_read_only_replay_completed"] is True
    assert receipt["sealed_q2_truth_commitment_replay_completed"] is True
    assert receipt["q3_action_audit"] == state.ZERO_ACTION_AUDIT
    assert all(value == 0 for value in state.ZERO_ACTION_AUDIT.values() if type(value) is int)
    assert all(value is False for value in state.ZERO_ACTION_AUDIT.values() if type(value) is bool)


def test_capsule_digest_is_independently_recomputed_and_full_binding() -> None:
    contract, prereg, capsule = _capsule()
    replayed = state.validate_capsule(capsule, contract, prereg.value)
    assert replayed == capsule

    unsigned = deepcopy(capsule)
    supplied = unsigned.pop("capsule_digest")
    independently_computed = "sha256:" + sha256(
        state.DOMAINS["capsule"] + state.canonical_json_bytes(unsigned)
    ).hexdigest()
    assert supplied == independently_computed
    assert capsule["source_closure_root"] == state.source_closure_root(
        capsule["source_closure_rows"]
    )
    assert capsule["execution_policy"] == state.EXECUTION_POLICY
    assert capsule["authority_boundary"] == state.AUTHORITY_BOUNDARY


def test_rehashed_producer_source_row_is_rejected() -> None:
    _, _, capsule = _capsule()
    rows = deepcopy(capsule["source_closure_rows"])
    rows.append(
        {
            "relative_path": "src/hegel_machine/phase3_m25_rows_v1.py",
            "byte_count": 1,
            "raw_sha256": "sha256:" + "00" * 32,
        }
    )
    with pytest.raises(state.Q3CompletionReplayError, match="row count"):
        state.validate_source_closure_rows(rows)


def test_authorization_skeleton_is_digest_bound_and_all_actions_zero() -> None:
    contract, prereg, capsule = _capsule()
    authorization = state.build_authorization(
        contract, prereg.value, capsule
    )
    assert state.validate_authorization(
        authorization, contract, prereg.value, capsule
    ) == authorization
    unsigned = deepcopy(authorization)
    supplied = unsigned.pop("authorization_digest")
    expected = "sha256:" + sha256(
        state.DOMAINS["authorization"] + state.canonical_json_bytes(unsigned)
    ).hexdigest()
    assert supplied == expected
    assert authorization["capsule_digest"] == capsule["capsule_digest"]
    assert authorization["authorized_scope"] == state.AUTHORIZED_SCOPE
    for key in (
        "maximum_internal_popen_count",
        "maximum_actor_start_count",
        "maximum_docker_invocation_count",
        "maximum_measurement_count",
        "maximum_target_open_count",
        "maximum_role_evaluation_count",
        "maximum_split_access_count",
        "maximum_q1_archive_read_count",
    ):
        assert authorization[key] == 0
    for key in (
        "network_allowed",
        "certificate_signing_allowed",
        "outside_certificate_issuance_allowed",
        "synthesis_allowed",
        "mdl_allowed",
        "external_scientific_claim_allowed",
        "active_transition_allowed",
        "production_allowed",
        "adoption_allowed",
        "promotion_allowed",
        "current_pointer_update_allowed",
    ):
        assert authorization[key] is False

    mutated = deepcopy(authorization)
    mutated["maximum_target_open_count"] = 1
    with pytest.raises(state.Q3CompletionReplayError, match="authorization replay"):
        state.validate_authorization(mutated, contract, prereg.value, capsule)


@pytest.mark.parametrize(
    "kind",
    ["capsule", "authorization", "consumption", "terminal", "artifact"],
)
def test_each_sealed_q2_raw_input_byte_drift_is_rejected(kind: str) -> None:
    contract = state.load_contract(tool.DEFAULT_CONTRACT)
    prereg = state.load_preregistration(Path(contract.value["preregistration_path"]))
    selected = tool._read_q2_selected(contract)
    original = selected[kind]
    selected[kind] = bytes((original[0] ^ 1,)) + original[1:]

    with pytest.raises(state.Q3CompletionReplayError, match="raw SHA-256"):
        state.replay_q2_predecessor_v1(
            prereg.value,
            capsule_payload=selected["capsule"],
            authorization_payload=selected["authorization"],
            consumption_payload=selected["consumption"],
            terminal_payload=selected["terminal"],
            artifact_payload=selected["artifact"],
        )


def test_pending_is_same_capsule_and_never_consumed() -> None:
    contract, prereg, capsule = _capsule()
    authorization = state.build_authorization(contract, prereg.value, capsule)
    pending = state.build_pending(
        contract,
        prereg.value,
        capsule,
        phase="SOURCE_CLOSURE",
        condition="Q3_CAPSULE_SOURCE_CLOSURE_DIFFERS",
        detail="repair local source then replay the same capsule",
        authorization=authorization,
    )
    assert pending["resume_capsule_digest"] == capsule["capsule_digest"]
    assert pending["authorization_consumption_count"] == 0
    assert pending["logical_campaign_count"] == 0
    assert pending["q3_action_audit"] == state.ZERO_ACTION_AUDIT


def test_complete_and_invalidated_terminals_are_exactly_replayed() -> None:
    contract, prereg, capsule = _capsule()
    authorization = state.build_authorization(contract, prereg.value, capsule)
    consumption = state.build_consumption(
        contract, prereg.value, capsule, authorization
    )
    replay = tool._replay_q2(contract, prereg)
    artifact = replay.candidate.canonical_bytes
    complete = state.build_complete_terminal(
        contract,
        prereg.value,
        capsule,
        authorization,
        consumption,
        artifact,
    )
    assert state.validate_terminal(
        complete,
        contract,
        prereg.value,
        capsule,
        authorization,
        consumption,
        artifact_payload=artifact,
    ) == complete

    changed = deepcopy(complete)
    changed["outcome"]["campaign_complete"] = False
    with pytest.raises(state.Q3CompletionReplayError, match="terminal replay"):
        state.validate_terminal(
            changed,
            contract,
            prereg.value,
            capsule,
            authorization,
            consumption,
            artifact_payload=artifact,
        )

    invalidated = state.build_invalidated_terminal(
        contract,
        prereg.value,
        capsule,
        authorization,
        consumption,
        failure_phase="POST_CONSUMPTION_FINALIZATION",
        failure_code="Q3_CANDIDATE_ARTIFACT_MISSING_AFTER_CONSUMPTION",
        failure_detail="sealed candidate absent",
    )
    assert state.validate_terminal(
        invalidated,
        contract,
        prereg.value,
        capsule,
        authorization,
        consumption,
    ) == invalidated
    with pytest.raises(state.Q3CompletionReplayError, match="forbids"):
        state.validate_terminal(
            invalidated,
            contract,
            prereg.value,
            capsule,
            authorization,
            consumption,
            artifact_payload=artifact,
        )
