from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
STATE_ROOT = Path("/home/erzhu419/.local/state/hegel-machine")
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hegel_machine import phase3_q5b_completion_replay_v1 as state  # noqa: E402
from hegel_machine import phase3_q5b_purpose5_trust_amendment_v1 as wire  # noqa: E402


def _q5a_payloads() -> dict[str, bytes]:
    return {
        "capsule": (
            STATE_ROOT / "q5a-purpose5-signer-genesis-v1-run/capsule.json"
        ).read_bytes(),
        "authorization": (
            STATE_ROOT
            / "q5a-purpose5-signer-genesis-v1-ledger/owner.authorization.json"
        ).read_bytes(),
        "consumption": (
            STATE_ROOT
            / "q5a-purpose5-signer-genesis-v1-ledger/campaign.consumed.json"
        ).read_bytes(),
        "terminal": (
            STATE_ROOT
            / "q5a-purpose5-signer-genesis-v1-ledger/campaign.terminal.json"
        ).read_bytes(),
        "artifact": (
            ROOT / "artifacts/phase3_q5a_purpose5_signer_genesis_candidate_v1.cbor"
        ).read_bytes(),
        "trajectory": (
            STATE_ROOT / "q5a-purpose5-signer-genesis-v1-run/trajectory.jsonl"
        ).read_bytes(),
    }


def _predecessor(payloads: dict[str, bytes] | None = None) -> dict[str, object]:
    selected = _q5a_payloads() if payloads is None else payloads
    capsule = state.strict_json_bytes(selected["capsule"], "Q5a capsule")
    authorization = state.strict_json_bytes(
        selected["authorization"], "Q5a authorization"
    )
    consumption = state.strict_json_bytes(selected["consumption"], "Q5a consumption")
    terminal = state.strict_json_bytes(selected["terminal"], "Q5a terminal")
    candidate = wire.decode_sealed_q5a_candidate_v1(selected["artifact"])
    return {
        "source_commit": capsule["source_commit"],
        "source_closure_root": capsule["source_closure_root"],
        "capsule": {
            "raw_sha256": state.raw_sha256(selected["capsule"]),
            "capsule_digest": capsule["capsule_digest"],
        },
        "authorization": {
            "raw_sha256": state.raw_sha256(selected["authorization"]),
            "authorization_digest": authorization["authorization_digest"],
        },
        "consumption": {
            "raw_sha256": state.raw_sha256(selected["consumption"]),
            "record_digest": consumption["record_digest"],
        },
        "terminal": {
            "raw_sha256": state.raw_sha256(selected["terminal"]),
            "record_digest": terminal["record_digest"],
        },
        "artifact": {
            "raw_sha256": state.raw_sha256(selected["artifact"]),
            "candidate_root": "sha256:" + candidate.candidate_root.hex(),
            "unpinned_genesis_root": (
                "sha256:" + candidate.unpinned_genesis_root.hex()
            ),
            "genesis_only_status_root": (
                "sha256:" + candidate.genesis_only_status_root.hex()
            ),
            "manifest_roots": [
                "sha256:" + item.manifest_root.hex()
                for item in candidate.signer_manifests
            ],
            "key_ids": [item.key_id.hex() for item in candidate.signer_manifests],
        },
        "trajectory": {
            "raw_sha256": state.raw_sha256(selected["trajectory"]),
            "event_count": 7,
        },
        "selected_input_total_bytes": sum(len(item) for item in selected.values()),
    }


def _preregistration(
    payloads: dict[str, bytes] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": state.PREREGISTRATION_SCHEMA,
        "preregistration_id": state.PREREGISTRATION_ID,
        "status": "Q5B_SPEC_FROZEN_NOT_AUTHORIZED",
        "predecessor_q5a": _predecessor(payloads),
    }


def _replay(
    *,
    payloads: dict[str, bytes] | None = None,
    preregistration: dict[str, object] | None = None,
    source_commit: str = "a" * 40,
    source_closure_root: str = "sha256:" + "b" * 64,
) -> state.Q5aSealedReplayV1:
    selected = _q5a_payloads() if payloads is None else payloads
    prereg = _preregistration(selected) if preregistration is None else preregistration
    return state.replay_q5a_predecessor_v1(
        prereg,
        q5b_source_commit=source_commit,
        q5b_source_closure_root=source_closure_root,
        capsule_payload=selected["capsule"],
        authorization_payload=selected["authorization"],
        consumption_payload=selected["consumption"],
        terminal_payload=selected["terminal"],
        artifact_payload=selected["artifact"],
        trajectory_payload=selected["trajectory"],
    )


def _contract(preregistration: dict[str, object]) -> state.LoadedJson:
    prereg_raw = state.canonical_json_bytes(preregistration)
    value = {
        "schema_version": state.CONTRACT_SCHEMA,
        "contract_id": state.CONTRACT_ID,
        "action_id": state.ACTION_ID,
        "preregistration_raw_sha256": state.raw_sha256(prereg_raw),
        "execution_policy": deepcopy(state.EXECUTION_POLICY),
        "authority_boundary": deepcopy(state.AUTHORITY_BOUNDARY),
    }
    raw = state.canonical_json_bytes(value)
    return state.LoadedJson(Path("q5b-test-contract.json"), raw, state.raw_sha256(raw), value)


def _source_rows() -> list[dict[str, object]]:
    return [
        {
            "relative_path": relative,
            "byte_count": len(relative.encode("ascii")),
            "raw_sha256": state.raw_sha256(relative.encode("ascii")),
        }
        for relative in state.Q5B_SOURCE_PATHS
    ]


def _initial_state():
    prereg = _preregistration()
    contract = _contract(prereg)
    rows = _source_rows()
    closure = state.source_closure_root(rows)
    replay = _replay(source_closure_root=closure)
    capsule = state.build_capsule(
        contract,
        prereg,
        {
            "source_commit": "a" * 40,
            "source_closure_rows": rows,
            "source_closure_root": closure,
            "run_id": "c" * 32,
            "created_at_unix_seconds": 1_788_500_000,
            "preflight_receipt": replay.preflight_receipt,
        },
    )
    authorization = state.build_authorization(contract, prereg, capsule)
    consumption = state.build_consumption(
        contract, prereg, capsule, authorization
    )
    return contract, prereg, capsule, authorization, consumption, replay


def test_exact_six_input_q5a_replay_derives_public_q5b_receipt() -> None:
    replay = _replay()
    preflight = replay.preflight_receipt
    assert preflight["selected_q5a_input_count"] == 6
    assert [item["kind"] for item in preflight["selected_q5a_inputs"]] == [
        "capsule", "authorization", "consumption", "terminal", "artifact",
        "trajectory",
    ]
    assert preflight["q5a_trajectory_audit"]["event_count"] == 7
    assert preflight["q5a_trajectory_audit"]["success_action_audit_replayed"] is True
    assert preflight["custody_root_access_count"] == 0
    assert preflight["private_key_access_count"] == 0
    assert replay.amendment_receipt.purpose5_active_signer_count == 3
    assert replay.amendment_receipt.global_active_transition_performed is False


def test_q5a_trajectory_drift_is_rejected() -> None:
    payloads = _q5a_payloads()
    trajectory = bytearray(payloads["trajectory"])
    trajectory[-2] ^= 1
    payloads["trajectory"] = bytes(trajectory)
    prereg = _preregistration(payloads)
    with pytest.raises(state.Q5bCompletionReplayError, match="trajectory"):
        _replay(payloads=payloads, preregistration=prereg)


def test_q5a_success_action_audit_is_exact() -> None:
    payloads = _q5a_payloads()
    terminal = state.strict_json_bytes(payloads["terminal"], "Q5a terminal")
    terminal["outcome"]["q5a_action_audit"]["signature_count"] = 1
    unsigned = deepcopy(terminal)
    unsigned.pop("record_digest")
    terminal["record_digest"] = state._digest(state.Q5A_DOMAINS["terminal"], unsigned)
    payloads["terminal"] = state.canonical_json_bytes(terminal)
    prereg = _preregistration(payloads)
    with pytest.raises(state.Q5bCompletionReplayError, match="completion boundary"):
        _replay(payloads=payloads, preregistration=prereg)


def test_q5a_capsule_q4_lineage_must_match_artifact_projection() -> None:
    payloads = _q5a_payloads()
    capsule = state.strict_json_bytes(payloads["capsule"], "Q5a capsule")
    capsule["preflight_receipt"]["q4_predecessor_receipt"][
        "q4_quotient_body_hash"
    ] = "sha256:" + "0" * 64
    unsigned = deepcopy(capsule)
    unsigned.pop("capsule_digest")
    capsule["capsule_digest"] = state._digest(state.Q5A_DOMAINS["capsule"], unsigned)
    payloads["capsule"] = state.canonical_json_bytes(capsule)
    authorization = state.strict_json_bytes(payloads["authorization"], "Q5a auth")
    authorization["capsule_digest"] = capsule["capsule_digest"]
    unsigned = deepcopy(authorization)
    unsigned.pop("authorization_digest")
    authorization["authorization_digest"] = state._digest(
        state.Q5A_DOMAINS["authorization"], unsigned
    )
    payloads["authorization"] = state.canonical_json_bytes(authorization)
    # The remaining chain intentionally stays sealed; replay must fail before
    # accepting the artifact's differing Q4 projection.
    prereg = _preregistration(payloads)
    with pytest.raises(state.Q5bCompletionReplayError):
        _replay(payloads=payloads, preregistration=prereg)


def test_capsule_authorization_consumption_and_pending_replay() -> None:
    contract, prereg, capsule, authorization, consumption, _ = _initial_state()
    assert state.validate_capsule(capsule, contract, prereg) == capsule
    assert state.validate_authorization(
        authorization, contract, prereg, capsule
    ) == authorization
    assert state.validate_consumption(
        consumption, contract, prereg, capsule, authorization
    ) == consumption
    pending = state.build_pending(
        contract,
        prereg,
        capsule,
        phase="SEALED_Q5A_REPLAY",
        condition="TEMPORARILY_UNAVAILABLE",
        detail="public file unavailable",
        authorization=authorization,
    )
    assert pending["authorization_consumption_count"] == 0
    assert pending["logical_campaign_count"] == 0
    assert pending["q5b_action_audit"] == state.ZERO_ACTION_AUDIT


def test_artifact_and_terminal_bind_outer_authority_and_sources() -> None:
    contract, prereg, capsule, authorization, consumption, replay = _initial_state()
    q5a_artifact = _q5a_payloads()["artifact"]
    receipt = state.validate_artifact_binding(
        replay.artifact_payload, q5a_artifact, capsule
    )
    assert receipt.q5a_capsule_digest == bytes.fromhex(
        capsule["preflight_receipt"]["q5a_capsule_digest"][7:]
    )
    terminal = state.build_complete_terminal(
        contract,
        prereg,
        capsule,
        authorization,
        consumption,
        replay.artifact_payload,
        q5a_artifact,
    )
    assert state.validate_terminal(
        terminal,
        contract,
        prereg,
        capsule,
        authorization,
        consumption,
        artifact_payload=replay.artifact_payload,
        q5a_artifact_payload=q5a_artifact,
    ) == terminal
    outcome = terminal["outcome"]
    assert outcome["purpose5_active_signer_count"] == 3
    assert outcome["purpose5_trust_genesis_pinned"] is True
    assert outcome["purpose5_current_key_status_accepted"] is True
    assert outcome["global_active_transition_performed"] is False
    assert outcome["formal_certificate_signed"] is False
    assert outcome["outside_certificate_issued"] is False
    assert outcome["q5b_action_audit"] == state.SUCCESS_ACTION_AUDIT


def test_receipt_with_self_consistent_but_wrong_outer_digest_is_rejected() -> None:
    _, _, capsule, _, _, _ = _initial_state()
    q5a_artifact = _q5a_payloads()["artifact"]
    preflight = capsule["preflight_receipt"]
    wrong = wire.build_q5b_trust_amendment_v1(
        q5a_artifact,
        q5a_capsule_digest=b"\xff" * 32,
        q5a_authorization_digest=bytes.fromhex(preflight["q5a_authorization_digest"][7:]),
        q5a_consumption_record_digest=bytes.fromhex(preflight["q5a_consumption_record_digest"][7:]),
        q5a_terminal_record_digest=bytes.fromhex(preflight["q5a_terminal_record_digest"][7:]),
        q5a_source_commit=preflight["q5a_source_commit"],
        q5a_source_closure_root=bytes.fromhex(preflight["q5a_source_closure_root"][7:]),
        q5b_source_commit=capsule["source_commit"],
        q5b_source_closure_root=bytes.fromhex(capsule["source_closure_root"][7:]),
    )
    with pytest.raises(state.Q5bCompletionReplayError, match="artifact replay"):
        state.validate_artifact_binding(wrong.canonical_bytes, q5a_artifact, capsule)


def test_invalidated_terminal_keeps_pin_and_global_active_false() -> None:
    contract, prereg, capsule, authorization, consumption, _ = _initial_state()
    terminal = state.build_invalidated_terminal(
        contract,
        prereg,
        capsule,
        authorization,
        consumption,
        failure_phase="ARTIFACT_PUBLICATION",
        failure_code="NO_VALID_ARTIFACT",
        failure_detail="synthetic",
    )
    outcome = terminal["outcome"]
    assert outcome["purpose5_active_signer_count"] == 0
    assert outcome["purpose5_trust_genesis_pinned"] is False
    assert outcome["purpose5_current_key_status_accepted"] is False
    assert outcome["global_active_transition_performed"] is False
    assert outcome["terminal_no_resume"] is True


def test_execution_policy_has_no_runtime_or_private_capability() -> None:
    assert state.EXECUTION_POLICY["maximum_internal_popen_count"] == 0
    assert state.EXECUTION_POLICY["maximum_worker_popen_count"] == 0
    assert state.EXECUTION_POLICY["maximum_worker_openssl_popen_count"] == 0
    assert state.EXECUTION_POLICY["maximum_custody_root_access_count"] == 0
    assert state.EXECUTION_POLICY["maximum_private_key_access_count"] == 0
    assert state.EXECUTION_POLICY["maximum_signature_count"] == 0
    assert state.EXECUTION_POLICY["maximum_certificate_issuance_count"] == 0
    assert state.EXECUTION_POLICY["global_active_transition_allowed"] is False
