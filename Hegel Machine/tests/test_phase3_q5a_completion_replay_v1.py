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

from hegel_machine import phase3_q5a_completion_replay_v1 as state  # noqa: E402
from hegel_machine import phase3_q5a_purpose5_signer_genesis_v1 as wire  # noqa: E402


Q4_PREDECESSOR = {
    "source_commit": "2b33dc5e4145962e93af8722afaa55445c8a63c5",
    "source_closure_root": (
        "sha256:583c337cf153df0ca18daba5cb79dbbf6c0969acbd27ccb5d1c3accd5dc3d602"
    ),
    "capsule": {
        "raw_sha256": (
            "sha256:49b7e14baf9b22be6d5715edd28019f54a8361a775cc548be057fd49b6ba39fd"
        ),
        "capsule_digest": (
            "sha256:dbb3e5933647d5db567c35a5ea32aeb9bbd2a535d8c2a30adedd4d4cbc2308c6"
        ),
    },
    "authorization": {
        "raw_sha256": (
            "sha256:fa34201ebe3352f917e824cd4805a05c7ad3a2f42f214d45fad374c335c87c60"
        ),
        "authorization_digest": (
            "sha256:c4363e907f32f5aab7f66bcd657af14aba75c1ce5358dbd1e7c227d32aa3a087"
        ),
    },
    "consumption": {
        "raw_sha256": (
            "sha256:35dff653f93201e59af3038ff44c0d4642bf298f05503c9749c3a6625876286a"
        ),
        "record_digest": (
            "sha256:7f5c53d1e83c973ca160d979e6883d541abee55686dcfd0cf4f0191309cb285f"
        ),
    },
    "terminal": {
        "raw_sha256": (
            "sha256:105d560b6f7a351e80df81b08b427d8f9cbf68e58f85d0f84e20f6c6d7320c2e"
        ),
        "record_digest": (
            "sha256:d5770333031378eb611be134312fb7b26a77e92b7641636ca215c6ac84ea507c"
        ),
    },
    "artifact": {
        "raw_sha256": (
            "sha256:8c04217f9ffa8619c9a827dff7af84791318fee61432217e4efc929419d9c00d"
        ),
        "capability_receipt_root": (
            "sha256:f45a3d0d37df747683d7de5c72f279a86fd7fc4220f3a782bb33ee0d36c9f0c8"
        ),
        "quotient_body_hash": (
            "sha256:873b10cb7ac9b6eec2c00543ccc755c86e2c20c7180101842b74a2d6856c03de"
        ),
    },
    "trajectory": {
        "raw_sha256": (
            "sha256:c0cba88fec4951d281b1068b7acf02a1c412b8c648d2c716ef440bcdc1591d36"
        ),
        "event_count": 3,
    },
}


def _q4_payloads() -> dict[str, bytes]:
    return {
        "capsule": (
            STATE_ROOT
            / "q4-formal-quotient-certificate-capability-v1-run/capsule.json"
        ).read_bytes(),
        "authorization": (
            STATE_ROOT
            / "q4-formal-quotient-certificate-capability-v1-ledger/owner.authorization.json"
        ).read_bytes(),
        "consumption": (
            STATE_ROOT
            / "q4-formal-quotient-certificate-capability-v1-ledger/campaign.consumed.json"
        ).read_bytes(),
        "terminal": (
            STATE_ROOT
            / "q4-formal-quotient-certificate-capability-v1-ledger/campaign.terminal.json"
        ).read_bytes(),
        "artifact": (
            ROOT / "artifacts/phase3_q4_formal_quotient_certificate_capability_v1.cbor"
        ).read_bytes(),
        "trajectory": (
            STATE_ROOT
            / "q4-formal-quotient-certificate-capability-v1-run/trajectory.jsonl"
        ).read_bytes(),
    }


def _preregistration() -> dict[str, object]:
    return {
        "schema_version": state.PREREGISTRATION_SCHEMA,
        "preregistration_id": state.PREREGISTRATION_ID,
        "status": "Q5A_SPEC_FROZEN_NOT_AUTHORIZED",
        "predecessor_q4": deepcopy(Q4_PREDECESSOR),
    }


def _replay(
    preregistration: dict[str, object] | None = None,
    payloads: dict[str, bytes] | None = None,
) -> state.Q4SealedReplayV1:
    selected = _q4_payloads() if payloads is None else payloads
    return state.replay_q4_predecessor_v1(
        _preregistration() if preregistration is None else preregistration,
        capsule_payload=selected["capsule"],
        authorization_payload=selected["authorization"],
        consumption_payload=selected["consumption"],
        terminal_payload=selected["terminal"],
        artifact_payload=selected["artifact"],
        trajectory_payload=selected["trajectory"],
    )


def _smoke(ordinal: int) -> dict[str, object]:
    return {
        "expected_gid": 1000,
        "expected_uid": 1000,
        "key_generation_count": 0,
        "openssl_path": "/usr/bin/openssl",
        "openssl_popen_count": 0,
        "public_key_derivation_count": 0,
        "purpose_id": 5,
        "schema": "hegel-phase3-q5a-purpose5-keygen-startup-smoke-receipt/1",
        "signature_count": 0,
        "signer_ordinal": ordinal,
        "signer_role": state.SIGNER_ROLE_NAMES[ordinal - 1],
        "status": "Q5A_PURPOSE5_KEYGEN_STARTUP_SMOKE_PASS",
    }


def _contract() -> state.LoadedJson:
    prereg_raw = state.canonical_json_bytes(_preregistration())
    value = {
        "schema_version": state.CONTRACT_SCHEMA,
        "contract_id": state.CONTRACT_ID,
        "action_id": state.ACTION_ID,
        "preregistration_raw_sha256": state.raw_sha256(prereg_raw),
        "execution_policy": deepcopy(state.EXECUTION_POLICY),
        "authority_boundary": deepcopy(state.AUTHORITY_BOUNDARY),
    }
    raw = state.canonical_json_bytes(value)
    return state.LoadedJson(Path("q5a-test-contract.json"), raw, state.raw_sha256(raw), value)


def _source_rows() -> list[dict[str, object]]:
    return [
        {
            "relative_path": relative,
            "byte_count": len(relative.encode("ascii")),
            "raw_sha256": state.raw_sha256(relative.encode("ascii")),
        }
        for relative in state.Q5_SOURCE_PATHS
    ]


def _capsule() -> tuple[state.LoadedJson, dict[str, object], dict[str, object]]:
    contract = _contract()
    prereg = _preregistration()
    q4 = _replay()
    preflight = state.build_preflight_receipt(
        q4.preflight_receipt,
        [_smoke(1), _smoke(2), _smoke(3)],
        fresh_root_admission_completed=True,
    )
    rows = _source_rows()
    capsule = state.build_capsule(
        contract,
        prereg,
        {
            "source_commit": "a" * 40,
            "source_closure_rows": rows,
            "source_closure_root": state.source_closure_root(rows),
            "run_id": "b" * 32,
            "created_at_unix_seconds": 1_788_400_000,
            "preflight_receipt": preflight,
        },
    )
    return contract, prereg, capsule


def _initial_state():
    contract, prereg, capsule = _capsule()
    authorization = state.build_authorization(contract, prereg, capsule)
    consumption = state.build_consumption(
        contract, prereg, capsule, authorization
    )
    return contract, prereg, capsule, authorization, consumption


def _worker_receipt(ordinal: int, public_key: bytes) -> dict[str, object]:
    return {
        "key_epoch": 1,
        "key_generation_count": 1,
        "key_id_hex": state.sha256(public_key).digest()[:16].hex(),
        "openssl_popen_count": 2,
        "private_key_exported": False,
        "public_key_derivation_count": 1,
        "public_key_hex": public_key.hex(),
        "purpose_id": 5,
        "schema": state.WORKER_RECEIPT_SCHEMA,
        "signature_count": 0,
        "signer_ordinal": ordinal,
        "signer_role": state.SIGNER_ROLE_NAMES[ordinal - 1],
        "status": state.WORKER_SUCCESS_STATUS,
    }


def _completed_receipts(count: int):
    contract, prereg, capsule, authorization, consumption = _initial_state()
    receipts: list[dict[str, object]] = []
    intents: list[dict[str, object]] = []
    for ordinal in range(1, count + 1):
        intent = state.build_keygen_intent(
            contract,
            prereg,
            capsule,
            authorization,
            consumption,
            receipts,
            created_at_unix_seconds=1_788_400_000 + ordinal,
        )
        public_key = bytes([ordinal]) * 32
        worker = _worker_receipt(ordinal, public_key)
        receipt = state.build_public_receipt(
            contract,
            prereg,
            capsule,
            authorization,
            consumption,
            receipts,
            intent,
            worker_receipt=worker,
            worker_public_receipt_raw_sha256=state.raw_sha256(
                state.canonical_json_bytes(worker)
            ),
            public_key_der_raw_sha256=state.raw_sha256(
                state.ED25519_SPKI_DER_PREFIX + public_key
            ),
        )
        intents.append(intent)
        receipts.append(receipt)
    return (
        contract,
        prereg,
        capsule,
        authorization,
        consumption,
        intents,
        receipts,
    )


def test_actual_q4_authority_and_trajectory_replay_is_exact() -> None:
    replay = _replay()
    receipt = replay.preflight_receipt
    assert receipt["selected_q4_authority_input_count"] == 5
    assert receipt["selected_q4_authority_input_total_bytes"] == 11_308
    assert receipt["q4_trajectory_audit"] == {
        "raw_sha256": Q4_PREDECESSOR["trajectory"]["raw_sha256"],
        "byte_count": 3_036,
        "event_count": 3,
        "events": [
            "CAPSULE_PREPARED_NOT_AUTHORIZED",
            "AUTHORIZATION_RECORDED_NOT_CONSUMED",
            "Q4_CAPABILITY_ADMISSION_COMPLETE_NOT_ISSUED",
        ],
        "all_action_audits_zero": True,
    }
    assert receipt["q4_quotient_body_hash"] == Q4_PREDECESSOR["artifact"]["quotient_body_hash"]
    assert receipt["q5a_action_audit"] == state.ZERO_ACTION_AUDIT


@pytest.mark.parametrize(
    "kind", ["capsule", "authorization", "consumption", "terminal", "artifact", "trajectory"]
)
def test_each_q4_input_raw_drift_is_rejected(kind: str) -> None:
    payloads = _q4_payloads()
    payloads[kind] = bytes((payloads[kind][0] ^ 1,)) + payloads[kind][1:]
    with pytest.raises(state.Q5aCompletionReplayError, match="raw SHA-256"):
        _replay(payloads=payloads)


def test_trajectory_shape_is_replayed_after_raw_admission() -> None:
    payloads = _q4_payloads()
    rows = [json.loads(line) for line in payloads["trajectory"].splitlines()]
    rows[2]["event"] = "WRONG_EVENT"
    payloads["trajectory"] = b"".join(state.canonical_json_bytes(row) for row in rows)
    prereg = _preregistration()
    prereg["predecessor_q4"]["trajectory"]["raw_sha256"] = state.raw_sha256(
        payloads["trajectory"]
    )
    with pytest.raises(state.Q5aCompletionReplayError, match="trajectory event replay"):
        _replay(prereg, payloads)


def test_capsule_preflight_binds_three_zero_action_startup_smokes() -> None:
    _, _, capsule = _capsule()
    preflight = capsule["preflight_receipt"]
    assert preflight["worker_startup_smoke_popen_count"] == 3
    assert preflight["preconsumption_key_generation_count"] == 0
    assert preflight["preconsumption_worker_openssl_popen_count"] == 0
    assert preflight["fresh_root_admission_completed"] is True
    assert [row["signer_ordinal"] for row in preflight["worker_startup_smoke_receipts"]] == [1, 2, 3]


def test_wrong_startup_smoke_role_is_rejected() -> None:
    q4 = _replay()
    smokes = [_smoke(1), _smoke(2), _smoke(3)]
    smokes[1]["signer_role"] = smokes[0]["signer_role"]
    with pytest.raises(state.Q5aCompletionReplayError, match="startup-smoke"):
        state.build_preflight_receipt(
            q4.preflight_receipt, smokes, fresh_root_admission_completed=True
        )


def test_capsule_preflight_must_match_frozen_q4_preregistration() -> None:
    contract = _contract()
    prereg = _preregistration()
    q4 = _replay()
    forged = deepcopy(q4.preflight_receipt)
    forged["q4_capsule_digest"] = "sha256:" + "00" * 32
    preflight = state.build_preflight_receipt(
        forged,
        [_smoke(1), _smoke(2), _smoke(3)],
        fresh_root_admission_completed=True,
    )
    rows = _source_rows()
    with pytest.raises(state.Q5aCompletionReplayError, match="preregistration"):
        state.build_capsule(
            contract,
            prereg,
            {
                "source_commit": "a" * 40,
                "source_closure_rows": rows,
                "source_closure_root": state.source_closure_root(rows),
                "run_id": "b" * 32,
                "created_at_unix_seconds": 1,
                "preflight_receipt": preflight,
            },
        )


def test_source_closure_is_exact_and_excludes_legacy_certificate() -> None:
    rows = _source_rows()
    assert state.validate_source_closure_rows(rows) == rows
    assert "src/hegel_machine/phase3_certificate_v1.py" not in state.Q5_SOURCE_PATHS
    assert "src/hegel_machine/phase3_q4_formal_quotient_certificate_capability_v1.py" not in state.Q5_SOURCE_PATHS


def test_authorization_and_consumption_are_candidate_only() -> None:
    contract, prereg, capsule, authorization, consumption = _initial_state()
    assert state.validate_authorization(authorization, contract, prereg, capsule) == authorization
    assert state.validate_consumption(
        consumption, contract, prereg, capsule, authorization
    ) == consumption
    assert authorization["key_generation_limit"] == 3
    assert authorization["worker_openssl_popen_limit"] == 6
    assert authorization["certificate_signing_allowed"] is False
    assert capsule["authority_boundary"]["q5a_disposition"] == (
        "GENESIS_CANDIDATE_ONLY_NOT_PINNED_NOT_SIGNED"
    )
    assert consumption["completed_prefix"] == []
    assert consumption["q5a_action_audit"] == state.ZERO_ACTION_AUDIT


def test_preconsumption_pending_is_zero_action_and_not_consumed() -> None:
    contract, prereg, capsule = _capsule()
    authorization = state.build_authorization(contract, prereg, capsule)
    pending = state.build_pending(
        contract,
        prereg,
        capsule,
        authorization=authorization,
        phase="LOCAL_DEPENDENCY",
        condition="WORKER_UNAVAILABLE",
        detail="test",
    )
    assert pending["pending_kind"] == "PRECONSUMPTION"
    assert pending["authorization_consumption_count"] == 0
    assert pending["logical_campaign_count"] == 0
    assert pending["completed_prefix"] == []
    assert pending["q5a_action_audit"] == state.ZERO_ACTION_AUDIT
    assert state.validate_pending(
        pending,
        contract,
        prereg,
        capsule,
        authorization=authorization,
    ) == pending


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_postconsumption_pending_accepts_clean_public_result_prefixes(count: int) -> None:
    contract, prereg, capsule, authorization, consumption, _, receipts = _completed_receipts(count)
    pending = state.build_pending(
        contract,
        prereg,
        capsule,
        authorization=authorization,
        consumption=consumption,
        completed_receipts=receipts,
        phase="NEXT_ROOT_ADMISSION",
        condition="ROOT_TEMPORARILY_UNAVAILABLE",
        detail="test",
    )
    assert pending["pending_kind"] == "POSTCONSUMPTION_CLEAN_CHECKPOINT"
    assert pending["completed_prefix"] == list(range(1, count + 1))
    assert pending["q5a_action_audit"] == state.q5a_action_audit(tuple(range(1, count + 1)))
    assert state.validate_pending(
        pending,
        contract,
        prereg,
        capsule,
        authorization=authorization,
        consumption=consumption,
        completed_receipts=receipts,
    ) == pending


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_clean_checkpoint_replays_exact_completed_prefix(count: int) -> None:
    contract, prereg, capsule, authorization, consumption, _, receipts = _completed_receipts(count)
    checkpoint = state.build_checkpoint(
        contract,
        prereg,
        capsule,
        authorization,
        consumption,
        receipts,
    )
    assert checkpoint["completed_prefix"] == list(range(1, count + 1))
    assert checkpoint["q5a_action_audit"] == state.q5a_action_audit(
        tuple(range(1, count + 1))
    )
    assert state.validate_checkpoint(
        checkpoint,
        contract,
        prereg,
        capsule,
        authorization,
        consumption,
        receipts,
    ) == checkpoint


def test_checkpoint_cannot_claim_a_completed_signer_without_its_receipt() -> None:
    contract, prereg, capsule, authorization, consumption, _, receipts = _completed_receipts(1)
    checkpoint = state.build_checkpoint(
        contract,
        prereg,
        capsule,
        authorization,
        consumption,
        receipts,
    )
    checkpoint["completed_prefix"].append(2)
    with pytest.raises(state.Q5aCompletionReplayError, match="checkpoint replay"):
        state.validate_checkpoint(
            checkpoint,
            contract,
            prereg,
            capsule,
            authorization,
            consumption,
            receipts,
        )


def test_full_prefix_pending_preserves_all_completed_action_counts() -> None:
    contract, prereg, capsule, authorization, consumption, _, receipts = _completed_receipts(3)
    pending = state.build_pending(
        contract,
        prereg,
        capsule,
        authorization=authorization,
        consumption=consumption,
        completed_receipts=receipts,
        phase="PRE_ARTIFACT_FINALIZATION",
        condition="PUBLICATION_TEMPORARILY_UNAVAILABLE",
        detail="test",
    )
    assert pending["completed_prefix"] == [1, 2, 3]
    assert pending["q5a_action_audit"] == state.SUCCESS_ACTION_AUDIT


def test_keygen_intents_and_public_receipts_form_one_strict_prefix() -> None:
    contract, _, capsule, authorization, consumption, intents, receipts = _completed_receipts(3)
    assert [row["signer_ordinal"] for row in intents] == [1, 2, 3]
    assert [row["signer_ordinal"] for row in receipts] == [1, 2, 3]
    assert state.validate_public_receipt_sequence(
        receipts, contract, capsule, authorization, consumption
    ) == receipts
    assert receipts[-1]["q5a_action_audit"] == state.SUCCESS_ACTION_AUDIT
    assert receipts[-1]["q5a_action_audit"]["worker_openssl_popen_count"] == 6
    assert receipts[-1]["q5a_action_audit"]["public_key_derivation_count"] == 3


def test_worker_key_id_and_public_der_bindings_are_recomputed() -> None:
    contract, prereg, capsule, authorization, consumption = _initial_state()
    intent = state.build_keygen_intent(
        contract, prereg, capsule, authorization, consumption, [],
        created_at_unix_seconds=1,
    )
    public_key = b"x" * 32
    worker = _worker_receipt(1, public_key)
    worker["key_id_hex"] = "00" * 16
    with pytest.raises(state.Q5aCompletionReplayError, match="worker public receipt"):
        state.build_public_receipt(
            contract, prereg, capsule, authorization, consumption, [], intent,
            worker_receipt=worker,
            worker_public_receipt_raw_sha256=state.raw_sha256(state.canonical_json_bytes(worker)),
            public_key_der_raw_sha256=state.raw_sha256(state.ED25519_SPKI_DER_PREFIX + public_key),
        )


def test_duplicate_public_keys_are_rejected() -> None:
    contract, prereg, capsule, authorization, consumption, _, receipts = _completed_receipts(1)
    intent = state.build_keygen_intent(
        contract, prereg, capsule, authorization, consumption, receipts,
        created_at_unix_seconds=2,
    )
    public_key = bytes.fromhex(receipts[0]["public_key_hex"])
    worker = _worker_receipt(2, public_key)
    with pytest.raises(state.Q5aCompletionReplayError, match="pairwise distinct"):
        state.build_public_receipt(
            contract, prereg, capsule, authorization, consumption, receipts, intent,
            worker_receipt=worker,
            worker_public_receipt_raw_sha256=state.raw_sha256(state.canonical_json_bytes(worker)),
            public_key_der_raw_sha256=state.raw_sha256(state.ED25519_SPKI_DER_PREFIX + public_key),
        )


def test_intent_without_public_receipt_is_terminal_and_conservatively_counted() -> None:
    contract, prereg, capsule, authorization, consumption, _, receipts = _completed_receipts(1)
    intent = state.build_keygen_intent(
        contract, prereg, capsule, authorization, consumption, receipts,
        created_at_unix_seconds=2,
    )
    terminal = state.build_invalidated_terminal(
        contract,
        prereg,
        capsule,
        authorization,
        consumption,
        receipts,
        failure_phase="SIGNER_2_WORKER",
        failure_code="KEYGEN_OUTCOME_UNKNOWN",
        failure_detail="test",
        unknown_intent=intent,
    )
    assert state.validate_terminal(
        terminal,
        contract,
        prereg,
        capsule,
        authorization,
        consumption,
        receipts,
        unknown_intent=intent,
    ) == terminal
    audit = terminal["outcome"]["q5a_action_audit"]
    assert audit["public_receipt_count"] == 1
    assert audit["key_generation_count"] == 2
    assert audit["worker_openssl_popen_count"] == 4
    assert terminal["outcome"]["terminal_no_resume"] is True


def test_complete_candidate_terminal_is_unpinned_unsigned_and_replayable() -> None:
    contract, prereg, capsule, authorization, consumption, _, receipts = _completed_receipts(3)
    candidate = state.build_candidate_from_receipts(capsule, receipts)
    artifact = candidate.canonical_bytes
    terminal = state.build_complete_terminal(
        contract, prereg, capsule, authorization, consumption, receipts, artifact
    )
    assert state.validate_terminal(
        terminal,
        contract,
        prereg,
        capsule,
        authorization,
        consumption,
        receipts,
        artifact_payload=artifact,
    ) == terminal
    outcome = terminal["outcome"]
    assert terminal["status"] == wire.Q5A_SUCCESS_STATUS
    assert outcome["disposition"] == "UNPINNED_NOT_SIGNED"
    assert outcome["purpose5_candidate_signer_count"] == 3
    assert outcome["purpose5_active_signer_count"] == 0
    assert outcome["purpose5_trust_genesis_pinned"] is False
    assert outcome["purpose5_current_key_status_accepted"] is False
    assert outcome["formal_certificate_signed"] is False
    assert outcome["certificate_envelope_published"] is False
    assert outcome["outside_certificate_issued"] is False
    assert outcome["q5a_action_audit"] == state.SUCCESS_ACTION_AUDIT


def test_artifact_drift_and_terminal_authority_forgery_are_rejected() -> None:
    contract, prereg, capsule, authorization, consumption, _, receipts = _completed_receipts(3)
    artifact = state.build_candidate_from_receipts(capsule, receipts).canonical_bytes
    terminal = state.build_complete_terminal(
        contract, prereg, capsule, authorization, consumption, receipts, artifact
    )
    drifted = bytes((artifact[0] ^ 1,)) + artifact[1:]
    with pytest.raises(state.Q5aCompletionReplayError, match="candidate replay rejected"):
        state.validate_artifact_binding(drifted, capsule, receipts)
    forged = deepcopy(terminal)
    forged["outcome"]["purpose5_trust_genesis_pinned"] = True
    with pytest.raises(state.Q5aCompletionReplayError, match="terminal replay"):
        state.validate_terminal(
            forged,
            contract,
            prereg,
            capsule,
            authorization,
            consumption,
            receipts,
            artifact_payload=artifact,
        )


def test_complete_artifact_to_terminal_partial_finalization_is_deterministic() -> None:
    contract, prereg, capsule, authorization, consumption, _, receipts = _completed_receipts(3)
    artifact = state.build_candidate_from_receipts(capsule, receipts).canonical_bytes
    first = state.build_complete_terminal(
        contract, prereg, capsule, authorization, consumption, receipts, artifact
    )
    second = state.build_complete_terminal(
        contract, prereg, capsule, authorization, consumption, receipts, artifact
    )
    assert first == second
    assert first["record_digest"] == second["record_digest"]
