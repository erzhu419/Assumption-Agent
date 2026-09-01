from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine import phase3_q6_completion_replay_v1 as state  # noqa: E402
from hegel_machine import phase3_q6_formal_quotient_certificate_issuance_v1 as wire  # noqa: E402

CONTRACT_PATH = ROOT / "config/phase3_q6_formal_quotient_certificate_issuance_actual_v1.json"


def _fixture() -> dict[str, object]:
    contract = state.load_contract(CONTRACT_PATH)
    prereg = state.load_preregistration(Path(contract.value["preregistration_path"]))
    q5b = {
        kind: Path(path).read_bytes()
        for kind, path in contract.value["q5b_inputs"].items()
    }
    q4 = Path(contract.value["q4_input"]["artifact"]).read_bytes()
    replay = state.replay_predecessors_v1(
        prereg.value, q5b_payloads=q5b, q4_artifact_payload=q4
    )
    preimage = wire.build_q6_signature_preimage_v1(
        replay.q4_artifact, replay.q5b_artifact,
        created_at_unix_seconds=1_800_000_000,
    )
    smoke = []
    for ordinal in state.SIGNER_ORDINALS:
        smoke.append({
            "expected_gid": 1000,
            "expected_key_id_hex": replay.preflight_receipt["role_ordered_key_ids"][ordinal - 1],
            "expected_message_byte_count": len(preimage.signature_message),
            "expected_message_raw_sha256": state.raw_sha256(preimage.signature_message),
            "expected_public_key_hex": replay.preflight_receipt["role_ordered_public_keys"][ordinal - 1],
            "expected_uid": 1000,
            "key_generation_count": 0,
            "key_import_count": 0,
            "openssl_path": "/usr/bin/openssl",
            "openssl_popen_count": 0,
            "private_key_access_count": 0,
            "purpose_id": 5,
            "schema": state.WORKER_SMOKE_SCHEMA,
            "signature_count": 0,
            "signer_ordinal": ordinal,
            "signer_role": state.SIGNER_ROLE_NAMES[ordinal - 1],
            "status": state.WORKER_SMOKE_STATUS,
        })
    preflight = state.build_preflight_receipt(replay, smoke, preimage)
    rows = []
    for relative in state.Q6_SOURCE_PATHS:
        payload = (ROOT / relative).read_bytes()
        rows.append({
            "relative_path": relative,
            "byte_count": len(payload),
            "raw_sha256": state.raw_sha256(payload),
        })
    capsule = state.build_capsule(
        contract, prereg.value,
        source_commit="a" * 40,
        source_closure_rows=rows,
        run_id="b" * 32,
        created_at_unix_seconds=1_800_000_001,
        preflight_receipt=preflight,
    )
    authorization = state.build_authorization(contract, prereg.value, capsule)
    consumption = state.build_consumption(
        contract, prereg.value, capsule, authorization
    )
    receipts = []
    signatures = []
    intents = []
    verification_intents = []
    for ordinal in state.SIGNER_ORDINALS:
        intent = state.build_signing_intent(
            contract, prereg.value, capsule, authorization, consumption,
            receipts, created_at_unix_seconds=1_800_000_010 + ordinal,
        )
        signature = bytes([ordinal]) * 64
        worker = state.expected_worker_receipt(
            intent=intent, signature_payload=signature
        )
        verification_intent = state.build_verification_intent(
            contract, capsule, authorization, consumption, receipts, intent,
            worker_receipt=worker, signature_payload=signature,
            created_at_unix_seconds=1_800_000_020 + ordinal,
        )
        receipt = state.build_signature_receipt(
            contract, prereg.value, capsule, authorization, consumption,
            receipts, intent, worker_receipt=worker,
            signature_payload=signature,
            verification_intent=verification_intent,
        )
        intents.append(intent)
        verification_intents.append(verification_intent)
        signatures.append(signature)
        receipts.append(receipt)
    checkpoint = state.build_checkpoint(
        contract, capsule, authorization, consumption, receipts
    )
    return locals()


def test_live_predecessor_and_complete_checkpoint_artifact_terminal_are_zero_crypto(monkeypatch):
    data = _fixture()
    calls = 0

    def forbidden(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("partial finalization must not verify crypto")

    monkeypatch.setattr(wire, "verify_ed25519_v1", forbidden)
    receipt = state.build_issuance_artifact(
        data["capsule"], data["receipts"], data["signatures"],
        q4_artifact_payload=data["q4"],
        q5b_artifact_payload=data["q5b"]["artifact"],
        complete_checkpoint_digest=data["checkpoint"]["checkpoint_digest"],
    )
    checked = state.validate_artifact_binding(
        receipt.canonical_bytes, receipt.envelope.canonical_bytes,
        data["capsule"], q4_artifact_payload=data["q4"],
        q5b_artifact_payload=data["q5b"]["artifact"],
        complete_checkpoint_digest=data["checkpoint"]["checkpoint_digest"],
        verified_checkpoints=data["receipts"],
    )
    terminal = state.build_complete_terminal(
        data["contract"], data["capsule"], data["authorization"],
        data["consumption"], receipt.canonical_bytes,
        receipt.envelope.canonical_bytes, checked, data["receipts"],
        complete_checkpoint_digest=data["checkpoint"]["checkpoint_digest"],
        pending_history_receipt={
            "raw_sha256": state.raw_sha256(b""), "byte_count": 0, "event_count": 0
        },
    )
    assert state.validate_terminal(
        terminal, data["contract"], data["capsule"], data["authorization"],
        data["consumption"], artifact_payload=receipt.canonical_bytes,
        envelope_payload=receipt.envelope.canonical_bytes,
        q4_artifact_payload=data["q4"],
        q5b_artifact_payload=data["q5b"]["artifact"],
        verified_checkpoints=data["receipts"],
        complete_checkpoint_digest=data["checkpoint"]["checkpoint_digest"],
        pending_history_receipt={
            "raw_sha256": state.raw_sha256(b""), "byte_count": 0, "event_count": 0
        },
    ) == terminal
    assert calls == 0


def test_tampered_smoke_signature_and_checkpoint_bindings_are_rejected():
    data = _fixture()
    bad_preflight = deepcopy(data["preflight"])
    bad_preflight["startup_smoke_receipts"][0]["signature_count"] = 1
    with pytest.raises(state.Q6CompletionReplayError):
        state.validate_preflight_receipt(bad_preflight)

    with pytest.raises(state.Q6CompletionReplayError):
        state.build_issuance_artifact(
            data["capsule"], data["receipts"],
            [b"x" * 64, *data["signatures"][1:]],
            q4_artifact_payload=data["q4"],
            q5b_artifact_payload=data["q5b"]["artifact"],
            complete_checkpoint_digest=data["checkpoint"]["checkpoint_digest"],
        )

    receipt = state.build_issuance_artifact(
        data["capsule"], data["receipts"], data["signatures"],
        q4_artifact_payload=data["q4"],
        q5b_artifact_payload=data["q5b"]["artifact"],
        complete_checkpoint_digest=data["checkpoint"]["checkpoint_digest"],
    )
    with pytest.raises(state.Q6CompletionReplayError):
        state.validate_artifact_binding(
            receipt.canonical_bytes, receipt.envelope.canonical_bytes,
            data["capsule"], q4_artifact_payload=data["q4"],
            q5b_artifact_payload=data["q5b"]["artifact"],
            complete_checkpoint_digest="sha256:" + "0" * 64,
            verified_checkpoints=data["receipts"],
        )


def test_invalidated_terminal_exactly_replays_observed_conservative_counts():
    data = _fixture()
    audit = state.q6_observed_action_audit(
        (), metadata_admission_count=1, signing_intent_count=1,
        worker_popen_count=1,
        openssl_sign_count=1, private_key_load_count=1,
        signature_output_count=1, worker_receipt_count=1,
        verification_intent_count=0, verification_count=0,
        valid_signature_count=0, verified_checkpoint_count=0,
        verified_checkpoint_publication_count=0,
    )
    terminal = state.build_invalidated_terminal(
        data["contract"], data["capsule"], data["authorization"],
        data["consumption"], (), failure_phase="SIGNER_1",
        failure_code="Q6_SIGNING_WORKER_OUTCOME_UNKNOWN",
        failure_detail="injected", observed_action_audit=audit,
        pending_history_receipt={
            "raw_sha256": state.raw_sha256(b""), "byte_count": 0, "event_count": 0
        },
    )
    assert state.validate_terminal(
        terminal, data["contract"], data["capsule"], data["authorization"],
        data["consumption"], pending_history_receipt={
            "raw_sha256": state.raw_sha256(b""), "byte_count": 0, "event_count": 0
        },
    ) == terminal
    tampered = deepcopy(terminal)
    tampered["outcome"]["q6_action_audit"]["worker_openssl_sign_popen_count"] = 0
    unsigned = deepcopy(tampered)
    unsigned.pop("record_digest")
    tampered["record_digest"] = state.raw_sha256(
        state.DOMAINS["terminal"] + state.canonical_json_bytes(unsigned)
    )
    with pytest.raises(state.Q6CompletionReplayError):
        state.validate_terminal(
            tampered, data["contract"], data["capsule"],
            data["authorization"], data["consumption"],
            pending_history_receipt={
                "raw_sha256": state.raw_sha256(b""), "byte_count": 0,
                "event_count": 0,
            },
        )
