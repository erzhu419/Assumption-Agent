from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine import (  # noqa: E402
    phase3_q6_formal_quotient_certificate_issuance_v1 as q6,
)
from hegel_machine.strict_cbor_v1 import (  # noqa: E402
    canonical_cbor_encode,
    content_hash,
    rfc6962_root,
)


def _h(value: int) -> bytes:
    return bytes((value,)) * 32


def _synthetic_q4_artifact() -> bytes:
    body = (
        1,
        q6.Q4_BODY_TAG,
        q6.Q4_BODY_SCHEMA_ID,
        q6.Q4_CLAIM_KIND,
        q6.Q4_DSL_VERSION,
        q6.Q4_CLOSURE_SEMANTICS_VERSION,
        1,
        _h(1),
        _h(2),
        q6.Q4_EQUIVALENCE_ID,
        _h(3),
        _h(4),
        _h(5),
        _h(6),
        _h(7),
        _h(8),
        _h(9),
        _h(10),
        _h(11),
        _h(12),
        418,
        0,
        rfc6962_root(()),
        (2, _h(13), _h(14), _h(15), 89, 1, (_h(16),)),
        _h(17),
        _h(18),
        _h(19),
        _h(20),
        _h(21),
        (1, bytes.fromhex("11" * 20)),
        _h(22),
        (1, bytes.fromhex("22" * 20)),
        _h(23),
        q6.Q4_CLAIM_LEVEL,
    )
    body_hash = content_hash(q6.Q4_BODY_ROOT_DOMAIN, body)
    receipt = (
        1,
        q6.Q4_RECEIPT_TAG,
        q6.Q4_RECEIPT_SCHEMA_ID,
        body,
        body_hash,
        body[28],
        q6.PURPOSE5_FINAL_CERTIFICATE_SIGNER,
        q6.Q4_MISSING_PREREQUISITES,
        0,
        3,
        0,
        3,
        q6.Q4_CAPABILITY_DISPOSITION,
        q6.Q4_SUCCESS_STATUS,
        True,
        False,
        False,
        False,
        False,
        False,
    )
    return canonical_cbor_encode(receipt)


def _manifest(ordinal: int, public_key: bytes) -> tuple[object, ...]:
    key_id = sha256(public_key).digest()[:16]
    return (
        1,
        q6.Q5A_SIGNER_MANIFEST_TAG,
        q6.Q5A_SIGNER_MANIFEST_SCHEMA_ID,
        q6.PURPOSE5_FINAL_CERTIFICATE_SIGNER,
        ordinal,
        q6.SIGNER_ROLE_NAMES[ordinal - 1].encode("ascii"),
        q6.INITIAL_KEY_EPOCH,
        key_id,
        public_key,
    )


def _synthetic_q5b_artifact(q4_payload: bytes) -> bytes:
    q4 = q6.decode_sealed_q4_capability_artifact_v1(q4_payload)
    # These synthetic public bytes deliberately yield key-id order 1,3,2.
    manifests = (
        _manifest(1, bytes((7,)) * 32),
        _manifest(2, bytes((2,)) * 32),
        _manifest(3, bytes((3,)) * 32),
    )
    manifest_roots = tuple(
        content_hash(q6.Q5A_SIGNER_MANIFEST_ROOT_DOMAIN, item)
        for item in manifests
    )
    key_ids = tuple(item[7] for item in manifests)
    unpinned = (
        1,
        q6.Q5A_TRUST_GENESIS_TAG,
        q6.Q5A_TRUST_GENESIS_SCHEMA_ID,
        5,
        1,
        manifests,
        manifest_roots,
        3,
        3,
        q6.Q6_SIGNATURE_PREIMAGE_POLICY_ID,
        True,
        False,
    )
    unpinned_root = content_hash(q6.Q5A_TRUST_GENESIS_ROOT_DOMAIN, unpinned)
    genesis_status = (
        1,
        q6.Q5A_KEY_STATUS_TAG,
        q6.Q5A_KEY_STATUS_SCHEMA_ID,
        unpinned,
        unpinned_root,
        1,
        manifest_roots,
        (),
        (),
        q6.Q6_SIGNATURE_PREIMAGE_POLICY_ID,
        False,
        0,
        0,
    )
    genesis_status_root = content_hash(
        q6.Q5A_KEY_STATUS_ROOT_DOMAIN, genesis_status
    )
    candidate = (
        1,
        q6.Q5A_CANDIDATE_TAG,
        q6.Q5A_CANDIDATE_SCHEMA_ID,
        _h(30),
        _h(31),
        _h(32),
        _h(33),
        q4.receipt_root,
        q4.body.body_hash,
        (1, bytes.fromhex("22" * 20)),
        _h(23),
        (1, bytes.fromhex("33" * 20)),
        _h(34),
        manifests,
        unpinned,
        unpinned_root,
        genesis_status,
        genesis_status_root,
        q6.Q6_SIGNATURE_PREIMAGE_POLICY_ID,
        True,
        3,
        0,
        0,
        q6.Q5A_ACTION_ID,
        q6.Q5A_SUCCESS_STATUS,
        q6.Q5A_CLAIM_LEVEL,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    )
    candidate_bytes = canonical_cbor_encode(candidate)
    candidate_root = content_hash(q6.Q5A_CANDIDATE_ROOT_DOMAIN, candidate)
    pinned = (
        1,
        q6.Q5B_PINNED_GENESIS_TAG,
        q6.Q5B_PINNED_GENESIS_SCHEMA_ID,
        5,
        1,
        candidate_root,
        unpinned_root,
        manifests,
        manifest_roots,
        key_ids,
        3,
        3,
        2,
        3,
        q6.Q6_SIGNATURE_PREIMAGE_POLICY_ID,
        True,
        True,
    )
    pinned_root = content_hash(q6.Q5B_PINNED_GENESIS_ROOT_DOMAIN, pinned)
    current = (
        1,
        q6.Q5B_CURRENT_STATUS_TAG,
        q6.Q5B_CURRENT_STATUS_SCHEMA_ID,
        5,
        pinned,
        pinned_root,
        1,
        manifest_roots,
        key_ids,
        (1, 2, 3),
        3,
        2,
        3,
        q6.Q6_SIGNATURE_PREIMAGE_POLICY_ID,
        True,
        True,
        0,
        0,
        0,
    )
    current_root = content_hash(q6.Q5B_CURRENT_STATUS_ROOT_DOMAIN, current)
    q5b = (
        1,
        q6.Q5B_RECEIPT_TAG,
        q6.Q5B_RECEIPT_SCHEMA_ID,
        _h(35),
        _h(36),
        _h(37),
        _h(38),
        sha256(candidate_bytes).digest(),
        candidate,
        candidate_root,
        unpinned_root,
        genesis_status_root,
        manifest_roots,
        key_ids,
        (1, bytes.fromhex("33" * 20)),
        _h(34),
        (1, bytes.fromhex("44" * 20)),
        _h(39),
        pinned,
        pinned_root,
        current,
        current_root,
        5,
        1,
        3,
        2,
        3,
        q6.Q6_SIGNATURE_PREIMAGE_POLICY_ID,
        True,
        q6.Q5B_ACTION_ID,
        q6.Q5B_SUCCESS_STATUS,
        q6.Q5B_CLAIM_LEVEL,
        True,
        True,
        True,
        0,
        0,
        0,
        False,
        False,
        False,
        False,
        False,
    )
    return canonical_cbor_encode(q5b)


def _predecessors() -> tuple[
    bytes,
    bytes,
    q6.SealedQ4CapabilityArtifactV1,
    q6.SealedQ5bTrustAmendmentArtifactV1,
    q6.Q6SignaturePreimageV1,
]:
    q4_payload = _synthetic_q4_artifact()
    q5b_payload = _synthetic_q5b_artifact(q4_payload)
    q4 = q6.decode_sealed_q4_capability_artifact_v1(q4_payload)
    q5b = q6.decode_sealed_q5b_trust_amendment_artifact_v1(q5b_payload)
    preimage = q6.build_q6_signature_preimage_v1(
        q4, q5b, created_at_unix_seconds=1_788_300_123
    )
    return q4_payload, q5b_payload, q4, q5b, preimage


def test_sealed_predecessors_restore_full_body_and_pinned_status() -> None:
    _, _, q4, q5b, preimage = _predecessors()
    assert len(q4.body.canonical_object) == 34
    assert q5b.q5a_candidate.q4_body_hash == q4.body.body_hash
    assert q5b.q5a_candidate.q4_capability_receipt_root == q4.receipt_root
    assert tuple(item.signer_ordinal for item in q5b.signer_manifests) == (1, 2, 3)
    assert q5b.current_status.canonical_object[6] == 1
    assert preimage.canonical_object()[3] == q6.Q6_SIGNATURE_PREIMAGE_POLICY_ID
    assert preimage.canonical_object()[4:7] == (2, 1_788_300_123, 1)
    assert preimage.canonical_object()[7] == q4.body.canonical_object
    assert q6.decode_q6_signature_preimage_v1(
        preimage.canonical_bytes, q4_artifact=q4, q5b_artifact=q5b
    ) == preimage


def test_sealed_predecessor_decoders_reject_nested_root_drift() -> None:
    q4_payload = _synthetic_q4_artifact()
    q4_value = list(q6.canonical_cbor_decode(q4_payload))
    q4_value[4] = _h(99)
    with pytest.raises(q6.Q6FormalIssuanceWireError, match="body binding"):
        q6.decode_sealed_q4_capability_artifact_v1(
            canonical_cbor_encode(q4_value)
        )

    q5b_payload = _synthetic_q5b_artifact(q4_payload)
    q5b_value = list(q6.canonical_cbor_decode(q5b_payload))
    q5b_value[21] = _h(98)
    with pytest.raises(q6.Q6FormalIssuanceWireError, match="object graph"):
        q6.decode_sealed_q5b_trust_amendment_artifact_v1(
            canonical_cbor_encode(q5b_value)
        )


def test_rfc8032_ed25519_verifier_and_strict_rejections() -> None:
    public_key = bytes.fromhex(
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
    )
    signature = bytes.fromhex(
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
        "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
    )
    q6.verify_ed25519_v1(public_key, signature, b"")

    changed = bytearray(signature)
    changed[0] ^= 1
    with pytest.raises(q6.Q6FormalIssuanceWireError, match="REJECT_Q6_SIGNATURE"):
        q6.verify_ed25519_v1(public_key, bytes(changed), b"")

    noncanonical_s = signature[:32] + q6._ED_L.to_bytes(32, "little")
    with pytest.raises(q6.Q6FormalIssuanceWireError, match="noncanonical"):
        q6.verify_ed25519_v1(public_key, noncanonical_s, b"")


def test_role_order_actions_become_key_sorted_q4_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, q5b, preimage = _predecessors()
    calls: list[tuple[bytes, bytes, bytes]] = []
    monkeypatch.setattr(
        q6,
        "verify_ed25519_v1",
        lambda public, signature, message: calls.append((public, signature, message)),
    )
    role_signatures = (bytes((1,)) * 64, bytes((2,)) * 64, bytes((3,)) * 64)
    envelope = q6.build_q4_compatible_envelope_v1(
        preimage, q5b, role_order_signatures=role_signatures
    )

    assert calls == []
    assert [record.signature[0] for record in envelope.signatures] == [1, 3, 2]
    assert tuple(record.key_id for record in envelope.signatures) == tuple(
        sorted(q5b.role_order_key_ids)
    )
    assert len(envelope.canonical_object()) == 7
    assert q6.decode_q4_compatible_envelope_v1(
        envelope.canonical_bytes,
        preimage=preimage,
        q5b_artifact=q5b,
    ) == envelope
    assert calls == []


def test_envelope_rejects_wrong_epoch_and_active_key_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, q5b, preimage = _predecessors()
    monkeypatch.setattr(q6, "verify_ed25519_v1", lambda *_args: None)
    envelope = q6.build_q4_compatible_envelope_v1(
        preimage,
        q5b,
        role_order_signatures=(bytes((1,)) * 64, bytes((2,)) * 64, bytes((3,)) * 64),
    )
    wrong_epoch = list(envelope.canonical_object())
    wrong_epoch[3] = 2
    with pytest.raises(q6.Q6FormalIssuanceWireError, match="envelope wire"):
        q6.decode_q4_compatible_envelope_v1(
            canonical_cbor_encode(wrong_epoch),
            preimage=preimage,
            q5b_artifact=q5b,
        )

    wrong_set = list(envelope.canonical_object())
    records = [list(item) for item in wrong_set[6]]
    records[0][0] = bytes((0,)) * 16
    records.sort(key=lambda item: item[0])
    wrong_set[6] = tuple(tuple(item) for item in records)
    with pytest.raises(q6.Q6FormalIssuanceWireError, match="active set"):
        q6.decode_q4_compatible_envelope_v1(
            canonical_cbor_encode(wrong_set),
            preimage=preimage,
            q5b_artifact=q5b,
        )


def test_aggregate_issuance_receipt_round_trip_and_success_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q4_payload, q5b_payload, _, q5b, _ = _predecessors()
    verification_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        q6,
        "verify_ed25519_v1",
        lambda *args: verification_calls.append(args),
    )
    receipt = q6.build_q6_formal_issuance_receipt_v1(
        q4_payload,
        q5b_payload,
        q5b_capsule_digest=_h(40),
        q5b_authorization_digest=_h(41),
        q5b_consumption_record_digest=_h(42),
        q5b_terminal_record_digest=_h(43),
        created_at_unix_seconds=1_788_300_123,
        role_order_signatures=(bytes((1,)) * 64, bytes((2,)) * 64, bytes((3,)) * 64),
        q6_source_commit="55" * 20,
        q6_source_closure_root=_h(44),
        complete_checkpoint_digest=_h(45),
    )
    replayed = q6.decode_q6_formal_issuance_receipt_v1(receipt.canonical_bytes)
    assert verification_calls == []
    assert replayed == receipt
    assert replayed.formal_certificate_signed is True
    assert replayed.certificate_envelope_published is True
    assert replayed.outside_certificate_issued is True
    assert replayed.external_authority is False
    assert replayed.global_active_transition_performed is False
    assert replayed.synthesis_released is False
    assert replayed.complete_checkpoint_digest == _h(45)
    assert replayed.canonical_object()[20] == q5b.role_order_key_ids
    assert replayed.canonical_object()[21] == tuple(sorted(q5b.role_order_key_ids))
    assert replayed.canonical_object()[25] == q6.Q6_SUCCESS_STATUS.encode("ascii")

    wrong_checkpoint = list(replayed.canonical_object())
    wrong_checkpoint[19] = bytes((1,)) * 31
    with pytest.raises(q6.Q6FormalIssuanceWireError, match="field 19"):
        q6.decode_q6_formal_issuance_receipt_v1(
            canonical_cbor_encode(wrong_checkpoint)
        )
    assert verification_calls == []


def test_wire_module_has_no_q4_q5_producer_or_process_import() -> None:
    source = (
        ROOT
        / "src/hegel_machine/phase3_q6_formal_quotient_certificate_issuance_v1.py"
    ).read_text(encoding="utf-8")
    assert "phase3_q4_formal_quotient_certificate_capability_v1" not in source
    assert "phase3_q5a_purpose5_signer_genesis_v1" not in source
    assert "phase3_q5b_purpose5_trust_amendment_v1" not in source
    assert "import subprocess" not in source
    assert "import openssl" not in source.lower()
