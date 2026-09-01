from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine import phase3_q5a_purpose5_signer_genesis_v1 as q5a
from hegel_machine.strict_cbor_v1 import canonical_cbor_encode, content_hash


Q4_ARTIFACT = (
    ROOT / "artifacts/phase3_q4_formal_quotient_certificate_capability_v1.cbor"
)

# Public-only opaque bytes.  No private key or deterministic private seed exists
# in this test module, and no signing backend is imported.
PUBLIC_KEYS = (
    bytes.fromhex("11" * 32),
    bytes.fromhex("42" * 32),
    bytes.fromhex("e7" * 32),
)


def _candidate() -> q5a.Q5aPurpose5SignerGenesisCandidateV1:
    sealed_q4 = q5a.decode_sealed_q4_capability_receipt_v1(Q4_ARTIFACT.read_bytes())
    return q5a.build_q5a_candidate_v1(
        q4_capsule_digest=bytes.fromhex(
            "dbb3e5933647d5db567c35a5ea32aeb9bbd2a535d8c2a30adedd4d4cbc2308c6"
        ),
        q4_authorization_digest=b"A" * 32,
        q4_consumption_record_digest=b"C" * 32,
        q4_terminal_record_digest=bytes.fromhex(
            "d5770333031378eb611be134312fb7b26a77e92b7641636ca215c6ac84ea507c"
        ),
        q4_capability_receipt_root=sealed_q4.receipt_root,
        q4_body_hash=sealed_q4.body_hash,
        q4_source_commit="2b33dc5e4145962e93af8722afaa55445c8a63c5",
        q4_source_closure_root=bytes.fromhex(
            "583c337cf153df0ca18daba5cb79dbbf6c0969acbd27ccb5d1c3accd5dc3d602"
        ),
        q5_source_commit="1" * 40,
        q5_source_closure_root=b"5" * 32,
        public_keys=PUBLIC_KEYS,
    )


def test_actual_q4_capability_replays_producer_free() -> None:
    sealed = q5a.decode_sealed_q4_capability_receipt_v1(Q4_ARTIFACT.read_bytes())
    assert sealed.receipt_root.hex() == (
        "f45a3d0d37df747683d7de5c72f279a86fd7fc4220f3a782bb33ee0d36c9f0c8"
    )
    assert sealed.q4_repository_commit_id.raw_digest.hex() == (
        "2b33dc5e4145962e93af8722afaa55445c8a63c5"
    )
    assert sealed.q4_source_closure_root.hex() == (
        "583c337cf153df0ca18daba5cb79dbbf6c0969acbd27ccb5d1c3accd5dc3d602"
    )
    assert sealed.body_hash == content_hash(
        q5a.Q4_BODY_ROOT_DOMAIN, sealed.body_canonical_object
    )


def test_three_signer_manifests_round_trip_and_derive_key_ids() -> None:
    candidate = _candidate()
    assert tuple(item.signer_ordinal for item in candidate.signer_manifests) == (
        q5a.SIGNER_ROLE_CUSTODIAN,
        q5a.SIGNER_ROLE_PYTHON_REPLAY,
        q5a.SIGNER_ROLE_RUST_REPLAY,
    )
    assert all(
        item.canonical_object()[3] == q5a.PURPOSE5_FINAL_CERTIFICATE_SIGNER
        and item.canonical_object()[5]
        == q5a.SIGNER_ROLE_NAMES[item.signer_ordinal - 1].encode("ascii")
        and item.canonical_object()[6] == q5a.INITIAL_KEY_EPOCH
        and item.key_id == sha256(item.public_key).digest()[:16]
        and q5a.decode_purpose5_signer_manifest_v1(item.canonical_bytes) == item
        for item in candidate.signer_manifests
    )
    assert len({item.key_id for item in candidate.signer_manifests}) == 3
    assert len({item.public_key for item in candidate.signer_manifests}) == 3


def test_manifest_rejects_wrong_role_epoch_purpose_and_key_id() -> None:
    manifest = _candidate().signer_manifests[0]
    with pytest.raises(q5a.Q5aPurpose5WireError, match="key id"):
        replace(manifest, key_id=b"X" * 16)

    for index, wrong in ((3, 4), (4, 4), (5, b"WRONG_ROLE"), (6, 0)):
        row = list(manifest.canonical_object())
        row[index] = wrong
        with pytest.raises(q5a.Q5aPurpose5WireError, match="signer"):
            q5a.decode_purpose5_signer_manifest_v1(canonical_cbor_encode(row))


def test_boolean_values_cannot_masquerade_as_numeric_wire_fields() -> None:
    candidate = _candidate()
    manifest_row = list(candidate.signer_manifests[0].canonical_object())
    manifest_row[4] = True
    with pytest.raises(q5a.Q5aPurpose5WireError, match="signer"):
        q5a.decode_purpose5_signer_manifest_v1(
            canonical_cbor_encode(manifest_row)
        )

    candidate_row = list(candidate.canonical_object())
    candidate_row[21] = False
    with pytest.raises(q5a.Q5aPurpose5WireError, match="candidate wire"):
        q5a.decode_q5a_candidate_v1(canonical_cbor_encode(candidate_row))

    q4_row = list(
        q5a.decode_sealed_q4_capability_receipt_v1(
            Q4_ARTIFACT.read_bytes()
        ).canonical_object
    )
    q4_row[8] = False
    with pytest.raises(q5a.Q5aPurpose5WireError, match="Q4 capability"):
        q5a.decode_sealed_q4_capability_receipt_v1(
            canonical_cbor_encode(q4_row)
        )


def test_genesis_requires_three_ordered_distinct_public_keys() -> None:
    manifests = _candidate().signer_manifests
    with pytest.raises(q5a.Q5aPurpose5WireError, match="three ordered distinct"):
        q5a.UnpinnedTrustGenesisCandidateV1(
            (manifests[0], manifests[0], manifests[2])
        )
    with pytest.raises(q5a.Q5aPurpose5WireError, match="three ordered distinct"):
        q5a.UnpinnedTrustGenesisCandidateV1(tuple(reversed(manifests)))
    with pytest.raises(q5a.Q5aPurpose5WireError, match="three ordered distinct"):
        q5a.UnpinnedTrustGenesisCandidateV1(manifests[:2])


def test_genesis_and_status_are_unpinned_noncurrent_candidates() -> None:
    candidate = _candidate()
    genesis = q5a.decode_unpinned_trust_genesis_candidate_v1(
        candidate.trust_genesis.canonical_bytes
    )
    status = q5a.decode_genesis_only_status_candidate_v1(
        candidate.key_status.canonical_bytes
    )
    assert genesis == candidate.trust_genesis
    assert status == candidate.key_status
    assert genesis.canonical_object()[-1] is False
    assert status.canonical_object()[7:9] == ((), ())
    assert status.canonical_object()[-3:] == (False, 0, 0)
    assert status.trust_genesis.genesis_root == genesis.genesis_root


def test_q6_policy_rejects_q4_body_only_message_as_issuance_gate() -> None:
    candidate = _candidate()
    assert q5a.Q4_BODY_ONLY_SIGNATURE_MESSAGE_FORBIDDEN_AS_ISSUANCE_GATE
    assert q5a.Q6_SIGNATURE_PREIMAGE_POLICY_ID == (
        b"hegel-q6-purpose5-pinned-status-bound-signature-preimage/1"
    )
    assert b"pinned-status-bound" in q5a.Q6_SIGNATURE_PREIMAGE_POLICY_ID
    assert candidate.trust_genesis.canonical_object()[-2:] == (True, False)
    assert candidate.canonical_object()[18:20] == (
        q5a.Q6_SIGNATURE_PREIMAGE_POLICY_ID,
        True,
    )


def test_aggregate_candidate_round_trips_and_binds_lineage() -> None:
    candidate = _candidate()
    replayed = q5a.decode_q5a_candidate_v1(candidate.canonical_bytes)
    assert replayed == candidate
    assert replayed.q4_capability_receipt_root.hex() == (
        "f45a3d0d37df747683d7de5c72f279a86fd7fc4220f3a782bb33ee0d36c9f0c8"
    )
    assert replayed.q4_repository_commit_id.raw_digest.hex() == (
        "2b33dc5e4145962e93af8722afaa55445c8a63c5"
    )
    assert replayed.q5_repository_commit_id.raw_digest == bytes.fromhex("1" * 40)
    assert replayed.q5_source_closure_root == b"5" * 32
    assert replayed.trust_genesis.signer_manifests == replayed.signer_manifests
    assert replayed.key_status.trust_genesis == replayed.trust_genesis
    assert replayed.candidate_root == content_hash(
        q5a.Q5A_CANDIDATE_ROOT_DOMAIN, replayed.canonical_object()
    )


def test_candidate_tail_cannot_claim_pin_current_signatures_or_issuance() -> None:
    candidate = _candidate()
    row = candidate.canonical_object()
    assert row[20:23] == (3, 0, 0)
    assert row[23:27] == (
        q5a.ACTION_ID.encode("ascii"),
        q5a.Q5A_SUCCESS_STATUS.encode("ascii"),
        q5a.Q5A_CLAIM_LEVEL.encode("ascii"),
        True,
    )
    assert row[27:] == (False,) * 7
    for index, wrong in (
        (21, 3),
        (22, 1),
        (27, True),
        (28, True),
        (29, True),
        (30, True),
        (31, True),
    ):
        forged = list(row)
        forged[index] = wrong
        with pytest.raises(q5a.Q5aPurpose5WireError, match="candidate wire"):
            q5a.decode_q5a_candidate_v1(canonical_cbor_encode(forged))


def test_nested_candidate_roots_are_recomputed_exactly() -> None:
    candidate = _candidate()
    row = list(candidate.canonical_object())
    row[15] = b"G" * 32
    with pytest.raises(q5a.Q5aPurpose5WireError, match="nested roots"):
        q5a.decode_q5a_candidate_v1(canonical_cbor_encode(row))

    row = list(candidate.canonical_object())
    status = list(row[16])
    status[4] = b"S" * 32
    row[16] = tuple(status)
    with pytest.raises(q5a.Q5aPurpose5WireError, match="status/genesis"):
        q5a.decode_q5a_candidate_v1(canonical_cbor_encode(row))


def test_fixed_public_fixture_freezes_all_candidate_roots() -> None:
    candidate = _candidate()
    assert tuple(item.manifest_root.hex() for item in candidate.signer_manifests) == (
        "d0f18f694bb0c5c650a6d6dbe0b6460d61c5d9ace43474dd55221b612d623e58",
        "969757deccae39c26967cfebe81b78844626372f276b9678fe1e870c20155f9a",
        "61581d47c08eea84c488946cb7a8467f337022c1db73dc8053c61ab8bdeaac3c",
    )
    assert candidate.trust_genesis.genesis_root.hex() == (
        "6190a7ef76649a06bc66dd7d362226866e038ea2e18297c3707c1b3dd94b3c9e"
    )
    assert candidate.key_status.status_root.hex() == (
        "014ae9aac1212eb6593a4be5b82a392102cb413838b4bfde93b4bf1ab8e60362"
    )
    assert candidate.candidate_root.hex() == (
        "409766c1a77592ed15ea919fbb15fbfa6cb4798e7df4b076b05f442b4db7e52a"
    )


def test_candidate_root_changes_with_any_public_identity() -> None:
    candidate = _candidate()
    changed = q5a.build_q5a_candidate_v1(
        q4_capsule_digest=candidate.q4_capsule_digest,
        q4_authorization_digest=candidate.q4_authorization_digest,
        q4_consumption_record_digest=candidate.q4_consumption_record_digest,
        q4_terminal_record_digest=candidate.q4_terminal_record_digest,
        q4_capability_receipt_root=candidate.q4_capability_receipt_root,
        q4_body_hash=candidate.q4_body_hash,
        q4_source_commit=candidate.q4_repository_commit_id.raw_digest.hex(),
        q4_source_closure_root=candidate.q4_source_closure_root,
        q5_source_commit=candidate.q5_repository_commit_id.raw_digest.hex(),
        q5_source_closure_root=candidate.q5_source_closure_root,
        public_keys=(PUBLIC_KEYS[0], PUBLIC_KEYS[1], bytes.fromhex("e8" * 32)),
    )
    assert changed.signer_manifests[2].manifest_root != (
        candidate.signer_manifests[2].manifest_root
    )
    assert changed.trust_genesis.genesis_root != candidate.trust_genesis.genesis_root
    assert changed.key_status.status_root != candidate.key_status.status_root
    assert changed.candidate_root != candidate.candidate_root


def test_repository_commit_wire_is_raw_sha1_not_ascii_hex() -> None:
    candidate = _candidate()
    assert candidate.canonical_object()[9][0] == 1
    assert len(candidate.canonical_object()[9][1]) == 20
    with pytest.raises(q5a.Q5aPurpose5WireError, match="commit"):
        q5a.RepositoryCommitIdV1(1, b"1" * 40)


def test_no_private_key_or_signing_surface_is_exported() -> None:
    exported = set(q5a.__all__)
    assert not any("private" in name.lower() for name in exported)
    assert not any(
        callable(getattr(q5a, name))
        and (name.lower().startswith("sign") or name.lower().startswith("verify"))
        for name in exported
    )
    assert q5a.Q5A_SUCCESS_STATUS.endswith("NOT_PINNED_NOT_SIGNED")
