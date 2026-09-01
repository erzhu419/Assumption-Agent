from __future__ import annotations

import ast
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine import phase3_q5b_purpose5_trust_amendment_v1 as q5b  # noqa: E402
from hegel_machine.strict_cbor_v1 import canonical_cbor_encode  # noqa: E402


Q5A_ARTIFACT = (
    ROOT / "artifacts/phase3_q5a_purpose5_signer_genesis_candidate_v1.cbor"
)
Q5A_CAPSULE = bytes.fromhex(
    "497b0d9ce42d4e2c23913127a73d8ff1dc607b90260b8468f72c6dc9cd883220"
)
Q5A_AUTHORIZATION = bytes.fromhex(
    "b27bc9727b23037e233997fc23e7e1d136e7c02cf69bfda819437b14d988c158"
)
Q5A_CONSUMPTION = bytes.fromhex(
    "4fb583b9b036473692ee73a930b3e3e718d0ede6d773ee6a1d5cbf5e6c44ab0a"
)
Q5A_TERMINAL = bytes.fromhex(
    "a07d87d5a87295411682bebf2353ac54e8528a95d3abba953e20f58826cb6120"
)
Q5A_SOURCE_COMMIT = "aaf01d960c9f772510106f8f0ec735fa1e772beb"
Q5A_SOURCE_CLOSURE = bytes.fromhex(
    "d006225ae79e6980c6109986c1ca8f69541ca79de2edfa7d28d7acc362fe3acb"
)

# A public source identity fixture, not a key or seed.  The actual Q5b freeze
# replaces these two values at the state/capsule layer.
Q5B_SOURCE_COMMIT = "b" * 40
Q5B_SOURCE_CLOSURE = b"B" * 32


def _artifact() -> bytes:
    return Q5A_ARTIFACT.read_bytes()


def _receipt(**overrides: object) -> q5b.Q5bPurpose5TrustAmendmentReceiptV1:
    values: dict[str, object] = {
        "q5a_capsule_digest": Q5A_CAPSULE,
        "q5a_authorization_digest": Q5A_AUTHORIZATION,
        "q5a_consumption_record_digest": Q5A_CONSUMPTION,
        "q5a_terminal_record_digest": Q5A_TERMINAL,
        "q5a_source_commit": Q5A_SOURCE_COMMIT,
        "q5a_source_closure_root": Q5A_SOURCE_CLOSURE,
        "q5b_source_commit": Q5B_SOURCE_COMMIT,
        "q5b_source_closure_root": Q5B_SOURCE_CLOSURE,
    }
    values.update(overrides)
    return q5b.build_q5b_trust_amendment_v1(_artifact(), **values)  # type: ignore[arg-type]


def test_actual_q5a_artifact_replays_without_q5a_producer_import() -> None:
    payload = _artifact()
    candidate = q5b.decode_sealed_q5a_candidate_v1(payload)
    assert sha256(payload).hexdigest() == (
        "ea0fd7a15b58d8b78dc2a39a0b47d0e1346987b1bbd7930ba348e981cf5348e7"
    )
    assert candidate.candidate_root.hex() == (
        "1b06225e48f28af06d9b3fcd3d4fd4075613970ebfab0c1305ccca1f7c94ec57"
    )
    assert candidate.unpinned_genesis_root.hex() == (
        "0b00012d833ef036d6d3be074ae7e7ef10339bd0fe0977324abda664cdf0437f"
    )
    assert candidate.genesis_only_status_root.hex() == (
        "079c795b88eed90dcc5384631e7af164f4fa2b7133cad524090853eff4ba1cef"
    )
    assert candidate.q4_capability_receipt_root.hex() == (
        "f45a3d0d37df747683d7de5c72f279a86fd7fc4220f3a782bb33ee0d36c9f0c8"
    )
    assert candidate.q4_body_hash.hex() == (
        "873b10cb7ac9b6eec2c00543ccc755c86e2c20c7180101842b74a2d6856c03de"
    )
    assert candidate.q4_source_commit_raw.hex() == (
        "2b33dc5e4145962e93af8722afaa55445c8a63c5"
    )
    assert candidate.q4_source_closure_root.hex() == (
        "583c337cf153df0ca18daba5cb79dbbf6c0969acbd27ccb5d1c3accd5dc3d602"
    )
    assert candidate.q5a_source_commit_raw.hex() == Q5A_SOURCE_COMMIT
    assert candidate.q5a_source_closure_root == Q5A_SOURCE_CLOSURE


def test_q4_artifact_lineage_is_distinct_from_q5a_outer_authority() -> None:
    receipt = _receipt()
    candidate = receipt.q5a_candidate
    assert candidate.q4_capsule_digest != receipt.q5a_capsule_digest
    assert candidate.q4_authorization_digest != receipt.q5a_authorization_digest
    assert candidate.q4_consumption_record_digest != (
        receipt.q5a_consumption_record_digest
    )
    assert candidate.q4_terminal_record_digest != receipt.q5a_terminal_record_digest
    assert candidate.q4_source_commit_raw != receipt.q5a_source_commit_raw
    assert candidate.q4_source_closure_root != receipt.q5a_source_closure_root


def test_actual_manifest_roots_and_key_ids_are_bound_exactly() -> None:
    receipt = _receipt()
    assert tuple(root.hex() for root in receipt.q5a_manifest_roots) == (
        "b92ebd24dd307aea11d28af5cf874b980c3658a8513b9d8e5458c2bf90aaea90",
        "213289ebf14093ec37f696e216c27209ff70f896ef674b5f0c0d5082590474ba",
        "8a6f9504ff15a73b57a05d2fcf9e63336c5db07bab219e2b8bc1f16c4339aacb",
    )
    assert tuple(key_id.hex() for key_id in receipt.q5a_key_ids) == (
        "568e0c23e25dd11be10ac41bbf8bd6cb",
        "b43d7064dafbf4151eb054d7ff281c5c",
        "6266c47bea6e2fc8c2aad410cf32d18a",
    )
    assert receipt.pinned_trust_genesis.manifest_roots == receipt.q5a_manifest_roots
    assert receipt.current_key_status.signer_manifests == (
        receipt.pinned_trust_genesis.signer_manifests
    )


def test_pinned_genesis_round_trip_freezes_three_active_signers_and_two_of_three() -> None:
    genesis = _receipt().pinned_trust_genesis
    assert q5b.decode_pinned_trust_genesis_v1(genesis.canonical_bytes) == genesis
    row = genesis.canonical_object()
    assert row[10:14] == (3, 3, 2, 3)
    assert row[14] == q5b.Q6_SIGNATURE_PREIMAGE_POLICY_ID
    assert row[15:] == (True, True)
    assert genesis.genesis_root.hex() == (
        "8a772a4c79ab7b6139eeaebc397d123e02248a2a307c58c558a3dc2f70f11d19"
    )


def test_current_status_is_epoch1_and_distinguishes_signer_active_from_global_active() -> None:
    receipt = _receipt()
    status = receipt.current_key_status
    assert q5b.decode_current_key_status_bundle_v1(status.canonical_bytes) == status
    assert status.current_key_epoch == 1
    assert status.purpose5_active_signer_count == 3
    assert status.canonical_object()[9:13] == ((1, 2, 3), 3, 2, 3)
    assert status.canonical_object()[14:16] == (True, True)
    assert status.canonical_object()[16:] == (0, 0, 0)
    assert receipt.purpose5_active_signer_count == 3
    assert receipt.purpose5_trust_genesis_pinned is True
    assert receipt.purpose5_current_key_status_accepted is True
    assert receipt.global_active_transition_performed is False
    assert q5b.GLOBAL_ACTIVE_TRANSITION_PERFORMED is False
    assert status.status_root.hex() == (
        "0eba7a69d2c77328537bc8cd5ee1c7935dc45caedb726ac91b1c304033e67c3c"
    )


def test_receipt_round_trip_binds_full_q5a_outer_and_source_lineage() -> None:
    receipt = _receipt()
    replayed = q5b.decode_q5b_trust_amendment_receipt_v1(receipt.canonical_bytes)
    assert replayed == receipt
    assert replayed.q5a_capsule_digest == Q5A_CAPSULE
    assert replayed.q5a_authorization_digest == Q5A_AUTHORIZATION
    assert replayed.q5a_consumption_record_digest == Q5A_CONSUMPTION
    assert replayed.q5a_terminal_record_digest == Q5A_TERMINAL
    assert replayed.q5a_artifact_raw_sha256 == sha256(_artifact()).digest()
    assert replayed.q5a_source_commit_raw.hex() == Q5A_SOURCE_COMMIT
    assert replayed.q5a_source_closure_root == Q5A_SOURCE_CLOSURE
    assert replayed.q5b_source_commit_raw.hex() == Q5B_SOURCE_COMMIT
    assert replayed.q5b_source_closure_root == Q5B_SOURCE_CLOSURE
    assert q5b.validate_q5b_receipt_against_q5a_artifact_v1(
        receipt.canonical_bytes, _artifact()
    ) == receipt
    assert receipt.receipt_root.hex() == (
        "b5d785c1dfe3decec573225d07877a1bcb33cd7c40958ac25b42ad37306a4695"
    )


def _authority_expectations() -> dict[str, object]:
    return {
        "expected_q5a_capsule_digest": Q5A_CAPSULE,
        "expected_q5a_authorization_digest": Q5A_AUTHORIZATION,
        "expected_q5a_consumption_record_digest": Q5A_CONSUMPTION,
        "expected_q5a_terminal_record_digest": Q5A_TERMINAL,
        "expected_q5a_artifact_raw_sha256": sha256(_artifact()).digest(),
        "expected_q5a_source_commit": Q5A_SOURCE_COMMIT,
        "expected_q5a_source_closure_root": Q5A_SOURCE_CLOSURE,
        "expected_q5b_source_commit": Q5B_SOURCE_COMMIT,
        "expected_q5b_source_closure_root": Q5B_SOURCE_CLOSURE,
    }


def test_external_authority_expectations_are_required_and_pass_actual_chain() -> None:
    receipt = _receipt()
    assert q5b.validate_q5b_receipt_authority_v1(
        receipt.canonical_bytes,
        _artifact(),
        **_authority_expectations(),  # type: ignore[arg-type]
    ) == receipt


@pytest.mark.parametrize(
    ("field", "wrong"),
    (
        ("expected_q5a_capsule_digest", b"1" * 32),
        ("expected_q5a_authorization_digest", b"2" * 32),
        ("expected_q5a_consumption_record_digest", b"3" * 32),
        ("expected_q5a_terminal_record_digest", b"4" * 32),
        ("expected_q5a_artifact_raw_sha256", b"5" * 32),
        ("expected_q5a_source_commit", "c" * 40),
        ("expected_q5a_source_closure_root", b"6" * 32),
        ("expected_q5b_source_commit", "d" * 40),
        ("expected_q5b_source_closure_root", b"7" * 32),
    ),
)
def test_external_authority_expectation_drift_is_rejected(
    field: str, wrong: object
) -> None:
    expectations = _authority_expectations()
    expectations[field] = wrong
    with pytest.raises(q5b.Q5bPurpose5TrustAmendmentError, match="AUTHORITY"):
        q5b.validate_q5b_receipt_authority_v1(
            _receipt().canonical_bytes,
            _artifact(),
            **expectations,  # type: ignore[arg-type]
        )


def test_success_is_current_but_unsigned_unissued_and_globally_inactive() -> None:
    row = _receipt().canonical_object()
    assert row[24:27] == (3, 2, 3)
    assert row[30] == q5b.Q5B_SUCCESS_STATUS.encode("ascii")
    assert q5b.Q5B_SUCCESS_STATUS.endswith(
        "ADMITTED_CURRENT_NOT_SIGNED_NOT_ISSUED"
    )
    # producer-free replay, Purpose-5 pin and current acceptance
    assert row[32:35] == (True, True, True)
    # keygen, private-read and signature counts
    assert row[35:38] == (0, 0, 0)
    # formal-sign, envelope, issuance, global ACTIVE, synthesis
    assert row[38:43] == (False, False, False, False, False)


@pytest.mark.parametrize("index", (9, 10, 11))
def test_candidate_genesis_and_status_root_drift_is_rejected(index: int) -> None:
    row = list(_receipt().canonical_object())
    row[index] = b"X" * 32
    with pytest.raises(q5b.Q5bPurpose5TrustAmendmentError, match="object graph"):
        q5b.decode_q5b_trust_amendment_receipt_v1(canonical_cbor_encode(row))


def test_manifest_root_and_key_id_drift_is_rejected() -> None:
    for index, replacement in (
        (12, (b"M" * 32,) * 3),
        (13, (b"K" * 16,) * 3),
    ):
        row = list(_receipt().canonical_object())
        row[index] = replacement
        with pytest.raises(q5b.Q5bPurpose5TrustAmendmentError, match="object graph"):
            q5b.decode_q5b_trust_amendment_receipt_v1(canonical_cbor_encode(row))


def test_q5a_artifact_raw_binding_is_recomputed_from_nested_candidate() -> None:
    receipt = _receipt()
    with pytest.raises(q5b.Q5bPurpose5TrustAmendmentError, match="object graph"):
        replace(receipt, q5a_artifact_raw_sha256=b"R" * 32)

    row = list(receipt.canonical_object())
    row[7] = b"R" * 32
    with pytest.raises(q5b.Q5bPurpose5TrustAmendmentError, match="object graph"):
        q5b.decode_q5b_trust_amendment_receipt_v1(canonical_cbor_encode(row))


def test_boolean_values_cannot_masquerade_as_numeric_wire_fields() -> None:
    genesis_row = list(_receipt().pinned_trust_genesis.canonical_object())
    genesis_row[10] = True
    with pytest.raises(q5b.Q5bPurpose5TrustAmendmentError, match="pinned genesis"):
        q5b.decode_pinned_trust_genesis_v1(canonical_cbor_encode(genesis_row))

    status_row = list(_receipt().current_key_status.canonical_object())
    status_row[10] = True
    with pytest.raises(q5b.Q5bPurpose5TrustAmendmentError, match="current status"):
        q5b.decode_current_key_status_bundle_v1(canonical_cbor_encode(status_row))

    receipt_row = list(_receipt().canonical_object())
    receipt_row[24] = True
    with pytest.raises(q5b.Q5bPurpose5TrustAmendmentError, match="receipt wire"):
        q5b.decode_q5b_trust_amendment_receipt_v1(
            canonical_cbor_encode(receipt_row)
        )


def test_m25_actor_trust_wire_is_not_reused_or_accepted() -> None:
    assert q5b.Q5B_PINNED_GENESIS_SCHEMA_ID != q5b.M25_ACTOR_TRUST_GENESIS_SCHEMA_ID
    assert q5b.Q5B_PINNED_GENESIS_ROOT_DOMAIN != (
        q5b.M25_ACTOR_TRUST_GENESIS_ROOT_DOMAIN
    )
    assert q5b.M25_PRE_M4_PURPOSE_SET == (1, 2, 3, 4)
    assert q5b.PURPOSE5_FINAL_CERTIFICATE_SIGNER == 5
    m25_actor_wire = (
        1,
        0x3111,
        q5b.M25_ACTOR_TRUST_GENESIS_SCHEMA_ID,
        b"I" * 16,
        ((1, b"R" * 32),),
        b"P" * 32,
        1,
        (1, b"C" * 20),
    )
    with pytest.raises(q5b.Q5bPurpose5TrustAmendmentError, match="pinned genesis"):
        q5b.decode_pinned_trust_genesis_v1(
            canonical_cbor_encode(m25_actor_wire)
        )


def test_source_lineage_changes_receipt_not_pinned_public_identity() -> None:
    base = _receipt()
    changed = _receipt(
        q5b_source_commit="c" * 40,
        q5b_source_closure_root=b"C" * 32,
    )
    assert changed.pinned_trust_genesis == base.pinned_trust_genesis
    assert changed.current_key_status == base.current_key_status
    assert changed.receipt_root != base.receipt_root


def test_no_keygen_private_read_signing_or_process_surface() -> None:
    source_path = ROOT / "src/hegel_machine/phase3_q5b_purpose5_trust_amendment_v1.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not {"subprocess", "socket", "os", "cryptography"}.intersection(imported)
    assert "phase3_q5a_purpose5_signer_genesis_v1" not in source
    exported = set(q5b.__all__)
    assert not any("private" in name.lower() or "keygen" in name.lower() for name in exported)
    assert not any(
        name.lower().startswith(("sign", "verify")) for name in exported
    )
