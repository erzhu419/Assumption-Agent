from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine import phase3_q4_formal_quotient_certificate_capability_v1 as q4
from hegel_machine.strict_cbor_v1 import canonical_cbor_encode


Q3_ARTIFACT = ROOT / "artifacts/phase3_q3_bounded_quotient_certificate_candidate_v1.cbor"


def _receipt() -> q4.Q4CapabilityReceiptV1:
    candidate = q4.decode_sealed_q3_candidate_v1(Q3_ARTIFACT.read_bytes())
    return q4.build_q4_capability_receipt_v1(
        q3_capsule_digest=bytes.fromhex(
            "f8459ae97648ae316a96a2d60da80e4c20bced20aeb518f400bc981f24ac7fc1"
        ),
        q3_authorization_digest=bytes.fromhex(
            "a72e3a8e644b43762561472b92465c2ea7344a8faf2ae495bf5a8a26e0a239a5"
        ),
        q3_consumption_record_digest=bytes.fromhex(
            "ee0f1a1bd626e90ff466bb339df31783d758c22c5b426b7f8a701436ce998354"
        ),
        q3_terminal_record_digest=bytes.fromhex(
            "bc0200feb3684c27975b6e212308c9d8b794f41b80dd7c666571e9474aab4624"
        ),
        q3_candidate_root=candidate.candidate_root,
        q3_source_closure_root=bytes.fromhex(
            "35613e7b0af7b97e94b1881a28dc286747c3466655fbf5aa49203636ffc20f6c"
        ),
        q3_source_commit="46956049481178177b395c3acde69bce28960dd7",
        q4_source_closure_root=b"Q" * 32,
        q4_source_commit="1" * 40,
        q3_candidate=candidate,
    )


def test_actual_q3_candidate_replays_without_q3_producer_import() -> None:
    candidate = q4.decode_sealed_q3_candidate_v1(Q3_ARTIFACT.read_bytes())
    assert candidate.candidate_root.hex() == (
        "8964461f54c8c45028d97b292e2d265b1ea00caf7b8df23a639f95b660005d8f"
    )
    assert candidate.dsl_version == "hegel-old-dsl-v1.6.0"
    assert candidate.closure_semantics_version == "hegel-quotient-closure-v1.0.1"
    assert len(candidate.null_matching_class_ids) == 1

    with pytest.raises(
        q4.Q4FormalCapabilityWireError,
        match="attributes do not match",
    ):
        replace(candidate, outside_truth_root=b"X" * 32)


def test_formal_body_and_capability_receipt_round_trip_exactly() -> None:
    receipt = _receipt()
    body = q4.decode_quotient_certificate_body_v1(receipt.body.canonical_bytes)
    replayed = q4.decode_q4_capability_receipt_v1(receipt.canonical_bytes)
    assert body == receipt.body
    assert replayed == receipt
    assert len(body.canonical_object()) == 34
    assert body.q4_repository_commit_id.raw_digest == bytes.fromhex("1" * 40)
    assert body.q4_source_closure_root == b"Q" * 32
    row = receipt.canonical_object()
    assert row[6] == q4.PURPOSE5_FINAL_CERTIFICATE_SIGNER
    assert tuple(value.decode("ascii") for value in row[7]) == (
        q4.MISSING_ISSUANCE_PREREQUISITES
    )
    assert row[8:12] == (0, 3, 0, 3)
    assert row[14:] == (True, False, False, False, False, False)


def test_wrong_body_tag_and_claim_elevation_are_rejected() -> None:
    receipt = _receipt()
    wrong_tag = list(receipt.body.canonical_object())
    wrong_tag[1] = 0x3E02
    with pytest.raises(q4.Q4FormalCapabilityWireError, match="body wire"):
        q4.decode_quotient_certificate_body_v1(canonical_cbor_encode(wrong_tag))

    elevated = list(receipt.body.canonical_object())
    elevated[-1] = b"EXTERNAL_AUTHORITY"
    with pytest.raises(q4.Q4FormalCapabilityWireError, match="body wire"):
        q4.decode_quotient_certificate_body_v1(canonical_cbor_encode(elevated))


def test_receipt_cannot_claim_signing_or_issuance() -> None:
    receipt = _receipt()
    forged = list(receipt.canonical_object())
    forged[16] = True
    with pytest.raises(q4.Q4FormalCapabilityWireError, match="receipt wire"):
        q4.decode_q4_capability_receipt_v1(canonical_cbor_encode(forged))


def test_future_envelope_codec_is_structural_and_three_of_three() -> None:
    body = _receipt().body
    signatures = tuple(
        q4.SignatureRecordV1(bytes((index,)) * 16, bytes((index,)) * 64)
        for index in (1, 2, 3)
    )
    envelope = q4.FinalCertificateEnvelopeV1(1_788_300_000, 1, body, signatures)
    assert q4.decode_final_certificate_envelope_v1(envelope.canonical_bytes) == envelope

    with pytest.raises(q4.Q4FormalCapabilityWireError, match="envelope"):
        q4.FinalCertificateEnvelopeV1(
            1_788_300_000, 1, body, tuple(reversed(signatures))
        )

    wire = list(envelope.canonical_object())
    wire[5] = b"X" * 32
    with pytest.raises(q4.Q4FormalCapabilityWireError, match="body hash"):
        q4.decode_final_certificate_envelope_v1(canonical_cbor_encode(wire))


def test_q4_body_repository_commit_is_raw_not_ascii_hex() -> None:
    body = _receipt().body
    assert body.q3_repository_commit_id.canonical_object()[0] == 1
    assert len(body.q3_repository_commit_id.canonical_object()[1]) == 20
    with pytest.raises(q4.Q4FormalCapabilityWireError, match="commit"):
        replace(
            body,
            q4_repository_commit_id=q4.RepositoryCommitIdV1(
                1, b"1" * 40
            ),
        )
