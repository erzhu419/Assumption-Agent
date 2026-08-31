"""Strict producer-free wire for the Phase-3A-Q4a formal capability.

The module accepts the already sealed Q3 candidate and constructs a complete
unsigned quotient-certificate body plus a capability receipt.  It contains no
key lookup, private-key, signing, filesystem, process, or network surface.
The formal envelope codec is structural only; Q4a never emits an envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, NoReturn

from .strict_cbor_v1 import (
    canonical_cbor_decode,
    canonical_cbor_encode,
    content_hash,
    rfc6962_root,
)


Q3_CANDIDATE_TAG: Final = 0x3D00
Q3_CANDIDATE_SCHEMA_ID: Final = (
    b"hegel-q3-bounded-quotient-certificate-candidate/1"
)
Q3_CANDIDATE_ROOT_DOMAIN: Final = (
    "HEGEL/Q3/BOUNDED_QUOTIENT_CERTIFICATE_CANDIDATE/V1"
)
Q3_SUCCESS_STATUS: Final = (
    b"Q3_BOUNDED_QUOTIENT_CERTIFICATE_CANDIDATE_COMPLETE"
)
Q3_CLAIM_LEVEL: Final = b"LOCAL_DISCLOSED_DEVELOPMENT_CONTROL"
Q3_EQUIVALENCE_ID: Final = b"exact_behavior_vector_with_bottom"
Q2_OUTSIDE_VERDICT: Final = b"OUTSIDE_FROZEN_QUOTIENT_CLOSURE"
Q2_IN_VERDICT: Final = b"IN_FROZEN_QUOTIENT_CLOSURE"

Q4_BODY_TAG: Final = 0x3E00
Q4_BODY_SCHEMA_ID: Final = b"hegel-q4-formal-quotient-certificate-body/1"
Q4_BODY_ROOT_DOMAIN: Final = "HEGEL/Q4/FORMAL_QUOTIENT_CERTIFICATE_BODY/V1"
Q4_RECEIPT_TAG: Final = 0x3E01
Q4_RECEIPT_SCHEMA_ID: Final = (
    b"hegel-q4-formal-quotient-certificate-capability-receipt/1"
)
Q4_RECEIPT_ROOT_DOMAIN: Final = (
    "HEGEL/Q4/FORMAL_QUOTIENT_CERTIFICATE_CAPABILITY_RECEIPT/V1"
)
Q4_SUCCESS_STATUS: Final = (
    "Q4_FORMAL_QUOTIENT_CERTIFICATE_CAPABILITY_ADMITTED_NOT_ISSUED"
)
Q4_CLAIM_KIND: Final = "OUTSIDE_FROZEN_QUOTIENT_CLOSURE"
Q4_DSL_VERSION: Final = "hegel-old-dsl-v1.6.0"
Q4_CLOSURE_SEMANTICS_VERSION: Final = "hegel-quotient-closure-v1.0.1"
Q4_EQUIVALENCE_ID: Final = "exact_behavior_vector_with_bottom"
Q4_CLAIM_LEVEL: Final = "LOCAL_DISCLOSED_DEVELOPMENT_CONTROL"
Q4_CAPABILITY_DISPOSITION: Final = "NOT_READY_FOR_ISSUANCE"

GIT_SHA1_ALGORITHM_ID: Final = 1
QUOTIENT_CERTIFICATE_KIND_ID: Final = 2
PURPOSE5_FINAL_CERTIFICATE_SIGNER: Final = 5
REQUIRED_ACTIVE_SIGNER_COUNT: Final = 3
REQUIRED_SIGNATURE_COUNT: Final = 3
SIGNATURE_MESSAGE_PREFIX: Final = b"HEGEL/Q4/QUOTIENT_CERT_SIGNATURE/V1\x00"
MISSING_ISSUANCE_PREREQUISITES: Final = (
    "PURPOSE5_TRUST_GENESIS_PIN",
    "PURPOSE5_CURRENT_KEY_STATUS_BUNDLE",
    "PURPOSE5_THREE_ACTIVE_SIGNERS",
    "THREE_OF_THREE_SIGNATURES",
)


class Q4FormalCapabilityWireError(ValueError):
    """One strict Q3/Q4 wire object differs from the frozen contract."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Q4FormalCapabilityWireError(code, detail)


def _bytes(value: object, length: int, label: str) -> bytes:
    if type(value) is not bytes or len(value) != length:
        _fail("REJECT_Q4_BYTES", f"{label} must be {length} bytes")
    return value


def _bytes32(value: object, label: str) -> bytes:
    return _bytes(value, 32, label)


def _ascii(value: object, label: str) -> str:
    if type(value) is not bytes or not value:
        _fail("REJECT_Q4_TEXT", f"{label} must be nonempty ASCII bytes")
    try:
        decoded = value.decode("ascii")
    except UnicodeDecodeError as error:
        _fail("REJECT_Q4_TEXT", f"{label}: {error}")
    if decoded.encode("ascii") != value:
        _fail("REJECT_Q4_TEXT", f"{label} does not replay")
    return decoded


@dataclass(frozen=True, slots=True)
class SealedQ3CandidateV1:
    q2_capsule_digest: bytes
    q2_authorization_digest: bytes
    q2_consumption_record_digest: bytes
    q2_terminal_record_digest: bytes
    q2_artifact_raw_sha256: bytes
    q2_artifact_root: bytes
    q1_completion_receipt_root: bytes
    q1_closure_bundle_root: bytes
    q1_dual_replay_agreement_root: bytes
    q1_target_blind_access_ledger_root: bytes
    outside_role_result_root: bytes
    null_role_result_root: bytes
    dsl_version: str
    closure_semantics_version: str
    outside_universe_root: bytes
    outside_truth_root: bytes
    null_universe_root: bytes
    null_truth_root: bytes
    null_matching_class_ids: tuple[bytes, ...]
    canonical_object: tuple[object, ...]

    def __post_init__(self) -> None:
        for name in (
            "q2_capsule_digest",
            "q2_authorization_digest",
            "q2_consumption_record_digest",
            "q2_terminal_record_digest",
            "q2_artifact_raw_sha256",
            "q2_artifact_root",
            "q1_completion_receipt_root",
            "q1_closure_bundle_root",
            "q1_dual_replay_agreement_root",
            "q1_target_blind_access_ledger_root",
            "outside_role_result_root",
            "null_role_result_root",
            "outside_universe_root",
            "outside_truth_root",
            "null_universe_root",
            "null_truth_root",
        ):
            _bytes32(getattr(self, name), name)
        if (
            self.dsl_version != Q4_DSL_VERSION
            or self.closure_semantics_version != Q4_CLOSURE_SEMANTICS_VERSION
            or len(self.null_matching_class_ids) != 1
            or any(
                type(item) is not bytes or len(item) != 32
                for item in self.null_matching_class_ids
            )
        ):
            _fail("REJECT_Q3_CANDIDATE", "candidate semantic binding differs")
        if self.canonical_object != self.projected_canonical_object():
            _fail(
                "REJECT_Q3_CANDIDATE",
                "candidate attributes do not match the sealed canonical row",
            )

    def projected_canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q3_CANDIDATE_TAG,
            Q3_CANDIDATE_SCHEMA_ID,
            self.q2_capsule_digest,
            self.q2_authorization_digest,
            self.q2_consumption_record_digest,
            self.q2_terminal_record_digest,
            self.q2_artifact_raw_sha256,
            self.q2_artifact_root,
            self.q1_completion_receipt_root,
            self.q1_closure_bundle_root,
            self.q1_dual_replay_agreement_root,
            self.q1_target_blind_access_ledger_root,
            (
                Q2_OUTSIDE_VERDICT,
                self.dsl_version.encode("ascii"),
                self.closure_semantics_version.encode("ascii"),
                1,
                self.outside_universe_root,
                Q3_EQUIVALENCE_ID,
                self.outside_truth_root,
                self.outside_role_result_root,
                418,
                0,
            ),
            (
                Q2_IN_VERDICT,
                2,
                self.null_universe_root,
                self.null_truth_root,
                self.null_role_result_root,
                89,
                1,
                self.null_matching_class_ids,
            ),
            Q3_SUCCESS_STATUS,
            Q3_CLAIM_LEVEL,
            True,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        )

    @property
    def candidate_root(self) -> bytes:
        return content_hash(
            Q3_CANDIDATE_ROOT_DOMAIN, self.projected_canonical_object()
        )


def decode_sealed_q3_candidate_v1(payload: bytes) -> SealedQ3CandidateV1:
    """Decode Q3 without importing its producer or any Q1/Q2 module."""

    if type(payload) is not bytes:
        raise TypeError("Q3 candidate payload must be exact bytes")
    try:
        value = canonical_cbor_decode(payload)
    except ValueError as error:
        _fail("REJECT_Q3_CANDIDATE", str(error))
    fixed_tail = (
        Q3_SUCCESS_STATUS,
        Q3_CLAIM_LEVEL,
        True,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    )
    if (
        type(value) is not tuple
        or len(value) != 36
        or value[:3] != (1, Q3_CANDIDATE_TAG, Q3_CANDIDATE_SCHEMA_ID)
        or type(value[13]) is not tuple
        or len(value[13]) != 10
        or type(value[14]) is not tuple
        or len(value[14]) != 8
        or value[15:] != fixed_tail
        or value[13][0] != Q2_OUTSIDE_VERDICT
        or value[13][3] != 1
        or value[13][5] != Q3_EQUIVALENCE_ID
        or value[13][8:] != (418, 0)
        or value[14][0] != Q2_IN_VERDICT
        or value[14][1] != 2
        or value[14][5:7] != (89, 1)
        or type(value[14][7]) is not tuple
    ):
        _fail("REJECT_Q3_CANDIDATE", "candidate wire differs")
    candidate = SealedQ3CandidateV1(
        q2_capsule_digest=value[3],
        q2_authorization_digest=value[4],
        q2_consumption_record_digest=value[5],
        q2_terminal_record_digest=value[6],
        q2_artifact_raw_sha256=value[7],
        q2_artifact_root=value[8],
        q1_completion_receipt_root=value[9],
        q1_closure_bundle_root=value[10],
        q1_dual_replay_agreement_root=value[11],
        q1_target_blind_access_ledger_root=value[12],
        outside_role_result_root=value[13][7],
        null_role_result_root=value[14][4],
        dsl_version=_ascii(value[13][1], "DSL version"),
        closure_semantics_version=_ascii(value[13][2], "closure version"),
        outside_universe_root=value[13][4],
        outside_truth_root=value[13][6],
        null_universe_root=value[14][2],
        null_truth_root=value[14][3],
        null_matching_class_ids=value[14][7],
        canonical_object=value,
    )
    if canonical_cbor_encode(value) != payload:
        _fail("REJECT_Q3_CANDIDATE", "candidate canonical replay differs")
    return candidate


@dataclass(frozen=True, slots=True)
class RepositoryCommitIdV1:
    algorithm_id: int
    raw_digest: bytes

    def __post_init__(self) -> None:
        if self.algorithm_id != GIT_SHA1_ALGORITHM_ID:
            _fail("REJECT_Q4_REPOSITORY_COMMIT", "only Git SHA-1 is frozen")
        _bytes(self.raw_digest, 20, "Git SHA-1 commit")

    @classmethod
    def from_hex(cls, value: object) -> "RepositoryCommitIdV1":
        if type(value) is not str or len(value) != 40:
            _fail("REJECT_Q4_REPOSITORY_COMMIT", "commit must be 40 lower hex")
        if any(character not in "0123456789abcdef" for character in value):
            _fail("REJECT_Q4_REPOSITORY_COMMIT", "commit must be 40 lower hex")
        return cls(GIT_SHA1_ALGORITHM_ID, bytes.fromhex(value))

    def canonical_object(self) -> tuple[object, ...]:
        return (self.algorithm_id, self.raw_digest)


def _decode_repository_commit(value: object) -> RepositoryCommitIdV1:
    if type(value) is not tuple or len(value) != 2:
        _fail("REJECT_Q4_REPOSITORY_COMMIT", "commit wire differs")
    return RepositoryCommitIdV1(value[0], value[1])


@dataclass(frozen=True, slots=True)
class QuotientCertificateBodyV1:
    q1_completion_receipt_root: bytes
    q1_closure_bundle_root: bytes
    q1_dual_replay_agreement_root: bytes
    q1_target_blind_access_ledger_root: bytes
    q2_capsule_digest: bytes
    q2_authorization_digest: bytes
    q2_consumption_record_digest: bytes
    q2_terminal_record_digest: bytes
    q2_artifact_root: bytes
    outside_role_result_root: bytes
    outside_universe_root: bytes
    outside_truth_root: bytes
    null_role_result_root: bytes
    null_universe_root: bytes
    null_truth_root: bytes
    null_matching_class_ids: tuple[bytes, ...]
    q3_capsule_digest: bytes
    q3_authorization_digest: bytes
    q3_consumption_record_digest: bytes
    q3_terminal_record_digest: bytes
    q3_candidate_root: bytes
    q3_repository_commit_id: RepositoryCommitIdV1
    q3_source_closure_root: bytes
    q4_repository_commit_id: RepositoryCommitIdV1
    q4_source_closure_root: bytes

    def __post_init__(self) -> None:
        for name in (
            "q1_completion_receipt_root",
            "q1_closure_bundle_root",
            "q1_dual_replay_agreement_root",
            "q1_target_blind_access_ledger_root",
            "q2_capsule_digest",
            "q2_authorization_digest",
            "q2_consumption_record_digest",
            "q2_terminal_record_digest",
            "q2_artifact_root",
            "outside_role_result_root",
            "outside_universe_root",
            "outside_truth_root",
            "null_role_result_root",
            "null_universe_root",
            "null_truth_root",
            "q3_capsule_digest",
            "q3_authorization_digest",
            "q3_consumption_record_digest",
            "q3_terminal_record_digest",
            "q3_candidate_root",
            "q3_source_closure_root",
            "q4_source_closure_root",
        ):
            _bytes32(getattr(self, name), name)
        if (
            type(self.q3_repository_commit_id) is not RepositoryCommitIdV1
            or type(self.q4_repository_commit_id) is not RepositoryCommitIdV1
            or len(self.null_matching_class_ids) != 1
            or any(
                type(item) is not bytes or len(item) != 32
                for item in self.null_matching_class_ids
            )
        ):
            _fail("REJECT_Q4_BODY", "body fixed binding differs")

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q4_BODY_TAG,
            Q4_BODY_SCHEMA_ID,
            Q4_CLAIM_KIND.encode("ascii"),
            Q4_DSL_VERSION.encode("ascii"),
            Q4_CLOSURE_SEMANTICS_VERSION.encode("ascii"),
            1,
            self.outside_universe_root,
            self.outside_truth_root,
            Q4_EQUIVALENCE_ID.encode("ascii"),
            self.q1_completion_receipt_root,
            self.q1_closure_bundle_root,
            self.q1_dual_replay_agreement_root,
            self.q1_target_blind_access_ledger_root,
            self.q2_capsule_digest,
            self.q2_authorization_digest,
            self.q2_consumption_record_digest,
            self.q2_terminal_record_digest,
            self.q2_artifact_root,
            self.outside_role_result_root,
            418,
            0,
            rfc6962_root(()),
            (
                2,
                self.null_universe_root,
                self.null_truth_root,
                self.null_role_result_root,
                89,
                1,
                self.null_matching_class_ids,
            ),
            self.q3_capsule_digest,
            self.q3_authorization_digest,
            self.q3_consumption_record_digest,
            self.q3_terminal_record_digest,
            self.q3_candidate_root,
            self.q3_repository_commit_id.canonical_object(),
            self.q3_source_closure_root,
            self.q4_repository_commit_id.canonical_object(),
            self.q4_source_closure_root,
            Q4_CLAIM_LEVEL.encode("ascii"),
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def body_hash(self) -> bytes:
        return content_hash(Q4_BODY_ROOT_DOMAIN, self.canonical_object())

    @property
    def signature_message(self) -> bytes:
        return SIGNATURE_MESSAGE_PREFIX + self.body_hash


def decode_quotient_certificate_body_v1(payload: bytes) -> QuotientCertificateBodyV1:
    if type(payload) is not bytes:
        raise TypeError("Q4 body payload must be exact bytes")
    try:
        value = canonical_cbor_decode(payload)
    except ValueError as error:
        _fail("REJECT_Q4_BODY", str(error))
    if (
        type(value) is not tuple
        or len(value) != 34
        or value[:10]
        != (
            1,
            Q4_BODY_TAG,
            Q4_BODY_SCHEMA_ID,
            Q4_CLAIM_KIND.encode("ascii"),
            Q4_DSL_VERSION.encode("ascii"),
            Q4_CLOSURE_SEMANTICS_VERSION.encode("ascii"),
            1,
            value[7] if len(value) > 7 else None,
            value[8] if len(value) > 8 else None,
            Q4_EQUIVALENCE_ID.encode("ascii"),
        )
        or value[20:23] != (418, 0, rfc6962_root(()))
        or type(value[23]) is not tuple
        or len(value[23]) != 7
        or value[23][0] != 2
        or value[23][4:6] != (89, 1)
        or type(value[23][6]) is not tuple
        or value[33] != Q4_CLAIM_LEVEL.encode("ascii")
    ):
        _fail("REJECT_Q4_BODY", "body wire differs")
    body = QuotientCertificateBodyV1(
        q1_completion_receipt_root=value[10],
        q1_closure_bundle_root=value[11],
        q1_dual_replay_agreement_root=value[12],
        q1_target_blind_access_ledger_root=value[13],
        q2_capsule_digest=value[14],
        q2_authorization_digest=value[15],
        q2_consumption_record_digest=value[16],
        q2_terminal_record_digest=value[17],
        q2_artifact_root=value[18],
        outside_role_result_root=value[19],
        outside_universe_root=value[7],
        outside_truth_root=value[8],
        null_role_result_root=value[23][3],
        null_universe_root=value[23][1],
        null_truth_root=value[23][2],
        null_matching_class_ids=value[23][6],
        q3_capsule_digest=value[24],
        q3_authorization_digest=value[25],
        q3_consumption_record_digest=value[26],
        q3_terminal_record_digest=value[27],
        q3_candidate_root=value[28],
        q3_repository_commit_id=_decode_repository_commit(value[29]),
        q3_source_closure_root=value[30],
        q4_repository_commit_id=_decode_repository_commit(value[31]),
        q4_source_closure_root=value[32],
    )
    if body.canonical_bytes != payload:
        _fail("REJECT_Q4_BODY", "body replay differs")
    return body


@dataclass(frozen=True, slots=True)
class Q4CapabilityReceiptV1:
    body: QuotientCertificateBodyV1

    def __post_init__(self) -> None:
        if type(self.body) is not QuotientCertificateBodyV1:
            _fail("REJECT_Q4_RECEIPT", "receipt body type differs")

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q4_RECEIPT_TAG,
            Q4_RECEIPT_SCHEMA_ID,
            self.body.canonical_object(),
            self.body.body_hash,
            self.body.q3_candidate_root,
            PURPOSE5_FINAL_CERTIFICATE_SIGNER,
            tuple(item.encode("ascii") for item in MISSING_ISSUANCE_PREREQUISITES),
            0,
            REQUIRED_ACTIVE_SIGNER_COUNT,
            0,
            REQUIRED_SIGNATURE_COUNT,
            Q4_CAPABILITY_DISPOSITION.encode("ascii"),
            Q4_SUCCESS_STATUS.encode("ascii"),
            True,
            False,
            False,
            False,
            False,
            False,
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def receipt_root(self) -> bytes:
        return content_hash(Q4_RECEIPT_ROOT_DOMAIN, self.canonical_object())


def decode_q4_capability_receipt_v1(payload: bytes) -> Q4CapabilityReceiptV1:
    if type(payload) is not bytes:
        raise TypeError("Q4 receipt payload must be exact bytes")
    try:
        value = canonical_cbor_decode(payload)
    except ValueError as error:
        _fail("REJECT_Q4_RECEIPT", str(error))
    fixed_tail = (
        PURPOSE5_FINAL_CERTIFICATE_SIGNER,
        tuple(item.encode("ascii") for item in MISSING_ISSUANCE_PREREQUISITES),
        0,
        REQUIRED_ACTIVE_SIGNER_COUNT,
        0,
        REQUIRED_SIGNATURE_COUNT,
        Q4_CAPABILITY_DISPOSITION.encode("ascii"),
        Q4_SUCCESS_STATUS.encode("ascii"),
        True,
        False,
        False,
        False,
        False,
        False,
    )
    if (
        type(value) is not tuple
        or len(value) != 20
        or value[:3] != (1, Q4_RECEIPT_TAG, Q4_RECEIPT_SCHEMA_ID)
        or type(value[3]) is not tuple
        or value[6:] != fixed_tail
    ):
        _fail("REJECT_Q4_RECEIPT", "receipt wire differs")
    body = decode_quotient_certificate_body_v1(canonical_cbor_encode(value[3]))
    if value[4] != body.body_hash or value[5] != body.q3_candidate_root:
        _fail("REJECT_Q4_RECEIPT", "receipt body binding differs")
    receipt = Q4CapabilityReceiptV1(body)
    if receipt.canonical_bytes != payload:
        _fail("REJECT_Q4_RECEIPT", "receipt replay differs")
    return receipt


def build_q4_capability_receipt_v1(
    *,
    q3_capsule_digest: bytes,
    q3_authorization_digest: bytes,
    q3_consumption_record_digest: bytes,
    q3_terminal_record_digest: bytes,
    q3_candidate_root: bytes,
    q3_source_closure_root: bytes,
    q3_source_commit: str,
    q4_source_closure_root: bytes,
    q4_source_commit: str,
    q3_candidate: SealedQ3CandidateV1,
) -> Q4CapabilityReceiptV1:
    if type(q3_candidate) is not SealedQ3CandidateV1:
        _fail("REJECT_Q4_BRIDGE", "Q3 candidate exact type differs")
    if q3_candidate.candidate_root != q3_candidate_root:
        _fail("REJECT_Q4_BRIDGE", "Q3 candidate root differs")
    body = QuotientCertificateBodyV1(
        q1_completion_receipt_root=q3_candidate.q1_completion_receipt_root,
        q1_closure_bundle_root=q3_candidate.q1_closure_bundle_root,
        q1_dual_replay_agreement_root=q3_candidate.q1_dual_replay_agreement_root,
        q1_target_blind_access_ledger_root=(
            q3_candidate.q1_target_blind_access_ledger_root
        ),
        q2_capsule_digest=q3_candidate.q2_capsule_digest,
        q2_authorization_digest=q3_candidate.q2_authorization_digest,
        q2_consumption_record_digest=q3_candidate.q2_consumption_record_digest,
        q2_terminal_record_digest=q3_candidate.q2_terminal_record_digest,
        q2_artifact_root=q3_candidate.q2_artifact_root,
        outside_role_result_root=q3_candidate.outside_role_result_root,
        outside_universe_root=q3_candidate.outside_universe_root,
        outside_truth_root=q3_candidate.outside_truth_root,
        null_role_result_root=q3_candidate.null_role_result_root,
        null_universe_root=q3_candidate.null_universe_root,
        null_truth_root=q3_candidate.null_truth_root,
        null_matching_class_ids=q3_candidate.null_matching_class_ids,
        q3_capsule_digest=q3_capsule_digest,
        q3_authorization_digest=q3_authorization_digest,
        q3_consumption_record_digest=q3_consumption_record_digest,
        q3_terminal_record_digest=q3_terminal_record_digest,
        q3_candidate_root=q3_candidate_root,
        q3_repository_commit_id=RepositoryCommitIdV1.from_hex(q3_source_commit),
        q3_source_closure_root=q3_source_closure_root,
        q4_repository_commit_id=RepositoryCommitIdV1.from_hex(q4_source_commit),
        q4_source_closure_root=q4_source_closure_root,
    )
    receipt = Q4CapabilityReceiptV1(body)
    # Exercise the frozen decoder before returning bytes to a controller.
    if decode_q4_capability_receipt_v1(receipt.canonical_bytes) != receipt:
        _fail("REJECT_Q4_BRIDGE", "capability receipt round trip differs")
    return receipt


@dataclass(frozen=True, slots=True)
class SignatureRecordV1:
    key_id: bytes
    signature: bytes

    def __post_init__(self) -> None:
        _bytes(self.key_id, 16, "signature key id")
        _bytes(self.signature, 64, "Ed25519 signature")

    def canonical_object(self) -> tuple[object, ...]:
        return (self.key_id, self.signature)


@dataclass(frozen=True, slots=True)
class FinalCertificateEnvelopeV1:
    created_at_unix_seconds: int
    key_epoch: int
    body: QuotientCertificateBodyV1
    signatures: tuple[SignatureRecordV1, ...]

    def __post_init__(self) -> None:
        if (
            type(self.created_at_unix_seconds) is not int
            or self.created_at_unix_seconds < 0
            or type(self.key_epoch) is not int
            or self.key_epoch < 0
            or type(self.body) is not QuotientCertificateBodyV1
            or type(self.signatures) is not tuple
            or len(self.signatures) != REQUIRED_SIGNATURE_COUNT
            or any(type(item) is not SignatureRecordV1 for item in self.signatures)
            or tuple(sorted(item.key_id for item in self.signatures))
            != tuple(item.key_id for item in self.signatures)
            or len({item.key_id for item in self.signatures}) != len(self.signatures)
        ):
            _fail("REJECT_Q4_ENVELOPE", "envelope fixed binding differs")

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            QUOTIENT_CERTIFICATE_KIND_ID,
            self.created_at_unix_seconds,
            self.key_epoch,
            self.body.canonical_object(),
            self.body.body_hash,
            tuple(item.canonical_object() for item in self.signatures),
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())


def decode_final_certificate_envelope_v1(payload: bytes) -> FinalCertificateEnvelopeV1:
    """Decode a future envelope structurally; this does not verify signatures."""

    if type(payload) is not bytes:
        raise TypeError("Q4 envelope payload must be exact bytes")
    try:
        value = canonical_cbor_decode(payload)
    except ValueError as error:
        _fail("REJECT_Q4_ENVELOPE", str(error))
    if (
        type(value) is not tuple
        or len(value) != 7
        or value[:2] != (1, QUOTIENT_CERTIFICATE_KIND_ID)
        or type(value[4]) is not tuple
        or type(value[6]) is not tuple
        or len(value[6]) != REQUIRED_SIGNATURE_COUNT
    ):
        _fail("REJECT_Q4_ENVELOPE", "envelope wire differs")
    body = decode_quotient_certificate_body_v1(canonical_cbor_encode(value[4]))
    if value[5] != body.body_hash:
        _fail("REJECT_Q4_ENVELOPE", "body hash differs")
    signatures = tuple(
        SignatureRecordV1(row[0], row[1])
        if type(row) is tuple and len(row) == 2
        else _fail("REJECT_Q4_ENVELOPE", "signature record wire differs")
        for row in value[6]
    )
    envelope = FinalCertificateEnvelopeV1(value[2], value[3], body, signatures)
    if envelope.canonical_bytes != payload:
        _fail("REJECT_Q4_ENVELOPE", "envelope replay differs")
    return envelope


__all__ = [
    "FinalCertificateEnvelopeV1",
    "MISSING_ISSUANCE_PREREQUISITES",
    "PURPOSE5_FINAL_CERTIFICATE_SIGNER",
    "Q4CapabilityReceiptV1",
    "Q4FormalCapabilityWireError",
    "Q4_BODY_ROOT_DOMAIN",
    "Q4_BODY_SCHEMA_ID",
    "Q4_BODY_TAG",
    "Q4_RECEIPT_ROOT_DOMAIN",
    "Q4_RECEIPT_SCHEMA_ID",
    "Q4_RECEIPT_TAG",
    "Q4_SUCCESS_STATUS",
    "QuotientCertificateBodyV1",
    "RepositoryCommitIdV1",
    "SealedQ3CandidateV1",
    "SignatureRecordV1",
    "build_q4_capability_receipt_v1",
    "decode_final_certificate_envelope_v1",
    "decode_q4_capability_receipt_v1",
    "decode_quotient_certificate_body_v1",
    "decode_sealed_q3_candidate_v1",
]
