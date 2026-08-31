"""Producer-free strict wire for the Phase-3A-Q3 bounded certificate candidate.

This module decodes only the already sealed Q2 result artifact.  It cannot
generate target rows, read a split, enumerate a Q1 archive, or rerun the Q2
membership fold.  The emitted object is an unsigned local development
candidate, not a formal certificate and not authority to start synthesis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, NoReturn

from .strict_cbor_v1 import (
    canonical_cbor_decode,
    canonical_cbor_encode,
    content_hash,
)


Q2_ROLE_RESULT_TAG: Final = 0x3C00
Q2_MEMBERSHIP_ARTIFACT_TAG: Final = 0x3C01
Q2_ROLE_RESULT_SCHEMA_ID: Final = b"hegel-q2-role-membership-result/1"
Q2_MEMBERSHIP_ARTIFACT_SCHEMA_ID: Final = b"hegel-q2-membership-artifact/1"
Q2_ROLE_RESULT_ROOT_DOMAIN: Final = "HEGEL/Q2/ROLE_MEMBERSHIP_RESULT/V1"
Q2_MEMBERSHIP_ARTIFACT_ROOT_DOMAIN: Final = "HEGEL/Q2/MEMBERSHIP_ARTIFACT/V1"
Q2_ROLE_MATCH_PROFILE_ID: Final = b"BOOL_BIT_EXACT_PREDICATE_MATCH_V1"
Q2_PASS_STATUS: Final = b"Q2_ROLE_MEMBERSHIP_PASS"
Q2_OUTSIDE_VERDICT: Final = b"OUTSIDE_FROZEN_QUOTIENT_CLOSURE"
Q2_IN_VERDICT: Final = b"IN_FROZEN_QUOTIENT_CLOSURE"

Q3_CANDIDATE_TAG: Final = 0x3D00
Q3_CANDIDATE_SCHEMA_ID: Final = (
    b"hegel-q3-bounded-quotient-certificate-candidate/1"
)
Q3_CANDIDATE_ROOT_DOMAIN: Final = (
    "HEGEL/Q3/BOUNDED_QUOTIENT_CERTIFICATE_CANDIDATE/V1"
)
Q3_SUCCESS_STATUS: Final = (
    "Q3_BOUNDED_QUOTIENT_CERTIFICATE_CANDIDATE_COMPLETE"
)
Q3_CLAIM_LEVEL: Final = "LOCAL_DISCLOSED_DEVELOPMENT_CONTROL"
Q3_EQUIVALENCE_ID: Final = "exact_behavior_vector_with_bottom"


class Q3CertificateWireError(ValueError):
    """A sealed Q2 input or Q3 candidate differs from the strict wire."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Q3CertificateWireError(code, detail)


def _bytes32(value: object, label: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        _fail("REJECT_Q3_DIGEST", f"{label} must be 32 bytes")
    return value


def _ascii(value: object, label: str) -> str:
    if type(value) is not bytes or not value:
        _fail("REJECT_Q3_TEXT", f"{label} must be nonempty ASCII bytes")
    try:
        result = value.decode("ascii")
    except UnicodeDecodeError as error:
        _fail("REJECT_Q3_TEXT", f"{label}: {error}")
    if result.encode("ascii") != value:
        _fail("REJECT_Q3_TEXT", f"{label} did not replay")
    return result


@dataclass(frozen=True, slots=True)
class Q2RoleProjectionV1:
    role_id: int
    input_signature_id: int
    universe_root: bytes
    truth_root: bytes
    q1_signature_manifest_root: bytes
    q1_saturation_state_root: bytes
    q1_class_archive_root: bytes
    evaluated_class_count: int
    ordered_matching_class_ids: tuple[bytes, ...]
    match_count: int
    membership_verdict: str
    result_root: bytes
    canonical_row: tuple[object, ...]

    def __post_init__(self) -> None:
        if self.role_id not in (1, 2) or self.input_signature_id != self.role_id:
            _fail("REJECT_Q2_ROLE_RESULT", "role/signature identity differs")
        for name in (
            "universe_root",
            "truth_root",
            "q1_signature_manifest_root",
            "q1_saturation_state_root",
            "q1_class_archive_root",
            "result_root",
        ):
            _bytes32(getattr(self, name), name)
        if type(self.evaluated_class_count) is not int or self.evaluated_class_count < 1:
            _fail("REJECT_Q2_ROLE_RESULT", "evaluated class count differs")
        if (
            type(self.ordered_matching_class_ids) is not tuple
            or any(
                type(value) is not bytes or len(value) != 32
                for value in self.ordered_matching_class_ids
            )
            or tuple(sorted(self.ordered_matching_class_ids))
            != self.ordered_matching_class_ids
            or len(set(self.ordered_matching_class_ids))
            != len(self.ordered_matching_class_ids)
        ):
            _fail("REJECT_Q2_ROLE_RESULT", "matching class IDs differ")
        if self.match_count != len(self.ordered_matching_class_ids):
            _fail("REJECT_Q2_ROLE_RESULT", "match count differs")
        expected = (
            Q2_IN_VERDICT.decode("ascii")
            if self.match_count
            else Q2_OUTSIDE_VERDICT.decode("ascii")
        )
        if self.membership_verdict != expected:
            _fail("REJECT_Q2_ROLE_RESULT", "membership verdict differs")

    @property
    def result_root_id(self) -> str:
        return "sha256:" + self.result_root.hex()


@dataclass(frozen=True, slots=True)
class Q2ArtifactProjectionV1:
    capsule_digest: bytes
    authorization_digest: bytes
    consumption_record_digest: bytes
    q1_completion_receipt_root: bytes
    q1_closure_bundle_root: bytes
    q1_dual_replay_agreement_root: bytes
    q1_target_blind_access_ledger_root: bytes
    role_results: tuple[Q2RoleProjectionV1, Q2RoleProjectionV1]
    status: str
    observed_wall_time_ns: int
    observed_peak_rss_bytes: int
    artifact_root: bytes
    canonical_object: tuple[object, ...]
    canonical_bytes: bytes

    def __post_init__(self) -> None:
        for name in (
            "capsule_digest",
            "authorization_digest",
            "consumption_record_digest",
            "q1_completion_receipt_root",
            "q1_closure_bundle_root",
            "q1_dual_replay_agreement_root",
            "q1_target_blind_access_ledger_root",
            "artifact_root",
        ):
            _bytes32(getattr(self, name), name)
        if (
            type(self.role_results) is not tuple
            or len(self.role_results) != 2
            or tuple(row.role_id for row in self.role_results) != (1, 2)
        ):
            _fail("REJECT_Q2_ARTIFACT", "role order differs")
        if self.status != Q2_PASS_STATUS.decode("ascii"):
            _fail("REJECT_Q2_ARTIFACT", "Q2 predecessor is not PASS")
        if (
            type(self.observed_wall_time_ns) is not int
            or self.observed_wall_time_ns < 1
            or type(self.observed_peak_rss_bytes) is not int
            or self.observed_peak_rss_bytes < 1
        ):
            _fail("REJECT_Q2_ARTIFACT", "resource audit differs")


def _decode_q2_role_result(value: object) -> Q2RoleProjectionV1:
    if (
        type(value) is not tuple
        or len(value) != 14
        or value[:3]
        != (1, Q2_ROLE_RESULT_TAG, Q2_ROLE_RESULT_SCHEMA_ID)
        or value[3] not in (1, 2)
        or value[4] != value[3]
        or type(value[10]) is not int
        or type(value[11]) is not tuple
        or type(value[12]) is not int
    ):
        _fail("REJECT_Q2_ROLE_RESULT", "role result wire differs")
    for index, label in (
        (5, "universe root"),
        (6, "truth root"),
        (7, "signature manifest root"),
        (8, "saturation root"),
        (9, "class archive root"),
    ):
        _bytes32(value[index], label)
    if any(type(item) is not bytes or len(item) != 32 for item in value[11]):
        _fail("REJECT_Q2_ROLE_RESULT", "matching class ID wire differs")
    verdict = _ascii(value[13], "membership verdict")
    expected_match_count = len(value[11])
    if value[12] != expected_match_count:
        _fail("REJECT_Q2_ROLE_RESULT", "encoded match count differs")
    result_root = content_hash(Q2_ROLE_RESULT_ROOT_DOMAIN, value)
    return Q2RoleProjectionV1(
        role_id=value[3],
        input_signature_id=value[4],
        universe_root=value[5],
        truth_root=value[6],
        q1_signature_manifest_root=value[7],
        q1_saturation_state_root=value[8],
        q1_class_archive_root=value[9],
        evaluated_class_count=value[10],
        ordered_matching_class_ids=value[11],
        match_count=value[12],
        membership_verdict=verdict,
        result_root=result_root,
        canonical_row=value,
    )


def decode_sealed_q2_artifact_v1(payload: bytes) -> Q2ArtifactProjectionV1:
    """Decode the sealed Q2 artifact without importing any Q2 producer."""

    if type(payload) is not bytes:
        raise TypeError("Q2 artifact payload must be exact bytes")
    try:
        value = canonical_cbor_decode(payload)
    except ValueError as error:
        _fail("REJECT_Q2_ARTIFACT", str(error))
    fixed_tail = (
        (1, 1),
        (1, 1),
        b"COMPLETE",
        True,
        False,
        True,
        None,
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
        or len(value) != 29
        or value[:3]
        != (
            1,
            Q2_MEMBERSHIP_ARTIFACT_TAG,
            Q2_MEMBERSHIP_ARTIFACT_SCHEMA_ID,
        )
        or value[10] != Q2_ROLE_MATCH_PROFILE_ID
        or type(value[11]) is not tuple
        or len(value[11]) != 2
        or value[12] != Q2_PASS_STATUS
        or value[15:] != fixed_tail
    ):
        _fail("REJECT_Q2_ARTIFACT", "artifact wire differs")
    for index, label in (
        (3, "capsule digest"),
        (4, "authorization digest"),
        (5, "consumption digest"),
        (6, "Q1 completion receipt root"),
        (7, "Q1 closure bundle root"),
        (8, "Q1 dual replay root"),
        (9, "Q1 access-ledger root"),
    ):
        _bytes32(value[index], label)
    roles = tuple(_decode_q2_role_result(row) for row in value[11])
    if roles[0].match_count != 0 or roles[1].match_count < 1:
        _fail("REJECT_Q2_ARTIFACT", "Q2 PASS role disposition differs")
    canonical = canonical_cbor_encode(value)
    if canonical != payload:
        _fail("REJECT_Q2_ARTIFACT", "artifact canonical replay differs")
    return Q2ArtifactProjectionV1(
        capsule_digest=value[3],
        authorization_digest=value[4],
        consumption_record_digest=value[5],
        q1_completion_receipt_root=value[6],
        q1_closure_bundle_root=value[7],
        q1_dual_replay_agreement_root=value[8],
        q1_target_blind_access_ledger_root=value[9],
        role_results=(roles[0], roles[1]),
        status=_ascii(value[12], "Q2 status"),
        observed_wall_time_ns=value[13],
        observed_peak_rss_bytes=value[14],
        artifact_root=content_hash(Q2_MEMBERSHIP_ARTIFACT_ROOT_DOMAIN, value),
        canonical_object=value,
        canonical_bytes=payload,
    )


@dataclass(frozen=True, slots=True)
class Q3CertificateCandidateV1:
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
        if any(
            type(value) is not bytes or len(value) != 32
            for value in self.null_matching_class_ids
        ) or len(self.null_matching_class_ids) != 1:
            _fail("REJECT_Q3_CANDIDATE", "null matching class IDs differ")
        for name in ("dsl_version", "closure_semantics_version"):
            value = getattr(self, name)
            if type(value) is not str or not value or not value.isascii():
                _fail("REJECT_Q3_CANDIDATE", f"{name} differs")

    def canonical_object(self) -> tuple[object, ...]:
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
                Q3_EQUIVALENCE_ID.encode("ascii"),
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
            Q3_SUCCESS_STATUS.encode("ascii"),
            Q3_CLAIM_LEVEL.encode("ascii"),
            True,   # producer-free read-only replay completed
            0,      # target opens in Q3
            0,      # role evaluations in Q3
            0,      # split accesses in Q3
            0,      # Q1 archive reads in Q3
            0,      # measurements in Q3
            0,      # internal Popen in Q3
            0,      # actors in Q3
            0,      # Docker in Q3
            False,  # network accessed
            False,  # formal certificate issued
            False,  # external authority
            False,  # synthesis run
            False,  # MDL run
            False,  # adoption
            False,  # promotion
            False,  # ACTIVE
            False,  # production
            False,  # current pointer
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def candidate_root(self) -> bytes:
        return content_hash(Q3_CANDIDATE_ROOT_DOMAIN, self.canonical_object())


def decode_q3_certificate_candidate_v1(payload: bytes) -> Q3CertificateCandidateV1:
    if type(payload) is not bytes:
        raise TypeError("Q3 candidate payload must be exact bytes")
    try:
        value = canonical_cbor_decode(payload)
    except ValueError as error:
        _fail("REJECT_Q3_CANDIDATE", str(error))
    fixed_tail = (
        Q3_SUCCESS_STATUS.encode("ascii"),
        Q3_CLAIM_LEVEL.encode("ascii"),
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
        or value[13][5] != Q3_EQUIVALENCE_ID.encode("ascii")
        or value[13][8:] != (418, 0)
        or value[14][0] != Q2_IN_VERDICT
        or value[14][1] != 2
        or value[14][5:7] != (89, 1)
        or type(value[14][7]) is not tuple
    ):
        _fail("REJECT_Q3_CANDIDATE", "candidate wire differs")
    candidate = Q3CertificateCandidateV1(
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
        closure_semantics_version=_ascii(
            value[13][2], "closure semantics version"
        ),
        outside_universe_root=value[13][4],
        outside_truth_root=value[13][6],
        null_universe_root=value[14][2],
        null_truth_root=value[14][3],
        null_matching_class_ids=value[14][7],
    )
    if candidate.canonical_bytes != payload:
        _fail("REJECT_Q3_CANDIDATE", "candidate replay differs")
    return candidate


__all__ = [
    "Q2ArtifactProjectionV1",
    "Q2RoleProjectionV1",
    "Q3CertificateCandidateV1",
    "Q3CertificateWireError",
    "Q3_CANDIDATE_ROOT_DOMAIN",
    "Q3_CLAIM_LEVEL",
    "Q3_EQUIVALENCE_ID",
    "Q3_SUCCESS_STATUS",
    "decode_q3_certificate_candidate_v1",
    "decode_sealed_q2_artifact_v1",
]
