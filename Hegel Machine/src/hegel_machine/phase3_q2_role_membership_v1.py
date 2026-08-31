"""Pure Phase-3A-Q2 role-membership fold over typed Q1 class records.

The caller supplies target bits explicitly.  This module does not locate,
generate, or open target/split inputs and does not publish a campaign result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, NoReturn

from . import phase3_m25_rows_v1 as _role_rows
from . import phase3_q0_quotient_contract_v1 as _q0
from . import phase3_q1_formal_archive_contract_v1 as _q1_formal
from . import phase3_q1_quotient_contract_v1 as _q1_quotient
from .phase3_q2_admission_v1 import (
    Q1CompletionAdmissionV1,
    Q2RoleCommitmentV1,
    Q2_ROLE_COMMITMENTS,
    ROLE_MATCH_PROFILE_ID,
)
from .phase3_q2_archive_reader_v1 import Q1ClassStreamV1
from .strict_cbor_v1 import canonical_cbor_decode, canonical_cbor_encode, content_hash


IN_CLOSURE = "IN_FROZEN_QUOTIENT_CLOSURE"
OUTSIDE_CLOSURE = "OUTSIDE_FROZEN_QUOTIENT_CLOSURE"
PASS_STATUS = "Q2_ROLE_MEMBERSHIP_PASS"
COMPLETE_FAIL_STATUS = "Q2_ROLE_MEMBERSHIP_COMPLETE_FAIL"

Q2_ROLE_RESULT_TAG: Final = 0x3C00
Q2_MEMBERSHIP_ARTIFACT_TAG: Final = 0x3C01
Q2_ROLE_RESULT_SCHEMA_ID: Final = b"hegel-q2-role-membership-result/1"
Q2_MEMBERSHIP_ARTIFACT_SCHEMA_ID: Final = b"hegel-q2-membership-artifact/1"
Q2_ROLE_RESULT_ROOT_DOMAIN: Final = "HEGEL/Q2/ROLE_MEMBERSHIP_RESULT/V1"
Q2_MEMBERSHIP_ARTIFACT_ROOT_DOMAIN: Final = "HEGEL/Q2/MEMBERSHIP_ARTIFACT/V1"


class Q2RoleMembershipError(ValueError):
    """The supplied class stream or target-bit vector is not exact."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Q2RoleMembershipError(code, detail)


def _target_bits(value: object, row_count: int) -> tuple[int, ...]:
    if type(value) is not tuple or len(value) != row_count or any(
        type(bit) is not int or bit not in (0, 1) for bit in value
    ):
        _fail("REJECT_Q2_TARGET_BITS", "target truth is not the exact Bit row tuple")
    return value


def verified_role_target_bits_v1(
    rows: _role_rows.TypedRoleRows,
    commitment: Q2RoleCommitmentV1,
) -> tuple[int, ...]:
    """Open and bind one role truth object; callers must invoke post-consumption."""

    if type(rows) is not _role_rows.TypedRoleRows:
        raise TypeError("rows must be TypedRoleRows")
    if type(commitment) is not Q2RoleCommitmentV1:
        raise TypeError("commitment must be Q2RoleCommitmentV1")
    expected_role_name = {1: "odd", 2: "sink"}[commitment.role_id]
    try:
        rows.validate()
    except ValueError as error:
        _fail("REJECT_Q2_TARGET_ROWS", str(error))
    if (
        rows.role_name != expected_role_name
        or rows.input_signature_id != commitment.input_signature_id
        or len(rows.universe_rows) != commitment.row_count
        or len(rows.truth_rows) != commitment.row_count
        or rows.universe_root.hex() != commitment.universe_root[7:]
        or rows.truth_root.hex() != commitment.truth_root[7:]
    ):
        _fail("REJECT_Q2_TARGET_ROWS", "role rows differ from the frozen commitment")
    return _target_bits(
        tuple(row[5] for row in rows.truth_rows),
        commitment.row_count,
    )


def q1_cell_matches_target_bit_v1(
    output_sort_id: _q1_quotient.OutputSortId,
    cell: _q1_formal.Q1BehaviorCellV1,
    target_bit: int,
) -> bool:
    """Bridge one Q1 cell to the already-qualified Q0 matcher exactly."""

    if not isinstance(output_sort_id, _q1_quotient.OutputSortId):
        raise TypeError("output_sort_id must be the Q1 OutputSortId")
    if type(cell) is not _q1_formal.Q1BehaviorCellV1:
        raise TypeError("cell must be Q1BehaviorCellV1")
    try:
        q0_sort = _q0.OutputSortId(int(output_sort_id))
        q0_cell = (
            _q0.BehaviorCellV1.exact(cell.value)
            if cell.defined
            else _q0.BehaviorCellV1.bottom()
        )
        q1_wire = canonical_cbor_encode(cell.canonical_object(output_sort_id))
        q0_wire = canonical_cbor_encode(q0_cell.canonical_object(q0_sort))
    except ValueError as error:
        _fail("REJECT_Q2_CELL_BRIDGE", str(error))
    if q1_wire != q0_wire:
        _fail("REJECT_Q2_CELL_BRIDGE", "Q1 and matcher cell wires differ")
    return _q0.bool_bit_role_match_v1(q0_sort, q0_cell, target_bit)


def q1_class_matches_truth_v1(
    record: _q1_formal.Q1QuotientClassRecordV1,
    target_bits: tuple[int, ...],
) -> bool:
    """Return true only when every row matches under the frozen profile."""

    if type(record) is not _q1_formal.Q1QuotientClassRecordV1:
        raise TypeError("record must be Q1QuotientClassRecordV1")
    bits = _target_bits(target_bits, len(record.behavior.cells))
    output_sort_id = record.behavior.output_sort_id
    if output_sort_id not in (
        _q1_quotient.OutputSortId.BOOL,
        _q1_quotient.OutputSortId.BIT,
    ):
        return False
    return all(
        q1_cell_matches_target_bit_v1(output_sort_id, cell, bit)
        for cell, bit in zip(record.behavior.cells, bits, strict=True)
    )


@dataclass(frozen=True, slots=True)
class Q2RoleMembershipResultV1:
    commitment: Q2RoleCommitmentV1
    q1_signature_manifest_root: bytes
    q1_saturation_state_root: bytes
    q1_class_archive_root: bytes
    evaluated_class_count: int
    ordered_matching_class_ids: tuple[bytes, ...]
    membership_verdict: str

    def __post_init__(self) -> None:
        if type(self.commitment) is not Q2RoleCommitmentV1:
            _fail("REJECT_Q2_ROLE_RESULT", "role commitment type differs")
        for name in (
            "q1_signature_manifest_root",
            "q1_saturation_state_root",
            "q1_class_archive_root",
        ):
            value = getattr(self, name)
            if type(value) is not bytes or len(value) != 32:
                _fail("REJECT_Q2_ROLE_RESULT", f"{name} must be 32 bytes")
        if type(self.evaluated_class_count) is not int or self.evaluated_class_count < 1:
            _fail("REJECT_Q2_ROLE_RESULT", "evaluated class count differs")
        if type(self.ordered_matching_class_ids) is not tuple or any(
            type(value) is not bytes or len(value) != 32
            for value in self.ordered_matching_class_ids
        ):
            _fail("REJECT_Q2_ROLE_RESULT", "matching class IDs differ")
        if tuple(sorted(self.ordered_matching_class_ids)) != self.ordered_matching_class_ids:
            _fail("REJECT_Q2_ROLE_RESULT", "matching class IDs are not ordered")
        if (
            len(set(self.ordered_matching_class_ids))
            != len(self.ordered_matching_class_ids)
            or self.match_count > self.evaluated_class_count
        ):
            _fail("REJECT_Q2_ROLE_RESULT", "matching class ID cardinality differs")
        expected_verdict = IN_CLOSURE if self.ordered_matching_class_ids else OUTSIDE_CLOSURE
        if self.membership_verdict != expected_verdict:
            _fail("REJECT_Q2_ROLE_RESULT", "membership verdict differs from match count")

    @property
    def match_count(self) -> int:
        return len(self.ordered_matching_class_ids)

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q2_ROLE_RESULT_TAG,
            Q2_ROLE_RESULT_SCHEMA_ID,
            self.commitment.role_id,
            self.commitment.input_signature_id,
            bytes.fromhex(self.commitment.universe_root[7:]),
            bytes.fromhex(self.commitment.truth_root[7:]),
            self.q1_signature_manifest_root,
            self.q1_saturation_state_root,
            self.q1_class_archive_root,
            self.evaluated_class_count,
            self.ordered_matching_class_ids,
            self.match_count,
            self.membership_verdict.encode("ascii"),
        )

    @property
    def result_root(self) -> bytes:
        return content_hash(Q2_ROLE_RESULT_ROOT_DOMAIN, self.canonical_object())

    @property
    def passes_preregistered_disposition(self) -> bool:
        if self.commitment.pass_match_count_relation == "EQUALS_ZERO":
            count_passes = self.match_count == 0
        elif self.commitment.pass_match_count_relation == "GREATER_THAN_ZERO":
            count_passes = self.match_count > 0
        else:  # pragma: no cover - commitment registry closes this branch
            raise AssertionError("unregistered match-count relation")
        return (
            count_passes
            and self.membership_verdict
            == self.commitment.pass_membership_verdict
        )


def evaluate_role_membership_v1(
    class_stream: Q1ClassStreamV1,
    commitment: Q2RoleCommitmentV1,
    target_bits: tuple[int, ...],
) -> Q2RoleMembershipResultV1:
    """Evaluate one complete frozen role against one sealed Q1 class stream."""

    if type(class_stream) is not Q1ClassStreamV1:
        raise TypeError("class_stream must be Q1ClassStreamV1")
    if type(commitment) is not Q2RoleCommitmentV1:
        raise TypeError("commitment must be Q2RoleCommitmentV1")
    bits = _target_bits(target_bits, commitment.row_count)
    if (
        class_stream.input_signature_id != commitment.input_signature_id
        or class_stream.universe_root.hex() != commitment.universe_root[7:]
        or any(len(record.behavior.cells) != commitment.row_count for record in class_stream.records)
    ):
        _fail("REJECT_Q2_ROLE_BINDING", "role and Q1 class stream differ")
    matching = tuple(
        record.class_id
        for record in class_stream.records
        if q1_class_matches_truth_v1(record, bits)
    )
    verdict = IN_CLOSURE if matching else OUTSIDE_CLOSURE
    return Q2RoleMembershipResultV1(
        commitment,
        class_stream.signature_manifest_root,
        class_stream.saturation_state_root,
        class_stream.class_archive_root,
        len(class_stream.records),
        matching,
        verdict,
    )


def _decode_role_result_object_v1(value: object) -> Q2RoleMembershipResultV1:
    if (
        type(value) is not tuple
        or len(value) != 14
        or value[:3] != (1, Q2_ROLE_RESULT_TAG, Q2_ROLE_RESULT_SCHEMA_ID)
        or type(value[3]) is not int
        or value[3] not in (1, 2)
        or value[4] != value[3]
        or type(value[11]) is not tuple
        or type(value[13]) is not bytes
    ):
        _fail("REJECT_Q2_ROLE_RESULT", "role result wire differs")
    commitment = Q2_ROLE_COMMITMENTS[value[3] - 1]
    if (
        value[5] != bytes.fromhex(commitment.universe_root[7:])
        or value[6] != bytes.fromhex(commitment.truth_root[7:])
    ):
        _fail("REJECT_Q2_ROLE_RESULT", "role commitment roots differ")
    try:
        verdict = value[13].decode("ascii")
    except UnicodeDecodeError as error:
        _fail("REJECT_Q2_ROLE_RESULT", str(error))
    result = Q2RoleMembershipResultV1(
        commitment,
        value[7],
        value[8],
        value[9],
        value[10],
        value[11],
        verdict,
    )
    if value[12] != result.match_count or result.canonical_object() != value:
        _fail("REJECT_Q2_ROLE_RESULT", "role result replay differs")
    return result


@dataclass(frozen=True, slots=True)
class Q2MembershipArtifactV1:
    capsule_digest: bytes
    authorization_digest: bytes
    consumption_record_digest: bytes
    q1_completion_receipt_root: bytes
    q1_closure_bundle_root: bytes
    q1_dual_replay_agreement_root: bytes
    q1_target_blind_access_ledger_root: bytes
    role_results: tuple[Q2RoleMembershipResultV1, Q2RoleMembershipResultV1]
    observed_wall_time_ns: int
    observed_peak_rss_bytes: int

    def __post_init__(self) -> None:
        for name in (
            "capsule_digest",
            "authorization_digest",
            "consumption_record_digest",
            "q1_completion_receipt_root",
            "q1_closure_bundle_root",
            "q1_dual_replay_agreement_root",
            "q1_target_blind_access_ledger_root",
        ):
            value = getattr(self, name)
            if type(value) is not bytes or len(value) != 32:
                _fail("REJECT_Q2_ARTIFACT", f"{name} must be 32 bytes")
        if (
            type(self.role_results) is not tuple
            or len(self.role_results) != 2
            or tuple(row.commitment.role_id for row in self.role_results) != (1, 2)
        ):
            _fail("REJECT_Q2_ARTIFACT", "role result order differs")
        if (
            type(self.observed_wall_time_ns) is not int
            or self.observed_wall_time_ns < 1
            or type(self.observed_peak_rss_bytes) is not int
            or self.observed_peak_rss_bytes < 1
        ):
            _fail("REJECT_Q2_ARTIFACT", "observed resource audit differs")

    @property
    def status(self) -> str:
        return combine_q2_role_results_v1(*self.role_results)

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q2_MEMBERSHIP_ARTIFACT_TAG,
            Q2_MEMBERSHIP_ARTIFACT_SCHEMA_ID,
            self.capsule_digest,
            self.authorization_digest,
            self.consumption_record_digest,
            self.q1_completion_receipt_root,
            self.q1_closure_bundle_root,
            self.q1_dual_replay_agreement_root,
            self.q1_target_blind_access_ledger_root,
            ROLE_MATCH_PROFILE_ID.encode("ascii"),
            tuple(row.canonical_object() for row in self.role_results),
            self.status.encode("ascii"),
            self.observed_wall_time_ns,
            self.observed_peak_rss_bytes,
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

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def artifact_root(self) -> bytes:
        return content_hash(
            Q2_MEMBERSHIP_ARTIFACT_ROOT_DOMAIN,
            self.canonical_object(),
        )


def build_q2_membership_artifact_v1(
    capsule_digest: bytes,
    authorization_digest: bytes,
    consumption_record_digest: bytes,
    admission: Q1CompletionAdmissionV1,
    outside: Q2RoleMembershipResultV1,
    null_control: Q2RoleMembershipResultV1,
    *,
    observed_wall_time_ns: int,
    observed_peak_rss_bytes: int,
) -> Q2MembershipArtifactV1:
    if type(admission) is not Q1CompletionAdmissionV1:
        raise TypeError("admission must be Q1CompletionAdmissionV1")
    return Q2MembershipArtifactV1(
        capsule_digest,
        authorization_digest,
        consumption_record_digest,
        admission.ordered_output_slots[7][2],
        admission.ordered_output_slots[4][2],
        admission.ordered_output_slots[5][2],
        admission.ordered_output_slots[6][2],
        (outside, null_control),
        observed_wall_time_ns,
        observed_peak_rss_bytes,
    )


def decode_q2_membership_artifact_v1(payload: bytes) -> Q2MembershipArtifactV1:
    if type(payload) is not bytes:
        _fail("REJECT_Q2_ARTIFACT", "artifact must be exact bytes")
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
        != (1, Q2_MEMBERSHIP_ARTIFACT_TAG, Q2_MEMBERSHIP_ARTIFACT_SCHEMA_ID)
        or value[10] != ROLE_MATCH_PROFILE_ID.encode("ascii")
        or type(value[11]) is not tuple
        or len(value[11]) != 2
        or value[15:] != fixed_tail
        or type(value[12]) is not bytes
    ):
        _fail("REJECT_Q2_ARTIFACT", "artifact wire differs")
    results = tuple(_decode_role_result_object_v1(row) for row in value[11])
    artifact = Q2MembershipArtifactV1(
        value[3],
        value[4],
        value[5],
        value[6],
        value[7],
        value[8],
        value[9],
        results,
        value[13],
        value[14],
    )
    if value[12] != artifact.status.encode("ascii") or artifact.canonical_bytes != payload:
        _fail("REJECT_Q2_ARTIFACT", "artifact replay differs")
    return artifact


def combine_q2_role_results_v1(
    outside: Q2RoleMembershipResultV1,
    null_control: Q2RoleMembershipResultV1,
) -> str:
    """Project the two complete role results to the preregistered Q2 status."""

    if type(outside) is not Q2RoleMembershipResultV1 or type(
        null_control
    ) is not Q2RoleMembershipResultV1:
        raise TypeError("both role results must be Q2RoleMembershipResultV1")
    if (
        outside.commitment.role_id != 1
        or null_control.commitment.role_id != 2
    ):
        _fail("REJECT_Q2_ROLE_ORDER", "role results are not in frozen order")
    return (
        PASS_STATUS
        if outside.passes_preregistered_disposition
        and null_control.passes_preregistered_disposition
        else COMPLETE_FAIL_STATUS
    )


__all__ = [
    "COMPLETE_FAIL_STATUS",
    "IN_CLOSURE",
    "OUTSIDE_CLOSURE",
    "PASS_STATUS",
    "Q2MembershipArtifactV1",
    "Q2RoleMembershipError",
    "Q2RoleMembershipResultV1",
    "build_q2_membership_artifact_v1",
    "combine_q2_role_results_v1",
    "decode_q2_membership_artifact_v1",
    "evaluate_role_membership_v1",
    "q1_cell_matches_target_bit_v1",
    "q1_class_matches_truth_v1",
    "verified_role_target_bits_v1",
]
