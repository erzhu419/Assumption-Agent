"""Target-blind formal archive wire for the Phase-3A Q1 quotient closure.

This module freezes value identity, strict CBOR arrays, ordering, archive
framing, and root-DAG wire shape/local self-hash validation.  The strict
partition/manifest/bundle DAG assembler remains pending.  Importing or
exercising this module does not start Q1, read target truth or split state,
populate a formal output slot, or issue a certificate.  Production roots remain
null until a separately admitted Q1 run.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
from hashlib import sha256
from typing import Final, Iterable, NoReturn, Sequence

from .phase3_q1_quotient_contract_v1 import (
    FutureAdmissibilitySignatureV1,
    NormalizationProfileId,
    OutputSortId,
    normalization_witness_capacity_v1,
)
from .phase3_q1_universe_v1 import production_universe_v1
from .strict_ast_shrink6_v1 import decode_shrink6_canonical_ast
from .strict_cbor_v1 import (
    canonical_cbor_decode,
    canonical_cbor_encode,
    content_hash,
    rfc6962_root,
)


DSL_VERSION: Final = "hegel-old-dsl-v1.6.0"
DSL_FREEZE_VERSION: Final = "hegel-freeze-p2b-p3-v1.6.0"
CLOSURE_SEMANTICS_VERSION: Final = "hegel-quotient-closure-v1.0.1"
ARCHIVE_WIRE_VERSION: Final = "hegel-q1-archive-wire-v1.0.0"
PROJECTION_FREEZE_VERSION: Final = (
    "hegel-freeze-p3a-q05a-q1-projection-v1.0.0"
)
PROJECTION_PROFILE_ID: Final = "hegel-q1-archive-projection-profile-v1"

Q1_SEMANTIC_BINDING_TAG: Final = 0x3700
Q1_BEHAVIOR_BLOB_TAG: Final = 0x3701
Q1_CONSTRUCTION_SIGNATURE_TAG: Final = 0x3702
Q1_REPRESENTATIVE_PROGRAM_TAG: Final = 0x3703
Q1_CONTINUATION_COHORT_TAG: Final = 0x3704
Q1_QUOTIENT_CLASS_TAG: Final = 0x3705
Q1_SEMANTIC_COVERAGE_TAG: Final = 0x3706
Q1_FIXED_POINT_TAG: Final = 0x3707
Q1_ARCHIVE_CHUNK_MANIFEST_TAG: Final = 0x3708
Q1_SIGNATURE_ARCHIVE_MANIFEST_TAG: Final = 0x3709
Q1_CLOSURE_BUNDLE_TAG: Final = 0x370A
Q1_ARCHIVE_PROJECTION_PROFILE_TAG: Final = 0x370B
Q1_ARCHIVE_PROJECTION_RESULT_TAG: Final = 0x370C

Q1_TAG_REGISTRY: Final = (
    (Q1_SEMANTIC_BINDING_TAG, b"Q1_SEMANTIC_BINDING_MANIFEST"),
    (Q1_BEHAVIOR_BLOB_TAG, b"Q1_BEHAVIOR_BLOB"),
    (Q1_CONSTRUCTION_SIGNATURE_TAG, b"Q1_CONSTRUCTION_SIGNATURE"),
    (Q1_REPRESENTATIVE_PROGRAM_TAG, b"Q1_REPRESENTATIVE_PROGRAM_RECORD"),
    (Q1_CONTINUATION_COHORT_TAG, b"Q1_CONTINUATION_COHORT_RECORD"),
    (Q1_QUOTIENT_CLASS_TAG, b"Q1_QUOTIENT_CLASS_RECORD"),
    (Q1_SEMANTIC_COVERAGE_TAG, b"Q1_SEMANTIC_COVERAGE_RECORD"),
    (Q1_FIXED_POINT_TAG, b"Q1_FIXED_POINT_RECORD"),
    (Q1_ARCHIVE_CHUNK_MANIFEST_TAG, b"Q1_ARCHIVE_CHUNK_MANIFEST"),
    (Q1_SIGNATURE_ARCHIVE_MANIFEST_TAG, b"Q1_SIGNATURE_ARCHIVE_MANIFEST"),
    (Q1_CLOSURE_BUNDLE_TAG, b"Q1_CLOSURE_BUNDLE"),
    (Q1_ARCHIVE_PROJECTION_PROFILE_TAG, b"Q1_ARCHIVE_PROJECTION_PROFILE"),
    (Q1_ARCHIVE_PROJECTION_RESULT_TAG, b"Q1_ARCHIVE_PROJECTION_RESULT"),
)

SEMANTIC_BINDING_SCHEMA_ID: Final = b"hegel-q1-semantic-binding-manifest/1"
BEHAVIOR_BLOB_SCHEMA_ID: Final = b"hegel-q1-behavior-blob/1"
CONSTRUCTION_SIGNATURE_SCHEMA_ID: Final = b"hegel-q1-construction-signature/1"
REPRESENTATIVE_PROGRAM_SCHEMA_ID: Final = b"hegel-q1-representative-program/1"
CONTINUATION_COHORT_SCHEMA_ID: Final = b"hegel-q1-continuation-cohort/1"
QUOTIENT_CLASS_SCHEMA_ID: Final = b"hegel-q1-quotient-class/1"
SEMANTIC_COVERAGE_SCHEMA_ID: Final = b"hegel-q1-semantic-coverage/1"
FIXED_POINT_SCHEMA_ID: Final = b"hegel-q1-fixed-point/1"
ARCHIVE_CHUNK_MANIFEST_SCHEMA_ID: Final = b"hegel-q1-archive-chunk-manifest/1"
SIGNATURE_ARCHIVE_MANIFEST_SCHEMA_ID: Final = (
    b"hegel-q1-signature-archive-manifest/1"
)
CLOSURE_BUNDLE_SCHEMA_ID: Final = b"hegel-q1-closure-bundle/1"
ARCHIVE_PROJECTION_PROFILE_SCHEMA_ID: Final = (
    b"hegel-q1-archive-projection-profile/1"
)
ARCHIVE_PROJECTION_RESULT_SCHEMA_ID: Final = (
    b"hegel-q1-archive-projection-result/1"
)
PROJECTION_PARTITION_ROW_SCHEMA_ID: Final = (
    b"hegel-q1-archive-projection-partition-row/1"
)
PARTITION_STREAM_COMMITMENT_SCHEMA_ID: Final = (
    b"hegel-q1-partition-stream-commitment/1"
)
PARTITION_EXTERNAL_SORT_SCHEMA_ID: Final = (
    b"hegel-q1-partition-external-sort-projection/1"
)
APPLICATION_KEY_SCHEMA_ID: Final = b"hegel-q1-semantic-application-key/1"
STREAM_DESCRIPTOR_SCHEMA_ID: Final = b"hegel-q1-stream-descriptor/1"

SEMANTIC_BINDING_ROOT_DOMAIN: Final = "HEGEL/Q1/SEMANTIC_BINDING/V1"
BEHAVIOR_ID_DOMAIN: Final = "HEGEL/Q1/BEHAVIOR_ID/V1"
CONSTRUCTION_SIGNATURE_ID_DOMAIN: Final = "HEGEL/Q1/CONSTRUCTION_SIGNATURE_ID/V1"
PROGRAM_ID_DOMAIN: Final = "HEGEL/Q1/PROGRAM_ID/V1"
PROGRAM_RECORD_ID_DOMAIN: Final = "HEGEL/Q1/PROGRAM_RECORD_ID/V1"
COHORT_ID_DOMAIN: Final = "HEGEL/Q1/COHORT_ID/V1"
COHORT_RECORD_ID_DOMAIN: Final = "HEGEL/Q1/COHORT_RECORD_ID/V1"
CLASS_RECORD_ID_DOMAIN: Final = "HEGEL/Q1/CLASS_RECORD_ID/V1"
APPLICATION_ID_DOMAIN: Final = "HEGEL/Q1/APPLICATION_ID/V1"
COVERAGE_RECORD_ID_DOMAIN: Final = "HEGEL/Q1/COVERAGE_RECORD_ID/V1"
FIXED_POINT_ROOT_DOMAIN: Final = "HEGEL/Q1/FIXED_POINT_RECORD/V1"
FRAMED_BLOB_HASH_DOMAIN: Final = "HEGEL/Q1/FRAMED_BLOB/V1"
CHUNK_MANIFEST_RECORD_ID_DOMAIN: Final = "HEGEL/Q1/CHUNK_MANIFEST_RECORD_ID/V1"
SIGNATURE_MANIFEST_ROOT_DOMAIN: Final = "HEGEL/Q1/SIGNATURE_ARCHIVE_MANIFEST/V1"
SIGNATURE_SATURATION_STATE_ROOT_DOMAIN: Final = (
    "HEGEL/Q1/SIGNATURE_SATURATION_STATE/V1"
)
CLOSURE_BUNDLE_ROOT_DOMAIN: Final = "HEGEL/Q1/CLOSURE_BUNDLE/V1"
PROJECTION_PROFILE_ROOT_DOMAIN: Final = "HEGEL/Q1/ARCHIVE_PROJECTION_PROFILE/V1"
PROJECTION_RESULT_ROOT_DOMAIN: Final = "HEGEL/Q1/PREFLIGHT/PROJECTION_RESULT/V1"
PARTITION_STREAM_COMMITMENT_DOMAIN: Final = (
    "HEGEL/Q1/PREFLIGHT/PARTITION_STREAM_COMMITMENT/V1"
)
PARTITION_EXTERNAL_SORT_ROOT_DOMAIN: Final = (
    "HEGEL/Q1/PREFLIGHT/PARTITION_EXTERNAL_SORT/V1"
)

MAX_RECORDS_PER_CHUNK: Final = 4096
MAX_CHUNK_FRAMED_BYTES: Final = 16_777_216
FRAME_LENGTH_BYTES: Final = 4
COMPRESSION_ID_NONE: Final = 0
ENDPOINT_RUN_METADATA_RESERVATION_BYTES: Final = 1_048_576
HOST_RUN_METADATA_RESERVATION_BYTES: Final = 1_048_576
MAX_RUN_METADATA_FRAME_COUNT: Final = 64
MAX_RUN_METADATA_FRAME_BYTES: Final = 16_384
EXTERNAL_SORT_RUN_PAYLOAD_LIMIT_BYTES: Final = 268_435_456
EXTERNAL_SORT_MERGE_FAN_IN: Final = 16
EXTERNAL_SORT_RUN_MAGIC: Final = b"HGQ1RUN1"
EXTERNAL_SORT_RUN_HEADER_BYTES: Final = 68
EXTERNAL_SORT_ROW_LENGTH_BYTES: Final = 4
SCRATCH_ALLOCATION_BLOCK_BYTES: Final = 4096
SCRATCH_METADATA_RESERVE_BYTES_PER_LIVE_FILE: Final = 4096
SCRATCH_NONSPARSE_REQUIRED: Final = True
EXTERNAL_SORT_STREAM_ORDER: Final = (1, 2, 3, 4)
EXTERNAL_SORT_SIGNATURE_ORDER: Final = (1, 2)

LEAF_COVERAGE_CODES: Final = tuple(range(0x0000, 0x032A))
OPERATOR_COVERAGE_CODES: Final = (
    0x1000,
    0x1001,
    0x1002,
    0x1003,
    0x2001,
    0x2002,
    0x2003,
    0x2005,
    0x2006,
    0x3001,
    0x3002,
    0x4002,
)
CONSTRUCTION_DEPTHS: Final = (1, 2, 3)
EXPECTED_COVERAGE_RECORD_COUNT: Final = 810 + 12 * 3

Q1_OUTPUT_SLOT_NAMES: Final = (
    b"odd_signature_archive_manifest_root",
    b"odd_signature_saturation_state_root",
    b"sink_signature_archive_manifest_root",
    b"sink_signature_saturation_state_root",
    b"q1_closure_bundle_root",
    b"q1_dual_replay_agreement_root",
    b"q1_target_blind_access_ledger_root",
    b"q1_completion_receipt_root",
)

Q1_RESOURCE_GUARD_REGISTRY: Final = (
    (1, b"RAW_OPERATOR_APPLICATIONS"),
    (2, b"BEHAVIOR_CLASSES"),
    (3, b"VISIBLE_FRONTIER_TOTAL"),
    (4, b"VISIBLE_FRONTIER_PER_CLASS"),
    (5, b"CONTINUATION_BANK_TOTAL"),
    (6, b"CONTINUATION_BANK_PER_CLASS"),
    (7, b"WORK_QUEUE_POINTS"),
    (8, b"SATURATION_ROUNDS"),
    (9, b"OUTPUT_BYTES"),
    (10, b"SCRATCH_BYTES"),
    (11, b"RESIDENT_MEMORY"),
    (12, b"WALL_TIME"),
)

RATIONAL_VALUE_GRID: Final = frozenset(
    Fraction(numerator, denominator)
    for numerator in range(-64, 65)
    for denominator in range(1, 9)
)


class ArchiveStreamKindId(IntEnum):
    PROGRAM = 1
    COHORT = 2
    CLASS = 3
    COVERAGE = 4


class CoverageCodeKindId(IntEnum):
    LEAF = 1
    OPERATOR = 2


class Q1ArchiveContractError(ValueError):
    """Stable fail-closed rejection from the Q1 formal archive contract."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Q1ArchiveContractError(code, detail)


def _uint(value: object, name: str, maximum: int = (1 << 64) - 1) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        _fail("REJECT_Q1_ARCHIVE_UINT", f"{name} is outside uint range")
    return value


def _bytes(value: object, name: str, length: int | None = None) -> bytes:
    if type(value) is not bytes or (length is not None and len(value) != length):
        suffix = " bytes" if length is None else f" exactly {length} bytes"
        _fail("REJECT_Q1_ARCHIVE_BYTES", f"{name} must be{suffix}")
    return value


def _root32(value: object, name: str) -> bytes:
    return _bytes(value, name, 32)


def _input_binding(input_signature_id: object, universe_root: object) -> tuple[int, bytes]:
    if type(input_signature_id) is not int or input_signature_id not in (1, 2):
        _fail("REJECT_Q1_INPUT_SIGNATURE", "input signature must be exact int 1 or 2")
    root = _root32(universe_root, "universe_root")
    expected = production_universe_v1(input_signature_id).universe_root
    if root != expected:
        _fail("REJECT_Q1_UNIVERSE_BINDING", "universe root differs from frozen rows")
    return input_signature_id, root


def _strict_replay(payload: bytes, name: str) -> tuple[object, ...]:
    _bytes(payload, name)
    try:
        value = canonical_cbor_decode(payload)
    except ValueError as error:
        _fail("REJECT_Q1_ARCHIVE_CBOR", f"{name}: {error}")
    if type(value) is not tuple or canonical_cbor_encode(value) != payload:
        _fail("REJECT_Q1_ARCHIVE_CBOR", f"{name} is not canonical array CBOR")
    return value


@dataclass(frozen=True, slots=True)
class Q1BehaviorCellV1:
    defined: bool
    value: object = None

    @classmethod
    def bottom(cls) -> "Q1BehaviorCellV1":
        return cls(False, None)

    @classmethod
    def exact(cls, value: object) -> "Q1BehaviorCellV1":
        return cls(True, value)

    def canonical_object(self, output_sort_id: OutputSortId) -> tuple[object, ...]:
        if type(self.defined) is not bool:
            _fail("REJECT_Q1_BEHAVIOR_CELL", "defined must be exact bool")
        if not self.defined:
            if self.value is not None:
                _fail("REJECT_Q1_BEHAVIOR_CELL", "bottom cannot carry payload")
            return (0,)
        value = self.value
        if output_sort_id is OutputSortId.BOOL:
            if type(value) is not bool:
                _fail("REJECT_Q1_BEHAVIOR_CELL", "Bool requires exact bool")
            payload: object = value
        elif output_sort_id is OutputSortId.BIT:
            if type(value) is not int or value not in (0, 1):
                _fail("REJECT_Q1_BEHAVIOR_CELL", "Bit requires exact int 0 or 1")
            payload = value
        elif output_sort_id is OutputSortId.SIGN:
            if type(value) is not int or value not in (-1, 0, 1):
                _fail("REJECT_Q1_BEHAVIOR_CELL", "Sign is outside -1,0,1")
            payload = value
        elif output_sort_id is OutputSortId.BOUNDED_INT:
            if type(value) is not int or not -8 <= value <= 8:
                _fail("REJECT_Q1_BEHAVIOR_CELL", "BoundedInt is outside [-8,8]")
            payload = value
        elif output_sort_id is OutputSortId.RATIONAL_VALUE:
            if type(value) is not Fraction or value not in RATIONAL_VALUE_GRID:
                _fail("REJECT_Q1_BEHAVIOR_CELL", "RationalValue is outside exact grid")
            payload = (value.numerator, value.denominator)
        else:
            _fail("REJECT_Q1_BEHAVIOR_CELL", "unregistered output sort")
        return (1, payload)


@dataclass(frozen=True, slots=True)
class Q1BehaviorBlobV1:
    input_signature_id: int
    universe_root: bytes
    output_sort_id: OutputSortId
    cells: tuple[Q1BehaviorCellV1, ...]

    def __post_init__(self) -> None:
        _input_binding(self.input_signature_id, self.universe_root)
        if not isinstance(self.output_sort_id, OutputSortId):
            _fail("REJECT_Q1_BEHAVIOR_BLOB", "output sort is unregistered")
        expected_rows = len(production_universe_v1(self.input_signature_id).rows)
        if type(self.cells) is not tuple or len(self.cells) != expected_rows:
            _fail("REJECT_Q1_BEHAVIOR_BLOB", "behavior row count differs")
        if any(type(cell) is not Q1BehaviorCellV1 for cell in self.cells):
            _fail("REJECT_Q1_BEHAVIOR_BLOB", "cells must be exact Q1BehaviorCellV1")
        for cell in self.cells:
            cell.canonical_object(self.output_sort_id)

    def canonical_object(self) -> tuple[object, ...]:
        cells = tuple(cell.canonical_object(self.output_sort_id) for cell in self.cells)
        return (
            1,
            Q1_BEHAVIOR_BLOB_TAG,
            BEHAVIOR_BLOB_SCHEMA_ID,
            self.input_signature_id,
            self.universe_root,
            int(self.output_sort_id),
            len(cells),
            cells,
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def behavior_id(self) -> bytes:
        return content_hash(BEHAVIOR_ID_DOMAIN, self.canonical_object())


def construction_signature_object_v1(
    signature: FutureAdmissibilitySignatureV1,
) -> tuple[object, ...]:
    if type(signature) is not FutureAdmissibilitySignatureV1:
        raise TypeError("signature must be FutureAdmissibilitySignatureV1")
    return (
        1,
        Q1_CONSTRUCTION_SIGNATURE_TAG,
        CONSTRUCTION_SIGNATURE_SCHEMA_ID,
        *signature.resource_tuple(),
    )


def construction_signature_id_v1(signature: FutureAdmissibilitySignatureV1) -> bytes:
    return content_hash(
        CONSTRUCTION_SIGNATURE_ID_DOMAIN,
        construction_signature_object_v1(signature),
    )


@dataclass(frozen=True, slots=True)
class Q1RepresentativeProgramRecordV1:
    input_signature_id: int
    universe_root: bytes
    program_index: int
    class_id: bytes
    canonical_ast_cbor: bytes
    canonical_ast_hash: bytes
    construction_signature: FutureAdmissibilitySignatureV1

    def __post_init__(self) -> None:
        _input_binding(self.input_signature_id, self.universe_root)
        _uint(self.program_index, "program_index", 0xFFFFFFFF)
        _root32(self.class_id, "class_id")
        _bytes(self.canonical_ast_cbor, "canonical_ast_cbor")
        _root32(self.canonical_ast_hash, "canonical_ast_hash")
        try:
            replay = decode_shrink6_canonical_ast(self.canonical_ast_cbor)
        except ValueError as error:
            _fail("REJECT_Q1_PROGRAM_AST", str(error))
        if replay.digest != self.canonical_ast_hash:
            _fail("REJECT_Q1_PROGRAM_AST_HASH", "AST hash differs from strict replay")
        from .phase3_q1_quotient_contract_v1 import future_signature_from_ast_v1

        if future_signature_from_ast_v1(replay) != self.construction_signature:
            _fail("REJECT_Q1_PROGRAM_SIGNATURE", "signature differs from AST replay")

    @property
    def signature_id(self) -> bytes:
        return construction_signature_id_v1(self.construction_signature)

    @property
    def program_id(self) -> bytes:
        return content_hash(
            PROGRAM_ID_DOMAIN,
            (
                self.input_signature_id,
                self.universe_root,
                self.canonical_ast_cbor,
                self.canonical_ast_hash,
                construction_signature_object_v1(self.construction_signature),
            ),
        )

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q1_REPRESENTATIVE_PROGRAM_TAG,
            REPRESENTATIVE_PROGRAM_SCHEMA_ID,
            self.input_signature_id,
            self.universe_root,
            self.program_index,
            self.program_id,
            self.class_id,
            self.canonical_ast_cbor,
            self.canonical_ast_hash,
            construction_signature_object_v1(self.construction_signature),
            self.signature_id,
        )

    @property
    def record_id(self) -> bytes:
        return content_hash(PROGRAM_RECORD_ID_DOMAIN, self.canonical_object())

    @property
    def sort_key(self) -> tuple[object, ...]:
        replay = decode_shrink6_canonical_ast(self.canonical_ast_cbor)
        return (
            replay.metrics.depth,
            replay.metrics.node_count,
            int(self.construction_signature.output_sort_id),
            replay.root_operator_id,
            self.canonical_ast_cbor,
        )


@dataclass(frozen=True, slots=True)
class Q1CohortWitnessV1:
    rank: int
    program_id: bytes
    canonical_ast_hash: bytes

    def canonical_object(self) -> tuple[object, ...]:
        _uint(self.rank, "witness_rank", 0xFF)
        _root32(self.program_id, "program_id")
        _root32(self.canonical_ast_hash, "canonical_ast_hash")
        return (self.rank, self.program_id, self.canonical_ast_hash)


@dataclass(frozen=True, slots=True)
class Q1ContinuationCohortRecordV1:
    input_signature_id: int
    universe_root: bytes
    cohort_index: int
    class_id: bytes
    construction_signature: FutureAdmissibilitySignatureV1
    witnesses: tuple[Q1CohortWitnessV1, ...]
    visible_frontier_cohort: bool

    def __post_init__(self) -> None:
        _input_binding(self.input_signature_id, self.universe_root)
        _uint(self.cohort_index, "cohort_index", 0xFFFFFFFF)
        _root32(self.class_id, "class_id")
        if type(self.witnesses) is not tuple or not self.witnesses:
            _fail("REJECT_Q1_COHORT", "cohort witnesses must be nonempty tuple")
        capacity = normalization_witness_capacity_v1(
            self.construction_signature.output_sort_id
        )
        if len(self.witnesses) > capacity:
            _fail("REJECT_Q1_COHORT", "witness count exceeds sort capacity")
        rows = tuple(witness.canonical_object() for witness in self.witnesses)
        if tuple(row[0] for row in rows) != tuple(range(len(rows))):
            _fail("REJECT_Q1_COHORT", "witness ranks are not contiguous")
        if len({row[1] for row in rows}) != len(rows):
            _fail("REJECT_Q1_COHORT", "duplicate witness program")
        if type(self.visible_frontier_cohort) is not bool:
            _fail("REJECT_Q1_COHORT", "visible frontier flag must be exact bool")

    @property
    def signature_id(self) -> bytes:
        return construction_signature_id_v1(self.construction_signature)

    @property
    def cohort_id(self) -> bytes:
        return content_hash(
            COHORT_ID_DOMAIN,
            (
                self.input_signature_id,
                self.universe_root,
                self.class_id,
                self.signature_id,
            ),
        )

    def canonical_object(self) -> tuple[object, ...]:
        capacity = normalization_witness_capacity_v1(
            self.construction_signature.output_sort_id
        )
        return (
            1,
            Q1_CONTINUATION_COHORT_TAG,
            CONTINUATION_COHORT_SCHEMA_ID,
            self.input_signature_id,
            self.universe_root,
            self.cohort_index,
            self.cohort_id,
            self.class_id,
            construction_signature_object_v1(self.construction_signature),
            self.signature_id,
            capacity,
            len(self.witnesses),
            tuple(witness.canonical_object() for witness in self.witnesses),
            self.visible_frontier_cohort,
        )

    @property
    def record_id(self) -> bytes:
        return content_hash(COHORT_RECORD_ID_DOMAIN, self.canonical_object())


@dataclass(frozen=True, slots=True)
class Q1QuotientClassRecordV1:
    input_signature_id: int
    universe_root: bytes
    class_index: int
    behavior: Q1BehaviorBlobV1
    first_cohort_index: int
    cohort_count: int
    class_cohort_subtree_root: bytes
    bank_point_count: int
    visible_cohort_count: int
    visible_frontier_point_count: int
    visible_frontier_subtree_root: bytes
    minimum_mdl_q32: int

    def __post_init__(self) -> None:
        _input_binding(self.input_signature_id, self.universe_root)
        _uint(self.class_index, "class_index", 0xFFFFFFFF)
        if type(self.behavior) is not Q1BehaviorBlobV1:
            _fail("REJECT_Q1_CLASS", "behavior must be exact Q1BehaviorBlobV1")
        if (
            self.behavior.input_signature_id != self.input_signature_id
            or self.behavior.universe_root != self.universe_root
        ):
            _fail("REJECT_Q1_CLASS", "behavior binding differs from class")
        _uint(self.first_cohort_index, "first_cohort_index", 0xFFFFFFFF)
        _uint(self.cohort_count, "cohort_count", 0xFFFFFFFF)
        if self.cohort_count < 1:
            _fail("REJECT_Q1_CLASS", "class must contain at least one cohort")
        _root32(self.class_cohort_subtree_root, "class_cohort_subtree_root")
        _uint(self.bank_point_count, "bank_point_count", 0xFFFFFFFF)
        _uint(self.visible_cohort_count, "visible_cohort_count", 0xFFFFFFFF)
        _uint(self.visible_frontier_point_count, "visible_frontier_point_count", 0xFFFFFFFF)
        _root32(self.visible_frontier_subtree_root, "visible_frontier_subtree_root")
        _uint(self.minimum_mdl_q32, "minimum_mdl_q32")
        if not 1 <= self.bank_point_count <= 2 * self.cohort_count:
            _fail("REJECT_Q1_CLASS", "bank point count is outside cohort capacity")
        if not 1 <= self.visible_cohort_count <= self.cohort_count:
            _fail("REJECT_Q1_CLASS", "visible cohort count is invalid")
        if not self.visible_cohort_count <= self.visible_frontier_point_count <= self.bank_point_count:
            _fail("REJECT_Q1_CLASS", "visible frontier point count is invalid")

    @property
    def class_id(self) -> bytes:
        return self.behavior.behavior_id

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q1_QUOTIENT_CLASS_TAG,
            QUOTIENT_CLASS_SCHEMA_ID,
            self.input_signature_id,
            self.universe_root,
            self.class_index,
            self.behavior.canonical_object(),
            self.class_id,
            self.first_cohort_index,
            self.cohort_count,
            self.class_cohort_subtree_root,
            self.bank_point_count,
            self.visible_cohort_count,
            self.visible_frontier_point_count,
            self.visible_frontier_subtree_root,
            self.minimum_mdl_q32,
        )

    @property
    def record_id(self) -> bytes:
        return content_hash(CLASS_RECORD_ID_DOMAIN, self.canonical_object())


def semantic_application_key_v1(
    input_signature_id: int,
    universe_root: bytes,
    construction_depth: int,
    coverage_code: int,
    operator_parameters: tuple[object, ...],
    ordered_child_program_ids: tuple[bytes, ...],
) -> tuple[object, ...]:
    _input_binding(input_signature_id, universe_root)
    _uint(construction_depth, "construction_depth", 3)
    _uint(coverage_code, "coverage_code", 0xFFFF)
    if type(operator_parameters) is not tuple:
        _fail("REJECT_Q1_APPLICATION_KEY", "operator parameters must be tuple")
    if type(ordered_child_program_ids) is not tuple or any(
        type(value) is not bytes or len(value) != 32
        for value in ordered_child_program_ids
    ):
        _fail("REJECT_Q1_APPLICATION_KEY", "child program IDs must be root tuple")
    return (
        1,
        APPLICATION_KEY_SCHEMA_ID,
        input_signature_id,
        universe_root,
        construction_depth,
        coverage_code,
        operator_parameters,
        ordered_child_program_ids,
    )


def semantic_application_id_v1(application_key: tuple[object, ...]) -> bytes:
    return content_hash(APPLICATION_ID_DOMAIN, application_key)


@dataclass(frozen=True, slots=True)
class Q1SemanticCoverageRecordV1:
    input_signature_id: int
    universe_root: bytes
    construction_depth: int
    coverage_code: int
    eligible_application_count: int
    eligible_application_root: bytes
    processed_application_count: int
    processed_application_root: bytes
    strict_admitted_count: int
    strict_admission_root: bytes
    unique_canonical_ast_count: int
    rewrite_collapse_count: int

    def __post_init__(self) -> None:
        _input_binding(self.input_signature_id, self.universe_root)
        _uint(self.construction_depth, "construction_depth", 3)
        _uint(self.coverage_code, "coverage_code", 0xFFFF)
        if self.construction_depth == 0:
            if self.coverage_code not in LEAF_COVERAGE_CODES:
                _fail("REJECT_Q1_COVERAGE", "depth-zero coverage code is not a leaf")
        elif self.coverage_code not in OPERATOR_COVERAGE_CODES:
            _fail("REJECT_Q1_COVERAGE", "construction coverage code is unregistered")
        for name in (
            "eligible_application_count",
            "processed_application_count",
            "strict_admitted_count",
            "unique_canonical_ast_count",
            "rewrite_collapse_count",
        ):
            _uint(getattr(self, name), name, 0xFFFFFFFF)
        _root32(self.eligible_application_root, "eligible_application_root")
        _root32(self.processed_application_root, "processed_application_root")
        _root32(self.strict_admission_root, "strict_admission_root")
        if self.eligible_application_count != self.processed_application_count:
            _fail("REJECT_Q1_COVERAGE", "eligible and processed counts differ")
        if self.eligible_application_root != self.processed_application_root:
            _fail("REJECT_Q1_COVERAGE", "eligible and processed roots differ")
        if self.strict_admitted_count > self.processed_application_count:
            _fail("REJECT_Q1_COVERAGE", "strict admitted count exceeds processed")
        if self.unique_canonical_ast_count > self.strict_admitted_count:
            _fail("REJECT_Q1_COVERAGE", "unique AST count exceeds strict admitted")
        if self.rewrite_collapse_count > self.strict_admitted_count:
            _fail("REJECT_Q1_COVERAGE", "rewrite count exceeds strict admitted")
        if self.construction_depth == 0 and (
            self.eligible_application_count != 1
            or self.strict_admitted_count != 1
            or self.unique_canonical_ast_count != 1
            or self.rewrite_collapse_count != 0
        ):
            _fail("REJECT_Q1_COVERAGE", "each frozen leaf row must admit once")

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q1_SEMANTIC_COVERAGE_TAG,
            SEMANTIC_COVERAGE_SCHEMA_ID,
            self.input_signature_id,
            self.universe_root,
            self.construction_depth,
            self.coverage_code,
            self.eligible_application_count,
            self.eligible_application_root,
            self.processed_application_count,
            self.processed_application_root,
            self.strict_admitted_count,
            self.strict_admission_root,
            self.unique_canonical_ast_count,
            self.rewrite_collapse_count,
        )

    @property
    def record_id(self) -> bytes:
        return content_hash(COVERAGE_RECORD_ID_DOMAIN, self.canonical_object())


def expected_coverage_registry_v1() -> tuple[tuple[int, int], ...]:
    rows = tuple((0, code) for code in LEAF_COVERAGE_CODES) + tuple(
        (depth, code)
        for depth in CONSTRUCTION_DEPTHS
        for code in OPERATOR_COVERAGE_CODES
    )
    if len(rows) != EXPECTED_COVERAGE_RECORD_COUNT:
        raise AssertionError("Q1 coverage registry cardinality drift")
    return rows


@dataclass(frozen=True, slots=True)
class Q1FixedPointRecordV1:
    input_signature_id: int
    universe_root: bytes
    projection_profile_root: bytes
    raw_application_count: int
    strict_admitted_count: int
    rewrite_collapse_count: int
    maximum_depth: int
    maximum_nodes: int
    structural_boundary_depth: int
    work_queue_empty: bool
    zero_delta_full_boundary: bool
    all_eligible_covered: bool
    final_class_delta: int
    final_cohort_delta: int
    final_frontier_delta: int
    final_bank_delta: int
    program_count: int
    class_count: int
    cohort_count: int
    bank_point_count: int
    frontier_point_count: int
    maximum_bank_points_per_class: int
    maximum_frontier_points_per_class: int
    program_archive_root: bytes
    bank_archive_root: bytes
    class_archive_root: bytes
    coverage_archive_root: bytes

    def __post_init__(self) -> None:
        _input_binding(self.input_signature_id, self.universe_root)
        for name in (
            "projection_profile_root",
            "program_archive_root",
            "bank_archive_root",
            "class_archive_root",
            "coverage_archive_root",
        ):
            _root32(getattr(self, name), name)
        for name in (
            "raw_application_count",
            "strict_admitted_count",
            "rewrite_collapse_count",
            "maximum_depth",
            "maximum_nodes",
            "structural_boundary_depth",
            "final_class_delta",
            "final_cohort_delta",
            "final_frontier_delta",
            "final_bank_delta",
            "program_count",
            "class_count",
            "cohort_count",
            "bank_point_count",
            "frontier_point_count",
            "maximum_bank_points_per_class",
            "maximum_frontier_points_per_class",
        ):
            _uint(getattr(self, name), name, 0xFFFFFFFF)
        for name in (
            "work_queue_empty",
            "zero_delta_full_boundary",
            "all_eligible_covered",
        ):
            if type(getattr(self, name)) is not bool:
                _fail("REJECT_Q1_FIXED_POINT", f"{name} must be exact bool")
        if self.maximum_depth != 3 or self.maximum_nodes != 6:
            _fail("REJECT_Q1_FIXED_POINT", "formal fixed point requires depth3/node6")
        if self.structural_boundary_depth != 4:
            _fail("REJECT_Q1_FIXED_POINT", "structural boundary must be depth4")
        if not (
            self.work_queue_empty
            and self.zero_delta_full_boundary
            and self.all_eligible_covered
            and self.final_class_delta == 0
            and self.final_cohort_delta == 0
            and self.final_frontier_delta == 0
            and self.final_bank_delta == 0
        ):
            _fail("REJECT_Q1_FIXED_POINT", "fixed-point terminal evidence is incomplete")
        if (
            self.raw_application_count < len(LEAF_COVERAGE_CODES)
            or self.strict_admitted_count > self.raw_application_count
            or self.rewrite_collapse_count > self.strict_admitted_count
        ):
            _fail("REJECT_Q1_FIXED_POINT", "application count relation differs")
        if (
            self.program_count != self.bank_point_count
            or not 1 <= self.class_count <= self.cohort_count <= self.bank_point_count
            or self.bank_point_count > 2 * self.cohort_count
            or not self.class_count <= self.frontier_point_count <= self.bank_point_count
            or not 1 <= self.maximum_frontier_points_per_class
            <= self.maximum_bank_points_per_class
            <= self.bank_point_count
        ):
            _fail("REJECT_Q1_FIXED_POINT", "quotient cardinality relation differs")

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q1_FIXED_POINT_TAG,
            FIXED_POINT_SCHEMA_ID,
            self.input_signature_id,
            self.universe_root,
            self.projection_profile_root,
            self.raw_application_count,
            self.strict_admitted_count,
            self.rewrite_collapse_count,
            self.maximum_depth,
            self.maximum_nodes,
            self.structural_boundary_depth,
            self.work_queue_empty,
            self.zero_delta_full_boundary,
            self.all_eligible_covered,
            self.final_class_delta,
            self.final_cohort_delta,
            self.final_frontier_delta,
            self.final_bank_delta,
            self.program_count,
            self.class_count,
            self.cohort_count,
            self.bank_point_count,
            self.frontier_point_count,
            self.maximum_bank_points_per_class,
            self.maximum_frontier_points_per_class,
            self.program_archive_root,
            self.bank_archive_root,
            self.class_archive_root,
            self.coverage_archive_root,
        )

    @property
    def record_root(self) -> bytes:
        return content_hash(FIXED_POINT_ROOT_DOMAIN, self.canonical_object())


def frame_canonical_record_v1(record: object) -> bytes:
    payload = canonical_cbor_encode(record)
    if len(payload) > 0xFFFFFFFF:
        _fail("REJECT_Q1_FRAME", "record exceeds u32 frame length")
    return len(payload).to_bytes(FRAME_LENGTH_BYTES, "big") + payload


def replay_framed_records_v1(payload: bytes) -> tuple[tuple[object, ...], ...]:
    _bytes(payload, "framed_payload")
    offset = 0
    records: list[tuple[object, ...]] = []
    while offset < len(payload):
        if offset + FRAME_LENGTH_BYTES > len(payload):
            _fail("REJECT_Q1_FRAME", "truncated frame length")
        length = int.from_bytes(payload[offset : offset + FRAME_LENGTH_BYTES], "big")
        offset += FRAME_LENGTH_BYTES
        if offset + length > len(payload):
            _fail("REJECT_Q1_FRAME", "truncated frame payload")
        encoded = payload[offset : offset + length]
        offset += length
        records.append(_strict_replay(encoded, "framed_record"))
    return tuple(records)


@dataclass(frozen=True, slots=True)
class Q1ArchiveChunkManifestV1:
    input_signature_id: int
    universe_root: bytes
    stream_kind_id: ArchiveStreamKindId
    chunk_index: int
    first_record_index: int
    record_count: int
    first_record_id: bytes
    last_record_id: bytes
    record_subtree_root: bytes
    framed_blob_hash: bytes
    framed_blob_length: int

    def __post_init__(self) -> None:
        _input_binding(self.input_signature_id, self.universe_root)
        if not isinstance(self.stream_kind_id, ArchiveStreamKindId):
            _fail("REJECT_Q1_CHUNK", "stream kind is unregistered")
        for name in ("chunk_index", "first_record_index", "record_count"):
            _uint(getattr(self, name), name, 0xFFFFFFFF)
        if not 1 <= self.record_count <= MAX_RECORDS_PER_CHUNK:
            _fail("REJECT_Q1_CHUNK", "record count is outside chunk bound")
        for name in (
            "first_record_id",
            "last_record_id",
            "record_subtree_root",
            "framed_blob_hash",
        ):
            _root32(getattr(self, name), name)
        _uint(self.framed_blob_length, "framed_blob_length", MAX_CHUNK_FRAMED_BYTES)
        if self.framed_blob_length < self.record_count * FRAME_LENGTH_BYTES:
            _fail("REJECT_Q1_CHUNK", "framed blob is too short")

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q1_ARCHIVE_CHUNK_MANIFEST_TAG,
            ARCHIVE_CHUNK_MANIFEST_SCHEMA_ID,
            self.input_signature_id,
            self.universe_root,
            int(self.stream_kind_id),
            self.chunk_index,
            self.first_record_index,
            self.record_count,
            self.first_record_id,
            self.last_record_id,
            self.record_subtree_root,
            self.framed_blob_hash,
            self.framed_blob_length,
        )

    @property
    def record_id(self) -> bytes:
        return content_hash(CHUNK_MANIFEST_RECORD_ID_DOMAIN, self.canonical_object())


@dataclass(frozen=True, slots=True)
class Q1StreamDescriptorV1:
    stream_kind_id: ArchiveStreamKindId
    record_count: int
    archive_root: bytes
    framed_stream_bytes: int
    chunk_count: int
    chunk_manifest_subtree_root: bytes

    def canonical_object(self) -> tuple[object, ...]:
        if not isinstance(self.stream_kind_id, ArchiveStreamKindId):
            _fail("REJECT_Q1_STREAM", "stream kind is unregistered")
        _uint(self.record_count, "record_count", 0xFFFFFFFF)
        _root32(self.archive_root, "archive_root")
        _uint(self.framed_stream_bytes, "framed_stream_bytes")
        _uint(self.chunk_count, "chunk_count", 0xFFFFFFFF)
        _root32(self.chunk_manifest_subtree_root, "chunk_manifest_subtree_root")
        if (self.record_count == 0) != (self.chunk_count == 0):
            _fail("REJECT_Q1_STREAM", "empty record/chunk counts disagree")
        return (
            1,
            STREAM_DESCRIPTOR_SCHEMA_ID,
            int(self.stream_kind_id),
            self.record_count,
            self.archive_root,
            self.framed_stream_bytes,
            self.chunk_count,
            self.chunk_manifest_subtree_root,
        )


@dataclass(frozen=True, slots=True)
class Q1SignatureArchiveManifestV1:
    input_signature_id: int
    universe_root: bytes
    universe_row_count: int
    semantic_binding_root: bytes
    projection_profile_root: bytes
    stream_descriptors: tuple[Q1StreamDescriptorV1, ...]
    fixed_point_record_root: bytes
    saturation_state_root: bytes
    chunk_manifest_count: int
    chunk_manifest_archive_root: bytes

    def __post_init__(self) -> None:
        _input_binding(self.input_signature_id, self.universe_root)
        expected_rows = len(production_universe_v1(self.input_signature_id).rows)
        if type(self.universe_row_count) is not int or self.universe_row_count != expected_rows:
            _fail("REJECT_Q1_SIGNATURE_MANIFEST", "universe row count differs")
        for name in (
            "semantic_binding_root",
            "projection_profile_root",
            "fixed_point_record_root",
            "saturation_state_root",
            "chunk_manifest_archive_root",
        ):
            _root32(getattr(self, name), name)
        if type(self.stream_descriptors) is not tuple or tuple(
            descriptor.stream_kind_id for descriptor in self.stream_descriptors
        ) != tuple(ArchiveStreamKindId):
            _fail("REJECT_Q1_SIGNATURE_MANIFEST", "four stream descriptors differ")
        _uint(self.chunk_manifest_count, "chunk_manifest_count", 0xFFFFFFFF)
        expected_chunks = sum(item.chunk_count for item in self.stream_descriptors)
        if self.chunk_manifest_count != expected_chunks:
            _fail("REJECT_Q1_SIGNATURE_MANIFEST", "chunk manifest count differs")

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q1_SIGNATURE_ARCHIVE_MANIFEST_TAG,
            SIGNATURE_ARCHIVE_MANIFEST_SCHEMA_ID,
            self.input_signature_id,
            self.universe_root,
            self.universe_row_count,
            self.semantic_binding_root,
            self.projection_profile_root,
            MAX_RECORDS_PER_CHUNK,
            MAX_CHUNK_FRAMED_BYTES,
            tuple(item.canonical_object() for item in self.stream_descriptors),
            self.fixed_point_record_root,
            self.saturation_state_root,
            self.chunk_manifest_count,
            self.chunk_manifest_archive_root,
        )

    @property
    def manifest_root(self) -> bytes:
        return content_hash(SIGNATURE_MANIFEST_ROOT_DOMAIN, self.canonical_object())


@dataclass(frozen=True, slots=True)
class Q1ClosureBundleV1:
    semantic_binding_root: bytes
    projection_profile_root: bytes
    signature_rows: tuple[tuple[int, bytes, bytes, bytes], ...]

    def __post_init__(self) -> None:
        _root32(self.semantic_binding_root, "semantic_binding_root")
        _root32(self.projection_profile_root, "projection_profile_root")
        if type(self.signature_rows) is not tuple or len(self.signature_rows) != 2:
            _fail("REJECT_Q1_CLOSURE_BUNDLE", "exactly two signature rows required")
        for expected_id, row in zip((1, 2), self.signature_rows, strict=True):
            if type(row) is not tuple or len(row) != 4 or type(row[0]) is not int:
                _fail("REJECT_Q1_CLOSURE_BUNDLE", "signature row is malformed")
            if row[0] != expected_id:
                _fail("REJECT_Q1_CLOSURE_BUNDLE", "signature order differs")
            _input_binding(row[0], row[1])
            _root32(row[2], "signature_manifest_root")
            _root32(row[3], "signature_saturation_state_root")

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q1_CLOSURE_BUNDLE_TAG,
            CLOSURE_BUNDLE_SCHEMA_ID,
            self.semantic_binding_root,
            self.projection_profile_root,
            2,
            self.signature_rows,
        )

    @property
    def bundle_root(self) -> bytes:
        return content_hash(CLOSURE_BUNDLE_ROOT_DOMAIN, self.canonical_object())


@dataclass(frozen=True, slots=True)
class Q1SemanticBindingManifestV1:
    child_dsl_root: bytes
    operator_semantics_root: bytes
    identifier_registry_root: bytes
    canonical_ast_root: bytes
    canonical_cbor_root: bytes
    mdl_profile_id: bytes
    q0_receipt_root: bytes
    full_v16_leaf_manifest_root: bytes
    preregistration_document_sha256: bytes
    post_shrink6_document_sha256: bytes

    def __post_init__(self) -> None:
        for name in (
            "child_dsl_root",
            "operator_semantics_root",
            "identifier_registry_root",
            "canonical_ast_root",
            "canonical_cbor_root",
            "q0_receipt_root",
            "full_v16_leaf_manifest_root",
            "preregistration_document_sha256",
            "post_shrink6_document_sha256",
        ):
            _root32(getattr(self, name), name)
        _bytes(self.mdl_profile_id, "mdl_profile_id")

    def canonical_object(self) -> tuple[object, ...]:
        universes = tuple(
            (
                input_signature_id,
                production_universe_v1(input_signature_id).universe_root,
                len(production_universe_v1(input_signature_id).rows),
            )
            for input_signature_id in (1, 2)
        )
        return (
            1,
            Q1_SEMANTIC_BINDING_TAG,
            SEMANTIC_BINDING_SCHEMA_ID,
            DSL_VERSION.encode("ascii"),
            DSL_FREEZE_VERSION.encode("ascii"),
            CLOSURE_SEMANTICS_VERSION.encode("ascii"),
            self.child_dsl_root,
            self.operator_semantics_root,
            self.identifier_registry_root,
            self.canonical_ast_root,
            self.canonical_cbor_root,
            self.mdl_profile_id,
            self.q0_receipt_root,
            self.full_v16_leaf_manifest_root,
            universes,
            self.preregistration_document_sha256,
            self.post_shrink6_document_sha256,
        )

    @property
    def manifest_root(self) -> bytes:
        return content_hash(SEMANTIC_BINDING_ROOT_DOMAIN, self.canonical_object())


def projection_profile_object_v1(
    *,
    semantic_binding_root: bytes,
    coverage_registry_root: bytes,
    resource_guard_registry: tuple[tuple[int, bytes], ...],
) -> tuple[object, ...]:
    _root32(semantic_binding_root, "semantic_binding_root")
    _root32(coverage_registry_root, "coverage_registry_root")
    expected_coverage_root = rfc6962_root(expected_coverage_registry_v1())
    if coverage_registry_root != expected_coverage_root:
        _fail(
            "REJECT_Q1_PROJECTION_PROFILE",
            "coverage registry root differs from the frozen 846-row registry",
        )
    if type(resource_guard_registry) is not tuple or any(
        type(row) is not tuple
        or len(row) != 2
        or type(row[0]) is not int
        or type(row[1]) is not bytes
        for row in resource_guard_registry
    ):
        _fail("REJECT_Q1_PROJECTION_PROFILE", "resource guards must be typed rows")
    if resource_guard_registry != Q1_RESOURCE_GUARD_REGISTRY:
        _fail(
            "REJECT_Q1_PROJECTION_PROFILE",
            "resource guards differ from the frozen 12-row registry",
        )
    return (
        1,
        Q1_ARCHIVE_PROJECTION_PROFILE_TAG,
        ARCHIVE_PROJECTION_PROFILE_SCHEMA_ID,
        ARCHIVE_WIRE_VERSION.encode("ascii"),
        PROJECTION_FREEZE_VERSION.encode("ascii"),
        PROJECTION_PROFILE_ID.encode("ascii"),
        semantic_binding_root,
        Q1_TAG_REGISTRY,
        expected_coverage_registry_v1(),
        coverage_registry_root,
        tuple(int(value) for value in ArchiveStreamKindId),
        MAX_RECORDS_PER_CHUNK,
        MAX_CHUNK_FRAMED_BYTES,
        FRAME_LENGTH_BYTES,
        COMPRESSION_ID_NONE,
        b"FRAME_U32BE_LENGTH_PLUS_CANONICAL_CBOR",
        b"CHUNK_CLOSE_BEFORE_NEXT_RECORD_EXCEEDS_RECORD_OR_FRAMED_BYTE_LIMIT",
        (
            b"PROGRAM_U8_DEPTH_U16_NODES_U8_SORT_U16_ROOT_OPERATOR_AST_CBOR",
            b"COHORT_CLASS_ID_SIGNATURE_ID_SIGNATURE_CBOR",
            b"CLASS_ID_BEHAVIOR_CBOR",
            b"COVERAGE_U8_DEPTH_U16_COVERAGE_CODE",
        ),
        ENDPOINT_RUN_METADATA_RESERVATION_BYTES,
        HOST_RUN_METADATA_RESERVATION_BYTES,
        MAX_RUN_METADATA_FRAME_COUNT,
        MAX_RUN_METADATA_FRAME_BYTES,
        EXTERNAL_SORT_RUN_PAYLOAD_LIMIT_BYTES,
        EXTERNAL_SORT_MERGE_FAN_IN,
        EXTERNAL_SORT_RUN_MAGIC,
        EXTERNAL_SORT_RUN_HEADER_BYTES,
        EXTERNAL_SORT_ROW_LENGTH_BYTES,
        SCRATCH_ALLOCATION_BLOCK_BYTES,
        SCRATCH_METADATA_RESERVE_BYTES_PER_LIVE_FILE,
        SCRATCH_NONSPARSE_REQUIRED,
        EXTERNAL_SORT_STREAM_ORDER,
        EXTERNAL_SORT_SIGNATURE_ORDER,
        b"STABLE_K_WAY_MERGE_CONTIGUOUS_RUN_INDEX_GROUPS",
        b"SEAL_HASH_REOPEN_VERIFY_THEN_FREE_INPUT_GROUP",
        b"NO_RANDOM_OR_TIME_COMPONENT_IN_RUN_FILE_NAME",
        b"RUN_ROW_U32BE_KEY_LENGTH_KEY_U32BE_RECORD_LENGTH_CANONICAL_RECORD",
        b"SCRATCH_CHARGE_CEIL_FILE_SIZE_TO_4096_PLUS_4096_PER_LIVE_FILE",
        resource_guard_registry,
        Q1_OUTPUT_SLOT_NAMES,
        b"DEPTH_BARRIER_DIRECT_FULL_BANK",
        b"RAW_AND_SEMANTIC_COVERAGE_EXACT_EQUAL_WORK_QUEUE_HIGH_WATER_MAX",
        b"COUNTING_DISCARD_USES_IDENTICAL_ENCODER_AND_FIXED_ROOT_PLACEHOLDERS",
    )


def projection_profile_root_v1(
    *,
    semantic_binding_root: bytes,
    coverage_registry_root: bytes,
    resource_guard_registry: tuple[tuple[int, bytes], ...],
) -> bytes:
    return content_hash(
        PROJECTION_PROFILE_ROOT_DOMAIN,
        projection_profile_object_v1(
            semantic_binding_root=semantic_binding_root,
            coverage_registry_root=coverage_registry_root,
            resource_guard_registry=resource_guard_registry,
        ),
    )


def partition_stream_commitment_v1(
    *,
    input_signature_id: int,
    universe_root: bytes,
    raw_application_count: int,
    behavior_class_count: int,
    cohort_count: int,
    bank_point_count: int,
    frontier_point_count: int,
    maximum_bank_points_per_class: int,
    maximum_frontier_points_per_class: int,
    program_record_count: int,
    coverage_record_count: int,
    projected_record_stream_bytes: int,
    projected_chunk_manifest_stream_bytes: int,
    stream_diagnostic_commitments: tuple[bytes, ...],
) -> bytes:
    """Bind the four ordered stream projections and their semantic counts."""

    _input_binding(input_signature_id, universe_root)
    for name, value in (
        ("raw_application_count", raw_application_count),
        ("behavior_class_count", behavior_class_count),
        ("cohort_count", cohort_count),
        ("bank_point_count", bank_point_count),
        ("frontier_point_count", frontier_point_count),
        ("maximum_bank_points_per_class", maximum_bank_points_per_class),
        ("maximum_frontier_points_per_class", maximum_frontier_points_per_class),
        ("program_record_count", program_record_count),
        ("coverage_record_count", coverage_record_count),
        ("projected_record_stream_bytes", projected_record_stream_bytes),
        (
            "projected_chunk_manifest_stream_bytes",
            projected_chunk_manifest_stream_bytes,
        ),
    ):
        _uint(value, name)
    if type(stream_diagnostic_commitments) is not tuple or len(
        stream_diagnostic_commitments
    ) != len(ArchiveStreamKindId):
        _fail(
            "REJECT_Q1_PARTITION_STREAM_COMMITMENT",
            "four ordered stream commitments are required",
        )
    for index, root in enumerate(stream_diagnostic_commitments, start=1):
        _root32(root, f"stream_diagnostic_commitments[{index}]")
    return content_hash(
        PARTITION_STREAM_COMMITMENT_DOMAIN,
        (
            1,
            PARTITION_STREAM_COMMITMENT_SCHEMA_ID,
            input_signature_id,
            universe_root,
            raw_application_count,
            behavior_class_count,
            cohort_count,
            bank_point_count,
            frontier_point_count,
            maximum_bank_points_per_class,
            maximum_frontier_points_per_class,
            program_record_count,
            coverage_record_count,
            projected_record_stream_bytes,
            projected_chunk_manifest_stream_bytes,
            stream_diagnostic_commitments,
        ),
    )


def partition_external_sort_root_v1(
    *,
    input_signature_id: int,
    universe_root: bytes,
    external_sort_stream_roots: tuple[bytes, ...],
    projected_peak_scratch_bytes: int,
) -> bytes:
    """Bind all ordered per-stream sort ledgers and their partition high-water."""

    _input_binding(input_signature_id, universe_root)
    _uint(projected_peak_scratch_bytes, "projected_peak_scratch_bytes")
    if type(external_sort_stream_roots) is not tuple or len(
        external_sort_stream_roots
    ) != len(ArchiveStreamKindId):
        _fail(
            "REJECT_Q1_PARTITION_EXTERNAL_SORT",
            "four ordered external-sort roots are required",
        )
    for index, root in enumerate(external_sort_stream_roots, start=1):
        _root32(root, f"external_sort_stream_roots[{index}]")
    return content_hash(
        PARTITION_EXTERNAL_SORT_ROOT_DOMAIN,
        (
            1,
            PARTITION_EXTERNAL_SORT_SCHEMA_ID,
            input_signature_id,
            universe_root,
            external_sort_stream_roots,
            projected_peak_scratch_bytes,
        ),
    )


@dataclass(frozen=True, slots=True)
class Q1ProjectionPartitionRowV1:
    input_signature_id: int
    universe_root: bytes
    raw_application_count: int
    behavior_class_count: int
    cohort_count: int
    bank_point_count: int
    frontier_point_count: int
    maximum_bank_points_per_class: int
    maximum_frontier_points_per_class: int
    peak_work_queue_points: int
    program_record_count: int
    coverage_record_count: int
    projected_record_stream_bytes: int
    projected_chunk_manifest_stream_bytes: int
    projected_fixed_point_frame_bytes: int
    projected_signature_manifest_frame_bytes: int
    projected_partition_payload_bytes: int
    projected_peak_scratch_bytes: int
    stream_diagnostic_commitments: tuple[bytes, ...]
    diagnostic_stream_commitment: bytes
    external_sort_stream_roots: tuple[bytes, ...]
    external_sort_projection_root: bytes

    def __post_init__(self) -> None:
        _input_binding(self.input_signature_id, self.universe_root)
        for name in (
            "raw_application_count",
            "behavior_class_count",
            "cohort_count",
            "bank_point_count",
            "frontier_point_count",
            "maximum_bank_points_per_class",
            "maximum_frontier_points_per_class",
            "peak_work_queue_points",
            "program_record_count",
            "coverage_record_count",
            "projected_record_stream_bytes",
            "projected_chunk_manifest_stream_bytes",
            "projected_fixed_point_frame_bytes",
            "projected_signature_manifest_frame_bytes",
            "projected_partition_payload_bytes",
            "projected_peak_scratch_bytes",
        ):
            _uint(getattr(self, name), name)
        if self.coverage_record_count != EXPECTED_COVERAGE_RECORD_COUNT:
            _fail(
                "REJECT_Q1_PROJECTION_PARTITION",
                "coverage record count differs from 846",
            )
        if self.program_record_count != self.bank_point_count:
            _fail(
                "REJECT_Q1_PROJECTION_PARTITION",
                "one program record is required per continuation-bank point",
            )
        if (
            self.raw_application_count < len(LEAF_COVERAGE_CODES)
            or not 1
            <= self.behavior_class_count
            <= self.cohort_count
            <= self.bank_point_count
            <= 2 * self.cohort_count
            or not self.behavior_class_count
            <= self.frontier_point_count
            <= self.bank_point_count
            or not 1
            <= self.maximum_frontier_points_per_class
            <= self.maximum_bank_points_per_class
            <= self.bank_point_count
        ):
            _fail(
                "REJECT_Q1_PROJECTION_PARTITION",
                "partition cardinality relation differs",
            )
        expected_payload = (
            self.projected_record_stream_bytes
            + self.projected_chunk_manifest_stream_bytes
            + self.projected_fixed_point_frame_bytes
            + self.projected_signature_manifest_frame_bytes
        )
        if self.projected_partition_payload_bytes != expected_payload:
            _fail(
                "REJECT_Q1_PROJECTION_PARTITION",
                "partition payload formula differs",
            )
        for name in ("stream_diagnostic_commitments", "external_sort_stream_roots"):
            roots = getattr(self, name)
            if type(roots) is not tuple or len(roots) != len(ArchiveStreamKindId):
                _fail(
                    "REJECT_Q1_PROJECTION_PARTITION",
                    f"{name} must contain four ordered roots",
                )
            for index, root in enumerate(roots, start=1):
                _root32(root, f"{name}[{index}]")
        _root32(self.diagnostic_stream_commitment, "diagnostic_stream_commitment")
        _root32(self.external_sort_projection_root, "external_sort_projection_root")
        expected_stream_commitment = partition_stream_commitment_v1(
            input_signature_id=self.input_signature_id,
            universe_root=self.universe_root,
            raw_application_count=self.raw_application_count,
            behavior_class_count=self.behavior_class_count,
            cohort_count=self.cohort_count,
            bank_point_count=self.bank_point_count,
            frontier_point_count=self.frontier_point_count,
            maximum_bank_points_per_class=self.maximum_bank_points_per_class,
            maximum_frontier_points_per_class=self.maximum_frontier_points_per_class,
            program_record_count=self.program_record_count,
            coverage_record_count=self.coverage_record_count,
            projected_record_stream_bytes=self.projected_record_stream_bytes,
            projected_chunk_manifest_stream_bytes=(
                self.projected_chunk_manifest_stream_bytes
            ),
            stream_diagnostic_commitments=self.stream_diagnostic_commitments,
        )
        if self.diagnostic_stream_commitment != expected_stream_commitment:
            _fail(
                "REJECT_Q1_PROJECTION_PARTITION",
                "partition stream commitment preimage differs",
            )
        expected_sort_root = partition_external_sort_root_v1(
            input_signature_id=self.input_signature_id,
            universe_root=self.universe_root,
            external_sort_stream_roots=self.external_sort_stream_roots,
            projected_peak_scratch_bytes=self.projected_peak_scratch_bytes,
        )
        if self.external_sort_projection_root != expected_sort_root:
            _fail(
                "REJECT_Q1_PROJECTION_PARTITION",
                "partition external-sort root preimage differs",
            )

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            PROJECTION_PARTITION_ROW_SCHEMA_ID,
            self.input_signature_id,
            self.universe_root,
            self.raw_application_count,
            self.behavior_class_count,
            self.cohort_count,
            self.bank_point_count,
            self.frontier_point_count,
            self.maximum_bank_points_per_class,
            self.maximum_frontier_points_per_class,
            self.peak_work_queue_points,
            self.program_record_count,
            self.coverage_record_count,
            self.projected_record_stream_bytes,
            self.projected_chunk_manifest_stream_bytes,
            self.projected_fixed_point_frame_bytes,
            self.projected_signature_manifest_frame_bytes,
            self.projected_partition_payload_bytes,
            self.projected_peak_scratch_bytes,
            self.stream_diagnostic_commitments,
            self.diagnostic_stream_commitment,
            self.external_sort_stream_roots,
            self.external_sort_projection_root,
        )


@dataclass(frozen=True, slots=True)
class Q1ArchiveProjectionResultV1:
    projection_profile_root: bytes
    semantic_binding_root: bytes
    partition_rows: tuple[Q1ProjectionPartitionRowV1, ...]
    projected_closure_bundle_frame_bytes: int
    projected_archive_payload_bytes_per_endpoint: int
    projected_endpoint_total_output_bytes: int
    projected_endpoint_peak_scratch_bytes: int
    projected_host_replay_output_bytes: int
    projected_host_replay_peak_scratch_bytes: int

    def __post_init__(self) -> None:
        _root32(self.projection_profile_root, "projection_profile_root")
        _root32(self.semantic_binding_root, "semantic_binding_root")
        expected_profile_root = projection_profile_root_v1(
            semantic_binding_root=self.semantic_binding_root,
            coverage_registry_root=rfc6962_root(expected_coverage_registry_v1()),
            resource_guard_registry=Q1_RESOURCE_GUARD_REGISTRY,
        )
        if self.projection_profile_root != expected_profile_root:
            _fail(
                "REJECT_Q1_PROJECTION_RESULT",
                "projection profile root differs from the semantic binding",
            )
        if type(self.partition_rows) is not tuple or len(self.partition_rows) != 2:
            _fail("REJECT_Q1_PROJECTION_RESULT", "two projection partition rows required")
        if any(type(row) is not Q1ProjectionPartitionRowV1 for row in self.partition_rows):
            _fail("REJECT_Q1_PROJECTION_RESULT", "projection partition row type differs")
        if tuple(row.input_signature_id for row in self.partition_rows) != (1, 2):
            _fail("REJECT_Q1_PROJECTION_RESULT", "partition rows are out of order")
        for name in (
            "projected_archive_payload_bytes_per_endpoint",
            "projected_closure_bundle_frame_bytes",
            "projected_endpoint_total_output_bytes",
            "projected_endpoint_peak_scratch_bytes",
            "projected_host_replay_output_bytes",
            "projected_host_replay_peak_scratch_bytes",
        ):
            _uint(getattr(self, name), name)
        if self.projected_endpoint_total_output_bytes != (
            self.projected_archive_payload_bytes_per_endpoint
            + ENDPOINT_RUN_METADATA_RESERVATION_BYTES
        ):
            _fail("REJECT_Q1_PROJECTION_RESULT", "endpoint output formula differs")
        if self.projected_archive_payload_bytes_per_endpoint != (
            sum(row.projected_partition_payload_bytes for row in self.partition_rows)
            + self.projected_closure_bundle_frame_bytes
        ):
            _fail(
                "REJECT_Q1_PROJECTION_RESULT",
                "global archive payload formula differs",
            )
        if self.projected_host_replay_output_bytes != HOST_RUN_METADATA_RESERVATION_BYTES:
            _fail(
                "REJECT_Q1_PROJECTION_RESULT",
                "host replay output must equal its frozen metadata reservation",
            )
        if self.projected_host_replay_peak_scratch_bytes != 0:
            _fail(
                "REJECT_Q1_PROJECTION_RESULT",
                "streaming host replay must not allocate regular-file scratch",
            )
        if self.projected_endpoint_peak_scratch_bytes != max(
            row.projected_peak_scratch_bytes for row in self.partition_rows
        ):
            _fail(
                "REJECT_Q1_PROJECTION_RESULT",
                "endpoint peak scratch must equal the frozen sequential-partition maximum",
            )

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q1_ARCHIVE_PROJECTION_RESULT_TAG,
            ARCHIVE_PROJECTION_RESULT_SCHEMA_ID,
            self.projection_profile_root,
            self.semantic_binding_root,
            tuple(row.canonical_object() for row in self.partition_rows),
            self.projected_closure_bundle_frame_bytes,
            self.projected_archive_payload_bytes_per_endpoint,
            self.projected_endpoint_total_output_bytes,
            self.projected_endpoint_peak_scratch_bytes,
            self.projected_host_replay_output_bytes,
            self.projected_host_replay_peak_scratch_bytes,
            0,
            0,
            0,
            None,
            (None,) * len(Q1_OUTPUT_SLOT_NAMES),
            0,
            False,
            None,
            False,
        )

    @property
    def diagnostic_root(self) -> bytes:
        return content_hash(PROJECTION_RESULT_ROOT_DOMAIN, self.canonical_object())


def archive_root_v1(records: Sequence[object]) -> bytes:
    return rfc6962_root(tuple(records))


def framed_blob_hash_v1(payload: bytes) -> bytes:
    _bytes(payload, "framed_blob")
    return content_hash(FRAMED_BLOB_HASH_DOMAIN, (payload,))


def signature_saturation_state_root_v1(
    *,
    input_signature_id: int,
    universe_root: bytes,
    semantic_binding_root: bytes,
    projection_profile_root: bytes,
    program_archive_root: bytes,
    cohort_archive_root: bytes,
    class_archive_root: bytes,
    coverage_archive_root: bytes,
    fixed_point_record_root: bytes,
) -> bytes:
    _input_binding(input_signature_id, universe_root)
    for name, value in (
        ("semantic_binding_root", semantic_binding_root),
        ("projection_profile_root", projection_profile_root),
        ("program_archive_root", program_archive_root),
        ("cohort_archive_root", cohort_archive_root),
        ("class_archive_root", class_archive_root),
        ("coverage_archive_root", coverage_archive_root),
        ("fixed_point_record_root", fixed_point_record_root),
    ):
        _root32(value, name)
    return content_hash(
        SIGNATURE_SATURATION_STATE_ROOT_DOMAIN,
        (
            1,
            b"hegel-q1-signature-saturation-state/1",
            input_signature_id,
            universe_root,
            semantic_binding_root,
            projection_profile_root,
            program_archive_root,
            cohort_archive_root,
            class_archive_root,
            coverage_archive_root,
            fixed_point_record_root,
        ),
    )


def canonical_archive_order_v1(
    records: Iterable[object],
    *,
    stream_kind_id: ArchiveStreamKindId,
) -> tuple[object, ...]:
    material = tuple(records)
    expected_type: type[object]
    if stream_kind_id is ArchiveStreamKindId.PROGRAM:
        expected_type = Q1RepresentativeProgramRecordV1
        key = lambda row: row.sort_key  # type: ignore[union-attr]
    elif stream_kind_id is ArchiveStreamKindId.COHORT:
        expected_type = Q1ContinuationCohortRecordV1
        key = lambda row: (  # type: ignore[union-attr]
            row.class_id,
            row.signature_id,
            canonical_cbor_encode(construction_signature_object_v1(row.construction_signature)),
        )
    elif stream_kind_id is ArchiveStreamKindId.CLASS:
        expected_type = Q1QuotientClassRecordV1
        key = lambda row: (row.class_id, row.behavior.canonical_bytes)  # type: ignore[union-attr]
    elif stream_kind_id is ArchiveStreamKindId.COVERAGE:
        expected_type = Q1SemanticCoverageRecordV1
        key = lambda row: (row.construction_depth, row.coverage_code)  # type: ignore[union-attr]
    else:
        raise TypeError("stream_kind_id must be ArchiveStreamKindId")
    if any(type(row) is not expected_type for row in material):
        _fail("REJECT_Q1_ARCHIVE_ORDER", "record type differs from stream kind")
    ordered = tuple(sorted(material, key=key))
    if material != ordered:
        _fail("REJECT_Q1_ARCHIVE_ORDER", "records are not canonically ordered")
    if stream_kind_id is ArchiveStreamKindId.PROGRAM and tuple(
        row.program_index for row in material  # type: ignore[union-attr]
    ) != tuple(range(len(material))):
        _fail("REJECT_Q1_ARCHIVE_ORDER", "program indices are not contiguous")
    if stream_kind_id is ArchiveStreamKindId.COHORT and tuple(
        row.cohort_index for row in material  # type: ignore[union-attr]
    ) != tuple(range(len(material))):
        _fail("REJECT_Q1_ARCHIVE_ORDER", "cohort indices are not contiguous")
    if stream_kind_id is ArchiveStreamKindId.CLASS and tuple(
        row.class_index for row in material  # type: ignore[union-attr]
    ) != tuple(range(len(material))):
        _fail("REJECT_Q1_ARCHIVE_ORDER", "class indices are not contiguous")
    if stream_kind_id is ArchiveStreamKindId.COVERAGE and tuple(
        (row.construction_depth, row.coverage_code) for row in material  # type: ignore[union-attr]
    ) != expected_coverage_registry_v1():
        _fail("REJECT_Q1_ARCHIVE_ORDER", "coverage registry is incomplete")
    return material


__all__ = [
    "APPLICATION_KEY_SCHEMA_ID",
    "ARCHIVE_WIRE_VERSION",
    "ArchiveStreamKindId",
    "ENDPOINT_RUN_METADATA_RESERVATION_BYTES",
    "EXTERNAL_SORT_MERGE_FAN_IN",
    "EXTERNAL_SORT_RUN_HEADER_BYTES",
    "EXTERNAL_SORT_RUN_MAGIC",
    "EXTERNAL_SORT_RUN_PAYLOAD_LIMIT_BYTES",
    "EXTERNAL_SORT_ROW_LENGTH_BYTES",
    "EXTERNAL_SORT_SIGNATURE_ORDER",
    "EXTERNAL_SORT_STREAM_ORDER",
    "EXPECTED_COVERAGE_RECORD_COUNT",
    "HOST_RUN_METADATA_RESERVATION_BYTES",
    "MAX_CHUNK_FRAMED_BYTES",
    "MAX_RECORDS_PER_CHUNK",
    "PROJECTION_FREEZE_VERSION",
    "PROJECTION_PROFILE_ID",
    "Q1ArchiveChunkManifestV1",
    "Q1ArchiveContractError",
    "Q1ArchiveProjectionResultV1",
    "Q1BehaviorBlobV1",
    "Q1BehaviorCellV1",
    "Q1ClosureBundleV1",
    "Q1CohortWitnessV1",
    "Q1ContinuationCohortRecordV1",
    "Q1FixedPointRecordV1",
    "Q1QuotientClassRecordV1",
    "Q1ProjectionPartitionRowV1",
    "Q1RepresentativeProgramRecordV1",
    "Q1SemanticBindingManifestV1",
    "Q1SemanticCoverageRecordV1",
    "Q1SignatureArchiveManifestV1",
    "Q1StreamDescriptorV1",
    "Q1_OUTPUT_SLOT_NAMES",
    "Q1_RESOURCE_GUARD_REGISTRY",
    "Q1_TAG_REGISTRY",
    "SCRATCH_ALLOCATION_BLOCK_BYTES",
    "SCRATCH_METADATA_RESERVE_BYTES_PER_LIVE_FILE",
    "SCRATCH_NONSPARSE_REQUIRED",
    "archive_root_v1",
    "canonical_archive_order_v1",
    "construction_signature_id_v1",
    "construction_signature_object_v1",
    "expected_coverage_registry_v1",
    "frame_canonical_record_v1",
    "framed_blob_hash_v1",
    "projection_profile_object_v1",
    "projection_profile_root_v1",
    "partition_external_sort_root_v1",
    "partition_stream_commitment_v1",
    "replay_framed_records_v1",
    "semantic_application_id_v1",
    "semantic_application_key_v1",
    "signature_saturation_state_root_v1",
]
