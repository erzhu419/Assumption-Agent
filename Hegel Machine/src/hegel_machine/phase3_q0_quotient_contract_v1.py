"""Machine contract for Phase-3A-Q0 exact quotient qualification.

This module freezes identity, typed behavior cells, continuation signatures,
and Pareto dominance for the target-blind C3 qualification track.  It does
not enumerate the full language, read target truth, generate formal roots, or
inherit the historical M3 24/24 readiness state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from fractions import Fraction
from typing import Final, Iterable

from .strict_cbor_v1 import canonical_cbor_encode, content_hash, rfc6962_root


DSL_VERSION: Final = "hegel-old-dsl-v1.6.0"
DSL_FREEZE_VERSION: Final = "hegel-freeze-p2b-p3-v1.6.0"
CLOSURE_SEMANTICS_VERSION: Final = "hegel-quotient-closure-v1.0.1"
Q0_FREEZE_VERSION: Final = "hegel-freeze-p3a-q0-v1.0.1"
Q0_QUALIFICATION_ID: Final = (
    "hegel-phase3a-q0-exact-quotient-qualification-v1"
)
NORMATIVE_DOCUMENT_ID: Final = (
    "hegel-phase3-post-shrink6-quotient-direction-v1"
)
NORMATIVE_DOCUMENT_PATH: Final = (
    "Hegel Machine/docs/"
    "Hegel_Machine_Phase3_Post_Shrink6_Quotient_Direction_Decision.md"
)
NORMATIVE_DOCUMENT_SHA256: Final = (
    "1df8d3ff3ede2cbead98e7901a3e82b91c460ad1d5eb0d1af78938e7b2d23b95"
)
SOURCE_Y_COMMIT: Final = "5217568303d5c7f902682c092750f637c64f080a"
EVIDENCE_Z_COMMIT: Final = "ea98157f5d6eb2930ab28dda8f3a6839b343673c"

Q0_EXECUTION_STATE: Final = "NOT_RUN"
Q0_FORMAL_ROOTS: Final = None
Q0_FORMAL_ROOTS_GENERATED: Final = False
Q0_TARGET_TRUTH_ACCESS_ALLOWED: Final = False
Q0_SPLIT_ACCESS_ALLOWED: Final = False
Q0_ROLE_EVALUATION_ALLOWED: Final = False
Q0_OLD_M3_GATE_COUNT_INHERITED: Final = 0
Q0_READINESS_GATE_TOTAL: Final = 14

Q0_PROJECTION_ID: Final = "hegel-q0-micro-projection-v1"
Q0_PROBE_INPUT_SIGNATURE_ID: Final = 0x7001
Q0_PROJECTION_MAX_AST_DEPTH: Final = 2
Q0_PROJECTION_MAX_NODE_COUNT: Final = 4
Q0_PROJECTION_MAX_TOP_LEVEL_CLAUSES: Final = 2
Q0_PROJECTION_MAX_AGGREGATE_LEAVES: Final = 1
Q0_PROJECTION_ROW_COUNT: Final = 4
Q0_INHERITED_MAX_SCALAR_PARAMETER_OCCURRENCES: Final = 3
Q0_INHERITED_MAX_SCOPE_CLAUSES: Final = 2
Q0_INHERITED_MAX_DISTINCT_BIT_SLOTS: Final = 4

Q0_MAX_RAW_APPLICATIONS: Final = 5_000
Q0_MAX_CANONICAL_SYNTAX: Final = 2_000
Q0_MAX_BEHAVIOR_CLASSES: Final = 2_000
Q0_MAX_FRONTIER_POINTS: Final = 2_000
Q0_MAX_FRONTIER_POINTS_PER_CLASS: Final = 64
Q0_MAX_CONTINUATION_BANK_POINTS: Final = 2_000
Q0_MAX_CONTINUATION_BANK_POINTS_PER_CLASS: Final = 64
Q0_MAX_SATURATION_ROUNDS: Final = 4
Q0_MAX_OUTPUT_BYTES: Final = 64 * 1024 * 1024
Q0_MAX_WALL_TIME_SECONDS: Final = 300
Q0_MAX_MEMORY_BYTES: Final = 512 * 1024 * 1024

RATIONAL_VALUE_GRID: Final = frozenset(
    Fraction(numerator, denominator)
    for numerator in range(-64, 65)
    for denominator in range(1, 9)
)

Q0_BEHAVIOR_BLOB_TAG: Final = 0x3601
Q0_CONSTRUCTION_SIGNATURE_TAG: Final = 0x3602
Q0_FRONTIER_ENTRY_TAG: Final = 0x3603
Q0_QUOTIENT_CLASS_TAG: Final = 0x3604
Q0_SATURATION_RECEIPT_TAG: Final = 0x3605
Q0_PROBE_INPUT_TAG: Final = 0x3606

BEHAVIOR_BLOB_SCHEMA_ID: Final = b"hegel-q0-behavior-blob/1"
CONSTRUCTION_SIGNATURE_SCHEMA_ID: Final = b"hegel-q0-construction-signature/1"
FRONTIER_ENTRY_SCHEMA_ID: Final = b"hegel-q0-frontier-entry/1"
QUOTIENT_CLASS_SCHEMA_ID: Final = b"hegel-q0-quotient-class/1"
SATURATION_RECEIPT_SCHEMA_ID: Final = b"hegel-q0-saturation-receipt/1"
PROBE_INPUT_SCHEMA_ID: Final = b"hegel-q0-probe-input/1"
ENDPOINT_STATE_SCHEMA_ID: Final = b"hegel-q0-oracle-endpoint-state/1"
SEMANTIC_BINDING_SCHEMA_ID: Final = b"hegel-q0-semantic-binding/1"
PROJECTION_MANIFEST_SCHEMA_ID: Final = b"hegel-q0-projection-manifest/1"

BEHAVIOR_ID_DOMAIN: Final = "HEGEL/Q0/BEHAVIOR_ID/V1"
FRONTIER_ENTRY_ID_DOMAIN: Final = "HEGEL/Q0/FRONTIER_ENTRY_ID/V1"
QUOTIENT_CLASS_RECORD_ID_DOMAIN: Final = "HEGEL/Q0/QUOTIENT_CLASS_RECORD_ID/V1"
SATURATION_RECEIPT_ROOT_DOMAIN: Final = "HEGEL/Q0/SATURATION_RECEIPT/V1"
PROBE_UNIVERSE_ROOT_DOMAIN: Final = "HEGEL/Q0/PROBE_UNIVERSE_ROOT/V1"
ENDPOINT_STATE_ROOT_DOMAIN: Final = "HEGEL/Q0/ORACLE_ENDPOINT_STATE/V1"
SEMANTIC_BINDING_ROOT_DOMAIN: Final = "HEGEL/Q0/SEMANTIC_BINDING/V1"
PROJECTION_MANIFEST_ROOT_DOMAIN: Final = "HEGEL/Q0/PROJECTION_MANIFEST/V1"
ROLE_MATCH_PROFILE_ID: Final = "BOOL_BIT_EXACT_PREDICATE_MATCH_V1"
Q0_ENDPOINT_PASS_STATUS: Final = (
    "SINGLE_IMPLEMENTATION_EXHAUSTIVE_ORACLE_EQUIVALENCE_PASS"
)

# These five roots are the already sealed, non-formal v1.6 diagnostic
# bindings.  Q0 remains a qualification track, so it reuses their exact
# preimages rather than silently substituting the unrelated earlier formal
# static roots.  The Q0 semantic-binding root below additionally binds the
# target-blind adapter, projection, and probe universe.
Q0_CHILD_DSL_SPEC_ROOT: Final = bytes.fromhex(
    "da5ed2db33a88a0912d5003999f787cc26ba18564876615773a82bb742d9f8ae"
)
Q0_OPERATOR_SEMANTICS_ROOT: Final = bytes.fromhex(
    "922e48ada22dfa8621a4d516e07ec9aa7dc8fc10c165d1cafc963575aed5ec03"
)
Q0_IDENTIFIER_REGISTRY_ROOT: Final = bytes.fromhex(
    "64c9415f7759eec140e439030c5a5374851b9024d7d4849b52b995704ba76ff1"
)
Q0_CANONICAL_AST_SCHEMA_ROOT: Final = bytes.fromhex(
    "5de72fc51e27e5501561ffda6b05522f4941d1a13c4b324f5edcc15fa584a0bd"
)
Q0_CANONICAL_CBOR_PROFILE_ROOT: Final = bytes.fromhex(
    "ef0008912962de9da322eaeea6e421e1e58d16be152f968298774af0fd3249ab"
)

Q0_LEAF_COVERAGE_CODES: Final = tuple(range(0x0000, 0x000F))
Q0_UNARY_COVERAGE_CODES: Final = tuple(0x1000 + index for index in range(4))
Q0_BINARY_COVERAGE_CODES: Final = tuple(
    0x2000 + index for index in (1, 2, 3, 5, 6)
)
Q0_APPROX_COVERAGE_CODES: Final = (0x3001, 0x3002)
Q0_AND2_COVERAGE_CODE: Final = 0x4002
Q0_COVERAGE_CODES: Final = (
    Q0_LEAF_COVERAGE_CODES
    + Q0_UNARY_COVERAGE_CODES
    + Q0_BINARY_COVERAGE_CODES
    + Q0_APPROX_COVERAGE_CODES
    + (Q0_AND2_COVERAGE_CODE,)
)
Q0_COVERAGE_RECORD_LENGTH: Final = 6

Q0_ALLOWED_UNARY_OPERATOR_IDS: Final = (0, 1, 2, 3)
Q0_ALLOWED_BINARY_OPERATOR_IDS: Final = (1, 2, 3, 5, 6)
Q0_ALLOWED_APPROX_TOLERANCE_IDS: Final = (1, 2)
Q0_TOP_LEVEL_AND_ARITY: Final = 2
Q0_FROZEN_LEAF_CANONICAL_NODES: Final = (
    (0, 0, 1),
    (0, 0, 3),
    (0, 0, 5),
    (0, 1, 0),
    (0, 1, 1),
    (0, 2),
    (0, 3, 0, 3, 0, ()),
    (0, 3, 1, 3, 0, ()),
    (0, 3, 5, 3, 0, ()),
    (0, 3, 0, 0, 0, ()),
    (0, 3, 0, 3, 1, ()),
    (0, 3, 0, 3, 0, ((0, True),)),
    (0, 3, 1, 1, 0, ()),
    (0, 4, 0),
    (0, 5, 0),
)

Q0_PROBE_SOURCE_ROWS: Final = (
    (
        1,
        0x3401,
        (1, 0x3401, b"hegel-odd-input/1", 5, (0, 1, 0, 1, 0)),
    ),
    (
        1,
        0x3401,
        (1, 0x3401, b"hegel-odd-input/1", 8, (1, 0, 1, 0, 1, 0, 1, 0)),
    ),
    (2, 0x3402, (1, 0x3402, b"hegel-sink-input/1", 0, 0, 0, 0)),
    (2, 0x3402, (1, 0x3402, b"hegel-sink-input/1", 4, 1, 2, 3)),
)


class OutputSortId(IntEnum):
    BOOL = 1
    BIT = 2
    SIGN = 3
    BOUNDED_INT = 4
    RATIONAL_VALUE = 5


class CellTag(IntEnum):
    UNDEFINED = 0
    DEFINED = 1


class NormalizationProfileId(IntEnum):
    GENERAL = 0
    ABSOLUTE_ROOT = 1
    CONST_NEGATIVE_ONE = 2
    CONST_ZERO = 3
    CONST_POSITIVE_ONE = 4
    TOP_LEVEL_AND2 = 5


class Q0TerminalStatusId(IntEnum):
    NOT_RUN = 0
    RUNNING = 1
    DUAL_EXHAUSTIVE_ORACLE_EQUIVALENCE_PASS = 2
    INCONCLUSIVE_RESOURCE_LIMIT = 3
    FAIL_SEMANTICS_MISMATCH = 4
    FAIL_IMPLEMENTATION_DISAGREEMENT = 5


class Q0ResourceGuardId(IntEnum):
    RAW_OPERATOR_APPLICATIONS = 1
    CANONICAL_SYNTAX_PROGRAMS = 2
    BEHAVIOR_CLASSES = 3
    TOTAL_FRONTIER_POINTS = 4
    FRONTIER_POINTS_PER_CLASS = 5
    SATURATION_ROUNDS = 6
    WALL_TIME = 7
    RESIDENT_MEMORY = 8
    OUTPUT_BYTES = 9
    TOTAL_CONTINUATION_BANK_POINTS = 10
    CONTINUATION_BANK_POINTS_PER_CLASS = 11


Q0_RESOURCE_GUARD_REGISTRY: Final = tuple(
    (int(guard_id), guard_id.name.encode("ascii"))
    for guard_id in Q0ResourceGuardId
)


class QuotientContractError(ValueError):
    """Stable fail-closed error at the quotient contract boundary."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> "None":
    raise QuotientContractError(code, detail)


def _uint(value: object, name: str, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        _fail("REJECT_Q0_SIGNATURE", f"{name} is outside 0..{maximum}")
    return value


@dataclass(frozen=True, slots=True)
class Q0ProbeInputV1:
    """The exact public composite universe used only by Q0 qualification."""

    rows: tuple[tuple[object, ...], ...] = Q0_PROBE_SOURCE_ROWS

    def __post_init__(self) -> None:
        # Python equality aliases bool and int (``True == 1``).  Wire identity
        # must instead be type-exact, so compare deterministic CBOR bytes.
        if (
            type(self.rows) is not tuple
            or canonical_cbor_encode(self.rows)
            != canonical_cbor_encode(Q0_PROBE_SOURCE_ROWS)
        ):
            _fail(
                "REJECT_Q0_PROBE_INPUT",
                "Q0 probe rows differ from the frozen ordered four-row projection",
            )
        from .phase3_q0_input_adapter_v1 import (
            observation_environment_from_object_v1,
        )

        for source_signature_id, source_tag, source_object in self.rows:
            environment = observation_environment_from_object_v1(source_object)
            if (
                type(source_signature_id) is not int
                or type(source_tag) is not int
                or environment.input_signature_id != source_signature_id
                or environment.input_object_tag != source_tag
                or canonical_cbor_encode(environment.canonical_input_object)
                != canonical_cbor_encode(source_object)
            ):
                _fail(
                    "REJECT_Q0_PROBE_INPUT",
                    "typed source-row replay differs from its Q0 binding",
                )

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q0_PROBE_INPUT_TAG,
            PROBE_INPUT_SCHEMA_ID,
            Q0_PROBE_INPUT_SIGNATURE_ID,
            Q0_PROJECTION_ROW_COUNT,
            self.rows,
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def universe_root(self) -> bytes:
        return content_hash(PROBE_UNIVERSE_ROOT_DOMAIN, self.canonical_object())

    def observation_environments(self) -> tuple[object, ...]:
        from .phase3_q0_input_adapter_v1 import (
            observation_environment_from_object_v1,
        )

        return tuple(
            observation_environment_from_object_v1(source_object)
            for _, _, source_object in self.rows
        )


def q0_semantic_binding_object_v1() -> tuple[object, ...]:
    """Return the exact target-blind semantic identity used by Q0 endpoints."""

    from .phase3_q0_input_adapter_v1 import ADAPTER_SCHEMA_ID

    return (
        1,
        SEMANTIC_BINDING_SCHEMA_ID,
        DSL_VERSION.encode("ascii"),
        DSL_FREEZE_VERSION.encode("ascii"),
        CLOSURE_SEMANTICS_VERSION.encode("ascii"),
        Q0_FREEZE_VERSION.encode("ascii"),
        Q0_QUALIFICATION_ID.encode("ascii"),
        bytes.fromhex(NORMATIVE_DOCUMENT_SHA256),
        q0_projection_manifest_root_v1(),
        Q0_CHILD_DSL_SPEC_ROOT,
        Q0_OPERATOR_SEMANTICS_ROOT,
        Q0_IDENTIFIER_REGISTRY_ROOT,
        Q0_CANONICAL_AST_SCHEMA_ROOT,
        Q0_CANONICAL_CBOR_PROFILE_ROOT,
        ADAPTER_SCHEMA_ID.encode("ascii"),
        Q0_PROJECTION_ID.encode("ascii"),
        Q0ProbeInputV1().universe_root,
    )


def q0_semantic_binding_root_v1() -> bytes:
    return content_hash(
        SEMANTIC_BINDING_ROOT_DOMAIN,
        q0_semantic_binding_object_v1(),
    )


def q0_projection_manifest_object_v1() -> tuple[object, ...]:
    """Canonical content of the exact executable Q0 micro projection."""

    capacities = tuple(
        (int(sort_id), normalization_witness_capacity_v1(sort_id))
        for sort_id in OutputSortId
    )
    return (
        1,
        PROJECTION_MANIFEST_SCHEMA_ID,
        Q0_PROJECTION_ID.encode("ascii"),
        Q0_FROZEN_LEAF_CANONICAL_NODES,
        Q0_ALLOWED_UNARY_OPERATOR_IDS,
        Q0_ALLOWED_BINARY_OPERATOR_IDS,
        Q0_ALLOWED_APPROX_TOLERANCE_IDS,
        Q0_TOP_LEVEL_AND_ARITY,
        (
            Q0_PROJECTION_MAX_AST_DEPTH,
            Q0_PROJECTION_MAX_NODE_COUNT,
            Q0_PROJECTION_MAX_TOP_LEVEL_CLAUSES,
            Q0_PROJECTION_MAX_AGGREGATE_LEAVES,
            Q0_INHERITED_MAX_SCALAR_PARAMETER_OCCURRENCES,
            Q0_INHERITED_MAX_SCOPE_CLAUSES,
            Q0_INHERITED_MAX_DISTINCT_BIT_SLOTS,
        ),
        (
            Q0_MAX_RAW_APPLICATIONS,
            Q0_MAX_CANONICAL_SYNTAX,
            Q0_MAX_BEHAVIOR_CLASSES,
            Q0_MAX_FRONTIER_POINTS,
            Q0_MAX_FRONTIER_POINTS_PER_CLASS,
            Q0_MAX_CONTINUATION_BANK_POINTS,
            Q0_MAX_CONTINUATION_BANK_POINTS_PER_CLASS,
            Q0_MAX_SATURATION_ROUNDS,
            Q0_MAX_OUTPUT_BYTES,
            Q0_MAX_WALL_TIME_SECONDS,
            Q0_MAX_MEMORY_BYTES,
        ),
        capacities,
        b"LEX_MIN_REAL_AST_UP_TO_SORT_CAPACITY",
        b"EXPAND_EACH_BANK_REP_ONCE_REGARDLESS_OF_VISIBLE_DOMINANCE",
        b"PUBLIC_CLASS_ARCHIVE_VISIBLE_FRONTIER_ONLY",
        Q0_RESOURCE_GUARD_REGISTRY,
        Q0_COVERAGE_CODES,
        Q0_COVERAGE_RECORD_LENGTH,
    )


def q0_projection_manifest_root_v1() -> bytes:
    return content_hash(
        PROJECTION_MANIFEST_ROOT_DOMAIN,
        q0_projection_manifest_object_v1(),
    )


@dataclass(frozen=True, slots=True)
class BehaviorCellV1:
    """One explicit bottom or sort-typed exact output cell."""

    defined: bool
    value: object = None

    @classmethod
    def bottom(cls) -> "BehaviorCellV1":
        return cls(False, None)

    @classmethod
    def exact(cls, value: object) -> "BehaviorCellV1":
        return cls(True, value)

    def canonical_object(self, output_sort_id: OutputSortId) -> tuple[object, ...]:
        if type(self.defined) is not bool:
            _fail("REJECT_Q0_BEHAVIOR_CELL", "defined must be a bool")
        if not self.defined:
            if self.value is not None:
                _fail("REJECT_Q0_BEHAVIOR_CELL", "bottom must not carry a value")
            return (int(CellTag.UNDEFINED),)
        value = self.value
        if output_sort_id is OutputSortId.BOOL:
            if type(value) is not bool:
                _fail("REJECT_Q0_BEHAVIOR_CELL", "Bool cell requires canonical bool")
            payload: object = value
        elif output_sort_id is OutputSortId.BIT:
            if type(value) is not int or value not in (0, 1):
                _fail("REJECT_Q0_BEHAVIOR_CELL", "Bit cell requires uint 0 or 1")
            payload = value
        elif output_sort_id is OutputSortId.SIGN:
            if type(value) is not int or value not in (-1, 0, 1):
                _fail("REJECT_Q0_BEHAVIOR_CELL", "Sign cell is outside -1,0,1")
            payload = value
        elif output_sort_id is OutputSortId.BOUNDED_INT:
            if type(value) is not int or not -8 <= value <= 8:
                _fail("REJECT_Q0_BEHAVIOR_CELL", "BoundedInt cell is outside [-8,8]")
            payload = value
        elif output_sort_id is OutputSortId.RATIONAL_VALUE:
            if type(value) is not Fraction:
                _fail("REJECT_Q0_BEHAVIOR_CELL", "RationalValue requires Fraction")
            if value.denominator <= 0:
                _fail("REJECT_Q0_BEHAVIOR_CELL", "rational denominator must be positive")
            if value not in RATIONAL_VALUE_GRID:
                _fail(
                    "REJECT_Q0_BEHAVIOR_CELL",
                    "RationalValue cell is outside the frozen exact grid",
                )
            payload = (value.numerator, value.denominator)
        else:  # pragma: no cover - IntEnum closes this branch
            _fail("REJECT_Q0_BEHAVIOR_CELL", "unknown output sort")
        return (int(CellTag.DEFINED), payload)


@dataclass(frozen=True, slots=True)
class BehaviorBlobV1:
    input_signature_id: int
    frozen_universe_root: bytes
    output_sort_id: OutputSortId
    cells: tuple[BehaviorCellV1, ...]

    def __post_init__(self) -> None:
        _uint(self.input_signature_id, "input_signature_id", 0xFFFF)
        if type(self.frozen_universe_root) is not bytes or len(self.frozen_universe_root) != 32:
            _fail("REJECT_Q0_BEHAVIOR_BLOB", "frozen_universe_root must be 32 bytes")
        if not isinstance(self.output_sort_id, OutputSortId):
            _fail("REJECT_Q0_BEHAVIOR_BLOB", "output_sort_id must be OutputSortId")
        if not self.cells:
            _fail("REJECT_Q0_EMPTY_UNIVERSE", "behavior universe must be nonempty")
        if any(not isinstance(cell, BehaviorCellV1) for cell in self.cells):
            _fail("REJECT_Q0_BEHAVIOR_BLOB", "cells must be BehaviorCellV1")

    def canonical_object(self) -> tuple[object, ...]:
        encoded_cells = tuple(
            cell.canonical_object(self.output_sort_id) for cell in self.cells
        )
        return (
            1,
            Q0_BEHAVIOR_BLOB_TAG,
            BEHAVIOR_BLOB_SCHEMA_ID,
            self.input_signature_id,
            self.frozen_universe_root,
            int(self.output_sort_id),
            len(encoded_cells),
            encoded_cells,
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def behavior_id(self) -> bytes:
        return content_hash(BEHAVIOR_ID_DOMAIN, self.canonical_object())


@dataclass(frozen=True, slots=True)
class FutureAdmissibilitySignatureV1:
    output_sort_id: OutputSortId
    ast_depth: int
    ast_node_count: int
    scalar_parameter_occurrence_count: int
    aggregate_leaf_count: int
    distinct_bit_slot_bitmap: int
    scope_clause_count: int
    top_level_clause_count: int
    old_law_composition_depth: int
    normalization_profile_id: NormalizationProfileId
    mdl_length_q32: int

    def __post_init__(self) -> None:
        if not isinstance(self.output_sort_id, OutputSortId):
            _fail("REJECT_Q0_SIGNATURE", "output_sort_id must be OutputSortId")
        _uint(self.ast_depth, "ast_depth", 0xFF)
        _uint(self.ast_node_count, "ast_node_count", 0xFFFF)
        _uint(
            self.scalar_parameter_occurrence_count,
            "scalar_parameter_occurrence_count",
            0xFF,
        )
        _uint(self.aggregate_leaf_count, "aggregate_leaf_count", 0xFF)
        _uint(self.distinct_bit_slot_bitmap, "distinct_bit_slot_bitmap", 0xFF)
        _uint(self.scope_clause_count, "scope_clause_count", 0xFF)
        _uint(self.top_level_clause_count, "top_level_clause_count", 0xFF)
        _uint(self.old_law_composition_depth, "old_law_composition_depth", 0xFF)
        if self.old_law_composition_depth != 0:
            _fail(
                "REJECT_Q0_UNREPRESENTED_LAW_COMPOSITION",
                "v1.6 program AST has no old-law composition node; depth must be zero",
            )
        if not isinstance(self.normalization_profile_id, NormalizationProfileId):
            _fail("REJECT_Q0_SIGNATURE", "normalization profile is unregistered")
        _uint(self.mdl_length_q32, "mdl_length_q32", (1 << 64) - 1)

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q0_CONSTRUCTION_SIGNATURE_TAG,
            CONSTRUCTION_SIGNATURE_SCHEMA_ID,
            int(self.output_sort_id),
            self.ast_depth,
            self.ast_node_count,
            self.scalar_parameter_occurrence_count,
            self.aggregate_leaf_count,
            self.distinct_bit_slot_bitmap,
            self.scope_clause_count,
            self.top_level_clause_count,
            self.old_law_composition_depth,
            int(self.normalization_profile_id),
            self.mdl_length_q32,
        )

    def dominates(self, other: "FutureAdmissibilitySignatureV1") -> bool:
        """Return whether self can safely replace other for continuation."""

        if not isinstance(other, FutureAdmissibilitySignatureV1):
            raise TypeError("other must be FutureAdmissibilitySignatureV1")
        if (
            self.output_sort_id != other.output_sort_id
            or self.normalization_profile_id != other.normalization_profile_id
        ):
            return False
        subset = (
            self.distinct_bit_slot_bitmap | other.distinct_bit_slot_bitmap
        ) == other.distinct_bit_slot_bitmap
        left = (
            self.ast_depth,
            self.ast_node_count,
            self.scalar_parameter_occurrence_count,
            self.aggregate_leaf_count,
            self.scope_clause_count,
            self.top_level_clause_count,
            self.old_law_composition_depth,
            self.mdl_length_q32,
        )
        right = (
            other.ast_depth,
            other.ast_node_count,
            other.scalar_parameter_occurrence_count,
            other.aggregate_leaf_count,
            other.scope_clause_count,
            other.top_level_clause_count,
            other.old_law_composition_depth,
            other.mdl_length_q32,
        )
        no_worse = subset and all(a <= b for a, b in zip(left, right, strict=True))
        strict = self.distinct_bit_slot_bitmap != other.distinct_bit_slot_bitmap or any(
            a < b for a, b in zip(left, right, strict=True)
        )
        return no_worse and strict


def future_signature_from_ast_v1(canonical_ast: object) -> FutureAdmissibilitySignatureV1:
    """Recompute the complete continuation signature from one strict AST."""

    from .phase3_m3_bounded_enumerator_v1 import program_mdl_length_q32
    from .strict_ast_v1 import CanonicalAst

    if not isinstance(canonical_ast, CanonicalAst):
        raise TypeError("canonical_ast must be CanonicalAst")
    sort_by_name = {
        "Bool": OutputSortId.BOOL,
        "Bit": OutputSortId.BIT,
        "Sign": OutputSortId.SIGN,
        "BoundedInt": OutputSortId.BOUNDED_INT,
        "RationalValue": OutputSortId.RATIONAL_VALUE,
    }
    try:
        output_sort_id = sort_by_name[canonical_ast.metrics.output_sort]
    except KeyError:
        _fail("REJECT_Q0_SIGNATURE", "strict AST carries an unknown output sort")

    node = canonical_ast.value[1]
    if not isinstance(node, tuple) or not node or type(node[0]) is not int:
        _fail("REJECT_Q0_SIGNATURE", "strict AST node is malformed")
    profile = NormalizationProfileId.GENERAL
    if node[0] == 0 and len(node) == 3 and node[1] == 0:
        profile = {
            1: NormalizationProfileId.CONST_NEGATIVE_ONE,
            3: NormalizationProfileId.CONST_ZERO,
            5: NormalizationProfileId.CONST_POSITIVE_ONE,
        }.get(node[2], NormalizationProfileId.GENERAL)
    elif node[0] == 1 and len(node) == 3 and node[1] == 2:
        profile = NormalizationProfileId.ABSOLUTE_ROOT
    elif (
        node[0] == 4
        and len(node) == 2
        and isinstance(node[1], tuple)
        and len(node[1]) == 2
    ):
        profile = NormalizationProfileId.TOP_LEVEL_AND2

    bitmask = 0
    for slot in canonical_ast.metrics.distinct_bit_slots:
        bitmask |= 1 << slot
    return FutureAdmissibilitySignatureV1(
        output_sort_id=output_sort_id,
        ast_depth=canonical_ast.metrics.depth,
        ast_node_count=canonical_ast.metrics.node_count,
        scalar_parameter_occurrence_count=(
            canonical_ast.metrics.scalar_parameter_occurrences
        ),
        aggregate_leaf_count=canonical_ast.metrics.aggregate_leaf_count,
        distinct_bit_slot_bitmap=bitmask,
        scope_clause_count=canonical_ast.metrics.scope_clause_count,
        top_level_clause_count=canonical_ast.metrics.top_level_clause_count,
        old_law_composition_depth=0,
        normalization_profile_id=profile,
        mdl_length_q32=program_mdl_length_q32(canonical_ast),
    )


def _validate_q0_canonical_node_v1(node: object) -> None:
    """Reject any strict-v1.6 construct outside the exact Q0 subgrammar."""

    if not isinstance(node, tuple) or not node or type(node[0]) is not int:
        _fail("REJECT_Q0_PROJECTION_GRAMMAR", "canonical node is malformed")
    tag = node[0]
    if tag == 0:
        encoded = canonical_cbor_encode(node)
        allowed = {
            canonical_cbor_encode(leaf)
            for leaf in Q0_FROZEN_LEAF_CANONICAL_NODES
        }
        if encoded not in allowed:
            _fail(
                "REJECT_Q0_PROJECTION_GRAMMAR",
                "leaf is outside the frozen fifteen-leaf Q0 manifest",
            )
        return
    if tag == 1:
        if (
            len(node) != 3
            or type(node[1]) is not int
            or node[1] not in Q0_ALLOWED_UNARY_OPERATOR_IDS
        ):
            _fail("REJECT_Q0_PROJECTION_GRAMMAR", "unary node is outside Q0")
        _validate_q0_canonical_node_v1(node[2])
        return
    if tag == 2:
        if (
            len(node) != 4
            or type(node[1]) is not int
            or node[1] not in Q0_ALLOWED_BINARY_OPERATOR_IDS
        ):
            _fail("REJECT_Q0_PROJECTION_GRAMMAR", "binary node is outside Q0")
        _validate_q0_canonical_node_v1(node[2])
        _validate_q0_canonical_node_v1(node[3])
        return
    if tag == 3:
        if (
            len(node) != 5
            or type(node[1]) is not int
            or node[1] != 0
            or type(node[4]) is not int
            or node[4] not in Q0_ALLOWED_APPROX_TOLERANCE_IDS
        ):
            _fail(
                "REJECT_Q0_PROJECTION_GRAMMAR",
                "approximate-equality node is outside Q0",
            )
        _validate_q0_canonical_node_v1(node[2])
        _validate_q0_canonical_node_v1(node[3])
        return
    if tag == 4:
        if (
            len(node) != 2
            or not isinstance(node[1], tuple)
            or len(node[1]) != Q0_TOP_LEVEL_AND_ARITY
        ):
            _fail("REJECT_Q0_PROJECTION_GRAMMAR", "conjunction is not Q0 AND2")
        for child in node[1]:
            _validate_q0_canonical_node_v1(child)
        return
    _fail("REJECT_Q0_PROJECTION_GRAMMAR", "canonical tag is outside Q0")


@dataclass(frozen=True, slots=True)
class FrontierEntryV1:
    signature: FutureAdmissibilitySignatureV1
    normalization_witness_rank: int
    representative_ast_cbor: bytes
    representative_ast_hash: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.signature, FutureAdmissibilitySignatureV1):
            _fail("REJECT_Q0_FRONTIER_ENTRY", "signature has the wrong type")
        capacity = normalization_witness_capacity_v1(
            self.signature.output_sort_id
        )
        if (
            type(self.normalization_witness_rank) is not int
            or not 0 <= self.normalization_witness_rank < capacity
        ):
            _fail(
                "REJECT_Q0_FRONTIER_ENTRY",
                "normalization witness rank is outside the output-sort capacity",
            )
        if type(self.representative_ast_cbor) is not bytes or not self.representative_ast_cbor:
            _fail("REJECT_Q0_FRONTIER_ENTRY", "representative AST CBOR is empty")
        if type(self.representative_ast_hash) is not bytes or len(self.representative_ast_hash) != 32:
            _fail("REJECT_Q0_FRONTIER_ENTRY", "representative AST hash must be 32 bytes")
        # Import lazily so the behavior/signature wire remains lightweight,
        # while every actual frontier representative is nevertheless replayed
        # through the authoritative v1.6 strict admission boundary.
        from .strict_ast_shrink6_v1 import decode_shrink6_canonical_ast

        try:
            replay = decode_shrink6_canonical_ast(self.representative_ast_cbor)
        except ValueError as error:
            _fail(
                "REJECT_Q0_FRONTIER_AST",
                f"representative is not an admitted v1.6 canonical AST: {error}",
            )
        if replay.digest != self.representative_ast_hash:
            _fail(
                "REJECT_Q0_FRONTIER_AST_HASH",
                "representative AST hash does not match strict replay",
            )
        _validate_q0_canonical_node_v1(replay.value[1])
        metrics = replay.metrics
        if (
            metrics.depth > Q0_PROJECTION_MAX_AST_DEPTH
            or metrics.node_count > Q0_PROJECTION_MAX_NODE_COUNT
            or metrics.top_level_clause_count
            > Q0_PROJECTION_MAX_TOP_LEVEL_CLAUSES
            or metrics.aggregate_leaf_count > Q0_PROJECTION_MAX_AGGREGATE_LEAVES
            or metrics.scalar_parameter_occurrences
            > Q0_INHERITED_MAX_SCALAR_PARAMETER_OCCURRENCES
            or metrics.scope_clause_count > Q0_INHERITED_MAX_SCOPE_CLAUSES
            or len(metrics.distinct_bit_slots)
            > Q0_INHERITED_MAX_DISTINCT_BIT_SLOTS
        ):
            _fail(
                "REJECT_Q0_PROJECTION_LIMIT",
                "frontier representative is outside the frozen Q0 micro projection",
            )
        if future_signature_from_ast_v1(replay) != self.signature:
            _fail(
                "REJECT_Q0_FRONTIER_SIGNATURE",
                "continuation signature does not match strict AST replay",
            )

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q0_FRONTIER_ENTRY_TAG,
            FRONTIER_ENTRY_SCHEMA_ID,
            self.signature.canonical_object(),
            self.normalization_witness_rank,
            self.representative_ast_cbor,
            self.representative_ast_hash,
        )

    @property
    def entry_id(self) -> bytes:
        return content_hash(FRONTIER_ENTRY_ID_DOMAIN, self.canonical_object())


def normalization_witness_capacity_v1(output_sort_id: OutputSortId) -> int:
    """Return the exact distinct-child witness capacity required by v1.6."""

    if not isinstance(output_sort_id, OutputSortId):
        raise TypeError("output_sort_id must be OutputSortId")
    # ``difference(x, x)`` and AND deduplication inspect canonical child
    # identity.  Their maximum repeated child arity is two.  Other output
    # sorts feed no identity-sensitive rewrite in the surviving grammar.
    return 2 if output_sort_id in {
        OutputSortId.BOOL,
        OutputSortId.RATIONAL_VALUE,
    } else 1


def pareto_frontier_v1(entries: Iterable[FrontierEntryV1]) -> tuple[FrontierEntryV1, ...]:
    """Return the deterministic MDL/multiplicity-aware continuation frontier."""

    material = tuple(entries)
    if any(not isinstance(entry, FrontierEntryV1) for entry in material):
        raise TypeError("entries must contain only FrontierEntryV1")
    by_exact_signature: dict[
        tuple[object, ...], dict[bytes, FrontierEntryV1]
    ] = {}
    for entry in material:
        key = entry.signature.canonical_object()
        cohort = by_exact_signature.setdefault(key, {})
        prior = cohort.get(entry.representative_ast_cbor)
        if prior is not None and prior.representative_ast_hash != entry.representative_ast_hash:
            _fail(
                "FAIL_SHA256_PREIMAGE_COLLISION",
                "one representative AST has two strict hashes",
            )
        cohort[entry.representative_ast_cbor] = entry

    cohorts: list[tuple[FutureAdmissibilitySignatureV1, tuple[FrontierEntryV1, ...]]] = []
    for cohort in by_exact_signature.values():
        ordered = tuple(sorted(cohort.values(), key=lambda entry: entry.representative_ast_cbor))
        signature = ordered[0].signature
        capacity = normalization_witness_capacity_v1(signature.output_sort_id)
        ranked = tuple(
            replace(entry, normalization_witness_rank=rank)
            for rank, entry in enumerate(ordered[:capacity])
        )
        cohorts.append((signature, ranked))

    retained: list[FrontierEntryV1] = []
    for index, (signature, cohort) in enumerate(cohorts):
        dominated = any(
            other_signature.dominates(signature)
            and len(other_cohort) >= len(cohort)
            for other_index, (other_signature, other_cohort) in enumerate(cohorts)
            if other_index != index
        )
        if not dominated:
            retained.extend(cohort)
    return tuple(
        sorted(
            retained,
            key=lambda entry: (
                canonical_cbor_encode(entry.signature.canonical_object()),
                entry.normalization_witness_rank,
                entry.representative_ast_cbor,
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class QuotientClassRecordV1:
    """One exact behavior class and its supplied visible continuation frontier.

    Closure completeness is established separately by the bound cohort bank,
    fixed-point state, and host replay rather than by this record alone.
    """

    class_index: int
    behavior: BehaviorBlobV1
    frontier: tuple[FrontierEntryV1, ...]

    def __post_init__(self) -> None:
        _uint(self.class_index, "class_index", 0xFFFFFFFF)
        if not isinstance(self.behavior, BehaviorBlobV1):
            _fail("REJECT_Q0_QUOTIENT_CLASS", "behavior has the wrong type")
        probe = Q0ProbeInputV1()
        if (
            self.behavior.input_signature_id != Q0_PROBE_INPUT_SIGNATURE_ID
            or self.behavior.frozen_universe_root != probe.universe_root
            or len(self.behavior.cells) != Q0_PROJECTION_ROW_COUNT
        ):
            _fail(
                "REJECT_Q0_BEHAVIOR_BINDING",
                "quotient behavior is not bound to the exact four-row Q0 probe",
            )
        if not self.frontier or any(
            not isinstance(entry, FrontierEntryV1) for entry in self.frontier
        ):
            _fail(
                "REJECT_Q0_QUOTIENT_CLASS",
                "frontier must contain at least one FrontierEntryV1",
            )
        expected = pareto_frontier_v1(self.frontier)
        if self.frontier != expected:
            _fail(
                "REJECT_Q0_QUOTIENT_CLASS",
                "supplied frontier is not in canonical Pareto ordering",
            )
        if any(
            entry.signature.output_sort_id != self.behavior.output_sort_id
            for entry in self.frontier
        ):
            _fail(
                "REJECT_Q0_QUOTIENT_CLASS",
                "frontier output sort differs from behavior output sort",
            )
        from .phase3_q0_evaluator_v1 import (
            Q0EvaluatorError,
            evaluate_canonical_ast_on_environments_v1,
        )
        from .strict_ast_shrink6_v1 import decode_shrink6_canonical_ast

        environments = probe.observation_environments()
        for entry in self.frontier:
            replay = decode_shrink6_canonical_ast(entry.representative_ast_cbor)
            try:
                raw_cells = evaluate_canonical_ast_on_environments_v1(
                    replay,
                    environments,  # type: ignore[arg-type]
                )
            except Q0EvaluatorError as error:
                _fail(
                    "REJECT_Q0_FRONTIER_BEHAVIOR_REPLAY",
                    f"representative evaluation failed: {error}",
                )
            from .phase3_q0_input_adapter_v1 import BOTTOM

            replay_cells = tuple(
                BehaviorCellV1.bottom()
                if value is BOTTOM
                else BehaviorCellV1.exact(value)
                for value in raw_cells
            )
            replay_behavior = BehaviorBlobV1(
                Q0_PROBE_INPUT_SIGNATURE_ID,
                probe.universe_root,
                entry.signature.output_sort_id,
                replay_cells,
            )
            if replay_behavior.canonical_bytes != self.behavior.canonical_bytes:
                _fail(
                    "REJECT_Q0_FRONTIER_BEHAVIOR_MISMATCH",
                    "frontier representative does not replay to its behavior class",
                )

    @property
    def minimum_mdl_length_q32(self) -> int:
        return min(entry.signature.mdl_length_q32 for entry in self.frontier)

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q0_QUOTIENT_CLASS_TAG,
            QUOTIENT_CLASS_SCHEMA_ID,
            self.class_index,
            self.behavior.canonical_object(),
            self.behavior.behavior_id,
            len(self.frontier),
            tuple(entry.canonical_object() for entry in self.frontier),
            self.minimum_mdl_length_q32,
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def record_id(self) -> bytes:
        return content_hash(
            QUOTIENT_CLASS_RECORD_ID_DOMAIN, self.canonical_object()
        )


def quotient_class_archive_root_v1(
    records: Iterable[QuotientClassRecordV1],
) -> bytes:
    material = tuple(records)
    if any(not isinstance(record, QuotientClassRecordV1) for record in material):
        raise TypeError("records must contain only QuotientClassRecordV1")
    expected = tuple(
        sorted(
            material,
            key=lambda record: (
                record.behavior.behavior_id,
                record.behavior.canonical_bytes,
            ),
        )
    )
    if material != expected:
        _fail(
            "REJECT_Q0_QUOTIENT_ARCHIVE",
            "quotient classes are not in canonical behavior order",
        )
    if tuple(record.class_index for record in material) != tuple(range(len(material))):
        _fail(
            "REJECT_Q0_QUOTIENT_ARCHIVE",
            "quotient class indices are not contiguous",
        )
    seen: dict[bytes, bytes] = {}
    for record in material:
        prior = seen.get(record.behavior.behavior_id)
        if prior is not None and prior != record.behavior.canonical_bytes:
            _fail(
                "FAIL_SHA256_PREIMAGE_COLLISION",
                "one behavior digest has two distinct canonical preimages",
            )
        if prior is not None:
            _fail(
                "REJECT_Q0_QUOTIENT_ARCHIVE",
                "duplicate behavior class",
            )
        seen[record.behavior.behavior_id] = record.behavior.canonical_bytes
    return rfc6962_root([record.canonical_object() for record in material])


def _root32(value: object, name: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        _fail("REJECT_Q0_SATURATION_RECEIPT", f"{name} must be 32 bytes")
    return value


@dataclass(frozen=True, slots=True)
class Q0SaturationReceiptV1:
    """Host-only receipt created after both independent endpoints agree."""

    terminal_status_id: Q0TerminalStatusId
    syntax_raw_operator_application_count: int
    quotient_raw_operator_application_count: int
    canonical_syntax_program_count: int
    behavior_class_count: int
    frontier_point_count: int
    maximum_frontier_points_per_class: int
    saturation_round_count: int
    syntax_program_archive_root: bytes
    syntax_oracle_class_archive_root: bytes
    quotient_engine_class_archive_root: bytes
    syntax_operator_coverage_root: bytes
    quotient_operator_coverage_root: bytes
    python_implementation_root: bytes
    rust_implementation_root: bytes
    python_endpoint_output_root: bytes
    rust_endpoint_output_root: bytes
    host_replay_class_archive_root: bytes

    def __post_init__(self) -> None:
        if self.terminal_status_id is not Q0TerminalStatusId.DUAL_EXHAUSTIVE_ORACLE_EQUIVALENCE_PASS:
            _fail(
                "REJECT_Q0_SATURATION_RECEIPT",
                "the dual PASS receipt is absent for NOT_RUN, partial, or failed runs",
            )
        bounded_counts = (
            (
                self.syntax_raw_operator_application_count,
                "syntax_raw_operator_application_count",
                Q0_MAX_RAW_APPLICATIONS,
            ),
            (
                self.quotient_raw_operator_application_count,
                "quotient_raw_operator_application_count",
                Q0_MAX_RAW_APPLICATIONS,
            ),
            (
                self.canonical_syntax_program_count,
                "canonical_syntax_program_count",
                Q0_MAX_CANONICAL_SYNTAX,
            ),
            (
                self.behavior_class_count,
                "behavior_class_count",
                Q0_MAX_BEHAVIOR_CLASSES,
            ),
            (
                self.frontier_point_count,
                "frontier_point_count",
                Q0_MAX_FRONTIER_POINTS,
            ),
            (
                self.maximum_frontier_points_per_class,
                "maximum_frontier_points_per_class",
                Q0_MAX_FRONTIER_POINTS_PER_CLASS,
            ),
            (
                self.saturation_round_count,
                "saturation_round_count",
                Q0_MAX_SATURATION_ROUNDS,
            ),
        )
        for value, name, maximum in bounded_counts:
            if type(value) is not int or not 1 <= value <= maximum:
                _fail(
                    "REJECT_Q0_SATURATION_RECEIPT",
                    f"{name} must be in 1..{maximum}",
                )
        if not (
            self.behavior_class_count
            <= self.canonical_syntax_program_count
            and self.behavior_class_count
            <= self.frontier_point_count
            and self.maximum_frontier_points_per_class
            <= self.frontier_point_count
        ):
            _fail(
                "REJECT_Q0_SATURATION_RECEIPT",
                "class, syntax, and frontier counts are inconsistent",
            )
        root_fields = (
            "syntax_program_archive_root",
            "syntax_oracle_class_archive_root",
            "quotient_engine_class_archive_root",
            "syntax_operator_coverage_root",
            "quotient_operator_coverage_root",
            "python_implementation_root",
            "rust_implementation_root",
            "python_endpoint_output_root",
            "rust_endpoint_output_root",
            "host_replay_class_archive_root",
        )
        for name in root_fields:
            _root32(getattr(self, name), name)
        if not (
            self.syntax_oracle_class_archive_root
            == self.quotient_engine_class_archive_root
            == self.host_replay_class_archive_root
        ):
            _fail(
                "REJECT_Q0_SATURATION_RECEIPT",
                "syntax, quotient, and host class archive roots differ",
            )

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q0_SATURATION_RECEIPT_TAG,
            SATURATION_RECEIPT_SCHEMA_ID,
            Q0_QUALIFICATION_ID.encode("ascii"),
            DSL_VERSION.encode("ascii"),
            CLOSURE_SEMANTICS_VERSION.encode("ascii"),
            Q0_FREEZE_VERSION.encode("ascii"),
            Q0_PROJECTION_ID.encode("ascii"),
            Q0ProbeInputV1().universe_root,
            int(self.terminal_status_id),
            self.syntax_raw_operator_application_count,
            self.quotient_raw_operator_application_count,
            self.canonical_syntax_program_count,
            self.behavior_class_count,
            self.frontier_point_count,
            self.maximum_frontier_points_per_class,
            self.saturation_round_count,
            True,  # zero-delta full round completed
            True,  # work queue empty
            True,  # all typed operator x frontier tuples closed
            True,  # no resource guard hit
            True,  # exhaustive syntax oracle completed
            self.syntax_program_archive_root,
            self.syntax_oracle_class_archive_root,
            self.quotient_engine_class_archive_root,
            self.syntax_operator_coverage_root,
            self.quotient_operator_coverage_root,
            self.python_implementation_root,
            self.rust_implementation_root,
            self.python_endpoint_output_root,
            self.rust_endpoint_output_root,
            self.host_replay_class_archive_root,
            Q0_READINESS_GATE_TOTAL,
            (1 << Q0_READINESS_GATE_TOTAL) - 1,
            0,  # Q1 NOT_RUN
            None,  # Q1 output root
            0,  # Q2 NOT_RUN
            False,  # role evaluation performed
            None,  # M3 formal roots
            False,  # outside certificate issued
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def receipt_root(self) -> bytes:
        return content_hash(
            SATURATION_RECEIPT_ROOT_DOMAIN, self.canonical_object()
        )


def bool_bit_role_match_v1(
    program_sort_id: OutputSortId,
    program_cell: BehaviorCellV1,
    target_bit: int,
) -> bool:
    """Explicit Q2 predicate adapter; it never changes quotient class identity."""

    if type(target_bit) is not int or target_bit not in (0, 1):
        _fail("REJECT_Q0_TARGET_BIT", "target truth must be uint 0 or 1")
    if program_sort_id not in (OutputSortId.BOOL, OutputSortId.BIT):
        return False
    if not isinstance(program_cell, BehaviorCellV1):
        raise TypeError("program_cell must be BehaviorCellV1")
    if not program_cell.defined:
        return False
    if program_sort_id is OutputSortId.BOOL and type(program_cell.value) is bool:
        return int(program_cell.value) == target_bit
    if (
        program_sort_id is OutputSortId.BIT
        and type(program_cell.value) is int
        and program_cell.value in (0, 1)
    ):
        return program_cell.value == target_bit
    return False


Q0_READINESS_GATES: Final = (
    "NORMATIVE_DIRECTION_BYTES_BOUND",
    "V16_DSL_TYPING_AND_REGISTRY_ROOTS_QUALIFIED",
    "INPUT_SIGNATURE_OBSERVATION_ADAPTERS_QUALIFIED",
    "BEHAVIOR_AND_BOTTOM_CODEC_QUALIFIED",
    "UNIVERSE_ONLY_BINDINGS_QUALIFIED",
    "EXACT_EQUIVALENCE_CONTRACT_QUALIFIED",
    "CONSTRUCTION_SIGNATURE_QUALIFIED",
    "PARETO_DOMINANCE_AND_MDL_QUALIFIED",
    "PER_OPERATOR_CONGRUENCE_QUALIFIED",
    "STRUCTURAL_INDUCTION_COMPLETENESS_QUALIFIED",
    "EXHAUSTIVE_MICRO_ORACLE_EQUALITY_QUALIFIED",
    "COLLISION_BOTTOM_SORT_ADVERSARIAL_VECTORS_PASS",
    "TARGET_TRUTH_AND_SPLIT_INPUT_ISOLATION_PASS",
    "DUAL_HOST_AGREEMENT_Q1_OUTPUTS_NULL_NOT_RUN",
)

if len(Q0_READINESS_GATES) != Q0_READINESS_GATE_TOTAL:
    raise AssertionError("Q0 readiness gate count drift")


__all__ = [name for name in globals() if name.isupper()] + [
    "BehaviorBlobV1",
    "BehaviorCellV1",
    "CellTag",
    "FrontierEntryV1",
    "FutureAdmissibilitySignatureV1",
    "NormalizationProfileId",
    "OutputSortId",
    "Q0ProbeInputV1",
    "Q0ResourceGuardId",
    "Q0SaturationReceiptV1",
    "Q0TerminalStatusId",
    "QuotientClassRecordV1",
    "QuotientContractError",
    "bool_bit_role_match_v1",
    "future_signature_from_ast_v1",
    "normalization_witness_capacity_v1",
    "pareto_frontier_v1",
    "q0_projection_manifest_object_v1",
    "q0_projection_manifest_root_v1",
    "q0_semantic_binding_object_v1",
    "q0_semantic_binding_root_v1",
    "quotient_class_archive_root_v1",
]
