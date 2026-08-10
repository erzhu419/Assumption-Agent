"""Target-blind continuation contract for the Phase-3A-Q1 preflight.

Only construction resources that can affect a future admitted continuation
live in this module.  In particular, it has no behavior-to-target adapter,
truth row, split object, role matcher, certificate, or formal Q1 wire.  The
canonical object below is diagnostic identity for the capacity preflight; the
later Q1 formal-wire amendment must allocate its own numeric tags.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final, NoReturn

from .strict_ast_v1 import CanonicalAst


DIAGNOSTIC_SIGNATURE_SCHEMA_ID: Final = (
    b"hegel-phase3a-q1-future-admissibility-signature-diagnostic/1"
)
Q32_SCALE: Final = 1 << 32


class Q1QuotientContractError(ValueError):
    """Stable fail-closed error from the target-blind Q1 contract."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Q1QuotientContractError(code, detail)


def _uint(value: object, name: str, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        _fail("REJECT_Q1_CONSTRUCTION_SIGNATURE", f"{name} is outside uint range")
    return value


class OutputSortId(IntEnum):
    BOOL = 1
    BIT = 2
    SIGN = 3
    BOUNDED_INT = 4
    RATIONAL_VALUE = 5


class NormalizationProfileId(IntEnum):
    GENERAL = 0
    ABSOLUTE_ROOT = 1
    CONST_NEGATIVE_ONE = 2
    CONST_ZERO = 3
    CONST_POSITIVE_ONE = 4
    TOP_LEVEL_AND2 = 5


@dataclass(frozen=True, slots=True)
class FutureAdmissibilitySignatureV1:
    """Complete target-independent resource label for future composition."""

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
            _fail("REJECT_Q1_CONSTRUCTION_SIGNATURE", "output sort is unregistered")
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
                "REJECT_Q1_UNREPRESENTED_LAW_COMPOSITION",
                "v1.6 has no old-law-composition AST node",
            )
        if not isinstance(self.normalization_profile_id, NormalizationProfileId):
            _fail(
                "REJECT_Q1_CONSTRUCTION_SIGNATURE",
                "normalization profile is unregistered",
            )
        _uint(self.mdl_length_q32, "mdl_length_q32", (1 << 64) - 1)

    def canonical_object(self) -> tuple[object, ...]:
        """Return the tag-free diagnostic preimage in frozen field order."""

        return (
            1,
            DIAGNOSTIC_SIGNATURE_SCHEMA_ID,
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

    def resource_tuple(self) -> tuple[int, ...]:
        """Return only the semantic resource fields, excluding wire identity."""

        return (
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
        no_worse = subset and all(
            a <= b for a, b in zip(left, right, strict=True)
        )
        strict = self.distinct_bit_slot_bitmap != other.distinct_bit_slot_bitmap or any(
            a < b for a, b in zip(left, right, strict=True)
        )
        return no_worse and strict


def _elias_delta_length(one_based_index: int) -> int:
    if type(one_based_index) is not int or one_based_index < 1:
        _fail("FAIL_Q1_MDL_LENGTH", "invalid registry index")
    floor = one_based_index.bit_length() - 1
    return floor + 2 * ((floor + 1).bit_length() - 1) + 1


def _scope_length(extension: object) -> int:
    if not isinstance(extension, tuple) or len(extension) not in (0, 1, 2):
        _fail("FAIL_Q1_MDL_LENGTH", "invalid scope extension")
    return (1 if len(extension) == 0 else 2) + 3 * len(extension)


def _node_mdl_bits(node: object) -> int:
    if not isinstance(node, tuple) or not node or type(node[0]) is not int:
        _fail("FAIL_Q1_MDL_LENGTH", "malformed canonical node")
    tag = node[0]
    if tag == 0:
        if len(node) < 2 or type(node[1]) is not int:
            _fail("FAIL_Q1_MDL_LENGTH", "malformed leaf")
        leaf = node[1]
        if leaf == 0 and len(node) == 3:
            return 2 + 3 + 3
        if leaf == 1 and len(node) == 3:
            return 2 + 3 + _elias_delta_length(int(node[2]) + 1)
        if leaf == 2 and len(node) == 2:
            return 2 + 3
        if leaf == 3 and len(node) == 6:
            return 2 + 3 + 3 + 2 + 1 + _scope_length(node[5])
        if leaf == 4 and len(node) == 3:
            return 2 + 3 + _elias_delta_length(int(node[2]) + 1)
        if leaf == 5 and len(node) == 3:
            return 2 + 3 + _elias_delta_length(int(node[2]) + 1)
        _fail("FAIL_Q1_MDL_LENGTH", "unknown old-DSL leaf")
    if tag == 1 and len(node) == 3:
        return 2 + 2 + _node_mdl_bits(node[2])
    if tag == 2 and len(node) == 4:
        return 2 + 3 + _node_mdl_bits(node[2]) + _node_mdl_bits(node[3])
    if tag == 3 and len(node) == 5:
        return 3 + 1 + _node_mdl_bits(node[2]) + _node_mdl_bits(node[3]) + 2
    if tag == 4 and len(node) == 2 and isinstance(node[1], tuple):
        shape = {1: 4, 2: 5, 3: 6}.get(len(node[1]))
        if shape is None:
            _fail("FAIL_Q1_MDL_LENGTH", "invalid conjunction arity")
        return shape + sum(_node_mdl_bits(child) for child in node[1])
    _fail("FAIL_Q1_MDL_LENGTH", "unknown canonical node")


def program_mdl_length_q32_v1(canonical_ast: CanonicalAst) -> int:
    """Return the inherited exact fixed-prefix old-program Q32 length."""

    if not isinstance(canonical_ast, CanonicalAst) or canonical_ast.value[0] != 1:
        raise TypeError("canonical_ast must be CanonicalAstV1")
    return _node_mdl_bits(canonical_ast.value[1]) * Q32_SCALE


def future_signature_from_ast_v1(
    canonical_ast: CanonicalAst,
) -> FutureAdmissibilitySignatureV1:
    """Recompute the complete continuation signature from one admitted AST."""

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
        _fail("REJECT_Q1_CONSTRUCTION_SIGNATURE", "unknown output sort")

    node = canonical_ast.value[1]
    if not isinstance(node, tuple) or not node or type(node[0]) is not int:
        _fail("REJECT_Q1_CONSTRUCTION_SIGNATURE", "canonical AST is malformed")
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
        mdl_length_q32=program_mdl_length_q32_v1(canonical_ast),
    )


def normalization_witness_capacity_v1(output_sort_id: OutputSortId) -> int:
    """Return the distinct-child witness capacity required by v1.6."""

    if not isinstance(output_sort_id, OutputSortId):
        raise TypeError("output_sort_id must be OutputSortId")
    return 2 if output_sort_id in {
        OutputSortId.BOOL,
        OutputSortId.RATIONAL_VALUE,
    } else 1


__all__ = [
    "DIAGNOSTIC_SIGNATURE_SCHEMA_ID",
    "FutureAdmissibilitySignatureV1",
    "NormalizationProfileId",
    "OutputSortId",
    "Q1QuotientContractError",
    "future_signature_from_ast_v1",
    "normalization_witness_capacity_v1",
    "program_mdl_length_q32_v1",
]
