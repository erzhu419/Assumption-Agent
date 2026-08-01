"""Machine-readable freeze for ``hegel-old-dsl-v1.0.0``.

This module freezes the finite old-language surface, exact value grids,
typing, bottom rules, search bounds, and the two Phase-3A benchmark
contracts.  It is a *specification artifact*, not a closure result.  In
particular, importing this module does not enumerate the language and cannot
support an ``OUTSIDE_FROZEN_CLOSURE`` certificate.

All collections that contribute to a content id are immutable and use exact
integers or numerator/denominator pairs.  Runtime helpers use
``fractions.Fraction`` and never binary floating point.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from itertools import product
from types import MappingProxyType
from typing import Final, Mapping, Sequence

from .hashing import stable_hash


DSL_VERSION: Final = "hegel-old-dsl-v1.0.0"
DSL_FREEZE_SCHEMA_VERSION: Final = "hegel-old-dsl-freeze/1.0.0"
TARGET_ID: Final = "TARGET_P3A_GENERIC_ODD_REDUCTION_V1"
TARGET_NAME: Final = (
    "Generic Odd-Cardinality Reduction over Bounded Entity Sets"
)
SINK_CONTROL_ID: Final = "CONTROL_P3A_OBSERVED_OMITTED_SINK_V1"
SINK_CONTROL_NAME: Final = (
    "Observed Omitted-Channel Conservation Refinement Control"
)


@dataclass(frozen=True, order=True, slots=True)
class RationalAtom:
    """Canonical, reduced rational value suitable for content addressing."""

    numerator: int
    denominator: int = 1

    def __post_init__(self) -> None:
        if type(self.numerator) is not int or type(self.denominator) is not int:
            raise TypeError("rational numerator and denominator must be integers")
        if self.denominator <= 0:
            raise ValueError("rational denominator must be positive")
        reduced = Fraction(self.numerator, self.denominator)
        if (reduced.numerator, reduced.denominator) != (
            self.numerator,
            self.denominator,
        ):
            raise ValueError("RationalAtom must already be reduced")

    @classmethod
    def from_fraction(cls, value: Fraction) -> "RationalAtom":
        if type(value) is not Fraction:
            raise TypeError("value must be an exact Fraction")
        return cls(value.numerator, value.denominator)

    def as_fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)


def _rational_atom(value: int | Fraction) -> RationalAtom:
    exact = value if type(value) is Fraction else Fraction(value, 1)
    return RationalAtom.from_fraction(exact)


def _rational_grid() -> tuple[RationalAtom, ...]:
    values = {
        Fraction(numerator, denominator)
        for numerator in range(-64, 65)
        for denominator in range(1, 9)
    }
    return tuple(RationalAtom.from_fraction(value) for value in sorted(values))


RATIONAL_VALUE_GRID: Final = _rational_grid()
RATIONAL_VALUE_FRACTIONS: Final = frozenset(
    atom.as_fraction() for atom in RATIONAL_VALUE_GRID
)
RATIONAL_PARAMETER_GRID: Final = tuple(
    _rational_atom(value)
    for value in (
        -2,
        -1,
        Fraction(-1, 2),
        0,
        Fraction(1, 2),
        1,
        2,
    )
)
TOLERANCE_GRID: Final = tuple(
    _rational_atom(value) for value in (0, Fraction(1, 4), Fraction(1, 2))
)
INTERVAL_ENDPOINT_GRID: Final = (-8, -4, -2, -1, 0, 1, 2, 4, 8)


@dataclass(frozen=True, order=True, slots=True)
class ClosedIntervalAtom:
    lower: int
    upper: int

    def __post_init__(self) -> None:
        if self.lower not in INTERVAL_ENDPOINT_GRID:
            raise ValueError("lower endpoint is outside the frozen grid")
        if self.upper not in INTERVAL_ENDPOINT_GRID:
            raise ValueError("upper endpoint is outside the frozen grid")
        if self.lower > self.upper:
            raise ValueError("closed interval requires lower <= upper")


CLOSED_INTERVAL_GRID: Final = tuple(
    ClosedIntervalAtom(lower, upper)
    for lower in INTERVAL_ENDPOINT_GRID
    for upper in INTERVAL_ENDPOINT_GRID
    if lower <= upper
)


ENTITY_SLOTS: Final = tuple(f"e{index}" for index in range(8))
INDICES: Final = tuple(range(8))
QUANTITY_IDS: Final = ("q0", "q1")
CONTEXT_IDS: Final = ("c0", "c1", "c2", "c3")
ROLE_IDS: Final = ("r0", "r1", "r2", "r3")
SCALE_IDS: Final = ("s0", "s1")
TASK_IDS: Final = ("t0", "t1")


@dataclass(frozen=True, slots=True)
class ScopeSpec:
    scope_id: str
    semantic_rule: str
    include_auxiliary: bool
    context_clause_limit: int = 2
    context_registry: tuple[str, ...] = CONTEXT_IDS

    def __post_init__(self) -> None:
        if self.context_clause_limit != 2:
            raise ValueError("old DSL scopes allow at most two context clauses")
        if self.context_registry != CONTEXT_IDS:
            raise ValueError("scope context registry is frozen")


SCOPE_CATALOG: Final = (
    ScopeSpec(
        "scope_all_observed_v1",
        "include every observed entity",
        True,
    ),
    ScopeSpec(
        "scope_primary_only_v1",
        "include observed entities with auxiliary == false",
        False,
    ),
    ScopeSpec(
        "scope_boundary_only_v1",
        "include observed entities with boundary_member == true",
        True,
    ),
    ScopeSpec(
        "control_volume_all_observed_v1",
        "same control_volume_id and quantity_id; include auxiliary entities",
        True,
    ),
)
SCOPE_IDS: Final = tuple(spec.scope_id for spec in SCOPE_CATALOG)


@dataclass(frozen=True, slots=True)
class AggregateMapSpec:
    map_id: str
    input_sorts: tuple[str, ...]
    output_sort: str
    semantic_rule: str
    undefined_conditions: tuple[str, ...]


_MEASUREMENT_BOTTOM: Final = (
    "missing_measurement",
    "quantity_mismatch",
)
AGGREGATE_CATALOG: Final = (
    AggregateMapSpec(
        "sum_v1",
        ("EntitySet", "QuantityId"),
        "RationalValue",
        "exact sum(values)",
        _MEASUREMENT_BOTTOM,
    ),
    AggregateMapSpec(
        "count_nonzero_v1",
        ("EntitySet", "QuantityId"),
        "BoundedInt",
        "count(value != 0)",
        _MEASUREMENT_BOTTOM,
    ),
    AggregateMapSpec(
        "mean_v1",
        ("EntitySet", "QuantityId"),
        "RationalValue",
        "exact sum(values) / len(values)",
        _MEASUREMENT_BOTTOM + ("empty_entity_set",),
    ),
    AggregateMapSpec(
        "min_v1",
        ("EntitySet", "QuantityId"),
        "RationalValue",
        "minimum(values)",
        _MEASUREMENT_BOTTOM + ("empty_entity_set",),
    ),
    AggregateMapSpec(
        "max_v1",
        ("EntitySet", "QuantityId"),
        "RationalValue",
        "maximum(values)",
        _MEASUREMENT_BOTTOM + ("empty_entity_set",),
    ),
    AggregateMapSpec(
        "signed_balance_v1",
        ("EntitySet", "QuantityId"),
        "RationalValue",
        "exact sum(orientation(entity) * value(entity))",
        ("missing_orientation",) + _MEASUREMENT_BOTTOM,
    ),
)
AGGREGATE_MAP_IDS: Final = tuple(spec.map_id for spec in AGGREGATE_CATALOG)


@dataclass(frozen=True, slots=True)
class TransformSpec:
    transform_id: str
    semantic_rule: str
    adapter_only: bool = True
    old_dsl_composable: bool = False


TRANSFORM_CATALOG: Final = (
    TransformSpec("identity_v1", "x -> x"),
    TransformSpec("negate_v1", "x -> -x"),
    TransformSpec("scale_by_2_v1", "x -> 2*x"),
    TransformSpec("scale_by_half_v1", "x -> x/2"),
)
TRANSFORM_IDS: Final = tuple(spec.transform_id for spec in TRANSFORM_CATALOG)


@dataclass(frozen=True, slots=True)
class PrimitiveDomainSpec:
    sort_id: str
    values: tuple[object, ...]
    cardinality: int
    observable_values_only: bool = True

    def __post_init__(self) -> None:
        if type(self.cardinality) is not int or self.cardinality <= 0:
            raise ValueError("domain cardinality must be a positive integer")
        if len(self.values) != self.cardinality:
            raise ValueError(
                f"{self.sort_id} declares {self.cardinality} values but has "
                f"{len(self.values)}"
            )
        serialized = tuple(stable_hash(value) for value in self.values)
        if len(set(serialized)) != len(serialized):
            raise ValueError(f"{self.sort_id} domain values must be unique")


PRIMITIVE_DOMAINS: Final = (
    PrimitiveDomainSpec("Bool", (False, True), 2),
    PrimitiveDomainSpec("Bit", (0, 1), 2),
    PrimitiveDomainSpec("Sign", (-1, 0, 1), 3),
    PrimitiveDomainSpec("BoundedInt", tuple(range(-8, 9)), 17),
    PrimitiveDomainSpec("RationalValue", RATIONAL_VALUE_GRID, 663),
    PrimitiveDomainSpec("RationalParameter", RATIONAL_PARAMETER_GRID, 7),
    PrimitiveDomainSpec("Tolerance", TOLERANCE_GRID, 3),
    PrimitiveDomainSpec("IntervalEndpoint", INTERVAL_ENDPOINT_GRID, 9),
    PrimitiveDomainSpec("ClosedInterval", CLOSED_INTERVAL_GRID, 45),
    PrimitiveDomainSpec("EntitySlot", ENTITY_SLOTS, 8),
    PrimitiveDomainSpec("Index", INDICES, 8),
    PrimitiveDomainSpec("QuantityId", QUANTITY_IDS, 2),
    PrimitiveDomainSpec("ContextId", CONTEXT_IDS, 4),
    PrimitiveDomainSpec("RoleId", ROLE_IDS, 4),
    PrimitiveDomainSpec("ScaleId", SCALE_IDS, 2),
    PrimitiveDomainSpec("TaskId", TASK_IDS, 2),
    PrimitiveDomainSpec("ScopeId", SCOPE_IDS, 4),
    PrimitiveDomainSpec("AggregateMapId", AGGREGATE_MAP_IDS, 6),
    PrimitiveDomainSpec("TransformId", TRANSFORM_IDS, 4),
)
PRIMITIVE_SORT_IDS: Final = tuple(spec.sort_id for spec in PRIMITIVE_DOMAINS)
if len(set(PRIMITIVE_SORT_IDS)) != len(PRIMITIVE_SORT_IDS):
    raise RuntimeError("frozen primitive sort ids must be unique")
PRIMITIVE_DOMAIN_BY_ID: Final = MappingProxyType(
    {spec.sort_id: spec for spec in PRIMITIVE_DOMAINS}
)


@dataclass(frozen=True, slots=True)
class IdentifierRegistries:
    entity_slots: tuple[str, ...] = ENTITY_SLOTS
    quantity_ids: tuple[str, ...] = QUANTITY_IDS
    context_ids: tuple[str, ...] = CONTEXT_IDS
    role_ids: tuple[str, ...] = ROLE_IDS
    scale_ids: tuple[str, ...] = SCALE_IDS
    task_ids: tuple[str, ...] = TASK_IDS

    def __post_init__(self) -> None:
        expected = (
            ENTITY_SLOTS,
            QUANTITY_IDS,
            CONTEXT_IDS,
            ROLE_IDS,
            SCALE_IDS,
            TASK_IDS,
        )
        actual = (
            self.entity_slots,
            self.quantity_ids,
            self.context_ids,
            self.role_ids,
            self.scale_ids,
            self.task_ids,
        )
        if actual != expected:
            raise ValueError("private identifier registries are frozen")

    @property
    def content_id(self) -> str:
        return stable_hash(self, prefix="identifier_registry_")


IDENTIFIER_REGISTRIES: Final = IdentifierRegistries()


class BottomValue(str, Enum):
    BOTTOM = "⊥"


BOTTOM: Final = BottomValue.BOTTOM


@dataclass(frozen=True, slots=True)
class BottomAndEquivalenceSemantics:
    bottom_symbol: str = "⊥"
    bottom_is_observable: bool = False
    strict_required_child_propagation: bool = True
    bit_at_out_of_range_is_bottom: bool = True
    empty_mean_min_max_is_bottom: bool = True
    missing_measurement_orientation_or_quantity_mismatch_is_bottom: bool = True
    out_of_rational_grid_result_is_bottom: bool = True
    sign_zero: int = 0
    allow_nan: bool = False
    allow_infinity: bool = False
    allow_implicit_float_rounding: bool = False
    boolean_bit_equivalence: str = "exact_equality_tolerance_zero"
    rational_equivalence: str = "exact_fraction_equality"
    bottom_disqualifies_exact_match: bool = True


BOTTOM_AND_EQUIVALENCE: Final = BottomAndEquivalenceSemantics()


@dataclass(frozen=True, slots=True)
class TypedExpressionSpec:
    expression_id: str
    expression_class: str
    input_sorts: tuple[str, ...]
    output_sorts: tuple[str, ...]
    accepted_arities: tuple[int, ...]
    semantic_rule: str
    canonical_child_sort_groups: tuple[tuple[int, ...], ...] = ()


LEAF_EXPRESSIONS: Final = (
    TypedExpressionSpec(
        "scalar_const",
        "leaf",
        ("RationalParameter",),
        ("RationalValue",),
        (1,),
        "return the selected exact RationalParameter",
    ),
    TypedExpressionSpec(
        "bit_at",
        "leaf",
        ("Index",),
        ("Bit",),
        (1,),
        "return S[index], or bottom when index >= set_size",
    ),
    TypedExpressionSpec(
        "set_size",
        "leaf",
        (),
        ("BoundedInt",),
        (0,),
        "return the observed EntitySet cardinality",
    ),
    TypedExpressionSpec(
        "aggregate",
        "leaf",
        ("AggregateMapId", "ScopeId", "QuantityId"),
        ("RationalValue", "BoundedInt"),
        (3,),
        "evaluate the selected frozen aggregate over the selected frozen scope",
    ),
    TypedExpressionSpec(
        "context_flag",
        "leaf",
        ("ContextId",),
        ("Bool",),
        (1,),
        "return the observed boolean ContextId flag",
    ),
    TypedExpressionSpec(
        "task_flag",
        "leaf",
        ("TaskId",),
        ("Bool",),
        (1,),
        "return the observed boolean TaskId flag",
    ),
)

UNARY_OPERATORS: Final = (
    TypedExpressionSpec(
        "bit_to_scalar",
        "unary_operator",
        ("Bit",),
        ("RationalValue",),
        (1,),
        "exact Fraction(bit, 1)",
    ),
    TypedExpressionSpec(
        "int_to_scalar",
        "unary_operator",
        ("BoundedInt",),
        ("RationalValue",),
        (1,),
        "exact Fraction(integer, 1)",
    ),
    TypedExpressionSpec(
        "absolute",
        "unary_operator",
        ("RationalValue",),
        ("RationalValue",),
        (1,),
        "exact absolute value",
    ),
    TypedExpressionSpec(
        "sign",
        "unary_operator",
        ("RationalValue",),
        ("Sign",),
        (1,),
        "-1 if x < 0, 0 if x == 0, +1 if x > 0",
    ),
)

BINARY_OPERATORS: Final = (
    TypedExpressionSpec(
        "add",
        "binary_operator",
        ("RationalValue", "RationalValue"),
        ("RationalValue",),
        (2,),
        "exact left + right",
        ((0, 1),),
    ),
    TypedExpressionSpec(
        "difference",
        "binary_operator",
        ("RationalValue", "RationalValue"),
        ("RationalValue",),
        (2,),
        "exact left - right",
    ),
    TypedExpressionSpec(
        "equal_exact",
        "binary_operator",
        ("RationalValue", "RationalValue"),
        ("Bool",),
        (2,),
        "exact Fraction equality",
        ((0, 1),),
    ),
    TypedExpressionSpec(
        "less_equal",
        "binary_operator",
        ("RationalValue", "RationalValue"),
        ("Bool",),
        (2,),
        "exact Fraction left <= right",
    ),
    TypedExpressionSpec(
        "greater_equal",
        "binary_operator",
        ("RationalValue", "RationalValue"),
        ("Bool",),
        (2,),
        "exact Fraction left >= right",
    ),
    TypedExpressionSpec(
        "same_sign",
        "binary_operator",
        ("Sign", "Sign"),
        ("Bool",),
        (2,),
        "left == right",
        ((0, 1),),
    ),
    TypedExpressionSpec(
        "opposite_sign",
        "binary_operator",
        ("Sign", "Sign"),
        ("Bool",),
        (2,),
        "left == -right and left != 0",
        ((0, 1),),
    ),
)

TERNARY_OPERATORS: Final = (
    TypedExpressionSpec(
        "approx_equal",
        "ternary_operator",
        ("RationalValue", "RationalValue", "Tolerance"),
        ("Bool",),
        (3,),
        "exact abs(left - right) <= tolerance",
        ((0, 1),),
    ),
)

BOOLEAN_COMPOSITION: Final = (
    TypedExpressionSpec(
        "top_level_AND",
        "top_level_boolean_composition",
        ("Bool",),
        ("Bool",),
        (1, 2, 3),
        "exact conjunction of one, two, or three atomic Bool clauses",
        ((0, 1, 2),),
    ),
)

ALL_EXPRESSIONS: Final = (
    LEAF_EXPRESSIONS
    + UNARY_OPERATORS
    + BINARY_OPERATORS
    + TERNARY_OPERATORS
    + BOOLEAN_COMPOSITION
)

FORBIDDEN_FORMS: Final = (
    "OR",
    "XOR",
    "NOT(compound)",
    "modulo",
    "parity",
    "arbitrary_lookup_table",
    "recursive_fold",
    "user_defined_reducer",
    "case_ID_branch",
)


def _require_fraction(value: object, name: str) -> Fraction:
    if type(value) is not Fraction:
        raise TypeError(f"{name} must be fractions.Fraction")
    if value not in RATIONAL_VALUE_FRACTIONS:
        raise ValueError(f"{name} is outside RationalValue")
    return value


def _bounded_rational(value: Fraction) -> Fraction | BottomValue:
    if value not in RATIONAL_VALUE_FRACTIONS:
        return BOTTOM
    return value


def evaluate_operator(
    operator_id: str,
    children: Sequence[object],
) -> object:
    """Evaluate a frozen scalar operator with strict bottom propagation.

    This small reference helper pins the scalar truth conditions.  It is not
    the independent closure evaluator required by a formal certificate.
    """

    if any(child is BOTTOM for child in children):
        return BOTTOM

    arities = {
        spec.expression_id: spec.accepted_arities
        for spec in UNARY_OPERATORS
        + BINARY_OPERATORS
        + TERNARY_OPERATORS
        + BOOLEAN_COMPOSITION
    }
    if operator_id not in arities:
        raise ValueError(f"unknown frozen operator: {operator_id}")
    if len(children) not in arities[operator_id]:
        raise ValueError(f"invalid arity for {operator_id}")

    if operator_id == "bit_to_scalar":
        bit = children[0]
        if type(bit) is not int or bit not in (0, 1):
            raise TypeError("bit_to_scalar requires a Bit")
        return Fraction(bit, 1)
    if operator_id == "int_to_scalar":
        integer = children[0]
        if type(integer) is not int or not -8 <= integer <= 8:
            raise TypeError("int_to_scalar requires a BoundedInt")
        return Fraction(integer, 1)
    if operator_id in {"absolute", "sign"}:
        value = _require_fraction(children[0], operator_id)
        if operator_id == "absolute":
            return _bounded_rational(abs(value))
        return -1 if value < 0 else 1 if value > 0 else 0
    if operator_id in {
        "add",
        "difference",
        "equal_exact",
        "less_equal",
        "greater_equal",
    }:
        left = _require_fraction(children[0], operator_id)
        right = _require_fraction(children[1], operator_id)
        if operator_id == "add":
            return _bounded_rational(left + right)
        if operator_id == "difference":
            return _bounded_rational(left - right)
        if operator_id == "equal_exact":
            return left == right
        if operator_id == "less_equal":
            return left <= right
        return left >= right
    if operator_id in {"same_sign", "opposite_sign"}:
        left, right = children
        if (
            type(left) is not int
            or type(right) is not int
            or left not in (-1, 0, 1)
            or right not in (-1, 0, 1)
        ):
            raise TypeError(f"{operator_id} requires two Sign values")
        if operator_id == "same_sign":
            return left == right
        return left == -right and left != 0
    if operator_id == "approx_equal":
        left = _require_fraction(children[0], operator_id)
        right = _require_fraction(children[1], operator_id)
        tolerance = children[2]
        if type(tolerance) is not Fraction or RationalAtom.from_fraction(
            tolerance
        ) not in TOLERANCE_GRID:
            raise TypeError("approx_equal requires a frozen Tolerance")
        return abs(left - right) <= tolerance
    if operator_id == "top_level_AND":
        if any(type(child) is not bool for child in children):
            raise TypeError("top_level_AND requires Bool children")
        return all(children)
    raise AssertionError("unreachable frozen operator")


def evaluate_aggregate(
    map_id: str,
    values: Sequence[Fraction | BottomValue],
    *,
    orientations: Sequence[int | BottomValue] | None = None,
) -> object:
    """Evaluate one frozen aggregate on already scoped exact values."""

    if map_id not in AGGREGATE_MAP_IDS:
        raise ValueError(f"unknown frozen aggregate map: {map_id}")
    if any(value is BOTTOM for value in values):
        return BOTTOM
    exact_values = tuple(_require_fraction(value, map_id) for value in values)
    if map_id == "count_nonzero_v1":
        count = sum(value != 0 for value in exact_values)
        return count if -8 <= count <= 8 else BOTTOM
    if map_id in {"mean_v1", "min_v1", "max_v1"} and not exact_values:
        return BOTTOM
    if map_id == "sum_v1":
        return _bounded_rational(sum(exact_values, Fraction(0, 1)))
    if map_id == "mean_v1":
        return _bounded_rational(
            sum(exact_values, Fraction(0, 1)) / len(exact_values)
        )
    if map_id == "min_v1":
        return min(exact_values)
    if map_id == "max_v1":
        return max(exact_values)
    if orientations is None or len(orientations) != len(exact_values):
        return BOTTOM
    if any(orientation is BOTTOM for orientation in orientations):
        return BOTTOM
    if any(
        type(orientation) is not int or orientation not in (-1, 1)
        for orientation in orientations
    ):
        raise TypeError("signed_balance_v1 orientations must be -1 or +1")
    return _bounded_rational(
        sum(
            (orientation * value for orientation, value in zip(orientations, exact_values)),
            Fraction(0, 1),
        )
    )


@dataclass(frozen=True, slots=True)
class StructuralLimits:
    max_total_ast_depth: int = 4
    max_total_node_count: int = 7
    max_top_level_clauses: int = 3
    max_distinct_bit_slots: int = 4
    max_aggregate_leaves: int = 1
    max_scope_clauses: int = 2
    max_old_law_composition_depth: int = 2
    max_fitted_scalar_parameters: int = 3
    leaf_depth: int = 0
    operator_depth_rule: str = "1 + max(child_depth)"


STRUCTURAL_LIMITS: Final = StructuralLimits()


@dataclass(frozen=True, slots=True)
class ShrinkStep:
    order: int
    operation: str


SHRINK_ORDER: Final = (
    ShrinkStep(1, "remove mean_v1, min_v1, max_v1"),
    ShrinkStep(2, "reduce RationalParameter to {-1,0,1}"),
    ShrinkStep(3, "remove add; retain difference"),
    ShrinkStep(4, "reduce max_top_level_clauses from 3 to 2"),
    ShrinkStep(5, "reduce max_total_node_count from 7 to 6"),
    ShrinkStep(6, "reduce max_total_ast_depth from 4 to 3"),
)


class ClosureStatus(str, Enum):
    NOT_RUN = "NOT_RUN"
    COMPLETE = "COMPLETE"
    DSL_TOO_LARGE = "DSL_TOO_LARGE"
    INCONCLUSIVE_BUDGET = "INCONCLUSIVE_BUDGET"


@dataclass(frozen=True, slots=True)
class ClosureBudgetAndTraversal:
    canonical_counted_object: str = (
        "syntactically canonical programs before extensional quotient"
    )
    max_canonical_program_count: int = 50_000
    max_raw_operator_applications: int = 5_000_000
    raw_expansion_definition: str = (
        "attempt one operator token with one type-legal child tuple, whether "
        "canonicalization accepts, rejects, or deduplicates the parent AST"
    )
    traversal_sort_keys: tuple[str, ...] = (
        "total_ast_depth ascending",
        "total_node_count ascending",
        "output_sort_id ascending",
        "root_operator_id ascending",
        "canonical_ast_cbor bytes lexicographically ascending",
    )
    dynamic_programming_bucket: tuple[str, ...] = (
        "output_sort",
        "depth",
        "node_count",
    )
    commutative_child_order: str = "child canonical hash ascending"
    completion_requirements: tuple[str, ...] = (
        "enumeration_frontier_exhausted == true",
        "all_type_buckets_closed == true",
        "canonical_program_count <= 50000",
        "raw_expansion_limit_not_hit == true",
        "wall_clock_abort_not_hit == true",
    )
    replay_bound_roots: tuple[str, ...] = (
        "dsl_spec_root",
        "operator_semantics_root",
        "identifier_registry_root",
        "bounded_universe_root",
        "canonicalizer_source_root",
        "enumerator_source_root",
        "canonical_program_archive_root",
        "program_output_archive_root",
        "target_truth_table_root",
        "chunk_manifest_root",
        "enumeration_exhaustion_receipt_root",
        "container_image_digest",
        "repository_commit_sha",
    )


CLOSURE_BUDGET: Final = ClosureBudgetAndTraversal()


@dataclass(frozen=True, slots=True)
class DslExecutionState:
    surface_parameter_tables_frozen: bool = True
    strict_canonical_ast_schema_frozen: bool = True
    scalar_reference_semantics_present: bool = True
    canonicalizer_implemented: bool = False
    python_complete_enumerator_implemented: bool = False
    rust_complete_enumerator_implemented: bool = False
    closure_status: ClosureStatus = ClosureStatus.NOT_RUN
    outside_frozen_closure_certificate_issued: bool = False

    def __post_init__(self) -> None:
        if (
            not self.surface_parameter_tables_frozen
            or not self.strict_canonical_ast_schema_frozen
            or not self.scalar_reference_semantics_present
        ):
            raise ValueError(
                "the v1 surface tables and v1.0.2 strict AST schema are frozen"
            )
        if (
            self.canonicalizer_implemented
            or self.python_complete_enumerator_implemented
            or self.rust_complete_enumerator_implemented
        ):
            raise ValueError(
                "this freeze module cannot claim executable closure components"
            )
        if self.closure_status is not ClosureStatus.NOT_RUN:
            raise ValueError("this freeze module does not carry an executed closure")
        if self.outside_frozen_closure_certificate_issued:
            raise ValueError("a DSL freeze cannot issue an outside certificate")


DSL_EXECUTION_STATE: Final = DslExecutionState()


@dataclass(frozen=True, slots=True)
class OldDslFreeze:
    schema_version: str = DSL_FREEZE_SCHEMA_VERSION
    dsl_version: str = DSL_VERSION
    primitive_domains: tuple[PrimitiveDomainSpec, ...] = PRIMITIVE_DOMAINS
    identifier_registries: IdentifierRegistries = IDENTIFIER_REGISTRIES
    scope_catalog: tuple[ScopeSpec, ...] = SCOPE_CATALOG
    aggregate_catalog: tuple[AggregateMapSpec, ...] = AGGREGATE_CATALOG
    transform_catalog: tuple[TransformSpec, ...] = TRANSFORM_CATALOG
    leaves: tuple[TypedExpressionSpec, ...] = LEAF_EXPRESSIONS
    unary_operators: tuple[TypedExpressionSpec, ...] = UNARY_OPERATORS
    binary_operators: tuple[TypedExpressionSpec, ...] = BINARY_OPERATORS
    ternary_operators: tuple[TypedExpressionSpec, ...] = TERNARY_OPERATORS
    boolean_composition: tuple[TypedExpressionSpec, ...] = BOOLEAN_COMPOSITION
    forbidden_forms: tuple[str, ...] = FORBIDDEN_FORMS
    bottom_and_equivalence: BottomAndEquivalenceSemantics = BOTTOM_AND_EQUIVALENCE
    structural_limits: StructuralLimits = STRUCTURAL_LIMITS
    shrink_order: tuple[ShrinkStep, ...] = SHRINK_ORDER
    closure_budget: ClosureBudgetAndTraversal = CLOSURE_BUDGET
    execution_state: DslExecutionState = DSL_EXECUTION_STATE

    def __post_init__(self) -> None:
        if self.schema_version != DSL_FREEZE_SCHEMA_VERSION:
            raise ValueError("unknown old-DSL freeze schema")
        if self.dsl_version != DSL_VERSION:
            raise ValueError("old-DSL version is frozen")
        frozen_fields = (
            ("primitive_domains", self.primitive_domains, PRIMITIVE_DOMAINS),
            (
                "identifier_registries",
                self.identifier_registries,
                IDENTIFIER_REGISTRIES,
            ),
            ("scope_catalog", self.scope_catalog, SCOPE_CATALOG),
            ("aggregate_catalog", self.aggregate_catalog, AGGREGATE_CATALOG),
            ("transform_catalog", self.transform_catalog, TRANSFORM_CATALOG),
            ("leaves", self.leaves, LEAF_EXPRESSIONS),
            ("unary_operators", self.unary_operators, UNARY_OPERATORS),
            ("binary_operators", self.binary_operators, BINARY_OPERATORS),
            ("ternary_operators", self.ternary_operators, TERNARY_OPERATORS),
            ("boolean_composition", self.boolean_composition, BOOLEAN_COMPOSITION),
            ("forbidden_forms", self.forbidden_forms, FORBIDDEN_FORMS),
            (
                "bottom_and_equivalence",
                self.bottom_and_equivalence,
                BOTTOM_AND_EQUIVALENCE,
            ),
            ("structural_limits", self.structural_limits, STRUCTURAL_LIMITS),
            ("shrink_order", self.shrink_order, SHRINK_ORDER),
            ("closure_budget", self.closure_budget, CLOSURE_BUDGET),
            ("execution_state", self.execution_state, DSL_EXECUTION_STATE),
        )
        for field_name, actual, expected in frozen_fields:
            if actual != expected:
                raise ValueError(f"{field_name} differs from frozen old DSL v1")
        expression_ids = tuple(
            spec.expression_id
            for spec in (
                self.leaves
                + self.unary_operators
                + self.binary_operators
                + self.ternary_operators
                + self.boolean_composition
            )
        )
        if len(set(expression_ids)) != len(expression_ids):
            raise ValueError("old-DSL expression ids must be unique")

    @property
    def content_id(self) -> str:
        return stable_hash(self, prefix="dsl_spec_")

    @property
    def operator_semantics_id(self) -> str:
        return stable_hash(
            (
                self.primitive_domains,
                self.identifier_registries,
                self.scope_catalog,
                self.aggregate_catalog,
                self.transform_catalog,
                self.leaves,
                self.unary_operators,
                self.binary_operators,
                self.ternary_operators,
                self.boolean_composition,
                self.bottom_and_equivalence,
            ),
            prefix="operator_semantics_",
        )

    @property
    def rational_grid_id(self) -> str:
        return stable_hash(RATIONAL_VALUE_GRID, prefix="rational_grid_")


OLD_DSL_V1: Final = OldDslFreeze()


@dataclass(frozen=True, slots=True)
class OddReductionUniverseRow:
    universe_index: int
    set_size: int
    bits: tuple[int, ...]
    target_output: int


def odd_reduction(bits: Sequence[int]) -> int:
    if not 5 <= len(bits) <= 8:
        raise ValueError("Generic Odd Reduction requires set size 5..8")
    if any(type(bit) is not int or bit not in (0, 1) for bit in bits):
        raise TypeError("Generic Odd Reduction requires Bit values")
    return sum(bits) % 2


def _odd_universe() -> tuple[OddReductionUniverseRow, ...]:
    rows: list[OddReductionUniverseRow] = []
    for set_size in range(5, 9):
        for bits in product((0, 1), repeat=set_size):
            rows.append(
                OddReductionUniverseRow(
                    universe_index=len(rows),
                    set_size=set_size,
                    bits=bits,
                    target_output=odd_reduction(bits),
                )
            )
    return tuple(rows)


ODD_REDUCTION_UNIVERSE: Final = _odd_universe()


@dataclass(frozen=True, slots=True)
class SplitQuota:
    set_size: int
    discovery_train: int
    validation: int
    sealed_prediction: int
    discovery_per_label: int
    validation_per_label: int
    sealed_per_label: int

    def __post_init__(self) -> None:
        if self.discovery_train != 2 * self.discovery_per_label:
            raise ValueError("discovery split must be label-balanced")
        if self.validation != 2 * self.validation_per_label:
            raise ValueError("validation split must be label-balanced")
        if self.sealed_prediction != 2 * self.sealed_per_label:
            raise ValueError("sealed split must be label-balanced")
        if self.discovery_train + self.validation + self.sealed_prediction != 2 ** self.set_size:
            raise ValueError("split quotas must exhaust the per-size universe")


ODD_REDUCTION_SPLITS: Final = (
    SplitQuota(5, 12, 6, 14, 6, 3, 7),
    SplitQuota(6, 26, 12, 26, 13, 6, 13),
    SplitQuota(7, 52, 26, 50, 26, 13, 25),
    SplitQuota(8, 102, 52, 102, 51, 26, 51),
)


class TargetRegistryPredicate(str, Enum):
    COUNT_MOD_2_EQ_1 = "count mod 2 == 1"
    COUNT_MOD_3_EQ_1 = "count mod 3 == 1"
    COUNT_IN_PRIME_SET = "count in {2,3,5,7}"


@dataclass(frozen=True, slots=True)
class HiddenTargetRegistryEntry:
    priority: int
    target_id: str
    predicate: TargetRegistryPredicate
    positive_count: int
    universe_count: int = 480

    @property
    def prevalence(self) -> Fraction:
        return Fraction(self.positive_count, self.universe_count)


def _registry_target_output(predicate: TargetRegistryPredicate, count: int) -> int:
    if predicate is TargetRegistryPredicate.COUNT_MOD_2_EQ_1:
        return int(count % 2 == 1)
    if predicate is TargetRegistryPredicate.COUNT_MOD_3_EQ_1:
        return int(count % 3 == 1)
    if predicate is TargetRegistryPredicate.COUNT_IN_PRIME_SET:
        return int(count in {2, 3, 5, 7})
    raise AssertionError("unknown target predicate")


def _positive_count(predicate: TargetRegistryPredicate) -> int:
    return sum(
        _registry_target_output(predicate, sum(row.bits))
        for row in ODD_REDUCTION_UNIVERSE
    )


HIDDEN_TARGET_REGISTRY: Final = tuple(
    HiddenTargetRegistryEntry(priority, target_id, predicate, _positive_count(predicate))
    for priority, target_id, predicate in (
        (1, TARGET_ID, TargetRegistryPredicate.COUNT_MOD_2_EQ_1),
        (
            2,
            "TARGET_P3A_GENERIC_COUNT_MOD_3_EQ_1_V1",
            TargetRegistryPredicate.COUNT_MOD_3_EQ_1,
        ),
        (
            3,
            "TARGET_P3A_GENERIC_PRIME_COUNT_V1",
            TargetRegistryPredicate.COUNT_IN_PRIME_SET,
        ),
    )
)


def select_first_outside_target(
    exact_match_count_by_target: Mapping[str, int],
) -> HiddenTargetRegistryEntry | None:
    """Apply the preregistered fallback rule without making a closure claim."""

    for entry in HIDDEN_TARGET_REGISTRY:
        match_count = exact_match_count_by_target.get(entry.target_id)
        if type(match_count) is not int or match_count < 0:
            raise ValueError("every registry target needs a nonnegative match count")
        if (
            match_count == 0
            and Fraction(1, 4) <= entry.prevalence <= Fraction(3, 4)
        ):
            return entry
    return None


class TargetPreflightStatus(str, Enum):
    AWAITING_COMPLETE_CLOSURE = "AWAITING_COMPLETE_CLOSURE"
    IN_LANGUAGE_POSITIVE_CONTROL = "IN_LANGUAGE_POSITIVE_CONTROL"
    ELIGIBLE_FOR_OUTSIDE_FROZEN_CLOSURE_CERTIFICATION = (
        "ELIGIBLE_FOR_OUTSIDE_FROZEN_CLOSURE_CERTIFICATION"
    )


@dataclass(frozen=True, slots=True)
class OddReductionTargetSpec:
    target_id: str = TARGET_ID
    formal_name: str = TARGET_NAME
    input_type: str = "EntitySet[Bit]"
    set_sizes: tuple[int, ...] = (5, 6, 7, 8)
    relation: str = "sum(b(e) for e in S) mod 2 == 1"
    permutation_invariant: bool = True
    one_relation_for_all_sizes: bool = True
    per_size_lookup_allowed: bool = False
    entity_id_pattern_access_allowed: bool = False
    universe_rows: int = 480
    size_row_counts: tuple[tuple[int, int], ...] = (
        (5, 32),
        (6, 64),
        (7, 128),
        (8, 256),
    )
    split_rank_algorithm: str = (
        "HMAC-SHA256(K_split, target_id || set_size || label || "
        "canonical_bitstring)"
    )
    splits: tuple[SplitQuota, ...] = ODD_REDUCTION_SPLITS
    full_truth_table_visible_to_synthesis_agent: bool = False
    full_truth_table_use: tuple[str, ...] = (
        "old DSL extensional closure comparison",
        "target_truth_table_root generation",
        "exact old-language match decision",
        "independent replay certificate",
    )
    preflight_status: TargetPreflightStatus = (
        TargetPreflightStatus.AWAITING_COMPLETE_CLOSURE
    )
    outside_frozen_closure_certificate_issued: bool = False

    @property
    def diagnostic_universe_content_id(self) -> str:
        inputs = tuple(
            (row.universe_index, row.set_size, row.bits)
            for row in ODD_REDUCTION_UNIVERSE
        )
        return stable_hash(inputs, prefix="bounded_universe_")

    @property
    def diagnostic_target_table_content_id(self) -> str:
        table = tuple(
            (row.universe_index, row.bits, row.target_output)
            for row in ODD_REDUCTION_UNIVERSE
        )
        return stable_hash(table, prefix="target_truth_table_")

    @property
    def content_id(self) -> str:
        return stable_hash(self, prefix="target_spec_")


ODD_REDUCTION_TARGET: Final = OddReductionTargetSpec()


class BinaryXorStatus(str, Enum):
    TARGET_DESIGN_SANITY_ONLY = "TARGET_DESIGN_SANITY_ONLY"
    IN_LANGUAGE = "IN_LANGUAGE"


@dataclass(frozen=True, slots=True)
class BinaryXorSanitySpec:
    status: BinaryXorStatus = BinaryXorStatus.TARGET_DESIGN_SANITY_ONLY
    # This is the source-document spelling from section 2.6.  Under the
    # separately frozen operator typing it is not type-correct unless an
    # implicit Bit -> RationalValue coercion exists; no such coercion is
    # currently frozen.
    candidate_old_dsl_program: str = (
        "absolute(difference(bit_at(0), bit_at(1)))"
    )
    type_explicit_candidate_old_dsl_program: str = (
        "absolute(difference(bit_to_scalar(bit_at(0)), "
        "bit_to_scalar(bit_at(1))))"
    )
    implicit_bit_to_rational_coercion_frozen: bool = False
    source_candidate_typechecks_under_frozen_typing: bool = False
    truth_table: tuple[tuple[int, int, int], ...] = (
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
    )
    required_machine_evidence: tuple[str, ...] = (
        "closure enumeration status COMPLETE",
        "canonical witness or equivalent exists",
        "exact match on all four rows",
        "no undefined output",
        "Python and Rust replay agree on program hash and output root",
    )
    formal_language_verdict_issued: bool = False

    def __post_init__(self) -> None:
        if self.status is not BinaryXorStatus.TARGET_DESIGN_SANITY_ONLY:
            raise ValueError("binary XOR remains target-design sanity only")
        if (
            self.implicit_bit_to_rational_coercion_frozen
            or self.source_candidate_typechecks_under_frozen_typing
            or self.formal_language_verdict_issued
        ):
            raise ValueError("the source XOR witness has no executable verdict")


BINARY_XOR_SANITY: Final = BinaryXorSanitySpec()


@dataclass(frozen=True, slots=True)
class OmittedSinkUniverseRow:
    universe_index: int
    inflow_a: int
    inflow_b: int
    primary_outflow: int
    auxiliary_outflow: int
    full_balance_residual: int
    baseline_residual: int


def _sink_universe() -> tuple[OmittedSinkUniverseRow, ...]:
    rows: list[OmittedSinkUniverseRow] = []
    for inflow_a in range(5):
        for inflow_b in range(5):
            for primary_outflow in range(5):
                auxiliary_outflow = inflow_a + inflow_b - primary_outflow
                if 0 <= auxiliary_outflow <= 4:
                    rows.append(
                        OmittedSinkUniverseRow(
                            universe_index=len(rows),
                            inflow_a=inflow_a,
                            inflow_b=inflow_b,
                            primary_outflow=primary_outflow,
                            auxiliary_outflow=auxiliary_outflow,
                            full_balance_residual=(
                                inflow_a
                                + inflow_b
                                - primary_outflow
                                - auxiliary_outflow
                            ),
                            baseline_residual=(
                                inflow_a + inflow_b - primary_outflow
                            ),
                        )
                    )
    return tuple(rows)


OMITTED_SINK_UNIVERSE: Final = _sink_universe()


@dataclass(frozen=True, slots=True)
class ScopeSupportContract:
    discovery_support_total: int = 16
    discovery_support_per_scale: int = 8
    validation_support_total: int = 8
    sealed_support_total: int = 8
    discovery_d_zero_per_scale: int = 4
    discovery_d_positive_per_scale: int = 4


@dataclass(frozen=True, slots=True)
class NoFalseInventionContract:
    old_closure_exact_match_count_minimum: int = 1
    best_old_program_error_required: int = 0
    required_decision: str = "IN_LANGUAGE_REFINEMENT"
    promoted_new_symbol_count_required: int = 0
    outside_frozen_closure_certificate_count_required: int = 0
    sealed_false_invention_rate_required: RationalAtom = RationalAtom(0, 1)

    def passes(
        self,
        *,
        old_closure_exact_match_count: int,
        best_old_program_error: int,
        decision: str,
        promoted_new_symbol_count: int,
        outside_frozen_closure_certificate_count: int,
        sealed_false_invention_rate: Fraction,
    ) -> bool:
        return (
            old_closure_exact_match_count
            >= self.old_closure_exact_match_count_minimum
            and best_old_program_error == self.best_old_program_error_required
            and decision == self.required_decision
            and promoted_new_symbol_count
            == self.promoted_new_symbol_count_required
            and outside_frozen_closure_certificate_count
            == self.outside_frozen_closure_certificate_count_required
            and sealed_false_invention_rate
            == self.sealed_false_invention_rate_required.as_fraction()
        )


SINK_SUPPORT: Final = ScopeSupportContract()
NO_FALSE_INVENTION: Final = NoFalseInventionContract()


@dataclass(frozen=True, slots=True)
class ObservedOmittedSinkControlSpec:
    control_id: str = SINK_CONTROL_ID
    formal_name: str = SINK_CONTROL_NAME
    observed_channels: tuple[str, ...] = (
        "inflow_a",
        "inflow_b",
        "primary_outflow",
        "auxiliary_outflow",
    )
    all_channels_present_in_public_typed_evidence: bool = True
    latent_sink_allowed: bool = False
    value_domain: tuple[int, ...] = (0, 1, 2, 3, 4)
    generator_constraint: str = "a + b - c - d == 0; d = a + b - c"
    universe_rows: int = 85
    aggregation_map_id: str = "signed_balance_v1"
    correct_scope_id: str = "control_volume_all_observed_v1"
    # The source prose calls the baseline ``control_volume_primary_only_v1``
    # once, but that spelling is not one of the four frozen ScopeId values.
    # The machine contract binds the intended primary-only semantics to the
    # actual catalog member and retains the source label for auditability.
    baseline_scope_id: str = "scope_primary_only_v1"
    source_document_baseline_label: str = "control_volume_primary_only_v1"
    quantity_id: str = "q0"
    correct_old_dsl_program: str = (
        "approx_equal(aggregate(signed_balance_v1, "
        "control_volume_all_observed_v1, q0), 0, tolerance=0)"
    )
    correct_candidate_class: str = "scope_or_aggregation_refinement"
    support: ScopeSupportContract = SINK_SUPPORT
    no_false_invention: NoFalseInventionContract = NO_FALSE_INVENTION
    outside_frozen_closure_certificate_issued: bool = False

    @property
    def diagnostic_universe_content_id(self) -> str:
        inputs = tuple(
            (
                row.universe_index,
                row.inflow_a,
                row.inflow_b,
                row.primary_outflow,
                row.auxiliary_outflow,
            )
            for row in OMITTED_SINK_UNIVERSE
        )
        return stable_hash(inputs, prefix="bounded_universe_")

    @property
    def diagnostic_target_table_content_id(self) -> str:
        table = tuple(
            (row.universe_index, int(row.full_balance_residual == 0))
            for row in OMITTED_SINK_UNIVERSE
        )
        return stable_hash(table, prefix="target_truth_table_")

    @property
    def content_id(self) -> str:
        return stable_hash(self, prefix="sink_control_spec_")


OBSERVED_OMITTED_SINK_CONTROL: Final = ObservedOmittedSinkControlSpec()
