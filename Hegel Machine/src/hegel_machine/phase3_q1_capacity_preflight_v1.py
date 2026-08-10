"""Target-blind capacity preflight for the full v1.6 quotient surface.

This module is deliberately diagnostic.  It expands the complete latent
construction-signature cohort bank on each frozen production universe, but it
does not read target truth, split material, or role-match data; create formal
roots; or claim Q1 closure completion.  Its purpose is to measure the exact
direct-quotient construction before a production resource envelope is frozen.

The traversal uses a depth barrier.  Every operator increases raw AST depth by
one, so the immutable bank snapshot below depth ``d`` is a complete input to
the work queue for depth ``d``.  Newly retained representatives cannot feed
another application until the next barrier.  Visible Pareto points are report
data only: expansion always uses the complete latent cohort bank, including
dominated cohorts.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from itertools import combinations, combinations_with_replacement, product
import json
from time import monotonic
from typing import Final, Iterator, NoReturn, Sequence

from . import phase3_q0_input_adapter_v1 as _adapter
from . import phase3_q1_quotient_contract_v1 as _q1_contract
from .phase3_m3_bounded_enumerator_shrink6_v1 import _Shrink6Enumerator
from .phase3_q0_evaluator_v1 import evaluate_canonical_ast_on_environments_v1
from .phase3_q1_universe_v1 import ProductionUniverseV1, production_universe_v1
from .strict_ast_shrink6_v1 import (
    canonicalize_shrink6_source_ast,
    decode_shrink6_canonical_ast,
)
from .strict_ast_v1 import CanonicalAst, StrictAstError
from .strict_cbor_v1 import canonical_cbor_encode


SCHEMA_VERSION: Final = "hegel-phase3a-q1-capacity-preflight/1"
DSL_VERSION: Final = "hegel-old-dsl-v1.6.0"
CLOSURE_SEMANTICS_VERSION: Final = "hegel-quotient-closure-v1.0.1"
PREFLIGHT_ID: Final = "hegel-phase3a-q1-capacity-preflight-v1"

PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY: Final = (
    "PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY"
)
PREFLIGHT_CAPACITY_GUARD_HIT: Final = "PREFLIGHT_CAPACITY_GUARD_HIT"
LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED: Final = (
    "LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED"
)

# Compatibility aliases for the first local prototype API.  The values, and
# therefore every emitted terminal status, are the preregistered spellings.
PREFLIGHT_CLOSED: Final = PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY
INCONCLUSIVE_RESOURCE_LIMIT: Final = PREFLIGHT_CAPACITY_GUARD_HIT

REJECT_PREFLIGHT_INPUT_SIGNATURE: Final = "REJECT_PREFLIGHT_INPUT_SIGNATURE"
REJECT_FULL_NODE6_PREFLIGHT_NOT_AUTHORIZED: Final = (
    "REJECT_FULL_NODE6_PREFLIGHT_NOT_AUTHORIZED"
)

RESOURCE_GUARD_REGISTRY: Final = (
    (1, "RAW_OPERATOR_APPLICATIONS"),
    (2, "BEHAVIOR_CLASSES"),
    (3, "VISIBLE_FRONTIER_TOTAL"),
    (4, "VISIBLE_FRONTIER_PER_CLASS"),
    (5, "CONTINUATION_BANK_TOTAL"),
    (6, "CONTINUATION_BANK_PER_CLASS"),
    (7, "WORK_QUEUE_POINTS"),
    (8, "SATURATION_ROUNDS"),
    (9, "OUTPUT_BYTES"),
    (10, "SCRATCH_BYTES"),
    (11, "RESIDENT_MEMORY"),
    (12, "WALL_TIME"),
)
_REGISTERED_RESOURCE_GUARDS: Final = frozenset(
    name for _, name in RESOURCE_GUARD_REGISTRY
)
_RESOURCE_GUARD_ID_BY_NAME: Final = {
    name: guard_id for guard_id, name in RESOURCE_GUARD_REGISTRY
}

FAIL_PREFLIGHT_STRICT_BOUNDARY: Final = "FAIL_Q1_PREFLIGHT_STRICT_BOUNDARY"
FAIL_PREFLIGHT_AST_IDENTITY: Final = "FAIL_Q1_PREFLIGHT_AST_IDENTITY"
FAIL_PREFLIGHT_BEHAVIOR: Final = "FAIL_Q1_PREFLIGHT_BEHAVIOR"
FAIL_PREFLIGHT_HIGH_WATER_INVARIANT: Final = (
    "FAIL_Q1_PREFLIGHT_HIGH_WATER_INVARIANT"
)
FAIL_PREFLIGHT_LATE_LOWER_DEPTH_BANK_MUTATION: Final = (
    "FAIL_Q1_PREFLIGHT_LATE_LOWER_DEPTH_BANK_MUTATION"
)

LEAF_COUNT: Final = 810
OUTPUT_SORT_IDS: Final = {
    "Bool": 1,
    "Bit": 2,
    "Sign": 3,
    "BoundedInt": 4,
    "RationalValue": 5,
}
UNARY_NAMES: Final = {
    0: "bit_to_scalar",
    1: "int_to_scalar",
    2: "absolute",
    3: "sign",
}
BINARY_NAMES: Final = {
    1: "difference",
    2: "equal_exact",
    3: "less_equal",
    5: "same_sign",
    6: "opposite_sign",
}


class Q1CapacityPreflightError(RuntimeError):
    """Stable fail-closed error from the diagnostic quotient traversal."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class _ResourceLimit(Q1CapacityPreflightError):
    def __init__(self, guard_name: str, detail: str) -> None:
        if guard_name not in _REGISTERED_RESOURCE_GUARDS:
            _fail(
                FAIL_PREFLIGHT_STRICT_BOUNDARY,
                f"unregistered preflight resource guard {guard_name!r}",
            )
        super().__init__(INCONCLUSIVE_RESOURCE_LIMIT, f"{guard_name}: {detail}")
        self.resource_guard_id = _RESOURCE_GUARD_ID_BY_NAME[guard_name]
        self.resource_guard_name = guard_name


def _fail(code: str, detail: str) -> NoReturn:
    raise Q1CapacityPreflightError(code, detail)


def _require_explicit_nonfull_limits_v1(
    limits: PreflightLimitsV1 | None,
) -> PreflightLimitsV1:
    if type(limits) is not PreflightLimitsV1:
        _fail(
            REJECT_FULL_NODE6_PREFLIGHT_NOT_AUTHORIZED,
            "an explicit non-formal subset limit is required",
        )
    if limits.maximum_ast_depth == 3 and limits.maximum_ast_node_count == 6:
        _fail(
            REJECT_FULL_NODE6_PREFLIGHT_NOT_AUTHORIZED,
            "full depth-3/node-6 preflight requires the future admission token",
        )
    return limits


@dataclass(frozen=True, slots=True)
class PreflightLimitsV1:
    """Target-blind diagnostic limits, independent of the future Q1 envelope."""

    maximum_ast_depth: int = 3
    maximum_ast_node_count: int = 6
    maximum_raw_operator_applications: int = 4_294_967_295
    maximum_behavior_classes: int = 16_777_216
    maximum_continuation_bank_points: int = 67_108_864
    maximum_continuation_bank_points_per_class: int = 65_536
    maximum_visible_frontier_points: int = 33_554_432
    maximum_visible_frontier_points_per_class: int = 65_536
    maximum_work_queue_points: int = 67_108_864
    maximum_saturation_rounds: int = 16
    maximum_wall_time_seconds: int = 172_800

    def __post_init__(self) -> None:
        if type(self.maximum_ast_depth) is not int or not 0 <= self.maximum_ast_depth <= 3:
            raise ValueError("maximum_ast_depth must be in 0..3")
        if (
            type(self.maximum_ast_node_count) is not int
            or not 1 <= self.maximum_ast_node_count <= 6
        ):
            raise ValueError("maximum_ast_node_count must be in 1..6")
        for name in (
            "maximum_raw_operator_applications",
            "maximum_behavior_classes",
            "maximum_continuation_bank_points",
            "maximum_continuation_bank_points_per_class",
            "maximum_visible_frontier_points",
            "maximum_visible_frontier_points_per_class",
            "maximum_work_queue_points",
            "maximum_saturation_rounds",
            "maximum_wall_time_seconds",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class DepthBarrierRecordV1:
    depth: int
    barrier_kind: str
    eligible_raw_application_count: int
    strict_admitted_application_count: int
    rewrite_collapse_count: int
    new_behavior_class_count: int
    new_signature_cohort_count: int
    continuation_bank_mutation_count: int
    behavior_class_count_after_barrier: int
    signature_cohort_count_after_barrier: int
    continuation_bank_point_count_after_barrier: int
    visible_frontier_point_count_after_barrier: int


@dataclass(frozen=True, slots=True)
class Q1PartitionCapacityResultV1:
    schema_version: str
    preflight_id: str
    dsl_version: str
    closure_semantics_version: str
    input_signature_id: int
    universe_row_count: int
    universe_root: bytes
    limits: PreflightLimitsV1
    full_v16_structural_limits_applied: bool
    maximum_ast_depth: int
    maximum_ast_node_count: int
    terminal_status: str
    resource_guard_id: int | None
    resource_guard_name: str | None
    traversal_closed: bool
    frozen_leaf_count: int
    raw_operator_application_count: int
    strict_admitted_application_count: int
    rewrite_collapse_count: int
    behavior_class_count: int
    signature_cohort_count: int
    continuation_bank_point_count: int
    visible_frontier_point_count: int
    maximum_bank_points_per_class: int
    maximum_frontier_points_per_class: int
    peak_raw_operator_application_count: int
    peak_behavior_class_count: int
    peak_visible_frontier_point_count: int
    peak_visible_frontier_points_per_class: int
    peak_continuation_bank_point_count: int
    peak_continuation_bank_points_per_class: int
    peak_work_queue_points: int
    peak_saturation_round_count: int
    vector_cache_entry_count: int
    vector_cache_hit_count: int
    vector_cache_miss_count: int
    depth_barriers: tuple[DepthBarrierRecordV1, ...]
    diagnostic_only: bool
    formal_roots_generated: bool
    formal_roots: None
    target_truth_accessed: bool
    split_accessed: bool
    role_evaluation_performed: bool
    complete_claim_allowed: bool


@dataclass(frozen=True, slots=True)
class Q1CapacityPreflightResultV1:
    schema_version: str
    terminal_status: str
    partitions: tuple[Q1PartitionCapacityResultV1, ...]
    diagnostic_only: bool = True
    formal_roots_generated: bool = False
    formal_roots: None = None
    target_truth_accessed: bool = False
    split_accessed: bool = False
    role_evaluation_performed: bool = False
    complete_claim_allowed: bool = False


@dataclass(frozen=True, slots=True)
class Q1ImmutableCandidateApplicationV1:
    construction_depth: int
    coverage_code: int
    operator_parameters: tuple[object, ...]
    ordered_child_canonical_ast_cbors: tuple[bytes, ...]
    canonical_ast_cbor: bytes
    canonical_ast_hash: bytes
    rewrite_collapsed: bool

    def __post_init__(self) -> None:
        if type(self.construction_depth) is not int or not 0 <= self.construction_depth <= 3:
            raise TypeError("construction_depth must be exact int in 0..3")
        if type(self.coverage_code) is not int or not 0 <= self.coverage_code <= 0xFFFF:
            raise TypeError("coverage_code must be exact uint16")
        if type(self.operator_parameters) is not tuple:
            raise TypeError("operator_parameters must be exact tuple")
        if type(self.ordered_child_canonical_ast_cbors) is not tuple or any(
            type(value) is not bytes or not value
            for value in self.ordered_child_canonical_ast_cbors
        ):
            raise TypeError("ordered child ASTs must be non-empty byte tuples")
        if type(self.canonical_ast_cbor) is not bytes or not self.canonical_ast_cbor:
            raise TypeError("canonical_ast_cbor must be non-empty bytes")
        if type(self.canonical_ast_hash) is not bytes or len(self.canonical_ast_hash) != 32:
            raise TypeError("canonical_ast_hash must be 32 bytes")
        if type(self.rewrite_collapsed) is not bool:
            raise TypeError("rewrite_collapsed must be exact bool")
        try:
            output_ast = decode_shrink6_canonical_ast(self.canonical_ast_cbor)
        except (StrictAstError, ValueError) as error:
            _fail(FAIL_PREFLIGHT_AST_IDENTITY, f"candidate output AST: {error}")
        if (
            output_ast.cbor_bytes != self.canonical_ast_cbor
            or output_ast.digest != self.canonical_ast_hash
        ):
            _fail(FAIL_PREFLIGHT_AST_IDENTITY, "candidate output AST identity differs")

        if self.construction_depth == 0:
            if (
                not 0 <= self.coverage_code < LEAF_COUNT
                or self.operator_parameters
                or self.ordered_child_canonical_ast_cbors
                or self.rewrite_collapsed
            ):
                _fail(FAIL_PREFLIGHT_STRICT_BOUNDARY, "leaf candidate wire differs")
            leaf = _frozen_leaf_asts_v1(raw_cap=LEAF_COUNT)[self.coverage_code]
            if leaf.cbor_bytes != self.canonical_ast_cbor:
                _fail(FAIL_PREFLIGHT_AST_IDENTITY, "leaf manifest index differs")
            return

        child_asts: list[CanonicalAst] = []
        for child_cbor in self.ordered_child_canonical_ast_cbors:
            try:
                child = decode_shrink6_canonical_ast(child_cbor)
            except (StrictAstError, ValueError) as error:
                _fail(FAIL_PREFLIGHT_AST_IDENTITY, f"candidate child AST: {error}")
            if child.cbor_bytes != child_cbor:
                _fail(FAIL_PREFLIGHT_AST_IDENTITY, "candidate child replay differs")
            child_asts.append(child)
        children = tuple(child_asts)
        if not children or self.construction_depth != 1 + max(
            child.metrics.depth for child in children
        ):
            _fail(FAIL_PREFLIGHT_STRICT_BOUNDARY, "candidate construction depth differs")

        if self.coverage_code in (0x1000, 0x1001, 0x1002, 0x1003):
            if len(children) != 1 or self.operator_parameters:
                _fail(FAIL_PREFLIGHT_STRICT_BOUNDARY, "unary candidate arity differs")
            expected_node = (
                1,
                self.coverage_code - 0x1000,
                children[0].value[1],
            )
        elif self.coverage_code in (0x2001, 0x2002, 0x2003, 0x2005, 0x2006):
            if len(children) != 2 or self.operator_parameters:
                _fail(FAIL_PREFLIGHT_STRICT_BOUNDARY, "binary candidate arity differs")
            if self.coverage_code in (0x2002, 0x2005, 0x2006):
                commutative_keys = tuple(
                    (
                        sha256(canonical_cbor_encode(child.value[1])).digest(),
                        canonical_cbor_encode(child.value[1]),
                    )
                    for child in children
                )
                if commutative_keys != tuple(sorted(commutative_keys)):
                    _fail(
                        FAIL_PREFLIGHT_STRICT_BOUNDARY,
                        "commutative child order differs",
                    )
            expected_node = (
                2,
                self.coverage_code - 0x2000,
                children[0].value[1],
                children[1].value[1],
            )
        elif self.coverage_code in (0x3001, 0x3002):
            tolerance = self.coverage_code - 0x3000
            if len(children) != 2 or self.operator_parameters != (tolerance,):
                _fail(FAIL_PREFLIGHT_STRICT_BOUNDARY, "approx candidate wire differs")
            commutative_keys = tuple(
                (
                    sha256(canonical_cbor_encode(child.value[1])).digest(),
                    canonical_cbor_encode(child.value[1]),
                )
                for child in children
            )
            if commutative_keys != tuple(sorted(commutative_keys)):
                _fail(FAIL_PREFLIGHT_STRICT_BOUNDARY, "approx child order differs")
            expected_node = (
                3,
                0,
                children[0].value[1],
                children[1].value[1],
                tolerance,
            )
        elif self.coverage_code == 0x4002:
            if len(children) != 2 or self.operator_parameters:
                _fail(FAIL_PREFLIGHT_STRICT_BOUNDARY, "AND2 candidate wire differs")
            if (
                children[0].cbor_bytes == children[1].cbor_bytes
                or any(child.value[1][0] == 4 for child in children)
                or tuple(
                    (
                        child.metrics.depth,
                        child.metrics.node_count,
                        OUTPUT_SORT_IDS[child.metrics.output_sort],
                        child.root_operator_id,
                        child.cbor_bytes,
                    )
                    for child in children
                )
                != tuple(
                    sorted(
                        (
                            child.metrics.depth,
                            child.metrics.node_count,
                            OUTPUT_SORT_IDS[child.metrics.output_sort],
                            child.root_operator_id,
                            child.cbor_bytes,
                        )
                        for child in children
                    )
                )
            ):
                _fail(FAIL_PREFLIGHT_STRICT_BOUNDARY, "AND2 child order differs")
            expected_node = (4, tuple(child.value[1] for child in children))
        else:
            _fail(FAIL_PREFLIGHT_STRICT_BOUNDARY, "candidate coverage code is unknown")
        try:
            expected_ast = canonicalize_shrink6_source_ast(
                _canonical_node_to_source(expected_node)
            )
        except (StrictAstError, ValueError) as error:
            _fail(FAIL_PREFLIGHT_STRICT_BOUNDARY, f"candidate reconstruction: {error}")
        if expected_ast.cbor_bytes != self.canonical_ast_cbor:
            _fail(FAIL_PREFLIGHT_AST_IDENTITY, "candidate output differs from children")
        if self.rewrite_collapsed != (expected_ast.value[1] != expected_node):
            _fail(FAIL_PREFLIGHT_STRICT_BOUNDARY, "rewrite-collapse flag differs")


@dataclass(frozen=True, slots=True)
class _Program:
    ast: CanonicalAst
    behavior: tuple[object, ...]
    behavior_key: bytes

    @property
    def sort(self) -> str:
        return self.ast.metrics.output_sort

    @property
    def node(self) -> tuple[object, ...]:
        value = self.ast.value[1]
        assert isinstance(value, tuple)
        return value

    @property
    def global_key(self) -> tuple[int, int, int, int, bytes]:
        return (
            self.ast.metrics.depth,
            self.ast.metrics.node_count,
            OUTPUT_SORT_IDS[self.sort],
            self.ast.root_operator_id,
            self.ast.cbor_bytes,
        )

    @property
    def commutative_key(self) -> tuple[bytes, bytes]:
        encoded = canonical_cbor_encode(self.node)
        return sha256(encoded).digest(), encoded


@dataclass(frozen=True, slots=True)
class _Candidate:
    operator_code: int
    source_ast: tuple[object, ...]
    expected_node: tuple[object, ...]
    children: tuple[_Program, ...]


@dataclass(slots=True)
class _Cohort:
    signature: _q1_contract.FutureAdmissibilitySignatureV1
    representatives: dict[bytes, _Program]


@dataclass(slots=True)
class _BehaviorClass:
    behavior: tuple[object, ...]
    cohorts: dict[bytes, _Cohort]


@dataclass(frozen=True, slots=True)
class _SnapshotRepresentativeMaterialV1:
    canonical_ast_cbor: bytes
    canonical_ast_hash: bytes


@dataclass(frozen=True, slots=True)
class _SnapshotCohortMaterialV1:
    signature: _q1_contract.FutureAdmissibilitySignatureV1
    representatives: tuple[_SnapshotRepresentativeMaterialV1, ...]
    visible_frontier_member: bool


@dataclass(frozen=True, slots=True)
class _SnapshotClassMaterialV1:
    behavior_key: bytes
    behavior: tuple[object, ...]
    cohorts: tuple[_SnapshotCohortMaterialV1, ...]


def _canonical_cell(value: object) -> tuple[object, ...]:
    if value is _adapter.BOTTOM:
        return (0,)
    if type(value) is Fraction:
        return (1, (value.numerator, value.denominator))
    if type(value) is bool:
        return (1, value)
    if type(value) is int:
        return (1, value)
    _fail(FAIL_PREFLIGHT_BEHAVIOR, f"unsupported runtime value {value!r}")


def _behavior_key(sort: str, behavior: Sequence[object]) -> bytes:
    return canonical_cbor_encode(
        (
            OUTPUT_SORT_IDS[sort],
            tuple(_canonical_cell(value) for value in behavior),
        )
    )


def _canonical_node_to_source(node: tuple[object, ...]) -> tuple[object, ...]:
    tag = node[0]
    if tag == 0:
        leaf = node[1]
        if leaf in (0, 1, 4, 5):
            name = {0: "scalar_const", 1: "bit_at", 4: "context_flag", 5: "task_flag"}[leaf]
            return (name, node[2])
        if leaf == 2:
            return ("set_size",)
        if leaf == 3:
            return ("aggregate", node[2], node[3], node[4], node[5])
    if tag == 1:
        return (UNARY_NAMES[node[1]], _canonical_node_to_source(node[2]))
    if tag == 2:
        return (
            BINARY_NAMES[node[1]],
            _canonical_node_to_source(node[2]),
            _canonical_node_to_source(node[3]),
        )
    if tag == 3:
        return (
            "approx_equal",
            _canonical_node_to_source(node[2]),
            _canonical_node_to_source(node[3]),
            node[4],
        )
    if tag == 4:
        return ("top_level_AND",) + tuple(
            _canonical_node_to_source(child) for child in node[1]
        )
    _fail(FAIL_PREFLIGHT_STRICT_BOUNDARY, "unknown canonical AST node")


def _resource_eligible(
    children: Sequence[_Program],
    *,
    target_depth: int,
    limits: PreflightLimitsV1,
    conjunction: bool = False,
) -> bool:
    metrics = tuple(child.ast.metrics for child in children)
    depth = 1 + max(metric.depth for metric in metrics)
    nodes = 1 + sum(metric.node_count for metric in metrics)
    aggregate = sum(metric.aggregate_leaf_count for metric in metrics)
    scalar = sum(metric.scalar_parameter_occurrences for metric in metrics)
    scopes = sum(metric.scope_clause_count for metric in metrics)
    bits = frozenset().union(*(metric.distinct_bit_slots for metric in metrics))
    return (
        depth == target_depth
        and depth <= limits.maximum_ast_depth
        and nodes <= limits.maximum_ast_node_count
        and aggregate <= 1
        and scalar <= 3
        and scopes <= 2
        and len(bits) <= 4
        and (not conjunction or len(children) == 2)
    )


def _operator_candidates(
    programs: Sequence[_Program],
    *,
    target_depth: int,
    limits: PreflightLimitsV1,
) -> Iterator[_Candidate]:
    groups = {
        sort: tuple(
            sorted(
                (program for program in programs if program.sort == sort),
                key=lambda program: program.global_key,
            )
        )
        for sort in OUTPUT_SORT_IDS
    }

    for operator, input_sort in (
        (0, "Bit"),
        (1, "BoundedInt"),
        (2, "RationalValue"),
        (3, "RationalValue"),
    ):
        for child in groups[input_sort]:
            if not _resource_eligible(
                (child,), target_depth=target_depth, limits=limits
            ):
                continue
            source = (UNARY_NAMES[operator], _canonical_node_to_source(child.node))
            yield _Candidate(
                0x1000 + operator,
                source,
                (1, operator, child.node),
                (child,),
            )

    rational = groups["RationalValue"]
    for left, right in product(rational, repeat=2):
        if not _resource_eligible(
            (left, right), target_depth=target_depth, limits=limits
        ):
            continue
        left_source = _canonical_node_to_source(left.node)
        right_source = _canonical_node_to_source(right.node)
        yield _Candidate(
            0x2001,
            ("difference", left_source, right_source),
            (2, 1, left.node, right.node),
            (left, right),
        )
        yield _Candidate(
            0x2003,
            ("less_equal", left_source, right_source),
            (2, 3, left.node, right.node),
            (left, right),
        )

    commutative_rational = tuple(
        sorted(rational, key=lambda program: program.commutative_key)
    )
    for left, right in combinations_with_replacement(commutative_rational, 2):
        if not _resource_eligible(
            (left, right), target_depth=target_depth, limits=limits
        ):
            continue
        left_source = _canonical_node_to_source(left.node)
        right_source = _canonical_node_to_source(right.node)
        yield _Candidate(
            0x2002,
            ("equal_exact", left_source, right_source),
            (2, 2, left.node, right.node),
            (left, right),
        )
        for tolerance in (1, 2):
            yield _Candidate(
                0x3000 + tolerance,
                ("approx_equal", left_source, right_source, tolerance),
                (3, 0, left.node, right.node, tolerance),
                (left, right),
            )

    signs = tuple(sorted(groups["Sign"], key=lambda program: program.commutative_key))
    for left, right in combinations_with_replacement(signs, 2):
        if not _resource_eligible(
            (left, right), target_depth=target_depth, limits=limits
        ):
            continue
        for operator in (5, 6):
            yield _Candidate(
                0x2000 + operator,
                (
                    BINARY_NAMES[operator],
                    _canonical_node_to_source(left.node),
                    _canonical_node_to_source(right.node),
                ),
                (2, operator, left.node, right.node),
                (left, right),
            )

    bool_atoms = tuple(
        sorted(
            (program for program in groups["Bool"] if program.node[0] != 4),
            key=lambda program: canonical_cbor_encode(program.node),
        )
    )
    for left, right in combinations(bool_atoms, 2):
        if not _resource_eligible(
            (left, right),
            target_depth=target_depth,
            limits=limits,
            conjunction=True,
        ):
            continue
        yield _Candidate(
            0x4002,
            (
                "top_level_AND",
                _canonical_node_to_source(left.node),
                _canonical_node_to_source(right.node),
            ),
            (4, (left.node, right.node)),
            (left, right),
        )


@lru_cache(maxsize=None)
def _frozen_leaf_asts_v1(*, raw_cap: int) -> tuple[CanonicalAst, ...]:
    enumerator = _Shrink6Enumerator(raw_cap=max(LEAF_COUNT, raw_cap))
    for sort_id in range(1, 6):
        enumerator.leaves(sort_id)
    leaves = tuple(
        sorted(
            (program.ast for programs in enumerator.groups.values() for program in programs),
            key=lambda ast: (
                OUTPUT_SORT_IDS[ast.metrics.output_sort],
                ast.root_operator_id,
                ast.cbor_bytes,
            ),
        )
    )
    if len(leaves) != LEAF_COUNT or len({ast.cbor_bytes for ast in leaves}) != LEAF_COUNT:
        _fail(FAIL_PREFLIGHT_STRICT_BOUNDARY, "full v1.6 leaf manifest is not 810")
    return leaves


def immutable_candidate_applications_v1(
    continuation_bank_canonical_ast_cbors: tuple[bytes, ...],
    *,
    limits: PreflightLimitsV1,
) -> tuple[Q1ImmutableCandidateApplicationV1, ...]:
    """Export the unique target-blind candidate semantics as immutable views."""

    if type(continuation_bank_canonical_ast_cbors) is not tuple or any(
        type(value) is not bytes or not value
        for value in continuation_bank_canonical_ast_cbors
    ):
        raise TypeError("continuation bank ASTs must be an exact byte tuple")
    if len(set(continuation_bank_canonical_ast_cbors)) != len(
        continuation_bank_canonical_ast_cbors
    ):
        _fail(FAIL_PREFLIGHT_AST_IDENTITY, "continuation bank contains duplicate AST")
    _require_explicit_nonfull_limits_v1(limits)

    output: list[Q1ImmutableCandidateApplicationV1] = []
    for coverage_code, ast in enumerate(_frozen_leaf_asts_v1(raw_cap=LEAF_COUNT)):
        output.append(
            Q1ImmutableCandidateApplicationV1(
                construction_depth=0,
                coverage_code=coverage_code,
                operator_parameters=(),
                ordered_child_canonical_ast_cbors=(),
                canonical_ast_cbor=ast.cbor_bytes,
                canonical_ast_hash=ast.digest,
                rewrite_collapsed=False,
            )
        )

    programs: list[_Program] = []
    for ast_cbor in sorted(continuation_bank_canonical_ast_cbors):
        ast = decode_shrink6_canonical_ast(ast_cbor)
        if ast.cbor_bytes != ast_cbor:
            _fail(FAIL_PREFLIGHT_AST_IDENTITY, "continuation AST replay differs")
        programs.append(_Program(ast=ast, behavior=(), behavior_key=b""))
    program_tuple = tuple(programs)
    for depth in range(1, limits.maximum_ast_depth + 1):
        for candidate in _operator_candidates(
            program_tuple,
            target_depth=depth,
            limits=limits,
        ):
            try:
                ast = canonicalize_shrink6_source_ast(candidate.source_ast)
                replay = decode_shrink6_canonical_ast(ast.cbor_bytes)
            except StrictAstError as error:
                _fail(FAIL_PREFLIGHT_STRICT_BOUNDARY, str(error))
            if replay.cbor_bytes != ast.cbor_bytes or replay.digest != ast.digest:
                _fail(FAIL_PREFLIGHT_AST_IDENTITY, "candidate strict replay differs")
            parameters: tuple[object, ...] = (
                (candidate.operator_code - 0x3000,)
                if candidate.operator_code in (0x3001, 0x3002)
                else ()
            )
            output.append(
                Q1ImmutableCandidateApplicationV1(
                    construction_depth=depth,
                    coverage_code=candidate.operator_code,
                    operator_parameters=parameters,
                    ordered_child_canonical_ast_cbors=tuple(
                        child.ast.cbor_bytes for child in candidate.children
                    ),
                    canonical_ast_cbor=ast.cbor_bytes,
                    canonical_ast_hash=ast.digest,
                    rewrite_collapsed=ast.value[1] != candidate.expected_node,
                )
            )
    return tuple(output)


def _bottom(value: object) -> bool:
    return value is _adapter.BOTTOM


def _bounded_rational(value: Fraction) -> object:
    return value if value in _adapter.RATIONAL_VALUE_GRID else _adapter.BOTTOM


def _apply_candidate_vector(candidate: _Candidate) -> tuple[object, ...]:
    vectors = tuple(child.behavior for child in candidate.children)
    if not vectors or len({len(vector) for vector in vectors}) != 1:
        _fail(FAIL_PREFLIGHT_BEHAVIOR, "candidate child vector lengths differ")
    output: list[object] = []
    for cells in zip(*vectors, strict=True):
        if any(_bottom(cell) for cell in cells):
            output.append(_adapter.BOTTOM)
            continue
        code = candidate.operator_code
        if code == 0x1000:
            value = cells[0]
            if type(value) is not int or value not in (0, 1):
                _fail(FAIL_PREFLIGHT_BEHAVIOR, "bit_to_scalar received non-Bit")
            output.append(Fraction(value, 1))
        elif code == 0x1001:
            value = cells[0]
            if type(value) is not int or not -8 <= value <= 8:
                _fail(FAIL_PREFLIGHT_BEHAVIOR, "int_to_scalar received non-int")
            output.append(Fraction(value, 1))
        elif code == 0x1002:
            value = cells[0]
            if type(value) is not Fraction:
                _fail(FAIL_PREFLIGHT_BEHAVIOR, "absolute received non-rational")
            output.append(_bounded_rational(abs(value)))
        elif code == 0x1003:
            value = cells[0]
            if type(value) is not Fraction:
                _fail(FAIL_PREFLIGHT_BEHAVIOR, "sign received non-rational")
            output.append(-1 if value < 0 else 1 if value > 0 else 0)
        elif code in (0x2001, 0x2002, 0x2003, 0x3001, 0x3002):
            left, right = cells
            if type(left) is not Fraction or type(right) is not Fraction:
                _fail(FAIL_PREFLIGHT_BEHAVIOR, "rational binary received wrong sort")
            if code == 0x2001:
                output.append(_bounded_rational(left - right))
            elif code == 0x2002:
                output.append(left == right)
            elif code == 0x2003:
                output.append(left <= right)
            else:
                tolerance = Fraction(1, 4) if code == 0x3001 else Fraction(1, 2)
                output.append(abs(left - right) <= tolerance)
        elif code in (0x2005, 0x2006):
            left, right = cells
            if (
                type(left) is not int
                or type(right) is not int
                or left not in (-1, 0, 1)
                or right not in (-1, 0, 1)
            ):
                _fail(FAIL_PREFLIGHT_BEHAVIOR, "sign binary received wrong sort")
            output.append(left == right if code == 0x2005 else left == -right and left != 0)
        elif code == 0x4002:
            left, right = cells
            if type(left) is not bool or type(right) is not bool:
                _fail(FAIL_PREFLIGHT_BEHAVIOR, "AND2 received non-Bool")
            output.append(left and right)
        else:  # pragma: no cover - generator closes the registry
            _fail(FAIL_PREFLIGHT_BEHAVIOR, f"unknown operator code {code}")
    return tuple(output)


class _State:
    def __init__(
        self,
        universe: ProductionUniverseV1,
        limits: PreflightLimitsV1,
    ) -> None:
        self.universe = universe
        self.limits = limits
        self.start_time = monotonic()
        self.classes: dict[bytes, _BehaviorClass] = {}
        self.vector_cache: dict[tuple[int, tuple[bytes, ...]], tuple[object, ...]] = {}
        self.vector_cache_hits = 0
        self.vector_cache_misses = 0
        self.peak_visible_frontier_point_count = 0
        self.peak_visible_frontier_points_per_class = 0
        self.peak_work_queue_points = 0
        self.peak_saturation_round_count = 0
        self.raw_count = 0
        self.current_barrier_raw_count = 0
        self.strict_count = 0
        self.rewrite_count = 0
        self.barriers: list[DepthBarrierRecordV1] = []

    def _wall_guard(self) -> None:
        if monotonic() - self.start_time > self.limits.maximum_wall_time_seconds:
            raise _ResourceLimit("WALL_TIME", "preflight active wall time exceeded")

    def _counts(self) -> tuple[int, int, int, int, int, int]:
        class_count = len(self.classes)
        cohort_count = sum(len(item.cohorts) for item in self.classes.values())
        bank_by_class = tuple(
            sum(len(cohort.representatives) for cohort in item.cohorts.values())
            for item in self.classes.values()
        )
        frontier_by_class = tuple(self._visible_frontier_count(item) for item in self.classes.values())
        bank_count = sum(bank_by_class)
        frontier_count = sum(frontier_by_class)
        return (
            class_count,
            cohort_count,
            bank_count,
            frontier_count,
            max(bank_by_class, default=0),
            max(frontier_by_class, default=0),
        )

    @staticmethod
    def _visible_frontier_count(item: _BehaviorClass) -> int:
        material = tuple(item.cohorts.values())
        return sum(
            len(cohort.representatives)
            for index, cohort in enumerate(material)
            if _State._cohort_is_visible(material, index)
        )

    @staticmethod
    def _cohort_is_visible(
        material: tuple[_Cohort, ...],
        index: int,
    ) -> bool:
        cohort = material[index]
        return not any(
            other.signature.dominates(cohort.signature)
            and len(other.representatives) >= len(cohort.representatives)
            for other_index, other in enumerate(material)
            if other_index != index
        )

    def _guard_state(self) -> tuple[int, int, int, int, int, int]:
        classes, cohorts, bank, frontier, max_bank, max_frontier = self._counts()
        checks = (
            ("BEHAVIOR_CLASSES", classes, self.limits.maximum_behavior_classes),
            (
                "VISIBLE_FRONTIER_TOTAL",
                frontier,
                self.limits.maximum_visible_frontier_points,
            ),
            (
                "VISIBLE_FRONTIER_PER_CLASS",
                max_frontier,
                self.limits.maximum_visible_frontier_points_per_class,
            ),
            # Signature cohorts are internal construction detail.  They share
            # the preregistered total-bank guard rather than emitting an
            # unregistered diagnostic guard ID.
            (
                "CONTINUATION_BANK_TOTAL",
                cohorts,
                self.limits.maximum_continuation_bank_points,
            ),
            (
                "CONTINUATION_BANK_TOTAL",
                bank,
                self.limits.maximum_continuation_bank_points,
            ),
            (
                "CONTINUATION_BANK_PER_CLASS",
                max_bank,
                self.limits.maximum_continuation_bank_points_per_class,
            ),
            # The semantic vector cache is a deduplicated work reservoir, not
            # a separately registered closure resource.
            (
                "WORK_QUEUE_POINTS",
                len(self.vector_cache),
                self.limits.maximum_work_queue_points,
            ),
        )
        for guard, observed, maximum in checks:
            if observed > maximum:
                raise _ResourceLimit(guard, f"{observed} exceeds {maximum}")
        self._wall_guard()
        return classes, cohorts, bank, frontier, max_bank, max_frontier

    def _record_accepted_event_high_water(
        self,
        counts: tuple[int, int, int, int, int, int],
    ) -> None:
        """Commit guard-specific peaks only after the event is admitted."""

        frontier = counts[3]
        max_frontier = counts[5]
        work_queue = max(len(self.vector_cache), self.current_barrier_raw_count)
        self.peak_visible_frontier_point_count = max(
            self.peak_visible_frontier_point_count,
            frontier,
        )
        self.peak_visible_frontier_points_per_class = max(
            self.peak_visible_frontier_points_per_class,
            max_frontier,
        )
        self.peak_work_queue_points = max(
            self.peak_work_queue_points,
            work_queue,
        )

    def _consume_raw(self) -> None:
        if self.raw_count + 1 > self.limits.maximum_raw_operator_applications:
            raise _ResourceLimit(
                "RAW_OPERATOR_APPLICATIONS",
                "the next application would exceed the frozen diagnostic limit",
            )
        if self.current_barrier_raw_count + 1 > self.limits.maximum_work_queue_points:
            raise _ResourceLimit(
                "WORK_QUEUE_POINTS",
                "the next depth-barrier work item would exceed its limit",
            )
        self._wall_guard()
        self.raw_count += 1
        self.current_barrier_raw_count += 1

    def _event_counter_checkpoint(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.raw_count,
            self.current_barrier_raw_count,
            self.strict_count,
            self.rewrite_count,
            self.vector_cache_hits,
            self.vector_cache_misses,
        )

    def _restore_event_counters(
        self,
        checkpoint: tuple[int, int, int, int, int, int],
    ) -> None:
        (
            self.raw_count,
            self.current_barrier_raw_count,
            self.strict_count,
            self.rewrite_count,
            self.vector_cache_hits,
            self.vector_cache_misses,
        ) = checkpoint

    def insert(
        self,
        ast: CanonicalAst,
        behavior: tuple[object, ...],
        *,
        current_depth: int,
    ) -> tuple[int, int, int]:
        replay = decode_shrink6_canonical_ast(ast.cbor_bytes)
        if replay.cbor_bytes != ast.cbor_bytes or replay.digest != ast.digest:
            _fail(FAIL_PREFLIGHT_AST_IDENTITY, "strict AST replay identity differs")
        key = _behavior_key(ast.metrics.output_sort, behavior)
        item = self.classes.get(key)
        class_created = item is None
        class_delta = 0
        if item is None:
            item = _BehaviorClass(behavior, {})
            self.classes[key] = item
            class_delta = 1
        elif item.behavior != behavior:
            _fail(FAIL_PREFLIGHT_BEHAVIOR, "behavior key alias changed value")

        signature = _q1_contract.future_signature_from_ast_v1(ast)
        signature_key = canonical_cbor_encode(signature.canonical_object())
        cohort = item.cohorts.get(signature_key)
        cohort_created = cohort is None
        cohort_delta = 0
        if cohort is None:
            cohort = _Cohort(signature, {})
            item.cohorts[signature_key] = cohort
            cohort_delta = 1
        elif cohort.signature != signature:
            _fail(FAIL_PREFLIGHT_AST_IDENTITY, "signature key alias changed value")

        prior_representatives = dict(cohort.representatives)
        before = tuple(sorted(prior_representatives))
        candidates = dict(cohort.representatives)
        candidates[ast.cbor_bytes] = _Program(ast, behavior, key)
        capacity = _q1_contract.normalization_witness_capacity_v1(
            signature.output_sort_id
        )
        selected = tuple(sorted(candidates)[:capacity])
        cohort.representatives = {value: candidates[value] for value in selected}
        after = tuple(sorted(cohort.representatives))
        bank_delta = int(before != after)
        if ast.metrics.depth < current_depth and bank_delta:
            _fail(
                FAIL_PREFLIGHT_LATE_LOWER_DEPTH_BANK_MUTATION,
                "a later barrier changed an earlier-depth cohort bank",
            )
        try:
            guarded_counts = self._guard_state()
        except _ResourceLimit:
            # Section 6 of the preregistration admits a value equal to a
            # ceiling, but the next event must terminate *before acceptance*.
            # Restore the exact pre-event state before exposing diagnostics.
            if class_created:
                del self.classes[key]
            elif cohort_created:
                del item.cohorts[signature_key]
            else:
                cohort.representatives = prior_representatives
            raise
        self._record_accepted_event_high_water(guarded_counts)
        return class_delta, cohort_delta, bank_delta

    def bank_programs(self) -> tuple[_Program, ...]:
        values: dict[bytes, _Program] = {}
        for item in self.classes.values():
            for cohort in item.cohorts.values():
                for program in cohort.representatives.values():
                    prior = values.get(program.ast.cbor_bytes)
                    if prior is not None and prior.behavior_key != program.behavior_key:
                        _fail(FAIL_PREFLIGHT_BEHAVIOR, "one AST has two behaviors")
                    values[program.ast.cbor_bytes] = program
        return tuple(sorted(values.values(), key=lambda program: program.global_key))

    def immutable_snapshot_material(
        self,
    ) -> tuple[_SnapshotClassMaterialV1, ...]:
        """Copy the complete quotient state into canonical immutable rows."""

        output: list[_SnapshotClassMaterialV1] = []
        for behavior_key in sorted(self.classes):
            item = self.classes[behavior_key]
            cohort_rows = tuple(
                sorted(
                    item.cohorts.items(),
                    key=lambda row: row[0],
                )
            )
            material = tuple(cohort for _, cohort in cohort_rows)
            cohorts: list[_SnapshotCohortMaterialV1] = []
            for index, (_signature_key, cohort) in enumerate(cohort_rows):
                representatives = tuple(
                    _SnapshotRepresentativeMaterialV1(
                        canonical_ast_cbor=program.ast.cbor_bytes,
                        canonical_ast_hash=program.ast.digest,
                    )
                    for program in sorted(
                        cohort.representatives.values(),
                        key=lambda value: value.ast.cbor_bytes,
                    )
                )
                cohorts.append(
                    _SnapshotCohortMaterialV1(
                        signature=cohort.signature,
                        representatives=representatives,
                        visible_frontier_member=self._cohort_is_visible(
                            material,
                            index,
                        ),
                    )
                )
            output.append(
                _SnapshotClassMaterialV1(
                    behavior_key=behavior_key,
                    behavior=tuple(item.behavior),
                    cohorts=tuple(cohorts),
                )
            )
        return tuple(output)

    def candidate_vector(self, candidate: _Candidate) -> tuple[object, ...]:
        key = self._candidate_vector_key(candidate)
        cached = self.vector_cache.get(key)
        if cached is not None:
            self.vector_cache_hits += 1
            return cached
        if len(self.vector_cache) + 1 > self.limits.maximum_work_queue_points:
            raise _ResourceLimit(
                "WORK_QUEUE_POINTS",
                "the next semantic vector cache entry would exceed its limit",
            )
        value = _apply_candidate_vector(candidate)
        self.vector_cache[key] = value
        self.vector_cache_misses += 1
        return value

    @staticmethod
    def _candidate_vector_key(
        candidate: _Candidate,
    ) -> tuple[int, tuple[bytes, ...]]:
        return (
            candidate.operator_code,
            tuple(child.behavior_key for child in candidate.children),
        )

    def seed(self) -> None:
        if len(self.barriers) + 1 > self.limits.maximum_saturation_rounds:
            raise _ResourceLimit(
                "SATURATION_ROUNDS",
                "the leaf barrier would exceed the diagnostic round limit",
            )
        self.current_barrier_raw_count = 0
        before = self._counts()
        leaves = _frozen_leaf_asts_v1(
            raw_cap=self.limits.maximum_raw_operator_applications
        )
        environments = self.universe.observation_environments()
        bank_delta = 0
        for ast in leaves:
            checkpoint = self._event_counter_checkpoint()
            try:
                self._consume_raw()
                behavior = evaluate_canonical_ast_on_environments_v1(ast, environments)
                self.strict_count += 1
                deltas = self.insert(ast, behavior, current_depth=0)
            except _ResourceLimit:
                self._restore_event_counters(checkpoint)
                raise
            bank_delta += deltas[2]
        after = self._counts()
        self.barriers.append(
            DepthBarrierRecordV1(
                depth=0,
                barrier_kind="LEAF_SEED",
                eligible_raw_application_count=LEAF_COUNT,
                strict_admitted_application_count=LEAF_COUNT,
                rewrite_collapse_count=0,
                new_behavior_class_count=after[0] - before[0],
                new_signature_cohort_count=after[1] - before[1],
                continuation_bank_mutation_count=bank_delta,
                behavior_class_count_after_barrier=after[0],
                signature_cohort_count_after_barrier=after[1],
                continuation_bank_point_count_after_barrier=after[2],
                visible_frontier_point_count_after_barrier=after[3],
            )
        )
        self.peak_saturation_round_count = max(
            self.peak_saturation_round_count,
            len(self.barriers),
        )

    def expand_depth(self, depth: int) -> None:
        if len(self.barriers) + 1 > self.limits.maximum_saturation_rounds:
            raise _ResourceLimit(
                "SATURATION_ROUNDS",
                "the next depth barrier would exceed the diagnostic round limit",
            )
        self.current_barrier_raw_count = 0
        snapshot = self.bank_programs()
        if any(program.ast.metrics.depth >= depth for program in snapshot):
            _fail(
                FAIL_PREFLIGHT_AST_IDENTITY,
                "depth barrier snapshot contains a non-prior representative",
            )
        before = self._counts()
        raw_before = self.raw_count
        strict_before = self.strict_count
        rewrite_before = self.rewrite_count
        class_delta = cohort_delta = bank_mutations = 0
        for candidate in _operator_candidates(
            snapshot,
            target_depth=depth,
            limits=self.limits,
        ):
            checkpoint = self._event_counter_checkpoint()
            vector_key = self._candidate_vector_key(candidate)
            vector_was_cached = vector_key in self.vector_cache
            try:
                self._consume_raw()
                try:
                    ast = canonicalize_shrink6_source_ast(candidate.source_ast)
                    replay = decode_shrink6_canonical_ast(ast.cbor_bytes)
                except StrictAstError as error:
                    _fail(FAIL_PREFLIGHT_STRICT_BOUNDARY, str(error))
                if replay.cbor_bytes != ast.cbor_bytes or replay.digest != ast.digest:
                    _fail(FAIL_PREFLIGHT_AST_IDENTITY, "source/formal replay differs")
                self.strict_count += 1
                if ast.value[1] != candidate.expected_node:
                    self.rewrite_count += 1
                behavior = self.candidate_vector(candidate)
                deltas = self.insert(ast, behavior, current_depth=depth)
            except _ResourceLimit:
                if not vector_was_cached:
                    self.vector_cache.pop(vector_key, None)
                self._restore_event_counters(checkpoint)
                raise
            class_delta += deltas[0]
            cohort_delta += deltas[1]
            bank_mutations += deltas[2]
        after = self._counts()
        self.barriers.append(
            DepthBarrierRecordV1(
                depth=depth,
                barrier_kind="CONSTRUCTION_DEPTH",
                eligible_raw_application_count=self.raw_count - raw_before,
                strict_admitted_application_count=self.strict_count - strict_before,
                rewrite_collapse_count=self.rewrite_count - rewrite_before,
                new_behavior_class_count=class_delta,
                new_signature_cohort_count=cohort_delta,
                continuation_bank_mutation_count=bank_mutations,
                behavior_class_count_after_barrier=after[0],
                signature_cohort_count_after_barrier=after[1],
                continuation_bank_point_count_after_barrier=after[2],
                visible_frontier_point_count_after_barrier=after[3],
            )
        )
        self.peak_saturation_round_count = max(
            self.peak_saturation_round_count,
            len(self.barriers),
        )

    def close_structural_boundary(self) -> None:
        """Record the zero-delta barrier outside the admitted depth grammar."""

        self._wall_guard()
        if len(self.barriers) + 1 > self.limits.maximum_saturation_rounds:
            raise _ResourceLimit(
                "SATURATION_ROUNDS",
                "the terminal structural barrier would exceed the round limit",
            )
        depth = self.limits.maximum_ast_depth + 1
        counts = self._counts()
        self.barriers.append(
            DepthBarrierRecordV1(
                depth=depth,
                barrier_kind="STRUCTURAL_BOUNDARY",
                eligible_raw_application_count=0,
                strict_admitted_application_count=0,
                rewrite_collapse_count=0,
                new_behavior_class_count=0,
                new_signature_cohort_count=0,
                continuation_bank_mutation_count=0,
                behavior_class_count_after_barrier=counts[0],
                signature_cohort_count_after_barrier=counts[1],
                continuation_bank_point_count_after_barrier=counts[2],
                visible_frontier_point_count_after_barrier=counts[3],
            )
        )
        self.peak_saturation_round_count = max(
            self.peak_saturation_round_count,
            len(self.barriers),
        )


def _result_from_state(
    state: _State,
    *,
    terminal_status: str,
    resource_guard_id: int | None,
    resource_guard_name: str | None,
    traversal_closed: bool,
) -> Q1PartitionCapacityResultV1:
    classes, cohorts, bank, frontier, max_bank, max_frontier = state._counts()
    if (
        state.peak_visible_frontier_point_count < frontier
        or state.peak_visible_frontier_points_per_class < max_frontier
        or state.peak_work_queue_points < len(state.vector_cache)
        or state.peak_work_queue_points < state.current_barrier_raw_count
        or state.peak_saturation_round_count < len(state.barriers)
    ):
        _fail(
            FAIL_PREFLIGHT_HIGH_WATER_INVARIANT,
            "accepted-state high-water fell below its resident guard quantity",
        )
    return Q1PartitionCapacityResultV1(
        schema_version=SCHEMA_VERSION,
        preflight_id=PREFLIGHT_ID,
        dsl_version=DSL_VERSION,
        closure_semantics_version=CLOSURE_SEMANTICS_VERSION,
        input_signature_id=state.universe.input_signature_id,
        universe_row_count=len(state.universe.rows),
        universe_root=state.universe.universe_root,
        limits=state.limits,
        full_v16_structural_limits_applied=(
            state.limits.maximum_ast_depth == 3
            and state.limits.maximum_ast_node_count == 6
        ),
        maximum_ast_depth=state.limits.maximum_ast_depth,
        maximum_ast_node_count=state.limits.maximum_ast_node_count,
        terminal_status=terminal_status,
        resource_guard_id=resource_guard_id,
        resource_guard_name=resource_guard_name,
        traversal_closed=traversal_closed,
        frozen_leaf_count=LEAF_COUNT,
        raw_operator_application_count=state.raw_count,
        strict_admitted_application_count=state.strict_count,
        rewrite_collapse_count=state.rewrite_count,
        behavior_class_count=classes,
        signature_cohort_count=cohorts,
        continuation_bank_point_count=bank,
        visible_frontier_point_count=frontier,
        maximum_bank_points_per_class=max_bank,
        maximum_frontier_points_per_class=max_frontier,
        peak_raw_operator_application_count=state.raw_count,
        peak_behavior_class_count=classes,
        peak_visible_frontier_point_count=(
            state.peak_visible_frontier_point_count
        ),
        peak_visible_frontier_points_per_class=(
            state.peak_visible_frontier_points_per_class
        ),
        peak_continuation_bank_point_count=bank,
        peak_continuation_bank_points_per_class=max_bank,
        peak_work_queue_points=state.peak_work_queue_points,
        peak_saturation_round_count=state.peak_saturation_round_count,
        vector_cache_entry_count=len(state.vector_cache),
        vector_cache_hit_count=state.vector_cache_hits,
        vector_cache_miss_count=state.vector_cache_misses,
        depth_barriers=tuple(state.barriers),
        diagnostic_only=True,
        formal_roots_generated=False,
        formal_roots=None,
        target_truth_accessed=False,
        split_accessed=False,
        role_evaluation_performed=False,
        complete_claim_allowed=False,
    )


def _execute_q1_partition_capacity_preflight_state_v1(
    input_signature_id: int,
    *,
    limits: PreflightLimitsV1 | None = None,
) -> tuple[Q1PartitionCapacityResultV1, _State]:

    active_limits = PreflightLimitsV1() if limits is None else limits
    if type(input_signature_id) is not int or input_signature_id not in (1, 2):
        _fail(
            REJECT_PREFLIGHT_INPUT_SIGNATURE,
            "input_signature_id must be the exact integer 1 or 2",
        )
    if type(active_limits) is not PreflightLimitsV1:
        raise TypeError("limits must be PreflightLimitsV1")
    state = _State(production_universe_v1(input_signature_id), active_limits)
    try:
        state.seed()
        for depth in range(1, active_limits.maximum_ast_depth + 1):
            state.expand_depth(depth)
        state.close_structural_boundary()
    except _ResourceLimit as error:
        return (
            _result_from_state(
                state,
                terminal_status=INCONCLUSIVE_RESOURCE_LIMIT,
                resource_guard_id=error.resource_guard_id,
                resource_guard_name=error.resource_guard_name,
                traversal_closed=False,
            ),
            state,
        )
    full_v16_structural_limits = (
        active_limits.maximum_ast_depth == 3
        and active_limits.maximum_ast_node_count == 6
    )
    return (
        _result_from_state(
            state,
            terminal_status=(
                PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY
                if full_v16_structural_limits
                else LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED
            ),
            resource_guard_id=None,
            resource_guard_name=None,
            traversal_closed=True,
        ),
        state,
    )


def run_q1_partition_capacity_preflight_v1(
    input_signature_id: int,
    *,
    limits: PreflightLimitsV1 | None = None,
) -> Q1PartitionCapacityResultV1:
    """Run one target-blind input-signature preflight to a depth barrier."""

    if type(input_signature_id) is not int or input_signature_id not in (1, 2):
        _fail(
            REJECT_PREFLIGHT_INPUT_SIGNATURE,
            "input_signature_id must be the exact integer 1 or 2",
        )
    active_limits = _require_explicit_nonfull_limits_v1(limits)

    result, _state = _execute_q1_partition_capacity_preflight_state_v1(
        input_signature_id,
        limits=active_limits,
    )
    return result


def _run_q1_partition_snapshot_material_v1(
    input_signature_id: int,
    *,
    limits: PreflightLimitsV1 | None = None,
) -> tuple[
    Q1PartitionCapacityResultV1,
    tuple[_SnapshotClassMaterialV1, ...],
]:
    """Internal immutable export used only by the target-blind Q0.5a layer."""

    result, state = _execute_q1_partition_capacity_preflight_state_v1(
        input_signature_id,
        limits=limits,
    )
    return result, state.immutable_snapshot_material()


def run_q1_capacity_preflight_v1(
    *,
    limits: PreflightLimitsV1 | None = None,
) -> Q1CapacityPreflightResultV1:
    """Run both production signatures without creating a formal Q1 result."""

    active_limits = _require_explicit_nonfull_limits_v1(limits)
    partitions = tuple(
        run_q1_partition_capacity_preflight_v1(
            input_signature_id,
            limits=active_limits,
        )
        for input_signature_id in (1, 2)
    )
    partition_statuses = {item.terminal_status for item in partitions}
    if partition_statuses == {PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY}:
        status = PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY
    elif partition_statuses == {LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED}:
        status = LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED
    else:
        status = PREFLIGHT_CAPACITY_GUARD_HIT
    return Q1CapacityPreflightResultV1(
        schema_version=SCHEMA_VERSION,
        terminal_status=status,
        partitions=partitions,
    )


def _limits_diagnostic_object(limits: PreflightLimitsV1) -> dict[str, int]:
    return {
        "maximum_ast_depth": limits.maximum_ast_depth,
        "maximum_ast_node_count": limits.maximum_ast_node_count,
        "maximum_behavior_classes": limits.maximum_behavior_classes,
        "maximum_continuation_bank_points": (
            limits.maximum_continuation_bank_points
        ),
        "maximum_continuation_bank_points_per_class": (
            limits.maximum_continuation_bank_points_per_class
        ),
        "maximum_raw_operator_applications": (
            limits.maximum_raw_operator_applications
        ),
        "maximum_saturation_rounds": limits.maximum_saturation_rounds,
        "maximum_visible_frontier_points": (
            limits.maximum_visible_frontier_points
        ),
        "maximum_visible_frontier_points_per_class": (
            limits.maximum_visible_frontier_points_per_class
        ),
        "maximum_wall_time_seconds": limits.maximum_wall_time_seconds,
        "maximum_work_queue_points": limits.maximum_work_queue_points,
    }


def _barrier_diagnostic_object(row: DepthBarrierRecordV1) -> dict[str, object]:
    return {
        "barrier_kind": row.barrier_kind,
        "behavior_class_count_after_barrier": (
            row.behavior_class_count_after_barrier
        ),
        "continuation_bank_mutation_count": row.continuation_bank_mutation_count,
        "continuation_bank_point_count_after_barrier": (
            row.continuation_bank_point_count_after_barrier
        ),
        "depth": row.depth,
        "eligible_raw_application_count": row.eligible_raw_application_count,
        "new_behavior_class_count": row.new_behavior_class_count,
        "new_signature_cohort_count": row.new_signature_cohort_count,
        "rewrite_collapse_count": row.rewrite_collapse_count,
        "signature_cohort_count_after_barrier": (
            row.signature_cohort_count_after_barrier
        ),
        "strict_admitted_application_count": (
            row.strict_admitted_application_count
        ),
        "visible_frontier_point_count_after_barrier": (
            row.visible_frontier_point_count_after_barrier
        ),
    }


def _partition_diagnostic_object(
    result: Q1PartitionCapacityResultV1,
) -> dict[str, object]:
    return {
        "closure_semantics_version": result.closure_semantics_version,
        "complete_claim_allowed": result.complete_claim_allowed,
        "continuation_bank_point_count": result.continuation_bank_point_count,
        "depth_barriers": [
            _barrier_diagnostic_object(row) for row in result.depth_barriers
        ],
        "diagnostic_only": result.diagnostic_only,
        "dsl_version": result.dsl_version,
        "formal_roots": result.formal_roots,
        "formal_roots_generated": result.formal_roots_generated,
        "frozen_leaf_count": result.frozen_leaf_count,
        "full_v16_structural_limits_applied": (
            result.full_v16_structural_limits_applied
        ),
        "input_signature_id": result.input_signature_id,
        "limits": _limits_diagnostic_object(result.limits),
        "maximum_ast_depth": result.maximum_ast_depth,
        "maximum_ast_node_count": result.maximum_ast_node_count,
        "maximum_bank_points_per_class": result.maximum_bank_points_per_class,
        "maximum_frontier_points_per_class": (
            result.maximum_frontier_points_per_class
        ),
        "peak_behavior_class_count": result.peak_behavior_class_count,
        "peak_continuation_bank_point_count": (
            result.peak_continuation_bank_point_count
        ),
        "peak_continuation_bank_points_per_class": (
            result.peak_continuation_bank_points_per_class
        ),
        "peak_raw_operator_application_count": (
            result.peak_raw_operator_application_count
        ),
        "peak_saturation_round_count": result.peak_saturation_round_count,
        "peak_visible_frontier_points_per_class": (
            result.peak_visible_frontier_points_per_class
        ),
        "peak_visible_frontier_point_count": (
            result.peak_visible_frontier_point_count
        ),
        "peak_work_queue_points": result.peak_work_queue_points,
        "preflight_id": result.preflight_id,
        "raw_operator_application_count": result.raw_operator_application_count,
        "resource_guard_id": result.resource_guard_id,
        "resource_guard_name": result.resource_guard_name,
        "rewrite_collapse_count": result.rewrite_collapse_count,
        "role_evaluation_performed": result.role_evaluation_performed,
        "schema_version": result.schema_version,
        "signature_cohort_count": result.signature_cohort_count,
        "split_accessed": result.split_accessed,
        "strict_admitted_application_count": (
            result.strict_admitted_application_count
        ),
        "target_truth_accessed": result.target_truth_accessed,
        "terminal_status": result.terminal_status,
        "traversal_closed": result.traversal_closed,
        "universe_root": f"sha256:{result.universe_root.hex()}",
        "universe_row_count": result.universe_row_count,
        "vector_cache_entry_count": result.vector_cache_entry_count,
        "vector_cache_hit_count": result.vector_cache_hit_count,
        "vector_cache_miss_count": result.vector_cache_miss_count,
        "visible_frontier_point_count": result.visible_frontier_point_count,
        "behavior_class_count": result.behavior_class_count,
    }


def capacity_preflight_diagnostic_object_v1(
    result: Q1CapacityPreflightResultV1,
) -> dict[str, object]:
    """Return the stable v1 non-formal diagnostic JSON object.

    This object is deliberately not a formal CBOR object, archive root, Q1
    receipt, gate row, or dual-endpoint agreement.  Canonicalization exists so
    a Python diagnostic can be replayed byte-for-byte during engineering.
    """

    if type(result) is not Q1CapacityPreflightResultV1:
        raise TypeError("result must be Q1CapacityPreflightResultV1")
    return {
        "active_transition_allowed": False,
        "closure_semantics_version": CLOSURE_SEMANTICS_VERSION,
        "complete_claim_allowed": result.complete_claim_allowed,
        "diagnostic_only": result.diagnostic_only,
        "dsl_version": DSL_VERSION,
        "formal_roots": result.formal_roots,
        "formal_roots_generated": result.formal_roots_generated,
        "m3_formal_roots": None,
        "outside_certificate_issued": False,
        "partitions": [
            _partition_diagnostic_object(partition)
            for partition in result.partitions
        ],
        "preflight_id": PREFLIGHT_ID,
        "q1_formal_roots": None,
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_receipt": None,
        "q1_state": "NOT_RUN",
        "q2_state": "NOT_RUN",
        "role_evaluation_performed": result.role_evaluation_performed,
        "schema_version": result.schema_version,
        "split_accessed": result.split_accessed,
        "target_truth_accessed": result.target_truth_accessed,
        "terminal_status": result.terminal_status,
    }


def canonical_capacity_preflight_json_bytes_v1(
    result: Q1CapacityPreflightResultV1,
) -> bytes:
    """Encode the stable diagnostic object as canonical UTF-8 JSON + LF."""

    value = capacity_preflight_diagnostic_object_v1(result)
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


__all__ = [
    "CLOSURE_SEMANTICS_VERSION",
    "DSL_VERSION",
    "DepthBarrierRecordV1",
    "FAIL_PREFLIGHT_AST_IDENTITY",
    "FAIL_PREFLIGHT_BEHAVIOR",
    "FAIL_PREFLIGHT_HIGH_WATER_INVARIANT",
    "FAIL_PREFLIGHT_LATE_LOWER_DEPTH_BANK_MUTATION",
    "FAIL_PREFLIGHT_STRICT_BOUNDARY",
    "INCONCLUSIVE_RESOURCE_LIMIT",
    "LEAF_COUNT",
    "LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED",
    "PREFLIGHT_CAPACITY_GUARD_HIT",
    "PREFLIGHT_CLOSED",
    "PREFLIGHT_ID",
    "PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY",
    "PreflightLimitsV1",
    "Q1CapacityPreflightError",
    "Q1CapacityPreflightResultV1",
    "Q1ImmutableCandidateApplicationV1",
    "Q1PartitionCapacityResultV1",
    "REJECT_FULL_NODE6_PREFLIGHT_NOT_AUTHORIZED",
    "REJECT_PREFLIGHT_INPUT_SIGNATURE",
    "RESOURCE_GUARD_REGISTRY",
    "SCHEMA_VERSION",
    "canonical_capacity_preflight_json_bytes_v1",
    "capacity_preflight_diagnostic_object_v1",
    "immutable_candidate_applications_v1",
    "run_q1_capacity_preflight_v1",
    "run_q1_partition_capacity_preflight_v1",
]
