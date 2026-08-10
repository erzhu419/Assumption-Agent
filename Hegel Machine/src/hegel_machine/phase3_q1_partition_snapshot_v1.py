"""Immutable target-blind Q0.5a snapshots of one Q1 quotient partition.

The snapshot is diagnostic engineering state.  It contains no target truth,
split, role match, formal wire tag, formal root, gate row, receipt, or
certificate authority.  Every representative is retained as its real strict
canonical AST bytes and digest, and validation independently replays its
signature and exact behavior on the bound production universe.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Final, NoReturn

from . import phase3_q0_input_adapter_v1 as _adapter
from . import phase3_q1_capacity_preflight_v1 as _capacity
from .phase3_q0_evaluator_v1 import evaluate_canonical_ast_on_environments_v1
from .phase3_q1_quotient_contract_v1 import (
    FutureAdmissibilitySignatureV1,
    future_signature_from_ast_v1,
    normalization_witness_capacity_v1,
)
from .phase3_q1_universe_v1 import production_universe_v1
from .strict_ast_shrink6_v1 import decode_shrink6_canonical_ast
from .strict_ast_v1 import StrictAstError
from .strict_cbor_v1 import canonical_cbor_encode


SNAPSHOT_SCHEMA_VERSION: Final = "hegel-phase3a-q05a-partition-snapshot/1"
SNAPSHOT_ID: Final = "hegel-phase3a-q05a-target-blind-partition-snapshot-v1"

REJECT_Q1_PARTITION_SNAPSHOT: Final = "REJECT_Q1_PARTITION_SNAPSHOT"
REJECT_Q1_PARTITION_SNAPSHOT_INCOMPLETE: Final = (
    "REJECT_Q1_PARTITION_SNAPSHOT_INCOMPLETE"
)
REJECT_Q1_FULL_NODE6_NOT_AUTHORIZED: Final = (
    "REJECT_Q1_FULL_NODE6_NOT_AUTHORIZED"
)
REJECT_Q1_LOCAL_SNAPSHOT_SCOPE_NOT_AUTHORIZED: Final = (
    "REJECT_Q1_LOCAL_SNAPSHOT_SCOPE_NOT_AUTHORIZED"
)
FAIL_Q1_PARTITION_SNAPSHOT_REPLAY: Final = "FAIL_Q1_PARTITION_SNAPSHOT_REPLAY"
FAIL_SHA256_PREIMAGE_COLLISION: Final = "FAIL_SHA256_PREIMAGE_COLLISION"

_SORT_NAME_BY_ID: Final = {
    1: "Bool",
    2: "Bit",
    3: "Sign",
    4: "BoundedInt",
    5: "RationalValue",
}


class Q1PartitionSnapshotError(ValueError):
    """Stable fail-closed rejection from the Q0.5a snapshot boundary."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Q1PartitionSnapshotError(code, detail)


def _register_ast_digest_preimage_v1(
    seen_preimages: dict[bytes, bytes],
    digest: bytes,
    canonical_ast_cbor: bytes,
) -> None:
    prior = seen_preimages.get(digest)
    if prior is not None and prior != canonical_ast_cbor:
        _fail(
            FAIL_SHA256_PREIMAGE_COLLISION,
            "strict AST digest has different CBOR preimages",
        )
    seen_preimages[digest] = canonical_ast_cbor


def _exact_tuple(value: object, name: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        _fail(REJECT_Q1_PARTITION_SNAPSHOT, f"{name} must be an exact tuple")
    return value


def _exact_uint(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        _fail(REJECT_Q1_PARTITION_SNAPSHOT, f"{name} must be a uint")
    return value


@dataclass(frozen=True, slots=True)
class Q1BehaviorCellSnapshotV1:
    cell_tag: int
    boolean_value: bool | None = None
    integer_value: int | None = None
    rational_numerator: int | None = None
    rational_denominator: int | None = None

    def __post_init__(self) -> None:
        if type(self.cell_tag) is not int or self.cell_tag not in (0, 1):
            _fail(REJECT_Q1_PARTITION_SNAPSHOT, "cell_tag must be 0 or 1")
        if self.boolean_value is not None and type(self.boolean_value) is not bool:
            _fail(REJECT_Q1_PARTITION_SNAPSHOT, "boolean_value must be exact bool")
        for name in ("integer_value", "rational_numerator", "rational_denominator"):
            value = getattr(self, name)
            if value is not None and type(value) is not int:
                _fail(REJECT_Q1_PARTITION_SNAPSHOT, f"{name} must be exact int")
        present = (
            self.boolean_value is not None,
            self.integer_value is not None,
            self.rational_numerator is not None
            or self.rational_denominator is not None,
        )
        if self.cell_tag == 0 and any(present):
            _fail(REJECT_Q1_PARTITION_SNAPSHOT, "bottom cell must carry no value")
        if self.cell_tag == 1 and sum(present) != 1:
            _fail(
                REJECT_Q1_PARTITION_SNAPSHOT,
                "defined cell must carry exactly one typed value",
            )
        if (self.rational_numerator is None) != (self.rational_denominator is None):
            _fail(
                REJECT_Q1_PARTITION_SNAPSHOT,
                "rational numerator and denominator must be present together",
            )
        if self.rational_denominator is not None and self.rational_denominator <= 0:
            _fail(REJECT_Q1_PARTITION_SNAPSHOT, "rational denominator must be positive")

    def runtime_value(self, output_sort_id: int) -> object:
        if type(output_sort_id) is not int or output_sort_id not in _SORT_NAME_BY_ID:
            _fail(REJECT_Q1_PARTITION_SNAPSHOT, "output sort is unregistered")
        if self.cell_tag == 0:
            return _adapter.BOTTOM
        if output_sort_id == 1:
            if type(self.boolean_value) is not bool:
                _fail(REJECT_Q1_PARTITION_SNAPSHOT, "Bool cell has wrong payload")
            return self.boolean_value
        if output_sort_id in (2, 3, 4):
            if type(self.integer_value) is not int:
                _fail(REJECT_Q1_PARTITION_SNAPSHOT, "integer cell has wrong payload")
            if output_sort_id == 2 and self.integer_value not in (0, 1):
                _fail(REJECT_Q1_PARTITION_SNAPSHOT, "Bit cell is outside 0..1")
            if output_sort_id == 3 and self.integer_value not in (-1, 0, 1):
                _fail(REJECT_Q1_PARTITION_SNAPSHOT, "Sign cell is outside -1..1")
            if output_sort_id == 4 and not -8 <= self.integer_value <= 8:
                _fail(REJECT_Q1_PARTITION_SNAPSHOT, "BoundedInt cell is outside grid")
            return self.integer_value
        if (
            type(self.rational_numerator) is not int
            or type(self.rational_denominator) is not int
        ):
            _fail(REJECT_Q1_PARTITION_SNAPSHOT, "rational cell has wrong payload")
        value = Fraction(self.rational_numerator, self.rational_denominator)
        if (
            value.numerator != self.rational_numerator
            or value.denominator != self.rational_denominator
            or value not in _adapter.RATIONAL_VALUE_GRID
        ):
            _fail(REJECT_Q1_PARTITION_SNAPSHOT, "rational cell is not canonical grid value")
        return value


@dataclass(frozen=True, slots=True)
class Q1SnapshotRepresentativeV1:
    canonical_ast_cbor: bytes
    canonical_ast_hash: bytes

    def __post_init__(self) -> None:
        if type(self.canonical_ast_cbor) is not bytes or not self.canonical_ast_cbor:
            _fail(REJECT_Q1_PARTITION_SNAPSHOT, "canonical AST must be non-empty bytes")
        if type(self.canonical_ast_hash) is not bytes or len(self.canonical_ast_hash) != 32:
            _fail(REJECT_Q1_PARTITION_SNAPSHOT, "canonical AST hash must be 32 bytes")


@dataclass(frozen=True, slots=True)
class Q1SnapshotCohortV1:
    signature: FutureAdmissibilitySignatureV1
    representatives: tuple[Q1SnapshotRepresentativeV1, ...]
    visible_frontier_member: bool

    def __post_init__(self) -> None:
        if type(self.signature) is not FutureAdmissibilitySignatureV1:
            _fail(REJECT_Q1_PARTITION_SNAPSHOT, "signature has wrong exact type")
        representatives = _exact_tuple(self.representatives, "representatives")
        if not representatives or any(
            type(item) is not Q1SnapshotRepresentativeV1 for item in representatives
        ):
            _fail(REJECT_Q1_PARTITION_SNAPSHOT, "representatives are malformed")
        ast_bytes = tuple(item.canonical_ast_cbor for item in representatives)
        if ast_bytes != tuple(sorted(ast_bytes)) or len(set(ast_bytes)) != len(ast_bytes):
            _fail(REJECT_Q1_PARTITION_SNAPSHOT, "representatives are not canonical")
        if type(self.visible_frontier_member) is not bool:
            _fail(REJECT_Q1_PARTITION_SNAPSHOT, "frontier flag must be exact bool")

    @property
    def canonical_signature_key(self) -> bytes:
        return canonical_cbor_encode(self.signature.canonical_object())


@dataclass(frozen=True, slots=True)
class Q1BehaviorClassSnapshotV1:
    behavior_key: bytes
    output_sort_id: int
    behavior_cells: tuple[Q1BehaviorCellSnapshotV1, ...]
    cohorts: tuple[Q1SnapshotCohortV1, ...]
    visible_frontier_representative_hashes: tuple[bytes, ...]
    minimum_admitted_mdl_q32: int

    def __post_init__(self) -> None:
        if type(self.behavior_key) is not bytes or not self.behavior_key:
            _fail(REJECT_Q1_PARTITION_SNAPSHOT, "behavior key must be non-empty bytes")
        if type(self.output_sort_id) is not int or self.output_sort_id not in _SORT_NAME_BY_ID:
            _fail(REJECT_Q1_PARTITION_SNAPSHOT, "class output sort is unregistered")
        cells = _exact_tuple(self.behavior_cells, "behavior_cells")
        if not cells or any(type(item) is not Q1BehaviorCellSnapshotV1 for item in cells):
            _fail(REJECT_Q1_PARTITION_SNAPSHOT, "behavior cells are malformed")
        cohorts = _exact_tuple(self.cohorts, "cohorts")
        if not cohorts or any(type(item) is not Q1SnapshotCohortV1 for item in cohorts):
            _fail(REJECT_Q1_PARTITION_SNAPSHOT, "cohorts are malformed")
        signature_keys = tuple(item.canonical_signature_key for item in cohorts)
        if signature_keys != tuple(sorted(signature_keys)) or len(set(signature_keys)) != len(
            signature_keys
        ):
            _fail(REJECT_Q1_PARTITION_SNAPSHOT, "cohorts are not canonical")
        hashes = _exact_tuple(
            self.visible_frontier_representative_hashes,
            "visible_frontier_representative_hashes",
        )
        if any(type(value) is not bytes or len(value) != 32 for value in hashes):
            _fail(REJECT_Q1_PARTITION_SNAPSHOT, "frontier hashes are malformed")
        if hashes != tuple(sorted(hashes)) or len(set(hashes)) != len(hashes):
            _fail(REJECT_Q1_PARTITION_SNAPSHOT, "frontier hashes are not canonical")
        _exact_uint(self.minimum_admitted_mdl_q32, "minimum_admitted_mdl_q32")


@dataclass(frozen=True, slots=True)
class Q1PartitionSnapshotV1:
    schema_version: str
    snapshot_id: str
    preflight_id: str
    dsl_version: str
    closure_semantics_version: str
    input_signature_id: int
    universe_row_count: int
    universe_root: bytes
    limits: _capacity.PreflightLimitsV1
    terminal_status: str
    behavior_classes: tuple[Q1BehaviorClassSnapshotV1, ...]
    depth_barriers: tuple[_capacity.DepthBarrierRecordV1, ...]
    behavior_class_count: int
    signature_cohort_count: int
    continuation_bank_point_count: int
    visible_frontier_point_count: int
    raw_operator_application_count: int
    strict_admitted_application_count: int
    rewrite_collapse_count: int
    maximum_bank_points_per_class: int
    maximum_frontier_points_per_class: int
    vector_cache_entry_count: int
    vector_cache_hit_count: int
    vector_cache_miss_count: int
    peak_raw_operator_application_count: int
    peak_behavior_class_count: int
    peak_visible_frontier_point_count: int
    peak_visible_frontier_points_per_class: int
    peak_continuation_bank_point_count: int
    peak_continuation_bank_points_per_class: int
    peak_work_queue_points: int
    peak_saturation_round_count: int
    diagnostic_only: bool
    q1_state: str
    q1_gate_count: int
    q1_gate_mask: int
    q1_formal_roots: None
    q1_receipt: None
    q2_state: str
    m3_formal_roots: None
    target_truth_accessed: bool
    split_accessed: bool
    role_evaluation_performed: bool
    outside_certificate_issued: bool
    active_transition_allowed: bool

    def __post_init__(self) -> None:
        for name in (
            "schema_version",
            "snapshot_id",
            "preflight_id",
            "dsl_version",
            "closure_semantics_version",
            "terminal_status",
            "q1_state",
            "q2_state",
        ):
            if type(getattr(self, name)) is not str:
                _fail(REJECT_Q1_PARTITION_SNAPSHOT, f"{name} must be exact str")
        if type(self.input_signature_id) is not int or self.input_signature_id not in (1, 2):
            _fail(REJECT_Q1_PARTITION_SNAPSHOT, "input signature must be 1 or 2")
        _exact_uint(self.universe_row_count, "universe_row_count")
        if type(self.universe_root) is not bytes or len(self.universe_root) != 32:
            _fail(REJECT_Q1_PARTITION_SNAPSHOT, "universe root must be 32 bytes")
        if type(self.limits) is not _capacity.PreflightLimitsV1:
            _fail(REJECT_Q1_PARTITION_SNAPSHOT, "limits have wrong exact type")
        classes = _exact_tuple(self.behavior_classes, "behavior_classes")
        if any(type(item) is not Q1BehaviorClassSnapshotV1 for item in classes):
            _fail(REJECT_Q1_PARTITION_SNAPSHOT, "behavior classes are malformed")
        keys = tuple(item.behavior_key for item in classes)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            _fail(REJECT_Q1_PARTITION_SNAPSHOT, "behavior classes are not canonical")
        barriers = _exact_tuple(self.depth_barriers, "depth_barriers")
        if any(type(item) is not _capacity.DepthBarrierRecordV1 for item in barriers):
            _fail(REJECT_Q1_PARTITION_SNAPSHOT, "depth barriers are malformed")
        for name in (
            "behavior_class_count",
            "signature_cohort_count",
            "continuation_bank_point_count",
            "visible_frontier_point_count",
            "raw_operator_application_count",
            "strict_admitted_application_count",
            "rewrite_collapse_count",
            "maximum_bank_points_per_class",
            "maximum_frontier_points_per_class",
            "vector_cache_entry_count",
            "vector_cache_hit_count",
            "vector_cache_miss_count",
            "peak_raw_operator_application_count",
            "peak_behavior_class_count",
            "peak_visible_frontier_point_count",
            "peak_visible_frontier_points_per_class",
            "peak_continuation_bank_point_count",
            "peak_continuation_bank_points_per_class",
            "peak_work_queue_points",
            "peak_saturation_round_count",
            "q1_gate_count",
            "q1_gate_mask",
        ):
            _exact_uint(getattr(self, name), name)
        for name in (
            "diagnostic_only",
            "target_truth_accessed",
            "split_accessed",
            "role_evaluation_performed",
            "outside_certificate_issued",
            "active_transition_allowed",
        ):
            if type(getattr(self, name)) is not bool:
                _fail(REJECT_Q1_PARTITION_SNAPSHOT, f"{name} must be exact bool")


def _cell_from_runtime(value: object) -> Q1BehaviorCellSnapshotV1:
    if value is _adapter.BOTTOM:
        return Q1BehaviorCellSnapshotV1(cell_tag=0)
    if type(value) is bool:
        return Q1BehaviorCellSnapshotV1(cell_tag=1, boolean_value=value)
    if type(value) is int:
        return Q1BehaviorCellSnapshotV1(cell_tag=1, integer_value=value)
    if type(value) is Fraction:
        return Q1BehaviorCellSnapshotV1(
            cell_tag=1,
            rational_numerator=value.numerator,
            rational_denominator=value.denominator,
        )
    _fail(FAIL_Q1_PARTITION_SNAPSHOT_REPLAY, f"unsupported behavior value {value!r}")


def _cohort_visible(cohorts: tuple[Q1SnapshotCohortV1, ...], index: int) -> bool:
    cohort = cohorts[index]
    return not any(
        other.signature.dominates(cohort.signature)
        and len(other.representatives) >= len(cohort.representatives)
        for other_index, other in enumerate(cohorts)
        if other_index != index
    )


def _validate_q1_partition_snapshot_self_consistency_v1(
    snapshot: Q1PartitionSnapshotV1,
) -> None:
    """Replay every immutable class, signature, AST, behavior, and statistic."""

    if type(snapshot) is not Q1PartitionSnapshotV1:
        raise TypeError("snapshot must be Q1PartitionSnapshotV1")
    if (
        snapshot.schema_version != SNAPSHOT_SCHEMA_VERSION
        or snapshot.snapshot_id != SNAPSHOT_ID
        or snapshot.preflight_id != _capacity.PREFLIGHT_ID
        or snapshot.dsl_version != _capacity.DSL_VERSION
        or snapshot.closure_semantics_version != _capacity.CLOSURE_SEMANTICS_VERSION
    ):
        _fail(REJECT_Q1_PARTITION_SNAPSHOT, "snapshot identity differs")
    if snapshot.terminal_status not in (
        _capacity.LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED,
        _capacity.PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY,
    ):
        _fail(REJECT_Q1_PARTITION_SNAPSHOT_INCOMPLETE, "partition is not closed")
    if (
        snapshot.diagnostic_only is not True
        or snapshot.q1_state != "NOT_RUN"
        or snapshot.q1_gate_count != 0
        or snapshot.q1_gate_mask != 0
        or snapshot.q1_formal_roots is not None
        or snapshot.q1_receipt is not None
        or snapshot.q2_state != "NOT_RUN"
        or snapshot.m3_formal_roots is not None
        or snapshot.target_truth_accessed is not False
        or snapshot.split_accessed is not False
        or snapshot.role_evaluation_performed is not False
        or snapshot.outside_certificate_issued is not False
        or snapshot.active_transition_allowed is not False
    ):
        _fail(REJECT_Q1_PARTITION_SNAPSHOT, "downstream authority is not closed")

    universe = production_universe_v1(snapshot.input_signature_id)
    if (
        snapshot.universe_row_count != len(universe.rows)
        or snapshot.universe_root != universe.universe_root
    ):
        _fail(FAIL_Q1_PARTITION_SNAPSHOT_REPLAY, "universe binding differs")
    environments = universe.observation_environments()

    total_cohorts = 0
    total_bank = 0
    total_frontier = 0
    maximum_bank = 0
    maximum_frontier = 0
    seen_ast_bytes: set[bytes] = set()
    seen_hash_preimages: dict[bytes, bytes] = {}
    for class_row in snapshot.behavior_classes:
        if len(class_row.behavior_cells) != snapshot.universe_row_count:
            _fail(FAIL_Q1_PARTITION_SNAPSHOT_REPLAY, "behavior row count differs")
        behavior = tuple(
            cell.runtime_value(class_row.output_sort_id)
            for cell in class_row.behavior_cells
        )
        sort_name = _SORT_NAME_BY_ID[class_row.output_sort_id]
        if _capacity._behavior_key(sort_name, behavior) != class_row.behavior_key:
            _fail(FAIL_Q1_PARTITION_SNAPSHOT_REPLAY, "behavior identity differs")

        frontier_hashes: list[bytes] = []
        class_bank = 0
        for index, cohort in enumerate(class_row.cohorts):
            expected_visible = _cohort_visible(class_row.cohorts, index)
            if cohort.visible_frontier_member is not expected_visible:
                _fail(FAIL_Q1_PARTITION_SNAPSHOT_REPLAY, "frontier membership differs")
            if len(cohort.representatives) > normalization_witness_capacity_v1(
                cohort.signature.output_sort_id
            ):
                _fail(FAIL_Q1_PARTITION_SNAPSHOT_REPLAY, "cohort capacity differs")
            for representative in cohort.representatives:
                try:
                    ast = decode_shrink6_canonical_ast(
                        representative.canonical_ast_cbor
                    )
                except StrictAstError as error:
                    _fail(FAIL_Q1_PARTITION_SNAPSHOT_REPLAY, str(error))
                if (
                    ast.cbor_bytes != representative.canonical_ast_cbor
                    or ast.digest != representative.canonical_ast_hash
                ):
                    _fail(FAIL_Q1_PARTITION_SNAPSHOT_REPLAY, "AST identity differs")
                _register_ast_digest_preimage_v1(
                    seen_hash_preimages,
                    ast.digest,
                    ast.cbor_bytes,
                )
                if ast.cbor_bytes in seen_ast_bytes:
                    _fail(FAIL_Q1_PARTITION_SNAPSHOT_REPLAY, "AST occurs twice in bank")
                seen_ast_bytes.add(ast.cbor_bytes)
                if ast.metrics.output_sort != sort_name:
                    _fail(FAIL_Q1_PARTITION_SNAPSHOT_REPLAY, "AST output sort differs")
                if future_signature_from_ast_v1(ast) != cohort.signature:
                    _fail(FAIL_Q1_PARTITION_SNAPSHOT_REPLAY, "signature replay differs")
                replayed_behavior = evaluate_canonical_ast_on_environments_v1(
                    ast,
                    environments,
                )
                if replayed_behavior != behavior:
                    _fail(FAIL_Q1_PARTITION_SNAPSHOT_REPLAY, "behavior replay differs")
                if expected_visible:
                    frontier_hashes.append(ast.digest)
                class_bank += 1
        expected_hashes = tuple(sorted(frontier_hashes))
        if expected_hashes != class_row.visible_frontier_representative_hashes:
            _fail(FAIL_Q1_PARTITION_SNAPSHOT_REPLAY, "frontier hash set differs")
        expected_minimum_mdl = min(
            cohort.signature.mdl_length_q32 for cohort in class_row.cohorts
        )
        if class_row.minimum_admitted_mdl_q32 != expected_minimum_mdl:
            _fail(FAIL_Q1_PARTITION_SNAPSHOT_REPLAY, "minimum MDL differs")
        class_frontier = len(expected_hashes)
        total_cohorts += len(class_row.cohorts)
        total_bank += class_bank
        total_frontier += class_frontier
        maximum_bank = max(maximum_bank, class_bank)
        maximum_frontier = max(maximum_frontier, class_frontier)

    observed = (
        len(snapshot.behavior_classes),
        total_cohorts,
        total_bank,
        total_frontier,
        maximum_bank,
        maximum_frontier,
    )
    declared = (
        snapshot.behavior_class_count,
        snapshot.signature_cohort_count,
        snapshot.continuation_bank_point_count,
        snapshot.visible_frontier_point_count,
        snapshot.maximum_bank_points_per_class,
        snapshot.maximum_frontier_points_per_class,
    )
    if observed != declared:
        _fail(FAIL_Q1_PARTITION_SNAPSHOT_REPLAY, "snapshot count summary differs")

    barriers = snapshot.depth_barriers
    expected_depths = tuple(range(snapshot.limits.maximum_ast_depth + 2))
    expected_kinds = (
        "LEAF_SEED",
        *("CONSTRUCTION_DEPTH" for _ in range(snapshot.limits.maximum_ast_depth)),
        "STRUCTURAL_BOUNDARY",
    )
    if (
        not barriers
        or tuple(row.depth for row in barriers) != expected_depths
        or tuple(row.barrier_kind for row in barriers) != expected_kinds
        or barriers[-1].behavior_class_count_after_barrier
        != snapshot.behavior_class_count
        or barriers[-1].signature_cohort_count_after_barrier
        != snapshot.signature_cohort_count
        or barriers[-1].continuation_bank_point_count_after_barrier
        != snapshot.continuation_bank_point_count
        or barriers[-1].visible_frontier_point_count_after_barrier
        != snapshot.visible_frontier_point_count
    ):
        _fail(FAIL_Q1_PARTITION_SNAPSHOT_REPLAY, "depth barrier replay differs")
    previous_classes = previous_cohorts = previous_bank = 0
    for barrier in barriers:
        if type(barrier.barrier_kind) is not str:
            _fail(FAIL_Q1_PARTITION_SNAPSHOT_REPLAY, "barrier kind is not exact str")
        for name in (
            "depth",
            "eligible_raw_application_count",
            "strict_admitted_application_count",
            "rewrite_collapse_count",
            "new_behavior_class_count",
            "new_signature_cohort_count",
            "continuation_bank_mutation_count",
            "behavior_class_count_after_barrier",
            "signature_cohort_count_after_barrier",
            "continuation_bank_point_count_after_barrier",
            "visible_frontier_point_count_after_barrier",
        ):
            value = getattr(barrier, name)
            if type(value) is not int or value < 0:
                _fail(
                    FAIL_Q1_PARTITION_SNAPSHOT_REPLAY,
                    f"barrier field {name} is not a uint",
                )
        if (
            barrier.strict_admitted_application_count
            != barrier.eligible_raw_application_count
            or barrier.rewrite_collapse_count
            > barrier.strict_admitted_application_count
            or barrier.new_behavior_class_count
            != barrier.behavior_class_count_after_barrier - previous_classes
            or barrier.new_signature_cohort_count
            != barrier.signature_cohort_count_after_barrier - previous_cohorts
            or barrier.continuation_bank_point_count_after_barrier < previous_bank
            or barrier.continuation_bank_mutation_count
            < barrier.continuation_bank_point_count_after_barrier - previous_bank
            or barrier.visible_frontier_point_count_after_barrier
            > barrier.continuation_bank_point_count_after_barrier
        ):
            _fail(FAIL_Q1_PARTITION_SNAPSHOT_REPLAY, "barrier transition differs")
        previous_classes = barrier.behavior_class_count_after_barrier
        previous_cohorts = barrier.signature_cohort_count_after_barrier
        previous_bank = barrier.continuation_bank_point_count_after_barrier
    if (
        barriers[0].eligible_raw_application_count != _capacity.LEAF_COUNT
        or barriers[0].rewrite_collapse_count != 0
        or any(
            getattr(barriers[-1], name) != 0
            for name in (
                "eligible_raw_application_count",
                "strict_admitted_application_count",
                "rewrite_collapse_count",
                "new_behavior_class_count",
                "new_signature_cohort_count",
                "continuation_bank_mutation_count",
            )
        )
    ):
        _fail(FAIL_Q1_PARTITION_SNAPSHOT_REPLAY, "boundary barrier differs")
    if sum(row.eligible_raw_application_count for row in barriers) != (
        snapshot.raw_operator_application_count
    ):
        _fail(FAIL_Q1_PARTITION_SNAPSHOT_REPLAY, "raw barrier count differs")
    if sum(row.strict_admitted_application_count for row in barriers) != (
        snapshot.strict_admitted_application_count
    ):
        _fail(FAIL_Q1_PARTITION_SNAPSHOT_REPLAY, "strict barrier count differs")
    if sum(row.rewrite_collapse_count for row in barriers) != snapshot.rewrite_collapse_count:
        _fail(FAIL_Q1_PARTITION_SNAPSHOT_REPLAY, "rewrite barrier count differs")

    if (
        snapshot.peak_raw_operator_application_count
        != snapshot.raw_operator_application_count
        or snapshot.peak_behavior_class_count != snapshot.behavior_class_count
        or snapshot.peak_visible_frontier_point_count
        < snapshot.visible_frontier_point_count
        or snapshot.peak_visible_frontier_points_per_class
        < snapshot.maximum_frontier_points_per_class
        or snapshot.peak_continuation_bank_point_count
        != snapshot.continuation_bank_point_count
        or snapshot.peak_continuation_bank_points_per_class
        != snapshot.maximum_bank_points_per_class
        or snapshot.peak_work_queue_points < snapshot.vector_cache_entry_count
        or snapshot.peak_work_queue_points
        < max(row.eligible_raw_application_count for row in barriers)
        or snapshot.peak_saturation_round_count != len(barriers)
        or snapshot.vector_cache_entry_count != snapshot.vector_cache_miss_count
    ):
        _fail(FAIL_Q1_PARTITION_SNAPSHOT_REPLAY, "high-water summary differs")


def _build_q1_partition_snapshot_unvalidated_v1(
    input_signature_id: int,
    *,
    limits: _capacity.PreflightLimitsV1 | None = None,
) -> Q1PartitionSnapshotV1:
    """Run one target-blind partition and export its immutable final state."""

    result, material = _capacity._run_q1_partition_snapshot_material_v1(
        input_signature_id,
        limits=limits,
    )
    if not result.traversal_closed or result.resource_guard_id is not None:
        _fail(
            REJECT_Q1_PARTITION_SNAPSHOT_INCOMPLETE,
            "resource-limited or incomplete traversal has no final snapshot",
        )

    classes: list[Q1BehaviorClassSnapshotV1] = []
    for class_material in material:
        cohorts = tuple(
            Q1SnapshotCohortV1(
                signature=cohort.signature,
                representatives=tuple(
                    Q1SnapshotRepresentativeV1(
                        canonical_ast_cbor=representative.canonical_ast_cbor,
                        canonical_ast_hash=representative.canonical_ast_hash,
                    )
                    for representative in cohort.representatives
                ),
                visible_frontier_member=cohort.visible_frontier_member,
            )
            for cohort in class_material.cohorts
        )
        first_ast = decode_shrink6_canonical_ast(
            cohorts[0].representatives[0].canonical_ast_cbor
        )
        output_sort_id = _capacity.OUTPUT_SORT_IDS[first_ast.metrics.output_sort]
        frontier_hashes = tuple(
            sorted(
                representative.canonical_ast_hash
                for cohort in cohorts
                if cohort.visible_frontier_member
                for representative in cohort.representatives
            )
        )
        classes.append(
            Q1BehaviorClassSnapshotV1(
                behavior_key=class_material.behavior_key,
                output_sort_id=output_sort_id,
                behavior_cells=tuple(
                    _cell_from_runtime(value) for value in class_material.behavior
                ),
                cohorts=cohorts,
                visible_frontier_representative_hashes=frontier_hashes,
                minimum_admitted_mdl_q32=min(
                    cohort.signature.mdl_length_q32 for cohort in cohorts
                ),
            )
        )

    snapshot = Q1PartitionSnapshotV1(
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        snapshot_id=SNAPSHOT_ID,
        preflight_id=result.preflight_id,
        dsl_version=result.dsl_version,
        closure_semantics_version=result.closure_semantics_version,
        input_signature_id=result.input_signature_id,
        universe_row_count=result.universe_row_count,
        universe_root=result.universe_root,
        limits=result.limits,
        terminal_status=result.terminal_status,
        behavior_classes=tuple(classes),
        depth_barriers=result.depth_barriers,
        behavior_class_count=result.behavior_class_count,
        signature_cohort_count=result.signature_cohort_count,
        continuation_bank_point_count=result.continuation_bank_point_count,
        visible_frontier_point_count=result.visible_frontier_point_count,
        raw_operator_application_count=result.raw_operator_application_count,
        strict_admitted_application_count=result.strict_admitted_application_count,
        rewrite_collapse_count=result.rewrite_collapse_count,
        maximum_bank_points_per_class=result.maximum_bank_points_per_class,
        maximum_frontier_points_per_class=result.maximum_frontier_points_per_class,
        vector_cache_entry_count=result.vector_cache_entry_count,
        vector_cache_hit_count=result.vector_cache_hit_count,
        vector_cache_miss_count=result.vector_cache_miss_count,
        peak_raw_operator_application_count=(
            result.peak_raw_operator_application_count
        ),
        peak_behavior_class_count=result.peak_behavior_class_count,
        peak_visible_frontier_point_count=(
            result.peak_visible_frontier_point_count
        ),
        peak_visible_frontier_points_per_class=(
            result.peak_visible_frontier_points_per_class
        ),
        peak_continuation_bank_point_count=(
            result.peak_continuation_bank_point_count
        ),
        peak_continuation_bank_points_per_class=(
            result.peak_continuation_bank_points_per_class
        ),
        peak_work_queue_points=result.peak_work_queue_points,
        peak_saturation_round_count=result.peak_saturation_round_count,
        diagnostic_only=True,
        q1_state="NOT_RUN",
        q1_gate_count=0,
        q1_gate_mask=0,
        q1_formal_roots=None,
        q1_receipt=None,
        q2_state="NOT_RUN",
        m3_formal_roots=None,
        target_truth_accessed=False,
        split_accessed=False,
        role_evaluation_performed=False,
        outside_certificate_issued=False,
        active_transition_allowed=False,
    )
    _validate_q1_partition_snapshot_self_consistency_v1(snapshot)
    return snapshot


def build_q1_partition_snapshot_v1(
    input_signature_id: int,
    *,
    limits: _capacity.PreflightLimitsV1 | None = None,
) -> Q1PartitionSnapshotV1:
    """Export an explicitly bounded local snapshot; full node-6 remains gated."""

    if type(limits) is not _capacity.PreflightLimitsV1:
        _fail(
            REJECT_Q1_LOCAL_SNAPSHOT_SCOPE_NOT_AUTHORIZED,
            "an explicit bounded local-prototype limit is required",
        )
    if limits.maximum_ast_node_count > 3:
        _fail(
            REJECT_Q1_LOCAL_SNAPSHOT_SCOPE_NOT_AUTHORIZED,
            "the materialized local snapshot prototype is capped at three nodes",
        )

    return _build_q1_partition_snapshot_unvalidated_v1(
        input_signature_id,
        limits=limits,
    )


def validate_q1_partition_snapshot_v1(snapshot: Q1PartitionSnapshotV1) -> None:
    """Replay self-consistency and require exact equality with a fresh engine run."""

    if type(snapshot) is not Q1PartitionSnapshotV1:
        raise TypeError("snapshot must be Q1PartitionSnapshotV1")
    if snapshot.limits.maximum_ast_node_count > 3:
        _fail(
            REJECT_Q1_LOCAL_SNAPSHOT_SCOPE_NOT_AUTHORIZED,
            "untrusted replay exceeds the bounded three-node prototype scope",
        )
    _validate_q1_partition_snapshot_self_consistency_v1(snapshot)
    expected = _build_q1_partition_snapshot_unvalidated_v1(
        snapshot.input_signature_id,
        limits=snapshot.limits,
    )
    if snapshot != expected:
        _fail(
            FAIL_Q1_PARTITION_SNAPSHOT_REPLAY,
            "snapshot differs from the complete deterministic engine export",
        )


__all__ = [
    "FAIL_Q1_PARTITION_SNAPSHOT_REPLAY",
    "FAIL_SHA256_PREIMAGE_COLLISION",
    "Q1BehaviorCellSnapshotV1",
    "Q1BehaviorClassSnapshotV1",
    "Q1PartitionSnapshotError",
    "Q1PartitionSnapshotV1",
    "Q1SnapshotCohortV1",
    "Q1SnapshotRepresentativeV1",
    "REJECT_Q1_FULL_NODE6_NOT_AUTHORIZED",
    "REJECT_Q1_LOCAL_SNAPSHOT_SCOPE_NOT_AUTHORIZED",
    "REJECT_Q1_PARTITION_SNAPSHOT",
    "REJECT_Q1_PARTITION_SNAPSHOT_INCOMPLETE",
    "SNAPSHOT_ID",
    "SNAPSHOT_SCHEMA_VERSION",
    "build_q1_partition_snapshot_v1",
    "validate_q1_partition_snapshot_v1",
]
