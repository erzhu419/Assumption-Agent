"""Admission-gated full depth-3/node-6 capacity diagnostic.

This module is the additive successor to the bounded Q0.5b projection
qualification.  It deliberately reuses the already-qualified v1 quotient
semantics without changing their public fail-closed entrypoints.  The only
new authority is an exact, source-bound admission object supplied by the
outer one-shot campaign.

The result is diagnostic only.  It cannot open target truth, split or role
material, cannot populate a Q1 output slot, and cannot start Q1.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Final, NoReturn

from . import phase3_q1_capacity_preflight_v1 as _capacity
from . import phase3_q1_partition_snapshot_v1 as _snapshot


SCHEMA_VERSION: Final = "hegel-phase3a-q1-full-node6-capacity-preflight/1"
ADMISSION_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q1-full-node6-capacity-admission/1"
)
ACTION_ID: Final = "full-node6-capacity-preflight-v1"
TERMINAL_SATURATED: Final = _capacity.PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY
TERMINAL_GUARD_HIT: Final = _capacity.PREFLIGHT_CAPACITY_GUARD_HIT
Q1_STATE: Final = "NOT_RUN"
Q1_GATE_COUNT: Final = 0
Q1_GATE_MASK: Final = 0
Q1_OUTPUT_SLOT_COUNT: Final = 8

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class FullNode6CapacityPreflightError(RuntimeError):
    """Stable error raised before or during the diagnostic successor."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise FullNode6CapacityPreflightError(code, detail)


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _fail("REJECT_FULL_NODE6_ADMISSION", f"{label} must be sha256:<64hex>")
    return value


@dataclass(frozen=True, slots=True)
class FullNode6AdmissionV1:
    """Exact execution authority projected from one validated capsule.

    This object is intentionally insufficient on its own to consume owner
    authorization.  The outer campaign owns lookup, ledger consumption and
    process creation; this object only prevents an ungated library call from
    reaching the full traversal.
    """

    schema_version: str
    action_id: str
    capsule_digest: str
    source_commit: str
    predecessor_capsule_digest: str
    predecessor_terminal_receipt_digest: str
    predecessor_artifact_sha256: str
    projection_profile_root: str
    diagnostic_only: bool
    q1_state: str
    q1_gate_count: int
    q1_gate_mask: int
    q1_output_slots: tuple[None, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != ADMISSION_SCHEMA_VERSION
            or self.action_id != ACTION_ID
            or type(self.source_commit) is not str
            or _COMMIT.fullmatch(self.source_commit) is None
            or self.diagnostic_only is not True
            or self.q1_state != Q1_STATE
            or type(self.q1_gate_count) is not int
            or self.q1_gate_count != Q1_GATE_COUNT
            or type(self.q1_gate_mask) is not int
            or self.q1_gate_mask != Q1_GATE_MASK
            or type(self.q1_output_slots) is not tuple
            or self.q1_output_slots != (None,) * Q1_OUTPUT_SLOT_COUNT
        ):
            _fail("REJECT_FULL_NODE6_ADMISSION", "admission authority fields differ")
        _digest(self.capsule_digest, "capsule_digest")
        _digest(self.predecessor_capsule_digest, "predecessor_capsule_digest")
        _digest(
            self.predecessor_terminal_receipt_digest,
            "predecessor_terminal_receipt_digest",
        )
        _digest(self.predecessor_artifact_sha256, "predecessor_artifact_sha256")
        _digest(self.projection_profile_root, "projection_profile_root")


@dataclass(frozen=True, slots=True)
class FullNode6PartitionDiagnosticV1:
    input_signature_id: int
    universe_row_count: int
    universe_root: str
    maximum_ast_depth: int
    maximum_ast_node_count: int
    full_v16_structural_limits_applied: bool
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
    depth_barriers: tuple[tuple[object, ...], ...]

    def __post_init__(self) -> None:
        if (
            type(self.input_signature_id) is not int
            or self.input_signature_id not in (1, 2)
            or type(self.universe_row_count) is not int
            or self.universe_row_count != (480 if self.input_signature_id == 1 else 85)
            or _SHA256.fullmatch(self.universe_root) is None
            or self.maximum_ast_depth != 3
            or self.maximum_ast_node_count != 6
            or self.full_v16_structural_limits_applied is not True
            or self.terminal_status not in (TERMINAL_SATURATED, TERMINAL_GUARD_HIT)
            or type(self.traversal_closed) is not bool
            or type(self.depth_barriers) is not tuple
        ):
            _fail("REJECT_FULL_NODE6_RESULT", "partition header differs")
        numeric_fields = (
            "frozen_leaf_count",
            "raw_operator_application_count",
            "strict_admitted_application_count",
            "rewrite_collapse_count",
            "behavior_class_count",
            "signature_cohort_count",
            "continuation_bank_point_count",
            "visible_frontier_point_count",
            "maximum_bank_points_per_class",
            "maximum_frontier_points_per_class",
            "peak_raw_operator_application_count",
            "peak_behavior_class_count",
            "peak_visible_frontier_point_count",
            "peak_visible_frontier_points_per_class",
            "peak_continuation_bank_point_count",
            "peak_continuation_bank_points_per_class",
            "peak_work_queue_points",
            "peak_saturation_round_count",
            "vector_cache_entry_count",
            "vector_cache_hit_count",
            "vector_cache_miss_count",
        )
        if any(
            type(getattr(self, name)) is not int or getattr(self, name) < 0
            for name in numeric_fields
        ):
            _fail("REJECT_FULL_NODE6_RESULT", "partition count is not uint")
        if self.frozen_leaf_count != _capacity.LEAF_COUNT:
            _fail("REJECT_FULL_NODE6_RESULT", "frozen leaf count differs")
        if self.terminal_status == TERMINAL_SATURATED:
            if (
                not self.traversal_closed
                or self.resource_guard_id is not None
                or self.resource_guard_name is not None
            ):
                _fail("REJECT_FULL_NODE6_RESULT", "saturated terminal differs")
        elif (
            self.traversal_closed
            or type(self.resource_guard_id) is not int
            or type(self.resource_guard_name) is not str
        ):
            _fail("REJECT_FULL_NODE6_RESULT", "guard-hit terminal differs")
        if (
            self.strict_admitted_application_count
            > self.raw_operator_application_count
            or self.rewrite_collapse_count
            > self.strict_admitted_application_count
            or self.behavior_class_count > self.signature_cohort_count
            or self.signature_cohort_count > self.continuation_bank_point_count
            or self.visible_frontier_point_count
            > self.continuation_bank_point_count
            or self.peak_raw_operator_application_count
            < self.raw_operator_application_count
            or self.peak_behavior_class_count < self.behavior_class_count
            or self.peak_visible_frontier_point_count
            < self.visible_frontier_point_count
            or self.peak_visible_frontier_points_per_class
            < self.maximum_frontier_points_per_class
            or self.peak_continuation_bank_point_count
            < self.continuation_bank_point_count
            or self.peak_continuation_bank_points_per_class
            < self.maximum_bank_points_per_class
            or self.vector_cache_entry_count != self.vector_cache_miss_count
        ):
            _fail("REJECT_FULL_NODE6_RESULT", "partition count relation differs")

    @classmethod
    def from_snapshot(
        cls,
        value: _snapshot.Q1PartitionSnapshotV1,
    ) -> "FullNode6PartitionDiagnosticV1":
        if type(value) is not _snapshot.Q1PartitionSnapshotV1:
            raise TypeError("value must be Q1PartitionSnapshotV1")
        if (
            value.limits.maximum_ast_depth != 3
            or value.limits.maximum_ast_node_count != 6
        ):
            _fail("REJECT_FULL_NODE6_RESULT", "partition limits are not depth3/node6")
        return cls(
            input_signature_id=value.input_signature_id,
            universe_row_count=value.universe_row_count,
            universe_root="sha256:" + value.universe_root.hex(),
            maximum_ast_depth=value.limits.maximum_ast_depth,
            maximum_ast_node_count=value.limits.maximum_ast_node_count,
            full_v16_structural_limits_applied=True,
            terminal_status=value.terminal_status,
            resource_guard_id=None,
            resource_guard_name=None,
            traversal_closed=True,
            frozen_leaf_count=_capacity.LEAF_COUNT,
            raw_operator_application_count=value.raw_operator_application_count,
            strict_admitted_application_count=(
                value.strict_admitted_application_count
            ),
            rewrite_collapse_count=value.rewrite_collapse_count,
            behavior_class_count=value.behavior_class_count,
            signature_cohort_count=value.signature_cohort_count,
            continuation_bank_point_count=value.continuation_bank_point_count,
            visible_frontier_point_count=value.visible_frontier_point_count,
            maximum_bank_points_per_class=value.maximum_bank_points_per_class,
            maximum_frontier_points_per_class=(
                value.maximum_frontier_points_per_class
            ),
            peak_raw_operator_application_count=(
                value.peak_raw_operator_application_count
            ),
            peak_behavior_class_count=value.peak_behavior_class_count,
            peak_visible_frontier_point_count=(
                value.peak_visible_frontier_point_count
            ),
            peak_visible_frontier_points_per_class=(
                value.peak_visible_frontier_points_per_class
            ),
            peak_continuation_bank_point_count=(
                value.peak_continuation_bank_point_count
            ),
            peak_continuation_bank_points_per_class=(
                value.peak_continuation_bank_points_per_class
            ),
            peak_work_queue_points=value.peak_work_queue_points,
            peak_saturation_round_count=value.peak_saturation_round_count,
            vector_cache_entry_count=value.vector_cache_entry_count,
            vector_cache_hit_count=value.vector_cache_hit_count,
            vector_cache_miss_count=value.vector_cache_miss_count,
            depth_barriers=tuple(
                (
                    row.depth,
                    row.barrier_kind,
                    row.eligible_raw_application_count,
                    row.strict_admitted_application_count,
                    row.rewrite_collapse_count,
                    row.new_behavior_class_count,
                    row.new_signature_cohort_count,
                    row.continuation_bank_mutation_count,
                    row.behavior_class_count_after_barrier,
                    row.signature_cohort_count_after_barrier,
                    row.continuation_bank_point_count_after_barrier,
                    row.visible_frontier_point_count_after_barrier,
                )
                for row in value.depth_barriers
            ),
        )

    @classmethod
    def from_capacity_result(
        cls,
        value: _capacity.Q1PartitionCapacityResultV1,
    ) -> "FullNode6PartitionDiagnosticV1":
        """Build the terminal guard-hit row, for which no snapshot exists."""

        if type(value) is not _capacity.Q1PartitionCapacityResultV1:
            raise TypeError("value must be Q1PartitionCapacityResultV1")
        if (
            value.limits.maximum_ast_depth != 3
            or value.limits.maximum_ast_node_count != 6
        ):
            _fail("REJECT_FULL_NODE6_RESULT", "partition limits are not depth3/node6")
        return cls(
            input_signature_id=value.input_signature_id,
            universe_row_count=value.universe_row_count,
            universe_root="sha256:" + value.universe_root.hex(),
            maximum_ast_depth=value.maximum_ast_depth,
            maximum_ast_node_count=value.maximum_ast_node_count,
            full_v16_structural_limits_applied=(
                value.full_v16_structural_limits_applied
            ),
            terminal_status=value.terminal_status,
            resource_guard_id=value.resource_guard_id,
            resource_guard_name=value.resource_guard_name,
            traversal_closed=value.traversal_closed,
            frozen_leaf_count=value.frozen_leaf_count,
            raw_operator_application_count=value.raw_operator_application_count,
            strict_admitted_application_count=value.strict_admitted_application_count,
            rewrite_collapse_count=value.rewrite_collapse_count,
            behavior_class_count=value.behavior_class_count,
            signature_cohort_count=value.signature_cohort_count,
            continuation_bank_point_count=value.continuation_bank_point_count,
            visible_frontier_point_count=value.visible_frontier_point_count,
            maximum_bank_points_per_class=value.maximum_bank_points_per_class,
            maximum_frontier_points_per_class=(
                value.maximum_frontier_points_per_class
            ),
            peak_raw_operator_application_count=(
                value.peak_raw_operator_application_count
            ),
            peak_behavior_class_count=value.peak_behavior_class_count,
            peak_visible_frontier_point_count=(
                value.peak_visible_frontier_point_count
            ),
            peak_visible_frontier_points_per_class=(
                value.peak_visible_frontier_points_per_class
            ),
            peak_continuation_bank_point_count=(
                value.peak_continuation_bank_point_count
            ),
            peak_continuation_bank_points_per_class=(
                value.peak_continuation_bank_points_per_class
            ),
            peak_work_queue_points=value.peak_work_queue_points,
            peak_saturation_round_count=value.peak_saturation_round_count,
            vector_cache_entry_count=value.vector_cache_entry_count,
            vector_cache_hit_count=value.vector_cache_hit_count,
            vector_cache_miss_count=value.vector_cache_miss_count,
            depth_barriers=tuple(
                (
                    row.depth,
                    row.barrier_kind,
                    row.eligible_raw_application_count,
                    row.strict_admitted_application_count,
                    row.rewrite_collapse_count,
                    row.new_behavior_class_count,
                    row.new_signature_cohort_count,
                    row.continuation_bank_mutation_count,
                    row.behavior_class_count_after_barrier,
                    row.signature_cohort_count_after_barrier,
                    row.continuation_bank_point_count_after_barrier,
                    row.visible_frontier_point_count_after_barrier,
                )
                for row in value.depth_barriers
            ),
        )


@dataclass(frozen=True, slots=True)
class FullNode6CapacityDiagnosticV1:
    schema_version: str
    action_id: str
    capsule_digest: str
    source_commit: str
    terminal_status: str
    partitions: tuple[FullNode6PartitionDiagnosticV1, ...]
    diagnostic_only: bool = True
    q1_state: str = Q1_STATE
    q1_gate_count: int = Q1_GATE_COUNT
    q1_gate_mask: int = Q1_GATE_MASK
    q1_formal_roots: None = None
    q1_output_slots: tuple[None, ...] = (None,) * Q1_OUTPUT_SLOT_COUNT
    q1_receipt: None = None
    q2_state: str = "NOT_RUN"
    target_truth_accessed: bool = False
    split_accessed: bool = False
    role_evaluation_performed: bool = False
    m3_formal_roots: None = None
    outside_certificate_issued: bool = False
    active_transition_allowed: bool = False

    def __post_init__(self) -> None:
        signature_prefix = tuple(
            row.input_signature_id for row in self.partitions
        )
        if (
            self.schema_version != SCHEMA_VERSION
            or self.action_id != ACTION_ID
            or type(self.partitions) is not tuple
            or signature_prefix not in ((1,), (1, 2))
            or self.terminal_status not in (TERMINAL_SATURATED, TERMINAL_GUARD_HIT)
            or self.diagnostic_only is not True
            or self.q1_state != Q1_STATE
            or self.q1_gate_count != 0
            or self.q1_gate_mask != 0
            or self.q1_formal_roots is not None
            or self.q1_output_slots != (None,) * Q1_OUTPUT_SLOT_COUNT
            or self.q1_receipt is not None
            or self.q2_state != "NOT_RUN"
            or self.target_truth_accessed
            or self.split_accessed
            or self.role_evaluation_performed
            or self.m3_formal_roots is not None
            or self.outside_certificate_issued
            or self.active_transition_allowed
        ):
            _fail("REJECT_FULL_NODE6_RESULT", "diagnostic authority boundary differs")
        _digest(self.capsule_digest, "capsule_digest")
        if _COMMIT.fullmatch(self.source_commit) is None:
            _fail("REJECT_FULL_NODE6_RESULT", "source_commit differs")
        statuses = tuple(row.terminal_status for row in self.partitions)
        if self.terminal_status == TERMINAL_SATURATED:
            valid_terminal = (
                signature_prefix == (1, 2)
                and statuses == (TERMINAL_SATURATED, TERMINAL_SATURATED)
            )
        else:
            valid_terminal = statuses in (
                (TERMINAL_GUARD_HIT,),
                (TERMINAL_SATURATED, TERMINAL_GUARD_HIT),
            )
        if not valid_terminal:
            _fail("REJECT_FULL_NODE6_RESULT", "aggregate terminal status differs")


def _run_partition_once_v1(
    input_signature_id: int,
    limits: _capacity.PreflightLimitsV1,
) -> tuple[
    FullNode6PartitionDiagnosticV1,
    _snapshot.Q1PartitionSnapshotV1 | None,
]:
    """Execute exactly one capacity traversal for one signature."""

    result, state = _capacity._execute_q1_partition_capacity_preflight_state_v1(
        input_signature_id,
        limits=limits,
    )
    if not result.traversal_closed:
        return FullNode6PartitionDiagnosticV1.from_capacity_result(result), None
    # Transfer the final state into the immutable snapshot one class at a
    # time.  The full-node6 route must not retain the engine state, a complete
    # intermediate material copy and a complete snapshot simultaneously.
    # This is ownership transfer from the one traversal, not a second replay.
    snapshot = _snapshot_from_result_and_state_transfer_v1(result, state)
    return FullNode6PartitionDiagnosticV1.from_snapshot(snapshot), snapshot


def _snapshot_from_result_and_state_transfer_v1(
    result: _capacity.Q1PartitionCapacityResultV1,
    state: _capacity._State,
) -> _snapshot.Q1PartitionSnapshotV1:
    """Consume one completed engine state into an exact immutable snapshot."""

    if type(state) is not _capacity._State:
        raise TypeError("state must be the exact capacity engine state")
    if len(state.classes) != result.behavior_class_count:
        _fail("REJECT_FULL_NODE6_RESULT", "same-run state class count differs")

    # These construction caches are already represented by immutable counters
    # in ``result`` and are not inputs to any projection stream.
    state.vector_cache.clear()
    state.barriers.clear()

    classes: list[_snapshot.Q1BehaviorClassSnapshotV1] = []
    for behavior_key in tuple(sorted(state.classes)):
        item = state.classes.pop(behavior_key)
        cohort_rows = tuple(sorted(item.cohorts.items(), key=lambda row: row[0]))
        cohort_material = tuple(cohort for _, cohort in cohort_rows)
        cohorts: list[_snapshot.Q1SnapshotCohortV1] = []
        for index, (_signature_key, cohort) in enumerate(cohort_rows):
            representatives = tuple(
                _snapshot.Q1SnapshotRepresentativeV1(
                    canonical_ast_cbor=program.ast.cbor_bytes,
                    canonical_ast_hash=program.ast.digest,
                )
                for program in sorted(
                    cohort.representatives.values(),
                    key=lambda value: value.ast.cbor_bytes,
                )
            )
            cohorts.append(
                _snapshot.Q1SnapshotCohortV1(
                    signature=cohort.signature,
                    representatives=representatives,
                    visible_frontier_member=_capacity._State._cohort_is_visible(
                        cohort_material,
                        index,
                    ),
                )
            )
        exact_cohorts = tuple(cohorts)
        first_ast = _snapshot.decode_shrink6_canonical_ast(
            exact_cohorts[0].representatives[0].canonical_ast_cbor
        )
        frontier_hashes = tuple(
            sorted(
                representative.canonical_ast_hash
                for cohort in exact_cohorts
                if cohort.visible_frontier_member
                for representative in cohort.representatives
            )
        )
        classes.append(
            _snapshot.Q1BehaviorClassSnapshotV1(
                behavior_key=behavior_key,
                output_sort_id=_capacity.OUTPUT_SORT_IDS[
                    first_ast.metrics.output_sort
                ],
                behavior_cells=tuple(
                    _snapshot._cell_from_runtime(value) for value in item.behavior
                ),
                cohorts=exact_cohorts,
                visible_frontier_representative_hashes=frontier_hashes,
                minimum_admitted_mdl_q32=min(
                    cohort.signature.mdl_length_q32 for cohort in exact_cohorts
                ),
            )
        )
        item.cohorts.clear()
    return _snapshot_from_result_and_classes_v1(result, tuple(classes))


def _snapshot_from_result_and_material_v1(
    result: _capacity.Q1PartitionCapacityResultV1,
    material: tuple[_capacity._SnapshotClassMaterialV1, ...],
) -> _snapshot.Q1PartitionSnapshotV1:
    """Freeze the state returned by the same traversal without rerunning it."""

    classes: list[_snapshot.Q1BehaviorClassSnapshotV1] = []
    for class_material in material:
        cohorts = tuple(
            _snapshot.Q1SnapshotCohortV1(
                signature=cohort.signature,
                representatives=tuple(
                    _snapshot.Q1SnapshotRepresentativeV1(
                        canonical_ast_cbor=representative.canonical_ast_cbor,
                        canonical_ast_hash=representative.canonical_ast_hash,
                    )
                    for representative in cohort.representatives
                ),
                visible_frontier_member=cohort.visible_frontier_member,
            )
            for cohort in class_material.cohorts
        )
        first_ast = _snapshot.decode_shrink6_canonical_ast(
            cohorts[0].representatives[0].canonical_ast_cbor
        )
        frontier_hashes = tuple(
            sorted(
                representative.canonical_ast_hash
                for cohort in cohorts
                if cohort.visible_frontier_member
                for representative in cohort.representatives
            )
        )
        classes.append(
            _snapshot.Q1BehaviorClassSnapshotV1(
                behavior_key=class_material.behavior_key,
                output_sort_id=_capacity.OUTPUT_SORT_IDS[
                    first_ast.metrics.output_sort
                ],
                behavior_cells=tuple(
                    _snapshot._cell_from_runtime(value)
                    for value in class_material.behavior
                ),
                cohorts=cohorts,
                visible_frontier_representative_hashes=frontier_hashes,
                minimum_admitted_mdl_q32=min(
                    cohort.signature.mdl_length_q32 for cohort in cohorts
                ),
            )
        )
    return _snapshot_from_result_and_classes_v1(result, tuple(classes))


def _snapshot_from_result_and_classes_v1(
    result: _capacity.Q1PartitionCapacityResultV1,
    classes: tuple[_snapshot.Q1BehaviorClassSnapshotV1, ...],
) -> _snapshot.Q1PartitionSnapshotV1:
    value = _snapshot.Q1PartitionSnapshotV1(
        schema_version=_snapshot.SNAPSHOT_SCHEMA_VERSION,
        snapshot_id=_snapshot.SNAPSHOT_ID,
        preflight_id=result.preflight_id,
        dsl_version=result.dsl_version,
        closure_semantics_version=result.closure_semantics_version,
        input_signature_id=result.input_signature_id,
        universe_row_count=result.universe_row_count,
        universe_root=result.universe_root,
        limits=result.limits,
        terminal_status=result.terminal_status,
        behavior_classes=classes,
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
    _snapshot._validate_q1_partition_snapshot_self_consistency_v1(value)
    return value


def run_full_node6_capacity_preflight_v1(
    admission: FullNode6AdmissionV1,
) -> tuple[
    FullNode6CapacityDiagnosticV1,
    tuple[_snapshot.Q1PartitionSnapshotV1 | None, ...],
]:
    """Run Odd then Sink once each under an exact outer-campaign admission."""

    if type(admission) is not FullNode6AdmissionV1:
        _fail("REJECT_FULL_NODE6_ADMISSION", "exact admission object is required")
    limits = _capacity.PreflightLimitsV1()
    rows: list[FullNode6PartitionDiagnosticV1] = []
    snapshots: list[_snapshot.Q1PartitionSnapshotV1 | None] = []
    for signature_id in (1, 2):
        row, snapshot = _run_partition_once_v1(signature_id, limits)
        rows.append(row)
        snapshots.append(snapshot)
        if row.terminal_status != TERMINAL_SATURATED:
            # The two partitions are sequential by contract.  A clean guard
            # hit terminates the endpoint without measuring the later one.
            break
    terminal = (
        TERMINAL_SATURATED
        if len(rows) == 2
        and all(row.terminal_status == TERMINAL_SATURATED for row in rows)
        else TERMINAL_GUARD_HIT
    )
    return (
        FullNode6CapacityDiagnosticV1(
            schema_version=SCHEMA_VERSION,
            action_id=ACTION_ID,
            capsule_digest=admission.capsule_digest,
            source_commit=admission.source_commit,
            terminal_status=terminal,
            partitions=tuple(rows),
        ),
        tuple(snapshots),
    )


def bounded_engine_self_test_v1(
    maximum_ast_node_count: int,
) -> tuple[_capacity.Q1PartitionCapacityResultV1, ...]:
    """Exercise the inherited engine below node six for source qualification."""

    if (
        type(maximum_ast_node_count) is not int
        or not 1 <= maximum_ast_node_count <= 5
    ):
        _fail(
            "REJECT_FULL_NODE6_SELF_TEST_SCOPE",
            "bounded self-test node count must be in 1..5",
        )
    limits = _capacity.PreflightLimitsV1(
        maximum_ast_node_count=maximum_ast_node_count
    )
    return tuple(
        _capacity.run_q1_partition_capacity_preflight_v1(
            signature_id,
            limits=limits,
        )
        for signature_id in (1, 2)
    )


__all__ = [
    "ACTION_ID",
    "ADMISSION_SCHEMA_VERSION",
    "FullNode6AdmissionV1",
    "FullNode6CapacityDiagnosticV1",
    "FullNode6CapacityPreflightError",
    "FullNode6PartitionDiagnosticV1",
    "SCHEMA_VERSION",
    "TERMINAL_GUARD_HIT",
    "TERMINAL_SATURATED",
    "bounded_engine_self_test_v1",
    "run_full_node6_capacity_preflight_v1",
]
