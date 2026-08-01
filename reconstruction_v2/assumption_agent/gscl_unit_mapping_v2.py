"""Polynomial, source-free mapping proposals for exclusive GSCL units.

The v2 extractor wire owns exactly two object endpoints per generator.  This
module treats that generator and its endpoints as one indivisible unit.  For
each member of a fixed eight-operator closure it forms a rectangular unit
weight matrix and returns at most four deterministic k-best injective
assignments.  A Lawler/Murty partition uses the Hungarian algorithm as its
polynomial constrained-assignment oracle.

Only semantic scores rank proposals.  Generator kind, polarity, and
orientation are intentionally *not* used to prune unit edges; the existing
score-free ``verify_correspondence`` boundary remains responsible for those
checks.  The output is the existing ``MappingSearchResult`` envelope, so the
flat and full arms consume the same bound proposal set.

This module performs no model, filesystem, benchmark-source, or network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import hashlib
import json
from typing import Any, ClassVar, Mapping

import assumption_agent.gscl_narrative_correspondence_v1 as v1
from assumption_agent.gscl_narrative_correspondence_v1 import (
    GlobalOperator,
    MappingSearchResult,
    NarrativeContractError,
    NarrativeExtraction,
    OrientationMode,
    PairMappingProposal,
    SemanticScoreTable,
    SlotPermutation,
    StructuralMapping,
)


UNIT_MAPPING_SCHEMA_VERSION = "gscl.unit_mapping.v2"
UNIT_MAPPING_ALGORITHM = (
    "deterministic_lawler_murty_k4_rectangular_hungarian_v1"
)
MAX_EXCLUSIVE_UNITS = 21
K_BEST_ASSIGNMENTS_PER_OPERATOR = 4


def _operator_closure() -> tuple[GlobalOperator, ...]:
    return tuple(
        GlobalOperator(
            orientation_mode=orientation,
            invert_polarity=invert,
            slot_permutation=permutation,
        )
        for orientation in (
            OrientationMode.PRESERVING,
            OrientationMode.INVERTING,
        )
        for invert in (False, True)
        for permutation in (
            SlotPermutation.IDENTITY,
            SlotPermutation.REVERSE,
        )
    )


UNIT_OPERATOR_CLOSURE = _operator_closure()
MAX_CONSTRAINED_ASSIGNMENT_SUBPROBLEMS = (
    len(UNIT_OPERATOR_CLOSURE)
    * (
        1
        + (K_BEST_ASSIGNMENTS_PER_OPERATOR - 1)
        * MAX_EXCLUSIVE_UNITS
    )
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class UnitMappingSearchConfigV2:
    """Non-tunable v2 search contract.

    The capacity, K, and operator closure are part of the algorithm rather
    than effect-search knobs.  ``max_assignments`` is a compatibility
    property for the v1 ``MappingSearchResult`` envelope.  It bounds the
    number of constrained Hungarian subproblems solved by Lawler/Murty, not
    candidate assignments or search-tree leaves.
    """

    schema_version: ClassVar[str] = UNIT_MAPPING_SCHEMA_VERSION
    algorithm: ClassVar[str] = UNIT_MAPPING_ALGORITHM
    max_units: ClassVar[int] = MAX_EXCLUSIVE_UNITS
    k_best_per_operator: ClassVar[int] = (
        K_BEST_ASSIGNMENTS_PER_OPERATOR
    )
    operators: ClassVar[tuple[GlobalOperator, ...]] = UNIT_OPERATOR_CLOSURE
    max_assignments: ClassVar[int] = (
        MAX_CONSTRAINED_ASSIGNMENT_SUBPROBLEMS
    )

    def safe_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm": self.algorithm,
            "max_units": self.max_units,
            "k_best_per_operator": self.k_best_per_operator,
            "unit_arity": 2,
            "edge_score": (
                "generator_semantic_plus_slot0_semantic_plus_slot1_semantic"
            ),
            "tie_break": (
                "source_unit_order_then_lexicographically_smallest_"
                "target_unit_indices"
            ),
            "operators": [
                operator.safe_payload() for operator in self.operators
            ],
            "assignments_explored_semantics": (
                "constrained_hungarian_assignment_subproblems_solved"
            ),
            "constrained_assignment_subproblem_budget": (
                self.max_assignments
            ),
            "worst_case_solver_complexity": (
                "O(operator_count*k_best*unit_count^4)"
            ),
        }

    @property
    def config_hash(self) -> str:
        return _content_hash(self.safe_payload())


@dataclass(frozen=True)
class _ExclusiveUnit:
    generator_id: str
    endpoints: tuple[str, str]


@dataclass(frozen=True)
class _Assignment:
    target_indices: tuple[int, ...]
    primary_score: int


@dataclass(frozen=True)
class _AssignmentSubproblem:
    fixed_prefix: tuple[int, ...]
    forbidden_edges: frozenset[tuple[int, int]]

    @property
    def canonical_key(
        self,
    ) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
        return self.fixed_prefix, tuple(sorted(self.forbidden_edges))


class _UnitWireError(ValueError):
    def __init__(self, issue_id: str) -> None:
        self.issue_id = issue_id
        super().__init__(issue_id)


def _exclusive_units(
    extraction: NarrativeExtraction,
    *,
    side: str,
) -> tuple[_ExclusiveUnit, ...]:
    generators = extraction.generators
    if len(generators) > MAX_EXCLUSIVE_UNITS:
        raise _UnitWireError(f"{side}_unit_capacity_exceeded")
    if any(len(generator.slot_mention_ids) != 2 for generator in generators):
        raise _UnitWireError(f"{side}_unit_arity_invalid")

    endpoints = tuple(
        endpoint
        for generator in generators
        for endpoint in generator.slot_mention_ids
    )
    if len(set(endpoints)) != len(endpoints):
        raise _UnitWireError(f"{side}_endpoint_ownership_invalid")
    object_ids = extraction.hypergraph.object_mention_ids
    if set(endpoints) != set(object_ids) or len(endpoints) != len(object_ids):
        raise _UnitWireError(f"{side}_endpoint_coverage_invalid")

    return tuple(
        _ExclusiveUnit(
            generator_id=generator.generator_id,
            endpoints=(
                generator.slot_mention_ids[0],
                generator.slot_mention_ids[1],
            ),
        )
        for generator in generators
    )


def _score_lookup(
    rows: tuple[tuple[str, str, int], ...],
    *,
    source_ids: set[str],
    target_ids: set[str],
    prefix: str,
) -> dict[tuple[str, str], int]:
    result: dict[tuple[str, str], int] = {}
    for source_id, target_id, score in rows:
        if source_id not in source_ids or target_id not in target_ids:
            raise NarrativeContractError(f"{prefix}_score_ref_invalid")
        result[(source_id, target_id)] = score
    return result


def _target_endpoints_for_source_order(
    target: _ExclusiveUnit,
    permutation: SlotPermutation,
) -> tuple[str, str]:
    if permutation is SlotPermutation.IDENTITY:
        return target.endpoints
    if permutation is SlotPermutation.REVERSE:
        return (target.endpoints[1], target.endpoints[0])
    raise NarrativeContractError("unit_operator_permutation_invalid")


def _unit_weight_matrix(
    source_units: tuple[_ExclusiveUnit, ...],
    target_units: tuple[_ExclusiveUnit, ...],
    *,
    object_scores: Mapping[tuple[str, str], int],
    generator_scores: Mapping[tuple[str, str], int],
    permutation: SlotPermutation,
) -> tuple[tuple[int | None, ...], ...]:
    rows: list[tuple[int | None, ...]] = []
    for source in source_units:
        row: list[int | None] = []
        for target in target_units:
            target_endpoints = _target_endpoints_for_source_order(
                target, permutation
            )
            keys = (
                (source.generator_id, target.generator_id),
                (source.endpoints[0], target_endpoints[0]),
                (source.endpoints[1], target_endpoints[1]),
            )
            generator_score = generator_scores.get(keys[0])
            first_score = object_scores.get(keys[1])
            second_score = object_scores.get(keys[2])
            if (
                generator_score is None
                or first_score is None
                or second_score is None
            ):
                row.append(None)
            else:
                row.append(generator_score + first_score + second_score)
        rows.append(tuple(row))
    return tuple(rows)


def _hungarian_maximum_injection(
    primary_weights: tuple[tuple[int | None, ...], ...],
) -> _Assignment | None:
    """Return the unique primary/lexicographic optimum in O(n^2 m).

    A positional mixed-radix secondary score makes the target-index vector
    unique without repeated solves.  One unit of primary score dominates the
    complete secondary range, so integer primary optimality is exact.
    """

    row_count = len(primary_weights)
    if row_count == 0:
        return _Assignment(target_indices=(), primary_score=0)
    column_count = len(primary_weights[0])
    if (
        column_count < row_count
        or any(len(row) != column_count for row in primary_weights)
    ):
        return None

    base = column_count + 1
    primary_scale = base**row_count
    positional_scales = tuple(
        base ** (row_count - 1 - row) for row in range(row_count)
    )
    composite: tuple[tuple[int | None, ...], ...] = tuple(
        tuple(
            None
            if weight is None
            else (
                weight * primary_scale
                + (column_count - column) * positional_scales[row]
            )
            for column, weight in enumerate(weights)
        )
        for row, weights in enumerate(primary_weights)
    )

    # Rectangular Hungarian algorithm for minimum cost.  Costs are the
    # negated composite scores.  ``None`` is a forbidden edge.
    row_potential = [0] * (row_count + 1)
    column_potential = [0] * (column_count + 1)
    column_owner = [0] * (column_count + 1)
    predecessor = [0] * (column_count + 1)

    for incoming_row in range(1, row_count + 1):
        column_owner[0] = incoming_row
        current_column = 0
        best_reduced: list[int | None] = [None] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[current_column] = True
            current_row = column_owner[current_column]
            delta: int | None = None
            next_column = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                score = composite[current_row - 1][column - 1]
                if score is not None:
                    reduced = (
                        -score
                        - row_potential[current_row]
                        - column_potential[column]
                    )
                    if (
                        best_reduced[column] is None
                        or reduced < best_reduced[column]
                    ):
                        best_reduced[column] = reduced
                        predecessor[column] = current_column
                candidate = best_reduced[column]
                if candidate is not None and (
                    delta is None
                    or candidate < delta
                    or (candidate == delta and column < next_column)
                ):
                    delta = candidate
                    next_column = column
            if delta is None:
                return None
            for column in range(column_count + 1):
                if used[column]:
                    row_potential[column_owner[column]] += delta
                    column_potential[column] -= delta
                elif best_reduced[column] is not None:
                    best_reduced[column] -= delta
            current_column = next_column
            if column_owner[current_column] == 0:
                break

        while True:
            previous_column = predecessor[current_column]
            column_owner[current_column] = column_owner[previous_column]
            current_column = previous_column
            if current_column == 0:
                break

    target_indices = [-1] * row_count
    for column in range(1, column_count + 1):
        owner = column_owner[column]
        if owner:
            target_indices[owner - 1] = column - 1
    if any(column < 0 for column in target_indices):
        return None
    assignment = tuple(target_indices)
    primary_score = 0
    for row, column in enumerate(assignment):
        weight = primary_weights[row][column]
        if weight is None:
            raise NarrativeContractError(
                "unit_assignment_forbidden_edge_selected"
            )
        primary_score += weight
    return _Assignment(
        target_indices=assignment,
        primary_score=primary_score,
    )


def _solve_assignment_subproblem(
    primary_weights: tuple[tuple[int | None, ...], ...],
    subproblem: _AssignmentSubproblem,
) -> _Assignment | None:
    row_count = len(primary_weights)
    if row_count == 0:
        column_count = 0
    else:
        column_count = len(primary_weights[0])
    if (
        len(subproblem.fixed_prefix) > row_count
        or any(len(row) != column_count for row in primary_weights)
        or any(
            row < 0
            or row >= row_count
            or column < 0
            or column >= column_count
            for row, column in subproblem.forbidden_edges
        )
    ):
        raise NarrativeContractError("unit_assignment_subproblem_invalid")

    fixed_columns = subproblem.fixed_prefix
    if (
        len(set(fixed_columns)) != len(fixed_columns)
        or any(
            column < 0 or column >= column_count
            for column in fixed_columns
        )
    ):
        return None
    fixed_score = 0
    for row, column in enumerate(fixed_columns):
        weight = primary_weights[row][column]
        if (
            weight is None
            or (row, column) in subproblem.forbidden_edges
        ):
            return None
        fixed_score += weight

    available_columns = tuple(
        column
        for column in range(column_count)
        if column not in set(fixed_columns)
    )
    remaining_start = len(fixed_columns)
    if row_count - remaining_start > len(available_columns):
        return None
    reduced = tuple(
        tuple(
            None
            if (row, column) in subproblem.forbidden_edges
            else primary_weights[row][column]
            for column in available_columns
        )
        for row in range(remaining_start, row_count)
    )
    residual = _hungarian_maximum_injection(reduced)
    if residual is None:
        return None
    target_indices = fixed_columns + tuple(
        available_columns[index] for index in residual.target_indices
    )
    return _Assignment(
        target_indices=target_indices,
        primary_score=fixed_score + residual.primary_score,
    )


def _k_best_maximum_injections(
    primary_weights: tuple[tuple[int | None, ...], ...],
    *,
    k: int,
) -> tuple[tuple[_Assignment, ...], int]:
    """Return exact k-best assignments via disjoint Lawler partitions.

    Each queue node fixes a source-row prefix and forbids a finite set of
    row/column edges.  Children partition their parent's remaining feasible
    assignments by the first row that differs from the popped optimum.
    """

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise NarrativeContractError("unit_kbest_limit_invalid")
    row_count = len(primary_weights)
    root = _AssignmentSubproblem(
        fixed_prefix=(),
        forbidden_edges=frozenset(),
    )
    root_solution = _solve_assignment_subproblem(
        primary_weights, root
    )
    subproblems_solved = 1
    if root_solution is None:
        return (), subproblems_solved

    serial = 0
    queue: list[
        tuple[
            int,
            tuple[int, ...],
            int,
            _AssignmentSubproblem,
            _Assignment,
        ]
    ] = [
        (
            -root_solution.primary_score,
            root_solution.target_indices,
            serial,
            root,
            root_solution,
        )
    ]
    seen_subproblems = {root.canonical_key}
    seen_assignments: set[tuple[int, ...]] = set()
    results: list[_Assignment] = []

    while queue and len(results) < k:
        _, _, _, subproblem, solution = heapq.heappop(queue)
        if solution.target_indices in seen_assignments:
            raise NarrativeContractError("unit_kbest_partition_overlap")
        seen_assignments.add(solution.target_indices)
        results.append(solution)
        if len(results) == k:
            break

        branch_start = len(subproblem.fixed_prefix)
        for split_row in range(branch_start, row_count):
            child = _AssignmentSubproblem(
                fixed_prefix=solution.target_indices[:split_row],
                forbidden_edges=(
                    subproblem.forbidden_edges
                    | frozenset(
                        {
                            (
                                split_row,
                                solution.target_indices[split_row],
                            )
                        }
                    )
                ),
            )
            if child.canonical_key in seen_subproblems:
                continue
            seen_subproblems.add(child.canonical_key)
            child_solution = _solve_assignment_subproblem(
                primary_weights, child
            )
            subproblems_solved += 1
            if child_solution is None:
                continue
            serial += 1
            heapq.heappush(
                queue,
                (
                    -child_solution.primary_score,
                    child_solution.target_indices,
                    serial,
                    child,
                    child_solution,
                ),
            )

    return tuple(results), subproblems_solved


def _empty_result(
    *,
    source: NarrativeExtraction,
    target: NarrativeExtraction,
    scores: SemanticScoreTable,
    config: UnitMappingSearchConfigV2,
    reason_ids: tuple[str, ...],
    assignments_explored: int = 0,
) -> MappingSearchResult:
    return v1._make_search_result(
        source=source,
        target=target,
        scores=scores,
        config=config,
        proposals=(),
        assignments_explored=assignments_explored,
        budget_exhausted=False,
        reason_ids=reason_ids,
    )


def generate_unit_mapping_proposals_v2(
    source: NarrativeExtraction,
    target: NarrativeExtraction,
    scores: SemanticScoreTable,
    *,
    config: UnitMappingSearchConfigV2 | None = None,
) -> MappingSearchResult:
    """Build at most 32 exact k-best unit-level proposals.

    The result is directly consumable by the v1 ``choose_flat_arm`` and
    ``choose_full_arm`` functions.  An incompatible v2 unit wire yields an
    empty, hash-bound result with a stable reason; malformed core objects
    still raise ``NarrativeContractError`` at the trust boundary.
    """

    config = config or UnitMappingSearchConfigV2()
    if (
        not isinstance(source, NarrativeExtraction)
        or not isinstance(target, NarrativeExtraction)
        or not isinstance(scores, SemanticScoreTable)
        or not isinstance(config, UnitMappingSearchConfigV2)
    ):
        raise NarrativeContractError("unit_mapping_inputs_invalid")
    source.__post_init__()
    target.__post_init__()
    scores.__post_init__()

    try:
        source_units = _exclusive_units(source, side="source")
        target_units = _exclusive_units(target, side="target")
    except _UnitWireError as error:
        return _empty_result(
            source=source,
            target=target,
            scores=scores,
            config=config,
            reason_ids=(error.issue_id,),
        )
    if len(source_units) > len(target_units):
        return _empty_result(
            source=source,
            target=target,
            scores=scores,
            config=config,
            reason_ids=("unit_injection_impossible",),
        )

    source_objects = set(source.hypergraph.object_mention_ids)
    target_objects = set(target.hypergraph.object_mention_ids)
    source_generators = {unit.generator_id for unit in source_units}
    target_generators = {unit.generator_id for unit in target_units}
    object_scores = _score_lookup(
        scores.object_scores,
        source_ids=source_objects,
        target_ids=target_objects,
        prefix="object",
    )
    generator_scores = _score_lookup(
        scores.generator_scores,
        source_ids=source_generators,
        target_ids=target_generators,
        prefix="generator",
    )

    proposals: list[PairMappingProposal] = []
    subproblems_solved = 0
    saw_row_complete_matrix = False
    for operator in config.operators:
        weights = _unit_weight_matrix(
            source_units,
            target_units,
            object_scores=object_scores,
            generator_scores=generator_scores,
            permutation=operator.slot_permutation,
        )
        if all(any(weight is not None for weight in row) for row in weights):
            saw_row_complete_matrix = True
        assignments, operator_subproblems = _k_best_maximum_injections(
            weights,
            k=config.k_best_per_operator,
        )
        subproblems_solved += operator_subproblems
        if subproblems_solved > config.max_assignments:
            raise NarrativeContractError("unit_polynomial_bound_violated")
        if not assignments:
            continue

        for assignment in assignments:
            object_mapping: list[tuple[str, str]] = []
            generator_mapping: list[tuple[str, str]] = []
            for source_index, target_index in enumerate(
                assignment.target_indices
            ):
                source_unit = source_units[source_index]
                target_unit = target_units[target_index]
                mapped_endpoints = _target_endpoints_for_source_order(
                    target_unit, operator.slot_permutation
                )
                generator_mapping.append(
                    (source_unit.generator_id, target_unit.generator_id)
                )
                object_mapping.extend(
                    (
                        (
                            source_unit.endpoints[0],
                            mapped_endpoints[0],
                        ),
                        (
                            source_unit.endpoints[1],
                            mapped_endpoints[1],
                        ),
                    )
                )
            proposals.append(
                PairMappingProposal(
                    mapping=StructuralMapping(
                        source_semantic_hash=source.semantic_hash,
                        target_semantic_hash=target.semantic_hash,
                        object_mapping=tuple(object_mapping),
                        generator_mapping=tuple(generator_mapping),
                        operator=operator,
                    ),
                    semantic_score_micros=assignment.primary_score,
                )
            )

    if not proposals:
        reason = (
            "unit_injective_assignment_empty"
            if saw_row_complete_matrix
            else "unit_edge_domain_empty"
        )
        return _empty_result(
            source=source,
            target=target,
            scores=scores,
            config=config,
            reason_ids=(reason,),
            assignments_explored=subproblems_solved,
        )

    return v1._make_search_result(
        source=source,
        target=target,
        scores=scores,
        config=config,
        proposals=tuple(
            sorted(proposals, key=lambda proposal: proposal.proposal_hash)
        ),
        assignments_explored=subproblems_solved,
        budget_exhausted=False,
        reason_ids=(),
    )


__all__ = [
    "K_BEST_ASSIGNMENTS_PER_OPERATOR",
    "MAX_EXCLUSIVE_UNITS",
    "MAX_CONSTRAINED_ASSIGNMENT_SUBPROBLEMS",
    "UNIT_MAPPING_ALGORITHM",
    "UNIT_MAPPING_SCHEMA_VERSION",
    "UNIT_OPERATOR_CLOSURE",
    "UnitMappingSearchConfigV2",
    "generate_unit_mapping_proposals_v2",
]
