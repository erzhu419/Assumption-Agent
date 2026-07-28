"""Consumed-data qualification for a typed interacting evidence-set evaluator.

This module is deliberately not a fresh efficacy study.  It reuses only the
already-consumed HybridQA P6/E2 A_form, A_hold, and M_search packs to answer
one architecture question left open by the stopped marginal-action model:
can a fixed evaluator generalize when it scores a complete interacting top-5
evidence set rather than a sequence of independent replacement actions?

For each item the state space contains RAW and every distinct set obtained by
replacing one or two original RAW slots with candidates outside RAW that are
query anchored within two frozen typed edges.  No candidate is filtered or
sampled.  A fixed no-intercept, lambda-one convex set-energy model is trained
on exact complete-set utility deltas with item- and utility-stratum-balanced
weights.  Three leave-one-block-out folds are descriptive architecture
qualification only.  A single failed requirement stops this architecture.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from assumption_agent.benchmarks import (
    hybridqa_marginal_replacement_meta_development_v1 as base,
)
from assumption_agent.benchmarks import hybridqa_query_anchored_formal_runner_v1 as runner
from assumption_agent.benchmarks import hybridqa_query_anchored_operator_v1 as operator


VERSION = "hybridqa_set_interaction_meta_development_v1"
ARCHITECTURE_DECISION_SHA256 = (
    "fe9bc18d100a190faea21121ef0d934ea1fa222885fa031aba22d9d830ff9421"
)
PRIOR_MARGINAL_RESULT_SELF_SHA256 = (
    "78c882902ded2830b84a987c68262bb56d1459a444eed58046da923a88f636f5"
)
LABELED_BLOCKS = base.LABELED_BLOCKS
FAMILIES = base.FAMILIES
TOP_K = base.TOP_K
MAX_REPLACEMENTS = base.MAX_REPLACEMENTS
MAX_PATH_LENGTH = base.MAX_PATH_LENGTH
INTEGER_SCALE = base.INTEGER_SCALE
UTILITY_INTEGER_SCALE = base.UTILITY_INTEGER_SCALE
RIDGE_LAMBDA = 1.0
PROMOTION_ALPHA = Fraction(1, 10)
MAX_WORKERS = 8
EXPECTED_SET_ENERGY_NUMERIC_CANARY_SHA256 = ""

EXPECTED_LEGACY_POOLED = {
    "raw": {
        "total_utility": [781, 6],
        "complete_count": 56,
        "item_count": 108,
    },
    "marginal_v1": {
        "total_utility": [811, 6],
        "complete_count": 60,
        "item_count": 108,
    },
    "p6_path2": {
        "total_utility": [344, 3],
        "complete_count": 47,
        "item_count": 108,
    },
}
EXPECTED_LEGACY_FULL_PROJECTION_SHA256 = (
    "395f755517bd756bb16eb83c02dd15a6c9a46e10023561be2eaf89850fefe943"
)

FEATURE_ORDER = (
    "replacement_count_half",
    "removed_raw_dense_mean",
    "added_candidate_dense_mean",
    "final_dense_sum_delta_fifth",
    "final_dense_minimum_delta",
    "candidate_raw_rank_reciprocal_mean",
    "facet_coverage_mean_delta",
    "facet_coverage_minimum_delta",
    "facet_coverage_maximum_delta",
    "facet_newly_positive_fraction",
    "facet_lost_positive_fraction",
    "removed_raw_facet_deletion_loss_mean",
    "candidate_residual_facet_gain_mean",
    "joint_facet_coverage_synergy_mean",
    "negative_pairwise_semantic_redundancy_delta",
    "direct_anchor_fraction",
    "path_length_mean_half",
    "path_strength_mean",
    "candidate_table_row_fraction",
    "removed_table_row_fraction",
    "unit_type_change_fraction",
    "final_table_row_count_delta_fifth",
    "typed_pair_count_delta_tenth",
    "typed_pair_strength_delta_tenth",
    "adjacent_row_edge_count_delta_tenth",
    "row_to_passage_edge_count_delta_tenth",
    "shared_link_edge_count_delta_tenth",
    "connected_component_improvement_fifth",
    "largest_component_delta_fifth",
    "same_table_pair_count_delta_tenth",
    "cross_type_pair_count_delta_tenth",
    "candidate_pair_typed_edge_indicator",
    "candidate_pair_edge_strength",
    "candidate_pair_same_table_indicator",
    "candidate_pair_cross_type_indicator",
    "candidate_pair_distinct_anchor_facet_indicator",
    "candidate_pair_path_family_different_indicator",
    "candidate_pair_path_prefix_jaccard",
    "candidate_pair_path_family_disjoint_indicator",
    "candidate_pair_best_facet_distinct_indicator",
    "candidate_retained_typed_edge_fraction",
    "candidate_retained_edge_strength_mean",
    "candidate_retained_same_table_fraction",
    "candidate_retained_cross_type_fraction",
    "query_facet_count_eighth_times_replacement_fraction",
    "query_entity_facet_fraction_times_replacement_fraction",
    "query_numeric_facet_fraction_times_replacement_fraction",
    "query_relation_facet_fraction_times_replacement_fraction",
)

_OUTPUT_PAIR_LEFT = np.asarray(
    [left for left, _right in itertools.combinations(range(TOP_K), 2)],
    dtype=np.int64,
)
_OUTPUT_PAIR_RIGHT = np.asarray(
    [right for _left, right in itertools.combinations(range(TOP_K), 2)],
    dtype=np.int64,
)
_PAIR_BIT_WEIGHTS = np.asarray(
    [1 << index for index in range(math.comb(TOP_K, 2))],
    dtype=np.uint16,
)


class HybridQaSetInteractionError(RuntimeError):
    """A frozen set-space, feature, fit, or aggregate contract drifted."""


DiagnosticItem = base.DiagnosticItem
Corpus = base.Corpus
PortableCorpusIndex = base.PortableCorpusIndex
PolicyOutcome = base.PolicyOutcome


@dataclass(frozen=True)
class StateBatch:
    """One canonical fixed-replacement-count slice of the complete set space."""

    replacement_count: int
    outputs: np.ndarray
    features: np.ndarray
    utility_delta_ticks: np.ndarray
    complete: np.ndarray


@dataclass(frozen=True)
class ItemSufficientStatistics:
    gram: np.ndarray
    target: np.ndarray
    state_count: int
    non_noop_state_count: int
    stratum_count: int


@dataclass(frozen=True)
class _FeatureContext:
    global_ordinals: np.ndarray
    dense: np.ndarray
    coverage: np.ndarray
    table_row: np.ndarray
    raw_rank_reciprocal: np.ndarray
    direct_anchor: np.ndarray
    path_length_half: np.ndarray
    path_strength: np.ndarray
    edge_any: np.ndarray
    edge_strength: np.ndarray
    edge_family: np.ndarray
    same_table: np.ndarray
    cross_type: np.ndarray
    semantic_redundancy: np.ndarray
    pair_distinct_anchor_facet: np.ndarray
    pair_path_family_different: np.ndarray
    pair_path_prefix_jaccard: np.ndarray
    pair_path_family_disjoint: np.ndarray
    pair_best_facet_distinct: np.ndarray
    raw_facet_maximum: np.ndarray
    raw_dense_sum: float
    raw_dense_minimum: float
    raw_table_count: int
    raw_pair_count: float
    raw_pair_strength: float
    raw_family_counts: np.ndarray
    raw_component_count: int
    raw_largest_component: int
    raw_same_table_count: float
    raw_cross_type_count: float
    raw_semantic_redundancy: float
    deletion_loss_by_raw_slot: np.ndarray
    query_facet_type_counts: np.ndarray


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise HybridQaSetInteractionError("value is not canonical JSON") from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in body:
        raise HybridQaSetInteractionError("self-hash field already exists")
    return {**dict(body), field: stable_hash(body)}


def verify_self_hash(value: Mapping[str, Any], field: str) -> str:
    if not isinstance(value, Mapping):
        raise HybridQaSetInteractionError("self-hashed value is not a mapping")
    body = dict(value)
    declared = body.pop(field, None)
    if (
        not isinstance(declared, str)
        or len(declared) != 64
        or stable_hash(body) != declared
    ):
        raise HybridQaSetInteractionError(f"{field} drifted")
    return declared


def _sha256_float64(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array, dtype="<f8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def compute_set_energy_numeric_canary() -> dict[str, Any]:
    """Exercise the exact float64 Gram/solve/score path without source data."""

    row_count = 257
    dimension = len(FEATURE_ORDER)
    row = np.arange(1, row_count + 1, dtype=np.int64)[:, None]
    column = np.arange(3, dimension + 3, dtype=np.int64)[None, :]
    integer_matrix = (
        (row * column * 104_729 + row * row * 17 + column * 97)
        % 2_000_001
    ) - 1_000_000
    matrix = integer_matrix.astype(np.float64) / 1_000_000
    tick_registry = np.asarray((-12, -6, -3, 0, 2, 6, 12), dtype=np.int64)
    ticks = tick_registry[np.arange(row_count) % len(tick_registry)]
    gram = np.eye(dimension, dtype=np.float64) * RIDGE_LAMBDA
    target = np.zeros(dimension, dtype=np.float64)
    for tick in tick_registry:
        selected = matrix[ticks == tick]
        weight = 1.0 / (len(tick_registry) * len(selected))
        gram += selected.T @ selected * weight
        target += selected.T @ np.full(
            len(selected),
            tick / UTILITY_INTEGER_SCALE,
            dtype=np.float64,
        ) * weight
    weights = np.linalg.solve(gram, target)
    score_matrix = matrix[::7]
    scores = score_matrix @ weights
    if (
        weights.shape != (dimension,)
        or scores.shape != (37,)
        or not np.isfinite(weights).all()
        or not np.isfinite(scores).all()
    ):
        raise HybridQaSetInteractionError("numeric canary output drifted")
    payload = b"".join(
        np.ascontiguousarray(values, dtype="<f8").tobytes(order="C")
        for values in (gram, target, weights, scores)
    )
    return {
        "schema": f"{VERSION}_numeric_canary",
        "version": VERSION,
        "matrix_shape": [row_count, dimension],
        "score_count": len(scores),
        "best_score_index": int(np.argmax(scores)),
        "numpy_version": np.__version__,
        "float64_payload_sha256": hashlib.sha256(payload).hexdigest(),
    }


def verify_set_energy_numeric_canary() -> dict[str, Any]:
    receipt = compute_set_energy_numeric_canary()
    observed = receipt["float64_payload_sha256"]
    if (
        not EXPECTED_SET_ENERGY_NUMERIC_CANARY_SHA256
        or observed != EXPECTED_SET_ENERGY_NUMERIC_CANARY_SHA256
    ):
        raise HybridQaSetInteractionError(
            "set-energy numeric runtime canary drifted"
        )
    return {**receipt, "status": "verified_frozen_311_numeric_runtime"}


def complete_state_count(candidate_count: int) -> int:
    if type(candidate_count) is not int or candidate_count < 2:
        raise HybridQaSetInteractionError("candidate count is outside set-space bounds")
    return (
        1
        + TOP_K * candidate_count
        + math.comb(TOP_K, 2) * math.comb(candidate_count, 2)
    )


def _connectivity_lookup() -> tuple[np.ndarray, np.ndarray]:
    component_count = np.empty(1 << math.comb(TOP_K, 2), dtype=np.int8)
    largest_component = np.empty_like(component_count)
    pairs = tuple(itertools.combinations(range(TOP_K), 2))
    for mask in range(len(component_count)):
        parent = list(range(TOP_K))

        def find(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        for bit, (left, right) in enumerate(pairs):
            if mask & (1 << bit):
                left_root = find(left)
                right_root = find(right)
                if left_root != right_root:
                    parent[right_root] = left_root
        counts = Counter(find(index) for index in range(TOP_K))
        component_count[mask] = len(counts)
        largest_component[mask] = max(counts.values())
    component_count.setflags(write=False)
    largest_component.setflags(write=False)
    return component_count, largest_component


_COMPONENT_COUNT_BY_MASK, _LARGEST_COMPONENT_BY_MASK = _connectivity_lookup()


def _set_pair_arrays(
    context: _FeatureContext,
    outputs_local: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    left = outputs_local[:, _OUTPUT_PAIR_LEFT]
    right = outputs_local[:, _OUTPUT_PAIR_RIGHT]
    edge_any = context.edge_any[left, right]
    edge_strength = context.edge_strength[left, right]
    family = np.stack(
        [context.edge_family[index][left, right] for index in range(3)],
        axis=2,
    )
    same_table = context.same_table[left, right]
    cross_type = context.cross_type[left, right]
    redundancy = context.semantic_redundancy[left, right]
    mask = np.sum(
        edge_any.astype(np.uint16) * _PAIR_BIT_WEIGHTS[None, :],
        axis=1,
        dtype=np.uint16,
    )
    return (
        edge_any,
        edge_strength,
        family,
        same_table,
        cross_type,
        redundancy,
        mask,
    )


def _build_feature_context(
    *,
    item: DiagnosticItem,
    graph: operator.TypedCorpusGraph,
) -> _FeatureContext:
    expected_raw_order = tuple(
        sorted(
            range(operator.CORPUS_UNIT_COUNT),
            key=lambda ordinal: (
                -item.tensor.dense_relevance_ints[ordinal],
                ordinal,
            ),
        )
    )
    expected_ranks = [0] * operator.CORPUS_UNIT_COUNT
    for rank, ordinal in enumerate(expected_raw_order):
        expected_ranks[ordinal] = rank
    raw_set = set(expected_raw_order[:TOP_K])
    expected_candidates = tuple(
        ordinal
        for ordinal, record in enumerate(item.reachability)
        if ordinal not in raw_set
        and record.path_length is not None
        and record.path_length <= MAX_PATH_LENGTH
    )
    if (
        item.raw_top5 != expected_raw_order[:TOP_K]
        or item.raw_rank != tuple(expected_ranks)
        or len(item.reachability) != operator.CORPUS_UNIT_COUNT
        or any(
            record.unit_ordinal != ordinal
            for ordinal, record in enumerate(item.reachability)
        )
        or item.candidates != expected_candidates
        or len(item.candidates) < 2
    ):
        raise HybridQaSetInteractionError("item candidate universe drifted")
    global_ordinals = np.asarray((*item.raw_top5, *item.candidates), dtype=np.int64)
    local_count = len(global_ordinals)
    local_by_global = {
        int(ordinal): index for index, ordinal in enumerate(global_ordinals)
    }
    dense_ints = np.asarray(
        [item.tensor.dense_relevance_ints[int(value)] for value in global_ordinals],
        dtype=np.float64,
    )
    coverage_ints = np.asarray(
        [
            [row.semantic_coverage_ints[int(value)] for value in global_ordinals]
            for row in item.tensor.rows
        ],
        dtype=np.float64,
    )
    if (
        np.max(np.abs(dense_ints)) > INTEGER_SCALE
        or np.max(np.abs(coverage_ints)) > INTEGER_SCALE
    ):
        raise HybridQaSetInteractionError("quantized semantic range drifted")
    dense = dense_ints / INTEGER_SCALE
    coverage = np.maximum(coverage_ints / INTEGER_SCALE, 0.0)
    table_row = np.asarray(
        [
            graph.units[int(value)].unit_type == "table_row"
            for value in global_ordinals
        ],
        dtype=np.bool_,
    )
    raw_rank_reciprocal = np.asarray(
        [1.0 / (item.raw_rank[int(value)] + 1) for value in global_ordinals],
        dtype=np.float64,
    )
    direct_anchor = np.zeros(local_count, dtype=np.float64)
    path_length_half = np.zeros(local_count, dtype=np.float64)
    path_strength = np.zeros(local_count, dtype=np.float64)
    for local_index in range(TOP_K, local_count):
        reach = item.reachability[int(global_ordinals[local_index])]
        if reach.path_length not in {0, 1, 2}:
            raise HybridQaSetInteractionError("candidate reachability drifted")
        direct_anchor[local_index] = float(reach.direct_anchor)
        path_length_half[local_index] = reach.path_length / 2
        path_strength[local_index] = reach.path_strength_int / INTEGER_SCALE

    edge_any = np.zeros((local_count, local_count), dtype=np.bool_)
    edge_strength = np.zeros((local_count, local_count), dtype=np.float64)
    edge_family = np.zeros((3, local_count, local_count), dtype=np.bool_)
    for edge in graph.edges:
        left = local_by_global.get(edge.left_ordinal)
        right = local_by_global.get(edge.right_ordinal)
        if left is None or right is None:
            continue
        edge_any[left, right] = edge_any[right, left] = True
        strength = edge.strength_int / (edge.strength_int + INTEGER_SCALE)
        edge_strength[left, right] = edge_strength[right, left] = max(
            edge_strength[left, right],
            strength,
        )
        edge_family[edge.family_order, left, right] = True
        edge_family[edge.family_order, right, left] = True

    same_table = np.zeros((local_count, local_count), dtype=np.bool_)
    cross_type = np.zeros_like(same_table)
    for left in range(local_count):
        left_unit = graph.units[int(global_ordinals[left])]
        for right in range(left + 1, local_count):
            right_unit = graph.units[int(global_ordinals[right])]
            same = left_unit.table_key == right_unit.table_key
            cross = left_unit.unit_type != right_unit.unit_type
            same_table[left, right] = same_table[right, left] = same
            cross_type[left, right] = cross_type[right, left] = cross

    semantic_redundancy = np.zeros((local_count, local_count), dtype=np.float64)
    for left in range(local_count):
        overlap = np.minimum(
            coverage[:, left, None],
            coverage[:, left + 1 :],
        ).mean(axis=0)
        semantic_redundancy[left, left + 1 :] = overlap
        semantic_redundancy[left + 1 :, left] = overlap

    pair_distinct_anchor_facet = np.zeros_like(same_table)
    pair_path_family_different = np.zeros_like(same_table)
    pair_path_prefix_jaccard = np.zeros(
        (local_count, local_count),
        dtype=np.float64,
    )
    pair_path_family_disjoint = np.zeros_like(same_table)
    pair_best_facet_distinct = np.zeros_like(same_table)
    best_facets = np.argmax(coverage, axis=0)
    for left in range(TOP_K, local_count):
        left_reach = item.reachability[int(global_ordinals[left])]
        left_prefix = set(left_reach.path_unit_ordinals[:-1])
        left_families = set(left_reach.path_family_orders)
        for right in range(left + 1, local_count):
            right_reach = item.reachability[int(global_ordinals[right])]
            right_prefix = set(right_reach.path_unit_ordinals[:-1])
            right_families = set(right_reach.path_family_orders)
            distinct_facet = (
                left_reach.anchor_facet_i is not None
                and right_reach.anchor_facet_i is not None
                and left_reach.anchor_facet_i != right_reach.anchor_facet_i
            )
            family_different = left_families != right_families
            union = left_prefix.union(right_prefix)
            jaccard = (
                len(left_prefix.intersection(right_prefix)) / len(union)
                if union
                else 0.0
            )
            family_disjoint = bool(
                left_families
                and right_families
                and left_families.isdisjoint(right_families)
            )
            best_distinct = best_facets[left] != best_facets[right]
            pair_distinct_anchor_facet[left, right] = (
                pair_distinct_anchor_facet[right, left]
            ) = distinct_facet
            pair_path_family_different[left, right] = family_different
            pair_path_family_different[right, left] = family_different
            pair_path_prefix_jaccard[left, right] = pair_path_prefix_jaccard[
                right, left
            ] = jaccard
            pair_path_family_disjoint[left, right] = pair_path_family_disjoint[
                right, left
            ] = family_disjoint
            pair_best_facet_distinct[left, right] = pair_best_facet_distinct[
                right, left
            ] = best_distinct

    raw_local = np.arange(TOP_K, dtype=np.int64)[None, :]
    raw_left = raw_local[:, _OUTPUT_PAIR_LEFT]
    raw_right = raw_local[:, _OUTPUT_PAIR_RIGHT]
    raw_edges = edge_any[raw_left, raw_right]
    raw_strengths = edge_strength[raw_left, raw_right]
    raw_families = np.stack(
        [edge_family[index][raw_left, raw_right] for index in range(3)],
        axis=2,
    )
    raw_same_table = same_table[raw_left, raw_right]
    raw_cross_type = cross_type[raw_left, raw_right]
    raw_redundancy = semantic_redundancy[raw_left, raw_right]
    raw_mask = np.sum(
        raw_edges.astype(np.uint16) * _PAIR_BIT_WEIGHTS[None, :],
        axis=1,
        dtype=np.uint16,
    )
    raw_facet_maximum = np.max(coverage[:, :TOP_K], axis=1)
    deletion_loss_by_raw_slot = np.asarray(
        [
            np.mean(
                raw_facet_maximum
                - np.max(
                    coverage[
                        :,
                        [index for index in range(TOP_K) if index != slot],
                    ],
                    axis=1,
                )
            )
            for slot in range(TOP_K)
        ],
        dtype=np.float64,
    )
    facet_type_counts = np.asarray(
        [
            sum(facet.facet_type == family for facet in item.tensor.facets)
            for family in operator.FACET_TYPES
        ],
        dtype=np.float64,
    )
    context = _FeatureContext(
        global_ordinals=global_ordinals,
        dense=dense,
        coverage=coverage,
        table_row=table_row,
        raw_rank_reciprocal=raw_rank_reciprocal,
        direct_anchor=direct_anchor,
        path_length_half=path_length_half,
        path_strength=path_strength,
        edge_any=edge_any,
        edge_strength=edge_strength,
        edge_family=edge_family,
        same_table=same_table,
        cross_type=cross_type,
        semantic_redundancy=semantic_redundancy,
        pair_distinct_anchor_facet=pair_distinct_anchor_facet,
        pair_path_family_different=pair_path_family_different,
        pair_path_prefix_jaccard=pair_path_prefix_jaccard,
        pair_path_family_disjoint=pair_path_family_disjoint,
        pair_best_facet_distinct=pair_best_facet_distinct,
        raw_facet_maximum=raw_facet_maximum,
        raw_dense_sum=float(np.sum(dense[:TOP_K])),
        raw_dense_minimum=float(np.min(dense[:TOP_K])),
        raw_table_count=int(np.sum(table_row[:TOP_K])),
        raw_pair_count=float(np.sum(raw_edges)),
        raw_pair_strength=float(np.sum(raw_strengths)),
        raw_family_counts=np.sum(
            raw_families,
            axis=1,
            dtype=np.float64,
        )[0],
        raw_component_count=int(_COMPONENT_COUNT_BY_MASK[int(raw_mask[0])]),
        raw_largest_component=int(_LARGEST_COMPONENT_BY_MASK[int(raw_mask[0])]),
        raw_same_table_count=float(np.sum(raw_same_table)),
        raw_cross_type_count=float(np.sum(raw_cross_type)),
        raw_semantic_redundancy=float(np.sum(raw_redundancy)),
        deletion_loss_by_raw_slot=deletion_loss_by_raw_slot,
        query_facet_type_counts=facet_type_counts,
    )
    if (
        context.raw_family_counts.shape != (3,)
        or context.query_facet_type_counts.shape != (3,)
        or not all(
            np.isfinite(array).all()
            for array in (
                context.dense,
                context.coverage,
                context.edge_strength,
                context.semantic_redundancy,
                context.deletion_loss_by_raw_slot,
            )
        )
    ):
        raise HybridQaSetInteractionError("feature context drifted")
    return context


def _state_features(
    *,
    item: DiagnosticItem,
    context: _FeatureContext,
    slots: tuple[int, ...],
    candidate_local: np.ndarray,
    outputs_local: np.ndarray,
) -> np.ndarray:
    replacement_count = len(slots)
    if (
        replacement_count not in {1, 2}
        or candidate_local.ndim != 2
        or candidate_local.shape[1] != replacement_count
        or outputs_local.shape != (len(candidate_local), TOP_K)
        or tuple(sorted(set(slots))) != slots
        or any(not 0 <= slot < TOP_K for slot in slots)
    ):
        raise HybridQaSetInteractionError("state feature input drifted")
    # All following reductions use a unique set order.  This makes the
    # numerical representation, not merely its mathematical definition,
    # invariant to the arbitrary assignment of two candidates to deleted
    # RAW positions.
    candidate_local = np.sort(candidate_local, axis=1)
    outputs_local = np.sort(outputs_local, axis=1)
    row_count = len(outputs_local)
    replaced_local = np.asarray(slots, dtype=np.int64)
    retained_local = np.asarray(
        [index for index in range(TOP_K) if index not in slots],
        dtype=np.int64,
    )
    replacement_fraction = replacement_count / MAX_REPLACEMENTS

    selected_dense = context.dense[outputs_local]
    candidate_dense = context.dense[candidate_local]
    removed_dense = context.dense[replaced_local]
    selected_coverage = context.coverage[:, outputs_local]
    final_facet_maximum = np.max(selected_coverage, axis=2).T
    raw_facet_maximum = context.raw_facet_maximum[None, :]
    facet_delta = final_facet_maximum - raw_facet_maximum
    retained_facet_maximum = np.max(
        context.coverage[:, retained_local],
        axis=1,
    )
    candidate_coverage = np.transpose(
        context.coverage[:, candidate_local],
        (1, 2, 0),
    )
    residual_gain = np.maximum(
        candidate_coverage - retained_facet_maximum[None, None, :],
        0.0,
    )
    joint_synergy = np.zeros(row_count, dtype=np.float64)
    if replacement_count == 2:
        candidate_residuals = np.maximum(
            candidate_coverage - retained_facet_maximum[None, None, :],
            0.0,
        )
        joint_synergy = (
            np.mean(np.max(candidate_residuals, axis=1), axis=1)
            - np.max(np.mean(candidate_residuals, axis=2), axis=1)
        )

    (
        set_edges,
        set_strengths,
        set_families,
        set_same_table,
        set_cross_type,
        set_redundancy,
        set_masks,
    ) = _set_pair_arrays(context, outputs_local)
    set_components = _COMPONENT_COUNT_BY_MASK[set_masks]
    set_largest = _LARGEST_COMPONENT_BY_MASK[set_masks]

    pair_typed = np.zeros(row_count, dtype=np.float64)
    pair_strength = np.zeros(row_count, dtype=np.float64)
    pair_same_table = np.zeros(row_count, dtype=np.float64)
    pair_cross_type = np.zeros(row_count, dtype=np.float64)
    pair_distinct_anchor = np.zeros(row_count, dtype=np.float64)
    pair_family_different = np.zeros(row_count, dtype=np.float64)
    pair_prefix_jaccard = np.zeros(row_count, dtype=np.float64)
    pair_family_disjoint = np.zeros(row_count, dtype=np.float64)
    pair_best_facet_distinct = np.zeros(row_count, dtype=np.float64)
    if replacement_count == 2:
        left = candidate_local[:, 0]
        right = candidate_local[:, 1]
        pair_typed = context.edge_any[left, right].astype(np.float64)
        pair_strength = context.edge_strength[left, right]
        pair_same_table = context.same_table[left, right].astype(np.float64)
        pair_cross_type = context.cross_type[left, right].astype(np.float64)
        pair_distinct_anchor = context.pair_distinct_anchor_facet[
            left, right
        ].astype(np.float64)
        pair_family_different = context.pair_path_family_different[
            left, right
        ].astype(np.float64)
        pair_prefix_jaccard = context.pair_path_prefix_jaccard[left, right]
        pair_family_disjoint = context.pair_path_family_disjoint[
            left, right
        ].astype(np.float64)
        pair_best_facet_distinct = context.pair_best_facet_distinct[
            left, right
        ].astype(np.float64)

    candidate_retained_edges = context.edge_any[
        candidate_local[:, :, None],
        retained_local[None, None, :],
    ]
    candidate_retained_strength = context.edge_strength[
        candidate_local[:, :, None],
        retained_local[None, None, :],
    ]
    candidate_retained_same_table = context.same_table[
        candidate_local[:, :, None],
        retained_local[None, None, :],
    ]
    candidate_retained_cross_type = context.cross_type[
        candidate_local[:, :, None],
        retained_local[None, None, :],
    ]
    facet_count = len(item.tensor.facets)
    query_type_fractions = context.query_facet_type_counts / facet_count
    columns = (
        np.full(row_count, replacement_fraction),
        np.full(row_count, float(np.mean(removed_dense))),
        np.mean(candidate_dense, axis=1),
        (np.sum(selected_dense, axis=1) - context.raw_dense_sum) / TOP_K,
        np.min(selected_dense, axis=1) - context.raw_dense_minimum,
        np.mean(context.raw_rank_reciprocal[candidate_local], axis=1),
        np.mean(facet_delta, axis=1),
        np.min(final_facet_maximum, axis=1)
        - float(np.min(context.raw_facet_maximum)),
        np.max(final_facet_maximum, axis=1)
        - float(np.max(context.raw_facet_maximum)),
        np.mean(
            (raw_facet_maximum <= 0.0) & (final_facet_maximum > 0.0),
            axis=1,
        ),
        np.mean(
            (raw_facet_maximum > 0.0) & (final_facet_maximum <= 0.0),
            axis=1,
        ),
        np.full(
            row_count,
            float(np.mean(context.deletion_loss_by_raw_slot[replaced_local])),
        ),
        np.mean(residual_gain, axis=(1, 2)),
        joint_synergy,
        (context.raw_semantic_redundancy - np.sum(set_redundancy, axis=1))
        / math.comb(TOP_K, 2),
        np.mean(context.direct_anchor[candidate_local], axis=1),
        np.mean(context.path_length_half[candidate_local], axis=1),
        np.mean(context.path_strength[candidate_local], axis=1),
        np.mean(context.table_row[candidate_local], axis=1),
        np.full(
            row_count,
            float(np.mean(context.table_row[replaced_local])),
        ),
        np.abs(
            np.sum(context.table_row[candidate_local], axis=1)
            - float(np.sum(context.table_row[replaced_local]))
        )
        / replacement_count,
        (np.sum(context.table_row[outputs_local], axis=1) - context.raw_table_count)
        / TOP_K,
        (np.sum(set_edges, axis=1) - context.raw_pair_count)
        / math.comb(TOP_K, 2),
        (np.sum(set_strengths, axis=1) - context.raw_pair_strength)
        / math.comb(TOP_K, 2),
        (
            np.sum(set_families[:, :, 0], axis=1)
            - context.raw_family_counts[0]
        )
        / math.comb(TOP_K, 2),
        (
            np.sum(set_families[:, :, 1], axis=1)
            - context.raw_family_counts[1]
        )
        / math.comb(TOP_K, 2),
        (
            np.sum(set_families[:, :, 2], axis=1)
            - context.raw_family_counts[2]
        )
        / math.comb(TOP_K, 2),
        (context.raw_component_count - set_components) / TOP_K,
        (set_largest - context.raw_largest_component) / TOP_K,
        (np.sum(set_same_table, axis=1) - context.raw_same_table_count)
        / math.comb(TOP_K, 2),
        (np.sum(set_cross_type, axis=1) - context.raw_cross_type_count)
        / math.comb(TOP_K, 2),
        pair_typed,
        pair_strength,
        pair_same_table,
        pair_cross_type,
        pair_distinct_anchor,
        pair_family_different,
        pair_prefix_jaccard,
        pair_family_disjoint,
        pair_best_facet_distinct,
        np.mean(candidate_retained_edges, axis=(1, 2)),
        np.mean(candidate_retained_strength, axis=(1, 2)),
        np.mean(candidate_retained_same_table, axis=(1, 2)),
        np.mean(candidate_retained_cross_type, axis=(1, 2)),
        np.full(row_count, (facet_count / 8) * replacement_fraction),
        np.full(row_count, query_type_fractions[0] * replacement_fraction),
        np.full(row_count, query_type_fractions[1] * replacement_fraction),
        np.full(row_count, query_type_fractions[2] * replacement_fraction),
    )
    matrix = np.column_stack(columns).astype(np.float64, copy=False)
    if (
        matrix.shape != (row_count, len(FEATURE_ORDER))
        or not np.isfinite(matrix).all()
    ):
        raise HybridQaSetInteractionError("set feature matrix drifted")
    return matrix


def _utility_ticks(
    *,
    item: DiagnosticItem,
    outputs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gold = np.asarray(item.gold, dtype=np.int64)
    if len(gold) not in {1, 2, 3} or UTILITY_INTEGER_SCALE % len(gold):
        raise HybridQaSetInteractionError("gold utility scale drifted")
    hits = np.sum(
        np.any(outputs[:, :, None] == gold[None, None, :], axis=2),
        axis=1,
    )
    complete = hits == len(gold)
    raw_hits = len(set(item.raw_top5).intersection(item.gold))
    raw_complete = raw_hits == len(gold)
    raw_ticks = (
        raw_hits * (UTILITY_INTEGER_SCALE // len(gold))
        + int(raw_complete) * UTILITY_INTEGER_SCALE
    )
    ticks = (
        hits * (UTILITY_INTEGER_SCALE // len(gold))
        + complete.astype(np.int64) * UTILITY_INTEGER_SCALE
        - raw_ticks
    )
    return ticks.astype(np.int16), complete


def iter_state_batches(
    *,
    item: DiagnosticItem,
    graph: operator.TypedCorpusGraph,
    context: _FeatureContext | None = None,
) -> Iterable[StateBatch]:
    """Yield every non-noop final set once in canonical deterministic order."""

    feature_context = context or _build_feature_context(item=item, graph=graph)
    candidate_count = len(item.candidates)
    raw_local = np.arange(TOP_K, dtype=np.int64)
    for slot in range(TOP_K):
        candidate_indices = np.arange(candidate_count, dtype=np.int64)
        candidate_local = (candidate_indices + TOP_K)[:, None]
        outputs_local = np.tile(raw_local, (candidate_count, 1))
        outputs_local[:, slot] = candidate_local[:, 0]
        outputs = feature_context.global_ordinals[outputs_local]
        ticks, complete = _utility_ticks(item=item, outputs=outputs)
        yield StateBatch(
            replacement_count=1,
            outputs=outputs,
            features=_state_features(
                item=item,
                context=feature_context,
                slots=(slot,),
                candidate_local=candidate_local,
                outputs_local=outputs_local,
            ),
            utility_delta_ticks=ticks,
            complete=complete,
        )

    left_candidate, right_candidate = np.triu_indices(candidate_count, k=1)
    candidate_local_pair = np.column_stack(
        (left_candidate + TOP_K, right_candidate + TOP_K)
    ).astype(np.int64, copy=False)
    for left_slot, right_slot in itertools.combinations(range(TOP_K), 2):
        outputs_local = np.tile(raw_local, (len(candidate_local_pair), 1))
        outputs_local[:, left_slot] = candidate_local_pair[:, 0]
        outputs_local[:, right_slot] = candidate_local_pair[:, 1]
        outputs = feature_context.global_ordinals[outputs_local]
        ticks, complete = _utility_ticks(item=item, outputs=outputs)
        yield StateBatch(
            replacement_count=2,
            outputs=outputs,
            features=_state_features(
                item=item,
                context=feature_context,
                slots=(left_slot, right_slot),
                candidate_local=candidate_local_pair,
                outputs_local=outputs_local,
            ),
            utility_delta_ticks=ticks,
            complete=complete,
        )


def item_sufficient_statistics(
    *,
    item: DiagnosticItem,
    graph: operator.TypedCorpusGraph,
) -> ItemSufficientStatistics:
    """Accumulate exact per-stratum sufficient statistics without row storage."""

    dimension = len(FEATURE_ORDER)
    stratum_grams: dict[int, np.ndarray] = {
        0: np.zeros((dimension, dimension), dtype=np.float64)
    }
    stratum_targets: dict[int, np.ndarray] = {
        0: np.zeros(dimension, dtype=np.float64)
    }
    stratum_counts: Counter[int] = Counter({0: 1})  # explicit RAW no-op
    non_noop_count = 0
    context = _build_feature_context(item=item, graph=graph)
    for batch in iter_state_batches(item=item, graph=graph, context=context):
        matrix = batch.features
        ticks = batch.utility_delta_ticks.astype(np.int64)
        non_noop_count += len(matrix)
        for tick in np.unique(ticks):
            key = int(tick)
            selected = matrix[ticks == tick]
            gram = stratum_grams.setdefault(
                key,
                np.zeros((dimension, dimension), dtype=np.float64),
            )
            target = stratum_targets.setdefault(
                key,
                np.zeros(dimension, dtype=np.float64),
            )
            gram += selected.T @ selected
            target += selected.T @ np.full(
                len(selected),
                key / UTILITY_INTEGER_SCALE,
                dtype=np.float64,
            )
            stratum_counts[key] += len(selected)
    expected = complete_state_count(len(item.candidates))
    if non_noop_count + 1 != expected or set(stratum_grams) != set(stratum_counts):
        raise HybridQaSetInteractionError("complete state enumeration drifted")
    stratum_count = len(stratum_counts)
    gram = np.zeros((dimension, dimension), dtype=np.float64)
    target = np.zeros(dimension, dtype=np.float64)
    for tick in sorted(stratum_counts):
        weight = 1.0 / (stratum_count * stratum_counts[tick])
        gram += stratum_grams[tick] * weight
        target += stratum_targets[tick] * weight
    if (
        not np.isfinite(gram).all()
        or not np.isfinite(target).all()
        or not np.allclose(gram, gram.T, rtol=0.0, atol=1e-12)
    ):
        raise HybridQaSetInteractionError("item sufficient statistics drifted")
    return ItemSufficientStatistics(
        gram=gram,
        target=target,
        state_count=expected,
        non_noop_state_count=non_noop_count,
        stratum_count=stratum_count,
    )


def fit_set_energy(
    *,
    items: Sequence[DiagnosticItem],
    statistics: Mapping[str, ItemSufficientStatistics],
) -> tuple[np.ndarray, dict[str, Any]]:
    dimension = len(FEATURE_ORDER)
    gram = np.eye(dimension, dtype=np.float64) * RIDGE_LAMBDA
    target = np.zeros(dimension, dtype=np.float64)
    state_count = 0
    stratum_count = 0
    for item in items:
        row = statistics.get(item.commitment)
        if not isinstance(row, ItemSufficientStatistics):
            raise HybridQaSetInteractionError("training statistic is unavailable")
        gram += row.gram
        target += row.target
        state_count += row.state_count
        stratum_count += row.stratum_count
    try:
        weights = np.linalg.solve(gram, target)
    except np.linalg.LinAlgError as exc:
        raise HybridQaSetInteractionError("set-energy solve failed") from exc
    if weights.shape != (dimension,) or not np.isfinite(weights).all():
        raise HybridQaSetInteractionError("set-energy weights drifted")
    receipt = {
        "feature_count": dimension,
        "fit": "no_intercept_lambda_one_L2_convex_solve",
        "item_count": len(items),
        "lambda": "1",
        "complete_state_count_including_noop": state_count,
        "summed_item_utility_stratum_count": stratum_count,
        "per_item_total_weight": "1",
        "per_item_utility_delta_stratum_total_weights": "equal",
        "within_stratum_state_weights": "equal_complete_enumeration",
        "weights_float64_le_sha256": _sha256_float64(weights),
        "weights_persisted": False,
    }
    return weights, receipt


def select_set_and_oracle(
    *,
    item: DiagnosticItem,
    graph: operator.TypedCorpusGraph,
    weights: np.ndarray,
) -> tuple[PolicyOutcome, PolicyOutcome]:
    if weights.shape != (len(FEATURE_ORDER),) or not np.isfinite(weights).all():
        raise HybridQaSetInteractionError("selection weights drifted")
    learned_output = item.raw_top5
    learned_replacements = 0
    learned_score = 0.0
    oracle_output = item.raw_top5
    oracle_replacements = 0
    oracle_delta_ticks = 0
    context = _build_feature_context(item=item, graph=graph)
    for batch in iter_state_batches(item=item, graph=graph, context=context):
        scores = batch.features @ weights
        learned_index = int(np.argmax(scores))
        candidate_score = float(scores[learned_index])
        if not math.isfinite(candidate_score):
            raise HybridQaSetInteractionError("set-energy score is nonfinite")
        if candidate_score > learned_score:
            learned_score = candidate_score
            learned_output = tuple(
                int(value) for value in batch.outputs[learned_index]
            )
            learned_replacements = batch.replacement_count
        oracle_index = int(np.argmax(batch.utility_delta_ticks))
        candidate_ticks = int(batch.utility_delta_ticks[oracle_index])
        if candidate_ticks > oracle_delta_ticks:
            oracle_delta_ticks = candidate_ticks
            oracle_output = tuple(
                int(value) for value in batch.outputs[oracle_index]
            )
            oracle_replacements = batch.replacement_count
    return (
        PolicyOutcome(learned_output, learned_replacements),
        PolicyOutcome(oracle_output, oracle_replacements),
    )


def _fraction_payload(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _arm_summary(rows: Sequence[tuple[Fraction, bool]]) -> dict[str, Any]:
    return {
        "total_utility": _fraction_payload(
            sum((row[0] for row in rows), Fraction(0))
        ),
        "complete_count": sum(row[1] for row in rows),
        "item_count": len(rows),
    }


def _comparison_summary(
    treatment: Sequence[tuple[Fraction, bool]],
    control: Sequence[tuple[Fraction, bool]],
) -> dict[str, Any]:
    deltas = tuple(
        left[0] - right[0]
        for left, right in zip(treatment, control, strict=True)
    )
    return {
        "total_utility_delta": _fraction_payload(
            sum(deltas, Fraction(0))
        ),
        "complete_count_delta": sum(
            left[1] - right[1]
            for left, right in zip(treatment, control, strict=True)
        ),
        "positive_item_count": sum(delta > 0 for delta in deltas),
        "negative_item_count": sum(delta < 0 for delta in deltas),
        "zero_item_count": sum(delta == 0 for delta in deltas),
        "exact_one_sided_magnitude_sign_flip_p": _fraction_payload(
            base.exact_sign_flip_p(deltas)
        ),
    }


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    arm_names = (
        "raw",
        "set_learned",
        "set_oracle",
        "marginal_v1",
        "p6_path2",
    )
    arms = {arm: [row[arm] for row in rows] for arm in arm_names}
    return {
        "arms": {arm: _arm_summary(values) for arm, values in arms.items()},
        "comparisons": {
            "set_learned_minus_raw": _comparison_summary(
                arms["set_learned"], arms["raw"]
            ),
            "set_learned_minus_marginal_v1": _comparison_summary(
                arms["set_learned"], arms["marginal_v1"]
            ),
            "set_learned_minus_p6_path2": _comparison_summary(
                arms["set_learned"], arms["p6_path2"]
            ),
            "set_oracle_minus_raw": _comparison_summary(
                arms["set_oracle"], arms["raw"]
            ),
            "marginal_v1_minus_raw": _comparison_summary(
                arms["marginal_v1"], arms["raw"]
            ),
            "marginal_v1_minus_p6_path2": _comparison_summary(
                arms["marginal_v1"], arms["p6_path2"]
            ),
            "p6_path2_minus_raw": _comparison_summary(
                arms["p6_path2"], arms["raw"]
            ),
        },
        "set_learned_replacement_count": sum(
            row["set_learned_replacements"] for row in rows
        ),
        "set_oracle_replacement_count": sum(
            row["set_oracle_replacements"] for row in rows
        ),
        "complete_state_count": {
            "minimum": min(row["complete_state_count"] for row in rows),
            "maximum": max(row["complete_state_count"] for row in rows),
            "sum": sum(row["complete_state_count"] for row in rows),
        },
    }


def _positive(summary: Mapping[str, Any], comparison: str) -> bool:
    numerator, _denominator = summary["comparisons"][comparison][
        "total_utility_delta"
    ]
    return numerator > 0


def _legacy_safe_projection(
    summary: Mapping[str, Any],
    *,
    include_p6_minus_raw: bool,
) -> dict[str, Any]:
    arms = summary.get("arms")
    comparisons = summary.get("comparisons")
    if not isinstance(arms, Mapping) or not isinstance(comparisons, Mapping):
        raise HybridQaSetInteractionError("legacy safe projection drifted")
    comparison_names = [
        "marginal_v1_minus_raw",
        "marginal_v1_minus_p6_path2",
    ]
    if include_p6_minus_raw:
        comparison_names.append("p6_path2_minus_raw")
    return {
        "arms": {
            arm: dict(arms[arm])
            for arm in ("raw", "marginal_v1", "p6_path2")
        },
        "comparisons": {
            comparison: dict(comparisons[comparison])
            for comparison in comparison_names
        },
    }


def evaluate_crossfit(
    *,
    items: Sequence[DiagnosticItem],
    graph: operator.TypedCorpusGraph,
) -> dict[str, Any]:
    by_block = {
        block: tuple(item for item in items if item.block == block)
        for block in LABELED_BLOCKS
    }
    if (
        tuple(by_block) != LABELED_BLOCKS
        or sum(map(len, by_block.values())) != len(items)
        or len({item.commitment for item in items}) != len(items)
    ):
        raise HybridQaSetInteractionError("cross-fit blocks drifted")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        statistic_rows = tuple(
            executor.map(
                lambda item: item_sufficient_statistics(item=item, graph=graph),
                items,
            )
        )
    statistics = {
        item.commitment: row
        for item, row in zip(items, statistic_rows, strict=True)
    }
    fold_models: dict[str, tuple[np.ndarray, dict[str, Any], np.ndarray, dict[str, Any]]] = {}
    for held_block in LABELED_BLOCKS:
        train_items = tuple(
            item
            for block in LABELED_BLOCKS
            if block != held_block
            for item in by_block[block]
        )
        set_weights, set_fit = fit_set_energy(
            items=train_items,
            statistics=statistics,
        )
        marginal_weights, marginal_fit = base.fit_marginal_ridge(
            items=train_items,
            graph=graph,
        )
        fold_models[held_block] = (
            set_weights,
            set_fit,
            marginal_weights,
            marginal_fit,
        )

    fold_receipts: dict[str, Any] = {}
    pooled_rows: list[dict[str, Any]] = []
    for held_block in LABELED_BLOCKS:
        (
            set_weights,
            set_fit,
            marginal_weights,
            marginal_fit,
        ) = fold_models[held_block]

        def evaluate_item(item: DiagnosticItem) -> dict[str, Any]:
            set_learned, set_oracle = select_set_and_oracle(
                item=item,
                graph=graph,
                weights=set_weights,
            )
            marginal = base.apply_learned_policy(
                item=item,
                graph=graph,
                weights=marginal_weights,
            )
            p6 = operator.run_recipe(
                recipe_id="R3_P6_PATH2_B2",
                graph=graph,
                semantic_tensor=item.tensor,
            ).output_top5
            return {
                "block": held_block,
                "family": item.family,
                "raw": runner.item_utility(item.raw_top5, item.gold),
                "set_learned": runner.item_utility(
                    set_learned.output, item.gold
                ),
                "set_oracle": runner.item_utility(
                    set_oracle.output, item.gold
                ),
                "marginal_v1": runner.item_utility(
                    marginal.output, item.gold
                ),
                "p6_path2": runner.item_utility(p6, item.gold),
                "set_learned_replacements": set_learned.replacements,
                "set_oracle_replacements": set_oracle.replacements,
                "complete_state_count": complete_state_count(
                    len(item.candidates)
                ),
            }

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            held_rows = list(executor.map(evaluate_item, by_block[held_block]))
        pooled_rows.extend(held_rows)
        fold_receipts[held_block] = {
            "held_block": held_block,
            "train_blocks": [
                block for block in LABELED_BLOCKS if block != held_block
            ],
            "set_fit": set_fit,
            "legacy_marginal_fit": marginal_fit,
            "aggregate": _summarize(held_rows),
            "families": {
                family: _summarize(
                    [row for row in held_rows if row["family"] == family]
                )
                for family in FAMILIES
            },
        }

    pooled = _summarize(pooled_rows)
    pooled_families = {
        family: _summarize(
            [row for row in pooled_rows if row["family"] == family]
        )
        for family in FAMILIES
    }
    legacy_projection = {
        "pooled": _legacy_safe_projection(
            pooled,
            include_p6_minus_raw=False,
        ),
        "pooled_families": {
            family: _legacy_safe_projection(
                pooled_families[family],
                include_p6_minus_raw=False,
            )
            for family in FAMILIES
        },
        "folds": {
            block: {
                "aggregate": _legacy_safe_projection(
                    fold_receipts[block]["aggregate"],
                    include_p6_minus_raw=True,
                ),
                "families": {
                    family: _legacy_safe_projection(
                        fold_receipts[block]["families"][family],
                        include_p6_minus_raw=True,
                    )
                    for family in FAMILIES
                },
            }
            for block in LABELED_BLOCKS
        },
    }
    observed_legacy_projection_sha256 = stable_hash(legacy_projection)
    legacy_reproduction = {
        arm: pooled["arms"][arm] == expected
        for arm, expected in EXPECTED_LEGACY_POOLED.items()
    }
    legacy_reproduction.update(
        {
            "expected_full_projection_sha256": (
                EXPECTED_LEGACY_FULL_PROJECTION_SHA256
            ),
            "observed_full_projection_sha256": (
                observed_legacy_projection_sha256
            ),
            "full_fold_and_family_projection_exact": (
                observed_legacy_projection_sha256
                == EXPECTED_LEGACY_FULL_PROJECTION_SHA256
            ),
        }
    )
    learned_raw_p = Fraction(
        *pooled["comparisons"]["set_learned_minus_raw"][
            "exact_one_sided_magnitude_sign_flip_p"
        ]
    )
    learned_marginal_p = Fraction(
        *pooled["comparisons"]["set_learned_minus_marginal_v1"][
            "exact_one_sided_magnitude_sign_flip_p"
        ]
    )
    requirements = {
        "complete_set_oracle_positive_every_block_and_family": all(
            _positive(
                fold_receipts[block]["aggregate"],
                "set_oracle_minus_raw",
            )
            and all(
                _positive(
                    fold_receipts[block]["families"][family],
                    "set_oracle_minus_raw",
                )
                for family in FAMILIES
            )
            for block in LABELED_BLOCKS
        ),
        "set_learned_positive_every_held_block": all(
            _positive(
                fold_receipts[block]["aggregate"],
                "set_learned_minus_raw",
            )
            for block in LABELED_BLOCKS
        ),
        "set_learned_positive_every_pooled_family": all(
            _positive(
                pooled_families[family],
                "set_learned_minus_raw",
            )
            for family in FAMILIES
        ),
        "set_learned_vs_raw_pooled_exact_p_at_most_point_one": (
            learned_raw_p <= PROMOTION_ALPHA
        ),
        "set_learned_positive_over_fixed_p6_path2_pooled": _positive(
            pooled,
            "set_learned_minus_p6_path2",
        ),
        "set_learned_positive_over_marginal_v1_pooled": _positive(
            pooled,
            "set_learned_minus_marginal_v1",
        ),
        "set_learned_vs_marginal_v1_pooled_exact_p_at_most_point_one": (
            learned_marginal_p <= PROMOTION_ALPHA
        ),
        "legacy_safe_aggregates_exactly_reproduced": all(
            legacy_reproduction[arm]
            for arm in ("raw", "marginal_v1", "p6_path2")
        ),
        "legacy_full_fold_and_family_projection_exactly_reproduced": (
            legacy_reproduction["full_fold_and_family_projection_exact"]
        ),
    }
    return {
        "folds": fold_receipts,
        "pooled": pooled,
        "pooled_families": pooled_families,
        "legacy_reproduction": legacy_reproduction,
        "go_requirements": requirements,
        "decision": (
            "GO_ONE_INDEPENDENT_CONFIRMATORY_STUDY"
            if all(requirements.values())
            else "STOP_SET_INTERACTION_ARCHITECTURE"
        ),
    }


def build_safe_result(
    *,
    corpus: Corpus,
    items: Sequence[DiagnosticItem],
    evaluation: Mapping[str, Any],
    encoder: runner.Encoder,
    freeze_self_sha256: str,
) -> dict[str, Any]:
    body = {
        "schema": f"{VERSION}_safe_result",
        "version": VERSION,
        "status": "complete",
        "architecture_decision_self_sha256": ARCHITECTURE_DECISION_SHA256,
        "prior_marginal_result_self_sha256": PRIOR_MARGINAL_RESULT_SELF_SHA256,
        "implementation_freeze_self_sha256": freeze_self_sha256,
        "scope": {
            "fresh_efficacy_claim": False,
            "source_or_cohort_newly_consumed": False,
            "consumed_blocks": list(LABELED_BLOCKS),
            "item_count": len(items),
            "block_counts": dict(
                sorted(Counter(item.block for item in items).items())
            ),
            "family_counts": dict(
                sorted(Counter(item.family for item in items).items())
            ),
            "per_item_or_private_content_persisted": False,
        },
        "mechanism": {
            "state_space": "complete_noop_plus_all_distinct_one_or_two_replacement_sets",
            "candidate_filter_or_sampling": False,
            "maximum_replacements": MAX_REPLACEMENTS,
            "maximum_typed_path_length": MAX_PATH_LENGTH,
            "candidate_must_be_outside_original_RAW_top5": True,
            "set_invariant_candidate_assignment": True,
            "explicit_no_op_feature_and_score": "zero",
            "feature_order": list(FEATURE_ORDER),
            "fit": "item_and_utility_stratum_balanced_no_intercept_lambda_one_L2",
            "global_argmax_not_sequential_policy": True,
        },
        "bindings": {
            "corpus_pack_sha256": corpus.pack_sha256,
            "graph_sha256": corpus.graph.graph_sha256,
            "base_portable_encoder_receipt_sha256": (
                base.portable_encoder_receipt_sha256(encoder)
            ),
            "minilm_canary_receipt_sha256": base.stable_hash(
                dict(encoder.canary_receipt)
            ),
            "set_energy_numeric_canary_receipt_sha256": stable_hash(
                verify_set_energy_numeric_canary()
            ),
        },
        "evaluation": dict(evaluation),
        "activity_counts": {
            "new_source_download": 0,
            "fresh_selection": 0,
            "official_TEST_access": 0,
            "online_or_API_evaluation": 0,
            "HippoRAG_candidate_or_feature_access": 0,
            "retry_replay_resample": 0,
            "candidate_filter_or_sample": 0,
        },
    }
    return self_hashed(body, "result_self_sha256")


__all__ = [
    "ARCHITECTURE_DECISION_SHA256",
    "EXPECTED_LEGACY_FULL_PROJECTION_SHA256",
    "EXPECTED_LEGACY_POOLED",
    "EXPECTED_SET_ENERGY_NUMERIC_CANARY_SHA256",
    "FEATURE_ORDER",
    "FAMILIES",
    "HybridQaSetInteractionError",
    "ItemSufficientStatistics",
    "LABELED_BLOCKS",
    "MAX_PATH_LENGTH",
    "MAX_REPLACEMENTS",
    "MAX_WORKERS",
    "PRIOR_MARGINAL_RESULT_SELF_SHA256",
    "PROMOTION_ALPHA",
    "RIDGE_LAMBDA",
    "StateBatch",
    "VERSION",
    "build_safe_result",
    "complete_state_count",
    "compute_set_energy_numeric_canary",
    "evaluate_crossfit",
    "fit_set_energy",
    "item_sufficient_statistics",
    "iter_state_batches",
    "select_set_and_oracle",
    "self_hashed",
    "stable_hash",
    "verify_self_hash",
    "verify_set_energy_numeric_canary",
]
