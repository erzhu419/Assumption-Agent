"""Outcome-blind FEVEROUS evaluator feature production.

This sibling module turns one terminal P6 action trace into the exact eight
coordinates consumed by :mod:`feverous_e2_evaluator_v1`.  Its public producer
interfaces accept only a trusted typed graph, a complete semantic tensor, an
operator action trace, an opaque item commitment, and two externally verified
receipt hashes.  There is no source-record, late-result, comparator-output, or
open-ended feature input.

Every semantic and replacement calculation uses the complete 8192-unit
corpus.  The producer re-executes the frozen operator and requires exact trace
equality before trusting an ordered top five.  Exact :class:`fractions.Fraction`
features and content-free audit counts are retained in a self-hashed producer
trace; the nested evaluator :class:`RecipeTrace` uses the evaluator's frozen
80-digit Decimal encoding.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import json
from itertools import combinations
import re
from typing import Iterable, Mapping, Sequence

from assumption_agent.benchmarks import (
    feverous_p6_query_anchored_operator_v1 as operator,
)
from assumption_agent.benchmarks.feverous_e2_evaluator_v1 import (
    FEATURE_ORDER as EVALUATOR_FEATURE_ORDER,
    RECIPE_IDS as EVALUATOR_RECIPE_IDS,
    RecipeTrace,
)


VERSION = "feverous_e2_feature_producer_v1"
CORPUS_UNIT_COUNT = 8192
TOP_K = 5
MAXIMUM_QUERY_ANCHORED_PATH_LENGTH = 2
INTEGER_SCALE = 1_000_000
PAIR_COUNT_AT_FIVE = 10

FEATURE_ORDER = (
    "direct_facet_coverage",
    "residual_facet_coverage",
    "deletion_mean_coverage_drop",
    "deletion_minimum_coverage_drop",
    "same_type_replacement_mean_coverage_drop",
    "query_anchored_path_coverage",
    "dense_relevance_mass",
    "negative_pairwise_redundancy",
)
RECIPE_IDS = (
    "R0_DENSE5",
    "R1_P6_DIRECT_B2",
    "R2_P6_PATH1_B2",
    "R3_P6_PATH2_B2",
)

if FEATURE_ORDER != EVALUATOR_FEATURE_ORDER:
    raise RuntimeError("FEVEROUS evaluator feature schema drifted")
if RECIPE_IDS != EVALUATOR_RECIPE_IDS or RECIPE_IDS != operator.RECIPE_IDS:
    raise RuntimeError("FEVEROUS operator/evaluator recipe registry drifted")
if CORPUS_UNIT_COUNT != operator.CORPUS_UNIT_COUNT:
    raise RuntimeError("FEVEROUS closed-corpus size drifted")

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class FeverousFeatureProducerError(ValueError):
    """A trusted input, frozen feature, or producer receipt drifted."""


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
        raise FeverousFeatureProducerError(
            "feature receipt is not canonical JSON"
        ) from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _stream_hash(header: object, rows: Iterable[object]) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_bytes(header))
    for row in rows:
        digest.update(b"\n")
        digest.update(_canonical_bytes(row))
    return digest.hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise FeverousFeatureProducerError(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def _fraction_payload(value: Fraction) -> list[int]:
    if not isinstance(value, Fraction):
        raise FeverousFeatureProducerError("exact feature is not a Fraction")
    return [value.numerator, value.denominator]


FEATURE_BEHAVIOR_CONTRACT = {
    "closed_corpus_unit_count": CORPUS_UNIT_COUNT,
    "dense_relevance_mass": "sum_selected_quantized_dense_divided_by_1000000",
    "direct_facet_coverage": (
        "mean_over_facets_of_maximum_selected_quantized_semantic_coverage_"
        "divided_by_1000000"
    ),
    "feature_order": list(FEATURE_ORDER),
    "intervention_deletion": (
        "full_top5_coverage_minus_each_exact_four_unit_deletion_coverage_"
        "then_exact_mean_and_minimum"
    ),
    "negative_pairwise_redundancy": (
        "negative_mean_over_ten_selected_pairs_and_all_facets_of_minimum_"
        "positive_quantized_semantic_coverage_divided_by_1000000"
    ),
    "query_anchored_path_coverage": (
        "fraction_of_selected_units_reachable_at_length_zero_one_or_two_from_"
        "the_union_of_positive_direct_claim_facet_anchors"
    ),
    "replacement": (
        "for_each_selected_slot_scan_all_8192_units_select_best_coverage_"
        "unselected_exact_same_atomic_type_then_full_minus_best;_zero_if_none"
    ),
    "residual_facet_coverage": (
        "mean_over_facets_of_max_zero_full_top5_maximum_minus_retained_RAW_"
        "top3_maximum_divided_by_1000000"
    ),
    "top_k": TOP_K,
    "version": VERSION,
}
FEATURE_BEHAVIOR_SHA256 = stable_hash(FEATURE_BEHAVIOR_CONTRACT)


@dataclass(frozen=True)
class ExactFeatureVector:
    direct_facet_coverage: Fraction
    residual_facet_coverage: Fraction
    deletion_mean_coverage_drop: Fraction
    deletion_minimum_coverage_drop: Fraction
    same_type_replacement_mean_coverage_drop: Fraction
    query_anchored_path_coverage: Fraction
    dense_relevance_mass: Fraction
    negative_pairwise_redundancy: Fraction

    def __post_init__(self) -> None:
        for name in FEATURE_ORDER:
            if not isinstance(getattr(self, name), Fraction):
                raise FeverousFeatureProducerError(
                    f"{name} is not an exact Fraction"
                )

    def as_mapping(self) -> dict[str, Fraction]:
        return {name: getattr(self, name) for name in FEATURE_ORDER}

    def payload(self) -> dict[str, list[int]]:
        return {
            name: _fraction_payload(getattr(self, name))
            for name in FEATURE_ORDER
        }


@dataclass(frozen=True)
class FeatureDiagnostics:
    full_facet_maxima_ints: tuple[int, ...]
    retained_raw_top3_facet_maxima_ints: tuple[int, ...]
    deletion_coverage_drops: tuple[Fraction, ...]
    replacement_coverage_drops: tuple[Fraction, ...]
    replacement_available: tuple[bool, ...]
    replacement_same_type_candidate_counts: tuple[int, ...]
    replacement_candidate_consideration_count: int
    replacement_scan_sha256: str
    query_anchored_reachable_selected_count: int
    query_anchored_reachable_corpus_count: int
    query_anchored_scan_sha256: str
    positive_pairwise_redundancy_sum_int: int

    def __post_init__(self) -> None:
        if (
            len(self.deletion_coverage_drops) != TOP_K
            or len(self.replacement_coverage_drops) != TOP_K
            or len(self.replacement_available) != TOP_K
            or len(self.replacement_same_type_candidate_counts) != TOP_K
        ):
            raise FeverousFeatureProducerError(
                "feature diagnostics do not contain five intervention slots"
            )
        if any(
            not isinstance(value, Fraction)
            for value in (
                *self.deletion_coverage_drops,
                *self.replacement_coverage_drops,
            )
        ):
            raise FeverousFeatureProducerError(
                "intervention diagnostics are not exact Fractions"
            )
        if self.replacement_candidate_consideration_count != (
            TOP_K * CORPUS_UNIT_COUNT
        ):
            raise FeverousFeatureProducerError(
                "replacement diagnostic did not scan the complete corpus"
            )
        if not 0 <= self.query_anchored_reachable_selected_count <= TOP_K:
            raise FeverousFeatureProducerError(
                "selected path-reachability count is invalid"
            )
        if not 0 <= self.query_anchored_reachable_corpus_count <= CORPUS_UNIT_COUNT:
            raise FeverousFeatureProducerError(
                "corpus path-reachability count is invalid"
            )
        if self.positive_pairwise_redundancy_sum_int < 0:
            raise FeverousFeatureProducerError(
                "positive redundancy sum is negative"
            )
        _require_sha256(self.replacement_scan_sha256, "replacement scan hash")
        _require_sha256(
            self.query_anchored_scan_sha256, "query-anchored scan hash"
        )

    def payload(self) -> dict[str, object]:
        return {
            "deletion_coverage_drops": [
                _fraction_payload(value)
                for value in self.deletion_coverage_drops
            ],
            "full_facet_maxima_ints": list(self.full_facet_maxima_ints),
            "positive_pairwise_redundancy_sum_int": (
                self.positive_pairwise_redundancy_sum_int
            ),
            "query_anchored_reachable_corpus_count": (
                self.query_anchored_reachable_corpus_count
            ),
            "query_anchored_reachable_selected_count": (
                self.query_anchored_reachable_selected_count
            ),
            "query_anchored_scan_sha256": self.query_anchored_scan_sha256,
            "replacement_available": list(self.replacement_available),
            "replacement_candidate_consideration_count": (
                self.replacement_candidate_consideration_count
            ),
            "replacement_coverage_drops": [
                _fraction_payload(value)
                for value in self.replacement_coverage_drops
            ],
            "replacement_same_type_candidate_counts": list(
                self.replacement_same_type_candidate_counts
            ),
            "replacement_scan_sha256": self.replacement_scan_sha256,
            "retained_raw_top3_facet_maxima_ints": list(
                self.retained_raw_top3_facet_maxima_ints
            ),
        }


@dataclass(frozen=True)
class FeatureProvenance:
    item_commitment_sha256: str
    recipe_id: str
    operator_version: str
    operator_graph_sha256: str
    operator_action_trace_sha256: str
    semantic_tensor_sha256: str
    query_sha256: str
    ordered_top5_behavior_sha256: str
    external_operator_receipt_sha256: str
    external_semantic_receipt_sha256: str
    feature_behavior_sha256: str
    provenance_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "item_commitment_sha256",
            "operator_graph_sha256",
            "operator_action_trace_sha256",
            "semantic_tensor_sha256",
            "query_sha256",
            "ordered_top5_behavior_sha256",
            "external_operator_receipt_sha256",
            "external_semantic_receipt_sha256",
            "feature_behavior_sha256",
            "provenance_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if self.recipe_id not in RECIPE_IDS:
            raise FeverousFeatureProducerError(
                "provenance recipe is outside the frozen registry"
            )
        if self.operator_version != operator.VERSION:
            raise FeverousFeatureProducerError("operator version drifted")

    def body(self) -> dict[str, object]:
        return {
            "external_operator_receipt_sha256": (
                self.external_operator_receipt_sha256
            ),
            "external_semantic_receipt_sha256": (
                self.external_semantic_receipt_sha256
            ),
            "feature_behavior_sha256": self.feature_behavior_sha256,
            "item_commitment_sha256": self.item_commitment_sha256,
            "operator_action_trace_sha256": (
                self.operator_action_trace_sha256
            ),
            "operator_graph_sha256": self.operator_graph_sha256,
            "operator_version": self.operator_version,
            "ordered_top5_behavior_sha256": (
                self.ordered_top5_behavior_sha256
            ),
            "query_sha256": self.query_sha256,
            "recipe_id": self.recipe_id,
            "semantic_tensor_sha256": self.semantic_tensor_sha256,
            "version": VERSION,
        }

    def payload(self) -> dict[str, object]:
        return {**self.body(), "provenance_sha256": self.provenance_sha256}


def recompute_provenance_sha256(provenance: FeatureProvenance) -> str:
    if not isinstance(provenance, FeatureProvenance):
        raise FeverousFeatureProducerError("provenance has the wrong type")
    return stable_hash(provenance.body())


@dataclass(frozen=True)
class FeatureProductionTrace:
    """One E2 RecipeTrace plus exact source-only production provenance."""

    recipe_trace: RecipeTrace
    exact_features: ExactFeatureVector
    diagnostics: FeatureDiagnostics
    provenance: FeatureProvenance
    production_trace_sha256: str

    def payload_body(self) -> dict[str, object]:
        return {
            "diagnostics": self.diagnostics.payload(),
            "exact_features": self.exact_features.payload(),
            "provenance": self.provenance.payload(),
            "recipe_trace": self.recipe_trace.payload(),
            "schema": f"{VERSION}_production_trace",
            "version": VERSION,
        }


def recompute_feature_production_sha256(
    trace: FeatureProductionTrace,
) -> str:
    if not isinstance(trace, FeatureProductionTrace):
        raise FeverousFeatureProducerError(
            "feature production trace has the wrong type"
        )
    return stable_hash(trace.payload_body())


def ordered_top5_behavior_sha256(
    *,
    graph: operator.TypedCorpusGraph,
    semantic_tensor: operator.QuerySemanticTensor,
    ordered_top5: Sequence[int],
) -> str:
    rows = tuple(ordered_top5)
    if (
        len(rows) != TOP_K
        or len(set(rows)) != TOP_K
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < CORPUS_UNIT_COUNT
            for value in rows
        )
    ):
        raise FeverousFeatureProducerError(
            "behavior is not an exact ordered top five"
        )
    return stable_hash(
        {
            "graph_sha256": graph.graph_sha256,
            "ordered_top5": list(rows),
            "query_sha256": semantic_tensor.query_sha256,
            "schema": "feverous_exact_ordered_top5_behavior_v1",
            "semantic_tensor_sha256": semantic_tensor.tensor_sha256,
        }
    )


def _facet_maxima(
    tensor: operator.QuerySemanticTensor, selected: Sequence[int]
) -> tuple[int, ...]:
    rows = tuple(selected)
    if not rows:
        raise FeverousFeatureProducerError(
            "semantic coverage received an empty selected set"
        )
    return tuple(
        max(row.semantic_coverage_ints[ordinal] for ordinal in rows)
        for row in tensor.rows
    )


def _coverage_from_maxima(maxima: Sequence[int]) -> Fraction:
    rows = tuple(maxima)
    if not rows:
        raise FeverousFeatureProducerError("claim has no semantic facets")
    return Fraction(sum(rows), len(rows) * INTEGER_SCALE)


def _query_anchored_reachable(
    graph: operator.TypedCorpusGraph,
    tensor: operator.QuerySemanticTensor,
) -> tuple[frozenset[int], str]:
    reachable = {
        ordinal
        for ordinal in range(CORPUS_UNIT_COUNT)
        if any(
            row.direct_anchor_strength_ints[ordinal] > 0
            for row in tensor.rows
        )
    }
    frontier = set(reachable)
    for _depth in range(MAXIMUM_QUERY_ANCHORED_PATH_LENGTH):
        following: set[int] = set()
        for ordinal in frontier:
            following.update(
                edge.neighbor_ordinal for edge in graph.neighbors[ordinal]
            )
        following.difference_update(reachable)
        reachable.update(following)
        frontier = following
        if not frontier:
            break
    frozen = frozenset(reachable)
    receipt = _stream_hash(
        {
            "graph_sha256": graph.graph_sha256,
            "maximum_path_length": MAXIMUM_QUERY_ANCHORED_PATH_LENGTH,
            "query_sha256": tensor.query_sha256,
            "semantic_tensor_sha256": tensor.tensor_sha256,
        },
        (
            [ordinal, ordinal in frozen]
            for ordinal in range(CORPUS_UNIT_COUNT)
        ),
    )
    return frozen, receipt


def _replacement_diagnostics(
    *,
    graph: operator.TypedCorpusGraph,
    tensor: operator.QuerySemanticTensor,
    selected: tuple[int, int, int, int, int],
    full_coverage: Fraction,
) -> tuple[
    tuple[Fraction, ...],
    tuple[bool, ...],
    tuple[int, ...],
    int,
    str,
]:
    selected_set = set(selected)
    drops: list[Fraction] = []
    availability: list[bool] = []
    same_type_counts: list[int] = []
    consideration_count = 0
    digest = hashlib.sha256()
    digest.update(
        _canonical_bytes(
            {
                "corpus_size": CORPUS_UNIT_COUNT,
                "scan": "five_full_corpus_exact_same_type_replacements",
            }
        )
    )
    for slot, removed in enumerate(selected):
        reduced = selected[:slot] + selected[slot + 1 :]
        reduced_maxima = _facet_maxima(tensor, reduced)
        removed_type = graph.units[removed].unit_type
        best: Fraction | None = None
        eligible_count = 0
        for ordinal in range(CORPUS_UNIT_COUNT):
            consideration_count += 1
            eligible = (
                ordinal not in selected_set
                and graph.units[ordinal].unit_type == removed_type
            )
            candidate_coverage: Fraction | None = None
            if eligible:
                eligible_count += 1
                candidate_maxima = tuple(
                    max(
                        reduced_maxima[row.facet_i],
                        row.semantic_coverage_ints[ordinal],
                    )
                    for row in tensor.rows
                )
                candidate_coverage = _coverage_from_maxima(candidate_maxima)
                if best is None or candidate_coverage > best:
                    best = candidate_coverage
            digest.update(b"\n")
            digest.update(
                _canonical_bytes(
                    [
                        slot,
                        ordinal,
                        eligible,
                        None
                        if candidate_coverage is None
                        else _fraction_payload(candidate_coverage),
                    ]
                )
            )
        same_type_counts.append(eligible_count)
        availability.append(best is not None)
        # The frozen contract defines the feature drop itself as zero when no
        # exact-type replacement exists; it does not substitute deletion loss.
        drops.append(Fraction(0) if best is None else full_coverage - best)
    return (
        tuple(drops),
        tuple(availability),
        tuple(same_type_counts),
        consideration_count,
        digest.hexdigest(),
    )


def _compute_exact_features(
    *,
    graph: operator.TypedCorpusGraph,
    tensor: operator.QuerySemanticTensor,
    selected: tuple[int, int, int, int, int],
    retained_raw_top3: tuple[int, int, int],
    reachable: frozenset[int],
    reachable_scan_sha256: str,
) -> tuple[ExactFeatureVector, FeatureDiagnostics]:
    full_maxima = _facet_maxima(tensor, selected)
    raw_maxima = _facet_maxima(tensor, retained_raw_top3)
    full_coverage = _coverage_from_maxima(full_maxima)
    residual = Fraction(
        sum(
            max(0, full - retained)
            for full, retained in zip(full_maxima, raw_maxima)
        ),
        len(full_maxima) * INTEGER_SCALE,
    )

    deletion_drops = tuple(
        full_coverage
        - _coverage_from_maxima(
            _facet_maxima(tensor, selected[:slot] + selected[slot + 1 :])
        )
        for slot in range(TOP_K)
    )
    deletion_mean = sum(deletion_drops, Fraction(0)) / TOP_K
    deletion_minimum = min(deletion_drops)

    (
        replacement_drops,
        replacement_available,
        replacement_counts,
        replacement_considerations,
        replacement_scan_sha256,
    ) = _replacement_diagnostics(
        graph=graph,
        tensor=tensor,
        selected=selected,
        full_coverage=full_coverage,
    )
    replacement_mean = sum(replacement_drops, Fraction(0)) / TOP_K

    reachable_selected = sum(ordinal in reachable for ordinal in selected)
    path_coverage = Fraction(reachable_selected, TOP_K)
    dense_mass = Fraction(
        sum(tensor.dense_relevance_ints[ordinal] for ordinal in selected),
        INTEGER_SCALE,
    )

    redundancy_sum = 0
    for left, right in combinations(selected, 2):
        for row in tensor.rows:
            redundancy_sum += min(
                max(0, row.semantic_coverage_ints[left]),
                max(0, row.semantic_coverage_ints[right]),
            )
    negative_redundancy = -Fraction(
        redundancy_sum,
        PAIR_COUNT_AT_FIVE * len(tensor.rows) * INTEGER_SCALE,
    )

    features = ExactFeatureVector(
        direct_facet_coverage=full_coverage,
        residual_facet_coverage=residual,
        deletion_mean_coverage_drop=deletion_mean,
        deletion_minimum_coverage_drop=deletion_minimum,
        same_type_replacement_mean_coverage_drop=replacement_mean,
        query_anchored_path_coverage=path_coverage,
        dense_relevance_mass=dense_mass,
        negative_pairwise_redundancy=negative_redundancy,
    )
    diagnostics = FeatureDiagnostics(
        full_facet_maxima_ints=full_maxima,
        retained_raw_top3_facet_maxima_ints=raw_maxima,
        deletion_coverage_drops=deletion_drops,
        replacement_coverage_drops=replacement_drops,
        replacement_available=replacement_available,
        replacement_same_type_candidate_counts=replacement_counts,
        replacement_candidate_consideration_count=replacement_considerations,
        replacement_scan_sha256=replacement_scan_sha256,
        query_anchored_reachable_selected_count=reachable_selected,
        query_anchored_reachable_corpus_count=len(reachable),
        query_anchored_scan_sha256=reachable_scan_sha256,
        positive_pairwise_redundancy_sum_int=redundancy_sum,
    )
    return features, diagnostics


def _make_production_trace(
    *,
    item_commitment_sha256: str,
    action_trace: operator.ActionTrace,
    graph: operator.TypedCorpusGraph,
    tensor: operator.QuerySemanticTensor,
    exact_features: ExactFeatureVector,
    diagnostics: FeatureDiagnostics,
    external_operator_receipt_sha256: str,
    external_semantic_receipt_sha256: str,
) -> FeatureProductionTrace:
    behavior = ordered_top5_behavior_sha256(
        graph=graph,
        semantic_tensor=tensor,
        ordered_top5=action_trace.output_top5,
    )
    recipe_trace = RecipeTrace.from_mapping(
        item_commitment_sha256=item_commitment_sha256,
        recipe_id=action_trace.recipe_id,
        behavior_sha256=behavior,
        features=exact_features.as_mapping(),
    )
    provenance = FeatureProvenance(
        item_commitment_sha256=item_commitment_sha256,
        recipe_id=action_trace.recipe_id,
        operator_version=operator.VERSION,
        operator_graph_sha256=graph.graph_sha256,
        operator_action_trace_sha256=action_trace.trace_sha256,
        semantic_tensor_sha256=tensor.tensor_sha256,
        query_sha256=tensor.query_sha256,
        ordered_top5_behavior_sha256=behavior,
        external_operator_receipt_sha256=external_operator_receipt_sha256,
        external_semantic_receipt_sha256=external_semantic_receipt_sha256,
        feature_behavior_sha256=FEATURE_BEHAVIOR_SHA256,
        provenance_sha256="0" * 64,
    )
    provenance = replace(
        provenance,
        provenance_sha256=recompute_provenance_sha256(provenance),
    )
    produced = FeatureProductionTrace(
        recipe_trace=recipe_trace,
        exact_features=exact_features,
        diagnostics=diagnostics,
        provenance=provenance,
        production_trace_sha256="0" * 64,
    )
    produced = replace(
        produced,
        production_trace_sha256=recompute_feature_production_sha256(produced),
    )
    verify_feature_production_trace(
        produced,
        external_operator_receipt_sha256=external_operator_receipt_sha256,
        external_semantic_receipt_sha256=external_semantic_receipt_sha256,
    )
    return produced


def _validate_common_inputs(
    *,
    item_commitment_sha256: str,
    external_operator_receipt_sha256: str,
    external_semantic_receipt_sha256: str,
) -> None:
    _require_sha256(item_commitment_sha256, "item commitment")
    _require_sha256(
        external_operator_receipt_sha256, "external operator receipt"
    )
    _require_sha256(
        external_semantic_receipt_sha256, "external semantic receipt"
    )


def produce_e2_recipe_trace(
    *,
    item_commitment_sha256: str,
    graph: operator.TypedCorpusGraph,
    semantic_tensor: operator.QuerySemanticTensor,
    action_trace: operator.ActionTrace,
    external_operator_receipt_sha256: str,
    external_semantic_receipt_sha256: str,
) -> FeatureProductionTrace:
    """Produce one exact E2 RecipeTrace after semantic operator replay."""

    _validate_common_inputs(
        item_commitment_sha256=item_commitment_sha256,
        external_operator_receipt_sha256=external_operator_receipt_sha256,
        external_semantic_receipt_sha256=external_semantic_receipt_sha256,
    )
    if not isinstance(action_trace, operator.ActionTrace):
        raise FeverousFeatureProducerError("operator trace has the wrong type")
    try:
        expected = operator.run_recipe(
            recipe_id=action_trace.recipe_id,
            graph=graph,
            semantic_tensor=semantic_tensor,
        )
    except (operator.FeverousP6OperatorError, TypeError) as exc:
        raise FeverousFeatureProducerError(
            "operator graph, tensor, or action trace is not trusted"
        ) from exc
    if action_trace != expected:
        raise FeverousFeatureProducerError(
            "operator action trace does not equal frozen semantic replay"
        )
    reachable, reachable_hash = _query_anchored_reachable(
        graph, semantic_tensor
    )
    features, diagnostics = _compute_exact_features(
        graph=graph,
        tensor=semantic_tensor,
        selected=action_trace.output_top5,
        retained_raw_top3=action_trace.retained_raw_top3,
        reachable=reachable,
        reachable_scan_sha256=reachable_hash,
    )
    return _make_production_trace(
        item_commitment_sha256=item_commitment_sha256,
        action_trace=action_trace,
        graph=graph,
        tensor=semantic_tensor,
        exact_features=features,
        diagnostics=diagnostics,
        external_operator_receipt_sha256=external_operator_receipt_sha256,
        external_semantic_receipt_sha256=external_semantic_receipt_sha256,
    )


def produce_complete_e2_recipe_matrix(
    *,
    item_commitment_sha256: str,
    graph: operator.TypedCorpusGraph,
    semantic_tensor: operator.QuerySemanticTensor,
    action_traces: Sequence[operator.ActionTrace],
    external_operator_receipt_sha256: str,
    external_semantic_receipt_sha256: str,
) -> tuple[FeatureProductionTrace, ...]:
    """Produce exactly four recipe traces with shared physical feature work."""

    _validate_common_inputs(
        item_commitment_sha256=item_commitment_sha256,
        external_operator_receipt_sha256=external_operator_receipt_sha256,
        external_semantic_receipt_sha256=external_semantic_receipt_sha256,
    )
    supplied = tuple(action_traces)
    if len(supplied) != len(RECIPE_IDS) or any(
        not isinstance(trace, operator.ActionTrace) for trace in supplied
    ):
        raise FeverousFeatureProducerError(
            "feature matrix must contain exactly four operator traces"
        )
    by_recipe: dict[str, operator.ActionTrace] = {}
    for trace in supplied:
        if trace.recipe_id in by_recipe:
            raise FeverousFeatureProducerError(
                "feature matrix contains a duplicate recipe trace"
            )
        by_recipe[trace.recipe_id] = trace
    if set(by_recipe) != set(RECIPE_IDS):
        raise FeverousFeatureProducerError(
            "feature matrix recipe registry is incomplete"
        )
    try:
        expected = operator.run_all_recipes(
            graph=graph,
            semantic_tensor=semantic_tensor,
        )
    except (operator.FeverousP6OperatorError, TypeError) as exc:
        raise FeverousFeatureProducerError(
            "operator graph, tensor, or action matrix is not trusted"
        ) from exc
    expected_by_recipe = {trace.recipe_id: trace for trace in expected}
    if any(by_recipe[recipe] != expected_by_recipe[recipe] for recipe in RECIPE_IDS):
        raise FeverousFeatureProducerError(
            "operator action matrix does not equal frozen semantic replay"
        )

    reachable, reachable_hash = _query_anchored_reachable(
        graph, semantic_tensor
    )
    # Identical ordered outputs have identical label-free features.  Reuse the
    # expensive five-by-8192 replacement scan while retaining action-specific
    # provenance and recipe identity.
    feature_cache: dict[
        tuple[int, int, int, int, int],
        tuple[ExactFeatureVector, FeatureDiagnostics],
    ] = {}
    produced: list[FeatureProductionTrace] = []
    for recipe_id in RECIPE_IDS:
        action = by_recipe[recipe_id]
        cached = feature_cache.get(action.output_top5)
        if cached is None:
            cached = _compute_exact_features(
                graph=graph,
                tensor=semantic_tensor,
                selected=action.output_top5,
                retained_raw_top3=action.retained_raw_top3,
                reachable=reachable,
                reachable_scan_sha256=reachable_hash,
            )
            feature_cache[action.output_top5] = cached
        features, diagnostics = cached
        produced.append(
            _make_production_trace(
                item_commitment_sha256=item_commitment_sha256,
                action_trace=action,
                graph=graph,
                tensor=semantic_tensor,
                exact_features=features,
                diagnostics=diagnostics,
                external_operator_receipt_sha256=(
                    external_operator_receipt_sha256
                ),
                external_semantic_receipt_sha256=(
                    external_semantic_receipt_sha256
                ),
            )
        )
    return tuple(produced)


def verify_feature_production_trace(
    trace: FeatureProductionTrace,
    *,
    external_operator_receipt_sha256: str,
    external_semantic_receipt_sha256: str,
) -> str:
    """Verify self hashes, exact-to-E2 conversion, and external provenance."""

    if not isinstance(trace, FeatureProductionTrace):
        raise FeverousFeatureProducerError(
            "feature production trace has the wrong type"
        )
    operator_receipt = _require_sha256(
        external_operator_receipt_sha256, "external operator receipt"
    )
    semantic_receipt = _require_sha256(
        external_semantic_receipt_sha256, "external semantic receipt"
    )
    if (
        trace.provenance.external_operator_receipt_sha256 != operator_receipt
        or trace.provenance.external_semantic_receipt_sha256
        != semantic_receipt
    ):
        raise FeverousFeatureProducerError(
            "feature trace is outside the external provenance freeze"
        )
    if trace.provenance.feature_behavior_sha256 != FEATURE_BEHAVIOR_SHA256:
        raise FeverousFeatureProducerError(
            "feature behavior contract drifted"
        )
    if (
        recompute_provenance_sha256(trace.provenance)
        != trace.provenance.provenance_sha256
    ):
        raise FeverousFeatureProducerError("feature provenance self hash drifted")
    if (
        trace.recipe_trace.item_commitment_sha256
        != trace.provenance.item_commitment_sha256
        or trace.recipe_trace.recipe_id != trace.provenance.recipe_id
        or trace.recipe_trace.behavior_sha256
        != trace.provenance.ordered_top5_behavior_sha256
    ):
        raise FeverousFeatureProducerError(
            "nested E2 trace disagrees with provenance"
        )
    expected_recipe = RecipeTrace.from_mapping(
        item_commitment_sha256=trace.provenance.item_commitment_sha256,
        recipe_id=trace.provenance.recipe_id,
        behavior_sha256=trace.recipe_trace.behavior_sha256,
        features=trace.exact_features.as_mapping(),
    )
    if expected_recipe != trace.recipe_trace:
        raise FeverousFeatureProducerError(
            "nested E2 trace disagrees with exact features"
        )
    _require_sha256(trace.production_trace_sha256, "production trace hash")
    observed = recompute_feature_production_sha256(trace)
    if observed != trace.production_trace_sha256:
        raise FeverousFeatureProducerError(
            "feature production trace self hash drifted"
        )
    return observed


def verify_feature_production_against_inputs(
    trace: FeatureProductionTrace,
    *,
    graph: operator.TypedCorpusGraph,
    semantic_tensor: operator.QuerySemanticTensor,
    action_trace: operator.ActionTrace,
    external_operator_receipt_sha256: str,
    external_semantic_receipt_sha256: str,
) -> str:
    """Replay one production and reject even a consistently rehashed forgery."""

    verify_feature_production_trace(
        trace,
        external_operator_receipt_sha256=external_operator_receipt_sha256,
        external_semantic_receipt_sha256=external_semantic_receipt_sha256,
    )
    expected = produce_e2_recipe_trace(
        item_commitment_sha256=trace.provenance.item_commitment_sha256,
        graph=graph,
        semantic_tensor=semantic_tensor,
        action_trace=action_trace,
        external_operator_receipt_sha256=external_operator_receipt_sha256,
        external_semantic_receipt_sha256=external_semantic_receipt_sha256,
    )
    if trace != expected:
        raise FeverousFeatureProducerError(
            "feature production trace disagrees with trusted input replay"
        )
    return trace.production_trace_sha256


__all__ = [
    "CORPUS_UNIT_COUNT",
    "ExactFeatureVector",
    "FEATURE_BEHAVIOR_CONTRACT",
    "FEATURE_BEHAVIOR_SHA256",
    "FEATURE_ORDER",
    "FeatureDiagnostics",
    "FeatureProductionTrace",
    "FeatureProvenance",
    "FeverousFeatureProducerError",
    "INTEGER_SCALE",
    "MAXIMUM_QUERY_ANCHORED_PATH_LENGTH",
    "RECIPE_IDS",
    "TOP_K",
    "VERSION",
    "ordered_top5_behavior_sha256",
    "produce_complete_e2_recipe_matrix",
    "produce_e2_recipe_trace",
    "recompute_feature_production_sha256",
    "recompute_provenance_sha256",
    "stable_hash",
    "verify_feature_production_against_inputs",
    "verify_feature_production_trace",
]
