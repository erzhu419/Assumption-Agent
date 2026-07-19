"""Exact, label-free ERASER R7-minus-R0 feature bridge.

This module is the only production constructor for the evaluator runner's
eight-dimensional :class:`DifferenceTrace`.  It accepts verified public
operator objects, not a caller-supplied feature mapping.  Every feature is
recomputed from the complete graph/tensor, the independently verified R0/R7
action traces, and a self-hashed exact measurement of the canonical sentence
pairs induced by those two frozen top-five actions.  Measuring that exact
pair union after the actions are fixed avoids an unnecessary quadratic scan
of a full clinical article while retaining the frozen MiniLM and quantization
identity.  No source reader, label, rationale, classifier, HippoRAG result,
network client, or binary float enters this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import combinations
import re
from typing import Sequence

from assumption_agent.benchmarks import (
    eraser_evidence_inference_r7_e3_runner_v1 as runner,
)
from assumption_agent.benchmarks import (
    eraser_evidence_inference_r7_operator_v1 as operator,
)


VERSION = "eraser_evidence_inference_exact_feature_bridge_v1"
TOP_K = operator.TOP_K
FACET_COUNT = len(operator.FACET_TYPES)
PAIR_COUNT = TOP_K * (TOP_K - 1) // 2
QUANTIZATION_SCALE = operator.INTEGER_SCALE
MINILM_ASSET_MANIFEST_SHA256 = (
    "837180aeb37eaaae2ebf108d2e3e2cb381db4d80152f75ff1da178ea5e144e88"
)
FEATURE_ORDER = runner.FEATURE_ORDER
R0_RECIPE_ID, R7_RECIPE_ID = runner.RECIPE_IDS

_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class EraserExactFeatureBridgeError(RuntimeError):
    """A semantic measurement, operator binding, feature, or receipt drifted."""


stable_hash = operator.stable_hash


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise EraserExactFeatureBridgeError(f"{field} is not a lowercase SHA-256")
    return value


def _require_int(value: object, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EraserExactFeatureBridgeError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise EraserExactFeatureBridgeError(f"{field} is below its minimum")
    return value


def _fraction_payload(value: Fraction) -> list[int]:
    if not isinstance(value, Fraction):
        raise EraserExactFeatureBridgeError("feature value is not an exact Fraction")
    return [value.numerator, value.denominator]


def _sentence_identity_sha256(
    sentence_sha256s: Sequence[str],
) -> str:
    return stable_hash(
        [[ordinal, value] for ordinal, value in enumerate(sentence_sha256s)]
    )


def _canonical_pairs(selected: Sequence[int]) -> tuple[tuple[int, int], ...]:
    rows = tuple(selected)
    if (
        len(rows) != TOP_K
        or len(set(rows)) != TOP_K
        or any(type(value) is not int or value < 0 for value in rows)
    ):
        raise EraserExactFeatureBridgeError("selected ordinals are not an exact top five")
    return tuple(combinations(sorted(rows), 2))


def _validate_top5(selected: tuple[int, ...], sentence_count: int, field: str) -> None:
    if (
        not isinstance(selected, tuple)
        or len(selected) != TOP_K
        or len(set(selected)) != TOP_K
        or any(
            type(ordinal) is not int or not 0 <= ordinal < sentence_count
            for ordinal in selected
        )
    ):
        raise EraserExactFeatureBridgeError(f"{field} is not an in-corpus top five")


def _required_pair_union(
    r0_top5: Sequence[int], r7_top5: Sequence[int]
) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(set(_canonical_pairs(r0_top5)) | set(_canonical_pairs(r7_top5))))


@dataclass(frozen=True)
class SelectedPairSemanticReceipt:
    """Exact MiniLM cosines for the action-induced canonical pair union."""

    graph_sha256: str
    semantic_tensor_sha256: str
    sentence_sha256s: tuple[str, ...]
    sentence_identity_sha256: str
    minilm_asset_manifest_sha256: str
    r0_top5: tuple[int, ...]
    r7_top5: tuple[int, ...]
    pair_rows: tuple[tuple[int, int, int], ...]
    receipt_sha256: str

    @property
    def sentence_count(self) -> int:
        return len(self.sentence_sha256s)

    def payload(self) -> dict[str, object]:
        return {
            **_selected_pair_receipt_body(self),
            "receipt_sha256": self.receipt_sha256,
        }


def _selected_pair_receipt_body(
    receipt: SelectedPairSemanticReceipt,
) -> dict[str, object]:
    return {
        "full_square_scan_required": False,
        "graph_sha256": receipt.graph_sha256,
        "measurement_scope": "exact_union_of_R0_and_R7_canonical_top5_pairs",
        "minilm_asset_manifest_sha256": receipt.minilm_asset_manifest_sha256,
        "pair_rows": [list(row) for row in receipt.pair_rows],
        "quantization": "cosine_times_1000000_math_fsum_then_Python_round_v1",
        "quantization_scale": QUANTIZATION_SCALE,
        "r0_top5": list(receipt.r0_top5),
        "r7_top5": list(receipt.r7_top5),
        "required_pair_count": len(receipt.pair_rows),
        "schema": f"{VERSION}_selected_pair_semantic_receipt",
        "semantic_tensor_sha256": receipt.semantic_tensor_sha256,
        "sentence_count": receipt.sentence_count,
        "sentence_identity_sha256": receipt.sentence_identity_sha256,
        "sentence_sha256s": list(receipt.sentence_sha256s),
        "version": VERSION,
    }


def recompute_selected_pair_receipt_sha256(
    receipt: SelectedPairSemanticReceipt,
) -> str:
    if not isinstance(receipt, SelectedPairSemanticReceipt):
        raise EraserExactFeatureBridgeError("selected-pair receipt has wrong type")
    return stable_hash(_selected_pair_receipt_body(receipt))


def _verify_selected_pair_shape_and_hash(
    receipt: SelectedPairSemanticReceipt,
) -> str:
    if not isinstance(receipt, SelectedPairSemanticReceipt):
        raise EraserExactFeatureBridgeError("selected-pair receipt has wrong type")
    for value, field in (
        (receipt.graph_sha256, "pair receipt graph hash"),
        (receipt.semantic_tensor_sha256, "pair receipt tensor hash"),
        (receipt.sentence_identity_sha256, "sentence identity hash"),
        (receipt.minilm_asset_manifest_sha256, "MiniLM asset manifest hash"),
        (receipt.receipt_sha256, "selected-pair receipt hash"),
    ):
        _require_sha256(value, field)
    if receipt.minilm_asset_manifest_sha256 != MINILM_ASSET_MANIFEST_SHA256:
        raise EraserExactFeatureBridgeError("selected-pair receipt uses an unfrozen model")
    if (
        not isinstance(receipt.sentence_sha256s, tuple)
        or any(not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None for value in receipt.sentence_sha256s)
    ):
        raise EraserExactFeatureBridgeError("sentence identity registry is malformed")
    sentence_count = len(receipt.sentence_sha256s)
    if sentence_count < TOP_K:
        raise EraserExactFeatureBridgeError("selected-pair corpus is smaller than top five")
    if receipt.sentence_identity_sha256 != _sentence_identity_sha256(receipt.sentence_sha256s):
        raise EraserExactFeatureBridgeError("sentence identity hash drifted")
    _validate_top5(receipt.r0_top5, sentence_count, "R0 pair action")
    _validate_top5(receipt.r7_top5, sentence_count, "R7 pair action")
    expected_pairs = _required_pair_union(receipt.r0_top5, receipt.r7_top5)
    if (
        not isinstance(receipt.pair_rows, tuple)
        or any(
            not isinstance(row, tuple) or len(row) != 3
            for row in receipt.pair_rows
        )
    ):
        raise EraserExactFeatureBridgeError("selected-pair row is malformed")
    if tuple((row[0], row[1]) for row in receipt.pair_rows) != expected_pairs:
        raise EraserExactFeatureBridgeError(
            "selected-pair registry is incomplete or noncanonical"
        )
    for row in receipt.pair_rows:
        if (
            not isinstance(row, tuple)
            or len(row) != 3
            or any(type(value) is not int for value in row)
            or not 0 <= row[0] < row[1] < sentence_count
            or not -QUANTIZATION_SCALE <= row[2] <= QUANTIZATION_SCALE
        ):
            raise EraserExactFeatureBridgeError("selected-pair row is malformed")
    if recompute_selected_pair_receipt_sha256(receipt) != receipt.receipt_sha256:
        raise EraserExactFeatureBridgeError("selected-pair receipt self hash drifted")
    return receipt.receipt_sha256


def build_selected_pair_semantic_receipt(
    *,
    graph: operator.QueryAnchoredSentenceGraph,
    semantic_tensor: operator.QuerySemanticTensor,
    r0_top5: Sequence[int],
    r7_top5: Sequence[int],
    pair_rows: Sequence[Sequence[int]],
    minilm_asset_manifest_sha256: str = MINILM_ASSET_MANIFEST_SHA256,
) -> SelectedPairSemanticReceipt:
    """Bind exactly the pair measurements required by two fixed actions."""

    try:
        operator.verify_query_anchored_graph(graph, semantic_tensor)
    except operator.EraserR7OperatorError as exc:
        raise EraserExactFeatureBridgeError("operator graph/tensor verification failed") from exc
    sentence_sha256s = tuple(unit.sentence_sha256 for unit in graph.units)
    receipt = SelectedPairSemanticReceipt(
        graph_sha256=graph.graph_sha256,
        semantic_tensor_sha256=semantic_tensor.tensor_sha256,
        sentence_sha256s=sentence_sha256s,
        sentence_identity_sha256=_sentence_identity_sha256(sentence_sha256s),
        minilm_asset_manifest_sha256=minilm_asset_manifest_sha256,
        r0_top5=tuple(r0_top5),
        r7_top5=tuple(r7_top5),
        pair_rows=tuple(tuple(row) for row in pair_rows),
        receipt_sha256="0" * 64,
    )
    receipt = replace(
        receipt,
        receipt_sha256=recompute_selected_pair_receipt_sha256(receipt),
    )
    verify_selected_pair_semantic_receipt(
        receipt, graph=graph, semantic_tensor=semantic_tensor
    )
    return receipt


def verify_selected_pair_semantic_receipt(
    receipt: SelectedPairSemanticReceipt,
    *,
    graph: operator.QueryAnchoredSentenceGraph | None = None,
    semantic_tensor: operator.QuerySemanticTensor | None = None,
) -> str:
    result = _verify_selected_pair_shape_and_hash(receipt)
    if (graph is None) != (semantic_tensor is None):
        raise EraserExactFeatureBridgeError("pair input verification requires graph and tensor")
    if graph is not None and semantic_tensor is not None:
        try:
            operator.verify_query_anchored_graph(graph, semantic_tensor)
        except operator.EraserR7OperatorError as exc:
            raise EraserExactFeatureBridgeError("operator graph/tensor verification failed") from exc
        if (
            receipt.graph_sha256 != graph.graph_sha256
            or receipt.semantic_tensor_sha256 != semantic_tensor.tensor_sha256
            or receipt.sentence_sha256s != tuple(unit.sentence_sha256 for unit in graph.units)
        ):
            raise EraserExactFeatureBridgeError("selected-pair ordinal binding drifted")
    return result


def _pair_values(
    receipt: SelectedPairSemanticReceipt,
    selected: Sequence[int],
) -> tuple[int, ...]:
    pairs = _canonical_pairs(selected)
    values = {(left, right): value for left, right, value in receipt.pair_rows}
    try:
        return tuple(values[pair] for pair in pairs)
    except KeyError as exc:
        raise EraserExactFeatureBridgeError("selected pair is absent from receipt") from exc


def _validate_loo_matrix(
    rows: tuple[tuple[int, int, int], ...], field: str
) -> None:
    if (
        not isinstance(rows, tuple)
        or len(rows) != TOP_K
        or any(not isinstance(row, tuple) or len(row) != FACET_COUNT for row in rows)
        or any(type(value) is not int for row in rows for value in row)
    ):
        raise EraserExactFeatureBridgeError(
            f"{field} must contain exact five-by-three coverage deltas"
        )


@dataclass(frozen=True)
class ExactFeatureComputationReceipt:
    item_commitment_sha256: str
    sentence_count: int
    graph_sha256: str
    semantic_tensor_sha256: str
    selected_pair_semantic_receipt_sha256: str
    r0_action_trace_sha256: str
    r7_action_trace_sha256: str
    r0_operator_behavior_sha256: str
    r7_operator_behavior_sha256: str
    r0_runner_behavior_sha256: str
    r7_runner_behavior_sha256: str
    r0_top5: tuple[int, ...]
    r7_top5: tuple[int, ...]
    r0_facet_maxima_ints: tuple[int, int, int]
    r7_facet_maxima_ints: tuple[int, int, int]
    minimum_positive_anchor_strength: int
    r0_dense_relevance_mass_int: int
    r7_dense_relevance_mass_int: int
    r0_leave_one_out_coverage_deltas: tuple[tuple[int, int, int], ...]
    r7_leave_one_out_coverage_deltas: tuple[tuple[int, int, int], ...]
    edge_deletion_action_change_indicator: int
    r0_pair_similarity_ints: tuple[int, ...]
    r7_pair_similarity_ints: tuple[int, ...]
    features: tuple[Fraction, ...]
    feature_receipt_sha256: str

    def payload(self) -> dict[str, object]:
        return {
            **_feature_receipt_body(self),
            "feature_receipt_sha256": self.feature_receipt_sha256,
        }


def _feature_receipt_body(
    receipt: ExactFeatureComputationReceipt,
) -> dict[str, object]:
    return {
        "edge_deletion_action_change_indicator": (
            receipt.edge_deletion_action_change_indicator
        ),
        "feature_order": list(FEATURE_ORDER),
        "feature_values": [_fraction_payload(value) for value in receipt.features],
        "graph_sha256": receipt.graph_sha256,
        "item_commitment_sha256": receipt.item_commitment_sha256,
        "labels_gold_family_Hippo_or_external_feature_map_accessed": False,
        "minimum_positive_anchor_strength": (
            receipt.minimum_positive_anchor_strength
        ),
        "online_evaluator_calls": 0,
        "r0_action_trace_sha256": receipt.r0_action_trace_sha256,
        "r0_dense_relevance_mass_int": receipt.r0_dense_relevance_mass_int,
        "r0_facet_maxima_ints": list(receipt.r0_facet_maxima_ints),
        "r0_leave_one_out_coverage_deltas": [
            list(row) for row in receipt.r0_leave_one_out_coverage_deltas
        ],
        "r0_operator_behavior_sha256": receipt.r0_operator_behavior_sha256,
        "r0_pair_ordinal_pairs": [
            list(pair) for pair in _canonical_pairs(receipt.r0_top5)
        ],
        "r0_pair_similarity_ints": list(receipt.r0_pair_similarity_ints),
        "r0_runner_behavior_sha256": receipt.r0_runner_behavior_sha256,
        "r0_top5": list(receipt.r0_top5),
        "r7_action_trace_sha256": receipt.r7_action_trace_sha256,
        "r7_dense_relevance_mass_int": receipt.r7_dense_relevance_mass_int,
        "r7_facet_maxima_ints": list(receipt.r7_facet_maxima_ints),
        "r7_leave_one_out_coverage_deltas": [
            list(row) for row in receipt.r7_leave_one_out_coverage_deltas
        ],
        "r7_operator_behavior_sha256": receipt.r7_operator_behavior_sha256,
        "r7_pair_ordinal_pairs": [
            list(pair) for pair in _canonical_pairs(receipt.r7_top5)
        ],
        "r7_pair_similarity_ints": list(receipt.r7_pair_similarity_ints),
        "r7_runner_behavior_sha256": receipt.r7_runner_behavior_sha256,
        "r7_top5": list(receipt.r7_top5),
        "schema": f"{VERSION}_feature_computation_receipt",
        "semantic_tensor_sha256": receipt.semantic_tensor_sha256,
        "sentence_count": receipt.sentence_count,
        "selected_pair_semantic_receipt_sha256": (
            receipt.selected_pair_semantic_receipt_sha256
        ),
        "source": "verified_operator_graph_tensor_actions_and_exact_selected_pair_union",
        "version": VERSION,
    }


def recompute_feature_receipt_sha256(
    receipt: ExactFeatureComputationReceipt,
) -> str:
    if not isinstance(receipt, ExactFeatureComputationReceipt):
        raise EraserExactFeatureBridgeError("feature computation receipt has wrong type")
    return stable_hash(_feature_receipt_body(receipt))


def _features_from_receipt(
    receipt: ExactFeatureComputationReceipt,
) -> tuple[Fraction, ...]:
    outside_raw = Fraction(len(set(receipt.r7_top5).difference(receipt.r0_top5)))
    coverage_gain = Fraction(
        sum(receipt.r7_facet_maxima_ints) - sum(receipt.r0_facet_maxima_ints),
        FACET_COUNT,
    )
    new_facets = Fraction(
        sum(
            r7_value > r0_value
            for r0_value, r7_value in zip(
                receipt.r0_facet_maxima_ints,
                receipt.r7_facet_maxima_ints,
            )
        )
    )
    minimum_anchor = Fraction(receipt.minimum_positive_anchor_strength)
    dense_delta = Fraction(
        receipt.r7_dense_relevance_mass_int
        - receipt.r0_dense_relevance_mass_int
    )
    r0_loo_total = sum(
        value for row in receipt.r0_leave_one_out_coverage_deltas for value in row
    )
    r7_loo_total = sum(
        value for row in receipt.r7_leave_one_out_coverage_deltas for value in row
    )
    deletion_mean_delta = Fraction(
        r7_loo_total - r0_loo_total,
        TOP_K * FACET_COUNT,
    )
    edge_change = Fraction(receipt.edge_deletion_action_change_indicator)
    negative_redundancy_delta = Fraction(
        sum(receipt.r0_pair_similarity_ints)
        - sum(receipt.r7_pair_similarity_ints),
        PAIR_COUNT,
    )
    return (
        outside_raw,
        coverage_gain,
        new_facets,
        minimum_anchor,
        dense_delta,
        deletion_mean_delta,
        edge_change,
        negative_redundancy_delta,
    )


def verify_feature_computation_receipt(
    receipt: ExactFeatureComputationReceipt,
) -> str:
    if not isinstance(receipt, ExactFeatureComputationReceipt):
        raise EraserExactFeatureBridgeError("feature computation receipt has wrong type")
    for value, field in (
        (receipt.item_commitment_sha256, "item commitment"),
        (receipt.graph_sha256, "graph hash"),
        (receipt.semantic_tensor_sha256, "semantic tensor hash"),
        (
            receipt.selected_pair_semantic_receipt_sha256,
            "selected-pair semantic receipt hash",
        ),
        (receipt.r0_action_trace_sha256, "R0 action hash"),
        (receipt.r7_action_trace_sha256, "R7 action hash"),
        (receipt.r0_operator_behavior_sha256, "R0 operator behavior hash"),
        (receipt.r7_operator_behavior_sha256, "R7 operator behavior hash"),
        (receipt.r0_runner_behavior_sha256, "R0 runner behavior hash"),
        (receipt.r7_runner_behavior_sha256, "R7 runner behavior hash"),
        (receipt.feature_receipt_sha256, "feature receipt hash"),
    ):
        _require_sha256(value, field)
    _require_int(receipt.sentence_count, "sentence count", minimum=TOP_K)
    _validate_top5(receipt.r0_top5, receipt.sentence_count, "R0 output")
    _validate_top5(receipt.r7_top5, receipt.sentence_count, "R7 output")
    for values, field in (
        (receipt.r0_facet_maxima_ints, "R0 facet maxima"),
        (receipt.r7_facet_maxima_ints, "R7 facet maxima"),
    ):
        if (
            not isinstance(values, tuple)
            or len(values) != FACET_COUNT
            or any(type(value) is not int for value in values)
        ):
            raise EraserExactFeatureBridgeError(f"{field} are malformed")
    _require_int(
        receipt.minimum_positive_anchor_strength,
        "minimum positive anchor strength",
        minimum=0,
    )
    _require_int(receipt.r0_dense_relevance_mass_int, "R0 dense mass")
    _require_int(receipt.r7_dense_relevance_mass_int, "R7 dense mass")
    _validate_loo_matrix(
        receipt.r0_leave_one_out_coverage_deltas, "R0 leave-one-out"
    )
    _validate_loo_matrix(
        receipt.r7_leave_one_out_coverage_deltas, "R7 leave-one-out"
    )
    if receipt.edge_deletion_action_change_indicator not in (0, 1):
        raise EraserExactFeatureBridgeError("edge deletion indicator is not binary")
    for values, field in (
        (receipt.r0_pair_similarity_ints, "R0 pair similarities"),
        (receipt.r7_pair_similarity_ints, "R7 pair similarities"),
    ):
        if (
            not isinstance(values, tuple)
            or len(values) != PAIR_COUNT
            or any(
                type(value) is not int
                or not -QUANTIZATION_SCALE <= value <= QUANTIZATION_SCALE
                for value in values
            )
        ):
            raise EraserExactFeatureBridgeError(f"{field} are malformed")
    if (
        not isinstance(receipt.features, tuple)
        or len(receipt.features) != len(FEATURE_ORDER)
        or any(not isinstance(value, Fraction) for value in receipt.features)
        or receipt.features != _features_from_receipt(receipt)
    ):
        raise EraserExactFeatureBridgeError("exact eight-dimensional feature vector drifted")
    if recompute_feature_receipt_sha256(receipt) != receipt.feature_receipt_sha256:
        raise EraserExactFeatureBridgeError("feature computation receipt self hash drifted")
    return receipt.feature_receipt_sha256


@dataclass(frozen=True)
class ExactDifferenceTraceBuild:
    difference_trace: runner.DifferenceTrace
    feature_receipt: ExactFeatureComputationReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.difference_trace, runner.DifferenceTrace):
            raise EraserExactFeatureBridgeError("runner difference trace has wrong type")
        verify_feature_computation_receipt(self.feature_receipt)
        trace = self.difference_trace
        receipt = self.feature_receipt
        if (
            trace.item_commitment_sha256 != receipt.item_commitment_sha256
            or trace.sentence_count != receipt.sentence_count
            or trace.r0_action_trace_sha256 != receipt.r0_action_trace_sha256
            or trace.r7_action_trace_sha256 != receipt.r7_action_trace_sha256
            or trace.r0_behavior_sha256 != receipt.r0_runner_behavior_sha256
            or trace.r7_behavior_sha256 != receipt.r7_runner_behavior_sha256
            or trace.r0_top5 != receipt.r0_top5
            or trace.r7_top5 != receipt.r7_top5
            or trace.features != receipt.features
        ):
            raise EraserExactFeatureBridgeError(
                "runner DifferenceTrace drifted from exact feature receipt"
            )


def _verified_action(
    *,
    action: operator.ActionTrace,
    expected_recipe_id: str,
    graph: operator.QueryAnchoredSentenceGraph,
    semantic_tensor: operator.QuerySemanticTensor,
) -> operator.ActionTrace:
    if not isinstance(action, operator.ActionTrace):
        raise EraserExactFeatureBridgeError("operator action has the wrong type")
    if action.recipe_id != expected_recipe_id:
        raise EraserExactFeatureBridgeError("operator action recipe binding drifted")
    try:
        operator.verify_action_trace(
            action,
            graph=graph,
            semantic_tensor=semantic_tensor,
        )
    except operator.EraserR7OperatorError as exc:
        raise EraserExactFeatureBridgeError("operator action verification failed") from exc
    return action


def build_exact_difference_trace(
    *,
    item_commitment_sha256: str,
    graph: operator.QueryAnchoredSentenceGraph,
    semantic_tensor: operator.QuerySemanticTensor,
    r0_action: operator.ActionTrace,
    r7_action: operator.ActionTrace,
    selected_pair_semantic_receipt: SelectedPairSemanticReceipt,
) -> ExactDifferenceTraceBuild:
    """Derive the frozen exact 8D vector; no external feature map is accepted."""

    _require_sha256(item_commitment_sha256, "item commitment")
    try:
        operator.verify_query_anchored_graph(graph, semantic_tensor)
    except operator.EraserR7OperatorError as exc:
        raise EraserExactFeatureBridgeError("operator graph/tensor verification failed") from exc
    verify_selected_pair_semantic_receipt(
        selected_pair_semantic_receipt,
        graph=graph,
        semantic_tensor=semantic_tensor,
    )
    r0 = _verified_action(
        action=r0_action,
        expected_recipe_id=R0_RECIPE_ID,
        graph=graph,
        semantic_tensor=semantic_tensor,
    )
    r7 = _verified_action(
        action=r7_action,
        expected_recipe_id=R7_RECIPE_ID,
        graph=graph,
        semantic_tensor=semantic_tensor,
    )
    sentence_count = len(graph.units)
    r0_top5 = r0.output_top5
    r7_top5 = r7.output_top5
    _validate_top5(r0_top5, sentence_count, "R0 output")
    _validate_top5(r7_top5, sentence_count, "R7 output")
    if (
        selected_pair_semantic_receipt.r0_top5 != r0_top5
        or selected_pair_semantic_receipt.r7_top5 != r7_top5
    ):
        raise EraserExactFeatureBridgeError(
            "selected-pair receipt is bound to different actions"
        )

    r0_maxima = operator.facet_maxima_ints(semantic_tensor, r0_top5)
    r7_maxima = operator.facet_maxima_ints(semantic_tensor, r7_top5)
    if (
        r0.behavior.selected_facet_maxima_ints != r0_maxima
        or r7.behavior.selected_facet_maxima_ints != r7_maxima
    ):
        raise EraserExactFeatureBridgeError("operator behavior facet maxima drifted")
    anchor_strengths = tuple(
        path.anchor_strength_int
        for step in r7.selection_steps
        for path in step.facet_paths
    )
    minimum_anchor = min(anchor_strengths) if anchor_strengths else 0
    r0_dense_mass = sum(
        semantic_tensor.dense_relevance_ints[ordinal] for ordinal in r0_top5
    )
    r7_dense_mass = sum(
        semantic_tensor.dense_relevance_ints[ordinal] for ordinal in r7_top5
    )
    r0_loo = operator.sentence_leave_one_out_coverage_deltas(
        semantic_tensor, r0_top5
    )
    r7_loo = operator.sentence_leave_one_out_coverage_deltas(
        semantic_tensor, r7_top5
    )
    edge_change = int(
        any(
            witness.selected_ordinals_changed
            for witness in r7.edge_deletion_witnesses
        )
    )
    r0_pairs = _pair_values(selected_pair_semantic_receipt, r0_top5)
    r7_pairs = _pair_values(selected_pair_semantic_receipt, r7_top5)
    r0_runner_behavior = runner.behavior_sha256(
        item_commitment_sha256=item_commitment_sha256,
        recipe_id=R0_RECIPE_ID,
        selected_ordinals=r0_top5,
    )
    r7_runner_behavior = runner.behavior_sha256(
        item_commitment_sha256=item_commitment_sha256,
        recipe_id=R7_RECIPE_ID,
        selected_ordinals=r7_top5,
    )
    receipt = ExactFeatureComputationReceipt(
        item_commitment_sha256=item_commitment_sha256,
        sentence_count=sentence_count,
        graph_sha256=graph.graph_sha256,
        semantic_tensor_sha256=semantic_tensor.tensor_sha256,
        selected_pair_semantic_receipt_sha256=(
            selected_pair_semantic_receipt.receipt_sha256
        ),
        r0_action_trace_sha256=r0.trace_sha256,
        r7_action_trace_sha256=r7.trace_sha256,
        r0_operator_behavior_sha256=r0.behavior_sha256,
        r7_operator_behavior_sha256=r7.behavior_sha256,
        r0_runner_behavior_sha256=r0_runner_behavior,
        r7_runner_behavior_sha256=r7_runner_behavior,
        r0_top5=r0_top5,
        r7_top5=r7_top5,
        r0_facet_maxima_ints=r0_maxima,
        r7_facet_maxima_ints=r7_maxima,
        minimum_positive_anchor_strength=minimum_anchor,
        r0_dense_relevance_mass_int=r0_dense_mass,
        r7_dense_relevance_mass_int=r7_dense_mass,
        r0_leave_one_out_coverage_deltas=r0_loo,
        r7_leave_one_out_coverage_deltas=r7_loo,
        edge_deletion_action_change_indicator=edge_change,
        r0_pair_similarity_ints=r0_pairs,
        r7_pair_similarity_ints=r7_pairs,
        features=(),
        feature_receipt_sha256="0" * 64,
    )
    receipt = replace(receipt, features=_features_from_receipt(receipt))
    receipt = replace(
        receipt,
        feature_receipt_sha256=recompute_feature_receipt_sha256(receipt),
    )
    verify_feature_computation_receipt(receipt)
    difference = runner.DifferenceTrace.from_mapping(
        item_commitment_sha256=item_commitment_sha256,
        sentence_count=sentence_count,
        r0_action_trace_sha256=r0.trace_sha256,
        r7_action_trace_sha256=r7.trace_sha256,
        r0_top5=r0_top5,
        r7_top5=r7_top5,
        features={
            name: receipt.features[index]
            for index, name in enumerate(FEATURE_ORDER)
        },
    )
    return ExactDifferenceTraceBuild(
        difference_trace=difference,
        feature_receipt=receipt,
    )


def verify_exact_difference_trace_build(
    build: ExactDifferenceTraceBuild,
    *,
    item_commitment_sha256: str | None = None,
    graph: operator.QueryAnchoredSentenceGraph | None = None,
    semantic_tensor: operator.QuerySemanticTensor | None = None,
    r0_action: operator.ActionTrace | None = None,
    r7_action: operator.ActionTrace | None = None,
    selected_pair_semantic_receipt: SelectedPairSemanticReceipt | None = None,
) -> str:
    if not isinstance(build, ExactDifferenceTraceBuild):
        raise EraserExactFeatureBridgeError("exact DifferenceTrace build has wrong type")
    build.__post_init__()
    supplied = (
        item_commitment_sha256,
        graph,
        semantic_tensor,
        r0_action,
        r7_action,
        selected_pair_semantic_receipt,
    )
    if any(value is not None for value in supplied):
        if any(value is None for value in supplied):
            raise EraserExactFeatureBridgeError(
                "full reconstruction requires every bridge input"
            )
        assert item_commitment_sha256 is not None
        assert graph is not None and semantic_tensor is not None
        assert r0_action is not None and r7_action is not None
        assert selected_pair_semantic_receipt is not None
        expected = build_exact_difference_trace(
            item_commitment_sha256=item_commitment_sha256,
            graph=graph,
            semantic_tensor=semantic_tensor,
            r0_action=r0_action,
            r7_action=r7_action,
            selected_pair_semantic_receipt=selected_pair_semantic_receipt,
        )
        if build != expected:
            raise EraserExactFeatureBridgeError(
                "exact DifferenceTrace build does not reconstruct from inputs"
            )
    return build.feature_receipt.feature_receipt_sha256


__all__ = [
    "ExactDifferenceTraceBuild",
    "ExactFeatureComputationReceipt",
    "EraserExactFeatureBridgeError",
    "FACET_COUNT",
    "FEATURE_ORDER",
    "MINILM_ASSET_MANIFEST_SHA256",
    "PAIR_COUNT",
    "QUANTIZATION_SCALE",
    "SelectedPairSemanticReceipt",
    "VERSION",
    "build_exact_difference_trace",
    "build_selected_pair_semantic_receipt",
    "recompute_feature_receipt_sha256",
    "recompute_selected_pair_receipt_sha256",
    "stable_hash",
    "verify_exact_difference_trace_build",
    "verify_feature_computation_receipt",
    "verify_selected_pair_semantic_receipt",
]
