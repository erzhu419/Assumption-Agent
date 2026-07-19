"""Pure-offline evaluator and scoring core for ERASER Evidence Inference R7/E3.

The module deliberately has no source reader, action operator, HippoRAG runtime,
controller, network client, or model API.  It consumes only already-verified,
content-free R0/R7 action bindings and exact feature differences.  Labels may be
joined only by :func:`fit_e3` for ``A_form`` or :func:`score_anchor` for the two
late measurement blocks.

The historical HybridQA formal runner remains the trust root for canonical
receipt hashing, exact decimal conversion/linear algebra, and the exact
magnitude-preserving sign-flip implementation.  Action-trace hashes and ordered
top-five behavior hashes are separate commitments: behavior hashes are
recomputed here, while action hashes are carried from the independently
verified operator trace and are never compared with behavior hashes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
import hashlib
import hmac
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks import feverous_e2_evaluator_v1 as evaluator_math
from assumption_agent.benchmarks import (
    hybridqa_query_anchored_formal_runner_v1 as shared,
)


VERSION = "eraser_evidence_inference_r7_e3_runner_v1"
BLOCK_ORDER = ("A_form", "F_search", "A_hold", "M_search")
BLOCK_COUNTS = {"A_form": 48, "F_search": 36, "A_hold": 30, "M_search": 30}
FAMILIES = (
    "SIGNIFICANTLY_DECREASED",
    "NO_SIGNIFICANT_DIFFERENCE",
    "SIGNIFICANTLY_INCREASED",
)
BLOCK_FAMILY_COUNTS = {
    "A_hold": {family: 10 for family in FAMILIES},
    "M_search": {family: 10 for family in FAMILIES},
}
RECIPE_IDS = ("R0_DENSE5", "R7_QUERY_ANCHORED_ATOMIC_PATH_BUNDLE")
PAIRWISE_COMPARISONS = (
    "E3_minus_E0",
    "E3_minus_HippoRAG",
    "E3_minus_RAW",
)
PAIRWISE_BASELINES = {
    "E3_minus_E0": "E0",
    "E3_minus_HippoRAG": "HippoRAG",
    "E3_minus_RAW": "RAW",
}
FEATURE_ORDER = (
    "outside_RAW5_sentence_count",
    "official_ICO_coverage_gain",
    "new_official_ICO_facet_count",
    "minimum_positive_anchor_strength",
    "dense_relevance_mass_delta",
    "deletion_mean_ICO_coverage_drop_delta",
    "edge_deletion_action_change_indicator",
    "negative_pairwise_semantic_redundancy_delta",
)
TOP_K = 5
FOLD_COUNT = 4
RIDGE_LAMBDA = Decimal(1)
DECIMAL_PRECISION = 80
PROMOTION_ALPHA = Fraction(1, 10)


class EraserEvidenceInferenceRunnerError(RuntimeError):
    """A frozen evaluator, receipt, policy, or score contract drifted."""


stable_hash = shared.stable_hash


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, *, field: str) -> str:
    if not _is_sha256(value):
        raise EraserEvidenceInferenceRunnerError(f"{field} is not a lowercase sha256")
    return str(value)


def _canonical_json_text(value: object) -> str:
    try:
        return shared._canonical_json_text(value)
    except shared.HybridQaFormalRunnerError as exc:
        raise EraserEvidenceInferenceRunnerError("value is not canonical JSON") from exc


def _mapping_from_canonical_json(value: str, *, field: str) -> dict[str, Any]:
    try:
        return shared._mapping_from_canonical_json(value, field=field)
    except shared.HybridQaFormalRunnerError as exc:
        raise EraserEvidenceInferenceRunnerError(str(exc)) from exc


def _self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    try:
        return shared._self_hashed(body, field)
    except shared.HybridQaFormalRunnerError as exc:
        raise EraserEvidenceInferenceRunnerError(str(exc)) from exc


def _verify_self_hashed(
    receipt: Mapping[str, Any], *, schema: str, field: str
) -> str:
    try:
        return shared._verify_self_hashed(receipt, schema=schema, field=field)
    except shared.HybridQaFormalRunnerError as exc:
        raise EraserEvidenceInferenceRunnerError(str(exc)) from exc


def _fraction_payload(value: Fraction) -> list[int]:
    if not isinstance(value, Fraction):
        raise EraserEvidenceInferenceRunnerError("value is not an exact Fraction")
    return [value.numerator, value.denominator]


def _fraction_from_payload(value: object, *, field: str) -> Fraction:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(part) is not int for part in value)
        or value[1] <= 0
    ):
        raise EraserEvidenceInferenceRunnerError(f"{field} is not a fraction")
    result = Fraction(value[0], value[1])
    if value != [result.numerator, result.denominator]:
        raise EraserEvidenceInferenceRunnerError(f"{field} is not reduced")
    return result


def _exact_fraction(value: object, *, field: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise EraserEvidenceInferenceRunnerError(
            f"{field} must be an exact integer or Fraction"
        )
    return Fraction(value)


def _to_decimal(value: object, *, field: str) -> Decimal:
    try:
        return evaluator_math._to_decimal(value, field)
    except evaluator_math.FeverousEvaluatorError as exc:
        raise EraserEvidenceInferenceRunnerError(str(exc)) from exc


def _decimal_text(value: Decimal) -> str:
    try:
        return evaluator_math._decimal_text(value)
    except evaluator_math.FeverousEvaluatorError as exc:
        raise EraserEvidenceInferenceRunnerError(str(exc)) from exc


def behavior_sha256(
    *, item_commitment_sha256: str, recipe_id: str, selected_ordinals: Sequence[int]
) -> str:
    """Recompute the content-free ordered-output behavior commitment."""

    _require_sha256(item_commitment_sha256, field="item commitment")
    if recipe_id not in RECIPE_IDS:
        raise EraserEvidenceInferenceRunnerError("recipe is outside the registry")
    selected = tuple(selected_ordinals)
    if (
        len(selected) != TOP_K
        or len(set(selected)) != TOP_K
        or any(type(value) is not int or value < 0 for value in selected)
    ):
        raise EraserEvidenceInferenceRunnerError("behavior output is not a top five")
    return stable_hash(
        {
            "consumer": VERSION,
            "item_commitment_sha256": item_commitment_sha256,
            "ordered_top5": list(selected),
        }
    )


@dataclass(frozen=True)
class DifferenceTrace:
    """One exact, content-free R7-minus-R0 observation for one item."""

    item_commitment_sha256: str
    sentence_count: int
    r0_action_trace_sha256: str
    r7_action_trace_sha256: str
    r0_behavior_sha256: str
    r7_behavior_sha256: str
    r0_top5: tuple[int, ...]
    r7_top5: tuple[int, ...]
    features: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.item_commitment_sha256, field="item commitment")
        _require_sha256(self.r0_action_trace_sha256, field="R0 action trace")
        _require_sha256(self.r7_action_trace_sha256, field="R7 action trace")
        _require_sha256(self.r0_behavior_sha256, field="R0 behavior")
        _require_sha256(self.r7_behavior_sha256, field="R7 behavior")
        if type(self.sentence_count) is not int or self.sentence_count < TOP_K:
            raise EraserEvidenceInferenceRunnerError("sentence count is invalid")
        for recipe_id, selected, declared in (
            (RECIPE_IDS[0], self.r0_top5, self.r0_behavior_sha256),
            (RECIPE_IDS[1], self.r7_top5, self.r7_behavior_sha256),
        ):
            if (
                not isinstance(selected, tuple)
                or len(selected) != TOP_K
                or len(set(selected)) != TOP_K
                or any(
                    type(value) is not int
                    or not 0 <= value < self.sentence_count
                    for value in selected
                )
            ):
                raise EraserEvidenceInferenceRunnerError(
                    f"{recipe_id} output is not an in-corpus top five"
                )
            expected = behavior_sha256(
                item_commitment_sha256=self.item_commitment_sha256,
                recipe_id=recipe_id,
                selected_ordinals=selected,
            )
            if declared != expected:
                raise EraserEvidenceInferenceRunnerError(
                    f"{recipe_id} behavior hash was not independently recomputed"
                )
        if not isinstance(self.features, tuple) or len(self.features) != len(
            FEATURE_ORDER
        ):
            raise EraserEvidenceInferenceRunnerError(
                "difference trace must contain exactly eight features"
            )
        normalized = tuple(
            _exact_fraction(value, field=FEATURE_ORDER[index])
            for index, value in enumerate(self.features)
        )
        discrete = {
            "outside_RAW5_sentence_count": (normalized[0], 0, TOP_K),
            "new_official_ICO_facet_count": (normalized[2], 0, 3),
            "minimum_positive_anchor_strength": (normalized[3], 0, None),
            "dense_relevance_mass_delta": (normalized[4], None, None),
        }
        for name, (value, minimum, maximum) in discrete.items():
            if (
                value.denominator != 1
                or (minimum is not None and value < minimum)
                or (maximum is not None and value > maximum)
            ):
                raise EraserEvidenceInferenceRunnerError(
                    f"{name} violates its exact integer range"
                )
        if normalized[6] not in {Fraction(0), Fraction(1)}:
            raise EraserEvidenceInferenceRunnerError(
                "edge_deletion_action_change_indicator is not binary"
            )
        object.__setattr__(self, "features", normalized)

    @classmethod
    def from_mapping(
        cls,
        *,
        item_commitment_sha256: str,
        sentence_count: int,
        r0_action_trace_sha256: str,
        r7_action_trace_sha256: str,
        r0_top5: Sequence[int],
        r7_top5: Sequence[int],
        features: Mapping[str, object],
    ) -> "DifferenceTrace":
        if set(features) != set(FEATURE_ORDER):
            missing = sorted(set(FEATURE_ORDER) - set(features))
            extra = sorted(set(features) - set(FEATURE_ORDER))
            raise EraserEvidenceInferenceRunnerError(
                f"fixed feature schema drifted; missing={missing}, extra={extra}"
            )
        r0 = tuple(r0_top5)
        r7 = tuple(r7_top5)
        return cls(
            item_commitment_sha256=item_commitment_sha256,
            sentence_count=sentence_count,
            r0_action_trace_sha256=r0_action_trace_sha256,
            r7_action_trace_sha256=r7_action_trace_sha256,
            r0_behavior_sha256=behavior_sha256(
                item_commitment_sha256=item_commitment_sha256,
                recipe_id=RECIPE_IDS[0],
                selected_ordinals=r0,
            ),
            r7_behavior_sha256=behavior_sha256(
                item_commitment_sha256=item_commitment_sha256,
                recipe_id=RECIPE_IDS[1],
                selected_ordinals=r7,
            ),
            r0_top5=r0,
            r7_top5=r7,
            features=tuple(
                _exact_fraction(features[name], field=name) for name in FEATURE_ORDER
            ),
        )

    @property
    def behavior_distinct(self) -> bool:
        return self.r0_behavior_sha256 != self.r7_behavior_sha256

    def payload(self) -> dict[str, object]:
        return {
            "item_commitment_sha256": self.item_commitment_sha256,
            "sentence_count": self.sentence_count,
            "action_trace_sha256": {
                RECIPE_IDS[0]: self.r0_action_trace_sha256,
                RECIPE_IDS[1]: self.r7_action_trace_sha256,
            },
            "behavior_sha256": {
                RECIPE_IDS[0]: self.r0_behavior_sha256,
                RECIPE_IDS[1]: self.r7_behavior_sha256,
            },
            "ordered_top5": {
                RECIPE_IDS[0]: list(self.r0_top5),
                RECIPE_IDS[1]: list(self.r7_top5),
            },
            "R7_minus_R0_features": [
                _fraction_payload(value) for value in self.features
            ],
        }


def _normalize_traces(traces: Sequence[DifferenceTrace]) -> tuple[DifferenceTrace, ...]:
    if not traces or any(not isinstance(trace, DifferenceTrace) for trace in traces):
        raise EraserEvidenceInferenceRunnerError("difference trace matrix is invalid")
    canonical = tuple(sorted(traces, key=lambda trace: trace.item_commitment_sha256))
    if len({trace.item_commitment_sha256 for trace in canonical}) != len(canonical):
        raise EraserEvidenceInferenceRunnerError("item commitment is duplicated")
    return canonical


def build_feature_receipt(
    *, block: str, traces: Sequence[DifferenceTrace]
) -> dict[str, Any]:
    """Build the explicit exact eight-dimensional R7-minus-R0 receipt."""

    if block not in BLOCK_COUNTS:
        raise EraserEvidenceInferenceRunnerError("feature block is invalid")
    canonical = _normalize_traces(traces)
    if len(canonical) != BLOCK_COUNTS[block]:
        raise EraserEvidenceInferenceRunnerError("feature block item count drifted")
    commitments = [trace.item_commitment_sha256 for trace in canonical]
    body = {
        "schema": f"{VERSION}_feature_receipt",
        "version": VERSION,
        "block": block,
        "item_count": len(canonical),
        "trace_count": len(canonical),
        "recipe_registry": list(RECIPE_IDS),
        "feature_basis": "one_exact_R7_minus_R0_vector_per_item",
        "fixed_R7_minus_R0_feature_order": list(FEATURE_ORDER),
        "feature_value_encoding": "reduced_integer_fraction_pair_v1",
        "trace_matrix_sha256": stable_hash([trace.payload() for trace in canonical]),
        "action_trace_matrix_sha256": stable_hash(
            [
                [
                    trace.item_commitment_sha256,
                    trace.r0_action_trace_sha256,
                    trace.r7_action_trace_sha256,
                ]
                for trace in canonical
            ]
        ),
        "behavior_matrix_sha256": stable_hash(
            [
                [
                    trace.item_commitment_sha256,
                    trace.r0_behavior_sha256,
                    trace.r7_behavior_sha256,
                ]
                for trace in canonical
            ]
        ),
        "item_commitment_set_sha256": stable_hash(commitments),
        "action_and_behavior_sha256_equality_required": False,
        "behavior_sha256_recomputed_from_ordered_top5": True,
        "action_trace_sha256_requires_independent_operator_verification": True,
        "labels_utility_family_RAW_or_Hippo_accessed": False,
        "online_evaluator_calls": 0,
        "raw_content_persisted": False,
    }
    return _self_hashed(body, "feature_receipt_sha256")


@dataclass(frozen=True)
class FeatureSeal:
    block: str
    traces: tuple[DifferenceTrace, ...]
    feature_receipt_sha256: str
    trace_matrix_sha256: str
    item_commitment_set_sha256: str

    def __post_init__(self) -> None:
        if self.block not in BLOCK_COUNTS or not isinstance(self.traces, tuple):
            raise EraserEvidenceInferenceRunnerError("feature seal is invalid")
        canonical = _normalize_traces(self.traces)
        if canonical != self.traces:
            raise EraserEvidenceInferenceRunnerError("feature traces are not canonical")
        receipt = build_feature_receipt(block=self.block, traces=self.traces)
        observed = (
            self.feature_receipt_sha256,
            self.trace_matrix_sha256,
            self.item_commitment_set_sha256,
        )
        expected = (
            receipt["feature_receipt_sha256"],
            receipt["trace_matrix_sha256"],
            receipt["item_commitment_set_sha256"],
        )
        if observed != expected:
            raise EraserEvidenceInferenceRunnerError("feature seal binding drifted")

    @property
    def receipt(self) -> dict[str, Any]:
        return build_feature_receipt(block=self.block, traces=self.traces)

    @property
    def item_commitments(self) -> tuple[str, ...]:
        return tuple(trace.item_commitment_sha256 for trace in self.traces)

    @property
    def by_item(self) -> Mapping[str, DifferenceTrace]:
        return {trace.item_commitment_sha256: trace for trace in self.traces}


def seal_feature_matrix(
    *, block: str, traces: Sequence[DifferenceTrace]
) -> FeatureSeal:
    canonical = _normalize_traces(traces)
    receipt = build_feature_receipt(block=block, traces=canonical)
    return FeatureSeal(
        block=block,
        traces=canonical,
        feature_receipt_sha256=receipt["feature_receipt_sha256"],
        trace_matrix_sha256=receipt["trace_matrix_sha256"],
        item_commitment_set_sha256=receipt["item_commitment_set_sha256"],
    )


E3Model = evaluator_math.E2Model


def _population_scaler(
    traces: Sequence[DifferenceTrace],
) -> tuple[tuple[Decimal, ...], tuple[Decimal, ...]]:
    if not traces:
        raise EraserEvidenceInferenceRunnerError("cannot fit an empty scaler")
    rows = tuple(
        tuple(_to_decimal(value, field="feature") for value in trace.features)
        for trace in traces
    )
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        count = Decimal(len(rows))
        means = tuple(
            sum((row[index] for row in rows), Decimal(0)) / count
            for index in range(len(FEATURE_ORDER))
        )
        variances = tuple(
            sum(
                ((row[index] - means[index]) ** 2 for row in rows), Decimal(0)
            )
            / count
            for index in range(len(FEATURE_ORDER))
        )
        stds = tuple(
            Decimal(0) if variance == 0 else context.sqrt(variance)
            for variance in variances
        )
    return means, stds


def _fit_model(
    traces: Sequence[DifferenceTrace], utilities: Mapping[str, Fraction]
) -> E3Model:
    means, stds = _population_scaler(traces)
    width = len(FEATURE_ORDER)
    gram = [[Decimal(0) for _ in range(width)] for _ in range(width)]
    rhs = [Decimal(0) for _ in range(width)]
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        for trace in traces:
            x = tuple(
                Decimal(0)
                if stds[index] == 0
                else (
                    _to_decimal(trace.features[index], field=FEATURE_ORDER[index])
                    - means[index]
                )
                / stds[index]
                for index in range(width)
            )
            delta = utilities[trace.item_commitment_sha256]
            y = Decimal(delta.numerator) / Decimal(delta.denominator)
            for left in range(width):
                rhs[left] += x[left] * y
                for right in range(width):
                    gram[left][right] += x[left] * x[right]
        for index in range(width):
            gram[index][index] += RIDGE_LAMBDA
        try:
            beta = evaluator_math._solve_linear_system(gram, rhs)
        except evaluator_math.FeverousEvaluatorError as exc:
            raise EraserEvidenceInferenceRunnerError("ridge fit failed") from exc
    return E3Model(means, stds, beta)


def _balanced_fold_assignment(items: Sequence[str], secret: bytes) -> dict[str, int]:
    if not isinstance(secret, bytes) or len(secret) != 32:
        raise EraserEvidenceInferenceRunnerError(
            "fold secret must contain exactly 32 bytes"
        )
    rows = tuple(items)
    if len(rows) != len(set(rows)) or any(not _is_sha256(item) for item in rows):
        raise EraserEvidenceInferenceRunnerError("fold commitments are invalid")
    ordered = sorted(
        rows,
        key=lambda item: (
            hmac.new(
                secret,
                f"{VERSION}:A_form:balanced_fold:{item}".encode("ascii"),
                hashlib.sha256,
            ).digest(),
            item,
        ),
    )
    return {item: rank % FOLD_COUNT for rank, item in enumerate(ordered)}


def _crossfit_diagnostic(
    *,
    model: E3Model,
    traces: Sequence[DifferenceTrace],
    utilities: Mapping[str, Fraction],
) -> dict[str, Any]:
    correct_direction = 0
    nonzero_utility = 0
    squared_error = Decimal(0)
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        for trace in traces:
            prediction = model.predict(trace.features)
            actual = utilities[trace.item_commitment_sha256]
            actual_decimal = Decimal(actual.numerator) / Decimal(actual.denominator)
            squared_error += (prediction - actual_decimal) ** 2
            if actual:
                nonzero_utility += 1
                correct_direction += int((prediction > 0) == (actual > 0))
        mse = squared_error / Decimal(len(traces))
    return {
        "held_item_count": len(traces),
        "held_nonzero_utility_count": nonzero_utility,
        "held_correct_direction_count": correct_direction,
        "held_mean_squared_error": _decimal_text(mse),
    }


@dataclass(frozen=True)
class E3FitSeal:
    a_form_features: FeatureSeal
    model: E3Model
    receipt_json: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.a_form_features, FeatureSeal)
            or self.a_form_features.block != "A_form"
            or not isinstance(self.model, E3Model)
        ):
            raise EraserEvidenceInferenceRunnerError("E3 fit dependencies are invalid")
        receipt = _mapping_from_canonical_json(self.receipt_json, field="E3 fit receipt")
        _verify_self_hashed(
            receipt,
            schema=f"{VERSION}_e3_fit_receipt",
            field="fit_receipt_sha256",
        )
        required = {
            "version": VERSION,
            "block": "A_form",
            "feature_receipt_sha256": self.a_form_features.feature_receipt_sha256,
            "trace_matrix_sha256": self.a_form_features.trace_matrix_sha256,
            "item_commitment_set_sha256": (
                self.a_form_features.item_commitment_set_sha256
            ),
            "item_count": BLOCK_COUNTS["A_form"],
            "observation_count": BLOCK_COUNTS["A_form"],
            "observation": "one_R7_minus_R0_feature_and_utility_delta_per_item",
            "ridge_lambda": "1",
            "intercept": False,
            "scaler": "A_form_population_mean_population_std_zero_variance_to_zero",
            "decimal_contract": {
                "precision": DECIMAL_PRECISION,
                "rounding": "ROUND_HALF_EVEN",
                "binary_float_inputs": False,
            },
            "fold_count": FOLD_COUNT,
            "fold_policy": "private_HMAC_SHA256_order_then_balanced_rank_mod_4_v1",
            "crossfit_descriptive_only": True,
            "final_fit_count": 1,
            "model": self.model.payload(),
            "utility_values_persisted": False,
            "F_search_accessed": False,
            "A_hold_or_M_search_accessed": False,
            "online_evaluator_calls": 0,
            "raw_content_persisted": False,
        }
        if any(receipt.get(key) != value for key, value in required.items()):
            raise EraserEvidenceInferenceRunnerError("E3 fit receipt semantics drifted")
        for field in ("utility_delta_matrix_sha256", "fold_assignment_sha256"):
            _require_sha256(receipt.get(field), field=field)
        crossfit = receipt.get("crossfit")
        if (
            not isinstance(crossfit, list)
            or len(crossfit) != FOLD_COUNT
            or [row.get("fold") for row in crossfit] != list(range(FOLD_COUNT))
            or any(row.get("fit_item_count") != 36 for row in crossfit)
            or any(row.get("held_item_count") != 12 for row in crossfit)
        ):
            raise EraserEvidenceInferenceRunnerError("crossfit receipt drifted")

    @property
    def receipt(self) -> dict[str, Any]:
        return _mapping_from_canonical_json(self.receipt_json, field="E3 fit receipt")

    @property
    def fit_receipt_sha256(self) -> str:
        return str(self.receipt["fit_receipt_sha256"])


def fit_e3(
    *,
    feature_seal: FeatureSeal,
    utility_deltas: Mapping[str, Fraction | int],
    fold_secret: bytes,
) -> E3FitSeal:
    """Fit exactly one no-intercept lambda-one E3 model on sealed A_form."""

    if not isinstance(feature_seal, FeatureSeal) or feature_seal.block != "A_form":
        raise EraserEvidenceInferenceRunnerError("A_form must be feature-sealed")
    commitments = feature_seal.item_commitments
    if set(utility_deltas) != set(commitments):
        raise EraserEvidenceInferenceRunnerError("utility delta alignment drifted")
    utilities = {
        item: _exact_fraction(utility_deltas[item], field="utility delta")
        for item in commitments
    }
    if any(not -2 <= value <= 2 for value in utilities.values()):
        raise EraserEvidenceInferenceRunnerError("utility delta is outside [-2, 2]")
    fold_by_item = _balanced_fold_assignment(commitments, fold_secret)
    diagnostics: list[dict[str, Any]] = []
    for fold in range(FOLD_COUNT):
        fit_rows = tuple(
            trace
            for trace in feature_seal.traces
            if fold_by_item[trace.item_commitment_sha256] != fold
        )
        held_rows = tuple(
            trace
            for trace in feature_seal.traces
            if fold_by_item[trace.item_commitment_sha256] == fold
        )
        model = _fit_model(fit_rows, utilities)
        diagnostic = _crossfit_diagnostic(
            model=model, traces=held_rows, utilities=utilities
        )
        diagnostics.append(
            {"fold": fold, "fit_item_count": len(fit_rows), **diagnostic}
        )
    final_model = _fit_model(feature_seal.traces, utilities)
    assignment_payload = [[item, fold_by_item[item]] for item in commitments]
    body = {
        "schema": f"{VERSION}_e3_fit_receipt",
        "version": VERSION,
        "block": "A_form",
        "feature_receipt_sha256": feature_seal.feature_receipt_sha256,
        "trace_matrix_sha256": feature_seal.trace_matrix_sha256,
        "item_commitment_set_sha256": feature_seal.item_commitment_set_sha256,
        "item_count": len(commitments),
        "observation_count": len(commitments),
        "observation": "one_R7_minus_R0_feature_and_utility_delta_per_item",
        "ridge_lambda": "1",
        "intercept": False,
        "scaler": "A_form_population_mean_population_std_zero_variance_to_zero",
        "decimal_contract": {
            "precision": DECIMAL_PRECISION,
            "rounding": "ROUND_HALF_EVEN",
            "binary_float_inputs": False,
        },
        "fold_count": FOLD_COUNT,
        "fold_policy": "private_HMAC_SHA256_order_then_balanced_rank_mod_4_v1",
        "fold_assignment_sha256": stable_hash(assignment_payload),
        "crossfit_descriptive_only": True,
        "crossfit": diagnostics,
        "final_fit_count": 1,
        "model": final_model.payload(),
        "utility_delta_matrix_sha256": stable_hash(
            [[item, _fraction_payload(utilities[item])] for item in commitments]
        ),
        "utility_values_persisted": False,
        "F_search_accessed": False,
        "A_hold_or_M_search_accessed": False,
        "online_evaluator_calls": 0,
        "raw_content_persisted": False,
    }
    receipt = _self_hashed(body, "fit_receipt_sha256")
    return E3FitSeal(
        a_form_features=feature_seal,
        model=final_model,
        receipt_json=_canonical_json_text(receipt),
    )


def route_e3(model: E3Model, trace: DifferenceTrace) -> str:
    if not isinstance(model, E3Model) or not isinstance(trace, DifferenceTrace):
        raise EraserEvidenceInferenceRunnerError("E3 route input is invalid")
    prediction = model.predict(trace.features)
    return RECIPE_IDS[1] if prediction.is_finite() and prediction > 0 else RECIPE_IDS[0]


@dataclass(frozen=True)
class PolicySeal:
    f_search_features: FeatureSeal
    fit: E3FitSeal
    receipt_json: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.f_search_features, FeatureSeal)
            or self.f_search_features.block != "F_search"
            or not isinstance(self.fit, E3FitSeal)
        ):
            raise EraserEvidenceInferenceRunnerError("policy dependencies are invalid")
        receipt = _mapping_from_canonical_json(
            self.receipt_json, field="F policy receipt"
        )
        _verify_self_hashed(
            receipt,
            schema=f"{VERSION}_policy_receipt",
            field="policy_receipt_sha256",
        )
        routes = tuple(
            (
                trace.item_commitment_sha256,
                RECIPE_IDS[0],
                route_e3(self.fit.model, trace),
            )
            for trace in self.f_search_features.traces
        )
        counts = Counter(route[2] for route in routes)
        distinct_r7 = sum(
            route[2] == RECIPE_IDS[1] and trace.behavior_distinct
            for route, trace in zip(routes, self.f_search_features.traces, strict=True)
        )
        required = {
            "version": VERSION,
            "block": "F_search",
            "F_feature_receipt_sha256": (
                self.f_search_features.feature_receipt_sha256
            ),
            "A_form_feature_receipt_sha256": (
                self.fit.a_form_features.feature_receipt_sha256
            ),
            "fit_receipt_sha256": self.fit.fit_receipt_sha256,
            "item_commitment_set_sha256": (
                self.f_search_features.item_commitment_set_sha256
            ),
            "item_count": BLOCK_COUNTS["F_search"],
            "E0_routing": "always_R0_DENSE5",
            "E3_routing": "R7_iff_frozen_prediction_strictly_positive_else_R0",
            "route_matrix_sha256": stable_hash([list(row) for row in routes]),
            "E3_route_counts": {
                recipe: counts.get(recipe, 0) for recipe in RECIPE_IDS
            },
            "behavior_distinct_R7_route_count": distinct_r7,
            "labels_gold_utility_family_RAW_or_Hippo_accessed": False,
            "A_hold_authorized": True,
            "M_search_authorized_before_A_hold_promotion": False,
            "online_evaluator_calls": 0,
            "raw_content_persisted": False,
        }
        if any(receipt.get(key) != value for key, value in required.items()):
            raise EraserEvidenceInferenceRunnerError("F policy semantics drifted")

    @property
    def receipt(self) -> dict[str, Any]:
        return _mapping_from_canonical_json(
            self.receipt_json, field="F policy receipt"
        )

    @property
    def policy_receipt_sha256(self) -> str:
        return str(self.receipt["policy_receipt_sha256"])

    def e3_recipe_for(self, trace: DifferenceTrace) -> str:
        return route_e3(self.fit.model, trace)


def freeze_f_policy(
    *, feature_seal: FeatureSeal, fit_seal: E3FitSeal
) -> PolicySeal:
    if (
        not isinstance(feature_seal, FeatureSeal)
        or feature_seal.block != "F_search"
        or not isinstance(fit_seal, E3FitSeal)
    ):
        raise EraserEvidenceInferenceRunnerError("F policy inputs are invalid")
    if set(feature_seal.item_commitments).intersection(
        fit_seal.a_form_features.item_commitments
    ):
        raise EraserEvidenceInferenceRunnerError("A_form and F_search overlap")
    routes = tuple(
        (
            trace.item_commitment_sha256,
            RECIPE_IDS[0],
            route_e3(fit_seal.model, trace),
        )
        for trace in feature_seal.traces
    )
    counts = Counter(route[2] for route in routes)
    distinct_r7 = sum(
        route[2] == RECIPE_IDS[1] and trace.behavior_distinct
        for route, trace in zip(routes, feature_seal.traces, strict=True)
    )
    body = {
        "schema": f"{VERSION}_policy_receipt",
        "version": VERSION,
        "block": "F_search",
        "F_feature_receipt_sha256": feature_seal.feature_receipt_sha256,
        "A_form_feature_receipt_sha256": (
            fit_seal.a_form_features.feature_receipt_sha256
        ),
        "fit_receipt_sha256": fit_seal.fit_receipt_sha256,
        "item_commitment_set_sha256": feature_seal.item_commitment_set_sha256,
        "item_count": len(feature_seal.traces),
        "E0_routing": "always_R0_DENSE5",
        "E3_routing": "R7_iff_frozen_prediction_strictly_positive_else_R0",
        "route_matrix_sha256": stable_hash([list(row) for row in routes]),
        "E3_route_counts": {recipe: counts.get(recipe, 0) for recipe in RECIPE_IDS},
        "behavior_distinct_R7_route_count": distinct_r7,
        "labels_gold_utility_family_RAW_or_Hippo_accessed": False,
        "A_hold_authorized": True,
        "M_search_authorized_before_A_hold_promotion": False,
        "online_evaluator_calls": 0,
        "raw_content_persisted": False,
    }
    receipt = _self_hashed(body, "policy_receipt_sha256")
    return PolicySeal(
        f_search_features=feature_seal,
        fit=fit_seal,
        receipt_json=_canonical_json_text(receipt),
    )


def freeze_f_policies(
    *, feature_seal: FeatureSeal, fit_seal: E3FitSeal
) -> PolicySeal:
    """Compatibility spelling matching the earlier formal runner API."""

    return freeze_f_policy(feature_seal=feature_seal, fit_seal=fit_seal)


@dataclass(frozen=True)
class AnchorLabel:
    item_commitment_sha256: str
    gold_ordinals: tuple[int, ...]
    family: str

    def __post_init__(self) -> None:
        _require_sha256(self.item_commitment_sha256, field="label commitment")
        if (
            not isinstance(self.gold_ordinals, tuple)
            or not self.gold_ordinals
            or len(set(self.gold_ordinals)) != len(self.gold_ordinals)
            or any(type(value) is not int or value < 0 for value in self.gold_ordinals)
        ):
            raise EraserEvidenceInferenceRunnerError(
                "flattened gold sentence union is invalid"
            )
        if self.family not in FAMILIES:
            raise EraserEvidenceInferenceRunnerError("relation family drifted")


@dataclass(frozen=True)
class HippoRetrieval:
    item_commitment_sha256: str
    sentence_count: int
    top5: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.item_commitment_sha256, field="Hippo commitment")
        if (
            type(self.sentence_count) is not int
            or self.sentence_count < TOP_K
            or not isinstance(self.top5, tuple)
            or len(self.top5) != TOP_K
            or len(set(self.top5)) != TOP_K
            or any(
                type(value) is not int or not 0 <= value < self.sentence_count
                for value in self.top5
            )
        ):
            raise EraserEvidenceInferenceRunnerError(
                "HippoRAG result is not an in-corpus top five"
            )

    def payload(self) -> list[object]:
        return [self.item_commitment_sha256, self.sentence_count, list(self.top5)]


@dataclass(frozen=True)
class HippoRetrievalSeal:
    block: str
    rows: tuple[HippoRetrieval, ...]
    retrieval_matrix_sha256: str
    item_commitment_set_sha256: str

    def __post_init__(self) -> None:
        if self.block not in {"A_hold", "M_search"}:
            raise EraserEvidenceInferenceRunnerError("Hippo block is invalid")
        if (
            not isinstance(self.rows, tuple)
            or len(self.rows) != BLOCK_COUNTS[self.block]
            or any(not isinstance(row, HippoRetrieval) for row in self.rows)
            or self.rows
            != tuple(sorted(self.rows, key=lambda row: row.item_commitment_sha256))
            or len({row.item_commitment_sha256 for row in self.rows}) != len(self.rows)
        ):
            raise EraserEvidenceInferenceRunnerError("Hippo matrix drifted")
        payload = [row.payload() for row in self.rows]
        commitments = [row.item_commitment_sha256 for row in self.rows]
        if (
            self.retrieval_matrix_sha256 != stable_hash(payload)
            or self.item_commitment_set_sha256 != stable_hash(commitments)
        ):
            raise EraserEvidenceInferenceRunnerError("Hippo seal binding drifted")

    @property
    def by_item(self) -> Mapping[str, HippoRetrieval]:
        return {row.item_commitment_sha256: row for row in self.rows}


def seal_hippo_retrievals(
    *, block: str, rows: Sequence[HippoRetrieval]
) -> HippoRetrievalSeal:
    if any(not isinstance(row, HippoRetrieval) for row in rows):
        raise EraserEvidenceInferenceRunnerError("Hippo rows contain a foreign type")
    canonical = tuple(sorted(rows, key=lambda row: row.item_commitment_sha256))
    return HippoRetrievalSeal(
        block=block,
        rows=canonical,
        retrieval_matrix_sha256=stable_hash([row.payload() for row in canonical]),
        item_commitment_set_sha256=stable_hash(
            [row.item_commitment_sha256 for row in canonical]
        ),
    )


def item_utility(
    selected: Sequence[int], gold_union: Sequence[int]
) -> tuple[Fraction, bool]:
    """Exact flattened-union utility: recall plus complete-union bonus."""

    output = tuple(selected)
    gold = tuple(gold_union)
    if (
        len(output) != TOP_K
        or len(set(output)) != TOP_K
        or not gold
        or len(set(gold)) != len(gold)
        or any(type(value) is not int or value < 0 for value in (*output, *gold))
    ):
        raise EraserEvidenceInferenceRunnerError("utility input is invalid")
    overlap = len(set(output).intersection(gold))
    complete = overlap == len(gold)
    return Fraction(overlap, len(gold)) + int(complete), complete


def _sign_fraction(raw: object, *, field: str) -> Fraction:
    if (
        not isinstance(raw, Mapping)
        or set(raw) != {"numerator", "denominator"}
        or type(raw.get("numerator")) is not int
        or type(raw.get("denominator")) is not int
        or raw["denominator"] <= 0
    ):
        raise EraserEvidenceInferenceRunnerError(f"{field} is not a fraction")
    result = Fraction(raw["numerator"], raw["denominator"])
    if dict(raw) != {
        "numerator": result.numerator,
        "denominator": result.denominator,
    }:
        raise EraserEvidenceInferenceRunnerError(f"{field} is not reduced")
    return result


def _sign_flip_payload(deltas: Sequence[Fraction]) -> dict[str, Any]:
    try:
        result = evaluator_math.exact_magnitude_preserving_sign_flip(deltas)
    except evaluator_math.FeverousEvaluatorError as exc:
        raise EraserEvidenceInferenceRunnerError("exact sign-flip failed") from exc
    payload = result.payload()
    payload["test"] = "eraser_one_sided_exact_magnitude_preserving_sign_flip_v1"
    payload["consumer"] = VERSION
    return payload


def _validate_sign_flip_payload(
    value: object, *, field: str, item_count: int
) -> tuple[Fraction, Fraction, bool]:
    expected_keys = {
        "test",
        "consumer",
        "observed_net_U",
        "nonzero_pair_count",
        "p_value",
        "alpha",
        "positive_observed_net",
        "exact_p_at_or_below_alpha",
        "promoted",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise EraserEvidenceInferenceRunnerError(f"{field} sign-flip schema drifted")
    net = _sign_fraction(value["observed_net_U"], field=f"{field} net")
    exact_p = _sign_fraction(value["p_value"], field=f"{field} p")
    alpha = _sign_fraction(value["alpha"], field=f"{field} alpha")
    nonzero = value["nonzero_pair_count"]
    promoted = value["promoted"]
    if (
        value["test"]
        != "eraser_one_sided_exact_magnitude_preserving_sign_flip_v1"
        or value["consumer"] != VERSION
        or type(nonzero) is not int
        or not 0 <= nonzero <= item_count
        or not 0 <= exact_p <= 1
        or alpha != PROMOTION_ALPHA
        or type(value["positive_observed_net"]) is not bool
        or value["positive_observed_net"] is not (net > 0)
        or type(value["exact_p_at_or_below_alpha"]) is not bool
        or value["exact_p_at_or_below_alpha"] is not (exact_p <= alpha)
        or type(promoted) is not bool
        or promoted is not (net > 0 and exact_p <= alpha)
    ):
        raise EraserEvidenceInferenceRunnerError(
            f"{field} sign-flip semantics drifted"
        )
    return net, exact_p, promoted


def _route_and_action_aggregates(
    *, traces: Sequence[DifferenceTrace], policy: PolicySeal
) -> tuple[dict[str, str], dict[str, int], int, dict[str, Any]]:
    """Recompute frozen E3 routes and label-free R7 action aggregates."""

    if not isinstance(policy, PolicySeal):
        raise EraserEvidenceInferenceRunnerError("route aggregate policy is invalid")
    canonical = _normalize_traces(traces)
    routes = {
        trace.item_commitment_sha256: policy.e3_recipe_for(trace)
        for trace in canonical
    }
    route_counts = Counter({recipe: 0 for recipe in RECIPE_IDS})
    for recipe in routes.values():
        route_counts[recipe] += 1
    candidate_outside = sum((trace.features[0] for trace in canonical), Fraction(0))
    candidate_edge_change = sum(int(trace.features[6]) for trace in canonical)
    candidate_distinct = sum(trace.behavior_distinct for trace in canonical)
    activated = tuple(
        trace
        for trace in canonical
        if routes[trace.item_commitment_sha256] == RECIPE_IDS[1]
    )
    activated_outside = sum(
        (trace.features[0] for trace in activated), Fraction(0)
    )
    activated_edge_change = sum(int(trace.features[6]) for trace in activated)
    activated_distinct = sum(trace.behavior_distinct for trace in activated)
    aggregates = {
        "R7_candidate_item_count": len(canonical),
        "R7_candidate_behavior_distinct_count": candidate_distinct,
        "R7_candidate_outside_RAW5_sentence_count_sum": _fraction_payload(
            candidate_outside
        ),
        "R7_candidate_edge_deletion_action_change_count": candidate_edge_change,
        "E3_activated_R7_item_count": len(activated),
        "E3_activated_behavior_distinct_R7_count": activated_distinct,
        "E3_activated_outside_RAW5_sentence_count_sum": _fraction_payload(
            activated_outside
        ),
        "E3_activated_edge_deletion_action_change_count": activated_edge_change,
    }
    return routes, dict(route_counts), activated_distinct, aggregates


@dataclass(frozen=True)
class AnchorScoreSeal:
    block: str
    anchor_features: FeatureSeal
    hippo_retrievals: HippoRetrievalSeal
    policies: PolicySeal
    a_hold_authorization: "AnchorScoreSeal | None"
    receipt_json: str

    def __post_init__(self) -> None:
        if (
            self.block not in {"A_hold", "M_search"}
            or not isinstance(self.anchor_features, FeatureSeal)
            or self.anchor_features.block != self.block
            or not isinstance(self.hippo_retrievals, HippoRetrievalSeal)
            or self.hippo_retrievals.block != self.block
            or not isinstance(self.policies, PolicySeal)
        ):
            raise EraserEvidenceInferenceRunnerError("anchor dependencies are invalid")
        if self.block == "A_hold":
            if self.a_hold_authorization is not None:
                raise EraserEvidenceInferenceRunnerError(
                    "A_hold cannot carry M authorization"
                )
            authorization_sha = None
        else:
            if (
                not isinstance(self.a_hold_authorization, AnchorScoreSeal)
                or self.a_hold_authorization.block != "A_hold"
                or not self.a_hold_authorization.evaluator_promoted
                or self.a_hold_authorization.policies.policy_receipt_sha256
                != self.policies.policy_receipt_sha256
            ):
                raise EraserEvidenceInferenceRunnerError(
                    "M_search lacks promoted policy-matched A_hold authorization"
                )
            authorization_sha = self.a_hold_authorization.score_receipt_sha256
        receipt = _mapping_from_canonical_json(
            self.receipt_json, field=f"{self.block} score receipt"
        )
        _verify_self_hashed(
            receipt,
            schema=f"{VERSION}_{self.block}_score_receipt",
            field="score_receipt_sha256",
        )
        required = {
            "version": VERSION,
            "block": self.block,
            "item_count": BLOCK_COUNTS[self.block],
            "logical_RAW_HippoRAG_Agent_work_units": 3 * BLOCK_COUNTS[self.block],
            "anchor_feature_receipt_sha256": (
                self.anchor_features.feature_receipt_sha256
            ),
            "policy_receipt_sha256": self.policies.policy_receipt_sha256,
            "hipporag_retrieval_matrix_sha256": (
                self.hippo_retrievals.retrieval_matrix_sha256
            ),
            "item_commitment_set_sha256": (
                self.anchor_features.item_commitment_set_sha256
            ),
            "A_hold_authorization_score_receipt_sha256": authorization_sha,
            "E0_routing": "always_R0_DENSE5",
            "family_item_counts": BLOCK_FAMILY_COUNTS[self.block],
            "item_level_utility_values_persisted": False,
            "online_evaluator_calls": 0,
            "raw_content_persisted": False,
        }
        if any(receipt.get(key) != value for key, value in required.items()):
            raise EraserEvidenceInferenceRunnerError("anchor receipt semantics drifted")
        expected_keys = {
            "schema",
            "version",
            "block",
            "item_count",
            "logical_RAW_HippoRAG_Agent_work_units",
            "anchor_feature_receipt_sha256",
            "policy_receipt_sha256",
            "hipporag_retrieval_matrix_sha256",
            "item_commitment_set_sha256",
            "late_opened_label_matrix_sha256",
            "A_hold_authorization_score_receipt_sha256",
            "E0_routing",
            "E3_route_counts",
            "behavior_distinct_R7_route_count",
            "R7_action_aggregates",
            "E3_minus_E0",
            "E3_minus_HippoRAG",
            "E3_minus_RAW",
            "pairwise_total_U",
            "pairwise_family_sums",
            "family_item_counts",
            "complete_counts",
            "complete_counts_by_family",
            "pairwise_complete_count_deltas",
            "pairwise_complete_count_deltas_by_family",
            "Hippo_cross_relation_passed",
            "RAW_block_passed",
            "A_hold_real_domain_primary_passed",
            "evaluator_promoted",
            "M_L5_passed",
            "cross_relation_stability_passed",
            "RAW_advantage_overcome",
            "item_level_utility_values_persisted",
            "online_evaluator_calls",
            "raw_content_persisted",
            "score_receipt_sha256",
        }
        if set(receipt) != expected_keys:
            raise EraserEvidenceInferenceRunnerError(
                "anchor score receipt key schema drifted"
            )
        route_counts = receipt.get("E3_route_counts")
        (
            _expected_routes,
            expected_route_counts,
            expected_distinct_routes,
            expected_action_aggregates,
        ) = _route_and_action_aggregates(
            traces=self.anchor_features.traces, policy=self.policies
        )
        observed_action_aggregates = receipt.get("R7_action_aggregates")
        if (
            not isinstance(route_counts, Mapping)
            or set(route_counts) != set(RECIPE_IDS)
            or any(type(value) is not int or value < 0 for value in route_counts.values())
            or sum(route_counts.values()) != BLOCK_COUNTS[self.block]
            or dict(route_counts) != expected_route_counts
            or not isinstance(observed_action_aggregates, Mapping)
            or stable_hash(observed_action_aggregates)
            != stable_hash(expected_action_aggregates)
        ):
            raise EraserEvidenceInferenceRunnerError("E3 route counts drifted")
        _require_sha256(
            receipt.get("late_opened_label_matrix_sha256"),
            field="late-opened label matrix",
        )
        e0_net, _e0_p, e0_passed = _validate_sign_flip_payload(
            receipt.get("E3_minus_E0"), field="E3-minus-E0", item_count=30
        )
        hippo_net, _hippo_p, hippo_passed = _validate_sign_flip_payload(
            receipt.get("E3_minus_HippoRAG"),
            field="E3-minus-HippoRAG",
            item_count=30,
        )
        raw_net, _raw_p, raw_passed = _validate_sign_flip_payload(
            receipt.get("E3_minus_RAW"), field="E3-minus-RAW", item_count=30
        )
        pairwise_nets = {
            "E3_minus_E0": e0_net,
            "E3_minus_HippoRAG": hippo_net,
            "E3_minus_RAW": raw_net,
        }
        total_payload = receipt.get("pairwise_total_U")
        if not isinstance(total_payload, Mapping) or set(total_payload) != set(
            PAIRWISE_COMPARISONS
        ):
            raise EraserEvidenceInferenceRunnerError("pairwise totals drifted")
        totals = {
            comparison: _fraction_from_payload(
                total_payload[comparison], field=f"{comparison} total"
            )
            for comparison in PAIRWISE_COMPARISONS
        }
        if totals != pairwise_nets:
            raise EraserEvidenceInferenceRunnerError(
                "pairwise totals disagree with sign-flip nets"
            )
        family_payload = receipt.get("pairwise_family_sums")
        if not isinstance(family_payload, Mapping) or set(family_payload) != set(
            PAIRWISE_COMPARISONS
        ):
            raise EraserEvidenceInferenceRunnerError("pairwise family schema drifted")
        family_sums: dict[str, dict[str, Fraction]] = {}
        for comparison in PAIRWISE_COMPARISONS:
            row = family_payload[comparison]
            if not isinstance(row, Mapping) or set(row) != set(FAMILIES):
                raise EraserEvidenceInferenceRunnerError(
                    f"{comparison} family schema drifted"
                )
            family_sums[comparison] = {
                family: _fraction_from_payload(
                    row[family], field=f"{comparison} {family} family sum"
                )
                for family in FAMILIES
            }
            if sum(family_sums[comparison].values(), Fraction(0)) != totals[
                comparison
            ]:
                raise EraserEvidenceInferenceRunnerError(
                    f"{comparison} family sums do not partition its total"
                )
        complete = receipt.get("complete_counts")
        if (
            not isinstance(complete, Mapping)
            or set(complete) != {"E0", "E3", "HippoRAG", "RAW"}
            or any(type(value) is not int or not 0 <= value <= 30 for value in complete.values())
        ):
            raise EraserEvidenceInferenceRunnerError("complete counts drifted")
        complete_by_family = receipt.get("complete_counts_by_family")
        if not isinstance(complete_by_family, Mapping) or set(
            complete_by_family
        ) != {"E0", "E3", "HippoRAG", "RAW"}:
            raise EraserEvidenceInferenceRunnerError(
                "complete-by-family schema drifted"
            )
        for arm, row in complete_by_family.items():
            if (
                not isinstance(row, Mapping)
                or set(row) != set(FAMILIES)
                or any(type(value) is not int or not 0 <= value <= 10 for value in row.values())
                or sum(row.values()) != complete[arm]
            ):
                raise EraserEvidenceInferenceRunnerError(
                    f"{arm} complete-by-family counts drifted"
                )
        expected_complete_deltas = {
            comparison: complete["E3"] - complete[baseline]
            for comparison, baseline in PAIRWISE_BASELINES.items()
        }
        observed_complete_deltas = receipt.get("pairwise_complete_count_deltas")
        if (
            not isinstance(observed_complete_deltas, Mapping)
            or set(observed_complete_deltas) != set(PAIRWISE_COMPARISONS)
            or any(
                type(value) is not int or not -30 <= value <= 30
                for value in observed_complete_deltas.values()
            )
            or dict(observed_complete_deltas) != expected_complete_deltas
        ):
            raise EraserEvidenceInferenceRunnerError(
                "pairwise complete-count deltas drifted"
            )
        expected_complete_family_deltas = {
            comparison: {
                family: complete_by_family["E3"][family]
                - complete_by_family[baseline][family]
                for family in FAMILIES
            }
            for comparison, baseline in PAIRWISE_BASELINES.items()
        }
        observed_family_complete_deltas = receipt.get(
            "pairwise_complete_count_deltas_by_family"
        )
        if not isinstance(observed_family_complete_deltas, Mapping) or set(
            observed_family_complete_deltas
        ) != set(PAIRWISE_COMPARISONS):
            raise EraserEvidenceInferenceRunnerError(
                "pairwise family complete-count delta schema drifted"
            )
        for comparison, row in observed_family_complete_deltas.items():
            if (
                not isinstance(row, Mapping)
                or set(row) != set(FAMILIES)
                or any(
                    type(value) is not int or not -10 <= value <= 10
                    for value in row.values()
                )
            ):
                raise EraserEvidenceInferenceRunnerError(
                    f"{comparison} family complete-count deltas drifted"
                )
        if dict(observed_family_complete_deltas) != expected_complete_family_deltas:
            raise EraserEvidenceInferenceRunnerError(
                "pairwise family complete-count deltas drifted"
            )
        distinct_routes = receipt.get("behavior_distinct_R7_route_count")
        if (
            type(distinct_routes) is not int
            or not 0 <= distinct_routes <= route_counts[RECIPE_IDS[1]]
            or distinct_routes != expected_distinct_routes
        ):
            raise EraserEvidenceInferenceRunnerError("R7 route count drifted")
        hippo_cross_relation = hippo_passed and all(
            value > 0
            for value in family_sums["E3_minus_HippoRAG"].values()
        )
        a_hold_promotion = e0_passed and distinct_routes > 0
        raw_block = raw_passed and complete["E3"] >= complete["RAW"]
        expected = {
            "Hippo_cross_relation_passed": hippo_cross_relation,
            "RAW_block_passed": raw_block,
            "A_hold_real_domain_primary_passed": (
                hippo_cross_relation if self.block == "A_hold" else None
            ),
            "evaluator_promoted": (
                a_hold_promotion if self.block == "A_hold" else None
            ),
            # M_L5 is exactly the preregistered paired E3-minus-E0 test.  The
            # behavior-distinct condition belongs only to A_hold promotion.
            "M_L5_passed": e0_passed if self.block == "M_search" else None,
            "cross_relation_stability_passed": (
                self.a_hold_authorization.receipt["Hippo_cross_relation_passed"]
                and hippo_cross_relation
                if self.block == "M_search"
                else None
            ),
            "RAW_advantage_overcome": (
                self.a_hold_authorization.receipt["RAW_block_passed"] and raw_block
                if self.block == "M_search"
                else None
            ),
        }
        if any(
            (receipt.get(key) is not None if value is None else type(receipt.get(key)) is not bool)
            or receipt.get(key) != value
            for key, value in expected.items()
        ):
            raise EraserEvidenceInferenceRunnerError(
                "anchor derived decision semantics drifted"
            )

    @property
    def receipt(self) -> dict[str, Any]:
        return _mapping_from_canonical_json(
            self.receipt_json, field=f"{self.block} score receipt"
        )

    @property
    def score_receipt_sha256(self) -> str:
        return str(self.receipt["score_receipt_sha256"])

    @property
    def evaluator_promoted(self) -> bool:
        value = self.receipt.get("evaluator_promoted")
        return value if type(value) is bool else False


def score_anchor(
    *,
    block: str,
    labels: Sequence[AnchorLabel],
    anchor_feature_seal: FeatureSeal,
    hippo_retrieval_seal: HippoRetrievalSeal,
    policy_seal: PolicySeal,
    a_hold_authorization: AnchorScoreSeal | None = None,
) -> AnchorScoreSeal:
    """Score one late block after all Agent/RAW/Hippo outputs are sealed."""

    if block not in {"A_hold", "M_search"}:
        raise EraserEvidenceInferenceRunnerError("anchor block is invalid")
    if (
        not isinstance(anchor_feature_seal, FeatureSeal)
        or anchor_feature_seal.block != block
        or not isinstance(hippo_retrieval_seal, HippoRetrievalSeal)
        or hippo_retrieval_seal.block != block
        or not isinstance(policy_seal, PolicySeal)
    ):
        raise EraserEvidenceInferenceRunnerError("anchor inputs are not sealed")
    if block == "A_hold" and a_hold_authorization is not None:
        raise EraserEvidenceInferenceRunnerError("A_hold cannot use authorization")
    if block == "M_search" and (
        not isinstance(a_hold_authorization, AnchorScoreSeal)
        or not a_hold_authorization.evaluator_promoted
        or a_hold_authorization.policies.policy_receipt_sha256
        != policy_seal.policy_receipt_sha256
        or set(a_hold_authorization.anchor_features.item_commitments).intersection(
            anchor_feature_seal.item_commitments
        )
    ):
        raise EraserEvidenceInferenceRunnerError("M_search is not authorized")
    if any(not isinstance(label, AnchorLabel) for label in labels):
        raise EraserEvidenceInferenceRunnerError("anchor labels contain a foreign type")
    labels_by_item = {label.item_commitment_sha256: label for label in labels}
    commitments = anchor_feature_seal.item_commitments
    traces_by_item = anchor_feature_seal.by_item
    hippo_by_item = hippo_retrieval_seal.by_item
    if (
        len(labels_by_item) != len(labels)
        or set(labels_by_item) != set(commitments)
        or set(hippo_by_item) != set(commitments)
        or hippo_retrieval_seal.item_commitment_set_sha256
        != anchor_feature_seal.item_commitment_set_sha256
    ):
        raise EraserEvidenceInferenceRunnerError(
            "anchor commitment-keyed alignment drifted"
        )
    family_counts = Counter(label.family for label in labels_by_item.values())
    if dict(family_counts) != BLOCK_FAMILY_COUNTS[block]:
        raise EraserEvidenceInferenceRunnerError("anchor family counts drifted")

    delta_vectors: dict[str, list[Fraction]] = {
        comparison: [] for comparison in PAIRWISE_COMPARISONS
    }
    family_deltas: dict[str, dict[str, list[Fraction]]] = {
        comparison: defaultdict(list) for comparison in PAIRWISE_COMPARISONS
    }
    complete = {arm: 0 for arm in ("E0", "E3", "HippoRAG", "RAW")}
    complete_by_family = {
        arm: {family: 0 for family in FAMILIES}
        for arm in ("E0", "E3", "HippoRAG", "RAW")
    }
    (
        routes_by_item,
        route_counts,
        behavior_distinct_r7_routes,
        r7_action_aggregates,
    ) = _route_and_action_aggregates(
        traces=anchor_feature_seal.traces, policy=policy_seal
    )
    for item in commitments:
        trace = traces_by_item[item]
        label = labels_by_item[item]
        hippo = hippo_by_item[item]
        if hippo.sentence_count != trace.sentence_count or any(
            value >= trace.sentence_count for value in label.gold_ordinals
        ):
            raise EraserEvidenceInferenceRunnerError("sentence-corpus alignment drifted")
        e3_recipe = routes_by_item[item]
        e3_output = trace.r7_top5 if e3_recipe == RECIPE_IDS[1] else trace.r0_top5
        scored = {
            "E0": item_utility(trace.r0_top5, label.gold_ordinals),
            "E3": item_utility(e3_output, label.gold_ordinals),
            "HippoRAG": item_utility(hippo.top5, label.gold_ordinals),
            "RAW": item_utility(trace.r0_top5, label.gold_ordinals),
        }
        for arm, (_utility, is_complete) in scored.items():
            complete[arm] += int(is_complete)
            complete_by_family[arm][label.family] += int(is_complete)
        item_deltas = {
            comparison: scored["E3"][0] - scored[baseline][0]
            for comparison, baseline in PAIRWISE_BASELINES.items()
        }
        for comparison, delta in item_deltas.items():
            delta_vectors[comparison].append(delta)
            family_deltas[comparison][label.family].append(delta)

    tests = {
        comparison: _sign_flip_payload(delta_vectors[comparison])
        for comparison in PAIRWISE_COMPARISONS
    }
    e0_test = tests["E3_minus_E0"]
    hippo_test = tests["E3_minus_HippoRAG"]
    raw_test = tests["E3_minus_RAW"]
    pairwise_totals = {
        comparison: sum(delta_vectors[comparison], Fraction(0))
        for comparison in PAIRWISE_COMPARISONS
    }
    pairwise_family_sums = {
        comparison: {
            family: sum(family_deltas[comparison][family], Fraction(0))
            for family in FAMILIES
        }
        for comparison in PAIRWISE_COMPARISONS
    }
    e0_passed = bool(e0_test["promoted"])
    hippo_cross_relation = bool(hippo_test["promoted"]) and all(
        value > 0
        for value in pairwise_family_sums["E3_minus_HippoRAG"].values()
    )
    a_hold_promotion = e0_passed and behavior_distinct_r7_routes > 0
    raw_block = bool(raw_test["promoted"]) and complete["E3"] >= complete["RAW"]
    pairwise_complete_deltas = {
        comparison: complete["E3"] - complete[baseline]
        for comparison, baseline in PAIRWISE_BASELINES.items()
    }
    pairwise_complete_family_deltas = {
        comparison: {
            family: complete_by_family["E3"][family]
            - complete_by_family[baseline][family]
            for family in FAMILIES
        }
        for comparison, baseline in PAIRWISE_BASELINES.items()
    }
    authorization_sha = (
        a_hold_authorization.score_receipt_sha256
        if a_hold_authorization is not None
        else None
    )
    body = {
        "schema": f"{VERSION}_{block}_score_receipt",
        "version": VERSION,
        "block": block,
        "item_count": len(commitments),
        "logical_RAW_HippoRAG_Agent_work_units": 3 * len(commitments),
        "anchor_feature_receipt_sha256": anchor_feature_seal.feature_receipt_sha256,
        "policy_receipt_sha256": policy_seal.policy_receipt_sha256,
        "hipporag_retrieval_matrix_sha256": (
            hippo_retrieval_seal.retrieval_matrix_sha256
        ),
        "item_commitment_set_sha256": anchor_feature_seal.item_commitment_set_sha256,
        "late_opened_label_matrix_sha256": stable_hash(
            [
                [item, list(labels_by_item[item].gold_ordinals), labels_by_item[item].family]
                for item in commitments
            ]
        ),
        "A_hold_authorization_score_receipt_sha256": authorization_sha,
        "E0_routing": "always_R0_DENSE5",
        "E3_route_counts": {recipe: route_counts[recipe] for recipe in RECIPE_IDS},
        "behavior_distinct_R7_route_count": behavior_distinct_r7_routes,
        "R7_action_aggregates": r7_action_aggregates,
        "E3_minus_E0": e0_test,
        "E3_minus_HippoRAG": hippo_test,
        "E3_minus_RAW": raw_test,
        "pairwise_total_U": {
            comparison: _fraction_payload(pairwise_totals[comparison])
            for comparison in PAIRWISE_COMPARISONS
        },
        "pairwise_family_sums": {
            comparison: {
                family: _fraction_payload(
                    pairwise_family_sums[comparison][family]
                )
                for family in FAMILIES
            }
            for comparison in PAIRWISE_COMPARISONS
        },
        "family_item_counts": dict(family_counts),
        "complete_counts": complete,
        "complete_counts_by_family": complete_by_family,
        "pairwise_complete_count_deltas": pairwise_complete_deltas,
        "pairwise_complete_count_deltas_by_family": (
            pairwise_complete_family_deltas
        ),
        "Hippo_cross_relation_passed": hippo_cross_relation,
        "RAW_block_passed": raw_block,
        "A_hold_real_domain_primary_passed": (
            hippo_cross_relation if block == "A_hold" else None
        ),
        "evaluator_promoted": a_hold_promotion if block == "A_hold" else None,
        "M_L5_passed": e0_passed if block == "M_search" else None,
        "cross_relation_stability_passed": (
            a_hold_authorization.receipt["Hippo_cross_relation_passed"]
            and hippo_cross_relation
            if block == "M_search"
            else None
        ),
        "RAW_advantage_overcome": (
            a_hold_authorization.receipt["RAW_block_passed"] and raw_block
            if block == "M_search"
            else None
        ),
        "item_level_utility_values_persisted": False,
        "online_evaluator_calls": 0,
        "raw_content_persisted": False,
    }
    receipt = _self_hashed(body, "score_receipt_sha256")
    return AnchorScoreSeal(
        block=block,
        anchor_features=anchor_feature_seal,
        hippo_retrievals=hippo_retrieval_seal,
        policies=policy_seal,
        a_hold_authorization=a_hold_authorization,
        receipt_json=_canonical_json_text(receipt),
    )


__all__ = [
    "AnchorLabel",
    "AnchorScoreSeal",
    "BLOCK_COUNTS",
    "BLOCK_FAMILY_COUNTS",
    "BLOCK_ORDER",
    "DECIMAL_PRECISION",
    "DifferenceTrace",
    "E3FitSeal",
    "E3Model",
    "EraserEvidenceInferenceRunnerError",
    "FAMILIES",
    "FEATURE_ORDER",
    "FOLD_COUNT",
    "FeatureSeal",
    "HippoRetrieval",
    "HippoRetrievalSeal",
    "PAIRWISE_BASELINES",
    "PAIRWISE_COMPARISONS",
    "PROMOTION_ALPHA",
    "PolicySeal",
    "RECIPE_IDS",
    "RIDGE_LAMBDA",
    "TOP_K",
    "VERSION",
    "behavior_sha256",
    "build_feature_receipt",
    "fit_e3",
    "freeze_f_policy",
    "freeze_f_policies",
    "item_utility",
    "route_e3",
    "score_anchor",
    "seal_feature_matrix",
    "seal_hippo_retrievals",
    "stable_hash",
]
