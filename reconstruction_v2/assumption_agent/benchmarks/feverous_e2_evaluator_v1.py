"""Deterministic FEVEROUS E0/E2 evaluator mathematics.

This module is deliberately independent of FEVEROUS source readers and formal
controllers.  It consumes only already-terminal, content-free recipe traces and,
for the A_form fit, exact utility values supplied by the caller.  In particular,
it cannot read labels, evidence, challenge families, or F_search outcomes.

The implementation follows ``feverous_p6_e2_evaluator_design_v1``:

* four fixed recipes and eight fixed continuous trace coordinates;
* E0 is the exact mean of per-coordinate average midranks;
* E2 is a no-intercept, lambda-one, equally item-weighted pairwise ridge;
* A_form population standardisation maps zero-variance coordinates to zero;
* four HMAC folds are descriptive only, followed by one all-A_form fit; and
* F_search freezes one global E0 policy and one global E2 policy without labels.

All fitting arithmetic uses a local 80-digit, half-even ``Decimal`` context.
Inputs must be exact integers, Fractions, Decimals, or decimal strings; binary
floats are rejected.  Public receipts contain canonical strings/rationals and
are semantic-self-hashed using canonical JSON.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
import hashlib
import hmac
import json
import math
from typing import Any, Mapping, Sequence


VERSION = "feverous_e2_evaluator_v1"
RECIPE_IDS = (
    "R0_DENSE5",
    "R1_P6_DIRECT_B2",
    "R2_P6_PATH1_B2",
    "R3_P6_PATH2_B2",
)
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
FORBIDDEN_FEATURES = frozenset(
    {
        "challenge_family",
        "SUPPORTS_or_REFUTES_label",
        "gold_or_evidence_id",
        "RAW_or_Hippo_agreement",
        "recipe_id",
        "page_or_item_identity",
    }
)
FOLD_COUNT = 4
RIDGE_LAMBDA = Decimal(1)
PAIR_WEIGHT = Fraction(1, 6)
DECIMAL_PRECISION = 80
PROMOTION_ALPHA = Fraction(1, 10)
BLOCK_ITEM_COUNTS = {"A_form": 96, "F_search": 48}


class FeverousEvaluatorError(ValueError):
    """Fail-closed error for malformed traces, labels, or receipts."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, field: str) -> str:
    if not _is_sha256(value):
        raise FeverousEvaluatorError(f"{field} must be a lowercase sha256")
    return str(value)


def _to_decimal(value: object, field: str = "numeric value") -> Decimal:
    if isinstance(value, bool) or isinstance(value, float):
        raise FeverousEvaluatorError(f"{field} must be exact, not binary float")
    try:
        with localcontext() as context:
            context.prec = DECIMAL_PRECISION
            context.rounding = ROUND_HALF_EVEN
            if isinstance(value, Decimal):
                result = +value
            elif isinstance(value, Fraction):
                result = Decimal(value.numerator) / Decimal(value.denominator)
            elif isinstance(value, int):
                result = Decimal(value)
            elif isinstance(value, str):
                result = Decimal(value)
            else:
                raise FeverousEvaluatorError(f"{field} has an unsupported type")
    except InvalidOperation as exc:
        raise FeverousEvaluatorError(f"{field} is not a finite decimal") from exc
    if not result.is_finite():
        raise FeverousEvaluatorError(f"{field} is not finite")
    return Decimal(0) if result == 0 else result


def _decimal_text(value: Decimal) -> str:
    value = _to_decimal(value)
    if value == 0:
        return "0"
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _fraction_payload(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _fraction_from_payload(value: object, field: str) -> Fraction:
    if not isinstance(value, Mapping):
        raise FeverousEvaluatorError(f"{field} must be an exact rational")
    numerator = value.get("numerator")
    denominator = value.get("denominator")
    if (
        type(numerator) is not int
        or type(denominator) is not int
        or denominator <= 0
    ):
        raise FeverousEvaluatorError(f"{field} must be an exact rational")
    result = Fraction(numerator, denominator)
    if result.numerator != numerator or result.denominator != denominator:
        raise FeverousEvaluatorError(f"{field} rational is not reduced")
    return result


def _canonical_hash(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise FeverousEvaluatorError("receipt payload is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = dict(body)
    if field in value:
        raise FeverousEvaluatorError(f"{field} already exists")
    value[field] = _canonical_hash(value)
    return value


def _verify_self_hash(
    receipt: Mapping[str, Any], *, schema: str, field: str
) -> str:
    body = dict(receipt)
    declared = _require_sha256(body.pop(field, None), field)
    if receipt.get("schema") != schema or _canonical_hash(body) != declared:
        raise FeverousEvaluatorError(f"{schema} self-hash drifted")
    return declared


@dataclass(frozen=True)
class RecipeTrace:
    """One content-free, fixed-schema recipe trace for one item."""

    item_commitment_sha256: str
    recipe_id: str
    behavior_sha256: str
    features: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.item_commitment_sha256, "item commitment")
        _require_sha256(self.behavior_sha256, "ordered-top5 behavior commitment")
        if self.recipe_id not in RECIPE_IDS:
            raise FeverousEvaluatorError("recipe is outside the frozen registry")
        if len(self.features) != len(FEATURE_ORDER):
            raise FeverousEvaluatorError("trace must contain exactly eight features")
        normalized = tuple(
            _to_decimal(value, f"feature {FEATURE_ORDER[index]}")
            for index, value in enumerate(self.features)
        )
        object.__setattr__(self, "features", normalized)

    @classmethod
    def from_mapping(
        cls,
        *,
        item_commitment_sha256: str,
        recipe_id: str,
        behavior_sha256: str,
        features: Mapping[str, object],
    ) -> "RecipeTrace":
        """Build a trace while rejecting recipe identity and all schema drift."""

        supplied = set(features)
        forbidden = supplied.intersection(FORBIDDEN_FEATURES)
        if forbidden:
            raise FeverousEvaluatorError(
                "forbidden evaluator feature(s): " + ", ".join(sorted(forbidden))
            )
        expected = set(FEATURE_ORDER)
        if supplied != expected:
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            raise FeverousEvaluatorError(
                f"fixed feature schema drifted; missing={missing}, extra={extra}"
            )
        return cls(
            item_commitment_sha256=item_commitment_sha256,
            recipe_id=recipe_id,
            behavior_sha256=behavior_sha256,
            features=tuple(_to_decimal(features[name], name) for name in FEATURE_ORDER),
        )

    def payload(self) -> dict[str, object]:
        return {
            "item_commitment_sha256": self.item_commitment_sha256,
            "recipe_id": self.recipe_id,
            "behavior_sha256": self.behavior_sha256,
            "features": [_decimal_text(value) for value in self.features],
        }


@dataclass(frozen=True)
class E2Model:
    """Frozen scaler and final no-intercept ridge coefficients."""

    population_mean: tuple[Decimal, ...]
    population_std: tuple[Decimal, ...]
    beta: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        width = len(FEATURE_ORDER)
        if not all(
            len(values) == width
            for values in (self.population_mean, self.population_std, self.beta)
        ):
            raise FeverousEvaluatorError("E2 model width drifted")
        means = tuple(_to_decimal(value, "population mean") for value in self.population_mean)
        stds = tuple(_to_decimal(value, "population std") for value in self.population_std)
        betas = tuple(_to_decimal(value, "ridge beta") for value in self.beta)
        if any(value < 0 for value in stds):
            raise FeverousEvaluatorError("population standard deviation is negative")
        object.__setattr__(self, "population_mean", means)
        object.__setattr__(self, "population_std", stds)
        object.__setattr__(self, "beta", betas)

    def standardize(self, features: Sequence[object]) -> tuple[Decimal, ...]:
        if len(features) != len(FEATURE_ORDER):
            raise FeverousEvaluatorError("feature width drifted during E2 scoring")
        values = tuple(_to_decimal(value) for value in features)
        with localcontext() as context:
            context.prec = DECIMAL_PRECISION
            context.rounding = ROUND_HALF_EVEN
            return tuple(
                Decimal(0) if std == 0 else (value - mean) / std
                for value, mean, std in zip(
                    values, self.population_mean, self.population_std
                )
            )

    def predict(self, features: Sequence[object]) -> Decimal:
        standardized = self.standardize(features)
        with localcontext() as context:
            context.prec = DECIMAL_PRECISION
            context.rounding = ROUND_HALF_EVEN
            return sum(
                (coefficient * value for coefficient, value in zip(self.beta, standardized)),
                Decimal(0),
            )

    def payload(self) -> dict[str, object]:
        return {
            "population_mean": [_decimal_text(value) for value in self.population_mean],
            "population_std": [_decimal_text(value) for value in self.population_std],
            "beta": [_decimal_text(value) for value in self.beta],
        }


@dataclass(frozen=True)
class FeverousItemUtility:
    """Exact recall-at-five plus complete-set bonus for one canonical set."""

    distinct_gold_hits: int
    canonical_gold_count: int
    complete: bool
    value: Fraction


@dataclass(frozen=True)
class ExactSignFlipResult:
    observed_net_u: Fraction
    nonzero_pair_count: int
    exact_p: Fraction
    promoted: bool

    def payload(self) -> dict[str, object]:
        return {
            "test": "feverous_one_sided_exact_magnitude_preserving_sign_flip_v1",
            "observed_net_U": _fraction_payload(self.observed_net_u),
            "nonzero_pair_count": self.nonzero_pair_count,
            "p_value": _fraction_payload(self.exact_p),
            "alpha": _fraction_payload(PROMOTION_ALPHA),
            "positive_observed_net": self.observed_net_u > 0,
            "exact_p_at_or_below_alpha": self.exact_p <= PROMOTION_ALPHA,
            "promoted": self.promoted,
        }


def _normalize_matrix(
    traces: Sequence[RecipeTrace],
) -> tuple[tuple[str, tuple[RecipeTrace, ...]], ...]:
    if not traces:
        raise FeverousEvaluatorError("trace matrix is empty")
    rows: dict[str, dict[str, RecipeTrace]] = {}
    for trace in traces:
        if not isinstance(trace, RecipeTrace):
            raise FeverousEvaluatorError("trace matrix contains a foreign type")
        by_recipe = rows.setdefault(trace.item_commitment_sha256, {})
        if trace.recipe_id in by_recipe:
            raise FeverousEvaluatorError("duplicate item/recipe trace")
        by_recipe[trace.recipe_id] = trace
    normalized: list[tuple[str, tuple[RecipeTrace, ...]]] = []
    for item in sorted(rows):
        if set(rows[item]) != set(RECIPE_IDS):
            raise FeverousEvaluatorError("each item must contain all four recipes once")
        normalized.append((item, tuple(rows[item][recipe] for recipe in RECIPE_IDS)))
    return tuple(normalized)


def _trace_matrix_sha256(
    matrix: Sequence[tuple[str, Sequence[RecipeTrace]]],
) -> str:
    return _canonical_hash(
        [trace.payload() for _, rows in matrix for trace in rows]
    )


def build_feature_receipt(
    *, block: str, traces: Sequence[RecipeTrace]
) -> dict[str, Any]:
    """Seal a content-free trace matrix without utility or label access."""

    if block not in BLOCK_ITEM_COUNTS:
        raise FeverousEvaluatorError("unknown evaluator block")
    matrix = _normalize_matrix(traces)
    if len(matrix) != BLOCK_ITEM_COUNTS[block]:
        raise FeverousEvaluatorError(
            f"{block} must contain exactly {BLOCK_ITEM_COUNTS[block]} items"
        )
    body = {
        "schema": f"{VERSION}_feature_receipt",
        "version": VERSION,
        "block": block,
        "item_count": len(matrix),
        "trace_count": len(matrix) * len(RECIPE_IDS),
        "recipe_registry": list(RECIPE_IDS),
        "fixed_feature_order": list(FEATURE_ORDER),
        "feature_value_encoding": "canonical_decimal_string_v1",
        "trace_matrix_sha256": _trace_matrix_sha256(matrix),
        "labels_or_utility_accessed": False,
        "recipe_id_used_as_feature": False,
        "raw_content_persisted": False,
    }
    return _self_hashed(body, "feature_receipt_sha256")


def verify_feature_receipt(
    receipt: Mapping[str, Any], *, block: str, traces: Sequence[RecipeTrace]
) -> str:
    declared = _verify_self_hash(
        receipt,
        schema=f"{VERSION}_feature_receipt",
        field="feature_receipt_sha256",
    )
    expected = build_feature_receipt(block=block, traces=traces)
    if dict(receipt) != expected:
        raise FeverousEvaluatorError("feature receipt semantic binding drifted")
    return declared


def average_midranks(values: Mapping[str, Decimal]) -> dict[str, Fraction]:
    """Return exact ascending average midranks; the largest value ranks highest."""

    if set(values) != set(RECIPE_IDS):
        raise FeverousEvaluatorError("midrank input must contain the four recipes")
    ordered = sorted(
        ((recipe, _to_decimal(value)) for recipe, value in values.items()),
        key=lambda row: (row[1], row[0]),
    )
    result: dict[str, Fraction] = {}
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and ordered[stop][1] == ordered[start][1]:
            stop += 1
        # One-based positions start+1 through stop have this exact average.
        midrank = Fraction((start + 1) + stop, 2)
        for index in range(start, stop):
            result[ordered[index][0]] = midrank
        start = stop
    return result


def e0_item_scores(
    traces: Sequence[RecipeTrace],
) -> dict[str, dict[str, Fraction]]:
    """Compute exact balanced E0 scores for every item/recipe."""

    matrix = _normalize_matrix(traces)
    result: dict[str, dict[str, Fraction]] = {}
    for item, rows in matrix:
        sums = {recipe: Fraction(0) for recipe in RECIPE_IDS}
        for coordinate in range(len(FEATURE_ORDER)):
            ranks = average_midranks(
                {row.recipe_id: row.features[coordinate] for row in rows}
            )
            for recipe in RECIPE_IDS:
                sums[recipe] += ranks[recipe]
        result[item] = {
            recipe: sums[recipe] / len(FEATURE_ORDER) for recipe in RECIPE_IDS
        }
    return result


def _normalize_utilities(
    matrix: Sequence[tuple[str, Sequence[RecipeTrace]]],
    utilities: Mapping[tuple[str, str], Fraction | int],
) -> dict[tuple[str, str], Fraction]:
    expected = {(item, recipe) for item, _ in matrix for recipe in RECIPE_IDS}
    if set(utilities) != expected:
        raise FeverousEvaluatorError("A_form utility matrix coverage drifted")
    result: dict[tuple[str, str], Fraction] = {}
    for key, value in utilities.items():
        if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
            raise FeverousEvaluatorError("utility must be an exact Fraction or integer")
        utility = Fraction(value)
        if not 0 <= utility <= 2:
            raise FeverousEvaluatorError("utility is outside the frozen zero-through-two range")
        result[key] = utility
    return result


def _population_scaler(
    rows: Sequence[RecipeTrace],
) -> tuple[tuple[Decimal, ...], tuple[Decimal, ...]]:
    if not rows:
        raise FeverousEvaluatorError("cannot fit a scaler without traces")
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        count = Decimal(len(rows))
        means = tuple(
            sum((row.features[index] for row in rows), Decimal(0)) / count
            for index in range(len(FEATURE_ORDER))
        )
        variances = tuple(
            sum(
                ((row.features[index] - means[index]) ** 2 for row in rows),
                Decimal(0),
            )
            / count
            for index in range(len(FEATURE_ORDER))
        )
        stds = tuple(
            Decimal(0) if variance == 0 else context.sqrt(variance)
            for variance in variances
        )
    return means, stds


def _standardize(
    features: Sequence[Decimal],
    means: Sequence[Decimal],
    stds: Sequence[Decimal],
) -> tuple[Decimal, ...]:
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        return tuple(
            Decimal(0) if std == 0 else (value - mean) / std
            for value, mean, std in zip(features, means, stds)
        )


def _pair_rows(
    matrix: Sequence[tuple[str, Sequence[RecipeTrace]]],
    utilities: Mapping[tuple[str, str], Fraction],
    means: Sequence[Decimal],
    stds: Sequence[Decimal],
) -> list[tuple[tuple[Decimal, ...], Fraction]]:
    pairs: list[tuple[tuple[Decimal, ...], Fraction]] = []
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        for item, traces in matrix:
            standardized = {
                trace.recipe_id: _standardize(trace.features, means, stds)
                for trace in traces
            }
            for left_index, left in enumerate(RECIPE_IDS):
                for right in RECIPE_IDS[left_index + 1 :]:
                    x = tuple(
                        a - b
                        for a, b in zip(standardized[left], standardized[right])
                    )
                    y = utilities[(item, left)] - utilities[(item, right)]
                    pairs.append((x, y))
    return pairs


def _solve_linear_system(
    matrix: Sequence[Sequence[Decimal]], vector: Sequence[Decimal]
) -> tuple[Decimal, ...]:
    width = len(vector)
    augmented = [list(row) + [vector[index]] for index, row in enumerate(matrix)]
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        for column in range(width):
            pivot = max(range(column, width), key=lambda row: abs(augmented[row][column]))
            if augmented[pivot][column] == 0:
                raise FeverousEvaluatorError("ridge system is unexpectedly singular")
            if pivot != column:
                augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
            pivot_value = augmented[column][column]
            augmented[column] = [value / pivot_value for value in augmented[column]]
            for row in range(width):
                if row == column:
                    continue
                factor = augmented[row][column]
                if factor == 0:
                    continue
                augmented[row] = [
                    value - factor * pivot_entry
                    for value, pivot_entry in zip(augmented[row], augmented[column])
                ]
    return tuple(Decimal(0) if row[-1] == 0 else row[-1] for row in augmented)


def _fit_model(
    matrix: Sequence[tuple[str, Sequence[RecipeTrace]]],
    utilities: Mapping[tuple[str, str], Fraction],
) -> tuple[E2Model, list[tuple[tuple[Decimal, ...], Fraction]]]:
    rows = [trace for _, item_rows in matrix for trace in item_rows]
    means, stds = _population_scaler(rows)
    pairs = _pair_rows(matrix, utilities, means, stds)
    width = len(FEATURE_ORDER)
    gram = [[Decimal(0) for _ in range(width)] for _ in range(width)]
    rhs = [Decimal(0) for _ in range(width)]
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        # Every item contributes six unordered pairs at exact weight 1/6.
        # Multiplying the complete normal equation by six avoids representing
        # the recurring decimal 1/6: (sum xx' + 6 I) beta = sum xy is exactly
        # equivalent to ((1/6) sum xx' + I) beta = (1/6) sum xy.
        ridge_after_exact_rescale = Decimal(PAIR_WEIGHT.denominator)
        for x, y_fraction in pairs:
            y = Decimal(y_fraction.numerator) / Decimal(y_fraction.denominator)
            for left in range(width):
                rhs[left] += x[left] * y
                for right in range(width):
                    gram[left][right] += x[left] * x[right]
        for index in range(width):
            gram[index][index] += RIDGE_LAMBDA * ridge_after_exact_rescale
        beta = _solve_linear_system(gram, rhs)
    return E2Model(means, stds, beta), pairs


def _fold_for(item_commitment_sha256: str, secret: bytes) -> int:
    if not isinstance(secret, bytes) or len(secret) < 16:
        raise FeverousEvaluatorError("private fold HMAC secret must contain at least 16 bytes")
    _require_sha256(item_commitment_sha256, "item commitment")
    digest = hmac.new(
        secret,
        f"{VERSION}:A_form:four_fold:{item_commitment_sha256}".encode("ascii"),
        hashlib.sha256,
    ).digest()
    return int.from_bytes(digest[:8], "big") % FOLD_COUNT


def _prediction_error(
    model: E2Model,
    matrix: Sequence[tuple[str, Sequence[RecipeTrace]]],
    utilities: Mapping[tuple[str, str], Fraction],
) -> tuple[int, int, Decimal]:
    correct = 0
    non_tie = 0
    squared_error = Decimal(0)
    pair_count = 0
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        for item, traces in matrix:
            predicted = {trace.recipe_id: model.predict(trace.features) for trace in traces}
            for left_index, left in enumerate(RECIPE_IDS):
                for right in RECIPE_IDS[left_index + 1 :]:
                    predicted_delta = predicted[left] - predicted[right]
                    exact_delta = utilities[(item, left)] - utilities[(item, right)]
                    exact_decimal = Decimal(exact_delta.numerator) / Decimal(
                        exact_delta.denominator
                    )
                    squared_error += (predicted_delta - exact_decimal) ** 2
                    pair_count += 1
                    if exact_delta != 0:
                        non_tie += 1
                        correct += (predicted_delta > 0) == (exact_delta > 0)
        mse = Decimal(0) if pair_count == 0 else squared_error / Decimal(pair_count)
    return correct, non_tie, mse


def fit_e2_a_form(
    *,
    traces: Sequence[RecipeTrace],
    utilities: Mapping[tuple[str, str], Fraction | int],
    fold_hmac_secret: bytes,
    feature_receipt: Mapping[str, Any],
) -> tuple[E2Model, dict[str, Any]]:
    """Run descriptive four-fold diagnostics, then exactly one final A_form fit."""

    matrix = _normalize_matrix(traces)
    if len(matrix) != BLOCK_ITEM_COUNTS["A_form"]:
        raise FeverousEvaluatorError("A_form fit item count drifted")
    receipt = dict(feature_receipt)
    feature_sha = verify_feature_receipt(
        receipt, block="A_form", traces=traces
    )
    normalized_utilities = _normalize_utilities(matrix, utilities)
    fold_by_item = {
        item: _fold_for(item, fold_hmac_secret) for item, _ in matrix
    }
    folds: list[dict[str, Any]] = []
    for fold in range(FOLD_COUNT):
        fit_matrix = tuple(row for row in matrix if fold_by_item[row[0]] != fold)
        held_matrix = tuple(row for row in matrix if fold_by_item[row[0]] == fold)
        if not fit_matrix:
            raise FeverousEvaluatorError("HMAC cross-fit produced an empty fit partition")
        fit_keys = {(item, recipe) for item, _ in fit_matrix for recipe in RECIPE_IDS}
        fit_utilities = {
            key: value for key, value in normalized_utilities.items() if key in fit_keys
        }
        fold_model, _ = _fit_model(fit_matrix, fit_utilities)
        correct, non_tie, mse = _prediction_error(
            fold_model, held_matrix, normalized_utilities
        )
        folds.append(
            {
                "fold": fold,
                "fit_item_count": len(fit_matrix),
                "held_item_count": len(held_matrix),
                "held_pair_count": 6 * len(held_matrix),
                "held_non_tie_pair_count": non_tie,
                "held_preference_correct_count": correct,
                "held_pair_mean_squared_error": _decimal_text(mse),
                "fit_model_sha256": _canonical_hash(fold_model.payload()),
            }
        )
    model, pairs = _fit_model(matrix, normalized_utilities)
    utility_payload = [
        {
            "item_commitment_sha256": item,
            "recipe_id": recipe,
            "utility": _fraction_payload(normalized_utilities[(item, recipe)]),
        }
        for item, _ in matrix
        for recipe in RECIPE_IDS
    ]
    assignment_payload = [
        {"item_commitment_sha256": item, "fold": fold_by_item[item]}
        for item, _ in matrix
    ]
    body = {
        "schema": f"{VERSION}_fit_receipt",
        "version": VERSION,
        "block": "A_form",
        "feature_receipt_sha256": feature_sha,
        "trace_matrix_sha256": receipt["trace_matrix_sha256"],
        "utility_matrix_sha256": _canonical_hash(utility_payload),
        "utility_values_persisted": False,
        "item_count": len(matrix),
        "pair_count": len(pairs),
        "pair_count_per_item": 6,
        "pair_weight": _fraction_payload(PAIR_WEIGHT),
        "fixed_feature_order": list(FEATURE_ORDER),
        "scaler": "population_mean_and_population_standard_deviation_v1",
        "zero_variance_maps_to_zero": True,
        "ridge_lambda": "1",
        "intercept": False,
        "decimal_contract": {
            "precision": DECIMAL_PRECISION,
            "rounding": "ROUND_HALF_EVEN",
            "binary_float_inputs": False,
        },
        "fold_policy": "private_HMAC_SHA256_four_fold_v1",
        "fold_secret_sha256": hashlib.sha256(fold_hmac_secret).hexdigest(),
        "fold_assignment_sha256": _canonical_hash(assignment_payload),
        "crossfit_descriptive_only": True,
        "crossfit": folds,
        "final_fit_count": 1,
        "model": model.payload(),
        "F_search_accessed": False,
        "A_hold_accessed": False,
        "online_evaluator_calls": 0,
        "raw_content_persisted": False,
    }
    return model, _self_hashed(body, "fit_receipt_sha256")


def model_from_fit_receipt(
    receipt: Mapping[str, Any],
    *,
    feature_receipt_sha256: str,
    fit_receipt_sha256: str,
) -> E2Model:
    """Verify public fit semantics and recover the frozen model without labels."""

    declared = _verify_self_hash(
        receipt,
        schema=f"{VERSION}_fit_receipt",
        field="fit_receipt_sha256",
    )
    if declared != _require_sha256(
        fit_receipt_sha256, "externally frozen fit receipt"
    ):
        raise FeverousEvaluatorError("fit receipt is outside the external freeze")
    expected_keys = {
        "schema",
        "version",
        "block",
        "feature_receipt_sha256",
        "trace_matrix_sha256",
        "utility_matrix_sha256",
        "utility_values_persisted",
        "item_count",
        "pair_count",
        "pair_count_per_item",
        "pair_weight",
        "fixed_feature_order",
        "scaler",
        "zero_variance_maps_to_zero",
        "ridge_lambda",
        "intercept",
        "decimal_contract",
        "fold_policy",
        "fold_secret_sha256",
        "fold_assignment_sha256",
        "crossfit_descriptive_only",
        "crossfit",
        "final_fit_count",
        "model",
        "F_search_accessed",
        "A_hold_accessed",
        "online_evaluator_calls",
        "raw_content_persisted",
        "fit_receipt_sha256",
    }
    if set(receipt) != expected_keys:
        raise FeverousEvaluatorError("fit receipt key schema drifted")
    required = {
        "version": VERSION,
        "block": "A_form",
        "feature_receipt_sha256": _require_sha256(
            feature_receipt_sha256, "feature receipt binding"
        ),
        "utility_values_persisted": False,
        "item_count": BLOCK_ITEM_COUNTS["A_form"],
        "pair_count_per_item": 6,
        "pair_weight": _fraction_payload(PAIR_WEIGHT),
        "fixed_feature_order": list(FEATURE_ORDER),
        "scaler": "population_mean_and_population_standard_deviation_v1",
        "zero_variance_maps_to_zero": True,
        "ridge_lambda": "1",
        "intercept": False,
        "decimal_contract": {
            "precision": DECIMAL_PRECISION,
            "rounding": "ROUND_HALF_EVEN",
            "binary_float_inputs": False,
        },
        "fold_policy": "private_HMAC_SHA256_four_fold_v1",
        "crossfit_descriptive_only": True,
        "final_fit_count": 1,
        "F_search_accessed": False,
        "A_hold_accessed": False,
        "online_evaluator_calls": 0,
        "raw_content_persisted": False,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise FeverousEvaluatorError("fit receipt frozen semantics drifted")
    if receipt.get("pair_count") != 6 * BLOCK_ITEM_COUNTS["A_form"]:
        raise FeverousEvaluatorError("fit receipt pair count drifted")
    for field in (
        "trace_matrix_sha256",
        "utility_matrix_sha256",
        "fold_secret_sha256",
        "fold_assignment_sha256",
    ):
        _require_sha256(receipt.get(field), field)
    crossfit = receipt.get("crossfit")
    crossfit_keys = {
        "fold",
        "fit_item_count",
        "held_item_count",
        "held_pair_count",
        "held_non_tie_pair_count",
        "held_preference_correct_count",
        "held_pair_mean_squared_error",
        "fit_model_sha256",
    }
    if (
        not isinstance(crossfit, list)
        or len(crossfit) != FOLD_COUNT
        or any(not isinstance(row, Mapping) for row in crossfit)
        or [row.get("fold") for row in crossfit] != list(range(FOLD_COUNT))
        or any(set(row) != crossfit_keys for row in crossfit)
    ):
        raise FeverousEvaluatorError("fit receipt cross-fit drifted")
    held_total = 0
    for row in crossfit:
        held = row.get("held_item_count")
        fit = row.get("fit_item_count")
        non_tie = row.get("held_non_tie_pair_count")
        correct = row.get("held_preference_correct_count")
        if (
            type(held) is not int
            or type(fit) is not int
            or held < 0
            or fit < 1
            or fit + held != BLOCK_ITEM_COUNTS["A_form"]
            or row.get("held_pair_count") != 6 * held
            or type(non_tie) is not int
            or type(correct) is not int
            or not 0 <= correct <= non_tie <= 6 * held
        ):
            raise FeverousEvaluatorError("fit receipt cross-fit counts drifted")
        _to_decimal(row.get("held_pair_mean_squared_error"), "cross-fit MSE")
        _require_sha256(row.get("fit_model_sha256"), "cross-fit model hash")
        held_total += held
    if held_total != BLOCK_ITEM_COUNTS["A_form"]:
        raise FeverousEvaluatorError("fit receipt fold coverage drifted")
    payload = receipt.get("model")
    if not isinstance(payload, Mapping) or set(payload) != {
        "population_mean",
        "population_std",
        "beta",
    }:
        raise FeverousEvaluatorError("fit receipt model payload drifted")
    try:
        model = E2Model(
            tuple(_to_decimal(value) for value in payload["population_mean"]),
            tuple(_to_decimal(value) for value in payload["population_std"]),
            tuple(_to_decimal(value) for value in payload["beta"]),
        )
    except (KeyError, TypeError) as exc:
        raise FeverousEvaluatorError("fit receipt model payload drifted") from exc
    if model.payload() != dict(payload):
        raise FeverousEvaluatorError("fit receipt model encoding is noncanonical")
    return model


def verify_fit_receipt(
    receipt: Mapping[str, Any],
    *,
    traces: Sequence[RecipeTrace],
    utilities: Mapping[tuple[str, str], Fraction | int],
    fold_hmac_secret: bytes,
    feature_receipt: Mapping[str, Any],
) -> E2Model:
    """Recompute the A_form fit so even a rehashed parameter tamper fails."""

    feature_sha = verify_feature_receipt(
        feature_receipt, block="A_form", traces=traces
    )
    declared_fit_sha = _require_sha256(
        receipt.get("fit_receipt_sha256"), "fit receipt"
    )
    model_from_fit_receipt(
        receipt,
        feature_receipt_sha256=feature_sha,
        fit_receipt_sha256=declared_fit_sha,
    )
    expected_model, expected = fit_e2_a_form(
        traces=traces,
        utilities=utilities,
        fold_hmac_secret=fold_hmac_secret,
        feature_receipt=feature_receipt,
    )
    if dict(receipt) != expected:
        raise FeverousEvaluatorError("fit receipt A_form recomputation drifted")
    return expected_model


def _global_scores(
    matrix: Sequence[tuple[str, Sequence[RecipeTrace]]], model: E2Model
) -> tuple[dict[str, Fraction], dict[str, Decimal]]:
    e0_by_item = e0_item_scores(
        [trace for _, rows in matrix for trace in rows]
    )
    e0 = {
        recipe: sum(
            (e0_by_item[item][recipe] for item, _ in matrix), Fraction(0)
        )
        / len(matrix)
        for recipe in RECIPE_IDS
    }
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        e2 = {
            recipe: sum(
                (
                    model.predict(next(row for row in rows if row.recipe_id == recipe).features)
                    for _, rows in matrix
                ),
                Decimal(0),
            )
            / Decimal(len(matrix))
            for recipe in RECIPE_IDS
        }
    return e0, e2


def freeze_f_policies(
    *,
    traces: Sequence[RecipeTrace],
    feature_receipt: Mapping[str, Any],
    fit_receipt: Mapping[str, Any],
    expected_a_form_feature_receipt_sha256: str,
    expected_fit_receipt_sha256: str,
) -> dict[str, Any]:
    """Freeze both global F policies using features only (there is no label input)."""

    matrix = _normalize_matrix(traces)
    f_feature_sha = verify_feature_receipt(
        feature_receipt, block="F_search", traces=traces
    )
    a_feature_sha = _require_sha256(
        expected_a_form_feature_receipt_sha256,
        "externally frozen A_form feature receipt",
    )
    model = model_from_fit_receipt(
        fit_receipt,
        feature_receipt_sha256=a_feature_sha,
        fit_receipt_sha256=expected_fit_receipt_sha256,
    )
    fit_sha = _require_sha256(
        expected_fit_receipt_sha256, "externally frozen fit receipt"
    )
    e0, e2 = _global_scores(matrix, model)
    e0_recipe = min(RECIPE_IDS, key=lambda recipe: (-e0[recipe], recipe))
    e2_recipe = min(RECIPE_IDS, key=lambda recipe: (-e2[recipe], recipe))
    same_recipe = e0_recipe == e2_recipe
    identical_behavior = all(
        next(row for row in rows if row.recipe_id == e0_recipe).behavior_sha256
        == next(row for row in rows if row.recipe_id == e2_recipe).behavior_sha256
        for _, rows in matrix
    )
    unidentifiable = same_recipe or identical_behavior
    body = {
        "schema": f"{VERSION}_policy_receipt",
        "version": VERSION,
        "block": "F_search",
        "F_feature_receipt_sha256": f_feature_sha,
        "A_form_feature_receipt_sha256": a_feature_sha,
        "fit_receipt_sha256": fit_sha,
        "trace_matrix_sha256": feature_receipt["trace_matrix_sha256"],
        "item_count": len(matrix),
        "recipe_registry": list(RECIPE_IDS),
        "E0_recipe_scores": {
            recipe: _fraction_payload(e0[recipe]) for recipe in RECIPE_IDS
        },
        "E2_recipe_scores": {
            recipe: _decimal_text(e2[recipe]) for recipe in RECIPE_IDS
        },
        "E0_selected_recipe_id": e0_recipe,
        "E2_selected_recipe_id": e2_recipe,
        "ascending_recipe_id_tie_break": True,
        "same_recipe": same_recipe,
        "identical_all_F_ordered_top5": identical_behavior,
        "status": (
            "valid_unidentifiable_nonpromotion"
            if unidentifiable
            else "formed_distinct_frozen_policies"
        ),
        "A_hold_authorized": True,
        "A_hold_primary_authorized": True,
        "A_hold_evaluator_comparison_identifiable": not unidentifiable,
        "M_search_authorized_before_A_hold_promotion": False,
        "runner_up_or_objective_change_authorized": False,
        "labels_gold_utility_or_family_accessed": False,
        "online_evaluator_calls": 0,
        "raw_content_persisted": False,
    }
    return _self_hashed(body, "policy_receipt_sha256")


def verify_policy_receipt(
    receipt: Mapping[str, Any],
    *,
    traces: Sequence[RecipeTrace],
    feature_receipt: Mapping[str, Any],
    fit_receipt: Mapping[str, Any],
    expected_a_form_feature_receipt_sha256: str,
    expected_fit_receipt_sha256: str,
) -> str:
    declared = _verify_self_hash(
        receipt,
        schema=f"{VERSION}_policy_receipt",
        field="policy_receipt_sha256",
    )
    expected = freeze_f_policies(
        traces=traces,
        feature_receipt=feature_receipt,
        fit_receipt=fit_receipt,
        expected_a_form_feature_receipt_sha256=(
            expected_a_form_feature_receipt_sha256
        ),
        expected_fit_receipt_sha256=expected_fit_receipt_sha256,
    )
    if dict(receipt) != expected:
        raise FeverousEvaluatorError("F policy receipt semantic binding drifted")
    return declared


def item_utility(
    top5: Sequence[str | int], canonical_gold: Sequence[str | int]
) -> FeverousItemUtility:
    """Return exact canonical-element recall@5 plus the complete-set bonus."""

    if len(top5) != 5 or len(set(top5)) != 5:
        raise FeverousEvaluatorError("top5 must contain exactly five distinct units")
    if not 2 <= len(canonical_gold) <= 5 or len(set(canonical_gold)) != len(
        canonical_gold
    ):
        raise FeverousEvaluatorError("canonical gold must contain two through five units")
    if any(isinstance(value, bool) or not isinstance(value, (str, int)) for value in top5):
        raise FeverousEvaluatorError("top5 unit identity type is invalid")
    if any(
        isinstance(value, bool) or not isinstance(value, (str, int))
        for value in canonical_gold
    ):
        raise FeverousEvaluatorError("gold unit identity type is invalid")
    hits = len(set(top5).intersection(canonical_gold))
    complete = hits == len(canonical_gold)
    utility = Fraction(hits, len(canonical_gold)) + int(complete)
    return FeverousItemUtility(hits, len(canonical_gold), complete, utility)


def exact_magnitude_preserving_sign_flip(
    deltas: Sequence[Fraction | int],
) -> ExactSignFlipResult:
    """Compute an exact one-sided paired sign-flip test for rational utilities."""

    if not deltas:
        raise FeverousEvaluatorError("paired utility delta vector is empty")
    normalized: list[Fraction] = []
    for value in deltas:
        if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
            raise FeverousEvaluatorError("paired utility deltas must be exact rationals")
        normalized.append(Fraction(value))
    common_denominator = 1
    for value in normalized:
        common_denominator = math.lcm(common_denominator, value.denominator)
    integer_deltas = [
        value.numerator * (common_denominator // value.denominator)
        for value in normalized
    ]
    observed = sum(integer_deltas)
    magnitudes = [abs(value) for value in integer_deltas if value]
    distribution: Counter[int] = Counter({0: 1})
    for magnitude in magnitudes:
        updated: Counter[int] = Counter()
        for subtotal, count in distribution.items():
            updated[subtotal + magnitude] += count
            updated[subtotal - magnitude] += count
        distribution = updated
    p_value = Fraction(
        sum(count for subtotal, count in distribution.items() if subtotal >= observed),
        1 << len(magnitudes),
    )
    net = sum(normalized, Fraction(0))
    return ExactSignFlipResult(
        observed_net_u=net,
        nonzero_pair_count=len(magnitudes),
        exact_p=p_value,
        promoted=net > 0 and p_value <= PROMOTION_ALPHA,
    )


__all__ = [
    "BLOCK_ITEM_COUNTS",
    "DECIMAL_PRECISION",
    "E2Model",
    "ExactSignFlipResult",
    "FEATURE_ORDER",
    "FOLD_COUNT",
    "FORBIDDEN_FEATURES",
    "FeverousEvaluatorError",
    "FeverousItemUtility",
    "PAIR_WEIGHT",
    "PROMOTION_ALPHA",
    "RECIPE_IDS",
    "RIDGE_LAMBDA",
    "RecipeTrace",
    "average_midranks",
    "build_feature_receipt",
    "e0_item_scores",
    "exact_magnitude_preserving_sign_flip",
    "fit_e2_a_form",
    "freeze_f_policies",
    "item_utility",
    "model_from_fit_receipt",
    "verify_feature_receipt",
    "verify_fit_receipt",
    "verify_policy_receipt",
]
