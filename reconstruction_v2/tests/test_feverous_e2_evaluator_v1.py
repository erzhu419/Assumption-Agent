from __future__ import annotations

import copy
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
import hashlib
import inspect
import json

import pytest

from assumption_agent.benchmarks.feverous_e2_evaluator_v1 import (
    BLOCK_ITEM_COUNTS,
    DECIMAL_PRECISION,
    FEATURE_ORDER,
    RECIPE_IDS,
    FeverousEvaluatorError,
    RecipeTrace,
    build_feature_receipt,
    e0_item_scores,
    exact_magnitude_preserving_sign_flip,
    fit_e2_a_form,
    freeze_f_policies,
    item_utility,
    verify_feature_receipt,
    verify_fit_receipt,
    verify_policy_receipt,
)


SECRET = b"synthetic-fixed-private-fold-secret-v1"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _trace(
    item: int,
    recipe_index: int,
    features: list[int | Decimal],
    *,
    behavior: str | None = None,
) -> RecipeTrace:
    return RecipeTrace.from_mapping(
        item_commitment_sha256=_sha(f"item-{item}"),
        recipe_id=RECIPE_IDS[recipe_index],
        behavior_sha256=_sha(behavior or f"behavior-{item}-{recipe_index}"),
        features=dict(zip(FEATURE_ORDER, features)),
    )


def _matrix(
    item_count: int,
    feature_fn,
    *,
    behavior_fn=None,
) -> list[RecipeTrace]:
    return [
        _trace(
            item,
            recipe,
            list(feature_fn(item, recipe)),
            behavior=(None if behavior_fn is None else behavior_fn(item, recipe)),
        )
        for item in range(item_count)
        for recipe in range(len(RECIPE_IDS))
    ]


def _utilities(item_count: int, value_fn) -> dict[tuple[str, str], Fraction]:
    return {
        (_sha(f"item-{item}"), RECIPE_IDS[recipe]): Fraction(value_fn(item, recipe))
        for item in range(item_count)
        for recipe in range(len(RECIPE_IDS))
    }


def _rehash(receipt: dict[str, object], field: str) -> None:
    body = dict(receipt)
    body.pop(field)
    raw = json.dumps(
        body,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    receipt[field] = hashlib.sha256(raw).hexdigest()


def _one_dimensional_a_form(item_count: int = BLOCK_ITEM_COUNTS["A_form"]):
    traces = _matrix(
        item_count,
        lambda _item, recipe: [3 - recipe, 0, 0, 0, 0, 0, 0, 0],
    )
    utilities = _utilities(
        item_count,
        lambda _item, recipe: Fraction(3 - recipe, 3),
    )
    feature_receipt = build_feature_receipt(block="A_form", traces=traces)
    model, fit_receipt = fit_e2_a_form(
        traces=traces,
        utilities=utilities,
        fold_hmac_secret=SECRET,
        feature_receipt=feature_receipt,
    )
    return traces, utilities, feature_receipt, model, fit_receipt


def test_e0_balanced_midranks_prevent_one_coordinate_lexicographic_domination() -> None:
    traces = [
        _trace(0, 0, [100, 0, 0, 0, 0, 0, 0, 0]),
        _trace(0, 1, [1, 1, 1, 1, 1, 1, 1, 1]),
        _trace(0, 2, [0, 0, 0, 0, 0, 0, 0, 0]),
        _trace(0, 3, [0, 0, 0, 0, 0, 0, 0, 0]),
    ]
    scores = e0_item_scores(traces)[_sha("item-0")]
    assert scores[RECIPE_IDS[0]] == Fraction(9, 4)
    assert scores[RECIPE_IDS[1]] == Fraction(31, 8)
    assert scores[RECIPE_IDS[1]] > scores[RECIPE_IDS[0]]


def test_positive_coordinate_rescale_preserves_e0_and_e2_standardized_fit() -> None:
    item_count = BLOCK_ITEM_COUNTS["A_form"]

    def features(item: int, recipe: int) -> list[int]:
        return [
            5 * recipe + item,
            (recipe + 2 * item) % 7,
            2 * recipe - item,
            recipe * (item % 3),
            item - recipe,
            (item + recipe) % 5,
            11 + recipe,
            -(item + 3 * recipe),
        ]

    traces = _matrix(item_count, features)
    scaled = _matrix(
        item_count,
        lambda item, recipe: [100 * features(item, recipe)[0], *features(item, recipe)[1:]],
    )
    utilities = _utilities(
        item_count,
        lambda item, recipe: Fraction((item + 2 * recipe) % 7, 6),
    )
    assert e0_item_scores(traces) == e0_item_scores(scaled)
    base_features = build_feature_receipt(block="A_form", traces=traces)
    scaled_features = build_feature_receipt(block="A_form", traces=scaled)
    base_model, _ = fit_e2_a_form(
        traces=traces,
        utilities=utilities,
        fold_hmac_secret=SECRET,
        feature_receipt=base_features,
    )
    scaled_model, _ = fit_e2_a_form(
        traces=scaled,
        utilities=utilities,
        fold_hmac_secret=SECRET,
        feature_receipt=scaled_features,
    )
    assert base_model.beta == scaled_model.beta
    for base, changed in zip(traces, scaled):
        assert base_model.predict(base.features) == scaled_model.predict(changed.features)


def test_lambda_one_weighted_ridge_matches_closed_form_and_recovers_direction() -> None:
    traces, _utilities_map, _feature_receipt, model, _fit = _one_dimensional_a_form()
    assert model.beta[0] > 0
    assert all(value == 0 for value in model.beta[1:])

    # Exact 1/6 rescaling: sum(x*y)/(sum(x^2)+6).
    by_item: dict[str, dict[str, RecipeTrace]] = {}
    for trace in traces:
        by_item.setdefault(trace.item_commitment_sha256, {})[trace.recipe_id] = trace
    numerator = Decimal(0)
    denominator = Decimal(6)
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        for rows in by_item.values():
            z = {
                recipe: model.standardize(rows[recipe].features)[0]
                for recipe in RECIPE_IDS
            }
            values = {
                recipe: Decimal(3 - index) / Decimal(3)
                for index, recipe in enumerate(RECIPE_IDS)
            }
            for left_index, left in enumerate(RECIPE_IDS):
                for right in RECIPE_IDS[left_index + 1 :]:
                    x = z[left] - z[right]
                    y = values[left] - values[right]
                    numerator += x * y
                    denominator += x * x
        expected = numerator / denominator
    # The generic pivoted solver and the scalar formula round in a different
    # final operation order; they agree to the declared 79 significant digits.
    assert abs(model.beta[0] - expected) <= Decimal("1e-79")


def test_population_zero_variance_dimensions_map_to_zero() -> None:
    item_count = BLOCK_ITEM_COUNTS["A_form"]
    traces = _matrix(item_count, lambda _item, _recipe: [7] * len(FEATURE_ORDER))
    utilities = _utilities(
        item_count, lambda item, recipe: Fraction((item + recipe) % 3, 2)
    )
    feature_receipt = build_feature_receipt(block="A_form", traces=traces)
    model, receipt = fit_e2_a_form(
        traces=traces,
        utilities=utilities,
        fold_hmac_secret=SECRET,
        feature_receipt=feature_receipt,
    )
    assert model.population_std == (Decimal(0),) * len(FEATURE_ORDER)
    assert model.beta == (Decimal(0),) * len(FEATURE_ORDER)
    assert receipt["zero_variance_maps_to_zero"] is True


def test_recipe_identity_and_any_schema_extra_are_forbidden_features() -> None:
    features = {name: 0 for name in FEATURE_ORDER}
    features["recipe_id"] = 1
    with pytest.raises(FeverousEvaluatorError, match="forbidden evaluator feature"):
        RecipeTrace.from_mapping(
            item_commitment_sha256=_sha("item"),
            recipe_id=RECIPE_IDS[0],
            behavior_sha256=_sha("behavior"),
            features=features,
        )
    features.pop("recipe_id")
    features["undeclared_coordinate"] = 1
    with pytest.raises(FeverousEvaluatorError, match="feature schema drifted"):
        RecipeTrace.from_mapping(
            item_commitment_sha256=_sha("item"),
            recipe_id=RECIPE_IDS[0],
            behavior_sha256=_sha("behavior"),
            features=features,
        )


def test_crossfit_is_fixed_hmac_descriptive_and_final_fit_is_once() -> None:
    traces, utilities, feature_receipt, model, receipt = _one_dimensional_a_form()
    model_again, receipt_again = fit_e2_a_form(
        traces=list(reversed(traces)),
        utilities=utilities,
        fold_hmac_secret=SECRET,
        feature_receipt=feature_receipt,
    )
    assert model_again == model
    assert receipt_again == receipt
    assert receipt["crossfit_descriptive_only"] is True
    assert receipt["final_fit_count"] == 1
    assert [row["fold"] for row in receipt["crossfit"]] == [0, 1, 2, 3]
    assert sum(row["held_item_count"] for row in receipt["crossfit"]) == 96

    _, other_secret = fit_e2_a_form(
        traces=traces,
        utilities=utilities,
        fold_hmac_secret=b"a-different-synthetic-fold-secret-v1",
        feature_receipt=feature_receipt,
    )
    assert other_secret["fold_assignment_sha256"] != receipt["fold_assignment_sha256"]
    assert other_secret["model"] == receipt["model"]


def test_f_policy_api_has_no_label_or_utility_input_and_receipt_records_no_access() -> None:
    _a_traces, _utilities_map, a_features, _model, fit_receipt = (
        _one_dimensional_a_form()
    )
    assert "utilities" not in inspect.signature(freeze_f_policies).parameters
    assert "labels" not in inspect.signature(freeze_f_policies).parameters
    f_traces = _matrix(
        BLOCK_ITEM_COUNTS["F_search"],
        lambda _item, recipe: [3 - recipe, recipe, recipe, recipe, recipe, recipe, recipe, recipe],
    )
    f_features = build_feature_receipt(block="F_search", traces=f_traces)
    policy = freeze_f_policies(
        traces=f_traces,
        feature_receipt=f_features,
        fit_receipt=fit_receipt,
        expected_a_form_feature_receipt_sha256=a_features[
            "feature_receipt_sha256"
        ],
        expected_fit_receipt_sha256=fit_receipt["fit_receipt_sha256"],
    )
    assert policy["labels_gold_utility_or_family_accessed"] is False
    assert "utility_matrix_sha256" not in policy


def test_distinct_selected_recipes_with_same_f_behavior_are_unidentifiable() -> None:
    _a_traces, _utilities_map, a_features, _model, fit_receipt = (
        _one_dimensional_a_form()
    )

    def features(_item: int, recipe: int) -> list[int]:
        if recipe == 0:
            return [100, 0, 0, 0, 0, 0, 0, 0]
        if recipe == 1:
            return [10, 10, 10, 10, 10, 10, 10, 10]
        return [recipe, 0, 0, 0, 0, 0, 0, 0]

    f_traces = _matrix(
        BLOCK_ITEM_COUNTS["F_search"],
        features,
        behavior_fn=lambda item, recipe: (
            f"same-selected-{item}" if recipe in {0, 1} else f"other-{item}-{recipe}"
        ),
    )
    f_features = build_feature_receipt(block="F_search", traces=f_traces)
    policy = freeze_f_policies(
        traces=f_traces,
        feature_receipt=f_features,
        fit_receipt=fit_receipt,
        expected_a_form_feature_receipt_sha256=a_features[
            "feature_receipt_sha256"
        ],
        expected_fit_receipt_sha256=fit_receipt["fit_receipt_sha256"],
    )
    assert policy["E0_selected_recipe_id"] == RECIPE_IDS[1]
    assert policy["E2_selected_recipe_id"] == RECIPE_IDS[0]
    assert policy["same_recipe"] is False
    assert policy["identical_all_F_ordered_top5"] is True
    assert policy["status"] == "valid_unidentifiable_nonpromotion"
    assert policy["A_hold_authorized"] is True
    assert policy["A_hold_primary_authorized"] is True
    assert policy["A_hold_evaluator_comparison_identifiable"] is False
    assert policy["runner_up_or_objective_change_authorized"] is False


def test_feature_fit_and_policy_receipt_tampering_fails_closed_even_if_rehashed() -> None:
    traces, utilities, feature_receipt, _model, fit_receipt = (
        _one_dimensional_a_form()
    )
    feature_tamper = copy.deepcopy(feature_receipt)
    feature_tamper["labels_or_utility_accessed"] = True
    _rehash(feature_tamper, "feature_receipt_sha256")
    with pytest.raises(FeverousEvaluatorError, match="semantic binding"):
        verify_feature_receipt(feature_tamper, block="A_form", traces=traces)

    fit_tamper = copy.deepcopy(fit_receipt)
    fit_tamper["model"]["beta"][0] = "999"
    _rehash(fit_tamper, "fit_receipt_sha256")
    with pytest.raises(FeverousEvaluatorError, match="recomputation drifted"):
        verify_fit_receipt(
            fit_tamper,
            traces=traces,
            utilities=utilities,
            fold_hmac_secret=SECRET,
            feature_receipt=feature_receipt,
        )

    f_traces = _matrix(
        BLOCK_ITEM_COUNTS["F_search"],
        lambda _item, recipe: [3 - recipe, recipe, recipe, recipe, recipe, recipe, recipe, recipe],
    )
    f_features = build_feature_receipt(block="F_search", traces=f_traces)
    policy = freeze_f_policies(
        traces=f_traces,
        feature_receipt=f_features,
        fit_receipt=fit_receipt,
        expected_a_form_feature_receipt_sha256=feature_receipt[
            "feature_receipt_sha256"
        ],
        expected_fit_receipt_sha256=fit_receipt["fit_receipt_sha256"],
    )
    forged_fit = copy.deepcopy(fit_receipt)
    forged_fit["model"]["beta"][0] = "999"
    _rehash(forged_fit, "fit_receipt_sha256")
    with pytest.raises(FeverousEvaluatorError, match="external freeze"):
        freeze_f_policies(
            traces=f_traces,
            feature_receipt=f_features,
            fit_receipt=forged_fit,
            expected_a_form_feature_receipt_sha256=feature_receipt[
                "feature_receipt_sha256"
            ],
            expected_fit_receipt_sha256=fit_receipt[
                "fit_receipt_sha256"
            ],
        )
    policy_tamper = copy.deepcopy(policy)
    policy_tamper["E2_selected_recipe_id"] = RECIPE_IDS[-1]
    _rehash(policy_tamper, "policy_receipt_sha256")
    with pytest.raises(FeverousEvaluatorError, match="semantic binding"):
        verify_policy_receipt(
            policy_tamper,
            traces=f_traces,
            feature_receipt=f_features,
            fit_receipt=fit_receipt,
            expected_a_form_feature_receipt_sha256=feature_receipt[
                "feature_receipt_sha256"
            ],
            expected_fit_receipt_sha256=fit_receipt[
                "fit_receipt_sha256"
            ],
        )


def test_formal_block_sizes_and_presealed_A_form_receipt_are_mandatory() -> None:
    short_a = _matrix(12, lambda _item, _recipe: [0] * len(FEATURE_ORDER))
    short_f = _matrix(8, lambda _item, _recipe: [0] * len(FEATURE_ORDER))
    with pytest.raises(FeverousEvaluatorError, match="exactly 96"):
        build_feature_receipt(block="A_form", traces=short_a)
    with pytest.raises(FeverousEvaluatorError, match="exactly 48"):
        build_feature_receipt(block="F_search", traces=short_f)
    assert inspect.signature(fit_e2_a_form).parameters[
        "feature_receipt"
    ].default is inspect.Parameter.empty


def test_exact_utility_and_rational_signflip_use_new_feverous_types() -> None:
    utility = item_utility(["a", "b", "c", "d", "e"], ["a", "b", "z"])
    assert utility.distinct_gold_hits == 2
    assert utility.complete is False
    assert utility.value == Fraction(2, 3)
    complete = item_utility(["a", "b", "c", "d", "e"], ["a", "b"])
    assert complete.value == 2

    result = exact_magnitude_preserving_sign_flip([Fraction(1, 2)] * 5)
    assert result.observed_net_u == Fraction(5, 2)
    assert result.exact_p == Fraction(1, 32)
    assert result.promoted is True
    assert result.payload()["test"].startswith("feverous_")
