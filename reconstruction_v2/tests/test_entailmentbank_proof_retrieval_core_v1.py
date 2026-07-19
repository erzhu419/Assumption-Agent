from __future__ import annotations

from dataclasses import replace

import pytest

from assumption_agent.benchmarks import entailmentbank_proof_retrieval_core_v1 as core


def _tensor(index: int) -> core.ItemTensor:
    commitment = f"{index + 1:064x}"
    rows = []
    for ordinal in range(25):
        descending = round((24 - ordinal) * core.INTEGER_SCALE / 24)
        ascending = round(ordinal * core.INTEGER_SCALE / 24)
        rows.append(
            (
                descending,
                ascending,
                ascending,
                descending,
                descending,
                ascending,
                descending,
                (ordinal % 5) * 100_000,
            )
        )
    pair = [[0] * 25 for _ in range(25)]
    for ordinal in range(25):
        pair[ordinal][ordinal] = core.INTEGER_SCALE
    pair[0][24] = pair[24][0] = core.INTEGER_SCALE
    pair[1][23] = pair[23][1] = 900_000
    return core.ItemTensor(
        commitment,
        tuple(rows),
        tuple(tuple(row) for row in pair),
    )


def _item(index: int) -> core.LabelFreeItem:
    return core.LabelFreeItem(
        f"{index + 1:064x}",
        f"Which fact answers private question {index}?",
        f"private answer {index}",
        f"private hypothesis {index} alpha beta",
        tuple(f"private node {index} {ordinal} alpha" for ordinal in range(25)),
    )


def _label(index: int) -> core.ItemLabel:
    return core.ItemLabel(f"{index + 1:064x}", "TWO_LEAF", (0, 1))


def _g_model(count: int = 6) -> core.QuantizedRidgeModel:
    return core.fit_g_model(
        tuple(_tensor(index) for index in range(count)),
        tuple(_label(index) for index in range(count)),
    )


def test_token_f1_and_pair_matrix_are_exact_and_symmetric() -> None:
    assert core.tokens(" A, a BETA 12 ") == frozenset({"a", "beta", "12"})
    assert core.token_f1("alpha beta", "beta gamma") == 500_000
    texts = tuple(f"fact {ordinal} common" for ordinal in range(25))
    matrix = core.build_pair_token_f1(texts)
    assert matrix[0][0] == core.INTEGER_SCALE
    assert matrix[0][1] == matrix[1][0] == 666_667


def test_recipe_registry_is_frozen_cartesian_16_and_actions_are_deterministic() -> None:
    assert len(core.RECIPE_REGISTRY) == 16
    assert len({recipe.recipe_id for recipe in core.RECIPE_REGISTRY}) == 16
    assert {
        (recipe.seed, recipe.alpha) for recipe in core.RECIPE_REGISTRY
    } == {
        (seed, alpha)
        for seed in core.SEED_REGISTRY
        for alpha in core.ALPHA_REGISTRY
    }
    tensor = _tensor(0)
    model = _g_model()
    nli_recipe = next(
        recipe.recipe_id
        for recipe in core.RECIPE_REGISTRY
        if recipe.seed == "NLI_HYPOTHESIS" and recipe.alpha == 0
    )
    minilm_recipe = next(
        recipe.recipe_id
        for recipe in core.RECIPE_REGISTRY
        if recipe.seed == "MINILM_HYPOTHESIS" and recipe.alpha == 0
    )
    first = core.execute_recipe(tensor, model, nli_recipe)
    assert first == core.execute_recipe(tensor, model, nli_recipe)
    assert first.selected_ordinals == (0, 1, 2, 3, 4)
    assert core.execute_recipe(tensor, model, minilm_recipe).selected_ordinals == (
        24,
        23,
        22,
        21,
        20,
    )
    with pytest.raises(core.EntailmentBankCoreError, match="self hash"):
        replace(first, selected_ordinals=(0, 1, 2, 3, 5))


def test_g_model_is_quantized_roundtrippable_and_bool_payload_fails_closed() -> None:
    model = _g_model()
    assert model.feature_count == core.NODE_FEATURE_COUNT
    assert model.training_row_count == 150
    assert core.QuantizedRidgeModel.from_payload(model.payload()) == model
    bad = model.payload()
    bad["feature_count"] = True
    with pytest.raises(core.EntailmentBankCoreError, match="payload"):
        core.QuantizedRidgeModel.from_payload(bad)


def test_direct_utility_and_evaluator_model_cover_the_full_recipe_registry() -> None:
    model = _g_model()
    items = tuple(_item(index) for index in range(3))
    tensors = tuple(_tensor(index) for index in range(3))
    labels = tuple(_label(index) for index in range(3))
    feature_registry: dict[str, dict[str, tuple[int, ...]]] = {}
    training_rows = []
    targets = []
    for item, tensor, label in zip(items, tensors, labels, strict=True):
        per_recipe = {}
        for recipe in core.RECIPE_REGISTRY:
            action = core.execute_recipe(tensor, model, recipe.recipe_id)
            features = core.evaluator_features(item, tensor, model, action)
            assert len(features) == core.EVALUATOR_FEATURE_COUNT
            per_recipe[recipe.recipe_id] = features
            training_rows.append(features)
            targets.append(core.direct_utility(action.selected_ordinals, label))
        feature_registry[item.item_commitment_sha256] = per_recipe
    assert core.direct_utility((0, 1, 2, 3, 4), labels[0]) == 3
    assert core.direct_utility((0, 2, 3, 4, 5), labels[0]) == 1
    e1 = core.fit_e1_model(training_rows, targets)
    assert e1.feature_count == core.EVALUATOR_FEATURE_COUNT
    q0, e0_totals = core.select_global_recipe(feature_registry, evaluator="E0")
    q1, e1_totals = core.select_global_recipe(
        feature_registry, evaluator="E1", e1_model=e1
    )
    assert q0 in core.RECIPE_BY_ID and q1 in core.RECIPE_BY_ID
    assert set(e0_totals) == set(e1_totals) == set(core.RECIPE_BY_ID)


def test_exact_one_sided_signflip_uses_nonzero_magnitude_dp() -> None:
    assert core.exact_one_sided_signflip((0, 0)) == {
        "observed_sum": 0,
        "nonzero_pair_count": 0,
        "tail_numerator": 1,
        "tail_denominator": 1,
    }
    assert core.exact_one_sided_signflip((1, 1, 1, 1)) == {
        "observed_sum": 4,
        "nonzero_pair_count": 4,
        "tail_numerator": 1,
        "tail_denominator": 16,
    }
    mixed = core.exact_one_sided_signflip((2, -1, 0))
    assert mixed["observed_sum"] == 1
    assert mixed["nonzero_pair_count"] == 2
    assert mixed["tail_numerator"] == 1
    assert mixed["tail_denominator"] == 2
