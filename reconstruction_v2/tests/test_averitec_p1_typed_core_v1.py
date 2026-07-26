from __future__ import annotations

from fractions import Fraction

from assumption_agent.benchmarks.averitec_p1_typed_core_v1 import (
    CAUSE,
    COMPARE,
    CONTEXT,
    DIRECT,
    EFFECT,
    NUMBER,
    QUERY_VARIANT_IDS,
    QUOTE,
    R0_DIRECT_DENSE,
    R1_CAUSAL_CHAIN,
    RECIPE_IDS,
    SCALE,
    SOURCE,
    AFormAction,
    AFormSlate,
    compare,
    compute_action_features,
    exact_sign_flip,
    fit_e1,
    materialize_recipe_actions,
    model_from_payload,
    model_payload,
    select_e0,
    select_e1,
    typed_query_variants,
    utility,
)


def _coordinates() -> dict[str, list[int]]:
    direct = [900_000, 800_000, 700_000, 600_000, 500_000, 400_000]
    return {
        DIRECT: direct,
        CAUSE: [100_000, 990_000, 100_000, 100_000, 100_000, 100_000],
        EFFECT: [100_000, 100_000, 980_000, 100_000, 100_000, 100_000],
        QUOTE: [100_000, 100_000, 100_000, 970_000, 100_000, 100_000],
        SOURCE: [100_000, 100_000, 100_000, 100_000, 960_000, 100_000],
        NUMBER: [100_000, 100_000, 100_000, 100_000, 100_000, 950_000],
        COMPARE: [940_000, 100_000, 100_000, 100_000, 100_000, 100_000],
        CONTEXT: [100_000, 100_000, 100_000, 930_000, 100_000, 100_000],
    }


def test_typed_queries_and_actions_are_closed_and_total() -> None:
    variants = typed_query_variants("A claim about an event and 42 percent.")
    assert tuple(variants) == QUERY_VARIANT_IDS
    documents = [f"document {index} evidence" for index in range(6)]
    actions = materialize_recipe_actions(
        document_texts=documents,
        variant_scores=_coordinates(),
    )
    assert tuple(actions) == RECIPE_IDS
    assert actions[R0_DIRECT_DENSE].top5_document_ordinals == (0, 1, 2, 3, 4)
    assert (
        actions[R1_CAUSAL_CHAIN].top5_document_ordinals
        != actions[R0_DIRECT_DENSE].top5_document_ordinals
    )
    assert len(set(actions[R1_CAUSAL_CHAIN].top5_document_ordinals)) == 5
    assert select_e0(actions) == R0_DIRECT_DENSE
    features = compute_action_features(
        action=actions[R1_CAUSAL_CHAIN],
        document_texts=documents,
    )
    assert len(features.values) == 8
    assert all(-SCALE <= value <= SCALE for value in features.values)


def test_e1_learns_a_behavior_distinct_recipe_and_roundtrips() -> None:
    documents = [f"document {index} evidence" for index in range(6)]
    actions = materialize_recipe_actions(
        document_texts=documents,
        variant_scores=_coordinates(),
    )
    slates = []
    for _ in range(12):
        rows = []
        for recipe_id in RECIPE_IDS:
            rows.append(
                AFormAction(
                    recipe_id=recipe_id,
                    features=compute_action_features(
                        action=actions[recipe_id],
                        document_texts=documents,
                    ),
                    utility=(
                        Fraction(1, 1)
                        if recipe_id == R1_CAUSAL_CHAIN
                        else Fraction(0, 1)
                    ),
                )
            )
        slates.append(AFormSlate(tuple(rows)))
    model = fit_e1(slates)
    restored = model_from_payload(model_payload(model))
    assert restored == model
    assert (
        select_e1(
            model=restored,
            actions=actions,
            document_texts=documents,
        )
        == R1_CAUSAL_CHAIN
    )


def test_exact_fraction_utility_and_reference_tail() -> None:
    assert utility(
        top5_document_ordinals=(0, 1, 2, 3, 4),
        qrel_document_ordinals=(0, 2, 5),
    ) == Fraction(2, 3)
    result = compare(
        [Fraction(1, 1), Fraction(1, 1), Fraction(1, 1), Fraction(1, 1)],
        [Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)],
    )
    assert result.net_utility == 4
    assert result.positive_count == 4
    assert result.reference_tail == Fraction(1, 16)


def test_exact_sign_flip_is_bounded_for_36_heterogeneous_denominators() -> None:
    deltas = [Fraction(1, denominator) for denominator in range(2, 38)]
    assert exact_sign_flip(deltas) == Fraction(1, 2**36)
