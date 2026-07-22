from __future__ import annotations

from fractions import Fraction
import inspect
import itertools

import numpy as np
import pytest

from assumption_agent.benchmarks import tatqa_p20_typed_evaluator_core_v1 as core


def _plan() -> core.TypedPlan:
    return core.TypedPlan(
        entity_facets=("Acme",),
        metric_facets=("revenue",),
        time_facets=("2024",),
        operation="COMPARE",
        relation_query="Acme revenue in 2024",
    )


def _unit(
    unit_id: str,
    facets: tuple[int, int, int, int],
    *,
    edges: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0),
    operand: int = 0,
    similarity: int = 0,
) -> core.CanonicalUnit:
    return core.CanonicalUnit(
        unit_id=unit_id,
        facet_coverage=facets,
        typed_edge_features=edges,
        numeric_or_time_operand_coverage=operand,
        full_question_similarity=similarity,
    )


def _units() -> tuple[core.CanonicalUnit, ...]:
    # P:1 and P:2 are deliberately below P0's top five.  Their fourth typed
    # edge coordinate gives them query-anchored cross-modal residual authority.
    return (
        _unit("T:0", (1, 1, 1, 1), edges=(5, 0, 0, 0, 0), similarity=100),
        _unit("T:1", (1, 1, 1, 0), edges=(4, 0, 0, 0, 0), similarity=90),
        _unit("P:3", (1, 1, 0, 0), edges=(3, 0, 0, 0, 0), similarity=80),
        _unit("T:2", (1, 0, 1, 0), edges=(2, 0, 0, 0, 0), operand=1, similarity=70),
        _unit("T:3", (1, 0, 1, 0), edges=(2, 0, 0, 0, 0), operand=1, similarity=60),
        _unit("P:1", (0, 0, 0, 0), edges=(0, 0, 0, 9, 0), operand=5, similarity=1),
        _unit("P:2", (0, 0, 0, 0), edges=(0, 0, 0, 9, 0), operand=5, similarity=1),
    )


def test_strict_plan_validation_and_totalizer_are_deterministic_and_total() -> None:
    mapping = _plan().payload()
    assert core.validate_typed_plan(mapping) == _plan()
    with pytest.raises(core.TatqaP20TypedEvaluatorError, match="schema drifted"):
        core.validate_typed_plan({**mapping, "answer": "forbidden"})
    with pytest.raises(core.TatqaP20TypedEvaluatorError, match="between 1 and 4"):
        core.validate_typed_plan({**mapping, "entity_facets": []})

    malformed = {
        "entity_facets": ["  Acme  ", "acme", 7, "Subsidiary"],
        "metric_facets": [],
        "time_facets": [" 2024 ", "2024", None],
        "operation": " ratio ",
        "relation_query": "",
        "ignored_extra": {"anything": True},
    }
    kwargs = {
        "fallback_relation_query": "Which metric relates Acme and Subsidiary?",
        "fallback_metric_facets": ("margin",),
    }
    first = core.totalize_typed_plan(malformed, **kwargs)
    second = core.totalize_typed_plan(malformed, **kwargs)
    assert first == second
    assert first.entity_facets == ("Acme", "Subsidiary")
    assert first.metric_facets == ("margin",)
    assert first.time_facets == ("2024",)
    assert first.operation == "RATIO"
    assert first.relation_query == "Which metric relates Acme and Subsidiary?"

    for arbitrary in (None, 4, [], object(), {}, {"operation": object()}):
        assert isinstance(core.totalize_typed_plan(arbitrary), core.TypedPlan)


def test_canonical_unit_and_plan_width_validation_fail_closed() -> None:
    with pytest.raises(core.TatqaP20TypedEvaluatorError, match="canonical unit ID"):
        _unit("table-0", (1, 1, 1, 1))
    with pytest.raises(core.TatqaP20TypedEvaluatorError, match="zero or one"):
        _unit("T:0", (1, 2, 0, 0))
    wrong_width = tuple(
        _unit(f"T:{index}", (1, 0, 0)) for index in range(5)
    )
    with pytest.raises(core.TatqaP20TypedEvaluatorError, match="coverage width"):
        core.build_p0_action(_plan(), wrong_width)


def test_p0_order_and_p1_residual_can_expand_outside_p0_top5() -> None:
    plan = _plan()
    units = _units()
    full_p0 = core.rank_p0_units(plan, units)
    assert full_p0[:5] == ("T:0", "T:1", "P:3", "T:2", "T:3")
    p0, p1 = core.build_action_pair(plan, units)
    assert p0.selected_unit_ids == full_p0[:5]
    assert p1.selected_unit_ids == ("T:0", "T:1", "P:3", "P:1", "P:2")
    assert p1.selected_unit_ids[:3] == p0.selected_unit_ids[:3]
    assert set(p1.selected_unit_ids[3:]).isdisjoint(p0.selected_unit_ids)
    assert p1.feature_mapping()["P1_outside_P0_unit_count"] == 2
    assert p1.feature_mapping()["cross_modal_query_anchored_path_delta"] == 18


def test_p1_query_independent_components_have_zero_authority_and_totalize_by_p0() -> None:
    plan = _plan()
    units = list(_units()[:5])
    # Large operand/relevance values alone are not query-anchored authority.
    units.extend(
        (
            _unit("P:1", (0, 0, 0, 0), operand=1_000, similarity=1_000),
            _unit("P:2", (0, 0, 0, 0), operand=2_000, similarity=2_000),
        )
    )
    p0 = core.build_p0_action(plan, units)
    p1 = core.build_p1_action(plan, units, p0)
    assert p1.selected_unit_ids == p0.selected_unit_ids
    assert p1.feature_vector == (0,) * len(core.FEATURE_ORDER)


def test_fixed_feature_order_rejects_gold_answer_family_and_baseline_extras() -> None:
    assert core.FEATURE_ORDER == (
        "typed_facet_coverage_delta",
        "numeric_or_time_operand_coverage_delta",
        "cross_modal_query_anchored_path_delta",
        "dense_relevance_mass_delta",
        "selected_unit_redundancy_delta",
        "P1_outside_P0_unit_count",
    )
    clean = {name: 0 for name in core.FEATURE_ORDER}
    assert core.feature_vector(clean) == (0,) * len(core.FEATURE_ORDER)
    for forbidden in (
        "answer",
        "family",
        "gold_unit",
        "HippoRAG_ranking",
        "RAW_ranking",
        "baseline_score",
    ):
        with pytest.raises(core.TatqaP20TypedEvaluatorError, match="schema drifted"):
            core.feature_vector({**clean, forbidden: 1})

    dataclass_fields = set(core.CanonicalUnit.__dataclass_fields__)
    assert not dataclass_fields.intersection(
        {"gold", "answer", "family", "raw_ranking", "hipporag_ranking"}
    )
    for function in (
        core.rank_p0_units,
        core.rank_p1_units,
        core.p1_minus_p0_features,
        core.build_action_pair,
        core.fit_paired_delta_ridge,
        core.select_evaluator_action,
    ):
        names = set(inspect.signature(function).parameters)
        assert not names.intersection(
            {"gold", "answer", "family", "raw", "hipporag", "baseline"}
        )


def test_redundancy_delta_uses_canonical_undirected_pairs() -> None:
    plan = _plan()
    units = _units()
    p0 = core.build_p0_action(plan, units)
    redundancy = {
        ("T:2", "T:3"): 7,
        ("P:1", "P:2"): 2,
    }
    p1 = core.build_p1_action(
        plan, units, p0, redundancy_features=redundancy
    )
    assert p1.feature_mapping()["selected_unit_redundancy_delta"] == -5
    with pytest.raises(core.TatqaP20TypedEvaluatorError, match="reverse"):
        core.build_p1_action(
            plan,
            units,
            p0,
            redundancy_features={
                ("T:2", "T:3"): 1,
                ("T:3", "T:2"): 1,
            },
        )


def test_lambda_one_population_standardized_ridge_matches_closed_form() -> None:
    x_values = (-2, -1, 1, 2)
    rows = [(value, 0, 0, 0, 0, 0) for value in x_values]
    targets = tuple(Fraction(value) for value in x_values)
    model = core.fit_paired_delta_ridge(rows, targets)
    assert model.solver == "numpy_float64_solve_v1"
    assert model.intercept == pytest.approx(0.0, abs=1e-15)
    assert model.population_mean == (0.0,) * len(core.FEATURE_ORDER)
    assert model.population_std[1:] == (0.0,) * (len(core.FEATURE_ORDER) - 1)
    z = np.asarray(x_values, dtype=np.float64) / model.population_std[0]
    expected = float(np.dot(z, np.asarray(x_values)) / (np.dot(z, z) + 1.0))
    assert model.coefficients[0] == pytest.approx(expected, rel=1e-14)
    assert model.coefficients[1:] == (0.0,) * (len(core.FEATURE_ORDER) - 1)


def test_ridge_intercept_is_unpenalized_and_zero_variance_maps_to_zero() -> None:
    rows = [(7, 7, 7, 7, 7, 7)] * 3
    model = core.fit_paired_delta_ridge(
        rows, (Fraction(1), Fraction(2), Fraction(3))
    )
    assert model.intercept == pytest.approx(2.0)
    assert model.population_std == (0.0,) * len(core.FEATURE_ORDER)
    assert model.coefficients == (0.0,) * len(core.FEATURE_ORDER)
    assert model.predict((999, 999, 999, 999, 999, 999)) == pytest.approx(2.0)
    with pytest.raises(core.TatqaP20TypedEvaluatorError, match="exact"):
        core.fit_paired_delta_ridge(rows, (1.0, 2.0, 3.0))


def test_e0_e1_selector_uses_strict_positive_prediction_and_no_tie_switch() -> None:
    p0, p1 = core.build_action_pair(_plan(), _units())
    zero = core.PairedDeltaRidgeModel(
        population_mean=(0.0,) * len(core.FEATURE_ORDER),
        population_std=(1.0,) * len(core.FEATURE_ORDER),
        intercept=0.0,
        coefficients=(0.0,) * len(core.FEATURE_ORDER),
        solver="numpy_float64_solve_v1",
    )
    positive = core.PairedDeltaRidgeModel(
        population_mean=(0.0,) * len(core.FEATURE_ORDER),
        population_std=(1.0,) * len(core.FEATURE_ORDER),
        intercept=1.0,
        coefficients=(0.0,) * len(core.FEATURE_ORDER),
        solver="numpy_float64_solve_v1",
    )
    assert core.select_e0_action(p0, p1) is p0
    assert core.select_e1_action(zero, p0, p1) is p0
    assert core.select_e1_action(positive, p0, p1) is p1
    assert core.select_evaluator_action(
        "E1", p0_action=p0, p1_action=p1, model=positive
    ) is p1


def test_exact_fraction_utility() -> None:
    top5 = ("T:0", "T:1", "T:2", "P:3", "P:1")
    assert core.item_utility(top5, ("T:0", "P:1")) == Fraction(2)
    assert core.item_utility(top5, ("T:0", "P:2", "P:4")) == Fraction(1, 3)
    assert isinstance(core.item_utility(top5, ("T:0",)), Fraction)
    with pytest.raises(core.TatqaP20TypedEvaluatorError, match="one through five"):
        core.item_utility(top5, ())


def _brute_sign_flip(deltas: tuple[Fraction, ...]) -> Fraction:
    denominator = 1
    for value in deltas:
        denominator = np.lcm(denominator, value.denominator)
    integers = tuple(
        value.numerator * (int(denominator) // value.denominator) for value in deltas
    )
    observed = sum(integers)
    magnitudes = tuple(abs(value) for value in integers if value)
    possible = tuple(
        sum(sign * magnitude for sign, magnitude in zip(signs, magnitudes))
        for signs in itertools.product((-1, 1), repeat=len(magnitudes))
    )
    return Fraction(sum(value >= observed for value in possible), len(possible))


@pytest.mark.parametrize(
    "deltas",
    (
        (Fraction(0), Fraction(0)),
        (Fraction(1, 2),) * 4,
        (Fraction(3, 5), Fraction(-1, 2), Fraction(2, 3), Fraction(0)),
        (Fraction(-3), Fraction(-1), Fraction(1)),
    ),
)
def test_exact_magnitude_preserving_sign_flip_matches_brute_force(
    deltas: tuple[Fraction, ...]
) -> None:
    result = core.exact_magnitude_preserving_sign_flip(deltas)
    expected = _brute_sign_flip(deltas)
    assert result.observed_net_u == sum(deltas, Fraction(0))
    assert result.exact_p == expected
    assert result.promoted is (
        result.observed_net_u > 0 and expected <= Fraction(1, 10)
    )


def test_canonical_action_and_behavior_hashes_freeze_ties_and_order() -> None:
    p0, p1 = core.build_action_pair(_plan(), _units())
    assert p0.action_sha256 == core.canonical_action_hash(p0)
    assert p0.behavior_sha256 == core.canonical_behavior_hash(p0)
    assert p0.action_sha256 == core.canonical_action_hash(
        core.Action(
            policy_id=p0.policy_id,
            plan=core.validate_typed_plan(p0.plan.payload()),
            selected_unit_ids=tuple(p0.selected_unit_ids),
            feature_vector=tuple(p0.feature_vector),
        )
    )
    assert p0.action_sha256 != p1.action_sha256

    # Behavior deliberately ignores policy and plan internals but binds order.
    p1_same_behavior = core.Action(
        policy_id=core.P1_POLICY_ID,
        plan=p0.plan,
        selected_unit_ids=p0.selected_unit_ids,
        feature_vector=(0,) * len(core.FEATURE_ORDER),
    )
    assert p0.action_sha256 != p1_same_behavior.action_sha256
    assert p0.behavior_sha256 == p1_same_behavior.behavior_sha256
    reordered = (
        p0.selected_unit_ids[1],
        p0.selected_unit_ids[0],
        *p0.selected_unit_ids[2:],
    )
    assert core.canonical_behavior_hash(reordered) != p0.behavior_sha256
