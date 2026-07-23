from __future__ import annotations

from fractions import Fraction
import inspect
import json
import math

import numpy as np
import pytest

from assumption_agent.benchmarks import birco_p1_typed_constraint_e4_core_v1 as core


def _facet(
    ordinal: int,
    facet_type: str,
    text: str,
    weight: int = 4,
) -> core.TypedFacet:
    return core.TypedFacet(ordinal, facet_type, text, weight)


def _plan(*, requires: bool = True, excluded: bool = True) -> core.TypedFacetPlan:
    facets = (
        _facet(0, "REQUIRED", "primary condition", 4),
        _facet(1, "REQUIRED", "dependent condition", 4),
        _facet(2, "EXCLUDED" if excluded else "PREFERRED", "negative condition", 1),
    )
    edges = (core.TypedFacetEdge(0, 1, "REQUIRES"),) if requires else ()
    return core.TypedFacetPlan(facets, edges)


def _candidate(
    ordinal: int,
    rows: tuple[tuple[int, int, int | None], ...],
    *,
    unit_count: int = 4,
) -> core.CandidateFacetEvidence:
    return core.CandidateFacetEvidence(
        candidate_ordinal=ordinal,
        evidence_unit_count=unit_count,
        facet_evidence=tuple(
            core.FacetEvidence(facet, support, contradiction, evidence)
            for facet, (support, contradiction, evidence) in enumerate(rows)
        ),
    )


def _matrix() -> core.CandidateFacetEvidenceMatrix:
    # Candidate zero has stronger mass/bottleneck/flow, but its two required
    # facets compete for one evidence unit.  Candidate one has three distinct
    # evidence units and therefore wins R4's capacity-first lexicographic key.
    return core.CandidateFacetEvidenceMatrix(
        (
            _candidate(0, ((4, 0, 0), (4, 0, 0), (0, 4, 1))),
            _candidate(1, ((3, 0, 0), (3, 0, 1), (1, 0, 2))),
            _candidate(2, ((1, 0, 0), (1, 0, 1), (1, 0, 2))),
            _candidate(3, ((1, 0, 0), (1, 0, 1), (1, 0, 2))),
        )
    )


def test_strict_plan_schema_weight_and_all_edge_types_are_one_dag() -> None:
    plan = _plan()
    assert core.validate_typed_plan(plan.payload()) == plan
    short_shape = {
        "facets": [
            {"ordinal": 0, "type": "REQUIRED", "text": "a", "weight": 4},
            {"ordinal": 1, "type": "PREFERRED", "text": "b", "weight": 1},
        ],
        "edges": [{"source": 0, "target": 1, "type": "REFINES"}],
    }
    assert core.validate_typed_plan(short_shape).facets[0].weight == 4

    with pytest.raises(core.BircoP1CoreError, match="weight"):
        core.TypedFacet(0, "REQUIRED", "bad", 5)
    drifted = plan.payload()
    drifted["facets"][0]["candidate_id"] = "forbidden"  # type: ignore[index]
    with pytest.raises(core.BircoP1CoreError, match="schema drifted"):
        core.validate_typed_plan(drifted)

    # The complete typed graph is acyclic, not just its REQUIRES projection.
    with pytest.raises(core.BircoP1CoreError, match="acyclic"):
        core.TypedFacetPlan(
            (_facet(0, "REQUIRED", "a"), _facet(1, "REQUIRED", "b")),
            (
                core.TypedFacetEdge(0, 1, "REQUIRES"),
                core.TypedFacetEdge(1, 0, "CONTRASTS_WITH"),
            ),
        )


def test_planner_totalizer_is_deterministic_total_and_cycle_safe() -> None:
    malformed = {
        "facets": [
            {"ordinal": 8, "type": " temporal ", "text": "  2024   result ", "weight": 2},
            {"ordinal": 2, "facet_type": "PREFERRED", "text": "2024 result", "weight": 99},
            {"ordinal": 3, "facet_type": "not-a-type", "text": "other clause"},
            None,
        ],
        "edges": [
            {"source": 8, "target": 3, "type": "REQUIRES"},
            {"source": 3, "target": 8, "type": "REFINES"},
            {"source": 8, "target": 8, "type": "REQUIRES"},
        ],
        "ignored": "totalizer may ignore malformed extras",
    }
    first = core.totalize_typed_plan(malformed, fallback_clauses=("fallback",))
    second = core.totalize_typed_plan(malformed, fallback_clauses=("fallback",))
    assert first == second
    assert tuple(facet.ordinal for facet in first.facets) == (0, 1)
    assert first.facets[0].text == "2024 result"
    assert first.facets[0].facet_type == "TEMPORAL"
    assert first.facets[0].weight == 2
    assert first.facets[1].facet_type == "REQUIRED"
    assert len(first.edges) == 1  # The reverse edge that closes a cycle is dropped.

    for arbitrary in (None, 7, object(), [], {}, {"facets": "wrong"}):
        result = core.totalize_typed_plan(arbitrary)
        assert isinstance(result, core.TypedFacetPlan)
        assert 2 <= len(result.facets) <= 12


def test_candidate_matrix_is_complete_strict_and_invalid_batch_totalizes_once() -> None:
    plan = _plan()
    matrix = _matrix()
    assert core.validate_candidate_matrix(matrix.payload(), plan) == matrix
    incomplete = matrix.payload()
    incomplete["candidates"][0]["facet_evidence"].pop()  # type: ignore[index]
    with pytest.raises(core.BircoP1CoreError, match="every candidate"):
        core.validate_candidate_matrix(incomplete, plan)
    with pytest.raises(core.BircoP1CoreError, match="outside"):
        core.CandidateFacetEvidence(
            candidate_ordinal=0,
            evidence_unit_count=1,
            facet_evidence=(core.FacetEvidence(0, 4, 0, 5),),
        )

    arbitrary = {
        "candidates": [
            {
                "candidate_ordinal": 1,
                "facet_evidence": [
                    {"facet_ordinal": 0, "support": 99, "contradiction": -3, "evidence_unit_ordinal": 0},
                    {"facet_ordinal": 0, "support": 1, "contradiction": 1, "evidence_unit_ordinal": 0},
                    {"facet_ordinal": 2, "support": "bad", "contradiction": 2, "evidence_unit_ordinal": 9},
                ],
            }
        ]
    }
    totalized = core.totalize_candidate_matrix(
        arbitrary, plan=plan, evidence_unit_counts=(2, 1)
    )
    assert totalized.candidate_count == 2
    assert totalized.candidates[0].facet_evidence == (
        core.FacetEvidence(0, 0, 0, None),
        core.FacetEvidence(1, 0, 0, None),
        core.FacetEvidence(2, 0, 0, None),
    )
    second = totalized.candidates[1]
    assert second.facet_evidence[0] == core.FacetEvidence(0, 4, 0, 0)
    assert second.facet_evidence[1] == core.FacetEvidence(1, 0, 0, None)
    assert second.facet_evidence[2] == core.FacetEvidence(2, 0, 2, None)


def test_all_recipes_return_complete_permutations_and_freeze_ordinal_ties() -> None:
    plan = _plan()
    matrix = _matrix()
    rankings = core.build_recipe_rankings(plan, matrix)
    assert tuple(rankings) == core.RECIPE_IDS
    for recipe, ranking in rankings.items():
        assert ranking.recipe_id == recipe
        assert set(ranking.candidate_ordinals) == set(range(matrix.candidate_count))
        assert len(ranking.candidate_ordinals) == matrix.candidate_count
        # Candidate two and three are exact semantic ties; ordinal two wins.
        assert ranking.candidate_ordinals.index(2) < ranking.candidate_ordinals.index(3)
    assert rankings[core.R1_WEIGHTED_MASS].candidate_ordinals[0] == 0
    assert rankings[core.R2_BOTTLENECK].candidate_ordinals[0] == 0
    assert rankings[core.R3_DEPENDENCY_FLOW].candidate_ordinals[0] == 0
    assert rankings[core.R4_CAPACITY_MATCH].candidate_ordinals[0] == 1
    with pytest.raises(core.BircoP1CoreError, match="full permutation"):
        core.validate_full_permutation((0, 0, 1, 2), 4)


def test_capacity_one_assignment_and_r3_requires_flow_are_substantive() -> None:
    plan = _plan()
    matrix = _matrix()
    zero = core.solve_capacity_assignment(plan, matrix.candidates[0])
    one = core.solve_capacity_assignment(plan, matrix.candidates[1])
    assert zero.assigned_facet_count == 1
    assert one.assigned_facet_count == 3
    assert one.bottleneck_support == 1
    assert zero.bottleneck_support == 0

    contradicted = _candidate(
        0, ((0, 4, 0), (4, 4, 1), (0, 0, 2))
    )
    empty = core.solve_capacity_assignment(plan, contradicted)
    assert empty.facet_to_evidence == ()
    assert empty.assigned_facet_count == 0

    broken_dependency = core.CandidateFacetEvidenceMatrix(
        (
            _candidate(0, ((0, 0, None), (4, 0, 0), (0, 0, None))),
            _candidate(1, ((1, 0, 0), (1, 0, 1), (0, 0, None))),
        )
    )
    # Candidate zero's dependent support cannot flow through an unsatisfied
    # prerequisite, so candidate one wins even though zero has raw support four.
    assert core.rank_r3_dependency_flow(plan, broken_dependency).candidate_ordinals[0] == 1


def test_e0_policy_order_is_exact() -> None:
    assert core.select_e0_recipe(_plan(requires=True, excluded=True)) == core.R3_DEPENDENCY_FLOW
    assert core.select_e0_recipe(_plan(requires=False, excluded=True)) == core.R2_BOTTLENECK
    assert core.select_e0_recipe(_plan(requires=False, excluded=False)) == core.R4_CAPACITY_MATCH


def test_twelve_action_features_are_finite_label_free_and_schema_closed() -> None:
    plan = _plan()
    matrix = _matrix()
    rankings = core.build_recipe_rankings(plan, matrix)
    assert core.FEATURE_ORDER == (
        "plan_facet_count",
        "required_facet_fraction",
        "exclusion_or_eligibility_fraction",
        "dependency_edge_fraction",
        "top10_mean_support",
        "top10_minimum_required_support",
        "top10_satisfied_facet_fraction",
        "top10_contradiction_negative",
        "top1_to_top2_margin",
        "score_entropy_negative",
        "top10_distinct_evidence_assignment_fraction",
        "single-facet-removal_rank_stability",
    )
    for ranking in rankings.values():
        values = core.compute_action_features(plan, matrix, ranking)
        assert len(values) == 12
        assert all(math.isfinite(value) for value in values)
        assert values[0] == 3
        assert 0 <= values[-1] <= 1
    clean = {name: 0 for name in core.FEATURE_ORDER}
    with pytest.raises(core.BircoP1CoreError, match="forbidden E4"):
        core.validate_action_features({**clean, "recipe_id": 1})
    with pytest.raises(core.BircoP1CoreError, match="schema drifted"):
        core.validate_action_features({**clean, "undeclared": 1})


def _training_slates(count: int = 30) -> tuple[core.E4TrainingSlate, ...]:
    slates = []
    for item in range(count):
        features = {}
        utilities = {}
        for index, recipe in enumerate(core.RECIPE_IDS):
            vector = [0.0] * len(core.FEATURE_ORDER)
            vector[0] = float(3 - index) + item / 100.0
            vector[1] = (item % 3) / 10.0  # utility-independent nuisance coordinate
            features[recipe] = vector
            utilities[recipe] = (3 - index) * 300_000_000
        slates.append(core.make_e4_training_slate(features, utilities))
    return tuple(slates)


def test_deterministic_listwise_softmax_fit_and_laplace_uncertainty() -> None:
    slates = _training_slates()
    first = core.fit_e4(slates)
    second = core.fit_e4(tuple(reversed(slates)))
    assert first.coefficients == pytest.approx(second.coefficients, abs=1e-11)
    assert first.coefficients[0] > 0
    assert all(abs(value) <= 1e-9 for value in first.coefficients[1:])
    covariance = np.asarray(first.laplace_covariance)
    assert covariance.shape == (12, 12)
    assert np.allclose(covariance, covariance.T)
    assert np.linalg.eigvalsh(covariance).min() > 0
    assert first.solver.startswith("numpy_deterministic_lbfgs")
    assert first.iterations <= 256
    assert first.converged is True
    with pytest.raises(core.BircoP1CoreError, match="exactly 30"):
        core.fit_e4(slates[:-1])

    features = {
        recipe: slates[0].actions[index].features
        for index, recipe in enumerate(core.RECIPE_IDS)
    }
    selection = core.select_e4_recipe(
        first, features, e0_recipe_id=core.R4_CAPACITY_MATCH
    )
    assert selection.selected_recipe_id == core.R1_WEIGHTED_MASS
    assert first.predict_standard_error(features[core.R1_WEIGHTED_MASS]) >= 0


def test_e4_ties_resolve_to_e0_then_recipe_name_without_recipe_id_feature() -> None:
    covariance = tuple(
        tuple(1.0 if row == column else 0.0 for column in range(12))
        for row in range(12)
    )
    zero = core.E4Model(
        population_mean=(0.0,) * 12,
        population_std=(1.0,) * 12,
        coefficients=(0.0,) * 12,
        laplace_covariance=covariance,
        solver="synthetic",
        iterations=0,
        converged=True,
        objective=0.0,
    )
    features = {recipe: (0.0,) * 12 for recipe in core.RECIPE_IDS}
    selection = core.select_e4_recipe(
        zero, features, e0_recipe_id=core.R3_DEPENDENCY_FLOW
    )
    assert selection.selected_recipe_id == core.R3_DEPENDENCY_FLOW
    assert "recipe_id" not in inspect.signature(core.E4Model.predict_mean).parameters


def test_linear_gain_metrics_integer_utility_and_public_report_are_aggregate_only() -> None:
    qrels = {0: 3.0, 1: 2.0, 2: 1.0, 3: 0.0, 4: 0.5, 5: 0.0}
    ideal = (0, 1, 2, 4, 3, 5)
    report = core.score_full_permutation(ideal, qrels)
    assert report.ndcg_at_10 == pytest.approx(1.0)
    assert report.recall_at_5 == pytest.approx(1.0)
    assert report.integer_utility == 1_000_000_000
    worse = core.score_full_permutation(tuple(reversed(ideal)), qrels)
    assert 0 < worse.ndcg_at_10 < 1
    with pytest.raises(core.BircoP1CoreError, match="full permutation"):
        core.score_full_permutation(ideal[:-1], qrels)

    payload = report.payload()
    assert set(payload).isdisjoint(
        {
            "query_id",
            "candidate_id",
            "candidate_ordinals",
            "document_id",
            "document_text",
            "qrels",
            "ranking",
        }
    )
    serialized = json.dumps(payload, sort_keys=True)
    assert "primary condition" not in serialized
    assert "candidate_ordinals" not in serialized


def test_descriptive_binomial_tail_ties_and_e4_promotion() -> None:
    assert core.descriptive_binomial_tail(5, 0) == Fraction(1, 32)
    assert core.descriptive_binomial_tail(0, 0) == 1
    challenger = (2,) * 5 + (1,) * 2
    incumbent = (1,) * 5 + (1,) * 2
    summary = core.paired_utility_summary(challenger, incumbent)
    assert (summary.gains, summary.harms, summary.ties) == (5, 0, 2)
    assert summary.descriptive_reference_tail == Fraction(1, 32)
    assert core.decide_a_hold_e4_promotion(
        challenger, incumbent, f_identifiability_passed=True
    ).promoted
    assert not core.decide_a_hold_e4_promotion(
        challenger, incumbent, f_identifiability_passed=False
    ).promoted


def test_f_identifiability_reality_primary_and_m_search_family_rules() -> None:
    families = tuple(index // 10 for index in range(30))
    e0 = ((0, 1, 2),) * 30
    e4 = list(e0)
    e4[0] = (1, 0, 2)
    e4[10] = (0, 2, 1)
    e4[11] = (2, 1, 0)
    identifiable = core.assess_f_identifiability(tuple(e4), e0, families)
    assert identifiable.passed
    assert identifiable.differing_ranking_count == 3
    assert identifiable.differing_family_count == 2
    identical = core.assess_f_identifiability(e0, e0, families)
    assert not identical.passed
    assert identical.differing_ranking_count == 0

    # Distinct recipe identities with one identical observable full ranking do
    # not satisfy behavior identifiability.
    recipe_rankings = core.build_recipe_rankings(_plan(), _matrix())
    same_order = tuple(
        recipe_rankings[core.R1_WEIGHTED_MASS].candidate_ordinals
    )
    assert same_order == recipe_rankings[core.R2_BOTTLENECK].candidate_ordinals
    same_behavior = core.assess_f_identifiability(
        (recipe_rankings[core.R1_WEIGHTED_MASS],) * 30,
        (recipe_rankings[core.R2_BOTTLENECK],) * 30,
        families,
    )
    assert same_behavior.differing_ranking_count == 0
    assert not same_behavior.passed

    # Ten strict gains in every family give a tiny descriptive reference tail
    # and positive family sums against both comparators.
    agent = (2,) * 30
    baseline = (1,) * 30
    reality = core.decide_a_hold_reality_primary(
        agent, baseline, baseline, families
    )
    assert reality.passed
    assert reality.raw_family_integer_deltas == (10, 10, 10)
    m_decision = core.decide_m_search_e4_improvement(agent, baseline, families)
    assert m_decision.passed

    # M_search permits a zero family but requires at least two positive ones.
    mixed_e4 = (1,) * 10 + (2,) * 20
    mixed_e0 = (1,) * 30
    assert core.decide_m_search(mixed_e4, mixed_e0, families).passed
    only_one = (1,) * 20 + (2,) * 10
    assert not core.decide_m_search(only_one, mixed_e0, families).passed


def test_core_has_no_source_network_api_or_secret_inputs() -> None:
    forbidden_names = {
        "query_id",
        "candidate_id",
        "document_id",
        "document_text",
        "api_key",
        "secret",
        "source_path",
        "url",
    }
    for function in (
        core.rank_candidates,
        core.compute_action_features,
        core.fit_e4,
        core.score_full_permutation,
        core.decide_a_hold_e4_promotion,
    ):
        assert forbidden_names.isdisjoint(inspect.signature(function).parameters)
