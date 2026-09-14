"""Rejected v053: a reopened hypothesis offers a control without pre-buying it.

Native GP action values decide between retained ordinary controls and the
research offer. Controlled vector callbacks exercise sharing without simulation.
Only the frozen experiment is loaded; these are not default-behavior claims.
"""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from test_cognition_sc_kg_environment import project_root
from test_cognition_objective_incumbent import saved_incumbent_failure
from test_cognition_objective_replacement_runtime import _runtime
from test_cognition_objective_reentry import _rejected_goal
from reentry_value_experiment import ObjectiveResearch, use_frozen_reentry_value_optimizer
from hegel_machine.cognition.decision_experiments import _predict


def test_reentry_offer_does_not_register_until_the_control_is_selected():
    goal, rows = _rejected_goal()
    goal = ObjectiveResearch(goal.spec, state=goal.to_dict())
    before = deepcopy(goal.state["scope_attempts"])

    def forecast(point):
        return {"features": deepcopy(point), "objective_mean": point["x1"] + 1.,
                "objective_noise": .001, "feasible": True}

    offer = goal.next_control(rows, forecast, scope_start=0, stage=8, remaining_calls=3)
    assert offer is not None and offer["reentry"] is True
    assert goal.state["pending"] is None and not goal.state["requests"]
    assert not goal.state["outcomes"] and goal.state["scope_attempts"] == before
    assert goal.state["scope_research"]["validation"] == before[0]["validation"]
    request = goal.select_terminal_offer(offer, stage=8)
    assert request["features"] == offer["features"]
    assert goal.state["pending"] == request and goal.state["requests"] == [request]
    assert goal.select_terminal_offer(offer, stage=8) == request
    assert goal.state["requests"] == [request] and not goal.state["outcomes"]


def _reentry_runtime(project_root, monkeypatch, saved, *, phase="monitoring"):
    use_frozen_reentry_value_optimizer(monkeypatch)
    algorithm, adapter, goal, rows = _runtime(project_root, monkeypatch, saved)
    goal = ObjectiveResearch(goal.spec, state=goal.to_dict())
    algorithm.objective_research = goal
    goal.state["replacement_research"] = algorithm.objective_replacement.state
    algorithm.config = replace(algorithm.config, N=60, exact_kg_mc_samples=2)
    archive = deepcopy(goal.state["scope_research"])
    archive.update(phase="completed", status="future_predictions_do_not_improve")
    for index, row in enumerate(archive["validation"]):
        row.update(candidate_error=1. if index == 0 else 0.,
                   previous_error=0. if index == 0 else .2)
    goal.state.update(scope_research=None, scope_attempts=[archive], constraint_concept=None)
    goal._resume_rejected(rows, stage=58, remaining_calls=2)
    assert goal.state["scope_research"]["reentry"]
    algorithm.recovery.state["phase"] = phase
    monkeypatch.setattr(algorithm, "_consider_constraint_applicability", lambda: None)
    monkeypatch.setattr(algorithm.recovery, "next_control", lambda: None)
    point = (13, 27, 71)
    features = dict(zip(goal.spec.feature_names, (value / 100. for value in point)))
    candidate_mean = _predict(goal.spec, archive["candidate"], features)
    offer = {"features": features, "purpose": "objective_confirmation",
        "reason": "validate_objective_scope", "stage": 58, "scope_start": 0,
        "candidate_mean": candidate_mean, "previous_mean": candidate_mean + 1.,
        "objective_noise": .001, "reentry": True, "disagreement": 100.}
    monkeypatch.setattr(goal, "next_control", lambda *args, **kwargs: deepcopy(offer))
    return algorithm, adapter, goal, offer, point


@pytest.mark.parametrize("phase", ["monitoring", "unresolved"])
@pytest.mark.parametrize("already_native", [False, True])
def test_reentry_keeps_every_native_action_and_deduplicates_an_overlapping_offer(
        project_root, monkeypatch, saved_incumbent_failure, phase, already_native):
    algorithm, adapter, goal, _, point = _reentry_runtime(
        project_root, monkeypatch, saved_incumbent_failure, phase=phase)
    native = [(25, 50, 75), *((point,) if already_native else ())]
    monkeypatch.setattr(type(algorithm).__mro__[1], "_generate_candidates",
        lambda self, iteration: (list(native), {item: "native_test" for item in native}))
    before = deepcopy(goal.state["scope_research"]["validation"])
    candidates, sources = algorithm._generate_candidates(59)
    assert candidates == [*native, *((point,) if not already_native else ())]
    assert candidates.count(point) == 1
    assert sources[point] == ("native_test" if already_native else "objective_reentry_choice")
    assert set(native).issubset(candidates)
    assert goal.state["pending"] is None and not goal.state["requests"]
    assert goal.state["scope_research"]["validation"] == before
    assert algorithm.config.N == 60 and len(algorithm.history) == 58
    assert adapter.simulator_calls == 0


@pytest.mark.parametrize("selected_role", ["native", "research"])
def test_only_the_selected_control_registers_and_its_vector_is_shared_exactly_once(
        project_root, monkeypatch, saved_incumbent_failure, selected_role):
    algorithm, adapter, goal, _, research_point = _reentry_runtime(
        project_root, monkeypatch, saved_incumbent_failure)
    native = (25, 50, 75)
    monkeypatch.setattr(type(algorithm).__mro__[1], "_generate_candidates",
        lambda self, iteration: ([native], {native: "native_test"}))
    candidates, _ = algorithm._generate_candidates(59)
    # Explicit opposite value orderings test both outcomes of the choice seam.
    # The separate test below exercises the actual native GP value calculation.
    selected = native if selected_role == "native" else research_point
    values = np.asarray([1. if point == selected else 0. for point in candidates])
    chosen = candidates[int(np.argmax(values))]
    features = dict(zip(goal.spec.feature_names, (value / 100. for value in chosen)))
    target = _predict(goal.spec, goal.state["scope_research"]["candidate"], features)
    prior_validation = deepcopy(goal.state["scope_research"]["validation"])
    before_goal_requests = len(goal.state["requests"])
    algorithm._last_terminal_pool = [point for point, _ in algorithm.history]
    purchases, constraint_observations = [], []

    def replay(selected_point, rng):
        purchases.append(tuple(selected_point))
        assert len(algorithm.history) == 58
        if selected_role == "native":
            assert goal.state["pending"] is None
            assert goal.state["shared_pending"]["expected_global_index"] == 58
            assert len(goal.state["requests"]) == before_goal_requests
        else:
            assert goal.state["pending"]["features"] == features
            assert goal.state["shared_pending"] is None
            assert len(goal.state["requests"]) == before_goal_requests + 1
        assert goal.state["scope_research"]["validation"] == prior_validation
        return np.asarray([target, -1.])

    monkeypatch.setattr(adapter, "simulate", replay)
    monkeypatch.setattr(algorithm.recovery, "observe",
        lambda *args, **kwargs: constraint_observations.append(kwargs["context"]["global_index"]))
    algorithm._simulate_and_store(chosen)
    assert purchases == [chosen] and constraint_observations == [58]
    assert len(goal.state["scope_research"]["validation"]) == len(prior_validation) + 1
    assert goal.state["scope_research"]["validation"][-1]["global_index"] == 58
    assert len(goal.state["shared_outcomes"]) + len(goal.state["outcomes"]) == 1
    assert goal.state["pending"] is None and goal.state["shared_pending"] is None
    assert not algorithm._terminal_goal_offers
    assert algorithm.config.N == 60 and len(algorithm.history) == 59
    assert adapter.simulator_calls == 0


def test_native_gp_values_do_not_add_disagreement_and_leave_unselected_evidence_unconsumed(
        project_root, monkeypatch, saved_incumbent_failure):
    algorithm, adapter, goal, _, research_point = _reentry_runtime(
        project_root, monkeypatch, saved_incumbent_failure)
    native = (25, 50, 75)
    monkeypatch.setattr(type(algorithm).__mro__[1], "_generate_candidates",
        lambda self, iteration: ([native], {native: "native_test"}))
    candidates, _ = algorithm._generate_candidates(59)
    pool = list(dict.fromkeys([*(point for point, _ in algorithm.history), *candidates]))
    original_goal = goal.to_dict()
    parameters = [(model.a.copy(), model.C.copy()) for model in algorithm.gpr]
    random_state = deepcopy(algorithm.rng.bit_generator.state)
    first = algorithm._exact_posterior_update_scores(candidates, pool)
    algorithm.rng.bit_generator.state = random_state
    algorithm._terminal_goal_offers[research_point]["disagreement"] = 1e12
    second = algorithm._exact_posterior_update_scores(candidates, pool)
    np.testing.assert_allclose(second, first, rtol=1e-12, atol=1e-12)
    assert len(first) == 2 and np.all(np.isfinite(first))
    assert goal.to_dict() == original_goal
    for model, (a, covariance) in zip(algorithm.gpr, parameters):
        np.testing.assert_array_equal(model.a, a)
        np.testing.assert_array_equal(model.C, covariance)
    assert len(algorithm.history) == 58 and adapter.simulator_calls == 0


def test_all_clipped_reentry_values_use_real_bounded_refinement_without_an_installed_goal_head(
        project_root, monkeypatch, saved_incumbent_failure):
    algorithm, adapter, goal, _, research_point = _reentry_runtime(
        project_root, monkeypatch, saved_incumbent_failure)
    algorithm.config = replace(algorithm.config, exact_kg_mc_samples=1)
    native = (25, 50, 75)
    base = type(algorithm).__mro__[1]
    monkeypatch.setattr(base, "_generate_candidates",
        lambda self, iteration: ([native], {native: "native_test"}))
    candidates, _ = algorithm._generate_candidates(59)
    pool = list(dict.fromkeys([*(point for point, _ in algorithm.history), *candidates]))
    assert not algorithm.objective_scope_revisions
    original_scores = base._exact_posterior_update_scores
    native_score_one = algorithm._exact_posterior_update_score_one
    refinements = []

    def all_clipped(self, points, terminal_pool, **kwargs):
        # Use the real current terminal value and MC1 setup, then control only
        # the initial all-clipped trigger. Refinement fantasies remain native.
        original_scores(self, points, terminal_pool, **kwargs)
        self._last_exact_kg_raw_scores = np.asarray([-1., -2.])
        self._last_exact_kg_expected_values = self._last_exact_kg_current_value + np.asarray([1., 2.])
        return np.zeros(2)

    def score_one(point, innovations, terminal_pool, current, *args, **kwargs):
        if algorithm._value_precision_round is not None:
            assert terminal_pool is pool and len(innovations) == 1
            refinements.append(tuple(point))
        return native_score_one(point, innovations, terminal_pool, current, *args, **kwargs)

    monkeypatch.setattr(base, "_exact_posterior_update_scores", all_clipped)
    monkeypatch.setattr(algorithm, "_exact_posterior_update_score_one", score_one)
    before_goal = goal.to_dict()
    parameters = [(model.a.copy(), model.C.copy()) for model in algorithm.gpr]
    costs = algorithm.objective_diagnostics["value_precision"]
    before_evaluations = costs["evaluations"]
    result = algorithm._exact_posterior_update_scores(candidates, pool)
    event = costs["rounds"][-1]
    assert set(refinements) == {native, research_point}
    assert 0 < len(refinements) == event["evaluations"] <= event["allowance"]
    assert event["allowance"] == (costs["evaluation_budget"] - before_evaluations) // 2
    assert costs["evaluations"] - before_evaluations == len(refinements)
    assert event["score_one_calls"] == event["terminal_value_calls"] == len(refinements)
    assert sum(result) == 1. and event["selection_semantics"] == "policy_indicator"
    assert goal.to_dict() == before_goal
    for model, (a, covariance) in zip(algorithm.gpr, parameters):
        np.testing.assert_array_equal(model.a, a)
        np.testing.assert_array_equal(model.C, covariance)
    assert len(algorithm.history) == 58 and adapter.simulator_calls == 0


def test_without_a_reentry_offer_the_original_scores_return_without_extra_refinement(
        project_root, monkeypatch, saved_incumbent_failure):
    algorithm, adapter, _, _, _ = _reentry_runtime(project_root, monkeypatch, saved_incumbent_failure)
    algorithm.config = replace(algorithm.config, exact_kg_mc_samples=1)
    assert not algorithm.objective_scope_revisions and not algorithm._terminal_goal_offers
    calls = []
    expected = np.zeros(2)

    def original_scores(self, candidates, pool, **kwargs):
        calls.append((list(candidates), list(pool)))
        self._last_exact_kg_raw_scores = np.asarray([-1., -2.])
        self._last_exact_kg_current_value = 1.
        return expected

    monkeypatch.setattr(type(algorithm).__mro__[1], "_exact_posterior_update_scores", original_scores)
    monkeypatch.setattr(algorithm, "_exact_posterior_update_score_one",
        lambda *args, **kwargs: pytest.fail("ordinary scores must not open reentry refinement"))
    candidates = [(25, 50, 75), (13, 27, 71)]
    result = algorithm._exact_posterior_update_scores(candidates, candidates)
    assert result is expected and calls == [(candidates, candidates)]
    assert not algorithm.objective_diagnostics["reentry_value_comparisons"]
    assert algorithm.objective_diagnostics["value_precision"]["evaluations"] == 0
    assert not algorithm.objective_diagnostics["value_precision"]["rounds"]
    assert len(algorithm.history) == 58 and adapter.simulator_calls == 0
