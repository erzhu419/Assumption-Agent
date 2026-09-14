"""Use saved boundary-failure vectors in the actual SC-KG fantasy scorer."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from test_cognition_sc_kg_environment import project_root, _saved_terminal_failure_optimizer


ROOT = Path(__file__).resolve().parents[1]
CASE = "research_scope_3_objective_only_change_noise_0.03"


@pytest.fixture
def boundary_runtime(project_root, monkeypatch):
    saved = json.loads((ROOT / "artifacts/sc_kg_goal_completion_20260911/runs" /
                        f"{CASE}_hegel_goal_completion.json").read_text())
    algorithm, adapter, environment = _saved_terminal_failure_optimizer(project_root, monkeypatch, CASE)
    monkeypatch.setattr(adapter, "simulate", lambda *args, **kwargs: pytest.fail("no new physical vector"))
    monkeypatch.setattr(environment, "observe", lambda *args, **kwargs: pytest.fail("no research draw"))
    algorithm._fit_initial_belief(algorithm._initial_samples())
    assert [list(point) for point, _ in algorithm.history] == saved["initial_points"]
    np.testing.assert_allclose([output for _, output in algorithm.history], saved["initial_outputs"])
    # Replay the paid prefix with the same normal GP/HVD updates. The supported
    # goal model is installed at its recorded time, after its four future rows.
    revision = saved["objective_research"]["scope_revisions"][0]
    for vector in saved["simulations"][:28]:
        point, output = tuple(vector["point"]), np.asarray(vector["output"])
        means = [model.posterior_mean(point) for model in algorithm.gpr]
        epistemic = [model.posterior_var(point) for model in algorithm.gpr]
        noises = [algorithm.variance_model.predict_variance(i, point, adapter) for i in range(2)]
        algorithm.history.append((point, output.copy()))
        algorithm.observations.setdefault(point, []).append(output.copy())
        for i, model in enumerate(algorithm.gpr):
            model.update(point, float(output[i]), noises[i])
            values = [float(y[i]) for y in algorithm.observations[point]]
            algorithm.variance_model.update(i, np.asarray(point), float(output[i]), means[i], model, adapter,
                epistemic_var=epistemic[i], replicate_count=len(values),
                replicate_variance=float(np.var(values, ddof=1)) if len(values) > 1 else None)
        if len(algorithm.history) == 53:
            algorithm.objective_research.state["accepted_revision"] = deepcopy(revision)
            algorithm._rebase_objective_scope(deepcopy(revision), stage=53)
    assert len(algorithm.history) == 60
    goal = algorithm.objective_research
    goal.state.update(scope_start=0, last_monitored=59, monitoring=[], scope_research=None)
    algorithm.recovery.state["phase"] = "monitoring"
    monkeypatch.setattr(algorithm.recovery, "next_control", lambda: None)
    monkeypatch.setattr(algorithm, "_consider_constraint_applicability", lambda: None)
    return algorithm, adapter, environment, saved


def test_real_boundary_repeat_and_new_goal_use_uncertain_noise_in_native_updates(boundary_runtime, monkeypatch):
    algorithm, adapter, environment, saved = boundary_runtime
    # Noise evidence is bought before the final goal-pool search consumes its
    # allocation. No fit, extra observation or total budget is added.
    goal = algorithm.objective_research
    goal.state["compute_cost"] = goal.state["compute_budget"] - 6000
    candidates, _ = algorithm._generate_candidates(28)
    evidence = algorithm._terminal_noise_evidence
    assert evidence["degrees_of_freedom"] == 3
    assert all(index < 60 for group in evidence["groups"] for index in group["indices"])
    assert 2 <= len(candidates) <= 4
    assert any(point in algorithm.observations for point in candidates)
    assert any(point not in algorithm.observations for point in candidates)
    assert goal.state["pending"] is None
    before = [(model.a.copy(), model.C.copy()) for model in algorithm.gpr]
    noise_before = {key: deepcopy(value) for key, value in algorithm.variance_model.__dict__.items()
                    if isinstance(value, dict) and 0 in value and 1 in value}
    original_update = algorithm._exact_primary_fantasy_update
    updates = []

    def update(point, y, observations, means, noise, epistemic):
        if algorithm._value_precision_round is not None:
            updates.append({"point": tuple(point), "y": list(y), "noise": list(noise),
                            "observations": len(observations)})
            # The sampled scale changes y; it is not supplied to the learner.
            assert noise == pytest.approx([
                algorithm.variance_model.predict_variance(i, point, adapter) for i in range(2)])
        return original_update(point, y, observations, means, noise, epistemic)

    monkeypatch.setattr(algorithm, "_exact_primary_fantasy_update", update)
    pool = list(dict.fromkeys([point for point, _ in algorithm.history] + candidates))
    scores = algorithm._exact_posterior_update_scores(candidates, pool)
    event = algorithm.objective_diagnostics["value_precision"]["rounds"][-1]
    assert event["noise_evidence"]["sensitive_repeats"]
    assert event["noise_quadrature_points"] == 4
    assert event["allowance"] == 128
    assert 8 * 4 * len(candidates) <= event["evaluations"] <= 128
    assert event["evaluations"] == len(updates) == event["score_one_calls"]
    assert event["evaluations"] == 4 * event["innovation_evaluations"]
    assert algorithm.objective_diagnostics["value_precision"]["evaluations"] == event["evaluations"]
    assert sum(scores) == 1. and set(scores) <= {0., 1.}
    for point in candidates:
        actual = [row for row in updates if row["point"] == point]
        assert len({row["y"][1] for row in actual}) > 4
        assert all(row["observations"] == len(algorithm.observations.get(point, [])) for row in actual)
    for model, (a, covariance) in zip(algorithm.gpr, before):
        np.testing.assert_array_equal(model.a, a)
        np.testing.assert_array_equal(model.C, covariance)
    assert all(repr(algorithm.variance_model.__dict__[key]) == repr(value)
               for key, value in noise_before.items())
    assert len(algorithm.history) == 60 and adapter.simulator_calls == 0
    assert environment.costs["simulator_calls"] == 32
    assert goal.state["compute_cost"] <= goal.state["compute_budget"]
    assert goal.state["pending"] is None


def test_no_valuation_budget_preserves_native_scores_and_does_not_buy_noise_research(boundary_runtime, monkeypatch):
    algorithm, _, _, _ = boundary_runtime
    costs = algorithm.objective_diagnostics["value_precision"]
    costs["evaluations"] = costs["evaluation_budget"]
    monkeypatch.setattr(algorithm, "_boundary_noise_evidence",
                        lambda remaining: pytest.fail("no affordable complete comparison"))
    candidates, _ = algorithm._generate_candidates(28)
    assert algorithm._terminal_noise_evidence is None
    parent = type(algorithm).__mro__[1]
    expected = np.asarray([.03 + index * .01 for index in range(len(candidates))])

    def original(self, points, pool, secondary_terminal_mode=None):
        self._last_exact_kg_raw_scores = expected.copy()
        self._last_exact_kg_expected_values = 3. - expected
        self._last_exact_kg_current_value = 3.
        return expected.copy()

    monkeypatch.setattr(parent, "_exact_posterior_update_scores", original)
    actual = algorithm._exact_posterior_update_scores(candidates, candidates)
    np.testing.assert_array_equal(actual, expected)
    assert costs["evaluations"] == costs["evaluation_budget"] and not costs["rounds"]


def test_same_future_label_updates_one_noise_belief_used_by_fantasy_and_actual_recommendation(boundary_runtime):
    from statistics import NormalDist

    algorithm, adapter, _, _ = boundary_runtime
    evidence = algorithm._boundary_noise_evidence(4)
    algorithm._install_boundary_noise(evidence)
    pool = [(22, 59, 34), (14, 50, 59)]
    parent = type(algorithm).__mro__[1]
    original = parent._terminal_certificate_components(algorithm, algorithm.gpr, algorithm.variance_model,
                                                       pool, observations=algorithm.observations)
    mixed = algorithm._terminal_certificate_components(algorithm.gpr, algorithm.variance_model,
                                                       pool, observations=algorithm.observations)
    actual = algorithm._certification_result(algorithm.gpr[1].posterior_mean_many(pool), pool)
    np.testing.assert_allclose(mixed["margin"], actual.margin, rtol=1e-12, atol=1e-12)
    assert mixed["margin"][0] > original["margin"][0]
    assert original["margin"][0] <= 0. < mixed["margin"][0]
    assert mixed["margin"][1] < 0.
    value = algorithm._terminal_value_from_models(algorithm.gpr, algorithm.variance_model,
                                                pool, observations=algorithm.observations)
    # Every scale describes the same two prospective recommendations. The
    # learner must choose one point after mixing their tails, not one point
    # per hidden noise value followed by an average of those oracle minima.
    z = NormalDist().inv_cdf(1. - adapter.alpha)
    base = np.asarray(mixed["noise_reference_margin"]) - z * np.sqrt(mixed["noise_reference"])
    margins = [base + z * np.sqrt(mixed["noise_reference"] * scale) for scale in evidence["scales"]]
    oracle_values = [float(np.min(np.where(margin <= 0., mixed["objective"], np.inf)))
                     for margin in margins]
    assert len(set(oracle_values)) == 2
    assert value == pytest.approx([0., 0., mixed["objective"][1]])
    assert value[2] > np.dot(evidence["weights"], oracle_values)
    before = algorithm._boundary_noise_posterior()
    point = pool[0]
    old = algorithm.observations[point][-1]
    future = algorithm._future_observations_with_fantasy(
        algorithm.observations, point, np.asarray([old[0], old[1] + .2]))
    posterior = algorithm._boundary_noise_posterior(future)
    future_certificate = algorithm._terminal_certificate_components(
        algorithm.gpr, algorithm.variance_model, pool, observations=future)
    assert posterior["new_variance_comparisons"] == 1
    assert posterior["tail"]["variance_factor"] > before["tail"]["variance_factor"]
    assert future_certificate["margin"][0] > mixed["margin"][0]
    assert algorithm._boundary_noise_posterior() == before
    assert len(algorithm.history) == 60 and adapter.simulator_calls == 0
    # The existing constraint-repair path can reject a newly acquired outlier
    # after a first forecast. That same paid label must then stop voting on
    # noise, even though its history row remains and the stage is unchanged.
    outlier = np.asarray([old[0], old[1] + .2])
    algorithm.history.append((point, outlier))
    algorithm.observations[point].append(outlier)
    assert algorithm._boundary_noise_posterior()["new_variance_comparisons"] == 1
    algorithm.rejected_constraint_indices.add(60)
    rejected = algorithm._boundary_noise_posterior()
    assert rejected["new_variance_comparisons"] == 0
    assert rejected["evidence"]["weights"] == before["evidence"]["weights"]


def test_saved_real_repeat_has_positive_value_when_it_restores_the_common_certificate(boundary_runtime):
    algorithm, adapter, _, _ = boundary_runtime
    algorithm._install_boundary_noise(algorithm._boundary_noise_evidence(4))
    vector = json.loads((ROOT / "artifacts/sc_kg_goal_recompetition_20260911/revision2/runs" /
        f"{CASE}_hegel_goal_recompetition.json").read_text())["simulations"][28]
    point = tuple(vector["point"])
    paid = list(algorithm.observations)
    certificate = algorithm._terminal_certificate_components(algorithm.gpr, algorithm.variance_model,
                                                              paid, observations=algorithm.observations)
    # An actual paid control catalog with no current certificates reaches the
    # old normalized-fallback branch. No predictions or gains are fabricated.
    pool = [p for p, margin in zip(paid, certificate["margin"]) if margin > 0.]
    assert point in pool
    current = algorithm._terminal_value_from_models(algorithm.gpr, algorithm.variance_model,
                                                   pool, observations=algorithm.observations)
    assert current[0] == 1. and current[1] > 0.
    legacy = algorithm._terminal_value_from_models(algorithm.gpr, algorithm.variance_model,
        pool, observations=algorithm.observations, terminal_mode_override="hard_certified")
    means = [model.posterior_mean(point) for model in algorithm.gpr]
    epistemic = [model.posterior_var(point) for model in algorithm.gpr]
    noises = [algorithm.variance_model.predict_variance(i, point, adapter) for i in range(2)]
    innovation = [(vector["output"][i] - means[i]) / np.sqrt(noises[i] + epistemic[i]) for i in range(2)]
    prediction = algorithm._exact_posterior_update_score_one(point, np.asarray([innovation]), pool,
                                                           current, return_diagnostics=True)
    future = prediction["expected_terminal_value"]
    assert future[0] == 0. and future[1] == 0.
    assert future[2] > 1.  # Objective units cannot outweigh restored feasibility.
    assert legacy < future[2]  # The old mixed-units subtraction has the wrong sign.
    assert prediction["component_gain"][0] == 1.
    assert prediction["raw_score"] == 1. and prediction["score"] > 0.
    assert len(algorithm.history) == 60 and adapter.simulator_calls == 0
