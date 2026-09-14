"""An executable constraint remains a contender beside its adaptive GP readout.

The default retains the model competition and shares existing observations.
The explicitly experimental tests load the frozen extra-nomination variant.
These use the actual optional SC-KG GP and clone implementations with controlled
labels; no research, optimizer, or evaluation simulator draws are purchased.
"""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from test_cognition_sc_kg_environment import project_root
from test_cognition_objective_incumbent import saved_incumbent_failure
from test_cognition_objective_replacement_runtime import _runtime
from constraint_nomination_experiment import use_frozen_nomination_optimizer
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.sc_kg_environment import SCKGEnvironment, compile_sc_kg_program


@pytest.fixture
def competition(project_root):
    environment = SCKGEnvironment(project_root, dimension=3, seed=841000071)
    adapter = compile_sc_kg_program(environment, Program("feature", name="x1"))
    from core.gpr import ParametricGPR
    from algorithms.single_olhkg import SingleOLHKGAlgorithm
    from hegel_machine.cognition.constraint_retention import install_program_competition
    original = ParametricGPR(3, lambda_i=.01, prior_var=.04,
        normalize_func=adapter.normalize, basis_map=adapter.gpr_basis_map(1))
    original.set_parametric_prior(np.array([0., 1.4]), .01, .04)
    cloner = object.__new__(SingleOLHKGAlgorithm)._clone_gpr_for_exact_kg
    mixture = install_program_competition(original, cloner)
    return original, mixture, cloner, environment


def test_installation_retains_both_explanations_without_changing_the_existing_model(competition):
    original, mixture, _, environment = competition
    affine, program = mixture._finite_mixture_components
    assert affine is not original and program is not original
    assert affine.basis_map is original.basis_map and program.basis_map is original.basis_map
    np.testing.assert_array_equal(original.a, [0., 1.4])
    np.testing.assert_array_equal(affine.a, original.a)
    np.testing.assert_array_equal(program.a, [0., 1.])
    np.testing.assert_array_equal(mixture._finite_mixture_weights, [.5, .5])
    assert mixture.sampled_set == affine.sampled_set == program.sampled_set == []
    assert mixture._finite_mixture_update_count == 0
    assert environment.costs["simulator_calls"] == 0


def test_existing_fitted_samples_initialize_both_components_without_voting_twice(competition):
    original, _, clone, _ = competition
    from hegel_machine.cognition.constraint_retention import install_program_competition
    points = [(20, 40, 60), (80, 60, 40)]
    for point, label in zip(points, (.7, -.2)):
        original.update(point, label, .01)
    before = original.a.copy(), original.C.copy()
    assert np.any(original.a[2:] != 0.)
    mixture = install_program_competition(original, clone)
    affine, program = mixture._finite_mixture_components
    np.testing.assert_array_equal(affine.a, before[0])
    np.testing.assert_array_equal(affine.C, before[1])
    np.testing.assert_array_equal(program.C, before[1])
    np.testing.assert_array_equal(program.a, [0., 1., 0., 0.])
    assert mixture.sampled_set == affine.sampled_set == program.sampled_set == points
    np.testing.assert_allclose(program.posterior_mean_many(points), [.2, .8])
    np.testing.assert_array_equal(mixture._finite_mixture_weights, [.5, .5])
    assert mixture._finite_mixture_update_count == 0
    np.testing.assert_array_equal(original.a, before[0])
    np.testing.assert_array_equal(original.C, before[1])


def test_frozen_program_keeps_its_mean_while_both_predictors_learn_uncertainty(competition):
    original, mixture, _, _ = competition
    affine, program = mixture._finite_mixture_components
    points = [(0, 0, 0), (30, 70, 20), (60, 50, 80), (100, 100, 100)]
    expected = np.array([point[0] / 100. for point in points])
    before_variance = program.posterior_var(points[-1])
    old_affine = affine.posterior_mean_many(points)
    old_program = deepcopy(program.basis_map.program.to_dict())
    mixture.update(points[-1], 2., .01)
    mixture.update(points[1], -.5, .01)
    np.testing.assert_array_equal(program.a[:2], [0., 1.])
    np.testing.assert_array_equal(program.a[2:], np.zeros(len(program.sampled_set)))
    np.testing.assert_allclose(program.posterior_mean_many(points), expected, atol=1e-12)
    assert program.posterior_var(points[-1]) < before_variance
    assert not np.allclose(affine.posterior_mean_many(points), old_affine)
    assert program.basis_map.program.to_dict() == old_program
    assert mixture.sampled_set == affine.sampled_set == program.sampled_set == [points[-1], points[1]]
    np.testing.assert_array_equal(original.a, [0., 1.4])
    assert original.sampled_set == []
    assert mixture._finite_mixture_update_count == 2


@pytest.mark.parametrize("target,program_wins", [(1., True), (1.4, False)])
def test_one_shared_label_uses_preupdate_likelihood_and_can_support_either_explanation(
        competition, target, program_wins):
    _, mixture, _, _ = competition
    point, noise = (100, 50, 50), .01
    components = mixture._finite_mixture_components
    means = np.array([model.posterior_mean(point) for model in components])
    variances = np.array([model.posterior_var(point) + noise for model in components])
    log_mass = np.log(mixture._finite_mixture_weights) - .5 * (
        np.log(2. * np.pi * variances) + (target - means) ** 2 / variances)
    expected = np.exp(log_mass - np.max(log_mass))
    expected /= sum(expected)
    mixture.update(point, target, noise)
    np.testing.assert_allclose(mixture._finite_mixture_weights, expected, rtol=1e-12, atol=1e-12)
    assert bool(mixture._finite_mixture_weights[1] > .5) is program_wins
    assert mixture._finite_mixture_update_count == 1
    assert all(model.sampled_set == [point] for model in components)


def test_later_counterexamples_can_withdraw_support_for_the_retained_program(competition):
    _, mixture, _, _ = competition
    point = (100, 50, 50)
    mixture.update(point, 1., .01)
    supported = float(mixture._finite_mixture_weights[1])
    assert supported > .5
    for _ in range(4):
        mixture.update(point, 1.4, .01)
    assert mixture._finite_mixture_weights[1] < supported
    assert mixture._finite_mixture_weights[0] > .5
    assert mixture._finite_mixture_components[1].posterior_mean(point) == pytest.approx(1.)
    assert mixture._finite_mixture_update_count == 5


def test_native_fantasy_clone_keeps_component_updates_out_of_the_live_model(competition):
    _, mixture, clone, environment = competition
    point = (80, 30, 40)
    before_mean, before_covariance = mixture.a.copy(), mixture.C.copy()
    before_weights = mixture._finite_mixture_weights.copy()
    components = [(model.a.copy(), model.C.copy(), list(model.sampled_set))
                  for model in mixture._finite_mixture_components]
    fantasy = clone(mixture)
    fantasy.update(point, 1.8, .01)
    np.testing.assert_array_equal(mixture.a, before_mean)
    np.testing.assert_array_equal(mixture.C, before_covariance)
    np.testing.assert_array_equal(mixture._finite_mixture_weights, before_weights)
    for live, imagined, (a, covariance, sampled) in zip(
            mixture._finite_mixture_components, fantasy._finite_mixture_components, components):
        assert live is not imagined
        np.testing.assert_array_equal(live.a, a)
        np.testing.assert_array_equal(live.C, covariance)
        assert live.sampled_set == sampled == []
        assert imagined.sampled_set == [point]
    assert fantasy._finite_mixture_components[1].posterior_mean(point) == pytest.approx(.8)
    assert mixture._finite_mixture_update_count == 0 and fantasy._finite_mixture_update_count == 1
    assert environment.costs["simulator_calls"] == 0


def test_native_moment_match_keeps_the_variance_between_the_two_explanations(competition):
    _, mixture, _, _ = competition
    point = (100, 50, 50)
    for update in (False, True):
        if update:
            mixture.update(point, 1.2, .01)
        weights = mixture._finite_mixture_weights
        components = mixture._finite_mixture_components
        means = np.array([model.posterior_mean(point) for model in components])
        variances = np.array([model.posterior_var(point) for model in components])
        average = float(weights @ means)
        within = float(weights @ variances)
        between = float(weights @ ((means - average) ** 2))
        assert between > 0.
        assert mixture.posterior_mean(point) == pytest.approx(average)
        assert mixture.posterior_var(point) == pytest.approx(within + between, rel=1e-10, abs=1e-12)
        assert mixture.posterior_var(point) > within


def _retention_runtime(project_root, monkeypatch, saved):
    algorithm, adapter, goal, rows = _runtime(project_root, monkeypatch, saved)
    algorithm._rebase_scope({"scope_start": 44, "program": adapter.basis.program}, stage=58)
    algorithm.config = replace(algorithm.config, N=60)
    algorithm.objective_scope_revisions = [{"scope_start": 42}]
    algorithm.recovery.state["phase"] = "monitoring"
    algorithm.recovery.state["costs"]["compute_cost"] = 0
    monkeypatch.setattr(algorithm, "_consider_constraint_applicability", lambda: None)
    monkeypatch.setattr(algorithm.recovery, "next_control", lambda: None)
    return algorithm, adapter, goal, rows


def _experimental_runtime(project_root, monkeypatch, saved):
    use_frozen_nomination_optimizer(monkeypatch)
    return _retention_runtime(project_root, monkeypatch, saved)


def test_scope_rebase_installs_a_frozen_program_beside_its_actual_fitted_readout(
        project_root, monkeypatch, saved_incumbent_failure):
    algorithm, adapter, _, _ = _retention_runtime(project_root, monkeypatch, saved_incumbent_failure)
    mixture = algorithm.constraint_models["learned"]
    affine, frozen, coefficients = mixture._finite_mixture_components
    points = [(13, 27, 71), (41, 62, 83)]
    actual_program = [adapter.basis.features(point)[0] for point in points]
    np.testing.assert_allclose(frozen.posterior_mean_many(points), actual_program, atol=1e-12)
    assert mixture.sampled_set == affine.sampled_set == frozen.sampled_set == coefficients.sampled_set
    assert len(mixture.sampled_set) > 0 and mixture._finite_mixture_update_count == 0
    assert algorithm.constraint_retention["scopes"][-1]["program"] == adapter.basis.program.to_dict()
    np.testing.assert_allclose(algorithm.constraint_retention["scopes"][-1]["initial_weights"], [1/3] * 3)
    assert algorithm.active_constraint_basis == "learned"
    assert adapter.simulator_calls == 0 and len(algorithm.history) == 58


def _strong_disagreement(algorithm):
    mixture = algorithm.constraint_models["learned"]
    affine, frozen = mixture._finite_mixture_components
    affine.a = frozen.a.copy()
    affine.a[0] += 2.
    return mixture


def _goal_offer(point):
    return {"features": {name: value / 100. for name, value in zip(("x1", "x2", "x3"), point)},
            "purpose": "objective_confirmation", "reason": "test_terminal_goal",
            "stage": 58, "scope_start": 42}


def test_experimental_discrimination_and_goal_confirmation_keep_the_original_native_candidate_pool(
        project_root, monkeypatch, saved_incumbent_failure):
    algorithm, adapter, goal, _ = _experimental_runtime(project_root, monkeypatch, saved_incumbent_failure)
    _strong_disagreement(algorithm)
    offered = algorithm._constraint_retention_offer()
    assert offered is not None
    discriminating = tuple(int(round(offered["features"][name] * 100))
                           for name in goal.spec.feature_names)
    public = {tuple(int(round(features[name] * 100)) for name in goal.spec.feature_names)
              for probe in adapter.spec.available_probes for features in probe.sequence}
    assert discriminating in public
    repeated = next(point for point, _ in algorithm.history if point != discriminating)
    native = [(97, 98, 99)]
    monkeypatch.setattr(goal, "terminal_offers", lambda *args, **kwargs: [_goal_offer(repeated)])
    monkeypatch.setattr(type(algorithm).__mro__[1], "_generate_candidates",
        lambda self, iteration: (list(native), {native[0]: "native_test_control"}))
    before_requests = deepcopy(goal.state["requests"])
    candidates, sources = algorithm._generate_candidates(59)
    assert {native[0], repeated, discriminating}.issubset(candidates)
    assert sources[native[0]] == "native_test_control"
    assert algorithm._terminal_goal_offers[repeated]["purpose"] == "objective_confirmation"
    assert algorithm._terminal_goal_offers[discriminating]["purpose"] == "constraint_discrimination"
    assert algorithm.constraint_retention["nomination_compute"] > 0
    assert algorithm.recovery.state["costs"]["compute_cost"] <= algorithm.recovery.state["compute_budget"]
    assert not algorithm.constraint_retention["requests"]
    assert goal.state["requests"] == before_requests and goal.state["pending"] is None
    assert len(algorithm.history) == 58 and adapter.simulator_calls == 0


def test_experimental_offers_use_native_kg_without_adding_the_nomination_proxy_to_action_value(
        project_root, monkeypatch, saved_incumbent_failure):
    algorithm, adapter, goal, _ = _experimental_runtime(project_root, monkeypatch, saved_incumbent_failure)
    algorithm.config = replace(algorithm.config, exact_kg_mc_samples=2)
    repeated = algorithm.history[-1][0]
    discriminating = (13, 27, 71)
    algorithm._terminal_goal_offers = {
        repeated: _goal_offer(repeated),
        discriminating: {**_goal_offer(discriminating), "purpose": "constraint_discrimination",
                         "prediction": {"information_proxy": 1.}}}
    candidates = [repeated, discriminating]
    pool = list(dict.fromkeys([*(point for point, _ in algorithm.history), discriminating]))
    mixture = algorithm.constraint_models["learned"]
    before = [(model.a.copy(), model.C.copy()) for model in mixture._finite_mixture_components]
    weights = mixture._finite_mixture_weights.copy()
    random_state = deepcopy(algorithm.rng.bit_generator.state)
    first = algorithm._exact_posterior_update_scores(candidates, pool)
    algorithm.rng.bit_generator.state = random_state
    algorithm._terminal_goal_offers[discriminating]["prediction"]["information_proxy"] = 1e9
    second = algorithm._exact_posterior_update_scores(candidates, pool)
    np.testing.assert_allclose(second, first, rtol=1e-12, atol=1e-12)
    assert len(first) == 2 and np.all(np.isfinite(first))
    np.testing.assert_array_equal(mixture._finite_mixture_weights, weights)
    for model, (a, covariance) in zip(mixture._finite_mixture_components, before):
        np.testing.assert_array_equal(model.a, a)
        np.testing.assert_array_equal(model.C, covariance)
    assert mixture._finite_mixture_update_count == 0
    assert not algorithm.constraint_retention["requests"] and goal.state["pending"] is None
    assert len(algorithm.history) == 58 and adapter.simulator_calls == 0


@pytest.mark.parametrize("reason", ["no_calls", "no_compute", "same_predictions"])
def test_experimental_unaffordable_or_uninformative_discrimination_preserves_the_original_goal_actions(
        project_root, monkeypatch, saved_incumbent_failure, reason):
    algorithm, adapter, goal, _ = _experimental_runtime(project_root, monkeypatch, saved_incumbent_failure)
    mixture = _strong_disagreement(algorithm)
    if reason == "no_calls":
        algorithm.config = replace(algorithm.config, N=58)
    elif reason == "no_compute":
        algorithm.recovery.state["costs"]["compute_cost"] = algorithm.recovery.state["compute_budget"]
    else:
        mixture._finite_mixture_components[0].a = mixture._finite_mixture_components[1].a.copy()
    assert algorithm._constraint_retention_offer() is None
    assert not algorithm.constraint_retention["offers"]
    assert not algorithm.constraint_retention["requests"]
    if reason != "no_calls":
        existing = [(13, 27, 71), (42, 63, 84)]
        monkeypatch.setattr(goal, "terminal_offers",
                            lambda *args, **kwargs: [_goal_offer(point) for point in existing])
        candidates, _ = algorithm._generate_candidates(59)
        assert candidates == existing
    assert len(algorithm.history) == 58 and adapter.simulator_calls == 0


@pytest.mark.parametrize("active", ["learned", "default"])
def test_one_selected_vector_updates_the_retained_competition_once_in_active_or_shadow_role(
        project_root, monkeypatch, saved_incumbent_failure, active):
    algorithm, adapter, goal, _ = _retention_runtime(project_root, monkeypatch, saved_incumbent_failure)
    mixture = algorithm.constraint_models["learned"]
    algorithm.active_constraint_basis = active
    algorithm.gpr[1] = algorithm.constraint_models[active]
    algorithm.variance_model = algorithm.constraint_variances[active]
    algorithm.adoption_policy.state["selected"] = active
    goal.state["scope_research"] = None
    algorithm._recovery_control_purpose = "optimization"
    point = tuple(saved_incumbent_failure["simulations"][26]["point"])
    output = np.asarray(saved_incumbent_failure["simulations"][26]["output"])
    expected_means = [float(model.posterior_mean(point)) for model in mixture._finite_mixture_components]
    before = [(model.a.copy(), model.C.copy()) for model in mixture._finite_mixture_components]
    before_weights = mixture._finite_mixture_weights.copy()
    draws, goal_labels, constraint_labels = [], [], []
    original_goal_observe = goal.observe

    def replay_vector(selected, rng):
        draws.append(tuple(selected))
        assert len(algorithm.history) == 58 and tuple(selected) == point
        assert algorithm._recovery_control_purpose == "optimization"
        assert not algorithm.constraint_retention["observations"]
        np.testing.assert_array_equal(mixture._finite_mixture_weights, before_weights)
        for model, (a, covariance) in zip(mixture._finite_mixture_components, before):
            np.testing.assert_array_equal(model.a, a)
            np.testing.assert_array_equal(model.C, covariance)
        return output.copy()

    def goal_observe(*args, **kwargs):
        goal_labels.append(kwargs["context"]["global_index"])
        return original_goal_observe(*args, **kwargs)

    monkeypatch.setattr(adapter, "simulate", replay_vector)
    monkeypatch.setattr(goal, "observe", goal_observe)
    monkeypatch.setattr(algorithm.recovery, "observe",
        lambda *args, **kwargs: constraint_labels.append(kwargs["context"]["global_index"]))
    # The test exercises this adapter's update seam, independently of the
    # unrelated native recommendation bookkeeping that run() invokes next.
    monkeypatch.setattr(type(algorithm).__mro__[1], "_update_posterior_dominance_incumbent",
                        lambda *args, **kwargs: None)
    algorithm._simulate_and_store(point)
    pending = algorithm._pending_adoption
    assert pending["retained_prediction"]["means"] == expected_means
    assert pending["retained_prediction"]["weights_before"] == before_weights.tolist()
    assert mixture._finite_mixture_update_count == 0
    # These are the two active updates performed by native run() before its
    # post-update callback; the adapter performs the shadow update itself.
    for index, model in enumerate(algorithm.gpr):
        model.update(point, float(output[index]),
                     algorithm.variance_model.predict_variance(index, point, adapter))
    algorithm._update_posterior_dominance_incumbent(stage=59, reason="test_saved_vector")
    record, = algorithm.constraint_retention["observations"]
    assert record["means"] == expected_means and record["target"] == float(output[1])
    assert record["purpose"] == "optimization"
    assert record["weights_before"] == before_weights.tolist()
    assert record["weights_after"] == mixture._finite_mixture_weights.tolist()
    assert mixture._finite_mixture_update_count == 1
    assert algorithm.constraint_retention["component_updates"] == 3
    assert draws == [point] and goal_labels == constraint_labels == [58]
    assert len(algorithm.history) == 59 and algorithm.config.N == 60
    assert adapter.simulator_calls == 0  # One saved vector was replayed.


def test_default_keeps_existing_candidates_and_uses_native_mixture_kg_without_extra_probe_search(
        project_root, monkeypatch, saved_incumbent_failure):
    algorithm, adapter, goal, _ = _retention_runtime(project_root, monkeypatch, saved_incumbent_failure)
    algorithm.config = replace(algorithm.config, exact_kg_mc_samples=2)
    repeated = algorithm.history[-1][0]
    native = (13, 27, 71)
    monkeypatch.setattr(goal, "terminal_offers", lambda *args, **kwargs: [_goal_offer(repeated)])
    monkeypatch.setattr(type(algorithm).__mro__[1], "_generate_candidates",
        lambda self, iteration: ([native], {native: "native_test_control"}))
    before_cost = algorithm.recovery.state["costs"]["compute_cost"]
    before_retention = deepcopy(algorithm.constraint_retention)
    candidates, sources = algorithm._generate_candidates(59)
    assert candidates == [native, repeated]
    assert sources == {native: "native_test_control", repeated: "objective_terminal_choice"}
    assert algorithm.constraint_retention == before_retention
    assert algorithm.recovery.state["costs"]["compute_cost"] == before_cost
    mixture = algorithm.constraint_models["learned"]
    before = [(model.a.copy(), model.C.copy()) for model in mixture._finite_mixture_components]
    weights = mixture._finite_mixture_weights.copy()
    pool = list(dict.fromkeys([*(point for point, _ in algorithm.history), native]))
    scores = algorithm._exact_posterior_update_scores(candidates, pool)
    assert len(scores) == 2 and np.all(np.isfinite(scores))
    np.testing.assert_array_equal(mixture._finite_mixture_weights, weights)
    for model, (a, covariance) in zip(mixture._finite_mixture_components, before):
        np.testing.assert_array_equal(model.a, a)
        np.testing.assert_array_equal(model.C, covariance)
    assert mixture._finite_mixture_update_count == 0
    assert len(algorithm.history) == 58 and adapter.simulator_calls == 0
