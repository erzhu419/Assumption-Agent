"""Independent coefficients retain accepted spatial structure.

Actual optional SC-KG GPs consume controlled labels on the same paid controls;
these tests make no physical simulator calls.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from test_cognition_sc_kg_environment import project_root
from test_cognition_objective_incumbent import saved_incumbent_failure
from test_cognition_objective_replacement_runtime import _runtime
from hegel_machine.cognition.constraint_retention import (
    condition_coefficient_readout, install_program_competition, program_terms,
)
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.sc_kg_environment import SCKGEnvironment, compile_sc_kg_program


def test_decomposition_combines_coefficients_and_removes_canceled_terms():
    x, y = (Program("feature", name=name) for name in ("x1", "x2"))
    program = Program("add", children=(Program("const", value=2.), Program("add", children=(
        Program("sub", children=(Program("mul", children=(Program("const", value=3.), x)), x)),
        Program("mul", children=(Program("sub", children=(y, Program("mul", children=(
            Program("const", value=2.), y)))), Program("const", value=2.)))))))
    intercept, terms, coefficients = program_terms(program)
    assert intercept == 2. and dict(zip(terms, coefficients)) == {x: 2., y: -2.}
    canceled = Program("add", children=(Program("sub", children=(x, x)),
        Program("mul", children=(Program("const", value=0.), y))))
    assert program_terms(canceled) == (0., (), ())


def test_decomposition_keeps_products_minima_and_branches_executable():
    x, y = (Program("feature", name=name) for name in ("x1", "x2"))
    square = Program("mul", children=(x, x))
    minimum = Program("min", children=(x, y))
    branch = Program("if", children=(Program("lt", children=(x, Program("const", value=.5))), x, y))
    program = Program("mul", children=(Program("const", value=3.), Program("sub", children=(
        Program("add", children=(square, Program("mul", children=(Program("const", value=2.), minimum)))),
        branch))))
    intercept, terms, coefficients = program_terms(program)
    assert intercept == 0.
    assert dict(zip(terms, coefficients)) == {square: 3., minimum: 6., branch: -3.}
    for a, b in ((.2, .7), (.8, .3), (.5, .5)):
        row = Observation({"x1": a, "x2": b}, 0.)
        assert intercept + sum(c * term.predict(row) for term, c in zip(terms, coefficients)) == pytest.approx(
            program.predict(row))


def test_callable_coefficients_expand_without_losing_the_definition_mean():
    arg = Program("arg", name="v")
    square = Definition("square", ("v",), Program("mul", children=(arg, arg)))
    weighted = Definition("weighted_square", ("v",), Program("add", children=(
        Program("const", value=1.), Program("mul", children=(Program("const", value=2.),
            Program("call", name=square.definition_id, children=(arg,), definition=square))))))
    x = Program("feature", name="x1")
    call = Program("call", name=weighted.definition_id, children=(x,), definition=weighted)
    intercept, terms, coefficients = program_terms(call)
    assert intercept == 1. and coefficients == (2.,)
    assert terms == (Program("mul", children=(x, x)),)
    row = Observation({"x1": .3}, 0.)
    assert intercept + coefficients[0] * terms[0].predict(row) == pytest.approx(call.predict(row))


@pytest.fixture
def coefficient_competition(project_root):
    environment = SCKGEnvironment(project_root, dimension=3, seed=841000171)
    from core.gpr import ParametricGPR
    from algorithms.single_olhkg import SingleOLHKGAlgorithm
    from hegel_machine.cognition.sc_kg_environment import HegelProgramTermBasis

    x, y = (Program("feature", name=name) for name in ("x1", "x2"))
    program = Program("add", children=(Program("const", value=2.), Program("add", children=(
        Program("mul", children=(Program("const", value=3.), x)),
        Program("mul", children=(Program("const", value=4.), Program("mul", children=(y, y))))))))
    adapter = compile_sc_kg_program(environment, program)
    basis = HegelProgramTermBasis(adapter.basis)
    original = ParametricGPR(3, lambda_i=.01, prior_var=.2,
        normalize_func=adapter.normalize, basis_map=adapter.basis)
    original.set_parametric_prior(np.array([.2, 1.1]), .01, .2)
    coefficients = ParametricGPR(3, lambda_i=.01, prior_var=.2,
        normalize_func=adapter.normalize, basis_map=basis)
    points = [(10, 20, 30), (80, 20, 40), (20, 80, 50), (80, 80, 60), (40, 50, 70)]
    history = []
    for point in [*points, points[0]]:
        target = .5 + point[0] / 100. - 2 * (point[1] / 100.) ** 2
        original.update(point, target, .02)
        coefficients.update(point, target, .02)
        history.append((point, [0., target]))
    assert condition_coefficient_readout(coefficients, history, .02)
    clone = object.__new__(SingleOLHKGAlgorithm)._clone_gpr_for_exact_kg
    mixture = install_program_competition(original, clone, coefficient_model=coefficients)
    return original, coefficients, mixture, clone, environment, history


def test_independent_basis_preserves_program_and_normalizes_its_terms(coefficient_competition):
    original, coefficients, _, _, environment, _ = coefficient_competition
    basis = coefficients.basis_map
    assert basis.program == original.basis_map.program
    assert basis.feature_names == original.basis_map.feature_names
    assert (basis.lower, basis.upper) == (original.basis_map.lower, original.basis_map.upper)
    assert basis.feature_dim == len(basis.terms) == 2
    points = [(20, 40, 60), (70, 90, 10)]
    features = basis.features_many(points)
    assert features.shape == (2, 2)
    for index, point in enumerate(points):
        row = Observation(dict(zip(basis.feature_names, (value / 100 for value in point))), 0.)
        np.testing.assert_allclose(features[index], [term.predict(row) for term in basis.terms])
        np.testing.assert_allclose(basis.features(point), features[index])
        assert np.r_[1., features[index]] @ basis.program_coefficients == pytest.approx(basis.program.predict(row))
    assert environment.costs["simulator_calls"] == 0


def test_affine_mean_and_full_covariance_survive_the_coordinate_change(coefficient_competition):
    original, coefficients, mixture, _, environment, _ = coefficient_competition
    affine, frozen, independent = mixture._finite_mixture_components
    assert affine.p == frozen.p == independent.p == coefficients.p == 3
    points = [*original.sampled_set, (64, 30, 20), (28, 60, 20)]
    old_design = original.augmented_feature_matrix(points)
    new_design = affine.augmented_feature_matrix(points)
    np.testing.assert_allclose(new_design @ affine.a, old_design @ original.a, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(new_design @ affine.C @ new_design.T,
        old_design @ original.C @ old_design.T, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(affine.posterior_var_many(points), original.posterior_var_many(points),
                               rtol=1e-11, atol=1e-12)
    np.testing.assert_array_equal(independent.a, coefficients.a)
    np.testing.assert_array_equal(independent.C, coefficients.C)
    np.testing.assert_array_equal(frozen.a[:frozen.p], coefficients.basis_map.program_coefficients)
    np.testing.assert_array_equal(frozen.a[frozen.p:], np.zeros(len(original.sampled_set)))
    assert all(component.sampled_set == original.sampled_set for component in (affine, frozen, independent))
    np.testing.assert_allclose(mixture._finite_mixture_weights, [1/3] * 3)
    assert mixture._finite_mixture_update_count == 0 and environment.costs["simulator_calls"] == 0


def test_independent_readout_separates_controls_collapsed_by_the_original_program(coefficient_competition):
    original, coefficients, mixture, _, _, history = coefficient_competition
    points = [(64, 30, 20), (28, 60, 20)]
    affine, _, independent = mixture._finite_mixture_components
    assert all(point not in original.sampled_set for point in points)
    assert original.basis_map.features(points[0]) == pytest.approx(original.basis_map.features(points[1]))
    assert affine.posterior_mean(points[0]) == pytest.approx(affine.posterior_mean(points[1]))
    expected = [.5 + point[0] / 100. - 2 * (point[1] / 100.) ** 2 for point in points]
    np.testing.assert_allclose(independent.posterior_mean_many(points), expected, atol=1e-10)
    assert abs(expected[0] - expected[1]) > .5
    # Replicates share one local deviation and reduce measurement noise.
    design = coefficients.basis_matrix(coefficients.sampled_set)
    counts = np.array([sum(control == point for control, _ in history)
                       for point in coefficients.sampled_set])
    expected_covariance = np.linalg.inv(design.T @ (design / (.01 + .02 / counts)[:, None]))
    np.testing.assert_allclose(coefficients.C[:coefficients.p, :coefficients.p], expected_covariance)


def test_one_future_label_updates_three_preobserved_likelihoods_and_resets_only_frozen_mean(
        coefficient_competition):
    original, coefficients, mixture, _, environment, _ = coefficient_competition
    original_state = original.a.copy(), original.C.copy(), coefficients.a.copy(), coefficients.C.copy()
    point, target, noise = (90, 70, 20), -.1, .02
    components = mixture._finite_mixture_components
    old_count = len(original.sampled_set)
    means = np.array([component.posterior_mean(point) for component in components])
    variances = np.array([component.posterior_var(point) + noise for component in components])
    mass = np.log(mixture._finite_mixture_weights) - .5 * (
        np.log(2 * np.pi * variances) + (target - means) ** 2 / variances)
    expected = np.exp(mass - max(mass))
    expected /= sum(expected)
    mixture.update(point, target, noise)
    np.testing.assert_allclose(mixture._finite_mixture_weights, expected, rtol=1e-12, atol=1e-12)
    assert mixture._finite_mixture_update_count == 1
    assert all(len(component.sampled_set) == old_count + 1 for component in components)
    assert all(component.sampled_set[-1] == point for component in components)
    frozen = components[1]
    np.testing.assert_array_equal(frozen.a[:frozen.p], frozen.basis_map.program_coefficients)
    np.testing.assert_array_equal(frozen.a[frozen.p:], np.zeros(old_count + 1))
    for control in [point, (5, 95, 55), original.sampled_set[0]]:
        assert frozen.posterior_mean(control) == pytest.approx(original.basis_map.features(control)[0])
    for current, before in zip((original.a, original.C, coefficients.a, coefficients.C), original_state):
        np.testing.assert_array_equal(current, before)
    assert environment.costs["simulator_calls"] == 0


def test_native_fantasy_clone_keeps_all_three_components_out_of_live_state(coefficient_competition):
    _, _, mixture, clone, environment, _ = coefficient_competition
    before = mixture.a.copy(), mixture.C.copy(), mixture._finite_mixture_weights.copy()
    components = [(component.a.copy(), component.C.copy(), list(component.sampled_set))
                  for component in mixture._finite_mixture_components]
    fantasy = clone(mixture)
    fantasy.update((90, 70, 20), -.1, .02)
    for current, expected in zip((mixture.a, mixture.C, mixture._finite_mixture_weights), before):
        np.testing.assert_array_equal(current, expected)
    for live, imagined, (mean, covariance, points) in zip(
            mixture._finite_mixture_components, fantasy._finite_mixture_components, components):
        assert live is not imagined
        np.testing.assert_array_equal(live.a, mean)
        np.testing.assert_array_equal(live.C, covariance)
        assert live.sampled_set == points and len(imagined.sampled_set) == len(points) + 1
    assert mixture._finite_mixture_update_count == 0 and fantasy._finite_mixture_update_count == 1
    assert environment.costs["simulator_calls"] == 0


def test_nonidentified_paid_design_keeps_the_original_posterior(coefficient_competition):
    from core.gpr import ParametricGPR

    _, fitted, _, _, _, _ = coefficient_competition
    model = ParametricGPR(3, lambda_i=.01, prior_var=.2, basis_map=fitted.basis_map)
    history = [((10, 20, 30), [0., .4]), ((80, 20, 40), [0., 1.1])]
    for point, output in history:
        model.update(point, output[1], .02)
    mean, covariance, version = model.a.copy(), model.C.copy(), model._state_version
    assert not condition_coefficient_readout(model, history, .02)
    np.testing.assert_array_equal(model.a, mean)
    np.testing.assert_array_equal(model.C, covariance)
    assert model._state_version == version


def test_conditioned_covariance_makes_one_future_update_match_the_complete_paid_fit(coefficient_competition):
    _, coefficients, _, clone, _, history = coefficient_competition
    online, batch = clone(coefficients), clone(coefficients)
    point, target = (90, 70, 20), -.1
    online.update(point, target, .02)
    batch.update(point, target, .02)
    assert condition_coefficient_readout(batch, [*history, (point, [0., target])], .02)
    controls = [*coefficients.sampled_set, point, (64, 30, 20), (28, 60, 20)]
    np.testing.assert_allclose(online.posterior_mean_many(controls), batch.posterior_mean_many(controls),
                               rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(online.posterior_var_many(controls), batch.posterior_var_many(controls),
                               rtol=1e-9, atol=1e-10)


def test_only_an_accepted_constraint_scope_changes_the_initial_v056_readouts(
        project_root, monkeypatch, saved_incumbent_failure):
    import test_cognition_sc_kg_environment as runtime_tests

    source = Path(__file__).resolve().parents[1] / (
        "artifacts/sc_kg_dual_value_20260912/revision3/method_source/sc_kg_environment.py")
    module_spec = importlib.util.spec_from_file_location(
        "hegel_machine.cognition._v056_constraint_initialization_reference", source)
    frozen = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(frozen)
    algorithm, adapter, _, _ = _runtime(project_root, monkeypatch, saved_incumbent_failure)
    with monkeypatch.context() as baseline_context:
        baseline_context.setattr(runtime_tests, "initialize_sc_kg_optimizer", frozen.initialize_sc_kg_optimizer)
        baseline, old_adapter, _, _ = _runtime(project_root, baseline_context, saved_incumbent_failure)
    # Preserve the saved-prefix helper's explicit accepted-revision seam after
    # restoring the initializer used by the rest of this module.
    monkeypatch.setattr(baseline.recovery, "acknowledge_revision", lambda: None)
    points = [*algorithm.constraint_models["default"].sampled_set[:3], (13, 27, 71), (41, 62, 83)]
    for name in ("default", "learned"):
        live, old = algorithm.constraint_models[name], baseline.constraint_models[name]
        np.testing.assert_array_equal(live.a, old.a)
        np.testing.assert_array_equal(live.C, old.C)
        np.testing.assert_allclose(live.posterior_mean_many(points), old.posterior_mean_many(points))
        np.testing.assert_allclose(live.posterior_var_many(points), old.posterior_var_many(points))
    assert algorithm.constraint_retention["paid_design_fits"] == 0
    assert algorithm.constraint_retention["paid_design_fit_records"] == 0

    for optimizer, problem in ((algorithm, adapter), (baseline, old_adapter)):
        optimizer._rebase_scope({"scope_start": 44, "program": problem.basis.program}, stage=58)
    assert algorithm.constraint_retention["paid_design_fits"] == 2
    assert algorithm.constraint_retention["paid_design_fit_records"] == 28
    np.testing.assert_array_equal(algorithm.constraint_models["default"].a,
                                  baseline.constraint_models["default"].a)
    np.testing.assert_array_equal(algorithm.constraint_models["default"].C,
                                  baseline.constraint_models["default"].C)
    affine, retained, coefficients = algorithm.constraint_models["learned"]._finite_mixture_components
    assert affine.p == retained.p == coefficients.p > 2
    np.testing.assert_array_equal(retained.a[:retained.p], retained.basis_map.program_coefficients)
    np.testing.assert_allclose(retained.posterior_mean_many(points),
        [adapter.basis.features(point)[0] for point in points])
    assert adapter.simulator_calls == old_adapter.simulator_calls == 0
    assert len(algorithm.history) == len(baseline.history) == 58
