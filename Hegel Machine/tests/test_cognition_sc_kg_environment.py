"""Real optional SC-KG measurements and consumption of learned programs."""

from pathlib import Path
import json

import pytest

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Observation, Probe, SearchBudget
from hegel_machine.cognition.sc_kg_environment import (
    SCKGEnvironment, compile_sc_kg_program, initialize_sc_kg_optimizer,
)
from hegel_machine.cognition.sparse_search import sparse_candidates


@pytest.fixture(scope="module")
def project_root():
    root = Path(__file__).resolve().parents[3] / "KG_op" / "SC-OLH-KG"
    if not (root / "problems" / "rzdt.py").is_file():
        pytest.skip("the optional local SC-KG checkout is unavailable")
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    return root


def test_public_interface_executes_actual_quantized_sc_kg_controls(project_root):
    environment = SCKGEnvironment(project_root, problem="rzdt2", seed=841000017)
    other = SCKGEnvironment(project_root, problem="rzdt1", seed=841000017)
    assert environment.spec == other.spec
    assert environment.spec.context["objective_direction"] == "minimize"
    assert environment.spec.context["constraint_upper_bound"] == "0.0"
    controls = {name: .254 for name in environment.spec.feature_names}
    row, = environment.observe(Probe("quantized", (controls,)))
    assert set(row.features.values()) == {.25}
    assert row.history == ()
    from problems.rzdt import RZDT2
    import numpy as np
    measured = RZDT2(d=5, L=100, sigma=0, heteroscedastic=False).simulate(
        (25,) * 5, np.random.default_rng(841000017))[2]
    assert row.target == measured
    assert environment.acquisitions[0]["integer_points"] == [[25] * 5]
    assert environment.costs == {"observation_cost": 1, "simulator_calls": 1}


def test_replicates_are_paid_and_evaluation_preserves_learning_stream(project_root):
    environment = SCKGEnvironment(project_root, seed=841000018, noise=.1)
    clone = SCKGEnvironment(project_root, seed=841000018, noise=.1)
    probe = environment.spec.initial_probes[0]
    first = environment.observe(probe)
    assert first == clone.observe(probe)
    evaluation = environment.evaluation_rows(seed=841000019, count=70)
    second = environment.observe(probe)
    assert second == clone.observe(probe)
    assert [row.target for row in first] != [row.target for row in second]
    assert len(evaluation) == 70
    assert environment.costs["simulator_calls"] == 16
    assert environment.evaluation_costs["simulator_calls"] == 70
    assert len(environment.evaluation_acquisitions) == 2


def test_state_policy_public_measurements_preserve_hidden_family_and_full_vectors(project_root):
    import numpy as np
    environment = SCKGEnvironment(project_root, problem="state_policy_rzdt1",
        dimension=3, seed=841000029)
    from problems.rzdt import StatePolicyRZDT1
    ordinary = SCKGEnvironment(project_root, dimension=3, seed=841000029)
    assert environment.spec == ordinary.spec
    rows = environment.observe(environment.spec.initial_probes[0])
    raw = StatePolicyRZDT1(d=3, L=100, sigma=0., heteroscedastic=False)
    rng = np.random.default_rng(841000029)
    for row, vector in zip(rows, environment.acquisitions[0]["vector_outputs"]):
        measured = raw.simulate(environment.integer_point(row.features), rng)
        np.testing.assert_array_equal(vector, measured)
        assert row.target == measured[2]
    assert environment.costs["simulator_calls"] == 8


def test_learned_sc_kg_relation_is_consumed_by_its_real_gp(project_root):
    source = SCKGEnvironment(project_root, problem="rzdt2", seed=841000021)
    observed = [row for probe in (*source.spec.initial_probes, *source.spec.available_probes[:2])
                for row in source.observe(probe)]
    candidates = sparse_candidates(observed, budget=SearchBudget(24000), max_history=0)
    program = min(candidates, key=lambda item: sum(
        (item.program.predict(row) - row.target) ** 2 for row in observed)).program

    target = SCKGEnvironment(project_root, problem="rzdt2", seed=841000022)
    adapter = compile_sc_kg_program(target, program)
    from core.gpr import ParametricGPR
    import numpy as np
    model = ParametricGPR(d=5, lambda_i=1e-12, prior_var=1e6,
                         basis_map=adapter.gpr_basis_map(output_index=1))
    wrong = compile_sc_kg_program(target, Program("feature", name="x1"))
    linear = ParametricGPR(d=5, lambda_i=1e-12, prior_var=1e6,
                          basis_map=wrong.gpr_basis_map(output_index=1))
    training = target.observe(target.spec.initial_probes[0])
    for row in training:
        point = target.integer_point(row.features)
        model.update(point, row.target, sigma2_hat=1e-8)
        linear.update(point, row.target, sigma2_hat=1e-8)
    heldout = target.evaluation_rows(seed=841000023, count=64)
    points = [target.integer_point(row.features) for row in heldout]
    labels = np.asarray([row.target for row in heldout])
    before = adapter.basis.evaluations
    predictions = model.posterior_mean_many(points)
    assert adapter.basis.evaluations >= before + len(points)
    learned_error = float(np.mean((predictions - labels) ** 2))
    linear_error = float(np.mean((linear.posterior_mean_many(points) - labels) ** 2))
    assert learned_error < 1e-9
    assert learned_error < linear_error / 1000
    assert model.p == 2
    assert adapter.gpr_basis_map(output_index=0) is None
    response = adapter.simulate(points[0], np.random.default_rng(841000024))
    assert len(response) == 2 and response[1] == labels[0]
    assert adapter.simulator_calls == 1


def test_export_preserves_callable_body_and_refuses_historical_programs(project_root):
    environment = SCKGEnvironment(project_root, seed=841000025)
    definition = Definition("learned_square", ("value",),
        Program("mul", children=(Program("arg", name="value"),) * 2))
    program = Program("call", name=definition.definition_id,
                      children=(Program("feature", name="x1"),), definition=definition)
    adapter = compile_sc_kg_program(environment, program)
    assert adapter.basis.features((25,) * 5) == [.0625]
    assert adapter.basis.program == program
    for unsupported in (Program("lag", name="x1", lag=1), Program("available", lag=1),
                        Program("feature", name="absent")):
        with pytest.raises(ValueError):
            compile_sc_kg_program(environment, unsupported)
    assert environment.costs["simulator_calls"] == 0


def test_invalid_public_control_is_rejected_before_any_measurement(project_root):
    environment = SCKGEnvironment(project_root)
    outside = {name: .5 for name in environment.spec.feature_names}
    outside["x1"] = 1.5
    with pytest.raises(ValueError, match="normalized controls"):
        environment.observe(Probe("outside", (outside,)))
    with pytest.raises(ValueError, match="every public feature"):
        environment.observe(Probe("missing", ({"x1": .5},)))
    assert not environment.acquisitions


def test_json_snapshot_resumes_noise_and_costs_without_rebuying(project_root):
    source = SCKGEnvironment(project_root, problem="rzdt2", seed=841000031, noise=.1)
    source.observe(source.spec.initial_probes[0])
    source.evaluation_rows(count=7)
    snapshot = json.loads(json.dumps(source.snapshot()))
    resumed = SCKGEnvironment(project_root, problem="rzdt2", seed=841000031, noise=.1)
    resumed.restore(snapshot)
    assert resumed.acquisitions == source.acquisitions
    assert resumed.costs == source.costs == {"observation_cost": 8, "simulator_calls": 8}
    assert resumed.evaluation_costs == source.evaluation_costs
    probe = source.spec.initial_probes[1]
    assert resumed.observe(probe) == source.observe(probe)
    assert resumed.snapshot() == source.snapshot()
    resumed.acquisitions[0]["targets"][0] += 1
    assert source.acquisitions[0]["targets"] != resumed.acquisitions[0]["targets"]


def test_all_paid_vector_replicates_initialize_real_optimizer_without_new_calls(project_root):
    import numpy as np
    environment = SCKGEnvironment(project_root, problem="rzdt1", seed=841000041, noise=.03)
    probes = (*environment.spec.initial_probes, environment.spec.available_probes[0],
              environment.spec.initial_probes[0])
    for probe in probes:
        environment.observe(probe)
    saved = json.loads(json.dumps(environment.snapshot()))
    resumed = SCKGEnvironment(project_root, problem="rzdt1", seed=841000041, noise=.03)
    resumed.restore(saved)
    adapter = compile_sc_kg_program(resumed, Program("feature", name="x1"))
    assert len(adapter.paid_observations) == 32
    assert len({row["point"] for row in adapter.paid_observations}) == 24
    from algorithms.single_olhkg import SingleOLHKGConfig
    config = SingleOLHKGConfig(N=34, n0=8, K1=3, K2=0, posterior_pool_size=8,
        posterior_keep=1, axis_candidate_count=0, state_candidate_count=0,
        use_state_coupling=False, use_state_basis=False, lambda_coupling=0.,
        variance_mode="pooled", exact_kg_mc_samples=1, eval_pool_size=8,
        evaluate_interval=0, truth_pool_diagnostics=False, checkpoint_interval=0,
        use_recommendation_refinement=False, recommendation_axis_oracle=False,
        seed=841000042)
    algorithm = initialize_sc_kg_optimizer(adapter, config)
    result = algorithm.run()
    assert algorithm.config.n0 == adapter.loaded_observation_count == 32
    assert adapter.research_simulator_calls == 32
    assert adapter.simulator_calls == 2
    assert result["n_simulations"] == len(algorithm.history) == 34
    assert len(algorithm.variance_model.records[1]) == 34
    for (point, output), record in zip(algorithm.history, adapter.paid_observations):
        assert point == record["point"]
        np.testing.assert_array_equal(output, record["output"])
    first = adapter.paid_observations[0]
    repeat = adapter.paid_observations[24]
    assert first["point"] == repeat["point"]
    assert first["output"] != repeat["output"]
    assert first["output"][0] == .5 * sum(first["vector_output"][:2])
    assert adapter.basis.evaluations > 32


def test_missing_old_vector_outputs_cannot_be_reconstructed_for_initialization(project_root):
    environment = SCKGEnvironment(project_root, seed=841000043)
    environment.observe(environment.spec.initial_probes[0])
    environment.acquisitions[0].pop("vector_outputs")
    adapter = compile_sc_kg_program(environment, Program("feature", name="x1"))
    from algorithms.single_olhkg import SingleOLHKGConfig
    with pytest.raises(ValueError, match="lack actual vector outputs"):
        initialize_sc_kg_optimizer(adapter, SingleOLHKGConfig(N=32))
    assert adapter.simulator_calls == 0


def _adoption_setup(project_root, seed=841000051, total=38):
    environment = SCKGEnvironment(project_root, problem="rzdt2", seed=seed, noise=.03)
    for probe in (*environment.spec.initial_probes, *environment.spec.available_probes[:2]):
        environment.observe(probe)
    from algorithms.single_olhkg import SingleOLHKGConfig
    config = SingleOLHKGConfig(N=total, K1=3, K2=0, posterior_pool_size=8,
        posterior_keep=1, axis_candidate_count=0, state_candidate_count=0,
        use_state_coupling=False, use_state_basis=False, lambda_coupling=0.,
        variance_mode="pooled", exact_kg_mc_samples=1, eval_pool_size=8,
        evaluate_interval=0, truth_pool_diagnostics=False, checkpoint_interval=0,
        use_recommendation_refinement=False, recommendation_axis_oracle=False,
        seed=seed + 1)
    return environment, config


class _ScriptedAdoption:
    def __init__(self, selections):
        self.selections = selections
        self.selected = "default"
        self.predictions = []

    def observe(self, default_mean, learned_mean, target, *, noise_variance, context):
        self.predictions.append((default_mean, learned_mean, target, noise_variance, context))
        self.selected = self.selections[min(len(self.predictions) - 1, len(self.selections) - 1)]
        return self.selected


def test_adoption_switches_real_acquisition_models_and_retracts_after_measurement(project_root):
    environment, config = _adoption_setup(project_root)
    adapter = compile_sc_kg_program(environment, Program("feature", name="x1"))
    policy = _ScriptedAdoption(["default", "learned", "learned", "default"])
    algorithm = initialize_sc_kg_optimizer(adapter, config, adoption_policy=policy)
    scored = []
    original_score = algorithm.acquisition.score

    def record_score(candidates, objective, constraint, variance, *args, **kwargs):
        scored.append((constraint, variance))
        return original_score(candidates, objective, constraint, variance, *args, **kwargs)

    algorithm.acquisition.score = record_score
    algorithm.run()
    expected = ["default", "default", "learned", "learned", "default", "default"]
    assert len(scored) == len(policy.predictions) == adapter.simulator_calls == 6
    for (model, variance), name in zip(scored, expected):
        assert model is algorithm.constraint_models[name]
        assert variance is algorithm.constraint_variances[name]
    assert algorithm.gpr[1] is algorithm.constraint_models["default"]
    assert algorithm.active_constraint_basis == policy.selected == "default"
    assert len(algorithm.history) == adapter.loaded_observation_count + 6 == 38
    assert [event["selected_after"] for event in algorithm.adoption_events] == [
        "default", "learned", "learned", "default", "default", "default"]
    for name, model in algorithm.constraint_models.items():
        assert set(model.sampled_set) == {point for point, _ in algorithm.history}
        assert len(algorithm.constraint_variances[name].records[1]) == 38
    assert algorithm.adoption_diagnostics["additional_fit_records"] == 32
    assert algorithm.adoption_diagnostics["paired_prediction_rows"] == 12
    assert algorithm.adoption_diagnostics["shadow_gp_updates"] == 6
    assert algorithm.adoption_diagnostics["shadow_hvd_updates"] == 12
    # Logged means match the original optimizer's predictions before that
    # label was measured, rather than its posterior after fitting the label.
    for row, event in zip(algorithm.iteration_log, algorithm.adoption_events):
        assert row["mu_before"][1] == pytest.approx(event[event["selected_before"] + "_mean"])
        assert row["Y_observed"][1] == event["target"]


def test_never_adopt_preserves_original_default_actions_and_measurement_stream(project_root):
    import numpy as np
    from hegel_machine.cognition.sc_kg_environment import HegelProgramProblem

    class DefaultProblem(HegelProgramProblem):
        def gpr_basis_map(self, output_index=1):
            return None

    environment, config = _adoption_setup(project_root, seed=841000061)
    program = Program("feature", name="x1")
    default = initialize_sc_kg_optimizer(DefaultProblem(environment, program), config)
    adaptive = initialize_sc_kg_optimizer(compile_sc_kg_program(environment, program),
        config, adoption_policy=_ScriptedAdoption(["default"]))
    default_result, adaptive_result = default.run(), adaptive.run()
    assert [row["x_selected"] for row in default.iteration_log] == [
        row["x_selected"] for row in adaptive.iteration_log]
    for (_, first), (_, second) in zip(default.history, adaptive.history):
        np.testing.assert_array_equal(first, second)
    assert default_result["x_recommended"] == adaptive_result["x_recommended"]
    assert default.problem.simulator_calls == adaptive.problem.simulator_calls == 6
    assert adaptive.adoption_diagnostics["policy_observations"] == 6


@pytest.mark.parametrize("change_after", [12, 20])
def test_real_joint_failure_rebases_supported_models_within_budget(project_root, change_after):
    from importlib.util import module_from_spec, spec_from_file_location
    import numpy as np
    from hegel_machine.cognition.joint_model_revision import JointModelRevision
    from hegel_machine.cognition.memory import KnowledgeLibrary
    from hegel_machine.cognition.meta_prior_research import MetaPriorResearch
    from hegel_machine.cognition.models import ResearchConfig
    from hegel_machine.cognition.prior_adoption import PriorAdoption
    seed = 1043010067
    environment = SCKGEnvironment(project_root, problem="state_policy_rzdt1", dimension=3, seed=seed)
    from algorithms.single_olhkg import SingleOLHKGConfig
    report = MetaPriorResearch(ResearchConfig(observation_budget=32, compute_budget=60000,
        max_rounds=2, minimum_improvement=1e-7, seed=seed), KnowledgeLibrary()).run(environment)
    assert report["status"] == "completed" and report["family_id"] == "T11"
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_sc_kg_prior_mechanism_change.py"
    module_spec = spec_from_file_location("hegel_mechanism_change_case", script)
    module = module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    adapter = module.MechanismChangeProblem(environment, Program.from_dict(report["program"]),
                                            change_after=change_after)
    recovery = JointModelRevision(environment.spec, seed=seed)
    config = SingleOLHKGConfig(N=64, K1=6, K2=0, posterior_pool_size=32, posterior_keep=3,
        axis_candidate_count=0, state_candidate_count=0, n_thr=2,
        use_problem_initial_samples=False, use_boundary_initial_samples=False,
        use_recommendation_refinement=False, recommendation_axis_oracle=False,
        recommendation_axis_candidate_count=0, use_state_coupling=False,
        use_state_basis=False, lambda_coupling=0., variance_mode="pooled",
        acquisition_mode="exact_mc", exact_kg_mc_samples=1, exact_kg_jobs=1,
        eval_pool_size=32, evaluate_interval=0, truth_pool_diagnostics=False,
        checkpoint_interval=0, seed=seed + 4000009)
    algorithm = initialize_sc_kg_optimizer(adapter, config,
        adoption_policy=PriorAdoption(), recovery=recovery)
    original_objective = algorithm.gpr[0]
    singleton_rounds = {}
    original_scores = algorithm._exact_posterior_update_scores
    original_measure = adapter.simulate

    def score_singleton_with_native_actions(candidates, terminal_pool, *args, **kwargs):
        offers = algorithm._terminal_goal_offers
        goal_points = [point for point, offer in offers.items()
                       if offer["purpose"] != "constraint_discrimination"]
        if len(goal_points) == 1:
            nominated = goal_points[0]
            paid = {point for point, _ in algorithm.history}
            if nominated in paid:
                assert nominated in candidates and len(set(candidates)) > 1
                assert any(point not in paid for point in candidates)
                assert paid.issubset(set(terminal_pool))
                assert algorithm.objective_research.state["pending"] is None
                singleton_rounds[len(algorithm.history)] = {"nominated": nominated,
                    "requests_before": len(algorithm.objective_research.state["requests"]),
                    "offers": {point: offer["purpose"] for point, offer in offers.items()},
                    "candidates": list(candidates)}
        return original_scores(candidates, terminal_pool, *args, **kwargs)

    def measure_singleton_or_native(point, rng=None):
        check = singleton_rounds.get(len(algorithm.history))
        if check is not None:
            chosen = tuple(point) == check["nominated"]
            selected_purpose = check["offers"].get(tuple(point))
            check["selected_goal"] = chosen
            check["selected_offer"] = selected_purpose is not None
            assert tuple(point) in check["candidates"]
            assert len(algorithm.objective_research.state["requests"]) == check["requests_before"] + int(chosen)
            if chosen:
                assert algorithm.objective_research.state["pending"] is not None
            else:
                assert algorithm.objective_research.state["pending"] is None
                assert algorithm._recovery_control_purpose == (
                    "constraint_discrimination" if selected_purpose == "constraint_discrimination"
                    else "optimization")
        return original_measure(point, rng)

    algorithm._exact_posterior_update_scores = score_singleton_with_native_actions
    adapter.simulate = measure_singleton_or_native
    result = algorithm.run()
    assert result["n_simulations"] == len(algorithm.history) == 64
    assert adapter.loaded_observation_count == adapter.simulator_calls == 32
    if change_after == 12:
        assert singleton_rounds
        assert any(not row["selected_goal"] for row in singleton_rounds.values())
        assert algorithm.objective_diagnostics["terminal_native_pool_rounds"] == len(singleton_rounds)
        assert algorithm.objective_diagnostics["terminal_native_pool_candidates"] > len(singleton_rounds)
        assert algorithm.objective_diagnostics["terminal_outside_offer_actions"] == sum(
            not row["selected_offer"] for row in singleton_rounds.values())
    assert len(algorithm.scope_revisions) == 1
    revision = algorithm.scope_revisions[0]
    scope_start = 32 + change_after
    assert revision["scope_start"] == scope_start
    assert algorithm.active_scope_start == scope_start
    objective_installed = (not revision["preserved_objective_head"]
        or algorithm.objective_scope_revisions
        or algorithm.objective_diagnostics["replacement_installs"])
    if objective_installed:
        assert algorithm.gpr[0] is not original_objective
    else:
        assert algorithm.gpr[0] is original_objective
    assert adapter.basis.program.to_dict() != report["program"]
    scope_rows = 64 - scope_start
    assert len(algorithm.history[scope_start:]) == scope_rows
    repeated = {tuple(choice["point"]) for choice in algorithm.objective_diagnostics["terminal_choices"]
                if choice["action_kind"] == "replicate"}
    for variance in algorithm.constraint_variances.values():
        for index, start in ((0, algorithm.active_objective_scope_start), (1, scope_start)):
            paid = algorithm.history[start:]
            assert {point for point, _ in variance.records[index]} == {point for point, _ in paid}
            for point in repeated:
                values = [float(output[index]) for control, output in paid if control == point]
                if len(values) >= 2:
                    expected = max(float(np.var(values, ddof=1)), variance.floor)
                    assert [value for control, value in variance.records[index]
                            if control == point] == pytest.approx([expected])
    assert sum(map(len, algorithm.observations.values())) == scope_rows
    for (point, output), paid in zip(algorithm.history, adapter.paid_observations):
        assert point == paid["point"]
        np.testing.assert_array_equal(output, paid["output"])
    episode = recovery.to_dict()["episodes"][0]
    accepted = episode["accepted_revision"]
    assert set(accepted["training_indices"]).isdisjoint(accepted["validation_indices"])
    assert len(accepted["validation_indices"]) == 4
    actual_forecasts = sum(len(event["candidate_predictions"]) for event in algorithm.adoption_events)
    assert algorithm.recovery_diagnostics["candidate_prediction_rows"] == actual_forecasts
    assert actual_forecasts >= 4
    assert algorithm.recovery_diagnostics["scope_model_fits"] == (
        2 + int(not revision["preserved_objective_head"]))
    for purpose in ("confirmation", "coverage", "validation"):
        calls = sum(row["candidate_source_selected"] == f"revision_{purpose}"
                    for row in algorithm.iteration_log)
        assert calls == algorithm.recovery_diagnostics[f"{purpose}_requests"]
        assert calls == recovery.to_dict()["costs"][f"{purpose}_calls"]
        assert calls == sum(event["purpose"] == purpose for event in algorithm.adoption_events)
    confirmation, = recovery.to_dict()["confirmations"]
    assert confirmation["confirmed"]
    anchor = algorithm.adoption_events[confirmation["anchor_index"]]
    repeated = algorithm.adoption_events[confirmation["observation_index"]]
    assert anchor["purpose"] == "optimization" and repeated["purpose"] == "confirmation"
    assert anchor["point"] == repeated["point"]
    assert confirmation["frozen_predictions"] == {
        name: anchor[f"{name}_mean"] for name in ("default", "learned")}
    validation = [recovery.to_dict()["observations"][index]["observation"]
                  for index in accepted["validation_indices"]]
    assert max(abs(adapter.basis.program.predict(Observation(row["features"], row["target"]))
                   - row["target"]) for row in validation) < 1e-6
    # Independent post-run scoring is charged to a separate environment and
    # never returned to the optimizer or the candidate-selection controller.
    evaluation = SCKGEnvironment(project_root, problem="rzdt2", dimension=3, seed=seed + 11)
    heldout = evaluation.evaluation_rows(count=8)
    assert max(abs(adapter.basis.program.predict(row) - row.target) for row in heldout) < 1e-6
    assert evaluation.evaluation_costs["simulator_calls"] == 8
    assert adapter.simulator_calls == 32
    if change_after == 20:
        opened = next(event for event in recovery.to_dict()["events"]
                      if event["action"] == "reopen_hypothesis_research")
        remaining = config.N - opened["stage"]
        assert remaining < 8 + 4
        assert episode["planned_coverage"] == remaining - 4
        assert episode["reserved_validation"] == 4
        assert (algorithm.recovery_diagnostics["coverage_requests"]
                + algorithm.recovery_diagnostics["validation_requests"]) == remaining
        assert revision["stage"] == config.N


def test_real_promising_uncertified_control_can_compete_for_a_repeat_without_relaxing_recommendation(project_root, monkeypatch):
    """An actual noisy winner loses certification and may buy evidence to recover it."""
    _check_uncertified_nomination(project_root, monkeypatch, "objective_only_change_1_noise_0.03",
        "research_promising_constraint_certificate", "sc_kg_terminal_value_v040_20260910")


def test_real_supported_uncertified_neighbor_uses_native_kg_and_paid_observation(project_root, monkeypatch):
    _check_uncertified_nomination(project_root, monkeypatch, "value_precision_1_objective_only_change_noise_0.03",
        "supported_goal_with_uncertain_constraint", "sc_kg_value_precision_v041_20260910")


def _check_uncertified_nomination(project_root, monkeypatch, case_id, reason, frozen_directory):
    import numpy as np
    algorithm, adapter, environment = _saved_terminal_failure_optimizer(project_root, monkeypatch,
        case_id, frozen_directory=frozen_directory)
    original_offers = algorithm.objective_research.terminal_offers
    original_scores = algorithm._exact_posterior_update_scores
    original_measure = adapter.simulate
    nominations, purchased = {}, []

    def certificate(point):
        return algorithm._terminal_certificate_components(algorithm.gpr, algorithm.variance_model,
            [point], observations=algorithm.observations)

    def actual_offers(rows, forecast, **context):
        request_count = len(algorithm.objective_research.state["requests"])
        offers = original_offers(rows, forecast, **context)
        repairs = [offer for offer in offers if offer["reason"] == reason]
        assert len(repairs) <= 1 and len(offers) <= 4
        if repairs:
            offer = repairs[0]
            point = environment.integer_point(offer["features"])
            assert certificate(point)["margin"][0] > 0.
            assert offer["feasible"] is False
            if reason == "supported_goal_with_uncertain_constraint":
                anchor = environment.integer_point(offer["support_anchor"]["features"])
                assert all(point != control for control, _ in algorithm.history)
                changed = [name for name in environment.spec.feature_names
                           if offer["features"][name] != offer["support_anchor"]["features"][name]]
                assert len(changed) == 1
                name = changed[0]
                low, high = environment.spec.feature_bounds[name]
                actual_cell = round(32 * (offer["features"][name] - low) / (high - low))
                anchor_cell = round(32 * (offer["support_anchor"]["features"][name] - low) / (high - low))
                assert abs(actual_cell - anchor_cell) == 1
                assert offer["prediction"]["constraint_mean"] <= 0.
                assert offer["prediction"]["predicted_gain"] > offer["prediction"]["gain_uncertainty"] + 1e-7
                assert offer["purpose"] == "objective_exploration"
            else:
                anchor = point
                assert offer["purpose"] == "objective_confirmation"
            actual = [float(output[1]) for index, (control, output) in enumerate(algorithm.history)
                      if index >= algorithm.active_scope_start and control == anchor
                      and index not in algorithm.rejected_constraint_indices]
            assert actual and np.mean(actual) <= 0.
            assert algorithm.objective_research.state["pending"] is None
            assert len(algorithm.objective_research.state["requests"]) == request_count
            nominations[len(algorithm.history)] = {"point": point, "request_count": request_count,
                                                    "scored": False}
        return offers

    def native_scores(candidates, terminal_pool, **kwargs):
        check = nominations.get(len(algorithm.history))
        if check is not None:
            assert check["point"] in candidates
            assert {point for point, _ in algorithm.history}.issubset(set(terminal_pool))
            assert algorithm.objective_research.state["pending"] is None
            margin_before = float(certificate(check["point"])["margin"][0])
        scores = original_scores(candidates, terminal_pool, **kwargs)
        if check is not None:
            check["scored"] = True
            # Nomination and native fantasies confer no actual certificate.
            assert certificate(check["point"])["margin"][0] == margin_before > 0.
        return scores

    def actual_measure(point, rng=None):
        check = nominations.get(len(algorithm.history))
        if check is not None and tuple(point) == check["point"]:
            pending = algorithm.objective_research.state["pending"]
            assert pending["reason"] == reason
            assert len(algorithm.objective_research.state["requests"]) == check["request_count"] + 1
            purchased.append(tuple(point))
        return original_measure(point, rng)

    algorithm.objective_research.terminal_offers = actual_offers
    algorithm._exact_posterior_update_scores = native_scores
    adapter.simulate = actual_measure
    result = algorithm.run()
    assert nominations and all(row["scored"] for row in nominations.values()), algorithm.objective_diagnostics["terminal_candidate_sets"]
    assert purchased, {"nominations": nominations, "choices": algorithm.objective_diagnostics["terminal_choices"]}
    assert certificate(tuple(result["x_recommended"]))["margin"][0] <= 0.
    assert algorithm.objective_research.state["pending"] is None
    assert len(algorithm.history) == result["n_simulations"] == 64
    assert adapter.research_simulator_calls == adapter.simulator_calls == 32


def test_candidate_competition_predicts_every_quantized_point_before_real_simulation(project_root, monkeypatch):
    environment, config = _adoption_setup(project_root, seed=841000081, total=36)
    first = Program("feature", name="x1")
    second = Program("mul", children=(Program("feature", name="x1"),) * 2)
    forecasts = []
    original_predict = Program.predict

    def record_prediction(program, observation, context=None):
        result = original_predict(program, observation, context)
        if program is first or program is second:
            forecasts.append((program, dict(observation.features), result))
        return result

    monkeypatch.setattr(Program, "predict", record_prediction)

    class MeasuredCompetition:
        state = {"phase": "collecting"}
        candidates = {"linear": first, "square": second}
        candidate = first
        accepted_revision = None
        control_purpose = "coverage"

        def next_control(self):
            return {name: .254 for name in environment.spec.feature_names}

        def observe(self, observation, default_mean, learned_mean, *, candidate_mean,
                    candidate_predictions, noise_variance, context):
            assert candidate_predictions == {"linear": .25, "square": .0625}
            assert candidate_mean == .25
            assert set(observation.features.values()) == {.25}
            assert len(forecasts) == 2 * adapter.simulator_calls

    adapter = compile_sc_kg_program(environment, Program("feature", name="x1"))
    original_simulate = adapter.simulate

    def check_before_measurement(point, rng=None):
        assert len(forecasts) == 2 * (adapter.simulator_calls + 1)
        assert tuple(point) == (25,) * 5
        return original_simulate(point, rng)

    adapter.simulate = check_before_measurement
    algorithm = initialize_sc_kg_optimizer(adapter, config, recovery=MeasuredCompetition())
    result = algorithm.run()
    assert result["n_simulations"] == len(algorithm.history) == 36
    assert adapter.research_simulator_calls == 32 and adapter.simulator_calls == 4
    assert algorithm.recovery_diagnostics["candidate_prediction_rows"] == len(forecasts) == 8
    assert all(event["candidate_predictions"] == {"linear": .25, "square": .0625}
               for event in algorithm.adoption_events)


def test_actual_noisy_repeat_rejects_one_off_error_using_frozen_predictions(project_root):
    from hegel_machine.cognition.joint_model_revision import JointModelRevision
    from hegel_machine.cognition.prior_adoption import PriorAdoption

    environment, config = _adoption_setup(project_root, seed=841000091, total=48)
    adapter = compile_sc_kg_program(environment, Program("feature", name="x1"))
    recovery = JointModelRevision(environment.spec, seed=841000091)
    original_simulate = adapter.simulate
    measurements = []

    def one_off_instrument_error(point, rng=None):
        measured = original_simulate(point, rng)
        measurements.append({"point": tuple(point), "output": measured.copy()})
        # Four genuine new predictions establish the controller's reference.
        # The fifth real instrument reading alone is corrupted; later calls
        # still execute the original noisy simulator and preserve its RNG.
        if adapter.simulator_calls == 5:
            measured[1] += 1.
        return measured

    adapter.simulate = one_off_instrument_error
    algorithm = initialize_sc_kg_optimizer(adapter, config,
        adoption_policy=PriorAdoption(), recovery=recovery)
    result = algorithm.run()
    assert result["n_simulations"] == len(algorithm.history) == 48
    assert adapter.loaded_observation_count == adapter.research_simulator_calls == 32
    assert adapter.simulator_calls == len(measurements) == 16
    state = recovery.to_dict()
    confirmation = state["confirmations"][0]
    assert confirmation["anchor_index"] == 4 and confirmation["observation_index"] == 5
    assert all(not item["confirmed"] for item in state["confirmations"])
    assert not state["episodes"]
    anchor = state["observations"][4]
    repeat = state["observations"][5]
    assert anchor["context"]["purpose"] in {"optimization", "objective_exploration"}
    assert repeat["context"]["purpose"] == "confirmation"
    assert anchor["observation"]["features"] == repeat["observation"]["features"]
    assert measurements[4]["point"] == measurements[5]["point"]
    assert measurements[4]["output"][1] != measurements[5]["output"][1]
    assert confirmation["frozen_predictions"] == {
        name: anchor[f"{name}_mean"] for name in ("default", "learned")}
    assert repeat["confirmation_predictions"] == confirmation["frozen_predictions"]
    # The GP assimilated the outlier before the repeat. Testing its updated
    # residual would incorrectly treat the good repeat as another failure.
    assert repeat["default_error"] > confirmation["threshold"]
    assert confirmation["repeat_errors"]["default"] < confirmation["threshold"]
    assert any(abs(repeat[f"{name}_mean"] - confirmation["frozen_predictions"][name]) > 1e-3
               for name in ("default", "learned"))
    # Subsequent noisy controls may independently need confirmation; they must
    # be paid and accounted for without producing another rejected observation.
    confirmation_count = len(state["confirmations"])
    assert confirmation_count == state["costs"]["confirmation_calls"]
    assert state["costs"]["coverage_calls"] == state["costs"]["validation_calls"] == 0
    assert algorithm.recovery_diagnostics["confirmation_requests"] == confirmation_count
    assert algorithm.iteration_log[5]["candidate_source_selected"] == "revision_confirmation"
    assert sum(row["context"]["purpose"] == "confirmation"
               for row in algorithm.adoption_policy.to_dict()["observations"]) == confirmation_count
    assert sum(row["candidate_source_selected"] == "revision_confirmation"
               for row in algorithm.iteration_log) == confirmation_count
    assert algorithm.rejected_constraint_indices == {36}
    assert len(algorithm.observation_repairs) == 1
    assert len(state["observation_repairs"]) == 1
    assert sum(map(len, algorithm.observations.values())) == 48
    assert algorithm.problem.basis.program == Program("feature", name="x1")
    assert algorithm.active_scope_start == 0 and not algorithm.scope_revisions


def test_observation_repair_restores_real_posteriors_and_filters_future_replicates(project_root):
    from copy import deepcopy
    import numpy as np
    from hegel_machine.cognition.joint_model_revision import JointModelRevision

    environment, config = _adoption_setup(project_root, seed=841000091, total=48)
    adapter = compile_sc_kg_program(environment, Program("feature", name="x1"))
    recovery = JointModelRevision(environment.spec, seed=841000091)
    algorithm = initialize_sc_kg_optimizer(adapter, config, recovery=recovery)
    original_simulate = adapter.simulate
    original_repair = algorithm._repair_constraint_observation
    original_candidates = algorithm._generate_candidates
    checks, revisits = [], []

    def one_bad_output(point, rng=None):
        measured = original_simulate(point, rng)
        if adapter.simulator_calls == 5:
            measured[1] += 1.
        return measured

    def check_repair(repair, pending, stage):
        saved = algorithm._confirmation_predictors
        expected = {name: algorithm._clone_gpr_for_exact_kg(model)
                    for name, model in saved["models"].items()}
        for name, model in expected.items():
            sigma2 = saved["variances"][name].predict_variance(1, pending["point"], adapter)
            model.update(pending["point"], pending["output"][1], sigma2)
        objective = algorithm.gpr[0]
        objective_state = (objective.a.copy(), objective.C.copy())
        objective_variances = {name: deepcopy(variance.records[0])
                               for name, variance in algorithm.constraint_variances.items()}
        original_repair(repair, pending, stage)
        assert algorithm.gpr[0] is objective
        np.testing.assert_array_equal(objective.a, objective_state[0])
        np.testing.assert_array_equal(objective.C, objective_state[1])
        for name, model in algorithm.constraint_models.items():
            np.testing.assert_allclose(model.a, expected[name].a)
            np.testing.assert_allclose(model.C, expected[name].C)
            assert algorithm.constraint_variances[name].records[0] == objective_variances[name]
        checks.append(stage)

    def repeat_after_repair(iteration):
        if algorithm.observation_repairs and not revisits:
            point = algorithm.history[36][0]
            revisits.append(point)
            algorithm._recovery_control_purpose = "optimization"
            return [point], {point: "test_retained_control"}
        return original_candidates(iteration)

    adapter.simulate = one_bad_output
    algorithm._repair_constraint_observation = check_repair
    algorithm._generate_candidates = repeat_after_repair
    result = algorithm.run()
    assert checks == [38] and len(revisits) == 1
    assert result["n_simulations"] == len(algorithm.history) == 48
    assert algorithm.recovery_diagnostics["filtered_replicate_updates"] >= 1
    point = revisits[0]
    retained = [float(output[1]) for index, (control, output) in enumerate(algorithm.history)
                if control == point and index != 36]
    assert algorithm._constraint_values(point) == retained
    assert len(algorithm.observations[point]) == len(retained) + 1
    assert algorithm.history[36][1][1] == recovery.to_dict()["observations"][4]["observation"]["target"]
    for variance in algorithm.constraint_variances.values():
        actual = [residual for control, residual in variance.records[1] if control == point]
        assert actual == pytest.approx([max(float(np.var(retained, ddof=1)), variance.floor)])

    # The real exact-MC acquisition must use the same retained replicate values
    # in its hypothetical posterior; otherwise the rejected value returns as
    # large predicted measurement noise on this next experiment.
    means = [model.posterior_mean(point) for model in algorithm.gpr]
    epistemic = [model.posterior_var(point) for model in algorithm.gpr]
    noise = [algorithm.variance_model.predict_variance(i, point, adapter) for i in range(2)]
    fantasy_y = [means[0] + .01, means[1] + .01]
    _, fantasy = algorithm._exact_primary_fantasy_update(np.asarray(point), fantasy_y,
        algorithm.observations[point], means, noise, epistemic)
    future_variance = max(float(np.var(retained + [fantasy_y[1]], ddof=1)), fantasy.floor)
    assert [value for control, value in fantasy.records[1] if control == point] == pytest.approx([future_variance])
    assert algorithm.recovery_diagnostics["filtered_fantasy_updates"] >= 1
    objective_values = [float(output[0]) for output in algorithm.observations[point]] + [fantasy_y[0]]
    assert [value for control, value in fantasy.records[0] if control == point] == pytest.approx([
        max(float(np.var(objective_values, ddof=1)), fantasy.floor)])

    # Legacy recommendation routines use empirical control means. Their input
    # view retains the mean of both real objective draws, while constraint
    # replication count and mean exclude only the disputed output.
    view = algorithm._constraint_decision_view()
    assert np.mean([row[0] for row in view.observations[point]]) == pytest.approx(
        np.mean([row[0] for row in algorithm.observations[point]]))
    assert [row[1] for row in view.observations[point]] == retained
    assert len(view.observations[point]) + 1 == len(algorithm.observations[point])

    # A later valid revision can select a segment spanning this disputed row.
    # Refitting that scope must not make its rejected constraint trusted again.
    raw_history = deepcopy(algorithm.history)
    objective = algorithm.gpr[0]
    objective_state = (objective.a.copy(), objective.C.copy())
    objective_variances = {name: deepcopy(variance.records[0])
                           for name, variance in algorithm.constraint_variances.items()}
    acknowledgments = []
    recovery.acknowledge_revision = lambda: acknowledgments.append(True)
    algorithm._rebase_scope({"scope_start": 32, "program": adapter.basis.program}, stage=48)
    assert acknowledgments == [True]
    assert algorithm.scope_revisions[-1]["scope_records"] == 16
    assert algorithm.scope_revisions[-1]["constraint_scope_records"] == 15
    assert algorithm.scope_revisions[-1]["preserved_objective_head"] is True
    assert algorithm.recovery_diagnostics["scope_fit_records"] == 2 * 15
    assert algorithm.gpr[0] is objective
    np.testing.assert_array_equal(objective.a, objective_state[0])
    np.testing.assert_array_equal(objective.C, objective_state[1])
    for name, variance in algorithm.constraint_variances.items():
        assert variance.records[0] == objective_variances[name]
        assert len(variance.records[1]) == 15
    for (actual_point, actual), (saved_point, saved) in zip(algorithm.history, raw_history):
        assert actual_point == saved_point
        np.testing.assert_array_equal(actual, saved)
    assert sum(map(len, algorithm.observations.values())) == 16
    assert len(algorithm.history) == adapter.research_simulator_calls + adapter.simulator_calls == 48


@pytest.mark.parametrize("constraint_pending", [False, True])
def test_independent_objective_queries_share_the_paid_loop_and_yield_to_constraint_research(
        project_root, monkeypatch, constraint_pending):
    from hegel_machine.cognition import objective_research
    from hegel_machine.cognition.joint_model_revision import JointModelRevision

    environment, config = _adoption_setup(project_root, seed=841000111, total=34)
    observed, planning = [], []

    class GoalResearch:
        accepted_revision = None

        def __init__(self, spec):
            self.spec = spec
            self.state = {"scope_research": None, "pending": None}

        def prepare_shared_control(self, rows, forecast, features, **context):
            assert len(rows) == len(algorithm.history) == context["stage"]
            assert len(observed) == len(rows) - 32
            if context["purpose"] == "optimization":
                assert not constraint_pending and self.state["scope_research"] is None
                return None
            assert constraint_pending
            assert context["purpose"] == "confirmation"
            assert set(features.values()) == {.75}
            return None

        def next_control(self, rows, forecast, **context):
            assert not constraint_pending
            assert len(rows) == context["stage"] == 32
            assert context["remaining_calls"] == 2
            assert context["scope_start"] == 0
            assert rows[0]["context"]["objective"] == algorithm.history[0][1][0]
            actual = forecast({name: .254 for name in self.spec.feature_names})
            assert set(actual["features"].values()) == {.25}
            assert actual["feasible"] == (actual["constraint_margin"] <= 0.)
            planning.append(actual)
            return {"features": actual["features"], "purpose": "objective_exploration",
                    "reason": "measured_goal_opportunity"}

        def observe(self, features, output, *, context):
            observed.append((features, output, context))
            # Only the first paid query is prescribed; the second follows the
            # ordinary SC proposal path without restarting mechanism research.
            self.next_control = lambda *args, **kwargs: None

    class PendingConstraintResearch:
        candidates = {}
        candidate = accepted_revision = None
        control_purpose = "confirmation"
        state = {"phase": "confirming"}

        def next_control(self):
            return {name: .754 for name in environment.spec.feature_names}

        def observe(self, *args, **kwargs):
            pass

    monkeypatch.setattr(objective_research, "ObjectiveResearch", GoalResearch)
    adapter = compile_sc_kg_program(environment, Program("feature", name="x1"))
    recovery = (PendingConstraintResearch() if constraint_pending else
                JointModelRevision(environment.spec, seed=841000111))
    algorithm = initialize_sc_kg_optimizer(adapter, config, recovery=recovery)
    result = algorithm.run()
    assert result["n_simulations"] == len(algorithm.history) == 34
    assert adapter.research_simulator_calls == 32 and adapter.simulator_calls == 2
    assert len(observed) == algorithm.objective_diagnostics["observations"] == 2
    first = observed[0]
    assert first[1] == list(algorithm.history[32][1])
    assert first[2]["stage"] == 33 and first[2]["global_index"] == 32
    assert not algorithm.scope_revisions
    if constraint_pending:
        assert not planning
        assert set(first[0].values()) == {.75}
        assert first[2]["purpose"] == "confirmation"
        assert algorithm.recovery_diagnostics["confirmation_requests"] == 2
    else:
        assert len(planning) == 1
        assert set(first[0].values()) == {.25}
        assert first[2]["purpose"] == "objective_exploration"
        assert algorithm.objective_diagnostics["objective_exploration_requests"] == 1
        assert algorithm.objective_diagnostics["forecast_rows"] == 1
        for key in ("mean_prediction_rows", "epistemic_prediction_rows", "noise_prediction_rows"):
            assert algorithm.objective_diagnostics[key] == 2
        assert not recovery.to_dict()["episodes"]


def test_terminal_gp_ranking_is_not_overridden_by_one_low_objective_reading(project_root, monkeypatch):
    import numpy as np
    from hegel_machine.cognition.joint_model_revision import JointModelRevision

    environment, config = _adoption_setup(project_root, seed=841000121, total=32)
    adapter = compile_sc_kg_program(environment, Program("feature", name="x1"))
    algorithm = initialize_sc_kg_optimizer(adapter, config,
        recovery=JointModelRevision(environment.spec, seed=841000121))
    algorithm._fit_initial_belief(algorithm._initial_samples())
    low_reading, posterior_best = (25,) * 5, (75,) * 5
    # A supported noisy observation disagrees with the cross-control posterior.
    # Both controls have equally supported, feasible constraint observations.
    algorithm.observations = {low_reading: [np.asarray([-.05, -.4])],
                              posterior_best: [np.asarray([.5, -.4])]}
    monkeypatch.setattr(algorithm.gpr[0], "posterior_mean_many", lambda points:
                        np.asarray([.7 if tuple(point) == low_reading else .5 for point in points]))
    monkeypatch.setattr(algorithm.gpr[1], "posterior_mean_many", lambda points: np.full(len(points), -.4))
    monkeypatch.setattr(algorithm, "_constraint_certification_epistemic_many",
                        lambda model, points: np.full(len(points), 1e-10))
    monkeypatch.setattr(algorithm.variance_model, "predict_certification_variance_many",
                        lambda output, points, problem: np.full(len(points), 1e-6))
    researcher = algorithm.objective_research
    algorithm.objective_research = None
    legacy, legacy_details = algorithm._solve_posterior_recommendation(pool=[low_reading, posterior_best])
    algorithm.objective_research = researcher
    selected, details = algorithm._solve_posterior_recommendation(pool=[low_reading, posterior_best])
    assert legacy == low_reading and legacy_details["observed_incumbent_used"]
    assert selected == posterior_best and not details["observed_incumbent_used"]
    assert algorithm.observations[low_reading][0][0] == -.05


def test_real_objective_only_change_validates_and_rebuilds_only_the_goal_head(project_root):
    from copy import deepcopy
    import numpy as np
    from hegel_machine.cognition.joint_model_revision import JointModelRevision
    environment = SCKGEnvironment(project_root, problem="rzdt2", dimension=3, seed=841000131)
    from algorithms.single_olhkg import SingleOLHKGConfig
    simulate = environment._problem.simulate
    calls = []

    def instrument(point, rng=None):
        vector = np.asarray(simulate(point, rng), dtype=float)
        x1, x2 = point[0] / 100., point[1] / 100.
        value = .5 + x1 if len(calls) < 44 else .5 + (1 - x1) ** 2 + .5 * (1 - x2)
        vector[:2] = value
        calls.append((tuple(point), vector.copy()))
        return vector

    environment._problem.simulate = instrument
    for probe in (*environment.spec.initial_probes, *environment.spec.available_probes[:2]):
        environment.observe(probe)
    config = SingleOLHKGConfig(N=64, K1=3, K2=0, posterior_pool_size=8, posterior_keep=1,
        axis_candidate_count=0, state_candidate_count=0, use_state_coupling=False,
        use_state_basis=False, lambda_coupling=0., variance_mode="pooled",
        exact_kg_mc_samples=1, eval_pool_size=8, evaluate_interval=0,
        truth_pool_diagnostics=False, checkpoint_interval=0,
        use_recommendation_refinement=False, recommendation_axis_oracle=False, seed=841000132)
    adapter = compile_sc_kg_program(environment, Program("feature", name="x1"))
    recovery = JointModelRevision(environment.spec, seed=841000131)
    algorithm = initialize_sc_kg_optimizer(adapter, config, recovery=recovery)
    original_rebase = algorithm._rebase_objective_scope
    checks = []
    terminal_rounds = {}
    original_offers = algorithm.objective_research.terminal_offers
    original_scores = algorithm._exact_posterior_update_scores
    original_measure = adapter.simulate

    def three_actual_actions(rows, forecast, **context):
        offered = original_offers(rows, forecast, **context)
        if algorithm.objective_research.state["scope_research"] is not None:
            return offered
        # Fix only the proposal list for this routing test. The installed goal
        # head earned its real four future observations, and all GP/HVD, KG
        # scores, selection and subsequent simulator outputs remain real.
        observed = list(dict.fromkeys(point for point, output in
            algorithm.history[algorithm.active_objective_scope_start:] if output[1] <= 0.))
        assert len(observed) >= 2
        paid = {point for point, _ in algorithm.history}
        new = next((100, 100, coordinate) for coordinate in range(100, -1, -1)
                   if (100, 100, coordinate) not in paid)
        points = [new, *observed[:2]]
        offers = [{"features": dict(zip(environment.spec.feature_names, (value / 100. for value in point))),
                   "purpose": "objective_exploration" if index == 0 else "objective_confirmation",
                   "reason": "integration_terminal_alternative", "stage": context["stage"],
                   "scope_start": context["scope_start"]}
                  for index, point in enumerate(points)]
        terminal_rounds[len(algorithm.history)] = {"points": points, "paid": paid,
            "request_count": len(algorithm.objective_research.state["requests"]), "score_calls": 0}
        assert algorithm.objective_research.state["pending"] is None
        return offers

    def score_actual_actions(candidates, terminal_pool, *args, **kwargs):
        check = terminal_rounds.get(len(algorithm.history))
        if check is not None:
            assert list(candidates) == check["points"]
            assert len(candidates) == 3
            assert candidates[0] not in check["paid"] and all(point in check["paid"] for point in candidates[1:])
            assert check["paid"].issubset(set(terminal_pool)) and len(terminal_pool) > 3
            assert algorithm.objective_research.state["pending"] is None
            assert len(algorithm.objective_research.state["requests"]) == check["request_count"]
            check["score_calls"] += 1
        return original_scores(candidates, terminal_pool, *args, **kwargs)

    def measure_selected(point, rng=None):
        check = terminal_rounds.get(len(algorithm.history))
        if check is not None:
            pending = algorithm.objective_research.state["pending"]
            assert tuple(point) in check["points"] and pending is not None
            assert environment.integer_point(pending["features"]) == tuple(point)
            assert len(algorithm.objective_research.state["requests"]) == check["request_count"] + 1
        return original_measure(point, rng)

    algorithm.objective_research.terminal_offers = three_actual_actions
    algorithm._exact_posterior_update_scores = score_actual_actions
    adapter.simulate = measure_selected

    def apply_goal_only(revision, stage):
        models = dict(algorithm.constraint_models)
        state = {name: (model.a.copy(), model.C.copy()) for name, model in models.items()}
        variances = {name: deepcopy(value.records[1]) for name, value in algorithm.constraint_variances.items()}
        prior, scope = adapter.basis.program, algorithm.active_scope_start
        original_rebase(revision, stage)
        for name, model in algorithm.constraint_models.items():
            assert model is models[name]
            np.testing.assert_array_equal(model.a, state[name][0])
            np.testing.assert_array_equal(model.C, state[name][1])
            assert algorithm.constraint_variances[name].records[1] == variances[name]
        assert adapter.basis.program == prior and algorithm.active_scope_start == scope
        checks.append(stage)

    algorithm._rebase_objective_scope = apply_goal_only
    result = algorithm.run()
    assert result["n_simulations"] == len(algorithm.history) == len(calls) == 64
    assert adapter.loaded_observation_count == 32 and adapter.simulator_calls == 32
    assert len(checks) == len(algorithm.objective_scope_revisions) == 1
    revision = algorithm.objective_scope_revisions[0]
    assert revision["scope_start"] >= 44 and algorithm.active_objective_scope_start == revision["scope_start"]
    assert len(revision["validation_indices"]) == 4
    assert set(revision["validation_indices"]).isdisjoint(revision["training_indices"])
    assert revision["validation_mse"] < revision["previous_mse"]
    assert not algorithm.scope_revisions and not recovery.to_dict()["episodes"]
    assert algorithm.objective_diagnostics["head_refits"] == 1
    assert algorithm.objective_diagnostics["head_fit_records"] == len(revision["fit_indices"])
    assert algorithm.objective_research.to_dict()["compute_cost"] <= 20000
    assert set(terminal_rounds) == {60, 61, 62, 63}
    assert all(item["score_calls"] == 1 for item in terminal_rounds.values())
    assert algorithm.objective_diagnostics["terminal_rounds"] == 4
    assert algorithm.objective_diagnostics["terminal_candidates"] == 12
    assert algorithm.objective_diagnostics["terminal_selected_offers"] == 4
    assert len(algorithm.objective_diagnostics["terminal_choices"]) == 4
    for row in algorithm.iteration_log[-4:]:
        assert row["exact_kg_active_action_count"] == row["n_candidates"] == 3
        assert row["terminal_pool_size"] > 3 and row["acquisition_mode"] == "exact_mc"
    assert algorithm.objective_research.state["pending"] is None
    assert not algorithm.objective_research.state["confirmation_queue"]

    # An independently validated goal head keeps its complete state when a
    # constraint revision selects either an earlier or a later evidence scope.
    from hegel_machine.cognition.sc_kg_environment import HegelProgramBasis
    goal_start = algorithm.active_objective_scope_start
    goal_program = Program.from_dict(revision["program"])
    goal = algorithm.gpr[0]
    coefficients, covariance = goal.a.copy(), goal.C.copy()
    goal_variances = {name: {key: deepcopy(values[0]) for key, values in variance.__dict__.items()
                            if isinstance(values, dict) and 0 in values}
                      for name, variance in algorithm.constraint_variances.items()}
    assert isinstance(algorithm.gpr[0].basis_map, HegelProgramBasis)
    assert algorithm.gpr[0].basis_map.program == goal_program
    assert algorithm.gpr[0].p == 2
    recovery.acknowledge_revision = lambda: None
    for constraint_start in (40, 47):
        old_constraints = dict(algorithm.constraint_models)
        old_fits = algorithm.recovery_diagnostics["scope_model_fits"]
        old_rows = algorithm.recovery_diagnostics["scope_fit_records"]
        algorithm._rebase_scope({"scope_start": constraint_start, "program": adapter.basis.program}, stage=64)
        assert algorithm.gpr[0] is goal
        np.testing.assert_array_equal(goal.a, coefficients)
        np.testing.assert_array_equal(goal.C, covariance)
        assert goal.basis_map.program == goal_program and goal.p == 2
        assert algorithm.active_objective_scope_start == goal_start
        assert algorithm.active_scope_start == constraint_start
        assert algorithm.scope_revisions[-1]["preserved_objective_head"]
        assert algorithm.scope_revisions[-1]["objective_scope_start"] == goal_start
        assert algorithm.scope_revisions[-1]["objective_scope_records"] == 64 - goal_start
        assert algorithm.scope_revisions[-1]["constraint_scope_records"] == 64 - constraint_start
        assert algorithm.recovery_diagnostics["scope_model_fits"] - old_fits == 2
        assert algorithm.recovery_diagnostics["scope_fit_records"] - old_rows == 2 * (64 - constraint_start)
        for name, variance in algorithm.constraint_variances.items():
            assert algorithm.constraint_models[name] is not old_constraints[name]
            output_state = {key: values[0] for key, values in variance.__dict__.items()
                            if isinstance(values, dict) and 0 in values}
            np.testing.assert_equal(output_state, goal_variances[name])
            assert len(variance.records[1]) == 64 - constraint_start
    controls = [(0, 0, 0), (25, 50, 75), (100, 100, 0)]
    expected = [.5 + (1 - point[0] / 100.) ** 2 + .5 * (1 - point[1] / 100.)
                for point in controls]
    np.testing.assert_allclose(algorithm.gpr[0].posterior_mean_many(controls), expected,
                               atol=.03, rtol=0.)
    # The active SC view now omits valid goal observations at indices 44--46.
    # Fantasy replication must still include those actual objective labels.
    point = algorithm.history[goal_start][0]
    assert algorithm._has_previous_objective(point)
    means = [float(model.posterior_mean(point)) for model in algorithm.gpr]
    epistemic = [float(model.posterior_var(point)) for model in algorithm.gpr]
    noise = [float(algorithm.variance_model.predict_variance(i, point, adapter)) for i in range(2)]
    output = np.asarray([means[0] + .3, means[1]])
    _, fantasy = algorithm._exact_primary_fantasy_update(np.asarray(point), output,
        algorithm.observations.get(point, []), means, noise, epistemic)
    values = algorithm._objective_values(point) + [float(output[0])]
    expected_variance = max(float(np.var(values, ddof=1)), fantasy.floor)
    assert [value for control, value in fantasy.records[0] if control == point] == pytest.approx([expected_variance])
    assert len(algorithm.history) == len(calls) == 64


def test_actual_measured_objective_changes_revision_queries_with_the_same_budget(project_root):
    from dataclasses import replace
    from importlib.util import module_from_spec, spec_from_file_location
    import numpy as np
    from hegel_machine.cognition.joint_model_revision import JointModelRevision
    from hegel_machine.cognition.memory import KnowledgeLibrary
    from hegel_machine.cognition.meta_prior_research import MetaPriorResearch
    from hegel_machine.cognition.models import ResearchConfig

    seed = 1043010067
    environment = SCKGEnvironment(project_root, problem="state_policy_rzdt1", dimension=3, seed=seed)
    report = MetaPriorResearch(ResearchConfig(observation_budget=32, compute_budget=60000,
        max_rounds=2, minimum_improvement=1e-7, seed=seed), KnowledgeLibrary()).run(environment)
    from algorithms.single_olhkg import SingleOLHKGConfig
    settings = json.loads((Path(__file__).resolve().parents[1]
                          / "config/cognition_sc_kg_optimizer.json").read_text())
    config = replace(SingleOLHKGConfig(**settings), N=64, K1=6, posterior_pool_size=32,
        posterior_keep=3, eval_pool_size=32, seed=seed + 4000009)
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_sc_kg_prior_mechanism_change.py"
    module_spec = spec_from_file_location("hegel_goal_change_case", script)
    module = module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    class PublicObjectiveProblem(module.MechanismChangeProblem):
        def __init__(self, sign):
            super().__init__(environment, Program.from_dict(report["program"]), change_after=12)
            self.sign = sign

        def simulate(self, point, rng=None):
            output = super().simulate(point, rng)
            if self.simulator_calls > self.change_after:
                output[0] *= self.sign
            return output

        def true_objective(self, point):
            value = self.base.true_objective(point)
            return value * (self.sign if self.simulator_calls > self.change_after else 1.)

        def true_outputs(self, point):
            return np.asarray([self.true_objective(point), self.base.true_constraint_mean(point)])

    runs = []
    for sign in (1., -1.):
        adapter = PublicObjectiveProblem(sign)
        recovery = JointModelRevision(environment.spec, seed=seed)
        algorithm = initialize_sc_kg_optimizer(adapter, config, recovery=recovery)
        result = algorithm.run()
        assert result["n_simulations"] == len(algorithm.history) == 64
        assert adapter.research_simulator_calls == adapter.simulator_calls == 32
        assert len(algorithm.scope_revisions) == 1
        episode = recovery.to_dict()["episodes"][0]
        assert episode["accepted_revision"]["scope_start"] == 44
        assert len(episode["accepted_revision"]["validation_indices"]) == 4
        proposal = episode["proposals"][0]
        plan = proposal["experiment_plan"]
        assert plan["training_indices"] == proposal["training_indices"]
        assert plan["objective_model"]["training_rows"] == len(plan["training_indices"])
        assert set(plan["training_indices"]).isdisjoint(episode["accepted_revision"]["validation_indices"])
        assert 0 < plan["compute_cost"] < recovery.to_dict()["costs"]["compute_cost"]
        assert [item["purpose"] for item in plan["selected"]] == [
            "objective_improvement", "objective_coverage",
            "hypothesis_disagreement", "hypothesis_disagreement"]
        validation_points = [row["x_selected"] for row in algorithm.iteration_log
                             if row["candidate_source_selected"] == "revision_validation"]
        assert [tuple(point) for point in validation_points] == [
            environment.integer_point(item["point"]) for item in plan["selected"]]
        assert len(validation_points) == recovery.to_dict()["costs"]["validation_calls"] == 4
        runs.append((algorithm, episode, validation_points))
    first, second = runs
    # Both arms see the same paid controls and constraint data up to the
    # validation plan; only the public measured objective is reversed.
    first_proposal, second_proposal = (run[1]["proposals"][0] for run in runs)
    assert first_proposal["competitors"] == second_proposal["competitors"]
    first_indices = first_proposal["training_indices"]
    assert first_indices == second_proposal["training_indices"]
    training_end = 32 + max(first_indices) + 1
    assert [(row[0], row[1][1]) for row in first[0].history[:training_end]] == [
        (row[0], row[1][1]) for row in second[0].history[:training_end]]
    for index in first_indices:
        first_row = first[0].recovery.to_dict()["observations"][index]
        second_row = second[0].recovery.to_dict()["observations"][index]
        assert first_row["observation"] == second_row["observation"]
        assert first_row["context"]["objective"] == -second_row["context"]["objective"]
    assert first[2] != second[2]


def _saved_terminal_failure_optimizer(project_root, monkeypatch, case_id,
        *, frozen_directory="sc_kg_terminal_value_v040_20260910"):
    from hegel_machine.cognition.joint_model_revision import JointModelRevision
    from hegel_machine.cognition.prior_adoption import PriorAdoption
    from hegel_machine.cognition.sc_kg_environment import HegelProgramProblem

    root = Path(__file__).resolve().parents[1]
    frozen = root / "artifacts" / frozen_directory
    if not (frozen / "protocol.json").exists():
        pytest.skip("the saved real terminal MC1 failure is unavailable")
    spec = json.loads((frozen / "protocol.json").read_text())
    case = next(item for item in spec["cases"] if item["case_id"] == case_id)
    source = json.loads((frozen / "research" / f"{case['research_id']}.json").read_text())
    monkeypatch.syspath_prepend(str(root / "scripts"))
    from run_sc_kg_objective_exploration_study import controlled_problem
    environment = SCKGEnvironment(project_root, **case["task"])
    environment.restore(source["optimizer_initial_environment_state"])
    adapter = controlled_problem(HegelProgramProblem, environment,
        Program.from_dict(source["result"]["program"]), case)
    from algorithms.single_olhkg import SingleOLHKGConfig
    config = SingleOLHKGConfig(N=64, seed=case["task"]["seed"] + spec["optimizer_seed_offset"],
                              **spec["optimizer_settings"])
    recovery = JointModelRevision(environment.spec,
        minimum_error=spec["revision"]["minimum_error"], batch_size=spec["revision"]["batch_size"],
        seed=case["task"]["seed"] + spec["revision"]["seed_offset"], compute_budget=20000)
    algorithm = initialize_sc_kg_optimizer(adapter, config,
        adoption_policy=PriorAdoption(), recovery=recovery)
    return algorithm, adapter, environment


def test_real_negative_mc1_terminal_comparison_uses_native_fantasies_and_keeps_measurement_rng(project_root, monkeypatch):
    """Replay the existing all-negative failure; no GP or acquisition values are mocked."""
    from copy import deepcopy
    import numpy as np
    algorithm, adapter, _ = _saved_terminal_failure_optimizer(project_root, monkeypatch,
        "research_scope_3_objective_only_change_noise_0.03")
    original_scores = algorithm._exact_posterior_update_scores
    original_one = algorithm._exact_posterior_update_score_one
    original_plan = algorithm._exact_kg_sample_plan
    plans, extra_calls, checked = [], [], []
    scoring = {}

    def sample_plan(count):
        plans.append(count)
        return original_plan(count)

    def native_draw(point, innovations, pool, current, *args, **kwargs):
        if algorithm._value_precision_round is not None:
            assert pool is scoring["pool"]
            assert current == algorithm._last_exact_kg_current_value
            assert len(innovations) == 1
            extra_calls.append((len(algorithm.history) + 1, tuple(point)))
        return original_one(point, innovations, pool, current, *args, **kwargs)

    def exact_scores(candidates, terminal_pool, **kwargs):
        scoring["pool"] = terminal_pool
        plan_count = len(plans)
        expected_rng = np.random.default_rng()
        expected_rng.bit_generator.state = deepcopy(algorithm.rng.bit_generator.state)
        expected_rng.standard_normal((1, 2))
        expected_rng.random(1)
        parameters = [(model.a.copy(), model.C.copy()) for model in algorithm.gpr]
        variance = {key: deepcopy(value) for key, value in algorithm.variance_model.__dict__.items()
                    if isinstance(value, dict) and 0 in value and 1 in value}
        history = deepcopy(algorithm.history)
        previous = len(algorithm.objective_diagnostics["value_precision"]["rounds"])
        result = original_scores(candidates, terminal_pool, **kwargs)
        assert len(plans) == plan_count + 1 and plans[-1] == 1
        assert algorithm.rng.bit_generator.state == expected_rng.bit_generator.state
        rounds = algorithm.objective_diagnostics["value_precision"]["rounds"]
        if len(rounds) > previous:
            event = rounds[-1]
            assert event["stage"] >= 61 and algorithm.objective_scope_revisions
            assert len(algorithm._terminal_goal_offers) >= 1
            assert {point for point, _ in history}.issubset(set(terminal_pool))
            assert max(event["initial_raw"]) < 0. and set(event["initial_clipped"]) == {0.}
            assert event["selection_semantics"] == "policy_indicator"
            assert list(candidates[int(np.argmax(result))]) == event["selected_point"]
            by_point = {tuple(row["point"]): row for row in event["action_statistics"]}
            estimates = [by_point[tuple(point)]["mean_raw"] for point in candidates]
            np.testing.assert_allclose(algorithm._last_exact_kg_raw_scores, estimates)
            np.testing.assert_allclose(algorithm._last_exact_kg_expected_values,
                                       event["current_terminal_value"] - np.asarray(estimates))
            assert event["selected_raw_value"] == by_point[tuple(event["selected_point"])]["mean_raw"]
            assert event["evaluations"] == event["score_one_calls"] == event["terminal_value_calls"]
            assert event["terminal_pool_rows"] == event["evaluations"] * len(terminal_pool)
            assert event["evaluations"] <= event["allowance"]
            assert 8 <= event["draws_per_action"] <= 16
            for model, (mean, covariance) in zip(algorithm.gpr, parameters):
                np.testing.assert_array_equal(model.a, mean)
                np.testing.assert_array_equal(model.C, covariance)
            np.testing.assert_equal({key: algorithm.variance_model.__dict__[key] for key in variance}, variance)
            np.testing.assert_equal(algorithm.history, history)
            checked.append(event)
        return result

    algorithm._exact_kg_sample_plan = sample_plan
    algorithm._exact_posterior_update_score_one = native_draw
    algorithm._exact_posterior_update_scores = exact_scores
    algorithm.run()
    costs = algorithm.objective_diagnostics["value_precision"]
    assert checked and checked[0]["stage"] == 63
    assert 0 < costs["evaluations"] == len(extra_calls) <= 512
    assert costs["score_one_calls"] == costs["terminal_value_calls"] == costs["evaluations"]
    assert all(event["seconds"] > 0 and event["original_mc1_seconds"] > 0 for event in checked)
    assert len(algorithm.history) == adapter.research_simulator_calls + adapter.simulator_calls == 64
    assert adapter.simulator_calls == 32


def test_objective_scope_conditions_each_paid_label_once_from_the_configured_prior(project_root, monkeypatch):
    """A saved noisy-residual scope agrees with one batch Gaussian likelihood."""
    from copy import deepcopy
    import numpy as np
    from hegel_machine.cognition.sc_kg_environment import HegelProgramBasis
    algorithm, adapter, _ = _saved_terminal_failure_optimizer(project_root, monkeypatch,
        "forward_1_mechanism_change_noise_0.00")
    from core.gpr import ParametricGPR
    saved = json.loads((Path(__file__).resolve().parents[1]
        / "artifacts/sc_kg_terminal_value_v040_20260910/runs/forward_1_mechanism_change_noise_0.00_terminal_value.json").read_text())
    revision = saved["objective_scope_revisions"][0]
    start, end = revision["scope_start"], revision["stage"]
    history = [(tuple(point), np.asarray(output)) for point, output in
               zip(saved["initial_points"], saved["initial_outputs"])]
    history += [(tuple(row["point"]), np.asarray(row["output"])) for row in saved["simulations"]]
    algorithm.history = deepcopy(history[:end])
    algorithm.observations = {}
    for point, output in algorithm.history[start:]:
        algorithm.observations.setdefault(point, []).append(output)
    algorithm.active_scope_start = start
    algorithm.objective_research.state["accepted_revision"] = deepcopy(revision)
    # Past-scope labels cannot influence the accepted goal posterior.
    for _, output in algorithm.history[:start]:
        output[0] += 1000.
    from problems.single_objective import ScalarizedProblem
    adapter.base = ScalarizedProblem(adapter.next_problem)
    original_update = ParametricGPR.update
    updates, prior = [], {}

    def record_update(model, point, target, noise):
        if isinstance(model.basis_map, HegelProgramBasis) and model not in algorithm.constraint_models.values():
            if not updates:
                prior.update(mean=model.a.copy(), covariance=model.C.copy(), deviation=model.lambda_i)
            updates.append((tuple(point), float(target), float(noise)))
        return original_update(model, point, target, noise)

    with monkeypatch.context() as patch:
        patch.setattr(ParametricGPR, "update", record_update)
        algorithm._rebase_objective_scope(revision, stage=end)
    assert [(point, target) for point, target, _ in updates] == [
        (point, float(output[0])) for point, output in history[start:end]]
    assert len(updates) == end - start == 14
    np.testing.assert_array_equal(prior["mean"], [0., 1.])
    np.testing.assert_array_equal(prior["covariance"], np.eye(2) * algorithm.config.prior_var)
    model = algorithm.gpr[0]
    points = [point for point, _, _ in updates]
    design = model.augmented_feature_matrix(points)
    targets = np.asarray([target for _, target, _ in updates])
    noises = np.asarray([noise for _, _, noise in updates])
    prior_mean = np.r_[prior["mean"], np.zeros(len(model.sampled_set))]
    prior_variance = np.r_[np.diag(prior["covariance"]),
                          np.full(len(model.sampled_set), prior["deviation"])]
    precision = np.diag(1. / prior_variance) + design.T @ (design / noises[:, None])
    expected_covariance = np.linalg.inv(precision)
    expected_mean = np.linalg.solve(precision,
        prior_mean / prior_variance + design.T @ (targets / noises))
    np.testing.assert_allclose(model.a, expected_mean, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(model.C, expected_covariance, atol=1e-9, rtol=1e-9)
    # The same paid replicate has equal weight inside the initial scope or as
    # its last sequential update, with the same fixed observation variance.
    duplicate = next(index for index, point in enumerate(points) if point in points[:index])
    delayed = ParametricGPR(adapter.d, normalize_func=adapter.normalize, basis_map=model.basis_map)
    delayed.set_parametric_prior(prior["mean"], prior["deviation"], algorithm.config.prior_var)
    for index in [index for index in range(len(updates)) if index != duplicate] + [duplicate]:
        delayed.update(*updates[index])
    np.testing.assert_allclose(delayed.posterior_mean_many(points), model.posterior_mean_many(points), atol=1e-9)
    np.testing.assert_allclose(delayed.posterior_var_many(points), model.posterior_var_many(points), atol=1e-9)
    assert np.max(np.abs(model.a[model.p:])) > .01
    costs = algorithm.objective_diagnostics
    assert costs["head_posterior_updates"] == costs["head_posterior_noise_predictions"] == 14
    assert algorithm.objective_scope_revisions[-1]["posterior_updates"] == 14
    assert costs["head_posterior_seconds"] > 0.
    assert len(algorithm.history) == end and adapter.simulator_calls == 0
