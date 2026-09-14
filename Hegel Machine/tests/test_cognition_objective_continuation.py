"""Resume frozen goal validation and consume already selected real observations.

Saved labels replay the data-consumption boundary; these tests do not call a
simulator or claim an improved optimization trajectory.
"""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

import test_cognition_sc_kg_environment as sc_tests
from test_cognition_sc_kg_environment import project_root
from hegel_machine.cognition.objective_research import ObjectiveResearch


CASE = "value_precision_3_mechanism_change_noise_0.03"
DIRECTORY = "sc_kg_active_noise_v046_20260910"


@pytest.fixture(scope="module")
def saved_failure():
    root = Path(sc_tests.__file__).resolve().parents[1]
    path = root / "artifacts" / DIRECTORY / "runs" / f"{CASE}_active_noise_research.json"
    return json.loads(path.read_text())


def _paused_optimizer(project_root, monkeypatch, saved_failure):
    algorithm, adapter, environment = sc_tests._saved_terminal_failure_optimizer(
        project_root, monkeypatch, CASE, frozen_directory=DIRECTORY)
    algorithm._fit_initial_belief(algorithm._initial_samples())
    history = [(tuple(point), np.asarray(output)) for point, output in
               zip(saved_failure["initial_points"], saved_failure["initial_outputs"])]
    history += [(tuple(row["point"]), np.asarray(row["output"]))
                for row in saved_failure["simulations"][:25]]
    algorithm.history, algorithm.observations = history, {}
    for point, output in history:
        algorithm.observations.setdefault(point, []).append(output)
    algorithm.recovery.state = deepcopy(saved_failure["recovery"])
    algorithm.recovery.state["observations"] = algorithm.recovery.state["observations"][:25]
    algorithm.objective_research = ObjectiveResearch(environment.spec,
        state=saved_failure["objective_research"])
    assert len(history) == 57
    assert algorithm.recovery.state["phase"] == "unresolved"
    assert len(algorithm.objective_research.state["scope_research"]["validation"]) == 3
    return algorithm, adapter, environment


@pytest.mark.parametrize("case", ["ready", "insufficient_calls", "collecting",
                                  "no_candidate", "unrelated_pending", "active_constraint"])
def test_only_affordable_existing_validation_can_resume_after_constraint_failure(
        project_root, monkeypatch, saved_failure, case):
    algorithm, adapter, environment = _paused_optimizer(project_root, monkeypatch, saved_failure)
    research = algorithm.objective_research.state["scope_research"]
    point = tuple(saved_failure["simulations"][25]["point"])
    features = {name: value / 100. for name, value in zip(environment.spec.feature_names, point)}
    algorithm.config = replace(algorithm.config, N=58)
    monkeypatch.setattr(algorithm, "_consider_constraint_applicability", lambda: None)
    # Existing terminal offers must not take over the unresolved route.
    algorithm.objective_scope_revisions = [{"scope_start": 44}]
    monkeypatch.setattr(algorithm.objective_research, "terminal_offers",
        lambda *args, **kwargs: pytest.fail("unresolved validation must not open terminal offers"))
    native = [(98, 98, 98)]
    monkeypatch.setattr(type(algorithm).__mro__[1], "_generate_candidates",
        lambda self, iteration: (list(native), {native[0]: "native_test_control"}))
    if case == "insufficient_calls":
        research["validation"] = research["validation"][:2]
    elif case == "collecting":
        research["phase"] = "collecting"
    elif case == "no_candidate":
        research["candidate"] = None
    elif case == "unrelated_pending":
        algorithm.objective_research.state["pending"] = {
            "reason": "supported_goal_improvement", "purpose": "objective_exploration",
            "features": features}
    elif case == "active_constraint":
        algorithm.recovery.state["phase"] = "collecting"
        algorithm.recovery.state["controls"] = [features]
    calls = []

    def continuation(rows, forecast, **context):
        assert case == "ready"
        assert context["stage"] == len(rows) == 57
        assert context["remaining_calls"] == 1
        calls.append(context)
        return algorithm.objective_research._record_request(features, "objective_confirmation",
            "validate_objective_scope", context["stage"], candidate_mean=1., previous_mean=2.)

    monkeypatch.setattr(algorithm.objective_research, "next_control", continuation)
    candidates, sources = algorithm._generate_candidates(58)
    if case == "ready":
        assert candidates == [point] and sources[point] == "objective_confirmation"
        assert len(calls) == 1
    elif case == "active_constraint":
        assert candidates == [point] and sources[point] == "revision_coverage"
        assert not calls
    else:
        assert candidates == native and not calls
    assert len(algorithm.history) == 57 and adapter.simulator_calls == 0


@pytest.mark.parametrize("phase", [None, "collecting", "missing_candidate"])
def test_ordinary_optimization_does_not_open_or_fit_an_unready_goal(
        project_root, monkeypatch, saved_failure, phase):
    algorithm, adapter, _ = _paused_optimizer(project_root, monkeypatch, saved_failure)
    goal = algorithm.objective_research
    if phase is None:
        goal.state["scope_research"] = None
    elif phase == "collecting":
        goal.state["scope_research"]["phase"] = phase
    else:
        goal.state["scope_research"]["candidate"] = None
    before = goal.to_dict()
    point = {name: .5 for name in algorithm.problem.spec.feature_names}
    result = goal.prepare_shared_control(algorithm._objective_evidence(),
        lambda features: pytest.fail("an unready goal must not forecast ordinary controls"),
        point, scope_start=0, stage=57, remaining_calls=7, purpose="optimization")
    assert result is None and goal.to_dict() == before
    assert adapter.simulator_calls == 0


def test_ordinary_selected_vector_is_predicted_before_replay_and_consumed_once(
        project_root, monkeypatch, saved_failure):
    algorithm, adapter, _ = _paused_optimizer(project_root, monkeypatch, saved_failure)
    goal = algorithm.objective_research
    research = goal.state["scope_research"]
    frozen = deepcopy(research["candidate"])
    row = saved_failure["simulations"][25]
    point, output = tuple(row["point"]), np.asarray(row["output"])
    old_mean = saved_failure["actions"][25]["mu_before"][0]
    # The old mean is the saved pre-measurement forecast. Noise is obtained
    # from the actual paid-initialization HVD and does not set the gain test.
    noise = float(algorithm.variance_model.predict_variance(0, point, adapter))
    forecasts, measurements = [], []

    def forecast(features):
        assert len(algorithm.history) == 57 and not measurements
        assert research["candidate"] == frozen
        forecasts.append(deepcopy(features))
        return {"features": deepcopy(features), "objective_mean": old_mean, "objective_noise": noise}

    def replay_selected(selected, rng=None):
        assert tuple(selected) == point
        assert len(forecasts) == 1
        pending = goal.state["shared_pending"]
        assert pending["purpose"] == "optimization"
        assert pending["expected_global_index"] == 57
        assert pending["candidate_mean"] == pytest.approx(4.3248077118102035)
        assert research["candidate"] == frozen and len(research["validation"]) == 3
        assert goal.accepted_revision is None
        measurements.append(tuple(selected))
        return output.copy()

    monkeypatch.setattr(algorithm, "_objective_forecast", forecast)
    monkeypatch.setattr(adapter, "simulate", replay_selected)
    algorithm._recovery_control_purpose = "optimization"
    result = algorithm._simulate_and_store(point)
    np.testing.assert_array_equal(result, output)
    assert len(algorithm.history) == 58 and measurements == [point]
    assert goal.state["shared_pending"] is None
    continuing = goal.state["scope_research"]
    assert continuing["candidate"] == frozen and len(continuing["validation"]) == 4
    final = continuing["validation"][-1]
    assert final["source"] == "shared_optimization_control"
    assert final["global_index"] == 57
    assert final["candidate_error"] == pytest.approx(5.298408549691019e-6)
    assert final["previous_error"] == pytest.approx(13.750493972802436)
    # A positive but uncertain gain cannot install the candidate, and now
    # retains it until the next complete pair of independent future controls.
    assert continuing["status"] == "inconclusive_future_predictions"
    assert continuing["validation_target"] == 6
    assert goal.accepted_revision is None
    before = len(continuing["validation"])
    goal.observe(dict(zip(algorithm.problem.spec.feature_names, np.asarray(point) / 100.)), output,
        context={"stage": 58, "global_index": 57, "purpose": "optimization"})
    assert len(continuing["validation"]) == before
    assert adapter.simulator_calls == 0


def test_reusing_a_validation_control_cannot_supply_the_missing_fourth_validation(
        project_root, monkeypatch, saved_failure):
    algorithm, adapter, _ = _paused_optimizer(project_root, monkeypatch, saved_failure)
    goal = algorithm.objective_research
    research = goal.state["scope_research"]
    before = deepcopy(research)
    point = deepcopy(research["validation"][0]["features"])
    pending = goal.prepare_shared_control(algorithm._objective_evidence(),
        lambda features: pytest.fail("a used validation point must not receive another forecast"),
        point, scope_start=0, stage=57, remaining_calls=7, purpose="optimization")
    assert pending is None and goal.state["scope_research"] == before
    assert len(research["validation"]) == 3 and goal.accepted_revision is None
    assert adapter.simulator_calls == 0


@pytest.mark.parametrize("validation_count,remaining,reachable", [(4, 2, True), (4, 1, False), (5, 1, True)])
def test_unresolved_constraint_allows_only_the_affordable_remainder_of_an_extended_goal_round(
        project_root, monkeypatch, saved_failure, validation_count, remaining, reachable):
    algorithm, adapter, environment = _paused_optimizer(project_root, monkeypatch, saved_failure)
    research = algorithm.objective_research.state["scope_research"]
    for index in range(3, validation_count):
        row = saved_failure["simulations"][index + 22]
        point, output = tuple(row["point"]), np.asarray(row["output"])
        algorithm.history.append((point, output))
        algorithm.observations.setdefault(point, []).append(output)
        research["validation"].append({"features": dict(zip(environment.spec.feature_names,
            np.asarray(point) / 100.)), "global_index": len(algorithm.history) - 1,
            "previous_error": 10., "candidate_error": .01})
    research.update(validation_target=6, status="inconclusive_future_predictions")
    stage = len(algorithm.history)
    algorithm.config = replace(algorithm.config, N=stage + remaining)
    monkeypatch.setattr(algorithm, "_consider_constraint_applicability", lambda: None)
    native = [(98, 98, 98)]
    monkeypatch.setattr(type(algorithm).__mro__[1], "_generate_candidates",
        lambda self, iteration: (list(native), {native[0]: "native_test_control"}))
    offered = tuple(saved_failure["simulations"][stage - 32]["point"])
    features = dict(zip(environment.spec.feature_names, np.asarray(offered) / 100.))
    calls = []

    def next_control(rows, forecast, **context):
        assert reachable and len(rows) == stage and context["remaining_calls"] == remaining
        calls.append(context)
        return algorithm.objective_research._record_request(features, "objective_confirmation",
            "validate_objective_scope", stage, candidate_mean=1., previous_mean=2.)

    monkeypatch.setattr(algorithm.objective_research, "next_control", next_control)
    points, sources = algorithm._generate_candidates(stage + 1)
    assert points == ([offered] if reachable else native)
    assert len(calls) == int(reachable)
    if reachable:
        assert sources[offered] == "objective_confirmation"
    assert len(algorithm.history) == stage and adapter.simulator_calls == 0
