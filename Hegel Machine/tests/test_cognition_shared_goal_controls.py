"""A purchased goal action also tests an unresolved, independently frozen model.

The saved 58-row SC-KG prefix awaits its sixth comparison. Tests replay
its next measured vector; the runtime cases use the real initialized GP and
optimizer wrapper without making another physical simulator call.
The three tail-selection cases explicitly retain the rejected experiment.
"""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from test_cognition_sc_kg_environment import project_root
from test_cognition_objective_incumbent import saved_incumbent_failure
from test_cognition_objective_replacement_runtime import _live_goal, _runtime
from hegel_machine.cognition.objective_research import ObjectiveResearch


GOAL_PURPOSES = ("objective_exploration", "objective_confirmation")


@pytest.fixture(scope="module")
def frozen_shared_tail_research():
    source = Path(__file__).resolve().parents[1] / (
        "artifacts/sc_kg_shared_tail_20260912/method_source/objective_research.py")
    module_spec = importlib.util.spec_from_file_location(
        "hegel_machine.cognition._shared_tail_experiment_objective_research", source)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.ObjectiveResearch


def _action(goal, label, purpose, *, point=None):
    point = point or label["request"]["features"]
    return goal._record_request(point, purpose,
        "measured_and_posterior_incumbents_disagree", 58,
        incumbent_measured_mean=label["objective"] + .25)


def _forecast(label):
    return lambda features: {
        "features": deepcopy(features),
        "objective_mean": label["request"]["previous_mean"],
        "objective_noise": label["request"]["objective_noise"],
    }


def _prepare(goal, rows, label, purpose, *, forecast=None, point=None):
    return goal.prepare_shared_control(rows, forecast or _forecast(label),
        point or label["request"]["features"], scope_start=0,
        stage=len(rows), remaining_calls=64 - len(rows), purpose=purpose)


def _consume(goal, label, purpose, *, point=None):
    point = point or label["request"]["features"]
    context = {**label["context"], "purpose": purpose}
    goal.observe(point, [label["objective"], label["constraint"]], context=context)
    return context


@pytest.mark.parametrize("purpose", GOAL_PURPOSES)
def test_goal_action_resolves_saved_uncertainty_and_records_its_measured_outcome(
        saved_incumbent_failure, purpose):
    goal, rows, label = _live_goal(saved_incumbent_failure)
    frozen = deepcopy(goal.state["scope_research"])
    assert len(frozen["validation"]) == 5
    assert frozen["validation_rounds"][-1]["decision"] == "inconclusive"
    assert frozen["status"] == "inconclusive_future_predictions"
    action = _action(goal, label, purpose)
    cost = goal.state["compute_cost"]
    shared = _prepare(goal, rows, label, purpose)
    assert shared is not None and shared["expected_global_index"] == len(rows)
    assert goal.state["pending"] == action
    assert goal.state["shared_pending"] == shared
    assert goal.state["compute_cost"] == cost + 1
    assert not goal.state["canceled_requests"]
    assert not goal.state["outcomes"] and not goal.state["shared_outcomes"]
    assert goal.state["scope_research"]["candidate"] == frozen["candidate"]

    context = _consume(goal, label, purpose)
    outcome, = goal.state["outcomes"]
    shared_outcome, = goal.state["shared_outcomes"]
    assert outcome["request"] == action
    assert outcome["measured_improvement"] == pytest.approx(.25)
    assert shared_outcome["request"] == shared
    assert outcome["objective"] == shared_outcome["objective"] == label["objective"]
    assert goal.state["pending"] is None and goal.state["shared_pending"] is None
    validation = goal.state["scope_research"]["validation"]
    assert len(validation) == 6 and validation[-1]["global_index"] == 58
    assert validation[-1]["source"] == "shared_goal_control"
    assert validation[-1]["candidate_error"] == pytest.approx(
        (label["objective"] - shared["candidate_mean"]) ** 2)
    assert validation[-1]["previous_error"] == pytest.approx(
        (label["objective"] - shared["previous_mean"]) ** 2)
    assert goal.accepted_revision["model"] == frozen["candidate"]
    assert len(goal.accepted_revision["validation_indices"]) == 6

    # Re-delivery of the already consumed vector cannot make another action or
    # another independent model comparison.
    before = goal.to_dict()
    goal.observe(shared["features"], [label["objective"], label["constraint"]],
                 context=context)
    assert goal.to_dict() == before and len(rows) == 58


@pytest.mark.parametrize("purpose", GOAL_PURPOSES)
def test_repeated_compared_goal_control_keeps_action_without_charging_a_prediction(
        saved_incumbent_failure, purpose):
    goal, rows, label = _live_goal(saved_incumbent_failure)
    point = deepcopy(goal.state["scope_research"]["validation"][0]["features"])
    action = _action(goal, label, purpose, point=point)
    before = goal.to_dict()
    shared = _prepare(goal, rows, label, purpose, point=point,
        forecast=lambda features: pytest.fail("this control already compared the same models"))
    assert shared is None and goal.state["pending"] == action
    assert goal.state["compute_cost"] == before["compute_cost"]
    _consume(goal, label, purpose, point=point)
    assert len(goal.state["outcomes"]) == 1
    assert goal.state["outcomes"][0]["measured_improvement"] == pytest.approx(.25)
    assert not goal.state["shared_outcomes"] and not goal.state["shared_predictions"]
    assert goal.state["scope_research"] == before["scope_research"]
    assert goal.state["pending"] is None and not goal.state["canceled_requests"]


def test_goal_validation_itself_keeps_its_original_prediction_and_only_one_comparison(
        saved_incumbent_failure):
    goal, rows, label = _live_goal(saved_incumbent_failure)
    prediction = label["request"]
    original = goal._record_request(prediction["features"], "objective_confirmation",
        "validate_objective_scope", 58, candidate_mean=prediction["candidate_mean"],
        previous_mean=prediction["previous_mean"], objective_noise=prediction["objective_noise"])
    before = goal.to_dict()
    assert _prepare(goal, rows, label, "objective_confirmation",
        forecast=lambda features: pytest.fail("an existing validation prediction must be reused")) == original
    assert goal.to_dict() == before
    _consume(goal, label, "objective_confirmation")
    assert len(goal.state["outcomes"]) == 1 and not goal.state["shared_outcomes"]
    assert len(goal.state["scope_research"]["validation"]) == 6
    assert goal.state["scope_research"]["validation"][-1]["source"] == "objective_control"
    assert goal.state["pending"] is None and goal.state["shared_pending"] is None


def test_json_resume_preserves_both_pending_predictions_until_the_same_paid_label(
        saved_incumbent_failure):
    goal, rows, label = _live_goal(saved_incumbent_failure)
    action = _action(goal, label, "objective_confirmation")
    shared = _prepare(goal, rows, label, "objective_confirmation")
    state = json.loads(json.dumps(goal.to_dict()))
    restored = ObjectiveResearch(goal.spec, state=state)
    assert restored.to_dict() == state
    assert _prepare(restored, rows, label, "objective_confirmation",
        forecast=lambda features: pytest.fail("resume must not replace a frozen prediction")) == shared
    assert restored.state["pending"] == action and restored.to_dict() == state
    _consume(restored, label, "objective_confirmation")
    assert len(restored.state["outcomes"]) == len(restored.state["shared_outcomes"]) == 1
    assert restored.accepted_revision is not None


def test_selecting_another_goal_control_cancels_only_the_unpurchased_old_action(
        saved_incumbent_failure):
    goal, rows, label = _live_goal(saved_incumbent_failure)
    original = _action(goal, label, "objective_confirmation",
                       point={"x1": .11, "x2": .21, "x3": .31})
    shared = _prepare(goal, rows, label, "objective_confirmation")
    assert goal.state["pending"] is None and goal.state["shared_pending"] == shared
    cancellation, = goal.state["canceled_requests"]
    assert cancellation["request"] == original
    _consume(goal, label, "objective_confirmation")
    assert not goal.state["outcomes"] and len(goal.state["shared_outcomes"]) == 1
    assert len(goal.state["scope_research"]["validation"]) == 6


@pytest.mark.parametrize("purpose", GOAL_PURPOSES)
def test_goal_action_without_an_active_candidate_does_not_reopen_rejected_research(
        saved_incumbent_failure, monkeypatch, purpose):
    goal, rows, label = _live_goal(saved_incumbent_failure)
    rejected = deepcopy(goal.state["scope_research"])
    rejected.update(phase="completed", status="future_predictions_do_not_improve")
    goal.state.update(scope_research=None, scope_attempts=[rejected])
    action = _action(goal, label, purpose)
    before = goal.to_dict()
    monkeypatch.setattr(goal, "_resume_rejected", lambda *args, **kwargs:
        pytest.fail("sharing a selected goal action must not reopen a rejected candidate"))
    assert _prepare(goal, rows, label, purpose,
        forecast=lambda point: pytest.fail("there is no active candidate to validate")) is None
    assert goal.to_dict() == before
    _consume(goal, label, purpose)
    assert goal.state["outcomes"][0]["request"] == action
    assert goal.state["scope_research"] is None and goal.state["scope_attempts"] == [rejected]
    assert goal.state["pending"] is None and not goal.state["shared_outcomes"]


def test_frozen_tail_preserves_the_collecting_candidates_first_validation_slot(
        saved_incumbent_failure, monkeypatch, frozen_shared_tail_research):
    goal, rows, _ = _live_goal(saved_incumbent_failure)
    goal = frozen_shared_tail_research(goal.spec, state=goal.to_dict())
    research = goal.state["scope_research"]
    research.update(phase="collecting", candidate=None, validation=[])
    selected = {"purpose": "objective_confirmation", "reason": "validate_objective_scope"}
    calls = []

    def complete_candidate(paid, forecast, stage, remaining_calls):
        calls.append((paid, stage, remaining_calls))
        assert research["phase"] == "collecting"
        return selected

    monkeypatch.setattr(goal, "_scope_decision_confirmation", lambda *args:
        pytest.fail("taking the first of four validation slots makes completion impossible"))
    monkeypatch.setattr(goal, "_research_scope", complete_candidate)
    assert goal.next_control(rows, lambda point: None, scope_start=0,
                             stage=60, remaining_calls=4) == selected
    assert len(calls) == 1 and calls[0] == (rows, 60, 4)


@pytest.mark.parametrize("both_compared", [True, False])
def test_frozen_tail_leaves_already_compared_competitors_for_new_experiments(
        saved_incumbent_failure, both_compared, frozen_shared_tail_research):
    goal, rows, _ = _live_goal(saved_incumbent_failure)
    goal = frozen_shared_tail_research(goal.spec, state=goal.to_dict())
    research = goal.state["scope_research"]
    controls = [deepcopy(row["features"]) for row in research["validation"][:2]]
    paid = [row for row in rows if row["observation"]["features"] in controls
            and row["context"]["global_index"] >= research["scope_start"]]
    means = [(sum(row["context"]["objective"] for row in paid
                  if row["observation"]["features"] == point)
              / sum(row["observation"]["features"] == point for row in paid), point)
             for point in controls]
    empirical = min(means, key=lambda item: item[0])[1]
    predicted = next(point for point in controls if point != empirical)
    if not both_compared:
        research["validation"] = [row for row in research["validation"]
                                  if row["features"] != predicted]
    assert len(research["validation"]) >= 4
    predictions = []

    def forecast(point):
        predictions.append(deepcopy(point))
        return {"features": deepcopy(point), "objective_mean": float(point != predicted),
                "objective_noise": .001, "feasible": True}

    request = goal.next_control(paid, forecast, scope_start=0, stage=60, remaining_calls=4)
    assert len(predictions) == 2
    if both_compared:
        assert request is None and goal.state["pending"] is None
        assert not goal.state["requests"]
    else:
        assert request["features"] == predicted
        assert request["reason"] == "resolve_current_goal_incumbent"
    assert not goal.state["shared_predictions"] and not goal.state["outcomes"]


@pytest.mark.parametrize("purpose", GOAL_PURPOSES)
def test_runtime_goal_draw_freezes_real_gp_predictions_then_consumes_both_uses_once(
        project_root, monkeypatch, saved_incumbent_failure, purpose):
    algorithm, adapter, goal, rows = _runtime(
        project_root, monkeypatch, saved_incumbent_failure)
    algorithm.recovery.state["phase"] = "monitoring"
    _, _, label = _live_goal(saved_incumbent_failure)
    features = label["request"]["features"]
    point = tuple(round(features[name] * 100) for name in goal.spec.feature_names)
    vector = np.asarray([label["objective"], label["constraint"]])
    action = _action(goal, label, purpose)
    expected_gp = float(algorithm.gpr[0].posterior_mean(point))
    algorithm._recovery_control_purpose = purpose
    calls = {"draw": 0, "goal": 0}
    original_observe = goal.observe

    def replay_measured_vector(selected, rng):
        calls["draw"] += 1
        assert tuple(selected) == point and len(algorithm.history) == 58
        assert goal.state["pending"] == action
        shared = goal.state["shared_pending"]
        assert shared["expected_global_index"] == 58
        assert shared["previous_mean"] == pytest.approx(expected_gp)
        assert len(goal.state["scope_research"]["validation"]) == 5
        assert not goal.state["outcomes"] and not goal.state["shared_outcomes"]
        return vector.copy()

    def consume_goal(*args, **kwargs):
        calls["goal"] += 1
        assert calls["draw"] == 1 and len(algorithm.history) == 59
        return original_observe(*args, **kwargs)

    monkeypatch.setattr(adapter, "simulate", replay_measured_vector)
    monkeypatch.setattr(goal, "observe", consume_goal)
    result = algorithm._simulate_and_store(point)
    np.testing.assert_array_equal(result, vector)
    assert calls == {"draw": 1, "goal": 1}
    assert len(goal.state["outcomes"]) == len(goal.state["shared_outcomes"]) == 1
    research = goal.state["scope_research"] or goal.state["scope_attempts"][-1]
    assert len(research["validation"]) == 6
    assert goal.state["pending"] is None and goal.state["shared_pending"] is None
    assert not goal.state["canceled_requests"]
    assert len(algorithm.history) == 59 and len(rows) == 58
    assert adapter.simulator_calls == 0  # The callback replayed an existing draw.
