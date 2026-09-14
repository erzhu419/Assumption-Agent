"""Research-action experience must change real queries and receive real returns."""

import json

import pytest

from hegel_machine.cognition.action_policy import ResearchActionPolicy
from hegel_machine.cognition.controller import CognitiveAgent
from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, Probe, ResearchConfig, TaskSpec


class RecordingEnvironment:
    def __init__(self, environment):
        self.environment = environment
        self.spec = environment.spec
        self.queries = []

    def observe(self, probe):
        rows = self.environment.observe(probe)
        self.queries.append((probe, rows))
        return rows


class ConstantMeasurements:
    """A measured solved task may finish with an experiment awaiting feedback."""

    def __init__(self):
        self.spec = TaskSpec(
            "stable_measurement", "Predict the measured response.", ("x",),
            (Probe("initial", tuple({"x": x} for x in (-1., -.4, .3, .8))),), (),
            feature_bounds={"x": (-1., 1.)}, max_probe_length=8)

    def observe(self, probe):
        rows, history = [], []
        for features in probe.sequence:
            rows.append(Observation(dict(features), .25, tuple(history)))
            history.append({**features, "__target__": .25})
        return rows


def state_experiment(favored):
    environment = RecordingEnvironment(make_suite(401, "continual")[9])
    library = KnowledgeLibrary()
    # Seed the return of earlier research actions, not a solution program or
    # any extra observations available to this task's controller.
    policy = ResearchActionPolicy([Observation({}, 0.), Observation({}, 1.)],
                                  environment.spec, library.action_experience)
    for condition in ("transfer_counterexample", "new_counterexample", "model_residual", "adequate"):
        policy.set_condition(condition)
        for name in ("transition", "task_value", "representative", "refit", "representation"):
            policy.update(name, before_loss=1., after_loss=.1 if name == favored else 1.5,
                          observation_cost=4, compute_cost=8)
    config = ResearchConfig(seed=401, partition="continual", max_rounds=2)
    result = CognitiveAgent(config, library).run(environment)
    return environment, library, result


def program_before(actions, event_index, fallback):
    current = Program("const", value=fallback)
    for action in actions[:event_index]:
        if action.action == "revise":
            current = Program.from_dict(action.details["after_program"])
        elif action.action == "rollback":
            current = Program.from_dict(action.details["restored_program"])
    return current


@pytest.mark.parametrize("favored", ["transition", "task_value", "representative"])
def test_current_observed_boundary_overrides_unconditioned_imported_action_returns(favored):
    environment, _, result = state_experiment(favored)
    index, planning = next((index, action) for index, action in enumerate(result.actions)
                           if action.action == "plan_experiment" and "action_policy" in action.details)
    decision = planning.details["action_policy"]
    assert set(decision["eligible_actions"]) >= {"transition", "task_value", "representative"}
    assert decision["selected_action"] == "transition"
    assert decision["reason"] == "resolve_observed_boundary"
    assert planning.details["boundary_support"]
    assert decision["context"].startswith("squared:binary:multiple:")
    acquired = next(action for action in result.actions[index + 1:] if action.action == "observe")
    probe, rows = next((probe, rows) for probe, rows in environment.queries
                      if probe.probe_id == acquired.details["probe_id"])
    assert probe.probe_id == planning.details["selected_probe_id"]
    assert probe.cost == decision["quoted_observation_cost"] == len(rows)
    assert result.observation_cost == sum(probe.cost for probe, _ in environment.queries)
    assert result.observation_cost <= 96 and result.compute_cost <= 24000


def test_pending_action_survives_json_and_receives_new_common_trajectory_feedback():
    first_environment = RecordingEnvironment(ConstantMeasurements())
    library = KnowledgeLibrary()
    config = ResearchConfig(seed=61, observation_budget=48, compute_budget=9000, max_rounds=3)
    first = CognitiveAgent(config, library).run(first_environment)
    state = json.loads(json.dumps(first.research_state))
    assert state["pending_actions"]
    pending = state["pending_actions"][0]
    assert pending["observation_cost"] > 0 and pending["compute_cost"] > 0
    restored_library = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    resumed_environment = RecordingEnvironment(ConstantMeasurements())
    resumed = CognitiveAgent(config, restored_library).run(resumed_environment, research_state=state)

    index, feedback = next((index, action) for index, action in enumerate(resumed.actions)
                           if action.action == "action_feedback" and action.details["action"] == pending["name"])
    preceding_query = next(action for action in reversed(resumed.actions[:index]) if action.action == "observe")
    probe, common_rows = next((probe, rows) for probe, rows in resumed_environment.queries
                             if probe.probe_id == preceding_query.details["probe_id"])
    assert probe.probe_id.startswith("representative_")
    assert all(probe.sequence != old.sequence for old, _ in first_environment.queries)
    assert all(probe.probe_id != "initial" for probe, _ in resumed_environment.queries)
    details = feedback.details
    assert details["reference_rows"] == len(common_rows)
    assert details["last_before_loss"] == pytest.approx(
        loss(Program.from_dict(pending["program"]), common_rows, resumed.fallback, .5))
    assert details["last_after_loss"] == pytest.approx(loss(resumed.program, common_rows, resumed.fallback, .5))
    assert details["mean_observation_cost"] == pending["observation_cost"] + len(common_rows)
    assert details["mean_compute_cost"] > pending["compute_cost"]
    persisted = restored_library.action_experience[details["context"]][pending["name"]]
    assert persisted["trials"] == 1 and persisted["last_gain"] == 0.


def test_a_real_transition_counterexample_changes_theory_and_action_experience():
    environment, library, result = state_experiment("transition")
    experiment_index = next(index for index, action in enumerate(result.actions)
                            if action.action == "plan_experiment"
                            and action.details.get("action_policy", {}).get("selected_action") == "transition")
    counterexample = next(action for action in result.actions[experiment_index + 1:]
                          if action.action == "targeted_prediction_check")
    assert counterexample.details["further_revision_needed"]
    feedback_index, feedback = next((index, action) for index, action in enumerate(result.actions)
                                   if action.action == "action_feedback" and action.details["action"] == "transition")
    assert feedback.round > counterexample.round
    before = program_before(result.actions, experiment_index, result.fallback)
    after = program_before(result.actions, feedback_index, result.fallback)
    assert after != before
    preceding_query = next(action for action in reversed(result.actions[:feedback_index]) if action.action == "observe")
    _, common_rows = next((probe, rows) for probe, rows in environment.queries
                          if probe.probe_id == preceding_query.details["probe_id"])
    before_loss = loss(before, common_rows, result.fallback, .5)
    after_loss = loss(after, common_rows, result.fallback, .5)
    assert before_loss > after_loss
    details = feedback.details
    assert details["last_before_loss"] == pytest.approx(before_loss)
    assert details["last_after_loss"] == pytest.approx(after_loss)
    assert details["reference_rows"] == len(common_rows)
    actual = library.action_experience[details["context"]]["transition"]
    assert actual["trials"] == 2 and actual["expected_gain"] != .9
    assert actual["mean_observation_cost"] > 4 and actual["mean_compute_cost"] > 8
