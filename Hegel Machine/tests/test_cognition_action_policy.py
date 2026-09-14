import json
from dataclasses import replace

from hegel_machine.cognition.action_policy import ActionOption, ResearchActionPolicy
from hegel_machine.cognition.models import Observation, TaskSpec


def specification(**changes):
    spec = TaskSpec("first", "A public objective", ("x", "z"), (), ())
    return replace(spec, **changes)


ROWS = [Observation({"x": -1.0, "z": 0.2}, 0.0),
        Observation({"x": 1.0, "z": 0.3}, 1.0)]
OPTIONS = [ActionOption("transition", 8, 80), ActionOption("task_value", 8, 80)]


def choose(policy, options=OPTIONS, **remaining):
    return policy.choose(options, observation_remaining=remaining.get("observations", 96),
                         compute_remaining=remaining.get("compute", 2400))


def outcome(policy, name, after, *, observations=8, compute=80):
    policy.update(name, before_loss=1.0, after_loss=after,
                  observation_cost=observations, compute_cost=compute)


def test_action_experience_transfers_across_names_but_respects_the_fixed_objective():
    experience = {}
    prediction = ResearchActionPolicy(ROWS, specification(), experience)
    outcome(prediction, "transition", 0.2)
    outcome(prediction, "task_value", 0.8)
    risk = ResearchActionPolicy(ROWS, specification(decision_costs=(9.0, 1.0)), experience)
    outcome(risk, "transition", 1.2)
    outcome(risk, "task_value", 0.4)

    renamed = [Observation({"r": row.features["x"], "s": row.features["z"]}, row.target) for row in ROWS]
    other_spec = specification(task_id="unrelated_name", feature_names=("r", "s"), context={"system_id": "other"})
    transferred = ResearchActionPolicy(renamed, other_spec, experience)
    assert choose(transferred).name == "transition"
    assert choose(risk).name == "task_value"
    different_risk = ResearchActionPolicy(ROWS, specification(decision_costs=(1.0, 9.0)), experience)
    assert different_risk.context != risk.context
    assert different_risk.experience == {}


def test_new_negative_result_changes_selection_without_forced_negative_retries():
    policy = ResearchActionPolicy(ROWS, specification(), {})
    outcome(policy, "transition", 0.2)
    outcome(policy, "task_value", 0.5)
    assert choose(policy).name == "transition"
    outcome(policy, "transition", 2.0)
    assert choose(policy).name == "task_value"
    for _ in range(12):
        assert choose(policy).name == "task_value"
    outcome(policy, "task_value", 2.0)
    assert choose(policy) is None
    assert policy.last_decision["reason"] == "no_measured_positive_return"


def test_unknown_and_neutral_actions_have_bounded_exploration():
    policy = ResearchActionPolicy(ROWS, specification(), {})
    outcome(policy, "transition", 0.2)
    assert choose(policy).name == "task_value"
    outcome(policy, "task_value", 1.0)
    assert choose(policy).name == "transition"
    outcome(policy, "transition", 2.0)
    assert choose(policy).name == "task_value"
    assert policy.last_decision["reason"] == "resolve_neutral_first_trial"
    outcome(policy, "task_value", 1.0)
    assert choose(policy) is None


def test_observation_and_compute_budgets_are_independent_constraints():
    policy = ResearchActionPolicy(ROWS, specification(), {})
    options = [ActionOption("transition", 12, 30), ActionOption("refit", 0, 200)]
    assert choose(policy, options, observations=8, compute=240).name == "refit"
    assert choose(policy, options, observations=16, compute=100).name == "transition"
    assert choose(policy, options, observations=8, compute=100) is None


def test_actual_downstream_search_cost_changes_the_value_of_a_cheap_probe():
    policy = ResearchActionPolicy(ROWS, specification(), {})
    outcome(policy, "transition", 0.6, compute=1800)
    outcome(policy, "task_value", 0.7, compute=80)
    assert choose(policy).name == "task_value"


def test_resource_scarcity_changes_choice_without_exchanging_budget_units():
    policy = ResearchActionPolicy(ROWS, specification(), {})
    options = [ActionOption("representative", 20, 20), ActionOption("refit", 0, 600)]
    outcome(policy, "representative", 0.6, observations=20, compute=20)
    outcome(policy, "refit", 0.6, observations=0, compute=600)
    assert choose(policy, options, observations=100, compute=600).name == "representative"
    assert choose(policy, options, observations=20, compute=6000).name == "refit"


def test_feedback_and_next_choice_survive_json_save_restore():
    experience = {}
    policy = ResearchActionPolicy(ROWS, specification(decision_costs=(9.0, 1.0)), experience)
    options = [ActionOption("refit", 0, 150), ActionOption("calibration", 8, 80)]
    outcome(policy, "refit", 1.1, observations=0, compute=150)
    policy.update("calibration", before_loss=0.2, after_loss=0.1, baseline_loss=1.0,
                  observation_cost=8, compute_cost=80)
    restored_experience = json.loads(json.dumps(experience))
    restored = ResearchActionPolicy(ROWS, specification(decision_costs=(90.0, 10.0)), restored_experience)
    assert choose(policy, options).name == choose(restored, options).name == "calibration"
    assert restored.experience["calibration"]["expected_gain"] == 0.1
    outcome(restored, "calibration", 1.5)
    assert choose(restored, options) is None
    assert experience[policy.context]["calibration"]["trials"] == 1
    assert restored_experience[restored.context]["calibration"]["trials"] == 2


def test_observed_conditions_change_actions_and_delayed_feedback_credits_its_origin():
    experience = {}
    policy = ResearchActionPolicy(ROWS, specification(), experience)
    origin = policy.set_condition("transfer_counterexample")
    outcome(policy, "transition", 1.2)
    outcome(policy, "task_value", .2)
    assert choose(policy).name == "task_value"
    current = policy.set_condition("new_counterexample")
    outcome(policy, "transition", .2)
    outcome(policy, "task_value", 1.2)
    assert choose(policy).name == "transition"
    assert current != origin and origin.startswith(policy.base_context)

    policy.update("task_value", before_loss=1., after_loss=2., observation_cost=8,
                  compute_cost=80, context=origin)
    assert policy.context == current
    assert experience[current]["task_value"]["trials"] == 1
    assert experience[origin]["task_value"]["trials"] == 2
    restored = ResearchActionPolicy(ROWS, specification(), json.loads(json.dumps(experience)))
    restored.set_condition("transfer_counterexample")
    assert choose(restored) is None
    restored.set_condition("new_counterexample")
    assert choose(restored).name == "transition"
