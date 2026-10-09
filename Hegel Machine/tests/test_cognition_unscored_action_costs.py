"""Unscored executions retain real costs without manufacturing response gains."""

import json

import pytest

from hegel_machine.cognition.action_policy import ActionOption, ResearchActionPolicy
from hegel_machine.cognition.models import TaskSpec
from hegel_machine.cognition.unified_session import UnifiedResearchSession, laboratory_for
from test_cognition_object_action_selection import BASE, CONFIG, continue_tasks, serialize


SPEC = TaskSpec("execution", "Public response", ("x",), (), ())


def test_no_candidate_is_an_attempt_but_not_a_measured_response_trial():
    experience = {}
    policy = ResearchActionPolicy([], SPEC, experience)
    recorded = policy.record_unscored_attempt("diagnose", status="no_candidate",
        observation_cost=7, measurement_cost=14, compute_cost=90)
    assert recorded["trials"] == 0 and recorded["unscored_attempts"] == 1
    assert recorded["no_candidate_attempts"] == 1
    assert not {"expected_gain", "last_gain", "last_before_loss", "last_after_loss", "trial_outcomes"} & recorded.keys()
    restored = ResearchActionPolicy([], SPEC, json.loads(json.dumps(experience)))
    choice = restored.choose([ActionOption("diagnose", 0, 1), ActionOption("new_direction", 0, 20)],
        observation_remaining=100, measurement_remaining=100, compute_remaining=1000)
    assert choice.name == "new_direction"
    assert restored._estimated_costs(restored.experience["diagnose"]) == {
        "observation_cost": 7., "measurement_cost": 14., "compute_cost": 90.}


@pytest.mark.parametrize("status, adjusted_gain", [("no_candidate", .4), ("unresolved", .8)])
def test_execution_failure_and_unresolved_cost_do_not_rewrite_a_real_gain(status, adjusted_gain):
    experience = {}
    policy = ResearchActionPolicy([], SPEC, experience)
    initial = policy.update("revise", before_loss=1., after_loss=.2,
        observation_cost=0, measurement_cost=0, compute_cost=10)
    origin = {"context": policy.context, "action": "revise", "trial": 1, "credited_gain": initial["last_gain"]}
    policy.revise_feedback(experience, origin, status="supported", before_loss=1., after_loss=.2)
    policy.record_unscored_attempt("revise", status=status,
        observation_cost=0, measurement_cost=0, compute_cost=30)
    restored = ResearchActionPolicy([], SPEC, json.loads(json.dumps(experience)))
    record = restored.experience["revise"]
    assert record["trials"] == 1 and record["expected_gain"] == pytest.approx(.8)
    assert record["trial_outcomes"]["1"]["status"] == "supported"
    assert restored.choose([ActionOption("revise", 0, 10), ActionOption("new_direction", 0, 20)],
        observation_remaining=100, compute_remaining=1000).name == "new_direction"
    assert restored.choose([ActionOption("revise", 0, 10)],
        observation_remaining=100, compute_remaining=1000).name == "revise"
    decision = restored.last_decision
    assert decision["adjusted_gain"] == pytest.approx(adjusted_gain)
    assert decision["attempt_count"] == 2
    assert decision["estimated_costs"]["compute_cost"] == 20
    assert decision["dominant_resource_share"] == pytest.approx(.02)
    # A later real paired response is another measured trial, while the past
    # execution and its costs remain part of the action's experience.
    restored.update("revise", before_loss=1., after_loss=.4, observation_cost=0, compute_cost=10)
    assert restored.experience["revise"]["trials"] == 2
    assert restored.experience["revise"]["unscored_attempts"] == 1
    assert restored.experience["revise"]["unscored_compute_cost"] == 30


def test_real_failed_relation_search_is_retained_before_input_repair_and_after_json():
    source = serialize(UnifiedResearchSession(CONFIG,
        tasks=[{**BASE, "seed": 221}], measurement_budget=4608).run())
    task = {**BASE, "seed": 441, "actuator_curvature": .31}
    state = continue_tasks(source, [task])[0]
    report = state["results"][-1]["research"]
    assert report["status"] == "completed" and report["confirmation_loss"] <= 1e-7
    failed = [event for event in report["actions"]
              if event["action"] == "propose_object_research" and event["status"] == "no_candidate"]
    assert failed
    restored = UnifiedResearchSession(state=serialize(state))
    for event in failed:
        context, action = event["context"], event["direction"]
        record = restored.library.action_experience[context][action]
        assert event["compute_cost"] > 0
        assert record["trials"] == 0 and record["no_candidate_attempts"] == 1
        assert record == event["unscored_feedback"]
        for resource in ("observation", "measurement", "compute"):
            assert record[f"unscored_{resource}_cost"] == event[f"{resource}_cost"]
        assert "expected_gain" not in record and "trial_outcomes" not in record
    first = next(item for item in report["actions"] if item["action"] == "choose_object_research")
    with laboratory_for(task) as laboratory:
        policy = ResearchActionPolicy([], laboratory.spec, restored.library.action_experience)
        policy.set_condition(first["context"][len(policy.base_context) + 1:])
        options = [ActionOption(**item) for item in first["options"]]
        chosen = policy.choose(options, observation_remaining=first["observation_remaining"],
            measurement_remaining=first["measurement_remaining"], compute_remaining=first["compute_remaining"])
    assert chosen.name != first["selected_action"]
    assert policy.last_decision["trials"] > 0
    assert any(event["action"] == "object_research_feedback" and event["direction"] == chosen.name
               and event["status"] == "supported" and event["before_loss"] > event["after_loss"]
               for event in report["actions"])
