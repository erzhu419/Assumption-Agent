"""Current paid method questions compete without erasing older failed returns."""

from copy import deepcopy
from dataclasses import replace
import gzip
import importlib
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.action_policy import ActionOption, ResearchActionPolicy
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, ResearchConfig
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_session import UnifiedResearchSession, laboratory_for


@pytest.fixture(scope="module")
def recorded_numeric():
    path = Path(__file__).parent / "fixtures/method_applicability_root10.json.gz"
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


@pytest.fixture(scope="module")
def numeric_run(recorded_numeric):
    library = KnowledgeLibrary.from_dict(deepcopy(recorded_numeric["before_task_library"]))
    config = replace(ResearchConfig(**recorded_numeric["config"]),
                     seed=recorded_numeric["task"]["seed"])
    agent = UnifiedResearchAgent(config, library=library,
        measurement_budget=recorded_numeric["measurement_budget"])
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
        factory = importlib.import_module("run_observed_condition_decoding_study").laboratory
        with factory(recorded_numeric["task"]) as laboratory:
            spec = agent._spec(laboratory, ("reading",))
            report = agent.run(laboratory)
            physical_costs = dict(laboratory.costs)
    print(json.dumps({"current_method_question": {
        "status": report["status"], "costs": report["costs"],
        "confirmation_loss": report["confirmation_loss"],
        "joint_choices": [{key: event.get(key) for key in
            ("context", "reason", "selected_action", "compute_remaining")}
            for event in report["actions"] if event["action"] == "choose_condition_response_research"],
        "full_response_fits": [{key: event.get(key) for key in
            ("fitting_rows", "compute_cost", "candidate_count", "fitting_priority")}
            for event in report["actions"] if event["action"] == "refit_observed_condition_response"],
        "method_refits": [{key: event.get(key) for key in
            ("draft_id", "compute_cost", "training_loss", "validation_loss")}
            for event in report["actions"] if event["action"] == "replay_research_draft"]}}))
    return report, agent.library.to_dict(), physical_costs, spec


def test_paid_method_misfit_receives_an_actual_joint_response_trial(recorded_numeric, numeric_run):
    old = recorded_numeric["recorded_decisions"][-1]
    assert old["reason"] == "no_measured_positive_return"
    assert old["eligible_actions"] == ["joint_observed_condition_response"]
    assert old["counterexample_loss"] > 1.
    report, _, physical_costs, _ = numeric_run
    actions = report["actions"]
    applicability = [event for event in actions
                     if event["action"] == "assess_condition_method_applicability"]
    assert applicability and any(event["compute_cost"] > 0 for event in applicability)
    assert sum(event["compute_cost"] > 0 for event in applicability) == 1
    assert applicability[-1]["compute_cost"] == 0
    assert any(method["current_response_status"] == "misfit"
               for event in applicability
               for method in event["method_applicability"]["executing_methods"])
    contexts = {event["context"] for event in applicability}
    choices = [(index, event) for index, event in enumerate(actions)
               if event["action"] == "choose_condition_response_research"
               and event["selected_action"] == "joint_observed_condition_response"]
    assert choices and choices[0][1]["context"] in contexts
    choice_index = choices[0][0]
    fits = [(index, event) for index, event in enumerate(actions)
            if index > choice_index and event["action"] in {
                "refit_observed_condition_response", "replay_research_draft"}
            and event.get("comparison_basis") == "original_response"]
    assert fits and any(event["compute_cost"] > 0 for _, event in fits)
    assert any(event.get("training_loss") is not None
               and event.get("validation_loss") is not None
               and max(event["training_loss"], event["validation_loss"])
                   < recorded_numeric["config"]["minimum_improvement"]
               for _, event in fits)
    origin = next(event for event in actions[choice_index + 1:]
                  if event["action"] in {"research_action_feedback", "research_action_pending"}
                  and event["research_action"] == "joint_observed_condition_response")
    assert origin["compute_cost"] >= sum(event["compute_cost"] for event in applicability)
    assert any(event["action"] == "acquire"
               and event["purpose"] == "distinguish_observed_condition_response"
               for event in actions[choice_index + 1:])
    for resource in ("observation_cost", "measurement_cost"):
        assert report["costs"][resource] == physical_costs[resource]
        assert physical_costs[resource] == sum(event[resource] for event in actions
                                               if event["action"] == "acquire")
    assert report["costs"]["observation_cost"] <= recorded_numeric["config"]["observation_budget"]
    assert report["costs"]["measurement_cost"] <= recorded_numeric["measurement_budget"]
    assert report["costs"]["compute_cost"] <= recorded_numeric["config"]["compute_budget"]
    confirmations = [(index, event) for index, event in enumerate(actions)
                     if index > fits[0][0] and event["action"] == "validate_frozen_program"]
    assert confirmations
    for index, event in confirmations:
        purchase_index, purchase = next((i, item) for i, item in reversed(list(enumerate(actions[:index])))
            if item["action"] == "acquire" and item["purpose"] == "confirm_frozen_program")
        assert purchase_index > fits[0][0]
        batch = next(item for item in report["batches"]
                     if item["probe"]["probe_id"] == purchase["probe_id"])
        assert event["observation_rows"] == len(batch["observations"])
        if "program" in event:
            assert event["loss"] == pytest.approx(loss(Program.from_dict(event["program"]),
                [Observation(**row) for row in batch["observations"]], report["fallback"], .5))
    if report["status"] == "completed":
        assert confirmations and report["confirmation_loss"] <= recorded_numeric["config"]["minimum_improvement"]


def test_json_keeps_old_negative_returns_and_the_same_current_question(recorded_numeric, numeric_run):
    report, knowledge, _, spec = numeric_run
    action = "joint_observed_condition_response"
    base = "squared:numeric:multiple:condition_response:paid_observation"
    old_record = recorded_numeric["before_task_library"]["action_experience"][base][action]
    assert old_record["trials"] == 3 and old_record["expected_gain"] < 0
    assert knowledge["action_experience"][base][action] == old_record
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(knowledge)))
    event = next(item for item in report["actions"]
                 if item["action"] == "assess_condition_method_applicability")
    rows = [Observation(**row) for batch in report["batches"][:4] for row in batch["observations"]]
    policy = ResearchActionPolicy(rows, spec, restored.action_experience)
    policy.set_condition("condition_response:paid_observation")
    saved = deepcopy(restored.action_experience)
    context = policy.set_action_knowledge(action, deepcopy(event["method_applicability"]))
    assert context == event["context"]
    assert policy.set_action_knowledge(action, json.loads(json.dumps(event["method_applicability"]))) == context
    assert restored.action_experience == saved
    # The already paid generic failure remains a stop after JSON recovery.
    # This uses the real declined quote and remaining resources, not made-up loss.
    old_decision = recorded_numeric["recorded_decisions"][-1]
    policy = ResearchActionPolicy(rows, spec, restored.action_experience)
    policy.set_condition("condition_response:paid_observation")
    quote = old_decision["options"][0]
    option = ActionOption(quote["name"], quote["observation_cost"], quote["compute_cost"],
                          measurement_cost=quote["measurement_cost"])
    assert policy.choose([option], observation_remaining=old_decision["observation_remaining"],
        measurement_remaining=old_decision["measurement_remaining"],
        compute_remaining=old_decision["compute_remaining"],
        minimum_improvement=recorded_numeric["config"]["minimum_improvement"]) is None
    assert policy.last_decision["reason"] == "no_measured_positive_return"


def test_normal_risk_run_preserves_budget_for_paid_counterevidence_and_new_confirmation():
    path = Path(__file__).parent / "fixtures/risk_early_budget_development.json.gz"
    with gzip.open(path, "rt") as stream:
        recorded = json.load(stream)
    old = recorded["recorded_late_choice"]
    assert old["reason"] == "no_affordable_action" and not old["eligible_actions"]
    assert old["observation_remaining"] < min(option["observation_cost"] for option in old["options"])
    config = ResearchConfig(**recorded["config"])
    assert (config.observation_budget, recorded["measurement_budget"], config.compute_budget) == (256, 1024, 160000)
    labs = []

    def factory(arguments):
        laboratory = laboratory_for(arguments)
        labs.append(laboratory)
        return laboratory

    session = UnifiedResearchSession(config, tasks=[recorded["task"]],
        measurement_budget=recorded["measurement_budget"], lab_factory=factory)
    state = json.loads(json.dumps(session.run(max_tasks=1)))
    report = state["results"][0]["research"]
    actions = report["actions"]
    assert report["risk_quantile"] == .9 and report["loss_kind"] == "pinball"
    reuse_index, reuse = next((index, event) for index, event in enumerate(actions)
                              if event["action"] == "reuse_paid_risk_response_views")
    assert reuse["new_observations"] == 0
    bank_purchases = [event for event in actions[:reuse_index] if event["action"] == "acquire"]
    assert len(bank_purchases) == 3
    assert sum(event["observation_cost"] for event in bank_purchases) == (
        reuse["fitting_rows"] + reuse["validation_rows"])
    assert all(event["channels"] == reuse["channels"] for event in bank_purchases)
    agenda_index, agenda = next((index, event) for index, event in enumerate(actions)
                               if event["action"] == "assess_paid_risk_agenda")
    assert agenda["compute_cost"] > 0 and agenda["training_rows"] > 0 and agenda["validation_rows"] > 0
    choice_index, choice = next((index, event) for index, event in enumerate(actions)
        if index > agenda_index and event["action"] == "choose_risk_research"
        and event["selected_action"] is not None)
    propose_index, proposal = next((index, event) for index, event in enumerate(actions)
        if index > choice_index and event["action"] == "propose_risk_research"
        and event["status"] == "requires_fresh_confirmation")
    assert proposal["compute_cost"] > 0 and proposal["selected_kind"] is not None
    first_confirmation = next(index for index, event in enumerate(actions)
                              if event["action"] == "validate_frozen_program")
    assert agenda_index < first_confirmation
    if propose_index > first_confirmation:
        prior_confirmation = [event for event in actions[:choice_index]
                              if event["action"] == "validate_frozen_program"][-1]
        assert not prior_confirmation["accepted"]
        assert not prior_confirmation["noise_assessment"]["adequate"]
    feedback_index, feedback = next((index, event) for index, event in enumerate(actions)
        if index > propose_index and event["action"] == "risk_research_feedback"
        and event["before_loss"] is not None)
    purchase_index, purchase = next((index, event) for index, event in enumerate(actions)
        if propose_index < index < feedback_index and event["action"] == "acquire"
        and event["probe_id"] == feedback["probe_id"])
    assert purchase_index > propose_index and purchase["observation_cost"] > 0
    batch = next(item for item in report["batches"]
                 if item["probe"]["probe_id"] == feedback["probe_id"])
    rows = [Observation(**row) for row in batch["observations"]]
    assert feedback["risk_quantile"] == .9 and feedback["observation_rows"] == len(rows)
    assert feedback["before_loss"] == pytest.approx(loss(Program.from_dict(feedback["baseline_program"]),
        rows, feedback["baseline_fallback"], .9))
    assert feedback["after_loss"] == pytest.approx(loss(Program.from_dict(feedback["program"]),
        rows, feedback["fallback"], .9))
    assert any(event["action"] == "preserve_research_draft"
               and event.get("domain") == "risk_method" for event in actions)
    for resource in ("observation_cost", "measurement_cost"):
        assert report["costs"][resource] == sum(laboratory.costs[resource] for laboratory in labs)
        assert report["costs"][resource] == sum(event[resource] for event in actions
                                               if event["action"] == "acquire")
    assert report["costs"]["observation_cost"] <= 256
    assert report["costs"]["measurement_cost"] <= 1024
    assert report["costs"]["compute_cost"] <= 160000
    assert state["costs"] == {resource: report["costs"][resource]
                             for resource in ("observation_cost", "measurement_cost", "compute_cost")}
    print(json.dumps({"early_paid_risk_research": {"status": report["status"],
        "costs": report["costs"], "reason": agenda["reason"],
        "research_action": choice["selected_action"], "feedback_status": feedback["status"],
        "fresh_before": feedback["before_loss"], "fresh_after": feedback["after_loss"]}}))
