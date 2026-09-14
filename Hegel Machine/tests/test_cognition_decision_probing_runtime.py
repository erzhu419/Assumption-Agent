"""Actual Controller acquisition, parameter repair and fresh history comparison."""

from dataclasses import replace
import json

from hegel_machine.cognition import abstractions, controller
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, ResearchConfig, SearchBudget
from hegel_machine.cognition.workspace import ResearchWorkspace
from test_cognition_decision_probing import ResponseApparatus, acquired, policies


class RecordingApparatus(ResponseApparatus):
    def __init__(self, feedback):
        super().__init__(feedback)
        self.probes, self.responses = [], []

    def observe(self, probe):
        rows = super().observe(probe)
        self.probes.append(probe)
        self.responses.append(rows)
        return rows


def prepared(feedback=0.):
    environment = RecordingApparatus(feedback)
    rows, pairs = acquired(environment), policies()
    workspace = ResearchWorkspace(environment.spec.task_id, observations=rows, reference=list(rows),
        queried=list(environment.probes), incumbent=pairs[1][0], bank=[pairs[0][0]], fallback=0.,
        initial_loss=1., uncertainty_model={"kind": "nominal"},
        requested_method={"name": "refit", "method_id": "known_refit", "context": "numeric", "condition": {}})
    environment.probes.clear()
    environment.responses.clear()
    config = ResearchConfig(seed=7, observation_budget=64, compute_budget=16000,
        max_rounds=2, use_memory=False, use_abstractions=False, state_search=False,
        joint_planning=False, use_warm_start=False)
    return environment, workspace, config


def test_controller_repairs_after_targeted_evidence_and_earns_return_on_two_unseen_representative_trials(monkeypatch):
    environment, workspace, config = prepared()
    original_fit = abstractions.fit_template
    actual_fits = []

    def observe_real_fit(template, mapping, observations, *args, **kwargs):
        result = original_fit(template, mapping, observations, *args, **kwargs)
        actual_fits.append((list(observations), result))
        return result

    monkeypatch.setattr(abstractions, "fit_template", observe_real_fit)
    result = controller.CognitiveAgent(config).run(environment,
        research_state=json.loads(json.dumps(workspace.to_dict())))
    plan = next(action for action in result.actions if action.action == "plan_experiment"
                and action.details.get("policy") == "decision_history_contrast")
    target_ids = {item["probe_id"] for item in plan.details["preparations"]}
    reference_ids = set(plan.details["reference_probe_ids"])
    targeted = [block for probe, block in zip(environment.probes, environment.responses) if probe.probe_id in target_ids]
    reference = [block for probe, block in zip(environment.probes, environment.responses) if probe.probe_id in reference_ids]
    assert len(targeted) == len(reference) == 2
    assert all(not block[0].history for block in [*targeted, *reference])
    targeted_rows = {id(row) for block in targeted for row in block}
    reference_rows = {id(row) for block in reference for row in block}
    fit_rows, fitted = next((rows, fit) for rows, fit in actual_fits if targeted_rows <= {id(row) for row in rows})
    assert not reference_rows & {id(row) for row in fit_rows}
    assert fitted is not None
    assert result.program == policies()[0][0].program and result.validation_loss == 0.
    comparisons = [action for action in result.actions if action.action == "compare"]
    assert comparisons[0].details["adopt"] and comparisons[0].details["fresh_gain"] > 0.
    feedback = next(action for action in result.actions if action.action == "action_feedback"
                    and action.details["action"] == "task_value")
    assert feedback.details["reference_rows"] == sum(map(len, reference)) == 16
    assert feedback.details["last_after_loss"] == 0. < feedback.details["last_before_loss"]
    assert feedback.details["mean_observation_cost"] == plan.details["observation_cost"] == 24
    assert feedback.details["mean_compute_cost"] >= plan.compute_cost + feedback.compute_cost
    assert any(action.action == "defer_method_step" and action.details["method_id"] == "known_refit"
               for action in result.actions)
    assert not any(action.action == "execute_method_step" and action.details.get("method_id") == "known_refit"
                   for action in result.actions)
    assert any(action.action == "refit" for action in result.actions)
    assert result.observation_cost == sum(probe.cost for probe in environment.probes) <= config.observation_budget
    assert result.compute_cost <= config.compute_budget
    assert result.research_state["decision_probe_conditions"] == [plan.details["condition"]]


def test_real_zero_return_is_retained_and_saved_condition_blocks_rebuying_parameter_drift():
    environment, workspace, config = prepared(.5)
    config = replace(config, risk_learning=False, observation_budget=40, max_rounds=1)
    first = controller.CognitiveAgent(config).run(environment, research_state=workspace.to_dict())
    plan = next(action for action in first.actions if action.action == "plan_experiment"
                and action.details.get("policy") == "decision_history_contrast")
    feedback = next(action for action in first.actions if action.action == "action_feedback"
                    and action.details["action"] == "task_value")
    assert feedback.details["last_gain"] == 0.
    restored = ResearchWorkspace.from_dict(json.loads(json.dumps(first.research_state)))
    restored.incumbent = Candidate("numeric_repair", policies(.6)[1][0].program, "refit")
    restored.bank = [policies()[0][0]]
    restored.uncertainty_model = {"kind": "nominal"}
    second = controller.CognitiveAgent(config).run(environment, research_state=restored.to_dict())
    assert not any(action.action == "plan_experiment" and action.details.get("policy") == "decision_history_contrast"
                   for action in second.actions)
    assert second.research_state["decision_probe_conditions"] == [plan.details["condition"]]


def test_controller_does_not_buy_the_target_pair_when_the_two_reference_trials_cannot_fit():
    environment, workspace, config = prepared()
    result = controller.CognitiveAgent(replace(config, observation_budget=23, max_rounds=1)).run(
        environment, research_state=workspace.to_dict())
    assert not any(probe.probe_id.startswith("decision_history_") for probe in environment.probes)
    assert result.research_state["decision_probe_conditions"] == []
    assert result.observation_cost <= 23


def test_actual_successful_preparations_enter_existing_experiment_memory_and_can_be_recalled():
    environment, workspace, config = prepared()
    library = KnowledgeLibrary()
    result = controller.CognitiveAgent(replace(config, use_memory=True), library).run(
        environment, research_state=workspace.to_dict())
    learned = [action for action in result.actions if action.action == "learn_experiment"
               and action.reason == "Retain a history intervention that actually separated the frozen explanations"]
    assert len(learned) == 2 and sum(action.compute_cost for action in learned) == 48
    assert all(action.details["before_distance"] == 0. and action.details["observed_loss_gap"] > 0.
               for action in learned)
    assert len(library.experiments.records) == 2
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    work = SearchBudget(500)
    offers = restored.experiments.propose(environment.spec, [candidate for candidate, _ in policies()],
        workspace.observations, fallback=0., budget=work, remaining=24)
    assert offers and work.spent > 0
    assert all(any(probe.sequence == observed.sequence for observed in environment.probes)
               for probe in offers)
