"""Ordinary continuation selects the complete deployed decision on new trials."""

from copy import deepcopy
import json
from statistics import mean

import pytest

from hegel_machine.cognition import controller
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, TaskSpec
from hegel_machine.cognition.uncertainty import action_and_interval, decision_loss
from hegel_machine.cognition.workspace import ResearchWorkspace, observation_from_dict


class MeasuredResponse:
    spec = TaskSpec("continued_complete_decision", "Predict the instrument reading", ("x",), (), (),
                    feature_bounds={"x": (-1., 1.)}, risk_quantile=.9, decision_costs=(.9, .1))

    def __init__(self):
        self.queries, self.responses = [], []

    def observe(self, probe):
        rows, history = [], []
        for point in probe.sequence:
            target = point["x"] + 1.
            rows.append(Observation(dict(point), target, tuple(history)))
            history.append({**point, "__target__": target})
        self.queries.append(probe)
        self.responses.append(rows)
        return rows


@pytest.mark.parametrize("old_offset,adopt", [(1., False), (.2, True)])
def test_changed_program_without_calibration_method_must_earn_its_complete_decision(monkeypatch, old_offset, adopt):
    environment = MeasuredResponse()
    previous_program = Program("feature", name="x")
    challenger_program = Program("add", children=(previous_program, Program("const", value=.5)))
    previous_model = {"kind": "global_interval", "action_rule": "empirical_cost_quantile",
                      "action_offset": old_offset, "residual_interval": [old_offset] * 2,
                      "status": "measured_selection"}
    acquired = [Probe("already_acquired_left", tuple({"x": x} for x in (-1., -.7, -.4, -.1))),
                Probe("already_acquired_right", tuple({"x": x} for x in (.1, .4, .7, 1.)))]
    rows = [row for probe in acquired for row in environment.observe(probe)]
    previous = Candidate("deployed", previous_program, "refit")
    challenger = Candidate("retained_challenger", challenger_program, "refit")
    workspace = ResearchWorkspace(environment.spec.task_id, observations=rows, reference=rows,
        queried=acquired, incumbent=previous, bank=[challenger], fallback=0., initial_loss=1.,
        uncertainty_model=deepcopy(previous_model))
    environment.queries.clear()
    environment.responses.clear()

    # Isolate the two already retained executable alternatives and the route
    # where no optional calibration method was selected. Its new nominal
    # action must still beat the complete old deployment on real new probes.
    monkeypatch.setattr(controller, "synthesize", lambda *args, **kwargs: [])
    choose = controller.ResearchActionPolicy.choose

    def no_optional_calibration(self, options, **kwargs):
        if all(option.name == "calibration" for option in options):
            self.last_decision = {"selected_action": None, "reason": "no_measured_positive_return"}
            return None
        return choose(self, options, **kwargs)

    monkeypatch.setattr(controller.ResearchActionPolicy, "choose", no_optional_calibration)
    def unselected_fit(*args, **kwargs):
        raise AssertionError("The unselected calibration method must not execute")

    monkeypatch.setattr(controller, "fit_decision_model", unselected_fit)
    config = ResearchConfig(seed=43, observation_budget=48, compute_budget=9000,
        max_rounds=1, strategy="fixed", use_memory=False, use_abstractions=False,
        active_probes=False, state_search=False, joint_planning=False, use_warm_start=False)
    result = controller.CognitiveAgent(config).run(environment,
        research_state=json.loads(json.dumps(workspace.to_dict())))
    plan = next(action for action in result.actions if action.action == "plan_calibration")
    assert plan.details["selected_action"] is None
    assert any(action.action == "revise" and action.details["after_program"] == challenger_program.to_dict()
               for action in result.actions)
    acquisitions = [action for action in result.actions if action.action == "observe"]
    followup_ids = {action.details["probe_id"] for action in acquisitions
                   if action.reason == "Compare complete decision plans on subsequent trajectories"}
    followup = [block for probe, block in zip(environment.queries, environment.responses)
                if probe.probe_id in followup_ids]
    assert len(followup) == 2 and all(not block[0].history for block in followup)
    comparison = next(action for action in result.actions if action.action == "compare_decision_plan")
    assert comparison.details["trajectories"] == 2
    assert comparison.details["adopted"] is adopt
    assert comparison.details["program_changed"]
    for block, measured in zip(followup, comparison.details["trajectory_comparisons"]):
        def block_loss(program, model):
            return mean(decision_loss(action_and_interval(program, row, 0., model,
                environment.spec.decision_costs)[0], row.target, environment.spec.decision_costs) for row in block)
        assert measured["before_loss"] == pytest.approx(block_loss(previous_program, previous_model))
        assert measured["after_loss"] == pytest.approx(block_loss(challenger_program, {"kind": "nominal"}))

    expected_program = challenger_program if adopt else previous_program
    expected_model = {"kind": "nominal"} if adopt else previous_model
    assert result.program == expected_program and result.uncertainty_model == expected_model
    assert result.research_state["incumbent"]["program"] == expected_program.to_dict()
    assert result.research_state["uncertainty_model"] == expected_model
    if not adopt:
        assert any(item["program"] == challenger_program.to_dict() for item in result.research_state["bank"])
        assert any(action.action == "retain_decision_plan" for action in result.actions)
    reference = [observation_from_dict(row) for row in result.research_state["reference"]]
    assert result.validation_loss == pytest.approx(loss(result.program, reference, result.fallback, .9))
    assert result.validation_loss == pytest.approx(.45 if adopt else .9)
    assert comparison.compute_cost == comparison.details["compute_cost"] == 2 * sum(map(len, followup))
    assert result.compute_cost >= comparison.compute_cost + 2 * len(reference)
    assert result.compute_cost <= config.compute_budget
    assert result.observation_cost == sum(probe.cost for probe in environment.queries) <= config.observation_budget
