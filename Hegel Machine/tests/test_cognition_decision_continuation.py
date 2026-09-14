"""Continued calibration must preserve the actual deployed decision rule."""

from copy import deepcopy

import pytest

from hegel_machine.cognition import controller
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, TaskSpec
from hegel_machine.cognition.workspace import ResearchWorkspace


@pytest.mark.parametrize("previous_offset,proposed_offset", [(1., .5), (1., 1.5), (.5, 1.)])
@pytest.mark.parametrize("calibration_available", [True, False])
def test_continued_calibration_compares_actual_old_actions_and_saves_the_selected_rule(monkeypatch, previous_offset, proposed_offset, calibration_available):
    predictor = Program("feature", name="x")
    previous = {"kind": "global_interval", "action_rule": "empirical_cost_quantile",
                "action_offset": previous_offset, "residual_interval": [previous_offset] * 2, "status": "measured_selection"}
    proposed = {**previous, "action_offset": proposed_offset, "residual_interval": [proposed_offset] * 2}

    class Environment:
        spec = TaskSpec("continued_decisions", "Predict the reading", ("x",), (), (),
                        feature_bounds={"x": (-1., 1.)}, risk_quantile=.9, decision_costs=(.9, .1))

        def observe(self, probe):
            rows, history = [], []
            for point in probe.sequence:
                target = point["x"] + 1.
                rows.append(Observation(point, target, tuple(history)))
                history.append({**point, "__target__": target})
            return rows

    environment = Environment()
    paid = Probe("paid", tuple({"x": x} for x in (-1., -.5, 0., .5, 1.)))
    rows = environment.observe(paid)
    candidate = Candidate("saved", predictor, "refit")
    workspace = ResearchWorkspace(environment.spec.task_id, observations=rows, reference=rows,
        incumbent=candidate, fallback=0., initial_loss=1., uncertainty_model=deepcopy(previous))
    monkeypatch.setattr(controller, "synthesize", lambda *args, **kwargs: [])
    received = {}

    if not calibration_available:
        choose = controller.ResearchActionPolicy.choose

        def choose_without_calibration(self, options, **kwargs):
            return None if all(option.name == "calibration" for option in options) else choose(self, options, **kwargs)

        monkeypatch.setattr(controller.ResearchActionPolicy, "choose", choose_without_calibration)

    def fit(program, train, validation, **kwargs):
        assert calibration_available, "No research should refit a retained calibration without subsequent evidence."
        received["fit_rows"] = [*train, *validation]
        return deepcopy(proposed)

    monkeypatch.setattr(controller, "fit_decision_model", fit)
    result = controller.CognitiveAgent(ResearchConfig(observation_budget=48, compute_budget=6000,
        max_rounds=1, strategy="fixed", use_memory=False)).run(environment, research_state=workspace.to_dict())
    assert result.program == predictor
    improves = calibration_available and proposed_offset == 1.
    expected = proposed if improves else previous
    assert result.uncertainty_model == expected
    assert result.research_state["uncertainty_model"] == expected
    comparison = next(action.details for action in result.actions if action.action == "compare_calibration")
    if not calibration_available:
        assert comparison["compute_cost"] == 0
        assert not received
        return
    assert comparison["before_loss"] == pytest.approx(.9 * (1. - previous_offset))
    assert (comparison["gain"] > 0.) is improves
    feedback = next(action.details for action in result.actions
                    if action.action == "action_feedback" and action.details["action"] == "calibration")
    if improves:
        assert feedback["last_gain"] > 0.
    else:
        assert feedback["last_gain"] == pytest.approx(0.)
    measured = [action for action in result.actions if action.action == "observe"
                and action.reason == "Compare complete decision plans on subsequent trajectories"]
    assert len(measured) == 2
    assert sum(action.observation_cost for action in measured) == 16
    assert len(received["fit_rows"]) == len(result.research_state["observations"]) - sum(
        action.observation_cost for action in measured)
    assert result.observation_cost <= 48 and result.compute_cost <= 6000
