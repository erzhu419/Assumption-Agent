"""A lone noisy risk trajectory must not make every positive gain acceptable."""

import json
from math import sqrt
from pathlib import Path
from statistics import mean, pstdev

import pytest

from hegel_machine.cognition import controller
from hegel_machine.cognition.evaluation import point_loss, predict
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Probe, ResearchConfig, TaskSpec
from hegel_machine.cognition.risk_search import trajectory_blocks
from hegel_machine.cognition.workspace import ResearchWorkspace, observation_from_dict


FIXTURE = json.loads((Path(__file__).parent / "fixtures" /
    "cognition_v021_single_risk_trajectory_learning.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["task_id"])
def test_single_paid_risk_trajectory_keeps_dispersion_screening(monkeypatch, case):
    """Replay the first n11/f04 decision, keeping actual scoring and costs."""
    public = case["spec"]
    spec = TaskSpec(case["task_id"], "Predict the measured response",
        tuple(public["feature_names"]), (), (),
        risk_quantile=public["risk_quantile"], decision_costs=tuple(public["decision_costs"]),
        feature_bounds={name: tuple(bounds) for name, bounds in public["feature_bounds"].items()},
        max_probe_length=public["max_probe_length"])
    initial = [observation_from_dict(row) for row in case["initial_observations"]]
    fresh = [observation_from_dict(row) for row in case["fresh_observations"]]
    probe = Probe(case["probe"]["probe_id"], tuple(case["probe"]["sequence"]))
    incumbent = Program.from_dict(case["incumbent_program"])
    challenger = Program.from_dict(case["candidate_program"])
    assert len(trajectory_blocks(fresh)) == 1

    class PaidObservationReplay:
        def __init__(self):
            self.spec, self.queries = spec, []

        def observe(self, selected_probe):
            assert selected_probe == probe
            self.queries.append(selected_probe)
            return list(fresh)

    environment = PaidObservationReplay()
    # Freeze the real proposed program and the real paid query. Screening,
    # prediction, ranking, the support decision, and all accounting stay live.
    candidate = Candidate("frozen_first_challenger", challenger, case["candidate_operation"])
    workspace = ResearchWorkspace(spec.task_id, observations=initial, bank=[candidate])
    monkeypatch.setattr(controller, "synthesize", lambda *args, **kwargs: [])
    # This fixture contains only the first paid query; it cannot supply the
    # independent final calibration trials now requested by the controller.
    monkeypatch.setattr(controller, "representative_probe",
                        lambda *args, **kwargs: probe if kwargs["index"] == 0 else None)
    choose = controller.ResearchActionPolicy.choose

    def no_extra_calibration(self, options, **kwargs):
        if all(option.name == "calibration" for option in options):
            self.last_decision = {"selected_action": None, "reason": "frozen_first_comparison"}
            return None
        return choose(self, options, **kwargs)

    monkeypatch.setattr(controller.ResearchActionPolicy, "choose", no_extra_calibration)
    config = ResearchConfig(seed=case["query_seed"], observation_budget=48, compute_budget=6000,
        max_rounds=1, strategy="fixed", use_memory=False, use_abstractions=False,
        active_probes=False, synthesize_probes=False, state_search=False,
        joint_planning=False, use_warm_start=False)
    result = controller.CognitiveAgent(config).run(environment, research_state=workspace.to_dict())
    comparison = next(action for action in result.actions if action.action == "compare")
    details = comparison.details
    differences = [point_loss(predict(incumbent, row, result.fallback), row.target, spec.risk_quantile)
                   - point_loss(predict(challenger, row, result.fallback), row.target, spec.risk_quantile)
                   for row in fresh]
    gain = mean(differences)
    margin = .5 * pstdev(differences) / sqrt(len(differences))

    # Both released runs chose this best candidate and had positive gains, but
    # mistakenly set margin=gain. Real within-trajectory dispersion rejects it.
    released = case["released_comparison"]
    assert released["adopt"] and released["dispersion_margin"] == released["evidence_gain"]
    assert details["best_expression"] == challenger.describe() == released["best_expression"]
    assert details["evidence_gain"] == pytest.approx(released["evidence_gain"])
    assert 0 < gain < margin
    assert details["dispersion_margin"] == pytest.approx(margin)
    assert details["support_unit"] == "observation"
    assert not details["adopt"]
    assert result.program == incumbent
    assert comparison.compute_cost == 2 * (len(initial) + 2 * len(fresh))
    assert result.compute_cost >= comparison.compute_cost
    assert environment.queries == [probe]
    assert result.observation_cost == probe.cost
