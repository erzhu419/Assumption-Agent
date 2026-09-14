"""Initial calibration competes with the same predictor's nominal decision."""

from copy import deepcopy
import json
from statistics import mean

import pytest

from hegel_machine.cognition import controller
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, TaskSpec
from hegel_machine.cognition.uncertainty import action_and_interval, decision_loss, risk_calibration_quote
from hegel_machine.cognition.workspace import ResearchWorkspace, observation_from_dict


class TrajectoryResponses:
    spec = TaskSpec("initial_measured_calibration", "Predict the measured response", ("x",), (), (),
                    feature_bounds={"x": (-5., 5.)}, risk_quantile=.9, decision_costs=(.9, .1))

    def __init__(self, followup_offsets):
        self.followup_offsets = iter(followup_offsets)
        self.queries, self.responses = [], []
        self.fitted = None

    def observe(self, probe):
        final_trial = int(probe.probe_id.split("_")[-1]) >= 100
        if final_trial:
            assert self.fitted is not None  # outcomes arrive after model freezing
        offset = next(self.followup_offsets) if final_trial else .25
        rows, history = [], []
        for point in probe.sequence:
            target = point["x"] + offset
            rows.append(Observation(dict(point), target, tuple(history)))
            history.append({**point, "__target__": target})
        self.queries.append(probe)
        self.responses.append(rows)
        return rows


def initial_run(monkeypatch, offsets, *, observation_budget=96, max_rounds=1, resumed=False):
    environment = TrajectoryResponses(offsets)
    predictor = Program("feature", name="x")
    acquired = [Probe(f"already_{i}", tuple({"x": x} for x in (-5., -3., -1., 1., 3., 5.)))
                for i in range(3)]
    rows = [row for probe in acquired for row in environment.observe(probe)]
    candidate = Candidate("measured_predictor", predictor, "refit")
    workspace = ResearchWorkspace(environment.spec.task_id, observations=rows, queried=acquired, bank=[candidate])
    if resumed:
        workspace.incumbent = candidate
        workspace.uncertainty_model = {"kind": "nominal"}
        workspace.fallback, workspace.initial_loss = 0., 1.
    environment.queries.clear()
    environment.responses.clear()
    monkeypatch.setattr(controller, "synthesize", lambda *args, **kwargs: [])
    fit = controller.fit_decision_model

    def capture_fit(program, training, validation, **kwargs):
        model = fit(program, training, validation, **kwargs)
        environment.fitted = {"program": program, "model": deepcopy(model),
            "row_ids": {id(row) for row in [*training, *validation]},
            "training_rows": len(training), "validation_rows": len(validation),
            "training_trajectories": sum(not row.history for row in training),
            "compute_cost": kwargs["budget"].spent}
        return model

    monkeypatch.setattr(controller, "fit_decision_model", capture_fit)
    config = ResearchConfig(seed=47, observation_budget=observation_budget, compute_budget=9000,
        max_rounds=max_rounds, strategy="fixed", use_memory=False, use_abstractions=False,
        active_probes=False, synthesize_probes=False, state_search=False,
        joint_planning=False, use_warm_start=False)
    result = controller.CognitiveAgent(config).run(environment, research_state=workspace.to_dict())
    return environment, result, config


@pytest.mark.parametrize("offsets,adopt", [([0., 0.], False), ([.25, .25], True)])
def test_initial_calibration_needs_new_complete_decision_support(monkeypatch, offsets, adopt):
    environment, result, config = initial_run(monkeypatch, offsets)
    frozen = environment.fitted
    comparison = next(action for action in result.actions if action.action == "compare_initial_calibration")
    details = comparison.details
    assert frozen["program"] == result.program == Program("feature", name="x")
    assert frozen["model"]["status"] == "measured_selection"
    assert frozen["model"]["action_offset"] + frozen["model"]["calibration_offset"] == pytest.approx(.25)
    assert details["previous_model"] == {"kind": "nominal"}
    assert details["candidate_model"] == frozen["model"]
    assert details["adopted"] is adopt
    followup = [rows for probe, rows in zip(environment.queries, environment.responses)
                if int(probe.probe_id.split("_")[-1]) >= 100]
    assert len(followup) == 2
    assert all(id(row) not in frozen["row_ids"] for rows in followup for row in rows)
    for rows, recorded in zip(followup, details["trajectory_comparisons"]):
        for key, model in (("before_loss", details["previous_model"]), ("after_loss", frozen["model"])):
            actual = mean(decision_loss(action_and_interval(result.program, row, result.fallback,
                model, environment.spec.decision_costs)[0], row.target, environment.spec.decision_costs)
                for row in rows)
            assert recorded[key] == pytest.approx(actual)
    assert result.uncertainty_model == (frozen["model"] if adopt else {"kind": "nominal"})
    assert comparison.compute_cost == 2 * sum(map(len, followup))
    reference = [observation_from_dict(row) for row in result.research_state["reference"]]
    assert result.validation_loss == pytest.approx(loss(result.program, reference, result.fallback, .9))
    assert result.compute_cost >= frozen["compute_cost"] + comparison.compute_cost + 2 * len(reference)
    assert result.compute_cost <= config.compute_budget
    assert result.observation_cost == sum(probe.cost for probe in environment.queries) <= config.observation_budget


def test_initial_conflict_uses_reserved_pair_without_refitting(monkeypatch):
    environment, result, config = initial_run(monkeypatch, [0., .25, .25, .25], max_rounds=5)
    initial = next(action for action in result.actions if action.action == "compare_initial_calibration")
    extra = next(action for action in result.actions if action.action == "extend_decision_comparison")
    assert not initial.details["adopted"] and initial.details["trajectories"] == 2
    assert extra.details["adopted"] and extra.details["trajectories"] == 4
    assert extra.compute_cost == initial.compute_cost == 32
    assert result.uncertainty_model == environment.fitted["model"]
    # Every pre-fit newly purchased row fits inside the 64-observation learning
    # allocation. The two independent pairs use the reserved 32 observations.
    learning_cost = sum(probe.cost for probe in environment.queries
                        if int(probe.probe_id.split("_")[-1]) < 100)
    assert learning_cost <= config.observation_budget - 32
    assert result.observation_cost == learning_cost + 32 <= config.observation_budget


def test_unfunded_initial_comparison_keeps_nominal_and_exact_fitted_question(monkeypatch):
    environment, result, config = initial_run(monkeypatch, [], observation_budget=12)
    comparison = next(action for action in result.actions if action.action == "compare_initial_calibration")
    assert environment.fitted["model"]["status"] == "measured_selection"
    assert comparison.details["trajectories"] == 0 and comparison.compute_cost == 0
    assert result.uncertainty_model == {"kind": "nominal"}
    restored = ResearchWorkspace.from_dict(json.loads(json.dumps(result.research_state)))
    state = restored.decision_frontier["evidence_states"][0]
    assert state["status"] == "continue"
    assert state["previous_program"] == state["candidate_program"] == result.program.to_dict()
    assert state["previous_model"] == {"kind": "nominal"}
    assert state["candidate_model"] == environment.fitted["model"]
    assert state["comparison"]["compute_cost"] == 0
    assert result.observation_cost <= config.observation_budget


def test_calibration_quote_covers_real_fit_and_history_comparison(monkeypatch):
    environment, _, _ = initial_run(monkeypatch, [.25, .25])
    frozen = environment.fitted
    quote = risk_calibration_quote(training_rows=frozen["training_rows"],
        validation_rows=frozen["validation_rows"], training_trajectories=frozen["training_trajectories"],
        feature_count=1, history_depth=3, max_probe_length=24)
    assert frozen["compute_cost"] <= quote["fit_compute_cost"]
    assert quote["trajectory_length"] == 16
    assert quote["comparison_observation_cost"] == quote["followup_observation_cost"] == 32
    assert quote["comparison_compute_cost"] == quote["followup_compute_cost"] == 64


def test_unfunded_continuation_does_not_report_an_old_data_zero_return(monkeypatch):
    _, result, _ = initial_run(monkeypatch, [], observation_budget=12, resumed=True)
    comparison = next(action for action in result.actions if action.action == "compare_calibration")
    assert comparison.details["trajectories"] == 0
    feedback = next(action.details for action in result.actions if action.action == "continuation_feedback")
    assert feedback["execution_complete"] is False
    assert feedback["gain"] is None and feedback["normalized_gain"] is None
    assert result.research_state["decision_frontier"]["evidence_states"][0]["status"] == "continue"
