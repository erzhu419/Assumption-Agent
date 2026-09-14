"""Persistent residuals reopen paid measurements before buying another sweep."""

from dataclasses import replace
import json

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import make_lab
from hegel_machine.cognition.measurement_research import MeasurementResearchAgent
from hegel_machine.cognition.models import ResearchConfig


def _config():
    return ResearchConfig(observation_budget=96, compute_budget=60000,
                          max_rounds=3, seed=7, use_memory=False)


def test_improving_but_persistent_residual_reuses_paid_views_then_tests_new_predictions(tmp_path):
    config = _config()
    # A short first search acquires the complete pilot readings, but cannot
    # yet discover their nonlinear state representation. The continuation has
    # the same total physical budget and a larger computation allowance.
    with make_lab(1, 68348) as laboratory:
        initial = MeasurementResearchAgent(replace(config, compute_budget=480),
            measurement_budget=384).run(laboratory, max_steps=1)
        assert initial["selected_channels"] == []
        assert len(initial["research_state"]["apparatus_acquisitions"]) == 2
        assert all(set(batch["channels"]) == {"sensor_a", "sensor_b"}
                   for batch in initial["research_state"]["batches"])
        residual = MeasurementResearchAgent(config, measurement_budget=384).run(
            laboratory, research_state=initial["research_state"], max_steps=1)

    question = residual["measurement_question"]
    assert residual["research_state"]["stage"] == "reselect"
    assert question["reason"] == "persistent_prediction_residual"
    assert config.minimum_improvement < question["current_validation_loss"] < question["previous_validation_loss"]
    assert not residual["measurement_events"]
    original_program = Program.from_dict(residual["episode"]["program"])
    original_fallback = residual["episode"]["fallback"]
    path = tmp_path / "unfinished_measurement_question.json"
    path.write_text(json.dumps(residual["research_state"]))

    with make_lab(1, 68348, backend="process") as laboratory:
        reconsidered = MeasurementResearchAgent(config, measurement_budget=384).run(
            laboratory, research_state=json.loads(path.read_text()), max_steps=1)
        event, = [item for item in reconsidered["measurement_events"]
                  if item["action"] == "recompare_measurements"]
        assert event["evidence_source"] == "acquired_trajectories"
        assert event["adopted"] and reconsidered["selected_channels"] == ["sensor_a"]
        assert event["costs"]["observation_cost"] == event["costs"]["measurement_cost"] == 0
        assert event["costs"]["compute_cost"] > 0
        assert reconsidered["research_state"]["apparatus_acquisitions"] == residual["research_state"]["apparatus_acquisitions"]
        assert reconsidered["pilot_observations"] == json.loads(json.dumps(initial["pilot_observations"]))
        assert reconsidered["research_state"]["stage"] == "research"
        assert reconsidered["costs"]["observation_cost"] == residual["costs"]["observation_cost"]
        assert reconsidered["costs"]["measurement_cost"] == residual["costs"]["measurement_cost"]
        path.write_text(json.dumps(reconsidered["research_state"]))

    with make_lab(1, 68348, backend="process") as laboratory:
        completed = MeasurementResearchAgent(config, measurement_budget=384).run(
            laboratory, research_state=json.loads(path.read_text()))
        assert completed["status"] == "completed"
        assert completed["costs"]["observation_cost"] > reconsidered["costs"]["observation_cost"]
        assert completed["costs"]["measurement_cost"] > reconsidered["costs"]["measurement_cost"]
        assert len([query for query in laboratory.acquisitions
                    if query["probe_id"] in {"instrument_pilot_0", "instrument_pilot_1"}]) == 2
        assert completed["episode"]["validation_loss"] < config.minimum_improvement
        fresh = laboratory.evaluation_rows(completed["selected_channels"], seed=719, length=32, trajectories=3)
        learned = Program.from_dict(completed["episode"]["program"])
        learned_loss = loss(learned, fresh, completed["episode"]["fallback"], .5)
        retained_loss = loss(original_program, fresh, original_fallback, .5)
        assert learned_loss < .02 and learned_loss < .1 * retained_loss
        for kind, limit in (("observation", 96), ("measurement", 384), ("compute", 60000)):
            assert completed["costs"][f"{kind}_cost"] <= limit


def test_sufficient_view_finishes_without_reopening_or_buying_unused_instruments():
    config = _config()
    with make_lab(0, 18313) as laboratory:
        agent = MeasurementResearchAgent(config, measurement_budget=384)
        initial = agent.run(laboratory, max_steps=1)
        assert initial["selected_channels"] == ["sensor_a"]
        assert initial["research_state"]["view_validation_loss"] < 1e-8
        completed = agent.run(laboratory, research_state=json.loads(json.dumps(initial["research_state"])))
        assert completed["status"] == "completed"
        assert not completed["measurement_events"] and not completed["measurement_question"]
        assert completed["episode"]["validation_loss"] < 1e-8
        assert all(query["channels"] == ["sensor_a"] for query in laboratory.acquisitions[2:])
