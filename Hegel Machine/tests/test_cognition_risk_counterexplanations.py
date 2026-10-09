"""Retain paid noisy counterexplanations and recalibrate genuinely new mechanisms."""

import json
from pathlib import Path

import pytest

from hegel_machine.cognition.evaluation import loss, predict
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.research_graph import ResearchGraph
from hegel_machine.cognition.risk_research_actions import RiskResearchActions
from hegel_machine.cognition.unified_research import UnifiedResearchAgent, _view
from hegel_machine.cognition.unified_session import laboratory_for
from test_cognition_observation_research import ActualPurchases


FIXTURE = Path(__file__).with_name("fixtures") / "risk_counterexplanation_95401901.json"


def paid_failure(*, execution_budget=240000):
    saved = json.loads(FIXTURE.read_text())
    laboratory = laboratory_for(saved["task"])
    config = ResearchConfig(observation_budget=384, compute_budget=240000,
        minimum_improvement=1e-7, max_rounds=4, active_probes=False, seed=saved["task"]["seed"])
    agent = UnifiedResearchAgent(config, measurement_budget=1536)
    agent._learn_instruments = agent._instrument_knowledge(laboratory)
    agent._revise_inputs = agent._revisable_inputs(laboratory)
    budget, actions = SearchBudget(execution_budget), []
    paid = ActualPurchases(laboratory, budget, actions)
    for item in saved["probes"]:
        paid.acquire(tuple(item["channels"]), "replay_paid_failure_controls",
            probe=Probe(item["probe_id"], tuple(item["sequence"])))
    channels = tuple(saved["parent_channels"])
    blocks = [[row for _, row in batch] for batch in paid.paired_for(channels)]
    training, validation = [row for block in blocks[:-1] for row in block], blocks[-1]
    program = Program.from_dict(saved["parent_program"])
    parent = {"candidate": Candidate("retained_paid_source", program, "retained"),
              "kind": "retained_response", "family_id": "T02", "fallback": 0.}
    # This is a fresh execution of the original diagnostic on actual readings.
    # The fixture supplies no response values, supported gains or oracle model.
    diagnostic_budget = SearchBudget(240000)
    support = agent._support(program, training, validation,
                             laboratory.spec_for(channels), diagnostic_budget)
    controller = RiskResearchActions(agent, laboratory, acquire=paid.acquire,
        paired_for=paid.paired_for, command_batches=paid.command_batches,
        view_rows=lambda rows, source, target: _view(rows, laboratory.spec_for(target).feature_names),
        transitions_for=lambda *args: [], budget=budget, actions=actions, transfer=[],
        graph=ResearchGraph(), remaining=paid.remaining, length=24)
    return agent, laboratory, paid, budget, actions, controller, parent, channels, support


def test_independently_supported_counterexplanation_becomes_an_executable_retained_incumbent():
    agent, lab, paid, budget, actions, controller, parent, channels, support = paid_failure()
    counterexample = support["structural_counterexample"]
    assert support["reason"] == "predictable_residual" and not support["adequate"]
    assert counterexample["gain"] > counterexample["margin"]
    assert support["counterexample_baseline_program"] == parent["candidate"].program.to_dict()
    assert support["counterexample_fallback"] == 0.
    result = controller.propose(parent, channels, noise_assessment=support)
    retained = next(item for item in actions if item["action"] == "retain_risk_counterexplanation")
    assert retained["selected"] and not retained["confirmed"]
    assert retained["retained_loss"] < retained["previous_loss"]
    base, correction = Program.from_dict(retained["baseline_program"]), Program.from_dict(retained["correction"])
    explanation = Program.from_dict(retained["program"])
    rows = [row for batch in paid.paired_for(channels) for _, row in batch]
    assert all(explanation.predict(row) == pytest.approx(
        predict(base, row, retained["fallback"]) + predict(correction, row, 0.)) for row in rows)
    original = result["comparison_parent"]
    assert original["candidate"].program == explanation and original["fallback"] == 0.
    assert original in result["finalists"]
    assert not result["confirmed"] and not any(item["action"] == "risk_research_feedback" for item in actions)

    fresh = paid.acquire(channels, "confirm_retained_counterexplanation", rows_count=48)
    selected, selected_loss = agent._compare_calibration(original, result["selected"], fresh,
        lab.spec_for(channels), budget, 0., actions)
    # Compare new responses directly; no fixed terminal loss or private noise
    # distribution is used as either a training target or an acceptance rule.
    assert selected_loss <= loss(explanation, fresh, 0., .9)
    assert selected_loss < loss(parent["candidate"].program, fresh, 0., .9)
    fresh_support = agent._support(selected["candidate"].program, result["training"], fresh,
        lab.spec_for(channels), budget)
    event = controller.settle(selected, fresh, channels, selected_channels=channels,
        confirmed=fresh_support["adequate"], fallback=selected.get("fallback", 0.),
        probe_id=paid.batches[-1]["probe"].probe_id)
    assert event["before_loss"] == pytest.approx(loss(explanation, fresh, 0., .9))
    if selected["candidate"].program == explanation:
        assert event["before_loss"] == event["after_loss"]
        assert event["feedback"]["last_gain"] == 0.


def test_actual_mechanism_revision_reopens_uncertainty_once_for_its_new_predictions():
    _, _, paid, budget, actions, controller, parent, channels, _ = paid_failure()
    first = controller.propose(parent, channels)
    assert first["risk_action_origin"]["action"] == "revise_uncertainty"
    # No fresh response has settled this visit; changing away from it records
    # an unresolved attempt, without manufacturing a success or failure gain.
    mechanism = controller.propose(first["selected"], first["channels"])
    assert mechanism["risk_action_origin"]["action"] == "revise_mechanism"
    rows = [row for batch in paid.paired_for(channels) for _, row in batch]
    old, new = first["selected"]["candidate"].program, mechanism["selected"]["candidate"].program
    assert max(abs(predict(old, row, 0.) - predict(new, row, 0.)) for row in rows) > 1e-7
    start = len(actions)
    recalibrated = controller.propose(mechanism["selected"], mechanism["channels"])
    reopened = next(item for item in actions[start:] if item["action"] == "choose_risk_research")
    assert "revise_uncertainty" in reopened["eligible_actions"]
    proposed = [first, mechanism, recalibrated]
    if recalibrated["risk_action_origin"]["action"] == "remeasure_inputs":
        # Genuine untried input research can precede reopened calibration in
        # the common policy. It must not consume the new calibration option.
        new = recalibrated["selected"]["candidate"].program
        recalibrated = controller.propose(recalibrated["selected"], recalibrated["channels"])
        proposed.append(recalibrated)
    assert recalibrated["risk_action_origin"]["action"] == "revise_uncertainty"
    assert recalibrated["risk_action_origin"]["baseline_program"] == new.to_dict()
    assert len(controller.uncertainty_bases) == 2
    assert all(not item["confirmed"] for item in proposed)
    assert not any(item["action"] == "risk_research_feedback" for item in actions)
    assert 0 < budget.spent <= budget.limit


def test_unfunded_further_research_keeps_the_supported_explanation_without_claiming_fresh_success():
    _, _, _, budget, actions, controller, parent, channels, support = paid_failure(execution_budget=192)
    # The retained comparison itself costs two predictions on each of 96 paid
    # rows. No budget remains to fit a tail or purchase a fresh confirmation.
    result = controller.propose(parent, channels, noise_assessment=support)
    retained = next(item for item in actions if item["action"] == "retain_risk_counterexplanation")
    assert retained["selected"]
    assert result["selected"]["candidate"].program.to_dict() == retained["program"]
    assert not result["confirmed"] and controller.pending is None
    assert budget.spent == 192
    assert not any(item["action"] == "risk_research_feedback" for item in actions)
