"""The legacy chooser must distinguish evidence gaps from fitted zero error."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.action_policy import ResearchActionPolicy
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.program_experiment_methods import ProgramExperimentMethods
from hegel_machine.cognition.temporal_research_choice import choose_temporal_research
from scripts.run_temporal_legacy_method_llm import prepare_episode


@pytest.fixture
def episode(tmp_path):
    agent, lab = prepare_episode(tmp_path, dict(case="event", stage=0, seed=1154,
        center=.21, half_width=1e-4, response_gain=2.),
        InquiryLLMConfig(observation_budget=240, confirmation_rows=48, compute_budget=100000))
    instrument = json.loads((tmp_path / "instrument.json").read_text())
    hypotheses = json.loads((tmp_path / "hypotheses.json").read_text())
    return agent, lab, instrument, hypotheses


def choose(episode, *, records=None, hypotheses=None, **kwargs):
    agent, lab, instrument, original = episode
    return choose_temporal_research(agent.observations if records is None else records,
        original if hypotheses is None else hypotheses, lab.public_spec(), instrument=instrument,
        methods=ProgramExperimentMethods(), observation_remaining=kwargs.pop("observation_remaining", 168),
        compute_remaining=kwargs.pop("compute_remaining", 50000), confirmation_rows=48, **kwargs)


def test_paid_zero_tie_uses_existing_policy_and_original_disagreement_planner(episode, monkeypatch):
    agent, lab, instrument, _ = episode
    before = deepcopy(instrument), deepcopy(lab.costs)
    calls, old = [], ResearchActionPolicy.choose

    def wrapped(self, options, **kwargs):
        calls.append([option.name for option in options])
        return old(self, options, **kwargs)

    monkeypatch.setattr(ResearchActionPolicy, "choose", wrapped)
    experience = {}
    result = choose(episode, experience=experience)
    assert result["diagnosis"] == "response_ambiguity"
    assert calls == [["discriminate_response"]]
    assert result["recommendation"]["operator"] == "response_disagreement"
    assert len(result["hypotheses"]) == 5
    assert all(item["mse"] == 0. and item["rows"] == 72 for item in result["hypotheses"])
    assert result["response_plan"]["forecast_status"] == "predicted_distinction"
    assert result["response_plan"]["predicted_separation"] > 0.
    assert result["response_plan"]["proposal_source"] == "program_disagreement"
    assert result["decision"]["quoted_observation_cost"] == 52
    assert result["compute_cost"] > 72 * 5
    assert result["compute_cost"] <= 50000
    assert (instrument, lab.costs) == before
    # Reading the policy does not fabricate a trial or measured return.
    assert all(not actions for actions in experience.values())
    assert not agent.confirmations


def test_actual_discriminating_observation_retains_unique_candidate_without_refit(episode):
    agent, lab, _, _ = episode
    first = choose(episode)
    agent._execute("query", {"actions": [first["response_plan"]["action"]]}, lab)
    result = choose(episode)
    assert result["diagnosis"] == "one_response_survives_paid_evidence"
    assert result["recommendation"]["operator"] == "retain_response"
    assert result["recommendation"]["path"] == "models/candidate_03.py"
    assert sum(item["status"] == "fits_paid_evidence" for item in result["hypotheses"]) == 1
    assert result["response_plan"] is None
    assert result["decision"]["quoted_observation_cost"] == 48
    assert agent.model is None and not agent.confirmations


def test_paid_calibration_counterexample_precedes_response_adoption(episode):
    agent, _, instrument, _ = episode
    records = deepcopy(agent.observations)
    # A single measured error can disappear in the average. The old assessor's
    # worst-row criterion must still trigger calibration repair.
    records[0]["measurements"][0]["values"]["meter"] += (36e-10) ** .5
    before = deepcopy(instrument)
    result = choose(episode, records=records)
    assert result["paid_meter_status"] == "counterexample"
    assert result["diagnosis"] == "instrument_counterexample"
    assert result["recommendation"]["operator"] == "instrument"
    assert result["response_plan"] is None
    assert result["paid_meter_assessment"]["field_losses"]["meter"]
    assert instrument == before


def test_discrimination_reserves_the_full_final_confirmation(episode):
    result = choose(episode, observation_remaining=51)
    assert result["response_plan"]["action"] is not None
    assert result["recommendation"] is None
    assert result["decision"]["reason"] == "no_affordable_action"
    assert result["decision"]["options"][0]["observation_cost"] == 52


def test_response_failures_revise_law_only_when_paid_calibration_is_consistent(episode):
    agent, _, _, candidates = episode
    first = choose(episode)
    agent._execute("query", {"actions": [first["response_plan"]["action"]]}, episode[1])
    result = choose(episode, hypotheses=[candidates[-1]])
    assert result["paid_meter_status"] == "consistent"
    assert result["diagnosis"] == "response_counterexample"
    assert result["recommendation"]["operator"] == "response"


def test_missing_readings_do_not_become_simulated_paid_response_evidence(episode):
    records = deepcopy(episode[0].observations)
    for record in records:
        record["measurements"] = []
    result = choose(episode, records=records)
    assert result["diagnosis"] == "response_evidence_unavailable"
    assert result["recommendation"] is None
    assert result["hypotheses"][-1]["status"] == "fits_paid_evidence"
    assert all(item["status"] == "unassessed" for item in result["hypotheses"][:-1])


def test_paid_comparison_cannot_overrun_compute_or_confirm_partial_candidate_set(episode):
    result = choose(episode, compute_remaining=73)
    assert result["diagnosis"] == "insufficient_compute_for_paid_comparison"
    assert result["compute_cost"] == 73
    assert result["recommendation"] is None


def test_existing_measured_policy_returns_can_decline_an_unprofitable_direction(episode):
    context = "squared:numeric:single:temporal:response_ambiguity"
    experience = {context: {"discriminate_response": {
        "trials": 2, "expected_gain": -.3, "mean_observation_cost": 52,
        "mean_compute_cost": 600}}}
    before = deepcopy(experience[context])
    result = choose(episode, experience=experience)
    assert result["decision"]["reason"] == "no_measured_positive_return"
    assert result["recommendation"] is None
    assert experience[context] == before


def test_retention_reserves_finish_scoring_of_paid_rows_as_well_as_fresh_confirmation(episode):
    agent, lab, _, _ = episode
    first = choose(episode)
    agent._execute("query", {"actions": [first["response_plan"]["action"]]}, lab)
    affordable = choose(episode)
    assert affordable["diagnosis"] == "one_response_survives_paid_evidence"
    assert affordable["decision"]["quoted_compute_cost"] == 76 + 48
    # Forty-eight remaining units would pay only for new confirmation. Finish
    # first evaluates all 76 paid rows, and must not be recommended without it.
    insufficient = choose(episode, compute_remaining=affordable["compute_cost"] + 48)
    assert insufficient["recommendation"] is None
    assert insufficient["decision"]["reason"] == "no_affordable_action"


def test_selected_records_still_reserve_scoring_all_rows_at_finish(episode):
    agent, lab, _, _ = episode
    first = choose(episode)
    agent._execute("query", {"actions": [first["response_plan"]["action"]]}, lab)
    subset = [agent.observations[-1]]
    available = choose(episode, records=subset, finish_scoring_rows=len(agent.rows))
    assert available["diagnosis"] == "one_response_survives_paid_evidence"
    assert available["paid_rows"] == 4
    assert available["decision"]["quoted_compute_cost"] == 76 + 48
    insufficient = choose(episode, records=subset, finish_scoring_rows=len(agent.rows),
        compute_remaining=available["compute_cost"] + 4 + 48)
    assert insufficient["recommendation"] is None
    assert insufficient["decision"]["reason"] == "no_affordable_action"
