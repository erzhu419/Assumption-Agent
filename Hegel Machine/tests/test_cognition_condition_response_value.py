"""Price the remaining response question from paid evidence and future costs."""

from copy import deepcopy
import gzip
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.condition_marginal_value import (
    CONDITION_FOLLOWUP_ACTION, condition_followup_policy, record_condition_information,
)
from hegel_machine.cognition.condition_research_returns import consider_condition_research
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Probe, SearchBudget, TaskSpec
from hegel_machine.cognition.object_continuation import observations
from hegel_machine.cognition.observed_condition_experiments import plan_observed_condition_probe
from hegel_machine.cognition.observed_condition_uncertainty import assess_observed_condition_uncertainty


@pytest.fixture
def root04():
    path = Path(__file__).parent / "fixtures/condition_response_value_root04.json.gz"
    case = json.loads(gzip.decompress(path.read_bytes()))
    saved = case["plan"]["probe"]
    case["plan"]["probe"] = Probe(saved["probe_id"], tuple(saved["sequence"]))
    return case


def decision(case, **overrides):
    limits = {key: value for key, value in case["resources"].items()
              if key != "compute_spent_before_decision"}
    return consider_condition_research(case["state"], case["plan"], case["previous"],
        previous_experiments=case["previous_experiments"],
        marginal_record=case["marginal_record"], **{**limits, **overrides})


def measured_information_record(case):
    specification = TaskSpec("paid_condition", "", ("a", "b", "c", "meter"), (), (),
        feature_bounds={name: (-1.5, 1.5) for name in ("a", "b", "c")})
    selected = {"candidate": Candidate("paid_response",
        Program.from_dict(case["state"]["program"]), "retained_incumbent")}
    policy = condition_followup_policy(specification, [], selected, {})
    for experiment in case["previous_experiments"]:
        record_condition_information(policy, experiment["research_return"])
    return policy.action_record(CONDITION_FOLLOWUP_ACTION)


def test_real_small_contractions_calibrate_hits_and_the_size_of_remaining_response_gain(root04):
    assert root04["state"]["fit_rows"] == 204
    assert root04["state"]["expected_prediction_ambiguity"] > 1.
    assert all(0. < item["research_return"]["relative_interval_reduction"] < .2
               for item in root04["previous_experiments"])
    root04["marginal_record"] = measured_information_record(root04)
    record = root04["marginal_record"]
    assert record["information_gain_sum"] < .3  # The measured contraction stays small.
    assert record["information_event_sum"] == 2  # Both predicted information events occurred.
    assert record.get("expected_gain", 0.) == record.get("trials", 0) == 0
    experienced = decision(root04, compute_remaining=240000)
    without_history = deepcopy(root04)
    without_history["marginal_record"] = {}
    uncalibrated = decision(without_history, compute_remaining=240000)
    assert experienced["information_yield"] == 1.  # These are real hits.
    assert 0. < experienced["expected_information_fraction"] < 1.
    assert 0. < experienced["expected_information_value"] < uncalibrated["expected_information_value"]
    assert experienced["predicted_response_risk_reduction"] > 0.
    assert not experienced["buy"] and not experienced["reopened"]
    # The same partial information can repay a less scarce pool. Its measured
    # size does not become zero merely because no prediction credit was earned.
    funded = decision(root04, compute_remaining=240000,
        observation_remaining=320, measurement_remaining=320)
    assert funded["buy"] and funded["expected_information_value"] == experienced["expected_information_value"]


def test_wide_partial_question_cannot_lend_its_hit_probability_to_current_narrow_gap(root04):
    whole = decision(root04)
    current_only = deepcopy(root04)
    current_only["plan"]["interval_exposure"] = current_only["plan"]["interval_exposure"][:1]
    current = decision(current_only)
    assert whole["probability_of_new_information"] == pytest.approx(
        current["probability_of_new_information"])
    assert .4 < whole["probability_of_new_information"] < .45
    assert root04["previous_decision"]["probability_of_new_information"] > .99
    assert whole["predicted_response_risk_reduction"] == pytest.approx(
        current["predicted_response_risk_reduction"])
    unrelated = deepcopy(root04)
    unrelated["plan"]["interval_exposure"] = unrelated["plan"]["interval_exposure"][1:]
    absent = decision(unrelated)
    assert not absent["buy"]
    assert absent["probability_of_new_information"] == 0.


def test_actual_planner_keeps_partial_candidate_without_claiming_its_wide_interval_as_current_value(root04):
    rows = observations(root04["observations"])
    specification = TaskSpec("paid_condition", "", ("a", "b", "c", "meter"), (), (),
        feature_bounds={name: (-1.5, 1.5) for name in ("a", "b", "c")}, max_probe_length=64)
    selected = {"candidate": Candidate("paid_response",
        Program.from_dict(root04["state"]["program"]), "research_program"),
        "condition_uncertainty": root04["state"]}
    partial = deepcopy(root04["partial_competitor"])
    saved = partial["candidate"]
    partial["candidate"] = Candidate(saved["candidate_id"],
        Program.from_dict(saved["program"]), saved["operation"])
    partial["condition_uncertainty"] = assess_observed_condition_uncertainty(
        specification, rows, partial, SearchBudget(5000))
    planned = plan_observed_condition_probe(specification, rows, selected, SearchBudget(15000),
        finalists=[partial], previous_experiments=root04["previous_experiments"])
    assert planned["experiment_revision_basis"] == "unmeasured_temporal_boundary"
    assert planned["paid_reading_rows"] == 204
    assert len(planned["interval_exposure"]) == 2
    assert any(item["candidate_id"] == saved["candidate_id"]
               for item in planned["competing_hypotheses"])
    # A broad local nomination cannot increase the current remaining bracket's
    # value. A new paid sampling model may select a different real background.
    current_only = {**partial, "structure_nominations": [], "partial_support": []}
    narrow = plan_observed_condition_probe(specification, rows, selected, SearchBudget(15000),
        finalists=[current_only], previous_experiments=root04["previous_experiments"])
    assert planned["boundary_disagreement"] == pytest.approx(narrow["boundary_disagreement"])
    assert planned["fixed_control_background"] == narrow["fixed_control_background"]
    assert root04["plan"]["boundary_disagreement"] > 190.


def test_paid_planning_is_subtracted_from_estimate_without_refunding_the_bill_or_future_work(root04):
    frozen = deepcopy(root04)
    unpaid = decision(root04, compute_spent_before_decision=0)
    partial = decision(root04, compute_spent_before_decision=10000)
    planned = decision(root04,
        compute_spent_before_decision=root04["resources"]["compute_spent_before_decision"])
    assert unpaid["estimated_remaining_compute_cost"] == 27017
    assert partial["estimated_remaining_compute_cost"] == 17017
    assert planned["estimated_remaining_compute_cost"] == root04["resources"]["compute_quote"]
    assert planned["resource_share"] < unpaid["resource_share"]
    assert not unpaid["buy"] and planned["buy"]
    assert root04 == frozen
    # A large paid prefix cannot buy the actual future work on credit.
    short = decision(root04, compute_spent_before_decision=50000,
        compute_remaining=root04["resources"]["compute_quote"] + root04["resources"]["compute_reserve"] - 1)
    assert not short["buy"]
    assert short["reason"] == "leave_fresh_confirmation_affordable"
    assert short["estimated_remaining_compute_cost"] >= root04["resources"]["compute_quote"]


@pytest.mark.parametrize("output_scale", [.01, 100.])
def test_output_amplitude_and_reading_units_preserve_the_same_investment(root04, output_scale):
    original = decision(root04)
    changed = deepcopy(root04)
    scale_squared = output_scale ** 2
    for question in (changed["state"], changed["previous"]["last_question"]):
        question["original_target_loss"] *= scale_squared
        question["expected_prediction_ambiguity"] *= scale_squared
        for condition in question["conditions"]:
            condition["threshold_intervals"] = [[23. + 1000. * x for x in interval]
                for interval in condition["threshold_intervals"]]
            for trial in condition.get("prediction_trials", []):
                trial["threshold_interval"] = [23. + 1000. * x for x in trial["threshold_interval"]]
                trial["expected_prediction_ambiguity"] *= scale_squared
    for item in changed["plan"]["interval_exposure"]:
        item["threshold_interval"] = [23. + 1000. * x for x in item["threshold_interval"]]
        item["response_disagreement"] *= scale_squared
    for item in changed["previous_experiments"]:
        item["threshold_interval"] = [23. + 1000. * x for x in item["threshold_interval"]]
    result = decision(changed)
    for key in ("buy", "reopened", "response_risk_known"):
        assert result[key] == original[key]
    for key in ("probability_of_new_information", "expected_information_value", "resource_share"):
        assert result[key] == pytest.approx(original[key])
    assert result["predicted_response_risk_reduction"] == pytest.approx(
        original["predicted_response_risk_reduction"] * scale_squared)


def test_easy_low_risk_condition_does_not_supply_probability_to_a_different_risky_condition(root04):
    root04["plan"]["interval_exposure"] = root04["plan"]["interval_exposure"][:1]
    root04["plan"]["interval_exposure"][0]["estimated_probability"] = .001
    root04["previous_experiments"] = []
    root04["previous"] = {}
    condition = deepcopy(root04["state"]["conditions"][0])
    condition["condition_field"] = "other_meter"
    condition["prediction_trials"][0]["expected_prediction_ambiguity"] = 1e-4
    root04["state"]["conditions"].append(condition)
    other = deepcopy(root04["plan"]["interval_exposure"][0])
    other.update(condition_field="other_meter", estimated_probability=1.)
    root04["plan"]["interval_exposure"].append(other)
    result = decision(root04)
    assert result["response_risk_known"]
    assert result["expected_information_value"] < .03
    assert not result["buy"]


def test_zero_remaining_response_risk_stops_but_unknown_risk_is_not_zero(root04):
    root04["marginal_record"] = {"expected_gain": 1., "trials": 2}
    root04["state"]["expected_prediction_ambiguity"] = 0.
    for condition in root04["state"]["conditions"]:
        for trial in condition["prediction_trials"]:
            trial["expected_prediction_ambiguity"] = 0.
    settled = decision(root04, compute_remaining=240000)
    assert root04["state"]["status"] == "unresolved"
    assert settled["response_risk_known"] and not settled["buy"]
    assert settled["reason"] == "no_remaining_targeted_response_risk"
    root04["state"]["expected_prediction_ambiguity"] = None
    for condition in root04["state"]["conditions"]:
        for trial in condition["prediction_trials"]:
            trial["expected_prediction_ambiguity"] = None
    unknown = decision(root04, compute_remaining=240000)
    assert not unknown["response_risk_known"] and unknown["buy"]
