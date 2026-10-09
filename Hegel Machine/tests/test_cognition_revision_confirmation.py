"""Confirm a concrete repair against its frozen predecessor on fresh responses."""

from copy import deepcopy
import gzip
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.condition_confirmation import acquire_condition_confirmation
from hegel_machine.cognition.condition_marginal_value import (
    condition_followup_policy, record_condition_followup,
)
from hegel_machine.cognition.evaluation import loss, predict
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, Probe, SearchBudget, TaskSpec
from hegel_machine.cognition.observed_condition_experiments import plan_observed_condition_probe
from hegel_machine.cognition.observed_condition_uncertainty import assess_observed_condition_uncertainty
from test_cognition_method_revisit_opportunity import real_method_stream
from test_cognition_observed_condition_uncertainty import READINGS, entry, observations, response, spec


def _rows(batches):
    result = []
    for batch in batches:
        history = []
        for saved in batch["observations"]:
            result.append(Observation(saved["features"], saved["target"], tuple(history)))
            history.append({**saved["features"], "__target__": saved["target"]})
    return result


@pytest.fixture(scope="module")
def paid_cases():
    path = Path(__file__).parent / "fixtures/paid_confirmation_root07.json.gz"
    return json.loads(gzip.decompress(path.read_bytes()))["cases"]


@pytest.mark.parametrize("case_index", [0, 1], ids=["paid_p2", "paid_p7"])
def test_paid_refutation_keeps_the_old_law_in_confirmation_and_a_real_miss_earns_no_gain(
        paid_cases, case_index):
    case = paid_cases[case_index]
    task = TaskSpec(**case["spec"])
    paid = _rows(case["paid_batches_before_confirmation"])
    selected = {"candidate": Candidate("revised", Program.from_dict(case["program"]), "research_program"),
        "fallback": case["fallback"], "channels": tuple(case["channels"])}
    baseline = {"candidate": Candidate("retained", Program.from_dict(case["baseline_program"]), "retained"),
        "fallback": case["baseline_fallback"], "channels": tuple(case["channels"])}
    assert loss(baseline["candidate"].program, paid, baseline["fallback"], .5) > 1.
    assert loss(selected["candidate"].program, paid, selected["fallback"], .5) < 1e-7
    # The real revised threshold and the old threshold delimit the question.
    # The old law's measured failure must not remove it from this comparison.
    budget = SearchBudget(15000)
    plan = plan_observed_condition_probe(task, paid, selected, budget,
        confirmation_baseline=baseline, length=24)
    assert plan is not None
    strip, = case["diagnosis"]["changed_threshold_intervals"]
    assert plan["threshold_interval"] == strip
    assert plan["sampling_question_intervals"] == [strip]
    assert plan["expected_disagreement"] > 1e-7
    assert 0 < plan["compute_cost"] == budget.spent <= 15000
    assert {item["candidate_id"] for item in plan["competing_hypotheses"]} == {"revised", "retained"}
    assert all(item["kind"] == "response" for item in plan["competing_hypotheses"])
    assert all(set(point) == set(task.feature_bounds) for point in plan["probe"].sequence)
    assert any(strip[0] < row.features["meter"] < strip[1] for row in paid[-24:])

    # Replay the independently saved MISS on its actual requested controls.
    # It is never passed to the planner or substituted as a new probe outcome.
    fresh = _rows([case["recorded_fresh_batch"]])
    old_probe = case["recorded_fresh_batch"]["probe"]
    replay_plan = {**plan, "probe": Probe(old_probe["probe_id"], tuple(old_probe["sequence"]))}
    frozen_selected, actions, purchases = deepcopy(selected), [], []
    confirmation_budget = SearchBudget(20000)

    def acquire(channels, purpose, *, probe):
        assert channels == tuple(case["channels"])
        assert purpose == "confirm_continuation_program"
        assert probe.probe_id == old_probe["probe_id"]
        assert list(probe.sequence) == old_probe["sequence"]
        purchases.append(probe)
        return fresh

    measured = acquire_condition_confirmation(task, selected, replay_plan,
        confirmation_budget, actions, acquire=acquire,
        remaining=lambda: {"observation_remaining": 24 * (1 - len(purchases)),
                           "measurement_remaining": 24 * (1 - len(purchases)),
                           "compute_remaining": confirmation_budget.remaining},
        rows_for=lambda channels: [*paid, *fresh], channels=tuple(case["channels"]),
        measurement_rate=1, tolerance=1e-7)
    assert measured is fresh and len(purchases) == 1
    assert {key: value for key, value in selected.items()
            if key != "pair_confirmation_returns"} == frozen_selected
    # Only this recorded independent batch is funded by the fixture. Its pair
    # evaluation is paid, without inventing a new-cut information outcome.
    event = next(item for item in actions if item.get("outcomes"))
    outcome, = event["outcomes"]
    assert not outcome["distinguished"] and not outcome["response_counterexample"]
    assert selected["pair_confirmation_returns"]["batches"] == 1
    assert not any(item["action"] == "allocate_condition_confirmation" for item in actions)
    assert confirmation_budget.spent == 2 * len(fresh)
    before = loss(baseline["candidate"].program, measured, baseline["fallback"], .5)
    after = loss(selected["candidate"].program, measured, selected["fallback"], .5)
    assert before == after
    assert all(predict(baseline["candidate"].program, row, baseline["fallback"])
        == predict(selected["candidate"].program, row, selected["fallback"]) for row in measured)
    policy = condition_followup_policy(task, paid, baseline, {})
    record = record_condition_followup(policy, before_loss=before, after_loss=after,
        observation_cost=len(measured), measurement_cost=len(measured), compute_cost=2 * len(measured))
    assert record["trials"] == 1 and record["expected_gain"] == 0.


@pytest.mark.parametrize("old_lag", [1, 2], ids=["different_history", "same_ast"])
def test_reset_forecast_uses_each_fallback_and_the_new_laws_shared_response_history(old_lag):
    proposed = Program("lag", name="__target__", lag=2)
    retained = Program("lag", name="__target__", lag=old_lag)
    selected = {"candidate": Candidate("new", proposed, "research_program"), "fallback": 5.}
    baseline = {"candidate": Candidate("old", retained, "retained"), "fallback": -2.}
    rows, history = [], []
    for index in range(24):
        features = {"x": (index % 5 - 2) / 2., "meter": (index % 7 - 3) / 4.}
        row = Observation(features, 5., tuple(history))
        rows.append(row)
        history.append({**features, "__target__": 5.})
    plan = plan_observed_condition_probe(spec(), rows, selected, SearchBudget(15000),
        confirmation_baseline=baseline, length=24)
    assert plan is not None
    assert plan["confirmation_pair"]["baseline_fallback"] == -2.
    assert plan["confirmation_pair"]["fallback"] == 5.
    assert len(plan["competing_hypotheses"]) == 2
    assert plan["startup_disagreement"] == 49.
    # After its missing history is filled by the new law's actual fallback,
    # both laws predict 5. A synthetic zero startup would create another gap.
    assert plan["duration_disagreement"] == pytest.approx(49. * old_lag / 24)
    assert plan["boundary_disagreement"] == 0.
    assert plan["expected_disagreement"] == pytest.approx(49. * old_lag / 24)


@pytest.mark.parametrize("width", [3, 7])
def test_new_cut_endpoint_ambiguity_does_not_multiply_an_existing_response_offset(width):
    proposed = response(width=width, difference=20.)
    retained = Program("add", children=(proposed, Program("const", value=3.)))
    rows = observations(proposed, READINGS)
    selected = {**entry(proposed), "fallback": 0.}
    selected["condition_uncertainty"] = assess_observed_condition_uncertainty(
        spec(), rows, selected, SearchBudget(20000))
    assert selected["condition_uncertainty"]["expected_prediction_ambiguity"] > 9.
    baseline = {"candidate": Candidate("old", retained, "retained"), "fallback": 0.}
    plan = plan_observed_condition_probe(spec(), rows, selected, SearchBudget(15000),
        confirmation_baseline=baseline, length=24)
    assert plan is not None
    # Every real trajectory differs by exactly 3, including startup. Moving
    # a reading into the shared cut cannot increase that old/new difference.
    assert plan["duration_disagreement"] == pytest.approx(9.)
    assert plan["boundary_disagreement"] == pytest.approx(0.)
    assert plan["expected_disagreement"] == pytest.approx(9.)
    assert all(item["kind"] == "response" for item in plan["competing_hypotheses"])


def test_real_continuation_confirms_the_same_frozen_pair_that_earns_its_measured_return(
        real_method_stream):
    stream = real_method_stream
    previous = stream["queued"]["results"][0]["research"]
    report = stream["final"]["results"][0]["research"]
    events = report["actions"][len(previous["actions"]):]
    plans = [event for event in events if event["action"] == "plan_condition_confirmation"
             and event.get("confirmation_pair")]
    assert plans
    # The initial method repair is credited against the original incumbent;
    # later research phases are credited against their adjacent frozen law.
    first = plans[0]
    comparison = next(event for event in events
        if event["action"] == "compare_successor_method_on_fresh_response"
        and event["probe_id"] == first["probe_id"])
    pair = first["confirmation_pair"]
    for key in ("baseline_program", "baseline_fallback", "program", "fallback"):
        assert pair[key] == comparison[key]
    assert pair["baseline_program"] == previous["program"]
    purchased = next(item for item in stream["purchases"][stream["before"]:]
        if item["probe"].probe_id == first["probe_id"])
    assert comparison["before_loss"] == pytest.approx(loss(Program.from_dict(pair["baseline_program"]),
        purchased["rows"], pair["baseline_fallback"], .5))
    assert comparison["after_loss"] == pytest.approx(loss(Program.from_dict(pair["program"]),
        purchased["rows"], pair["fallback"], .5))
    for plan in plans[1:]:
        paired = next(event for event in events
            if event["action"] == "compare_method_followup_on_fresh_response"
            and event["probe_id"] == plan["probe_id"])
        for key in ("baseline_program", "baseline_fallback", "program", "fallback"):
            assert plan["confirmation_pair"][key] == paired[key]
    assert report["batches"][:len(previous["batches"])] == previous["batches"]
