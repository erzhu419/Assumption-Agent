"""Price actual information size and complete instrument work separately."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.condition_marginal_value import (
    CONDITION_FOLLOWUP_ACTION, condition_followup_policy,
    instrument_cycle_cost_estimate, record_condition_followup, record_condition_information,
    record_instrument_revisit_cost,
)
from hegel_machine.cognition.condition_research_returns import consider_condition_research
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate
from test_cognition_condition_instrument_applicability import forecast, measured
from test_cognition_condition_research_returns import decide, paid, plan, state


def marginal_policy():
    spec, rows = measured()
    selected = {"candidate": Candidate("retained_response",
        Program.from_dict(state()["program"]), "retained")}
    return condition_followup_policy(spec, rows, selected, {}), forecast(spec, rows)


def test_small_repeated_contractions_cost_less_information_than_large_contractions():
    decisions = []
    for fraction in (.01, .7):
        policy, experiment = marginal_policy()
        question = state()
        history = None
        for _ in range(5):
            left, right = question["conditions"][0]["threshold_intervals"][0]
            narrower = state(interval=(left, right - fraction * (right - left)))
            history = paid(history, question, narrower, experiment)
            record_condition_information(policy, history["last_outcome"])
            question = narrower
        # Use the remaining question's current exposure, keeping the same paid
        # instrument context and its real contraction outcomes.
        experiment["interval_exposure"][0]["threshold_interval"] = question["conditions"][0]["threshold_intervals"][0]
        record = policy.action_record(CONDITION_FOLLOWUP_ACTION)
        record_condition_followup(policy, before_loss=0., after_loss=0.,
            observation_cost=48, measurement_cost=48, compute_cost=30000)
        result = decide(question, experiment, history, marginal_record=record)
        assert result["information_yield"] == 1.
        assert result["empirical_marginal_prediction_gain"] == 0.
        assert result["expected_information_value"] > 0.
        decisions.append(result)
    assert decisions[0]["expected_information_fraction"] < decisions[1]["expected_information_fraction"]
    assert decisions[0]["predicted_response_risk_reduction"] < decisions[1]["predicted_response_risk_reduction"]


def test_current_task_information_size_survives_deleted_transfer_memory_and_json():
    question, history = state(), None
    experiment = plan(.1)
    for _ in range(8):
        left, right = question["conditions"][0]["threshold_intervals"][0]
        narrower = state(interval=(left, right - .01 * (right - left)))
        history = paid(history, question, narrower, experiment)
        question = narrower
    experiment["interval_exposure"][0]["threshold_interval"] = question["conditions"][0]["threshold_intervals"][0]
    restored = json.loads(json.dumps(history))
    decision = decide(question, experiment, restored, marginal_record={},
        observation_remaining=144, measurement_remaining=144)
    assert decision["expected_information_fraction"] == pytest.approx(1.08 / 9.)
    assert not decision["buy"]
    contradicted = {**question, "status": "contradicted", "original_target_loss": .4}
    reopened = decide(contradicted, experiment, restored,
        marginal_record={}, new_counterexample=True)
    assert reopened["buy"] and reopened["reopened"]


def test_instrument_cost_learning_uses_completed_scoped_cycle_and_excludes_paid_screen():
    policy, experiment = marginal_policy()
    reserve = dict(observation_cost=48, measurement_cost=48, compute_cost=110000)
    assert instrument_cycle_cost_estimate({}, experiment, reserve) == reserve
    record_condition_followup(policy, before_loss=0., after_loss=0.,
        observation_cost=48, measurement_cost=48, compute_cost=50000)
    record_instrument_revisit_cost(policy,
        observation_cost=48, measurement_cost=48, compute_cost=50000,
        sampling_context=experiment["sampling_context"],
        screening_compute_cost=20000)
    record = policy.action_record(CONDITION_FOLLOWUP_ACTION)
    assert record["mean_compute_cost"] == 50000  # Complete paid attribution stays intact.
    assert record["instrument_revisits"][0]["compute_cost"] == 30000
    same = instrument_cycle_cost_estimate(json.loads(json.dumps(record)), experiment, reserve)
    assert same == {**reserve, "compute_cost": 70000}
    shorter = deepcopy(record)
    shorter["instrument_revisits"][0].update(observation_cost=32, measurement_cost=0)
    still_required = instrument_cycle_cost_estimate(shorter, experiment, reserve)
    assert still_required["observation_cost"] == still_required["measurement_cost"] == 48
    changed = forecast(*measured(slope=-1.))
    distant = instrument_cycle_cost_estimate(record, changed, reserve)
    assert same["compute_cost"] < distant["compute_cost"] < reserve["compute_cost"]


def test_method_improvements_and_incomplete_cycles_do_not_teach_cheap_instrument_work():
    policy, experiment = marginal_policy()
    reserve = dict(observation_cost=48, measurement_cost=48, compute_cost=110000)
    record_condition_followup(policy, before_loss=100., after_loss=0.,
        observation_cost=24, measurement_cost=24, compute_cost=1000)
    record_condition_followup(policy, observation_cost=24, measurement_cost=24, compute_cost=1000)
    record = policy.action_record(CONDITION_FOLLOWUP_ACTION)
    assert record["trials"] == 1 and record["unscored_attempts"] == 1
    assert instrument_cycle_cost_estimate(record, experiment, reserve) == reserve


def test_expected_full_cycle_expenditure_prices_value_without_weakening_hard_reserve():
    question, experiment = state(), plan(.02)
    resources = dict(observation_remaining=480, measurement_remaining=480,
        compute_remaining=240000, measurement_rate=1, compute_quote=80000,
        observation_reserve=24, measurement_reserve=24, compute_reserve=30000)
    expected = dict(observation_cost=48, measurement_cost=48, compute_cost=30000)
    old = consider_condition_research(question, experiment, **resources,
        expected_cycle_costs={**expected, "compute_cost": 110000})
    measured = consider_condition_research(question, experiment, **resources,
        expected_cycle_costs=expected)
    assert not old["buy"] and measured["buy"]
    assert measured["resource_share"] == pytest.approx(30000 / 240000)
    assert measured["estimated_remaining_compute_cost"] == 30000
    short = consider_condition_research(question, experiment,
        **{**resources, "compute_remaining": 109999}, expected_cycle_costs=expected)
    assert not short["buy"] and short["reason"] == "leave_fresh_confirmation_affordable"
    observation_limited = consider_condition_research(question, experiment,
        **{**resources, "observation_remaining": 48}, expected_cycle_costs=expected)
    assert observation_limited["resource_share"] == 1.
    assert not observation_limited["buy"]


def test_information_size_is_scoped_to_actual_instrument_context_without_double_counting():
    policy, experiment = marginal_policy()
    question = state()
    after = state(interval=(-.2, .198))
    history = None
    for _ in range(5):
        history = paid(history, question, after, experiment)
        record_condition_information(policy, history["last_outcome"])
    record = policy.action_record(CONDITION_FOLLOWUP_ACTION)
    local = decide(after, experiment, history)
    transferred = decide(after, experiment, marginal_record=record)
    combined = decide(after, experiment, history, marginal_record=record)
    assert combined["expected_information_fraction"] == min(
        local["expected_information_fraction"], transferred["expected_information_fraction"])
    different = forecast(*measured(slope=-1.))
    changed = decide(after, different, marginal_record=record)
    assert changed["expected_information_fraction"] > transferred["expected_information_fraction"]


def test_new_fresh_narrowing_reprices_without_bypassing_value_and_cost():
    previous = {"batches": 1, "last_question": state()}
    narrower = state(interval=(-.1, .1))
    experiment = plan(.0001, interval=(-.1, .1))
    tiny = decide(narrower, experiment, previous)
    assert not tiny["buy"] and not tiny["reopened"]
    assert tiny["expected_marginal_value"] < tiny["resource_share"]
    funded = decide(narrower, plan(.1, interval=(-.1, .1)), previous)
    assert funded["buy"] and not funded["reopened"]
    contradicted = {**narrower, "status": "contradicted", "original_target_loss": .4}
    reopened = decide(contradicted, experiment, previous)
    assert reopened["buy"] and reopened["reopened"]
