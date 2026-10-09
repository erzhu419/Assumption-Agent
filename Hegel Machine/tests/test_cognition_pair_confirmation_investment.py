"""Buy useful fresh evidence for a frozen pair and return the first real result."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.condition_confirmation import (
    acquire_condition_confirmation, pair_confirmation_reopen_required,
)
from hegel_machine.cognition.evaluation import loss, predict
from hegel_machine.cognition.models import Observation, Probe, SearchBudget
from hegel_machine.cognition.observed_condition_uncertainty import assess_observed_condition_uncertainty
from hegel_machine.cognition.pair_confirmation_investment import (
    consider_pair_confirmation, record_pair_confirmation,
)
from test_cognition_observed_condition_uncertainty import entry, observations, response, spec


def _setup(*, reading_probability=.1, background=0., observation_limit=240, compute_limit=200000):
    new, old = response(.1, width=1), response(-.1, width=1)
    selected = {**entry(new, nested=False), "fallback": 0., "channels": ("meter",)}
    paid = observations(new, [-1., -.5, .5, 1.] * 6)
    selected["condition_uncertainty"] = assess_observed_condition_uncertainty(
        spec(), paid, selected, SearchBudget(20000))
    selected["condition_research_returns"] = {"separate_new_cut_history": True}
    pair = {"baseline_program": old.to_dict(), "baseline_fallback": 0.,
            "program": new.to_dict(), "fallback": 0., "basis": "frozen_response_predictions"}

    def plan(probe_id, point=background, probability=reading_probability):
        return {"probe": Probe(probe_id, tuple({"x": point} for _ in range(24))),
            "confirmation_pair": deepcopy(pair), "condition_field": "meter",
            "threshold_interval": [-.1, .1], "expected_disagreement": 4. * probability,
            "interval_exposure": [{"condition_field": "meter", "threshold_interval": [-.1, .1],
                "estimated_probability": probability, "response_disagreement": 4.}],
            "fixed_control_background": {"x": point}}

    return dict(selected=selected, frozen=deepcopy(selected), paid=paid, prefix=list(paid),
        old=old, new=new, plan=plan, first=plan("funded_pair"),
        budget=SearchBudget(compute_limit), actions=[], purchases=[], planning=[],
        observation_limit=observation_limit)


def _run(case, outcomes):
    def remaining():
        rows = sum(len(item["rows"]) for item in case["purchases"])
        return {"observation_remaining": case["observation_limit"] - rows,
                "measurement_remaining": case["observation_limit"] - rows,
                "compute_remaining": case["budget"].remaining}

    def acquire(channels, purpose, *, probe):
        assert channels == ("meter",) and purpose == "confirm_continuation_program"
        assert all(set(point) == {"x"} for point in probe.sequence)
        assert probe.probe_id not in {item["probe"].probe_id for item in case["purchases"]}
        assert probe.cost <= remaining()["observation_remaining"]
        index = len(case["purchases"])
        assert index < len(outcomes), "an earlier useful batch must not be overwritten by more sampling"
        reading, target_kind = outcomes[index]
        rows, history = [], []
        for point in probe.sequence:
            features = {**point, "meter": reading(point) if callable(reading) else reading}
            sample = Observation(features, 0., tuple(history))
            target = (5. if target_kind == "common_mismatch" else
                predict(case[target_kind], sample, 0.))
            rows.append(Observation(features, target, tuple(history)))
            history.append({**features, "__target__": target})
        case["purchases"].append({"probe": probe, "rows": rows})
        case["paid"].extend(rows)
        return rows

    result = acquire_condition_confirmation(spec(), case["selected"], case["first"],
        case["budget"], case["actions"], acquire=acquire, remaining=remaining,
        rows_for=lambda channels: case["paid"], channels=("meter",),
        measurement_rate=1, tolerance=1e-7)
    assert case["paid"][:len(case["prefix"])] == case["prefix"]
    assert {key: value for key, value in case["selected"].items()
            if key != "pair_confirmation_returns"} == case["frozen"]
    assert case["budget"].spent <= case["budget"].limit
    return result


def _next_plan(monkeypatch, case, *, probability=.9, point=0.):
    from hegel_machine.cognition import condition_confirmation

    def planner(task, rows, selected, budget, **kwargs):
        assert rows == case["paid"]
        assert len(rows) > len(case["prefix"])
        assert selected["candidate"].program == case["new"]
        assert kwargs["confirmation_baseline"]["candidate"].program == case["old"]
        assert kwargs["confirmation_baseline"]["fallback"] == 0.
        assert budget.spend(17)
        case["planning"].append({"paid_rows": len(rows), "compute_cost": 17})
        return case["plan"]("next_pair", point=point, probability=probability)

    monkeypatch.setattr(condition_confirmation, "plan_observed_condition_probe", planner)


def test_a_funded_miss_buys_valuable_confirmation_and_keeps_all_paid_batches(monkeypatch):
    case = _setup()
    _next_plan(monkeypatch, case)
    result = _run(case, [(.5, "new"), (0., "new")])
    assert len(case["purchases"]) == 2
    assert result is case["purchases"][1]["rows"]
    assert len(case["paid"]) == len(case["prefix"]) + 48
    assert len(case["planning"]) == 1
    event = next(item for item in case["actions"] if item.get("outcomes"))
    assert len(event["outcomes"]) == 2
    assert not event["outcomes"][0]["distinguished"]
    assert not event["outcomes"][0]["common_response_mismatch"]
    assert event["outcomes"][1]["distinguished"]
    assert event["outcomes"][1]["selected_loss"] == 0.
    assert not event["outcomes"][1]["common_response_mismatch"]
    assert event["outcomes"][1]["baseline_loss"] > 0.
    assert sum(item["observation_cost"] for item in event["outcomes"]) == 48
    assert sum(item["measurement_cost"] for item in event["outcomes"]) == 48
    assert sum(item["compute_cost"] for item in event["outcomes"]) == 96
    assert event["evaluation_compute_cost"] == 96
    assert event["planning_compute_cost"] == 17
    assert case["budget"].spent == 96 + 17
    assert case["selected"]["pair_confirmation_returns"]["batches"] == 2


def test_the_next_actual_probe_uses_a_better_background_from_all_paid_instrument_rows():
    case = _setup(background=-.75)
    # This instrument has learned reading=x from varying controls. The already
    # funded background misses; real planning can steer x into the changed cut.
    paid, history = [], []
    for reading in [-1., -.5, .5, 1.] * 6:
        features = {"x": reading, "meter": reading}
        row = Observation(features, 0., tuple(history))
        target = predict(case["new"], row, 0.)
        paid.append(Observation(features, target, tuple(history)))
        history.append({**features, "__target__": target})
    case["paid"], case["prefix"] = paid, list(paid)
    result = _run(case, [(lambda point: point["x"], "new"),
                         (lambda point: point["x"], "new")])
    assert len(case["purchases"]) == 2
    first, second = case["purchases"]
    assert first["probe"].sequence != second["probe"].sequence
    assert all(-.1 <= point["x"] < .1 for point in second["probe"].sequence)
    assert result is second["rows"]
    assert any(predict(case["new"], row, 0.) != predict(case["old"], row, 0.) for row in result)


@pytest.mark.parametrize("initial_probability,planning_calls", [(.00001, 0), (.1, 1)],
                         ids=["before_replanning", "after_replanning"])
def test_low_information_value_keeps_the_real_miss_without_buying_more(
        monkeypatch, initial_probability, planning_calls):
    case = _setup(reading_probability=initial_probability)
    _next_plan(monkeypatch, case, probability=.00001)
    result = _run(case, [(.5, "new")])
    assert len(case["purchases"]) == 1
    assert len(case["planning"]) == planning_calls
    assert case["budget"].spent == 48 + 17 * planning_calls
    assert result is case["purchases"][0]["rows"]
    event = next(item for item in case["actions"] if item.get("outcomes"))
    assert not event["outcomes"][0]["distinguished"]
    assert not event["outcomes"][0]["response_counterexample"]
    assert event["outcomes"][0]["baseline_loss"] == event["outcomes"][0]["selected_loss"] == 0.
    assert event["decisions"][-1]["buy"] is False
    assert event["decisions"][-1]["stage"] == (
        "after_replanning" if planning_calls else "before_replanning")
    assert event["decisions"][-1]["expected_information_value"] < event["decisions"][-1]["resource_share"]


@pytest.mark.parametrize("observation_limit,compute_limit", [(24, 200000), (240, 20000)],
                         ids=["observations_exhausted", "preserve_funded_settlement"])
def test_no_second_batch_is_bought_without_its_measurements_and_settlement_budget(
        monkeypatch, observation_limit, compute_limit):
    case = _setup(observation_limit=observation_limit, compute_limit=compute_limit)
    _next_plan(monkeypatch, case)
    result = _run(case, [(.5, "new")])
    assert len(case["purchases"]) == 1
    assert result is case["purchases"][0]["rows"]
    assert not case["planning"]
    assert case["budget"].spent == 48


@pytest.mark.parametrize("reading,target_kind", [(0., "new"), (0., "old"), (.5, "common_mismatch")],
                         ids=["new_wins", "new_is_refuted", "both_miss_target"])
def test_first_distinction_or_response_counterexample_returns_before_any_further_plan(
        monkeypatch, reading, target_kind):
    from hegel_machine.cognition import condition_confirmation

    case = _setup()
    monkeypatch.setattr(condition_confirmation, "plan_observed_condition_probe",
        lambda *args, **kwargs: pytest.fail("actual response evidence must return to the caller immediately"))
    result = _run(case, [(reading, target_kind)])
    assert result is case["purchases"][0]["rows"]
    assert len(case["purchases"]) == 1
    event = next(item for item in case["actions"] if item.get("outcomes"))
    outcome, = event["outcomes"]
    assert outcome["response_counterexample"]
    if target_kind == "common_mismatch":
        assert outcome["common_response_mismatch"]
        assert event["stop_reason"] == "reopen_response_model_after_common_mismatch"
        assert case["selected"]["pair_confirmation_returns"]["reopen_required"]
    else:
        assert not outcome["common_response_mismatch"]
    assert outcome["distinguished"] == (target_kind != "common_mismatch")
    if target_kind == "new":
        assert outcome["selected_loss"] == 0. < outcome["baseline_loss"]
    elif target_kind == "old":
        assert outcome["baseline_loss"] == 0. < outcome["selected_loss"]
    else:
        assert outcome["baseline_loss"] == outcome["selected_loss"] > 0.


def test_pair_reopen_receipt_is_scoped_to_the_current_probe():
    actions = [{"action": "allocate_pair_confirmation", "outcomes": [
        {"probe_id": "pair:0", "common_response_mismatch": True}]}]
    assert pair_confirmation_reopen_required(actions, "pair:0")
    assert not pair_confirmation_reopen_required(actions, "pair:1")


def test_a_high_probability_miss_changes_investment_only_for_its_same_restored_pair():
    case = _setup(reading_probability=.9)
    plan = case["first"]
    available = dict(paid_loss=0., observation_remaining=40, measurement_remaining=40,
                     compute_remaining=100000, measurement_rate=1, compute_quote=48)
    assert consider_pair_confirmation(plan, **available)["buy"]
    miss = observations(case["new"], [.5] * 24)
    history = record_pair_confirmation(None, plan, miss, SearchBudget(1000), measurement_rate=1)
    restored = json.loads(json.dumps(history))
    reconsidered = consider_pair_confirmation(plan, restored, **available)
    assert not reconsidered["buy"]
    assert reconsidered["sampling_reliability"] < 1.
    other = deepcopy(plan)
    other["confirmation_pair"]["baseline_program"] = response(-.2, width=1).to_dict()
    other["threshold_interval"] = [-.2, .1]
    other["interval_exposure"][0]["threshold_interval"] = [-.2, .1]
    different_pair = consider_pair_confirmation(other, restored, **available)
    assert different_pair["buy"] and different_pair["current_pair_trials"] == 0
    assert different_pair["sampling_reliability"] == 1.
    assert history == restored  # Repricing does not consume or alter paid evidence.


def test_likely_separation_is_not_worth_buying_when_it_addresses_little_paid_response_error():
    case = _setup(reading_probability=.2)
    available = dict(observation_remaining=240, measurement_remaining=240,
                     compute_remaining=100000, measurement_rate=1, compute_quote=48)
    fit = consider_pair_confirmation(case["first"], paid_loss=0., **available)
    measured = [Observation({"x": 0., "meter": .5}, 12.) for _ in range(24)]
    paid_error = loss(case["new"], measured, 0., .5)
    mismatched = consider_pair_confirmation(case["first"], paid_loss=paid_error, **available)
    assert fit["probability_of_new_information"] == mismatched["probability_of_new_information"]
    assert fit["buy"] and not mismatched["buy"]
    assert mismatched["response_risk_fraction"] < fit["response_risk_fraction"]
