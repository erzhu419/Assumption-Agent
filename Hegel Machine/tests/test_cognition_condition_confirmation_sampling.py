"""Only newly measured responses can resolve a frozen condition question."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.condition_confirmation import acquire_condition_confirmation
from hegel_machine.cognition.models import Observation, Probe, SearchBudget
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.observed_condition_uncertainty import assess_observed_condition_uncertainty
from hegel_machine.cognition.observed_condition_experiments import refresh_condition_sampling
from hegel_machine.cognition.condition_research_returns import consider_condition_research, record_condition_research
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_cognition_observed_condition_uncertainty import (
    READINGS, entry, observations, response, spec,
)


@pytest.mark.parametrize("informative_second_batch", [False, True])
def test_frozen_condition_buys_fresh_batches_until_actual_information_or_its_paid_limit(
        informative_second_batch):
    actual, proposed = response(.1), response()
    measured = observations(actual, READINGS)
    selected = entry(proposed)
    selected["condition_uncertainty"] = assess_observed_condition_uncertainty(
        spec(), measured, selected, SearchBudget(20000))
    old_state = deepcopy(selected["condition_uncertainty"])
    # A low-yield instrument forecast can justify several trials. The observed
    # reading remains an outcome; every request contains the public x only.
    plan = {"probe": Probe("frozen_cut", tuple({"x": 0.} for _ in range(24))),
        "expected_disagreement": .1, "interval_exposure": [{"condition_field": "meter",
            "threshold_interval": [-.2, .2], "estimated_probability": .01,
            "response_disagreement": 10.}]}
    budget, actions, purchases = SearchBudget(320000), [], []
    spent = {"observation_cost": 0, "measurement_cost": 0}

    def remaining():
        return {"observation_remaining": 288 - spent["observation_cost"],
                "measurement_remaining": 288 - spent["measurement_cost"],
                "compute_remaining": budget.remaining}

    def acquire(channels, purpose, *, probe):
        assert channels == ("meter",) and purpose == "confirm_continuation_program"
        assert all(set(point) == {"x"} for point in probe.sequence)
        assert all(probe.probe_id != old["probe"].probe_id for old in purchases)
        readings = READINGS[:24]
        if informative_second_batch and len(purchases) == 1:
            readings = [.05, .15, *READINGS[:22]]
        rows = observations(actual, readings)
        measured.extend(rows)
        purchases.append({"probe": probe, "rows": rows})
        spent["observation_cost"] += len(rows)
        spent["measurement_cost"] += len(rows)
        return rows

    last = acquire_condition_confirmation(spec(), selected, plan, budget, actions,
        acquire=acquire, remaining=remaining, rows_for=lambda channels: measured,
        channels=("meter",), measurement_rate=1, tolerance=1e-7)
    event, = actions
    assert event["planned_rows"] > 48
    assert selected["candidate"].program == proposed
    assert event["program"] == proposed.to_dict()
    assert last is purchases[-1]["rows"]
    assert event["acquired_rows"] == spent["observation_cost"] == spent["measurement_cost"]
    assert event["probe_ids"] == [item["probe"].probe_id for item in purchases]
    assert not event["outcomes"][0]["informative"]
    assert remaining()["observation_remaining"] >= event["confirmation_reserve"]
    assert remaining()["measurement_remaining"] >= event["confirmation_reserve"]
    assert budget.remaining >= event["revision_compute_reserve"]
    assert all(not item["changed_program"] for item in event["outcomes"])
    current = selected["condition_uncertainty"]
    assert current["status"] == "unresolved"
    if informative_second_batch:
        assert len(purchases) == 2
        assert event["acquired_rows"] < event["planned_rows"]
        assert event["stop_reason"] == "new_condition_evidence"
        assert event["outcomes"][-1]["interval_narrowed"]
        assert current["conditions"][0]["threshold_intervals"] == [[.05, .15]]
        assert current["conditions"][0]["needs_threshold_revision"]
        assert current["original_target_loss"] > 1e-7
    else:
        assert event["acquired_rows"] == event["planned_rows"]
        assert event["stop_reason"] == "priced_allocation_exhausted"
        assert not any(item["informative"] for item in event["outcomes"])
        assert current["conditions"][0]["threshold_intervals"] == old_state["conditions"][0]["threshold_intervals"]
        assert current["original_target_loss"] == 0.


def test_followup_calibration_keeps_new_confirmation_misses_after_older_condition_experiments(monkeypatch):
    from hegel_machine.cognition import condition_response_actions, observed_condition_experiments

    rows = observations(response(.1), READINGS)
    selected = entry(response())
    selected.update(training_loss=0., validation_loss=0., kind="conditional", fallback=0.)
    state = assess_observed_condition_uncertainty(spec(), rows, selected, SearchBudget(20000))
    selected["condition_uncertainty"] = state
    plan = {"probe": Probe("older_condition_probe", tuple({"x": 0.} for _ in range(24))),
        "condition_field": "meter", "threshold_interval": [-.2, .2],
        "threshold_intervals": [{"condition_field": "meter", "threshold_interval": [-.2, .2]}],
        "fixed_control_background": {"x": 0.}, "experiment_revision_basis": "retained_cut",
        "expected_disagreement": .01, "previous_experiment_count": 0,
        "interval_exposure": [{"condition_field": "meter", "threshold_interval": [-.2, .2],
            "estimated_probability": .001, "response_disagreement": 10.}]}
    history = record_condition_research(None, state, state, plan,
        observation_cost=24, measurement_cost=24, compute_cost=200)
    old_outcome = deepcopy(history["last_outcome"])
    misses = []
    for _ in range(2):
        history = record_condition_research(history, state, state, plan,
            observation_cost=24, measurement_cost=24, compute_cost=200)
        misses.append(deepcopy(history["last_outcome"]))
    selected["condition_research_returns"] = history
    actions = [
        {"action": "plan_competing_observed_conditions", **plan},
        {"action": "revise_competing_observed_conditions", "measured_rows": 24,
         "research_return": old_outcome},
        {"action": "plan_condition_confirmation", **plan, "probe_id": "later_confirmation"},
        {"action": "allocate_condition_confirmation", "outcomes": misses,
         "program": selected["candidate"].program.to_dict(), "acquired_rows": 48,
         "probe_ids": ["later_confirmation", "later_confirmation:batch1"]},
    ]
    # Only candidate enumeration is suppressed. The real follow-up consumer
    # reconstructs its history, prices the next experiment, and writes the
    # actual investment decision from these paid outcomes.
    monkeypatch.setattr(condition_response_actions, "_partial_response_competitors", lambda *a, **k: [])
    observed_histories = []

    def existing_plan(*args, **kwargs):
        observed_histories.append(deepcopy(kwargs["previous_experiments"]))
        return {**plan, "previous_experiment_count": len(kwargs["previous_experiments"])}

    monkeypatch.setattr(observed_condition_experiments, "plan_observed_condition_probe", existing_plan)
    agent = UnifiedResearchAgent(ResearchConfig(observation_budget=288, compute_budget=320000),
                                 measurement_budget=288)
    condition_response_actions.research_observed_condition_competition(agent, spec(),
        selected, rows, rows[-10:], [], SearchBudget(320000), actions,
        acquire=lambda *args, **kwargs: pytest.fail("the low-yield next trial should not be purchased"),
        remaining=lambda: {"observation_remaining": 96, "measurement_remaining": 96,
                           "compute_remaining": 320000}, channels=("meter",), measurement_rate=1)
    assert len(observed_histories) == 1
    assert len(observed_histories[0]) == 3
    decision = next(event for event in actions if event["action"] == "choose_condition_research_investment")
    assert decision["current_question_trials"] == 3
    assert decision["current_question_information_batches"] == 0
    assert decision["current_question_predicted_information_batches"] == pytest.approx(
        history["current_question_returns"]["predicted_information_batches"])
    assert not decision["buy"]


def test_a_paid_instrument_miss_updates_the_next_batch_forecast_before_buying_again():
    actual, proposed = response(.1), response()
    measured = observations(actual, READINGS)
    selected = entry(proposed)
    selected["condition_uncertainty"] = assess_observed_condition_uncertainty(
        spec(), measured, selected, SearchBudget(20000))
    plan = {"probe": Probe("changing_instrument", tuple({"x": 0.} for _ in range(24))),
        "condition_field": "meter", "fixed_control_background": {"x": 0.},
        "expected_disagreement": .1, "interval_exposure": [{"condition_field": "meter",
            "threshold_interval": [-.2, .2], "estimated_probability": .01,
            "response_disagreement": 10.}],
        "sampling_context": {"basis": "paid_local_control_residuals",
            "normalized_background": [.5], "neighbourhood_radius": .5}}
    plan["sampling_context"] = refresh_condition_sampling(
        spec(), measured, plan, SearchBudget(15000))["sampling_context"]
    frozen = deepcopy(plan)
    budget, actions, purchases = SearchBudget(320000), [], []

    def remaining():
        return {"observation_remaining": 216 - 24 * len(purchases),
                "measurement_remaining": 216 - 24 * len(purchases),
                "compute_remaining": budget.remaining}

    def acquire(channels, purpose, *, probe):
        assert all(point == {"x": 0.} for point in probe.sequence)
        rows, history = [], []
        for _ in probe.sequence:
            features = {"x": 0., "meter": 10.}
            target = actual.predict(Observation(features, 0., tuple(history)))
            rows.append(Observation(features, target, tuple(history)))
            history.append({**features, "__target__": target})
        purchases.append(probe)
        measured.extend(rows)
        return rows

    last = acquire_condition_confirmation(spec(), selected, plan, budget, actions,
        acquire=acquire, remaining=remaining, rows_for=lambda channels: measured,
        channels=("meter",), measurement_rate=1, tolerance=1e-7)
    event, = actions
    assert event["planned_rows"] > 24
    assert len(purchases) == 2 and event["acquired_rows"] == 48 and len(last) == 24
    assert len(measured) == len(READINGS) + 48
    first_update, update = event["sampling_updates"]
    assert first_update["updated"] and first_update["paid_reading_rows"] == len(READINGS) + 24
    assert update["updated"] and update["paid_reading_rows"] == len(measured)
    first_decision, decision = event["decisions"]
    assert first_decision["buy"]
    assert not decision["buy"]
    assert update["interval_exposure"][0]["estimated_probability"] < .01
    stale = consider_condition_research(selected["condition_uncertainty"], frozen,
        selected["condition_research_returns"], **remaining(), measurement_rate=1,
        compute_quote=15048, observation_reserve=24, measurement_reserve=24,
        compute_reserve=event["revision_compute_reserve"])
    assert stale["buy"]  # Keeping the old instrument model would buy a third batch.
    assert update["compute_cost"] > 0
    # The first successful refresh belongs to the second batch's return;
    # the final declined refresh is paid once in the enclosing budget only.
    assert budget.spent == update["compute_cost"] + sum(item["compute_cost"] for item in event["outcomes"])
    assert selected["candidate"].program == proposed
    assert selected["condition_uncertainty"]["status"] == "unresolved"
    assert plan == frozen
