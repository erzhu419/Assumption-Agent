"""Sampling failures transfer with a measured instrument and its current query."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.condition_marginal_value import (
    CONDITION_FOLLOWUP_ACTION, condition_followup_policy,
    record_condition_information, sampling_applicability,
)
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.observed_condition_experiments import refresh_condition_sampling
from test_cognition_condition_research_returns import decide, paid, plan, state


def measured(*, slope=1., noise=(-.8, -.4, -.2, -.1, .1, .2, .4, .8)):
    spec = TaskSpec("paid_instrument", "", ("x", "meter"), (), (),
                    feature_bounds={"x": (-1., 1.)})
    rows = [Observation({"x": x, "meter": slope * x + value}, 0.)
            for x in (-1., 0., 1.) for value in noise]
    return spec, rows


def forecast(spec, rows, *, interval=(-.2, .2), point=0.):
    experiment = {**plan(interval=interval), "condition_field": "meter",
                  "fixed_control_background": {"x": point}}
    return refresh_condition_sampling(spec, rows, experiment, SearchBudget(15000))


def test_same_controls_do_not_make_a_different_instrument_or_query_equivalent():
    spec, rows = measured()
    current = forecast(spec, rows)
    different = forecast(*measured(slope=-1.))
    tail = forecast(spec, rows, interval=(.4, .7))
    noise = forecast(*measured(noise=(-.8, -.8, -.8, -.8, .8, .8, .8, .8)))
    context = current["sampling_context"]
    assert sampling_applicability(context, context) == 1.
    for alternative in (different, tail, noise):
        assert alternative["sampling_context"]["normalized_background"] == context["normalized_background"]
        assert 0. < sampling_applicability(context, alternative["sampling_context"]) < 1.
    nearby = forecast(spec, rows, point=1e-6)
    assert sampling_applicability(context, nearby["sampling_context"]) == pytest.approx(1., abs=1e-8)


def test_applicable_misses_survive_and_inapplicable_misses_fade_without_prediction_credit():
    spec, rows = measured()
    current, other = forecast(spec, rows), forecast(*measured(slope=-1.))
    question = state()
    selected = {"candidate": Candidate("response", Program.from_dict(question["program"]), "retained")}
    experience = {}
    policy = condition_followup_policy(spec, rows, selected, experience)
    for _ in range(4):
        outcome = paid(None, question, question, current)["last_outcome"]
        record_condition_information(policy, outcome)
    stored = json.loads(json.dumps(policy.action_record(CONDITION_FOLLOWUP_ACTION)))
    frozen = deepcopy(stored)
    same = decide(question, current, marginal_record=stored)
    changed = decide(question, other, marginal_record=stored)
    assert same["information_yield"] < changed["information_yield"] < 1.
    assert same["marginal_prediction_trials"] == changed["marginal_prediction_trials"] == 0
    assert same["empirical_marginal_prediction_gain"] == changed["empirical_marginal_prediction_gain"] == 0.
    assert stored == frozen and stored["information_batches"] == 4


def test_paid_forecast_context_is_invariant_to_renaming_and_positive_units():
    spec, rows = measured()
    original = forecast(spec, rows)
    renamed = replace(spec, task_id="renamed", feature_names=("command", "reading"),
        feature_bounds={"command": (3., 17.)})
    transformed = [Observation({"command": 10. + 7. * row.features["x"],
                               "reading": -30. + 11. * row.features["meter"]}, row.target)
                   for row in rows]
    experiment = {**plan(interval=(-32.2, -27.8)), "condition_field": "reading",
                  "fixed_control_background": {"command": 10.}}
    experiment["interval_exposure"][0]["condition_field"] = "reading"
    updated = refresh_condition_sampling(renamed, transformed, experiment, SearchBudget(15000))
    assert sampling_applicability(original["sampling_context"], updated["sampling_context"]) == pytest.approx(1., abs=1e-8)


def test_descriptor_uses_paid_readings_and_keeps_full_update_cost():
    spec, rows = measured()
    frozen = deepcopy(rows)
    experiment = {**plan(), "condition_field": "meter", "fixed_control_background": {"x": 0.}}
    budget = SearchBudget(15000)
    current = refresh_condition_sampling(spec, rows, experiment, budget)
    descriptor = current["sampling_context"]["reading_forecast"]
    assert descriptor["paid_rows"] == len(rows)
    assert descriptor["local_rows"] == 8
    assert descriptor["question_cdf"][0][0] < descriptor["question_cdf"][0][1]
    assert current["sampling_update_compute_cost"] == budget.spent > 2 * len(rows)
    assert current["paid_reading_rows"] == len(rows) and rows == frozen
