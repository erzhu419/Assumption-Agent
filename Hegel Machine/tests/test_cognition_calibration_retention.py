"""A newly fitted calibration must earn replacement on independent trials."""

import json
from copy import deepcopy
from statistics import mean

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.uncertainty import (
    action_and_interval,
    decision_loss,
    fit_decision_model,
    select_decision_model,
)


def trajectory(residuals, *, start=0.0):
    rows, history = [], []
    for index, residual in enumerate(residuals):
        x = start + index / 10
        rows.append(Observation({"x": x}, x + residual, tuple(history)))
        history.append({"x": x, "target": x + residual})
    return rows


def offset_model(offset):
    return {"kind": "nominal", "action_rule": "empirical_cost_quantile",
            "action_offset": 0.0, "calibration_offset": offset,
            "residual_interval": [-1.0, 1.0]}


def fitted_models():
    predictor = Program("feature", name="x")
    training = [row for start in (-2.0, 0.0, 2.0)
                for row in trajectory([0.0] * 8, start=start)]
    old = fit_decision_model(predictor, training, trajectory([0.0] * 8),
                             fallback=0.0, costs=(.5, .5), budget=SearchBudget(10000))
    candidate = fit_decision_model(predictor, training, trajectory([.8] * 8),
                                   fallback=0.0, costs=(.5, .5), budget=SearchBudget(10000))
    assert old["status"] == candidate["status"] == "measured_selection"
    assert candidate["calibration_offset"] > old["calibration_offset"] + .5
    return predictor, old, candidate


def test_actual_refit_that_improves_its_calibration_rows_can_fail_followup():
    predictor, old, candidate = fitted_models()
    # The apparent shift in the fitting block fails to recur on either new
    # trajectory. The fitted candidate's own score is not replacement evidence.
    assert candidate["calibration_decision_loss"] < candidate["validation_decision_loss"]
    followup = trajectory([-.02, .01, .02, -.01], start=-.5)
    followup += trajectory([.01, -.02, -.01, .02], start=.7)
    budget = SearchBudget(16)
    chosen, details = select_decision_model(predictor, old, candidate, followup,
                                            fallback=0, costs=(.5, .5), budget=budget)
    assert chosen == old
    assert details["gain"] < -.3
    assert details["selected"] == "previous"
    assert details["compute_cost"] == budget.spent == 16
    assert details["observation_cost"] == 0
    assert details["trajectories"] == 2


def test_repeated_real_shift_supports_new_calibration_and_fresh_decisions():
    predictor, old, candidate = fitted_models()
    followup = trajectory([.78, .81, .82, .79], start=-.5)
    followup += trajectory([.81, .78, .79, .82], start=.7)
    originals = deepcopy((old, candidate))
    chosen, details = select_decision_model(predictor, old, candidate, followup,
                                            fallback=0, costs=(.5, .5), budget=SearchBudget(16))
    assert details["adopted"]
    assert details["gain"] > details["support_margin"]
    assert chosen == candidate
    assert (old, candidate) == originals
    # Selection uses no evaluation query. These separate observations only
    # check the resulting deployed decision rule after it has been chosen.
    fresh = trajectory([.77, .83, .80, .79], start=3)
    def loss(model):
        return mean(decision_loss(action_and_interval(predictor, row, 0, model, (.5, .5))[0],
                                  row.target, (.5, .5)) for row in fresh)
    assert loss(chosen) < loss(old) / 10
    chosen["residual_interval"][0] = -99
    assert candidate == originals[1]


def test_public_decision_costs_change_which_calibration_is_supported():
    predictor = Program("feature", name="x")
    followup = trajectory([.1, .1, .1, .9]) + trajectory([.1, .1, .1, .9], start=2)
    previous, candidate = {}, offset_model(.8)
    outcomes = []
    for costs in ((.95, .05), (.05, .95)):
        chosen, details = select_decision_model(predictor, previous, candidate, followup,
                                                fallback=0, costs=costs, budget=SearchBudget(16))
        outcomes.append(chosen)
        assert details["fixed_costs"] == list(costs)
    assert outcomes == [candidate, previous]


def test_positive_average_gain_with_conflicting_trajectories_is_insufficient():
    predictor = Program("feature", name="x")
    followup = trajectory([1.0] * 12) + trajectory([.1] * 12, start=2)
    previous, candidate = {}, offset_model(1.0)
    chosen, details = select_decision_model(predictor, previous, candidate, followup,
                                            fallback=0, costs=(1., 1.), budget=SearchBudget(48))
    assert chosen == previous
    assert 0 < details["gain"] < details["support_margin"]
    # Twelve correlated rows in one trial do not manufacture twelve
    # independent trials or reduce the uncertainty of the paired gains.
    assert details["standard_error"] == pytest.approx(.9)
    assert details["trajectories"] == 2


@pytest.mark.parametrize("rows,limit,reason", [
    ([], 100, "insufficient_trajectory_support"),
    (trajectory([1.] * 4), 100, "insufficient_trajectory_support"),
    (trajectory([1.] * 40), 100, "insufficient_trajectory_support"),
    (trajectory([1.] * 4) + trajectory([1.] * 4), 15, "insufficient_budget"),
])
def test_absent_independent_support_or_budget_preserves_saved_model(rows, limit, reason):
    previous, candidate = offset_model(.2), offset_model(1.)
    saved = json.loads(json.dumps(previous))
    budget = SearchBudget(limit)
    chosen, details = select_decision_model(Program("feature", name="x"), saved, candidate, rows,
                                            fallback=0, costs=(.5, .5), budget=budget)
    assert chosen == previous
    assert chosen is not saved
    assert details["reason"] == reason
    assert details["compute_cost"] == budget.spent == 0
    assert details["before_loss"] is None
    assert json.loads(json.dumps(details)) == details


def test_nominal_candidate_can_replace_an_unhelpful_saved_offset():
    previous, candidate = offset_model(1.), {"kind": "nominal"}
    followup = trajectory([0.] * 4) + trajectory([0.] * 4, start=2)
    chosen, details = select_decision_model(Program("feature", name="x"), previous, candidate,
                                            followup, fallback=0, costs=(.5, .5), budget=SearchBudget(16))
    assert chosen == candidate
    assert details["before_loss"] == pytest.approx(.5)
    assert details["after_loss"] == 0


def test_equal_actual_decisions_retain_old_model_despite_changed_interval_metadata():
    previous, candidate = offset_model(.2), offset_model(.2)
    candidate["residual_interval"] = [-5., 10.]
    followup = trajectory([.2] * 4) + trajectory([.3] * 4, start=2)
    chosen, details = select_decision_model(Program("feature", name="x"), previous, candidate,
                                            followup, fallback=0, costs=(.9, .1), budget=SearchBudget(16))
    assert chosen == previous
    assert details["gain"] == 0
    assert not details["adopted"]
