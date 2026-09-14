"""A deployed program and calibration compete together on subsequent trials."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.uncertainty import fit_decision_model, select_decision_plan
from hegel_machine.cognition.workspace import observation_from_dict


@pytest.fixture(scope="module")
def learning_cases():
    path = Path(__file__).parent / "fixtures" / "cognition_v020_risk_continuation_learning.json"
    return json.loads(path.read_text())["tasks"]


def test_actual_noisy_revisit_does_not_replace_the_deployed_linear_decision_with_its_constant_parent(learning_cases):
    task = learning_cases[0]
    rows = [observation_from_dict(row) for row in task["observations"]]
    first = task["visits"][0]
    previous = Program.from_dict(first["program"])
    # This parent already existed as the initial empirical quantile. Neither
    # its value nor the saved calibrated linear rule used revisit responses.
    candidate = Program("const", value=first["fallback"])
    assert candidate == Program.from_dict(task["visits"][1]["program"])
    followup = [rows[index] for index in task["reference_indices"]
                if index >= first["observations_acquired"]]
    work = SearchBudget(2 * len(followup))
    program, model, details = select_decision_plan(previous, first["uncertainty_model"], candidate, {},
        followup, fallback=first["fallback"], costs=task["spec"]["decision_costs"], budget=work)
    assert details["trajectories"] == 5
    assert details["gain"] == pytest.approx(.011165521197904343)
    assert 0 < details["gain"] < details["support_margin"]
    assert sum(row["gain"] < 0 for row in details["trajectory_comparisons"]) == 3
    assert program == previous and model == first["uncertainty_model"]
    assert details["program_changed"] and details["selected"] == "previous"
    assert work.spent == details["compute_cost"] == 80


def test_actual_nonlinear_risk_revision_can_replace_the_old_complete_decision_on_later_paid_trajectories(learning_cases):
    task = learning_cases[1]
    rows = [observation_from_dict(row) for row in task["observations"]]
    first, second = task["visits"]
    previous, candidate = Program.from_dict(first["program"]), Program.from_dict(second["program"])
    prior_rows = first["observations_acquired"]
    reference = [index for index in task["reference_indices"] if index >= prior_rows]
    first_comparison_start = reference[0]
    # The candidate was fitted before the first resumed comparison. Fit its
    # residual model from that same previously acquired prefix, then exclude
    # the first comparison that originally selected this candidate program.
    fitting = SearchBudget(10000)
    candidate_model = fit_decision_model(candidate, rows[:prior_rows], rows[prior_rows:first_comparison_start],
        fallback=first["fallback"], costs=task["spec"]["decision_costs"], budget=fitting)
    followup = [rows[index] for index in reference if index >= first_comparison_start + 8]
    assert fitting.spent > 0 and candidate_model["status"] == "measured_selection"
    originals = deepcopy((first["uncertainty_model"], candidate_model))
    work = SearchBudget(2 * len(followup))
    program, model, details = select_decision_plan(previous, first["uncertainty_model"],
        candidate, candidate_model, followup, fallback=first["fallback"],
        costs=task["spec"]["decision_costs"], budget=work)
    assert details["trajectories"] == 4
    assert all(row["gain"] > 0 for row in details["trajectory_comparisons"])
    assert details["gain"] > details["support_margin"]
    assert program == candidate and model == candidate_model and details["adopted"]
    assert (first["uncertainty_model"], candidate_model) == originals
    assert work.spent == details["compute_cost"] == 64


def _trajectory(values):
    rows, history = [], []
    for value in values:
        rows.append(Observation({"x": value}, 2., tuple(history)))
        history.append({"x": value, "__target__": 2.})
    return rows


def _offset(value):
    return {"kind": "global_interval", "action_rule": "empirical_cost_quantile",
            "action_offset": value, "residual_interval": [value, value]}


def test_changed_predictor_does_not_rebind_the_old_calibration_to_the_new_program():
    previous, candidate = Program("const", value=0.), Program("const", value=2.)
    old_model = _offset(2.)
    followup = _trajectory([-1., -.5, 0., .5]) + _trajectory([.2, .4, .6, .8])
    selected, model, details = select_decision_plan(previous, old_model, candidate, {}, followup,
        fallback=0., costs=(.9, .1), budget=SearchBudget(16))
    # Both deployed actions equal 2. Applying the old offset around the new
    # predictor would invent a loss for the old plan and a spurious advantage.
    assert details["before_loss"] == details["after_loss"] == details["gain"] == 0.
    assert selected == previous and model == old_model


@pytest.mark.parametrize("single_trajectory,budget_limit,reason", [
    (True, 100, "insufficient_trajectory_support"),
    (False, 15, "insufficient_budget"),
])
def test_unfinished_comparison_preserves_both_parts_of_the_previous_decision(single_trajectory, budget_limit, reason):
    previous, candidate = Program("const", value=0.), Program("const", value=2.)
    old_model = _offset(1.)
    followup = _trajectory([-1., -.5, 0., .5])
    if not single_trajectory:
        followup += _trajectory([.2, .4, .6, .8])
    original = deepcopy(old_model)
    work = SearchBudget(budget_limit)
    selected, model, details = select_decision_plan(previous, old_model, candidate, {}, followup,
        fallback=0., costs=(.9, .1), budget=work)
    assert selected == previous and model == old_model and model is not old_model
    assert old_model == original and details["reason"] == reason
    assert details["compute_cost"] == work.spent == 0
