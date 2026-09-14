"""Fixed-cost decisions must improve through measured uncertainty models."""

from statistics import mean

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.uncertainty import action_and_interval, decision_loss, fit_decision_model


def residual_data():
    training = ([Observation({"context": -1.0}, target) for target in (-0.1, 0.0, 0.1, -0.05, 0.05, 0.0)]
                + [Observation({"context": 1.0}, target) for target in (8.0, 9.0, 10.0, 11.0, 12.0, 10.0)])
    validation = [Observation({"context": context}, target)
                  for context, target in ((-1.0, -0.03), (-1.0, 0.04), (1.0, 9.5), (1.0, 10.5))]
    return training, validation


def test_conditional_uncertainty_changes_actions_and_reduces_fixed_cost():
    training, validation = residual_data()
    predictor = Program("const", value=0.0)
    fixed_costs = (0.9, 0.1)
    budget = SearchBudget(200)
    model = fit_decision_model(predictor, training, validation, fallback=0.0,
                              costs=fixed_costs, budget=budget)
    assert model["status"] == "measured_selection"
    assert model["kind"] == "conditional_interval"
    assert model["fixed_costs"] == list(fixed_costs)
    measured = {candidate["kind"]: candidate["loss"] for candidate in model["alternatives"]}
    assert measured["conditional_interval"] < measured["global_interval"] < measured["nominal"]

    # New outcomes are not used to construct or select the interval model.
    held_out = [Observation({"context": -1.0}, 0.02), Observation({"context": 1.0}, 10.2)]
    conditional_actions = [action_and_interval(predictor, row, 0.0, model, fixed_costs)[0]
                           for row in held_out]
    nominal_actions = [action_and_interval(predictor, row, 0.0, {"kind": "nominal"}, fixed_costs)[0]
                       for row in held_out]
    assert conditional_actions[1] - conditional_actions[0] > 8
    conditional_cost = mean(decision_loss(action, row.target, fixed_costs)
                            for action, row in zip(conditional_actions, held_out))
    nominal_cost = mean(decision_loss(action, row.target, fixed_costs)
                        for action, row in zip(nominal_actions, held_out))
    assert conditional_cost < nominal_cost
    assert 0 < budget.spent <= budget.limit


def test_no_uncertainty_selection_claim_without_budget_for_validation():
    training, validation = residual_data()
    budget = SearchBudget(len(training))
    model = fit_decision_model(Program("const", value=0.0), training, validation,
                              fallback=0.0, costs=(0.9, 0.1), budget=budget)
    assert model == {"kind": "nominal", "status": "insufficient_budget"}
    assert budget.spent == budget.limit
