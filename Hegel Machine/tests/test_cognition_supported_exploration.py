"""Research a nearby goal improvement without granting a constraint certificate."""

from copy import deepcopy

import pytest

from test_cognition_terminal_offers import _forecast, _installed_head, _offers, _paid_row


def _boundary_forecast(rows, *, constraint_mean=-.03, quantize=None):
    controls = [row["observation"]["features"] for row in rows]

    def forecast(point):
        actual = point if point in controls else (quantize(point) if quantize else point)
        result = _forecast(actual)
        if actual not in controls:
            result.update(feasible=False, constraint_mean=constraint_mean,
                          constraint_variance=.001, constraint_noise=.001, constraint_margin=.01)
        return result

    return forecast


def test_supported_local_direction_can_compete_before_it_has_a_constraint_certificate():
    controller, rows = _installed_head()
    forecast = _boundary_forecast(rows)
    before = controller.to_dict()
    offers = _offers(controller, rows, forecast)
    question, = [item for item in offers if item["purpose"] == "objective_exploration"]
    anchor = question["support_anchor"]
    point = question["features"]
    assert question["reason"] == "supported_goal_with_uncertain_constraint"
    assert question["feasible"] is False
    assert forecast(point)["feasible"] is False
    assert question["prediction"]["constraint_margin"] > 0.
    assert point["x1"] == anchor["features"]["x1"]
    assert point["x2"] > anchor["features"]["x2"]
    assert round(point["x2"] * 32) == round(anchor["features"]["x2"] * 32) + 1
    assert question["prediction"]["predicted_gain"] > question["prediction"]["gain_uncertainty"]
    assert anchor["constraint_measured_mean"] <= 0.
    assert len(offers) <= 4 and controller.state["pending"] is None
    assert controller.state["requests"] == before["requests"]
    assert controller.state["compute_cost"] > before["compute_cost"]


def test_a_favorable_nominal_far_prediction_does_not_supply_local_measurement_support():
    controller, rows = _installed_head()
    # The real apparatus maps every attempted new point to a distant location.
    forecast = _boundary_forecast(rows, quantize=lambda point: {"x1": 1., "x2": 1.})
    assert all(item["purpose"] == "objective_confirmation" for item in _offers(controller, rows, forecast))


@pytest.mark.parametrize("rebuttal", ["constraint_mean", "goal_gp", "goal_program"])
def test_nominal_constraint_or_goal_rebuttal_does_not_become_an_uncertified_experiment(rebuttal):
    controller, rows = _installed_head()
    forecast = _boundary_forecast(rows, constraint_mean=.02 if rebuttal == "constraint_mean" else -.03)
    ordinary = forecast

    def opposing_prediction(point):
        result = ordinary(point)
        if not result["feasible"]:
            if rebuttal == "goal_gp":
                result["objective_mean"] = 2.
            elif rebuttal == "goal_program" and point["x2"] > .1:
                # Only controls that worsen the unchanged fitted goal relation
                # have nominally favorable constraints; no revised coefficients.
                result["constraint_mean"] = .02
        return result

    assert all(item["purpose"] == "objective_confirmation"
               for item in _offers(controller, rows, opposing_prediction))


def test_actual_constraint_counterexample_withdraws_the_same_direction_without_losing_the_paid_row():
    controller, rows = _installed_head()
    forecast = _boundary_forecast(rows)
    question = next(item for item in _offers(controller, rows, forecast)
                    if item["purpose"] == "objective_exploration")
    request = controller.select_terminal_offer(question, stage=60)
    point = request["features"]
    row = _paid_row(point, -point["x2"], rows[-1]["context"]["global_index"] + 1, request["purpose"])
    row["observation"]["target"] = .2
    controller.observe(point, [-point["x2"], .2], context=row["context"])
    rows.append(row)
    later = _offers(controller, rows, _boundary_forecast(rows), remaining=3)
    assert all(item["features"] != point for item in later)
    assert controller.state["outcomes"][-1]["constraint"] == .2
    assert controller.state["requests"][-1] == request
    assert controller.state["pending"] is None
    assert rows[-1] == row
    assert controller.state["compute_cost"] <= controller.state["compute_budget"]


def test_quantized_neighbor_prediction_and_support_refer_to_the_actual_paid_control():
    controller, rows = _installed_head()
    forecast = _boundary_forecast(rows, quantize=lambda point: {**point, "x2": round(point["x2"], 2)})
    question = next(item for item in _offers(controller, rows, forecast)
                    if item["purpose"] == "objective_exploration")
    point = question["features"]
    assert point["x2"] == round(point["x2"], 2)
    assert question["prediction"]["model_mean"] == pytest.approx(-point["x2"], abs=1e-6)
    assert question["prediction"]["objective_mean"] == -point["x2"]
    assert round(point["x2"] * 32) - round(question["support_anchor"]["features"]["x2"] * 32) == 1
