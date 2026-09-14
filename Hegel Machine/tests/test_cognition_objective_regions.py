"""Decision regions expose local joint evidence without spending new labels."""

from copy import deepcopy
from itertools import combinations, product

import pytest

from hegel_machine.cognition.decision_experiments import _predict
from hegel_machine.cognition.models import SearchBudget, TaskSpec
from hegel_machine.cognition.objective_regions import propose_decision_regions


def spec(dimension=2):
    names = tuple("xyz"[:dimension])
    return TaskSpec("regions", "Goal-local identification", names, (), (),
        feature_bounds={name: (0., 1.) for name in names},
        context={"constraint_upper_bound": "0", "objective_direction": "minimize"})


def competing_models():
    # The additive optimum is (1/2, 1/2). The coupled explanation's is (1/3, 1/3).
    return {"degree": 2, "coefficients": [.5, -1., -1., 1., 1.],
            "interaction_challenger": {"degree": 2,
                "coefficients": [.5, -1., -1., 1., 1., 1.],
                "interactions": [[0, 1]],
                "interaction_evidence": {"decision": "inconclusive"}}}


def paid(point, objective, *, constraint=-1., status="accepted"):
    return {"observation": {"features": point, "target": constraint},
            "observation_status": status, "context": {"objective": objective}}


def test_both_candidate_minima_and_paid_feasible_winner_remain_anchors():
    task, model = spec(), competing_models()
    rows = [paid({"x": .75, "y": .25}, .1),
            paid({"x": 0., "y": 0.}, -10., constraint=1.),
            paid({"x": 1., "y": 1.}, -20., status="disputed")]
    before = deepcopy((model, rows))
    budget = SearchBudget(1000)
    points = propose_decision_regions(task, model, rows, budget)
    assert points[0] == {"x": .5, "y": .5}
    assert points[1] == pytest.approx({"x": 1/3, "y": 1/3})
    assert points[2] == rows[0]["observation"]["features"]
    assert {"x": 0., "y": 0.} not in points
    assert {"x": 1., "y": 1.} not in points
    assert (model, rows) == before
    assert len(points) <= 3 * (1 + 4 + 4)
    values = [min(_predict(task, model, point),
                  _predict(task, model["interaction_challenger"], point))
              for point in points[3:]]
    assert values == sorted(values)
    assert 0 < budget.spent <= 8 + 2 * 3 * (1 + 4 + 4)


def test_local_joint_points_distinguish_axis_equivalent_explanations():
    task = spec()
    model = {"degree": 2, "coefficients": [.5, -1., -1., 1., 1.],
             "interaction_challenger": {"degree": 2,
                 "coefficients": [.75, -1.5, -1.5, 1., 1., 1.],
                 "interactions": [[0, 1]]}}
    challenger = model["interaction_challenger"]
    axes = [{"x": value, "y": .5} for value in (0., .25, .75, 1.)]
    axes += [{"x": .5, "y": value} for value in (0., .25, .75, 1.)]
    assert all(_predict(task, model, point) == _predict(task, challenger, point)
               for point in axes)
    points = propose_decision_regions(task, model,
        [paid(point, _predict(task, model, point)) for point in axes], SearchBudget(1000))
    assert any(abs(_predict(task, model, point) -
                   _predict(task, challenger, point)) > 1e-6 for point in points)
    assert any(point["x"] != .5 and point["y"] != .5 and
               abs(point["x"] - .5) == 4/32 and
               abs(point["y"] - .5) == 4/32 for point in points)


@pytest.mark.parametrize("limit", [0, 1, 2, 9, 10, 14, 19])
def test_budget_prices_each_actual_prediction_and_never_overflows(monkeypatch, limit):
    import hegel_machine.cognition.objective_regions as module
    predictions = []
    original = module._predict

    def counted(task, explanation, point):
        predictions.append(dict(point))
        return original(task, explanation, point)

    monkeypatch.setattr(module, "_predict", counted)
    budget = SearchBudget(limit)
    model = competing_models()
    points = propose_decision_regions(spec(), model, [], budget)
    solve_cost = 8 if limit >= 10 else 0
    assert budget.spent == len(predictions) + solve_cost
    assert budget.spent <= limit
    assert all(point in predictions for point in points)
    assert len({tuple(point.items()) for point in points}) == len(points)


def test_three_dimensional_pool_keeps_all_axis_and_joint_signs_and_stays_local():
    task = spec(3)
    model = {"degree": 2, "coefficients": [.75, -1., -1., -1., 1., 1., 1.]}
    points = propose_decision_regions(task, model, [], SearchBudget(1000))
    assert len(points) == 19
    assert points[0] == {"x": .5, "y": .5, "z": .5}
    for name, sign in product(task.feature_names, (-1, 1)):
        assert {**points[0], name: .5 + sign * 4/32} in points
    for first, second in combinations(task.feature_names, 2):
        for first_sign, second_sign in product((-1, 1), repeat=2):
            assert {**points[0], first: .5 + first_sign * 4/32,
                    second: .5 + second_sign * 4/32} in points
    assert all(abs(value - .5) <= 4/32 for point in points for value in point.values())


def test_unvalidated_low_order_model_cannot_remove_disfavored_paid_winner_directions():
    task = spec()
    model = {"degree": 1, "coefficients": [0., 10., 1.]}
    winner = {"x": .5, "y": .5}
    points = propose_decision_regions(task, model, [paid(winner, .1)], SearchBudget(1000))
    negative, positive = {**winner, "x": .5 - 4/32}, {**winner, "x": .5 + 4/32}
    assert _predict(task, model, negative) < _predict(task, model, positive)
    assert negative in points and positive in points
    for x_sign, y_sign in product((-1, 1), repeat=2):
        assert {"x": .5 + x_sign * 4/32, "y": .5 + y_sign * 4/32} in points


def test_tight_budget_reaches_joint_directions_before_exhausting_axis_sweeps():
    task = spec(3)
    model = {"degree": 2, "coefficients": [.75, -1., -1., -1., 1., 1., 1.]}
    points = propose_decision_regions(task, model, [], SearchBudget(5))
    assert len(points) == 5
    assert sum(sum(value != .5 for value in point.values()) == 2 for point in points) == 3


def test_clipped_boundary_points_remain_distinct_and_within_public_bounds():
    task = spec()
    model = {"degree": 1, "coefficients": [0., 1., 1.]}
    points = propose_decision_regions(task, model, [], SearchBudget(1000))
    assert points[0] == {"x": 0., "y": 0.}
    assert len({tuple(point.items()) for point in points}) == len(points)
    assert all(0 <= value <= 4/32 for point in points for value in point.values())
    assert {"x": 4/32, "y": 4/32} in points
