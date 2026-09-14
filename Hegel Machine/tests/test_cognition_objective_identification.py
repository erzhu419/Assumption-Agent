"""Paid endpoint observations must lead to a feasible distinguishing experiment."""

from copy import deepcopy
from random import Random

import pytest

from hegel_machine.cognition.models import TaskSpec
from hegel_machine.cognition.objective_research import ObjectiveResearch


def _constraint(point):
    return .02 - abs(point["x1"] - .5)


def _forecast(point):
    actual = {name: round(value, 2) for name, value in point.items()}
    constraint = _constraint(actual)
    return {"features": actual, "objective_mean": actual["x2"],
            "objective_variance": 0., "objective_noise": 0.,
            "constraint_mean": constraint, "constraint_variance": 0.,
            "feasible": constraint <= 0.}


def _quadratic(point):
    return 2. + point["x1"] ** 2 + point["x2"] ** 2


def _endpoint_equivalent(point):
    return 2. + point["x1"] + point["x2"] ** 2


def _append(rows, point, value, purpose="optimization", *, constraint=_constraint):
    index = len(rows)
    context = {"global_index": index, "stage": index + 1, "purpose": purpose, "objective": value}
    rows.append({"observation": {"features": deepcopy(point), "target": constraint(point)},
                 "context": context})
    return context


def _endpoint_scope():
    spec = TaskSpec("goal_identification", "Minimize the measured objective", ("x1", "x2"), (), (),
        feature_bounds={"x1": (0., 1.), "x2": (0., 1.)},
        context={"objective_direction": "minimize", "constraint_upper_bound": "0.0"})
    controller = ObjectiveResearch(spec, compute_budget=20000)
    rng = Random(817)
    rows = []
    for _ in range(32):
        point = {name: round(rng.uniform(.05, .95), 2) for name in spec.feature_names}
        _append(rows, point, point["x2"])
    request = controller.next_control(rows, _forecast, scope_start=0, stage=32, remaining_calls=32)
    assert request is not None and request["reason"] == "supported_goal_improvement"
    point = request["features"]
    context = _append(rows, point, point["x2"], request["purpose"])
    controller.observe(point, [point["x2"], _constraint(point)], context=context)
    scope_start = len(rows)
    for point in ({"x1": 0., "x2": .1}, {"x1": 1., "x2": .2},
                  {"x1": 0., "x2": .8}, {"x1": 1., "x2": .9}):
        assert _quadratic(point) == _endpoint_equivalent(point)
        _append(rows, point, _quadratic(point))
    return controller, rows, scope_start


def test_research_finds_a_feasible_measurement_that_distinguishes_models_identical_at_paid_endpoints():
    controller, rows, scope_start = _endpoint_scope()
    paid = deepcopy(rows)
    before = controller.to_dict()
    request = controller.next_control(rows, _forecast, scope_start=0,
        stage=len(rows), remaining_calls=64 - len(rows))
    assert request is not None and request["reason"] == "identify_objective_scope"
    point = request["features"]
    assert _forecast(point)["feasible"]
    assert 0. < point["x1"] < 1. and point["x1"] != .5
    # The experiment resolves the actual ambiguity rather than merely visiting
    # another endpoint or asserting that a numerical design has higher rank.
    assert abs(_quadratic(point) - _endpoint_equivalent(point)) > 1e-6
    assert rows == paid and controller.accepted_revision is None
    assert controller.to_dict()["scope_research"]["scope_start"] == scope_start
    assert len(controller.to_dict()["outcomes"]) == len(before["outcomes"])
    context = _append(rows, point, _quadratic(point), request["purpose"])
    controller.observe(point, [_quadratic(point), _constraint(point)], context=context)
    state = controller.to_dict()
    assert state["outcomes"][-1]["objective"] == pytest.approx(_quadratic(point))
    assert state["outcomes"][-1]["objective"] != pytest.approx(_endpoint_equivalent(point))
    assert rows[:-1] == paid and len(rows) == len(paid) + 1 <= 64
    assert 0 < state["compute_cost"] <= state["compute_budget"] == 20000


def test_one_distinguishing_measurement_does_not_end_research_before_the_relation_can_predict_new_controls():
    controller, rows, scope_start = _endpoint_scope()
    first = controller.next_control(rows, _forecast, scope_start=0,
        stage=len(rows), remaining_calls=64 - len(rows))
    point = first["features"]
    context = _append(rows, point, _quadratic(point), first["purpose"])
    controller.observe(point, [_quadratic(point), _constraint(point)], context=context)
    paid = deepcopy(rows)
    # Withholding the only informative control leaves the original ambiguity.
    # A second independently measured control is needed before treating that
    # explanation as one whose predictions have been checked out of sample.
    request = controller.next_control(rows, _forecast, scope_start=0,
        stage=len(rows), remaining_calls=64 - len(rows))
    assert request is not None and request["reason"] == "identify_objective_scope"
    assert _forecast(request["features"])["feasible"]
    assert request["features"] not in [row["observation"]["features"] for row in rows[scope_start:]]
    state = controller.to_dict()
    assert state["scope_research"]["scope_start"] == scope_start
    assert state["scope_research"]["phase"] == "collecting"
    assert not state["scope_research"]["validation"] and controller.accepted_revision is None
    assert len(rows) + 1 + 4 <= 64  # The next real probe still leaves future validation.
    assert rows == paid and state["compute_cost"] <= 20000


def test_identification_expands_from_a_measured_feasible_center_to_escape_a_single_axis():
    names = ("x1", "x2", "x3")
    spec = TaskSpec("feasible_identification", "Minimize the measured objective", names, (), (),
        feature_bounds={name: (0., 1.) for name in names},
        context={"objective_direction": "minimize", "constraint_upper_bound": "0.0"})
    controller = ObjectiveResearch(spec)
    constraint = lambda point: .4 - point["x2"]

    def forecast(point):
        actual = {name: round(value, 2) for name, value in point.items()}
        return {"features": actual, "objective_mean": actual["x2"],
                "objective_variance": 0., "objective_noise": 0.,
                "constraint_mean": constraint(actual), "constraint_variance": 0.,
                "feasible": constraint(actual) <= 0.}

    rng = Random(817)
    rows = []
    for _ in range(32):
        point = {"x1": round(rng.uniform(.05, .95), 2),
                 "x2": round(rng.uniform(.45, .95), 2),
                 "x3": round(rng.uniform(.05, .95), 2)}
        _append(rows, point, point["x2"], constraint=constraint)
    request = controller.next_control(rows, forecast, scope_start=0, stage=32, remaining_calls=32)
    if request is not None:
        point = request["features"]
        context = _append(rows, point, point["x2"], request["purpose"], constraint=constraint)
        controller.observe(point, [point["x2"], constraint(point)], context=context)
    scope_start = len(rows)
    # The first two controls have the shape of the actual development failure:
    # at the initial center, varying x1 or x3 cannot cure its infeasible x2.
    changed = [{"x1": .14, "x2": .11, "x3": .89}, {"x1": .03, "x2": .13, "x3": .61},
               {"x1": .12, "x2": .16, "x3": .81}, {"x1": .08, "x2": .19, "x3": .66}]
    objective = lambda point: 2. + sum(value ** 2 for value in point.values())
    for point in changed:
        _append(rows, point, objective(point), constraint=constraint)
    paid = deepcopy(rows)
    acquired = []
    while len(rows) < 60:
        request = controller.next_control(rows, forecast, scope_start=0,
            stage=len(rows), remaining_calls=64 - len(rows))
        if request is None or request["reason"] != "identify_objective_scope":
            break
        point = request["features"]
        assert constraint(point) <= 0.
        context = _append(rows, point, objective(point), request["purpose"], constraint=constraint)
        controller.observe(point, [objective(point), constraint(point)], context=context)
        acquired.append(point)
        if point["x1"] != changed[0]["x1"] or point["x3"] != changed[0]["x3"]:
            break
    assert acquired and any(point["x1"] != changed[0]["x1"] or point["x3"] != changed[0]["x3"]
                            for point in acquired)
    assert rows[:len(paid)] == paid and len(rows) + 4 <= 64
    assert controller.to_dict()["scope_research"]["scope_start"] == scope_start
    assert controller.to_dict()["compute_cost"] <= 20000


def test_a_newly_measured_feasible_control_can_supply_the_fourth_frozen_prediction_validation():
    controller, rows, scope_start = _endpoint_scope()
    first_validation = None
    while len(rows) < 60:
        request = controller.next_control(rows, _forecast, scope_start=0,
            stage=len(rows), remaining_calls=64 - len(rows))
        assert request is not None
        if request["reason"] == "validate_objective_scope":
            first_validation = request
            break
        point = request["features"]
        context = _append(rows, point, _quadratic(point), request["purpose"])
        controller.observe(point, [_quadratic(point), _constraint(point)], context=context)
    assert first_validation is not None
    frozen = deepcopy(controller.to_dict()["scope_research"]["candidate"])
    available = [first_validation["features"]]
    for row in rows[scope_start:]:
        point = row["observation"]["features"]
        if _forecast(point)["feasible"] and point not in available:
            available.append(deepcopy(point))
        if len(available) == 3:
            break
    assert len(available) == 3

    def forecast(point):
        prediction = _forecast(point)
        prediction["feasible"] = prediction["features"] in available
        return prediction

    request = first_validation
    for index in range(3):
        if index:
            request = controller.next_control(rows, forecast, scope_start=0,
                stage=len(rows), remaining_calls=64 - len(rows))
        assert request is not None and request["reason"] == "validate_objective_scope"
        point = request["features"]
        context = _append(rows, point, _quadratic(point), request["purpose"])
        controller.observe(point, [_quadratic(point), _constraint(point)], context=context)
    state = controller.to_dict()
    assert len(state["scope_research"]["validation"]) == 3 and controller.accepted_revision is None
    assert controller.next_control(rows, forecast, scope_start=0,
        stage=len(rows), remaining_calls=64 - len(rows)) is None
    # Ordinary optimization then supplies a useful new feasible point. Its
    # existing label must not become an unpredicted fourth validation outcome.
    new_point = {"x1": .37, "x2": .61}
    assert new_point not in state["scope_research"]["validation_pool"]
    available.append(new_point)
    ordinary = _append(rows, new_point, _quadratic(new_point))
    request = controller.next_control(rows, forecast, scope_start=0,
        stage=len(rows), remaining_calls=64 - len(rows))
    assert request is not None and request["features"] == new_point
    assert request["reason"] == "validate_objective_scope"
    state = controller.to_dict()
    assert state["scope_research"]["candidate"] == frozen
    assert len(state["scope_research"]["validation"]) == 3 and controller.accepted_revision is None
    context = _append(rows, new_point, _quadratic(new_point), request["purpose"])
    controller.observe(new_point, [_quadratic(new_point), _constraint(new_point)], context=context)
    revision = controller.accepted_revision
    assert revision is not None and len(revision["validation_indices"]) == 4
    assert ordinary["global_index"] not in revision["validation_indices"]
    assert context["global_index"] in revision["validation_indices"]
    assert not set(revision["training_indices"]) & set(revision["validation_indices"])
    assert len(rows) <= 64 and controller.to_dict()["compute_cost"] <= 20000
