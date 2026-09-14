"""Constraint-selected vector draws can test independently frozen goal models."""

from copy import deepcopy
from random import Random

import pytest

from hegel_machine.cognition.models import TaskSpec
from hegel_machine.cognition.objective_research import ObjectiveResearch


def _goal(point):
    return 2. + point["x1"] + point["x2"] ** 2


def _append(rows, point, *, changed=True, purpose="coverage"):
    objective = _goal(point) if changed else point["x2"]
    context = {"global_index": len(rows), "stage": len(rows) + 1,
               "purpose": purpose, "objective": objective}
    rows.append({"observation": {"features": deepcopy(point), "target": .2},
                 "context": context})
    return [objective, .2], context


def _frozen_shared_candidate():
    spec = TaskSpec("shared_goal_evidence", "Minimize the measured objective", ("x1", "x2"), (), (),
        feature_bounds={"x1": (0., 1.), "x2": (0., 1.)},
        context={"objective_direction": "minimize", "constraint_upper_bound": "0.0"})
    controller, rows = ObjectiveResearch(spec), []
    rng = Random(941)
    points = [{"x1": round(rng.uniform(.1, .9), 2), "x2": round(rng.uniform(.1, .9), 2)}
              for _ in range(64)]
    forecasts = []

    def forecast(point):
        forecasts.append(deepcopy(point))
        return {"features": deepcopy(point), "objective_mean": point["x2"],
                "objective_noise": 0., "feasible": False}

    for point in points[:32]:
        _append(rows, point, changed=False)
    # From the first online call, all controls belong to constraint research.
    # Goal research can fit/monitor paid data, but never selects another query.
    for point in points[32:]:
        request = controller.prepare_shared_control(rows, forecast, point,
            scope_start=0, stage=len(rows), remaining_calls=64 - len(rows), purpose="coverage")
        if request is not None:
            return controller, rows, point, request, forecast, forecasts
        output, context = _append(rows, point)
        controller.observe(point, output, context=context)
    raise AssertionError("identifiable paid constraint experiments should freeze a goal candidate")


def test_four_constraint_vectors_validate_the_goal_without_buying_another_experiment():
    controller, rows, point, request, forecast, forecasts = _frozen_shared_candidate()
    frozen = deepcopy(controller.to_dict()["scope_research"])
    assert frozen["scope_start"] == 32
    assert not controller.to_dict()["requests"]
    assert not frozen["validation"] and controller.accepted_revision is None
    assert forecasts == [point]  # No goal pool or alternate control was scanned.
    assert request["features"] == point
    initial_rows = deepcopy(rows)
    controls = [point, {"x1": .23, "x2": .45}, {"x1": .42, "x2": .61}, {"x1": .53, "x2": .67}]
    purposes = ["coverage", "validation", "applicability_probe", "applicability_confirmation"]
    for index, (point, purpose) in enumerate(zip(controls, purposes)):
        if index:
            request = controller.prepare_shared_control(rows, forecast, point,
                scope_start=len(initial_rows) if index >= 2 else 0,
                stage=len(rows), remaining_calls=64 - len(rows), purpose=purpose)
        assert request is not None and request["features"] == point
        assert controller.to_dict()["scope_research"]["candidate"] == frozen["candidate"]
        output, context = _append(rows, point, purpose=purpose)
        controller.observe(point, output, context=context)
        validation = controller.to_dict()["scope_research"]["validation"]
        assert len(validation) == index + 1
        assert validation[-1]["previous_error"] == pytest.approx((output[0] - request["previous_mean"]) ** 2)
        if index < 3:
            assert controller.accepted_revision is None
    revision = controller.accepted_revision
    assert revision is not None and len(revision["validation_indices"]) == 4
    assert set(revision["validation_indices"]).isdisjoint(revision["training_indices"])
    assert revision["training_indices"] == frozen["training_indices"]
    assert rows[:len(initial_rows)] == initial_rows and len(rows) == len(initial_rows) + 4 <= 64
    assert revision["validation_mse"] < revision["previous_mse"]
    state = controller.to_dict()
    assert len(state["shared_predictions"]) == len(state["shared_outcomes"]) == 4
    assert not state["requests"] and not state["outcomes"]
    assert state["compute_cost"] <= state["compute_budget"] == 20000
    assert {row["source"] for row in state["scope_research"]["validation"]} == {"shared_constraint_control"}


def test_shared_predictions_cannot_consume_training_or_retrospectively_supplied_labels():
    controller, rows, point, request, forecast, forecasts = _frozen_shared_candidate()
    before = controller.to_dict()
    prepared = controller.prepare_shared_control(rows, forecast, point,
        scope_start=0, stage=len(rows), remaining_calls=64 - len(rows), purpose="coverage")
    assert prepared == request and len(forecasts) == 1
    assert controller.to_dict()["compute_cost"] == before["compute_cost"]
    output = [_goal(point), .2]
    with pytest.raises(ValueError, match="next real observation"):
        controller.observe(point, output, context={"global_index": len(rows) - 1, "purpose": "coverage"})
    assert not controller.to_dict()["scope_research"]["validation"]
    # The same real row consumed twice also cannot create another validation.
    output, context = _append(rows, point)
    controller.observe(point, output, context=context)
    controller.observe(point, output, context=context)
    assert len(controller.to_dict()["scope_research"]["validation"]) == 1


def test_same_control_constraint_confirmation_does_not_make_a_second_goal_validation():
    controller, rows, point, _, forecast, forecasts = _frozen_shared_candidate()
    output, context = _append(rows, point)
    controller.observe(point, output, context=context)
    count = len(forecasts)
    request = controller.prepare_shared_control(rows, forecast, point,
        scope_start=0, stage=len(rows), remaining_calls=64 - len(rows), purpose="confirmation")
    assert request is None and len(forecasts) == count
    output, context = _append(rows, point, purpose="confirmation")
    controller.observe(point, output, context=context)
    assert len(controller.to_dict()["scope_research"]["validation"]) == 1
    assert len(controller.to_dict()["shared_outcomes"]) == 1
    assert controller.accepted_revision is None


def test_observation_without_an_earlier_shared_prediction_does_not_validate_a_candidate():
    controller, rows, point, _, _, _ = _frozen_shared_candidate()
    output, context = _append(rows, point)
    controller.observe(point, output, context=context)
    output, context = _append(rows, {"x1": .31, "x2": .54}, purpose="validation")
    controller.observe(rows[-1]["observation"]["features"], output, context=context)
    assert len(controller.to_dict()["scope_research"]["validation"]) == 1
    assert controller.accepted_revision is None
