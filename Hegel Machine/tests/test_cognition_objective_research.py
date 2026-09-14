"""Goal experiments earn their queries from paid evidence and public forecasts."""

from copy import deepcopy
import json
from random import Random

import pytest

from hegel_machine.cognition.models import TaskSpec
from hegel_machine.cognition.objective_research import ObjectiveResearch


def _spec():
    return TaskSpec("independent_goal", "Minimize a measured objective", ("x1", "x2"), (), (),
        feature_bounds={"x1": (0., 1.), "x2": (0., 1.)},
        context={"objective_direction": "minimize", "constraint_upper_bound": "0.0"})


def _rows(objective, *, offset=32, count=16):
    rng = Random(615)
    points = [{"x1": rng.uniform(.05, .95), "x2": rng.uniform(.05, .95)}
              for _ in range(count)]
    return [{"observation": {"features": point, "target": -1.},
             "context": {"objective": objective(point), "global_index": offset + index,
                         "stage": offset + index + 1}}
            for index, point in enumerate(points)]


def _forecast(objective):
    # This is a frozen public predictor, not an environment oracle. Tests vary
    # it independently of the measured objective supplied in _rows.
    def forecast(point):
        return {"features": deepcopy(point), "objective_mean": objective(point),
                "objective_variance": 0., "objective_noise": 0.,
                "constraint_mean": -1., "constraint_variance": 0., "feasible": True}
    return forecast


def _request(controller, rows, forecast, *, scope=32, remaining=16):
    return controller.next_control(rows, forecast, scope_start=scope,
        stage=rows[-1]["context"]["stage"], remaining_calls=remaining)


def test_paid_objective_direction_changes_an_independent_experiment_without_constraint_counterexamples():
    controls = []
    for direction in (1., -1.):
        objective = lambda point: direction * point["x2"]
        rows = _rows(objective)
        before = deepcopy(rows)
        controller = ObjectiveResearch(_spec())
        request = _request(controller, rows, _forecast(objective))
        assert request is not None and request["purpose"] == "objective_exploration"
        assert objective(request["features"]) < min(row["context"]["objective"] for row in rows)
        assert rows == before
        controls.append(request["features"])
    assert controls[0]["x2"] == pytest.approx(0., abs=.01)
    assert controls[1]["x2"] == pytest.approx(1., abs=.01)


def test_a_goal_model_alone_cannot_force_a_query_when_the_current_predictor_refutes_its_improvement():
    controller = ObjectiveResearch(_spec())
    rows = _rows(lambda point: point["x2"])
    assert _request(controller, rows, _forecast(lambda point: 10.)) is None


def test_repeated_controls_do_not_create_evidence_for_an_unidentified_objective_model():
    controller = ObjectiveResearch(_spec())
    rows = _rows(lambda point: point["x2"])
    for row in rows:
        row["observation"]["features"] = {"x1": .4, "x2": .6}
        row["context"]["objective"] = .6
    assert _request(controller, rows, _forecast(lambda point: point["x2"])) is None


def test_an_additive_model_without_predictive_support_does_not_manufacture_a_goal_opportunity():
    controller = ObjectiveResearch(_spec())
    points = [{"x1": x, "x2": y} for x in (.1, .3, .5, .7, .9)
              for y in (.1, .3, .5, .7, .9)]
    # A pure interaction has zero additive projection on the observed grid.
    rows = [{"observation": {"features": point, "target": -1.},
             "context": {"objective": 100 * (point["x1"] - .5) * (point["x2"] - .5),
                         "global_index": 32 + index, "stage": 33 + index}}
            for index, point in enumerate(points)]
    assert _request(controller, rows, _forecast(lambda point: -100.)) is None


@pytest.mark.parametrize("budget", [0, 20000])
def test_missing_value_or_missing_compute_does_not_spend_a_measurement(budget):
    controller = ObjectiveResearch(_spec(), compute_budget=budget)
    objective = (lambda point: point["x2"]) if budget == 0 else (lambda point: 1.)
    rows = _rows(objective)
    assert _request(controller, rows, _forecast(objective)) is None
    assert len(rows) == 16


def test_constraint_scope_cancels_the_request_but_goal_replacement_needs_its_own_future_evidence():
    controller = ObjectiveResearch(_spec())
    old_rows = _rows(lambda point: point["x2"])
    old = _request(controller, old_rows, _forecast(lambda point: point["x2"]))
    assert old["features"]["x2"] < .01
    old_model = deepcopy(controller.to_dict()["model"])
    rows = old_rows + _rows(lambda point: -point["x2"], offset=48, count=8)
    request = _request(controller, rows, _forecast(lambda point: point["x2"]), scope=48, remaining=8)
    state = controller.to_dict()
    assert state["canceled_requests"][-1]["request"] == old
    assert state["model"] == old_model
    assert state["scope_research"]["scope_start"] == 48
    assert state["scope_research"]["training_indices"] == list(range(48, 56))
    assert not state["scope_research"]["validation"] and controller.accepted_revision is None
    for count in range(4):
        assert request is not None and request["reason"] == "validate_objective_scope"
        point = request["features"]
        index = rows[-1]["context"]["global_index"] + 1
        context = {"global_index": index, "stage": index + 1,
                   "objective": -point["x2"], "purpose": request["purpose"]}
        rows.append({"observation": {"features": point, "target": -1.}, "context": context})
        controller.observe(point, [-point["x2"], -1.], context=context)
        if count < 3:
            assert controller.accepted_revision is None
            request = _request(controller, rows, _forecast(lambda value: value["x2"]),
                               scope=48, remaining=64 - context["stage"])
    revision = controller.accepted_revision
    assert revision is not None and revision["scope_start"] == 48
    assert revision["training_indices"] == list(range(48, 56))
    assert revision["validation_indices"] == list(range(56, 60))
    assert controller.to_dict()["model"] == old_model
    controller.acknowledge_revision()
    assert controller.to_dict()["evidence_start"] == 48
    assert controller.to_dict()["model"]["coefficients"][2] < 0.


def test_json_resume_preserves_an_unmeasured_request_without_refitting_or_inventing_feedback():
    controller = ObjectiveResearch(_spec())
    rows = _rows(lambda point: point["x2"])
    forecast = _forecast(lambda point: point["x2"])
    request = _request(controller, rows, forecast)
    assert request is not None
    pending = json.loads(json.dumps(controller.to_dict()))
    restored = ObjectiveResearch(_spec(), state=pending)
    assert _request(restored, rows, forecast) == request
    assert json.loads(json.dumps(restored.to_dict())) == pending


def test_a_future_failed_experiment_is_recorded_as_negative_value_and_refitting_waits_for_new_evidence():
    controller = ObjectiveResearch(_spec())
    rows = _rows(lambda point: point["x2"])
    forecast = _forecast(lambda point: point["x2"])
    request = _request(controller, rows, forecast)
    state = controller.to_dict()
    assert len(state["fits"]) == 1 and not state["outcomes"]
    assert request["prediction"]["predicted_gain"] > 0.
    # This later real outcome contradicts the previously favorable prediction.
    context = {"purpose": request["purpose"], "global_index": 48, "stage": 49}
    controller.observe(request["features"], [2., -1.], context=context)
    state = controller.to_dict()
    outcome, = state["outcomes"]
    assert outcome["request"] == request
    assert outcome["objective"] == 2. and outcome["measured_improvement"] < 0.
    assert state["pending"] is None and len(state["requests"]) == 1
    for index in range(4):
        point = request["features"] if index == 0 else {"x1": .1 * index, "x2": .1 * index}
        rows.append({"observation": {"features": point, "target": -1.},
                     "context": {"objective": 2. if index == 0 else point["x2"],
                                 "global_index": 48 + index, "stage": 49 + index}})
        _request(controller, rows, forecast)
        assert len(controller.to_dict()["fits"]) == (1 if index < 3 else 2)
    assert controller.to_dict()["outcomes"][0] == outcome


def _replicate_rows(first, second):
    rows = []
    for point, targets in (({"x1": .2, "x2": .2}, first), ({"x1": .8, "x2": .8}, second)):
        for target in targets:
            rows.append({"observation": {"features": deepcopy(point), "target": -1.},
                         "context": {"objective": target, "global_index": 32 + len(rows),
                                     "stage": 33 + len(rows)}})
    return rows


def _noisy_ranking_forecast(point):
    return {"features": deepcopy(point), "objective_mean": .2 if point["x1"] < .5 else .1,
            "objective_variance": .04, "objective_noise": .04,
            "constraint_mean": -1., "constraint_variance": 0., "feasible": True}


def test_terminal_research_uses_repeat_means_instead_of_chasing_the_lowest_noise_draw():
    controller = ObjectiveResearch(_spec())
    rows = _replicate_rows([-1., 1., 1.], [.1, .1, .1])
    assert min(row["context"]["objective"] for row in rows[:3]) == -1.
    # The average of the first control is worse and both empirical/posterior
    # rankings prefer the second. There is no ranking conflict to investigate.
    assert _request(controller, rows, _noisy_ranking_forecast, remaining=4) is None
    assert not controller.to_dict()["requests"]


def test_a_live_noisy_ranking_conflict_buys_one_pair_of_actual_repeats_within_the_same_budget():
    controller = ObjectiveResearch(_spec())
    rows = _replicate_rows([-.2, .2], [.1, .1])
    points = []
    for remaining, value in ((4, .2), (3, .1)):
        request = _request(controller, rows, _noisy_ranking_forecast, remaining=remaining)
        assert request["purpose"] == "objective_confirmation"
        points.append(request["features"])
        state = controller.to_dict()
        assert len(state["outcomes"]) == len(points) - 1
        context = {"purpose": request["purpose"], "stage": 33 + len(rows),
                   "global_index": 32 + len(rows)}
        controller.observe(request["features"], [value, -1.], context=context)
        rows.append({"observation": {"features": deepcopy(request["features"]), "target": -1.},
                     "context": {**context, "objective": value}})
    assert points == [{"x1": .2, "x2": .2}, {"x1": .8, "x2": .8}]
    assert len(rows) == 6 and len(controller.to_dict()["outcomes"]) == 2
    assert _request(controller, rows, _noisy_ranking_forecast, remaining=2) is None
    assert len(controller.to_dict()["requests"]) == 2


def test_a_ranking_conflict_with_only_one_call_left_cannot_start_a_two_control_confirmation():
    controller = ObjectiveResearch(_spec())
    rows = _replicate_rows([-.2, .2], [.1, .1])
    assert _request(controller, rows, _noisy_ranking_forecast, remaining=1) is None
    assert not controller.to_dict()["requests"]


def test_an_infeasible_posterior_favorite_does_not_displace_a_feasible_measured_incumbent():
    controller = ObjectiveResearch(_spec())
    rows = _replicate_rows([-.2, .2], [.1, .1])

    def forecast(point):
        prediction = _noisy_ranking_forecast(point)
        if point["x1"] > .5:
            prediction.update(feasible=False, constraint_mean=1.)
        return prediction

    assert _request(controller, rows, forecast, remaining=4) is None
    assert not controller.to_dict()["requests"]


def test_unchanged_or_incomplete_evidence_does_not_refit_or_rescan_the_gp_pool():
    controller = ObjectiveResearch(_spec())
    rows = _rows(lambda point: point["x2"])
    forecasts = []

    def forecast(point):
        forecasts.append(dict(point))
        return _forecast(lambda features: features["x2"])(point)

    request = _request(controller, rows, forecast)
    controller.observe(request["features"], [request["features"]["x2"], -1.],
        context={"purpose": request["purpose"], "global_index": 48, "stage": 49})
    count = len(forecasts)
    cost = controller.to_dict()["compute_cost"]
    assert _request(controller, rows, forecast) is None
    assert _request(controller, rows, forecast) is None
    assert len(forecasts) == count and controller.to_dict()["compute_cost"] == cost
    for index in range(3):
        point = {"x1": .1 * index, "x2": .2}
        rows.append({"observation": {"features": point, "target": -1.},
                     "context": {"objective": .2, "global_index": 48 + index, "stage": 49 + index}})
        assert _request(controller, rows, forecast) is None
    assert len(forecasts) == count
    assert controller.to_dict()["planning_passes"] == len(controller.to_dict()["fits"]) == 1


def test_lazy_gp_screen_does_not_discard_the_first_feasible_offer_beyond_twelve_unusable_points():
    controller = ObjectiveResearch(_spec())
    rows = _rows(lambda point: point["x2"])
    calls = []

    def forecast(point):
        calls.append(dict(point))
        return {**_forecast(lambda features: features["x2"])(point), "feasible": point["x2"] >= .4}

    request = _request(controller, rows, forecast)
    assert request is not None
    assert .4 <= request["features"]["x2"] < .45
    assert len(calls) > 12
    assert controller.to_dict()["screened_controls"] > controller.to_dict()["screened_gp_controls"]


def _changed_goal_trial(*, validation_supports=True):
    controller = ObjectiveResearch(_spec())
    old = lambda point: point["x2"]
    new = lambda point: -point["x2"]
    rows = _rows(old)
    forecast = _forecast(old)
    request = _request(controller, rows, forecast)
    controller.observe(request["features"], [old(request["features"]), -1.],
        context={"purpose": request["purpose"], "global_index": 48, "stage": 49})
    rows.extend(_rows(new, offset=48, count=4))
    for _ in range(12):
        request = _request(controller, rows, forecast, remaining=64 - rows[-1]["context"]["stage"])
        if request is None:
            break
        objective = (old if request["reason"] == "validate_objective_scope" and not validation_supports else new)
        index = rows[-1]["context"]["global_index"] + 1
        value = objective(request["features"])
        context = {"purpose": request["purpose"], "global_index": index, "stage": index + 1}
        controller.observe(request["features"], [value, -1.], context=context)
        rows.append({"observation": {"features": request["features"], "target": -1.},
                     "context": {**context, "objective": value}})
        if controller.accepted_revision is not None or controller.to_dict()["scope_attempts"]:
            break
    return controller, rows, forecast


def test_paid_objective_mismatch_earns_a_later_scope_only_after_future_predictions_improve():
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.models import Observation
    controller, rows, _ = _changed_goal_trial()
    revision = controller.accepted_revision
    assert revision is not None and revision["scope_start"] == 48
    assert len(revision["validation_indices"]) == 4
    assert set(revision["training_indices"]).isdisjoint(revision["validation_indices"])
    assert revision["validation_mse"] < revision["previous_mse"]
    program = Program.from_dict(revision["program"])
    assert program.predict(Observation({"x1": .7, "x2": .3}, 0.)) == pytest.approx(-.3, abs=1e-7)
    assert controller.to_dict()["compute_cost"] <= 20000
    controller.acknowledge_revision()
    assert controller.to_dict()["evidence_start"] == 48
    assert len(controller.to_dict()["scope_revisions"]) == 1


def test_refuted_objective_scope_is_archived_and_new_evidence_can_restart_planning():
    controller, rows, forecast = _changed_goal_trial(validation_supports=False)
    state = controller.to_dict()
    assert controller.accepted_revision is None
    assert state["scope_research"] is None and len(state["scope_attempts"]) == 1
    plans = state["planning_passes"]
    assert _request(controller, rows, forecast, remaining=16) is None
    for row in _rows(lambda point: point["x2"], offset=rows[-1]["context"]["global_index"] + 1, count=4):
        rows.append(row)
        _request(controller, rows, forecast, remaining=16)
    assert controller.to_dict()["planning_passes"] > plans
    assert len(controller.to_dict()["scope_attempts"]) == 1
