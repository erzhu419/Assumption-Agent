"""A validated goal head nominates purchases without pre-buying repeats."""

from copy import deepcopy
import json
from random import Random

import pytest

from hegel_machine.cognition.models import TaskSpec
from hegel_machine.cognition.objective_research import ObjectiveResearch


def _spec():
    return TaskSpec("terminal_goal", "Minimize the observed goal", ("x1", "x2"), (), (),
        feature_bounds={"x1": (0., 1.), "x2": (0., 1.)},
        context={"objective_direction": "minimize", "constraint_upper_bound": "0.0"})


def _forecast(point):
    return {"features": deepcopy(point), "objective_mean": -point["x2"],
        "objective_variance": .0001, "objective_noise": .035 ** 2,
        "constraint_mean": -1., "constraint_variance": 0., "feasible": True}


def _paid_row(point, target, index, purpose=None):
    return {"observation": {"features": deepcopy(point), "target": -1.},
        "context": {"objective": target, "global_index": index, "stage": index + 1,
                    "purpose": purpose}}


def _installed_head(*, acknowledge=True, start=32):
    controller = ObjectiveResearch(_spec())
    rng = Random(615)
    points = [{"x1": rng.uniform(.05, .95), "x2": rng.uniform(.05, .95)} for _ in range(16)]
    rows = [_paid_row(point, point["x2"], index + start) for index, point in enumerate(points)]

    def old_forecast(point):
        return {**_forecast(point), "objective_mean": point["x2"], "feasible": point["x2"] <= .95}

    controller.next_control(rows, old_forecast, scope_start=start, stage=start + 16, remaining_calls=16)
    rows += [_paid_row(point, -point["x2"], index + start + 16) for index, point in enumerate(points[:8])]
    for count in range(4):
        request = controller.next_control(rows, old_forecast, scope_start=start + 16,
            stage=len(rows) + start, remaining_calls=8 - count)
        assert request["reason"] == "validate_objective_scope"
        point = request["features"]
        row = _paid_row(point, -point["x2"], len(rows) + start, request["purpose"])
        rows.append(row)
        controller.observe(point, [-point["x2"], -1.], context=row["context"])
        if count < 3:
            assert controller.accepted_revision is None
    assert controller.accepted_revision is not None
    if acknowledge:
        controller.acknowledge_revision()
    assert len(rows) == 28
    return controller, rows


def _offers(controller, rows, forecast=_forecast, remaining=4):
    return controller.terminal_offers(rows, forecast, scope_start=controller.state["scope_start"],
        stage=rows[-1]["context"]["stage"], remaining_calls=remaining)


def _acquire(controller, rows, offer, target=None):
    request = controller.select_terminal_offer(offer, stage=rows[-1]["context"]["stage"])
    point = request["features"]
    target = -point["x2"] if target is None else target
    row = _paid_row(point, target, rows[-1]["context"]["global_index"] + 1, request["purpose"])
    controller.observe(point, [target, -1.], context=row["context"])
    rows.append(row)
    return request


def test_accepted_but_not_installed_head_cannot_nominate_terminal_actions():
    controller, rows = _installed_head(acknowledge=False)
    before = controller.to_dict()
    assert not _offers(controller, rows)
    assert controller.to_dict() == before
    controller.acknowledge_revision()
    assert any(offer["purpose"] == "objective_exploration" for offer in _offers(controller, rows))


def test_small_noisy_ranking_conflict_keeps_a_new_control_in_the_same_candidate_set():
    controller, rows = _installed_head()
    distinct = {tuple(row["observation"]["features"].values()) for row in rows[16:]}
    first, second = sorted(distinct, key=lambda point: -point[1])[:2]

    def forecast(point):
        result = _forecast(point)
        if tuple(point.values()) == second:
            result["objective_mean"] = -first[1] - .00328
        return result

    before = controller.to_dict()
    offers = _offers(controller, rows, forecast)
    assert len(offers) == 3
    assert {offer["purpose"] for offer in offers} == {"objective_confirmation", "objective_exploration"}
    assert controller.to_dict()["pending"] is None
    assert controller.to_dict()["requests"] == before["requests"]
    assert controller.to_dict()["outcomes"] == before["outcomes"]
    assert not controller.to_dict()["confirmation_queue"]
    new = next(offer for offer in offers if offer["purpose"] == "objective_exploration")
    request = _acquire(controller, rows, new)
    assert len(controller.to_dict()["requests"]) == len(before["requests"]) + 1
    assert controller.to_dict()["outcomes"][-1]["request"] == request
    assert controller.to_dict()["pending"] is None
    # One paid outcome, rather than another four-row batch, can change the
    # nominations. No previously planned second repeat survives the purchase.
    later = _offers(controller, rows, forecast, remaining=3)
    assert all(offer["purpose"] == "objective_confirmation" for offer in later)
    assert any(offer["features"] == new["features"] for offer in later)
    assert not controller.to_dict()["confirmation_queue"]


def test_last_single_call_can_still_nominate_new_evidence_and_failed_feedback_stays_paid():
    controller, rows = _installed_head()
    for index in range(3):
        point = {"x1": .3, "x2": .4 + .01 * index}
        rows.append(_paid_row(point, -point["x2"], len(rows) + 32))
    before = controller.to_dict()
    offers = _offers(controller, rows, remaining=1)
    offer = next(item for item in offers if item["purpose"] == "objective_exploration")
    _acquire(controller, rows, offer, target=2.)
    after = controller.to_dict()
    assert rows[-1]["context"]["stage"] == 64
    assert after["outcomes"][-1]["measured_improvement"] < 0.
    assert len(after["requests"]) == len(before["requests"]) + 1
    assert after["outcomes"][-1]["objective"] == 2.
    assert after["compute_cost"] <= after["compute_budget"]


def test_quantization_reuses_actual_controls_and_does_not_invent_an_unpaid_new_location():
    controller, rows = _installed_head()
    controls = [row["observation"]["features"] for row in rows[16:]]

    def forecast(point):
        # The apparatus permits only this measured discrete control catalog.
        actual = min(controls, key=lambda item: sum((item[name] - point[name]) ** 2 for name in point))
        return _forecast(actual)

    offers = _offers(controller, rows, forecast)
    assert offers and all(offer["purpose"] == "objective_confirmation" for offer in offers)
    keys = [tuple(offer["features"].values()) for offer in offers]
    assert len(keys) == len(set(keys))
    assert all(offer["features"] in controls for offer in offers)


def test_new_offer_predictions_describe_the_actual_quantized_control():
    controller, rows = _installed_head()
    controls = [row["observation"]["features"] for row in rows[16:]]

    def forecast(point):
        actual = point if point in controls else {**point, "x2": round(point["x2"] * .96, 2)}
        return _forecast(actual)

    offer = next(item for item in _offers(controller, rows, forecast)
                 if item["purpose"] == "objective_exploration")
    assert offer["features"]["x2"] < 1.
    assert offer["prediction"]["model_mean"] == pytest.approx(-offer["features"]["x2"], abs=1e-6)
    assert offer["prediction"]["objective_mean"] == -offer["features"]["x2"]


@pytest.mark.parametrize("remaining_compute", [0, 160])
def test_terminal_search_shares_remaining_compute_and_charges_old_point_forecasts(remaining_compute):
    controller, rows = _installed_head()
    state = controller.to_dict()
    state["compute_budget"] = state["compute_cost"] + remaining_compute
    controller = ObjectiveResearch(_spec(), state=state)
    calls = []

    def forecast(point):
        calls.append(deepcopy(point))
        return _forecast(point)

    before = controller.to_dict()["compute_cost"]
    offers = _offers(controller, rows, forecast)
    spent = controller.to_dict()["compute_cost"] - before
    assert len(calls) <= spent <= remaining_compute // 4
    assert controller.to_dict()["retained_compute"] == remaining_compute - remaining_compute // 4
    assert not controller.to_dict()["pending"]
    if remaining_compute:
        assert offers and calls
    else:
        assert not offers and not calls


def test_selected_pending_offer_survives_json_without_second_forecast_or_request():
    controller, rows = _installed_head()
    offer = _offers(controller, rows)[0]
    request = controller.select_terminal_offer(offer, stage=60)
    controller = ObjectiveResearch(_spec(), state=json.loads(json.dumps(controller.to_dict())))
    before = controller.to_dict()

    def unused_forecast(point):
        raise AssertionError("an already selected paid query needs no new nomination")

    assert _offers(controller, rows, unused_forecast) == [request]
    assert controller.select_terminal_offer(request, stage=60) == request
    assert controller.to_dict() == before


def test_new_scope_failure_keeps_four_future_validation_queries_ahead_of_terminal_offers():
    controller, rows = _installed_head(start=24)
    # A second measured shift arrives while another four-call validation is
    # still affordable; only actual observations trigger the new research.
    for index in range(8):
        point = {"x1": .1 + .1 * (index % 4), "x2": .2 + .13 * (index // 2)}
        rows.append(_paid_row(point, 2. + point["x2"], rows[-1]["context"]["global_index"] + 1))
    assert not _offers(controller, rows)
    monitoring = deepcopy(controller.to_dict()["monitoring"])
    request = controller.next_control(rows, _forecast, scope_start=controller.state["scope_start"],
        stage=rows[-1]["context"]["stage"], remaining_calls=4)
    assert controller.to_dict()["monitoring"] == monitoring
    assert request["reason"] == "validate_objective_scope"
    assert not controller.to_dict()["scope_research"]["validation"]
    assert controller.accepted_revision is None


def test_better_measured_point_can_research_its_lost_certificate_and_adverse_repeat_withdraws_it():
    controller, rows = _installed_head()
    promising = {"x1": .24, "x2": .99}
    row = _paid_row(promising, -.99, 60)
    row["observation"]["target"] = -.02
    rows.append(row)

    def forecast(point):
        result = _forecast(point)
        if point == promising:
            result.update(feasible=False, constraint_mean=-.02,
                          constraint_variance=.01, constraint_margin=.04)
        return result

    previous_requests = deepcopy(controller.to_dict()["requests"])
    offers = _offers(controller, rows, forecast, remaining=3)
    research, = [offer for offer in offers if offer["reason"] == "research_promising_constraint_certificate"]
    assert research["features"] == promising and research["feasible"] is False
    assert research["constraint_measured_mean"] == -.02 and research["constraint_replicates"] == 1
    assert research["prediction"]["constraint_margin"] == .04
    assert research["prediction"]["supported_gain"] > 0
    assert research["certified_incumbent"] != promising
    assert len(offers) <= 4 and controller.to_dict()["requests"] == previous_requests
    request = controller.select_terminal_offer(research, stage=61)
    denied = _paid_row(promising, -.99, 61, request["purpose"])
    denied["observation"]["target"] = .1
    rows.append(denied)
    controller.observe(promising, [-.99, .1], context=denied["context"])
    later = _offers(controller, rows, forecast, remaining=2)
    assert not any(offer["features"] == promising for offer in later)
    assert not controller.to_dict()["confirmation_queue"]
    assert controller.to_dict()["outcomes"][-1]["constraint"] == .1
    assert [row["observation"]["target"] for row in rows[-2:]] == [-.02, .1]


def test_only_the_strongest_supported_certificate_question_is_nominated():
    controller, rows = _installed_head()
    questions = [{"x1": .1, "x2": .96}, {"x1": .2, "x2": .99}]
    for point in questions:
        row = _paid_row(point, -point["x2"], rows[-1]["context"]["global_index"] + 1)
        row["observation"]["target"] = -.01
        rows.append(row)

    def forecast(point):
        return {**_forecast(point), "feasible": point not in questions}

    offers = _offers(controller, rows, forecast, remaining=2)
    question, = [item for item in offers if item["reason"] == "research_promising_constraint_certificate"]
    assert question["features"] == questions[1]
    assert len(offers) <= 4


def test_lost_certificate_alone_is_not_a_reason_to_repeat_a_worse_goal_point():
    controller, rows = _installed_head()
    poor = {"x1": .24, "x2": .1}
    rows.append(_paid_row(poor, -.1, 60))

    def forecast(point):
        return {**_forecast(point), "feasible": point != poor}

    offers = _offers(controller, rows, forecast, remaining=3)
    assert not any(offer["reason"] == "research_promising_constraint_certificate" for offer in offers)


def test_no_certified_point_can_still_offer_one_measured_certificate_question_without_declaring_feasibility():
    controller, rows = _installed_head()

    def forecast(point):
        return {**_forecast(point), "feasible": False, "constraint_margin": .1}

    offers = _offers(controller, rows, forecast)
    question, = offers
    best_paid = min(rows[16:], key=lambda row: row["context"]["objective"])
    assert question["features"] == best_paid["observation"]["features"]
    assert question["reason"] == "research_promising_constraint_certificate"
    assert question["feasible"] is False and question["certified_incumbent"] is None
    assert question["prediction"]["constraint_margin"] == .1
    assert "predicted_gain" not in question["prediction"]
    assert controller.to_dict()["pending"] is None


def test_untrusted_constraint_rows_cannot_supply_nominal_support_for_certificate_research():
    controller, rows = _installed_head()
    unsupported = {"x1": .24, "x2": .99}
    row = _paid_row(unsupported, -.99, 60)
    row["observation_status"] = "disputed"
    rows.append(row)

    def forecast(point):
        return {**_forecast(point), "feasible": point != unsupported}

    assert not any(offer["features"] == unsupported
                   for offer in _offers(controller, rows, forecast, remaining=3))


def test_a_measured_goal_rebuttal_withdraws_the_certificate_question_even_if_program_still_prefers_it():
    controller, rows = _installed_head()
    promising = {"x1": .24, "x2": .99}
    rows.append(_paid_row(promising, -.99, 60))

    def forecast(point):
        return {**_forecast(point), "feasible": point != promising}

    question, = [offer for offer in _offers(controller, rows, forecast, remaining=3)
                 if offer["reason"] == "research_promising_constraint_certificate"]
    saved_model = deepcopy(controller.to_dict()["model"])
    _acquire(controller, rows, question, target=-.1)
    later = _offers(controller, rows, forecast, remaining=2)
    assert controller.to_dict()["model"] == saved_model
    assert controller.to_dict()["scope_research"] is None
    assert not any(offer["features"] == promising for offer in later)
    assert controller.to_dict()["outcomes"][-1]["objective"] == -.1
    assert rows[-1]["observation"]["target"] == -1.
