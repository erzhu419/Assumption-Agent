"""Actual later research returns must select executable proxy functions."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.research_evaluator import ResearchEvaluator, research_features


def offer(residual):
    return {"kind": "revision", "action": "representation", "observation_cost": 16, "compute_cost": 6000,
        "profile": {"observation_count": 32, "residual": {"normalized": residual, "state": "unresolved"},
            "objective": {"loss": "squared", "quantile": .5},
            "representation": {"history_depth": 0},
            "data": {"max_history": 3, "independent_trajectories": True}}}


def measure(evaluator, x, gain, *, fit_budget=1200):
    prediction_budget = SearchBudget(256)
    prediction = evaluator.predict(offer(x), budget=prediction_budget)
    assert prediction["compute_cost"] == prediction_budget.spent
    budget = SearchBudget(fit_budget)
    update = evaluator.observe(prediction["forecast"],
        {"execution_complete": True, "reference_rows": 16, "normalized_gain": gain}, budget=budget)
    assert update["compute_cost"] == budget.spent <= fit_budget
    return prediction, update


def learn_linear(evaluator):
    training = [0., .2, .4, .6, .8, 1.]
    for x in training:
        prediction, _ = measure(evaluator, x, .8 * x - .4)
        assert prediction["expected_gain"] is None
    assert evaluator.current is None and evaluator.candidates
    # These four outcomes occur after the candidate function was frozen.
    for x in (0., 1., 0., 1.):
        measure(evaluator, x, .8 * x - .4)
    assert evaluator.current is not None


def test_later_measured_returns_adopt_a_function_that_changes_predictions():
    evaluator = ResearchEvaluator()
    learn_linear(evaluator)
    low = evaluator.predict(offer(.25), budget=SearchBudget(10))
    high = evaluator.predict(offer(.75), budget=SearchBudget(10))
    assert low["expected_gain"] == pytest.approx(-.2, abs=1e-8)
    assert high["expected_gain"] == pytest.approx(.2, abs=1e-8)
    assert low["program"]["op"] != "const"
    adoption = next(event for event in evaluator.events if event["event"] == "adopt_research_evaluator")
    assert adoption["comparison"]["trials"] == 4
    assert adoption["comparison"]["gain"] > adoption["comparison"]["support_margin"]


def test_unpaid_or_unfinished_returns_do_not_change_models_or_training_data():
    evaluator = ResearchEvaluator()
    before = evaluator.to_dict()
    unavailable = evaluator.predict(offer(.5), budget=SearchBudget(0))
    assert unavailable["expected_gain"] is None and unavailable["forecast"] is None
    assert evaluator.to_dict() == before
    forecast = evaluator.predict(offer(.5), budget=SearchBudget(1))["forecast"]
    feedback = {"execution_complete": True, "reference_rows": 16, "normalized_gain": .7}
    assert evaluator.observe(forecast, feedback, budget=SearchBudget(0))["status"] == "insufficient_budget"
    assert evaluator.records == [] and evaluator.current is None
    assert len(evaluator.deferred) == 1
    after_delivery = evaluator.to_dict()
    assert evaluator.observe(forecast, {**feedback, "execution_complete": False},
        budget=SearchBudget(1200))["status"] == "unmeasured_return"
    assert evaluator.to_dict() == after_delivery
    evaluator = ResearchEvaluator(json.loads(json.dumps(after_delivery)))
    assert evaluator.flush(budget=SearchBudget(1))["compute_cost"] == 1
    assert len(evaluator.records) == 1 and evaluator.current is None
    assert evaluator.deferred == []


def test_pending_comparisons_and_selected_forecasts_restore_without_refitting():
    evaluator = ResearchEvaluator()
    for x in (0., .2, .4, .6, .8, 1.):
        measure(evaluator, x, .8 * x - .4)
    measure(evaluator, 0., -.4)
    prediction = evaluator.predict(offer(1.), budget=SearchBudget(10))
    saved = json.loads(json.dumps({"evaluator": evaluator.to_dict(), "selected_forecast": prediction["forecast"]}))
    restored = ResearchEvaluator(saved["evaluator"])
    feedback = {"execution_complete": True, "reference_rows": 16, "normalized_gain": .4}
    # Zero fitting allowance remains after paying the already frozen errors.
    price = 1 + len(prediction["forecast"]["predictions"])
    expected = evaluator.observe(prediction["forecast"], feedback, budget=SearchBudget(price))
    actual = restored.observe(saved["selected_forecast"], feedback, budget=SearchBudget(price))
    assert actual == expected and restored.to_dict() == evaluator.to_dict()
    for x in (0., 1.):
        measure(restored, x, .8 * x - .4)
    assert restored.current is not None


def test_counterexamples_withdraw_an_adopted_function_to_unknown():
    evaluator = ResearchEvaluator()
    learn_linear(evaluator)
    # A paid forecast of +.4 repeatedly precedes a measured return of -.4.
    # No fitting budget is supplied: only the previously frozen predictions
    # and their paid new errors can cause the withdrawal.
    for _ in range(4):
        prediction = evaluator.predict(offer(1.), budget=SearchBudget(10))
        price = 1 + len(prediction["forecast"]["predictions"])
        evaluator.observe(prediction["forecast"],
            {"execution_complete": True, "reference_rows": 16, "normalized_gain": -.4},
            budget=SearchBudget(price))
    assert evaluator.current is None
    assert evaluator.predict(offer(1.), budget=SearchBudget(10))["expected_gain"] is None
    assert evaluator.events[-1]["event"] == "withdraw_research_evaluator"


def test_loss_of_fit_can_propose_a_different_conditional_representation():
    evaluator = ResearchEvaluator()
    for x in (0., .2, .4, .6, .8, 1.):
        measure(evaluator, x, -.6 if x < .5 else .6)
    assert any(model["program"]["op"] == "if" for model in evaluator.candidates)
    for x in (0., 1., .2, .8):
        measure(evaluator, x, -.6 if x < .5 else .6)
    assert evaluator.current["program"]["op"] == "if"
    assert evaluator.predict(offer(.4), budget=SearchBudget(10))["expected_gain"] == pytest.approx(-.6)
    assert evaluator.predict(offer(.6), budget=SearchBudget(10))["expected_gain"] == pytest.approx(.6)


def test_prediction_errors_rewrite_an_already_supported_linear_function():
    evaluator = ResearchEvaluator()
    learn_linear(evaluator)
    previous = evaluator.current["model_id"]
    before = evaluator.predict(offer(.4), budget=SearchBudget(10))["expected_gain"]
    for x in (.4, .6, 0., 1.) * 3:
        measure(evaluator, x, -.6 if x < .5 else .6)
    assert evaluator.current["model_id"] != previous
    assert evaluator.current["program"]["op"] == "if"
    after = evaluator.predict(offer(.4), budget=SearchBudget(10))["expected_gain"]
    assert (after + .6) ** 2 < (before + .6) ** 2
    replacements = [event for event in evaluator.events if event["event"] == "adopt_research_evaluator"
                    and event["previous_model_id"] is not None]
    assert replacements and all(event["comparison"]["supported"] for event in replacements)


def test_identity_and_final_evaluation_are_not_features_and_prediction_does_not_accumulate_state():
    evaluator = ResearchEvaluator()
    first = offer(.3)
    changed = deepcopy(first)
    changed.update(task_id="different", seed=7919, evaluation_loss=0., expected_gain=.999)
    changed["profile"].update(task_id="also_ignored", evaluation={"loss": 0.})
    assert research_features(first) == research_features(changed)
    before = evaluator.to_dict()
    for _ in range(20):
        evaluator.predict(changed, budget=SearchBudget(10))
    assert evaluator.to_dict() == before


def test_the_old_nonzero_estimate_must_actually_lose_before_a_new_function_replaces_it():
    evaluator = ResearchEvaluator()

    def observed(x, target, old):
        query = {**offer(x), "expected_gain": old}
        prediction = evaluator.predict(query, budget=SearchBudget(256))
        assert prediction["forecast"]["baseline_expected_gain"] == old
        evaluator.observe(prediction["forecast"], {"execution_complete": True,
            "reference_rows": 16, "normalized_gain": target}, budget=SearchBudget(1200))

    for x in (0., .2, .4, .6, .8, 1., 0., 1., 0., 1.):
        target = .8 * x - .4
        observed(x, target, target)  # the previous estimate is already correct
    assert evaluator.current is None
    assert not any(event["event"] == "adopt_research_evaluator" for event in evaluator.events)
    # The old condition mean now overestimates low-residual opportunities.
    # The executable input-dependent model must win on later actual returns.
    for x in (0., .2, 0., .2):
        observed(x, .8 * x - .4, .3)
    assert evaluator.current is not None
    assert evaluator.predict({**offer(.1), "expected_gain": .3},
        budget=SearchBudget(10))["expected_gain"] == pytest.approx(-.32, abs=1e-8)
