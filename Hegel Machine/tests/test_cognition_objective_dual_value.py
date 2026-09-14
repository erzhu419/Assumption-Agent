"""Price current-mechanism repeats separately from independent model tests.

The saved shared-tail failure supplies paid prefixes. Forecast constants below
come from replaying that prefix with its actual GP before stages 60 and 62;
the decision tests acquire no labels and never pass future observations in.
"""

from copy import deepcopy
import json
from math import pi, sqrt
from pathlib import Path

import pytest

from hegel_machine.cognition.models import TaskSpec
from hegel_machine.cognition.decision_experiments import _predict
import hegel_machine.cognition.objective_research as research_module
from hegel_machine.cognition.objective_research import ObjectiveResearch, repeat_decision_value
from hegel_machine.cognition.validation_continuation import evidence_summary
from test_cognition_objective_completion import observations, spec as completion_spec


@pytest.mark.parametrize("repeat_incumbent", [True, False])
def test_repeat_value_uses_gaussian_update_and_true_value_label_covariance(repeat_incumbent):
    if repeat_incumbent:
        incumbent = dict(objective_mean=0., current_mean=4.)
        alternative = dict(objective_mean=1., current_mean=1.)
    else:
        incumbent = dict(objective_mean=1., current_mean=1.)
        alternative = dict(objective_mean=2., current_mean=-2.)
    for point in (incumbent, alternative):
        point.update(objective_variance=1., objective_noise=3., current_variance=1.)
    before = deepcopy((incumbent, alternative))
    value = repeat_decision_value(incumbent, alternative, repeat_incumbent=repeat_incumbent)
    assert value["gp_update_weight"] == pytest.approx(.25)
    assert value["expected_updated_mean"] == pytest.approx(1.)
    assert value["predictive_variance"] == pytest.approx(4.)
    assert value["switch_probability"] == pytest.approx(.5)
    # The threshold is the predictive mean. Half of the 3-unit mean gap is
    # recovered, plus covariance / predictive_sd times the standard density.
    assert value["gain"] == pytest.approx(1.5 + 1 / (2 * sqrt(2 * pi)))
    assert (incumbent, alternative) == before


def test_repeat_with_no_gp_update_cannot_change_the_recommendation():
    incumbent = dict(objective_mean=0., current_mean=4., objective_variance=0.,
                     objective_noise=3., current_variance=1.)
    alternative = {**incumbent, "objective_mean": 1., "current_mean": 1.}
    value = repeat_decision_value(incumbent, alternative, repeat_incumbent=True)
    assert value["gp_update_weight"] == value["gain"] == value["switch_probability"] == 0.
    assert value["expected_updated_mean"] == 0.


@pytest.fixture(scope="module")
def failed_shared_tail():
    source = Path(__file__).resolve().parents[1] / (
        "artifacts/sc_kg_shared_tail_20260912/runs/"
        "unknown_revision_1_center_interaction_change_hegel_shared_tail.json")
    return json.loads(source.read_text())


def _prefix(record, stage):
    names = ("x1", "x2", "x3")
    spec = TaskSpec("paid-tail-failure", "Resolve current goal value", names, (), (),
        feature_bounds={name: (0., 1.) for name in names},
        context={"objective_direction": "minimize", "constraint_upper_bound": "0"})
    paid = [(i, point, output) for i, (point, output) in enumerate(
        zip(record["initial_points"], record["initial_outputs"]))]
    paid.extend((31 + item["new_call"], item["point"], item["output"])
                for item in record["simulations"] if 31 + item["new_call"] < stage)
    rows = [{"observation": {"features": dict(zip(names, (value / 100 for value in point))),
                              "target": output[1]},
             "context": {"global_index": index, "stage": index + 1, "objective": output[0]}}
            for index, point, output in paid]
    research = deepcopy(record["objective_research"]["scope_attempts"][-1])
    research["validation"] = [row for row in research["validation"] if row["global_index"] < stage]
    research["validation_rounds"] = [row for row in research["validation_rounds"]
                                     if row["stage"] <= stage]
    research.update(phase="validating", validation_target=len(research["validation"]) + 1,
                    status="inconclusive_future_predictions")
    assert research["candidate"]["training_end"] == 53 < stage
    goal = ObjectiveResearch(spec)
    goal.state.update(scope_start=0, evidence_start=0, scope_research=research)
    return goal, rows


def _prefix_forecasts(stage):
    # Native GP/HVD replay forecasts, before receiving the next saved vector.
    if stage == 60:
        means = (.37115095096333106, .3978310389569931)
        variances = (.00018463828622582645, .0003330222922621569)
        noise = .0007621845786777539
    else:
        assert stage == 62
        means = (.38755192347654965, .39668910442944305)
        variances = (.00014859482511664102, .00024066708385231086)
        noise = .0008675329002502875
    return {point: {"features": dict(zip(("x1", "x2", "x3"), point)),
                    "objective_mean": mean, "objective_variance": variance,
                    "objective_noise": noise, "feasible": True}
            for point, mean, variance in zip(((.25, .78, .69), (.06, .69, .73)), means, variances)}


def _forecast(predictions):
    def forecast(point):
        key = tuple(point[name] for name in ("x1", "x2", "x3"))
        return deepcopy(predictions.get(key, {"features": point, "feasible": False}))
    return forecast


def test_actual_compared_point_still_has_value_when_a_validation_slot_is_reserved(failed_shared_tail):
    goal, rows = _prefix(failed_shared_tail, 62)
    validation = deepcopy(goal.state["scope_research"]["validation"])
    predictions = _prefix_forecasts(62)
    assert len(rows) == 62 and len(validation) == 8
    assert all(any(row["features"] == item["features"] for row in validation)
               for item in predictions.values())
    request = goal._scope_decision_confirmation(rows, _forecast(predictions), 62, 2)
    assert request["features"] == {"x1": .25, "x2": .78, "x3": .69}
    assert request["reason"] == "resolve_current_goal_value"
    assert request["expected_decision_gain"] > 0
    assert request["validation_calls_needed"] == 1
    assert all(not offer["advances_validation"] for offer in request["value_offers"])
    assert all(offer["preserves_validation_budget"] for offer in request["value_offers"])
    assert request["current_incumbent"]["current_mean"] == pytest.approx(.45544960452430194)
    other = request["current_alternative"]
    assert other["current_count"] == 2
    assert other["current_mean"] == pytest.approx(.3806298588610115)
    assert other["current_variance"] == pytest.approx(other["objective_noise"] / 2)
    assert goal.state["scope_research"]["validation"] == validation
    assert evidence_summary(validation)["mean_gain"] < 0
    assert request["validation_plan"] is None
    assert not goal.state["outcomes"] and not goal.state["shared_predictions"]


@pytest.mark.parametrize("needed", [1, 2])
def test_repeat_cannot_spend_the_last_slots_needed_for_distinct_validation(failed_shared_tail, needed):
    goal, rows = _prefix(failed_shared_tail, 62)
    research = goal.state["scope_research"]
    research["validation_target"] = len(research["validation"]) + needed
    request = goal._scope_decision_confirmation(rows, _forecast(_prefix_forecasts(62)), 62, needed)
    assert request is None and goal.state["pending"] is None
    assert not goal.state["requests"]


def test_unmeasured_current_point_is_calibrated_before_repeat_value_is_estimated(
        failed_shared_tail, monkeypatch):
    goal, rows = _prefix(failed_shared_tail, 60)
    predictions = _prefix_forecasts(60)
    monkeypatch.setattr(research_module, "repeat_decision_value", lambda *args, **kwargs:
        pytest.fail("the questioned GP cannot supply an unmeasured current value for valuation"))
    request = goal._scope_decision_confirmation(rows, _forecast(predictions), 60, 4)
    assert request["reason"] == "calibrate_current_goal_value"
    assert request["advances_validation"]
    assert "expected_decision_gain" not in request and "value_offers" not in request
    incumbent = request["current_incumbent"]
    assert incumbent["current_count"] == 0
    assert incumbent["current_mean"] == incumbent["objective_mean"]
    assert incumbent["current_variance"] == incumbent["objective_variance"]
    assert incumbent["current_mean"] != pytest.approx(.3375316373085575)
    other_goal, changed = _prefix(failed_shared_tail, 60)
    for row in changed:
        if row["context"]["global_index"] < 41:
            row["context"]["objective"] = -1e6
    other = other_goal._scope_decision_confirmation(changed, _forecast(predictions), 60, 4)
    assert other == request


def test_positive_actual_candidate_can_buy_a_more_valuable_validation_pair():
    source = Path(__file__).resolve().parents[1] / (
        "artifacts/sc_kg_shared_tail_20260912/runs/"
        "goal_completion_2_center_change_hegel_shared_tail.json")
    record = json.loads(source.read_text())
    goal = ObjectiveResearch(completion_spec())
    saved = record["objective_research"]
    revision = saved["scope_revisions"][0]
    records = sorted((row for row in [*saved["outcomes"], *saved["shared_outcomes"]]
                      if row["request"]["reason"] == "validate_objective_scope"
                      and row["context"]["global_index"] < 61),
                     key=lambda row: row["context"]["global_index"])
    validation = [{"features": deepcopy(row["request"]["features"]),
                   "global_index": row["context"]["global_index"],
                   "candidate_error": (row["objective"] - row["request"]["candidate_mean"]) ** 2,
                   "previous_error": (row["objective"] - row["request"]["previous_mean"]) ** 2}
                  for row in records]
    evidence = evidence_summary(validation)
    assert len(validation) == 12 and evidence["decision"] == "inconclusive"
    assert evidence["mean_gain"] == pytest.approx(.05161923182979782)
    research = {"scope_start": 40, "phase": "validating", "validation": validation,
                "validation_target": 14, "candidate": deepcopy(revision["model"]),
                "training_indices": revision["training_indices"]}
    goal.state.update(scope_start=0, evidence_start=0, scope_research=research)
    paid = list(enumerate(zip(record["initial_points"], record["initial_outputs"])))
    paid.extend((31 + row["new_call"], (row["point"], row["output"]))
                for row in record["simulations"] if 31 + row["new_call"] < 61)
    rows = [{"observation": {"features": dict(zip(goal.spec.feature_names,
                              (value / 100 for value in point))), "target": output[1]},
             "context": {"global_index": index, "objective": output[0]}}
            for index, (point, output) in paid]
    noise = .0014951377601833858
    # These two incumbent forecasts are from the real stage-61 GP. Two other
    # already paid controls receive controlled quarter-unit disagreements to
    # exercise the prospective planner without manufacturing validation labels.
    forecasts = {
        (.28, .58, .69): (.3899129550630398, .0001644120413240702),
        (.12, .67, .78): (.4257368307419247, .0003373505618788662),
    }
    probes = {tuple(value / 100 for value in point) for point in record["initial_points"][:2]}

    def forecast(point):
        key = tuple(point[name] for name in goal.spec.feature_names)
        if key in forecasts:
            mean, variance = forecasts[key]
        elif key in probes:
            mean, variance = _predict(goal.spec, research["candidate"], point) + .25, .0005
        else:
            return {"features": deepcopy(point), "feasible": False}
        return {"features": deepcopy(point), "objective_mean": mean,
                "objective_variance": variance, "objective_noise": noise, "feasible": True}

    before = deepcopy(research)
    request = goal._scope_decision_confirmation(rows, forecast, 61, 3)
    assert request["reason"] == "validate_objective_scope"
    plan = request["validation_plan"]
    assert plan["new_calls"] == 2 and plan["expected_gain"] > 0
    assert plan["expected_gain_per_call"] > max(row["gain"] for row in request["value_offers"])
    assert request["features"] not in [row["features"] for row in validation]
    assert research == before and not goal.state["outcomes"]


@pytest.mark.parametrize("entry", ["active", "shared"])
@pytest.mark.parametrize("favors_candidate", [True, False])
def test_last_real_label_resolves_the_available_round_without_relaxing_evidence(entry, favors_candidate):
    source = Path(__file__).resolve().parents[1] / (
        "artifacts/sc_kg_dual_value_20260912/revision2/runs/"
        "goal_completion_2_center_change_hegel_dual_value.json")
    saved = json.loads(source.read_text())
    goal = ObjectiveResearch(completion_spec())
    research = deepcopy(saved["objective_research"]["scope_research"])
    goal.state.update(scope_start=0, evidence_start=0, scope_research=research)
    before = deepcopy(research["validation"])
    assert len(before) == 14 and research["validation_target"] == 16
    assert evidence_summary(before)["decision"] == "inconclusive"
    compared = [row["features"] for row in before]
    point = next(point for point in research["validation_pool"] if point not in compared)
    candidate = _predict(goal.spec, research["candidate"], point)
    previous = candidate + sqrt(.1 if favors_candidate else 1.)
    paid = list(enumerate(zip(saved["initial_points"], saved["initial_outputs"])))
    paid.extend((31 + row["new_call"], (row["point"], row["output"]))
                for row in saved["simulations"] if 31 + row["new_call"] < 63)
    rows = [{"observation": {"features": dict(zip(goal.spec.feature_names,
                              (value / 100 for value in control))), "target": output[1]},
             "context": {"global_index": index, "objective": output[0]}}
            for index, (control, output) in paid]
    assert len(rows) == 63

    def forecast(features):
        return {"features": deepcopy(features), "objective_mean": previous,
                "objective_variance": .001, "objective_noise": .001,
                "feasible": features == point}

    if entry == "active":
        request = goal.next_control(rows, forecast, scope_start=0, stage=63, remaining_calls=1)
    else:
        goal._record_request(point, "objective_confirmation", "resolve_current_goal_value", 63)
        request = goal.prepare_shared_control(rows, forecast, point,
            scope_start=0, stage=63, remaining_calls=1, purpose="objective_confirmation")
    assert research["validation_target"] == 15
    assert research["validation"] == before and goal.accepted_revision is None
    assert request["reason"] == "validate_objective_scope" and request["features"] not in compared
    assert not goal.state["outcomes"] and not goal.state["shared_outcomes"]
    target = candidate if favors_candidate else previous
    goal.observe(point, [target, -1.], context={"global_index": 63, "stage": 64,
                                               "purpose": "objective_confirmation"})
    assert len(research["validation"]) == 15
    assert research["validation"][:-1] == before
    observed = research["validation"][-1]
    assert observed["global_index"] == 63 and observed["features"] == point
    assert observed["candidate_error"] == pytest.approx((target - candidate) ** 2)
    assert observed["previous_error"] == pytest.approx((target - previous) ** 2)
    evidence = evidence_summary(research["validation"])
    assert research["validation_rounds"][-1] == {**evidence, "stage": 64}
    assert evidence["decision"] == ("supported" if favors_candidate else "rejected")
    assert (goal.accepted_revision is not None) == favors_candidate
    assert len(goal.state["outcomes"]) == 1
    assert len(goal.state["shared_outcomes"]) == int(entry == "shared")
    assert goal.state["pending"] is None and goal.state["shared_pending"] is None


def test_last_complete_validation_window_actively_nominates_an_uncompared_point(failed_shared_tail):
    goal, rows = _prefix(failed_shared_tail, 62)
    research = goal.state["scope_research"]
    research.pop("reentry", None)  # The ordinary candidate path also needs an active last slot.
    compared = [row["features"] for row in research["validation"]]
    before = deepcopy(research["candidate"])

    def forecast(point):
        return {"features": deepcopy(point), "objective_mean": .5,
                "objective_variance": .001, "objective_noise": .001, "feasible": True}

    assert goal._research_scope(rows, forecast, 62, 2) is None
    request = goal._research_scope(rows, forecast, 63, 1)
    assert request is not None and request["reason"] == "validate_objective_scope"
    assert request["features"] not in compared
    assert research["candidate"] == before and len(research["validation"]) == 8
    assert not goal.state["outcomes"]


def test_collecting_with_four_calls_fits_and_starts_validation_before_repeating(monkeypatch):
    goal = ObjectiveResearch(completion_spec())
    rows = observations(20, affine=True)
    for row in rows:
        row["context"]["global_index"] += 40
    goal.state.update(scope_start=0, evidence_start=0, scope_research={
        "scope_start": 40, "phase": "collecting", "candidate": None, "validation": [],
        "coverage_calls": 0})
    monkeypatch.setattr(goal, "_scope_decision_confirmation", lambda *args:
        pytest.fail("a terminal repeat would leave fewer than four validation slots"))
    request = goal.next_control(rows, lambda point: {
        "features": deepcopy(point), "objective_mean": 0., "objective_variance": .1,
        "objective_noise": .1, "feasible": True},
        scope_start=0, stage=60, remaining_calls=4)
    assert request["reason"] == "validate_objective_scope"
    assert goal.state["scope_research"]["phase"] == "validating"
    assert goal.state["scope_research"]["candidate"]["cv_loss"] < 1e-8
    assert not goal.state["outcomes"]
