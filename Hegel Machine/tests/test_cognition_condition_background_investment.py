"""A sampling failure applies to its measured background, not every experiment."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.condition_marginal_value import (
    CONDITION_FOLLOWUP_ACTION, condition_followup_policy,
    record_condition_followup, record_condition_information,
    sampling_applicability,
)
from hegel_machine.cognition.condition_response_actions import research_observed_condition_competition
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.observed_condition_experiments import (
    _interval_exposure, _local_residuals, refresh_condition_sampling,
)
from hegel_machine.cognition.object_continuation import observations
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_cognition_condition_research_returns import decide, paid, plan, state
from test_cognition_condition_response_value import root04


def scoped_plan(background=0., radius=.1):
    return {**plan(.1), "sampling_context": {
        "basis": "paid_local_control_residuals",
        "normalized_background": [background], "neighbourhood_radius": radius}}


def test_fitted_paid_response_funds_measurement_without_rebuilding_known_competitors(root04):
    rows = observations(root04["observations"])
    training, validation = rows[:-24], rows[-24:]
    spec = TaskSpec("paid_condition", "", ("a", "b", "c", "meter"), (), (),
        feature_bounds={name: (-1.5, 1.5) for name in ("a", "b", "c")}, max_probe_length=64)
    program = Program.from_dict(root04["state"]["program"])
    steps = [{"gate_coordinate": {"op": "feature", "name": item["condition_field"]},
              "threshold_interval": item["threshold_interval"], "threshold": item["threshold"]}
             for item in root04["state"]["conditions"]]
    selected = {"candidate": Candidate("paid_response", program, "research_program"),
        "research_program": {"selection_steps": steps}, "fallback": 0.,
        "training_loss": loss(program, training, 0., .5),
        "validation_loss": loss(program, validation, 0., .5),
        "condition_uncertainty": deepcopy(root04["state"]),
        "condition_research_returns": deepcopy(root04["previous"])}
    partial = deepcopy(root04["partial_competitor"])
    saved = partial["candidate"]
    partial["candidate"] = Candidate(saved["candidate_id"],
        Program.from_dict(saved["program"]), saved["operation"])
    partial["validation_loss"] = loss(partial["candidate"].program, validation, 0., .5)
    actions = [{"action": "distinguish_paid_condition_duration",
                "competing_structures": partial["structure_nominations"]}]
    for experiment in root04["previous_experiments"]:
        actions.extend([{**experiment, "action": "plan_competing_observed_conditions"},
            {"action": "revise_competing_observed_conditions", "measured_rows": 24,
             "research_return": experiment["research_return"]}])
    existing = len(actions)
    agent = UnifiedResearchAgent(ResearchConfig(observation_budget=320, compute_budget=320000,
        use_memory=False, minimum_improvement=1e-7), measurement_budget=320)
    budget, purchases = SearchBudget(77126), []

    def acquire(channels, purpose, *, probe):
        purchases.append(probe)
        return []  # Stop at the next purchase; the fixture contains only paid evidence.

    result, retained, finalists = research_observed_condition_competition(agent, spec,
        selected, training, validation, [partial], budget, actions, acquire=acquire,
        remaining=lambda: {"observation_remaining": 116, "measurement_remaining": 116},
        channels=("meter",), measurement_rate=1)
    current = actions[existing:]
    choice, = [item for item in current if item["action"] == "choose_condition_research_investment"]
    assert choice["research_focus"] == "distinguish_current_response"
    assert not any(item["action"] == "fit_partial_structure_competitors" for item in current)
    assert finalists == [partial] and retained == training
    assert result["candidate"].program == program
    # Cold 0.162 replay of these same paid rows, nominations, and limits spent
    # 29,961 planning units (14,940 on another partial fit), 30,633 total.
    assert 0 < choice["planning_compute_cost"] < 29961
    assessment = sum(item["compute_cost"] for item in current
                     if item["action"] == "retain_observed_condition_question")
    assert budget.spent == assessment + choice["planning_compute_cost"] < 30633
    assert choice["buy"] and len(purchases) == 1


def test_misses_remain_for_same_background_and_fade_continuously_for_distant_backgrounds():
    question, experiment, history = state(), scoped_plan(), None
    for _ in range(4):
        history = paid(history, question, question, experiment)
    restored = json.loads(json.dumps(history))
    frozen = deepcopy(restored)
    same = decide(question, experiment, restored)
    nearby = decide(question, scoped_plan(1e-6), restored)
    distant = decide(question, scoped_plan(1.), restored)
    assert same["current_question_trials"] == 4
    assert same["sampling_reliability"] < distant["sampling_reliability"] < 1.
    assert nearby["sampling_reliability"] == pytest.approx(same["sampling_reliability"], abs=1e-9)
    assert same["expected_information_value"] < distant["expected_information_value"]
    assert restored == frozen


def test_global_information_transfer_keeps_prediction_credit_but_scopes_sampling_misses():
    spec = TaskSpec("sampling", "", ("x", "meter"), (), (), feature_bounds={"x": (-1., 1.)})
    selected = {"candidate": Candidate("response", Program.from_dict(state()["program"]), "retained")}
    policy = condition_followup_policy(spec, [], selected, {})
    record_condition_followup(policy, before_loss=1., after_loss=.8,
        observation_cost=24, measurement_cost=24, compute_cost=2000)
    question, experiment = state(), scoped_plan()
    for _ in range(4):
        outcome = paid(None, question, question, experiment)["last_outcome"]
        record_condition_information(policy, outcome)
    record = policy.action_record(CONDITION_FOLLOWUP_ACTION)
    frozen = deepcopy(record)
    same = decide(question, experiment, marginal_record=record)
    distant = decide(question, scoped_plan(1.), marginal_record=record)
    assert same["information_yield"] < distant["information_yield"] < 1.
    assert same["empirical_marginal_prediction_gain"] == pytest.approx(.2)
    assert distant["empirical_marginal_prediction_gain"] == same["empirical_marginal_prediction_gain"]
    assert same["marginal_prediction_trials"] == distant["marginal_prediction_trials"] == 1
    assert record == frozen


def test_unscoped_old_information_cannot_discredit_a_new_measured_sampling_model():
    question, history = state(), None
    for _ in range(4):
        history = paid(history, question, question, plan())
    current = decide(question, scoped_plan(), history)
    assert current["sampling_reliability"] == 1.
    assert current["current_question_trials"] == 0
    assert history["batches"] == 4 and history["observation_cost"] == 96


def test_paid_local_readings_change_the_forecast_and_repeated_backgrounds_keep_nearby_misses():
    far = [Observation({"x": -1., "meter": value}, 0.)
           for value in (-2., -1., 1., 2.) for _ in range(4)]
    local = [Observation({"x": 1., "meter": value}, 0.)
             for value in (-.5, -.3, -.2, -.1, .1, .2, .3, .5)]
    paid_rows = [*far, *local]
    budget = SearchBudget(1000)
    _, residuals, context = _local_residuals(paid_rows,
        [row.features["meter"] for row in paid_rows], ["x"], {"x": (-1., 1.)}, {"x": 1.}, budget)
    assert len(residuals) == len(local)
    assert budget.spent >= 2 * len(paid_rows)
    _, _, nearby = _local_residuals(paid_rows,
        [row.features["meter"] for row in paid_rows], ["x"], {"x": (-1., 1.)}, {"x": 1. - 1e-6}, budget)
    assert sampling_applicability(context, nearby) == pytest.approx(1., abs=1e-9)
    before = _interval_exposure(sorted(residuals), -.05, .05, budget)
    acquired = [Observation({"x": 1., "meter": .5 + index * .001}, 0.) for index in range(24)]
    measured = [*paid_rows, *acquired]
    _, updated, _ = _local_residuals(measured,
        [row.features["meter"] for row in measured], ["x"], {"x": (-1., 1.)}, {"x": 1.}, budget)
    after = _interval_exposure(sorted(updated), -.05, .05, budget)
    assert len(updated) == len(local) + len(acquired)
    assert 0. < after["estimated_probability"] < before["estimated_probability"]


def test_real_paid_readings_refit_fixed_background_forecast_and_pay_for_the_update(root04):
    spec = TaskSpec("paid_condition", "", ("a", "b", "c", "meter"), (), (),
        feature_bounds={name: (-1.5, 1.5) for name in ("a", "b", "c")}, max_probe_length=64)
    rows = observations(root04["observations"])
    experiment = deepcopy(root04["previous_experiments"][-1])
    point = experiment["fixed_control_background"]
    acquired = [row for row in rows if all(row.features[name] == value for name, value in point.items())]
    earlier = [row for row in rows if row not in acquired]
    assert len(acquired) == 24
    experiment["probe"] = Probe("paid_fixed_background", tuple(dict(point) for _ in acquired))
    frozen = deepcopy(experiment)
    first_budget, next_budget = SearchBudget(15000), SearchBudget(15000)
    before = refresh_condition_sampling(spec, earlier, experiment, first_budget)
    after = refresh_condition_sampling(spec, rows, before, next_budget)
    assert before["paid_reading_rows"] + 24 == after["paid_reading_rows"] == len(rows)
    assert before["probe"] == after["probe"] == experiment["probe"]
    assert before["fixed_control_background"] == after["fixed_control_background"] == point
    assert before["sampling_context"]["normalized_background"] == after["sampling_context"]["normalized_background"]
    assert before["interval_exposure"][0]["estimated_probability"] != pytest.approx(
        after["interval_exposure"][0]["estimated_probability"])
    assert 0 < before["sampling_update_compute_cost"] == first_budget.spent
    assert 0 < after["sampling_update_compute_cost"] == next_budget.spent
    assert experiment == frozen
