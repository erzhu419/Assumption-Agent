"""A paid narrow bracket must change the next actual research purchase."""

from copy import deepcopy
import gzip
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.condition_research_returns import consider_condition_research
from hegel_machine.cognition.condition_response_actions import research_observed_condition_competition
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.observed_condition_experiments import plan_observed_condition_probe
from hegel_machine.cognition.observed_condition_uncertainty import _replace_threshold
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


def paid_prefix():
    with gzip.open(Path(__file__).parent / "fixtures/observed_condition_paid_root02.json.gz", "rt") as f:
        recorded = json.load(f)
    episodes = []
    for episode in recorded["episodes"]:
        rows, history = [], []
        for item in episode:
            rows.append(Observation(item["features"], item["target"], tuple(history)))
            history.append({**item["features"], "__target__": item["target"]})
        episodes.append(rows)
    validation = episodes[recorded["validation_episode"]]
    training = [row for i, episode in enumerate(episodes)
                if i != recorded["validation_episode"] for row in episode]
    spec = TaskSpec("paid_condition_prefix", "Predict measured responses under legal commands.",
        tuple(recorded["feature_names"]), (), (),
        feature_bounds={k: tuple(v) for k, v in recorded["feature_bounds"].items()}, max_probe_length=64)
    program = Program.from_dict(recorded["program"])
    selected = {"candidate": Candidate("paid_response", program, "research_program"),
        "research_program": {"selection_steps": recorded["selection_steps"]},
        "condition_uncertainty": recorded["condition_uncertainty"],
        "condition_research_returns": recorded["condition_research_returns"],
        "training_loss": loss(program, training, 0., .5),
        "validation_loss": loss(program, validation, 0., .5), "fallback": 0.}
    return recorded, spec, training, validation, selected


def test_actual_current_bracket_and_paid_failures_rewrite_the_next_forecast():
    recorded, spec, training, validation, selected = paid_prefix()
    rows = [*training, *validation]
    state = selected["condition_uncertainty"]
    interval = state["conditions"][0]["threshold_interval"]
    previous = [x for x in recorded["paid_actions"] if x["action"] == "plan_competing_observed_conditions"]
    budget = SearchBudget(15000)
    plan = plan_observed_condition_probe(spec, rows, selected, budget, previous_experiments=previous)
    assert plan is not None
    assert len(rows) == state["fit_rows"] == 144
    assert plan["threshold_interval"] == interval
    assert [x["threshold_interval"] for x in plan["interval_exposure"]] == [interval]
    assert plan["threshold_intervals"][0]["source_operation"] == "current_paid_condition_support"
    assert plan["previous_experiment_count"] == 4
    assert plan["experiment_revision_basis"] == "unmeasured_temporal_boundary"
    assert plan["forecast_length"] == len(plan["probe"].sequence) == 24
    assert all(plan["fixed_control_background"] != x["fixed_control_background"] for x in previous)
    assert all(set(point) == set(spec.feature_bounds) for point in plan["probe"].sequence)
    assert 0 < plan["compute_cost"] == budget.spent <= budget.limit
    decision = consider_condition_research(state, plan, selected["condition_research_returns"],
        observation_remaining=recorded["observation_remaining"],
        measurement_remaining=recorded["measurement_remaining"],
        compute_remaining=200000 - budget.spent, measurement_rate=1, compute_quote=16008,
        observation_reserve=24, measurement_reserve=24, compute_reserve=96)
    # The recorded old wide recipe forecast called this nearly certain. The
    # remaining paid-supported gap cannot justify five more such purchases.
    assert decision["probability_of_new_information"] < .2
    assert not decision["buy"]
    assert decision["reason"] == "retain_question_for_more_valuable_future_evidence"


@pytest.mark.parametrize("refuted_cut", [False, True])
def test_consumer_refreshes_old_finalist_support_and_stops_an_uninformative_purchase(refuted_cut):
    recorded, spec, training, validation, selected = paid_prefix()
    finalist = {**selected, "candidate": Candidate("older_frontier",
        selected["candidate"].program, "research_program"), "condition_uncertainty": None}
    if refuted_cut:
        program = _replace_threshold(selected["candidate"].program, "meter",
            selected["condition_uncertainty"]["conditions"][0]["threshold"], .02)
        # This older cut genuinely fitted the original 48 paid rows. The new
        # purchased readings now refute it, so its broad bracket is not viable.
        original = [row for episode in recorded["episodes"][:4] for row in episode]
        first_rows, history = [], []
        for episode in recorded["episodes"][:4]:
            history = []
            for item in episode:
                first_rows.append(Observation(item["features"], item["target"], tuple(history)))
                history.append({**item["features"], "__target__": item["target"]})
        assert len(original) == 48 and loss(program, first_rows, 0., .5) < 1e-7
        assert loss(program, [*training, *validation], 0., .5) > 1e-7
        finalist["candidate"] = Candidate("older_frontier", program, "research_program")
        finalist.update(training_loss=0., validation_loss=0.)
    agent = UnifiedResearchAgent(ResearchConfig(observation_budget=320, compute_budget=200000,
        use_memory=False, minimum_improvement=1e-7), measurement_budget=320)
    actions = deepcopy(recorded["paid_actions"])
    purchases = []

    def acquire(channels, purpose, *, probe):
        purchases.append(probe)
        return []

    budget = SearchBudget(200000)
    result, updated, _ = research_observed_condition_competition(agent, spec, selected,
        training, validation, [finalist], budget, actions, acquire=acquire,
        remaining=lambda: {"observation_remaining": recorded["observation_remaining"],
                           "measurement_remaining": recorded["measurement_remaining"]},
        channels=("meter",), measurement_rate=1)
    assert not purchases
    assert updated == training
    assert result["candidate"].program == selected["candidate"].program
    assert result["condition_uncertainty"]["status"] == "unresolved"
    assert result["condition_uncertainty"]["expected_prediction_ambiguity"] > 1e-7
    choices = [x for x in actions if x["action"] == "choose_condition_research_investment"]
    assert len(choices) == 1 and not choices[0]["buy"]
    assert choices[0]["previous_experiment_count"] == 4
    assert choices[0]["reason"] == "retain_question_for_more_valuable_future_evidence"
    interval = result["condition_uncertainty"]["conditions"][0]["threshold_interval"]
    assert all(x["threshold_interval"] == interval for x in choices[0]["threshold_intervals"])
    assert choices[0]["probability_of_new_information"] < .2
    assert choices[0]["compute_spent_before_decision"] == choices[0]["planning_compute_cost"] > 0
    assert choices[0]["estimated_remaining_compute_cost"] >= choices[0]["quoted_compute_cost"]
    assert 0 < budget.spent <= budget.limit
