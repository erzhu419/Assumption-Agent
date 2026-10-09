"""Use the already purchased response counterexample before buying readings."""

from copy import deepcopy
import gzip
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.condition_response_actions import research_observed_condition_competition
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.research_graph import ResearchGraph
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


def paid_prefix():
    with gzip.open(Path(__file__).parent / "fixtures/condition_revision_paid_root02.json.gz", "rt") as stream:
        recorded = json.load(stream)
    episodes = []
    for episode in recorded["episodes"]:
        rows, history = [], []
        for item in episode["observations"]:
            rows.append(Observation(item["features"], item["target"], tuple(history)))
            history.append({**item["features"], "__target__": item["target"]})
        episodes.append(rows)
    validation = episodes[recorded["validation_episode"]]
    training = [row for index, episode in enumerate(episodes)
                if index != recorded["validation_episode"] for row in episode]
    program = Program.from_dict(recorded["program"])
    selected = {"candidate": Candidate("paid_response", program, "research_program"),
        "research_program": {"selection_steps": deepcopy(recorded["selection_steps"])},
        "condition_uncertainty": deepcopy(recorded["condition_uncertainty"]),
        "condition_research_returns": deepcopy(recorded["condition_research_returns"]),
        "training_loss": loss(program, training, 0., .5),
        "validation_loss": loss(program, validation, 0., .5), "fallback": 0.}
    spec = TaskSpec("paid_response_counterexample", "Predict measured responses under legal commands.",
        tuple(recorded["feature_names"]), (), (),
        feature_bounds={key: tuple(value) for key, value in recorded["feature_bounds"].items()},
        max_probe_length=recorded["max_probe_length"])
    return recorded, spec, training, validation, selected, deepcopy(recorded["paid_actions"])


def agent(strategy="adaptive"):
    return UnifiedResearchAgent(ResearchConfig(observation_budget=320, compute_budget=320000,
        use_memory=False, minimum_improvement=1e-7, strategy=strategy), measurement_budget=320)


def assert_paid_repair(result, before, training, validation, events):
    program = result["candidate"].program
    assert program != before["candidate"].program
    assert loss(program, training, 0., .5) < 1e-7
    assert loss(program, validation, 0., .5) < 1e-7
    assert result["condition_uncertainty"]["status"] == "unresolved"
    assert not result["condition_uncertainty"]["conditions"][0]["needs_threshold_revision"]
    revision, = [item for item in events if item["action"] == "revise_condition_on_paid_response"]
    assert revision["changed_program"] and revision["requires_fresh_confirmation"]
    assert revision["evidence_rows"] == 192 and revision["measured_rows"] == 0
    assert revision["after_loss"] < 1e-7 < revision["before_loss"]
    assert revision["after_validation_loss"] <= revision["before_validation_loss"] + 1e-7
    assert revision["compute_cost"] > 0
    saved = json.loads(json.dumps(result["research_program"]))
    graph = ResearchGraph.from_dict(saved["graph"])
    assert graph.nodes[saved["output_node"]].answer == program
    gate = next(item for item in saved["selection_steps"]
                if item["operation"] == "discover_conditional_coordinates")
    condition = result["condition_uncertainty"]["conditions"][0]
    assert gate["threshold"] == condition["threshold"]
    assert gate["threshold_interval"] == condition["threshold_interval"]
    returns, old_returns = result["condition_research_returns"], before["condition_research_returns"]
    for resource in ("batches", "observation_cost", "measurement_cost"):
        assert returns[resource] == old_returns[resource]
    assert returns["compute_cost"] == old_returns["compute_cost"] + revision["compute_cost"]
    assert returns["last_question"]["program"] == program.to_dict()


@pytest.mark.parametrize("remaining_rows", [0, 24])
def test_actual_paid_counterexample_repairs_without_funding_another_purchase(remaining_rows):
    recorded, spec, training, validation, selected, actions = paid_prefix()
    before, frozen_rows = deepcopy(selected), deepcopy([*training, *validation])
    assert len(frozen_rows) == 192
    condition = selected["condition_uncertainty"]["conditions"][0]
    assert condition["needs_threshold_revision"]
    assert condition["conditional_loss"] < 1e-7
    assert selected["condition_uncertainty"]["original_target_loss"] == pytest.approx(.05379979467487)
    # The old run repaired only after this additional batch, which contains
    # no reading in the interval already supported by the 192 purchased rows.
    left, right = condition["threshold_interval"]
    assert not any(left < row["features"]["meter"] < right
                   for row in recorded["later_experiment"]["observations"])

    def acquire(*args, **kwargs):
        pytest.fail("The available rows must remain reserved for fresh confirmation")

    start = len(actions)
    budget = SearchBudget(15000)
    result, updated, _ = research_observed_condition_competition(agent(), spec, selected,
        training, validation, [], budget, actions,
        acquire=acquire if remaining_rows else None,
        remaining=lambda: {"observation_remaining": remaining_rows,
                           "measurement_remaining": remaining_rows},
        channels=("meter",), measurement_rate=1)
    assert_paid_repair(result, before, training, validation, actions[start:])
    assert not any(item["action"] == "plan_competing_observed_conditions" for item in actions[start:])
    assert updated == training and [*training, *validation] == frozen_rows
    assert selected == before
    assert 0 < budget.spent <= budget.limit


def test_repair_precedes_a_purchase_and_json_resume_does_not_repeat_it():
    _, spec, training, validation, selected, actions = paid_prefix()
    before, frozen_rows = deepcopy(selected), deepcopy([*training, *validation])
    start = len(actions)
    purchases = []

    def acquire(channels, purpose, *, probe):
        events = actions[start:]
        revision = next(item for item in events if item["action"] == "revise_condition_on_paid_response")
        retained = next(item for item in reversed(events)
                        if item["action"] == "retain_observed_condition_question")
        assert revision["after_loss"] < 1e-7 and revision["changed_program"]
        assert not retained["conditions"][0]["needs_threshold_revision"]
        assert retained["conditions"][0]["threshold"] != before["condition_uncertainty"]["conditions"][0]["threshold"]
        assert events.index(revision) < next(index for index, item in enumerate(events)
            if item["action"] == "plan_competing_observed_conditions")
        assert channels == ("meter",) and purpose == "distinguish_competing_observed_conditions"
        assert all(set(point) == set(spec.feature_bounds) for point in probe.sequence)
        purchases.append(probe)
        return []

    budget = SearchBudget(80000)
    result, updated, _ = research_observed_condition_competition(agent(), spec, selected,
        training, validation, [], budget, actions, acquire=acquire,
        remaining=lambda: {"observation_remaining": 128, "measurement_remaining": 128},
        channels=("meter",), measurement_rate=1)
    assert_paid_repair(result, before, training, validation, actions[start:])
    assert updated == training and [*training, *validation] == frozen_rows
    assert bool(purchases) == any(item["action"] == "plan_competing_observed_conditions"
                                  for item in actions[start:])
    payload = json.loads(json.dumps({"program": result["candidate"].program.to_dict(),
        **{key: result[key] for key in ("research_program", "condition_uncertainty",
            "condition_research_returns", "training_loss", "validation_loss", "fallback")}}))
    restored = {key: value for key, value in payload.items() if key != "program"}
    restored["candidate"] = Candidate("paid_response", Program.from_dict(payload["program"]), "research_program")
    restored_before, resumed_actions = deepcopy(restored), []
    resumed_budget = SearchBudget(15000)
    resumed, resumed_rows, _ = research_observed_condition_competition(agent(), spec, restored,
        training, validation, [], resumed_budget, resumed_actions, acquire=None)
    assert resumed["candidate"].program == result["candidate"].program
    assert resumed["condition_research_returns"] == result["condition_research_returns"]
    assert resumed_rows == training and [*training, *validation] == frozen_rows
    assert restored == restored_before
    assert not any(item["action"] == "revise_condition_on_paid_response" for item in resumed_actions)
    assert 0 < resumed_budget.spent <= resumed_budget.limit


def test_fixed_strategy_keeps_the_same_paid_counterexample_for_review():
    _, spec, training, validation, selected, actions = paid_prefix()
    before, start = deepcopy(selected), len(actions)
    result, updated, _ = research_observed_condition_competition(agent("fixed"), spec, selected,
        training, validation, [], SearchBudget(15000), actions, acquire=None)
    assert result["candidate"].program == before["candidate"].program
    assert loss(result["candidate"].program, training, 0., .5) > 1e-7
    assert result["condition_uncertainty"]["conditions"][0]["needs_threshold_revision"]
    assert result["condition_research_returns"] == before["condition_research_returns"]
    assert updated == training
    assert not any(item["action"] == "revise_condition_on_paid_response" for item in actions[start:])
