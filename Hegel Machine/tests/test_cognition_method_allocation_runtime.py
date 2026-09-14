"""A learned method changes session allocation and then earns actual feedback."""

import json

from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.research_allocation_demo import run_method_allocation
from hegel_machine.cognition.controller import CognitiveAgent
from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.models import ResearchConfig


def test_source_episode_teaches_a_method_that_revisits_and_solves_the_earlier_question():
    library = KnowledgeLibrary()
    report = run_method_allocation(library)
    summary = report["summary"]
    assert summary["earlier_question_unresolved"]
    assert summary["earlier_validation_loss"] > .5
    assert summary["source_validation_loss"] < 1e-10
    assert summary["selection_reason"] == "learned_method_opportunity"
    assert summary["selected_task_index"] == 0 and summary["unvisited_question_still_pending"]
    assert summary["method_steps_executed"][0]["name"] == "representation"
    identifier = summary["method_opportunity"]["method_id"]
    assert summary["method_steps_executed"][0]["method_id"] == identifier
    measured = next(step for step in summary["measured_method_returns"] if step["method_id"] == identifier)
    assert measured["gain"] > .9 and measured["compute_cost"] > 0 and measured["observation_cost"] > 0
    assert summary["continued_validation_loss"] < 1e-10
    for scope in ("source", "session"):
        assert summary[f"{scope}_compute_cost"] <= summary[f"{scope}_compute_budget"]
        assert summary[f"{scope}_observation_cost"] <= summary[f"{scope}_observation_budget"]
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    method = next(item for item in restored.methods.methods if item["method_id"] == identifier)
    assert method["confirmations"] >= 1
    returns = method["continuation_outcomes"]
    assert any(outcome["gain"] > .9 for outcome in returns)
    assert all(outcome["observation_cost"] > 0 and outcome["compute_cost"] > 0 for outcome in returns)
    assert any(step["method_id"] == identifier for step in report["checkpoint"]["workspaces"]["0"]["method_trace"])
    continuation = next(episode for episode in report["checkpoint"]["episodes"]
                        if episode["scheduler"].get("method_opportunity"))
    allocation = continuation["scheduler"]
    plan = allocation["research_opportunity"]["research_plan"]
    assert plan["max_rounds"] == 1
    assert plan["compute_budget"] == allocation["allocated_compute_budget"]
    assert plan["observation_budget"] == allocation["allocated_observation_budget"]
    assert continuation["compute_cost"] <= plan["compute_budget"] < 9000
    assert continuation["observation_cost"] <= plan["observation_budget"] < 48
    assert allocation["research_return"]["execution_complete"]


def test_actual_prior_method_continues_its_unfinished_sequence_on_the_revisited_target():
    library = KnowledgeLibrary()
    prior = CognitiveAgent(ResearchConfig(seed=503, partition="continual"), library).run(make_suite(503, "continual")[0])
    assert prior.validation_loss < 1e-10
    report = run_method_allocation(library)
    summary = report["summary"]
    trace = summary["measured_method_returns"]
    identifier = summary["method_opportunity"]["method_id"]
    first = next(step for step in trace if step["method_id"] == identifier and step["action"] == "refit")
    # Parameter fitting now makes a measured partial improvement on the new
    # trajectory. It still leaves a residual requiring the sequence's next
    # structural step; an exact zero return was an incidental older result.
    assert first["execution_complete"] and first["gain"] >= 0.
    assert first["before_loss"] >= first["after_loss"] > .002
    assert first["compute_cost"] > 0 and first["observation_cost"] > 0
    following = next(step for step in trace if step["method_id"] == first["method_id"] and step["action"] == "representation")
    assert trace.index(first) < trace.index(following)
    execution = [step for step in summary["method_steps_executed"] if step["method_id"] == identifier]
    assert [(step["step"], step["name"]) for step in execution] == [(0, "refit"), (1, "representation")]
    assert following["execution_complete"] and following["observation_cost"] > 0 and following["compute_cost"] > 0
    assert following["gain"] > .9
    assert summary["source_validation_loss"] < 1e-10
    assert summary["continued_validation_loss"] < 1e-10
