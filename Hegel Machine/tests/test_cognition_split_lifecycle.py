"""A paid split becomes usable knowledge, while supported sharing can survive."""

import json

import pytest

from hegel_machine.cognition.evolution_research import run_evolution_research
from hegel_machine.cognition.knowledge_evolution import KnowledgeEvolution
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.selection import rank_candidates


@pytest.fixture(scope="module")
def lifecycle():
    library = KnowledgeLibrary()
    report = run_evolution_research(library)
    return report, KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))


def test_split_is_accepted_on_a_different_new_trajectory_and_becomes_a_later_call(lifecycle):
    report, library = lifecycle
    summary = report["summary"]
    assert summary["split_proposed"] and summary["split_adopted"]
    assert summary["split_fresh_mse"] < 1e-10
    assert summary["split_transfer_called"] and summary["split_transfer_fresh_mse"] < 1e-10
    record = next(item for item in library.evolution.merges if item.get("split_concept_id") == summary["split_concept_id"])
    event = next(item for item in record["comparisons"] if item["operation"] == "split_adopted")
    assert event["fresh_rows"] == 8
    assert event["shared_loss"]["fresh"] > .1 > event["separate_loss"]["fresh"]
    # Equal candidate origins make the adoption a real rollback. A proposed
    # operation label is not used as a substitute for the selected computation.
    assert any(item["action"] == "rollback" for item in report["episodes"]["split"]["actions"])
    assert event["selected_operation"] == "composition"
    counterexample_points = {row["features"]["x"] for row in record["counterexamples"]}
    split_points = {row["features"]["x"] for row in report["episodes"]["split"]["research_state"]["observations"][-8:]}
    assert split_points.isdisjoint(counterexample_points)
    consumer = Program.from_dict(report["episodes"]["split_transfer"]["program"])
    assert consumer.op == "call" and consumer.definition.source_concept_id == summary["split_concept_id"]
    assert report["episodes"]["split_transfer"]["selected_memory_id"] == summary["split_concept_id"]
    assert summary["split_transfer"]["offered_calls"] > 1


def test_split_knowledge_keeps_component_dependencies_and_revises_the_failed_merge(lifecycle):
    report, library = lifecycle
    split_id = report["summary"]["split_concept_id"]
    record = next(item for item in library.evolution.merges if item.get("split_concept_id") == split_id)
    split = next(concept for concept in library.concepts if concept.concept_id == split_id)
    merged_id = record["merged_concept_id"]
    merged = next(concept for concept in library.concepts if concept.concept_id == merged_id)
    original_sources = set(report["summary"]["split_parent_ids"])
    assert len(original_sources) == 2 and merged_id not in original_sources
    assert set(split.parents) == original_sources
    assert split.status != "revoked" and any(scope.verified for scope in split.scopes)
    assert merged.status in {"restricted", "revoked"} and merged.counterexamples
    # Retiring an invalid compression must not retire the separately executable
    # source computations. Removing either real component must remove its user.
    retained = library.without([merged_id])
    assert split_id in {concept.concept_id for concept in retained.concepts}
    removed = library.without([next(iter(original_sources))])
    assert split_id not in {concept.concept_id for concept in removed.concepts}
    use = next(item for item in split.reuse_history if item["task_id"] == "evolution:split_transfer"
               and item.get("retained"))
    assert use["evidence_status"] == "measured" and use["comparison_compute"] > 0
    assert "evolution:split_transfer" in split.successful_tasks
    assert any(scope.task_id == "evolution:split_transfer" and scope.verified for scope in split.scopes)
    assert sum(item["task_id"] == "evolution:split_transfer" and item["kind"] == "executed_program"
               for item in split.reuse_history) == 1


def test_valid_merge_remains_selected_when_the_old_separate_law_loses_on_new_inputs(lifecycle):
    report, _ = lifecycle
    assert report["summary"]["valid_merge_retained"]
    assert report["summary"]["valid_merge_fresh_mse"] < 1e-12
    control = report["episodes"]["merge_control"]
    decision = next(action["details"] for action in control["actions"] if action["action"] == "merge_retained")
    assert decision["shared_loss"]["fresh"] < 1e-12
    assert decision["separate_loss"]["fresh"] > .1
    assert not any(action["action"] in {"split_adopted", "rollback"} for action in control["actions"])
    assert control["observation_cost"] == 8


def test_a_split_proposal_does_not_bind_new_knowledge_before_paid_selection(lifecycle, monkeypatch):
    report, library = lifecycle
    source = next(item for item in library.evolution.merges if item.get("split_concept_id"))
    evolution = KnowledgeEvolution.from_dict({"merges": [source]})
    record = evolution.merges[0]
    record["status"] = "counterexample_reopened"
    record.pop("split_concept_id")
    shared, separate = Program.from_dict(source["rewritten"]), Program.from_dict(source["original"])
    assert evolution.bind_concept("premature", separate) == []
    # New data favor sharing despite the previous proposal. Score first, then
    # forbid further predictions while completing the actual decision.
    rows = [Observation({"x": x}, 2*x*x) for x in (-1.6, -.7, .25, .55, 1.6)]
    current = Candidate("shared", shared, "concept_merge")
    proposed = Candidate("old_parent", separate, "composition")
    budget = SearchBudget(4 * len(rows))
    assessed = rank_candidates([current, proposed], rows, [], fresh=rows, fallback=0., q=.5,
                               budget=budget, complexity_weight=.001)
    assert assessed[0].candidate == current
    spec = TaskSpec("new_comparison", "Predict the measured response", ("x",), (), ())

    def no_unpaid_prediction(*args, **kwargs):
        raise AssertionError("recording the already-paid decision must not predict again")

    monkeypatch.setattr(Program, "predict", no_unpaid_prediction)
    events = evolution.record_comparison(shared, current, assessed, spec, len(rows))
    assert events[0]["operation"] == "split_rejected"
    assert record["status"] == "adopted" and "split_concept_id" not in record
    assert evolution.bind_concept("not_selected", separate) == []
    assert budget.spent == budget.limit
