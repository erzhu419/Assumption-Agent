"""Paid failed transfers as well as successful reuse affect future knowledge."""

import json

from hegel_machine.cognition.controller import CognitiveAgent
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate
from test_cognition_runtime import _LinearEnvironment, _small_config


def test_rejected_source_records_real_comparison_cost_and_survives_session_serialization():
    library = KnowledgeLibrary()
    source, _ = library.learn(Candidate("known", Program("feature", name="x"), "refit"),
                              _LinearEnvironment("source").spec)
    result = CognitiveAgent(_small_config(), library).run(_LinearEnvironment("changed", slope=-1.))
    feedback = next(action.details for action in result.actions if action.action == "knowledge_return")
    rejected = [row for row in feedback["returns"] if row["source_id"] == source
                and row["template_id"] is None and not row["retained"]]
    assert rejected and rejected[0]["paired_gain"] < 0
    assert rejected[0]["comparison_compute"] > 0
    assert all(item["evidence_rows"] > 0 for item in rejected[0]["comparisons"])
    adapted = sum(action.compute_cost for action in result.actions if action.action == "adapt_knowledge")
    assert feedback["unique_adaptation_compute"] <= adapted
    assert feedback["episode_cost_is_context_only"]
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    concept = next(item for item in restored.concepts if item.concept_id == source)
    assert any(row.get("kind") == "research_return" and row["paired_gain"] < 0
               for row in concept.reuse_history)


def test_unneeded_state_is_removed_before_knowledge_becomes_a_callable_source():
    library = KnowledgeLibrary()
    x = Program("feature", name="x")
    wrapped = Program("accumulate", value=0., children=(x, Program("const", value=0.)))
    source, created = library.learn(Candidate("memory_coordinate", wrapped, "representation"),
                                    _LinearEnvironment("source").spec)
    assert created and library.concepts[0].program == x
    duplicate, created = library.learn(Candidate("current_coordinate", x, "refit"),
                                       _LinearEnvironment("same_law").spec)
    assert duplicate == source and not created
    result = CognitiveAgent(_small_config(strategy="fixed"), library).run(_LinearEnvironment("transfer"))
    assert result.selected_memory_id == source
    assert result.program.history_depth == 0
    feedback = next(action.details for action in result.actions if action.action == "knowledge_return")
    assert any(row["retained"] and row["comparison_compute"] > 0 for row in feedback["returns"])
