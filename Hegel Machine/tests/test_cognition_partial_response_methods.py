"""Unfinished response knowledge must change the real successor selection."""

from copy import deepcopy
import json

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_research import UnifiedResearchAgent

from test_cognition_decoded_observed_conditions import (
    DecodedObservedLab, assert_actual_accounting, assert_current_graph, random_probe)
from test_cognition_numeric_draft_replay import broad_source


def test_restored_partial_response_is_selected_and_retained_by_full_agent():
    source, original = broad_source()
    saved_source = deepcopy(original["research_program"])
    library = KnowledgeLibrary.from_dict(json.loads(json.dumps(source.to_dict())))
    settings = dict(width=3, placement="inside_window", names=("u", "v", "w"),
                    meter="reading", threshold=.173, meter_scale=1.7, response_scale=1.25)
    laboratory = DecodedObservedLab(seed=4511, **settings)
    config = ResearchConfig(seed=4512, observation_budget=320, compute_budget=320000,
        max_rounds=2, max_candidates=8, minimum_improvement=1e-7)
    report = UnifiedResearchAgent(config, library=library, measurement_budget=320).run(laboratory)
    choices = [item for item in report["actions"] if item["action"] == "select_partial_response_method"]
    assert choices and choices[0]["draft_id"] == original["draft_id"]
    assert choices[0]["requires_fresh_confirmation"]
    assert report["research_draft_id"] == original["draft_id"]
    assert report["confirmation_loss"] < 1e-12
    assert report["status"] == "unresolved" and not report["concepts_added"]
    fresh_lab = DecodedObservedLab(seed=4513, **settings)
    fresh = fresh_lab.acquire(random_probe(fresh_lab.spec, 4514, 64), ("reading",)).observations
    assert loss(Program.from_dict(report["program"]), fresh, report["fallback"], .5) < 1e-6
    assert_actual_accounting(report, laboratory)
    assert_current_graph(report, fresh)
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    assert restored.research_drafts[0]["research_program"] == saved_source
    assert any(item["parent_draft_id"] == original["draft_id"]
               for item in restored.research_drafts)
    assert not restored.concepts
    restored_report = json.loads(json.dumps(report))
    assert restored_report["condition_research_returns"] == report["condition_research_returns"]
    alternative = next(item for item in restored_report["continuation_state"]["alternatives"]
                       if item.get("research_draft_id") == original["draft_id"])
    assert 0 < alternative["condition_research_returns"]["batches"] <= report["condition_research_returns"]["batches"]
