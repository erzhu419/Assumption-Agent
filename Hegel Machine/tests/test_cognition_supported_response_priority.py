"""New paid time structure receives fitting funds before unrelated old trials."""

from copy import deepcopy
import json
from pathlib import Path

from hegel_machine.cognition.condition_response_actions import _observed_response_claims
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


def paid_episode(values):
    rows, history = [], []
    for item in values:
        rows.append(Observation(item["features"], item["target"], tuple(history)))
        history.append({**item["features"], "__target__": item["target"]})
    return rows


def paid_case(*, memory):
    fixture = json.loads((Path(__file__).parent / "fixtures" /
                          "supported_response_paid_root09.json").read_text())
    episodes = [paid_episode(values) for values in fixture["episodes"]]
    training = [row for episode in episodes for row in episode]
    validation = episodes[fixture["validation_episode"]]
    confirmation = paid_episode(fixture["confirmation_episode"])
    spec = TaskSpec("retained_paid_response", "Predict the original measured response.",
        tuple(fixture["feature_names"]), (), (), feature_bounds=fixture["feature_bounds"],
        max_probe_length=64)
    library = KnowledgeLibrary.from_dict({"research_drafts": fixture["research_drafts"]})
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7,
        max_candidates=8, use_memory=memory, proposer="search"), library=library)
    selected = {"candidate": Candidate("retained_response",
        Program.from_dict(fixture["retained_program"]), "retained")}
    return fixture, spec, agent, selected, training, validation, confirmation


def test_latest_paid_structure_refits_compact_method_before_broad_old_draft():
    fixture, spec, agent, selected, training, validation, confirmation = paid_case(memory=True)
    assert len(training) == 252 and len(confirmation) == 24
    assert all(row not in training for row in confirmation)
    before = deepcopy([draft["research_program"] for draft in agent.library.research_drafts])
    actions = deepcopy(fixture["duration_events"])
    event_count, budget = len(actions), SearchBudget(80000)
    claims = _observed_response_claims(agent, spec, training, selected, budget,
        actions, validation, fixture["current_supported_evidence"])
    best = min(claims, key=lambda item: max(item["training_loss"], item["validation_loss"]))
    assert max(best["training_loss"], best["validation_loss"]) < 1e-7
    assert loss(selected["candidate"].program, confirmation, 0., .5) > 1e-7
    assert loss(best["candidate"].program, confirmation, 0., .5) < 1e-7
    replay = [action for action in actions[event_count:] if action["action"] == "replay_research_draft"]
    assert len(replay) == 1 and replay[0]["draft_id"] == "research_draft_0002"
    assert replay[0]["response_basis_terms"] == 3
    assert replay[0]["fitting_priority"] == "current_supported_structure"
    assert best["candidate"].program.required_features() == {"a", "b", "meter"}
    current = [step for step in best["research_program"]["selection_steps"]
               if not step.get("historical") and step.get("width")]
    assert current and current[-1]["width"] == 5 and current[-1]["placement"] == "outside_window"
    assert budget.spent < 40000 and budget.remaining > 40000
    assert [draft["research_program"] for draft in agent.library.research_drafts] == before
    assert all(not draft.get("resolved_concept_id") for draft in agent.library.research_drafts)


def test_current_supported_response_competes_with_paid_alternatives_without_memory():
    fixture, spec, agent, selected, training, validation, confirmation = paid_case(memory=False)
    actions = deepcopy(fixture["duration_events"])
    event_count, budget = len(actions), SearchBudget(80000)
    claims = _observed_response_claims(agent, spec, training, selected, budget,
        actions, validation, fixture["current_supported_evidence"])
    fitting = [action for action in actions[event_count:]
               if action["action"] == "refit_observed_condition_response"]
    assert len(fitting) == 1 and fitting[0]["fitting_priority"] == "current_supported_structure"
    assert fitting[0]["candidate_count"] > 0 and fitting[0]["compute_cost"] > 0
    assert fitting[0]["evidence"]
    assert {item["condition_identified"] for item in fitting[0]["structure_nominations"]} == {True, False}
    best = min(claims, key=lambda item: max(item["training_loss"], item["validation_loss"]))
    assert max(best["training_loss"], best["validation_loss"]) < 1e-7
    assert loss(best["candidate"].program, confirmation, 0., .5) < 1e-7
    assert budget.spent <= 80000
