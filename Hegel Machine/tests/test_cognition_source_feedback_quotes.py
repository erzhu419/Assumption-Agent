"""Pending ordinary successor quotes react to measured, scoped source returns."""

from copy import deepcopy
import json

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import Concept, KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.research_portfolio import choose_research_visit
from hegel_machine.cognition.unified_session import UnifiedResearchSession


def _condition(name="same_boundary"):
    return {"objective": "predict_response", "data": {"boundary": name}}


def _library(*sources):
    concepts = []
    for identifier, returns in sources:
        concept = Concept(identifier, Program("feature", name="x"), .5,
                          f"learned_{identifier}", "refit")
        concept.reuse_history.extend(deepcopy(returns))
        concepts.append(concept)
    return KnowledgeLibrary(concepts=concepts)


def _return(condition, gain, *, cost=10):
    return {"kind": "research_return", "condition": deepcopy(condition),
            "comparisons": [{"comparison_id": "paid", "accepted": gain > 0,
                "program": _offer(0, "fixture_source", condition)["candidate"]["program"]}],
            "retained_normalized_gain": gain, "adaptation_compute": cost,
            "comparison_compute": cost}


def _offer(position, source, condition, *, expected=1.0, label=""):
    return {"task_position": position, "source_task_position": 1,
            "candidate": {"kind": "successor_knowledge",
                "program": Program("const", value=expected).to_dict(),
                "fallback": 0.0, "source_concept_ids": [source],
                "transfer_evidence": [{"source_id": source,
                    "condition": deepcopy(condition)}], "label": label},
            "relative_gain": expected,
            "expected_loss_reduction": expected,
            "loss_kind": "squared_error",
            "minimum_resources": {"observation_cost": 1,
                "measurement_cost": 1, "compute_cost": 1}}


def _session(library):
    return UnifiedResearchSession(
        ResearchConfig(observation_budget=32, compute_budget=1000,
                       active_probes=False, seed=911),
        tasks=[{"task_index": 0, "seed": 911},
               {"task_index": 0, "seed": 912}],
        library=library, measurement_budget=32)


def test_matching_return_reprices_only_the_same_condition():
    matched = _condition()
    different = _condition("different_boundary")
    library = _library(
        ("source_good", [_return(matched, .4)]),
        ("source_unknown", []),
    )
    session = _session(library)
    exact = _offer(0, "source_good", matched, label="exact")
    different_offer = _offer(0, "source_good", different, label="different")
    unknown = _offer(1, "source_unknown", matched, label="unknown")
    session.state["pending_revisits"] = [exact, different_offer, unknown]

    session._refresh_source_feedback()

    pending = {item["candidate"]["label"]: item
               for item in session.state["pending_revisits"]}
    assert set(pending) == {"exact", "different", "unknown"}
    assert pending["exact"]["source_return_status"] == ["positive"]
    assert pending["exact"]["knowledge_return_matches"] == 1
    assert pending["exact"]["knowledge_return_score"] > 0.0
    assert pending["different"]["source_return_status"] == ["unknown"]
    assert pending["different"]["knowledge_return_score"] == 0.0
    assert pending["unknown"]["source_return_status"] == ["unknown"]
    assert not [event for event in session.state["scheduler_events"]
                if event["action"] == "withdraw_successor_after_negative_return"]


def test_negative_return_withdraws_only_its_bound_offer():
    condition = _condition()
    other_condition = _condition("other_boundary")
    library = _library(
        ("source_bad", [_return(condition, -.3)]),
        ("source_other", []),
    )
    session = _session(library)
    rejected = _offer(0, "source_bad", condition, label="rejected")
    same_source_other_condition = _offer(0, "source_bad", other_condition,
                                         label="same_source_other_condition")
    unrelated = _offer(1, "source_other", condition, label="unrelated")
    session.state["pending_revisits"] = [rejected, same_source_other_condition,
                                           unrelated]

    session._refresh_source_feedback()

    labels = [item["candidate"]["label"] for item in session.state["pending_revisits"]]
    assert labels == ["same_source_other_condition", "unrelated"]
    events = [event for event in session.state["scheduler_events"]
              if event["action"] == "withdraw_successor_after_negative_return"]
    assert len(events) == 1
    assert events[0]["source_concept_ids"] == ["source_bad"]
    assert events[0]["matching_returns"] == 1


def test_source_feedback_json_restore_does_not_duplicate_pending_quotes_or_events():
    condition = _condition()
    library = _library(("source_good", [_return(condition, .2)]),
                       ("source_unknown", []))
    session = _session(library)
    session.state["pending_revisits"] = [
        _offer(0, "source_good", condition, label="good"),
        _offer(1, "source_unknown", condition, label="unknown"),
    ]
    session._refresh_source_feedback()
    before_pending = deepcopy(session.state["pending_revisits"])
    before_events = deepcopy(session.state["scheduler_events"])

    restored = UnifiedResearchSession(state=json.loads(json.dumps(session.to_dict())))
    restored._refresh_source_feedback()

    assert restored.state["pending_revisits"] == before_pending
    assert restored.state["scheduler_events"] == before_events


def test_measured_return_breaks_equal_successor_investment_tie():
    condition = _condition()
    library = _library(
        ("source_low", [_return(condition, .1, cost=10)]),
        ("source_high", [_return(condition, .8, cost=10)]),
    )
    session = _session(library)
    session.state["pending_revisits"] = [
        _offer(0, "source_low", condition, label="low"),
        _offer(0, "source_high", condition, label="high"),
    ]
    session._refresh_source_feedback()
    offers = session.state["pending_revisits"]

    def allocate(offer, others, resources):
        return dict(offer["minimum_resources"])

    choice = choose_research_visit(offers,
        {"observation_cost": 10, "measurement_cost": 10, "compute_cost": 20},
        SearchBudget(20), allocation_for=allocate)

    assert choice["offer"]["candidate"]["label"] == "high"
    assert choice["decision"]["selected_knowledge_return_score"] > 0.0
    assert choice["decision"]["selected_indices"] == [1]
