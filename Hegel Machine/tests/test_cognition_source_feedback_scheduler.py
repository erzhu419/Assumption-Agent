"""Focused scheduler checks for measured source-return feedback."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import Concept, KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.research_portfolio import choose_research_visit
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_revisits import offer_successor_knowledge
from hegel_machine.cognition.unified_session import UnifiedResearchSession
from test_cognition_unified_revisits import revisited_stream
from test_cognition_unified_source_returns import failed_successor, source_competition


def _condition(name="same_boundary"):
    return {"objective": "predict_response", "data": {"boundary": name}}


def _library(source_id, returns=()):
    concept = Concept(source_id, Program("feature", name="x"), .5,
                      f"learned_{source_id}", "refit")
    concept.reuse_history.extend(deepcopy(list(returns)))
    return KnowledgeLibrary(concepts=[concept])


def _return(condition, gain):
    return {"kind": "research_return", "condition": deepcopy(condition),
            "comparisons": [{"comparison_id": "paid",
                "program": _offer(0, "fixture_source", condition)["candidate"]["program"]}],
            "retained_normalized_gain": gain, "adaptation_compute": 10,
            "comparison_compute": 10}


def _offer(position, source, condition, *, label=""):
    return {"task_position": position, "source_task_position": 0,
            "candidate": {"kind": "successor_knowledge", "label": label,
                "program": Program("const", value=1.).to_dict(),
                "fallback": 0., "source_concept_ids": [source],
                "transfer_evidence": [{"source_id": source,
                    "condition": deepcopy(condition)}]},
            "relative_gain": 1., "expected_loss_reduction": 1.,
            "loss_kind": "squared_error",
            "minimum_resources": {"observation_cost": 1,
                "measurement_cost": 1, "compute_cost": 1}}


def _session(library):
    return UnifiedResearchSession(
        ResearchConfig(observation_budget=8, compute_budget=100,
                       active_probes=False, seed=911),
        tasks=[{"task_index": 0, "seed": 911}], library=library,
        measurement_budget=8)


def test_source_feedback_state_reports_positive_negative_and_unknown():
    condition = _condition()
    other = _condition("different_boundary")
    library = KnowledgeLibrary(concepts=[
        Concept("positive", Program("feature", name="x"), .5, "learned_positive", "refit"),
        Concept("negative", Program("feature", name="x"), .5, "learned_negative", "refit"),
        Concept("unknown", Program("feature", name="x"), .5, "learned_unknown", "refit"),
    ])
    library.concepts[0].reuse_history.append(_return(condition, .4))
    library.concepts[1].reuse_history.append(_return(condition, -.3))
    session = _session(library)

    positive = session._source_feedback_state(_offer(0, "positive", condition))
    negative = session._source_feedback_state(_offer(0, "negative", condition))
    unknown = session._source_feedback_state(_offer(0, "unknown", condition))
    unmatched = session._source_feedback_state(_offer(0, "positive", other))

    assert positive[1:] == (1, ["positive"], positive[3])
    assert positive[0] > 0.
    assert negative[1:] == (1, ["negative"], negative[3])
    assert negative[0] < 0.
    assert unknown == (0., 0, ["unknown"], unknown[3])
    assert unmatched == (0., 0, ["unknown"], unmatched[3])


def test_positive_unresolved_return_is_provisional_not_confirmed():
    """A positive paired gain cannot bias reuse before the successor resolves."""
    condition = _condition()
    provisional = _return(condition, .4)
    provisional["confirmation_status"] = "provisional"
    session = _session(_library("provisional", [provisional]))

    state = session._source_feedback_state(_offer(0, "provisional", condition))

    assert state[0] == 0.
    assert state[1] == 1
    assert state[2] == ["provisional"]


def test_negative_return_remains_actionable_alongside_provisional_positive():
    condition = _condition()
    provisional = _return(condition, .4)
    provisional["confirmation_status"] = "provisional"
    negative = _return(condition, -.2)
    negative["confirmation_status"] = "rejected"
    session = _session(_library("mixed", [provisional, negative]))

    state = session._source_feedback_state(_offer(0, "mixed", condition))

    assert state[2] == ["negative"]
    assert state[0] < 0.


def test_refresh_source_feedback_withdraws_all_negative_pending_offer():
    condition = _condition()
    session = _session(_library("bad", [_return(condition, -.3)]))
    session.state["pending_revisits"] = [_offer(0, "bad", condition, label="bad")]

    session._refresh_source_feedback()

    assert session.state["pending_revisits"] == []
    event = session.state["scheduler_events"][-1]
    assert event["action"] == "withdraw_successor_after_negative_return"
    assert event["source_concept_ids"] == ["bad"]
    assert event["matching_returns"] == 1


def test_positive_source_score_breaks_equal_visit_gain_tie():
    condition = _condition()
    library = KnowledgeLibrary(concepts=[
        Concept("unknown", Program("feature", name="x"), .5, "learned_unknown", "refit"),
        Concept("positive", Program("feature", name="x"), .5, "learned_positive", "refit"),
    ])
    library.concepts[1].reuse_history.append(_return(condition, .4))
    session = _session(library)
    session.state["pending_revisits"] = [
        _offer(0, "unknown", condition, label="unknown"),
        _offer(0, "positive", condition, label="positive"),
    ]
    session._refresh_source_feedback()

    def allocate(offer, others, remaining):
        return dict(offer["minimum_resources"])

    choice = choose_research_visit(
        session.state["pending_revisits"],
        {"observation_cost": 4, "measurement_cost": 4, "compute_cost": 20},
        SearchBudget(20), allocation_for=allocate)

    assert choice["offer"]["candidate"]["label"] == "positive"
    assert choice["decision"]["selected_indices"] == [1]
    assert choice["decision"]["selected_knowledge_return_score"] > 0.


def test_source_feedback_json_roundtrip_is_idempotent():
    condition = _condition()
    session = _session(_library("positive", [_return(condition, .4)]))
    session.state["pending_revisits"] = [_offer(0, "positive", condition, label="positive")]
    session._refresh_source_feedback()
    saved = json.loads(json.dumps(session.to_dict()))

    restored = UnifiedResearchSession(state=saved)
    assert restored.to_dict() == saved
    restored._refresh_source_feedback()
    assert restored.to_dict() == saved


def test_actual_positive_return_refreshes_queued_offer_without_rescreen(revisited_stream):
    """A paid successor return updates the old queued quote in place.

    ``queued`` is the real pre-visit quote and ``final`` contains the
    ``research_return`` produced by executing that quote.  Reusing both
    checkpoints keeps the feedback causal without manufacturing an outcome in
    the scheduler test.
    """
    queued = revisited_stream["queued"]
    final = revisited_stream["final"]
    session = UnifiedResearchSession(state=final, lab_factory=revisited_stream["factory"])
    offer = deepcopy(queued["pending_revisits"][0])
    before_candidate = deepcopy(offer["candidate"])
    before_screening = session.state["screening_compute_cost"]
    before_opened = len(revisited_stream["opened"])
    session.state["pending_revisits"] = [offer]

    session._refresh_source_feedback()

    refreshed = session.state["pending_revisits"]
    assert len(refreshed) == 1
    assert refreshed[0]["candidate"] == before_candidate
    assert refreshed[0]["source_return_status"] == ["positive"]
    assert refreshed[0]["knowledge_return_matches"] == 1
    assert refreshed[0]["knowledge_return_score"] > 0.
    assert session.state["screening_compute_cost"] == before_screening
    assert len(revisited_stream["opened"]) == before_opened
    event = session.state["scheduler_events"][-1]
    assert event["action"] == "refresh_source_feedback"
    assert event["compute_cost"] == 0
    assert event["scheduled"] is True


def test_actual_positive_return_breaks_near_tie_against_unknown_source(revisited_stream):
    """Measured source value changes the next visit, without changing its gain."""
    queued = revisited_stream["queued"]
    final = revisited_stream["final"]
    session = UnifiedResearchSession(state=final, lab_factory=revisited_stream["factory"])

    positive = deepcopy(queued["pending_revisits"][0])
    unknown = deepcopy(positive)
    unknown["candidate"]["source_concept_ids"] = ["unmeasured_rival"]
    unknown["candidate"]["transfer_evidence"] = [
        deepcopy(unknown["candidate"]["transfer_evidence"][0])]
    unknown["candidate"]["transfer_evidence"][0]["source_id"] = "unmeasured_rival"
    session.library.concepts.append(Concept(
        "unmeasured_rival", Program("feature", name="u"), .5,
        "unmeasured_rival", "refit"))
    session.state["pending_revisits"] = [positive, unknown]
    session._refresh_source_feedback()

    assert [offer["source_return_status"] for offer in session.state["pending_revisits"]] == [
        ["positive"], ["unknown"]]
    assert positive["expected_loss_reduction"] == unknown["expected_loss_reduction"]

    def allocate(offer, others, remaining):
        return dict(offer["minimum_resources"])

    choice = choose_research_visit(
        session.state["pending_revisits"],
        {"observation_cost": 200, "measurement_cost": 200, "compute_cost": 10000},
        SearchBudget(10000), allocation_for=allocate)

    assert choice["offer"] is session.state["pending_revisits"][0]
    assert choice["decision"]["selected_indices"] == [0]
    assert choice["decision"]["selected_expected_loss_reduction"] == pytest.approx(
        unknown["expected_loss_reduction"])
    assert choice["decision"]["selected_knowledge_return_score"] > 0.


def test_actual_negative_return_withdraws_pending_offer_without_rescreen(source_competition):
    """A measured failed successor is withdrawn before another visit is bought."""
    case, good, warm, cold = source_competition
    session = _session(warm.library)
    offer = deepcopy(case["offer"])
    offer.update(task_position=0, source_task_position=0, screening_compute_cost=0)
    rival = deepcopy(offer)
    rival["candidate"]["source_concept_ids"] = [good]
    rival["candidate"]["transfer_evidence"][0]["source_id"] = good
    session.state["pending_revisits"] = [offer, rival]
    before_opened = len(session.state.get("scheduler_events", []))

    session._refresh_source_feedback()

    assert len(session.state["pending_revisits"]) == 1
    assert session.state["pending_revisits"][0]["candidate"]["source_concept_ids"] == [good]
    assert session.state["pending_revisits"][0]["source_return_status"] == ["unknown"]
    event = session.state["scheduler_events"][-1]
    assert event["action"] == "withdraw_successor_after_negative_return"
    assert event["source_concept_ids"] == [case["source_id"]]
    assert event["matching_returns"] == 1
    assert event["compute_cost"] == 0
    assert len(session.state["scheduler_events"]) == before_opened + 1


def test_real_positive_and_negative_returns_reprice_real_pending_successors(source_competition):
    """Actual accepted/rejected continuations update two real queued offers."""
    case, good, warm, cold = source_competition
    library = KnowledgeLibrary.from_dict(json.loads(json.dumps(warm.library.to_dict())))
    config = ResearchConfig(observation_budget=100, compute_budget=20000,
        minimum_improvement=1e-7, max_candidates=16, active_probes=False,
        risk_learning=False, seed=1237)

    # First earn a positive return from the repaired source on a fresh target
    # report. This is a real continuation, not a fabricated history entry.
    positive_previous = deepcopy(case["previous"])
    positive_previous["task_id"] = "positive_source_target"
    positive_lab = type(case["lab"])(case["spec"])
    positive_agent = UnifiedResearchAgent(config, library=library, measurement_budget=0)
    positive_budget = SearchBudget(4000)
    positive_offer = offer_successor_knowledge(
        positive_agent, positive_lab, positive_previous, {good}, positive_budget)
    assert positive_offer is not None
    positive_report = positive_agent.run(
        positive_lab, previous=positive_previous,
        continuation_candidate=positive_offer["candidate"])
    positive_return = next(row for row in positive_report["knowledge_return"]["returns"]
                           if row["source_id"] == good)
    assert positive_return["retained"] and positive_return["paired_gain"] > 0.
    assert positive_report["status"] == "completed"
    assert positive_return["visit_context"]["final_loss"] < config.minimum_improvement

    # Independently screen two successor offers from the same paid prefix.
    # Rescreening the failed source can produce a different refit. Its
    # source's old negative return must not withdraw this untested program.
    pending_previous = deepcopy(case["previous"])
    pending_previous["task_id"] = "pending_source_target"
    pending = []
    for position, source_id in enumerate((case["source_id"], good)):
        lab = type(case["lab"])(case["spec"])
        agent = UnifiedResearchAgent(config, library=library, measurement_budget=0)
        budget = SearchBudget(4000)
        offer = offer_successor_knowledge(agent, lab, pending_previous, {source_id}, budget)
        assert offer is not None
        offer.update(task_position=position, source_task_position=2)
        pending.append(offer)

    # The matched no-return arm sees both real offers. Its first purchase
    # should favor the larger immediate gain with a cheaper one-confirmation
    # quote, even though the lower-gain source occupies the earlier position.
    cold_library = KnowledgeLibrary.from_dict(json.loads(json.dumps(cold.library.to_dict())))
    cold_pending = []
    for position, source_id in enumerate((case["source_id"], good)):
        lab = type(case["lab"])(case["spec"])
        agent = UnifiedResearchAgent(config, library=cold_library, measurement_budget=0)
        budget = SearchBudget(4000)
        offer = offer_successor_knowledge(agent, lab, pending_previous, {source_id}, budget)
        assert offer is not None
        offer.update(task_position=position, source_task_position=2)
        cold_pending.append(offer)
    cold_session = UnifiedResearchSession(config, tasks=[
        {"task_index": 0, "seed": 1237}, {"task_index": 0, "seed": 1238}],
        library=cold_library, measurement_budget=0)
    cold_session.state["pending_revisits"] = cold_pending
    cold_session._refresh_source_feedback()

    def allocate(offer, others, remaining):
        return dict(offer["minimum_resources"])

    cold_choice = choose_research_visit(
        cold_session.state["pending_revisits"],
        {"observation_cost": 100, "measurement_cost": 100, "compute_cost": 10000},
        SearchBudget(10000), allocation_for=allocate)
    assert cold_pending[0]["minimum_resources"]["observation_cost"] == 24
    assert cold_pending[1]["minimum_resources"]["observation_cost"] == 12
    assert "paid_selection" not in cold_pending[0]["candidate"]
    assert "paid_selection" in cold_pending[1]["candidate"]
    assert cold_pending[1]["expected_loss_reduction"] > cold_pending[0]["expected_loss_reduction"]
    assert cold_choice["offer"]["candidate"]["source_concept_ids"] == [good]
    assert cold_choice["decision"]["selected_indices"] == [1, 0]

    frozen_failed = deepcopy(case["offer"])
    frozen_failed.update(task_position=0, source_task_position=2)
    assert (Program.from_dict(frozen_failed["candidate"]["program"]).expanded().simplified()
            != Program.from_dict(pending[0]["candidate"]["program"]).expanded().simplified())
    # Keep the actual previously quoted binding alongside the fresh refit.
    # Only that frozen quote has the measured counterexample.
    pending.insert(0, frozen_failed)

    session = UnifiedResearchSession(config, tasks=[
        {"task_index": 0, "seed": 1237}, {"task_index": 0, "seed": 1238}],
        library=library, measurement_budget=0)
    session.state["pending_revisits"] = pending
    before_screening = session.state["screening_compute_cost"]
    session.lab_factory = lambda item: (_ for _ in ()).throw(
        AssertionError("source feedback must not reopen a laboratory"))
    session._refresh_source_feedback()

    assert frozen_failed not in session.state["pending_revisits"]
    assert [item["candidate"]["source_concept_ids"]
            for item in session.state["pending_revisits"]] == [[case["source_id"]], [good]]
    assert session.state["pending_revisits"][0]["source_return_status"] == ["unknown"]
    retained = session.state["pending_revisits"][1]
    assert retained["source_return_status"] == ["positive"]
    assert retained["knowledge_return_matches"] == 1
    assert retained["knowledge_return_score"] > 0.
    assert session.state["screening_compute_cost"] == before_screening
    chosen = choose_research_visit(
        session.state["pending_revisits"],
        {"observation_cost": 100, "measurement_cost": 100, "compute_cost": 10000},
        SearchBudget(10000), allocation_for=allocate)
    assert chosen["offer"]["candidate"]["source_concept_ids"] == [good]
    # Both arms choose the denser first purchase. Feedback removes the
    # disproven binding, while the distinct refit remains a possible companion.
    assert chosen["decision"]["selected_indices"] == [1, 0]
    actions = session.state["scheduler_events"]
    assert [item["action"] for item in actions] == [
        "withdraw_successor_after_negative_return", "refresh_source_feedback"]
    assert all(item["compute_cost"] == 0 for item in actions)
