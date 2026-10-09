"""Ordinary Unified successors earn source returns from actual new responses."""

from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from hegel_machine.cognition.evaluation import initial_prediction, loss
from hegel_machine.cognition.knowledge_returns import return_priority, transfer_condition
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import MeasurementBatch
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.meta_prior_research import _sources
from hegel_machine.cognition.research_programs import _plan
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_revisits import (
    _joint_successor_candidate, _successor_candidates, offer_successor_knowledge)
from hegel_machine.cognition.unified_session import UnifiedResearchSession
from test_cognition_conflict_transfer_refit import ConflictLagEnvironment, conflict_learning, refit
from test_cognition_unified_revisits import revisited_stream


def test_actual_success_records_source_comparisons_and_only_source_work(revisited_stream):
    stream = revisited_stream
    report = stream["final"]["results"][0]["research"]
    visit, = stream["final"]["revisits"]
    feedback = report["knowledge_return"]
    origins = visit["candidate"]["transfer_evidence"]
    records = feedback["returns"]
    assert records and any(row["retained_normalized_gain"] > 0 for row in records)
    assert {row["source_id"] for row in records} <= {row["source_id"] for row in origins}
    purchases = {item["probe_id"]: item for item in stream["purchases"][stream["before_purchases"]:]}
    comparisons = {event["probe_id"]: event for event in report["actions"]
                   if event["action"] == "compare_successor_knowledge_on_fresh_response"}
    for record in records:
        origin = next(item for item in origins if item["source_id"] == record["source_id"])
        assert record["condition"] == origin["condition"]
        assert record["adaptation_compute"] == sum(item["compute_cost"] for item in record["shared_adaptation_work"])
        assert record["comparison_compute"] == sum(item["compute_cost"] for item in record["comparisons"])
        # Candidate adaptation was already paid by screening; only the frozen
        # comparison is executed in this confirmation visit.
        assert record["adaptation_compute"] <= stream["final"]["screening_compute_cost"]
        assert record["comparison_compute"] == report["visit_costs"]["compute_cost"]
        for comparison in record["comparisons"]:
            event = comparisons[comparison["comparison_id"]]
            rows = purchases[event["probe_id"]]["rows"]
            assert comparison["loss"] == pytest.approx(loss(
                Program.from_dict(comparison["program"]), rows, event["fallback"], .5))
            assert comparison["baseline_loss"] == event["before_loss"]
            assert comparison["compute_cost"] == 2 * len(rows)
    assert feedback["episode_cost_is_context_only"]
    assert feedback["unique_adaptation_compute"] <= stream["final"]["screening_compute_cost"]


def test_joint_successor_requires_residual_and_keeps_two_source_lineage():
    spec = TaskSpec("joint-successor", "current plus lag", ("x",), (), (),
                    feature_bounds={"x": (-2., 2.)})
    x = Program("feature", name="x")
    lag = Program("lag", name="x", lag=1)
    a = Candidate("single-a", x, "reuse", parents=("a",), memory_id="a")
    b = Candidate("single-b", lag, "reuse", parents=("b",), memory_id="b")
    rows = [Observation({"x": float(i)}, float(i) + (float(i - 1) if i else 0.),
                         tuple({"x": float(j)} for j in range(i))) for i in range(1, 9)]
    origins = {candidate.candidate_id: [{"source_id": source, "template_id": source,
        "adaptation_work": []}] for candidate, source in ((a, "a"), (b, "b"))}
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7),
                                 library=KnowledgeLibrary())
    result = _joint_successor_candidate(agent, spec, rows, SearchBudget(3000),
                                         [(a, loss(a.program, rows, 0., .5)),
                                          (b, loss(b.program, rows, 0., .5))], origins)
    assert result is not None
    candidate, measured, evidence = result
    assert measured < min(loss(a.program, rows, 0., .5), loss(b.program, rows, 0., .5))
    assert candidate.parents == ("a", "b")
    assert {item["source_id"] for item in evidence} == {"a", "b"}
    assert {tuple(item["joint_source_ids"]) for item in evidence} == {("a", "b")}


def test_completed_json_session_does_not_credit_source_twice(revisited_stream):
    stream = revisited_stream
    saved = json.loads(json.dumps(stream["final"]))
    session = UnifiedResearchSession(state=saved, lab_factory=stream["factory"])
    purchases = len(stream["purchases"])
    assert json.loads(json.dumps(session.run())) == saved
    assert len(stream["purchases"]) == purchases
    for returned in saved["results"][0]["research"]["knowledge_return"]["returns"]:
        concept = next(item for item in session.library.concepts if item.concept_id == returned["source_id"])
        assert concept.reuse_history.count(returned) == 1


@pytest.fixture(scope="module")
def failed_successor():
    source_spec, target, source, old, _, _ = conflict_learning()

    class NumericConflictLab(ConflictLagEnvironment):
        channels = ()

        def spec_for(self, channels=()):
            assert not channels
            return self.spec

        def acquire(self, probe, channels=()):
            assert not channels
            rows = self.observe(probe)
            return MeasurementBatch(rows, len(rows), 0, ())

    library = KnowledgeLibrary()
    source_id, _ = library.learn(Candidate("paid_source", source, "representation"), source_spec)
    fallback = initial_prediction(old, .5)
    # This legal continuation uses the archived paid prefix and source program;
    # it is not an assertion that this exact scheduler state was historical.
    previous = {"status": "unresolved", "program": source.to_dict(),
        "selected_channels": [], "fallback": fallback, "actions": [],
        "selection_loss": loss(source, old, fallback, .5), "confirmation_loss": None,
        "costs": {"observation_cost": len(old), "measurement_cost": 0, "compute_cost": len(old)},
        "batches": [{"probe": asdict(probe), "channels": [],
            "observations": [asdict(row) for row in old[index * 3:(index + 1) * 3]]}
            for index, probe in enumerate(target.initial_probes)]}
    config = ResearchConfig(observation_budget=len(old) + 24, compute_budget=20000,
        minimum_improvement=1e-7, max_rounds=1, active_probes=False,
        risk_learning=False, seed=1237)
    lab = NumericConflictLab(target)
    agent = UnifiedResearchAgent(config, library=library, measurement_budget=0)
    screening = SearchBudget(4000)
    offer = offer_successor_knowledge(agent, lab, previous, {source_id}, screening)
    assert offer is not None and offer["old_evidence_gain"] > 0
    report = agent.run(lab, previous=previous, continuation_candidate=offer["candidate"])
    return {"library": library, "source_id": source_id, "old": old, "spec": target,
        "source_spec": source_spec, "previous": previous, "report": report,
        "lab": lab, "offer": offer, "screening": screening}


def test_fresh_failed_successor_records_negative_attempt_after_incumbent_retained(failed_successor):
    case = failed_successor
    report = case["report"]
    records = report["knowledge_return"]["returns"]
    negative = [row for row in records if row["source_id"] == case["source_id"] and row["paired_gain"] < 0]
    assert negative and all(not row["retained"] and row["retained_normalized_gain"] < 0 for row in negative)
    events = [row for row in report["actions"] if row["action"] == "compare_successor_knowledge_on_fresh_response"]
    assert events and events[0]["after_loss"] > events[0]["before_loss"]
    batches = {probe.probe_id: case["lab"].observed[index * 12:(index + 1) * 12]
               for index, probe in enumerate(case["lab"].queries)}
    for event in events:
        rows = batches[event["probe_id"]]
        assert event["after_loss"] == pytest.approx(loss(Program.from_dict(event["program"]), rows, event["fallback"], .5))
    confirmation, = [row for row in report["actions"] if row["action"] == "validate_continued_program"]
    assert confirmation["frozen_paid_selection"]
    assert report["program"] == case["previous"]["program"]
    assert confirmation["loss"] == pytest.approx(events[0]["before_loss"])
    assert negative[0]["condition"] == transfer_condition(case["spec"], case["old"])
    assert negative[0]["comparison_compute"] == sum(row["compute_cost"] for row in negative[0]["comparisons"])
    assert report["visit_costs"]["observation_cost"] == len(case["lab"].observed) == 12


def test_real_negative_return_withdraws_a_still_queued_same_binding(failed_successor):
    """A measured failed successor is removed before a second paid visit."""
    case = failed_successor
    session = UnifiedResearchSession(
        ResearchConfig(observation_budget=64, compute_budget=20000,
                       minimum_improvement=1e-7, active_probes=False, seed=1237),
        tasks=[{"task_index": 0, "seed": 1237}], library=case["library"],
        measurement_budget=0)
    queued = deepcopy(case["offer"])
    queued.update(task_position=0, source_task_position=0)
    session.state["pending_revisits"] = [queued]
    session._refresh_source_feedback()

    assert session.state["pending_revisits"] == []
    event = next(item for item in session.state["scheduler_events"]
                 if item["action"] == "withdraw_successor_after_negative_return")
    assert event["source_concept_ids"] == [case["source_id"]]
    assert event["matching_returns"] == 1
    # The concept remains available for a different condition/target.
    assert next(item for item in case["library"].concepts
                if item.concept_id == case["source_id"]).status != "revoked"


def test_failed_source_return_is_conditional_and_survives_json(failed_successor):
    case = failed_successor
    library = KnowledgeLibrary.from_dict(json.loads(json.dumps(case["library"].to_dict())))
    source = next(row for row in library.concepts if row.concept_id == case["source_id"])
    condition = transfer_condition(case["spec"], case["old"])
    assert return_priority(source.reuse_history, condition)[0] > 0
    fresh_condition = transfer_condition(case["spec"], case["lab"].observed)
    assert fresh_condition != condition
    assert return_priority(source.reuse_history, fresh_condition) == (0., 0.)
    assert source.status != "revoked"


def test_actual_failed_binding_is_consumed_without_banning_source_or_new_target(failed_successor):
    case = failed_successor
    library = KnowledgeLibrary.from_dict(json.loads(json.dumps(case["library"].to_dict())))
    event = next(row for row in case["report"]["actions"]
                 if row["action"] == "compare_successor_knowledge_on_fresh_response")
    failed = Program.from_dict(event["program"])
    assert library.transfer_refuted(case["spec"], failed)
    assert not library.transfer_refuted(case["spec"], Program.from_dict(case["previous"]["program"]))
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7), library=library)
    current, _ = _successor_candidates(agent, case["spec"], case["old"], SearchBudget(4000), {case["source_id"]})
    assert current and all(row.program != failed for row in current)
    future = replace(case["spec"], task_id="another_task_not_refuted")
    assert not library.transfer_refuted(future, failed)
    reopened, _ = _successor_candidates(agent, future, case["old"], SearchBudget(4000), {case["source_id"]})
    assert any(row.program == failed for row in reopened)


@pytest.fixture(scope="module")
def source_competition(failed_successor):
    case = failed_successor
    warm = KnowledgeLibrary.from_dict(json.loads(json.dumps(case["library"].to_dict())))
    paid = case["old"] + case["lab"].observed
    repaired, work = refit(warm, case["source_id"], case["spec"], paid)
    assert loss(repaired.program, paid, 0., .5) < 1e-10
    good, created = warm.learn(Candidate("paid_repair", repaired.program, "representation"),
        replace(case["source_spec"], task_id="paid_repair_source"))
    assert created
    # Store executable one-term refit procedures for these actual two laws.
    # Their ordering test attributes no new success to this packaging step.
    for identifier in (good, case["source_id"]):
        source = next(row for row in warm.concepts if row.concept_id == identifier)
        measured = loss(source.program, paid, 0., .5)
        recipe = _plan({"fit_terms": [source.program], "coefficients": [0., 1.],
                        "program": source.program, "loss": measured}, [], len(paid), 0)
        warm.meta_priors.append({"source_concept_id": identifier,
            "research_program": recipe, "source_features": list(case["spec"].feature_names)})
    cold = KnowledgeLibrary.from_dict(json.loads(json.dumps(warm.to_dict())))
    for source in cold.concepts:
        source.reuse_history = [row for row in source.reuse_history if row.get("kind") != "research_return"]
    config = ResearchConfig(max_candidates=2, minimum_improvement=1e-7)
    return case, good, UnifiedResearchAgent(config, library=warm), UnifiedResearchAgent(config, library=cold)


def test_actual_negative_return_changes_callable_source_selection(source_competition):
    case, good, warm, cold = source_competition
    future = replace(case["spec"], task_id="new_target_same_paid_condition")
    learned_calls = warm._recall(future, observations=case["old"])
    unlearned_calls = cold._recall(future, observations=case["old"])
    assert _sources(learned_calls[0]) == [good]
    assert _sources(unlearned_calls[0]) == [case["source_id"]]
    # The same bad source remains available; only the condition-matched order changes.
    assert set(source for call in learned_calls for source in _sources(call)) == {good, case["source_id"]}
    other_rows = case["lab"].observed
    assert warm._recall(future, observations=other_rows) == cold._recall(future, observations=other_rows)


def test_actual_negative_return_changes_replay_queue_without_global_exclusion(source_competition):
    case, good, warm, cold = source_competition
    future = replace(case["spec"], task_id="future_replay_same_paid_condition")
    calls = warm._recall(future, observations=case["old"])
    measured_queue = warm._replay_program_queue(future, calls, observations=case["old"])
    cold_queue = cold._replay_program_queue(future, calls, observations=case["old"])
    assert [row[0] for row in measured_queue] == [good, case["source_id"]]
    assert [row[0] for row in cold_queue] == [case["source_id"], good]
    assert warm._replay_program_queue(future, calls, observations=case["lab"].observed) == cold._replay_program_queue(
        future, calls, observations=case["lab"].observed)
