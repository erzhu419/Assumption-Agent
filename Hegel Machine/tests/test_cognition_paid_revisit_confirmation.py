"""A paid source selection needs one independent confirmation, not a new fit."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.unified_continuation import paid_view
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_revisits import offer_successor_knowledge
from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab


CONFIG = ResearchConfig(observation_budget=192, compute_budget=120000,
    minimum_improvement=1e-7, max_rounds=4, max_candidates=16,
    active_probes=False, seed=11811)


def persisted(value):
    return json.loads(json.dumps(value))


class RecordedLab(UnknownTaskLab):
    def restore(self, records):
        self.restoring = True
        try:
            return super().restore(records)
        finally:
            self.restoring = False

    def acquire(self, probe, channels=()):
        batch = super().acquire(probe, channels)
        if not getattr(self, "restoring", False):
            self.new_purchases.append((probe, batch))
        return batch


@pytest.fixture(scope="module")
def learned_selection():
    library = KnowledgeLibrary()
    with UnknownTaskLab(8, 11811) as lab:
        previous = UnifiedResearchAgent(replace(CONFIG, compute_budget=512),
            measurement_budget=768).run(lab)
    assert previous["status"] == "unresolved"
    # Learn the source by ordinary research on another realized trajectory;
    # no private apparatus equation supplies the transferred program.
    with UnknownTaskLab(8, 11812) as lab:
        source = UnifiedResearchAgent(CONFIG, measurement_budget=768,
            library=library).run(lab)
    assert source["status"] == "completed" and source["concepts_added"]
    with UnknownTaskLab(8, 11811) as lab:
        screening = SearchBudget(45000)
        offer = offer_successor_knowledge(UnifiedResearchAgent(CONFIG, library=library),
            lab, previous, set(source["concepts_added"]), screening)
    assert offer is not None and offer["candidate"].get("paid_selection")
    assert screening.spent > 0
    return persisted({"previous": previous, "offer": offer,
        "knowledge": library.to_dict(), "screening_compute_cost": screening.spent})


def resume(data, *, profile="development", extra_rows=24):
    previous, candidate = data["previous"], data["offer"]["candidate"]
    quote = data["offer"]["minimum_resources"]
    agent = UnifiedResearchAgent(replace(CONFIG,
        observation_budget=previous["costs"]["observation_cost"] + extra_rows,
        compute_budget=previous["costs"]["compute_cost"] + quote["compute_cost"]),
        measurement_budget=previous["costs"]["measurement_cost"] + max(
            quote["measurement_cost"], extra_rows * 4),
        library=KnowledgeLibrary.from_dict(data["knowledge"]))
    with RecordedLab(8, 11811, profile=profile) as lab:
        lab.new_purchases = []
        current = agent.run(lab, previous=persisted(previous),
            continuation_candidate=persisted(candidate))
    return agent, lab, current


def test_paid_selection_confirms_frozen_source_in_24_rows_without_refit(learned_selection, monkeypatch):
    data = learned_selection
    candidate = data["offer"]["candidate"]

    def unexpected_fit(*args, **kwargs):
        pytest.fail("A frozen paid selection must not be fitted or reselected before confirmation")

    monkeypatch.setattr("hegel_machine.cognition.unified_research._fit", unexpected_fit)
    monkeypatch.setattr(UnifiedResearchAgent, "_best", unexpected_fit)
    _, lab, current = resume(data)
    previous = data["previous"]
    events = current["actions"][len(previous["actions"]):]
    assert current["status"] == "completed"
    assert current["program"] == candidate["program"]
    assert current["fallback"] == candidate["fallback"]
    assert current["visit_costs"]["observation_cost"] == 24
    assert current["visit_costs"]["compute_cost"] == 48
    assert len(lab.new_purchases) == 1
    probe, batch = lab.new_purchases[0]
    assert batch.observation_cost == 24
    assert current["visit_costs"]["measurement_cost"] == batch.measurement_cost
    assert current["batches"][:len(previous["batches"])] == previous["batches"]
    assert probe.probe_id not in candidate["paid_selection"]["evidence_batches"]
    assert all(list(probe.sequence) != old["probe"]["sequence"] for old in previous["batches"])
    assert [event["purpose"] for event in events if event["action"] == "acquire"] == [
        "confirm_continuation_program"]
    assert not any(event["action"] == "compare_continuation_on_paid_evidence" for event in events)
    confirmation, = [event for event in events if event["action"] == "validate_continued_program"]
    assert confirmation["accepted"] and confirmation["selection_probe_id"] is None
    assert confirmation["selection_evidence_batches"] == candidate["paid_selection"]["evidence_batches"]
    paired, = [event for event in events
               if event["action"] == "compare_successor_knowledge_on_fresh_response"]
    assert paired["probe_id"] == confirmation["probe_id"] == probe.probe_id
    assert paired["source_concept_ids"] == candidate["source_concept_ids"]
    assert paired["after_loss"] == pytest.approx(loss(
        Program.from_dict(candidate["program"]), batch.observations, candidate["fallback"], .5))
    assert paired["before_loss"] == pytest.approx(loss(
        Program.from_dict(previous["program"]), batch.observations, previous["fallback"], .5))
    returns = current["knowledge_return"]["returns"]
    assert returns and all(item["retained"] and item["value_status"] == "positive" for item in returns)
    for item in returns:
        # The existing return policy asks for replicated transfer under
        # residual-bearing conditions, even though this task is confirmed.
        repeated = item["condition"]["response_signature"].get("residual_bin", 0) > 0
        assert item["confirmation_status"] == ("provisional" if repeated else "confirmed")
        assert item["final_status"] == "completed"
        assert len(item["comparisons"]) == 1
        assert item["comparisons"][0]["comparison_id"] == probe.probe_id
    assert {item["source_id"] for item in returns} == set(candidate["source_concept_ids"])


def test_failed_frozen_source_stops_after_one_confirmation_with_more_budget(learned_selection, monkeypatch):
    # The laboratory's supported confirmation profile changes the response
    # after the old paid frontier. The agent sees only the fresh measurements.
    def unexpected_research(*args, **kwargs):
        pytest.fail("Failed one-confirmation packages must stop before additional research")

    monkeypatch.setattr(UnifiedResearchAgent, "_support", unexpected_research)
    _, lab, current = resume(learned_selection, profile="confirmation", extra_rows=96)
    previous = learned_selection["previous"]
    events = current["actions"][len(previous["actions"]):]
    assert current["status"] == "unresolved"
    assert current["concepts_added"] == []
    assert current["visit_costs"]["observation_cost"] == 24
    assert len(lab.new_purchases) == 1
    confirmation, = [event for event in events if event["action"] == "validate_continued_program"]
    assert not confirmation["accepted"]
    paired, = [event for event in events
               if event["action"] == "compare_successor_knowledge_on_fresh_response"]
    assert paired["program"] == learned_selection["offer"]["candidate"]["program"]
    assert paired["after_loss"] > CONFIG.minimum_improvement
    assert current["confirmation_loss"] > CONFIG.minimum_improvement
    returns = current["knowledge_return"]["returns"]
    assert returns and all(item["confirmation_status"] != "confirmed" for item in returns)


def test_receipt_matches_real_old_common_view(learned_selection):
    data = deepcopy(learned_selection)
    previous, candidate = data["previous"], data["offer"]["candidate"]
    receipt = candidate["paid_selection"]
    with UnknownTaskLab(8, 11811) as lab:
        rows = [row for batch in paid_view(lab, previous, receipt["channels"]) for row in batch]
    assert len(rows) == receipt["evidence_rows"]
    assert receipt["baseline_loss"] == pytest.approx(loss(
        Program.from_dict(receipt["baseline_program"]), rows, receipt["baseline_fallback"], .5))
    assert receipt["candidate_loss"] == pytest.approx(loss(
        Program.from_dict(candidate["program"]), rows, candidate["fallback"], .5))
    assert receipt["candidate_loss"] <= CONFIG.minimum_improvement
