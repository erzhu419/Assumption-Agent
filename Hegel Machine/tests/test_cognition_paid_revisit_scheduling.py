"""Previously unexposed knowledge can fund one fresh check of a paid task."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_session import UnifiedResearchSession
from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab


CONFIG = ResearchConfig(observation_budget=192, compute_budget=120000,
    minimum_improvement=1e-7, max_rounds=4, max_candidates=16,
    active_probes=False, seed=11811)
TARGET = {"task_index": 8, "seed": 11811}


@pytest.fixture(scope="module")
def acquired_checkpoint():
    previous = UnifiedResearchAgent(replace(CONFIG, compute_budget=512)).run(UnknownTaskLab(**TARGET))
    assert previous["status"] == "unresolved"
    library = KnowledgeLibrary()
    source = UnifiedResearchAgent(CONFIG, library=library).run(UnknownTaskLab(8, seed=11812))
    assert source["status"] == "completed" and source["concepts_added"]
    return previous, library.to_dict(), source["concepts_added"]


def session_from_paid(acquired_checkpoint, remaining_rows, *, compute_budget=120000, factory=None):
    previous, library, _ = acquired_checkpoint
    task = {**TARGET, "research_limits": {
        "observation_budget": previous["costs"]["observation_cost"] + remaining_rows}}
    session = UnifiedResearchSession(replace(CONFIG, compute_budget=compute_budget), tasks=[task],
        library=KnowledgeLibrary.from_dict(library), lab_factory=factory)
    session.state["results"] = [{"task_position": 0, "available_concepts": [],
                                 "research": deepcopy(previous)}]
    session.state["status"] = "paused"
    return session


def tracked_factory(purchases):
    class RecordedLab(UnknownTaskLab):
        def restore(self, records):
            self.restoring = True
            try:
                return super().restore(records)
            finally:
                self.restoring = False

        def acquire(self, probe, channels=()):
            result = super().acquire(probe, channels)
            if not getattr(self, "restoring", False):
                purchases.append((self.seed, probe.probe_id, probe.cost))
            return result

        def evaluation_rows(self, *args, **kwargs):
            raise AssertionError("successor scheduling read a terminal outcome")

    return lambda item: RecordedLab(**item)


def test_restored_paid_task_discovers_existing_source_and_spends_one_fresh_trajectory(acquired_checkpoint):
    purchases = []
    factory = tracked_factory(purchases)
    original = json.loads(json.dumps(
        session_from_paid(acquired_checkpoint, 32, factory=factory).to_dict()))
    original_ids = {item["concept_id"] for item in original["knowledge"]["concepts"]}
    assert original["results"][0]["available_concepts"] == []
    session = UnifiedResearchSession(state=deepcopy(original), lab_factory=factory)
    final = session.run()
    previous = original["results"][0]["research"]
    current = final["results"][0]["research"]
    assert current["status"] == "completed"
    visit, = final["revisits"]
    assert visit["source_task_position"] is None  # The source predates this session.
    assert set(visit["candidate"]["source_concept_ids"]) <= original_ids
    assert sum(rows for _, _, rows in purchases) == 24
    assert current["batches"][:len(previous["batches"])] == previous["batches"]
    assert current["costs"]["observation_cost"] - previous["costs"]["observation_cost"] == 24
    assert final["screening_compute_cost"] > 0
    restored = UnifiedResearchSession(state=json.loads(json.dumps(final)), lab_factory=factory)
    before = len(purchases)
    again = restored.run()
    assert len(purchases) == before
    assert again["screening_compute_cost"] == final["screening_compute_cost"]
    assert again["revisits"] == final["revisits"]


def test_twenty_three_rows_cannot_purchase_the_required_fresh_confirmation(acquired_checkpoint):
    purchases = []
    session = session_from_paid(acquired_checkpoint, 23, factory=tracked_factory(purchases))
    before = session.to_dict()
    result = session.run()
    assert result["results"][0]["research"] == before["results"][0]["research"]
    assert result["revisits"] == [] and purchases == []
    assert result["screening_compute_cost"] == before["screening_compute_cost"]


def test_same_paid_bank_is_not_rescreened_until_compute_allowance_increases(acquired_checkpoint):
    purchases = []
    factory = tracked_factory(purchases)
    previous, _, _ = acquired_checkpoint
    # Enough to inspect the existing bank, insufficient to complete candidate
    # adaptation and comparison. No observation is generated to force a retry.
    limited = previous["costs"]["compute_cost"] + 6 * previous["costs"]["observation_cost"] + 32 * 24 + 300
    session = session_from_paid(acquired_checkpoint, 32, compute_budget=limited, factory=factory)
    first = session.run()
    assert first["results"][0]["research"]["status"] == "unresolved"
    assert first["screening_compute_cost"] > 0 and purchases == []
    assert first["results"][0]["successor_screen"]["source_ids"]
    resumed = UnifiedResearchSession(state=json.loads(json.dumps(first)), lab_factory=factory)
    repeated = resumed.run()
    assert repeated["screening_compute_cost"] == first["screening_compute_cost"]
    assert purchases == []
    increased = deepcopy(repeated)
    increased["config"]["compute_budget"] = CONFIG.compute_budget
    funded = UnifiedResearchSession(state=increased, lab_factory=factory).run()
    assert funded["results"][0]["research"]["status"] == "completed"
    assert funded["screening_compute_cost"] > repeated["screening_compute_cost"]
    assert sum(rows for _, _, rows in purchases) == 24


def test_new_unrelated_production_does_not_own_an_existing_external_source(acquired_checkpoint):
    purchases = []
    session = session_from_paid(acquired_checkpoint, 32, factory=tracked_factory(purchases))
    incoming = {item.concept_id for item in session.library.concepts}
    new_task = {"task_index": 0, "seed": 19812}
    report = UnifiedResearchAgent(CONFIG, library=session.library).run(UnknownTaskLab(**new_task))
    added = set(report["concepts_added"]) - incoming
    assert report["status"] == "completed" and added
    # The new task's actual work has its own fully spent observation allowance,
    # leaving the same 32 rows available to the earlier unresolved task.
    session.state["tasks"].append({**new_task, "research_limits": {
        "observation_budget": report["costs"]["observation_cost"]}})
    session.state["results"].append({"task_position": 1, "available_concepts": sorted(incoming),
                                     "research": report})
    session._offer_new_knowledge(1, sorted(added))
    offer, = session.state["pending_revisits"]
    assert set(offer["candidate"]["source_concept_ids"]) & incoming
    assert offer["source_task_position"] is None
    event = next(item for item in reversed(session.state["scheduler_events"])
                 if item["action"] == "try_successor_knowledge" and item["scheduled"])
    assert event["source_task_position"] is None
    assert set(event["previously_unavailable_concept_ids"]) & incoming
    assert purchases == []  # Scheduling used the already paid trajectories only.
