"""Later learned concepts reopen an unresolved task using its paid evidence."""

from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.meta_prior_claims import compile_claims
from hegel_machine.cognition.models import Candidate, Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_session import UnifiedResearchSession
from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab
from hegel_machine.cognition.unified_revisits import _merge_joint_transfer_candidates


TASKS = [{"task_index": 8, "seed": 11811}, {"task_index": 8, "seed": 11812}]
CONFIG = ResearchConfig(observation_budget=192, compute_budget=120000,
    minimum_improvement=1e-7, max_rounds=4, max_candidates=16,
    active_probes=False, seed=11811)
RESOURCES = ("observation_cost", "measurement_cost", "compute_cost")


def test_nonadjacent_duplicate_preserves_other_candidate_and_all_origins():
    first = Candidate("a", Program("feature", name="x"), "reuse", parents=("source_a",))
    other = Candidate("b", Program("feature", name="y"), "reuse", parents=("source_b",))
    duplicate = replace(first, candidate_id="c", parents=("source_c",))
    later_duplicate = replace(first, candidate_id="d", parents=("source_d",))
    candidates = [first, other, duplicate, later_duplicate]
    origins = {item.candidate_id: [{"source_id": item.parents[0]}] for item in candidates}
    merged, evidence = _merge_joint_transfer_candidates(candidates, origins, "target", 1)
    assert [item.candidate_id for item in merged] == ["a", "b"]
    assert merged[1] == other
    assert merged[0].parents == ("source_a", "source_c", "source_d")
    assert [item["source_id"] for item in evidence["a"]] == ["source_a", "source_c", "source_d"]
    assert all(item["joint_source_ids"] == ["source_a", "source_c", "source_d"]
               for item in evidence["a"])
    assert evidence["b"] == [{"source_id": "source_b"}]


def persisted(value):
    return json.loads(json.dumps(value))


@pytest.fixture(scope="module")
def revisited_stream(tmp_path_factory):
    opened, purchases, evaluations, checkpoints = [], [], [], []
    checkpoint_dir = tmp_path_factory.mktemp("unified_revisits")

    def save(name, state):
        path = checkpoint_dir / name
        path.write_text(json.dumps(state))
        return json.loads(path.read_text())

    class RecordedLab(UnknownTaskLab):
        def restore(self, records):
            self.restoring_paid_process = True
            try:
                return super().restore(records)
            finally:
                self.restoring_paid_process = False

        def acquire(self, probe, channels=()):
            batch = super().acquire(probe, channels)
            if not getattr(self, "restoring_paid_process", False):
                purchases.append({"seed": self.seed, "probe_id": probe.probe_id,
                    "channels": tuple(channels), "rows": list(batch.observations),
                    "observation_cost": batch.observation_cost,
                    "measurement_cost": batch.measurement_cost})
            return batch

        def evaluation_rows(self, *args, **kwargs):
            evaluations.append(self.seed)
            return super().evaluation_rows(*args, **kwargs)

    def factory(item):
        opened.append(item["seed"])
        return RecordedLab(**item)

    first_session = UnifiedResearchSession(CONFIG, tasks=TASKS,
        measurement_budget=768, lab_factory=factory)
    # A performs a real initial visit with limited compute. Its public probe
    # length is unchanged; B then learns with the normal per-task allowance.
    # The accumulated report is a legal partial session, with unused compute
    # available to its later knowledge-driven revisit.
    with factory(TASKS[0]) as laboratory:
        previous = UnifiedResearchAgent(replace(CONFIG, compute_budget=512),
            measurement_budget=768, library=first_session.library).run(laboratory)
    first_session.state["results"].append({"task_position": 0,
        "available_concepts": [], "research": previous})
    first_session.state["status"] = "paused"
    paused = save("paused.json", first_session.to_dict())
    resumed = UnifiedResearchSession(state=paused, lab_factory=factory)
    queued = save("queued.json", resumed.run(max_tasks=1, on_checkpoint=checkpoints.append))
    before_purchases, before_opened = len(purchases), len(opened)
    continuation = UnifiedResearchSession(state=queued, lab_factory=factory)
    final = save("final.json", continuation.run(max_tasks=1, on_checkpoint=checkpoints.append))
    return {"paused": paused, "queued": queued, "final": final,
        "opened": opened, "purchases": purchases, "evaluations": evaluations,
        "before_purchases": before_purchases, "before_opened": before_opened,
        "factory": factory, "checkpoints": checkpoints}


def test_later_real_knowledge_schedules_and_restores_the_unresolved_task(revisited_stream):
    stream = revisited_stream
    paused, queued, final = (stream[key] for key in ("paused", "queued", "final"))
    assert paused["results"][0]["research"]["status"] == "unresolved"
    assert paused["pending_tasks"] == [1] and not paused["pending_revisits"]
    learned = set(queued["results"][1]["research"]["concepts_added"])
    assert learned and queued["results"][1]["research"]["status"] == "completed"
    assert queued["pending_tasks"] == [] and queued["pending_revisits"]
    with pytest.raises(ValueError, match="finish learning"):
        UnifiedResearchSession(state=queued, lab_factory=stream["factory"]).evaluate()
    assert stream["evaluations"] == []
    visit, = final["revisits"]
    assert visit["task_position"] == 0 and visit["source_task_position"] == 1
    assert set(visit["candidate"]["source_concept_ids"]) & learned
    assert visit["previous"] == paused["results"][0]["research"]
    # Resume the target, then allow its newly learned concept to challenge the
    # other completed task on paid evidence. No extra source observations are
    # needed when the explanations make no distinguishable new prediction.
    assert stream["opened"][stream["before_opened"]:] == [task["seed"] for task in TASKS]
    assert final["results"][1] == queued["results"][1]
    assert final["results"][0]["research"]["status"] == "completed"
    assert final["pending_revisits"] == [] and final["status"] == "completed"
    # No new task or knowledge arrived. JSON restoration must not recycle the
    # same evidence-based opportunity into another paid visit.
    restored = UnifiedResearchSession(state=final, lab_factory=stream["factory"])
    opened, purchases = len(stream["opened"]), len(stream["purchases"])
    assert persisted(restored.run()) == final
    assert len(stream["opened"]) == opened and len(stream["purchases"]) == purchases


def test_revisit_reuses_old_evidence_and_compares_frozen_candidates_on_new_responses(revisited_stream):
    stream = revisited_stream
    old = stream["paused"]["results"][0]["research"]
    current = stream["final"]["results"][0]["research"]
    assert current["batches"][:len(old["batches"])] == old["batches"]
    old_ids = {batch["probe"]["probe_id"] for batch in old["batches"]}
    purchases = stream["purchases"][stream["before_purchases"]:]
    assert purchases and all(item["seed"] == TASKS[0]["seed"] for item in purchases)
    assert not old_ids & {item["probe_id"] for item in purchases}
    selected, = [item for item in current["actions"][len(old["actions"]):]
                 if item["action"] == "reuse_paid_continuation_selection"]
    assert set(selected["evidence_batches"]) == old_ids
    assert selected["new_observations"] == 0
    assert selected["candidate_loss"] <= CONFIG.minimum_improvement
    comparisons = [item for item in current["actions"][len(old["actions"]):]
                   if item["action"] == "compare_successor_knowledge_on_fresh_response"]
    assert comparisons
    by_probe = {item["probe_id"]: item for item in purchases}
    for event in comparisons:
        batch = by_probe[event["probe_id"]]
        rows = batch["rows"]
        assert event["program"] == selected["program"]
        before = loss(Program.from_dict(old["program"]), rows, old["fallback"], .5)
        after = loss(Program.from_dict(event["program"]), rows,
                     event["fallback"], .5)
        assert event["before_loss"] == pytest.approx(before)
        assert event["after_loss"] == pytest.approx(after)
    by_id = {batch["probe"]["probe_id"]: batch for batch in current["batches"]}
    confirmations = [item for item in current["actions"][len(old["actions"]):]
                     if item["action"] == "validate_continued_program"]
    assert confirmations
    for event in confirmations:
        confirmed = by_id[event["probe_id"]]["probe"]
        assert event["selection_probe_id"] is None and event["frozen_paid_selection"]
        assert set(event["selection_evidence_batches"]) == old_ids
        assert confirmed["probe_id"] not in old_ids
        assert all(confirmed["sequence"] != by_id[probe_id]["probe"]["sequence"]
                   for probe_id in old_ids)
    assert len(purchases) == 1
    assert current["visit_costs"]["observation_cost"] == sum(item["observation_cost"] for item in purchases)
    assert current["visit_costs"]["measurement_cost"] == sum(item["measurement_cost"] for item in purchases)


def test_session_accounts_screening_and_incremental_research_before_independent_scoring(revisited_stream):
    stream, final = revisited_stream, revisited_stream["final"]
    old = stream["paused"]["results"][0]["research"]
    current = final["results"][0]["research"]
    visit, = final["revisits"]
    for resource in RESOURCES:
        assert current["costs"][resource] == old["costs"][resource] + current["visit_costs"][resource]
        assert visit["visit_costs"][resource] == current["visit_costs"][resource]
        expected = sum(row["research"]["costs"][resource] for row in final["results"])
        if resource == "compute_cost":
            expected += final["screening_compute_cost"]
        assert final["costs"][resource] == expected
    assert final["screening_compute_cost"] > 0
    assert final["costs"]["observation_cost"] <= len(TASKS) * CONFIG.observation_budget
    assert final["costs"]["measurement_cost"] <= len(TASKS) * 768
    assert final["costs"]["compute_cost"] <= len(TASKS) * CONFIG.compute_budget
    scoring = UnifiedResearchSession(state=final, lab_factory=stream["factory"])
    before = deepcopy(scoring.to_dict())
    scored = scoring.evaluate(length=24, trajectories=2)
    assert scored["costs"] == before["costs"] and scored["knowledge"] == before["knowledge"]
    assert len(scored["evaluation"]) == len(TASKS)
    assert scored["evaluation"][0]["mse"] < CONFIG.minimum_improvement
    assert all(item["rows"] == item["observation_cost"] == item["compute_cost"] == 48
               for item in scored["evaluation"])
    assert stream["evaluations"] == [item["seed"] for item in TASKS]
    opened, evaluated = len(stream["opened"]), len(stream["evaluations"])
    assert scoring.evaluate(length=24, trajectories=2) == scored
    assert len(stream["opened"]) == opened and len(stream["evaluations"]) == evaluated


def test_new_learned_knowledge_without_paid_evidence_gain_does_not_buy_a_revisit():
    # A short, constant public control sequence is exactly explained by its
    # fitted intercept but has no independent confirmation yet.
    old_lab = UnknownTaskLab(0, 8711)
    old_probe = Probe("short_unconfirmed", tuple({"u": 0.} for _ in range(8)))
    old_rows = old_lab.acquire(old_probe).observations
    incumbent = Program("const", value=old_rows[0].target)
    assert loss(incumbent, old_rows, 0., .5) == 0.
    previous = {"status": "unresolved", "program": incumbent.to_dict(),
        "selected_channels": [], "fallback": old_rows[0].target,
        "costs": {"observation_cost": 8, "measurement_cost": 0, "compute_cost": 8},
        "batches": [{"probe": asdict(old_probe), "channels": [],
                     "observations": [asdict(row) for row in old_rows]}]}
    # The successor learns from actual varied controls, then checks a separate
    # actual trajectory. No private apparatus formula supplies its program.
    source = UnknownTaskLab(0, 8911)
    training = source.acquire(Probe("varied_source", tuple(
        {"u": (index - 8) / 8} for index in range(16)))).observations
    validation = source.acquire(Probe("independent_source", tuple(
        {"u": (index - 7.5) / 9} for index in range(16)))).observations
    work = SearchBudget(10000)
    proposal = next(item["candidate"] for item in compile_claims(source.spec, training, work)
                    if item["kind"] == "affine")
    assert loss(proposal.program, validation, 0., .5) <= 1e-7
    library = KnowledgeLibrary()
    concept_id, created = library.learn(proposal, source.spec, training)
    assert created
    acquisitions = []

    class NoExtraReadings(UnknownTaskLab):
        def acquire(self, probe, channels=()):
            acquisitions.append(probe.probe_id)
            return super().acquire(probe, channels)

    session = UnifiedResearchSession(CONFIG, library=library, measurement_budget=1536,
        tasks=[{"task_index": 0, "seed": 8711}, {"task_index": 0, "seed": 8911}],
        lab_factory=lambda item: NoExtraReadings(**item))
    session.state["results"] = [{"task_position": 0, "research": previous},
        {"task_position": 1, "research": {"status": "completed",
         "costs": {"observation_cost": 32, "measurement_cost": 0,
                   "compute_cost": work.spent + len(validation)}}}]
    session._offer_new_knowledge(1, [concept_id])
    result = session.to_dict()
    event = result["scheduler_events"][-1]
    assert event["source_concept_ids"] == [concept_id]
    assert event["compute_cost"] > 0 and event["old_evidence_gain"] <= 0.
    assert not event["scheduled"] and not result["pending_revisits"]
    assert acquisitions == []
    assert result["results"][0]["research"] == previous
