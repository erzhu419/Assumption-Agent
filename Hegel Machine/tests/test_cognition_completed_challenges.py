"""Later real knowledge challenges completed claims beyond their paid history."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.evaluation import loss, predict
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.meta_prior_research import _sources
from hegel_machine.cognition.models import Observation, ResearchConfig
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_session import UnifiedResearchSession
from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab


CONFIG = ResearchConfig(observation_budget=192, compute_budget=120000,
    minimum_improvement=1e-7, max_rounds=4, active_probes=False, seed=11811)
TASKS = [
    {"task_index": 8, "seed": 11811, "research_limits": {
        "observation_budget": 16, "compute_budget": 512, "measurement_budget": 768}},
    {"task_index": 8, "seed": 11812},
]


def restore(value):
    return json.loads(json.dumps(value))


def recording_factory(purchases):
    class RecordedLab(UnknownTaskLab):
        restoring = False

        def restore(self, acquisitions):
            self.restoring = True
            try:
                return super().restore(acquisitions)
            finally:
                self.restoring = False

        def acquire(self, probe, channels=()):
            batch = super().acquire(probe, channels)
            if not self.restoring:
                purchases.append({"seed": self.seed, "probe": probe,
                    "rows": list(batch.observations), "channels": tuple(channels),
                    "observation_cost": batch.observation_cost,
                    "measurement_cost": batch.measurement_cost})
            return batch

    return lambda item: RecordedLab(**{key: value for key, value in item.items()
                                      if key != "research_limits"})


@pytest.fixture(scope="module")
def challenged_session(tmp_path_factory):
    purchases = []
    factory = recording_factory(purchases)
    directory = tmp_path_factory.mktemp("completed_challenges")

    def save(name, value):
        path = directory / name
        path.write_text(json.dumps(value))
        return json.loads(path.read_text())

    session = UnifiedResearchSession(CONFIG, tasks=TASKS, measurement_budget=768,
                                     lab_factory=factory)
    first = save("first.json", session.run(max_tasks=1))
    successor = UnifiedResearchSession(state=first, lab_factory=factory)
    queued = save("queued.json", successor.run(max_tasks=1))
    boundary = len(purchases)
    resumed = UnifiedResearchSession(state=queued, lab_factory=factory)
    final = save("final.json", resumed.run(max_tasks=1))
    return {"first": first, "queued": queued, "final": final,
            "purchases": purchases, "boundary": boundary, "factory": factory}


def test_equivalent_short_histories_are_actually_distinguished_before_completed_claim_is_revised(challenged_session):
    stream = challenged_session
    old = stream["first"]["results"][0]["research"]
    queued, final = stream["queued"], stream["final"]
    assert old["status"] == "completed" and old["confirmation_loss"] < 1e-7
    depth = max(len(batch["probe"]["sequence"]) for batch in old["batches"])
    assert depth == 2
    offer, = queued["pending_revisits"]
    candidate, plan = offer["candidate"], offer["candidate"]["challenge"]
    old_program, new_program = Program.from_dict(old["program"]), Program.from_dict(candidate["program"])
    old_rows = [Observation(row["features"], row["target"], tuple(row["history"]))
                for batch in old["batches"] for row in batch["observations"]]
    assert loss(new_program, old_rows, candidate["fallback"], .5) < 1e-7
    assert max(abs(predict(old_program, row, old["fallback"])
                   - predict(new_program, row, candidate["fallback"])) for row in old_rows) < 1e-5
    assert len(plan["probe"]["sequence"]) > depth
    assert plan["predicted_separation"] > CONFIG.minimum_improvement
    research = final["results"][0]["research"]
    events = research["actions"][len(old["actions"]):]
    comparison = next(item for item in events if item["action"] == "compare_completed_hypotheses")
    paid = stream["purchases"][stream["boundary"]:]
    batch = next(item for item in paid if item["probe"].probe_id == comparison["probe_id"])
    assert comparison["counterexample"]
    assert comparison["before_loss"] == pytest.approx(loss(old_program, batch["rows"], old["fallback"], .5))
    assert comparison["after_loss"] == pytest.approx(loss(new_program, batch["rows"], candidate["fallback"], .5))
    assert comparison["before_loss"] > 1e-7 and comparison["after_loss"] < 1e-7
    confirmation = next(item for item in events if item["action"] == "validate_continued_program")
    confirmed = next(item for item in paid if item["probe"].probe_id == confirmation["probe_id"])
    assert confirmed["probe"].sequence != batch["probe"].sequence
    assert len(confirmed["rows"]) > depth and confirmation["accepted"]
    assert research["status"] == "completed" and research["confirmation_loss"] < 1e-7
    assert research["program"] != old["program"]
    scope = research["scope_revision"]
    assert scope["previous_program"] == old["program"]
    assert scope["previous_status"] == "completed" and scope["counterexamples"]
    assert scope["validated_trajectories"] == [
        {"probe_id": batch["probe"]["probe_id"], "length": len(batch["probe"]["sequence"])}
        for batch in old["batches"]]
    # The restriction must change future executable calls, not just decorate
    # this report. Short measured successes remain callable after JSON reload.
    library = KnowledgeLibrary.from_dict(restore(final["knowledge"]))
    original_id, = old["concepts_added"]
    concept = next(item for item in library.concepts if item.concept_id == original_id)
    spec = UnknownTaskLab(8, 11811).spec
    local = library._scope_for(concept, spec)
    assert concept.status == "restricted" and local.predicate
    condition = Program.from_dict(local.predicate)
    assert all(condition.predict(row) for row in old_rows)
    counterexamples = [row for row in batch["rows"]
                      if (predict(old_program, row, old["fallback"]) - row.target) ** 2 > 1e-7]
    assert counterexamples and all(not condition.predict(row) for row in counterexamples)
    recalled = UnifiedResearchAgent(CONFIG, library=library)
    recalled._revise_inputs = False
    old_calls = [program for program in recalled._recall(spec) if original_id in _sources(program)]
    assert old_calls
    assert all(loss(program, old_rows, local.fallback, .5) < 1e-7 for program in old_calls)
    assert all(predict(program, counterexamples[0], local.fallback) == pytest.approx(local.fallback)
               for program in old_calls)


def test_challenge_json_resume_preserves_paid_evidence_and_the_actual_per_task_resource_pool(challenged_session):
    stream = challenged_session
    old, final = stream["first"]["results"][0]["research"], stream["final"]
    research = final["results"][0]["research"]
    assert research["batches"][:len(old["batches"])] == old["batches"]
    old_ids = {batch["probe"]["probe_id"] for batch in old["batches"]}
    paid = stream["purchases"][stream["boundary"]:]
    assert paid and not old_ids & {batch["probe"].probe_id for batch in paid}
    for name in ("observation_cost", "measurement_cost"):
        assert research["visit_costs"][name] == sum(batch[name] for batch in paid)
        assert research["costs"][name] == old["costs"][name] + research["visit_costs"][name]
    assert final["costs"]["observation_cost"] <= 208
    assert final["costs"]["measurement_cost"] <= 1536
    assert final["costs"]["compute_cost"] <= 120512
    assert final["costs"]["compute_cost"] == final["screening_compute_cost"] + sum(
        item["research"]["costs"]["compute_cost"] for item in final["results"])
    session = UnifiedResearchSession(state=restore(final), lab_factory=stream["factory"])
    count = len(stream["purchases"])
    assert restore(session.run()) == final and len(stream["purchases"]) == count
    scoring = session.evaluate(length=24, trajectories=2)
    assert scoring["evaluation"][0]["mse"] <= CONFIG.minimum_improvement
    assert scoring["costs"] == final["costs"]


def test_unfunded_challenge_keeps_the_completed_program_and_its_existing_support(challenged_session):
    state = challenged_session["queued"]
    previous = state["results"][0]["research"]
    offer, = state["pending_revisits"]
    library = KnowledgeLibrary.from_dict(state["knowledge"])
    purchases = []
    config = replace(CONFIG, observation_budget=previous["costs"]["observation_cost"],
                     compute_budget=previous["costs"]["compute_cost"])
    with recording_factory(purchases)(TASKS[0]) as lab:
        report = UnifiedResearchAgent(config, library=library,
            measurement_budget=previous["costs"]["measurement_cost"]).run(lab,
                previous=previous, continuation_candidate=deepcopy(offer["candidate"]))
    assert report["status"] == "completed" and report["program"] == previous["program"]
    assert report["confirmation_loss"] == previous["confirmation_loss"]
    assert report["batches"] == previous["batches"] and purchases == []
    event = next(item for item in report["actions"][len(previous["actions"]):]
                 if item["action"] == "completed_task_challenge")
    assert event["outcome"] == "deferred" and not event["counterexample"]
    assert report["scope_revision"]["status"] == "challenge_deferred"
    assert report["scope_revision"]["counterexamples"] == []


def test_real_response_can_reject_a_later_wrong_challenger_without_revoking_the_old_program(challenged_session):
    state = challenged_session["queued"]
    correct = state["results"][1]["research"]
    short = state["results"][0]["research"]
    offer, = state["pending_revisits"]
    challenger = {"program": short["program"], "channels": short["selected_channels"],
        "fallback": short["fallback"], "source_concept_ids": short["concepts_added"],
        "kind": "successor_knowledge", "challenge": deepcopy(offer["candidate"]["challenge"])}
    # Reverse two actually learned explanations: the continuous-state source
    # is already correct here, while the short-history rival is only locally
    # supported. New physical responses must retain the former.
    challenger["challenge"]["probe"]["probe_id"] = "later_short_history_challenge"
    sequence = challenger["challenge"]["probe"]["sequence"]
    challenger["challenge"]["hypotheses"] = [
        {"hypothesis_id": name, "candidate_id": name, "program": payload["program"],
         "predictions": [predict(Program.from_dict(payload["program"]),
             Observation(point, 0., tuple(sequence[:index])), payload["fallback"])
             for index, point in enumerate(sequence)]}
        for name, payload in (("completed_incumbent", correct), ("later_short_history", short))]
    purchases = []
    library = KnowledgeLibrary.from_dict(state["knowledge"])
    before = restore(library.to_dict())
    with recording_factory(purchases)(TASKS[1]) as lab:
        report = UnifiedResearchAgent(CONFIG, measurement_budget=768, library=library).run(
            lab, previous=correct, continuation_candidate=challenger)
    assert purchases
    comparison = next(item for item in report["actions"][len(correct["actions"]):]
                      if item["action"] == "compare_completed_hypotheses")
    assert comparison["before_loss"] < 1e-7 and comparison["after_loss"] > 1e-7
    assert not comparison["counterexample"]
    assert report["program"] == correct["program"] and report["status"] == "completed"
    assert report["scope_revision"]["status"] == "challenge_rejected"
    assert report["scope_revision"]["counterexamples"] == []
    assert not any(item["action"] == "validate_continued_program"
                   for item in report["actions"][len(correct["actions"]):])
    assert restore(library.to_dict()) == before
