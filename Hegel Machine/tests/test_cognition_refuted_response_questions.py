"""Paid counterexamples constrain expanded refits and later response questions."""

from copy import deepcopy
from dataclasses import replace
import gzip
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget
from hegel_machine.cognition.noise_replication import assess_noise_replication
from hegel_machine.cognition.object_continuation import decode_entry
from hegel_machine.cognition.response_research_questions import response_question_offer
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_session import UnifiedResearchSession
from test_cognition_noise_research_reopening import laboratory


@pytest.fixture
def case():
    path = Path(__file__).parent / "fixtures/target_withdrawal_root07_question.json.gz"
    payload = json.loads(gzip.decompress(path.read_bytes()))
    payload["task"].pop("family")
    payload["task"].pop("research_limits")
    return payload


def remember_failure(library, spec, payload, failed=0, *, sources=None):
    item = payload["failed_entries"][failed]
    entry = decode_entry(item["entry"])
    return library.record_transfer_counterexample(spec, entry["candidate"],
        sources or entry["method_source_ids"], evidence=item["outcome"],
        probe_id=payload["counterexample_probe_id"])


def quote(previous, question, library=None, spec=None):
    return response_question_offer(previous, question, 64,
        {"observation_cost": 48, "measurement_cost": 0, "compute_cost": 500},
        resource_limits={"observation_budget": 240, "compute_budget": 240000},
        is_refuted=(lambda program: library.transfer_refuted(spec, program)) if library is not None else None)


def test_measured_expanded_refit_is_withdrawn_without_blame_on_unconsumed_sources(case):
    library = KnowledgeLibrary.from_dict(case["library"])
    lab = laboratory(case["task"])
    parent = decode_entry(case["previous"]["response_questions"][0]["parent"])
    assert parent["candidate"].program.op != "call"
    first, repeat = [[Observation(row["features"], row["target"], tuple(row["history"]))
        for row in rows] for rows in case["repeated_counterexample_rows"]]
    measured = assess_noise_replication(parent["candidate"].program, first, repeat, SearchBudget(1000))
    assert measured["repeatable_error"] and not measured["noise_supported"]
    unrelated, _ = library.learn(Candidate("unrelated", Program("const", value=123.), "refit"), lab.spec)
    before = deepcopy(next(c for c in library.concepts if c.concept_id == unrelated).reuse_history)
    event = remember_failure(library, lab.spec, case, sources=["concept_0001", unrelated])
    assert event["source_ids"] == ["concept_0001"]
    library = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    assert library.transfer_refuted(lab.spec, parent["candidate"].program)
    assert next(c for c in library.concepts if c.concept_id == unrelated).reuse_history == before
    source = next(c for c in library.concepts if c.concept_id == "concept_0001")
    assert source.status == "available" and not source.excluded_tasks
    assert not library.transfer_refuted(replace(lab.spec, task_id=source.learned_on), source.program)
    candidate = decode_entry(case["previous"]["response_questions"][0]["candidate"])
    assert not library.transfer_refuted(lab.spec, candidate["candidate"].program)
    assert library.record_transfer_counterexample(lab.spec, candidate["candidate"],
        candidate["method_source_ids"], evidence={"status": "unresolved"}, probe_id="unresolved") is None


def test_refuted_parent_cannot_win_back_confirmation_on_a_lucky_new_batch(case):
    previous = case["previous"]
    library = KnowledgeLibrary.from_dict(case["library"])
    lab = laboratory({**case["task"], "noise_seed": 19})
    remember_failure(library, lab.spec, case)
    question = previous["response_questions"][0]
    offer = quote(previous, question, library, lab.spec)
    assert offer is not None  # The different, unrefuted candidate still merits confirmation.
    config = ResearchConfig(**{**case["config"], "seed": case["task"]["seed"],
        "observation_budget": previous["costs"]["observation_cost"] + 48,
        "compute_budget": previous["costs"]["compute_cost"] + 500})
    report = UnifiedResearchAgent(config, measurement_budget=0, library=library).run(
        lab, previous=previous, continuation_candidate=offer["candidate"])
    new_batches = report["batches"][len(previous["batches"]):]
    assert len(new_batches) == 2
    assert new_batches[0]["probe"]["sequence"] == new_batches[1]["probe"]["sequence"]
    assert report["batches"][:len(previous["batches"])] == previous["batches"]
    parent = decode_entry(question["parent"])["candidate"].program
    candidate = decode_entry(question["candidate"])["candidate"].program
    rows = [Observation(row["features"], row["target"], tuple(row["history"]))
            for batch in new_batches for row in batch["observations"]]
    assert loss(parent, rows, previous["fallback"], .5) < loss(candidate, rows, previous["fallback"], .5)
    event = next(a for a in report["actions"] if a["action"] == "resolve_response_question")
    assert "parent" in event["refuted_programs"]
    assert event["outcome"] == "candidate" and report["program"] == candidate.to_dict()
    assert report["visit_costs"]["observation_cost"] == 48
    assert report["visit_costs"]["compute_cost"] <= 500
    assert library.transfer_refuted(lab.spec, parent)


def test_refuting_parent_keeps_unconfirmed_candidate_pending(case):
    previous = case["previous"]
    question = previous["response_questions"][0]
    question["next_probe_length"] = 8
    lab = laboratory(case["task"])
    library = KnowledgeLibrary.from_dict(case["library"])
    remember_failure(library, lab.spec, case)
    offer = quote(previous, question, library, lab.spec)
    config = ResearchConfig(**{**case["config"], "seed": case["task"]["seed"],
        "observation_budget": previous["costs"]["observation_cost"] + 16,
        "compute_budget": previous["costs"]["compute_cost"] + 80})
    report = UnifiedResearchAgent(config, measurement_budget=0, library=library).run(
        lab, previous=previous, continuation_candidate=offer["candidate"])
    assert not report["noise_assessment"]["adequate"]
    assert not report["noise_assessment"]["repeatable_error"]
    assert report["status"] == "unresolved"
    assert report["response_questions"][0]["status"] == "pending"
    assert report["program"] == question["candidate"]["candidate"]["program"]
    assert report["visit_costs"]["observation_cost"] == 16
    assert report["visit_costs"]["compute_cost"] == 80


@pytest.mark.parametrize("retain_third_incumbent", [False, True])
def test_two_measured_refutations_stop_both_requeue_and_an_already_queued_purchase(
        case, retain_third_incumbent):
    previous = case["previous"]
    question = previous["response_questions"][0]
    retained_program = deepcopy(question["candidate"]["candidate"]["program"])
    # Save the offer at the time it really could enter the queue, before a
    # later incumbent changes its value and paid evidence refutes both rivals.
    question["candidate"] = deepcopy(case["failed_entries"][1]["entry"])
    stale_offer = quote(previous, question)
    assert stale_offer is not None
    if retain_third_incumbent:
        # A later answer need not be either of an older question's rivals.
        previous["program"] = retained_program
    previous["status"] = "completed"
    lab = laboratory(case["task"])
    library = KnowledgeLibrary.from_dict(case["library"])
    assert remember_failure(library, lab.spec, case, 0)
    assert remember_failure(library, lab.spec, case, 1)
    library = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    assert quote(previous, question, library, lab.spec) is None
    config = ResearchConfig(**{**case["config"], "seed": case["task"]["seed"],
        "observation_budget": previous["costs"]["observation_cost"] + 48,
        "compute_budget": previous["costs"]["compute_cost"] + 500})
    session = UnifiedResearchSession(config, tasks=[case["task"]], measurement_budget=0,
        library=library, lab_factory=laboratory)
    session.state["results"] = [{"task_position": 0, "research": previous}]
    session._offer_response_questions()
    assert not session.state["pending_revisits"]
    report = UnifiedResearchAgent(config, measurement_budget=0, library=library).run(
        lab, previous=previous, continuation_candidate=stale_offer["candidate"])
    assert lab.costs["observation_cost"] == 0 and not lab.acquisitions
    assert not any(report["visit_costs"].values())
    assert report["batches"] == previous["batches"]
    assert report["response_questions"][0]["parent"] == question["parent"]
    assert report["response_questions"][0]["candidate"] == question["candidate"]
    assert report["response_questions"][0]["status"] == "refuted"
    assert report["program"] == previous["program"]
    assert report["status"] == ("completed" if retain_third_incumbent else "unresolved")
    if not retain_third_incumbent:
        assert not report["noise_assessment"]["adequate"]
