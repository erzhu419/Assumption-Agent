"""Price surviving hypotheses against the answer now supported by paid data."""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
import gzip
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig
from hegel_machine.cognition.object_continuation import encode_entry, observations
from hegel_machine.cognition.response_research_questions import response_question_offer
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_session import UnifiedResearchSession
from test_cognition_noise_research_reopening import laboratory
from test_cognition_response_question_investment import (
    CONFIG, PaidResponseLab, pending_response_rivals, persisted,
)


@pytest.fixture
def root03():
    path = Path(__file__).parent / "fixtures/current_question_root03.json.gz"
    return json.loads(gzip.decompress(path.read_bytes()))


def session_for(payload, *, compute_remaining=10000):
    previous = deepcopy(payload["after"])
    task = {key: value for key, value in payload["task"].items()
            if key not in {"family", "research_limits"}}
    config = ResearchConfig(**{**payload["config"], "seed": task["seed"],
        "observation_budget": previous["costs"]["observation_cost"] + 144,
        "compute_budget": previous["costs"]["compute_cost"] + compute_remaining})
    session = UnifiedResearchSession(config, tasks=[task], measurement_budget=0,
                                     lab_factory=laboratory)
    session.state["results"] = [{"task_position": 0, "research": previous}]
    session.state["status"] = "paused"
    return session


def test_real_adoption_reprices_shared_paid_evidence_and_changes_question_order(root03):
    session = session_for(root03)
    before = deepcopy(session.state["results"][0]["research"])
    # A repeated trajectory improves its noise estimate; it is not another
    # independent choice of commands. Give each distinct command trajectory
    # its original row weight when comparing the stored frozen responses.
    repeats = Counter(json.dumps(batch["probe"]["sequence"], sort_keys=True) for batch in before["batches"])

    def paid_loss(program, fallback):
        weighted, count = 0., 0.
        for batch in before["batches"]:
            rows = observations(batch["observations"])
            weight = len(rows) / repeats[json.dumps(batch["probe"]["sequence"], sort_keys=True)]
            weighted += weight * loss(program, rows, fallback, .5)
            count += weight
        return weighted / count

    incumbent_loss = paid_loss(Program.from_dict(before["program"]), before["fallback"])
    pending = {q["question_id"]: q for q in before["response_questions"] if q["status"] == "pending"}
    session._offer_response_questions()
    offers = session.state["pending_revisits"]
    assert offers, "small uncertain gains retain a payable discriminating experiment"
    for offer in offers:
        question = pending[offer["question_id"]]
        candidate = question["candidate"]
        candidate_loss = paid_loss(Program.from_dict(candidate["candidate"]["program"]), candidate["fallback"])
        assert offer["old_evidence_gain"] == pytest.approx(incumbent_loss - candidate_loss)
        assert abs(offer["old_evidence_gain"]) < abs(question["observed_gain"]) / 10
    assert max(offers, key=lambda item: item["relative_gain"])["question_id"] == "response_question_0004"
    # Repricing informs resource choice; it neither adopts a rival nor fabricates
    # a fresh confirmation from the rows already used to select the incumbent.
    current = session.state["results"][0]["research"]
    for key in ("program", "confirmation_loss", "status", "batches", "concepts_added"):
        assert current[key] == before[key]
    for old, new in zip(before["response_questions"], current["response_questions"]):
        for key in ("parent", "candidate", "observed_gain", "uncertainty_margin", "gains", "probe_ids"):
            assert new[key] == old[key]


def test_paid_repricing_is_charged_once_and_reused_after_session_restore(root03):
    session = session_for(root03)
    before = session.to_dict()["costs"]
    session._offer_response_questions()
    quoted = persisted(session.to_dict())
    assert quoted["screening_compute_cost"] > 0
    assert quoted["costs"]["compute_cost"] - before["compute_cost"] == quoted["screening_compute_cost"]
    assert quoted["costs"]["observation_cost"] == before["observation_cost"]
    restored = UnifiedResearchSession(state=quoted, lab_factory=laboratory)
    restored._offer_response_questions()
    repeated = restored.to_dict()
    assert repeated["costs"] == quoted["costs"]
    assert repeated["pending_revisits"] == quoted["pending_revisits"]
    # A further real paid pair invalidates the valuation, even with the same
    # incumbent. Charging it is separate from charging fresh acquisition.
    report = restored.state["results"][0]["research"]
    report["batches"].extend(deepcopy(root03["additional_paid_batches"]))
    report["costs"]["observation_cost"] += sum(
        len(batch["observations"]) for batch in root03["additional_paid_batches"])
    restored._offer_response_questions()
    assert restored.state["screening_compute_cost"] > quoted["screening_compute_cost"]


def test_insufficient_repricing_budget_does_not_reuse_obsolete_parent_price(root03):
    session = session_for(root03, compute_remaining=400)
    old = persisted(session.to_dict())
    session._offer_response_questions()
    current = session.to_dict()
    assert not current["pending_revisits"]
    assert current["costs"]["compute_cost"] <= old["costs"]["compute_cost"] + 400
    assert current["costs"]["observation_cost"] == old["costs"]["observation_cost"]
    assert current["results"][0]["research"]["program"] == old["results"][0]["research"]["program"]
    assert [q["status"] for q in current["results"][0]["research"]["response_questions"]] == [
        q["status"] for q in old["results"][0]["research"]["response_questions"]]


def test_zero_mean_gain_with_unresolved_response_difference_still_merits_an_experiment():
    lab = PaidResponseLab(1., 16, noise=0.)
    parent = Program("const", value=2.)
    incumbent = Program("feature", name="x")
    rival = Program("mul", children=(Program("const", value=-1.), incumbent))
    entry = lambda program: encode_entry({"candidate": Candidate(program.describe(), program, "proposal"),
                                          "kind": "affine", "fallback": 0., "validation_loss": 1.})
    rows = [Observation({"x": x}, 1., ()) for x in (-1., 1.) * 12]
    question = {"question_id": "opposite_response", "parent": entry(parent), "candidate": entry(rival),
        "observed_gain": 1., "uncertainty_margin": 1., "rows": 24, "reference_loss": 1.,
        "probe_ids": ["paid"], "status": "pending", "pilot_only": True,
        "gains": [], "reference_losses": [], "candidate_losses": []}
    previous = {"task_id": lab.spec.task_id, "status": "unresolved", "program": incumbent.to_dict(),
        "expression": incumbent.describe(), "selected_kind": "affine", "fallback": 0.,
        "selected_channels": [], "selection_loss": 2., "confirmation_loss": None,
        "batches": [{"probe": {"probe_id": "paid", "sequence": [{"x": row.features["x"]} for row in rows]},
                     "channels": [], "observations": [asdict(row) for row in rows]}],
        "costs": {"observation_cost": 24, "measurement_cost": 0, "compute_cost": 48, "queries": 1},
        "actions": [], "used_sources": [], "concepts_added": [], "response_questions": [question]}
    session = UnifiedResearchSession(ResearchConfig(observation_budget=192, compute_budget=20000),
        tasks=[{"seed": 16}], measurement_budget=0, lab_factory=lambda item: lab)
    session.state["results"] = [{"task_position": 0, "research": previous}]
    session._offer_response_questions()
    offer, = session.state["pending_revisits"]
    assert offer["old_evidence_gain"] == pytest.approx(0.)
    assert offer["relative_gain"] > 0
    assert question["status"] == "pending"
    assert previous["program"] == incumbent.to_dict()


def test_direct_saved_offer_reprices_before_purchase_and_preserves_the_confirmed_answer(root03):
    lab, previous, questions, first, _ = pending_response_rivals()
    limits = {"observation_budget": CONFIG.observation_budget, "compute_budget": CONFIG.compute_budget}
    available = {"observation_cost": 144, "measurement_cost": 0, "compute_cost": 10000}
    first_offer, stale_rival = [response_question_offer(previous, question, 24, available,
        resource_limits=limits) for question in questions]
    adopted = persisted(UnifiedResearchAgent(CONFIG, measurement_budget=0).run(
        lab, previous=previous, continuation_candidate=first_offer["candidate"]))
    assert adopted["program"] == first["candidate"].program.to_dict()
    assert adopted["status"] == "completed"
    assert len(adopted["batches"]) > len(previous["batches"])
    purchases = len(lab.acquisitions)
    for allowance in (100, 10000):
        config = replace(CONFIG, compute_budget=adopted["costs"]["compute_cost"] + allowance)
        report = UnifiedResearchAgent(config, measurement_budget=0).run(
            lab, previous=persisted(adopted), continuation_candidate=stale_rival["candidate"])
        assert len(lab.acquisitions) == purchases
        assert report["program"] == adopted["program"]
        assert report["status"] == "completed"
        assert report["confirmation_loss"] == adopted["confirmation_loss"]
        assert report["response_questions"][1]["status"] == "pending"
        assert report["visit_costs"]["observation_cost"] == 0
        assert report["batches"] == adopted["batches"]
        assert report["visit_costs"]["compute_cost"] == (
            0 if allowance == 100 else 3 * adopted["costs"]["observation_cost"])
        if allowance == 10000:
            assert report["response_questions"][1]["current_quote"]["expected_gain"] <= CONFIG.minimum_improvement
            again = UnifiedResearchAgent(config, measurement_budget=0).run(
                lab, previous=persisted(report), continuation_candidate=stale_rival["candidate"])
            assert again["costs"] == report["costs"]
            assert not any(again["visit_costs"].values())
    # A normally priced real root03 offer pays for its fresh comparison only;
    # the session's current-evidence scoring must not be charged a second time.
    session = session_for(root03)
    session._offer_response_questions()
    offer = max(session.state["pending_revisits"], key=lambda item: item["relative_gain"])
    paid = session.state["results"][0]["research"]
    real_lab = laboratory(session.state["tasks"][0])
    report = UnifiedResearchAgent(session.config, measurement_budget=0).run(
        real_lab, previous=paid, continuation_candidate=offer["candidate"])
    assert len(report["batches"]) == len(paid["batches"]) + 2
    assert report["visit_costs"]["compute_cost"] == offer["minimum_resources"]["compute_cost"]
    assert report["visit_costs"]["observation_cost"] == offer["minimum_resources"]["observation_cost"]
