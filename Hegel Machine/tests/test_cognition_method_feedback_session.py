"""Cached method queues consume response evidence without paying for a new fit.

The root11 fixture supplies a real paid quote and legacy route feedback. Queue
transitions use explicitly synthetic response feedback; actual response producers
are covered by test_cognition_method_response_attribution and the stream tests.
"""

from copy import deepcopy
import gzip
import importlib
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.action_policy import ResearchActionPolicy
from hegel_machine.cognition.knowledge_changes import knowledge_snapshot
from hegel_machine.cognition.numeric_method_revisits import METHOD_RESPONSE_ACTION, METHOD_VISIT_ACTION
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.unified_continuation import paid_view
from hegel_machine.cognition.unified_session import UnifiedResearchSession


RESOURCES = ("observation_cost", "measurement_cost", "compute_cost")


@pytest.fixture(scope="module")
def recorded():
    path = Path(__file__).parent / "fixtures/method_feedback_root11.json.gz"
    return json.loads(gzip.decompress(path.read_bytes()))


@pytest.fixture(scope="module")
def laboratory_factory():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
        return importlib.import_module("run_observed_condition_decoding_study").laboratory


def restore_histories(state):
    for record in state["results"]:
        for batch in record["research"]["batches"]:
            history = []
            for row in batch["observations"]:
                row["history"] = list(history)
                history.append({**row["features"], "__target__": row["target"]})


def no_laboratory(_arguments):
    raise AssertionError("Refreshing a purchased method quote must not open a laboratory")


@pytest.fixture(scope="module")
def screened(recorded, laboratory_factory):
    state = deepcopy(recorded["state"])
    restore_histories(state)
    session = UnifiedResearchSession(state=state, lab_factory=laboratory_factory)
    before = session._costs()
    assert session._remaining() == recorded["remaining"]
    session._offer_retained_methods(recorded["target_position"])
    return {"state": session.to_dict(), "before": before}


def restored(screened):
    return UnifiedResearchSession(state=json.loads(json.dumps(screened["state"])),
        lab_factory=no_laboratory)


def quote(session):
    attempt, = session.state["method_revisit_attempts"]
    return attempt["cached_quote"]


def apply_synthetic_response_feedback(session):
    """Isolate queue behavior; this is not the old fixture's measured route gain."""
    cached = quote(session)
    context = cached["pricing_inputs"]["method_context"].replace(METHOD_VISIT_ACTION, METHOD_RESPONSE_ACTION)
    session.library.action_experience[context] = {METHOD_RESPONSE_ACTION: {
        "knowledge_applicability": deepcopy(cached["candidate"]["method_opportunity"]["method_applicability"]),
        "trials": 1, "expected_gain": 1., "last_gain": 1., "mean_gain": 1.,
        "mean_compute_cost": 0.}}
    session._refresh_method_feedback()
    return context


def test_zero_gain_paid_screen_retains_the_executable_quote(screened, recorded):
    session = restored(screened)
    cached = quote(session)
    assert not session.state["pending_revisits"]
    assert cached["expected_marginal_value"] == 0.
    assert cached["method_feedback"]["trials"] == 0
    assert cached["old_evidence_gain"] < 0.
    assert cached["candidate"]["method_opportunity"]["predicted_separation"] == 0.
    assert cached["candidate"]["research_draft_id"] == recorded["source_draft_id"]
    assert session._costs()["compute_cost"] > screened["before"]["compute_cost"]
    for resource in RESOURCES[:2]:
        assert session._costs()[resource] == screened["before"][resource]


def test_legacy_route_success_does_not_reopen_a_paid_response_quote(screened, recorded):
    session = restored(screened)
    before = session._costs()
    session.library.action_experience.update(deepcopy(recorded["actual_method_feedback"]))
    session._refresh_method_feedback()
    assert not session.state["pending_revisits"]
    assert quote(session)["method_feedback"]["trials"] == 0
    assert quote(session)["expected_marginal_value"] == 0.
    assert session._costs() == before


def test_applicable_response_feedback_reopens_the_same_paid_plan_without_screening(screened, recorded):
    session = restored(screened)
    old = deepcopy(quote(session))
    before = session._costs()
    apply_synthetic_response_feedback(session)
    offer, = session.state["pending_revisits"]
    assert offer["expected_marginal_value"] > 0.
    assert offer["method_feedback"]["trials"] == 1
    assert offer["method_feedback"]["expected_gain"] == 1.
    assert offer["candidate"]["program"] == old["candidate"]["program"]
    assert offer["candidate"]["method_experiment"] == old["candidate"]["method_experiment"]
    assert offer["candidate"]["research_draft_id"] == recorded["source_draft_id"]
    assert offer["pricing_inputs"] == old["pricing_inputs"]
    assert session._costs() == before


def test_unrelated_feedback_and_json_refresh_do_not_pay_or_schedule_twice(screened, recorded):
    session = restored(screened)
    original_quote = deepcopy(quote(session))
    before = session._costs()
    context, values = next(iter(recorded["actual_method_feedback"].items()))
    unrelated = deepcopy(values)
    unrelated[METHOD_VISIT_ACTION]["knowledge_applicability"]["commands"] += 1
    session.library.action_experience[context + "_unrelated"] = unrelated
    session._refresh_method_feedback()
    assert not session.state["pending_revisits"]
    assert quote(session) == original_quote
    apply_synthetic_response_feedback(session)
    state = json.loads(json.dumps(session.to_dict()))
    session = UnifiedResearchSession(state=state, lab_factory=no_laboratory)
    events = deepcopy(session.state["scheduler_events"])
    pending = deepcopy(session.state["pending_revisits"])
    session._refresh_method_feedback()
    session._refresh_method_feedback()
    assert session.state["pending_revisits"] == pending
    assert session.state["scheduler_events"] == events
    assert session._costs() == before


def test_adverse_fresh_revision_withdraws_only_the_unexecuted_offer(screened, recorded):
    session = restored(screened)
    context = apply_synthetic_response_feedback(session)
    assert len(session.state["pending_revisits"]) == 1
    before = session._costs()
    drafts = deepcopy(session.library.research_drafts)
    ResearchActionPolicy.revise_feedback(session.library.action_experience,
        {"context": context, "action": METHOD_RESPONSE_ACTION, "trial": 1, "credited_gain": 1.},
        status="refuted", before_loss=1., after_loss=2., evidence_rows=24)
    session._refresh_method_feedback()
    assert not session.state["pending_revisits"]
    assert quote(session)["expected_marginal_value"] == 0.
    assert session.library.research_drafts == drafts
    assert session._costs() == before


def test_resolved_cached_winner_restores_unchanged_paid_competitor(screened, recorded):
    session = restored(screened)
    apply_synthetic_response_feedback(session)
    winning_attempt, = session.state["method_revisit_attempts"]
    winner, = session.state["pending_revisits"]
    # The winner is the real paid quote. Add a separate, previously paid
    # alternative to test queue selection, not this alternative's accuracy.
    # A different fitted coefficient makes its executable method distinct;
    # its structural applicability and measured method feedback stay the same.
    rival_source = deepcopy(session.library.research_drafts[0])
    rival_source["draft_id"] = "fixture_competing_paid_method"
    rival_source["source_task_id"] = "fixture_other_source"
    procedure = rival_source["research_program"]
    answer = procedure["graph"]["nodes"][procedure["output_node"]]["answer"]

    def change_coefficient(node):
        if node["op"] == "const" and node["value"] != 0.:
            node["value"] *= 2.
            return True
        return any(change_coefficient(child) for child in node.get("children", []))

    assert change_coefficient(answer)
    session.library.research_drafts.append(rival_source)
    rival = deepcopy(winner)
    candidate = rival["candidate"]
    candidate["research_draft_id"] = rival_source["draft_id"]
    candidate["research_program"] = deepcopy(procedure)
    candidate["program"] = deepcopy(answer)
    candidate["method_experiment"]["source_draft_ids"] = [rival_source["draft_id"]]
    candidate["method_opportunity"]["draft_ids"] = [rival_source["draft_id"]]
    candidate["method_opportunity"]["source_task_ids"] = [rival_source["source_task_id"]]
    rival_attempt = {**deepcopy(winning_attempt), "draft_id": rival_source["draft_id"],
        "method": knowledge_snapshot(session.library)["draft"][rival_source["draft_id"]],
        "cached_quote": deepcopy(rival)}
    session.state["method_revisit_attempts"].append(rival_attempt)
    session.state["screening_compute_cost"] += rival["screening_compute_cost"]
    session._queue_knowledge_offer(0, None, rival, rival["screening_compute_cost"],
        [rival_attempt], {"action": "fixture_paid_competing_quote", "scheduled": False})
    assert session.state["pending_revisits"][0]["candidate"]["research_draft_id"] == recorded["source_draft_id"]
    before_costs = session._costs()
    before_feedback = deepcopy(rival_attempt["cached_quote"]["method_feedback"])
    session.library.research_drafts[0]["resolved_concept_id"] = "fixture_resolved_source"
    session._refresh_method_feedback()
    remaining, = session.state["pending_revisits"]
    assert remaining["candidate"]["research_draft_id"] == rival_source["draft_id"]
    assert "cached_quote" not in winning_attempt
    assert rival_attempt["cached_quote"]["method_feedback"] == before_feedback
    assert session._costs() == before_costs


@pytest.mark.parametrize("resource", RESOURCES)
def test_reopened_method_still_requires_its_complete_hard_minimum(screened, recorded, resource):
    session = restored(screened)
    apply_synthetic_response_feedback(session)
    offer, = session.state["pending_revisits"]
    minimum = offer["minimum_resources"]
    inputs = offer["pricing_inputs"]
    assert minimum["compute_cost"] >= inputs["compute_quote"]
    assert minimum["observation_cost"] >= inputs["observation_quote"]
    limits = {**minimum, resource: minimum[resource] - 1}
    before = session._costs()
    assert not session._perform_revisit(offer, allocation_limits=limits)
    assert session.state["scheduler_events"][-1]["reason"] == "remaining_resources"
    assert session._costs() == before


def test_response_funded_plan_executes_and_measures_its_actual_predictor(
        screened, recorded, laboratory_factory, monkeypatch):
    session = restored(screened)
    apply_synthetic_response_feedback(session)
    offer, = session.state["pending_revisits"]
    # Isolate this one purchased continuation; subsequent library-wide
    # proposal screens are separate work and are exercised by stream studies.
    monkeypatch.setattr(session, "_offer_new_knowledge", lambda *args, **kwargs: None)
    monkeypatch.setattr(session, "_offer_response_questions", lambda: None)
    session.lab_factory = laboratory_factory
    previous = deepcopy(session.state["results"][0]["research"])
    before = session._costs()
    assert session._perform_revisit(offer)
    current = session.state["results"][0]["research"]
    visit = session.state["revisits"][-1]
    pairs = current["continuation"]["method_fresh_comparisons"]
    assert pairs and pairs[-1]["observation_rows"] > 0
    pair = pairs[-1]
    with laboratory_factory(session.state["tasks"][0]) as lab:
        batches = paid_view(lab, current, pair["attempted_channels"])
    payloads = [batch for batch in current["batches"]
                if set(pair["attempted_channels"]) <= set(batch["channels"])]
    fresh = next(rows for payload, rows in zip(payloads, batches)
                 if payload["probe"]["probe_id"] == pair["probe_id"])
    assert pair["before_loss"] == pytest.approx(loss(
        Program.from_dict(pair["baseline_program"]), fresh, pair["baseline_fallback"], .5))
    assert pair["attempted_after_loss"] == pytest.approx(loss(
        Program.from_dict(pair["attempted_program"]), fresh, pair["attempted_fallback"], .5))
    assert current["batches"][:len(previous["batches"])] == previous["batches"]
    assert len(current["batches"]) > len(previous["batches"])
    for resource in RESOURCES:
        assert session._costs()[resource] - before[resource] == visit["visit_costs"][resource]
    assert visit["method_visit_outcome"]["whole_visit_costs"]["compute_cost"] == (
        visit["visit_costs"]["compute_cost"] + offer["screening_compute_cost"])
    assert not any(attempt.get("cached_quote") for attempt in session.state["method_revisit_attempts"])
