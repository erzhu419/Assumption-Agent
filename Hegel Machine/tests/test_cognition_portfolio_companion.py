"""A chosen companion is repriced on evidence, not rejected by a smaller pool."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.condition_marginal_value import CONDITION_FOLLOWUP_ACTION
from hegel_machine.cognition.instrument_condition_revisits import instrument_condition_offer
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.research_portfolio import choose_research_visit
from hegel_machine.cognition.unified_session import UnifiedResearchSession
from test_cognition_instrument_condition_revisits import paid_case, laboratory_factory, POOL
from test_cognition_shared_condition_session import session_for


@pytest.fixture(scope="module")
def prepared(paid_case, laboratory_factory):
    session = session_for(paid_case, laboratory_factory, POOL)
    session.state["pending_revisits"] = session._offer_instrument_conditions([])
    offer, = session.state["pending_revisits"]
    assert offer["investment_decision"]["buy"]
    # This is a controlled queue transition around a real paid screen. It does
    # not recreate the root07/root09 historical first visit or its outcome.
    session._remember_portfolio_companion(
        {"decision": {"companion": {"index": 0}}}, session._remaining())
    return session.to_dict()


def restored(prepared, laboratory_factory):
    return UnifiedResearchSession(state=json.loads(json.dumps(prepared)),
                                  lab_factory=laboratory_factory)


def set_remaining(session, resources):
    costs = session._costs()
    session.state["tasks"][0]["research_limits"] = {
        key.replace("_cost", "_budget"): costs[key] + amount
        for key, amount in resources.items()}


def reduced_pool(session):
    # The real complete floor is 110504 compute. A 150000-compute remainder
    # affords it but rejects its .245 local value. No gain or plan is changed.
    pool = {"observation_cost": 144, "measurement_cost": 144, "compute_cost": 150000}
    set_remaining(session, pool)
    return pool


def test_same_paid_offer_survives_balance_only_repricing_and_json(prepared, laboratory_factory):
    session = restored(prepared, laboratory_factory)
    pool = reduced_pool(session)
    screen = session.state["condition_revisit_quotes"][0]["screen"]
    assert instrument_condition_offer(screen, pool, experience=session.library.action_experience) is None
    before = deepcopy(session.state["condition_revisit_quotes"])
    cost = session.state["screening_compute_cost"]
    session._offer_response_questions()
    offer, = session.state["pending_revisits"]
    decision = offer["investment_decision"]
    assert decision["buy"] and not decision["local_buy"]
    assert decision["reason"] == "selected_portfolio_companion"
    assert decision["admission_resource_share"] < decision["expected_marginal_value"]
    assert decision["expected_marginal_value"] < decision["resource_share"]
    assert offer["minimum_resources"] == prepared["pending_revisits"][0]["minimum_resources"]
    session = restored(session.to_dict(), laboratory_factory)
    session._offer_response_questions()
    assert session.state["pending_revisits"] == [offer]
    assert session.state["condition_revisit_quotes"] == before
    assert session.state["screening_compute_cost"] == cost


@pytest.mark.parametrize("resource", ["observation_cost", "measurement_cost", "compute_cost"])
def test_companion_does_not_borrow_its_independent_confirmation_floor(
        prepared, laboratory_factory, resource):
    session = restored(prepared, laboratory_factory)
    screen = session.state["condition_revisit_quotes"][0]["screen"]
    minimum = prepared["pending_revisits"][0]["minimum_resources"]
    pool = {**POOL, resource: minimum[resource] - 1}
    assert instrument_condition_offer(screen, pool, experience=session.library.action_experience,
        admission_resources=session.state["portfolio_companion"]["admission_resources"]) is None


def test_new_applicable_negative_information_can_revoke_old_admission(prepared, laboratory_factory):
    session = restored(prepared, laboratory_factory)
    pool = reduced_pool(session)
    screen = session.state["condition_revisit_quotes"][0]["screen"]
    experience = session.library.action_experience
    record = experience[screen["marginal_context"]][CONDITION_FOLLOWUP_ACTION]
    # Controlled feedback input: 30 predicted hits in this SAME paid sampling
    # context yielded none. It is not a claim about the fixture's actual runs.
    record["sampling_information"] = [{"sampling_context": deepcopy(screen["plan"]["sampling_context"]),
        "information_event_sum": 0, "predicted_information_event_sum": 30., "batches": 30}]
    before = session.state["screening_compute_cost"]
    assert instrument_condition_offer(screen, pool, experience=experience,
        admission_resources=session.state["portfolio_companion"]["admission_resources"]) is None
    session._offer_response_questions()
    assert not session.state["pending_revisits"]
    assert session.state["screening_compute_cost"] == before


def test_reselecting_the_same_companion_keeps_its_original_admission(prepared, laboratory_factory):
    session = restored(prepared, laboratory_factory)
    original = deepcopy(session.state["portfolio_companion"])
    remaining = reduced_pool(session)
    session._offer_response_questions()
    session._remember_portfolio_companion(
        {"decision": {"companion": {"index": 0}}}, remaining)
    assert session.state["portfolio_companion"] == original
    session._offer_response_questions()
    assert session.state["pending_revisits"][0]["investment_decision"]["reason"] == "selected_portfolio_companion"


def test_stronger_exclusive_rival_can_win_and_clears_companion_admission(prepared, laboratory_factory):
    session = restored(prepared, laboratory_factory)
    pool = reduced_pool(session)
    session._offer_response_questions()
    companion, = session.state["pending_revisits"]
    # A separate complete research action occupies this same resource pool.
    # Its explicit gain is a unit-test alternative, not fabricated feedback.
    rival = {"task_position": 1, "basis": "old_paid_evidence", "candidate": {},
        "minimum_resources": {**pool, "compute_cost": pool["compute_cost"] - 4},
        "expected_loss_reduction": 2 * companion["expected_loss_reduction"],
        "loss_kind": "squared_error"}
    session.state["pending_revisits"] = [companion, rival]
    choice = choose_research_visit(session.state["pending_revisits"], pool, SearchBudget(4),
                                   allocation_for=session._revisit_allocation)
    assert choice["offer"] is rival
    assert choice["decision"]["companion"] is None
    session._remember_portfolio_companion(choice, pool)
    assert "portfolio_companion" not in session.state
    assert session._offer_instrument_conditions([]) == []


@pytest.fixture(scope="module")
def executed(prepared, laboratory_factory):
    session = restored(prepared, laboratory_factory)
    reduced_pool(session)
    before = session.to_dict()
    # Isolate this admitted purchase; planning/selection/continuation and its
    # post-visit instrument refresh are real. Other new knowledge is separate.
    session._offer_new_knowledge = lambda *args, **kwargs: None
    result = session.run(max_tasks=1)
    return before, result


def test_restored_companion_reaches_real_execution_and_fresh_confirmation(executed):
    before, result = executed
    visit, = result["revisits"]
    old, current = before["results"][0]["research"], result["results"][0]["research"]
    assert visit["basis"] == "unresolved_instrument_condition"
    assert current["batches"][:len(old["batches"])] == old["batches"]
    acquired = current["batches"][len(old["batches"]):]
    assert len(acquired) >= 2
    pairs = [item for item in current["continuation"]["method_marginal_comparisons"]
             if item.get("before_loss") is not None]
    assert pairs
    fresh_ids = {batch["probe"]["probe_id"] for batch in acquired[1:]}
    assert all(pair["probe_id"] in fresh_ids for pair in pairs)
    assert len({pair["probe_id"] for pair in pairs}) == len(pairs)
    assert all(pair["probe_id"] != acquired[0]["probe"]["probe_id"] for pair in pairs)
    selected, = [event for event in result["scheduler_events"]
                 if event["action"] == "choose_research_portfolio"]
    for key in ("observation_cost", "measurement_cost", "compute_cost"):
        assert visit["visit_costs"][key] <= selected["allocation"][key]
        assert result["costs"][key] - before["costs"][key] == visit["visit_costs"][key] + (
            result["screening_compute_cost"] - before["screening_compute_cost"] if key == "compute_cost" else 0)


def test_new_actual_paid_batches_do_not_inherit_old_companion_admission(
        prepared, executed, laboratory_factory):
    session = restored(prepared, laboratory_factory)
    reduced_pool(session)
    session.state["results"][0]["research"] = deepcopy(executed[1]["results"][0]["research"])
    # Reuse the real completed continuation, including its fresh evidence;
    # never edit targets, clear counterexamples, or restore the former law.
    cost = session.state["screening_compute_cost"]
    session._offer_instrument_conditions([])
    assert "portfolio_companion" not in session.state
    assert session.state["screening_compute_cost"] > cost
    cached, = session.state["condition_revisit_quotes"]
    assert cached["evidence"]["paid_batches"] == len(executed[1]["results"][0]["research"]["batches"])
