"""Complete research purchases compete with the visits they would displace."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.research_portfolio import choose_research_visit
from hegel_machine.cognition.unified_session import UnifiedResearchSession, retain_revisit_offer
from test_cognition_instrument_condition_revisits import paid_case, laboratory_factory, POOL
from test_cognition_shared_condition_session import session_for


def resources(observations, compute, measurement=None):
    return {"observation_cost": observations,
            "measurement_cost": observations if measurement is None else measurement,
            "compute_cost": compute}


def offer(position, gain, minimum, *, allocation=None):
    return {"task_position": position, "expected_loss_reduction": gain,
            "loss_kind": "squared_error", "minimum_resources": minimum,
            "execution_allocation": minimum if allocation is None else allocation}


def complete_allocation(item, others, remaining):
    # These unit offers describe fixed complete visits, including confirmation.
    # The session integration below uses the actual executor's allocator.
    return item["execution_allocation"]


def test_two_complete_visits_beat_the_more_dense_single_and_price_its_opportunity_cost():
    candidates = [offer(0, 6., resources(60, 10)),
                  offer(1, 4., resources(50, 80)),
                  offer(2, 4., resources(50, 80))]
    pool = resources(100, 169)
    original = deepcopy(candidates)
    work = SearchBudget(9)
    choice = choose_research_visit(candidates, pool, work, allocation_for=complete_allocation)
    # The 6/.6 single is denser than either 4/.5 visit, but prevents either
    # companion; the two 4-unit visits fit together after paying comparison.
    assert choice["offer"] is candidates[1]
    assert choice["allocation"] == resources(50, 80)
    decision = choice["decision"]
    assert decision["selected_indices"] == [1, 2]
    assert decision["portfolio_expected_loss_reduction"] == 8.
    assert decision["alternative_expected_loss_reduction"] == 6.
    assert decision["retained_companion_expected_loss_reduction"] == 4.
    assert decision["opportunity_cost"] == 2.
    assert decision["net_expected_loss_reduction"] == 2.
    assert decision["companion"]["allocation"] == resources(50, 80)
    assert work.spent == decision["compute_cost"] == 9
    assert decision["remaining_resources"] == resources(100, 160)
    assert candidates == original and pool == resources(100, 169)


def test_alternative_laws_of_the_same_task_cannot_be_counted_as_two_repairs():
    candidates = [offer(0, 6., resources(60, 10)),
                  offer(1, 4., resources(50, 80)),
                  offer(1, 4., resources(50, 80))]
    result = choose_research_visit(candidates, resources(100, 169), SearchBudget(9),
                                  allocation_for=complete_allocation)
    assert result["offer"] is candidates[0]
    assert result["decision"]["selected_indices"] == [0]
    assert result["decision"]["companion"] is None
    assert result["decision"]["portfolio_expected_loss_reduction"] == 6.


def test_companion_must_fit_after_the_first_complete_execution_allocation():
    candidates = [offer(0, 5., resources(48, 100), allocation=resources(72, 180)),
                  offer(1, 4., resources(48, 100), allocation=resources(72, 180))]
    result = choose_research_visit(candidates, resources(120, 404), SearchBudget(4),
                                  allocation_for=complete_allocation)
    # Two minimum quotes fit, but two executable visits with their allowed
    # confirmation work do not. Selecting the first cannot promise both gains.
    assert result["offer"] is candidates[0]
    assert result["allocation"] == resources(72, 180)
    assert result["decision"]["portfolio_expected_loss_reduction"] == 5.
    assert result["decision"]["companion"] is None
    assert result["decision"]["opportunity_cost"] == 4.


def test_feasible_reverse_order_is_retained_when_the_denser_first_visit_would_block_it():
    candidates = [offer(0, 20., resources(60, 60)),
                  offer(1, 8., resources(40, 40))]

    def allocation_for(item, others, remaining):
        # Different continuation types can use different complete visit caps.
        # Visit 0 first would leave too little for 1, while executing 1 first
        # constrains visit 0 to a complete 60-row continuation that still fits.
        return resources(80, 80) if item is candidates[0] and others else item["minimum_resources"]

    work = SearchBudget(4)
    result = choose_research_visit(candidates, resources(100, 104), work,
                                  allocation_for=allocation_for)
    assert result["offer"] is candidates[1]
    assert result["decision"]["selected_indices"] == [1, 0]
    assert result["decision"]["portfolio_expected_loss_reduction"] == 28.
    assert result["allocation"] == resources(40, 40)
    assert result["decision"]["companion"]["allocation"] == resources(60, 60)
    assert work.spent == 4


@pytest.mark.parametrize("knowledge_tie_tolerance", [0., 1e-8])
@pytest.mark.parametrize("reverse", [False, True])
def test_pair_orders_keep_the_denser_first_visit_regardless_of_queue_order(
        knowledge_tie_tolerance, reverse):
    expensive = offer(0, 4., resources(60, 60))
    cheap = offer(1, 4., resources(20, 20))
    candidates = [cheap, expensive] if reverse else [expensive, cheap]
    choice = choose_research_visit(candidates, resources(100, 104), SearchBudget(4),
        allocation_for=complete_allocation, knowledge_tie_tolerance=knowledge_tie_tolerance)
    assert choice["offer"] is cheap
    assert choice["decision"]["portfolio_expected_loss_reduction"] == 8.
    assert choice["decision"]["companion"]["task_position"] == expensive["task_position"]
    assert choice["allocation"] == resources(20, 20)


def test_same_task_distinct_plans_survive_until_allocation_and_same_plan_only_refreshes():
    first = {**offer(0, 5., resources(48, 100)), "basis": "old_paid_evidence",
             "candidate": {"program": {"op": "const", "value": 1.}}}
    rival = {**offer(0, 4., resources(48, 100)), "basis": "new_partial_method_distinguishing_experiment",
             "candidate": {"program": {"op": "const", "value": 2.}}}
    pending = []
    assert retain_revisit_offer(pending, first)
    assert retain_revisit_offer(pending, rival)
    assert pending == [first, rival]
    # The common minimum says nothing about the two continuations' complete
    # allocations, so it cannot justify discarding the lower raw estimate.
    refreshed = {**deepcopy(first), "expected_loss_reduction": 3.}
    assert retain_revisit_offer(pending, refreshed)
    assert len(pending) == 2 and pending == [rival, refreshed]


@pytest.mark.parametrize("resource", ["observation_cost", "measurement_cost", "compute_cost"])
def test_comparison_cannot_spend_the_independent_confirmation_minimum(resource):
    minimum = resources(48, 100)
    pool = resources(48, 101)
    pool[resource] -= 1
    work = SearchBudget(1)
    result = choose_research_visit([offer(0, 3., minimum)], pool, work,
                                  allocation_for=complete_allocation)
    assert result is not None and result["offer"] is None
    assert result["decision"]["reason"] == "no_affordable_positive_portfolio"
    assert work.spent == result["decision"]["compute_cost"] == 1


def test_unsupported_or_unfunded_comparison_does_not_claim_a_supported_stop():
    valid = offer(0, 1., resources(48, 100))
    unsupported = {**offer(1, 1., resources(48, 100)), "loss_kind": "pinball"}

    def must_not_allocate(*args):
        raise AssertionError("An unsupported or unpaid comparison cannot allocate a visit")

    work = SearchBudget(100)
    assert choose_research_visit([valid, unsupported], resources(200, 1000), work,
                                allocation_for=must_not_allocate) is None
    assert work.spent == 0
    unfunded = SearchBudget(0)
    assert choose_research_visit([valid], resources(200, 1000), unfunded,
                                allocation_for=must_not_allocate) is None
    assert unfunded.spent == 0
    zero = choose_research_visit([offer(0, 0., resources(48, 100))], resources(200, 1000),
                                work, allocation_for=must_not_allocate)
    assert zero["offer"] is None and zero["decision"]["selected_indices"] == []
    assert zero["decision"]["reason"] == "no_affordable_positive_portfolio"
    assert work.spent == 1


def test_knowledge_return_resolves_only_a_measured_near_tie():
    minimum = resources(48, 100)
    unknown = {**offer(0, 1.0, minimum), "knowledge_return_score": 0.}
    # Same task position keeps the test at the single-visit choice boundary;
    # otherwise the bounded portfolio correctly buys both visits.
    confirmed = {**offer(0, 1.0 - 1e-10, minimum), "knowledge_return_score": .5}

    raw = choose_research_visit([unknown, confirmed], resources(200, 1000),
        SearchBudget(100), allocation_for=complete_allocation,
        knowledge_tie_tolerance=0.)
    assert raw["offer"] is unknown
    assert raw["decision"]["selection_basis"] == "largest_affordable_single_or_pair_return"

    resolved = choose_research_visit([unknown, confirmed], resources(200, 1000),
        SearchBudget(100), allocation_for=complete_allocation,
        knowledge_tie_tolerance=1e-8)
    assert resolved["offer"] is confirmed
    assert resolved["decision"]["selection_basis"] == "measured_return_within_gain_resolution"
    assert resolved["decision"]["gain_tie_tolerance"] == pytest.approx(1e-8)
    assert resolved["decision"]["gain_tied_portfolios"] == [[0], [1]]


def test_real_instrument_below_local_value_admission_is_not_quoted_or_purchased(
        paid_case, laboratory_factory):
    session = session_for(paid_case, laboratory_factory, POOL)
    initially_priced, = session._offer_instrument_conditions([])
    screen_cost = session.state["screening_compute_cost"]
    assert screen_cost > 0
    minimum = initially_priced["minimum_resources"]
    old = deepcopy(session.state["results"][0]["research"])
    # Affording a complete experiment and confirmation does not establish
    # value. This small remaining pool cannot justify this actual forecast.
    pool = {**minimum, "compute_cost": minimum["compute_cost"] + screen_cost + 1}
    session.state["tasks"][0]["research_limits"] = {
        key.replace("_cost", "_budget"): old["costs"][key] + value
        for key, value in pool.items()}
    session._offer_response_questions()
    assert not session.state["pending_revisits"]
    assert session.state["screening_compute_cost"] == screen_cost
    session = UnifiedResearchSession(state=json.loads(json.dumps(session.to_dict())),
                                     lab_factory=laboratory_factory)
    before = session._costs()
    result = session.run(max_tasks=1)
    assert result["revisits"] == []
    assert result["results"][0]["research"] == old
    assert result["costs"] == before
    assert result["screening_compute_cost"] == screen_cost
    assert not any(event["action"] == "choose_research_portfolio"
                   for event in result["scheduler_events"])


def test_real_admitted_instrument_is_globally_selected_and_independently_confirmed(
        paid_case, laboratory_factory, monkeypatch):
    session = session_for(paid_case, laboratory_factory, POOL)
    session._offer_response_questions()
    candidate, = session.state["pending_revisits"]
    assert candidate["investment_decision"]["buy"]
    assert candidate["expected_loss_reduction"] > 0.
    old = deepcopy(session.state["results"][0]["research"])
    screen_cost = session.state["screening_compute_cost"]
    session = UnifiedResearchSession(state=json.loads(json.dumps(session.to_dict())),
                                     lab_factory=laboratory_factory)
    # Subsequent knowledge proposal discovery is separate from this one
    # purchase; retain actual queue repricing, allocation and continuation.
    monkeypatch.setattr(session, "_offer_new_knowledge", lambda *args, **kwargs: None)
    before = session._costs()
    result = session.run(max_tasks=1)
    visit, = result["revisits"]
    current = result["results"][0]["research"]
    selected, = [event for event in result["scheduler_events"]
                 if event["action"] == "choose_research_portfolio"]
    assert selected["compute_cost"] == 1
    assert current["batches"][:len(old["batches"])] == old["batches"]
    bought = current["batches"][len(old["batches"]):]
    assert len(bought) >= 2
    frozen_plan = candidate["candidate"]["instrument_condition_experiment"]["plan"]
    assert bought[0]["probe"]["sequence"] == frozen_plan["probe"]["sequence"]
    batches = {batch["probe"]["probe_id"]: batch for batch in bought}
    assert len(batches) == len(bought)
    assert sum(len(batch["observations"]) for batch in bought) == visit["visit_costs"]["observation_cost"]
    confirmed = [item for item in current["continuation"]["method_marginal_comparisons"]
                 if item["before_loss"] is not None]
    assert confirmed
    for comparison in confirmed:
        assert comparison["probe_id"] != bought[0]["probe"]["probe_id"]
        batch = batches[comparison["probe_id"]]
        rows = [Observation(row["features"], row["target"], tuple(row["history"]))
                for row in batch["observations"]]
        assert comparison["observation_rows"] == len(rows)
        assert comparison["before_loss"] == pytest.approx(loss(
            Program.from_dict(comparison["baseline_program"]), rows, comparison["baseline_fallback"], .5))
        assert comparison["after_loss"] == pytest.approx(loss(
            Program.from_dict(comparison["program"]), rows, comparison["fallback"], .5))
    screen_delta = result["screening_compute_cost"] - screen_cost
    assert screen_delta >= selected["compute_cost"]
    for resource, allocation in selected["allocation"].items():
        assert visit["resource_limits"][resource] - old["costs"][resource] == allocation
        assert visit["visit_costs"][resource] <= allocation
        assert result["costs"][resource] - before[resource] == (
            visit["visit_costs"][resource] + (screen_delta if resource == "compute_cost" else 0))
        assert session._remaining()[resource] >= 0


def test_exact_gain_tie_prefers_lower_complete_allocation():
    # Equal forecasts should preserve the scarce pool for the next visit when
    # no measured source return separates the two plans.
    expensive = offer(0, 1., resources(80, 80), allocation=resources(80, 80))
    cheap = offer(0, 1., resources(20, 20), allocation=resources(20, 20))
    choice = choose_research_visit([expensive, cheap], resources(100, 100),
        SearchBudget(10), allocation_for=complete_allocation, knowledge_tie_tolerance=0.)
    assert choice["offer"] is cheap
    assert choice["decision"]["selected_indices"] == [1]
    assert choice["decision"]["selected_resource_share"] == pytest.approx(20 / 96)
