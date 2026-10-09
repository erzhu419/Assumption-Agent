"""Research families compete on the pool remaining after paid screening."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.instrument_condition_revisits import instrument_condition_offer
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.numeric_method_revisits import reprice_numeric_method_offer
from hegel_machine.cognition.unified_session import UnifiedResearchSession
from test_cognition_instrument_condition_revisits import (
    POOL, laboratory_factory, paid_case, screened,
)


def method_offer(costs, *, gain=.5, trials=0, observed_gain=None, information=1.):
    return {"candidate": {"kind": "successor_partial_method", "method_experiment": {},
        "method_opportunity": {"paid_counterexample_loss": 1.,
            "predicted_separation": 0., "observed_visit_trials": trials,
            "observed_expected_gain": observed_gain,
            "confirmation_allocation": {"calibrated_information_probability": information},
            "recommended_resources": dict(costs), "planning_compute_cost": 30000,
            "draft_ids": ["draft"]}},
        "basis": "new_partial_method_distinguishing_experiment",
        "old_evidence_gain": gain, "minimum_resources": dict(costs)}


def test_equal_value_and_future_cost_have_equal_method_and_instrument_priority(screened):
    instrument = instrument_condition_offer(screened["screen"], POOL)
    assert instrument is not None
    decision = instrument["investment_decision"]
    method = method_offer(instrument["expected_resource_costs"],
        gain=decision["expected_marginal_value"])
    before = deepcopy(method)
    reprice_numeric_method_offer(method, POOL)
    assert method["relative_gain"] == pytest.approx(instrument["relative_gain"])
    assert method["resource_share"] == pytest.approx(decision["resource_share"])
    assert method["candidate"] == before["candidate"]
    assert method["minimum_resources"] == before["minimum_resources"]
    # Already paid screening still belongs to attribution, but does not alter
    # future execution cost or the priority on the same remaining pool.
    method["candidate"]["method_opportunity"]["planning_compute_cost"] *= 10
    reprice_numeric_method_offer(method, POOL)
    assert method["relative_gain"] == pytest.approx(instrument["relative_gain"])


def test_repricing_keeps_learned_return_and_sampling_probability():
    costs = dict(observation_cost=72, measurement_cost=72, compute_cost=320000)
    method = method_offer(costs, gain=.8, trials=3, observed_gain=-.2, information=.4)
    reprice_numeric_method_offer(method, POOL)
    assert method["expected_marginal_value"] == pytest.approx(.24)
    initial = deepcopy(method)
    smaller = {key: value / 2 for key, value in POOL.items()}
    reprice_numeric_method_offer(method, smaller)
    assert method["relative_gain"] == pytest.approx(initial["relative_gain"] / 2)
    assert method["candidate"] == initial["candidate"]
    assert method["expected_resource_costs"] == costs
    assert method["minimum_resources"] == costs


def test_shared_basis_does_not_read_numeric_evidence_from_a_risk_method():
    risk = {"basis": "new_partial_method_distinguishing_experiment",
        "candidate": {"risk_method": {"steps": ["compare_risk"]}},
        "relative_gain": .7, "minimum_resources": dict(POOL)}
    before = deepcopy(risk)
    assert reprice_numeric_method_offer(risk, POOL) == before


def test_queued_method_is_repriced_before_replacing_it_with_a_new_paid_offer():
    config = ResearchConfig(observation_budget=100, compute_budget=1000, use_memory=False)
    session = UnifiedResearchSession(config, tasks=[{}], measurement_budget=100)
    session.state["results"] = [{"task_position": 0, "research": {
        "costs": dict(observation_cost=0, measurement_cost=0, compute_cost=0)}}]
    costs = dict(observation_cost=50, measurement_cost=50, compute_cost=100)
    incumbent = method_offer(costs, gain=.5)
    incumbent.update(task_position=0, source_task_position=1)
    reprice_numeric_method_offer(incumbent, {key: value * 10 for key, value in session._remaining().items()})
    stale_priority = incumbent["relative_gain"]
    session.state["pending_revisits"] = [incumbent]
    proposed = method_offer(costs, gain=.6)
    event = {"scheduled": False}
    session._queue_knowledge_offer(0, 2, proposed, 100, [], event)
    selected, = session.state["pending_revisits"]
    assert selected is proposed and event["scheduled"]
    assert selected["relative_gain"] < stale_priority
    assert incumbent["relative_gain"] < selected["relative_gain"]
    assert incumbent["pricing_resources"] == selected["pricing_resources"] == session._remaining()


def test_all_instrument_screens_are_paid_before_the_mixed_queue_is_priced(
        paid_case, screened, monkeypatch):
    config = ResearchConfig(**{**paid_case["config"], "use_memory": False})
    class Laboratory:
        spec = SimpleNamespace(max_probe_length=24, risk_quantile=.5)
        channels = [SimpleNamespace(name="meter", cost_per_row=1)]
        def __init__(self, task):
            self.cost = task["screen_fee"]
        def __enter__(self):
            return self
        def __exit__(self, *unused):
            pass
    session = UnifiedResearchSession(config, tasks=[
        {"screen_fee": 1000}, {"screen_fee": 7000}], measurement_budget=320,
        lab_factory=Laboratory)
    for position, task in enumerate(session.state["tasks"]):
        previous = deepcopy(paid_case["report"])
        task["research_limits"] = {key.replace("_cost", "_budget"):
            previous["costs"][key] + value for key, value in POOL.items()}
        session.state["results"].append({"task_position": position, "research": previous})
    fees = []
    def screen(agent, laboratory, previous, budget):
        assert budget.spend(laboratory.cost)
        fees.append(laboratory.cost)
        result = deepcopy(screened["screen"])
        result["compute_cost"] = laboratory.cost
        return result
    monkeypatch.setattr("hegel_machine.cognition.instrument_condition_revisits."
        "screen_instrument_condition_revisit", screen)
    method = method_offer(dict(observation_cost=72, measurement_cost=72, compute_cost=320000))
    method.update(task_position=0, source_task_position=1)
    session.state["pending_revisits"] = [method]
    session._offer_response_questions()
    assert fees == [1000, 7000]
    remaining = session._remaining()
    instruments = [offer for offer in session.state["pending_revisits"]
                   if offer["basis"] == "unresolved_instrument_condition"]
    assert len(instruments) == 2
    assert method["pricing_resources"] == remaining
    for offer in instruments:
        decision = offer["investment_decision"]
        actual = max(offer["expected_resource_costs"][key] / remaining[key] for key in remaining)
        assert decision["resource_share"] == pytest.approx(actual)
    first = deepcopy(session.state["pending_revisits"])
    session._offer_response_questions()
    assert session.state["pending_revisits"] == first
    assert fees == [1000, 7000]
    assert session.state["screening_compute_cost"] == 8000
