"""Unresolved instrument risk can use visited-task savings after JSON restore."""

from copy import deepcopy
import json
import pytest

from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_session import UnifiedResearchSession
from test_cognition_instrument_condition_revisits import paid_case, laboratory_factory, POOL


def session_for(paid_case, laboratory_factory, pool):
    previous = deepcopy(paid_case["report"])
    task = {**paid_case["task"], "research_limits": {
        name.replace("_cost", "_budget"): previous["costs"][name] + amount
        for name, amount in pool.items()}}
    session = UnifiedResearchSession(ResearchConfig(**paid_case["config"]),
        tasks=[task], measurement_budget=320, lab_factory=laboratory_factory)
    session.state["results"] = [{"task_position": 0, "available_concepts": [],
                                  "research": previous}]
    return session


@pytest.mark.parametrize("compute", [1000, 96513])
def test_instrument_screen_survives_json_and_reprices_only_visited_savings(
        paid_case, laboratory_factory, compute):
    session = session_for(paid_case, laboratory_factory,
        dict(observation_cost=116, measurement_cost=116, compute_cost=compute))
    assert session._offer_instrument_conditions([]) == []
    screening = session.state["screening_compute_cost"]
    assert screening > 0
    quote, = session.state["condition_revisit_quotes"]
    assert (quote["screen"] is not None) == (compute == 96513)
    saved = json.loads(json.dumps(session.to_dict()))
    restored = UnifiedResearchSession(state=saved, lab_factory=laboratory_factory)
    assert restored._offer_instrument_conditions([]) == []
    assert restored.state["screening_compute_cost"] == screening
    # Another task's allocation stays reserved until it has actually run.
    remaining = restored._remaining()
    restored.state["tasks"].append(deepcopy(paid_case["task"]))
    assert restored._remaining() == remaining
    # This budget-only completed task contributes its real unused allocation;
    # no observations or knowledge from it enter the instrument forecast.
    restored.state["results"].append({"task_position": 1, "research": {
        "status": "completed", "selected_channels": [],
        "costs": dict(observation_cost=40, measurement_cost=40, compute_cost=1000)}})
    offers = restored._offer_instrument_conditions([])
    # Funding a complete cycle is separate from deciding it is worthwhile.
    # More real savings can make the same frozen plan worth buying.
    after_first_saving = restored.state["screening_compute_cost"]
    restored.state["tasks"].append(deepcopy(paid_case["task"]))
    restored.state["results"].append({"task_position": 2, "research": {
        "status": "completed", "selected_channels": [],
        "costs": dict(observation_cost=40, measurement_cost=40, compute_cost=1000)}})
    offers = restored._offer_instrument_conditions([])
    offer, = offers
    assert offer["basis"] == "unresolved_instrument_condition"
    assert offer["task_position"] == offer["source_task_position"] == 0
    assert offer["investment_decision"]["buy"]
    assert restored.state["screening_compute_cost"] == after_first_saving
    if compute == 96513:
        assert restored.state["screening_compute_cost"] == screening
        assert restored.state["condition_revisit_quotes"] == saved["condition_revisit_quotes"]
    else:
        assert restored.state["screening_compute_cost"] > screening
        current = restored.state["condition_revisit_quotes"][0]
        assert current["screen"] is not None
        # Earlier unfunded planning belongs to this same research question.
        # Its fee reaches the eventual experience without billing it again.
        assert current["screening_compute_cost"] == restored.state["screening_compute_cost"]
        assert current["screen"]["compute_cost"] == current["screening_compute_cost"]
        screening = restored.state["screening_compute_cost"]
        assert restored._offer_instrument_conditions([]) == offers
        assert restored.state["screening_compute_cost"] == screening


def test_session_executes_the_shared_offer_and_retains_actual_budget_and_evidence(
        paid_case, laboratory_factory):
    session = session_for(paid_case, laboratory_factory, POOL)
    session.state["pending_revisits"] = session._offer_instrument_conditions([])
    offer, = session.state["pending_revisits"]
    old = deepcopy(paid_case["report"])
    # Isolate one complete quoted cycle; the session still prices subsequent
    # opportunities, which can consume the remaining pool on later steps.
    session.state["pending_revisits"].remove(offer)
    assert session._perform_revisit(offer, allocation_limits=offer["minimum_resources"])
    result = session.to_dict()
    visit, = result["revisits"]
    current = result["results"][0]["research"]
    assert visit["basis"] == "unresolved_instrument_condition"
    assert current["batches"][:len(old["batches"])] == old["batches"]
    assert current["visit_costs"]["observation_cost"] == 48
    assert len(current["batches"]) == len(old["batches"]) + 2
    assert current["batches"][-1]["probe"]["probe_id"] != current["batches"][-2]["probe"]["probe_id"]
    for resource, maximum in offer["minimum_resources"].items():
        assert current["costs"][resource] == old["costs"][resource] + current["visit_costs"][resource]
        assert current["visit_costs"][resource] <= maximum
        assert result["costs"][resource] == current["costs"][resource] + (
            result["screening_compute_cost"] if resource == "compute_cost" else 0)
        assert session._remaining()[resource] >= 0
