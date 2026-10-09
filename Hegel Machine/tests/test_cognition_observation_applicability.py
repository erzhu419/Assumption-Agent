"""Actual method availability scopes observation research experience across tasks."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.dynamic_instrument_lab import DynamicInstrumentLab
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, SearchBudget
from hegel_machine.cognition.observation_research import research_observation_actions
from hegel_machine.cognition.research_graph import ResearchGraph
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_session import UnifiedResearchSession

from test_cognition_observation_research import ActualPurchases, CONFIG, assert_fresh_solution, chosen


SEED = 95201900
TASKS = [
    {"environment": "ambiguous_instruments", "task_index": 0, "seed": SEED,
     "apparatus_seed": SEED, "instrument_mode": "recurrent"},
    {"environment": "ambiguous_instruments", "task_index": 1, "seed": SEED + 10007,
     "apparatus_seed": SEED, "instrument_mode": "recurrent"},
    {"environment": "dynamic_instruments", "task_index": 0, "seed": SEED + 20014,
     "apparatus_seed": SEED + 500009, "profile": "confirmation"},
    {"environment": "dynamic_instruments", "task_index": 2, "seed": SEED + 40028,
     "apparatus_seed": SEED + 500009, "profile": "confirmation"},
]


@pytest.fixture(scope="module")
def stream():
    """All knowledge and action returns originate in this complete local stream."""
    session = UnifiedResearchSession(replace(CONFIG, seed=SEED, max_candidates=16),
        tasks=TASKS, measurement_budget=1536)
    snapshots = []
    for _ in TASKS:
        state = session.run(max_tasks=1)
        # Full task state, including concepts, instrumentation and policy
        # experience, crosses each boundary together.
        snapshots.append(json.loads(json.dumps(state)))
        session = UnifiedResearchSession(state=snapshots[-1])
    assert snapshots[-1]["status"] == "completed"
    return snapshots


def test_real_remeasurement_success_does_not_override_a_later_applicable_zero_meter_revision(stream):
    later = stream[3]["results"][-1]["research"]
    # The session can now complete this task by replaying a learned acquisition
    # DAG. Retain its response knowledge but remove that separate method supply
    # in this branch, so the native observation choice under test really runs.
    library = KnowledgeLibrary.from_dict(stream[1]["knowledge"])
    library.research_workflows.methods.clear()
    source_agent = UnifiedResearchAgent(replace(CONFIG, seed=SEED, max_candidates=16),
        measurement_budget=1536, library=library)
    with DynamicInstrumentLab(0, SEED + 20014, SEED + 500009, "confirmation") as laboratory:
        unknown = source_agent.run(laboratory)
    source_state = {"knowledge": library.to_dict(), "results": [{"research": unknown}]}
    earlier_choices = chosen(unknown["actions"])
    assert earlier_choices
    earlier = next(item for item in earlier_choices if item["selected_action"] == "remeasure_inputs")
    assert earlier["control_revision_applicability"]["status"] == "no_retained_coordinates"
    earlier_record = source_state["knowledge"]["action_experience"][earlier["context"]]["remeasure_inputs"]
    assert earlier_record["trials"] > 0 and earlier_record["expected_gain"] > 0
    assert any(item["action"] == "observation_research_feedback" and
               item["research_action"] == "remeasure_inputs" and item["after_loss"] <= 1e-7
               for item in unknown["actions"])

    assert later["status"] == "completed" and later["confirmation_loss"] <= 1e-7
    # A learned acquisition workflow can now solve the whole successor before
    # this controller is entered. Exercise the actual applicability decision
    # from the same paid prefix, with measurements still available to choose.
    agent, laboratory, paid, actions, _, result, _ = controller_from_state(
        source_state, measurement_budget=1536)
    with laboratory:
        current = chosen(actions)[0]
        assert current["control_revision_applicability"]["status"] == "ready"
        assert current["control_revision_applicability"]["ready_coordinates"] > 0
        assert current["context"] != earlier["context"]
        assert current["selected_action"] == "revise_mechanism"
        assert "remeasure_inputs" in current["eligible_actions"]
        assert laboratory.costs["measurement_cost"] == 0
        assert agent.library.action_experience[earlier["context"]]["remeasure_inputs"] == earlier_record
        prepared = [item for item in actions if item["action"] == "prepare_control_revision"]
        assert prepared and all(item["compute_cost"] > 0 for item in prepared)
        assert_fresh_solution(laboratory, result)


def controller_from_state(state, *, use_memory=True, compute_budget=360000, pilot_length=24,
                          measurement_budget=0):
    config = replace(CONFIG, use_memory=use_memory)
    agent = UnifiedResearchAgent(config, measurement_budget=measurement_budget,
        library=KnowledgeLibrary.from_dict(json.loads(json.dumps(state["knowledge"]))))
    agent._learn_instruments = agent._revise_inputs = True
    laboratory = DynamicInstrumentLab(2, SEED + 40028, SEED + 500009, "confirmation")
    compute, actions = SearchBudget(compute_budget), []
    paid = ActualPurchases(laboratory, compute, actions, measurement_budget=measurement_budget)
    for _ in range(2):
        paid.acquire((), "actual_command_pilot", rows_count=pilot_length)
    parent = {"candidate": Candidate("actual_previous_response",
        Program.from_dict(state["results"][-1]["research"]["program"]), "retained")}
    before = deepcopy(agent.library.to_dict())
    result = research_observation_actions(agent, laboratory, acquire=paid.acquire,
        paired_for=paid.paired_for, command_batches=paid.command_batches, budget=compute,
        actions=actions, transfer=[], graph=ResearchGraph(), remaining=paid.remaining,
        length=24, parent=parent, failed_probe=paid.batches[-1]["probe"])
    return agent, laboratory, paid, actions, compute, result, before


def test_disabling_memory_does_not_write_preparation_or_feedback_into_the_restored_book(stream):
    agent, laboratory, paid, actions, _, result, before = controller_from_state(stream[2], use_memory=False)
    assert_fresh_solution(laboratory, result)
    assert laboratory.costs["measurement_cost"] == 0
    assert chosen(actions)[0]["selected_action"] == "revise_mechanism"
    assert agent.library.action_experience == before["action_experience"]
    assert agent.library.program_experiment_methods.to_dict() == before["program_experiment_methods"]
    assert not any(item["action"] == "observation_research_feedback" for item in actions)


def test_no_compute_budget_cannot_claim_a_prepared_revision_or_buy_more_observations(stream):
    agent, laboratory, paid, actions, compute, result, before = controller_from_state(stream[2], compute_budget=0)
    assert not result["confirmed"] and result["selected"] is None
    assert compute.spent == 0
    assert laboratory.costs["observation_cost"] == 48
    assert laboratory.costs["measurement_cost"] == 0
    assert not chosen(actions)
    assert not any(item["action"] in {"prepare_control_revision", "observation_research_feedback"}
                   for item in actions)
    assert agent.library.action_experience == before["action_experience"]
