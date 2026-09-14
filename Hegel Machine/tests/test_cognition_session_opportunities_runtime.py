"""The selected research route executes and earns its own common-row return."""

from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.research_allocation_demo import run_method_allocation
from hegel_machine.cognition.session import ResearchSession
from test_cognition_experiment_frontier_runtime import _saved_question


def test_selected_learned_method_executes_without_spending_or_rewarding_the_saved_experiment(monkeypatch):
    library = KnowledgeLibrary()
    learned = run_method_allocation(library)
    choice = deepcopy(learned["summary"]["method_opportunity"])
    assert choice["expected_gain"] > 0.
    identifier = choice["method_id"]
    assert any(step["method_id"] == identifier and step["gain"] > 0.
               for step in learned["summary"]["measured_method_returns"])

    session, environment, workspace = _saved_question()
    session.config = replace(session.config, use_memory=True,
                             learn_research_policy=True, use_abstractions=False)
    session.state["config"] = asdict(session.config)
    session.library = library
    session.limits["max_visits"] = 4
    original_records = deepcopy(library.research_opportunities.outcomes)

    def recommend(*args, trace=(), **kwargs):
        # Supply an actually learned positive method to isolate the scheduler's
        # competition from its separate numeric-to-binary applicability rule.
        # The selected representation step and every apparatus query run live.
        if any(step.get("method_id") == identifier for step in trace):
            return None
        return deepcopy(choice)

    monkeypatch.setattr(library.methods, "opportunity", recommend)
    pending = session.to_dict()["tasks"][0]
    assert [offer["kind"] for offer in pending["research_offers"]] == ["method", "experiment"]
    assert pending["next_method"]["method_id"] == identifier
    assert pending["next_experiment"] is None
    result = session.run(max_steps=1)
    episode = result["episodes"][0]
    executed = [action["details"] for action in episode["actions"]
                if action["action"] == "execute_method_step"]
    assert any(step["method_id"] == identifier and step["name"] == "representation"
               for step in executed)
    assert not any(action["action"] == "execute_research_question" for action in episode["actions"])
    assert workspace.experiment_frontier["probe"]["probe_id"] not in {
        probe.probe_id for probe in environment.queries}
    trace = result["workspaces"]["0"]["method_trace"]
    assert any(step.get("method_id") == identifier and step["execution_complete"]
               and step["compute_cost"] > 0. for step in trace)

    records = result["knowledge"]["research_opportunities"]["outcomes"]
    assert records[:len(original_records)] == original_records
    added = records[len(original_records):]
    assert len(added) == 1 and added[0]["condition"]["kind"] == "method"
    feedback = next(action["details"] for action in episode["actions"]
                    if action["action"] == "continuation_feedback")
    assert added[0]["execution_complete"] and added[0]["reference_rows"] > 0
    assert added[0]["gain"] == max(-1., min(1., feedback["normalized_gain"]))
    assert added[0]["reported_normalized_gain"] == feedback["normalized_gain"]
    assert added[0]["raw_gain"] == feedback["gain"]
    assert added[0]["compute_cost"] == episode["compute_cost"]
    assert added[0]["observation_cost"] == sum(probe.cost for probe in environment.queries)


def test_actual_zero_return_survives_json_and_keeps_the_question_without_repeating_its_experiment():
    session, environment, workspace = _saved_question()
    session.limits["max_visits"] = 4
    completed = session.run(max_steps=1)
    episode = completed["episodes"][0]
    record = episode["scheduler"]["research_return"]
    assert record["condition"]["kind"] == "experiment"
    assert record["gain"] == 0. and record["reference_rows"] > 0
    assert record["execution_complete"]
    question = completed["workspaces"]["0"]["experiment_frontier"]
    assert 0. < question["uncertainty"] < workspace.experiment_frontier["uncertainty"]
    assert completed["tasks"][0]["unresolved"]

    # Give ample resources and unused visits: the measured return, not a spent
    # quota or a visit cap, must suppress the next same-support experiment.
    saved = json.loads(json.dumps(completed))
    restored = ResearchSession(session.config, [environment], state=saved,
        observation_budget=completed["total_observation_cost"] + 128,
        compute_budget=completed["total_compute_cost"] + 80000, max_visits=5)
    before_queries = len(environment.queries)
    resumed = restored.run(max_steps=1)
    assert resumed["pending"] == [] and len(resumed["episodes"]) == 1
    assert len(environment.queries) == before_queries
    assert resumed["total_compute_cost"] == completed["total_compute_cost"]
    assert resumed["workspaces"]["0"]["experiment_frontier"] == saved["workspaces"]["0"]["experiment_frontier"]
    assert resumed["knowledge"]["research_opportunities"] == completed["knowledge"]["research_opportunities"]
    assert resumed["tasks"][0]["revisit_basis"] == "no_observed_continuation_gain"


@pytest.mark.parametrize("short_resource", ["observation", "compute"])
def test_unfunded_valid_experiment_remains_pending_and_executes_after_resources_are_added(short_resource):
    session, environment, workspace = _saved_question()
    before = session.to_dict()
    quote = before["tasks"][0]["next_research"]
    limits = {"observation_budget": before["total_observation_cost"] + 32,
              "compute_budget": before["total_compute_cost"] + 18000}
    limits[f"{short_resource}_budget"] = (before[f"total_{short_resource}_cost"]
                                          + quote[f"{short_resource}_cost"] - 1)
    unfunded = ResearchSession(session.config, [environment], state=json.loads(json.dumps(before)),
                                max_visits=4, **limits)
    stopped = unfunded.run(max_steps=1)
    assert stopped["status"] == "budget_exhausted" and stopped["pending"] == [0]
    assert stopped["tasks"][0]["revisit_basis"] == "unfunded_research_opportunity"
    assert not environment.queries and not stopped["episodes"]
    assert not stopped["knowledge"]["research_opportunities"]["outcomes"]
    assert stopped["workspaces"]["0"]["experiment_frontier"] == json.loads(json.dumps(workspace.experiment_frontier))

    funded = ResearchSession(session.config, [environment], state=json.loads(json.dumps(stopped)),
        observation_budget=before["total_observation_cost"] + 32,
        compute_budget=before["total_compute_cost"] + 18000, max_visits=4)
    completed = funded.run(max_steps=1)
    assert len(completed["episodes"]) == 1
    assert environment.queries[0].probe_id == workspace.experiment_frontier["probe"]["probe_id"]
    assert completed["episodes"][0]["scheduler"]["research_return"]["execution_complete"]
