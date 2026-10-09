"""Successor methods repay full visits on newly purchased common responses."""

from copy import deepcopy
from dataclasses import replace
from math import ceil

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, SearchBudget
from hegel_machine.cognition.numeric_method_revisits import (
    METHOD_VISIT_ACTION, METHOD_RESPONSE_ACTION, method_response_applicability,
    offer_numeric_method_experiment, refresh_numeric_method_quote)
from hegel_machine.cognition.unified_continuation import paid_view
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_cognition_method_revisit_opportunity import UnifiedResearchSession

from test_cognition_method_revisit_opportunity import (
    CONFIG, TASKS, MethodLab, persisted, real_method_stream)


def test_whole_visit_and_screening_are_credited_once_from_the_same_fresh_response(real_method_stream):
    stream = real_method_stream
    offer, = stream["queued"]["pending_revisits"]
    previous = stream["queued"]["results"][0]["research"]
    report = stream["final"]["results"][0]["research"]
    outcome = report["continuation"]["method_visit_outcome"]
    paired = outcome["fresh_comparison"]
    assert paired["observation_rows"] == 24
    lab = MethodLab([], **TASKS[0])
    batches = paid_view(lab, report, paired["channels"])
    measured = [payload for payload in report["batches"]
                if set(paired["channels"]) <= set(payload["channels"])]
    batch = next(rows for payload, rows in zip(measured, batches)
                 if payload["probe"]["probe_id"] == paired["probe_id"])
    assert paired["before_loss"] == pytest.approx(loss(
        Program.from_dict(previous["program"]), batch, previous["fallback"], .5))
    assert paired["after_loss"] == pytest.approx(loss(
        Program.from_dict(report["program"]), batch, report["fallback"], .5))
    assert paired["after_loss"] < paired["before_loss"]
    assert paired["probe_id"] != offer["candidate"]["method_experiment"]["probe"]["probe_id"]
    costs = outcome["whole_visit_costs"]
    assert costs["compute_cost"] == (report["visit_costs"]["compute_cost"]
        + offer["screening_compute_cost"])
    experience = stream["final"]["knowledge"]["action_experience"][outcome["method_context"]][METHOD_VISIT_ACTION]
    assert experience["trials"] == 1
    for resource in ("compute", "observation", "measurement"):
        assert experience[f"mean_{resource}_cost"] == costs[f"{resource}_cost"]
        assert experience["execution_costs"][f"{resource}_cost"] == report["visit_costs"][f"{resource}_cost"]
    assert experience["execution_costs"]["visits"] == 1
    drafts = stream["final"]["knowledge"]["research_drafts"]
    assert all(outcome not in draft["outcomes"] for draft in drafts)
    assert outcome["response_confirmations"]
    for comparison in report["continuation"]["method_fresh_comparisons"]:
        if not comparison.get("attempted_source_draft_ids"):
            continue
        for source in comparison["attempted_source_draft_ids"]:
            draft = next(item for item in drafts if item["draft_id"] == source)
            evidence, = [item for item in draft["outcomes"]
                if item.get("kind") == "method_response_confirmation"
                and item["probe_id"] == comparison["probe_id"]]
            assert evidence["program"] == comparison["attempted_program"]
            assert evidence["after_loss"] == comparison["attempted_after_loss"]
            assert evidence["retained_loss"] == comparison["after_loss"]
            assert evidence["method_applicability"] == comparison["attempted_method_applicability"]
    resumed = UnifiedResearchSession(state=persisted(stream["final"]), lab_factory=stream["factory"])
    assert persisted(resumed.run()) == stream["final"]


def test_next_quote_separates_route_execution_cost_from_actual_response_structure(real_method_stream):
    stream = real_method_stream
    original_offer, = stream["queued"]["pending_revisits"]
    previous = stream["paused"]["results"][0]["research"]
    library = KnowledgeLibrary.from_dict(persisted(stream["final"]["knowledge"]))
    before = persisted(library.to_dict())
    budget = SearchBudget(40000)
    lab = MethodLab([], **TASKS[0])
    offer = offer_numeric_method_experiment(UnifiedResearchAgent(CONFIG, library=library,
        measurement_budget=320), lab, previous,
        {original_offer["candidate"]["research_draft_id"]}, budget)
    assert offer is not None
    opportunity = offer["candidate"]["method_opportunity"]
    completed = stream["final"]["results"][0]["research"]
    # Reprice the original executable route: its actual execution expense is
    # known, regardless of which fitted structure eventually earned a return.
    cached = deepcopy(original_offer)
    refresh_numeric_method_quote(cached, library.action_experience,
        dict(observation_cost=1000, measurement_cost=1000, compute_cost=1000000))
    assert cached["method_feedback"]["route_trials"] == 1
    expected_costs = {key: completed["visit_costs"][key]
        for key in ("observation_cost", "measurement_cost", "compute_cost")}
    assert cached["candidate"]["method_opportunity"]["observed_visit_costs"] == expected_costs
    assert cached["minimum_resources"]["compute_cost"] >= ceil(expected_costs["compute_cost"])
    assert cached["minimum_resources"]["compute_cost"] < original_offer["minimum_resources"]["compute_cost"]
    assert persisted(library.to_dict()) == before and not lab.purchases
    incumbent = {"candidate": Candidate("measured_old_response",
        Program.from_dict(previous["program"]), "retained_incumbent")}
    actual = Program.from_dict(offer["candidate"]["program"])
    assert method_response_applicability(actual, lab.spec_for(("meter",)), incumbent) == (
        opportunity["method_applicability"])
    matching = [actions[METHOD_RESPONSE_ACTION] for actions in library.action_experience.values()
        if METHOD_RESPONSE_ACTION in actions and actions[METHOD_RESPONSE_ACTION]["knowledge_applicability"]
        == opportunity["method_applicability"]]
    assert opportunity["observed_response_trials"] == sum(item["trials"] for item in matching)
    assert opportunity["observed_response_gain"] == (matching[0]["expected_gain"] if matching else None)

    renamed = MethodLab([], seed=4817, names=("u", "v", "w"), meter="reading")
    names = dict(zip(lab.spec_for(("meter",)).feature_names, renamed.spec_for(("reading",)).feature_names))

    def rename(program):
        return replace(program, name=names.get(program.name, program.name),
            children=tuple(rename(child) for child in program.children))

    assert method_response_applicability(rename(actual), renamed.spec_for(("reading",)), incumbent) == (
        opportunity["method_applicability"])
    refitted = Program("mul", children=(Program("const", value=2.3), actual))
    assert method_response_applicability(refitted, lab.spec_for(("meter",)), incumbent) == (
        opportunity["method_applicability"])

    temporal = Program("window_sum", lag=3, children=(actual,))
    assert method_response_applicability(temporal, lab.spec_for(("meter",)), incumbent) != (
        opportunity["method_applicability"])


@pytest.mark.parametrize("other_rows", [48, "tight"])
def test_priced_method_preserves_other_payable_research_in_the_visited_pool(real_method_stream, other_rows):
    stream = real_method_stream
    state = persisted(stream["queued"])
    offer, = state["pending_revisits"]
    # Another already visited task has an independently executable method
    # waiting. A method's first duration fit does not receive its reserve.
    other = deepcopy(offer)
    other["task_position"] = 1
    pool = UnifiedResearchSession(state=state)._remaining()
    if other_rows == "tight":
        other_rows = pool["observation_cost"] - offer["minimum_resources"]["observation_cost"]
    other["minimum_resources"] = {"observation_cost": other_rows,
        "measurement_cost": other_rows,
        "compute_cost": (pool["compute_cost"] - offer["minimum_resources"]["compute_cost"]) // 2}
    state["pending_revisits"] = [other]
    session = UnifiedResearchSession(state=state, lab_factory=stream["factory"])
    remaining = session._remaining()
    assert all(remaining[key] >= offer["minimum_resources"][key] + other["minimum_resources"][key]
               for key in remaining)
    assert session._perform_revisit(offer)
    actual, = session.state["revisits"]
    previous = stream["queued"]["results"][0]["research"]
    for resource in remaining:
        allocated = actual["resource_limits"][resource] - previous["costs"][resource]
        assert offer["minimum_resources"][resource] <= allocated <= (
            remaining[resource] - other["minimum_resources"][resource])
        assert session._remaining()[resource] >= other["minimum_resources"][resource]
