"""A paid successor experiment finishes its frozen response comparison first."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from test_cognition_method_revisit_opportunity import CONFIG, real_method_stream, UnifiedResearchSession


def test_priced_method_completes_new_confirmation_before_another_research_purchase(real_method_stream):
    stream = real_method_stream
    state = deepcopy(stream["queued"])
    offer, = state["pending_revisits"]
    # This is the actual small learned allocation that previously lost its
    # confirmation to further condition research in the observed study.
    offer["minimum_resources"]["compute_cost"] = 123528
    previous = deepcopy(state["results"][0]["research"])
    session = UnifiedResearchSession(state=state, lab_factory=stream["factory"])
    purchase_start = len(stream["purchases"])
    assert session._perform_revisit(offer, allocation_limits={
        "observation_cost": 72, "measurement_cost": 72, "compute_cost": 123528})
    report = session.state["results"][0]["research"]
    actions = report["actions"][len(previous["actions"]):]
    acquired = [item for item in actions if item["action"] == "acquire"]
    assert [item["purpose"] for item in acquired[:3]] == [
        "successor_partial_method_experiment", "compare_continuation_candidates",
        "confirm_continuation_program"]
    paired = report["continuation"]["method_visit_outcome"]["fresh_comparison"]
    assert paired is not None
    purchases = stream["purchases"][purchase_start:]
    measured = next(item for item in purchases if item["probe"].probe_id == paired["probe_id"])
    assert paired["probe_id"] not in {item["probe_id"] for item in acquired[:2]}
    assert paired["before_loss"] == pytest.approx(loss(
        Program.from_dict(previous["program"]), measured["rows"], previous["fallback"], .5))
    assert paired["after_loss"] == pytest.approx(loss(
        Program.from_dict(report["program"]), measured["rows"], report["fallback"], .5))
    assert paired["after_loss"] < paired["before_loss"]
    assert report["visit_costs"]["observation_cost"] == sum(
        item["observation_cost"] for item in purchases) == 72
    assert report["visit_costs"]["compute_cost"] <= 123528


def test_method_fit_without_payable_fresh_response_keeps_the_measured_incumbent(real_method_stream):
    stream = real_method_stream
    state = deepcopy(stream["queued"])
    offer, = state["pending_revisits"]
    offer["minimum_resources"].update(observation_cost=48, measurement_cost=48,
                                     compute_cost=123528)
    previous = deepcopy(state["results"][0]["research"])
    session = UnifiedResearchSession(state=state, lab_factory=stream["factory"])
    assert session._perform_revisit(offer, allocation_limits={
        "observation_cost": 48, "measurement_cost": 48, "compute_cost": 123528})
    report = session.state["results"][0]["research"]
    assert report["program"] == previous["program"]
    assert report["status"] == "unresolved"
    assert report["continuation"]["method_visit_outcome"]["fresh_comparison"] is None
    assert report["visit_costs"]["observation_cost"] <= 48
    assert report["visit_costs"]["compute_cost"] <= 123528


def test_method_confirmation_purchases_the_unresolved_cut_distinction(real_method_stream):
    stream = real_method_stream
    old = stream["queued"]["results"][0]["research"]
    report = stream["final"]["results"][0]["research"]
    actions = report["actions"][len(old["actions"]):]
    plan = next(item for item in actions if item["action"] == "plan_condition_confirmation")
    assert plan["expected_disagreement"] > CONFIG.minimum_improvement
    assert plan["planning_compute_cost"] > 0
    allocation = next(item for item in actions if item["action"] == "allocate_condition_confirmation")
    confirmation = next(item for item in actions
        if item["action"] == "validate_continued_program" and item["probe_id"] == allocation["probe_ids"][-1])
    purchase = next(item for item in stream["purchases"][stream["before"]:]
        if item["probe"].probe_id == confirmation["probe_id"])
    # The planner chooses public commands; the meter is purchased, not set.
    controls = set(old["batches"][0]["probe"]["sequence"][0])
    assert all(set(point) == controls for point in purchase["probe"].sequence)
    assert all("meter" in row.features for row in purchase["rows"])
    assert set(allocation["probe_ids"]).isdisjoint(batch["probe"]["probe_id"] for batch in old["batches"])
    assert confirmation["loss"] == pytest.approx(loss(
        Program.from_dict(confirmation["program"]), purchase["rows"], report["fallback"], .5))
    assert not confirmation["accepted"]
    assert report["status"] == "unresolved"


def test_declined_condition_experiment_does_not_buy_the_same_confirmation_again(
        real_method_stream, monkeypatch):
    from hegel_machine.cognition import condition_response_actions

    stream = real_method_stream
    state = deepcopy(stream["queued"])
    offer, = state["pending_revisits"]
    offer["minimum_resources"].update(observation_cost=96, measurement_cost=96,
                                     compute_cost=123528)
    previous = deepcopy(state["results"][0]["research"])
    decisions = []

    def decline(agent, spec, selected, training, validation, finalists, *args, **kwargs):
        decisions.append(selected["candidate"].program)
        return selected, training, finalists

    # Isolate the decision to retain the unresolved cut; the method experiment,
    # response comparison, and confirmation all still use actual lab targets.
    monkeypatch.setattr(condition_response_actions,
                        "research_observed_condition_competition", decline)
    session = UnifiedResearchSession(state=state, lab_factory=stream["factory"])
    assert session._perform_revisit(offer, allocation_limits={
        "observation_cost": 96, "measurement_cost": 96, "compute_cost": 123528})
    report = session.state["results"][0]["research"]
    actions = report["actions"][len(previous["actions"]):]
    assert len(decisions) == 1
    assert [item["purpose"] for item in actions if item["action"] == "acquire"] == [
        "successor_partial_method_experiment", "compare_continuation_candidates",
        "confirm_continuation_program"]
    assert report["condition_uncertainty"]["status"] == "unresolved"
    assert report["status"] == "unresolved"
    assert any(item["action"] == "stop_condition_followup_research" for item in actions)
    assert report["visit_costs"]["observation_cost"] == 72


def test_real_method_miss_stops_unproductive_extra_batches_without_claiming_support(real_method_stream):
    stream = real_method_stream
    before = stream["queued"]["results"][0]["research"]
    after = stream["final"]["results"][0]["research"]
    actions = after["actions"][len(before["actions"]):]
    plan = next(item for item in actions if item["action"] == "plan_condition_confirmation")
    allocation = next(item for item in actions if item["action"] == "allocate_condition_confirmation")
    batches = [batch for batch in after["batches"]
               if batch["probe"]["probe_id"] in allocation["probe_ids"]]
    assert len(batches) == 1
    assert len({batch["probe"]["probe_id"] for batch in batches}) == len(batches)
    assert all(batch["probe"]["sequence"] == batches[0]["probe"]["sequence"] for batch in batches)
    left, right = plan["threshold_interval"]
    assert not any(left < row["features"][plan["condition_field"]] < right
                   for batch in batches for row in batch["observations"])
    assert allocation["acquired_rows"] < allocation["planned_rows"]
    assert allocation["acquired_rows"] == sum(
        len(batch["observations"]) for batch in batches)
    assert not any(outcome["informative"] for outcome in allocation["outcomes"])
    assert allocation["stop_reason"] == "retain_question_for_more_valuable_future_evidence"
    decision = allocation["decisions"][-1]
    assert not decision["buy"]
    assert decision["expected_marginal_value"] <= decision["resource_share"]
    assert decision["empirical_marginal_prediction_gain"] == 0.
    assert after["status"] == after["condition_uncertainty"]["status"] == "unresolved"
    assert not after["concepts_added"]
    confirmation = next(item for item in actions if item["action"] == "validate_continued_program")
    assert confirmation["probe_id"] == allocation["probe_ids"][-1]
    assert confirmation["program"] == allocation["program"]
    assert not confirmation["accepted"]
    visit, = stream["final"]["revisits"]
    for resource in ("observation_cost", "measurement_cost", "compute_cost"):
        assert after["costs"][resource] <= visit["resource_limits"][resource]
    unused_observations = visit["resource_limits"]["observation_cost"] - after["costs"]["observation_cost"]
    assert unused_observations >= allocation["confirmation_reserve"]
