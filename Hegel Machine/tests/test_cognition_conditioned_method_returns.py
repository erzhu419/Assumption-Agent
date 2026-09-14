"""Observed method outcomes apply to evidence conditions and actual revisit costs."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.action_policy import ActionOption
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.method_conditions import method_profile, method_applicability, same_research_condition
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.research_methods import ResearchMethods
from test_cognition_method_returns import method_evidence, measured_step
from test_cognition_method_conditions import spec, rows


def boundary(width, *, depth=1, previous=False):
    return {"boundary_support": [{"coordinate": {"u": 1.}, "history_depth": depth,
                "previous_state": previous, "observed_interval": [-width, width],
                "relative_width": width, "free_controls": []}],
            "unresolved_boundary_width": width}


def test_equal_marginal_controls_do_not_equate_different_observed_state_coverage():
    def observations(swap):
        return [Observation({"u": value}, float(value > 0),
                            ({"u": 0., "__target__": float((value > 0) != swap)},))
                for value in (-.8, -.3, .3, .8)]
    source = method_profile(spec(), observations(False), Program("feature", name="u"), residual_loss=.2)
    target = method_profile(spec(), observations(True), Program("feature", name="u"), residual_loss=.2)
    assert source["data"]["coverage"] == target["data"]["coverage"]
    assert source["data"]["conditional_coverage"] != target["data"]["conditional_coverage"]
    restored = json.loads(json.dumps(source))
    assert method_applicability(restored, target, "transition")["status"] == "explore"
    assert not same_research_condition(restored, target, "transition")


def test_paid_boundary_contraction_reopens_a_previously_unproductive_action():
    arguments = dict(spec=spec(), observations=rows(history=True, binary=True),
                     incumbent=Program("feature", name="u"), residual_loss=.1)
    before = method_profile(**arguments, boundary_evidence=boundary(.2))
    pending = [{"action": "transition", "gain": 0., "profile": before}]
    unchanged = method_profile(**arguments, trace=pending, boundary_evidence=boundary(.2))
    contracted = method_profile(**arguments, trace=pending, boundary_evidence=boundary(.05))
    assert before["data"]["coverage"] == contracted["data"]["coverage"]
    assert method_applicability(before, unchanged, "transition")["status"] == "unsupported"
    assert method_applicability(before, contracted, "transition")["can_try"]
    # Moving to another prior response is a new transfer hypothesis, even
    # though the same public control and the same total rows were acquired.
    other = method_profile(**arguments, boundary_evidence=boundary(.2, previous=True))
    assert method_applicability(before, other, "transition")["status"] == "explore"


def test_a_neutral_different_condition_does_not_erase_the_method_useful_condition():
    task, observations, source, context, condition = method_evidence()
    book = ResearchMethods()
    identifier = book.observe(context, [measured_step("representation", .8, context, source)],
                              condition=condition, source_task="source")
    book.observe(context, [measured_step("representation", .6, context, source, method_id=identifier)],
                 condition=condition, source_task="useful_transfer")
    changed = method_profile(task.spec, observations,
        Program("mul", children=(Program("feature", name="x"), Program("feature", name="x"))),
        residual_loss=1., baseline_loss=2.)
    book.observe(context, [measured_step("representation", 0., context, changed, method_id=identifier)],
                 condition=condition, source_task="unhelpful_transfer")
    assert book.methods[0]["observed_gain"] == 0.
    restored = ResearchMethods(json.loads(json.dumps(book.to_dict())))
    options = [ActionOption("refit", 0, 100), ActionOption("representation", 8, 600)]
    arguments = dict(task_id="new_task", observation_remaining=100, compute_remaining=5000, ordinary=options[0])
    assert restored.choose(context, condition, options, [], profile=source, **arguments) == options[1]
    assert restored.last_choice["changes_action"]
    assert restored.last_choice["return_estimate"]["gain"] == pytest.approx(.6)
    assert restored.choose(context, condition, options, [], profile=changed, **arguments) is None


def test_conditioned_return_must_repay_cost_before_replacing_a_better_ordinary_action():
    _, _, profile, context, condition = method_evidence()
    book = ResearchMethods()
    identifier = book.observe(context, [measured_step("representation", .8, context, profile)],
                              condition=condition, source_task="source")
    book.observe(context, [measured_step("representation", .6, context, profile, method_id=identifier)],
                 condition=condition, source_task="transfer")
    options = [ActionOption("refit", 0, 100), ActionOption("representation", 8, 600)]
    arguments = dict(task_id="new", profile=profile, observation_remaining=100, compute_remaining=5000,
                     ordinary=options[0])
    assert book.choose(context, condition, options, [], ordinary_evidence={
        "expected_gain": .1, "dominant_resource_share": .1}, **arguments) == options[1]
    assert book.choose(context, condition, options, [], ordinary_evidence={
        "expected_gain": .9, "dominant_resource_share": .01}, **arguments) is None


@pytest.mark.parametrize("gain", [0., -.1, .3])
def test_complete_continuation_return_replaces_source_promise_and_prices_whole_visit(gain):
    _, _, profile, context, condition = method_evidence()
    book = ResearchMethods()
    identifier = book.observe(context, [measured_step("representation", .9, context, profile)],
                              condition=condition, source_task="source")
    feedback = {"normalized_gain": gain, "gain": gain * 2., "reference_rows": 16,
                "observation_cost": 24, "compute_cost": 2400}
    saved = deepcopy(feedback)
    result = book.observe_continuation(identifier, profile, feedback, source_task="continued_task")
    assert feedback == saved
    assert result["gain"] == gain and result["raw_gain"] == gain * 2.
    restored = ResearchMethods(json.loads(json.dumps(book.to_dict())))
    limits = dict(task_id="another", profile=profile, observation_remaining=100, compute_remaining=10000,
                  visit_observation_budget=32, visit_compute_budget=3000)
    opportunity = restored.opportunity(context, condition, **limits)
    if gain <= 0:
        assert opportunity is None
    else:
        assert opportunity["expected_gain"] == gain
        assert opportunity["compute_cost"] == 2400 and opportunity["observation_cost"] == 24
        assert opportunity["return_estimate"]["basis"] == "measured_continuation_return"
        assert restored.opportunity(context, condition, **{**limits, "visit_compute_budget": 2399}) is None


def test_supported_method_cannot_replace_a_live_boundary_with_undirected_coverage():
    _, _, profile, context, condition = method_evidence()
    book = ResearchMethods()
    identifier = book.observe(context, [measured_step("representative", .9, context, profile)],
                              condition=condition, source_task="source")
    book.observe(context, [measured_step("representative", .7, context, profile, method_id=identifier)],
                 condition=condition, source_task="transfer")
    coverage = ActionOption("representative", 8, 8, (None, {}))
    bracket = ActionOption("transition", 8, 8, (None, boundary(.2)))
    assert book.choose(context, condition, [coverage, bracket], [], ordinary=bracket,
        task_id="new", profile=profile, observation_remaining=100, compute_remaining=10000) is None


def test_unfinished_execution_does_not_supply_successful_method_evidence():
    _, _, profile, context, condition = method_evidence()
    book = ResearchMethods()
    result = book.observe(context, [measured_step("transition", .8, context, profile, complete=False)],
                          condition=condition, source_task="source")
    assert result is None and book.methods == []


def test_successful_transfer_teaches_a_new_coverage_condition_after_restore():
    task, _, source, context, condition = method_evidence()
    learned_rows = [Observation({"x": x}, x + .25) for x in (-.4, -.2, .2, .4)]
    target = method_profile(task.spec, learned_rows, Program("const", value=0.),
                            residual_loss=1., baseline_loss=2.)
    assert method_applicability(source, target, "representation")["status"] == "explore"
    book = ResearchMethods()
    identifier = book.observe(context, [measured_step("representation", .8, context, source)],
                              condition=condition, source_task="source")
    book.observe(context, [measured_step("representation", .6, context, target, method_id=identifier)],
                 condition=condition, source_task="new_condition")
    restored = ResearchMethods(json.loads(json.dumps(book.to_dict())))
    options = [ActionOption("refit", 0, 100), ActionOption("representation", 8, 600)]
    assert restored.choose(context, condition, options, [], ordinary=options[0], task_id="renamed_target",
        profile=target, observation_remaining=100, compute_remaining=5000) == options[1]
    assert restored.last_choice["applicability"]["status"] == "supported"
    assert restored.last_choice["return_estimate"]["gain"] == .6


def test_next_step_quote_keeps_the_return_of_its_remaining_sequence():
    _, _, profile, context, condition = method_evidence()
    book = ResearchMethods()
    book.observe(context, [measured_step("refit", .1, context, profile)],
                 condition=condition, source_task="short_source")
    sequence = book.observe(context, [measured_step("refit", .1, context, profile),
        measured_step("representation", .9, context, profile)], condition=condition, source_task="long_source")
    quote = book.opportunity(context, condition, profile=profile, task_id="target",
                             observation_remaining=100, compute_remaining=10000)
    assert quote["method_id"] == sequence
    assert quote["expected_gain"] == .9
    assert quote["compute_cost"] == 600  # Begin the first step, not repeat the whole sequence.
    assert quote["return_estimate"]["basis"] == "remaining_sequence_return"


def test_neutral_matching_transfer_is_not_averaged_with_the_original_positive_promise():
    _, _, profile, context, condition = method_evidence()
    book = ResearchMethods()
    identifier = book.observe(context, [measured_step("representation", .9, context, profile)],
                              condition=condition, source_task="source")
    book.observe(context, [measured_step("representation", 0., context, profile, method_id=identifier)],
                 condition=condition, source_task="actual_transfer")
    restored = ResearchMethods(json.loads(json.dumps(book.to_dict())))
    options = [ActionOption("representation", 8, 600)]
    assert restored.choose(context, condition, options, [], ordinary=options[0], task_id="another",
        profile=profile, observation_remaining=100, compute_remaining=10000) is None
    assert restored.opportunity(context, condition, task_id="another", profile=profile,
        observation_remaining=100, compute_remaining=10000) is None
