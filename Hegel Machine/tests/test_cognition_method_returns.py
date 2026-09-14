"""Method promises must survive actual target costs and common new evidence."""

import json

import pytest

from hegel_machine.cognition.action_policy import ActionOption, ResearchActionPolicy
from hegel_machine.cognition.controller import CognitiveAgent
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.experiment_knowledge import research_context
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.method_conditions import method_profile
from hegel_machine.cognition.models import Candidate, Probe, ResearchConfig
from hegel_machine.cognition.research_methods import ResearchMethods
from hegel_machine.cognition.session import ResearchSession
from hegel_machine.cognition.workspace import ResearchWorkspace
from test_cognition_session import MeasuredTask, config as session_config


def method_evidence():
    task = MeasuredTask("source")
    observations = task.observe(task.spec.initial_probes[0])
    incumbent = Program("const", value=0.)
    profile = method_profile(task.spec, observations, incumbent, residual_loss=1., baseline_loss=2.)
    context = ResearchActionPolicy(observations, task.spec, {}).set_condition("model_residual")
    return task, observations, profile, context, research_context(task.spec, observations)


def measured_step(action, gain, context, profile, *, method_id=None, complete=True,
                  compute_cost=600, observation_cost=8):
    return {"action": action, "gain": gain, "context": context, "profile": profile,
            "method_id": method_id, "execution_complete": complete,
            "compute_cost": compute_cost, "observation_cost": observation_cost}


def test_opportunity_caps_source_gain_by_target_headroom_and_prices_the_next_visit():
    task, observations, profile, context, condition = method_evidence()
    book = ResearchMethods()
    book.observe(context, [measured_step("representation", .9, context, profile)],
                 condition=condition, source_task="source")
    target = method_profile(task.spec, [*observations, *observations], Program("const", value=0.),
                            residual_loss=1., baseline_loss=2.)
    limits = {"observation_remaining": 1000, "compute_remaining": 200000,
              "task_id": "target", "profile": target, "maximum_gain": .03,
              "visit_observation_budget": 16, "visit_compute_budget": 2000}
    opportunity = book.opportunity(context, condition, **limits)
    assert opportunity["applicability"]["status"] == "supported"
    assert opportunity["source_gain"] == .9
    assert opportunity["expected_gain"] == .03
    assert opportunity["compute_cost"] == 1200  # Twice as many actual fitted rows.
    share = max(opportunity["observation_cost"] / 1000, opportunity["compute_cost"] / 200000)
    assert opportunity["value"] == pytest.approx(.03 / share)
    assert book.opportunity(context, condition, **{**limits, "visit_compute_budget": 1199}) is None
    assert book.opportunity(context, condition, **{**limits, "visit_observation_budget": 7}) is None
    assert book.opportunity(context, condition, **{**limits, "maximum_gain": 0.}) is None


def test_opportunity_buys_an_executable_next_step_not_the_entire_method_again():
    _, _, profile, context, condition = method_evidence()
    book = ResearchMethods()
    identifier = book.observe(context, [
        measured_step("representative", .1, context, profile, compute_cost=100, observation_cost=12),
        measured_step("representation", .7, context, profile, compute_cost=500, observation_cost=0)],
        condition=condition, source_task="source")
    trace = [measured_step("representative", 0., context, profile, method_id=identifier,
                           compute_cost=100, observation_cost=12)]
    opportunity = book.opportunity(context, condition, task_id="target", trace=trace, profile=profile,
        observation_remaining=20, compute_remaining=2000, maximum_gain=.2,
        visit_observation_budget=4, visit_compute_budget=500)
    assert opportunity["compute_cost"] == 500
    assert opportunity["observation_cost"] == 0
    assert opportunity["expected_gain"] == .2


def test_realized_target_cost_is_not_scaled_twice_on_the_next_transfer():
    task, observations, source, context, condition = method_evidence()
    book = ResearchMethods()
    identifier = book.observe(context, [measured_step("representation", .9, context, source)],
                              condition=condition, source_task="source")
    target = method_profile(task.spec, [*observations, *observations], Program("const", value=0.),
                            residual_loss=1., baseline_loss=2.)
    book.observe(context, [measured_step("representation", .8, context, target,
        method_id=identifier, compute_cost=1200)], condition=condition, source_task="target")
    restored = ResearchMethods(json.loads(json.dumps(book.to_dict())))
    quote = restored.opportunity(context, condition, task_id="another", profile=target,
        observation_remaining=100, compute_remaining=10000, visit_compute_budget=1200)
    assert quote is not None
    assert quote["compute_cost"] == 1200


def test_a_source_only_method_does_not_replace_a_different_target_plan():
    _, _, profile, context, condition = method_evidence()
    book = ResearchMethods()
    book.observe(context, [measured_step("representation", .9, context, profile)],
                 condition=condition, source_task="source")
    refit = ActionOption("refit", 0, 100)
    representation = ActionOption("representation", 0, 600)
    arguments = dict(task_id="target", profile=profile, observation_remaining=100, compute_remaining=5000)
    assert book.choose(context, condition, [refit, representation], [], ordinary=refit, **arguments) is None
    # It still earns a transfer trial when the target independently selects
    # this action; source evidence has not been discarded or disabled.
    assert book.choose(context, condition, [refit, representation], [],
                       ordinary=representation, **arguments) == representation


def test_a_supported_coverage_method_preserves_a_current_distinguishing_probe():
    _, _, profile, context, condition = method_evidence()
    book = ResearchMethods()
    identifier = book.observe(context, [measured_step("representative", .9, context, profile)],
                              condition=condition, source_task="source")
    book.observe(context, [measured_step("representative", .5, context, profile, method_id=identifier)],
                 condition=condition, source_task="earlier_target")
    coverage = ActionOption("representative", 8, 8, (None, {}))
    distinguishing = ActionOption("task_value", 8, 8, (None, {"expected_risk_reduction": .1}))
    arguments = dict(task_id="new_target", profile=profile, observation_remaining=100, compute_remaining=5000)
    assert book.choose(context, condition, [coverage, distinguishing], [],
                       ordinary=distinguishing, **arguments) is None
    assert book.choose(context, condition, [coverage, distinguishing], [],
                       ordinary=coverage, **arguments) == coverage


@pytest.mark.parametrize("execution_complete", [True, False])
def test_another_method_id_cannot_resell_a_complete_neutral_action(execution_complete):
    _, _, profile, context, condition = method_evidence()
    book = ResearchMethods()
    first = book.observe(context, [measured_step("representation", .9, context, profile)],
                         condition=condition, source_task="source_a")
    other = book.observe(context, [measured_step("representation", .8, context, profile),
                                  measured_step("representative", .1, context, profile)],
                         condition=condition, source_task="source_b")
    assert first != other
    trace = [measured_step("representation", 0., context, profile, method_id=first, complete=execution_complete)]
    restored = ResearchMethods(json.loads(json.dumps(book.to_dict())))
    options = [ActionOption("representation", 0, 600)]
    chosen = restored.choose(context, condition, options, trace, task_id="target", profile=profile,
                              observation_remaining=100, compute_remaining=5000)
    opportunity = restored.opportunity(context, condition, trace=trace, profile=profile, task_id="target",
                                        observation_remaining=100, compute_remaining=5000, maximum_gain=.2)
    if execution_complete:
        assert chosen is None
        assert opportunity is None
    else:
        assert chosen.name == "representation"
        assert restored.last_choice["method_id"] == other
        assert opportunity["method_id"] == other


@pytest.mark.parametrize("correct_incumbent", [True, False])
def test_real_continuation_compares_saved_and_final_program_on_the_same_new_slice(correct_incumbent):
    task = MeasuredTask("same_slice")
    training_probe = task.spec.initial_probes[0]
    reference_probe = Probe("old_reference", tuple({"x": value} for value in (-.9, -.2, .2, .9)))
    training, reference = task.observe(training_probe), task.observe(reference_probe)
    before = (Program("add", children=(Program("feature", name="x"), Program("const", value=.25)))
              if correct_incumbent else Program("const", value=.25))
    candidate = Candidate("saved", before, "refit")
    fallback = .25
    initial_loss = loss(Program("const", value=fallback), training, fallback, .5)
    workspace = ResearchWorkspace(task.spec.task_id, observations=training, reference=reference,
        queried=[training_probe, reference_probe], incumbent=candidate, evidence_champion=candidate,
        bank=[candidate], fallback=fallback, initial_loss=initial_loss)
    config = ResearchConfig(observation_budget=24, compute_budget=5000, max_rounds=1,
                            seed=43, use_memory=False, learn_research_policy=False)
    saved = json.loads(json.dumps(workspace.to_dict()))
    episode = CognitiveAgent(config).run(task, research_state=saved)
    feedback = next(action for action in episode.actions if action.action == "continuation_feedback")
    restored = ResearchWorkspace.from_dict(episode.research_state)
    new_reference = restored.reference[len(reference):]
    assert new_reference and feedback.details["reference_rows"] == len(new_reference)
    assert feedback.compute_cost == 2 * len(new_reference)
    expected_before = loss(before, new_reference, fallback, .5)
    expected_after = loss(episode.program, new_reference, fallback, .5)
    assert feedback.details["before_loss"] == pytest.approx(expected_before, abs=1e-15)
    assert feedback.details["after_loss"] == pytest.approx(expected_after, abs=1e-15)
    assert feedback.details["gain"] == pytest.approx(expected_before - expected_after, abs=1e-15)
    assert task.observed == len(training) + len(reference) + episode.observation_cost
    if correct_incumbent:
        assert episode.program == before
        assert feedback.details["program_changed"] is False
        assert feedback.details["gain"] == 0.
    else:
        assert episode.program != before
        assert feedback.details["program_changed"] is True
        assert feedback.details["gain"] > .1


def test_complete_unproductive_revisit_stops_old_residual_until_new_evidence_arrives():
    config = session_config(use_memory=False, learn_research_policy=False)
    task = MeasuredTask("hard", nonlinear=True)
    limits = {"observation_budget": 128, "compute_budget": 24000, "max_visits": 4}
    session = ResearchSession(config, [task], **limits)
    finished = session.run()
    current = finished["tasks"][0]
    assert finished["completed_visits"] == 2
    assert current["continuation_return"]["gain"] <= 0.
    assert current["validation_loss"] > config.minimum_improvement
    assert current["normalized_excess_loss"] > 1.
    assert current["revisit_basis"] == "no_observed_continuation_gain"
    # Larger stale loss estimates cannot manufacture a third measured return.
    snapshot = json.loads(json.dumps(finished))
    snapshot["tasks"][0].update(validation_loss=100., normalized_excess_loss=100.)
    restored = ResearchSession(config, [task], state=snapshot, **limits)
    observed = task.observed
    stable = restored.run()
    assert stable["completed_visits"] == 2
    assert task.observed == observed
    probe = Probe("external_followup", tuple({"x": value} for value in (-.75, -.15, .15, .75)))
    newly_measured = task.observe(probe)
    restored.add_evidence(0, probe, newly_measured)
    continued = restored.run(max_steps=1)
    assert continued["completed_visits"] == 3
    assert continued["episodes"][-1]["scheduler"]["reason"] == "new_evidence"
    assert continued["total_observation_cost"] > stable["total_observation_cost"] + probe.cost
    assert continued["total_observation_cost"] <= limits["observation_budget"]
    assert continued["total_compute_cost"] <= limits["compute_budget"]


def test_method_selection_prices_the_executable_visit_before_comparing_returns():
    _, _, profile, context, condition = method_evidence()
    book = ResearchMethods()
    expensive = book.observe(context, [measured_step("representation", .9, context, profile)],
                             condition=condition, source_task="representation_source")
    affordable = book.observe(context, [measured_step("refit", .4, context, profile)],
                              condition=condition, source_task="refit_source")
    calls = []

    def whole_visit(method, estimate):
        calls.append(method["method_id"])
        return {"observation_budget": 16, "compute_budget": 7000 if method["method_id"] == expensive else 2200,
                "max_rounds": 1}

    selected = book.opportunity(context, condition, task_id="target", profile=profile,
        observation_remaining=100, compute_remaining=5000, visit_quote=whole_visit)
    assert selected["method_id"] == affordable
    assert selected["compute_cost"] == selected["investment_compute_cost"] == 2200
    assert selected["research_plan"]["compute_budget"] == 2200
    assert len(calls) == len(set(calls)) == 2
    assert selected["return_estimate"]["compute_cost"] == 600
