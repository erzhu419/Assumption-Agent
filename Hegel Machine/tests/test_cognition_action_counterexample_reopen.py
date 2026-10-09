"""A frozen law's real new counterexample can reopen a failed direction."""

import json

from hegel_machine.cognition.action_policy import ActionOption, ResearchActionPolicy
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, TaskSpec


SPEC = TaskSpec("measured_response", "Predict acquired responses", ("x",), (), ())
PROGRAM = Program("feature", name="x")
OLD = [Observation({"x": x}, x) for x in (-1., -.5, .5, 1.)]
NEW = [Observation({"x": x}, x * x) for x in (-1., -.5, .5, 1.)]
OPTION = ActionOption("revise_response", 4, 200)
EVIDENCE = {"probe_id": "purchased_new_response", "frozen_program": PROGRAM.to_dict()}


def policy(experience):
    result = ResearchActionPolicy(OLD, SPEC, experience)
    result.set_condition("measured_response_misfit")
    return result


def choose(result):
    return result.choose([OPTION], observation_remaining=16, compute_remaining=1000,
                         minimum_improvement=1e-7)


def test_actual_refutation_reopens_once_and_preserves_the_failed_retry_after_json():
    experience = {}
    initial = policy(experience)
    initial.update(OPTION.name, before_loss=1., after_loss=2.,
                   observation_cost=4, compute_cost=200)
    original = initial.action_context(OPTION.name)
    assert choose(initial) is None
    fit_loss, counterexample_loss = loss(PROGRAM, OLD, 0., .5), loss(PROGRAM, NEW, 0., .5)
    reopened = initial.set_action_counterexample(OPTION.name, EVIDENCE,
        fit_loss=fit_loss, counterexample_loss=counterexample_loss, tolerance=1e-7)
    assert reopened != original and choose(initial).name == OPTION.name
    assert experience[original][OPTION.name]["expected_gain"] < 0
    initial.update(OPTION.name, before_loss=counterexample_loss,
        after_loss=counterexample_loss * 1.1, observation_cost=4, compute_cost=200)
    assert choose(initial) is None
    restored = policy(json.loads(json.dumps(experience)))
    assert restored.set_action_counterexample(OPTION.name, EVIDENCE, fit_loss=fit_loss,
        counterexample_loss=counterexample_loss, tolerance=1e-7) == reopened
    assert choose(restored) is None


def test_fitting_misfit_or_another_accurate_batch_does_not_reopen_old_failure():
    experience = {}
    initial = policy(experience)
    initial.update(OPTION.name, before_loss=1., after_loss=2.,
                   observation_cost=4, compute_cost=200)
    context = initial.action_context(OPTION.name)
    for fit_loss, current_loss in [(loss(PROGRAM, NEW, 0., .5), loss(PROGRAM, NEW, 0., .5)),
                                   (loss(PROGRAM, OLD, 0., .5), loss(PROGRAM, OLD, 0., .5))]:
        assert initial.set_action_counterexample(OPTION.name, EVIDENCE, fit_loss=fit_loss,
            counterexample_loss=current_loss, tolerance=1e-7) == context
        assert choose(initial) is None


def test_new_counterexample_still_requires_an_affordable_actual_action():
    initial = policy({})
    initial.update(OPTION.name, before_loss=1., after_loss=2.,
                   observation_cost=4, compute_cost=200)
    initial.set_action_counterexample(OPTION.name, EVIDENCE,
        fit_loss=loss(PROGRAM, OLD, 0., .5), counterexample_loss=loss(PROGRAM, NEW, 0., .5),
        tolerance=1e-7)
    assert initial.choose([OPTION], observation_remaining=3, compute_remaining=1000,
                          minimum_improvement=1e-7) is None
