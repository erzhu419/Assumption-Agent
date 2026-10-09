"""Later evidence replaces one research trial rather than inventing another."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.action_policy import ActionOption, ResearchActionPolicy
from hegel_machine.cognition.models import Observation, TaskSpec


def policy(experience=None):
    return ResearchActionPolicy([Observation({'x': 0.}, 1.)],
        TaskSpec('public_task', 'Predict response', ('x',), (), ()),
        {} if experience is None else experience)


def record_trial(agent, gain, compute=40):
    feedback = agent.update('program', before_loss=1., after_loss=1. - gain,
        observation_cost=4, measurement_cost=8, compute_cost=compute)
    return {'context': agent.context, 'action': 'program',
            'trial': feedback['trials'], 'credited_gain': feedback['last_gain']}


def test_older_trial_correction_replaces_its_exact_mean_and_ema_contribution():
    experience = {}
    agent = policy(experience)
    origin = record_trial(agent, .8)
    record_trial(agent, .6, 50)
    latest = record_trial(agent, -.2, 60)
    revised = agent.revise_feedback(experience, origin, status='refuted',
        before_loss=2., after_loss=2.8, compute_cost=12, evidence_rows=12)
    assert revised['trials'] == 3
    assert revised['expected_gain'] == pytest.approx(.25 * -.4 + .25 * .6 + .5 * -.2)
    assert revised['mean_gain'] == pytest.approx((-.4 + .6 - .2) / 3)
    assert revised['mean_compute_cost'] == 54
    assert revised['mean_observation_cost'] == 4
    assert revised['mean_measurement_cost'] == 8
    assert revised['last_gain'] == pytest.approx(-.2)  # Later trial remains the latest.

    restored = json.loads(json.dumps(experience))
    same = agent.revise_feedback(restored, deepcopy(origin), status='refuted',
        before_loss=2., after_loss=2.8, evidence_rows=12)
    assert same == revised
    recovered = agent.revise_feedback(restored, deepcopy(origin), status='supported',
        before_loss=2., after_loss=.4, evidence_rows=12)
    assert recovered['expected_gain'] == pytest.approx(.25 * .8 + .25 * .6 + .5 * -.2)
    assert recovered['mean_gain'] == pytest.approx((.8 + .6 - .2) / 3)
    assert revised['trial_outcomes']['1']['status'] == 'refuted'  # Earlier trace stays frozen.
    final = agent.revise_feedback(restored, latest, status='supported',
        before_loss=2., after_loss=0., evidence_rows=12)
    assert final['last_gain'] == 1.
    assert final['last_before_loss'] == 2.
    assert final['trials'] == 3


def test_confirmed_direction_precedes_exploration_until_new_evidence_refutes_it():
    experience = {}
    agent = policy(experience)
    options = [ActionOption('program', 0, 40), ActionOption('structure', 0, 40)]
    def choose():
        return agent.choose(options, observation_remaining=100, compute_remaining=1000)
    origin = record_trial(agent, .8)
    assert choose().name == 'structure'  # Initial fit alone does not suppress exploration.
    agent.revise_feedback(experience, origin, status='supported',
        before_loss=1., after_loss=.1, evidence_rows=24)
    assert choose().name == 'program'
    assert agent.last_decision['reason'] == 'confirmed_direction_before_exploration'

    newer = record_trial(agent, .8)
    feedback = agent.revise_feedback(experience, newer, status='refuted',
        before_loss=1., after_loss=.5, evidence_rows=24)
    assert feedback['expected_gain'] > 0  # A partial improvement is still measured improvement.
    assert choose().name == 'structure'  # It is no longer evidence of a solved direction.
    assert feedback['trials'] == 2


def test_paid_counterexample_withdraws_support_without_fabricating_loss_feedback():
    experience = {}
    agent = policy(experience)
    origin = record_trial(agent, .75)
    confirmed = agent.revise_feedback(experience, origin, status='supported',
        before_loss=4., after_loss=1., evidence_rows=16)
    refuted = agent.revise_feedback(experience, origin, status='refuted')
    assert refuted['expected_gain'] == confirmed['expected_gain']
    assert refuted['mean_gain'] == confirmed['mean_gain']
    assert refuted['mean_compute_cost'] == confirmed['mean_compute_cost']
    assert refuted['trial_outcomes']['1']['status'] == 'refuted'
    assert refuted['trial_outcomes']['1']['evidence_rows'] == 16
    assert confirmed['trial_outcomes']['1']['status'] == 'supported'

