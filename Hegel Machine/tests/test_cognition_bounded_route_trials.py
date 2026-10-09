"""A bounded cheaper route can be tried without consuming its known fallback."""

import json

import pytest

from hegel_machine.cognition.action_policy import ActionOption, ResearchActionPolicy
from hegel_machine.cognition.models import Observation, TaskSpec


ROWS = [Observation({'x': -1., 'z': 0.}, .1), Observation({'x': 1., 'z': .3}, 1.2)]
SPEC = TaskSpec('observed_task', 'predict measurements', ('x', 'z'), (), ())
KNOWN = ActionOption('mechanism_revision', 0, 100)


def confirmed_policy():
    policy = ResearchActionPolicy(ROWS, SPEC, {})
    feedback = policy.update(KNOWN.name, before_loss=1., after_loss=.2,
                             observation_cost=20, measurement_cost=100, compute_cost=800)
    origin = {'context': policy.context, 'action': KNOWN.name, 'trial': feedback['trials'],
              'credited_gain': feedback['last_gain']}
    policy.charge_followup(policy._store, origin, observation_cost=10,
                           measurement_cost=50, compute_cost=200)
    policy.revise_feedback(policy._store, origin, status='supported')
    return policy


def choose(policy, trial, *, observations=50, measurements=200, compute=2000):
    return policy.choose([KNOWN, trial], observation_remaining=observations,
        measurement_remaining=measurements, compute_remaining=compute)


def trial(cap=300, **costs):
    return ActionOption('coordinate_reconstruction', costs.get('observations', 0),
        costs.get('compute', 20), payload=(None, {'compute_limit': cap}),
        measurement_cost=costs.get('measurements', 0))


def test_bounded_cheaper_trial_keeps_all_measured_followup_costs_available():
    policy = confirmed_policy()
    bounded = trial(observations=4, measurements=20)
    assert choose(policy, bounded) == bounded
    decision = policy.last_decision
    assert decision['reason'] == 'bounded_trial_before_confirmed_direction'
    assert decision['fallback_action'] == KNOWN.name
    assert decision['fallback_costs'] == {
        'observation_cost': 30., 'measurement_cost': 150., 'compute_cost': 1000.}
    assert decision['bounded_compute_limit'] == 300
    assert decision['dominant_resource_share'] == pytest.approx(.15)


@pytest.mark.parametrize('resources', [
    {'compute': 1299}, {'observations': 33}, {'measurements': 169}])
def test_trial_does_not_spend_the_fallback_reserve(resources):
    policy = confirmed_policy()
    assert choose(policy, trial(observations=4, measurements=20), **resources) == KNOWN
    assert policy.last_decision['reason'] == 'confirmed_direction_before_exploration'
    assert 'fallback_action' not in policy.last_decision


def test_unbounded_entry_quote_and_expensive_cap_cannot_claim_cheap_exploration():
    policy = confirmed_policy()
    unbounded = ActionOption('coordinate_reconstruction', 0, 1)
    assert choose(policy, unbounded) == KNOWN
    # Its tiny entry quote does not stand in for the actual whole-trial cap.
    assert choose(policy, trial(cap=1800, compute=1), compute=4000, measurements=10000,
                  observations=1000) == KNOWN


def test_zero_observation_quote_cannot_hide_three_times_the_known_compute_bill():
    policy = ResearchActionPolicy(ROWS, SPEC, {})
    known = ActionOption('coordinate_reconstruction', 0, 5449)
    feedback = policy.update(known.name, before_loss=1., after_loss=.2,
                             observation_cost=24, compute_cost=5449)
    policy.revise_feedback(policy._store,
        {'context': policy.context, 'action': known.name, 'trial': feedback['trials'],
         'credited_gain': feedback['last_gain']}, status='supported')
    expensive = ActionOption('mechanism_revision', 0, 216,
                              payload=(None, {'compute_limit': 16192}))
    # The nominal dominant share is smaller for the unknown direction only
    # because its quote has no follow-up observations. Its compute is costlier.
    assert 16192 / 240000 < 24 / 152
    selected = policy.choose([known, expensive], observation_remaining=152,
                              compute_remaining=240000)
    assert selected == known
    assert policy.last_decision['reason'] == 'confirmed_direction_before_exploration'


@pytest.mark.parametrize('outcome', ['no_candidate', 'refuted'])
def test_failed_bounded_trial_is_not_repeated_after_json_restore(outcome):
    policy = confirmed_policy()
    bounded = trial()
    assert choose(policy, bounded) == bounded
    if outcome == 'no_candidate':
        policy.record_unscored_attempt(bounded.name, status='no_candidate', observation_cost=0,
                                       compute_cost=300)
    else:
        policy.update(bounded.name, before_loss=1., after_loss=1.2, observation_cost=0, compute_cost=300)
    restored = ResearchActionPolicy(ROWS, SPEC, json.loads(json.dumps(policy._store)))
    assert choose(restored, bounded) == KNOWN
    assert restored.last_decision['reason'] == 'measured_improvement_per_resource'


def test_confirmed_cheap_trial_later_competes_by_its_actual_measured_return():
    policy = confirmed_policy()
    bounded = trial()
    assert choose(policy, bounded) == bounded
    feedback = policy.update(bounded.name, before_loss=1., after_loss=.2,
                             observation_cost=0, compute_cost=250)
    policy.revise_feedback(policy._store,
        {'context': policy.context, 'action': bounded.name, 'trial': feedback['trials'],
         'credited_gain': feedback['last_gain']}, status='supported')
    assert choose(policy, bounded) == bounded
    assert policy.last_decision['reason'] == 'measured_improvement_per_resource'
