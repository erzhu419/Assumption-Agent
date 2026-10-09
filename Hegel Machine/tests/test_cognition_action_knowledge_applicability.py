"""A newly executable replay gets its own appraisal without erasing experience."""

from copy import deepcopy
import json

from hegel_machine.cognition.action_policy import ActionOption, ResearchActionPolicy
from hegel_machine.cognition.models import Observation, TaskSpec


ROWS = [Observation({'x': -1., 'z': 0.}, .1), Observation({'x': 1., 'z': .3}, 1.2)]
SPEC = TaskSpec('observed_task', 'predict measurements', ('x', 'z'), (), ())
OLD = [{'source_id': 'source_linear', 'terms': ['x', 'z'], 'bindings': ['x', 'z']}]
NEW = [*OLD, {'source_id': 'source_composition', 'terms': ['x*x', 'x*z', 'z*z'], 'bindings': ['x', 'z']}]


def policy(experience=None):
    result = ResearchActionPolicy(ROWS, SPEC, experience if experience is not None else {})
    result.set_condition('unified:coordinates:calls:no_parent')
    return result


def observed(result, action, after=.2, compute=1000):
    feedback = result.update(action, before_loss=1., after_loss=after,
                              observation_cost=24, compute_cost=compute)
    return {'context': result.action_context(action), 'action': action,
            'trial': feedback['trials'], 'credited_gain': feedback['last_gain']}


def test_new_executable_knowledge_reopens_only_the_affected_direction():
    result = policy()
    confirmed = observed(result, 'coordinate_reconstruction')
    result.revise_feedback(result._store, confirmed, status='supported')
    observed(result, 'program_replay', after=.99, compute=200)
    base = deepcopy(result._store[result.context])
    context = result.set_action_knowledge('program_replay', NEW)
    replay = ActionOption('program_replay', 0, 80, payload=(None, {'compute_limit': 160}))
    reconstruction = ActionOption('coordinate_reconstruction', 0, 1000)
    chosen = result.choose([reconstruction, replay], observation_remaining=100, compute_remaining=5000)
    assert chosen == replay
    assert result.last_decision['reason'] == 'bounded_trial_before_confirmed_direction'
    assert result.last_decision['context'] == context != result.context
    assert result.last_decision['fallback_context'] == result.context
    assert result.last_decision['trials'] == 0
    assert result._store[result.context] == base
    observed(result, 'program_replay', after=0., compute=120)
    assert result.action_record('program_replay')['trials'] == 1
    assert result._store[result.context] == base


def test_unchanged_and_returning_knowledge_reuse_their_original_paid_outcomes_after_json():
    result = policy()
    old_context = result.set_action_knowledge('program_replay', OLD)
    observed(result, 'program_replay', after=1.1, compute=300)
    old_record = deepcopy(result.action_record('program_replay'))
    new_context = result.set_action_knowledge('program_replay', NEW)
    assert new_context != old_context
    assert result.action_record('program_replay')['trials'] == 0
    result.record_unscored_attempt('program_replay', status='no_candidate', observation_cost=0,
                                   compute_cost=160)
    restored = policy(json.loads(json.dumps(result._store)))
    assert restored.set_action_knowledge('program_replay', NEW) == new_context
    assert restored.action_record('program_replay')['no_candidate_attempts'] == 1
    assert restored.set_action_knowledge('program_replay', OLD) == old_context
    assert restored.action_record('program_replay') == old_record
    assert restored.set_action_knowledge('program_replay', deepcopy(OLD)) == old_context


def test_delayed_confirmation_and_followup_stay_with_the_original_knowledge_trial():
    result = policy()
    old_context = result.set_action_knowledge('program_replay', OLD)
    origin = observed(result, 'program_replay', after=.2, compute=200)
    new_context = result.set_action_knowledge('program_replay', NEW)
    new_before = deepcopy(result.action_record('program_replay'))
    result.charge_followup(result._store, origin, observation_cost=8, measurement_cost=4, compute_cost=50)
    result.revise_feedback(result._store, origin, status='refuted', before_loss=1., after_loss=1.3,
                            compute_cost=8, evidence_rows=8)
    assert result.action_context('program_replay') == new_context
    assert result.action_record('program_replay') == new_before
    old = result._store[old_context]['program_replay']
    assert old['trial_outcomes']['1']['status'] == 'refuted'
    assert old['mean_compute_cost'] == 258
    assert old['mean_observation_cost'] == 32 and old['mean_measurement_cost'] == 4


def test_revised_executable_terms_or_binding_are_new_applicability_not_other_action_resets():
    result = policy()
    observed(result, 'callable_transfer', after=.1, compute=400)
    callable_before = deepcopy(result.action_record('callable_transfer'))
    first = result.set_action_knowledge('program_replay', OLD)
    revised_terms = deepcopy(OLD)
    revised_terms[0]['terms'] = ['x*x', 'z']
    second = result.set_action_knowledge('program_replay', revised_terms)
    revised_binding = deepcopy(revised_terms)
    revised_binding[0]['bindings'] = ['z', 'x']
    third = result.set_action_knowledge('program_replay', revised_binding)
    assert len({first, second, third}) == 3
    assert result.action_context('callable_transfer') == result.context
    assert result.action_record('callable_transfer') == callable_before
    assert result.set_action_knowledge('program_replay', []) == result.context
