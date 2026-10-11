"""Paid tuning responses can return an unfinished frozen trial to its parent."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition import world_reference_recalibration as recipe
from hegel_machine.cognition import world_response_hypotheses as research
from hegel_machine.cognition.world_response_transfer import KNOWLEDGE_PATH
from test_world_reference_hypothesis import PATH
from test_world_response_controller import _questions, _state
from test_world_response_transfer_stage import (
    _forbid_new_response_members,
    _public_initial_state,
)


def _unfinished_trial():
    test = {
        'number': 4, 'family': 'old_square', 'prediction': 260.,
        'question': 'response_4_physical', 'response_question': 'response_4',
        'started': 20, 'response_forecasts': {'a': True, 'b': False, 'c': True},
    }
    state = {
        'target_numbers': [4, 5], 'completed': [],
        'response_research': {
            'question_id': 'response_4',
            'selection': {'consequence_scope': 'frozen_numeric_response_members'},
            'frozen_transfer': {'status': 'selected'},
        },
    }
    feedback = {
        'requested_setting': 260.,
        'observed_responses': [
            {'crystal': 4, 'frequency': 300., 'activated': False, 'evidence': [23]}],
        'remaining_hypotheses': ['b'], 'refuted_hypotheses': ['a', 'c'],
    }
    question = {'id': test['question'], 'status': 'inconclusive',
                'answer': feedback, 'evidence': [23]}
    return state, test, question


@pytest.mark.parametrize('remaining', [['b'], []])
def test_paid_response_retires_trial_when_no_requested_supporter_remains(remaining):
    state, test, question = _unfinished_trial()
    question['answer']['remaining_hypotheses'] = remaining
    if not remaining:
        question['answer']['refuted_hypotheses'].append('b')
    before = deepcopy((state, test, question))

    assert recipe._response_trial_lost_support(state, test, question)
    assert (state, test, question) == before
    assert not state['completed']
    assert question['status'] == 'inconclusive'


def test_one_surviving_positive_supporter_keeps_original_experiment():
    state, test, question = _unfinished_trial()
    question['answer'].update(remaining_hypotheses=['b', 'c'], refuted_hypotheses=['a'])

    assert not recipe._response_trial_lost_support(state, test, question)


@pytest.mark.parametrize('change', [
    'local_family', 'old_question', 'authored', 'reference', 'control_preparation',
    'no_observed_response', 'no_positive_forecast', 'physical_supported', 'physical_refuted',
])
def test_redecision_requires_its_own_paid_unfinished_frozen_target_trial(change):
    state, test, question = _unfinished_trial()
    if change == 'local_family':
        state['response_research']['selection']['consequence_scope'] = 'declared_response_family'
    elif change == 'old_question':
        state['response_research']['question_id'] = 'response_5'
    elif change == 'authored':
        test['authored_experiment'] = True
    elif change == 'reference':
        test['restore_frequency'] = 260.
    elif change == 'control_preparation':
        test['control_preparation'] = {'boundary': 0.}
    elif change == 'no_observed_response':
        question['answer']['observed_responses'] = []
        question['evidence'] = []
    elif change == 'no_positive_forecast':
        test['response_forecasts'] = {'a': False, 'b': False, 'c': False}
    else:
        question['status'] = change.removeprefix('physical_')

    assert not recipe._response_trial_lost_support(state, test, question)


def test_common_positive_prediction_without_measurement_does_not_complete_target():
    state, test, question = _unfinished_trial()
    test['response_forecasts'] = {'a': True, 'b': True}
    question['answer'].update(remaining_hypotheses=['a', 'b'], refuted_hypotheses=[])
    question['answer']['observed_responses'] = []
    question['evidence'] = []

    assert not recipe._response_trial_lost_support(state, test, question)
    assert state['completed'] == []
    assert question['status'] == 'inconclusive'


def test_paid_intermediate_counterexample_returns_to_parent_without_losing_real_activation(
        tmp_path, monkeypatch):
    agent, lab, knowledge = _public_initial_state(tmp_path)
    # A real success must be preserved even when a new response band could
    # support the old request. Repairing the negative-only case is tested in
    # test_world_pending_band_revision through the same public interface.
    lab.required[4] = 300.
    monkeypatch.setattr(research, '_propose_response_members', _forbid_new_response_members)

    for _ in range(12):
        agent._run_procedure(lab, PATH, max_steps=1)
        state = _state(agent)
        if state.get('tests'):
            break
    else:
        pytest.fail('An acquired intermediate refutation never returned the trial to its parent')

    retired, = state['tests']
    assert retired['number'] == 4 and retired['prediction'] == 260.
    assert retired['status'] == 'inconclusive'
    assert 'test' not in state
    assert retired['family'] not in state['rejected'].get('4', [])
    retirement = retired['retirement']
    assert retirement['reason'] == 'intermediate_responses_refuted_all_requested_members'
    assert retirement['remaining_hypotheses'] == []
    assert retirement['requested_supporters'] == [
        key for key, positive in retired['response_forecasts'].items() if positive]
    assert retirement['evidence']
    physical = _questions(agent)[retired['question']]
    assert physical['status'] == 'inconclusive'
    assert physical['answer']['remaining_hypotheses'] == []
    assert any(row['frequency'] == 300. and row['activated']
               for row in physical['answer']['observed_responses'])
    assert not any(row['frequency'] == 260. for row in state['response_settings']
                   if row['crystal'] == 4)
    assert lab.frequencies[4] == 300.
    assert lab.active(4)
    assert 'band_revision' not in retired

    following = agent._run_procedure(lab, PATH, max_steps=1)
    proposal = following['steps'][0]['proposal']
    assert proposal['arguments']['operator'] == 'reference_response'
    assert proposal['arguments']['stage'] == 'frozen_recall'
    state = _state(agent)
    assert state['response_context']['number'] == 5
    assert state['completed'] == [4]
    assert state['response_research']['frozen_transfer']['status'] == 'refuted'
    assert not state.get('fit_bindings')
    assert lab.frequencies[4] == 300.
    assert _questions(agent)[retired['question']]['status'] == 'inconclusive'

    actions = deepcopy(lab.actions)
    fitted = agent._run_procedure(lab, PATH, max_steps=1)
    assert fitted['steps'][0]['proposal']['arguments']['operator'] == 'research_update'
    assert _state(agent)['fit_bindings']
    assert lab.actions == actions
    assert agent.costs['model_calls'] == 0
    assert json.loads((agent.workspace / KNOWLEDGE_PATH).read_text()) == knowledge
