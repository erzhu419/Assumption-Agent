"""Revise a failed inherited band without refitting its surviving program."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition import world_response_hypotheses as research
from hegel_machine.cognition import world_reference_recalibration as recipe
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.world_response_transfer import (
    KNOWLEDGE_PATH, export_response_knowledge, forecast_response_knowledge,
)
from test_world_reference_hypothesis import PATH, ReferenceLab
from test_world_intermediate_redecision import _unfinished_trial
from test_world_response_controller import _questions, _state
from test_world_response_transfer_stage import (
    _forbid_new_response_members,
    _public_initial_state,
)


def _fixed_program_state():
    program = Program('feature', name='measurement').to_dict()
    rows = [{'crystal': n, 'measurements': {
        'measurement': {'value': float(n), 'evidence': n}}} for n in (1, 2, 4)]
    rows[0]['frequency'] = {'value': 1., 'evidence': 1}
    rows[1]['frequency'] = {'value': 4., 'evidence': 2}
    hypothesis = {
        'id': 'old_a', 'family': 'linear', 'program': program,
        'lower': -.5, 'upper': 4.5, 'target_prediction': 4.,
        'predictions_by_crystal': {'1': 1., '2': 2., '4': 4.},
    }
    return {
        'target_numbers': [4], 'completed': [],
        'test': {'number': 4, 'prediction': 5., 'response_question': 'old_response',
                 'response_forecasts': {'old_a': True}},
        'research_table': {'references': rows[:2], 'targets': rows[2:]},
        'response_research': {'question_id': 'old_response', 'hypotheses': [hypothesis]},
        'response_settings': [
            {'crystal': 1, 'frequency': 1., 'activated': True, 'evidence': [1]},
            {'crystal': 2, 'frequency': 4., 'activated': True, 'evidence': [2]},
            {'crystal': 4, 'frequency': 8., 'activated': False, 'evidence': [10]},
        ],
    }


def test_positive_residual_hull_revises_band_without_refitting_or_restoring_old_member(monkeypatch):
    state = _fixed_program_state()
    before = deepcopy(state)
    monkeypatch.setattr(research, '_fit_linear', _forbid_new_response_members)

    result = research.revise_pending_response_band(state)

    assert state == before
    assert result['status'] == 'supported_continuation'
    hypothesis, = result['hypotheses']
    assert hypothesis['id'] == 'revised_band_0'
    assert hypothesis['program'] == before['response_research']['hypotheses'][0]['program']
    assert (hypothesis['lower'], hypothesis['upper']) == (0., 2.)
    assert hypothesis['source_hypotheses'] == ['old_a']
    assert hypothesis['target_prediction'] == 4.
    assert hypothesis['predictions_by_crystal'] == {'1': 1., '2': 2., '4': 4.}
    assert result['requested_setting'] == 5. and result['number'] == 4
    assert {1, 2, 10} <= set(result['evidence'])
    assert result['compute_cost'] > 0
    assert state['completed'] == []
    assert before['response_research']['hypotheses'][0]['upper'] == 4.5


def test_revision_cannot_ignore_a_paid_negative_inside_positive_hull():
    state = _fixed_program_state()
    state['response_settings'].append(
        {'crystal': 1, 'frequency': 2., 'activated': False, 'evidence': [11]})

    result = research.revise_pending_response_band(state)

    assert result['status'] == 'no_supported_continuation'
    assert result['hypotheses'] == []


def test_request_outside_positive_hull_gets_two_testable_bands_without_confirmation():
    state = _fixed_program_state()
    state['test']['prediction'] = 7.
    before = deepcopy(state)

    result = research.revise_pending_response_band(state)

    assert result['status'] == 'supported_continuation'
    narrow, wide = result['hypotheses']
    assert (narrow['lower'], narrow['upper']) == (0., 2.)
    assert (wide['lower'], wide['upper']) == (0., 3.)
    assert narrow['requested_prediction'] is False
    assert wide['requested_prediction'] is True
    assert result['positive_supporters'] == [wide['id']]
    assert narrow['program'] == wide['program']
    assert state == before and state['completed'] == []
    # Both proposals explain every paid response; the unobserved request is
    # precisely their distinguishing trial, rather than a new positive label.
    for hypothesis in result['hypotheses']:
        for row in state['response_settings']:
            residual = row['frequency'] - hypothesis['predictions_by_crystal'][str(row['crystal'])]
            assert (hypothesis['lower'] <= residual <= hypothesis['upper']) == row['activated']


@pytest.mark.parametrize('requested_setting', [8., 9., 2., 3.])
def test_paid_negative_boundaries_block_request_inclusive_of_the_boundary(requested_setting):
    state = _fixed_program_state()
    state['test']['prediction'] = requested_setting
    # Residual -1 supplies the lower exclusive boundary; residual 4 above is
    # the upper boundary. A shared connected band cannot cross either negative.
    state['response_settings'].append(
        {'crystal': 1, 'frequency': 0., 'activated': False, 'evidence': [11]})

    result = research.revise_pending_response_band(state)

    assert result['status'] == 'no_supported_continuation'
    assert result['hypotheses'] == []
    assert result['positive_supporters'] == []


def test_duplicate_programs_share_revision_and_retain_original_supporter_sources():
    state = _fixed_program_state()
    duplicate = deepcopy(state['response_research']['hypotheses'][0])
    duplicate.update(id='old_b', lower=-1., upper=8.)
    state['response_research']['hypotheses'].append(duplicate)
    state['test']['response_forecasts']['old_b'] = True

    result = research.revise_pending_response_band(state)

    hypothesis, = result['hypotheses']
    assert hypothesis['source_hypotheses'] == ['old_a', 'old_b']
    assert (hypothesis['lower'], hypothesis['upper']) == (0., 2.)


def test_request_nonsupporters_are_not_repurposed_as_pending_trial_revisions():
    state = _fixed_program_state()
    state['test']['response_forecasts']['old_a'] = False

    result = research.revise_pending_response_band(state)

    assert result['status'] == 'no_supported_continuation'
    assert result['hypotheses'] == []


@pytest.mark.parametrize('remaining', [['narrow'], ['narrow', 'wide']])
def test_only_a_live_positive_revision_can_keep_tuning_the_request(tmp_path, monkeypatch, remaining):
    state, test, question = _unfinished_trial()
    test.update(prediction=472., crystal=204, reactor=104,
                band_revision={'hypotheses': [{'id': 'narrow'}, {'id': 'wide'}],
                               'positive_supporters': ['wide']})
    question['answer'].update(requested_setting=472., remaining_hypotheses=[],
        band_revision={'remaining_hypotheses': remaining,
                       'refuted_hypotheses': [] if 'wide' in remaining else ['wide']})
    question['answer']['observed_responses'][0]['frequency'] = 470.
    state.update(test=test, rejected={})
    # The ordinary public menu can approach 472 via 470. At that readout a
    # revised [0, 3] band around 469 can fail while [0, 0] survives, but the
    # surviving narrow band supplies no positive reason to continue to 472.
    lab = ReferenceLab()
    lab.installed.add(4)
    lab.frequencies[4] = 470.
    lab.dialog_number, lab.in_dialog = 4, True
    view = lab.public_observation()
    observations = [{'index': 23, 'action': {'action': 'TALK', 'arg1': 104},
                     'observation': view}]
    state_file = tmp_path / 'state.json'
    state_file.write_text(json.dumps(state))
    monkeypatch.setattr(recipe, 'STATE', state_file)

    selected = recipe.select_actions({'current_observation': view}, observations,
                                    {'program_path': PATH, 'questions': [question]})

    saved = json.loads(state_file.read_text())
    if 'wide' in remaining:
        assert selected['actions'] == [{'chosen_dialog_option_int': 5}]
        assert saved['test']['prediction'] == 472.
        assert not saved.get('tests')
    else:
        assert selected['actions'] == [{'chosen_dialog_option_int': 1}]
        assert 'test' not in saved
        retired, = saved['tests']
        assert retired['status'] == 'inconclusive' and retired['retirement']
        assert retired['family'] not in saved['rejected'].get('4', [])


@pytest.mark.parametrize('actual_positive', [False, True])
def test_export_rechecks_revised_bands_on_later_evidence_before_transferring(actual_positive):
    state = _fixed_program_state()
    state['test']['prediction'] = 7.
    revision = research.revise_pending_response_band(state)
    revision['question_id'] = 'paid_band_revision'
    state['tests'] = [{'band_revision': deepcopy(revision)}]
    state['response_settings'].append(
        {'crystal': 4, 'frequency': 7., 'activated': actual_positive, 'evidence': [12]})
    before = deepcopy(state)

    knowledge = export_response_knowledge(state, [], 'revised_source')

    assert state == before
    model, = knowledge['models']
    assert model['program'] == revision['hypotheses'][0]['program']
    assert (model['lower'], model['upper']) == (0., 3. if actual_positive else 2.)
    assert model['source_question'] == 'paid_band_revision'
    assert knowledge['rejected_count'] == 2  # old wide band and the newly refuted alternative
    assert knowledge['source_response_count'] == len(state['response_settings'])
    forecast, = forecast_response_knowledge(knowledge, {'measurement': 6.}, [8.5])
    assert forecast['prediction'] is actual_positive


def _wide_band_trial(tmp_path, *, actual_setting=260.):
    agent, lab, knowledge = _public_initial_state(tmp_path)
    lab.required[4] = actual_setting
    knowledge['models'][0]['upper'] = 40.
    (agent.workspace / KNOWLEDGE_PATH).write_text(json.dumps(knowledge))
    return agent, lab, knowledge


@pytest.mark.parametrize('actual_setting, expected_status', [(260., 'supported'), (255., 'refuted')])
def test_real_intermediate_band_revision_continues_once_and_its_answer_needs_fresh_evidence(
        tmp_path, monkeypatch, actual_setting, expected_status):
    agent, lab, knowledge = _wide_band_trial(tmp_path, actual_setting=actual_setting)
    monkeypatch.setattr(research, '_propose_response_members', _forbid_new_response_members)
    revision_proposals = []
    revision = None
    for _ in range(20):
        outcome = agent._run_procedure(lab, PATH, max_steps=1)
        proposal = outcome['steps'][0]['proposal']
        if proposal.get('arguments', {}).get('stage') == 'pending_band_revision':
            revision_proposals.append(deepcopy(proposal))
        state = _state(agent)
        if state.get('test', {}).get('band_revision') and revision is None:
            revision = deepcopy(state['test']['band_revision'])
            assert lab.frequencies[4] == 300.
            assert revision['status'] == 'supported_continuation'
            revised, = revision['hypotheses']
            assert (revised['lower'], revised['upper']) == (0., 0.)
            assert state['response_feedback']['remaining_hypotheses'] == []
            assert _questions(agent)[state['test']['question']]['status'] == 'inconclusive'
            assert not state['completed']
        if state.get('tests'):
            break
    else:
        pytest.fail('The revised pending request did not reach an actual physical answer')

    assert revision is not None and len(revision_proposals) == 1
    tested, = state['tests']
    assert tested['prediction'] == 260. and tested['status'] == expected_status
    assert tested['band_revision'] == revision
    assert 'retirement' not in tested
    assert lab.frequencies[4] == 260.
    assert lab.active(4) == (expected_status == 'supported')
    physical = _questions(agent)[tested['question']]
    assert physical['status'] == expected_status
    feedback = physical['answer']['response_feedback']
    assert feedback['remaining_hypotheses'] == []
    assert feedback['band_revision']['remaining_hypotheses'] == (
        [revision['hypotheses'][0]['id']] if expected_status == 'supported' else [])
    assert feedback['band_revision']['refuted_hypotheses'] == (
        [] if expected_status == 'supported' else [revision['hypotheses'][0]['id']])
    assert not state.get('fit_bindings')
    assert agent.costs['model_calls'] == 0
    assert json.loads((agent.workspace / KNOWLEDGE_PATH).read_text()) == knowledge

    if expected_status == 'refuted':
        # A failed revision returns to the original researcher; it cannot invent
        # a second wider band solely to keep this already refuted request alive.
        following = agent._run_procedure(lab, PATH, max_steps=1)
        proposal = following['steps'][0]['proposal']
        assert proposal['arguments']['operator'] == 'reference_response'
        assert proposal['arguments']['stage'] == 'frozen_recall'
        assert 4 not in _state(agent)['completed']
