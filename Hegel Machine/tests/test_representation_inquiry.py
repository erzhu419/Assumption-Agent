"""Scripted researchers expose their own input contracts on actual paid rows."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.representation_inquiry import open_representation_inquiry
from test_common_archive_llm_inquiry import ContextSavingProvider, RestorableConfirmationLab
from test_common_archive_online_research import ScriptedProvider
from test_scientific_revision_inquiry import prepared, prefix
from test_inquiry_revision import action


OPAQUE = 'def predict(row): return 0.\n'
BROKEN = 'def predict(:\n pass\n'
EXPOSED = '''def represent(row):
    return [0.]
def fit_features(features, targets):
    global mean
    mean = sum(targets) / len(targets)
def predict_features(features): return mean
'''


def initial_steps():
    return [
        ('finish', {'model': 'parent.py'}),
        ('finish', {'lesson': 'Local research needs parent synthesis.'}),
        ('evaluate', {'model': 'candidate.py', 'code': OPAQUE}),
    ]


def representation_questions(agent):
    return [q for q in agent.questions if q.get('origin') == 'representation_research']


def paused_at_automatic_review(path):
    provider = ContextSavingProvider(initial_steps())
    agent = prepared(path, provider, calls=10)
    provider.workspace = agent.workspace
    lab = RestorableConfirmationLab()
    prefix(agent, lab)
    saved = []

    def checkpoint(result):
        if (result['inquiry_stack'] and
                result['inquiry_stack'][-1]['question'].get('origin') == 'representation_research'):
            saved.append(deepcopy(result))
            raise InterruptedError('Representation review saved before its first provider call')

    with pytest.raises(InterruptedError, match='Representation review saved'):
        agent.run(lab, on_checkpoint=checkpoint)
    return agent, lab, saved[-1]


def test_failed_opaque_parent_opens_real_research_and_returns_executed_collision(tmp_path):
    provider = ScriptedProvider(initial_steps() + [
        ('finish', {'model': 'candidate.py', 'code': BROKEN}),
        ('finish', {'model': 'exposed.py', 'code': EXPOSED,
                    'lesson': 'The exposed input collapses distinct paid responses.'}),
        ('run', {'code': "print('Parent must repair the measured input ambiguity')"}),
    ])
    agent = prepared(tmp_path, provider, calls=6)
    lab = RestorableConfirmationLab()
    prefix(agent, lab)
    result = agent.run(lab)

    opened = provider.requests[3]['llm_inquiry']['assignment']
    assert opened['origin'] == 'representation_research'
    assert opened['representation_model'] == 'candidate.py'
    assert opened['source_at_start'] == OPAQUE
    assert 'fit_features' in opened['question'] and 'predict_features' in opened['question']
    # A syntax failure returns to the same researcher; local finish is not a submission.
    assert provider.requests[4]['llm_inquiry']['question_id'] == opened['id']
    assert provider.requests[4]['events'][-1]['feedback']['status'] == 'inquiry_delivery_failed'
    assert not provider.requests[5].get('llm_inquiry')
    question = representation_questions(agent)[0]
    assert question['source_at_return'] == BROKEN
    assert question['returned_model'] == 'exposed.py'
    assert question['returned_model_source'] == EXPOSED
    assert agent._parent_research()['parent_research']['evaluated_model'] == 'exposed.py'
    assert question['representation_return']['model_calls'] == 2
    evaluation = next(event['feedback']['evaluation'] for event in result['events']
                      if event.get('arguments', {}).get('code') == EXPOSED)
    assert evaluation['representation_diagnostic']['empirical_mse_floor'] > 0
    assert evaluation['representation_diagnostic']['witnesses']
    conflict = next(q for q in agent.questions if q['id'] == 'representation:exposed.py')
    assert conflict['status'] == 'open'
    assert agent._active_question_id == conflict['id']
    assert len(agent.confirmations) == 1 and result['fresh_confirmation']['passed'] is False
    assert result['model'] is None and lab.paid == result['costs']['observation_cost'] == 3


def test_same_source_is_attempted_once_changed_draft_reopens_and_missing_source_is_not_invented(tmp_path):
    agent, lab, checkpoint = paused_at_automatic_review(tmp_path)
    assert checkpoint['inquiry_stack'][-1]['question']['source_at_start'] == OPAQUE
    assert checkpoint['costs']['model_calls'] == 3
    agent._return_inquiry('inquiry_finished')
    assert not open_representation_inquiry(agent)
    assert len(representation_questions(agent)) == 1

    changed = OPAQUE + '# An authored revision.\n'
    agent._save('candidate.py', changed)
    assert open_representation_inquiry(agent)
    second = representation_questions(agent)[-1]
    assert second['source_at_start'] == changed
    returned = changed + '# Local work returned without a supported replacement.\n'
    agent._save('candidate.py', returned)
    agent._return_inquiry('inquiry_finished')
    assert second['source_at_return'] == returned
    assert not open_representation_inquiry(agent)
    assert len(representation_questions(agent)) == 2

    owner = agent._parent_research()
    owner['parent_research']['evaluated_model'] = 'missing_author_model.py'
    assert not open_representation_inquiry(agent)
    assert not (tmp_path / 'missing_author_model.py').exists()
    assert len(representation_questions(agent)) == 2
    assert lab.paid == 3 and agent.costs['model_calls'] == 3


def test_existing_checkpoint_restores_representation_session_without_new_paid_rows(tmp_path):
    old, old_lab, previous = paused_at_automatic_review(tmp_path / 'old')
    provider = ScriptedProvider([
        ('finish', {'lesson': 'Return this unfinished representation research.'}),
        ('run', {'code': "print('Parent retains the unresolved model and evidence')"}),
    ])
    restored = prepared(tmp_path / 'restored', provider, calls=5)
    lab = RestorableConfirmationLab()
    restored.restore_research(previous, old.workspace, lab, confirmation_seed=1709)
    result = restored.run(lab)

    first = provider.requests[0]['llm_inquiry']['assignment']
    assert first['origin'] == 'representation_research'
    assert first['source_at_start'] == OPAQUE
    assert first['representation_model'] == 'candidate.py'
    assert not provider.requests[1].get('llm_inquiry')
    assert len(representation_questions(restored)) == 1
    assert representation_questions(restored)[0]['source_at_return'] == OPAQUE
    assert result['paid_observations'] == previous['paid_observations']
    assert lab.actions == [] and lab.paid == old_lab.paid == 3
    assert result['costs']['model_calls'] == result['costs']['reported_tokens'] == 5
    assert result['model'] is None and result['fresh_confirmation']['passed'] is False


def test_frozen_and_code_agents_do_not_open_automatic_representation_research(tmp_path):
    agent, _, _ = paused_at_automatic_review(tmp_path)
    agent._return_inquiry('inquiry_finished')
    agent._save('candidate.py', OPAQUE + '# A new draft.\n')
    agent.arm = 'code_react'
    assert not open_representation_inquiry(agent)
    agent.arm = 'hegel'
    agent.provider = None
    assert not open_representation_inquiry(agent)
    assert len(representation_questions(agent)) == 1


def test_root_paid_rejection_exposes_actual_collision_before_any_confirmation(tmp_path):
    provider = ScriptedProvider([
        ('query', {'actions': [action(0., 2.)]}),
        ('evaluate', {'model': 'candidate.py', 'code': OPAQUE}),
        ('finish', {'model': 'exposed.py', 'code': EXPOSED,
                    'lesson': 'This input mapping cannot explain the paid observations.'}),
        ('run', {'code': "print('Parent decides how to revise its existing source')"}),
    ])
    agent = prepared(tmp_path, provider, calls=4)
    lab = RestorableConfirmationLab()
    result = agent.run(lab)

    assignment = provider.requests[2]['llm_inquiry']['assignment']
    assert assignment['origin'] == 'representation_research'
    assert assignment['root_paid_rejection'] == {
        'event_index': 1, 'model': 'candidate.py', 'paid_rows': 2, 'mse': 2.}
    assert 'source_catalog.json' in assignment['question']
    assert 'complete relevant functions' in assignment['question']
    returned = representation_questions(agent)[0]
    assert returned['returned_model'] == 'exposed.py'
    assert returned['research_return']['parent_id'] == 'task'
    parent = agent._parent_research()['parent_research']
    assert parent['evaluated_model'] == 'exposed.py'
    assert parent['phase'] == 'representation_conflict'
    assert parent['returned_event_indices'] == [2]
    assert not provider.requests[3].get('llm_inquiry')
    assert result['model'] is None and not agent.confirmations
    assert lab.paid == result['costs']['observation_cost'] == 2
    assert result['costs']['model_calls'] == 4


@pytest.mark.parametrize('condition', ['supported', 'stale_rows', 'unscored',
                                     'already_exposed', 'execution_failed', 'later_edit'])
def test_root_review_uses_only_current_failed_opaque_evaluation(tmp_path, condition):
    agent = prepared(tmp_path, ScriptedProvider([]), calls=4)
    lab = RestorableConfirmationLab()
    agent._execute('query', {'actions': [action(0., 2.)]}, lab)
    arguments = {'model': 'candidate.py', 'code': OPAQUE}
    feedback = agent._execute('evaluate', arguments, lab)
    agent.events.append({'action': 'evaluate', 'arguments': arguments, 'feedback': feedback})
    candidate = agent.research_candidates['candidate.py']
    if condition == 'supported':
        feedback['mse'] = 0.
    elif condition == 'stale_rows':
        candidate['paid_rows'] -= 1
    elif condition == 'unscored':
        candidate['status'] = 'new_paid_rows_unscored'
    elif condition == 'already_exposed':
        feedback['representation_diagnostic'] = {'status': 'no_exact_conflicts'}
    elif condition == 'execution_failed':
        feedback['returncode'] = 1
    elif condition == 'later_edit':
        agent._save('candidate.py', OPAQUE + '# A later untested edit.\n')
        agent.events.append({'action': 'write', 'arguments': {'path': 'candidate.py'},
                             'feedback': {'written': 'candidate.py'}})
    assert not open_representation_inquiry(agent)
    assert not representation_questions(agent)
    assert not agent._inquiry_stack
    assert lab.paid == 2 and agent.costs['model_calls'] == 0
