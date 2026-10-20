"""Executed ordinary child returns exercise parent research, not model ability."""

from copy import deepcopy

import pytest

from test_common_archive_execution_repair import BROKEN, REPAIRED
from test_common_archive_llm_inquiry import ContextSavingProvider, RestorableConfirmationLab
from test_common_archive_online_research import ScriptedProvider
from test_parent_research_synthesis import IDENTITY, is_running
from test_scientific_revision_inquiry import prepared, prefix


CHILD = {'id': 'paid_contrast', 'question': 'Check the paid response contrast',
         'parent_id': 'task', 'depends_on': []}
READOUT_PATH = 'shared/generic_readout.py'
READOUT = """def readout(result, spec, memory):
    rows = [(i, row) for record in result['observations']
            for i, row in zip(record['row_indices'], record['rows'])]
    (left, a), (right, b) = rows[:2]
    return {'status': 'supported',
            'answer': 'The tested paid response contrast equals the public-control contrast.',
            'evidence': [left, right],
            'checks': [{'id': 'paid_difference',
                'terms': [{'row': right, 'coefficient': 1.},
                          {'row': left, 'coefficient': -1.}],
                'relation': 'eq',
                'expected': b['features']['u'] - a['features']['u'],
                'atol': 1e-10}]}
"""


def old_review_steps():
    return [
        ('finish', {'model': 'parent.py'}),
        ('finish', {'lesson': 'No executable parent revision was found.'}),
        ('finish', {'inconclusive': True, 'lesson': 'Leave the older review unresolved.'}),
        ('evaluate', {'model': 'parent.py'}),
    ]


def checked_child_steps():
    return [
        ('run', {'operator': 'inquire', 'question': CHILD, 'max_model_calls': 3}),
        ('run', {'operator': 'analyze_evidence', 'path': READOUT_PATH,
                 'record_indices': [0, 1]}),
    ]


def configured(path, steps, *, calls=None, provider_type=ScriptedProvider):
    provider = provider_type(steps)
    agent = prepared(path, provider, calls=len(steps) if calls is None else calls)
    if isinstance(provider, ContextSavingProvider):
        provider.workspace = agent.workspace
    # This fixture studies generic child returns, with the parent's actual
    # scalar input already exposed rather than a separate representation task.
    agent._save('parent.py', (agent.workspace / 'parent.py').read_text()
        + "\ndef represent(row): return row['features']['u']\n"
          "def predict_features(value): return predict({'features': {'u': value}})\n")
    agent._save(READOUT_PATH, READOUT)
    lab = RestorableConfirmationLab()
    prefix(agent, lab)
    return agent, lab, provider


def question(result, identifier):
    return next(q for q in result['online_questions'] if q['id'] == identifier)


def test_checked_ordinary_return_keeps_parent_active_until_new_confirmation(tmp_path):
    steps = old_review_steps() + checked_child_steps() + [
        ('run', {'code': "print('Parent must compose the checked child result')"}),
        ('evaluate', {'model': 'parent.py'}),
        ('evaluate', {'model': 'revision.py', 'code': IDENTITY}),
        ('finish', {'model': 'revision.py'}),
    ]
    agent, lab, provider = configured(tmp_path, steps)

    result = agent.run(lab)

    returned = provider.requests[6]
    assert is_running(returned)
    assert not returned.get('llm_inquiry')
    assert returned['parent_research_focus']['parent_id'] == 'task'
    assert returned['evaluated_model']['source_model'] == 'parent.py'
    assert returned['costs']['observation_cost'] == 3
    fact = returned['evidence_notebook']['facts']['research_return:paid_contrast']
    assert fact['status'] == 'answered'
    assert returned['evidence_notebook']['facts']['revision:0']['evidence_status'] == 'supported'
    assert is_running(provider.requests[7])  # Ordinary Python did not complete synthesis.
    assert is_running(provider.requests[8])  # Re-evaluating the old source did not complete it.
    assert is_running(provider.requests[9])
    assert provider.requests[9]['parent_research_focus']['phase'] == 'awaiting_independent_confirmation'
    owner = question(result, returned['parent_research_focus']['question_id'])
    assert owner['parent_research']['status'] == 'independent_confirmation_passed'
    old = next(q for q in result['online_questions']
               if q.get('origin') == 'fresh_confirmation_failure')
    assert old['parent_research']['status'] == 'inconclusive'
    child = question(result, CHILD['id'])
    assert child['research_return']['parent_id'] == 'task'
    assert child['research_return']['revision_indices'] == [0]
    assert child['evidence_status'] == 'supported'
    assert child['check_results'][0]['status'] == 'satisfied'
    assert result['fresh_confirmation']['passed'] and result['model'] == 'revision.py'
    assert len(agent.confirmations) == 2
    assert result['costs']['observation_cost'] == lab.paid == 5
    assert result['costs']['model_calls'] == result['costs']['reported_tokens'] == len(steps)


def test_evaluated_ordinary_child_model_returns_without_automatic_confirmation(tmp_path):
    steps = old_review_steps() + [
        ('run', {'operator': 'inquire', 'question': CHILD, 'max_model_calls': 3}),
        ('finish', {'model': 'local.py', 'code': IDENTITY,
                    'lesson': 'The parent receives an evaluated candidate, not confirmation.'}),
        ('run', {'code': "print('Parent still decides whether to use the local model')"}),
        ('finish', {'model': 'local.py'}),
    ]
    agent, lab, provider = configured(tmp_path, steps)

    result = agent.run(lab)

    returned = provider.requests[6]
    assert is_running(returned)
    assert returned['costs']['observation_cost'] == 3
    assert returned['evaluated_model']['source_model'] == 'local.py'
    fact = returned['evidence_notebook']['facts']['research_return:paid_contrast']
    assert fact['evaluation']['mse'] == pytest.approx(0.)
    child = question(result, CHILD['id'])
    assert child['status'] == 'open' and 'answer' not in child
    assert child['research_return']['revision_indices'] == []
    assert child['research_return']['evaluation']['paid_rows'] == 3
    owner = question(result, returned['parent_research_focus']['question_id'])
    assert owner['parent_research']['status'] == 'independent_confirmation_passed'
    assert result['fresh_confirmation']['passed'] and result['model'] == 'local.py'
    assert len(agent.confirmations) == 2 and result['costs']['observation_cost'] == lab.paid == 5
    assert result['costs']['model_calls'] == result['costs']['reported_tokens'] == len(steps)


def test_inconclusive_stays_closed_until_a_changed_checked_child_returns(tmp_path):
    changed_readout = READOUT.replace('rows[:2]', 'rows[1:3]').replace(
        'The tested paid response contrast', 'The second tested paid response contrast')
    steps = old_review_steps() + checked_child_steps() + [
        ('finish', {'inconclusive': True, 'lesson': 'No parent predictor is ready.'}),
        ('run', {'code': "print('Continue after explicit inconclusive')"}),
        ('write', {'path': READOUT_PATH, 'code': changed_readout}),
        *checked_child_steps(),
        ('run', {'code': "print('A changed checked child result needs parent synthesis')"}),
    ]
    agent, lab, provider = configured(tmp_path, steps)

    result = agent.run(lab)

    assert is_running(provider.requests[6])
    assert not is_running(provider.requests[7])
    assert not is_running(provider.requests[8])
    assert not is_running(provider.requests[9])
    assert is_running(provider.requests[11])
    owner = question(result, provider.requests[11]['parent_research_focus']['question_id'])
    assert owner['parent_research']['status'] == 'running'
    assert owner['parent_research_history'][-1]['status'] == 'inconclusive'
    assert owner['parent_research_history'][-1]['returned_event_indices'] != (
        owner['parent_research']['returned_event_indices'])
    child = question(result, CHILD['id'])
    assert child['evidence'] == [1, 2]
    assert child['research_return']['revision_indices'] == [1]
    old = next(q for q in result['online_questions']
               if q.get('origin') == 'fresh_confirmation_failure')
    assert old['parent_research']['status'] == 'inconclusive'
    assert result['model'] is None and len(agent.confirmations) == 1
    assert result['costs']['observation_cost'] == lab.paid == 3


@pytest.mark.parametrize('legacy_snapshot', [False, True])
def test_restored_child_return_uses_existing_events_without_reexecution_or_charges(
        tmp_path, monkeypatch, legacy_snapshot):
    steps = old_review_steps() + checked_child_steps() + [
        ('run', {'code': "print('Parent remains pending')"}),
    ]
    old, lab, provider = configured(tmp_path / 'old', steps,
                                    provider_type=ContextSavingProvider)
    previous = old.run(lab)
    expected = deepcopy(question(previous, CHILD['id'])['research_return'])
    if legacy_snapshot:
        question(previous, CHILD['id']).pop('research_return')
        owner = question(previous, 'task')
        owner.pop('review_return')
        owner.pop('parent_research')
    costs = deepcopy(previous['costs'])
    events = deepcopy(previous['events'])
    paid = deepcopy(previous['paid_observations'])
    later = ScriptedProvider([('run', {'code': "print('Resume parent without replaying the child')"})])
    restored = prepared(tmp_path / 'restored', later, calls=len(steps) + 1)
    fresh = RestorableConfirmationLab()
    original_python = restored._python

    def no_reexecution(*args, **kwargs):
        raise AssertionError('Restore must extract the paid return, not execute it again')

    monkeypatch.setattr(restored, '_python', no_reexecution)
    recovered = restored.restore_research(previous, old.workspace, fresh, confirmation_seed=1709)

    assert recovered['new_observations'] == 0
    assert restored.costs == costs and fresh.paid == costs['observation_cost']
    assert fresh.actions == [] and later.requests == []
    assert restored.events == events and restored.observations == paid
    assert question(restored.result('restored'), CHILD['id'])['research_return'] == expected
    monkeypatch.setattr(restored, '_python', original_python)
    result = restored.run(fresh)
    assert is_running(later.requests[0])
    assert 'research_return:paid_contrast' in later.requests[0]['evidence_notebook']['facts']
    assert result['costs']['model_calls'] == costs['model_calls'] + 1
    assert result['costs']['reported_tokens'] == costs['reported_tokens'] + 1
    assert result['costs']['observation_cost'] == fresh.paid == costs['observation_cost']
    assert fresh.actions == [] and result['paid_observations'] == paid
    assert result['model'] is None and len(restored.confirmations) == 1


def test_unchecked_child_delivery_does_not_start_parent_synthesis(tmp_path):
    steps = old_review_steps() + [
        ('run', {'operator': 'inquire', 'question': CHILD, 'max_model_calls': 3}),
        ('run', {'code': "print('An executed calculation with no checked scientific answer')"}),
        ('finish', {'lesson': 'The local relation remains unresolved.'}),
        ('run', {'code': "print('Parent retains ordinary research control')"}),
    ]
    agent, lab, provider = configured(tmp_path, steps)

    result = agent.run(lab)

    assert not is_running(provider.requests[-1])
    child = question(result, CHILD['id'])
    assert child['status'] == 'open' and not child.get('parent_research')
    assert not question(result, 'task').get('parent_research')
    assert result['model'] is None and len(agent.confirmations) == 1
    assert result['costs']['observation_cost'] == lab.paid == 3


def test_execution_repair_return_is_not_a_scientific_parent_obligation(tmp_path):
    steps = [
        ('run', {'path': 'calculation.py'}),
        ('run', {'path': 'calculation.py'}),
        ('write', {'path': 'calculation.py', 'code': REPAIRED}),
        ('run', {'path': 'calculation.py'}),
        ('finish', {'lesson': 'Execution repaired.', 'artifacts': ['calculation.py']}),
        ('run', {'code': "print('Parent decides its scientific next action')"}),
    ]
    agent, lab, provider = configured(tmp_path, steps)
    agent._save('calculation.py', BROKEN)

    result = agent.run(lab)

    repair = next(q for q in result['online_questions'] if q.get('origin') == 'host_execution_repair')
    assert not repair.get('research_return')
    assert not is_running(provider.requests[-1])
    assert not question(result, 'task').get('parent_research')
    assert result['model'] is None and not agent.confirmations
    assert result['costs']['observation_cost'] == lab.paid == 1
    assert result['costs']['compute_cost'] == 0
