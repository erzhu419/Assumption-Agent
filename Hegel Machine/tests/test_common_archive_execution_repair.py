"""Shared runtime repair is not evidence of autonomous scientific discovery."""

from copy import deepcopy

import pytest

from test_common_archive_llm_inquiry import ROOT, buy_prefix
from test_common_archive_online_research import ScriptedProvider, prepared
from test_executed_evidence_notebook import AccountedPublicLab
from test_inquiry_revision import action


BROKEN = 'values = [1, 3]\nprint(sum(values) / len(missing_values))\n'
REPAIRED = 'values = [1, 3]\nprint(sum(values) / len(values))\n'


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
def test_repeated_run_error_enters_local_repair_then_returns_to_parent(tmp_path, arm):
    provider = ScriptedProvider([
        ('run', {'path': 'calculation.py', 'research_question': ROOT,
                 'notes': 'Parent scientific investigation.'}),
        ('run', {'path': 'calculation.py', 'research_question': ROOT,
                 'notes': 'Parent still needs this calculation.'}),
        ('write', {'path': 'calculation.py', 'code': REPAIRED,
                   'notes': 'Local execution repair.'}),
        ('run', {'path': 'calculation.py'}),
        ('finish', {'lesson': 'The edited calculation executes and prints 2.0.',
                    'artifacts': ['calculation.py']}),
        ('run', {'code': "print('Parent research continues')"}),
    ])
    agent, lab = prepared(tmp_path, arm, provider, calls=6), AccountedPublicLab()
    agent._save('calculation.py', BROKEN)
    buy_prefix(agent, lab)
    original_assignment = deepcopy(next(q for q in agent.questions if q['id'] == 'task')['assignment'])
    untouched = []

    def checkpoint(result):
        if result['costs']['model_calls'] == 2:
            untouched.append((tmp_path / 'calculation.py').read_text())

    result = agent.run(lab, on_checkpoint=checkpoint)

    child, parent = provider.requests[2], provider.requests[5]
    assignment = child['llm_inquiry']['assignment']
    assert assignment['origin'] == 'host_execution_repair'
    assert assignment['failed_program_path'] == 'calculation.py'
    assert assignment['failure_event_indices'] == [1, 2]
    assert assignment['parent_assignment'] == original_assignment
    assert assignment['parent_question'] == ROOT['question']
    assert assignment['execution_failure']['stderr'] == [
        "NameError: name 'missing_values' is not defined"]
    assert child['costs']['model_calls'] == 2 and child['llm_inquiry']['depth'] == 1
    assert untouched == [BROKEN]  # The runtime supplied no source repair.
    assert provider.requests[4]['events'][-1]['feedback']['stdout'] == '2.0\n'
    assert not parent.get('llm_inquiry')
    assert parent['research_notes'] == 'Parent still needs this calculation.'
    assert (tmp_path / 'calculation.py').read_text() == REPAIRED
    returned = result['events'][2]['feedback']
    assert returned['returncode'] == 1  # Original paid failure remains intact.
    assert returned['inquiry_result']['status'] == 'inquiry_finished'
    assert returned['inquiry_result']['model_calls'] == 3
    assert returned['inquiry_result']['artifacts'] == ['calculation.py']
    question = returned['inquiry_result']['question']
    assert question['status'] == 'open' and 'answer' not in question
    assert result['costs']['model_calls'] == result['costs']['reported_tokens'] == 6
    assert len(result['events']) == 7 and len(provider.requests) == 6
    assert lab.paid == 2 and len(lab.actions) == 1
    assert result['costs']['compute_cost'] == 0
    assert result['model'] is None and not agent.confirmations
    assert not agent._inquiry_stack


def test_repair_errors_do_not_nest_and_repair_cannot_buy_observations(tmp_path):
    provider = ScriptedProvider([
        ('run', {'path': 'calculation.py'}),
        ('run', {'path': 'calculation.py'}),
        ('run', {'path': 'calculation.py'}),
        ('query', {'actions': [action(2.)]}),
        ('finish', {'lesson': 'Unresolved execution repair.'}),
        ('run', {'code': "print('Return to parent')"}),
    ])
    agent, lab = prepared(tmp_path, 'hegel', provider, calls=6), AccountedPublicLab()
    agent._save('calculation.py', BROKEN)
    buy_prefix(agent, lab)
    result = agent.run(lab)

    assert len(result['llm_inquiries']) == 1
    assert provider.requests[3]['llm_inquiry']['depth'] == 1
    assert 'return scientific actions' in provider.requests[4]['events'][-1]['feedback']['error']
    assert not provider.requests[5].get('llm_inquiry')
    assert lab.paid == 2 and len(lab.actions) == 1
    assert result['costs']['model_calls'] == 6 and result['model'] is None


def test_scientific_mismatch_does_not_trigger_execution_repair(tmp_path):
    provider = ScriptedProvider([
        ('evaluate', {'model': 'old.py'}),
        ('evaluate', {'model': 'old.py'}),
        ('run', {'code': "print('Scientific research remains with the parent')"}),
    ])
    agent, lab = prepared(tmp_path, 'hegel', provider, calls=3), AccountedPublicLab()
    buy_prefix(agent, lab)
    result = agent.run(lab)
    assert all(event['feedback']['mse'] > 0 for event in result['events'][1:3])
    assert all(request.get('llm_inquiry', {}).get('assignment', {}).get('origin')
               != 'host_execution_repair' for request in provider.requests)
    assert all(question.get('origin') != 'host_execution_repair'
               for question in result['online_questions'])
    assert all('execution_repair' not in event['feedback'] for event in result['events'])
    assert result['model'] is None
    assert lab.paid == 2


def test_frozen_failures_cannot_open_live_repair(tmp_path):
    agent, lab = prepared(tmp_path, 'hegel'), AccountedPublicLab()
    agent._save('calculation.py', BROKEN)
    buy_prefix(agent, lab)
    for _ in range(2):
        feedback = agent._execute('run', {'path': 'calculation.py'}, lab)
        agent.events.append({'action': 'run', 'arguments': {'path': 'calculation.py'},
                             'feedback': feedback})
        before = deepcopy(agent.costs)
        agent._after_model_turn(lab)
        assert agent.costs == before
    assert not agent._inquiry_stack and not agent.llm_inquiries
    assert agent.costs['model_calls'] == 0 and lab.paid == 2
