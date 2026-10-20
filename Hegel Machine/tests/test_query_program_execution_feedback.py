"""Program execution facts survive a stop or a separately executed proposal."""

import json

import pytest

from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from test_anonymous_single_submission import prepared


def test_computation_then_stop_reports_actual_execution_and_authored_output(tmp_path):
    source = '''import json
def select_actions(spec, records, memory):
    total = sum(range(1000))
    with open('computation.json', 'w') as stream:
        json.dump({'total': total}, stream)
    print('completed_work=' + str(total))
    return {'stop': 'The attempted calculation did not produce a submission.'}
'''
    agent, lab = prepared(tmp_path, source)

    feedback = agent._query({'path': METHOD_PATH}, lab)

    assert feedback['status'] == 'method_stopped'
    assert json.loads((tmp_path / 'computation.json').read_text()) == {'total': 499500}
    execution = feedback['program_execution']
    assert execution['path'] == METHOD_PATH
    assert execution['returncode'] == 0 and execution['timeout'] is False
    assert execution['python_execution_seconds'] > 0
    assert execution['python_execution_seconds'] == agent.costs['python_execution_seconds']
    assert execution['stderr'].strip() == 'completed_work=499500'
    assert agent.method_results[-1]['feedback']['program_execution'] == execution
    assert agent.costs['observation_cost'] == agent.costs['model_calls'] == 0
    assert 'mse' not in feedback and 'fresh_confirmation' not in feedback


def test_proposal_time_excludes_the_following_evaluation(tmp_path, monkeypatch):
    source = '''def select_actions(spec, records, memory):
    return {'action': 'evaluate', 'arguments': {'model': 'good.py'}}
'''
    agent, lab = prepared(tmp_path, source)
    execute = agent._execute

    def timed_evaluation(action, arguments, laboratory):
        if action == 'evaluate':
            assert arguments == {'model': 'good.py'}
            agent.costs['python_execution_seconds'] += 7.
            return {'returncode': 0, 'mse': 2., 'evaluation_scope': 'paid_evaluation'}
        return execute(action, arguments, laboratory)

    monkeypatch.setattr(agent, '_execute', timed_evaluation)
    feedback = agent._query({'path': METHOD_PATH}, lab)

    assert feedback['mse'] == 2.
    assert feedback['program_action']['action'] == 'evaluate'
    execution = feedback['program_execution']
    assert execution['returncode'] == 0 and execution['timeout'] is False
    assert 'stderr' not in execution
    assert execution['python_execution_seconds'] > 0
    assert agent.costs['python_execution_seconds'] == pytest.approx(
        execution['python_execution_seconds'] + 7.)


def test_failed_program_retains_execution_facts_without_a_proposed_action(tmp_path):
    source = '''def select_actions(spec, records, memory):
    print('diagnostic:' + 'x' * 13000)
    raise ValueError('fixture calculation failed')
'''
    agent, lab = prepared(tmp_path, source)
    feedback = agent._query({'path': METHOD_PATH}, lab)

    execution = feedback['program_execution']
    assert execution['returncode'] == feedback['returncode'] != 0
    assert execution['timeout'] is feedback['timeout'] is False
    assert execution['python_execution_seconds'] == agent.costs['python_execution_seconds']
    assert len(execution['stderr']) == 12000
    assert execution['stderr'].endswith('ValueError: fixture calculation failed\n')
    assert agent.method_results[-1]['proposal'] is None
    assert agent.costs['observation_cost'] == 0


def test_successful_acquisition_keeps_its_program_execution(tmp_path):
    source = '''def select_actions(spec, records, memory):
    return {'actions': [{'kind': 'trajectory', 'sequence': [{'u': .5}]}]}
'''
    agent, lab = prepared(tmp_path, source)
    feedback = agent._query({'path': METHOD_PATH}, lab)

    assert feedback['observation_cost'] == agent.costs['observation_cost'] == 1
    assert feedback['program_execution']['returncode'] == 0
    assert feedback['program_execution']['timeout'] is False
    assert feedback['program_execution']['python_execution_seconds'] > 0
    assert len(agent.observations) == 1
