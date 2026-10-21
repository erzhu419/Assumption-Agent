"""Registered organization runs on the same procedure account and action trace."""

from dataclasses import replace
import json

from hegel_machine.cognition.temporal_paired_method import execute_procedure
from test_procedure_question_readout import PublicLab, install, make_agent


def scene(tmp_path, source):
    agent, lab = make_agent(tmp_path), PublicLab()
    agent.config = replace(agent.config, confirmation_rows=2)
    install(agent, 'shared/author.py', source)
    return agent, lab


def test_each_real_procedure_action_sees_before_work_and_checkpoints_after_work(tmp_path):
    agent, lab = scene(tmp_path, """import json
def select_actions(spec,records,memory):
    turn=json.load(open('shared/turn.json'))['turn']
    return {'action':'write','arguments':{
        'path':'shared/seen_'+str(turn)+'.txt','code':str(turn)}}
""")
    before_calls, after_calls, checkpoints = [], [], []

    def before(laboratory):
        assert laboratory is lab
        before_calls.append(len(before_calls) + 1)
        agent._execute('write', {'path': 'shared/turn.json',
            'code': json.dumps({'turn': before_calls[-1]})}, laboratory)
        return True

    def after(laboratory):
        assert laboratory is lab
        after_calls.append(len(agent.method_results))
        agent.events.append({'action': 'organization',
            'feedback': {'completed_author_actions': after_calls[-1]},
            'usage': {'backend': 'test_organization', 'total_tokens': 0}})

    result = execute_procedure(agent, lab, 'shared/author.py', max_steps=2,
        record_events=True, before_step=before, after_step=after,
        on_checkpoint=checkpoints.append)

    assert result['status'] == 'method_step_limit'
    assert before_calls == after_calls == [1, 2]
    assert (agent.workspace / 'shared/seen_1.txt').read_text() == '1'
    assert (agent.workspace / 'shared/seen_2.txt').read_text() == '2'
    assert [event['action'] for event in agent.events] == ['query', 'organization'] * 2
    assert checkpoints[-1]['events'] == agent.events
    assert checkpoints[-1]['events'][-1]['feedback']['completed_author_actions'] == 2
    assert agent.costs['model_calls'] == agent.costs['observation_cost'] == lab.rows == 0


def test_before_work_exhausting_the_original_python_allowance_prevents_author_action(tmp_path):
    agent, lab = scene(tmp_path, """def select_actions(spec,records,memory):
    raise AssertionError('This proposal must never run after its account is exhausted')
""")
    original_config = agent.config
    completed, checkpoints = [], []

    def before(laboratory):
        agent.costs['python_execution_seconds'] = agent.config.execution_seconds
        return True

    outcome = execute_procedure(agent, lab, 'shared/author.py', record_events=True,
        before_step=before, after_step=lambda laboratory: completed.append(True),
        on_checkpoint=checkpoints.append)

    assert outcome['status'] == 'python_execution_budget_exhausted' and not outcome['steps']
    assert not completed and not agent.events and not agent.method_results
    assert agent.config == original_config
    assert checkpoints[-1]['costs']['python_execution_seconds'] == original_config.execution_seconds


def test_repeated_failure_stays_on_its_author_event_after_organization_appends_an_event(tmp_path):
    agent, lab = scene(tmp_path, """def select_actions(spec,records,memory):
    return {'action':'run','arguments':{'code':"raise ValueError('unfinished analysis')"}}
""")
    checkpoints = []

    def after(laboratory):
        agent.events.append({'action': 'organization', 'feedback': {'status': 'completed'},
            'usage': {'backend': 'test_organization', 'total_tokens': 0}})

    outcome = execute_procedure(agent, lab, 'shared/author.py', max_steps=4,
        record_events=True, after_step=after, on_checkpoint=checkpoints.append)

    assert outcome['status'] == 'method_no_progress'
    author_events = [event for event in agent.events if event['action'] == 'query']
    organization_events = [event for event in agent.events if event['action'] == 'organization']
    assert len(author_events) == len(organization_events) == 2
    assert author_events[-1]['feedback']['no_progress'] == outcome['no_progress']
    assert all('no_progress' not in event['feedback'] for event in organization_events)
    assert checkpoints[-1]['events'] == agent.events
    assert not agent.model and lab.rows == 0 and agent.costs['model_calls'] == 0
