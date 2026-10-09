"""Live procedure plumbing with authored toy programs, not LLM adoption evidence."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from test_discoveryworld_agent import ANALYZE, InstrumentLab, LIMITS, ScriptedProvider


PROCEDURE = """import json
def select_actions(spec, observations, memory):
    state = json.load(open('research_state.json'))
    assert state['method_results'] == memory['method_results']
    questions = {q['id']: q for q in memory['questions']}
    if 'parent' not in questions:
        return {'action': 'run', 'arguments': {'operator': 'question',
            'id': 'parent', 'question': 'Which intervention is warranted?'}}
    if 'instrument' not in memory['answers']:
        return {'question': {'id': 'instrument', 'parent_id': 'parent',
                    'question': 'What does the instrument report?'},
                'actions': [{'action': 'MEASURE'}],
                'after_observation': [{'action': 'evaluate', 'arguments': {
                    'path': 'analysis.py', 'question_id': 'instrument'}}]}
    assert memory['answers']['instrument']['status'] == 'supported'
    assert memory['answers']['instrument']['evidence'] == [0]
    if not memory['methods']:
        return {'action': 'run', 'arguments': {'operator': 'method',
            'id': 'tuning', 'path': 'tune.py', 'requires': ['instrument'],
            'applicability': 'An instrument reading has been interpreted'}}
    return {'action': 'query', 'arguments': {'method_id': 'tuning',
                                           'question': {'id': 'parent'}}}
"""

TUNE = """def select_actions(spec, observations, memory):
    child = memory['answers']['instrument']
    assert child['status'] == 'supported'
    assert child['evidence'] == [0]
    assert any(item['feedback'].get('after_observation_results')
               for item in memory['method_results'])
    return [{'action': 'TUNE', 'arg1': 2 * child['answer']['gain']}]
"""


def make_agent(tmp_path, steps, *, arm='hegel', lab=None, limits=None, files=None):
    for name, source in (files or {}).items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    provider = ScriptedProvider(steps)
    agent = DiscoveryWorldAgent(provider, tmp_path, arm=arm, limits=limits or LIMITS)
    return agent, provider, lab or InstrumentLab()


@pytest.mark.parametrize('arm', ['code_archive', 'hegel'])
def test_one_live_procedure_returns_child_answer_and_changes_parent_intervention(tmp_path, arm):
    class RegistrationLab(InstrumentLab):
        def act(self, action):
            questions = {q['id']: q for q in agent.questions}
            assert questions['instrument']['opened_before_observation'] == 0
            assert questions['instrument']['parent_id'] == 'parent'
            if action['action'] == 'TUNE':
                assert questions['instrument']['answer'] == {'gain': 3}
                assert questions['parent']['status'] == 'reopened'
            return super().act(action)

    agent, provider, lab = make_agent(tmp_path, [
        ('run', {'operator': 'procedure', 'path': 'procedure.py', 'max_steps': 8,
                 'research_question': {'id': 'task', 'question': 'Solve public task',
                                       'parent_id': None, 'depends_on': []}}),
    ], arm=arm, lab=RegistrationLab(), files={
        'procedure.py': PROCEDURE, 'analysis.py': ANALYZE, 'tune.py': TUNE})
    result = agent.run(lab)

    assert result['status'] == 'completed'
    assert result['score']['completedSuccessfully']
    assert lab.actions == [{'action': 'MEASURE'}, {'action': 'TUNE', 'arg1': 6}]
    assert result['costs']['model_calls'] == len(provider.requests) == 1
    assert result['costs']['environment_steps'] == 2
    assert result['costs']['python_execution_seconds'] > 0
    assert result['research']['child_answers_returned'] == 1
    assert result['research']['methods'][0]['available']
    child = next(q for q in result['research']['questions'] if q['id'] == 'instrument')
    assert child['evidence'] == child['acquired_observations'] == [0]
    assert agent.method_results
    expected_final_step = {'action': 'query', 'arguments': {
        'method_id': 'tuning', 'question': {'id': 'parent'}}}
    assert agent.method_results[-1]['path'] == 'procedure.py'
    assert agent.method_results[-1]['proposal'] == expected_final_step
    assert result['events'][0]['feedback']['steps'][-1]['proposal'] == expected_final_step
    saved = json.loads((tmp_path / 'research_state.json').read_text())
    assert saved['method_results'] == agent.method_results
    assert saved['answers']['instrument']['answer'] == {'gain': 3}
    assert 'EVALUATOR_ONLY_GOLD_RULE' not in json.dumps(saved)
    assert (tmp_path / 'procedure.py').read_text() == PROCEDURE


def test_analysis_only_program_reuses_paid_records_and_explicit_continuation_wins(tmp_path):
    program = """def select_actions(spec, observations, memory):
    assert len(observations) == 1
    return {'question': {'id': 'instrument', 'question': 'Interpret the existing reading'},
            'actions': [], 'after_observation': [
                {'action': 'write', 'arguments': {'path': 'unused.txt', 'code': 'wrong'}}]}
"""
    agent, _, lab = make_agent(tmp_path, [
        ('query', {'actions': [{'action': 'MEASURE'}]}),
        ('query', {'path': 'analysis_only.py',
                   'research_question': {'id': 'task', 'question': 'Solve public task',
                                         'parent_id': None, 'depends_on': []},
                   'after_observation': [
            {'action': 'evaluate', 'arguments': {'path': 'analysis.py', 'question_id': 'instrument'}},
            {'operator': 'question', 'id': 'next', 'question': 'Which test should follow?'}]}),
        ('finish', {}),
    ], files={'analysis_only.py': program, 'analysis.py': ANALYZE})
    result = agent.run(lab)

    assert lab.actions == [{'action': 'MEASURE'}]
    questions = {q['id']: q for q in result['research']['questions']}
    assert questions['task']['question'] == 'Solve public task'
    assert questions['instrument']['answer'] == {'gain': 3}
    assert questions['instrument']['evidence'] == [0]
    assert questions['instrument']['acquired_observations'] == []
    assert questions['next']['status'] == 'open'
    assert not (tmp_path / 'unused.txt').exists()
    assert not result['events'][1]['feedback'].get('error')


def test_failed_acquisition_is_paid_and_stops_procedure_before_analysis(tmp_path):
    class FailedLab(InstrumentLab):
        def act(self, action):
            self.actions.append(deepcopy(action))
            self.environment_steps += 1
            return {'action': action, 'feedback': {'success': False, 'message': 'Instrument unavailable'},
                    'observation': self.public_observation()}

    program = """def select_actions(spec, observations, memory):
    return {'question': {'id': 'instrument', 'question': 'Read the instrument'},
            'actions': [{'action': 'MEASURE'}],
            'after_observation': [{'action': 'write', 'arguments': {
                'path': 'analysis_ran.txt', 'code': 'should not run'}}]}
"""
    agent, _, lab = make_agent(tmp_path, [
        ('run', {'operator': 'procedure', 'path': 'procedure.py', 'max_steps': 8}),
        ('finish', {}),
    ], arm='code_archive', lab=FailedLab(), files={'procedure.py': program})
    result = agent.run(lab)

    assert lab.environment_steps == len(agent.observations) == 1
    assert result['costs']['environment_steps'] == 1
    assert agent.observations[0]['feedback']['success'] is False
    question = next(q for q in result['research']['questions'] if q['id'] == 'instrument')
    assert question['acquired_observations'] == [0]
    assert question['status'] == 'open'
    assert not (tmp_path / 'analysis_ran.txt').exists()
    assert len(agent.method_results) == 1
    assert agent.method_results[0]['feedback'].get('error')
    assert result['events'][0]['feedback']['status'] == 'method_error'


def test_explicit_stop_returns_to_model_without_spending_observations(tmp_path):
    program = """def select_actions(spec, observations, memory):
    return {'stop': 'No discriminating experiment has been authored'}
"""
    agent, provider, lab = make_agent(tmp_path, [
        ('run', {'operator': 'procedure', 'path': 'procedure.py', 'max_steps': 8}),
        ('finish', {}),
    ], files={'procedure.py': program})
    result = agent.run(lab)

    assert len(agent.method_results) == 1
    assert result['events'][0]['feedback']['status'] == 'method_stopped'
    assert lab.actions == []
    assert result['costs']['model_calls'] == len(provider.requests) == 2
    assert not result['score']['completedSuccessfully']


def test_multiple_runs_in_one_procedure_preserve_each_source_and_complete_output(tmp_path):
    sources = ["print('FIRST:' + 'a' * 13000)", "print('SECOND:' + 'b' * 13000)"]
    program = f"""def select_actions(spec, observations, memory):
    sources = {sources!r}
    step = len(memory['method_results'])
    if step == len(sources):
        return {{'stop': 'Both analyses completed'}}
    return {{'action': 'run', 'arguments': {{'code': sources[step]}}}}
"""
    agent, _, lab = make_agent(tmp_path, [
        ('run', {'operator': 'procedure', 'path': 'procedure.py', 'max_steps': 8}),
        ('finish', {}),
    ], files={'procedure.py': program})
    result = agent.run(lab)

    assert result['events'][0]['feedback']['status'] == 'method_stopped'
    assert len(agent.method_results) == 3 and lab.actions == []
    feedback = [entry['feedback'] for entry in agent.method_results[:2]]
    output_paths = [entry['stdout_file'] for entry in feedback]
    assert output_paths[0] != output_paths[1]
    for index, entry in enumerate(feedback):
        expected = ('FIRST:' + 'a' * 13000 if index == 0 else 'SECOND:' + 'b' * 13000) + '\n'
        assert entry['returncode'] == 0 and entry['stdout_truncated']
        assert entry['stdout'] == expected[-12000:]
        assert (tmp_path / entry['stdout_file']).read_text() == expected
        source_path = entry['stdout_file'].removesuffix('.stdout.txt')
        assert (tmp_path / source_path).read_text() == sources[index]


@pytest.mark.parametrize('batch_size', [1, 2])
def test_environment_budget_stops_acquisition_but_analyzes_a_completed_batch(tmp_path, batch_size):
    program = f"""def select_actions(spec, observations, memory):
    return {{'question': {{'id': 'instrument', 'question': 'Read the instrument'}},
            'actions': [{{'action': 'MEASURE'}}] * {batch_size},
            'after_observation': [{{'action': 'write', 'arguments': {{
                'path': 'analysis_ran.txt', 'code': 'analyze the completed batch'}}}}]}}
"""
    agent, _, lab = make_agent(tmp_path, [
        ('run', {'operator': 'procedure', 'path': 'procedure.py', 'max_steps': 8}),
    ], limits={**LIMITS, 'environment_steps': 1}, files={'procedure.py': program})
    result = agent.run(lab)

    assert result['status'] == 'environment_budget_exhausted'
    assert lab.environment_steps == len(agent.observations) == 1
    assert len(agent.method_results) == 1
    assert (tmp_path / 'analysis_ran.txt').exists() == (batch_size == 1)


def test_wall_time_expiry_after_acquisition_skips_continuation(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr('hegel_machine.cognition.discoveryworld_agent.monotonic', lambda: clock[0])

    class ExpiringLab(InstrumentLab):
        def act(self, action):
            result = super().act(action)
            clock[0] = LIMITS['task_wall_seconds'] + 1
            return result

    program = """def select_actions(spec, observations, memory):
    return {'actions': [{'action': 'MEASURE'}], 'after_observation': [
        {'action': 'write', 'arguments': {'path': 'analysis_ran.txt', 'code': 'should not run'}}]}
"""
    agent, _, lab = make_agent(tmp_path, [
        ('run', {'operator': 'procedure', 'path': 'procedure.py', 'max_steps': 8}),
    ], lab=ExpiringLab(), files={'procedure.py': program})
    result = agent.run(lab)

    assert result['status'] == 'wall_time_exhausted'
    assert lab.environment_steps == len(agent.observations) == 1
    assert len(agent.method_results) == 1
    assert not (tmp_path / 'analysis_ran.txt').exists()
