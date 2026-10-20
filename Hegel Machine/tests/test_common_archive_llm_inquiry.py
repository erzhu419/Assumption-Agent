"""Scripted child research exercises the shared online loop, not model ability."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.interrupted_online_inquiry import recover_interrupted_online_run

from test_common_archive_online_research import ConfirmationLab, ScriptedProvider, prepared
from test_common_archive_resume import SEED
from test_executed_evidence_notebook import AccountedPublicLab
from test_inquiry_revision import action


ROOT = {'id': 'task', 'question': 'Predict the process', 'parent_id': None, 'depends_on': []}
CHILD = {'id': 'local_relation', 'question': 'Test the local response contrast',
         'parent_id': 'task', 'depends_on': []}
MODEL = "def predict(row): return row['features']['u']\n"
READOUT = """def readout(result, spec, memory):
    return {'status': 'supported', 'answer': {'predictor': 'shared/child_model.py'},
            'evidence': [0, 1], 'checks': [{'id': 'local_difference',
                'terms': [{'row': 1, 'coefficient': 1}, {'row': 0, 'coefficient': -1}],
                'relation': 'eq', 'expected': 1., 'atol': 1e-10,
                'scope': {'question': 'The two paid reset responses differ by one.'}}]}
"""


def buy_prefix(agent, lab):
    feedback = agent._execute('query', {'actions': [action(0., 1.)], 'question': ROOT}, lab)
    agent.events.append({'step': -1, 'action': 'query', 'arguments': {},
                         'feedback': feedback, 'usage': {'total_tokens': 0}})


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
def test_child_repairs_python_then_checked_answer_returns_parent_for_confirmation(tmp_path, arm):
    repair = ("from pathlib import Path\n"
              f"Path('shared/child_model.py').write_text({MODEL!r})\n"
              f"Path('shared/local_readout.py').write_text({READOUT!r})\n")
    provider = ScriptedProvider([
        ('run', {'operator': 'inquire', 'question': CHILD, 'max_model_calls': 5,
                 'notes': 'Parent: test the child before selecting a predictor.'}),
        ('run', {'code': 'def broken(:\n pass\n', 'notes': 'Child: inspect my execution.'}),
        ('run', {'code': repair, 'notes': 'Child: repair syntax and expose an executable model.'}),
        ('run', {'operator': 'analyze_evidence', 'path': 'shared/local_readout.py',
                 'research_question': CHILD}),
        ('evaluate', {'model': 'shared/child_model.py', 'research_question': ROOT}),
        ('finish', {'model': 'shared/child_model.py', 'notes': 'Parent: confirm the returned model.'}),
    ])
    agent, lab = prepared(tmp_path, arm, provider, calls=6), ConfirmationLab()
    buy_prefix(agent, lab)

    result = agent.run(lab)

    assert result['status'] == 'completed' and result['fresh_confirmation']['passed']
    assert result['costs']['model_calls'] == result['costs']['reported_tokens'] == 6
    assert len(provider.requests) == 6 and lab.paid == 4
    assert len(lab.actions) == 2  # One paid prefix and one final independent check.
    child_request = provider.requests[1]
    assert child_request['llm_inquiry']['question_id'] == CHILD['id']
    assert child_request['research_root']['id'] == CHILD['id']
    repaired_request = provider.requests[2]
    assert 'SyntaxError' in repaired_request['events'][-1]['feedback']['stderr']
    assert repaired_request['research_notes'] == 'Child: inspect my execution.'
    parent_request = provider.requests[4]
    assert not parent_request.get('llm_inquiry')
    if arm == 'hegel':
        assert parent_request['research_root']['id'] == 'task'
    assert parent_request['research_notes'] == 'Parent: test the child before selecting a predictor.'
    assert parent_request['costs']['observation_cost'] == 2
    question = next(q for q in result['online_questions'] if q['id'] == CHILD['id'])
    assert question['evidence_status'] == 'supported'
    assert question['check_results'][0]['status'] == 'satisfied'
    assert question['answer'] == {'predictor': 'shared/child_model.py'}
    assert (tmp_path / 'shared/child_model.py').read_text() == MODEL
    assert result['model'] == 'shared/child_model.py'


def test_local_finish_preserves_open_question_and_child_notebook_for_next_invocation(tmp_path):
    provider = ScriptedProvider([
        ('run', {'operator': 'inquire', 'question': CHILD, 'max_model_calls': 4,
                 'notes': 'Parent notebook.'}),
        ('run', {'code': "print('Need another test')", 'notes': 'Child still needs an observation.'}),
        ('finish', {'lesson': 'Unresolved; return to parent.'}),
        ('run', {'operator': 'inquire', 'question': CHILD, 'max_model_calls': 4}),
        ('finish', {'lesson': 'Resume the same unresolved research.'}),
        ('run', {'code': "print('Parent retains its own work')"}),
    ])
    agent, lab = prepared(tmp_path, 'hegel', provider, calls=6), AccountedPublicLab()
    buy_prefix(agent, lab)
    result = agent.run(lab)
    first = next(event['feedback'] for event in result['events']
                 if event.get('arguments', {}).get('operator') == 'inquire')

    assert first['status'] == 'inquiry_finished'
    assert agent.model is None and not agent.confirmations
    assert first['question']['status'] == 'open' and 'answer' not in first['question']
    assert (tmp_path / 'research_notes.md').read_text() == 'Parent notebook.'
    assert provider.requests[4]['research_notes'] == 'Child still needs an observation.'
    assert any(event['action'] == 'finish' and
               event['arguments']['lesson'] == 'Unresolved; return to parent.'
               for event in provider.requests[4]['events'])
    assert agent.costs['model_calls'] == 6 and lab.paid == 2
    assert (tmp_path / 'research_notes.md').read_text() == 'Parent notebook.'


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
def test_ordinary_child_finish_evaluates_returned_model_before_parent_handoff(tmp_path, arm):
    provider = ScriptedProvider([
        ('run', {'operator': 'inquire', 'question': CHILD, 'max_model_calls': 3,
                 'notes': 'Parent awaits the evaluated child artifact.'}),
        ('finish', {'model': 'shared/child_model.py', 'code': 'def predict(:\n pass\n'}),
        ('finish', {'model': 'shared/child_model.py', 'code': MODEL,
                    'lesson': 'The parent can now assess this executable candidate.'}),
        ('run', {'code': "print('Parent receives the paid evaluation')"}),
    ])
    agent, lab = prepared(tmp_path, arm, provider, calls=4), AccountedPublicLab()
    buy_prefix(agent, lab)

    result = agent.run(lab)

    retry = provider.requests[2]
    assert retry['llm_inquiry']['question_id'] == CHILD['id']
    assert retry['events'][-1]['feedback']['status'] == 'inquiry_delivery_failed'
    assert 'SyntaxError' in retry['events'][-1]['feedback']['stderr']
    parent = provider.requests[3]
    assert not parent.get('llm_inquiry')
    assert parent['research_notes'] == 'Parent awaits the evaluated child artifact.'
    delivery = parent['events'][-1]['feedback']
    assert delivery['status'] == 'inquiry_finished'
    assert delivery['evaluation']['returncode'] == 0
    assert delivery['evaluation']['mse'] == pytest.approx(0.)
    assert delivery['evaluation']['evaluated_model']['evaluated_rows'] == 2
    assert 'shared/child_model.py' in delivery['inquiry_return']['artifacts']
    assert delivery['inquiry_return']['question']['status'] == 'open'
    assert result['evaluated_model']['mse'] == pytest.approx(0.)
    assert agent.model is None and not agent.confirmations
    assert result['costs']['observation_cost'] == lab.paid == 2
    assert len(lab.actions) == 1


def test_nested_child_shares_parent_and_global_call_limits(tmp_path):
    nested = {'id': 'nested', 'question': 'Inspect a sub-assumption',
              'parent_id': CHILD['id'], 'depends_on': []}
    provider = ScriptedProvider([
        ('run', {'operator': 'inquire', 'question': CHILD, 'max_model_calls': 20}),
        ('run', {'operator': 'inquire', 'question': nested, 'max_model_calls': 20}),
        ('run', {'code': "print('One nested calculation')"}),
        ('run', {'code': "print('The direct parent adopts or revises the nested work')"}),
        ('run', {'code': "print('The root still has its last global call')"}),
    ])
    agent, lab = prepared(tmp_path, 'hegel', provider, calls=5), AccountedPublicLab()
    buy_prefix(agent, lab)

    result = agent.run(lab)

    assert result['status'] == 'model_call_budget_exhausted'
    assert result['costs']['model_calls'] == result['costs']['reported_tokens'] == 5
    assert len(provider.requests) == 5
    assert provider.requests[1]['llm_inquiry']['depth'] == 1
    assert provider.requests[2]['llm_inquiry']['depth'] == 2
    assert provider.requests[2]['llm_inquiry']['remaining_model_calls'] == 1
    assert provider.requests[3]['llm_inquiry']['question_id'] == CHILD['id']
    assert provider.requests[3]['llm_inquiry']['remaining_model_calls'] == 1
    assert not provider.requests[4].get('llm_inquiry')
    assert result['model'] is None and not agent.confirmations and lab.paid == 2


def test_frozen_agent_cannot_open_paid_child_loop(tmp_path):
    agent, lab = prepared(tmp_path, 'hegel'), AccountedPublicLab()
    buy_prefix(agent, lab)
    before = deepcopy(agent.costs)
    with pytest.raises(ValueError, match='provider|live model|frozen'):
        agent._execute('run', {'operator': 'inquire', 'question': CHILD, 'max_model_calls': 2}, lab)
    assert agent.costs == before and lab.paid == 2
    assert agent.model is None and not agent.confirmations


class RestorableConfirmationLab(AccountedPublicLab, ConfirmationLab):
    pass


class ContextSavingProvider(ScriptedProvider):
    """Match the live provider's pre-action context artifact for restoration."""

    def next_action(self, request):
        (self.workspace / 'research_context.json').write_text(json.dumps(request))
        return super().next_action(request)


@pytest.mark.parametrize('interrupted_run', [False, True])
def test_checkpoint_inside_child_resumes_scope_without_rebuying_parent_call(tmp_path, interrupted_run):
    original = ContextSavingProvider([
        ('run', {'operator': 'inquire', 'question': CHILD, 'max_model_calls': 4,
                 'notes': 'Parent notebook before delegation.'}),
        ('write', {'path': 'shared/child_model.py', 'code': MODEL,
                   'notes': 'Child notebook after writing its unfinished hypothesis.'}),
    ])
    agent, lab = prepared(tmp_path / 'old', 'hegel', original, calls=6), RestorableConfirmationLab()
    original.workspace = agent.workspace
    buy_prefix(agent, lab)
    checkpoints = []

    def stop_after_write(result):
        if result['events'][-1]['action'] == 'write':
            checkpoints.append(deepcopy(result))
            raise InterruptedError('Stop at the actual child checkpoint')

    with pytest.raises(InterruptedError):
        agent.run(lab, on_checkpoint=stop_after_write)
    previous = checkpoints[-1]
    assert previous['costs']['model_calls'] == 2
    assert previous['inquiry_stack'][-1]['question_id'] == CHILD['id']
    if interrupted_run:
        (agent.workspace / 'active.json').write_text(json.dumps(previous))
        code = "print('Child calculation interrupted before return')"
        executed_path = f"scratch_{len(previous['events']):03d}.py"
        (agent.workspace / executed_path).write_text(code)
        (agent.workspace / 'research_notes.md').write_text('Child notebook for its interrupted calculation.')
        (agent.workspace / 'code_action_003.json').write_text(json.dumps({
            'action': 'run', 'arguments_json': '{}', 'source_code': code,
            'research_notes': 'Child notebook for its interrupted calculation.',
            'research_question': None}))
        (agent.workspace / 'code_call_003.jsonl').write_text(json.dumps({
            'type': 'turn.completed', 'usage': {'input_tokens': 1, 'output_tokens': 0,
                                             'cached_input_tokens': 0}}) + '\n')
        previous = recover_interrupted_online_run(agent.workspace, model='scripted',
            interruption={'executed_path': executed_path, 'interrupted_python_seconds': 0.1},
            result_path=agent.workspace / 'interrupted_training.json')
        assert previous['costs']['model_calls'] == 3
        assert previous['inquiry_stack'][-1]['question_id'] == CHILD['id']
        assert previous['events'][-1]['feedback']['research_question_id'] == CHILD['id']

    continuation = ScriptedProvider([
        ('run', {'path': 'shared/child_model.py'}),
        ('finish', {'lesson': 'The artifact is ready for the parent to test.'}),
        ('finish', {'model': 'shared/child_model.py'}),
    ])
    resumed, fresh = prepared(tmp_path / 'resumed', 'hegel', continuation, calls=6), RestorableConfirmationLab()
    restored = resumed.restore_research(previous, agent.workspace, fresh, confirmation_seed=SEED)
    assert restored['new_observations'] == 0 and fresh.actions == []
    result = resumed.run(fresh)

    first = continuation.requests[0]
    assert first['llm_inquiry']['question_id'] == CHILD['id']
    assert first['research_notes'] == ('Child notebook for its interrupted calculation.' if interrupted_run
        else 'Child notebook after writing its unfinished hypothesis.')
    assert first['events'][-1]['action'] == ('run' if interrupted_run else 'write')
    assert first['costs']['model_calls'] == 2 + int(interrupted_run)
    if interrupted_run:
        assert first['events'][-1]['feedback']['status'] == 'runtime_interrupted'
        assert len(previous['events']) - 1 in first['llm_inquiry']['event_indices']
        child_memory = next(question for question in first['online_research']['questions']
                            if question['id'] == CHILD['id'])
        assert any(analysis['event_index'] == len(previous['events']) - 1
                   for analysis in child_memory['executed_analyses'])
        assert sum(event.get('feedback', {}).get('status') == 'runtime_interrupted'
                   for event in result['events']) == 1
    parent = continuation.requests[2]
    assert not parent.get('llm_inquiry')
    assert parent['research_notes'] == 'Parent notebook before delegation.'
    assert result['status'] == 'completed' and result['fresh_confirmation']['passed']
    assert result['costs']['model_calls'] == result['costs']['reported_tokens'] == 5 + int(interrupted_run)
    assert fresh.paid == 4 and len(fresh.actions) == 1
    assert sum(event.get('arguments', {}).get('operator') == 'inquire'
               for event in result['events']) == 1
    assert result['inquiry_stack'] == []
