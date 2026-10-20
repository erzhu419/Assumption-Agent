"""Method-author reasoning survives fresh sessions without entering transfer memory."""

from copy import deepcopy
import json
from pathlib import Path

from hegel_machine.cognition import code_agent
from hegel_machine.cognition.anonymous_method_learning import (
    AnonymousResearchMethodLearner, EMPTY_METHOD)
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMAgent, InquiryLLMConfig
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from test_python_method_learning import archived_predictor


class AuthorInterfaceLab:
    task_interface = 'Use only the public fixture interface.'

    def public_spec(self):
        return {'controls': ['u']}


def make_author(workspace, provider=None):
    def unexpected_practice(*args):
        raise AssertionError('Updating author reasoning must not buy a practice.')

    return AnonymousResearchMethodLearner(provider, workspace,
        knowledge={'files': {METHOD_PATH: EMPTY_METHOD}}, cases={'case_a': {}},
        lab_factory=lambda case: AuthorInterfaceLab(), prepare_trial=unexpected_practice,
        config=InquiryLLMConfig(max_model_calls=2, observation_budget=576,
            confirmation_rows=96, token_budget=10000, execution_seconds=30.))


def test_author_notebook_flows_through_real_provider_requests_checkpoint_and_restore(
        tmp_path, monkeypatch):
    contexts = []
    notes = ['Inspection is planned; no new empirical result yet.',
             'The inspection returned the retained source; its behavior remains untested.']

    def run_provider(command, prompt, timeout, cwd):
        context = json.loads(prompt)
        index = len(contexts)
        assert context['research_notes'] == ('' if index == 0 else notes[0])
        assert context['files'][METHOD_PATH] == EMPTY_METHOD
        schema = json.loads(Path(command[command.index('--output-schema') + 1]).read_text())
        assert 'research_notes' in schema['required']
        if index == 0:
            payload = {'action': 'run', 'arguments_json': '{}',
                'source_code': f'print(open({METHOD_PATH!r}).read())',
                'research_notes': notes[index]}
        else:
            assert context['events'][-1]['feedback']['returncode'] == 0
            payload = {'action': 'write',
                'arguments_json': json.dumps({'path': METHOD_PATH}),
                'source_code': EMPTY_METHOD, 'research_notes': notes[index]}
        contexts.append(context)
        Path(command[command.index('--output-last-message') + 1]).write_text(json.dumps(payload))
        return 0, json.dumps({'type': 'turn.completed', 'usage': {
            'input_tokens': 31, 'output_tokens': 7}}), '', False

    monkeypatch.setattr(code_agent, '_run', run_provider)
    workspace = tmp_path / 'author'
    provider = code_agent.CodexCodeProvider(workspace, executable='scripted-provider-test')
    author = make_author(workspace, provider)
    checkpoints = []
    result = author.run(on_checkpoint=lambda value: checkpoints.append(deepcopy(value)))
    assert len(contexts) == len(checkpoints) == 2
    assert [value['author_research_notes'] for value in checkpoints] == notes
    assert result['status'] == 'model_call_budget_exhausted'
    assert result['author_research_notes'] == notes[-1]
    assert (workspace / 'research_notes.md').read_text() == notes[-1]
    assert (workspace / METHOD_PATH).read_text() == EMPTY_METHOD
    assert result['practice_trials'] == []
    assert result['costs']['observation_cost'] == 0
    assert result['outgoing_knowledge']['files'] == {METHOD_PATH: EMPTY_METHOD}

    resumed = make_author(tmp_path / 'resumed')
    resumed.restore_practice(result)
    assert resumed._request(resumed.interface)['research_notes'] == notes[-1]
    assert resumed.result('researching')['author_research_notes'] == notes[-1]
    assert resumed.files[METHOD_PATH] == EMPTY_METHOD
    assert resumed.practice_trials == []
    assert resumed.costs['observation_cost'] == 0
    consumer = InquiryLLMAgent(None, tmp_path / 'consumer', arm='hegel',
        knowledge=result['outgoing_knowledge'], source_task=False)
    assert 'research_notes.md' not in consumer.files
    assert not (consumer.workspace / 'research_notes.md').exists()


def test_old_result_starts_an_empty_notebook_and_ordinary_actions_keep_authored_notes(tmp_path):
    author = make_author(tmp_path / 'author')
    previous = author.result('model_call_budget_exhausted')
    previous.pop('author_research_notes')
    resumed = make_author(tmp_path / 'resumed')
    resumed.restore_practice(previous)
    assert resumed._request(resumed.interface)['research_notes'] == ''
    arguments = {'path': METHOD_PATH, 'code': EMPTY_METHOD,
                 'notes': 'This is an author interpretation, not a published method.'}
    feedback = resumed._execute('write', arguments, resumed.interface)
    assert 'edit_practice' not in feedback
    assert arguments['notes']  # Executed argument history keeps its authored interpretation.
    resumed._execute('run', {'code': 'print(1)'}, resumed.interface)
    assert resumed._request(resumed.interface)['research_notes'] == arguments['notes']
    assert resumed.result('model_call_budget_exhausted')['outgoing_knowledge']['files'] == {
        METHOD_PATH: EMPTY_METHOD}
    assert resumed.practice_trials == []


def test_restoring_a_practiced_draft_preserves_the_author_notebook(tmp_path, archived_predictor):
    from test_python_method_learning import make_learner, practice

    template, _ = make_learner(tmp_path / 'template', archived_predictor)
    author = AnonymousResearchMethodLearner(None, tmp_path / 'author',
        knowledge=template.knowledge, cases=template.cases,
        lab_factory=template.lab_factory, prepare_trial=template.prepare_trial,
        practice_limits=template.practice_limits, config=template.config)
    trial = practice(author, 'case_a')
    assert trial['fresh_confirmation']['passed']
    costs = deepcopy(author.costs)
    source = author.files[METHOD_PATH]
    notes = 'This practiced source has paid support on case_a; other cases remain unresolved.'
    restored = author._execute('run', {'operator': 'restore_method',
        'trial': trial['directory'], 'notes': notes}, author.interface)
    assert restored['restored_method'] == trial['directory']
    assert author._request(author.interface)['research_notes'] == notes
    assert author.files[METHOD_PATH] == source
    assert len(author.practice_trials) == 1
    assert author.costs == costs
