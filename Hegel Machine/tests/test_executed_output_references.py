"""Executed output references retain originals without recopying old results."""

from copy import deepcopy
import json

from hegel_machine.cognition.code_provider_context import prepare_code_context
from test_common_archive_resume import execute
from test_executed_evidence_notebook import evidence_scene


def test_analysis_fact_references_complete_stream_files_after_truncation(tmp_path):
    agent, lab = evidence_scene(tmp_path, 'hegel')
    feedback = execute(agent, lab, 'run', {'code': (
        "import sys\nprint('original stdout start\\n' + 'o' * 13000)\n"
        "print('original stderr start\\n' + 'e' * 14000, file=sys.stderr)\n")})

    fact = agent._request(lab)['evidence_notebook']['facts'][
        f'analysis:{len(agent.events) - 1}']

    for stream in ('stdout', 'stderr'):
        original = (tmp_path / feedback[f'{stream}_file']).read_text()
        assert len(original) > len(feedback[stream]) == 12000
        assert fact[f'{stream}_characters'] == len(original)
        assert fact[f'{stream}_truncated'] is True
        assert fact[stream] == {'full_content_file': feedback[f'{stream}_file'],
                                'characters': len(original)}
        assert original.startswith(f'original {stream} start\n')


def test_old_fact_keeps_original_file_when_numeric_context_is_compacted(tmp_path):
    text = 'stored computation\n' * 700
    events = [{'action': 'run', 'arguments': {'path': f'scratch_{i:03d}.py'},
               'feedback': {'returncode': 0, 'stdout': str(i)}} for i in range(10)]
    fact = {'kind': 'executed_python_analysis', 'event_index': 0,
            'stdout': text, 'stdout_file': 'scratch_000.py.stdout.txt',
            'stdout_characters': 17000, 'stdout_truncated': True}
    request = {'events': events,
        'online_research': {'questions': []},
        'laboratory_observations': [{'rows': [{'target': 0., 'features': {},
                                               'history': []}]}],
        'evidence_notebook': {'facts': {'analysis:0': fact}}}

    context = prepare_code_context(tmp_path, request, schema={}, instructions='test')

    actual = context['evidence_notebook']['facts']['analysis:0']
    assert actual['stdout'] == {'full_content_file': 'scratch_000.py.stdout.txt',
                                'characters': 17000}
    assert actual['stdout_characters'] == 17000
    assert actual['stdout_truncated'] is True


def test_reprinted_old_stdout_blocks_keep_new_residual_and_original_sources(tmp_path):
    original = 'paid model and record diagnostics\n' * 150
    events = []
    texts = [original, original + '\n---stderr---\n',
             'EVENT 55\n' + original + 'EVENT 56\n' + original + '\n---stderr---\n',
             original + '\nnew residual mse=0.125\n']
    for index, text in enumerate(texts):
        events.append({'action': 'run', 'arguments': {'code': 'print(old_output)'},
            'feedback': {'returncode': 0, 'stdout': text, 'stderr': 'new stderr',
                         'stdout_file': f'scratch_{index:03d}.py.stdout.txt',
                         'stdout_characters': len(text), 'stdout_truncated': False}})
    request = {'events': events, 'online_research': {'questions': []},
        'evidence_notebook': {'facts': {'analysis:3': {
            'kind': 'executed_python_analysis', 'event_index': 3,
            'stdout': texts[-1], 'stdout_file': 'scratch_003.py.stdout.txt'}}}}
    before = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions='test')

    latest = context['events'][-1]['feedback']
    assert latest['stdout']['parts'] == [
        {'full_content_file': 'scratch_000.py.stdout.txt', 'source_event_index': 0},
        '\nnew residual mse=0.125\n']
    assert latest['stdout']['full_content_file'] == 'scratch_003.py.stdout.txt'
    assert latest['stderr'] == 'new stderr'
    assert context['evidence_notebook']['facts']['analysis:3']['stdout'] == latest['stdout']
    assert original not in json.dumps(context)
    assert request == before
    assert json.loads((tmp_path / 'research_context.json').read_text()) == before


def test_short_results_and_execution_errors_are_not_deduplicated(tmp_path):
    request = {'events': [
        {'action': 'run', 'arguments': {'code': "print('mse=0.125')"},
         'feedback': {'returncode': 0, 'stdout': 'mse=0.125\n', 'stderr': ''}},
        {'action': 'run', 'arguments': {'code': "raise ValueError('new failure')"},
         'feedback': {'returncode': 1, 'stdout': 'mse=0.125\n',
                      'stderr': 'ValueError: new failure'}}]}

    context = prepare_code_context(tmp_path, request, schema={}, instructions='test')

    assert context['events'][0]['feedback']['stdout'] == 'mse=0.125\n'
    assert context['events'][1]['feedback'] == request['events'][1]['feedback']
