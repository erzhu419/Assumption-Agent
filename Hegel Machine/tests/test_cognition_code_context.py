"""The real provider can inspect all paid evidence without resending it each turn."""

from copy import deepcopy
import json
from pathlib import Path

from hegel_machine.cognition import code_agent


def test_context_preview_keeps_full_evidence_retrievable_and_latest_error_visible(tmp_path, monkeypatch):
    observations = [{'features': {'u': index / 100}, 'target': index / 10,
                     'trajectory_start': index % 9 == 0} for index in range(73)]
    program = 'def predict(row):\n    return row["features"]["u"]\n'
    events = [
        {'action': 'query', 'arguments': {'sequence': [{'u': row / 100} for row in range(64)]},
         'feedback': {'observations_file': 'observations.json', 'observation_cost': 64}},
        {'action': 'run', 'arguments': {'code': 'print("old diagnostic")'},
         'feedback': {'returncode': 0, 'stdout': 'old diagnostic\n' * 500, 'stderr': ''}},
        {'action': 'run', 'arguments': {'code': 'raise ValueError("refit needed")'},
         'feedback': {'returncode': 1, 'stdout': '', 'stderr': 'ValueError: refit needed'}},
    ]
    request = {'observations': observations, 'events': events,
               'files': {'shared/model.py': program, 'shared/revised.py': 'new explanation'},
               'session_knowledge': {'files': {'shared/model.py': program, 'shared/revised.py': 'old explanation'}, 'lessons': ['check applicability']}}
    original = deepcopy(request)
    transmitted = []

    def completion(command, payload, timeout, cwd):
        disabled = [command[index + 1] for index, value in enumerate(command[:-1]) if value == "--disable"]
        enabled = [command[index + 1] for index, value in enumerate(command[:-1]) if value == "--enable"]
        assert "code_mode_host" not in disabled and "code_mode_host" in enabled
        assert "shell_tool" in disabled and "multi_agent" in disabled
        context = json.loads(payload)
        transmitted.append(context)
        stored = json.loads((Path(cwd) / context['evidence']['full_context_file']).read_text())
        assert stored == original
        assert len(context['observations']) == 12
        assert context['observations'][0]['row_index'] == 61
        assert context['evidence']['observation_count'] == 73
        assert context['observations'][-1]['trajectory_start']
        assert len(context['events']) == len(events)
        assert context['events'][0]['arguments']['sequence_rows'] == 64
        assert 'sequence' not in context['events'][0]['arguments']
        assert 'stdout' not in context['events'][1]['feedback']
        assert context['events'][-1]['feedback']['stderr'] == 'ValueError: refit needed'
        assert context['files']['shared/model.py'] == program
        assert context['session_knowledge']['shared_files'] == ['shared/model.py', 'shared/revised.py']
        assert context['session_knowledge']['files'] == {'shared/revised.py': 'old explanation'}
        assert context['session_knowledge']['lessons'] == ['check applicability']
        destination = Path(command[command.index('--output-last-message') + 1])
        destination.write_text(json.dumps({'action': 'run', 'arguments_json': '{"code":"print(1)"}'}))
        warning = {'type': 'item.completed', 'item': {'type': 'error', 'message': 'runtime diagnostic'}}
        completed = {'type': 'turn.completed', 'usage': {'input_tokens': 10, 'output_tokens': 2}}
        return 0, json.dumps(warning) + '\n' + json.dumps(completed), '', False

    monkeypatch.setattr(code_agent, '_run', completion)
    reply = code_agent.CodexCodeProvider(tmp_path, executable='/test/codex').next_action(request)
    assert reply['action'] == 'run'
    assert reply['usage']['total_tokens'] == 12
    assert reply['usage']['runtime_errors'] == ['runtime diagnostic']
    assert request == original
    assert len(json.dumps(transmitted[0])) < len(json.dumps(original)) / 2
