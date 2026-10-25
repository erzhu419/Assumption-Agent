"""Remove irrelevant metadata work without changing retained score inputs."""

import json
from pathlib import Path

from hegel_machine.cognition import code_agent
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program


def write(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def pair(tmp_path, *, literal=True):
    roots = [tmp_path / 'current', tmp_path / 'snapshot']
    for root in roots:
        write(root, 'model.py', export_temporal_program(Program('const', value=1.)) if literal else
              'import helper\ndef predict(row): return helper.value\n')
        write(root, 'params.json', '{"value": 1}')
        write(root, 'model_observations.json', '[{"features":{"u":0},"target":1}]')
        write(root, 'data/input.json', '{"value": 2}')
        write(root, 'helper.py', 'value=1\n')
    return roots


def test_closed_siblings_skip_metadata_but_keep_nonpython_and_nested_inputs(tmp_path, monkeypatch):
    current, snapshot = pair(tmp_path)
    for root in (current, snapshot):
        for index in range(30):
            write(root, f'candidate_{index}.py', 'def predict(row): return 0.\n')
        write(root, 'data/helper.py', 'value=3\n')
    is_file = Path.is_file

    def no_sibling_stat(path):
        assert not (path.parent in {current, snapshot} and path.name.startswith('candidate_'))
        return is_file(path)

    monkeypatch.setattr(Path, 'is_file', no_sibling_stat)
    assert code_agent.python_model_context_matches(current, 'model.py', snapshot, 'model.py')
    # All these same-context files remain part of the old cache contract.
    for name, value in [('params.json', '{"value": 3}'),
                        ('model_observations.json', '[]'),
                        ('data/input.json', '{"value": 4}'),
                        ('data/helper.py', 'value=9\n')]:
        path = current / name
        original = path.read_text()
        path.write_text(value)
        assert not code_agent.python_model_context_matches(current, 'model.py', snapshot, 'model.py')
        path.write_text(original)


def test_general_python_still_checks_dynamic_helper_and_fitting_inputs(tmp_path):
    current, snapshot = pair(tmp_path, literal=False)
    assert code_agent.python_model_context_matches(current, 'model.py', snapshot, 'model.py')
    write(current, 'helper.py', 'value=7\n')
    assert not code_agent.python_model_context_matches(current, 'model.py', snapshot, 'model.py')
    write(current, 'helper.py', 'value=1\n')
    write(current, 'model_observations.json', '[{"features":{"u":0},"target":4}]')
    assert not code_agent.python_model_context_matches(current, 'model.py', snapshot, 'model.py')


def test_same_immutable_paid_trace_is_not_read_twice_and_retargeting_invalidates(tmp_path, monkeypatch):
    current, snapshot = pair(tmp_path, literal=False)
    original = write(tmp_path, 'canonical/paid_trace.json', '{"paid": [1,2,3]}')
    other = write(tmp_path, 'different/paid_trace.json', '{"paid": [1,2,4]}')
    relative = 'archive/source_000/paid_trace.json'
    for root in (current, snapshot):
        target = root / relative
        target.parent.mkdir(parents=True)
        target.symlink_to(original)
    read_bytes = Path.read_bytes
    calls = []

    def counted(path):
        calls.append(path)
        return read_bytes(path)

    monkeypatch.setattr(Path, 'read_bytes', counted)
    assert code_agent.python_model_context_matches(current, 'model.py', snapshot, 'model.py')
    assert all(path.name != 'paid_trace.json' for path in calls)
    (current / relative).unlink()
    (current / relative).symlink_to(other)
    assert not code_agent.python_model_context_matches(current, 'model.py', snapshot, 'model.py')
    assert any(path.name == 'paid_trace.json' for path in calls)


def test_selected_literal_and_new_bindings_cannot_use_old_score(tmp_path):
    current, snapshot = pair(tmp_path)
    write(current, 'model.py', export_temporal_program(Program('const', value=2.)))
    assert not code_agent.python_model_context_matches(current, 'model.py', snapshot, 'model.py')
    write(current, 'model.py', export_temporal_program(Program('const', value=1.)))
    write(current, 'model_bindings.json', json.dumps({'model.py': {'child': 'helper.py'}}))
    assert not code_agent.python_model_context_matches(current, 'model.py', snapshot, 'model.py')


def test_same_mtime_same_length_parameter_edit_still_invalidates(tmp_path):
    import os

    current, snapshot = pair(tmp_path)
    path = current / 'params.json'
    stat = path.stat()
    path.write_text('{"value": 2}')
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_size == (snapshot / 'params.json').stat().st_size
    assert not code_agent.python_model_context_matches(current, 'model.py', snapshot, 'model.py')
