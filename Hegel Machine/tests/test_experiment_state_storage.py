"""Host state writers keep their last recovery boundary when disk writes fail."""

import errno
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import checkpoint_storage
from hegel_machine.cognition.cli import _save_json
from hegel_machine.cognition.code_provider_context import load_code_context, prepare_code_context
from hegel_machine.cognition.delayed_feedback import _write_result
from hegel_machine.cognition.interrupted_online_inquiry import recover_interrupted_online_run
from test_cognition_unified_code_control import SharedProvider, session
from test_interrupted_online_inquiry import stopped_run


def disk_full(_descriptor):
    raise OSError(errno.ENOSPC, 'fixture disk full')


@pytest.mark.parametrize('writer', [_save_json, _write_result])
def test_host_checkpoint_failure_preserves_complete_prior_result(tmp_path, monkeypatch, writer):
    path = tmp_path / 'checkpoint.json'
    previous = {'events': [{'status': 'failed', 'evidence': [2, 5]}], 'costs': {'compute': 17}}
    writer(path, previous)
    monkeypatch.setattr(checkpoint_storage.os, 'fsync', disk_full)

    with pytest.raises(OSError, match='fixture disk full'):
        writer(path, {**previous, 'new_candidate': {'parameters': [1., 2.]}})

    assert json.loads(path.read_text()) == previous
    assert not list(tmp_path.glob('*.writing'))


def test_provider_request_failure_preserves_saved_source_versions(tmp_path, monkeypatch):
    previous = {'files': {'model.py': 'earlier source'},
        'session_knowledge': {'files': {'shared/old.py': '# original evidence\n' * 5000}}}
    prepare_code_context(tmp_path, previous, schema={}, instructions='original')
    monkeypatch.setattr(checkpoint_storage.os, 'fsync', disk_full)

    with pytest.raises(OSError, match='fixture disk full'):
        prepare_code_context(tmp_path, {**previous, 'files': {'model.py': 'changed source'}},
                             schema={}, instructions='changed')

    assert load_code_context(tmp_path) == previous
    assert (tmp_path / 'code_agent_instructions.txt').read_text() == 'original'


def test_interruption_export_failure_keeps_source_and_prior_export(stopped_run, monkeypatch):
    source, previous, _payload, stop = stopped_run
    path = source / 'interrupted_training.json'
    retained = recover_interrupted_online_run(source, model='fixture', interruption=stop, result_path=path)
    original = path.read_bytes()
    monkeypatch.setattr(checkpoint_storage.os, 'fsync', disk_full)

    with pytest.raises(OSError, match='fixture disk full'):
        recover_interrupted_online_run(source, model='fixture',
            interruption={**stop, 'interrupted_python_seconds': 12.}, result_path=path)

    assert path.read_bytes() == original
    assert json.loads(path.read_text()) == retained
    assert json.loads((source / 'active.json').read_text()) == previous
    assert not (source / 'must_not_execute.txt').exists()


@pytest.mark.parametrize('session_kind', ['unified', 'code'])
def test_common_successor_uses_shared_snapshot_copy_and_keeps_writes_independent(
        tmp_path, monkeypatch, session_kind):
    from hegel_machine.cognition import code_session, unified_code_control
    from hegel_machine.cognition.code_agent import CodeAgentConfig

    module = unified_code_control if session_kind == 'unified' else code_session
    control = (session(tmp_path) if session_kind == 'unified' else
        code_session.CodeResearchSession(tmp_path, backend='local',
            tasks=[{'task_index': 0, 'seed': seed} for seed in (113, 127)],
            config=CodeAgentConfig(observation_budget=2, measurement_budget=2,
                compute_budget=32, max_model_calls=3, execution_seconds=10),
            provider_factory=lambda _: SharedProvider()))
    control.run(max_tasks=1)
    original = tmp_path / 'task_000/shared/parameters.bin'
    original.write_bytes(b'paid fitted parameters\0' * 4096)
    copied = []
    copy_file = module._copy_model_file

    def record_copy(source, target):
        copied.append((Path(source), Path(target)))
        return copy_file(source, target)

    monkeypatch.setattr(module, '_copy_model_file', record_copy)
    control.run(max_tasks=1)
    current = tmp_path / 'task_001/shared/parameters.bin'

    assert (original, current) in copied
    assert current.read_bytes() == original.read_bytes()
    assert original.stat().st_ino != current.stat().st_ino
    with current.open('r+b') as stream:
        stream.write(b'refitted parameters')
    assert original.read_bytes().startswith(b'paid fitted parameters\0')


def test_world_practice_reuses_declared_paid_archive_before_author_writes(tmp_path):
    from test_world_method_learning import prepared

    learner, _labs, _providers = prepared(tmp_path)
    name = 'archive/source_000/paid_trace.json'
    original = tmp_path / name
    original.parent.mkdir(parents=True)
    text = json.dumps({'observations': ['already paid'] * 2000})
    original.write_text(text)
    learner.knowledge['files'][name] = text
    source = (tmp_path / 'shared/consolidated_method.py').read_text()

    outcome = learner._practice('source_initial')
    retained = tmp_path / outcome['directory'] / name
    method = tmp_path / outcome['directory'] / 'shared/consolidated_method.py'

    assert retained.is_symlink() and retained.resolve() == original
    assert retained.read_text() == text
    assert method.read_text() == source
    method.write_text('independent next draft')
    assert (tmp_path / 'shared/consolidated_method.py').read_text() == source
