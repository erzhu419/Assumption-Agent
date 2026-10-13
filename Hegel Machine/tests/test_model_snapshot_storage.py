"""Snapshot storage keeps execution inputs without walking excluded history."""

from dataclasses import asdict
import errno
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hegel_machine.cognition import code_agent
from hegel_machine.cognition.code_agent import (
    evaluate_python_model, python_model_context_matches, retain_evaluated_python_model,
)
from hegel_machine.cognition.inquiry_revision import _sources
from hegel_machine.cognition.models import Observation


def write(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_snapshot_retains_archived_data_imports_and_independent_fitting_inputs(tmp_path):
    source = tmp_path / 'source'
    training = [Observation({'u': 0.}, 2.)]
    write(source, 'observations.json', json.dumps([asdict(row) for row in training]))
    write(source, 'model_observations.json', json.dumps([asdict(row) for row in training]))
    write(source, 'basis_config.py', 'FACTOR = 3.\n')
    write(source, 'archive/source_000/parameter.json', '{"bias": 5}')
    write(source, 'archive/source_000/paid_trace.json', '{"events": ["retained evidence"]}')
    write(source, 'archive/source_000/__pycache__/helper.pyc', 'cached bytecode')
    write(source, 'research_state.json', '{"experiment_comparisons": ["host record"]}')
    write(source, 'research_context.json', '{"files": {"host request": "long history"}}')
    write(source, 'model.py', '''import json
import basis_config
from pathlib import Path
value = 0.
def fit(rows):
    global value
    parameter = json.loads(Path('archive/source_000/parameter.json').read_text())
    value = rows[0]['target'] + parameter['bias']
    Path('fitted_parameter.json').write_text(json.dumps(value))
def predict(row):
    return value + basis_config.FACTOR * row['features']['u']
''')
    snapshots = [retain_evaluated_python_model(source, 'model.py', [], 0.,
        destination=f'_experiment_models/comparison_{index}/_evaluated_model')
        for index in range(2)]
    for saved in snapshots:
        path = Path(saved['workspace'])
        assert not (path / 'research_state.json').exists()
        assert not (path / 'research_context.json').exists()
        assert not list(path.rglob('*.pyc'))
        assert (path / 'archive/source_000/paid_trace.json').read_text() == (
            source / 'archive/source_000/paid_trace.json').read_text()
    write(source, 'basis_config.py', 'FACTOR = 100.\n')
    write(source, 'archive/source_000/parameter.json', '{"bias": 100}')
    write(source, 'model_observations.json', '[{"features": {"u": 0}, "target": 100}]')
    first, second = (Path(item['workspace']) for item in snapshots)
    evaluated = evaluate_python_model(first, 'model.py', [Observation({'u': 4.}, 19.)])
    assert evaluated['returncode'] == 0, evaluated
    assert evaluated['predictions'] == [19.]
    assert (first / 'fitted_parameter.json').is_file()
    assert not (second / 'fitted_parameter.json').exists()
    assert not (source / 'fitted_parameter.json').exists()
    assert evaluate_python_model(second, 'model.py', [Observation({'u': 4.}, 19.)])['mse'] == 0


def test_source_record_prunes_retained_history_before_directory_traversal(tmp_path, monkeypatch):
    write(tmp_path, 'model.py', 'def predict(row): return 1.\n')
    write(tmp_path, 'shared/helper.py', 'FACTOR = 2.\n')
    excluded = tmp_path / '_experiment_models/comparison_0000/candidate_0000/_evaluated_model'
    write(excluded, 'archive/source_000/_evaluated_model/model.py', 'def predict(row): return -1.\n')
    original_scandir = os.scandir

    def scandir(path):
        assert Path(path) != excluded, 'source export descended into an excluded snapshot'
        return original_scandir(path)

    monkeypatch.setattr(os, 'scandir', scandir)
    if hasattr(Path, '_accessor'):  # Python 3.10 pathlib caches os.scandir.
        monkeypatch.setattr(Path._accessor, 'scandir', scandir)
    sources = _sources(SimpleNamespace(workspace=tmp_path), 'model.py')
    assert sources == {'model': 'model.py', 'files': {
        'model.py': 'def predict(row): return 1.\n',
        'shared/helper.py': 'FACTOR = 2.\n',
    }}


def test_parent_and_bound_child_fits_cannot_overwrite_another_candidate(tmp_path):
    source = tmp_path / 'source'
    child = '_observable_models/meter/_evaluated_model'
    fitting = '''import json
from pathlib import Path
def fit(rows):
    global value
    value = json.loads(Path('parameter.json').read_text()) + rows[0]['target']
    Path('parameter.json').write_text(json.dumps(value))
    Path('fit_created.json').write_text(json.dumps(value))
'''
    write(source, 'parent.py', fitting + '''def bind_models(models):
    global child
    child = models['meter']
def predict(row): return value + child(row)
''')
    write(source, 'parameter.json', '5')
    write(source, 'model_observations.json', '[{"features":{"u":0},"target":3}]')
    write(source, f'{child}/child.py', fitting + 'def predict(row): return value\n')
    write(source, f'{child}/parameter.json', '7')
    write(source, f'{child}/model_observations.json', '[{"features":{"u":0},"target":2}]')
    write(source, 'model_bindings.json', json.dumps({'parent.py': {'meter': f'{child}/child.py'}}))
    snapshots = [Path(retain_evaluated_python_model(source, 'parent.py', [], 0.,
        destination=f'_experiment_models/candidate_{index}/_evaluated_model')['workspace'])
        for index in range(2)]
    score = evaluate_python_model(snapshots[0], 'parent.py', [Observation({'u': 0.}, 17.)])
    assert score['returncode'] == 0, score
    assert score['mse'] == 0
    for directory in (source, snapshots[1]):
        assert (directory / 'parameter.json').read_text() == '5'
        assert (directory / child / 'parameter.json').read_text() == '7'
        assert not (directory / 'fit_created.json').exists()
        assert not (directory / child / 'fit_created.json').exists()
    assert json.loads((snapshots[0] / 'parameter.json').read_text()) == 8
    assert json.loads((snapshots[0] / child / 'parameter.json').read_text()) == 9
    assert evaluate_python_model(snapshots[1], 'parent.py', [Observation({'u': 0.}, 17.)])['mse'] == 0


def test_known_context_history_is_omitted_without_discarding_same_named_user_data(tmp_path):
    source = tmp_path / 'source'
    nested = 'archive/source_000/_evaluated_model'
    for directory in ('', nested):
        for name in ('research_state.json', 'research_context.json', 'method_decision.json'):
            write(source / directory, name, '{"host_history": true}')
        # Inside an ordinary data folder these names are model inputs, not
        # host records, even when the folder sits below an archived context.
        for name in ('result.json', 'research_state.json', 'method_decision.json'):
            write(source / directory, f'data/{name}', '3')
    write(source, 'archive/source_000/paid_trace.json', '{"events": ["actual old evidence"]}')
    write(source, f'{nested}/child.py', 'def predict(row): return 2.\n')
    write(source, 'observations.json', '[{"features":{"u":0},"target":6}]')
    write(source, 'model.py', '''import json
def predict(row):
    return json.load(open('data/result.json')) + json.load(open(
        'archive/source_000/_evaluated_model/data/method_decision.json'))
''')
    destination = tmp_path / 'candidate/_evaluated_model'
    retained = retain_evaluated_python_model(source, 'model.py', [], 0., destination=destination)
    for directory in ('', nested):
        for name in ('research_state.json', 'research_context.json', 'method_decision.json'):
            assert not (destination / directory / name).exists()
            assert (source / directory / name).read_text() == '{"host_history": true}'
        for name in ('result.json', 'research_state.json', 'method_decision.json'):
            assert (destination / directory / 'data' / name).read_text() == '3'
    assert (destination / 'archive/source_000/paid_trace.json').read_text() == (
        source / 'archive/source_000/paid_trace.json').read_text()
    assert evaluate_python_model(destination, retained['model'], [Observation({'u': 0.}, 6.)])['mse'] == 0
    assert python_model_context_matches(tmp_path / 'candidate', '_evaluated_model/model.py', source, 'model.py')


@pytest.mark.parametrize('unsupported_errno', [errno.EOPNOTSUPP, errno.ENOTTY, errno.EXDEV])
def test_unsupported_clone_copies_the_same_contents_and_metadata(tmp_path, monkeypatch, unsupported_errno):
    source, destination = tmp_path / 'parameters.json', tmp_path / 'snapshot.json'
    source.write_text('{"coefficient": 3.25}\n')
    source.chmod(0o640)
    os.utime(source, ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000))

    def unavailable(*args):
        raise OSError(unsupported_errno, 'filesystem cannot clone this copy')

    monkeypatch.setattr(code_agent.fcntl, 'ioctl', unavailable)
    code_agent._copy_model_file(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_mode == source.stat().st_mode
    assert destination.stat().st_mtime_ns == source.stat().st_mtime_ns
    destination.write_text('independent new fit')
    assert source.read_text() == '{"coefficient": 3.25}\n'


@pytest.mark.parametrize('storage_errno', [errno.EIO, errno.ENOSPC])
def test_storage_failure_is_reported_without_silent_copy_retry(tmp_path, monkeypatch, storage_errno):
    source, destination = tmp_path / 'parameters.json', tmp_path / 'snapshot.json'
    source.write_text('{"coefficient": 3.25}\n')

    def failed(*args):
        raise OSError(storage_errno, 'actual storage failure')

    def forbidden_copy(*args, **kwargs):
        pytest.fail('An I/O or capacity failure must not be disguised as unsupported cloning.')

    monkeypatch.setattr(code_agent.fcntl, 'ioctl', failed)
    monkeypatch.setattr(code_agent.shutil, 'copy2', forbidden_copy)
    with pytest.raises(OSError) as caught:
        code_agent._copy_model_file(source, destination)
    assert caught.value.errno == storage_errno
    assert source.read_text() == '{"coefficient": 3.25}\n'
