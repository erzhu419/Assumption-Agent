"""Immutable source traces are shared; fitted execution contexts stay independent."""

import errno
import json
from pathlib import Path
import shutil

from hegel_machine.cognition import code_agent
from hegel_machine.cognition.code_agent import evaluate_python_model, retain_evaluated_python_model
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from scripts import run_anonymous_method_learning as study
from test_common_archive_online_research import ConfirmationLab


TRACE = 'archive/source_000/paid_trace.json'
REPEATED = 'archive/source_001/_evaluated_model/archive/source_000/paid_trace.json'
DISTINCT = 'archive/source_002/paid_trace.json'
MODEL = '''import json
from pathlib import Path
value = 0.
def fit(rows):
    global value
    value = json.loads(Path('parameter.json').read_text())['bias']
    Path('fit_created.json').write_text(json.dumps(value))
def predict(row):
    gain = json.loads(Path('archive/source_000/paid_trace.json').read_text())['gain']
    return value + gain * row['features']['u']
'''


def test_missing_nested_trace_reuses_existing_declared_source_without_rewriting(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    original = source / TRACE
    original.parent.mkdir(parents=True)
    content = '{"gain": 3, "events": ["already paid"]}'
    original.write_text(content)
    files = {REPEATED: content, TRACE: content}

    def no_trace_write(path, *args, **kwargs):
        raise AssertionError(f'Existing immutable evidence was rewritten: {path}')

    monkeypatch.setattr(Path, 'write_text', no_trace_write)
    references = code_agent.archived_paid_trace_references(
        files, tmp_path / 'reference_archive', source_workspace=source)
    assert references[REPEATED] == references[TRACE] == original.resolve()
    assert (tmp_path / 'reference_archive' / REPEATED).is_symlink()
    assert not (source / REPEATED).exists()


def prepare_trials(tmp_path, monkeypatch):
    def unavailable(*args):
        raise OSError(errno.EOPNOTSUPP, 'ext4 cannot clone this copy')

    monkeypatch.setattr(code_agent.fcntl, 'ioctl', unavailable)
    monkeypatch.setattr(study, 'laboratory', lambda case: ConfirmationLab())
    trace = json.dumps({'gain': 3, 'events': ['paid evidence'] * 10000})
    knowledge = {'files': {'model.py': MODEL, 'parameter.json': '{"bias": 5}',
        TRACE: trace, REPEATED: trace, DISTINCT: '{"gain": 4,"events":[]}'}}
    limits = InquiryLLMConfig(observation_budget=100, confirmation_rows=3,
                             compute_budget=10000, execution_seconds=30.)
    prepare = study.trial_preparer(knowledge, [], tmp_path / 'reference_archive')
    case = {'prefix_seed': 1, 'confirmation_seed': 2}
    method = 'def select_actions(spec, records, memory): return {"stop": "fixture"}\n'
    write_text = Path.write_text

    def no_trial_trace_write(path, *args, **kwargs):
        assert path.name != 'paid_trace.json', 'trial constructor rewrote immutable paid evidence'
        return write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'write_text', no_trial_trace_write)
    payloads = [prepare(case, tmp_path / f'trial_{i}', method, limits) for i in range(2)]
    monkeypatch.setattr(Path, 'write_text', write_text)
    return knowledge, limits, payloads


def test_ext4_trials_and_snapshots_share_paid_trace_but_keep_fits_independent(tmp_path, monkeypatch):
    _, _, payloads = prepare_trials(tmp_path, monkeypatch)
    first, second = (payload['agent'].workspace for payload in payloads)
    canonical = first / TRACE
    assert canonical.is_symlink()
    assert (first / REPEATED).resolve() == canonical.resolve() == (second / TRACE).resolve()
    assert (second / REPEATED).resolve() == canonical.resolve()
    assert (first / DISTINCT).resolve() != canonical.resolve()
    snapshots = [Path(retain_evaluated_python_model(first, 'model.py', [], 0.,
        destination=tmp_path / f'candidate_{i}/_evaluated_model')['workspace']) for i in range(2)]
    for snapshot in snapshots:
        assert (snapshot / TRACE).is_symlink()
        assert (snapshot / TRACE).resolve() == canonical.resolve()
        assert not (snapshot / 'parameter.json').is_symlink()
        assert not (snapshot / 'model_observations.json').is_symlink()
        assert not (snapshot / 'model.py').is_symlink()
    shutil.rmtree(first)
    result = evaluate_python_model(snapshots[0], 'model.py', [Observation({'u': 2.}, 11.)])
    assert result['returncode'] == 0 and result['mse'] == 0, result
    assert (snapshots[0] / 'fit_created.json').is_file()
    assert not (snapshots[1] / 'fit_created.json').exists()
    assert not (second / 'fit_created.json').exists()
    (snapshots[0] / 'parameter.json').write_text('{"bias": 100}')
    assert json.loads((snapshots[1] / 'parameter.json').read_text()) == {'bias': 5}
    assert evaluate_python_model(snapshots[1], 'model.py', [Observation({'u': 2.}, 11.)])['mse'] == 0


def test_restore_replaces_materialized_paid_trace_with_same_canonical_reference(tmp_path, monkeypatch):
    knowledge, limits, payloads = prepare_trials(tmp_path, monkeypatch)
    old, lab = payloads[0]['agent'], payloads[0]['laboratory']
    old._execute('evaluate', {'model': 'model.py'}, lab)
    (old.workspace / 'research_context.json').write_text(json.dumps(old._request(lab)))
    previous = old.result('researching')
    current = CommonArchiveInquiryAgent(None, tmp_path / 'restored', arm='hegel',
        knowledge=knowledge, source_task=False, config=limits)
    assert not (current.workspace / TRACE).is_symlink()
    restored_lab = TemporalInquiryLab(task_index=1)
    current.restore_research(previous, old.workspace, restored_lab, confirmation_seed=2)
    assert (current.workspace / TRACE).is_symlink()
    assert (current.workspace / TRACE).resolve() == (old.workspace / TRACE).resolve()
    assert (current.workspace / '_evaluated_model' / TRACE).is_symlink()
    assert current.costs == old.costs and current.rows == old.rows
    assert restored_lab.costs['observation_cost'] == old.costs['observation_cost']
    result = evaluate_python_model(current.workspace, '_evaluated_model/model.py',
                                   [Observation({'u': 2.}, 11.)])
    assert result['returncode'] == 0 and result['mse'] == 0, result
