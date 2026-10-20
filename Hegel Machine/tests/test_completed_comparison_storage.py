"""Completed fold evidence stays once; later fits get independent local files."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import common_archive_inquiry
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from test_common_archive_resume import SEED, agent_at, checkpoint, saved_research


REPORT = 'shared/comparison/report.json'
PARAMETER = 'shared/comparison/tree_fold_0.joblib'
OTHER_PARAMETER = 'shared/comparison/linear_fold_0.joblib'
RESIDUALS = 'shared/comparison/residuals.json'


def add_comparison(workspace, *, complete=True):
    (workspace / REPORT).parent.mkdir(parents=True, exist_ok=True)
    (workspace / PARAMETER).write_bytes(b'original tree parameters\0' * 4096)
    (workspace / OTHER_PARAMETER).write_bytes(b'original linear parameters\0' * 2048)
    report = {'complete': complete, 'aggregation': 'total held-out SSE / held-out rows',
              'folds': [{'train_records': [0, 2], 'validation_records': [1, 3],
                         'candidates': {'tree': {'parameter_path': PARAMETER, 'mse': 2.7},
                                        'linear': {'parameter_path': OTHER_PARAMETER, 'mse': 3.4}}}],
              'candidates': {'tree': {'folds': [{'parameter_path': PARAMETER}]},
                             'incomplete': {'parameter_path': None}}}
    (workspace / REPORT).write_text(json.dumps(report))
    (workspace / RESIDUALS).write_text('[{"record": 1, "residual": 1.25}]\n')
    (workspace / 'shared/comparison/selected_candidate.py').write_text(
        'def predict(row): return 0.\n')
    return report


@pytest.mark.parametrize('restore_entry', ['restore_research', 'restore_practice_research'])
def test_completed_report_references_parameters_without_copying_or_altering_evidence(
        tmp_path, monkeypatch, restore_entry):
    old, lab, previous = checkpoint(tmp_path)
    report = add_comparison(old.workspace)
    original_report = (old.workspace / REPORT).read_bytes()
    parameters = {old.workspace / name for name in (PARAMETER, OTHER_PARAMETER)}
    ordinary_copy = common_archive_inquiry._copy_model_file

    def no_parameter_copy(source, destination):
        assert Path(source) not in parameters, 'completed fold parameters reached the copy writer'
        return ordinary_copy(source, destination)

    monkeypatch.setattr(common_archive_inquiry, '_copy_model_file', no_parameter_copy)
    continued = agent_at(tmp_path / 'continued')
    restored = getattr(continued, restore_entry)(previous, old.workspace,
        TemporalInquiryLab(task_index=1), confirmation_seed=SEED,
        retained_comparison_reports=[REPORT])
    expected = deepcopy(report)
    expected['folds'][0]['candidates']['tree']['parameter_path'] = str(old.workspace / PARAMETER)
    expected['folds'][0]['candidates']['linear']['parameter_path'] = str(old.workspace / OTHER_PARAMETER)
    expected['candidates']['tree']['folds'][0]['parameter_path'] = str(old.workspace / PARAMETER)
    assert json.loads((continued.workspace / REPORT).read_text()) == expected
    assert not list(continued.workspace.rglob('*.joblib'))
    assert (old.workspace / REPORT).read_bytes() == original_report
    assert (continued.workspace / RESIDUALS).read_bytes() == (old.workspace / RESIDUALS).read_bytes()
    assert (continued.workspace / 'shared/comparison/selected_candidate.py').read_bytes() == (
        old.workspace / 'shared/comparison/selected_candidate.py').read_bytes()
    assert continued.observations == old.observations and continued.costs == old.costs
    assert restored['completed_comparison_storage'] == {
        'reports': [REPORT],
        'retained_parameter_files': sorted(str(path) for path in parameters),
        'parameter_bytes_not_copied': sum(path.stat().st_size for path in parameters),
    }


def test_repeated_restore_keeps_references_and_later_local_fit_preserves_old_parameters(tmp_path):
    old, lab, previous = checkpoint(tmp_path)
    add_comparison(old.workspace)
    original_parameters = (old.workspace / PARAMETER).read_bytes()
    continued = agent_at(tmp_path / 'continued')
    continued_lab = TemporalInquiryLab(task_index=1)
    continued.restore_research(previous, old.workspace, continued_lab, confirmation_seed=SEED,
                               retained_comparison_reports=[REPORT])
    continued_result = saved_research(continued, continued_lab)
    again = agent_at(tmp_path / 'again')
    restored = again.restore_research(continued_result, continued.workspace,
        TemporalInquiryLab(task_index=1), confirmation_seed=SEED,
        retained_comparison_reports=[REPORT])
    assert not list(again.workspace.rglob('*.joblib'))
    assert restored['completed_comparison_storage']['parameter_bytes_not_copied'] == 0
    assert (again.workspace / REPORT).read_bytes() == (continued.workspace / REPORT).read_bytes()

    # A repeated authored fit writes its original relative filename, not a link
    # into the paid old scene. Its newly completed report names this new version.
    (continued.workspace / PARAMETER).write_bytes(b'new separately fitted parameters')
    revised = json.loads((continued.workspace / REPORT).read_text())
    revised['folds'][0]['candidates']['tree']['parameter_path'] = PARAMETER
    revised['candidates']['tree']['folds'][0]['parameter_path'] = PARAMETER
    (continued.workspace / REPORT).write_text(json.dumps(revised))
    next_version = agent_at(tmp_path / 'next_version')
    next_version.restore_practice_research(continued_result, continued.workspace,
        TemporalInquiryLab(task_index=1), confirmation_seed=SEED,
        retained_comparison_reports=[REPORT])
    restored_report = json.loads((next_version.workspace / REPORT).read_text())
    assert restored_report['folds'][0]['candidates']['tree']['parameter_path'] == str(
        continued.workspace / PARAMETER)
    assert restored_report['folds'][0]['candidates']['linear']['parameter_path'] == str(
        old.workspace / OTHER_PARAMETER)
    assert not list(next_version.workspace.rglob('*.joblib'))
    assert (old.workspace / PARAMETER).read_bytes() == original_parameters
    assert (continued.workspace / PARAMETER).read_bytes() == b'new separately fitted parameters'


@pytest.mark.parametrize('complete,declared', [(False, True), (True, False)])
def test_unfinished_or_undeclared_comparison_retains_independent_writable_parameters(
        tmp_path, complete, declared):
    old, lab, previous = checkpoint(tmp_path)
    add_comparison(old.workspace, complete=complete)
    original_parameters = (old.workspace / PARAMETER).read_bytes()
    continued = agent_at(tmp_path / 'continued')
    restored = continued.restore_research(previous, old.workspace,
        TemporalInquiryLab(task_index=1), confirmation_seed=SEED,
        retained_comparison_reports=[REPORT] if declared else ())
    assert (continued.workspace / REPORT).read_bytes() == (old.workspace / REPORT).read_bytes()
    assert (continued.workspace / PARAMETER).read_bytes() == original_parameters
    assert not (continued.workspace / PARAMETER).is_symlink()
    (continued.workspace / PARAMETER).write_bytes(b'continue the unfinished fit')
    assert (old.workspace / PARAMETER).read_bytes() == original_parameters
    if declared:
        assert restored['completed_comparison_storage'] == {
            'reports': [], 'retained_parameter_files': [], 'parameter_bytes_not_copied': 0}
    else:
        assert 'completed_comparison_storage' not in restored
