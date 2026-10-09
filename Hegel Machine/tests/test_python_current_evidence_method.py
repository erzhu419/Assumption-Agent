"""The integrated program chooses repairs from real current-record validation."""

import json
from pathlib import Path

from hegel_machine.cognition import python_current_evidence_method
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from test_python_method_learning import archived_predictor, publish
from test_python_scoped_method_learning import make_scoped_learner


def test_current_evidence_program_preserves_reuse_and_selects_both_real_repairs(
        tmp_path, archived_predictor):
    # Reuse the numerical five-case fixture, but execute the actual implementation
    # source rather than its separate test-only research procedure.
    source = Path(python_current_evidence_method.__file__).read_text()
    archive = Path(archived_predictor['snapshot_workspace'])
    original_files = {str(path.relative_to(archive)): path.read_bytes()
                      for path in archive.rglob('*') if path.is_file()}
    learner, prepared = make_scoped_learner(tmp_path / 'learner', archived_predictor)
    learner._execute('write', {'path': METHOD_PATH, 'code': source}, learner.interface)
    outcome = publish(learner)

    assert outcome.get('published') == METHOD_PATH, outcome
    assert len(prepared) == 5
    coverage = outcome['current_source_coverage']
    assert coverage['missing_cases'] == []
    assert coverage['prior_reuse_cases'] == ['alpha']
    assert coverage['scoped_revision'] == {
        'zero_meter_scoped_recovery_cases': ['delta'],
        'measured_recovery_cases': ['epsilon'], 'distinct_cases': True}

    trials = {trial['case']: trial for trial in learner.practice_trials}
    for case, trial in trials.items():
        assert trial['status'] == 'completed', (case, trial)
        assert trial['method_source'] == source
        assert trial['costs']['model_calls'] == trial['costs']['reported_tokens'] == 0
    assert trials['alpha']['selected_prior_predictor']
    assert trials['alpha']['costs']['measurement_cost'] == 0
    assert trials['beta']['costs']['measurement_cost'] == 0
    assert trials['gamma']['costs']['measurement_cost'] > 0

    scoped = trials['delta']['scoped_revision']['zero_meter_scoped_recoveries'][0]
    assert trials['delta']['costs']['measurement_cost'] == 0
    assert scoped['validation_mse'] < 1e-10 < scoped['fresh_mse']
    assert scoped['refit_validation_mse'] < 1e-10
    assert scoped['recovery_mse'] < 1e-10
    assert scoped['records'] == [4, 5]
    assert scoped['validation_records'] == scoped['new_validation_records'] == [5]
    assert scoped['rejection_step'] < scoped['refit_step'] < scoped['recovery_step']

    changed = trials['epsilon']
    measured = changed['scoped_revision']['measured_recoveries'][0]
    trace = json.loads((learner.workspace / changed['directory'] / 'result.json').read_text())
    current_refits = [(index, feedback) for index, step in enumerate(trace['method_results'])
                      for feedback in [step['feedback'],
                                       *step['feedback'].get('after_observation_results', [])]
                      if feedback.get('operator') == 'refit'
                      and feedback.get('split', {}).get('records') == [4, 5]]
    assert len(current_refits) == 1
    refit_step, rejected_fit = current_refits[0]
    assert rejected_fit['status'] == 'refit_inadequate_on_validation'
    assert rejected_fit['validation']['mse'] > 1e-10
    assert rejected_fit['split']['validation_records'] == [5]
    assert measured['rejection_step'] < refit_step < measured['instrument_step']
    assert measured['instrument_step'] < measured['response_step'] < measured['recovery_step']
    first_meter = next(index for index, record in enumerate(trace['paid_observations'])
                       if record.get('measurements'))
    assert first_meter >= rejected_fit['paid_records_at_refit']
    assert measured['recovery_mse'] < 1e-10

    # The same archive remains reusable after all five incompatible target tasks.
    assert original_files == {str(path.relative_to(archive)): path.read_bytes()
                              for path in archive.rglob('*') if path.is_file()}
    assert learner.costs['model_calls'] == learner.costs['reported_tokens'] == 0
