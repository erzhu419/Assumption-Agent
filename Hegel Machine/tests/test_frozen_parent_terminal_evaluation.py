"""Frozen terminal readout reuses the evaluator without research or refitting."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import experiment_runtime, registered_source_control as control
from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from scripts import run_anonymous_revision_inquiry as study


def fixture_registration(tmp_path):
    models = {}
    for arm, program in [('remove', Program('const', value=0.)),
                         ('keep', Program('feature', name='u'))]:
        workspace = tmp_path / 'models' / arm
        workspace.mkdir(parents=True)
        (workspace / 'model.py').write_text(export_temporal_program(program))
        models[arm] = {'workspace': str(workspace), 'model': 'model.py'}
    paid = tmp_path / 'laboratory_observations.json'
    paid.write_text(json.dumps([{'rows': [asdict(Observation({'u': 1.}, -100., ())) ]}]))
    return {'execution_admitted': True, 'models': models,
        'evaluation': {'case': 'synthetic', 'kind': 'hysteresis', 'seed': 13,
            'evaluation_seed': 17, 'rows': 512, 'trajectory_length': 48, 'mse_threshold': 1e-10},
        'paid_observations_ref': {'file': str(paid), 'json_path': []},
        'exposure_audit_ref': {'file': 'bounded_synthetic_audit.json'},
        'prior_costs_ref': {'file': 'retained_costs.json'}, 'execution_timeout_seconds': 30.}


def synthetic_rows():
    rows = []
    while len(rows) < 512:
        history = []
        for _ in range(min(48, 512-len(rows))):
            rows.append(Observation({'u': 1.}, 1., tuple(deepcopy(history))))
            history.append({'u': 1.})
    return rows


def test_frozen_pair_generates_one_batch_once_and_preserves_models_and_costs(tmp_path, monkeypatch):
    registration = fixture_registration(tmp_path)
    output = tmp_path / 'terminal'
    generation, model_calls, workers_used = [], [], []
    originals = {arm: (Path(model['workspace']) / model['model']).read_bytes()
                 for arm, model in registration['models'].items()}

    def evaluation_rows(self, *, length, seed):
        assert json.loads((output / 'registration.json').read_text()) == registration
        assert length == 512 and seed == 17
        generation.append((length, seed))
        return synthetic_rows()

    evaluate = study.evaluate_python_model
    def counted(workspace, model, rows, **kwargs):
        model_calls.append((workspace, model, rows))
        return evaluate(workspace, model, rows, **kwargs)

    def serial(train, jobs, workers):
        workers_used.append(workers)
        return [train(job) for job in jobs]

    monkeypatch.setattr(AnonymousHistoryLab, 'evaluation_rows', evaluation_rows)
    monkeypatch.setattr(AnonymousHistoryLab, 'execute', lambda *args: pytest.fail('no acquisition'))
    monkeypatch.setattr(AnonymousHistoryLab, 'confirmation_actions', lambda *args: pytest.fail('no confirmation'))
    monkeypatch.setattr(study, 'evaluate_python_model', counted)
    monkeypatch.setattr(experiment_runtime, 'run_process_trials', serial)
    result = control.run_frozen_parent_evaluation(registration, output, workers=8)
    assert generation == [(512, 17)] and len(model_calls) == 2
    assert model_calls[0][2] is model_calls[1][2]
    assert workers_used == [8]
    assert result['status'] == 'completed'
    assert result['arms']['remove']['evaluation']['mse'] == 1.
    assert result['arms']['keep']['evaluation']['mse'] == 0.
    assert not result['arms']['remove']['evaluation']['passed']
    assert result['arms']['keep']['evaluation']['passed']
    assert result['pair']['trajectory_mse_keep_minus_remove'] == [-1.] * 11
    assert result['pair']['status'] == 'fully_compared'
    assert result['pair']['compared_trajectories'] == 11
    assert result['pair']['unscored_trajectories'] == 0
    assert result['arms']['keep']['regions']['history_lt24']['rows'] == 264
    assert result['arms']['keep']['regions']['history_ge24']['rows'] == 248
    assert result['arms']['keep']['regions']['seen_public_inputs']['rows'] == 11
    assert result['arms']['keep']['regions']['unseen_public_inputs']['rows'] == 501
    assert result['costs']['compute_cost'] == result['costs']['scored_rows'] == 1024
    assert result['costs']['terminal_observations_once'] == 512
    assert result['costs']['new_fits'] == result['costs']['model_calls'] == 0
    assert result['costs']['prior_costs_ref'] == registration['prior_costs_ref']
    assert result['research_selection'] == result['fresh_confirmation'] == 'not_performed'
    for arm, model in registration['models'].items():
        assert (Path(model['workspace']) / model['model']).read_bytes() == originals[arm]
        saved = json.loads((output / arm / 'terminal_predictions.json').read_text())
        assert len(saved['predictions']) == 512
        assert 'predictions' not in result['arms'][arm]['evaluation']
    assert len(json.loads((output / 'terminal_rows.json').read_text())) == 512
    resources = json.loads((output / 'resources.json').read_text())
    assert resources['allocated_output_bytes_before_resource_receipt'] > 0
    assert resources['disk_scope'] == 'allocated_output_file_blocks_not_net_filesystem_growth'


def test_failed_frozen_execution_retains_quote_and_elapsed_time_without_retry(tmp_path, monkeypatch):
    registration = fixture_registration(tmp_path)
    monkeypatch.setattr(AnonymousHistoryLab, 'evaluation_rows', lambda *args, **kwargs: synthetic_rows())
    monkeypatch.setattr(experiment_runtime, 'run_process_trials',
                        lambda train, jobs, workers: [train(job) for job in jobs])
    attempts = []
    def evaluate(workspace, model, rows, **kwargs):
        attempts.append(Path(workspace).name)
        if Path(workspace).name == 'keep':
            raise ValueError('synthetic execution failed')
        return {'returncode': 0, 'mse': 1., 'predictions': [0.] * len(rows)}
    monkeypatch.setattr(study, 'evaluate_python_model', evaluate)
    result = control.run_frozen_parent_evaluation(registration, tmp_path / 'terminal', workers=1)
    assert attempts == ['remove', 'keep']
    assert result['status'] == 'completed_with_failures'
    assert result['costs']['compute_cost'] == 1024
    assert result['costs']['scored_rows'] is None
    assert result['costs']['known_scored_rows'] == 512
    assert result['costs']['requested_rows'] == result['costs']['attempted_rows'] == 1024
    failed_costs = result['arms']['keep']['costs']
    assert failed_costs['scored_rows'] is None
    assert failed_costs['requested_rows'] == failed_costs['attempted_rows'] == 512
    assert failed_costs['compute_cost'] == 512
    assert failed_costs['python_execution_seconds'] > 0
    assert result['arms']['keep']['trajectories'][0]['status'] == 'unscored'
    assert result['costs']['python_execution_seconds'] > 0
    assert result['pair']['mse_keep_minus_remove'] is None
    assert result['pair']['trajectory_mse_keep_minus_remove'] == [None] * 11
    assert result['pair']['status'] == 'incomplete'
    assert result['pair']['compared_trajectories'] == 0
    assert result['pair']['unscored_trajectories'] == 11
    assert 'synthetic execution failed' in result['arms']['keep']['evaluation']['error']
    assert not (tmp_path / 'terminal/keep/terminal_predictions.json').exists()


@pytest.mark.parametrize('change', ['unadmitted', 'threshold', 'fit'])
def test_invalid_or_unfrozen_registration_stops_before_terminal_labels(tmp_path, monkeypatch, change):
    registration = fixture_registration(tmp_path)
    if change == 'unadmitted':
        registration['execution_admitted'] = False
    elif change == 'threshold':
        registration['evaluation']['mse_threshold'] = .1
    else:
        Path(registration['models']['keep']['workspace'], 'model.py').write_text(
            'def fit(rows):\n    raise ValueError("must not fit")\ndef predict(row):\n    return 0.\n')
    monkeypatch.setattr(AnonymousHistoryLab, 'evaluation_rows',
                        lambda *args, **kwargs: pytest.fail('labels must remain unopened'))
    with pytest.raises(ValueError):
        control.run_frozen_parent_evaluation(registration, tmp_path / 'terminal')
    assert not (tmp_path / 'terminal/terminal_rows.json').exists()


def test_frozen_cli_dispatches_only_terminal_entry(tmp_path, monkeypatch, capsys):
    from scripts import prepare_pr3_source_successor as entry
    registration = fixture_registration(tmp_path)
    spec = tmp_path / 'registration.json'
    spec.write_text(json.dumps(registration))
    output = tmp_path / 'terminal'
    calls = []
    def run(registered, destination, *, workers):
        calls.append((registered, destination, workers))
        return {'status': 'completed'}
    monkeypatch.setattr(control, 'run_frozen_parent_evaluation', run)
    monkeypatch.setattr(entry, 'prepare', lambda *args: pytest.fail('must not start training'))
    monkeypatch.setattr(entry.sys, 'argv', ['prepare_pr3_source_successor.py',
        '--stage', 'frozen-parent-evaluation', '--spec', str(spec), '--output', str(output)])
    entry.main()
    assert calls == [(registration, output, 8)]
    assert json.loads(capsys.readouterr().out)['status'] == 'completed'
