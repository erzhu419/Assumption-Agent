"""The complete old pool is common paid work, then both outputs freeze."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import registered_source_control as driver
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program


def setup(tmp_path, monkeypatch, *, challenger_complete=True, same_model=False):
    from hegel_machine.cognition import experiment_runtime
    from scripts import prepare_pr3_source_successor as factory

    costs = {'compute_cost': 1000, 'observation_cost': 169, 'measurement_cost': 0,
             'python_execution_seconds': 10., 'model_calls': 0, 'reported_tokens': 0}
    cost_file = tmp_path / 'old.json'
    cost_file.write_text(json.dumps(costs))
    limits = {'compute': 1000, 'python_seconds': 100., 'activity_seconds': 100.,
        'native_host_seconds': 30., 'per_unit_compute': 300,
        'max_candidates': 8, 'new_observation_rows': 0}
    registration = {'execution_authorized': True, 'approval_ref': {'file': 'explicit_approval'},
        'maximum_physical_cores': 8,
        'source_scene': {'costs_ref': {'file': str(cost_file), 'json_path': []},
            'unknown_cost_ref': {'file': 'old_unknown_hold'}},
        'source_costs_ref': {'file': str(cost_file), 'json_path': []},
        'material': {'diagnostic_subject': 'original0165'},
        'cells': {name: {'limits': {**limits, 'research_units': int(name == '10')},
                        'scope': {'paid_rows': 193, 'pool_models': 5200,
                                  'max_exports': 0 if name == '00' else 28},
                        'evaluation_ref': {'file': 'frozen_registration', 'json_path': ['evaluation']}}
                  for name in ('00', '10')},
        'components': {'Qeval_predictions': {'compute': 48, 'observations': 24,
            'python_seconds_per_arm': 10., 'activity_seconds_per_arm': 10.}}}
    called, labels = [], []

    def processes(function, jobs, workers):
        assert workers == 8 and len(jobs) == 1
        return [function(job) for job in jobs]

    monkeypatch.setattr(experiment_runtime, 'run_process_trials', processes)
    monkeypatch.setattr(experiment_runtime, '_initialize_worker', lambda cpus: None)

    def cell(single, output):
        name = output.name
        called.append((name, deepcopy(single)))
        assert single['limits']['new_observation_rows'] == 0
        assert single['scope'] == registration['cells'][name]['scope']
        assert single['material']['diagnostic_subject'] == 'original0165'
        if name == '00':
            assert single['limits']['research_units'] == 0
            assert single['limits']['compute'] == 976
            paid = costs
            new, wall, python = 130, 3., 2.
            model = 'actual_strong_pool_winner.py'
        else:
            assert single['limits']['research_units'] == 1
            source = single['source_scene']
            assert source['shared_preparation_ref']['file'].endswith('00/summary.json')
            assert source['training_ref']['file'].endswith('00/workspace/training.json')
            assert single['limits']['compute'] == 1000 - 130 - 24
            assert single['limits']['python_seconds'] == 100. - 2. - 10.
            assert single['limits']['activity_seconds'] == 100. - 3. - 10.
            paid = json.loads(Path(source['costs_ref']['file']).read_text())['costs']
            new, wall, python = 77, 2., 1.
            model = 'actual_repaired_winner.py'
        output.mkdir(parents=True)
        workspace = output / 'workspace'
        workspace.mkdir()
        (workspace / 'training.json').write_text('{}')
        frozen = output / 'frozen'
        frozen.mkdir()
        value = .2 if name == '00' or same_model else .3
        (frozen / model).write_text(export_temporal_program(Program('const', value=value)))
        after = {**paid, 'compute_cost': paid['compute_cost'] + new,
                 'python_execution_seconds': paid['python_execution_seconds'] + python}
        result = {'status': 'completed', 'execution_complete': name == '00' or challenger_complete,
            'frozen_model': {'workspace': str(frozen), 'model': model},
            'costs': after,
            'new_stage_costs': {key: after[key] - paid[key] for key in paid},
            'timing': {'activity_wall_seconds': wall},
            'continuation_state_ref': {'file': str(workspace / 'training.json'), 'json_path': []},
            'continuation_data_ref': {'file': str(workspace / 'training.json'),
                                      'json_path': ['paid_observations']},
            'continuation_costs_ref': {'file': str(output / 'summary.json'), 'json_path': ['costs']}}
        (output / 'summary.json').write_text(json.dumps(result))
        return result

    monkeypatch.setattr(driver, 'run_registered_representation_research', cell)

    def factories(reg, output):
        def prepare():
            manifest = json.loads((output / 'frozen_outputs.json').read_text())
            assert set(manifest['models']) == {'00', '10'}
            assert all(Path(value['workspace'], value['model']).exists()
                       for value in manifest['models'].values())
            assert not manifest['evaluation_labels_opened']
            labels.append('purchased_once')
            path = output / 'shared_evaluation_panel.json'
            path.write_text(json.dumps({'rows': [{'features': {'u': 0.},
                'history': [{'u': 0.}] * index, 'target': .5} for index in range(24)],
                'costs_once': {'observation_cost': 24, 'model_calls': 0}}))
            return {'file': str(path), 'json_path': []}

        def evaluate(name, model, reference, account):
            assert labels == ['purchased_once']
            assert account.config.compute_budget == 2000
            assert model['model'] == ('actual_strong_pool_winner.py' if name == '00'
                                      else 'actual_repaired_winner.py')
            account.costs['compute_cost'] += 24
            account.costs['python_execution_seconds'] += .1
            value = .2 if name == '00' or same_model else .3
            mse = (value - .5) ** 2
            (output / name / 'frozen_evaluation.json').write_text(json.dumps({
                'frozen_model': model,
                'evaluation_rows_ref': {'file': str(output / 'shared_evaluation_panel.json'),
                                        'json_path': ['rows']},
                'evaluation': {'returncode': 0, 'mse': mse, 'predictions': [value] * 24,
                               'python_execution_seconds': .1}}))
            return {'loss': mse, 'coverage': 24, 'completed': False,
                    'quality': {'loss': mse, 'threshold': 1e-10}}

        return {'prepare_evaluation': prepare, 'evaluate_frozen': evaluate}

    monkeypatch.setattr(factory, 'make_retained_method_pairing_factories', factories)
    return registration, called, labels


def test_unauthorized_pair_never_creates_a_run(tmp_path):
    output = tmp_path / 'run'
    with pytest.raises(ValueError, match='approve'):
        driver.run_registered_representation_pair({'execution_authorized': False}, output)
    assert not output.exists()


def test_common_paid_old_pool_is_reused_and_labels_follow_both_actual_frozen_winners(tmp_path, monkeypatch):
    registration, called, labels = setup(tmp_path, monkeypatch)
    result = driver.run_registered_representation_pair(registration, tmp_path / 'run')
    assert result['status'] == 'completed_development_pairing'
    assert [name for name, _ in called] == ['00', '10']
    assert labels == ['purchased_once']
    assert result['costs_once']['compute_cost'] == 130 + 77 + 48
    assert result['costs_once']['observation_cost'] == 24
    assert result['policy_costs_including_shared_preparation']['00']['compute_cost'] == 130 + 24
    assert result['policy_costs_including_shared_preparation']['10']['compute_cost'] == 130 + 77 + 24
    assert result['evaluation_lifecycle']['all_outputs_frozen_before_labels']


def test_incomplete_repair_blocks_shared_purchase_without_erasing_its_paid_work(tmp_path, monkeypatch):
    registration, _, labels = setup(tmp_path, monkeypatch, challenger_complete=False)
    result = driver.run_registered_representation_pair(registration, tmp_path / 'run')
    assert result['status'] == 'development_pairing_incomplete'
    assert labels == [] and not result['evaluation_lifecycle']['labels_revealed']
    assert result['costs_once']['compute_cost'] == 130 + 77
    assert result['costs_once']['observation_cost'] == 0
    failed = json.loads((tmp_path / 'run/10/outcome.json').read_text())
    assert failed['new_stage_costs']['compute_cost'] == 77


def test_failed_shared_acquisition_keeps_unknown_exposure_and_cost_instead_of_zero(tmp_path, monkeypatch):
    from scripts import prepare_pr3_source_successor as factory
    registration, _, labels = setup(tmp_path, monkeypatch)
    original = factory.make_retained_method_pairing_factories

    def failing_factory(reg, output):
        callbacks = original(reg, output)

        def failed_prepare():
            labels.append('partial_acquisition_not_committed')
            raise OSError('synthetic interrupted panel save')

        return {**callbacks, 'prepare_evaluation': failed_prepare}

    monkeypatch.setattr(factory, 'make_retained_method_pairing_factories', failing_factory)
    result = driver.run_registered_representation_pair(registration, tmp_path / 'run')
    assert result['status'] == 'development_pairing_incomplete'
    assert result['costs_once'] is None and result['shared_readout_costs_once'] is None
    assert result['evaluation_lifecycle']['labels_revealed'] is None
    assert labels == ['partial_acquisition_not_committed']
    saved = json.loads((tmp_path / 'run/00/outcome.json').read_text())
    assert saved['new_stage_costs']['compute_cost'] == 130


def test_identical_frozen_model_reuses_prediction_cash_but_attributes_cost_to_both_policies(tmp_path, monkeypatch):
    from scripts import prepare_pr3_source_successor as factory
    registration, _, labels = setup(tmp_path, monkeypatch, same_model=True)
    original = factory.make_retained_method_pairing_factories
    predicted = []

    def counting_factory(reg, output):
        callbacks = original(reg, output)

        def evaluate(name, *args):
            predicted.append(name)
            return callbacks['evaluate_frozen'](name, *args)

        return {**callbacks, 'evaluate_frozen': evaluate}

    monkeypatch.setattr(factory, 'make_retained_method_pairing_factories', counting_factory)
    result = driver.run_registered_representation_pair(registration, tmp_path / 'run')
    assert result['status'] == 'completed_development_pairing'
    assert predicted == ['00'] and labels == ['purchased_once']
    assert result['costs_once']['compute_cost'] == 130 + 77 + 24
    assert result['policy_costs_including_shared_preparation']['00']['compute_cost'] == 130 + 24
    assert result['policy_costs_including_shared_preparation']['10']['compute_cost'] == 130 + 77 + 24
    assert result['policy_costs_including_shared_preparation']['10']['python_execution_seconds'] == pytest.approx(3.1)
    challenger = json.loads((tmp_path / 'run/10/outcome.json').read_text())
    assert challenger['frozen_prediction_costs']['compute_cost'] == 0
    assert challenger['policy_frozen_prediction_costs']['compute_cost'] == 24
    saved = json.loads((tmp_path / 'run/10/frozen_evaluation.json').read_text())
    assert saved['evaluation']['predictions_ref'] == {
        'file': str(tmp_path / 'run/00/frozen_evaluation.json'),
        'json_path': ['evaluation', 'predictions']}
    comparison = json.loads((tmp_path / 'run/pair_00_10.json').read_text())
    assert comparison['frozen_predictive_comparison']['challenger_minus_baseline'] == 0.
