"""Preparing retained columns batches one real reader call per prior roster."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hegel_machine.cognition import code_agent
from hegel_machine.cognition import retained_forecasts as module


def test_unchanged_forecast_roster_reads_once_in_order_without_prediction_or_cost(tmp_path, monkeypatch):
    public = [{'features': {'u': .25}, 'history': []}]
    roles = {'u': {'role': 'control', 'unit': 'dimensionless', 'time': 'current'}}
    source, snapshot = tmp_path / '_evaluated_model', tmp_path / 'retained_context'
    source.mkdir()
    snapshot.mkdir()
    names = [f'candidate_{index:03d}.py' for index in range(16)]
    models = [str(Path('_evaluated_model') / name) for name in names]
    for name in names:
        text = 'def predict(row): raise AssertionError("paid predictor must not rerun")\n'
        (source / name).write_text(text)
        (snapshot / name).write_text(text)
    basis = {'models': models, 'meaning_version': 1, 'role_signature': roles,
        'output_unit': 'dimensionless', 'context_ref': {'apparatus': 'public_fixture'},
        'prediction_inputs': public, 'paid_scope': {'kind': 'available_paid_cutoff_not_fitting_view',
            'records': 0, 'rows': 0, 'source_ref': {'file': 'laboratory_observations.json', 'json_path': []}}}
    records = [{'model': model, 'prediction_inputs_ref': {'file': 'saved_forecasts.json',
        'json_path': ['prediction_inputs']}, 'returncode': 0, 'predictions': [float(index)]}
        for index, model in enumerate(models)]
    records[-1] = {**records[-1], 'returncode': 1,
        'reason': 'forecast_compute_budget_exhausted'}
    records[-1].pop('predictions')
    columns = [{'model': model, 'forecast_ref': {'file': 'saved_forecasts.json',
        'json_path': ['forecasts', index], 'model': model}} for index, model in enumerate(models)]
    doc = {'basis': basis, 'contexts': [{'source_relative_directory': '_evaluated_model',
        'models': names, 'snapshot_directory': str(snapshot)}],
        'prediction_inputs': public, 'forecasts': records, 'columns': columns}
    saved = tmp_path / 'saved_forecasts.json'
    saved.write_text(json.dumps(doc))
    original_text = saved.read_text()
    question = {'id': 'retained_prediction', 'meaning_version': 1,
        'theory_target': {'role_signature': roles, 'output_unit': basis['output_unit'],
            'context_ref': basis['context_ref']}, 'python_forecast_execution': {
                'version': 1, 'source_ref': {'file': 'saved_forecasts.json', 'json_path': []}}}
    agent = SimpleNamespace(workspace=tmp_path, questions=[question], observations=[], rows=[],
        method_results=[], config=SimpleNamespace(compute_budget=17, execution_seconds=1.),
        costs={'compute_cost': 17, 'python_execution_seconds': .25, 'observation_cost': 0})
    before = deepcopy(agent.costs)
    load, read = module.load_source_values, Path.read_text
    batches, payload_reads = [], []

    def load_batch(refs, workspace):
        batches.append(deepcopy(refs))
        return load(refs, workspace)

    def counted_read(path, *args, **kwargs):
        if path == saved:
            payload_reads.append(path)
        return read(path, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail('unchanged paid forecast triggered new fitting, scoring or context copying')

    monkeypatch.setattr(module, '_public_forecast_rows', lambda *args: deepcopy(public))
    monkeypatch.setattr(module, 'load_source_values', load_batch)
    monkeypatch.setattr(Path, 'read_text', counted_read)
    monkeypatch.setattr(code_agent, 'evaluate_python_model', forbidden)
    monkeypatch.setattr(code_agent, '_copy_execution_context', forbidden)
    result = module.prepare_python_forecasts(agent, question, None, models, compute_budget=0)
    assert [len(refs) for refs in batches] == [1, len(models), 1]
    assert batches[1] == [column['forecast_ref'] for column in columns]
    assert len(payload_reads) == 3  # prior metadata, one addressed batch, final table resolution
    assert result['cache_hit'] and result['status'] == 'scoped_unknown'
    assert result['ready_count'] == len(models) - 1 and result['unknown_count'] == 1
    assert [entry['forecast_ref'] for entry in result['theories']] == [column['forecast_ref'] for column in columns]
    assert result['new_compute_cost'] == result['new_python_execution_seconds'] == 0
    assert agent.costs == before and read(saved) == original_text
    assert not (tmp_path / '_experiment_models').exists()


def test_restored_common_scene_reuses_old_context_columns_and_keeps_failure_unknown(tmp_path, monkeypatch):
    from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
    from test_inquiry_revision import PublicLab
    from test_ordinary_python_forecasts import catalog_scene
    from test_question_routing import probe, route

    prior, lab, question, source, models = catalog_scene(tmp_path / 'original', failing=True)
    first = route(prior, lab, question=question, experiments=[probe(prior)])
    assert first['python_forecasts']['ready_count'] == 2
    assert first['python_forecasts']['unknown_count'] == 1
    owner = next(item for item in prior.questions if item['id'] == question['id'])
    forecast_file = prior.workspace / owner['python_forecast_execution']['source_ref']['file']
    document = json.loads(forecast_file.read_text())
    declarations = [{'id': str(index), 'forecast_ref': column['forecast_ref']}
        for index, column in enumerate(document['columns'])]
    original_columns = module.resolve_forecast_references(declarations, prior.workspace)
    prior._prepare_program_state()
    (prior.workspace / 'research_context.json').write_text(json.dumps(prior._request(lab)))
    previous = prior.result('researching')
    restored = CommonArchiveInquiryAgent(None, tmp_path / 'restored', arm=prior.arm,
        source_task=False, config=deepcopy(prior.config))
    fresh_lab = PublicLab()
    fresh_lab.costs = {'observation_cost': 0}
    restored.restore_research(previous, prior.workspace, fresh_lab, confirmation_seed=0)
    restored_owner = next(item for item in restored.questions if item['id'] == question['id'])
    before = deepcopy(restored.costs)

    def forbidden(*args, **kwargs):
        pytest.fail('restored paid forecast triggered new fitting, scoring or copying')

    monkeypatch.setattr(code_agent, 'evaluate_python_model', forbidden)
    monkeypatch.setattr(code_agent, '_copy_execution_context', forbidden)
    result = module.prepare_python_forecasts(restored, restored_owner, fresh_lab, models, compute_budget=0)
    assert result['cache_hit'] and result['ready_count'] == 2 and result['unknown_count'] == 1
    assert result['new_compute_cost'] == result['new_python_execution_seconds'] == 0
    assert [item['forecast_ref'] for item in result['theories']] == [item['forecast_ref'] for item in declarations]
    restored_columns = module.resolve_forecast_references(declarations, restored.workspace)
    assert [item.get('_retained_forecast') for item in restored_columns] == [
        item.get('_retained_forecast') for item in original_columns]
    assert restored_columns[-1]['source_status'] == 'unavailable_retained_forecast'
    assert restored_columns[-1]['forecast_resolution'] == original_columns[-1]['forecast_resolution']
    assert restored.costs == before and fresh_lab.paid == 0
    assert (source / 'params.json').read_text() == '0'
    assert not (restored.workspace / forecast_file.relative_to(prior.workspace).parent / 'forecasts_0002.json').exists()
