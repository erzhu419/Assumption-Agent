"""Saved Python predictions join PR2 without executing or inventing Programs."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.question_selection import select_question_plan
from hegel_machine.cognition.retained_forecasts import resolve_forecast_references
from hegel_machine.cognition.theory_matching import prepare_theories_for_question
from test_question_routing import (CONTEXT, future_question, parent, probe, route,
                                    scene, theory)


def columns(tmp_path):
    rows = [{'features': {'u': float(value)}, 'history': []} for value in range(4)]
    forecasts = [{'model': 'frozen/linear.py', 'returncode': 0,
                  'prediction_inputs': deepcopy(rows), 'predictions': [0., 1., 2., 3.]},
                 {'model': 'frozen/square.py', 'returncode': 0,
                  'prediction_inputs': deepcopy(rows), 'predictions': [0., 1., 4., 9.]}]
    path = tmp_path / 'saved_forecasts.json'
    path.write_text(json.dumps({'experiment_comparisons': [{'forecasts': forecasts}]}))
    declarations = []
    for index, forecast in enumerate(forecasts):
        entry = theory(('linear', 'square')[index])
        entry.pop('program')
        entry['forecast_ref'] = {'file': str(path),
            'json_path': ['experiment_comparisons', 0, 'forecasts', index], 'model': forecast['model']}
        declarations.append(entry)
    return path, forecasts, declarations


def prepare(agent, declarations):
    question = future_question(agent)
    resolved = resolve_forecast_references(declarations, agent.workspace)
    return question, resolved, prepare_theories_for_question(question, resolved, [])


def test_same_file_columns_load_once_without_changing_declarations_or_copying_python(tmp_path, monkeypatch):
    import hegel_machine.cognition.retained_forecasts as module

    _, _, declarations = columns(tmp_path)
    original = deepcopy(declarations)
    calls, load = [], module.load_source_values

    def counted(refs, workspace):
        calls.append(deepcopy(refs))
        return load(refs, workspace)

    monkeypatch.setattr(module, 'load_source_values', counted)
    resolved = resolve_forecast_references(declarations, tmp_path)
    assert len(calls) == 1
    assert declarations == original
    assert all(entry['source_status'] == 'retained_forecast_ready' for entry in resolved)
    assert all('program' not in entry and 'python_source' not in entry for entry in resolved)
    assert all(entry['forecast_source_version']['stat_size'] > 0 for entry in resolved)


def test_resolved_columns_use_real_target_queries_original_groups_and_public_panel_validation(tmp_path):
    agent, lab = scene(tmp_path)
    _, _, declarations = columns(tmp_path)
    question, resolved, prepared = prepare(agent, declarations)
    assert not prepared['blocked'] and len(prepared['theories']) == 2
    assert all(item['structural_match'] == item['confirmation'] == 'not_established'
               for item in prepared['prepared'])
    result = select_question_plan(question, prepared['theories'], [probe(agent)],
        context=CONTEXT, public_prediction_rows=lab.prediction_rows)
    assert result['status'] == 'selected' and result['scores'][0]['G'] == 1.
    assert [group['task_answer'] for group in result['groups']] == [3., 9.]
    assert result['public_row_validation']['validated_experiments'] == ['distinguishing_probe']
    assert result['support_updated'] is False and result['evidence'] == []
    assert all('program' not in entry for entry in prepared['theories'])
    assert resolved[0]['support']['status'] == 'candidate'


@pytest.mark.parametrize('changed', ['model', 'failed', 'length', 'missing_file'])
def test_unavailable_column_preserves_unknown_without_refuting_its_scientific_model(tmp_path, changed):
    path, forecasts, declarations = columns(tmp_path)
    if changed == 'model':
        declarations[1]['forecast_ref']['model'] = 'another/model.py'
    elif changed == 'missing_file':
        declarations[1]['forecast_ref']['file'] = str(tmp_path / 'absent.json')
    else:
        if changed == 'failed':
            forecasts[1]['returncode'] = 1
        else:
            forecasts[1]['predictions'].pop()
        path.write_text(json.dumps({'experiment_comparisons': [{'forecasts': forecasts}]}))
    agent, lab = scene(tmp_path / 'workspace')
    question, resolved, prepared = prepare(agent, declarations)
    assert resolved[0]['source_status'] == 'retained_forecast_ready'
    assert resolved[1]['source_status'] == 'unavailable_retained_forecast'
    assert resolved[1]['forecast_resolution']['status'] == 'scoped_unknown'
    assert 'compatible' not in resolved[1] and resolved[1]['support']['status'] == 'candidate'
    assert prepared['blocked'][0]['reason'] == 'retained_forecast_unavailable'
    result = select_question_plan(question, resolved, [probe(agent)], context=CONTEXT)
    assert result['status'] == 'scoped_unknown' and result['selected_experiment'] is None


@pytest.mark.parametrize('changed', ['missing_row', 'conflict'])
def test_exact_public_lookup_never_fills_or_averages_unstored_predictions(tmp_path, changed):
    path, forecasts, declarations = columns(tmp_path)
    if changed == 'missing_row':
        forecasts[1]['prediction_inputs'].pop()
        forecasts[1]['predictions'].pop()
    else:
        forecasts[1]['prediction_inputs'].append(deepcopy(forecasts[1]['prediction_inputs'][-1]))
        forecasts[1]['predictions'].append(99.)
    path.write_text(json.dumps({'experiment_comparisons': [{'forecasts': forecasts}]}))
    agent, lab = scene(tmp_path / 'workspace')
    question, _, prepared = prepare(agent, declarations)
    result = select_question_plan(question, prepared['theories'], [probe(agent)], context=CONTEXT)
    assert result['status'] == 'scoped_unknown'
    assert result['unknown_reason'] == 'incomplete_candidate_predictions'
    assert any(not group['known'] for group in result['groups'])
    assert result['selected_experiment'] is None


@pytest.mark.parametrize('changed', ['context', 'unit', 'role', 'mapping'])
def test_saved_numeric_columns_cannot_silently_change_measurement_contract(tmp_path, changed):
    agent, _ = scene(tmp_path)
    _, _, declarations = columns(tmp_path)
    question = future_question(agent)
    if changed == 'context':
        question['context_ref'] = {'scene': 'other'}
    elif changed == 'unit':
        question['theory_target']['output_unit'] = 'mV'
    elif changed == 'role':
        question['theory_target']['role_signature']['u']['time'] = 'past'
    else:
        question['allowed_mappings'] = [{'feature_map': {'u': 'x'}, 'status': 'tested'}]
    prepared = prepare_theories_for_question(question,
        resolve_forecast_references(declarations, tmp_path), [])
    assert prepared['theories'] == [] and len(prepared['blocked']) == 2
    assert all(item['required_work']['type'] == 'AuthorWorkRequest' for item in prepared['blocked'])


def test_changed_public_panel_row_is_rejected_before_saved_predictions_are_compared(tmp_path):
    agent, lab = scene(tmp_path)
    _, _, declarations = columns(tmp_path)
    question, _, prepared = prepare(agent, declarations)
    plan = probe(agent)
    plan['query']['binding']['features']['u'] = 3.
    result = select_question_plan(question, prepared['theories'], [plan],
        context=CONTEXT, public_prediction_rows=lab.prediction_rows)
    assert result['status'] == 'scoped_unknown'
    assert result['excluded'][0]['reason'] == 'public_row_query_mismatch'


def test_identity_mapping_metadata_keeps_the_saved_measurement_contract(tmp_path):
    agent, _ = scene(tmp_path)
    _, _, declarations = columns(tmp_path)
    question = future_question(agent)
    question['allowed_mappings'] = [{'feature_map': {'u': 'u'}, 'role_map': {'drive': 'drive'},
        'unit_conversions': {'u': {'from': 'V', 'to': 'V', 'scale': 1., 'offset': 0.}},
        'output_conversion': {'from': 'V', 'to': 'V', 'scale': 1., 'offset': 0.},
        'status': 'tested', 'test_evidence': [{'file': 'already_paid.json', 'json_path': [0]}]}]
    prepared = prepare_theories_for_question(question,
        resolve_forecast_references(declarations, tmp_path), [])
    assert len(prepared['theories']) == 2 and not prepared['blocked']


def test_common_route_keeps_only_refs_and_invalidates_cache_when_source_column_changes(tmp_path, monkeypatch):
    agent, lab = scene(tmp_path)
    path, forecasts, declarations = columns(tmp_path)
    question = future_question(agent)
    question['candidate_theory_ids'] = ['linear', 'square']
    paid = lab.paid

    def no_python(*args, **kwargs):
        raise AssertionError('a retained column must never execute a Python predictor')

    monkeypatch.setattr(agent, '_python', no_python)
    first = route(agent, lab, question=question, theories=declarations, experiments=[probe(agent)])
    assert first['status'] == 'selected'
    assert all('program' not in item and '_retained_forecast' not in item
               for item in agent.theory_index.values())
    assert not agent.theory_consolidation.get('candidates')
    costs = deepcopy(agent.costs)
    again = route(agent, lab, question={'id': question['id']})
    assert again['cache_hit'] and agent.costs == costs
    forecasts[1]['predictions'][-1] = 8.
    path.write_text(json.dumps({'experiment_comparisons': [{'forecasts': forecasts}]}))
    changed = route(agent, lab, question={'id': question['id']})
    assert not changed['cache_hit']
    assert [group['task_answer'] for group in changed['selection']['groups']] == [3., 8.]
    assert parent(agent)['pending_query']['actions'] == probe(agent)['actions']
    assert lab.paid == paid and agent.costs['model_calls'] == 0
