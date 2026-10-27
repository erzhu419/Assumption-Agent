"""External paid predictors use the same private forecast contexts as local ones."""

from copy import deepcopy
import json
from pathlib import Path

from hegel_machine.cognition.retained_forecasts import resolve_forecast_references
from test_ordinary_python_forecasts import catalog_scene, forecasts
from test_question_routing import parent, probe, route


def external_scene(tmp_path, *, failing=False):
    agent, lab, question, local, old_models = catalog_scene(tmp_path / 'target', failing=failing)
    source = tmp_path / 'old_paid_archive' / '_evaluated_model'
    source.parent.mkdir()
    local.rename(source)
    models = [str(source / Path(model).name) for model in old_models]
    native = agent.workspace / '_evaluated_model' / 'local.py'
    native.parent.mkdir()
    native.write_text('def predict(row): return -3.\n')
    models.insert(1, str(native.relative_to(agent.workspace)))
    agent.knowledge['finite_native_author']['predictor_paths'] = models
    return agent, lab, question, source, models


def test_external_and_local_roster_predicts_in_order_with_private_fit_and_cache_reuse(tmp_path, monkeypatch):
    import hegel_machine.cognition.code_agent as execution

    agent, lab, question, source, models = external_scene(tmp_path)
    paid = lab.paid
    first = route(agent, lab, question=question, experiments=[probe(agent)])
    report = first['python_forecasts']
    assert report['ready_count'] == report['model_count'] == 3 and report['unknown_count'] == 0
    assert report['new_compute_cost'] == 6
    assert first['status'] == 'selected'
    assert [group['task_answer'] for group in first['selection']['groups']] == [10., -3., 16.]
    assert [entry['forecast_ref']['model'] for entry in forecasts(agent)] == models
    assert parent(agent)['pending_query']['actions'] == probe(agent)['actions']
    document = json.loads((agent.workspace / report['source_ref']['file']).read_text())
    assert [entry['model'] for entry in document['forecasts']] == models
    assert [item['source_relative_directory'] for item in document['contexts']] == [str(source), '_evaluated_model']
    for context in document['contexts']:
        assert Path(context['snapshot_directory']).is_relative_to(agent.workspace)
    external = document['contexts'][0]
    assert (Path(external['snapshot_directory']) / 'params.json').read_text() == '0'
    assert (source / 'params.json').read_text() == '0'  # Each fit wrote only its temporary execution.
    assert all(Path(item['source_context_ref']['workspace']).is_relative_to(agent.workspace)
        for item in document['forecasts'])
    before = deepcopy(agent.costs)

    def forbidden(*args, **kwargs):
        raise AssertionError('retained external forecast must not refit or score again')

    monkeypatch.setattr(execution, 'evaluate_python_model', forbidden)
    second = route(agent, lab, question={'id': question['id']})
    assert second['cache_hit'] and second['python_forecasts']['cache_hit']
    assert agent.costs == before and lab.paid == paid


def test_a_genuine_external_failure_stays_unknown_without_losing_the_roster_or_old_bill(tmp_path):
    agent, lab, question, source, models = external_scene(tmp_path, failing=True)
    first = route(agent, lab, question=question, experiments=[probe(agent)])
    assert first['python_forecasts']['ready_count'] == 3
    assert first['python_forecasts']['unknown_count'] == 1
    assert first['status'] == 'scoped_unknown'
    entries = forecasts(agent)
    assert [item['forecast_ref']['model'] for item in entries] == models
    assert entries[-1]['source_status'] == 'unavailable_retained_forecast'
    unknown = [group for group in first['selection']['groups'] if not group['known']]
    assert [group['theory_ids'] for group in unknown] == [[entries[-1]['id']]]
    original_ref = deepcopy(entries[-1]['forecast_ref'])
    original = (agent.workspace / original_ref['file']).read_text()
    costs = deepcopy(agent.costs)
    second = route(agent, lab, question={'id': question['id']})
    assert second['cache_hit'] and agent.costs == costs
    failed, = resolve_forecast_references([entries[-1]], agent.workspace)
    assert failed['source_status'] == 'unavailable_retained_forecast'
    assert (agent.workspace / original_ref['file']).read_text() == original
    assert (source / 'params.json').read_text() == '0'


def test_changed_external_fit_view_recomputes_only_that_context_and_keeps_old_predictions(tmp_path):
    agent, lab, question, source, models = external_scene(tmp_path)
    first = route(agent, lab, question=question, experiments=[probe(agent)])
    old_entries = deepcopy(forecasts(agent))
    (source / 'model_observations.json').write_text(json.dumps([
        {'features': {'u': 0.}, 'history': [], 'target': 8.}]))
    changed = route(agent, lab, question={'id': question['id']})
    assert not changed['cache_hit'] and changed['python_forecasts']['new_compute_cost'] == 4
    assert [group['task_answer'] for group in changed['selection']['groups']] == [11., -3., 17.]
    assert changed['python_forecasts']['source_ref'] != first['python_forecasts']['source_ref']
    assert [item['forecast_ref']['model'] for item in forecasts(agent)] == models
    old = resolve_forecast_references(old_entries, agent.workspace)
    assert [item['_retained_forecast']['predictions'] for item in old] == [[10., 9.], [-3., -3.], [16., 11.]]
    assert (source / 'params.json').read_text() == '0'
