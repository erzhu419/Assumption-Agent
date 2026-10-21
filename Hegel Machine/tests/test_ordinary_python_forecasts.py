"""Common forecasts its complete ordinary catalog without fitting public labels."""

from copy import deepcopy
import json
from pathlib import Path

from hegel_machine.cognition.question_routing import resolved_theories
from hegel_machine.cognition.retained_forecasts import resolve_forecast_references
from test_question_routing import future_question, parent, probe, route, scene, theory


def catalog_scene(tmp_path, *, failing=False):
    agent, lab = scene(tmp_path)
    question = future_question(agent)
    source = agent.workspace / 'archive' / 'batch' / '_evaluated_model'
    source.mkdir(parents=True)
    (source / 'params.json').write_text('0')
    (source / 'model_observations.json').write_text(json.dumps([
        {'features': {'u': 0.}, 'history': [], 'target': 7.}]))
    models = []
    for name, expression in [('linear', "row['features']['u']"),
                             ('square', "row['features']['u'] ** 2")]:
        (source / f'{name}.py').write_text(
            'import json\nfrom pathlib import Path\n'
            'def fit(rows):\n'
            "    assert json.loads(Path('params.json').read_text()) == 0\n"
            "    Path('params.json').write_text(json.dumps(rows[0]['target']))\n"
            'def predict(row):\n'
            f"    return json.loads(Path('params.json').read_text()) + {expression}\n")
        models.append(str((source / f'{name}.py').relative_to(agent.workspace)))
    if failing:
        (source / 'failed.py').write_text("raise ValueError('retained failed predictor')\n")
        models.append(str((source / 'failed.py').relative_to(agent.workspace)))
    agent.knowledge['finite_native_author'] = {'question': deepcopy(question),
        'predictor_paths': models, 'predictor_catalog_ref': {'file': 'catalog.json', 'json_path': []}}
    return agent, lab, question, source, models


def forecasts(agent):
    return [entry for entry in resolved_theories(agent)
            if entry.get('forecast_scope', {}).get('kind') == 'ordinary_python_catalog']


def allocated(root):
    return sum(path.lstat().st_blocks * 512 for path in root.rglob('*') if not path.is_symlink())


def test_common_catalog_forecasts_keep_original_fit_view_shared_input_and_private_writes(tmp_path, monkeypatch):
    import hegel_machine.cognition.code_agent as module

    agent, lab, question, source, models = catalog_scene(tmp_path)
    copies, actual_copy = [], module._copy_execution_context

    def counted(origin, destination, selected):
        copies.append((Path(origin), Path(destination), list(selected)))
        return actual_copy(origin, destination, selected)

    monkeypatch.setattr(module, '_copy_execution_context', counted)
    paid, before_compute = lab.paid, agent.costs['compute_cost']
    first = route(agent, lab, question=question, experiments=[probe(agent)])
    report = first['python_forecasts']
    assert report['status'] == 'ready' and report['model_count'] == report['ready_count'] == 2
    assert report['new_compute_cost'] == 4  # two public rows per original predictor
    assert report['new_python_execution_seconds'] > 0.
    assert agent.costs['compute_cost'] - before_compute == first['new_compute_cost']
    assert first['status'] == 'selected' and first['selection']['scores'][0]['G'] == 1.
    assert [group['task_answer'] for group in first['selection']['groups']] == [10., 16.]
    members = [identifier for group in first['selection']['groups'] for identifier in group['theory_ids']]
    assert members == [entry['id'] for entry in forecasts(agent)]
    assert parent(agent)['pending_query']['actions'] == probe(agent)['actions']
    assert len(copies) == 3 and copies[0][2] == ['linear.py', 'square.py']
    assert len({origin for origin, _, _ in copies[1:]}) == 1
    assert (source / 'params.json').read_text() == '0'
    doc = json.loads((agent.workspace / report['source_ref']['file']).read_text())
    assert len(doc['contexts']) == 1 and len(doc['prediction_inputs']) == 2
    snapshot = Path(doc['contexts'][0]['snapshot_directory'])
    assert (snapshot / 'params.json').read_text() == '0'
    assert all(entry['fitting_view_ref']['file'] == str(snapshot / 'model_observations.json')
               for entry in doc['forecasts'])
    assert all('mse' not in entry and 'loss' not in entry and 'source' not in entry
               and 'files' not in entry for entry in doc['forecasts'])
    assert all('program' not in entry and entry['support']['status'] == 'candidate'
               for entry in agent.theory_index.values())
    assert len(agent.theory_index) == 2 and agent.candidate_paths == []
    assert lab.paid == paid and agent.costs['model_calls'] == 0 and not agent.confirmations
    root = agent.workspace / '_experiment_models'
    before = deepcopy(agent.costs), allocated(root), sorted(str(path) for path in root.rglob('*'))
    again = route(agent, lab, question={'id': question['id']})
    assert again['cache_hit'] and again['python_forecasts']['cache_hit']
    assert again['python_forecasts']['new_compute_cost'] == 0
    assert again['python_forecasts']['new_python_execution_seconds'] == 0
    assert before == (agent.costs, allocated(root), sorted(str(path) for path in root.rglob('*')))
    assert before[1] > 0 and len(copies) == 3


def test_failed_python_column_remains_unknown_in_actual_selection_and_complete_catalog(tmp_path):
    agent, lab, question, _, models = catalog_scene(tmp_path, failing=True)
    result = route(agent, lab, question=question, experiments=[probe(agent)])
    assert result['python_forecasts']['model_count'] == 3
    assert result['python_forecasts']['ready_count'] == 2
    assert result['python_forecasts']['unknown_count'] == 1
    entries = forecasts(agent)
    assert len(entries) == 3 and entries[-1]['forecast_ref']['model'] == models[-1]
    assert entries[-1]['source_status'] == 'unavailable_retained_forecast'
    assert entries[-1]['support']['status'] == 'candidate' and 'compatible' not in entries[-1]
    unknown = [group for group in result['selection']['groups'] if not group['known']]
    assert [group['theory_ids'] for group in unknown] == [[entries[-1]['id']]]
    assert result['selection']['unknown_reason'] == 'incomplete_candidate_predictions'
    assert result['status'] == 'scoped_unknown' and result['selection']['selected_experiment'] is None
    assert agent.costs['observation_cost'] == lab.paid == 2


def test_funded_mission_limits_forecasts_then_resumes_only_unfunded_columns(tmp_path, monkeypatch):
    import hegel_machine.cognition.code_agent as module

    agent, lab, question, _, _ = catalog_scene(tmp_path)
    calls, evaluate = [], module.evaluate_python_model

    def counted(*args, **kwargs):
        calls.append(kwargs.get('research_compute_budget'))
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(module, 'evaluate_python_model', counted)
    mission = {'mission_id': 'small_theory', 'purpose': 'theory_learning', 'allocation': {'compute': 2}}
    before = agent.costs['compute_cost']
    first = route(agent, lab, question=question, experiments=[probe(agent)], mission=mission)
    assert first['python_forecasts']['ready_count'] == first['python_forecasts']['unknown_count'] == 1
    assert first['new_compute_cost'] == agent.costs['compute_cost'] - before == 2
    assert agent.research_mission['resource_usage']['compute'] == 2 and calls == [0]
    costs = deepcopy(agent.costs)
    again = route(agent, lab, question={'id': question['id']})
    assert again['cache_hit'] and agent.costs == costs and calls == [0]
    expanded = {**mission, 'allocation': {'compute': 100}}
    final = route(agent, lab, question={'id': question['id']}, mission=expanded)
    assert final['python_forecasts']['status'] == 'ready'
    assert final['python_forecasts']['new_compute_cost'] == 2 and len(calls) == 2
    assert agent.research_mission['resource_usage']['compute'] == agent.costs['compute_cost'] - before
    assert agent.research_mission['resource_usage']['compute'] <= 100
    assert len(forecasts(agent)) == 2


def test_explicit_roster_is_exact_and_source_change_leaves_original_forecast_readable(tmp_path):
    agent, lab, question, source, _ = catalog_scene(tmp_path)
    original = route(agent, lab, question=question, experiments=[probe(agent)])
    original_entry = deepcopy(forecasts(agent)[0])
    (source / 'model_observations.json').write_text(json.dumps([
        {'features': {'u': 0.}, 'history': [], 'target': 8.}]))
    changed = route(agent, lab, question={'id': question['id']})
    assert not changed['cache_hit'] and changed['python_forecasts']['new_compute_cost'] == 4
    assert changed['python_forecasts']['source_ref'] != original['python_forecasts']['source_ref']
    assert [group['task_answer'] for group in changed['selection']['groups']] == [11., 17.]
    old, = resolve_forecast_references([original_entry], agent.workspace)
    assert old['_retained_forecast']['predictions'] == [10., 9.]
    explicit = future_question(agent, 'explicit')
    explicit['candidate_theory_ids'] = ['declared']
    # Even a catalog explicitly assigned to this question cannot expand its roster.
    agent.knowledge['finite_native_author']['question'] = deepcopy(explicit)
    before = agent.costs['python_execution_seconds']
    result = route(agent, lab, question=explicit, theories=[theory('declared')], experiments=[probe(agent)])
    assert result['python_forecasts']['status'] == 'not_needed'
    assert agent.costs['python_execution_seconds'] == before
    assert [identifier for group in result['selection']['groups'] for identifier in group['theory_ids']] == ['declared']
