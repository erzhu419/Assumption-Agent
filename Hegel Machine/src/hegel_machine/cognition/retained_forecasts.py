"""Resolve saved Python prediction columns without executing their predictors.

The columns are finite execution results, not scientific Programs or new
measurements. Public feature/history identity remains necessary at every read.
"""

from copy import deepcopy
import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

from .models import Observation
from .research_episodes import load_source_values


def resolve_forecast_references(theories, workspace):
    """Read each small source file once, attaching transient columns to copies.

    ``forecast_ref`` names file, json_path and the exact recorded model. File
    size/mtime are retained for the caller's existing cache invalidation.
    Missing or invalid sources remain UNKNOWN rather than model refutations.
    """
    resolved, payloads = deepcopy(theories), {}
    for theory in resolved:
        reference = theory.get('forecast_ref')
        if reference is None:
            continue
        theory.pop('_retained_forecast', None)
        theory.pop('forecast_resolution', None)
        theory['source_status'] = 'unavailable_retained_forecast'
        try:
            if (not isinstance(reference, dict) or not isinstance(reference.get('file'), str)
                    or not isinstance(reference.get('json_path'), list)
                    or not isinstance(reference.get('model'), str) or not reference['model']):
                raise ValueError('complete_forecast_reference_required')
            path = Path(reference['file'])
            path = path if path.is_absolute() else Path(workspace) / path
            key = str(path.resolve())
            if key not in payloads:
                try:
                    stat = path.stat()
                    version = {'file': key, 'stat_size': stat.st_size, 'stat_mtime_ns': stat.st_mtime_ns}
                    loaded, = load_source_values([{'file': key, 'json_path': []}], workspace)
                    payloads[key] = (loaded['value'], version, None)
                except (OSError, ValueError) as error:
                    payloads[key] = (None, {'file': key}, error)
            value, version, error = payloads[key]
            theory['forecast_source_version'] = deepcopy(version)
            if error is not None:
                raise ValueError('retained_forecast_file_unavailable') from error
            for component in reference['json_path']:
                value = value[component]
            if not isinstance(value, dict) or value.get('model') != reference['model']:
                raise ValueError('recorded_forecast_model_mismatch')
            if value.get('returncode', 0) != 0:
                raise ValueError('retained_forecast_execution_failed')
            inputs, predictions = value.get('prediction_inputs'), value.get('predictions')
            if inputs is None and value.get('prediction_inputs_ref'):
                inputs_ref = value['prediction_inputs_ref']
                inputs_path = Path(inputs_ref['file'])
                inputs_path = inputs_path if inputs_path.is_absolute() else Path(workspace) / inputs_path
                inputs_key = str(inputs_path.resolve())
                if inputs_key not in payloads:
                    loaded, = load_source_values([{'file': inputs_key, 'json_path': []}], workspace)
                    payloads[inputs_key] = (loaded['value'], {}, None)
                inputs = payloads[inputs_key][0]
                for component in inputs_ref['json_path']:
                    inputs = inputs[component]
            if (not isinstance(inputs, list) or not isinstance(predictions, list) or not inputs
                    or len(inputs) != len(predictions)):
                raise ValueError('complete_retained_forecast_columns_required')
            if any(not isinstance(row, dict) or not isinstance(row.get('features'), dict)
                   or not isinstance(row.get('history'), list) for row in inputs):
                raise ValueError('retained_forecast_public_rows_required')
            if any(type(value) not in (int, float) or not isfinite(value) for value in predictions):
                raise ValueError('finite_retained_predictions_required')
            theory['_retained_forecast'] = {'model': value['model'],
                'prediction_inputs': [{key: deepcopy(row[key]) for key in ('features', 'history')}
                                      for row in inputs], 'predictions': deepcopy(predictions)}
            theory['source_status'] = 'retained_forecast_ready'
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            theory['forecast_resolution'] = {'status': 'scoped_unknown', 'reason': str(error)}
    return resolved


def retained_forecast_predictor(table):
    """Return an exact public-row lookup; an unrecorded row stays UNKNOWN."""
    def predict(public):
        values = [prediction for row, prediction in zip(table['prediction_inputs'], table['predictions'])
                  if row['features'] == public['features'] and row['history'] == public['history']]
        if not values:
            raise ValueError('public_row_missing_from_retained_forecast')
        if any(value != values[0] for value in values[1:]):
            raise ValueError('conflicting_predictions_for_same_public_row')
        return values[0]

    return predict


def _public_forecast_rows(question, laboratory):
    """Read the already declared public bindings, without looking up labels."""
    from .question_selection import _query_matches_actions

    rows = []

    def collect(query):
        operation = query.get('op')
        if operation == 'Readout':
            if 'finite_readouts' in query:
                for case in query['finite_readouts']:
                    collect(case)
            else:
                binding = query.get('binding') or {}
                if not _query_matches_actions(query, [binding.get('action')], laboratory.prediction_rows):
                    raise ValueError('public_row_query_mismatch')
                row = {'features': deepcopy(binding['features']), 'history': deepcopy(binding['history'])}
                if row not in rows:
                    rows.append(row)
        elif operation == 'Difference':
            collect(query['left'])
            collect(query['right'])
        elif operation in {'CheckApplication', 'RelationResidual'}:
            collect(query['query'])
        elif operation == 'AllOnFiniteSet':
            for case in query['finite_cases']:
                collect(case)
        else:
            raise ValueError('finite_public_prediction_query_required')

    collect(question.get('query_semantics', {}).get('query_expression') or {})
    for plan in question.get('resolution_experiments', []):
        if not _query_matches_actions(plan['query'], plan['actions'], laboratory.prediction_rows):
            raise ValueError('public_row_query_mismatch')
        collect(plan['query'])
        for branch in plan.get('branches', []):
            if not _query_matches_actions(branch['query'], branch['actions'], laboratory.prediction_rows):
                raise ValueError('public_row_query_mismatch')
            collect(branch['query'])
    return rows


def prepare_python_forecasts(agent, question, laboratory, models, *, compute_budget):
    """Freeze the ordinary catalog on this question's finite public inputs.

    Inputs are shared by columns and source contexts by their actual directory.
    Existing fitting hooks run in independent temporary contexts on their saved
    training view. Failed/unfunded columns remain UNKNOWN. No paid score,
    scientific Program, observation or model support is invented here.
    """
    from .code_agent import (_bound_model_contexts, _copy_execution_context,
        evaluate_python_model, python_model_context, python_model_context_matches)
    from .python_research_routing import ordinary_paid_assessment_refs

    preparation_started = monotonic()
    result = {'status': 'not_needed', 'model_count': len(models), 'ready_count': 0,
        'unknown_count': 0, 'new_compute_cost': 0, 'new_python_execution_seconds': 0.,
        'preparation_wall_seconds': 0., 'cache_hit': False, 'theories': [], 'support_updated': False}
    if not models:
        return result
    target = question.get('theory_target') or {}
    roles = target.get('role_signature')
    try:
        if (not isinstance(roles, dict) or target.get('output_unit') is None
                or target.get('context_ref') is None
                or any(not isinstance(role, dict) or any(role.get(key) is None
                       for key in ('role', 'unit', 'time')) for role in roles.values())):
            raise ValueError('typed_python_forecast_contract_required')
        public = _public_forecast_rows(question, laboratory)
        if not public or any(set(row['features']) - set(roles) for row in public):
            raise ValueError('typed_public_forecast_rows_required')
    except (KeyError, TypeError, ValueError) as error:
        return {**result, 'status': 'author_work_request', 'unknown_count': len(models),
            'reason': str(error)}
    basis = {'models': list(models), 'meaning_version': question.get('meaning_version', 1),
        'role_signature': deepcopy(roles), 'output_unit': target['output_unit'],
        'context_ref': deepcopy(target['context_ref']), 'prediction_inputs': public,
        'paid_scope': {'kind': 'available_paid_cutoff_not_fitting_view',
            'records': len(agent.observations), 'rows': len(agent.rows),
            'source_ref': {'file': 'laboratory_observations.json', 'json_path': []}}}
    prior = question.get('python_forecast_execution') or {}
    previous = None
    if prior.get('source_ref'):
        loaded, = load_source_values([prior['source_ref']], agent.workspace)
        previous = loaded['value']
    same = previous is not None and previous['basis'] == basis
    previous_groups = {item['source_relative_directory']: item for item in previous.get('contexts', [])} if same else {}
    groups, errors = {}, {}
    for model in models:
        try:
            directory, inner = python_model_context(agent.workspace, model)
            if not (directory / inner).is_file():
                raise FileNotFoundError(model)
            # The ordinary resolver also accepts frozen external archives.
            # Keep their absolute identity; execution snapshots remain local.
            key = str(directory.relative_to(agent.workspace)
                      if directory.is_relative_to(agent.workspace) else directory)
            group = groups.setdefault(key, {'source_directory': str(directory),
                'source_relative_directory': key, 'models': []})
            group['models'].append(inner)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors[model] = str(error)
    for key, group in groups.items():
        directory = group['source_directory']
        old = previous_groups.get(key)
        unchanged = bool(old and old['models'] == group['models'])
        if unchanged:
            snapshot = Path(old['snapshot_directory'])
            unchanged = python_model_context_matches(directory, group['models'][0], snapshot, group['models'][0])
            checked = set()
            for inner in group['models']:
                for child, name in _bound_model_contexts(Path(directory), inner):
                    if child == Path(directory) or (str(child), name) in checked:
                        continue
                    checked.add((str(child), name))
                    unchanged = unchanged and python_model_context_matches(child, name,
                        snapshot / child.relative_to(directory), name)
        group['unchanged'] = unchanged
        if unchanged:
            group['snapshot_directory'] = old['snapshot_directory']
    old_columns = {item['model']: item for item in previous.get('columns', [])} if same else {}
    assessment_refs = ordinary_paid_assessment_refs(agent, question, models)
    allowance = min(max(0, compute_budget), max(0, agent.config.compute_budget - agent.costs['compute_cost']))
    version = prior.get('version', 0) + 1
    root = Path('_experiment_models') / f'python_forecasts_{agent.questions.index(question):04d}_{basis["meaning_version"]:04d}'
    file = root / f'forecasts_{version:04d}.json'
    doc = {'basis': deepcopy(basis), 'contexts': [], 'prediction_inputs': public, 'forecasts': [], 'columns': []}
    changed, before = False, agent.costs['compute_cost']
    # Thousands of columns can address one file. Share the source reader's
    # existing per-call payload cache across this retained roster.
    old_references = [old_columns[model]['forecast_ref'] for model in models if model in old_columns]
    previous_payloads = {(reference['file'], tuple(reference['json_path'])): loaded['value']
        for reference, loaded in zip(old_references,
            load_source_values(old_references, agent.workspace))} if old_references else {}
    for model in models:
        group = None
        if model not in errors:
            directory, inner = python_model_context(agent.workspace, model)
            group = groups[str(directory.relative_to(agent.workspace)
                               if directory.is_relative_to(agent.workspace) else directory)]
        old = old_columns.get(model)
        reused = None
        if old:
            reference = old['forecast_ref']
            key = (reference['file'], tuple(reference['json_path']))
            recorded = previous_payloads[key]
            unfunded = allowance - (agent.costs['compute_cost'] - before) < len(public)
            retryable = recorded.get('reason') in {'forecast_compute_budget_exhausted',
                                                   'research_compute_budget_exhausted'}
            if ((group and group['unchanged'] and (not retryable or unfunded))
                    or (recorded.get('reason') == 'forecast_compute_budget_exhausted' and unfunded)):
                reused = deepcopy(old)
        if reused:
            doc['columns'].append(reused)
            continue
        changed = True
        entry = {'model': model, 'prediction_inputs_ref': {'file': str(agent.workspace / file),
            'json_path': ['prediction_inputs']},
            'paid_scope': deepcopy(basis['paid_scope'])}
        if model in assessment_refs:
            entry['paid_assessment_ref'] = deepcopy(assessment_refs[model])
        doc['forecasts'].append(entry)
        doc['columns'].append({'model': model, 'forecast_ref': {
            'file': str(file), 'json_path': ['forecasts', len(doc['forecasts']) - 1], 'model': model}})
        available = min(allowance - (agent.costs['compute_cost'] - before),
                        agent.config.compute_budget - agent.costs['compute_cost'])
        if available < len(public):
            entry.update(returncode=1, reason='forecast_compute_budget_exhausted')
            continue
        try:
            if model in errors:
                raise ValueError(errors[model])
            if not group.get('snapshot_directory'):
                snapshot = agent.workspace / root / f'input_{version:04d}_{list(groups).index(group["source_relative_directory"]):04d}' / '_evaluated_model'
                _copy_execution_context(directory, snapshot, group['models'])
                group['snapshot_directory'] = str(snapshot)
            snapshot = Path(group['snapshot_directory'])
            entry['source_context_ref'] = {'workspace': str(snapshot), 'model': inner}
            training = snapshot / 'model_observations.json'
            if not training.is_file():
                training = snapshot / 'observations.json'
            if training.is_file():
                entry['fitting_view_ref'] = {'file': str(training), 'json_path': []}
            remaining = agent.config.execution_seconds - agent.costs['python_execution_seconds']
            if remaining <= 0:
                raise ValueError('Python execution time budget exhausted')
            with TemporaryDirectory(prefix='python-forecast-', dir=agent.workspace / root) as temporary:
                execution = Path(temporary) / '_evaluated_model'
                _copy_execution_context(snapshot, execution, [inner])
                agent.costs['compute_cost'] += len(public)
                worker_started = monotonic()
                try:
                    scored = evaluate_python_model(agent.workspace, str((execution / inner).relative_to(agent.workspace)),
                        [Observation(row['features'], 0., tuple(row['history'])) for row in public],
                        timeout_seconds=remaining, research_compute_budget=max(0, available - len(public)))
                finally:
                    elapsed = monotonic() - worker_started
                    result['new_python_execution_seconds'] += elapsed
                    agent.costs['python_execution_seconds'] += elapsed
                agent.costs['compute_cost'] += scored.get('research_prediction_count', 0)
                entry.update({key: value for key, value in scored.items() if key not in {
                    'mse', 'loss', 'model_dependencies', 'research_bindings', 'representation_diagnostic',
                    'training_representation_diagnostic'}})
        except (OSError, ValueError, TypeError, KeyError) as error:
            entry.update(returncode=1, error=f'{type(error).__name__}: {error}')
    if not changed:
        source_ref = prior['source_ref']
        doc = previous
        result['cache_hit'] = True
    else:
        doc['contexts'] = [{key: value for key, value in group.items() if key != 'unchanged'}
                           for group in groups.values() if group.get('snapshot_directory')]
        agent._save(str(file), json.dumps(doc, ensure_ascii=False))
        source_ref = {'file': str(file), 'json_path': []}
        question['python_forecast_execution'] = {'version': version, 'source_ref': source_ref}
    columns = resolve_forecast_references([{'id': str(index), 'forecast_ref': item['forecast_ref']}
                                         for index, item in enumerate(doc['columns'])], agent.workspace)
    result['ready_count'] = sum('_retained_forecast' in column for column in columns)
    result['unknown_count'] = len(models) - result['ready_count']
    for index, item in enumerate(doc['columns']):
        result['theories'].append({'id': f'python_forecast:{question["id"]}@{basis["meaning_version"]}:{index}',
            'forecast_ref': deepcopy(item['forecast_ref']), 'role_signature': deepcopy(roles),
            'output_unit': target['output_unit'], 'context_ref': deepcopy(target['context_ref']),
            'forecast_scope': {'kind': 'ordinary_python_catalog',
                'question_id': question['id'], 'meaning_version': basis['meaning_version'],
                'paid_scope': deepcopy(basis['paid_scope'])},
            'support': {'status': 'candidate', 'evidence': [], 'independent_confirmation': False}})
    result.update(status='ready' if not result['unknown_count'] else 'scoped_unknown',
        source_ref=source_ref, new_compute_cost=agent.costs['compute_cost'] - before,
        preparation_wall_seconds=monotonic() - preparation_started)
    return result
