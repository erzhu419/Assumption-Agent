"""Execute authored representation/experiment revisions in the existing inquiry loop."""

from copy import deepcopy
import json
import os
from pathlib import Path
from statistics import pvariance
import sys
from time import monotonic

from .code_agent import (
    evaluate_python_model, python_model_context, retain_evaluated_python_model,
)
from .language import Program
from .models import Candidate, Probe, SearchBudget
from .probing import choose_probe, value_forecast_experiments


REVISION_INTERFACE = """
A query or select_actions result may include revision:{id:'r1',
affected_object:'representation', models:['old.py','new.py'], readout:'readout.py',
target:'response', parent_and_scope:{...}, ...}. Supply the actual experiment
through the usual actions. Additional proposal fields are retained as authored. Each named model
joins the common comparison on already paid rows; the better executed old model
is preserved. Before buying these actions, the runtime executes every proposed
predictor on their public control histories and saves the predictions, source
and paid-evidence cutoff. Optional fit sees only evidence available before the
experiment. After purchase, prospective MSE uses these saved predictions, never
a refit using the new labels. This diagnostic is separate from fresh finish.
Optional readout defines readout(result,public_spec,memory). result includes the
proposal, prior_fit, forecasts, observations and record_indices. Each forecast
has model, predictions, prediction_inputs and, after acquisition, prospective_mse.
Return {status:'supported'|'refuted'|'inconclusive',answer:...,evidence:[row IDs]}.
The answer is authored analysis, not a runtime claim of truth. It returns to the
query question and parent; a negative child answer can inform further research.
Queries returned by compare_experiments name saved predictors with their own
fitting contexts. proposal.original_models maps those executable paths back to
the originally offered paths; later paid labels do not change their fitting view.
research_state.json exposes revisions and questions, including returned answers,
to the next program invocation. All arms may use this same interface. Failing
forecasts and analyses remain recorded. No physical mechanism is supplied.
For a purchased channel, set target to its public name and include that channel
in every action's channels. Candidate fit and prospective MSE then use only
that channel's readings. These candidates cannot replace the response predictor.
Channel revision forecasts name frozen executable paths, so a readout can return
forecast.model directly for parent bindings. proposal.original_models preserves
the originally offered paths.
"""


_READOUT_RUNNER = """import contextlib,json,sys
payload=json.load(sys.stdin)
with contextlib.redirect_stdout(sys.stderr):
    namespace={'__name__':'revision_readout','__file__':payload['path']}
    exec(compile(payload['source'],payload['path'],'exec'),namespace)
    answer=namespace['readout'](payload['result'],payload['spec'],payload['memory'])
print(json.dumps(answer,allow_nan=False))
"""


def _sources(agent, model):
    """Capture executable source, including ordinary local Python dependencies."""
    directory, inner = python_model_context(agent.workspace, model)
    sources = {}
    # Retained contexts were already excluded from this source record. Prune
    # them before walking: repeated comparisons can otherwise traverse every
    # preceding candidate's complete archive just to discard it afterwards.
    for root, directories, filenames in os.walk(directory):
        directories[:] = [name for name in directories if name != '_evaluated_model']
        for name in filenames:
            if name.endswith('.py'):
                path = Path(root) / name
                sources[str(path.relative_to(directory))] = path.read_text(encoding='utf-8')
    return {'model': inner, 'files': sources}


def _forecast_models(agent, models, rows, forecasts, *, model_contexts=None):
    """Execute authored predictors on public inputs, without acquiring labels."""
    if len(rows) * len(models) > agent.config.compute_budget - agent.costs['compute_cost']:
        raise ValueError('prospective candidate-row scoring budget exhausted')
    for model in models:
        execution_model = (model_contexts or {}).get(model, model)
        forecast = {'model': model, 'prediction_inputs': [
            {'features': deepcopy(row.features), 'history': deepcopy(row.history)} for row in rows]}
        if execution_model != model:
            forecast['execution_model'] = execution_model
        forecasts.append(forecast)
        try:
            forecast['source'] = _sources(agent, execution_model)
            remaining = agent.config.execution_seconds - agent.costs['python_execution_seconds']
            if remaining <= 0:
                raise ValueError('Python execution time budget exhausted')
            agent.costs['compute_cost'] += len(rows)
            started = monotonic()
            try:
                result = evaluate_python_model(agent.workspace, execution_model, rows, timeout_seconds=remaining)
            finally:
                agent.costs['python_execution_seconds'] += monotonic() - started
            # Placeholder target errors are not scientific results.
            forecast.update({key: value for key, value in result.items() if key not in {'mse', 'loss'}})
        except (ValueError, TypeError, KeyError, OSError) as error:
            forecast.update(returncode=1, error=f'{type(error).__name__}: {error}')


def _compare_paid_models(agent, models, prior_fit, *, model_contexts=None,
                         target='response', laboratory=None):
    """Reuse paid scoring and retention after saving every offered forecast."""
    rows = agent.rows if target == 'response' else laboratory.target_rows(agent.observations, target)
    prior_models = sorted(models, key=lambda model: '_evaluated_model' not in model.split('/'))
    for model in prior_models:
        if not rows:
            break
        execution_model = (model_contexts or {}).get(model, model)
        try:
            if target == 'response':
                score = agent._score(execution_model)
                agent._retain_evaluation(execution_model, score)
            else:
                from .observable_research import evaluate_observable
                score = evaluate_observable(agent, {'target': target, 'model': execution_model}, laboratory)
            prior_fit.append({'model': model, **score})
        except (ValueError, TypeError, KeyError, OSError) as error:
            prior_fit.append({'model': model, 'error': f'{type(error).__name__}: {error}'})


def _comparison_contexts(agent, models, record, *, target='response', laboratory=None,
                         context_kind='comparison'):
    """Keep offered predictors callable after incumbent retention replaces its view."""
    contexts = {}
    for index, model in enumerate(models):
        candidate = {'source_model': model}
        record['candidate_contexts'].append(candidate)
        try:
            directory, inner = python_model_context(agent.workspace, model)
            if not (directory / inner).is_file():
                raise FileNotFoundError(model)
            destination = (f"_experiment_models/{context_kind}_{record['index']:04d}/"
                           f"candidate_{index:04d}/_evaluated_model")
            training = (laboratory.target_rows(agent.observations, target)
                        if target != 'response' and '_evaluated_model' not in model.split('/') else None)
            retained = retain_evaluated_python_model(
                agent.workspace, model, [], None, destination=destination, training_observations=training)
            if training is not None:
                from .observable_research import _indices
                (agent.workspace / destination / 'observable_support.json').write_text(json.dumps({
                    'target': target, 'training_observation_indices': _indices(agent.observations, target)}))
            frozen_model = f"{destination}/{retained['model']}"
            contexts[model] = candidate['model'] = frozen_model
        except (ValueError, TypeError, KeyError, OSError) as error:
            candidate['error'] = f'{type(error).__name__}: {error}'
    return contexts


def prepare_revision(agent, proposal, actions, laboratory, *, experiment_path=None):
    """Freeze predictions and compare candidates before the first new observation."""
    record = {'index': len(agent.revisions), 'proposal': deepcopy(proposal),
              'status': 'preparing', 'evidence_cutoff': {
                  'records': len(agent.observations), 'rows': len(agent.rows)},
              'actions': deepcopy(actions), 'forecasts': [], 'prior_fit': [],
              'experiment_path': experiment_path}
    agent.revisions.append(record)
    try:
        if (not isinstance(proposal, dict) or not proposal.get('id')
                or not proposal.get('affected_object')):
            raise ValueError('revision needs an id and affected_object')
        models = proposal.get('models')
        if not isinstance(models, list) or not models or any(not isinstance(m, str) or not m for m in models):
            raise ValueError('revision.models must be a nonempty list of Python predictor paths')
        if not actions:
            raise ValueError('a prospective revision requires actual query actions')
        target = record['target'] = proposal.get('target', 'response')
        if target != 'response':
            laboratory.target_rows(agent.observations, target)
            if any(target not in action.get('channels', []) for action in actions):
                raise ValueError('every action must include the prediction target in channels')
        rows_by_action = [laboratory.prediction_rows(action) for action in actions]
        rows = [row for group in rows_by_action for row in group]
        record['action_row_counts'] = [len(group) for group in rows_by_action]
        if len(rows) * len(models) > agent.config.compute_budget - agent.costs['compute_cost']:
            raise ValueError('prospective candidate-row scoring budget exhausted')
        if experiment_path:
            record['experiment_source'] = (agent.workspace / experiment_path).read_text(encoding='utf-8')
        readout = proposal.get('readout')
        if readout:
            record['readout_source'] = (agent.workspace / readout).read_text(encoding='utf-8')
        if target != 'response':
            record['candidate_contexts'] = []
            contexts = _comparison_contexts(agent, models, record, target=target,
                                           laboratory=laboratory, context_kind='revision')
            originals = proposal.get('original_models', {})
            record['proposal']['original_models'] = {
                contexts.get(model, model): originals.get(model, model) for model in models}
            models = record['proposal']['models'] = [contexts.get(model, model) for model in models]
        # Forecast all candidates before retention can replace an incumbent path.
        _forecast_models(agent, models, rows, record['forecasts'])
        # Score retained paths before a better new model can replace their
        # fitting directory; forecasts already captured the original context.
        _compare_paid_models(agent, models, record['prior_fit'], target=target, laboratory=laboratory)
        record['evaluated_model'] = agent._evaluated_summary()
        record['status'] = 'predictions_frozen'
    except (ValueError, TypeError, KeyError, OSError, AttributeError) as error:
        record.update(status='preparation_failed', error=f'{type(error).__name__}: {error}')
    return record


def compare_experiments(agent, models, experiments, laboratory, *, mse_threshold, target='response',
                        reference_actions=None):
    """Compare only authored Python hypotheses and experiments before purchase.

    The forecast-column adapter follows world_experiment_choice: predictions
    become feature projections for the existing numerical trajectory chooser.
    It neither constructs scientific alternatives nor executes a selected probe.
    """
    record = {'index': len(agent.experiment_comparisons), 'models': deepcopy(models), 'target': target,
              'experiments': deepcopy(experiments),
              'evidence_cutoff': {'records': len(agent.observations), 'rows': len(agent.rows)},
              'mse_threshold': mse_threshold, 'prior_fit': [], 'forecasts': [],
              'candidate_contexts': [],
              'comparisons': [], 'eligible_models': [], 'selected_experiment': None,
              'query_arguments': None, 'status': 'preparing',
              'diagnosis': 'preparing', 'comparison_scope': 'authored_experiments_only'}
    if reference_actions is not None:
        record['reference_actions'] = deepcopy(reference_actions)
        record['selection_policy'] = 'expected_reference_risk_reduction'
    agent.experiment_comparisons.append(record)
    before = agent.costs['compute_cost']
    try:
        if (not isinstance(models, list) or not models
                or any(not isinstance(model, str) or not model for model in models)
                or len(set(models)) != len(models)):
            raise ValueError('models must list distinct existing Python predictor paths')
        if not isinstance(experiments, list) or not experiments:
            raise ValueError('experiments must contain authored experiment objects')
        paid_rows = (agent.rows if target == 'response'
                     else laboratory.target_rows(agent.observations, target))
        rows, seen = [], set()
        for experiment in experiments:
            identifier, actions = experiment.get('id'), experiment.get('actions')
            if not isinstance(identifier, str) or not identifier or identifier in seen:
                raise ValueError('each experiment needs a unique nonempty id')
            if not isinstance(actions, list) or not actions:
                raise ValueError('each experiment needs authored public actions')
            if target != 'response' and any(target not in action.get('channels', []) for action in actions):
                raise ValueError('every action must include the prediction target in channels')
            seen.add(identifier)
            groups = [laboratory.prediction_rows(action) for action in actions]
            inputs = [row for group in groups for row in group]
            record['comparisons'].append({'id': identifier, 'actions': deepcopy(actions),
                'row_start': len(rows), 'row_count': len(inputs),
                'action_row_counts': [len(group) for group in groups],
                'observation_cost': sum(laboratory.action_cost(action) for action in actions),
                'forecasts': [], 'predicted_separation': None})
            rows.extend(inputs)
        reference_start = len(rows)
        if reference_actions is not None:
            if not isinstance(reference_actions, list) or not reference_actions:
                raise ValueError('reference_actions must contain public prediction actions')
            if target != 'response' and any(target not in action.get('channels', [])
                                           for action in reference_actions):
                raise ValueError('reference actions must include the prediction target in channels')
            rows.extend(row for action in reference_actions for row in laboratory.prediction_rows(action))
        # Both prospective and paid execution use the same saved fitting view.
        # Paid retention may replace _evaluated_model before the actor purchases
        # a selected experiment, so that incumbent alias cannot be its identity.
        contexts = _comparison_contexts(agent, models, record, target=target, laboratory=laboratory)
        _forecast_models(agent, models, rows, record['forecasts'], model_contexts=contexts)
        _compare_paid_models(agent, models, record['prior_fit'], model_contexts=contexts,
                             target=target, laboratory=laboratory)
        record['evaluated_model'] = agent._evaluated_summary()
        eligible = record['eligible_models'] = [fit['model'] for fit in record['prior_fit']
            if fit.get('returncode') == 0 and isinstance(fit.get('mse'), (int, float))
            and fit['mse'] <= mse_threshold]
        for experiment in record['comparisons']:
            start, count = experiment['row_start'], experiment['row_count']
            for forecast in record['forecasts']:
                detail = {key: deepcopy(value) for key, value in forecast.items()
                          if key not in {'source', 'predictions', 'prediction_inputs'}}
                if 'predictions' in forecast:
                    detail['predictions'] = forecast['predictions'][start:start + count]
                experiment['forecasts'].append(detail)
        if not paid_rows:
            record['diagnosis'] = 'no_paid_support'
        elif not eligible:
            record['diagnosis'] = 'needs_model_revision'
        elif len(eligible) == 1:
            record['diagnosis'] = 'single_supported_candidate'
        elif any(forecast.get('returncode') != 0 for forecast in record['forecasts']
                 if forecast['model'] in eligible):
            record['diagnosis'] = 'forecast_execution_failed'
        else:
            columns = [f'forecast_{index}' for index in range(len(eligible))]
            candidates = [Candidate(model, Program('feature', name=column), 'authored_forecast')
                          for model, column in zip(eligible, columns)]
            budget = SearchBudget(agent.config.compute_budget - agent.costs['compute_cost'])
            best, best_value = None, 0.
            complete = True
            for experiment in record['comparisons']:
                forecasts = {forecast['model']: forecast['predictions']
                             for forecast in experiment['forecasts'] if forecast['model'] in eligible}
                probe = Probe(experiment['id'], tuple(
                    {column: forecasts[model][index] for model, column in zip(eligible, columns)}
                    for index in range(experiment['row_count'])))
                selected, separation = choose_probe([probe], candidates, 0., budget)
                if selected is None:
                    complete = False
                    break
                experiment['predicted_separation'] = separation
                if separation > best_value:
                    best, best_value = experiment, separation
            agent.costs['compute_cost'] += budget.spent
            record['diagnosis'] = ('comparison_budget_exhausted' if not complete else
                'discriminating_experiment_available' if best is not None else 'observational_ambiguity')
            if complete and reference_actions is not None:
                reference_predictions = {
                    forecast['model']: forecast['predictions'][reference_start:]
                    for forecast in record['forecasts'] if forecast['model'] in eligible}
                panels = [{
                    'id': experiment['id'], 'observation_cost': experiment['observation_cost'],
                    'predictions': [next(forecast['predictions'] for forecast in experiment['forecasts']
                                         if forecast['model'] == model) for model in eligible],
                } for experiment in record['comparisons']]
                # All retained instances meet the same paid-fit criterion. The
                # finite empirical prior has equal mass, including alternative
                # parameter fits when the author supplies them. It is not a
                # posterior over unoffered families or their parameter spaces.
                noise_scale = max(min(fit['mse'] for fit in record['prior_fit']
                                      if fit['model'] in eligible),
                                  pvariance(row.target for row in paid_rows) * 1e-6, 1e-8)
                value_budget = SearchBudget(agent.config.compute_budget - agent.costs['compute_cost'])
                value = value_forecast_experiments(
                    [reference_predictions[model] for model in eligible], panels,
                    weights=[1. / len(eligible)] * len(eligible),
                    noise_scale=noise_scale, budget=value_budget)
                agent.costs['compute_cost'] += value_budget.spent
                record['task_value'] = {**value, 'noise_scale': noise_scale,
                    'reference_rows': len(rows) - reference_start,
                    'candidate_weights': [{'model': model, 'weight': 1. / len(eligible)} for model in eligible],
                    'scope': 'Expected MSE reduction on authored reference inputs, conditional on the '
                        'supplied supported instances and finite observation channel. No reference '
                        'labels, unoffered models or future refitting outcomes were consulted.'}
                best = next((experiment for experiment in record['comparisons']
                             if experiment['id'] == value['selected_experiment_id']), None)
                record['diagnosis'] = ('comparison_budget_exhausted' if value['status'] == 'budget_exhausted'
                    else 'task_relevant_experiment_available' if best is not None
                    else 'no_expected_task_risk_reduction')
            if complete and best is not None:
                record['selected_experiment'] = {key: deepcopy(best[key]) for key in
                    ('id', 'actions', 'observation_cost', 'predicted_separation')}
                record['selected_experiment']['revision'] = {
                    'id': f"comparison_{record['index']}_{best['id']}",
                    'affected_object': 'prediction', 'target': target,
                    'models': [contexts[model] for model in eligible],
                    'original_models': {contexts[model]: model for model in eligible},
                    'comparison_index': record['index']}
                record['query_arguments'] = {key: deepcopy(record['selected_experiment'][key])
                                             for key in ('actions', 'revision')}
        record['status'] = 'compared'
    except (ValueError, TypeError, KeyError, OSError, AttributeError) as error:
        record.update(status='comparison_failed', diagnosis='comparison_failed',
                      error=f'{type(error).__name__}: {error}')
    record['compute_cost'] = agent.costs['compute_cost'] - before
    agent._prepare_program_state()
    return deepcopy(record)


def complete_revision(agent, record, observations, record_indices, laboratory, question=None, *, error=None):
    """Score frozen predictions against purchased rows and return authored analysis."""
    record.update(observations=deepcopy(observations), record_indices=list(record_indices),
                  status='acquisition_failed' if error else 'observed')
    if error:
        record['acquisition_error'] = error
    target = record.get('target', 'response')
    rows = (laboratory.target_rows(observations, target) if target != 'response' else
            [row for batch in observations for row in laboratory.observation_rows(batch)])
    targets = [row.target for row in rows]
    for forecast in record['forecasts']:
        predicted = forecast.get('predictions')
        if forecast.get('returncode') == 0 and predicted is not None:
            forecast['scored_rows'] = len(targets)
            forecast['complete_experiment'] = len(targets) == len(predicted)
            forecast['prospective_mse'] = (sum((a-b)**2 for a, b in zip(predicted, targets)) / len(targets)
                if targets and forecast['complete_experiment'] else None)
    if record['proposal'].get('readout') and observations:
        analyze_revision(agent, record, laboratory, question)
    agent._prepare_program_state()
    return deepcopy(record)


def analyze_revision(agent, record, laboratory, question=None, *, path=None):
    """Run an authored readout on this experiment's saved predictions and evidence.

    Retrying analysis does not refit the pre-experiment models or repurchase rows.
    The original readout and every later executed revision remain inspectable.
    """
    path = path or record['proposal'].get('readout')
    if not path or not record.get('observations'):
        raise ValueError('analysis requires a readout path and an acquired revision')
    source = ((agent.workspace / path).read_text(encoding='utf-8')
              if path != record['proposal'].get('readout') or record.get('analysis_attempts')
              else record.get('readout_source') or (agent.workspace / path).read_text(encoding='utf-8'))
    attempt = {'path': path, 'source': source, 'paid_rows': len(agent.rows)}
    record.setdefault('analysis_attempts', []).append(attempt)
    record.pop('readout_error', None)
    try:
        agent._prepare_program_state()
        code, stdout, stderr, timed_out = agent._python(
            [sys.executable, '-c', _READOUT_RUNNER], json.dumps({
                'source': source, 'path': path,
                'result': record, 'spec': laboratory.public_spec(), 'memory': agent.knowledge}))
        execution = {'returncode': code, 'stderr': stderr, 'timeout': timed_out}
        record['readout_execution'] = execution
        attempt['execution'] = deepcopy(execution)
        if code:
            raise ValueError(stderr or 'authored readout failed')
        answer = json.loads(stdout)
        attempt['authored_answer'] = deepcopy(answer)
        if (not isinstance(answer, dict) or answer.get('status') not in
                {'supported', 'refuted', 'inconclusive'} or 'answer' not in answer):
            raise ValueError('readout must return status and answer')
        evidence = answer.get('evidence')
        if (not isinstance(evidence, list) or not evidence
                or any(type(i) is not int or not 0 <= i < len(agent.rows) for i in evidence)):
            raise ValueError('readout evidence must reference actual paid response row indices')
        record['authored_answer'] = deepcopy(answer)
        record['analysis_is_agent_authored'] = True
        if question is not None:
            question.update({key: deepcopy(answer[key]) for key in ('status', 'answer', 'evidence')})
            question['reviewed_through_observation'] = len(agent.rows) - 1
            question['revision_index'] = record['index']
            returned = getattr(agent, '_return_revision_answer', None)
            if callable(returned):
                returned(question, record, answer)
            else:
                parent = next((q for q in reversed(agent.questions)
                               if q['id'] == question.get('parent_id')), None)
                if parent is not None:
                    parent.setdefault('child_answers', []).append({
                        'question_id': question['id'], 'revision_index': record['index'], **deepcopy(answer)})
        record['status'] = 'answered'
    except (ValueError, TypeError, KeyError, OSError) as failure:
        attempt['error'] = record['readout_error'] = f'{type(failure).__name__}: {failure}'
    agent._prepare_program_state()
    return deepcopy(record)
