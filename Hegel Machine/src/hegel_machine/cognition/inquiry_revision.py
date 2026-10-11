"""Execute authored representation/experiment revisions in the existing inquiry loop."""

from copy import deepcopy
import json
import sys
from time import monotonic

from .code_agent import evaluate_python_model, python_model_context
from .language import Program
from .models import Candidate, Probe, SearchBudget
from .probing import choose_probe


REVISION_INTERFACE = """
A query or select_actions result may include revision:{id:'r1',
affected_object:'representation', models:['old.py','new.py'], readout:'readout.py',
parent_and_scope:{...}, ...}. Supply the actual experiment through the usual
actions. Additional proposal fields are retained as authored. Each named model
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
research_state.json exposes revisions and questions, including returned answers,
to the next program invocation. All arms may use this same interface. Failing
forecasts and analyses remain recorded. No physical mechanism is supplied.
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
    for path in directory.rglob('*.py'):
        relative = path.relative_to(directory)
        if '_evaluated_model' not in relative.parts:
            sources[str(relative)] = path.read_text(encoding='utf-8')
    return {'model': inner, 'files': sources}


def _forecast_models(agent, models, rows, forecasts):
    """Execute authored predictors on public inputs, without acquiring labels."""
    if len(rows) * len(models) > agent.config.compute_budget - agent.costs['compute_cost']:
        raise ValueError('prospective candidate-row scoring budget exhausted')
    for model in models:
        forecast = {'model': model, 'prediction_inputs': [
            {'features': deepcopy(row.features), 'history': deepcopy(row.history)} for row in rows]}
        forecasts.append(forecast)
        try:
            forecast['source'] = _sources(agent, model)
            remaining = agent.config.execution_seconds - agent.costs['python_execution_seconds']
            if remaining <= 0:
                raise ValueError('Python execution time budget exhausted')
            agent.costs['compute_cost'] += len(rows)
            started = monotonic()
            try:
                result = evaluate_python_model(agent.workspace, model, rows, timeout_seconds=remaining)
            finally:
                agent.costs['python_execution_seconds'] += monotonic() - started
            # Placeholder target errors are not scientific results.
            forecast.update({key: value for key, value in result.items() if key not in {'mse', 'loss'}})
        except (ValueError, TypeError, KeyError, OSError) as error:
            forecast.update(returncode=1, error=f'{type(error).__name__}: {error}')


def _compare_paid_models(agent, models, prior_fit):
    """Reuse paid scoring and retention after saving every offered forecast."""
    prior_models = sorted(models, key=lambda model: '_evaluated_model' not in model.split('/'))
    for model in prior_models:
        if not agent.rows:
            break
        try:
            score = agent._score(model)
            agent._retain_evaluation(model, score)
            prior_fit.append({'model': model, **score})
        except (ValueError, TypeError, KeyError, OSError) as error:
            prior_fit.append({'model': model, 'error': f'{type(error).__name__}: {error}'})


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
        # Forecast all candidates before retention can replace an incumbent path.
        _forecast_models(agent, models, rows, record['forecasts'])
        # Score retained paths before a better new model can replace their
        # fitting directory; forecasts already captured the original context.
        _compare_paid_models(agent, models, record['prior_fit'])
        record['evaluated_model'] = agent._evaluated_summary()
        record['status'] = 'predictions_frozen'
    except (ValueError, TypeError, KeyError, OSError, AttributeError) as error:
        record.update(status='preparation_failed', error=f'{type(error).__name__}: {error}')
    return record


def compare_experiments(agent, models, experiments, laboratory, *, mse_threshold):
    """Compare only authored Python hypotheses and experiments before purchase.

    The forecast-column adapter follows world_experiment_choice: predictions
    become feature projections for the existing numerical trajectory chooser.
    It neither constructs scientific alternatives nor executes a selected probe.
    """
    record = {'index': len(agent.experiment_comparisons), 'models': deepcopy(models),
              'experiments': deepcopy(experiments),
              'evidence_cutoff': {'records': len(agent.observations), 'rows': len(agent.rows)},
              'mse_threshold': mse_threshold, 'prior_fit': [], 'forecasts': [],
              'comparisons': [], 'eligible_models': [], 'selected_experiment': None,
              'query_arguments': None, 'status': 'preparing',
              'diagnosis': 'preparing', 'comparison_scope': 'authored_experiments_only'}
    agent.experiment_comparisons.append(record)
    before = agent.costs['compute_cost']
    try:
        if (not isinstance(models, list) or not models
                or any(not isinstance(model, str) or not model for model in models)
                or len(set(models)) != len(models)):
            raise ValueError('models must list distinct existing Python predictor paths')
        if not isinstance(experiments, list) or not experiments:
            raise ValueError('experiments must contain authored experiment objects')
        rows, seen = [], set()
        for experiment in experiments:
            identifier, actions = experiment.get('id'), experiment.get('actions')
            if not isinstance(identifier, str) or not identifier or identifier in seen:
                raise ValueError('each experiment needs a unique nonempty id')
            if not isinstance(actions, list) or not actions:
                raise ValueError('each experiment needs authored public actions')
            seen.add(identifier)
            groups = [laboratory.prediction_rows(action) for action in actions]
            inputs = [row for group in groups for row in group]
            record['comparisons'].append({'id': identifier, 'actions': deepcopy(actions),
                'row_start': len(rows), 'row_count': len(inputs),
                'action_row_counts': [len(group) for group in groups],
                'observation_cost': sum(laboratory.action_cost(action) for action in actions),
                'forecasts': [], 'predicted_separation': None})
            rows.extend(inputs)
        _forecast_models(agent, models, rows, record['forecasts'])
        _compare_paid_models(agent, models, record['prior_fit'])
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
        if not agent.rows:
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
            if complete and best is not None:
                record['selected_experiment'] = {key: deepcopy(best[key]) for key in
                    ('id', 'actions', 'observation_cost', 'predicted_separation')}
                record['selected_experiment']['revision'] = {
                    'id': f"comparison_{record['index']}_{best['id']}",
                    'affected_object': 'prediction', 'models': list(eligible),
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
    targets = [row.target for batch in observations for row in laboratory.observation_rows(batch)]
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
