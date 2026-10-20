"""Retain authored auxiliary predictors and return their evidence to parent research.

This is the ordinary-Python connection to the older instrument/assess/bind flow.
The researcher chooses the target, source and parent composition. No scientific
family, fitting routine or response equation is selected here.
"""

from copy import deepcopy
import json
from pathlib import Path
from time import monotonic

from .code_agent import evaluate_python_model, python_model_context, retain_evaluated_python_model
from .online_inquiry_llm import CONFIRMATION_MSE_THRESHOLD


OBSERVABLE_INTERFACE = """
Auxiliary model interface (identical tools in both conditions):
evaluate {target:'meter',model:'shared/submodel.py',code:'optional Python source'}
scores that public channel using ONLY its purchased readings. fit(rows) receives
that target's labels, with the same public control/history inputs as predict(row).
The task's root observations.json still contains response rows. Use fit(rows) for
the target-specific training input. Omit target or use 'response' for normal scoring.
observable_models holds independently retained predictors and their measured support.
Retention freezes the files and fitting rows. It does not confirm the parent model.
New readings of that target cause charged evaluation of the frozen old predictor;
ordinary response-only purchases do not invalidate its measurement support.
later_readings reports error on readings outside that predictor's saved fitting
evidence, separately from the all-row mse. Both errors must meet the task tolerance
for support, so a large old fit cannot hide a failed later prediction.

To use one retained predictor in another, supply evaluate/finish arguments
{model:'parent.py',bindings:{chosen_name:'exact retained model path'}}. The parent
source defines bind_models(predictors), fit(rows) if needed, and predict(row).
bind_models receives chosen_name -> callable(row); store and call it in your own
fit/predict code. Child initialization uses its saved fitting rows and directory.
You author the composition, including any transformed rows or histories it needs.
Bindings persist for that model path; bindings:{} explicitly removes them.
model_dependencies reports actual calls, not just a declared dependency. Both
conditions may use these models or implement their own ordinary Python composition.
Paid fit is not independent validation. Use experiments or final fresh confirmation
to challenge a composition; auxiliary model retention never submits the response.
"""


def _indices(records, target):
    return [item['observation_index'] for record in records
            for item in record.get('measurements', []) if target in item['values']]


def _score(agent, model, rows, *, refit, later_positions=()):
    if not rows:
        raise ValueError('buy readings of the selected target before evaluating it')
    if agent.costs['compute_cost'] + len(rows) > agent.config.compute_budget:
        raise ValueError('candidate-row scoring budget exhausted')
    remaining = agent.config.execution_seconds - agent.costs['python_execution_seconds']
    if remaining <= 0:
        raise ValueError('Python execution time budget exhausted')
    agent.costs['compute_cost'] += len(rows)
    started = monotonic()
    result = evaluate_python_model(agent.workspace, model, rows, timeout_seconds=remaining,
                                  training_observations=rows if refit else None)
    agent.costs['compute_cost'] += result.get('research_prediction_count', 0)
    agent.costs['python_execution_seconds'] += monotonic() - started
    predictions = result.pop('predictions', None)
    if predictions is not None and later_positions:
        errors = [predictions[index] - (rows[index].target if hasattr(rows[index], 'target')
                                        else rows[index]['target']) for index in later_positions]
        result['later_readings'] = {'rows': len(errors),
            'mse': sum(error ** 2 for error in errors) / len(errors),
            'max_abs_error': max(abs(error) for error in errors)}
    return result


def _saved_training_indices(agent, model, filename, field, selected):
    context, _ = python_model_context(agent.workspace, model)
    path = context / filename
    support = json.loads(path.read_text()) if path.exists() else {}
    return support.get('training_observation_indices') if support.get(field) == selected else None


def _later_positions(indices, training_indices):
    if training_indices is None:
        return []
    fitted = set(training_indices)
    return [position for position, index in enumerate(indices) if index not in fitted]


def _update_support(entry, result, indices, *, mse_threshold=CONFIRMATION_MSE_THRESHOLD):
    entry.update(mse=result.get('mse'), evaluated_observation_indices=list(indices),
                 mse_threshold=mse_threshold, loss_kind='mse')
    entry.pop('later_readings', None)
    if 'later_readings' in result:
        entry['later_readings'] = deepcopy(result['later_readings'])
    if result.get('returncode') != 0:
        entry['support'] = 'execution_failed'
    elif (result['mse'] > mse_threshold
          or result.get('later_readings', {}).get('mse', 0.) > mse_threshold):
        entry['support'] = 'refuted_on_paid_rows'
    else:
        entry['support'] = ('paid_evaluation' if entry['training_observation_indices'] is None else
                           'paid_fit' if indices == entry['training_observation_indices']
                            else 'supported_on_later_readings')


def return_model_answer(agent, target, question=None, *, entry=None, kind='executable_observable'):
    """Hegel schedules a parent return; flat Code receives the same model evidence."""
    entry = agent.observable_models[target] if entry is None else entry
    conditional = kind == 'executable_conditional'
    field = 'conditional_component' if conditional else 'observable_target'
    if question is None:
        question = next((q for q in agent.questions if q.get(field) == target), None)
    if question is None or question['id'] == 'task':
        if agent.arm != 'hegel':
            return
        prefix = 'conditional' if conditional else 'observable'
        assignment = (f'Predict the selected {target} component from its authored paid observation view.'
                      if conditional else f'Predict the selected {target} readings from public control history.')
        question = agent._open_question({'id': f'{prefix}:{target}',
            'question': assignment,
            'parent_id': 'task', 'depends_on': []})
        question['origin'] = 'framework_organization_of_author_selected_target'
    question[field] = target
    question['status'] = entry['support']
    question['model_answer'] = deepcopy(entry)
    parent = next((q for q in agent.questions if q['id'] == question.get('parent_id')), None)
    if parent is not None:
        parent.setdefault('child_answers', []).append({
            'question_id': question['id'], 'kind': kind, **deepcopy(entry)})
        parent['status'] = 'reopened'
    if agent.arm == 'hegel':
        supported = entry['support'] in {'paid_fit', 'paid_evaluation', 'supported_on_later_readings'}
        agent._active_question_id = parent['id'] if supported and parent else question['id']


def refresh_observables(agent, laboratory):
    """Reassess only when newly purchased readings actually extend a target."""
    outcomes = []
    for target, entry in agent.observable_models.items():
        indices = _indices(agent.observations, target)
        if indices == entry['evaluated_observation_indices']:
            continue
        entry['support'] = 'needs_reassessment'
        try:
            result = _score(agent, entry['model'],
                            laboratory.target_rows(agent.observations, target), refit=False,
                            later_positions=_later_positions(indices, entry['training_observation_indices']))
        except ValueError as error:
            result = {'error': str(error)}
        else:
            _update_support(entry, result, indices, mse_threshold=agent.config.confirmation_mse_threshold)
        outcomes.append({'target': target, 'evaluation': result, 'retained': deepcopy(entry)})
        return_model_answer(agent, target)
    return outcomes


def evaluate_observable(agent, arguments, laboratory, question=None):
    target, model = arguments['target'], arguments.get('model', 'model.py')
    rows = laboratory.target_rows(agent.observations, target)
    indices = _indices(agent.observations, target)
    refresh = refresh_observables(agent, laboratory)
    retained_source = '_evaluated_model' in Path(model).parts
    training_indices = (_saved_training_indices(agent, model, 'observable_support.json', 'target', target)
                        if retained_source else indices)
    result = _score(agent, model, rows, refit=not retained_source,
                    later_positions=_later_positions(indices, training_indices))
    result.update(target=target, executed_path=model, observation_indices=indices)
    if refresh:
        result['reassessments'] = refresh
    incumbent = agent.observable_models.get(target)
    if incumbent is not None and model == incumbent['model']:
        _update_support(incumbent, result, indices, mse_threshold=agent.config.confirmation_mse_threshold)
    elif result.get('returncode') == 0:
        # Never compare a new fit with an old candidate scored on fewer rows.
        comparable = incumbent is None or incumbent['evaluated_observation_indices'] == indices
        better = incumbent is None or incumbent.get('mse') is None or result['mse'] < incumbent['mse']
        if comparable and better:
            version = 0 if incumbent is None else incumbent['version'] + 1
            directory = Path('_observable_models') / target / str(version) / '_evaluated_model'
            saved = retain_evaluated_python_model(agent.workspace, model, [], result['mse'],
                destination=agent.workspace / directory,
                training_observations=None if retained_source else rows)
            # A previously frozen model retains its original paid fitting view.
            entry = {'target': target, 'version': version,
                'model': str(directory / saved['model']), 'source_model': model,
                'training_observation_indices': deepcopy(training_indices)}
            _update_support(entry, result, indices, mse_threshold=agent.config.confirmation_mse_threshold)
            agent.observable_models[target] = entry
            (agent.workspace / directory / 'observable_support.json').write_text(json.dumps(entry))
        elif not comparable:
            result['retention_deferred'] = 'saved target model still needs assessment on current paid readings'
    if target in agent.observable_models:
        return_model_answer(agent, target, question)
        result['observable_model'] = deepcopy(agent.observable_models[target])
    return result


def dependency_focus(agent):
    """Carry an executable answer to the unresolved parent, without a solution hint."""
    active = next((q for q in agent.questions if q['id'] == agent._active_question_id), None)
    return {'question': deepcopy(active),
            'available_models': deepcopy(agent.observable_models),
            'available_components': deepcopy(agent.conditional_models),
            'parent_predictor': agent._evaluated_summary(),
            'instruction': ('Advance the active question using its actual returned evidence. '
                'If a submodel is supported, decide how to use or challenge it in the unresolved '
                'parent relation; preserve its executable result. If new evidence refutes a '
                'dependency, repair or replace that premise before relying on it again. '
                'You choose the equations, further decomposition, experiments and stopping. '
                'A child paid fit does not establish the parent response.')}
