"""Return repeated failed predictions to the existing executable research loop."""

from copy import deepcopy
from math import isfinite


def generalization_failures(confirmations, threshold):
    """Keep actual well-fitted prediction failures since the last supported test.

    A prediction execution error is not a measured failure. The retained attempts
    concern submitted predictors, without identifying or excluding model families.
    """
    failed = []
    next_cutoff = None
    for index in range(len(confirmations) - 1, -1, -1):
        confirmation = confirmations[index]
        if confirmation.get('passed') is True:
            break
        mse = confirmation.get('mse')
        if (confirmation.get('passed') is not False or confirmation.get('returncode') != 0
                or confirmation.get('timeout') or not isinstance(mse, (int, float))
                or not isfinite(mse)):
            continue
        if mse <= confirmation.get('mse_threshold', threshold):
            break
        submitted = confirmation.get('submitted_candidate', {})
        paid_mse = submitted.get('mse')
        if (not isinstance(paid_mse, (int, float)) or not isfinite(paid_mse)
                or paid_mse > confirmation.get('mse_threshold', threshold)):
            break
        cutoff = submitted.get('paid_rows')
        indices = confirmation.get('observation_indices', [])
        if (next_cutoff is not None and isinstance(cutoff, int)
                and (cutoff >= next_cutoff or indices and max(indices) >= next_cutoff)):
            break
        if isinstance(cutoff, int):
            next_cutoff = cutoff
        failed.append({'confirmation_index': index,
            **{key: deepcopy(confirmation[key]) for key in
               ('mse', 'mse_threshold', 'rows', 'observation_indices', 'submitted_candidate')
               if key in confirmation}})
    return list(reversed(failed))


def _execute_review_refit(agent, arguments, laboratory, origin):
    before = len(agent.operator_results)
    try:
        result = agent._execute('run', arguments, laboratory)
    except (ValueError, TypeError, KeyError, OSError) as error:
        result = {'operator': 'refit', 'status': 'refit_execution_failed',
                  'error': str(error)}
        if len(agent.operator_results) > before:
            result.update({key: deepcopy(value) for key, value in
                           agent.operator_results[-1].items()
                           if key not in {'index', 'arguments'}})
    if len(agent.operator_results) == before:
        agent.operator_results.append({'index': before, 'arguments': deepcopy(arguments),
                                       **deepcopy(result)})
    agent.operator_results[-1]['trigger_origin'] = origin
    return result


def resume_generalization_review(agent, question, laboratory):
    """Continue only an unexecuted review's original refit on explicit resumption."""
    previous = question.get('whole_trajectory_refit', {})
    review = question.get('generalization_review', {})
    original = review.get('refit_arguments')
    if previous.get('status') != 'refit_execution_failed' or not original:
        return None
    retries = review.setdefault('refit_retries', [])
    prior_arguments = retries[-1]['arguments'] if retries else original
    # Common consumes the question association before recording the operator.
    prior_arguments = {key: value for key, value in prior_arguments.items()
                       if key != 'research_question'}
    prior_index = next((entry['index'] for entry in reversed(agent.operator_results)
                        if entry['arguments'] == prior_arguments), None)
    arguments = deepcopy(original)
    number = len(retries) + 1
    while (agent.workspace / (original['path'] + f'_resume_{number:04d}')).exists():
        number += 1
    arguments['path'] = original['path'] + f'_resume_{number:04d}'
    before_costs = deepcopy(agent.costs)
    result = _execute_review_refit(agent, arguments, laboratory, 'explicit_question_resume')
    split, validation = result.get('split', {}), result.get('validation', {})
    retry = {'origin': 'explicit_question_resume', 'previous_status': previous['status'],
        'previous_operator_result_index': prior_index,
        'operator_result_index': agent.operator_results[-1]['index'],
        'arguments': deepcopy(arguments), 'status': result.get('status'),
        'validation_mse': validation.get('mse'), 'validation_returncode': validation.get('returncode'),
        'train_rows': split.get('train_rows'), 'holdout_rows': split.get('holdout_rows'),
        'costs': {key: value - before_costs[key] for key, value in agent.costs.items()
                  if isinstance(value, (int, float)) and isinstance(before_costs[key], (int, float))}}
    retries.append(retry)
    question['whole_trajectory_refit'] = deepcopy(result)
    if split:
        review.update(training_records=deepcopy(split['training_records']),
                      validation_records=deepcopy(split['validation_records']))
    if 'parent_research' in question:
        question['parent_research']['method_review_result'] = {
            **{key: deepcopy(result[key]) for key in
               ('status', 'source_model', 'tested_model', 'error', 'paid_rows_at_refit') if key in result},
            'validation_mse': validation.get('mse'),
            'validation_returncode': validation.get('returncode'),
            **{key: deepcopy(split[key]) for key in
               ('train_rows', 'holdout_rows', 'validation_records') if key in split}}
    return deepcopy(retry)


def prepare_generalization_review(agent, question, confirmation, laboratory):
    """Test the latest fitting procedure with two failed trajectory blocks held out."""
    if 'whole_trajectory_refit' in question:
        return
    failures = generalization_failures(agent.confirmations, agent.config.confirmation_mse_threshold)
    if len(failures) < 2:
        return
    question.update(review_scope='prediction_method_revision', review_model_calls=12)
    question.setdefault('generalization_review', {'failed_attempts': failures,
        'evidence_scope': 'Frozen submitted predictors and their actual independent '
            'prediction failures on this paid task; no whole-family refutation.'})
    submitted = confirmation.get('submitted_candidate', {})
    cutoff = submitted.get('paid_rows')
    latest_rows = set(confirmation.get('observation_indices', []))
    expected = latest_rows | set(failures[-2].get('observation_indices', []))
    validation, covered = [], set()
    for index, record in enumerate(agent.observations):
        rows = set(record.get('row_indices', []))
        if rows.intersection(expected):
            if not rows.issubset(expected):
                break
            validation.append(index)
            covered.update(rows)
    else:
        if (latest_rows and covered == expected and type(cutoff) is int
                and min(latest_rows) == cutoff):
            training = [index for index, record in enumerate(agent.observations)
                if index not in validation and record.get('row_indices')
                and max(record['row_indices']) < cutoff]
            arguments = {'operator': 'refit', 'model': submitted.get('model'),
                'path': 'shared/generalization/' + question['id'],
                'records': training + validation, 'validation_records': validation,
                'research_question': {'id': question['id']}}
            review = question['generalization_review']
            review.update(training_cutoff=cutoff, training_records=training,
                          validation_confirmation_indices=[f['confirmation_index'] for f in failures[-2:]],
                          validation_records=validation, refit_arguments=deepcopy(arguments))
            result = _execute_review_refit(agent, arguments, laboratory, 'prediction_method_revision')
            split = result.get('split', {})
            if split:
                review.update(training_records=deepcopy(split['training_records']),
                              validation_records=deepcopy(split['validation_records']))
            question['whole_trajectory_refit'] = deepcopy(result)
            return
    question['whole_trajectory_refit'] = {
        'operator': 'refit', 'status': 'confirmation_record_mapping_unavailable',
        'error': 'The failed confirmations are not fully mapped to complete paid experiment records '
                 'at its submitted fitting cutoff; no refit or acquisition was performed.'}


PREDICTION_METHOD_QUESTION = (
    'The retained well-fitted submitted predictors repeatedly failed on independent trajectories. '
    'Investigate how the current explanation and fitting method generalize, using the frozen '
    'failed submissions, their actual counterexamples, and whole_trajectory_refit. That result '
    'tests the latest fitting procedure with the two recent failed trajectory blocks jointly '
    'held out, including counterexamples a later submission had incorporated into fitting; '
    'it neither rejects its whole family nor independently confirms a replacement. If the '
    'calculation is unavailable or failed to execute, interpret that execution limitation '
    'separately from the measured prediction failures. Revisit your own earlier executable '
    'analyses and local findings; choose which remain applicable and what they leave unexplained. '
    'Propose competing explanations or applicability conditions, derive testable differences, '
    'and design an executable distinguishing experiment when one is available. Choose all '
    'scientific content yourself. Assess a revised fitting procedure on separate complete '
    'paid trajectories using the existing refit tool when useful. Return an executable result '
    'and its measured support, or an unresolved finding. Local finish returns to the parent; '
    'only a later parent submission obtains independent confirmation.'
)
