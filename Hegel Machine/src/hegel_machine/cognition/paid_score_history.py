"""Retain comparison decisions without copying immutable paid score payloads."""

from copy import deepcopy


_PAYLOAD_FIELDS = frozenset((
    'model_dependencies', 'research_bindings', 'representation_diagnostic',
    'training_representation_diagnostic', 'paid_relation_diagnostic',
))


def retain_comparison_history(result):
    """Own the changing decision; address only an already saved complete score.

    Live evaluation feedback remains complete. Failed, incomplete and unstable
    executions lack an assessment_ref and retain their full original feedback.
    """
    if result.get('operator') != 'research_choice' or 'evaluations' not in result:
        return deepcopy(result)
    retained = {key: deepcopy(value) for key, value in result.items() if key != 'evaluations'}
    retained['evaluations'] = []
    for evaluation in result['evaluations']:
        reference = evaluation.get('paid_score_reuse', {}).get('assessment_ref')
        if reference is None or 'score_result_ref' in evaluation:
            retained['evaluations'].append(deepcopy(evaluation))
            continue
        saved = {key: deepcopy(value) for key, value in evaluation.items()
                 if key not in _PAYLOAD_FIELDS and key != 'paid_residuals'}
        saved['score_result_ref'] = deepcopy(reference)
        residuals = evaluation.get('paid_residuals')
        if residuals is not None:
            saved['paid_residuals'] = {key: deepcopy(value) for key, value in residuals.items()
                                       if key != 'examples'}
            if 'examples' in residuals:
                saved['paid_residuals']['examples_ref'] = {
                    **deepcopy(reference),
                    'json_path': [*reference['json_path'], 'paid_residuals', 'examples']}
        retained['evaluations'].append(saved)
    retained['paid_score_history'] = 'assessment_references'
    return share_owned_comparison_metadata(retained)


def share_owned_comparison_metadata(result):
    """Share equal read-only snapshots inside one already owned comparison."""
    parent_states, incumbents = [], []

    def shared(value, known):
        for item in known:
            if item == value:
                return item
        known.append(value)
        return value

    for evaluation in result.get('evaluations', []):
        if 'evaluated_model' in evaluation:
            evaluation['evaluated_model'] = shared(evaluation['evaluated_model'], parent_states)
        decision = evaluation.get('retention_decision')
        if decision is not None and 'incumbent' in decision:
            decision['incumbent'] = shared(decision['incumbent'], incumbents)
        reference = evaluation.get('paid_score_reuse', {}).get('assessment_ref')
        if reference is not None and evaluation.get('score_result_ref') == reference:
            evaluation['score_result_ref'] = reference
    return result


def retain_event_history(events):
    """Own restored events without first copying their old repeated score payloads."""
    return [{key: (retain_comparison_history(value)
                   if key == 'feedback' and isinstance(value, dict) else deepcopy(value))
             for key, value in event.items()} for event in events]


def _comparison_histories(agent):
    collections = [(getattr(agent, 'operator_results', []), None),
                   (agent.events, 'feedback'),
                   (getattr(agent, 'method_results', []), 'feedback')]
    collections.extend((question.get('after_observation_results', []), None)
                       for question in agent.questions)
    for collection, field in collections:
        for index, entry in enumerate(collection):
            result = entry if field is None else entry.get(field)
            if isinstance(result, dict) and result.get('operator') == 'research_choice':
                yield collection, index, field, result


def compact_agent_comparisons(agent):
    """Compact appended histories; already retained comparisons need no new scan."""
    for collection, index, field, result in _comparison_histories(agent):
        if result.get('paid_score_history') == 'assessment_references':
            continue
        retained = retain_comparison_history(result)
        if field is None:
            collection[index] = retained
        else:
            collection[index][field] = retained


def restore_comparison_history(agent):
    """Restore metadata sharing once after JSON state has become agent-owned."""
    for collection, index, field, result in _comparison_histories(agent):
        if result.get('paid_score_history') == 'assessment_references':
            share_owned_comparison_metadata(result)
        else:
            retained = retain_comparison_history(result)
            if field is None:
                collection[index] = retained
            else:
                collection[index][field] = retained



def restore_score_history_value(value, load_reference):
    """Expand one addressed score or residual view, preserving occurrence aliases."""
    if not isinstance(value, dict):
        return value
    if 'score_result_ref' in value:
        restored = {**load_reference(value['score_result_ref']),
                    **{key: item for key, item in value.items() if key != 'score_result_ref'}}
        if 'paid_residuals' in restored:
            restored['paid_residuals'] = restore_score_history_value(
                restored['paid_residuals'], load_reference)
        return restored
    if 'examples_ref' in value and 'examples' not in value:
        return {**{key: item for key, item in value.items() if key != 'examples_ref'},
                'examples': load_reference(value['examples_ref'])}
    return value
