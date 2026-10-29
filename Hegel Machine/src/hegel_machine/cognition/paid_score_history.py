"""Retain comparison decisions without copying immutable paid score payloads."""

from copy import deepcopy


def retain_paid_assessment_references(agent, references):
    """Make declared paid executions available to the ordinary score matcher.

    Importing a reference does not establish reuse: the live comparator still
    checks its full paid scope and the actual executable model context.
    """
    from .checkpoint_storage import read_json_checkpoint, write_json_checkpoint
    from .method_action_research import read_method_reference

    path = agent.workspace / '_paid_python_scores/_evaluated_model/assessments.json'
    index = read_json_checkpoint(path) if path.is_file() else {
        'scopes': [], 'assessments': [], 'contexts': []}
    imported = existing = scopes = 0
    for reference in references:
        if reference in index['assessments']:
            existing += 1
            continue
        assessment = read_method_reference(reference, agent.workspace)
        scope = assessment['scope_ref']
        if scope not in index['scopes']:
            index['scopes'].append(deepcopy(scope))
            scopes += 1
        index['assessments'].append(deepcopy(reference))
        imported += 1
    if imported:
        write_json_checkpoint(path, index, indent=None)
    return {'status': 'paid_assessment_references_retained',
        'imported_assessments': imported, 'existing_assessments': existing,
        'imported_scopes': scopes, 'assessment_refs': deepcopy(references),
        'compute_cost': 0, 'new_scores': 0, 'new_fits': 0}


def paid_comparison_scope_owner(feedback, *, question, rows, public_spec, workspace=None):
    """Read a missing report owner from its actual paid numerical scopes.

    Early direct comparisons lost Common's question header after scoring. Each
    evaluation still names its original complete scope. Verify those scopes;
    never infer an owner from a model path or the expected question alone.
    """
    from dataclasses import asdict
    import json
    from pathlib import Path
    from .checkpoint_storage import read_json_checkpoint
    from .research_episodes import SEMANTIC_FIELDS
    from .temporal_operator_agent import paid_python_evaluation_contract

    evaluations = feedback.get('evaluations', [])
    if (not evaluations or feedback.get('truncated') or feedback.get('error')
            or feedback.get('models_requested') != feedback.get('models_completed')
            or feedback.get('models_completed') != len(evaluations)):
        return None
    scopes = {}
    for evaluation in evaluations:
        reference = evaluation.get('paid_score_reuse', {}).get('scope_ref')
        if not reference or reference.get('json_path'):
            return None
        path = Path(workspace or Path.cwd()) / reference['file']
        scopes[str(path)] = path
    expected_rows = json.loads(json.dumps([
        asdict(row) if hasattr(row, 'features') else row for row in rows]))
    expected_question = json.loads(json.dumps({key: question[key] for key in
        ('id', 'meaning_version', *SEMANTIC_FIELDS) if key in question}))
    expected_question['meaning_version'] = question.get('meaning_version', 1)
    expected_spec = json.loads(json.dumps(public_spec))
    contract = paid_python_evaluation_contract()
    for path in scopes.values():
        saved = read_json_checkpoint(path, keys={
            'rows', 'question', 'public_spec', 'evaluation_contract'})
        if (saved.get('rows') != expected_rows or saved.get('question') != expected_question
                or saved.get('public_spec') != expected_spec
                or saved.get('evaluation_contract') != contract):
            return None
    return {'research_question_id': question['id'],
            'question_meaning_version': question.get('meaning_version', 1)}


_PAYLOAD_FIELDS = frozenset((
    'model_dependencies', 'research_bindings', 'representation_diagnostic',
    'training_representation_diagnostic', 'paid_relation_diagnostic',
))


def retain_frozen_readout_score(agent, *, model, parent_assessment_ref,
                                panel_ref, evaluation_ref, public_spec, question_id):
    """Join already paid predictions without rerunning their frozen model.

    The new MSE covers the union. Representation diagnostics keep their original
    partial scopes: old evaluator outputs did not retain every represented row,
    so this does not invent a complete union representation or a new MSE floor.
    """
    from copy import copy
    from dataclasses import asdict
    from pathlib import Path
    from .checkpoint_storage import write_json_checkpoint
    from .method_action_research import read_method_reference
    from .code_agent import literal_python_model_context_matches
    from .online_inquiry_llm import _prediction_residuals
    from .temporal_operator_agent import (
        _closed_prediction_prefix_scope, _closed_prediction_values)

    parent = read_method_reference(parent_assessment_ref, agent.workspace)
    panel = read_method_reference(panel_ref, agent.workspace)
    frozen = read_method_reference(evaluation_ref, agent.workspace)
    evaluation = frozen['evaluation']
    context = frozen['frozen_model']
    if ((agent.workspace / model).read_bytes()
            != (Path(context['workspace']) / context['model']).read_bytes()
            or not literal_python_model_context_matches(agent.workspace, model,
                parent['execution_context']['workspace'], parent['execution_context']['model'])
            or evaluation.get('returncode') != 0 or evaluation.get('timeout')):
        raise ValueError('paid readout must belong to the same successfully executed frozen model')
    scope = read_method_reference(parent['scope_ref'], agent.workspace)
    old_rows = scope['rows']
    current_rows = [asdict(row) for row in agent.rows]
    # JSON transport turns history tuples into lists in both the scorer and
    # saved readout. Compare that same public/target execution basis here.
    import json
    if json.loads(json.dumps(current_rows)) != [*old_rows, *panel['rows']]:
        raise ValueError('paid prediction pieces do not cover the current ordered development rows')
    prefix_state = parent.get('closed_prediction_state')
    if prefix_state is not None:
        values = _closed_prediction_values({**deepcopy(parent_assessment_ref),
            'json_path': [*parent_assessment_ref.get('json_path', []), 'closed_prediction_state']},
            agent.workspace)['predictions']
        parts = [{**deepcopy(parent_assessment_ref), 'json_path': [
            *parent_assessment_ref.get('json_path', []), 'closed_prediction_state', 'predictions']}]
        if prefix_state.get('prefix_ref') is not None:
            # Keep the already supported prefix chain, not only its last delta.
            parts = [{'closed_prediction_state_ref': {**deepcopy(parent_assessment_ref),
                'json_path': [*parent_assessment_ref.get('json_path', []), 'closed_prediction_state']}}]
    else:
        parts = deepcopy(parent['result']['paid_prediction_parts'])
        values = []
        for reference in parts:
            values.extend(_closed_prediction_values(reference['closed_prediction_state_ref'],
                agent.workspace)['predictions'] if 'closed_prediction_state_ref' in reference
                else read_method_reference(reference, agent.workspace))
    suffix = evaluation['predictions']
    if len(values) != len(old_rows) or len(suffix) != len(panel['rows']):
        raise ValueError('paid prediction references have incomplete row coverage')
    view = copy(agent)
    view._active_question_id = question_id
    comparison = view._paid_python_score_scope(public_spec,
        assessment_refs=[parent_assessment_ref])
    if not _closed_prediction_prefix_scope(scope, comparison['basis']):
        raise ValueError('paid model task, context or numerical evaluation contract changed')
    predictions = [*values, *suffix]
    mse = sum((prediction - row.target) ** 2
              for prediction, row in zip(predictions, agent.rows)) / len(predictions)
    suffix_ref = {**deepcopy(evaluation_ref), 'json_path': [
        *evaluation_ref.get('json_path', []), 'evaluation', 'predictions']}
    result = {**deepcopy(evaluation), 'mse': mse, 'loss': mse,
        'evaluation_rows': len(predictions),
        'evaluation_scope': 'paid_prediction', 'external_python_wall_seconds': 0.,
        'paid_residuals': _prediction_residuals(agent.rows, predictions, model,
            evaluation_scope='paid_response_evaluation'),
        'paid_prediction_parts': [*parts, suffix_ref],
        'representation_diagnostic_scope_ref': {**deepcopy(panel_ref),
            'json_path': [*panel_ref.get('json_path', []), 'rows']},
        'paid_readout_sources': [deepcopy(parent_assessment_ref), deepcopy(evaluation_ref)]}
    result.pop('predictions', None)
    result.pop('python_execution_seconds', None)
    # This new assessment owns only aggregation metadata and source references.
    # Its input predictions and all original execution fees remain at source.
    index = comparison['index']
    path = comparison['directory'] / f"assessment_{len(index['assessments']):06d}.json"
    assessment = {'model': model, 'source_workspace': str(agent.workspace),
        'scope_ref': comparison['scope_ref'],
        'execution_context': {'workspace': context['workspace'], 'model': context['model']},
        'execution_costs': {'compute_cost': 0, 'python_execution_seconds': 0.}, 'result': result}
    write_json_checkpoint(path, assessment)
    reference = {'file': str(path), 'json_path': []}
    index['assessments'].append(reference)
    write_json_checkpoint(comparison['index_file'], index, indent=None)
    return {'status': 'retained_frozen_prediction_union', 'assessment_ref': reference,
        'rows': len(predictions), 'reused_prefix_rows': len(values), 'reused_readout_rows': len(suffix),
        'compute_cost': 0, 'python_execution_seconds': 0., 'model_executions': 0,
        'representation_scope': 'original_partial_readout_not_full_union'}


def _retain_score_history(evaluation):
    reference = evaluation.get('paid_score_reuse', {}).get('assessment_ref')
    if reference is None:
        return deepcopy(evaluation)
    saved = {key: deepcopy(value) for key, value in evaluation.items()
             if key not in _PAYLOAD_FIELDS and key not in {'paid_residuals', 'retained_comparison'}}
    saved['score_result_ref'] = deepcopy(reference)
    residuals = evaluation.get('paid_residuals')
    if residuals is not None:
        saved['paid_residuals'] = {key: deepcopy(value) for key, value in residuals.items()
                                   if key != 'examples'}
        if 'examples' in residuals:
            saved['paid_residuals']['examples_ref'] = {
                **deepcopy(reference),
                'json_path': [*reference['json_path'], 'paid_residuals', 'examples']}
    if 'retained_comparison' in evaluation:
        saved['retained_comparison'] = _retain_score_history(evaluation['retained_comparison'])
    return saved


def _compact_retained_comparisons(result):
    """Upgrade old owned histories once during recovery, without replaying scores."""
    for evaluation in result.get('evaluations', []):
        incumbent = evaluation.get('retained_comparison')
        if (isinstance(incumbent, dict)
                and incumbent.get('paid_score_reuse', {}).get('assessment_ref') is not None
                and ('score_result_ref' not in incumbent
                     or any(key in incumbent for key in _PAYLOAD_FIELDS)
                     or 'examples' in incumbent.get('paid_residuals', {}))):
            evaluation['retained_comparison'] = _retain_score_history(incumbent)


def retain_comparison_history(result):
    """Own the changing decision; address only an already saved complete score.

    Live evaluation feedback remains complete. Failed, incomplete and unstable
    executions lack an assessment_ref and retain their full original feedback.
    """
    if result.get('operator') != 'research_choice' or 'evaluations' not in result:
        return deepcopy(result)
    retained = {key: deepcopy(value) for key, value in result.items() if key != 'evaluations'}
    retained['evaluations'] = [_retain_score_history(evaluation)
                               for evaluation in result['evaluations']]
    retained['paid_score_history'] = 'assessment_references'
    return share_owned_comparison_metadata(retained)


def share_owned_comparison_metadata(result):
    """Share equal read-only snapshots inside one already owned comparison."""
    parent_states, incumbents, incumbent_scores = [], [], []

    def shared(value, known):
        for item in known:
            if item == value:
                return item
        known.append(value)
        return value

    for evaluation in result.get('evaluations', []):
        if 'retained_comparison' in evaluation:
            evaluation['retained_comparison'] = shared(
                evaluation['retained_comparison'], incumbent_scores)
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
            _compact_retained_comparisons(result)
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
        if 'retained_comparison' in restored:
            restored['retained_comparison'] = restore_score_history_value(
                restored['retained_comparison'], load_reference)
        return restored
    if 'examples_ref' in value and 'examples' not in value:
        return {**{key: item for key, item in value.items() if key != 'examples_ref'},
                'examples': load_reference(value['examples_ref'])}
    return value
