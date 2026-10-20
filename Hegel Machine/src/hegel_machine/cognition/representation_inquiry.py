"""Ask the live researcher to expose its own predictor's complete input contract."""

from copy import deepcopy
from math import isfinite


def _root_paid_rejection(agent):
    """A root researcher can encounter an opaque bad fit before any child exists."""
    event = agent.events[-1]
    arguments, evaluation = event.get('arguments', {}), event.get('feedback', {})
    if (event.get('action') not in {'evaluate', 'finish'}
            or 'component' in arguments or arguments.get('target', 'response') != 'response'):
        return None
    model = arguments.get('model', 'model.py')
    candidate = agent.research_candidates.get(model, {})
    mse = evaluation.get('mse')
    if (evaluation.get('returncode') != 0
            or evaluation.get('representation_diagnostic', {}).get('status') != 'not_exposed'
            or not isinstance(mse, (int, float)) or not isfinite(mse)
            or mse <= agent.config.confirmation_mse_threshold
            or candidate.get('paid_rows') != len(agent.rows)
            or candidate.get('status') != 'evaluated'):
        return None
    parent_id = evaluation.get('research_question_id') or 'task'
    owner = next((q for q in agent.questions if q['id'] == parent_id
                  and q.get('parent_id') is None), None)
    if owner is None:
        return None
    return owner, {'parent_id': owner['id'], 'evaluated_model': model,
                   'phase': 'representation_unassessed'}


def open_representation_inquiry(agent):
    """Reuse the ordinary child loop once for each actual authored model source."""
    if (agent.arm != 'hegel' or agent.provider is None or agent.model is not None
            or agent._inquiry_stack or not agent.events
            or any(q.get('flat_review', {}).get('status') == 'running' for q in agent.questions)
            or agent.config.max_model_calls - agent.costs['model_calls'] <= 1
            or agent.costs['python_execution_seconds'] >= agent.config.execution_seconds
            or agent.costs['reported_tokens'] is None
            or agent.costs['reported_tokens'] >= agent.config.token_budget):
        return False
    owner = agent._parent_research()
    root_rejection = owner is None
    if owner is None:
        rejected = _root_paid_rejection(agent)
        if rejected is None:
            return False
        owner, focus = rejected
    else:
        focus = owner['parent_research']
    model = focus.get('evaluated_model')
    evaluation = agent.research_candidates.get(model, {}).get('current_evaluation', {})
    unassessed_method = (focus.get('phase') == 'prediction_method_revision'
        and evaluation.get('returncode') == 0
        and evaluation.get('representation_diagnostic', {}).get('status') == 'not_exposed')
    if focus.get('phase') != 'representation_unassessed' and not unassessed_method:
        return False
    if not model or not (agent.workspace / model).is_file():
        return False
    source = (agent.workspace / model).read_text()
    previous = [q for q in agent.questions if q.get('origin') == 'representation_research']
    if any((q.get('representation_model') == model
            and source in (q.get('source_at_start'), q.get('source_at_return')))
           or (q.get('returned_model') == model and q.get('returned_model_source') == source)
           for q in previous):
        return False
    question = {
        'id': f'representation_research_{len(previous)}',
        'origin': 'representation_research',
        'parent_id': focus['parent_id'], 'depends_on': [],
        'parent_research_question_id': owner['id'],
        'representation_model': model, 'source_at_start': source,
        'question': f'Investigate the actual input representation of your existing predictor {model}. '
            'Inspect its current source and expose the complete inputs and routing it actually uses '
            'through the existing represent(row), predict_features(features), and, when fitting is '
            'needed, fit_features(features, targets) interface. Adapt the fitting path as well as '
            'prediction; the explicit interface does not call the old fit(rows). Initially preserve '
            'your own scientific hypothesis while making its input contract executable. Include '
            'all information your downstream predictor actually needs, rather than an unrelated '
            'diagnostic view. Evaluate the authored model on the existing paid records, inspect '
            'the computed collision witnesses and execution errors, and research any actual '
            'conflict using those records. Inspect relevant fitting and representation code in '
            'the inherited research programs and source_catalog.json before rebuilding it. '
            'Read the complete relevant functions; archived fitted coefficients are not current '
            'evidence. Choose whether to reuse, refit or revise that source using the current paid '
            'counterexamples. Choose any scientific revision yourself. Do not purchase '
            'new observations in this local investigation; return any request for new evidence to '
            'the parent. Return your executable artifact and the measured diagnostic, or state '
            'what remains unresolved. Local finish returns to the parent. No observed collision '
            'and successful execution do not establish independent confirmation.'}
    if root_rejection:
        question['root_paid_rejection'] = {
            'event_index': len(agent.events) - 1, 'model': model,
            'paid_rows': len(agent.rows), 'mse': evaluation['mse']}
    parent_event = len(agent.events) - 1
    result = agent._start_inquiry({'question': question, 'max_model_calls': 8},
                                  parent_event_index=parent_event)
    focus['representation_inquiry_id'] = question['id']
    agent.events[parent_event].setdefault('feedback', {})['representation_research'] = result
    return True


def record_representation_return(agent, frame, result):
    """Retain the actual returned draft so an unresolved edit is not audited twice."""
    if result['question'].get('origin') != 'representation_research':
        return
    question = next(q for q in agent.questions if q['id'] == frame['question_id'])
    model = question['representation_model']
    path = agent.workspace / model
    question['source_at_return'] = path.read_text() if path.is_file() else None
    question['representation_return'] = {key: deepcopy(result[key]) for key in
        ('status', 'model_calls', 'artifacts', 'event_indices') if key in result}
    if question.get('root_paid_rejection'):
        # A real child return creates the ordinary parent synthesis; no failed
        # fresh confirmation or prior child is needed to reach that existing loop.
        question['research_return'] = {
            **deepcopy(question['representation_return']),
            'question_id': question['id'], 'parent_id': question['parent_id'],
            'assignment': deepcopy(question['assignment']), 'revision_indices': []}
        agent._prepare_parent_research()
        owner = next(q for q in agent.questions
                     if q['id'] == question['parent_research_question_id'])
        if owner.get('parent_research'):
            owner['parent_research']['evaluated_model'] = model
    for index in reversed(result.get('event_indices', [])):
        event = agent.events[index]
        args, feedback = event.get('arguments', {}), event.get('feedback', {})
        evaluation = feedback.get('evaluation', {})
        if (event.get('action') != 'finish' or not args.get('model')
                or 'component' in args or args.get('target', 'response') != 'response'
                or evaluation.get('returncode') != 0 or evaluation.get('error')):
            continue
        returned = evaluation.get('executed_path') or args['model']
        path = agent.workspace / returned
        if path.is_file():
            question.update(returned_model=returned, returned_model_source=path.read_text())
            owner = next(q for q in agent.questions
                         if q['id'] == question['parent_research_question_id'])
            if owner.get('parent_research'):
                owner['parent_research']['evaluated_model'] = returned
        break
