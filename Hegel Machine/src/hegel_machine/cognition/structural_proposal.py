"""Reuse the historical authored-draft -> paid trial -> revision work loop."""

from copy import deepcopy

from .structural_inquiry import execute_structural_research, structural_summary


STRUCTURAL_PROPOSAL_INSTRUCTION = """
This is the host-scheduled structural proposal stage in structural_proposal.
Use your failed prospective predictions and paid counterexamples to propose an
executable explanation. Choose its coordinates, history, scientific terms,
conditions, parameters and competing alternatives yourself. The earlier
prediction method and its measured failures remain available.
Write your bind(public_spec,observations,memory) draft to the assigned proposal_path.
Writing that file immediately executes the existing research_update contract on
paid evidence and returns structural_trial before your next decision. Its ordinary
Python rows and executable hypotheses/basis/candidate_additions are yours to author;
the host supplies no scientific family or formula. The same question retains the
tested candidate bank and rejected programs. Supply candidate_additions explicitly
when revising a bank that already exists; changing hypotheses does not replace it.
Inspect actual errors, contradictions and surviving predictions, then revise your
draft, calculate in Python, propose or select an experiment, or return an unresolved
finding. A refuted local candidate does not end this proposal stage. The ordinary
query action can execute your chosen experiment; writing a draft buys no observations.
Its returned next_action is a suggestion that you choose whether to execute.
Exported models are executable local explanations. Evaluate a chosen predictor
before adopting or composing it. Local finish {model:'your_predictor.py'} evaluates
your explicitly returned predictor on paid evidence and returns it to the parent;
finish {lesson:'unresolved finding'} can instead return without a predictor.
Neither a draft trial nor a local finish submits or independently confirms the
parent. The enclosing parent chooses its next step and ordinary fresh confirmation.
"""


def open_structural_proposal(agent):
    """Schedule one draft-practice stage per failed-confirmation method review."""
    if (agent.arm != 'hegel' or agent.provider is None or agent.model is not None
            or agent._inquiry_stack or not agent.events
            or any(q.get('flat_review', {}).get('status') == 'running' for q in agent.questions)
            or agent.config.max_model_calls - agent.costs['model_calls'] <= 1
            or agent.costs['python_execution_seconds'] >= agent.config.execution_seconds
            or agent.costs['compute_cost'] >= agent.config.compute_budget
            or agent.costs['reported_tokens'] is None
            or agent.costs['reported_tokens'] >= agent.config.token_budget
            or agent.config.observation_budget - agent.costs['observation_cost']
                < agent.config.confirmation_rows):
        return False
    owner = agent._parent_research()
    if owner is None or owner.get('structural_proposal') is not None:
        return False
    focus = owner['parent_research']
    if focus['status'] != 'running' or focus.get('phase') != 'prediction_method_revision':
        return False
    previous = None
    if agent.relation_research:
        # A manually adopted bank still needs no extra host stage. An earlier
        # host proposal, however, must not permanently block structural work
        # after a later independent failure of the same parent investigation.
        if owner.get('origin') != 'fresh_confirmation_failure':
            return False
        confirmation = owner['failed_confirmation']['confirmation_index']
        by_id = {question['id']: question for question in agent.questions}
        for candidate in reversed(agent.questions):
            prior_owner = by_id.get(candidate.get('parent_research_question_id'), {})
            prior = prior_owner.get('structural_proposal', {})
            if (candidate.get('origin') == 'structural_proposal'
                    and candidate['id'] in agent.relation_research
                    and candidate.get('parent_id') == focus['parent_id']
                    and prior.get('origin') == 'host_scheduled_structural_proposal'
                    and prior.get('question_id') == candidate['id']
                    and prior_owner.get('origin') == 'fresh_confirmation_failure'
                    and prior_owner['failed_confirmation']['confirmation_index'] < confirmation):
                previous = candidate
                break
        if previous is None:
            return False
    identifier = 'structure:' + owner['id']
    path = 'shared/structural_proposals/' + owner['id'] + '/binding.py'
    question = {
        'id': identifier, 'origin': 'structural_proposal',
        'parent_id': focus['parent_id'], 'depends_on': [],
        'parent_research_question_id': owner['id'], 'proposal_path': path,
        'question': 'Propose and test new executable structural hypotheses from the actual '
            'paid counterexamples and failed predictions in the enclosing method review. '
            'Choose the scientific coordinates, terms, applicability and alternatives yourself. '
            'Author bind(public_spec,observations,memory) at ' + path + '. Each write to this '
            'draft executes the existing structural researcher on paid evidence and returns '
            'its actual errors, candidate losses and suggested distinguishing experiment. '
            'Use that feedback to revise, explicitly add alternatives to the retained bank, '
            'select an experiment, or report an unresolved finding. Evaluate any chosen '
            'executable predictor and explicitly return it to the parent when ready. '
            'This host-scheduled stage provides no scientific answer; local work does not '
            'establish independent parent confirmation.'}
    if previous is not None:
        # The original review retains its return and event references. Continue
        # the same authored file/session/bank; only this invocation's caller
        # changes. Opening the inquiry itself executes no draft or experiment.
        question = {**deepcopy(previous), 'parent_research_question_id': owner['id']}
        identifier, path = question['id'], question['proposal_path']
    parent_event = len(agent.events) - 1
    result = agent._start_inquiry({'question': question, 'max_model_calls': 12},
                                  parent_event_index=parent_event)
    owner['structural_proposal'] = {
        'question_id': identifier, 'proposal_path': path, 'status': 'running',
        'origin': 'host_scheduled_structural_proposal',
        'started_at_model_call': agent.costs['model_calls']}
    agent.events[parent_event].setdefault('feedback', {})['structural_proposal'] = result
    return True


def _trial_summary(feedback):
    summary = {key: deepcopy(feedback[key]) for key in
        ('operator', 'research_question_id', 'binding_path', 'status', 'error', 'next_action')
        if key in feedback}
    relation = feedback.get('relation_research', {})
    summary['relation_research'] = {key: deepcopy(relation[key]) for key in
        ('status', 'diagnosis', 'round_index', 'compute_cost', 'selected_program',
         'selected_probe', 'exported_models', 'export_errors', 'bound_row_count') if key in relation}
    summary['relation_research'].update(
        candidate_count=len(relation.get('candidates', [])),
        rejected_candidate_count=len(relation.get('rejected_candidates', [])),
        full_state_path='research_state.json.relation_research')
    summary['relation_research']['candidates'] = [
        {'candidate_index': index, **{key: deepcopy(item[key]) for key in
            ('status', 'fit_mse', 'validation_mse', 'confirmation_mse', 'paid_fit_residuals')
            if key in item}}
        for index, item in enumerate(relation.get('candidates', []))]
    return summary


def trial_structural_proposal(agent, action, arguments, laboratory):
    """Execute only this author's designated draft after its ordinary write."""
    if not agent._inquiry_stack:
        return None
    question = agent._inquiry_stack[-1]['question']
    if (question.get('origin') != 'structural_proposal' or action != 'write'
            or arguments.get('path') != question['proposal_path']):
        return None
    try:
        feedback = execute_structural_research(agent, {'path': question['proposal_path']},
                                              laboratory, question)
    except (ValueError, KeyError, TypeError, StopIteration, OSError) as error:
        feedback = {'operator': 'research_update', 'status': 'structural_trial_failed',
                    'research_question_id': question['id'], 'binding_path': question['proposal_path'],
                    'error': f'{type(error).__name__}: {error}'}
        previous = agent.relation_research.get(question['id'])
        if previous is not None:
            feedback['relation_research'] = structural_summary(previous)
    question['last_structural_trial'] = _trial_summary(feedback)
    return feedback


def extract_existing_return_evaluation(agent, event_indices):
    """Read the actual local finish without changing its history or paid cutoff."""
    for index in reversed(event_indices):
        event = agent.events[index]
        arguments, feedback = event.get('arguments', {}), event.get('feedback', {})
        evaluation = feedback.get('evaluation', {})
        if (event.get('action') != 'finish' or not arguments.get('model')
                or 'component' in arguments or arguments.get('target', 'response') != 'response'
                or evaluation.get('returncode') != 0 or evaluation.get('error')):
            continue
        model = evaluation.get('executed_path') or arguments['model']
        summary = {'model': model, 'event_index': index,
            **{key: deepcopy(evaluation[key]) for key in
               ('returncode', 'mse', 'evaluation_scope', 'paid_residuals') if key in evaluation}}
        paid_rows = evaluation.get('paid_residuals', {}).get('evaluated_rows', evaluation.get('paid_rows'))
        if paid_rows is not None:
            summary['paid_rows'] = paid_rows
        return summary
    return None


def record_structural_return(agent, frame, result):
    """Keep the actual local return; the parent still decides adoption and confirmation."""
    if result['question'].get('origin') != 'structural_proposal':
        return
    question = frame['question']
    returned = {key: deepcopy(result[key]) for key in
                ('status', 'model_calls', 'artifacts', 'event_indices') if key in result}
    owner = next(q for q in agent.questions
                 if q['id'] == question['parent_research_question_id'])
    evaluation = extract_existing_return_evaluation(agent, returned.get('event_indices', []))
    if evaluation is not None and (agent.workspace / evaluation['model']).is_file():
        returned['evaluation'] = evaluation
        question['returned_model'] = evaluation['model']
        owner['structural_proposal']['returned_model'] = evaluation['model']
        owner['parent_research']['evaluated_model'] = evaluation['model']
    question['structural_return'] = returned
    owner['structural_proposal'].update(status=result['status'],
        returned_at_model_call=agent.costs['model_calls'], returned=deepcopy(returned))


def structural_proposal_context(agent):
    """Describe the active author's actual draft and trial without copying its paid rows."""
    if not agent._inquiry_stack:
        return None
    question = agent._inquiry_stack[-1]['question']
    if question.get('origin') != 'structural_proposal':
        return None
    bank = agent.relation_research.get(question['id'], {})
    return {
        'origin': 'host_scheduled_structural_proposal', 'question_id': question['id'],
        'parent_research_question_id': question['parent_research_question_id'],
        'proposal_path': question['proposal_path'],
        'last_trial': deepcopy(question.get('last_structural_trial')),
        'candidate_bank': {'candidate_count': len(bank.get('candidates', [])),
            'rejected_candidate_count': len(bank.get('rejected_candidates', [])),
            'round_index': bank.get('round_index'),
            'full_state_path': 'research_state.json.relation_research'}}
