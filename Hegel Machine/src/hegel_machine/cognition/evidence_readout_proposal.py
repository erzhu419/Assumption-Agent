"""Execute an author's readout of the active question's paid evidence."""

from copy import deepcopy
import json

from .inquiry_revision import analyze_paid_evidence, readout_record_references


EVIDENCE_READOUT_INSTRUCTION = """
This active question already has purchased observations. Author
readout(result, public_spec, memory) at evidence_readout_proposal.path. Writing
that assigned file immediately executes the existing analyze_evidence interface
on the listed laboratory record_indices and returns evidence_readout_trial.
result is a Python dictionary: result['observations'] contains those actual
records, each with 'rows' and global response 'row_indices';
each record's 'record_index' is its position in laboratory_observations.json.
'observation_index' is the first response row ID, not the record ID. The
proposal's record_references maps the selected record IDs to response row IDs
and exposes their actual control_runs, including any initial history. Runs name
public inputs at inclusive reset-relative steps, not inferred experiment roles.
Compare these complete inputs before interpreting a numerical contrast.
result['analysis_scope'] describes their exact scope. Evidence and checks must
reference response row IDs from the selected records. To compare additional paid
experiments, explicitly include their record indices in analyze_evidence or
write {path:proposal.path,code:...,record_indices:[...]}. The explicit list
selects both previous reference records and new records from the paid archive.
Changing the source or its notes alone does not change the input selection.
Choose your own
claims, coordinates, numerical checks, expected values and conclusions. Return
{status:'supported'|'refuted'|'inconclusive', answer:..., evidence:[response row IDs]}.
Optional checks use the existing terms:[{row:ID,coefficient:number}], relation,
expected, atol and scope contract. Actual checks determine evidence_status and
return to this same question and its parent. Inspect errors and revise the same
file when needed. Ordinary executed analyses remain available. This readout
neither buys observations nor refits, submits or confirms a response predictor.
"""


def prepare_evidence_readout(agent, question):
    """Assign record references without changing or executing the scientific question."""
    current = next((q for q in agent.questions if q['id'] == question['id']), None)
    if current is None or agent._active_question_id != current['id']:
        return None
    reviewed = {index for record in agent.revisions
                if record.get('question_id') == current['id']
                and record.get('status') == 'answered'
                for index in record.get('returned_answer', {}).get('evidence', [])}
    acquired = set(current.get('acquired_evidence', [])) - reviewed
    if not acquired:
        return None
    indices = [index for index, record in enumerate(agent.observations)
               if acquired.intersection(record.get('row_indices', []))]
    if not indices:
        return None
    proposal = current.setdefault('evidence_readout_proposal', {})
    proposal.update(question_id=current['id'],
                    path=f"shared/evidence_readouts/{current['id']}/readout.py",
                    record_indices=indices,
                    record_references=readout_record_references(
                        [agent.observations[i] for i in indices], indices),
                    response_row_count=sum(len(agent.observations[i]['row_indices'])
                                           for i in indices))
    return deepcopy(proposal)


def _active_proposal(agent):
    question = (agent._inquiry_stack[-1]['question'] if agent._inquiry_stack else
                next((q for q in agent.questions if q['id'] == agent._active_question_id), None))
    if question is None:
        return None, None
    proposal = question.get('evidence_readout_proposal')
    if proposal is None or proposal['question_id'] != question['id']:
        return None, None
    proposal['record_references'] = readout_record_references(
        [agent.observations[i] for i in proposal['record_indices']],
        proposal['record_indices'])
    return question, proposal


def _trial_summary(record, question, proposal):
    feedback = {'operator': 'analyze_evidence', 'research_question_id': question['id'],
                'path': proposal['path'], 'revision_index': record['index'],
                'record_indices': list(record['record_indices'])}
    feedback.update({key: deepcopy(record[key]) for key in
                     ('status', 'evidence_status', 'readout_error', 'analysis_scope')
                     if key in record})
    if 'check_results' in record:
        feedback['check_results'] = [{key: deepcopy(value) for key, value in check.items()
                                      if key != 'witnesses'}
                                     for check in record['check_results']]
    if 'returned_answer' in record:
        answer = record['returned_answer']
        if len(json.dumps(answer)) <= 4000:
            feedback['returned_answer'] = deepcopy(answer)
        else:
            feedback['returned_answer_source'] = {
                'file': 'research_state.json',
                'json_path': ['revisions', record['index'], 'returned_answer']}
    return feedback


def trial_evidence_readout(agent, action, arguments, laboratory):
    """Run only the active author's assigned write through the existing readout engine."""
    question, proposal = _active_proposal(agent)
    if (proposal is None or action != 'write'
            or arguments.get('path') != proposal['path']):
        return None
    try:
        record = analyze_paid_evidence(agent, proposal['path'], laboratory, question,
            record_indices=arguments.get('record_indices', proposal['record_indices']))
        feedback = _trial_summary(record, question, proposal)
    except (ValueError, TypeError, KeyError, OSError) as error:
        feedback = {'operator': 'analyze_evidence', 'status': 'readout_trial_failed',
                    'research_question_id': question['id'], 'path': proposal['path'],
                    'record_indices': list(proposal['record_indices']),
                    'readout_error': f'{type(error).__name__}: {error}'}
    if feedback.get('readout_error'):
        feedback['evidence_selection'] = {
            'selected_record_indices': deepcopy(feedback['record_indices']),
            'available_records': len(agent.observations),
            'archive': 'laboratory_observations.json',
            'instruction': 'The readout receives only selected records. To include an old '
                'reference, change record_indices in this action or in the next write. '
                'Source edits and notes do not add records.',
            'analyze_action': {'action': 'run', 'arguments': {
                'operator': 'analyze_evidence', 'path': proposal['path'],
                'record_indices': deepcopy(feedback['record_indices']),
                'research_question': {'id': question['id']}}}}
    proposal['last_trial'] = deepcopy(feedback)
    return feedback


def evidence_readout_context(agent):
    """Expose only an unfinished paid scope; retain completed proposals in the archive."""
    question, proposal = _active_proposal(agent)
    if proposal is None:
        return None
    remaining = set(proposal['record_indices'])
    # Explicit analyze_evidence/analyze_revision calls also complete this work.
    # A later repair of an older revision must take precedence over its index.
    by_index = {record['index']: record for record in agent.revisions}
    records = []
    for event in reversed(agent.events):
        feedback = event.get('feedback', {})
        for key in ('revision', 'evidence_readout_trial'):
            result = feedback.get(key)
            if isinstance(result, dict):
                index = result.get('revision_index', result.get('index'))
                if index in by_index:
                    records.append(by_index[index])
                elif result.get('readout_error'):
                    records.append({**result,
                        'question_id': result.get('research_question_id')})
    records.extend(record for record in reversed(agent.revisions)
                   if record.get('analysis_attempts') and record not in records)
    for record in records:
        if (record.get('question_id') != question['id']
                or record.get('target', 'response') != 'response'):
            continue
        scope = set(record.get('analysis_scope', {}).get(
            'record_indices', record.get('record_indices', [])))
        if not remaining.intersection(scope):
            continue
        answer = record.get('returned_answer') or {}
        attempt = (record.get('analysis_attempts') or [{}])[-1]
        if (record.get('status') != 'answered' or record.get('readout_error')
                or attempt.get('error') or answer.get('status') not in
                    {'supported', 'refuted', 'inconclusive'}):
            return deepcopy(proposal)
        remaining.difference_update(scope)
        if not remaining:
            return None
    return deepcopy(proposal)
