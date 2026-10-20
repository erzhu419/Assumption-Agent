"""Paid fixture readouts test execution and handoff, not scientific discovery."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.evidence_readout_proposal import (
    EVIDENCE_READOUT_INSTRUCTION, evidence_readout_context,
    prepare_evidence_readout, trial_evidence_readout,
)
from test_inquiry_revision import PublicLab, action
from test_scientific_generalization import execute, prepared
from test_structural_proposal import U, binding_source
from test_common_archive_online_research import ScriptedProvider


PARENT = {'id': 'task', 'question': 'Explain the public response',
          'parent_id': None, 'depends_on': []}
QUESTION = {'id': 'paid_pairs', 'question': 'Do the matched paid responses agree?',
            'parent_id': 'task', 'depends_on': []}


def paid_scene(path, *, provider=None, calls=20):
    agent, lab = prepared(path, provider, calls=calls), PublicLab()
    execute(agent, lab, 'query', {'actions': [action(9)], 'question': PARENT})
    execute(agent, lab, 'query', {'actions': [action(2, 2), action(4, 4)],
                                'question': QUESTION})
    question = next(q for q in agent.questions if q['id'] == QUESTION['id'])
    return agent, lab, question


def open_readout(agent, question):
    agent._start_inquiry({'question': {'id': question['id'],
                                    'parent_id': question['parent_id']},
                          'max_model_calls': 4})
    proposal = prepare_evidence_readout(agent, question)
    return proposal['path']


def readout_source(*, expected=0., answer='The local paired contrast is zero.'):
    return f'''def readout(result, public_spec, memory):
    assert result['record_indices'] == [1, 2]
    assert public_spec['controls'] == ['u']
    indices = [index for record in result['observations']
               for index in record['row_indices']]
    assert indices == [1, 2, 3, 4]
    return {{'status': 'supported', 'answer': {answer!r}, 'evidence': indices,
             'checks': [{{'id': 'paired_contrast',
                         'terms': [{{'row': indices[0], 'coefficient': 1}},
                                   {{'row': indices[1], 'coefficient': -1}}],
                         'relation': 'eq', 'expected': {expected!r}, 'atol': 1e-9,
                         'scope': {{'protocol': 'the first matched paid pair'}}}}]}}
'''


def test_preparation_selects_question_records_without_execution_or_scientific_changes(tmp_path):
    agent, lab, question = paid_scene(tmp_path)
    question.pop('evidence_readout_proposal')
    before = deepcopy((agent.costs, agent.observations, agent.questions, agent.revisions))
    before_actions = deepcopy(lab.actions)
    note = agent.workspace / 'research_notes.md'
    note.write_text('Existing author analysis remains unresolved.')

    proposal = prepare_evidence_readout(agent, question)

    assert proposal == {'question_id': question['id'],
                        'path': 'shared/evidence_readouts/paid_pairs/readout.py',
                        'record_indices': [1, 2], 'response_row_count': 4,
                        'record_references': [
                            {'record_index': 1, 'observation_index': 1,
                             'response_row_ranges': [[1, 2]], 'initial_history_length': 0,
                             'control_runs': [{'steps': [0, 1], 'controls': {'u': 2}}]},
                            {'record_index': 2, 'observation_index': 3,
                             'response_row_ranges': [[3, 4]], 'initial_history_length': 0,
                             'control_runs': [{'steps': [0, 1], 'controls': {'u': 4}}]}]}
    assert question['evidence_readout_proposal'] == proposal
    unchanged = deepcopy(agent.questions)
    next(q for q in unchanged if q['id'] == question['id']).pop('evidence_readout_proposal')
    assert (agent.costs, agent.observations, unchanged, agent.revisions) == before
    assert lab.actions == before_actions and lab.paid == 5
    assert agent.provider.requests == [] and agent.costs['model_calls'] == 0
    assert not agent._inquiry_stack and not (agent.workspace / proposal['path']).exists()
    assert note.read_text() == 'Existing author analysis remains unresolved.'
    assert 'observations' not in proposal and 'rows' not in proposal
    assert evidence_readout_context(agent) == proposal


def test_selected_readout_keeps_archive_record_ids_distinct_from_response_row_ids(tmp_path):
    agent, lab = prepared(tmp_path), PublicLab()
    agent.config = replace(agent.config, observation_budget=100)
    execute(agent, lab, 'query', {'actions': [action(i, i + 1) for i in range(20)],
                                'question': PARENT})
    execute(agent, lab, 'query', {'actions': [action(i, i + 1) for i in range(20, 26)],
                                'question': QUESTION})
    question = next(q for q in agent.questions if q['id'] == QUESTION['id'])
    path = open_readout(agent, question)
    original = deepcopy(agent.observations)
    references = question['evidence_readout_proposal']['record_references']
    assert [item['record_index'] for item in references] == list(range(20, 26))
    assert [item['observation_index'] for item in references] == list(range(40, 52, 2))
    source = '''import json
def readout(result, public_spec, memory):
    archive = json.load(open('laboratory_observations.json'))
    assert result['record_indices'] == list(range(20, 26))
    by_id = {record['record_index']: record for record in result['observations']}
    assert list(by_id) == list(range(20, 26))
    for index, record in by_id.items():
        assert record['observation_index'] == 2 * index
        assert record['row_indices'] == [2 * index, 2 * index + 1]
        assert {key: value for key, value in record.items()
                if key != 'record_index'} == archive[index]
    assert result['analysis_scope']['record_references'][0] == {
        'record_index': 20, 'observation_index': 40, 'response_row_ranges': [[40, 41]],
        'initial_history_length': 0,
        'control_runs': [{'steps': [0, 0], 'controls': {'u': 20}},
                         {'steps': [1, 1], 'controls': {'u': 21}}]}
    return {'status': 'inconclusive', 'answer': 'Only record addressing was inspected.',
            'evidence': [row for record in by_id.values() for row in record['row_indices']]}
'''
    result = execute(agent, lab, 'write', {'path': path, 'code': source})
    trial = result['evidence_readout_trial']
    assert trial['status'] == 'answered' and 'readout_error' not in trial
    assert trial['analysis_scope']['record_references'] == references
    assert agent.observations == original
    assert all('record_index' not in record for record in agent.observations)
    assert lab.paid == 52 and agent.costs['model_calls'] == 0


@pytest.mark.parametrize('action_kind', ['write', 'analyze'])
def test_readout_can_explicitly_include_a_previously_reviewed_reference(tmp_path, action_kind):
    agent, lab, question = paid_scene(tmp_path)
    path = open_readout(agent, question)
    execute(agent, lab, 'write', {'path': path, 'code': readout_source()})
    execute(agent, lab, 'query', {'actions': [action(2, 2)],
                                'question': {'id': question['id']}})
    proposal = prepare_evidence_readout(agent, question)
    assert proposal['record_indices'] == [3]
    source = '''def readout(result, public_spec, memory):
    records = {r['record_index']: r for r in result['observations']}
    reference, new = records[1], records[3]
    evidence = reference['row_indices'] + new['row_indices']
    return {'status':'supported', 'answer':'The selected measured responses agree.',
        'evidence':evidence, 'checks':[{'id':'measured_difference',
            'terms':[{'row':reference['row_indices'][0],'coefficient':-1},
                     {'row':new['row_indices'][0],'coefficient':1}],
            'relation':'eq','expected':0.}]}
'''
    before = deepcopy((agent.observations, lab.actions, agent.confirmations))
    failed = execute(agent, lab, 'write', {'path': path, 'code': source})['evidence_readout_trial']
    assert 'KeyError: 1' in failed['readout_error']
    scope = failed['evidence_selection']
    assert scope['selected_record_indices'] == [3]
    assert scope['available_records'] == 4
    if action_kind == 'write':
        fixed = execute(agent, lab, 'write', {'path': path, 'code': source,
                         'record_indices': [1, 3]})['evidence_readout_trial']
    else:
        retry = deepcopy(scope['analyze_action'])
        retry['arguments']['record_indices'] = [1, 3]
        fixed = execute(agent, lab, retry['action'], retry['arguments'])['revision']
    assert fixed['evidence_status'] == 'supported'
    assert fixed['analysis_scope']['record_indices'] == [1, 3]
    assert fixed['analysis_scope']['response_row_ranges'] == [[1, 2], [5, 6]]
    assert (agent.observations, lab.actions, agent.confirmations) == before
    assert agent.costs['model_calls'] == 0 and lab.paid == 7
    assert 'readout_error' in agent.revisions[-2]


def test_current_existing_question_is_required_and_an_answered_revision_is_not_reopened(tmp_path):
    agent, lab, question = paid_scene(tmp_path)
    count = len(agent.questions)
    assert prepare_evidence_readout(agent, {'id': 'not_a_question'}) is None
    assert prepare_evidence_readout(agent, agent.questions[0]) is None
    question['acquired_evidence'] = []
    assert prepare_evidence_readout(agent, question) is None
    assert len(agent.questions) == count

    question['acquired_evidence'] = [1, 2, 3, 4]
    path = open_readout(agent, question)
    execute(agent, lab, 'write', {'path': path, 'code': readout_source()})
    agent._active_question_id = question['id']
    before = deepcopy(agent.questions)
    assert prepare_evidence_readout(agent, question) is None
    assert agent.questions == before and len(agent.revisions) == 1


def test_restored_pending_proposal_renders_record_references_without_query_or_source_edit(tmp_path):
    agent, lab, question = paid_scene(tmp_path)
    proposal = question['evidence_readout_proposal']
    references = proposal.pop('record_references')
    retained = deepcopy(proposal)
    source = agent.workspace / proposal['path']
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text('# Retained author source; do not execute while rendering.\n')
    before = deepcopy((agent.observations, lab.actions, agent.costs, agent.revisions))

    request = agent._request(lab)

    assert request['evidence_readout_proposal'] == {**retained, 'record_references': references}
    assert source.read_text() == '# Retained author source; do not execute while rendering.\n'
    assert (agent.observations, lab.actions, agent.costs, agent.revisions) == before
    assert lab.paid == 5 and not agent.confirmations and agent.costs['model_calls'] == 0


def test_new_paid_rows_under_an_answered_question_receive_their_own_readout_scope(tmp_path):
    agent, lab, question = paid_scene(tmp_path)
    path = open_readout(agent, question)
    execute(agent, lab, 'write', {'path': path, 'code': readout_source()})
    first_answer = deepcopy(agent.revisions[0]['returned_answer'])
    execute(agent, lab, 'query', {'actions': [action(7, 7)],
                                'question': {'id': question['id']}})
    costs = deepcopy(agent.costs)

    proposal = prepare_evidence_readout(agent, question)

    assert proposal['record_indices'] == [3] and proposal['response_row_count'] == 2
    assert proposal['path'] == path and proposal['last_trial']['revision_index'] == 0
    assert evidence_readout_context(agent) == proposal
    assert EVIDENCE_READOUT_INSTRUCTION in agent._request(lab)['task_interface']
    assert agent.costs == costs and lab.paid == 7
    source = '''def readout(result, public_spec, memory):
    assert result['record_indices'] == [3]
    indices = result['observations'][0]['row_indices']
    assert indices == [5, 6]
    return {'status': 'supported', 'answer': 'The newly purchased pair agrees.',
            'evidence': indices, 'checks': [{'id': 'new_pair',
                'terms': [{'row': 5, 'coefficient': 1}, {'row': 6, 'coefficient': -1}],
                'relation': 'eq', 'expected': 0., 'atol': 1e-9}]}
'''
    result = execute(agent, lab, 'write', {'path': path, 'code': source})
    assert result['evidence_readout_trial']['evidence_status'] == 'supported'
    assert result['evidence_readout_trial']['analysis_scope']['response_row_ranges'] == [[5, 6]]
    assert agent.revisions[0]['returned_answer'] == first_answer
    assert agent.questions[0]['child_answers'][-1]['evidence'] == [5, 6]
    assert lab.paid == 7 and agent.costs['model_calls'] == 0


def terminal_paid_scene(path, status, provider):
    agent, lab, question = paid_scene(path, provider=provider, calls=3)
    execute(agent, lab, 'query', {'actions': [action(7, 7)],
                                'question': {'id': question['id']}})
    execute(agent, lab, 'write', {'path': 'historical_readout.py',
        'code': readout_source(expected=1. if status == 'refuted' else 0.)})
    execute(agent, lab, 'run', {'operator': 'analyze_evidence',
        'path': 'historical_readout.py', 'record_indices': [1, 2],
        'research_question': {'id': question['id']}})
    assert question['status'] == status
    return agent, lab, question


@pytest.mark.parametrize('status', ['supported', 'refuted'])
def test_reopened_checked_question_reaches_author_and_waits_for_new_readout(tmp_path, status):
    path = f"shared/evidence_readouts/{QUESTION['id']}/readout.py"
    fresh_source = '''def readout(result, public_spec, memory):
    assert result['record_indices'] == [3]
    return {'status': 'supported', 'answer': 'The newly purchased pair agrees.',
            'evidence': [5, 6], 'checks': [{'id': 'new_pair',
                'terms': [{'row': 5, 'coefficient': 1}, {'row': 6, 'coefficient': -1}],
                'relation': 'eq', 'expected': 0., 'atol': 1e-9}]}
'''
    provider = ScriptedProvider([
        ('run', {'code': "print('Inspect the existing paid work')"}),
        ('write', {'path': path, 'code': fresh_source}),
        ('run', {'code': "print('Parent receives the new checked answer')"}),
    ])
    agent, lab, question = terminal_paid_scene(tmp_path, status, provider)
    old_question = deepcopy(question)
    old_revision = deepcopy(agent.revisions[0])
    old_answers = deepcopy(agent.questions[0]['child_answers'])
    execute(agent, lab, 'run', {'operator': 'inquire',
        'question': {'id': question['id'], 'parent_id': question['parent_id']},
        'max_model_calls': 2})
    proposal = prepare_evidence_readout(agent, question)

    agent._before_model_turn(lab)
    assert agent._inquiry_stack[-1]['question_id'] == question['id']
    assert question['status'] == 'reopened'
    for key in ('answer', 'revision_index', 'check_results', 'evidence', 'assignment'):
        assert question[key] == old_question[key]
    assert agent.questions[0]['child_answers'] == old_answers
    assert proposal['record_indices'] == [3]
    result = agent.run(lab)

    for request in provider.requests[:2]:
        assert request['llm_inquiry']['question_id'] == question['id']
        assert request['evidence_readout_proposal']['record_indices'] == [3]
        assert request['llm_inquiry']['question']['status'] == 'reopened'
    assert provider.requests[1]['events'][-1]['feedback']['returncode'] == 0
    assert not provider.requests[2].get('llm_inquiry')
    assert agent.llm_inquiries[question['id']]['status'] == 'answered'
    assert agent.llm_inquiries[question['id']]['model_calls'] == 2
    assert agent.revisions[0] == old_revision
    assert agent.questions[0]['child_answers'][:len(old_answers)] == old_answers
    answer = agent.questions[0]['child_answers'][-1]
    assert answer['evidence'] == [5, 6] and answer['evidence_status'] == 'supported'
    assert answer['revision_index'] == 1
    assert result['costs']['model_calls'] == 3 and lab.paid == 7
    assert not agent.confirmations and agent.model is None


def test_reopened_question_without_new_answer_returns_at_its_normal_call_budget(tmp_path):
    provider = ScriptedProvider([
        ('run', {'code': "print('Inspect the paid work')"}),
        ('run', {'code': "print('The new evidence is still unresolved')"}),
        ('run', {'code': "print('Parent retains the unresolved result')"}),
    ])
    agent, lab, question = terminal_paid_scene(tmp_path, 'refuted', provider)
    old_question = deepcopy(question)
    old_revision = deepcopy(agent.revisions[0])
    old_answers = deepcopy(agent.questions[0]['child_answers'])
    execute(agent, lab, 'run', {'operator': 'inquire',
        'question': {'id': question['id'], 'parent_id': question['parent_id']},
        'max_model_calls': 2})
    prepare_evidence_readout(agent, question)

    agent._before_model_turn(lab)
    assert agent._inquiry_stack[-1]['question_id'] == question['id']
    agent.run(lab)

    assert all(request.get('llm_inquiry') for request in provider.requests[:2])
    assert not provider.requests[2].get('llm_inquiry')
    returned = provider.requests[2]['events'][-1]['feedback']['inquiry_return']
    assert returned['status'] == 'model_call_budget_exhausted'
    assert returned['model_calls'] == 2 and returned['question']['status'] == 'reopened'
    assert agent.llm_inquiries[question['id']]['status'] == 'model_call_budget_exhausted'
    for key in ('answer', 'revision_index', 'check_results', 'evidence', 'assignment'):
        assert question[key] == old_question[key]
    assert agent.revisions == [old_revision]
    assert agent.questions[0]['child_answers'] == old_answers
    assert lab.paid == 7 and not agent.confirmations and agent.model is None


def test_assigned_write_executes_paid_checks_and_returns_actual_parent_answer(tmp_path):
    agent, lab, question = paid_scene(tmp_path)
    original = {key: deepcopy(question[key]) for key in
                ('id', 'question', 'parent_id', 'depends_on', 'assignment')}
    path = open_readout(agent, question)
    note = agent.workspace / 'research_notes.md'
    note.write_text('The author retains the same local notebook.')
    before = deepcopy((agent.observations, lab.actions, agent.confirmations))
    costs = deepcopy(agent.costs)
    result = execute(agent, lab, 'write', {'path': path, 'code': readout_source()})

    trial = result['evidence_readout_trial']
    assert trial['status'] == 'answered' and trial['evidence_status'] == 'supported'
    assert trial['record_indices'] == [1, 2]
    assert trial['analysis_scope']['response_row_ranges'] == [[1, 4]]
    assert trial['analysis_scope']['response_row_count'] == 4
    assert trial['check_results'][0]['value'] == 0.
    assert trial['check_results'][0]['status'] == 'satisfied'
    assert 'witnesses' not in trial['check_results'][0]
    assert trial['returned_answer']['answer'] == 'The local paired contrast is zero.'
    parent = next(q for q in agent.questions if q['id'] == 'task')
    answer = parent['child_answers'][-1]
    assert answer['question_id'] == question['id'] and answer['status'] == 'supported'
    assert answer['check_results'][0]['witnesses'][0]['row'] == 1
    assert {key: question[key] for key in original} == original
    assert note.read_text() == 'The author retains the same local notebook.'
    assert (agent.observations, lab.actions, agent.confirmations) == before
    assert agent.costs['observation_cost'] == costs['observation_cost'] == lab.paid == 5
    assert agent.costs['model_calls'] == costs['model_calls'] == 0
    assert agent.costs['compute_cost'] - costs['compute_cost'] == 2
    assert agent.model is None and not agent.relation_research
    record = agent.revisions[trial['revision_index']]
    assert record['actions'] == record['forecasts'] == [] and 'observations' not in record
    assert question['evidence_readout_proposal']['last_trial'] == trial
    request = agent._request(lab)
    assert 'evidence_readout_proposal' not in request
    assert EVIDENCE_READOUT_INSTRUCTION not in request['task_interface']
    assert evidence_readout_context(agent) is None
    assert question['evidence_readout_proposal']['last_trial']['evidence_status'] == 'supported'
    assert 'readout(result, public_spec, memory)' in EVIDENCE_READOUT_INSTRUCTION
    saved = json.loads((agent.workspace / 'research_state.json').read_text())
    assert saved['revisions'][0]['check_results'][0]['witnesses'][1]['row'] == 2


def test_wrong_authored_relation_is_measured_and_returns_refutation(tmp_path):
    agent, lab, question = paid_scene(tmp_path)
    path = open_readout(agent, question)
    result = execute(agent, lab, 'write', {'path': path, 'code': readout_source(expected=1.)})

    trial = result['evidence_readout_trial']
    assert trial['evidence_status'] == 'refuted'
    assert trial['returned_answer']['authored_status'] == 'supported'
    assert trial['returned_answer']['status'] == question['status'] == 'refuted'
    assert trial['check_results'][0]['residual'] == -1.
    assert agent.questions[0]['child_answers'][-1]['status'] == 'refuted'
    assert agent.revisions[0]['authored_answer']['checks'][0]['expected'] == 1.
    assert lab.paid == 5 and not agent.confirmations and agent.model is None


def test_author_parent_revision_keeps_checked_answer_bound_to_the_actual_caller(tmp_path):
    agent, lab, question = paid_scene(tmp_path)
    path = open_readout(agent, question)
    assignment = deepcopy(question['assignment'])
    frame_assignment = deepcopy(agent._inquiry_stack[-1]['assignment'])
    before = deepcopy((agent.observations, lab.actions, agent.confirmations))
    result = execute(agent, lab, 'write', {'path': path, 'code': readout_source(),
        'research_question': {'id': question['id'], 'parent_id': None}})

    assert question['parent_id'] is None
    assert question['question'] == QUESTION['question']
    assert question['assignment'] == assignment and assignment['parent_id'] == 'task'
    frame = agent._inquiry_stack[-1]
    assert frame['parent_question'] == 'task' and frame['assignment'] == frame_assignment
    trial = result['evidence_readout_trial']
    assert trial['evidence_status'] == 'supported'
    caller = next(q for q in agent.questions if q['id'] == frame['parent_question'])
    returned = caller['child_answers'][-1]
    assert returned['question_id'] == question['id'] and returned['status'] == 'supported'
    assert returned['evidence'] == [1, 2, 3, 4]
    assert returned['check_results'][0]['status'] == 'satisfied'
    assert (agent.observations, lab.actions, agent.confirmations) == before
    assert lab.paid == 5 and agent.model is None


def test_author_parent_revision_keeps_structural_answer_bound_to_the_actual_caller(tmp_path):
    agent, lab, question = paid_scene(tmp_path)
    open_readout(agent, question)
    assignment = deepcopy(question['assignment'])
    frame_assignment = deepcopy(agent._inquiry_stack[-1]['assignment'])
    execute(agent, lab, 'write', {'path': 'local_relation.py',
        'code': binding_source(question['id'], [U])})
    before = deepcopy((agent.observations, lab.actions, agent.confirmations))
    result = execute(agent, lab, 'run', {'operator': 'research_update',
        'path': 'local_relation.py',
        'research_question': {'id': question['id'], 'parent_id': None}})

    assert question['parent_id'] is None and question['assignment'] == assignment
    frame = agent._inquiry_stack[-1]
    assert frame['parent_question'] == 'task' and frame['assignment'] == frame_assignment
    caller = next(q for q in agent.questions if q['id'] == frame['parent_question'])
    returned = caller['child_answers'][-1]
    assert returned['operator'] == 'research_update'
    assert returned['question_id'] == question['id']
    assert returned['answer'] == result['returned_answer']['answer']
    assert returned['answer']['candidates'][0]['fit_mse'] == 0.
    assert (agent.observations, lab.actions, agent.confirmations) == before
    assert lab.paid == 5 and agent.model is None


def test_readout_exception_returns_to_same_author_who_can_repair_the_assigned_file(tmp_path):
    agent, lab, question = paid_scene(tmp_path)
    path = open_readout(agent, question)
    failed = execute(agent, lab, 'write', {'path': path,
        'code': "def readout(result, public_spec, memory): raise ValueError('fixture analysis failed')\n"})

    trial = failed['evidence_readout_trial']
    assert 'fixture analysis failed' in trial['readout_error']
    assert 'returned_answer' not in trial and not agent.questions[0].get('child_answers')
    request = agent._request(lab)
    assert request['llm_inquiry']['question_id'] == question['id']
    assert request['evidence_readout_proposal']['last_trial'] == trial
    assert question['parent_id'] == 'task' and question['status'] == 'observed'

    repaired = execute(agent, lab, 'write', {'path': path, 'code': readout_source()})
    assert repaired['evidence_readout_trial']['evidence_status'] == 'supported'
    assert len(agent.revisions) == 2
    assert 'fixture analysis failed' in agent.revisions[0]['readout_error']
    assert agent.revisions[1]['status'] == 'answered'
    assert lab.paid == 5 and agent.costs['model_calls'] == 0


def test_unrelated_writes_and_inactive_question_never_execute_a_readout(tmp_path):
    agent, lab, question = paid_scene(tmp_path)
    proposal = prepare_evidence_readout(agent, question)
    agent._active_question_id = 'task'
    saved = execute(agent, lab, 'write', {'path': proposal['path'], 'code': readout_source()})
    assert 'evidence_readout_trial' not in saved and not agent.revisions
    path = open_readout(agent, question)
    costs = deepcopy(agent.costs)
    unrelated = execute(agent, lab, 'write', {'path': 'ordinary.py', 'code': 'value = 1\n'})
    assert 'evidence_readout_trial' not in unrelated and not agent.revisions
    assert trial_evidence_readout(agent, 'run', {'path': path}, lab) is None
    assert agent.costs == costs and lab.paid == 5


def test_ordinary_paid_question_offers_checked_readout_and_returns_actual_parent_result(tmp_path):
    agent, lab, question = paid_scene(tmp_path)
    proposal = question['evidence_readout_proposal']
    before = deepcopy((agent.observations, lab.actions, agent.confirmations))
    costs = deepcopy(agent.costs)

    request = agent._request(lab)

    assert not agent._inquiry_stack and 'llm_inquiry' not in request
    assert request['evidence_readout_proposal'] == proposal
    rendered = next(q for q in request['online_research']['questions']
                    if q['id'] == question['id'])
    assert rendered['evidence_readout_proposal'] == proposal
    assert proposal['record_indices'] == [1, 2]
    assert 'Choose your own' in request['task_interface']
    assert not agent.revisions and not agent.questions[0].get('child_answers')

    result = execute(agent, lab, 'write', {'path': proposal['path'],
                                          'code': readout_source(expected=1.)})

    trial = result['evidence_readout_trial']
    assert trial['check_results'][0]['value'] == 0.
    assert trial['evidence_status'] == trial['returned_answer']['status'] == 'refuted'
    assert trial['analysis_scope']['response_row_ranges'] == [[1, 4]]
    answer = agent.questions[0]['child_answers'][-1]
    assert answer['question_id'] == question['id']
    assert answer['check_results'][0]['witnesses'][0]['row'] == 1
    assert answer['evidence_status'] == 'refuted'
    assert agent._active_question_id == 'task' and not agent._inquiry_stack
    assert (agent.observations, lab.actions, agent.confirmations) == before
    assert agent.costs['model_calls'] == costs['model_calls'] == 0
    assert agent.costs['observation_cost'] == costs['observation_cost'] == lab.paid == 5
    assert agent.model is None


def test_zero_purchase_query_failure_preserves_existing_scope_and_creates_no_new_scope(tmp_path):
    agent, lab, question = paid_scene(tmp_path)
    before = deepcopy((question['evidence_readout_proposal'], agent.observations,
                       agent.costs, agent.revisions))

    failure = execute(agent, lab, 'query', {'actions': [action(7, fail=True)],
                                          'question': {'id': question['id']}})

    assert failure['error'] == 'apparatus unavailable' and failure['observations'] == []
    assert (question['evidence_readout_proposal'], agent.observations,
            agent.costs, agent.revisions) == before
    rejected = {'id': 'rejected', 'question': 'Inspect this proposed experiment',
                'parent_id': 'task', 'depends_on': []}
    failure = execute(agent, lab, 'query', {'actions': [action(8, fail=True)],
                                          'question': rejected})
    assert failure['observations'] == []
    active = next(q for q in agent.questions if q['id'] == 'rejected')
    assert not active['acquired_evidence'] and 'evidence_readout_proposal' not in active
    assert 'evidence_readout_proposal' not in agent._request(lab)
    assert lab.paid == 5 and not agent.confirmations and not agent.revisions


def test_partial_paid_query_offers_only_actual_acquired_records(tmp_path):
    agent, lab = prepared(tmp_path), PublicLab()
    execute(agent, lab, 'query', {'actions': [action(9)], 'question': PARENT})

    partial = execute(agent, lab, 'query', {
        'actions': [action(2, 2), action(4, fail=True)], 'question': QUESTION})

    assert partial['error'] == 'apparatus unavailable'
    proposal = partial['question']['evidence_readout_proposal']
    assert proposal['record_indices'] == [1] and proposal['response_row_count'] == 2
    assert partial['question']['acquired_evidence'] == [1, 2]
    assert agent._request(lab)['evidence_readout_proposal'] == proposal
    assert len(agent.observations) == 2 and lab.paid == 3
    assert not agent.revisions and not agent.confirmations


def test_ordinary_repeated_query_advances_scope_without_reexecuting_old_readout(tmp_path):
    agent, lab, question = paid_scene(tmp_path)
    path = question['evidence_readout_proposal']['path']
    first = execute(agent, lab, 'write', {'path': path, 'code': readout_source()})
    old_revision = deepcopy(agent.revisions[0])
    old_parent_answer = deepcopy(agent.questions[0]['child_answers'][0])

    result = execute(agent, lab, 'query', {'actions': [action(7, 7)],
                                          'question': {'id': question['id']}})

    proposal = result['question']['evidence_readout_proposal']
    assert proposal['record_indices'] == [3] and proposal['response_row_count'] == 2
    assert proposal['last_trial'] == first['evidence_readout_trial']
    assert agent.revisions == [old_revision]
    assert agent.questions[0]['child_answers'] == [old_parent_answer]
    assert agent._request(lab)['evidence_readout_proposal'] == proposal
    source = '''def readout(result, public_spec, memory):
    assert result['record_indices'] == [3]
    rows = result['observations'][0]['row_indices']
    assert rows == [5, 6]
    return {'status': 'supported', 'answer': 'The new pair agrees.',
            'evidence': rows, 'checks': [{'id': 'new_pair',
                'terms': [{'row': rows[0], 'coefficient': 1},
                          {'row': rows[1], 'coefficient': -1}],
                'relation': 'eq', 'expected': 0., 'atol': 1e-9}]}
'''
    second = execute(agent, lab, 'write', {'path': path, 'code': source})
    assert second['evidence_readout_trial']['evidence_status'] == 'supported'
    assert second['evidence_readout_trial']['analysis_scope']['response_row_ranges'] == [[5, 6]]
    assert agent.revisions[0] == old_revision
    assert agent.questions[0]['child_answers'][0] == old_parent_answer
    assert agent.questions[0]['child_answers'][-1]['evidence'] == [5, 6]
    assert lab.paid == 7 and agent.costs['model_calls'] == 0 and not agent.confirmations


def test_large_authored_answer_is_referenced_while_full_revision_remains_saved(tmp_path):
    agent, lab, question = paid_scene(tmp_path)
    path = open_readout(agent, question)
    authored = 'Local paid discussion. ' * 250
    result = execute(agent, lab, 'write', {'path': path,
                                          'code': readout_source(answer=authored)})
    trial = result['evidence_readout_trial']
    assert 'returned_answer' not in trial
    assert trial['returned_answer_source'] == {
        'file': 'research_state.json', 'json_path': ['revisions', 0, 'returned_answer']}
    assert agent.revisions[0]['returned_answer']['answer'] == authored
    assert agent.questions[0]['child_answers'][-1]['answer'] == authored
    assert trial['check_results'][0]['status'] == 'satisfied'
    assert lab.paid == 5 and not agent.confirmations and agent.model is None


@pytest.mark.parametrize('status', ['supported', 'refuted', 'inconclusive'])
def test_completed_scope_leaves_no_repeated_assignment_without_requiring_all_rows_checked(
        tmp_path, status):
    agent, lab, question = paid_scene(tmp_path)
    path = open_readout(agent, question)
    source = f'''def readout(result, public_spec, memory):
    return {{'status': {status!r}, 'answer': 'The selected paid scope was examined.',
            'evidence': [result['observations'][0]['row_indices'][0]]}}
'''
    execute(agent, lab, 'write', {'path': path, 'code': source})
    before = deepcopy((question['evidence_readout_proposal'], agent.revisions, agent.costs))

    request = agent._request(lab)

    assert evidence_readout_context(agent) is None
    assert 'evidence_readout_proposal' not in request
    assert EVIDENCE_READOUT_INSTRUCTION not in request['task_interface']
    assert agent.revisions[0]['record_indices'] == [1, 2]
    assert agent.revisions[0]['returned_answer']['evidence'] == [1]
    assert (question['evidence_readout_proposal'], agent.revisions, agent.costs) == before
    assert lab.paid == 5 and not agent.confirmations and agent.model is None


def test_explicit_paid_analysis_completes_assignment_but_partial_scope_stays_pending(tmp_path):
    agent, lab, question = paid_scene(tmp_path)
    open_readout(agent, question)
    original = deepcopy(question['evidence_readout_proposal'])
    source = '''def readout(result, public_spec, memory):
    return {'status': 'inconclusive', 'answer': 'No general law claimed.',
            'evidence': [i for r in result['observations'] for i in r['row_indices']]}
'''
    execute(agent, lab, 'write', {'path': 'explicit_readout.py', 'code': source})
    execute(agent, lab, 'run', {'operator': 'analyze_evidence',
        'path': 'explicit_readout.py', 'record_indices': [1],
        'research_question': {'id': question['id']}})
    assert evidence_readout_context(agent) == original
    assert agent._request(lab)['evidence_readout_proposal'] == original

    execute(agent, lab, 'run', {'operator': 'analyze_evidence',
        'path': 'explicit_readout.py', 'record_indices': [2],
        'research_question': {'id': question['id']}})
    request = agent._request(lab)
    assert evidence_readout_context(agent) is None
    assert EVIDENCE_READOUT_INSTRUCTION not in request['task_interface']
    assert question['evidence_readout_proposal'] == original
    assert 'last_trial' not in original and lab.paid == 5


def test_later_error_reopens_completed_readout_even_when_reanalyzing_an_older_revision(tmp_path):
    agent, lab, question = paid_scene(tmp_path)
    path = open_readout(agent, question)
    execute(agent, lab, 'write', {'path': path, 'code': readout_source()})
    execute(agent, lab, 'run', {'operator': 'analyze_evidence',
        'path': path, 'record_indices': [1, 2],
        'research_question': {'id': question['id']}})
    archived = deepcopy(question['evidence_readout_proposal'])
    assert evidence_readout_context(agent) is None
    execute(agent, lab, 'write', {'path': 'explicit_repair.py',
        'code': "def readout(result, public_spec, memory): raise ValueError('new revision failed')\n"})

    execute(agent, lab, 'run', {'operator': 'analyze_revision', 'revision_index': 0,
        'path': 'explicit_repair.py', 'research_question': {'id': question['id']}})

    assert agent.revisions[0]['status'] == 'answered'
    assert agent.revisions[0]['analysis_attempts'][-1]['error']
    assert evidence_readout_context(agent) == archived
    assert agent._request(lab)['evidence_readout_proposal'] == archived
    execute(agent, lab, 'write', {'path': 'explicit_repair.py', 'code': readout_source()})
    execute(agent, lab, 'run', {'operator': 'analyze_revision', 'revision_index': 0,
        'path': 'explicit_repair.py', 'research_question': {'id': question['id']}})
    assert evidence_readout_context(agent) is None
    assert question['evidence_readout_proposal'] == archived
    assert len(agent.revisions) == 2 and lab.paid == 5


def test_later_success_supersedes_an_earlier_record_whose_readout_never_started(tmp_path):
    agent, lab, question = paid_scene(tmp_path)
    path = open_readout(agent, question)
    with pytest.raises(FileNotFoundError):
        execute(agent, lab, 'run', {'operator': 'analyze_evidence',
            'path': 'not_written.py', 'record_indices': [1, 2],
            'research_question': {'id': question['id']}})
    assert agent.revisions[0]['status'] == 'observed'
    assert not agent.revisions[0].get('analysis_attempts')

    execute(agent, lab, 'write', {'path': path, 'code': readout_source()})

    assert evidence_readout_context(agent) is None
    assert 'evidence_readout_proposal' not in agent._request(lab)
    assert len(agent.revisions) == 2 and agent.revisions[1]['status'] == 'answered'
