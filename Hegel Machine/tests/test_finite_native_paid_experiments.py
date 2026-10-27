"""Paid reset trajectories are facts, not new data when an author revisits them."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.finite_native_author import (
    _attempted, paid_experiment_record_indices, select_actions,
)
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMAgent, InquiryLLMConfig
from test_finite_native_author import declaration, local_memory
from test_inquiry_revision import PublicLab, action
from test_native_condition_research import boundary_research_state


class ResetLab(PublicLab):
    def public_spec(self):
        return {**super().public_spec(), 'reset': 'Each trajectory starts from reset.'}


def scene(tmp_path):
    contract = declaration()
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm='code_react', source_task=False,
        knowledge={'files': {}, 'finite_native_author': contract},
        config=InquiryLLMConfig(observation_budget=100, compute_budget=1000,
            execution_seconds=30., max_model_calls=0))
    return agent, ResetLab(), contract


def query(agent, lab, question, actions, **kwargs):
    return agent._execute('query', {'research_question': deepcopy(question),
        'actions': deepcopy(actions), **kwargs}, lab)


def test_paid_pair_is_reused_after_other_rows_and_rebound_to_another_parent(tmp_path):
    agent, lab, contract = scene(tmp_path)
    question = contract['question']
    pair = [action(-.4, -.4), action(-.4, .5)]
    query(agent, lab, question, pair)
    query(agent, lab, question, [action(.8)])
    agent.config = replace(agent.config, observation_budget=5)
    before = deepcopy((agent.observations, agent.rows, agent.costs, lab.actions))
    changed_parent = {**question, 'id': 'different_parent_use', 'parent_id': 'task'}
    feedback = query(agent, lab, changed_parent, pair)
    assert feedback['record_indices'] == feedback['reused_record_indices'] == [0, 1]
    assert feedback['observations'] == [] and feedback['observation_cost'] == 0
    assert feedback['reused_observation_refs'] == [
        {'file': 'laboratory_observations.json', 'json_path': [0]},
        {'file': 'laboratory_observations.json', 'json_path': [1]}]
    assert (agent.observations, agent.rows, agent.costs, lab.actions) == before
    owner = next(q for q in agent.questions if q['id'] == changed_parent['id'])
    assert owner['evidence'] == [0, 1, 2, 3]
    assert owner['acquired_evidence'] == [] and owner['reused_record_indices'] == [0, 1]


def test_mixed_batch_only_buys_new_action_and_remaps_actual_continuation_indices(tmp_path, monkeypatch):
    agent, lab, contract = scene(tmp_path)
    question = contract['question']
    query(agent, lab, question, [action(-.4), action(.8)])
    agent.config = replace(agent.config, observation_budget=3)
    original = agent._execute
    received = []
    def execute(kind, arguments, laboratory):
        if kind == 'run' and arguments.get('operator') == 'fixture_paid_readout':
            received.append(deepcopy(arguments))
            return {'operator': 'fixture_paid_readout', 'new_compute_cost': 0}
        return original(kind, arguments, laboratory)
    monkeypatch.setattr(agent, '_execute', execute)
    feedback = query(agent, lab, question, [action(-.4), action(.3)], after_observation=[{
        'operator': 'fixture_paid_readout', 'records': 'acquired',
        'record_indices': [0, 2, 3],
        'matched_pairs': [{'reference_record_index': 2, 'record_index': 3, 'control': 'u'}]}])
    assert feedback['record_indices'] == [0, 2]
    assert feedback['reused_record_indices'] == [0]
    assert feedback['observation_cost'] == 1
    assert len(agent.rows) == len(agent.observations) == lab.paid == 3
    assert received == [{'operator': 'fixture_paid_readout', 'records': [0, 2],
        'record_indices': [0, 2],
        'matched_pairs': [{'reference_record_index': 0, 'record_index': 2, 'control': 'u'}]}]
    assert len(feedback['observations']) == 1


@pytest.mark.parametrize('purpose', [
    {'answer_type': 'distribution'}, {'kind': 'measurement'}, {'kind': 'instrument'},
    {'context_ref': {'apparatus': 'another_context'}},
])
def test_new_measurement_purpose_or_context_still_collects(tmp_path, purpose):
    agent, lab, contract = scene(tmp_path)
    question = contract['question']
    query(agent, lab, question, [action(-.4)])
    feedback = query(agent, lab, {**question, **purpose}, [action(-.4)])
    assert 'reused_record_indices' not in feedback
    assert feedback['observation_cost'] == 1 and len(agent.rows) == lab.paid == 2
    if 'context_ref' in purpose:
        assert agent.observations[-1]['experiment_context_ref'] == purpose['context_ref']
        assert paid_experiment_record_indices([action(-.4)], agent.observations[-1:],
            lab.public_spec(), question, contract, prediction_rows=lab.prediction_rows) == [None]


def test_paid_match_uses_actual_history_and_does_not_choose_conflicting_answers(tmp_path):
    agent, lab, contract = scene(tmp_path)
    question = contract['question']
    query(agent, lab, question, [action(-.4, .5)])
    source = deepcopy(agent.observations)
    source[0]['rows'][0]['history'] = [{'u': 1.}]
    assert paid_experiment_record_indices([action(-.4, .5)], source,
        lab.public_spec(), question, contract, prediction_rows=lab.prediction_rows) == [None]
    source = deepcopy(agent.observations) * 2
    source[1] = deepcopy(source[1])
    source[1]['rows'][1]['target'] += 1.
    assert paid_experiment_record_indices([action(-.4, .5)], source,
        lab.public_spec(), question, contract, prediction_rows=lab.prediction_rows) == [None]


def test_author_skips_answered_pair_even_after_paid_row_count_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    contract = declaration()
    question = contract['question']
    lab = ResetLab()
    pair = [action(-.4), action(.5)]
    records = [lab.execute(item) for item in pair]
    records.append(lab.execute(action(.9)))
    old = {'action': 'query', 'arguments': {'actions': pair}, 'research_question': question}
    assert _attempted(old, [{'proposal': old, 'feedback': {'record_indices': [0, 1]},
        'evidence_after': {'rows': 2}}], 3)
    proposed = {'action': 'query', 'arguments': {'actions': pair}}
    latest = {'status': 'needs_matched_evidence', 'candidates': [],
        'proposed_experiments': [{'purpose': 'matched_evidence', 'query_action': proposed}]}
    memory = local_memory(operator_results=[{'operator': 'native_condition_research',
        'research_question_id': question['id'], 'native_condition_research': latest}])
    result = select_actions(lab.public_spec(), records, memory)
    assert result['stop']['request']['reason'] == 'paid_experiments_already_answered'
    assert result['stop']['request']['paid_answers'][0]['record_indices'] == [0, 1]


def test_generic_query_without_finite_contract_keeps_original_acquisition(tmp_path):
    agent, lab, contract = scene(tmp_path)
    agent.knowledge.pop('finite_native_author')
    query(agent, lab, contract['question'], [action(-.4)])
    feedback = query(agent, lab, contract['question'], [action(-.4)])
    assert 'reused_record_indices' not in feedback and lab.paid == 2


def test_explicit_prospective_revision_still_buys_its_new_measurement(tmp_path):
    contract = declaration()
    agent = InquiryLLMAgent(None, tmp_path, arm='code_react', source_task=False,
        knowledge={'files': {}, 'finite_native_author': contract},
        config=InquiryLLMConfig(observation_budget=100, compute_budget=1000,
            execution_seconds=30., max_model_calls=0))
    lab = ResetLab()
    agent._execute('write', {'path': 'model.py',
        'code': "def predict(row): return row['features']['u']\n"}, lab)
    agent._execute('query', {'question': contract['question'], 'actions': [action(-.4)]}, lab)
    feedback = agent._execute('query', {'question': contract['question'], 'actions': [action(-.4)],
        'revision': {'id': 'r', 'affected_object': 'representation', 'models': ['model.py'],
            'parent_and_scope': {'parent': 'task', 'scope': 'public controls'}}}, lab)
    assert 'reused_record_indices' not in feedback and lab.paid == 2
    assert feedback['record_indices'] == [1] and feedback['revision']['evidence_cutoff']['rows'] == 1


def test_reused_boundary_continuation_names_the_original_paid_record(tmp_path, monkeypatch):
    agent, lab, contract = scene(tmp_path)
    question = contract['question']
    query(agent, lab, question, [action(.5)])
    original = agent._execute
    received = []
    def execute(kind, arguments, laboratory):
        if kind == 'run' and arguments.get('operator') == 'fixture_boundary_readout':
            received.append(deepcopy(arguments))
            return {'operator': 'fixture_boundary_readout', 'new_compute_cost': 0}
        return original(kind, arguments, laboratory)
    monkeypatch.setattr(agent, '_execute', execute)
    result = query(agent, lab, question, [action(.5)], after_observation=[{
        'operator': 'fixture_boundary_readout',
        'boundary_update': {'round_index': 0, 'proposal_index': 0}}])
    assert result['record_indices'] == [0] and lab.paid == 1
    assert received[0]['boundary_update'] == {
        'round_index': 0, 'proposal_index': 0, 'paid_record_indices': [0]}


def test_native_boundary_consumes_an_existing_paid_point_without_structure_refit(
        boundary_research_state, monkeypatch):
    from hegel_machine.cognition import native_condition_research as native
    from hegel_machine.cognition.models import SearchBudget
    from test_native_condition_research import (
        append_boundary_measurement, forbid_structure_refit, public_spec,
    )
    records, previous, proposal = deepcopy(boundary_research_state)
    records = append_boundary_measurement(records, proposal)
    # The selected fit used the old trajectories while another paid reader had
    # already acquired this reset point. It is not a new append after that fit.
    previous['observed_record_count'] = len(records)
    before_records, before_result = deepcopy(records), deepcopy(previous)
    forbid_structure_refit(monkeypatch)
    result = native.update_native_boundary(public_spec(), records, SearchBudget(10_000),
        previous=previous, proposal=proposal, observation_budget=1,
        paid_record_indices=[len(records) - 1])
    assert records == before_records and previous == before_result
    assert result['boundary_update']['status'] == 'boundary_refined'
    assert result['boundary_update']['observation_index'] == len(records) - 1
    assert result['fit_record_indices'] == list(range(len(records)))
    assert result['boundary_update']['after_interval'][1] < proposal['threshold_interval'][1]


def boundary_callback_query():
    return {'action': 'query', 'research_question': declaration()['question'], 'arguments': {
        'actions': [action(.5)], 'after_observation': [{
            'operator': 'native_condition_research', 'research_question': {'id': 'source_native'},
            'boundary_update': {'round_index': 2, 'proposal_index': 0}, 'max_compute': 10_000}]}}


def completed_boundary_query(proposal):
    callback = deepcopy(proposal['arguments']['after_observation'][0])
    callback['boundary_update']['paid_record_indices'] = [7]
    return {'proposal': deepcopy(proposal), 'feedback': {'record_indices': [7],
        'after_observation_results': [{'arguments': callback,
            'operator': 'native_condition_research', 'research_question_id': 'source_native',
            'question_meaning_version': 1, 'native_condition_research': {'boundary_update': {
                'status': 'boundary_refined', 'observation_index': 7}}}]},
        'evidence_after': {'rows': 83}}


def test_paid_query_without_its_new_callback_is_not_completed():
    proposal = boundary_callback_query()
    old = deepcopy(proposal)
    old['arguments'].pop('after_observation')
    result = {'proposal': old, 'feedback': {'record_indices': [7]},
        'evidence_after': {'rows': 83}}
    assert not _attempted(proposal, [result], 83, paid_record_indices=[7])
    assert _attempted(old, [result], 99, paid_record_indices=[7])


@pytest.mark.parametrize('difference', [
    'round', 'proposal', 'question', 'meaning', 'record', 'error', 'unfinished',
])
def test_boundary_completion_requires_actual_same_analysis_and_paid_evidence(difference):
    proposal = boundary_callback_query()
    result = completed_boundary_query(proposal)
    outcome = result['feedback']['after_observation_results'][0]
    if difference == 'round':
        outcome['arguments']['boundary_update']['round_index'] = 1
    elif difference == 'proposal':
        outcome['arguments']['boundary_update']['proposal_index'] = 1
    elif difference == 'question':
        outcome['research_question_id'] = 'another_question'
    elif difference == 'meaning':
        outcome['question_meaning_version'] = 2
    elif difference == 'record':
        outcome['arguments']['boundary_update']['paid_record_indices'] = [6]
    elif difference == 'error':
        outcome['error'] = 'interrupted boundary analysis'
    else:
        outcome['native_condition_research']['boundary_update']['status'] = 'compute_budget_exhausted'
    assert not _attempted(proposal, [result], 83, paid_record_indices=[7])


@pytest.mark.parametrize('status', ['boundary_refined', 'structure_mismatch'])
def test_completed_boundary_analysis_including_negative_result_is_reused(status):
    proposal = boundary_callback_query()
    result = completed_boundary_query(proposal)
    result['feedback']['after_observation_results'][0]['native_condition_research'][
        'boundary_update']['status'] = status
    assert _attempted(proposal, [result], 90, paid_record_indices=[7])


def test_author_does_not_repeat_completed_analysis_when_old_proposal_remains(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    proposal = boundary_callback_query()
    lab = ResetLab()
    records = [lab.execute(action(.5))]
    latest = {'status': 'candidates_require_evaluation', 'candidates': [],
        'remaining_boundary': {'threshold_interval': [0., .125]},
        'proposed_experiments': [{'purpose': 'conditional_boundary', 'query_action': proposal}]}
    memory = local_memory(operator_results=[{'operator': 'native_condition_research',
        'research_question_id': 'source_native', 'native_condition_research': latest}])
    assert select_actions(lab.public_spec(), records, memory) == proposal
    completed = completed_boundary_query(proposal)
    completed['feedback']['record_indices'] = [0]
    outcome = completed['feedback']['after_observation_results'][0]
    outcome['arguments']['boundary_update']['paid_record_indices'] = [0]
    outcome['native_condition_research']['boundary_update']['observation_index'] = 0
    memory['program_results'] = [completed]
    selected = select_actions(lab.public_spec(), records, memory)
    assert selected['stop']['request']['reason'] == 'paid_experiments_already_answered'
    assert selected['stop']['request']['paid_answers'][0]['record_indices'] == [0]


def test_updated_question_meaning_requires_its_own_boundary_analysis(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    proposal = boundary_callback_query()
    old_completion = completed_boundary_query(proposal)
    old_completion['feedback']['record_indices'] = [0]
    old_outcome = old_completion['feedback']['after_observation_results'][0]
    old_outcome['arguments']['boundary_update']['paid_record_indices'] = [0]
    old_outcome['native_condition_research']['boundary_update']['observation_index'] = 0
    proposal.pop('research_question')
    lab = ResetLab()
    records = [lab.execute(action(.5))]
    memory = local_memory(questions=[{'id': 'task'}, {'id': 'source_native', 'meaning_version': 2}],
        operator_results=[{'operator': 'native_condition_research',
            'research_question_id': 'source_native', 'question_meaning_version': 2,
            'native_condition_research': {'status': 'candidates_require_evaluation', 'candidates': [],
                'remaining_boundary': {'threshold_interval': [0., .125]},
                'proposed_experiments': [{'purpose': 'conditional_boundary', 'query_action': proposal}]}}])
    memory['finite_native_author']['question']['meaning_version'] = 2
    memory['program_results'] = [old_completion]
    selected = select_actions(lab.public_spec(), records, memory)
    assert selected['action'] == 'query' and selected['research_question']['meaning_version'] == 2
    current_completion = deepcopy(old_completion)
    current_completion['feedback']['after_observation_results'][0]['question_meaning_version'] = 2
    memory['program_results'].append(current_completion)
    completed = select_actions(lab.public_spec(), records, memory)
    assert completed['stop']['request']['reason'] == 'paid_experiments_already_answered'


def test_author_consumes_old_paid_boundary_then_selects_the_next_action(
        tmp_path, monkeypatch, boundary_research_state):
    from hegel_machine.cognition import native_condition_research as native
    from hegel_machine.cognition.models import SearchBudget
    from test_native_condition_inquiry import PaidConditionalLab
    from test_native_condition_research import (
        append_boundary_measurement, forbid_structure_refit, observed, public_spec,
    )

    records, previous, proposal = deepcopy(boundary_research_state)
    rounds = [previous]
    previous['round_index'] = 0
    forbid_structure_refit(monkeypatch)
    for _ in range(2):
        records = append_boundary_measurement(records, proposal)
        previous = native.update_native_boundary(public_spec(), records, SearchBudget(10_000),
            previous=previous, proposal=proposal, observation_budget=1)
        previous['round_index'] = len(rounds)
        rounds.append(previous)
        proposal, = previous['proposed_experiments']
    before_interval = proposal['threshold_interval']
    records = append_boundary_measurement(records, proposal)
    previous['observed_record_count'] = len(records)
    contract = declaration()
    target = contract['question']['theory_target']
    role = target['role_signature']['u']
    target['role_signature'] = {name: deepcopy(role) for name in public_spec()['controls']}
    contract['boundary_resolution'] = .01
    callback = {'operator': 'native_condition_research', 'research_question': {'id': 'source_native'},
        'boundary_update': {'round_index': 2, 'proposal_index': 0}, 'max_compute': 10_000}
    proposal['query_action']['arguments']['after_observation'] = [callback]
    for entry in rounds:
        entry['exported_models'] = []
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm='code_react', source_task=False,
        knowledge={'files': {}, 'finite_native_author': contract},
        config=InquiryLLMConfig(observation_budget=100, compute_budget=100_000,
            execution_seconds=30., max_model_calls=0))
    agent.observations = deepcopy(records)
    agent.rows = [row for item in records for row in observed(item['rows'])]
    agent.costs['observation_cost'] = len(agent.rows)
    agent.questions = [{'id': 'task'}, {**deepcopy(contract['question']),
        'evidence': list(previous['fit_evidence']), 'native_condition_research': rounds}]
    agent.operator_results = [{'operator': 'native_condition_research',
        'arguments': {'max_compute': 10_000,
            'matched_pairs': [{'reference_record_index': 0, 'record_index': 1}]},
        'research_question_id': 'source_native', 'native_condition_research': deepcopy(previous)}]
    lab = PaidConditionalLab()
    lab.paid = len(agent.rows)
    lab.actions = [deepcopy(item['action']) for item in records]
    monkeypatch.setattr(lab, 'execute', lambda *_: pytest.fail('A paid boundary must not be purchased again'))
    monkeypatch.chdir(tmp_path)
    memory = {**agent.knowledge, 'research_state': {'questions': agent.questions,
        'operator_results': agent.operator_results}}
    selected = select_actions(lab.public_spec(), records, memory)
    assert selected['action'] == 'query'
    assert selected['arguments']['after_observation'][0]['boundary_update'] == callback['boundary_update']
    before_records, before_paid = deepcopy(agent.observations), lab.paid
    feedback = agent._execute('query', {**selected['arguments'],
        'research_question': selected['research_question']}, lab)
    assert 'error' not in feedback and 'after_observation_error' not in feedback
    outcome, = feedback['after_observation_results']
    update = outcome['native_condition_research']['boundary_update']
    assert update['status'] == 'boundary_refined'
    assert update['before_interval'] == before_interval == [0., .125]
    assert update['after_interval'] == [.0625, .125]
    assert feedback['record_indices'] == feedback['reused_record_indices'] == [7]
    assert feedback['observations'] == [] and feedback['observation_cost'] == 0
    assert agent.observations == before_records and lab.paid == before_paid == 83
    attempted = {'proposal': selected, 'feedback': feedback, 'evidence_after': {'rows': 83}}
    assert _attempted(selected, [attempted], 83, paid_record_indices=[7])
    memory['program_results'] = [attempted]
    result = select_actions(lab.public_spec(), records, memory)
    assert result['action'] == 'run' and result['arguments']['operator'] == 'research_choice'
    assert result['arguments']['models'] == [item['model']
        for item in outcome['native_condition_research']['exported_models']]
    (tmp_path / 'paid_boundary_trace.json').write_text(json.dumps({
        'rows': before_paid, 'record_indices': feedback['record_indices'],
        'new_observations': len(feedback['observations']), 'observation_cost': feedback['observation_cost'],
        'before_interval': update['before_interval'], 'after_interval': update['after_interval'],
        'boundary_compute_cost': outcome['native_condition_research']['compute_cost'],
        'next_action': result}, indent=2))
