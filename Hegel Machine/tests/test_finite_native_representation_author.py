"""The common finite author exposes the same evidence-triggered repair to ABC."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.finite_native_author import (
    select_actions, _representation_action, _retained_representation_parent,
    export_finite_native_author,
)
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.question_routing import register_theories
from hegel_machine.cognition.question_continuation import parent_revision_obligation
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from scripts.run_anonymous_revision_inquiry import RegisteredOrganizationAgent
from test_finite_native_author import declaration
from test_native_representation_inquiry import (
    agent_with_records, measured_failure_basis, competing_representation_basis, learned_representation_basis,
)
from test_native_representation_routing import PublicOnlyLab


def scene(tmp_path, group, *, old=None):
    library = tmp_path / 'library.json'
    library.write_text('{"candidates":{}}')
    contract = declaration()
    contract['predictor_paths'] = ['old.py']
    contract['native_representation_plan'] = {'parent_theory_ids': ['old'],
        'validation_records': [4], 'compute_limit': 50000, 'max_candidates': 4}
    agent = RegisteredOrganizationAgent(None, tmp_path / group, arm=group, source_task=False,
        research_organization={'phase': 'target', 'initial_library_path': str(library),
                               'pr3_library_path': str(library)},
        knowledge={'finite_native_author': contract},
        config=replace(InquiryLLMConfig(), compute_budget=150000, max_model_calls=0))
    records = agent_with_records(tmp_path / 'records')
    agent.observations = records.observations
    agent.rows = [Observation(row['features'], row['target'], tuple(row['history']))
                  for row in records.rows]
    old = old or Program('const', value=0.)
    agent._save('old.py', export_temporal_program(old))
    register_theories(agent, [{'id': 'old', 'program': old.to_dict(),
        **{key: deepcopy(contract['question']['theory_target'][key])
           for key in ('role_signature', 'output_unit', 'context_ref')}}])
    question = agent._open_question(deepcopy(contract['question']))
    question['revision_candidate_exports'] = [{'model': 'old.py',
        'theory_ref': {'id': 'old', 'version': 1}, 'program_source_ref': {
            'file': 'research_state.json', 'json_path': ['theory_index', 'old', 'program'], 'version': 1}}]
    proposal = {'action': 'run', 'arguments': {'operator': 'research_choice',
        'models': ['old.py'], 'research_question': deepcopy(contract['question'])}}
    lab = PublicOnlyLab()
    choice = agent._execute(proposal['action'], proposal['arguments'], lab)
    agent._prepare_program_state()
    memory = {**agent.knowledge, 'research_state': agent._program_state(),
              'program_results': [{'proposal': proposal, 'feedback': choice,
                'evidence_after': {'rows': len(agent.rows)}}]}
    return agent, lab, contract, choice, memory


@pytest.mark.parametrize('group', ['archive', 'fixed_pr2', 'pr3'])
def test_all_registered_arms_execute_the_same_explicit_native_representation_action(tmp_path, monkeypatch, group):
    agent, lab, contract, choice, memory = scene(tmp_path, group)
    monkeypatch.chdir(agent.workspace)
    proposal = select_actions(lab.public_spec(), agent.observations, memory)
    assert proposal.get('action') == 'run', proposal
    args = proposal['arguments']
    assert args['operator'] == 'research_route'
    assert args['research_question']['id'] == 'representation:old.py'
    assert args['native_representation_plan']['compute_limit'] == 50000
    assert args['native_representation_plan']['validation_records'] == [4]
    assert args['native_representation_plan']['parent_theory_ids'] == [
        'old', 'retained_representation_parent:old.py']
    before = dict(agent.costs)
    result = agent._execute('run', args, lab)
    assert result['status'] == 'native_representation_candidates_ready'
    assert result['selected_model'] in agent.candidate_paths
    assert result['native_representation']['selected']['validation_mse'] < 1e-10
    assert agent._evaluated_summary()['source_model'] == 'old.py'
    assert agent.costs['model_calls'] == before['model_calls'] == 0
    assert agent.costs['observation_cost'] == before['observation_cost']
    assert agent.arm == 'code_react'  # The action works without Hegel-only hooks.


def test_proven_collision_requests_missing_plan_without_spending_legacy_search_allowance(tmp_path):
    agent, _, contract, choice, memory = scene(tmp_path, 'archive')
    contract.pop('native_representation_plan')
    result = _representation_action(contract, contract['question'], choice, agent._program_state(),
                                    memory, len(agent.rows))
    assert result['stop']['request']['reason'] == 'native_representation_plan_required'
    assert contract['max_compute'] == 300


def test_opaque_actual_parent_remains_an_author_question_instead_of_borrowing_old_parent(tmp_path):
    agent, _, contract, choice, memory = scene(tmp_path, 'archive')
    state = agent._program_state()
    state['questions'] = [{'id': 'task'}]
    result = _representation_action(contract, contract['question'], choice, state, memory, len(agent.rows))
    assert result['stop']['request']['reason'] == 'actual_retained_representation_program_required'
    assert result['stop']['request']['source_model'] == 'old.py'


def test_standalone_common_author_executes_the_new_route_from_its_comparison_feedback(tmp_path):
    agent, lab, _, _, _ = scene(tmp_path, 'archive')
    agent._save('shared/author.py', export_finite_native_author())
    result = agent._execute('run', {'operator': 'procedure', 'path': 'shared/author.py',
                                  'max_steps': 2, 'confirm': False}, lab)
    operators = [entry['proposal']['arguments']['operator'] for entry in result['steps']]
    assert operators == ['research_choice', 'research_route'], result
    assert result['steps'][-1]['feedback']['status'] == 'native_representation_candidates_ready'
    assert agent.costs['model_calls'] == agent.costs['observation_cost'] == 0


def test_new_actual_export_retains_its_program_address_and_gets_its_own_child_identity():
    contract = declaration()
    contract['native_representation_plan'] = {'parent_theory_ids': ['old'],
        'validation_records': [1], 'compute_limit': 1000, 'max_candidates': 4}
    new = Program('const', value=.5).to_dict()
    source = {'file': 'research_state.json', 'json_path': [
        'questions', 1, 'native_representation_research', 0, 'candidates', 0]}
    first_result = {'exported_models': [{'model': 'new.py', 'candidate_id': 'native_representation:0',
        'source_ref': source}], 'native_representation': {'candidates': [
            {'id': 'native_representation:0', 'program': new}]}}
    state = {'questions': [], 'method_results': [{'feedback': first_result}],
        'theory_index': {'old': {'program': Program('const', value=195.).to_dict()}}}
    choice = {'evaluated_model': {'source_model': 'new.py'}, 'evaluations': [
        {'path': 'new.py', 'returncode': 0, 'representation_diagnostic': {
            'conflicting_groups': 1, 'empirical_mse_floor': .1}}]}
    action = _representation_action(contract, contract['question'], choice, state, {}, 10)
    assert action['arguments']['research_question']['id'] == 'representation:new.py'
    actual = action['arguments']['theories'][0]
    assert actual['program'] == new
    assert actual['id'] == 'retained_representation_parent:new.py'
    assert actual['program_source_ref'] == {'file': source['file'],
                                          'json_path': [*source['json_path'], 'program']}
    assert actual['support']['independent_confirmation'] is False
    assert action['arguments']['native_representation_plan']['parent_theory_ids'] == [
        'old', 'retained_representation_parent:new.py']


def test_unchanged_completed_native_attempt_is_not_originally_rerun():
    contract = declaration()
    contract['native_representation_plan'] = {'parent_theory_ids': ['old'],
        'validation_records': [1], 'compute_limit': 1000, 'max_candidates': 4}
    state = {'questions': [{'id': 'source', 'native_condition_research': [{'candidates': [
        {'model': 'old.py', 'program': Program('const', value=0.).to_dict()}]}]}]}
    choice = {'evaluated_model': {'source_model': 'old.py'}, 'evaluations': [
        {'path': 'old.py', 'returncode': 0, 'representation_diagnostic': {
            'conflicting_groups': 1, 'empirical_mse_floor': .1}}]}
    first = _representation_action(contract, contract['question'], choice, state, {}, 10)
    repeated = _representation_action(contract, contract['question'], choice, state,
        {'program_results': [{'proposal': first, 'evidence_after': {'rows': 10}}]}, 10)
    assert repeated['stop']['request']['reason'] == (
        'representation_research_return_requires_supported_revision_or_new_evidence')


def test_canonical_native_export_resolves_only_its_real_program_address(tmp_path):
    program = Program('feature', name='u').to_dict()
    canonical = tmp_path / 'native.json'
    canonical.write_text(json.dumps({'candidates': [{'program': program}]}))
    result = {'native_representation_ref': {'file': str(canonical), 'json_path': []},
        'native_representation': {'candidates': [{'id': 'native_representation:0',
            'program_ref': {'file': str(canonical), 'json_path': ['candidates', 0, 'program']}}]},
        'exported_models': [{'model': 'new.py', 'candidate_id': 'native_representation:0',
            'source_ref': {'file': str(canonical), 'json_path': ['candidates', 0]}}]}
    parent = _retained_representation_parent({'questions': []}, [{'feedback': result}],
                                            'new.py', declaration()['question']['theory_target'])
    assert parent['program'] == program
    assert parent['program_source_ref'] == {'file': str(canonical),
                                          'json_path': ['candidates', 0, 'program']}


def test_retained_old_meaning_parent_keeps_its_own_native_program_address():
    program = Program('feature', name='u').to_dict()
    state = {'questions': [{'id': 'source', 'meaning_version': 2, 'meaning_history': [
        {'id': 'source', 'meaning_version': 1, 'native_condition_research': [
            {'candidates': [{'model': 'old.py', 'program': program}]}]}]}]}
    parent = _retained_representation_parent(state, [], 'old.py', declaration()['question']['theory_target'])
    assert parent['program'] == program
    assert parent['program_source_ref']['json_path'] == ['questions', 0, 'meaning_history', 0,
        'native_condition_research', 0, 'candidates', 0, 'program']


@pytest.mark.parametrize('group', ['archive', 'fixed_pr2', 'pr3'])
def test_actual_new_export_binds_new_parent_and_creates_a_separate_revision_meaning(tmp_path, group):
    agent, lab, contract, _, _ = scene(tmp_path, group)
    question = next(q for q in agent.questions if q['id'] == contract['question']['id'])
    question['theory_target']['revision_fitting'] = {}
    question['use_sites'] = [{'kind': 'supply_component', 'model': 'parent.py', 'slot': 'response'}]
    agent._bind_revision_target(question)
    first = parent_revision_obligation(agent, question, {'parent_revision_completed': False})
    assert first['theory_target']['parent_theory_id'] == 'old'
    first_ref = {'question_id': first['id'], 'meaning_version': first['meaning_version']}
    first['frozen_recall_execution'] = {'cursor': {'next_candidate_id': 'original_parent_cursor'},
        'result': {'search_completed': False}}
    # This fixture supplies an actual exported draft; the full real comparison
    # must adopt it before parent binding can use its exact Program address.
    program = Program('feature', name='u')
    candidate_path = 'native_representations/actual_new.py'
    agent._save(candidate_path, export_temporal_program(program))
    canonical = agent.workspace / 'native.json'
    canonical.write_text(json.dumps({'candidates': [{'id': 'new', 'program': program.to_dict()}]}))
    child = agent._open_question({'id': 'representation:old.py', 'question': 'Retain the actual representation draft.',
                                 'kind': 'representation',
                                 'parent_id': 'task', 'depends_on': []})
    child['routing_execution'] = {'result': {'native_representation': {'candidates': [{'id': 'new'}]},
        'exported_models': [{'model': candidate_path, 'candidate_id': 'new', 'source_ref': {
            'file': str(canonical), 'json_path': ['candidates', 0]}}]}}
    compared = agent._execute('run', {'operator': 'research_choice', 'models': ['old.py', candidate_path],
        'research_question': {'id': question['id']}}, lab)
    assert compared['evaluated_model']['source_model'] == candidate_path
    agent._bind_revision_target(question)
    identifier = question['theory_target']['parent_theory_id']
    assert identifier == f'retained_representation_parent:{candidate_path}'
    assert agent.theory_index[identifier]['program'] == program.to_dict()
    assert agent.theory_index[identifier]['support']['independent_confirmation'] is False
    second = parent_revision_obligation(agent, question, {'parent_revision_completed': False})
    assert second['id'] == first_ref['question_id']
    assert second['meaning_version'] == first_ref['meaning_version'] + 1
    assert second['theory_target']['parent_theory_id'] == identifier
    assert 'frozen_recall_execution' not in second
    assert second['meaning_history'][0]['frozen_recall_execution']['cursor'] == {
        'next_candidate_id': 'original_parent_cursor'}


@pytest.mark.parametrize('warrant', ['failures', 'alternative', 'semantics', 'learned'])
def test_formal_author_and_actual_native_consume_each_independent_warrant(tmp_path, monkeypatch, warrant):
    agent, lab, contract, choice, memory = scene(tmp_path, 'archive', old=Program('feature', name='u'))
    assert choice['evaluations'][0]['representation_diagnostic']['conflicting_groups'] == 0
    current = next(item for item in agent.questions if item['id'] == contract['question']['id'])
    current['use_sites'] = [{'kind': 'supply_component', 'model': 'old.py', 'slot': 'representation'}]
    if warrant == 'failures':
        basis = measured_failure_basis(agent, Program('window_sum', lag=3,
            children=(Program('feature', name='u'),)))
    elif warrant == 'alternative':
        basis = competing_representation_basis(agent, tmp_path)
    elif warrant == 'learned':
        basis = learned_representation_basis(tmp_path)
    else:
        agent.questions.append({'id': 'history_semantics', 'kind': 'representation', 'status': 'open',
            'use_sites': deepcopy(current['use_sites']),
            'answer_resolution_ref': {'evaluator': 'representation_diagnostic_and_paid_comparison',
                                      'parent_model': 'old.py'}})
        basis = {'semantic_question_ref': {'kind': 'question', 'id': 'history_semantics', 'version': 1}}
    current['representation_research_basis'] = basis
    agent._prepare_program_state()
    memory['research_state'] = agent._program_state()
    monkeypatch.chdir(agent.workspace)
    proposal = select_actions(lab.public_spec(), agent.observations, memory)
    assert proposal.get('action') == 'run', proposal
    assert proposal['arguments']['operator'] == 'research_route'
    child = proposal['arguments']['research_question']
    assert child['representation_research_basis'] == basis
    assert child['use_sites'] == current['use_sites']
    result = agent._execute('run', proposal['arguments'], lab)
    assert result['status'] == 'native_representation_candidates_ready'
    assert result['native_representation']['alternative_representation_basis']
    assert result['native_representation']['origin'] == child['origin']
    assert result['native_representation']['origin'] != 'computed_representation_conflict'
    assert result['native_representation']['candidates']
    assert agent.costs['model_calls'] == agent.costs['observation_cost'] == 0


def test_missing_reference_requests_author_work_instead_of_counting_a_field_as_evidence(tmp_path):
    agent, _, contract, choice, memory = scene(tmp_path, 'archive', old=Program('feature', name='u'))
    current = next(item for item in agent.questions if item['id'] == contract['question']['id'])
    current['representation_research_basis'] = {'alternative_ref': {
        'file': str(tmp_path / 'missing.json'), 'json_path': []}}
    result = _representation_action(contract, contract['question'], choice,
        agent._program_state(), memory, len(agent.rows))
    assert result['stop']['request']['reason'] == 'representation_research_warrant_unresolved'



def test_standalone_author_consumes_saved_competing_program_without_collision(tmp_path):
    agent, lab, contract, _, _ = scene(tmp_path, 'archive', old=Program('feature', name='u'))
    current = next(item for item in agent.questions if item['id'] == contract['question']['id'])
    current['representation_research_basis'] = competing_representation_basis(agent, tmp_path)
    agent._save('shared/author.py', export_finite_native_author())
    result = agent._execute('run', {'operator': 'procedure', 'path': 'shared/author.py',
                                  'max_steps': 2, 'confirm': False}, lab)
    operators = [entry['proposal']['arguments']['operator'] for entry in result['steps']]
    assert operators == ['research_choice', 'research_route'], result
    native = result['steps'][-1]['feedback']['native_representation']
    assert native['alternative_representation_basis'] == 'retained_alternative_has_different_task_predictions'
    assert native['origin'] == result['steps'][-1]['proposal']['arguments']['research_question']['origin']
    assert native['origin'] != 'computed_representation_conflict'
    assert native['candidates'] and agent.costs['model_calls'] == agent.costs['observation_cost'] == 0
