"""Finite control-flow checks use fixture rows, never a scientific simulator."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.finite_native_author import export_finite_native_author, select_actions
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from test_inquiry_revision import PublicLab, action
from test_question_routing import CONTEXT, ROLES, SQUARE, future_question, probe, route, theory, scene


POLICY = 'shared/finite_native_author.py'


class ChoiceLab(PublicLab):
    def public_spec(self):
        return {**super().public_spec(), 'max_probe_length': 24,
                'feature_bounds': {'u': [-10., 10.]}}


def declaration():
    return {'question': {'id': 'source_native', 'question': 'Predict the public scalar response from paid controls.',
        'parent_id': 'task', 'depends_on': [], 'context_ref': deepcopy(CONTEXT),
        'theory_target': {'role_signature': deepcopy(ROLES), 'output_unit': 'V',
                          'context_ref': deepcopy(CONTEXT)}},
        'predictor_paths': [], 'predictor_catalog_ref': {'file': 'retained_source_catalog.json',
            'json_path': [], 'scope': 'all sources including failed_source'},
        'max_compute': 300, 'max_candidates': 16, 'boundary_resolution': .1}


def local_memory(**state):
    return {'finite_native_author': declaration(), 'research_state': {
        'questions': [{'id': 'task'}], 'operator_results': [], **state}}


def observations():
    return [{'rows': [{'features': {'u': 2.}, 'target': 2., 'history': []}], 'row_indices': [0]}]


def test_missing_typed_contract_or_complete_catalog_is_explicit_author_work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    memory = local_memory()
    memory['finite_native_author']['question']['theory_target'].pop('output_unit')
    result = select_actions({'controls': ['u']}, observations(), memory)
    assert result['stop']['request']['reason'] == 'public_typed_research_contract_required'
    memory = local_memory()
    memory['finite_native_author'].pop('predictor_paths')
    assert select_actions({'controls': ['u']}, observations(), memory)['stop']['request'][
        'reason'] == 'complete_retained_predictor_catalog_required'
    memory = local_memory()
    memory['finite_native_author']['predictor_paths'] = ['missing_old.py']
    stopped = select_actions({'controls': ['u']}, observations(), memory)['stop']['request']
    assert stopped['reason'] == 'retained_predictor_source_missing'
    assert stopped['required'] == ['missing_old.py']
    assert stopped['catalog_ref'] == memory['finite_native_author']['predictor_catalog_ref']


def test_native_search_and_old_experiment_continuation_use_only_existing_actions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    memory = local_memory()
    proposed = select_actions({'controls': ['u']}, observations(), memory)
    assert proposed['arguments']['operator'] == 'native_condition_research'
    assert proposed['arguments']['research_question'] == memory['finite_native_author']['question']
    assert proposed['arguments']['max_compute'] == 300
    query = {'action': 'query', 'arguments': {'actions': [action(1., 2.)],
        'after_observation': [{'operator': 'native_condition_research',
                              'research_question': {'id': 'source_native'}}]}}
    latest = {'status': 'needs_matched_evidence', 'candidates': [],
              'proposed_experiments': [{'purpose': 'matched_evidence', 'query_action': deepcopy(query)}]}
    memory['research_state']['operator_results'] = [{'operator': 'native_condition_research',
        'research_question_id': 'source_native', 'native_condition_research': latest}]
    assert select_actions({'controls': ['u']}, observations(), memory) == {
        **query, 'research_question': memory['finite_native_author']['question']}
    latest.update(status='structural_mismatch', proposed_experiments=[])
    result = select_actions({'controls': ['u']}, observations(), memory)
    assert result['stop']['request']['reason'] == 'executable_revision_or_scope_required'
    assert result['stop']['request']['native_status'] == 'structural_mismatch'
    assert latest['status'] == 'structural_mismatch'


def test_common_procedure_compares_all_old_native_and_validated_paths_and_preserves_old_winner(tmp_path):
    contract = declaration()
    contract['predictor_paths'] = ['archive/good.py', 'archive/failed.py']
    files = {POLICY: export_finite_native_author(),
        'archive/good.py': "def predict(row): return row['features']['u']\n",
        'archive/failed.py': "def predict(row): return 0.\n",
        'native.py': "def predict(row): return -1.\n",
        'validated.py': "def predict(row): return 1.\n"}
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm='code_react', source_task=False,
        knowledge={'files': files, 'finite_native_author': contract},
        config=InquiryLLMConfig(observation_budget=20, compute_budget=3000, confirmation_rows=2,
                               execution_seconds=30., max_model_calls=0))
    lab = ChoiceLab()
    agent._execute('query', {'actions': [action(2., 3.)]}, lab)
    agent.operator_results.extend([
        {'operator': 'refit', 'status': 'refit_supported_on_validation', 'path': 'validated.py'},
        {'operator': 'native_condition_research', 'research_question_id': 'source_native',
         'native_condition_research': {'status': 'candidates_require_evaluation',
             'candidates': [{'model': 'native.py', 'fit_mse': 0.}]}}])
    feedback = agent._execute('run', {'operator': 'procedure', 'path': POLICY,
                                    'max_steps': 3, 'confirm': False}, lab)
    assert any(item['operator'] == 'research_choice' for item in agent.operator_results), feedback
    choice = next(item for item in agent.operator_results if item['operator'] == 'research_choice')
    assert 'evaluations' in choice, choice
    assert [entry['path'] for entry in choice['evaluations']] == [
        'archive/good.py', 'archive/failed.py', 'native.py', 'validated.py']
    assert choice['evaluated_model']['source_model'] == 'archive/good.py'
    assert choice['next_action']['action'] == 'finish'
    assert feedback['status'] == 'confirmation_not_authorized'
    assert agent.model is None and not agent.confirmations
    assert lab.paid == agent.costs['observation_cost'] == 2 and len(lab.actions) == 1
    assert agent.costs['compute_cost'] > 0 and agent.costs['model_calls'] == 0
    assert agent.knowledge['finite_native_author']['predictor_paths'] == contract['predictor_paths']


def test_common_procedure_actually_calls_the_bounded_native_operator_on_fixture_prefix(tmp_path):
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm='code_react', source_task=False,
        knowledge={'files': {POLICY: export_finite_native_author()},
                   'finite_native_author': declaration()},
        config=InquiryLLMConfig(observation_budget=20, compute_budget=3000, confirmation_rows=2,
                               execution_seconds=30., max_model_calls=0))
    lab = ChoiceLab()
    agent._execute('query', {'actions': [action(2., 3.)]}, lab)
    before = agent.costs['compute_cost']
    result = agent._execute('run', {'operator': 'procedure', 'path': POLICY, 'max_steps': 1}, lab)
    executed, = agent.operator_results
    assert executed['operator'] == 'native_condition_research' and 'error' not in executed
    assert executed['arguments']['max_compute'] == 300
    assert executed['research_question_id'] == 'source_native'
    assert executed['native_condition_research']['status'] == 'needs_matched_evidence'
    assert 0 < agent.costs['compute_cost'] - before <= 300
    assert result['steps'][0]['proposal']['arguments']['operator'] == 'native_condition_research'
    assert lab.paid == 2 and len(lab.actions) == 1 and agent.costs['model_calls'] == 0


def test_actual_common_procedure_consumes_the_prepared_PR2_parent_query(tmp_path):
    agent, lab = scene(tmp_path)
    current = future_question(agent)
    selected = route(agent, lab, question=current,
        theories=[theory('square', SQUARE), theory('zero', Program('const', value=0.))],
        experiments=[probe(agent)])
    assert selected['status'] == 'selected'
    contract = declaration()
    contract['question'] = current
    agent.knowledge['finite_native_author'] = contract
    agent._save(POLICY, export_finite_native_author())
    paid_before = lab.paid
    feedback = agent._execute('run', {'operator': 'procedure', 'path': POLICY,
                                    'max_steps': 1, 'confirm': False}, lab)
    proposal = feedback['steps'][0]['proposal']
    assert proposal['action'] == 'query' and proposal['arguments'] == {'pending_question': 'task'}
    parent = next(item for item in agent.questions if item['id'] == 'task')
    assert parent['executed_pending_queries'][-1]['source']['question_id'] == current['id']
    assert 'pending_query' not in parent
    assert lab.paid == paid_before + 1
    assert agent.costs['model_calls'] == 0 and not agent.confirmations


def test_PR2_purchase_compares_the_retained_bank_before_pricing_current_data_research(tmp_path):
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm='code_react', source_task=False,
        knowledge={'files': {POLICY: export_finite_native_author()}},
        config=InquiryLLMConfig(observation_budget=20, compute_budget=3000, confirmation_rows=2,
                               execution_seconds=30., max_model_calls=0))
    lab = ChoiceLab()
    agent._open_question({'id': 'task', 'question': 'Predict the public fixture response.'})
    agent._execute('query', {'actions': [action(1.)]}, lab)
    current = future_question(agent)
    current['candidate_theory_ids'] = ['square', 'zero']
    contract = declaration()
    contract['question'] = current
    agent.knowledge['finite_native_author'] = contract
    initial = agent._execute('run', {'operator': 'native_condition_research',
        'research_question': current, 'programs': [Program('const', value=5.).to_dict(),
            Program('const', value=0.).to_dict()],
        'max_compute': contract['max_compute'], 'max_candidates': contract['max_candidates']}, lab)
    owner = next(item for item in agent.questions if item['id'] == current['id'])
    original = deepcopy(owner['native_condition_research'][0])
    assert initial['native_condition_research']['fit_evidence'] == [0]
    assert original['candidates'] and all(item['fit_mse'] > 0 for item in original['candidates'])
    # A current comparison now precedes acquisition. Keep this test on the
    # unresolved branch; an already correct native candidate would finish.
    compared = agent._execute('run', {'operator': 'procedure', 'path': POLICY,
        'max_steps': 1, 'confirm': False}, lab)
    assert compared['steps'][0]['proposal']['arguments']['operator'] == 'research_choice'
    assert not agent.operator_results[-1].get('next_action')
    selected = route(agent, lab, question=current,
        theories=[theory('square', SQUARE), theory('zero', Program('const', value=0.))],
        experiments=[probe(agent)])
    assert selected['status'] == 'selected'
    parent = next(item for item in agent.questions if item['id'] == 'task')
    frozen_pending = deepcopy(parent['pending_query'])
    assert 'after_observation' not in frozen_pending
    paid_before, compute_before = lab.paid, agent.costs['compute_cost']
    result = agent._execute('run', {'operator': 'procedure', 'path': POLICY,
                                  'max_steps': 2, 'confirm': False}, lab)
    query, compared = result['steps']
    assert query['proposal']['arguments'] == {'pending_question': 'task'}
    executed_query = agent.method_results[-2]
    assert executed_query['evidence_before']['rows'] == 1 and executed_query['evidence_after']['rows'] == 2
    assert compared['proposal']['arguments']['operator'] == 'research_choice'
    assert compared['feedback']['paid_rows'] == 2
    assert compared['proposal']['arguments']['models'] == [
        candidate['model'] for candidate in original['candidates']]
    assert owner['native_condition_research'][0] == original
    assert len(owner['native_condition_research']) == 1
    assert lab.paid == paid_before + 1
    assert agent.costs['compute_cost'] >= compute_before
    assert lab.paid == paid_before + 1 and len(lab.actions) == 2
    assert parent['executed_pending_queries'][-1]['query'] == frozen_pending
    assert not agent.confirmations and agent.costs['model_calls'] == 0


@pytest.mark.parametrize('record_indices', [None, [0]])
def test_native_after_observation_already_saw_new_records_without_repeated_search(tmp_path, record_indices):
    contract = declaration()
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm='code_react', source_task=False,
        knowledge={'files': {POLICY: export_finite_native_author()}, 'finite_native_author': contract},
        config=InquiryLLMConfig(observation_budget=20, compute_budget=3000, confirmation_rows=2,
                               execution_seconds=30., max_model_calls=0))
    lab = ChoiceLab()
    agent._execute('query', {'actions': [action(1.)]}, lab)
    native = {'operator': 'native_condition_research', 'research_question': contract['question'],
        'max_compute': contract['max_compute'], 'max_candidates': contract['max_candidates']}
    agent._execute('run', {**native, 'programs': [Program('feature', name='u').to_dict()]}, lab)
    continuation = deepcopy(native)
    if record_indices is not None:
        continuation['record_indices'] = record_indices
    feedback = agent._execute('query', {'actions': [action(2.)],
                                      'after_observation': [continuation]}, lab)
    assert 'after_observation_error' not in feedback
    owner = next(item for item in agent.questions if item['id'] == contract['question']['id'])
    updated = deepcopy(owner['native_condition_research'])
    assert updated[-1]['observed_record_count'] == 2
    assert updated[-1]['fit_record_indices'] == ([0, 1] if record_indices is None else [0])
    compute_before = agent.costs['compute_cost']
    result = agent._execute('run', {'operator': 'procedure', 'path': POLICY,
                                  'max_steps': 1, 'confirm': False}, lab)
    assert result['steps'][0]['proposal']['arguments']['operator'] == 'research_choice'
    assert owner['native_condition_research'] == updated
    assert agent.costs['compute_cost'] > compute_before
    assert lab.paid == 2 and not agent.confirmations and agent.costs['model_calls'] == 0


def test_structured_missing_input_obligation_survives_the_actual_procedure(tmp_path):
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm='code_react', source_task=False,
        knowledge={'files': {POLICY: export_finite_native_author()}},
        config=InquiryLLMConfig(observation_budget=10, compute_budget=1000,
                               confirmation_rows=0, execution_seconds=30., max_model_calls=0))
    lab = PublicLab()
    result = agent._execute('run', {'operator': 'procedure', 'path': POLICY, 'max_steps': 1}, lab)
    request = result['steps'][0]['feedback']['reason']
    assert result['status'] == 'method_stopped' and request['status'] == 'author_work_request'
    assert request['request']['reason'] == 'public_typed_research_contract_required'
    assert lab.paid == agent.costs['model_calls'] == 0


def test_author_source_update_reuses_completed_choice_then_refit_and_published_same_bank(tmp_path, monkeypatch):
    contract = declaration()
    contract['predictor_paths'] = ['archive/_evaluated_model/old.py']
    previous_source = export_finite_native_author() + '\n# previous frozen author version\n'
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm='code_react', source_task=False,
        knowledge={'files': {POLICY: previous_source,
            contract['predictor_paths'][0]: 'def predict(row): return 0.\n'},
            'finite_native_author': contract},
        config=InquiryLLMConfig(observation_budget=20, compute_budget=10000, confirmation_rows=2,
                               execution_seconds=30., max_model_calls=0))
    lab = ChoiceLab()
    agent._execute('query', {'actions': [action(2.), action(3.)]}, lab)
    # Isolate same-bank refitting from representation-conflict research: these
    # predictors retain distinct u values but have deliberately wrong gains.
    programs = [Program('mul', children=(Program('const', value=gain),
        Program('feature', name='u'))).to_dict() for gain in (1.25, 1.5)]
    basis = [Program('feature', name='u').to_dict()]
    agent._execute('run', {'operator': 'native_condition_research',
        'research_question': contract['question'], 'programs': programs, 'basis': basis,
        'max_compute': contract['max_compute'], 'max_candidates': contract['max_candidates']}, lab)
    owner = next(item for item in agent.questions if item['id'] == 'source_native')
    original = deepcopy(owner['native_condition_research'][0])
    first = agent._execute('run', {'operator': 'procedure', 'path': POLICY,
        'max_steps': 1, 'confirm': False}, lab)
    choice = first['steps'][0]['feedback']
    assert choice['operator'] == 'research_choice'
    assert choice['next_action']['arguments']['operator'] == 'refit'
    assert choice['paid_rows'] == 2 and len(choice['evaluations']) == 3
    first_result = deepcopy(agent.method_results[-1])
    assert agent.program_contexts[POLICY]['source'] == previous_source
    agent._save(POLICY, export_finite_native_author())
    agent.knowledge['finite_native_author']['max_compute'] = 600

    def no_new_purchase(*args, **kwargs):
        pytest.fail('the same-paid continuation must not acquire another fixture record')

    def no_new_python_score(*args, **kwargs):
        pytest.fail('unchanged paid execution and literal aliases must not predict again')

    monkeypatch.setattr(lab, 'execute', no_new_purchase)
    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model',
                        no_new_python_score)
    before = deepcopy(agent.costs)
    result = agent._execute('run', {'operator': 'procedure', 'path': POLICY,
        'max_steps': 3, 'confirm': False}, lab)
    steps = result['steps']
    assert [entry['proposal']['arguments']['operator'] for entry in steps] == [
        'refit', 'native_condition_research', 'research_choice'], result
    assert steps[0]['feedback']['status'] == 'refit_unavailable'
    resumed = steps[1]['proposal']['arguments']
    assert resumed['max_compute'] == contract['max_compute']
    assert resumed['max_candidates'] == contract['max_candidates']
    assert resumed['programs'] == programs and resumed['basis'] == basis
    assert owner['native_condition_research'][0] == original
    assert len(owner['native_condition_research']) == 2
    expected = contract['predictor_paths'] + [candidate['model']
        for round_ in owner['native_condition_research'] for candidate in round_['candidates']]
    assert steps[-1]['proposal']['arguments']['models'] == expected
    assert [item['path'] for item in steps[-1]['feedback']['evaluations']] == expected
    compared = steps[-1]['feedback']
    assert compared['reused_score_count'] == 5
    assert compared['new_scored_count'] == 0
    for retained, repeated in zip(choice['evaluations'], compared['evaluations'][:3]):
        before_reuse, reuse = retained['paid_score_reuse'], repeated['paid_score_reuse']
        assert reuse['assessment_ref'] == before_reuse['assessment_ref']
        assert reuse['scope_ref'] == before_reuse['scope_ref']
    from hegel_machine.cognition.code_agent import literal_python_model_context_matches, python_model_context
    for retained, alias in zip(choice['evaluations'][1:], compared['evaluations'][3:]):
        reuse = alias['paid_score_reuse']
        assert reuse['matched_model'] == retained['path']
        assert reuse['match_basis'] == 'identical_closed_literal_execution'
        assert reuse['assessment_ref'] == retained['paid_score_reuse']['assessment_ref']
        original_context, original_model = python_model_context(agent.workspace, retained['path'])
        assert literal_python_model_context_matches(agent.workspace, alias['path'],
                                                   original_context, original_model)
    from pathlib import Path
    expected_rows = [{'features': row.features, 'target': row.target,
                      'history': list(row.history)} for row in agent.rows]
    for evaluation in compared['evaluations']:
        reuse = evaluation['paid_score_reuse']
        assert reuse['additional_scoring_compute'] == reuse['additional_scoring_python_seconds'] == 0
        assert json.loads(Path(reuse['scope_ref']['file']).read_text())['rows'] == expected_rows
    from hegel_machine.cognition.paid_score_history import retain_comparison_history
    retained_first = deepcopy(first_result)
    retained_first['feedback'] = retain_comparison_history(first_result['feedback'])
    assert agent.method_results[0] == retained_first
    assert agent.program_contexts[POLICY]['source'] == export_finite_native_author()
    assert agent.program_contexts[POLICY]['start'] == 1
    assert 0 < agent.costs['compute_cost'] - before['compute_cost']
    assert lab.paid == agent.costs['observation_cost'] == before['observation_cost'] == 2
    assert agent.costs['model_calls'] == 0 and not agent.confirmations

    # The same published action at the same paid scope is attempted once even
    # after its result was reused; it must not become another native search.
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'research_state.json').write_text(json.dumps(agent._program_state()))
    next_proposal = select_actions(lab.public_spec(), agent.observations,
        {'finite_native_author': agent.knowledge['finite_native_author']})
    assert next_proposal.get('arguments', {}).get('operator') != 'native_condition_research'


@pytest.mark.parametrize('changed', ['finish', 'other_question', 'other_meaning', 'selected_records', 'attempted'])
def test_same_bank_requires_actual_current_full_scope_refit_failure_and_keeps_finish_first(
        tmp_path, monkeypatch, changed):
    monkeypatch.chdir(tmp_path)
    question = declaration()['question']
    memory = local_memory(questions=[{'id': 'task', 'pending_query': {'actions': []}},
        {**deepcopy(question), 'meaning_version': 1}])
    memory['finite_native_author']['max_compute'] = 100000
    model = 'native.py'
    (tmp_path / model).write_text('def predict(row): return 0.\n')
    latest = {'round_index': 0, 'status': 'candidates_require_evaluation',
        'candidates': [{'model': model}], 'observed_record_count': 1}
    state = memory['research_state']
    state['operator_results'] = [{'operator': 'native_condition_research',
        'research_question_id': question['id'], 'question_meaning_version': 1,
        'native_condition_research': latest}]
    choice = {'operator': 'research_choice', 'research_question_id': question['id'],
        'question_meaning_version': 1, 'paid_rows': 1, 'evaluations': [{'path': model}]}
    state['method_results'] = [
        {'proposal': {'action': 'run', 'arguments': {'operator': 'research_choice', 'models': [model]}},
         'feedback': choice, 'evidence_after': {'rows': 1}},
        {'proposal': {'action': 'run', 'arguments': {'operator': 'refit', 'model': model}},
         'feedback': {'status': 'refit_unavailable', 'research_question_id': question['id'],
                      'question_meaning_version': 1}, 'evidence_after': {'rows': 1}}]
    entry = {'question_id': question['id'], 'round_index': 0, 'paid_rows': 1,
        'current_paid_rows': 1, 'observed_record_count': 1, 'fit_scope': 'all_paid_rows',
        'source': {'file': 'research_state.json',
                   'json_path': ['questions', 1, 'native_condition_research', 0]},
        'resume_action': {'action': 'run', 'arguments': {'operator': 'native_condition_research',
            'research_question': {'id': question['id']}, 'max_compute': 300, 'max_candidates': 16}}}
    if changed == 'finish':
        choice['next_action'] = {'action': 'finish', 'arguments': {'model': model}}
    elif changed == 'other_question':
        entry['question_id'] = 'different_question'
    elif changed == 'other_meaning':
        state['method_results'][-1]['feedback']['question_meaning_version'] = 2
    elif changed == 'selected_records':
        entry['fit_scope'] = 'selected_records'
    (tmp_path / 'method_decision.json').write_text(json.dumps({'research_options': [
        {'id': 'native_condition_research', 'current_executions': [entry]}]}))
    proposed = select_actions({'controls': ['u']}, observations(), memory)
    if changed == 'attempted':
        assert proposed['arguments'] == entry['resume_action']['arguments']
        state['method_results'].append({'proposal': deepcopy(proposed),
            'feedback': {'operator': 'native_condition_research',
                'research_question_id': question['id'], 'question_meaning_version': 1},
            'evidence_after': {'rows': 1}})
        proposed = select_actions({'controls': ['u']}, observations(), memory)
    if changed == 'finish':
        assert proposed['action'] == 'finish'
    else:
        assert proposed['action'] == 'query'
        assert proposed['arguments'] == {'pending_question': 'task'}
