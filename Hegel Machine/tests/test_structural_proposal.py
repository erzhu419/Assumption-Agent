"""Executed author fixtures test structural handoff, not autonomous discovery."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.code_provider_context import prepare_code_context
from hegel_machine.cognition.language import Program
from test_common_archive_llm_inquiry import RestorableConfirmationLab
from test_common_archive_online_research import ScriptedProvider
from test_inquiry_revision import action
from test_scientific_generalization import FEATURE_MEMORIZER, execute, failed_scene, prepared


U = Program('feature', name='u').to_dict()
ZERO = Program('const', value=0.).to_dict()


def binding_source(question_id, hypotheses, *, additions=(), basis=None):
    """Explicit fixture laws keep this test outside the scientific experiment."""
    declaration = {'hypotheses': hypotheses, 'candidate_additions': list(additions)}
    if basis is not None:
        declaration['basis'] = basis
    declarations = json.dumps(declaration)
    return f'''import json
DECLARATIONS = json.loads({declarations!r})
def bind(spec, observations, memory):
    assert memory['question_id'] == {question_id!r}
    rows = []
    for record in observations:
        for row, index in zip(record['rows'], record['row_indices']):
            rows.append({{'features': row['features'], 'history': row['history'],
                         'target': row['target'], 'evidence': [index]}})
    return {{**DECLARATIONS, 'rows': rows}}
'''


def reviewed_scene(path, *, calls=20, provider=None):
    agent, lab = failed_scene(path, source=FEATURE_MEMORIZER, provider=provider)
    agent.config = replace(agent.config, max_model_calls=calls)
    assert agent._before_model_turn(lab)
    execute(agent, lab, 'finish', {'lesson': 'The fitting procedure remains inadequate.'})
    agent._settle_inquiries()
    agent._prepare_parent_research()
    assert agent._parent_research()['parent_research']['phase'] == 'prediction_method_revision'
    return agent, lab


def open_proposal(agent, lab):
    assert agent._before_model_turn(lab)
    frame = agent._inquiry_stack[-1]
    question = frame['question']
    assert question['origin'] == 'structural_proposal'
    return question, question['proposal_path']


def test_repeated_confirmed_failures_schedule_one_real_author_stage_without_purchase(tmp_path):
    agent, lab = reviewed_scene(tmp_path)
    before, observations, actions = deepcopy(agent.costs), deepcopy(agent.observations), deepcopy(lab.actions)
    source = (agent.workspace / 'candidate.py').read_text()
    owner = agent._parent_research()
    baseline = deepcopy(owner['parent_research']['baseline_parent'])
    question, path = open_proposal(agent, lab)

    assert question['id'] == 'structure:' + owner['id']
    assert question['parent_research_question_id'] == owner['id']
    assert path.endswith('/binding.py')
    request = agent._request(lab)
    assert request['llm_inquiry']['question_id'] == question['id']
    assert path in json.dumps(request)
    assert agent._inquiry_stack[-1]['call_limit'] == 12
    assert agent.provider.requests == []
    assert agent.costs == before and agent.observations == observations and lab.actions == actions
    assert agent.model is None and not agent.relation_research
    assert (agent.workspace / 'candidate.py').read_text() == source
    assert owner['parent_research']['baseline_parent'] == baseline

    execute(agent, lab, 'finish', {'lesson': 'No structural candidate yet.', 'inconclusive': True})
    agent._before_model_turn(lab)
    agent._before_model_turn(lab)
    assert not agent._inquiry_stack
    assert len([q for q in agent.questions if q.get('origin') == 'structural_proposal']) == 1
    assert agent._parent_research()['parent_research']['status'] == 'running'
    assert agent.costs == before and lab.paid == 6


def test_refuted_trial_returns_to_same_author_and_explicit_addition_keeps_old_failure(tmp_path):
    provider = ScriptedProvider([])
    agent, lab = reviewed_scene(tmp_path, calls=4, provider=provider)
    question, path = open_proposal(agent, lab)
    before_rows, before_paid, before_actions = deepcopy(agent.observations), lab.paid, deepcopy(lab.actions)
    before_cost = deepcopy(agent.costs)
    provider.steps = [
        ('write', {'path': path, 'code': binding_source(question['id'], [ZERO])}),
        ('write', {'path': path, 'code': binding_source(question['id'], [U], additions=[U])}),
        ('finish', {'lesson': 'The new explicit fixture candidate fits paid evidence; no fresh test.'}),
        ('run', {'code': "print('The parent receives the structural result before choosing confirmation')"}),
    ]
    result = agent.run(lab)

    first, second = provider.requests[:2]
    assert first['llm_inquiry']['question_id'] == second['llm_inquiry']['question_id'] == question['id']
    first_result = second['events'][-1]['feedback']['structural_trial']
    assert first_result['relation_research']['candidates'][0]['status'] == 'refuted'
    assert first_result['relation_research']['exported_models'] == []
    bank = result['relation_research'][question['id']]
    assert bank['round_index'] == 1
    assert bank['candidates'][0]['program'] == ZERO
    assert bank['candidates'][0]['status'] == 'refuted'
    assert bank['candidates'][0]['fit_mse'] == first_result['relation_research']['candidates'][0]['fit_mse']
    assert bank['candidates'][1]['program'] == U and bank['candidates'][1]['status'] == 'plausible'
    assert bank['exported_models']
    assert len(provider.requests) == result['costs']['model_calls'] == 4
    assert result['costs']['reported_tokens'] - before_cost['reported_tokens'] == 4
    assert result['costs']['compute_cost'] > before_cost['compute_cost']
    assert result['costs']['python_execution_seconds'] > before_cost['python_execution_seconds']
    assert result['paid_observations'] == before_rows and lab.paid == before_paid
    assert lab.actions == before_actions
    assert not provider.requests[3].get('llm_inquiry')
    parent = provider.requests[3]['online_research']['parent_returns']
    answers = [a for entry in parent for a in entry['answers'] if a['question_id'] == question['id']]
    assert answers[-1]['operator'] == 'research_update'
    assert answers[-1]['answer']['models']
    assert result['model'] is None and len(agent.confirmations) == 2
    assert agent._parent_research()['parent_research']['status'] == 'running'


def test_refuted_basis_expansion_is_fitted_and_residuals_reach_same_author(tmp_path):
    provider = ScriptedProvider([])
    agent, lab = reviewed_scene(tmp_path, calls=4, provider=provider)
    question, path = open_proposal(agent, lab)
    paid, observations = lab.paid, deepcopy(agent.observations)
    provider.steps = [
        ('write', {'path': path,
                   'code': binding_source(question['id'], [], basis=[ZERO])}),
        ('write', {'path': path,
                   'code': binding_source(question['id'], [], basis=[ZERO, U])}),
        ('finish', {'lesson': 'The revised fixture fits paid data; confirmation remains open.'}),
        ('run', {'code': "print('Parent chooses whether to confirm')"}),
    ]
    result = agent.run(lab)

    feedback = provider.requests[1]['structural_proposal']['last_trial']['relation_research']
    assert feedback['candidates'][0]['status'] == 'refuted'
    errors = feedback['candidates'][0]['paid_fit_residuals']
    assert errors['evaluation_scope'] == 'paid_relation_fit'
    assert errors['evaluated_rows'] == paid
    assert errors['examples'][0]['squared_error'] > 0
    for example in errors['examples']:
        index = example['evidence'][0]
        assert example['target'] == agent.rows[index].target
        assert example['squared_error'] == (example['prediction'] - example['target']) ** 2
    bank = result['relation_research'][question['id']]
    assert bank['round_index'] == 1 and len(bank['candidates']) == 2
    assert bank['candidates'][0]['status'] == 'refuted'
    assert bank['candidates'][1]['status'] == 'plausible'
    assert bank['candidates'][1]['fit_mse'] < 1e-12
    assert bank['compute_cost'] == 2 * paid
    assert bank['exported_models']
    assert provider.requests[2]['llm_inquiry']['question_id'] == question['id']
    assert provider.requests[2]['structural_proposal']['last_trial']['relation_research'][
        'candidates'][1]['paid_fit_residuals']['evaluated_rows'] == paid
    assert result['paid_observations'] == observations and lab.paid == paid
    assert result['model'] is None and len(agent.confirmations) == 2


def test_invalid_binding_returns_execution_error_then_same_author_can_repair(tmp_path):
    provider = ScriptedProvider([])
    agent, lab = reviewed_scene(tmp_path, calls=4, provider=provider)
    question, path = open_proposal(agent, lab)
    provider.steps = [
        ('write', {'path': path, 'code': 'def bind(:\n'}),
        ('write', {'path': path, 'code': binding_source(question['id'], [U])}),
        ('finish', {'lesson': 'The repaired authored binding executes.'}),
        ('run', {'code': "print('Parent decision remains independent')"}),
    ]
    result = agent.run(lab)
    repair_request = provider.requests[1]
    assert repair_request['llm_inquiry']['question_id'] == question['id']
    assert 'SyntaxError' in json.dumps(repair_request['events'][-1]['feedback'])
    assert repair_request['relation_research'] == {}
    assert result['relation_research'][question['id']]['round_index'] == 0
    assert result['relation_research'][question['id']]['selected_program'] == U
    assert result['costs']['model_calls'] == 4 and lab.paid == 6
    assert result['model'] is None and len(agent.confirmations) == 2


def test_stage_budget_returns_to_parent_without_reopening_or_extra_model_calls(tmp_path):
    provider = ScriptedProvider([])
    agent, lab = reviewed_scene(tmp_path, calls=3, provider=provider)
    question, path = open_proposal(agent, lab)
    assert agent._inquiry_stack[-1]['call_limit'] == 2
    provider.steps = [
        ('write', {'path': path, 'code': binding_source(question['id'], [ZERO])}),
        ('run', {'code': "print('The authored candidate remains refuted')"}),
        ('run', {'code': "print('Parent still has a decision turn')"}),
    ]
    result = agent.run(lab)
    assert provider.requests[1]['llm_inquiry']['question_id'] == question['id']
    assert not provider.requests[2].get('llm_inquiry')
    assert result['costs']['model_calls'] == len(provider.requests) == 3
    assert len([q for q in agent.questions if q.get('origin') == 'structural_proposal']) == 1
    assert agent.llm_inquiries[question['id']]['model_calls'] == 2
    assert not agent._inquiry_stack and lab.paid == 6
    assert result['model'] is None


def test_restore_retains_active_author_bank_old_parent_and_paid_account(tmp_path):
    original, lab = reviewed_scene(tmp_path / 'original')
    question, path = open_proposal(original, lab)
    execute(original, lab, 'write', {'path': path, 'code': binding_source(question['id'], [ZERO])})
    original._prepare_program_state()
    previous = original.result('interrupted_inside_structural_proposal')
    before = deepcopy(original.costs)
    owner = next(q for q in original.questions if q['id'] == question['parent_research_question_id'])
    baseline = deepcopy(owner['parent_research']['baseline_parent'])
    restored = prepared(tmp_path / 'restored', source=FEATURE_MEMORIZER)
    fresh = RestorableConfirmationLab()
    report = restored.restore_practice_research(previous, original.workspace, fresh, confirmation_seed=1709)
    assert report['new_observations'] == 0
    assert restored.relation_research == original.relation_research
    assert restored.costs == before and fresh.paid == 6 and fresh.actions == []
    assert restored._request(fresh)['llm_inquiry']['question_id'] == question['id']
    restored_owner = next(q for q in restored.questions if q['id'] == owner['id'])
    assert restored_owner['parent_research']['baseline_parent'] == baseline
    execute(restored, fresh, 'write', {'path': path,
        'code': binding_source(question['id'], [U], additions=[U])})
    bank = restored.relation_research[question['id']]
    assert bank['round_index'] == 1 and bank['candidates'][0]['program'] == ZERO
    assert bank['candidates'][0]['status'] == 'refuted'
    assert bank['candidates'][1]['program'] == U
    assert fresh.paid == 6 and fresh.actions == [] and restored.model is None
    assert len(restored.confirmations) == 2


def test_explicit_local_model_return_scores_paid_rows_without_confirming_parent(tmp_path):
    agent, lab = reviewed_scene(tmp_path)
    owner = agent._parent_research()
    baseline, old_source = deepcopy(owner['parent_research']['baseline_parent']), (
        agent.workspace / 'candidate.py').read_text()
    question, path = open_proposal(agent, lab)
    execute(agent, lab, 'write', {'path': path, 'code': binding_source(question['id'], [U])})
    exported = agent.relation_research[question['id']]['exported_models'][0]['model']
    before_paid, before_actions = lab.paid, deepcopy(lab.actions)
    result = execute(agent, lab, 'finish', {'model': exported,
        'lesson': 'This executable fixture fits paid rows; independent confirmation is unresolved.'})
    assert result['evaluation']['returncode'] == 0 and result['evaluation']['mse'] == 0.
    agent._before_model_turn(lab)
    assert not agent._inquiry_stack
    assert question['returned_model'] == exported
    assert owner['structural_proposal']['returned_model'] == exported
    assert owner['parent_research']['evaluated_model'] == exported
    assert owner['parent_research']['baseline_parent'] == baseline
    assert owner['parent_research']['status'] == 'running'
    assert (agent.workspace / 'candidate.py').read_text() == old_source
    assert path in question['structural_return']['artifacts']
    assert exported in question['structural_return']['artifacts']
    assert agent.model is None and len(agent.confirmations) == 2
    assert lab.paid == before_paid and lab.actions == before_actions


@pytest.mark.parametrize('old_saved_shape', [False, True])
def test_partial_return_evaluation_survives_later_rows_provider_window_and_restore(tmp_path, old_saved_shape):
    agent, lab = reviewed_scene(tmp_path / 'source')
    execute(agent, lab, 'evaluate', {'model': 'better.py',
        'code': "def represent(row): return row['features']['u']\n"
                "def predict_features(value): return value + .01\n"})
    incumbent = deepcopy(agent._evaluated_summary())
    incumbent_source = (agent.workspace / incumbent['model']).read_text()
    question, _ = open_proposal(agent, lab)
    owner = next(q for q in agent.questions
                 if q['id'] == question['parent_research_question_id'])
    execute(agent, lab, 'write', {'path': 'partial.py',
        'code': "def represent(row): return row['features']['u']\n"
                "def predict_features(value): return value + .062\n"})
    rows, paid, confirmations = deepcopy(agent.observations), lab.paid, deepcopy(agent.confirmations)
    finish_index = len(agent.events)
    result = execute(agent, lab, 'finish', {'model': 'partial.py',
        'lesson': 'An approximate predictor is available; its parent remains unresolved.'})
    residuals = deepcopy(result['evaluation']['paid_residuals'])
    agent._before_model_turn(lab)

    assert result['evaluation']['mse'] > agent.config.confirmation_mse_threshold
    assert result['evaluation']['mse'] > incumbent['mse']
    assert not agent._inquiry_stack and agent.model is None
    assert agent._evaluated_summary() == incumbent
    assert owner['parent_research']['phase'] == 'prediction_method_revision'
    assert lab.paid == paid and agent.confirmations == confirmations
    assert agent.observations == rows
    returned_at = owner['structural_proposal']['returned_at_model_call']
    if old_saved_shape:
        question['structural_return'].pop('evaluation')
        owner['structural_proposal']['returned'].pop('evaluation')
        agent.research_candidates['partial.py']['current_evaluation'].pop('paid_residuals')

    sibling = {'id': 'next_investigation', 'question': 'Inspect later paid evidence',
               'parent_id': 'task', 'depends_on': []}
    execute(agent, lab, 'query', {'actions': [action(.3, .7)], 'question': sibling})
    for index in range(10):
        execute(agent, lab, 'write', {'path': 'scratch_note.txt', 'code': str(index),
            'research_question': sibling, 'notes': 'Later investigation without old citations.'})
    before = deepcopy((agent.questions, agent.events, agent.costs))
    context = prepare_code_context(agent.workspace, agent._request(lab), schema={}, instructions='test')
    returned = context['evidence_notebook']['facts']['research_return:' + question['id']]
    evaluation = returned['evaluation']

    assert context['event_history']['preview_start'] > finish_index
    assert evaluation['event_index'] == finish_index and evaluation['model'] == 'partial.py'
    assert evaluation['mse'] == result['evaluation']['mse'] and evaluation['returncode'] == 0
    assert evaluation['paid_rows'] == paid < context['evidence_notebook']['current_paid_rows']
    assert evaluation['paid_residuals'] == residuals
    assert residuals['evaluation_scope'] == 'paid_response_evaluation'
    assert residuals['model'] == 'partial.py' and residuals['evaluated_rows'] == paid
    assert len(residuals['examples']) == 3
    cached = agent.research_candidates['partial.py']['current_evaluation']
    if old_saved_shape:
        assert 'evaluation' not in question['structural_return']
        assert 'paid_residuals' not in cached
    else:
        assert cached['paid_residuals'] == residuals
    assert 'supported' not in returned and 'fresh_confirmation' not in evaluation
    if old_saved_shape:
        assert 'evaluation' not in owner['structural_proposal']['returned']
    else:
        assert owner['structural_proposal']['returned']['evaluation'] == evaluation
    assert owner['structural_proposal']['returned_at_model_call'] == returned_at
    assert owner['parent_research']['status'] == 'running'
    assert owner['parent_research']['phase'] == 'prediction_method_revision'
    assert agent._evaluated_summary()['model'] == incumbent['model']
    assert (agent.workspace / incumbent['model']).read_text() == incumbent_source
    assert agent.confirmations == confirmations and lab.paid == paid + 2
    assert agent.observations[:len(rows)] == rows
    assert (agent.questions, agent.events, agent.costs) == before

    previous = agent.result('model_call_budget_exhausted')
    restored = prepared(tmp_path / 'restored', calls=21)
    fresh = RestorableConfirmationLab()
    restored.restore_research(previous, agent.workspace, fresh, confirmation_seed=1709)
    fact = restored._request(fresh)['evidence_notebook']['facts']['research_return:' + question['id']]
    assert fact == returned
    assert restored.costs == agent.costs and restored.observations == agent.observations
    assert restored.confirmations == confirmations and fresh.actions == []
    assert restored.model is None
    restored_owner = next(q for q in restored.questions if q['id'] == owner['id'])
    assert restored_owner['structural_proposal']['returned_at_model_call'] == returned_at
    if old_saved_shape:
        restored_question = next(q for q in restored.questions if q['id'] == question['id'])
        assert 'evaluation' not in restored_question['structural_return']


def test_manually_adopted_bank_skips_additional_host_stage(tmp_path):
    agent, lab = reviewed_scene(tmp_path)
    relation = {'id': 'author_selected_relation', 'question': 'Fixture author relation',
                'parent_id': 'task', 'depends_on': []}
    execute(agent, lab, 'write', {'path': 'manual_binding.py',
        'code': binding_source(relation['id'], [U])})
    execute(agent, lab, 'run', {'operator': 'research_update', 'path': 'manual_binding.py',
                               'research_question': relation})
    before = deepcopy(agent.costs)
    agent._before_model_turn(lab)
    assert not any(q.get('origin') == 'structural_proposal' for q in agent.questions)
    assert not agent._inquiry_stack and agent.costs == before and lab.paid == 6

    failed = execute(agent, lab, 'finish', {'model': 'candidate.py'})
    assert failed['fresh_confirmation']['passed'] is False
    agent._before_model_turn(lab)
    assert agent._inquiry_stack[-1]['question']['origin'] == 'fresh_confirmation_failure'
    execute(agent, lab, 'finish', {'lesson': 'The author keeps its manually adopted relation bank.'})
    agent._before_model_turn(lab)
    assert not agent._inquiry_stack
    assert not any(q.get('origin') == 'structural_proposal' for q in agent.questions)
    assert list(agent.relation_research) == [relation['id']]


def test_ordinary_write_and_other_child_file_do_not_execute_a_structural_binding(tmp_path):
    ordinary = prepared(tmp_path / 'ordinary', source=FEATURE_MEMORIZER)
    fresh = RestorableConfirmationLab()
    saved = execute(ordinary, fresh, 'write', {'path': 'binding.py', 'code': 'def bind(:\n'})
    assert saved['written'] == 'binding.py' and 'structural_trial' not in saved
    assert not ordinary.relation_research and ordinary.costs['compute_cost'] == 0

    agent, lab = reviewed_scene(tmp_path / 'active')
    open_proposal(agent, lab)
    before = deepcopy(agent.costs)
    result = execute(agent, lab, 'write', {'path': 'unrelated.py', 'code': 'x = 7\n'})
    assert result['written'] == 'unrelated.py' and 'structural_trial' not in result
    assert not agent.relation_research and agent.costs == before and lab.paid == 6


def refuted_proposal_scene(path, *, return_predictor=False):
    agent, lab = reviewed_scene(path)
    owner = agent._parent_research()
    question, binding = open_proposal(agent, lab)
    for _ in range(5):
        execute(agent, lab, 'write', {'path': binding,
            'code': binding_source(question['id'], [ZERO])})
    assert agent.relation_research[question['id']]['round_index'] == 4
    assert agent.relation_research[question['id']]['status'] == 'refuted'
    returned = {'lesson': 'The authored fixture structure remains refuted.'}
    if return_predictor:
        execute(agent, lab, 'write', {'path': 'partial.py',
            'code': "def represent(row): return row['features']['u']\n"
                    "def predict_features(value): return value + .01\n"})
        returned['model'] = 'partial.py'
    execute(agent, lab, 'finish', returned)
    agent._before_model_turn(lab)
    assert not agent._inquiry_stack
    return agent, lab, owner, question, binding


def later_failed_review(agent, lab):
    failed = execute(agent, lab, 'finish', {'model': 'candidate.py'})
    assert failed['fresh_confirmation']['passed'] is False
    assert agent._before_model_turn(lab)
    owner = agent._inquiry_stack[-1]['question']
    assert owner['origin'] == 'fresh_confirmation_failure'
    before = deepcopy((agent.costs, agent.observations, lab.actions))
    execute(agent, lab, 'finish', {'lesson': 'The later independent failure needs structural revision.'})
    assert agent._before_model_turn(lab)
    assert (agent.costs, agent.observations, lab.actions) == before
    return owner


def test_later_failure_reuses_authored_bank_and_returns_new_trial_to_new_review(tmp_path):
    agent, lab, old_owner, question, path = refuted_proposal_scene(tmp_path)
    old_return = deepcopy(old_owner['structural_proposal'])
    old_bank = deepcopy(agent.relation_research[question['id']])
    old_notes = deepcopy(agent.llm_inquiries[question['id']])
    old_confirmation = deepcopy(agent.confirmations)
    old_observations = deepcopy(agent.observations)
    new_owner = later_failed_review(agent, lab)

    resumed = agent._inquiry_stack[-1]['question']
    assert resumed['id'] == question['id']
    assert resumed['proposal_path'] == path
    assert resumed['parent_research_question_id'] == new_owner['id'] != old_owner['id']
    assert len([q for q in agent.questions if q.get('origin') == 'structural_proposal']) == 1
    assert agent.relation_research[resumed['id']] == old_bank
    assert old_owner['structural_proposal'] == old_return
    assert agent.llm_inquiries[resumed['id']]['notes'] == old_notes['notes']
    assert agent.llm_inquiries[resumed['id']]['invocations'] == old_notes['invocations'] + 1
    assert agent.confirmations[:-1] == old_confirmation
    assert agent.observations[:len(old_observations)] == old_observations
    assert (agent.workspace / path).read_text() == binding_source(resumed['id'], [ZERO])

    before_rows, before_paid = deepcopy(agent.observations), lab.paid
    before_actions, before_cost = deepcopy(lab.actions), deepcopy(agent.costs)
    trial = execute(agent, lab, 'write', {'path': path,
        'code': binding_source(resumed['id'], [U], additions=[U])})
    bank = agent.relation_research[resumed['id']]
    assert trial['structural_trial']['research_question_id'] == resumed['id']
    assert bank['round_index'] == old_bank['round_index'] + 1
    assert bank['round_history'][:-1] == old_bank['round_history']
    assert bank['candidates'][0]['program'] == ZERO
    assert bank['candidates'][0]['status'] == 'refuted'
    assert bank['candidates'][0]['fit_evidence'] == old_bank['candidates'][0]['fit_evidence']
    assert bank['candidates'][1]['program'] == U
    assert bank['candidates'][1]['status'] == 'plausible'
    assert agent.costs['compute_cost'] > before_cost['compute_cost']
    exported = bank['exported_models'][0]['model']
    execute(agent, lab, 'finish', {'model': exported,
        'lesson': 'The explicit revised fixture fits paid rows; the parent decides confirmation.'})
    agent._before_model_turn(lab)

    assert new_owner['structural_proposal']['question_id'] == resumed['id']
    assert new_owner['structural_proposal']['returned']['evaluation']['model'] == exported
    assert new_owner['parent_research']['evaluated_model'] == exported
    assert old_owner['structural_proposal'] == old_return
    assert agent.observations == before_rows and lab.paid == before_paid
    assert lab.actions == before_actions
    assert len(agent.confirmations) == len(old_confirmation) + 1
    assert agent.model is None and not agent._inquiry_stack


def test_refuted_bank_does_not_restart_without_another_independent_failure(tmp_path):
    agent, lab, owner, question, _ = refuted_proposal_scene(tmp_path)
    before = deepcopy((agent.costs, agent.relation_research, owner['structural_proposal'],
                       agent.llm_inquiries[question['id']], lab.actions))
    for _ in range(3):
        agent._before_model_turn(lab)
    assert not agent._inquiry_stack
    assert (agent.costs, agent.relation_research, owner['structural_proposal'],
            agent.llm_inquiries[question['id']], lab.actions) == before


def test_previous_local_predictor_is_not_a_new_return_for_later_review(tmp_path):
    agent, lab, old_owner, question, _ = refuted_proposal_scene(tmp_path, return_predictor=True)
    old_return = deepcopy(question['structural_return'])
    old_owner_return = deepcopy(old_owner['structural_proposal'])
    assert question['returned_model'] == 'partial.py'
    new_owner = later_failed_review(agent, lab)

    assert question['structural_return'] == old_return
    assert new_owner['structural_proposal']['status'] == 'running'
    assert not {'returned', 'returned_model', 'returned_at_model_call'} & new_owner[
        'structural_proposal'].keys()
    assert new_owner['parent_research'].get('evaluated_model') != 'partial.py'
    execute(agent, lab, 'finish', {'inconclusive': True,
        'lesson': 'No revised predictor is returned from this invocation.'})
    agent._before_model_turn(lab)

    returned = new_owner['structural_proposal']['returned']
    assert 'evaluation' not in returned and 'returned_model' not in new_owner['structural_proposal']
    assert min(returned['event_indices']) > max(old_return['event_indices'])
    assert new_owner['parent_research']['status'] == 'running'
    assert old_owner['structural_proposal'] == old_owner_return
    assert agent.model is None and not agent._inquiry_stack


@pytest.mark.parametrize('active', [True, False])
def test_restoring_reentered_stage_neither_replays_trial_nor_reopens_finished_review(tmp_path, active):
    agent, lab, old_owner, question, _ = refuted_proposal_scene(tmp_path / 'source')
    old_return = deepcopy(old_owner['structural_proposal'])
    new_owner = later_failed_review(agent, lab)
    if not active:
        execute(agent, lab, 'finish', {'inconclusive': True,
            'lesson': 'No new fixture structure; leave this invocation unresolved.'})
        agent._before_model_turn(lab)
        assert not agent._inquiry_stack
    agent._prepare_program_state()
    prepare_code_context(agent.workspace, agent._request(lab), schema={}, instructions='test')
    previous = agent.result('interrupted' if active else 'unresolved')
    restored = prepared(tmp_path / 'restored', source=FEATURE_MEMORIZER)
    fresh = RestorableConfirmationLab()
    report = restored.restore_research(previous, agent.workspace, fresh, confirmation_seed=1709)
    before = deepcopy((restored.costs, restored.relation_research, restored.llm_inquiries,
                       restored.confirmations, restored.observations))
    restored._before_model_turn(fresh)
    restored._before_model_turn(fresh)

    assert report['new_observations'] == 0 and fresh.actions == []
    assert (restored.costs, restored.relation_research, restored.llm_inquiries,
            restored.confirmations, restored.observations) == before
    assert restored.costs == agent.costs and restored.confirmations == agent.confirmations
    restored_old = next(q for q in restored.questions if q['id'] == old_owner['id'])
    assert restored_old['structural_proposal'] == old_return
    restored_new = next(q for q in restored.questions if q['id'] == new_owner['id'])
    assert restored_new['structural_proposal']['question_id'] == question['id']
    assert bool(restored._inquiry_stack) is active
    if active:
        resumed = restored._request(fresh)['structural_proposal']
        assert resumed['question_id'] == question['id']
        assert resumed['parent_research_question_id'] == new_owner['id']
