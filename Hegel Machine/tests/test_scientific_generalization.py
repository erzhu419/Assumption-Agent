"""Executed fixtures check scientific-review control, not autonomous discovery."""

from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.scientific_generalization import (
    generalization_failures, prepare_generalization_review, resume_generalization_review)
from hegel_machine.cognition import temporal_operator_agent
from scripts import run_scientific_revision_comparison as runner
from test_common_archive_llm_inquiry import RestorableConfirmationLab
from test_common_archive_online_research import ScriptedProvider
from test_inquiry_revision import action


QUESTION = {'id': 'task', 'question': 'Predict the public response',
            'parent_id': None, 'depends_on': []}
MEMORIZER = """from collections import Counter
table = {}
def fit(rows):
    global table
    groups = Counter(r['history'][0]['u'] if r['history'] else r['features']['u'] for r in rows)
    assert set(groups.values()) == {2}, 'a trajectory was split across fitting and validation'
    table = {r['features']['u']: r['target'] for r in rows}
def predict(row):
    assert 'target' not in row
    return table.get(row['features']['u'], 0.)
"""
FEATURE_MEMORIZER = """from collections import Counter
table = {}
def represent(row):
    h, f = row['history'], row['features']
    return {'trajectory': h[0]['u'] if h else f['u'], 'u': f['u']}
def fit_features(features, targets):
    global table
    groups = Counter(f['trajectory'] for f in features)
    assert set(groups.values()) == {2}, 'a trajectory was split across fitting and validation'
    table = {f['u']: y for f, y in zip(features, targets)}
def predict_features(features):
    return table.get(features['u'], 0.)
"""


def prepared(path, provider=None, *, source=MEMORIZER, calls=20):
    return CommonArchiveInquiryAgent(provider or ScriptedProvider([]), path, arm='hegel',
        source_task=False, confirmation_seed=1709,
        config=InquiryLLMConfig(observation_budget=20, confirmation_rows=2,
            compute_budget=2000, execution_seconds=30., max_model_calls=calls),
        knowledge={'files': {'candidate.py': source,
            'shared/knowledge.json': json.dumps({'hypotheses': [], 'research_programs': [],
                                                'support_questions': []})}})


def execute(agent, lab, selected, arguments):
    result = agent._execute(selected, deepcopy(arguments), lab)
    agent.events.append({'step': len(agent.events), 'action': selected,
        'arguments': deepcopy(arguments), 'feedback': result, 'usage': {'total_tokens': 0}})
    return result


def failed_scene(path, *, source=MEMORIZER, provider=None):
    agent, lab = prepared(path, provider, source=source), RestorableConfirmationLab()
    execute(agent, lab, 'query', {'actions': [action(.25, .75)], 'question': QUESTION})
    for _ in range(2):
        result = execute(agent, lab, 'finish', {'model': 'candidate.py'})
        assert result['mse'] == 0.
        assert result['fresh_confirmation']['passed'] is False
    return agent, lab


def failure(*, paid_mse=0., fresh_mse=.1, returncode=0, passed=False):
    return {'passed': passed, 'returncode': returncode, 'mse': fresh_mse,
            'submitted_candidate': {'mse': paid_mse, 'model': 'candidate.py'}}


@pytest.mark.parametrize('invalid', [
    failure(returncode=1), failure(fresh_mse=None), failure(fresh_mse=float('nan')),
    failure(fresh_mse=float('inf')), failure(paid_mse=float('nan')),
])
def test_execution_errors_and_nonfinite_errors_do_not_make_two_measured_failures(invalid):
    assert len(generalization_failures([failure(), invalid], 1e-6)) < 2


def test_review_requires_repeated_low_paid_fit_failure_and_success_resets_it():
    assert len(generalization_failures([failure()], 1e-6)) < 2
    assert len(generalization_failures([failure(), failure()], 1e-6)) == 2
    assert len(generalization_failures([failure(), failure(paid_mse=.01)], 1e-6)) < 2
    assert not generalization_failures([failure(), failure(), failure(passed=True, fresh_mse=0.)], 1e-6)


@pytest.mark.parametrize('source', [MEMORIZER, FEATURE_MEMORIZER])
def test_review_refits_without_split_trajectories_new_labels_or_source_changes(tmp_path, source):
    agent, lab = failed_scene(tmp_path, source=source)
    before_costs, before_observations = deepcopy(agent.costs), deepcopy(agent.observations)
    before_actions = deepcopy(lab.actions)
    before_source = (agent.workspace / 'candidate.py').read_text()
    assert agent._before_model_turn(lab)
    question = next(q for q in agent.questions if q.get('generalization_review'))

    review = question['generalization_review']
    assert len(review['failed_attempts']) == 2
    assert question['review_scope'] == 'prediction_method_revision'
    assert question['review_model_calls'] == 12
    refit = question['whole_trajectory_refit']
    assert refit['validation']['returncode'] == 0
    assert refit['validation']['mse'] > 1e-6
    assert refit['status'] == 'refit_inadequate_on_validation'
    assert agent.costs['compute_cost'] > before_costs['compute_cost']
    assert agent.costs['python_execution_seconds'] > before_costs['python_execution_seconds']
    assert agent.costs['model_calls'] == before_costs['model_calls']
    assert agent.costs['observation_cost'] == before_costs['observation_cost']
    assert agent.observations == before_observations and lab.actions == before_actions
    assert (agent.workspace / 'candidate.py').read_text() == before_source == source
    assert review['validation_confirmation_indices'] == [0, 1]
    assert review['validation_records'] == [1, 2]
    args = review['refit_arguments']
    assert args['validation_records'] == [1, 2]
    assert args['records'] == [0, 1, 2]
    assert set(review['training_records']).isdisjoint(args['validation_records'])
    assert review['training_records'] == [0]


def test_real_repeated_failures_enter_method_assignment_and_paid_fit_does_not_clear_it(tmp_path):
    agent, lab = failed_scene(tmp_path, source=FEATURE_MEMORIZER)
    assert agent._before_model_turn(lab)
    request = agent._request(lab)
    assignment = request['scientific_revision_assignment']
    assert assignment['review_scope'] == 'prediction_method_revision'
    question = next(q for q in agent.questions if q['id'] == assignment['id'])
    assert question['review_model_calls'] == 12
    assert request['llm_inquiry']['assignment']['review_scope'] == 'prediction_method_revision'
    assert question['status'] != 'refuted'

    paid = execute(agent, lab, 'evaluate', {'model': 'candidate.py'})
    assert paid['mse'] == 0.
    execute(agent, lab, 'finish', {'lesson': 'The unchanged memorizer is unresolved.'})
    agent._before_model_turn(lab)
    assert agent._inquiry_stack[-1]['question']['origin'] == 'structural_proposal'
    execute(agent, lab, 'finish', {'lesson': 'No revised structural declaration in this fixture.'})
    agent._before_model_turn(lab)
    parent = agent._request(lab)
    assert not parent.get('llm_inquiry')
    assert parent['parent_research_focus']['phase'] == 'prediction_method_revision'
    execute(agent, lab, 'evaluate', {'model': 'candidate.py'})
    agent._before_model_turn(lab)
    assert agent._request(lab)['parent_research_focus']['phase'] == 'prediction_method_revision'
    assert agent.model is None and lab.paid == 6


def test_incomplete_confirmation_record_mapping_does_not_refit_or_buy_evidence(tmp_path):
    agent, lab = failed_scene(tmp_path)
    incomplete = deepcopy(agent.confirmations[-1])
    incomplete['observation_indices'] = incomplete['observation_indices'][:1]
    question = agent._open_question({'id': 'incomplete_mapping', 'parent_id': 'task',
                                    'question': 'Review the mapped confirmation records.'})
    before, before_operators, before_paid = deepcopy(agent.costs), deepcopy(agent.operator_results), lab.paid
    prepare_generalization_review(agent, question, incomplete, lab)
    assert question['whole_trajectory_refit']['status'] == 'confirmation_record_mapping_unavailable'
    assert agent.costs == before and agent.operator_results == before_operators and lab.paid == before_paid
    assert 'refit_arguments' not in question['generalization_review']
    assert agent.model is None and question['status'] != 'refuted'


def test_method_revision_keeps_existing_representation_research_available(tmp_path):
    agent, lab = failed_scene(tmp_path, source=FEATURE_MEMORIZER)
    agent._before_model_turn(lab)
    execute(agent, lab, 'finish', {'lesson': 'The original fitting method remains unresolved.'})
    agent._before_model_turn(lab)
    assert agent._inquiry_stack[-1]['question']['origin'] == 'structural_proposal'
    execute(agent, lab, 'finish', {'lesson': 'No revised structural declaration in this fixture.'})
    agent._before_model_turn(lab)
    execute(agent, lab, 'evaluate', {'model': 'new_opaque.py',
                                    'code': 'def predict(row): return 0.\n'})
    assert agent._parent_research()['parent_research']['phase'] == 'prediction_method_revision'
    assert agent._before_model_turn(lab)
    assert agent._inquiry_stack[-1]['question']['origin'] == 'representation_research'
    assert agent._inquiry_stack[-1]['question']['representation_model'] == 'new_opaque.py'
    assert lab.paid == 6 and agent.model is None


def test_same_confirmation_refit_and_checkpoint_restore_do_not_pay_twice(tmp_path):
    old, lab = failed_scene(tmp_path / 'old', source=FEATURE_MEMORIZER)
    assert old._before_model_turn(lab)
    old._prepare_program_state()
    previous = old.result('interrupted_before_first_method_review_call')
    before = deepcopy(old.costs)
    assert not old._before_model_turn(lab)
    assert old.costs == before
    original = next(q for q in previous['online_questions'] if q.get('generalization_review'))

    restored = prepared(tmp_path / 'restored', source=FEATURE_MEMORIZER)
    fresh = RestorableConfirmationLab()
    report = restored.restore_practice_research(previous, old.workspace, fresh,
                                               confirmation_seed=1709)
    assert report['new_observations'] == 0
    assert restored.costs == before
    restored._before_model_turn(fresh)
    assert restored.costs == before and fresh.actions == [] and fresh.paid == 6
    current = next(q for q in restored.questions if q['id'] == original['id'])
    assert current['generalization_review'] == original['generalization_review']
    assert current['whole_trajectory_refit'] == original['whole_trajectory_refit']
    assert len([q for q in restored.questions if q.get('generalization_review')]) == 1
    assert restored._request(fresh)['llm_inquiry']['assignment']['review_scope'] == 'prediction_method_revision'


def test_first_measured_failure_keeps_eight_call_scientific_revision(tmp_path):
    agent, lab = prepared(tmp_path), RestorableConfirmationLab()
    execute(agent, lab, 'query', {'actions': [action(.25, .75)], 'question': QUESTION})
    execute(agent, lab, 'finish', {'model': 'candidate.py'})
    assert agent._before_model_turn(lab)
    request = agent._request(lab)
    assignment = request['scientific_revision_assignment']
    assert assignment.get('review_scope') != 'prediction_method_revision'
    question = next(q for q in agent.questions if q['id'] == assignment['id'])
    assert not question.get('generalization_review')
    assert agent._inquiry_stack[-1]['call_limit'] == 8
    assert lab.paid == 4


def test_one_successful_fresh_submission_keeps_existing_path(tmp_path):
    provider = ScriptedProvider([('finish', {'model': 'candidate.py'})])
    agent = prepared(tmp_path, provider, source="def predict(row): return row['features']['u']\n", calls=1)
    lab = RestorableConfirmationLab()
    execute(agent, lab, 'query', {'actions': [action(.25, .75)], 'question': QUESTION})
    result = agent.run(lab)
    assert result['status'] == 'completed' and result['fresh_confirmation']['passed']
    assert result['costs']['model_calls'] == 1 and result['costs']['observation_cost'] == 4
    assert not any(q.get('generalization_review') for q in result['online_questions'])
    assert not provider.requests[0].get('scientific_revision_assignment')


@pytest.fixture
def interrupted_review(tmp_path, monkeypatch):
    agent, lab = failed_scene(tmp_path / 'old', source=FEATURE_MEMORIZER)

    def interrupted_copy(workspace, model, destination):
        destination.mkdir(parents=True)
        (destination / 'partial.txt').write_text('Retain the interrupted context.')
        raise OSError(28, 'No space left on device')

    with monkeypatch.context() as patch:
        patch.setattr(temporal_operator_agent, '_copy_model_context', interrupted_copy)
        assert agent._before_model_turn(lab)
    question = next(q for q in agent.questions if q.get('generalization_review'))
    assert question['whole_trajectory_refit']['status'] == 'refit_execution_failed'
    execute(agent, lab, 'finish', {'lesson': 'The original review could not execute.'})
    agent._settle_inquiries()
    agent._prepare_parent_research()
    execute(agent, lab, 'finish', {'inconclusive': True, 'lesson': 'The method remains unresolved.'})
    assert question['parent_research']['status'] == 'inconclusive'
    return agent, lab, question


def test_explicit_resume_runs_original_refit_once_and_keeps_failed_history(interrupted_review):
    agent, lab, question = interrupted_review
    original = deepcopy(question['generalization_review']['refit_arguments'])
    old_results, before = deepcopy(agent.operator_results), deepcopy(agent.costs)
    observations, lab_actions = deepcopy(agent.observations), deepcopy(lab.actions)
    source = (agent.workspace / 'candidate.py').read_text()
    occupied = agent.workspace / (original['path'] + '_resume_0001')
    occupied.mkdir()
    (occupied / 'partial.txt').write_text('An earlier interrupted output must not be reused.')

    retry = resume_generalization_review(agent, question, lab)

    assert retry['arguments']['path'] == original['path'] + '_resume_0002'
    assert {k: v for k, v in retry['arguments'].items() if k != 'path'} == {
        k: v for k, v in original.items() if k != 'path'}
    assert question['generalization_review']['refit_arguments'] == original
    assert agent.operator_results[:-1] == old_results
    assert retry['previous_operator_result_index'] == old_results[-1]['index']
    assert 'No space left on device' in old_results[-1]['error']
    assert agent.operator_results[-1]['trigger_origin'] == 'explicit_question_resume'
    refit = question['whole_trajectory_refit']
    assert retry['operator_result_index'] == agent.operator_results[-1]['index']
    assert retry['status'] == refit['status'] == 'refit_inadequate_on_validation'
    assert refit['source_model'] == original['model']
    assert refit['split']['training_records'] == [0]
    assert refit['split']['validation_records'] == original['validation_records'] == [1, 2]
    assert refit['validation']['returncode'] == 0 and refit['validation']['mse'] > 1e-6
    assert retry['validation_mse'] == refit['validation']['mse']
    assert retry['validation_returncode'] == 0
    assert retry['train_rows'] == 2 and retry['holdout_rows'] == 4
    assert retry['costs']['compute_cost'] == 6
    assert retry['costs']['python_execution_seconds'] > 0
    for key in ('observation_cost', 'measurement_cost', 'model_calls', 'reported_tokens'):
        assert retry['costs'][key] == 0 and agent.costs[key] == before[key]
    assert agent.observations == observations and lab.actions == lab_actions
    assert agent.provider.requests == [] and agent.model is None
    assert (agent.workspace / 'candidate.py').read_text() == source
    assert (agent.workspace / original['path'] / '_evaluated_model' / 'partial.txt').is_file()
    assert (occupied / 'partial.txt').is_file()
    focus = question['parent_research']
    assert focus['status'] == 'inconclusive'
    assert focus['method_review_result']['validation_mse'] == refit['validation']['mse']
    paid = deepcopy(agent.costs)
    assert resume_generalization_review(agent, question, lab) is None
    assert agent.costs == paid


@pytest.mark.parametrize('status', [
    'refit_supported_on_validation', 'refit_inadequate_on_validation',
    'refit_unavailable', 'confirmation_record_mapping_unavailable',
])
def test_resume_does_not_recompute_measured_or_unavailable_reviews(interrupted_review, status):
    agent, lab, question = interrupted_review
    question['whole_trajectory_refit']['status'] = status
    before, operators = deepcopy(agent.costs), deepcopy(agent.operator_results)
    assert resume_generalization_review(agent, question, lab) is None
    assert agent.costs == before and agent.operator_results == operators
    assert 'refit_retries' not in question['generalization_review']


def test_resume_without_original_refit_arguments_does_nothing(interrupted_review):
    agent, lab, question = interrupted_review
    question['generalization_review'].pop('refit_arguments')
    before, operators = deepcopy(agent.costs), deepcopy(agent.operator_results)
    assert resume_generalization_review(agent, question, lab) is None
    assert agent.costs == before and agent.operator_results == operators


@pytest.mark.parametrize('limit,cost', [
    ('compute_budget', 'compute_cost'), ('execution_seconds', 'python_execution_seconds'),
])
def test_resume_uses_existing_operator_budget_and_retains_failed_retry(interrupted_review, limit, cost):
    agent, lab, question = interrupted_review
    agent.config = replace(agent.config, **{limit: agent.costs[cost]})
    before, operators, paid = deepcopy(agent.costs), len(agent.operator_results), lab.paid
    retry = resume_generalization_review(agent, question, lab)
    assert retry['status'] == question['whole_trajectory_refit']['status'] == 'refit_execution_failed'
    assert 'budget exhausted' in question['whole_trajectory_refit']['error']
    assert len(agent.operator_results) == operators + 1
    assert agent.costs['compute_cost'] == before['compute_cost']
    assert agent.costs['model_calls'] == before['model_calls']
    assert agent.costs['observation_cost'] == before['observation_cost'] and lab.paid == paid
    assert agent.provider.requests == []


def test_runner_explicit_same_question_resume_refits_before_starting_inquiry(
        interrupted_review, tmp_path, monkeypatch):
    old, lab, question = interrupted_review
    old._prepare_program_state()
    (old.workspace / 'research_context.json').write_text(json.dumps(old._request(lab)))
    previous = old.result('model_call_budget_exhausted')
    archive = tmp_path / 'initial_archive.json'
    archive.write_text(json.dumps(old.knowledge))
    case = {'id': 'case', 'confirmation_seed': 1709}
    plan = {'online': {'runtime_snapshot': [], 'arms': ['recursive'], 'model': 'fixture',
        'limits': asdict(old.config)}, 'resume_result': str(old.workspace / 'training.json'),
        'resume_workspace': str(old.workspace), 'resume_question': question['id'],
        'source_study': str(tmp_path), 'initial_archive': str(archive),
        'comparison': 'Same paid fixture', 'next_stage': 'No additional acquisition.'}

    def forbidden(*args, **kwargs):
        raise AssertionError('Host refit resumption must not call an LLM or buy observations')

    def inspect(part, output, *, knowledge, prepare_prefix, agent_factory, **kwargs):
        provider = ScriptedProvider([])
        agent = agent_factory(provider, output / 'recursive', arm='recursive',
            knowledge=knowledge, source_task=False, config=InquiryLLMConfig(**part['limits']))
        current = RestorableConfirmationLab()
        extra = prepare_prefix(agent, current, case)
        selected = next(q for q in agent.questions if q['id'] == question['id'])
        assert selected['whole_trajectory_refit']['status'] == 'refit_inadequate_on_validation'
        assert selected['parent_research']['status'] == 'inconclusive'
        assert agent._inquiry_stack[-1]['question_id'] == question['id']
        assert agent._inquiry_stack[-1]['assignment']['whole_trajectory_refit'] == selected['whole_trajectory_refit']
        assert agent.costs['compute_cost'] == previous['costs']['compute_cost'] + 6
        assert agent.costs['python_execution_seconds'] > previous['costs']['python_execution_seconds']
        assert agent.costs['observation_cost'] == previous['costs']['observation_cost']
        assert agent.costs['model_calls'] == previous['costs']['model_calls']
        assert agent.observations == old.observations and current.actions == []
        assert provider.requests == [] and agent.model is None
        report = json.loads((agent.workspace / 'restoration.json').read_text())
        retry = selected['generalization_review']['refit_retries'][-1]
        assert report['generalization_review_retry'] == retry
        assert report['new_observations'] == 0
        assert report['host_scheduled_resume']['question_id'] == question['id']
        assert extra['prior_research_cost'] == previous['costs']
        return {'status': 'restored_fixture', 'trials': []}

    monkeypatch.setattr(runner, 'run_study', inspect)
    result = runner.run_comparison(plan, tmp_path / 'resumed', provider_factory=forbidden,
                                   previous=previous)
    assert result['status'] == 'restored_fixture'
