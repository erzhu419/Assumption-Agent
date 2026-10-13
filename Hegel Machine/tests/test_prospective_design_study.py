"""Prospective-design study wiring; scripted publications are not scientific evidence."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.anonymous_method_learning import SingleSubmissionMethodAgent
from hegel_machine.cognition.code_provider_context import prepare_code_context
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts import run_anonymous_method_learning as learning
from scripts import run_anonymous_method_revision as revision
from scripts.experimental_design_author import ExperimentalDesignMethodLearner
from scripts.experimental_design_control import (
    ExperimentalDesignMethodAgent, RandomExperimentMethodAgent)
from test_anonymous_method_learning import action, memoryless_source, read_json, write_json
from test_recurrent_method_revision import SOURCE, revision_sources


@pytest.fixture
def acquisition_source(revision_sources):
    source, repairs = revision_sources
    repaired = revision.protocol(source, repair_study=repairs)
    write_json(source / 'protocol.json', repaired)
    write_json(source / 'summary.json', {'prior_author_phases': []})
    acquired = revision.protocol(source, design_experiments=True)
    write_json(source / 'protocol.json', acquired)
    previous = read_json(source / 'author/training.json')
    previous['practice_trials'] = []
    for index in range(36):
        case = list(acquired['practice_cases'])[index % 6]
        directory = f'trials/trial_{index:03d}'
        path = source / 'author' / directory / METHOD_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(SOURCE)
        previous['practice_trials'].append({
            'case': case, 'directory': directory, 'method_source': SOURCE,
            'status': 'completed', 'costs': {'observation_cost': 168},
            'selected_prior_predictor': index % 6 == 0,
            'fresh_confirmation': {'passed': True, 'mse': 0., 'rows': 96},
            'experimental_design': {'design_supported': True},
        })
    previous['costs']['observation_cost'] = 36 * 168
    write_json(source / 'author/training.json', previous)
    return source


def test_protocol_requires_new_case_support_and_fixes_matched_research_allowance(acquisition_source):
    old = read_json(acquisition_source / 'protocol.json')
    plan = revision.protocol(acquisition_source, prospective_design=True, timeout_seconds=610.)
    assert plan['prospective_design'] and plan['design_experiments']
    assert list(plan['practice_cases']) == [f'design_case_{i:03d}' for i in range(6)]
    assert list(plan['practice_cases'].values()) == list(old['practice_cases'].values())
    assert set(plan['practice_cases']).isdisjoint(old['practice_cases'])
    assert plan['previous_practice_cases'] == {
        f'design_case_{i:03d}': name for i, name in enumerate(old['practice_cases'])}
    assert plan['transfer_cases'][:6] == old['transfer_cases'][:6]
    fresh = plan['transfer_cases'][6:]
    assert {case['apparatus_seed'] for case in fresh} == {101231031, 101232040}
    assert {(case['parameter_pair'], case['kind']) for case in fresh} == {
        (block, kind) for block in (0, 1) for kind in ('recurrent', 'finite2')}
    historical_seeds = {case['apparatus_seed'] for case in
                        [*old['practice_cases'].values(), *old['transfer_cases']]}
    assert {case['apparatus_seed'] for case in fresh}.isdisjoint(historical_seeds)
    for case in fresh:
        assert case['prefix_seed'] == case['apparatus_seed'] + 100003
        assert case['confirmation_seed'] == case['apparatus_seed'] + 200003
        assert case['evaluation_seed'] == case['apparatus_seed'] + 300007
    assert plan['prefix_rows'] == 72
    assert plan['limits']['confirmation_rows'] == 96
    assert plan['limits']['observation_budget'] == 72 + 192 + 96
    assert plan['method_steps'] == 32
    assert plan['provider']['timeout_seconds'] == 610.
    with pytest.raises(ValueError, match='one published method'):
        revision.protocol(acquisition_source, prospective_design=True, design_experiments=True)


def test_all_36_restored_practices_remain_experience_without_new_design_support(
        tmp_path, acquisition_source, monkeypatch):
    plan = revision.protocol(acquisition_source, prospective_design=True)
    plan['runtime_snapshot'] = []
    plan['author_limits']['max_model_calls'] = 1
    output = tmp_path / 'restore'

    def forbidden(*args, **kwargs):
        raise AssertionError('Do not replay old research or transfer an unpublished design.')

    monkeypatch.setattr(revision, 'reproduce_counterexample', forbidden)
    monkeypatch.setattr(revision, 'attach_repair_archive', forbidden)
    monkeypatch.setattr(revision, 'frozen_transfers', forbidden)

    class InspectAuthor:
        def next_action(self, request):
            context = read_json(output / 'author/resume_context.json')
            assert len(context['previous_practice']) == 36
            assert context['current_source_coverage']['confirmed_cases'] == []
            assert context['current_source_coverage']['missing_cases'] == sorted(plan['practice_cases'])
            assert request['files'][METHOD_PATH] == SOURCE
            assert request['costs']['observation_cost'] == 0
            assert 'prospective experiment-design method' in request['task_interface']
            return action('run', {'code': "print('Retain inherited experience.')"})

    summary = revision.run_revision(plan, output, provider_factory=lambda _: InspectAuthor())
    result = read_json(output / 'author/training.json')
    assert len(result['practice_trials']) == summary['restored_practice_trials'] == 36
    assert summary['new_practice_trials'] == 0
    assert summary['author_costs']['observation_cost'] == 0
    assert not summary['method_published']
    assert summary['unavailable_target_episodes'] == 30
    assert set(summary['frozen']) == {'original', 'revised', 'random_choice'}
    assert all(not records for records in summary['frozen'].values())
    assert read_json(output / 'author/experiment_design_feedback.json')['design_supported_cases'] == []


def test_three_arms_share_protocol_and_new_source_before_any_terminal_scoring(
        tmp_path, acquisition_source, monkeypatch):
    plan = revision.protocol(acquisition_source, prospective_design=True)
    plan['runtime_snapshot'] = []
    knowledge = read_json(acquisition_source / 'initial_archive.json')
    revised = SOURCE.replace('fixture source', 'prospective fixture source')
    order, memories = [], {}

    def publish(author, *, on_checkpoint):
        payload = author.prepare_trial(plan['practice_cases']['design_case_000'],
            author.workspace / 'fixture_practice', revised, InquiryLLMConfig(**plan['limits']))
        assert type(payload['agent']) is ExperimentalDesignMethodAgent
        assert isinstance(payload['agent'], SingleSubmissionMethodAgent)
        assert payload['agent'].costs['observation_cost'] == 72
        assert payload['agent'].config.observation_budget == 360
        author.model, author.published_source = METHOD_PATH, revised
        return author.result('completed')

    def targets(part, branch, memory, sources, method_source, *, agent_type):
        assert part['transfer_cases'] == plan['transfer_cases']
        assert agent_type is (RandomExperimentMethodAgent if branch.name == 'random_choice'
                              else ExperimentalDesignMethodAgent)
        assert issubclass(agent_type, SingleSubmissionMethodAgent)
        assert part['limits']['observation_budget'] == 360
        assert part['method_steps'] == 32
        assert method_source == (SOURCE if branch.name == 'original' else revised)
        assert memory['files'][METHOD_PATH] == method_source
        assert memory['source_catalog'] == knowledge['source_catalog']
        assert all(memory['files'][name] == text for name, text in knowledge['files'].items()
                   if name not in {METHOD_PATH, 'shared/knowledge.json'})
        memories[branch.name] = deepcopy(memory)
        order.append(('research', branch.name))
        return [(case, {'branch': branch.name, 'costs': {'observation_cost': 360, 'compute_cost': 5},
                        'events': [], 'method_results': []}) for case in part['transfer_cases']]

    def score(part, results):
        assert order[:3] == [('research', 'original'), ('research', 'revised'),
                             ('research', 'random_choice')]
        order.append(('score', results[0][1]['branch']))
        return [{'case': case['id'], 'kind': case['kind'], 'status': 'completed',
                 'costs': result['costs'], 'terminal': {'passed': True}}
                for case, result in results]

    monkeypatch.setattr(ExperimentalDesignMethodLearner, 'run', publish)
    monkeypatch.setattr(revision, 'frozen_transfers', targets)
    monkeypatch.setattr(revision, 'score_frozen', score)
    summary = revision.run_revision(plan, tmp_path / 'three_arms', provider_factory=lambda _: None)
    assert summary['method_published'] and summary['source_changed']
    assert memories['revised'] == memories['random_choice']
    assert summary['unavailable_target_episodes'] == 0
    assert len(summary['choice_pairs']) == 10
    assert all(pair['observation_delta'] == pair['compute_delta'] == 0
               for pair in summary['choice_pairs'])
    for records in summary['frozen'].values():
        assert len(records) == 10
        assert all(record['research_protocol'] == 'single_submission' for record in records)
    assert [pair['group'] for pair in summary['pairs']] == ['old_target'] * 6 + ['new_parameter'] * 4


def test_unresolved_design_retains_predictive_publication_without_target_fallback(
        tmp_path, acquisition_source, monkeypatch):
    plan = revision.protocol(acquisition_source, prospective_design=True)
    plan['runtime_snapshot'] = []

    def prediction_only(author, *, on_checkpoint):
        for index, case in enumerate(plan['practice_cases']):
            author.practice_trials.append({
                'case': case, 'directory': f'trials/new_{index:03d}', 'method_source': SOURCE,
                'status': 'completed', 'costs': {'observation_cost': 360},
                'selected_prior_predictor': index == 0,
                'fresh_confirmation': {'passed': True, 'mse': 0., 'rows': 96},
                'experimental_design': {'prediction_supported': True, 'design_supported': False},
            })
        feedback = author._execute('finish', {'path': METHOD_PATH, 'lesson': 'Predictive fixture.'},
                                   author.interface)
        assert feedback['task_status'] == 'prediction_supported_design_unresolved'
        assert not feedback.get('published')
        assert author.published_source is None
        return author.result('model_call_budget_exhausted')

    def forbidden(*args, **kwargs):
        raise AssertionError('Prediction alone must not trigger transfer of a draft design method.')

    monkeypatch.setattr(ExperimentalDesignMethodLearner, 'run', prediction_only)
    monkeypatch.setattr(revision, 'frozen_transfers', forbidden)
    output = tmp_path / 'unresolved'
    summary = revision.run_revision(plan, output, provider_factory=lambda _: None)
    assert not summary['method_published']
    assert summary['unavailable_target_episodes'] == 30
    assert summary['predictive_publication']
    assert summary['predictive_publication']['confirmed_cases'] == sorted(plan['practice_cases'])
    assert not read_json(output / 'published_method.json')['published']
    assert all(not records for records in summary['frozen'].values())


@pytest.mark.parametrize('agent_type', [ExperimentalDesignMethodAgent, RandomExperimentMethodAgent])
def test_frozen_factory_uses_the_requested_control_without_live_calls(
        tmp_path, acquisition_source, monkeypatch, agent_type):
    plan = revision.protocol(acquisition_source, prospective_design=True)
    plan['transfer_cases'] = plan['transfer_cases'][:1]
    knowledge = read_json(acquisition_source / 'initial_archive.json')
    sources = read_json(acquisition_source / 'source_outcomes.json')
    executed = []

    def execute(agent, lab, *, max_steps, on_checkpoint):
        executed.append(type(agent))
        assert max_steps == 32
        assert agent.frozen_method_source == SOURCE
        assert agent.costs['observation_cost'] == 72
        assert agent.config.max_model_calls == agent.config.token_budget == 0
        assert agent.config.observation_budget == 360
        return {'status': 'method_stopped', 'costs': deepcopy(agent.costs)}

    monkeypatch.setattr(agent_type, 'execute_method', execute)
    results = learning.frozen_transfers(plan, tmp_path / 'factory', knowledge, sources,
        SOURCE, agent_type=agent_type)
    assert executed == [agent_type]
    assert len(results) == 1


@pytest.fixture
def interrupted_design(tmp_path, acquisition_source, monkeypatch):
    # Historical method learning must remain separate from this new account.
    inherited = read_json(acquisition_source / 'author/training.json')
    inherited['costs']['model_calls'] = 181
    write_json(acquisition_source / 'author/training.json', inherited)
    plan = revision.protocol(acquisition_source, prospective_design=True, timeout_seconds=600.)
    plan['runtime_snapshot'] = []
    failed = tmp_path / 'provider_failed'
    unfinished = SOURCE.replace('fixture source', 'unfinished prospective draft')

    def no_targets(*args, **kwargs):
        raise AssertionError('An unfinished design must not run target research.')

    monkeypatch.setattr(revision, 'frozen_transfers', no_targets)

    class InterruptedAuthor:
        calls = 0

        def next_action(self, request):
            self.calls += 1
            if self.calls == 1:
                decision = action('write', {'path': METHOD_PATH, 'code': unfinished})
                decision['usage']['total_tokens'] = 31537
                return decision
            return {'action': 'provider_error',
                    'error': {'provider': 'Fixture provider rejected its next input.'},
                    'usage': {'total_tokens': None}}

    summary = revision.run_revision(plan, failed, provider_factory=lambda _: InterruptedAuthor())
    previous_path = failed / 'author/training.json'
    previous = read_json(previous_path)
    assert previous['status'] == 'provider_failed'
    assert previous['costs']['model_calls'] == 2
    assert previous['costs']['observation_cost'] == 72
    assert len(previous['practice_trials']) == 37
    assert previous['practice_trials'][-1]['status'] == 'method_stopped'
    assert previous['outgoing_knowledge']['files'][METHOD_PATH] == unfinished
    assert not (failed / 'baseline_reproduction.json').exists()
    return acquisition_source, plan, previous_path, previous, summary, unfinished


def test_provider_resume_keeps_protocol_and_debits_only_this_account(interrupted_design):
    source, old, path, previous, summary, _ = interrupted_design
    resumed = revision.protocol(source, continue_author=path, timeout_seconds=600.)
    for key in ('study', 'practice_cases', 'previous_practice_cases', 'transfer_cases',
                'old_target_ids', 'limits', 'prefix_rows', 'evaluation', 'comparison',
                'practice_protocol', 'outcome_protocol', 'method_steps', 'scope', 'provider'):
        assert resumed[key] == old[key]
    assert resumed['prospective_design'] and resumed['design_experiments']
    assert resumed['author_total_limits'] == old['author_limits']
    assert resumed['author_limits']['max_model_calls'] == 94
    assert resumed['author_limits']['observation_budget'] == 36_000 - 72
    assert resumed['author_limits']['token_budget'] == 20_000_000 - 31537
    assert resumed['author_limits']['compute_budget'] == (
        old['author_limits']['compute_budget'] - previous['costs']['compute_cost'])
    assert resumed['author_limits']['execution_seconds'] == (
        old['author_limits']['execution_seconds'] - previous['costs']['python_execution_seconds'])
    assert resumed['continuation_usage']['known_tokens_charged'] == 31537
    assert resumed['continuation_usage']['missing_usage_calls'] == 1
    assert resumed['prior_author_phases'][:-1] == summary['prior_author_phases']
    assert resumed['prior_author_phases'][-1]['result'] == str(path)
    assert resumed['prior_author_phases'][-1]['costs'] == previous['costs']
    assert sum(phase['costs']['model_calls'] for phase in summary['prior_author_phases']) >= 181


def test_runtime_resume_uses_saved_account_without_a_completed_summary(
        tmp_path, interrupted_design, monkeypatch):
    source, old, path, previous, summary, unfinished = interrupted_design
    previous = deepcopy(previous)
    previous['status'] = 'runtime_interrupted'
    previous['costs']['model_calls'] = 1
    previous['events'] = previous['events'][:1]
    recovered_path = path.with_name('runtime_interrupted.json')
    write_json(recovered_path, previous)
    (path.parent.parent / 'summary.json').unlink()
    resumed = revision.protocol(source, continue_author=recovered_path)
    assert resumed['author_limits']['max_model_calls'] == old['author_limits']['max_model_calls'] - 1
    assert resumed['author_limits']['token_budget'] == old['author_limits']['token_budget'] - 31537
    assert resumed['author_limits']['observation_budget'] == old['author_limits']['observation_budget'] - 72
    assert resumed['prior_author_phases'][:-1] == old['prior_author_phases']
    assert resumed['prior_author_phases'][-1]['status'] == 'runtime_interrupted'
    assert resumed['continuation_usage']['missing_usage_calls'] == 0

    def no_repurchase(*args, **kwargs):
        raise AssertionError('Restore paid practice; do not execute the interrupted edit again.')

    monkeypatch.setattr(ExperimentalDesignMethodLearner, '_practice', no_repurchase)
    resumed['author_limits']['max_model_calls'] = 1
    output = tmp_path / 'runtime_continued'

    class InspectingAuthor:
        def next_action(self, request):
            context = read_json(output / 'author/resume_context.json')
            assert context['previous_status'] == 'runtime_interrupted'
            assert len(context['previous_practice']) == 37
            assert request['files'][METHOD_PATH] == unfinished
            assert request['costs']['observation_cost'] == 0
            return action('run', {'code': "print('Inspect retained practice.')"})

    result = revision.run_revision(resumed, output, provider_factory=lambda _: InspectingAuthor())
    assert result['new_practice_trials'] == 0
    assert result['author_costs']['observation_cost'] == 0
    assert result['old_author_costs']['observation_cost'] == summary['old_author_costs']['observation_cost'] + 72


def test_edit_practice_keeps_actual_feedback_until_provider_context_is_saved(
        tmp_path, interrupted_design):
    _, _, path, previous, _, _ = interrupted_design
    event = previous['events'][0]
    assert event['action'] == 'write'
    assert 'error' not in event['feedback']
    practice = event['feedback']['edit_practice']
    trial = previous['practice_trials'][-1]
    result = read_json(path.parent / trial['directory'] / 'result.json')
    assert practice['directory'] == trial['directory']
    assert practice['status'] == result['status'] == 'method_stopped'
    assert practice['costs'] == result['costs']
    assert practice['steps'] == result['method_results']
    assert practice['steps'] and practice['steps'][0]['proposal']['stop'] == 'unfinished prospective draft'
    assert 'full_trace_file' not in practice

    workspace = tmp_path / 'provider_context'
    request = {'events': [deepcopy(event)]}
    context = prepare_code_context(workspace, request, schema={}, instructions='Fixture context.')
    stored = read_json(workspace / 'research_context.json')
    assert stored == request
    assert stored['events'][0]['feedback']['edit_practice'] == practice
    compact = context['events'][0]['feedback']['edit_practice']
    assert compact['full_trace_file'] == f"{trial['directory']}/result.json"
    assert read_json(path.parent / compact['full_trace_file'])['method_results'] == practice['steps']


def test_provider_resume_restores_unfinished_source_and_paid_failure_without_repurchase(
        tmp_path, interrupted_design, monkeypatch):
    source, _, path, previous, first, unfinished = interrupted_design
    resumed_plan = revision.protocol(source, continue_author=path, timeout_seconds=600.)
    continued = tmp_path / 'continued'
    failed_trial = previous['practice_trials'][-1]
    old_trial = path.parent / failed_trial['directory']
    old_result = (old_trial / 'result.json').read_bytes()
    inspected = []

    def no_repurchase(*args, **kwargs):
        raise AssertionError('Restoration and unchanged inspection must not repurchase practice.')

    monkeypatch.setattr(revision, 'reproduce_counterexample', no_repurchase)
    monkeypatch.setattr(ExperimentalDesignMethodLearner, '_practice', no_repurchase)

    class InspectingAuthor:
        def __init__(self, workspace):
            self.workspace = workspace

        def next_action(self, request):
            if inspected:
                return {'action': 'provider_error',
                        'error': {'provider': 'Fixture stops after inspecting its restored state.'},
                        'usage': {'total_tokens': None}}
            inspected.append(True)
            assert request['files'][METHOD_PATH] == unfinished
            assert request['costs']['observation_cost'] == 0
            restored = read_json(self.workspace / 'resume_context.json')
            assert restored['previous_status'] == 'provider_failed'
            assert len(restored['previous_practice']) == 37
            assert restored['previous_practice'][-1]['directory'] == failed_trial['directory']
            assert restored['current_source_coverage']['confirmed_cases'] == []
            assert (self.workspace / failed_trial['directory'] / METHOD_PATH).read_text() == unfinished
            assert (self.workspace / failed_trial['directory'] / 'result.json').read_bytes() == old_result
            return action('run', {'code': "print('Inspect the retained paid failure.')"})

    summary = revision.run_revision(resumed_plan, continued, provider_factory=InspectingAuthor)
    latest = read_json(continued / 'author/training.json')
    assert inspected == [True]
    assert latest['status'] == 'provider_failed'
    assert latest['outgoing_knowledge']['files'][METHOD_PATH] == unfinished
    assert latest['practice_trials'] == previous['practice_trials']
    assert summary['restored_practice_trials'] == 37
    assert summary['new_practice_trials'] == 0
    assert summary['author_costs']['observation_cost'] == 0
    assert summary['old_author_costs']['observation_cost'] == (
        first['old_author_costs']['observation_cost'] + 72)
    assert summary['unavailable_target_episodes'] == 30
    assert not summary['method_published']
    assert not (continued / 'baseline_reproduction.json').exists()
    assert all(not values for values in summary['frozen'].values())
    assert (old_trial / 'result.json').read_bytes() == old_result
