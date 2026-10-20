"""Expose actual source support without choosing an author's retrieval action."""

from copy import deepcopy
from dataclasses import replace
import json
import pytest

from hegel_machine.cognition.code_provider_context import load_code_context, prepare_code_context
from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.anonymous_method_learning import AnonymousResearchMethodLearner
from hegel_machine.cognition.interrupted_method_author import recover_interrupted_method_author
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from test_interrupted_method_author import stopped_edit, write_json
from test_python_method_incumbent import ORIGIN, resumed_author, runner


def decision(learner):
    learner._prepare_program_state()
    summary = json.loads(learner.files['method_decision.json'])
    assert summary == json.loads((learner.workspace / 'method_decision.json').read_text())
    return summary


def test_effective_source_entry_points_to_the_last_selector_definition(tmp_path):
    source = ('def select_actions(spec, records, memory):\n'
              '    return {"stop": "shadowed"}\n\n'
              'def select_actions(spec, records, memory):\n'
              '    return {"stop": "effective"}\n')
    learner = AnonymousResearchMethodLearner(None, tmp_path / 'author',
        knowledge={'files': {runner.METHOD_PATH: source}}, cases={'case_0': {}},
        lab_factory=lambda _: AnonymousHistoryLab(kind='memoryless', seed=42),
        config=InquiryLLMConfig(), prepare_trial=None)
    entry = decision(learner)['current_source_entry']
    assert entry['effective_definition'] == {'line': 4, 'end_line': 5}
    assert entry['shadowed_definitions'] == [{'line': 1, 'end_line': 2}]
    assert (learner.workspace / runner.METHOD_PATH).read_text() == source


def test_latest_saved_native_summary_refreshes_without_reopening_old_histories(tmp_path):
    source = 'def select_actions(spec, records, memory): return {"stop": "inspect"}\n'
    learner = AnonymousResearchMethodLearner(None, tmp_path / 'author',
        knowledge={'files': {runner.METHOD_PATH: source}}, cases={'case_0': {}},
        lab_factory=lambda _: AnonymousHistoryLab(kind='memoryless', seed=42),
        config=InquiryLLMConfig(), prepare_trial=None)
    research = {'kind': 'remaining_parameter_ambiguity',
        'next_experiment': {'action': 'query', 'arguments': {'actions': []}}}
    native = {'status': 'candidates_require_evaluation', 'candidates': [],
              'research_decision': research, 'remaining_boundary': {'threshold_interval': [.1, .3]}}
    saved = {'status': 'completed', 'method_results': [{'proposal': {'action': 'run'},
        'feedback': {'native_condition_research': native}}]}
    old_summary = {'research_steps': [], 'latest_native_result': {'status': native['status']}}
    trial = {'case': 'case_0', 'directory': 'trials/trial_239', 'method_source': source,
             'status': 'completed', 'costs': {}, 'fresh_confirmation': {'passed': True},
             'execution_summary': old_summary}
    older = {**deepcopy(trial), 'directory': 'trials/trial_235'}
    for entry, content in [(older, 'Older large trace must not be reopened'),
                           (trial, json.dumps(saved))]:
        path = learner.workspace / entry['directory'] / 'result.json'
        path.parent.mkdir(parents=True)
        path.write_text(content)
    learner.practice_trials.extend([older, trial])
    before = deepcopy(learner.practice_trials)
    refreshed = learner._paid_practice_view(trial)['execution_summary']['latest_native_result']
    assert refreshed['research_decision'] == research
    assert refreshed['remaining_boundary']['threshold_interval'] == [.1, .3]
    assert learner._paid_practice_view(older)['execution_summary'] == old_summary
    assert learner.practice_trials == before
    assert all(cost == 0 for cost in learner.costs.values())


def test_incumbent_failure_does_not_replace_selected_source_research_context(tmp_path):
    source = 'def select_actions(spec, records, memory): return {"stop": "draft"}\n'
    learner = AnonymousResearchMethodLearner(None, tmp_path / 'author',
        knowledge={'files': {runner.METHOD_PATH: source}}, cases={'case_0': {}},
        lab_factory=lambda _: AnonymousHistoryLab(kind='memoryless', seed=42),
        config=InquiryLLMConfig(), prepare_trial=None)
    current = {'case': 'case_0', 'directory': 'trials/trial_167', 'method_source': source,
        'status': 'confirmation_failed', 'costs': {'observation_cost': 216},
        'selected_prior_predictor': False, 'fresh_confirmation': {'passed': False, 'mse': .48},
        'execution_summary': {'research_steps': [],
            'latest_evaluation': {'model': 'native/candidate.py', 'mse': 1e-20},
            'latest_native_result': {'proposed_experiment_preview': [
                {'purpose': 'conditional_boundary', 'observation_cost': 1}]}}}
    incumbent = {**deepcopy(current), 'directory': 'trials/trial_168',
        'method_source': source + '# older scalar program\n', 'status': 'method_no_progress',
        'fresh_confirmation': None, 'execution_summary': {'research_steps': []},
        'no_progress': {'kind': 'repeated_execution_error', 'error': 'KeyError: u'}}
    learner.practice_trials.extend([current, incumbent])
    learner.events.append({'action': 'finish', 'arguments': {}, 'feedback': {
        'error': 'method publication practice failed', 'publication_practices': [current],
        'incumbent_comparison': {'old_trial': 'trials/trial_047',
            'current_failure_case': 'case_0', 'selected': False,
            'status': 'incumbent_practice_failed', 'practices': [incumbent]}}})
    costs, ledger = deepcopy(learner.costs), deepcopy(learner.practice_trials)
    view = decision(learner)
    assert view['latest_practice']['directory'] == 'trials/trial_168'
    assert view['latest_practice']['source_matches_current'] is False
    assert view['current_source_latest_failure']['directory'] == 'trials/trial_167'
    assert view['current_source_latest_evaluation']['directory'] == 'trials/trial_167'
    assert view['current_source_latest_failure']['source_matches_current'] is True
    assert view['current_source_latest_no_progress'] is None
    old_view = view['latest_incumbent_comparison']['practices'][0]
    assert old_view['directory'] == 'trials/trial_168' and old_view['source_matches_current'] is False
    context = prepare_code_context(learner.workspace, {
        'task_interface': learner.common_interface, 'files': dict(learner.files),
        'events': learner.events, 'observations': [], 'session_knowledge': learner.knowledge},
        schema={}, instructions='test')
    assert context['method_decision']['current_source_latest_failure'] == view['current_source_latest_failure']
    assert context['method_decision']['current_source_latest_failure']['execution_summary'][
        'latest_native_result']['proposed_experiment_preview'][0]['purpose'] == 'conditional_boundary'
    assert learner.costs == costs and learner.practice_trials == ledger and learner.published_source is None

    # A later confirmation of that exact source resolves the earlier failure;
    # restoring the older source redirects the focus to its own actual failure.
    passed = {**deepcopy(current), 'directory': 'trials/trial_169', 'status': 'completed',
              'fresh_confirmation': {'passed': True, 'mse': 1e-20}}
    learner.practice_trials.append(passed)
    assert decision(learner)['current_source_latest_failure'] is None
    (learner.workspace / runner.METHOD_PATH).write_text(incumbent['method_source'])
    restored = decision(learner)
    assert restored['current_source_latest_failure']['directory'] == 'trials/trial_168'
    assert restored['current_source_latest_evaluation'] is None


@pytest.mark.parametrize('already_has_predictor', [False, True, 'without_source'])
def test_restored_submitted_predictor_refreshes_selected_summary_only(
        tmp_path, monkeypatch, already_has_predictor):
    source = 'def select_actions(spec, records, memory): return {"stop": "draft"}\n'
    old = tmp_path / 'old'
    (old / 'trials/trial_001').mkdir(parents=True)
    result_file = old / 'trials/trial_001/result.json'
    result_file.write_text('{"status":"confirmation_failed"}')
    old_summary = {'research_steps': [], 'submission_count': 1,
                   'latest_evaluation': {'model': 'native/candidate.py', 'mse': 1e-20}}
    if already_has_predictor:
        old_summary['latest_submitted_predictor'] = {
            'model': 'native/candidate.py', 'paid_rows': 121}
    if already_has_predictor == 'without_source':
        old_summary['latest_submitted_predictor'].update(
            model_file='trials/trial_001/_submitted_models/model.py', submission_evidence={})
    selected = {'case': 'case_0', 'directory': 'trials/trial_001', 'method_source': source,
        'status': 'confirmation_failed', 'costs': {}, 'selected_prior_predictor': False,
        'fresh_confirmation': {'passed': False}, 'execution_summary': old_summary}
    unrelated = {**deepcopy(selected), 'directory': 'trials/trial_000',
                 'method_source': source + '# old source\n'}
    (old / unrelated['directory']).mkdir()
    (old / unrelated['directory'] / 'result.json').write_text('unread older result')
    previous = {'workspace': str(old), 'status': 'model_call_budget_exhausted',
        'outgoing_knowledge': {'files': {runner.METHOD_PATH: source}},
        'practice_trials': [unrelated, selected], 'events': []}
    learner = AnonymousResearchMethodLearner(None, tmp_path / 'restored',
        knowledge={'files': {runner.METHOD_PATH: source}}, cases={'case_0': {}},
        lab_factory=lambda _: AnonymousHistoryLab(kind='memoryless', seed=42),
        config=InquiryLLMConfig(), prepare_trial=None)
    learner.restore_practice(previous)
    refreshed = []

    def summarize(result, directory):
        refreshed.append(directory)
        return {**old_summary, 'latest_submitted_predictor': {
            'model': 'native/candidate.py', 'description': 'five-step measured candidate',
            'source_preview': {'preview': 'from sklearn.ensemble import ExtraTreesRegressor'},
            'submission_evidence': {'paid_before_submission': {
                'row_count': 121, 'trajectory_row_counts': [64, 8, 12, 12, 12, 12, 1]}}}}

    monkeypatch.setattr('hegel_machine.cognition.python_method_learning.practice_execution_summary', summarize)
    first, second = decision(learner), decision(learner)
    assert first == second and refreshed == ['trials/trial_001']
    assert first['current_source_latest_failure']['execution_summary']['latest_submitted_predictor'][
        'model'] == 'native/candidate.py'
    assert first['current_source_latest_failure']['execution_summary']['latest_submitted_predictor'][
        'submission_evidence']['paid_before_submission']['trajectory_row_counts'][0] == 64
    assert 'ExtraTreesRegressor' in first['latest_practice']['execution_summary'][
        'latest_submitted_predictor']['source_preview']['preview']
    stored_candidates = json.loads((learner.workspace / 'method_candidates.json').read_text())
    stored_latest = next(trial for entry in stored_candidates for trial in entry['trials']
                         if trial['directory'] == 'trials/trial_001')
    assert 'ExtraTreesRegressor' in stored_latest['execution_summary']['latest_submitted_predictor'][
        'source_preview']['preview']
    assert len(learner.practice_trials) == 2
    assert learner.practice_trials[-1]['execution_summary'] == old_summary
    assert json.loads(result_file.read_text()) == {'status': 'confirmation_failed'}


def test_real_paid_candidates_are_directly_visible_without_preselecting_a_source(tmp_path):
    learner, previous = resumed_author(tmp_path)
    source, costs = learner.files[runner.METHOD_PATH], deepcopy(learner.costs)
    summary = decision(learner)
    assert summary['current_source_coverage']['missing_cases'] == ['case_4']
    assert summary['latest_publication_failure']['case'] == 'case_4'
    entries = summary['candidates']
    assert [entry['trial'] for entry in entries] == [entry['trial'] for entry in
        json.loads(learner.files['method_candidates.json'])]
    old = next(entry for entry in entries if entry['trial'] == 'trials/trial_003')
    assert old['supported_cases'] == ['case_5'] and old['failed_cases'] == []
    assert old['missing_cases'] == [f'case_{i}' for i in range(5)]
    current = next(entry for entry in entries if entry['current'])
    assert current['supported_cases'] == ['case_0', 'case_1', 'case_2', 'case_3', 'case_5']
    assert current['failed_cases'] == ['case_4']
    assert learner.files[runner.METHOD_PATH] == source and learner.costs == costs
    assert learner.published_source is None and len(learner.practice_trials) == len(previous['practice_trials'])

    request = {'task_interface': learner.common_interface, 'files': dict(learner.files),
               'events': [], 'observations': [], 'session_knowledge': learner.knowledge}
    original = deepcopy(request)
    context = prepare_code_context(learner.workspace, request, schema={}, instructions='test')
    assert next(iter(context)) == 'method_decision'
    assert context['method_decision'] == summary
    for name in ('method_decision.json', 'method_candidates.json', 'resume_context.json'):
        assert context['files'][name] == {'full_content_file': name}
        assert (learner.workspace / name).is_file()
    assert context['files'][runner.METHOD_PATH] == source
    assert load_code_context(learner.workspace) == original
    assert request == original


def test_removed_supported_memory_does_not_acquire_support_from_other_failed_sources(tmp_path):
    original, previous = resumed_author(tmp_path / 'original')
    supported_sources = {trial['method_source'] for trial in original.practice_trials
        if any(item['status'] == 'completed' and
               (item.get('fresh_confirmation') or {}).get('passed') is True
               for item in original._latest_source_practices(trial['method_source']).values())}
    negative = deepcopy(previous)
    negative['practice_trials'] = [trial for trial in previous['practice_trials']
                                  if trial['method_source'] not in supported_sources]
    assert negative['practice_trials']
    plan = json.loads((ORIGIN / 'protocol.json').read_text())
    sources = json.loads((ORIGIN / 'existing_source_outcomes.json').read_text())['sources']
    learner = runner.prepare_author(None, tmp_path / 'unsupported', plan, plan['author_replicas'][2],
        sources, (ORIGIN / 'unchanged_method.py').read_text(), ORIGIN / 'paid_source_counterexample.json')
    learner.restore_practice(negative)
    summary = decision(learner)
    assert summary['candidates'] and all(not entry['supported_cases'] for entry in summary['candidates'])
    assert all(entry['failed_cases'] for entry in summary['candidates'])
    assert not (learner.workspace / 'trials/trial_003').exists()
    assert summary['current_source_coverage']['confirmed_cases'] == []
    assert learner._incumbent_trial(learner.files[runner.METHOD_PATH]) is None
    assert learner.published_source is None and all(cost == 0 for cost in learner.costs.values())


def test_explicit_summary_restore_action_keeps_the_original_publication_requirements(tmp_path):
    learner, _ = resumed_author(tmp_path)
    old = next(entry for entry in decision(learner)['candidates']
               if entry['trial'] == 'trials/trial_003')
    action = old['restore_action']
    costs = deepcopy(learner.costs)
    feedback = learner._execute(action['action'], action['arguments'], learner.interface)
    assert feedback['restored_method'] == old['trial']
    assert 'edit_practice' not in feedback and learner.costs == costs
    summary = decision(learner)
    assert next(entry for entry in summary['candidates'] if entry['current'])['trial'] == old['trial']
    assert summary['current_source_coverage']['confirmed_cases'] == ['case_5']
    assert learner.published_source is None
    learner.config = replace(learner.config, observation_budget=0)
    result = learner._execute('finish', {'path': runner.METHOD_PATH}, learner.interface)
    assert 'published' not in result and learner.published_source is None
    assert result['unconfirmed_cases'] == [f'case_{i}' for i in range(5)]
    assert learner.costs == costs


def test_resumed_no_progress_is_inline_and_later_practice_does_not_hide_its_evidence(
        tmp_path, stopped_edit):
    old, copied, _, _, source = stopped_edit
    no_progress = {'kind': 'repeated_experiment_comparison', 'comparison_indices': [0, 1],
        'evidence_cutoff': {'records': 2, 'rows': 72}, 'diagnosis': 'single_supported_candidate',
        'reason': 'The same forecasts were compared twice without new paid evidence.'}
    write_json(old / 'runtime_stop.json', {
        'reason': 'operator_stopped_repeated_no_progress', 'interrupted_python_seconds': 4.,
        'no_progress': no_progress})
    restored = recover_interrupted_method_author(old, copied)
    learner = AnonymousResearchMethodLearner(None, tmp_path / 'resumed',
        knowledge={'files': {runner.METHOD_PATH: source}}, cases={'case_0': {}, 'case_1': {}},
        lab_factory=lambda _: AnonymousHistoryLab(kind='memoryless', seed=42),
        config=InquiryLLMConfig(), prepare_trial=None)
    learner.restore_practice(restored)
    costs = deepcopy(learner.costs)

    def provider_view():
        summary = decision(learner)
        context = prepare_code_context(learner.workspace, {
            'task_interface': learner.common_interface, 'files': dict(learner.files),
            'events': learner.events, 'observations': [], 'session_knowledge': learner.knowledge},
            schema={}, instructions='test')
        assert context['files']['resume_context.json'] == {'full_content_file': 'resume_context.json'}
        assert context['method_decision'] == summary
        return context['method_decision']

    view = provider_view()
    assert view['latest_practice']['status'] == 'method_no_progress'
    assert view['latest_practice']['no_progress'] == no_progress
    assert view['current_source_latest_no_progress']['directory'] == 'trials/trial_001'
    assert 'steps' not in view['latest_practice'] and 'method_source' not in view['latest_practice']

    # A later operator interruption is a new outcome, not a refutation or a
    # repair of the identical source's already recorded stalled comparison.
    interrupted = {**deepcopy(learner.practice_trials[-1]),
                   'directory': 'trials/trial_002', 'status': 'runtime_interrupted'}
    interrupted.pop('no_progress')
    learner.practice_trials.append(interrupted)
    learner.events.append({'action': 'run', 'arguments': {'operator': 'practice', 'case': 'case_1'},
        'feedback': {key: value for key, value in interrupted.items() if key != 'method_source'}})
    view = provider_view()
    assert view['latest_practice']['directory'] == 'trials/trial_002'
    assert view['latest_practice']['status'] == 'runtime_interrupted'
    assert 'no_progress' not in view['latest_practice']
    assert view['current_source_latest_no_progress']['no_progress'] == no_progress
    assert view['current_source_latest_no_progress']['directory'] == 'trials/trial_001'

    (learner.workspace / runner.METHOD_PATH).write_text(source + '\n# New authored draft.\n')
    assert provider_view()['current_source_latest_no_progress'] is None
    assert learner.costs == costs and learner.model is None
