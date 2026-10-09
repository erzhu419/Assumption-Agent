"""Expose actual source support without choosing an author's retrieval action."""

from copy import deepcopy
from dataclasses import replace
import json

from hegel_machine.cognition.code_provider_context import prepare_code_context
from test_python_method_incumbent import ORIGIN, resumed_author, runner


def decision(learner):
    learner._prepare_program_state()
    summary = json.loads(learner.files['method_decision.json'])
    assert summary == json.loads((learner.workspace / 'method_decision.json').read_text())
    return summary


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
    assert json.loads((learner.workspace / 'research_context.json').read_text()) == original
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
