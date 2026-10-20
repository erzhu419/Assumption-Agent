"""Retained program choices keep paid cost tradeoffs without selecting a solver."""

from copy import deepcopy
import json
from pathlib import Path

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.anonymous_method_learning import AnonymousResearchMethodLearner
from hegel_machine.cognition.code_provider_context import prepare_code_context
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from hegel_machine.cognition.temporal_paired_practice import restore_paid_practice_summaries


OLD = 'def select_actions(spec, records, memory): return {"stop": "old fixture"}\n'
NEW = 'def select_actions(spec, records, memory): return {"stop": "new fixture"}\n'


def author(path):
    return AnonymousResearchMethodLearner(None, path, knowledge={'files': {METHOD_PATH: NEW}},
        cases={'case_a': {}, 'case_b': {}},
        lab_factory=lambda _: AnonymousHistoryLab(kind='memoryless', seed=42),
        config=InquiryLLMConfig(), prepare_trial=None)


def trial(index, source, case, observations, seconds, queries, passed=True):
    directory = f'trials/trial_{index:03d}'
    confirmation = {'passed': passed, 'rows': 96, 'mse': 0. if passed else 1.}
    return {'directory': directory, 'case': case, 'method_source': source,
        'status': 'completed' if passed else 'confirmation_failed',
        'selected_prior_predictor': case == 'case_a', 'fresh_confirmation': confirmation,
        'costs': {'observation_cost': observations, 'python_execution_seconds': seconds},
        'execution_summary': {'research_steps': [], 'action_counts': {'query': queries, 'model': 1},
            'step_count': queries + 1, 'submission_count': 1, 'confirmation_count': 1,
            'latest_submitted_predictor': {'model_file': f'{directory}/model.py'},
            'latest_confirmation': confirmation, 'result_file': f'{directory}/result.json'}}


def decision(learner):
    learner._prepare_program_state()
    return json.loads(learner.files['method_decision.json'])


def test_actual_provider_context_preserves_old_and_new_cost_tradeoffs(tmp_path):
    learner = author(tmp_path / 'author')
    learner.practice_trials = [trial(0, OLD, 'case_a', 552, 100., 3),
                              trial(1, OLD, 'case_b', 2120, 3279., 16, False)]
    # Later drafts must not evict the older program's relevant evidence.
    learner.practice_trials.extend(trial(i, OLD + f'# draft {i}\n', 'case_b', 900, 9., 6, False)
                                   for i in range(2, 10))
    learner.practice_trials.extend([trial(10, NEW, 'case_a', 808, 4., 5),
                                   trial(11, NEW, 'case_b', 808, 6., 5)])
    original = deepcopy(learner.practice_trials)
    costs = deepcopy(learner.costs)
    expected = decision(learner)
    context = prepare_code_context(learner.workspace, {
        'files': dict(learner.files), 'events': [], 'observations': [],
        'session_knowledge': learner.knowledge, 'research_notes': 'old note ' * 130000,
        'task_interface': learner.common_interface}, schema={}, instructions='test')
    assert len(json.dumps(context)) < 1_000_000
    assert context['method_decision'] == expected
    old, new = context['method_decision']['candidates'][0], context['method_decision']['candidates'][-1]
    old_a, old_b = old['latest_case_evidence']
    new_a, new_b = new['latest_case_evidence']
    assert old_a['costs']['observation_cost'] == 552 < new_a['costs']['observation_cost'] == 808
    assert old_a['costs']['python_execution_seconds'] == 100. > new_a['costs']['python_execution_seconds'] == 4.
    assert old_a['execution_summary']['action_counts']['query'] == 3
    assert new_b['execution_summary']['action_counts']['query'] == 5
    assert old_b['fresh_confirmation']['passed'] is False
    assert old['supported_cases'] == ['case_a'] and old['failed_cases'] == ['case_b']
    assert new['supported_cases'] == ['case_a', 'case_b'] and new['failed_cases'] == []
    assert old['restore_action']['arguments']['trial'] == 'trials/trial_000'
    assert expected['current_source_coverage']['confirmed_cases'] == ['case_a', 'case_b']
    assert learner.model is learner.published_source is None
    assert learner.practice_trials == original and learner.costs == costs
    assert (learner.workspace / METHOD_PATH).read_text() == NEW


def test_candidate_index_roundtrip_keeps_compact_costs_and_last_outcomes(tmp_path):
    learner = author(tmp_path / 'author')
    first = trial(0, OLD, 'case_a', 552, 100., 3)
    later_failure = trial(1, OLD, 'case_a', 600, 101., 4, False)
    newer = trial(2, NEW, 'case_b', 808, 4., 5)
    first['execution_summary']['research_steps'] = [{'large_paid_payload': 'x' * 10000}] * 20
    learner.practice_trials = [first, later_failure, newer]
    learner._refresh_method_candidates()
    index = json.loads(learner.files['method_candidates.json'])
    assert index == json.loads((learner.workspace / 'method_candidates.json').read_text())
    assert len(json.dumps(index)) < 6000
    for entry in index:
        for retained in entry['trials']:
            original = next(row for row in learner.practice_trials if row['directory'] == retained['directory'])
            assert retained['costs'] == original['costs']
            assert retained['execution_summary']['action_counts'] == original['execution_summary']['action_counts']
            assert 'research_steps' not in retained['execution_summary']
    resumed = author(tmp_path / 'resumed')
    sources = {row['directory']: row['method_source'] for row in learner.practice_trials}
    resumed.practice_trials = [{**row, 'method_source': sources[row['directory']]}
                              for entry in index for row in entry['trials']]
    view = decision(resumed)
    old, new = view['candidates']
    assert old['supported_cases'] == [] and old['failed_cases'] == ['case_a']
    assert old['latest_case_evidence'][0]['directory'] == later_failure['directory']
    assert old['latest_case_evidence'][0]['costs']['observation_cost'] == 600
    assert new['supported_cases'] == ['case_b']
    assert view['current_source_coverage']['missing_cases'] == ['case_a']
    assert resumed.model is resumed.published_source is None


def test_selected_legacy_metadata_is_loaded_once_without_changing_support(tmp_path, monkeypatch):
    selected = trial(0, OLD, 'case_a', 552, 100., 3, False)
    untouched = trial(1, NEW, 'case_b', 808, 4., 5)
    selected.pop('costs')
    selected.pop('execution_summary')
    trials = [selected, untouched]
    original = deepcopy(trials)
    path = tmp_path / selected['directory'] / 'result.json'
    path.parent.mkdir(parents=True)
    saved = {'status': 'completed', 'costs': {'observation_cost': 552,
        'python_execution_seconds': 100.}, 'method_results': [
            {'proposal': {'actions': []}, 'feedback': {'observation_cost': 128}},
            {'proposal': {'model': 'model.py'}, 'feedback': {'fresh_confirmation': {
                'passed': True, 'rows': 96, 'mse': 0.}}}]}
    path.write_text(json.dumps(saved))
    raw = path.read_bytes()
    # Neither unrelated trials nor adjacent final evaluation enter this view.
    (path.parent / 'evaluation.json').write_text('not paid development evidence')
    unrelated = tmp_path / untouched['directory'] / 'result.json'
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text('do not read this unselected legacy result')
    real_read, reads = Path.read_text, []

    def read(p, *args, **kwargs):
        reads.append(p)
        return real_read(p, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read)
    assert restore_paid_practice_summaries(trials, tmp_path, [selected['directory']]) == [selected['directory']]
    assert restore_paid_practice_summaries(trials, tmp_path, [selected['directory']]) == []
    assert reads == [path]
    assert selected['costs'] == saved['costs']
    assert selected['execution_summary']['action_counts'] == {'query': 1, 'model': 1}
    assert {key: selected[key] for key in original[0]} == original[0]
    assert untouched == original[1]
    assert path.read_bytes() == raw
