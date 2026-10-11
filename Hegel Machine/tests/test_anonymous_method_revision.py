"""Publication restoration tests use saved fixture support, not research results."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from hegel_machine.cognition.temporal_paired_practice import PairedMethodLearner
from hegel_machine.cognition.anonymous_research_archive import load_research_archive
from scripts import run_anonymous_method_revision as revision
from scripts.run_anonymous_method_learning import protocol as learning_protocol
from test_anonymous_method_learning import LEARNED_METHOD, action, memoryless_source, read_json, write_json


SOURCE = "def select_actions(spec, rows, memory):\n    return {'stop': 'fixture'}\n"


def learner(path, cases, source=SOURCE):
    return PairedMethodLearner(None, path, knowledge={'files': {METHOD_PATH: source}},
        cases=cases, lab_factory=lambda _: AnonymousHistoryLab(kind='memoryless', seed=42),
        config=InquiryLLMConfig())


def test_published_source_restore_keeps_support_but_requires_new_case_before_publication(tmp_path):
    old_cases = {'case_000': {}, 'case_001': {}}
    original = learner(tmp_path / 'original', old_cases)
    for index, case in enumerate(old_cases):
        directory = f'trials/trial_{index:03d}'
        trial = original.workspace / directory
        (trial / METHOD_PATH).parent.mkdir(parents=True)
        (trial / METHOD_PATH).write_text(SOURCE)
        (trial / 'result.json').write_text(json.dumps({'case': case, 'fixture_support': True}))
        original.practice_trials.append({'case': case, 'directory': directory,
            'method_source': SOURCE, 'status': 'completed', 'costs': {'observation_cost': 168},
            'selected_prior_predictor': index == 0,
            'fresh_confirmation': {'passed': True, 'mse': 0., 'rows': 96}})
    original.model, original.published_source = METHOD_PATH, SOURCE
    previous = original.result('completed')
    preserved = deepcopy(previous)
    old_bytes = {str(p.relative_to(original.workspace)): p.read_bytes()
                 for p in original.workspace.rglob('*') if p.is_file()}
    new = learner(tmp_path / 'revision', {**old_cases, 'case_006': {}}, source='# placeholder\n')

    context = new.restore_practice(previous)

    assert new.model is None and new.published_source is None
    assert new.result('researching')['method_source'] is None
    assert (new.workspace / METHOD_PATH).read_text() == SOURCE
    assert context['previous_status'] == 'completed'
    assert context['current_source_coverage'] == {
        'confirmed_cases': ['case_000', 'case_001'], 'missing_cases': ['case_006'],
        'prior_reuse_cases': ['case_000'], 'revised_cases': ['case_001']}
    assert new.practice_trials == previous['practice_trials']
    assert new.costs['observation_cost'] == new.costs['model_calls'] == 0
    for trial in previous['practice_trials']:
        assert (new.workspace / trial['directory'] / 'result.json').read_bytes() == (
            original.workspace / trial['directory'] / 'result.json').read_bytes()
    rejected = new._execute('finish', {'path': METHOD_PATH}, new.interface)
    assert rejected['unconfirmed_cases'] == ['case_006']
    assert new.model is None and new.published_source is None
    assert previous == preserved
    assert old_bytes == {str(p.relative_to(original.workspace)): p.read_bytes()
                         for p in original.workspace.rglob('*') if p.is_file()}


@pytest.mark.parametrize('method_source', [None, ''])
def test_completed_without_published_source_is_not_restorable(tmp_path, method_source):
    author = learner(tmp_path, {'case_000': {}})
    with pytest.raises(ValueError, match='actually published method'):
        author.restore_practice({'status': 'completed', 'method_source': method_source})


def published_fixture(tmp_path, memoryless_source, *, full_protocol=False):
    source = tmp_path / 'published_fixture'
    old_plan = learning_protocol(memoryless_source, model='scripted-test-only')
    if full_protocol:
        original_practice = list(old_plan['practice_cases'].values())
        original_targets = deepcopy(old_plan['transfer_cases'])
        old_plan['practice_cases'] = {
            f'case_{index * 2 + offset:03d}': {**original_practice[offset], 'kind': kind}
            for index, kind in enumerate(('memoryless', 'delay', 'hysteresis'))
            for offset in range(2)}
        old_plan['transfer_cases'] = [
            {**case, 'id': f'target_{index * 3 + offset:03d}', 'kind': kind}
            for index, kind in enumerate(('memoryless', 'delay', 'hysteresis'))
            for offset, case in enumerate(original_targets)]
    else:
        old_plan['practice_cases'] = {'case_000': old_plan['practice_cases']['case_000']}
        old_plan['transfer_cases'] = old_plan['transfer_cases'][:1]
    write_json(source / 'protocol.json', old_plan)
    knowledge, sources = load_research_archive(memoryless_source)
    old_method = """import json
def select_actions(spec, records, memory):
    if json.load(open('research_state.json'))['method_results']:
        return {'stop': 'The fixture prior did not confirm.'}
    return {'model': memory['research_roles']['predictors'][0]}
"""
    knowledge['files'][METHOD_PATH] = old_method
    write_json(source / 'initial_archive.json', knowledge)
    write_json(source / 'source_outcomes.json', sources)
    for name, contents in knowledge['files'].items():
        path = source / 'reference_archive' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    # Saved support is a restoration fixture, not a claim that this one-case
    # archive reproduces the earlier scientific publication cohort.
    old_author = learner(source / 'author', old_plan['practice_cases'], old_method)
    directory = 'trials/trial_000'
    (old_author.workspace / directory / METHOD_PATH).parent.mkdir(parents=True)
    (old_author.workspace / directory / METHOD_PATH).write_text(old_method)
    old_author.practice_trials = [{'case': 'case_000', 'directory': directory,
        'method_source': old_method, 'status': 'completed', 'costs': {'observation_cost': 168},
        'selected_prior_predictor': True,
        'fresh_confirmation': {'passed': True, 'mse': 0., 'rows': 96}}]
    old_author.model, old_author.published_source = METHOD_PATH, old_method
    old_author.costs['observation_cost'] = 168
    write_json(old_author.workspace / 'training.json', old_author.result('completed'))
    return source


@pytest.mark.parametrize('publishes', [True, False])
def test_revision_runner_preserves_counterexample_cost_and_frozen_order(
        tmp_path, memoryless_source, monkeypatch, publishes):
    """Real author/practice; mock only repeated target runs and their scorer."""
    source = published_fixture(tmp_path, memoryless_source)
    output = tmp_path / 'revision_run'
    before = {str(path.relative_to(source)): path.read_bytes()
              for path in source.rglob('*') if path.is_file()}

    plan = revision.protocol(source, model='scripted-test-only')
    # Small affine fixture isolates orchestration from the real counterexample's
    # research difficulty; source and changed object still execute real physics.
    plan['practice_cases']['case_006'] = {**plan['practice_cases']['case_006'], 'kind': 'memoryless'}
    plan['transfer_cases'] = plan['transfer_cases'][:1]
    plan['author_limits']['max_model_calls'] = 2 if publishes else 1
    order = []

    class Author:
        def next_action(self, request):
            assert 'effective_confirmation_seed' not in json.dumps(request['events'])
            if publishes and request['costs']['model_calls'] == 0:
                # Provider sees the pre-increment costs copied into the request.
                return action('write', {'path': METHOD_PATH, 'code': LEARNED_METHOD})
            return action('finish', {'path': METHOD_PATH, 'lesson': 'Scripted runtime revision.'})

    def targets(plan, branch, memory, sources, method_source):
        order.append('research:' + branch.name)
        case = plan['transfer_cases'][0]
        return [(case, {'costs': {'observation_cost': 17, 'compute_cost': 5},
                        'branch': branch.name})]

    def scores(plan, results):
        if not results:
            return []
        assert order[:2] == ['research:original', 'research:revised']
        order.append('score:' + results[0][1]['branch'])
        return [{'case': case['id'], 'kind': case['kind'], 'status': 'completed',
                 'costs': result['costs'], 'terminal': {'passed': True}}
                for case, result in results]

    monkeypatch.setattr(revision, 'frozen_transfers', targets)
    monkeypatch.setattr(revision, 'score_frozen', scores)
    summary = revision.run_revision(plan, output, provider_factory=lambda _: Author())
    learned = read_json(output / 'author' / 'training.json')
    baseline = read_json(output / 'baseline_reproduction.json')
    assert baseline['fresh_confirmation'] is None
    assert baseline['costs']['observation_cost'] == 72
    rejected = baseline['steps'][0]['feedback']
    assert rejected['returncode'] == 0 and rejected['mse'] > 1e-10 and rejected.get('error')
    assert baseline['effective_confirmation_seed'] == 62313561
    assert learned['events'][0]['feedback'].get('effective_confirmation_seed') is None
    new_trials = learned['practice_trials'][1:]
    charged = sum(trial['costs']['observation_cost'] for trial in new_trials)
    assert summary['author_costs']['observation_cost'] == charged
    assert new_trials[0]['costs'] == summary['baseline_costs']
    assert summary['old_author_costs']['observation_cost'] == 168
    if publishes:
        assert summary['method_published'] and summary['source_changed']
        assert order == ['research:original', 'research:revised', 'score:original', 'score:revised']
        assert summary['new_costs']['observation_cost'] == charged + 34
        # Editing loses old-source coverage: the revised source really practices
        # the new counterexample and the formerly supported case before publishing.
        revised = [trial for trial in new_trials if trial['method_source'] == LEARNED_METHOD]
        assert {trial['case'] for trial in revised} == {'case_006', 'case_000'}
        assert all(trial['fresh_confirmation']['passed'] for trial in revised)
        assert summary['unavailable_target_episodes'] == 0
    else:
        assert not summary['method_published'] and order == []
        assert summary['unavailable_target_episodes'] == 2
        assert summary['new_costs']['observation_cost'] == charged
        assert not (output / 'original').exists() and not (output / 'revised').exists()
    assert before == {str(path.relative_to(source)): path.read_bytes()
                      for path in source.rglob('*') if path.is_file()}


def test_continuation_restores_failed_author_and_paid_counterexample_without_repurchase(
        tmp_path, memoryless_source, monkeypatch):
    source = published_fixture(tmp_path, memoryless_source, full_protocol=True)
    failed, continued = tmp_path / 'failed_phase', tmp_path / 'continued_phase'
    plan = revision.protocol(source, model='scripted-test-only')
    plan['practice_cases']['case_006']['kind'] = 'memoryless'
    plan['author_limits']['max_model_calls'] = 1
    assert len(plan['practice_cases']) == 7 and len(plan['transfer_cases']) == 33

    def no_targets(*args, **kwargs):
        raise AssertionError('An unpublished author must not start frozen target research.')

    monkeypatch.setattr(revision, 'frozen_transfers', no_targets)

    class FailedAuthor:
        def next_action(self, request):
            return action('write', {'path': METHOD_PATH, 'code': SOURCE})

    first = revision.run_revision(plan, failed, provider_factory=lambda _: FailedAuthor())
    previous_path = failed / 'author' / 'training.json'
    previous = read_json(previous_path)
    baseline = read_json(failed / 'baseline_reproduction.json')
    baseline_result = read_json(failed / 'author' / baseline['directory'] / 'result.json')
    assert not first['method_published']
    assert previous['status'] == 'model_call_budget_exhausted'
    assert previous['outgoing_knowledge']['files'][METHOD_PATH] == SOURCE
    assert len(previous['practice_trials']) == 3  # Prior support, rejected prior, unsuccessful edit.
    assert previous['practice_trials'][-1]['status'] == 'method_stopped'
    assert baseline_result['fresh_confirmation'] is None
    assert baseline_result['costs']['observation_cost'] == 72
    rejected = baseline_result['method_results'][0]['feedback']
    assert rejected['returncode'] == 0 and rejected['mse'] > 1e-10 and rejected.get('error')
    original_bytes = {str(path): path.read_bytes() for root in (source, failed)
                      for path in root.rglob('*') if path.is_file()}

    resumed_plan = revision.protocol(source, model='scripted-test-only', continue_author=previous_path)
    assert resumed_plan['practice_cases'] == plan['practice_cases']
    assert resumed_plan['transfer_cases'] == plan['transfer_cases']
    assert resumed_plan['old_target_ids'] == plan['old_target_ids']
    assert len(resumed_plan['prior_author_phases']) == len(first['prior_author_phases']) + 1
    assert sum(phase['result'] == str(previous_path) for phase in resumed_plan['prior_author_phases']) == 1

    def no_reproduction(*args, **kwargs):
        raise AssertionError('Continuation must reuse the recorded counterexample without buying it again.')

    monkeypatch.setattr(revision, 'reproduce_counterexample', no_reproduction)
    seen = []

    class InspectingAuthor:
        def __init__(self, workspace):
            self.workspace = workspace

        def next_action(self, request):
            seen.append(deepcopy(request))
            assert request['costs']['model_calls'] == request['costs']['observation_cost'] == 0
            assert 'Revise the existing research program using its paid counterexample.' in request['task_interface']
            assert 'Turn the supplied actual research archives' not in request['task_interface']
            assert request['files'][METHOD_PATH] == SOURCE
            restored = read_json(self.workspace / 'resume_context.json')
            assert restored['previous_status'] == previous['status']
            assert len(restored['previous_practice']) == len(previous['practice_trials'])
            for trial in previous['practice_trials']:
                old_path = failed / 'author' / trial['directory'] / METHOD_PATH
                assert (self.workspace / trial['directory'] / METHOD_PATH).read_bytes() == old_path.read_bytes()
            ledger = read_json(self.workspace / 'counterexample.json')
            assert ledger == json.loads(request['files']['counterexample.json'])
            assert ledger['practice_case'] == 'case_006'
            assert ledger['executed_method']['source'] == baseline_result['method_source']
            assert ledger['executed_method']['source'] != SOURCE  # The rejected execution is not the last draft.
            assert ledger['steps'] and any('proposal' in step for step in ledger['steps'])
            failed_steps = [step for step in ledger['steps']
                if step.get('feedback', {}).get('error') and step['feedback'].get('mse', 0.) > 1e-10]
            assert failed_steps
            assert failed_steps[0]['feedback']['mse'] == rejected['mse']
            assert 'fresh_confirmation' not in failed_steps[0]['feedback']
            assert read_json(self.workspace / ledger['paid_records_file']) == baseline_result['paid_observations']
            assert read_json(self.workspace / ledger['full_trace_file'])
            return action('run', {'code': "import json\nprint(json.load(open('counterexample.json'))['status'])"})

    summary = revision.run_revision(resumed_plan, continued, provider_factory=InspectingAuthor)
    latest = read_json(continued / 'author' / 'training.json')
    assert len(seen) == 1
    assert not summary['method_published'] and summary['baseline_reused']
    assert summary['author_costs']['observation_cost'] == summary['new_costs']['observation_cost'] == 0
    assert summary['author_costs']['model_calls'] == 1
    assert summary['restored_practice_trials'] == len(previous['practice_trials'])
    assert summary['new_practice_trials'] == 0
    assert latest['practice_trials'] == previous['practice_trials']
    assert summary['old_author_costs']['observation_cost'] == (
        first['old_author_costs']['observation_cost'] + previous['costs']['observation_cost'])
    assert summary['prior_author_phases'] == resumed_plan['prior_author_phases']
    assert read_json(continued / 'baseline_reproduction.json') == baseline
    assert summary['unavailable_target_episodes'] == 66
    assert not (continued / 'original').exists() and not (continued / 'revised').exists()
    assert all(Path(name).read_bytes() == contents for name, contents in original_bytes.items())
