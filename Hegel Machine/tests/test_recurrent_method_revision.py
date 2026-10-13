"""Paid archive and method-transfer orchestration, not discovery evidence."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.anonymous_research_archive import load_research_archive
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts import run_anonymous_method_revision as revision
from scripts.run_anonymous_method_learning import protocol as learning_protocol, trial_preparer
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from test_anonymous_method_learning import action, memoryless_source, read_json, write_json
from test_anonymous_method_revision import learner


SOURCE = "def select_actions(spec, records, memory):\n    return {'stop': 'fixture source'}\n"


@pytest.fixture
def revision_sources(tmp_path, memoryless_source):
    source, repairs = tmp_path / 'published', tmp_path / 'repairs'
    old = learning_protocol(memoryless_source)
    old['practice_cases'] = {f'case_{i:03d}': {'kind': 'recurrent', 'seed': 71000 + i,
        'apparatus_seed': 71000 + i, 'prefix_seed': 72000 + i, 'confirmation_seed': 73000 + i}
        for i in range(4)}
    old['transfer_cases'] = [{'id': f'target_{i:03d}', 'kind': 'recurrent', 'seed': 81000 + i,
        'apparatus_seed': 81000 + i, 'prefix_seed': 82000 + i, 'confirmation_seed': 83000 + i,
        'evaluation_seed': 84000 + i} for i in range(6)]
    write_json(source / 'protocol.json', old)
    knowledge, records = load_research_archive(memoryless_source)
    knowledge['files'][METHOD_PATH] = SOURCE
    write_json(source / 'initial_archive.json', knowledge)
    write_json(source / 'source_outcomes.json', records)
    author = learner(source / 'author', old['practice_cases'], SOURCE)
    for index, case in enumerate(old['practice_cases']):
        directory = f'trials/trial_{index:03d}'
        path = author.workspace / directory / METHOD_PATH
        path.parent.mkdir(parents=True)
        path.write_text(SOURCE)
        author.practice_trials.append({'case': case, 'directory': directory,
            'method_source': SOURCE, 'status': 'completed', 'costs': {'observation_cost': 168},
            'selected_prior_predictor': index == 0,
            'fresh_confirmation': {'passed': True, 'mse': 0., 'rows': 96}})
    author.model, author.published_source = METHOD_PATH, SOURCE
    author.costs['observation_cost'] = 672
    write_json(author.workspace / 'training.json', author.result('completed'))
    cases = [{'id': f'case_{i:03d}', 'kind': kind, 'seed': 91000 + i // 2,
        'apparatus_seed': 91000 + i // 2, 'prefix_seed': 92000 + i // 2,
        'confirmation_seed': 93000 + i // 2, 'evaluation_seed': 94000 + i // 2}
        for i, kind in enumerate(('recurrent', 'finite2', 'finite2', 'recurrent'))]
    write_json(repairs / 'protocol.json', {'cases': cases, 'arms': ['code_react', 'hegel']})
    for case in cases:
        for arm in ('code_react', 'hegel'):
            workspace = repairs / case['id'] / arm
            passed = case['kind'] == 'recurrent' or arm == 'hegel'
            files = {'model.py': "import json\np = json.load(open('parameters.json'))\n"
                     "def predict(row):\n    return p['value']\n",
                     'parameters.json': '{"value": 0.0}', METHOD_PATH: SOURCE,
                     'notes.md': 'Actual fixture research, including rejected attempts.'}
            paid = [{'row_indices': [0], 'rows': [{'features': {'u': 0.}, 'target': 1.}],
                     'measurements': []}]
            archive = {'files': files, 'lessons': ['Fixture paid experience.'], 'question_traces': []}
            write_json(workspace / 'archive.json', archive)
            write_json(workspace / 'training.json', {'workspace': str(workspace),
                'status': 'completed' if passed else 'token_budget_exhausted',
                'model': 'model.py' if passed else None, 'paid_observations': paid,
                'events': [{'action': 'run', 'arguments': {'operator': 'procedure'},
                            'feedback': {'error': 'An earlier predictor was rejected.'}}],
                'costs': {'observation_cost': 72, 'model_calls': 1},
                'outgoing_knowledge': archive,
                'fresh_confirmation': {'passed': passed, 'mse': 0. if passed else 1., 'rows': 96}})
            write_json(workspace / 'evaluation.json', {'private_terminal_marker': 991827})
    return source, repairs


def test_protocol_keeps_original_support_and_fixes_new_targets_and_allowances(revision_sources):
    source, repairs = revision_sources
    old = read_json(source / 'protocol.json')
    plan = revision.protocol(source, repair_study=repairs)
    assert list(plan['practice_cases']) == [f'case_{i:03d}' for i in range(6)]
    assert {key: plan['practice_cases'][key] for key in old['practice_cases']} == old['practice_cases']
    assert plan['repair_cases'] == {'case_001': 'case_004', 'case_002': 'case_005'}
    assert plan['transfer_cases'][:6] == old['transfer_cases']
    new = plan['transfer_cases'][6:]
    assert len(new) == 4 and len({case['apparatus_seed'] for case in new}) == 2
    assert {(case['parameter_pair'], case['kind']) for case in new} == {
        (block, kind) for block in (0, 1) for kind in ('recurrent', 'finite2')}
    consumed = [*old['practice_cases'].values(), *old['transfer_cases'],
                *read_json(repairs / 'protocol.json')['cases']]
    assert {case['apparatus_seed'] for case in new}.isdisjoint(case['apparatus_seed'] for case in consumed)
    assert plan['author_limits']['max_model_calls'] == 96
    assert plan['author_limits']['token_budget'] == 20_000_000
    assert plan['author_limits']['observation_budget'] == 36_000
    assert plan['author_limits']['execution_seconds'] == 7200.
    assert plan['author_limits']['compute_budget'] == 4_000_000
    assert plan['limits']['observation_budget'] == 1152
    assert plan['limits']['execution_seconds'] == 3600.
    assert plan['provider']['timeout_seconds'] == 600.
    assert 'scripts/run_anonymous_method_revision.py' in plan['runtime_snapshot']


def test_author_only_experience_preserves_paths_failures_and_paid_support(
        tmp_path, revision_sources, monkeypatch):
    source, repairs = revision_sources
    previous = read_json(source / 'author/training.json')
    knowledge = read_json(source / 'initial_archive.json')
    before = {str(path): path.read_bytes() for root in (source, repairs)
              for path in root.rglob('*') if path.is_file()}
    plan = revision.protocol(source, repair_study=repairs)
    plan['author_limits']['max_model_calls'] = 1
    plan['runtime_snapshot'] = ['scripts/run_anonymous_method_revision.py']
    output = tmp_path / 'revision'

    def no_repurchase(*args, **kwargs):
        raise AssertionError('Historical repairs are already paid; do not reacquire them.')

    monkeypatch.setattr(revision, 'reproduce_counterexample', no_repurchase)
    monkeypatch.setattr(revision, 'frozen_transfers', no_repurchase)

    class InspectingAuthor:
        def next_action(self, request):
            assert request['files'][METHOD_PATH] == SOURCE
            assert request['costs']['observation_cost'] == request['costs']['model_calls'] == 0
            assert not any(name.startswith('repair_archive/') for name in request['files'])
            catalog = json.loads(request['files']['source_catalog.json'])
            assert request['session_knowledge']['source_catalog'] == catalog
            assert request['session_knowledge']['files']['source_catalog.json'] == request['files']['source_catalog.json']
            assert not any(name.startswith('repair_archive/')
                           for name in request['session_knowledge']['files'])
            assert catalog[:2] == knowledge['source_catalog']
            assert len(catalog) == 10
            assert sum(entry['status'] == 'token_budget_exhausted' for entry in catalog[2:]) == 2
            for entry in catalog[2:]:
                assert entry['directory'].startswith('repair_archive/archive/source_')
                assert (output / 'author' / entry['paid_trace']).is_file()
                assert (output / 'author' / entry['directory'] / 'model.py').read_text().startswith('import json')
                if entry['model']:
                    assert (output / 'author' / entry['model']).is_file()
            evidence = json.loads(request['files']['repair_evidence.json'])
            assert len(evidence['records']) == 4
            assert {row['case'] for row in evidence['records']} == {'case_004', 'case_005'}
            for record in evidence['records']:
                assert (output / 'author' / record['ledger']).is_file()
                assert (output / 'author' / record['observations']).is_file()
            resume = read_json(output / 'author/resume_context.json')
            assert resume['current_source_coverage']['missing_cases'] == ['case_004', 'case_005']
            assert len(resume['previous_practice']) == 4
            return action('run', {'code': "print('inspection only')"})

    summary = revision.run_revision(plan, output, provider_factory=lambda _: InspectingAuthor())
    learned = read_json(output / 'author/training.json')
    assert learned['practice_trials'] == previous['practice_trials']
    assert not summary['method_published'] and summary['unavailable_target_episodes'] == 20
    assert summary['baseline_reused'] and summary['baseline_costs']['observation_cost'] == 576
    assert summary['author_costs']['observation_cost'] == summary['new_costs']['observation_cost'] == 0
    assert summary['new_practice_trials'] == 0
    assert read_json(output / 'source_outcomes.json') == read_json(source / 'source_outcomes.json')
    assert not (output / 'reference_archive/repair_archive').exists()
    assert not (output / 'author/repair_archive/archive/source_000/_evaluated_model/evaluation.json').exists()
    assert before == {str(path): path.read_bytes() for root in (source, repairs)
                      for path in root.rglob('*') if path.is_file()}
    with pytest.raises(ValueError, match='one fixed allowance'):
        revision.protocol(source, continue_author=output / 'author/training.json')


def test_published_pair_uses_original_knowledge_and_scores_after_both_cohorts(
        tmp_path, revision_sources, monkeypatch):
    """Mock publication and targets only; inspect the real author and trial setup."""
    source, repairs = revision_sources
    knowledge = read_json(source / 'initial_archive.json')
    plan = revision.protocol(source, repair_study=repairs)
    plan['runtime_snapshot'] = []
    order, memories = [], []
    revised = SOURCE.replace('fixture source', 'fixture revised')

    def publish(author, *, on_checkpoint):
        # Author-only catalog changes must not alter the captured practice K0.
        payload = author.prepare_trial(plan['practice_cases']['case_004'],
            author.workspace / 'fixture_trial', revised, InquiryLLMConfig(**plan['limits']))
        agent = payload['agent']
        assert agent.knowledge['source_catalog'] == knowledge['source_catalog']
        assert not (agent.workspace / 'repair_archive').exists()
        assert read_json(agent.workspace / 'source_catalog.json') == knowledge['source_catalog']
        author.model, author.published_source = METHOD_PATH, revised
        return author.result('completed')

    def targets(plan, branch, memory, records, method_source):
        order.append('research:' + branch.name)
        memories.append(deepcopy(memory))
        assert memory['source_catalog'] == knowledge['source_catalog']
        assert all(memory['files'][name] == contents for name, contents in knowledge['files'].items()
                   if name != METHOD_PATH)
        assert not any(name.startswith('repair_archive/') for name in memory['files'])
        assert len(records) == 2
        if branch.name == 'original':
            # The real trial receives the old context and pays its ordinary prefix.
            payload = trial_preparer(memory, records, branch / 'reference_archive')(
                plan['practice_cases']['case_004'], branch / 'fixture_trial', method_source,
                InquiryLLMConfig(**plan['limits']))
            agent = payload['agent']
            assert agent.costs['observation_cost'] == 72
            assert not (agent.workspace / 'repair_archive').exists()
            assert read_json(agent.workspace / 'source_catalog.json') == knowledge['source_catalog']
        return [(case, {'costs': {'observation_cost': 17, 'compute_cost': 5}, 'branch': branch.name})
                for case in plan['transfer_cases']]

    def scores(plan, results):
        assert order[:2] == ['research:original', 'research:revised']
        order.append('score:' + results[0][1]['branch'])
        return [{'case': case['id'], 'kind': case['kind'], 'status': 'completed',
                 'costs': result['costs'], 'terminal': {'passed': True}}
                for case, result in results]

    monkeypatch.setattr(revision.AnonymousResearchMethodLearner, 'run', publish)
    monkeypatch.setattr(revision, 'frozen_transfers', targets)
    monkeypatch.setattr(revision, 'score_frozen', scores)
    summary = revision.run_revision(plan, tmp_path / 'paired', provider_factory=lambda _: None)
    assert memories[0]['files'][METHOD_PATH] == SOURCE
    assert memories[1]['files'][METHOD_PATH] == revised
    assert order == ['research:original', 'research:revised', 'score:original', 'score:revised']
    assert summary['source_changed'] and len(summary['pairs']) == 10
    assert sum(row['group'] == 'old_target' for row in summary['pairs']) == 6
    assert sum(row['group'] == 'new_parameter' for row in summary['pairs']) == 4
    assert summary['new_costs']['observation_cost'] == 340
    assert summary['live_consumer_model_calls'] == 0
