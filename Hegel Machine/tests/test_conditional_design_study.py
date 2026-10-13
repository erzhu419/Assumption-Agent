"""Mixed-task protocol wiring; scripted publication is not scientific evidence."""

from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts import run_anonymous_method_learning as learning
from scripts import run_anonymous_method_revision as revision
from scripts.experimental_design_author import ExperimentalDesignMethodLearner
from scripts.experimental_design_control import (
    ExperimentalDesignMethodAgent, RandomExperimentMethodAgent, VarianceExperimentMethodAgent)
from test_anonymous_method_learning import memoryless_source, read_json, write_json
from test_prospective_design_study import acquisition_source
from test_recurrent_method_revision import SOURCE, revision_sources


@pytest.fixture
def prospective_source(acquisition_source):
    plan = revision.protocol(acquisition_source, prospective_design=True)
    write_json(acquisition_source / 'protocol.json', plan)
    previous = read_json(acquisition_source / 'author/training.json')
    renamed = {old: new for new, old in plan['previous_practice_cases'].items()}
    for trial in previous['practice_trials']:
        trial['case'] = renamed[trial['case']]
    write_json(acquisition_source / 'author/training.json', previous)
    return acquisition_source


def test_mixed_plan_preserves_all_ten_old_targets_and_six_practice_conditions(prospective_source):
    old = read_json(prospective_source / 'protocol.json')
    before = deepcopy(old)
    plan = revision.protocol(prospective_source, conditional_design=True, timeout_seconds=610.)
    assert len(old['old_target_ids']) == 6 and len(old['transfer_cases']) == 10
    assert plan['old_target_ids'] == [case['id'] for case in old['transfer_cases']]
    for previous, current in zip(old['transfer_cases'], plan['transfer_cases'][:10]):
        assert {key: value for key, value in current.items() if key != 'limits'} == previous
        assert current['limits'] == {**old['limits'], 'confirmation_mse_threshold': 1e-10}
    assert list(plan['practice_cases'])[:6] == list(old['practice_cases'])
    for name, previous in old['practice_cases'].items():
        current = plan['practice_cases'][name]
        assert {key: value for key, value in current.items() if key != 'limits'} == previous
        assert current['limits']['observation_budget'] == old['limits']['observation_budget'] == 360
        assert current['limits']['confirmation_mse_threshold'] == 1e-10
    practice = list(plan['practice_cases'].values())[6:]
    targets = plan['transfer_cases'][10:]
    assert len(practice) == 3 and len(targets) == 12
    assert {case['stage'] for case in practice} == {1, 2, 4}
    assert {(case['parameter_pair'], case['stage']) for case in targets} == {
        (block, stage) for block in range(4) for stage in (1, 2, 4)}
    historical = [*old['practice_cases'].values(), *old['transfer_cases']]
    for key in ('apparatus_seed', 'prefix_seed', 'confirmation_seed', 'evaluation_seed'):
        assert {case[key] for case in targets}.isdisjoint(case[key] for case in practice)
        assert {case[key] for case in [*practice, *targets]}.isdisjoint(
            case[key] for case in historical if key in case)
    assert all(case['limits']['confirmation_mse_threshold'] == 1e-6 for case in [*practice, *targets])
    assert plan['model'] == 'gpt-6-luna' and plan['provider']['timeout_seconds'] == 610.
    assert read_json(prospective_source / 'protocol.json') == before


@pytest.mark.parametrize('case_index', [0, 6, 7, 8])
def test_preparation_caps_remaining_account_and_buys_public_72_row_prefix(
        tmp_path, prospective_source, case_index):
    plan = revision.protocol(prospective_source, conditional_design=True)
    case = list(plan['practice_cases'].values())[case_index]
    knowledge = read_json(prospective_source / 'initial_archive.json')
    prepare = learning.trial_preparer(knowledge, [], tmp_path / 'reference',
                                     agent_type=ExperimentalDesignMethodAgent)
    remaining = InquiryLLMConfig(observation_budget=180, compute_budget=200, execution_seconds=25.,
        max_model_calls=0, token_budget=0, confirmation_rows=96)
    payload = prepare(case, tmp_path / f'practice_{case_index}', SOURCE, remaining)
    agent, lab = payload['agent'], payload['laboratory']
    for key in ('observation_budget', 'compute_budget', 'execution_seconds', 'max_model_calls', 'token_budget'):
        assert getattr(agent.config, key) <= getattr(remaining, key)
    assert agent.costs['observation_cost'] == len(agent.rows) == 72
    assert sum(len(batch['rows']) for batch in agent.observations) == 72
    assert agent.config.confirmation_mse_threshold == (1e-10 if case_index == 0 else 1e-6)
    spec = lab.public_spec()
    assert not {'stage', 'seed', 'apparatus_seed', 'threshold', 'width', 'kind'} & spec.keys()
    controls = set(spec['controls'])
    assert all(set(row.features) == controls for row in agent.rows)
    assert all(set(point) == controls for row in agent.rows for point in row.history)
    assert '__target__' not in json.dumps(agent.observations)
    if case_index == 0:
        assert len(agent.observations) == 24
        assert max(len(row.history) for row in agent.rows) == 2
        larger = replace(remaining, observation_budget=900, compute_budget=500000, execution_seconds=9000.)
        capped = prepare(case, tmp_path / 'capped_old', SOURCE, larger)['agent']
        assert capped.config.observation_budget == 360
        assert capped.config.compute_budget == case['limits']['compute_budget']


def test_mixed_runner_pairs_cases_across_different_cohorts_after_all_research(
        tmp_path, prospective_source, monkeypatch):
    plan = revision.protocol(prospective_source, conditional_design=True)
    plan['runtime_snapshot'] = []
    old_ids = set(plan['old_target_ids'])
    new_ids = {case['id'] for case in plan['transfer_cases']} - old_ids
    revised_source = SOURCE.replace('fixture source', 'conditional fixture source')
    indexes = {case['id']: index for index, case in enumerate(plan['transfer_cases'])}
    offsets = {'original': 100, 'revised': 101, 'random_choice': 104, 'variance_choice': 108}
    classes = {'original': ExperimentalDesignMethodAgent, 'revised': ExperimentalDesignMethodAgent,
        'random_choice': RandomExperimentMethodAgent, 'variance_choice': VarianceExperimentMethodAgent}
    order, memories = [], {}

    def publish(author, *, on_checkpoint):
        public = json.loads(author.files['practice_interfaces.json'])
        assert set(public) == set(plan['practice_cases'])
        for name, interface in public.items():
            assert set(interface) == {'public_spec', 'limits'}
            assert not {'stage', 'seed', 'apparatus_seed', 'threshold', 'width', 'kind'} & interface['public_spec'].keys()
            assert interface['limits']['confirmation_mse_threshold'] == (
                1e-6 if plan['practice_cases'][name]['kind'] == 'conditional_window' else 1e-10)
        coverage = author._current_source_coverage()
        assert coverage['confirmed_cases'] == list(plan['practice_cases'])[:6]
        assert coverage['missing_cases'] == list(plan['practice_cases'])[6:]
        assert author.costs['observation_cost'] == 0
        author.model, author.published_source = METHOD_PATH, revised_source
        return author.result('completed')

    def targets(part, branch, memory, sources, method_source, *, agent_type):
        name = branch.name
        actual = {case['id'] for case in part['transfer_cases']}
        assert actual == (old_ids if name == 'original' else old_ids | new_ids if name == 'revised' else new_ids)
        assert agent_type is classes[name]
        assert part['method_steps'] == 32
        assert method_source == (SOURCE if name == 'original' else revised_source)
        assert memory['files'][METHOD_PATH] == method_source
        memories[name] = deepcopy(memory)
        order.append(('research', name))
        # Deliberately change order: pairing by zip would silently match wrong cases.
        selected = list(reversed(part['transfer_cases'])) if name == 'revised' else part['transfer_cases']
        return [(case, {'branch': name, 'events': [], 'method_results': [],
            'costs': {'observation_cost': offsets[name] + indexes[case['id']],
                      'compute_cost': 2 * offsets[name] + indexes[case['id']]}}) for case in selected]

    def score(part, results):
        assert order[:4] == [('research', name) for name in offsets]
        order.append(('score', results[0][1]['branch']))
        return [{'case': case['id'], 'kind': case['kind'], 'status': 'completed',
            'costs': result['costs'], 'terminal': {'passed': True}} for case, result in results]

    monkeypatch.setattr(ExperimentalDesignMethodLearner, 'run', publish)
    monkeypatch.setattr(revision, 'frozen_transfers', targets)
    monkeypatch.setattr(revision, 'score_frozen', score)
    result = revision.run_revision(plan, tmp_path / 'mixed', provider_factory=lambda _: None)
    assert result['method_published'] and result['source_changed']
    assert {name: len(rows) for name, rows in result['frozen'].items()} == {
        'original': 10, 'revised': 22, 'random_choice': 12, 'variance_choice': 12}
    assert memories['revised'] == memories['random_choice'] == memories['variance_choice']
    assert len(result['pairs']) == 10 and all(pair['observation_delta'] == 1 for pair in result['pairs'])
    assert {pair['case'] for pair in result['pairs']} == old_ids
    assert len(result['choice_pairs']) == len(result['variance_pairs']) == 12
    assert all(pair['observation_delta'] == -3 and pair['compute_delta'] == -6 for pair in result['choice_pairs'])
    assert all(pair['observation_delta'] == -7 and pair['compute_delta'] == -14 for pair in result['variance_pairs'])
    assert result['unavailable_target_episodes'] == 0


def test_unpublished_mixed_method_leaves_exactly_56_target_slots_unavailable(
        tmp_path, prospective_source, monkeypatch):
    plan = revision.protocol(prospective_source, conditional_design=True)
    plan['runtime_snapshot'] = []
    monkeypatch.setattr(ExperimentalDesignMethodLearner, 'run',
        lambda author, **kwargs: author.result('model_call_budget_exhausted'))

    def forbidden(*args, **kwargs):
        raise AssertionError('An unpublished draft must not enter frozen transfer.')

    monkeypatch.setattr(revision, 'frozen_transfers', forbidden)
    result = revision.run_revision(plan, tmp_path / 'unpublished', provider_factory=lambda _: None)
    assert not result['method_published']
    assert result['unavailable_target_episodes'] == 56
    assert result['new_practice_trials'] == result['author_costs']['observation_cost'] == 0


def test_terminal_scoring_keeps_old_exactness_and_new_public_risk_tolerance(
        tmp_path, prospective_source, monkeypatch):
    plan = revision.protocol(prospective_source, conditional_design=True)
    cases = [plan['transfer_cases'][0], plan['transfer_cases'][10]]
    monkeypatch.setattr(learning, 'laboratory', lambda case: SimpleNamespace(
        evaluation_rows=lambda **kwargs: []))
    monkeypatch.setattr(learning, 'evaluate_python_model', lambda *args, **kwargs:
        {'returncode': 0, 'mse': 5e-7, 'predictions': []})
    results = []
    for case in cases:
        workspace = tmp_path / case['id']
        workspace.mkdir()
        results.append((case, {'workspace': str(workspace), 'model': 'model.py', 'status': 'completed',
            'costs': {}, 'fresh_confirmation': {'passed': True}, 'method_results': []}))
    scored = learning.score_frozen(plan, results)
    assert [record['terminal']['mse_threshold'] for record in scored] == [1e-10, 1e-6]
    assert [record['terminal']['passed'] for record in scored] == [False, True]
