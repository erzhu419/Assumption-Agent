"""Comparison wiring tests use no paid model calls and do not test discovery."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from random import Random

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from scripts import run_astra_code_comparison as comparison
from test_common_archive_resume import checkpoint, SEED


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def forbidden(*args, **kwargs):
    raise AssertionError('This check must not make a paid model call')


@pytest.fixture
def plan(tmp_path):
    baseline = tmp_path / 'successful_astra'
    old = tmp_path / 'pre_astra'
    base_limits = asdict(InquiryLLMConfig(observation_budget=2240,
        execution_seconds=7200., compute_budget=640000, confirmation_rows=96))
    source_limits = {**base_limits, 'max_model_calls': 424, 'token_budget': 67_000_000,
                     'confirmation_mse_threshold': 1e-6}
    cases = [
        {'id': 'old', 'kind': 'recurrent', 'seed': 129,
         'apparatus_seed': 129, 'prefix_seed': 1132, 'confirmation_seed': 2132,
         'evaluation_seed': 3132, 'limits': {'observation_budget': 360,
                                            'confirmation_mse_threshold': 1e-10}},
        {'id': 'new', 'kind': 'conditional_window', 'stage': 1, 'seed': 213,
         'apparatus_seed': 213, 'noise_sigma': 0., 'prefix_seed': 1216,
         'confirmation_seed': 2216, 'evaluation_seed': 3216,
         'limits': {'confirmation_mse_threshold': 1e-6}}]
    write(baseline / 'protocol.json', {'resume_result': str(old / 'training.json'),
        'resume_workspace': str(old), 'source_study': str(old),
        'online': {'arms': ['hegel'], 'model': 'gpt-6-astra',
            'provider': {'reasoning_effort': 'high', 'timeout_seconds': 1800.},
            'cases': [cases[1]], 'limits': source_limits,
            'evaluation': {'rows': 512, 'mse_threshold': 1e-6}},
        'method': {'source': str(old), 'limits': base_limits, 'transfer_cases': cases,
                   'evaluation': {'rows': 512, 'mse_threshold': 1e-10}}})
    write(old / 'initial_archive.json', {'files': {'old.py': '# prior knowledge'},
                                       'manifest': {}, 'lessons': []})
    write(baseline / 'initial_archive.json', {'files': {'solution.py': '# forbidden new solution'}})
    return comparison.protocol(baseline)


def test_protocol_preserves_pre_astra_boundary_and_source_allocation(plan):
    assert plan['resume_result'].endswith('pre_astra/training.json')
    assert plan['source_archive'] == plan['target_archive']
    assert 'successful_astra' not in plan['target_archive']
    assert plan['online']['arms'] == ['code_react']
    assert plan['online']['model'] == 'gpt-6-astra'
    assert plan['online']['provider']['reasoning_effort'] == 'high'
    assert plan['online']['limits']['max_model_calls'] == 424
    assert plan['online']['limits']['token_budget'] == 67_000_000


def test_real_restore_keeps_only_pre_astra_paid_scene(tmp_path, monkeypatch):
    old, lab, previous = checkpoint(tmp_path)
    write(old.workspace / 'training.json', previous)
    write(old.workspace / 'initial_archive.json', old.knowledge)
    plan = {'resume_result': str(old.workspace / 'training.json'),
        'resume_workspace': str(old.workspace),
        'source_archive': str(old.workspace / 'initial_archive.json'),
        'online': {'cases': [{'confirmation_seed': SEED}]}}

    def run(part, output, *, prepare_prefix, knowledge, **kwargs):
        agent = CommonArchiveInquiryAgent(None, output / 'code_react', arm='code_react',
            knowledge=knowledge, source_task=False)
        current = TemporalInquiryLab(task_index=1)
        monkeypatch.setattr(current, 'execute', forbidden)
        restored = prepare_prefix(agent, current, part['cases'][0])
        assert agent.arm == 'code_react'
        assert agent.rows == old.rows
        assert agent.events == comparison.read_json(plan['resume_result'])['events']
        assert agent.costs == old.costs
        assert current.costs == lab.costs
        assert restored['prior_research_cost'] == previous['costs']
        assert agent.confirmations == json.loads(json.dumps(old.confirmations))
        assert agent.knowledge == old.knowledge
        return {'status': 'fixture_restoration_only'}

    monkeypatch.setattr(comparison, 'run_study', run)
    assert comparison.source_study(plan, tmp_path / 'code', forbidden)['status'] == 'fixture_restoration_only'


@pytest.mark.parametrize('index,stage', [(0, None), (1, 1), (1, 2), (1, 4)])
def test_target_prefix_matches_frozen_protocol(plan, tmp_path, index, stage):
    case = deepcopy(plan['method']['transfer_cases'][index])
    if stage is not None:
        case['stage'] = stage
    lab = comparison.laboratory(case)
    agent = CommonArchiveInquiryAgent(None, tmp_path / 'target', arm='code_react',
        source_task=False, config=comparison.target_limits(plan, case))
    comparison.prepare_target_prefix(agent, lab, case)
    rng = Random(case['prefix_seed'])
    if index == 0:
        expected = [{'kind': 'trajectory', 'channels': [],
            'sequence': [{'u': rng.uniform(-1., 1.)} for _ in range(3)]} for _ in range(24)]
    else:
        expected = comparison.laboratory(case).confirmation_actions(72, rng)
    assert agent.events[0]['arguments']['actions'] == expected
    assert len(agent.rows) == agent.costs['observation_cost'] == 72
    assert agent.costs['model_calls'] == 0


def test_case_limits_keep_physical_budgets_and_different_thresholds(plan):
    old, new = [comparison.target_limits(plan, case) for case in plan['method']['transfer_cases']]
    assert (old.observation_budget, new.observation_budget) == (360, 2240)
    assert (old.confirmation_mse_threshold, new.confirmation_mse_threshold) == (1e-10, 1e-6)
    for limits in (old, new):
        assert limits.max_model_calls == 96
        assert limits.token_budget == 10_000_000
        assert limits.execution_seconds == 7200.


def test_completed_failure_is_preserved_without_provider_call(plan, tmp_path):
    case = plan['method']['transfer_cases'][0]
    old = {'status': 'token_budget_exhausted', 'model': None}
    write(tmp_path / 'targets' / case['id'] / 'code_react/training.json', old)
    assert comparison.train_target(plan, tmp_path, case, {}, forbidden) == (case, old)


def test_source_failure_does_not_cancel_targets_and_scoring_waits(plan, tmp_path, monkeypatch):
    finished = set()
    monkeypatch.setattr(comparison, 'source_study', lambda *args: {'status': 'confirmation_failed'})

    def train(current, output, case, knowledge, provider):
        assert knowledge == comparison.read_json(plan['target_archive'])
        assert set(knowledge['files']) == {'old.py'}
        # Each target may mutate its own imported copy; no rescue result is used.
        finished.add(case['id'])
        return case, {'status': 'fixture_complete', 'fresh_confirmation': {'passed': True}}

    def score(method, results):
        assert finished == {'old', 'new'}
        assert [case['id'] for case, _ in results] == ['old', 'new']
        return [{'case': case['id'], 'fresh_confirmation_passed': True,
                 'terminal': {'passed': True}} for case, _ in results]

    monkeypatch.setattr(comparison, 'train_target', train)
    monkeypatch.setattr(comparison, 'score_frozen', score)
    result = comparison.run_comparison(plan, tmp_path / 'comparison', provider_factory=forbidden)
    assert result['source']['status'] == 'confirmation_failed'
    assert result['target_terminal_passed'] == result['target_count'] == 2


def test_target_does_not_inherit_rescue_or_sibling_source(plan, tmp_path, monkeypatch):
    case = plan['method']['transfer_cases'][1]
    limits = comparison.target_limits(plan, case)
    monkeypatch.setattr(comparison, 'target_limits', lambda *args: replace(limits, max_model_calls=0))
    knowledge = comparison.read_json(plan['target_archive'])
    _, result = comparison.train_target(plan, tmp_path, case, knowledge, lambda path: None)
    assert result['status'] == 'model_call_budget_exhausted'
    assert result['inherited_files'] == ['old.py']
    assert result['costs']['model_calls'] == 0
    assert knowledge == comparison.read_json(plan['target_archive'])
