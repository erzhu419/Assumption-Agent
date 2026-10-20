"""Offline contract checks; the scripted affine fixture is not discovery evidence."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from hegel_machine.cognition.anonymous_method_learning import EMPTY_METHOD
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts import run_astra_code_frozen_comparison as comparison
from test_anonymous_method_learning import FITTED_MODEL


REFIT = """import json
def select_actions(spec, records, memory):
    steps = json.load(open('research_state.json'))['method_results']
    if not steps:
        return {'action':'write','arguments':{'path':'local.py','code':__MODEL__}}
    return {'model':'local.py'}
""".replace('__MODEL__', repr(FITTED_MODEL))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def costs():
    return dict(observation_cost=0, measurement_cost=0, compute_cost=0,
                python_execution_seconds=0., model_calls=0, reported_tokens=0)


def fixture_plan(tmp_path):
    old = tmp_path / 'old'
    prior = tmp_path / 'prior_author'
    baseline = tmp_path / 'hegel_astra'
    code = tmp_path / 'code_source'
    practice = {f'design_case_{i:03d}': {'kind': 'memoryless', 'seed': 100 + i,
        'prefix_seed': 200 + i, 'confirmation_seed': 300 + i} for i in range(9)}
    targets = [{'id': f'target_{i:03d}', **case, 'evaluation_seed': 400 + i}
               for i, case in enumerate(practice.values())]
    targets[0]['limits'] = {'observation_budget': 360, 'confirmation_mse_threshold': 1e-10}
    limits = asdict(InquiryLLMConfig(observation_budget=2240, confirmation_rows=96,
        execution_seconds=7200., compute_budget=640000, max_model_calls=96,
        token_budget=10_000_000, confirmation_mse_threshold=1e-6))
    method = {'source': str(old), 'continued_author': str(prior / 'training.json'),
        'model': 'gpt-6-astra', 'provider': {'reasoning_effort': 'high', 'timeout_seconds': 1800.},
        'author_limits': asdict(InquiryLLMConfig(max_model_calls=96, token_budget=20_000_000,
            observation_budget=120000, compute_budget=8_000_000, execution_seconds=36000.)),
        'limits': limits, 'prefix_rows': 72, 'practice_cases': practice,
        'transfer_cases': targets, 'evaluation': {'rows': 32}, 'method_steps': 32,
        'prior_author_phases': [{'result': str(prior / 'training.json'), 'costs': costs()}]}
    write(baseline / 'protocol.json', {'method': method})
    write(baseline / 'method/protocol.json', {'online_research_experience': 'FORBIDDEN_HEGEL_NEW_SOURCE'})
    write(baseline / 'method/published_method.json', {'source': 'FORBIDDEN_HEGEL_NEW_METHOD'})
    memory = {'files': {METHOD_PATH: EMPTY_METHOD}, 'manifest': {},
              'lessons': [], 'research_roles': {'predictors': []}}
    write(old / 'initial_archive.json', memory)
    write(old / 'source_outcomes.json', [])
    write(old / 'author/training.json', {'status': 'completed', 'method_source': EMPTY_METHOD,
        'method_path': METHOD_PATH, 'method_entry': {'id': 'old', 'path': METHOD_PATH},
        'lesson': 'Historical program', 'practice_trials': []})
    trials = [{'case': 'design_case_000', 'directory': f'trials/trial_{i:03d}',
        'method_source': EMPTY_METHOD, 'status': 'confirmation_failed', 'costs': costs(),
        'selected_prior_predictor': False, 'fresh_confirmation': {'passed': False}} for i in range(99)]
    write(prior / 'training.json', {'status': 'token_budget_exhausted', 'workspace': str(prior),
        'method_source': None, 'outgoing_knowledge': memory, 'practice_trials': trials, 'events': []})
    write(code / 'training.json', {'status': 'completed', 'agent_arm': 'code_react', 'model': 'model.py',
        'fresh_confirmation': {'passed': True, 'mse': 0., 'rows': 96},
        'costs': costs(), 'continuation_cost': costs()})
    write(code / 'evaluation.json', {'passed': True})
    (code / 'model.py').write_text('# own Code source fixture')
    return comparison.protocol(baseline, code)


def test_compiler_uses_pre_astra_archive_99_trials_and_only_own_confirmed_source(tmp_path):
    plan = fixture_plan(tmp_path)
    assert 'FORBIDDEN' not in json.dumps(plan)
    assert plan['author_limits']['max_model_calls'] == 96
    assert plan['author_limits']['token_budget'] == 20_000_000
    assert plan['provider']['reasoning_effort'] == 'high'
    (tmp_path / 'compiled').mkdir()
    author, original, sources = comparison.prepare_author(plan, tmp_path / 'compiled', lambda _: None)
    assert len(author.practice_trials) == 99
    assert author.arm == 'code_react' and author.questions == []
    assert 'counterexamples, visible on the next invocation' not in author.common_interface
    evidence = json.loads(author.files['online_research_experience.json'])
    assert (author.workspace / evidence['model']).resolve() == Path(plan['online_research_experience']) / 'model.py'
    assert 'model.py' not in original['files']
    assert sources == []
    assert author.costs == costs()
    assert (author.workspace / 'trials/trial_098').is_symlink()
    draft = author.result('runtime_interrupted')
    assert draft['outgoing_knowledge']['files'] == {METHOD_PATH: EMPTY_METHOD}


def test_plain_coding_requires_explicit_practice_but_all_nine_predictive_cases_publish(tmp_path):
    plan = fixture_plan(tmp_path)
    (tmp_path / 'compiled').mkdir()
    author, _, _ = comparison.prepare_author(plan, tmp_path / 'compiled', lambda _: None)
    feedback = author._execute('write', {'path': METHOD_PATH, 'code': REFIT}, author.interface)
    assert 'edit_practice' not in feedback
    assert len(author.practice_trials) == 99
    outcome = author._execute('finish', {'path': METHOD_PATH, 'lesson': 'Scripted affine fixture'}, author.interface)
    assert outcome['published'] == METHOD_PATH
    assert len(outcome['publication_practices']) == 9
    coverage = outcome['current_source_coverage']
    assert len(coverage['confirmed_cases']) == 9
    assert coverage['missing_cases'] == coverage['prior_reuse_cases'] == []
    assert len(coverage['revised_cases']) == 9
    assert author.result('completed')['method_source'] == REFIT
    assert author.costs['model_calls'] == 0
    # Changing the exact program removes coverage; publication must execute it.
    author._execute('write', {'path': METHOD_PATH, 'code': EMPTY_METHOD}, author.interface)
    assert len(author._current_source_coverage()['missing_cases']) == 9


def test_real_target_has_no_provider_no_hegel_root_and_keeps_per_case_limits(tmp_path):
    plan = fixture_plan(tmp_path)
    knowledge = comparison.read_json(Path(plan['source']) / 'initial_archive.json')
    prepare = comparison.trial_preparer(knowledge, [], tmp_path / 'reference',
                                        agent_type=comparison.FrozenCodeProgramAgent)
    cfg = InquiryLLMConfig(**{**plan['limits'], 'max_model_calls': 0, 'token_budget': 0})
    for index in (0, 1):
        case = plan['transfer_cases'][index]
        payload = prepare(case, tmp_path / f'target_{index}', REFIT, cfg)
        agent = payload['agent']
        assert agent.provider is None and agent.arm == 'code_react'
        assert agent.questions == [] and agent._active_question_id is None
        assert agent.config.max_model_calls == agent.config.token_budget == 0
        assert agent.config.observation_budget == (360 if index == 0 else 2240)
        assert agent.config.confirmation_mse_threshold == (1e-10 if index == 0 else 1e-6)
        assert agent.config.execution_seconds == 7200. and agent.config.compute_budget == 640000
        assert agent.costs['observation_cost'] == len(agent.rows) == 72
        result = agent.execute_method(payload['laboratory'], max_steps=32)
        assert result['status'] == 'completed' and result['fresh_confirmation']['passed']
        assert result['agent_arm'] == 'code_react' and result['costs']['model_calls'] == 0
        assert len(agent.confirmations) == 1


def test_frozen_completed_failure_is_reused_and_scoring_waits_for_all_targets(tmp_path, monkeypatch):
    plan = fixture_plan(tmp_path)
    output = tmp_path / 'compiled'
    publication = {'status': 'completed', 'method_source': REFIT, 'method_path': METHOD_PATH,
        'method_entry': {'id': 'fixture', 'path': METHOD_PATH}, 'lesson': '',
        'practice_trials': [], 'costs': costs(), 'events': []}
    write(output / 'author/training.json', publication)
    write(output / 'author_origin.json', {'fixture': True})
    write(output / 'source_outcomes.json', [])
    saved_failure = {'status': 'method_no_progress', 'model': None, 'costs': costs()}
    first = plan['transfer_cases'][0]
    write(output / 'frozen' / first['id'] / 'training.json', saved_failure)
    loaded = comparison.frozen_transfers({**plan, 'transfer_cases': [first]}, output, {}, [], REFIT,
                                         agent_type=comparison.FrozenCodeProgramAgent)
    assert loaded == [(first, saved_failure)]
    stopped = []

    def freeze(part, destination, memory, sources, source, **kwargs):
        assert kwargs['agent_type'] is comparison.FrozenCodeProgramAgent
        assert part['transfer_cases'] == plan['transfer_cases']
        stopped.extend(case['id'] for case in part['transfer_cases'])
        return [(case, deepcopy(saved_failure)) for case in part['transfer_cases']]

    def score(part, rows):
        assert stopped == [case['id'] for case in plan['transfer_cases']]
        return [{'case': case['id'], 'fresh_confirmation_passed': False,
            'terminal': {'passed': False}, 'costs': costs()} for case, _ in rows]

    def forbidden(_):
        raise AssertionError('Frozen continuation must never construct a model provider')

    monkeypatch.setattr(comparison, 'frozen_transfers', freeze)
    monkeypatch.setattr(comparison, 'score_frozen', score)
    summary = comparison.run_comparison(plan, output, provider_factory=forbidden)
    assert summary['target_count'] == len(plan['transfer_cases'])
    assert summary['target_model_calls'] == summary['target_terminal_passed'] == 0
