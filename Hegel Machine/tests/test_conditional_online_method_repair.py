"""Online-to-method phase wiring; scripted outcomes are not discovery evidence."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from random import Random

import pytest

from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from scripts import run_anonymous_method_revision as revision
from scripts import run_conditional_online_method_repair as repair
from test_anonymous_method_learning import memoryless_source, read_json, write_json
from test_conditional_design_study import prospective_source
from test_prospective_design_study import acquisition_source
from test_recurrent_method_revision import SOURCE, revision_sources


def costs(observations=0, calls=0, tokens=0, compute=0, seconds=0.):
    return {'observation_cost': observations, 'measurement_cost': 0,
            'model_calls': calls, 'reported_tokens': tokens,
            'compute_cost': compute, 'python_execution_seconds': seconds}


@pytest.fixture
def failed_conditional_source(tmp_path, prospective_source):
    """Keep a failed author and 99 trial identities, with trial095's paid state."""
    source = tmp_path / 'failed_conditional'
    plan = revision.protocol(prospective_source, conditional_design=True)
    plan['runtime_snapshot'] = []
    plan['author_total_limits'] = deepcopy(plan['author_limits'])
    # This is the exhausted account's remaining limit, not a new development allowance.
    plan['author_limits']['token_budget'] = 11
    write_json(source / 'protocol.json', plan)
    earlier = {'result': 'earlier/training.json', 'status': 'provider_failed',
               'costs': costs(observations=72, calls=1, tokens=None),
               'usage': {'known_tokens': 7, 'missing_usage_calls': 1,
                         'input_tokens': 7, 'cached_input_tokens': 0}}
    write_json(source / 'summary.json', {'prior_author_phases': [earlier]})
    for filename in ('initial_archive.json', 'source_outcomes.json'):
        write_json(source / filename, read_json(prospective_source / filename))
    author = read_json(prospective_source / 'author/training.json')
    author.update(workspace=str(source / 'author'), status='token_budget_exhausted',
                  model=None, method_source=None, costs=costs(6432, 69, 19754849),
                  events=[{'action': 'run', 'usage': {'total_tokens': 19754849}}])
    author['practice_trials'] = [{
        'case': 'design_case_006', 'directory': f'trials/trial_{index:03d}',
        'method_source': SOURCE, 'status': 'confirmation_failed',
        'costs': costs(192), 'fresh_confirmation': {'passed': False, 'mse': 3.95, 'rows': 96}}
        for index in range(99)]
    write_json(source / 'author/training.json', author)
    selected = source / 'author/trials/trial_095'
    state = {'status': 'confirmation_failed', 'workspace': str(selected), 'model': None,
             'costs': costs(192, compute=840, seconds=5.5),
             'fresh_confirmation': {'passed': False, 'mse': 3.95, 'rows': 96},
             'events': [], 'paid_observations': [], 'method_results': [],
             'outgoing_knowledge': read_json(source / 'initial_archive.json')}
    write_json(selected / 'result.json', state)
    write_json(selected / 'research_state.json', {'costs': state['costs']})
    return source


def test_new_development_keeps_paid_account_and_all_registered_conditions(failed_conditional_source):
    source = failed_conditional_source
    previous_plan = read_json(source / 'protocol.json')
    previous_author = read_json(source / 'author/training.json')
    plan = repair.protocol(source)
    method = plan['method']
    assert plan['restored_practice_trials'] == 99
    assert plan['practice_index'] == 95
    assert plan['online']['cases'][0]['confirmation_seed'] == (
        previous_plan['practice_cases']['design_case_006']['confirmation_seed'] + 95)
    assert method['practice_cases'] == previous_plan['practice_cases']
    assert len(method['practice_cases']) == 9
    assert method['transfer_cases'] == previous_plan['transfer_cases']
    assert len(method['transfer_cases']) == 22
    assert method['old_target_ids'] == previous_plan['old_target_ids']
    assert len(method['old_target_ids']) == 10
    assert method['continued_author'] == str(source / 'author/training.json')
    assert method['author_limits'] == previous_plan['author_total_limits']
    assert method['prior_author_phases'][:-1] == read_json(source / 'summary.json')['prior_author_phases']
    assert method['prior_author_phases'][-1]['costs'] == previous_author['costs']
    assert method['prior_author_phases'][0]['costs']['reported_tokens'] is None
    assert plan['online']['limits']['observation_budget'] == plan['online_allowance']['observation_budget'] + 192
    assert plan['online']['limits']['compute_budget'] == plan['online_allowance']['compute_budget'] + 840
    assert plan['online']['limits']['execution_seconds'] == plan['online_allowance']['execution_seconds'] + 5.5
    assert plan['online']['limits']['max_model_calls'] == plan['online_allowance']['max_model_calls']
    assert plan['online']['limits']['confirmation_mse_threshold'] == 1e-6
    assert read_json(source / 'protocol.json') == previous_plan
    assert read_json(source / 'author/training.json') == previous_author


def test_explicit_new_allowances_do_not_reuse_exhausted_token_remainder(failed_conditional_source):
    online = InquiryLLMConfig(max_model_calls=4, token_budget=300, observation_budget=600,
        compute_budget=1000, execution_seconds=20., confirmation_rows=24)
    author = InquiryLLMConfig(max_model_calls=6, token_budget=500, observation_budget=900,
        compute_budget=2000, execution_seconds=40., confirmation_rows=0)
    plan = repair.protocol(failed_conditional_source, online_limits=online, author_limits=author)
    assert plan['online']['limits']['token_budget'] == 300
    assert plan['online']['limits']['observation_budget'] == 792
    assert plan['online']['limits']['confirmation_rows'] == 96
    assert plan['method']['author_limits'] == asdict(author)
    assert plan['method']['author_total_limits'] == asdict(author)
    assert plan['method']['prior_author_phases'][-1]['costs']['reported_tokens'] == 19754849


@pytest.fixture
def failed_native_source(failed_conditional_source):
    source = failed_conditional_source
    plan = read_json(source / 'protocol.json')
    plan.pop('conditional_design')
    case = deepcopy(plan['practice_cases']['design_case_006'])
    case.pop('evaluation_seed', None)
    while len(plan['practice_cases']) < 12:
        plan['practice_cases'][f"design_case_{len(plan['practice_cases']):03d}"] = deepcopy(case)
    old_ids = set(plan['old_target_ids'])
    plan['transfer_cases'] = ([c for c in plan['transfer_cases'] if c['id'] in old_ids]
        + [c for c in plan['transfer_cases'] if c['id'] not in old_ids][:4])
    plan.update(study='luna_native_research_method_compilation',
        source_model='_evaluated_model/older_model.py',
        online_research_experience='earlier_confirmed_online',
        author_limits={**plan['author_total_limits'], 'max_model_calls': 32,
                       'token_budget': 12_000_000})
    write_json(source / 'protocol.json', plan)
    author = read_json(source / 'author/training.json')
    author['practice_trials'] = [{**deepcopy(author['practice_trials'][0]),
        'case': 'design_case_011', 'directory': f'trials/trial_{index:03d}'}
        for index in range(175)]
    write_json(source / 'author/training.json', author)
    old = read_json(source / 'author/trials/trial_095/result.json')
    selected = source / 'author/trials/trial_174'
    old['workspace'] = str(selected)
    write_json(selected / 'result.json', old)
    return source


def test_native_failed_practice_keeps_twelve_cases_and_fourteen_targets(failed_native_source):
    source = failed_native_source
    old = read_json(source / 'protocol.json')
    author = read_json(source / 'author/training.json')
    plan = repair.protocol(source, trial='trial_174')
    assert plan['restored_practice_trials'] == 175
    assert plan['practice_index'] == 174
    assert plan['target_episodes'] == 14
    assert plan['method']['practice_cases'] == old['practice_cases']
    assert len(plan['method']['practice_cases']) == 12
    assert plan['method']['transfer_cases'] == old['transfer_cases']
    assert len(plan['method']['old_target_ids']) == 10
    case, = plan['online']['cases']
    assert case['confirmation_seed'] == old['practice_cases']['design_case_011']['confirmation_seed'] + 174
    assert case['evaluation_seed'] == case['confirmation_seed'] + 60_000_043
    assert case['evaluation_seed'] not in {c['evaluation_seed'] for c in old['transfer_cases']}
    assert 'evaluation_seed' not in plan['method']['practice_cases']['design_case_011']
    assert plan['online_allowance']['max_model_calls'] == 32
    assert plan['online_allowance']['token_budget'] == 12_000_000
    assert plan['online_allowance']['observation_budget'] == 4096
    assert plan['online']['limits']['observation_budget'] == 4096 + 192
    assert plan['online']['limits']['compute_budget'] == 2_000_000 + 840
    assert plan['method']['author_limits'] == old['author_limits']
    assert plan['method']['prior_author_phases'][-1]['costs'] == author['costs']
    assert read_json(source / 'protocol.json') == old
    assert read_json(source / 'author/training.json') == author


@pytest.mark.parametrize('changes', [
    {'status': 'model_call_budget_exhausted'}, {'model': None},
    {'confirmation': False}, {'terminal': False},
])
def test_native_online_failure_cannot_enter_compilation(
        tmp_path, failed_native_source, monkeypatch, changes):
    plan = repair.protocol(failed_native_source, trial='trial_174')
    plan['online']['runtime_snapshot'] = []
    monkeypatch.setattr(repair, 'run_study', lambda part, output, **kwargs:
                        completed_online(part, output, **changes))

    def forbidden(*args, **kwargs):
        raise AssertionError('The twelve-case author requires a confirmed source first.')

    monkeypatch.setattr(repair, 'run_compilation', forbidden)
    monkeypatch.setattr(repair, 'run_revision', forbidden)
    result = repair.run_pipeline(plan, tmp_path / 'native_failed', provider_factory=forbidden)
    assert result['method_published'] is False
    assert result['target_episodes_completed'] == 0
    assert result['target_episodes_pending'] == 14


@pytest.mark.parametrize('published', [False, True])
def test_native_confirmed_research_uses_current_compiler_and_source_model(
        tmp_path, failed_native_source, monkeypatch, published):
    source = failed_native_source
    plan = repair.protocol(source, trial='trial_174')
    plan['online']['runtime_snapshot'] = []
    old = read_json(source / 'protocol.json')
    output = tmp_path / 'native_pipeline'

    def forbidden(*args, **kwargs):
        raise AssertionError('Current compilation must not use the legacy 56-target driver.')

    def compile_method(part, destination, **kwargs):
        assert part['practice_cases'] == old['practice_cases']
        assert part['transfer_cases'] == old['transfer_cases']
        assert part['source'] == old['source']
        assert part['continued_author'] == str(source / 'author/training.json')
        assert part['online_research_experience'] == str(output / 'online/design_case_011/hegel')
        assert part['source_model'] == '_evaluated_model/current_model.py'
        assert part['source_model'] != old['source_model']
        assert part['previous_confirmed_source']['workspace'] == old['online_research_experience']
        assert part['previous_confirmed_source']['model'] == old['source_model']
        assert part['source_research_costs'] == costs(288, 2, 30, compute=936, seconds=6.5)
        assert part['prior_author_phases'][-1]['costs'] == costs(96, 2, 30, compute=96, seconds=1.)
        assert part['prior_author_phases'][-1]['usage']['known_tokens'] == 30
        assert destination == output / 'method'
        return {'method_published': published,
            'retention': [{}] * (10 if published else 0),
            'new_parameters': [{}] * (4 if published else 0)}

    monkeypatch.setattr(repair, 'run_study', lambda part, destination, **kwargs:
        completed_online(part, destination, model='_evaluated_model/current_model.py'))
    monkeypatch.setattr(repair, 'run_compilation', compile_method)
    monkeypatch.setattr(repair, 'run_revision', forbidden)
    result = repair.run_pipeline(plan, output, provider_factory=forbidden)
    assert result['status'] == ('completed' if published else 'method_unpublished')
    assert result['target_episodes_completed'] == (14 if published else 0)
    assert result['target_episodes_pending'] == (0 if published else 14)
    assert result['online_additional_costs'] == costs(96, 2, 30, compute=96, seconds=1.)
    assert read_json(source / 'protocol.json') == old


def completed_online(plan, output, *, status='completed', model='model.py',
                     confirmation=True, terminal=True):
    case = plan['cases'][0]
    workspace = output / case['id'] / 'hegel'
    extra = costs(96, 2, 30, compute=96, seconds=1.)
    cumulative = costs(288, 2, 30, compute=936, seconds=6.5)
    record = {'status': status, 'workspace': str(workspace), 'model': model,
              'costs': cumulative, 'prior_research_cost': costs(192, compute=840, seconds=5.5),
              'continuation_cost': extra,
              'events': [{'action': 'finish', 'usage': {'total_tokens': 30}}],
              'fresh_confirmation': {'passed': confirmation, 'mse': 0. if confirmation else 1., 'rows': 96},
              'outgoing_knowledge': {'files': {'online_only.py': 'learned fixture predictor'},
                                     'lessons': [], 'question_traces': []}}
    write_json(workspace / 'training.json', record)
    write_json(workspace / 'archive.json', record['outgoing_knowledge'])
    evaluation = {'passed': terminal, 'mse': 0. if terminal else 1.}
    write_json(workspace / 'evaluation.json', evaluation)
    summary = {'status': 'completed', 'trials': [{
        'case': case['id'], 'arm': 'hegel', 'status': status, 'model': model,
        'costs': cumulative, 'continuation_cost': extra,
        'fresh_confirmation_passed': confirmation, 'terminal': evaluation}]}
    write_json(output / 'summary.json', summary)
    return summary


@pytest.mark.parametrize('changes', [
    {'status': 'token_budget_exhausted'}, {'model': None},
    {'confirmation': False}, {'terminal': False},
])
def test_incomplete_online_research_never_enters_method_authoring(
        tmp_path, failed_conditional_source, monkeypatch, changes):
    plan = repair.protocol(failed_conditional_source)
    plan['online']['runtime_snapshot'] = []
    old_author = read_json(failed_conditional_source / 'author/training.json')
    calls = []

    def online(part, output, **kwargs):
        calls.append('online')
        return completed_online(part, output, **changes)

    def forbidden(*args, **kwargs):
        raise AssertionError('Unconfirmed online research cannot compile or transfer a method.')

    monkeypatch.setattr(repair, 'run_study', online)
    monkeypatch.setattr(repair, 'run_revision', forbidden)
    result = repair.run_pipeline(plan, tmp_path / 'pipeline', provider_factory=forbidden)
    assert result['status'] in {'online_research_incomplete', 'online_confirmation_failed'}
    assert calls == ['online']
    assert read_json(failed_conditional_source / 'author/training.json') == old_author


@pytest.mark.parametrize('published', [False, True])
def test_only_confirmed_online_experience_enters_method_phase_with_new_costs(
        tmp_path, failed_conditional_source, monkeypatch, published):
    source = failed_conditional_source
    plan = repair.protocol(source)
    plan['online']['runtime_snapshot'] = []
    old_plan = read_json(source / 'protocol.json')
    old_author = read_json(source / 'author/training.json')
    old_knowledge = read_json(source / 'initial_archive.json')
    order = []
    output = tmp_path / 'pipeline'

    def provider(*args, **kwargs):
        raise AssertionError('This wiring test must not call an LLM.')

    def online(part, destination, **kwargs):
        assert kwargs['provider_factory'] is provider
        assert kwargs['knowledge'] == old_knowledge
        assert callable(kwargs['prepare_prefix'])
        order.append('online_confirmed')
        return completed_online(part, destination)

    def compile_method(part, destination, **kwargs):
        assert order == ['online_confirmed']
        assert kwargs['provider_factory'] is provider
        assert part['practice_cases'] == old_plan['practice_cases']
        assert part['transfer_cases'] == old_plan['transfer_cases']
        assert part['old_target_ids'] == old_plan['old_target_ids']
        assert part['source'] == old_plan['source']
        assert part['continued_author'] == str(source / 'author/training.json')
        assert part['online_research_experience'] == str(output / 'online/design_case_006/hegel')
        assert part['prior_author_phases'][:-1] == plan['method']['prior_author_phases']
        new = part['prior_author_phases'][-1]
        assert new['phase'] == 'online_source_research'
        assert new['costs'] == costs(96, 2, 30, compute=96, seconds=1.)
        assert new['costs']['observation_cost'] != 288
        assert new['usage']['known_tokens'] == 30
        order.append('compile_method')
        frozen = {name: [{}] * (count if published else 0) for name, count in
                  {'original': 10, 'revised': 22, 'random_choice': 12, 'variance_choice': 12}.items()}
        result = {'method_published': published, 'frozen': frozen,
                  'unavailable_target_episodes': 0 if published else 56}
        write_json(destination / 'summary.json', result)
        return result

    monkeypatch.setattr(repair, 'run_study', online)
    monkeypatch.setattr(repair, 'run_revision', compile_method)
    result = repair.run_pipeline(plan, output, provider_factory=provider)
    assert order == ['online_confirmed', 'compile_method']
    assert result['status'] == ('completed' if published else 'method_unpublished')
    assert result['target_episodes_completed'] == (56 if published else 0)
    assert result['target_episodes_pending'] == (0 if published else 56)
    assert result['online_additional_costs'] == costs(96, 2, 30, compute=96, seconds=1.)
    assert read_json(source / 'protocol.json') == old_plan
    assert read_json(source / 'author/training.json') == old_author
    assert read_json(source / 'initial_archive.json') == old_knowledge


@pytest.fixture
def exhausted_online_source(tmp_path, failed_conditional_source, monkeypatch):
    plan = repair.protocol(failed_conditional_source)
    plan['online']['runtime_snapshot'] = []
    monkeypatch.setattr(repair, 'run_study', lambda part, output, **kwargs:
        completed_online(part, output, status='token_budget_exhausted',
                         model=None, confirmation=False, terminal=False))
    output = tmp_path / 'previous_online'
    repair.run_pipeline(plan, output, provider_factory=lambda workspace: None)
    return output


def test_finished_online_rescue_retains_cumulative_account_and_paid_lineage(exhausted_online_source):
    source = exhausted_online_source
    old_plan = read_json(source / 'protocol.json')
    old_summary = read_json(source / 'summary.json')
    previous = repair.read_json(old_summary['online_training'])
    plan = repair.rescue_protocol(source)
    assert plan['resume_result'] == old_summary['online_training']
    assert plan['source_study'] == old_plan['source_study']
    assert plan['practice_result'] == old_plan['practice_result']
    assert plan['restored_practice_trials'] == 99
    assert plan['online']['cases'] == old_plan['online']['cases']
    assert plan['online']['limits']['max_model_calls'] == previous['costs']['model_calls'] + 32
    assert plan['online']['limits']['token_budget'] == previous['costs']['reported_tokens'] + 4_000_000
    assert plan['online']['limits']['observation_budget'] == previous['costs']['observation_cost'] + 4096
    assert plan['online']['limits']['confirmation_mse_threshold'] == 1e-6
    assert plan['method']['practice_cases'] == old_plan['method']['practice_cases']
    assert plan['method']['transfer_cases'] == old_plan['method']['transfer_cases']
    phases = plan['method']['prior_author_phases']
    assert phases[:-1] == old_plan['method']['prior_author_phases']
    assert phases[-1]['costs'] == previous['continuation_cost']
    assert phases[-1]['costs']['observation_cost'] == 96  # not cumulative 288
    assert phases[-1]['usage']['known_tokens'] == 30
    assert read_json(source / 'protocol.json') == old_plan
    assert read_json(source / 'summary.json') == old_summary


def test_rescue_uses_online_restore_without_replaying_initial_practice(
        tmp_path, exhausted_online_source, monkeypatch):
    plan = repair.rescue_protocol(exhausted_online_source)
    plan['online']['runtime_snapshot'] = []
    previous = repair.read_json(plan['resume_result'])
    output = tmp_path / 'rescue'
    restored = []

    class Agent:
        workspace = output / 'online' / 'design_case_006' / 'hegel'

        def restore_practice_research(self, *args, **kwargs):
            raise AssertionError('The online account must not restart from trial095.')

        def restore_research(self, state, source, lab, *, confirmation_seed):
            assert state == previous
            assert source == plan['resume_workspace']
            assert confirmation_seed == plan['online']['cases'][0]['confirmation_seed']
            restored.append(state)
            return {'new_observations': 0, 'costs_before': state['costs']}

        def result(self, status):
            return {**deepcopy(previous), 'status': status}

    def online(part, destination, **kwargs):
        Agent.workspace.mkdir(parents=True, exist_ok=True)
        prefix = kwargs['prepare_prefix'](Agent(), object(), part['cases'][0])
        assert prefix['prior_research_cost'] == previous['costs']
        assert prefix['resume_origin'] == plan['resume_result']
        summary = completed_online(part, destination, status='token_budget_exhausted',
                                   model=None, confirmation=False, terminal=False)
        path = Agent.workspace / 'training.json'
        training = read_json(path)
        training['events'] = previous['events'] + training['events']
        training['costs'] = {key: value + training['continuation_cost'][key]
                             for key, value in previous['costs'].items()}
        write_json(path, training)
        return summary

    def forbidden(*args, **kwargs):
        raise AssertionError('An unconfirmed rescue cannot author or run the 56 targets.')

    monkeypatch.setattr(repair, 'run_study', online)
    monkeypatch.setattr(repair, 'run_revision', forbidden)
    result = repair.run_pipeline(plan, output, provider_factory=forbidden)
    assert restored == [previous]
    assert result['online_prior_costs'] == previous['costs']
    assert result['online_cumulative_costs']['observation_cost'] == 384
    assert result['online_additional_costs']['observation_cost'] == 96
    assert result['online_prior_events'] == 1 and result['online_added_events'] == 1
    assert result['target_episodes_completed'] == 0
    assert result['target_episodes_pending'] == 56
    assert read_json(output / 'restoration.json')['restored_model_calls'] == 2
    assert repair.read_json(plan['resume_result']) == previous


@pytest.fixture
def interrupted_online_source(exhausted_online_source, monkeypatch):
    """An interrupted third paid call follows two completed online decisions."""
    source = exhausted_online_source
    summary = read_json(source / 'summary.json')
    previous = repair.read_json(summary['online_training'])
    recovered = deepcopy(previous)
    recovered['status'] = 'runtime_interrupted'
    recovered['costs']['model_calls'] += 1
    recovered['costs']['reported_tokens'] += 11
    recovered['costs']['python_execution_seconds'] += 2.
    recovered['events'].append({'step': 2, 'action': 'run',
        'arguments': {'code': 'print("pending paid computation")'},
        'feedback': {'status': 'runtime_interrupted', 'execution_seconds': 2.},
        'usage': {'total_tokens': 11}})
    recovered.pop('continuation_cost', None)
    (source / 'summary.json').unlink()
    write_json(source / 'operator_pause.json', {'status': 'stopped_for_repair',
        'interrupted_python_seconds': 2., 'executed_path': 'scratch_001.py'})
    recoveries = []

    def recover(workspace, *, model, interruption, result_path):
        assert str(workspace) == previous['workspace']
        assert interruption['interrupted_python_seconds'] == 2.
        recoveries.append(model)
        write_json(result_path, recovered)
        return deepcopy(recovered)

    monkeypatch.setattr(repair, 'recover_interrupted_online_run', recover)
    return source, recovered, recoveries


def test_interrupted_plan_keeps_limits_and_accounts_pending_paid_call_once(interrupted_online_source):
    source, recovered, recoveries = interrupted_online_source
    old = read_json(source / 'protocol.json')
    plan = repair.interrupted_protocol(source)
    assert plan['online']['limits'] == old['online']['limits']
    assert plan['online_allowance'] == old['online_allowance']
    assert plan['online']['cases'] == old['online']['cases']
    assert plan['method']['practice_cases'] == old['method']['practice_cases']
    assert plan['method']['transfer_cases'] == old['method']['transfer_cases']
    assert plan['method']['old_target_ids'] == old['method']['old_target_ids']
    assert plan['target_episodes'] == 56
    assert repair.read_json(plan['resume_result']) == recovered
    phases = plan['method']['prior_author_phases']
    assert phases[:-1] == old['method']['prior_author_phases']
    assert phases[-1]['costs'] == costs(96, 3, 41, compute=96, seconds=3.)
    assert phases[-1]['usage']['known_tokens'] == 41
    assert phases[-1]['costs']['observation_cost'] != recovered['costs']['observation_cost']
    assert repair.interrupted_protocol(source) == plan
    assert recoveries == ['gpt-6-luna']
    assert read_json(source / 'protocol.json') == old


def test_interrupted_older_config_adopts_run_slice_without_granting_new_budget(interrupted_online_source):
    source, _, _ = interrupted_online_source
    old = read_json(source / 'protocol.json')
    old['online']['limits'].pop('run_timeout_seconds')
    write_json(source / 'protocol.json', old)
    plan = repair.interrupted_protocol(source)
    assert plan['online']['limits'] == {
        **old['online']['limits'], 'run_timeout_seconds': 300.}


def test_interrupted_pipeline_restores_paid_decision_and_waits_for_confirmation(
        tmp_path, interrupted_online_source, monkeypatch):
    source, recovered, _ = interrupted_online_source
    plan = repair.interrupted_protocol(source)
    plan['online']['runtime_snapshot'] = []
    output = tmp_path / 'continued_interrupted'
    restored = []

    class Agent:
        workspace = output / 'online' / 'design_case_006' / 'hegel'

        def restore_practice_research(self, *args, **kwargs):
            raise AssertionError('Do not restart the interrupted research from trial095.')

        def restore_research(self, previous, origin, lab, *, confirmation_seed):
            assert previous == recovered
            assert origin == plan['resume_workspace']
            assert confirmation_seed == plan['online']['cases'][0]['confirmation_seed']
            restored.append(previous)
            return {'new_observations': 0, 'costs_before': previous['costs']}

        def result(self, status):
            return {**deepcopy(recovered), 'status': status}

    def online(part, destination, **kwargs):
        Agent.workspace.mkdir(parents=True)
        prefix = kwargs['prepare_prefix'](Agent(), object(), part['cases'][0])
        assert prefix['prior_research_cost'] == recovered['costs']
        new = costs(calls=1, tokens=7)
        continued = deepcopy(recovered)
        continued.update(status='model_call_budget_exhausted', continuation_cost=new)
        continued['costs'] = {key: value + new[key] for key, value in recovered['costs'].items()}
        continued['events'].append({'step': recovered['costs']['model_calls'],
            'action': 'write', 'arguments': {'path': 'revised.py', 'code': '# revised by fixture'},
            'feedback': {'written': 'revised.py'}, 'usage': {'total_tokens': 7}})
        write_json(Agent.workspace / 'training.json', continued)
        return {'trials': [{'case': part['cases'][0]['id'], 'terminal': {'passed': False}}]}

    def forbidden(*args, **kwargs):
        raise AssertionError('No method publication or target experiment before independent confirmation.')

    monkeypatch.setattr(repair, 'run_study', online)
    monkeypatch.setattr(repair, 'run_revision', forbidden)
    result = repair.run_pipeline(plan, output, provider_factory=forbidden)
    assert restored == [recovered]
    assert result['online_prior_costs'] == recovered['costs']
    assert result['online_additional_costs'] == costs(calls=1, tokens=7)
    assert result['online_cumulative_costs']['observation_cost'] == recovered['costs']['observation_cost']
    assert result['online_cumulative_costs']['reported_tokens'] == 48
    assert result['target_episodes_completed'] == 0
    assert result['target_episodes_pending'] == 56
    assert read_json(output / 'restoration.json')['restored_model_calls'] == 3


def test_real_interrupted_restore_continues_after_paid_reply_without_repurchase(
        tmp_path, failed_conditional_source, monkeypatch):
    from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
    from test_common_archive_resume import execute

    # This isolated pipeline starts from an empty paid account; its ordinary
    # query and pending CLI reply below are recovered through the real runtime.
    source = failed_conditional_source
    original = read_json(source / 'author/trials/trial_095/result.json')
    original['costs'] = costs()
    write_json(source / 'author/trials/trial_095/result.json', original)
    old = repair.protocol(source, online_limits=InquiryLLMConfig(
        max_model_calls=3, token_budget=1000, observation_budget=200,
        execution_seconds=60., confirmation_rows=96))
    old['online']['runtime_snapshot'] = []
    interrupted = tmp_path / 'real_interrupted'
    write_json(interrupted / 'protocol.json', old)
    case = old['online']['cases'][0]
    workspace = interrupted / 'online' / case['id'] / 'hegel'
    lab = repair.laboratory(case)
    agent = CommonArchiveInquiryAgent(None, workspace, arm='hegel', source_task=False,
        config=InquiryLLMConfig(**old['online']['limits']),
        confirmation_seed=case['confirmation_seed'],
        knowledge=read_json(source / 'initial_archive.json'))
    execute(agent, lab, 'query', {'actions': lab.confirmation_actions(2, Random(71))})
    write_json(workspace / 'active.json', agent.result('researching'))
    write_json(workspace / 'research_context.json', agent._request(lab))
    write_json(workspace / 'code_action_002.json', {'action': 'run', 'arguments_json': '{}',
        'source_code': "print('interrupted computation')", 'research_notes': 'Inspect this computation.'})
    (workspace / 'code_call_002.jsonl').write_text(json.dumps({'type': 'turn.completed',
        'usage': {'input_tokens': 100, 'output_tokens': 25, 'cached_input_tokens': 80}}) + '\n')
    pending_path = f'scratch_{len(agent.events):03d}.py'
    (workspace / pending_path).write_text("print('interrupted computation')")
    write_json(interrupted / 'operator_pause.json', {'status': 'stopped_for_repair',
        'interrupted_python_seconds': 2., 'executed_path': pending_path})
    plan = repair.interrupted_protocol(interrupted, model='gpt-6-astra')
    assert plan['online']['model'] == plan['method']['model'] == 'gpt-6-astra'
    assert old['online']['model'] == 'gpt-6-luna'
    recovered = read_json(Path(plan['resume_result']))
    assert recovered['events'][-1]['usage']['model'] == 'gpt-6-luna'
    plan['online']['runtime_snapshot'] = []

    class Provider:
        def next_action(self, request):
            assert self.calls == 2
            self.calls += 1
            assert request['events'][-1]['feedback']['status'] == 'runtime_interrupted'
            assert request['events'][-1]['usage']['model'] == 'gpt-6-luna'
            assert request['costs']['reported_tokens'] == 126
            assert len(request['observations']) == 2
            return {'action': 'write', 'arguments': {'path': 'revised.py', 'code': '# revised'},
                    'usage': {'backend': 'scripted_runtime_check', 'total_tokens': 1}}

    factory = repair.laboratory

    def no_repurchase(case):
        restored_lab = factory(case)

        def forbidden(*args, **kwargs):
            raise AssertionError('Paid observations and the interrupted run must not be replayed.')

        restored_lab.execute = forbidden
        return restored_lab

    monkeypatch.setattr(repair, 'laboratory', no_repurchase)
    result = repair.run_pipeline(plan, tmp_path / 'real_continued', provider_factory=lambda _: Provider())
    assert result['online_additional_costs'] == costs(calls=1, tokens=1)
    assert result['online_cumulative_costs']['model_calls'] == 3
    assert result['online_cumulative_costs']['reported_tokens'] == 127
    assert result['online_cumulative_costs']['observation_cost'] == 2
    assert result['target_episodes_completed'] == 0
    assert result['target_episodes_pending'] == 56


@pytest.mark.parametrize('builder, fixture_name', [
    ('protocol', 'failed_conditional_source'),
    ('rescue_protocol', 'exhausted_online_source'),
    ('interrupted_protocol', 'interrupted_online_source'),
])
def test_source_evaluation_metadata_matches_generated_reset_trajectories(
        request, builder, fixture_name):
    source = request.getfixturevalue(fixture_name)
    if isinstance(source, tuple):
        source = source[0]
    old = read_json(source / 'protocol.json')
    evaluation = old['evaluation'] if builder == 'protocol' else old['online']['evaluation']
    evaluation['trajectory_length'] = 24  # Earlier protocol's descriptive value.
    write_json(source / 'protocol.json', old)

    plan = getattr(repair, builder)(source)
    case = plan['online']['cases'][0]
    declared = plan['online']['evaluation']
    rows = repair.laboratory(case).evaluation_rows(length=declared['rows'],
                                                  seed=case['evaluation_seed'])

    assert declared['trajectory_length'] == max(len(row.history) + 1 for row in rows) == 64
    assert len(rows) == declared['rows'] == evaluation['rows']
    assert declared['mse_threshold'] == 1e-6
    assert case['evaluation_seed'] == old.get('online', plan['online'])['cases'][0]['evaluation_seed']
    assert read_json(source / 'protocol.json') == old


def test_interrupted_cli_uses_recorded_reasoning_effort_for_online_and_method(tmp_path, monkeypatch):
    source, output = tmp_path / 'stopped', tmp_path / 'continued'
    plan = {stage: {'provider': {'reasoning_effort': 'high'}} for stage in ('online', 'method')}
    monkeypatch.setattr(repair.sys, 'argv', ['repair', '--resume-interrupted-study', str(source),
                                          '--output', str(output)])

    def restore(path, **kwargs):
        assert path == source
        assert kwargs == {'model': 'gpt-6-luna', 'timeout_seconds': 1800.}
        return plan

    constructed = []

    def provider(workspace, **kwargs):
        constructed.append((workspace, kwargs))
        return object()

    def pipeline(actual_plan, destination, *, provider_factory):
        assert actual_plan is plan and destination == output
        for stage in ('online', 'method'):
            assert plan[stage]['provider']['auth_mode'] == 'test_subscription'
            provider_factory(output / stage)
        return {'status': 'online_research_incomplete', 'method_published': False,
                'target_episodes_completed': 0, 'target_episodes_pending': 56}

    monkeypatch.setattr(repair, 'interrupted_protocol', restore)
    monkeypatch.setattr(repair, 'verify_codex_subscription', lambda: 'test_subscription')
    monkeypatch.setattr(repair, 'CodexCodeProvider', provider)
    monkeypatch.setattr(repair, 'run_pipeline', pipeline)
    repair.main()
    assert constructed == [(output / stage, {'model': 'gpt-6-luna',
        'reasoning_effort': 'high', 'timeout_seconds': 1800.}) for stage in ('online', 'method')]
