"""Online-to-method phase wiring; scripted outcomes are not discovery evidence."""

from copy import deepcopy
from dataclasses import asdict

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
