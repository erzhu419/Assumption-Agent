"""Recurrent method adapters with scripted failures, not discovery evidence.

The fixture contains no fitted scientific solution. Its zero predictor must fail
paid evidence, so these checks establish archive, physics and publication wiring.
"""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.anonymous_method_learning import (
    AnonymousResearchMethodLearner, EMPTY_METHOD,
)
from hegel_machine.cognition.anonymous_research_archive import load_research_archive
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts.run_anonymous_method_learning import (
    frozen_transfers, laboratory, protocol, run_learning, score_frozen, trial_preparer,
)
from test_anonymous_method_learning import action, read_json, write_json


ZERO_METHOD = """import json
def select_actions(spec, records, memory):
    steps = json.load(open('research_state.json'))['method_results']
    if not steps:
        return {'action': 'write', 'arguments': {
            'path': 'zero.py', 'code': 'def predict(row):\\n    return 0.0\\n'}}
    if len(steps) == 1:
        return {'model': 'zero.py'}
    return {'stop': 'The scripted zero predictor contradicts paid evidence.'}
"""


@pytest.fixture
def recurrent_source(tmp_path):
    root = tmp_path / 'sources'
    cases = [{'id': f'case_{index:03d}', 'kind': 'recurrent', 'seed': seed,
              'apparatus_seed': seed + 37, 'prefix_seed': seed + 100003,
              'confirmation_seed': seed + 200003, 'evaluation_seed': seed + 300007}
             for index, seed in enumerate((71221031, 71222040))]
    write_json(root / 'protocol.json', {'cases': cases, 'arms': ['code_react', 'hegel']})
    for case in cases:
        for arm in ('code_react', 'hegel'):
            workspace = root / case['id'] / arm
            write_json(workspace / 'training.json', {
                'workspace': str(workspace), 'status': 'model_call_budget_exhausted',
                'model': None, 'events': [], 'costs': {'model_calls': 1},
            })
            write_json(workspace / 'archive.json', {
                'files': {'notes.md': f"Scripted unresolved source {case['id']} {arm}."},
                'lessons': [], 'question_traces': [],
            })
            # Terminal labels must not enter the author archive, even if present.
            write_json(workspace / 'evaluation.json', {'private_terminal_marker': 991827})
    return root


def method_memory(source):
    knowledge, sources = load_research_archive(source)
    knowledge['files'][METHOD_PATH] = EMPTY_METHOD
    return knowledge, sources


def test_recurrent_protocol_preserves_sources_and_changes_actual_apparatus(recurrent_source):
    original = read_json(recurrent_source / 'protocol.json')
    plan = protocol(recurrent_source, model='scripted-interface-only')
    practice = list(plan['practice_cases'].values())
    targets = plan['transfer_cases']
    assert len(practice) == 4 and len(targets) == 6
    assert len(plan['live_control_cases']) == 2
    assert plan['evaluation']['trajectory_length'] == 24
    assert '24 independent three-step' in plan['prefix_protocol']
    source_seeds = {case['apparatus_seed'] for case in original['cases']}
    assert {case['apparatus_seed'] for case in practice[::2]} == source_seeds
    assert source_seeds.isdisjoint(case['apparatus_seed'] for case in practice[1::2])
    practice_seeds = {case['apparatus_seed'] for case in practice}
    assert practice_seeds.isdisjoint(case['apparatus_seed'] for case in targets)
    assert len({case['apparatus_seed'] for case in targets}) == 6
    assert {case['seed'] for case in practice}.isdisjoint(case['seed'] for case in targets)
    assert len({case['confirmation_seed'] for case in targets}) == 6
    assert len({case['evaluation_seed'] for case in targets}) == 6

    probe = {'kind': 'trajectory', 'channels': ['meter'],
             'sequence': [{'u': value} for value in (0.4, -0.7, 0.3, 0.6)]}
    source = laboratory(original['cases'][0]).execute(deepcopy(probe))
    returned = laboratory(practice[0]).execute(deepcopy(probe))
    changed = laboratory(practice[1]).execute(deepcopy(probe))
    assert source['rows'] == returned['rows']
    assert source['measurements'] == returned['measurements']
    assert source['rows'] != changed['rows']
    assert source['measurements'] != changed['measurements']

    knowledge, records = load_research_archive(recurrent_source)
    assert len(records) == len(knowledge['source_catalog']) == 4
    assert {(row['case'], row['arm']) for row in records} == {
        (case['id'], arm) for case in original['cases'] for arm in ('code_react', 'hegel')}
    assert all(row['status'] == 'model_call_budget_exhausted' for row in records)
    assert len([path for path in knowledge['files'] if path.endswith('notes.md')]) == 4
    assert 'private_terminal_marker' not in json.dumps(knowledge)


def test_recurrent_practice_buys_short_prefix_and_exposes_actual_contract(recurrent_source, tmp_path):
    plan = protocol(recurrent_source, model='scripted-interface-only')
    knowledge, sources = method_memory(recurrent_source)
    prepare = trial_preparer(knowledge, sources, tmp_path / 'reference_archive')
    case = plan['practice_cases']['case_001']
    payload = prepare(case, tmp_path / 'practice', EMPTY_METHOD,
                      InquiryLLMConfig(**plan['limits']))
    agent, lab = payload['agent'], payload['laboratory']
    assert isinstance(lab, TemporalInquiryLab)
    assert lab.public_spec()['max_probe_length'] == 24
    assert lab.public_spec()['channels'][0]['name'] == 'meter'
    prefix = agent.events[0]
    assert len(prefix['arguments']['actions']) == 24
    assert all(len(item['sequence']) == 3 and item['channels'] == []
               for item in prefix['arguments']['actions'])
    records = prefix['feedback']['observations']
    assert sum(len(record['rows']) for record in records) == 72
    assert all(not record['measurements'] for record in records)
    assert all([len(row['history']) for row in record['rows']] == [0, 1, 2]
               for record in records)
    assert agent.costs['observation_cost'] == 72
    assert agent.costs['measurement_cost'] == 0

    author = AnonymousResearchMethodLearner(None, tmp_path / 'author', knowledge=knowledge,
        cases=plan['practice_cases'], lab_factory=laboratory,
        config=InquiryLLMConfig(**plan['author_limits']),
        practice_limits=InquiryLLMConfig(**plan['limits']), prepare_trial=prepare)
    assert lab.task_interface in author.common_interface
    assert "channels:['meter']" in author.common_interface
    assert 'auxiliary readings and past response targets are unavailable' in author.common_interface
    assert 'three trajectories' not in author.common_interface
    assert 'Three paid 24-row' not in author.common_interface


def test_frozen_recurrent_program_receives_paid_refutation_without_llm(recurrent_source, tmp_path):
    plan = protocol(recurrent_source, model='scripted-interface-only')
    plan['transfer_cases'] = plan['transfer_cases'][:1]
    knowledge, sources = method_memory(recurrent_source)
    results = frozen_transfers(plan, tmp_path / 'frozen_study', knowledge, sources, ZERO_METHOD)
    _, result = results[0]
    assert result['status'] == 'method_stopped'
    assert result['model'] is None
    assert result['costs']['model_calls'] == result['costs']['reported_tokens'] == 0
    assert result['costs']['observation_cost'] == 72
    assert result['fresh_confirmation'] is None
    assert 'paid evidence contradicts' in result['method_results'][1]['feedback']['error']
    assert result['method_source'] == ZERO_METHOD
    scores = score_frozen(plan, results)
    assert scores[0]['terminal'] == {'status': 'no_submitted_model', 'mse': None, 'passed': False}


def test_failed_recurrent_practice_does_not_publish_or_start_consumers(recurrent_source, tmp_path):
    plan = protocol(recurrent_source, model='scripted-interface-only')
    plan['author_limits']['max_model_calls'] = 2
    output = tmp_path / 'failed_study'
    requested_workspaces = []

    class UnsuccessfulAuthor:
        def next_action(self, request):
            if not request['events']:
                return action('write', {'path': METHOD_PATH, 'code': ZERO_METHOD})
            feedback = request['events'][-1]['feedback']['edit_practice']
            assert feedback['status'] == 'method_stopped'
            return action('finish', {'path': METHOD_PATH, 'lesson': 'Unresolved scripted failure.'})

    def provider_factory(workspace):
        requested_workspaces.append(workspace)
        assert workspace.name == 'author'
        return UnsuccessfulAuthor()

    summary = run_learning(plan, output, provider_factory=provider_factory)
    assert requested_workspaces == [output / 'author']
    assert len(read_json(output / 'source_outcomes.json')) == 4
    assert not summary['method_published']
    assert summary['unavailable_frozen_targets'] == 6
    assert summary['unavailable_live_episodes'] == 4
    assert summary['frozen'] == [] and summary['live_controls'] is None
    assert summary['author_costs']['observation_cost'] >= 72
    published = read_json(output / 'published_method.json')
    assert not published['published'] and published['source'] is None
    assert not (output / 'transfer_archive.json').exists()
    assert not (output / 'frozen').exists() and not (output / 'controls').exists()
