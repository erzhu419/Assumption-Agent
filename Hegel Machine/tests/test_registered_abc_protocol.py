"""Explicit ABC sources retain the declared limits through execution and billing."""

from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import run_partial_role_successors as runner
from scripts.run_partial_role_parent import load, save


def explicit():
    old = runner.program_successor_protocol({'seed': 7, 'objects': 4,
        'relation_radius': .9, 'actuator_curvature': .1})
    return {'tasks': [{**old['tasks'][2], 'seed': 106110601, 'apparatus_seed': 106110601}],
        'limits': {'compute_cost': 300000, 'observation_cost': 320, 'measurement_cost': 3840},
        'activity_seconds': 1800, 'terminal': deepcopy(old['terminal']),
        'research_config': {**old['research_config'], 'compute_budget': 300000, 'observation_budget': 320}}


def test_protocol_quotes_all_three_arms_and_shared_terminal_without_changing_old_four_cases():
    source = {'seed': 7, 'objects': 4, 'relation_radius': .9, 'actuator_curvature': .1}
    old = runner.program_successor_protocol(source)
    assert old['budget_quote']['total_new_compute_upper'] == 5762136
    assert old['budget_quote']['total_new_observation_upper'] == 7568
    protocol = explicit()
    current = runner.program_successor_protocol(source, protocol)
    assert current['tasks'] == protocol['tasks']
    assert current['budget_quote']['total_new_compute_upper'] == 900534
    assert current['budget_quote']['total_new_observation_upper'] == 1316
    assert current['budget_quote']['training_activity_seconds_upper'] == 5400
    protocol['tasks'].append({**protocol['tasks'][0], 'seed': 106110602, 'objects': 3})
    second = runner.program_successor_protocol(source, protocol)
    assert second['budget_quote']['terminal_compute_upper'] == 3 * (178 + 148)


@pytest.mark.parametrize('field', ['duplicate', 'arms', 'budget', 'threshold', 'drafts'])
def test_explicit_protocol_does_not_relax_frozen_scientific_or_resource_contracts(field):
    declared = explicit()
    if field == 'duplicate':
        declared['tasks'] *= 2
    elif field == 'arms':
        declared['arms'] = ['archive', 'pr3']
    elif field == 'budget':
        declared['research_config']['compute_budget'] = 480000
    elif field == 'threshold':
        declared['terminal']['mse_tolerance'] = .01
    else:
        declared['research_config']['max_candidates'] = 8
    with pytest.raises(ValueError):
        runner.program_successor_protocol({'seed': 7, 'objects': 4,
            'relation_radius': .9, 'actuator_curvature': .1}, declared)


def test_registration_consumes_fixed_protocol_and_checks_recent_source_refs(tmp_path, monkeypatch):
    import run_learned_structure_successor

    paths = {name: tmp_path / (name + '.json') for name in
             ('program_source', 'method', 'task_source', 'source_pool', 'source_costs', 'budget_contract', 'successor_protocol')}
    save(tmp_path / 'proposal.json', {'proposal': {'baseline_research_program': {'kind': 'old'}, 'candidates': []}})
    save(paths['program_source'], {'source_concept_id': 'old', 'source_ref': {'file': 'original.json'},
        'proposal_ref': {'file': str(tmp_path / 'proposal.json')}})
    save(paths['method'], {'research_program': {'kind': 'old'}})
    save(paths['task_source'], {'task': {'seed': 7, 'objects': 4, 'relation_radius': .9, 'actuator_curvature': .1},
        'historical_knowledge_ref': {'store': 'unused', 'unit': 'old'}})
    save(paths['source_pool'], {'scores': [{'candidate_ref': {'file': 'retained.json'}}]})
    save(paths['source_costs'], {'occurrences': []})
    save(paths['budget_contract'], {'allocations': {'abc': {'compute_cost': 900534, 'observation_cost': 1316}}})
    save(paths['successor_protocol'], explicit())
    args = SimpleNamespace(**paths, output=tmp_path / 'registered', knowledge_root=tmp_path / 'knowledge',
        raw_source=None, method_conditions=None, workers=1)
    monkeypatch.setattr(runner, 'program_knowledge', lambda *args:
        SimpleNamespace(to_dict=lambda: {'meta_priors': [{'source_concept_id': 'old'}]}))
    fake = SimpleNamespace(execute=lambda *args: [], close=lambda: None)
    monkeypatch.setattr(runner, 'HistoryStore', lambda *args, **kwargs:
                        SimpleNamespace(connection=fake, close=lambda: None))
    monkeypatch.setattr(run_learned_structure_successor, 'freeze_runtime', lambda *args: None)
    registered = runner.register_programs(args)
    assert registered['tasks'] == explicit()['tasks'] and registered['current_budget_admitted']
    assert registered['limits']['compute_cost'] == 300000
    assert registered['successor_protocol_ref'] == {'file': str(paths['successor_protocol'])}
    recent = tmp_path / 'recent.json'
    save(recent, explicit()['tasks'][0])
    save(paths['successor_protocol'], {**explicit(), 'prior_source_task_refs': [{'file': str(recent)}]})
    args.output = tmp_path / 'must_not_run'
    with pytest.raises(ValueError, match='already exposed'):
        runner.register_programs(args)
    assert not args.output.exists()


def test_failed_native_run_uses_registered_deadline_and_unknown_cost_bound(tmp_path, monkeypatch):
    registered = runner.program_successor_protocol({'seed': 7, 'objects': 4,
        'relation_radius': .9, 'actuator_curvature': .1}, explicit())
    registered['activity_seconds'] = 743
    monkeypatch.setattr(runner, '_SHARED', {'output': tmp_path, 'registration': registered,
        'knowledge': {}, 'raw': None, 'method': None})
    monkeypatch.setattr(runner.KnowledgeLibrary, 'from_dict', lambda data: object())

    class Agent:
        failure_evidence_ref, failure_compute_lower_bound = None, 13

        def __init__(self, config, **kwargs):
            assert config.compute_budget == 300000 and config.observation_budget == 320

        def run(self, laboratory):
            raise RuntimeError('retained failed attempt')

    class Lab:
        costs = {'observation_cost': 2, 'measurement_cost': 3}
        acquisitions = []

        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    deadlines = []

    @contextmanager
    def deadline(seconds, **kwargs):
        deadlines.append(seconds)
        yield

    monkeypatch.setattr(runner, 'CommonObjectResearchAgent', Agent)
    monkeypatch.setattr(runner, 'AnonymousObjectLab', Lab)
    monkeypatch.setattr(runner, 'python_execution_deadline', deadline)
    result = runner.train_one((106110601, 'archive'))
    summary = load(result['summary_ref'])
    assert summary['compute_unknown_upper_bound'] == 300000
    assert summary['confirmed_compute_lower_bound'] == 13
    assert deadlines == [743]
    assert load(tmp_path / 'runs/106110601/archive/failure.json')['compute_upper_bound'] == 300000


def test_interrupted_attempts_keep_custom_unknown_upper_bounds_without_reexecution(tmp_path):
    protocol = explicit()
    save(tmp_path / 'registration.json', {**protocol, 'method_kind': 'transfer_partition',
        'current_budget_admitted': True, 'source_costs_ref': {'file': 'old_bill.json'}})
    for arm in runner.ARMS:
        (tmp_path / 'runs/106110601' / arm).mkdir(parents=True)
    runner.run(tmp_path)
    summary = load(tmp_path / 'training_summary.json')
    assert len(summary['outcomes']) == 3
    assert all(item['unknown_upper_bounds'] == protocol['limits'] for item in summary['outcomes'])
