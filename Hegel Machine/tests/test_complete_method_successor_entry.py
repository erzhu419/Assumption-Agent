"""Confirmed source gating and frozen-method registration; no physical target runs."""

from copy import deepcopy

import pytest

from scripts import review_paid_object_pool, validate_partition_successor as runner
from scripts.run_partial_role_parent import load, save
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate
from test_transfer_partitions import observations, proposal


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    source, output = tmp_path / 'source', tmp_path / 'successor'
    body = proposal()['candidates'][0]['research_program']
    body['transfer_partition']['second_stage'] = {
        'operation': 'resolve_relation_intervals', 'compute_limit': 15000}
    baseline = deepcopy(body)
    baseline['transfer_partition'] = {'kind': 'Pi3', 'max_attempts': 48,
        'alternatives': [{'fit_terms': [body['graph']['nodes'][key]['answer']
            for key in body['term_nodes']]} for _ in range(2)]}
    save(source / 'methods.json', {'Pi3': baseline, 'complete': body})
    save(source / 'baseline.json', {'id': 'actual:oldPi3', 'execution': {
        'research_program_ref': runner.ref(source / 'methods.json', 'Pi3')}})
    task = {'seed': 11, 'objects': 4, 'apparatus_seed': 11, 'name_seed': 3,
        'wiring_seed': 5, 'actuator_curvature': .11, 'relation_direction': 'left',
        'variant': 'exchange', 'profile': 'confirmation', 'readout': 'window_total_event',
        'relation_radius': .9}
    save(source / 'task.json', task)
    method_ref, selected_ref = runner.ref(source / 'methods.json', 'complete'), runner.ref(source / 'selected.json')
    save(source / 'registration.json', {'source_task_ref': runner.ref(source / 'task.json'),
        'research_method_ref': method_ref, 'quality_threshold': 1e-7,
        'training_rows': 96, 'active_wall_seconds': 1800})
    save(source / 'receipt.json', {'execution_complete': True, 'full_comparison': True,
        'selected_ref': selected_ref, 'executed_method_ref': method_ref, 'selected_loss': 1e-21})
    save(source / 'confirmation.json', {'status': 'confirmed', 'selected_ref': selected_ref,
        'source_task_ref': runner.ref(source / 'task.json')})
    save(source / 'costs.json', {'cumulative_protocol_observations': 4737})
    save(source / 'allocation.json', {'allocation_id': 'complete_successor',
        'source_frontier_ref': runner.ref(source / 'retained_frontier.json'),
        'maximum_compute': 90000, 'new_observation_limit': 168,
        'protocol_observation_limit': 8000})
    entries = [{'candidate': Candidate(str(index), Program('const', value=float(index % 2)), 'old')}
        for index in range(262)]
    refs = [(runner.ref(source / f'old{index}.json'), entry) for index, entry in enumerate(entries)]
    monkeypatch.setattr(review_paid_object_pool, 'parent_inputs', lambda _: (
        {'selected_ref': selected_ref}, {}, observations(count=120),
        [{'rows': 24} for _ in range(5)], refs, entries, {}, {}))
    monkeypatch.setattr(runner, 'freeze_entry', lambda *args: None)
    request = {'source': str(source), 'seed': 106110501, 'prior_source_task_refs': [],
        'confirmation_ref': runner.ref(source / 'confirmation.json'),
        'baseline_method_ref': runner.ref(source / 'methods.json', 'Pi3'),
        'baseline_artifact_ref': runner.ref(source / 'baseline.json'),
        'source_model_ref': runner.ref(source / 'model.json'),
        'allocation_ref': runner.ref(source / 'allocation.json'),
        'source_costs_ref': runner.ref(source / 'costs.json'),
        'prior_method_review': str(source / 'review')}
    path = tmp_path / 'request.json'
    save(path, request)
    return source, output, path, request, baseline, body, task


@pytest.mark.parametrize('change', [
    {'status': 'confirmation_failed'}, {'selected_ref': {'file': 'another_model.json'}},
    {'source_task_ref': {'file': 'another_source.json'}}])
def test_failed_or_unrelated_confirmation_does_not_register(prepared, change):
    source, output, request, *_ = prepared
    save(source / 'confirmation.json', {**load(source / 'confirmation.json'), **change})
    with pytest.raises(ValueError, match='Confirm this same frozen source'):
        runner.register_complete_method_successor(request, output)
    assert not output.exists()


def test_actual_complete_body_and_full_pool_are_frozen_without_target_scores(prepared):
    source, output, request, supplied, baseline, body, task = prepared
    cfg = runner.register_complete_method_successor(request, output)
    assert load(output / 'frozen_methods.json') == {'P0': baseline, 'Pi1': body}
    assert load(source / 'methods.json') == {'Pi3': baseline, 'complete': body}
    assert cfg['source_pool_size'] == 262
    pool = load(output / 'source_pool.json')
    assert len(pool['scores']) == 262 and not pool['target_scores_available']
    assert all(set(score) == {'candidate_ref'} for score in pool['scores'])
    assert cfg['native_method_artifact_refs']['P0'] == supplied['baseline_artifact_ref']
    assert cfg['research_config']['max_candidates'] == 4
    assert cfg['confirmation_seed'] == supplied['seed'] + 1000003
    assert cfg['task'] == {**task, 'seed': supplied['seed'], 'apparatus_seed': supplied['seed'],
        'relation_radius': runner.task_settings(supplied['seed'])['relation_radius']}
    assert cfg['method_quotes']['Pi1']['geometry_compute'] == 96 * 3 * 2
    assert cfg['equal_method_compute_limit'] == max(q['required_compute'] for q in cfg['method_quotes'].values())
    assert cfg['total_new_compute_upper'] < cfg['total_compute_limit'] == 90000
    assert cfg['common_comparison_upper'] == 2 * 120
    assert cfg['previous_observation_cost'] == 4737


@pytest.mark.parametrize('change,reason', [
    ({'seed': 11}, 'already in the retained source cohort'),
    ({'maximum_compute': 1}, 'Fund the complete frozen pair'),
    ({'protocol_observation_limit': 4800}, 'unchanged protocol observation budget')])
def test_registration_does_not_repeat_sources_or_increase_limits(prepared, change, reason):
    source, output, request, supplied, *_ = prepared
    if 'seed' in change:
        save(request, {**supplied, **change})
    else:
        save(source / 'allocation.json', {**load(source / 'allocation.json'), **change})
    with pytest.raises(ValueError, match=reason):
        runner.register_complete_method_successor(request, output)
    assert not output.exists()


@pytest.mark.parametrize('denied', [False, True])
def test_completed_negative_search_gets_readout_but_denied_search_does_not(tmp_path, monkeypatch, denied):
    from dataclasses import dataclass
    from hegel_machine.cognition import anonymous_object_lab
    from hegel_machine.cognition.models import Observation, Probe, TaskSpec
    from hegel_machine.cognition.object_continuation import encode_entry

    save(tmp_path / 'old.json', encode_entry({'candidate': Candidate('old',
        Program('const', value=0.), 'retained'), 'kind': 'retained', 'family_id': 'T03'}))
    save(tmp_path / 'pool.json', {'scores': [{'candidate_ref': runner.ref(tmp_path / 'old.json')}]})
    save(tmp_path / 'model.json', {'raw_sensor_features': ['x'], 'raw_control_features': ['u']})
    save(tmp_path / 'frozen_methods.json', {'P0': {}, 'Pi1': {
        'transfer_partition': {'second_stage': {'operation': 'resolve_relation_intervals'}}}})
    save(tmp_path / 'registration.json', {'source': 'synthetic', 'remaining_compute': 10000,
        'previous_compute': 0, 'total_compute_limit': 10000,
        'source_pool_ref': runner.ref(tmp_path / 'pool.json'),
        'source_model_ref': runner.ref(tmp_path / 'model.json'), 'research_config': {},
        'task': {}, 'active_wall_seconds': 60, 'quality_threshold': 1e-7,
        'operations': {'P0': 'program_replay', 'Pi1': 'program_replay'},
        'equal_method_compute_limit': 500, 'per_arm_comparison_reserve': 480,
        'confirmation_rows': 48, 'confirmation_seed': 42, 'previous_observation_cost': 0,
        'previous_measurement_cost': 0, 'source_cost_refs': []})
    monkeypatch.setattr(runner, 'ROOT', (tmp_path / 'runtime').resolve())

    @dataclass
    class Batch:
        observations: list
        transitions: list
        channels: tuple = ('fine',)

    class Lab:
        def __init__(self, **kwargs):
            self.observation_cost = self.measurement_cost = self.queries = 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        @property
        def costs(self):
            return {'observation_cost': self.observation_cost, 'measurement_cost': self.measurement_cost,
                'queries': self.queries}
        def spec_for(self, channels):
            return TaskSpec('toy', 'test', ('x',) if channels else ('u',), (), ())
        def random_probe(self, seed, *, length, name):
            return Probe(name, tuple({'u': float(i)} for i in range(length)))
        def acquire(self, probe, channels):
            self.observation_cost += len(probe.sequence)
            self.queries += 1
            return Batch([Observation({'x': point['u']}, 1.) for point in probe.sequence], [])

    monkeypatch.setattr(anonymous_object_lab, 'AnonymousObjectLab', Lab)
    def replay(method, rows, budget, **kwargs):
        budget.spend(budget.limit + 1 if denied else 5)
        return {'candidate': Candidate('negative', Program('const', value=.5), 'replay'),
            'kind': 'replay', 'family_id': 'T03', 'training_loss': .25,
            'transfer_partition_search': {'completed': True},
            'transfer_partition_followup': {'status': 'first_stage_training_unexplained'}}
    monkeypatch.setattr(runner, 'replay_research_program', replay)
    confirmations = []
    def confirm(*args, **kwargs):
        confirmations.append(True)
        assert load(tmp_path / 'Pi1/outcome.json')['followup_prerequisite_unmet']
        return {'status': 'synthetic_completed_readout', 'shared_actual_costs': {
            'observation_cost': 48, 'measurement_cost': 0, 'compute_cost': 96}}
    monkeypatch.setattr(runner, 'confirm_pair', confirm)
    receipt = runner.run(tmp_path)
    assert bool(confirmations) is not denied
    assert receipt['arms']['Pi1']['execution_complete'] is not denied
