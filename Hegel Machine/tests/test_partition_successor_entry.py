"""A synthetic successor exercises budgets, the full pool and frozen readout ordering."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import validate_partition_successor as runner
from scripts.run_partial_role_parent import load, save
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, Probe, TaskSpec
from hegel_machine.cognition.object_continuation import encode_entry
from test_transfer_partitions import proposal


def test_allocation_keeps_equal_available_caps_without_reserving_unused_pi0_search():
    baseline = proposal()['baseline_research_program']
    quote = runner.allocation(70000, 32, baseline)
    assert quote['total_new_compute_upper'] == 70000
    assert quote['common_comparison_upper'] == 32 * 120
    assert quote['per_arm_comparison_reserve'] == 4 * 120
    assert quote['baseline_fit_upper'] == 5 * 96
    assert quote['equal_method_compute_limit'] > 60000
    with pytest.raises(ValueError):
        runner.allocation(100, 32, baseline)
    reserved = runner.allocation(70000, 34, baseline, training_rows=32, working_rows=16,
        method_return_reserve=100)
    assert reserved['total_new_compute_upper'] == 70000
    assert reserved['method_return_compute_limit'] == 100
    assert reserved['common_comparison_upper'] == 34 * 48


def test_register_waits_for_source_completion_without_creating_target(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'target'
    save(source / 'receipt.json', {'status': 'compute_unresolved'})
    save(source / 'registration.json', {})
    with pytest.raises(ValueError, match='second segment'):
        runner.register(source, target)
    assert not target.exists()


def test_paid_source_screen_keeps_history_and_stops_without_method_labels(tmp_path, monkeypatch):
    from dataclasses import asdict
    source, ledger, output = (tmp_path / name for name in ('old', 'ledger', 'screen'))
    monkeypatch.setattr(runner, 'freeze_entry', lambda *args: None)
    report_ref = runner.ref(source / 'report.json')
    save(source / 'registration.json', {'quality_threshold': 1e-7})
    save(source / 'prepared_state.json', {'source_references': {
        'report': report_ref, 'training': [runner.ref(source / 'report.json', 'batches', 0)],
        'validation': runner.ref(source / 'report.json', 'batches', 1)}})
    rows = [Observation({'x': 5.}, 2., ({'x': 2.},)), Observation({'x': 8.}, 5., ({'x': 5.},))]
    save(source / 'report.json', {'batches': [{}, {}, {'observations': [asdict(row) for row in rows]}]})
    candidate_ref = runner.ref(source / 'selected.json')
    save(source / 'selected.json', encode_entry({'candidate': Candidate('old',
        Program('lag', name='x', lag=1), 'retained')}))
    save(source / 'challenger/outcome.json', {'execution_complete': True,
        'full_comparison': True, 'selected_ref': candidate_ref})
    save(ledger / 'receipt.json', {'cumulative_compute': 100, 'remaining_compute': 50})
    result = runner.check_paid_source(source, ledger, output, 2)
    assert result['status'] == 'retained_winner_explains_paid_batch'
    assert result['mse'] == 0. and result['predictions'] == [2., 5.]
    assert result['cumulative_compute'] == 102 and result['remaining_compute'] == 48
    assert result['new_fits'] == result['new_observations'] == result['method_labels_created'] == 0
    assert runner.check_paid_source(source, ledger, output, 2)['new_compute_cost'] == 0
    with pytest.raises(ValueError, match='Reuse the paid score'):
        runner.check_paid_source(source, ledger, tmp_path / 'unneeded', 0)


@pytest.mark.parametrize('short,interrupted,solved,preparation', [
    (False, False, False, False), (False, True, False, False),
    (True, False, False, False), (True, False, True, False),
    (False, False, False, True), (False, False, True, True)])
def test_same_full_pool_and_development_feedback_precede_one_frozen_confirmation(tmp_path, monkeypatch, short, interrupted, solved, preparation):
    # All responses below are synthetic; this test never constructs the
    # registered physical lab or inspects the new successor's outcomes.
    from dataclasses import dataclass
    from scripts import run_learned_structure_successor as existing
    from hegel_machine.cognition import anonymous_object_lab

    x = Program('feature', name='x')
    correct = Program('add', children=(Program('mul', children=(x, x)), Program('const', value=.2)))
    old = [correct if solved else Program('const', value=0.), x, Program('const', value=1.)]
    source_scores = []
    for index, program in enumerate(old):
        location = tmp_path / 'old' / f'{index}.json'
        save(location, encode_entry({'candidate': Candidate(f'old{index}', program, 'retained'),
            'kind': 'retained', 'family_id': 'T03', 'training_loss': 1.}))
        source_scores.append({'candidate_ref': runner.ref(location), 'loss': 1.})
    save(tmp_path / 'old_pool.json', {'scores': source_scores, 'full_comparison': True})
    save(tmp_path / 'old_model.json', {'raw_sensor_features': ['x'], 'raw_control_features': ['u']})
    save(tmp_path / 'frozen_methods.json', {'P0': {'name': 'old'}, 'Pi1': {'name': 'new'}})
    cfg = {'source': 'synthetic_source', 'remaining_compute': 10000, 'previous_compute': 180000,
        'total_compute_limit': 270010, 'source_pool_ref': runner.ref(tmp_path / 'old_pool.json'),
        'source_model_ref': runner.ref(tmp_path / 'old_model.json'), 'research_config': {'max_candidates': 4},
        'task': {}, 'active_wall_seconds': 60, 'quality_threshold': 1e-7,
        'operations': {'P0': 'program_replay', 'Pi1': 'program_replay'},
        'equal_method_compute_limit': 5000, 'per_arm_comparison_reserve': 480,
        'confirmation_rows': 48, 'confirmation_seed': 42,
        'previous_observation_cost': 243, 'previous_measurement_cost': 2916,
        'source_cost_refs': [{'file': 'old/receipt.json'}]}
    if short:
        cfg.update(task={'seed': runner.CONTROL_SEED}, training_rows=32,
            training_probe_rows=[16, 16], working_rows=16, prior_method_review='prior_review',
            retained_method_pair_ref={'file': 'prior_review/pair.json'})
    if preparation:
        cfg.update(evaluation_role='paid_development_preparation',
            canonical_source='frozen_pair', knowledge_source='complete_current_pool')
    save(tmp_path / 'registration.json', cfg)
    monkeypatch.setattr(runner, 'ROOT', (tmp_path / 'runtime').resolve())
    calls = []
    returns = []
    if short:
        from scripts import return_partition_pair
        def record_return(source, output, prior_review, retained_pair=None):
            assert load(source / 'receipt.json')['confirmation']['status'] == 'frozen_pair_readout_completed'
            assert retained_pair == cfg['retained_method_pair_ref']
            returns.append((source, output, prior_review))
            return {'status': 'returned_existing_result', 'new_compute_cost': 20}
        monkeypatch.setattr(return_partition_pair, 'return_pair', record_return)

    @dataclass
    class Batch:
        observations: list
        transitions: list
        channels: tuple = ('fine',)

    class Lab:
        def __init__(self, **kw):
            self.observation_cost = self.measurement_cost = self.queries = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        @property
        def costs(self):
            return {'observation_cost': self.observation_cost,
                    'measurement_cost': self.measurement_cost, 'queries': self.queries}

        def spec_for(self, channels):
            return TaskSpec('synthetic', 'Synthetic public measurements', ('x',) if channels else ('u',), (), ())

        def random_probe(self, seed, *, length, name):
            return Probe(name, tuple({'u': -1. + 2. * i / length} for i in range(length)))

        def acquire(self, probe, channels):
            if len(probe.sequence) == 48:
                assert len(calls) == 2
                assert (tmp_path / 'confirmation/frozen_predictors.json').exists()
            rows = [Observation({'x': point['u']}, point['u'] ** 2 + .2) for point in probe.sequence]
            self.observation_cost += len(rows)
            self.measurement_cost += 12 * len(rows)
            self.queries += 1
            return Batch(rows, [])

    monkeypatch.setattr(anonymous_object_lab, 'AnonymousObjectLab', Lab)
    monkeypatch.setattr(existing, 'AnonymousObjectLab', Lab)

    def replay(payload, training, budget, **kwargs):
        calls.append((deepcopy(payload), len(training), len(kwargs['feedback_rows']), budget.limit))
        assert budget.spend(5)
        if interrupted:
            raise RuntimeError('synthetic interruption after paid work')
        return {'candidate': Candidate(payload['name'], correct if payload['name'] == 'new' else x, 'replay'),
            'kind': 'replay', 'family_id': 'T03', 'training_loss': 1.,
            'transfer_partition_search': {'completed': True},
            'transfer_partition_followup': {'execution_complete': True}}

    monkeypatch.setattr(runner, 'replay_research_program', replay)
    if interrupted:
        with pytest.raises(RuntimeError, match='synthetic interruption'):
            runner.run(tmp_path)
        failed = load(tmp_path / 'failure.json')
        assert failed['compute_known'] == 3 * 120 + 5
        assert failed['observation_cost'] == 120
        assert not (tmp_path / 'confirmation').exists()
        return
    receipt = runner.run(tmp_path)
    count, work = (32, 16) if short else (96, 24)
    if preparation:
        assert calls == [] and receipt['arms'] == {} and receipt['confirmation'] is None
        assert receipt['status'] == ('common_pool_already_explained' if solved else 'paid_development_prepared')
        assert receipt['full_comparison'] and receipt['candidate_occurrences'] == 3
        assert len(receipt['scores']) == receipt['unique_programs'] == 3
        assert receipt['selected_mse'] == min(item['mse'] for item in receipt['scores'])
        assert [item['candidate_refs'][0] for item in receipt['scores']] == [
            item['candidate_ref'] for item in source_scores]
        assert receipt['compute_cost'] == 3 * (count + work)
        assert receipt['observation_cost'] == count + work
        assert receipt['method_labels_created'] == 0
        assert len(receipt['training_refs']) == 4 and len(receipt['feedback_refs']) == 1
        assert not (tmp_path / 'confirmation').exists()
        return
    if solved:
        assert calls == [] and receipt['arms'] == {}
        assert returns == []
        assert receipt['status'] == 'common_pool_already_explained'
        assert receipt['confirmation'] is None
        assert receipt['observation_cost'] == count + work
        assert not (tmp_path / 'confirmation').exists()
        return
    assert calls == [({'name': 'old'}, count, 0, 5000), ({'name': 'new'}, count, work, 5000)]
    assert receipt['source_pool_size'] == 3
    assert receipt['observation_cost'] == count + work + 48
    assert receipt['cumulative_observation_cost'] == 243 + count + work + 48
    assert receipt['measurement_cost'] == 12 * (count + work + 48)
    assert receipt['confirmation']['arms']['Pi1']['passed']
    assert not receipt['confirmation']['arms']['P0']['passed']
    assert receipt['compute_cost'] == 4 * (count + work) + 10 + 96
    common = load(tmp_path / 'common_comparison.json')
    assert [item['candidate_ref'] for item in common['scores']] == [item['candidate_ref'] for item in source_scores]
    assert len(common['data_refs']) == (3 if short else 5)
    assert not receipt['independent_architecture_support']
    if short:
        assert len(returns) == 1
        assert receipt['method_return']['status'] == 'returned_existing_result'
        assert 'method_return' not in load(tmp_path / 'receipt.json')


def test_control_preserves_canonical_bodies_all_prior_entries_and_original_limits(tmp_path, monkeypatch):
    source, original, knowledge, review, output = [tmp_path / name for name in
        ('positive', 'original', 'knowledge', 'review', 'control')]
    bodies = proposal()
    p0, p1 = bodies['baseline_research_program'], bodies['candidates'][0]['research_program']
    p1['transfer_partition']['second_stage'] = {'operation': 'resolve_relation_intervals'}
    frozen = {'P0': p0, 'Pi1': p1, 'canonical_source_ref': {'file': 'first/frozen.json'}}
    task = {'seed': runner.SEED, 'apparatus_seed': runner.SEED, 'name_seed': 11,
        'wiring_seed': 12, 'actuator_curvature': .17, 'relation_radius': .9}
    save(source / 'registration.json', {'original_source': str(original),
        'operations': {'P0': 'program_replay', 'Pi1': 'program_replay'},
        'source_model_ref': {'file': 'retained/model.json'}, 'task': task,
        'research_config': {'seed': runner.SEED}, 'quality_threshold': 1e-7,
        'observation_limit': 512, 'measurement_limit': 6144,
        'total_compute_limit': 270010})
    save(source / 'receipt.json', {'status': 'paired_methods_completed',
        'arms': {'P0': {'execution_complete': True}, 'Pi1': {'execution_complete': True}},
        'cumulative_observation_cost': 411, 'cumulative_measurement_cost': 4932})
    save(source / 'frozen_methods.json', frozen)
    save(original / 'registration.json', {'knowledge_root': str(knowledge),
        'historical_knowledge_ref': {'store': 'synthetic'}, 'task': {'seed': 1}})
    save(knowledge / 'registration.json', {'source_task_ref': runner.ref(knowledge / 'lab.json')})
    save(knowledge / 'lab.json', {'relation_radius': 1.16})
    old = [{'candidate_ref': {'file': f'old/{index}.json'}, 'loss': .5} for index in range(32)]
    save(source / 'common_comparison.json', {'scores': old, 'full_comparison': True})
    for arm in ('P0', 'Pi1'):
        save(source / arm / 'comparison.json', {'full_comparison': True, 'new_scores': [
            {'candidate_ref': runner.ref(source / arm / 'new_entry.json'), 'loss': .1}]})
    save(review / 'receipt.json', {'cumulative_compute': 213300, 'remaining_compute': 56710})

    class History:
        def __init__(self, *args, **kwargs):
            self.connection = self
        def execute(self, *args):
            return []
        def close(self):
            pass

    monkeypatch.setattr(runner, 'HistoryStore', History)
    monkeypatch.setattr(runner, 'freeze_entry', lambda *args: None)
    cfg = runner.register_control(source, review, output)
    assert load(output / 'frozen_methods.json') == frozen
    assert cfg['source_pool_size'] == 34
    assert load(output / 'source_pool.json')['scores'][:32] == old
    assert cfg['task']['relation_radius'] == 1.16
    assert cfg['task']['seed'] == cfg['task']['apparatus_seed'] == runner.CONTROL_SEED
    assert all(cfg['task'][key] == task[key] for key in ('name_seed', 'wiring_seed', 'actuator_curvature'))
    assert cfg['previous_compute'] == 213300 and cfg['remaining_compute'] == 56710
    assert cfg['total_new_compute_upper'] == 56710 and cfg['total_compute_limit'] == 270010
    assert cfg['method_return_compute_limit'] == 100
    assert cfg['training_probe_rows'] == [16, 16] and cfg['working_rows'] == 16
    assert cfg['confirmation_rows'] == 48 and cfg['quality_threshold'] == 1e-7
    assert cfg['retained_method_pair_ref'] == runner.ref(review / 'pair.json')
