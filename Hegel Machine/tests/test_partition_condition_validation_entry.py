"""Condition validation freezes actual proposals and preserves the rebound pool."""

from copy import deepcopy
from dataclasses import asdict, dataclass

import pytest

from scripts import validate_partition_successor as runner
from scripts.run_partial_role_parent import load, save
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, Probe, SearchBudget, TaskSpec
from hegel_machine.cognition.object_continuation import encode_entry, decode_entry
from test_transfer_partitions import proposal
from test_v8_paired_method_conditions import learn, pair


def fixture(tmp_path, monkeypatch):
    source, canonical, review, original, knowledge = [tmp_path / name for name in
        ('source103', 'canonical', 'review', 'original', 'knowledge')]
    proposed = proposal()
    methods = {'P0': proposed['baseline_research_program'],
               'Pi1': proposed['candidates'][0]['research_program']}
    methods['Pi1']['transfer_partition']['second_stage'] = {'compute_limit': 15000}
    group = learn([pair(0), pair(1), pair(2, stage='complete', challenger_cost=200)])
    for side, arm in [('baseline', 'P0'), ('challenger', 'Pi1')]:
        save(canonical / 'frozen_methods.json', methods)
        artifact = runner.ref(review / f'{side}.json')
        save(review / f'{side}.json', {'execution': {'entry': 'replay_research_program',
            'research_program_ref': runner.ref(canonical / 'frozen_methods.json', arm)}})
        group['baseline_method_artifact_ref' if side == 'baseline' else 'method_artifact_ref'] = artifact
    group['method_ref'] = {'id': 'method', 'version': 1}
    save(review / 'receipt.json', {'return': {'research_result': {
        'method_action_research': {'paired_condition_research': [group]}}}})
    task = {'seed': 71, 'objects': 4, 'relation_radius': .8, 'name_seed': 12,
            'wiring_seed': 13, 'actuator_curvature': .16}
    save(canonical / 'registration.json', {'original_source': str(original), 'task': task,
        'source_model_ref': runner.ref(canonical / 'model.json'), 'research_config': {},
        'operations': {'P0': 'program_replay', 'Pi1': 'program_replay'},
        'quality_threshold': 1e-7, 'active_wall_seconds': 1800})
    save(canonical / 'public_spec.json', {'feature_names': ['drive', 'zero']})
    save(canonical / 'model.json', {'raw_sensor_features': ['drive', 'zero']})
    save(original / 'registration.json', {'knowledge_root': str(knowledge),
        'historical_knowledge_ref': {'store': 'synthetic'}, 'task': {'seed': 72}})
    save(knowledge / 'registration.json', {'source_task_ref': runner.ref(knowledge / 'lab.json')})
    save(knowledge / 'lab.json', {'relation_radius': 1.16})
    save(source / 'frozen_methods.json', methods)
    mappings = [{'drive': 'current', 'zero': 'other'}, {'drive': 'other', 'zero': 'current'}]
    save(source / 'registration.json', {'feature_mappings': mappings})
    scores = []
    for index, program in enumerate([Program('feature', name='current'), Program('const', value=0.)]):
        entry = {'candidate': Candidate(f'old{index}', program, 'retained')}
        location = runner.ref(source / f'old{index}.json')
        save(source / f'old{index}.json', encode_entry(entry))
        scores.append({'candidate_ref': location, 'loss': .5})
    save(source / 'common_comparison.json', {'scores': scores, 'full_comparison': True})
    for arm in ('P0', 'Pi1'):
        location = runner.ref(source / arm / 'new.json')
        save(source / arm / 'new.json', encode_entry({'candidate': Candidate(arm,
            Program('feature', name='other'), 'retained')}))
        save(source / arm / 'comparison.json', {'full_comparison': True,
            'new_scores': [{'candidate_ref': location, 'loss': .1}]})
    allocations = {}
    for suffix, seed in [('held', 106030301), ('changed', 106030302)]:
        allocation_id = 'condition_validation_' + suffix
        allocations[allocation_id] = {'allocation_id': allocation_id, 'maximum_compute': 250000,
            'new_observation_limit': 168, 'seed': seed, 'training_rows': 96,
            'working_rows': 24, 'confirmation_rows': 48, 'factor': suffix}
    save(tmp_path / 'protocol.json', {'allocations': allocations})

    class History:
        def __init__(self, *a, **k):
            self.connection = self
        def execute(self, *a):
            return []
        def close(self):
            pass

    monkeypatch.setattr(runner, 'HistoryStore', History)
    monkeypatch.setattr(runner, 'freeze_entry', lambda *a: None)
    return source, canonical, review, group, methods


def test_missing_development_condition_does_not_register_or_read_world(tmp_path):
    review = tmp_path / 'review'
    save(review / 'receipt.json', {'return': {'research_result': {
        'method_action_research': {'paired_condition_research': []}}}})
    output = tmp_path / 'unstarted'
    result = runner.register_condition_validation(tmp_path / 'source', tmp_path / 'canonical',
        review, output, {'file': 'not_needed'})
    assert result['status'] == 'development_condition_required'
    assert result['new_compute_cost'] == result['new_observations'] == 0 and not output.exists()


def test_both_sources_freeze_same_conditions_all_candidates_and_separate_allowances(tmp_path, monkeypatch):
    source, canonical, review, group, methods = fixture(tmp_path, monkeypatch)
    registered = []
    for suffix in ('held', 'changed'):
        output = tmp_path / suffix
        cfg = runner.register_condition_validation(source, canonical, review, output,
            runner.ref(tmp_path / 'protocol.json', 'allocations', 'condition_validation_' + suffix))
        registered.append(cfg)
        assert cfg['source_pool_size'] == 8  # 4 actual source entries, both admitted maps
        entries = load(output / 'bound_source_pool.json')
        assert len({str(item['source_candidate_ref']) for item in entries}) == 4
        assert all(decode_entry(item)['candidate'].program.required_features() <= {'drive', 'zero'}
                   for item in entries)
        assert load(output / 'frozen_conditions.json') == [group]
        assert load(output / 'frozen_methods.json') == methods
        assert cfg['data_role'] == 'condition_validation'
        assert cfg['previous_compute'] == cfg['previous_observation_cost'] == 0
        assert cfg['total_compute_limit'] == cfg['total_new_compute_upper'] == 250000
        assert cfg['observation_limit'] == 168 and cfg['method_return_compute_limit'] == 2000
        assert cfg['training_probe_rows'] == [24] * 4 and cfg['working_rows'] == 24
        assert cfg['confirmation_rows'] == 48 and cfg['quality_threshold'] == 1e-7
    assert registered[0]['task']['relation_radius'] == 1.16
    assert registered[0]['task']['relation_radius'] != registered[1]['task']['relation_radius']
    for key in ('name_seed', 'wiring_seed', 'actuator_curvature'):
        assert registered[0]['task'][key] == registered[1]['task'][key]
    assert load(source / 'registration.json')['feature_mappings'][0]['drive'] == 'current'


def test_noninvertible_binding_cannot_silently_prune_source_pool(tmp_path, monkeypatch):
    source, canonical, review, _, _ = fixture(tmp_path, monkeypatch)
    save(source / 'registration.json', {'feature_mappings': [{'drive': 'current', 'zero': 'current'}]})
    with pytest.raises(ValueError, match='bijective'):
        runner.register_condition_validation(source, canonical, review, tmp_path / 'held',
            runner.ref(tmp_path / 'protocol.json', 'allocations', 'condition_validation_held'))
    assert not (tmp_path / 'held').exists()


def test_condition_predictions_are_frozen_before_any_method_result(tmp_path, monkeypatch):
    from scripts import return_partition_pair
    group = learn([pair(0), pair(1), pair(2, stage='complete', challenger_cost=200)])
    frozen = deepcopy(group)
    save(tmp_path / 'frozen_conditions.json', [group])
    monkeypatch.setattr(return_partition_pair, 'paid_pre_call_profile',
        lambda *a: pair(0)['baseline']['profile'])
    cfg = {'frozen_conditions_ref': runner.ref(tmp_path / 'frozen_conditions.json')}
    budget = SearchBudget(10000)
    result = runner.record_frozen_condition_prediction(tmp_path, cfg, {}, budget)
    assert result['before_method_execution'] and not result['ordinary_guard_admitted']
    assert result['compute_cost'] == budget.spent == len(group['candidates'])
    assert all(item['helpful'] for item in result['predictions'][0]['candidate_predictions'])
    assert load(tmp_path / 'frozen_conditions.json') == [frozen]
    assert not (tmp_path / 'P0').exists() and not (tmp_path / 'Pi1').exists()


def paid_validation_fixture(tmp_path, monkeypatch):
    from test_transfer_partitions import second_stage_fixture
    source, canonical, review, group, _ = fixture(tmp_path, monkeypatch)
    entry, training, feedback, _ = second_stage_fixture()
    pi1 = entry['research_program']
    pi3 = deepcopy(pi1)
    terms = runner.research_program_fit_contract(pi1)['fit_terms']
    pi3['transfer_partition'] = {'kind': 'Pi3', 'max_attempts': 2, 'max_candidates': 4,
        'alternatives': [{'fit_terms': [term.to_dict() for term in terms]}]}
    methods = {'Pi1': pi1, 'Pi3': pi3}
    names = {'baseline': 'Pi1', 'challenger': 'Pi3'}
    artifacts = {}
    for side, key in (('baseline', 'baseline_method_artifact_ref'), ('challenger', 'method_artifact_ref')):
        arm = names[side]
        artifacts[arm] = group[key]
        save(group[key]['file'], {'id': 'fixed:' + arm, 'version': 1, 'kind': 'native_research_segment',
            'execution': {'entry': 'replay_research_program',
                'research_program_ref': runner.ref(canonical / 'frozen_methods.json', arm)}})
    save(canonical / 'frozen_methods.json', methods)
    save(canonical / 'registration.json', {'evaluation_arm_names': names,
        'native_method_artifact_refs': artifacts, 'research_config': {}, 'active_wall_seconds': 60})
    addressed = runner.ref(review / 'addressed_condition.json')
    save(addressed['file'], group)
    frozen = runner.ref(tmp_path / 'frozen_condition.json')
    save(frozen['file'], {'condition_research_ref': addressed})
    # The explicit reference must win; no search of this later receipt may
    # substitute another family or decide which candidate to evaluate.
    save(review / 'receipt.json', {'return': {'research_result': {
        'method_action_research': {'paired_condition_research': []}}}})
    common = tmp_path / 'new_common'
    save(common / 'frozen_methods.json', methods)
    model_ref = runner.ref(common / 'model.json')
    features = tuple(training[0].features)
    save(model_ref['file'], {'raw_sensor_features': features, 'raw_control_features': ['control']})
    old_ref = runner.ref(common / 'old.json')
    save(old_ref['file'], encode_entry({'candidate': Candidate('old', Program('const', value=0.), 'retained')}))
    save(common / 'source_pool.json', {'scores': [{'candidate_ref': old_ref, 'loss': 0.}]})
    task_ref = runner.ref(common / 'registration.json', 'task')
    cohort = [task_ref, runner.ref(tmp_path / 'other_common/registration.json', 'task')]
    cfg = {'source': str(source), 'evaluation_role': 'paid_development_preparation',
        'frozen_condition_ref': frozen, 'canonical_source': str(canonical), 'knowledge_source': str(source),
        'source_task_ref': task_ref, 'source_model_ref': model_ref, 'source_cost_refs': [],
        'source_pool_ref': runner.ref(common / 'source_pool.json'), 'task': {'seed': 99},
        'remaining_compute': 200000, 'previous_compute': 0, 'total_compute_limit': 200000,
        'previous_observation_cost': 0, 'previous_measurement_cost': 0, 'active_wall_seconds': 60,
        'training_probe_rows': [len(training)], 'working_rows': len(feedback),
        'quality_threshold': 1e-7, 'research_config': {}, 'feature_mappings': [{}],
        'prior_method_review': str(review), 'retained_method_pair_ref': runner.ref(canonical / 'pair.json')}
    save(common / 'registration.json', cfg)
    return common, canonical, review, cohort, group, methods, training, feedback


def test_paid_condition_preparation_freezes_only_the_addressed_canonical_family(tmp_path, monkeypatch):
    common, canonical, review, cohort, group, methods, *_ = paid_validation_fixture(tmp_path, monkeypatch)
    result = runner.prepare_paid_condition_validation(common, review, cohort_refs=cohort)
    assert load(common / 'frozen_conditions.json') == [group]
    assert load(common / 'frozen_methods.json') == load(canonical / 'frozen_methods.json') == methods
    assert result['condition_research_ref'] == runner.ref(review / 'addressed_condition.json')
    assert result['condition_validation_cohort_refs'] == cohort
    assert result['evaluation_arm_names'] == {'baseline': 'Pi1', 'challenger': 'Pi3'}
    assert result['data_role'] == 'condition_validation'
    assert result['condition_prediction_compute_upper'] == len(group['candidates'])
    assert not (common / 'observations').exists() and not (common / 'condition_prediction.json').exists()


@pytest.mark.parametrize('mismatch', ['started', 'body', 'artifact', 'cohort'])
def test_paid_condition_preparation_rejects_late_or_different_experiments(tmp_path, monkeypatch, mismatch):
    common, _, review, cohort, group, methods, *_ = paid_validation_fixture(tmp_path, monkeypatch)
    if mismatch == 'started':
        save(common / 'started.json', {})
    elif mismatch == 'body':
        methods['Pi3']['transfer_partition']['max_attempts'] += 1
        save(common / 'frozen_methods.json', methods)
    elif mismatch == 'artifact':
        changed = load(group['method_artifact_ref']['file'])
        changed['id'] = 'different_method'
        location = runner.ref(review / 'different_artifact.json')
        save(location['file'], changed)
        group['method_artifact_ref'] = location
        save(review / 'addressed_condition.json', group)
    else:
        cohort = cohort[1:]
    with pytest.raises(ValueError):
        runner.prepare_paid_condition_validation(common, review, cohort_refs=cohort)
    assert not (common / 'frozen_conditions.json').exists()


@pytest.mark.parametrize('outside,solved', [(False, False), (True, False), (False, True)])
def test_new_common_forecast_precedes_both_methods_and_is_reused_unchanged(tmp_path, monkeypatch, outside, solved):
    from scripts import return_partition_pair
    from hegel_machine.cognition import anonymous_object_lab
    common, _, review, cohort, group, _, training, feedback = paid_validation_fixture(tmp_path, monkeypatch)
    runner.prepare_paid_condition_validation(common, review, cohort_refs=cohort)
    profile = deepcopy(pair(0)['baseline']['profile'])
    if outside:
        profile['research']['stage'] = 'previously_unseen_stage'

    def pre_call(source, cfg, compared):
        assert not (source / 'Pi1').exists() and not (source / 'Pi3').exists()
        assert cfg['equal_method_compute_limit'] == max(cfg['pair_quote']['method_compute_upper'].values())
        return profile
    monkeypatch.setattr(return_partition_pair, 'paid_pre_call_profile', pre_call)

    @dataclass
    class Batch:
        observations: list
        transitions: list

    class Lab:
        def __init__(self, **kwargs):
            self.observation_cost = self.measurement_cost = 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def spec_for(self, channels):
            return TaskSpec('new_source', 'Synthetic public interface',
                tuple(training[0].features) if channels else ('control',), (), ())
        def random_probe(self, seed, *, length, name):
            return Probe(name, tuple({'control': float(i)} for i in range(length)))
        def acquire(self, probe, channels):
            rows = training if probe.probe_id == 'train0' else feedback
            if solved:
                rows = [Observation(row.features, 0., row.history) for row in rows]
            self.observation_cost += len(rows)
            self.measurement_cost += len(rows) * len(rows[0].features)
            return Batch(rows, [])
    monkeypatch.setattr(anonymous_object_lab, 'AnonymousObjectLab', Lab)
    monkeypatch.setattr(runner, 'ROOT', common / 'runtime')
    monkeypatch.setattr(runner, 'replay_research_program', lambda *a, **kw: pytest.fail('common must stop before methods'))
    monkeypatch.setattr(runner, 'confirm_pair', lambda *a, **kw: pytest.fail('no world confirmation in this contract'))
    receipt = runner.run(common)
    predicted_ref = receipt['condition_prediction_ref']
    predicted = load(predicted_ref['file'])
    raw_prediction = (common / 'condition_prediction.json').read_bytes()
    assert predicted['before_method_execution'] and not predicted['ordinary_guard_admitted']
    assert predicted['predictions'][0]['public_coverage']['can_apply'] is not outside
    assert predicted['predictions'][0]['candidate_predictions']
    assert receipt['compute_cost'] == len(training) + len(feedback) + len(group['candidates'])
    assert receipt['arms'] == {} and receipt['confirmation'] is None
    output = tmp_path / 'paired'
    cfg = runner.register_paid_pair(common, output)
    if solved:
        assert cfg['status'] == 'paid_scene_already_explained_no_method_label'
        assert not output.exists()
        return
    assert cfg['condition_prediction_ref'] == predicted_ref
    assert cfg['frozen_conditions_ref'] == runner.ref(common / 'frozen_conditions.json')
    assert cfg['condition_validation_cohort_refs'] == cohort and cfg['data_role'] == 'condition_validation'
    assert cfg['equal_method_compute_limit'] == load(common / 'registration.json')['equal_method_compute_limit']
    calls = []
    def replay(body, rows, budget, **kwargs):
        assert (common / 'condition_prediction.json').read_bytes() == raw_prediction
        calls.append(body['transfer_partition']['kind'])
        budget.spend(1)
        return {'candidate': Candidate('same_old', Program('const', value=0.), 'retained'),
            'training_loss': 1., 'transfer_partition_search': {'completed': True},
            'transfer_partition_followup': {'status': 'first_stage_training_unexplained'}}
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    monkeypatch.setattr(runner, 'replay_research_program', replay)
    monkeypatch.setattr(runner, 'record_frozen_condition_prediction',
        lambda *a, **kw: pytest.fail('the original forecast is already paid'))
    monkeypatch.setattr(return_partition_pair, 'return_pair', lambda *a, **kw: {'status': 'returned'})
    result = runner.run_paid_pair(output)
    assert calls == ['Pi1', 'Pi3']
    assert result['condition_prediction_ref'] == predicted_ref
    assert result['data_role'] == 'condition_validation' and result['new_observations'] == 0
    assert result['compute_cost'] == 2
    assert (common / 'condition_prediction.json').read_bytes() == raw_prediction
