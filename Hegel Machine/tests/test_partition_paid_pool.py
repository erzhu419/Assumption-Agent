"""The paid gate retains late candidates and scores only compatible missing rows."""

from dataclasses import asdict

import pytest

from scripts import validate_partition_successor as runner
from scripts.run_partial_role_parent import load, save
from hegel_machine.cognition import parallel_prediction
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, TaskSpec
from hegel_machine.cognition.object_continuation import encode_entry


def setup_scene(tmp_path, monkeypatch):
    source, screen, output = [tmp_path / name for name in ('old', 'screen', 'pool')]
    monkeypatch.setattr(runner, 'freeze_entry', lambda *args: None)
    monkeypatch.setattr(runner, 'ROOT', (output / 'runtime').resolve())
    x = Program('feature', name='x')
    twice = Program('mul', children=(Program('const', value=2.), x))

    def entry(name, program):
        # This deliberately wrong training_loss must never be reused.
        return encode_entry({'candidate': Candidate(name, program, 'retained'), 'training_loss': 0.})

    entries = [entry('old_x', x), entry('late_twice', twice), entry('late_zero', Program('const', value=0.))]
    report = source / 'report.json'
    rows = [Observation({'x': 1.}, 3., ()), Observation({'x': 2.}, 4., ()),
        Observation({'u': 1.}, 100., ()), Observation({'x': 3.}, 5., ())]
    save(report, {'batches': [{'observations': [asdict(row)]} for row in rows],
        'object_continuation_state': {'selected': entries[1], 'finalists': [entries[0]],
            'candidate_frontiers': [{'entries': [entries[2]]}]}})
    working = runner.ref(report, 'batches', 1)
    save(source / 'prepared_state.json', {'source_references': {'report': runner.ref(report),
        'validation': working}, 'independent_validation': True,
        'spec': {'feature_names': ['x'], 'risk_quantile': .5}})
    save(source / 'evaluation_contract.json', {'role': 'paid_working_comparison', 'data_ref': working})
    for arm in ('baseline', 'challenger'):
        save(source / arm / 'entry.json', entries[0])
        save(source / arm / 'comparison.json', {'full_comparison': True, 'scores': [
            {'candidate_ref': runner.ref(source / arm / 'entry.json'), 'loss': 4.}]})
    save(screen / 'receipt.json', {'source': str(source), 'candidate_ref': runner.ref(source / 'challenger/entry.json'),
        'data_ref': runner.ref(report, 'batches', 3), 'mse': 4., 'threshold': 1e-7,
        'cumulative_compute': 100, 'remaining_compute': 50})
    return screen, output


def test_pool_reuses_actual_working_and_screen_scores_and_keeps_late_models(tmp_path, monkeypatch):
    screen, output = setup_scene(tmp_path, monkeypatch)
    cfg = runner.register_paid_pool(screen, output, tmp_path / 'topic')
    assert cfg['total_paid_rows'] == 3 and cfg['maximum_compute'] == 7
    assert len(cfg['candidate_refs']) == 5
    assert cfg['other_view_refs'][0]['rows'] == 1
    calls = []
    real = parallel_prediction.predict_rows
    monkeypatch.setattr(parallel_prediction, 'predict_rows', lambda program, rows:
        (calls.append(len(rows)) or real(program, rows)))
    result = runner.run_paid_pool(output)
    assert calls == [1, 3, 3]
    assert result['full_comparison'] and result['candidate_occurrences'] == 5
    assert result['unique_programs'] == 3 and result['compute_cost'] == 7
    assert result['scores'][0]['mse'] == 4.  # ignores unbound training_loss=0
    assert result['selected_mse'] == pytest.approx(2. / 3.)
    assert result['selected_ref']['json_path'] == ['object_continuation_state', 'selected']
    assert result['cumulative_compute'] == 107 and result['remaining_compute'] == 43
    assert result['method_review_ref'] == runner.ref(tmp_path / 'topic/common/questions.json')
    assert result['new_fits'] == result['new_observations'] == result['method_labels_created'] == 0
    assert runner.run_paid_pool(output)['new_compute_cost'] == 0 and calls == [1, 3, 3]


def test_failed_prediction_preserves_paid_partial_result_without_full_pool_claim(tmp_path, monkeypatch):
    screen, output = setup_scene(tmp_path, monkeypatch)
    runner.register_paid_pool(screen, output, tmp_path / 'topic')
    real = parallel_prediction.predict_rows
    count = 0

    def interrupted(program, rows):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError('actual failed prediction')
        return real(program, rows)

    monkeypatch.setattr(parallel_prediction, 'predict_rows', interrupted)
    with pytest.raises(RuntimeError, match='actual failed'):
        runner.run_paid_pool(output)
    failed = load(output / 'failure.json')
    assert not failed['full_comparison'] and len(failed['completed_scores']) == 1
    assert failed['compute_cost'] == 4 and failed['cumulative_compute'] == 104
    assert not (output / 'receipt.json').exists()


def test_current_bindings_keep_old_pool_reuse_scores_and_do_not_label_unfunded_pair(tmp_path, monkeypatch):
    screen, paid = setup_scene(tmp_path, monkeypatch)
    runner.register_paid_pool(screen, paid, tmp_path / 'topic')
    previous = runner.run_paid_pool(paid)
    state_path = tmp_path / 'old/prepared_state.json'
    state = load(state_path)
    state['source_references']['training'] = [runner.ref(tmp_path / 'old/report.json', 'batches', 0)]
    state['runtime_state'] = {'object_model_ref': runner.ref(tmp_path / 'model.json')}
    save(state_path, state)
    save(tmp_path / 'old/registration.json', {'source_task_ref': {'source_id': 'historical103'}})
    save(tmp_path / 'model.json', {})
    current, canonical, output = [tmp_path / name for name in ('current', 'canonical', 'bound')]
    public = TaskSpec('same_scene', 'Synthetic paid input', ('x',), (), ())
    save(current / 'public_spec.json', asdict(public))
    methods = {'P0': {'frozen': 'P0'}, 'Pi1': {'frozen': 'Pi1'}}
    save(canonical / 'frozen_methods.json', methods)
    save(canonical / 'registration.json', {'source_model_ref': runner.ref(tmp_path / 'model.json')})
    x = Program('feature', name='x')
    new = Program('mul', children=(Program('const', value=3.), x))
    location = runner.ref(current / 'candidate.json')
    save(current / 'candidate.json', encode_entry({'candidate': Candidate('current3x', new, 'retained')}))
    save(current / 'source_pool.json', {'scores': [{'candidate_ref': location}]})
    for arm in ('P0', 'Pi1'):
        save(current / arm / 'comparison.json', {'full_comparison': True,
            'new_scores': [{'candidate_ref': location}]})
    monkeypatch.setattr(runner, 'paid_bindings', lambda *args: ([{'x': 'x'}, {'x': 'x'}], public, {}))
    monkeypatch.setattr(runner, 'paid_pair_quote', lambda *args:
        {'total_new_compute_upper': 200, 'confirmation_compute': 0})
    monkeypatch.setattr(runner, 'ROOT', (output / 'runtime').resolve())
    cfg = runner.register_paid_binding_pool(paid, current, canonical, output)
    assert cfg['maximum_compute'] == 3 and cfg['bound_references'] == 2
    assert not cfg['method_pair_fundable_after_common_upper']
    assert load(output / 'frozen_methods.json') == methods
    calls = []
    real = parallel_prediction.predict_rows
    monkeypatch.setattr(parallel_prediction, 'predict_rows', lambda program, rows:
        (calls.append(program) or real(program, rows)))
    result = runner.run_paid_binding_pool(output)
    assert calls == [new]
    assert result['candidate_occurrences'] == 7 and result['unique_programs'] == 4
    assert result['reused_programs'] == 3 and result['new_programs'] == 1
    assert result['selected_ref'] == previous['selected_ref']
    assert result['compute_cost'] == 3 and result['remaining_compute'] == previous['remaining_compute'] - 3
    assert result['status'] == 'bound_pool_residual_pair_unfunded'
    assert result['new_fits'] == result['new_observations'] == result['method_labels_created'] == 0
    assert not result['independent_support']
    assert runner.run_paid_binding_pool(output)['new_compute_cost'] == 0 and len(calls) == 1


def test_paid_pair_quote_preserves_bodies_all_mappings_and_complete_followup():
    from copy import deepcopy
    from test_transfer_partitions import second_stage_fixture
    entry, training, feedback, _ = second_stage_fixture()
    pi1 = entry['research_program']
    p0 = deepcopy(pi1)
    p0['transfer_partition'] = {'kind': 'Pi0', 'released_groups': [], 'max_attempts': 1, 'max_candidates': 4}
    bodies = {'P0': p0, 'Pi1': pi1}
    before = deepcopy(bodies)
    identity = {name: name for name in training[0].features}
    quoted = runner.paid_pair_quote(bodies, [identity, identity], training, feedback,
        len(training) + len(feedback))
    assert bodies == before
    assert len(quoted['method_quotes']['Pi1']) == 2
    assert all(item['attempt_limit'] == 48 and item['followup_compute_upper'] == 15000
        for item in quoted['method_quotes']['Pi1'])
    assert quoted['total_new_compute_upper'] == sum(quoted['method_compute_upper'].values()) + 8 * 32 + 1000
    assert quoted['max_candidates'] == 4 and quoted['confirmation_compute'] == 0


def test_pair_quote_uses_actual_body_followup_instead_of_arm_label():
    from copy import deepcopy
    from test_transfer_partitions import second_stage_fixture
    entry, training, feedback, _ = second_stage_fixture()
    complete = entry['research_program']
    discrete = deepcopy(complete)
    terms = runner.research_program_fit_contract(complete)['fit_terms']
    discrete['transfer_partition'] = {'kind': 'Pi3', 'max_attempts': 2, 'max_candidates': 4,
        'alternatives': [{'fit_terms': [term.to_dict() for term in terms]}]}
    methods = {'parameter_method': complete, 'structure_method': discrete}
    result = runner.paid_pair_quote(methods, [{}], training, feedback,
        len(training) + len(feedback), evaluation_arm_names={
            'baseline': 'parameter_method', 'challenger': 'structure_method'})
    assert set(result['method_quotes']) == set(methods)
    assert result['method_quotes']['parameter_method'][0]['followup_compute_upper'] == 15000
    assert result['method_quotes']['structure_method'][0]['followup_compute_upper'] == 0


def test_separate_pair_allocation_preserves_old_balance_and_only_executes_paid_material(tmp_path, monkeypatch):
    from copy import deepcopy
    from scripts import return_partition_pair
    source, canonical, knowledge, output = [tmp_path / name for name in ('bound', 'canonical', 'knowledge', 'pair')]
    monkeypatch.setattr(runner, 'freeze_entry', lambda *args: None)
    monkeypatch.setattr(runner, 'ROOT', (output / 'runtime').resolve())
    public = TaskSpec('historical103', 'Synthetic paid source', ('x',), (), ())
    save(source / 'public_spec.json', asdict(public))
    bodies = {'P0': {'method': 'P0'}, 'Pi1': {'method': 'Pi1'}}
    save(source / 'frozen_methods.json', bodies)
    save(canonical / 'registration.json', {'research_config': {}, 'active_wall_seconds': 60})
    retained_pair = runner.ref(tmp_path / 'original_paid_pair.json')
    save(knowledge / 'registration.json', {'retained_method_pair_ref': retained_pair})
    batch = runner.ref(source / 'paid_batch.json')
    rows = [Observation({'x': 1.}, 2.), Observation({'x': 2.}, 4.)]
    save(source / 'paid_batch.json', {'observations': [asdict(row) for row in rows]})
    location = runner.ref(source / 'old_candidate.json')
    save(source / 'old_candidate.json', encode_entry({'candidate': Candidate('old', Program('const', value=0.), 'old')}))
    score = {'candidate_refs': [location], 'mse': 10., 'rows': 2}
    old_receipt = {'full_comparison': True, 'selected_mse': 10., 'cumulative_compute': 100,
        'remaining_compute': 20, 'scores': [score]}
    save(source / 'receipt.json', old_receipt)
    quote = {'total_new_compute_upper': 100, 'equal_method_compute_limit': 40,
        'per_arm_comparison_reserve': 8, 'method_return_compute_limit': 4}
    cfg = {'threshold': 1e-7, 'pair_quote': quote, 'canonical_source': str(canonical),
        'knowledge_source': str(knowledge), 'source_task_ref': {'source_id': 'original103'},
        'target_model_ref': runner.ref(source / 'model.json'),
        'method_review_ref': runner.ref(tmp_path / 'topic/common/questions.json'),
        'feature_mappings': [{'x': 'x'}, {'x': 'x'}], 'data_refs': [batch],
        'training_refs': [batch], 'feedback_refs': []}
    save(source / 'registration.json', cfg)
    unfunded = runner.register_paid_pair(source, output)
    assert unfunded['status'] == 'complete_paired_unit_unfunded_not_method_negative'
    assert not output.exists() and unfunded['method_labels_created'] == 0
    allocation = runner.ref(tmp_path / 'protocol.json', 'allocations', 'paid_source_103')
    save(tmp_path / 'protocol.json', {'allocations': {'paid_source_103': {
        'allocation_id': 'paid_source_103', 'maximum_compute': 100,
        'new_observation_limit': 0, 'source_pool_ref': runner.ref(source / 'receipt.json')}}})
    registered = runner.register_paid_pair(source, output, allocation_ref=allocation)
    assert registered['previous_compute'] == 0 and registered['remaining_compute'] == 100
    assert registered['original_ledger']['remaining_compute'] == 20
    assert registered['retained_method_pair_ref'] == retained_pair
    assert load(output / 'frozen_methods.json') == bodies
    invocations, returns = [], []

    def replay(body, paid_training, budget, **kwargs):
        invocations.append((deepcopy(body), list(paid_training), kwargs))
        assert budget.spend(3)
        program = Program('feature', name='x')
        if body['method'] == 'Pi1':
            program = Program('mul', children=(Program('const', value=2.), program))
        return {'candidate': Candidate(body['method'], program, 'research_program'),
            'transfer_partition_search': {'completed': True},
            'transfer_partition_followup': {'execution_complete': True}}

    monkeypatch.setattr(runner, 'replay_research_program', replay)
    monkeypatch.setattr(return_partition_pair, 'return_pair', lambda *args, **kwargs:
        (returns.append((args, kwargs)) or {'status': 'recorded_actual_pair'}))
    result = runner.run_paid_pair(output)
    assert len(invocations) == 4 and all(item[1] == rows for item in invocations)
    assert result['status'] == 'paid_methods_completed'
    assert result['arms']['Pi1']['selected_loss'] == 0.
    assert result['compute_cost'] == 16  # two fits per arm, one shared-program score per arm
    assert result['new_observations'] == result['model_calls'] == 0
    assert result['allocation_ref'] == allocation and result['original_ledger']['cumulative_compute'] == 100
    assert len(returns) == 1 and returns[0][1]['retained_pair'] == retained_pair
    assert load(source / 'receipt.json') == old_receipt
    assert runner.run_paid_pair(output)['new_compute_cost'] == 0 and len(invocations) == 4


@pytest.mark.parametrize('search_complete', (True, False))
@pytest.mark.parametrize('wrapped_batch', (True, False))
def test_pi1_pi3_pair_follows_actual_method_contract_and_keeps_unfinished_unknown(tmp_path, monkeypatch,
                                                                                 search_complete, wrapped_batch):
    from scripts import return_partition_pair
    output = tmp_path / 'different_methods'
    rows = [Observation({'x': 1.}, 2.), Observation({'x': 2.}, 4.)]
    for index, row in enumerate(rows):
        batch = {'observations': [asdict(row)]}
        save(output / f'row{index}.json', {'batch': batch} if wrapped_batch else batch)
    data_refs = [runner.ref(output / f'row{i}.json') for i in range(2)]
    candidate_ref = runner.ref(output / 'old.json')
    save(output / 'old.json', encode_entry({'candidate': Candidate('old', Program('const', value=0.), 'old')}))
    save(output / 'pool.json', {'scores': [{'candidate_refs': [candidate_ref], 'mse': 10.}]})
    save(output / 'public_spec.json', asdict(TaskSpec('historical', 'Paid', ('x',), (), ())))
    save(output / 'frozen_methods.json', {'Pi1': {'kind': 'Pi1', 'transfer_partition': {
        'second_stage': {'operation': 'resolve_relation_intervals'}}}, 'Pi3': {'kind': 'Pi3'}})
    save(output / 'registration.json', {'evaluation_arm_names': {'baseline': 'Pi1', 'challenger': 'Pi3'},
        'total_new_compute_upper': 30, 'remaining_compute': 50, 'previous_compute': 0,
        'source_pool_ref': runner.ref(output / 'pool.json'), 'training_refs': data_refs[:1],
        'feedback_refs': data_refs[1:], 'data_refs': data_refs, 'research_config': {'max_candidates': 4},
        'equal_method_compute_limit': 10, 'per_arm_comparison_reserve': 8,
        'feature_mappings': [{'x': 'x'}], 'active_wall_seconds': 60, 'quality_threshold': 1e-7,
        'source_cost_refs': [], 'original_ledger': {}, 'ledger_scope': 'separate_paid_development',
        'prior_method_review': 'unchanged_topic'})
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    calls, returns = [], []

    def replay(body, training, budget, **kwargs):
        calls.append((body['kind'], kwargs['feedback_rows']))
        assert training == rows[:1] and budget.spend(3)
        program = Program('feature', name='x')
        if body['kind'] == 'Pi1':
            return {'candidate': Candidate('Pi1', program, 'method'), 'training_loss': 1.,
                'transfer_partition_search': {'completed': search_complete},
                'transfer_partition_followup': {'status': 'first_stage_training_unexplained',
                    'execution_complete': False}}
        return {'candidate': Candidate('Pi3', Program('mul', children=(Program('const', value=2.), program)), 'method'),
            'training_loss': 0., 'transfer_partition_search': {'completed': True}}

    monkeypatch.setattr(runner, 'replay_research_program', replay)
    monkeypatch.setattr(return_partition_pair, 'return_pair', lambda *a, **kw:
        (returns.append(kw) or {'status': 'actual_pair_returned'}))
    result = runner.run_paid_pair(output)
    assert calls == [('Pi1', rows[1:]), ('Pi3', ())]
    assert result['arms']['Pi1']['executions'][0]['followup_prerequisite_unmet']
    assert result['arms']['Pi3']['execution_complete'] and result['arms']['Pi3']['selected_loss'] == 0.
    assert result['status'] == ('paid_methods_completed' if search_complete else 'incomplete_method_not_negative')
    assert bool(returns) == search_complete


@pytest.mark.parametrize('solved', (False, True))
def test_prepared_common_registers_same_canonical_pair_without_source_fit_reuse(tmp_path, monkeypatch, solved):
    from copy import deepcopy
    from test_transfer_partitions import second_stage_fixture
    source, canonical, output = [tmp_path / name for name in ('common', 'canonical', 'paired')]
    entry, training, feedback, _ = second_stage_fixture()
    pi1 = entry['research_program']
    pi3 = deepcopy(pi1)
    terms = runner.research_program_fit_contract(pi1)['fit_terms']
    pi3['transfer_partition'] = {'kind': 'Pi3', 'max_attempts': 2, 'max_candidates': 4,
        'alternatives': [{'fit_terms': [term.to_dict() for term in terms]}]}
    methods = {'Pi1': pi1, 'Pi3': pi3}
    names = {'baseline': 'Pi1', 'challenger': 'Pi3'}
    artifacts = {arm: runner.ref(canonical / (arm + '_method.json')) for arm in methods}
    retained_pair = runner.ref(canonical / 'method_return/pair.json')
    save(canonical / 'frozen_methods.json', methods)
    save(canonical / 'registration.json', {'evaluation_arm_names': names,
        'native_method_artifact_refs': artifacts, 'research_config': {}, 'active_wall_seconds': 60,
        'paid_fit_inputs': {'Pi3': {'fits': ['old-source-fit-not-current-data']}},
        'shared_paid_scores': ['old-source-score-not-current-data']})
    save(source / 'frozen_methods.json', methods)
    public = TaskSpec('new_source', 'New common view', tuple(training[0].features), (), ())
    save(source / 'public_spec.json', asdict(public))
    training_ref, feedback_ref = [runner.ref(source / (name + '.json')) for name in ('training', 'feedback')]
    save(source / 'training.json', {'batch': {'observations': [asdict(row) for row in training]}})
    save(source / 'feedback.json', {'batch': {'observations': [asdict(row) for row in feedback]}})
    save(source / 'prepared_state.json', {'training_refs': [training_ref], 'validation_refs': [feedback_ref]})
    source_cfg = {'evaluation_role': 'paid_development_preparation', 'quality_threshold': 1e-7,
        'canonical_source': str(canonical), 'knowledge_source': 'full-protected-pool',
        'source_model_ref': runner.ref(source / 'model.json'), 'source_task_ref': {'source_id': 'new_source'},
        'prior_method_review': str(canonical / 'method_return'), 'retained_method_pair_ref': retained_pair,
        'feature_mappings': [{}], 'source_identity': 'registered_new_physical_source',
        'evaluation_arm_names': {'baseline': 'ignored_request_name', 'challenger': 'ignored_request_name2'}}
    save(source / 'registration.json', source_cfg)
    receipt = {'full_comparison': True, 'selected_mse': 0. if solved else .5,
        'cumulative_compute': 120, 'remaining_compute': 100000, 'rows': len(training) + len(feedback),
        'data_refs': [training_ref, feedback_ref], 'scores': []}
    save(source / 'receipt.json', receipt)
    monkeypatch.setattr(runner, 'freeze_entry', lambda *args: None)
    result = runner.register_paid_pair(source, output)
    if solved:
        assert result['status'] == 'paid_scene_already_explained_no_method_label'
        assert not output.exists()
        return
    assert result['evaluation_arm_names'] == names
    assert result['native_method_artifact_refs'] == artifacts
    assert result['retained_method_pair_ref'] == retained_pair
    assert result['prior_method_review'] == source_cfg['prior_method_review']
    assert result['canonical_source'] == str(canonical)
    assert result['knowledge_source'] == 'full-protected-pool'
    assert result['source_model_ref'] == source_cfg['source_model_ref']
    assert result['data_refs'] == [training_ref, feedback_ref]
    assert result['training_refs'] == [training_ref] and result['feedback_refs'] == [feedback_ref]
    assert result['method_quotes']['Pi1'][0]['fit_rows'] == len(training)
    assert result['method_quotes']['Pi1'][0]['followup_compute_upper'] == 15000
    assert result['method_quotes']['Pi3'][0]['followup_compute_upper'] == 0
    assert result['confirmation_compute'] == 0
    assert 'paid_fit_inputs' not in result and 'shared_paid_scores' not in result
    assert load(output / 'frozen_methods.json') == methods
    assert load(source / 'receipt.json') == receipt
