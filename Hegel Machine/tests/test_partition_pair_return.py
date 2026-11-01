from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from scripts.return_partition_pair import bind_methods, paid_pre_call_profile, refresh_pair_profiles, return_pair
from scripts.run_native_decision_pair import ref
from scripts.run_partial_role_parent import load, save
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, TaskSpec
from hegel_machine.cognition.object_continuation import encode_entry


def test_new_source_reuses_the_actual_canonical_method_body(tmp_path):
    first, second = tmp_path / 'first', tmp_path / 'second'
    bodies = {'P0': {'program': 'source_fixed'}, 'Pi1': {'program': 'two_stage'}}
    save(first / 'frozen_methods.json', bodies)
    save(second / 'frozen_methods.json', deepcopy(bodies))
    original = bind_methods(first, tmp_path / 'return1')
    pair = {}
    for side, method in original.items():
        save(tmp_path / (side + '.json'), method)
        pair[side + '_ref'] = ref(tmp_path / (side + '.json'))
    save(tmp_path / 'pair.json', pair)
    reused = bind_methods(second, tmp_path / 'return2', ref(tmp_path / 'pair.json'))
    assert reused == original
    artifact = load(original['challenger']['method_artifact_ref']['file'])
    assert artifact['execution']['entry'] == 'replay_research_program'
    assert artifact['execution']['research_program_ref']['file'] == str(first / 'frozen_methods.json')
    bodies['Pi1']['program'] = 'changed'
    save(second / 'frozen_methods.json', bodies)
    with pytest.raises(ValueError, match='differs'):
        bind_methods(second, tmp_path / 'bad', ref(tmp_path / 'pair.json'))


def test_saved_return_does_not_reopen_science_or_create_another_source(tmp_path):
    save(tmp_path / 'receipt.json', {'status': 'complete', 'compute_cost': 14, 'paired_invocations': 1})
    result = return_pair(tmp_path / 'unneeded_source', tmp_path, tmp_path / 'unneeded_topic')
    assert result['new_compute_cost'] == 0 and result['reused']
    assert result['paired_invocations'] == 1


def test_paid_profile_keeps_pre_use_roles_and_commanded_control_coverage(tmp_path, monkeypatch):
    measured = tuple(f'm{index}' for index in range(16))
    controls = tuple(f'c{index}' for index in range(13))
    public = TaskSpec('anonymous', 'Public measurements', measured, (), (),
        feature_bounds={name: (-1., 1.) for name in controls}, max_probe_length=24)
    save(tmp_path / 'public_spec.json', asdict(public))
    save(tmp_path / 'prepared_state.json', {'public_spec_ref': ref(tmp_path / 'public_spec.json')})
    groups = [{'object_id': f'object{index}', **{role: measured[4 * index + offset]
        for offset, role in enumerate(('q', 'x', 'u', 'f'))}} for index in range(4)]
    model_ref = ref(tmp_path / 'paid_model.json')
    save(tmp_path / 'paid_model.json', {'groups': groups})
    # No measurement row contains a public control name. The retained commands
    # are the evidence of control coverage, not the predictor's input vectors.
    rows = [asdict(Observation({name: float(step) for name in measured}, float(step)))
        for step in range(2)]
    data_ref = ref(tmp_path / 'training.json')
    save(tmp_path / 'training.json', {'batch': {'observations': rows}, 'probe': {
        'sequence': [{name: value for name in controls} for value in (-.9, .9)]}})
    candidate_ref = ref(tmp_path / 'old_candidate.json')
    save(tmp_path / 'old_candidate.json', encode_entry({'candidate': Candidate(
        'old', Program('const', value=0.), 'retained')}))
    compared = {'data_refs': [data_ref], 'scores': [{'candidate_ref': candidate_ref, 'loss': .25}]}
    cfg = {'source_model_ref': model_ref, 'research_config': {},
        'quality_threshold': 1e-7, 'equal_method_compute_limit': 1000}
    # Neither a task generator nor a completed-method file is needed. Deliberate
    # later-stage values must not replace the retained pre-call loss or roles.
    save(tmp_path / 'Pi1/outcome.json', {'selected_loss': 0., 'object_groups': 99})
    profile = paid_pre_call_profile(tmp_path, cfg, compared)
    assert profile['data']['object_groups'] == 4
    assert profile['data']['object_fields'] == ['f', 'q', 'u', 'x']
    assert len(profile['data']['coverage']) == 13
    assert all(item['observed'] and item['varying'] and item['bins'] == [0, 3]
        for item in profile['data']['coverage'])
    assert profile['residual']['loss'] == .25
    assert profile['recovery_basis']['source_model_ref'] == model_ref
    assert profile['recovery_basis']['control_probe_refs'] == [data_ref]
    assert profile['recovery_basis']['uses_post_method_predictions'] is False

    # The first real successor has a registered task but no saved public spec.
    # Recovering its interface must never buy another observation.
    save(tmp_path / 'prepared_state.json', {'comparison_ref': ref(tmp_path / 'common_comparison.json')})
    (tmp_path / 'public_spec.json').unlink()
    cfg['task'] = {'seed': 106030101}

    class RegisteredPublicInterface:
        def __init__(self, **task):
            assert task == cfg['task']

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def spec_for(self, channels):
            assert channels == ('fine',)
            return public

        def acquire(self, *args, **kwargs):
            raise AssertionError('Paid profile recovery must not acquire observations')

    monkeypatch.setattr('scripts.return_partition_pair.AnonymousObjectLab', RegisteredPublicInterface)
    recovered = paid_pre_call_profile(tmp_path, cfg, compared)
    assert {key: value for key, value in recovered.items() if key != 'recovery_basis'} == {
        key: value for key, value in profile.items() if key != 'recovery_basis'}
    assert recovered['recovery_basis']['public_interface'] == {
        'kind': 'public_interface_reconstructed_from_registration',
        'registration_ref': ref(tmp_path / 'registration.json'), 'new_observations': 0}


def test_profile_refresh_preserves_two_origins_and_uses_latest_ledger(tmp_path, monkeypatch):
    from test_native_paired_method_feedback import common, deliver, route
    import scripts.return_partition_pair as entry

    prior, output = tmp_path / 'prior', tmp_path / 'refreshed'
    agent = common(prior)
    pair_refs = [deliver(agent, tmp_path, name, cheaper=cheaper)[1]
        for name, cheaper in (('positive', True), ('negative', False))]
    route(agent)
    save(prior / 'receipt.json', {'cumulative_compute': 500, 'remaining_compute': 1000})
    originals = {}
    for location in pair_refs:
        pair = load(location['file'])
        source = Path(pair['origin_ref']['file']).parent
        save(source / 'registration.json', {})
        save(source / 'common_comparison.json', {})
        for path in (Path(location['file']), Path(pair['baseline_ref']['file']),
                     Path(pair['challenger_ref']['file'])):
            originals[path] = path.read_bytes()

    def restored(source, cfg, compared):
        old = load(source / 'profile.json')
        return {**old, 'data': {'object_groups': 4, 'object_fields': ['f', 'q', 'u', 'x']}}

    monkeypatch.setattr(entry, 'paid_pre_call_profile', restored)
    result = refresh_pair_profiles(prior, output, pair_refs)
    groups = result['return']['research_result']['method_action_research']['paired_condition_research']
    assert len(groups) == 1 and groups[0]['paired_invocations'] == 2
    assert [item['preference'] for item in groups[0]['preferences']] == ['challenger', 'baseline']
    assert result['compute_cost'] > 0
    assert result['cumulative_compute'] == 500 + result['compute_cost']
    assert result['remaining_compute'] == 1000 - result['compute_cost']
    assert result['new_invocations'] == result['new_fits'] == result['new_predictions'] == 0
    for location, updated in zip(pair_refs, result['updated_pair_refs']):
        old, new = load(location['file']), load(updated['file'])
        assert old['origin_ref'] == new['origin_ref']
        before, after = load(old['challenger_ref']['file']), load(new['challenger_ref']['file'])
        assert before['loss'] == after['loss'] and before['costs'] == after['costs']
        assert before['method_artifact_ref'] == after['method_artifact_ref']
        assert load(after['profile_ref']['file'])['data']['object_groups'] == 4
    assert all(path.read_bytes() == raw for path, raw in originals.items())
    repeated = refresh_pair_profiles(prior, output, pair_refs)
    assert repeated['reused'] and repeated['new_compute_cost'] == 0


@pytest.mark.parametrize('separate_allocation', [False, True])
@pytest.mark.parametrize('arm_names', [None, {'baseline': 'Pi1', 'challenger': 'Pi3'}])
def test_completed_paid_pair_returns_to_same_topic_without_inventing_future_readout(tmp_path, monkeypatch,
                                                                                  separate_allocation, arm_names):
    from test_native_paired_method_feedback import common, deliver, route
    import scripts.return_partition_pair as entry

    prior, source, output = [tmp_path / name for name in ('prior', 'paid103', 'return')]
    agent = common(prior)
    deliver(agent, tmp_path, 'previous_paid_source', cheaper=True)
    route(agent)
    names = arm_names or {'baseline': 'P0', 'challenger': 'Pi1'}
    save(source / 'frozen_methods.json', {names['baseline']: {'body': 'original'},
        names['challenger']: {'body': 'alternative'}})
    cfg = {'evaluation_role': 'paid_working_comparison', 'method_return_compute_limit': 1000,
        'source_task_ref': {'source_id': 'historical103'}, 'training_rows': 125, 'working_rows': 129,
        'quality_threshold': 1e-7, 'equal_method_compute_limit': 100,
        'per_arm_comparison_reserve': 8, 'active_wall_seconds': 60}
    if arm_names:
        cfg['evaluation_arm_names'] = arm_names
        cfg['native_method_artifact_refs'] = {}
        for arm in names.values():
            path = source / (arm + '_method.json')
            save(path, {'id': 'native:parent_partition:paid103:' + arm, 'version': 1,
                'kind': 'native_research_segment', 'execution': {
                    'entry': 'replay_research_program',
                    'research_program_ref': ref(source / 'frozen_methods.json', arm)}})
            cfg['native_method_artifact_refs'][arm] = ref(path)
    save(source / 'registration.json', cfg)
    save(source / 'common_comparison.json', {'data_refs': [ref(source / 'paid.json')]})
    outcomes = {name: {'execution_complete': True, 'full_comparison': True,
        'selected_loss': measured, 'compute_cost': cost, 'selected_ref': ref(source / name / 'selected.json')}
        for name, measured, cost in [(names['baseline'], .1, 5), (names['challenger'], 0., 10)]}
    save(source / 'receipt.json', {'status': 'paid_methods_completed', 'arms': outcomes,
        'remaining_compute': 1000, 'cumulative_compute': 15,
        'execution_timing': {'phase_seconds': {names['baseline']: 1., names['challenger']: 2.}}})
    monkeypatch.setattr(entry, 'paid_pre_call_profile', lambda *args: {'data': {}, 'residual': {'loss': 1.}})
    original = (source / 'receipt.json').read_bytes()
    save(tmp_path / 'allocation.json', {'allocation_id': 'method_returns',
        'maximum_compute': 5000, 'new_observation_limit': 0})
    cohort = [{'source_id': 'held'}, {'source_id': 'changed'}]
    result = return_pair(source, output, prior,
        **({'allocation_ref': ref(tmp_path / 'allocation.json'),
            'validation_cohort_refs': cohort} if separate_allocation else {}))
    pair = load(output / 'pair.json')
    assert pair['physical_source_ref'] == cfg['source_task_ref']
    assert pair['evaluation_arm_names'] == names
    for side, arm in names.items():
        actual = load(pair[side + '_ref']['file'])
        assert actual['actual_outcome_ref'] == ref(source / arm / 'outcome.json')
        if arm_names:
            assert actual['method_artifact_ref'] == cfg['native_method_artifact_refs'][arm]
    if arm_names:
        reviewed = result['return']['research_result']['method_action_research']
        learned = next(item for item in reviewed['paired_condition_research']
            if item['method_ref']['id'] == 'native:parent_partition:paid103:Pi3')
        assert learned['baseline_method_ref']['id'] == 'native:parent_partition:paid103:Pi1'
        assert learned['paired_invocations'] == 1
        assert learned['general_guard_supported'] is False
        assert learned['condition'] is None
        # A second independently addressed development source keeps the same
        # actual artifact and joins the existing group, rather than inventing
        # another Pi1/Pi3 identity for identical frozen bodies.
        second, second_return = tmp_path / 'second_paid', tmp_path / 'second_return'
        save(second / 'frozen_methods.json', load(source / 'frozen_methods.json'))
        save(second / 'registration.json', {**cfg, 'source_task_ref': {'source_id': 'different_source'}})
        save(second / 'common_comparison.json', {'data_refs': [ref(second / 'paid.json')]})
        second_receipt = deepcopy(load(source / 'receipt.json'))
        second_receipt.update(remaining_compute=3000)
        save(second / 'receipt.json', second_receipt)
        returned_again = return_pair(second, second_return, output, retained_pair=ref(output / 'pair.json'))
        reviewed_again = returned_again['return']['research_result']['method_action_research']
        same_group = next(item for item in reviewed_again['paired_condition_research']
            if item['method_ref']['id'] == learned['method_ref']['id'])
        assert same_group['paired_invocations'] == 2
        assert same_group['baseline_method_ref'] == learned['baseline_method_ref']
        assert reviewed_again['paired_method_exclusions'] == []
        actual = load(second_return / 'challenger_outcome.json')
        assert actual['method_artifact_ref'] == cfg['native_method_artifact_refs']['Pi3']
    assert pair['data_role'] == 'development' and pair['evaluation_role'] == 'paid_working_comparison'
    assert 'paired_future_ref' not in pair and 'frozen_predictors_ref' not in pair
    assert result['new_fits'] == result['new_predictions'] == result['new_observations'] == 0
    assert result['condition_learning_role'] == 'exposed_paid_working_comparison_not_new_validation'
    assert result['compute_cost'] > 0
    if separate_allocation:
        assert pair['condition_validation_cohort_refs'] == cohort
        assert result['cumulative_compute'] == result['compute_cost']
        assert result['remaining_compute'] == 5000 - result['compute_cost']
        assert result['allocation_id'] == 'method_returns'
    assert (source / 'receipt.json').read_bytes() == original


def test_incomplete_changed_method_pair_stays_unknown_without_feedback_labels(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'returned'
    save(source / 'registration.json', {'evaluation_role': 'paid_working_comparison',
        'evaluation_arm_names': {'baseline': 'Pi1', 'challenger': 'Pi3'}})
    save(source / 'receipt.json', {'status': 'incomplete_method_not_negative', 'arms': {
        'Pi1': {'execution_complete': True, 'full_comparison': True},
        'Pi3': {'execution_complete': False, 'full_comparison': True}}})
    result = return_pair(source, output, tmp_path / 'unneeded_topic')
    assert result['status'] == 'incomplete_method_not_negative'
    assert result['independent_method_gain'] is None and not result['method_pair_created']
    assert result['new_compute_cost'] == 0
    assert not output.exists()
