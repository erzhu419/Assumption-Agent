"""The historical mapping intervention changes construction checks, not truth."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def replay_module():
    spec = importlib.util.spec_from_file_location('relation_mapping_ablation',
        ROOT / 'scripts/replay_relation_mapping_ablation.py')
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / 'scripts'))
        spec.loader.exec_module(module)
    if not module.POINTER.exists() or not Path(json.loads(module.POINTER.read_text())['archive']).is_file():
        pytest.skip('The retained historical shared archive is not mounted')
    return module


@pytest.fixture(scope='module')
def replay(replay_module, tmp_path_factory):
    return replay_module.replay_relation_mapping_ablation(
        output=tmp_path_factory.mktemp('relation_mapping') / 'receipt.json')


def test_same_typed_holes_are_distinguished_by_the_actual_learned_history_constraints(replay):
    protocol = replay['protocol']
    assert protocol['training_rows'] == 8 and protocol['old_baseline_program_count'] == 344
    assert protocol['finite_domain_size'] == 385
    assert protocol['source_cutoffs'] == [
        {'episode_index': 1, 'action_cutoff': 18}, {'episode_index': 2, 'action_cutoff': 34}]
    assert protocol['original_generation_allocation'] == 20000
    assert protocol['original_generation_status'] == 'UNKNOWN' and protocol['original_generation_budget_exhausted']
    first, second = protocol['hole_contracts']
    assert {key: value for key, value in first.items() if key != 'name'} == {
        key: value for key, value in second.items() if key != 'name'}
    assert protocol['construction_only']
    original, permuted = replay['arms']['original'], replay['arms']['permuted']
    assert original['binding_recheck']['status_counts'] == {'completed': 385}
    assert permuted['binding_recheck']['status_counts'] == {'not_applicable': 385}
    assert permuted['binding_recheck']['binding_checks']['outcomes'] == [
        {'status': 'not_applicable', 'violated_constraint_indices': [0, 2, 3]}]
    example = permuted['binding_recheck']['excluded_examples'][0]['excluded'][0]
    assert example['reason'] == 'learned_property_constraint_violated'
    assert example['constraint_checks'][2]['left_value'] == 0
    assert example['constraint_checks'][2]['right_value'] == 1
    assert original['candidate_count'] == 728 and permuted['candidate_count'] == 344
    assert len(original['numerically_refuted_bindings']) == 1


def test_old_programs_numeric_failure_and_native_partial_adoption_are_preserved(replay):
    assert replay['counterexample_replay']['status'] == 'refuted'
    assert replay['counterexample_replay']['checks'][0]['status'] == 'failed'
    assert replay['protocol']['same_total_compute_budget'] == 150000
    assert replay['protocol']['minimum_improvement'] == .002
    assert replay['protocol']['complexity_weight'] == .001
    for arm in replay['arms'].values():
        assert arm['all_old_programs_preserved'] and arm['source_reconstruction_preserved']
        assert arm['retained_numeric_failure_present'] and arm['retained_failed_checks_present']
        assert arm['old_candidate_count'] == 344 and len(arm['admitted_program_ids']) == 16
        assert arm['selection']['status'] == 'tied'
        assert arm['selection']['tied_experiments'] == ['initial_00', 'initial_01', 'initial_02']
        assert not arm['selection']['pending_query_prepared']
        comparison = arm['partial_comparison']
        assert comparison['native_gate_adopt']
        assert comparison['before_program_id'] == 'program_0017'
        assert comparison['after_program_id'] == 'program_0016'
        assert not comparison['actual_source_deployment_changed']
        assert arm['later_excerpt']['after_mse'] == pytest.approx(.0018749999637355207)
        assert not arm['later_excerpt']['used_for_admission_or_adoption']
        assert arm['new_paid_observations'] == arm['new_provider_calls'] == 0
        compute = arm['compute']
        assert compute['total'] == sum(value for key, value in compute.items() if key not in {'total', 'remaining'})
        assert compute['total'] + compute['remaining'] == 150000
    assert replay['comparison']['same_partial_parent'] and replay['comparison']['same_later_predictions']
    assert not replay['comparison']['scientific_gain_established']
    assert replay['comparison']['complete_historical_comparison'] == 'UNKNOWN_missing14of16rows'


def test_a_later_target_cannot_change_the_permutation_binding_checks_admission_or_parent(
        replay_module, replay, monkeypatch, tmp_path):
    recover = replay_module.recover_t07_evidence

    def changed_later_target(payload, source):
        evidence = recover(payload, source)
        evidence['later_rows'][-1] = replace(evidence['later_rows'][-1],
                                            target=evidence['later_rows'][-1].target + 10.)
        return evidence

    monkeypatch.setattr(replay_module, 'recover_t07_evidence', changed_later_target)
    changed = replay_module.replay_relation_mapping_ablation(output=tmp_path / 'changed_later.json')
    assert changed['forecast_table'] == replay['forecast_table']
    for name, original in replay['arms'].items():
        current = changed['arms'][name]
        for key in ('mapping', 'binding_recheck', 'admitted_program_ids', 'partial_comparison', 'selection'):
            assert current[key] == original[key]
        assert current['later_excerpt']['after_mse'] != original['later_excerpt']['after_mse']


def test_interrupted_binding_check_is_unknown_and_preserves_real_spent_compute(replay_module):
    data = replay_module.reconstruct_inputs()
    frozen = data['receipt']['frozen_relation']
    domain = replay_module._domain(data['receipt'])
    components = {f'old:{i:04d}': replay_module.declaration(f'old:{i:04d}', entry['candidate'].program)
                  for i, entry in enumerate(data['baseline']['candidates'])}
    names = tuple(replay_module.Definition.from_dict(frozen['definition']).arguments)
    result = replay_module.recheck_bindings(frozen, components, domain, names[1:] + names[:1], compute_budget=1)
    assert result['status'] == 'UNKNOWN' and result['budget_exhausted']
    assert result['actual_compute_cost'] == 1
    assert result['status_counts'] == {'UNKNOWN': 1} and result['proposals'] == []
    assert result['processed_bindings'] == 1
    assert result['binding_checks']['outcomes'] == [{'status': 'UNKNOWN', 'violated_constraint_indices': []}]
