"""Diagnose frozen proposals without learning from the held continuation."""
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def audit_module():
    specification = importlib.util.spec_from_file_location('cross_skeleton_consumption',
        ROOT / 'scripts/audit_cross_skeleton_consumption.py')
    module = importlib.util.module_from_spec(specification)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / 'scripts'))
        specification.loader.exec_module(module)
    if not module.POINTER.is_file():
        pytest.skip('Historical source pointer is not available')
    if not Path(json.loads(module.POINTER.read_text())['archive']).is_file():
        pytest.skip('Historical shared archive is not mounted')
    return module


@pytest.fixture(scope='module')
def consumption(audit_module, tmp_path_factory):
    return audit_module.audit_consumption(output=tmp_path_factory.mktemp('consumption') / 'receipt.json')


def test_all_saved_candidates_reconstruct_and_were_actually_scored_and_admitted(consumption):
    reconstruction = consumption['reconstruction']
    assert reconstruction['program_count'] == 728
    assert reconstruction['maximum_native_prefix_mse_difference'] == 0
    assert reconstruction['maximum_program_size_difference'] == 0
    assert reconstruction['all_new_candidates_scored']
    admission = consumption['admission']
    assert len(admission['new_admitted_program_ids']) == len(admission['displaced_old_program_ids']) == 9
    assert len(admission['old_still_admitted_program_ids']) == 7
    assert admission['all_raw_old_programs_remain_available']
    assert not consumption['diagnosis']['candidate_scoring_or_admission_disconnected']


def test_prefix_score_tradeoff_is_not_mistaken_for_native_parent_adoption(consumption):
    rows = {row['program_id']: row for row in consumption['highlighted_candidates']}
    objective = consumption['parent_objective']
    short = rows[objective['ranking_winner_in_both_arms']]
    accurate_old = rows[objective['old_best_by_prefix_mse']]
    assert short['prefix_mse'] > accurate_old['prefix_mse']
    assert short['score'] < accurate_old['score']
    assert short['score'] == pytest.approx(short['prefix_mse'] + .001 * short['size'])
    adoption = consumption['parent_adoption']
    assert adoption['actual_frozen_incumbent_program_id'] == 'program_0017'
    assert adoption['ranking_winner_program_id'] == 'program_0004'
    assert 0 < adoption['ranking_winner_score_gain_over_incumbent'] < adoption['minimum_improvement']
    assert adoption['ranking_winner_evidence_loss_change_over_incumbent'] > 0
    assert adoption['fresh_comparison_rows_after_freeze'] == 0
    assert not adoption['scientific_adoption_established']
    assert adoption['gate_status'] == 'deferred_no_fresh_comparison'


def test_accurate_old_program_and_new_prefix_overfit_are_both_retained(consumption):
    objective = consumption['parent_objective']
    rows = {row['program_id']: row for row in consumption['highlighted_candidates']}
    old = rows[objective['old_best_by_prefix_mse']]
    new = rows[objective['new_best_by_prefix_mse']]
    assert old['prefix_mse'] < 1e-12 and new['prefix_mse'] < 1e-12
    assert old['posthoc_continuation_squared_error'] < 1e-12
    assert new['posthoc_continuation_squared_error'] > .1
    assert consumption['candidate_distributions']['new']['posthoc_continuation_predictions_different_from_ranking_winner'] > 0
    assert not consumption['held_diagnostic']['used_to_change_candidates_ranking_or_budget']


def test_original_readout_misses_startup_and_tied_sets_are_distinct(consumption):
    contract = consumption['experiment_contract']
    assert contract['readout_step'] == 1
    assert not contract['startup_hole_executes_at_original_panel_readouts']
    assert contract['new_mature_values_are_existing_old_component_values']
    assert contract['remove_tied_ids_recovered_from_saved_scores'] == [
        'representative_0', 'representative_1', 'representative_2']
    assert contract['keep_tied_ids_recovered_from_saved_scores'] == [
        'initial_00', 'initial_01', 'initial_02']
    assert contract['both_selected_action_ids'] == [None, None]
    assert consumption['bounded_enumeration']['allocation_exhausted']
    assert consumption['bounded_enumeration']['complete_cross_product_not_claimed']


def test_changed_held_label_cannot_select_a_new_program_or_change_score_diagnosis(audit_module, consumption, monkeypatch, tmp_path):
    load = audit_module.load_source

    def changed_future(pointer):
        payload, source = load(pointer)
        payload['episodes'][6]['memory_events'][0]['observation']['history'][0]['__target__'] = 1e6
        return payload, source

    monkeypatch.setattr(audit_module, 'load_source', changed_future)
    changed = audit_module.audit_consumption(output=tmp_path / 'changed_label.json')
    assert changed['parent_objective'] == consumption['parent_objective']
    assert changed['admission'] == consumption['admission']
    assert changed['bounded_enumeration'] == consumption['bounded_enumeration']
    original_rows = consumption['highlighted_candidates']
    changed_rows = changed['highlighted_candidates']
    assert [row['program_id'] for row in changed_rows] == [row['program_id'] for row in original_rows]
    for before, after in zip(original_rows, changed_rows):
        assert after['score'] == before['score']
        assert after['posthoc_continuation_prediction'] == before['posthoc_continuation_prediction']
        assert after['posthoc_continuation_squared_error'] != before['posthoc_continuation_squared_error']


def test_common_route_alignment_preserves_admissions_budget_and_real_ties(audit_module, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    specification = importlib.util.spec_from_file_location('selection_contract_replay',
        ROOT / 'scripts/replay_cross_skeleton_selection_contract.py')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    result = module.replay_selection_contract(output=tmp_path / 'contract.json')
    original = json.loads(audit_module.RECEIPT.read_text())
    assert result['protocol']['auxiliary_raw_readout_diagnostic_compute'] == 32
    for name, counts in [('original_step1_declared_resolution', (51, 3)),
                         ('aligned_step0_native', (49, 20))]:
        arms = result['contracts'][name]['arms']
        for arm, pairs in zip(('remove', 'keep'), counts):
            row = arms[arm]
            assert row['admitted_program_ids'] == original['arms'][arm]['admitted_program_ids']
            assert row['candidate_count'] == (344 if arm == 'remove' else 728)
            assert row['effective_pairs'] == pairs and row['status'] == 'tied'
            assert row['selected_experiment_id'] is None and not row['pending_query_prepared']
            assert not row['support_updated'] and row['observation_cost'] == 0
            assert row['compute']['matched_protocol_total'] == original['arms'][arm]['compute']['total']
            assert row['compute']['matched_protocol_total'] <= result['protocol']['same_total_compute_budget']
            assert row['compute']['actual_common_route'] == 96
            assert row['public_row_validation']['available'] and not row['excluded']
            if name == 'aligned_step0_native':
                assert row['raw_missing_history_count'] == 4
                assert all(group['known'] for group in row['groups'])
                assert row['tied_experiments'] == ['initial_00', 'initial_01', 'initial_02']
    assert result['conclusion']['aligned_native_contract_still_tied']
    assert not result['conclusion']['scientific_or_parent_adoption_gain_established']
