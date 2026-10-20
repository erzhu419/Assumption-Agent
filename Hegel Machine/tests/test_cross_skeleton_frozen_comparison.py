"""Real retained comparison, with its missing rows and original outcomes."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def replay_module():
    specification = importlib.util.spec_from_file_location('retained_frozen_comparison',
        ROOT / 'scripts/replay_cross_skeleton_frozen_comparison.py')
    module = importlib.util.module_from_spec(specification)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / 'scripts'))
        specification.loader.exec_module(module)
    if not module.POINTER.is_file() or not Path(json.loads(module.POINTER.read_text())['archive']).is_file():
        pytest.skip('The retained shared historical archive is not mounted')
    return module


@pytest.fixture(scope='module')
def replay(replay_module, tmp_path_factory):
    return replay_module.replay_frozen_comparison(output=tmp_path_factory.mktemp('frozen') / 'receipt.json')


def test_equal_task_information_is_separate_from_equal_numeric_forecasts(replay):
    for arm, covered in [('remove', 49), ('keep', 20)]:
        selection = replay['arms'][arm]['selection']
        assert selection['status'] == 'tied'
        assert selection['tied_experiments'] == ['initial_00', 'initial_01', 'initial_02']
        assert selection['selected_experiment'] is None and not selection['pending_query_prepared']
        diagnosis = selection['tie_diagnostics']
        assert diagnosis['status'] == 'complete'
        assert all(row['task_relevant_covered_pairs'] == covered for row in diagnosis['experiments'])
        assert all(row['same_task_relevant_coverage'] and row['same_pairwise_separation']
                   and not row['same_candidate_predictions'] for row in diagnosis['comparisons'])
    assert not replay['protocol']['forced_tie_break']


def test_only_two_unique_postfreeze_labels_enter_the_comparison(replay):
    evidence = replay['retained_evidence']
    assert [row['id'] for row in evidence['comparison_refs']] == ['t07:action65:step0', 't07:action65:step1']
    assert evidence['coverage']['65']['recovered_unique_rows'] == 2
    assert evidence['coverage']['67']['recovered_unique_rows'] == 0
    assert evidence['historical_costs']['postfreeze_observation_cost'] == 16
    assert evidence['duplicates']['repeated_label_references_merged'] == 2
    assert evidence['comparison_refs'][0]['label_scored_in_prior_continuation_diagnostics']
    assert not replay['protocol']['comparison_is_new_or_blind']
    assert not replay['protocol']['generation_and_fitting_changed']
    assert replay['protocol']['candidate_fit_or_refit_calls'] == 0


def test_partial_native_adoption_changes_predictions_without_claiming_full_ablation(replay):
    for row in replay['arms'].values():
        assert row['comparison']['native_gate_adopt_on_available_rows']
        assert not row['comparison']['complete_original_comparison_replayed']
        assert row['partial_replay_parent']['before_program_id'] == 'program_0017'
        assert row['partial_replay_parent']['after_program_id'] == 'program_0016'
        assert not row['partial_replay_parent']['actual_deployment_or_source_revision']
        assert row['later_excerpt']['predictions_changed']
        assert 0 < row['later_excerpt']['after_mse'] < row['later_excerpt']['before_mse']
        assert not row['later_excerpt']['independent_new_confirmation']
    assert replay['conclusion']['same_partial_replay_parent']
    assert replay['conclusion']['same_later_excerpt_predictions']
    assert replay['conclusion']['complete_same_material_ablation_status'].startswith('unresolved')
    # The original complete run's recorded success is preserved in its own
    # scope; missing rows cannot be replaced with that aggregate summary.
    assert replay['original_run_outcome']['comparison_summary']['adopt']
    assert replay['original_run_outcome']['prediction_check_summary']['revision_loss'] < 1e-12
    assert not replay['original_run_outcome']['final_revision_used_as_new_candidate']


def test_later_target_cannot_change_frozen_admissions_gate_or_predictions(replay_module, replay, monkeypatch, tmp_path):
    recover = replay_module.recover_t07_evidence

    def changed_last_later_target(payload, source):
        evidence = recover(payload, source)
        final = evidence['later_rows'][-1]
        evidence['later_rows'][-1] = replace(final, target=final.target + 10.)
        return evidence

    monkeypatch.setattr(replay_module, 'recover_t07_evidence', changed_last_later_target)
    changed = replay_module.replay_frozen_comparison(output=tmp_path / 'changed_later.json')
    assert changed['forecast_table'] == replay['forecast_table']
    for arm in ('remove', 'keep'):
        before, after = replay['arms'][arm], changed['arms'][arm]
        for key in ('frozen_admitted_program_ids', 'comparison', 'partial_replay_parent', 'selection'):
            assert after[key] == before[key]
        assert after['later_excerpt']['after_mse'] != before['later_excerpt']['after_mse']


def test_all_diagnostic_work_fits_original_budget_without_new_acquisition(replay):
    for arm, expected in [('remove', 6579), ('keep', 29851)]:
        row = replay['arms'][arm]
        assert row['compute']['new_native_ranking'] == 192
        assert row['compute']['total_including_diagnostics'] == expected
        assert expected <= replay['protocol']['same_compute_allocation']
        assert row['new_paid_observations'] == row['new_provider_calls'] == 0
    assert replay['retained_evidence']['scope']['restored_final_evaluation_examples'] == 4
    assert replay['retained_evidence']['scope']['original_final_evaluation_rows'] == 192
