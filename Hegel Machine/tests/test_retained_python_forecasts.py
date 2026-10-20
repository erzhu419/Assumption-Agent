"""The retained Python forecast table keeps real histories, offsets and failures."""

import importlib.util
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import inquiry_revision


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def replay_module():
    path = ROOT / 'scripts/replay_retained_python_forecasts.py'
    specification = importlib.util.spec_from_file_location('retained_python_forecasts', path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    if not (module.SOURCE / 'research_state.json').exists():
        pytest.skip('The retained target_000 source is not mounted')
    return module


def test_real_complete_48_rows_reproduce_native_score_and_original_choice_without_model_execution(
        replay_module, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('A saved-forecast replay attempted to execute or refit a Python predictor')
    monkeypatch.setattr(inquiry_revision, 'evaluate_python_model', forbidden)
    monkeypatch.setattr(inquiry_revision, 'retain_evaluated_python_model', forbidden)
    result = replay_module.replay(output=tmp_path / 'receipt.json')
    assert result['status'] == 'completed' and result['row_pairing']['actual_successor_rows'] == 48
    assert result['row_pairing']['feature_history_mismatches'] == 0
    rows = result['row_pairing']['rows']
    assert rows[0] == [24, 0, 72, 0] and rows[23] == [24, 23, 95, 23]
    assert rows[24] == [25, 0, 96, 24] and rows[-1] == [25, 23, 119, 47]
    assert [row['mse'] for row in result['forecasts']] == [2.3685389106332595, 93.53979797386548]
    assert [[row['mse'] for row in model['per_record']] for model in result['forecasts']] == [
        [.17727431058287194, 4.559803510683647], [180.71352334276693, 6.366072604963958]]
    assert all(row['saved_mse_reproduced'] and row['mse'] == row['complete_revision_mse']
               and row['scoped_status'] == 'refuted' for row in result['forecasts'])
    assert result['failure_preservation']['all_frozen_successor_instances_refuted']
    assert len(result['failure_preservation']['all_offered_models']) == 3
    assert len(result['failure_preservation']['eligible_models']) == 2
    assert result['same_selected_experiment'] and result['selection']['selected_experiment_id'] == 'alternating_and_hold'
    assert [row['predicted_separation'] for row in result['selection']['scores']] == [
        21.458360718750047, 1.0635812875905102]
    assert result['panel_outcomes'][1] == {'id': 'switches_and_reversals', 'status': 'UNKNOWN',
        'record_indices': [], 'reason': 'panel_not_executed_no_labels'}
    assert result['native_completion']['readout_declared'] is False
    assert result['protocol']['PR2_route_executed'] is False
    assert result['protocol']['new_fit_runs'] == result['protocol']['new_predictor_executions'] == 0
    assert result['original_account']['costs']['observation_cost'] == 216
    assert result['original_account']['paid_trace_rows'] == 120
    assert result['original_account']['comparison_compute'] == 696
    assert result['replay_compute']['total'] == 384
    assert result['replay_compute']['within_original_compute_cap_including_original_account']


@pytest.mark.parametrize('damaged', ['history', 'incomplete_trajectory'])
def test_misaligned_public_history_or_missing_successor_row_cannot_be_scored_as_complete(
        replay_module, tmp_path, damaged):
    state = json.loads((replay_module.SOURCE / 'research_state.json').read_text())
    observations = json.loads((replay_module.SOURCE / 'laboratory_observations.json').read_text())
    if damaged == 'history':
        observations[25]['rows'][0]['history'] = [{'u': 99.}]
    else:
        observations[25]['rows'].pop()
    (tmp_path / 'research_state.json').write_text(json.dumps(state))
    (tmp_path / 'laboratory_observations.json').write_text(json.dumps(observations))
    with pytest.raises(ValueError, match='Frozen input differs|incomplete'):
        replay_module.replay(source=tmp_path, output=tmp_path / 'bad_receipt.json')
    assert not (tmp_path / 'bad_receipt.json').exists()


def test_successor_labels_change_only_the_check_not_frozen_candidate_or_panel_choice(replay_module, tmp_path):
    state = json.loads((replay_module.SOURCE / 'research_state.json').read_text())
    original = replay_module.frozen_selection(state['experiment_comparisons'][0], state['limits']['compute_budget'])
    observations = json.loads((replay_module.SOURCE / 'laboratory_observations.json').read_text())
    for index in (24, 25):
        for row in observations[index]['rows']:
            row['target'] += 10.
    (tmp_path / 'research_state.json').write_text(json.dumps(state))
    (tmp_path / 'laboratory_observations.json').write_text(json.dumps(observations))
    result = replay_module.replay(source=tmp_path, output=tmp_path / 'changed_labels.json')
    assert result['selection'] == original
    assert result['protocol']['candidate_and_panel_selection_before_successor_labels']
    assert all(not row['saved_mse_reproduced'] for row in result['forecasts'])
    assert result['native_completion']['record_indices'] == [24, 25]
