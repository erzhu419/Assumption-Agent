"""A completed joint revision can confirm once without restarting research."""

import json

import pytest

from scripts import revise_learned_structure_successor as entry


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def paid_source(tmp_path):
    original, source = tmp_path / 'original', tmp_path / 'repair/joint_paid'
    write(original / 'registration.json', {'confirmation_rows': 48, 'confirmation_seed': 107029108,
        'source_task_ref': {'id': 'public_task'}, 'observation_limit': 512, 'measurement_limit': 6144})
    write(original / 'receipt.json', {'cumulative_costs': {
        'compute_cost': 84607, 'observation_cost': 195, 'measurement_cost': 2340}})
    write(original / 'challenger/outcome.json', {'selected_ref': {'file': 'retained_parent.json'}})
    write(source / 'registration.json', {'source': str(original),
        'registered_original_total': 270010, 'source_cost_refs': [{'file': 'paid_old_costs.json'}]})
    write(source / 'receipt.json', {'full_comparison': True, 'all_paid_passed': True,
        'fresh_confirmation': None, 'threshold': 1e-7, 'compared_candidates': 28,
        'cumulative_current_successor_compute': 153660})
    write(source.parent / 'post_freeze/common/questions.json', [{
        'id': 'task:method_choice', 'method_use_plan': {'native_return_refs': [1] * 11}}])
    return source


def test_registration_uses_paid_completion_and_original_caps_without_refit(tmp_path, monkeypatch):
    source = paid_source(tmp_path)
    monkeypatch.setattr(entry, 'restore', lambda *a: pytest.fail('registration repeated research restoration'))
    frozen = []
    monkeypatch.setattr(entry, 'freeze_runtime', lambda output, parent: frozen.append((output, parent)))
    result = entry.register_paid_confirmation(source, tmp_path / 'confirm')
    assert result['confirmation_seed'] == 107046125
    assert result['confirmation_rows'] == 48 and result['threshold'] == 1e-7
    assert result['remaining_compute'] == 116350 and result['method_review_compute_limit'] == 13
    assert result['comparison_kind'] == 'additional_research_vs_retained_parent'
    assert result['complete_pool'] == 28 and frozen


@pytest.mark.parametrize('change', [
    {'full_comparison': False}, {'all_paid_passed': False}, {'fresh_confirmation': {'passed': False}},
])
def test_incomplete_or_already_confirmed_unit_does_not_restart(tmp_path, monkeypatch, change):
    source = paid_source(tmp_path)
    receipt = json.loads((source / 'receipt.json').read_text())
    write(source / 'receipt.json', {**receipt, **change})
    monkeypatch.setattr(entry, 'freeze_runtime', lambda *a: pytest.fail('invalid unit froze code'))
    with pytest.raises(ValueError, match='completed paid revision'):
        entry.register_paid_confirmation(source, tmp_path / 'confirm')


def test_confirmation_preserves_existing_observation_limit(tmp_path, monkeypatch):
    source = paid_source(tmp_path)
    original = tmp_path / 'original'
    receipt = json.loads((original / 'receipt.json').read_text())
    receipt['cumulative_costs']['observation_cost'] = 500
    write(original / 'receipt.json', receipt)
    monkeypatch.setattr(entry, 'freeze_runtime', lambda *a: pytest.fail('unfunded readout froze code'))
    with pytest.raises(ValueError, match='observation or measurement'):
        entry.register_paid_confirmation(source, tmp_path / 'confirm')
