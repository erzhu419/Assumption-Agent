"""An old selected radius can reuse a newer paid partition of the same response."""

from copy import deepcopy
from dataclasses import asdict, replace
from statistics import mean

import pytest

from scripts import review_paid_object_pool as runner
from scripts.run_partial_role_parent import load, save
from hegel_machine.cognition.structural_research_program import research_program_fit_contract
from test_paid_interval_probe_return import paid_probe_scene, forbid_scientific_restart


def current_parent_with_newer_partition(tmp_path, monkeypatch):
    source, old_entries, rows = paid_probe_scene(tmp_path, monkeypatch)
    # All old radii still explain this older evidence. Its completed partition
    # will belong to radius .75, while the later pool selects old radius .775.
    for index, distance in enumerate((.72, .78) * 3, 168):
        row = replace(rows[index], features={**rows[index].features, 'x1': distance})
        rows[index] = replace(row, target=old_entries[0]['candidate'].program.predict(row))
    original = load(source / 'paid_pool/registration.json')
    for location, start in zip(original['appended_observation_refs'], (168, 171), strict=True):
        save(location['file'], {'probe': {'probe_id': f'paid{start}',
            'sequence': [row.features for row in rows[start:start + 3]]},
            'batch': {'observations': [asdict(row) for row in rows[start:start + 3]]}})
    pool = load(source / 'paid_pool/receipt.json')
    for score, entry in zip(pool['scores'], old_entries, strict=True):
        score['mse'] = mean((entry['candidate'].program.predict(row) - row.target) ** 2 for row in rows)
    save(source / 'paid_pool/receipt.json', pool)
    donor = tmp_path / 'completed174'
    runner.register_interval_probe_return(source, donor)
    main_root = runner.ROOT
    monkeypatch.setattr(runner, 'ROOT', donor / 'runtime')
    donor_receipt = runner.run_interval_probe_return(donor)
    monkeypatch.setattr(runner, 'ROOT', main_root)
    assert donor_receipt['execution_complete']
    inherited = load(donor / 'paid_pool/receipt.json')
    old_index = max(range(len(old_entries)), key=lambda index: next(iter(old_entries[index]['parameter_values'].values())))
    parent = old_entries[old_index]
    assert parent['research_program']['evaluation_rows'] == 168
    parent_ref = runner.ref(source / 'old_entries.json', old_index)
    assert parent_ref != donor_receipt['selected_ref']
    assert next(item['mse'] for item in inherited['scores'] if parent_ref in item['candidate_refs']) == 0.
    extra = []
    for index, distance in enumerate((.77, .78, .79)):
        row = replace(rows[168 + index], features={**rows[168 + index].features, 'x1': distance})
        extra.append(replace(row, target=parent['candidate'].program.predict(row)))
    current = tmp_path / 'current177'
    new_data = current / 'observations.json'
    save(new_data, {'probe': {'probe_id': 'latest', 'sequence': [row.features for row in extra]},
        'batch': {'observations': [asdict(row) for row in extra]}})
    all_rows = [*rows, *extra]
    for item in inherited['scores']:
        entry = runner.retained_entry(item['candidate_refs'][0], donor, 0)
        item['mse'] = mean((entry['candidate'].program.predict(row) - row.target) ** 2 for row in all_rows)
        item['rows'] = 177
    inherited.update(rows=177, selected_ref=parent_ref, selected_mse=0.)
    save(current / 'paid_pool/receipt.json', inherited)
    save(current / 'paid_pool/registration.json', {**load(donor / 'paid_pool/registration.json'),
        'appended_observation_refs': [*original['appended_observation_refs'], runner.ref(new_data)],
        'feedback_rows': 81, 'comparison_rows': 177})
    save(current / 'retained_frontier.json', {'full_comparison': True,
        'original_pool_ref': runner.ref(current / 'paid_pool/receipt.json'),
        'appended_comparison_refs': [], 'selected_ref': parent_ref, 'selected_mse': 0.,
        'complete_current_pool_occurrences': inherited['candidate_occurrences']})
    save(current / 'registration.json', {**load(source / 'registration.json'), 'parent_ref': parent_ref})
    save(current / 'receipt.json', {'status': 'response_distinguished', 'full_comparison': True,
        'remaining_compute': 3659, 'selected_ref': parent_ref, 'selected_loss': 0.})
    request = tmp_path / 'partition_request.json'
    save(request, {'partition_source': str(donor), 'reserved_compute': 96 + 177})
    return source, donor, current, request, parent, all_rows


def test_old_selected_parent_reuses_completed174_partition_and_only_processes_three_new_rows(tmp_path, monkeypatch):
    from hegel_machine.cognition import relation_uncertainty

    source, donor, current, request, parent, _ = current_parent_with_newer_partition(tmp_path, monkeypatch)
    protected = {path: path.read_bytes() for root in (source, donor, current) for path in root.rglob('*.json')}
    output = tmp_path / 'updated177'
    cfg = runner.register_interval_probe_return(current, output, request=request)
    assert (cfg['training_rows'], cfg['partition_rows'], cfg['comparison_rows']) == (96, 174, 177)
    assert (cfg['available_compute'], cfg['compute_limit'], cfg['reserved_compute']) == (3659, 3386, 273)
    assert cfg['partition_ref'] != cfg['parent_ref']
    assert runner.read_method_reference(cfg['partition_parent_score_ref'], output) == 0.
    forbid_scientific_restart(monkeypatch)
    examined = []
    distances = relation_uncertainty._distances

    def collect_new_geometry(relations, frames, budget):
        frames = list(frames)
        examined.extend(frames)
        return distances(relations, frames, budget)

    monkeypatch.setattr(relation_uncertainty, '_distances', collect_new_geometry)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    result = runner.run_interval_probe_return(output)
    assert result['status'] == 'paid_interval_updated' and result['execution_complete']
    assert result['parent_program_unchanged'] and result['selected_loss'] == 0.
    assert result['new_fits'] == result['new_observations'] == result['model_calls'] == 0
    assert len(examined) == 3 and [row.features['x1'] for row in examined] == [.77, .78, .79]
    assert result['remaining_compute'] == 3659 - result['compute_cost'] >= 273
    update = load(output / 'interval_result.json')
    assert update['reuse_statistics']['geometry_paid_rows_reused'] == 174
    assert update['before_intervals'][0]['radius_intervals'] == [[.72, .78]]
    assert update['after_intervals'][0]['radius_intervals'] == [[.77, .78]]
    selected = runner.retained_entry(result['selected_ref'], output, 0)
    assert selected['candidate'].program == parent['candidate'].program
    body = selected['research_program']
    old_fit, fit = (research_program_fit_contract(entry['research_program']) for entry in (parent, selected))
    assert (body['fit_rows'], body['evaluation_rows']) == (96, 177)
    assert len(fit['fit_terms']) == 8 and fit['fit_terms'] == old_fit['fit_terms']
    assert fit['coefficients'] == old_fit['coefficients']
    assert body['selection_steps'][-1]['parameters'][0]['radius_intervals'] == [[.77, .78]]
    old_scores = load(current / 'paid_pool/receipt.json')['scores']
    new_scores = load(output / 'paid_pool/receipt.json')['scores']
    for old in old_scores:
        new = next(item for item in new_scores if old['candidate_refs'][0] in item['candidate_refs'])
        assert new['mse'] == old['mse']
    assert all(path.read_bytes() == value for path, value in protected.items())
    assert runner.run_interval_probe_return(output)['new_compute_cost'] == 0
    assert len(examined) == 3
    gate = result['confirmation_gate']
    assert gate['all_paid_squared_error_upper_bound'] == 0.
    assert not gate['admitted'] and gate['competition_review_needed']
    # Exact working predictions do not permit a new panel while the original
    # interval/competition condition remains unresolved.
    from hegel_machine.cognition import anonymous_object_lab
    save(tmp_path / 'task.json', {'seed': 123})
    monkeypatch.setattr(anonymous_object_lab, 'AnonymousObjectLab',
        lambda **kwargs: pytest.fail('Unresolved competition must stop before preparing any panel'))
    confirmation = tmp_path / 'blocked_confirmation'
    with pytest.raises(ValueError, match='remaining-competition confirmation gate'):
        runner.register_confirmation(output, confirmation)
    assert not confirmation.exists()


def test_partition_from_different_paid_prefix_is_rejected_before_start(tmp_path, monkeypatch):
    _, _, current, request, _, _ = current_parent_with_newer_partition(tmp_path, monkeypatch)
    cfg = load(current / 'paid_pool/registration.json')
    different = deepcopy(runner.read_method_reference(cfg['data_refs'][0], current))
    different['batch']['observations'][0]['target'] += 1.
    different_path = current / 'different_source.json'
    save(different_path, different)
    cfg['data_refs'][0] = runner.ref(different_path)
    save(current / 'paid_pool/registration.json', cfg)
    output = tmp_path / 'incorrect_donor'
    with pytest.raises(ValueError, match='same paid prefix'):
        runner.register_interval_probe_return(current, output, request=request)
    assert not output.exists()
