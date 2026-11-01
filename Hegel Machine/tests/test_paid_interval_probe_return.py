"""A purchased distinguishing probe updates intervals without refitting its readout."""

from dataclasses import asdict, replace
from statistics import mean

import pytest

from scripts import review_paid_object_pool as runner
from scripts.run_partial_role_parent import load, save
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import SearchBudget, TaskSpec
from hegel_machine.cognition.object_continuation import encode_entry
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
from hegel_machine.cognition.structural_research_program import research_program_fit_contract
from hegel_machine.cognition.transfer_partitions import finalize_interval_candidates
from test_interval_candidate_finalization import evaluated_interval_candidates


def paid_probe_scene(tmp_path, monkeypatch, *, balance=8434):
    """Keep 96 fitted rows, 168 previous interval rows, and six later paid rows."""
    source = tmp_path / 'response_probe'
    common = source / 'paid_pool'
    initial, _, rows, _, _, _ = evaluated_interval_candidates()
    for index, distance in enumerate((.74, .76) * 3, 168):
        row = replace(rows[index], features={**rows[index].features, 'x1': distance})
        rows[index] = replace(row, target=initial['candidate'].program.predict(row))
    budget = SearchBudget(15000)
    research = RelationIntervalResearch(initial['candidate'].program, [], budget,
        research_program=initial['research_program'])
    entries = research.alternatives(rows[:168], budget, 1e-7,
        experiments=[rows[96:144], rows[144:168]], interval_precision=0., parameter_index=0)
    finalize_interval_candidates(initial, entries, research, rows[:96], rows[96:168],
        feedback_experiments=[rows[96:144], rows[144:168]])
    selected_index = next(index for index, entry in enumerate(entries)
        if entry['candidate'].program == initial['candidate'].program)
    save(source / 'old_entries.json', [encode_entry(entry) for entry in entries])
    candidate_refs = [runner.ref(source / 'old_entries.json', index) for index in range(len(entries))]
    selected_ref = candidate_refs[selected_index]
    scores = [{'candidate_refs': [location], 'mse': mean(
        (entry['candidate'].program.predict(row) - row.target) ** 2 for row in rows)}
        for location, entry in zip(candidate_refs, entries, strict=True)]
    public = TaskSpec('anonymous', 'Paid observations', tuple(rows[0].features), (), (),
        feature_bounds={key: (-2., 2.) for key in rows[0].features}, max_probe_length=96)
    save(common / 'public_spec.json', asdict(public))
    refs = []
    for index, (start, end) in enumerate(((0, 96), (96, 144), (144, 168), (168, 171), (171, 174))):
        path = common / f'probe{index}.json'
        save(path, {'probe': {'probe_id': f'paid{index}',
            'sequence': [row.features for row in rows[start:end]]},
            'batch': {'observations': [asdict(row) for row in rows[start:end]]}})
        refs.append(runner.ref(path))
    save(common / 'registration.json', {'data_refs': refs[:3],
        'appended_observation_refs': refs[3:], 'public_spec_ref': runner.ref(common / 'public_spec.json'),
        'training_rows': 96, 'feedback_rows': 78, 'comparison_rows': 174, 'quality_threshold': 1e-7})
    save(common / 'receipt.json', {'full_comparison': True, 'rows': 174, 'scores': scores})
    save(source / 'retained_frontier.json', {'full_comparison': True,
        'original_pool_ref': runner.ref(common / 'receipt.json'), 'appended_comparison_refs': [],
        'complete_current_pool_occurrences': len(entries),
        'selected_ref': selected_ref, 'selected_mse': scores[selected_index]['mse']})
    save(source / 'registration.json', {'source_task_ref': runner.ref(tmp_path / 'task.json'),
        'active_wall_seconds': 30, 'quality_threshold': 1e-7,
        'candidate_refs': candidate_refs, 'parent_ref': selected_ref,
        'prior_method_review': str(tmp_path / 'previous_method_return'),
        'object_model_ref': runner.ref(source / 'object_model.json'),
        'source_costs_ref': runner.ref(source / 'old_costs.json'),
        'source_cost_refs': [runner.ref(source / 'old_costs.json')],
        'experiment_methods_ref': runner.ref(source / 'program_experiment_methods.json'),
        'binding_ref': runner.ref(source / 'instrument_binding.json')})
    save(source / 'receipt.json', {'status': 'response_distinguished',
        'execution_complete': True, 'full_comparison': True,
        'selected_ref': selected_ref, 'selected_loss': scores[selected_index]['mse'],
        'remaining_compute': balance, 'candidate_refs': candidate_refs,
        'source_task_ref': runner.ref(tmp_path / 'task.json')})
    save(source / 'program_experiment_methods.json', [])
    save(source / 'instrument_models.json', [])
    save(source / 'instrument_binding.json',
        {name: Program('feature', name=name).to_dict() for name in public.feature_names})
    monkeypatch.setattr(runner, 'freeze_entry', lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, 'write_text_incrementally', lambda *args, **kwargs: None)
    return source, entries, rows


def forbid_scientific_restart(monkeypatch):
    from hegel_machine.cognition import anonymous_object_lab, research_programs, transfer_partitions
    from scripts import continue_successor_relation_intervals

    def forbidden(*args, **kwargs):
        pytest.fail('The paid interval return must not acquire, refit, or rescore the old pool')

    monkeypatch.setattr(anonymous_object_lab.AnonymousObjectLab, 'acquire', forbidden)
    monkeypatch.setattr(research_programs._Fitter, 'fit', forbidden)
    monkeypatch.setattr(transfer_partitions, 'continue_transfer_partition', forbidden)
    monkeypatch.setattr(continue_successor_relation_intervals, 'extend_scores', forbidden)


def test_supported_parent_consumes_new_probe_intervals_without_repeating_paid_research(tmp_path, monkeypatch):
    from hegel_machine.cognition import relation_uncertainty

    source, original_entries, _ = paid_probe_scene(tmp_path, monkeypatch)
    protected = {path: path.read_bytes() for path in source.rglob('*.json')}
    output = tmp_path / 'returned_intervals'
    cfg = runner.register_interval_probe_return(source, output)
    assert (cfg['training_rows'], cfg['partition_rows'], cfg['comparison_rows']) == (96, 168, 174)
    assert (cfg['available_compute'], cfg['compute_limit'], cfg['reserved_compute']) == (8434, 5338, 3096)
    original = runner.retained_entry(cfg['parent_ref'], output, 0)
    old_fit = research_program_fit_contract(original['research_program'])
    assert len(old_fit['fit_terms']) == 8
    forbid_scientific_restart(monkeypatch)
    examined = []
    distances = relation_uncertainty._distances

    def measured_distances(relations, frames, budget):
        frames = list(frames)
        examined.extend(frames)
        return distances(relations, frames, budget)

    monkeypatch.setattr(relation_uncertainty, '_distances', measured_distances)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    result = runner.run_interval_probe_return(output)
    assert result['status'] == 'paid_interval_updated' and result['execution_complete']
    assert result['selected_loss'] == 0. and result['parent_program_unchanged']
    assert result['new_fits'] == result['new_observations'] == result['model_calls'] == 0
    assert result['remaining_compute'] == 8434 - result['compute_cost'] >= 3096
    assert len(examined) == 6 and {frame.features['x1'] for frame in examined} == {.74, .76}
    update = load(output / 'interval_result.json')
    assert update['fixed_readout_only'] and update['new_fits'] == 0
    assert update['before_intervals'][0]['radius_intervals'] == [[.7, .8]]
    assert update['after_intervals'][0]['radius_intervals'] == [[.74, .76]]
    assert update['reuse_statistics']['geometry_paid_rows_reused'] == 168
    selected = runner.retained_entry(result['selected_ref'], output, 0)
    assert selected['candidate'].program == original['candidate'].program
    assert result['selected_ref'] != cfg['parent_ref']
    fit = research_program_fit_contract(selected['research_program'])
    assert fit['fit_terms'] == old_fit['fit_terms'] and fit['coefficients'] == old_fit['coefficients']
    assert (selected['research_program']['fit_rows'], selected['research_program']['evaluation_rows']) == (96, 174)
    assert selected['research_program']['selection_steps'][-1]['parameters'][0]['radius_intervals'] == [[.74, .76]]
    before_scores = load(source / 'paid_pool/receipt.json')['scores']
    after_scores = load(output / 'paid_pool/receipt.json')['scores']
    for before in before_scores:
        after = next(item for item in after_scores if before['candidate_refs'][0] in item['candidate_refs'])
        assert after['mse'] == before['mse']
    frontier, _, rows, _, refs, pool, restored, _ = runner.parent_inputs(output)
    assert len(rows) == 174 and len(refs) == len(pool) == len(original_entries) + result['new_candidates']
    assert frontier['selected_ref'] == result['selected_ref']
    assert restored['research_program']['evaluation_rows'] == 174
    assert all(path.read_bytes() == value for path, value in protected.items())
    previous_count = len(examined)
    reused = runner.run_interval_probe_return(output)
    assert reused['reused'] and reused['new_compute_cost'] == 0 and len(examined) == previous_count


def test_unfinished_interval_research_keeps_paid_parent_and_cannot_claim_success(tmp_path, monkeypatch):
    source, original_entries, _ = paid_probe_scene(tmp_path, monkeypatch)
    output = tmp_path / 'unfinished_intervals'
    cfg = runner.register_interval_probe_return(source, output)
    forbid_scientific_restart(monkeypatch)

    def unfinished(self, rows, budget, tolerance, *, experiments):
        assert len(rows) == 174 and [len(part) for part in experiments] == [48, 24, 3, 3]
        assert budget.spend(17)
        assert not budget.spend(budget.limit + 1)
        self.status = 'compute_exhausted'
        return []

    monkeypatch.setattr(RelationIntervalResearch, 'alternatives', unfinished)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    result = runner.run_interval_probe_return(output)
    assert result['status'] == 'interval_update_incomplete' and not result['execution_complete']
    assert result['selected_ref'] == cfg['parent_ref'] and result['selected_loss'] == 0.
    assert result['candidate_refs'] == [] and result['new_candidates'] == 0
    assert result['pool_occurrences'] == len(original_entries)
    assert result['remaining_compute'] == 8434 - result['compute_cost']
    assert result['new_fits'] == result['new_observations'] == 0
    update = load(output / 'interval_result.json')
    assert update['status'] == 'compute_exhausted' and not update['execution_complete']
    assert runner.run_interval_probe_return(output)['new_compute_cost'] == 0
