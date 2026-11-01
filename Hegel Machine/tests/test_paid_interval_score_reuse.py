"""Completed-cell errors remain reusable through the paid interval entry."""

from dataclasses import replace

from scripts import review_paid_object_pool as runner
from scripts.run_partial_role_parent import load, save
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.object_continuation import decode_entry, encode_entry
from test_paid_interval_partition_donor import current_parent_with_newer_partition
from test_paid_interval_probe_return import paid_probe_scene, forbid_scientific_restart


def test_paid_cell_reference_finishes_with_330_balance_and_original_confirmation_reserve(tmp_path, monkeypatch):
    import test_paid_interval_probe_return as fixture

    evaluated = fixture.evaluated_interval_candidates

    def directional_response():
        initial, research, rows, experiments, candidates, budget = evaluated()

        def directional(value):
            if isinstance(value, list):
                return [directional(item) for item in value]
            if isinstance(value, dict):
                value = {key: directional(item) for key, item in value.items()}
                if value.get('op') == 'neighbor_sum':
                    value['name'] = 'right'
                if value.get('relation') == 'symmetric':
                    value['relation'] = 'right'
            return value

        initial = decode_entry(directional(encode_entry(initial)))
        rows = [replace(row, target=initial['candidate'].program.predict(row)) for row in rows]
        return initial, research, rows, experiments, candidates, budget

    monkeypatch.setattr(fixture, 'evaluated_interval_candidates', directional_response)
    _, donor, current, request, parent, rows = current_parent_with_newer_partition(tmp_path, monkeypatch)
    receipt = load(current / 'receipt.json')
    save(current / 'receipt.json', dict(receipt, remaining_compute=330))
    save(request, {'partition_source': str(donor), 'reserved_compute': 96})
    protected = {path: path.read_bytes() for root in (donor, current) for path in root.rglob('*.json')}
    output = tmp_path / 'paid_cell_return'
    cfg = runner.register_interval_probe_return(current, output, request=request)
    assert (cfg['available_compute'], cfg['compute_limit'], cfg['reserved_compute']) == (330, 234, 96)
    assert cfg['paid_cell_score_ref'] is not None
    stored = runner.retained_entry(cfg['paid_cell_score_ref'], output, 0)
    assert len(stored['paid_row_squared_errors']) == cfg['partition_rows'] == 174
    assert stored['candidate'].program != parent['candidate'].program
    assert 'paid_row_squared_errors' not in load(output / 'registration.json')
    forbid_scientific_restart(monkeypatch)
    original_predict = Program.predict
    original_inputs = runner.parent_inputs
    paid_ids = set()
    predictions = []

    def restore_paid_inputs(*args, **kwargs):
        result = original_inputs(*args, **kwargs)
        paid_ids.update(map(id, result[2][:174]))
        return result

    def only_new_rows(program, row, *args, **kwargs):
        assert id(row) not in paid_ids, 'Paid prefix predictions must be read from saved row errors'
        predictions.append(row)
        return original_predict(program, row, *args, **kwargs)

    monkeypatch.setattr(runner, 'parent_inputs', restore_paid_inputs)
    monkeypatch.setattr(Program, 'predict', only_new_rows)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    result = runner.run_interval_probe_return(output)
    assert result['status'] == 'paid_interval_updated' and result['execution_complete']
    assert result['paid_cell_scores_reused'] and result['paid_cell_score_ref'] == cfg['paid_cell_score_ref']
    assert result['compute_cost'] <= 234 and result['remaining_compute'] >= 96
    assert result['parent_program_unchanged'] and result['selected_loss'] == 0.
    assert predictions and all(any(row.features == new.features for new in rows[174:])
                              for row in predictions)
    assert result['new_fits'] == result['new_observations'] == result['model_calls'] == 0
    update = load(output / 'interval_result.json')
    assert update['reuse_statistics']['cell_score_rows_bound'] == 174
    assert update['reuse_statistics']['cell_predictions_reused'] >= 174
    assert not result['confirmation_gate']['admitted']
    assert result['confirmation_gate']['competition_review_needed']
    old_scores = load(current / 'paid_pool/receipt.json')['scores']
    new_scores = load(output / 'paid_pool/receipt.json')['scores']
    for old in old_scores:
        new = next(item for item in new_scores if old['candidate_refs'][0] in item['candidate_refs'])
        assert new['mse'] == old['mse']
    assert all(path.read_bytes() == content for path, content in protected.items())
    count = len(predictions)
    assert runner.run_interval_probe_return(output)['new_compute_cost'] == 0
    assert len(predictions) == count


def test_aggregate_only_parent_keeps_existing_scoring_path_and_default_reserve(tmp_path, monkeypatch):
    source, _, _ = paid_probe_scene(tmp_path, monkeypatch)
    path = source / 'old_entries.json'
    entries = load(path)
    for entry in entries:
        entry.pop('paid_row_squared_errors', None)
    save(path, entries)
    output = tmp_path / 'aggregate_only_return'
    cfg = runner.register_interval_probe_return(source, output)
    assert cfg['paid_cell_score_ref'] is None
    assert cfg['reserved_compute'] == 3096
    forbid_scientific_restart(monkeypatch)
    monkeypatch.setattr(runner, 'ROOT', output / 'runtime')
    result = runner.run_interval_probe_return(output)
    assert result['execution_complete'] and result['parent_program_unchanged']
    assert not result['paid_cell_scores_reused'] and result['selected_loss'] == 0.
    assert not result['confirmation_gate']['admitted']
