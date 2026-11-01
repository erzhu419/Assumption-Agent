"""Completed interval execution returns without rewriting its original receipt."""

from pathlib import Path

import pytest

from scripts import review_paid_object_pool as runner
from scripts.run_partial_role_parent import load, save
from hegel_machine.cognition.method_action_research import read_method_reference
from test_native_partition_proposals import interval_execution_scene, forbidden


@pytest.mark.parametrize('separate_allocation,partial_first', [(False, False), (True, False), (False, True)])
def test_missing_interval_producer_fields_are_recovered_before_ordinary_method_return(
        tmp_path, monkeypatch, separate_allocation, partial_first):
    from hegel_machine.cognition import transfer_partitions
    from hegel_machine.cognition.language import Program

    prior = tmp_path / 'prior'
    prior.mkdir()
    common, _, evidence_ref, _, _ = interval_execution_scene(prior)
    evidence = read_method_reference(evidence_ref, common.workspace)
    from hegel_machine.cognition.structural_research_program import research_program_fit_contract
    for key in ('before_entry_ref', 'after_entry_ref'):
        payload = read_method_reference(evidence[key], common.workspace)
        fit = research_program_fit_contract(payload['research_program'])
        save(Path(evidence[key]['file']), {**payload, 'program': fit['program'].to_dict(),
            'training_loss': 0., 'validation_loss': 0.})
    owner = next(q for q in common.questions if q['id'] == 'task:method_choice')
    owner['method_use_plan']['native_return_refs'] = []
    save(prior / 'common/questions.json', common.questions)
    raw_ref = evidence['method_execution_ref']
    raw = read_method_reference(raw_ref, common.workspace)
    raw.pop('fixed_readout_only')
    raw.pop('new_fits')
    save(Path(raw_ref['file']), raw)
    source = tmp_path / 'completed_interval'
    cfg = {'prior_method_review': str(prior), 'source_task_ref': evidence['source_task_ref'],
        'parent_ref': evidence['before_entry_ref'], 'research_action': 'paid_interval_uncertainty_update'}
    save(source / 'registration.json', cfg)
    actual = {'status': 'paid_interval_updated', 'execution_complete': True,
        'full_comparison': True, 'selected_ref': evidence['after_entry_ref'],
        'parent_ref': evidence['before_entry_ref'], 'remaining_compute': 221,
        'new_fits': 0, 'method_invoked': True,
        'executed_method_ref': evidence['executed_method_ref'], 'method_execution_ref': raw_ref,
        'comparison_ref': evidence['comparison_ref'],
        'confirmation_gate': {'admitted': False, 'stop_reason': 'unresolved'}}
    save(source / 'receipt.json', actual)
    original = Path(raw_ref['file']).read_bytes()
    monkeypatch.setattr(transfer_partitions, 'propose_transfer_partitions', forbidden)
    monkeypatch.setattr(Program, 'predict', forbidden)
    allowance = 2000 if separate_allocation else 221
    allocation_ref = None
    if separate_allocation:
        allocation_ref = runner.ref(tmp_path / 'allocation.json')
        allocation = {'source_return_ref': runner.ref(source / 'receipt.json'), 'maximum_compute': allowance}
        save(Path(allocation_ref['file']), allocation)
    partial = None
    if partial_first:
        partial = tmp_path / 'partial'
        incomplete = runner.return_parent(source, partial, maximum_compute=2)
        assert incomplete['status'] == 'scoped_unknown'
    result = runner.return_parent(source, tmp_path / 'returned', maximum_compute=allowance,
        allocation_ref=allocation_ref, previous_return=partial)
    assert result['status'] == 'complete_native_return_review'
    reviewed = result['return']['research_result']['method_action_research']
    returned = next(item for item in reviewed['existing_method_executions']
        if item['method_execution_ref'] == result['method_execution_ref'])
    assert returned['execution_complete']
    recovered = read_method_reference(result['method_execution_ref'], tmp_path)
    assert recovered['fixed_readout_only'] and recovered['new_fits'] == 0
    assert recovered['raw_execution_ref'] == raw_ref
    assert recovered['native_result_ref'] == runner.ref(source / 'receipt.json')
    if partial_first:
        assert result['method_execution_ref'] == incomplete['method_execution_ref']
        assert not (tmp_path / 'returned/interval_execution_return.json').exists()
    assert result['compute_cost'] <= allowance
    if separate_allocation:
        assert result['remaining_compute'] == allowance - result['compute_cost']
        assert result['original_ledger']['remaining_compute'] == 221
        assert load(Path(allocation_ref['file'])) == allocation
    assert result['new_fits'] == result['new_predictions'] == result['new_observations'] == 0
    assert not result['independent_support'] and not result['method_pair_created']
    assert Path(raw_ref['file']).read_bytes() == original
    assert load(source / 'receipt.json') == actual
    assert runner.return_parent(source, tmp_path / 'returned')['new_compute_cost'] == 0
