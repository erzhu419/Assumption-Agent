"""Existing interval executions use contract checks, not proposal searches."""

from pathlib import Path

import pytest

from hegel_machine.cognition import method_action_research as methods
from test_native_method_return_route import route, save
from test_native_partition_proposals import interval_execution_scene, forbidden


def test_existing_execution_finishes_below_new_proposal_quote_and_reuses_completion(tmp_path, monkeypatch):
    common, _, _, before, after = interval_execution_scene(tmp_path)
    required = 2 + methods.quote_existing_interval_return(before, after)
    assert required < 2 + methods.quote_native_partition_analysis(before, after)
    monkeypatch.setattr(methods, 'quote_native_partition_analysis', forbidden)
    from hegel_machine.cognition import transfer_partitions
    monkeypatch.setattr(transfer_partitions, 'propose_transfer_partitions', forbidden)

    result = route(common, allowance=required)
    assert result['status'] == 'complete_native_return_review'
    assert result['new_compute_cost'] == required
    action = result['method_action_research']
    assert action['relative_labels_created'] == 0
    assert action['existing_method_executions'][0]['execution_complete']
    proposal, = action['transfer_partition_proposals']
    assert proposal['candidate_refs'] == []
    assert proposal['status'] == 'existing_method_execution_returned'
    bundle = methods.read_method_reference(proposal['proposal_ref'], common.workspace)
    assert bundle['analysis_compute_cost'] == required - 2
    assert bundle['review_compute_cost'] == required - 2
    saved = Path(proposal['proposal_ref']['file']).read_bytes()
    monkeypatch.setattr(methods, '_existing_method_execution_return', forbidden)
    assert route(common, allowance=required)['new_compute_cost'] == 0
    assert Path(proposal['proposal_ref']['file']).read_bytes() == saved


def test_existing_execution_quotes_complete_return_before_publishing_when_unfunded(tmp_path, monkeypatch):
    common, _, _, before, after = interval_execution_scene(tmp_path)
    required = 2 + methods.quote_existing_interval_return(before, after)
    from hegel_machine.cognition import transfer_partitions
    monkeypatch.setattr(transfer_partitions, 'propose_transfer_partitions', forbidden)
    result = route(common, allowance=required - 1)
    assert result['status'] == 'scoped_unknown'
    action = result['method_action_research']
    assert action['reason'] == 'existing_method_return_check_not_funded'
    assert action['required_compute'] == required
    assert result['new_compute_cost'] == 2
    assert not list((common.workspace / 'method_feedback').glob('transfer_partition_*.json'))


@pytest.mark.parametrize('change', ['changed_readout', 'missing_fixed_readout', 'missing_new_fits'])
def test_nonmatching_or_undeclared_execution_keeps_full_analysis_and_budget(tmp_path, monkeypatch, change):
    common, _, evidence_ref, before, after = interval_execution_scene(
        tmp_path, changed_readout=change == 'changed_readout')
    if change.startswith('missing_'):
        evidence = methods.read_method_reference(evidence_ref, common.workspace)
        execution_ref = evidence['method_execution_ref']
        execution = methods.read_method_reference(execution_ref, common.workspace)
        execution.pop('fixed_readout_only' if change == 'missing_fixed_readout' else 'new_fits')
        save(Path(execution_ref['file']), execution)
    required = 2 + methods.quote_native_partition_analysis(before, after)
    from hegel_machine.cognition import transfer_partitions
    propose = transfer_partitions.propose_transfer_partitions
    monkeypatch.setattr(transfer_partitions, 'propose_transfer_partitions', forbidden)
    result = route(common, allowance=required - 1)
    assert result['status'] == 'scoped_unknown'
    assert result['new_compute_cost'] == 2
    assert result['method_action_research']['required_compute'] == required
    assert result['method_action_research']['reason'] == 'native_partition_analysis_not_funded'
    assert not list((common.workspace / 'method_feedback').glob('transfer_partition_*.json'))
    monkeypatch.setattr(transfer_partitions, 'propose_transfer_partitions', propose)
    funded = route(common, allowance=required)
    assert funded['new_compute_cost'] == required
    assert not funded['method_action_research'].get('existing_method_executions')
