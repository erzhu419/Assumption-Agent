"""ABC rebuilds the actual sequential method without receiving its fitted answer."""

from copy import deepcopy
from pathlib import Path

import pytest

from hegel_machine.cognition.common_object_research import CommonObjectResearchAgent
from hegel_machine.cognition.method_action_research import read_method_reference
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.source_method_reconstruction import reconstruct_partition_source
from scripts.run_partial_role_successors import frozen_program_bank
from scripts.run_partial_role_parent import save
from test_native_method_composition import composition_source
from test_native_method_return_route import route
from test_program_source_abc import source_fixture


def paid_source(tmp_path):
    sequential = tmp_path / 'sequential'
    sequential.mkdir()
    common, _, _, _, _ = composition_source(sequential)
    formed = route(common, allowance=1000000)['method_action_research']['staged_method_proposals']
    proposal_ref = next(item['proposal_ref'] for item in formed if item.get('candidate_count') == 1)
    sequence = read_method_reference(proposal_ref, common.workspace)
    method = sequence['proposal']['candidates'][0]
    earlier = tmp_path / 'earlier'
    earlier.mkdir()
    raw, _, proposal = source_fixture(earlier)
    for name in ('before_ref', 'after_ref'):
        raw[name]['file'] = str(earlier / raw[name]['file'])
    from hegel_machine.cognition.transfer_partitions import propose_transfer_partitions
    proposal = propose_transfer_partitions(read_method_reference(raw['before_ref'], earlier),
        read_method_reference(raw['after_ref'], earlier), source_ref=raw['source_ref'],
        before_ref=raw['before_ref'], after_ref=raw['after_ref'])
    save(earlier / 'proposal.json', {'proposal': proposal})
    fitted = proposal['candidates'][0]['research_program']
    save(earlier / 'study.json', {'basis': {'minimum_improvement': 1e-7}, 'methods': [{
        'index': 1, 'execution_complete': True, 'training_loss': 0., 'validation_loss': .1,
        'research_program': fitted}]})
    raw.update(proposal_ref={'file': str(earlier / 'proposal.json')},
        study_ref={'file': str(earlier / 'study.json')},
        composition_execution_return_ref=sequence['source_execution_return_ref'],
        composition_baseline_pair_ref=sequence['baseline_input_provenance_ref'])
    return raw, proposal, method, sequence


def test_ABC_reconstructs_same_full_composition_bank_with_real_analysis_cost(tmp_path, monkeypatch):
    from hegel_machine.cognition import research_programs, staged_method_research

    raw, proposal, method, sequence = paid_source(tmp_path)
    old_files = {path: path.read_bytes() for root in ('sequential', 'earlier')
                 for path in (tmp_path / root).rglob('*.json')}
    frozen = frozen_program_bank(raw, method, proposal, workspace=tmp_path)
    native, calls = staged_method_research.propose_native_method_compositions, []

    def form(*args, **kwargs):
        calls.append(args[2])
        return native(*args, **kwargs)

    monkeypatch.setattr(staged_method_research, 'propose_native_method_compositions', form)
    monkeypatch.setattr(research_programs._Fitter, 'fit',
                        lambda *args, **kwargs: pytest.fail('source reconstruction must not fit target answers'))
    banks, costs = {}, {}
    for arm in ('archive', 'fixed_pr2', 'pr3'):
        agent = CommonObjectResearchAgent(workspace=tmp_path / arm, organization=arm,
            program_sources=[deepcopy(raw)], learned_programs=deepcopy(frozen) if arm == 'pr3' else [])
        budget = SearchBudget(100000)
        costs[arm] = agent._prepare_program_sources(budget)
        banks[arm] = [item['research_program'] for item in agent.learned_programs]
        assert agent._prepare_program_sources(budget) == 0
        if arm != 'pr3':
            rebuilt = agent.program_reconstruction_receipts[0]
            assert rebuilt['status'] == 'reconstructed'
            assert rebuilt['composition_reconstruction']['compute_cost'] == sequence['analysis_compute_cost']
            assert rebuilt['new_fits'] == rebuilt['new_observations'] == 0
    assert banks['archive'] == banks['fixed_pr2'] == banks['pr3']
    assert banks['archive'][0] == method['research_program']
    assert len(banks['archive']) >= 5  # composition, components, prior one-/two-stage and baseline
    assert costs['archive'] == costs['fixed_pr2'] > 0 and costs['pr3'] == 0
    assert len(calls) == 2  # A/B really invoked the ordinary proposer, C reused paid knowledge.
    assert all(path.read_bytes() == content for path, content in old_files.items())


@pytest.mark.parametrize('failure', ['missing_prefix', 'incomplete_execution', 'wrong_producer'])
def test_missing_real_sequence_cannot_be_replaced_with_a_parallel_method(tmp_path, failure):
    raw, proposal, method, sequence = paid_source(tmp_path)
    if failure == 'missing_prefix':
        raw.pop('composition_baseline_pair_ref')
    elif failure == 'incomplete_execution':
        path = Path(raw['composition_execution_return_ref']['file'])
        record = read_method_reference(raw['composition_execution_return_ref'], tmp_path)
        save(path, {**record, 'execution_complete': False})
    else:
        pair = read_method_reference(raw['composition_baseline_pair_ref'], tmp_path)
        for side in ('baseline', 'challenger'):
            path = Path(pair[side + '_ref']['file'])
            outcome = read_method_reference(pair[side + '_ref'], tmp_path)
            save(path, {**outcome, 'actual_outcome_ref': {'file': str(tmp_path / 'unrelated.json')}})
    result = reconstruct_partition_source(raw, tmp_path / 'rebuild', SearchBudget(100000))
    assert result['status'] == 'composition_dependency_unresolved'
    assert result['methods']  # every paid earlier method remains usable
    assert all(not item['research_program'].get('staged_research') for item in result['methods'])
