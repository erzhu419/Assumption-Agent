"""Forked returns continue the actual formed-method topic without rebuilding it."""

from copy import deepcopy
from pathlib import Path

from scripts.return_partition_pair import merge_pair_returns
from hegel_machine.cognition.method_action_research import read_method_reference
from test_staged_method_research import fixture
from test_native_paired_method_feedback import save, route


def test_merge_keeps_formed_proposal_two_new_origins_and_reuses_completed_inputs(tmp_path, monkeypatch):
    agent, group, _ = fixture(tmp_path)
    formed = route(agent)
    prior = tmp_path
    save(prior / 'receipt.json', {'compute_cost': agent.costs['compute_cost'], 'return': formed})
    source = formed['method_action_research']['staged_method_proposals'][0]
    old_proposal = Path(source['proposal_ref']['file']).read_bytes()
    stage_artifact = save(tmp_path / 'staged_artifact.json', {
        'id': 'staged_actual_candidate', 'version': 1, 'kind': 'native_research_segment',
        'execution': {'entry': 'replay_research_program',
            'research_program_ref': source['research_program_refs'][0]}})
    original_pair = read_method_reference(group['pair_refs'][0], agent.workspace)
    template = read_method_reference(original_pair['challenger_ref'], agent.workspace)
    refs, originals = [], []
    for i in range(2):
        root = tmp_path / f'fork{i}'
        # The actual old successors froze the same body at two different
        # paths. Those paths do not create two different method contents.
        artifact = read_method_reference(stage_artifact, agent.workspace)
        artifact['execution']['research_program_ref'] = save(root / 'frozen_method.json',
            read_method_reference(source['research_program_refs'][0], agent.workspace))
        fork_artifact = save(root / 'staged_artifact.json', artifact)
        contract = {name + '_ref': save(root / (name + '.json'), {'source': i})
            for name in ('start_state', 'data', 'split', 'evaluation')}
        contract['quality_contract_ref'] = save(root / 'quality.json', {'threshold': 1e-7})
        contract['resource_contract_ref'] = save(root / 'resource.json', {'compute_limit': 1000})
        baseline = {**deepcopy(template), **contract, 'loss': 0., 'completed': True,
            'costs': {'compute_cost': 30, 'observation_cost': 0, 'measurement_cost': 0}}
        challenger = {**deepcopy(baseline), 'method_ref': {'id': 'staged_actual_candidate', 'version': 1},
            'method_artifact_ref': fork_artifact, 'loss': .2 if i else 0.,
            'completed': not i, 'costs': {'compute_cost': 20, 'observation_cost': 0, 'measurement_cost': 0}}
        pair = {**deepcopy(original_pair), 'baseline_ref': save(root / 'baseline.json', baseline),
            'challenger_ref': save(root / 'challenger.json', challenger),
            'origin_ref': {'file': str(root / 'source.json')},
            'physical_source_ref': {'file': str(root / 'source.json')},
            'baseline_action_content': {'method': 'fixed_Pi1'},
            'challenger_action_content': {'method': 'actual_staged_program'}}
        refs.append(save(root / 'pair.json', pair))
        originals.append(Path(refs[-1]['file']).read_bytes())
    allocation = save(tmp_path / 'allocation.json', {
        'allocation_id': 'paid_stage_remaining', 'maximum_compute': 2000, 'new_observation_limit': 0})
    # Only new condition evaluation is allowed: the old stage content already
    # has an executed formation receipt and must be reused by reference.
    from hegel_machine.cognition import staged_method_research
    monkeypatch.setattr(staged_method_research, 'staged_program',
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('old proposal must be reused')))
    merged = merge_pair_returns(prior, tmp_path / 'merged', refs, allocation)
    review = merged['return']['research_result']['method_action_research']
    assert merged['compute_cost'] <= 2000
    assert merged['new_fits'] == merged['new_predictions'] == merged['new_observations'] == 0
    assert review['staged_method_proposals'][0]['reused']
    new = next(item for item in review['paired_condition_research']
               if item['method_ref']['id'] == 'staged_actual_candidate')
    assert new['paired_invocations'] == 2 and new['pair_refs'] == refs
    assert not new['general_guard_supported'] and new['condition'] is None
    assert Path(source['proposal_ref']['file']).read_bytes() == old_proposal
    assert [Path(ref['file']).read_bytes() for ref in refs] == originals
    repeat = merge_pair_returns(prior, tmp_path / 'merged', refs, allocation)
    assert repeat['reused'] and repeat['new_compute_cost'] == 0
