"""Paid counterexamples prune frozen stage conditions without resynthesis."""
from copy import deepcopy
from pathlib import Path
import json
import pytest

from hegel_machine.cognition.common_object_research import record_native_parent_research_return
from hegel_machine.cognition.method_action_research import read_method_reference
from hegel_machine.cognition.staged_method_research import (
    baseline_return_profile, record_staged_method_practice, revise_staged_method_conditions)
from test_native_paired_method_feedback import route, save
from test_staged_method_research import fixture


def delivered_stage_pair(agent, root, body, prefix_loss, *, complete=True, mismatched=False):
    stage = body['staged_research']
    methods = {'baseline': stage['continuation'], 'challenger': body}
    artifacts = {side: save(root / (side + '_method.json'), {
        'id': 'fixed' if side == 'baseline' else 'staged', 'version': 1,
        'kind': 'native_research_segment', 'execution': {'entry': 'replay_research_program',
            'research_program_ref': save(root / (side + '_body.json'), program)}})
        for side, program in methods.items()}
    contracts = {key + '_ref': save(root / (key + '.json'), {'source': str(root)})
        for key in ('start_state', 'data', 'split', 'evaluation')}
    contracts['quality_contract_ref'] = save(root / 'quality.json', {'threshold': 1e-7})
    contracts['resource_contract_ref'] = save(root / 'resources.json', {'compute_limit': 10000})
    profile = save(root / 'profile.json', {'residual': {'loss': 100.}})
    sides = {}
    for side, cost in (('baseline', 100), ('challenger', 10)):
        actual = {'execution_complete': True, 'full_comparison': True, 'new_candidates': 1,
                  'development_rows': 24, 'selected_loss': 0.}
        if side == 'challenger':
            actual['staged_research_result'] = {
                'baseline': {'execution_complete': complete, 'all_paid_loss': prefix_loss},
                'guard': {'profile': baseline_return_profile(
                    {'fit_loss': prefix_loss, 'execution_complete': True}, 1e-7)}}
        save(root / side / 'comparison.json', {'full_comparison': True,
            'new_scores': [{'loss': 0., 'candidate_ref': {'file': str(root / side / 'new_entry.json')}}]})
        actual_ref = save(root / side / 'outcome.json', actual)
        value = {**contracts, 'profile_ref': profile, 'actual_outcome_ref': actual_ref,
            'method_artifact_ref': artifacts[side], 'method_ref': {
                'id': 'fixed' if side == 'baseline' else 'staged', 'version': 1},
            'status': 'completed', 'execution_complete': True, 'completed': True,
            'full_comparison': True, 'loss': 0., 'coverage': 24,
            'costs': {'compute_cost': cost, 'observation_cost': 0, 'measurement_cost': 0}}
        if mismatched and side == 'baseline':
            value['data_ref'] = save(root / 'different_data.json', {'different': True})
        sides[side] = save(root / (side + '_return.json'), value)
    pair_ref = save(root / 'pair.json', {'comparison_kind': 'matched_native_segments',
        'baseline_ref': sides['baseline'], 'challenger_ref': sides['challenger'],
        'origin_ref': {'file': str(root / 'registration.json')},
        'physical_source_ref': {'file': str(root / 'registration.json')},
        'data_role': 'condition_validation', 'identification_target': 'fixed_data_method_effect',
        'baseline_action_content': {'method': 'fixed'}, 'challenger_action_content': {'method': 'staged'}})
    returned = save(root / 'return.json', {'status': 'returned', 'method_pair_ref': pair_ref,
        'method_ref': {'id': 'staged', 'version': 1}, 'method_artifact_ref': artifacts['challenger']})
    record_native_parent_research_return(agent, source_task_ref={'file': str(root / 'registration.json')},
        result_ref=returned, mission={'purpose': 'method_learning', 'allocation': {'compute': 100000}})
    return {'baseline_method_artifact_ref': artifacts['baseline'],
        'method_artifact_ref': artifacts['challenger'], 'pair_refs': [pair_ref]}


def setup(tmp_path, **kwargs):
    agent, _, _ = fixture(tmp_path)
    proposal = route(agent)['method_action_research']['staged_method_proposals'][0]
    body = read_method_reference(proposal['research_program_refs'][0], agent.workspace)
    owner = next(q for q in agent.questions if q['id'] == 'task:method_choice')
    held = delivered_stage_pair(agent, tmp_path / 'held', body, 0.)
    changed = delivered_stage_pair(agent, tmp_path / 'changed', body, .05, **kwargs)
    held['pair_refs'].extend(changed['pair_refs'])
    return agent, owner, [held], proposal, body


def test_paid_prefix_counterexample_refutes_old_threshold_without_relearning(tmp_path, monkeypatch):
    agent, owner, groups, proposal, body = setup(tmp_path)
    frozen = Path(proposal['proposal_ref']['file']).read_bytes()
    monkeypatch.setattr('hegel_machine.cognition.condition_learning.learn_condition',
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError('Do not synthesize another threshold')))
    result = revise_staged_method_conditions(agent, owner, groups, 1000)
    assert result['status'] == 'staged_conditions_revised'
    assert result['refuted_indices'] == [0] and result['survivor_indices'] == [1, 2]
    assert not result['general_guard_supported'] and not result['condition_resynthesis']
    revision = read_method_reference(result['revision_ref'], agent.workspace)
    assert result['original_development_sources'] == 3 and result['new_paid_sources'] == 2
    evidence = [item for item in revision['paid_stage_evidence']
                if not item.get('original_development_support')]
    assert [item['preference']['preference'] for item in evidence] == ['baseline', 'challenger']
    assert [item['profile']['residual']['loss'] for item in evidence] == [0., .05]
    assert all(item['incremental_costs']['keep_prefix']['compute_cost'] == 0
        and item['incremental_costs']['append_continuation']['compute_cost'] == 100 for item in evidence)
    revised = revision['research_program']['staged_research']
    source = read_method_reference(proposal['proposal_ref'], agent.workspace)
    assert revised['conditions'] == [source['proposal']['candidates'][i]['research_program'][
        'staged_research']['condition'] for i in (1, 2)]
    assert revised['baseline'] == body['staged_research']['baseline']
    assert revised['continuation'] == body['staged_research']['continuation']
    assert Path(proposal['proposal_ref']['file']).read_bytes() == frozen
    assert result['new_fits'] == result['new_predictions'] == result['new_observations'] == 0
    paid = agent.costs['compute_cost']
    reused = revise_staged_method_conditions(agent, owner, groups, 0)
    assert reused['reused'] and reused['new_compute_cost'] == 0
    assert agent.costs['compute_cost'] == paid


def test_incomplete_prefix_is_not_a_refutation_or_successful_condition(tmp_path):
    agent, owner, groups, _, _ = setup(tmp_path, complete=False)
    result = revise_staged_method_conditions(agent, owner, groups, 1000)
    revision = read_method_reference(result['revision_ref'], agent.workspace)
    assert result['refuted_indices'] == []
    assert result['new_paid_sources'] == 1 and result['usable_sources'] == 4
    assert revision['excluded_returns'][0]['reason'] == 'complete_measured_prefix_and_continuation_required'
    assert not result['general_guard_supported']


def test_changed_data_contract_cannot_supply_stage_preference(tmp_path):
    agent, owner, groups, _, _ = setup(tmp_path, mismatched=True)
    result = revise_staged_method_conditions(agent, owner, groups, 1000)
    revision = read_method_reference(result['revision_ref'], agent.workspace)
    assert result['refuted_indices'] == [] and result['new_paid_sources'] == 1
    assert revision['excluded_returns'][0]['preference']['reason'] == 'start_data_or_evaluation_contract_differs'


def test_complete_budget_required_and_common_route_publishes_revision(tmp_path):
    agent, owner, groups, _, _ = setup(tmp_path)
    before = agent.costs['compute_cost']
    denied = revise_staged_method_conditions(agent, owner, groups, 1)
    assert denied['budget_incomplete'] and denied['new_compute_cost'] == 0
    assert agent.costs['compute_cost'] == before
    result = route(agent)['method_action_research']
    assert result['status'] == 'method_condition_revision_pending_validation'
    assert result['reason'] == 'staged_conditions_revised'
    assert result['staged_method_revision']['refuted_indices'] == [0]
    assert 'completed_agenda_basis' not in owner


def paid_practice(tmp_path, agent, revised, *, complete=True, wrong_body=False):
    method = read_method_reference(revised['research_program_ref'], agent.workspace)
    if wrong_body:
        method['staged_research']['conditions'] = []
    sources = [{'source_task_ref': {'id': label}, 'pool_size': 246} for label in ('held', 'changed')]
    save(tmp_path / 'registration.json', {'method_source_ref': revised['research_program_ref'],
        'method_ref': save(tmp_path / 'frozen_method.json', method), 'sources': sources})
    return save(tmp_path / 'receipt.json', {'status': 'paid_stage_paths_completed' if complete
        else 'paid_stage_path_incomplete_not_negative', 'sources': [{
            'source_task_ref': source['source_task_ref'], 'old_pool_size': 246,
            'execution_complete': complete or index == 0, 'full_comparison': complete or index == 0,
            'staged_research_result': {'execution_complete': complete or index == 0,
                'status': 'baseline_supported' if index == 0 else 'continuation_selected',
                'selected_stage': 'baseline' if index == 0 else 'continuation'}}
            for index, source in enumerate(sources)]})


def test_actual_paid_practice_advances_cached_revision_without_research(tmp_path):
    agent, owner, groups, _, _ = setup(tmp_path)
    revised = revise_staged_method_conditions(agent, owner, groups, 1000)
    practice_ref = paid_practice(tmp_path / 'practice', agent, revised)
    returned = record_staged_method_practice(agent, owner,
        revision_ref=revised['revision_ref'], practice_ref=practice_ref, compute_limit=4)
    assert returned['status'] == 'staged_method_practice_completed'
    assert returned['new_compute_cost'] == 4 and returned['execution_complete']
    costs = deepcopy(agent.costs)
    published = revise_staged_method_conditions(agent, owner, groups, 0)
    assert published['status'] == 'staged_conditions_practiced'
    assert published['required_work'] == 'assess_new_source_validation_for_revised_method'
    assert published['new_compute_cost'] == 0 and agent.costs == costs
    assert not published['independent_support'] and not published['general_guard_supported']
    repeated = record_staged_method_practice(agent, owner,
        revision_ref=revised['revision_ref'], practice_ref=practice_ref, compute_limit=0)
    assert repeated['cache_hit'] and repeated['new_compute_cost'] == 0 and agent.costs == costs
    # Ordinary Common research sees the updated practice, not a stale request
    # to execute the same already completed stages once again.
    routed = route(agent)['method_action_research']
    assert routed['staged_method_revision']['status'] == 'staged_conditions_practiced'
    assert routed['required_work'] == published['required_work']


def test_incomplete_paid_practice_remains_unresolved_and_wrong_body_rejected(tmp_path):
    agent, owner, groups, _, _ = setup(tmp_path)
    revised = revise_staged_method_conditions(agent, owner, groups, 1000)
    wrong = paid_practice(tmp_path / 'wrong', agent, revised, wrong_body=True)
    with pytest.raises(ValueError, match='different frozen method'):
        record_staged_method_practice(agent, owner, revision_ref=revised['revision_ref'],
            practice_ref=wrong, compute_limit=100)
    incomplete = paid_practice(tmp_path / 'incomplete', agent, revised, complete=False)
    returned = record_staged_method_practice(agent, owner, revision_ref=revised['revision_ref'],
        practice_ref=incomplete, compute_limit=4)
    assert returned['status'] == 'incomplete_not_negative'
    assert not returned['execution_complete']
    published = revise_staged_method_conditions(agent, owner, groups, 0)
    assert published['status'] == 'staged_condition_practice_incomplete'
    assert 'without_repeating_completed_sources' in published['required_work']
