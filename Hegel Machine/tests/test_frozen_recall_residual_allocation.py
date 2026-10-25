"""Unused first-pass slices fund deferred work inside the unchanged allowance."""
from copy import deepcopy

import pytest

from hegel_machine.cognition import theory_consolidation as consolidation
from test_retained_recall_budget_preparation import retained_scene
from test_theory_consolidation import assert_no_new_science


def set_minimum(agent, question, args, identifier, minimum):
    retained = args['_retained_execution_inputs'][identifier]
    target = question['theory_target']
    inputs = {'candidate_id': identifier,
        'theory_id': f'frozen_recall:{question["id"]}@1:{identifier}',
        'role_signature': deepcopy(target['role_signature']), 'output_unit': target['output_unit'],
        'context_ref': deepcopy(target['context_ref']), 'evidence_checks': [], **retained,
        'parent_theory_id': target['parent_theory_id'], 'fitting': args['fitting'], 'allocation': {}}
    basis = consolidation._relation_request_basis(retained['_retained_parent'],
        retained['_retained_components'], inputs,
        agent.theory_consolidation['candidates'][identifier], consolidation.LIBRARY_PATH)
    key = f'relation_transfer:{inputs["theory_id"]}'
    agent.theory_consolidation.setdefault('executions', {})[key] = {
        'request_basis': basis, 'cursor': {'layer': 4, 'offset': 2},
        'search_completed': False, 'result': {'minimum_compute_for_progress': minimum}}
    consolidation._save(agent)
    return key


@pytest.mark.parametrize('identifiers', [('old19', 'peer', 'new20'), ('new20', 'peer', 'old19')])
def test_residual_funds_known_minimum_without_repeating_dispatched_objects(
        tmp_path, monkeypatch, identifiers):
    agent, lab, question, args = retained_scene(tmp_path, identifiers)
    target = identifiers[0]
    key = set_minimum(agent, question, args, target, 15)
    saved = deepcopy(agent.theory_consolidation['executions'][key])
    args['allocation']['compute'] = 40
    calls = []

    def transfer(agent, inputs, laboratory=None):
        identifier = inputs['candidate_id']
        calls.append((identifier, inputs['allocation']['compute']))
        spent = 15 if identifier == target else 1
        agent.costs['compute_cost'] += spent
        return {'status': 'completed', 'generated_theories': [],
                'new_compute_cost': spent, 'search_completed': True}

    monkeypatch.setattr(consolidation, 'transfer_theory', transfer)
    result = consolidation.recall_frozen_theories(agent, question, args)
    assert [identifier for identifier, _ in calls] == [*identifiers[1:], target]
    assert calls[-1][1] >= 15
    assert result['initial_unfunded_objects'] == [{'candidate_id': target,
        'required_compute': 20, 'allocated_compute': 13}]
    assert result['unfunded_objects'] == [] and result['pending_object_ids'] == []
    assert result['search_completed'] and result['status'] == 'completed'
    assert result['new_compute_cost'] == 33 <= 40
    assert result['residual_reoffers'][0]['status'] == 'transfer_called'
    assert result['object_cursor']['next_candidate_id'] == identifiers[1]
    # Shared scheduling does not modify an object's numerical continuation.
    assert agent.theory_consolidation['executions'][key] == saved
    assert_no_new_science(agent, lab)


def test_preparation_shortfall_reclaims_residual_but_mismatches_are_not_retried(
        tmp_path, monkeypatch):
    agent, lab, question, args = retained_scene(tmp_path, ('target', 'mismatch_a', 'mismatch_b'))
    for identifier in ('mismatch_a', 'mismatch_b'):
        agent.theory_consolidation['candidates'][identifier]['output_unit'] = 'A'
    consolidation._save(agent)
    args['allocation']['compute'] = 12
    calls = []

    def transfer(agent, inputs, laboratory=None):
        calls.append(inputs['candidate_id'])
        agent.costs['compute_cost'] += 1
        return {'status': 'completed', 'generated_theories': [],
                'new_compute_cost': 1, 'search_completed': True}

    monkeypatch.setattr(consolidation, 'transfer_theory', transfer)
    result = consolidation.recall_frozen_theories(agent, question, args)
    assert calls == ['target']
    assert result['initial_unfunded_objects'] == [{'candidate_id': 'target',
        'required_compute': 5, 'allocated_compute': 4}]
    assert result['new_compute_cost'] == 10
    assert result['search_completed'] and not result['pending_object_ids']
    assert len(result['excluded']) == 2
    assert_no_new_science(agent, lab)


def test_comparison_reserve_cannot_fund_a_deferred_object(tmp_path, monkeypatch):
    agent, lab, question, args = retained_scene(tmp_path, ('target', 'peer', 'mismatch'))
    set_minimum(agent, question, args, 'target', 15)
    agent.theory_consolidation['candidates']['mismatch']['output_unit'] = 'A'
    consolidation._save(agent)
    args['allocation']['compute'] = 40
    calls = []

    def transfer(agent, inputs, laboratory=None):
        calls.append(inputs['candidate_id'])
        agent.costs['compute_cost'] += 1
        return {'status': 'completed', 'generated_theories': [],
                'new_compute_cost': 1, 'reserved_comparison_compute': 15,
                'search_completed': True}

    monkeypatch.setattr(consolidation, 'transfer_theory', transfer)
    result = consolidation.recall_frozen_theories(agent, question, args)
    assert calls == ['peer'] and result['reserved_comparison_compute'] == 15
    assert result['new_compute_cost'] == 8
    assert result['residual_reoffers'] == [{'candidate_id': 'target',
        'status': 'minimum_still_unfunded', 'required_compute': 20,
        'available_compute': 17, 'new_compute_cost': 0}]
    assert result['pending_object_ids'] == ['target'] and not result['search_completed']
    assert_no_new_science(agent, lab)


def test_dispatched_incomplete_work_does_not_receive_a_second_slice(tmp_path, monkeypatch):
    agent, lab, question, args = retained_scene(tmp_path)
    calls = []

    def transfer(agent, inputs, laboratory=None):
        calls.append(inputs['candidate_id'])
        agent.costs['compute_cost'] += 1
        return {'status': 'UNKNOWN', 'generated_theories': [],
                'new_compute_cost': 1, 'search_completed': False}

    monkeypatch.setattr(consolidation, 'transfer_theory', transfer)
    result = consolidation.recall_frozen_theories(agent, question, args)
    assert calls == ['old_object'] and result['new_compute_cost'] == 6
    assert result['initial_unfunded_objects'] == [] and result['residual_reoffers'] == []
    assert result['pending_object_ids'] == ['old_object'] and not result['search_completed']
    assert_no_new_science(agent, lab)
