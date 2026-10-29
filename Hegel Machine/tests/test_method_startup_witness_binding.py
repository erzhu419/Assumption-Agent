"""A revised method must carry its paid parent witnesses into local allocation."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.finite_native_author import _bind_paid_startup_witnesses
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.research_programs import export_research_selection_program
from test_native_representation_method_selection import prepared, current_memory


def witness(models, *, mature=0, startup=2, guard=1, status='observed'):
    return {'models': models, 'execution_ref': {'file': '/paid/assessment.json', 'json_path': [models[0]]},
        'transition': {'status': status, 'guard_depth': guard,
            'mature_first_divergences': mature, 'startup_first_divergences': startup,
            'source_refs': [{'file': '/paid/assessment.json', 'json_path': ['closed_prediction_state']}]}}


def selection_plan():
    return {'condition': Program('feature', name='whole').to_dict(),
        'when_true': {'first_segment': {'current_option': {
            'id': 'native_representation:whole_state', 'family': 'N'}}},
        'when_false': {'first_segment': {'current_option': {
            'id': 'native_representation:ordinary', 'family': 'N'},
            'parent_binding': 'paid_startup_witnesses'}}}


def test_binding_keeps_complete_pool_cap_and_one_alias_per_paid_execution():
    action = {'action': 'run', 'arguments': {'native_representation_plan': {
        'parent_theory_ids': ['unhelpful', 'matched', 'matched_alias', 'second'],
        'max_candidates': 4, 'compute_limit': 30000, 'validation_records': [28]}}}
    before = deepcopy(action)
    evidence = [witness(['missing.py']), witness(['unhelpful.py'], mature=3),
        witness(['matched.py', 'matched_alias.py']), witness(['second.py']),
        witness(['unhelpful.py'], mature=None), witness(['unhelpful.py'], guard=None)]
    result = _bind_paid_startup_witnesses(action, {'method_candidate_pool': {
        'status': 'observed', 'witnesses': evidence}})
    plan = result['arguments']['native_representation_plan']
    assert plan['startup_parent_priority'] == ['matched', 'second']
    assert [entry['execution_ref'] for entry in plan['startup_parent_witness_refs']] == [
        evidence[2]['execution_ref'], evidence[3]['execution_ref']]
    assert all(plan[key] == value for key, value in before['arguments']['native_representation_plan'].items())


@pytest.mark.parametrize('condition,supported,expected_binding', [
    (0, True, True), (1, True, False), (0, False, False)])
def test_compiled_method_only_binds_its_supported_false_branch(
        tmp_path, monkeypatch, condition, supported, expected_binding):
    agent, lab, _, _ = prepared(tmp_path, monkeypatch)
    memory = current_memory(agent, lab)
    memory.update(ordinary_method_path=str(agent.workspace / 'ordinary.py'),
        method_condition_features={'whole': condition},
        method_condition_support={'can_apply': supported},
        method_candidate_pool={'status': 'observed', 'witnesses': [witness(['old.py'])]})
    before_costs, before_observations = dict(agent.costs), deepcopy(agent.observations)
    compiled = {}
    exec(export_research_selection_program(selection_plan()), compiled)
    chosen = compiled['select_actions'](lab.public_spec(), agent.observations, memory)
    native = chosen['arguments']['native_representation_plan']
    assert ('startup_parent_priority' in native) is expected_binding
    if expected_binding:
        assert native['startup_parent_priority'] == ['old']
        assert native['parent_theory_ids'] == ['old', 'constant']
        assert native['max_candidates'] == 4
    if condition and supported:
        assert native['search_directions'] == ['discrete_state']
    original_plan = selection_plan()
    original_plan['when_false'] = {}
    old_compiled = {}
    exec(export_research_selection_program(original_plan), old_compiled)
    old_choice = old_compiled['select_actions'](lab.public_spec(), agent.observations, memory)
    assert 'startup_parent_priority' not in old_choice['arguments']['native_representation_plan']
    if not expected_binding:
        assert chosen == old_choice
    assert agent.costs == before_costs
    assert agent.observations == before_observations


def test_missing_or_unknown_paid_witnesses_do_not_invent_parent_support():
    original = {'arguments': {'native_representation_plan': {
        'parent_theory_ids': ['old'], 'max_candidates': 4}}}
    for pool in ({}, {'status': 'unknown', 'witnesses': [witness(['old.py'])]},
                 {'status': 'observed', 'witnesses': [witness(['old.py'], guard=None)]}):
        assert _bind_paid_startup_witnesses(deepcopy(original), {'method_candidate_pool': pool}) == original


def test_witness_binding_cannot_silently_attach_to_whole_state_option():
    plan = selection_plan()
    plan['when_false']['first_segment']['current_option']['id'] = 'native_representation:whole_state'
    with pytest.raises(ValueError, match='ordinary local'):
        export_research_selection_program(plan)
