"""A paid structure design is followed by its actual, newly quoted query."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.finite_native_author import research_action_semantics
from hegel_machine.cognition.research_programs import (
    export_research_selection_program, propose_research_action_revisions,
)


def planning_option():
    proposal = {'action': 'run', 'arguments': {'operator': 'native_condition_research',
        'structure_planning': {'source_round_index': 10, 'evidence_index': 0,
            'candidate_indices': [3, 7], 'observed_record_count': 11,
            'length': 12, 'proposal_compute': 120}, 'max_compute': 120}}
    return {'family': 'N', 'proposal': proposal,
        'semantic_action': research_action_semantics(proposal),
        'proposal_ref': {'file': 'method_decision.json', 'json_path': ['research_options', 0]},
        'complete_quote': {'status': 'available', 'compute_cost': 500,
            'planning_compute': 120, 'following_research_compute': 200,
            'bounded_research_compute': 320, 'comparison_compute': 160, 'finish_compute': 20,
            'observation_cost': 12, 'paid_rows': 157, 'comparison_paid_rows': 169,
            'retained_pool_models': 4991, 'max_exported_models': 29,
            'estimate_is_completion_guarantee': False},
        'followup_complete_quote': {'status': 'available', 'research_unit': {
            'search_compute': 200, 'completion_reserve': {
                'compute_cost': 180, 'retained_pool_models': 4991, 'max_exported_models': 29}}}}


def ordinary():
    return {'action': 'query', 'arguments': {'actions': [
        {'kind': 'trajectory', 'sequence': [{'u': .4}]}]}}


def test_finite_generator_binds_structure_planning_to_its_actual_future_query():
    option = planning_option()
    before = deepcopy(option)
    result = propose_research_action_revisions(ordinary(), [option],
        source_refs=[{'file': 'paid_parent.json', 'json_path': ['failure']}])
    assert len(result['candidates']) == 1
    candidate = result['candidates'][0]
    plan = candidate['research_selection_plan']
    assert plan['first_segment'] == option['semantic_action']
    assert plan['second_segment'] == {'followup_of_first_segment': True,
                                      'purpose': 'conditional_history_structure'}
    assert candidate['proposal_ref'] == option['proposal_ref']
    assert candidate['complete_quote'] == option['complete_quote']
    assert candidate['quote_scope'] == 'bounded_two_segment_capacity_followup_requires_actual_complete_quote'
    assert candidate['followup_complete_quote'] == option['followup_complete_quote']
    assert candidate['followup_quote']['status'] == 'requires_actual_first_return'
    assert candidate['followup_quote']['capacity_is_actual_spend_or_selected_query'] is False
    assert candidate['status'] == 'unmeasured_local_research_intervention'
    assert candidate['source_support_inherited'] is False
    assert option == before
    # Compilation stores the binding, not a fabricated pulse/step choice.
    namespace = {}
    exec(export_research_selection_program(plan), namespace)
    assert namespace['_FROZEN_PLAN'] == plan
    assert 'actions' not in plan['second_segment']


def test_regular_native_research_does_not_gain_an_unrelated_dynamic_second_step():
    option = planning_option()
    del option['proposal']['arguments']['structure_planning']
    option['semantic_action'] = research_action_semantics(option['proposal'])
    candidate = propose_research_action_revisions(ordinary(), [option])['candidates'][0]
    assert 'second_segment' not in candidate['research_selection_plan']
    assert 'followup_quote' not in candidate


def test_planning_price_alone_cannot_be_sold_as_the_two_segment_capacity():
    option = planning_option()
    del option['followup_complete_quote']
    option['complete_quote'] = {'status': 'available', 'compute_cost': 120,
                               'observation_cost': 0}
    result = propose_research_action_revisions(ordinary(), [option])
    assert not result['candidates']
    assert result['grouped_trials'][0]['reason'] == 'complete_structure_followup_capacity_required'


@pytest.mark.parametrize('change', ['wrong_purpose', 'no_planning', 'wrong_operator', 'extra_selector'])
def test_compiler_rejects_unbound_or_arbitrary_dynamic_followups(change):
    first = planning_option()['semantic_action']
    second = {'followup_of_first_segment': True, 'purpose': 'conditional_history_structure'}
    if change == 'wrong_purpose':
        second['purpose'] = 'any_experiment'
    elif change == 'no_planning':
        first['arguments'].pop('structure_planning')
    elif change == 'wrong_operator':
        first['arguments']['operator'] = 'theory_transfer'
    else:
        second['actions'] = [{'u': .5}]
    with pytest.raises(ValueError, match='structural followup'):
        export_research_selection_program({'first_segment': first, 'second_segment': second})
