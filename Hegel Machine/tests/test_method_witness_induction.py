"""Method content follows recorded component use, not labels or future answers."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.research_programs import (
    derive_paid_witness_binding_revision, propose_research_action_revisions,
    export_research_selection_program,
)


def state(mature, startup):
    u = Program('feature', name='u')
    threshold = lambda value: Program('ge', children=(u, Program('const', value=value)))
    return Program('state_replay', children=(Program('if', children=(
        Program('available', lag=1), threshold(mature), threshold(startup))),)).to_dict()


def old_plan():
    return {'ordinary_method_path': 'ordinary.py',
        'condition': Program('feature', name='whole').to_dict(),
        'condition_feature_paths': {'whole': ['research', 'candidate_pool', 'min_mature_first_divergences']},
        'when_true': {'first_segment': {'current_option': {
            'id': 'native_representation:whole_state', 'family': 'N'}}},
        'when_false': {}, 'source_refs': [{'file': 'old_method.json'}]}


def experience(name, missed=False):
    ref = {'file': name + '/request.json', 'json_path': []}
    local = {'id': 'local', 'program': state(.8 if missed else .2, .6),
        'search_direction': 'startup_revision', 'training_mse': .1 if missed else 0.,
        'validation_mse': .1 if missed else 0., 'development_mse': .1 if missed else 0.}
    whole = {'id': 'whole', 'program': state(.2, .6), 'search_direction': 'discrete_state',
        'training_mse': 0., 'validation_mse': 0., 'development_mse': 0.}
    selected = {key: value for key, value in (whole if missed else local).items()
                if key != 'program'}
    selected['program_ref'] = {'json_path': ['candidates', 1 if missed else 0, 'program']}
    return {'source_task_ref': ref,
        'source_refs': [{'file': name + '/native.json'}, {'file': name + '/profile.json'}],
        'profile': {'research': {'candidate_pool': {'status': 'observed', 'witnesses': [
            {'models': ['old.py'], 'execution_ref': {'file': name + '/paid.json'},
             'transition': {'status': 'observed', 'guard_depth': 1,
                 'startup_first_divergences': 3, 'mature_first_divergences': 0}}]}}},
        'parent_programs': [{'id': 'old', 'program': state(.2, -.2)}],
        'native_result': {'comparison_completed': True,
            'research_completion': 'bounded_search_completed', 'parent_theory_ids': ['old'],
            'mse_threshold': 1e-10, 'candidates': [local, whole] if missed else [local],
            'selected': selected,
            'search_directions': [{'direction': 'startup_revision'}] +
                ([{'direction': 'discrete_state'}] if missed else [])}}


@pytest.mark.parametrize('layout', ['selected_program_ref', 'inline_program'])
def test_real_opposite_component_returns_generate_only_the_binding_revision(layout):
    inputs, previous = [experience('success'), experience('missed', True)], old_plan()
    if layout == 'inline_program':
        for source in inputs:
            native = source['native_result']
            selected = native['selected']
            selected['program'] = native['candidates'][selected.pop('program_ref')['json_path'][1]]['program']
    before = deepcopy((inputs, previous))
    result = derive_paid_witness_binding_revision(inputs, previous, compute_limit=100)
    candidate = result['candidate']
    assert result['status'] == 'proposed_from_research_relations'
    assert result['analysis_compute_cost'] > 0
    revised = candidate['research_selection_plan']
    assert revised['when_false']['first_segment'] == {'current_option': {
        'id': 'native_representation:ordinary', 'family': 'N'},
        'parent_binding': 'paid_startup_witnesses'}
    assert {key: value for key, value in revised.items() if key != 'when_false'} == {
        key: value for key, value in previous.items() if key != 'when_false'}
    assert candidate['source_support_inherited'] is False
    assert candidate['status'] == 'unmeasured_local_research_intervention'
    assert len(candidate['source_refs']) == 4
    assert (inputs, previous) == before
    compile(export_research_selection_program(revised), 'derived_method.py', 'exec')


@pytest.mark.parametrize('defect', [
    'same_source', 'just_success_labels', 'no_paid_witness', 'wrong_mature_component',
    'startup_not_available_locally', 'already_bound_execution', 'incomplete_comparison',
])
def test_unrelated_or_repeated_success_cannot_become_binding_evidence(defect):
    inputs = [experience('success'), experience('missed', True)]
    missed = inputs[1]
    if defect == 'same_source':
        missed['source_task_ref'] = deepcopy(inputs[0]['source_task_ref'])
    elif defect == 'just_success_labels':
        missed['native_result']['candidates'] = []
        missed['relative_effect'] = 'binding wins'
    elif defect == 'no_paid_witness':
        missed['profile']['research']['candidate_pool']['witnesses'][0]['transition']['mature_first_divergences'] = 1
    elif defect == 'wrong_mature_component':
        missed['native_result']['candidates'][-1]['program'] = state(.7, .6)
    elif defect == 'startup_not_available_locally':
        missed['native_result']['candidates'][0]['program'] = state(.8, .9)
    elif defect == 'already_bound_execution':
        missed['native_result']['startup_parent_binding'] = {'priority': ['old']}
    else:
        missed['native_result']['comparison_completed'] = False
    assert derive_paid_witness_binding_revision(inputs, old_plan(), compute_limit=100)['candidate'] is None


def test_budget_stops_inference_and_bound_prior_is_not_copied_as_discovery():
    inputs = [experience('success'), experience('missed', True)]
    limited = derive_paid_witness_binding_revision(inputs, old_plan(), compute_limit=3)
    assert limited['status'] == 'analysis_budget_unresolved'
    assert limited['analysis_compute_cost'] == 3
    assert limited['candidate'] is None
    bound = old_plan()
    bound['when_false']['first_segment'] = {'current_option': {
        'id': 'native_representation:ordinary', 'family': 'N'},
        'parent_binding': 'paid_startup_witnesses'}
    assert derive_paid_witness_binding_revision(inputs, bound, compute_limit=100)[
        'status'] == 'original_unbound_local_branch_required'


def test_existing_proposer_retains_p0_and_two_method_limit():
    ordinary = {'action': 'run', 'arguments': {'operator': 'research_route',
        'native_representation_plan': {'parent_theory_ids': ['old'], 'max_candidates': 4}}}
    option = {'id': 'native_representation:ordinary', 'family': 'N',
        'proposal': ordinary, 'semantic_action': {'action': 'run', 'arguments': ordinary['arguments']},
        'complete_quote': {'status': 'available', 'max_exported_models': 4},
        'proposal_ref': {'file': 'current.json', 'json_path': ['ordinary']}}
    whole = deepcopy(option)
    whole['id'] = 'native_representation:whole_state'
    whole['semantic_action']['arguments']['native_representation_plan']['search_directions'] = ['discrete_state']
    result = propose_research_action_revisions(ordinary, [option, whole],
        method_experiences=[experience('success'), experience('missed', True)],
        prior_selection_plan=old_plan(), compute_limit=100)
    assert result['baseline']['id'] == 'P0'
    assert len(result['candidates']) == 2
    assert result['candidates'][0]['research_selection_plan']['when_false']['first_segment'][
        'parent_binding'] == 'paid_startup_witnesses'
    assert result['candidates'][0]['complete_quote']['max_exported_models'] == 4
    assert result['analysis_compute_cost'] > 0
