"""Typed contextual induction uses native Program arguments and finite bindings."""
from copy import deepcopy
from itertools import product

import pytest

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.revision_relations import (
    consolidate_revision_relations, instantiate_relation, propose_relation,
)


def feature(name='u'):
    return Program('feature', name=name)


def constant(value):
    return Program('const', value=float(value))


def binary(operator, left, right):
    return Program(operator, children=(left, right))


ROLES = {'u': {'role': 'drive', 'unit': 'V', 'time': 'current'}}
CONTEXT = {'scene': 'target', 'reset': 'per_trajectory'}


def revision(identifier, after, **extra):
    reference = {'file': 'retained.json', 'json_path': ['episodes', identifier, 'after_program']}
    return {'episode_id': identifier, 'source_id': identifier, 'source_ref': reference,
        'before_ref': {'file': 'retained.json', 'json_path': ['episodes', identifier, 'before_program']},
        'after_ref': reference, 'action_ref': {'file': 'retained.json', 'json_path': ['episodes', identifier, 'action']},
        'before_program': constant(0).to_dict(), 'after_program': after.to_dict(),
        'role_signature': deepcopy(ROLES), 'output_unit': 'V',
        'context_ref': {'source': identifier, 'reset': 'per_trajectory'},
        'context_family_ref': 'declared_fixture_family', 'evidence_refs': [], 'cost_refs': [], 'failure_refs': [],
        **extra}


def component(identifier, program, **extra):
    return {'id': identifier, 'program': program.to_dict(), 'version': 1,
        'source_ref': {'file': 'existing_components.json', 'json_path': [identifier]},
        'role_signature': deepcopy(ROLES), 'output_unit': 'V', 'context_ref': deepcopy(CONTEXT), **extra}


def guarded_sources():
    condition = Program('available', lag=1)
    numeric = Program('if', children=(condition, binary('add', Program('lag', name='u', lag=1), feature()), feature()))
    boolean = Program('if', children=(condition, binary('ge', Program('lag', name='u', lag=1), constant(0)),
                                     binary('ge', feature(), constant(0))))
    return [revision('a', numeric), revision('b', boolean)]


def relation(sources=None):
    result = consolidate_revision_relations(sources or guarded_sources(), allocation={'compute': 10000})
    assert result['status'] == 'completed'
    candidate, = result['candidates']
    return candidate


def test_different_numeric_and_boolean_bodies_compile_a_native_context_without_new_scientific_support():
    sources = guarded_sources()
    original = deepcopy(sources)
    candidate = relation(sources)
    definition = Definition.from_dict(candidate['definition'])
    assert definition.body.op == 'if' and definition.body.children[0] == Program('available', lag=1)
    assert [child.op for child in definition.body.children[1:]] == ['arg', 'arg']
    assert candidate['kind'] == 'program_context_relation' and candidate['construction_only'] is True
    assert candidate['constraint_scope'] == 'optional_proposer_only'
    assert candidate['active_library'] is False and candidate['transfer_status'] == 'not_established'
    assert candidate['scientific_support_added'] is False and candidate['independent_confirmation'] is False
    for source, inverse in zip(sources, candidate['source_bindings']):
        full = Program.from_dict(source['after_program'])
        arguments = {}
        for name, reference in inverse['bindings_refs'].items():
            argument = full
            for index in reference['program_path']:
                argument = argument.children[index]
            arguments[name] = argument
            assert reference['source_ref'] == source['after_ref']
        rebuilt = Program('call', name=definition.definition_id, definition=definition,
                          children=tuple(arguments[name] for name in definition.arguments)).expanded()
        assert rebuilt == full
        assert all(item['exact'] for item in candidate['source_reconstruction'])
        assert 'bindings' not in inverse and 'program' not in inverse
    assert original == sources


def test_an_unrelated_add_mul_context_is_learned_from_its_actual_trees():
    left = binary('add', binary('mul', Program('abs', children=(feature(),)), constant(2)), constant(1))
    right = binary('add', binary('mul', binary('sub', feature(), constant(3)), constant(2)), constant(1))
    candidate = relation([revision('a', left), revision('b', right)])
    body = Definition.from_dict(candidate['definition']).body
    assert body == binary('add', binary('mul', Program('arg', name='hole_0'), constant(2)), constant(1))
    result = propose_relation(candidate, [component('old_coordinate', binary('max', feature(), constant(0)))])
    proposal, = result['proposals']
    program = Program.from_dict(proposal['program'])
    assert program.predict(Observation({'u': 4.}, 0.)) == 9.
    assert program.predict(Observation({'u': -4.}, 0.)) == 1.


def test_repeated_typed_subtree_pairs_use_one_lexical_argument():
    left = Program('abs', children=(feature(),))
    right = binary('sub', feature(), constant(3))
    candidate = relation([revision('a', binary('add', left, left)), revision('b', binary('add', right, right))])
    definition = Definition.from_dict(candidate['definition'])
    assert definition.arguments == ('hole_0',)
    assert definition.body.children == (Program('arg', name='hole_0'), Program('arg', name='hole_0'))


def test_program_properties_create_fixed_and_equality_constraints_and_filter_only_this_proposer():
    candidate = relation()
    constraints = candidate['learned_constraints']
    assert any(item['op'] == 'fixed' and item['left']['object'] == 'program' and item['value'] == 0
               for item in constraints)
    assert any(item['op'] == 'equal' and item['left']['object'] == 'pattern' and item['left']['property'] == 'lag'
               for item in constraints)
    components = [component('mature', Program('lag', name='u', lag=1)), component('current', feature())]
    before = deepcopy(components)
    result = propose_relation(candidate, components, allocation={'compute': 1000})
    assert result['status'] == 'completed' and len(result['proposals']) == 1
    assert result['affects_existing_candidates'] is False and components == before
    proposal = result['proposals'][0]
    assert proposal['bindings']['hole_0']['component_id'] == 'mature'
    assert proposal['bindings']['hole_1']['component_id'] == 'current'
    assert all(check['status'] == 'passed' for check in proposal['constraint_checks'])
    program = Program.from_dict(proposal['program'])
    assert program.predict(Observation({'u': 2.}, 0.)) == 2.
    assert program.predict(Observation({'u': 2.}, 0., ({'u': 7.},))) == 7.


def test_fixed_lag_one_is_not_promoted_to_a_lag_two_law_or_global_filter():
    candidate = relation()
    bindings = {'hole_0': component('lag_two', Program('lag', name='u', lag=2)),
                'hole_1': component('current', feature())}
    original = deepcopy(bindings)
    result = instantiate_relation(candidate, bindings, allocation={'compute': 1000})
    assert result['status'] == 'not_applicable' and result['proposals'] == []
    assert result['affects_existing_candidates'] is False and bindings == original
    assert any(check['status'] == 'violated' for check in result['excluded'][0]['constraint_checks'])


def test_same_child_with_different_feature_roles_is_not_silently_fixed_from_the_first_source():
    left = revision('a', binary('add', feature(), Program('abs', children=(feature(),))))
    right = revision('b', binary('add', feature(), binary('sub', feature(), constant(1))))
    right['role_signature']['u']['role'] = 'another_observable'
    candidate = relation([left, right])
    assert candidate['fixed_role_signature'] == {}
    assert Definition.from_dict(candidate['definition']).body.children[0].op == 'arg'


def test_existing_ordered_constant_and_role_parameter_family_is_not_a_new_context_relation():
    left = revision('a', binary('add', constant(.3), binary('mul', constant(1.4), feature())))
    right = revision('b', binary('add', constant(-.1), binary('mul', constant(1.7), feature('v'))),
        role_signature={'v': deepcopy(ROLES['u'])})
    result = consolidate_revision_relations([left, right], allocation={'compute': 1000})
    assert result['status'] == 'completed' and result['candidates'] == []
    assert result['excluded'] == [{'episode_ids': ['a', 'b'], 'status': 'existing_family',
                                  'reason': 'existing_ordered_parameter_family'}]
    assert result['actual_compute_cost'] > 0 and not result['budget_exhausted']


@pytest.mark.parametrize('missing', ['after_ref', 'source_id', 'output_unit', 'context_ref', 'role_signature'])
def test_missing_real_provenance_or_type_contract_is_unknown(missing):
    sources = guarded_sources()
    del sources[0][missing]
    result = consolidate_revision_relations(sources)
    assert result['status'] == 'UNKNOWN' and result['candidates'] == []
    assert result['excluded'] and result['support_updated'] is False


def test_two_notes_or_revisions_from_one_source_are_not_cross_episode_evidence():
    sources = guarded_sources()
    sources[1]['source_id'] = sources[0]['source_id']
    result = consolidate_revision_relations(sources)
    assert result['status'] == 'UNKNOWN' and result['candidates'] == []
    assert result['excluded'][0]['reason'] == 'distinct_episode_sources_required'


def test_unrelated_roots_degenerate_to_a_hole_and_remain_unknown():
    result = consolidate_revision_relations([revision('a', binary('add', feature(), constant(1))),
        revision('b', binary('mul', feature(), constant(2)))])
    assert result['status'] == 'UNKNOWN' and result['candidates'] == []
    assert result['excluded'][0]['reason'] == 'no_nontrivial_shared_construction_context'


@pytest.mark.parametrize('changed', ['unit', 'time', 'context'])
def test_incompatible_source_contracts_do_not_force_a_relation(changed):
    sources = guarded_sources()
    if changed == 'unit':
        sources[1]['output_unit'] = 'A'
    elif changed == 'time':
        sources[1]['role_signature']['u']['time'] = 'another_clock'
    else:
        sources[1]['context_family_ref'] = 'different_family'
    result = consolidate_revision_relations(sources)
    assert result['status'] == 'UNKNOWN' and result['candidates'] == []


def test_failed_numeric_source_results_remain_referenced_without_promoting_world_support():
    sources = guarded_sources()
    sources[0]['failure_refs'] = [{'file': 'retained.json', 'json_path': ['numeric_failure']}]
    sources[0]['source_outcomes'] = {'status': 'failed', 'terminal_loss': .15}
    candidate = relation(sources)
    assert candidate['counterexample_refs'] == sources[0]['failure_refs']
    assert candidate['source_outcomes'][0]['source_outcomes'] == sources[0]['source_outcomes']
    assert candidate['scientific_support_added'] is False and candidate['active_library'] is False
    assert candidate['encoding_cost']['new_definition_bytes'] > 100
    assert candidate['encoding_cost']['formal_mdl'] is False


@pytest.mark.parametrize('changed', ['unit', 'time', 'context', 'missing_contract'])
def test_explicit_component_binding_obeys_units_time_and_current_context(changed):
    bindings = {'hole_0': component('mature', Program('lag', name='u', lag=1)),
                'hole_1': component('current', feature())}
    if changed == 'unit':
        bindings['hole_0']['output_unit'] = 'A'
    elif changed == 'time':
        bindings['hole_0']['role_signature']['u']['time'] = 'another_clock'
    elif changed == 'context':
        bindings['hole_0']['context_ref'] = {'scene': 'another'}
    else:
        del bindings['hole_0']['role_signature']['u']['unit']
    result = instantiate_relation(relation(), bindings)
    assert result['status'] == 'UNKNOWN' and result['proposals'] == []


@pytest.mark.parametrize('allocation', [{'compute': 0}, {'compute': 5}, {'compute': 1000, 'max_groups': 0}])
def test_actual_induction_budget_cutoffs_are_unknown(allocation):
    result = consolidate_revision_relations(guarded_sources(), allocation)
    assert result['status'] == 'UNKNOWN' and result['budget_exhausted']
    assert result['actual_compute_cost'] <= allocation['compute'] and result['candidates'] == []


def test_actual_component_enumeration_budget_is_charged_and_truncates_unknown():
    result = propose_relation(relation(), [component('mature', Program('lag', name='u', lag=1)),
        component('current', feature())], allocation={'compute': 1})
    assert result['status'] == 'UNKNOWN' and result['budget_exhausted']
    assert result['actual_compute_cost'] <= 1 and result['proposals'] == []
    assert result['new_observations'] == result['llm_calls'] == 0


def test_learned_unary_prefilter_preserves_the_small_complete_binding_result():
    candidate = relation()
    mature = component('mature', Program('lag', name='u', lag=1))
    current = component('current', feature())
    fallback = component('constant', constant(1))
    components = [mature, current, fallback]
    original = deepcopy(components)
    complete = []
    for left, right in product(components, repeat=2):
        result = instantiate_relation(candidate, {'hole_0': left, 'hole_1': right})
        complete.extend(Program.from_dict(item['program']) for item in result['proposals'])
    filtered = propose_relation(candidate, components, allocation={'compute': 1000})
    expected = [Program('if', children=(Program('available', lag=1),
                Program('lag', name='u', lag=1), startup)) for startup in (feature(), constant(1))]
    assert complete == expected
    assert [Program.from_dict(item['program']) for item in filtered['proposals']] == expected
    assert components == original and filtered['affects_existing_candidates'] is False
    assert filtered['status'] == 'completed' and filtered['actual_compute_cost'] <= 1000
