"""Typed contextual induction uses native Program arguments and finite bindings."""
from copy import deepcopy
from itertools import product

import pytest

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.revision_relations import (
    consolidate_revision_relations, instantiate_relation, propose_relation,
    propose_revision,
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
    candidate, = [item for item in result['candidates'] if item['kind'] == 'program_context_relation']
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
    assert result['status'] == 'completed'
    assert not any(item['kind'] == 'program_context_relation' for item in result['candidates'])
    # The explicit before/after records can now also retain a joint proposal;
    # without measured source failure it remains unavailable for application.
    for candidate in result['candidates']:
        assert candidate['kind'] == 'program_revision_relation'
        attempted = propose_revision(candidate, component('unchanged', constant(0.)), [],
            evidence_checks=[{'source_ref': {'file': 'unmeasured.json', 'json_path': []},
                'observation': {'features': {'u': 1.}, 'target': 2., 'history': []}}])
        assert not attempted['proposals']
        assert attempted['reason'] == 'source_measured_discrepancy_required'
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


def edit_sources():
    old = [Program('abs', children=(feature(),)), binary('sub', feature(), constant(3))]
    result = []
    for index, before in enumerate(old):
        after = binary('add', before, constant(index + 1))
        row = {'features': {'u': 2.}, 'target': after.predict(Observation({'u': 2.}, 0.)), 'history': []}
        result.append(revision(str(index), after, before_program=before.to_dict(), counterexamples=[{
            'source_ref': {'file': 'paid_fixture.json', 'json_path': [index]},
            'observation': row, 'tolerance': 0.}]))
    return result


def edit_candidate(sources=None):
    compiled = consolidate_revision_relations(sources or edit_sources())
    assert compiled['status'] == 'completed'
    return next(item for item in compiled['candidates'] if item['kind'] == 'program_revision_relation')


def test_joint_revision_reconstructs_both_sides_and_anchors_the_real_parent():
    sources = edit_sources()
    candidate = edit_candidate(sources)
    assert all(item['before_exact'] and item['after_exact'] for item in candidate['source_reconstruction'])
    assert candidate['parent_holes'] == ['hole_0']
    assert candidate['before_pattern']['op'] == 'arg'
    assert Definition.from_dict(candidate['definition']).body.children[0].name == 'hole_0'
    parent = component('current_parent', binary('max', feature(), constant(0)))
    pool = [component('one', constant(1)), component('two', constant(2)), component('old', feature())]
    checks = [{'source_ref': {'file': 'paid_target.json', 'json_path': [0]},
               'observation': {'features': {'u': 2.}, 'target': 3., 'history': []}, 'tolerance': 0.}]
    proposed = propose_revision(candidate, parent, pool, evidence_checks=checks)
    assert proposed['status'] == 'completed' and proposed['attempted_bindings'] == len(pool)
    assert proposed['parent_binding']['remaining_holes'] == ['hole_1']
    assert all(item['bindings']['hole_0']['component_id'] == 'current_parent' for item in proposed['proposals'])
    assert all(not item['support_updated'] for item in proposed['proposals'])
    changed = propose_revision(candidate, component('another_parent', Program('abs', children=(feature(),))),
                               pool, evidence_checks=checks)
    first = Program.from_dict(proposed['proposals'][0]['program'])
    second = Program.from_dict(changed['proposals'][0]['program'])
    assert first.predict(Observation({'u': -2.}, 0.)) != second.predict(Observation({'u': -2.}, 0.))
    assert all('program' not in item and 'bindings' not in item for item in candidate['source_bindings'])
    known = deepcopy(checks)
    known[0]['observation']['target'] = 2.
    assert propose_revision(candidate, parent, pool, evidence_checks=known)['status'] == 'not_applicable'
    altered = deepcopy(sources)
    for source in altered:
        row = source['counterexamples'][0]['observation']
        row['target'] = Program.from_dict(source['before_program']).predict(Observation(row['features'], 0.))
    unsupported = propose_revision(edit_candidate(altered), parent, pool, evidence_checks=checks)
    assert unsupported['status'] == 'UNKNOWN' and unsupported['reason'] == 'source_measured_discrepancy_required'


def test_timeless_parent_is_compatible_with_a_clocked_source_without_erasing_real_clocks():
    before = [constant(1.), binary('ge', feature(), constant(0.))]
    records = []
    for index, old in enumerate(before):
        after = binary('add', old, feature())
        records.append(revision(str(index), after, before_program=old.to_dict(), counterexamples=[{
            'source_ref': {'file': 'paid_fixture.json', 'json_path': [index]},
            'observation': {'features': {'u': 2.}, 'target': 3., 'history': []}}]))
    candidate = edit_candidate(records)
    assert all(item['before_exact'] and item['after_exact'] for item in candidate['source_reconstruction'])
    assert candidate['parent_holes']
    conflicting = deepcopy(records)
    # Both source expressions now depend on genuinely different declared clocks.
    conflicting[0]['before_program'] = feature().to_dict()
    conflicting[0]['role_signature']['u']['time'] = 'other_clock'
    result = consolidate_revision_relations(conflicting)
    assert not any(item['kind'] == 'program_revision_relation' for item in result['candidates'])
    assert any(item['reason'] == 'incompatible_subtree_unit_or_time' for item in result['excluded'])


def test_fixed_and_relational_history_explanations_stay_competing_and_empirical():
    condition = Program('available', lag=1)
    lag = Program('lag', name='u', lag=1)
    sources = [revision(str(index), Program('if', children=(condition, before, startup)),
        before_program=before.to_dict(), counterexamples=[{
            'source_ref': {'file': 'paid_startup.json', 'json_path': [index]},
            'observation': {'features': {'u': 2.}, 'target': 2., 'history': []}}])
        for index, (before, startup) in enumerate([(lag, feature()),
            (binary('mul', constant(2), lag), Program('abs', children=(feature(),)))])]
    candidate = edit_candidate(sources)
    assert len(candidate['constraint_hypotheses']) >= 2
    assert all(item['status'] == 'empirical_candidate' for item in candidate['constraint_hypotheses'])
    parent = component('parent_two', Program('lag', name='u', lag=2))
    checks = [{'source_ref': {'file': 'target_startup.json', 'json_path': [0]},
        'observation': {'features': {'u': 2.}, 'target': 2., 'history': []}}]
    proposed = propose_revision(candidate, parent, [component('current', feature())], evidence_checks=checks)
    assert proposed['status'] == 'completed' and proposed['proposals']
    assert all(Program.from_dict(item['program']).children[0].lag == 2 for item in proposed['proposals'])
    assert all(item['constraint_hypothesis'] != 'observed_fixed' for item in proposed['proposals'])
    assert not candidate['global_support'] and not candidate['independent_confirmation']


def test_bounded_component_search_round_robins_and_resumes_a_small_cursor():
    candidate = relation()
    lag = Program('lag', name='u', lag=1)
    components = [component('mature0', lag), component('mature1', binary('mul', constant(2), lag)),
        component('mature2', binary('add', lag, feature())), component('startup0', feature()),
        component('startup1', constant(0)), component('startup2', Program('abs', children=(feature(),)))]
    first = propose_relation(candidate, components, allocation={'compute': 100})
    assert first['status'] == 'UNKNOWN' and first['budget_exhausted']
    assert len({item['bindings']['hole_0']['component_id'] for item in first['proposals']}) > 1
    resumed = propose_relation(candidate, components, allocation={'compute': 100}, cursor=first['cursor'])
    assert resumed['cursor'] != first['cursor']
    assert first['actual_compute_cost'] <= 100 and resumed['actual_compute_cost'] <= 100
    assert {str(item['program']) for item in first['proposals']}.isdisjoint(
        {str(item['program']) for item in resumed['proposals']})
