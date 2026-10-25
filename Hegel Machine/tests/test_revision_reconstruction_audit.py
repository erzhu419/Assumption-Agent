"""Pure constructor/contract controls; no observation or world evaluation."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.language import Definition, Program
from scripts.audit_revision_reconstruction import audit_reconstruction


def arg(name):
    return Program('arg', name=name)


def add(left, right):
    return Program('add', children=(left, right))


def candidate(identifier, before, after, names=('parent', 'new', 'shared')):
    constraints = [{'op': 'fixed', 'left': {'object': 'program',
        'property': 'minimum_history_depth'}, 'value': 0}]
    constraints.extend({'op': 'equal', 'left': {'object': 'hole', 'name': name,
        'property': 'minimum_history_depth'}, 'right': {'object': 'program',
        'property': 'minimum_history_depth'}} for name in names)
    return {'id': identifier, 'version': 1, 'kind': 'program_revision_relation',
        'before_pattern': before.to_dict(),
        'definition': Definition(identifier, tuple(names), after).to_dict(),
        'holes': [{'name': name, 'value_type': 'scalar', 'output_unit': 'U',
                   'time_contract': ['current']} for name in names],
        'parent_holes': ['parent', 'shared'], 'learned_constraints': [],
        'constraint_hypotheses': [{'id': 'observed_fixed', 'constraints': constraints,
                                  'assignments': []}],
        'trigger_kind': 'measured_parent_discrepancy',
        'trigger_checks': [{'before': {'failed': True}, 'after': {'failed': True}}],
        'scope': {'context_family_ref': 'fixture_family'}, 'output_unit': 'U',
        'fixed_role_signature': {}, 'constraint_scope': 'optional_proposer_only',
        'source_ids': ['source_a', 'source_b'],
        'source_reconstruction': [{'before_exact': True, 'after_exact': True}],
        'counterexample_refs': [{'file': 'saved_failure.json', 'json_path': ['failure']}],
        'active_library': False, 'global_support': False}


@pytest.fixture
def entries():
    before = add(arg('parent'), arg('shared'))
    old = candidate('old', before, add(arg('new'), arg('shared')))
    new = candidate('new', before, add(add(arg('new'), arg('shared')), arg('shared')))
    new['purpose'] = 'structural_revision'
    new['parameter_relations'] = []
    return ({'candidate': old, 'reference': {'file': 'old.json', 'json_path': ['candidate']}},
            {'candidate': new, 'reference': {'file': 'new.json', 'json_path': ['candidate']}})


def run(entries, compute=25000):
    old, target = entries
    return audit_reconstruction([old], [target], {'compute': compute},
                                material_refs={'old_archive': {'file': 'complete_archive.json'}})


def test_old_nested_definition_reconstructs_macro_and_both_contract_directions(entries, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('representation audit must not evaluate world predictions')

    monkeypatch.setattr(Program, 'predict', forbidden)
    original = deepcopy(entries)
    receipt = run(entries)
    assert receipt['status'] == 'completed'
    assert 0 < receipt['actual_new_compute'] <= 25000
    proof = receipt['targets'][0]['certificate']
    assert proof['before_exact'] and proof['after_exact']
    compound = proof['composition_dag']['new']
    assert compound['kind'] == 'old_definition_call'
    assert compound['constructor_ref'] == entries[0]['reference']
    assert compound['exact_ordered_expansion']
    assert compound['unused_arguments'] == ['parent']
    assert proof['typed_domain_equivalence']['all_target_arguments_constrained_by_original_call_arguments']
    domain = proof['old_constraint_domain_entailed'][0]
    assert domain['target_domain_implies_composed_old_domain']
    assert domain['composed_old_domain_implies_target_domain']
    assert domain['unbound_arg_default_depth_used'] is False
    assert proof['classification'] == 'derived_reconstructible_with_old_closure'
    assert proof['objects_or_classes_merged'] is False
    assert proof['whole_original_family_equivalent_to_target'] is False
    assert proof['original_evidence_refs']['counterexample_refs']['file'] == 'old.json'
    assert receipt['new_predictions'] == receipt['new_fits'] == receipt['new_observations'] == 0
    assert entries == original


def test_shared_before_after_binding_mismatch_is_not_a_reconstruction(entries):
    target = entries[1]['candidate']
    target['definition']['body'] = add(arg('new'), arg('parent')).to_dict()
    receipt = run(entries)
    assert receipt['status'] == 'UNKNOWN'
    assert any(item.get('reason') == 'parent_repeated_subtree_mismatch'
               for item in receipt['targets'][0]['attempts'])


@pytest.mark.parametrize('changed_field,value', [('output_unit', 'V'),
                                               ('time_contract', ['previous'])])
def test_unit_or_time_conflict_cannot_be_imported_as_a_target_filter(entries, changed_field, value):
    entries[1]['candidate']['holes'][1][changed_field] = value
    receipt = run(entries)
    assert receipt['status'] == 'UNKNOWN'
    assert not any(item.get('status') == 'certified' for item in receipt['targets'][0]['attempts'])


def test_missing_declared_arg_depth_does_not_use_arg_default_zero(entries):
    constraints = entries[1]['candidate']['constraint_hypotheses'][0]['constraints']
    constraints[:] = [item for item in constraints if item['left'].get('name') != 'new']
    receipt = run(entries)
    assert receipt['status'] == 'UNKNOWN'
    assert any(item.get('reason') == 'unconstrained_argument_depth'
               for item in receipt['targets'][0]['attempts'])


def test_unsupported_depth_rule_is_unknown_instead_of_general_equivalence(entries):
    lagged = Program('lag', name='u', lag=1)
    entries[1]['candidate']['definition']['body'] = add(lagged, arg('shared')).to_dict()
    receipt = run(entries)
    assert receipt['status'] == 'UNKNOWN'


def test_bounded_audit_preserves_unknown_and_actual_work(entries):
    receipt = run(entries, compute=5)
    assert receipt['status'] == 'UNKNOWN'
    assert receipt['budget_exhausted']
    assert receipt['actual_new_compute'] == 5
    assert receipt['last_denied_request']['available_compute'] == 0
    assert receipt['targets'][0]['reason'] == 'bounded_reconstruction_budget_exhausted'


def test_direct_frozen_availability_is_positive_control_without_merging(entries):
    old, target = entries
    receipt = audit_reconstruction([old, target], [target], {'compute': 25000},
                                    old_constructors=[old])
    assert receipt['targets'][0]['status'] == 'certified_available_contract'
    assert receipt['targets'][0]['certificate']['original_ref'] == target['reference']
    assert receipt['targets'][0]['certificate']['world_equivalence_or_support_added'] is False


def test_symbolic_if_condition_cannot_assume_the_current_max_depth_rule():
    # The scalar condition hole can later bind Program('available', lag=1).
    # Native if then uses its guarded-startup rule, not a maximum of depths.
    names = ('condition', 'parent', 'new', 'shared')
    conditional = lambda mature: Program('if', children=(arg('condition'), mature, arg('shared')))
    before = conditional(arg('parent'))
    old = candidate('old_if', before, conditional(arg('new')), names)
    target = candidate('new_if', before, conditional(conditional(arg('new'))), names)
    for value in (old, target):
        value['parent_holes'] = ['condition', 'parent', 'shared']
        value['holes'][0]['output_unit'] = 'dimensionless'
    entries = ({'candidate': old, 'reference': {'file': 'old_if.json', 'json_path': []}},
               {'candidate': target, 'reference': {'file': 'new_if.json', 'json_path': []}})
    receipt = run(entries)
    assert receipt['status'] == 'UNKNOWN'
    assert any(item.get('reason') == 'symbolic_if_condition_history_rule_unknown'
               for item in receipt['targets'][0]['attempts'])


def test_top_level_learned_constraint_is_not_silently_ignored(entries):
    entries[0]['candidate']['learned_constraints'] = [{'op': 'fixed', 'left': {
        'object': 'hole', 'name': 'new', 'property': 'minimum_history_depth'}, 'value': 1}]
    receipt = run(entries)
    assert receipt['status'] == 'UNKNOWN'
    assert any(item.get('reason') == 'old_constraint_domain_not_entailed'
               for item in receipt['targets'][0]['attempts'])


def compound_if_entries():
    """Same typed multiplication/conditional/zero shape as the saved failure."""
    u = Program('feature', name='u')
    guard = Program('ge', children=(u, arg('threshold')))
    branch = Program('if', children=(guard, u, Program('const', value=0.)))
    term = lambda coefficient: Program('mul', children=(arg(coefficient), branch))
    before = add(arg('parent_offset'), term('parent_coefficient'))
    after = add(arg('new_offset'), term('new_coefficient'))
    old_names = ('parent_offset', 'parent_coefficient', 'threshold', 'new_offset', 'new_coefficient')
    old = candidate('old_conditional', before, after, old_names)
    target = candidate('nested_conditional', before, add(after, term('outer_coefficient')),
                       (*old_names, 'outer_coefficient'))
    for value in (old, target):
        value['parent_holes'] = ['parent_offset', 'parent_coefficient', 'threshold']
        value['output_unit'] = '1'
        value['fixed_role_signature'] = {'u': {'role': 'public_control', 'unit': '1', 'time': 'current'}}
        for hole in value['holes']:
            hole['output_unit'] = 'dimensionless' if 'coefficient' in hole['name'] else '1'
    return ({'candidate': old, 'reference': {'file': 'old_conditional.json', 'json_path': []}},
            {'candidate': target, 'reference': {'file': 'nested_conditional.json', 'json_path': []}})


def test_original_hole_and_arg_contracts_type_zero_through_compound_factor():
    receipt = run(compound_if_entries())
    assert receipt['status'] == 'completed'
    proof = receipt['targets'][0]['certificate']
    assert proof['typed_bindings']['new_offset']['output_unit'] == '1'
    assert proof['composition_dag']['new_offset']['exact_ordered_expansion']


def test_literal_one_unit_is_not_an_alias_for_dimensionless():
    entries = compound_if_entries()
    threshold = next(item for item in entries[1]['candidate']['holes'] if item['name'] == 'threshold')
    threshold['output_unit'] = 'dimensionless'
    receipt = run(entries)
    assert receipt['status'] == 'UNKNOWN'
    assert not any(item.get('status') == 'certified' for item in receipt['targets'][0]['attempts'])
