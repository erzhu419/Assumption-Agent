"""Saved-instance classification controls, without generating world evidence."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.language import Definition, Program
from scripts.audit_finite_revision_search import classify_saved_proposals


def arg(name):
    return Program('arg', name=name)


def add(left, right):
    return Program('add', children=(left, right))


@pytest.fixture
def saved():
    roles = {name: {'role': name, 'unit': '1', 'time': 'current'} for name in ('u', 'v')}
    parent = {'id': 'actual_parent', 'version': 1,
        'program': add(Program('const', value=1.), Program('feature', name='u')).to_dict(),
        'source_ref': {'file': 'saved.json', 'json_path': ['before']},
        'program_source_ref': {'file': 'saved.json', 'json_path': ['before', 'program'], 'version': 1},
        'role_signature': roles, 'output_unit': '1',
        'context_ref': {'source_id': 'physical_source_a'}, 'context_family_ref': 'declared_family'}
    entry = {**deepcopy(parent), 'id': 'actual_generated',
        'program': add(Program('const', value=3.), Program('feature', name='u')).to_dict(),
        'source_ref': {'file': 'saved.json', 'json_path': ['generated']},
        'program_source_ref': {'file': 'saved.json', 'json_path': ['generated', 'program'], 'version': 1},
        'source_status': 'refuted', 'compatible': False,
        'consolidation_ref': {'parent': {'component_id': parent['id'], 'version': 1,
                                        'source_ref': deepcopy(parent['source_ref'])}}}
    constraints = [{'op': 'fixed', 'left': {'object': 'program',
                                           'property': 'minimum_history_depth'}, 'value': 0}]
    constraints.extend({'op': 'equal', 'left': {'object': 'hole', 'name': name,
        'property': 'minimum_history_depth'}, 'right': {'object': 'program',
                                                      'property': 'minimum_history_depth'}}
                       for name in ('p', 'q', 'shared'))
    target = {'id': 'posthoc_target', 'kind': 'program_revision_relation',
        'before_pattern': add(arg('p'), arg('shared')).to_dict(),
        'definition': Definition('posthoc', ('p', 'q', 'shared'),
                                 add(arg('q'), arg('shared'))).to_dict(),
        'holes': [{'name': name, 'value_type': 'scalar', 'output_unit': '1',
                   'time_contract': ['current']} for name in ('p', 'q', 'shared')],
        'parent_holes': ['p', 'shared'], 'learned_constraints': [],
        'constraint_hypotheses': [{'id': 'observed_fixed', 'constraints': constraints, 'assignments': []}],
        'trigger_checks': [{'before': {'failed': True}}],
        'trigger_kind': 'measured_parent_discrepancy',
        'fixed_role_signature': {'u': roles['u']}, 'output_unit': '1',
        'scope': {'context_family_ref': 'declared_family'}}
    return parent, entry, {'candidate': target, 'reference': {
        'file': 'frozen.json', 'json_path': ['candidates', target['id']]}}


def classify(saved, compute=5000):
    parent, entry, target = saved
    return classify_saved_proposals(parent, [entry], [target], {'compute': compute})


def test_actual_pair_is_typed_checked_without_promoting_refuted_entry(saved, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('posthoc classification cannot evaluate world evidence')

    monkeypatch.setattr(Program, 'predict', forbidden)
    original = deepcopy(saved)
    result = classify(saved)
    assert result['status'] == 'completed'
    record = result['classifications'][0]
    assert record['after_only_match']['status'] == 'matched'
    assert record['declared_generation_parent']['exact']
    pair = record['generation_parent_pair_contract']
    assert pair['status'] == 'matched' and pair['before_exact'] and pair['after_exact']
    assert pair['scope']['measured_source_failure_gate']
    assert record['scientific_status'] == 'refuted' and record['compatible'] is False
    assert not result['scientific_status_changed']
    assert result['new_predictions'] == result['new_fits'] == result['new_observations'] == 0
    assert 0 < result['actual_compute_cost'] <= 5000
    assert saved == original


def test_after_only_match_does_not_recover_a_lost_generation_parent(saved):
    saved[1]['consolidation_ref']['parent']['component_id'] = 'other_actual_parent'
    result = classify(saved)
    record = result['classifications'][0]
    assert record['after_only_match']['status'] == 'matched'
    assert not record['declared_generation_parent']['exact']
    assert record['generation_parent_pair_contract']['status'] == 'UNKNOWN'
    assert result['status'] == 'UNKNOWN'


def test_shared_before_after_subtree_is_not_rebound_during_pair_check(saved):
    saved[1]['program'] = add(Program('const', value=3.), Program('feature', name='v')).to_dict()
    result = classify(saved)
    record = result['classifications'][0]
    assert record['after_only_match']['status'] == 'matched'
    assert record['generation_parent_pair_contract'] == {
        'status': 'not_matched', 'stage': 'shared_after', 'reason': 'parent_repeated_subtree_mismatch'}


def test_explicit_one_and_dimensionless_units_are_not_merged(saved):
    saved[2]['candidate']['holes'][1]['output_unit'] = 'dimensionless'
    record = classify(saved)['classifications'][0]
    assert record['generation_parent_pair_contract'] == {
        'status': 'UNKNOWN', 'reason': 'component_output_unit_mismatch'}


def test_actual_top_level_property_constraint_is_checked(saved):
    saved[2]['candidate']['learned_constraints'] = [{'op': 'fixed', 'left': {
        'object': 'hole', 'name': 'q', 'property': 'minimum_history_depth'}, 'value': 1}]
    pair = classify(saved)['classifications'][0]['generation_parent_pair_contract']
    assert pair['status'] == 'not_matched'
    assert pair['reason'] == 'actual_instance_constraint_violated'
    assert pair['constraint_checks'][0]['checks'][0]['status'] == 'violated'


def test_budget_incomplete_preserves_actual_cost_and_unknown(saved):
    result = classify(saved, compute=5)
    assert result['status'] == 'UNKNOWN' and result['budget_exhausted']
    assert result['actual_compute_cost'] == 5
    assert result['last_denied_request']['available_compute'] == 0
    assert result['classifications'][0]['generation_parent_pair_contract']['reason'] == 'posthoc_budget_exhausted'


def test_first_complete_witness_stops_each_target_without_pruning_saved_pool(saved):
    parent, entry, target = saved
    entries = [entry, *[deepcopy(entry) for _ in range(50)]]
    result = classify_saved_proposals(parent, entries, [target, deepcopy(target)])
    assert result['status'] == 'completed' and result['attempted_count'] == 2
    assert result['unstarted_count'] == 100 and len(entries) == 51
    assert [item['witness_classification_index'] for item in result['targets']] == [0, 1]


def test_historical_before_is_distinct_from_the_declared_generator_parent(saved):
    historical = deepcopy(saved[0])
    historical.update(id='actual_previous_incumbent', program=add(
        Program('const', value=7.), Program('feature', name='u')).to_dict())
    historical['program_source_ref'] = {'file': 'history.json', 'json_path': ['before', 'program']}
    saved[0]['historical_before'] = historical
    result = classify(saved)
    record = result['classifications'][0]
    assert record['declared_generation_parent']['exact']
    assert record['declared_generation_parent']['resolved_program_equal_historical_before'] is False
    assert record['generation_parent_pair_contract']['status'] == 'matched'
    assert record['historical_before_pair_contract']['status'] == 'matched'
    assert record['historical_before_pair_contract']['claim'] == 'actual_saved_instance_not_generic_family_equivalence'


def test_shared_equal_literal_must_have_equal_actual_before_after_units(saved):
    parent, entry, target = saved
    parent['program'] = Program('mul', children=(Program('const', value=1.),
                                                  Program('feature', name='u'))).to_dict()
    entry['program'] = add(Program('const', value=3.), Program('const', value=1.)).to_dict()
    entry['time_contract'] = ['current']
    candidate = target['candidate']
    candidate['before_pattern'] = Program('mul', children=(arg('shared'),
                                                           Program('feature', name='u'))).to_dict()
    candidate['definition'] = Definition('shared_unit', ('q', 'shared'),
                                          add(arg('q'), arg('shared'))).to_dict()
    candidate['holes'] = [hole for hole in candidate['holes'] if hole['name'] != 'p']
    candidate['holes'][1]['output_unit'] = 'dimensionless'
    candidate['parent_holes'] = ['shared']
    constraints = candidate['constraint_hypotheses'][0]['constraints']
    constraints[:] = [item for item in constraints if item['left'].get('name') != 'p']
    record = classify(saved)['classifications'][0]
    assert record['after_only_match']['status'] == 'matched'
    assert record['generation_parent_pair_contract'] == {
        'status': 'UNKNOWN', 'reason': 'actual_shared_subtree_contract_conflict'}


def test_actual_constructor_and_question_refs_restore_missing_parent_metadata(saved, tmp_path):
    parent, entry, target = saved
    u = Program('feature', name='u')
    guard = Program('ge', children=(u, Program('const', value=1.)))
    coefficient = Program('if', children=(guard, Program('const', value=1.), Program('const', value=0.)))
    signal = Program('if', children=(guard, u, Program('const', value=0.)))
    parent['program'] = Program('mul', children=(coefficient, signal)).to_dict()
    entry['program'] = Program('mul', children=(Program('const', value=2.), signal)).to_dict()
    c = target['candidate']
    c['before_pattern'] = Program('mul', children=(arg('p'), arg('shared'))).to_dict()
    c['definition'] = Definition('typed_if', ('p', 'q', 'shared'),
                                  Program('mul', children=(arg('q'), arg('shared')))).to_dict()
    for hole in c['holes'][:2]:
        hole['output_unit'] = 'dimensionless'
    # First, the actual undeclared compound is unknown, not given its target unit.
    parent.pop('context_family_ref')
    source = {'id': 'actual_source_constructor', 'version': 1,
        'definition': Definition('origin', ('coeff', 'signal'), Program('mul', children=(
            arg('coeff'), arg('signal')))).to_dict(),
        'holes': [{'name': name, 'value_type': 'scalar', 'output_unit': unit,
                   'time_contract': ['current']} for name, unit in
                  [('coeff', 'dimensionless'), ('signal', '1')]]}
    binding = {'candidate_id': source['id'], 'candidate_version': 1,
               'components': {'coeff': {'component_id': 'original_coeff'},
                              'signal': {'component_id': 'original_signal'}}}
    path = tmp_path / 'original.json'
    path.write_text(json.dumps({'candidate': source, 'binding': binding,
        'program': parent['program'], 'target': {'context_ref': parent['context_ref'],
                                               'context_family_ref': 'declared_family'}}))
    reference = lambda key: {'file': str(path), 'json_path': [key]}
    parent['contextual_scope_ref'] = reference('target')
    assert classify(saved)['classifications'][0]['generation_parent_pair_contract']['status'] == 'UNKNOWN'
    parent['subtree_contract_source'] = {f'{key}_ref': reference(key)
                                       for key in ('candidate', 'binding', 'program')}
    original = deepcopy(saved)
    result = classify(saved)
    pair = result['classifications'][0]['generation_parent_pair_contract']
    assert result['status'] == 'completed'
    assert pair['scope']['parent_context_family_source_ref'] == reference('target')
    assert pair['binding_refs']['p']['output_unit'] == 'dimensionless'
    assert saved == original
    # An explicit different family is never replaced by the old question's family.
    parent['context_family_ref'] = 'other_explicit_family'
    assert classify(saved)['classifications'][0]['generation_parent_pair_contract']['status'] == 'UNKNOWN'
