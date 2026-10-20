"""Finite matching contracts; no provider, purchases or theory confirmation."""
from copy import deepcopy

import pytest

from hegel_machine.cognition.abstractions import AbstractTemplate
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.theory_matching import (
    match_theories, prepare_theories_for_question, typed_program_match,
)


def feature(name='x'):
    return Program('feature', name=name)


def affine(slope=1., name='x'):
    return Program('mul', children=(Program('const', value=slope), feature(name)))


def theory(identifier='old', program=None):
    return {'id': identifier, 'family_id': 'linear',
            'program': (program or affine()).to_dict(), 'parameter_paths': [[0]],
            'role_signature': {'x': {'role': 'drive', 'unit': 'V', 'time': 'current'}},
            'output_unit': 'V', 'context_ref': {'instrument': 'meter1', 'reset': 'per_trajectory'},
            'declared_scope_ref': 'bounded_drive', 'measured_coverage_ref': 'bounded_drive',
            'premise_refs': [], 'support': {'status': 'empirical_support', 'evidence': [0, 1],
                'context_ref': {'instrument': 'meter1', 'reset': 'per_trajectory'}}}


def pair(program=None):
    source = theory(program=program)
    return source, {**deepcopy(source), 'id': 'target'}


def test_exact_typed_instance_reuses_support_without_granting_confirmation():
    source, target = pair()
    result = typed_program_match(source, target)
    assert result['status'] == 'REUSE_INSTANCE' and result['relation'] == 'exact_compute'
    assert result['node_comparisons'] == result['compute_cost'] == 3
    assert result['mapping']['feature_map'] == {'x': 'x'}
    assert result['confirmation'] == 'not_established'
    assert source['id'] == 'old' and target['id'] == 'target'


def test_only_declared_constants_offer_refitting_and_record_old_values():
    source, target = pair()
    target['program'] = affine(2.).to_dict()
    result = typed_program_match(source, target)
    assert result['status'] == 'REFIT_FAMILY' and result['family_id'] == 'linear'
    assert result['parameter_changes'] == [{'source_path': [0], 'target_path': [0],
        'source_value': 1., 'target_value': 2., 'constraint': 'requires_refit'}]
    source['parameter_paths'] = []
    assert typed_program_match(source, target)['status'] == 'NO_COMPATIBLE_CANDIDATE_FOUND'


@pytest.mark.parametrize('source_program,target_program', [
    (Program('lag', name='x', lag=1), Program('lag', name='x', lag=2)),
    (Program('window_sum', lag=2, children=(feature(),)),
     Program('window_sum', lag=3, children=(feature(),))),
    (Program('if', children=(feature(), Program('const', value=1.), Program('const', value=2.))),
     Program('if', children=(feature(), Program('const', value=2.), Program('const', value=1.)))),
    (Program('sub', children=(feature(), Program('const', value=1.))),
     Program('sub', children=(Program('const', value=1.), feature()))),
])
def test_ordered_branches_history_windows_and_operands_are_not_rearranged(source_program, target_program):
    source, target = pair(source_program)
    source['parameter_paths'] = target['parameter_paths'] = []
    target['program'] = target_program.to_dict()
    assert typed_program_match(source, target)['status'] == 'NO_COMPATIBLE_CANDIDATE_FOUND'


def test_repeated_source_variable_requires_one_consistent_mapping():
    source, target = pair(Program('add', children=(feature(), feature())))
    target['program'] = Program('add', children=(feature('u'), feature('v'))).to_dict()
    assert typed_program_match(source, target)['reason'] == 'inconsistent_variable_mapping'


def test_consistent_variable_renaming_retains_mapping_and_requires_actual_rebinding():
    source, target = pair()
    target['program'] = affine(name='u').to_dict()
    target['role_signature'] = {'u': deepcopy(source['role_signature']['x'])}
    result = typed_program_match(source, target)
    assert result['status'] == 'REBIND_THEN_TEST'
    assert result['mapping']['feature_map'] == {'x': 'u'}
    assert result['mapping_cost'] == 1


def test_role_and_unit_changes_require_explicit_mapping_before_they_are_candidates():
    source, target = pair()
    target['role_signature']['x'].update(role='sensor_reading', unit='mV')
    target['output_unit'] = 'mV'
    assert typed_program_match(source, target)['status'] == 'NO_COMPATIBLE_CANDIDATE_FOUND'
    mapping = {'feature_map': {'x': 'x'}, 'role_map': {'drive': 'sensor_reading'},
               'unit_conversions': {'x': {'from': 'V', 'to': 'mV', 'scale': 1000., 'offset': 0.}},
               'output_conversion': {'from': 'V', 'to': 'mV', 'scale': 1000., 'offset': 0.},
               'status': 'proposed', 'fit_evidence': [2]}
    result = typed_program_match(source, target, allowed_mappings=[mapping])
    assert result['status'] == 'REBIND_THEN_TEST'
    assert result['evidence']['mapping_fit'] == [2] and result['evidence']['mapping_test'] == []
    mapping.update(status='tested', test_evidence=[3], context_ref=deepcopy(target['context_ref']))
    checked = typed_program_match(source, target, allowed_mappings=[mapping])
    assert checked['status'] == 'SCOPED_REUSE'
    assert checked['evidence']['mapping_test'] == [3]
    assert checked['confirmation'] == 'not_established'


def test_multiple_measurement_mappings_remain_alternatives():
    source, target = pair()
    target['role_signature']['x']['unit'] = 'reading'
    mappings = [{'unit_conversions': {'x': {'from': 'V', 'to': 'reading', 'scale': scale, 'offset': 0.}}}
                for scale in (2., 3.)]
    result = typed_program_match(source, target, allowed_mappings=mappings)
    assert result['status'] == 'COMPARE_ALTERNATIVES' and len(result['mappings']) == 2
    assert result['mapping'] == {}


def test_changed_measurement_must_be_rebound_before_refitting_parameters():
    source, target = pair()
    target['program'] = affine(2.).to_dict()
    target['role_signature']['x']['unit'] = 'reading'
    mapping = {'unit_conversions': {'x': {'from': 'V', 'to': 'reading', 'scale': 2., 'offset': 0.}}}
    result = typed_program_match(source, target, allowed_mappings=[mapping])
    assert result['status'] == 'REBIND_THEN_TEST'
    assert result['parameter_changes'][0]['target_value'] == 2.


def test_temporal_role_change_cannot_be_disguised_as_variable_renaming():
    source, target = pair()
    target['role_signature']['x']['time'] = 'next_step'
    assert typed_program_match(source, target)['reason'] == 'time_role_mismatch'


def test_missing_types_are_unresolved_instead_of_guessed_from_names():
    source, target = pair()
    source['role_signature']['x'].pop('unit')
    assert typed_program_match(source, target)['status'] == 'UNKNOWN_MATCH'


def test_measured_scope_does_not_expand_to_declared_or_requested_scope():
    source, target = pair()
    target['declared_scope_ref'] = 'all_drives'
    result = typed_program_match(source, target)
    assert result['status'] == 'SCOPED_REUSE'
    assert result['coverage'] == {'declared': 'bounded_drive', 'measured': 'bounded_drive', 'requested': 'all_drives'}
    source.pop('measured_coverage_ref')
    assert typed_program_match(source, target)['reason'] == 'missing_measured_coverage'


def test_current_premise_versions_are_required_even_when_programs_match():
    source, target = pair()
    premise = {'question_id': 'calibration', 'meaning_version': 1, 'answer_version': 2}
    source['premise_refs'] = [premise]
    target['required_premises'] = [deepcopy(premise)]
    assert typed_program_match(source, target)['reason'] == 'missing_current_premise_support'
    target['premise_statuses'] = {'calibration': {**premise, 'status': 'supported'}}
    assert typed_program_match(source, target)['status'] == 'REUSE_INSTANCE'
    target['premise_statuses']['calibration']['meaning_version'] = 2
    assert typed_program_match(source, target)['status'] == 'UNKNOWN_MATCH'


def test_new_required_premise_cannot_be_skipped_because_old_theory_has_none():
    source, target = pair()
    target['required_premises'] = [{'id': 'new_sensor_condition'}]
    assert typed_program_match(source, target)['status'] == 'UNKNOWN_MATCH'


def test_authored_support_without_actual_evidence_or_context_is_not_instance_reuse():
    source, target = pair()
    source['support']['evidence'] = []
    assert typed_program_match(source, target)['reason'] == 'missing_executed_source_support'
    source['support']['evidence'] = [0]
    source.pop('context_ref')
    target.pop('context_ref')
    assert typed_program_match(source, target)['reason'] == 'missing_context_contract'


def test_actual_refutation_overrides_an_old_positive_support_record():
    source, target = pair()
    source['compatible'] = False
    result = typed_program_match(source, target)
    assert result['status'] == 'PROPOSE_REVISION'
    assert result['reason'] == 'recorded_candidate_incompatible_in_current_scope'


def test_unavailable_retained_source_is_unresolved_rather_than_a_negative_result():
    source, target = pair()
    source.update(compatible=False, source_status='unavailable_retained_version')
    result = typed_program_match(source, target)
    assert result['status'] == 'UNKNOWN_MATCH' and result['reason'] == 'retained_version_unavailable'


def test_budget_counts_actual_node_comparisons_and_never_means_refutation():
    source, target = pair()
    complete = typed_program_match(source, target, compute_budget=3)
    assert complete['status'] == 'REUSE_INSTANCE' and complete['remaining_budget'] == 0
    incomplete = typed_program_match(source, target, compute_budget=2)
    assert incomplete['status'] == 'UNKNOWN_MATCH' and incomplete['reason'] == 'incomplete_match_search'
    assert incomplete['node_comparisons'] == incomplete['compute_cost'] == 2


def test_library_definitions_compare_the_actual_ordered_body_with_bounded_expansion():
    body = Program('mul', children=(Program('const', value=2.), Program('arg', name='a')))
    first = Definition('first', ('a',), body)
    alias = Definition('alias', ('a',), body)
    source, target = pair(Program('call', name='first', children=(feature(),), definition=first))
    target['program'] = Program('call', name='alias', children=(feature(),), definition=alias).to_dict()
    assert typed_program_match(source, target)['status'] == 'REUSE_INSTANCE'
    assert typed_program_match(source, target, max_expansion_depth=0)['reason'] == 'definition_expansion_limit'
    different = Definition('first', ('a',), Program('abs', children=(Program('arg', name='a'),)))
    target['program'] = Program('call', name='first', children=(feature(),), definition=different).to_dict()
    assert typed_program_match(source, target)['status'] == 'NO_COMPATIBLE_CANDIDATE_FOUND'


def test_native_abstract_template_reuses_its_explicit_parameter_paths():
    template = AbstractTemplate('linear', 'source', affine(), ('x',), ('x',), ((0,),))
    source, target = pair()
    source['program'] = template
    source.pop('parameter_paths')
    target['program'] = template.instantiate({'x': 'x'}, [3.]).to_dict()
    assert typed_program_match(source, target)['status'] == 'REFIT_FAMILY'


def test_python_only_uses_exact_saved_source_fit_and_binding():
    source, target = pair()
    for record in (source, target):
        record.pop('program')
        record.update(python_source='def predict(row): return row["x"]\n',
                      fit_state={'slope': 1.}, binding={'x': 'drive'})
    assert typed_program_match(source, target)['status'] == 'REUSE_INSTANCE'
    target['python_source'] = 'def predict(item): return item["x"]\n'
    assert typed_program_match(source, target)['reason'] == 'opaque_python_has_no_exact_instance_match'
    target['python_source'] = source['python_source']
    target['role_signature']['x']['unit'] = 'mV'
    assert typed_program_match(source, target)['status'] == 'REBIND_THEN_TEST'


def signature(values, panel='offered_public_panel'):
    return {'panel_ref': panel, 'readout_ref': 'response', 'scales': [1.] * len(values), 'values': values}


def test_behavior_match_is_only_on_fixed_panel_and_new_probe_can_split_it():
    source, target = pair(Program('abs', children=(feature(),)))
    target['program'] = Program('mul', children=(feature(), feature())).to_dict()
    source['behavior_signature'] = target['behavior_signature'] = signature([1., 1.])
    result = typed_program_match(source, target)
    assert result['status'] == 'SCOPED_REUSE' and result['relation'] == 'scoped_behavior_match'
    source['behavior_signature'] = signature([1., 1., 2.], 'expanded_panel')
    target['behavior_signature'] = signature([1., 1., 4.], 'expanded_panel')
    split = typed_program_match(source, target)
    assert split['status'] == 'COMPARE_ALTERNATIVES'
    assert split['behavior']['normalized_differences'] == [0., 0., 2.]
    assert source['id'] != target['id']


def test_refit_candidate_keeps_fixed_panel_differences_in_its_result():
    source, target = pair()
    target['program'] = affine(2.).to_dict()
    source['behavior_signature'] = signature([1., 2.])
    target['behavior_signature'] = signature([2., 4.])
    result = typed_program_match(source, target)
    assert result['status'] == 'REFIT_FAMILY'
    assert result['behavior']['normalized_differences'] == [1., 2.]
    assert result['compute_cost'] == 3 + 2


def test_T06_threshold_neighbors_never_become_a_transitive_equivalence_class():
    records = [theory(str(i), Program('const', value=value)) for i, value in enumerate((0., .8, 1.6))]
    for record, value in zip(records, (0., .8, 1.6)):
        record.update(parameter_paths=[], behavior_signature=signature([value]), behavior_tolerance=1.)
    assert typed_program_match(records[0], records[1])['status'] == 'SCOPED_REUSE'
    assert typed_program_match(records[1], records[2])['status'] == 'SCOPED_REUSE'
    assert typed_program_match(records[0], records[2])['status'] == 'COMPARE_ALTERNATIVES'
    assert [record['id'] for record in records] == ['0', '1', '2']


def test_signatures_from_different_panels_readouts_or_units_do_not_match():
    source, target = pair(Program('abs', children=(feature(),)))
    target['program'] = Program('mul', children=(feature(), feature())).to_dict()
    source['behavior_signature'] = signature([1.])
    target['behavior_signature'] = signature([1.], 'other_panel')
    assert typed_program_match(source, target)['status'] == 'NO_COMPATIBLE_CANDIDATE_FOUND'
    target['behavior_signature'] = signature([1.])
    target['output_unit'] = 'mV'
    assert typed_program_match(source, target)['status'] == 'NO_COMPATIBLE_CANDIDATE_FOUND'


def test_finite_library_result_preserves_matches_and_only_claims_search_within_that_pool():
    source, target = pair()
    question = {'id': 'q', 'theory_target': target, 'context_ref': target['context_ref'],
                'declared_scope_ref': target['declared_scope_ref']}
    result = match_theories(question, [source, {**deepcopy(source), 'id': 'second_source'}])
    assert result['status'] == 'REUSE_INSTANCE' and len(result['matches']) == 2
    assert [item['source_id'] for item in result['matches']] == ['old', 'second_source']
    missing = match_theories(question, [])
    assert missing['status'] == 'NO_COMPATIBLE_CANDIDATE_FOUND' and missing['reason'] == 'finite_library_search'
    assert match_theories({'id': 'uncompiled'}, [])['status'] == 'UNKNOWN_MATCH'


def test_truncated_library_search_is_unknown_unless_a_supported_instance_is_already_found():
    source, target = pair()
    other = theory('other', Program('abs', children=(feature(),)))
    incomplete = match_theories({'id': 'q'}, [other, source], target=target, compute_budget=1)
    assert incomplete['status'] == 'UNKNOWN_MATCH' and not incomplete['search_complete']
    sufficient = match_theories({'id': 'q'}, [source, other], target=target, compute_budget=3)
    assert sufficient['status'] == 'REUSE_INSTANCE' and not sufficient['search_complete']
    assert sufficient['compute_cost'] == 3 and len(sufficient['matches']) == 2


def prepare_question(target):
    return {'id': 'current', 'theory_target': target, 'context_ref': deepcopy(target['context_ref']),
            'query_semantics': {'mode': 'model_query', 'query_expression': {'op': 'Readout',
                'binding': {'features': {'x': 2., 'u': 2.}, 'history': []}}}}


def test_prepare_blocks_future_inputs_in_current_public_binding():
    source, target = pair()
    source['role_signature']['x']['time'] = 'next_step'
    question = prepare_question(target)
    result = prepare_theories_for_question(question, [source], match_theories(question, [source]))
    assert result['theories'] == []
    assert result['blocked'][0]['reason'] == 'time_role_mismatch'
    assert result['blocked'][0]['required_work']['type'] == 'AuthorWorkRequest'


def test_prepare_keeps_structurally_different_legitimate_alternatives():
    source, target = pair(Program('abs', children=(feature(),)))
    target['program'] = affine().to_dict()
    question = prepare_question(target)
    matches = match_theories(question, [source])
    assert matches['matches'][0]['status'] == 'NO_COMPATIBLE_CANDIDATE_FOUND'
    result = prepare_theories_for_question(question, [source], matches)
    assert [record['id'] for record in result['theories']] == ['old'] and result['blocked'] == []


def test_prepare_keeps_new_structure_without_empirical_support_as_a_hypothesis():
    source, target = pair(Program('abs', children=(feature(),)))
    target['program'] = affine().to_dict()
    source['support'] = {'status': 'proposed', 'evidence': []}
    question = prepare_question(target)
    result = prepare_theories_for_question(question, [source], match_theories(question, [source]))
    assert [record['id'] for record in result['theories']] == ['old']
    assert result['theories'][0]['support'] == {'status': 'proposed', 'evidence': []}
    assert not result['support_updated'] and result['confirmation'] == 'not_established'


def test_prepare_tested_rename_changes_actual_forward_inputs_without_editing_retained_program():
    from hegel_machine.cognition.models import Observation
    source, target = pair()
    target['program'] = affine(name='u').to_dict()
    target['role_signature'] = {'u': deepcopy(source['role_signature']['x'])}
    question = prepare_question(target)
    question['allowed_mappings'] = [{'feature_map': {'x': 'u'}, 'status': 'tested',
        'test_evidence': [4], 'context_ref': deepcopy(target['context_ref'])}]
    before = deepcopy(source)
    result = prepare_theories_for_question(question, [source], match_theories(question, [source]))
    assert len(result['theories']) == 1 and result['blocked'] == []
    rebound = result['theories'][0]
    assert Program.from_dict(rebound['program']).predict(Observation({'u': 7.}, 0., ())) == 7.
    assert rebound['role_signature'] == target['role_signature']
    assert rebound['prepared_binding']['test_evidence'] == [4]
    assert source == before


def test_prepare_tested_input_and_output_units_change_actual_selector_predictions():
    from hegel_machine.cognition.question_selection import select_question_plan

    source, target = pair(affine(2.))
    target['program'] = affine(2., name='u').to_dict()
    target['role_signature'] = {'u': {**source['role_signature']['x'], 'unit': 'mV'}}
    target['output_unit'] = 'mV'
    question = prepare_question(target)
    action = {'kind': 'trajectory', 'sequence': [{'u': 1500.}]}
    question['query_semantics']['query_expression'] = {'op': 'Readout', 'record_index': 0, 'step': 0,
        'units': 'mV', 'binding': {'action': action, 'features': {'u': 1500.}, 'history': []}}
    mapping = {'feature_map': {'x': 'u'}, 'status': 'tested', 'test_evidence': [4],
        'context_ref': deepcopy(target['context_ref']),
        'unit_conversions': {'x': {'from': 'V', 'to': 'mV', 'scale': 1000., 'offset': 500.}},
        'output_conversion': {'from': 'V', 'to': 'mV', 'scale': 1000., 'offset': 7.}}
    question['allowed_mappings'] = [mapping]
    before = deepcopy(source)
    prepared = prepare_theories_for_question(question, [source], match_theories(question, [source]))
    assert prepared['blocked'] == []
    rebound, = prepared['theories']
    assert rebound['output_unit'] == 'mV' and rebound['role_signature'] == target['role_signature']
    assert rebound['support'] == {'status': 'candidate', 'evidence': [], 'context_ref': target['context_ref']}
    assert rebound['prepared_binding']['source_support'] == source['support']
    assert rebound['prepared_binding']['unit_conversions'] == mapping['unit_conversions']
    assert source == before and source['parameter_paths'] == [[0]]
    assert rebound['prepared_binding']['source_parameter_paths'] == [[0]]
    assert rebound['parameter_paths'] == []  # Unit constants are fixed, not new fit parameters.
    other = {**deepcopy(rebound), 'id': 'other', 'program': Program('const', value=0.).to_dict()}
    experiment = {'id': 'unit_probe', 'actions': [action],
        'query': deepcopy(question['query_semantics']['query_expression']), 'tolerance': 0., 'observation_cost': 1.}
    selection = select_question_plan(question, [rebound, other], [experiment])
    assert selection['status'] == 'selected' and selection['scores'][0]['G'] == 1.
    group = next(group for group in selection['groups'] if source['id'] in group['theory_ids'])
    assert group['task_answer'] == pytest.approx(2007.)
    assert selection['support_updated'] is False and selection['evidence'] == []


@pytest.mark.parametrize('status,evidence', [('proposed', [4]), ('tested', [])])
def test_prepare_unit_mapping_waits_for_executed_target_mapping_check(status, evidence):
    source, target = pair()
    target['role_signature']['x']['unit'] = 'mV'
    question = prepare_question(target)
    question['allowed_mappings'] = [{'feature_map': {'x': 'x'}, 'status': status,
        'evidence': evidence, 'context_ref': deepcopy(target['context_ref']),
        'unit_conversions': {'x': {'from': 'V', 'to': 'mV', 'scale': 1000., 'offset': 0.}}}]
    result = prepare_theories_for_question(question, [source], match_theories(question, [source]))
    assert result['theories'] == []
    assert result['blocked'][0]['reason'] == 'mapping_or_measurement_not_tested'


@pytest.mark.parametrize('program,expected', [
    (Program('lag', name='x', lag=1), 1.),
    (Program('window_sum', lag=2, children=(feature(),)), 3.),
    (Program('event_delta', lag=1, children=(feature(),)), 1.),
])
def test_affine_input_binding_retains_actual_historical_and_reset_semantics(program, expected):
    from hegel_machine.cognition.models import Observation

    source, target = pair(program)
    target['role_signature']['x']['unit'] = 'target_coordinate'
    question = prepare_question(target)
    question['allowed_mappings'] = [{'status': 'tested', 'test_evidence': [4],
        'context_ref': deepcopy(target['context_ref']),
        'unit_conversions': {'x': {'from': 'V', 'to': 'target_coordinate', 'scale': 2., 'offset': 10.}}}]
    prepared = prepare_theories_for_question(question, [source], match_theories(question, [source]))
    bound = Program.from_dict(prepared['theories'][0]['program'])
    assert bound.predict(Observation({'x': 14.}, 0., ({'x': 12.},))) == expected
    assert source['program'] == program.to_dict()


def test_affine_output_binding_converts_reserved_historical_response_in_source_units():
    from hegel_machine.cognition.models import Observation

    source, target = pair(Program('lag', name='__target__', lag=1))
    target['output_unit'] = 'mV'
    question = prepare_question(target)
    question['allowed_mappings'] = [{'status': 'tested', 'test_evidence': [4],
        'context_ref': deepcopy(target['context_ref']),
        'output_conversion': {'from': 'V', 'to': 'mV', 'scale': 1000., 'offset': 7.}}]
    prepared = prepare_theories_for_question(question, [source], [])
    bound = Program.from_dict(prepared['theories'][0]['program'])
    assert bound.predict(Observation({'x': 0.}, 0., ({'x': 0., '__target__': 2007.},))) == pytest.approx(2007.)


def test_two_tested_numeric_mappings_remain_alternatives_instead_of_deduplicating_by_names():
    source, target = pair()
    target['role_signature']['x']['unit'] = 'reading'
    question = prepare_question(target)
    question['allowed_mappings'] = [{'status': 'tested', 'test_evidence': [4],
        'context_ref': deepcopy(target['context_ref']),
        'unit_conversions': {'x': {'from': 'V', 'to': 'reading', 'scale': scale, 'offset': 0.}}}
        for scale in (2., 3.)]
    prepared = prepare_theories_for_question(question, [source], match_theories(question, [source]))
    assert prepared['theories'] == []
    assert prepared['blocked'][0]['reason'] == 'multiple_runnable_bindings_need_selection'
    assert len(prepared['blocked'][0]['required_work']['required_contract']['mappings']) == 2


def test_prepare_opaque_python_is_retained_as_work_instead_of_sent_to_program_selector():
    source, target = pair()
    source.pop('program')
    source.update(python_source='def predict(row): return row["x"]\n', fit_state={}, binding={'x': 'drive'})
    question = prepare_question(target)
    before = deepcopy(source)
    result = prepare_theories_for_question(question, [source], match_theories(question, [source]))
    assert result['theories'] == []
    assert result['blocked'][0]['reason'] == 'opaque_python_needs_executable_predictor_binding'
    assert source == before


def test_prepare_refit_uses_only_an_existing_declared_action_or_an_author_obligation():
    source, target = pair()
    target['program'] = affine(2.).to_dict()
    question = prepare_question(target)
    matches = match_theories(question, [source])
    absent = prepare_theories_for_question(question, [source], matches)
    assert absent['theories'] == []
    assert absent['blocked'][0]['required_work']['object_type'] == 'fitted_instance'
    action = {'action': 'run', 'arguments': {'operator': 'refit', 'model': 'old.py', 'validation_records': [1]}}
    source['refit_action'] = action
    existing = prepare_theories_for_question(question, [source], matches)
    assert existing['blocked'][0]['required_work'] == action
    assert 'action' not in absent['blocked'][0]['required_work']


def test_prepare_blocks_missing_current_premise_even_for_a_new_structure():
    source, target = pair(Program('abs', children=(feature(),)))
    target['program'] = affine().to_dict()
    source['premise_refs'] = [{'question_id': 'calibration', 'answer_version': 2}]
    question = prepare_question(target)
    question['required_premises'] = deepcopy(source['premise_refs'])
    result = prepare_theories_for_question(question, [source], match_theories(question, [source]))
    assert result['theories'] == [] and result['blocked'][0]['reason'] == 'missing_current_premise_support'


def test_prepare_lag_only_fields_are_typed_and_can_use_public_history():
    source, target = pair(Program('lag', name='past_drive', lag=1))
    role = {'role': 'drive', 'unit': 'V', 'time': 'current'}
    source['role_signature'] = {'past_drive': {**role, 'time': 'next_step'}}
    target['role_signature'] = {'past_drive': role}
    question = prepare_question(target)
    question['query_semantics']['query_expression']['binding'] = {
        'features': {'x': 2.}, 'history': [{'past_drive': 1.}]}
    result = prepare_theories_for_question(question, [source], [])
    assert result['theories'] == [] and result['blocked'][0]['reason'] == 'time_role_mismatch'
    source['role_signature']['past_drive'] = deepcopy(role)
    accepted = prepare_theories_for_question(question, [source], [])
    assert [item['id'] for item in accepted['theories']] == ['old'] and accepted['blocked'] == []
