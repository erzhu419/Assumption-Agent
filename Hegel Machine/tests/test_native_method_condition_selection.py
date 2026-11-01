"""Native callers consume the same complete-method condition evidence as Common."""

from copy import deepcopy
import json

from hegel_machine.cognition.method_conditions import (
    native_condition_bindings, select_native_method_programs, method_condition_support,
)
from test_v8_paired_method_conditions import pair, learn


def setup_case():
    pairs = [pair(0), pair(1), pair(2, stage='complete', challenger_cost=200),
        pair(3, role='condition_validation'),
        pair(4, stage='complete', challenger_cost=200, role='condition_validation')]
    research = learn(pairs)
    baseline = {'transfer_partition': {'kind': 'Pi0'}}
    challenger = {'transfer_partition': {'kind': 'Pi1',
        'second_stage': {'on': 'remaining_residual', 'operation': 'resolve_relation_intervals'}}}
    records = [{'source_concept_id': 'old', 'research_program': baseline},
        {'source_concept_id': 'learned', 'research_program': challenger},
        {'source_concept_id': 'other', 'research_program': {'unchanged': True}}]
    binding = {'baseline_program': baseline, 'challenger_program': challenger,
        'research': research, 'condition_ref': {'file': 'paid_conditions.json'}}
    return pairs, records, binding


def test_validated_condition_changes_actual_queue_and_preserves_complete_local_branch():
    pairs, records, binding = setup_case()
    before = deepcopy(records)
    positive = select_native_method_programs(records, pairs[0]['baseline']['profile'], [binding])
    negative = select_native_method_programs(records, pairs[2]['baseline']['profile'], [binding])
    assert positive['programs'] == records[1:]
    assert negative['programs'] == [records[0], records[2]]
    assert positive['compute_cost'] == len(binding['research']['candidates'])
    assert positive['programs'][0]['research_program']['transfer_partition']['second_stage']
    assert records == before


def test_incomplete_new_evidence_retires_old_guard_without_erasing_original_queue():
    pairs, records, binding = setup_case()
    latest = deepcopy(binding)
    latest['research']['general_guard_supported'] = False
    result = select_native_method_programs(records, pairs[0]['baseline']['profile'], [binding, latest])
    assert result['programs'] == records and result['compute_cost'] == 0
    assert len(result['decisions']) == 1
    assert result['decisions'][0]['reason'] == 'paired_condition_not_validated'


def test_missing_public_numbers_and_changed_complete_body_do_not_inherit_method_credit():
    pairs, records, binding = setup_case()
    profile = deepcopy(pairs[0]['baseline']['profile'])
    profile['research']['budget']['remaining_compute'] = None
    result = select_native_method_programs(records, profile, [binding])
    assert result['programs'] == records and result['compute_cost'] == 0
    assert result['decisions'][0]['reason'] == 'outside_measured_public_coverage'
    records[1]['research_program'] = {'transfer_partition': {'kind': 'Pi1'}}
    result = select_native_method_programs(records, pairs[0]['baseline']['profile'], [binding])
    assert result['programs'] == records
    assert result['decisions'][0]['reason'] == 'both_bound_complete_methods_required'


def test_disagreeing_conditions_retain_both_competitors():
    pairs, records, binding = setup_case()
    binding['research']['candidates'].append({'condition': {'op': 'const', 'value': 0., 'children': []}})
    result = select_native_method_programs(records, pairs[0]['baseline']['profile'], [binding])
    assert result['programs'] == records
    assert result['decisions'][0]['reason'] == 'competing_conditions_unresolved'


def test_validated_source_range_reaches_consumer_without_using_validation_to_fit_formula():
    pairs, records, binding = setup_case()
    for item in pairs[-2:]:
        for side in ('baseline', 'challenger'):
            item[side]['profile']['research']['budget']['remaining_compute'] = 2000
    learned = learn(pairs)
    assert learned['general_guard_supported']
    assert learned['candidates'] == binding['research']['candidates']
    assert method_condition_support(pairs[-2]['baseline']['profile'], learned)['can_apply']
    binding['research'] = learned
    result = select_native_method_programs(records, pairs[-2]['baseline']['profile'], [binding])
    assert result['decisions'][0]['selected'] == 'challenger'
    outside = deepcopy(pairs[-2]['baseline']['profile'])
    outside['research']['budget']['remaining_compute'] = 2001
    assert not method_condition_support(outside, learned)['can_apply']


def test_binding_loads_actual_method_bodies_instead_of_matching_version_names(tmp_path):
    _, records, binding = setup_case()
    for side, record in zip(('baseline', 'challenger'), records):
        (tmp_path / (side + '_program.json')).write_text(json.dumps(record['research_program']))
        (tmp_path / (side + '_method.json')).write_text(json.dumps({'execution': {
            'entry': 'replay_research_program',
            'research_program_ref': {'file': str(tmp_path / (side + '_program.json'))}}}))
    group = {**binding['research'], 'method_ref': {'id': 'method', 'version': 1},
        'baseline_method_artifact_ref': {'file': str(tmp_path / 'baseline_method.json')},
        'method_artifact_ref': {'file': str(tmp_path / 'challenger_method.json')}}
    actual = native_condition_bindings([group], tmp_path)
    assert actual[0]['baseline_program'] == binding['baseline_program']
    assert actual[0]['challenger_program'] == binding['challenger_program']
