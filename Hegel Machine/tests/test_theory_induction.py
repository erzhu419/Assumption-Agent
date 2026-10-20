"""Source reconstruction, fixed encoding costs and scoped counterexamples."""
from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.abstractions import AbstractTemplate
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.theory_induction import consolidate_theories, instantiate_induction


def feature(name='x'):
    return Program('feature', name=name)


def affine(name='x', slope=2., intercept=1.):
    return Program('add', children=(Program('mul', children=(Program('const', value=slope), feature(name))),
                                    Program('const', value=intercept)))


def episode(identifier='a', name='x', slope=2., intercept=1., source_id=None):
    return {'episode_id': identifier, 'source_id': source_id or identifier,
            'source_ref': {'source_id': source_id or identifier, 'round_index': 0},
            'program': affine(name, slope, intercept).to_dict(),
            'role_signature': {name: {'role': 'drive', 'unit': 'V', 'time': 'current'}},
            'output_unit': 'V', 'context_ref': {'instrument': 'meter', 'reset': 'per_trajectory'},
            'declared_scope_ref': 'bounded_drive',
            'evidence_refs': [{'source_id': source_id or identifier, 'record_index': 0}]}


def test_ordered_affine_sources_reconstruct_through_existing_abstract_template():
    originals = [episode(), episode('b', 'u', 4., -3.)]
    before = deepcopy(originals)
    result = consolidate_theories(originals)
    candidate, = result['candidates']
    assert result['status'] == 'completed'
    assert candidate['frozen'] and candidate['global_support'] is False
    assert candidate['transfer_status'] == 'not_established'
    assert candidate['source_counterexample_status'] == 'not_tested'
    assert candidate['parameter_paths'] == [[0, 0], [1]]
    assert candidate['role_signature'] == {'role_0': {'role': 'drive', 'unit': 'V', 'time': 'current'}}
    template = AbstractTemplate.from_dict(candidate['template'])
    for original, instance in zip(originals, candidate['instances']):
        assert instantiate_induction(candidate, instance) == Program.from_dict(original['program'])
        assert template.instantiate(instance['mapping'], instance['parameters']) == Program.from_dict(original['program'])
        assert instance['inverse_mapping'] == {next(iter(original['role_signature'])): 'role_0'}
    assert originals == before
    assert result['actual_compute_cost'] == 22
    assert all(item['exact'] for item in candidate['source_reconstruction'])


@pytest.mark.parametrize('change', ['role', 'unit', 'time', 'context', 'output', 'lag', 'window', 'order'])
def test_actual_type_time_context_and_order_changes_are_not_hard_unified(change):
    left, right = episode(), episode('b', 'u')
    if change in {'role', 'unit', 'time'}:
        right['role_signature']['u'][change] = 'different'
    elif change == 'context':
        right['context_ref']['instrument'] = 'another'
    elif change == 'output':
        right['output_unit'] = 'A'
    elif change == 'lag':
        left['program'] = Program('lag', name='x', lag=1).to_dict()
        right['program'] = Program('lag', name='u', lag=2).to_dict()
    elif change == 'window':
        left['program'] = Program('window_sum', lag=2, children=(feature(),)).to_dict()
        right['program'] = Program('window_sum', lag=3, children=(feature('u'),)).to_dict()
    else:
        left['program'] = Program('sub', children=(feature(), Program('const', value=1.))).to_dict()
        right['program'] = Program('sub', children=(Program('const', value=1.), feature('u'))).to_dict()
    result = consolidate_theories([left, right])
    assert result['status'] == 'completed' and result['candidates'] == []
    assert len(result['author_work']) == 2


def test_variable_sharing_cannot_become_two_distinct_observables():
    left, right = episode(), episode('b', 'u')
    left['program'] = Program('add', children=(feature(), feature())).to_dict()
    right['program'] = Program('add', children=(feature('u'), feature('v'))).to_dict()
    right['role_signature']['v'] = deepcopy(right['role_signature']['u'])
    assert consolidate_theories([left, right])['candidates'] == []


def test_duplicate_notes_and_same_source_rounds_do_not_add_support_origins_t21():
    left, right = episode(source_id='one_trace'), episode('b', 'u', source_id='one_trace')
    left['evidence_refs'][0]['note'] = 'first summary'
    right['evidence_refs'][0]['note'] = 'second summary'
    right['source_ref']['round_index'] = 1
    candidate, = consolidate_theories([left, right])['candidates']
    assert candidate['independent_source_count'] == 1
    assert len(candidate['source_evidence_refs']) == 1
    assert len(candidate['instances']) == 2
    assert not candidate['active_library']


def test_record_indices_from_distinct_sources_are_distinct_and_file_notes_are_not_votes():
    left, right = episode(), episode('b', 'u')
    left['evidence_refs'] = [{'record_index': 0}]
    right['evidence_refs'] = [{'record_index': 0}]
    candidate, = consolidate_theories([left, right])['candidates']
    assert len(candidate['source_evidence_refs']) == 2
    right['source_id'] = 'a'
    for item, note in ((left, 'first'), (right, 'another')):
        item['evidence_refs'] = [{'file': 'laboratory_observations.json', 'json_path': [0], 'note': note}]
    candidate, = consolidate_theories([left, right])['candidates']
    assert len(candidate['source_evidence_refs']) == 1


def test_source_failure_outcomes_remain_referenced_without_copying_observations():
    left, right = episode(), episode('b', 'u')
    left.update({'status': 'refuted', 'failure_refs': [{'file': 'research_state.json', 'json_path': ['revisions', 3]}],
                 'adoption': {'occurred': False},
                 'independent_results': [{'source_ref': {'file': 'confirmation.json'}, 'passed': False}]})
    candidate, = consolidate_theories([left, right])['candidates']
    outcome = candidate['source_outcomes'][0]
    assert outcome['status'] == 'refuted' and outcome['failure_refs'] == left['failure_refs']
    assert not outcome['adoption']['occurred'] and outcome['independent_results'] == left['independent_results']
    assert 'program' not in outcome and 'observation' not in outcome


def test_numeric_counterexample_and_textual_claim_remain_visible_t22():
    left, right = episode(), episode('b', 'u')
    left['counterexamples'] = [{'source_ref': {'source_id': 'a', 'record_index': 8},
                              'claim': 'response always equals 99',
                              'observation': {'features': {'x': 1.}, 'target': 99., 'history': []}}]
    candidate, = consolidate_theories([left, right])['candidates']
    check, = candidate['counterexample_checks']
    assert check['prediction'] == 3. and check['actual'] == 99.
    assert check['claim'] == 'response always equals 99'
    assert check['status'] == 'failed' and check['domain'] == 'source'
    assert candidate['source_counterexample_status'] == 'failed'
    assert all(item['exact'] for item in candidate['source_reconstruction'])
    assert not candidate['active_library']


def test_named_lookup_bodies_parameters_mappings_and_exceptions_still_cost_bytes_t23():
    left, right = episode(), episode('b', 'u', 4., -3.)
    long_body = feature()
    for index in range(20):
        long_body = Program('if', children=(Program('lt', children=(feature(), Program('const', value=float(index)))),
                                            Program('const', value=float(index * index)), long_body))
    definition = Definition('tiny_name', ('input',), long_body)
    call = Program('call', name='tiny_name', definition=definition, children=(feature(),))
    left['program'] = call.to_dict()
    right['program'] = call.remap_features({'x': 'u'}).to_dict()
    # A definition body reading concrete x is deliberately not an argument
    # rename; keep that actual source's x contract rather than pretending it is u.
    right['role_signature']['x'] = deepcopy(left['role_signature']['x'])
    left['exceptions'] = [{'full_original_payload': list(range(120))}]
    left['lookup_table'] = [{'x': index, 'y': index * index} for index in range(100)]
    candidate, = consolidate_theories([left, right])['candidates']
    cost = candidate['encoding_cost']
    assert cost['definition_bytes'] > 4000
    assert cost['mapping_and_parameter_bytes'] > 100
    assert cost['selector_bytes'] > 100
    assert cost['exception_bytes'] > 1000
    assert cost['parameter_precision'] == 'original_json_float' and not cost['formal_mdl']
    assert cost['candidate_bytes'] >= cost['definition_bytes'] + cost['exception_bytes']


def test_baseline_counts_original_named_programs_instead_of_expanding_their_storage_cost():
    body = Program('mul', children=(Program('arg', name='input'), Program('const', value=3.)))
    definition = Definition('shared_original', ('input',), body)
    call = Program('call', name='shared_original', definition=definition, children=(feature(),))
    left, right = episode(), episode('b', 'u')
    left['program'], right['program'] = call.to_dict(), call.remap_features({'x': 'u'}).to_dict()
    candidate, = consolidate_theories([left, right])['candidates']
    baseline = {'programs': [left['program'], right['program']],
                'instances': [{'source_id': 'a', 'program_index': 0}, {'source_id': 'b', 'program_index': 1}],
                'exceptions': []}
    expected = len(json.dumps(baseline, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8'))
    assert candidate['encoding_cost']['original_bytes'] == expected
    assert all(item['comparison'] == 'ordered_expanded_Program' for item in candidate['source_reconstruction'])


def test_transfer_counterexample_blocks_its_binding_without_erasing_source_reconstruction_t24():
    left, right = episode(), episode('b', 'u', 4., -3.)
    left['counterexamples'] = [{'source_ref': {'source_id': 'new_domain', 'record_index': 1},
                              'context_ref': {'instrument': 'other'},
                              'mapping': {'role_0': 'z'}, 'parameters': [2., 1.],
                              'observation': {'features': {'z': 1.}, 'target': 30.}}]
    candidate, = consolidate_theories([left, right])['candidates']
    assert candidate['source_counterexample_status'] == 'not_tested'
    assert all(item['exact'] for item in candidate['source_reconstruction'])
    assert candidate['blocked_transfer_bindings'][0]['mapping'] == {'role_0': 'z'}
    assert candidate['counterexample_checks'][0]['domain'] == 'transfer'
    assert candidate['counterexample_checks'][0]['status'] == 'failed'
    assert candidate['transfer_status'] == 'not_established'


def test_arbitrary_python_remains_author_work_and_missing_units_are_not_inferred():
    python = {**episode(), 'program': None, 'predictor_path': 'existing_predictor.py'}
    missing = episode('b')
    del missing['role_signature']['x']['unit']
    result = consolidate_theories([python, missing])
    assert result['candidates'] == []
    assert result['author_work'][0]['reason'] == 'arbitrary_python_requires_authored_Program'
    assert result['excluded'][0]['reason'] == 'missing_role_unit_or_time_contract'


@pytest.mark.parametrize('allocation', [{'compute': 0}, {'compute': 8}, {'compute': 1000, 'max_groups': 0}])
def test_real_truncation_is_unknown_and_charged_inside_allocation(allocation):
    result = consolidate_theories([episode(), episode('b', 'u')], allocation=allocation)
    assert result['status'] == 'UNKNOWN' and result['budget_exhausted']
    assert result['actual_compute_cost'] <= allocation['compute']
    assert result['candidates'] == []
    assert result['new_observations'] == result['llm_calls'] == 0


def test_historical_fold_prediction_steps_are_charged_and_insufficient_budget_stays_unknown():
    left, right = episode(), episode('b', 'u')
    left['program'] = Program('accumulate', value=1., children=(feature(), Program('const', value=0.))).to_dict()
    right['program'] = Program('accumulate', value=1., children=(feature('u'), Program('const', value=0.))).to_dict()
    left['counterexamples'] = [{'source_ref': 'retained_fold',
                              'observation': {'features': {'x': 1.}, 'target': 3., 'history': [{'x': 1.}, {'x': 1.}]}}]
    result = consolidate_theories([left, right])
    candidate, = result['candidates']
    assert candidate['counterexample_checks'][0]['status'] == 'passed'
    assert result['actual_compute_cost'] == 20
    truncated = consolidate_theories([left, right], allocation={'compute': 18})
    assert truncated['status'] == 'UNKNOWN' and truncated['candidates'] == []


def test_true_shared_ordered_fragment_uses_existing_extractor_with_its_own_type():
    left, right = episode(), episode('b', 'u')
    fragment_left = Program('abs', children=(feature(),))
    fragment_right = Program('abs', children=(feature('u'),))
    left['program'] = Program('add', children=(fragment_left, Program('const', value=1.))).to_dict()
    right['program'] = Program('mul', children=(fragment_right, Program('const', value=3.))).to_dict()
    for item in (left, right):
        item['fragment_contracts'] = {'0': {'output_unit': 'V'}}
    result = consolidate_theories([left, right])
    candidate, = result['candidates']
    assert candidate['kind'] == 'shared_subexpression'
    assert [instance['source_path'] for instance in candidate['instances']] == [[0], [0]]
    assert instantiate_induction(candidate, candidate['instances'][0]) == fragment_left
    assert all(item['target'] == 'source_fragment' for item in candidate['source_reconstruction'])


def test_distinct_contexts_require_explicit_family_but_remain_unconfirmed():
    left, right = episode(), episode('b', 'u')
    right['context_ref'] = {'instrument': 'another'}
    left['context_family_ref'] = right['context_family_ref'] = 'declared_meter_family'
    candidate, = consolidate_theories([left, right])['candidates']
    assert len(candidate['source_contexts']) == 2
    assert candidate['scope']['context_ref'] == 'declared_meter_family'
    assert candidate['transfer_status'] == 'not_established'
