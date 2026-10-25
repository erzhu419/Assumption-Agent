"""Conflicts limit an exact target view, without inventing world evidence."""

from copy import deepcopy

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.revision_parameter_relations import (
    diagnose_parameter_targets, learn_parameter_relations, target_conflicts,
)


def test_retained_equal_pair_has_a_necessary_bound_even_before_sample_gate():
    values = [.5053650313216267, .5051538816938639,
              .5049580761603409, .5047678342783982]
    features = [{'parent': -.25, 'condition': -.6, 'paid_rows': 97 + i} for i in range(4)]
    budget = SearchBudget(100)
    report = target_conflicts(features, values, range(4), ['parent', 'condition'], 1e-10, budget)
    assert report['status'] == 'conflict'
    witness = report['witnesses'][0]
    assert witness['source_record_indices'] == [0, 1, 2, 3]
    assert witness['necessary_max_abs_error_lower_bound'] == .00029859852161423506
    assert witness['target_values'] == values
    assert report['actual_compute_cost'] == budget.spent == 4


def test_no_observed_conflict_does_not_establish_input_sufficiency_or_use_held():
    features = [{'parent': 1., 'condition': 2.}, {'parent': 1., 'condition': 2.},
                {'parent': 1., 'condition': 2.}]
    result = target_conflicts(features, [3., 3., 100.], [0, 1],
                              ['parent', 'condition'], 1e-10, SearchBudget(100))
    assert result['status'] == 'UNKNOWN'
    assert result['checked_indices'] == [0, 1]
    assert not result['witnesses']
    assert 'not_sufficiency' in result['reason']


def test_budget_cut_keeps_already_proved_conflict_and_actual_fee():
    budget = SearchBudget(2)
    result = target_conflicts([{'x': 0.}] * 4, [0., 1., 2., 3.], range(4), ['x'],
                              1e-10, budget)
    assert result['budget_exhausted'] and result['status'] == 'conflict'
    assert result['checked_indices'] == [0, 1]
    assert result['witnesses'][0]['necessary_max_abs_error_lower_bound'] == .5
    assert result['actual_compute_cost'] == budget.spent == 2


def test_conflict_bound_and_intersection_use_each_raw_coordinate_scale():
    features = [{'x': 1.}] * 3
    # The canonical spread exceeds 2*tolerance, yet differently scaled raw
    # tolerance intervals have a shared canonical value.
    allowed = target_conflicts(features, [0., 2.2e-10], [0, 1], ['x'], 1e-10,
        SearchBudget(100), coordinate_scales=[.5, 2.])
    assert allowed['status'] == 'UNKNOWN' and not allowed['witnesses']
    # Conversely a small canonical spread cannot relax a small raw scale.
    rejected = target_conflicts(features, [0., 3e-11, 1e-11], range(3), ['x'], 1e-10,
        SearchBudget(100), coordinate_scales=[.01, -.1, 2.])
    witness = rejected['witnesses'][0]
    assert rejected['status'] == 'conflict'
    assert abs(witness['necessary_max_abs_error_lower_bound'] - 3e-11 / .11) < 1e-24
    assert witness['common_canonical_interval']['lower'] > witness['common_canonical_interval']['upper']
    assert witness['error_bound_coordinate'] == 'saved_raw_parameter'


def scalar_records(levels=((1., 2., 3.), (4., 5., 6.))):
    candidate = {'id': 'explicit_scalar_fixture', 'parent_holes': ['before'], 'holes': [
        {'name': name, 'output_unit': 'dimensionless'} for name in ('before', 'after')]}
    records = []
    for source, values in enumerate(levels):
        for index, value in enumerate(values):
            before = {'file': f'source_{source}.json', 'json_path': [index, 'before']}
            after = {'file': f'source_{source}.json', 'json_path': [index, 'after']}
            records.append({'source_id': f'physical_{source}', 'episode_id': f'{source}:{index}',
                'before_ref': before, 'after_ref': after, 'source_ref': after,
                'bindings': {'before': Program('const', value=value),
                             'after': Program('const', value=2 * value + 1)},
                'parameter_role_contracts': {name: {'status': 'known',
                    'role_key': ['explicit_fixture', name], 'coordinate_scale': 1.,
                    'coefficient_unit': 'dimensionless'} for name in ('before', 'after')}})
    return candidate, records


CONTRACT = {'operations': ['refit'], 'max_structures': 2,
            'risk_quantile': .5, 'tolerance': 1e-10}


def test_full_view_conflict_precedes_global_sample_gate_and_never_fits(monkeypatch):
    candidate, records = scalar_records(((1., 2.), (1., 2.)))
    records[2]['bindings']['after'] = Program('const', value=4.)

    def forbidden(*args, **kwargs):
        raise AssertionError('a contradictory training view must not fit')

    monkeypatch.setattr('hegel_machine.cognition.revision_parameter_relations.fit_parameter_slots', forbidden)
    result = learn_parameter_relations(candidate, records, SearchBudget(1000), search_contract=CONTRACT)
    assert not result['accepted'] and not result['attempts']
    assert result['reason'] == 'all_exact_parameter_target_views_conflict'
    witness = result['target_conflicts']['after']['witnesses'][0]
    assert witness['source_record_indices'] == [0, 2]
    assert witness['necessary_max_abs_error_lower_bound'] == .5
    assert any(action.get('output_hole') == 'after' for action in result['next_actions'])


def test_attempt_projection_conflict_precedes_its_variation_gate():
    candidate, records = scalar_records(((1., 2., 3.), (1., 2., 3.)))
    for index, record in enumerate(records):
        source = index // 3
        record['bindings']['after'] = Program('const', value=2 * record['bindings']['before'].value + source + 1)
        record['evidence_context'] = {'features': {'condition': source},
            'role_signature': {'condition': {'role': 'measured_condition',
                'unit': 'dimensionless', 'time': 'before_revision'}},
            'selection': 'retained_source_condition', 'source_refs': [record['before_ref']]}
    result = learn_parameter_relations(candidate, records, SearchBudget(5000),
                                      search_contract={**CONTRACT, 'max_structures': 4})
    attempt = next(item for item in result['attempts'] if item['input_holes'] == ['before']
                   and not item['evidence_inputs'])
    assert attempt['status'] == 'refuted' and 'fitting' not in attempt
    assert 'distinct_training_inputs' not in attempt
    assert attempt['target_conflict']['status'] == 'conflict'
    assert result['target_conflicts']['after']['status'] == 'UNKNOWN'


def test_role_coordinates_are_canonical_and_missing_state_requests_work_without_vetoing_empirical_map():
    candidate, records = scalar_records()
    original = deepcopy(records)
    for record in records[3:]:
        record['parameter_role_contracts']['before']['coordinate_scale'] = 2.
        record['bindings']['before'] = Program('const', value=record['bindings']['before'].value / 2)
        record['parameter_role_contracts']['after']['coordinate_scale'] = 3.
        record['bindings']['after'] = Program('const', value=record['bindings']['after'].value / 3)
    saved = deepcopy(records)
    diagnosis = diagnose_parameter_targets(candidate, records, SearchBudget(1000))
    assert diagnosis['constants']['before'] == [record['bindings']['before'].value for record in original]
    assert diagnosis['constants']['after'] == [record['bindings']['after'].value for record in original]
    result = learn_parameter_relations(candidate, records, SearchBudget(5000), search_contract=CONTRACT)
    mapping = result['accepted'][0]
    assert mapping['purpose'] == 'exact_edit_map' and mapping['parameter_coordinate'] == 'canonical_basis'
    assert mapping['input_roles']['before']['source_coordinate_scales'] == [1.] * 3 + [2.] * 3
    assert mapping['output_role']['source_coordinate_scales'] == [1.] * 3 + [3.] * 3
    assert mapping['target_contract']['generating_process'] == 'UNKNOWN'
    assert mapping['target_contract']['target_identifiability'] == 'UNKNOWN'
    assert mapping['status'] == 'empirical_candidate' and not mapping['scientific_support']
    assert any(action['reason'] == 'saved_parameter_generating_process_not_established'
               for action in result['next_actions'])
    assert records == saved


def test_actual_guard_history_role_overrides_same_hole_and_explicit_fixture_contract(monkeypatch):
    candidate, records = scalar_records()
    feature = Program('feature', name='x')
    candidate['before_pattern'] = Program('mul', children=(Program('arg', name='before'), feature)).to_dict()
    candidate['definition'] = Definition('after_fixture', ('after',),
        Program('mul', children=(Program('arg', name='after'), feature))).to_dict()
    for index, record in enumerate(records):
        record.update(_roles={'x': {'role': 'controlled_input', 'unit': 'dimensionless', 'time': 'current'}},
                      _output_unit='dimensionless')
        record['_before_program'] = Program('mul', children=(record['bindings']['before'], feature))
        basis = feature if index < 3 else Program('window_sum', children=(feature,), lag=2)
        record['_after_program'] = Program('mul', children=(record['bindings']['after'], basis))

    def forbidden(*args, **kwargs):
        raise AssertionError('different computational roles cannot be a shared exact target')

    monkeypatch.setattr('hegel_machine.cognition.revision_parameter_relations.fit_parameter_slots', forbidden)
    result = learn_parameter_relations(candidate, records, SearchBudget(5000), search_contract=CONTRACT)
    assert not result['attempts'] and not result['accepted']
    assert result['reason'] == 'common_computational_parent_and_after_parameter_roles_required'
    assert result['excluded'][0]['hole'] == 'after'
    assert result['excluded'][0]['source_record_indices'] == [3, 4, 5]


def test_small_coordinate_scale_cannot_relax_original_raw_acceptance_tolerance(monkeypatch):
    candidate, records = scalar_records()
    for record in records:
        record['parameter_role_contracts']['after']['coordinate_scale'] = .01

    def deliberately_inexact(template, mapping, rows, budget, quantile, *, report):
        report.update(candidate_methods=['initial'], parameter_identifiability='UNKNOWN', status='UNKNOWN')
        return [Program('add', children=(
            Program('mul', children=(Program('const', value=.02), Program('feature', name='before'))),
            Program('const', value=.01 + 5e-11)))]

    monkeypatch.setattr('hegel_machine.cognition.revision_parameter_relations.fit_parameter_slots',
                        deliberately_inexact)
    result = learn_parameter_relations(candidate, records, SearchBudget(5000), search_contract=CONTRACT)
    assert not result['accepted']
    checks = next(item['checks'][0] for item in result['attempts'] if item.get('checks'))
    assert all(abs(row['canonical_residual']) < CONTRACT['tolerance']
               and abs(row['residual']) > CONTRACT['tolerance']
               and row['status'] == 'failed' and row['coordinate_scale'] == .01
               for label in ('training', 'held') for row in checks[label])
    assert all(row['expected_parameter'] == records[row['source_record_index']]['bindings']['after'].value
               for label in ('training', 'held') for row in checks[label])
