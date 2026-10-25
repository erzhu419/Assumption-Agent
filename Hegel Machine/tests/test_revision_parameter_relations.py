"""Parameter edit maps are metadata hypotheses, with actual held revisions."""

from copy import deepcopy
import json
from random import Random

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.revision_parameter_relations import (
    learn_joint_parameter_relations, learn_parameter_relations,
)
from hegel_machine.cognition.synthesis import OPERATIONS


CONTRACT = {'operations': ['refit'], 'max_structures': 2, 'risk_quantile': .5, 'tolerance': 1e-10}


def sources(tmp_path, *, values=((1., 2., 3.), (4., 5., 6.)), bad_held=False):
    candidate = {'id': 'metadata_fixture', 'parent_holes': ['old_coefficient'], 'holes': [
        {'name': 'old_coefficient', 'output_unit': 'dimensionless', 'time_contract': ['current']},
        {'name': 'new_coefficient', 'output_unit': 'dimensionless', 'time_contract': ['current']}]}
    records = []
    for source_index, levels in enumerate(values):
        path = tmp_path / f'source_{source_index}.json'
        revisions = []
        for index, value in enumerate(levels):
            old = Program('const', value=value)
            new = Program('const', value=2. * value + 1.
                + (.25 if bad_held and source_index == 1 and index == len(levels) - 1 else 0.))
            before = {'file': str(path), 'json_path': ['revisions', index, 'before']}
            after = {'file': str(path), 'json_path': ['revisions', index, 'after']}
            revisions.append({'before': old.to_dict(), 'after': new.to_dict()})
            records.append({'source_id': f'physical_{source_index}',
                'episode_id': f'source_{source_index}:revision_{index}',
                'source_ref': after, 'before_ref': before, 'after_ref': after,
                'failure_refs': [{'file': str(path), 'json_path': ['source_failure']}],
                'bindings': {'old_coefficient': old, 'new_coefficient': new},
                'parameter_role_contracts': {name: {
                    'role_key': {'fixture_coordinate': role}, 'coordinate_scale': 1.,
                    'coefficient_unit': 'dimensionless', 'status': 'known'}
                    for name, role in (('old_coefficient', 'before_response_coefficient'),
                                       ('new_coefficient', 'after_response_coefficient'))}})
        path.write_text(json.dumps({'source_failure': True, 'revisions': revisions}))
    return candidate, records


def learn(candidate, records, limit=1000):
    budget = SearchBudget(limit, spent=3)
    before = budget.spent
    result = learn_parameter_relations(candidate, records, budget, search_contract=CONTRACT)
    assert result['actual_compute_cost'] == budget.spent - before
    assert budget.spent <= budget.limit
    return result


def test_old_grammar_learns_nontrivial_edit_map_and_checks_each_source_last_revision(tmp_path):
    candidate, records = sources(tmp_path)
    original = deepcopy((candidate, records))
    result = learn(candidate, records)
    assert (candidate, records) == original
    assert result['status'] == 'completed' and not result['budget_exhausted']
    assert len(result['accepted']) == 1 and result['actual_compute_cost'] > 0
    mapping = result['accepted'][0]
    assert mapping['output_hole'] == 'new_coefficient'
    assert mapping['input_holes'] == ['old_coefficient']
    assert mapping['output_unit'] == 'dimensionless'
    program = Program.from_dict(mapping['program'])
    assert abs(program.predict(Observation({'old_coefficient': 8.}, 0.)) - 17.) < CONTRACT['tolerance']
    assert [check['episode_id'] for check in mapping['training_checks']] == [
        'source_0:revision_0', 'source_0:revision_1', 'source_1:revision_0', 'source_1:revision_1']
    assert [check['episode_id'] for check in mapping['held_checks']] == [
        'source_0:revision_2', 'source_1:revision_2']
    assert all(check['status'] == 'passed' for check in mapping['training_checks'] + mapping['held_checks'])
    assert mapping['status'] == 'empirical_candidate' and not mapping['scientific_support']
    assert result['support_updated'] is False and result['independent_confirmation'] is False
    assert all(check['source_ref'] and result['source_records'][check['source_record_index']]['failure_refs']
               for check in mapping['held_checks'])
    assert len(result['source_records']) == len(records)
    assert all('bindings' not in record for record in result['source_records'])
    # Original failures live once, rather than once per fitted residual/check.
    assert json.dumps(result).count('"failure_refs"') == len(records)
    assert all('before_ref' not in check and 'failure_refs' not in check
               for check in mapping['source_refs'] + mapping['training_checks'] + mapping['held_checks'])


def test_held_counterexample_rejects_same_learned_map_without_refitting_on_held(tmp_path):
    candidate, records = sources(tmp_path)
    good = learn(candidate, records)
    _, bad_records = sources(tmp_path, bad_held=True)
    bad = learn(candidate, bad_records)
    assert good['accepted'] and bad['accepted'] == []
    fitted = next(item for attempt in bad['attempts'] for item in attempt['checks']
                  if item['program'] == good['accepted'][0]['program'])
    assert all(check['status'] == 'passed' for check in fitted['training'])
    assert [check['status'] for check in fitted['held']] == ['passed', 'failed']
    assert fitted['held'][-1]['after_ref'] == bad_records[-1]['after_ref']
    assert abs(fitted['held'][-1]['residual']) > .24
    assert not bad['scientific_support']


@pytest.mark.parametrize('values', [((1., 2.), (3., 4.)), ((1., 2., 3.), (1., 2., 4.))])
def test_two_training_points_or_two_distinct_inputs_cannot_establish_free_affine_map(tmp_path, values):
    candidate, records = sources(tmp_path, values=values)
    result = learn(candidate, records)
    assert result['status'] == 'UNKNOWN' and result['accepted'] == []
    assert not result['budget_exhausted']
    assert result.get('reason') == 'insufficient_parameter_training_points' or (
        result['attempts'][-1]['reason'] == 'insufficient_parameter_variation_or_points_for_free_parameters')


def test_budget_truncation_is_unknown_and_retains_actual_spent_without_mutating_sources(tmp_path):
    candidate, records = sources(tmp_path)
    original = deepcopy(records)
    result = learn(candidate, records, limit=26)
    assert result['status'] == 'UNKNOWN' and result['budget_exhausted']
    assert result['accepted'] == [] and 0 < result['actual_compute_cost'] <= 23
    assert records == original


def test_unit_mismatch_stays_unknown_instead_of_learning_numeric_cross_unit_affine_rule(tmp_path):
    candidate, records = sources(tmp_path)
    candidate['holes'][0]['output_unit'] = 's'
    candidate['holes'][1]['output_unit'] = 'V'
    for record in records:
        record['parameter_role_contracts']['old_coefficient']['coefficient_unit'] = 's'
        record['parameter_role_contracts']['new_coefficient']['coefficient_unit'] = 'V'
    result = learn(candidate, records)
    assert result['accepted'] == [] and result['status'] == 'UNKNOWN'
    assert result['attempts'][-1]['reason'] == 'incompatible_generated_unit'
    assert 'fitting' not in result['attempts'][-1]


def evidence_sources(tmp_path):
    candidate, records = sources(tmp_path, values=((1., 2., 3.), (1., 2., 3.)))
    name = 'measured_failure:control'
    for record in records:
        source = int(record['source_id'].rsplit('_', 1)[1])
        value = record['bindings']['old_coefficient'].value
        record['bindings']['new_coefficient'] = Program('const', value=2. * value + 3. * source + 1.)
        record['evidence_context'] = {'features': {name: float(source)},
            'role_signature': {name: {'role': 'measured_failure_control', 'unit': 'dimensionless',
                                      'time': 'before_revision'}},
            'source_refs': [{'file': record['after_ref']['file'],
                             'json_path': ['measured_failures', record['episode_id']]}],
            'selection': 'first_measured_parent_failure'}
    return candidate, records, name


def test_measured_condition_is_an_executable_input_and_after_constant_remains_target(tmp_path):
    candidate, records, name = evidence_sources(tmp_path)
    original = deepcopy(records)
    unconditional = deepcopy(records)
    for record in unconditional:
        record.pop('evidence_context')
    assert not learn(candidate, unconditional)['accepted']

    result = learn(candidate, records)
    assert records == original
    assert result['input_scope'] == 'before_parameters_and_measured_evidence'
    mapping = result['accepted'][0]
    assert mapping['evidence_inputs'] == [name]
    assert mapping['input_holes'] == ['old_coefficient']
    assert mapping['input_contracts'][name] == {
        'role': 'measured_failure_control', 'unit': 'dimensionless', 'time': 'before_revision',
        'selection': 'first_measured_parent_failure'}
    program = Program.from_dict(mapping['program'])
    assert abs(program.predict(Observation({'old_coefficient': 8., name: 1.}, 0.)) - 20.) < 1e-8
    assert abs(program.predict(Observation({'old_coefficient': 8., name: 0.}, 0.)) - 17.) < 1e-8
    assert any(attempt['evidence_inputs'] == [name] for attempt in result['attempts'])
    assert [check['expected_parameter'] for check in mapping['held_checks']] == [7., 10.]
    assert all(result['source_records'][check['source_record_index']]['evidence_context']['source_refs']
               for check in mapping['held_checks'])
    assert json.dumps(result).count('"failure_refs"') == len(records)


def test_changed_held_condition_rejects_map_without_using_it_as_parameter_label(tmp_path):
    candidate, records, name = evidence_sources(tmp_path)
    good = learn(candidate, records)
    records[-1]['evidence_context']['features'][name] = 0.
    bad = learn(candidate, records)
    assert bad['accepted'] == []
    checked = next(check for attempt in bad['attempts'] for check in attempt['checks']
                   if check['program'] == good['accepted'][0]['program'])
    assert all(row['status'] == 'passed' for row in checked['training'])
    assert checked['held'][-1]['expected_parameter'] == 10.
    assert abs(checked['held'][-1]['predicted_parameter'] - 7.) < 1e-8
    assert checked['held'][-1]['status'] == 'failed'
    assert bad['source_records'][checked['held'][-1]['source_record_index']]['evidence_context'][
        'features'][name] == 0.


@pytest.mark.parametrize('mismatch', ['missing_feature', 'unit', 'selection'])
def test_incomplete_or_conflicting_evidence_contract_is_not_filled_or_retyped(tmp_path, mismatch):
    candidate, records, name = evidence_sources(tmp_path)
    context = records[-1]['evidence_context']
    if mismatch == 'missing_feature':
        context['features'].pop(name)
    elif mismatch == 'unit':
        context['role_signature'][name]['unit'] = 'V'
    else:
        context['selection'] = 'last_measured_parent_failure'
    result = learn(candidate, records)
    assert result['accepted'] == []
    assert result['input_scope'] == 'before_parameters_only'
    assert result['excluded'][0]['feature'] == name
    assert all(not attempt['evidence_inputs'] for attempt in result['attempts'])


def wide_evidence_sources(tmp_path):
    """Thirteen actual metadata inputs, but only a sparse pair predicts the edit."""
    candidate, records = sources(tmp_path, values=((1., 2., 3., 4., 5.), (6., 7., 8., 9., 10.)))
    extra = [f'old_parameter_{index}' for index in range(1, 8)]
    candidate['parent_holes'].extend(extra)
    candidate['holes'].extend({'name': name, 'output_unit': 'dimensionless',
                               'time_contract': ['current']} for name in extra)
    evidence = [f'evidence:coordinate_{index}' for index in range(5)]
    rng = Random(7)
    for record in records:
        for name in extra:
            record['bindings'][name] = Program('const', value=float(rng.randrange(-4, 5)))
            record['parameter_role_contracts'][name] = {
                'role_key': {'fixture_coordinate': f'independent_nuisance_{name}'},
                'coordinate_scale': 1., 'coefficient_unit': 'dimensionless', 'status': 'known'}
        features = {name: float(rng.randrange(-4, 5)) for name in evidence}
        record['evidence_context'] = {'features': features,
            'role_signature': {name: {'role': 'measured_control', 'unit': 'dimensionless',
                                      'time': 'before_revision'} for name in evidence},
            'selection': 'first_measured_parent_failure',
            'source_refs': [{'file': record['after_ref']['file'],
                             'json_path': ['measured_failures', record['episode_id']]}]}
        record['bindings']['new_coefficient'] = Program('const', value=(
            2. * record['bindings']['old_coefficient'].value + 3. * features[evidence[0]] + 1.))
    return candidate, records, evidence[0]


def joint_learn(candidate, records, *, maximum=128, limit=20000):
    budget = SearchBudget(limit, spent=3)
    result = learn_joint_parameter_relations(candidate, records, budget, search_contract={
        **CONTRACT, 'operations': list(OPERATIONS), 'max_structures': maximum})
    assert result['actual_compute_cost'] == budget.spent - 3
    assert budget.spent <= budget.limit
    return result


def test_sparse_joint_frontier_fits_with_thirteen_inputs_and_preserves_old_frontier(tmp_path):
    candidate, records, evidence = wide_evidence_sources(tmp_path)
    original = deepcopy((candidate, records))
    old = learn_parameter_relations(candidate, records, SearchBudget(20000), search_contract={
        **CONTRACT, 'operations': list(OPERATIONS), 'max_structures': 64})
    old_joint = [item for item in old['attempts'] if item['input_holes'] and item['evidence_inputs']]
    assert len(old['attempts']) == 64 and old_joint
    assert all('fitting' not in item for item in old_joint)
    assert not old['accepted']

    result = joint_learn(candidate, records)
    assert (candidate, records) == original
    assert result['input_scope'] == 'sparse_parent_evidence_pairs'
    assert len(result['attempts']) == 128
    first_layer = result['attempts'][:40]
    assert all(item['scheduling_phase'] == 'refit' for item in first_layer)
    assert len({tuple(item['input_group']) for item in first_layer}) == 40
    assert all(item['input_holes'] and item['evidence_inputs'] for item in result['attempts'])
    stats = result['frontier']['outputs']['new_coefficient']
    assert stats['scheduling_phases'] == {'refit': 40, 'composition': 40, 'conditional': 40, 'remaining': 8}
    assert stats['skipped_non_joint_count'] > 0
    assert sum(item['count'] for item in stats['skipped']) == stats['skipped_non_joint_count']
    assert stats['limit_reached'] and not stats['complete']
    assert result['frontier_truncated'] and result['status'] == 'UNKNOWN'
    assert not result['budget_exhausted']
    mapping = next(item for item in result['accepted'] if item['input_holes'] == ['old_coefficient']
                   and item['evidence_inputs'] == [evidence])
    assert all(item['status'] == 'passed' for item in mapping['training_checks'] + mapping['held_checks'])
    assert not mapping['scientific_support']
    assert json.dumps(result).count('"failure_refs"') == len(records)


def test_sparse_joint_held_parameter_counterexample_rejects_unchanged_fitted_map(tmp_path):
    candidate, records, _ = wide_evidence_sources(tmp_path)
    good = joint_learn(candidate, records, maximum=40)
    program = good['accepted'][0]['program']
    value = records[-1]['bindings']['new_coefficient'].value
    records[-1]['bindings']['new_coefficient'] = Program('const', value=value + .25)
    bad = joint_learn(candidate, records, maximum=40)
    assert not bad['accepted']
    checks = next(item for attempt in bad['attempts'] for item in attempt['checks']
                  if item['program'] == program)
    assert all(item['status'] == 'passed' for item in checks['training'])
    assert checks['held'][-1]['status'] == 'failed'
    assert abs(checks['held'][-1]['residual']) > .24
    source = bad['source_records'][checks['held'][-1]['source_record_index']]
    assert source['after_ref'] == records[-1]['after_ref'] and source['failure_refs']


def test_sparse_joint_budget_cut_preserves_partial_work_and_actual_fee(tmp_path):
    candidate, records, _ = wide_evidence_sources(tmp_path)
    result = joint_learn(candidate, records, limit=500)
    assert result['status'] == 'UNKNOWN' and result['budget_exhausted']
    assert 0 < result['actual_compute_cost'] <= 497
    assert result['attempts'] and len(result['attempts']) < 128
    assert result['reason'] == 'parameter_relation_compute_budget_exhausted'
    assert not result['frontier']['outputs']['new_coefficient']['complete']
