"""Fixed-tree provenance and scope for the retained numerical audit."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Observation, SearchBudget


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/audit_retained_parameter_fits.py'
SPEC = importlib.util.spec_from_file_location('retained_fit_audit', SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def c(value):
    return Program('const', value=float(value))


def f(name):
    return Program('feature', name=name)


def node(op, *children):
    return Program(op, children=children)


def head(*terms):
    program = c(0)
    for term in terms:
        program = node('add', program, node('mul', c(1), term))
    return program


def test_robust_threshold_one_is_fixed_while_branch_zeros_and_head_are_free():
    initial = node('add', head(f('parent'), f('evidence')),
                   node('if', node('ge', f('evidence'), c(1)), c(0), c(0)))
    paths = audit.recover_parameter_paths(initial, 'robustification', 5, ['parent', 'evidence'])
    assert (1, 0, 1) not in paths
    assert (1, 1) in paths and (1, 2) in paths
    template = audit.AbstractTemplate('fixed', 'source', initial, (), (), paths)
    ridge = template.instantiate({}, [2., 3., 4., 5., 6.])
    report = {'search_completed': True, 'budget_exhausted': False,
        'parameter_scope': 'declared_const_slots_only', 'actual_fitted_rows': 8,
        'actual_compute_cost': 64}
    audit.validate_original_fit(initial, ridge, paths, report, 8)
    changed_guard = node('add', ridge.children[0],
                         node('if', node('ge', f('evidence'), c(0)), *ridge.children[1].children[1:]))
    with pytest.raises(audit.MissingFixedStructure, match='fixed_tree_or_threshold'):
        audit.validate_original_fit(initial, changed_guard, paths, report, 8)


def test_composition_recovers_outer_coefficients_without_freeing_the_interaction():
    initial = head(f('parent'), f('evidence'), node('mul', f('parent'), f('evidence')))
    paths = audit.recover_parameter_paths(initial, 'composition', 4, ['parent', 'evidence'])
    assert len(paths) == 4
    assert all(audit.at(initial, path).op == 'const' for path in paths)
    assert [audit.at(initial, path).value for path in paths] == [0., 1., 1., 1.]
    seeded = head(node('mul', c(2), f('parent')), f('evidence'))
    with pytest.raises(audit.MissingFixedStructure, match='supported_saved_factory'):
        audit.recover_parameter_paths(seeded, 'refit', 3, ['parent', 'evidence'])


def test_selected_matrix_preserves_guard_and_charges_zero_and_unit_predictions():
    initial = node('add', head(f('parent'), f('evidence')),
                   node('if', node('ge', f('evidence'), c(1)), c(0), c(0)))
    paths = audit.recover_parameter_paths(initial, 'robustification', 5, ['parent', 'evidence'])
    template = audit.AbstractTemplate('matrix', 'source', initial, (), (), paths)
    budget = SearchBudget(100)
    arrays = audit.pilot_matrix(template, [Observation({'parent': .5, 'evidence': 0.}, 7.),
        Observation({'parent': 2., 'evidence': 1.}, 9.)], budget)
    assert arrays['design'].tolist() == [[1., .5, 0., 0., 1.], [1., 2., 1., 1., 0.]]
    assert arrays['shifted_targets'].tolist() == [7., 9.]
    assert budget.spent == 12


def test_metadata_uses_exact_retained_before_after_and_keeps_after_out_of_features(tmp_path):
    path = tmp_path / 'sources.json'
    path.write_text(json.dumps({'revisions': [
        {'before': c(2).to_dict(), 'after': c(7).to_dict()},
        {'before': c(3).to_dict(), 'after': c(11).to_dict()}]}))
    records = []
    for i in range(2):
        records.append({'source_id': 'physical',
            'before_ref': audit.reference(path, 'revisions', i, 'before'),
            'after_ref': audit.reference(path, 'revisions', i, 'after'),
            'evidence_context': {'features': {'evidence:residual': i + .5}}})
    candidate = {'id': 'retained', 'before_pattern': Program('arg', name='before').to_dict(),
        'definition': Definition('after', ('before', 'after'), Program('arg', name='after')).to_dict(),
        'holes': [{'name': 'before'}, {'name': 'after'}], 'parent_holes': ['before']}
    search = {'candidate_id': 'retained', 'source_records': records,
              'split': {'training': [{'source_record_index': 0}], 'held': [{'source_record_index': 1}]}}
    original = deepcopy((search, candidate))
    scopes = audit.reconstruct_metadata({6: search}, {'retained': candidate},
                                        tmp_path / 'summary.json', SearchBudget(100))
    actual = scopes['6']['records']
    assert actual[0]['features'] == {'before': 2., 'evidence:residual': .5}
    assert actual[0]['targets'] == {'after': 7.}
    assert actual[1]['targets'] == {'after': 11.}
    assert (search, candidate) == original


@pytest.mark.parametrize('bad_held', [False, True])
def test_rank_unknown_does_not_erase_actual_train_and_held_outcomes(monkeypatch, tmp_path, bad_held):
    initial = head(f('parent'), f('evidence'))
    paths = audit.recover_parameter_paths(initial, 'refit', 3, ['parent', 'evidence'])
    template = audit.AbstractTemplate('case', 'source', initial, (), (), paths)
    ridge = template.instantiate({}, [0., 0., 0.])
    stable = template.instantiate({}, [1., 2., 0.])
    records = [{'source_record_index': i, 'features': {'parent': float(i), 'evidence': float(i)},
                'targets': {'after': 1. + 2. * i + (1. if bad_held and i == 3 else 0.)}}
               for i in range(4)]
    scope = {'training_indices': [0, 1, 2], 'held_indices': [3], 'records': records}
    checks = []
    for program in (initial, ridge):
        check = {'program': program.to_dict()}
        for label, indices in (('training', [0, 1, 2]), ('held', [3])):
            check[label] = [{'source_record_index': i, 'expected_parameter': records[i]['targets']['after'],
                             'residual': -records[i]['targets']['after']} for i in indices]
        checks.append(check)
    fitting = {'search_completed': True, 'budget_exhausted': False,
        'parameter_scope': 'declared_const_slots_only', 'actual_fitted_rows': 3,
        'actual_compute_cost': 18}
    search = {'candidate_id': 'source', 'contract': {'tolerance': 1e-10}, 'attempts': [{
        'checks': checks, 'fitting': fitting, 'operation': 'refit', 'output_hole': 'after',
        'input_group': ['parent', 'evidence']}]}

    def fake_fit(template, mapping, rows, budget, q, *, report):
        assert [row.target for row in rows] == [1., 3., 5.]
        assert all(set(row.features) == {'parent', 'evidence'} for row in rows)
        assert budget.spend(21)
        report.update(status='UNKNOWN', parameter_identifiability='UNKNOWN',
            candidate_methods=['initial', 'ridge_lstsq', 'stable_lstsq'],
            linear_diagnostics={'status': 'computed', 'rank_deficient': True,
                'train_residual_max_abs': 0., 'linear_l2_residual_sum_squares': 0.},
            losses={stable: 0.})
        return [initial, ridge, stable]

    monkeypatch.setattr(audit, 'fit_parameter_slots', fake_fit)
    budget = SearchBudget(100)
    result = audit.audit_case(6, 0, search, scope, paths, tmp_path / 'summary.json', budget)
    assert result['parameter_identifiability'] == 'UNKNOWN'
    assert result['stable_training_pass'] is True
    assert result['stable_held_pass'] is (not bad_held)
    assert result['observed_outcome'] == ('training_pass_held_failed' if bad_held else 'training_and_held_pass')
    assert result['new_compute_cost'] == 22 == budget.spent
    assert 'losses' not in result['fitting_report']
    assert result['ridge_reproduced_exactly']
    assert not result['scientific_support_added']
