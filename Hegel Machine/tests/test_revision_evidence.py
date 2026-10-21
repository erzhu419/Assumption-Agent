"""Shared condition inputs use cited before predictions and physical units."""

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.revision_evidence import evidence_context
from hegel_machine.cognition.revision_relations import _check_generated_units, _MissingContract


ROLES = {'u': {'role': 'drive', 'unit': 'V', 'time': 'current'}}


def example(target, index=0):
    return {'source_ref': {'file': 'retained_paid.json', 'json_path': [index]},
        'observation': {'features': {'u': 2.}, 'target': target, 'history': [{'u': 1.}]},
        'tolerance': 1e-5, 'criterion': {'evaluated_rows': 97}}


def test_source_and_target_read_same_first_measured_failure_from_before_only():
    parent = Program('feature', name='u')
    examples = [example(2.), example(5., 1)]
    measured = [{'status': 'computed', 'failed': False, 'prediction': 2.},
                {'status': 'computed', 'failed': True, 'prediction': 2.}]
    first, second = SearchBudget(20), SearchBudget(20)
    source = evidence_context(parent, examples, ROLES, 'V', measured, first)
    target = evidence_context(parent, examples, ROLES, 'V', measured, second)
    assert source == target
    assert first.spent == second.spent == 7  # Two examined rows plus five inputs.
    assert source['source_refs'] == [examples[1]['source_ref']]
    assert source['features']['evidence:residual'] == 3.
    assert source['features']['evidence:prediction'] == 2.
    assert source['features']['evidence:history_depth'] == 1
    assert source['features']['evidence:paid_rows'] == 97
    assert all('after' not in name and 'source_id' not in name for name in source['features'])


def test_absent_or_agreeing_measurement_cannot_supply_a_failure_condition():
    parent = Program('feature', name='u')
    budget = SearchBudget(20)
    result = evidence_context(parent, [example(2.)], ROLES, 'V', [], budget)
    assert result['status'] == 'UNKNOWN' and not result['features']
    assert budget.spent == 2  # Examined row plus ordinary before prediction.


def test_typed_threshold_inherits_comparison_unit_without_allowing_numeric_unit_change():
    feature = Program('feature', name='residual')
    roles = {'residual': {'role': 'signed_residual', 'unit': 'V', 'time': 'before_revision'}}
    threshold = Program('ge', children=(feature, Program('const', value=.5)))
    assert _check_generated_units(threshold, 'dimensionless', roles, {}, SearchBudget(20)) == 'dimensionless'
    with pytest.raises(_MissingContract, match='incompatible_generated_unit'):
        _check_generated_units(feature, 'dimensionless', roles, {}, SearchBudget(20))
