"""Comparison thresholds retain declared operand units during source preparation."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.revision_relations import (
    _MissingContract, _prepare, consolidate_revision_relations,
)
from test_revision_relations import binary, constant, edit_sources, feature, revision


@pytest.mark.parametrize('unit', ['1', 'V'])
def test_retained_first_pair_threshold_path_inherits_the_actual_feature_unit(unit):
    # Preserve the first failing historical guard and its original child path;
    # irrelevant sibling sums are omitted from this static contract fixture.
    threshold_path = (0, 0, 1, 1, 0, 1)
    for threshold, body in ((-.25, feature()),
                            (-.55, Program('window_sum', lag=2, children=(feature(),)))):
        guarded = Program('if', children=(binary('ge', feature(), constant(threshold)), body, constant(0)))
        program = binary('add', binary('add', binary('add', constant(.5),
            binary('mul', constant(1.2), guarded)), constant(0)), constant(0))
        source = revision('typed', program,
            role_signature={'u': {'role': 'public_control', 'unit': unit, 'time': 'current'}}, output_unit=unit)
        original = deepcopy(source)
        budget = SearchBudget(1000)
        prepared = _prepare(source, program, budget)
        assert prepared['contracts'][threshold_path]['output_unit'] == unit
        assert prepared['contracts'][threshold_path]['time_contract'] == ['current']
        assert prepared['contracts'][threshold_path[:-1]]['output_unit'] == 'dimensionless'
        assert source == original and budget.spent > program.size


@pytest.mark.parametrize('conflict', ['second_feature', 'declared_threshold'])
def test_comparison_operand_unit_conflicts_are_not_hidden_by_threshold_propagation(conflict):
    right = feature('v') if conflict == 'second_feature' else constant(.5)
    program = Program('if', children=(binary('ge', feature(), right), feature(), constant(0)))
    source = revision('conflict', program, role_signature={
        'u': {'role': 'drive', 'unit': 'V', 'time': 'current'},
        'v': {'role': 'current', 'unit': 'A', 'time': 'current'}})
    if conflict == 'declared_threshold':
        source['subtree_contracts'] = {'0.1': {'output_unit': 'A'}}
    with pytest.raises(_MissingContract, match='incompatible_comparison_operand_units'):
        _prepare(source, program, SearchBudget(1000))


def test_two_unknown_threshold_units_do_not_become_dimensionless_or_a_revision_answer():
    records = edit_sources()
    for index, record in enumerate(records):
        parent = Program.from_dict(record['before_program'])
        after = Program('if', children=(binary('lt', constant(0), constant(index + 1)), parent, constant(0)))
        record['after_program'] = after.to_dict()
        prepared = _prepare(record, after, SearchBudget(1000))
        assert prepared['contracts'][(0, 0)]['output_unit'] is None
        assert prepared['contracts'][(0, 1)]['output_unit'] is None
    result = consolidate_revision_relations(records, object_type='program_revision_relation')
    assert result['status'] == 'UNKNOWN' and result['candidates'] == []
    assert result['excluded'][0]['reason'] == 'missing_subtree_output_unit_or_time'


def test_undeclared_compound_operand_does_not_get_a_unit_from_the_comparison_result():
    opaque = binary('mul', feature(), feature('v'))
    program = Program('if', children=(binary('ge', opaque, constant(.5)), feature(), constant(0)))
    source = revision('unknown_product', program, role_signature={
        'u': {'role': 'voltage', 'unit': 'V', 'time': 'current'},
        'v': {'role': 'current', 'unit': 'A', 'time': 'current'}})
    prepared = _prepare(source, program, SearchBudget(1000))
    assert prepared['contracts'][(0, 0)]['output_unit'] is None
    assert prepared['contracts'][(0, 1)]['output_unit'] is None
