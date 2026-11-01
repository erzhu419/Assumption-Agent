"""A completed paid partition exposes its precision result in the same return."""

from dataclasses import replace

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
from test_cognition_relation_partition_reuse import law, rows


def single_research():
    return RelationIntervalResearch(law(.6), [], SearchBudget(10000),
        structure={'fits': [{'radius': .6}], 'radius_interval': [.5, .7]})


@pytest.mark.parametrize('half_width,expected', [
    (2e-7, 'resolved_to_precision'), (.01, 'unresolved')])
def test_newly_measured_bracket_reports_precision_without_a_second_pass(half_width, expected):
    research = single_research()
    paid = rows(law(.6), (.1, .6 - half_width, .6 + half_width, 1.2) * 2)
    entries = research.alternatives(paid, SearchBudget(10000), experiments=[paid])
    assert entries
    assert research.parameters[0]['radius_intervals'] == [[.6 - half_width, .6 + half_width]]
    assert all(entry['validation_loss'] == 0. for entry in entries)
    assert research.status == expected
    research.adopt(next(entry for entry in entries if entry['candidate'].program == law(.6)))
    assert research.status == expected


def test_several_precise_supported_cells_remain_unresolved():
    research = single_research()
    research.parameters[0]['radius_intervals'] = [[.59, .5900004], [.61, .6100004]]
    paid = rows(law(.6), (.1, 1.2) * 4)
    entries = research.alternatives(paid, SearchBudget(10000))
    assert entries and len(research.parameters[0]['radius_intervals']) == 2
    assert research.status == 'unresolved'


def two_parameter_research(*, left_precise=False, right_pending=False):
    def symbolic(node, symbol):
        if node.op == 'neighbor_sum':
            return replace(node, children=(Program('arg', name=symbol), *node.children[1:]))
        return replace(node, children=tuple(symbolic(child, symbol) for child in node.children))

    left = law(.6)
    right = law(.8).remap_features({name: 'right_' + name for name in left.required_features()})
    current = Program('add', children=(left, right))
    template = Program('add', children=(symbolic(left, 'left_radius'), symbolic(right, 'right_radius')))
    parameters = [
        {'symbol': 'left_radius', 'radius': .6, 'source_concept_id': None,
         'radius_intervals': [[.5999998, .6000002]] if left_precise else [[.5, .7]]},
        {'symbol': 'right_radius', 'radius': .8, 'source_concept_id': None,
         'radius_intervals': [[.7999998, .8000002]] if right_pending else [[.7, .9]]}]
    if right_pending:
        parameters[1].update(kernel_prior_status='pending', kernel_priors=[{
            'source_concept_id': 'right_source', 'radius_intervals': [[.7999998, .8000002]]}])
    saved = {'selection_steps': [{'operation': 'resolve_relation_intervals',
        'parameter_template': template.to_dict(), 'parameters': parameters}]}
    research = RelationIntervalResearch(current, [], SearchBudget(10000), research_program=saved)
    paid, history = [], []
    for index, row in enumerate(rows(left, (.1, .5999998, .6000002, 1.2) * 2)):
        right_features = {**row.features, 'x1': (.1, 1.2)[index % 2]}
        fields = {**row.features, **{'right_' + name: value for name, value in right_features.items()}}
        measured = Observation(fields, 0., tuple(history))
        paid.append(replace(measured, target=current.predict(measured)))
        history.append(fields)
    return research, paid


@pytest.mark.parametrize('right_pending', [False, True])
def test_finishing_one_parameter_does_not_finish_a_wide_or_pending_second_parameter(right_pending):
    research, paid = two_parameter_research(right_pending=right_pending)
    entries = research.alternatives(paid, SearchBudget(20000), parameter_index=0)
    assert entries
    assert research.parameters[0]['radius_intervals'] == [[.5999998, .6000002]]
    assert research.status == 'unresolved'


def test_requesting_an_already_precise_parameter_does_not_resolve_the_untouched_parameter():
    research, paid = two_parameter_research(left_precise=True)
    assert research.alternatives(paid, SearchBudget(20000), parameter_index=0) == []
    assert research.status == 'unresolved'


def test_unfinished_partition_is_not_reported_as_precise():
    research = single_research()
    paid = rows(law(.6), (.1, .5999998, .6000002, 1.2) * 2)
    assert research.alternatives(paid, SearchBudget(0)) == []
    assert research.status == 'compute_exhausted'
