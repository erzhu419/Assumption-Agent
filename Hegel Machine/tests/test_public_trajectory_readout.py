"""Construct public trajectory vectors without opening historical data."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope='module')
def builder():
    root = Path(__file__).resolve().parents[1]
    specification = importlib.util.spec_from_file_location('public_trajectory_readout',
        root / 'scripts/replay_retained_forecast_routing.py')
    module = importlib.util.module_from_spec(specification)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(root / 'scripts'))
        specification.loader.exec_module(module)
    return module


def trajectory(*values):
    return {'kind': 'trajectory', 'channels': [], 'sequence': [{'u': value} for value in values]}


def test_explicit_source_contract_preserves_two_reset_step_and_history_order(builder):
    actions = [trajectory(.1, .2, .3), trajectory(-.1, -.2)]
    context = {'episode': 'current_source', 'output_unit': 'response_A'}
    query = builder.full_panel_readout(actions, 7, context=context, units='response_A')
    assert query['op'] == 'Readout' and query['statistic_ref'] == {'kind': 'vector'}
    rows = query['finite_readouts']
    assert [(row['record_index'], row['step']) for row in rows] == [(7, 0), (7, 1), (7, 2), (8, 0), (8, 1)]
    assert [row['binding']['features'] for row in rows] == [{'u': value} for value in (.1, .2, .3, -.1, -.2)]
    assert [row['binding']['history'] for row in rows] == [[], [{'u': .1}],
        [{'u': .1}, {'u': .2}], [], [{'u': -.1}]]
    assert all(row['context_ref'] == context and row['units'] == 'response_A' for row in rows)
    assert [row['binding']['action'] for row in rows] == [actions[0]] * 3 + [actions[1]] * 2


def test_four_declared_twelve_step_trajectories_keep_all_48_coordinates(builder):
    actions = [trajectory(*[offset + step / 100 for step in range(12)]) for offset in (0., .2, -.2, .4)]
    rows = builder.full_panel_readout(actions, 24, context={'source': 'new'}, units='new_unit')['finite_readouts']
    assert len(rows) == 48
    assert [(row['record_index'], row['step']) for row in rows] == [
        (24 + index, step) for index in range(4) for step in range(12)]
    assert all(rows[index * 12]['binding']['history'] == [] for index in range(4))
    assert all(rows[index * 12 + 11]['binding']['history'] == action['sequence'][:11]
               for index, action in enumerate(actions))


def test_source_context_and_units_do_not_leak_between_calls_or_into_legacy_defaults(builder):
    actions = [trajectory(.5), trajectory(-.5)]
    first = builder.full_panel_readout(actions, 0, context={'episode': 'A'}, units='unit_A')
    second = builder.full_panel_readout(actions, 0, context={'episode': 'B'}, units='unit_B')
    legacy = builder.full_panel_readout(actions, 0)
    for query, context, units in [(first, {'episode': 'A'}, 'unit_A'),
                                   (second, {'episode': 'B'}, 'unit_B'),
                                   (legacy, builder.CONTEXT, builder.UNIT)]:
        assert all(row['context_ref'] == context and row['units'] == units for row in query['finite_readouts'])


def test_construction_and_result_edits_do_not_modify_input_actions_or_context(builder):
    actions, context = [trajectory(.1, .2), trajectory(-.1, -.2)], {'episode': {'id': 'A'}}
    before = deepcopy((actions, context))
    query = builder.full_panel_readout(actions, 3, context=context, units='response')
    assert (actions, context) == before
    row = query['finite_readouts'][1]
    row['binding']['action']['sequence'][0]['u'] = 999.
    row['binding']['features']['u'] = 999.
    row['binding']['history'][0]['u'] = 999.
    row['context_ref']['episode']['id'] = 'changed'
    assert (actions, context) == before
    assert query['finite_readouts'][0]['context_ref'] == context
