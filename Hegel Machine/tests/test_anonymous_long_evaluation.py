"""Frozen evaluator panels preserve the public controls and reset semantics."""

from random import Random

import pytest

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab, CONTROL_LEVELS


def sequence(*values):
    return [{'u': value} for value in values]


def test_explicit_long_panels_reset_and_do_not_spend_query_budget():
    lab = AnonymousHistoryLab('hysteresis', seed=31)
    high_hold = sequence(1., *([0.] * 47))
    reset_hold = sequence(*([0.] * 48))
    before = lab.costs
    rows = lab.evaluation_sequences([high_hold, reset_hold, high_hold])
    assert len(rows) == 144
    assert rows[:48] == rows[96:]
    assert rows[48:96] == lab.evaluation_sequences([reset_hold])
    assert rows[47].target == 1.
    assert rows[0].history == rows[48].history == rows[96].history == ()
    assert len(rows[47].history) == len(rows[95].history) == 47
    assert all(set(row.features) == {'u'} for row in rows)
    assert all(set(point) == {'u'} for row in rows for point in row.history)
    assert lab.costs == before


@pytest.mark.parametrize('values', [[0.] * 49, [.15], []])
def test_evaluator_rejects_unpublished_inputs(values):
    lab = AnonymousHistoryLab()
    with pytest.raises(ValueError):
        lab.evaluation_sequences([sequence(*values)])
    assert lab.costs == {'observation_cost': 0, 'measurement_cost': 0, 'queries': 0}


def test_long_evaluation_does_not_expand_ordinary_query_limit():
    lab = AnonymousHistoryLab()
    action = {'kind': 'trajectory', 'sequence': sequence(*([0.] * 25)), 'channels': []}
    with pytest.raises(ValueError, match='1 to 24'):
        lab.execute(action)
    assert lab.costs['observation_cost'] == 0
    assert len(lab.evaluation_sequences([action['sequence']])) == 25


@pytest.mark.parametrize('kind', ['memoryless', 'delay', 'hysteresis'])
def test_explicit_panel_preserves_seeded_terminal_readout(kind):
    lab, rng = AnonymousHistoryLab(kind, seed=31), Random(733)
    controls = [{'u': rng.choice(CONTROL_LEVELS)} for _ in range(110)]
    explicit = lab.evaluation_sequences([controls[:48], controls[48:96], controls[96:]])
    assert explicit == lab.evaluation_rows(length=110, seed=733)
