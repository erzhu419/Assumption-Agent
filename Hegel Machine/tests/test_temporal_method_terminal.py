"""A terminal activation remains an actual legal experiment."""

from random import Random

import pytest

from hegel_machine.cognition.models import Probe
from hegel_machine.cognition.temporal_method_lab import TemporalMethodLab


def test_registered_event_target_repairs_unreachable_random_prefix():
    lab = TemporalMethodLab("event", 4, 1376, center=.37, response_gain=1.)
    before = dict(lab.costs)
    rows = lab.evaluation_rows(seed=1000011, trajectories=4, length=32)
    assert len(rows) == 128 and lab.costs == before
    assert all(-1. <= row.features["u"] <= 1. for row in rows)
    assert all(-1. <= past["u"] <= 1. for row in rows for past in row.history)
    for offset in (32, 96):
        assert rows[offset + 2].features["meter"] == pytest.approx(.37, abs=1e-12)
        assert rows[offset + 2].target == 1.
        assert rows[offset + 3].target == -1.


def test_already_reachable_terminal_trajectories_are_unchanged():
    lab = TemporalMethodLab("event", 0, 1154, center=.21, response_gain=2.)
    rng, expected = Random(1000011), []
    delay, gain = lab._parameters()
    for trajectory in range(4):
        controls = [rng.uniform(-1., 1.) for _ in range(32)]
        if trajectory % 2:
            controls = [rng.uniform(-.1, .1) for _ in range(32)]
            index = delay + 1
            past = controls[index - delay]
            controls[index] = (.21 - gain * past - .15 * past ** 2) / .4
            assert -1. <= controls[index] <= 1.
            controls[index + 1] = 0.
        expected.extend(lab.acquire(Probe(f"old:{trajectory}", tuple({"u": value}
            for value in controls)), ("meter",)).observations)
    assert lab.evaluation_rows(seed=1000011, trajectories=4, length=32) == expected
