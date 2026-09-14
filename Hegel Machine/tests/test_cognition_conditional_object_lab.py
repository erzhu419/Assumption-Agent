"""Quantity conditions change paid transport observations while conserving mass."""

import json

import pytest

from hegel_machine.cognition.models import Probe
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_session import laboratory_for


def fluxes(before, direction, condition, radius, coupling):
    q = [before[f"measured_q_o{i}"] for i in range(3)]
    x = [before[f"measured_x_o{i}"] for i in range(3)]
    sign = 1 if direction == "right" else -1
    permitted = lambda source, target: (q[source] > q[target] if condition == "source_greater"
                                         else q[source] < q[target])
    # An independent edge matrix: column sums are incoming quantities and row
    # sums are outgoing quantities. Source quantities may be negative.
    edges = [[q[source] if 0 < sign * (x[target] - x[source]) < radius
              and permitted(source, target) else 0. for target in range(3)] for source in range(3)]
    return [coupling * (sum(edges[source][i] for source in range(3)) - sum(edges[i])) for i in range(3)]


@pytest.mark.parametrize("direction", ["right", "left"])
@pytest.mark.parametrize("condition", ["source_greater", "source_less"])
def test_paid_transitions_match_conditional_transport_and_conserve_signed_quantities(direction, condition):
    lab = UnifiedObjectLab(objects=3, readout="event", relation_radius=.8,
        relation_direction=direction, relation_condition=condition)
    point = {"q_o0": -.7, "q_o1": .4, "q_o2": -.2,
             "x_o0": -.6, "x_o1": 0., "x_o2": .6,
             "u_o0": .03, "u_o1": -.01, "u_o2": .02, "focus": 1.}
    probe = Probe("change_focus_only", (point, {**point, "focus": 0.}))
    batch = lab.acquire(probe, ("fine",))
    expected = fluxes(batch.transitions[0]["before"], direction, condition, .8, lab._lab._coupling)
    assert any(abs(value) > .01 for value in expected)
    for transition in batch.transitions:
        before, after = transition["before"], transition["after"]
        actual = [after[f"measured_q_o{i}"] - before[f"measured_q_o{i}"] - before[f"measured_u_o{i}"]
                  for i in range(3)]
        assert actual == pytest.approx(expected)
        assert sum(actual) == pytest.approx(0., abs=1e-12)
    assert batch.observations[0].target == 0.
    assert batch.observations[1].target == pytest.approx(expected[0] - expected[1])
    assert batch.observation_cost == 2 and batch.measurement_cost == 18


@pytest.mark.parametrize("condition,first_open", [("source_greater", True), ("source_less", False)])
def test_fixed_positions_and_swapped_quantities_switch_the_edge_condition(condition, first_open):
    lab = UnifiedObjectLab(objects=2, relation_radius=.8,
        relation_direction="right", relation_condition=condition)
    point = {"q_o0": .7, "q_o1": -.3, "x_o0": 0., "x_o1": .4,
             "u_o0": 0., "u_o1": 0., "focus": 0.}
    probe = Probe("swap_quantities", (point, {**point, "q_o0": -.3, "q_o1": .7},
                                       {**point, "q_o0": .2, "q_o1": .2}))
    batch = lab.acquire(probe, ("fine",))
    outgoing = [row.features["measured_q_o0"] - row.target for row in batch.observations]
    first = lab._lab._coupling * batch.observations[0].features["measured_q_o0"]
    second = lab._lab._coupling * batch.observations[1].features["measured_q_o0"]
    assert outgoing == pytest.approx([first if first_open else 0., 0. if first_open else second, 0.])


@pytest.mark.parametrize("direction", ["symmetric", "right", "left"])
def test_always_is_the_existing_default_and_symmetric_ignores_transport_conditions(direction):
    default = UnifiedObjectLab(objects=3, relation_direction=direction)
    always = UnifiedObjectLab(objects=3, relation_direction=direction, relation_condition="always")
    probe = default.random_probe(137, length=8)
    assert default.acquire(probe, ("fine",)) == always.acquire(probe, ("fine",))
    if direction == "symmetric":
        for condition in ("source_greater", "source_less"):
            conditioned = UnifiedObjectLab(objects=3, relation_condition=condition)
            assert conditioned.acquire(probe, ("fine",)) == always.acquire(probe, ("fine",))


@pytest.mark.parametrize("environment", ["objects", "anonymous_objects"])
def test_condition_is_private_and_survives_restore_and_independent_evaluation(environment):
    settings = {"environment": environment, "objects": 3, "apparatus_seed": 213,
        "readout": "window_total_event", "relation_direction": "left", "relation_radius": .8,
        "relation_condition": "source_greater"}
    with laboratory_for(settings) as lab, laboratory_for({**settings, "relation_condition": "source_less"}) as other:
        assert lab.channels == other.channels
        assert all(lab.spec_for(view) == other.spec_for(view) for view in lab.measurement_views)
        probe = lab.random_probe(713, length=10)
        batch = lab.acquire(probe, ("fine",))
        costs = lab.costs.copy()
        assert lab.evaluation_rows(("fine",), seed=713, length=10, trajectories=1) == batch.observations
        assert lab.costs == costs
        saved = json.loads(json.dumps(lab.acquisitions))
        lab.restore(saved)
        assert lab.costs == costs
        assert lab.acquire(probe, ("fine",)) == batch
        assert any(abs(a.target - b.target) > 1e-8 for a, b in zip(
            batch.observations, other.observe(probe, ("fine",))))
