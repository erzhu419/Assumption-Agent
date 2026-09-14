"""Directional transport must be observable while retaining total quantity."""

import json

import pytest

from hegel_machine.cognition.models import Probe
from hegel_machine.cognition.structure_lab import StructureLab
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_session import laboratory_for


def setup(mirror=1.):
    return {"q_o0": .1, "q_o1": .7, "q_o2": -.4,
            "x_o0": 0., "x_o1": .5 * mirror, "x_o2": -.5 * mirror,
            "u_o0": .02, "u_o1": -.03, "u_o2": .01, "focus": 0.}


@pytest.mark.parametrize("direction,source,target", [("right", 2, 1), ("left", 1, 2)])
def test_direction_selects_actual_transport_and_equal_opposite_quantity_changes(direction, source, target):
    lab = UnifiedObjectLab(objects=3, relation_radius=.8, relation_direction=direction)
    point = setup()
    result = lab.acquire(Probe("direction", (point,)), ("fine",))
    before, after = result.transitions[0]["before"], result.transitions[0]["after"]
    flows = [after[f"measured_q_o{i}"] - before[f"measured_q_o{i}"] - before[f"measured_u_o{i}"]
             for i in range(3)]
    incoming = lab._lab._coupling * before[f"measured_q_o{source}"]
    outgoing = lab._lab._coupling * before["measured_q_o0"]
    assert flows[0] == pytest.approx(incoming - outgoing)
    assert flows[source] == pytest.approx(-incoming)
    assert flows[target] == pytest.approx(outgoing)
    assert sum(flows) == pytest.approx(0., abs=1e-12)
    assert result.observations[0].target == after["measured_q_o0"]
    # Changing the selected object cannot change any physical quantity update.
    changed = lab.acquire(Probe("read_another_object", ({**point, "focus": 1.},)), ("fine",))
    assert all(changed.transitions[0]["after"][f"measured_q_o{i}"] == after[f"measured_q_o{i}"]
               for i in range(3))


def test_mirror_distinguishes_one_sided_relations_but_not_symmetric_exchange():
    probe = Probe("mirror", (setup(), setup(-1.)))
    batches = {direction: UnifiedObjectLab(objects=3, readout="event", relation_radius=.8,
        relation_direction=direction).acquire(probe, ("fine",))
        for direction in ("symmetric", "right", "left")}
    assert batches["symmetric"].observations[1].target == pytest.approx(0.)
    assert abs(batches["right"].observations[1].target) > .1
    assert batches["right"].observations[1].target == pytest.approx(-batches["left"].observations[1].target)
    assert all(batch.observations[0].target == 0. for batch in batches.values())
    assert all(batch.observation_cost == 2 and batch.measurement_cost == 18 for batch in batches.values())
    for direction in batches:
        total = UnifiedObjectLab(objects=3, readout="total", relation_radius=.8, relation_direction=direction)
        rows = total.observe(probe)
        assert rows[0].target == pytest.approx(rows[1].target)


@pytest.mark.parametrize("environment", ["objects", "anonymous_objects"])
def test_task_factory_direction_and_radius_survive_replay_and_terminal_without_public_disclosure(environment):
    settings = {"environment": environment, "seed": 317, "objects": 3, "apparatus_seed": 71,
                "readout": "window_total_event", "relation_radius": .8, "relation_direction": "right"}
    with laboratory_for(settings) as lab, laboratory_for({**settings, "relation_direction": "left"}) as other:
        assert lab.channels == other.channels
        assert all(lab.spec_for(view) == other.spec_for(view) for view in lab.measurement_views)
        probe = lab.random_probe(831, length=9)
        observed = lab.acquire(probe, ("fine",))
        original_costs = lab.costs.copy()
        assert lab.evaluation_rows(("fine",), seed=831, length=9, trajectories=1) == observed.observations
        assert lab.costs == original_costs
        saved = json.loads(json.dumps(lab.acquisitions))
        lab.restore(saved)
        assert lab.costs == original_costs
        assert lab.acquire(probe, ("fine",)) == observed
        assert any(abs(a.target - b.target) > 1e-8 for a, b in zip(observed.observations,
            other.observe(probe, ("fine",))))


def test_default_structure_keeps_existing_symmetric_updates_and_measurements():
    default = StructureLab(objects=3, seed=71)
    explicit = StructureLab(objects=3, seed=71, relation_direction="symmetric")
    probe = default.random_probe(837, length=9)
    expected = default.acquire(probe)
    assert explicit.acquire(probe) == expected
    for row, transition in zip(expected.observations, expected.transitions):
        before = transition["before"]
        for index in range(3):
            flow = default._coupling * sum(before[f"q_o{j}"] - before[f"q_o{index}"]
                for j in range(3) if j != index and abs(before[f"x_o{j}"] - before[f"x_o{index}"]) < default._radius)
            assert transition["after"][f"q_o{index}"] == before[f"q_o{index}"] + before[f"u_o{index}"] + flow
