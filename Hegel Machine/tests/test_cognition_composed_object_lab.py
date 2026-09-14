import json

import pytest

from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab


COMPOSED_READOUTS = (
    "total_event", "focus_window", "filtered_total", "window_total_event",
)


@pytest.mark.parametrize("readout", COMPOSED_READOUTS)
def test_composed_response_matches_independently_observed_components_and_resets(readout):
    lab = UnifiedObjectLab(readout=readout, apparatus_seed=73, seed=900)
    probe = lab.random_probe(503, length=9)
    components = {
        name: UnifiedObjectLab(readout=name, apparatus_seed=73).observe(probe, ("fine",))
        for name in ("total", "focus", "event", "window")
    }
    expected, filtered, combined = [], 0.0, []
    for index in range(probe.cost):
        total, focus, event, window = (components[name][index].target
                                      for name in ("total", "focus", "event", "window"))
        combined.append(total + .7 * event)
        filtered = .8 * filtered + total
        expected.append({"total_event": combined[-1],
                         "focus_window": focus + .6 * window,
                         "filtered_total": filtered,
                         "window_total_event": sum(combined[-3:])}[readout])
    assert components["event"][0].target == 0.0
    rows = lab.observe(probe, ("fine",))
    assert [row.target for row in rows] == pytest.approx(expected)
    assert not rows[0].history
    assert rows[1].history[0]["__target__"] == expected[0]
    assert lab.observe(probe, ("fine",)) == rows


@pytest.mark.parametrize("readout", COMPOSED_READOUTS)
def test_composed_response_has_the_same_target_at_each_measurement_scale(readout):
    lab = UnifiedObjectLab(readout=readout, apparatus_seed=83, rename_prefix="other_")
    probe = lab.random_probe(1205, length=7)
    commands = lab.acquire(probe)
    fine = lab.acquire(probe, ("fine",))
    before = dict(lab.costs)
    projected = lab.project_rows(fine.observations, ("fine",), ("coarse",))
    projected_transitions = lab.project_transitions(fine.transitions, ("fine",), ("coarse",))
    assert lab.costs == before
    coarse = lab.acquire(probe, ("coarse",))
    assert projected == coarse.observations
    assert projected_transitions == coarse.transitions
    assert [row.target for row in commands.observations] == [row.target for row in fine.observations]
    assert fine.measurement_cost == 4 * coarse.measurement_cost


def test_response_choice_and_apparatus_seed_do_not_change_the_public_interface():
    reference = UnifiedObjectLab(readout="focus")
    for readout in COMPOSED_READOUTS:
        lab = UnifiedObjectLab(readout=readout, seed=101, apparatus_seed=307)
        assert lab.seed == 101
        assert reference.channels == lab.channels
        for channels in lab.measurement_views:
            assert reference.spec_for(channels) == lab.spec_for(channels)


def test_fixed_apparatus_seed_survives_task_changes_evaluation_and_restore():
    first = UnifiedObjectLab(readout="filtered_total", seed=2, apparatus_seed=311,
                             profile="confirmation")
    second = UnifiedObjectLab(readout="filtered_total", seed=1009, apparatus_seed=311,
                              profile="confirmation")
    changed = UnifiedObjectLab(readout="filtered_total", seed=2, apparatus_seed=313,
                               profile="confirmation")
    probe = first.random_probe(711, length=8)
    original = first.acquire(probe, ("fine",))
    assert original == second.acquire(probe, ("fine",))
    assert original.observations != changed.observe(probe, ("fine",))
    before = dict(first.costs)
    assert first.evaluation_rows(("fine",), seed=711, length=8, trajectories=1) == original.observations
    assert first.costs == before
    first.restore(json.loads(json.dumps(first.acquisitions)))
    assert first.costs == before
    assert first.acquire(probe, ("fine",)) == original


def test_omitting_apparatus_seed_preserves_the_existing_seed_behavior():
    for profile in ("development", "confirmation"):
        original = UnifiedObjectLab(seed=103, readout="window", profile=profile)
        explicit = UnifiedObjectLab(seed=999, apparatus_seed=103, readout="window", profile=profile)
        probe = original.random_probe(981, length=6)
        assert original.acquire(probe, ("fine",)) == explicit.acquire(probe, ("fine",))


def test_changed_actuator_calibration_preserves_the_public_task_and_interaction_structure():
    original = UnifiedObjectLab(readout="total_event", apparatus_seed=9013)
    changed = UnifiedObjectLab(readout="total_event", apparatus_seed=9013, actuator_curvature=.31)
    assert original.channels == changed.channels
    for channels in original.measurement_views:
        assert original.spec_for(channels) == changed.spec_for(channels)
    probe = original.random_probe(1217, length=7)
    before = original.acquire(probe, ("fine",))
    after = changed.acquire(probe, ("fine",))
    assert before.observations != after.observations
    for point, row in zip(probe.sequence, after.observations):
        for index in range(4):
            command = point[f"q_o{index}"]
            assert row.features[f"measured_q_o{index}"] == pytest.approx(command + .31 * command ** 3)
    assert changed.project_rows(after.observations, ("fine",), ("coarse",)) == changed.observe(probe, ("coarse",))


def test_explicit_calibration_survives_evaluation_and_restore():
    lab = UnifiedObjectLab(readout="window_total_event", apparatus_seed=9013,
                           actuator_curvature=.31, rename_prefix="shifted_")
    probe = lab.random_probe(1709, length=8)
    original = lab.acquire(probe, ("fine",))
    before = dict(lab.costs)
    assert lab.evaluation_rows(("fine",), seed=1709, length=8, trajectories=1) == original.observations
    assert lab.costs == before
    acquisitions = json.loads(json.dumps(lab.acquisitions))
    lab.restore(acquisitions)
    assert lab.costs == before
    assert lab.acquire(probe, ("fine",)) == original


def test_relation_radius_changes_transitions_without_changing_sensors_or_total():
    original = UnifiedObjectLab(readout="total_event", apparatus_seed=9013, relation_radius=.6)
    changed = UnifiedObjectLab(readout="total_event", apparatus_seed=9013, relation_radius=1.35)
    assert original.channels == changed.channels
    for channels in original.measurement_views:
        assert original.spec_for(channels) == changed.spec_for(channels)
    probe = original.random_probe(1217, length=24)
    before = original.acquire(probe, ("fine",))
    after = changed.acquire(probe, ("fine",))
    assert [row.features for row in before.observations] == [row.features for row in after.observations]
    assert [item["before"] for item in before.transitions] == [item["before"] for item in after.transitions]
    assert [item["after"] for item in before.transitions] != [item["after"] for item in after.transitions]
    assert [row.target for row in before.observations] != [row.target for row in after.observations]
    totals = [UnifiedObjectLab(readout="total", apparatus_seed=9013, relation_radius=radius)
              .observe(probe) for radius in (.6, 1.35)]
    assert [row.target for row in totals[0]] == pytest.approx([row.target for row in totals[1]])


def test_changed_relation_composes_and_survives_evaluation_and_restore():
    settings = dict(apparatus_seed=9013, relation_radius=1.35)
    lab = UnifiedObjectLab(readout="window_total_event", **settings)
    probe = lab.random_probe(1709, length=8)
    totals = UnifiedObjectLab(readout="total", **settings).observe(probe)
    events = UnifiedObjectLab(readout="event", **settings).observe(probe)
    combined = [total.target + .7 * event.target for total, event in zip(totals, events)]
    original = lab.acquire(probe, ("fine",))
    assert [row.target for row in original.observations] == pytest.approx(
        [sum(combined[max(0, index - 2):index + 1]) for index in range(len(combined))])
    before = dict(lab.costs)
    assert lab.evaluation_rows(("fine",), seed=1709, length=8, trajectories=1) == original.observations
    assert lab.costs == before
    lab.restore(json.loads(json.dumps(lab.acquisitions)))
    assert lab.costs == before
    assert lab.acquire(probe, ("fine",)) == original
