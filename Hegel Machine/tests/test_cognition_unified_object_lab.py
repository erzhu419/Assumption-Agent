import json

import pytest

from hegel_machine.cognition.models import Probe
from hegel_machine.cognition.structure_lab import ObjectSchema
from hegel_machine.cognition.unified_object_lab import ObjectMeasurementBatch, UnifiedObjectLab


def test_command_view_does_not_supply_unpaid_actual_quantities():
    lab = UnifiedObjectLab(readout="total")
    point = {name: .8 if name.startswith("q_") else 0.
             for name in lab.spec.feature_bounds}
    probe = Probe("same_setup", (point, point))
    commands = lab.acquire(probe)
    coarse = lab.acquire(probe, ("coarse",))
    fine = lab.acquire(probe, ("fine",))
    assert commands.observations[0].features == point
    assert fine.observations[0].features["measured_q_o0"] > point["q_o0"]
    assert commands.observations[0].target == coarse.observations[0].target == fine.observations[0].target
    assert commands.measurement_cost == 0
    assert coarse.measurement_cost == 6
    assert fine.measurement_cost == 24
    assert lab.costs == {"observation_cost": 6, "measurement_cost": 30, "queries": 3}
    assert coarse.observations[1].history[0]["measured_q_o0_o1_o2_o3"] == sum(
        fine.observations[0].features[f"measured_q_o{i}"] for i in range(4))
    assert not set(fine.observations[0].features) & set(point)


def test_readout_and_law_are_not_published_by_the_public_view():
    first = UnifiedObjectLab(readout="total", variant="exchange", seed=12)
    second = UnifiedObjectLab(readout="event", variant="anchored", seed=103)
    assert first.channels == second.channels
    for channels in first.measurement_views:
        assert first.spec_for(channels) == second.spec_for(channels)
    assert "object_schema" not in first.spec.context
    schema = ObjectSchema.from_dict(json.loads(first.spec_for(("fine",)).context["object_schema"]))
    assert schema.objects == ("o0", "o1", "o2", "o3")
    assert set(schema.fields["o0"]) == {"q", "x", "u", "f"}
    with pytest.raises(ValueError, match="choose"):
        first.acquire(first.random_probe(41, length=2), ("coarse", "fine"))


def test_renamed_instruments_keep_the_schema_and_histories_consistent():
    lab = UnifiedObjectLab(rename_prefix="meter_", readout="window")
    plain = UnifiedObjectLab(readout="window")
    probe = lab.random_probe(181, length=5)
    for channels in (("coarse",), ("fine",)):
        spec = lab.spec_for(channels)
        schema = ObjectSchema.from_dict(json.loads(spec.context["object_schema"]))
        assert schema.controls["o0"]["q"] == "q_o0"
        assert set(spec.feature_names) == set(schema.feature_names)
        renamed_rows, plain_rows = lab.observe(probe, channels), plain.observe(probe, channels)
        for row, original in zip(renamed_rows, plain_rows):
            assert row.target == original.target
            assert row.features == {key.replace("measured_", "measured_meter_", 1): value
                                    for key, value in original.features.items()}
            assert all(set(past) == {*spec.feature_names, "__target__"} for past in row.history)


def test_evaluation_is_independent_of_learning_costs_and_restore_replays_events():
    with UnifiedObjectLab(readout="event", profile="confirmation") as lab:
        initial = lab.acquire(lab.random_probe(181, length=5), ("fine",))
        before = dict(lab.costs)
        rows = lab.evaluation_rows(("fine",), seed=510, length=7, trajectories=2)
        assert len(rows) == 14
        assert len(rows[0].history) == len(rows[7].history) == 0
        assert rows == lab.evaluation_rows(("fine",), seed=510, length=7, trajectories=2)
        assert lab.costs == before
        acquisitions = json.loads(json.dumps(lab.acquisitions))
        lab.restore(acquisitions)
        assert lab.costs == before
        assert lab.acquire(Probe(acquisitions[0]["probe_id"], tuple(acquisitions[0]["sequence"])),
                           ("fine",)).observations == initial.observations


@pytest.mark.parametrize("prefix", ["", "meter_"])
def test_paid_fine_measurements_project_to_the_same_cheaper_real_view(prefix):
    lab = UnifiedObjectLab(rename_prefix=prefix, readout="window")
    probe = lab.random_probe(181, length=8)
    rows = lab.observe(probe, ("fine",))
    before = dict(lab.costs)
    projected = lab.project_rows(rows, ("fine",), ("coarse",))
    assert lab.costs == before
    assert lab.projectable_views(("fine",)) == (("coarse",), ("fine",))
    assert lab.project_rows(rows, ("fine",), ("fine",)) is rows
    assert projected == lab.observe(probe, ("coarse",))
    assert lab.measurement_cost - before["measurement_cost"] == 3 * len(rows)
    with pytest.raises(ValueError, match="only fine"):
        lab.project_rows(projected, ("coarse",), ("fine",))


def test_paid_transitions_preserve_real_conservation_and_project_with_renamed_fields():
    lab = UnifiedObjectLab(rename_prefix="sensor_", readout="focus")
    probe = lab.random_probe(471, length=8)
    commands = lab.acquire(probe)
    assert commands.transitions == []
    fine = lab.acquire(probe, ("fine",))
    assert len(fine.transitions) == len(fine.observations)
    for row, transition in zip(fine.observations, fine.transitions):
        assert transition["before"] == row.features
        assert abs(sum(transition["after"][f"measured_sensor_q_o{i}"]
                       - transition["before"][f"measured_sensor_q_o{i}"]
                       - transition["before"][f"measured_sensor_u_o{i}"]
                       for i in range(4))) < 1e-12
    before = dict(lab.costs)
    projected = lab.project_transitions(fine.transitions, ("fine",), ("coarse",))
    assert lab.costs == before
    assert projected == lab.acquire(probe, ("coarse",)).transitions
    assert ObjectMeasurementBatch.from_dict(json.loads(json.dumps(fine.to_dict()))) == fine
