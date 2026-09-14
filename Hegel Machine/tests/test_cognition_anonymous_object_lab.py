from dataclasses import asdict
import json

import pytest

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.models import Probe
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab


@pytest.mark.parametrize("objects", [3, 4])
def test_anonymous_interface_discloses_neither_roles_nor_object_partition(objects):
    first = AnonymousObjectLab(objects=objects, apparatus_seed=9013, name_seed=241, wiring_seed=17)
    changed = AnonymousObjectLab(objects=objects, apparatus_seed=9017, name_seed=241, wiring_seed=29,
                                 readout="window_total_event", variant="sink", actuator_curvature=.31)
    assert first.channels == changed.channels
    for channels in first.measurement_views:
        spec = first.spec_for(channels)
        assert spec == changed.spec_for(channels)
        assert spec.context == {}
        assert set(spec.feature_bounds) == set(first.spec.feature_names)
        payload = json.dumps(asdict(spec))
        assert all(word not in payload for word in ("object_schema", "measurement_scale", "q_o", "x_o", "u_o", "f_o"))
    assert first._evaluation_schema().fields != changed._evaluation_schema().fields
    assert first._evaluation_schema().controls != changed._evaluation_schema().controls
    assert set(first.spec.feature_names).isdisjoint(first.spec_for(("fine",)).feature_names)
    assert set(first.spec_for(("fine",)).feature_names).isdisjoint(first.spec_for(("coarse",)).feature_names)
    for channel in first.channels:
        assert all(word not in channel.description for word in ("quantity", "position", "focus", "member", "object"))
    renamed = AnonymousObjectLab(objects=objects, apparatus_seed=9013, name_seed=997, wiring_seed=17)
    assert set(renamed.spec.feature_names).isdisjoint(first.spec.feature_names)


@pytest.mark.parametrize("channels", [(), ("coarse",), ("fine",)])
def test_anonymous_values_histories_and_transitions_match_the_physical_acquisition(channels):
    settings = dict(objects=3, apparatus_seed=9013, readout="window_total_event", actuator_curvature=.31)
    plain = UnifiedObjectLab(**settings)
    anonymous = AnonymousObjectLab(**settings, name_seed=509, wiring_seed=37)
    original_probe = plain.random_probe(1217, length=7)
    probe = Probe("physical_comparison", tuple({anonymous._control_names[name]: value
        for name, value in point.items()} for point in original_probe.sequence))
    original = plain.acquire(original_probe, channels)
    observed = anonymous.acquire(probe, channels)
    mapping = anonymous._measurement_names[channels] if channels else anonymous._control_names
    for expected, actual in zip(original.observations, observed.observations):
        assert actual.target == expected.target
        assert actual.features == {mapping[name]: value for name, value in expected.features.items()}
        assert actual.history == tuple({mapping.get(name, name): value for name, value in point.items()}
                                       for point in expected.history)
    assert observed.transitions == [{side: {mapping[name]: value for name, value in values.items()}
        for side, values in transition.items()} for transition in original.transitions]
    assert observed.observation_cost == original.observation_cost
    assert observed.measurement_cost == original.measurement_cost
    assert anonymous.acquisitions[0]["sequence"] == list(probe.sequence)


def test_anonymous_banks_do_not_offer_a_hidden_object_projection():
    lab = AnonymousObjectLab(objects=3, name_seed=211)
    batch = lab.acquire(lab.random_probe(3137, length=4), ("fine",))
    assert lab.projectable_views(("fine",)) == (("fine",),)
    before = dict(lab.costs)
    assert lab.project_rows(batch.observations, ("fine",), ("fine",)) is batch.observations
    with pytest.raises(ValueError, match="no supplied projection"):
        lab.project_rows(batch.observations, ("fine",), ("coarse",))
    with pytest.raises(ValueError, match="no supplied projection"):
        lab.project_transitions(batch.transitions, ("fine",), ("coarse",))
    assert lab.costs == before


def test_name_and_wiring_survive_evaluation_and_saved_acquisition_restore():
    lab = AnonymousObjectLab(objects=3, readout="window_total_event", variant="nonlocal",
        apparatus_seed=9013, actuator_curvature=.47, name_seed=821, wiring_seed=103)
    probe = lab.random_probe(7013, length=8)
    batch = lab.acquire(probe, ("fine",))
    before = dict(lab.costs)
    assert lab.evaluation_rows(("fine",), seed=7013, length=8, trajectories=1) == batch.observations
    assert lab.costs == before
    truth = lab._evaluation_schema().to_dict()
    saved = json.loads(json.dumps(lab.acquisitions))
    lab.restore(saved)
    assert lab.costs == before
    assert lab._evaluation_schema().to_dict() == truth
    assert lab.acquire(probe, ("fine",)) == batch


def test_total_response_does_not_identify_which_quantity_belongs_to_a_focus_indicator():
    settings = dict(objects=3, apparatus_seed=9013, name_seed=631, wiring_seed=47)
    total = AnonymousObjectLab(readout="total", **settings)
    focus = AnonymousObjectLab(readout="focus", **settings)
    truth = total._evaluation_schema()
    point = total.random_probe(1039, length=1).sequence[0]
    probe = Probe("vary_single_selector", tuple({**point, truth.focus_control: float(index)} for index in range(3)))
    total_batch = total.acquire(probe, ("fine",))
    focus_batch = focus.acquire(probe, ("fine",))
    assert max(row.target for row in total_batch.observations) - min(row.target for row in total_batch.observations) < 1e-12
    assert max(row.target for row in focus_batch.observations) - min(row.target for row in focus_batch.observations) > .1
    for side in ("before", "after"):
        for obj in truth.objects:
            for role in ("q", "x", "u"):
                name = truth.fields[obj][role]
                assert len({row[side][name] for row in total_batch.transitions}) == 1
            assert len({row[side][truth.fields[obj]["f"]] for row in total_batch.transitions}) == 2
