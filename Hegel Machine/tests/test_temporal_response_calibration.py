"""Vary response physics independently without changing existing calibrations."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.ambiguous_instrument_lab import AmbiguousInstrumentLab
from hegel_machine.cognition.models import Probe
from hegel_machine.cognition.stateful_instrument_lab import StatefulInstrumentLab
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab


CONTROLS = (.8, 0., 0., -.7, 0., .4, -.3, 0.)
PROBE = Probe("calibration", tuple({"u": value} for value in CONTROLS))
LABS = [(StatefulInstrumentLab, {"states": 2}),
        (AmbiguousInstrumentLab, {"instrument_mode": "recurrent"}),
        (AmbiguousInstrumentLab, {"instrument_mode": "finite2"})]


@pytest.mark.parametrize("lab_type,options,expected", [
    (*LABS[0], [(1.115790306314788, .8878235158405771),
                (-.6485103931366194, .7982768461867323)]),
    (*LABS[1], [(.7849055279076913, .3611652504620122),
                (-.3327460691761024, .5089116003176494)]),
    (*LABS[2], [(.7849055279076913, .3611652504620122),
                (-.6867923369192298, .8111872083919552)]),
])
def test_original_calibrations_remain_exact(lab_type, options, expected):
    # Captured before adding response_seed: checks both immediate and delayed output.
    settings = dict(task_index=2, seed=17, apparatus_seed=113, **options)
    original = lab_type(**settings).observe(PROBE, ["meter"])
    assert [(original[i].features["meter"], original[i].target) for i in (0, 3)] == expected
    assert original == lab_type(**settings, response_seed=None).observe(PROBE, ["meter"])
    assert original == lab_type(**settings, response_seed=113).observe(PROBE, ["meter"])


@pytest.mark.parametrize("lab_type,options", LABS)
def test_response_change_preserves_instrument_and_survives_restore(lab_type, options):
    settings = dict(task_index=1, apparatus_seed=113, **options)
    original = lab_type(**settings).observe(PROBE, ["meter"])
    changed = lab_type(**settings, response_seed=823)
    changed_rows = changed.observe(PROBE, ["meter"])
    assert [row.features for row in changed_rows] == [row.features for row in original]
    assert max(abs(left.target - right.target)
               for left, right in zip(changed_rows, original)) > .01
    acquisitions, costs = deepcopy(changed.acquisitions), changed.costs
    changed.observe(PROBE)
    changed.restore(acquisitions)
    assert changed.costs == costs
    assert changed.observe(PROBE, ["meter"]) == changed_rows


@pytest.mark.parametrize("lab_type,options", LABS)
def test_new_instrument_can_keep_original_response_law(lab_type, options):
    donor = lab_type(task_index=1, apparatus_seed=113, **options)
    donor_rows = donor.observe(PROBE, ["meter"])
    changed = lab_type(task_index=1, apparatus_seed=823, response_seed=113, **options)
    rows = changed.observe(PROBE, ["meter"])
    assert max(abs(left.features["meter"] - right.features["meter"])
               for left, right in zip(rows, donor_rows)) > .01
    # Apply the donor's response law to the new, actually observed instrument output.
    previous = 0.
    for row in rows:
        reading = row.features["meter"]
        expected = (donor._response_offset + donor._response_control_gain * row.features["u"]
                    + donor._response_square_gain * reading * reading
                    + donor._response_history_gain * previous)
        assert row.target == expected
        previous = reading


@pytest.mark.parametrize("lab_type,options", LABS)
def test_evaluation_clone_keeps_independent_response_physics(lab_type, options):
    lab = lab_type(task_index=1, seed=17, apparatus_seed=113, response_seed=823, **options)
    evaluation = lab.evaluation_rows(["meter"], seed=1903, length=8, trajectories=2)
    assert lab.costs["observation_cost"] == 0
    for offset in (0, 8):
        rows = evaluation[offset:offset + 8]
        replay = Probe("evaluation_replay", tuple({"u": row.features["u"]} for row in rows))
        assert lab.observe(replay, ["meter"]) == rows


@pytest.mark.parametrize("mode", ["finite2", "recurrent"])
def test_temporal_adapter_keeps_response_setting_private_and_in_evaluation(mode):
    settings = dict(task_index=1, apparatus_seed=113, instrument_mode=mode)
    original = TemporalInquiryLab(**settings)
    changed = TemporalInquiryLab(**settings, response_seed=823)
    assert changed.public_spec() == original.public_spec()
    assert "response_seed" not in changed.public_spec()
    action = {"kind": "trajectory", "sequence": list(PROBE.sequence), "channels": ["meter"]}
    old_record, new_record = original.execute(action), changed.execute(action)
    assert old_record["measurements"] == new_record["measurements"]
    assert old_record["rows"] != new_record["rows"]
    costs = changed.costs
    evaluation = changed.evaluation_rows(length=48, seed=1903)
    assert changed.costs == costs
    for offset in (0, 24):
        rows = evaluation[offset:offset + 24]
        replay = {"kind": "trajectory", "sequence": [row.features for row in rows], "channels": []}
        assert changed.observation_rows(changed.execute(replay)) == rows
