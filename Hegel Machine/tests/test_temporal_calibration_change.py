"""A scheduled apparatus update changes real paid rows, not support feedback."""

import json

import pytest

from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab


def trajectory(count=24, *, meter=True):
    controls = (.8, -.3, 0., .5, -.7, .2)
    return {"kind": "trajectory", "sequence": [{"u": controls[i % len(controls)]}
                                                 for i in range(count)],
            "channels": ["meter"] if meter else []}


@pytest.mark.parametrize("mode", ["finite2", "recurrent"])
@pytest.mark.parametrize("post", [
    {"apparatus_seed": 823, "response_seed": 113},
    {"apparatus_seed": 113, "response_seed": 823},
    {"apparatus_seed": 823, "response_seed": 823},
])
def test_real_change_preserves_prefix_and_global_paid_accounting(mode, post):
    settings = {"task_index": 1, "apparatus_seed": 113, "response_seed": 113,
                "instrument_mode": mode}
    shifted = TemporalInquiryLab(**settings,
        calibration_change={"after_rows": 72, **post})
    stable = TemporalInquiryLab(**settings,
        calibration_change={"after_rows": 72, "apparatus_seed": 113, "response_seed": 113})
    final = TemporalInquiryLab(**{**settings, **post})
    old_physics = shifted._laboratory
    for _ in range(3):
        assert shifted.execute(trajectory()) == stable.execute(trajectory())
    assert shifted.costs == {"observation_cost": 72, "measurement_cost": 216, "queries": 3}
    assert [record["probe_id"] for record in old_physics.acquisitions] == [
        "inquiry_0", "inquiry_1", "inquiry_2"]

    for index in (72, 96):
        changed_record, stable_record = shifted.execute(trajectory()), stable.execute(trajectory())
        final_record = final.execute(trajectory())
        assert changed_record["row_indices"] == list(range(index, index + 24))
        assert changed_record["rows"] == final_record["rows"]
        changed_meters = [row["values"] for row in changed_record["measurements"]]
        stable_meters = [row["values"] for row in stable_record["measurements"]]
        assert changed_meters == [row["values"] for row in final_record["measurements"]]
        assert (changed_meters == stable_meters) == (post["apparatus_seed"] == 113)
        assert changed_record["rows"] != stable_record["rows"]
    assert shifted.costs == {"observation_cost": 120, "measurement_cost": 360, "queries": 5}
    assert [record["probe_id"] for record in shifted._laboratory.acquisitions] == [
        "inquiry_3", "inquiry_4"]
    assert shifted.public_spec() == stable.public_spec()
    public = json.dumps(shifted.public_spec())
    assert shifted.public_spec()["settled_after_observation_rows"] == 72
    assert "settled apparatus" in shifted.public_spec()["prediction_objective"]
    assert "seed" not in public and "823" not in public and "113" not in public


def test_change_waits_for_next_whole_trajectory():
    settings = {"task_index": 1, "apparatus_seed": 113, "response_seed": 113}
    lab = TemporalInquiryLab(**settings,
        calibration_change={"after_rows": 72, "apparatus_seed": 823, "response_seed": 113})
    old, final = TemporalInquiryLab(**settings), TemporalInquiryLab(**{**settings, "apparatus_seed": 823})
    for count in (24, 24, 12, 24):
        assert lab.execute(trajectory(count)) == old.execute(trajectory(count))
    assert lab.costs["observation_cost"] == 84
    record = lab.execute(trajectory(12, meter=False))
    assert record["rows"] == final.execute(trajectory(12, meter=False))["rows"]
    assert record["row_indices"] == list(range(84, 96))
    assert lab.costs == {"observation_cost": 96, "measurement_cost": 252, "queries": 5}


@pytest.mark.parametrize("mode", ["finite2", "recurrent"])
def test_unused_evaluation_clone_generates_settled_physics_without_hidden_rows(mode):
    settings = {"task_index": 1, "apparatus_seed": 113, "response_seed": 113,
                "instrument_mode": mode}
    change = {"after_rows": 72, "apparatus_seed": 823, "response_seed": 991}
    lab = TemporalInquiryLab(**settings, calibration_change=change)
    clone = TemporalInquiryLab(**lab._settings)
    final = TemporalInquiryLab(**{**settings, "apparatus_seed": 823, "response_seed": 991})
    expected = final.evaluation_rows(length=53, seed=1709)
    assert clone.evaluation_rows(length=53, seed=1709) == expected
    assert clone.costs == {"observation_cost": 0, "measurement_cost": 0, "queries": 0}
    assert not clone._laboratory.acquisitions
    for _ in range(4):
        lab.execute(trajectory())
    paid_costs = lab.costs
    assert lab.evaluation_rows(length=53, seed=1709) == expected
    assert lab.costs == paid_costs
