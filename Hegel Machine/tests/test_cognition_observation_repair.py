"""A contradicted measurement can lose fitting influence without losing evidence."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.joint_model_revision import JointModelRevision
from hegel_machine.cognition.models import Observation, TaskSpec


def _observe(controller, point, target, *, default=0., learned=0., noise=.001,
             total=64, offset=32):
    index = len(controller.to_dict()["observations"])
    controller.observe(Observation(point, target), default, learned,
        noise_variance=noise,
        context={"stage": offset + index + 1, "global_index": offset + index,
                 "remaining_calls": total - offset - index - 1})


def _reference(noise=.001, dimension=1, count=4):
    names = tuple(f"x{i + 1}" for i in range(dimension))
    spec = TaskSpec("observation_repair", "Predict a measured response", names, (), (),
                    feature_bounds={name: (0., 1.) for name in names})
    controller = JointModelRevision(spec)
    for _ in range(count):
        _observe(controller, {name: .5 for name in names}, 0., noise=noise)
    return controller


@pytest.mark.parametrize("point,anchor,repeat,default,learned,reference", [
    ({"x1": .85, "x2": .97, "x3": .82}, 1.471636122519039,
     .4542606214992885, .5979461016757267, .5145122631221596,
     .0007039122508461197),
    ({"x1": .18, "x2": .33, "x3": .70}, .9821780360304033,
     -.12261043860059079, -.030024703388926155, -.06641411166197322,
     .0008568809843697343),
])
def test_paid_counterexamples_can_repair_a_measurement_when_only_one_old_predictor_is_supported(
        point, anchor, repeat, default, learned, reference):
    # These are the stage 45 / 46 measurements of the two 0.35 outlier cases.
    # Their repeat supports the learned predictor, but not the default one.
    controller = _reference(reference, dimension=3, count=12)
    _observe(controller, point, anchor, default=default, learned=learned)
    assert controller.observation_repair is None
    _observe(controller, controller.next_control(), repeat, default=anchor, learned=anchor)
    state = controller.to_dict()
    confirmation, = state["confirmations"]
    assert not confirmation["confirmed"]
    assert confirmation["repeat_errors"]["default"] > confirmation["threshold"]
    assert confirmation["repeat_errors"]["learned"] < confirmation["threshold"]
    repair = controller.observation_repair
    assert repair["observation_index"] == 12 and repair["global_index"] == 44
    assert repair["confirmation_index"] == 13 and repair["confirmation_global_index"] == 45
    assert repair["output_index"] == 1
    assert repair["reason"] == "isolated_observation_conflict"
    assert repair["repeat_target"] == repeat
    assert state["phase"] == "monitoring" and not state["episodes"]
    assert not state["observation_repairs"]  # A proposal is not an installed repair.

    controller.acknowledge_observation_repair()
    installed = controller.to_dict()
    assert controller.observation_repair is None
    assert installed["observation_repairs"] == [repair]
    assert installed["observations"][12]["observation_status"] == "disputed"
    for old_row, new_row in zip(state["observations"], installed["observations"]):
        assert old_row["observation"] == new_row["observation"]
        assert old_row["default_error"] == new_row["default_error"]
        assert old_row["learned_error"] == new_row["learned_error"]
    assert len(installed["observations"]) == 14
    assert installed["costs"] == state["costs"]
    assert installed["costs"]["confirmation_calls"] == 1
    assert installed["costs"]["coverage_calls"] == installed["costs"]["validation_calls"] == 0
    assert installed["reference_scale"] == state["reference_scale"] == reference


def test_opposite_large_residuals_leave_the_measurement_unresolved_instead_of_discarding_it():
    controller = _reference()
    _observe(controller, {"x1": .37}, 1.)
    _observe(controller, controller.next_control(), -1., default=1., learned=1.)
    state = controller.to_dict()
    confirmation, = state["confirmations"]
    assert not confirmation["confirmed"] and not confirmation["same_direction"]
    assert min(confirmation["repeat_errors"].values()) > confirmation["threshold"]
    assert controller.observation_repair is None and not state["observation_repairs"]
    assert [row["observation"]["target"] for row in state["observations"][-2:]] == [1., -1.]
    assert all(row.get("observation_status") != "disputed" for row in state["observations"])


def test_a_normal_noise_excursion_and_compatible_repeat_keep_both_observations():
    controller = _reference()
    # The first value lies just outside the 3-sigma prediction interval. The
    # independent repeat agrees with the old model; their difference remains
    # compatible with the uncertainty of two measurements.
    _observe(controller, {"x1": .37}, .1)
    assert controller.control_purpose == "confirmation"
    _observe(controller, controller.next_control(), -.02, default=.1, learned=.1)
    state = controller.to_dict()
    assert not state["confirmations"][0]["confirmed"]
    assert controller.observation_repair is None and not state["observation_repairs"]
    assert not state["episodes"] and len(state["observations"]) == 6


def test_pending_repair_survives_json_resume_without_repeating_measurements_or_rewriting_targets():
    original = _reference()
    _observe(original, {"x1": .37}, 1.)
    _observe(original, original.next_control(), .01, default=1., learned=1., noise=100.)
    serialized = json.loads(json.dumps(original.to_dict()))
    restored = JointModelRevision(original.spec, state=serialized)
    assert restored.observation_repair == original.observation_repair
    assert restored.observation_repair["global_index"] == 36
    detached = restored.observation_repair
    detached["global_index"] = -1
    assert restored.observation_repair["global_index"] == 36
    paid = deepcopy(serialized["observations"])
    for controller in (original, restored):
        assert controller.next_control() is None
        controller.acknowledge_observation_repair()
        state = json.loads(json.dumps(controller.to_dict()))
        assert [row["observation"] for row in state["observations"]] == [
            row["observation"] for row in paid]
        assert state["costs"]["confirmation_calls"] == 1
        assert state["reference_scale"] == .001
    assert json.loads(json.dumps(original.to_dict())) == restored.to_dict()


def test_repairing_a_spike_does_not_suppress_research_when_the_mechanism_later_changes():
    controller = _reference()
    _observe(controller, {"x1": .37}, 1.)
    _observe(controller, controller.next_control(), .01, default=1., learned=1.)
    repair = controller.observation_repair
    controller.acknowledge_observation_repair()
    _observe(controller, {"x1": .2}, 0.)
    before_change = len(controller.to_dict()["observations"])
    _observe(controller, {"x1": .73}, 1. + .73 ** 2)
    assert controller.control_purpose == "confirmation"
    point = controller.next_control()
    value = 1. + point["x1"] ** 2
    # Even a predictor that assimilated the new measurement cannot erase the
    # persistent contradiction with its prediction before the change.
    _observe(controller, point, value, default=value, learned=value)
    state = controller.to_dict()
    episode, = state["episodes"]
    assert episode["trigger_indices"] == [before_change, before_change + 1]
    assert episode["scope_start"] == 32 + before_change
    assert controller.control_purpose == "coverage"
    assert controller.observation_repair is None
    assert state["observation_repairs"] == [repair]
    assert state["observations"][4]["observation"]["target"] == 1.
    assert state["observations"][4]["observation_status"] == "disputed"
    assert state["costs"]["confirmation_calls"] == 2
    assert state["costs"]["coverage_calls"] == state["costs"]["validation_calls"] == 0
