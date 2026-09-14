"""Paid repeats answer frozen prediction questions before reopening research."""

import json

import pytest

from hegel_machine.cognition.joint_model_revision import JointModelRevision
from hegel_machine.cognition.models import Observation, TaskSpec


def _observe(research, point, target, *, default=0., learned=0., noise=.001, total=20):
    index = len(research.to_dict()["observations"])
    research.observe(Observation(point, target), default, learned, noise_variance=noise,
                     context={"stage": index + 1, "global_index": index,
                              "remaining_calls": total - index - 1})


def _reference(total=20):
    spec = TaskSpec("repeat_question", "Predict measured responses", ("x",), (), (),
                    feature_bounds={"x": (0., 1.)})
    research = JointModelRevision(spec)
    for x in (.1, .3, .6, .9):
        _observe(research, {"x": x}, 0., total=total)
    return research


def test_requesting_a_repeat_does_not_claim_confirmation_or_spend_an_unmeasured_call():
    research = _reference()
    point = {"x": .37}
    _observe(research, point, 1., default=0., learned=.1)
    state = research.to_dict()
    assert state["phase"] == "confirming"
    assert research.next_control() == point and research.control_purpose == "confirmation"
    assert state["confirmation"]["frozen_predictions"] == {"default": 0., "learned": .1}
    assert not state["confirmations"] and not state["episodes"]
    assert research.accepted_revision is None and research.candidate is None
    assert research.observation_repair is None
    assert state["costs"]["confirmation_calls"] == 0
    assert len(state["observations"]) == 5


def test_a_persistent_counterexample_uses_frozen_predictions_even_when_the_updated_gp_is_perfect():
    research = _reference()
    point = {"x": .37}
    _observe(research, point, 1., default=0., learned=.1)
    threshold = research.to_dict()["confirmation"]["threshold"]
    repeated = research.next_control()
    assert repeated == point
    _observe(research, repeated, 1.02, default=1.02, learned=1.02, noise=100.)
    state = research.to_dict()
    confirmation, = state["confirmations"]
    assert confirmation["confirmed"] and confirmation["same_direction"]
    assert min(confirmation["repeat_errors"].values()) > threshold
    assert state["observations"][-1]["default_error"] == state["observations"][-1]["learned_error"] == 0.
    assert state["reference_scale"] == state["reference_noise"] == .001
    episode, = state["episodes"]
    assert episode["scope_start"] == 4 and episode["trigger_indices"] == [4, 5]
    assert episode["row_indices"] == [4, 5]
    assert episode["trigger_window"]["evidence_kind"] == "repeated_control"
    assert research.control_purpose == "coverage" and research.accepted_revision is None
    assert research.observation_repair is None
    assert state["costs"]["confirmation_calls"] == 1
    assert state["costs"]["coverage_calls"] == state["costs"]["validation_calls"] == 0


@pytest.mark.parametrize("repeat_target", [0., -1.])
def test_an_isolated_spike_or_opposite_residual_does_not_support_the_original_counterexample(repeat_target):
    research = _reference()
    _observe(research, {"x": .37}, 1.)
    point = research.next_control()
    # Assimilating the first spike would make a normal repeat look wrong to
    # today's GP. Confirmation continues to ask about yesterday's prediction.
    _observe(research, point, repeat_target, default=1., learned=1.)
    state = research.to_dict()
    confirmation, = state["confirmations"]
    assert not confirmation["confirmed"]
    assert state["phase"] == "monitoring" and not state["episodes"]
    assert research.next_control() is None and research.accepted_revision is None
    assert state["costs"]["confirmation_calls"] == 1
    assert state["costs"]["coverage_calls"] == state["costs"]["validation_calls"] == 0
    assert len(state["observations"]) == 6  # The unsuccessful experiment remains paid evidence.
    assert confirmation["anchor_index"] not in state["pending_window"]
    if repeat_target < 0:
        assert min(confirmation["repeat_errors"].values()) > confirmation["threshold"]
        assert not confirmation["same_direction"]
        assert research.observation_repair is None
    else:
        assert research.observation_repair["observation_index"] == confirmation["anchor_index"]


def test_json_resume_preserves_the_pending_control_and_frozen_question(tmp_path):
    original = _reference()
    _observe(original, {"x": .37}, 1., default=.1, learned=.2)
    path = tmp_path / "pending_repeat.json"
    path.write_text(json.dumps(original.to_dict()))
    restored = JointModelRevision(original.spec, state=json.loads(path.read_text()))
    assert restored.next_control() == original.next_control()
    assert restored.to_dict()["confirmation"] == original.to_dict()["confirmation"]
    for controller in (original, restored):
        _observe(controller, controller.next_control(), 1.01, default=1.01, learned=1.01)
    assert json.loads(json.dumps(original.to_dict())) == json.loads(json.dumps(restored.to_dict()))
    assert restored.to_dict()["costs"]["confirmation_calls"] == 1
    assert len(restored.to_dict()["confirmations"]) == 1


@pytest.mark.parametrize("remaining", [0, 7])
def test_confirmation_is_deferred_when_it_cannot_leave_room_for_research_and_validation(remaining):
    total = 5 + remaining
    research = _reference(total)
    _observe(research, {"x": .37}, 1., total=total)
    state = research.to_dict()
    assert state["phase"] == "monitoring"
    assert research.next_control() is None and research.control_purpose is None
    assert state["confirmation"] is None and state["costs"]["confirmation_calls"] == 0
    event, = state["events"]
    assert event["action"] == "defer_counterexample_confirmation"
    assert event["remaining_calls"] == remaining
    assert len(state["observations"]) == 5 and state["pending_window"] == [4]
