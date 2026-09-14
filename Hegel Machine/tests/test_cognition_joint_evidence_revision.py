"""A changed goal can buy a constraint question, but cannot answer it."""

import json

import pytest

from hegel_machine.cognition.joint_model_revision import JointModelRevision
from hegel_machine.cognition.models import Observation, TaskSpec


POINT = {"x1": .37}


def _spec():
    return TaskSpec("temporal_evidence", "Predict paid measurements", ("x1",), (), (),
                    feature_bounds={"x1": (0., 1.)})


def _feed(research, target, *, point=None, default=None, learned=None):
    point = point or research.next_control() or POINT
    index = len(research.to_dict()["observations"])
    predictions = {key: program.predict(Observation(point, 0.))
                   for key, program in research.candidates.items()}
    research.observe(Observation(point, target), target if default is None else default,
        target if learned is None else learned, noise_variance=.001,
        candidate_predictions=predictions,
        context={"stage": 33 + index, "global_index": 32 + index,
                 "remaining_calls": 31 - index, "objective": -target})


def _referenced():
    research = JointModelRevision(_spec())
    # Historical GP approximation error is larger than measurement noise.
    # Repeating one measured control need not inherit that approximation error.
    for x in (.1, .3, .6, .9):
        _feed(research, 0., point={"x1": x}, default=.3, learned=.3)
    assert research.to_dict()["reference_scale"] == pytest.approx(.09)
    assert research.to_dict()["reference_noise"] == pytest.approx(.001)
    return research


def _start(research, *, target=0., count=1, scope=35, remaining=None):
    stage = 32 + len(research.to_dict()["observations"])
    return research.start_applicability_probe(POINT,
        [{"global_index": i + 12, "target": target} for i in range(count)],
        goal_scope_start=scope,
        context={"stage": stage, "remaining_calls": 64 - stage if remaining is None else remaining})


def test_current_gp_accuracy_does_not_prevent_a_paid_temporal_constraint_question():
    research = _referenced()
    history = research.to_dict()["observations"]
    assert _start(research)
    assert research.control_purpose == "applicability_probe"
    assert research.to_dict()["costs"]["applicability_probe_calls"] == 0
    _feed(research, .5)
    assert research.control_purpose == "applicability_confirmation"
    assert research.next_control() == POINT
    assert not research.to_dict()["episodes"]
    _feed(research, .51)
    state = research.to_dict()
    record, = state["applicability_probes"]
    assert record["confirmed"] and record["threshold"] == pytest.approx(.018)
    episode, = state["episodes"]
    assert episode["scope_start"] == 36
    assert episode["row_indices"] == episode["trigger_indices"] == [4, 5]
    assert episode["trigger_window"]["evidence_kind"] == "measured_constraint_change"
    assert state["observations"][:4] == history
    assert all(row["default_error"] == row["learned_error"] == 0.
               for row in state["observations"][4:])
    assert state["reference_scale"] == pytest.approx(.09)
    assert state["costs"]["applicability_probe_calls"] == 1
    assert state["costs"]["applicability_confirmation_calls"] == 1


def test_a_changed_goal_with_stable_constraint_does_not_reopen_or_repeat_the_attempt():
    research = _referenced()
    assert _start(research)
    _feed(research, .04)
    state = research.to_dict()
    assert state["phase"] == "monitoring" and not state["episodes"]
    assert state["applicability_probes"][0]["status"] == "reference_retained"
    assert not research.observation_repair
    assert state["costs"]["applicability_probe_calls"] == 1
    assert state["costs"]["applicability_confirmation_calls"] == 0
    assert not _start(research)
    assert _start(research, scope=36)


@pytest.mark.parametrize("old_readings,requires_confirmation", [(1, False), (4, True)])
def test_paid_old_replicates_reduce_reference_mean_uncertainty(old_readings, requires_confirmation):
    research = _referenced()
    assert _start(research, count=old_readings)
    _feed(research, .12)
    assert (research.control_purpose == "applicability_confirmation") is requires_confirmation
    assert not research.to_dict()["episodes"]
    assert not research.observation_repair


def test_json_restores_frozen_temporal_question_and_refuted_outlier_keeps_paid_history():
    research = _referenced()
    assert _start(research)
    _feed(research, 1.)
    state = json.loads(json.dumps(research.to_dict()))
    restored = JointModelRevision(_spec(), state=state)
    assert restored.next_control() == POINT
    assert restored.control_purpose == "applicability_confirmation"
    _feed(restored, .01)
    _feed(research, .01)
    assert json.loads(json.dumps(restored.to_dict())) == json.loads(json.dumps(research.to_dict()))
    repair = restored.observation_repair
    assert repair["global_index"] == 36 and repair["confirmation_global_index"] == 37
    assert repair["observation_index"] == 4 and repair["confirmation_index"] == 5
    assert repair["reason"] == "isolated_observation_conflict" and repair["output_index"] == 1
    before = restored.to_dict()
    restored.acknowledge_observation_repair()
    after = restored.to_dict()
    assert not after["episodes"] and len(after["observations"]) == 6
    assert [row["observation"] for row in after["observations"]] == [
        row["observation"] for row in before["observations"]]
    assert after["observations"][4]["observation_status"] == "disputed"
    assert after["costs"] == before["costs"]
    assert after["observation_repairs"] == [repair]


def test_opposing_large_new_readings_do_not_confirm_change_or_delete_either_reading():
    research = _referenced()
    assert _start(research)
    _feed(research, .5)
    _feed(research, -.5)
    state = research.to_dict()
    assert state["phase"] == "monitoring" and not state["episodes"]
    assert state["applicability_probes"][0]["status"] == "inconsistent_new_readings"
    assert not research.observation_repair
    assert [row["observation"]["target"] for row in state["observations"][-2:]] == [.5, -.5]


def test_nine_remaining_calls_cover_two_repeats_three_new_controls_and_four_future_tests():
    research = _referenced()
    while len(research.to_dict()["observations"]) < 23:
        _feed(research, 0.)
    assert not _start(research, remaining=8)
    assert _start(research)
    response = lambda point: 1. + point["x1"] ** 2
    _feed(research, response(POINT))
    _feed(research, response(POINT))
    assert research.to_dict()["episodes"][-1]["planned_coverage"] == 3
    controls = []
    while research.control_purpose == "coverage":
        point = research.next_control()
        controls.append(point["x1"])
        _feed(research, response(point), default=0., learned=0.)
    assert len(set(controls)) == 3 and POINT["x1"] not in controls
    assert research.control_purpose == "validation"
    frozen = research.to_dict()["competitors"]
    for index in range(4):
        _feed(research, response(research.next_control()), default=0., learned=0.)
        assert research.to_dict()["competitors"] == frozen
        if index < 3:
            assert research.accepted_revision is None
    accepted = research.accepted_revision
    assert accepted is not None and accepted["scope_start"] == 55
    assert len(accepted["validation_indices"]) == 4
    assert not set(accepted["training_indices"]) & set(accepted["validation_indices"])
    assert min(accepted["training_indices"]) == 23
    state = research.to_dict()
    assert len(state["observations"]) == 32
    assert state["observations"][-1]["context"]["stage"] == 64
    assert state["costs"]["compute_cost"] <= state["compute_budget"] == 20000
    assert state["costs"]["coverage_calls"] == 3
    assert state["costs"]["validation_calls"] == 4


def _active_research(phase):
    research = _referenced()
    _feed(research, 1. + POINT["x1"] ** 2, default=0., learned=0.)
    assert research.control_purpose == "confirmation"
    _feed(research, 1. + POINT["x1"] ** 2, default=0., learned=0.)
    if phase == "validating":
        while research.control_purpose == "coverage":
            _feed(research, 1. + research.next_control()["x1"] ** 2, default=0., learned=0.)
    assert research.to_dict()["phase"] == phase
    return research


@pytest.mark.parametrize("phase", ["collecting", "validating"])
def test_later_goal_evidence_can_confirm_a_new_constraint_scope_during_old_research(phase):
    research = _active_research(phase)
    previous = research.to_dict()
    start = 32 + len(previous["observations"])
    assert _start(research, target=1. + POINT["x1"] ** 2, scope=38)
    _feed(research, -1.)
    _feed(research, -1.01)
    state = research.to_dict()
    first, second = state["episodes"]
    assert first["status"] == "superseded_by_new_evidence"
    assert first["row_indices"] == previous["episodes"][0]["row_indices"]
    assert first["proposals"] == previous["episodes"][0]["proposals"]
    assert second["scope_start"] == start
    assert second["row_indices"] == [start - 32, start - 31]
    assert state["costs"]["compute_cost"] == previous["costs"]["compute_cost"]
    assert state["costs"]["validation_calls"] == previous["costs"]["validation_calls"]
    assert state["observations"][:start - 32] == previous["observations"]


@pytest.mark.parametrize("phase", ["collecting", "validating"])
def test_refuted_temporal_probe_restores_the_frozen_plan_without_counting_it_as_validation(phase):
    research = _active_research(phase)
    before = research.to_dict()
    target = 1. + POINT["x1"] ** 2
    assert not _start(research, target=target, scope=before["episodes"][0]["scope_start"])
    assert _start(research, target=target, scope=38)
    _feed(research, -2.)
    research = JointModelRevision(_spec(), state=json.loads(json.dumps(research.to_dict())))
    _feed(research, target)
    after = research.to_dict()
    assert after["phase"] == phase
    assert after["controls"] == before["controls"]
    assert after["candidate"] == before["candidate"]
    assert after["competitors"] == before["competitors"]
    assert after["episodes"][0]["validation_indices"] == before["episodes"][0]["validation_indices"]
    assert after["costs"]["validation_calls"] == before["costs"]["validation_calls"]
    assert after["episodes"][0]["row_indices"][-2:] == [len(before["observations"]), len(before["observations"]) + 1]
    assert after["observation_repair"]["observation_index"] == len(before["observations"])


def test_pending_ordinary_confirmation_keeps_priority_over_a_goal_suggested_question():
    research = _referenced()
    _feed(research, 1., default=0., learned=0.)
    before = research.to_dict()
    assert research.control_purpose == "confirmation"
    assert not _start(research)
    assert research.to_dict() == before


def test_interrupting_research_reserves_enough_calls_to_finish_its_original_plan_if_refuted():
    research = _active_research("collecting")
    before = research.to_dict()
    assert len(before["controls"]) == 4
    # The new-scope minimum of nine would not also preserve the old four
    # controls plus four validation draws after two paid temporal readings.
    assert not _start(research, scope=38, remaining=9)
    assert research.to_dict() == before
    assert _start(research, scope=38, remaining=10)
