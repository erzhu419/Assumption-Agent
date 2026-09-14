"""Research scopes remain hypotheses when later paid predictions contradict them."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.joint_model_revision import JointModelRevision
from hegel_machine.cognition.models import Observation, TaskSpec


def _spec():
    names = ("x1", "x2", "x3")
    return TaskSpec("research_scope", "Predict a measured response", names, (), (),
                    feature_bounds={name: (0., 1.) for name in names})


def _feed(research, point, target, *, default=0., learned=0., noise=.001, total=64):
    index = len(research.to_dict()["observations"])
    predictions = {key: program.predict(Observation(point, 0.))
                   for key, program in research.candidates.items()}
    research.observe(Observation(point, target), default, learned,
        candidate_predictions=predictions, noise_variance=noise,
        context={"stage": 33 + index, "global_index": 32 + index,
                 "remaining_calls": total - 33 - index})


def _response(point):
    return 1. + point["x1"] ** 2


def _research(*, compute_budget=20000):
    research = JointModelRevision(_spec(), compute_budget=compute_budget)
    for x in (.1, .3, .6, .9):
        _feed(research, {"x1": x, "x2": .4, "x3": .7}, 0.)
    point = {"x1": .2, "x2": .3, "x3": .4}
    _feed(research, point, _response(point))
    _feed(research, research.next_control(), _response(point),
          default=_response(point), learned=_response(point))
    assert research.control_purpose == "coverage"
    return research


def _supported_coverage(research, count):
    for _ in range(count):
        point = research.next_control()
        target = _response(point)
        _feed(research, point, target, default=target, learned=target)


def _second_failure(research):
    point = research.next_control()
    prediction = _response(point)
    _feed(research, point, prediction + 2., default=prediction, learned=prediction)
    assert research.control_purpose == "confirmation"
    assert research.next_control() == point
    return point, prediction


def test_actual_paid_prediction_prefix_requests_a_new_confirmation_during_coverage():
    # Exact public predictions/measurements from the v0.37 development failure
    # new_confirmation_2_mechanism_change_noise_0.03. Stop at the new request:
    # the old run's stage 46 measured another point and cannot answer it.
    prefix = [
        ([8, 45, 77], -.09420924975908536, -.09986574590819464, -.11489422173363775, .0006769518527161122),
        ([68, 93, 63], .17208263448722833, .1697627022901389, .12404478967469229, .0006564384632398665),
        ([25, 59, 70], -.1601213063396993, -.18220228141504868, -.1550482875426201, .0006371317437328115),
        ([26, 66, 70], -.22790076772319434, -.1660904868083241, -.15974467800098963, .0006189282653404456),
        ([26, 69, 70], -.20459137407302008, -.23629737696137132, -.18276154015696242, .000601736091303211),
        ([56, 41, 96], .045399550934327015, .21106466158758264, .1907621861145188, .0005854732239706919),
        ([56, 41, 96], .06310896781337696, .04598158077762417, .08224523956913862, .0005700662970240947),
        ([97, 73, 5], .9564907255406483, 1.291171577932702, .9493705175788237, .0005741926581582524),
        ([16, 32, 81], -.012721214533525992, -.03964848081222028, -.06581085962464198, .0005594700258977844),
        ([62, 81, 65], .04692822699637848, .04501532552172305, .011030594357493158, .0005454835252503395),
        ([85, 92, 91], .4408154087566722, .3994309183943483, .4623468083671957, .0005321792929271605),
        ([49, 39, 19], .16399987657909584, .21262043742039588, .2122365145472226, .0005195085954765139),
        ([70, 15, 45], -.01913467408496519, .5464413999944422, .4227004342579988, .0005074272327910135),
    ]
    research = JointModelRevision(_spec(), seed=1348110188)
    for index, (integer_point, target, default, learned, noise) in enumerate(prefix):
        if research.next_control() is not None:
            assert [round(100 * research.next_control()[name]) for name in research.spec.feature_names] == integer_point
        point = dict(zip(research.spec.feature_names, (value / 100 for value in integer_point)))
        _feed(research, point, target, default=default, learned=learned, noise=noise)
        if index == 11:
            assert research.control_purpose == "coverage"
    state = research.to_dict()
    assert research.control_purpose == "confirmation"
    assert research.next_control() == {"x1": .70, "x2": .15, "x3": .45}
    assert state["confirmation"]["anchor_index"] == 12
    assert state["reference_scale"] == pytest.approx(.0010863645802843382)
    assert len(state["episodes"]) == 1 and len(state["observations"]) == 13
    assert state["costs"]["confirmation_calls"] == 1  # The second repeat is not measured yet.
    assert state["costs"]["coverage_calls"] == 6


@pytest.mark.parametrize("phase", ["collecting", "validating"])
def test_a_disconfirmed_scope_question_restores_the_prior_research_queue_and_frozen_candidates(phase):
    research = _research()
    _supported_coverage(research, 4 if phase == "collecting" else 8)
    before = research.to_dict()
    assert before["phase"] == phase
    frozen = research.candidates
    point, prediction = _second_failure(research)
    # Opposite, still-large residuals do not support a new scope or authorize
    # deleting either reading as an outlier.
    _feed(research, point, prediction - 2., default=prediction + 2., learned=prediction + 2.)
    state = research.to_dict()
    assert state["phase"] == phase
    assert state["controls"] == before["controls"][1:]
    assert research.candidates == frozen
    assert research.observation_repair is None
    assert len(state["episodes"]) == 1
    assert state["episodes"][0]["scope_start"] == before["episodes"][0]["scope_start"]
    assert state["observations"][:len(before["observations"])] == before["observations"]
    assert len(state["observations"]) == len(before["observations"]) + 2
    assert state["costs"]["confirmation_calls"] == 2


@pytest.mark.parametrize("phase", ["collecting", "validating"])
def test_a_confirmed_new_scope_preserves_the_old_episode_and_reserves_future_validation(phase):
    research = _research()
    _supported_coverage(research, 4 if phase == "collecting" else 8)
    before = research.to_dict()
    anchor_index = len(before["observations"])
    point, prediction = _second_failure(research)
    _feed(research, point, prediction + 2., default=prediction + 2., learned=prediction + 2.)
    state = research.to_dict()
    previous, current = state["episodes"]
    assert previous["status"] == "superseded_by_new_evidence"
    assert previous["scope_start"] == before["episodes"][0]["scope_start"]
    assert previous["proposals"] == before["episodes"][0]["proposals"]
    assert current["scope_start"] == 32 + anchor_index
    assert current["trigger_indices"] == current["row_indices"] == [anchor_index, anchor_index + 1]
    assert current["reserved_validation"] == 4
    remaining = state["observations"][-1]["context"]["remaining_calls"]
    assert current["planned_coverage"] + current["reserved_validation"] <= remaining
    assert state["observations"][:len(before["observations"])] == before["observations"]
    assert state["costs"]["confirmation_calls"] == 2
    assert state["costs"]["coverage_calls"] >= before["costs"]["coverage_calls"]
    assert research.accepted_revision is None and research.control_purpose == "coverage"


def test_persistent_unrecovered_failure_does_not_restart_research_at_every_new_control():
    research = _research()
    while research.control_purpose == "coverage":
        point = research.next_control()
        _feed(research, point, _response(point))
        assert research.control_purpose != "confirmation"
    assert research.control_purpose == "validation"
    while research.control_purpose == "validation":
        point = research.next_control()
        _feed(research, point, _response(point))
        assert research.control_purpose != "confirmation"
    assert len(research.to_dict()["episodes"]) == 1
    assert research.to_dict()["costs"]["confirmation_calls"] == 1


@pytest.mark.parametrize("phase", ["collecting", "validating"])
def test_a_counterexample_at_the_last_control_resumes_the_deferred_batch_operation(phase):
    research = _research()
    _supported_coverage(research, 7 if phase == "collecting" else 8)
    if phase == "validating":
        for _ in range(3):
            point = research.next_control()
            target = _response(point)
            _feed(research, point, target, default=target, learned=target)
    assert len(research.to_dict()["controls"]) == 1
    point, prediction = _second_failure(research)
    _feed(research, point, prediction - 2., default=prediction + 2., learned=prediction + 2.)
    state = research.to_dict()
    assert len(state["episodes"]) == 1
    if phase == "collecting":
        assert state["phase"] == "validating" and len(state["episodes"][0]["proposals"]) == 1
    else:
        validation = state["episodes"][0]["proposals"][0]["validation"]
        assert not validation["supported"]
        assert len(validation["observation_indices"]) >= 4
        assert state["phase"] == "collecting"
    assert state["costs"]["confirmation_calls"] == 2


def test_json_resume_keeps_the_interrupted_research_and_frozen_scope_question():
    research = _research()
    _supported_coverage(research, 8)
    point, prediction = _second_failure(research)
    pending = json.loads(json.dumps(research.to_dict()))
    restored = JointModelRevision(research.spec, state=pending)
    assert restored.next_control() == point
    assert restored.candidates == research.candidates
    for controller in (research, restored):
        _feed(controller, controller.next_control(), prediction - 2.,
              default=prediction + 2., learned=prediction + 2.)
    assert json.loads(json.dumps(research.to_dict())) == json.loads(json.dumps(restored.to_dict()))
    assert restored.control_purpose == "validation"


def test_unresolved_research_keeps_independent_recovery_evidence_without_inventing_new_compute():
    research = _research(compute_budget=0)
    _supported_coverage(research, 8)
    state = research.to_dict()
    assert state["phase"] == "unresolved" and state["costs"]["compute_cost"] == 0
    point = {"x1": .2, "x2": .3, "x3": .4}
    # An unsuccessful new prediction clears the earlier recovery segment.
    _feed(research, point, 3., default=1., learned=1.)
    for _ in range(4):
        _feed(research, point, 1., default=1., learned=1.)
    assert len(research.to_dict()["research_recovery_indices"]) == 1
    for x in (.3, .4, .5):
        _feed(research, {"x1": x, "x2": .3, "x3": .4}, 1., default=1., learned=1.)
    state = research.to_dict()
    assert len(state["research_recovery_indices"]) == 4
    assert state["observations"][-1]["context"]["remaining_calls"] >= 9
    _feed(research, {"x1": .6, "x2": .3, "x3": .4}, 3., default=1., learned=1.)
    state = research.to_dict()
    assert state["phase"] == "unresolved" and research.next_control() is None
    assert state["costs"]["compute_cost"] == state["compute_budget"] == 0
    assert state["costs"]["confirmation_calls"] == 1
    assert len(state["episodes"]) == 1


def test_one_predictor_still_failing_breaks_the_required_joint_recovery_segment():
    research = _research()
    _supported_coverage(research, 3)
    point = research.next_control()
    target = _response(point)
    _feed(research, point, target, default=target + 1., learned=target)
    assert not research.to_dict()["research_recovery_indices"]
    _supported_coverage(research, 3)
    point = research.next_control()
    target = _response(point)
    _feed(research, point, target + 2., default=target, learned=target)
    assert research.control_purpose == "validation"
    assert len(research.to_dict()["episodes"]) == 1
    assert research.to_dict()["costs"]["confirmation_calls"] == 1


@pytest.mark.parametrize("phase", ["collecting", "validating"])
def test_a_repaired_last_measurement_loses_fit_influence_but_keeps_its_raw_evidence(phase):
    research = _research()
    _supported_coverage(research, 7 if phase == "collecting" else 8)
    if phase == "validating":
        for _ in range(3):
            point = research.next_control()
            target = _response(point)
            _feed(research, point, target, default=target, learned=target)
    before = research.to_dict()
    anchor_index = len(before["observations"])
    point, prediction = _second_failure(research)
    _feed(research, point, prediction, default=prediction + 2., learned=prediction + 2.)
    repair = research.observation_repair
    assert repair["observation_index"] == anchor_index
    state = research.to_dict()
    episode, = state["episodes"]
    assert anchor_index in episode["row_indices"]
    assert state["observations"][anchor_index]["observation"]["target"] == prediction + 2.
    if phase == "collecting":
        proposal, = episode["proposals"]
        assert anchor_index not in proposal["training_indices"]
        assert anchor_index + 1 in proposal["training_indices"]
        assert research.candidate.predict(Observation(point, 0.)) == pytest.approx(prediction, abs=1e-6)
    else:
        validation = episode["proposals"][0]["validation"]
        assert validation["supported"]
        assert anchor_index not in validation["observation_indices"]
        assert anchor_index + 1 in validation["observation_indices"]
        assert state["observations"][anchor_index + 1]["candidate_predictions"] == state["observations"][anchor_index]["candidate_predictions"]
        assert research.accepted_revision is not None
        assert state["costs"]["validation_calls"] == 4
    research.acknowledge_observation_repair()
    state = research.to_dict()
    assert state["observations"][anchor_index]["observation_status"] == "disputed"
    assert state["observations"][anchor_index]["observation"]["target"] == prediction + 2.
    assert len(state["observations"]) == anchor_index + 2
    assert state["costs"]["confirmation_calls"] == 2


def test_the_new_segment_can_fit_and_pass_fresh_measurements_with_the_original_total_budget():
    research = _research()
    _supported_coverage(research, 4)
    anchor_index = len(research.to_dict()["observations"])
    point, prediction = _second_failure(research)
    _feed(research, point, prediction + 2., default=prediction + 2., learned=prediction + 2.)
    while research.control_purpose == "coverage":
        point = research.next_control()
        _feed(research, point, _response(point) + 2., default=_response(point), learned=_response(point))
    assert research.control_purpose == "validation"
    frozen = deepcopy(research.to_dict()["competitors"])
    while research.control_purpose == "validation":
        point = research.next_control()
        _feed(research, point, _response(point) + 2., default=_response(point), learned=_response(point))
        assert research.to_dict()["competitors"] == frozen
    accepted = research.accepted_revision
    assert accepted is not None and min(accepted["training_indices"]) == anchor_index
    assert not set(accepted["training_indices"]) & set(accepted["validation_indices"])
    point = {"x1": .73, "x2": .1, "x3": .9}
    assert accepted["program"].predict(Observation(point, 0.)) == pytest.approx(_response(point) + 2., abs=1e-6)
    state = research.to_dict()
    assert state["costs"]["compute_cost"] <= state["compute_budget"] == 20000
    assert len(state["observations"]) + 32 <= 64
    assert state["observations"][-1]["context"]["remaining_calls"] == 64 - 32 - len(state["observations"])
    assert len(state["episodes"]) == 2
