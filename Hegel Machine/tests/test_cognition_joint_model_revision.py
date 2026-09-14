"""Persistent paid failures reopen hypotheses and require future confirmation."""

import json
from dataclasses import replace
from random import Random

import pytest

from hegel_machine.cognition.joint_model_revision import JointModelRevision
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec


def _spec(dimension=1):
    names = tuple(f"x{i + 1}" for i in range(dimension))
    return TaskSpec("revision", "Predict measured responses", names, (), (),
                    feature_bounds={name: (0., 1.) for name in names})


def _feed(research, features, target, default=0., learned=0., *, noise=0., remaining=None, objective=None):
    index = len(research.to_dict()["observations"])
    candidate = research.candidate
    prediction = candidate.predict(Observation(features, 0.)) if candidate is not None else None
    predictions = {key: program.predict(Observation(features, 0.))
                   for key, program in research.candidates.items()}
    context = {"stage": index + 1, "global_index": index,
               "remaining_calls": 100 - index - 1 if remaining is None else remaining}
    if objective is not None:
        context["objective"] = objective
    research.observe(Observation(features, target), default, learned, prediction,
        candidate_predictions=predictions,
        noise_variance=noise, context=context)


def _trigger(research, response=lambda x: 1. + x * x, remaining=None):
    for x in (.1, .3, .6, .9):
        _feed(research, {"x1": x}, 0.)
    for index, x in enumerate((.1, .3, .6, .9)):
        left = None if remaining is None else remaining + 3 - index
        _feed(research, {"x1": x}, response(x), remaining=left)
        if research.control_purpose == "confirmation":
            point = research.next_control()
            _feed(research, point, response(point["x1"]),
                  remaining=None if left is None else left - 1)
            return
        if research.to_dict()["phase"] != "monitoring":
            return


def _collect(research, response):
    while research.to_dict()["phase"] == "collecting":
        features = research.next_control()
        _feed(research, features, response(features["x1"]))


def _confirm(research, response):
    while research.to_dict()["phase"] == "validating":
        features = research.next_control()
        _feed(research, features, response(features["x1"]))


def test_real_paid_forecasts_request_a_repeat_at_the_first_common_failure():
    # First sixteen pre-measurement forecasts copied from the immutable
    # sc_kg_prior_mechanism_change_v031_20260910 policy log. No mechanism labels.
    rows = [
        (-.029414235304278336, -.026554075860772314, -.026554081632653152, 3.3873422222841777e-6),
        (-.03130751350779881, -.03227449485826548, -.03227448979591843, 3.284998518578596e-6),
        (.36614634189036654, .3653803279245833, .3653806122448977, 3.188675032738049e-6),
        (.2991985827224057, .29879694642244803, .2987969387755101, 3.0978557460883906e-6),
        (-.16640160484431915, -.1727357204039584, -.17273571428571427, 3.012081975363713e-6),
        (.11961307418066869, .11886633094349146, .11886632653061209, 2.930944624678207e-6),
        (.022320343826956135, .029819387179804507, .029819387755102015, 2.8540776608708856e-6),
        (.5239170536511122, .5139520464291261, .5139520408163264, 2.7811525926434267e-6),
        (-.056429815012918105, -.05911837440926655, -.05911836734693883, 2.7118737778273423e-6),
        (.46940885714560476, .4728877507983935, .4728877551020406, 1e-8),
        (.431842165026344, .4313612225008844, .43136122448979586, 1e-8),
        (1.060581747860195, 1.0576163084778776, 1.057616326530612, 1e-8),
        (1.210785215679853, 1.212432641338735, -.20009999999999997, 1e-8),
        (.6478537587707489, .22312196228247372, .017499999999999995, .044338824615312626),
        (.6970513327629346, .13894212421017038, -.13639999999999997, .04429405263492934),
        (-.9087003099401869, .26249137263013145, -.06889999999999996, .04496465135500399),
    ]
    research = JointModelRevision(_spec(3))
    for index, (default, learned, target, noise) in enumerate(rows):
        research.observe(Observation({name: .5 for name in research.spec.feature_names}, target),
            default, learned, noise_variance=noise,
            context={"stage": 33 + index, "global_index": 32 + index, "remaining_calls": 31 - index})
        if research.to_dict()["phase"] != "monitoring":
            break
    state = research.to_dict()
    assert state["phase"] == "confirming"
    assert state["reference_scale"] == pytest.approx(3.2397178799223035e-6)
    event, = state["events"]
    assert event["action"] == "request_counterexample_confirmation" and event["stage"] == 45
    assert event["anchor_index"] == 12 and not state["episodes"]
    assert research.control_purpose == "confirmation"
    assert state["costs"]["confirmation_calls"] == 0
    # The following old rows came from other controls, so they cannot answer
    # this new same-point question. Their increased variance stays diagnostic.
    assert all(min((d-y)**2, (l-y)**2) < 9*n for d, l, y, n in rows[13:])


@pytest.mark.parametrize("noise", [0., .03])
def test_stable_measured_process_does_not_reopen_from_isolated_noise(noise):
    research = JointModelRevision(_spec())
    rng = Random(5701)
    for index in range(64):
        point = research.next_control()
        x = point["x1"] if point is not None else rng.random()
        expected = 1. + x * x
        residual = rng.gauss(0., noise)
        if index == 20:
            residual += .5  # One isolated counterexample does not support reopening.
        _feed(research, {"x1": x}, expected + residual,
              default=expected, learned=expected, noise=noise ** 2)
    assert research.to_dict()["phase"] == "monitoring"
    assert not research.to_dict()["episodes"]


def test_intermittent_common_failures_reopen_only_when_the_batch_is_complete():
    research = JointModelRevision(_spec())
    for _ in range(4):
        _feed(research, {"x1": .5}, 0.)
    for index, target in enumerate((1., 0., 2., 0.)):
        _feed(research, {"x1": .5}, target, remaining=7-index)
        if index < 3:
            assert research.to_dict()["phase"] == "monitoring"
    state = research.to_dict()
    assert state["phase"] == "validating"
    event, = [item for item in state["events"] if item["action"] == "reopen_hypothesis_research"]
    assert event["stage"] == 8 and event["observation_indices"] == [4, 5, 6, 7]
    assert event["window"]["anomalous_indices"] == [4, 6]
    assert event["window"]["default_mse"] == event["window"]["learned_mse"] == 1.25
    assert state["costs"]["confirmation_calls"] == 0
    assert not any(item["action"] == "request_counterexample_confirmation" for item in state["events"])


def test_one_arbitrarily_large_outlier_does_not_reopen_a_stable_process():
    research = JointModelRevision(_spec())
    for _ in range(4):
        _feed(research, {"x1": .5}, 0.)
    for target in (1e6, 0., 0., 0., 0., 0., 0., 0.):
        _feed(research, {"x1": .5}, target)
    state = research.to_dict()
    assert state["phase"] == "monitoring"
    assert not state["episodes"] and research.next_control() is None
    assert state["costs"]["coverage_calls"] == state["costs"]["validation_calls"] == 0


def test_new_scope_retains_easy_points_after_failure_but_excludes_earlier_normal_points():
    research = JointModelRevision(_spec())
    for _ in range(4):
        _feed(research, {"x1": .5}, 0.)
    for index, target in enumerate((0., 1., 0., 2.)):
        _feed(research, {"x1": .5}, target, remaining=8-index)
    state = research.to_dict()
    episode, = state["episodes"]
    assert episode["scope_start"] == 5
    assert episode["trigger_indices"] == episode["row_indices"] == [5, 6, 7]
    assert episode["trigger_window"]["observation_indices"] == [4, 5, 6, 7]
    assert len(state["observations"]) == 8


def test_sliding_evidence_reopens_when_failures_cross_an_old_batch_boundary():
    research = JointModelRevision(_spec())
    for _ in range(4):
        _feed(research, {"x1": .5}, 0.)
    for index, target in enumerate((0., 0., 0., 1., 0., 2.)):
        _feed(research, {"x1": .5}, target, remaining=10-index)
        if index < 5:
            assert research.to_dict()["phase"] == "monitoring"
    event, = [item for item in research.to_dict()["events"] if item["action"] == "reopen_hypothesis_research"]
    assert event["stage"] == 10 and event["scope_start"] == 7
    assert event["observation_indices"] == [7, 8, 9]


def test_fitted_hypothesis_needs_four_future_outcomes_before_support():
    research = JointModelRevision(_spec())
    response = lambda x: 1. + x * x
    _trigger(research, response)
    _collect(research, response)
    proposed = research.candidate.to_dict()
    competitors = research.to_dict()["competitors"]
    assert research.accepted_revision is None
    for index in range(4):
        features = research.next_control()
        _feed(research, features, response(features["x1"]))
        assert research.candidate.to_dict() == proposed
        assert research.to_dict()["competitors"] == competitors
        if index < 3:
            assert research.accepted_revision is None
    accepted = research.accepted_revision
    assert accepted["program"].predict(Observation({"x1": .37}, 0.)) == pytest.approx(response(.37), abs=1e-7)
    assert not set(accepted["training_indices"]) & set(accepted["validation_indices"])
    assert len(accepted["validation_indices"]) == 4
    assert research.to_dict()["phase"] == "awaiting_rebase"
    state = research.to_dict()
    measured = state["episodes"][-1]["proposals"][-1]["validation"]["competitors"]
    assert {item["candidate_id"] for item in measured} == set(research.candidates)
    assert len(measured) > 1 and state["costs"]["validation_calls"] == 4
    assert all(len(state["observations"][index]["candidate_predictions"]) == len(measured)
               for index in accepted["validation_indices"])


def test_failed_future_predictions_do_not_become_an_accepted_replacement():
    research = JointModelRevision(_spec())
    response = lambda x: 1. + x * x
    _trigger(research, response)
    _collect(research, response)
    for _ in range(4):
        features = research.next_control()
        _feed(research, features, response(features["x1"]) + 2., remaining=0)
    assert research.accepted_revision is None
    assert research.to_dict()["phase"] == "unresolved"
    assert research.to_dict()["episodes"][-1]["status"] == "replacement_not_supported"
    assert not research.to_dict()["episodes"][-1]["proposals"][-1]["validation"]["supported"]


def test_old_counterexamples_cannot_be_erased_by_an_easy_validation_batch():
    research = JointModelRevision(_spec())
    for _ in range(4):
        _feed(research, {"x1": .5}, 0.)
    _feed(research, {"x1": .5}, 1.)
    _feed(research, research.next_control(), 3.)
    _collect(research, lambda x: 2.)
    assert research.to_dict()["candidate"]["training_loss"] > .1
    for _ in range(4):
        _feed(research, research.next_control(), 2., remaining=0)
    assert research.accepted_revision is None
    assert research.to_dict()["phase"] == "unresolved"


def test_insufficient_query_budget_keeps_the_question_open_without_proposing():
    research = JointModelRevision(_spec())
    _trigger(research, remaining=3)
    state = research.to_dict()
    assert state["phase"] == "unresolved"
    assert state["episodes"][-1]["status"] == "insufficient_query_budget"
    assert research.next_control() is None and research.accepted_revision is None
    assert not state["episodes"][-1]["proposals"]
    assert state["costs"]["coverage_calls"] == state["costs"]["validation_calls"] == 0


def test_insufficient_compute_preserves_paid_evidence_without_pretending_to_fit():
    research = JointModelRevision(_spec(), compute_budget=1)
    _trigger(research)
    _collect(research, lambda x: 1. + x*x)
    state = research.to_dict()
    assert state["phase"] == "unresolved"
    assert state["episodes"][-1]["status"] == "insufficient_compute_budget"
    assert state["costs"]["compute_cost"] <= 1 and state["costs"]["coverage_calls"] == 4
    assert research.accepted_revision is None and research.candidate is None


def test_eight_remaining_measurements_can_research_a_low_dimensional_response():
    research = JointModelRevision(_spec(3))
    response = lambda point: 1. + point["x2"] ** 2
    rng = Random(81)
    for _ in range(4):
        _feed(research, {name: rng.random() for name in research.spec.feature_names}, 0.)
    point = {name: rng.random() for name in research.spec.feature_names}
    _feed(research, point, response(point), remaining=8)
    assert research.control_purpose == "confirmation"
    frozen = None
    for index in range(8):
        point = research.next_control()
        _feed(research, point, response(point), remaining=7-index)
        if index == 0:
            episode, = research.to_dict()["episodes"]
            assert episode["planned_coverage"] == 3 and episode["reserved_validation"] == 4
        if index == 3:
            frozen = research.candidates
            assert frozen and research.accepted_revision is None
        if index >= 3:
            assert research.candidates == frozen
    accepted = research.accepted_revision
    assert accepted is not None
    costs = research.to_dict()["costs"]
    assert costs["confirmation_calls"] == 1 and costs["coverage_calls"] == 3
    assert costs["validation_calls"] == 4
    assert len(research.to_dict()["observations"]) == 13  # Four references, anchor and exactly eight paid calls.
    assert costs["compute_cost"] > 0
    for point in ({"x1": .91, "x2": .23, "x3": .81},
                  {"x1": .04, "x2": .77, "x3": .13}):
        assert accepted["program"].predict(Observation(point, 0.)) == pytest.approx(response(point), abs=1e-6)


def test_goal_validation_uses_current_scope_objective_evidence_and_keeps_four_future_measurements():
    spec = replace(_spec(3), context={"objective_direction": "minimize", "constraint_upper_bound": "0.0"})
    research = JointModelRevision(spec)
    rng = Random(813)
    for _ in range(32):
        point = {name: rng.random() for name in spec.feature_names}
        _feed(research, point, 0., objective=point["x2"])
    response = lambda point: -1. + .1 * point["x3"]
    point = {"x1": .2, "x2": .3, "x3": .4}
    _feed(research, point, response(point), objective=-point["x2"])
    while research.control_purpose in {"confirmation", "coverage"}:
        point = research.next_control()
        _feed(research, point, response(point), objective=-point["x2"])
    assert research.control_purpose == "validation"
    frozen = research.candidates
    state = research.to_dict()
    episode, = state["episodes"]
    assert episode["scope_start"] == 32
    proposal, = episode["proposals"]
    plan = proposal["experiment_plan"]
    assert plan["training_indices"] == proposal["training_indices"]
    assert min(plan["training_indices"]) == 32
    assert plan["objective_model"]["training_rows"] == len(plan["training_indices"])
    assert sum(item["purpose"] == "hypothesis_disagreement" for item in plan["selected"]) >= 2
    assert research.next_control()["x2"] > .99
    before = len(state["observations"])
    for index in range(4):
        point = research.next_control()
        _feed(research, point, response(point), objective=-point["x2"])
        assert research.candidates == frozen
        if index < 3:
            assert research.accepted_revision is None
    completed = research.to_dict()
    assert completed["costs"]["validation_calls"] == 4
    assert len(completed["observations"]) == before + 4
    assert research.accepted_revision is not None
    assert research.accepted_revision["training_indices"] == proposal["training_indices"]
    assert not set(proposal["training_indices"]) & set(research.accepted_revision["validation_indices"])


@pytest.mark.parametrize("resource", ["loo", "planning"])
def test_incomplete_model_comparison_or_experiment_planning_leaves_research_unresolved(resource):
    from hegel_machine.cognition.revision_candidates import propose_competitors
    from hegel_machine.cognition.workspace import observation_from_dict

    research = JointModelRevision(_spec())
    _trigger(research)
    while len(research.to_dict()["controls"]) > 1:
        point = research.next_control()
        _feed(research, point, 1. + point["x1"] ** 2)
    state = research.to_dict()
    point = research.next_control()
    last = Observation(point, 1. + point["x1"] ** 2)
    rows = [observation_from_dict(state["observations"][i]["observation"])
            for i in state["episodes"][-1]["row_indices"]] + [last]
    measured_budget = SearchBudget(20000)
    assert propose_competitors(research.spec, rows, measured_budget)
    # Enough to compare all fitted procedures leaves no units for selecting
    # distinguishing controls. Half that cost cannot complete all LOO folds.
    state["compute_budget"] = measured_budget.spent if resource == "planning" else measured_budget.spent // 2
    limited = JointModelRevision(research.spec, state=state)
    _feed(limited, point, last.target)
    result = limited.to_dict()
    assert result["phase"] == "unresolved"
    reason = "insufficient_experiment_planning_budget" if resource == "planning" else "insufficient_compute_budget"
    assert result["episodes"][-1]["status"] == reason
    assert 0 < result["costs"]["compute_cost"] <= state["compute_budget"]
    assert result["costs"]["coverage_calls"] == 4 and result["costs"]["validation_calls"] == 0
    assert limited.accepted_revision is None and limited.next_control() is None
    assert len(result["observations"]) == len(state["observations"]) + 1


def test_json_resume_preserves_the_frozen_candidate_and_pending_validation(tmp_path):
    research = JointModelRevision(_spec())
    response = lambda x: 1. + x*x
    _trigger(research)
    _collect(research, response)
    for _ in range(2):
        features = research.next_control()
        _feed(research, features, response(features["x1"]))
    path = tmp_path / "pending_revision.json"
    path.write_text(json.dumps(research.to_dict()))
    restored = JointModelRevision(_spec(), state=json.loads(path.read_text()))
    assert restored.candidate == research.candidate
    assert restored.candidates == research.candidates
    for _ in range(2):
        features = research.next_control()
        assert features == restored.next_control()
        for controller in (research, restored):
            _feed(controller, features, response(features["x1"]))
    assert json.loads(json.dumps(restored.to_dict())) == json.loads(json.dumps(research.to_dict()))
    assert restored.to_dict()["costs"]["validation_calls"] == 4


def test_a_second_common_failure_can_create_a_new_research_scope():
    research = JointModelRevision(_spec())
    first = lambda x: 1. + x*x
    _trigger(research)
    _collect(research, first)
    _confirm(research, first)
    accepted = research.accepted_revision["program"]
    research.acknowledge_revision()
    for x in (.1, .3, .6, .9):
        prediction = accepted.predict(Observation({"x1": x}, 0.))
        _feed(research, {"x1": x}, first(x), prediction, prediction)
    second = lambda x: -1. - 2*x*x
    point = {"x1": .1}
    prediction = accepted.predict(Observation(point, 0.))
    _feed(research, point, second(point["x1"]), prediction, prediction)
    point = research.next_control()
    prediction = accepted.predict(Observation(point, 0.))
    _feed(research, point, second(point["x1"]), prediction, prediction)
    assert len(research.to_dict()["episodes"]) == 2
    _collect(research, second)
    _confirm(research, second)
    replacement = research.accepted_revision
    assert replacement["program"].predict(Observation({"x1": .37}, 0.)) == pytest.approx(second(.37), abs=1e-7)
    assert research.to_dict()["episodes"][0]["status"] == "supported"
    assert replacement["scope_start"] > research.to_dict()["episodes"][0]["scope_start"]


def test_supported_optimizer_revision_returns_to_callable_knowledge_without_erasing_old_prior():
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.memory import KnowledgeLibrary
    from hegel_machine.cognition.meta_prior_session import MetaPriorSession
    from hegel_machine.cognition.models import Candidate

    spec = _spec()
    library = KnowledgeLibrary()
    initial = [Observation({"x1": x}, 0.) for x in (.1, .3, .6, .9)]
    old_id, _ = library.learn(Candidate("old_response", Program("const", value=0.), "meta_prior"),
                             spec, initial)
    old_prior = {"prior_id": "meta_prior_0001", "source_concept_id": old_id,
                 "task_id": spec.task_id, "family_id": "T02", "kind": "affine",
                 "status": "supported", "counterexamples": [], "source_features": ["x1"]}
    library.meta_priors.append(old_prior)
    research = JointModelRevision(spec)
    response = lambda x: 1. + x*x
    _trigger(research)
    _collect(research, response)
    _confirm(research, response)
    accepted = research.accepted_revision["program"]
    research.acknowledge_revision()
    session = MetaPriorSession(library=library)
    update, = session._retain_optimizer_revisions(spec, research.to_dict(), {old_id})
    new_id = update["source_concept_id"]
    assert new_id != old_id and update["created_concept"]
    assert old_prior in library.meta_priors and old_prior["status"] == "conditional"
    failure, = old_prior["counterexamples"]
    assert failure["replacement_concept_id"] == new_id and len(failure["observations"]) == 2
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    assert {old_id, new_id} <= {item.concept_id for item in restored.concepts}
    prior, = [item for item in restored.meta_priors if item["source_concept_id"] == new_id]
    assert prior["status"] == "supported" and "deployment_scope_" in prior["task_id"]
    assert prior["parent_concept_ids"] == [old_id] and prior["validation_loss"] < 1e-7
    calls = restored.callable_atoms(spec, structural_transfer=False, limit=len(restored.definitions))
    call = next(item for item in calls if item.definition.source_concept_id == new_id
                and item.expanded().simplified() == accepted.expanded().simplified())
    assert call.predict(Observation({"x1": .37}, 0.)) == pytest.approx(response(.37), abs=1e-7)
