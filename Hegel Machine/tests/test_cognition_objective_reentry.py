"""Reconsider an archived goal explanation using genuinely later predictions.

Controlled future labels test decisions; saved local-failure evidence supplies
one real continuation. No simulator labels are acquired by this file.
"""

from copy import deepcopy
from dataclasses import replace
import json
from math import sqrt
from pathlib import Path
from random import Random

import pytest

from hegel_machine.cognition.decision_experiments import _predict
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, Probe, SearchBudget, TaskSpec
from hegel_machine.cognition.objective_research import ObjectiveResearch, objective_program


def _row(point, target, index):
    return {"observation": {"features": deepcopy(point), "target": -1.},
            "context": {"global_index": index, "stage": index + 1, "objective": target}}


def _rejected_goal():
    spec = TaskSpec("rejected_goal", "Reconsider a measured objective", ("x1", "x2"), (), (),
        feature_bounds={"x1": (0., 1.), "x2": (0., 1.)},
        context={"objective_direction": "minimize", "constraint_upper_bound": "0.0"})
    goal = ObjectiveResearch(spec)
    candidate = {"degree": 1, "coefficients": [0., 1., 0.], "cv_loss": .0001,
                 "reference_loss": .1, "training_rows": 4, "training_end": 3,
                 "folds": [{"degree": 1, "coefficients": [0., 1., 0.]} for _ in range(3)]}
    points = [{"x1": .1 + .1 * index, "x2": .1} for index in range(8)]
    rows = [_row(point, point["x1"], index) for index, point in enumerate(points)]
    validation = [{"features": deepcopy(point), "global_index": index,
                   "target": point["x1"] + (1. if index == 4 else 0.),
                   "candidate_error": (1. if index == 4 else 0.),
                   "previous_error": (0. if index == 4 else .2)}
                  for index, point in enumerate(points[4:], start=4)]
    archive = {"scope_start": 0, "candidate": candidate, "phase": "completed",
               "status": "future_predictions_do_not_improve", "training_indices": [0, 1, 2, 3],
               "validation": validation, "validation_target": 4,
               "validation_rounds": [{"stage": 8, "count": 4, "decision": "rejected"}],
               "coverage_calls": 0}
    goal.state.update(scope_start=0, evidence_start=0, last_monitored=7, plan_rows=8,
                      scope_attempts=[archive])
    return goal, rows


def _future(goal, rows, *, gain, remaining=16, scope_start=0, point=None):
    point = point or {"x1": .11 + .04 * (len(rows) - 8), "x2": .73}
    archive = goal.state["scope_attempts"][-1]
    candidate = ((goal.state["scope_research"] or {}).get("candidate") or archive["candidate"])
    candidate_mean = _predict(goal.spec, candidate, point)
    target = candidate_mean + sqrt(max(0., -gain))
    previous = target + sqrt(max(0., gain))
    index = len(rows)
    request = goal.prepare_shared_control(rows, lambda features: {
        "features": deepcopy(features), "objective_mean": previous, "objective_noise": .001},
        point, scope_start=scope_start, stage=index, remaining_calls=remaining,
        purpose="optimization")
    assert request is not None and request["expected_global_index"] == index
    context = {"global_index": index, "stage": index + 1, "purpose": "optimization"}
    rows.append(_row(point, target, index))
    goal.observe(point, [target, -1.], context=context)
    return request, target, context


def test_reentry_keeps_the_archive_and_waits_for_the_next_label_before_comparing():
    goal, rows = _rejected_goal()
    archive, fits = deepcopy(goal.state["scope_attempts"]), deepcopy(goal.state["fits"])
    point = {"x1": .17, "x2": .73}
    request = goal.prepare_shared_control(rows, lambda features: {
        "features": deepcopy(features), "objective_mean": 1.17, "objective_noise": .001},
        point, scope_start=0, stage=8, remaining_calls=3, purpose="optimization")
    assert request is not None and request["candidate_mean"] == pytest.approx(.17)
    research = goal.state["scope_research"]
    assert research["reentry"]["attempt_index"] == 0
    assert research["reentry"]["representation_revised"] is False
    assert research["candidate"] == archive[0]["candidate"]
    assert research["validation"] == archive[0]["validation"]
    assert research["training_indices"] == archive[0]["training_indices"]
    assert goal.state["scope_attempts"] == archive and goal.state["fits"] == fits
    assert goal.accepted_revision is None and not goal.state["shared_outcomes"]
    assert len(rows) == 8


def test_later_support_can_rehabilitate_the_same_candidate_without_erasing_its_losses():
    goal, rows = _rejected_goal()
    archive = deepcopy(goal.state["scope_attempts"])
    for _ in range(8):
        _future(goal, rows, gain=1.)
        if goal.accepted_revision is not None:
            break
    accepted = goal.accepted_revision
    assert accepted is not None and accepted["model"] == archive[0]["candidate"]
    assert goal.state["scope_attempts"] == archive
    assert goal.state["scope_research"]["validation"][:4] == archive[0]["validation"]
    assert accepted["validation_indices"][:4] == [4, 5, 6, 7]
    assert all(index >= 4 for index in accepted["validation_indices"])
    assert set(accepted["training_indices"]).isdisjoint(accepted["validation_indices"])
    assert not goal.state["fits"]


def test_a_still_bad_candidate_stays_unadopted_without_repeated_archive_cycles():
    goal, rows = _rejected_goal()
    archive = deepcopy(goal.state["scope_attempts"])
    for _ in range(3):
        _future(goal, rows, gain=-1.)
        assert goal.accepted_revision is None
    assert goal.state["scope_attempts"][:1] == archive
    research = goal.state["scope_attempts"][-1]
    assert research["candidate"] == archive[0]["candidate"]
    assert research["reentry"]["attempt_index"] == 0
    assert len(research["validation"]) == 7 and goal.state["scope_research"] is None
    for _ in range(3):
        assert goal.prepare_shared_control(rows,
            lambda features: pytest.fail("a refuted reentry cannot restart the same contest"),
            {"x1": .93, "x2": .73}, scope_start=0, stage=len(rows),
            remaining_calls=10, purpose="optimization") is None
    assert len(goal.state["scope_attempts"]) == 2 and len(goal.state["reentry_attempts"]) == 1


def test_strong_existing_refutation_does_not_force_research_without_a_better_representation():
    goal, rows = _rejected_goal()
    for row in goal.state["scope_attempts"][0]["validation"]:
        row.update(candidate_error=1., previous_error=0.)
    archive = deepcopy(goal.state["scope_attempts"])
    assert goal.prepare_shared_control(rows,
        lambda features: pytest.fail("already decisive losses do not create an experiment"),
        {"x1": .17, "x2": .73}, scope_start=0, stage=8, remaining_calls=3,
        purpose="optimization") is None
    assert goal.state["scope_research"] is None and goal.accepted_revision is None
    assert goal.state["scope_attempts"] == archive
    assert goal.state["reentry_attempts"][-1]["status"] == "refuted_without_supported_representation"


def test_reentry_survives_json_and_a_shared_vector_is_counted_once():
    goal, rows = _rejected_goal()
    request, target, context = _future(goal, rows, gain=-1.)
    state = json.loads(json.dumps(goal.to_dict()))
    restored = ObjectiveResearch(goal.spec, state=state)
    assert restored.to_dict() == state
    restored.observe(request["features"], [target, -1.], context=context)
    assert restored.to_dict() == state
    _future(restored, rows, gain=1.)
    assert len(restored.state["shared_outcomes"]) == 2
    assert len(restored.state["scope_attempts"]) == 1


@pytest.mark.parametrize("scope_start", [0, 6])
def test_constraint_scope_change_does_not_rewrite_the_frozen_objective_scope(scope_start):
    goal, rows = _rejected_goal()
    archive = deepcopy(goal.state["scope_attempts"])
    _future(goal, rows, gain=-1., remaining=3, scope_start=scope_start)
    assert goal.state["scope_start"] == scope_start
    assert goal.state["scope_research"]["scope_start"] == 0
    assert goal.state["scope_research"]["candidate"] == archive[0]["candidate"]
    assert goal.state["scope_attempts"] == archive and not goal.state["fits"]


@pytest.mark.parametrize("exhausted", ["calls", "compute"])
def test_exhausted_budget_cannot_fit_or_acquire_reentry_evidence(exhausted):
    goal, rows = _rejected_goal()
    archive = deepcopy(goal.state["scope_attempts"])
    remaining = 0 if exhausted == "calls" else 3
    if exhausted == "compute":
        goal.state["compute_cost"] = goal.state["compute_budget"]
    request = goal.prepare_shared_control(rows,
        lambda features: pytest.fail("unfunded reentry cannot forecast an observation"),
        {"x1": .17, "x2": .73}, scope_start=0, stage=8,
        remaining_calls=remaining, purpose="optimization")
    assert request is None and goal.accepted_revision is None
    assert goal.state["scope_attempts"] == archive and not goal.state["fits"]
    assert goal.state["compute_cost"] <= goal.state["compute_budget"]
    assert not goal.state["shared_outcomes"]


def test_same_quantized_validation_control_cannot_create_new_reentry_support():
    goal, rows = _rejected_goal()
    archive = deepcopy(goal.state["scope_attempts"])
    repeated = archive[0]["validation"][0]["features"]
    request = goal.prepare_shared_control(rows, lambda features: {
        "features": deepcopy(repeated), "objective_mean": 1., "objective_noise": .001},
        {"x1": repeated["x1"] + .001, "x2": .1}, scope_start=0, stage=8,
        remaining_calls=3, purpose="optimization")
    assert request is None
    assert goal.state["scope_research"]["validation"] == archive[0]["validation"]
    assert goal.state["scope_attempts"] == archive and goal.accepted_revision is None


def _interaction_rows():
    goal, _ = _rejected_goal()
    rng = Random(7301)
    points = [{"x1": rng.uniform(.05, .95), "x2": rng.uniform(.05, .95)} for _ in range(32)]
    rows = [_row(point, 1. + .5 * point["x1"] - .2 * point["x2"]
                 + 2. * point["x1"] * point["x2"], index)
            for index, point in enumerate(points)]
    concept = Program("mul", children=(Program("feature", name="x1"),
                                       Program("feature", name="x2")))
    return goal, rows, concept


def test_constraint_concept_fit_and_compiled_program_make_the_same_new_predictions():
    from hegel_machine.cognition.objective_reentry import fit_repaired_objective
    goal, rows, concept = _interaction_rows()
    # These old-scope labels are deliberately incompatible with the new law.
    for row in rows[:6]:
        row["context"]["objective"] += 100.
    budget = SearchBudget(20000)
    model = fit_repaired_objective(goal.spec, rows,
        {"program": concept.to_dict(), "scope_start": 6}, budget)
    assert model is not None and model["concept"] == concept.to_dict()
    assert model["training_indices"] == list(range(6, 32)) and model["training_end"] == 31
    assert model["cv_loss"] < 1e-10 < model["base_cv_loss"]
    assert 0 < budget.spent <= budget.limit
    compiled = objective_program(goal.spec, model)
    for point in ({"x1": .12, "x2": .77}, {"x1": .86, "x2": .32}):
        target = 1. + .5 * point["x1"] - .2 * point["x2"] + 2. * point["x1"] * point["x2"]
        assert _predict(goal.spec, model, point) == pytest.approx(target, abs=1e-6)
        assert compiled.predict(Observation(point, 0.)) == pytest.approx(
            _predict(goal.spec, model, point), abs=1e-12)


@pytest.mark.parametrize("allowance", [0, 150])
def test_incomplete_representation_comparison_does_not_return_a_partly_evaluated_candidate(allowance):
    from hegel_machine.cognition.objective_reentry import fit_repaired_objective
    goal, rows, concept = _interaction_rows()
    before = deepcopy(rows)
    budget = SearchBudget(allowance)
    assert fit_repaired_objective(goal.spec, rows,
        {"program": concept.to_dict(), "scope_start": 0}, budget) is None
    assert budget.spent <= allowance and rows == before


@pytest.mark.parametrize("remaining", [3, 4])
def test_registered_concept_changes_representation_only_after_rejection_with_a_future_batch(remaining):
    goal, rows, concept = _interaction_rows()
    archive = deepcopy(goal.state["scope_attempts"])
    current_model = {"degree": 1, "coefficients": [1., .5, -.2], "training_end": 31,
                     "cv_loss": 1., "folds": []}
    goal.state["model"] = deepcopy(current_model)
    before_cost = goal.state["compute_cost"]
    goal.register_constraint_concept(concept.to_dict(), scope_start=6, stage=31)
    assert goal.state["scope_research"] is None and goal.accepted_revision is None
    assert goal.state["model"] == current_model and goal.state["scope_attempts"] == archive
    assert goal.state["compute_cost"] == before_cost and not goal.state["fits"]
    point = {"x1": .91, "x2": .83}
    request = goal.prepare_shared_control(rows, lambda features: {
        "features": deepcopy(features), "objective_mean": 5., "objective_noise": .001},
        point, scope_start=6, stage=32, remaining_calls=remaining, purpose="optimization")
    assert request is not None
    research = goal.state["scope_research"]
    assert research["reentry"]["representation_revised"] is (remaining == 4)
    assert goal.state["model"] == current_model and goal.accepted_revision is None
    assert goal.state["scope_attempts"] == archive
    if remaining == 3:
        assert research["candidate"] == archive[0]["candidate"]
        assert research["validation"] == archive[0]["validation"]
        assert "concept" not in research["candidate"]
    else:
        assert research["candidate"]["concept"] == concept.to_dict()
        assert research["scope_start"] == 6
        assert research["training_indices"] == list(range(6, 32))
        assert research["validation"] == [] and research["validation_target"] == 4
        assert request["expected_global_index"] == 32
        assert goal.state["compute_cost"] > before_cost


def test_merely_registering_a_concept_does_not_start_research_without_a_rejected_hypothesis():
    goal, rows, concept = _interaction_rows()
    goal.state["scope_attempts"] = []
    before = goal.to_dict()
    goal.register_constraint_concept(concept.to_dict(), scope_start=6, stage=31)
    assert goal.prepare_shared_control(rows,
        lambda point: pytest.fail("an unrelated registered concept does not start optimization research"),
        {"x1": .91, "x2": .83}, scope_start=6, stage=32, remaining_calls=4,
        purpose="optimization") is None
    for key in ("model", "fits", "scope_research", "scope_revisions", "compute_cost"):
        assert goal.state[key] == before[key]


def test_reentry_reaches_a_distinguishing_public_control_beyond_four_local_choices(monkeypatch):
    import hegel_machine.cognition.objective_reentry as reentry
    goal, rows = _rejected_goal()
    local = [{"x1": .01 * index, "x2": 0.} for index in range(1, 5)]
    request_point = {"x1": .913, "x2": .837}
    actual = {"x1": .91, "x2": .84}
    goal.spec = replace(goal.spec, available_probes=(Probe("new_region", (request_point,)),))
    monkeypatch.setattr(reentry, "objective_pool",
                        lambda spec, model, budget=None: deepcopy(local))
    seen = []

    def forecast(point):
        seen.append(deepcopy(point))
        selected = actual if point == request_point else point
        gap = 10. if point == request_point else .01
        return {"features": deepcopy(selected), "objective_mean": selected["x1"] + gap,
                "objective_noise": .001, "feasible": True}

    request = goal.next_control(rows, forecast, scope_start=0, stage=8, remaining_calls=3)
    assert all(point in seen for point in [*local, request_point])
    assert request["features"] == actual
    assert request["candidate_mean"] == pytest.approx(actual["x1"])
    assert request["previous_mean"] == pytest.approx(actual["x1"] + 10.)
    assert request["reentry"] is True and goal.accepted_revision is None
    assert goal.state["scope_research"]["validation"] == goal.state["scope_attempts"][0]["validation"]


def test_repaired_representation_earns_adoption_only_after_four_new_vector_labels():
    goal, rows, concept = _interaction_rows()
    archive = deepcopy(goal.state["scope_attempts"])
    goal.register_constraint_concept(concept.to_dict(), scope_start=6, stage=31)
    frozen = None
    for index in range(4):
        point = {"x1": .11 + .17 * index, "x2": .17 + .13 * index}
        target = 1. + .5 * point["x1"] - .2 * point["x2"] + 2. * point["x1"] * point["x2"]
        request = goal.prepare_shared_control(rows, lambda features: {
            "features": deepcopy(features), "objective_mean": target + 1., "objective_noise": .001},
            point, scope_start=6, stage=32 + index, remaining_calls=4 - index, purpose="optimization")
        assert request is not None and goal.accepted_revision is None
        research = goal.state["scope_research"]
        if frozen is None:
            frozen = deepcopy(research["candidate"])
        assert research["candidate"] == frozen and len(research["validation"]) == index
        rows.append(_row(point, target, 32 + index))
        goal.observe(point, [target, -1.], context={"global_index": 32 + index,
            "stage": 33 + index, "purpose": "optimization"})
        if index < 3:
            assert goal.accepted_revision is None
    revision = goal.accepted_revision
    assert revision is not None and revision["model"] == frozen
    assert revision["training_indices"] == list(range(6, 32))
    assert revision["validation_indices"] == list(range(32, 36))
    assert revision["reentry"]["representation_revised"] is True
    assert goal.state["scope_attempts"] == archive
    assert len(goal.state["shared_outcomes"]) == 4


def test_saved_late_local_rejection_can_observe_the_remaining_uncovered_control():
    root = Path(__file__).resolve().parents[1]
    source = root / ("artifacts/sc_kg_constraint_retention_v051_20260911/passive_mixture/runs/"
        "independent_goal_1_mechanism_change_noise_0.03_constraint_retention_passive_v051.json")
    saved = json.loads(source.read_text())
    names = ("x1", "x2", "x3")
    spec = TaskSpec("saved_reentry", "Reconsider the rejected objective", names, (), (),
        feature_bounds={name: (0., 1.) for name in names},
        context={"objective_direction": "minimize", "constraint_upper_bound": "0.0"})
    goal = ObjectiveResearch(spec, state=saved["objective_research"])
    archive = deepcopy(goal.state["scope_attempts"])
    fits = deepcopy(goal.state["fits"])
    records = list(zip(saved["initial_points"], saved["initial_outputs"]))
    records.extend((item["point"], item["output"]) for item in saved["simulations"][:29])
    rows = [_row(dict(zip(names, (value / 100. for value in point))), output[0], index)
            for index, (point, output) in enumerate(records)]
    assert len(rows) == 61 and goal.state["scope_research"] is None
    new_comparisons = []
    for global_index in range(61, 64):
        observed = saved["simulations"][global_index - 32]
        features = dict(zip(names, (value / 100. for value in observed["point"])))
        prior_mean = saved["actions"][global_index - 32]["mu_before"][0]
        request = goal.prepare_shared_control(rows, lambda point: {
            "features": deepcopy(point), "objective_mean": prior_mean, "objective_noise": .001},
            features, scope_start=goal.state["scope_start"], stage=global_index,
            remaining_calls=64 - global_index, purpose="optimization")
        if request is not None:
            assert request["expected_global_index"] == global_index
            assert request["candidate_mean"] == pytest.approx(
                _predict(spec, archive[0]["candidate"], features))
            new_comparisons.append(global_index)
        rows.append(_row(features, observed["output"][0], global_index))
        goal.observe(features, observed["output"], context={"global_index": global_index,
            "stage": global_index + 1, "purpose": "optimization"})
    # The first two purchased controls were already used against this GP.
    assert new_comparisons == [63]
    assert goal.state["scope_attempts"][:1] == archive and goal.state["fits"] == fits
    assert goal.state["reentry_attempts"][-1]["representation_revised"] is False
    assert len(rows) == 64
