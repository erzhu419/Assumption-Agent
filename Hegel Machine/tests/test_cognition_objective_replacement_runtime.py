"""Objective replacement competes on predictions made before paid labels.

Saved SC-KG histories initialize actual GP and controller states. Controlled
future labels exercise the selection rule; no test buys a simulator draw.
The final three goal-only scope cases load rejected revision 4 explicitly;
the other runtime cases exercise the production optimizer.
"""

from copy import deepcopy
import importlib.util
import json
from math import sqrt
from pathlib import Path

import numpy as np
import pytest

from test_cognition_sc_kg_environment import project_root, _saved_terminal_failure_optimizer
from test_cognition_objective_incumbent import (
    CASE, _paused_comparison, saved_incumbent_failure,
)
from hegel_machine.cognition.decision_experiments import _predict
from hegel_machine.cognition.objective_research import ObjectiveResearch
from hegel_machine.cognition.objective_replacement import ObjectiveReplacement
from hegel_machine.cognition.validation_continuation import evidence_summary


def _live_goal(saved):
    old, rows, label = _paused_comparison(saved)
    return ObjectiveResearch(old.spec, state=old.to_dict()), rows, label


def _comparison_rows(goal, count, gain):
    # These stand for predictions stored before these already observed labels.
    rows = deepcopy(goal.state["scope_research"]["validation"][:count])
    for row in rows:
        row.update(candidate_error=max(0., -gain), previous_error=max(0., gain))
    return rows


def _future_label(goal, rows, *, gain=1., point=None):
    index = len(rows)
    point = point or {"x1": .13 + .01 * (index - 58), "x2": .27, "x3": .71}
    candidate_mean = _predict(goal.spec, goal.state["scope_research"]["candidate"], point)
    target = candidate_mean + sqrt(max(0., -gain))
    previous_mean = target + sqrt(max(0., gain))
    request = goal.prepare_shared_control(rows, lambda features: {
        "features": deepcopy(features), "objective_mean": previous_mean,
        "objective_noise": .001}, point, scope_start=44, stage=index,
        remaining_calls=10, purpose="optimization")
    assert request is not None and request["expected_global_index"] == index
    context = {"global_index": index, "stage": index + 1,
               "purpose": "optimization", "objective": target}
    rows.append({"observation": {"features": deepcopy(point), "target": -1.},
                 "context": context})
    goal.observe(point, [target, -1.], context=context)


def test_supported_replacement_does_not_mix_old_advantage_into_current_comparison(
        saved_incumbent_failure):
    goal, rows, label = _live_goal(saved_incumbent_failure)
    before = deepcopy(goal.state["scope_research"])
    goal.incumbent_rebuilt(stage=58, scope_start=44)
    research = goal.state["scope_research"]
    assert research["candidate"] == before["candidate"]
    assert research["validation"] == before["validation"]
    assert research["training_indices"] == before["training_indices"]
    prediction = label["request"]
    goal.prepare_shared_control(rows, lambda features: {
        "features": deepcopy(features), "objective_mean": prediction["previous_mean"],
        "objective_noise": prediction["objective_noise"]}, prediction["features"],
        scope_start=44, stage=58, remaining_calls=6, purpose="optimization")
    goal.observe(prediction["features"], [label["objective"], label["constraint"]],
                 context=label["context"])
    assert evidence_summary(research["validation"])["decision"] == "supported"
    assert research["validation"][-1]["candidate_error"] > research["validation"][-1]["previous_error"]
    assert goal.accepted_revision is None and research["phase"] == "validating"
    assert not goal.state["fits"]


@pytest.mark.parametrize("gain", [1., -1.])
def test_four_preobserved_comparisons_decide_against_actual_replacement(
        saved_incumbent_failure, gain):
    goal, _, _ = _live_goal(saved_incumbent_failure)
    before = deepcopy(goal.state["scope_research"])
    comparison = _comparison_rows(goal, 4, gain)
    goal.incumbent_rebuilt(stage=58, scope_start=44, comparison=comparison)
    research = goal.state["scope_research"] or goal.state["scope_attempts"][-1]
    assert research["validation"] == before["validation"]
    assert research["candidate"] == before["candidate"]
    assert research["validation_rounds"][-1]["count"] == 4
    if gain > 0:
        assert goal.accepted_revision["model"] == before["candidate"]
        assert goal.accepted_revision["validation_indices"] == [row["global_index"] for row in comparison]
        assert goal.accepted_revision["previous_mse"] == pytest.approx(1.)
    else:
        assert goal.accepted_revision is None
        assert research["status"] == "future_predictions_do_not_improve"
    assert not goal.state["fits"]


def test_preobserved_pair_plus_future_pair_survives_json_resume_without_refit(
        saved_incumbent_failure):
    goal, rows, _ = _live_goal(saved_incumbent_failure)
    comparison = _comparison_rows(goal, 2, 1.)
    frozen = deepcopy(goal.state["scope_research"]["candidate"])
    goal.incumbent_rebuilt(stage=58, scope_start=44, comparison=comparison)
    _future_label(goal, rows)
    assert goal.accepted_revision is None
    state = json.loads(json.dumps(goal.to_dict()))
    restored = ObjectiveResearch(goal.spec, state=state)
    assert restored.to_dict() == state
    _future_label(restored, rows)
    assert restored.accepted_revision["validation_indices"] == [
        *(row["global_index"] for row in comparison), 58, 59]
    assert restored.accepted_revision["model"] == frozen
    assert restored.accepted_revision["validation_rounds"][-1]["count"] == 4
    assert not restored.state["fits"]


def test_same_control_gets_one_fresh_comparison_per_incumbent(saved_incumbent_failure):
    goal, rows, _ = _live_goal(saved_incumbent_failure)
    point = deepcopy(goal.state["scope_research"]["validation"][0]["features"])
    goal.incumbent_rebuilt(stage=58, scope_start=44)
    _future_label(goal, rows, point=point)
    before = goal.to_dict()
    assert goal.prepare_shared_control(rows,
        lambda features: pytest.fail("a second current-opponent comparison cannot add evidence"),
        point, scope_start=44, stage=59, remaining_calls=5,
        purpose="optimization") is None
    assert goal.state["scope_research"] == before["scope_research"]
    assert goal.accepted_revision is None


def test_uncertain_current_four_need_two_more_labels_even_with_old_support(saved_incumbent_failure):
    goal, rows, _ = _live_goal(saved_incumbent_failure)
    comparison = _comparison_rows(goal, 4, 0.)
    comparison[0]["previous_error"] = 10.
    goal.incumbent_rebuilt(stage=58, scope_start=44, comparison=comparison)
    assert goal.accepted_revision is None
    assert goal.state["scope_research"]["status"] == "inconclusive_future_predictions"
    _future_label(goal, rows, gain=10.)
    assert goal.accepted_revision is None
    _future_label(goal, rows, gain=10.)
    assert goal.accepted_revision["validation_rounds"][-1]["count"] == 6
    assert goal.accepted_revision["validation_indices"] == [
        *(row["global_index"] for row in comparison), 58, 59]


def _runtime(project_root, monkeypatch, saved):
    algorithm, adapter, environment = _saved_terminal_failure_optimizer(
        project_root, monkeypatch, CASE, frozen_directory="sc_kg_active_noise_v046_20260910")
    monkeypatch.setattr(adapter, "simulate", lambda *args, **kwargs:
                        pytest.fail("replacement transition cannot buy simulator labels"))
    monkeypatch.setattr(environment, "observe", lambda *args, **kwargs:
                        pytest.fail("replacement transition cannot acquire research labels"))
    algorithm._fit_initial_belief(algorithm._initial_samples())
    goal, rows, _ = _live_goal(saved)
    algorithm.objective_research = goal
    goal.state["replacement_research"] = algorithm.objective_replacement.state
    _set_history(algorithm, rows)
    algorithm.recovery.state = deepcopy(saved["recovery"])
    algorithm.recovery.state["phase"] = "validating"
    monkeypatch.setattr(algorithm.recovery, "acknowledge_revision", lambda: None)
    return algorithm, adapter, goal, rows


def _set_history(algorithm, rows):
    names = algorithm.objective_research.spec.feature_names
    algorithm.history = [(tuple(int(round(row["observation"]["features"][name] * 100))
                                for name in names),
                          np.asarray([row["context"]["objective"], row["observation"]["target"]]))
                         for row in rows]
    algorithm.observations = {}
    for point, output in algorithm.history:
        algorithm.observations.setdefault(point, []).append(output)


def _variance_predictions(algorithm):
    points = ((0, 0, 0), (100, 0, 0), (50, 30, 70), (100, 100, 100))
    return {name: [variance.predict_variance(0, point, algorithm.problem) for point in points]
            for name, variance in algorithm.constraint_variances.items()}


def _raw_comparisons(algorithm, *, count=4, candidate_loss=None):
    """Existing paid labels plus controlled, previously made error contrasts."""
    replacement = algorithm.objective_replacement
    replacement.propose(scope_start=44, stage=54)
    tag = algorithm._goal_candidate_tag()
    for index, (point, output) in enumerate(algorithm.history[54:54 + count], start=54):
        features = dict(zip(algorithm.objective_research.spec.feature_names,
                            (value / 100. for value in point)))
        target = float(output[0])
        candidate = (None if candidate_loss is None else
                     {"tag": tag, "mean": target + sqrt(candidate_loss)})
        replacement.prepare(features, target + 1., target, stage=index, candidate=candidate)
        replacement.observe(features, target, global_index=index)


def _promote_shared_interaction(algorithm, goal, rows):
    research = goal.state["scope_research"]
    base = research["candidate"]
    challenger = deepcopy(base)
    challenger.update(interactions=[[0, 1], [0, 2], [1, 2]])
    challenger["coefficients"][0] += 1.
    challenger["coefficients"].extend([.2, .2, .2])
    base["interaction_challenger"] = challenger
    research.update(validation=[], validation_target=4, validation_rounds=[], phase="validating")
    for index in range(58, 62):
        point = {"x1": .13 + .01 * (index - 58), "x2": .27, "x3": .71}
        target = _predict(goal.spec, challenger, point)
        request = goal.prepare_shared_control(rows, lambda features: {
            "features": features, "objective_mean": target + 2., "objective_noise": .001},
            point, scope_start=44, stage=index, remaining_calls=64-index, purpose="optimization")
        assert request["interaction_mean"] == pytest.approx(target)
        context = {"global_index": index, "stage": index + 1,
                   "purpose": "optimization", "objective": target}
        rows.append({"observation": {"features": point, "target": -1.}, "context": context})
        goal.observe(point, [target, -1.], context=context)
    assert goal.accepted_revision["model"]["interactions"] == [[0, 1], [0, 2], [1, 2]]
    assert goal.accepted_revision["model"]["training_end"] == base["training_end"]
    _set_history(algorithm, rows)


def test_base_predictions_cannot_defeat_a_same_prefix_promoted_interaction(
        project_root, monkeypatch, saved_incumbent_failure):
    algorithm, adapter, goal, rows = _runtime(project_root, monkeypatch, saved_incumbent_failure)
    base_tag = algorithm._goal_candidate_tag()
    _raw_comparisons(algorithm, candidate_loss=1.)
    _promote_shared_interaction(algorithm, goal, rows)
    full_tag = algorithm._goal_candidate_tag(goal.accepted_revision)
    assert base_tag[:2] == full_tag[:2] and base_tag != full_tag
    replacement = ObjectiveReplacement(json.loads(json.dumps(algorithm.objective_replacement.to_dict())))
    algorithm.objective_replacement = replacement
    goal.state["replacement_research"] = replacement.state
    assert len(replacement.comparisons(base_tag)) == 4
    assert replacement.comparisons(full_tag) == []
    replacement.confirm(44, stage=62)
    accepted, incumbent = goal.accepted_revision, algorithm.gpr[0]
    algorithm._consider_objective_replacement(stage=62)
    assert algorithm.gpr[0] is incumbent and goal.accepted_revision == accepted
    decision = replacement.state["decisions"][-1]
    assert decision["evidence"]["decision"] == "supported"
    assert decision["candidate_evidence"]["count"] == 0 and not decision["replace"]
    assert adapter.simulator_calls == 0


def test_rebuilt_incumbent_does_not_project_old_base_errors_onto_promoted_interaction(
        project_root, monkeypatch, saved_incumbent_failure):
    algorithm, adapter, goal, rows = _runtime(project_root, monkeypatch, saved_incumbent_failure)
    _raw_comparisons(algorithm, candidate_loss=1.)
    _promote_shared_interaction(algorithm, goal, rows)
    before = deepcopy(goal.state["scope_research"]["validation"])
    algorithm._rebase_goal_comparison(stage=62, start=44)
    research = goal.state["scope_research"]
    assert research["validation"] == before and research["incumbent_validation"] == []
    assert goal._comparison_rows(research) == [] and goal.accepted_revision is None
    assert research["status"] == "awaiting_current_incumbent"
    assert algorithm.objective_diagnostics["replacement_installs"][-1]["prepared_comparisons"] == 0
    assert adapter.simulator_calls == 0


def test_interaction_replacement_comparison_survives_json_for_the_same_model(
        project_root, monkeypatch, saved_incumbent_failure):
    algorithm, adapter, goal, rows = _runtime(project_root, monkeypatch, saved_incumbent_failure)
    _promote_shared_interaction(algorithm, goal, rows)
    tag = algorithm._goal_candidate_tag(goal.accepted_revision)
    _raw_comparisons(algorithm, candidate_loss=1.)
    replacement = ObjectiveReplacement(json.loads(json.dumps(algorithm.objective_replacement.to_dict())))
    revision = json.loads(json.dumps(goal.accepted_revision))
    assert algorithm._goal_candidate_tag(revision) == tag
    assert len(replacement.comparisons(tag)) == 4
    replacement.confirm(44, stage=62)
    assert replacement.decide(stage=62, accepted_candidate=tag)["replace"]
    assert adapter.simulator_calls == 0


@pytest.mark.parametrize("comparisons", [0, 3])
def test_constraint_rebase_without_objective_support_preserves_gp_and_noise(
        project_root, monkeypatch, saved_incumbent_failure, comparisons):
    algorithm, adapter, goal, _ = _runtime(project_root, monkeypatch, saved_incumbent_failure)
    before_goal, objective = goal.to_dict(), algorithm.gpr[0]
    old_constraint = algorithm.constraint_models["learned"]
    before_variance = _variance_predictions(algorithm)
    if comparisons:
        _raw_comparisons(algorithm, count=comparisons)
    algorithm._rebase_scope({"scope_start": 44, "program": adapter.basis.program}, stage=58)
    assert algorithm.gpr[0] is objective
    assert algorithm.active_objective_scope_start == 0
    assert algorithm.constraint_models["learned"] is not old_constraint
    assert algorithm.active_scope_start == 44
    # Constraint installation makes its accepted concept available to future
    # goal research; it does not revise any existing goal evidence or model.
    assert goal.to_dict()["constraint_concept"] == {
        "program": adapter.basis.program.to_dict(), "scope_start": 44, "stage": 58}
    assert {key: value for key, value in goal.to_dict().items()
            if key not in {"replacement_research", "constraint_concept"}} == {
        key: value for key, value in before_goal.items()
        if key not in {"replacement_research", "constraint_concept"}}
    assert _variance_predictions(algorithm) == before_variance
    assert algorithm.scope_revisions[-1]["preserved_objective_head"] is True
    assert adapter.simulator_calls == 0 and len(algorithm.history) == 58


def test_raw_replacement_requires_its_own_support_and_rebases_the_goal_opponent(
        project_root, monkeypatch, saved_incumbent_failure):
    algorithm, adapter, goal, _ = _runtime(project_root, monkeypatch, saved_incumbent_failure)
    original = goal.to_dict()
    objective = algorithm.gpr[0]
    _raw_comparisons(algorithm)
    algorithm._rebase_scope({"scope_start": 44, "program": adapter.basis.program}, stage=58)
    assert algorithm.gpr[0] is not objective
    assert algorithm.active_objective_scope_start == 44
    assert algorithm.active_scope_start == 44
    assert goal.accepted_revision is None
    assert goal.state["scope_research"]["candidate"] == original["scope_research"]["candidate"]
    assert goal.state["scope_research"]["validation"] == original["scope_research"]["validation"]
    assert goal.state["scope_research"]["incumbent_scope_start"] == 44
    assert algorithm.scope_revisions[-1]["preserved_objective_head"] is False
    assert adapter.simulator_calls == 0 and len(algorithm.history) == 58


@pytest.mark.parametrize("candidate_loss", [0., 1.])
def test_a_same_label_goal_survives_unless_replacement_also_beats_that_goal(
        project_root, monkeypatch, saved_incumbent_failure, candidate_loss):
    algorithm, adapter, goal, rows = _runtime(project_root, monkeypatch, saved_incumbent_failure)
    _future_label(goal, rows, gain=10.)
    _set_history(algorithm, rows)
    accepted = goal.accepted_revision
    assert accepted is not None
    objective = algorithm.gpr[0]
    _raw_comparisons(algorithm, candidate_loss=candidate_loss)
    algorithm._rebase_scope({"scope_start": 44, "program": adapter.basis.program}, stage=59)
    if candidate_loss == 0.:
        assert algorithm.gpr[0] is objective
        assert goal.accepted_revision == accepted
        assert algorithm.active_objective_scope_start == 0
    else:
        assert algorithm.gpr[0] is not objective
        assert goal.accepted_revision is None
        assert goal.state["scope_attempts"][-1]["status"] == "future_predictions_do_not_improve"
        assert algorithm.active_objective_scope_start == 44
    assert adapter.simulator_calls == 0 and len(algorithm.history) == 59


def test_confirmed_replacement_can_finish_after_constraint_installation(
        project_root, monkeypatch, saved_incumbent_failure):
    algorithm, adapter, goal, rows = _runtime(project_root, monkeypatch, saved_incumbent_failure)
    _set_history(algorithm, rows[:57])
    objective = algorithm.gpr[0]
    _raw_comparisons(algorithm, count=3)
    algorithm._rebase_scope({"scope_start": 44, "program": adapter.basis.program}, stage=57)
    constraint = algorithm.gpr[1]
    assert algorithm.gpr[0] is objective and algorithm.active_objective_scope_start == 0
    features = deepcopy(rows[57]["observation"]["features"])
    target = float(rows[57]["context"]["objective"])
    algorithm.objective_replacement.prepare(features, target + 1., target, stage=57)
    _set_history(algorithm, rows)
    algorithm.objective_replacement.observe(features, target, global_index=57)
    algorithm._consider_objective_replacement(stage=58)
    assert algorithm.gpr[0] is not objective and algorithm.gpr[1] is constraint
    assert algorithm.active_objective_scope_start == algorithm.active_scope_start == 44
    assert len(algorithm.scope_revisions) == 1
    assert adapter.simulator_calls == 0 and len(algorithm.history) == 58


def test_prospective_gp_prediction_matches_installation_from_the_same_paid_prefix(
        project_root, monkeypatch, saved_incumbent_failure):
    algorithm, adapter, goal, _ = _runtime(project_root, monkeypatch, saved_incumbent_failure)
    _raw_comparisons(algorithm)
    # This control already compared the goal to the original GP. It has never
    # compared this raw replacement, so the old comparison cannot exclude it.
    features = goal.state["scope_research"]["validation"][0]["features"]
    point = tuple(int(round(features[name] * 100)) for name in goal.spec.feature_names)
    objective = algorithm.gpr[0]
    mean, covariance = objective.a.copy(), objective.C.copy()
    before_validation = deepcopy(goal.state["scope_research"]["validation"])
    before_variance = _variance_predictions(algorithm)
    prefix = [(control, output.tolist()) for control, output in algorithm.history[44:]]
    fits = []
    original_fit = algorithm._fit_objective_model

    def record_fit(history, observations):
        fits.append([(control, output.tolist()) for control, output in history])
        assert len(algorithm.history) == 58
        return original_fit(history, observations)

    monkeypatch.setattr(algorithm, "_fit_objective_model", record_fit)
    algorithm._prepare_objective_replacement(point)
    pending = deepcopy(algorithm.objective_replacement.state["pending"])
    assert pending["features"] == features and pending["stage"] == 58
    assert pending["candidate"]["tag"] == [42, 52]
    assert fits == [prefix]
    assert algorithm.gpr[0] is objective
    np.testing.assert_array_equal(objective.a, mean)
    np.testing.assert_array_equal(objective.C, covariance)
    assert _variance_predictions(algorithm) == before_variance
    assert goal.state["scope_research"]["validation"] == before_validation
    assert len(algorithm.objective_replacement.state["validation"]) == 4
    algorithm._rebase_scope({"scope_start": 44, "program": adapter.basis.program}, stage=58)
    assert fits == [prefix, prefix]
    assert algorithm.gpr[0].posterior_mean(point) == pytest.approx(
        pending["replacement_mean"], rel=1e-12, abs=1e-12)
    assert algorithm.objective_diagnostics["replacement_fits"] == 1
    assert algorithm.objective_diagnostics["replacement_fit_records"] == 14
    assert adapter.simulator_calls == 0 and len(algorithm.history) == 58


def test_one_existing_vector_reaches_each_controller_once_after_the_forecasts(
        project_root, monkeypatch, saved_incumbent_failure):
    algorithm, adapter, goal, _ = _runtime(project_root, monkeypatch, saved_incumbent_failure)
    _, _, label = _live_goal(saved_incumbent_failure)
    features = label["request"]["features"]
    point = tuple(int(round(features[name] * 100)) for name in goal.spec.feature_names)
    vector = np.asarray([label["objective"], label["constraint"]])
    calls = {"simulated": 0, "goal": 0, "replacement": 0, "constraint": 0}
    goal_observe = goal.observe
    replacement_observe = algorithm.objective_replacement.observe
    algorithm._recovery_control_purpose = "optimization"

    def saved_vector(selected, rng):
        calls["simulated"] += 1
        assert tuple(selected) == point and len(algorithm.history) == 58
        assert goal.state["shared_pending"]["expected_global_index"] == 58
        assert algorithm.objective_replacement.state["pending"]["stage"] == 58
        assert not algorithm.objective_replacement.state["validation"]
        return vector.copy()

    def consume_goal(*args, **kwargs):
        calls["goal"] += 1
        assert calls["simulated"] == 1 and len(algorithm.history) == 59
        return goal_observe(*args, **kwargs)

    def consume_replacement(*args, **kwargs):
        calls["replacement"] += 1
        assert calls["simulated"] == 1 and len(algorithm.history) == 59
        return replacement_observe(*args, **kwargs)

    def consume_constraint(*args, **kwargs):
        calls["constraint"] += 1

    monkeypatch.setattr(adapter, "simulate", saved_vector)
    monkeypatch.setattr(goal, "observe", consume_goal)
    monkeypatch.setattr(algorithm.objective_replacement, "observe", consume_replacement)
    monkeypatch.setattr(algorithm.recovery, "observe", consume_constraint)
    result = algorithm._simulate_and_store(point)
    np.testing.assert_array_equal(result, vector)
    assert calls == {"simulated": 1, "goal": 1, "replacement": 1, "constraint": 1}
    row, = algorithm.objective_replacement.state["validation"]
    assert row["global_index"] == 58 and row["target"] == float(vector[0])
    assert algorithm.objective_replacement.state["pending"] is None
    assert adapter.simulator_calls == 0  # The callback replayed one saved draw.


def _frozen_goal_only_runtime(project_root, monkeypatch, saved):
    import test_cognition_sc_kg_environment as runtime_tests

    source = Path(__file__).resolve().parents[1] / (
        "artifacts/sc_kg_decision_regions_20260911/revision4/"
        "method_source/sc_kg_environment.py")
    module_spec = importlib.util.spec_from_file_location(
        "hegel_machine.cognition._decision_regions_revision4_runtime_tests", source)
    frozen = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(frozen)
    monkeypatch.setattr(runtime_tests, "initialize_sc_kg_optimizer",
                        frozen.initialize_sc_kg_optimizer)
    algorithm, adapter, goal, rows = _runtime(project_root, monkeypatch, saved)
    algorithm.recovery.state["phase"] = "monitoring"
    # This recorded event already established the objective counterexample;
    # the constraint controller has no corresponding revision to propose.
    trigger = next(event for event in saved["objective_research"]["events"]
                   if event["action"] == "research_objective_scope")
    goal.state["scope_research"]["trigger"] = [deepcopy(trigger)]
    assert algorithm.gpr[0].basis_map is None
    return algorithm, adapter, goal, rows


def test_frozen_raw_objective_scope_starts_comparison_before_a_goal_candidate_exists(
        project_root, monkeypatch, saved_incumbent_failure):
    algorithm, adapter, goal, _ = _frozen_goal_only_runtime(
        project_root, monkeypatch, saved_incumbent_failure)
    research = goal.state["scope_research"]
    research.update(phase="collecting", candidate=None, validation=[])
    objective, constraint = algorithm.gpr
    before_mean, before_covariance = objective.a.copy(), objective.C.copy()
    point = (13, 27, 71)
    algorithm._prepare_objective_replacement(point)
    replacement = algorithm.objective_replacement
    pending = replacement.state["pending"]
    assert replacement.state["scope_start"] == 42 and replacement.state["confirmed"]
    assert pending["stage"] == 58 and "candidate" not in pending
    assert pending["incumbent_mean"] == pytest.approx(objective.posterior_mean(point))
    expected = algorithm._fit_objective_model(*algorithm._objective_prefix(42))
    assert pending["replacement_mean"] == pytest.approx(expected.posterior_mean(point))
    assert algorithm.objective_diagnostics["replacement_fit_records"] == 16
    algorithm._consider_objective_replacement(stage=58)
    assert replacement.state["validation"] == []
    assert algorithm.gpr[0] is objective and algorithm.gpr[1] is constraint
    np.testing.assert_array_equal(objective.a, before_mean)
    np.testing.assert_array_equal(objective.C, before_covariance)
    assert algorithm.active_objective_scope_start == algorithm.active_scope_start == 0
    assert adapter.simulator_calls == 0


@pytest.mark.parametrize("winner", ["replacement", "incumbent"])
def test_frozen_goal_only_raw_scope_needs_new_paired_support_and_preserves_constraint_state(
        project_root, monkeypatch, saved_incumbent_failure, winner):
    algorithm, adapter, goal, rows = _frozen_goal_only_runtime(
        project_root, monkeypatch, saved_incumbent_failure)
    objective = algorithm.gpr[0]
    constraints = dict(algorithm.constraint_models)
    constraint_state = {name: (model.a.copy(), model.C.copy())
                        for name, model in constraints.items()}
    old_goal_validation = deepcopy(goal.state["scope_research"]["validation"])
    # Four previous comparisons matched this goal, but concerned scope 44.
    # They cannot count toward the newly questioned objective scope 42.
    _raw_comparisons(algorithm, count=4, candidate_loss=1.)
    tag = algorithm._goal_candidate_tag()
    assert len(algorithm.objective_replacement.comparisons(tag)) == 4
    prepared = []
    for offset, point in enumerate(((13, 27, 71), (14, 27, 71),
                                   (15, 27, 71), (16, 27, 71))):
        stage = len(rows)
        algorithm._prepare_objective_replacement(point)
        replacement = algorithm.objective_replacement
        pending = deepcopy(replacement.state["pending"])
        assert replacement.state["scope_start"] == 42
        assert len(replacement.state["validation"]) == offset
        assert len(replacement.comparisons(tag)) == offset
        prepared.append(pending)
        # Use actual GP forecasts, then choose a controlled future label at
        # one forecast to exercise both decisions without simulator calls.
        target = pending[f"{winner}_mean"]
        assert abs(pending["replacement_mean"] - pending["incumbent_mean"]) > 1e-4
        features = pending["features"]
        rows.append({"observation": {"features": deepcopy(features), "target": -1.},
                     "context": {"global_index": stage, "objective": target}})
        _set_history(algorithm, rows)
        replacement.observe(features, target, global_index=stage)
        algorithm._consider_objective_replacement(stage=stage+1)
        if offset < 3:
            assert algorithm.gpr[0] is objective
    decision = algorithm.objective_replacement.state["decisions"][-1]
    assert decision["evidence"]["count"] == 4
    assert decision["replace"] == (winner == "replacement")
    assert algorithm.recovery.state["phase"] == "monitoring"
    assert algorithm.active_scope_start == 0 and algorithm.scope_revisions == []
    for name, constraint in constraints.items():
        assert algorithm.constraint_models[name] is constraint
        np.testing.assert_array_equal(constraint.a, constraint_state[name][0])
        np.testing.assert_array_equal(constraint.C, constraint_state[name][1])
    if winner == "replacement":
        assert algorithm.gpr[0] is not objective and algorithm.gpr[0].basis_map is None
        assert algorithm.active_objective_scope_start == 42
        comparison = goal.state["scope_research"] or goal.state["scope_attempts"][-1]
        assert comparison["comparison_start"] == len(old_goal_validation)
        assert [row["global_index"] for row in comparison["incumbent_validation"]] == list(range(58, 62))
        assert comparison["validation"][:len(old_goal_validation)] == old_goal_validation
    else:
        assert algorithm.gpr[0] is objective and algorithm.active_objective_scope_start == 0
        assert goal.state["scope_research"]["validation"] == old_goal_validation
        assert "incumbent_validation" not in goal.state["scope_research"]
    assert adapter.simulator_calls == 0
