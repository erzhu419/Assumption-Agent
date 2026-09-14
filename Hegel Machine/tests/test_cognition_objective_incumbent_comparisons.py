"""Experimental v049: compare a candidate to a prospective rebuild.

All labels below are controlled decision examples using a real saved frozen
candidate. No simulator is called, and no prediction is reconstructed from the
label that evaluates it. The imported implementation is the frozen rejected
experiment; these tests do not enable it in the live v048 default.
"""

from copy import deepcopy
import json
from math import sqrt

import pytest
import numpy as np

from test_cognition_sc_kg_environment import project_root
from objective_incumbent_experiment import ObjectiveResearch, saved_experiment_optimizer
from test_cognition_objective_incumbent import CASE, _paused_comparison, saved_incumbent_failure
from hegel_machine.cognition.decision_experiments import _predict


def _start(saved):
    goal, rows, _ = _paused_comparison(saved)
    goal.state["scope_research"].update(validation=[], validation_target=4,
        validation_rounds=[], status="validating")
    return goal, rows


def _prepare(goal, rows, *, alternative_gain=1., scope_start=44, original_gain=4.):
    index = len(rows)
    point = {"x1": .13 + .01 * (index - 58), "x2": .27, "x3": .71}
    frozen = goal.state["scope_research"]["candidate"]
    candidate_mean = _predict(goal.spec, frozen, point)
    # For a negative gain the candidate errs, while the alternative is exact.
    target = candidate_mean + (sqrt(-alternative_gain) if alternative_gain < 0 else 0.)
    previous_mean = target + sqrt(original_gain + (target - candidate_mean) ** 2)
    request = goal.prepare_shared_control(rows, lambda features: {
        "features": deepcopy(features), "objective_mean": previous_mean,
        "objective_noise": .001}, point, scope_start=44, stage=index,
        remaining_calls=10, purpose="optimization")
    assert request is not None and request["expected_global_index"] == index
    if scope_start is not None:
        alternative_mean = target + (sqrt(alternative_gain) if alternative_gain >= 0 else 0.)
        goal.prepare_incumbent_comparison(scope_start=scope_start,
            previous_mean=alternative_mean, stage=index)
    return point, target, previous_mean


def _observe(goal, rows, prepared):
    point, target, previous_mean = prepared
    index = len(rows)
    context = {"global_index": index, "stage": index + 1,
               "purpose": "optimization", "objective": target}
    rows.append({"observation": {"features": deepcopy(point), "target": -1.},
                 "context": context})
    goal.observe(point, [target, -1.], context=context)
    return (target - previous_mean) ** 2


@pytest.mark.parametrize("alternative_gain", [1., -1.])
def test_same_label_support_is_reconsidered_against_four_precomputed_rebuild_predictions(
        saved_incumbent_failure, alternative_gain):
    goal, rows = _start(saved_incumbent_failure)
    frozen = deepcopy(goal.state["scope_research"]["candidate"])
    original_errors = []
    for _ in range(4):
        original_errors.append(_observe(goal, rows,
            _prepare(goal, rows, alternative_gain=alternative_gain)))
    assert goal.accepted_revision is not None  # The old incumbent loses either way.
    before = deepcopy(goal.state["scope_research"]["validation"])
    goal.incumbent_rebuilt(stage=62, scope_start=44)
    research = goal.state["scope_research"] or goal.state["scope_attempts"][-1]
    assert research["candidate"] == frozen
    assert research["validation"] == before
    assert [row["previous_error"] for row in research["validation"]] == original_errors
    assert research["validation_rounds"][-1]["count"] == 4
    if alternative_gain > 0:
        accepted = goal.accepted_revision
        assert accepted["model"] == frozen
        assert accepted["validation_indices"] == [58, 59, 60, 61]
        assert accepted["previous_mse"] == pytest.approx(1.)
        assert accepted["validation_mse"] == pytest.approx(0.)
    else:
        assert goal.accepted_revision is None
        assert research["status"] == "future_predictions_do_not_improve"
    assert len(rows) == 62 and not goal.state["fits"]


@pytest.mark.parametrize("future_scope", [None, 44])
def test_two_precomputed_comparisons_and_two_new_labels_complete_the_same_four_point_rule(
        saved_incumbent_failure, future_scope):
    goal, rows = _start(saved_incumbent_failure)
    for _ in range(2):
        _observe(goal, rows, _prepare(goal, rows))
    frozen = deepcopy(goal.state["scope_research"]["candidate"])
    goal.incumbent_rebuilt(stage=60, scope_start=44)
    assert goal.accepted_revision is None
    _observe(goal, rows, _prepare(goal, rows, scope_start=future_scope, original_gain=1.))
    assert goal.accepted_revision is None
    _observe(goal, rows, _prepare(goal, rows, scope_start=future_scope, original_gain=1.))
    accepted = goal.accepted_revision
    assert accepted["validation_indices"] == [58, 59, 60, 61]
    assert accepted["validation_rounds"][-1]["count"] == 4
    assert accepted["model"] == frozen and accepted["previous_mse"] == pytest.approx(1.)


def test_predictions_for_a_different_scope_cannot_save_four_new_comparisons(saved_incumbent_failure):
    goal, rows = _start(saved_incumbent_failure)
    for _ in range(4):
        _observe(goal, rows, _prepare(goal, rows, scope_start=43))
    assert goal.accepted_revision is not None
    goal.incumbent_rebuilt(stage=62, scope_start=44)
    assert goal.accepted_revision is None
    for _ in range(3):
        _observe(goal, rows, _prepare(goal, rows, scope_start=None))
        assert goal.accepted_revision is None
    _observe(goal, rows, _prepare(goal, rows, scope_start=None))
    assert goal.accepted_revision["validation_indices"] == [62, 63, 64, 65]


def test_uncertain_precomputed_four_keep_the_original_two_standard_error_rule(saved_incumbent_failure):
    goal, rows = _start(saved_incumbent_failure)
    for gain in (10., 0., 0., 0.):
        _observe(goal, rows, _prepare(goal, rows, alternative_gain=gain))
    assert goal.accepted_revision is not None
    goal.incumbent_rebuilt(stage=62, scope_start=44)
    research = goal.state["scope_research"]
    assert goal.accepted_revision is None
    assert research["status"] == "inconclusive_future_predictions"
    assert research["validation_rounds"][-1]["count"] == 4
    _observe(goal, rows, _prepare(goal, rows, scope_start=None, original_gain=10.))
    assert goal.accepted_revision is None
    _observe(goal, rows, _prepare(goal, rows, scope_start=None, original_gain=10.))
    assert goal.accepted_revision["validation_indices"] == list(range(58, 64))


def test_json_resume_preserves_a_forecast_without_creating_evidence_before_its_label(
        saved_incumbent_failure):
    goal, rows = _start(saved_incumbent_failure)
    for _ in range(3):
        _observe(goal, rows, _prepare(goal, rows))
    prepared = _prepare(goal, rows)
    validation = deepcopy(goal.state["scope_research"]["validation"])
    state = json.loads(json.dumps(goal.to_dict()))
    assert len(validation) == 3 and goal.accepted_revision is None
    assert "prospective_incumbent" in state["shared_pending"]
    resumed = ObjectiveResearch(goal.spec, state=state)
    assert resumed.to_dict() == state
    assert resumed.state["scope_research"]["validation"] == validation
    _observe(resumed, rows, prepared)
    resumed.incumbent_rebuilt(stage=62, scope_start=44)
    assert resumed.accepted_revision["validation_indices"] == [58, 59, 60, 61]
    assert resumed.accepted_revision["previous_mse"] == pytest.approx(1.)
    assert resumed.state["shared_pending"] is None


def test_no_selected_comparison_cannot_receive_a_prospective_forecast(saved_incumbent_failure):
    goal, _ = _start(saved_incumbent_failure)
    before = goal.to_dict()
    goal.prepare_incumbent_comparison(scope_start=44, previous_mean=2., stage=58)
    assert goal.to_dict() == before


@pytest.mark.parametrize("case", ["ready", "preserved_head", "collecting_constraint"])
def test_real_prospective_fit_matches_actual_rebase_using_only_the_paid_prefix(
        project_root, monkeypatch, saved_incumbent_failure, case):
    algorithm, adapter, _ = saved_experiment_optimizer(
        project_root, monkeypatch, CASE, frozen_directory="sc_kg_active_noise_v046_20260910")
    monkeypatch.setattr(adapter, "simulate", lambda *args, **kwargs:
                        pytest.fail("prospective comparisons cannot buy simulator labels"))
    algorithm._fit_initial_belief(algorithm._initial_samples())
    goal, rows, next_label = _paused_comparison(saved_incumbent_failure)
    algorithm.objective_research = goal
    algorithm.history = [(tuple(int(round(row["observation"]["features"][name] * 100))
                                for name in goal.spec.feature_names),
                          np.asarray([row["context"]["objective"], row["observation"]["target"]]))
                         for row in rows]
    algorithm.observations = {}
    for point, output in algorithm.history:
        algorithm.observations.setdefault(point, []).append(output)
    algorithm.recovery.state = deepcopy(saved_incumbent_failure["recovery"])
    algorithm.recovery.state["phase"] = "validating"
    assert algorithm.recovery.state["episodes"][-1]["scope_start"] == 44
    features = next_label["request"]["features"]
    point = tuple(int(round(features[name] * 100)) for name in goal.spec.feature_names)
    assert goal.prepare_shared_control(rows, algorithm._objective_forecast, features,
        scope_start=0, stage=58, remaining_calls=6, purpose="optimization") is not None
    if case == "preserved_head":
        algorithm.objective_scope_revisions = [{"scope_start": 0}]
    elif case == "collecting_constraint":
        algorithm.recovery.state["phase"] = "collecting"
    before = deepcopy(algorithm.objective_diagnostics)
    candidate = deepcopy(goal.state["scope_research"]["candidate"])
    incumbent = algorithm.gpr[0]
    expected_prefix = [(control, output.tolist()) for control, output in algorithm.history[44:]]
    fitting_inputs = []
    original_fit = type(algorithm)._rebuild_gpr_from_history

    def record_fit(self, output_index, *args, **kwargs):
        if output_index == 0:
            fitting_inputs.append([(control, output.tolist()) for control, output in self.history])
            assert len(algorithm.history) == 58
        return original_fit(self, output_index, *args, **kwargs)

    monkeypatch.setattr(type(algorithm), "_rebuild_gpr_from_history", record_fit)
    algorithm._prepare_objective_incumbent_comparison(point)
    assert len(algorithm.history) == 58 and algorithm.gpr[0] is incumbent
    assert goal.state["scope_research"]["candidate"] == candidate
    assert len(goal.state["scope_research"]["validation"]) == 5
    if case == "ready":
        prospective = deepcopy(goal.state["shared_pending"]["prospective_incumbent"])
        assert prospective["scope_start"] == 44 and prospective["stage"] == 58
        assert fitting_inputs == [expected_prefix]
        assert algorithm.objective_diagnostics["incumbent_comparison_fits"] - before[
            "incumbent_comparison_fits"] == 1
        assert algorithm.objective_diagnostics["incumbent_comparison_fit_records"] - before[
            "incumbent_comparison_fit_records"] == 14
        assert algorithm.objective_diagnostics["incumbent_comparison_prediction_rows"] - before[
            "incumbent_comparison_prediction_rows"] == 1
        # Independently install from the same prefix, before receiving label 58.
        monkeypatch.setattr(algorithm.recovery, "acknowledge_revision", lambda: None)
        algorithm._rebase_scope({"scope_start": 44, "program": adapter.basis.program}, stage=58)
        assert fitting_inputs == [expected_prefix, expected_prefix]
        assert float(algorithm.gpr[0].posterior_mean(point)) == pytest.approx(
            prospective["mean"], rel=1e-12, abs=1e-12)
    else:
        assert "prospective_incumbent" not in goal.state["shared_pending"]
        assert not fitting_inputs and algorithm.objective_diagnostics == before
    assert adapter.simulator_calls == 0
