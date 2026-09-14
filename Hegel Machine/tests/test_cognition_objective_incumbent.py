"""Experimental v049: rebuild starts a fresh comparison of the frozen candidate.

The saved v048 failure supplies its actual five old comparisons and first new
label. Later controlled labels exercise decision boundaries, not simulated
optimization outcomes. These tests target the frozen rejected experiment, not
the live v048 default, and make no physical simulator calls.
"""

from copy import deepcopy
import json
from math import sqrt
from pathlib import Path

import numpy as np
import pytest

from test_cognition_sc_kg_environment import project_root
from objective_incumbent_experiment import ObjectiveResearch, saved_experiment_optimizer
from hegel_machine.cognition.decision_experiments import _predict
from hegel_machine.cognition.models import TaskSpec
from hegel_machine.cognition.validation_continuation import evidence_summary


CASE = "completion_value_2_mechanism_change_noise_0.00"


@pytest.fixture(scope="module")
def saved_incumbent_failure():
    path = (Path(__file__).resolve().parents[1] / "artifacts" /
        "sc_kg_objective_validation_v048_20260910/runs" /
        f"{CASE}_objective_validation_v048.json")
    return json.loads(path.read_text())


def _paused_comparison(saved):
    names = ("x1", "x2", "x3")
    spec = TaskSpec("rebuilt_goal_comparison", "Minimize the measured objective", names, (), (),
        feature_bounds={name: (0., 1.) for name in names},
        context={"objective_direction": "minimize", "constraint_upper_bound": "0.0"})
    goal = ObjectiveResearch(spec)
    accepted = saved["objective_research"]["scope_revisions"][0]
    records = sorted((row for row in saved["objective_research"]["shared_outcomes"]
                      if row["request"]["reason"] == "validate_objective_scope"),
                     key=lambda row: row["context"]["global_index"])
    validation = [{"features": deepcopy(row["request"]["features"]),
        "global_index": row["context"]["global_index"], "target": row["objective"],
        "previous_error": (row["objective"] - row["request"]["previous_mean"]) ** 2,
        "candidate_error": (row["objective"] - row["request"]["candidate_mean"]) ** 2}
        for row in records[:5]]
    goal.state.update(scope_start=0, evidence_start=0,
        scope_research={"scope_start": 42, "phase": "validating",
            "status": "inconclusive_future_predictions", "validation_target": 6,
            "candidate": deepcopy(accepted["model"]), "validation": validation,
            "training_indices": deepcopy(accepted["training_indices"]),
            "validation_rounds": deepcopy(accepted["validation_rounds"][:1])})
    rows = [{"observation": {"features": dict(zip(names, (x / 100. for x in point))),
                             "target": output[1]},
             "context": {"global_index": i, "stage": i + 1, "objective": output[0]}}
            for i, (point, output) in enumerate(zip(saved["initial_points"],
                                                    saved["initial_outputs"]))]
    rows.extend(deepcopy(saved["recovery"]["observations"][:26]))
    assert len(rows) == 58 and len(records) == 6
    return goal, rows, records[-1]


def _future_control(goal, rows, gain):
    """Give two precomputed forecasts a controlled future squared-error gain."""
    index = len(rows)
    point = {"x1": .13 + .01 * (index - 58), "x2": .27, "x3": .71}
    candidate_mean = _predict(goal.spec, goal.state["scope_research"]["candidate"], point)
    previous_mean = candidate_mean + sqrt(abs(gain))
    request = goal.prepare_shared_control(rows, lambda features: {
        "features": deepcopy(features), "objective_mean": previous_mean,
        "objective_noise": .001}, point, scope_start=44, stage=index,
        remaining_calls=10, purpose="optimization")
    assert request is not None and request["expected_global_index"] == index
    output = candidate_mean if gain >= 0 else previous_mean
    context = {"global_index": index, "stage": index + 1,
               "purpose": "optimization", "objective": output}
    rows.append({"observation": {"features": deepcopy(point), "target": -1.},
                 "context": context})
    goal.observe(point, [output, -1.], context=context)


def test_saved_old_advantage_cannot_adopt_after_one_loss_against_the_rebuilt_incumbent(
        saved_incumbent_failure):
    goal, rows, next_label = _paused_comparison(saved_incumbent_failure)
    before = deepcopy(goal.state["scope_research"])
    goal.incumbent_rebuilt(stage=58, scope_start=44)
    research = goal.state["scope_research"]
    assert research["candidate"] == before["candidate"]
    assert research["training_indices"] == before["training_indices"]
    assert research["validation"] == before["validation"]
    assert research["validation_rounds"] == before["validation_rounds"]
    assert research["comparison_start"] == 5 and research["validation_target"] == 9

    prediction = next_label["request"]
    goal.prepare_shared_control(rows, lambda features: {
        "features": deepcopy(features), "objective_mean": prediction["previous_mean"],
        "objective_noise": prediction["objective_noise"]}, prediction["features"],
        scope_start=44, stage=58, remaining_calls=6, purpose="optimization")
    goal.observe(prediction["features"], [next_label["objective"], next_label["constraint"]],
                 context=next_label["context"])
    # The old mixed comparison really would accept this losing new candidate.
    assert evidence_summary(research["validation"])["decision"] == "supported"
    last = research["validation"][-1]
    assert last["candidate_error"] == pytest.approx(.11077524914500612)
    assert last["previous_error"] == pytest.approx(.006988449075886502)
    assert goal.accepted_revision is None and research["phase"] == "validating"
    assert research["candidate"] == before["candidate"] and not goal.state["fits"]


def test_four_current_losses_reject_despite_the_large_old_advantage(saved_incumbent_failure):
    goal, rows, _ = _paused_comparison(saved_incumbent_failure)
    goal.incumbent_rebuilt(stage=58, scope_start=44)
    frozen = deepcopy(goal.state["scope_research"]["candidate"])
    for _ in range(4):
        _future_control(goal, rows, -.01)
    assert goal.accepted_revision is None and goal.state["scope_research"] is None
    rejected = goal.state["scope_attempts"][-1]
    assert rejected["status"] == "future_predictions_do_not_improve"
    assert rejected["candidate"] == frozen and len(rejected["validation"]) == 9
    assert evidence_summary(rejected["validation"])["decision"] == "supported"
    assert rejected["validation_rounds"][-1]["count"] == 4


def test_four_current_wins_survive_json_resume_and_install_only_current_validation_indices(
        saved_incumbent_failure):
    goal, rows, _ = _paused_comparison(saved_incumbent_failure)
    goal.incumbent_rebuilt(stage=58, scope_start=44)
    frozen = deepcopy(goal.state["scope_research"]["candidate"])
    for _ in range(3):
        _future_control(goal, rows, 1.)
        assert goal.accepted_revision is None
    state = json.loads(json.dumps(goal.to_dict()))
    resumed = ObjectiveResearch(goal.spec, state=state)
    assert resumed.to_dict() == state
    _future_control(resumed, rows, 1.)
    revision = resumed.accepted_revision
    assert revision["validation_indices"] == [58, 59, 60, 61]
    assert revision["model"] == frozen
    assert revision["validation_mse"] == pytest.approx(0.)
    assert revision["previous_mse"] == pytest.approx(1.)
    assert revision["validation_rounds"][-1]["count"] == 4
    assert resumed.state["fits"] == state["fits"]
    assert len(resumed.state["scope_research"]["validation"]) == 9


def test_uncertain_current_four_extend_by_two_without_reusing_old_gains(saved_incumbent_failure):
    goal, rows, _ = _paused_comparison(saved_incumbent_failure)
    goal.incumbent_rebuilt(stage=58, scope_start=44)
    for gain in (10., 0., 0., 0.):
        _future_control(goal, rows, gain)
    research = goal.state["scope_research"]
    assert research["status"] == "inconclusive_future_predictions"
    assert research["validation_target"] == 11 and research["comparison_start"] == 5
    assert research["validation_rounds"][-1]["count"] == 4
    assert goal.accepted_revision is None
    _future_control(goal, rows, 10.)
    assert goal.accepted_revision is None
    _future_control(goal, rows, 10.)
    assert goal.accepted_revision["validation_indices"] == list(range(58, 64))


def test_rebuild_withdraws_same_label_support_and_cancels_old_comparator_predictions(
        saved_incumbent_failure):
    goal, rows, _ = _paused_comparison(saved_incumbent_failure)
    _future_control(goal, rows, 10.)
    assert goal.accepted_revision is not None
    research = deepcopy(goal.state["scope_research"])
    goal.state["pending"] = {"features": {"x1": .3, "x2": .3, "x3": .3},
                             "purpose": "objective_confirmation"}
    goal.state["shared_pending"] = {"features": {"x1": .4, "x2": .4, "x3": .4},
                                    "purpose": "optimization"}
    predictions = deepcopy(goal.state["shared_predictions"])
    goal.incumbent_rebuilt(stage=59, scope_start=44)
    reset = goal.state["scope_research"]
    assert goal.accepted_revision is None
    assert reset["candidate"] == research["candidate"]
    assert reset["validation"] == research["validation"]
    assert reset["phase"] == "validating" and reset["status"] == "awaiting_current_incumbent"
    assert reset["comparison_start"] == 6 and reset["validation_target"] == 10
    assert goal.state["pending"] is None and goal.state["shared_pending"] is None
    assert goal.state["shared_predictions"] == predictions


@pytest.mark.parametrize("phase", [None, "collecting"])
def test_without_a_frozen_candidate_rebuild_does_not_open_research(saved_incumbent_failure, phase):
    goal, _, _ = _paused_comparison(saved_incumbent_failure)
    goal.state["scope_research"] = (None if phase is None else
        {"phase": "collecting", "candidate": None, "scope_start": 42, "validation": []})
    before = goal.to_dict()
    goal.incumbent_rebuilt(stage=58, scope_start=44)
    assert goal.to_dict() == before


@pytest.mark.parametrize("preserve", [False, True])
def test_constraint_rebase_restarts_only_an_actually_rebuilt_objective_head(
        project_root, monkeypatch, saved_incumbent_failure, preserve):
    algorithm, adapter, _ = saved_experiment_optimizer(
        project_root, monkeypatch, CASE, frozen_directory="sc_kg_active_noise_v046_20260910")
    monkeypatch.setattr(adapter, "simulate", lambda *args, **kwargs:
                        pytest.fail("incumbent transition tests cannot buy simulator labels"))
    algorithm._fit_initial_belief(algorithm._initial_samples())
    goal, rows, _ = _paused_comparison(saved_incumbent_failure)
    algorithm.objective_research = goal
    algorithm.history = [(tuple(int(round(row["observation"]["features"][name] * 100))
                                for name in goal.spec.feature_names),
                          np.asarray([row["context"]["objective"], row["observation"]["target"]]))
                         for row in rows]
    algorithm.observations = {}
    for point, output in algorithm.history:
        algorithm.observations.setdefault(point, []).append(output)
    monkeypatch.setattr(algorithm.recovery, "acknowledge_revision", lambda: None)
    if preserve:
        algorithm.objective_scope_revisions = [{"scope_start": 0}]
    before = goal.to_dict()
    incumbent = algorithm.gpr[0]
    algorithm._rebase_scope({"scope_start": 44, "program": adapter.basis.program}, stage=58)
    if preserve:
        assert algorithm.gpr[0] is incumbent and goal.to_dict() == before
    else:
        assert algorithm.gpr[0] is not incumbent
        assert goal.state["scope_research"]["comparison_start"] == 5
        assert goal.state["scope_research"]["candidate"] == before["scope_research"]["candidate"]
    assert algorithm.scope_revisions[-1]["preserved_objective_head"] is preserve
    assert adapter.simulator_calls == 0
