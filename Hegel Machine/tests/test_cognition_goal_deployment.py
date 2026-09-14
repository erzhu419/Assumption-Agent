"""Deploy a frozen goal representation only after actual future comparison.

The saved stable-interaction prefix and vectors exercise real SC-KG models;
no optimizer, research, or evaluator simulator draw is acquired here.
"""

from copy import deepcopy
import json
from math import sqrt
from pathlib import Path

import numpy as np
import pytest

from hegel_machine.cognition.decision_experiments import _predict
from hegel_machine.cognition.models import Observation
from test_cognition_sc_kg_environment import project_root


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = "sc_kg_unknown_revision_confirmation_20260911"
CASE = "unknown_confirmation_1_stable_interaction"


@pytest.fixture(scope="module")
def saved_goal():
    path = ROOT / "artifacts" / DIRECTORY / "runs" / f"{CASE}_hegel_interaction_candidate.json"
    return json.loads(path.read_text())


@pytest.fixture
def deployment_runtime(project_root, monkeypatch, saved_goal):
    import test_cognition_sc_kg_environment as sc_tests

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import sc_kg_unknown_revision as task
    monkeypatch.setattr(sc_tests, "SCKGEnvironment", task.configure(sc_tests.SCKGEnvironment))
    algorithm, adapter, environment = sc_tests._saved_terminal_failure_optimizer(
        project_root, monkeypatch, CASE, frozen_directory=DIRECTORY)
    monkeypatch.setattr(adapter, "simulate", lambda *args, **kwargs:
                        pytest.fail("deployment must use a saved vector"))
    monkeypatch.setattr(environment, "observe", lambda *args, **kwargs:
                        pytest.fail("deployment cannot buy another research label"))
    algorithm._fit_initial_belief(algorithm._initial_samples())
    # Isolate goal deployment from unrelated constraint research and native
    # incumbent bookkeeping; GP/HVD updating and head installation stay real.
    monkeypatch.setattr(algorithm.recovery, "observe", lambda *args, **kwargs: None)
    monkeypatch.setattr(type(algorithm).__mro__[1], "_update_posterior_dominance_incumbent",
                        lambda *args, **kwargs: None)
    return algorithm, adapter, environment, saved_goal


def _start(algorithm):
    goal = algorithm.objective_research
    request = goal.next_control(algorithm._objective_evidence(), algorithm._objective_forecast,
        scope_start=0, stage=32, remaining_calls=32)
    assert request is None
    research = goal.state["scope_research"]
    assert research["deployment"] and research["phase"] == "validating"
    return goal, research


def _replay(algorithm, adapter, monkeypatch, saved_vector, *, output=None, before_label=None):
    point = tuple(saved_vector["point"])
    output = np.asarray(saved_vector["output"] if output is None else output)
    models = list(algorithm.gpr)
    mean = [model.posterior_mean(point) for model in models]
    epistemic = [model.posterior_var(point) for model in models]
    noise = [algorithm.variance_model.predict_variance(i, point, adapter) for i in range(2)]
    returned = []

    def saved_draw(selected, rng):
        assert tuple(selected) == point
        if before_label is not None:
            before_label()
        returned.append(point)
        return output.copy()

    monkeypatch.setattr(adapter, "simulate", saved_draw)
    algorithm._recovery_control_purpose = "optimization"
    algorithm._simulate_and_store(point)
    for i, model in enumerate(models):
        model.update(point, float(output[i]), noise[i])
    for i, model in enumerate(models):
        values = [float(y[i]) for y in algorithm.observations[point]]
        algorithm.variance_model.update(i, np.asarray(point), float(output[i]), mean[i], model, adapter,
            epistemic_var=epistemic[i], replicate_count=len(values),
            replicate_variance=float(np.var(values, ddof=1)) if len(values) > 1 else None)
    algorithm._update_posterior_dominance_incumbent(stage=len(algorithm.history), reason="saved_goal_vector")
    assert returned == [point] and adapter.simulator_calls == 0


def test_initial_representation_is_frozen_without_buying_or_installing_its_training_evidence(deployment_runtime):
    algorithm, adapter, environment, _ = deployment_runtime
    original = algorithm.gpr[0]
    parameters = original.a.copy(), original.C.copy()
    goal, research = _start(algorithm)
    assert [tuple(pair) for pair in research["candidate"]["interactions"]] == [(0, 1), (0, 2), (1, 2)]
    assert research["candidate"]["training_end"] == 31
    assert research["training_indices"] == list(range(32))
    assert research["validation"] == [] and research["candidate"] is not goal.state["model"]
    assert goal.accepted_revision is None and not algorithm.objective_scope_revisions
    assert algorithm.gpr[0] is original and original.basis_map is None
    np.testing.assert_array_equal(original.a, parameters[0])
    np.testing.assert_array_equal(original.C, parameters[1])
    state = deepcopy(research)
    rows = algorithm._objective_evidence()
    for row in rows:
        goal.observe(row["observation"]["features"],
            [row["context"]["objective"], row["observation"]["target"]],
            context={**row["context"], "purpose": "optimization"})
    goal._start_model_deployment(rows, stage=32)
    assert goal.state["scope_research"] == state
    assert goal.next_control(rows, lambda features: pytest.fail("passive deployment cannot nominate a query"),
        scope_start=0, stage=32, remaining_calls=32) is None
    assert not goal.state["requests"]
    assert len(algorithm.history) == environment.costs["simulator_calls"] == 32
    assert adapter.simulator_calls == 0


def test_the_shared_prediction_precedes_one_new_vector_and_training_labels_cannot_answer_it(
        deployment_runtime, monkeypatch):
    algorithm, adapter, _, saved = deployment_runtime
    goal, research = _start(algorithm)
    frozen = deepcopy(research["candidate"])
    point = saved["simulations"][0]["point"]
    features = {name: value / 100 for name, value in zip(goal.spec.feature_names, point)}

    def before_label():
        request = goal.state["shared_pending"]
        assert request["expected_global_index"] == 32
        assert request["candidate_mean"] == pytest.approx(_predict(goal.spec, frozen, features))
        assert not research["validation"] and len(algorithm.history) == 32
        with pytest.raises(ValueError, match="next real observation"):
            goal.observe(features, saved["simulations"][0]["output"],
                context={"global_index": 31, "stage": 32, "purpose": "optimization"})
        assert not research["validation"]

    _replay(algorithm, adapter, monkeypatch, saved["simulations"][0], before_label=before_label)
    assert research["candidate"] == frozen and len(research["validation"]) == 1
    assert research["validation"][0]["global_index"] == 32
    assert len(goal.state["shared_predictions"]) == len(goal.state["shared_outcomes"]) == 1
    before = goal.to_dict()
    goal.observe(features, saved["simulations"][0]["output"],
        context={"global_index": 32, "stage": 33, "purpose": "optimization"})
    assert goal.to_dict() == before and len(algorithm.history) == 33


def test_saved_future_support_installs_the_callable_goal_in_the_actual_gp_and_terminal_scores(
        deployment_runtime, monkeypatch):
    algorithm, adapter, _, saved = deployment_runtime
    goal, research = _start(algorithm)
    frozen = deepcopy(research["candidate"])
    original = algorithm.gpr[0]
    for index, vector in enumerate(saved["simulations"][:10], 1):
        _replay(algorithm, adapter, monkeypatch, vector)
        if index < 10:
            assert algorithm.gpr[0] is original and not algorithm.objective_scope_revisions
    revision, = algorithm.objective_scope_revisions
    installed = algorithm.gpr[0]
    assert installed is not original and installed.basis_map is not None
    assert revision["deployment"] and revision["stage"] == 42
    assert revision["model"] == frozen
    assert revision["training_indices"] == list(range(32))
    assert revision["validation_indices"] == list(range(32, 42))
    assert revision["fit_records"] == revision["posterior_updates"] == 42
    assert goal.accepted_revision is None
    assert len(goal.state["shared_predictions"]) == len(goal.state["shared_outcomes"]) == 10
    points = [tuple(vector["point"]) for vector in saved["simulations"][:4]]
    for point in points:
        features = {name: value / 100 for name, value in zip(goal.spec.feature_names, point)}
        assert installed.basis_map.program.predict(Observation(features, 0.)) == pytest.approx(
            _predict(goal.spec, frozen, features))
    terminal = algorithm._terminal_certificate_components(algorithm.gpr, algorithm.variance_model, points)
    np.testing.assert_allclose(terminal["objective"], installed.posterior_mean_many(points))
    assert not np.allclose(terminal["objective"], original.posterior_mean_many(points))
    assert len(algorithm.history) == 42 and algorithm.config.N == 64


def test_future_evidence_favoring_the_incumbent_keeps_the_original_gp(deployment_runtime, monkeypatch):
    algorithm, adapter, _, saved = deployment_runtime
    goal, research = _start(algorithm)
    original, frozen = algorithm.gpr[0], deepcopy(research["candidate"])
    for vector in saved["simulations"][:4]:
        point = tuple(vector["point"])
        output = [float(original.posterior_mean(point)), vector["output"][1]]
        _replay(algorithm, adapter, monkeypatch, vector, output=output)
        assert algorithm.gpr[0] is original and not algorithm.objective_scope_revisions
    rejected = goal.state["scope_attempts"][-1]
    assert rejected["deployment"] and rejected["status"] == "future_predictions_do_not_improve"
    assert rejected["candidate"] == frozen
    assert [row["global_index"] for row in rejected["validation"]] == list(range(32, 36))
    assert goal.accepted_revision is None and algorithm.objective_diagnostics["head_refits"] == 0


def test_absolute_model_failure_reopens_scope_research_while_deployment_is_inconclusive(deployment_runtime):
    algorithm, _, _, _ = deployment_runtime
    goal, research = _start(algorithm)
    rows = algorithm._objective_evidence()
    frozen = deepcopy(research["candidate"])
    for offset, (error, gain) in enumerate(((.01, .02), (1., .2), (.01, .02), (.01, .02))):
        point = {"x1": .11 + offset * .13, "x2": .47, "x3": .78}
        prediction = _predict(goal.spec, frozen, point)
        target = prediction + sqrt(error)
        previous = target + sqrt(error + gain)
        index = 32 + offset
        request = goal.prepare_shared_control(rows, lambda features: {
            "features": features, "objective_mean": previous, "objective_noise": .001}, point,
            scope_start=0, stage=index, remaining_calls=64-index, purpose="optimization")
        assert request is not None
        context = {"global_index": index, "stage": index+1, "purpose": "optimization", "objective": target}
        rows.append({"observation": {"features": point, "target": -1.}, "context": context})
        goal.observe(point, [target, -1.], context=context)
    assert research["status"] == "inconclusive_future_predictions" and goal.accepted_revision is None
    goal._monitor(rows)
    current = goal.state["scope_research"]
    assert current["scope_start"] == 32 and current["phase"] == "collecting"
    assert not current.get("deployment")
    assert goal.state["scope_attempts"][-1]["candidate"] == frozen
    assert len(goal.state["scope_attempts"][-1]["validation"]) == 4
    assert any(event["action"] == "reopen_during_goal_deployment" for event in goal.state["events"])
    assert not algorithm.objective_scope_revisions and len(algorithm.history) == 32
