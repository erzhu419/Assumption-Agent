"""Python tool feedback and held-out scoring share the fixed public risk loss."""

import pytest

from hegel_machine.cognition.code_agent import CodeAgent, CodeAgentConfig, evaluate_python_model
from hegel_machine.cognition.measurement_lab import MeasurementBatch
from hegel_machine.cognition.models import Observation, TaskSpec
from hegel_machine.cognition.unified_code_control import UnifiedCodeSession


class ConstantResponseLab:
    channels = ()

    def __init__(self, q):
        self.spec = TaskSpec("fixed_risk", "Predict a measured response", ("u",), (), (),
                             risk_quantile=q, feature_bounds={"u": (-1., 1.)}, max_probe_length=32)

    def acquire(self, probe, channels=()):
        rows, history = [], []
        for features in probe.sequence:
            rows.append(Observation(dict(features), 0., tuple(history)))
            history.append({**features, "__target__": 0.})
        return MeasurementBatch(rows, len(rows), 0, ())

    def evaluation_rows(self, channels=(), *, seed, length, trajectories):
        return [Observation({"u": 0.}, 0.) for _ in range(length * trajectories)]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class ComparingProvider:
    def next_action(self, request):
        q = request["capabilities"]["risk_quantile"]
        assert request["capabilities"]["loss_kind"] == ("mse" if q == .5 else "pinball")
        step = len(request["events"])
        if step == 0:
            action, arguments = "query", {"sequence": [{"u": -.5}, {"u": .5}]}
        elif step in {1, 3}:
            if step == 3:
                feedback = request["events"][-1]["feedback"]
                assert feedback["risk_quantile"] == q
                assert feedback["mse"] == 1.
                assert feedback["loss"] == (1. if q == .5 else q)
            action, arguments = "write", {"path": "model.py",
                "code": f"def predict(row):\n    return {-1. if step == 1 else 1.}\n"}
        elif step in {2, 4}:
            action, arguments = "evaluate", {"model": "model.py"}
        else:
            previous = request["events"][-1]["feedback"]
            assert previous["mse"] == 1.
            assert previous["loss"] == (1. if q == .5 else 1. - q)
            action, arguments = "finish", {"model": "model.py", "channels": []}
        return {"action": action, "arguments": arguments, "usage": {"total_tokens": 1}}


@pytest.mark.parametrize("q", [.5, .9])
def test_real_python_feedback_uses_public_fixed_risk_target(tmp_path, q):
    agent = CodeAgent(ComparingProvider(), tmp_path,
                      CodeAgentConfig(observation_budget=2, max_model_calls=6))
    report = agent.run(ConstantResponseLab(q))
    assert report["status"] == "completed"
    assert report["risk_quantile"] == q
    under, over = [event["feedback"] for event in report["events"] if event["action"] == "evaluate"]
    assert under["mse"] == over["mse"] == 1.
    if q == .9:
        assert under["loss"] == pytest.approx(9 * over["loss"])
    else:
        assert under["loss"] == over["loss"] == 1.
    retained = agent.to_dict()["evaluated_model"]
    assert retained["training_loss"] == over["loss"]
    assert retained["training_mse"] == over["mse"]
    assert retained["risk_quantile"] == q


def test_shared_code_session_keeps_training_and_final_loss_contract_aligned(tmp_path):
    session = UnifiedCodeSession(tmp_path, tasks=[{"seed": 71, "q": .9}, {"seed": 73, "q": .5}],
        config=CodeAgentConfig(observation_budget=2, max_model_calls=6),
        provider_factory=lambda _: ComparingProvider(), lab_factory=lambda item: ConstantResponseLab(item["q"]))
    report = session.run()
    scores = session.evaluate(length=3, trajectories=2)
    for result, score in zip(report["results"], scores):
        training = result["events"][-1]["feedback"]
        assert score["returncode"] == 0
        assert score["loss"] == pytest.approx(training["loss"])
        assert score["risk_quantile"] == training["risk_quantile"]
        assert score["loss_kind"] == training["loss_kind"]
        assert score["mse"] == training["mse"] == 1.


def test_direct_evaluator_without_new_keyword_preserves_squared_loss(tmp_path):
    (tmp_path / "model.py").write_text("def predict(row):\n    return 2.\n")
    result = evaluate_python_model(tmp_path, "model.py", [Observation({"u": 0.}, 5.)])
    assert result["loss"] == result["mse"] == 9.
    assert result["loss_kind"] == "mse"
    assert result["risk_quantile"] == .5
