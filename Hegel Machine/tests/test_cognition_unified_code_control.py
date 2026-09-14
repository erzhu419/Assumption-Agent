"""Protocol tests use a mock provider; they are not Code Agent outcome evidence."""

import json

import pytest

from hegel_machine.cognition.code_agent import CodeAgentConfig
from hegel_machine.cognition.measurement_lab import MeasurementLab
from hegel_machine.cognition.unified_code_control import UnifiedCodeSession


class SharedProvider:
    def next_action(self, request):
        # The task generator's private settings cannot become prior lessons.
        knowledge = json.dumps(request["session_knowledge"])
        assert "private_mechanism" not in knowledge
        assert "task_index" not in knowledge
        assert "profile" not in knowledge
        step = len(request["events"])
        if step == 0:
            action, arguments = "query", {"sequence": [{"u": -.4}, {"u": .6}], "channels": ["sensor_a"]}
        elif step == 1:
            inherited = request["session_knowledge"]["files"]
            if inherited:
                assert "shared/model.py" in request["files"]
                action, arguments = "run", {"code": "from shared.model import predict\nprint(predict({'features': {'u': .5}}))"}
            else:
                action, arguments = "write", {"path": "shared/model.py",
                    "code": "def predict(row):\n    return row['features']['u']\n"}
        else:
            action, arguments = "finish", {"model": "shared/model.py", "channels": [],
                "lesson": "A control-only starting model; refit before adoption.",
                "reused_files": ["shared/model.py"] if request["session_knowledge"]["files"] else []}
        return {"action": action, "arguments": arguments, "usage": {"total_tokens": 7}}


def session(tmp_path, *, state=None, laboratory_factory=None):
    return UnifiedCodeSession(tmp_path, state=state,
        tasks=None if state else [
            {"task_index": 0, "seed": 113, "profile": "private_mechanism"},
            {"task_index": 1, "seed": 127, "profile": "private_mechanism"}],
        config=None if state else CodeAgentConfig(observation_budget=2, measurement_budget=2,
            compute_budget=32, max_model_calls=3, execution_seconds=10),
        provider_factory=lambda _: SharedProvider(),
        lab_factory=laboratory_factory or (lambda item: MeasurementLab(item["task_index"], item["seed"])))


def test_common_code_sequence_restores_paid_prefix_and_reuses_own_python(tmp_path):
    original = session(tmp_path)
    checkpoints = []

    def interrupt(state):
        if state["active"] and len(state["active"]["agent_state"]["events"]) == 1:
            checkpoints.append(json.loads(json.dumps(state)))
            raise RuntimeError("interrupted after paid query")

    with pytest.raises(RuntimeError, match="paid query"):
        original.run(on_checkpoint=interrupt)
    restored = session(tmp_path, state=checkpoints[0])
    result = restored.run()
    assert result["status"] == "completed"
    assert result["costs"]["observation_cost"] == 4
    assert result["costs"]["measurement_cost"] == 4
    assert result["costs"]["model_calls"] == 6
    assert result["costs"]["reported_tokens"] == 42
    assert result["results"][1]["reused_files"] == ["shared/model.py"]
    assert result["results"][1]["events"][1]["feedback"]["stdout"] == "0.5\n"
    assert restored.run()["costs"] == result["costs"]


def test_final_evaluation_waits_for_stream_and_does_not_enter_knowledge(tmp_path):
    calls = []

    class EvaluationLab(MeasurementLab):
        def evaluation_rows(self, *args, **kwargs):
            calls.append(self.task_index)
            return super().evaluation_rows(*args, **kwargs)

    control = session(tmp_path,
        laboratory_factory=lambda item: EvaluationLab(item["task_index"], item["seed"]))
    control.run(max_tasks=1)
    with pytest.raises(ValueError, match="learning sequence"):
        control.evaluate()
    assert calls == []
    control.run()
    before = json.dumps(control.to_dict()["knowledge"], sort_keys=True)
    scores = control.evaluate(length=5, trajectories=2)
    assert [row["rows"] for row in scores] == [10, 10]
    assert all(row["mse"] >= 0 for row in scores)
    assert calls == [0, 1]
    assert json.dumps(control.to_dict()["knowledge"], sort_keys=True) == before
    assert control.evaluate(length=5, trajectories=2) == scores
    assert calls == [0, 1]


def test_code_loop_obeys_published_paid_instrument_limit(tmp_path):
    class OverMeasurement(SharedProvider):
        def next_action(self, request):
            decision = super().next_action(request)
            if len(request["events"]) == 0:
                decision["arguments"]["channels"] = ["sensor_a", "sensor_b"]
            return decision

    control = UnifiedCodeSession(tmp_path, tasks=[{"task_index": 0, "seed": 113}],
        config=CodeAgentConfig(observation_budget=2, measurement_budget=2, max_model_calls=1),
        lab_factory=lambda item: MeasurementLab(**item), provider_factory=lambda _: OverMeasurement())
    result = control.run()
    assert result["results"][0]["events"][0]["feedback"] == {"error": "measurement budget exhausted"}
    assert result["costs"]["measurement_cost"] == result["costs"]["observation_cost"] == 0


def test_default_common_laboratory_runs_the_existing_python_tool_loop(tmp_path):
    control = UnifiedCodeSession(tmp_path,
        tasks=[{"task_index": 0, "seed": 557, "profile": "confirmation"}],
        config=CodeAgentConfig(observation_budget=2, measurement_budget=2, max_model_calls=3),
        provider_factory=lambda _: SharedProvider())
    state = control.run()
    assert state["results"][0]["status"] == "completed"
    assert state["costs"]["observation_cost"] == state["costs"]["measurement_cost"] == 2
    assert control.evaluate(length=4, trajectories=1)[0]["returncode"] == 0


def test_agent_written_shared_binary_artifact_is_available_to_successor(tmp_path):
    class BinaryKnowledge(SharedProvider):
        def next_action(self, request):
            decision = super().next_action(request)
            if decision["action"] == "write":
                decision["arguments"]["code"] = (
                    "from pathlib import Path\n"
                    "def fit(observations):\n"
                    "    Path('shared/calibration.bin').write_bytes(bytes([173, 0, 255]))\n"
                    "def predict(row):\n"
                    "    return row['features']['u']\n")
            if decision["action"] == "run":
                assert "shared/calibration.bin" in request["session_knowledge"]["shared_artifacts"]
                decision["arguments"]["code"] = "from pathlib import Path\nprint(list(Path('shared/calibration.bin').read_bytes()))"
            return decision

    control = UnifiedCodeSession(tmp_path,
        tasks=[{"task_index": 0, "seed": seed} for seed in (557, 571)],
        config=CodeAgentConfig(observation_budget=2, measurement_budget=2, max_model_calls=3),
        provider_factory=lambda _: BinaryKnowledge())
    result = control.run()
    assert result["results"][1]["events"][1]["feedback"]["stdout"] == "[173, 0, 255]\n"
