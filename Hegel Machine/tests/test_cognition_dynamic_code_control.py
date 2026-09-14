"""Real Python execution checks the dynamic Code interface, not LLM quality."""

import json
from pathlib import Path
from random import Random

import pytest

from hegel_machine.cognition.code_agent import CodeAgentConfig
from hegel_machine.cognition.unified_code_control import UnifiedCodeSession


MODEL = """import numpy as np
def basis(row):
    assert 'target' not in row
    u, meter = row['features']['u'], row['features']['meter']
    previous = row['history'][-1]['meter'] if row['history'] else 0.
    return [1., u, meter, meter * meter, previous]
def fit(observations):
    global coefficients
    acquired = [row for row in observations if 'meter' in row['features']]
    inputs = [{'features': row['features'], 'history': row['history']} for row in acquired]
    coefficients = np.linalg.lstsq([basis(row) for row in inputs],
        [row['target'] for row in acquired], rcond=None)[0]
def predict(row):
    return float(np.dot(coefficients, basis(row)))
"""


def query(channels, length):
    random = Random(53)
    return {"sequence": [{"u": random.uniform(-1., 1.)} for _ in range(length)],
            "channels": channels}


def decision(action, arguments):
    return {"action": action, "arguments": arguments, "usage": {"total_tokens": 5}}


class FittingProvider:
    def next_action(self, request):
        capabilities = request["capabilities"]
        assert tuple(capabilities["feature_names"]) == ("u",)
        assert capabilities["feature_bounds"] == {"u": (-1., 1.)}
        assert capabilities["channels"][0]["name"] == "meter"
        assert capabilities["channels"][0]["cost_per_row"] == 3
        step = len(request["events"])
        if step == 0:
            return decision("query", query(["meter"], 32))
        if step == 1:
            return decision("query", query([], 3))
        assert request["observations"][0]["trajectory_start"]
        assert request["observations"][32]["trajectory_start"]
        if step == 2:
            if request["session_knowledge"]["files"]:
                assert "shared/model.py" in request["files"]
                assert len(request["session_knowledge"]["lessons"]) == 1
                return decision("run", {"code": "from shared.model import predict\nprint('shared-ready')"})
            return decision("write", {"path": "shared/model.py", "code": MODEL})
        if step == 3:
            return decision("evaluate", {"model": "shared/model.py", "channels": ["meter"]})
        assert request["events"][-1]["feedback"]["mse"] < 1e-20
        return decision("finish", {"model": "shared/model.py", "channels": ["meter"],
            "lesson": "Refit quadratic readings and prior readings on each task's acquired trajectories.",
            "reused_files": ["shared/model.py"] if request["session_knowledge"]["files"] else []})


def session(path, *, tasks=None, state=None, provider=FittingProvider, calls=5):
    return UnifiedCodeSession(path, state=state,
        tasks=None if state else tasks or [{"environment": "dynamic_instruments", "task_index": index,
            "seed": 113 + index, "apparatus_seed": 113} for index in (0, 3)],
        config=None if state else CodeAgentConfig(observation_budget=40, measurement_budget=96,
            compute_budget=256, execution_seconds=15, max_model_calls=calls),
        provider_factory=lambda _: provider())


def test_paid_dynamic_fit_resumes_shares_own_program_and_scores_independent_trajectories(tmp_path):
    original, checkpoints = session(tmp_path), []

    def interrupt(state):
        if state["active"] and len(state["active"]["agent_state"]["events"]) == 1:
            checkpoints.append(json.loads(json.dumps(state)))
            raise RuntimeError("after paid dynamic query")

    with pytest.raises(RuntimeError, match="paid dynamic query"):
        original.run(on_checkpoint=interrupt)
    restored = session(tmp_path, state=checkpoints[0])
    state = restored.run()
    assert state["status"] == "completed" and state["active"] is None
    assert state["costs"]["observation_cost"] == 70 and state["costs"]["measurement_cost"] == 192
    assert state["costs"]["compute_cost"] == 128  # Only the 32 compatible rows per evaluation.
    assert state["costs"]["model_calls"] == 10 and state["costs"]["reported_tokens"] == 50
    assert state["results"][1]["reused_files"] == ["shared/model.py"]
    assert state["results"][1]["events"][2]["feedback"]["stdout"] == "shared-ready\n"
    acquired = json.loads((tmp_path / "task_000" / "observations.json").read_text())
    assert acquired[0]["history"] == acquired[32]["history"] == []
    assert "meter" in acquired[31]["features"] and "meter" not in acquired[32]["features"]
    before = restored.to_dict()["knowledge"]
    terminal = restored.evaluate(length=12, trajectories=2)
    assert all(item["mse"] < 1e-20 and item["rows"] == 24 for item in terminal)
    assert all(item["measurement_cost"] == 72 for item in terminal)
    assert restored.to_dict()["knowledge"] == before


def test_control_only_prediction_projects_paid_current_and_history_but_fit_keeps_acquired_data(tmp_path):
    class ControlProvider:
        def next_action(self, request):
            step = len(request["events"])
            if step == 0:
                return decision("query", query(["meter"], 8))
            if step == 1:
                return decision("write", {"path": "model.py", "code": """def fit(rows):
    assert all('meter' in row['features'] for row in rows)
def predict(row):
    assert set(row['features']) == {'u'}
    assert all(set(point) <= {'u', '__target__'} for point in row['history'])
    return row['features']['u']
"""})
            return decision("finish", {"model": "model.py", "channels": []})

    control = session(tmp_path, provider=ControlProvider, calls=3,
        tasks=[{"environment": "dynamic_instruments", "task_index": 0, "seed": 113}])
    state = control.run()
    assert state["results"][0]["status"] == "completed"
    assert state["results"][0]["selected_channels"] == []
    assert state["costs"]["measurement_cost"] == 24
    result = control.evaluate(length=5, trajectories=2)[0]
    assert result["returncode"] == 0 and result["measurement_cost"] == 0


def test_later_failure_keeps_last_successful_python_and_its_paid_channel_view(tmp_path):
    class RetainedProvider:
        def next_action(self, request):
            step = len(request["events"])
            if step == 0:
                return decision("query", query(["meter"], 32))
            if step == 1:
                return decision("write", {"path": "model.py", "code": MODEL})
            if step == 2:
                return decision("evaluate", {"model": "model.py", "channels": ["meter"]})
            if step == 3:
                return decision("write", {"path": "model.py", "code": "def predict(row):\n    raise RuntimeError('failed revision')\n"})
            if step == 4:
                return decision("query", query([], 3))
            return decision("evaluate", {"model": "model.py", "channels": []})

    control = session(tmp_path, provider=RetainedProvider, calls=6,
        tasks=[{"environment": "dynamic_instruments", "task_index": 2, "seed": 127}])
    checkpoints = []

    def interrupt(state):
        if state["active"] and len(state["active"]["agent_state"]["events"]) == 3:
            checkpoints.append(json.loads(json.dumps(state)))
            raise RuntimeError("after successful Python evaluation")

    with pytest.raises(RuntimeError, match="successful Python evaluation"):
        control.run(on_checkpoint=interrupt)
    control = session(tmp_path, state=checkpoints[0], provider=RetainedProvider)
    state = control.run()
    result = state["results"][0]
    assert result["model_origin"] == "last_successful_evaluation"
    assert result["selected_channels"] == ["meter"] and result["model_source"] == MODEL
    assert "failed revision" in result["events"][-1]["feedback"]["stderr"]
    assert "failed revision" in (Path(result["research_workspace"]) / "model.py").read_text()
    assert (Path(result["workspace"]) / "model.py").read_text() == MODEL
    terminal = control.evaluate(length=12, trajectories=2)[0]
    assert terminal["returncode"] == 0 and terminal["mse"] < 1e-20
    assert terminal["measurement_cost"] == 72
