"""Provider protocol tests exercise real Python files, subprocesses and repair.

The deterministic providers here test the tool loop; the demonstration invokes
the actual configured Codex model and reports that run separately.
"""

import json
from math import sin

from hegel_machine.cognition.code_agent import CodeAgent, CodeAgentConfig, evaluate_python_model
from hegel_machine.cognition.measurement_lab import make_lab
from hegel_machine.cognition.models import Observation, ResearchConfig


FIT_SCRIPT = """import json
rows = json.load(open('observations.json'))
x = [[row['history'][-1]['__target__'] if row['history'] else 0.,
      row['features']['u'], row['features']['sensor_a']] for row in rows]
y = [row['target'] for row in rows]
m = [[sum(a[i]*a[j] for a in x) for j in range(3)] + [sum(a[i]*b for a,b in zip(x,y))] for i in range(3)]
for i in range(3):
    p = max(range(i,3), key=lambda j: abs(m[j][i]))
    m[i],m[p] = m[p],m[i]
    divisor = m[i][i]
    m[i] = [v/divisor for v in m[i]]
    for j in range(3):
        if i != j:
            scale = m[j][i]
            m[j] = [a-scale*b for a,b in zip(m[j],m[i])]
print(json.dumps([m[i][-1] for i in range(3)]))
"""


class RepairProtocolProvider:
    def next_action(self, request):
        index = len(request["events"])
        if index == 0:
            action, arguments = "query", {"sequence": [{"u": sin(index * 1.7)} for index in range(24)],
                                           "channels": ["sensor_a"]}
        elif index == 1:
            action, arguments = "write", {"path": "model.py", "code": "def predict(row):\n    return 1/0\n"}
        elif index == 2:
            action, arguments = "evaluate", {"model": "model.py"}
        elif index == 3:
            assert "ZeroDivisionError" in request["events"][-1]["feedback"]["stderr"]
            action, arguments = "run", {"code": FIT_SCRIPT}
        elif index == 4:
            assert request["events"][-1]["feedback"]["returncode"] == 0
            coefficients = json.loads(request["events"][-1]["feedback"]["stdout"])
            source = f"coefficients = {coefficients!r}\ndef predict(row):\n"
            source += "    state = 0.\n    for point in list(row['history'])+[row['features']]:\n"
            source += "        state = coefficients[0]*state + coefficients[1]*point['u'] + coefficients[2]*point['sensor_a']\n    return state\n"
            action, arguments = "write", {"path": "model.py", "code": source}
        else:
            action, arguments = "finish", {"model": "model.py", "channels": ["sensor_a"]}
        return {"action": action, "arguments": arguments,
                "usage": {"backend": "test_protocol", "total_tokens": 3}}


def test_python_agent_executes_fits_and_repairs_model_after_real_error(tmp_path):
    config = CodeAgentConfig(observation_budget=32, measurement_budget=64, compute_budget=1000)
    with make_lab(0, 46, backend="process") as laboratory:
        agent = CodeAgent(RepairProtocolProvider(), tmp_path, config)
        result = agent.run(laboratory)
        assert result["status"] == "completed"
        assert result["costs"]["observation_cost"] == result["costs"]["measurement_cost"] == 24
        assert result["costs"]["compute_cost"] == 48  # failed and repaired models were both actually run
        assert result["costs"]["python_execution_seconds"] > 0
        assert result["costs"]["reported_tokens"] == 18
        assert [event["action"] for event in result["events"]] == ["query", "write", "evaluate", "run", "write", "finish"]
        fresh = laboratory.evaluation_rows(("sensor_a",), seed=75, length=64, trajectories=2)
        before = dict(agent.costs)
        evaluation = evaluate_python_model(tmp_path, result["model"], fresh)
        assert evaluation["mse"] < 1e-20
        assert agent.costs == before
        assert "coefficients" in (tmp_path / "model.py").read_text()


def test_unavailable_model_token_usage_stops_further_calls_instead_of_counting_zero(tmp_path):
    class UnknownUsage:
        calls = 0

        def next_action(self, request):
            self.calls += 1
            return {"action": "query", "arguments": {"sequence": [{"u": .2}], "channels": []},
                    "usage": {"backend": "test_protocol", "total_tokens": None}}

    provider = UnknownUsage()
    result = CodeAgent(provider, tmp_path).run(make_lab())
    assert result["status"] == "token_usage_unavailable"
    assert provider.calls == result["costs"]["model_calls"] == 1
    assert result["costs"]["reported_tokens"] is None
    assert result["costs"]["observation_cost"] == 1


def test_budget_rejection_returns_action_feedback_without_acquiring_data(tmp_path):
    class TooMuch:
        def next_action(self, request):
            return {"action": "query", "arguments": {"sequence": [{"u": .2}] * 8, "channels": ["sensor_a"]},
                    "usage": {"backend": "test_protocol", "total_tokens": 1}}

    laboratory = make_lab()
    result = CodeAgent(TooMuch(), tmp_path, CodeAgentConfig(observation_budget=4, max_model_calls=2)).run(laboratory)
    assert result["status"] == "model_call_budget_exhausted"
    assert all(event["feedback"]["error"] == "observation budget exhausted" for event in result["events"])
    assert result["costs"]["observation_cost"] == laboratory.costs["observation_cost"] == 0


def test_reported_tokens_prevent_another_model_call_at_the_limit(tmp_path):
    class TokenLimit:
        def next_action(self, request):
            return {"action": "write", "arguments": {"path": "notes.txt", "code": "observations needed"},
                    "usage": {"backend": "test_protocol", "total_tokens": 5}}

    result = CodeAgent(TokenLimit(), tmp_path, CodeAgentConfig(token_budget=5)).run(make_lab())
    assert result["status"] == "token_budget_exhausted"
    assert result["costs"]["model_calls"] == 1 and result["costs"]["reported_tokens"] == 5


def test_evaluation_preserves_public_past_targets_but_withholds_current_target(tmp_path):
    (tmp_path / "model.py").write_text(
        "def predict(row):\n    assert 'target' not in row\n"
        "    return row['features']['u'] + row['history'][-1]['__target__']\n")
    row = Observation({"u": 2.}, 99., ({"u": 1., "__target__": 3.},))
    result = evaluate_python_model(tmp_path, "model.py", [row])
    assert result["predictions"] == [5.]
    assert result["mse"] == (99. - 5.) ** 2


def test_comparison_executes_both_arms_and_reuses_finished_results_without_new_calls(tmp_path, monkeypatch):
    from hegel_machine.cognition import code_comparison

    config = ResearchConfig(observation_budget=32, compute_budget=12000, max_rounds=1, seed=46)
    code_config = CodeAgentConfig(observation_budget=32, measurement_budget=160, compute_budget=12000)
    saved = []
    report = code_comparison.run_code_comparison(
        tmp_path, tasks=[0], config=config, code_config=code_config, measurement_budget=160,
        backend="local", provider_factory=lambda _: RepairProtocolProvider(), on_result=saved.append)
    result = report["results"][0]
    assert result["code_agent"]["status"] == "completed"
    assert result["code_agent"]["evaluation"]["mse"] < 1e-20
    for arm in ("code_agent", "hegel"):
        assert result[arm]["costs"]["observation_cost"] <= 32
        assert result[arm]["costs"]["measurement_cost"] <= 160
        assert result[arm]["evaluation"]["rows"] == 256
    assert saved[-1] == report

    def no_apparatus(*args, **kwargs):
        raise AssertionError("reusing a finished result launched another task")

    monkeypatch.setattr(code_comparison, "make_lab", no_apparatus)
    reused = code_comparison.run_code_comparison(
        tmp_path, tasks=[0], config=config, code_config=code_config, measurement_budget=160,
        backend="local", code_results=[result["code_agent"]], hegel_results=[result["hegel"]])
    assert reused == report
