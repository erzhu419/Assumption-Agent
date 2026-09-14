"""Public-interface checks with scripted providers and actual Python execution."""

import json

import pytest

from hegel_machine.cognition.code_agent import CodeAgent, CodeAgentConfig
from hegel_machine.cognition.unified_code_control import UnifiedCodeSession
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab


def descriptor(request, channels):
    capabilities = request["capabilities"]
    visible = json.dumps({"capabilities": capabilities,
                          "knowledge": request["session_knowledge"]})
    for private_key in ("readout", "variant", "task_index", "profile", "_radius", "_coupling"):
        assert f'"{private_key}"' not in visible
    views = capabilities["measurement_views"]
    assert [view["channels"] for view in views] == [[], ["coarse"], ["fine"]]
    return next(view for view in views if view["channels"] == channels)


def sequence(request):
    bounds = request["capabilities"]["feature_bounds"]
    return [{name: low + fraction * (high-low) for name, (low, high) in bounds.items()}
            for fraction in (.2, .8)]


def total_model(request, channels):
    schema = json.loads(descriptor(request, channels)["context"]["object_schema"])
    names = [fields[role] for fields in schema["fields"].values() for role in ("q", "u")]
    return (f"FIELDS = {names!r}\n"
            "def predict(row):\n"
            "    return sum(row['features'][name] for name in FIELDS)\n")


def decision(action, arguments):
    return {"action": action, "arguments": arguments, "usage": {"total_tokens": 1}}


@pytest.mark.parametrize("final_channels", [["coarse"], ["fine"]])
def test_common_code_controller_queries_alternative_views_and_scores_selected_features(tmp_path, final_channels):
    class Provider:
        def next_action(self, request):
            step = len(request["events"])
            view = descriptor(request, final_channels)
            assert set(view["feature_names"]).isdisjoint(request["capabilities"]["feature_bounds"])
            if step < 2:
                return decision("query", {"sequence": sequence(request),
                    "channels": ["coarse"] if step == 0 else ["fine"]})
            if step == 2:
                assert request["events"][-1]["feedback"]["channels"] == ["fine"]
                return decision("write", {"path": "model.py", "code": total_model(request, final_channels)})
            return decision("finish", {"model": "model.py", "channels": final_channels})

    control = UnifiedCodeSession(tmp_path,
        tasks=[{"environment": "objects", "readout": "total", "seed": 71},
               {"environment": "objects", "readout": "total", "seed": 97, "rename_prefix": "later_"}],
        config=CodeAgentConfig(observation_budget=4, measurement_budget=30, max_model_calls=4),
        provider_factory=lambda _: Provider())
    report = control.run()
    assert report["status"] == "completed"
    assert report["costs"]["observation_cost"] == 8
    assert report["costs"]["measurement_cost"] == 60
    assert all(result["selected_channels"] == final_channels for result in report["results"])
    before = json.dumps(report["knowledge"], sort_keys=True)
    scores = control.evaluate(length=8, trajectories=2)
    assert json.dumps(control.to_dict()["knowledge"], sort_keys=True) == before
    for result, score in zip(report["results"], scores):
        training = result["events"][-1]["feedback"]
        assert training["returncode"] == score["returncode"] == 0
        assert training["loss_kind"] == score["loss_kind"] == "mse"
        assert training["risk_quantile"] == score["risk_quantile"] == .5
        assert training["loss"] < 1e-25 and score["loss"] < 1e-25
        assert score["observation_cost"] == 16
        assert score["measurement_cost"] == 16 * (3 if final_channels == ["coarse"] else 12)
    # All purchased rows remain accessible even when finish selects one view.
    rows = json.loads((tmp_path / "task_000" / "observations.json").read_text())
    assert len(rows) == 4 and len(rows[0]["features"]) == 4 and len(rows[-1]["features"]) == 16


def test_fine_only_python_model_must_be_revised_before_claiming_coarse_measurements(tmp_path):
    class Provider:
        def next_action(self, request):
            step = len(request["events"])
            descriptor(request, ["fine"])
            if step == 0:
                return decision("query", {"sequence": sequence(request), "channels": ["fine"]})
            if step in (1, 4):
                channels = ["fine"] if step == 1 else ["coarse"]
                if step == 4:
                    assert request["events"][-1]["feedback"]["returncode"] != 0
                return decision("write", {"path": "model.py", "code": total_model(request, channels)})
            return decision("evaluate" if step == 2 else "finish",
                {"model": "model.py", "channels": ["fine"] if step == 2 else ["coarse"]})

    agent = CodeAgent(Provider(), tmp_path,
        CodeAgentConfig(observation_budget=2, measurement_budget=24, max_model_calls=6))
    report = agent.run(UnifiedObjectLab(readout="total", seed=71))
    assert report["status"] == "completed" and report["selected_channels"] == ["coarse"]
    assert report["events"][3]["feedback"]["returncode"] != 0
    assert report["events"][-1]["feedback"]["loss"] < 1e-25
    assert report["costs"]["observation_cost"] == 2
    assert report["costs"]["measurement_cost"] == 24


def test_incompatible_view_query_returns_feedback_and_spends_no_observations(tmp_path):
    class Provider:
        def next_action(self, request):
            step = len(request["events"])
            descriptor(request, ["coarse"])
            if step < 2:
                if step == 1:
                    assert "choose the command view" in request["events"][0]["feedback"]["error"]
                    assert request["costs"]["observation_cost"] == request["costs"]["measurement_cost"] == 0
                return decision("query", {"sequence": sequence(request),
                    "channels": ["coarse", "fine"] if step == 0 else ["coarse"]})
            if step == 2:
                return decision("write", {"path": "model.py", "code": total_model(request, ["coarse"])})
            return decision("finish", {"model": "model.py", "channels": ["coarse"]})

    agent = CodeAgent(Provider(), tmp_path,
        CodeAgentConfig(observation_budget=2, measurement_budget=30, max_model_calls=4))
    report = agent.run(UnifiedObjectLab(readout="total", seed=71))
    assert report["status"] == "completed"
    assert report["costs"]["observation_cost"] == 2
    assert report["costs"]["measurement_cost"] == 6
