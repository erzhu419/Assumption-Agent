import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dynamic_code_comparison", ROOT / "scripts/run_dynamic_code_comparison.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_terminal_waits_for_both_full_learning_sequences(tmp_path):
    for arm, completed in (("code", 1), ("hegel", 2)):
        study.save(tmp_path / f"{arm}_checkpoint.json", {"session": {
            "tasks": [{}, {}], "results": [{}] * completed}})
    with pytest.raises(ValueError, match="both complete learning sequences"):
        study.evaluate(tmp_path, {}, 151)
    assert not (tmp_path / "comparison.json").exists()


def test_unfinished_successful_model_keeps_its_terminal_score_and_separate_costs():
    protocol = json.loads((ROOT / "config/cognition_dynamic_code_comparison.json").read_text())
    costs = {"observation_cost": 20, "measurement_cost": 60, "compute_cost": 20,
             "model_calls": 12, "reported_tokens": 3000, "python_execution_seconds": .4}
    entry = {"status": "model_call_budget_exhausted", "model": "model.py",
        "model_origin": "last_successful_evaluation", "selected_channels": [], "costs": costs,
        "actual_costs": {"observation_cost": 20, "measurement_cost": 60, "queries": 1},
        "learning_wall_seconds": 30., "reused_files": [],
        "events": [{"action": "evaluate", "usage": {"input_tokens": 2800, "output_tokens": 200,
            "cached_input_tokens": 1000, "total_tokens": 3000, "wall_time_seconds": 29., "model": "real_provider_model"}}]}
    report = {"learning_wall_seconds": 30., "session": {"tasks": [{}], "results": [entry],
        "status": "completed", "costs": costs, "evaluation": [{"task_position": 0, "status": "completed",
            "loss": 1e-12, "mse": 1e-12, "observation_cost": 256, "measurement_cost": 0, "compute_cost": 256}]}}
    summary = study.summarize({"seed": 151, "status": "evaluated", "arms": {"code": report}}, protocol)["arms"]["code"]
    assert summary["solved"] == summary["fallback_evaluated_models"] == summary["measurement_free_solved"] == 1
    assert summary["tasks"][0]["status"] == "model_call_budget_exhausted"
    assert summary["mean_loss"] == 1e-12
    assert summary["learning_costs"]["observation_cost"] == 20
    assert summary["terminal_costs"]["observation_cost"] == 256
    assert summary["provider_usage"]["total_tokens"] == 3000
