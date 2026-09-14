import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_instrument_knowledge_study.py"
SPEC = importlib.util.spec_from_file_location("instrument_knowledge_study", SCRIPT)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_changed_calibration_keeps_the_original_tasks_and_apparatus():
    tasks = study.tasks_for(71)
    assert tasks[:8] == study.composition_tasks(71)
    assert [task["readout"] for task in tasks[8:]] == ["total_event", "total_event", "window_total_event"]
    assert all(task["apparatus_seed"] == tasks[1]["apparatus_seed"] for task in tasks[8:])
    assert all(task["actuator_curvature"] == .31 for task in tasks[8:])
    assert tasks[9]["rename_prefix"] == "recalibrated_"
    assert len({task["seed"] for task in tasks}) == 11


def test_baseline_prefix_reuse_preserves_evidence_and_counts_only_new_work(tmp_path):
    tasks = study.tasks_for(71)
    costs = {"observation_cost": 3, "measurement_cost": 2, "compute_cost": 7}
    research = {"status": "completed", "costs": costs, "selected_channels": [],
                "selected_kind": "test_model", "actions": []}
    baseline = {"tasks": tasks[:8], "results": [{"research": research} for _ in range(8)],
        "evaluation": [{"task_position": index, "observation_cost": 256,
                        "measurement_cost": 12, "loss": 0.} for index in range(8)],
        "knowledge": {"source": "existing"}, "costs": {key: 8 * value for key, value in costs.items()}}
    source = tmp_path / "source.json"
    source.write_text(json.dumps(baseline))
    resumed = study.resumed_prefix(source, tasks)
    assert resumed["results"] == baseline["results"]
    assert resumed["evaluation"] == baseline["evaluation"]
    assert resumed["knowledge"] == baseline["knowledge"]
    assert resumed["pending_tasks"] == [8, 9, 10]
    assert json.loads(source.read_text()) == baseline
    resumed["results"].extend({"research": research} for _ in range(3))
    resumed["evaluation"].extend({"task_position": index, "observation_cost": 256,
                                  "measurement_cost": 12, "loss": 0.}
                                 for index in range(8, 11))
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "71_no_instrument_knowledge.json").write_text(json.dumps(resumed))
    summary = study.summarize(tmp_path, {"experiment": "test", "seeds": [71],
        "baseline": str(source), "tasks_by_seed": {"71": tasks},
        "arms": ["no_instrument_knowledge"], "deterministic_solved_threshold": 1e-6})
    assert summary["actual_new_learning_costs"] == {key: 3 * value for key, value in costs.items()}
    assert summary["actual_new_evaluation_rows"] == 3 * 256
    assert summary["actual_new_terminal_evaluation_costs"]["measurement_cost"] == 3 * 12
    assert summary["arms"]["no_instrument_knowledge"]["all"]["terminal_evaluation_costs"]["measurement_cost"] == 11 * 12
    assert summary["arms"]["no_instrument_knowledge"]["all"]["solved"] == 11
