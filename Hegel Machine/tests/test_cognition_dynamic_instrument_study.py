import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dynamic_instrument_study", ROOT / "scripts/run_dynamic_instrument_study.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_stage_random_seeds_do_not_change_the_same_apparatus_between_tasks():
    protocol = json.loads(study.PROTOCOL.read_text())
    tasks = study.task_sequence(151, protocol)
    assert [task["task_index"] for task in tasks] == [0, 1, 2, 3, 4]
    assert len({task["seed"] for task in tasks}) == 5
    assert {task["apparatus_seed"] for task in tasks} == {151}
    assert all(task["environment"] == "dynamic_instruments" for task in tasks)


def test_summary_separates_paid_learning_from_terminal_and_rejects_wrong_free_predictions(tmp_path):
    protocol = {"development_seeds": [151], "formal_seeds": [], "stages": ["source", "same"],
                "terminal": {"solved_threshold": 1e-6}}
    def report(arm):
        entries, evaluation = [], []
        for stage in range(2):
            paid = arm == "v074"
            entries.append({"task_position": stage, "actual_costs": {"observation_cost": 24,
                "measurement_cost": 72 if paid else 0, "queries": 3}, "instrument_models_after": [],
                "research": {"selected_channels": ["meter"] if paid else [], "selected_kind": "test",
                    "costs": {"observation_cost": 24, "measurement_cost": 72 if paid else 0,
                              "compute_cost": 100}, "actions": []}})
            evaluation.append({"loss": 0. if paid or stage == 0 else .2,
                "observation_cost": 256, "measurement_cost": 768 if paid else 0, "compute_cost": 256})
        return {"results": entries, "evaluation": evaluation}
    for arm in study.ARMS:
        study.write(tmp_path / "runs" / f"151_{arm}.json", report(arm))
    # Absolute artifact paths may live outside the checkout in this temporary run.
    original_root = study.ROOT
    study.ROOT = tmp_path
    try:
        summary = study.summarize(tmp_path, protocol)["cohorts"]["development"]
    finally:
        study.ROOT = original_root
    assert summary["v074"]["all"]["learning_costs"]["measurement_cost"] == 144
    assert summary["v074"]["all"]["terminal_costs"]["measurement_cost"] == 1536
    new = summary["dynamic_instruments"]["all"]
    assert new["measurement_free_predictions"] == 2
    assert new["measurement_free_solved"] == 1
    assert new["incorrect_measurement_free_predictions"] == 1
    assert new["learning_costs"]["observation_cost"] == 48
