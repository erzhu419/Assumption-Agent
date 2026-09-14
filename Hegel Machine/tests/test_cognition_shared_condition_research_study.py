"""Study accounting uses completed runs and separates discovery from scoring."""

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("shared_condition_study", ROOT / "scripts/run_shared_condition_research_study.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_declared_fresh_seeds_preserve_the_entire_original_task_and_evaluation_protocol():
    original = json.loads(study.transfer.PROTOCOL.read_text())
    current = json.loads(study.PROTOCOL.read_text())
    for key in ("cases", "stages", "public_prior", "task_generation", "research_config", "measurement_budget", "terminal"):
        assert current[key] == original[key]
    assert current["cohorts"]["replication"] == original["formal_seeds"]
    assert set(current["cohorts"]["generalization"]).isdisjoint(original["formal_seeds"] + original["development_seeds"])
    for seed in current["cohorts"]["replication"]:
        assert study.report_path(Path("unused"), "v072", seed, "replication").is_file()


def test_metrics_count_actual_learning_cost_and_failed_short_acceptance():
    task = {"evaluation": {"loss": .25, "rows": 256, "costs": {"observation_cost": 256}},
        "actual_queries": 2, "research": {"costs": {"observation_cost": 8, "measurement_cost": 0, "compute_cost": 50},
            "actions": [{"action": "validate_frozen_program", "short_trial": True, "accepted": True},
                {"action": "plan_transfer_condition_experiment", "compute_cost": 20,
                 "transfer_scope": {"sharing": {"condition_questions": 1, "callable_hypotheses": 3,
                     "new_prediction_evaluations": 8, "reused_prediction_evaluations": 16}}},
                {"action": "test_transfer_condition_predictions", "plan": {"hypotheses": [1, 2, 3]},
                 "evidence": {"compute_cost": 6}}]}}
    metric = study.task_metrics(task)
    total = study.aggregate([metric])
    assert total["solved"] == 0 and total["false_short_transfer"] == 1
    assert total["costs"]["observation_cost"] == 8
    assert total["condition_planning_compute"] == 20
    assert total["complete_hypothesis_evaluations"] == 3
    assert total["sharing"]["condition_questions"] == 1
    assert total["sharing"]["callable_hypotheses"] == 3
    assert total["sharing"]["reused_prediction_evaluations"] == 16
