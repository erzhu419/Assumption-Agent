"""Shared fitting is judged beyond a common inactive observation prefix."""

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("shared_transfer_fitting_study", ROOT / "scripts/run_shared_transfer_fitting_study.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_common_inactive_predictions_do_not_make_distinct_laws_interchangeable():
    transfer = study.transfer
    transfer.load_runtime(ROOT / "src")
    protocol = json.loads(study.PROTOCOL.read_text())
    for stage, correct_index, wrong_index in ((0, 1, 4), (2, 2, 1), (3, 5, 2), (4, 1, 5)):
        settings = transfer.task_settings(protocol, 151, "narrow_windows", stage)
        laboratory = transfer.TransferLaboratory(settings)
        candidates = transfer.prior_programs(laboratory.spec)
        correct, wrong = candidates[correct_index], candidates[wrong_index]
        inactive = laboratory.acquire(transfer.Probe("inactive_prefix", (
            {name: 0. for name in settings["names"]},))).observations[0]
        assert correct.predict(inactive) == wrong.predict(inactive) == inactive.target == 0.
        right_result = transfer.evaluate(settings, {"program": correct.to_dict()}, protocol, 900 + stage)
        wrong_result = transfer.evaluate(settings, {"program": wrong.to_dict()}, protocol, 900 + stage)
        assert right_result["loss"] == 0.
        assert wrong_result["loss"] > protocol["terminal"]["solved_threshold"]


def test_fitting_comparison_retains_prior_budgets_and_full_sequence_settings():
    previous = json.loads(study.previous.PROTOCOL.read_text())
    current = json.loads(study.PROTOCOL.read_text())
    for key in ("cohorts", "cases", "stages", "public_prior", "task_generation", "research_config", "measurement_budget", "terminal"):
        assert current[key] == previous[key]


def test_summary_keeps_nested_work_separate_and_missing_old_telemetry_unmeasured():
    def task(compute, actions):
        return {"actual_queries": 3, "evaluation": {"loss": 0., "rows": 256},
            "research": {"costs": {"observation_cost": 24, "measurement_cost": 0, "compute_cost": compute},
                         "actions": actions}}

    old = task(100, [{"action": "refit_transfer_after_experiment", "compute_cost": 50}])
    new = task(60, [
        {"action": "shared_transfer_fit", "compute_cost": 20, "observation_rows": 8, "candidate_count": 4,
         "sharing": {"linear_solves": 1, "linear_solve_reuses": 3}},
        {"action": "shared_transfer_validation", "compute_cost": 10, "observation_rows": 8,
         "candidate_count": 4, "sharing": {"loss_row_evaluations": 8}},
        {"action": "refit_transfer_after_experiment", "compute_cost": 30}])
    later = task(80, [
        {"action": "shared_transfer_fit", "compute_cost": 35, "observation_rows": 16, "candidate_count": 4,
         "sharing": {"linear_solves": 2, "linear_solve_reuses": 2, "view_resets": 1}},
        {"action": "refit_transfer_after_experiment", "compute_cost": 45}])
    old_metric = study.task_metrics(old)
    assert old_metric["fitting"]["shared_transfer_fit"]["events"] == 0
    assert old_metric["fitting"]["shared_transfer_fit"]["sharing"] == {}
    total = study.aggregate([old_metric, study.task_metrics(new), study.task_metrics(later)])
    assert total["costs"] == {"observation_cost": 72, "measurement_cost": 0, "compute_cost": 240}
    assert total["refit_compute"] == 125
    assert total["fitting"]["shared_transfer_fit"]["compute_cost"] == 55
    assert total["fitting"]["shared_transfer_fit"]["sharing"] == {
        "linear_solves": 3, "linear_solve_reuses": 5, "view_resets": 1}
    assert total["fitting"]["shared_transfer_validation"]["compute_cost"] == 10
