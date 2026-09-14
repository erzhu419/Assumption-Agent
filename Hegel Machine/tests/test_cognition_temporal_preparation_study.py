import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("temporal_preparation_study", ROOT / "scripts/run_temporal_preparation_study.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_revisited_instrument_is_already_changed_during_paid_calibration():
    study.load_runtime(ROOT / "src")
    revisit = study.base.TemporalMethodLab("window", 5, 151, center=.13, response_gain=1.)
    drift = study.base.TemporalMethodLab("window", 3, 151, center=.13, response_gain=1.)
    repaired = study.base.TemporalMethodLab("window", 3, 151, center=.13, response_gain=1., calibration_rows=0)
    probe = study.base.Probe("first_calibration", tuple({"drive": value} for value in (.3, -.2, .6, .4, -.5, .2)))
    revisit_rows = revisit.acquire(probe, ("reading",)).observations
    repaired_rows = repaired.acquire(probe, ("reading",)).observations
    before_drift = drift.acquire(probe, ("reading",)).observations
    assert revisit_rows == repaired_rows
    assert revisit_rows != before_drift
    assert revisit.costs == {"observation_cost": 6, "measurement_cost": 18, "queries": 1}


def test_summary_separates_paid_revision_success_failure_and_later_use():
    def experiment(cost, feedback, *, parent=None, failed=False):
        return {"action": "paid_temporal_experiment", "plan": {"compute_cost": cost,
            "proposal": {"parent_method_id": parent} if parent else {}},
            "method_feedback": feedback, "assessment": {"compute_cost": 3,
                "calibration_status": "counterexample" if failed else "supported_on_probe"}}
    task = {"stage": 5, "costs": {"observation_cost": 66, "measurement_cost": 198, "compute_cost": 125, "queries": 5},
        "identified": True, "terminal": {"loss": 0., "rows": 128, "measurement_cost": 384}, "actions": [
            {"action": "paid_instrument_calibration", "compute_cost": 40},
            experiment(20, {"action": "revise_program_experiment_method", "method_id": "failed"}, parent="source", failed=True),
            experiment(30, {"action": "learn_program_experiment_method", "method_id": "variant"}, parent="source"),
            experiment(10, {"action": "use_program_experiment_method", "method_id": "variant"}),
            {"action": "learn_supported_hypothesis", "compute_cost": 16}]}
    records = [{"method_id": "source", "source_task_id": "151:window:0"},
               {"method_id": "variant", "source_task_id": "151:window:3", "parent_method_id": "source"}]
    protocol = json.loads(study.PROTOCOL.read_text())
    rows = study.task_rows({"seed": 151, "runtime_source": "actual_snapshot", "streams": [
        {"case": "window", "tasks": [task], "knowledge": {"program_experiment_methods": records}}]}, Path("actual.json"), protocol)
    result = study.aggregate(rows)
    assert result["revision_proposals"] == 2
    assert result["supported_revision_proposals"] == result["failed_revision_proposals"] == 1
    assert result["variant_uses"] == result["method_uses"] == 1
    assert result["original_method_uses"] == result["source_method_uses"] == 0
    assert result["costs"]["compute_cost"] == sum(result[key] for key in (
        "planning_compute", "calibration_compute", "assessment_compute", "knowledge_compute")) == 125
    assert result["costs"]["measurement_cost"] == 198
    assert result["terminal_measurement_cost"] == 384
