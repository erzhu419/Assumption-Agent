import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("temporal_method_study", ROOT / "scripts/run_temporal_method_study.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_paid_event_exit_distinguishes_the_event_law_from_instantaneous_alternatives():
    study.load_runtime(ROOT / "src")
    protocol = json.loads(study.PROTOCOL.read_text())
    laboratory = study.TemporalMethodLab("event", 2, center=.27, response_gain=2.)
    programs = study.candidate_programs(laboratory, .27, protocol)
    observations = laboratory.evaluation_rows(seed=1117)
    losses = [sum((program.predict(row) - row.target) ** 2 for row in observations) for program in programs]
    assert len(programs) == 5
    assert len([error for error in losses if error == 0.]) == 1
    assert any(row.target < 0. for row in observations)


def test_failed_method_replay_is_a_restriction_and_never_a_successful_use(tmp_path):
    task = {"stage": 0, "costs": {"observation_cost": 108, "measurement_cost": 324,
            "compute_cost": 171, "queries": 6}, "identified": True, "terminal": {"loss": 0.},
        "methods_after": 2, "concepts_after": 1, "actions": [
            {"action": "paid_instrument_calibration", "compute_cost": 40},
            {"action": "paid_temporal_experiment", "plan": {"compute_cost": 20, "fresh_synthesis_skipped": True},
             "assessment": {"compute_cost": 9, "calibration_status": "counterexample"},
             "method_feedback": {"action": "revise_program_experiment_method"}},
            {"action": "paid_instrument_calibration", "compute_cost": 40},
            {"action": "paid_temporal_experiment", "plan": {"compute_cost": 40, "fresh_synthesis_skipped": False},
             "assessment": {"compute_cost": 9, "calibration_status": "supported_on_probe"},
             "method_feedback": {"action": "learn_program_experiment_method"}},
            {"action": "learn_supported_hypothesis", "compute_cost": 13}]}
    for arm in study.ARMS:
        study.write(tmp_path / "runs" / f"151_{arm}.json", {"streams": [{"case": "window", "tasks": [task]}]})
    protocol = {"development_seeds": [151], "formal_seeds": [], "cases": ["window"],
                "stages": ["calibration_drift"], "terminal": {"solved_threshold": 1e-6}}
    old_root = study.ROOT
    study.ROOT = tmp_path
    try:
        result = study.summarize(tmp_path, protocol)["cohorts"]["development"]["temporal_methods"]["all"]
    finally:
        study.ROOT = old_root
    assert result["costs"]["compute_cost"] == 171
    assert sum(result[key] for key in ("planning_compute", "calibration_compute", "assessment_compute", "knowledge_compute")) == 171
    assert result["fresh_synthesis_skipped"] == result["method_revisions"] == result["recalibrations"] == 1
    assert result["method_uses"] == result["cross_task_method_uses"] == 0
    assert result["methods_learned"] == 1
