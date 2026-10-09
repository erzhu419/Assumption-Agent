#!/usr/bin/env python3
"""Learn selective preservation or reopening of a parent model's instrument."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.temporal_selective_practice import (
    SelectiveMethodLearner, selective_revision_evidence,
)
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json
from scripts.run_temporal_counterexample_learning import run_study
from scripts.run_temporal_failure_selectivity import CALIBRATIONS, shifted_task


PREVIOUS = ROOT / "artifacts/temporal_evidence_scope_learning_luna_20261006"


def protocol():
    plan = json.loads((PREVIOUS / "protocol.json").read_text())
    cases = deepcopy(plan["frozen_cases"])
    for index in range(2):
        for prior in CALIBRATIONS:
            for change in ("response_shift", "instrument_shift", "joint_shift"):
                serial = len(cases)
                cases.append({"case_id": f"episode_{serial:02d}", "prior": prior,
                    "change": change, "unchanged": False,
                    "task": shifted_task(prior, change, 187106007 + 1009 * index, serial)})
    plan.pop("preceding_author_phase", None)
    plan.update(study="llm_temporal_selective_revision",
        seed_method=str(PREVIOUS / "study/learning/result.json"),
        counterexample_case="case_b", frozen_cases=cases, regression_case_count=32,
        learning="Start from the actually published evidence-scoped method and its paid response-change "
            "practice trace. The previous method succeeds by refitting both components. Learn when "
            "fresh evidence permits retaining its earlier supporting instrument and when evidence "
            "requires reopening it. The source, conditions, experiments and repairs are model-authored.",
        publication="One exact source must confirm every opaque practice case, reuse an unchanged "
            "predictor, and demonstrate both post-parent-failure recovery paths: current measurements "
            "support the preserved instrument used by a revised response without instrument fitting; "
            "or current measurements refute it before instrument and response reconstruction. "
            "Criteria inspect paid execution only, not case names or hidden change labels.",
        transfer="Keep all thirty-two preceding cases exactly as regression controls. Add twelve "
            "preselected new-calibration cases: two new parameter seeds, two original donors and three "
            "change types. Old and new methods use identical observations limits, operators, confirmation "
            "and evaluation seeds, with no LLM calls or practice-fitted numerical models in transfer.",
        limitations="Explicitly requested selective repair in bounded recurrent families and a disclosed "
            "checkpoint. Old cases are regression controls. New parameter realizations are not new "
            "structural families. Reopening instrument research does not identify whether response "
            "coefficients also changed. Learning costs are additional; prior failed studies remain recorded.")
    return plan


def transfer_evidence(output, plan, arm):
    roles = {"first": "shared/meter.json", "second": "shared/current_instrument.json"}
    results = []
    for index, case in enumerate(plan["frozen_cases"]):
        path = Path(output) / "study" / arm / case["case_id"] / "training.json"
        task = json.loads(path.read_text())["tasks"][0]
        results.append({"case_id": case["case_id"], "change": case["change"],
            "cohort": "regression" if index < plan["regression_case_count"] else "new_calibration",
            **selective_revision_evidence(task, roles[case["prior"]])})
    return results


def run_experiment(output, *, provider_factory, auth_mode=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise ValueError("use a new output directory")
    plan = protocol()
    learned = json.loads(Path(plan["seed_method"]).read_text())
    trial = next(trial for trial in learned["practice_trials"]
        if trial["case"] == "case_b" and trial["status"] == "completed"
        and trial["method_source"] == learned["method_source"])
    source = PREVIOUS / "study/learning" / trial["directory"] / "result.json"
    counterexample = output / "paid_recovery_trace.json"
    write_json(counterexample, {"tasks": [json.loads(source.read_text())], "evaluation": None})
    plan["counterexample"] = str(counterexample.resolve())
    plan["supplied_trace_origin"] = str(source)
    write_json(output / "protocol.json", plan)
    for name in ("scripts/run_temporal_selective_revision.py",
                 "src/hegel_machine/cognition/temporal_selective_practice.py"):
        path = output / "source_at_run" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path)
    result = run_study(output / "study", provider_factory=provider_factory,
        auth_mode=auth_mode, study_protocol=plan, learner_class=SelectiveMethodLearner)
    if result["status"] == "completed":
        result["selective_transfer"] = {
            arm: transfer_evidence(output, plan, arm) for arm in ("original", "revised")}
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    auth = verify_codex_subscription()
    print(json.dumps(run_experiment(args.output, auth_mode=auth,
        provider_factory=lambda path: CodexCodeProvider(path, model="gpt-6-luna",
            reasoning_effort="medium", timeout_seconds=180)), indent=2))


if __name__ == "__main__":
    main()
