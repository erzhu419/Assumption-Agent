#!/usr/bin/env python3
"""Use observed regional prediction errors to choose re-entry experiments."""

import argparse
import json
from pathlib import Path

import run_sc_kg_reentry_value_study as value


METHOD = "reentry_regional_v053"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--first-development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "replay", "all"), default="plan")
    args = parser.parse_args()
    value.METHOD = METHOD
    controls = {key: str((args.first_development / "runs" / f"{key}_reentry_value_v053.json").resolve())
                for key in value.CONTROLS}
    spec, _ = value.prepare(args.source_study, args.output, {
        "experiment": "regional_evidence_for_reentered_objective_discrimination",
        "frozen_baseline_source": str((args.first_development / "baseline_source_v052").resolve()),
        "frozen_method_source": str((args.output / "method_source_regional").resolve()),
        "first_development_reference": str((args.first_development / "replay_result_brief.json").resolve()),
        "existing_control_records": controls, "expected_new_optimizer_runs": 62,
        "expected_new_online_calls": 1984, "expected_new_physical_calls": 4288,
        "prior_variant_physical_calls": 416, "expected_batch_physical_calls": 4704,
        "method": "Restore v0.52 direct re-entry experiments. For the frozen candidate's complete real "
            "validation history, label a control supported when candidate_error <=9*max(cv_loss,1e-7) "
            "and refuted otherwise. Reuse condition_learning.learn_condition to learn an experimental "
            "applicability predicate. Outside that predicate, normalize the candidate/incumbent mean "
            "gap by sqrt(noise + max(0,mean(refuted candidate_error)-cv_loss)); inside or without a "
            "supported predicate retain the original noise normalization. Relearn only when new "
            "validation arrives and charge all fitting and scoring work. Never exclude a control, "
            "delete old evidence or change the objective-head adoption rule. Pending requests record "
            "selection_value, evidence_error and experimental-condition support/status.",
        "method_selection": "The first native-competition development variant regressed one terminal "
            "objective and increased mean objective prediction loss. Preserve its416 physical calls. "
            "This second development proposal addresses repeatedly choosing regions where this same "
            "candidate already failed. The independently fixed fresh27 seeds remain unobserved.",
        "selection": "Execute all eight actual re-entry planning paths. Reuse the first variant's five "
            "executed no-entry controls, which kept the full paid trajectories, final outcomes and "
            "predictions unchanged. The three strongly refuted candidates remain outside this branch.",
        "accounting": "Eight new historical runs plus54 fresh paired runs cost1984 online calls. Fresh "
            "shared acquisition costs384 initial and1920 evaluator calls, totaling4288 physical. "
            "Development costs256 online calls and is included once. Reuse the five previously executed "
            "controls and all old initialization, evaluator and baseline records. The first variant's416 "
            "calls remain separate; the complete batch costs4704. Report experimental-condition learning "
            "within controller compute, plus native KG timing and forecast rows separately.",
        "source_loading": "Freeze the current v0.52 twelve-module baseline and the thirteen-module "
            "regional package including condition_learning.py. The method changes objective_reentry.py "
            "and records its selection metadata in objective_research.py; the SC adapter is v0.52."})
    if args.phase == "plan":
        print(json.dumps({"status": "prepared_not_run", "new_development_runs": 8,
            "reused_controls": 5, "full_runs": 62, "full_physical_calls": 4288}), flush=True)
    else:
        selected = None if args.phase == "all" else spec["development_cases"]
        baseline, refs = value.references(args.source_study, selected)
        value.run(spec, args.output, baseline, refs, args.phase)


if __name__ == "__main__":
    main()
