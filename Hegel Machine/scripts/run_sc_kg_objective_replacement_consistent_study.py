#!/usr/bin/env python3
"""Test consistent paired advantage after the strict replacement development failure."""

import argparse
import json
from pathlib import Path

import run_sc_kg_objective_replacement_study as replacement


METHOD = "objective_replacement_consistent_v050"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict-development", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "replay", "all"), default="plan")
    args = parser.parse_args()
    replacement.METHOD = METHOD
    spec, baseline, refs = replacement.prepare(args.source_study, args.output, {
        "experiment": "independent_objective_gp_replacement_with_consistent_paired_evidence",
        "frozen_method_source": str((args.output / "method_source_consistent").resolve()),
        "prior_negative_development": str((args.strict_development / "replay_result_brief.json").resolve()),
        "method": "Keep independent objective-GP replacement and prospective paid-vector comparisons. "
            "Support a proposal when the original two-standard-error gain rule passes, or positive mean "
            "gain exceeds1e-7 and the one-sided exact sign probability is at most0.05. Compute signs only "
            "for gains beyond plus/minus1e-7; exclude ties. Require at least four distinct real comparison "
            "controls. Four wins do not satisfy the sign alternative (p=0.0625), five do (p=0.03125). "
            "Apply this same independent-replacement rule to comparisons against an accepted objective "
            "candidate. The objective-program adoption rule remains unchanged. All fitting, retention, "
            "continuation and incumbent-rebase behavior remains as in the frozen strict replacement variant.",
        "method_selection": "Selected after the original fifteen development runs showed a real "
            "ten-of-ten prospective improvement being blocked by heterogeneous squared-loss magnitude. "
            "Preserve that failed snapshot and its480 physical calls. Run the same fifteen cases first; "
            "proceed to the original153 affected cases/four controls and predeclared27 fresh cases only "
            "if development is promising. No fresh measurements existed when choosing this variant.",
        "budget_scope": "The211 runs and9056 physical calls in this protocol are for this variant alone, "
            "including its development15 once. The earlier strict replacement development cost480 more "
            "physical calls and is never represented as a run of this variant."})
    if args.phase == "plan":
        print(json.dumps({"status": "prepared_not_run", "affected": len(spec["affected_cases"]),
            "controls": len(spec["unchanged_controls"]), "reuse": len(spec["reused_cases"]),
            "development": len(spec["development_cases"]), "fresh": 27,
            "new_physical_calls": spec["expected_new_physical_calls"]}), flush=True)
    else:
        replacement.run(spec, args.output, baseline, refs, args.phase)


if __name__ == "__main__":
    main()
