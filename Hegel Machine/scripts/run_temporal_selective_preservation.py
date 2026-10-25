#!/usr/bin/env python3
"""Preserve initial-instrument recovery while learning selective parent repair."""

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
from hegel_machine.cognition.temporal_selective_practice import SelectiveMethodLearner
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json
from scripts.run_temporal_calibration_transfer import run_frozen
from scripts.run_temporal_counterexample_learning import run_study
from scripts.run_temporal_failure_selectivity import CALIBRATIONS, shifted_task
from scripts.run_temporal_selective_revision import transfer_evidence


PREVIOUS = ROOT / "artifacts/temporal_selective_revision_luna_20261006"


def protocol():
    plan = json.loads((PREVIOUS / "protocol.json").read_text())
    plan.pop("supplied_trace_origin", None)
    previous = json.loads((PREVIOUS / "summary.json").read_text())
    # This observed regression becomes an explicit development case. Its stable
    # checkpoint keeps the public interface common without changing its physics.
    regression = next(case for case in previous["revised_method"]["cases"]
        if not case["confirmed_and_correct"] and case["prior"] == "first")
    old_case = next(case for case in plan["frozen_cases"] if case["case_id"] == regression["case_id"])
    practice = deepcopy(old_case["task"])
    practice["calibration_change"] = {"after_rows": 72,
        "apparatus_seed": practice["apparatus_seed"],
        "response_seed": practice.get("response_seed", practice["apparatus_seed"])}
    plan["practice_cases"]["case_e"] = practice
    plan["comparison_method"] = plan["seed_method"]
    plan["seed_method"] = str(PREVIOUS / "study/learning/result.json")
    plan["regression_case_count"] = len(plan["frozen_cases"])
    for index in range(2):
        for prior in CALIBRATIONS:
            for change in ("response_shift", "instrument_shift", "joint_shift"):
                serial = len(plan["frozen_cases"])
                plan["frozen_cases"].append({"case_id": f"episode_{serial:02d}", "prior": prior,
                    "change": change, "unchanged": False,
                    "task": shifted_task(prior, change, 217106007 + 1009 * index, serial)})
    plan.update(study="llm_temporal_selective_preservation", counterexample_case="case_e",
        preceding_selective_phase={"path": str(PREVIOUS / "summary.json"),
            "confirmed_and_correct": previous["revised_method"]["confirmed_and_correct"],
            "author_costs": previous["new_learning_costs"]},
        learning="The preceding selective program forgot an earlier initial-refutation recovery. "
            "One actual failed replay supplies paid feedback on a fifth opaque practice case. "
            "The original four cases and selective repair objective remain unchanged. The LLM "
            "must revise its own source, rather than receiving a repaired branch.",
        publication="Same-source fresh confirmation on all five cases, unchanged-predictor reuse, "
            "and the two measured post-parent-failure repair paths. Initial recovery is established "
            "by actual predictive confirmation in the additional opaque practice case.",
        transfer="The previous forty-four tasks are now regression/development controls. Twelve "
            "new parameter realizations are fixed before this author phase. Compare the final source "
            "against the earlier fully successful refit-both method, not just the regressed source. "
            "Both use the same original priors, operators, seeds and per-task limits.")
    return plan


def run_experiment(output, *, provider_factory, auth_mode=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise ValueError("use a new output directory")
    plan = protocol()
    plan["counterexample"] = str((output / "counterexample/episode_00/training.json").resolve())
    write_json(output / "protocol.json", plan)
    for name in ("scripts/run_temporal_selective_preservation.py",
                 "src/hegel_machine/cognition/checkpoint_storage.py",
                 "src/hegel_machine/cognition/temporal_selective_practice.py"):
        path = output / "source_at_run" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path)
    challenge = deepcopy(plan)
    challenge["frozen_cases"] = [{"case_id": "episode_00", "prior": "first",
        "change": "initial_refutation", "unchanged": False, "task": plan["practice_cases"]["case_e"]}]
    replay = run_frozen(plan["seed_method"], output / "counterexample", challenge)
    if replay["confirmed_and_correct"]:
        result = {"status": "regression_not_reproduced", "counterexample": replay}
    else:
        result = run_study(output / "study", provider_factory=provider_factory,
            auth_mode=auth_mode, study_protocol=plan, learner_class=SelectiveMethodLearner)
        result["preceding_selective_phase"] = plan["preceding_selective_phase"]
        result["new_counterexample_costs"] = replay["costs"]
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
