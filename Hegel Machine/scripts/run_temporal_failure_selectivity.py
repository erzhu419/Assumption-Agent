#!/usr/bin/env python3
"""Learn whether a parent counterexample requires instrument or response research."""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json
from scripts.run_temporal_calibration_transfer import CALIBRATIONS, run_frozen, snapshot
from scripts.run_temporal_counterexample_learning import protocol as earlier_protocol, run_study
from scripts.run_temporal_paired_method import task


SEED = ROOT / "artifacts/temporal_counterexample_learning_luna_20261006/learning/result.json"


def shifted_task(prior, change, next_seed, index):
    seed, stage = CALIBRATIONS[prior]
    settings = task(seed, stage, 600003 + 10007 * index)
    settings["evaluation_seed"] = seed + 6000003 + 100003 * index
    settings["calibration_change"] = {
        "after_rows": 72,
        "apparatus_seed": next_seed if change in ("instrument_shift", "joint_shift") else seed,
        "response_seed": next_seed if change in ("response_shift", "joint_shift") else seed}
    return settings


def protocol():
    previous = earlier_protocol()
    cases = []
    for prior in CALIBRATIONS:
        index = len(cases)
        cases.append({"case_id": f"episode_{index:02d}", "prior": prior,
            "change": "stable_checkpoint", "unchanged": True,
            "task": shifted_task(prior, "stable_checkpoint", 0, index)})
    for index in range(2):
        for prior in CALIBRATIONS:
            for change in ("response_shift", "instrument_shift", "joint_shift"):
                serial = len(cases)
                cases.append({"case_id": f"episode_{serial:02d}", "prior": prior,
                    "change": change, "unchanged": False,
                    "task": shifted_task(prior, change, 157106007 + 1009 * index, serial)})
    for case in previous["frozen_cases"]:
        cases.append({**deepcopy(case), "case_id": f"episode_{len(cases):02d}",
                      "change": "stationary_" + case["change"]})
    plan = deepcopy(previous)
    plan.update(study="llm_temporal_failure_selectivity", seed_method=str(SEED),
        practice_cases={name: shifted_task("first", change, 147106007, 100 + index)
            for index, (name, change) in enumerate(zip(
                ("case_a", "case_b", "case_c", "case_d"),
                ("stable_checkpoint", "response_shift", "instrument_shift", "joint_shift")))},
        author_limits=asdict(InquiryLLMConfig(observation_budget=2400, confirmation_rows=0,
            max_model_calls=24, token_budget=1200000, compute_budget=600000, execution_seconds=240)),
        frozen_cases=cases,
        learning="An actual failed execution on case_c supplies paid observations and feedback. "
            "Four opaque practice cases share the first donor and initial physics. The LLM must author "
            "its own research conditions and actions. No cause label or replacement policy is supplied.",
        publication="One exact source must freshly confirm all four cases with original-predictor reuse "
            "and revision. Fits may explicitly select paid record indices; history and its costs remain intact.",
        transfer="Fourteen checkpoint cases use two donor priors and new post-checkpoint calibration seeds; "
            "the eighteen previous stationary tasks are preserved as regression controls. Old and newly "
            "published methods receive the same original priors, seeds, operators and limits.",
        physical_intervention="The public 72-row checkpoint marks a possible recalibration between whole "
            "trajectories. Stable, response, instrument and joint changes have identical initial observations "
            "and the same public checkpoint. Which component changes and its parameters are hidden. "
            "Confirmation after the checkpoint and terminal evaluation use settled physics.",
        limitations="Bounded one-time calibration changes in supplied recurrent families. New measurements "
            "can establish instrument applicability, but refitting both components does not independently "
            "identify which response coefficients changed. The same parent failure may require more than "
            "response fitting. No noise, arbitrary drift or general causal-diagnosis claim is made.")
    return plan


def run_experiment(output, *, provider_factory, auth_mode=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise ValueError("use a new output directory")
    plan = protocol()
    plan["counterexample"] = str((output / "counterexample/episode_00/training.json").resolve())
    if auth_mode:
        plan["provider"]["auth_mode"] = auth_mode
    write_json(output / "protocol.json", plan)
    snapshot(output)
    destination = output / "source_at_run/scripts/run_temporal_failure_selectivity.py"
    shutil.copy2(Path(__file__), destination)
    challenge = deepcopy(plan)
    challenge["frozen_cases"] = [{"case_id": "episode_00", "prior": "first",
        "change": "instrument_shift", "unchanged": False, "task": plan["practice_cases"]["case_c"]}]
    counterexample = run_frozen(plan["seed_method"], output / "counterexample", challenge)
    if counterexample["confirmed_and_correct"]:
        result = {"status": "no_counterexample", "counterexample": counterexample}
    else:
        result = run_study(output / "study", provider_factory=provider_factory,
                           auth_mode=auth_mode, study_protocol=plan)
        result["new_counterexample_costs"] = counterexample["costs"]
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    auth = verify_codex_subscription()
    result = run_experiment(args.output, auth_mode=auth,
        provider_factory=lambda path: CodexCodeProvider(path, model="gpt-6-luna",
            reasoning_effort="medium", timeout_seconds=180))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
