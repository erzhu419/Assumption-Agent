#!/usr/bin/env python3
"""Freeze learned research and separate instrument from response calibration changes."""

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
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig, evaluate_chain
from hegel_machine.cognition.temporal_method_learning import method_memory_pair
from hegel_machine.cognition.temporal_paired_method import FrozenMethodAgent
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json
from scripts.run_temporal_method_learning import add_costs
from scripts.run_temporal_operator_llm import laboratory
from scripts.run_temporal_paired_method import DONOR, SECOND_DONOR, donor_memory
from scripts.run_temporal_paired_retention import (
    _case_result, protocol as retention_protocol, run_study as run_retention,
)


CALIBRATIONS = {"first": (98106007, 1), "second": (99406007, 2)}
SOURCE_FILES = (
    "scripts/run_temporal_calibration_transfer.py",
    "src/hegel_machine/cognition/stateful_instrument_lab.py",
    "src/hegel_machine/cognition/ambiguous_instrument_lab.py",
    "src/hegel_machine/cognition/temporal_inquiry.py",
    "src/hegel_machine/cognition/temporal_paired_method.py",
    "src/hegel_machine/cognition/temporal_operator_agent.py",
    "src/hegel_machine/cognition/online_inquiry_llm.py",
)


def protocol():
    cases = []

    def add(prior, instrument_seed, response_seed, change):
        old_seed, stage = CALIBRATIONS[prior]
        index = len(cases)
        task = {"seed": instrument_seed, "apparatus_seed": instrument_seed,
                "task_index": stage, "instrument_mode": "recurrent",
                "confirmation_seed": instrument_seed + 300003 + 10007 * index,
                "evaluation_seed": instrument_seed + 4000003 + 100003 * index}
        if response_seed is not None:
            task["response_seed"] = response_seed
        cases.append({"case_id": f"episode_{index:02d}", "prior": prior,
                      "unchanged": change == "stable", "change": change,
                      "instrument_changed": instrument_seed != old_seed,
                      "response_changed": change in ("response_only", "both"), "task": task})

    for prior, (seed, _) in CALIBRATIONS.items():
        add(prior, seed, None, "stable")
    for index, (prior, (seed, _)) in enumerate(CALIBRATIONS.items()):
        add(prior, seed, 117106007 + 1009 * index, "response_only")
    # Twelve new apparatus calibrations, each with old and new response laws.
    # These are finite-family parameter changes, not unseen structural families.
    for index in range(12):
        prior = ("first", "second")[index % 2]
        seed = 107106007 + 1009 * index
        add(prior, seed, CALIBRATIONS[prior][0], "instrument_only")
        add(prior, seed, None, "both")

    live = retention_protocol()
    live.update(study="llm_temporal_calibration_transfer", cases=deepcopy(
        [cases[index] for index in (0, 1, 2, 3, 4, 7)]),
        transfer="Twelve independent agents start from the original frozen donor memories and no current "
            "observations. No fitted target models, frozen-challenge traces or condition labels transfer. "
            "Both arms may use the same ordinary actions; inherited source changes are recorded.",
        terminal_order="All twelve live agents stop and save training before live terminal evaluation.",
        limitations="Six preselected development cases cover two stable priors, two independent response "
            "changes, one new instrument calibration with the old response law and one joint change. "
            "Both new instrument calibrations are from the frozen cohort fixed before any outcomes. "
            "New numerical parameters stay within the same recurrent family and supplied fit operators. "
            "The response law staying fixed does not keep a predictor using the old instrument correct.")
    return {"study": "temporal_independent_calibration_transfer",
            "frozen_cases": cases, "live_protocol": live,
            "frozen_limits": asdict(InquiryLLMConfig(observation_budget=240, confirmation_rows=48,
                max_model_calls=0, token_budget=0, compute_budget=100000, execution_seconds=120)),
            "max_program_steps": 8,
            "evaluation": {"length": 256, "mse_threshold": 1e-10},
            "question": "Does a fixed learned research procedure distinguish instrument validity from "
                "parent-prediction validity, and can the same LLM repair its use from current evidence?",
            "freeze": "Freeze the actually published source and original numerical priors before all cases. "
                "No researcher refits priors or edits the method. Strict frozen execution and live availability "
                "are separate cohorts. The live cohort is selected before frozen outcomes are observed.",
            "trajectories": "The fixed program retains its three authored acquisition trajectories. New "
                "confirmation and terminal seeds change only test trajectories. Live agents choose acquisitions.",
            "limitations": "Calibration transfer within a supplied recurrent family, not new structural primitives. "
                "Paired response interventions share instruments; case counts are not independent populations."}


def snapshot(output):
    for name in SOURCE_FILES:
        destination = output / "source_at_run" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)


def run_frozen(learned_result, output, plan):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    learned = json.loads(Path(learned_result).read_text())
    if learned["status"] != "completed":
        raise ValueError("calibration transfer requires a published method")
    learned["lesson"] = learned["outgoing_knowledge"]["lessons"][-1]
    originals = {"first": donor_memory(DONOR, 1, "shared/meter.json"),
                 "second": donor_memory(SECOND_DONOR, 0, "shared/current_instrument.json")}
    memories = {name: method_memory_pair(memory, learned)[0] for name, memory in originals.items()}
    config = InquiryLLMConfig(**plan["frozen_limits"])
    chains = []
    for case in plan["frozen_cases"]:
        path = output / case["case_id"]
        agent = FrozenMethodAgent(path, knowledge=memories[case["prior"]],
            method_source=learned["method_source"], config=config,
            confirmation_seed=case["task"]["confirmation_seed"])
        result = agent.execute_method(laboratory(case["task"]), max_steps=plan["max_program_steps"])
        result["task_position"] = 0
        chain = {"condition": "frozen_method", "tasks": [result], "evaluation": None}
        write_json(path / "training.json", chain)
        chains.append((case, chain))
        print(json.dumps({"cohort": "frozen", "case": case["case_id"], "status": result["status"]}), flush=True)
    results = []
    for case, chain in chains:
        evaluate_chain(chain, [case["task"]], lab_factory=laboratory,
            length=plan["evaluation"]["length"], timeout_seconds=config.execution_seconds)
        write_json(output / case["case_id"] / "result.json", chain)
        result = _case_result(chain, case, originals[case["prior"]], learned["method_path"],
                              plan["evaluation"]["mse_threshold"])
        result["change"] = case["change"]
        results.append(result)
    summary = {"study": plan["study"], "status": "completed", "cases": results,
        "confirmed_and_correct": sum(case["confirmed_and_correct"] for case in results),
        "false_confirmations": sum(case["false_confirmation"] for case in results),
        "costs": add_costs(case["costs"] for case in results)}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned-result", type=Path, required=True)
    parser.add_argument("--prior-learning-result", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "protocol.json").exists():
        raise ValueError("use a new output directory")
    plan = protocol()
    plan.update(learned_result=str(args.learned_result),
                prior_learning_results=[str(path) for path in args.prior_learning_result])
    auth = verify_codex_subscription()
    plan["live_protocol"]["provider"]["auth_mode"] = auth
    write_json(args.output / "protocol.json", plan)
    snapshot(args.output)
    frozen = run_frozen(args.learned_result, args.output / "frozen", plan)
    live = run_retention(args.learned_result, args.output / "live", auth_mode=auth,
        study_protocol=plan["live_protocol"], workers=args.workers,
        prior_learning_results=args.prior_learning_result,
        provider_factory=lambda path: CodexCodeProvider(path, model="gpt-6-luna",
            reasoning_effort="medium", timeout_seconds=180))
    summary = {"study": plan["study"], "status": "completed",
               "frozen": frozen, "live": live}
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
