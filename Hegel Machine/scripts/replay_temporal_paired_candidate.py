#!/usr/bin/env python3
"""Evaluate an interrupted learner's recorded candidate after a runtime repair."""

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig, evaluate_chain
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from hegel_machine.cognition.temporal_paired_method import FrozenMethodAgent
from scripts.run_online_inquiry_llm import write_json
from scripts.run_temporal_method_learning import add_costs
from scripts.run_temporal_operator_llm import laboratory
from scripts.run_temporal_paired_method import DONOR, SECOND_DONOR, donor_memory, protocol


def replay(candidate_run, output):
    candidate_run, output = Path(candidate_run), Path(output)
    source = (candidate_run / "learning" / METHOD_PATH).read_text()
    author = json.loads((candidate_run / "learning/active.json").read_text())
    writes = [event for event in author["events"] if event["action"] == "write"
              and event["arguments"].get("path") == METHOD_PATH
              and event["arguments"].get("code") == source]
    if not writes:
        raise ValueError("candidate must match a recorded author write")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise ValueError("use a new output directory")
    plan = protocol()
    for key in ("author_limits", "practice_cases", "learning", "publication", "seed_method"):
        plan.pop(key)
    plan.update(study="frozen_paired_candidate_after_runtime_repair", candidate_run=str(candidate_run),
        candidate_write_step=writes[-1]["step"],
        evidence_scope="Frozen replay of an actual LLM-authored candidate from an interrupted development "
            "run. No authoring state, trial observations or error feedback is restored. This is not a "
            "completed online learning/publication run. Four fresh workspaces start with their supplied "
            "prior calibration and run exactly this source; no LLM calls or host-written strategy.",
        runtime_repair="An initial model declaration now skips an absent paid-data diagnostic and goes "
            "directly to actual charged48-row fresh confirmation. Failed fresh predictions supply real "
            "counterexamples; an invocation error is not a predictive refutation.")
    write_json(output / "protocol.json", plan)
    (output / "candidate.py").write_text(source)
    for name in ("scripts/replay_temporal_paired_candidate.py", "scripts/run_temporal_paired_method.py",
                 "src/hegel_machine/cognition/checkpoint_storage.py",
                 "src/hegel_machine/cognition/temporal_paired_method.py",
                 "src/hegel_machine/cognition/temporal_operator_agent.py",
                 "src/hegel_machine/cognition/online_inquiry_llm.py"):
        destination = output / "source_at_run" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    memories = {"first": donor_memory(DONOR, 1, "shared/meter.json"),
                "second": donor_memory(SECOND_DONOR, 0, "shared/current_instrument.json")}
    config = InquiryLLMConfig(**plan["episode_limits"])
    chains = []
    for case in plan["frozen_cases"]:
        path = output / "frozen" / case["id"]
        agent = FrozenMethodAgent(path, knowledge=memories[case["prior"]], method_source=source,
                                  config=config, confirmation_seed=case["task"]["confirmation_seed"])
        result = agent.execute_method(laboratory(case["task"]), max_steps=plan["max_program_steps"])
        result["task_position"] = 0
        chain = {"tasks": [result], "evaluation": None}
        write_json(path / "training.json", chain)
        chains.append((case, chain))
    cases = []
    for case, chain in chains:
        evaluate_chain(chain, [case["task"]], lab_factory=laboratory, length=256,
                       timeout_seconds=config.execution_seconds)
        result, score = chain["tasks"][0], chain["evaluation"][0]
        write_json(output / "frozen" / case["id"] / "result.json", chain)
        prior = memories[case["prior"]]["research_roles"]["predictor"]
        outcomes = [event["feedback"] for event in result["events"]]
        cases.append({"id": case["id"], "prior": case["prior"], "unchanged": case["unchanged"],
            "status": result["status"], "model": result["model"],
            "selected_prior_predictor": result["model"] == prior
                and (Path(result["workspace"]) / prior).read_text() == memories[case["prior"]]["files"][prior],
            "confirmations": [{key: feedback["fresh_confirmation"].get(key) for key in ("mse", "passed", "rows")}
                              for feedback in outcomes if "fresh_confirmation" in feedback],
            "operators": [op["operator"] for feedback in outcomes
                          for op in feedback.get("after_observation_results", [])],
            "terminal_mse": score.get("mse"), "costs": result["costs"],
            "confirmed_and_correct": bool((result.get("fresh_confirmation") or {}).get("passed"))
                and score.get("mse") is not None and score["mse"] <= 1e-10})
    summary = {"study": plan["study"], "status": "completed", "candidate_write_step": writes[-1]["step"],
               "evidence_kind": "frozen_llm_authored_program_replay", "online_learning_completed": False,
               "frozen_cases": cases, "frozen_costs": add_costs(c[1]["tasks"][0]["costs"] for c in chains)}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.candidate_run, args.output), indent=2))


if __name__ == "__main__":
    main()
