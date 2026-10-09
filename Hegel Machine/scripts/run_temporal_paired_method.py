#!/usr/bin/env python3
"""Learn from unchanged/changed prior support, then execute one frozen program."""

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
from hegel_machine.cognition.temporal_paired_practice import PairedMethodLearner
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json
from scripts.run_temporal_method_learning import add_costs
from scripts.run_temporal_operator_llm import laboratory


DONOR = ROOT / "artifacts/temporal_evidence_continuation_luna_20261006/temporal_operators_dev_01/hegel/training.json"
SEED = ROOT / "artifacts/temporal_method_dependency_extended_luna_20261006/practice/result.json"
SECOND_DONOR = ROOT / "artifacts/temporal_method_dependency_extended_luna_20261006/changed_recurrent/method_deleted/training.json"


def donor_memory(path, position, instrument):
    task = json.loads(Path(path).read_text())["tasks"][position]
    if task["status"] != "completed" or not task["fresh_confirmation"]["passed"]:
        raise ValueError("paired practice requires an actually confirmed prior")
    memory = deepcopy(task["outgoing_knowledge"])
    memory["research_roles"] = {"instrument": instrument, "predictor": task["model"]}
    if any(path not in memory["files"] for path in memory["research_roles"].values()):
        raise ValueError("the declared prior instrument and predictor must exist")
    return memory


def task(seed, stage, confirmation_offset):
    return {"seed": seed, "apparatus_seed": seed, "task_index": stage,
            "instrument_mode": "recurrent", "confirmation_seed": seed + confirmation_offset}


def protocol():
    return {"study": "llm_temporal_paired_support_method", "model": "gpt-6-luna",
        "reasoning_effort": "medium", "donor": str(DONOR), "seed_method": str(SEED),
        "second_donor": str(SECOND_DONOR),
        "author_limits": asdict(InquiryLLMConfig(observation_budget=960, confirmation_rows=0,
            max_model_calls=20, token_budget=1000000, compute_budget=300000, execution_seconds=180)),
        "episode_limits": asdict(InquiryLLMConfig(observation_budget=240, confirmation_rows=48,
            max_model_calls=0, token_budget=0, compute_budget=100000, execution_seconds=60)),
        "max_program_steps": 8,
        "practice_cases": {"case_a": task(98106007, 2, 38191), "case_b": task(98106007, 1, 62861)},
        "frozen_cases": [
            {"id": "episode_00", "prior": "second", "unchanged": True, "task": task(99406007, 2, 75391)},
            {"id": "episode_01", "prior": "first", "unchanged": False, "task": task(98106007, 2, 97613)},
            {"id": "episode_02", "prior": "second", "unchanged": False, "task": task(99406007, 1, 87131)},
            {"id": "episode_03", "prior": "first", "unchanged": True, "task": task(98106007, 1, 94541)}],
        "learning": "Each paid practice invocation starts with the same frozen earlier knowledge and no "
            "current observations. The LLM authors and revises one select_actions procedure, chooses which "
            "unlabeled case to practice, and observes its real execution. No automatic initial refit or "
            "host-written experimental policy. Prior role paths are declared consistently in memory.",
        "publication": "Both cases must freshly confirm with exactly the final source, with actual unchanged "
            "prior-predictor reuse in at least one case and a confirmed revision in another. The learner is "
            "not told which case supports which choice. Source and numerical traces establish how its program "
            "makes those decisions. This is explicitly prompted reuse-and-revision learning; merely refitting "
            "both cases is insufficient for that goal.",
        "frozen_execution": "Freeze only source and lesson. Execute unchanged on original and second "
            "calibrations in four empty workspaces. The program selects all experiments, operators and final "
            "predictor, with no LLM making intermediate choices. Fresh confirmation costs48 observations. "
            "Only after all four stop is terminal256-row evaluation performed.",
        "cost_scope": "Report new authoring/practice and frozen executions separately. Existing donor "
            "and seed learning are reused historical work, not new calls. Terminal evaluation is separate.",
        "limitations": "One learning apparatus and one other previously studied apparatus, with fixed "
            "numerical operator families and supplied instrument/predictor roles. Second calibration and "
            "outcomes are unavailable during authoring. This is a mechanism study, not a population comparison "
            "or evidence of new primitive/role invention."}


def run_study(output, *, provider_factory, auth_mode=None, seed_method=None, seed_source=None,
              resume_practice=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise ValueError("use a new output directory")
    if resume_practice is not None:
        if seed_method is not None or seed_source is not None:
            raise ValueError("resumed practice restores its recorded inputs; do not supply a different seed")
        resume_path = Path(resume_practice)
        plan = json.loads((resume_path.parent.parent / "protocol.json").read_text())
        previous = json.loads(resume_path.read_text())
        seed_method = plan["seed_method"]
        if "seed_source" in plan:
            seed_source = resume_path.parent.parent / "source_at_run" / "seed_candidate.py"
        plan.update(resume_practice=str(resume_path),
                    resumption="Restore the exact saved candidate and paid practice trials after provider "
                               "failure. Begin a new authoring phase with separate costs; preserve original "
                               "predictive inputs. Earlier trial coverage remains bound to its source.")
    else:
        plan = protocol()
    seed_path = Path(seed_method) if seed_method is not None else SEED
    plan["seed_method"] = str(seed_path)
    if seed_source is not None:
        plan["seed_source"] = str(seed_source)
        plan["seed_status"] = ("unpublished_development_candidate; restored paid trial coverage"
                               if resume_practice is not None
                               else "unpublished_development_candidate; new empty practice state")
    plan["auth_mode"] = auth_mode
    write_json(output / "protocol.json", plan)
    for name in ("scripts/run_temporal_paired_method.py",
                 "src/hegel_machine/cognition/temporal_paired_practice.py",
                 "src/hegel_machine/cognition/temporal_paired_method.py",
                 "src/hegel_machine/cognition/temporal_operator_agent.py",
                 "src/hegel_machine/cognition/online_inquiry_llm.py"):
        destination = output / "source_at_run" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    first = donor_memory(Path(plan["donor"]), 1, "shared/meter.json")
    seed = json.loads(seed_path.read_text())
    initial, _ = method_memory_pair(first, seed)
    if seed_source is not None:
        source = Path(seed_source).read_text()
        initial["files"][seed["method_path"]] = source
        (output / "source_at_run" / "seed_candidate.py").write_text(source)
    author_path = output / "learning"
    learner = PairedMethodLearner(provider_factory(author_path), author_path, knowledge=initial,
        cases=plan["practice_cases"], lab_factory=laboratory,
        config=InquiryLLMConfig(**plan["author_limits"]))
    if resume_practice is not None:
        write_json(output / "restored_practice.json", learner.restore_practice(previous))
    learned = learner.run(on_checkpoint=lambda result: write_json(author_path / "active.json", result))
    write_json(author_path / "result.json", learned)
    summary = {"study": plan["study"], "model": plan["model"], "learning_status": learned["status"],
               "new_learning_costs": learned["costs"], "frozen_cases": []}
    if learned["status"] != "completed":
        summary["status"] = "method_not_published"
        write_json(output / "summary.json", summary)
        return summary
    # Load the second calibration only after publication. Neither its numeric
    # files nor its trial outcomes entered the authoring workspace.
    second = donor_memory(SECOND_DONOR, 0, "shared/current_instrument.json")
    memories = {name: method_memory_pair(memory, learned)[0]
                for name, memory in (("first", first), ("second", second))}
    chains = []
    config = InquiryLLMConfig(**plan["episode_limits"])
    for case in plan["frozen_cases"]:
        path = output / "frozen" / case["id"]
        agent = FrozenMethodAgent(path, knowledge=memories[case["prior"]],
            method_source=learned["method_source"], config=config,
            confirmation_seed=case["task"]["confirmation_seed"])
        result = agent.execute_method(laboratory(case["task"]), max_steps=plan["max_program_steps"])
        result["task_position"] = 0
        chain = {"tasks": [result], "evaluation": None}
        write_json(path / "training.json", chain)
        chains.append((case, chain))
    for case, chain in chains:
        evaluate_chain(chain, [case["task"]], lab_factory=laboratory, length=256,
                       timeout_seconds=config.execution_seconds)
        result, score = chain["tasks"][0], chain["evaluation"][0]
        write_json(output / "frozen" / case["id"] / "result.json", chain)
        operators = [outcome.get("operator") for event in result["events"]
                     for outcome in event.get("feedback", {}).get("after_observation_results", [])]
        summary["frozen_cases"].append({"id": case["id"], "prior": case["prior"],
            "unchanged": case["unchanged"], "status": result["status"], "model": result["model"],
            "fresh_mse": (result.get("fresh_confirmation") or {}).get("mse"),
            "terminal_mse": score.get("mse"), "costs": result["costs"], "operators": operators,
            "confirmed_and_correct": bool((result.get("fresh_confirmation") or {}).get("passed"))
                and score.get("mse") is not None and score["mse"] <= 1e-10})
    summary.update(status="completed", frozen_costs=add_costs(c[1]["tasks"][0]["costs"] for c in chains))
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-method", type=Path)
    parser.add_argument("--seed-source", type=Path,
                        help="Continue development from an unpublished candidate, with new paid practice")
    parser.add_argument("--resume-practice", type=Path,
                        help="Restore paid experimental state from a provider-failed learning result")
    args = parser.parse_args()
    auth = verify_codex_subscription()
    print(json.dumps(run_study(args.output, auth_mode=auth, seed_method=args.seed_method,
        seed_source=args.seed_source, resume_practice=args.resume_practice, provider_factory=lambda path:
        CodexCodeProvider(path, model="gpt-6-luna", reasoning_effort="medium", timeout_seconds=180)), indent=2))


if __name__ == "__main__":
    main()
