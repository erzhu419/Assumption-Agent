#!/usr/bin/env python3
"""Compare availability of a published paired method under the same live LLM."""

import argparse
from concurrent.futures import ThreadPoolExecutor
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
from hegel_machine.cognition.temporal_paired_method import ProcedureConsumerAgent
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json
from scripts.run_temporal_method_learning import CONDITIONS, add_costs
from scripts.run_temporal_operator_llm import laboratory
from scripts.run_temporal_paired_method import (
    DONOR, SECOND_DONOR, donor_memory, protocol as paired_protocol,
)


def protocol():
    cases = []
    for index, original in enumerate(paired_protocol()["frozen_cases"]):
        case = deepcopy(original)
        case["case_id"] = case.pop("id")
        seed = case["task"]["apparatus_seed"]
        case["task"]["confirmation_seed"] = seed + 200003 + 10007 * index
        case["task"]["evaluation_seed"] = seed + 2000003 + 100003 * index
        cases.append(case)
    return {"study": "llm_temporal_paired_method_retention", "model": "gpt-6-luna",
        "profile": "development", "cases": cases, "conditions": list(CONDITIONS),
        "donors": {"first": str(DONOR), "second": str(SECOND_DONOR)},
        "provider": {"backend": "codex_cli", "reasoning_effort": "medium", "timeout_seconds": 180},
        "transfer_limits": asdict(InquiryLLMConfig(observation_budget=240, confirmation_rows=48,
            compute_budget=100000, execution_seconds=120, max_model_calls=24, token_budget=600000)),
        "evaluation": {"length": 256, "mse_threshold": 1e-10},
        "intervention": "Each pair receives the same original predictive files, instrument, old plans, "
            "research_roles and published verbal lesson. Only the published method source and its manifest "
            "entry are removed. Practice-generated calibrations, models and observations are not transferred.",
        "consumer": "Both conditions use ProcedureConsumerAgent and the same single-invocation query, "
            "ordinary actions and paid shared operators. The LLM chooses whether to use, revise or reconstruct "
            "a procedure, acquire evidence, fit or submit a model. No initial refit or research strategy is supplied.",
        "transfer": "Eight independent agents start from frozen memory with no current paid data. Condition "
            "submission order alternates by case; no results or newly learned files flow between cases.",
        "terminal_order": "Generate terminal labels only after all eight transfer agents stop and every "
            "training.json is saved. The paired conditions use identical fresh and terminal seeds.",
        "cost_scope": "Report transfer costs, the published method's authoring/practice costs and explicitly "
            "supplied earlier learning-result costs separately. Add their common learning sum once per "
            "condition, not once per case. Earlier attempts are development continuations, not independent "
            "from-scratch replications. Historical donor and seed learning are reused and outside this "
            "new-work subtotal unless explicitly supplied as learning results.",
        "limitations": "Four mechanism cases from two previously studied apparatus calibrations, with fresh "
            "confirmation and terminal trajectories. This tests additional executable-method availability "
            "over shared predictive knowledge and a shared lesson; the deleted condition may reconstruct it. "
            "It is not a population estimate or a comparison with an agent lacking all prior knowledge."}


def _case_result(chain, case, original, method_path, threshold):
    task, score = chain["tasks"][0], chain["evaluation"][0]
    confirmation = task.get("fresh_confirmation") or {}
    correct = score.get("mse") is not None and score["mse"] <= threshold
    confirmed = bool(confirmation.get("passed"))
    prior = original["research_roles"]["predictor"]
    prior_file = Path(task["workspace"]) / prior
    selected_prior = (task.get("model") == prior and prior_file.is_file()
                      and prior_file.read_text(encoding="utf-8") == original["files"][prior])
    # A parsed program proposal establishes that select_actions actually ran.
    # An ordinary run of a file defining a function is not counted as an invocation.
    invocations = [trial for trial in task.get("method_results", [])
                   if trial.get("path") == method_path and trial.get("proposal") is not None]
    unchanged = [trial for trial in invocations
                 if trial["feedback"].get("inherited_source_unchanged")]
    prior_confirmations = []
    for event in task.get("events", []):
        feedback = event.get("feedback", {})
        fresh = feedback.get("fresh_confirmation")
        if not fresh:
            continue
        submitted = (feedback.get("method_submission") if event.get("action") == "query"
                     else event.get("arguments", {}).get("model") if event.get("action") == "finish" else None)
        unchanged_model = feedback.get("submitted_model_unchanged") if event.get("action") == "query" \
            else feedback.get("inherited_source_unchanged")
        if submitted == prior and unchanged_model:
            prior_confirmations.append({"step": event["step"], "passed": fresh["passed"], "mse": fresh.get("mse")})
    return {"case_id": case["case_id"], "condition": chain["condition"], "prior": case["prior"],
        "unchanged": case["unchanged"], "status": task["status"], "model": task.get("model"),
        "confirmed": confirmed, "confirmed_and_correct": confirmed and correct,
        "false_confirmation": confirmed and not correct, "fresh_mse": confirmation.get("mse"),
        "terminal_mse": score.get("mse"), "costs": task["costs"],
        "selected_prior_predictor": selected_prior, "prior_predictor_confirmations": prior_confirmations,
        "method_program_calls": len(invocations), "unchanged_inherited_method_calls": len(unchanged),
        "method_continuations": sum(bool(trial["feedback"].get("after_observation_results")) for trial in invocations),
        "method_feedback_errors": sum(bool(trial["feedback"].get("error") or
                                           trial["feedback"].get("after_observation_error")) for trial in invocations),
        "inherited_method_modified": method_path in task.get("inherited_files_changed", [])}


def run_study(learned_result, output, *, provider_factory, auth_mode=None, workers=2,
              prior_learning_results=(), study_protocol=None):
    output, learned_result = Path(output), Path(learned_result)
    prior_learning_results = [Path(path) for path in prior_learning_results]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise ValueError("use a new output directory; preserve earlier results")
    plan = deepcopy(study_protocol) if study_protocol is not None else protocol()
    plan.update(learned_result=str(learned_result), workers=workers,
                prior_learning_results=[str(path) for path in prior_learning_results],
                lesson_source="learning/result.json:outgoing_knowledge.lessons[-1]")
    if auth_mode:
        plan["provider"]["auth_mode"] = auth_mode
    write_json(output / "protocol.json", plan)
    for name in ("scripts/run_temporal_paired_retention.py", "scripts/run_temporal_paired_method.py",
                 "src/hegel_machine/cognition/checkpoint_storage.py",
                 "src/hegel_machine/cognition/temporal_paired_method.py",
                 "src/hegel_machine/cognition/temporal_method_learning.py",
                 "src/hegel_machine/cognition/temporal_operator_agent.py",
                 "src/hegel_machine/cognition/online_inquiry_llm.py"):
        destination = output / "source_at_run" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    learned = json.loads(learned_result.read_text(encoding="utf-8"))
    learning_components = []
    for path in prior_learning_results:
        prior = json.loads(path.read_text(encoding="utf-8"))
        learning_components.append({"path": str(path), "status": prior["status"], "costs": prior["costs"]})
    learning_components.append({"path": str(learned_result), "status": learned["status"], "costs": learned["costs"]})
    common_costs = add_costs(component["costs"] for component in learning_components)
    summary = {"study": plan["study"], "model": plan["model"], "learning_status": learned["status"],
               "learning_components": learning_components, "common_learning_costs": common_costs,
               "conditions": {}, "cases": []}
    if learned["status"] != "completed":
        summary["status"] = "method_not_published"
        write_json(output / "summary.json", summary)
        return summary
    # Paired publication already writes this final lesson into outgoing memory.
    # Earlier saved results omit the redundant top-level lesson field.
    learned["lesson"] = learned["outgoing_knowledge"]["lessons"][-1]
    originals = {"first": donor_memory(DONOR, 1, "shared/meter.json"),
                 "second": donor_memory(SECOND_DONOR, 0, "shared/current_instrument.json")}
    memories = {prior: dict(zip(CONDITIONS, method_memory_pair(memory, learned)))
                for prior, memory in originals.items()}
    for prior, variants in memories.items():
        for condition, memory in variants.items():
            path = output / "memories" / prior
            path.mkdir(parents=True, exist_ok=True)
            write_json(path / f"{condition}.json", memory)
    config = InquiryLLMConfig(**plan["transfer_limits"])

    def train(job):
        case, condition = job
        path = output / case["case_id"] / condition
        path.mkdir(parents=True, exist_ok=True)
        agent = ProcedureConsumerAgent(provider_factory(path), path, arm="hegel", config=config,
            knowledge=deepcopy(memories[case["prior"]][condition]), source_task=False,
            confirmation_seed=case["task"]["confirmation_seed"])
        result = agent.run(laboratory(case["task"]),
                          on_checkpoint=lambda current: write_json(path / "active.json", current))
        result["task_position"] = 0
        chain = {"case_id": case["case_id"], "condition": condition, "tasks": [result], "evaluation": None}
        write_json(path / "training.json", chain)
        print(json.dumps({"case": case["case_id"], "condition": condition, "status": result["status"]}), flush=True)
        return chain

    jobs = [(case, condition) for index, case in enumerate(plan["cases"])
            for condition in (CONDITIONS if index % 2 == 0 else CONDITIONS[::-1])]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        chains = list(executor.map(train, jobs))
    # No terminal evaluation enters a still-running agent's context.
    for chain in chains:
        case = next(case for case in plan["cases"] if case["case_id"] == chain["case_id"])
        evaluate_chain(chain, [case["task"]], lab_factory=laboratory,
                       length=plan["evaluation"]["length"], timeout_seconds=config.execution_seconds)
        write_json(output / chain["case_id"] / chain["condition"] / "result.json", chain)
        summary["cases"].append(_case_result(chain, case, originals[case["prior"]],
            learned["method_path"], plan["evaluation"]["mse_threshold"]))
    events = [event for chain in chains for event in chain["tasks"][0]["events"]]
    summary.update(status="completed", method_path=learned["method_path"],
        evidence_kind="real_external_model" if events and all(event.get("usage", {}).get("backend") == "codex_cli"
            for event in events) else "interface_test_not_llm_performance")
    for condition in CONDITIONS:
        selected = [case for case in summary["cases"] if case["condition"] == condition]
        costs = add_costs(case["costs"] for case in selected)
        summary["conditions"][condition] = {
            "registered_tasks": len(plan["cases"]), "attempted_tasks": len(selected),
            "statuses": [case["status"] for case in selected],
            "confirmed_and_correct": sum(case["confirmed_and_correct"] for case in selected),
            "false_confirmations": sum(case["false_confirmation"] for case in selected),
            "selected_prior_predictor_tasks": sum(case["selected_prior_predictor"] for case in selected),
            "method_program_calls": sum(case["method_program_calls"] for case in selected),
            "unchanged_inherited_method_calls": sum(case["unchanged_inherited_method_calls"] for case in selected),
            "transfer_costs": costs,
            "transfer_plus_paired_learning_costs": add_costs([costs, common_costs])}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned-result", type=Path, required=True)
    parser.add_argument("--prior-learning-result", type=Path, action="append", default=[],
                        help="Earlier development learning result whose costs are shared; repeat for each attempt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    auth = verify_codex_subscription()
    print(json.dumps(run_study(args.learned_result, args.output, workers=args.workers, auth_mode=auth,
        prior_learning_results=args.prior_learning_result,
        provider_factory=lambda path: CodexCodeProvider(path, model="gpt-6-luna",
            reasoning_effort="medium", timeout_seconds=180)), indent=2))


if __name__ == "__main__":
    main()
