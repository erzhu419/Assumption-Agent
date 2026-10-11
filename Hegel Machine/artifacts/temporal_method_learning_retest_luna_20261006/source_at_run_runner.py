#!/usr/bin/env python3
"""Learn a procedure from paid experience, then intervene on its availability."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig, evaluate_chain
from hegel_machine.cognition.temporal_method_learning import (
    MethodConsolidationAgent, load_training_experience, method_memory_pair,
)
from hegel_machine.cognition.temporal_operator_agent import TemporalOperatorAgent
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json
from scripts.run_temporal_operator_llm import laboratory


CONDITIONS = ("method_retained", "method_deleted")
COSTS = ("observation_cost", "measurement_cost", "compute_cost",
         "python_execution_seconds", "model_calls", "reported_tokens")


def add_costs(items):
    items = list(items)
    return {key: None if any(item.get(key) is None for item in items)
            else sum(item[key] for item in items) for key in COSTS}


def protocol(seed=99106007):
    config = InquiryLLMConfig(observation_budget=240, confirmation_rows=48,
        max_model_calls=24, token_budget=600000, execution_seconds=120., compute_budget=100000)
    consolidation = InquiryLLMConfig(max_model_calls=10, token_budget=200000,
                                    execution_seconds=60.)
    cases = [{"case_id": f"changed_{mode}", "task": {
        "task_index": 2, "seed": seed, "apparatus_seed": seed,
        "instrument_mode": mode, "confirmation_seed": seed + 7919,
        "evaluation_seed": seed + 1000003}}
        for mode in ("finite2", "recurrent")]
    return {"study": "llm_consolidated_temporal_method_intervention", "model": "gpt-6-luna",
        "profile": "development", "cases": cases, "conditions": list(CONDITIONS),
        "transfer_limits": asdict(config), "consolidation_limits": asdict(consolidation),
        "provider": {"backend": "codex_cli", "reasoning_effort": "medium", "timeout_seconds": 180},
        "evaluation": {"length": 256, "mse_threshold": 1e-10},
        "intervention": "Keep identical predictive files, instrument states, old acquisition plan and textual "
            "lessons; remove only the newly consolidated program and its manifest entry.",
        "learning": "Retrospective consolidation of a selected successful development experience. "
            "The LLM writes the procedure from paid actions and outcomes; no terminal results enter its context.",
        "publication": "Call the authored procedure on the actual public capabilities and empty new-task "
            "observations; require a nonempty initial action list. This check buys no labels and tests no predictor.",
        "transfer": "Freeze once before four independent prospective transfers. The LLM chooses whether to "
            "use or adapt a method. No transfer feedback enters another task.",
        "terminal_order": "Generate terminal labels only after all four transfer agents stop learning.",
        "cost_scope": "Record prior supplied learning, consolidation and transfer separately; include the "
            "same common learning cost once in each condition's complete sequence.",
        "limitations": "One learned procedure and one paired new apparatus. The removed method may be "
            "reconstructed from the shared textual lesson; this tests additional executable-method availability."}


def run_method_study(experience, output, *, provider_factory, seed=99106007, workers=2,
                     auth_mode=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise ValueError("use a new output directory; preserve earlier results")
    plan = protocol(seed)
    if auth_mode:
        plan["provider"]["auth_mode"] = auth_mode
    plan["training_experience"] = str(experience) if isinstance(experience, (str, Path)) else "provided_training_record"
    write_json(output / "protocol.json", plan)
    raw = json.loads(Path(experience).read_text()) if isinstance(experience, (str, Path)) else experience
    paid = load_training_experience(raw)
    prior_costs = add_costs(task["costs"] for task in raw.get("tasks", [raw]))
    learner_path = output / "consolidation"
    learner = MethodConsolidationAgent(provider_factory(learner_path), learner_path,
        experience=paid, public_spec=laboratory(plan["cases"][0]["task"]).public_spec(),
        config=InquiryLLMConfig(**plan["consolidation_limits"]))
    learned = learner.run(on_checkpoint=lambda result: write_json(learner_path / "active.json", result))
    write_json(learner_path / "result.json", learned)
    common = {"prior": prior_costs, "consolidation": learned["costs"],
              "combined": add_costs([prior_costs, learned["costs"]])}
    if learned["status"] != "completed":
        summary = {"study": plan["study"], "status": "method_not_published", "common_learning_costs": common,
                   "consolidation_status": learned["status"], "conditions": {}}
        write_json(output / "summary.json", summary)
        return summary
    memories = dict(zip(CONDITIONS, method_memory_pair(paid["outgoing_knowledge"], learned)))
    for condition, memory in memories.items():
        write_json(output / f"{condition}_memory.json", memory)
    config = InquiryLLMConfig(**plan["transfer_limits"])

    def train(job):
        case, condition = job
        path = output / case["case_id"] / condition
        agent = TemporalOperatorAgent(provider_factory(path), path, arm="hegel", config=config,
            knowledge=deepcopy(memories[condition]), source_task=False,
            confirmation_seed=case["task"]["confirmation_seed"])
        result = agent.run(laboratory(case["task"]),
                          on_checkpoint=lambda data: write_json(path / "active.json", data))
        result["task_position"] = 0
        chain = {"case_id": case["case_id"], "condition": condition, "tasks": [result], "evaluation": None}
        write_json(path / "training.json", chain)
        print(json.dumps({"case": case["case_id"], "condition": condition, "status": result["status"]}), flush=True)
        return chain

    jobs = [(case, condition) for index, case in enumerate(plan["cases"])
            for condition in (CONDITIONS if index % 2 == 0 else CONDITIONS[::-1])]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        chains = list(executor.map(train, jobs))
    for chain in chains:
        case = next(case for case in plan["cases"] if case["case_id"] == chain["case_id"])
        evaluate_chain(chain, [case["task"]], lab_factory=laboratory,
                       length=plan["evaluation"]["length"], timeout_seconds=config.execution_seconds)
        write_json(output / chain["case_id"] / chain["condition"] / "result.json", chain)
    all_events = learned["events"] + [event for chain in chains for event in chain["tasks"][0]["events"]]
    summary = {"study": plan["study"], "status": "completed", "model": plan["model"],
        "evidence_kind": "real_external_model" if all(event.get("usage", {}).get("backend") == "codex_cli"
            for event in all_events) else "interface_test_not_llm_performance",
        "method_path": learned["method_path"], "common_learning_costs": common, "conditions": {}}
    for condition in CONDITIONS:
        selected = [chain for chain in chains if chain["condition"] == condition]
        tasks = [chain["tasks"][0] for chain in selected]
        correct = [chain["evaluation"][0].get("mse") is not None and
                   chain["evaluation"][0]["mse"] <= plan["evaluation"]["mse_threshold"] for chain in selected]
        confirmed = [bool((task.get("fresh_confirmation") or {}).get("passed")) for task in tasks]
        calls = [event for task in tasks for event in task["events"] if event["action"] == "query"
                 and event["feedback"].get("executed_path") == learned["method_path"]]
        costs = add_costs(task["costs"] for task in tasks)
        summary["conditions"][condition] = {
            "registered_tasks": len(plan["cases"]), "attempted_tasks": len(tasks),
            "confirmed_and_correct": sum(a and b for a, b in zip(correct, confirmed)),
            "false_confirmations": sum(b and not a for a, b in zip(correct, confirmed)),
            "statuses": [task["status"] for task in tasks], "method_query_calls": len(calls),
            "successful_method_continuations": sum(bool(event["feedback"].get("after_observation_results"))
                and not event["feedback"].get("error") and not event["feedback"].get("after_observation_error")
                for event in calls),
            "revised_inherited_method_tasks": sum(learned["method_path"] in task["inherited_files_changed"] for task in tasks),
            "transfer_costs": costs, "costs_including_common_learning": add_costs([costs, common["combined"]])}
    usage = [event.get("usage", {}) for event in all_events]
    summary["reported_tokens_new_work"] = sum(item.get("total_tokens") or 0 for item in usage)
    summary["calls_without_token_usage"] = sum(not isinstance(item.get("total_tokens"), int) for item in usage)
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experience", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=99106007)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    auth_mode = verify_codex_subscription()
    def provider(path):
        return CodexCodeProvider(path, model="gpt-6-luna", reasoning_effort="medium", timeout_seconds=180)
    print(json.dumps(run_method_study(args.experience, args.output, provider_factory=provider,
        seed=args.seed, workers=args.workers, auth_mode=auth_mode), indent=2))


if __name__ == "__main__":
    main()
