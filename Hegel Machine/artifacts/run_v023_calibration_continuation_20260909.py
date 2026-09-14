"""Compare paid initial calibration and actual continuation on seven fixed risk tasks."""

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hegel_machine import __version__
from hegel_machine.cognition import controller, decision_research
from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.risk_search import trajectory_blocks
from hegel_machine.cognition.workspace import ResearchWorkspace
from run_method_continuation_study import _measure, _new_evaluation_rows

PARTITIONS = ("development", "confirmation", "forward", "continual", "stateful")
ARMS = ("nominal", "proposed_calibration", "stop", "short", "full_revision")


def retained(episode, reason, *, model=None, available=True):
    return {"program": deepcopy(episode["program"]), "fallback": episode["fallback"],
            "uncertainty_model": deepcopy(episode["uncertainty_model"] if model is None else model),
            "expression": episode["expression"], "stop_reason": reason,
            "available": available, "observation_cost": 0, "measurement_cost": 0, "compute_cost": 0,
            "actions": [], "research_state": deepcopy(episode["research_state"])}


def behavior(episode):
    actions = episode["actions"]
    counts = Counter(action["action"] for action in actions)
    keep = ("candidate_id", "adopted", "selected", "execution_complete", "gain", "normalized_gain",
            "before_loss", "after_loss", "reason", "status", "rows", "trajectories", "reference_rows",
            "support_margin", "standard_error", "program_changed", "evidence_status")
    events = []
    for action in actions:
        if "decision" in action["action"] or "calibration" in action["action"] or action["action"] == "continuation_feedback":
            details = action["details"]
            events.append({"round": action["round"], "action": action["action"],
                "observation_cost": action["observation_cost"], "compute_cost": action["compute_cost"],
                "details": {key: details[key] for key in keep if key in details},
                "trajectory_gains": [trial["gain"] for trial in details.get("trajectory_comparisons", [])]})
    frontier = (episode.get("research_state") or {}).get("decision_frontier", {})
    return {"action_counts": dict(counts), "decision_events": events,
            "pending_question": bool(frontier), "pending_candidates": max(0, len(frontier.get("policies", []))-1),
            "history_packages": sum(action["action"] == "plan_experiment" and action["details"].get("policy") == "decision_history_contrast" for action in actions),
            "short_targeted_packages": sum(action["action"] == "plan_decision_probe_package" and bool(action["details"].get("selected")) for action in actions),
            "extra_pairs": counts["extend_decision_question"] + counts["extend_decision_comparison"]}


def save(path, report):
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="v023_release_calibration_continuation")
    args = parser.parse_args()
    output = ROOT / f"artifacts/{args.label}_20260909.json"
    summary_path = output.with_name(output.stem + "_summary.json")
    if output.exists() or summary_path.exists():
        raise FileExistsError("Preserve previous experiments; use a new output label")
    config = ResearchConfig(seed=1237)
    sources = [(partition, index, environment) for partition in PARTITIONS
               for index, environment in enumerate(make_suite(config.seed, partition=partition))]
    roster = [(index, *source) for index, source in enumerate(sources) if source[2].spec.decision_costs is not None]
    if len(sources) != 65 or len(roster) != 7:
        raise ValueError("The fixed protocol contains 65 tasks, including seven public risk tasks")
    report = {"experiment": "paid_initial_calibration_and_actual_continuation", "version": __version__,
        "seed": 1237, "config": asdict(config), "partitions": list(PARTITIONS),
        "task_indices": [item[0] for item in roster], "status": "running", "tasks": [],
        "selection": "All seven public decision-cost tasks across the five seed1237 partitions, fixed before outcomes. Each task starts with an empty knowledge library.",
        "initial_capture": "Use the previous and candidate complete plans already saved by the production compare_initial_calibration action. No fitting or adoption logic is copied, and no production function is patched.",
        "arms": {
            "nominal": "The final initial predictor deployed with the standard nominal prediction rule. Zero new research; all initial fitting and observations remain charged.",
            "proposed_calibration": "The last actually fitted candidate calibration for that same final initial predictor, whether or not the controller adopted it. If unavailable, the arm is explicitly unavailable and not independently scored.",
            "stop": "The actual complete plan deployed by the initial visit, with no further research.",
            "short": "Run the production saved decision question from the same-time paid workspace and library; no question means no additional research.",
            "full_revision": "One complete production CognitiveAgent continuation from a separate copy of the same state."},
        "evaluation": "All initial visits and all continuation arms stop before any independent outcome is acquired. Every available plan receives identical 16-trajectory evaluation; costs are separate.",
        "research_costs": "Each initial and continuation visit keeps the 96-observation/24000-compute/5-round cap. Copying already fitted models adds no evidence or fit charge. These environments have zero instrument cost."}
    with output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    for global_index, partition, local_index, environment in roster:
        task = {"task_index": global_index, "task_id": environment.spec.task_id,
                "partition": partition, "local_index": local_index, "arms": {}}
        report["tasks"].append(task)
        try:
            initial_config = replace(config, partition=partition, seed=1237 + global_index * 10007)
            library = KnowledgeLibrary()
            initial = controller.CognitiveAgent(initial_config, library).run(environment).to_dict()
            knowledge = library.to_dict()
            workspace = json.loads(json.dumps(initial["research_state"]))
            visit_config = replace(initial_config, seed=initial_config.seed + 1000003)
            candidates = [action["details"] for action in initial["actions"] if action["action"] == "compare_initial_calibration"
                          and action["details"].get("candidate_program") == initial["program"]
                          and action["details"].get("previous_program") == initial["program"]]
            task.update(initial=initial, initial_knowledge=knowledge,
                        calibration_source={"action": "compare_initial_calibration", "matching_comparisons": len(candidates)},
                        visit_seed=visit_config.seed, saved_frontier=deepcopy(workspace.get("decision_frontier", {})))
            task["arms"]["nominal"] = retained(initial, "paid_predictor_nominal_comparator",
                model=candidates[-1]["previous_model"] if candidates else {"kind": "nominal"})
            task["arms"]["proposed_calibration"] = (
                retained(initial, "actual_paid_calibration_candidate", model=candidates[-1]["candidate_model"])
                if candidates else retained(initial, "no_paid_calibration_for_final_predictor", available=False))
            task["arms"]["stop"] = retained(initial, "no_additional_research")
            for arm in ("short", "full_revision"):
                task["active_arm"] = arm
                fresh_library = KnowledgeLibrary.from_dict(knowledge)
                fresh_environment = make_suite(config.seed, partition=partition)[local_index]
                if arm == "short":
                    if not workspace.get("decision_frontier"):
                        task["arms"][arm] = retained(initial, "no_executable_question")
                        continue
                    quote = decision_research.decision_quote(ResearchWorkspace.from_dict(workspace))
                    task["short_quote"] = quote
                    if quote["observation_cost"] > config.observation_budget or quote["compute_cost"] > config.compute_budget:
                        task["arms"][arm] = retained(initial, "unfunded_decision_question")
                        continue
                    result = decision_research.run_decision_research(fresh_environment, visit_config, fresh_library,
                                                                    research_state=deepcopy(workspace))
                else:
                    result = controller.CognitiveAgent(visit_config, fresh_library).run(fresh_environment,
                                                                                       research_state=deepcopy(workspace))
                task["arms"][arm] = {**result.to_dict(), "knowledge": fresh_library.to_dict(), "available": True, "measurement_cost": 0}
                save(output, report)
            task.pop("active_arm", None)
            for arm in task["arms"].values():
                arm["total_observation_cost"] = initial["observation_cost"] + arm["observation_cost"]
                arm["total_compute_cost"] = initial["compute_cost"] + arm["compute_cost"]
                arm["program_changed"] = arm["program"] != initial["program"]
                arm["model_changed"] = arm["uncertainty_model"] != initial["uncertainty_model"]
            save(output, report)
            print(json.dumps({"task_id": task["task_id"], "task_index": global_index, "research_stopped": True,
                "paid_calibration_comparisons": len(candidates), "saved_question": bool(task["saved_frontier"]),
                "additional_costs": {name: [arm["observation_cost"], arm["compute_cost"]] for name, arm in task["arms"].items()}}), flush=True)
        except Exception as error:
            report.update(status="failed", failure={"task_index": global_index,
                "arm": task.get("active_arm", "initial"), "error": repr(error)})
            save(output, report)
            raise
    # Independent outcomes are first requested here, after all seven tasks stop.
    for task in report["tasks"]:
        environment = make_suite(config.seed, partition=task["partition"])[task["local_index"]]
        evaluation_seed = task["visit_seed"] + 2000003
        rows = _new_evaluation_rows(environment, evaluation_seed, 16)
        blocks = trajectory_blocks(rows)
        for arm in task["arms"].values():
            if not arm["available"]:
                continue
            program = Program.from_dict(arm["program"])
            arm["evaluation"] = _measure(program, arm["fallback"], arm["uncertainty_model"], rows, environment.spec)
            arm["trajectory_decision_losses"] = [_measure(program, arm["fallback"], arm["uncertainty_model"], block,
                                                         environment.spec)["objective_loss"] for block in blocks]
        available = sum(arm["available"] for arm in task["arms"].values())
        task["evaluation_costs"] = {"seed": evaluation_seed, "observation_rows": len(rows), "trajectories": len(blocks),
            "measurement_cost": 0, "candidate_row_compute": available * 4 * len(rows), "plans": available,
            "computation": "Prediction and decisions, aggregate and per-trajectory reporting passes, for available plans only."}
        stopped = task["arms"]["stop"]["evaluation"]["objective_loss"]
        for arm in task["arms"].values():
            if arm["available"]:
                arm["decision_improvement_from_stopping"] = stopped-arm["evaluation"]["objective_loss"]
    report["status"] = "completed"
    save(output, report)
    summary = {key: value for key, value in report.items() if key != "tasks"}
    summary["source"] = str(output)
    summary["tasks"] = [{"task_index": task["task_index"], "task_id": task["task_id"], "partition": task["partition"],
        "initial_observation_cost": task["initial"]["observation_cost"], "initial_compute_cost": task["initial"]["compute_cost"],
        "initial_behavior": behavior(task["initial"]), "short_quote": task.get("short_quote"),
        "evaluation_costs": task["evaluation_costs"],
        "arms": {name: {**{key: arm[key] for key in (
            "available", "stop_reason", "observation_cost", "measurement_cost", "compute_cost", "total_observation_cost",
            "total_compute_cost", "program_changed", "model_changed")},
            "evaluation": arm.get("evaluation"), "trajectory_decision_losses": arm.get("trajectory_decision_losses"),
            "decision_improvement_from_stopping": arm.get("decision_improvement_from_stopping"), "behavior": behavior(arm)}
            for name, arm in task["arms"].items()}} for task in report["tasks"]]
    summary["arm_aggregates"] = {name: {
        "available_tasks": sum(task["arms"][name]["available"] for task in report["tasks"]),
        "mean_decision_loss": mean(task["arms"][name]["evaluation"]["objective_loss"] for task in report["tasks"]
                                   if task["arms"][name]["available"]) if any(task["arms"][name]["available"] for task in report["tasks"]) else None,
        "additional_observation_cost": sum(task["arms"][name]["observation_cost"] for task in report["tasks"]),
        "additional_compute_cost": sum(task["arms"][name]["compute_cost"] for task in report["tasks"])} for name in ARMS}
    comparable = [task for task in report["tasks"] if task["arms"]["proposed_calibration"]["available"]]
    summary["paired_initial_calibration"] = {"task_ids": [task["task_id"] for task in comparable],
        "mean_decision_loss": {name: mean(task["arms"][name]["evaluation"]["objective_loss"] for task in comparable)
                               if comparable else None for name in ("nominal", "proposed_calibration", "stop")}}
    with summary_path.open("x") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"summary": str(summary_path), "arm_aggregates": summary["arm_aggregates"]}), flush=True)


if __name__ == "__main__":
    main()
