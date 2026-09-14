#!/usr/bin/env python3
"""Run complete, matched research sessions with and without method selection."""

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.session import ResearchSession
from evaluate_cognition_session import evaluate_session
from run_method_continuation_study import without_method_selection


PARTITIONS = ("development", "confirmation", "forward", "continual", "stateful")


def _save(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _outcome(gain, threshold):
    return "methods_better" if gain > threshold else "ordinary_better" if gain < -threshold else "tied"


def _paired_summary(sessions, evaluations):
    threshold = sessions["methods"]["config"]["minimum_improvement"]
    tasks = []
    for left, right in zip(evaluations["methods"]["tasks"], evaluations["ordinary"]["tasks"]):
        if (left["task_index"], left["task_id"]) != (right["task_index"], right["task_id"]):
            raise ValueError("paired sessions must have identical task ordering")
        a, b = left["latest"], right["latest"]
        gain = b["loss"] - a["loss"]
        task = {"task_index": left["task_index"], "task_id": left["task_id"],
                "partition": a["partition"], "loss_kind": a["loss_kind"],
                "methods_loss": a["loss"], "ordinary_loss": b["loss"],
                "ordinary_minus_methods": gain,
                "numerical_outcome": _outcome(gain, 1e-9),
                "material_outcome": _outcome(gain, threshold),
                "methods_visits": left["visits"], "ordinary_visits": right["visits"]}
        if "decision" in a and "decision" in b:
            task["decision"] = {
                "methods_loss": a["decision"]["mean_decision_loss"],
                "ordinary_loss": b["decision"]["mean_decision_loss"],
                "ordinary_minus_methods": b["decision"]["mean_decision_loss"] - a["decision"]["mean_decision_loss"]}
        tasks.append(task)
    groups = {}
    for kind in sorted({task["loss_kind"] for task in tasks}):
        selected = [task for task in tasks if task["loss_kind"] == kind]
        groups[kind] = {"tasks": len(selected),
            "methods_mean_loss": mean(task["methods_loss"] for task in selected),
            "ordinary_mean_loss": mean(task["ordinary_loss"] for task in selected),
            "ordinary_minus_methods": mean(task["ordinary_minus_methods"] for task in selected),
            "numerical_outcomes": dict(Counter(task["numerical_outcome"] for task in selected)),
            "material_outcomes": dict(Counter(task["material_outcome"] for task in selected))}
    decisions = [task["decision"] for task in tasks if "decision" in task]
    costs = {}
    for arm, session in sessions.items():
        costs[arm] = {"status": session["status"], "completed_visits": session["completed_visits"],
            "visited_tasks": session["visited_tasks"], "observation_cost": session["total_observation_cost"],
            "compute_cost": session["total_compute_cost"], "unresolved_tasks": session["unresolved_tasks"],
            "method_choices": sum(bool(action["details"].get("learned_method"))
                for episode in session["episodes"] for action in episode["actions"]),
            "method_steps_executed": sum(action["action"] == "execute_method_step"
                for episode in session["episodes"] for action in episode["actions"]),
            "method_opportunities_selected": sum(episode["scheduler"]["reason"] == "learned_method_opportunity"
                for episode in session["episodes"]),
            "revisits": evaluations[arm]["revisits"]}
    return {"task_count": len(tasks), "material_improvement_threshold": threshold,
            "arms": costs, "by_prediction_loss": groups,
            "decision_loss": {"tasks": len(decisions),
                "methods_mean_loss": mean(row["methods_loss"] for row in decisions) if decisions else None,
                "ordinary_mean_loss": mean(row["ordinary_loss"] for row in decisions) if decisions else None,
                "ordinary_minus_methods": mean(row["ordinary_minus_methods"] for row in decisions) if decisions else None},
            "tasks": tasks}


def run_study(*, seed=919, output_directory=ROOT / "artifacts", label="v013", date="20260908"):
    config = ResearchConfig(seed=seed)
    prefix = f"hegel_machine_{label}_seed{seed}"
    paths = {arm: output_directory / f"{prefix}_{arm}_{date}.json" for arm in ("methods", "ordinary")}
    evaluation_paths = {arm: output_directory / f"{prefix}_{arm}_evaluation_{date}.json" for arm in paths}
    summary_path = output_directory / f"{prefix}_paired_{date}.json"
    output_directory.mkdir(parents=True, exist_ok=True)
    for path in [*paths.values(), *evaluation_paths.values(), summary_path]:
        if path.exists():
            raise FileExistsError(f"preserve the existing experiment: {path}")
    sessions = {}
    elapsed = {}
    for arm in ("methods", "ordinary"):
        environments = [environment for partition in PARTITIONS
                        for environment in make_suite(seed, partition=partition)]
        if len(environments) != 65:
            raise ValueError("this specified study requires the complete 65-task suite")
        session = ResearchSession(config, environments,
            observation_budget=65 * 96 * 2, compute_budget=65 * 24000 * 2, max_visits=2)
        session.state["partitions"] = list(PARTITIONS)
        context = without_method_selection() if arm == "ordinary" else nullcontext()
        started = perf_counter()

        def progress(checkpoint):
            latest = checkpoint["episodes"][-1]
            print(json.dumps({"arm": arm, "visits": checkpoint["completed_visits"],
                "task_index": latest["task_index"], "task_id": latest["task_id"],
                "task_visit": latest["visit"], "reason": latest["scheduler"]["reason"],
                "validation_loss": latest["validation_loss"],
                "observation_cost": checkpoint["total_observation_cost"],
                "compute_cost": checkpoint["total_compute_cost"],
                "elapsed_seconds": round(perf_counter() - started, 2)}), flush=True)

        with context:
            sessions[arm] = session.run(on_checkpoint=progress)
        elapsed[arm] = perf_counter() - started
        _save(paths[arm], sessions[arm])
        print(json.dumps({"arm": arm, "status": sessions[arm]["status"],
                          "checkpoint": str(paths[arm]), "elapsed_seconds": elapsed[arm]}), flush=True)

    # Independent outcomes are accessed only after both learning sessions stop.
    evaluations = {}
    for arm in paths:
        evaluations[arm] = evaluate_session(sessions[arm])
        _save(evaluation_paths[arm], evaluations[arm])
    report = {"experiment": "paired_complete_method_research_sessions", "status": "completed",
        "seed": seed, "config": asdict(config), "partitions": list(PARTITIONS),
        "limits_per_arm": sessions["methods"]["limits"],
        "intervention": "The ordinary arm uses without_method_selection(): only ResearchMethods.choose and opportunity return None. All other knowledge, policy learning, method feedback and research actions remain enabled.",
        "initial_state": "Each arm begins with a fresh knowledge library and the same complete ordered public task suite.",
        "evaluation": "evaluate_cognition_session.evaluate_session is called separately for each final checkpoint after both complete sessions stop; evaluation responses never enter research state.",
        "selection": f"Seed {seed} was specified before running either arm; this run is retained irrespective of sign.",
        "checkpoints": {arm: str(path) for arm, path in paths.items()},
        "evaluations": {arm: str(path) for arm, path in evaluation_paths.items()},
        "wall_time_seconds": elapsed, "summary": _paired_summary(sessions, evaluations)}
    _save(summary_path, report)
    print(json.dumps({key: value for key, value in report["summary"].items() if key != "tasks"}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=919)
    parser.add_argument("--output-directory", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--label", default="v013")
    parser.add_argument("--date", default="20260908")
    args = parser.parse_args()
    run_study(seed=args.seed, output_directory=args.output_directory, label=args.label, date=args.date)


if __name__ == "__main__":
    main()
