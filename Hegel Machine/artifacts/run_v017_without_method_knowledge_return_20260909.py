"""Specified seed 1237 ablation: omit only knowledge feedback to research methods."""

from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.session import ResearchSession
from evaluate_cognition_session import evaluate_session
from run_method_session_study import PARTITIONS, _save


def main():
    prefix = "hegel_machine_v017_without_method_knowledge_return_seed1237"
    checkpoint_path = ROOT / "artifacts" / f"{prefix}_methods_20260909.json"
    evaluation_path = ROOT / "artifacts" / f"{prefix}_methods_evaluation_20260909.json"
    summary_path = ROOT / "artifacts" / f"{prefix}_summary_20260909.json"
    for path in (checkpoint_path, evaluation_path, summary_path):
        if path.exists():
            raise FileExistsError(f"preserve the existing experiment: {path}")
    config = ResearchConfig(seed=1237)
    environments = [environment for partition in PARTITIONS
                    for environment in make_suite(1237, partition=partition)]
    session = ResearchSession(config, environments,
        observation_budget=65 * 96 * 2, compute_budget=65 * 24000 * 2, max_visits=2)
    session.state["partitions"] = list(PARTITIONS)
    started = perf_counter()

    def progress(checkpoint):
        episode = checkpoint["episodes"][-1]
        print(json.dumps({"arm": "without_method_knowledge_return",
            "visits": checkpoint["completed_visits"], "task_index": episode["task_index"],
            "task_id": episode["task_id"], "task_visit": episode["visit"],
            "reason": episode["scheduler"]["reason"],
            "observation_cost": checkpoint["total_observation_cost"],
            "compute_cost": checkpoint["total_compute_cost"],
            "elapsed_seconds": round(perf_counter() - started, 2)}), flush=True)

    with patch("hegel_machine.cognition.research_methods.ResearchMethods.observe_knowledge_transfer", return_value=None):
        checkpoint = session.run(on_checkpoint=progress)
    _save(checkpoint_path, checkpoint)
    # This arm stops before independent environment responses are accessed.
    evaluation = evaluate_session(checkpoint)
    _save(evaluation_path, evaluation)
    decisions = [task["latest"]["decision"]["mean_decision_loss"]
                 for task in evaluation["tasks"] if "decision" in task.get("latest", {})]
    report = {"experiment": "complete_session_without_method_knowledge_return",
        "status": checkpoint["status"], "seed": 1237, "config": asdict(config),
        "intervention": "Only ResearchMethods.observe_knowledge_transfer is temporarily replaced with a method returning None. Knowledge-producing method associations, knowledge return recording and ordering, existing method execution feedback and all other research remain enabled.",
        "initial_state": "Fresh knowledge library, identical 65-task order and budgets to the seed 1237 mainline methods arm.",
        "checkpoint": str(checkpoint_path), "evaluation": str(evaluation_path),
        "visited_tasks": checkpoint["visited_tasks"], "completed_visits": checkpoint["completed_visits"],
        "observation_cost": checkpoint["total_observation_cost"],
        "compute_cost": checkpoint["total_compute_cost"],
        "method_choices": sum(bool(action["details"].get("learned_method"))
            for episode in checkpoint["episodes"] for action in episode["actions"]),
        "method_steps_executed": sum(action["action"] == "execute_method_step"
            for episode in checkpoint["episodes"] for action in episode["actions"]),
        "method_overrides": [{"task_index": episode["task_index"], "task_id": episode["task_id"],
            "round": action["round"], "method_id": action["details"]["learned_method"]["method_id"]}
            for episode in checkpoint["episodes"] for action in episode["actions"]
            if action["details"].get("learned_method", {}).get("changes_action")],
        "metrics_by_loss": evaluation["metrics_by_loss"],
        "mean_decision_loss": mean(decisions) if decisions else None,
        "revisits": evaluation["revisits"], "wall_time_seconds": perf_counter() - started}
    _save(summary_path, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
