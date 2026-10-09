#!/usr/bin/env python3
"""Current offline research ablations on one matched continuing task sequence."""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.passive_research import FixedExperimentResearchAgent
from hegel_machine.cognition.unified_session import UnifiedResearchSession, task_sequence


ARMS = {
    "full": {},
    "fixed_library": {"strategy": "fixed", "use_memory": False},
    "passive_experiments": {"active_probes": False},
    "no_reuse": {"use_memory": False},
}


def protocol(seed=12901, *, config=None, order=(1, 2, 3), measurement_budget=768):
    config = config or ResearchConfig(observation_budget=192, compute_budget=120000,
        minimum_improvement=1e-7, max_rounds=4, seed=seed)
    return {
        "experiment": "current_offline_source_successor_ablation",
        "tasks": task_sequence(seed, order), "research_config": asdict(config),
        "measurement_budget": measurement_budget, "arms": deepcopy(ARMS),
        "evaluation": {"length": 64, "trajectories": 4, "seed_offset": 1000003},
        "solved_threshold": 1e-6,
        "budget_contract": "The same per-task observation, measurement and candidate-row caps; "
            "the complete sequence is bounded by their sum. Actual spending is reported separately.",
        "information_contract": "Every arm starts from its own empty knowledge library and receives "
            "only the public laboratory interface and its paid observations. All arms finish learning "
            "before any independent final score is requested.",
        "arm_contracts": {
            "full": "Current UnifiedResearchSession, including shared knowledge and funded revisits.",
            "fixed_library": "Existing fixed quadratic/two-step-history basis; no learned library.",
            "passive_experiments": "Existing FixedExperimentResearchAgent: four precommitted random "
                "trajectories and all instruments, learned representation/library; no adaptive revisits.",
            "no_reuse": "Current research and active experiments with cross-task memory disabled.",
        },
        "pending_external_controls": {
            "ordinary_code_agent": "not_run_external_model_deferred",
            "same_llm_module_ablations": "not_run_external_model_deferred",
        },
    }


def _train_arm(plan, arm, *, lab_factory=None):
    config = ResearchConfig(**{**plan["research_config"], **plan["arms"][arm]})
    session = UnifiedResearchSession(config, tasks=plan["tasks"],
        measurement_budget=plan["measurement_budget"], library=KnowledgeLibrary(),
        lab_factory=lab_factory)
    if arm != "passive_experiments":
        session.run()
        return session
    for position, item in enumerate(plan["tasks"]):
        available = [concept.concept_id for concept in session.library.concepts]
        with session.lab_factory(item) as laboratory:
            result = FixedExperimentResearchAgent(replace(config, seed=item["seed"]),
                measurement_budget=plan["measurement_budget"], library=session.library).run(laboratory)
        session.state["results"].append({"task_position": position,
            "available_concepts": available, "research": result})
    session.state["status"] = "completed"
    return session


def _compact(report, plan):
    scores = {row["task_position"]: row for row in report["evaluation"]}
    rows = []
    for entry in report["results"]:
        result, position = entry["research"], entry["task_position"]
        score = scores[position]
        transfer = result.get("transfer", [])
        confirmation_actions = [action for action in result.get("actions", [])
            if action.get("action") in {"defer_frozen_confirmation", "research_action_confirmation"}]
        rows.append({"task_position": position, "task": plan["tasks"][position],
            "status": result["status"], "mse": score["mse"],
            "solved": score["mse"] <= plan["solved_threshold"],
            "confirmation_loss": result.get("confirmation_loss"),
            "selected_kind": result["selected_kind"], "selected_channels": result["selected_channels"],
            "available_concepts": entry["available_concepts"],
            "used_sources": result.get("used_sources", []),
            # Keep transfer attempts separate from accepted source use. A
            # rejected transfer can still be the selected provisional model;
            # collapsing those fields made the full arm look as if it had
            # silently reused a source and obscured why no successor visit was
            # funded.
            "transfer_attempts": len(transfer),
            "accepted_transfers": sum(bool(row.get("accepted")) for row in transfer),
            "rejected_transfers": sum(row.get("status") == "counterexample"
                for row in transfer),
            "refuted_transfers": sum(row.get("accepted") is False for row in transfer),
            "confirmation_actions": [{key: action.get(key) for key in
                ("action", "status", "reason")} for action in confirmation_actions],
            "costs": result["costs"], "evaluation_rows": score["rows"]})
    costs = report["costs"]
    count = len(plan["tasks"])
    limits = {"observation_cost": count * plan["research_config"]["observation_budget"],
        "measurement_cost": count * plan["measurement_budget"],
        "compute_cost": count * plan["research_config"]["compute_budget"]}
    if any(costs[key] > limit for key, limit in limits.items()):
        raise ValueError("an arm exceeded the matched sequence resource limits")
    return {"status": report["status"], "tasks": rows, "costs": costs,
        "resource_limits": limits, "solved": sum(row["solved"] for row in rows),
        "mean_mse": mean(row["mse"] for row in rows),
        "revisit_count": len(report.get("revisits", [])),
        "screening_compute_cost": report.get("screening_compute_cost", 0),
        "learned_concepts": len(report["knowledge"]["concepts"])}


def run_benchmark(plan, *, lab_factory=None):
    sessions = {arm: _train_arm(plan, arm, lab_factory=lab_factory) for arm in plan["arms"]}
    results = {arm: _compact(session.evaluate(**plan["evaluation"]), plan)
        for arm, session in sessions.items()}
    full = results["full"]
    paired = {}
    for arm, result in results.items():
        if arm == "full":
            continue
        differences = [left["mse"] - right["mse"]
            for left, right in zip(full["tasks"], result["tasks"])]
        paired[arm] = {"full_minus_control_mse": differences,
            "mean_mse_difference": mean(differences),
            "full_wins_ties_losses": [sum(value < -1e-9 for value in differences),
                sum(abs(value) <= 1e-9 for value in differences), sum(value > 1e-9 for value in differences)],
            "cost_difference": {key: full["costs"][key] - result["costs"][key]
                for key in ("observation_cost", "measurement_cost", "compute_cost")}}
    return {"protocol": deepcopy(plan), "results": results, "paired_full_vs_control": paired,
        "scope": "Offline development sequence; these arms are algorithmic controls, not ordinary "
            "Code Agent or same-LLM evidence. Passive comparison also removes adaptive revisits.",
        "external_model_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=12901)
    parser.add_argument("--output", type=Path,
        default=ROOT / "artifacts/current_offline_ablation_20261004.json")
    args = parser.parse_args()
    plan = protocol(args.seed)
    # A completed cohort is not silently replaced by another code state.
    if args.output.exists():
        raise FileExistsError("preserve the completed result and choose another output path")
    result = run_benchmark(plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({arm: {key: report[key] for key in ("solved", "mean_mse", "costs")}
        for arm, report in result["results"].items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
