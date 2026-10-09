#!/usr/bin/env python3
"""Compare existing offline research arms on the six-task composition cycle."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from statistics import mean
import sys
from time import process_time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_composed_local_cycle import protocol as cycle_protocol
from run_current_offline_ablation import ARMS as PREVIOUS_ARMS, _compact, _train_arm
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary, _called_definitions
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_session import UnifiedResearchSession, laboratory_for


ARMS = {name: deepcopy(PREVIOUS_ARMS[name])
    for name in ("full", "fixed_library", "passive_experiments")}
ARMS["task_local_memory"] = {}


def protocol(seed=10078, *, apparatus_seed=9013, profile="development"):
    plan = cycle_protocol(seed, apparatus_seed=apparatus_seed, profile=profile)
    plan.update(experiment="composed_offline_research_comparison", arms=deepcopy(ARMS))
    plan["budget_contract"] = "Equal per-task caps for initial visits and equal summed sequence caps. " \
        "Full may pool unspent resources for revisits; task-local sessions retain only their own task's resources. " \
        "Research and terminal-evaluation costs are reported separately."
    plan["information_contract"] = "Each arm starts with empty knowledge. Task-local sessions restart " \
        "with an empty library on every task while preserving within-task learning and confirmation. " \
        "Every arm finishes all learning before any terminal score is requested; evaluation uses " \
        "each original task seed plus the common evaluation seed offset."
    plan["arm_contracts"] = {
        "full": "Current researcher with cross-task knowledge and pooled-budget revisits.",
        "fixed_library": "Fixed finite quadratic/two-step-history basis, no learned library; "
            "precommitted finest legal instrument view (fine).",
        "passive_experiments": "Four precommitted random trajectories with the finest legal instrument "
            "view (fine), learned representations and shared library; no adaptive revisits.",
        "task_local_memory": "The full researcher, including within-task learned instruments and "
            "confirmation reserves, independently completes each task with use_memory=True. "
            "No cross-task knowledge or resource borrowing; source IDs are local to each task.",
    }
    plan["pending_external_controls"] = {"ordinary_code_agent": "not_run_external_model_deferred",
        "same_llm_module_ablations": "not_run_external_model_deferred"}
    plan["acceptance"] = "Report every task's terminal MSE, fresh confirmation, and actual costs " \
        "for every arm, including failures. No quality-dependent case filtering."
    plan["scope"] = "One finite object-exchange family. These are offline algorithmic controls, " \
        "not ordinary Code Agent or same-LLM comparisons. Passive experiments also remove adaptive revisits."
    return plan


def _task_local_session(plan, lab_factory):
    config = ResearchConfig(**plan["research_config"])
    combined = UnifiedResearchSession(config, tasks=plan["tasks"],
        measurement_budget=plan["measurement_budget"], library=KnowledgeLibrary(), lab_factory=lab_factory)
    learned = []
    for position, task in enumerate(plan["tasks"]):
        session = UnifiedResearchSession(config, tasks=[task],
            measurement_budget=plan["measurement_budget"], library=KnowledgeLibrary(), lab_factory=lab_factory)
        session.run()
        entry = deepcopy(session.state["results"][0])
        entry["task_position"] = position
        combined.state["results"].append(entry)
        combined.state["screening_compute_cost"] += session.state["screening_compute_cost"]
        combined.state["revisits"].extend({**visit, "task_position": position}
            for visit in session.state["revisits"])
        learned.append(len(session.library.concepts))
    combined.state["status"] = "completed"
    # This empty-library container only evaluates the already frozen programs;
    # it never combines or trains on the independent task-local libraries.
    return combined, learned


def run_benchmark(plan, *, lab_factory=None):
    factory = lab_factory or laboratory_for
    sessions, training_cpu, local_learned = {}, {}, []
    for arm in plan["arms"]:
        start = process_time()
        if arm == "task_local_memory":
            sessions[arm], local_learned = _task_local_session(plan, factory)
        else:
            sessions[arm] = _train_arm(plan, arm, lab_factory=factory)
        training_cpu[arm] = process_time() - start
    results = {}
    tolerance = plan["research_config"]["minimum_improvement"]
    # Terminal scores cannot influence any arm's learning or resource allocation.
    for arm, session in sessions.items():
        start = process_time()
        report = session.evaluate(**plan["evaluation"])
        evaluation_cpu = process_time() - start
        result = _compact(report, plan)
        for row, entry, evaluation in zip(result["tasks"], report["results"], report["evaluation"]):
            research = entry["research"]
            definitions = _called_definitions(Program.from_dict(research["program"]))
            row["nested_call_sources"] = sorted({d.source_concept_id for d in definitions
                if d.source_concept_id})
            row["called_definition_ids"] = [d.definition_id for d in definitions]
            row["concepts_added"] = research.get("concepts_added", [])
            row["evaluation_costs"] = {key: evaluation[key]
                for key in ("observation_cost", "measurement_cost", "compute_cost")}
            row["fresh_confirmation_passed"] = row["status"] == "completed" \
                and row["confirmation_loss"] is not None and row["confirmation_loss"] <= tolerance
        result["fresh_confirmed"] = sum(row["fresh_confirmation_passed"] for row in result["tasks"])
        result["evaluation_costs"] = {key: sum(row["evaluation_costs"][key] for row in result["tasks"])
            for key in ("observation_cost", "measurement_cost", "compute_cost")}
        result["training_cpu_seconds"] = training_cpu[arm]
        result["evaluation_cpu_seconds"] = evaluation_cpu
        result["learned_concepts_scope"] = "shared_sequence_library"
        if arm == "task_local_memory":
            result["learned_concepts"] = sum(local_learned)
            result["per_task_learned_concepts"] = local_learned
            result["learned_concepts_scope"] = "sum_over_independent_task_libraries"
        results[arm] = result
    full, paired = results["full"], {}
    for arm, result in results.items():
        if arm == "full":
            continue
        differences = [left["mse"] - right["mse"]
            for left, right in zip(full["tasks"], result["tasks"])]
        costs = {key: full["costs"][key] - result["costs"][key]
            for key in ("observation_cost", "measurement_cost", "compute_cost")}
        paired[arm] = {"full_minus_control_mse": differences,
            "mean_mse_difference": mean(differences),
            "full_wins_ties_losses": [sum(v < -1e-9 for v in differences),
                sum(abs(v) <= 1e-9 for v in differences), sum(v > 1e-9 for v in differences)],
            "cost_difference": costs,
            "both_solve_all_with_full_pareto_saving": full["solved"] == result["solved"] == len(plan["tasks"])
                and all(value <= 0 for value in costs.values()) and any(value < 0 for value in costs.values())}
    return {"protocol": deepcopy(plan), "results": results,
        "paired_full_vs_control": paired, "external_model_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=10078)
    parser.add_argument("--apparatus-seed", type=int, default=9013)
    parser.add_argument("--profile", choices=("development", "confirmation"), default="development")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve the completed result and choose another output path")
    result = run_benchmark(protocol(args.seed, apparatus_seed=args.apparatus_seed, profile=args.profile))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({arm: {key: report[key] for key in ("solved", "fresh_confirmed", "mean_mse", "costs")}
        for arm, report in result["results"].items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
