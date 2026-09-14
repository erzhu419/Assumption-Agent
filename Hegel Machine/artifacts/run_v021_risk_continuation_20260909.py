"""Execute stop, a saved short decision question, and a full research visit."""

import argparse
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
from hegel_machine.cognition.controller import CognitiveAgent
from hegel_machine.cognition.decision_research import decision_quote, run_decision_research
from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.risk_search import trajectory_blocks
from hegel_machine.cognition.workspace import ResearchWorkspace
from run_method_continuation_study import _measure, _new_evaluation_rows


def stopped_arm(episode, library, reason):
    return {"program": deepcopy(episode["program"]), "fallback": episode["fallback"],
        "uncertainty_model": deepcopy(episode["uncertainty_model"]),
        "expression": episode["expression"], "stop_reason": reason,
        "observation_cost": 0, "compute_cost": 0, "actions": [],
        "research_state": deepcopy(episode["research_state"]), "knowledge": deepcopy(library)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="v021_risk_continuation")
    args = parser.parse_args()
    output = ROOT / f"artifacts/{args.label}_20260909.json"
    summary_path = output.with_name(output.stem + "_summary.json")
    if output.exists() or summary_path.exists():
        raise FileExistsError("Keep previous continuation experiments; choose a new label")
    config = ResearchConfig(seed=1237)
    sources = make_suite(config.seed, partition="development")
    roster = [index for index, environment in enumerate(sources) if environment.spec.decision_costs is not None]
    report = {"experiment": "actual_three_arm_risk_continuation", "version": __version__,
        "seed": config.seed, "partition": "development", "task_indices": roster,
        "config": asdict(config), "status": "running", "tasks": [],
        "selection": "All public decision-cost tasks in the preselected seed 1237 development suite.",
        "common_state": "One actual initial visit with its same-time knowledge and paid workspace per task. Every arm receives a separate copy of that exact state.",
        "arms": {"stop": "Zero additional research.",
            "reference_only": "Execute the actual saved decision_frontier with run_decision_research; no constructed challenger or fitting in the experiment driver. With no frontier, stop with no_executable_question.",
            "revision": "Execute one complete ordinary CognitiveAgent visit with the same per-visit limits."},
        "evaluation": "All tasks and arms stop before any independent outcome is requested. All three decisions are then scored on identical fresh trajectories, with evaluation costs separate."}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    for index in roster:
        library = KnowledgeLibrary()
        initial_config = replace(config, seed=config.seed + index * 10007)
        initial = CognitiveAgent(initial_config, library).run(sources[index]).to_dict()
        saved_knowledge = library.to_dict()
        workspace = json.loads(json.dumps(initial["research_state"]))
        visit_config = replace(initial_config, seed=initial_config.seed + 1000003)
        task = {"task_index": index, "task_id": sources[index].spec.task_id,
            "initial": initial, "initial_knowledge": saved_knowledge, "visit_seed": visit_config.seed,
            "saved_frontier": deepcopy(workspace.get("decision_frontier", {})),
            "arms": {"stop": stopped_arm(initial, saved_knowledge, "no_additional_research")}}
        for arm in ("reference_only", "revision"):
            fresh_library = KnowledgeLibrary.from_dict(saved_knowledge)
            environment = make_suite(config.seed, partition="development")[index]
            if arm == "reference_only":
                if not workspace.get("decision_frontier"):
                    task["arms"][arm] = stopped_arm(initial, saved_knowledge, "no_executable_question")
                    continue
                quote = decision_quote(ResearchWorkspace.from_dict(workspace))
                task["short_quote"] = quote
                if (quote["observation_cost"] > config.observation_budget or
                        quote["compute_cost"] > config.compute_budget):
                    task["arms"][arm] = stopped_arm(initial, saved_knowledge, "unfunded_decision_question")
                    continue
                result = run_decision_research(environment, visit_config, fresh_library,
                                               research_state=deepcopy(workspace))
            else:
                result = CognitiveAgent(visit_config, fresh_library).run(environment,
                                                                        research_state=deepcopy(workspace))
            task["arms"][arm] = {**result.to_dict(), "knowledge": fresh_library.to_dict()}
        for arm in task["arms"].values():
            arm["total_observation_cost"] = initial["observation_cost"] + arm["observation_cost"]
            arm["total_compute_cost"] = initial["compute_cost"] + arm["compute_cost"]
        report["tasks"].append(task)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"task_index": index, "research_stopped": True,
            "saved_question": bool(task["saved_frontier"]),
            "additional_costs": {arm: [value["observation_cost"], value["compute_cost"]]
                                 for arm, value in task["arms"].items()}}), flush=True)

    # Independent rows are deliberately acquired in a separate final pass.
    for task in report["tasks"]:
        evaluator = make_suite(config.seed, partition="development")[task["task_index"]]
        evaluation_seed = task["visit_seed"] + 2000003
        rows = _new_evaluation_rows(evaluator, evaluation_seed, 16)
        blocks = trajectory_blocks(rows)
        for arm in task["arms"].values():
            program = Program.from_dict(arm["program"])
            arm["evaluation"] = _measure(program, arm["fallback"], arm["uncertainty_model"], rows, evaluator.spec)
            arm["evaluation"]["trajectory_results"] = [
                _measure(program, arm["fallback"], arm["uncertainty_model"], block, evaluator.spec)
                for block in blocks]
        task["evaluation_costs"] = {"seed": evaluation_seed, "rows": len(rows),
            "trajectories": len(blocks), "candidate_row_compute": 3 * len(rows) * 4,
            "computation": "Three arms, prediction plus decisions, aggregate and trajectory reporting passes."}
        before = task["arms"]["stop"]["evaluation"]["objective_loss"]
        for arm in ("reference_only", "revision"):
            task["arms"][arm]["decision_improvement_from_stopping"] = (
                before - task["arms"][arm]["evaluation"]["objective_loss"])
    report["status"] = "completed"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    compact = {key: value for key, value in report.items() if key not in {"tasks"}}
    compact["tasks"] = [{"task_index": task["task_index"], "task_id": task["task_id"],
        "saved_question": bool(task["saved_frontier"]), "short_quote": task.get("short_quote"),
        "evaluation_costs": task["evaluation_costs"],
        "arms": {name: {key: value[key] for key in (
            "stop_reason", "expression", "observation_cost", "compute_cost", "total_observation_cost",
            "total_compute_cost", "evaluation")} for name, value in task["arms"].items()}}
        for task in report["tasks"]]
    compact["mean_decision_loss"] = {arm: mean(task["arms"][arm]["evaluation"]["objective_loss"]
        for task in report["tasks"]) for arm in ("stop", "reference_only", "revision")}
    compact["source"] = str(output)
    summary_path.write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summary": str(summary_path), "mean_decision_loss": compact["mean_decision_loss"]}), flush=True)


if __name__ == "__main__":
    main()
