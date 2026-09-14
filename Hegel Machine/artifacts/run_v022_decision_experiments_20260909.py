"""Compare actual short decision mechanisms and a full visit on fixed risk tasks."""

import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from statistics import mean
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hegel_machine import __version__
from hegel_machine.cognition import decision_research
from hegel_machine.cognition.controller import CognitiveAgent
from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.risk_search import trajectory_blocks
from hegel_machine.cognition.workspace import ResearchWorkspace
from run_method_continuation_study import _measure, _new_evaluation_rows

ARMS = ("stop", "fixed_representative", "targeted_package", "conflict_extra", "full_revision")


def stopped_arm(episode, library, reason):
    return {"program": deepcopy(episode["program"]), "fallback": episode["fallback"],
        "uncertainty_model": deepcopy(episode["uncertainty_model"]),
        "expression": episode["expression"], "stop_reason": reason,
        "observation_cost": 0, "measurement_cost": 0, "compute_cost": 0, "actions": [],
        "research_state": deepcopy(episode["research_state"]), "knowledge": deepcopy(library)}


def behavior(episode):
    """Keep actual actions, including failed designs and still-pending evidence."""
    actions = episode["actions"]
    counts = Counter(action["action"] for action in actions)
    observed = [{"reason": action["reason"], "observation_cost": action["observation_cost"],
                 "details": action["details"]} for action in actions if action["action"] == "observe"]
    decision = [{"action": action["action"], "reason": action["reason"],
                 "observation_cost": action["observation_cost"], "compute_cost": action["compute_cost"],
                 "details": action["details"]} for action in actions
                if "decision" in action["action"] or action["action"] == "continuation_feedback"]
    # Action costs explain parts of the fully charged EpisodeResult cost; they
    # are not treated as an independent total because reporting can be implicit.
    costs = {kind: {"observation_cost": sum(action["observation_cost"] for action in actions if action["action"] == kind),
                    "compute_cost": sum(action["compute_cost"] for action in actions if action["action"] == kind)}
             for kind in sorted(counts)}
    frontier = (episode.get("research_state") or {}).get("decision_frontier", {})
    return {"action_counts": dict(counts), "observed_probes": observed,
            "decision_actions": decision, "action_costs": costs,
            "remaining_decision_frontier": deepcopy(frontier)}


def write_report(path, report):
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="v022_release_decision_experiments")
    args = parser.parse_args()
    output = ROOT / f"artifacts/{args.label}_20260909.json"
    summary_path = output.with_name(output.stem + "_summary.json")
    if output.exists() or summary_path.exists():
        raise FileExistsError("Keep prior trials; choose a new output label")
    # This is the production helper, not a new experiment-facing wrapper.
    if not callable(getattr(decision_research, "_additional_decision_pair", None)):
        raise RuntimeError("Wait for the frozen 0.22 decision research runtime")
    config = ResearchConfig(seed=1237)
    sources = make_suite(config.seed, partition="development")
    roster = [index for index, environment in enumerate(sources) if environment.spec.decision_costs is not None]
    report = {"experiment": "actual_decision_mechanism_arms", "version": __version__,
        "seed": config.seed, "partition": "development", "task_indices": roster,
        "config": asdict(config), "status": "running", "tasks": [],
        "selection": "Every public decision-cost task in the preselected seed 1237 development suite.",
        "common_state": "An actual initial visit, with the same-time library and paid workspace copied separately into each arm.",
        "arms": {
            "stop": "Deploy the initial plan with zero additional research.",
            "fixed_representative": "Call run_decision_research; scoped experimental patches return {} from plan_decision_probes and [] from _additional_decision_pair. Evidence/adoption logic is unchanged.",
            "targeted_package": "Call run_decision_research; only _additional_decision_pair returns []. Targeted design, observations and ordinary reference evidence remain active.",
            "conflict_extra": "Call unmodified run_decision_research, including targeted design and at most one actual conditional extra pair. This is the production short arm.",
            "full_revision": "One normal CognitiveAgent visit with the same per-visit research limits."},
        "evaluation": "Every arm on every task stops before independent outcomes are acquired. All five plans are scored on identical new trajectories. Targeted rows are not whole-task gain estimates.",
        "costs": "Per-arm additional caps are 96 observations and 24000 computation. These cognitive environments have no instrument charge; measurement_cost is zero. Evaluation costs are separate."}
    with output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    for index in roster:
        task = {"task_index": index, "task_id": sources[index].spec.task_id, "arms": {}}
        report["tasks"].append(task)
        try:
            library = KnowledgeLibrary()
            initial_config = replace(config, seed=config.seed + index * 10007)
            initial = CognitiveAgent(initial_config, library).run(sources[index]).to_dict()
            saved_knowledge = library.to_dict()
            workspace = json.loads(json.dumps(initial["research_state"]))
            visit_config = replace(initial_config, seed=initial_config.seed + 1000003)
            task.update(initial=initial, initial_knowledge=saved_knowledge, visit_seed=visit_config.seed,
                        saved_frontier=deepcopy(workspace.get("decision_frontier", {})))
            task["arms"]["stop"] = stopped_arm(initial, saved_knowledge, "no_additional_research")
            for arm in ARMS[1:]:
                task["active_arm"] = arm
                fresh_library = KnowledgeLibrary.from_dict(saved_knowledge)
                environment = make_suite(config.seed, partition="development")[index]
                if arm != "full_revision":
                    if not workspace.get("decision_frontier"):
                        task["arms"][arm] = stopped_arm(initial, saved_knowledge, "no_executable_question")
                        continue
                    quote = decision_research.decision_quote(ResearchWorkspace.from_dict(workspace))
                    task["minimum_short_quote"] = quote
                    if quote["observation_cost"] > config.observation_budget or quote["compute_cost"] > config.compute_budget:
                        task["arms"][arm] = stopped_arm(initial, saved_knowledge, "unfunded_decision_question")
                        continue
                    with ExitStack() as stack:
                        if arm == "fixed_representative":
                            stack.enter_context(patch.object(decision_research, "plan_decision_probes", return_value={}))
                        if arm in {"fixed_representative", "targeted_package"}:
                            stack.enter_context(patch.object(decision_research, "_additional_decision_pair", return_value=[]))
                        result = decision_research.run_decision_research(environment, visit_config, fresh_library,
                                                                        research_state=deepcopy(workspace))
                else:
                    result = CognitiveAgent(visit_config, fresh_library).run(environment, research_state=deepcopy(workspace))
                task["arms"][arm] = {**result.to_dict(), "knowledge": fresh_library.to_dict(), "measurement_cost": 0}
                write_report(output, report)
            task.pop("active_arm", None)
            for arm in task["arms"].values():
                arm["total_observation_cost"] = initial["observation_cost"] + arm["observation_cost"]
                arm["total_compute_cost"] = initial["compute_cost"] + arm["compute_cost"]
                arm["program_changed"] = arm["program"] != initial["program"]
                arm["model_changed"] = arm["uncertainty_model"] != initial["uncertainty_model"]
            write_report(output, report)
            print(json.dumps({"task_id": task["task_id"], "research_stopped": True,
                "saved_question": bool(task["saved_frontier"]),
                "additional_costs": {name: [arm["observation_cost"], arm["compute_cost"]]
                                     for name, arm in task["arms"].items()}}), flush=True)
        except Exception as error:
            report.update(status="failed", failure={"task_index": index, "arm": task.get("active_arm", "initial"),
                                                    "error": repr(error)})
            write_report(output, report)
            raise

    # No evaluator has been called above. Independent outcomes cannot steer
    # later research arms or the development-task roster.
    for task in report["tasks"]:
        evaluator = make_suite(config.seed, partition="development")[task["task_index"]]
        evaluation_seed = task["visit_seed"] + 2000003
        rows = _new_evaluation_rows(evaluator, evaluation_seed, 16)
        blocks = trajectory_blocks(rows)
        for arm in task["arms"].values():
            program = Program.from_dict(arm["program"])
            arm["evaluation"] = _measure(program, arm["fallback"], arm["uncertainty_model"], rows, evaluator.spec)
            arm["evaluation"]["trajectory_results"] = [
                _measure(program, arm["fallback"], arm["uncertainty_model"], block, evaluator.spec) for block in blocks]
        task["evaluation_costs"] = {"seed": evaluation_seed, "observation_rows": len(rows),
            "trajectories": len(blocks), "candidate_row_compute": len(ARMS) * len(rows) * 4,
            "measurement_cost": 0,
            "computation": "Five arms, prediction plus decisions, aggregate and per-trajectory reporting passes."}
        before = task["arms"]["stop"]["evaluation"]["objective_loss"]
        for name, arm in task["arms"].items():
            arm["decision_improvement_from_stopping"] = before - arm["evaluation"]["objective_loss"]
            arm["trajectory_decision_improvements"] = [
                old["objective_loss"] - new["objective_loss"] for old, new in zip(
                    task["arms"]["stop"]["evaluation"]["trajectory_results"], arm["evaluation"]["trajectory_results"])]
    report["status"] = "completed"
    write_report(output, report)
    compact = {key: value for key, value in report.items() if key != "tasks"}
    compact["tasks"] = [{"task_index": task["task_index"], "task_id": task["task_id"],
        "saved_question": bool(task["saved_frontier"]), "minimum_short_quote": task.get("minimum_short_quote"),
        "evaluation_costs": task["evaluation_costs"],
        "arms": {name: {**{key: arm[key] for key in (
            "stop_reason", "expression", "observation_cost", "measurement_cost", "compute_cost", "total_observation_cost",
            "total_compute_cost", "program_changed", "model_changed", "decision_improvement_from_stopping", "trajectory_decision_improvements")},
            "evaluation": {key: value for key, value in arm["evaluation"].items() if key != "trajectory_results"},
            "behavior": behavior(arm)} for name, arm in task["arms"].items()}}
        for task in report["tasks"]]
    compact["mean_decision_loss"] = {name: mean(task["arms"][name]["evaluation"]["objective_loss"]
                                                    for task in report["tasks"]) for name in ARMS}
    compact["additional_total_costs"] = {name: {key: sum(task["arms"][name][key] for task in report["tasks"])
        for key in ("observation_cost", "measurement_cost", "compute_cost")} for name in ARMS}
    compact["source"] = str(output)
    with summary_path.open("x") as stream:
        json.dump(compact, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"summary": str(summary_path), "mean_decision_loss": compact["mean_decision_loss"],
                      "additional_total_costs": compact["additional_total_costs"]}), flush=True)


if __name__ == "__main__":
    main()
