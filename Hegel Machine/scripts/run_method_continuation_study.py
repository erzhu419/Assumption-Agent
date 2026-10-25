#!/usr/bin/env python3
"""Measure the return of one more research visit from a saved learning state."""

from collections import Counter
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict, replace
import argparse
import json
from pathlib import Path
from random import Random
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hegel_machine.cognition.checkpoint_storage import write_text_incrementally

from hegel_machine.cognition.controller import CognitiveAgent
from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.evaluation import decision_metrics, loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Probe, ResearchConfig
from hegel_machine.cognition.research_methods import ResearchMethods


@contextmanager
def without_method_selection():
    """Offline ablation: preserve other policy learning and past commitments."""
    choose, opportunity = ResearchMethods.choose, ResearchMethods.opportunity
    ResearchMethods.choose = lambda *_args, **_kwargs: None
    ResearchMethods.opportunity = lambda *_args, **_kwargs: None
    try:
        yield
    finally:
        ResearchMethods.choose, ResearchMethods.opportunity = choose, opportunity


def _sources(checkpoint):
    seed = checkpoint["config"]["seed"]
    sources = [(partition, index, environment.spec.task_id)
               for partition in checkpoint.get("partitions", [checkpoint["config"]["partition"]])
               for index, environment in enumerate(make_suite(seed, partition=partition))]
    if [item[2] for item in sources] != checkpoint["task_ids"]:
        raise ValueError("checkpoint task IDs do not match the current public task suite")
    return sources


def _new_evaluation_rows(environment, seed, trajectories):
    rng = Random(seed)
    spec, rows = environment.spec, []
    for index in range(trajectories):
        probe = Probe(f"continuation_evaluation_{index}", tuple(
            {name: rng.uniform(low, high) for name, (low, high) in spec.feature_bounds.items()}
            for _ in range(spec.max_probe_length)))
        rows.extend(environment.observe(probe))
    return rows


def _measure(program, fallback, model, rows, spec):
    prediction_loss = loss(program, rows, fallback, spec.risk_quantile)
    result = {"prediction_loss": prediction_loss, "objective_loss": prediction_loss,
              "loss_kind": "squared_error" if spec.risk_quantile == .5 else f"pinball@{spec.risk_quantile:g}"}
    if spec.decision_costs is not None:
        metrics = decision_metrics(program, rows, fallback, model, spec.decision_costs)
        result.update(decision_metrics=metrics, objective_loss=metrics["mean_decision_loss"],
                      loss_kind=f"decision@{spec.risk_quantile:g}")
    return result


def _pending(workspace, library):
    trace = workspace.get("method_trace", [])
    unfinished = []
    for method in library.methods.methods:
        completed = [step["action"] for step in trace if step.get("method_id") == method["method_id"]]
        if completed and completed == method["steps"][:len(completed)] and len(completed) < len(method["steps"]):
            unfinished.append({"method_id": method["method_id"], "completed": completed,
                               "remaining": method["steps"][len(completed):]})
    return {"actions_awaiting_feedback": deepcopy(workspace.get("pending_actions", [])),
            "requested_method": deepcopy(workspace.get("requested_method", {})),
            "unfinished_methods": unfinished}


def _action_trace(actions):
    detailed = {"execute_method_step", "plan_experiment", "action_feedback", "continuation_feedback",
                "compare", "resume_research", "observe", "repair_startup", "compare_calibration", "revise_uncertainty"}
    return [{"round": action.round, "action": action.action,
             "observation_cost": action.observation_cost, "compute_cost": action.compute_cost,
             **({"details": deepcopy(action.details)} if action.action in detailed else {})} for action in actions]


def _change(before, after):
    improvement = before - after
    tolerance = max(1e-9, abs(before) * 1e-6)
    return {"improvement": improvement,
            "relative_improvement": improvement / max(abs(before), 1e-9),
            "outcome": "improved" if improvement > tolerance else "worsened" if improvement < -tolerance else "unchanged"}


def run_study(checkpoint, *, tasks=None, seed=None, observation_budget=None, compute_budget=None,
              rounds=None, evaluation_trajectories=16, on_result=None):
    """Fork each saved workspace independently; evaluate only after both visits."""
    if evaluation_trajectories < 1:
        raise ValueError("evaluation trajectories must be positive")
    sources = _sources(checkpoint)
    selected = list(tasks) if tasks is not None else [task["task_index"] for task in checkpoint["tasks"]
                if task["unresolved"] and str(task["task_index"]) in checkpoint["workspaces"]]
    if any(index < 0 or index >= len(sources) or str(index) not in checkpoint["workspaces"] for index in selected):
        raise ValueError("continuation tasks must have an existing paid workspace")
    values = dict(checkpoint["config"])
    for key, value in (("observation_budget", observation_budget), ("compute_budget", compute_budget), ("max_rounds", rounds)):
        if value is not None:
            values[key] = value
    config = ResearchConfig(**values)
    if config.proposer != "search":
        raise ValueError("this offline continuation study requires the search proposer; it does not call external models")
    search_seed = config.seed if seed is None else seed
    report = {
        "experiment": "paired_method_continuation_return", "status": "running", "config": asdict(config),
        "source": {"status": checkpoint["status"], "completed_visits": checkpoint["completed_visits"],
                   "observation_cost": checkpoint["total_observation_cost"], "compute_cost": checkpoint["total_compute_cost"],
                   "session_visit_limit": checkpoint["limits"]["max_visits"]},
        "selection": "explicit task indices" if tasks is not None else "visited unresolved tasks in the saved learning state",
        "task_indices": selected, "search_seed": search_seed, "evaluation_trajectories": evaluation_trajectories,
        "intervention": "Only ResearchMethods.choose and opportunity return None in the ordinary arm; action policy, plan experience, executable knowledge, method feedback and saved commitments remain available.",
        "continuation": "Each task and arm starts independently from the same checkpoint knowledge and paid workspace; exactly one further controller visit is allowed regardless of the old session visit limit.",
        "evaluation": "New public control trajectories are acquired by a separate evaluator only after both research arms stop; all three saved programs are scored on those identical rows.",
        "tasks": [],
    }
    for task_index in selected:
        partition, local_index, task_id = sources[task_index]
        task_state = checkpoint["tasks"][task_index]
        saved_workspace = checkpoint["workspaces"][str(task_index)]
        visit_seed = search_seed + task_index * 10007 + task_state["visits"] * 1000003
        visit_config = replace(config, seed=visit_seed)
        baseline_program = Program.from_dict(saved_workspace["incumbent"]["program"])
        baseline_fallback = saved_workspace["fallback"]
        baseline_model = saved_workspace.get("uncertainty_model", {})
        result = {"task_index": task_index, "task_id": task_id, "partition": partition,
                  "source_visits": task_state["visits"], "seed": visit_seed,
                  "saved_observations": len(saved_workspace["observations"]), "arms": {}}
        evaluated = {"no_additional": (baseline_program, baseline_fallback, baseline_model)}
        for arm in ("methods", "ordinary"):
            library = KnowledgeLibrary.from_dict(checkpoint["knowledge"])
            workspace = deepcopy(saved_workspace)
            environment = make_suite(checkpoint["config"]["seed"], partition=partition)[local_index]
            context = without_method_selection() if arm == "ordinary" else nullcontext()
            with context:
                episode = CognitiveAgent(visit_config, library).run(environment, research_state=workspace)
            after_workspace = episode.research_state
            result["arms"][arm] = {
                "program": episode.program.to_dict(), "expression": episode.program.describe(),
                "fallback": episode.fallback, "uncertainty_model": deepcopy(episode.uncertainty_model),
                "validation_loss": episode.validation_loss, "stop_reason": episode.stop_reason,
                "observation_cost": episode.observation_cost, "compute_cost": episode.compute_cost,
                "wall_time_seconds": episode.wall_time_seconds,
                "observations_after": len(after_workspace["observations"]),
                "actions": _action_trace(episode.actions), "pending": _pending(after_workspace, library),
                "new_method_feedback": deepcopy(after_workspace.get("method_trace", [])[len(saved_workspace.get("method_trace", [])):]),
                "method_choices": [deepcopy(action.details["learned_method"]) for action in episode.actions
                                   if action.details.get("learned_method")],
            }
            evaluated[arm] = episode.program, episode.fallback, episode.uncertainty_model
        # No held-out response is read until both policies have finished.
        evaluator = make_suite(checkpoint["config"]["seed"], partition=partition)[local_index]
        evaluation_seed = visit_seed + 2000003
        rows = _new_evaluation_rows(evaluator, evaluation_seed, evaluation_trajectories)
        result["arms"]["no_additional"] = {
            "program": baseline_program.to_dict(), "expression": baseline_program.describe(),
            "fallback": baseline_fallback, "uncertainty_model": deepcopy(baseline_model),
            "observation_cost": 0, "compute_cost": 0, "wall_time_seconds": 0.,
            "observations_after": len(saved_workspace["observations"]), "actions": [],
            "pending": _pending(saved_workspace, KnowledgeLibrary.from_dict(checkpoint["knowledge"])),
        }
        for arm, (program, fallback, model) in evaluated.items():
            result["arms"][arm]["evaluation"] = _measure(program, fallback, model, rows, evaluator.spec)
        before = result["arms"]["no_additional"]["evaluation"]["objective_loss"]
        for arm in ("methods", "ordinary"):
            result["arms"][arm]["change_from_saved"] = _change(before, result["arms"][arm]["evaluation"]["objective_loss"])
        result["method_advantage"] = _change(result["arms"]["ordinary"]["evaluation"]["objective_loss"],
                                             result["arms"]["methods"]["evaluation"]["objective_loss"])
        result["evaluation_costs"] = {"seed": evaluation_seed, "observation_rows": len(rows),
                                      "candidate_row_compute": 3 * len(rows) * (2 if evaluator.spec.decision_costs is not None else 1)}
        report["tasks"].append(result)
        report["summary"] = summarize(report)
        if on_result is not None:
            on_result(deepcopy(report))
    report["status"] = "completed"
    report["summary"] = summarize(report)
    return report


def summarize(report):
    tasks = report["tasks"]
    threshold = report["config"]["minimum_improvement"]

    def material_outcome(improvement):
        return "improved" if improvement > threshold else "worsened" if improvement < -threshold else "unchanged"

    kinds = sorted({task["arms"]["no_additional"]["evaluation"]["loss_kind"] for task in tasks})
    summary = {"tasks": len(tasks), "arms": {}, "by_objective": {},
               "method_advantage_outcomes": dict(Counter(task["method_advantage"]["outcome"] for task in tasks)),
               "material_improvement_threshold": threshold,
               "method_advantage_material_outcomes": dict(Counter(material_outcome(task["method_advantage"]["improvement"])
                                                                  for task in tasks)),
               "evaluation_observation_rows": sum(task["evaluation_costs"]["observation_rows"] for task in tasks),
               "evaluation_candidate_row_compute": sum(task["evaluation_costs"]["candidate_row_compute"] for task in tasks)}
    for arm in ("methods", "ordinary", "no_additional"):
        records = [task["arms"][arm] for task in tasks]
        summary["arms"][arm] = {"observation_cost": sum(row["observation_cost"] for row in records),
                                "compute_cost": sum(row["compute_cost"] for row in records),
                                "outcomes": dict(Counter(row["change_from_saved"]["outcome"] for row in records if "change_from_saved" in row)),
                                "material_outcomes": dict(Counter(material_outcome(row["change_from_saved"]["improvement"])
                                                                   for row in records if "change_from_saved" in row)),
                                "method_choices": sum(len(row.get("method_choices", [])) for row in records)}
    for kind in kinds:
        chosen = [task for task in tasks if task["arms"]["no_additional"]["evaluation"]["loss_kind"] == kind]
        summary["by_objective"][kind] = {"tasks": len(chosen),
            **{arm: {"mean_objective_loss": mean(task["arms"][arm]["evaluation"]["objective_loss"] for task in chosen),
                     "mean_prediction_loss": mean(task["arms"][arm]["evaluation"]["prediction_loss"] for task in chosen)}
               for arm in ("methods", "ordinary", "no_additional")}}
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tasks", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--observation-budget", type=int)
    parser.add_argument("--compute-budget", "--budget", type=int, dest="compute_budget")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--evaluation-trajectories", type=int, default=16)
    args = parser.parse_args(argv)
    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(report):
        report["source"]["checkpoint"] = str(args.checkpoint.resolve())
        write_text_incrementally(args.output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    report = run_study(checkpoint, tasks=args.tasks, seed=args.seed, observation_budget=args.observation_budget,
                       compute_budget=args.compute_budget, rounds=args.rounds,
                       evaluation_trajectories=args.evaluation_trajectories, on_result=save)
    save(report)
    summary = {"status": report["status"], "source": report["source"], "task_indices": report["task_indices"], **report["summary"]}
    args.output.with_name(args.output.stem + "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
