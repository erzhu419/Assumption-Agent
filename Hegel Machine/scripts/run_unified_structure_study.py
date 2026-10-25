#!/usr/bin/env python3
"""Run one research controller and memory across scalar and object processes."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/hegel_machine/cognition"))
from checkpoint_storage import write_json_checkpoint
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_session import UnifiedResearchSession


SEEDS = (2949319099, 2949419113, 2949519131)
ARMS = {
    "default": {},
    "no_memory": {"use_memory": False},
    "passive_structure": {"active_probes": False},
    "fixed_library": {"strategy": "fixed", "use_memory": False},
}
TASKS = (
    {"task_index": 0, "profile": "development"},
    {"environment": "objects", "readout": "total"},
    {"environment": "objects", "readout": "focus"},
    {"task_index": 8, "profile": "development"},
    {"environment": "objects", "readout": "event"},
    {"environment": "objects", "readout": "window"},
    {"task_index": 12, "profile": "development"},
    {"environment": "objects", "readout": "total", "rename_prefix": "renamed_"},
    {"environment": "objects", "readout": "focus", "rename_prefix": "renamed_"},
    {"environment": "objects", "readout": "focus", "variant": "anchored"},
    {"environment": "objects", "readout": "focus", "variant": "nonlocal"},
    {"environment": "objects", "readout": "event", "rename_prefix": "later_"},
)


def write(path, payload):
    write_json_checkpoint(path, payload, indent=2, ensure_ascii=False, allow_nan=True)


def tasks_for(seed):
    return [{**({"objects": 4, "variant": "exchange"}
                if item.get("environment") == "objects" else {}),
             **item, "seed": seed + 10007 * position}
            for position, item in enumerate(TASKS)]


def configuration():
    return ResearchConfig(observation_budget=384, compute_budget=360000,
        minimum_improvement=1e-7, max_rounds=6, active_probes=True)


def make_protocol(seeds, arms):
    count = len(seeds) * len(arms) * len(TASKS)
    return {
        "experiment": "unified_object_relation_event_scale_research_v061",
        "seeds": list(seeds), "arms": {name: ARMS[name] for name in arms},
        "tasks_by_seed": {str(seed): tasks_for(seed) for seed in seeds},
        "research_config": asdict(configuration()), "measurement_budget": 4608,
        "evaluation": {"length": 64, "trajectories": 4, "seed_offset": 1000003},
        "deterministic_solved_threshold": 1e-6,
        "task_stream": "A scalar affine task, conserved object total, local object response, "
            "continuous state, event difference, event window, conditional risk, renamed total, "
            "renamed local response, anchored response, nonlocal response and renamed event. "
            "Every task has a new apparatus seed, including the recurring readouts, so renamed "
            "tasks also change physical coefficients and local interaction boundaries.",
        "learning_contract": "The driver chooses apparatus construction. Every arm runs the same "
            "UnifiedResearchAgent using the public bounds, measurement views and role schema; "
            "the learner is not given the readout, variant or private mechanism parameters. "
            "One knowledge library persists across the whole task stream when memory is enabled.",
        "evaluation_contract": "The entire learning sequence finishes before evaluation. Each "
            "fixed selected program is then scored on four independent 64-row trajectories "
            "through its selected measurement view. Evaluation rows never enter learning or "
            "memory. Only deterministic MSE tasks use the solved threshold; pinball loss is "
            "reported separately at the public quantile.",
        "endpoints": "Deterministic solved count and mean loss, risk pinball loss, observation, "
            "measurement and compute costs, selected views and program operators, confirmed "
            "source reuse, rejected transfers and actual boundary experiment actions. Object "
            "tasks and later successor tasks are also reported separately.",
        "scope": "Four algorithmic arms on the same new mixed cohort. This study makes no "
            "additional language-model calls and is not an ordinary Code Agent comparison.",
        "resume": "Each completed task and the terminal evaluation are checkpointed. Completed "
            "learning tasks are not repeated. Source snapshots keep a resumed cohort consistent.",
        "planned_learning_tasks": count,
        "maximum_learning_observations": count * 384,
        "maximum_learning_measurement_cost": count * 4608,
        "planned_evaluation_observations": count * 64 * 4,
    }


def prepare(output, *, seeds=None, arms=None):
    path = output / "protocol.json"
    saved = json.loads(path.read_text()) if path.exists() else None
    seeds = list(seeds or (saved["seeds"] if saved else SEEDS))
    arms = list(arms or (saved["arms"] if saved else ARMS))
    if len(set(seeds)) != len(seeds) or len(set(arms)) != len(arms):
        raise ValueError("seeds and arms must be unique")
    protocol = make_protocol(seeds, arms)
    if saved is not None and saved != protocol:
        raise ValueError("saved study protocol differs; use a new output directory")
    if saved is None:
        write(path, protocol)
    return protocol


def freeze(output):
    sources = list((ROOT / "src/hegel_machine/cognition").glob("*.py")) + [Path(__file__).resolve()]
    for source in sources:
        target = output / "method_source" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != source.read_bytes():
                raise ValueError(f"source changed during this cohort: {source.name}; use a new output directory")
        else:
            shutil.copyfile(source, target)


def research(output, protocol, *, selected_seed=None, selected_arm=None, max_tasks=None):
    if selected_seed is not None and selected_seed not in protocol["seeds"]:
        raise ValueError("selected seed is not in the prepared cohort")
    if selected_arm is not None and selected_arm not in protocol["arms"]:
        raise ValueError("selected arm is not in the prepared cohort")
    freeze(output)
    for seed in protocol["seeds"]:
        if selected_seed is not None and seed != selected_seed:
            continue
        for arm, overrides in protocol["arms"].items():
            if selected_arm is not None and arm != selected_arm:
                continue
            path = output / "runs" / f"{seed}_{arm}.json"
            saved = json.loads(path.read_text()) if path.exists() else None
            tasks = protocol["tasks_by_seed"][str(seed)]
            if saved and len(saved["evaluation"]) == len(tasks):
                continue
            config = ResearchConfig(**{**protocol["research_config"], "seed": seed, **overrides})
            session = UnifiedResearchSession(config, tasks=tasks, state=saved,
                measurement_budget=protocol["measurement_budget"])

            def checkpoint(report):
                write(path, report)
                if report["results"]:
                    result = report["results"][-1]["research"]
                    print(json.dumps({"seed": seed, "arm": arm,
                        "completed_tasks": len(report["results"]),
                        "evaluated_tasks": len(report["evaluation"]),
                        "selected_kind": result["selected_kind"],
                        "status": result["status"], "costs": report["costs"]}), flush=True)

            session.run(max_tasks=max_tasks, on_checkpoint=checkpoint)
            if len(session.state["results"]) == len(tasks):
                checkpoint(session.evaluate(**protocol["evaluation"]))


def program_details(payload):
    counts = Counter()

    def visit(node):
        counts[node.op] += 1
        for child in node.children:
            visit(child)

    visit(Program.from_dict(payload).expanded())
    return dict(counts)


def metrics(tasks, threshold):
    evaluated = [task for task in tasks if task["evaluation"] is not None]
    losses = {}
    for kind in ("mse", "pinball"):
        rows = [task["evaluation"] for task in evaluated if task["evaluation"]["loss_kind"] == kind]
        if rows:
            losses[kind] = {"tasks": len(rows), "mean_loss": mean(row["loss"] for row in rows)}
            if kind == "mse":
                losses[kind]["solved"] = sum(row["loss"] <= threshold for row in rows)
    return {"tasks": len(tasks), "evaluated_tasks": len(evaluated), "losses": losses,
        "research_statuses": dict(Counter(task["status"] for task in tasks)),
        "selected_kinds": dict(Counter(task["selected_kind"] for task in tasks)),
        "selected_views": dict(Counter(",".join(task["selected_channels"]) or "controls_only" for task in tasks)),
        "structural_programs": sum(any(op in task["operators"] for op in
            ("object_sum", "neighbor_sum", "event_delta", "window_sum")) for task in tasks),
        "confirmed_source_uses": sum(bool(task["source_calls"]) for task in tasks),
        "rejected_transfers": sum(task["rejected_transfers"] for task in tasks),
        "boundary_experiments": sum(len(task["boundary_acquisitions"]) for task in tasks),
        "boundary_revisions": sum(len(task["boundary_actions"]) for task in tasks),
        "learning_costs": {key: sum(task["costs"][key] for task in tasks)
                           for key in ("observation_cost", "measurement_cost", "compute_cost")}}


def summarize(output, protocol):
    runs = []
    for seed in protocol["seeds"]:
        for arm in protocol["arms"]:
            path = output / "runs" / f"{seed}_{arm}.json"
            if not path.exists():
                continue
            report = json.loads(path.read_text())
            scores = {row["task_position"]: row for row in report["evaluation"]}
            tasks = []
            for position, row in enumerate(report["results"]):
                result = row["research"]
                tasks.append({"task_position": position, **report["tasks"][position],
                    "status": result["status"], "selected_kind": result["selected_kind"],
                    "selected_channels": result["selected_channels"],
                    "operators": program_details(result["program"]),
                    "source_calls": result["used_sources"],
                    "method_source_ids": result.get("method_source_ids", []),
                    "rejected_transfers": sum(event["status"] == "counterexample"
                        for event in result["transfer"]),
                    "boundary_actions": [action for action in result["actions"]
                        if action.get("action") == "revise_structure_boundary"],
                    "boundary_acquisitions": [action for action in result["actions"]
                        if action.get("action") == "acquire"
                        and action.get("purpose") == "resolve_structure_boundary"],
                    "costs": result["costs"], "evaluation": scores.get(position)})
            runs.append({"seed": seed, "arm": arm, "status": report["status"], "tasks": tasks,
                "costs": report["costs"], "evaluation_costs": {
                    key: sum(row[key] for row in report["evaluation"])
                    for key in ("observation_cost", "measurement_cost", "compute_cost")}})
    threshold = protocol["deterministic_solved_threshold"]
    arms = {}
    for arm in protocol["arms"]:
        tasks = [task for run in runs if run["arm"] == arm for task in run["tasks"]]
        arms[arm] = {"all_tasks": metrics(tasks, threshold),
            "object_tasks": metrics([task for task in tasks if task.get("environment") == "objects"], threshold),
            "successor_tasks": metrics([task for task in tasks if task["task_position"] >= 7], threshold)}
    summary = {"experiment": protocol["experiment"], "arms": arms, "runs": runs,
        "planned_sequences": len(protocol["seeds"]) * len(protocol["arms"]),
        "completed_sequences": sum(len(run["tasks"]) == len(TASKS)
            and all(task["evaluation"] is not None for task in run["tasks"]) for run in runs),
        "actual_learning_observations": sum(run["costs"]["observation_cost"] for run in runs),
        "actual_learning_measurement_cost": sum(run["costs"]["measurement_cost"] for run in runs),
        "actual_learning_compute_cost": sum(run["costs"]["compute_cost"] for run in runs),
        "actual_evaluation_observations": sum(run["evaluation_costs"]["observation_cost"] for run in runs),
        "actual_evaluation_measurement_cost": sum(run["evaluation_costs"]["measurement_cost"] for run in runs)}
    write(output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "runs"}, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--arms", choices=tuple(ARMS), nargs="+")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--summary-only", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--max-tasks", type=int)
    arguments = parser.parse_args()
    if arguments.summary_only:
        protocol = json.loads((arguments.output / "protocol.json").read_text())
        summarize(arguments.output, protocol)
        return
    protocol = prepare(arguments.output, seeds=arguments.seeds, arms=arguments.arms)
    if arguments.prepare_only:
        print(json.dumps({"protocol": str(arguments.output / "protocol.json"),
            "planned_learning_tasks": protocol["planned_learning_tasks"],
            "planned_evaluation_observations": protocol["planned_evaluation_observations"]}), flush=True)
        return
    research(arguments.output, protocol, selected_seed=arguments.seed,
        selected_arm=arguments.arm, max_tasks=arguments.max_tasks)
    # A selected worker writes only its own runs. The coordinator summarizes
    # the shared cohort once, after the parallel seed workers have finished.
    if arguments.seed is None and arguments.arm is None:
        summarize(arguments.output, protocol)


if __name__ == "__main__":
    main()
