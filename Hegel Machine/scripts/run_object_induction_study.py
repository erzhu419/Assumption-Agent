#!/usr/bin/env python3
"""Learn anonymous object partitions, reuse them, and test a changed wiring."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.structure_lab import ObjectSchema
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_session import UnifiedResearchSession
from run_instrument_knowledge_study import metrics
from run_unified_composition_study import SEEDS, freeze, write


ARMS = ("default", "no_object_induction")
COST_KEYS = ("observation_cost", "measurement_cost", "compute_cost")
PHASES = {"all": (0, 9), "first_identification": (0, 1), "original_successors": (1, 5),
          "wiring_change": (5, 7), "renamed": (7, 9)}


class WithoutObjectInductionAgent(UnifiedResearchAgent):
    def _object_induction(self, laboratory):
        return False


def configuration(seed=71):
    return ResearchConfig(seed=seed, observation_budget=512, compute_budget=480000,
        minimum_improvement=1e-7, max_rounds=6, active_probes=True)


def tasks_for(seed):
    plant, count = dict(zip(SEEDS, ((10037, 3), (11047, 4), (12049, 3)))).get(seed, (9013, 3))
    tasks = []
    for index, readout in enumerate(("focus", "total", "event", "total_event", "window_total_event",
                                    "focus", "window_total_event", "focus", "window_total_event")):
        tasks.append({"environment": "anonymous_objects", "objects": count, "apparatus_seed": plant,
            "variant": "exchange", "readout": readout, "seed": seed + 10007 * index,
            "name_seed": seed + (53047 if index >= 7 else 42043),
            "wiring_seed": seed + (2609 if index >= 5 else 1709)})
    return tasks


def prepare(output, *, seeds=None, arms=None, task_count=None):
    path = output / "protocol.json"
    saved = json.loads(path.read_text()) if path.exists() else None
    seeds = list(seeds or (saved["seeds"] if saved else SEEDS))
    arms = list(arms or (saved["arms"] if saved else ARMS))
    task_count = task_count or (saved["positive_task_count"] if saved else 9)
    tasks = {str(seed): tasks_for(seed)[:task_count] for seed in seeds}
    protocol = {"experiment": "anonymous_object_induction_v065", "seeds": seeds, "arms": arms,
        "positive_task_count": task_count,
        "tasks_by_seed": tasks,
        "negative_tasks_by_seed": {str(seed): [{**tasks[str(seed)][0], "readout": "total",
            "seed": seed + 900007}] for seed in seeds},
        "research_config": asdict(configuration()), "measurement_budget": 6144,
        "evaluation": {"length": 64, "trajectories": 4, "seed_offset": 1000003},
        "deterministic_solved_threshold": 1e-6,
        "learning_contract": "Both arms begin with empty knowledge and share the same anonymous "
            "control bounds, sensor banks, transitions, observations and budgets. The control disables "
            "only _object_induction; no schema is supplied to either arm.",
        "public_contract": "Object counts, roles, membership, name and wiring seeds, readouts and "
            "apparatus parameters are evaluator-only settings. Public bounds define the controllable "
            "domain; their widths are not object-role labels. No private bank projection is available.",
        "wiring_contract": "Position 5 changes control and sensor wiring while preserving exactly "
            "the public names, bounds and order. Position 7 changes the anonymous names. Both "
            "changes are followed by a window composition on that same apparatus.",
        "evaluation_contract": "The complete learning stream finishes before independent prediction "
            "and object recovery scoring. Object recovery uses each task's actual learned_object_schema "
            "snapshot, never a later graph for an earlier task. Private evaluator schema is not feedback.",
        "negative_contract": "A separate empty-library total-only task must retain uncertainty in "
            "focus membership. It is run only for the default arm and never joins positive-task coverage.",
        "code_control": "No new ordinary Code calls. Prior schema-provided Code tasks do not form "
            "a matching anonymous-object comparison.",
        "planned_positive_tasks": len(seeds) * len(arms) * task_count,
        "planned_separate_negative_tasks": len(seeds)}
    if saved is not None and saved != protocol:
        raise ValueError("protocol changed; use a new output directory")
    if saved is None:
        write(path, protocol)
    return protocol


def object_recovery(inferred, truth):
    """Score acquired role/group hypotheses, disregarding arbitrary object IDs."""
    proposed = ObjectSchema.from_dict(inferred) if isinstance(inferred, dict) else inferred
    by_quantity = {truth.fields[obj]["q"]: obj for obj in truth.objects}
    count = len(truth.objects)
    result = {"object_count": count, "proposed_objects": len(proposed.objects) if proposed else 0,
        "complete_objects": 0, "correct_measurement_roles": 0, "correct_control_roles": 0,
        "correct_focus_pairs": 0, "focus_control_correct": bool(proposed and
            proposed.focus_control == truth.focus_control), "exact": False}
    if proposed:
        quantities = [proposed.fields[obj].get("q") for obj in proposed.objects]
        for obj, quantity in zip(proposed.objects, quantities):
            if quantity not in by_quantity or quantities.count(quantity) != 1:
                continue
            actual = by_quantity[quantity]
            fields = sum(proposed.fields[obj].get(role) == truth.fields[actual][role] for role in ("q", "x", "u", "f"))
            controls = sum(proposed.controls.get(obj, {}).get(role) == truth.controls[actual][role]
                           for role in ("q", "x", "u"))
            result["correct_measurement_roles"] += fields
            result["correct_control_roles"] += controls
            result["correct_focus_pairs"] += proposed.fields[obj].get("f") == truth.fields[actual]["f"]
            result["complete_objects"] += fields == 4 and controls == 3
        result["exact"] = (result["complete_objects"] == count == len(proposed.objects)
                           and result["focus_control_correct"])
    return result


def evaluate_objects(report):
    scores = []
    for task, entry in zip(report["tasks"], report["results"]):
        arguments = {key: value for key, value in task.items() if key != "environment"}
        with AnonymousObjectLab(**arguments) as lab:
            score = object_recovery(entry["research"].get("learned_object_schema"), lab._evaluation_schema())
        scores.append({"task_position": entry["task_position"], **score})
    report["object_evaluation"] = scores
    return report


def research(output, protocol, *, seed=None, arm=None, max_tasks=None, negative=False):
    freeze(output, "research")
    for source in (Path(__file__), ROOT / "scripts/run_instrument_knowledge_study.py"):
        frozen = output / "research_source" / source.name
        if frozen.exists() and frozen.read_bytes() != source.read_bytes():
            raise ValueError("study script changed during the cohort")
        if not frozen.exists():
            shutil.copyfile(source, frozen)
    for chosen_seed in protocol["seeds"]:
        if seed is not None and seed != chosen_seed:
            continue
        names = ("default",) if negative else protocol["arms"]
        for name in names:
            if arm is not None and name != arm:
                continue
            path = output / ("negative_runs" if negative else "runs") / f"{chosen_seed}_{name}.json"
            state = json.loads(path.read_text()) if path.exists() else None
            tasks = protocol["negative_tasks_by_seed" if negative else "tasks_by_seed"][str(chosen_seed)]
            if state and len(state.get("object_evaluation", [])) == len(tasks):
                continue
            session = UnifiedResearchSession(ResearchConfig(**{**protocol["research_config"], "seed": chosen_seed}),
                tasks=tasks, measurement_budget=protocol["measurement_budget"], state=state)

            def checkpoint(report):
                write(path, report)
                last = report["results"][-1]["research"]
                print(json.dumps({"seed": chosen_seed, "arm": name, "negative": negative,
                    "completed_tasks": len(report["results"]), "kind": last["selected_kind"],
                    "selected_channels": last["selected_channels"], "costs": report["costs"],
                    "object_model_id": last.get("object_model_id"),
                    "complete_object_schema": last.get("learned_object_schema") is not None}), flush=True)

            researcher = UnifiedResearchAgent if name == "default" else WithoutObjectInductionAgent
            with patch("hegel_machine.cognition.unified_research.UnifiedResearchAgent", researcher):
                session.run(max_tasks=max_tasks, on_checkpoint=checkpoint)
            if len(session.state["results"]) == len(tasks):
                checkpoint(evaluate_objects(session.evaluate(**protocol["evaluation"])))


def summarize(output, protocol):
    runs, negatives = [], []
    for directory, destination in (("runs", runs), ("negative_runs", negatives)):
        for path in sorted((output / directory).glob("*.json")):
            report = json.loads(path.read_text())
            seed, arm = path.stem.split("_", 1)
            scores = {row["task_position"]: row for row in report["evaluation"]}
            objects = {row["task_position"]: row for row in report.get("object_evaluation", [])}
            rows = []
            for index, entry in enumerate(report["results"]):
                result = entry["research"]
                rows.append({"position": index, "status": result["status"],
                    "loss": scores.get(index, {}).get("loss"), "costs": result["costs"],
                    "terminal_evaluation_costs": {key: scores.get(index, {}).get(key, 0) for key in COST_KEYS},
                    "channels": result["selected_channels"], "kind": result["selected_kind"],
                    "used_sources": result.get("used_sources", []), "object_model_id": result.get("object_model_id"),
                    "object_recovery": objects.get(index), "object_research": result.get("object_research"),
                    "object_actions": [event for event in result["actions"] if "object" in event["action"]]})
            destination.append({"seed": int(seed), "arm": arm, "tasks": rows,
                "completed": len(objects) == len(report["tasks"])})
    threshold = protocol["deterministic_solved_threshold"]
    arms = {}
    for arm in protocol["arms"]:
        rows = [row for run in runs if run["arm"] == arm for row in run["tasks"]]
        arms[arm] = {}
        for phase, (start, stop) in PHASES.items():
            selected = [row for row in rows if start <= row["position"] < stop]
            arms[arm][phase] = {**metrics(selected, threshold),
                "exact_object_recoveries": sum(bool(row["object_recovery"] and row["object_recovery"]["exact"])
                                               for row in selected)}
    summary = {"experiment": protocol["experiment"], "arms": arms, "runs": runs,
        "negative_runs": negatives, "completed_sequences": sum(run["completed"] for run in runs),
        "planned_sequences": len(protocol["seeds"]) * len(protocol["arms"])}
    write(output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key not in ("runs", "negative_runs")}, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--arms", choices=ARMS, nargs="+")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--task-count", type=int, choices=range(1, 10),
                        help="Declare a shorter development prefix; default is the full nine tasks")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--summary-only", action="store_true")
    mode.add_argument("--negative-only", action="store_true")
    args = parser.parse_args()
    protocol = prepare(args.output, seeds=args.seeds, arms=args.arms, task_count=args.task_count)
    if args.summary_only:
        summarize(args.output, protocol)
    elif not args.prepare_only:
        research(args.output, protocol, seed=args.seed, arm=args.arm, max_tasks=args.max_tasks,
                 negative=args.negative_only)


if __name__ == "__main__":
    main()
