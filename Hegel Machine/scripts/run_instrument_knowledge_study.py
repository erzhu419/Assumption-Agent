#!/usr/bin/env python3
"""Compare learned instrument representations through a hidden calibration change."""

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_session import UnifiedResearchSession
from run_unified_composition_study import SEEDS, configuration, freeze, tasks_for as composition_tasks, write


ARMS = ("default", "no_instrument_knowledge")
BASELINE = ROOT / "artifacts/unified_composition_20260913/comparison_revision3"


class WithoutInstrumentKnowledgeAgent(UnifiedResearchAgent):
    def _instrument_knowledge(self, laboratory):
        return False


def tasks_for(seed):
    tasks = composition_tasks(seed)
    apparatus = {key: tasks[1][key] for key in ("environment", "objects", "variant", "apparatus_seed")}
    for readout, prefix in (("total_event", ""), ("total_event", "recalibrated_"),
                            ("window_total_event", "")):
        tasks.append({**apparatus, "readout": readout, "rename_prefix": prefix,
                      "actuator_curvature": .31, "seed": seed + 10007 * len(tasks)})
    return tasks


def prepare(output, *, seeds=None, arms=None, baseline=None):
    path = output / "protocol.json"
    saved = json.loads(path.read_text()) if path.exists() else None
    seeds = list(seeds or (saved["seeds"] if saved else SEEDS))
    arms = list(arms or (saved["arms"] if saved else ARMS))
    baseline = Path(baseline or (saved["baseline"] if saved else BASELINE)).resolve()
    prefixes = {str(seed): str(baseline if baseline.is_file() else baseline / "runs" / f"{seed}_default.json")
                for seed in seeds}
    protocol = {
        "experiment": "instrument_knowledge_and_calibration_change",
        "seeds": seeds, "arms": arms, "baseline": str(baseline), "prefixes": prefixes,
        "tasks_by_seed": {str(seed): tasks_for(seed) for seed in seeds},
        "research_config": asdict(configuration()), "measurement_budget": 4608,
        "evaluation": {"length": 64, "trajectories": 4, "seed_offset": 1000003},
        "deterministic_solved_threshold": 1e-6,
        "ablation": "A local researcher subclass disables _instrument_knowledge for both fitting "
            "and command-based recall; all other research methods remain available.",
        "prefix_reuse": "The control arm reuses the unchanged original eight task results, "
            "knowledge and independent evaluation. It runs all three changed-calibration tasks. "
            "The default arm learns the complete eleven-task sequence from an empty library.",
        "calibration_change": "The physical apparatus keeps its interactions and readouts while "
            "the quantity actuator calibration changes. No calibration parameter, readout name, "
            "or hidden mechanism is supplied in learner inputs.",
        "evaluation_contract": "Every newly executed task is independently evaluated only after "
            "the entire learning sequence completes. Frozen prefix scores remain unchanged.",
        "code_control": "No new ordinary Code calls. Existing Code evidence is confined to the "
            "original eight matching tasks and provides no changed-calibration comparison.",
        "planned_logical_tasks": len(seeds) * len(arms) * 11,
        "planned_new_tasks": len(seeds) * sum(11 if arm == "default" else 3 for arm in arms),
    }
    if saved is not None and saved != protocol:
        raise ValueError("protocol changed; use a new output directory")
    if saved is None:
        write(path, protocol)
    return protocol


def resumed_prefix(path, tasks):
    state = deepcopy(json.loads(path.read_text()))
    if state["tasks"] != tasks[:8] or len(state["results"]) != 8 or len(state["evaluation"]) != 8:
        raise ValueError("control prefix must contain all eight identical completed and evaluated tasks")
    state["prefix_reuse"] = {"from": str(path.resolve()), "tasks": 8,
        "learning_costs": dict(state["costs"]),
        "evaluation_rows": sum(row["observation_cost"] for row in state["evaluation"])}
    state["tasks"] = tasks
    state["status"] = "paused"
    state["pending_tasks"] = [8, 9, 10]
    return state


def research(output, protocol, *, seed=None, arm=None, max_tasks=None):
    freeze(output, "research")
    frozen_script = output / "research_source" / Path(__file__).name
    if frozen_script.exists() and frozen_script.read_bytes() != Path(__file__).read_bytes():
        raise ValueError("instrument study script changed during the cohort")
    if not frozen_script.exists():
        shutil.copyfile(__file__, frozen_script)
    for chosen_seed in protocol["seeds"]:
        if seed is not None and seed != chosen_seed:
            continue
        for name in protocol["arms"]:
            if arm is not None and arm != name:
                continue
            path = output / "runs" / f"{chosen_seed}_{name}.json"
            state = json.loads(path.read_text()) if path.exists() else None
            tasks = protocol["tasks_by_seed"][str(chosen_seed)]
            if state and len(state["evaluation"]) == len(tasks):
                continue
            if state is None and name == "no_instrument_knowledge":
                state = resumed_prefix(Path(protocol["prefixes"][str(chosen_seed)]), tasks)
            session = UnifiedResearchSession(ResearchConfig(**{**protocol["research_config"],
                "seed": chosen_seed}), tasks=tasks, measurement_budget=protocol["measurement_budget"], state=state)

            def checkpoint(report):
                write(path, report)
                last = report["results"][-1]["research"]
                print(json.dumps({"seed": chosen_seed, "arm": name,
                    "completed_tasks": len(report["results"]), "kind": last["selected_kind"],
                    "selected_channels": last["selected_channels"], "costs": report["costs"]}), flush=True)

            researcher = UnifiedResearchAgent if name == "default" else WithoutInstrumentKnowledgeAgent
            with patch("hegel_machine.cognition.unified_research.UnifiedResearchAgent", researcher):
                session.run(max_tasks=max_tasks, on_checkpoint=checkpoint)
            if len(session.state["results"]) == len(tasks):
                checkpoint(session.evaluate(**protocol["evaluation"]))


def metrics(entries, threshold):
    return {"tasks": len(entries), "evaluated": sum(item["loss"] is not None for item in entries),
        "solved": sum(item["loss"] is not None and item["loss"] <= threshold for item in entries),
        "statuses": dict(Counter(item["status"] for item in entries)),
        "selected_channels": dict(Counter(",".join(item["channels"]) or "commands" for item in entries)),
        "learning_costs": {key: sum(item["costs"][key] for item in entries)
                           for key in ("observation_cost", "measurement_cost", "compute_cost")},
        "terminal_evaluation_costs": {
            key: sum(item.get("terminal_evaluation_costs", {}).get(key, 0) for item in entries)
            for key in ("observation_cost", "measurement_cost", "compute_cost")}}


def ordinary_code_reference(protocol):
    """Reference completed matching tasks; never run or extrapolate the Code control."""
    baseline = Path(protocol["baseline"])
    references = []
    for seed in protocol["seeds"]:
        path = baseline / "runs" / f"{seed}_ordinary_code.json"
        if not path.is_file():
            continue
        report = json.loads(path.read_text())
        if report["tasks"] != protocol["tasks_by_seed"][str(seed)][:8]:
            raise ValueError("ordinary Code reference does not match the original eight tasks")
        scores = report["evaluation"]
        reference = {"from": str(path.resolve()), "seed": seed, "tasks": 8,
            "primary_solved": sum(row.get("loss") is not None and
                row["loss"] <= protocol["deterministic_solved_threshold"] for row in scores),
            "statuses": dict(Counter(row["status"] for row in report["results"])),
            "learning_costs": report["costs"],
            "terminal_evaluation_costs": {key: sum(row.get(key, 0) for row in scores)
                for key in ("observation_cost", "measurement_cost", "compute_cost")},
            "scope": "Existing original-eight evidence only; no changed-calibration Code tasks were run."}
        secondary_path = baseline.parent / "comparison" / "code_comparison_results.json"
        if secondary_path.is_file():
            secondary = json.loads(secondary_path.read_text())
            if secondary["code_seed"] == seed:
                reference["secondary_authored_models"] = {
                    "from": str(secondary_path.resolve()),
                    **secondary["secondary_authored_models"]}
                reference["authored_programs_passing_primary_or_secondary"] = (
                    reference["primary_solved"] + secondary["secondary_authored_models"]["solved"])
                reference["delivery_interpretation"] = (
                    "Primary delivery passed five tasks. The other three already had budget-written "
                    "programs, which passed unchanged secondary execution with no new model calls. "
                    "This is eight passing authored programs, not eight primary deliveries.")
        references.append(reference)
    return references


def summarize(output, protocol):
    runs = []
    for path in sorted((output / "runs").glob("*.json")):
        report = json.loads(path.read_text())
        seed, arm = path.stem.split("_", 1)
        scores = {row["task_position"]: row for row in report["evaluation"]}
        reused = report.get("prefix_reuse", {}).get("tasks", 0)
        rows = [{"position": index, "status": entry["research"]["status"],
            "loss": scores.get(index, {}).get("loss"), "costs": entry["research"]["costs"],
            "terminal_evaluation_costs": {key: scores.get(index, {}).get(key, 0)
                for key in ("observation_cost", "measurement_cost", "compute_cost")},
            "channels": entry["research"]["selected_channels"],
            "kind": entry["research"]["selected_kind"],
            "used_sources": entry["research"].get("used_sources", []),
            "instrument_actions": [event for event in entry["research"]["actions"]
                                   if "instrument" in event["action"] or "calibrat" in event["action"]]}
            for index, entry in enumerate(report["results"])]
        runs.append({"seed": int(seed), "arm": arm, "tasks": rows,
            "prefix_reuse": report.get("prefix_reuse"), "reused_tasks": reused,
            "completed": len(scores) == 11,
            "new_learning_costs": metrics(rows[reused:], protocol["deterministic_solved_threshold"])["learning_costs"],
            "new_evaluation_rows": sum(row["observation_cost"] for index, row in scores.items() if index >= reused),
            "new_terminal_evaluation_costs": {
                key: sum(row.get(key, 0) for index, row in scores.items() if index >= reused)
                for key in ("observation_cost", "measurement_cost", "compute_cost")}})
    threshold = protocol["deterministic_solved_threshold"]
    arms = {}
    for arm in protocol["arms"]:
        rows = [row for run in runs if run["arm"] == arm for row in run["tasks"]]
        arms[arm] = {"all": metrics(rows, threshold),
            "original_eight": metrics([row for row in rows if row["position"] < 8], threshold),
            "changed_calibration": metrics([row for row in rows if row["position"] >= 8], threshold),
            "changed_successors": metrics([row for row in rows if row["position"] >= 9], threshold)}
    comparisons = {}
    if all(arm in arms for arm in ARMS):
        for subset in ("all", "original_eight", "changed_calibration", "changed_successors"):
            current, control = (arms[arm][subset] for arm in ARMS)
            comparisons[subset] = {
                category: {key: 100 * (current[category][key] / value - 1) if value else None
                    for key, value in control[category].items()}
                for category in ("learning_costs", "terminal_evaluation_costs")}
    summary = {"experiment": protocol["experiment"], "arms": arms, "runs": runs,
        "default_cost_change_percent_vs_no_instrument_knowledge": comparisons,
        "ordinary_code_references": ordinary_code_reference(protocol),
        "completed_sequences": sum(run["completed"] for run in runs),
        "planned_sequences": len(protocol["seeds"]) * len(protocol["arms"]),
        "actual_new_learning_costs": {key: sum(run["new_learning_costs"][key] for run in runs)
            for key in ("observation_cost", "measurement_cost", "compute_cost")},
        "actual_new_evaluation_rows": sum(run["new_evaluation_rows"] for run in runs),
        "actual_new_terminal_evaluation_costs": {
            key: sum(run["new_terminal_evaluation_costs"][key] for run in runs)
            for key in ("observation_cost", "measurement_cost", "compute_cost")}}
    write(output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "runs"}, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="Original comparison directory, or one completed development JSON")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--arms", choices=ARMS, nargs="+")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--max-tasks", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    protocol = prepare(args.output, seeds=args.seeds, arms=args.arms, baseline=args.baseline)
    if args.summary_only:
        summarize(args.output, protocol)
    elif not args.prepare_only:
        research(args.output, protocol, seed=args.seed, arm=args.arm, max_tasks=args.max_tasks)


if __name__ == "__main__":
    main()
