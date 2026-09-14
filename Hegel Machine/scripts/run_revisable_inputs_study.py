#!/usr/bin/env python3
"""Test local input revision against genuine and joint mechanism changes."""

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

from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_session import UnifiedResearchSession
from run_instrument_knowledge_study import metrics, tasks_for as instrument_tasks
from run_unified_composition_study import SEEDS, configuration, freeze, write


ARMS = ("default", "no_revisable_inputs")
BASELINE = ROOT / "artifacts/instrument_knowledge_20260913/comparison_v1"
COST_KEYS = ("observation_cost", "measurement_cost", "compute_cost")
PHASES = {"all": (0, 15), "original_eight": (0, 8), "calibration_only": (8, 11),
          "relation_only": (11, 13), "joint_change": (13, 15)}


class WithoutRevisableInputsAgent(UnifiedResearchAgent):
    def _revisable_inputs(self, laboratory):
        return False


def tasks_for(seed):
    tasks = instrument_tasks(seed)
    apparatus = {key: tasks[1][key] for key in ("environment", "objects", "apparatus_seed")}
    for variant, curvature, readout, prefix in (
        ("sink", .31, "total_event", ""),
        ("sink", .31, "window_total_event", "relation_"),
        ("nonlocal", .47, "total_event", ""),
        ("nonlocal", .47, "window_total_event", "joint_"),
    ):
        tasks.append({**apparatus, "variant": variant, "actuator_curvature": curvature,
            "readout": readout, "rename_prefix": prefix, "seed": seed + 10007 * len(tasks)})
    return tasks


def prepare(output, *, seeds=None, arms=None):
    path = output / "protocol.json"
    saved = json.loads(path.read_text()) if path.exists() else None
    seeds = list(seeds or (saved["seeds"] if saved else SEEDS))
    arms = list(arms or (saved["arms"] if saved else ARMS))
    protocol = {
        "experiment": "revisable_inputs_and_mechanism_change_v064",
        "seeds": seeds, "arms": arms,
        "historical_instrument_study": str(BASELINE / "summary.json"),
        "tasks_by_seed": {str(seed): tasks_for(seed) for seed in seeds},
        "research_config": asdict(configuration()), "measurement_budget": 4608,
        "evaluation": {"length": 64, "trajectories": 4, "seed_offset": 1000003},
        "deterministic_solved_threshold": 1e-6,
        "ablation": "A local subclass disables only _revisable_inputs; the previous instrument "
            "learning, command compilation and all other research remain enabled.",
        "learning_contract": "Both arms learn all fifteen tasks from an empty library using the "
            "same current researcher and residual language. The earlier v0.63 study remains historical "
            "evidence and supplies no learned prefixes to either arm.",
        "changes": "Positions 8-10 change calibration only. Positions 11-12 keep that calibration "
            "and add a quantity-dependent local sink. Positions 13-14 change both calibration "
            "and the interaction to a global quantity coupling. Each relationship change is followed "
            "by a renamed window composition task on that same changed apparatus.",
        "public_contract": "All object tasks share public controls, instrument descriptions and "
            "measurable roles. Variant labels, readouts and physical coefficients are evaluator-only "
            "task settings, absent from learner inputs.",
        "evaluation_contract": "The entire learning sequence finishes before four fresh 64-row "
            "trajectories per task are scored.",
        "endpoints": "Independent predictive coverage; learned measurement substitution; learning "
            "and terminal costs by phase; retained or revised callable sources and actual research actions.",
        "code_control": "No new Code calls; prior Code primary/secondary evidence covers only the "
            "matching original eight tasks, with no calibration or relation change comparison.",
        "planned_logical_tasks": len(seeds) * len(arms) * 15,
        "planned_new_tasks": len(seeds) * len(arms) * 15,
    }
    if saved is not None and saved != protocol:
        raise ValueError("protocol changed; use a new output directory")
    if saved is None:
        write(path, protocol)
    return protocol


def research(output, protocol, *, seed=None, arm=None, max_tasks=None):
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
        for name in protocol["arms"]:
            if arm is not None and arm != name:
                continue
            path = output / "runs" / f"{chosen_seed}_{name}.json"
            state = json.loads(path.read_text()) if path.exists() else None
            tasks = protocol["tasks_by_seed"][str(chosen_seed)]
            if state and len(state["evaluation"]) == len(tasks):
                continue
            session = UnifiedResearchSession(ResearchConfig(**{**protocol["research_config"],
                "seed": chosen_seed}), tasks=tasks, measurement_budget=protocol["measurement_budget"], state=state)

            def checkpoint(report):
                write(path, report)
                last = report["results"][-1]["research"]
                print(json.dumps({"seed": chosen_seed, "arm": name,
                    "completed_tasks": len(report["results"]), "kind": last["selected_kind"],
                    "selected_channels": last["selected_channels"], "costs": report["costs"]}), flush=True)

            researcher = UnifiedResearchAgent if name == "default" else WithoutRevisableInputsAgent
            with patch("hegel_machine.cognition.unified_research.UnifiedResearchAgent", researcher):
                session.run(max_tasks=max_tasks, on_checkpoint=checkpoint)
            if len(session.state["results"]) == len(tasks):
                checkpoint(session.evaluate(**protocol["evaluation"]))


def summarize(output, protocol):
    runs = []
    threshold = protocol["deterministic_solved_threshold"]
    for path in sorted((output / "runs").glob("*.json")):
        report = json.loads(path.read_text())
        seed, arm = path.stem.split("_", 1)
        scores = {row["task_position"]: row for row in report["evaluation"]}
        rows = []
        for index, entry in enumerate(report["results"]):
            result = entry["research"]
            plan = result.get("research_program") or {}
            program_trials = [alternative for event in result["actions"]
                if event["action"] == "execute_research_programs" for alternative in event["alternatives"]]
            rows.append({"position": index, "status": result["status"],
                "loss": scores.get(index, {}).get("loss"), "costs": result["costs"],
                "terminal_evaluation_costs": {key: scores.get(index, {}).get(key, 0) for key in COST_KEYS},
                "channels": result["selected_channels"], "kind": result["selected_kind"],
                "used_sources": result.get("used_sources", []),
                "selected_research_program": {
                    "kind": plan.get("kind"), "reused_procedure": bool(plan.get("reused_procedure")),
                    "source_ids": plan.get("source_ids", []), "status": plan.get("status")},
                "replayed_program_candidates": sum(bool(item.get("reused_procedure")) for item in program_trials),
                "revision_actions": [event for event in result["actions"] if any(word in event["action"]
                    for word in ("instrument", "calibrat", "revis", "mechanism", "structure"))]})
        runs.append({"seed": int(seed), "arm": arm, "tasks": rows,
            "saved_input_programs": [{"source_concept_id": prior["source_concept_id"],
                "research_program_preserved": bool(prior.get("input_research_program")),
                "source_dependencies": prior["input_program"]["source_dependencies"]}
                for prior in report["knowledge"].get("meta_priors", []) if prior.get("input_program")],
            "completed": len(scores) == 15, "new_work": metrics(rows, threshold)})
    arms = {}
    for arm in protocol["arms"]:
        rows = [row for run in runs if run["arm"] == arm for row in run["tasks"]]
        arms[arm] = {phase: metrics([row for row in rows if start <= row["position"] < stop], threshold)
            for phase, (start, stop) in PHASES.items()}
    differences = {}
    if all(arm in arms for arm in ARMS):
        for phase in PHASES:
            current, control = (arms[arm][phase] for arm in ARMS)
            differences[phase] = {category: {key: 100 * (current[category][key] / value - 1)
                if value else None for key, value in control[category].items()}
                for category in ("learning_costs", "terminal_evaluation_costs")}
    summary = {"experiment": protocol["experiment"], "arms": arms, "runs": runs,
        "default_cost_change_percent_vs_no_revisable_inputs": differences,
        "completed_sequences": sum(run["completed"] for run in runs),
        "planned_sequences": len(protocol["seeds"]) * len(protocol["arms"]),
        "actual_new_learning_costs": {key: sum(run["new_work"]["learning_costs"][key] for run in runs)
            for key in COST_KEYS},
        "actual_new_terminal_evaluation_costs": {
            key: sum(run["new_work"]["terminal_evaluation_costs"][key] for run in runs) for key in COST_KEYS}}
    old_summary = Path(protocol["historical_instrument_study"])
    if old_summary.is_file():
        summary["ordinary_code_references"] = json.loads(old_summary.read_text()).get("ordinary_code_references", [])
    write(output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "runs"}, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--arms", choices=ARMS, nargs="+")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--max-tasks", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    protocol = prepare(args.output, seeds=args.seeds, arms=args.arms)
    if args.summary_only:
        summarize(args.output, protocol)
    elif not args.prepare_only:
        research(args.output, protocol, seed=args.seed, arm=args.arm, max_tasks=args.max_tasks)


if __name__ == "__main__":
    main()
