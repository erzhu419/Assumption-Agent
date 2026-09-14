#!/usr/bin/env python3
"""Study measured competition between non-equivalent object explanations."""

import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/object_competition_20260913/v067_runtime"
SEEDS = (6060319271, 6060419287, 6060519299)
ARMS = ("v067", "competing_objects")
COST_KEYS = ("observation_cost", "measurement_cost", "compute_cost")
PHASES = {"all": (0, 4), "shared_mechanism": (0, 1), "competition": (1, 2),
          "reuse": (2, 3), "successor": (3, 4), "downstream": (2, 4)}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def tasks_for(seed):
    plant, count = dict(zip(SEEDS, ((31013, 2), (32003, 2), (33013, 2)))).get(seed, (27011, 2))
    return [{"environment": "anonymous_objects", "objects": count, "apparatus_seed": plant,
        "variant": "exchange", "readout": readout,
        "seed": seed + 10007 * index, "name_seed": seed + 63857, "wiring_seed": seed + 3709}
        for index, readout in enumerate(("focus", "position_focus", "position_focus", "event"))]


def load_runtime(source_root):
    source_root = source_root.resolve()
    sys.path.insert(0, str(source_root))
    module = importlib.import_module("hegel_machine.cognition.unified_research")
    loaded = Path(module.__file__).resolve()
    if not loaded.is_relative_to(source_root):
        raise RuntimeError(f"research imported from {loaded}, expected {source_root}")
    return module


def prepare(output, *, seeds=None):
    from hegel_machine.cognition.models import ResearchConfig
    path = output / "protocol.json"
    saved = json.loads(path.read_text()) if path.exists() else None
    seeds = list(seeds or (saved["seeds"] if saved else SEEDS))
    config = ResearchConfig(seed=97, observation_budget=512, compute_budget=480000,
        minimum_improvement=1e-7, max_rounds=6, active_probes=True)
    protocol = {"experiment": "measured_object_competition_v068", "seeds": seeds,
        "arms": list(ARMS), "tasks_by_seed": {str(seed): tasks_for(seed) for seed in seeds},
        "research_config": asdict(config), "measurement_budget": 6144,
        "evaluation": {"length": 64, "trajectories": 4, "seed_offset": 1000003},
        "deterministic_solved_threshold": 1e-6, "baseline_source": str(BASELINE),
        "contract": "Both versions start with empty knowledge, share anonymous observations, "
            "control bounds, physical transitions and budgets. Two objects leave their coordinate "
            "membership ambiguous in exchange dynamics. An ordinary focus response precedes a "
            "position-sensitive focus response, its repetition and an event question. Only the "
            "readout changes; physical role evidence remains unchanged. Both frozen algorithms "
            "use the same extended apparatus file. Complete streams finish before fresh full-domain "
            "prediction and private role scoring; terminal outcomes never become learning feedback.",
        "shared_apparatus": "shared_apparatus/structure_lab.py",
        "planned_tasks": len(seeds) * len(ARMS) * 4}
    if saved is not None and saved != protocol:
        raise ValueError("protocol changed; preserve this cohort and use a new output directory")
    if saved is None:
        write(path, protocol)
    return protocol


def load_shared_apparatus(output):
    """Give both algorithm versions the same new supported readout."""
    import importlib.util
    import hegel_machine.cognition.unified_object_lab as interface
    original = ROOT / "src/hegel_machine/cognition/structure_lab.py"
    target = output / "shared_apparatus/structure_lab.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(original, target)
    name = "hegel_machine.cognition.competition_apparatus"
    specification = importlib.util.spec_from_file_location(name, target)
    apparatus = importlib.util.module_from_spec(specification)
    sys.modules[name] = apparatus
    specification.loader.exec_module(apparatus)
    interface.StructureLab = apparatus.StructureLab
    return target


def score_objects(report):
    from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
    rows = []
    for task, entry in zip(report["tasks"], report["results"]):
        result = entry["research"]
        with AnonymousObjectLab(**{k: v for k, v in task.items() if k != "environment"}) as lab:
            truth = lab._evaluation_schema()
        quantities = {truth.fields[obj]["q"]: obj for obj in truth.objects}
        groups = (result.get("object_research") or {}).get("groups", [])
        correct, proposed, complete = 0, 0, 0
        claimed_quantities = [group.get("q") for group in groups]
        for group in groups:
            actual = quantities.get(group.get("q"))
            unique = claimed_quantities.count(group.get("q")) == 1
            fields = {role: group[role] for role in ("q", "x", "u", "f") if role in group}
            controls = group.get("controls", {})
            proposed += len(fields) + len(controls)
            if actual is not None and unique:
                matches = sum(value == truth.fields[actual].get(role) for role, value in fields.items())
                matches += sum(value == truth.controls[actual].get(role) for role, value in controls.items())
                correct += matches
                complete += matches == 7
        object_research = result.get("object_research") or {}
        focus_correct = object_research.get("focus_control") == truth.focus_control
        rows.append({"task_position": entry["task_position"], "objects": len(truth.objects),
            "correct_known_roles": correct, "proposed_known_roles": proposed,
            "complete_objects": complete, "focus_control_correct": focus_correct,
            "exact": complete == len(truth.objects) == len(groups) and focus_correct,
            "adopted_schema": result.get("learned_object_schema") is not None,
            "unresolved": object_research.get("unresolved", [])})
    report["object_evaluation"] = rows
    return report


def freeze(output, arm, source_root):
    source = source_root / "hegel_machine/cognition"
    for original in (*source.glob("*.py"), Path(__file__).resolve()):
        target = output / (arm + "_source") / original.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != original.read_bytes():
            raise ValueError(f"{arm} source changed during study: {original.name}")
        if not target.exists():
            shutil.copyfile(original, target)


def research(output, protocol, arm, source_root, *, seed=None, max_tasks=None):
    from hegel_machine.cognition.models import ResearchConfig
    from hegel_machine.cognition.unified_session import UnifiedResearchSession
    freeze(output, arm, source_root)
    for chosen in protocol["seeds"]:
        if seed is not None and chosen != seed:
            continue
        path = output / "runs" / f"{chosen}_{arm}.json"
        state = json.loads(path.read_text()) if path.exists() else None
        tasks = protocol["tasks_by_seed"][str(chosen)]
        if state and len(state.get("object_evaluation", [])) == len(tasks):
            continue
        session = UnifiedResearchSession(ResearchConfig(**{**protocol["research_config"], "seed": chosen}),
            tasks=tasks, state=state, measurement_budget=protocol["measurement_budget"])

        def checkpoint(report):
            write(path, report)
            last = report["results"][-1]["research"]
            print(json.dumps({"seed": chosen, "arm": arm, "learned": len(report["results"]),
                "evaluated": len(report["evaluation"]), "kind": last["selected_kind"],
                "channels": last["selected_channels"], "costs": report["costs"],
                "object_status": (last.get("object_research") or {}).get("status"),
                "adopted_schema": last.get("learned_object_schema") is not None}), flush=True)

        session.run(max_tasks=max_tasks, on_checkpoint=checkpoint)
        if len(session.state["results"]) == len(tasks):
            checkpoint(score_objects(session.evaluate(**protocol["evaluation"])))


def compact_model(model):
    return {"status": model.get("status"), "complete": model.get("complete"),
        "group_count": len(model.get("groups", [])),
        "unresolved": model.get("unresolved", []),
        "position_candidates": {field: value.get("quantities", [])
            for field, value in model.get("position_candidates", {}).items()},
        "focus_candidates": {field: value.get("quantities", [])
            for field, value in model.get("focus_candidates", {}).items()},
        "alternatives": {"count": len(model.get("schema_alternatives", {}).get("schemas", [])),
            "exhaustive": model.get("schema_alternatives", {}).get("exhaustive"),
            "status": model.get("schema_alternatives", {}).get("status")}}


def compact_equivalence(equivalence):
    if not equivalence:
        return {}
    return {**{key: value for key, value in equivalence.items()
        if value is None or isinstance(value, (str, int, float, bool))},
        "schema_count": len(equivalence.get("schemas", [])),
        "unresolved": equivalence.get("unresolved", []),
        "member_families": equivalence.get("member_families", [])}


def compact_competition(state):
    return {"status": state.get("status"), "revision": state.get("revision"),
        "schemas": len(state.get("schemas", [])), "exhaustive": state.get("exhaustive"),
        "program_classes": state.get("program_classes"),
        "hypotheses": [{key: item.get(key) for key in
            ("hypothesis_id", "status", "training_loss", "training_rows", "parent_hypothesis_id")}
            for item in state.get("hypotheses", [])],
        "counterexamples": len(state.get("counterexamples", [])),
        "compute_cost": state.get("compute_cost")}


def summarize(output, protocol):
    runs = []
    for path in sorted((output / "runs").glob("*.json")):
        report = json.loads(path.read_text())
        seed, arm = path.stem.split("_", 1)
        scores = {row["task_position"]: row for row in report["evaluation"]}
        objects = {row["task_position"]: row for row in report.get("object_evaluation", [])}
        rows = []
        for entry in report["results"]:
            index, result = entry["task_position"], entry["research"]
            rows.append({"position": index, "status": result["status"],
                "loss": scores.get(index, {}).get("loss"), "costs": result["costs"],
                "terminal_costs": {key: scores.get(index, {}).get(key, 0) for key in COST_KEYS},
                "kind": result["selected_kind"], "channels": result["selected_channels"],
                "object_evaluation": objects.get(index), "object_model_id": result.get("object_model_id"),
                "object_model": compact_model(result.get("object_research") or {}),
                "used_equivalent_program": bool(result.get("object_equivalence")),
                "confirmed_equivalent_program": any(event["action"] == "adopt_equivalent_object_program" for event in result["actions"]),
                "object_equivalence": compact_equivalence(result.get("object_equivalence") or {}),
                "competition": compact_competition(result.get("object_competition") or {}),
                "retained_competitions": [compact_competition(item)
                    for item in result.get("program_competitions", [])],
                "competition_experiments": [{"probe_id": event["probe_id"],
                    "forecast_status": event["plan"].get("forecast_status"),
                    "predicted_separation": event["plan"].get("predicted_separation"),
                    "actual_status": event["evidence"].get("status"),
                    "calibration_status": event["evidence"].get("calibration_status"),
                    "hypotheses": event["evidence"].get("hypotheses", [])}
                    for event in result["actions"] if event["action"] == "test_competing_object_predictions"],
                "used_sources": result.get("used_sources", []),
                "actions": [{key: value for key, value in event.items()
                    if value is None or isinstance(value, (str, int, float, bool))}
                    for event in result["actions"] if any(word in event["action"]
                    for word in ("object", "partial_roles", "dependenc", "equival", "compet"))]})
        runs.append({"seed": int(seed), "arm": arm, "tasks": rows,
                     "completed": len(objects) == len(report["tasks"])})
    arms = {}
    for arm in ARMS:
        arms[arm] = {}
        phases = PHASES
        for phase, (start, stop) in phases.items():
            rows = [row for run in runs if run["arm"] == arm for row in run["tasks"]
                    if start <= row["position"] < stop]
            arms[arm][phase] = {"tasks": len(rows), "evaluated": sum(row["loss"] is not None for row in rows),
                "solved": sum(row["loss"] is not None and row["loss"] <= 1e-6 for row in rows),
                "commands": sum(not row["channels"] for row in rows),
                "selected_equivalent_programs": sum(row["used_equivalent_program"] for row in rows),
                "confirmed_equivalent_programs": sum(row["confirmed_equivalent_program"] for row in rows),
                "competition_experiments": sum(len(row["competition_experiments"]) for row in rows),
                "retained_counterexamples": sum(item["counterexamples"] for row in rows
                    for item in row["retained_competitions"]),
                "used_prior_knowledge": sum(bool(row["used_sources"]) for row in rows),
                "learning_costs": {key: sum(row["costs"][key] for row in rows) for key in COST_KEYS},
                "terminal_costs": {key: sum(row["terminal_costs"][key] for row in rows) for key in COST_KEYS}}
    summary = {"experiment": protocol["experiment"], "arms": arms, "runs": runs,
               "completed_sequences": sum(run["completed"] for run in runs)}
    summary["comparison"] = {}
    for phase in PHASES:
        old, new = (arms[arm][phase] for arm in ARMS)
        if old["tasks"] and old["tasks"] == old["evaluated"] == new["tasks"] == new["evaluated"]:
            summary["comparison"][phase] = {
                "old_solved": old["solved"], "new_solved": new["solved"], "tasks_per_arm": old["tasks"],
                "learning_cost_change": {key: new["learning_costs"][key] - old["learning_costs"][key]
                    for key in COST_KEYS},
                "learning_cost_reduction_percent": {key: 100 * (1 - new["learning_costs"][key] /
                    old["learning_costs"][key]) if old["learning_costs"][key] else None for key in COST_KEYS}}
    write(output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "runs"}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm", choices=ARMS, default="competing_objects")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--max-tasks", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    source = (args.source_root or (BASELINE if args.arm == "v067" else ROOT / "src")).resolve()
    runtime = load_runtime(source)
    print(json.dumps({"arm": args.arm, "runtime": runtime.__file__}), flush=True)
    protocol = prepare(args.output, seeds=args.seeds)
    load_shared_apparatus(args.output)
    if args.summary_only:
        summarize(args.output, protocol)
    elif not args.prepare_only:
        research(args.output, protocol, args.arm, source, seed=args.seed, max_tasks=args.max_tasks)


if __name__ == "__main__":
    main()
