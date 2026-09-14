#!/usr/bin/env python3
"""Run complete-library research through dynamic instrument and mechanism drift."""

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/dynamic_instruments_20260913"
PROTOCOL = ARTIFACTS / "protocol.json"
ARMS = ("v074", "dynamic_instruments")
DEVELOPMENT_OUTPUTS = {"v074": "development_v1", "dynamic_instruments": "development_v2"}
COST_KEYS = ("observation_cost", "measurement_cost", "compute_cost")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_runtime(source):
    sys.path.insert(0, str(source.resolve()))
    names = {"models": ("ResearchConfig",), "unified_session": ("UnifiedResearchSession", "laboratory_for"),}
    for module_name, exports in names.items():
        module = importlib.import_module(f"hegel_machine.cognition.{module_name}")
        if not Path(module.__file__).resolve().is_relative_to(source.resolve()):
            raise RuntimeError(f"loaded {module.__file__}, expected {source}")
        globals().update({name: getattr(module, name) for name in exports})
    name = "hegel_machine.cognition.dynamic_instrument_lab"
    spec = importlib.util.spec_from_file_location(name, ARTIFACTS / "dynamic_instrument_lab.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    globals()["DynamicInstrumentLab"] = module.DynamicInstrumentLab


def task_sequence(seed, protocol):
    return [{"environment": "dynamic_instruments", "task_index": stage,
             "seed": seed + stage * protocol["config_seed_offset_per_stage"],
             "apparatus_seed": seed, "profile": "development"}
            for stage in range(len(protocol["stages"]))]


def run(output, protocol, arm, seed, source):
    laboratories = []

    def factory(item):
        laboratory = (DynamicInstrumentLab(**{key: value for key, value in item.items() if key != "environment"})
                      if arm == "v074" else laboratory_for(item))
        laboratories.append(laboratory)
        return laboratory

    session = UnifiedResearchSession(ResearchConfig(**protocol["research_config"], seed=seed),
        tasks=task_sequence(seed, protocol), measurement_budget=protocol["measurement_budget"], lab_factory=factory)
    for stage in range(len(protocol["stages"])):
        state = session.run(max_tasks=1)
        laboratory = laboratories[-1]
        report = state["results"][-1]["research"]
        if any(laboratory.costs[key] != report["costs"][key] for key in ("observation_cost", "measurement_cost")):
            raise RuntimeError("reported cost differs from actual laboratory acquisitions")
        state["results"][-1]["actual_costs"] = laboratory.costs
        state["results"][-1]["instrument_models_after"] = [{
            "model_id": model.get("model_id"), "status": model.get("status"),
            "fields": list(model.get("fields", {})), "unmodeled_fields": model.get("unmodeled_fields", {})}
            for model in state["knowledge"].get("instrument_models", [])]
        # A fresh runtime object receives the complete JSON library and session.
        session = UnifiedResearchSession(state=json.loads(json.dumps(state)), lab_factory=factory)
        print(json.dumps({"arm": arm, "seed": seed, "stage": stage, "status": report["status"],
            "selected_channels": report["selected_channels"], "kind": report["selected_kind"],
            "costs": report["costs"], "instrument_models": len(state["knowledge"].get("instrument_models", []))}), flush=True)
    state = session.evaluate(**{key: protocol["terminal"][key] for key in ("length", "trajectories", "seed_offset")})
    report = {"arm": arm, "seed": seed, "runtime_source": str(source.resolve().relative_to(ROOT)),
              "laboratory_source": str((ARTIFACTS / "dynamic_instrument_lab.py").relative_to(ROOT)), **state}
    write(output / "runs" / f"{seed}_{arm}.json", report)
    return report


def summarize(output, protocol):
    summaries, missing = {}, []
    for cohort, seeds in (("development", protocol["development_seeds"]), ("formal", protocol["formal_seeds"])):
        summaries[cohort] = {}
        for arm in ARMS:
            tasks = []
            for seed in seeds:
                path = output / "runs" / f"{seed}_{arm}.json"
                if cohort == "development" and not path.exists():
                    path = ARTIFACTS / DEVELOPMENT_OUTPUTS[arm] / "runs" / f"{seed}_{arm}.json"
                if not path.exists():
                    missing.append({"cohort": cohort, "seed": seed, "arm": arm})
                    continue
                result = json.loads(path.read_text())
                final_models = {model["model_id"]: model for model in result.get("knowledge", {}).get("instrument_models", [])}
                for entry, evaluation in zip(result["results"], result["evaluation"]):
                    research = entry["research"]
                    actions = research["actions"]
                    labels = sorted({action["action"] for action in actions if any(
                        token in action["action"] for token in ("instrument", "measurement", "input"))})
                    tasks.append({"seed": seed, "stage": entry["task_position"],
                        "source": str(path.resolve().relative_to(ROOT)),
                        "loss": evaluation["loss"], "solved": evaluation["loss"] <= protocol["terminal"]["solved_threshold"],
                        "costs": research["costs"], "actual_costs": entry["actual_costs"],
                        "terminal_costs": {key: evaluation[key] for key in COST_KEYS},
                        "selected_channels": research["selected_channels"],
                        "selected_kind": research["selected_kind"],
                        "instrument_models_after": entry["instrument_models_after"],
                        "instrument_fields_after": {model["model_id"]: {name: {key: field[key]
                            for key in ("kind", "degree", "history_depth") if key in field}
                            for name, field in final_models[model["model_id"]].get("fields", {}).items()
                            if name in model["fields"]}
                            for model in entry["instrument_models_after"] if model["model_id"] in final_models},
                        "actions": {label: sum(a["action"] == label for a in actions) for label in labels}})
            def aggregate(rows):
                labels = sorted({label for row in rows for label in row["actions"]})
                return {"tasks": len(rows), "solved": sum(row["solved"] for row in rows),
                    "max_loss": max((row["loss"] for row in rows), default=None),
                    "learning_costs": {key: sum(row["costs"][key] for row in rows) for key in COST_KEYS},
                    "terminal_costs": {key: sum(row["terminal_costs"][key] for row in rows) for key in COST_KEYS},
                    "measurement_free_predictions": sum(not row["selected_channels"] for row in rows),
                    "measurement_free_solved": sum(not row["selected_channels"] and row["solved"] for row in rows),
                    "incorrect_measurement_free_predictions": sum(not row["selected_channels"] and not row["solved"] for row in rows),
                    "actual_queries": sum(row["actual_costs"]["queries"] for row in rows),
                    "actions": {label: sum(row["actions"].get(label, 0) for row in rows) for label in labels}}
            summaries[cohort][arm] = {"all": aggregate(tasks),
                "stages": {name: aggregate([t for t in tasks if t["stage"] == stage])
                           for stage, name in enumerate(protocol["stages"])}, "tasks": tasks}
    result = {"cohorts": summaries, "missing_runs": missing,
        "development_history": {
            "initial_study": "artifacts/dynamic_instruments_20260913/development_v1/runs/151_dynamic_instruments.json",
            "initial_runtime_failures": "artifacts/dynamic_instruments_20260913/runtime_tests_v1.log",
            "selected_study": "artifacts/dynamic_instruments_20260913/development_v2/runs/151_dynamic_instruments.json",
            "selected_runtime_tests": "artifacts/dynamic_instruments_20260913/runtime_tests_v2.log"}}
    write(output / "summary.json", result)
    print(json.dumps({cohort: {arm: item["all"] for arm, item in arms.items()}
                      for cohort, arms in summaries.items()}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "comparison_v1")
    parser.add_argument("--config", type=Path, default=PROTOCOL)
    parser.add_argument("--arm", choices=ARMS, default="dynamic_instruments")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.config.read_text())
    if args.summary_only:
        summarize(args.output, protocol)
        return
    if args.seed not in protocol["development_seeds"] + protocol["formal_seeds"]:
        parser.error("seed must be declared in the frozen protocol")
    destination = args.output / "runs" / f"{args.seed}_{args.arm}.json"
    if destination.exists():
        parser.error(f"completed run already exists: {destination}")
    source = args.source or ARTIFACTS / ("v074_runtime" if args.arm == "v074" else "v075_runtime") / "src"
    if not source.is_dir():
        parser.error(f"freeze runtime first: {source}")
    load_runtime(source)
    run(args.output, protocol, args.arm, args.seed, source)


if __name__ == "__main__":
    main()
