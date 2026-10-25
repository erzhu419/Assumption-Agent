#!/usr/bin/env python3
"""Frozen numeric response forecasts and matched research on fresh objects."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
from math import ceil, floor, isclose
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/hegel_machine/cognition"))
from checkpoint_storage import write_json_checkpoint
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]
SOURCE = ROOT / "artifacts/response_retained_program_support_20261011"
OUTPUT = ROOT / "artifacts/response_numeric_transfer_20261011"
METHOD = "shared/consolidated_method.py"
KNOWLEDGE = "shared/frozen_response_knowledge.json"
ARMS = ("basis_only", "frozen_numeric")
SOURCES = tuple([f"cubic_original_{i}" for i in range(1, 5)]
    + [f"cubic_shifted_{i}" for i in (1, 2, 4)] + ["square_original", "square_shifted"])


def write(path, value):
    write_json_checkpoint(path, value, indent=2, ensure_ascii=True, allow_nan=True)


def source_knowledge(name):
    from hegel_machine.cognition.world_response_transfer import export_response_knowledge
    from scripts.run_response_composition_transfer import declarations
    directory = SOURCE / name / "control_preparation"
    trace = json.loads((directory / "research_trace.json").read_text())
    # These are the exact archived witnesses, not newly reconstructed band endpoints.
    history = []
    for qid, question in trace["questions"].items():
        models = (question.get("answer", {}).get("conditional_simplification", {})
                  .get("joint_revision", {}).get("models", []))
        if models:
            history.append({"question_id": qid, "hypotheses": deepcopy(models)})
    state = {"research_table": trace["research_table"],
             "response_settings": trace["response_settings"],
             "response_research": {}, "response_research_history": history}
    controller = (directory / "final_controller.py").read_text()
    knowledge = export_response_knowledge(state, declarations(controller), name)
    knowledge["source_artifact"] = str(directory / "research_trace.json")
    knowledge["archive_scope"] = "Exact archived joint witnesses rechecked on all final paid responses"
    return knowledge, controller


def prepare(output):
    from scripts.run_response_matched_adoption import prepare_bundle
    output.mkdir(parents=True, exist_ok=False)
    previous = json.loads((SOURCE / "protocol.json").read_text())
    cases, counts = {}, {}
    for name in SOURCES:
        knowledge, controller = source_knowledge(name)
        if not knowledge["models"]:
            raise ValueError("Predeclared source has no surviving exact model: " + name)
        counts[name] = len(knowledge["models"])
        package = prepare_bundle({"files": {METHOD: controller}, "methods": [{
            "id": "source_program", "path": METHOD, "parameters": {},
            "requires": [], "max_steps": 256}]})
        write(output / "sources" / name / "knowledge.json", knowledge)
        write(output / "sources" / name / "basis_bundle.json", package)
        numeric = deepcopy(package)
        numeric["files"][KNOWLEDGE] = json.dumps(knowledge)
        write(output / "sources" / name / "numeric_bundle.json", numeric)
        for condition in ("same_mechanism", "changed_intercept"):
            config = deepcopy(previous["cases"][name])
            config["signal_values"].update({"4": 5.25, "5": 12.5})
            if condition == "changed_intercept":
                config["intercept"] += 60.
            cases[name + "__" + condition] = {**config, "source_id": name, "condition": condition}
    limits = {**previous["limits"], "environment_steps": 512, "max_model_calls": 0,
        "python_seconds_per_call": 30, "python_seconds_per_task": 240, "task_wall_seconds": 3600}
    protocol = {"cases": cases, "arms": ARMS, "planned_executions": len(cases) * len(ARMS),
        "source_models": counts, "source_root": str(SOURCE), "limits": limits,
        "new_target_signals": [5.25, 12.5], "intercept_control_shift": 60.,
        "intervention": "Identical source declarations and controller; numeric arm additionally imports frozen coefficient/band models.",
        "forecast_rule": "Before target adaptation, test first unanimous positive and nearest unanimous negative from finite support endpoints/midpoints and adjacent grid points; retain ambiguous predictions and positive abstention.",
        "forecast_execution": "Each endpoint uses a fresh laboratory, public measurements and the existing tuner; other target stays uninstalled. Freeze predictions before target responses.",
        "scope": "New object values within a designed development fixture. Sources and mechanism families are consumed; not held-out benchmark or LLM discovery.",
        "storage": "Compact final research state and complete action trace; forecast public observations; no repeated agent snapshots."}
    modules = ("world_response_transfer", "world_response_hypotheses", "world_response_consequences",
        "world_reference_recalibration", "world_reference_response", "world_reference_hypothesis",
        "world_reference_expansion")
    for module in modules:
        path = output / "runtime" / (module + ".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text((ROOT / "src/hegel_machine/cognition" / path.name).read_text())
    protocol["runtime_snapshot"] = list(modules)
    write(output / "protocol.json", protocol)
    (output / Path(__file__).name).write_text(Path(__file__).read_text())
    write(output / "summary.json", summarize([], protocol))
    print(json.dumps({"status": "prepared", "executions": protocol["planned_executions"], "source_models": counts}), flush=True)


def measured_inputs(lab, number):
    """Acquire inputs through public instruments; never inspect fixture truth."""
    from scripts.run_response_matched_adoption import recipe
    records, features = [], {}

    def act(action):
        returned = lab.act(action)
        records.append({"index": len(records), **deepcopy(returned)})
        return returned["observation"]

    view = lab.public_observation()
    objects = recipe._objects(view)
    crystal = next(obj["uuid"] for obj in objects if obj.get("name") == "quantum crystal " + str(number))
    reactor = next(obj["uuid"] for obj in objects if obj["uuid"] == 100 + number)
    instruments = {"density": ("densitometer", r"density is approximately ([\d.]+)"),
        "gap": ("microscope", r"quantum gap is estimated to be ([\d.]+)"),
        "signal": ("signal meter", r"signal magnitude = ([\d.]+)")}
    for feature, (name, pattern) in instruments.items():
        uid = next(obj["uuid"] for obj in objects if obj.get("name") == name)
        if not any(obj["uuid"] == uid for obj in view["inventoryObjects"]):
            view = act({"action": "PICKUP", "arg1": uid})
        view = act({"action": "USE", "arg1": uid, "arg2": crystal})
        features[feature] = float(re.search(pattern, view["lastActionMessage"])[1])
    view = act({"action": "TALK", "arg1": reactor})
    grid = recipe._control_grids(records, [], set())[str(number)]
    return features, grid, records, crystal, reactor


def forecast_settings(knowledge, features, grid):
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.models import Observation
    from hegel_machine.cognition.world_response_hypotheses import _setting_proposals
    from hegel_machine.cognition.world_response_transfer import forecast_response_knowledge
    candidates = set()
    for model in knowledge["models"]:
        center = Program.from_dict(model["program"]).predict(Observation(features, 0.))
        candidates.update(_setting_proposals(center, model["lower"], model["upper"], grid))
        candidates.update(grid["origin"] + grid["step"] * index
            for index in (floor((center - grid["origin"]) / grid["step"]),
                          ceil((center - grid["origin"]) / grid["step"])))
    candidates |= {point + shift * grid["step"] for point in list(candidates) for shift in (-1, 1)}
    forecasts = forecast_response_knowledge(knowledge, features,
        sorted(point for point in candidates if grid["lower"] <= point <= grid["upper"]))
    positives = [row for row in forecasts if row["prediction"] is True]
    ambiguous = [row for row in forecasts if row["prediction"] is None]
    anchor = (positives or ambiguous or forecasts)[0]
    negatives = [row for row in forecasts if row["prediction"] is False]
    chosen = [positives[0]] if positives else ([ambiguous[0]] if ambiguous else [])
    if negatives:
        chosen.append(min(negatives, key=lambda row: (abs(row["setting"] - anchor["setting"]), row["setting"])))
    return {"forecasts": forecasts, "chosen": chosen, "positive_abstention": not positives,
            "ambiguous_settings": len(ambiguous)}


def execute_forecast(lab, number, setting):
    from scripts.run_response_matched_adoption import recipe
    features, grid, records, crystal, reactor = measured_inputs(lab, number)
    def act(action):
        returned = lab.act(action)
        records.append({"index": len(records), **deepcopy(returned)})
        return returned["observation"]
    # Measurement happens before installation. Only this target is installed.
    view = act(recipe._exit(lab.public_observation())["actions"][0])
    act({"action": "PICKUP", "arg1": crystal})
    act({"action": "PUT", "arg1": crystal, "arg2": reactor})
    view = act({"action": "TALK", "arg1": reactor})
    first = len(records) - 1
    while len(records) < 512:
        tuning = recipe._tune(view, setting)
        if not tuning:
            break
        view = act(tuning["actions"][0])
    _, actual = recipe._dialog(view)
    reached = isclose(actual, setting, abs_tol=1e-6)
    activated = recipe._activated(recipe._objects(view), reactor) if reached else None
    incidental = [{"index": row["index"], "setting": recipe._dialog(row["observation"])[1]}
        for row in records[first:] if recipe._activated(recipe._objects(row["observation"]), reactor)
        and not isclose(recipe._dialog(row["observation"])[1], setting, abs_tol=1e-6)]
    return {"requested_setting_reached": reached, "actual_setting": actual, "activated": activated,
            "intermediate_activations": incidental, "actions": len(records), "observations": records}


def run_forecasts(output, case):
    from scripts import consolidate_response_measurement as study
    protocol = json.loads((output / "protocol.json").read_text())
    config = protocol["cases"][case]
    study.CASES = {case: config}
    knowledge = json.loads((output / "sources" / config["source_id"] / "knowledge.json").read_text())
    directory = output / case / "forecasts"
    directory.mkdir(parents=True, exist_ok=False)
    plans = []
    for number in (4, 5):
        features, grid, records, _, _ = measured_inputs(study.TransferSignalLab(case), number)
        plans.append({"number": number, "features": features, "grid": grid,
            "measurement_observations": records, **forecast_settings(knowledge, features, grid)})
    # This file predates every installed-target response used to score it.
    write(directory / "predictions_before_execution.json", plans)
    results = []
    for plan in plans:
        for index, forecast in enumerate(plan["chosen"]):
            actual = execute_forecast(study.TransferSignalLab(case), plan["number"], forecast["setting"])
            observations = actual.pop("observations")
            write(directory / f"target_{plan['number']}_trial_{index}_observations.json", observations)
            results.append({"number": plan["number"], **forecast, **actual,
                "correct": (forecast["prediction"] == actual["activated"]
                    if forecast["prediction"] is not None and actual["requested_setting_reached"] else None)})
    summary = {"case": case, "condition": config["condition"], "source_id": config["source_id"],
        "positive_abstentions": sum(plan["positive_abstention"] for plan in plans),
        "ambiguous_settings": sum(plan["ambiguous_settings"] for plan in plans), "results": results,
        "model_calls": 0, "reported_tokens": 0}
    write(directory / "brief.json", summary)
    return summary


def run_case(output, case, arm):
    from scripts import run_response_matched_adoption as matched
    protocol = json.loads((output / "protocol.json").read_text())
    config = protocol["cases"][case]
    matched.study.CASES = {case: config}
    bundle = "numeric_bundle.json" if arm == "frozen_numeric" else "basis_bundle.json"
    package = json.loads((output / "sources" / config["source_id"] / bundle).read_text())
    destination = output / case / arm
    destination.mkdir(parents=True, exist_ok=False)
    with TemporaryDirectory(prefix="hegel-numeric-transfer-") as temporary:
        row = matched.study.run_transfer_case(case, arm, package, temporary, protocol["limits"],
            on_checkpoint=lambda data: write(destination / "active.json", {
                "status": data["status"], "costs": data["costs"], "score": data["score"]}))
        directory = Path(temporary) / case / arm
        result = json.loads((directory / "result.json").read_text())
        state = json.loads((directory / "agent/shared/reference_recalibration.json").read_text())
        result["reference_state"] = {key: state[key] for key in ("tests", "test") if key in state}
        points = matched.point_confirmations(result)
        row.update(condition=config["condition"], source_id=config["source_id"],
            point_confirmations=points, requested_point_confirmed={str(number): any(
                point["number"] == number and point["physically_confirmed"] for point in points) for number in (4, 5)},
            unsupported_positive_verdicts=sum(point["system_supported"] and not point["physically_confirmed"] for point in points))
        write(destination / "research_trace.json", {"score": result["score"],
            "questions": result["research"]["questions"],
            "state": {key: deepcopy(value) for key, value in state.items()
                if key not in ("response_research", "response_research_history", "family_history_checks")},
            "comparisons": [{key: deepcopy(value) for key, value in item.items()
                if key in ("question_id", "status", "response_count", "compute_cost", "selected_experiment", "frozen_transfer")}
                for item in state.get("response_research_history", [])]})
        write(destination / "action_trace.json", [{key: record.get(key) for key in ("index", "action", "feedback")}
            for record in result["observations"]])
        evidence = {index for question in result["research"]["questions"] for index in question.get("evidence", [])}
        if result["observations"]:
            evidence.add(result["observations"][-1]["index"])
        write(destination / "evidence_observations.json", [record for record in result["observations"] if record["index"] in evidence])
        controller = (directory / "agent" / METHOD).read_text()
        (destination / "final_controller.py").write_text(controller)
        # Final physical feedback can arrive after the last comparison. Freeze
        # surviving members and source-data band revisions before releasing it.
        from scripts.run_response_composition_transfer import declarations
        from hegel_machine.cognition.world_reference_recalibration import refresh_response_evidence
        from hegel_machine.cognition.world_response_transfer import export_response_knowledge

        estimators = declarations(controller)
        export_state = deepcopy(state)
        refresh_response_evidence(export_state, result['observations'],
                                  result['observations'][-1]['observation'], estimators)
        knowledge = export_response_knowledge(export_state, estimators, case)
        knowledge['source_artifact'] = str(destination / 'research_trace.json')
        knowledge['archive_scope'] = 'Surviving exact members and bands revised on final paid source responses; coefficient programs remain fixed'
        write(destination / 'learned_response_knowledge.json', knowledge)
        row['knowledge_export_compute'] = knowledge['compute_cost']
    write(destination / "brief.json", row)
    write(destination / "active.json", {"finished": True, "success": row["success"], "status": row["status"]})
    print(json.dumps({key: row[key] for key in ("case", "arm", "success", "requested_point_confirmed")}), flush=True)


def summarize(rows, protocol):
    return {"completed_executions": len(rows), "planned_executions": protocol["planned_executions"],
        "status": "finished" if len(rows) == protocol["planned_executions"] else "running" if rows else "prepared",
        "conditions": {condition: {arm: {"cases": len(selected), "successes": sum(row["success"] for row in selected),
            "requested_point_confirmed": {str(number): sum(row["requested_point_confirmed"][str(number)] for row in selected) for number in (4, 5)},
            "unsupported_positive_verdicts": sum(row["unsupported_positive_verdicts"] for row in selected),
            "costs": {key: sum(row["costs"].get(key, 0) for row in selected) for key in
                ("environment_steps", "research_compute_units", "python_execution_seconds", "model_calls", "reported_tokens")}}
            for arm in ARMS for selected in [[row for row in rows if row["arm"] == arm and row["condition"] == condition]]}
            for condition in ("same_mechanism", "changed_intercept")}}


def run(output, workers):
    protocol = json.loads((output / "protocol.json").read_text())
    jobs = [(case, arm) for case in protocol["cases"] for arm in ARMS]
    if any((output / case / arm).exists() for case, arm in jobs):
        raise ValueError("Preserve the existing batch; this directory has started runs")
    forecasts = [run_forecasts(output, case) for case in protocol["cases"]]
    write(output / "forecast_summary.json", forecasts)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(subprocess.run, [sys.executable, str(Path(__file__).resolve()),
            "--output", str(output), "--case", case, "--arm", arm], check=True): (case, arm) for case, arm in jobs}
        for future in as_completed(pending):
            future.result()
            case, arm = pending[future]
            rows.append(json.loads((output / case / arm / "brief.json").read_text()))
            rows.sort(key=lambda row: jobs.index((row["case"], row["arm"])))
            write(output / "brief.json", rows)
            write(output / "summary.json", summarize(rows, protocol))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--case")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--forecasts-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.prepare_only:
        prepare(output)
    elif args.case and args.arm:
        run_case(output, args.case, args.arm)
    elif args.case and args.forecasts_only:
        run_forecasts(output, args.case)
    elif args.run:
        run(output, args.workers)
    else:
        parser.error("Choose --prepare-only, --run, or --case with --arm/--forecasts-only")
