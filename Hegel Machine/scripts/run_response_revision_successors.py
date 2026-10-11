#!/usr/bin/env python3
"""Replay paid revisions, then compare retained numerical knowledge on new objects."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]
from scripts import run_response_numeric_transfer as original

SOURCE = ROOT / "artifacts/response_late_withdrawal_20261011"
OUTPUT = ROOT / "artifacts/response_revision_successors_20261011"
ARMS = ("old_only", "revised_only", "retained")
CONDITIONS = ("adapted_law", "original_law_return")
MODULES = ("world_response_consequences", "world_response_transfer", "world_response_hypotheses",
    "world_reference_recalibration", "world_reference_hypothesis", "world_reference_expansion",
    "world_reference_response")
write = original.write


def read(path):
    return json.loads(path.read_text())


def prepare(output):
    previous = read(SOURCE / "after/protocol.json")
    rows = read(SOURCE / "after/brief.json")
    cases = {row["case"]: deepcopy(previous["cases"][row["case"]])
             for row in rows if row["selected_then_refuted"]}
    original_cases = read(original.OUTPUT / "protocol.json")["cases"]
    successors = {}
    for name, config in cases.items():
        for condition in CONDITIONS:
            target = deepcopy(config)
            target.update(source_id=name, condition=condition, revision_source=name,
                          original_source_id=config["source_id"])
            target["signal_values"].update({"4": 6.5, "5": 13.5})
            if condition == "original_law_return":
                target["gain"] = original_cases[config["source_id"] + "__same_mechanism"]["gain"]
                target["gain_factor"] = 1.
            successors[name + "__" + condition] = target
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"source_cases": cases, "successor_cases": successors,
        "source_selection": "All 13 previous after-phase cases with actual selected-then-refuted archives; no source outcome or exported-model filter.",
        "source_artifact": str(SOURCE), "arms": list(ARMS), "conditions": list(CONDITIONS),
        "planned_source_replays": len(cases), "planned_successor_executions": len(successors) * len(ARMS),
        "limits": previous["limits"], "successor_target_signals": [6.5, 13.5],
        "intervention": "Same final source controller and declarations across all arms; only old, revised or jointly retained fixed numerical knowledge differs.",
        "source_replay": "Previous compact traces omit exact ordinary response hypotheses. Replay the frozen source inputs and four archived runtime modules, and export before temporary workspaces are removed. Report trajectory and cost equality; never infer coefficients from fixture truth.",
        "empty_revisions": "Keep every case. Revised-only has no numerical archive if export is empty; retained keeps the old archive unchanged.",
        "cost_scope": "Successor decision costs, common source learning costs and separate source-export compute are reported separately. Diagnostic end-of-run exports do not change the historical research account.",
        "scope": "Thirteen consumed revision branches from nine original development sources, not thirteen independent mechanisms or an LLM adoption experiment.",
        "model_calls": 0}
    write(output / "protocol.json", protocol)
    source = output / "source_replays"
    write(source / "protocol.json", {"cases": cases, "limits": previous["limits"],
        "arms": ["frozen_numeric"], "planned_executions": len(cases)})
    for config in cases.values():
        name = config["source_id"]
        package = read(SOURCE / "after/sources" / name / "numeric_bundle.json")
        write(source / "sources" / name / "numeric_bundle.json", package)
    for module in MODULES:
        archived = SOURCE / "after/runtime" / (module + ".py")
        selected = archived if archived.exists() else ROOT / "src/hegel_machine/cognition" / archived.name
        target = source / "runtime" / archived.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(selected.read_text())
    (output / Path(__file__).name).write_text(Path(__file__).read_text())
    summarize(output)
    print(json.dumps({"status": "prepared", "source_replays": len(cases),
        "successor_executions": len(successors) * len(ARMS)}), flush=True)


def load_runtime(directory):
    # The existing frozen-runtime experiments bind these modules before Agent imports.
    import hegel_machine.cognition as package
    for part in MODULES:
        name = "hegel_machine.cognition." + part
        spec = importlib.util.spec_from_file_location(name, directory / (part + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        setattr(package, part, module)
        spec.loader.exec_module(module)


def prepare_successors(output):
    from scripts.run_response_matched_adoption import prepare_bundle
    from scripts.run_response_composition_transfer import declarations
    from hegel_machine.cognition.world_response_transfer import combine_response_knowledge

    protocol = read(output / "protocol.json")
    for name in protocol["source_cases"]:
        source = output / "source_replays" / name / "frozen_numeric"
        if not (source / "learned_response_knowledge.json").exists():
            raise ValueError("Finish every source replay before preparing successors: " + name)
    directory = output / "successors"
    directory.mkdir(parents=True, exist_ok=False)
    inventory = []
    for name, config in protocol["source_cases"].items():
        source = output / "source_replays" / name / "frozen_numeric"
        original_package = read(output / "source_replays/sources" / config["source_id"] / "numeric_bundle.json")
        controller = (source / "final_controller.py").read_text()
        package = prepare_bundle({"files": {original.METHOD: controller},
            "methods": deepcopy(original_package["methods"])})
        assert declarations(package["files"][original.METHOD]) == declarations(controller)
        old = json.loads(original_package["files"][original.KNOWLEDGE])
        revised = read(source / "learned_response_knowledge.json")
        archives = {"old_only": old, "revised_only": revised,
                    "retained": combine_response_knowledge(old, revised)}
        inventory.append({"source_case": name, "old_models": len(old["models"]),
            "revised_models": len(revised["models"]), "retained_models": len(archives["retained"]["models"]),
            "source_export_compute": revised["compute_cost"], "empty_revision": not revised["models"]})
        for arm, knowledge in archives.items():
            candidate = deepcopy(package)
            if knowledge["models"]:
                candidate["files"][original.KNOWLEDGE] = json.dumps(knowledge)
            write(directory / arm / "sources" / name / "numeric_bundle.json", candidate)
    for arm in ARMS:
        write(directory / arm / "protocol.json", {"cases": protocol["successor_cases"],
            "limits": protocol["limits"], "arms": ["frozen_numeric"], "knowledge_arm": arm,
            "planned_executions": len(protocol["successor_cases"])})
    for module in MODULES:
        target = directory / "runtime" / (module + ".py")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((ROOT / "src/hegel_machine/cognition" / target.name).read_text())
    write(output / "knowledge_inventory.json", inventory)
    summarize(output)
    print(json.dumps({"status": "successors_prepared", "source_archives": len(inventory),
        "empty_revisions": sum(row["empty_revision"] for row in inventory)}), flush=True)


def aggregate(rows):
    return {"cases": len(rows), "successes": sum(row["success"] for row in rows),
        "requested_point_confirmed": {str(n): sum(row["requested_point_confirmed"][str(n)] for row in rows) for n in (4, 5)},
        "unsupported_positive_verdicts": sum(row["unsupported_positive_verdicts"] for row in rows),
        "costs": {key: sum(row["costs"].get(key, 0) for row in rows) for key in
            ("environment_steps", "research_compute_units", "python_execution_seconds", "model_calls", "reported_tokens")},
        "diagnostic_export_compute": sum(row.get("knowledge_export_compute", 0) for row in rows)}


def summarize(output):
    protocol = read(output / "protocol.json")
    replay_rows, preservation = [], []
    for name in protocol["source_cases"]:
        directory = output / "source_replays" / name / "frozen_numeric"
        if not (directory / "brief.json").exists():
            continue
        row = read(directory / "brief.json")
        replay_rows.append(row)
        previous = SOURCE / "after" / name / "frozen_numeric"
        old = read(previous / "brief.json")
        checks = {"action_trace_unchanged": read(previous / "action_trace.json") == read(directory / "action_trace.json"),
            "score_unchanged": read(previous / "research_trace.json")["score"] == read(directory / "research_trace.json")["score"],
            "endpoints_unchanged": all(old[key] == row[key] for key in
                ("success", "status", "point_confirmations", "requested_point_confirmed", "unsupported_positive_verdicts")),
            "research_compute_unchanged": old["costs"]["research_compute_units"] == row["costs"]["research_compute_units"]}
        knowledge = read(directory / "learned_response_knowledge.json")
        preservation.append({"case": name, "preserved": all(checks.values()), **checks,
            "exported_models": len(knowledge["models"]), "export_compute": knowledge["compute_cost"]})
    write(output / "source_replay_preservation.json", preservation)
    rows = []
    for arm in ARMS:
        for case in protocol["successor_cases"]:
            path = output / "successors" / arm / case / "frozen_numeric/brief.json"
            if path.exists():
                rows.append({**read(path), "arm": arm})
    paired = []
    by_pair = {(row["case"], row["arm"]): row for row in rows}
    for case in protocol["successor_cases"]:
        for treatment, baseline in (("revised_only", "old_only"), ("retained", "old_only"), ("retained", "revised_only")):
            if (case, treatment) not in by_pair or (case, baseline) not in by_pair:
                continue
            new, old = by_pair[case, treatment], by_pair[case, baseline]
            paired.append({"case": case, "condition": new["condition"], "treatment": treatment, "baseline": baseline,
                "success_before": old["success"], "success_after": new["success"],
                "requested_point_before": old["requested_point_confirmed"], "requested_point_after": new["requested_point_confirmed"],
                "actions_difference": new["costs"]["environment_steps"] - old["costs"]["environment_steps"],
                "compute_difference": new["costs"]["research_compute_units"] - old["costs"]["research_compute_units"]})
    summary = {"status": "finished" if len(rows) == protocol["planned_successor_executions"] else "running" if replay_rows else "prepared",
        "source_replays": {"completed": len(replay_rows), "planned": protocol["planned_source_replays"],
            "preserved": sum(row["preserved"] for row in preservation), "aggregate": aggregate(replay_rows)},
        "completed_successor_executions": len(rows), "planned_successor_executions": protocol["planned_successor_executions"],
        "conditions": {condition: {arm: aggregate([row for row in rows if row["condition"] == condition and row["arm"] == arm])
            for arm in ARMS} for condition in CONDITIONS}, "paired": paired}
    write(output / "brief.json", rows)
    write(output / "summary.json", summary)
    return summary


def run(output, replay, workers):
    protocol = read(output / "protocol.json")
    jobs = [("source_replays", case) for case in protocol["source_cases"]] if replay else [
        (arm, case) for case in protocol["successor_cases"] for arm in ARMS]
    def destination(arm, case):
        return (output / arm if replay else output / "successors" / arm) / case / "frozen_numeric"
    if any(destination(arm, case).exists() for arm, case in jobs):
        raise ValueError("Preserve the existing batch; this phase already has executions")
    errors = []
    def launch(arm, case):
        log = output / "logs" / arm / (case + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as stream:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--output", str(output),
                "--case", case, "--arm", arm], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        return result.returncode, str(log)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(launch, arm, case): (arm, case) for arm, case in jobs}
        for future in as_completed(pending):
            arm, case = pending[future]
            returncode, log = future.result()
            if returncode:
                errors.append({"arm": arm, "case": case, "returncode": returncode, "log": log})
            write(output / ("source_replay_errors.json" if replay else "successor_errors.json"), errors)
            result = summarize(output)
            print(json.dumps({"arm": arm, "case": case, "returncode": returncode,
                "source_replays": result["source_replays"]["completed"],
                "successor_executions": result["completed_successor_executions"]}), flush=True)
    if errors:
        raise SystemExit("Batch finished with recorded errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--replay-sources", action="store_true")
    parser.add_argument("--prepare-successors", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--case")
    parser.add_argument("--arm", choices=(*ARMS, "source_replays"))
    args = parser.parse_args()
    output = args.output.resolve()
    if args.case and args.arm:
        source = args.arm == "source_replays"
        directory = output / "source_replays" if source else output / "successors" / args.arm
        load_runtime(output / ("source_replays" if source else "successors") / "runtime")
        original.run_case(directory, args.case, "frozen_numeric")
    elif args.prepare:
        prepare(output)
    elif args.replay_sources:
        run(output, True, args.workers)
    elif args.prepare_successors:
        prepare_successors(output)
    elif args.run:
        run(output, False, args.workers)
    elif args.summarize:
        print(json.dumps(summarize(output), indent=2))
    else:
        parser.error("Choose a prepare, replay, run or summarize action")
