#!/usr/bin/env python3
"""Resume ordinary fitting after frozen numeric knowledge is withdrawn."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]
from scripts import run_response_numeric_transfer as original

SOURCE = original.OUTPUT
OUTPUT = ROOT / "artifacts/response_transfer_stage_resume_20261011"
write = original.write


def read(path):
    return json.loads(path.read_text())


def prepare(output):
    from scripts.run_response_matched_adoption import prepare_bundle
    from scripts.run_response_composition_transfer import declarations

    prior = read(SOURCE / "protocol.json")
    output.mkdir(parents=True, exist_ok=False)
    package_changes = []
    for name in prior["source_models"]:
        source = SOURCE / "sources" / name
        package = read(source / "numeric_bundle.json")
        refreshed = prepare_bundle(package)
        # Scientific declarations and exact frozen members are the fixed inputs.
        assert declarations(refreshed["files"][original.METHOD]) == declarations(package["files"][original.METHOD])
        assert refreshed["files"][original.KNOWLEDGE] == package["files"][original.KNOWLEDGE]
        assert refreshed["methods"] == package["methods"]
        write(output / "sources" / name / "numeric_bundle.json", refreshed)
        (output / "sources" / name / "knowledge.json").write_text((source / "knowledge.json").read_text())
        package_changes.append({"source_id": name,
            "controller_changed": refreshed["files"][original.METHOD] != package["files"][original.METHOD],
            "declarations_unchanged": True, "numeric_payload_unchanged": True})
    protocol = deepcopy(prior)
    protocol.update(arms=["frozen_numeric"], planned_executions=len(prior["cases"]),
        intervention="After fixed numerical members are refuted, resume the existing parent fit stage before wider response-family research.",
        original_protocol=str(SOURCE / "protocol.json"),
        reused_controls={arm: str(SOURCE / "brief.json") for arm in original.ARMS},
        reused_forecasts=str(SOURCE / "forecast_summary.json"),
        source_packages=package_changes,
        comparison_scope="Same consumed development panel; unchanged no-archive and old numeric runs are reused, not newly rerun.")
    reused = []
    for case in prior["cases"]:
        for arm in original.ARMS:
            path = SOURCE / case / arm / "brief.json"
            row = read(path)
            assert row["case"] == case and row["arm"] == arm
            reused.append({"case": case, "arm": arm, "brief": str(path),
                "action_trace": str(path.with_name("action_trace.json")),
                "success": row["success"], "status": row["status"]})
    for name in protocol["runtime_snapshot"]:
        path = output / "runtime" / (name + ".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text((ROOT / "src/hegel_machine/cognition" / path.name).read_text())
    (output / "runtime" / "run_response_matched_adoption.py").write_text(
        (ROOT / "scripts/run_response_matched_adoption.py").read_text())
    (output / Path(__file__).name).write_text(Path(__file__).read_text())
    write(output / "protocol.json", protocol)
    write(output / "reused_outcomes.json", reused)
    summarize(output)
    print(json.dumps({"status": "prepared", "new_executions": len(prior["cases"]),
        "reused_executions": len(reused), "forecasts_rerun": False}), flush=True)


def preservation(case, after):
    before = SOURCE / case / "frozen_numeric"
    old, new = read(before / "brief.json"), read(after / "brief.json")
    old_actions, new_actions = read(before / "action_trace.json"), read(after / "action_trace.json")
    old_score = read(before / "research_trace.json")["score"]
    new_score = read(after / "research_trace.json")["score"]
    endpoints = ("success", "status", "requested_point_confirmed", "point_confirmations",
                 "unsupported_positive_verdicts")
    checks = {"action_trace_unchanged": old_actions == new_actions,
        "score_unchanged": old_score == new_score,
        "confirmation_endpoints_unchanged": all(old[key] == new[key] for key in endpoints),
        "research_compute_unchanged": old["costs"]["research_compute_units"] == new["costs"]["research_compute_units"]}
    outcome = "passed" if all(checks.values()) else "failed"
    first_difference = next((index for index, (a, b) in enumerate(zip(old_actions, new_actions)) if a != b), None)
    if first_difference is None and len(old_actions) != len(new_actions):
        first_difference = min(len(old_actions), len(new_actions))
    return {"case": case, "assertion": outcome, **checks, "first_action_difference": first_difference,
        "before_actions": len(old_actions), "after_actions": len(new_actions)}


def summarize(output):
    protocol = read(output / "protocol.json")
    rows = [read(output / case / "frozen_numeric" / "brief.json") for case in protocol["cases"]
            if (output / case / "frozen_numeric" / "brief.json").exists()]
    original_rows = [read(SOURCE / case / arm / "brief.json")
        for case in protocol["cases"] for arm in original.ARMS]
    old = original.summarize(original_rows, read(SOURCE / "protocol.json"))
    new = original.summarize(rows, protocol)
    conditions = {condition: {"basis_only": old["conditions"][condition]["basis_only"],
        "frozen_numeric_before": old["conditions"][condition]["frozen_numeric"],
        "frozen_numeric_after": new["conditions"][condition]["frozen_numeric"]}
        for condition in new["conditions"]}
    preservation_results = [preservation(row["case"], output / row["case"] / "frozen_numeric")
        for row in rows if row["condition"] == "same_mechanism"]
    errors_path = output / "run_errors.json"
    errors = read(errors_path) if errors_path.exists() else []
    summary = {key: new[key] for key in ("completed_executions", "planned_executions", "status")}
    summary.update(conditions=conditions, reused_executions=len(original_rows),
        unchanged_forecasts=str(SOURCE / "forecast_summary.json"),
        same_mechanism_preservation=preservation_results, run_errors=errors,
        execution_statuses={status: sum(row["status"] == status for row in rows)
            for status in sorted({row["status"] for row in rows})})
    write(output / "brief.json", rows)
    write(output / "summary.json", summary)
    return summary


def run(output, workers):
    protocol = read(output / "protocol.json")
    cases = list(protocol["cases"])
    if any((output / case / "frozen_numeric").exists() for case in cases):
        raise ValueError("Preserve the existing batch; this directory has started runs")
    errors = []

    def launch(case):
        log = output / "logs" / (case + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as stream:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                "--output", str(output), "--case", case], cwd=ROOT,
                stdout=stream, stderr=subprocess.STDOUT)
        return result.returncode, str(log)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(launch, case): case for case in cases}
        for future in as_completed(pending):
            case = pending[future]
            returncode, log = future.result()
            if returncode:
                errors.append({"case": case, "returncode": returncode, "log": log})
            write(output / "run_errors.json", errors)
            summary = summarize(output)
            print(json.dumps({"case": case, "returncode": returncode,
                "completed": summary["completed_executions"], "planned": len(cases)}), flush=True)
    summary = summarize(output)
    if errors:
        summary["status"] = "finished_with_errors"
        write(output / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--case")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if arguments.prepare_only:
        prepare(output)
    elif arguments.case:
        original.run_case(output, arguments.case, "frozen_numeric")
    elif arguments.run:
        run(output, arguments.workers)
    elif arguments.summarize:
        print(json.dumps(summarize(output), indent=2))
    else:
        parser.error("Choose --prepare-only, --run, --summarize or --case")
