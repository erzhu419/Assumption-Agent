#!/usr/bin/env python3
"""Paired stage repair when initially usable numeric knowledge meets later data."""

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
OUTPUT = ROOT / "artifacts/response_late_withdrawal_20261011"
PRESERVED = ROOT / "artifacts/response_transfer_stage_resume_20261011"
MODULES = ("world_reference_recalibration", "world_reference_response",
           "world_response_hypotheses", "world_response_transfer")
FACTORS = {"gain_lower": .999, "gain_higher": 1.001}
write = original.write


def read(path):
    return json.loads(path.read_text())


def prepare_phase(output, phase, protocol):
    from scripts.run_response_matched_adoption import prepare_bundle
    from scripts.run_response_composition_transfer import declarations

    directory = output / phase
    directory.mkdir(parents=True, exist_ok=False)
    for name in protocol["source_models"]:
        package = read(SOURCE / "sources" / name / "numeric_bundle.json")
        refreshed = prepare_bundle(package)
        assert declarations(refreshed["files"][original.METHOD]) == declarations(package["files"][original.METHOD])
        assert refreshed["files"][original.KNOWLEDGE] == package["files"][original.KNOWLEDGE]
        assert refreshed["methods"] == package["methods"]
        write(directory / "sources" / name / "numeric_bundle.json", refreshed)
    for module in MODULES:
        target = directory / "runtime" / (module + ".py")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((ROOT / "src/hegel_machine/cognition" / target.name).read_text())
    write(directory / "protocol.json", {**protocol, "phase": phase,
        "planned_executions": len(protocol["cases"]), "runtime_snapshot": list(MODULES)})
    summarize(output)


def prepare(output):
    prior = read(SOURCE / "protocol.json")
    cases = {}
    for name in original.SOURCES:
        source = prior["cases"][name + "__same_mechanism"]
        for condition, factor in FACTORS.items():
            config = deepcopy(source)
            config.update(gain=source["gain"] * factor, condition=condition, gain_factor=factor)
            cases[name + "__" + condition] = config
    protocol = {"cases": cases, "arms": ["frozen_numeric"],
        "source_models": prior["source_models"], "limits": prior["limits"],
        "planned_executions": 2 * len(cases), "phases": ["before", "after"],
        "source_protocol": str(SOURCE / "protocol.json"), "gain_factors": FACTORS,
        "new_target_signals": [5.25, 12.5],
        "intervention": "Retain frozen knowledge and declarations; change only withdrawal-stage routing between phases.",
        "scope": "Fixed nine consumed source archives crossed with two small static gain changes; all conditions retained, including no late refutation.",
        "phase_execution": "Finish all before runs before runtime edits; prepare-after refreshes only the controller and limited runtime snapshots.",
        "preservation_reference": str(ROOT / "artifacts/response_transfer_stage_resume_20261011/summary.json"),
        "storage": "Reuse numeric.run_case compact research/action traces; no persistent full agent snapshots.",
        "model_calls": 0}
    output.mkdir(parents=True, exist_ok=False)
    write(output / "protocol.json", protocol)
    (output / Path(__file__).name).write_text(Path(__file__).read_text())
    prepare_phase(output, "before", protocol)
    print(json.dumps({"status": "prepared", "before_cases": len(cases), "after_cases": len(cases)}), flush=True)


def trajectory(directory):
    row, trace = read(directory / "brief.json"), read(directory / "research_trace.json")
    questions = trace["questions"]
    by_id = {q["id"]: q for q in questions}
    positions = {q["id"]: i for i, q in enumerate(questions)}
    comparisons = trace["comparisons"]
    selected = False
    transition = None
    for comparison in comparisons:
        status = comparison.get("frozen_transfer", {}).get("status")
        if status == "refuted" and selected:
            transition = comparison
            break
        selected |= status == "selected"
    summary = {"case": row["case"], "condition": row["condition"],
        "success": row["success"], "status": row["status"],
        "requested_point_confirmed": row["requested_point_confirmed"],
        "unsupported_positive_verdicts": row["unsupported_positive_verdicts"],
        "costs": row["costs"], "selected_then_refuted": transition is not None,
        "archive_statuses": [c.get("frozen_transfer", {}).get("status") for c in comparisons]}
    if transition is None:
        return summary
    qid = transition["question_id"]
    question, position = by_id[qid], positions[qid]
    earlier_successes = [point for point in row["point_confirmations"]
        if point["physically_confirmed"]
        and positions.get(point["question_id"], len(questions)) < position]
    later = [q for q in questions[position + 1:] if q["id"].startswith("reference_fit_")
        or q["id"] in {c["question_id"] for c in comparisons}]
    first_fit = next((q for q in later if q["id"].startswith("reference_fit_")), None)
    previous_fits = [q["id"] for q in questions[:position] if q["id"].startswith("reference_fit_")]
    family_in_same_call = "family_research" in question.get("answer", {})
    summary["withdrawal"] = {"question_id": qid,
        "opened_before_observation": question["opened_before_observation"],
        "compute_cost": transition["compute_cost"],
        "full_family_research_in_same_call": family_in_same_call,
        "earlier_confirmed_points": earlier_successes, "fit_questions_before_withdrawal": previous_fits,
        "first_later_research_question": later[0]["id"] if later else None,
        "first_later_fit_question": first_fit["id"] if first_fit else None,
        "full_family_research_after_withdrawal": [q["id"] for q in later
            if "family_research" in q.get("answer", {})]}
    return summary


def aggregate(rows):
    keys = ("environment_steps", "research_compute_units", "python_execution_seconds", "model_calls", "reported_tokens")
    return {"cases": len(rows), "successes": sum(row["success"] for row in rows),
        "requested_point_confirmed": {str(n): sum(row["requested_point_confirmed"][str(n)] for row in rows) for n in (4, 5)},
        "unsupported_positive_verdicts": sum(row["unsupported_positive_verdicts"] for row in rows),
        "late_refutations": sum(row["selected_then_refuted"] for row in rows),
        "late_refutations_after_confirmed_point": sum(bool(row.get("withdrawal", {}).get("earlier_confirmed_points")) for row in rows),
        "costs": {key: sum(row["costs"].get(key, 0) for row in rows) for key in keys}}


def summarize(output):
    protocol = read(output / "protocol.json")
    phases, all_rows = {}, {}
    for phase in protocol["phases"]:
        directory = output / phase
        rows = [trajectory(directory / case / "frozen_numeric") for case in protocol["cases"]
            if (directory / case / "frozen_numeric/brief.json").exists()]
        all_rows[phase] = rows
        phases[phase] = {"aggregate": aggregate(rows),
            "conditions": {condition: aggregate([row for row in rows if row["condition"] == condition]) for condition in FACTORS}}
        if directory.exists():
            write(directory / "brief.json", rows)
    count = sum(len(rows) for rows in all_rows.values())
    paired = []
    before = {row["case"]: row for row in all_rows["before"]}
    for row in all_rows["after"]:
        if row["case"] in before:
            old = before[row["case"]]
            paired.append({"case": row["case"], "success_before": old["success"], "success_after": row["success"],
                "requested_point_before": old["requested_point_confirmed"], "requested_point_after": row["requested_point_confirmed"],
                "actions_difference": row["costs"]["environment_steps"] - old["costs"]["environment_steps"],
                "compute_difference": row["costs"]["research_compute_units"] - old["costs"]["research_compute_units"]})
    summary = {"status": "finished" if count == protocol["planned_executions"] else "running" if count else "prepared",
        "completed_executions": count, "planned_executions": protocol["planned_executions"],
        "phases": phases, "paired": paired}
    write(output / "summary.json", summary)
    return summary


def preservation_summary(output):
    directory = output / "preservation"
    protocol = read(directory / "protocol.json")
    rows = []
    for case in protocol["cases"]:
        current, previous = directory / case / "frozen_numeric", PRESERVED / case / "frozen_numeric"
        if not (current / "brief.json").exists():
            continue
        old, new = read(previous / "brief.json"), read(current / "brief.json")
        endpoints = ("success", "status", "requested_point_confirmed", "point_confirmations", "unsupported_positive_verdicts")
        checks = {"actions_unchanged": read(previous / "action_trace.json") == read(current / "action_trace.json"),
            "score_unchanged": read(previous / "research_trace.json")["score"] == read(current / "research_trace.json")["score"],
            "endpoints_unchanged": all(old[key] == new[key] for key in endpoints),
            "compute_unchanged": old["costs"]["research_compute_units"] == new["costs"]["research_compute_units"]}
        rows.append({"case": case, "preserved": all(checks.values()), **checks,
            "costs_before": old["costs"], "costs_after": new["costs"]})
    summary = {"status": "finished" if len(rows) == len(protocol["cases"]) else "running",
        "completed_executions": len(rows), "planned_executions": len(protocol["cases"]),
        "source": str(PRESERVED), "preserved_cases": sum(row["preserved"] for row in rows), "cases": rows}
    write(directory / "summary.json", summary)
    return summary


def preserve(output, workers):
    previous = read(PRESERVED / "protocol.json")
    cases = {name: case for name, case in previous["cases"].items() if case["condition"] == "same_mechanism"}
    protocol = {**previous, "cases": cases, "arms": ["frozen_numeric"],
        "intervention": "After-stage runtime on the unchanged nine same-mechanism conditions; exact prior outcomes retained.",
        "source_protocol": str(PRESERVED / "protocol.json")}
    prepare_phase(output, "preservation", protocol)
    run(output, "preservation", workers)


def run(output, phase, workers):
    directory = output / phase
    protocol = read(directory / "protocol.json")
    cases = list(protocol["cases"])
    if any((directory / case / "frozen_numeric").exists() for case in cases):
        raise ValueError("Preserve the existing phase; this batch has started")
    errors = []

    def launch(case):
        log = directory / "logs" / (case + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as stream:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                "--output", str(output), "--phase", phase, "--case", case],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        return result.returncode, str(log)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(launch, case): case for case in cases}
        for future in as_completed(pending):
            case = pending[future]
            returncode, log = future.result()
            if returncode:
                errors.append({"case": case, "returncode": returncode, "log": log})
            write(directory / "run_errors.json", errors)
            summary = preservation_summary(output) if phase == "preservation" else summarize(output)
            completed = (summary["completed_executions"] if phase == "preservation"
                         else summary["phases"][phase]["aggregate"]["cases"])
            print(json.dumps({"phase": phase, "case": case, "returncode": returncode,
                "completed": completed}), flush=True)
    if errors:
        raise SystemExit("Phase finished with errors; see run_errors.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--prepare-after", action="store_true")
    parser.add_argument("--preserve", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--phase", choices=("before", "after", "preservation"), default="before")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--case")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.prepare:
        prepare(output)
    elif args.prepare_after:
        protocol = read(output / "protocol.json")
        if any(not (output / "before" / case / "frozen_numeric/brief.json").exists() for case in protocol["cases"]):
            raise ValueError("Finish the before phase before preparing after")
        prepare_phase(output, "after", protocol)
    elif args.preserve:
        preserve(output, args.workers)
    elif args.case:
        original.run_case(output / args.phase, args.case, "frozen_numeric")
    elif args.run:
        run(output, args.phase, args.workers)
    elif args.summarize:
        print(json.dumps(preservation_summary(output) if args.phase == "preservation" else summarize(output), indent=2))
    else:
        parser.error("Choose --prepare, --prepare-after, --preserve, --run, --summarize, or --case")
