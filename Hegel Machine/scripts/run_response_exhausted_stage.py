#!/usr/bin/env python3
"""Paired stage repair on the fixed 78 numerical-knowledge successors."""

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]
from hegel_machine.cognition.code_agent import _copy_model_file
from scripts import run_response_revision_successors as previous

SOURCE = previous.OUTPUT
OUTPUT = ROOT / "artifacts/response_exhausted_stage_20261011"
PILOT = "cubic_shifted_1__gain_lower__original_law_return"
ARMS = previous.ARMS
read, write = previous.read, previous.write


def prepare(output):
    old = read(SOURCE / "protocol.json")
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"baseline": str(SOURCE), "cases": old["successor_cases"],
        "arms": list(ARMS), "limits": old["limits"], "planned_executions": 78,
        "pilot": PILOT, "pilot_executions": 3,
        "cohort": "All 26 prior conditions and three knowledge arms, without outcome filtering.",
        "intervention": "Runtime stage selection after frozen-response feedback; refresh embedded stage helpers using existing prepare_bundle. Exact numerical knowledge, scientific declarations, methods metadata and resource limits are unchanged.",
        "baseline_reuse": "Use the completed 78 previous outputs as paired controls; do not rerun them.",
        "runtime": list(previous.MODULES),
        "pilot_reuse": "Run all three pilot arms first; run the other 75 using the same frozen runtime and include the pilot only once.",
        "cost_scope": "Charged research compute and environment actions are paired; final diagnostic export costs remain separate.",
        "scope": "Thirteen consumed revision branches from nine development sources; not independent mechanisms or LLM adoption.",
        "model_calls": 0}
    write(output / "protocol.json", protocol)
    for arm in ARMS:
        shutil.copytree(SOURCE / "successors" / arm / "sources", output / "after" / arm / "sources",
            copy_function=_copy_model_file)
        shutil.copyfile(SOURCE / "successors" / arm / "protocol.json", output / "after" / arm / "protocol.json")
    (output / Path(__file__).name).write_text(Path(__file__).read_text())
    summarize(output)
    print(json.dumps({"status": "prepared", "executions": 78, "pilot": PILOT}), flush=True)


def freeze_runtime(output):
    from scripts.run_response_matched_adoption import prepare_bundle
    from scripts.run_response_composition_transfer import declarations

    directory = output / "runtime"
    directory.mkdir(exist_ok=False)
    for module in previous.MODULES:
        shutil.copyfile(ROOT / "src/hegel_machine/cognition" / (module + ".py"), directory / (module + ".py"))
    # The execution harness is unchanged, but retain its exact version alongside
    # the stage modules to document the pilot/full-batch interface.
    shutil.copyfile(ROOT / "scripts/run_response_numeric_transfer.py", output / "run_response_numeric_transfer.py")
    changes = []
    for arm in ARMS:
        for path in sorted((output / "after" / arm / "sources").glob("*/numeric_bundle.json")):
            old = read(path)
            new = prepare_bundle(old)
            method = previous.original.METHOD
            before, after = old["files"][method], new["files"][method]
            assert declarations(before) == declarations(after)
            assert {k: v for k, v in old.items() if k != "files"} == {k: v for k, v in new.items() if k != "files"}
            assert {k: v for k, v in old["files"].items() if k != method} == {k: v for k, v in new["files"].items() if k != method}
            old_ast, new_ast = ast.parse(before), ast.parse(after)
            functions = lambda tree: {node.name: ast.dump(node) for node in tree.body if isinstance(node, ast.FunctionDef)}
            old_functions, new_functions = functions(old_ast), functions(new_ast)
            changed = [name for name in old_functions.keys() | new_functions.keys() if old_functions.get(name) != new_functions.get(name)]
            assert set(changed) <= {"_extend_hypotheses", "_point_needs_response"}, changed
            assert [ast.dump(n) for n in old_ast.body if not isinstance(n, ast.FunctionDef)] == [ast.dump(n) for n in new_ast.body if not isinstance(n, ast.FunctionDef)]
            write(path, new)
            changes.append({"arm": arm, "source": path.parent.name, "changed_functions": sorted(changed),
                "knowledge_declarations_methods_unchanged": True})
    write(output / "bundle_refresh.json", changes)
    print(json.dumps({"status": "runtime_frozen", "modules": len(previous.MODULES)}), flush=True)


def location(output, arm, case, before=False):
    return ((SOURCE / "successors") if before else (output / "after")) / arm / case / "frozen_numeric"


def stage_pattern(directory):
    trace = read(directory / "research_trace.json")
    questions = trace["questions"]
    ids = [q["id"] for q in questions]
    events = []
    for comparison in trace["comparisons"]:
        transfer = comparison.get("frozen_transfer", {})
        if transfer.get("status") != "exhausted":
            continue
        qid = comparison["question_id"]
        later = questions[ids.index(qid) + 1:] if qid in ids else []
        next_research = next((q for q in later if q["id"].startswith(("reference_fit_", "reference_response_"))
                              and not q["id"].endswith("_physical")), None)
        events.append({"question_id": qid, "stage": transfer.get("research_stage"),
            "retained_count": transfer.get("retained_count"), "status": comparison["status"],
            "compute_cost": comparison["compute_cost"],
            "selected_experiment": bool(comparison.get("selected_experiment")),
            "next_research": next_research["id"] if next_research else None,
            "source_counts": transfer.get("source_counts", {})})
    return {"comparisons": len(trace["comparisons"]),
        "stages": dict(Counter(c.get("frozen_transfer", {}).get("research_stage", "full_research")
                               for c in trace["comparisons"])),
        "compatible_exhausted_events": events}


def summarize(output):
    protocol = read(output / "protocol.json")
    before_rows, after_rows, pairs = [], [], []
    for case, config in protocol["cases"].items():
        for arm in ARMS:
            old_path, new_path = location(output, arm, case, True), location(output, arm, case)
            old = {**read(old_path / "brief.json"), "arm": arm}
            before_rows.append(old)
            if not (new_path / "brief.json").exists():
                continue
            new = {**read(new_path / "brief.json"), "arm": arm}
            after_rows.append(new)
            pairs.append({"case": case, "arm": arm, "condition": config["condition"],
                "success_before": old["success"], "success_after": new["success"],
                "requested_point_before": old["requested_point_confirmed"],
                "requested_point_after": new["requested_point_confirmed"],
                "actions_difference": new["costs"]["environment_steps"] - old["costs"]["environment_steps"],
                "compute_difference": new["costs"]["research_compute_units"] - old["costs"]["research_compute_units"],
                "actions_unchanged": read(old_path / "action_trace.json") == read(new_path / "action_trace.json"),
                "endpoints_unchanged": all(old[k] == new[k] for k in
                    ("success", "status", "point_confirmations", "requested_point_confirmed", "unsupported_positive_verdicts")),
                "stage_before": stage_pattern(old_path), "stage_after": stage_pattern(new_path)})
    completed = {(r["case"], r["arm"]) for r in after_rows}
    matched_before = [r for r in before_rows if (r["case"], r["arm"]) in completed]
    summary = {"status": "finished" if len(after_rows) == 78 else "running" if after_rows else "prepared",
        "completed_executions": len(after_rows), "planned_executions": 78,
        "pilot_completed": sum(r["case"] == PILOT for r in after_rows),
        "conditions": {condition: {arm: {
            "before": previous.aggregate([r for r in matched_before if r["condition"] == condition and r["arm"] == arm]),
            "after": previous.aggregate([r for r in after_rows if r["condition"] == condition and r["arm"] == arm])}
            for arm in ARMS} for condition in previous.CONDITIONS},
        "preservation": {"full_action_traces": sum(r["actions_unchanged"] for r in pairs),
            "endpoints": sum(r["endpoints_unchanged"] for r in pairs),
            "charged_compute": sum(r["compute_difference"] == 0 for r in pairs)}, "paired": pairs}
    write(output / "brief.json", after_rows)
    write(output / "summary.json", summary)
    return summary


def run(output, pilot, workers):
    protocol = read(output / "protocol.json")
    # This comparison answers whether the completed pilot still belongs to the
    # exact runtime under test. A changed implementation requires a new cohort.
    for module in previous.MODULES:
        if (output / "runtime" / (module + ".py")).read_text() != (ROOT / "src/hegel_machine/cognition" / (module + ".py")).read_text():
            raise ValueError("Runtime changed after freezing; preserve this cohort: " + module)
    if (output / "run_response_numeric_transfer.py").read_text() != (ROOT / "scripts/run_response_numeric_transfer.py").read_text():
        raise ValueError("Execution harness changed after freezing")
    jobs = [(arm, case) for case in protocol["cases"] for arm in ARMS
            if (case == PILOT if pilot else case != PILOT)]
    if not pilot and not all((location(output, arm, PILOT) / "brief.json").exists() for arm in ARMS):
        raise ValueError("Finish all three pilot arms before the remaining cohort")
    if any(location(output, arm, case).exists() for arm, case in jobs):
        raise ValueError("Preserve the already-started cohort")
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
            write(output / ("pilot_errors.json" if pilot else "batch_errors.json"), errors)
            summary = summarize(output)
            print(json.dumps({"case": case, "arm": arm, "returncode": returncode,
                "completed": summary["completed_executions"]}), flush=True)
    if errors:
        raise SystemExit("Batch finished with recorded errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--freeze-runtime", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--case")
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.case and args.arm:
        previous.load_runtime(output / "runtime")
        previous.original.run_case(output / "after" / args.arm, args.case, "frozen_numeric")
    elif args.prepare:
        prepare(output)
    elif args.freeze_runtime:
        freeze_runtime(output)
    elif args.pilot or args.run:
        run(output, args.pilot, args.workers)
    elif args.summarize:
        print(json.dumps(summarize(output), indent=2))
    else:
        parser.error("Choose prepare, freeze-runtime, pilot, run or summarize")
