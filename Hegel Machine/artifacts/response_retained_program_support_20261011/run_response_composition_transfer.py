#!/usr/bin/env python3
"""Compare archived composition runtimes on the existing twelve conditions."""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]
OUTPUT = ROOT / "artifacts/response_composition_transfer_20261011"
HISTORICAL = ROOT / "artifacts/response_structure_transfer_20261009"
ARMS = ("before", "auto_composition")
METHOD = "shared/consolidated_method.py"


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def declarations(source):
    return next(ast.literal_eval(node.value) for node in ast.parse(source).body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
            and target.id == "ESTIMATORS" for target in node.targets))


def runtime(arm):
    # Each invocation is a fresh child process; load before static agent imports.
    from scripts.diagnose_response_research_cycle import load_runtime
    selected = load_runtime(arm)
    from scripts import run_response_matched_adoption as matched
    from hegel_machine.cognition import discoveryworld_agent as agent_module
    selected["static_bindings_match"] = {
        "matched_recipe": matched.recipe is sys.modules["hegel_machine.cognition.world_reference_recalibration"],
        "agent_expansion": agent_module.execute_reference_expansion is
            sys.modules["hegel_machine.cognition.world_reference_expansion"].execute_reference_expansion,
        "agent_response": agent_module.execute_reference_response is
            sys.modules["hegel_machine.cognition.world_reference_response"].execute_reference_response}
    assert all(selected["static_bindings_match"].values()), "Runtime imported after static bindings"
    return selected, matched


def prepare_arm(output, arm):
    selected, matched = runtime(arm)
    original = json.loads((HISTORICAL / "without_cubic_bundle.json").read_text())
    prepared = matched.prepare_bundle(original)
    code = prepared["files"][METHOD]
    assert declarations(code) == declarations(original["files"][METHOD])
    assert len(declarations(code)) == 9
    selected["controller_has_basis_expansion_action"] = "basis_expansion_action" in code
    assert selected["controller_has_basis_expansion_action"] == (arm == "auto_composition")
    compile(code, str(output / f"{arm}_controller.py"), "exec")
    write(output / f"{arm}_bundle.json", prepared)
    write(output / f"{arm}_runtime.json", selected)
    (output / f"{arm}_controller.py").write_text(code)


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    # The archived protocol contains exactly successor_cases(), without importing
    # its agent dependencies into this orchestration process.
    previous = json.loads((HISTORICAL / "protocol.json").read_text())
    for arm in ARMS:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--output", str(output),
                        "--prepare-arm", arm], check=True)
    bundles = {arm: json.loads((output / f"{arm}_bundle.json").read_text()) for arm in ARMS}
    original = declarations(json.loads((HISTORICAL / "without_cubic_bundle.json").read_text())["files"][METHOD])
    assert len(original) == 9 and not any(item["id"] == "signal_cubic_5" for item in original)
    assert all(declarations(bundle["files"][METHOD]) == original for bundle in bundles.values())
    limits = {**previous["limits"], "environment_steps": 512, "max_model_calls": 0,
        "python_seconds_per_call": 30, "python_seconds_per_task": 240, "task_wall_seconds": 3600}
    protocol = {"cases": previous["cases"], "arms": list(ARMS), "planned_executions": 24,
        "source_bundle": str(HISTORICAL / "without_cubic_bundle.json"), "initial_declarations": original,
        "limits": limits, "procedure_max_steps": 256,
        "intervention": "Archived pre-composition versus candidate response integration modules, with matching nine original scientific declarations; embedded parent extension refreshed from its selected runtime.",
        "runtime": "Four response integration modules are archived by the prior cycle; shared dependencies and the immediate-completion fixture use this checkout. Each arm runs in an independent process.",
        "budget_change": "Single Python call allowance 5 to 30 seconds for both arms, preserving the original 240-second cumulative allowance and 512-action/256-step limits.",
        "scope": "Twelve fixed previously consumed development conditions, not random seeds or new held-out tasks. Zero LLM calls. Historical 24-run knowledge retention comparison is reference only, not this runtime baseline.",
        "confirmation": "Task completion, requested-point confirmation, and previously unobserved positive confirmation are separate. Incidental activation while approaching an untested requested point is not its confirmation.",
        "storage": "Keep compact state, complete action list, final source, and raw observations cited by point questions or composition checks plus the final public observation. No full repeated agent snapshots."}
    write(output / "protocol.json", protocol)
    (output / Path(__file__).name).write_text(Path(__file__).read_text())
    write(output / "summary.json", summarize([], protocol))
    print(json.dumps({"status": "prepared", "output": str(output), "planned_executions": 24}), flush=True)


def point_evidence(result, state, matched):
    result["reference_state"] = {key: state[key] for key in ("tests", "test") if key in state}
    confirmations = matched.point_confirmations(result)
    trials = {trial["question"]: trial for trial in state.get("tests", [])}
    if state.get("test"):
        trials[state["test"]["question"]] = state["test"]
    for row in confirmations:
        trial = trials[row["question_id"]]
        previous = []
        for record in result.get("observations", []):
            if record["index"] >= trial["started"]:
                continue
            view = matched.recipe._view(record)
            number, setting = matched.recipe._dialog(view)
            objects = matched.recipe._objects(view)
            if (number == row["number"] and setting is not None
                    and abs(setting - row["prediction"]) <= row["controller_tolerance"]
                    and matched.recipe._installed(objects, trial["crystal"], trial["reactor"])
                    and matched.recipe._activated(objects, trial["reactor"])):
                previous.append({"index": record["index"], "setting": setting})
        row["previous_positive_evidence"] = previous
        row["previously_positive"] = bool(previous)
        row["new_positive_confirmation"] = row["physically_confirmed"] and not previous
    return confirmations


def compact_expansion(expansion):
    compact = deepcopy(expansion)
    checks = compact.get("paid_family_checks", {})
    if checks:
        first = next(iter(checks.values()))
        compact["paid_evidence"] = {key: deepcopy(first[key]) for key in ("features", "responses")}
        for check in checks.values():
            assert check["features"] == first["features"] and check["responses"] == first["responses"]
        for check in checks.values():
            check.pop("features")
            check.pop("responses")
    return compact


def run_case(output, case, arm):
    selected, matched = runtime(arm)
    from scripts.diagnose_response_research_cycle import compact_comparison
    protocol = json.loads((output / "protocol.json").read_text())
    package = json.loads((output / f"{arm}_bundle.json").read_text())
    assert selected == {key: value for key, value in json.loads(
        (output / f"{arm}_runtime.json").read_text()).items() if key != "controller_has_basis_expansion_action"}
    destination = output / case / arm
    destination.mkdir(parents=True, exist_ok=False)
    matched.study.CASES = {case: protocol["cases"][case]}
    with TemporaryDirectory(prefix="hegel-composition-transfer-") as temporary:
        row = matched.study.run_transfer_case(case, arm, package, temporary, protocol["limits"],
            on_checkpoint=lambda data: write(destination / "active.json", {
                "status": data["status"], "costs": data["costs"], "score": data["score"]}))
        directory = Path(temporary) / case / arm
        result = json.loads((directory / "result.json").read_text())
        state = json.loads((directory / "agent/shared/reference_recalibration.json").read_text())
        source = (directory / "agent" / METHOD).read_text()
        confirmations = point_evidence(result, state, matched)
        expansions = [compact_expansion(item) for item in state.get("basis_expansions", [])]
        composition = [item for item in expansions if item.get("composition_attempts")]
        questions = {question["id"]: question for question in result["research"]["questions"]}
        evidence = {index for question in questions.values() for index in question.get("evidence", [])}
        evidence.update(index for trial in state.get("tests", [])
            for index in trial.get("preparation_evidence", []))
        evidence.update(index for recovery in state.get("setting_recoveries", [])
            for index in recovery.get("evidence", []))
        evidence.update(entry["index"] for point in confirmations for entry in point["previous_positive_evidence"])
        evidence.update(index for expansion in composition
            for check in expansion["paid_family_checks"].values() for index in check.get("evidence", []))
        observations = result.get("observations", [])
        if observations:
            evidence.add(observations[-1]["index"])
        final_declarations = declarations(source)
        initial = declarations(package["files"][METHOD])
        row.update(group=protocol["cases"][case]["group"],
            procedure_steps=len(result["execution"].get("steps", [])),
            composition_attempts=len(composition),
            generated_composition_declarations=sum(len(item["generated_hypotheses"]) for item in composition),
            added_declarations=[item for item in final_declarations if item not in initial],
            removed_declarations=[item for item in initial if item not in final_declarations],
            point_confirmations=confirmations,
            requested_point_confirmed={str(number): any(point["number"] == number
                and point["physically_confirmed"] for point in confirmations) for number in (4, 5)},
            new_positive_confirmed={str(number): any(point["number"] == number
                and point["new_positive_confirmation"] for point in confirmations) for number in (4, 5)},
            unsupported_positive_verdicts=sum(point["system_supported"] and not point["physically_confirmed"]
                                              for point in confirmations))
        write(destination / "evidence_observations.json", [record for record in observations if record["index"] in evidence])
        write(destination / "action_trace.json", [{key: record.get(key) for key in ("index", "action", "feedback")}
                                                  for record in observations])
        write(destination / "research_trace.json", {
            "score": result["score"], "tests": state.get("tests", []), "pending_test": state.get("test"),
            "setting_recoveries": state.get("setting_recoveries", []),
            "pending_prepared_recovery": state.get("prepared_recovery"),
            "questions": questions, "research_table": state.get("research_table"),
            "response_settings": state.get("response_settings", []), "basis_expansions": expansions,
            "comparisons": [compact_comparison(item) for item in state.get("response_research_history", [])],
            "current_comparison": compact_comparison(state.get("response_research", {})),
            "current_fixed_program_candidates": [{key: deepcopy(candidate[key]) for key in
                ("id", "family", "program", "status", "fit_sources", "response_count",
                 "target_prediction", "positive_residual_span", "lower_boundary_range", "upper_boundary_range")
                if key in candidate}
                for candidate in state.get("response_research", {}).get("candidates", [])
                if any(source.get("origin") in ("retained_fixed_program", "fixed_program_response_support")
                    for source in candidate.get("fit_sources", []))],
            "execution": {key: value for key, value in result["execution"].items() if key != "steps"}})
        (destination / "final_controller.py").write_text(source)
        write(destination / "brief.json", row)
        write(destination / "active.json", {"finished": True, "status": row["status"], "success": row["success"]})
        print(json.dumps({key: row[key] for key in ("case", "arm", "status", "success", "requested_point_confirmed")}), flush=True)


def summarize(rows, protocol):
    return {"completed_executions": len(rows), "planned_executions": protocol["planned_executions"],
        "status": "finished" if len(rows) == protocol["planned_executions"] else "prepared" if not rows else "running",
        "arms": {arm: {"cases": len(selected), "successes": sum(row["success"] for row in selected),
            "requested_point_confirmed": {str(number): sum(row["requested_point_confirmed"][str(number)] for row in selected) for number in (4, 5)},
            "new_positive_confirmed": {str(number): sum(row["new_positive_confirmed"][str(number)] for row in selected) for number in (4, 5)},
            "composition_cases": sum(row["composition_attempts"] > 0 for row in selected),
            "unsupported_positive_verdicts": sum(row["unsupported_positive_verdicts"] for row in selected),
            "costs": {key: sum(row["costs"].get(key, 0) for row in selected) for key in
                ("environment_steps", "research_compute_units", "python_execution_seconds", "model_calls")}}
            for arm in protocol["arms"] for selected in [[row for row in rows if row["arm"] == arm]]}}


def run(output, workers, *, case_names=None, entrypoint=None):
    protocol = json.loads((output / "protocol.json").read_text())
    all_jobs = [(case, arm) for case in protocol["cases"] for arm in protocol["arms"]]
    jobs = [(case, arm) for case, arm in all_jobs if case_names is None or case in case_names]
    if any((output / case / arm).exists() for case, arm in jobs):
        raise ValueError("This prepared batch has already started; preserve its executions")
    rows = [json.loads((output / case / arm / "brief.json").read_text())
        for case, arm in all_jobs if (output / case / arm / "brief.json").exists()]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(subprocess.run, [sys.executable, str(entrypoint or Path(__file__).resolve()),
            "--output", str(output), "--case", case, "--arm", arm], check=True): (case, arm)
            for case, arm in jobs}
        for future in as_completed(pending):
            case, arm = pending[future]
            future.result()
            rows.append(json.loads((output / case / arm / "brief.json").read_text()))
            rows.sort(key=lambda row: all_jobs.index((row["case"], row["arm"])))
            write(output / "brief.json", rows)
            write(output / "summary.json", summarize(rows, protocol))
    write(output / "paired_summary.json", {"pairs": [{"case": case,
        "rows": [row for row in rows if row["case"] == case]} for case in protocol["cases"]]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--prepare-arm", choices=ARMS)
    parser.add_argument("--case")
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.prepare_arm:
        prepare_arm(output, args.prepare_arm)
    elif args.case and args.arm:
        run_case(output, args.case, args.arm)
    elif args.prepare_only:
        prepare(output)
    elif args.run:
        run(output, args.workers)
    else:
        parser.error("Choose --prepare-only or --run")
