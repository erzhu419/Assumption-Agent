#!/usr/bin/env python3
"""Pair the old and repaired response-routing predicate on the existing six cases."""

import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]

from scripts import consolidate_response_measurement as study
from scripts import run_response_incumbent_frozen_transfer as frozen
from scripts.run_response_structure_transfer import source_parts
from hegel_machine.cognition import world_reference_recalibration as recipe

SOURCE = ROOT / "artifacts/response_incumbent_frozen_transfer_20261010/frozen_bundle.json"
OUTPUT = ROOT / "artifacts/response_cross_target_routing_20261010"
OLD_FUNCTION = OUTPUT / "old_point_needs_response.py"
ARMS = ("old_routing", "cross_target_routing")
FUNCTION = "_point_needs_response"


def function_node(source):
    return next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == FUNCTION)


def replace_function(source, replacement):
    node = function_node(source)
    lines = source.splitlines(keepends=True)
    revised = "".join(lines[:node.lineno - 1]) + replacement.rstrip() + "\n" + "".join(lines[node.end_lineno:])
    other_nodes = lambda code: [ast.dump(item, include_attributes=False) for item in ast.parse(code).body
        if not (isinstance(item, ast.FunctionDef) and item.name == FUNCTION)]
    assert other_nodes(source) == other_nodes(revised), "Only the routing predicate may differ"
    assert source_parts(source)[0] == source_parts(revised)[0], "Retain all 16 scientific declarations"
    compile(revised, "frozen_response_method.py", "exec")
    return revised


def routing_trace(steps, state):
    """Recover actual point verdicts preceding each response call, in event order."""
    tests = {item["question"]: item for item in state.get("tests", [])}
    if state.get("test"):
        tests[state["test"]["question"]] = state["test"]
    outcomes, calls, last_index, active_targets = {}, [], -1, []
    for step in steps:
        feedback = step.get("feedback", {})
        indices = feedback.get("observation_indices", [])
        if indices:
            last_index = max(last_index, max(indices))
        if feedback.get("action_results"):
            objects = feedback["action_results"][-1].get("public_objects", [])
            active_targets = [number for number in state.get("target_numbers", [])
                if any(item.get("uuid") == 100 + number and "(activated)" in item.get("name", "")
                       for item in objects)]
        for entry in [feedback, *feedback.get("after_observation_results", [])]:
            question = entry.get("question", {})
            test = tests.get(question.get("id"))
            if test and question.get("status") in ("supported", "refuted", "inconclusive"):
                outcomes[question["id"]] = {
                    "question_id": question["id"], "number": test["number"],
                    "family": test["family"], "prediction": test["prediction"],
                    "response_question": test.get("response_question"),
                    "authored_experiment": bool(test.get("authored_experiment")),
                    "restore_frequency": test.get("restore_frequency"),
                    "status": question["status"], "evidence": question.get("evidence", []),
                    "method_step": step["step"], "observations": last_index + 1}
        arguments = step.get("proposal", {}).get("arguments", step.get("arguments", {}))
        if arguments.get("operator") != "reference_response":
            continue
        number = arguments.get("context", {}).get("number", feedback.get("target_number"))
        point_failures = [deepcopy(value) for value in outcomes.values()
            if value["number"] == number and value["status"] == "refuted"
            and not value["response_question"]]
        calls.append({"method_step": step["step"], "observations": last_index + 1,
            "requested_target": number, "response_count": feedback.get("response_count"),
            "status": feedback.get("status"), "point_refutations_before_call": point_failures,
            "prior_other_target_support": [deepcopy(value) for value in outcomes.values()
                if value["number"] != number and value["status"] == "supported"],
            "other_targets_currently_activated": [target for target in active_targets if target != number],
            "prior_other_target_response_trials": [deepcopy(value) for value in outcomes.values()
                if value["number"] != number and value["response_question"]],
            "steps_since_first_point_refutation": (step["step"] - min(
                value["method_step"] for value in point_failures)) if point_failures else None,
            "selected_number": feedback.get("selected_number"),
            "selected_setting": feedback.get("selected_setting")})
    return {"point_and_response_outcomes": list(outcomes.values()), "response_calls": calls,
        "scope": "Ordered saved execution feedback; no future verdict is used to label an earlier response call."}


def run_case(case, config, arm, package, output, limits):
    study.CASES = {case: config}
    with TemporaryDirectory(prefix="hegel-cross-target-") as temporary:
        row = study.run_transfer_case(case, arm, package, temporary, limits)
        directory = Path(temporary) / case / arm
        result = json.loads((directory / "result.json").read_text())
        state = json.loads((directory / "agent/shared/reference_recalibration.json").read_text())
        steps = result["execution"].get("steps", [])
        trace = routing_trace(steps, state)
        history = [frozen.compact_comparison(item) for item in state.get("response_research_history", [])]
        scopes = []
        for item in history:
            revision = item["joint_revision"]
            if revision and revision["scope"] not in scopes:
                scopes.append(revision["scope"])
        source = (directory / "agent" / study.METHOD_PATH).read_text()
        declarations, _ = source_parts(source)
        row.update(group=config["group"], procedure_steps=len(steps),
            response_calls=len(trace["response_calls"]),
            response_calls_by_target=dict(Counter(str(item["requested_target"]) for item in trace["response_calls"])),
            joint_revision_scopes=len(scopes), final_declaration_count=len(declarations))
        destination = Path(output) / case / arm
        frozen.write(destination / "brief.json", row)
        current_state = {**state, "response_research_history": history}
        frozen.write(destination / "current_state.json", current_state)
        frozen.write(destination / "routing_trace.json", trace)
        frozen.write(destination / "research_trace.json", {
            "comparisons": history, "trials": state.get("tests", []), "pending_test": state.get("test"),
            "response_settings": state.get("response_settings", []), "score": result["score"],
            "execution": {key: value for key, value in result["execution"].items() if key != "steps"},
            "procedure_steps": len(steps),
            "last_public_observation": result.get("observations", [])[-1] if result.get("observations") else None})
        frozen.write(destination / "action_trace.json", [
            {key: record.get(key) for key in ("index", "action", "feedback")}
            for record in result.get("observations", [])])
        if not row["source_unchanged"]:
            (destination / "final_source.py").write_text(source)
        return row


def summarize(rows):
    arms = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        arms[arm] = {"cases": len(selected), "successes": sum(row["success"] for row in selected),
            "statuses": dict(Counter(row["status"] for row in selected)),
            "source_unchanged": sum(row["source_unchanged"] for row in selected),
            "joint_revision_scopes": sum(row["joint_revision_scopes"] for row in selected),
            "costs": {key: sum(row["costs"].get(key, 0) for row in selected) for key in
                ("environment_steps", "research_compute_units", "python_execution_seconds", "model_calls")}}
    return {"completed_executions": len(rows), "planned_executions": 12, "arms": arms}


def run(output=OUTPUT):
    output = Path(output).resolve()
    previous = json.loads(SOURCE.read_text())
    source = previous["files"][study.METHOD_PATH]
    old = OLD_FUNCTION.read_text()
    assert ast.dump(function_node(source)) == ast.dump(function_node(old)), "Archived old predicate must match the original bundle"
    current_recipe = Path(recipe.__file__).read_text()
    revised = ast.get_source_segment(current_recipe, function_node(current_recipe)) + "\n"
    if ast.dump(function_node(old)) == ast.dump(function_node(revised)):
        raise ValueError("The production routing predicate is still unchanged; do not run identical arms")
    packages = {ARMS[0]: deepcopy(previous), ARMS[1]: deepcopy(previous)}
    packages[ARMS[1]]["files"][study.METHOD_PATH] = replace_function(source, revised)
    panel = frozen.cases()
    limits = {**json.loads(frozen.LIMITS_FROM.read_text())["limits"], "max_model_calls": 0}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise ValueError("Use a new output directory; retain previous fixed-protocol results")
    for arm, package in packages.items():
        frozen.write(output / (arm + "_bundle.json"), package)
    (output / "old_point_needs_response.py").write_text(old)
    (output / "new_point_needs_response.py").write_text(revised)
    frozen.write(output / "protocol.json", {"source_bundle": str(SOURCE), "cases": panel, "arms": ARMS,
        "limits": limits, "procedure_max_steps": 256, "planned_executions": 12,
        "intervention": "Only the embedded _point_needs_response function differs; all other source AST statements and all 16 declarations are identical.",
        "runtime": "Both arms use the current shared runtime and TransferSignalLab with completion refreshed after every action.",
        "scope": "Same six already-used parameter interpolation conditions. Explicit host-controller repair, no learned strategy or unseen-task claim; zero LLM, no source state or fitted target parameters installed.",
        "historical_results": str(frozen.OUTPUT),
        "historical_boundary": "Earlier scores used the old completion fixture. They remain unchanged and are not the matched baseline for this comparison."})
    order = {(case, arm): index for index, (case, arm) in enumerate(
        (case, arm) for case in panel for arm in ARMS)}
    rows = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run_case, case, config, arm, packages[arm], output, limits): (case, arm)
            for case, config in panel.items() for arm in ARMS}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            rows.sort(key=lambda value: order[value["case"], value["arm"]])
            frozen.write(output / "brief.json", rows)
            frozen.write(output / "summary.json", summarize(rows))
            print(json.dumps(row), flush=True)
    pairs = []
    for case in panel:
        selected = [next(row for row in rows if row["case"] == case and row["arm"] == arm) for arm in ARMS]
        actions = [[item["action"] for item in json.loads((output / case / arm / "action_trace.json").read_text())] for arm in ARMS]
        pairs.append({"case": case, "rows": selected, "identical_action_sequence": actions[0] == actions[1]})
    frozen.write(output / "paired_summary.json", {"arms": ARMS, "pairs": pairs})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output)
