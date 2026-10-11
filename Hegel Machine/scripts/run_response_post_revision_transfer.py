#!/usr/bin/env python3
"""Run one frozen post-revision treatment against the saved six-case baseline."""

import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]

from scripts import run_response_cross_target_routing as routing
from scripts import run_response_incumbent_frozen_transfer as frozen
from scripts.run_response_structure_transfer import source_parts
from hegel_machine.cognition import world_reference_recalibration as recipe
from hegel_machine.cognition import world_response_hypotheses, world_response_consequences

BASELINE = ROOT / "artifacts/response_cross_target_routing_20261010"
OUTPUT = ROOT / "artifacts/response_post_revision_transfer_20261010"
ARM = "post_revision_research"
FUNCTION = "_extend_hypotheses"


def include_reference_research_question(source):
    """Carry the current assignment and its actual parent path into a new method copy."""
    name = "reference_research_question"
    tree = ast.parse(source)
    if not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == name for node in ast.walk(tree)):
        return source
    current = Path(recipe.__file__).read_text()
    helper = next(node for node in ast.parse(current).body
                  if isinstance(node, ast.FunctionDef) and node.name == name)
    existing = next((node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == name), None)
    lines = source.splitlines(keepends=True)
    replacement = ast.get_source_segment(current, helper) + "\n"
    if existing is not None:
        revised = ("".join(lines[:existing.lineno - 1]) + replacement
                   + "".join(lines[existing.end_lineno:]))
    else:
        extension = next(node for node in tree.body
                         if isinstance(node, ast.FunctionDef) and node.name == FUNCTION)
        revised = ("".join(lines[:extension.lineno - 1]) + replacement + "\n\n"
                   + "".join(lines[extension.lineno - 1:]))
    return revised.replace("reference_research_question(number, failure_file)",
                           "reference_research_question(number, failure_file, path)")


def frozen_package():
    package = json.loads((BASELINE / "cross_target_routing_bundle.json").read_text())
    source = package["files"][routing.study.METHOD_PATH]
    node = lambda code: next(item for item in ast.parse(code).body
        if isinstance(item, ast.FunctionDef) and item.name == FUNCTION)
    current = Path(recipe.__file__).read_text()
    old, new = node(source), node(current)
    replacement = ast.get_source_segment(current, new) + "\n"
    lines = source.splitlines(keepends=True)
    revised = "".join(lines[:old.lineno - 1]) + replacement + "".join(lines[old.end_lineno:])
    revised = include_reference_research_question(revised)
    other = lambda code: [ast.dump(item) for item in ast.parse(code).body
        if not (isinstance(item, ast.FunctionDef)
                and item.name in (FUNCTION, "reference_research_question"))]
    assert other(source) == other(revised), "Only the extension and its known helper may differ"
    assert source_parts(source)[0] == source_parts(revised)[0], "Retain every original declaration"
    assert len(source_parts(revised)[0]) == 16
    assert ast.dump(old) != ast.dump(new), "Do not launch unchanged source"
    compile(revised, "frozen_post_revision_method.py", "exec")
    package["files"][routing.study.METHOD_PATH] = revised
    return package


def summary(rows):
    return {"completed_executions": len(rows), "planned_executions": 6,
        "successes": sum(row["success"] for row in rows),
        "statuses": dict(Counter(row["status"] for row in rows)),
        "source_unchanged": sum(row["source_unchanged"] for row in rows),
        "joint_revision_scopes": sum(row["joint_revision_scopes"] for row in rows),
        "costs": {key: sum(row["costs"].get(key, 0) for row in rows) for key in
            ("environment_steps", "research_compute_units", "python_execution_seconds", "model_calls")}}


def run(output=OUTPUT):
    output = Path(output).resolve()
    protocol = json.loads((BASELINE / "protocol.json").read_text())
    panel, limits = deepcopy(protocol["cases"]), deepcopy(protocol["limits"])
    package = frozen_package()
    baseline = {row["case"]: row for row in json.loads((BASELINE / "brief.json").read_text())
        if row["arm"] == "cross_target_routing"}
    output.mkdir(parents=True, exist_ok=False)
    frozen.write(output / "frozen_bundle.json", package)
    for module in (world_response_hypotheses, world_response_consequences):
        (output / Path(module.__file__).name).write_text(Path(module.__file__).read_text())
    frozen.write(output / "protocol.json", {
        "baseline": str(BASELINE), "source_bundle": str(BASELINE / "cross_target_routing_bundle.json"),
        "cases": panel, "limits": limits, "procedure_max_steps": 256,
        "arm": ARM, "planned_executions": 6, "workers": 3,
        "intervention": "Only embedded _extend_hypotheses is replaced and its existing live-inquiry helper is included when missing. The frozen runtime also uses completed-target positive reference recovery and full-target-domain dual-branch comparison; the two changed runtime modules are archived here.",
        "source_scope": "All source AST statements outside the extension and its known helper, and all 16 declarations, are unchanged; no prior task state, fitted target parameters or model calls installed.",
        "comparison": "Historical matched six-condition cross_target_routing treatment; no baseline rerun. Same immediate-completion fixture and limits.",
        "scope": "Existing development parameter panel and controls. Host research-policy repair, not a learned strategy, unseen structure or recursive-advantage claim. Retain all outcomes without tuning.",
        "storage": "Reuse compact run_case outputs and retain current_state.json with only response_research_history compacted; the current comparison, research table and tests remain complete."})
    rows = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(routing.run_case, case, config, ARM, package, output, limits): case
            for case, config in panel.items()}
        for future in as_completed(futures):
            case = futures[future]
            try:
                rows.append(future.result())
            except Exception as error:
                frozen.write(output / case / ARM / "runner_error.json",
                    {"error": f"{type(error).__name__}: {error}"})
                raise
            rows.sort(key=lambda row: list(panel).index(row["case"]))
            frozen.write(output / "brief.json", rows)
            frozen.write(output / "summary.json", summary(rows))
            print(json.dumps(next(row for row in rows if row["case"] == case)), flush=True)
    pairs = []
    for row in rows:
        case = row["case"]
        old_actions = json.loads((BASELINE / case / "cross_target_routing/action_trace.json").read_text())
        new_actions = json.loads((output / case / ARM / "action_trace.json").read_text())
        pairs.append({"case": case, "rows": [baseline[case], row],
            "identical_action_sequence": [item["action"] for item in old_actions] ==
                                         [item["action"] for item in new_actions]})
    frozen.write(output / "paired_summary.json", {"arms": ["cross_target_routing", ARM], "pairs": pairs,
        "baseline_source": str(BASELINE), "baseline_reexecuted": False})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args().output)
