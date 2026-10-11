#!/usr/bin/env python3
"""Freeze the discovered cubic declaration and test its paired successor value."""

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

from scripts import consolidate_response_measurement as study


NEW_DECLARATION = "signal_cubic_5"
SIGNALS = {
    "original": {1: 1., 2: 2., 3: 3., 4: 4., 5: 395.5 ** .5},
    "shifted": {1: 1.25, 2: 2.75, 3: 4.25, 4: 6.75, 5: 18.25},
}
PARAMETERS = ((15.2, .6), (75.4, .85), (130.1, 1.1), (250.3, 1.2))


def successor_cases():
    cases = {}
    for distribution, values in SIGNALS.items():
        for index, (intercept, gain) in enumerate(PARAMETERS, start=1):
            cases[f"cubic_{distribution}_{index}"] = {
                "group": "cubic_transfer", "intercept": intercept, "gain": gain,
                "power": 3, "sensor": "informative", "signal_values": deepcopy(values)}
        intercept, gain = ((70.2, 9.5) if distribution == "original" else (180.4, 16.))
        cases[f"square_{distribution}"] = {
            "group": "wrong_structure_control", "intercept": intercept, "gain": gain,
            "power": 2, "sensor": "informative", "signal_values": deepcopy(values)}
        cases[f"aliased_{distribution}"] = {
            "group": "uninformative_control", "intercept": 75.4, "gain": .85,
            "power": 3, "sensor": "density", "signal_values": deepcopy(values)}
    return cases


def source_parts(source):
    node = next(node for node in ast.parse(source).body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
            and target.id == "ESTIMATORS" for target in node.targets))
    lines = source.splitlines(keepends=True)
    return list(ast.literal_eval(node.value)), "".join(lines[:node.lineno - 1] + lines[node.end_lineno:])


def frozen_packages(source):
    previous = (source / "source_before.py").read_text()
    retained = (source / "source_after.py").read_text()
    before, before_execution = source_parts(previous)
    after, after_execution = source_parts(retained)
    additions = [item for item in after if item not in before]
    if (before_execution != after_execution or
            [item for item in after if item.get("id") != NEW_DECLARATION] != before or
            len(additions) != 1 or additions[0].get("id") != NEW_DECLARATION):
        raise ValueError("This comparison requires unchanged execution and only the source-authored cubic addition")
    method = {"id": "frozen_response_method", "path": study.METHOD_PATH,
              "parameters": {}, "requires": [], "max_steps": 256}
    packages = {arm: {"files": {study.METHOD_PATH: code}, "methods": [deepcopy(method)]}
                for arm, code in (("without_cubic", previous), ("retain_cubic", retained))}
    return packages, additions[0]


def run_case(case, config, arm, package, output, limits):
    study.CASES = {case: config}
    row = study.run_transfer_case(case, arm, package, output, limits)
    directory = Path(output) / case / arm
    state = json.loads((directory / "agent/shared/reference_recalibration.json").read_text())
    current_source = (directory / "agent" / study.METHOD_PATH).read_text()
    final_declarations, _ = source_parts(current_source)
    families = state.get("response_research", {}).get("family_research", {}).get("families", [])
    row.update(group=config["group"], cubic={
        "retained_in_final_source": any(item.get("id") == NEW_DECLARATION for item in final_declarations),
        "point_tests": [{key: test.get(key) for key in ("number", "prediction", "status", "question")}
                        for test in state.get("tests", []) if test.get("family") == NEW_DECLARATION],
        "last_family_query_status": next((family["status"] for family in families
            if NEW_DECLARATION in family.get("source_ids", [family.get("id")])), "not_queried")},
        final_declaration_ids=[item["id"] for item in final_declarations])
    study.write_json(directory / "brief.json", row)
    return row


def summarize(rows):
    groups = {}
    for group in dict.fromkeys(row["group"] for row in rows):
        arms = {}
        for arm in ("without_cubic", "retain_cubic"):
            selected = [row for row in rows if row["group"] == group and row["arm"] == arm]
            arms[arm] = {"cases": len(selected), "successes": sum(row["success"] for row in selected),
                "actions": sum(row["costs"]["environment_steps"] for row in selected),
                "research_compute": sum(row["costs"]["research_compute_units"] for row in selected),
                "model_calls": sum(row["costs"]["model_calls"] for row in selected),
                "source_unchanged": sum(row["source_unchanged"] for row in selected),
                "cases_testing_cubic_points": sum(bool(row["cubic"]["point_tests"]) for row in selected),
                "last_cubic_family_query_statuses": dict(Counter(
                    row["cubic"]["last_family_query_status"] for row in selected)),
                "statuses": dict(Counter(row["status"] for row in selected))}
        groups[group] = arms
    return {"completed_executions": len(rows), "planned_executions": 24, "groups": groups,
        "scope": "Twelve fixed successor conditions, not twelve independent random seeds. "
                 "Point tests and the latest family query retain their separate scopes."}


def run(source, output, limits_from):
    source, output, limits_from = [Path(path).resolve() for path in (source, output, limits_from)]
    packages, added_declaration = frozen_packages(source)
    cases = successor_cases()
    limits = {**json.loads(limits_from.read_text())["limits"], "max_model_calls": 0}
    output.mkdir(parents=True, exist_ok=False)
    for arm, package in packages.items():
        study.write_json(output / f"{arm}_bundle.json", package)
    study.write_json(output / "protocol.json", {
        "source": str(source), "source_files": ["source_before.py", "source_after.py"],
        "cases": cases, "limits_from": str(limits_from), "limits": limits,
        "procedure_max_steps": 256, "planned_executions": 24, "model_calls": 0,
        "added_declaration": added_declaration,
        "arms": {"without_cubic": "The unchanged nine-declaration source before its cubic addition",
                 "retain_cubic": "The unchanged ten-declaration source after its autonomous cubic addition"},
        "packages": {arm: f"{arm}_bundle.json" for arm in packages},
        "source_cost_scope": "Previously paid discovery/review costs are not charged again to successors.",
        "scope": "Only the new cubic declaration differs. Both arms retain the same controller, "
                 "measurements, finite local search and limits. No source-world state or fitted parameters "
                 "are installed. Source edits during target execution are reported as local adaptation. "
                 "All fixed successor conditions are reported; controls do not prespecify failure."})
    rows = []
    order = {(case, arm): index for index, (case, arm) in enumerate(
        (case, arm) for case in cases for arm in packages)}
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_case, case, config, arm, package, output, limits): (case, arm)
            for case, config in cases.items() for arm, package in packages.items()}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            rows.sort(key=lambda value: order[value["case"], value["arm"]])
            study.write_json(output / "brief.json", rows)
            study.write_json(output / "summary.json", summarize(rows))
            print(json.dumps({"finished": futures[future], "completed_executions": len(rows),
                              "success": row["success"], "actions": row["costs"]["environment_steps"]}), flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limits-from", type=Path,
                        default=ROOT / "artifacts/response_family_routing_20261009/protocol.json")
    args = parser.parse_args()
    run(args.source, args.output, args.limits_from)
