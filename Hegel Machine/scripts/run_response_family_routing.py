#!/usr/bin/env python3
"""Compare repaired family queries and earlier response routing on the same four cases."""

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]

from scripts import consolidate_response_measurement as study
from hegel_machine.cognition import world_reference_recalibration as recipe
from hegel_machine.cognition import world_response_constraints as constraints
from hegel_machine.cognition.world_reference_hypothesis import _retain_declaration


def declarations(source):
    return list(next(ast.literal_eval(node.value) for node in ast.parse(source).body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ESTIMATORS"
                                               for t in node.targets)))


def paired_case(case, bundle, previous, output, limits):
    # Separate processes keep each original four-case fixture and output independent.
    study.CASES = {case: study.CASES[case]}
    return study.transfer(bundle, previous, Path(output) / case, limits)


def run(source, baseline, output):
    source, baseline, output = map(lambda p: Path(p).resolve(), (source, baseline, output))
    output.mkdir(parents=True, exist_ok=False)
    published = json.loads((source / "method_bundle.json").read_text())
    previous = published["files"][study.METHOD_PATH]
    current = Path(recipe.__file__).read_text()
    for declaration in declarations(previous):
        current = _retain_declaration(current, declaration)
    assert declarations(current) == declarations(previous), "Scientific declarations must remain unchanged"
    bundle = {"files": {study.METHOD_PATH: current}, "methods": [{
        "id": "response_routing_revision", "path": study.METHOD_PATH,
        "parameters": {}, "requires": [], "max_steps": 256}]}
    limits = json.loads((source / "protocol.json").read_text())["limits"]
    recorded = [row for row in json.loads((baseline / "brief.json").read_text())
                if row["arm"] == "consolidated"]
    study.write_json(output / "historical_baseline.json", recorded)
    study.write_json(output / "method_bundle.json", bundle)
    (output / "previous_source.py").write_text(previous)
    (output / "world_response_constraints.py").write_text(Path(constraints.__file__).read_text())
    study.write_json(output / "protocol.json", {
        "source": str(source), "recorded_baseline": str(baseline),
        "cases": deepcopy(study.CASES), "limits": limits, "model_calls": 0,
        "arms": {"historical_baseline": "Previously completed frozen publication with previous query solver",
                 "previous_source": "Same published program with repaired query solver",
                 "consolidated": "Same declarations with repaired query solver and earlier response routing"},
        "source_change": "Host controller repair; no new scientific declarations, measurements or answers supplied.",
        "scope": "Same four development conditions and limits; no new publication or model authorship claimed."})
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(paired_case, case, bundle, previous, output, limits): case
                   for case in study.CASES}
        for future in as_completed(futures):
            rows.extend(future.result())
            study.write_json(output / "brief.json", rows)
            print(json.dumps({"finished": futures[future], "completed_arms": len(rows)}), flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.source, args.baseline, args.output)
