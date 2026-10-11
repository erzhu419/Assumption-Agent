#!/usr/bin/env python3
"""Repeat the historical pre-cubic knowledge stage with two live agent arms."""

import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]

from scripts import run_response_matched_adoption as matched
from scripts.run_response_structure_transfer import source_parts

HISTORY = ROOT / "artifacts/response_family_recovery_20261009/changed_mechanism/source_before.py"
OUTPUT = ROOT / "artifacts/response_revision_replay_20261010"


def historical_knowledge(bundle, historical_source):
    """Use the actual earlier declarations with the current execution code."""
    prepared = deepcopy(bundle)
    path = matched.study.METHOD_PATH
    source = prepared["files"][path]
    declarations, _ = source_parts(historical_source)
    assignment = next(node for node in ast.parse(source).body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ESTIMATORS" for target in node.targets))
    lines = source.splitlines(keepends=True)
    prepared["files"][path] = ("".join(lines[:assignment.lineno - 1])
        + "ESTIMATORS = " + repr(declarations) + "\n" + "".join(lines[assignment.end_lineno:]))
    return prepared


def run(output=OUTPUT):
    prior = json.loads((matched.SOURCE / "protocol.json").read_text())
    bundle = historical_knowledge(json.loads(
        (matched.SOURCE / "completed_target_routing_bundle.json").read_text()), HISTORY.read_text())
    cases = {"changed_mechanism": deepcopy(matched.study.CASES["changed_mechanism"]),
             "aliased_shifted_mid": deepcopy(prior["cases"]["aliased_shifted_mid"])}
    result = matched.run(output, cases=cases, bundle=bundle, knowledge_scope={
        "declarations_from": str(HISTORY), "declaration_count": 9,
        "scope": "Replay the recorded knowledge state before the first cubic discovery, "
            "including its original rationales. Both arms start from zero observations with "
            "these same declarations and the current execution code. This is a repeated "
            "development revision task, not a new discovery benchmark. The separate full "
            "16-declaration archive and its completed adoption cohort are retained. "
            "The aliased control has no informative extra instrument; object-specific "
            "search is still possible, but more functions of identical readings cannot "
            "repair the shared deterministic response premise."})
    (Path(output) / Path(__file__).name).write_text(Path(__file__).read_text())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    print(json.dumps(run(parser.parse_args().output), indent=2))
