#!/usr/bin/env python3
"""Paired task-family interpolation with the unchanged current response method."""

import argparse
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
from scripts.run_response_structure_transfer import SIGNALS, source_parts


SOURCE = ROOT / "artifacts/response_condition_practice_20261009/ordinary_parent_resume_20261010/agent/shared/consolidated_method.py"
LIMITS_FROM = ROOT / "artifacts/response_structure_transfer_20261009/protocol.json"
OUTPUT = ROOT / "artifacts/response_incumbent_frozen_transfer_20261010"
ARMS = ("joint_revision", "without_joint_revision")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def cases():
    result = {}
    for label, intercept, gain in (("mid_low", 45.3, .725), ("mid_high", 190.2, 1.15)):
        for distribution in ("original", "shifted"):
            result[f"cubic_{distribution}_{label}"] = {
                "group": "cubic_interpolation", "intercept": intercept, "gain": gain,
                "power": 3, "sensor": "informative",
                "signal_values": deepcopy(SIGNALS[distribution])}
    result["square_shifted_mid"] = {
        "group": "wrong_structure_control", "intercept": 125.3, "gain": 12.75,
        "power": 2, "sensor": "informative", "signal_values": deepcopy(SIGNALS["shifted"])}
    result["aliased_shifted_mid"] = {
        "group": "aliased_control", "intercept": 45.3, "gain": .725,
        "power": 3, "sensor": "density", "signal_values": deepcopy(SIGNALS["shifted"])}
    return result


def compact_comparison(comparison):
    family = comparison.get("family_research", {})
    conditional = comparison.get("conditional_simplification", {})
    revision = conditional.get("joint_revision")
    return {key: deepcopy(comparison.get(key)) for key in
            ("question_id", "status", "response_count", "compute_cost", "selected_experiment")} | {
        "family_status": family.get("status"),
        "no_progress_cycles": family.get("no_progress_cycles"),
        "conditional_adopted": conditional.get("adopted"),
        "joint_revision": None if revision is None else {
            "scope": revision.get("scope"), "status": revision.get("status"),
            "compute_cost": revision.get("compute_cost"),
            "experiment": revision.get("experiment"),
            "model_count": len(revision.get("models", []))}}


def run_case(case, config, arm, package, output, limits):
    from hegel_machine.cognition import world_response_hypotheses

    original_revision = world_response_hypotheses._revise_stalled_support
    if arm == "without_joint_revision":
        world_response_hypotheses._revise_stalled_support = lambda *args: (None, None)
    try:
        study.CASES = {case: config}
        with TemporaryDirectory(prefix="hegel-frozen-response-") as temporary:
            row = study.run_transfer_case(case, arm, package, temporary, limits)
            directory = Path(temporary) / case / arm
            result = json.loads((directory / "result.json").read_text())
            state_path = directory / "agent/shared/reference_recalibration.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            history = [compact_comparison(item) for item in state.get("response_research_history", [])]
            scopes, revisions = [], []
            for item in history:
                revision = item["joint_revision"]
                if revision is not None and revision["scope"] not in scopes:
                    scopes.append(revision["scope"])
                    revisions.append(revision)
            final_source = (directory / "agent" / study.METHOD_PATH).read_text()
            before, _ = source_parts(package["files"][study.METHOD_PATH])
            after, _ = source_parts(final_source)
            row.update(group=config["group"], joint_revision_scopes=len(revisions),
                joint_revision_new_prediction_scopes=sum(item["status"] == "new_joint_target_prediction"
                                                        for item in revisions),
                procedure_steps=len(result["execution"].get("steps", [])),
                comparisons=len(history), response_count=len(state.get("response_settings", [])),
                final_declaration_count=len(after),
                added_declarations=[item for item in after if item not in before],
                removed_declarations=[item for item in before if item not in after])
            destination = Path(output) / case / arm
            write(destination / "brief.json", row)
            write(destination / "research_trace.json", {"comparisons": history,
                "trials": state.get("tests", []), "response_settings": state.get("response_settings", []),
                "score": result["score"], "execution": {
                    key: value for key, value in result["execution"].items() if key != "steps"},
                "procedure_steps": len(result["execution"].get("steps", [])),
                "last_action_public": next((step["feedback"]["action_results"][-1]
                    for step in reversed(result["execution"].get("steps", []))
                    if step.get("feedback", {}).get("action_results")), None)})
            write(destination / "action_trace.json", [
                {key: record.get(key) for key in ("index", "action", "feedback")}
                for record in result.get("observations", [])])
            if not row["source_unchanged"]:
                (destination / "final_source.py").write_text(final_source)
            return row
    finally:
        world_response_hypotheses._revise_stalled_support = original_revision


def summarize(rows):
    result = {"completed_executions": len(rows), "planned_executions": 12, "arms": {}}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        result["arms"][arm] = {"cases": len(selected),
            "successes": sum(row["success"] for row in selected),
            "source_unchanged": sum(row["source_unchanged"] for row in selected),
            "statuses": dict(Counter(row["status"] for row in selected)),
            "joint_revision_scopes": sum(row["joint_revision_scopes"] for row in selected),
            "joint_revision_new_prediction_scopes": sum(row["joint_revision_new_prediction_scopes"] for row in selected),
            "new_joint_prediction_cases": sum(row["joint_revision_new_prediction_scopes"] > 0 for row in selected),
            "costs": {key: sum(row["costs"].get(key, 0) for row in selected)
                for key in ("environment_steps", "research_compute_units", "python_execution_seconds", "model_calls")}}
    return result


def compare_saved_actions(output, rows):
    pairs = []
    for case in cases():
        selected = [next(row for row in rows if row["case"] == case and row["arm"] == arm)
                    for arm in ARMS]
        actions = [[item["action"] for item in json.loads(
            (output / case / arm / "action_trace.json").read_text())] for arm in ARMS]
        pairs.append({"case": case, "success": [row["success"] for row in selected],
            "statuses": [row["status"] for row in selected],
            "procedure_steps": [row["procedure_steps"] for row in selected],
            "actions": [row["costs"]["environment_steps"] for row in selected],
            "compute": [row["costs"]["research_compute_units"] for row in selected],
            "identical_action_sequence": actions[0] == actions[1],
            "joint_revision_scopes": [row["joint_revision_scopes"] for row in selected]})
    write(output / "paired_summary.json", {"arms": ARMS, "pairs": pairs,
        "all_action_sequences_identical": all(pair["identical_action_sequence"] for pair in pairs),
        "comparison_scope": "Compare saved action lists only; no new execution or score re-evaluation."})


def run(source=SOURCE, output=OUTPUT):
    source, output = Path(source).resolve(), Path(output).resolve()
    code = source.read_text()
    declarations, _ = source_parts(code)
    if len(declarations) != 16:
        raise ValueError("This fixed protocol requires the current 16-declaration source")
    panel = cases()
    limits = {**json.loads(LIMITS_FROM.read_text())["limits"], "max_model_calls": 0}
    package = {"files": {study.METHOD_PATH: code}, "methods": [{"id": "frozen_current_response",
        "path": study.METHOD_PATH, "parameters": {}, "requires": [], "max_steps": 256}]}
    output.mkdir(parents=True, exist_ok=False)
    write(output / "frozen_bundle.json", package)
    write(output / "protocol.json", {"source": str(source), "cases": panel, "arms": ARMS,
        "source_declaration_count": len(declarations), "limits": limits,
        "limits_from": str(LIMITS_FROM), "procedure_max_steps": 256, "planned_executions": 12,
        "intervention": "Only disable _revise_stalled_support in the paired worker; all other runtime code is identical.",
        "source_scope": "Same complete current 16-declaration source; no observation, fitted parameter, prior frequency or response-research state is installed.",
        "scope": "Six predeclared parameter interpolations in the existing task family. Historical signal layouts are reused. This tests repaired-policy breadth, not new structure discovery, LLM adoption or recursive advantage. All cases are reported without tuning.",
        "storage": "The existing runner uses a temporary workspace. Retain compact scientific and action traces, source changes and complete costs; discard duplicated runtime archives."})
    order = {(case, arm): index for index, (case, arm) in enumerate(
        (case, arm) for case in panel for arm in ARMS)}
    rows = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run_case, case, config, arm, package, output, limits): (case, arm)
                   for case, config in panel.items() for arm in ARMS}
        for future in as_completed(futures):
            case, arm = futures[future]
            try:
                row = future.result()
            except Exception as error:
                write(output / case / arm / "runner_error.json", {"error": f"{type(error).__name__}: {error}"})
                raise
            rows.append(row)
            rows.sort(key=lambda item: order[item["case"], item["arm"]])
            write(output / "brief.json", rows)
            write(output / "summary.json", summarize(rows))
            print(json.dumps(row), flush=True)
    compare_saved_actions(output, rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.source, args.output)
