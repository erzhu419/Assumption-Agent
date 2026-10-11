#!/usr/bin/env python3
"""Compare completed-target response routing on the existing fixed panel."""

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

BASELINE = ROOT / "artifacts/response_retained_revision_transfer_20261010"
OUTPUT = ROOT / "artifacts/response_completed_target_routing_20261010"
OLD_ARM = "previous_routing_current_runtime"
ARM = "completed_target_routing"
HIGH = "cubic_original_mid_high"


def summary(rows):
    treatment = [row for row in rows if row["arm"] == ARM]
    return {"completed_executions": len(rows), "planned_executions": 7,
        "treatment_cases": len(treatment),
        "treatment_successes": sum(row["success"] for row in treatment),
        "statuses": dict(Counter(row["status"] for row in treatment)),
        "source_unchanged": sum(row["source_unchanged"] for row in rows),
        "treatment_costs": {key: sum(row["costs"].get(key, 0) for row in treatment)
            for key in ("environment_steps", "research_compute_units",
                        "python_execution_seconds", "model_calls")}}


def evidence(directory):
    trace = json.loads((directory / "routing_trace.json").read_text())
    calls = [item for item in trace["response_calls"] if item["requested_target"] == 5]
    verdicts = trace["point_and_response_outcomes"]
    return {"first_target5_response": calls[0] if calls else None,
        "target4_point_verdicts": [item for item in verdicts
            if item["number"] == 4 and not item["response_question"]],
        "target5_point_refutations": [item for item in verdicts
            if item["number"] == 5 and not item["response_question"]
            and item["status"] == "refuted"]}


def run(output=OUTPUT):
    output = Path(output).resolve()
    prior = json.loads((BASELINE / "protocol.json").read_text())
    panel, limits = deepcopy(prior["cases"]), deepcopy(prior["limits"])
    old_package = json.loads((BASELINE / "frozen_bundle.json").read_text())
    source = old_package["files"][routing.study.METHOD_PATH]
    production = Path(recipe.__file__).read_text()
    revised = ast.get_source_segment(production, routing.function_node(production)) + "\n"
    assert ast.dump(routing.function_node(source)) != ast.dump(routing.function_node(revised))
    assert len(source_parts(source)[0]) == 16
    packages = {OLD_ARM: old_package, ARM: deepcopy(old_package)}
    packages[ARM]["files"][routing.study.METHOD_PATH] = routing.replace_function(source, revised)
    runtime = Path(world_response_hypotheses.__file__).read_text()
    assert runtime == (BASELINE / "final_cache/world_response_hypotheses.py").read_text(), \
        "Both arms must use the final retained-model cache runtime"
    baseline = {row["case"]: row for row in json.loads((BASELINE / "brief.json").read_text())}
    output.mkdir(parents=True, exist_ok=False)
    for arm, package in packages.items():
        frozen.write(output / (arm + "_bundle.json"), package)
    for module in (world_response_hypotheses, world_response_consequences):
        (output / Path(module.__file__).name).write_text(Path(module.__file__).read_text())
    (output / "new_point_needs_response.py").write_text(revised)
    (output / "old_point_needs_response.py").write_text(
        ast.get_source_segment(source, routing.function_node(source)) + "\n")
    frozen.write(output / "protocol.json", {
        "baseline": str(BASELINE), "source_bundle": str(BASELINE / "frozen_bundle.json"),
        "cases": panel, "limits": limits, "procedure_max_steps": 256,
        "arms": [OLD_ARM, ARM], "planned_executions": 7, "workers": 3,
        "intervention": "Only embedded _point_needs_response changes: a currently completed other target can route a refuted point prediction to the existing response researcher. All other AST and all 16 declarations are retained.",
        "comparison": "Rerun original_high old/new routing under the same final-cache runtime. Compare the other five treatments with saved prior outcomes; those historical outcomes predate the final-cache runtime fix, so the high pair alone isolates routing.",
        "runtime_phase_boundary": "Current world_response_hypotheses exactly equals the prior final_cache archive. The saved historical six-case panel used the earlier archived runtime, not final_cache. Both current runtime modules are saved here before execution.",
        "execution_order": [[HIGH, OLD_ARM], [HIGH, ARM]] +
            [[case, ARM] for case in panel if case != HIGH],
        "scope": "Consumed development parameter panel, not unseen structure or LLM adoption; same immediate-completion fixture, zero LLM calls, no installed solution frequency or extra budget, and no outcome-based tuning."})
    plan = [(HIGH, OLD_ARM), (HIGH, ARM)] + [(case, ARM) for case in panel if case != HIGH]
    rows = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(routing.run_case, case, panel[case], arm,
            packages[arm], output, limits): (case, arm) for case, arm in plan}
        for future in as_completed(futures):
            case, arm = futures[future]
            try:
                row = future.result()
            except Exception as error:
                frozen.write(output / case / arm / "runner_error.json",
                    {"error": f"{type(error).__name__}: {error}"})
                raise
            rows.append(row)
            rows.sort(key=lambda item: plan.index((item["case"], item["arm"])))
            frozen.write(output / "brief.json", rows)
            frozen.write(output / "summary.json", summary(rows))
            print(json.dumps(row), flush=True)
    pairs = []
    for case in panel:
        new_row = next(row for row in rows if row["case"] == case and row["arm"] == ARM)
        old_dir = output / case / OLD_ARM if case == HIGH else BASELINE / case / "retained_revision_confirmation"
        old_row = next(row for row in rows if row["case"] == case and row["arm"] == OLD_ARM) if case == HIGH else baseline[case]
        new_dir = output / case / ARM
        actions = [[item["action"] for item in json.loads((directory / "action_trace.json").read_text())]
                   for directory in (old_dir, new_dir)]
        prefix = next((index for index, (old, new) in enumerate(zip(*actions)) if old != new),
                      min(map(len, actions)))
        pairs.append({"case": case, "same_runtime_pair": case == HIGH,
            "baseline_directory": str(old_dir), "rows": [old_row, new_row],
            "identical_action_sequence": actions[0] == actions[1],
            "common_action_prefix_length": prefix,
            "evidence": [evidence(directory) for directory in (old_dir, new_dir)]})
    frozen.write(output / "paired_summary.json", {"pairs": pairs,
        "historical_source": str(BASELINE), "rerun_baseline_cases": [HIGH],
        "historical_summary": {"successes": sum(row["success"] for row in baseline.values()),
            "costs": {key: sum(row["costs"].get(key, 0) for row in baseline.values())
                for key in ("environment_steps", "research_compute_units", "python_execution_seconds", "model_calls")}}})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args().output)
