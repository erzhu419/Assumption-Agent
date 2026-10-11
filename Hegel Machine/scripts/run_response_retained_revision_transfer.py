#!/usr/bin/env python3
"""Compare finite retained-model confirmation with the saved six-case policy."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]

from scripts import run_response_cross_target_routing as routing
from scripts import run_response_incumbent_frozen_transfer as frozen
from scripts.run_response_post_revision_transfer import summary
from scripts.run_response_structure_transfer import source_parts
from hegel_machine.cognition import world_response_hypotheses, world_response_consequences

BASELINE = ROOT / "artifacts/response_post_revision_transfer_20261010"
OUTPUT = ROOT / "artifacts/response_retained_revision_transfer_20261010"
OLD_ARM = "post_revision_research"
ARM = "retained_revision_confirmation"


def continue_checkpoint(output=OUTPUT):
    """Continue the exact paid prefix under its original remaining limits."""
    from scripts.run_discoveryworld_method_repair import restore_repair
    from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent

    checkpoint = ROOT / "artifacts/response_joint_refutation_diagnostic_20261010"
    destination = Path(output) / "same_prefix_continuation"
    destination.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((checkpoint / "protocol.json").read_text())
    prefix = json.loads((checkpoint / "checkpoint.json").read_text())
    before = deepcopy(prefix["costs"])
    panel = json.loads((BASELINE / "protocol.json").read_text())
    case = protocol["case"]
    routing.study.CASES = {case: panel["cases"][case]}
    source = json.loads((BASELINE / "frozen_bundle.json").read_text())["files"][routing.study.METHOD_PATH]
    frozen.write(destination / "protocol.json", {
        "checkpoint": str(checkpoint), "case": case, "limits": protocol["limits"],
        "remaining_procedure_steps": protocol["remaining_procedure_steps"],
        "starting_costs": before,
        "source_bundle": str(BASELINE / "frozen_bundle.json"),
        "baseline_continuation": str(checkpoint / "continued"),
        "scope": "Exact paid observations and account restored; use the same previous frozen controller and current retained-model runtime. No nominated setting, imported diagnostic result, new model call or extra budget."})
    with TemporaryDirectory(prefix="hegel-retained-confirmation-") as temporary:
        lab = routing.study.TransferSignalLab(case)
        agent = DiscoveryWorldAgent(None, Path(temporary), arm="hegel", limits=protocol["limits"])
        agent.lab = lab
        replay = restore_repair(agent, lab, prefix, checkpoint / "agent")
        (agent.workspace / routing.study.METHOD_PATH).write_text(source)
        execution = agent._execute("run", {"operator": "procedure",
            "method_id": prefix["research"]["methods"][0]["id"],
            "max_steps": protocol["remaining_procedure_steps"]}, lab)
        result = agent.result()
        state = json.loads((agent.workspace / "shared/reference_recalibration.json").read_text())
        new_records = agent.observations[before["environment_steps"]:]
        delta = {key: value - before.get(key, 0) for key, value in result["costs"].items()}
        old = json.loads((checkpoint / "continued/summary.json").read_text())
        frozen.write(destination / "replay.json", replay)
        frozen.write(destination / "actions.json", new_records)
        frozen.write(destination / "decision_steps.json", execution["steps"])
        frozen.write(destination / "current_state.json", {
            **state, "response_research_history": [frozen.compact_comparison(item)
                for item in state.get("response_research_history", [])]})
        brief = {"status": execution["status"], "success": result["score"]["completedSuccessfully"],
            "procedure_steps": len(execution["steps"]), "starting_costs": before,
            "costs": result["costs"], "new_costs": delta,
            "baseline_success": old["success"], "baseline_new_costs": old["new_costs"],
            "last_proposal": execution["steps"][-1].get("proposal"),
            "last_feedback": execution["steps"][-1].get("feedback")}
        frozen.write(destination / "summary.json", brief)
        print(json.dumps({key: brief[key] for key in (
            "status", "success", "procedure_steps", "new_costs", "baseline_new_costs")}), flush=True)
        return brief


def run(output=OUTPUT):
    output = Path(output).resolve()
    protocol = json.loads((BASELINE / "protocol.json").read_text())
    panel, limits = deepcopy(protocol["cases"]), deepcopy(protocol["limits"])
    bundle_text = (BASELINE / "frozen_bundle.json").read_text()
    package = json.loads(bundle_text)
    assert len(source_parts(package["files"][routing.study.METHOD_PATH])[0]) == 16
    current = Path(world_response_hypotheses.__file__).read_text()
    assert current != (BASELINE / "world_response_hypotheses.py").read_text()
    # A runtime intervention must not silently include the prior turn's later
    # stop-message edit or a change to consequence projection.
    assert Path(world_response_consequences.__file__).read_text() == (
        BASELINE / "world_response_consequences.py").read_text()
    baseline = {row["case"]: row for row in json.loads((BASELINE / "brief.json").read_text())}
    output.mkdir(parents=True, exist_ok=False)
    (output / "frozen_bundle.json").write_text(bundle_text)
    (output / "world_response_hypotheses.py").write_text(current)
    frozen.write(output / "protocol.json", {
        "baseline": str(BASELINE), "source_bundle": str(BASELINE / "frozen_bundle.json"),
        "cases": panel, "limits": limits, "procedure_max_steps": 256,
        "arms": [OLD_ARM, ARM], "planned_executions": 6, "workers": 3,
        "intervention": "Only world_response_hypotheses changes: complete finite confirmation of already-paid joint models without another parameter fit or stagnation reset.",
        "source_scope": "The previous frozen bundle is copied exactly, retaining all 16 declarations, embedded controller functions and original task inputs.",
        "comparison": "Reuse the completed matched six-condition post_revision_research baseline. No baseline rerun; same immediate-completion fixture, cumulative limits and zero LLM calls.",
        "scope": "Consumed development parameter panel and controls; host policy repair, not new structure discovery, model adoption or recursive advantage. All outcomes retained without tuning."})
    rows = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(routing.run_case, case, config, ARM, package, output, limits): case
                   for case, config in panel.items()}
        for future in as_completed(futures):
            case = futures[future]
            try:
                row = future.result()
            except Exception as error:
                frozen.write(output / case / ARM / "runner_error.json",
                             {"error": f"{type(error).__name__}: {error}"})
                raise
            rows.append(row)
            rows.sort(key=lambda item: list(panel).index(item["case"]))
            frozen.write(output / "brief.json", rows)
            frozen.write(output / "summary.json", summary(rows))
            print(json.dumps(row), flush=True)
    pairs = []
    for row in rows:
        case = row["case"]
        old = json.loads((BASELINE / case / OLD_ARM / "action_trace.json").read_text())
        new = json.loads((output / case / ARM / "action_trace.json").read_text())
        pairs.append({"case": case, "rows": [baseline[case], row],
            "identical_action_sequence": [item["action"] for item in old] ==
                                         [item["action"] for item in new]})
    frozen.write(output / "paired_summary.json", {"arms": [OLD_ARM, ARM], "pairs": pairs,
                 "baseline_source": str(BASELINE), "baseline_reexecuted": False})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args().output)
