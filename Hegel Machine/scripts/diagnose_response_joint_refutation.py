#!/usr/bin/env python3
"""Recover the paid checkpoint just after the recorded joint trial is refuted."""

import argparse
import ast
from copy import deepcopy
import json
from math import ceil, floor
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]

from scripts import consolidate_response_measurement as study
from scripts.run_response_incumbent_frozen_transfer import write
from scripts.run_discoveryworld_method_repair import restore_repair
from scripts.run_response_structure_transfer import source_parts
from hegel_machine.cognition import world_reference_recalibration as recipe
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.world_method_transfer import install_method_bundle
from hegel_machine.cognition.world_reference_response import execute_reference_response
from hegel_machine.cognition.world_response_constraints import response_family_grid_support

SOURCE = ROOT / "artifacts/response_cross_target_routing_20261010"
OUTPUT = ROOT / "artifacts/response_joint_refutation_diagnostic_20261010"
CASE = "cubic_original_mid_low"
PREFIX_STEPS = 194


def continue_checkpoint(output=OUTPUT):
    """Resume this paid prefix with the current embedded continuation decision."""
    output = Path(output).resolve()
    destination = output / "continued"
    destination.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((output / "protocol.json").read_text())
    prefix = json.loads((output / "checkpoint.json").read_text())
    before = deepcopy(prefix["costs"])
    source = (output / "agent" / study.METHOD_PATH).read_text()
    template = Path(recipe.__file__).read_text()
    name = "_extend_hypotheses"
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    replacement = next(n for n in ast.parse(template).body
                       if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = source.splitlines(keepends=True)
    revised = ("".join(lines[:node.lineno - 1]) + ast.get_source_segment(template, replacement)
               + "\n" + "".join(lines[node.end_lineno:]))
    other_nodes = lambda code: [ast.dump(n, include_attributes=False) for n in ast.parse(code).body
        if not (isinstance(n, ast.FunctionDef) and n.name == name)]
    assert other_nodes(source) == other_nodes(revised)
    assert source_parts(source)[0] == source_parts(revised)[0]
    remaining = protocol["remaining_procedure_steps"]
    write(destination / "protocol.json", {**protocol, "prefix": str(output / "checkpoint.json"),
        "intervention": "Replace only embedded _extend_hypotheses with the current frozen template; "
                        "retain all declarations and other source AST nodes. Shared research runtime "
                        "uses the current frozen conditional-support decision algorithm.",
        "starting_costs": before, "continuation_procedure_steps": remaining,
        "scope": "One previously consumed public prefix, original cumulative limits and remaining "
                 "procedure steps, zero LLM; diagnostic comparison/projection costs are reported "
                 "separately and are not evidence or state imported into this continuation."})
    panel = json.loads((SOURCE / "protocol.json").read_text())
    study.CASES = {CASE: panel["cases"][CASE]}
    with TemporaryDirectory(prefix="hegel-joint-continuation-") as temporary:
        lab = study.TransferSignalLab(CASE)
        agent = DiscoveryWorldAgent(None, Path(temporary), arm="hegel", limits=protocol["limits"])
        agent.lab = lab
        replay = restore_repair(agent, lab, prefix, output / "agent")
        (agent.workspace / study.METHOD_PATH).write_text(revised)
        execution = agent._execute("run", {"operator": "procedure",
            "method_id": prefix["research"]["methods"][0]["id"], "max_steps": remaining}, lab)
        result = agent.result()
        state = json.loads((agent.workspace / "shared/reference_recalibration.json").read_text())
        new_records = agent.observations[before["environment_steps"]:]
        old = json.loads((SOURCE / CASE / "cross_target_routing/brief.json").read_text())
        old_tail = {key: old["costs"][key] - before[key]
                    for key in ("environment_steps", "research_compute_units", "model_calls")}
        delta = {key: value - before.get(key, 0) for key, value in result["costs"].items()}
        write(destination / "replay.json", replay)
        write(destination / "actions.json", new_records)
        write(destination / "comparison.json", state["response_research"])
        write(destination / "decision_steps.json", execution["steps"])
        write(destination / "agent/shared/reference_recalibration.json", state)
        write(destination / "agent/laboratory_observations.json", agent.observations)
        (destination / "agent" / study.METHOD_PATH).write_text(revised)
        summary = {"status": execution["status"], "success": result["score"]["completedSuccessfully"],
            "procedure_steps": len(execution["steps"]), "starting_costs": before,
            "costs": result["costs"], "new_costs": delta, "old_tail_costs": old_tail,
            "tail_cost_difference": {key: delta[key] - value for key, value in old_tail.items()},
            "comparison_status": state["response_research"]["status"],
            "last_proposal": execution["steps"][-1].get("proposal"),
            "last_feedback": execution["steps"][-1].get("feedback")}
        write(destination / "summary.json", summary)
        print(json.dumps({key: summary[key] for key in ("status", "success", "procedure_steps",
            "new_costs", "old_tail_costs", "tail_cost_difference", "comparison_status")}), flush=True)


def project_candidates(output=OUTPUT):
    """Evaluate existing public reference candidates on the saved target domain."""
    output = Path(output)
    state = json.loads((output / "agent/shared/reference_recalibration.json").read_text())
    comparison = json.loads((output / "comparison.json").read_text())
    conditional = comparison["conditional_simplification"]
    baseline = conditional["family_research"]["operational_search"]["support"]
    supports = conditional["assumption"]["supports"]
    responses = state["response_settings"]
    table = state["research_table"]
    features = {row["crystal"]: {key: value["value"] for key, value in row["measurements"].items()}
                for row in table["references"] + table["targets"]}
    number = comparison["target_number"]
    grid = state["control_grids"][str(number)]
    references = sorted({(e["number"], e["prediction"]) for e in comparison["experiments"]
                         if e["number"] in state["reference_numbers"]})
    completed_boundaries = []
    candidates = list(references)
    for crystal in state["completed"]:
        local = state["control_grids"][str(crystal)]
        observed = {r["frequency"] for r in responses if r["crystal"] == crystal}
        for row in responses:
            if row["crystal"] != crystal or not row["activated"]:
                continue
            position = (row["frequency"] - local["origin"]) / local["step"]
            for index in (ceil(position) - 1, floor(position) + 1):
                setting = local["origin"] + local["step"] * index
                if local["lower"] <= setting <= local["upper"]:
                    completed_boundaries.append({"crystal": crystal, "frequency": setting,
                        "positive_evidence": row["evidence"], "already_observed": setting in observed})
                    if setting not in observed and (crystal, setting) not in candidates:
                        candidates.append((crystal, setting))
    spent, results = 0, []
    for crystal, setting in candidates:
        branches = []
        for activated in (False, True):
            intervals, queries = [], []
            for support in supports:
                answer = response_family_grid_support(features, responses, support, number, grid,
                    queries=[{"crystal": crystal, "frequency": setting, "activated": activated}],
                    compute_limit=max(0, 50000 - spent))
                spent += answer["compute_cost"]
                queries.append(answer)
                intervals.extend(answer["intervals"])
            merged = []
            for left, right in sorted(intervals):
                if merged and left <= merged[-1][1] + 1:
                    merged[-1][1] = max(right, merged[-1][1])
                else:
                    merged.append([left, right])
            complete = all(q["status"] == "COMPLETE" for q in queries)
            count = sum(b - a + 1 for a, b in merged) if complete else None
            branches.append({"activated": activated, "status": "complete" if complete else "unknown",
                "remaining_settings": count, "intervals": merged if complete else None,
                "reduction": baseline["grid_count"] - count if complete else None,
                "compute_cost": sum(q["compute_cost"] for q in queries),
                "reasons": [q.get("reason") for q in queries]})
        results.append({"crystal": crystal, "frequency": setting, "branches": branches,
            "guaranteed_reduction": min(b["reduction"] for b in branches)
                if all(b["reduction"] is not None for b in branches) else None})
    result = {"baseline_support": baseline, "target_number": number, "results": results,
        "completed_target_boundary_candidates": completed_boundaries,
        "compute_limit": 50000, "compute_cost": spent, "new_world_actions": 0, "model_calls": 0,
        "scope": "Counterfactual projection from the saved paid public evidence, existing declared "
                 "supports and existing reference candidates; completed-target boundary proposals "
                 "use only the existing positive-readout plus/minus-one-grid rule."}
    write(output / "reference_projection.json", result)
    print(json.dumps({"compute_cost": spent, "results": [{"crystal": r["crystal"],
        "frequency": r["frequency"], "reductions": [b["reduction"] for b in r["branches"]],
        "guaranteed_reduction": r["guaranteed_reduction"]} for r in results]}), flush=True)


def run(output=OUTPUT):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((SOURCE / "protocol.json").read_text())
    package = json.loads((SOURCE / "cross_target_routing_bundle.json").read_text())
    expected = json.loads((SOURCE / CASE / "cross_target_routing/action_trace.json").read_text())
    limits = deepcopy(protocol["limits"])
    write(output / "protocol.json", {
        "source": str(SOURCE), "case": CASE, "limits": limits,
        "original_procedure_limit": 256, "prefix_procedure_steps": PREFIX_STEPS,
        "remaining_procedure_steps": 256 - PREFIX_STEPS,
        "scope": "Re-execute one unchanged frozen program to the already-recorded failed joint trial. "
                 "Stop before its next comparison, retain the paid public state and account, then "
                 "diagnose that comparison once without a new world action or model call. "
                 "The simulator configuration remains private to the laboratory; the researcher "
                 "receives only its ordinary public observations."})
    study.CASES = {CASE: protocol["cases"][CASE]}
    with TemporaryDirectory(prefix="hegel-joint-refutation-") as temporary:
        lab = study.TransferSignalLab(CASE)
        agent = DiscoveryWorldAgent(None, Path(temporary), arm="hegel", limits=limits)
        agent.lab = lab
        install_method_bundle(agent, package, lab)
        agent._question({"id": "task", "question": "Calibrate the two targets"})
        execution = agent._execute("run", {"operator": "procedure",
            "method_id": package["methods"][0]["id"], "max_steps": PREFIX_STEPS}, lab)
        state_path = agent.workspace / "shared/reference_recalibration.json"
        state = json.loads(state_path.read_text())
        actions = [{key: row.get(key) for key in ("index", "action", "feedback")}
                   for row in agent.observations]
        # If this prefix has changed, it cannot diagnose the recorded failure.
        assert actions == expected[:len(actions)]
        assert len(actions) == 148 and len(execution["steps"]) == PREFIX_STEPS
        assert any(row["prediction"] == 5901 and row["status"] == "refuted"
                   for row in state.get("tests", []))
        assert state["response_research"]["question_id"] == "reference_response_5_9"
        assert (agent.workspace / study.METHOD_PATH).read_text() == package["files"][study.METHOD_PATH]
        before = agent.result()
        before["episode_status"] = before["status"]
        before["method_results"] = []  # Prior full execution transcripts do not drive the policy.
        state.pop("bound_response_research", None)
        # Keep every prior question and current solver state. Compact only the
        # comparison archive, whose length supplies the next response-round ID.
        state["response_research_history"] = [{key: row[key] for key in
            ("question_id", "status", "response_count", "compute_cost")}
            for row in state.get("response_research_history", [])]
        write(output / "checkpoint.json", before)
        write(output / "agent/laboratory_observations.json", agent.observations)
        write(output / "agent/shared/reference_recalibration.json", state)
        (output / "agent" / study.METHOD_PATH).write_text(package["files"][study.METHOD_PATH])
        write(output / "prefix_summary.json", {"observations": len(actions),
            "steps": PREFIX_STEPS, "remaining_steps": 256 - PREFIX_STEPS,
            "action_prefix_equal": True, "current_question": state["response_research"]["question_id"],
            "costs": before["costs"], "status": execution["status"]})
        comparison = execute_reference_response(agent, lab, {"path": study.METHOD_PATH})
        write(output / "comparison.json", comparison)
        after = agent.result()
        write(output / "diagnostic_costs.json", {key: after["costs"][key] - value
            for key, value in before["costs"].items()})
        summary = {"status": comparison["status"], "response_count": comparison["response_count"],
            "compute_cost": comparison["compute_cost"], "selected_number": comparison.get("selected_number"),
            "selected_setting": comparison.get("selected_setting"), "model_calls": after["costs"]["model_calls"],
            "new_world_actions": len(agent.observations) - len(actions)}
        write(output / "summary.json", summary)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--project-checkpoint", action="store_true")
    parser.add_argument("--continue-checkpoint", action="store_true")
    args = parser.parse_args()
    (continue_checkpoint if args.continue_checkpoint else
     project_candidates if args.project_checkpoint else run)(args.output)
