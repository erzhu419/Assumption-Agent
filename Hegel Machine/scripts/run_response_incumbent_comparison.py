#!/usr/bin/env python3
"""Compare retained response programs on exactly the same already-paid evidence.

This is an offline diagnostic of existing candidate construction and research
selection. It neither supplies a known successful setting nor invokes a model.
"""
import argparse
import ast
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]

from hegel_machine.cognition.world_reference_recalibration import refresh_response_evidence
from hegel_machine.cognition.world_response_consequences import QUERY_ALLOWANCE, _families
from hegel_machine.cognition.world_response_constraints import query_response_family
from hegel_machine.cognition.world_response_hypotheses import propose_response_research

SOURCE = ROOT / "artifacts/response_condition_practice_20261009/ordinary_parent_resume_20261010"
LEARNED = ROOT / "artifacts/response_family_recovery_20261009/changed_mechanism/agent/shared/consolidated_method.py"
OUTPUT = ROOT / "artifacts/response_incumbent_comparison_20261010"
METHOD = "shared/consolidated_method.py"
BINDING = "shared/response_representation_proposal.py"


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def estimators(path):
    assignment = next(node for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
            and target.id == "ESTIMATORS" for target in node.targets))
    return list(ast.literal_eval(assignment.value))


def historical_declarations_on_current_controller():
    source = (SOURCE / "agent" / METHOD).read_text()
    module = ast.parse(source)
    assignment = next(node for node in module.body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
            and target.id == "ESTIMATORS" for target in node.targets))
    lines = source.splitlines(keepends=True)
    revised = "".join(lines[:assignment.lineno - 1]) + "ESTIMATORS = " + repr(tuple(estimators(LEARNED))) + "\n" + "".join(lines[assignment.end_lineno:])
    before = [ast.dump(node, include_attributes=False) for node in module.body if node is not assignment]
    after_module = ast.parse(revised)
    after = [ast.dump(node, include_attributes=False) for node in after_module.body
        if not (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
            and target.id == "ESTIMATORS" for target in node.targets))]
    assert before == after
    return revised


def prepare(output):
    workspace = SOURCE / "agent"
    records = json.loads((workspace / "laboratory_observations.json").read_text())
    saved = json.loads((workspace / "shared/reference_recalibration.json").read_text())
    old_estimators, current_estimators = estimators(LEARNED), estimators(workspace / METHOD)
    refresh_response_evidence(saved, records, records[-1]["observation"], current_estimators)
    # Keep only the actual solver inputs and prior selection state, not repeated
    # nested histories, transcripts or the multi-megabyte research request.
    state = {key: deepcopy(saved[key]) for key in ("completed", "target_numbers",
        "reference_numbers", "research_table", "response_settings", "control_grids",
        "control_grid", "response_research")}
    state["response_research"] = {key: deepcopy(state["response_research"][key])
        for key in ("family_research", "conditional_simplification")
        if key in state["response_research"]}
    spec = importlib.util.spec_from_file_location("saved_response_binding", workspace / BINDING)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    binding = module.bind({"current_observation": records[-1]["observation"]}, records, {})
    evidence = {record["index"] for record in records}
    assert all(set(row["evidence"]) <= evidence for row in binding["rows"])
    sessions = json.loads((workspace / "llm_inquiries.json").read_text())
    log_previous = sessions["proposal:" + BINDING]["proposal"]["comparison_result"]
    log_state = deepcopy(state)
    for row in log_state["research_table"]["references"] + log_state["research_table"]["targets"]:
        bound = next(item for item in binding["rows"] if item["crystal"] == row["crystal"])
        row["measurements"] = {key: {"value": value, "evidence": bound["evidence"],
            "computed_by": BINDING} for key, value in bound["features"].items()}
    log_state["response_research"] = {key: deepcopy(log_previous[key]) for key in
        ("family_research", "conditional_simplification") if key in log_previous}
    assert len(records) == 576 and len(state["response_settings"]) == 353
    assert log_state["response_settings"] == state["response_settings"]
    routes = {
        "historical_narrow": (deepcopy(state), old_estimators),
        "current_broad": (deepcopy(state), current_estimators),
        "authored_log": (log_state, binding["estimators"]),
    }
    write(output / "protocol.json", {
        "checkpoint": str(SOURCE), "learned_program": str(LEARNED),
        "prefix_observations": len(records), "prefix_unique_responses": len(state["response_settings"]),
        "previous_raw_comparison_responses": saved["response_research"]["response_count"],
        "previous_log_comparison_responses": log_previous["response_count"],
        "model_calls": 0, "new_observations": 0,
        "scope": "Same paid evidence and existing solvers. Historical route uses the complete "
            "source-task learned declaration list, containing a one-term cubic; current broad "
            "route uses the current parent declarations; authored log uses the saved author bind. "
            "Each route retains the existing relevant research history. The raw history is "
            "shared by both raw routes. The log route uses its actual entry paid-family checks. "
            "Diagnostic compatibility queries for raw routes are measured separately and do not "
            "change their production selection. No current-target successful setting, fitted "
            "coefficient or hidden simulator parameter is supplied. One consumed development case.",
        "routes": {name: {"estimators": values[1], "state": name + "_input.json"}
            for name, values in routes.items()},
    })
    for name, (input_state, _) in routes.items():
        write(output / (name + "_input.json"), input_state)
    return routes


def summarize(result):
    family = result.get("family_research", {})
    conditional = result.get("conditional_simplification", {})
    return {
        "status": result["status"], "response_count": result["response_count"],
        "compute_cost": result["compute_cost"],
        "candidates": dict(Counter(item["status"] for item in result["candidates"])),
        "compatible_sources": sorted({source["family"] for item in result["candidates"]
            if item["status"] == "compatible" for source in item.get("fit_sources", [])}),
        "hypothesis_count": len(result["hypotheses"]),
        "joint_parameter_search": result.get("joint_parameter_search", []),
        "selected_number": result.get("selected_number"),
        "selected_setting": result.get("selected_setting"),
        "selected_role": family.get("selected_role"),
        "nonlocal_progress": family.get("nonlocal_progress"),
        "no_progress_cycles": family.get("no_progress_cycles"),
        "family_statuses": [{key: value for key, value in item.items() if key in
            ("id", "source_ids", "status", "reason", "compute_cost", "support", "projection")}
            for item in family.get("families", [])],
        "conditional": {key: conditional[key] for key in ("status", "adopted", "compute_cost", "assumption")
            if key in conditional},
        "operational_fallback": compact_selection(result.get("operational_fallback")),
        "experiment_points": [{key: e[key] for key in ("number", "prediction", "family") if key in e}
            for e in result.get("experiments", [])],
    }


def compact_selection(selection):
    if not selection:
        return None
    return {key: deepcopy(value) for key, value in selection.items() if key != "hypotheses"}


def physical(output, route="historical_narrow"):
    """Run the unchanged frozen controller with the original remaining limits."""
    from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
    from scripts import consolidate_response_measurement as study
    from scripts.run_discoveryworld_prerequisite_check import replay_prefix
    from hegel_machine.cognition.world_reference_response import dispatch_response_premise

    output = Path(output)
    destination = output / (route + "_physical")
    destination.mkdir(parents=True, exist_ok=False)
    workspace = SOURCE / "agent"
    records = json.loads((workspace / "laboratory_observations.json").read_text())
    research = json.loads((workspace / "research_state.json").read_text())
    state = json.loads((workspace / "shared/reference_recalibration.json").read_text())
    limits = json.loads((SOURCE / "protocol.json").read_text())["limits"]
    # These arrays archive older comparisons. The actual current comparison and
    # all paid records remain; the diagnostic executes no LLM or historical lookup.
    state.pop("response_research_history", None)
    state.pop("bound_response_research", None)
    dispatch = route == "retained_dispatch"
    if not dispatch:
        comparison_route = "historical_narrow" if route == "historical_declarations_current_controller" else route
        state["response_research"] = json.loads((output / (comparison_route + "_comparison.json")).read_text())
        state["response_research"]["question_id"] = "paid_prefix_comparison"
    config = json.loads((ROOT / "artifacts/response_controller_adoption_20261009/protocol.json").read_text())
    study.CASES = {"task_01": config["cases"]["task_01"]["config"]}
    lab = study.TransferSignalLab("task_01")
    replay = replay_prefix(lab, {"records": records, "environment_steps": len(records)})
    before = deepcopy(research["costs"])
    before.update(observation_cost=before["environment_steps"],
                  compute_cost=before["research_compute_units"])
    with TemporaryDirectory(prefix="hegel-incumbent-controller-") as directory:
        agent = DiscoveryWorldAgent(None, Path(directory), arm="hegel", limits=limits)
        agent.lab, agent.observations = lab, deepcopy(records)
        agent.costs = deepcopy(before)
        agent.started -= before["wall_time_seconds"]
        agent.questions = [{key: deepcopy(value) for key, value in q.items() if key != "answer"}
            for q in research["questions"]]
        path = agent.workspace / METHOD
        path.parent.mkdir(parents=True, exist_ok=True)
        code = (historical_declarations_on_current_controller()
            if route == "historical_declarations_current_controller" else
            (LEARNED if route == "historical_narrow" else workspace / METHOD).read_text())
        path.write_text(code)
        write(agent.workspace / "shared/reference_recalibration.json", state)
        started = monotonic()
        dispatch_result = None
        if dispatch:
            class NoModelCalls:
                def __call__(self, *args, **kwargs):
                    raise AssertionError("The offline dispatch diagnostic must not call a model")

            agent.provider = NoModelCalls()
            # The saved episode used its model allowance. Two unspendable
            # slots open the dispatch precondition, solely for this integration
            # check; the sentinel fails instead of making a provider request.
            agent.limits["max_model_calls"] = before["model_calls"] + 2
            sessions = json.loads((workspace / "llm_inquiries.json").read_text())
            author = sessions["proposal:" + BINDING]
            agent.llm_inquiries = {"proposal:" + BINDING: {
                "question_id": author["question_id"], "notes": author["notes"], "proposal": {
                    "comparison_result": deepcopy(author["proposal"]["comparison_result"])}}}
            dispatched = dispatch_response_premise(agent, lab)
            advice = deepcopy(agent.research_advice)
            assert not dispatched and advice["diagnosis"] == "retained_response_experiment"
            assert agent.costs["model_calls"] == before["model_calls"]
            write(destination / "dispatch_advice.json", advice)
            dispatch_result = {"returned": dispatched, "advice": advice,
                "source_author": "proposal:" + BINDING,
                "original_model_call_ceiling": limits["max_model_calls"],
                "diagnostic_model_call_ceiling": agent.limits["max_model_calls"],
                "scope": "Actual source author comparison and raw paid state passed through the "
                    "production dispatcher. Two sentinel-only model slots satisfy its entry "
                    "condition; no model can be called. The runner executes the returned existing "
                    "action, so this tests host integration, not autonomous LLM adoption."}
            next_action = advice["next_action"]
            execution = agent._execute(next_action["action"], next_action["arguments"], lab)
        else:
            execution = agent._run_procedure(lab, METHOD, max_steps=512)
        elapsed = monotonic() - started
        final_state = json.loads((agent.workspace / "shared/reference_recalibration.json").read_text())
        after = {**agent.costs, "observation_cost": lab.environment_steps,
                 "compute_cost": agent.costs["research_compute_units"],
                 "wall_time_seconds": before["wall_time_seconds"] + elapsed}
        result = {
            "route": route, "replay": replay, "execution_status": execution["status"],
            "dispatch": dispatch_result,
            "source_contrast": ("Only ESTIMATORS in current controller replaced with the source-task "
                "learned declarations; every other AST statement is identical."
                if route == "historical_declarations_current_controller" else
                "Unchanged complete historical source" if route == "historical_narrow" else
                "Unchanged current source"),
            "score": lab.score(), "new_observations": len(agent.observations) - len(records),
            "new_costs": {key: value - before.get(key, 0) for key, value in after.items()},
            "limits": limits, "starting_costs": before,
            "trials": [{key: row.get(key) for key in ("number", "prediction", "family", "status", "actual_frequency", "activated")}
                for row in final_state.get("tests", [])[len(state.get("tests", [])):]],
            "last_comparison": summarize(final_state["response_research"]),
            "scope": "Frozen source program continuation, zero LLM. Paid physical prefix replayed; "
                "the original remaining world/Python/wall limits apply. Historical transcript-only "
                "comparison arrays are omitted from temporary controller memory; current scientific "
                "state, all public observations and prior question statuses are retained. No "
                "hand-selected setting or new scientific information is supplied.",
        }
        write(destination / "new_observations.json", agent.observations[len(records):])
        write(destination / "result.json", result)
        print(json.dumps({key: result[key] for key in
            ("route", "execution_status", "new_observations", "score", "trials")}), flush=True)
    return result


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    routes = prepare(output)
    summaries = {}
    for name, (state, declarations) in routes.items():
        features = {row["crystal"]: {key: value["value"] for key, value in row["measurements"].items()}
            for row in state["research_table"]["references"] + state["research_table"]["targets"]}
        started = monotonic()
        checks = {e["id"]: query_response_family(features, state["response_settings"], e,
            branch_limit=None, compute_limit=QUERY_ALLOWANCE) for e in _families(declarations)}
        check_seconds = monotonic() - started
        started = monotonic()
        result = propose_response_research(state, declarations, METHOD,
            paid_checks=checks if name == "authored_log" else None)
        seconds = monotonic() - started
        write(output / (name + "_comparison.json"), result)
        write(output / (name + "_compatibility.json"), checks)
        summary = summarize(result)
        summary.update(seconds=seconds, compatibility_seconds=check_seconds,
            compatibility_compute=sum(item["compute_cost"] for item in checks.values()),
            compatibility_statuses={key: item["status"] for key, item in checks.items()},
            compatibility_included_in_comparison_cost=name == "authored_log")
        summaries[name] = summary
        write(output / "summary.json", summaries)
        print(json.dumps({"route": name, "status": summary["status"],
            "compatible_sources": summary["compatible_sources"],
            "joint_searches": len(summary["joint_parameter_search"]),
            "setting": summary["selected_setting"], "role": summary["selected_role"],
            "compute": summary["compute_cost"], "seconds": seconds}), flush=True)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--physical", choices=("historical_narrow", "current_broad",
        "historical_declarations_current_controller", "retained_dispatch"))
    args = parser.parse_args()
    output = args.output or OUTPUT / args.label
    physical(output, args.physical) if args.physical else run(output)
