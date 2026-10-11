#!/usr/bin/env python3
"""Execute the existing research cycle on paid states, without a model author."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]
OUT = ROOT / "artifacts/response_research_cycle_20261011"
FAILURE = ROOT / "artifacts/response_assignment_resume_20261011/after_failure_connected/agent"
ALIASED = ROOT / "artifacts/response_structure_transfer_20261009/aliased_original/retain_cubic/agent"
LEARNED = ROOT / "artifacts/response_family_recovery_20261009/changed_mechanism/source_after.py"
METHOD = "shared/consolidated_method.py"


def load_runtime(name):
    """Choose archived modules before any agent imports bind their functions."""
    import hegel_machine.cognition as package

    variant = "candidate" if name.endswith("auto_composition") else "before"
    directory = OUT / ("runtime_" + variant)
    sources = {part: directory / (part + ".py") for part in (
        "world_reference_recalibration", "world_reference_hypothesis",
        "world_reference_expansion", "world_reference_response")}
    if variant == "before":
        # The before snapshot omitted this unchanged dependency; its original
        # archived copy differs from candidate only in the tool description.
        sources["world_reference_hypothesis"] = ROOT / (
            "artifacts/response_assignment_resume_20261011/common_runtime/src/"
            "hegel_machine/cognition/world_reference_hypothesis.py")
    for part, path in sources.items():
        module_name = "hegel_machine.cognition." + part
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        setattr(package, part, module)
        spec.loader.exec_module(module)
    return {"variant": variant, "frozen_modules": {key: str(path) for key, path in sources.items()},
            "scope": "The four response integration modules are archived; other shared dependencies use this checkout."}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def replace_declarations(source, declarations):
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "ESTIMATORS" for t in node.targets))
    lines = source.splitlines(keepends=True)
    return "".join(lines[:node.lineno - 1]) + "ESTIMATORS = " + repr(declarations) + "\n" + "".join(lines[node.end_lineno:])


def compact_comparison(item):
    family = item.get("family_research", {})
    return {key: deepcopy(item.get(key)) for key in
            ("question_id", "status", "response_count", "compute_cost", "selected_experiment")} | {
        "no_progress_cycles": family.get("no_progress_cycles"),
        "nonlocal_progress": family.get("nonlocal_progress"),
        "selected_role": family.get("selected_role"),
        "families": [{key: row.get(key) for key in ("id", "status", "source_ids")}
                     for row in family.get("families", [])]}


def run_case(name, output=OUT, *, prepare_only=False):
    runtime = load_runtime(name)
    from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
    from hegel_machine.cognition import discoveryworld_agent as agent_module
    from scripts import run_response_matched_adoption as matched
    from scripts.run_discoveryworld_prerequisite_check import replay_prefix
    from scripts.run_response_structure_transfer import source_parts

    source = ALIASED if name.startswith("aliased_") else FAILURE
    records = json.loads((source / "laboratory_observations.json").read_text())
    research = json.loads((source / "research_state.json").read_text())
    state = json.loads((source / "shared/reference_recalibration.json").read_text())
    before = deepcopy(research["costs"])
    old_count = len(state.get("response_research_history", []))
    old_trials = {row["question"] for row in state.get("tests", [])}
    declarations, _ = source_parts((source / METHOD).read_text())
    if name.startswith("failure_old_knowledge"):
        historical, _ = source_parts(LEARNED.read_text())
        declarations += [deepcopy(item) for item in historical if item["id"] == "signal_cubic_5"]
    # Scientific declarations use the saved state; the selected frozen recipe
    # determines whether the composition trigger is present in this arm.
    code = replace_declarations((FAILURE / METHOD).read_text(), declarations)
    code = matched.prepare_bundle({"files": {METHOD: code}})["files"][METHOD]
    runtime["controller_has_basis_expansion_action"] = "basis_expansion_action" in code
    runtime["static_bindings_match"] = {
        "matched_recipe": matched.recipe is sys.modules["hegel_machine.cognition.world_reference_recalibration"],
        "agent_expansion": agent_module.execute_reference_expansion is
            sys.modules["hegel_machine.cognition.world_reference_expansion"].execute_reference_expansion,
        "agent_response": agent_module.execute_reference_response is
            sys.modules["hegel_machine.cognition.world_reference_response"].execute_reference_response,
    }
    assert all(runtime["static_bindings_match"].values())
    assert runtime["controller_has_basis_expansion_action"] == name.endswith("auto_composition")
    destination = Path(output) / name
    if destination.exists():
        raise ValueError(f"Preserve completed diagnostic output: {destination}")
    write(destination / "prepared_runtime.json", runtime)
    (destination / "controller.py").write_text(code)
    if prepare_only:
        compile(code, str(destination / "controller.py"), "exec")
        return {"case": name, "status": "prepared_only", "runtime": runtime}
    limits = {**research["limits"], "max_model_calls": before["model_calls"],
              "environment_steps": len(records) + 256, "python_seconds_per_call": 30,
              "python_seconds_per_task": before["python_execution_seconds"] + 240,
              "task_wall_seconds": before["wall_time_seconds"] + 1800}
    if name.startswith("aliased_"):
        plan = json.loads((ROOT / "artifacts/response_structure_transfer_20261009/protocol.json").read_text())
        config = plan["cases"]["aliased_original"]
    else:
        plan = json.loads((ROOT / "artifacts/response_current_public_view_live_20261010/protocol.json").read_text())
        config = plan["cases"]["changed_mechanism"]
    matched.study.CASES = {name: config}
    lab = matched.study.TransferSignalLab(name)
    replay = replay_prefix(lab, {"records": records, "environment_steps": len(records)})
    write(destination / "protocol.json", {"source": str(source), "replay": replay,
        "prefix_costs": before, "prefix_observations": len(records),
        "prefix_unique_responses": len(state["response_settings"]), "limits": limits,
        "procedure_max_steps": 256, "declarations": declarations, "runtime": runtime,
        "knowledge_source": str(LEARNED) if name.startswith("failure_old_knowledge") else None})
    (destination / "controller.py").write_text(code)
    with TemporaryDirectory(prefix="hegel-research-cycle-") as temporary:
        agent = DiscoveryWorldAgent(None, Path(temporary), arm="hegel", limits=limits)
        agent.lab, agent.observations = lab, deepcopy(records)
        shutil.copytree(source / "shared", agent.workspace / "shared", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (agent.workspace / METHOD).write_text(code)
        for field, default in (("questions", []), ("methods", []), ("method_results", []),
                               ("relation_research", {}), ("research_advice", None),
                               ("research_focus", None), ("llm_inquiries", {})):
            setattr(agent, field, deepcopy(research.get(field, default)))
        agent._prerequisite_checked_through = research.get("prerequisite_checked_through", -1)
        agent.costs = deepcopy(before)
        agent.started -= before["wall_time_seconds"]
        agent._save_state()
        agent._on_checkpoint = lambda data: write(destination / "active.json", {
            "status": data["status"], "costs": data["costs"], "score": data["score"]})
        execution = agent._run_procedure(lab, METHOD, max_steps=256)
        final = agent.result()
        after_state = json.loads((agent.workspace / "shared/reference_recalibration.json").read_text())
        final_source = (agent.workspace / METHOD).read_text()
        final_declarations, _ = source_parts(final_source)
        comparisons = [compact_comparison(row) for row in after_state.get("response_research_history", [])[old_count:]]
        final["reference_state"] = {key: after_state[key] for key in ("tests", "test") if key in after_state}
        final["observations"] = agent.observations
        confirmations = [row for row in matched.point_confirmations(final)
                         if row["question_id"] not in old_trials]
        new_records = agent.observations[len(records):]
        activations = []
        for record in new_records:
            view = matched.recipe._view(record)
            number, setting = matched.recipe._dialog(view)
            objects = matched.recipe._objects(view)
            for crystal in (4, 5):
                if number == crystal and matched.recipe._activated(objects, 100 + crystal):
                    activations.append({"index": record["index"], "number": crystal, "setting": setting})
        result = {"case": name, "source": str(source), "replay": replay,
            "status": execution["status"], "success": final["score"]["completedSuccessfully"],
            "score": final["score"], "new_observations": len(new_records),
            "new_costs": {key: value - before.get(key, 0) for key, value in final["costs"].items()},
            "first_selected_experiment": next((row["selected_experiment"] for row in comparisons
                                                if row["selected_experiment"]), None),
            "comparisons": comparisons, "point_confirmations": confirmations,
            "actual_activations": activations,
            "trials": [row for row in after_state.get("tests", []) if row["question"] not in old_trials],
            "final_phase": after_state.get("phase"), "final_completed": after_state.get("completed"),
            "added_declarations": [item for item in final_declarations if item not in declarations],
            "removed_declarations": [item for item in declarations if item not in final_declarations],
            "scope": "Existing controller continuation, no new model calls. Old cubic availability is explicit frozen knowledge reuse, not autonomous discovery or evidence of model adoption."}
        if name.endswith("auto_composition"):
            result["scope"] = "Original scientific declarations and saved evidence; production runtime must generate and evaluate any new structure itself. No historical cubic, generated AST, coefficient or target setting is supplied. Zero-LLM finite composition is not unrestricted model discovery."
            if name.startswith("failure_old_knowledge"):
                result["scope"] = "The same explicitly retained historical cubic as the earlier successful control; the new production expansion trigger is active. Any further declarations are generated by the runtime from paid evidence. No current target setting or coefficient is supplied."
            write(destination / "final_reference_state.json", after_state)
            (destination / "final_controller.py").write_text(final_source)
        write(destination / "new_observations.json", new_records)
        write(destination / "result.json", result)
        write(destination / "active.json", {"finished": True, "status": result["status"],
            "success": result["success"], "new_observations": result["new_observations"]})
        print(json.dumps({k: result[k] for k in ("case", "status", "success", "new_observations", "new_costs")}), flush=True)
        return result


def main(output=OUT, *, prepare_only=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    names = ("failure_unchanged", "failure_old_knowledge", "aliased_retained")
    write(output / "protocol.json", {"cases": names, "model_calls": 0,
        "failure_boundary": str(FAILURE), "negative_boundary": str(ALIASED),
        "historical_knowledge": str(LEARNED),
        "failure_prefix_count": 223, "prefix_counted_model_calls": 8,
        "prefix_call_note": "Seven recorded choices and one intercepted decision boundary; counted eighth call sent no provider request.",
        "shared_continuation_allowance": {"procedure_steps": 256, "environment_steps": 256,
                                         "python_seconds": 240, "wall_seconds": 1800},
        "interpretation": "The failure pair separates existing controller effectiveness with the original declarations from explicit reuse of earlier independently learned structure. The saved aliased control retains that same kind of knowledge and all paid negative feedback. No target setting, coefficient, future proof or successful current-task outcome is installed."})
    rows = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {}
        for name in names:
            command = [sys.executable, str(Path(__file__).resolve()), "--case", name,
                       "--output-dir", str(output.resolve())]
            if prepare_only:
                command.append("--prepare-only")
            jobs[pool.submit(subprocess.run, command, check=True)] = name
        for future in as_completed(jobs):
            try:
                future.result()
                if prepare_only:
                    rows.append({"case": jobs[future], "status": "prepared_only"})
                else:
                    row = json.loads((output / jobs[future] / "result.json").read_text())
                    rows.append({key: row[key] for key in ("case", "status", "success", "new_observations", "new_costs", "point_confirmations")})
            except Exception as error:
                rows.append({"case": jobs[future], "error": f"{type(error).__name__}: {error}"})
                write(output / "summary.json", rows)
                raise
            write(output / "summary.json", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("failure_unchanged", "failure_old_knowledge",
        "aliased_retained", "failure_auto_composition", "aliased_auto_composition",
        "failure_old_knowledge_auto_composition"))
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Load archived modules and export the controller without constructing a world or agent")
    args = parser.parse_args()
    if args.case:
        run_case(args.case, args.output_dir, prepare_only=args.prepare_only)
    else:
        main(args.output_dir, prepare_only=args.prepare_only)
