#!/usr/bin/env python3
"""Reuse the twelve-condition comparison for prepared reference-control trials."""

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]
from scripts import run_response_composition_transfer as base

OUTPUT = ROOT / "artifacts/response_control_preparation_20261011"
ARMS = ("before", "control_preparation")
DIAGNOSTIC = ("cubic_original_1", "cubic_shifted_1")
MODULES = ("world_response_consequences", "world_response_hypotheses",
    "world_reference_recalibration", "world_reference_hypothesis",
    "world_reference_expansion", "world_reference_response")
CHANGED = MODULES[:3] + ("world_reference_response",)


def runtime(output, arm):
    """Bind the selected archived scientific modules before importing the agent."""
    import hegel_machine.cognition as package
    directory = output / ("runtime_before" if arm == "before" else "runtime_candidate")
    sources = {part: directory / (part + ".py") for part in MODULES}
    for part, path in sources.items():
        name = "hegel_machine.cognition." + part
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        setattr(package, part, module)
        spec.loader.exec_module(module)
    from scripts import run_response_matched_adoption as matched
    from hegel_machine.cognition import discoveryworld_agent as agent
    checks = {
        "matched_recipe": matched.recipe is package.world_reference_recalibration,
        "agent_expansion": agent.execute_reference_expansion is package.world_reference_expansion.execute_reference_expansion,
        "agent_response": agent.execute_reference_response is package.world_reference_response.execute_reference_response,
        "response_proposal": package.world_reference_response.propose_response_research is package.world_response_hypotheses.propose_response_research}
    assert all(checks.values()), "Scientific runtime loaded after static bindings"
    return {"variant": arm, "frozen_modules": {part: str(path) for part, path in sources.items()},
        "static_bindings_match": checks,
        "scope": "Six response modules are archived per arm; remaining unchanged dependencies use this checkout."}, matched


def prepare_arm(output, arm):
    selected, matched = runtime(output, arm)
    original = json.loads((base.HISTORICAL / "without_cubic_bundle.json").read_text())
    bundle = matched.prepare_bundle(original)
    code = bundle["files"][base.METHOD]
    assert base.declarations(code) == base.declarations(original["files"][base.METHOD])
    assert len(base.declarations(code)) == 9
    selected["controller_has_basis_expansion_action"] = "basis_expansion_action" in code
    assert selected["controller_has_basis_expansion_action"]
    compile(code, str(output / f"{arm}_controller.py"), "exec")
    base.write(output / f"{arm}_bundle.json", bundle)
    base.write(output / f"{arm}_runtime.json", selected)
    (output / f"{arm}_controller.py").write_text(code)


def prepare(output):
    # Preserve every pre-change module already saved by the implementation agents.
    if (output / "protocol.json").exists():
        raise ValueError("Preserve the already prepared comparison")
    for part in CHANGED:
        assert (output / "runtime_before" / (part + ".py")).is_file()
    saved_before = sum((output / "runtime_before" / (part + ".py")).is_file() for part in MODULES)
    source = ROOT / "src/hegel_machine/cognition"
    candidate = output / "runtime_candidate"
    candidate.mkdir()
    for part in MODULES:
        shutil.copyfile(source / (part + ".py"), candidate / (part + ".py"))
        if not (output / "runtime_before" / (part + ".py")).exists():
            shutil.copyfile(source / (part + ".py"), output / "runtime_before" / (part + ".py"))
    for name in (Path(__file__).name, "run_response_composition_transfer.py", "run_response_matched_adoption.py"):
        shutil.copyfile(ROOT / "scripts" / name, output / name)
    for arm in ARMS:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--output", str(output),
                        "--prepare-arm", arm], check=True)
    previous = json.loads((ROOT / "artifacts/response_composition_export_20261011/protocol.json").read_text())
    protocol = {**previous, "arms": list(ARMS), "planned_executions": 24,
        "intervention": "Existing composition and repaired historical exporter in both arms; candidate adds conditional reference-control preparation and new-grid reference recovery verified by actual activation.",
        "runtime": f"{saved_before} pre-change scientific modules were saved before editing; {len(MODULES) - saved_before} unchanged shared adapters were filled at preparation. Six modules per arm were archived before execution. Remaining unchanged dependencies use this checkout.",
        "execution_stages": {"diagnostic": list(DIAGNOSTIC), "remaining": [case for case in previous["cases"] if case not in DIAGNOSTIC]},
        "budget_change": "None relative to response_composition_export_20261011; zero model calls in both arms.",
        "preparation_evidence": "Full trial metadata and reference setting recoveries are retained. Research answers retain feasible target intervals and selected control preparations before and after paid observations."}
    if (output / "baseline_source.json").exists():
        protocol["previous_revision"] = json.loads((output / "baseline_source.json").read_text())
        if "intervention" in protocol["previous_revision"]:
            protocol["intervention"] = protocol["previous_revision"]["intervention"]
    assert protocol["limits"]["max_model_calls"] == 0
    base.write(output / "protocol.json", protocol)
    base.write(output / "summary.json", base.summarize([], protocol))
    print(json.dumps({"status": "prepared", "output": str(output), "planned_executions": 24}), flush=True)


def run_case(output, case, arm):
    base.runtime = lambda chosen: runtime(output, chosen)
    base.run_case(output, case, arm)
    path = output / case / arm / "research_trace.json"
    trace = json.loads(path.read_text())
    trace["control_preparation_trials"] = [trial for trial in trace["tests"] if trial.get("control_preparation")]
    trace["control_preparation_support"] = []
    for question, entry in trace["questions"].items():
        answer = entry.get("answer", {})
        if not isinstance(answer, dict):
            continue
        for scope, comparison in (("full", answer), ("conditional", answer.get("conditional_simplification", {}))):
            search = comparison.get("family_research", {}).get("operational_search", {})
            if search:
                trace["control_preparation_support"].append({"question": question, "scope": scope,
                    "remaining_settings": (search.get("support") or {}).get("grid_count"),
                    "intervals": (search.get("support") or {}).get("intervals"),
                    "selected_experiment": (comparison.get("selection") or {}).get("selected_experiment"),
                    "suggested_trial": search.get("suggested_trial")})
    base.write(path, trace)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepare-arm", choices=ARMS)
    parser.add_argument("--run", choices=("diagnostic", "remaining", "all"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--case")
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.prepare_arm:
        prepare_arm(output, args.prepare_arm)
    elif args.case and args.arm:
        run_case(output, args.case, args.arm)
    elif args.prepare_only:
        prepare(output)
    elif args.run:
        protocol = json.loads((output / "protocol.json").read_text())
        cases = protocol["cases"] if args.run == "all" else protocol["execution_stages"][args.run]
        base.run(output, args.workers, case_names=cases, entrypoint=Path(__file__).resolve())
    else:
        parser.error("Choose --prepare-only or --run diagnostic|remaining|all")
