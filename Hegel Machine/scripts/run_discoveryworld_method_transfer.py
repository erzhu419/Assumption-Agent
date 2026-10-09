#!/usr/bin/env python3
"""Frozen source-program transfer to the registered fresh Reactor evaluation tasks."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from scripts.run_discoveryworld_pilot import SELECTION, run_study
from scripts.run_online_inquiry_llm import verify_codex_subscription
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.discoveryworld_lab import DiscoveryWorldLab
from hegel_machine.cognition.world_method_transfer import install_method_bundle
from scripts.run_discoveryworld_pilot import write_json


def make_transfer_plan(bundle_path, *, calls=64, workers=4, target_ids=None):
    selection = json.loads(SELECTION.read_text())
    bundle_path = Path(bundle_path).resolve()
    bundle = json.loads(bundle_path.read_text())
    source = Path(bundle["source"])
    source_plan = json.loads((source.parents[1] / "protocol.json").read_text())
    source_result = json.loads((source / "result.json").read_text())
    ids = target_ids or ["dw_04_normal_s3", "dw_04_normal_s4"]
    development = set(bundle.get("development_task_ids", [])) | {source_result["task_id"]}
    if development.intersection(ids):
        raise ValueError("Transfer targets must exclude development source tasks")
    tasks = [deepcopy(next(task for task in selection["tasks"] if task["id"] == task_id))
             for task_id in ids]
    arms = ["code_archive", "hegel"]
    return {
        "study": "discoveryworld_frozen_source_method_transfer",
        "split": "evaluation", "selection_file": str(SELECTION),
        "tasks": tasks, "arms": arms,
        "jobs": [{"task_id": task["id"], "arm": arm} for index, task in enumerate(tasks)
                 for arm in arms[index % 2:] + arms[:index % 2]],
        "model": deepcopy(source_plan["model"]),
        "limits": {**source_plan["limits"], "max_model_calls": calls},
        "runtime": source_plan["runtime"], "workers": workers,
        "method_bundle": str(bundle_path),
        "auth_mode": verify_codex_subscription() if calls else "not_used",
        "source": str(source), "consolidation_costs": bundle["consolidation_costs"],
        "source_prior_costs": source_result["prior_costs"], "source_last_costs": source_result["costs"],
        "source_validation": bundle["validation"],
        "freeze": "The model-authored source bundle is copied before any target is opened. "
                  "No method or prompt changes during this registered pair of targets; keep every attempt.",
        "pairing": "Both agents start each fresh official seed from zero observations with the same "
                   "source program and separate prior-world archive, model, tools and resource ceilings. "
                   "No old object bindings, questions or supported answers are restored. "
                   "Each agent chooses whether to use, adapt or ignore the supplied source.",
        "scope": "Previously registered evaluation tasks; defaults use unopened seeds3/4 because seeds1/2 "
                 "were already explored, with seed2 reused as development. Not the full panel. "
                 "Tests same-experience organization and actual transferred-program adoption. "
                 "No archive-deletion arm: this does not identify the incremental causal value of memory. "
                 "A later edited target program is adaptation, not unchanged frozen execution.",
        "outcomes": selection["outcomes"], "fairness": selection["fairness"],
    }


def run_frozen_transfer(bundle_path, output, *, target_ids=None, reference_bundle=None):
    """Run the supplied programs directly; no model can repair a target run."""
    plan = make_transfer_plan(bundle_path, calls=0, workers=1, target_ids=target_ids)
    bundles = {"current_evidence": Path(bundle_path)}
    if reference_bundle:
        bundles["source_program"] = Path(reference_bundle)
        reference = json.loads(Path(reference_bundle).read_text())
        reference_source = json.loads((Path(reference["source"]) / "result.json").read_text())
        reference_development = set(reference.get("development_task_ids", [])) | {reference_source["task_id"]}
        if reference_development.intersection(task["id"] for task in plan["tasks"]):
            raise ValueError("Transfer targets must exclude reference development source tasks")
    plan.update(study="discoveryworld_frozen_procedure_transfer",
        arms=list(bundles), jobs=[{"task_id": t["id"], "arm": arm}
                                 for t in plan["tasks"] for arm in bundles],
        model={"backend": "none", "model_calls": 0}, auth_mode="not_used",
        procedure_steps=256,
        bundles={key: str(path.resolve()) for key, path in bundles.items()},
        freeze="Each supplied program is copied unchanged before target execution; no target edits or LLM calls.",
        pairing="Each frozen program starts the same task from zero observations with identical resource limits.",
        scope="Direct execution of frozen programs on registered tasks. This compares program "
              "behavior, not complete LLM agents. No target edits, provider or automatic repair.")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", plan)
    rows = []
    for task in plan["tasks"]:
        for arm, path in bundles.items():
            bundle = json.loads(path.read_text())
            directory = output / task["id"] / arm
            agent = DiscoveryWorldAgent(None, directory / "agent", arm="hegel", limits=plan["limits"])
            with DiscoveryWorldLab(task, plan["runtime"], agent.workspace) as lab:
                agent.lab = lab
                install_method_bundle(agent, bundle, lab)
                agent._question({"id": "task", "question": "\n".join(
                    row.get("description", "") for row in lab.public_observation().get("taskProgress", []))})
                execution = agent._execute("run", {"operator": "procedure",
                    "method_id": bundle["methods"][0]["id"], "max_steps": plan["procedure_steps"]}, lab)
                result = agent.result()
                result["costs"].update(observation_cost=lab.environment_steps,
                    compute_cost=result["costs"]["research_compute_units"])
                write_json(directory / "result.json", {**result, "execution": execution})
                row = {"task_id": task["id"], "arm": arm, "status": execution.get("status", "method_unavailable"),
                    "completedSuccessfully": result["score"]["completedSuccessfully"],
                    "scoreNormalized": result["score"]["scoreNormalized"], "costs": result["costs"]}
                rows.append(row)
                write_json(output / "brief.json", rows)
                print(json.dumps(row), flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calls", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--target-ids", nargs="+", help="Registered targets, excluding development sources")
    parser.add_argument("--frozen-procedure", action="store_true", help="Execute the exact program with zero LLM calls")
    parser.add_argument("--reference-bundle", type=Path, help="Earlier program for the same frozen execution comparison")
    args = parser.parse_args()
    if args.calls < 1 or args.workers < 1:
        parser.error("calls and workers must be positive")
    if args.frozen_procedure:
        run_frozen_transfer(args.bundle, args.output, target_ids=args.target_ids,
                            reference_bundle=args.reference_bundle)
    else:
        print(json.dumps(run_study(make_transfer_plan(args.bundle, calls=args.calls, workers=args.workers,
                                                    target_ids=args.target_ids),
                                   args.output), indent=2))
