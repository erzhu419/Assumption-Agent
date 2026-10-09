#!/usr/bin/env python3
"""Evaluate an engineering bridge of the existing evidence-dependent repair rule."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from time import monotonic
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from hegel_machine.cognition.python_scoped_method_learning import scoped_revision_evidence
from scripts.python_scoped_method_cases import protocol
from scripts.python_selective_method_cases import SOURCE_OUTCOMES
from scripts.run_python_method_learning_study import (
    InquiryLLMConfig, METHOD_PATH, execution_evidence, score_episode, selected_candidate,
    total_costs, trial_preparer, write_json,
)
from scripts.run_python_selective_method_study import aggregate

PREVIOUS = ROOT / "artifacts/python_scoped_method_comparison_luna_20261006"
METHOD = ROOT / "src/hegel_machine/cognition/python_current_evidence_method.py"
CONDITIONS = ("prior_published", "historical_bridge", "fixed")


def run_study(output):
    output = Path(output).resolve()
    sources = json.loads(SOURCE_OUTCOMES.read_text())["sources"]
    previous_plan = json.loads((PREVIOUS / "protocol.json").read_text())
    previous = json.loads((PREVIOUS / "summary.json").read_text())
    plan = protocol(sources)
    for key in ("practice_cases", "transfer_cases", "limits", "procedure_step_limit", "evaluation"):
        if plan[key] != previous_plan[key]:
            raise ValueError(f"historical reference reuse requires the unchanged {key}")
    plan.update(study="python_current_evidence_history_bridge", mode="engineer_integrated_history",
        conditions=list(CONDITIONS), model=None, method_source=str(METHOD),
        intervention="An engineering edit of the last authored candidate removes its residual-size "
            "gate. After refutation it tries current-record refitting and uses separate-trajectory "
            "validation to select confirmation or existing measured repair. Other branches remain.",
        controller="The frozen source runs through the existing Python method executor. No author "
            "model, provider, new hypothesis family or numerical fitter is introduced.",
        source="The original source archive supplies five development cases. Other two original "
            "archives supply ten unchanged targets; each retains the existing paid prefix and limits.",
        publication="This is engineer-integrated historical reuse, not LLM method publication. "
            "All five source cases must freshly confirm and demonstrate distinct zero-meter "
            "scoped and measured recoveries before running the ten target cases.",
        practice_confirmation="The first five practice streams use confirmation_seed plus the "
            "practice index, matching the existing learner's first-attempt convention.",
        transfer="The historical_bridge source is frozen before any execution. Reuse all twenty "
            "completed prior_published/fixed reference results from the preceding comparison.",
        terminal_order="Finish all source research, check only fresh confirmation and actual "
            "repair evidence, then finish all admitted target research before any terminal scoring.",
        failure_policy="If a source case fails or the two recovery witnesses are absent, retain "
            "every source result and skip all ten historical_bridge target slots.",
        accounting="New costs include only five source executions and admitted bridge targets. "
            "Prior reference, model-authoring and source-acquisition costs are retained separately.",
        scope="Development on the existing five conditions and two target archives. This establishes "
            "the engineering bridge, not autonomous method invention or unknown change-point discovery.")
    reused = [deepcopy(item) for item in previous["results"]
              if item["condition"] in ("prior_published", "fixed")]
    expected = {(spec["id"], condition) for spec in plan["transfer_cases"]
                for condition in ("prior_published", "fixed")}
    if len(reused) != len(expected) or {(item["target_id"], item["condition"]) for item in reused} != expected:
        raise ValueError("the twenty unchanged reference episodes must be available")
    source = METHOD.read_text()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", plan)
    write_json(output / "existing_source_outcomes.json", {"sources": sources})
    (output / "historical_bridge.py").write_text(source)
    for folder in ("src/hegel_machine/cognition", "scripts"):
        shutil.copytree(ROOT / folder, output / "source_at_run" / folder,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    def train(job):
        payload = trial_preparer(job["source"], plan)(job["spec"], job["path"], source,
            InquiryLLMConfig(**plan["limits"]))
        if "practice_index" in job:
            payload["agent"].confirmation_rng.seed(
                job["spec"]["confirmation_seed"] + job["practice_index"])
        started = monotonic()
        try:
            payload["training"] = payload["agent"].execute_method(payload["laboratory"],
                max_steps=plan["procedure_step_limit"])
        except Exception:
            payload["training"] = payload["agent"].result("execution_failed")
            payload["training"]["execution_error"] = traceback.format_exc()
        payload["research_wall_seconds"] = monotonic() - started
        evidence = scoped_revision_evidence(payload["training"])
        write_json(job["path"] / "training.json", payload["training"])
        write_json(job["path"] / "scoped_revision.json", evidence)
        print(json.dumps({"phase": job["cohort"], "case": job["id"],
            "status": payload["training"]["status"], "scoped_revision": evidence}), flush=True)
        return job, payload, evidence

    source_jobs = [{"id": name, "cohort": "source", "spec": spec,
        "source": sources[plan["author_source_index"]], "path": output / "source" / name,
        "practice_index": index}
        for index, (name, spec) in enumerate(plan["practice_cases"].items())]
    with ThreadPoolExecutor(max_workers=2) as pool:
        source_trials = list(pool.map(train, source_jobs))
    zero_meter = [job["id"] for job, _, evidence in source_trials
                  if evidence["zero_meter_scoped_recoveries"]]
    measured = [job["id"] for job, _, evidence in source_trials if evidence["measured_recoveries"]]
    confirmed = all(payload["training"]["status"] == "completed"
        and (payload["training"].get("fresh_confirmation") or {}).get("passed")
        for _, payload, _ in source_trials)
    admitted = confirmed and any(a != b for a in zero_meter for b in measured)
    target_jobs = [{"id": spec["id"], "cohort": "target", "spec": spec,
        "source": sources[spec["source_index"]], "path": output / spec["id"]}
        for spec in plan["transfer_cases"]] if admitted else []
    with ThreadPoolExecutor(max_workers=2) as pool:
        target_trials = list(pool.map(train, target_jobs))

    # Terminal outcomes are unavailable until every admitted research episode stops.
    source_results, executed = [], []
    for job, payload, evidence in [*source_trials, *target_trials]:
        result = score_episode(payload, plan)
        result.update(target_id=job["id"], kind=job["spec"]["kind"],
            condition="historical_bridge", cohort=job["cohort"], workspace=str(payload["path"]),
            unchanged_prior_selected=selected_candidate(payload) is not None,
            scoped_revision=evidence, **execution_evidence(payload["agent"], payload["training"]))
        write_json(job["path"] / "evaluation.json", result)
        (source_results if job["cohort"] == "source" else executed).append(result)
    results = [*reused, *executed]
    conditions = {condition: {"registered_targets": len(plan["transfer_cases"]),
        **aggregate([item for item in results if item["condition"] == condition]),
        "by_kind": {kind: aggregate([item for item in results
            if item["condition"] == condition and item["kind"] == kind])
            for kind in plan["target_kinds"]}} for condition in CONDITIONS}
    summary = {"study": plan["study"], "mode": plan["mode"], "new_model_calls": 0,
        "new_reported_tokens": 0, "source_gate_passed": admitted,
        "source_recovery_evidence": {"zero_meter_cases": zero_meter, "measured_cases": measured},
        "source_results": source_results, "source_summary": aggregate(source_results),
        "source_costs": total_costs([item["costs"] for item in source_results]),
        "new_target_costs": total_costs([item["costs"] for item in executed]),
        "new_work_costs": total_costs([item["costs"] for item in [*source_results, *executed]]),
        "reused_reference_episodes": len(reused), "reused_reference_summary": str(PREVIOUS / "summary.json"),
        "reused_reference_costs": total_costs([item["costs"] for item in reused]),
        "prior_method_development_costs": previous["all_method_development_costs"],
        "source_acquisition_costs": previous["source_acquisition_costs"],
        "conditions": conditions, "results": results,
        "skipped": [] if admitted else [{"target_id": spec["id"], "condition": "historical_bridge",
            "reason": "source_repair_requirement_not_met"} for spec in plan["transfer_cases"]],
        "scope": plan["scope"]}
    write_json(output / "summary.json", summary)
    return summary


def complete_parent_comparison(output):
    """Add the exact unmodified source after the first bridge run, without replaying it."""
    output = Path(output).resolve()
    summary = json.loads((output / "summary.json").read_text())
    plan = json.loads((output / "protocol.json").read_text())
    sources = json.loads((output / "existing_source_outcomes.json").read_text())["sources"]
    prior_result = PREVIOUS / "learning/result.json"
    candidate = json.loads(prior_result.read_text())["outgoing_knowledge"]["files"][METHOD_PATH]
    comparison = {"condition": "unmodified_candidate", "source_result": str(prior_result),
        "source_status": "unpublished_candidate_diagnostic", "cases": plan["transfer_cases"],
        "limits": plan["limits"], "source_frozen_before_execution": True,
        "registration": "The exact source parent was added after the bridge finished to isolate "
            "the local engineering edit. Existing source and bridge trials are not repeated.",
        "terminal_order": "Complete all ten parent-candidate research episodes before scoring them.",
        "scope": "Offline diagnostic of the previous unpublished source; this is not a newly "
            "published or independently authored LLM method."}
    write_json(output / "candidate_comparison_protocol.json", comparison)
    (output / "unmodified_candidate.py").write_text(candidate)
    shutil.copy2(__file__, output / "candidate_comparison_runner.py")

    def train(spec):
        path = output / f"{spec['id']}_unmodified_candidate"
        payload = trial_preparer(sources[spec["source_index"]], plan)(spec, path, candidate,
            InquiryLLMConfig(**plan["limits"]))
        started = monotonic()
        try:
            payload["training"] = payload["agent"].execute_method(payload["laboratory"],
                max_steps=plan["procedure_step_limit"])
        except Exception:
            payload["training"] = payload["agent"].result("execution_failed")
            payload["training"]["execution_error"] = traceback.format_exc()
        payload["research_wall_seconds"] = monotonic() - started
        evidence = scoped_revision_evidence(payload["training"])
        write_json(path / "training.json", payload["training"])
        write_json(path / "scoped_revision.json", evidence)
        print(json.dumps({"phase": "unmodified_candidate", "case": spec["id"],
            "status": payload["training"]["status"]}), flush=True)
        return spec, payload, evidence

    with ThreadPoolExecutor(max_workers=2) as pool:
        trials = list(pool.map(train, plan["transfer_cases"]))
    results = []
    for spec, payload, evidence in trials:
        result = score_episode(payload, plan)
        result.update(target_id=spec["id"], kind=spec["kind"], condition="unmodified_candidate",
            cohort="target", workspace=str(payload["path"]), scoped_revision=evidence,
            unchanged_prior_selected=selected_candidate(payload) is not None,
            **execution_evidence(payload["agent"], payload["training"]))
        write_json(payload["path"] / "evaluation.json", result)
        results.append(result)
    summary["conditions"]["unmodified_candidate"] = {
        "registered_targets": len(plan["transfer_cases"]), **aggregate(results),
        "by_kind": {kind: aggregate([item for item in results if item["kind"] == kind])
                    for kind in plan["target_kinds"]}}
    summary["results"].extend(results)
    summary["candidate_comparison"] = comparison
    summary["candidate_comparison_costs"] = total_costs([item["costs"] for item in results])
    summary["historical_bridge_target_costs"] = deepcopy(summary["new_target_costs"])
    summary["new_target_costs"] = total_costs([
        summary["new_target_costs"], summary["candidate_comparison_costs"]])
    summary["new_work_costs"] = total_costs([
        summary["new_work_costs"], summary["candidate_comparison_costs"]])
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_study(args.output)
    result = complete_parent_comparison(args.output)
    print(json.dumps({key: result[key] for key in
        ("source_gate_passed", "source_summary", "conditions", "new_work_costs")}, indent=2))
