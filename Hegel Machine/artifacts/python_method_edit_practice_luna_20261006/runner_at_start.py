#!/usr/bin/env python3
"""Revise an adopted program after a real response-validation counterexample."""

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

from scripts.run_python_method_adoption_study import (
    BRIDGE, SOURCE_OUTCOMES, CodexCodeProvider, InquiryLLMConfig, METHOD_PATH,
    ScopedPythonMethodLearner, aggregate, archive_memory, execution_evidence,
    laboratory, protocol, scoped_revision_evidence, score_episode, selected_candidate,
    total_costs, trial_preparer, usage_summary, verify_codex_subscription, write_json,
)
from scripts.run_temporal_counterexample_learning import supply_counterexample
from scripts.python_method_edit_practice import EditPracticeMethodLearner

DIAGNOSTIC = ROOT / "artifacts/python_late_response_diagnostic_20261006"
BASELINE = ROOT / "artifacts/python_method_reopening_luna_20261006"
KIND = "post_response_validation_response_change"
REVISION_INTERFACE = """
The inherited shared/consolidated_method.py is an existing historical program,
previously adopted by LLM authors. It preserves useful earlier branches but a new
paid execution refutes a prediction and leaves the investigation unresolved.
Inspect counterexample.json and the actual inherited source. Revise the research
program using that evidence, preserving branches that remain useful; then finish
to request paid practice and publication. The original five practice cases remain,
with one additional case. All six must freshly confirm under the same source and
the existing reuse, revision and scoped-recovery requirements. Supplied old evidence
grants no practice coverage. Only the published self-contained method and lesson
transfer; no hand-repaired program or new numerical operator is supplied.
"""


def late_case(spec):
    result = deepcopy(spec)
    result["kind"] = KIND
    result["task"]["calibration_change"] = {"after_rows": 167,
        "response_seed": result["task"]["response_seed"] + 500009}
    return result


def study_protocol(sources, *, edit_practice=False):
    plan = protocol(sources)
    plan["practice_cases"]["case_5"] = late_case(plan["practice_cases"]["case_2"])
    additions = []
    for spec in plan["transfer_cases"]:
        if spec["kind"] == "recurrent_instrument_change":
            item = late_case(spec)
            item["id"] = spec["id"].rsplit("_", 1)[0] + "_5"
            additions.append(item)
    plan["transfer_cases"].extend(additions)
    plan["target_kinds"].append(KIND)
    plan["author_limits"].update(max_model_calls=12, token_budget=600000,
        observation_budget=4800, compute_budget=3000000, execution_seconds=360.)
    replicas = []
    for index in range(3):
        cases, offset = deepcopy(plan["practice_cases"]), 810013 + index * 190027
        for spec in cases.values():
            spec["confirmation_seed"] += offset
        replicas.append({"id": f"author_{index}", "replica_index": index,
            "confirmation_seed_offset": offset, "practice_cases": cases})
    plan.update(study="python_adopted_method_response_reopening", mode="evidence_driven_method_revision",
        model="gpt-6-luna", provider={"backend": "codex_cli", "reasoning_effort": "high",
            "timeout_seconds": 180}, author_replicas=replicas, conditions=["unchanged", "revised"],
        intervention="Three independent authors start from the exact already-adopted historical "
            "program and one actual source-only paid failure. They may locally revise that source "
            "and request existing practice. No repaired source or new numerical solver is supplied.",
        source="Preserve the five original cases and append one recurrent-apparatus response change "
            "at the public 167-row checkpoint. Author confirmation streams advance by "
            "810013 + replica_index * 190027, retaining existing practice-index increments.",
        publication="The unchanged ScopedPythonMethodLearner checks all six cases and its existing "
            "reuse, revision and scoped-recovery requirements. No prescribed new repair trajectory "
            "or mandatory rejection during the revised procedure is added.",
        transfer="Preserve ten existing cross-source targets and append the new response-change "
            "condition for each of their two source archives. Unchanged control reuses its exact "
            "ten old outcomes and executes the two new targets. Each published modified program "
            "executes all twelve targets through the same tools, limits and frozen-method executor.",
        failure_policy="Three fixed author attempts; no adaptive continuation or replacement. "
            "Every unpublished author retains twelve skipped target slots.",
        terminal_order="Run the two new unchanged controls before authors, without showing their "
            "feedback. Finish every author and every new target research episode before terminal "
            "scoring. Neither source diagnostic nor author context contains terminal outcomes.",
        accounting="Charge new authors, two new controls and newly executed revised targets once. "
            "The source diagnostic and ten old controls are prior paid evidence, separately "
            "reported and not charged again. Reusing controls for multiple comparisons adds no trials.",
        source_evidence_reference=str(DIAGNOSTIC / "training.json"),
        scope="Local revision of a supplied research program after an actual paid counterexample "
            "within known apparatus families and a public change checkpoint. This is development "
            "evidence, not new hypothesis-language invention or unknown change-point generalization.")
    plan["edit_practice"] = edit_practice
    if edit_practice:
        plan.update(study="python_adopted_method_edit_practice_reopening",
            mode="edit_feedback_method_revision",
            intervention="Keep the same three author starts, source evidence, models and budgets. "
                "After an actual source edit the host runs the supplied counterexample case once "
                "through the existing paid practice path and returns its observed feedback. "
                "The author still chooses code changes and publication; no repaired code is supplied.",
            transfer="Reuse the completed baseline's twelve unchanged controls under identical "
                "target tasks, limits and frozen source. Every published modified source executes "
                "all twelve targets. Reused controls are not independent new executions.",
            terminal_order="Baseline controls are reused without exposing their outcomes to authors. "
                "Finish every author and newly required target research before new terminal scoring.",
            accounting="Charge new authors, including automatic paid practice, and newly executed "
                "revised targets. The diagnostic and twelve baseline controls retain prior costs "
                "without another charge; source acquisition remains the original archive cost.",
            control_reference=str(BASELINE / "summary.json"),
            scope=plan["scope"] + " The host initiates post-edit practice; success does not "
                "establish autonomous selection of when to begin the first experiment.")
    return plan


def reusable_controls(plan, reference_plan, reference_summary):
    old = {spec["id"]: spec for spec in reference_plan["transfer_cases"]}
    current = {spec["id"]: spec for spec in plan["transfer_cases"]}
    if any(current.get(name) != spec for name, spec in old.items()) or any(
            plan[key] != reference_plan[key] for key in ("limits", "procedure_step_limit", "evaluation")):
        raise ValueError("old control reuse requires identical target tasks and execution contract")
    results = [deepcopy(row) for row in reference_summary["results"]
               if row["condition"] == "historical_bridge"]
    if len(results) != len(old) or {row["target_id"] for row in results} != set(old):
        raise ValueError("the ten exact historical control executions must be available")
    for row in results:
        row.update(condition="unchanged", evidence_kind="reused_frozen_execution",
                   reference_summary=str(BRIDGE / "summary.json"))
    return results


def prepare_author(provider, path, plan, replica, sources, seed_source, counterexample_path):
    source = sources[plan["author_source_index"]]
    memory = archive_memory(source)
    memory["files"][METHOD_PATH] = seed_source
    learner_class = EditPracticeMethodLearner if plan.get("edit_practice") else ScopedPythonMethodLearner
    extra = {"counterexample_case": "case_5"} if plan.get("edit_practice") else {}
    learner = learner_class(provider, path, knowledge=memory, **extra,
        cases=replica["practice_cases"], lab_factory=laboratory,
        config=InquiryLLMConfig(**plan["author_limits"]), prepare_trial=trial_preparer(source, plan))
    supply_counterexample(learner, counterexample_path, case_label="case_5")
    learner.common_interface += REVISION_INTERFACE
    return learner


def target_jobs(plan, replica_id, learned, seed_source, output):
    if learned["status"] != "completed" or learned.get("method_source") is None:
        return [], [{"target_id": spec["id"], "reason": "method_not_published"}
                    for spec in plan["transfer_cases"]]
    if learned["method_source"] == seed_source:
        return [], []  # The same frozen source uses the single unchanged-control cohort.
    return [{"spec": spec, "replica_id": replica_id, "condition": "revised",
        "path": Path(output) / replica_id / spec["id"], "method_source": learned["method_source"]}
        for spec in plan["transfer_cases"]], []


def run_study(output, *, provider_factory=None, auth_mode=None, edit_practice=False):
    output = Path(output).resolve()
    sources = json.loads(SOURCE_OUTCOMES.read_text())["sources"]
    plan = study_protocol(sources, edit_practice=edit_practice)
    seed = (BRIDGE / "historical_bridge.py").read_text()
    diagnostic = json.loads((DIAGNOSTIC / "training.json").read_text())
    if plan["practice_cases"]["case_5"] != json.loads((DIAGNOSTIC / "spec.json").read_text()):
        raise ValueError("the supplied paid failure must belong to the declared additional source case")
    reference = json.loads((BRIDGE / "summary.json").read_text())
    if edit_practice:
        baseline_plan = json.loads((BASELINE / "protocol.json").read_text())
        baseline_summary = json.loads((BASELINE / "summary.json").read_text())
        if ((BASELINE / "unchanged_method.py").read_text() != seed or any(
                plan[key] != baseline_plan[key] for key in
                ("transfer_cases", "limits", "procedure_step_limit", "evaluation"))):
            raise ValueError("baseline control reuse requires the same frozen source and target contract")
        controls = deepcopy(baseline_summary["unchanged_control"]["results"])
        expected = {spec["id"] for spec in plan["transfer_cases"]}
        if len(controls) != len(expected) or {row["target_id"] for row in controls} != expected:
            raise ValueError("the complete twelve baseline controls must be available")
        for row in controls:
            row.update(evidence_kind="reused_frozen_execution",
                       reference_summary=str(BASELINE / "summary.json"))
    else:
        controls = reusable_controls(plan, json.loads((BRIDGE / "protocol.json").read_text()), reference)
    plan["provider"]["auth_mode"] = auth_mode
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", plan)
    write_json(output / "existing_source_outcomes.json", {"sources": sources})
    write_json(output / "paid_source_counterexample.json", {"tasks": [diagnostic]})
    (output / "unchanged_method.py").write_text(seed)
    shutil.copy2(__file__, output / "runner_at_start.py")
    if edit_practice:
        shutil.copy2(ROOT / "scripts/python_method_edit_practice.py", output / "edit_practice_hook_at_start.py")

    def train(job):
        spec = job["spec"]
        payload = trial_preparer(sources[spec["source_index"]], plan)(spec, job["path"],
            job["method_source"], InquiryLLMConfig(**plan["limits"]))
        started = monotonic()
        try:
            payload["training"] = payload["agent"].execute_method(payload["laboratory"],
                max_steps=plan["procedure_step_limit"])
        except Exception:
            payload["training"] = payload["agent"].result("execution_failed")
            payload["training"]["execution_error"] = traceback.format_exc()
        payload["research_wall_seconds"] = monotonic() - started
        write_json(job["path"] / "training.json", payload["training"])
        return job, payload

    control_jobs = [{"spec": spec, "replica_id": "unchanged", "condition": "unchanged",
        "path": output / "unchanged" / spec["id"], "method_source": seed}
        for spec in plan["transfer_cases"] if spec["kind"] == KIND and not edit_practice]
    with ThreadPoolExecutor(max_workers=2) as pool:
        trained_controls = list(pool.map(train, control_jobs))
    authors, jobs = [], []
    for replica in plan["author_replicas"]:
        path = output / replica["id"] / "learning"
        provider = provider_factory(path) if provider_factory else CodexCodeProvider(path,
            model=plan["model"], reasoning_effort="high", timeout_seconds=180)
        learner = prepare_author(provider, path, plan, replica, sources, seed,
                                 output / "paid_source_counterexample.json")

        def checkpoint(result):
            write_json(path / "active.json", result)
            event = result["events"][-1] if result["events"] else {}
            print(json.dumps({"phase": "author", "replica": replica["id"],
                "calls": result["costs"]["model_calls"], "action": event.get("action"),
                "coverage": event.get("feedback", {}).get("current_source_coverage")}), flush=True)

        try:
            learned = learner.run(on_checkpoint=checkpoint)
        except Exception:
            learned = learner.result("execution_failed")
            learned["execution_error"] = traceback.format_exc()
        learned["current_source_coverage"] = learner._current_source_coverage()
        write_json(path / "result.json", learned)
        added, skipped = target_jobs(plan, replica["id"], learned, seed, output)
        published = learned["status"] == "completed" and learned.get("method_source") is not None
        result = {"replica_id": replica["id"], "author_status": learned["status"],
            "method_published": published, "source_changed": bool(published and learned["method_source"] != seed),
            "author_costs": learned["costs"], "author_usage": usage_summary(learned),
            "current_source_coverage": learned["current_source_coverage"],
            "practice_trials": len(learned["practice_trials"]), "results": [], "skipped": skipped}
        write_json(output / replica["id"] / "author_summary.json", result)
        authors.append(result)
        jobs.extend(added)
    with ThreadPoolExecutor(max_workers=2) as pool:
        trained = list(pool.map(train, jobs))
    new_results, by_author = [], {item["replica_id"]: item for item in authors}
    # All authors and all new target research have stopped before terminal scoring.
    for job, payload in [*trained_controls, *trained]:
        result = score_episode(payload, plan)
        result.update(target_id=job["spec"]["id"], kind=job["spec"]["kind"],
            replica_id=job["replica_id"], condition=job["condition"],
            evidence_kind="new_frozen_execution", workspace=str(payload["path"]),
            unchanged_prior_selected=selected_candidate(payload) is not None,
            scoped_revision=scoped_revision_evidence(payload["training"]),
            **execution_evidence(payload["agent"], payload["training"]))
        write_json(job["path"] / "evaluation.json", result)
        (controls if job["condition"] == "unchanged" else by_author[job["replica_id"]]["results"]).append(result)
        new_results.append(result)
    for result in authors:
        if result["method_published"] and not result["source_changed"]:
            result["results"] = [{**deepcopy(row), "evidence_kind": "shared_control_execution"} for row in controls]
        result["target_summary"] = aggregate(result["results"])
        write_json(output / result["replica_id"] / "summary.json", result)
    summary = {"study": plan["study"], "registered_authors": len(authors),
        "published_authors": sum(item["method_published"] for item in authors),
        "changed_published_authors": sum(item["source_changed"] for item in authors), "authors": authors,
        "unchanged_control": {**aggregate(controls), "results": controls},
        "reused_control_count": len(controls) - len(trained_controls),
        "new_control_count": len(trained_controls), "new_revised_target_count": len(trained),
        "skipped_target_count": sum(len(item["skipped"]) for item in authors),
        "new_author_costs": total_costs([item["author_costs"] for item in authors]),
        "new_target_and_control_costs": total_costs([row["costs"] for row in new_results]),
        "new_work_costs": total_costs([*[item["author_costs"] for item in authors],
                                      *[row["costs"] for row in new_results]]),
        "author_usage": {key: sum(item["author_usage"][key] for item in authors)
                         for key in authors[0]["author_usage"]},
        "prior_paid_source_evidence": {"path": plan["source_evidence_reference"], "costs": diagnostic["costs"]},
        "reused_control_reference": str((BASELINE if edit_practice else BRIDGE) / "summary.json"),
        "source_acquisition_costs": reference["source_acquisition_costs"], "scope": plan["scope"]}
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--edit-practice", action="store_true",
                        help="Return paid counterexample practice immediately after source edits")
    args = parser.parse_args()
    result = run_study(args.output, auth_mode=verify_codex_subscription(), edit_practice=args.edit_practice)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"authors", "unchanged_control"}}, indent=2))
