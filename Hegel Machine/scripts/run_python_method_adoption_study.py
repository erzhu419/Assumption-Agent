#!/usr/bin/env python3
"""Ask independent authors to adopt or revise supplied historical research methods."""

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

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.python_scoped_method_learning import (
    ScopedPythonMethodLearner, scoped_revision_evidence,
)
from scripts.python_scoped_method_cases import protocol
from scripts.python_selective_method_cases import SOURCE_OUTCOMES
from scripts.run_python_method_learning_study import (
    InquiryLLMConfig, METHOD_PATH, archive_memory, execution_evidence, laboratory,
    score_episode, selected_candidate, total_costs, trial_preparer, usage_summary,
    verify_codex_subscription, write_json,
)
from scripts.run_python_selective_method_study import aggregate

BRIDGE = ROOT / "artifacts/python_current_evidence_bridge_20261006"
PREVIOUS = ROOT / "artifacts/python_scoped_method_comparison_luna_20261006"
LIBRARY_PATH = "shared/research_library/catalogue.json"
METHOD_NAMES = ("unmodified_candidate", "historical_bridge")
ADOPTION_INTERFACE = """
The available research library is listed in shared/research_library/catalogue.json.
Its two self-contained programs and compact paid source-practice evidence are
available in this workspace. They are supplied existing knowledge, not programs
you have invented. Neither is selected for you: shared/consolidated_method.py
still contains the earlier unpublished candidate. Inspect relevant evidence and
decide whether to reuse an available program unchanged or make a local revision.
Existing run {code:...} can copy a selected library source to
shared/consolidated_method.py; existing write can revise that source. You need not
retype a reusable program. Then finish {path:'shared/consolidated_method.py',
lesson:'...'} requests the normal five paid practices and publication. The library
evidence confers no current-source coverage: fresh practice must pass the original
unchanged gates. Preserve working branches and act on new practice feedback.
Only the final published self-contained source and lesson transfer. No library
files, source-practice observations or fitted predictors accompany that transfer.
"""


def study_protocol(sources):
    plan = protocol(sources)
    plan["author_limits"].update(max_model_calls=8, token_budget=300000)
    replicas = []
    for index in range(3):
        cases = deepcopy(plan["practice_cases"])
        offset = 310009 + index * 170003
        for spec in cases.values():
            spec["confirmation_seed"] += offset
        replicas.append({"id": f"author_{index}", "replica_index": index,
            "confirmation_seed_offset": offset, "practice_cases": cases,
            "library_order": list(METHOD_NAMES if index % 2 == 0 else METHOD_NAMES[::-1])})
    plan.update(study="python_existing_method_adoption_development", mode="supplied_method_adoption",
        model="gpt-6-luna", provider={"backend": "codex_cli", "reasoning_effort": "high",
            "timeout_seconds": 180}, author_replicas=replicas, conditions=["adopted_or_revised"],
        intervention="Three independent authors start from the same unpublished candidate. "
            "Two existing programs are supplied under alternating neutral file names with compact "
            "source-only practice evidence. Authors may copy a program or edit locally, then "
            "request the unchanged five-case scoped publication gate. No automatic adoption.",
        source="The same source archive and five development tasks; only fresh-confirmation "
            "streams advance by 310009 + replica_index * 170003. Existing practice-index "
            "increments remain. No prior practice coverage or target outcomes enters author context.",
        publication="Keep ScopedPythonMethodLearner unchanged: every case freshly confirms, "
            "with actual unchanged-prior reuse, revision, zero-meter scoped recovery and measured "
            "recovery in distinct cases. Supplied evidence does not confer publication.",
        transfer="Only published sources qualify. An exact supplied source with identical target "
            "specifications, limits, step limit and terminal settings reuses its ten frozen results, "
            "explicitly charging no new execution and claiming no independent transfer replication. "
            "Any modified published source executes all ten targets through the existing executor.",
        terminal_order="Finish all authors and all newly required target research before terminal "
            "scoring. Target results are never supplied to authors or used for publication.",
        failure_policy="Exactly three registered author attempts, with no adaptive continuation "
            "or replacement of failures. Unpublished authors leave ten target slots skipped.",
        accounting="Charge every new author once and only newly executed targets. Report reused "
            "source evidence and target executions separately, without charging them again. "
            "Unknown model usage remains unknown.",
        scope="Autonomous selection or local revision of supplied existing programs on familiar "
            "development tasks; not autonomous invention, unseen-family confirmation or broad "
            "reliability. A reused target outcome is historical evidence, not a new experiment.")
    return plan


def _compact_practice(case, result, evidence):
    fresh = result.get("fresh_confirmation") or {}
    return {"case": case, "status": result["status"],
        "fresh_confirmation": {key: fresh.get(key) for key in ("passed", "mse", "rows")},
        "costs": {key: result["costs"][key]
                  for key in ("observation_cost", "measurement_cost", "compute_cost")},
        "zero_meter_scoped_recoveries": len(evidence["zero_meter_scoped_recoveries"]),
        "measured_recoveries": len(evidence["measured_recoveries"])}


def source_evidence(method_sources):
    old = json.loads((PREVIOUS / "learning/result.json").read_text())
    current = {trial["case"]: trial for trial in old["practice_trials"]
               if trial["method_source"] == method_sources["unmodified_candidate"]}
    previous = [_compact_practice(case, trial, trial["scoped_revision"])
                for case, trial in sorted(current.items())]
    bridge = []
    for directory in sorted((BRIDGE / "source").glob("case_*")):
        result = json.loads((directory / "training.json").read_text())
        bridge.append(_compact_practice(directory.name, result, scoped_revision_evidence(result)))
    return {"unmodified_candidate": previous, "historical_bridge": bridge}


def library_memory(source, replica, method_sources, evidence):
    memory = archive_memory(source)
    memory["files"][METHOD_PATH] = method_sources["unmodified_candidate"]
    origins = {
        "unmodified_candidate": "Earlier LLM-authored candidate; not published by its original gate.",
        "historical_bridge": "Human engineering integration of historical 0.38/0.79/0.84 "
            "evidence-first repair into that candidate; not newly LLM-invented knowledge.",
    }
    entries = []
    for label, name in zip(("candidate_A", "candidate_B"), replica["library_order"]):
        path = f"shared/research_library/{label}.py"
        memory["files"][path] = method_sources[name]
        entries.append({"id": label, "path": path, "provenance": origins[name],
            "required_roles": ["predictor"],
            "contract": "select_actions(public_spec, records, memory); three paid 24-row "
                "control trajectories initially; existing refit, instrument, response and "
                "fresh-confirmation tools; self-contained source.",
            "prior_source_practices": deepcopy(evidence[name])})
    memory["files"][LIBRARY_PATH] = json.dumps({"candidates": entries,
        "evidence_scope": "Previously paid source practices only. No target outcomes or "
            "source coverage for this author. Select or revise using actual evidence."},
        separators=(",", ":"))
    return memory


def source_classification(source, method_sources):
    if source is None:
        return "unpublished"
    if source == method_sources["historical_bridge"]:
        return "exact_supplied"
    if source == method_sources["unmodified_candidate"]:
        return "exact_seed"
    return "modified"


def reusable_targets(source, plan, reference_plan, reference_summary, method_sources):
    condition = next((name for name in METHOD_NAMES if source == method_sources[name]), None)
    if condition is None or any(plan[key] != reference_plan[key] for key in
            ("transfer_cases", "limits", "procedure_step_limit", "evaluation")):
        return None
    rows = [deepcopy(item) for item in reference_summary["results"] if item["condition"] == condition]
    expected = {spec["id"] for spec in plan["transfer_cases"]}
    if len(rows) != len(expected) or {item["target_id"] for item in rows} != expected:
        return None
    for row in rows:
        row.update(evidence_kind="reused_frozen_execution", reference_condition=condition,
                   reference_summary=str(BRIDGE / "summary.json"))
    return rows


def prepare_author(provider, path, plan, replica, sources, method_sources, evidence):
    source = sources[plan["author_source_index"]]
    memory = library_memory(source, replica, method_sources, evidence)
    learner = ScopedPythonMethodLearner(provider, path, knowledge=memory,
        cases=replica["practice_cases"], lab_factory=laboratory,
        config=InquiryLLMConfig(**plan["author_limits"]), prepare_trial=trial_preparer(source, plan))
    learner.common_interface += ADOPTION_INTERFACE
    return learner


def run_study(output, *, provider_factory=None, auth_mode=None):
    output = Path(output).resolve()
    sources = json.loads(SOURCE_OUTCOMES.read_text())["sources"]
    method_sources = {name: (BRIDGE / f"{name}.py").read_text() for name in METHOD_NAMES}
    evidence, plan = source_evidence(method_sources), study_protocol(sources)
    reference_plan = json.loads((BRIDGE / "protocol.json").read_text())
    reference_summary = json.loads((BRIDGE / "summary.json").read_text())
    plan["provider"]["auth_mode"] = auth_mode
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", plan)
    write_json(output / "existing_source_outcomes.json", {"sources": sources})
    write_json(output / "supplied_source_evidence.json", evidence)
    for name, source in method_sources.items():
        (output / f"{name}.py").write_text(source)
    snapshot = output / "source_at_run"
    snapshot.mkdir()
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    for name in ("python_method_learning.py", "python_selective_method_learning.py",
                 "python_scoped_method_learning.py", "temporal_paired_practice.py"):
        shutil.copy2(ROOT / "src/hegel_machine/cognition" / name, snapshot / name)

    def author(replica):
        path = output / replica["id"] / "learning"
        provider = provider_factory(path) if provider_factory else CodexCodeProvider(path,
            model=plan["model"], reasoning_effort=plan["provider"]["reasoning_effort"],
            timeout_seconds=plan["provider"]["timeout_seconds"])
        learner = prepare_author(provider, path, plan, replica, sources, method_sources, evidence)

        def checkpoint(result):
            write_json(path / "active.json", result)
            event = result["events"][-1] if result["events"] else {}
            feedback = event.get("feedback", {})
            print(json.dumps({"phase": "author", "replica": replica["id"],
                "calls": result["costs"]["model_calls"], "action": event.get("action"),
                "error": feedback.get("error"),
                "coverage": feedback.get("current_source_coverage")}), flush=True)

        try:
            learned = learner.run(on_checkpoint=checkpoint)
        except Exception:
            learned = learner.result("execution_failed")
            learned["execution_error"] = traceback.format_exc()
        learned["current_source_coverage"] = learner._current_source_coverage()
        write_json(path / "result.json", learned)
        published = learned["status"] == "completed" and learned.get("method_source") is not None
        result = {"replica_id": replica["id"], "author_status": learned["status"],
            "method_published": published, "author_costs": learned["costs"],
            "author_usage": usage_summary(learned), "practice_trials": len(learned["practice_trials"]),
            "current_source_coverage": learned["current_source_coverage"],
            "source_classification": source_classification(learned.get("method_source"), method_sources),
            "final_file_classification": source_classification(
                learned["outgoing_knowledge"]["files"][METHOD_PATH], method_sources)}
        write_json(output / replica["id"] / "author_summary.json", result)
        return replica, learned, result

    # Persist each completed author before starting the next independent attempt.
    authored = [author(replica) for replica in plan["author_replicas"]]
    jobs, records = [], []
    for replica, learned, result in authored:
        reused = reusable_targets(learned.get("method_source"), plan, reference_plan,
            reference_summary, method_sources) if result["method_published"] else None
        result["results"], result["skipped"] = reused or [], []
        if not result["method_published"]:
            result["skipped"] = [{"target_id": spec["id"], "reason": "method_not_published"}
                                 for spec in plan["transfer_cases"]]
        elif reused is None:
            jobs.extend({"replica_id": replica["id"], "spec": spec,
                "method_source": learned["method_source"],
                "path": output / replica["id"] / spec["id"]} for spec in plan["transfer_cases"])
        records.append(result)

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

    with ThreadPoolExecutor(max_workers=2) as pool:
        trained = list(pool.map(train, jobs))
    # Every new research episode has stopped before any new terminal scoring.
    by_replica = {item["replica_id"]: item for item in records}
    new_results = []
    for job, payload in trained:
        result = score_episode(payload, plan)
        result.update(target_id=job["spec"]["id"], kind=job["spec"]["kind"],
            replica_id=job["replica_id"], condition="adopted_or_revised",
            evidence_kind="new_frozen_execution", workspace=str(payload["path"]),
            unchanged_prior_selected=selected_candidate(payload) is not None,
            scoped_revision=scoped_revision_evidence(payload["training"]),
            **execution_evidence(payload["agent"], payload["training"]))
        write_json(job["path"] / "evaluation.json", result)
        by_replica[job["replica_id"]]["results"].append(result)
        new_results.append(result)
    for result in records:
        result.update(target_summary=aggregate(result["results"]),
            reused_target_count=sum(row["evidence_kind"] == "reused_frozen_execution"
                                    for row in result["results"]),
            new_target_costs=total_costs([row["costs"] for row in result["results"]
                                         if row["evidence_kind"] == "new_frozen_execution"]))
        write_json(output / result["replica_id"] / "summary.json", result)
    reused = [row for result in records for row in result["results"]
              if row["evidence_kind"] == "reused_frozen_execution"]
    summary = {"study": plan["study"], "registered_authors": len(plan["author_replicas"]),
        "published_authors": sum(result["method_published"] for result in records),
        "exact_supplied_adoptions": sum(result["source_classification"] == "exact_supplied"
                                         for result in records),
        "authors": records, "new_target_count": len(new_results),
        "reused_target_records": len(reused),
        "unique_reused_target_executions": len({(row["reference_condition"], row["target_id"])
                                                 for row in reused}),
        "skipped_target_count": sum(len(result["skipped"]) for result in records),
        "new_author_costs": total_costs([result["author_costs"] for result in records]),
        "new_target_costs": total_costs([row["costs"] for row in new_results]),
        "new_work_costs": total_costs([*[result["author_costs"] for result in records],
                                      *[row["costs"] for row in new_results]]),
        "author_usage": {key: sum(result["author_usage"][key] for result in records)
                         for key in records[0]["author_usage"]},
        "source_acquisition_costs": reference_summary["source_acquisition_costs"],
        "prior_method_development_costs": reference_summary["prior_method_development_costs"],
        "reused_evidence_reference": str(BRIDGE / "summary.json"), "scope": plan["scope"]}
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run_study(args.output, auth_mode=verify_codex_subscription())
    print(json.dumps({key: value for key, value in summary.items() if key != "authors"}, indent=2))
