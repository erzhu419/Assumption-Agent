#!/usr/bin/env python3
"""Learn a revised research procedure from a paid parent-prediction counterexample."""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.code_provider_context import _compact_trajectory_payload
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH, method_memory_pair
from hegel_machine.cognition.temporal_paired_practice import PairedMethodLearner
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json
from scripts.run_temporal_calibration_transfer import CALIBRATIONS, run_frozen, snapshot
from scripts.run_temporal_operator_llm import laboratory
from scripts.run_temporal_paired_method import DONOR, donor_memory, task


SEED = ROOT / "artifacts/temporal_paired_state_resume_luna_20261006/learning/result.json"
COUNTEREXAMPLE = ROOT / "artifacts/temporal_calibration_transfer_luna_20261006/frozen/episode_02/training.json"


def protocol():
    cases = []

    def add(prior, apparatus_seed, response_seed, change):
        index = len(cases)
        settings = task(apparatus_seed, CALIBRATIONS[prior][1], 500003 + 10007 * index)
        settings["evaluation_seed"] = apparatus_seed + 5000003 + 100003 * index
        if response_seed is not None:
            settings["response_seed"] = response_seed
        cases.append({"case_id": f"episode_{index:02d}", "prior": prior,
                      "unchanged": change == "stable", "change": change, "task": settings})

    for prior, (seed, _) in CALIBRATIONS.items():
        add(prior, seed, None, "stable")
    for index in range(4):
        for prior, (seed, _) in CALIBRATIONS.items():
            add(prior, seed, 127106007 + 1009 * index, "response_only")
    for index in range(4):
        prior = ("first", "second")[index % 2]
        seed = 137106007 + 1009 * index
        add(prior, seed, CALIBRATIONS[prior][0], "instrument_only")
        add(prior, seed, None, "both")
    return {"study": "llm_temporal_parent_counterexample_method_revision",
        "model": "gpt-6-luna", "seed_method": str(SEED), "counterexample": str(COUNTEREXAMPLE),
        "provider": {"backend": "codex_cli", "reasoning_effort": "medium", "timeout_seconds": 180},
        "practice_cases": {"case_a": task(98106007, 2, 158191),
                           "case_b": task(98106007, 1, 162861),
                           "case_c": {**task(98106007, 1, 177013), "response_seed": 117106007}},
        "author_limits": asdict(InquiryLLMConfig(observation_budget=1440, confirmation_rows=0,
            max_model_calls=20, token_budget=1000000, compute_budget=300000, execution_seconds=180)),
        "frozen_cases": cases,
        "frozen_limits": asdict(InquiryLLMConfig(observation_budget=240, confirmation_rows=48,
            max_model_calls=0, token_budget=0, compute_budget=100000, execution_seconds=120)),
        "max_program_steps": 8, "evaluation": {"length": 256, "mse_threshold": 1e-10},
        "learning": "One actual paid failed trial is supplied with its observations and action feedback. "
            "Three opaque practice cases share the original first donor. The LLM authors the revision and "
            "chooses its practice order. No repaired program or branch implementation is supplied.",
        "publication": "The same final source must freshly confirm all three practice cases and demonstrate "
            "unchanged prior reuse and a revision. Earlier evidence is not publication coverage.",
        "transfer": "Original numerical priors plus the old or new published method and lesson, in empty "
            "workspaces on the same eighteen cases. Neither earlier failed traces nor practice-fitted models "
            "transfer. Four new response seeds per donor, and four new instrument seeds with old/new response laws.",
        "cost_scope": "Report new authoring and each frozen cohort separately. Supplied historical failure "
            "costs and previous interrupted author phases are retained separately, not charged again per task.",
        "limitations": "Explicitly prompted method repair within supplied operators and recurrent families. "
            "Paired cases share calibrations and are not independent population samples. Frozen transfer "
            "compares old versus newly authored research, not a new live-LLM retention/deletion benchmark."}


def _paid_research_events(prior):
    """Join recorded procedure declarations to their actual query events."""
    methods = {(step["step"], step["path"]): step
               for step in prior.get("method_results", [])}
    events = deepcopy(prior["events"])
    for event in events:
        if "proposal" in event:
            continue
        declared = event.get("feedback", {}).get("program_action")
        if declared is not None:
            event["proposal"] = deepcopy(declared)
        elif event.get("action") == "query":
            invocation = methods.get((event.get("step"), event.get("arguments", {}).get("path")))
            if invocation is not None and "proposal" in invocation:
                event["proposal"] = deepcopy(invocation["proposal"])
    return events


def _ledger_declaration(value, location):
    """Show decisions inline, and refer to the already saved acquisition inputs."""
    if not isinstance(value, dict):
        return deepcopy(value)
    result = deepcopy(value)
    for key in ("actions", "sequence"):
        if isinstance(result.get(key), list):
            result[key] = {"full_content_file": "counterexample_trace.json",
                           "json_path": [*location, key], "count": len(result[key])}
    if isinstance(result.get("arguments"), dict):
        result["arguments"] = _ledger_declaration(result["arguments"], [*location, "arguments"])
    return result


def _paid_evidence_ledger(prior):
    records = []
    for index, record in enumerate(prior["paid_observations"]):
        indices = record["row_indices"]
        records.append({"record": index, "row_interval": [min(indices), max(indices)],
            "rows": len(indices), "meter_rows": sum("meter" in item["values"]
                for item in record.get("measurements", []))})
    steps = []
    for index, event in enumerate(_paid_research_events(prior)):
        feedback = event.get("feedback", {})
        summary = {key: deepcopy(feedback[key]) for key in (
            "error", "method_submission", "mse", "observation_cost", "record_indices",
            "written", "returncode", "timeout", "status")
            if key in feedback}
        confirmation = feedback.get("fresh_confirmation")
        if isinstance(confirmation, dict):
            summary["fresh_confirmation"] = {key: deepcopy(confirmation[key])
                for key in ("passed", "mse", "rows") if key in confirmation}
            indices = confirmation.get("observation_indices", [])
            if indices:
                summary["fresh_confirmation"]["row_interval"] = [min(indices), max(indices)]
        outcomes = feedback.get("after_observation_results", [])
        if "operator" in feedback:
            outcomes = [feedback, *outcomes]
        operators = []
        for outcome in outcomes:
            item = {key: deepcopy(outcome[key]) for key in (
                "operator", "records", "path", "instrument", "support_rows", "error",
                "supported_fields", "refuted_fields", "ambiguous_fields", "validation",
                "final_fitting_rows", "paid_rows_at_refit", "source_model") if key in outcome}
            if isinstance(item.get("validation"), dict):
                item["validation"] = {key: item["validation"][key] for key in
                    ("mse", "returncode", "fit_available", "error", "timeout") if key in item["validation"]}
            arguments = outcome.get("arguments", event.get("arguments", {}))
            if "records" not in item and isinstance(arguments.get("records"), list):
                item["records"] = list(arguments["records"])
            split = outcome.get("split", {})
            if "training_records" in split and "validation_records" in split:
                item.update(training_records=list(split["training_records"]),
                            validation_records=list(split["validation_records"]),
                            final_fit_records=[*split["training_records"], *split["validation_records"]])
                item["records"] = list(split.get("records", item["final_fit_records"]))
            if "candidates" in outcome:
                item["candidate_validation"] = [{key: candidate[key]
                    for key in ("path", "validation_mse") if key in candidate}
                    for candidate in outcome["candidates"]]
            operators.append(item)
        if operators:
            summary["operators"] = operators
        steps.append({"step": event.get("step", index), "action": event["action"],
                      **{key: _ledger_declaration(event[key], [index, key])
                         for key in ("arguments", "proposal") if key in event},
                      "feedback": summary})
    return records, steps


def supply_counterexample(learner, path, *, case_label="case_c"):
    """Supply a paid single-task result or an existing tasks-envelope file."""
    payload = deepcopy(path) if isinstance(path, dict) else json.loads(Path(path).read_text())
    prior = payload["tasks"][0] if "tasks" in payload else payload
    records, steps = _paid_evidence_ledger(prior)
    trace = {"practice_case": case_label, "status": prior["status"], "costs": prior["costs"],
        "paid_records": records, "steps": steps,
        "full_trace_file": "counterexample_trace.json",
        "paid_records_file": "counterexample_observations.json",
        "use": "An earlier paid execution on this case. Investigate its rejected predictions, "
               "repair choices and costs. These earlier trials do not confirm your new source."}
    source = prior.get("method_source")
    method_path = prior.get("method_path") or METHOD_PATH
    source_record = "Earlier trial method_source."
    if source is None:
        source = prior.get("outgoing_knowledge", {}).get("files", {}).get(method_path)
        source_record = "Earlier trial outgoing_knowledge.files; this may differ from the inherited starting method."
    if source is not None:
        trace["executed_method"] = {"path": method_path, "source": source,
            "source_record": source_record}
    text = json.dumps(trace, indent=2)
    if source is not None and len(text) > 12000:
        source_file = "counterexample_method.py"
        (learner.workspace / source_file).write_text(source)
        trace["executed_method"].pop("source")
        trace["executed_method"]["source_file"] = source_file
        text = json.dumps(trace, indent=2)
    (learner.workspace / "counterexample.json").write_text(text)
    learner.files["counterexample.json"] = text
    write_json(learner.workspace / "counterexample_trace.json",
               _compact_trajectory_payload(_paid_research_events(prior)))
    write_json(learner.workspace / "counterexample_observations.json", prior["paid_observations"])
    learner.common_interface += ("\nAn earlier paid execution and its compact record/operation chronology are in "
        "counterexample.json; its executed_method, when present, identifies the program used in that trace, "
        "with complete source inline or at source_file. Complete paid records "
        "are in counterexample_observations.json. They are available to author inspection, not automatically "
        "copied into new practice tasks. Learn a reusable revision from this evidence and test it through "
        "your program. Keep original prior files intact; the final source must work on fresh task data.\n")
    return deepcopy(prior["costs"])


def run_study(output, *, provider_factory, auth_mode=None, resume_practice=None,
              study_protocol=None, learner_class=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise ValueError("use a new output directory")
    plan = deepcopy(study_protocol) if study_protocol is not None else protocol()
    previous = None
    if resume_practice is not None:
        resume_path = Path(resume_practice)
        plan = json.loads((resume_path.parent.parent / "protocol.json").read_text())
        previous = json.loads(resume_path.read_text())
        plan["resume_practice"] = str(resume_path)
    if auth_mode:
        plan["provider"]["auth_mode"] = auth_mode
    write_json(output / "protocol.json", plan)
    snapshot(output)
    for name in ("scripts/run_temporal_counterexample_learning.py",
                 "src/hegel_machine/cognition/temporal_paired_practice.py"):
        destination = output / "source_at_run" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    seed = json.loads(Path(plan["seed_method"]).read_text())
    seed["lesson"] = seed["outgoing_knowledge"]["lessons"][-1]
    original = donor_memory(DONOR, 1, "shared/meter.json")
    memory = method_memory_pair(original, seed)[0]
    path = output / "learning"
    learner = (learner_class or PairedMethodLearner)(provider_factory(path), path, knowledge=memory,
        cases=plan["practice_cases"], lab_factory=laboratory,
        config=InquiryLLMConfig(**plan["author_limits"]))
    historical_costs = supply_counterexample(learner, plan["counterexample"],
                                            case_label=plan.get("counterexample_case", "case_c"))
    if previous is not None:
        write_json(output / "restored_practice.json", learner.restore_practice(previous))
    learned = learner.run(on_checkpoint=lambda result: write_json(path / "active.json", result))
    learned_path = path / "result.json"
    write_json(learned_path, learned)
    summary = {"study": plan["study"], "learning_status": learned["status"],
               "new_learning_costs": learned["costs"], "reused_counterexample_costs": historical_costs}
    if previous is not None:
        summary["previous_author_phase"] = {"path": str(resume_practice),
                                            "status": previous["status"], "costs": previous["costs"]}
    if learned["status"] != "completed":
        summary["status"] = "method_not_published"
        write_json(output / "summary.json", summary)
        return summary
    # Both protocols were fixed before authoring. Only publication unlocks the
    # second donor and held-out settings; the agent never sees their results.
    summary["original_method"] = run_frozen(plan.get("comparison_method", plan["seed_method"]),
                                            output / "original", plan)
    summary["revised_method"] = run_frozen(learned_path, output / "revised", plan)
    summary["status"] = "completed"
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-practice", type=Path,
                        help="Restore an interrupted provider-failed author phase, preserving its paid trials")
    args = parser.parse_args()
    auth = verify_codex_subscription()
    print(json.dumps(run_study(args.output, auth_mode=auth, resume_practice=args.resume_practice,
        provider_factory=lambda path: CodexCodeProvider(path, model="gpt-6-luna",
            reasoning_effort="medium", timeout_seconds=180)), indent=2))


if __name__ == "__main__":
    main()
