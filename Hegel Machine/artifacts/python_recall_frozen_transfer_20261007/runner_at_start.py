#!/usr/bin/env python3
"""Fresh, zero-LLM transfer of every distinct program published in F.11."""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import resource
import shutil
import sys
from time import monotonic, process_time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from scripts import run_python_method_reopening_study as previous
from scripts.python_scoped_method_cases import case_spec

AUTHORS = ROOT / "artifacts/python_autonomous_recall_luna_20261007"
ORIGIN = ROOT / "artifacts/python_method_edit_practice_luna_20261006"
KINDS = ("unchanged", "response_only", "instrument_only", "both",
         "post_response_change", "post_instrument_change", "late_response_change", "old_return")


def read(path):
    return json.loads(Path(path).read_text())


def published_programs(summary, author_directory):
    """Literal source equality defines one executable; failed attempts stay listed."""
    programs, attempts = [], []
    for author in summary["authors"]:
        entry = {key: deepcopy(author[key]) for key in ("id", "condition", "status", "published", "costs")}
        entry["program_id"] = None
        if author["published"]:
            path = Path(author_directory) / author["id"] / "learning/result.json"
            result = read(path)
            if result["status"] != "completed" or not result.get("method_source"):
                raise ValueError("admit only an actually published final program")
            source = result["method_source"]
            program = next((item for item in programs if item["source"] == source), None)
            if program is None:
                program = {"id": f"published_{len(programs)}", "source": source,
                           "authors": [], "published_results": [], "role": "published_program"}
                programs.append(program)
            program["authors"].append(author["id"])
            program["published_results"].append(str(path))
            entry["program_id"] = program["id"]
        attempts.append(entry)
    return programs, attempts


def target_cases(sources):
    cases = []
    for source_index, source in enumerate(sources):
        if source["candidate"] is None:
            continue
        for kind_index, kind in enumerate(KINDS):
            base_kind = {"post_response_change": "post_validation_change",
                         "post_instrument_change": "post_validation_instrument_change",
                         "late_response_change": "recurrent_instrument_change",
                         "response_only": "response_parameter_change",
                         "both": "response_parameter_change", "old_return": "response_parameter_change",
                         "instrument_only": "recurrent_instrument_change"}.get(kind, kind)
            spec = case_spec(source, base_kind,
                seed_offset=24000017 + 100003 * kind_index, response_offset=34000019)
            spec.update(id=f"donor_{source_index}_{kind}", source_index=source_index, kind=kind)
            old_seed = source["case"]["apparatus_seed"]
            if kind in ("instrument_only", "both", "old_return"):
                spec["task"].update(apparatus_seed=old_seed + 44000021, instrument_mode="recurrent")
            if kind == "late_response_change":
                spec["task"]["calibration_change"] = {
                    "after_rows": 167, "response_seed": old_seed + 54000023}
            if kind == "old_return":
                spec["task"]["calibration_change"] = {"after_rows": 96,
                    "apparatus_seed": old_seed, "response_seed": old_seed, "instrument_mode": "finite2"}
            cases.append(spec)
    return cases


def protocol(sources, programs, attempts):
    original = read(ORIGIN / "protocol.json")
    cases = target_cases(sources)
    return {"study": "published_recall_program_fresh_transfer", "model": None,
        "programs": [{key: value for key, value in item.items() if key != "source"} for item in programs],
        "source_authors": attempts, "transfer_cases": cases,
        **{key: deepcopy(original[key]) for key in ("limits", "prefix", "procedure_step_limit", "evaluation")},
        "author_source_index": original["author_source_index"],
        "source_author_summary": str(AUTHORS / "summary.json"),
        "source_outcomes": str(previous.SOURCE_OUTCOMES),
        "freeze": "All actually published final sources are fixed by literal equality before any "
            "target runs. Seven equal publications create one program, not seven transfer replicas. "
            "All twenty author attempts, including failures, remain in source_authors.",
        "targets": "Three existing donor archives times eight registered conditions. New paid-prefix, "
            "confirmation and terminal trajectory seeds are fixed here. Response-only, instrument-only "
            "and joint changes use independently specified existing apparatus parameters; delayed "
            "changes and old-condition return use the existing public checkpoint interface.",
        "sample_size": "Twenty-four paired target specifications cover three donors and eight "
            "intervention types. This is a bounded mechanism extension, not a powered population test.",
        "pairing": "Every program receives the same donor predictor, paid initial trajectories, "
            "confirmation RNG seed and budgets on each target. All target research stops before "
            "terminal scoring. No outcomes select sources, cases or program changes.",
        "execution": "Existing PreparedPythonMethodAgent in code_react mode executes frozen Python. "
            "No LLM, authoring, host source selection or additional Hegel support gate is introduced.",
        "accounting": "Charge each new execution and its prefix once. Keep all prior author costs and "
            "all four archive-acquisition attempts, including the failed one, separate. Research CPU "
            "is host process CPU plus waited child user/system CPU, measured sequentially; Python "
            "execution seconds and charged compute remain separate recorded quantities.",
        "scope": "Four distinct published programs originate from one historical author regression "
            "state. Three target donor memories are not three independent method discoveries. One "
            "donor was used for authoring; two were not. These are existing finite-history/recurrent "
            "families and new parameter/intervention combinations, not open-domain generalization."}


def cpu_seconds():
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return process_time() + children.ru_utime + children.ru_stime


def train(spec, program, source, plan, path):
    start_cpu = cpu_seconds()
    payload = previous.trial_preparer(source, plan)(spec, path, program["source"],
        previous.InquiryLLMConfig(**plan["limits"]))
    payload["agent"].confirmation_rng.seed(spec["confirmation_seed"])
    start = monotonic()
    try:
        payload["training"] = payload["agent"].execute_method(payload["laboratory"],
            max_steps=plan["procedure_step_limit"])
    except Exception:
        payload["training"] = payload["agent"].result("execution_failed")
        payload["training"]["execution_error"] = traceback.format_exc()
    payload["research_wall_seconds"] = monotonic() - start
    payload["research_cpu_seconds"] = cpu_seconds() - start_cpu
    previous.write_json(path / "training.json", payload["training"])
    return payload


def aggregate(rows):
    return {"targets": len(rows),
        "confirmed_and_correct": sum(row["confirmed_and_correct"] for row in rows),
        "wrong_confirmations": sum(row["false_confirmation"] for row in rows),
        "statuses": dict(Counter(row["status"] for row in rows)),
        "costs": previous.total_costs([row["costs"] for row in rows]),
        "research_cpu_seconds": sum(row["research_cpu_seconds"] for row in rows),
        "research_wall_seconds": sum(row["research_wall_seconds"] for row in rows),
        "preparation_wall_seconds": sum(row["preparation_wall_seconds"] for row in rows),
        "terminal_cpu_seconds": sum(row["terminal_cpu_seconds"] for row in rows),
        "unchanged_prior_selected": sum(row["unchanged_prior_selected"] for row in rows)}


def run_study(output):
    output = Path(output).resolve()
    sources, authors = read(previous.SOURCE_OUTCOMES)["sources"], read(AUTHORS / "summary.json")
    programs, attempts = published_programs(authors, AUTHORS)
    programs.append({"id": "fixed_repair", "role": "existing_fixed_repair_control", "authors": [],
        "source": (previous.BRIDGE / "historical_bridge.py").read_text(),
        "source_path": str(previous.BRIDGE / "historical_bridge.py")})
    plan = protocol(sources, programs, attempts)
    output.mkdir(parents=True, exist_ok=False)
    previous.write_json(output / "protocol.json", plan)
    previous.write_json(output / "source_outcomes.json", {"sources": sources})
    for program in programs:
        (output / f"{program['id']}.py").write_text(program["source"])
    shutil.copy2(__file__, output / "runner_at_start.py")
    trained = []
    for index, spec in enumerate(plan["transfer_cases"]):
        order = programs[index % len(programs):] + programs[:index % len(programs)]
        for program in order:
            path = output / "targets" / spec["id"] / program["id"]
            payload = train(spec, program, sources[spec["source_index"]], plan, path)
            trained.append((spec, program, payload))
            print(json.dumps({"target": spec["id"], "program": program["id"],
                "status": payload["training"]["status"]}), flush=True)
    results = []
    for spec, program, payload in trained:
        start_cpu = cpu_seconds()
        row = previous.score_episode(payload, plan)
        row.update(target_id=spec["id"], source_index=spec["source_index"], kind=spec["kind"],
            program_id=program["id"], workspace=str(payload["path"]),
            research_cpu_seconds=payload["research_cpu_seconds"],
            terminal_cpu_seconds=cpu_seconds() - start_cpu,
            agent_arm=payload["training"]["agent_arm"],
            online_questions=deepcopy(payload["training"].get("online_questions", [])),
            unchanged_prior_selected=previous.selected_candidate(payload) is not None,
            **previous.execution_evidence(payload["agent"], payload["training"]))
        previous.write_json(payload["path"] / "evaluation.json", row)
        results.append(row)
    program_results = {program["id"]: {
        **aggregate(rows), "by_kind": {kind: aggregate([r for r in rows if r["kind"] == kind])
                                       for kind in KINDS},
        "by_donor": {str(index): aggregate([r for r in rows if r["source_index"] == index])
                     for index in sorted({r["source_index"] for r in rows})}}
        for program in programs for rows in [[r for r in results if r["program_id"] == program["id"]]]}
    summary = {"study": plan["study"], "registered_authors": len(attempts),
        "published_authors": sum(item["published"] for item in attempts),
        "distinct_published_programs": len(programs) - 1, "unique_target_specifications": len(plan["transfer_cases"]),
        "new_target_executions": len(results), "new_model_calls": sum(r["costs"]["model_calls"] for r in results),
        "source_author_conditions": deepcopy(authors["conditions"]),
        "source_author_attempts": [{**item, "unexecuted_target_slots": 0 if item["published"]
            else len(plan["transfer_cases"])} for item in attempts],
        "programs": program_results, "new_transfer": aggregate(results),
        "prior_author_costs": deepcopy(authors["new_author_costs"]),
        "prior_author_usage": deepcopy(authors["author_usage"]),
        "prior_source_acquisition_costs": previous.total_costs([source["costs"] for source in sources]),
        "prior_failed_source_acquisitions": sum(source["candidate"] is None for source in sources),
        "results": results, "scope": plan["scope"], "accounting": plan["accounting"]}
    previous.write_json(output / "summary.json", summary)
    previous.write_json(output / "brief.json", {key: value for key, value in summary.items()
        if key not in ("results", "source_author_attempts")})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_study(args.output)
    print(json.dumps({key: result[key] for key in ("distinct_published_programs",
        "unique_target_specifications", "new_target_executions", "new_model_calls")}, indent=2))
