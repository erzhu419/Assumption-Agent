#!/usr/bin/env python3
"""Recover omitted stopping traces from the unchanged 2026-10-05 controller."""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys
from time import process_time

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "artifacts/paper_scoped_retention_20261005"
FROZEN = ORIGINAL / "runtime"
COST_KEYS = ("observation_cost", "measurement_cost", "compute_cost")
sys.path.insert(0, str(FROZEN / "src"))
sys.path.insert(0, str(FROZEN / "scripts"))
from run_scoped_retention_comparison import _challenge, _conditional_library
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_session import UnifiedResearchSession, laboratory_for


def compact(value):
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items() if key not in {
            "program", "parent_program", "baseline_program", "original", "predictions",
            "sequence", "observations", "history", "features", "candidate"}}
    if isinstance(value, (list, tuple)):
        return [compact(item) for item in value]
    return value


def replay(apparatus, output):
    path = next(ORIGINAL.glob(f"case_{apparatus}_*.json"))
    original = json.loads(path.read_text())
    plan = original["protocol"]
    expected = original["results"]["global_discard"]["tasks"][3]
    config = ResearchConfig(**plan["research_config"])
    start = process_time()
    initial = UnifiedResearchSession(config, tasks=plan["tasks"],
        measurement_budget=plan["measurement_budget"], library=KnowledgeLibrary())
    prefix = initial.run(max_tasks=3)
    library = KnowledgeLibrary.from_dict(prefix["knowledge"])
    challenge = _challenge(plan, prefix, library, laboratory_for)
    before = _conditional_library(library, challenge)
    library = before.without({challenge["source_id"]})
    a, b = before.to_dict(), library.to_dict()
    deletion = {key: {"before": len(a[key]), "after": len(b[key])}
        for key in a if isinstance(a[key], (dict, list)) and a[key] != b[key]}
    deletion["unchanged_collections"] = [key for key in a
        if isinstance(a[key], (dict, list)) and a[key] == b[key]]
    charged = challenge["costs"]
    config = replace(config, observation_budget=config.observation_budget - charged["observation_cost"],
        compute_budget=config.compute_budget - charged["compute_cost"])
    session = UnifiedResearchSession(config, tasks=[plan["tasks"][3]], library=library,
        measurement_budget=plan["measurement_budget"] - charged["measurement_cost"])
    source = Path(UnifiedResearchAgent.run.__code__.co_filename)
    lines = source.read_text().splitlines()
    breaks, final = [], {}

    def trace(frame, event, arg):
        if frame.f_code is not UnifiedResearchAgent.run.__code__:
            return None
        names = ("attempt", "rounds", "confirmed", "relation_ambiguous",
            "relation_inputs_unresolved", "joint_roles_supported", "confirmation_loss",
            "spent_observations", "spent_measurements")
        if event == "line" and lines[frame.f_lineno - 1].strip() == "break":
            breaks.append({"line": frame.f_lineno,
                **{key: frame.f_locals.get(key) for key in names}})
        if event == "return":
            final.update({key: frame.f_locals.get(key) for key in names})
            actions = frame.f_locals.get("object_actions")
            final["attempted_object_directions"] = sorted(actions.attempted) if actions else []
            final["object_field_status"] = actions.field_status if actions else None
        return trace

    sys.settrace(trace)
    try:
        session.run()
    finally:
        sys.settrace(None)
    report = session.state["results"][0]["research"]
    evaluated = session.evaluate(**plan["evaluation"])["evaluation"][0]
    checks = {
        "prefix_costs": prefix["costs"] == original["common_prefix_costs"],
        "challenge": json.loads(json.dumps(challenge)) == original["challenge"],
        "status": report["status"] == expected["status"],
        "selected_kind": report["selected_kind"] == expected["selected_kind"],
        "costs": report["costs"] == expected["costs"],
        "confirmation_loss": report["confirmation_loss"] == expected["confirmation_loss"],
        "terminal_mse": evaluated["mse"] == expected["mse"],
    }
    result = {"apparatus_seed": apparatus, "original": str(path.relative_to(ROOT)),
        "runtime": str(FROZEN.relative_to(ROOT)), "replay_scope": "prefix, challenge, first discard task only",
        "external_model_calls": 0, "original_preserved": True, "exact_original_match": checks,
        "all_recorded_outcomes_match": all(checks.values()), "deletion": deletion,
        "final_control_state": final, "executed_breaks": breaks[-30:],
        "actions": compact(report["actions"]),
        "action_counts": dict(Counter(row["action"] for row in report["actions"])),
        "batches": [{key: row.get(key) for key in ("purpose", "observation_cost", "measurement_cost", "channels")}
            for row in report["batches"]],
        "scheduler_events": compact(session.state["scheduler_events"]),
        "revisits": compact(session.state["revisits"]),
        "pending_revisits": compact(session.state["pending_revisits"]),
        "result": {key: report.get(key) for key in ("status", "selected_kind", "confirmation_loss", "costs")},
        "terminal_mse": evaluated["mse"], "diagnostic_cpu_seconds": process_time() - start}
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"case_{apparatus}.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"case": apparatus, "match": all(checks.values()), "final": final,
        "last_break": breaks[-1:]}, sort_keys=True), flush=True)


def summarize(output):
    rows = []
    plan = json.loads((ORIGINAL / "protocol.json").read_text())
    for case in plan["cases"]:
        original = json.loads((ORIGINAL / f"case_{case['apparatus_seed']}_{case['profile']}.json").read_text())
        expected = original["results"]["global_discard"]["tasks"][3]
        if expected["solved"]:
            continue
        trace_path = output / f"case_{case['apparatus_seed']}.json"
        trace = json.loads(trace_path.read_text())
        assert trace["all_recorded_outcomes_match"], trace_path
        actions = trace["actions"]
        relation = [a for a in actions if a["action"] == "retain_relation_competition"]
        planners = [a for a in actions if a["action"] == "plan_inherited_relation_experiment"]
        remaining = {key: expected["resource_limits"][key] - expected["costs"][key]
            for key in COST_KEYS}
        # Quote another 24-row random acquisition and the observed rival set;
        # this prices existing primitives, without changing or executing policy.
        quote = {"observation_cost": 24,
            "measurement_cost": 24 * expected["evaluation_costs"]["measurement_cost"] / expected["evaluation_rows"],
            "compute_cost": 24 * (1 + len(relation[-1]["surviving_candidates"]) if relation else 1)}
        rows.append({"apparatus_seed": case["apparatus_seed"], "trace": str(trace_path.relative_to(ROOT)),
            "exact_original_match": trace["exact_original_match"],
            "cause": "relation_ambiguity_stop" if trace["final_control_state"]["relation_ambiguous"] else "false_confirmation",
            "control": trace["final_control_state"], "terminal_mse": expected["mse"],
            "remaining_resources": remaining,
            "last_planner_status": planners[-1]["forecast_status"] if planners else None,
            "last_planner_generation_cost": planners[-1].get("generation_cost") if planners else None,
            "surviving_rivals": len(relation[-1]["surviving_candidates"]) if relation else 0,
            "additional_random_comparison_quote": quote,
            "additional_random_comparison_affordable": all(quote[key] <= remaining[key] for key in COST_KEYS),
            "scheduler_events": len(trace["scheduler_events"]), "revisits": len(trace["revisits"]),
            "pending_revisits": len(trace["pending_revisits"]), "deletion": trace["deletion"],
            "diagnostic_cpu_seconds": trace["diagnostic_cpu_seconds"]})
    result = {"study": "post_hoc_frozen_first_revision_stopping_audit",
        "scope": "All original first-revision discard failures; original outcomes unchanged; no external model calls.",
        "original_cohort": len(plan["cases"]), "replayed_failures": len(rows),
        "exact_matches": len(rows), "causes": dict(Counter(row["cause"] for row in rows)),
        "unresolved_last_planners": dict(Counter(row["last_planner_status"] for row in rows
            if row["cause"] == "relation_ambiguity_stop")),
        "unresolved_confirmation_attempts": dict(Counter(str(row["control"]["attempt"] + 1) for row in rows
            if row["cause"] == "relation_ambiguity_stop")),
        "source_root": str(ORIGINAL.relative_to(ROOT)),
        "sources": {"stop": "runtime/src/hegel_machine/cognition/unified_research.py:4855,5150",
            "probe_failure": "runtime/src/hegel_machine/cognition/unified_research.py:4531",
            "local_generation_cap": "runtime/src/hegel_machine/cognition/competing_object_experiments.py:100",
            "source_deletion": "runtime/src/hegel_machine/cognition/memory.py:1323",
            "fresh_session_and_queue": "runtime/src/hegel_machine/cognition/unified_session.py:737"},
        "interpretation": "The 16 unresolved outcomes are stopping-policy failures to finish distinguishing relation candidates, not six-round or global account exhaustion. Four false confirmations instead stop after fresh agreement and fail independent evaluation. The intervention measures retained knowledge together with this frozen search/stopping controller.",
        "limitations": ["Additional random comparisons are affordable arithmetic quotes, not executed actions; usefulness and successful recovery are untested.",
            "No improved controller or ordinary executable-library baseline was tested; do not replace the original result or claim deletion is inherently impossible.",
            "Only the 20 first-revision terminal failures were replayed; this audit does not diagnose the separate last-return failure.",
            "Diagnostic replay CPU includes instrumentation and must not replace original experiment timing."],
        "cases": rows}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in {"cases", "sources"}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--apparatus", type=int)
    group.add_argument("--summarize", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/paper_retention_stopping_audit_20261007")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.output)
    else:
        replay(args.apparatus, args.output)
