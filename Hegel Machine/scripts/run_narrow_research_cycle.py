#!/usr/bin/env python3
"""Learn, reuse, reject an incompatible transfer, and reuse the original law."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_current_offline_ablation import protocol as control_protocol, _train_arm, _compact
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.meta_prior_research import _sources
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_session import UnifiedResearchSession


def protocol(seed=12901, *, profile="development"):
    config = ResearchConfig(observation_budget=96, compute_budget=24000,
        minimum_improvement=1e-7, max_rounds=1, seed=seed)
    plan = control_protocol(seed, config=config, order=(1, 1, 2, 1), measurement_budget=384)
    plan["experiment"] = "empty_library_narrow_research_cycle"
    for task in plan["tasks"]:
        task["profile"] = profile
    plan["evaluation"].update(length=64, trajectories=4)
    plan["phases"] = ["learn", "reuse", "changed_mechanism", "return_to_original"]
    plan["source_deletion"] = (
        "After the common first task, delete its learned concepts with KnowledgeLibrary.without; "
        "retain the same researcher, paid prefix, budgets and method experience. "
        "Both continuations may learn normally thereafter.")
    plan["acceptance"] = (
        "All four full tasks meet independent MSE<1e-6; actual source calls occur in phases 1/3; "
        "phase 2 rejects an old-source transfer on measured evidence, restricts its prior, "
        "and learns a correct replacement. Report source-deletion cost effects separately "
        "from the whole no-memory algorithm control and from fixed-library performance.")
    return plan


def _train_paired_sources(plan, lab_factory=None):
    prefix_session = UnifiedResearchSession(ResearchConfig(**plan["research_config"]),
        tasks=plan["tasks"], measurement_budget=plan["measurement_budget"],
        library=KnowledgeLibrary(), lab_factory=lab_factory)
    prefix = prefix_session.run(max_tasks=1)
    source_ids = set(prefix["results"][0]["research"]["concepts_added"])
    full = UnifiedResearchSession(state=prefix, lab_factory=lab_factory)
    removed = deepcopy(prefix)
    removed["knowledge"] = KnowledgeLibrary.from_dict(prefix["knowledge"]).without(source_ids).to_dict()
    deleted = UnifiedResearchSession(state=removed, lab_factory=lab_factory)
    full.run()
    deleted.run()
    return {"full": full, "delete_learned_source": deleted}, source_ids, prefix


def _details(session, plan):
    report = session.evaluate(**plan["evaluation"])
    result = _compact(report, plan)
    for summary, entry in zip(result["tasks"], report["results"]):
        research = entry["research"]
        summary["research_task_id"] = research["task_id"]
        summary["confirmation_loss"] = research["confirmation_loss"]
        summary["actual_call_sources"] = _sources(Program.from_dict(research["program"]))
        summary["concepts_added"] = research["concepts_added"]
        summary["transfers"] = [{key: event.get(key) for key in (
            "source_ids", "status", "accepted", "validation_loss", "validation_rows",
            "condition_status", "kind")} for event in research.get("transfer", [])]
        # Actual call syntax and measured outcomes remain reviewable without
        # storing every fitted candidate or copying terminal observations.
        summary["program"] = Program.from_dict(research["program"]).describe()
    result["source_priors"] = [{
        "source_concept_id": prior["source_concept_id"], "status": prior.get("status"),
        "counterexamples": [{key: event.get(key) for key in (
            "task_id", "loss", "kind", "replacement_concept_id")}
            for event in prior.get("counterexamples", [])],
    } for prior in report["knowledge"]["meta_priors"]]
    result["available_source_ids"] = [c["concept_id"] for c in report["knowledge"]["concepts"]]
    return result


def run_cycle(plan, *, lab_factory=None):
    sessions, source_ids, prefix = _train_paired_sources(plan, lab_factory)
    for arm in ("fixed_library", "passive_experiments", "no_reuse"):
        sessions[arm] = _train_arm(plan, arm, lab_factory=lab_factory)
    # No terminal score is requested until every arm has finished learning.
    results = {arm: _details(session, plan) for arm, session in sessions.items()}
    full, deleted = results["full"], results["delete_learned_source"]
    changed = full["tasks"][2]
    rejected = any(event["status"] == "counterexample" and not event["accepted"]
        and (event["validation_rows"] or 0) > 0
        and event["validation_loss"] is not None
        and event["validation_loss"] > plan["research_config"]["minimum_improvement"]
        and source_ids.intersection(event["source_ids"] or ()) for event in changed["transfers"])
    restricted = any(prior["source_concept_id"] in source_ids and prior["status"] == "conditional" and any(
        event["task_id"] == changed["research_task_id"]
        for event in prior["counterexamples"]) for prior in full["source_priors"])
    # Task IDs are used only to locate an emitted result here, never to choose
    # a program, an experiment, or a research action.
    checks = {
        "source_learned_from_empty_library": bool(source_ids)
            and not prefix["results"][0]["available_concepts"],
        "all_four_independent_scores_pass": full["solved"] == 4,
        "all_four_have_research_confirmation": all(task["status"] == "completed"
            and task["confirmation_loss"] is not None
            and task["confirmation_loss"] <= plan["research_config"]["minimum_improvement"]
            for task in full["tasks"]),
        "successor_calls_learned_source": bool(source_ids.intersection(full["tasks"][1]["actual_call_sources"])),
        "changed_task_rejects_old_source": rejected,
        "counterexample_restricts_source_prior": restricted,
        "changed_task_learns_replacement": changed["solved"] and bool(changed["concepts_added"]),
        "return_calls_original_source": bool(source_ids.intersection(full["tasks"][3]["actual_call_sources"])),
        "original_source_is_preserved": source_ids <= set(full["available_source_ids"]),
    }
    keys = ("observation_cost", "measurement_cost", "compute_cost")
    first = {key: full["tasks"][1]["costs"][key] - deleted["tasks"][1]["costs"][key]
             for key in keys}
    same_quality = full["tasks"][1]["solved"] and deleted["tasks"][1]["solved"]
    effect = {
        "first_successor_full_minus_deleted_cost": first,
        "first_successor_same_quality": same_quality,
        "first_successor_pareto_saving": same_quality
            and all(value <= 0 for value in first.values()) and any(value < 0 for value in first.values()),
        "sequence_full_minus_deleted_cost": {key: full["costs"][key] - deleted["costs"][key] for key in keys},
    }
    return {"protocol": deepcopy(plan), "source_ids": sorted(source_ids), "results": results,
        "cycle_checks": checks, "cycle_passed": all(checks.values()), "source_deletion_effect": effect,
        "external_model_calls": 0,
        "scope": "A finite four-task offline cycle; fixed-library and whole no-memory controls "
            "are reported separately from deletion of the actually learned source."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=12901)
    parser.add_argument("--profile", choices=("development", "confirmation"), default="development")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("keep the completed run and choose a new output")
    result = run_cycle(protocol(args.seed, profile=args.profile))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cycle_passed": result["cycle_passed"],
        "cycle_checks": result["cycle_checks"], "source_deletion_effect": result["source_deletion_effect"]}))


if __name__ == "__main__":
    main()
