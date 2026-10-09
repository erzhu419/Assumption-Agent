#!/usr/bin/env python3
"""Learn two components, revise their joint response, and reuse both versions."""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_narrow_research_cycle import _details
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary, _called_definitions
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_session import UnifiedResearchSession


def protocol(seed=10078, *, apparatus_seed=9013, profile="development"):
    config = ResearchConfig(observation_budget=384, compute_budget=360000,
        minimum_improvement=1e-7, max_rounds=6, active_probes=True, seed=seed)
    base = {"environment": "objects", "apparatus_seed": apparatus_seed, "profile": profile}
    tasks = [{**base, "readout": readout, "seed": seed + 10007 * i, **changes}
        for i, (readout, changes) in enumerate((
            ("total", {}), ("event", {}), ("total_event", {}),
            ("total_event", {"relation_radius": .45}), ("total_event", {}),
            ("total_event", {"relation_radius": .45}))) ]
    return {"experiment": "naturally_learned_composition_local_revision",
        "tasks": tasks, "research_config": asdict(config), "measurement_budget": 4608,
        "evaluation": {"length": 64, "trajectories": 4, "seed_offset": 1000003},
        "solved_threshold": 1e-6, "arms": {"full": [], "delete_a": ["a"]},
        "information_contract": "Learn A and B from an empty library, then clone the same paid "
            "two-task prefix. Delete A and derived knowledge in one arm; keep all research settings "
            "and allow relearning. Complete both continuations before independent evaluation.",
        "intervention": "Change only the exchange radius to .45. Total quantity A remains conserved; "
            "the event component B changes. Return to both old and changed apparatuses afterward.",
        "acceptance": "Full solves and confirms all six tasks; actual joint A/B calls are learned, "
            "old joint C has a measured counterexample, unchanged A remains in the revised AST, "
            "and later tasks call the original and revised compositions. Report deletion costs "
            "and quality separately; inherited source IDs alone do not prove unchanged definitions.",
        "scope": "One finite object-exchange family; local delta revision is allowed. "
            "The delete-A contrast measures A and its downstream learning effects, not a full factorial attribution."}


def run_cycle(plan):
    initial = UnifiedResearchSession(ResearchConfig(**plan["research_config"]),
        tasks=plan["tasks"], measurement_budget=plan["measurement_budget"], library=KnowledgeLibrary())
    prefix = initial.run(max_tasks=2)
    source_ids = {name: set(prefix["results"][i]["research"]["concepts_added"])
                  for i, name in enumerate(("a", "b"))}
    prefix_library = KnowledgeLibrary.from_dict(prefix["knowledge"])
    original_a = [d for d in prefix_library.definitions.values() if d.source_concept_id in source_ids["a"]]
    sessions, available = {}, {}
    for arm, masks in plan["arms"].items():
        state = deepcopy(prefix)
        removed = set().union(*(source_ids[name] for name in masks))
        state["knowledge"] = prefix_library.without(removed).to_dict()
        available[arm] = [c["concept_id"] for c in state["knowledge"]["concepts"]]
        sessions[arm] = UnifiedResearchSession(state=state)
        sessions[arm].run()
    results = {arm: _details(session, plan) for arm, session in sessions.items()}
    programs = {}
    for arm, session in sessions.items():
        programs[arm] = [Program.from_dict(entry["research"]["program"]) for entry in session.state["results"]]
        for row, program in zip(results[arm]["tasks"], programs[arm]):
            definitions = _called_definitions(program)
            row["nested_call_sources"] = sorted({d.source_concept_id for d in definitions if d.source_concept_id})
            row["called_definition_ids"] = [d.definition_id for d in definitions]
            row["unchanged_a_definition_called"] = any(d in original_a for d in definitions)
            row["program_ast"] = program.to_dict()
    full, deleted = results["full"], results["delete_a"]
    combined = set(full["tasks"][2]["concepts_added"])
    revised = set(full["tasks"][3]["concepts_added"])
    tolerance = plan["research_config"]["minimum_improvement"]
    checks = {
        "independent_components_learned_from_empty": bool(source_ids["a"] and source_ids["b"])
            and not prefix["results"][0]["available_concepts"]
            and source_ids["a"].isdisjoint(source_ids["b"])
            and source_ids["b"] <= set(available["delete_a"]),
        "all_six_independent_scores_pass": full["solved"] == 6,
        "all_six_have_fresh_confirmation": all(t["status"] == "completed"
            and t["confirmation_loss"] is not None and t["confirmation_loss"] <= tolerance
            for t in full["tasks"]),
        "joint_program_actually_calls_both_components": bool(combined and source_ids["a"] and source_ids["b"])
            and (source_ids["a"] | source_ids["b"]) <= set(full["tasks"][2]["nested_call_sources"]),
        "old_joint_has_measured_counterexample": any(e["status"] == "counterexample" and not e["accepted"]
            and (e["validation_rows"] or 0) > 0 and e["validation_loss"] is not None
            and e["validation_loss"] > tolerance and combined.intersection(e["source_ids"] or ())
            for e in full["tasks"][3]["transfers"]),
        "joint_prior_limited_after_counterexample": any(p["source_concept_id"] in combined
            and p["status"] == "conditional" and p["counterexamples"] for p in full["source_priors"]),
        "local_revision_preserves_exact_a_definition": full["tasks"][3]["unchanged_a_definition_called"],
        "changed_response_learned": bool(revised) and full["tasks"][3]["solved"],
        "old_composition_reused_on_return": bool(combined.intersection(full["tasks"][4]["nested_call_sources"])),
        "revised_composition_reused_on_successor": bool(revised.intersection(full["tasks"][5]["nested_call_sources"])),
    }
    keys = ("observation_cost", "measurement_cost", "compute_cost")
    difference = {k: full["costs"][k] - deleted["costs"][k] for k in keys}
    effect = {"full_minus_deleted_cost": difference, "solved": {a: r["solved"] for a, r in results.items()},
        "per_task_mse_difference": [a["mse"] - b["mse"] for a, b in zip(full["tasks"], deleted["tasks"])],
        "same_quality_pareto_saving": full["solved"] == deleted["solved"] == 6
            and all(v <= 0 for v in difference.values()) and any(v < 0 for v in difference.values())}
    return {"protocol": deepcopy(plan), "source_ids": {k: sorted(v) for k, v in source_ids.items()},
        "initial_available_source_ids": available, "results": results,
        "cycle_checks": checks, "cycle_passed": all(checks.values()),
        "source_deletion_effect": effect, "external_model_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=10078)
    parser.add_argument("--apparatus-seed", type=int, default=9013)
    parser.add_argument("--profile", choices=("development", "confirmation"), default="development")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve the completed result and choose another output")
    result = run_cycle(protocol(args.seed, apparatus_seed=args.apparatus_seed, profile=args.profile))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cycle_passed": result["cycle_passed"], "cycle_checks": result["cycle_checks"],
        "source_deletion_effect": result["source_deletion_effect"]}))


if __name__ == "__main__":
    main()
