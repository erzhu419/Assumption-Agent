#!/usr/bin/env python3
"""Two naturally learned sources, local transfer failure, and paired deletions."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_current_offline_ablation import protocol as control_protocol
from run_narrow_research_cycle import _details
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.meta_prior_research import _sources
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_session import UnifiedResearchSession


def protocol(seed=18013, *, profile="development"):
    config = ResearchConfig(observation_budget=96, compute_budget=24000,
        minimum_improvement=1e-7, max_rounds=1, seed=seed)
    plan = control_protocol(seed, config=config, order=(1, 2, 1, 2, 4, 1, 2), measurement_budget=384)
    for task in plan["tasks"]:
        task["profile"] = profile
    plan["experiment"] = "two_naturally_learned_sources_and_local_refutation"
    plan["arms"] = {"full": [], "delete_a": ["a"], "delete_b": ["b"], "delete_both": ["a", "b"]}
    plan["arm_contracts"] = {arm: {"delete_sources_after_common_prefix": masks}
                             for arm, masks in plan["arms"].items()}
    plan["information_contract"] = (
        "Learn both independent sources through the normal empty-library session. Clone that paid "
        "two-task prefix; remove only the indicated sources and descendants with library.without. "
        "All four arms keep the same researcher and may relearn. Finish all learning before scoring.")
    plan["phases"] = ["learn_a", "learn_b", "reuse_a", "reuse_b", "changed_b", "return_a", "return_b"]
    plan["acceptance"] = (
        "Full solves and freshly confirms all seven tasks; A and B are independently learned; "
        "both are actually called before and after a measured incompatible binding of B. "
        "Report both source-deletion marginal cost vectors and their interaction, including "
        "failed research and source learning. A prior marked conditional is not deletion of its source.")
    plan["scope"] = "Two distinct response laws in one finite task family; no claim of joint structural synthesis."
    return plan


def run_cycle(plan, *, lab_factory=None):
    session = UnifiedResearchSession(ResearchConfig(**plan["research_config"]),
        tasks=plan["tasks"], measurement_budget=plan["measurement_budget"],
        library=KnowledgeLibrary(), lab_factory=lab_factory)
    prefix = session.run(max_tasks=2)
    sources = {name: set(prefix["results"][i]["research"]["concepts_added"])
               for i, name in enumerate(("a", "b"))}
    sessions, initial_sources = {}, {}
    for arm, masks in plan["arms"].items():
        state = deepcopy(prefix)
        removed = set().union(*(sources[name] for name in masks))
        state["knowledge"] = KnowledgeLibrary.from_dict(prefix["knowledge"]).without(removed).to_dict()
        initial_sources[arm] = [c["concept_id"] for c in state["knowledge"]["concepts"]]
        sessions[arm] = UnifiedResearchSession(state=state, lab_factory=lab_factory)
        sessions[arm].run()
    results = {arm: _details(run, plan) for arm, run in sessions.items()}
    full = results["full"]
    changed = full["tasks"][4]
    tolerance = plan["research_config"]["minimum_improvement"]
    all_sources = sources["a"] | sources["b"]
    checks = {
        "two_sources_learned_from_empty_library": bool(sources["a"] and sources["b"])
            and not sources["a"].intersection(sources["b"])
            and not prefix["results"][0]["available_concepts"],
        "source_b_is_independent_of_a": bool(sources["b"])
            and not sources["a"].intersection(_sources(Program.from_dict(prefix["results"][1]["research"]["program"])))
            and sources["b"] <= set(initial_sources["delete_a"]),
        "all_seven_independent_scores_pass": full["solved"] == 7,
        "all_seven_have_research_confirmation": all(t["status"] == "completed"
            and t["confirmation_loss"] is not None and t["confirmation_loss"] <= tolerance
            for t in full["tasks"]),
        "changed_binding_of_b_has_measured_counterexample": any(
            event["status"] == "counterexample" and not event["accepted"]
            and (event["validation_rows"] or 0) > 0
            and event["validation_loss"] is not None and event["validation_loss"] > tolerance
            and sources["b"].intersection(event["source_ids"] or ()) for event in changed["transfers"]),
        "b_prior_records_changed_binding": any(prior["source_concept_id"] in sources["b"]
            and prior["status"] == "conditional" and any(e["task_id"] == changed["research_task_id"]
                for e in prior["counterexamples"]) for prior in full["source_priors"]),
        "changed_task_learns_replacement": changed["solved"] and bool(changed["concepts_added"]),
        "both_original_sources_preserved": all_sources <= set(full["available_source_ids"]),
    }
    for name, positions in (("a", (2, 5)), ("b", (3, 6))):
        checks[f"{name}_actually_called_before_and_after_change"] = all(
            bool(sources[name].intersection(full["tasks"][position]["actual_call_sources"]))
            for position in positions)

    cost_keys = ("observation_cost", "measurement_cost", "compute_cost")

    def difference(left, right):
        return {key: left[key] - right[key] for key in cost_keys}

    effects = {}
    for name, present, absent, first_position in (
            ("a_given_b", "full", "delete_a", 2), ("a_without_b", "delete_b", "delete_both", 2),
            ("b_given_a", "full", "delete_b", 3), ("b_without_a", "delete_a", "delete_both", 3)):
        left, right = results[present], results[absent]
        costs = difference(left["costs"], right["costs"])
        effects[name] = {
            "present_arm": present, "absent_arm": absent,
            "present_minus_absent_sequence_cost": costs,
            "both_arms_solve_all_tasks": left["solved"] == right["solved"] == 7,
            "first_matching_successor_cost": difference(left["tasks"][first_position]["costs"],
                                                        right["tasks"][first_position]["costs"]),
            "sequence_pareto_saving": left["solved"] == right["solved"] == 7
                and all(v <= 0 for v in costs.values()) and any(v < 0 for v in costs.values()),
        }
    interaction = {key: results["full"]["costs"][key] - results["delete_a"]["costs"][key]
        - results["delete_b"]["costs"][key] + results["delete_both"]["costs"][key] for key in cost_keys}
    return {"protocol": deepcopy(plan), "source_ids": {k: sorted(v) for k, v in sources.items()},
        "initial_available_source_ids": initial_sources, "results": results,
        "cycle_checks": checks, "cycle_passed": all(checks.values()),
        "source_deletion_effects": effects, "cost_interaction": interaction,
        "external_model_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=18013)
    parser.add_argument("--profile", choices=("development", "confirmation"), default="development")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("keep the completed run and choose a new output")
    result = run_cycle(protocol(args.seed, profile=args.profile))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cycle_passed": result["cycle_passed"],
        "source_deletion_effects": result["source_deletion_effects"]}))


if __name__ == "__main__":
    main()
