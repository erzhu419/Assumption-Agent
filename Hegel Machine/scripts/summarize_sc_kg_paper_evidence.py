#!/usr/bin/env python3
"""Collect existing SC-KG evidence for the paper; run no new experiments."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/kg_op_paper_integration_20261007/evidence.json"
SOURCES = {
    "program_learning": "artifacts/sc_kg_meta_prior_efficient_v030_20260910/summary.json",
    "data_reuse": "artifacts/sc_kg_optimizer_data_reuse_v030_20260910/summary.json",
    "goal_completion": "artifacts/sc_kg_goal_completion_20260911/completion_summary.json",
    "constraint_knowledge": "artifacts/sc_kg_constraint_knowledge_20260912/revision4/report_brief.json",
    "deployment_feedback": "artifacts/cognitive_direction_reset_20260912/summary.json",
}


def reduction_percent(before, after):
    return 100 * (before - after) / before


def main():
    records = {name: json.loads((ROOT / path).read_text()) for name, path in SOURCES.items()}
    completion = records["goal_completion"]["independent_confirmation"]
    knowledge = records["constraint_knowledge"]
    feedback = records["deployment_feedback"]["metrics"]
    first = completion["profile_results"]["overall"]["metrics"]
    later = knowledge["confirmation_metrics"]["overall"]["arms"]
    evidence = {
        "purpose": "Historical experiment integration into manuscript Sections 4.10, 5.16 and Appendix G",
        "new_experiments": 0,
        "sources": SOURCES,
        "scope": {
            "original_structural_properties": ["low_frequency", "orthogonality", "sparsity", "additivity"],
            "tested_baseline": "Native SC-KG with default quadratic GP features and shared paid initial observations",
            "researcher": "Offline numerical program search in a prescribed grammar; no LLM calls",
            "dimensions": {"early_program_learning": 5, "unknown_objective_confirmation": 3},
            "not_established": "Direct replacement of the full four-property prior in high-dimensional optimization",
        },
        "early_program_learning": records["program_learning"]["aggregate"],
        "same_data_optimization_control": {
            key: records["data_reuse"][key]
            for key in ("aggregate", "comparisons", "actual_new_research_calls", "actual_new_optimizer_calls")
        },
        "goal_completion_new_seed_confirmation": {
            key: completion[key]
            for key in ("cases", "profile_results", "new_costs", "goal_deployments",
                        "drift_installations", "unfinished_active_scopes", "source")
        },
        "later_new_seed_confirmation": {
            key: knowledge[key]
            for key in ("confirmation_metrics", "confirmation_costs", "confirmation_source",
                        "constraint_confirmation_metrics", "constraint_confirmation_costs",
                        "constraint_confirmation_source")
        },
        "deployment_feedback_development": feedback,
        "derived": {
            "v053_mean_gap_reduction_vs_native_percent": reduction_percent(
                first["native_sc_kg_shared_paid32"]["regret"], first["hegel_goal_completion"]["regret"]),
            "v057_mean_gap_reduction_vs_native_percent": reduction_percent(
                later["native_sc_kg_shared_paid32"]["regret"], later["hegel_constraint_knowledge"]["regret"]),
            "feedback_successor_research_compute_reduction_percent": reduction_percent(
                feedback["no_deployment_feedback"]["successor_research_compute"],
                feedback["interleaved"]["successor_research_compute"]),
        },
        "interpretation": {
            "decision_evidence": "Two separate frozen new-seed cohorts improve mean final optimality gap at 64 simulator calls per arm; extra research computation is not matched.",
            "attribution": "Early learned features do not improve over the same-data default model; later v057 ties v056 on terminal optimization, despite constraint prediction gains.",
            "reuse": "Deployment-created concepts are called in six of nine successor tasks, reducing research computation with unchanged final objectives in a development sequence.",
            "recursion": "Residual child research and deployment reopening are implemented; these optimizer comparisons do not isolate recursion from all other research components.",
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    print(OUTPUT.relative_to(ROOT))
    print(json.dumps(evidence["derived"], indent=2))


if __name__ == "__main__":
    main()
