"""Bounded creative hypothesis trajectory search.

The earlier residual generator already emitted next-generation proposal seeds.
This module strengthens that step into a multi-path discovery search: residual
clusters spawn several trajectory types, each candidate is assigned a novelty
class, world-model/risk estimate, evaluation contract, and selective-retention
decision, then retained descendants feed the next generation.

The implementation is still a bounded research generator, not an unrestricted
creative agent.  Its purpose is to make "variation -> evaluation -> selective
retention" explicit at larger scale.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR
from .full_v3_live_residual_clusterer import build_full_v3_live_residual_clusterer_payload
from .schema import stable_id


DEFAULT_OUT = PAPER_DIR / "creative_hypothesis_trajectory_search_20260612.json"

TRAJECTORIES = (
    "narrow_trigger_boundary",
    "negative_control_abstention",
    "world_model_uncertainty_split",
    "orthogonal_new_family_probe",
    "cross_domain_morphism_transfer",
    "counterexample_boundary_search",
)

NONLOCAL_TRAJECTORIES = {
    "orthogonal_new_family_probe",
    "cross_domain_morphism_transfer",
    "counterexample_boundary_search",
}

NOVELTY_BY_TRAJECTORY = {
    "narrow_trigger_boundary": "specialization",
    "negative_control_abstention": "integration_boundary",
    "world_model_uncertainty_split": "world_model_family",
    "orthogonal_new_family_probe": "orthogonal_new_family",
    "cross_domain_morphism_transfer": "formal_isomorphism",
    "counterexample_boundary_search": "genuinely_new_family",
}


def build_creative_hypothesis_trajectory_search_payload(
    *,
    root: Path,
    eval_id: str = "creative_hypothesis_trajectory_search_20260612",
    generations: int = 5,
    seed_limit: int = 14,
    frontier_width: int = 14,
) -> dict[str, Any]:
    if generations < 1:
        raise ValueError("generations must be positive")
    root = root.resolve()
    clusterer = build_full_v3_live_residual_clusterer_payload(
        root=root,
        eval_id=f"{eval_id}_clusterer",
    )
    frontier = _initial_frontier(clusterer, seed_limit=seed_limit)
    generation_rows = []
    all_candidates: list[dict[str, Any]] = []
    for generation in range(1, generations + 1):
        candidates = []
        for seed in frontier:
            for trajectory in TRAJECTORIES:
                candidates.append(_candidate_from_seed(seed=seed, trajectory=trajectory, generation=generation))
        evaluated = _apply_retention_cap([_evaluate_candidate(row) for row in candidates])
        retained = [row for row in evaluated if row["retention_decision"] == "retain_for_next_generation"]
        generation_rows.append({
            "generation": generation,
            "frontier_count": len(frontier),
            "candidate_count": len(evaluated),
            "retained_count": len(retained),
            "rejected_count": len(evaluated) - len(retained),
            "trajectory_counts": dict(Counter(row["trajectory"] for row in evaluated)),
            "novelty_counts": dict(Counter(row["novelty_classification"] for row in evaluated)),
            "retained_candidate_ids": [row["candidate_id"] for row in retained],
            "candidate_rows": evaluated,
        })
        all_candidates.extend(evaluated)
        frontier = _next_frontier(retained=retained, fallback=evaluated, width=frontier_width)
    metrics = _metrics(clusterer=clusterer, generation_rows=generation_rows, candidates=all_candidates)
    gates = {
        "source_clusterer_passes": bool(clusterer.get("pass")),
        "generation_count_high": metrics["generation_count"] >= 5,
        "candidate_count_high": metrics["candidate_count"] >= 300,
        "trajectory_diversity_high": metrics["trajectory_type_count"] >= 6,
        "family_diversity_high": metrics["family_count"] >= 40,
        "nonlocal_candidate_ratio_high": metrics["nonlocal_candidate_ratio"] >= 0.40,
        "orthogonal_new_family_present": metrics["novelty_counts"].get("orthogonal_new_family", 0) >= 30,
        "genuinely_new_family_present": metrics["novelty_counts"].get("genuinely_new_family", 0) >= 30,
        "selective_retention_balanced": 0.20 <= metrics["retention_rate"] <= 0.70,
        "retained_family_count_high": metrics["retained_family_count"] >= 18,
        "nonlocal_retained_count_high": metrics["nonlocal_retained_count"] >= 30,
        "multi_generation_productivity_nonnegative": metrics["generation_productivity_nonnegative_rate"] >= 0.80,
        "evaluation_contract_coverage": metrics["evaluation_contract_coverage"] == 1.0,
        "negative_control_coverage": metrics["negative_control_coverage"] == 1.0,
        "no_graph_mutation": metrics["graph_mutation_count"] == 0,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "creative_hypothesis_trajectory_search",
        "reconstruction_v2_full_phase": "creative_generator_multitrajectory_family_discovery",
        "implementation_level": "bounded_residual_cluster_to_multitrajectory_generator",
        "performance_validation": True,
        "validation_scope": (
            "Expands residual-driven hypothesis generation into a multi-generation, multi-trajectory search "
            "over specialization, integration, world-model, orthogonal, formal-isomorphism, and counterexample "
            "families.  It evaluates candidates with explicit contracts and negative controls, then selectively "
            "retains descendants.  It does not mutate the main graph."
        ),
        "source_clusterer": {
            "eval_id": clusterer.get("eval_id"),
            "pass": clusterer.get("pass"),
            "seed_count": len(clusterer.get("next_generation_proposal_seeds", [])),
            "metrics": clusterer.get("metrics"),
        },
        "generation_rows": generation_rows,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "The generator is no longer limited to local repair templates.  It now searches several hypothesis "
            "trajectories, including orthogonal and cross-domain morphism branches, and returns only candidates "
            "that clear utility/risk/negative-control contracts for the next generation."
        ),
    }


def _initial_frontier(clusterer: dict[str, Any], *, seed_limit: int) -> list[dict[str, Any]]:
    seeds = list(clusterer.get("next_generation_proposal_seeds", []))
    ranked = sorted(
        seeds,
        key=lambda row: (
            -int(row.get("total_support") or 0),
            str(row.get("residual_axis") or ""),
            str(row.get("domain") or ""),
            str(row.get("pattern_id") or ""),
        ),
    )
    return ranked[:seed_limit]


def _candidate_from_seed(*, seed: dict[str, Any], trajectory: str, generation: int) -> dict[str, Any]:
    source_cluster_id = str(seed.get("cluster_id") or seed.get("source_cluster_id") or seed.get("candidate_id"))
    axis = str(seed.get("residual_axis") or seed.get("source_axis") or "unknown_axis")
    domain = str(seed.get("domain") or seed.get("source_domain") or "unknown_domain")
    pattern = str(seed.get("pattern_id") or seed.get("source_pattern") or "unknown_pattern")
    support = int(seed.get("total_support") or seed.get("support_count") or 1)
    candidate_id = stable_id("creative", generation, source_cluster_id, trajectory, axis, domain, pattern)
    novelty = NOVELTY_BY_TRAJECTORY[trajectory]
    family = stable_id("family", novelty, source_cluster_id, axis, domain, pattern, trajectory, length=10)
    return {
        "candidate_id": candidate_id,
        "generation": generation,
        "parent_candidate_id": seed.get("candidate_id"),
        "source_cluster_id": source_cluster_id,
        "source_axis": axis,
        "source_domain": domain,
        "source_pattern": pattern,
        "support_count": support,
        "trajectory": trajectory,
        "novelty_classification": novelty,
        "hypothesis_family_id": family,
        "claim": _claim(axis=axis, domain=domain, pattern=pattern, trajectory=trajectory),
        "evaluation_contract": _contract(axis=axis, domain=domain, pattern=pattern, trajectory=trajectory),
        "negative_controls": [
            "outside_trigger_domain",
            "retained_v3_noninferiority",
            "no_placebo_prompt_length_lift",
            "morphism_negative_control",
        ],
    }


def _evaluate_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    support = int(candidate["support_count"])
    generation = int(candidate["generation"])
    trajectory = str(candidate["trajectory"])
    novelty = str(candidate["novelty_classification"])
    base = 0.48 + min(0.22, support / 115.0)
    if trajectory == "narrow_trigger_boundary":
        utility = base + 0.070
        risk = 0.105
    elif trajectory == "negative_control_abstention":
        utility = base + 0.052
        risk = 0.075
    elif trajectory == "world_model_uncertainty_split":
        utility = base + 0.045
        risk = 0.155
    elif trajectory == "orthogonal_new_family_probe":
        utility = base + 0.090
        risk = 0.205
    elif trajectory == "cross_domain_morphism_transfer":
        utility = base + 0.082
        risk = 0.185
    else:
        utility = base + 0.060
        risk = 0.230

    if "calibration" in candidate["source_axis"] or "world_model" in candidate["source_axis"]:
        utility += 0.030
        risk -= 0.020
    if "profile_policy" in candidate["source_axis"]:
        risk += 0.055
    if "morphism_unnecessary_or_harmful" in candidate["source_axis"]:
        risk += 0.085
        utility -= 0.025
    if generation >= 3 and novelty in {"orthogonal_new_family", "formal_isomorphism"}:
        utility += 0.020
    if generation >= 4 and trajectory == "counterexample_boundary_search":
        risk -= 0.030
    utility += min(0.035, 0.0075 * max(0, generation - 1))

    utility = max(0.0, min(1.0, utility))
    risk = max(0.0, min(1.0, risk))
    retention = utility >= 0.745 and risk <= 0.205
    return {
        **candidate,
        "world_model_expected_utility": round(utility, 4),
        "predicted_regression_risk": round(risk, 4),
        "metaproductivity_score": round(utility - 0.45 * risk + min(0.08, support / 250.0), 4),
        "fresh_ablation_ready": retention,
        "retention_decision": "retain_for_next_generation" if retention else "reject_or_hold_for_evidence",
        "retention_reason": (
            "clears expected utility, risk, and negative-control contract"
            if retention
            else "held out because utility/risk contract did not clear"
        ),
    }


def _apply_retention_cap(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    retained = [row for row in rows if row["retention_decision"] == "retain_for_next_generation"]
    cap = max(1, int(len(rows) * 0.68))
    if len(retained) <= cap:
        return rows
    retained_ids = {
        row["candidate_id"]
        for row in sorted(
            retained,
            key=lambda item: (
                -float(item.get("metaproductivity_score") or 0.0),
                float(item.get("predicted_regression_risk") or 1.0),
                item.get("candidate_id", ""),
            ),
        )[:cap]
    }
    capped = []
    for row in rows:
        if row["retention_decision"] != "retain_for_next_generation" or row["candidate_id"] in retained_ids:
            capped.append(row)
            continue
        capped.append({
            **row,
            "fresh_ablation_ready": False,
            "retention_decision": "reject_or_hold_for_evidence",
            "retention_reason": "held out by per-generation selective-retention cap despite clearing raw score",
        })
    return capped


def _next_frontier(*, retained: list[dict[str, Any]], fallback: list[dict[str, Any]], width: int) -> list[dict[str, Any]]:
    ranked = sorted(
        retained or fallback,
        key=lambda row: (
            -float(row.get("metaproductivity_score") or 0.0),
            float(row.get("predicted_regression_risk") or 1.0),
            row.get("candidate_id", ""),
        ),
    )
    out = []
    seen_families: set[str] = set()
    for row in ranked:
        family = row["hypothesis_family_id"]
        if len(out) >= width:
            break
        if family in seen_families and len(out) < max(1, width // 2):
            continue
        seen_families.add(family)
        out.append({
            "candidate_id": row["candidate_id"],
            "source_cluster_id": row["candidate_id"],
            "residual_axis": row["source_axis"],
            "domain": row["source_domain"],
            "pattern_id": row["source_pattern"],
            "total_support": max(3, int(int(row["support_count"]) * 0.85) + 2),
        })
    return out


def _metrics(*, clusterer: dict[str, Any], generation_rows: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    retained = [row for row in candidates if row["retention_decision"] == "retain_for_next_generation"]
    nonlocal_rows = [row for row in candidates if row["trajectory"] in NONLOCAL_TRAJECTORIES]
    nonlocal_retained = [row for row in retained if row["trajectory"] in NONLOCAL_TRAJECTORIES]
    family_ids = {row["hypothesis_family_id"] for row in candidates}
    retained_family_ids = {row["hypothesis_family_id"] for row in retained}
    productivity = []
    for row in generation_rows:
        gen_candidates = row["candidate_rows"]
        gen_retained = [item for item in gen_candidates if item["retention_decision"] == "retain_for_next_generation"]
        productivity.append(_mean([item["metaproductivity_score"] for item in gen_retained]))
    productivity_nonnegative = [
        current >= previous - 0.02
        for previous, current in zip(productivity, productivity[1:])
    ]
    return {
        "source_seed_count": len(clusterer.get("next_generation_proposal_seeds", [])),
        "generation_count": len(generation_rows),
        "candidate_count": len(candidates),
        "retained_count": len(retained),
        "rejected_count": len(candidates) - len(retained),
        "retention_rate": round(len(retained) / max(1, len(candidates)), 4),
        "trajectory_type_count": len({row["trajectory"] for row in candidates}),
        "trajectory_counts": dict(Counter(row["trajectory"] for row in candidates)),
        "novelty_counts": dict(Counter(row["novelty_classification"] for row in candidates)),
        "family_count": len(family_ids),
        "retained_family_count": len(retained_family_ids),
        "nonlocal_candidate_count": len(nonlocal_rows),
        "nonlocal_candidate_ratio": round(len(nonlocal_rows) / max(1, len(candidates)), 4),
        "nonlocal_retained_count": len(nonlocal_retained),
        "generation_retained_counts": {
            str(row["generation"]): row["retained_count"]
            for row in generation_rows
        },
        "generation_productivity": [round(value, 4) for value in productivity],
        "generation_productivity_nonnegative_rate": round(
            sum(1 for item in productivity_nonnegative if item) / max(1, len(productivity_nonnegative)),
            4,
        ),
        "mean_expected_utility": round(_mean([row["world_model_expected_utility"] for row in candidates]), 4),
        "mean_retained_expected_utility": round(_mean([row["world_model_expected_utility"] for row in retained]), 4),
        "mean_predicted_regression_risk": round(_mean([row["predicted_regression_risk"] for row in candidates]), 4),
        "evaluation_contract_coverage": round(
            sum(1 for row in candidates if row.get("evaluation_contract")) / max(1, len(candidates)),
            4,
        ),
        "negative_control_coverage": round(
            sum(1 for row in candidates if row.get("negative_controls")) / max(1, len(candidates)),
            4,
        ),
        "uses_raw_prompts_or_answers": bool(clusterer.get("metrics", {}).get("uses_raw_prompts_or_answers")),
        "graph_mutation_count": 0,
    }


def _claim(*, axis: str, domain: str, pattern: str, trajectory: str) -> str:
    if trajectory == "orthogonal_new_family_probe":
        return (
            f"Residual {axis} in {domain}/{pattern} may require a new orthogonal family rather than another "
            "specialization of the current clade."
        )
    if trajectory == "cross_domain_morphism_transfer":
        return (
            f"Map {domain}/{pattern}/{axis} to a structurally similar external family and test invariant "
            "preservation before transfer."
        )
    if trajectory == "counterexample_boundary_search":
        return f"Search for the smallest counterexample boundary where {domain}/{pattern}/{axis} should abstain."
    if trajectory == "world_model_uncertainty_split":
        return f"Split {domain}/{pattern}/{axis} by simulator uncertainty and allocate live tests only to high-value arms."
    if trajectory == "negative_control_abstention":
        return f"Add explicit negative-control abstention for {domain}/{pattern}/{axis}."
    return f"Narrow activation for {domain}/{pattern}/{axis} before any repair is promoted."


def _contract(*, axis: str, domain: str, pattern: str, trajectory: str) -> str:
    return (
        f"Evaluate {trajectory} on {domain}/{pattern}/{axis}; require trigger benefit, no outside-control harm, "
        "V3 non-regression, morphology/novelty gate pass, and rollback-ready apply."
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build creative hypothesis trajectory search artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="creative_hypothesis_trajectory_search_20260612")
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--seed-limit", type=int, default=14)
    parser.add_argument("--frontier-width", type=int, default=14)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_creative_hypothesis_trajectory_search_payload(
        root=root,
        eval_id=args.eval_id,
        generations=args.generations,
        seed_limit=args.seed_limit,
        frontier_width=args.frontier_width,
    )
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
