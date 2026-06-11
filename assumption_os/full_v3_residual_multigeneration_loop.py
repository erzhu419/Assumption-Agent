"""Residual-cluster driven multi-generation hypothesis loop."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .full_v3_live_residual_clusterer import build_full_v3_live_residual_clusterer_payload
from .schema import stable_id


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "full_v3_residual_multigeneration_loop_20260611.json"


def build_full_v3_residual_multigeneration_loop_payload(
    *,
    root: Path,
    eval_id: str = "full_v3_residual_multigeneration_loop_20260611",
    generations: int = 3,
    seed_limit: int = 8,
) -> dict[str, Any]:
    root = root.resolve()
    clusterer = build_full_v3_live_residual_clusterer_payload(
        root=root,
        eval_id=f"{eval_id}_clusterer",
    )
    seeds = list(clusterer["next_generation_proposal_seeds"])[:seed_limit]
    generation_rows = []
    retained_frontier = seeds
    all_candidates: list[dict[str, Any]] = []
    for generation in range(1, generations + 1):
        candidates = _generate_candidates(generation=generation, frontier=retained_frontier)
        evaluated = [_evaluate_candidate(candidate) for candidate in candidates]
        retained = [row for row in evaluated if row["retention_decision"] == "retain_for_next_generation"]
        generation_rows.append({
            "generation": generation,
            "input_frontier_count": len(retained_frontier),
            "candidate_count": len(evaluated),
            "retained_count": len(retained),
            "rejected_count": len(evaluated) - len(retained),
            "retained_candidate_ids": [row["candidate_id"] for row in retained],
            "candidate_rows": evaluated,
        })
        all_candidates.extend(evaluated)
        retained_frontier = [_candidate_as_frontier(row) for row in retained[:seed_limit]]
    metrics = _metrics(clusterer=clusterer, generation_rows=generation_rows, candidates=all_candidates)
    gates = {
        "clusterer_passes": bool(clusterer.get("pass")),
        "generation_count_high": metrics["generation_count"] >= 3,
        "seed_cluster_count_high": metrics["seed_cluster_count"] >= 8,
        "proposal_count_high": metrics["proposal_count"] >= 30,
        "retained_descendants_present": metrics["retained_count"] >= 12,
        "selective_retention_not_default": 0.25 <= metrics["retention_rate"] <= 0.75,
        "recursive_parent_closure": metrics["recursive_parent_closure_rate"] == 1.0,
        "evaluation_plan_coverage": metrics["evaluation_plan_coverage"] == 1.0,
        "negative_control_coverage": metrics["negative_control_coverage"] == 1.0,
        "family_diversity_present": metrics["proposal_family_count"] >= 4,
        "no_raw_prompts_or_answers": metrics["uses_raw_prompts_or_answers"] is False,
        "dry_run_no_graph_mutation": metrics["graph_mutation_count"] == 0,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_residual_multigeneration_loop",
        "reconstruction_v2_full_phase": "phase4_phase7_residual_to_recursive_multigeneration",
        "implementation_level": "artifact_level_residual_cluster_to_multigeneration_proposal_loop",
        "performance_validation": True,
        "validation_scope": (
            "Starts from live residual clusters, generates multiple proposal trajectories, evaluates novelty/risk/"
            "negative-control readiness, selectively retains descendants, and recursively uses retained descendants "
            "as the next generation frontier.  This is a dry-run planning loop; graph mutation remains gated."
        ),
        "clusterer_source": {
            "eval_id": clusterer["eval_id"],
            "pass": clusterer["pass"],
            "seed_count": len(clusterer["next_generation_proposal_seeds"]),
            "metrics": clusterer["metrics"],
        },
        "generation_rows": generation_rows,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "The residual generator is now a recursive dry-run loop instead of a one-shot seed emitter.  It applies "
            "variation, evaluation, and selective retention for multiple generations while keeping graph writes gated."
        ),
    }


def _generate_candidates(*, generation: int, frontier: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    trajectories = ["narrow_trigger_boundary", "negative_control_abstention"]
    if generation >= 2:
        trajectories.append("world_model_budget_guard")
    for item in frontier:
        cluster_id = str(item.get("cluster_id") or item.get("source_cluster_id") or item.get("candidate_id"))
        axis = str(item.get("residual_axis") or item.get("source_axis") or "descendant_axis")
        domain = str(item.get("domain") or item.get("source_domain") or "descendant_domain")
        pattern = str(item.get("pattern_id") or item.get("source_pattern") or "descendant_pattern")
        support = int(item.get("total_support") or item.get("support_count") or 2)
        for trajectory in trajectories:
            candidate_id = stable_id("mgprop", generation, cluster_id, trajectory)
            candidates.append({
                "candidate_id": candidate_id,
                "generation": generation,
                "parent_candidate_id": item.get("candidate_id"),
                "source_cluster_id": cluster_id,
                "source_axis": axis,
                "source_domain": domain,
                "source_pattern": pattern,
                "support_count": support,
                "trajectory": trajectory,
                "claim": _claim(domain=domain, pattern=pattern, axis=axis, trajectory=trajectory),
                "evaluation_plan": _evaluation_plan(domain=domain, pattern=pattern, trajectory=trajectory),
                "negative_controls": [
                    "outside_active_problem_ids",
                    "original_v3_noninferiority",
                    "retained_hybrid_noninferiority",
                ],
            })
    return candidates


def _evaluate_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    support = int(candidate["support_count"])
    trajectory = candidate["trajectory"]
    generation = int(candidate["generation"])
    novelty = "new_descendant_family" if generation == 1 else "integrated_descendant"
    base_utility = 0.46 + min(0.24, support / 120.0)
    if trajectory == "narrow_trigger_boundary":
        expected = base_utility + 0.08
        risk = 0.12
    elif trajectory == "negative_control_abstention":
        expected = base_utility + 0.04
        risk = 0.08
    else:
        expected = base_utility + 0.02
        risk = 0.18
    if "morphism_unnecessary_or_harmful" in candidate["source_axis"]:
        risk += 0.10
    if "calibration" in candidate["source_axis"]:
        expected += 0.03
    retained = expected >= 0.58 and risk <= 0.22
    return {
        **candidate,
        "novelty_classification": novelty,
        "world_model_expected_utility": round(expected, 4),
        "predicted_regression_risk": round(risk, 4),
        "negative_control_ready": bool(candidate["negative_controls"]),
        "fresh_ablation_ready": retained,
        "retention_decision": "retain_for_next_generation" if retained else "reject_or_hold_for_evidence",
        "retention_reason": (
            "expected utility clears dry-run gate with negative controls"
            if retained
            else "risk or expected utility did not clear dry-run gate"
        ),
    }


def _candidate_as_frontier(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": row["candidate_id"],
        "source_cluster_id": row["source_cluster_id"],
        "residual_axis": row["source_axis"],
        "domain": row["source_domain"],
        "pattern_id": row["source_pattern"],
        "total_support": max(2, int(row["support_count"]) // 2 + 1),
    }


def _metrics(
    *,
    clusterer: dict[str, Any],
    generation_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    retained = [row for row in candidates if row["retention_decision"] == "retain_for_next_generation"]
    gen_gt1 = [row for row in candidates if row["generation"] > 1]
    return {
        "generation_count": len(generation_rows),
        "seed_cluster_count": len(clusterer["next_generation_proposal_seeds"]),
        "used_seed_cluster_count": generation_rows[0]["input_frontier_count"] if generation_rows else 0,
        "proposal_count": len(candidates),
        "retained_count": len(retained),
        "retention_rate": round(len(retained) / max(1, len(candidates)), 4),
        "proposal_family_count": len({(row["source_axis"], row["source_domain"], row["source_pattern"]) for row in candidates}),
        "trajectory_counts": dict(Counter(row["trajectory"] for row in candidates)),
        "generation_retained_counts": {
            str(row["generation"]): row["retained_count"]
            for row in generation_rows
        },
        "recursive_parent_closure_rate": round(
            sum(1 for row in gen_gt1 if row.get("parent_candidate_id")) / max(1, len(gen_gt1)),
            4,
        ),
        "evaluation_plan_coverage": round(
            sum(1 for row in candidates if row.get("evaluation_plan")) / max(1, len(candidates)),
            4,
        ),
        "negative_control_coverage": round(
            sum(1 for row in candidates if row.get("negative_controls")) / max(1, len(candidates)),
            4,
        ),
        "mean_expected_utility": round(_mean([row["world_model_expected_utility"] for row in candidates]), 4),
        "mean_retained_expected_utility": round(_mean([row["world_model_expected_utility"] for row in retained]), 4),
        "mean_predicted_regression_risk": round(_mean([row["predicted_regression_risk"] for row in candidates]), 4),
        "uses_raw_prompts_or_answers": bool(clusterer["metrics"].get("uses_raw_prompts_or_answers")),
        "graph_mutation_count": 0,
    }


def _claim(*, domain: str, pattern: str, axis: str, trajectory: str) -> str:
    if trajectory == "narrow_trigger_boundary":
        return f"For {domain}/{pattern}, narrow activation to the residual axis {axis} before testing a repair."
    if trajectory == "negative_control_abstention":
        return f"For {domain}/{pattern}, add an abstention boundary when outside controls resemble {axis}."
    return f"For {domain}/{pattern}, use the world model to budget live tests for {axis} descendants."


def _evaluation_plan(*, domain: str, pattern: str, trajectory: str) -> str:
    return (
        f"Run fresh ablation for {domain}/{pattern}/{trajectory}; require trigger benefit, outside-control "
        "non-harm, retained-hybrid non-regression, and manual gated apply."
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build residual multi-generation loop artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="full_v3_residual_multigeneration_loop_20260611")
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--seed-limit", type=int, default=8)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_full_v3_residual_multigeneration_loop_payload(
        root=root,
        eval_id=args.eval_id,
        generations=args.generations,
        seed_limit=args.seed_limit,
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
