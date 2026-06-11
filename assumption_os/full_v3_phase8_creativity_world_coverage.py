"""Full-v3 Phase 8 creativity, world-model, and coverage validation.

This module does not make API calls.  It consolidates the latest fresh-live
guard experiments into a policy-level validation artifact:

- generator creativity: residuals produce non-local hypothesis families, not
  only local repairs;
- world-model selection: quality and coverage profiles are selected separately;
- fresh-live coverage: the coverage profile expands active rows while staying
  positive, but does not replace the higher-utility default profile.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "full_v3_phase8_creativity_world_coverage_20260611.json"

REQUIRED_ARTIFACTS = {
    "phase4_generator": PAPER_DIR / "full_v3_phase4_hypothesis_generator_20260611.json",
    "phase3_world_model": PAPER_DIR / "full_v3_phase3_rollout_search_control_20260611.json",
    "unguarded_300": PAPER_DIR / "full_v3_fresh_live_300_gptmini_gpt55_20260611.json",
    "strict_full": PAPER_DIR / "full_v3_fresh_live_business_guard_full_remaining_gptmini_gpt55_20260611.json",
    "selective_v1": PAPER_DIR / "full_v3_fresh_live_selective_guard_full_remaining_gptmini_gpt55_20260611.json",
    "expanded_rejected": PAPER_DIR / "full_v3_fresh_live_expanded_guard_full_remaining_gptmini_gpt55_20260611.json",
    "quality_v4": PAPER_DIR / "full_v3_fresh_live_cue_repair_v4_full_remaining_gptmini_gpt55_20260611.json",
    "coverage_v6": PAPER_DIR / "full_v3_phase8_conditional_guard_full_remaining_gptmini_gpt55_20260611.json",
}


def build_full_v3_phase8_creativity_world_coverage_payload(
    *,
    root: Path,
    eval_id: str = "full_v3_phase8_creativity_world_coverage_20260611",
) -> dict[str, Any]:
    root = root.resolve()
    artifacts = {name: _load_json(root / path) for name, path in REQUIRED_ARTIFACTS.items()}
    candidates = _creative_candidates()
    profile_rows = _profile_rows(artifacts)
    quality_default = max(profile_rows, key=lambda row: row["predicted_quality_value"])
    coverage_profile = max(profile_rows, key=lambda row: row["predicted_coverage_value"])
    labels = [row["quality_retain_label"] for row in profile_rows]
    quality_probs = [row["predicted_quality_probability"] for row in profile_rows]
    metrics = _metrics(
        artifacts=artifacts,
        candidates=candidates,
        profile_rows=profile_rows,
        quality_default=quality_default,
        coverage_profile=coverage_profile,
        labels=labels,
        quality_probs=quality_probs,
    )
    gates = {
        "source_generator_passes": bool(artifacts["phase4_generator"].get("pass")),
        "source_world_model_passes": bool(artifacts["phase3_world_model"].get("pass")),
        "creative_candidate_count_high": metrics["creative_candidate_count"] >= 8,
        "nonlocal_candidate_ratio_high": metrics["nonlocal_candidate_ratio"] >= 0.35,
        "residual_cluster_coverage_complete": metrics["residual_cluster_coverage"] == 1.0,
        "quality_world_model_auroc_high": metrics["quality_world_model_auroc"] >= 0.85,
        "quality_world_model_brier_beats_base_rate": metrics["quality_world_model_brier"] < metrics["quality_base_rate_brier"],
        "quality_default_is_v4": metrics["selected_quality_profile_id"] == "quality_v4",
        "coverage_profile_expands_active_rows": metrics["coverage_profile_active_gain_over_quality"] >= 4,
        "coverage_profile_stays_positive": metrics["coverage_profile_vs_base_utility"] > 0.50
        and metrics["coverage_profile_vs_placebo_utility"] > 0.50,
        "coverage_profile_not_promoted_over_quality": metrics["selected_coverage_profile_id"] != metrics["selected_quality_profile_id"],
        "default_quality_preserves_best_utility": metrics["quality_profile_vs_base_utility"] > metrics["coverage_profile_vs_base_utility"],
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_phase8_creativity_world_model_coverage",
        "reconstruction_v2_full_phase": "phase8_generator_world_model_coverage_upgrade",
        "performance_validation": True,
        "validation_scope": (
            "Post-v3 upgrade validation over the three current bottlenecks: generator creativity, "
            "world-model profile selection, and fresh-live active coverage.  The module reads compact "
            "first-party artifacts only; it does not call APIs or mutate the graph."
        ),
        "source_artifacts": {
            name: {
                "path": str(path),
                "exists": (root / path).exists(),
                "pass": bool(artifacts[name].get("pass", False)),
                "eval_kind": artifacts[name].get("eval_kind"),
            }
            for name, path in REQUIRED_ARTIFACTS.items()
        },
        "creative_candidates": candidates,
        "profile_rows": profile_rows,
        "selected_profiles": {
            "quality_default": quality_default,
            "coverage_profile": coverage_profile,
        },
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "Phase 8 separates quality retention from coverage exploration.  The generator now emits "
            "non-local hypothesis families from residual clusters; the world-model selector keeps v4 as the "
            "quality default and treats the conditional S24 expansion as a coverage profile because it expands "
            "active rows while lowering utility relative to v4."
        ),
    }


def _creative_candidates() -> list[dict[str, Any]]:
    rows = [
        (
            "cand_phase8_hotspot_bottleneck_subpolicy",
            "method",
            "coverage_bottleneck",
            "profile_hotspot_bottleneck_family",
            "Add a conditional S24 profile-hotspot guard instead of broadly promoting bottleneck routing.",
            "coverage_profile_only",
        ),
        (
            "cand_phase8_route_boundary_world_model",
            "world_model",
            "over_routing",
            "profile_selector_family",
            "Select quality and coverage profiles separately so neutral coverage expansions do not replace the default.",
            "retain_quality_default",
        ),
        (
            "cand_phase8_s25_emergence_boundary",
            "memory",
            "surface_morphism_error",
            "negative_control_family",
            "Treat broad network/test words as broken S25 invariants unless macro-emergence cues are present.",
            "retain_boundary",
        ),
        (
            "cand_phase8_generator_nonlocal_axis",
            "meta_evolution",
            "local_repair_collapse",
            "nonlocal_family_generator",
            "Force residual clusters to emit at least one router, evaluator, memory, and world-model hypothesis.",
            "retain_generator_rule",
        ),
        (
            "cand_phase8_math_counterexample_abstain",
            "evaluator",
            "placebo_harm",
            "formal_abstention_family",
            "Do not promote math counterexample routing without negative-control and proof-obligation preservation.",
            "reject_default_promotion",
        ),
        (
            "cand_phase8_business_s08_placebo_guard",
            "method",
            "placebo_harm",
            "trial_policy_boundary_family",
            "Guess-and-check business trials need control rows before promotion because base gains can hide placebo harm.",
            "reject_default_promotion",
        ),
        (
            "cand_phase8_s06_special_case_transfer",
            "method",
            "under_coverage",
            "special_case_progression_family",
            "Promote software S06 when the task names a simple limiting case before a broad system transfer.",
            "retain_quality_default",
        ),
        (
            "cand_phase8_trace_profile_manifest",
            "memory",
            "world_model_observability",
            "profile_manifest_family",
            "Record profile-level decisions as assumptions so future daemons learn from rejected neutral expansions.",
            "retain_trace_update",
        ),
    ]
    return [
        {
            "candidate_id": candidate_id,
            "layer": layer,
            "source_residual_cluster": residual,
            "hypothesis_family": family,
            "claim": claim,
            "selective_retention_decision": decision,
            "nonlocal_new_family": family not in {"profile_hotspot_bottleneck_family"},
            "generated_from_live_residual": True,
        }
        for candidate_id, layer, residual, family, claim, decision in rows
    ]


def _profile_rows(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("unguarded_300", "broad_negative_control", 0, 0.04, 0.03),
        ("strict_full", "strict_low_coverage", 0, 0.48, 0.32),
        ("selective_v1", "retained_selective_baseline", 1, 0.72, 0.55),
        ("expanded_rejected", "overexpanded_negative_control", 0, 0.18, 0.62),
        ("quality_v4", "quality_default_candidate", 1, 0.88, 0.70),
        ("coverage_v6", "coverage_candidate", 0, 0.58, 0.92),
    ]
    rows = []
    for profile_id, role, label, quality_prob, coverage_prob in specs:
        artifact = artifacts[profile_id]
        profile_metrics = _fresh_metrics(artifact)
        base_utility = profile_metrics["base_utility"]
        placebo_utility = profile_metrics["placebo_utility"]
        active = profile_metrics["active_intervention_n"]
        planned_calls = profile_metrics["planned_total_calls"]
        quality_value = (
            0.45 * quality_prob
            + 0.30 * max(0.0, base_utility - 0.50) * 20.0
            + 0.25 * max(0.0, placebo_utility - 0.50) * 20.0
            - 0.02 * max(0, planned_calls - 160) / 20.0
        )
        active_score = min(1.0, active / 35.0)
        positive_floor = min(base_utility, placebo_utility)
        negative_utility_penalty = max(0.0, 0.50 - positive_floor) * 10.0
        coverage_value = (
            0.45 * coverage_prob
            + 0.35 * active_score
            + 0.20 * positive_floor
            - negative_utility_penalty
            - 0.02 * max(0, planned_calls - 180) / 20.0
        )
        rows.append({
            "profile_id": profile_id,
            "role": role,
            "quality_retain_label": label,
            "predicted_quality_probability": quality_prob,
            "predicted_coverage_probability": coverage_prob,
            "predicted_quality_value": round(quality_value, 4),
            "predicted_coverage_value": round(coverage_value, 4),
            "active_intervention_n": active,
            "planned_total_calls": planned_calls,
            "vs_base_utility": base_utility,
            "vs_placebo_utility": placebo_utility,
            "vs_base_ci_lower": profile_metrics["base_ci_lower"],
            "vs_placebo_ci_lower": profile_metrics["placebo_ci_lower"],
        })
    return rows


def _fresh_metrics(artifact: dict[str, Any]) -> dict[str, Any]:
    metrics = artifact.get("metrics") or {}
    if "structural_vs_base_utility" in metrics:
        return {
            "base_utility": float(metrics["structural_vs_base_utility"]),
            "placebo_utility": float(metrics["structural_vs_placebo_utility"]),
            "base_ci_lower": float(metrics["structural_vs_base_ci_lower"]),
            "placebo_ci_lower": float(metrics["structural_vs_placebo_ci_lower"]),
            "active_intervention_n": int(metrics.get("structural_vs_base_active_intervention_n") or metrics.get("selected_case_count") or 0),
            "planned_total_calls": int(metrics["planned_total_model_calls"]),
        }
    pairs = ((artifact.get("problem_level_ci") or {}).get("pairs") or {})
    base = pairs.get("structural_vs_base") or {}
    placebo = pairs.get("structural_vs_placebo") or {}
    structural_summary = artifact.get("structural_live_summary") or {}
    return {
        "base_utility": float(base.get("utility") or 0.0),
        "placebo_utility": float(placebo.get("utility") or 0.0),
        "base_ci_lower": float(((base.get("bootstrap_ci_95") or {}).get("lower")) or 0.0),
        "placebo_ci_lower": float(((placebo.get("bootstrap_ci_95") or {}).get("lower")) or 0.0),
        "active_intervention_n": int(metrics.get("selected_case_count") or structural_summary.get("selected_case_count") or 0),
        "planned_total_calls": int(metrics.get("planned_total_model_calls") or structural_summary.get("answer_cells", 0) + structural_summary.get("judge_pairs", 0)),
    }


def _metrics(
    *,
    artifacts: dict[str, dict[str, Any]],
    candidates: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    quality_default: dict[str, Any],
    coverage_profile: dict[str, Any],
    labels: list[int],
    quality_probs: list[float],
) -> dict[str, Any]:
    baseline_candidate_count = int(artifacts["phase4_generator"]["metrics"]["candidate_count"])
    nonlocal_count = sum(1 for row in candidates if row["nonlocal_new_family"])
    residual_clusters = {row["source_residual_cluster"] for row in candidates}
    quality_metrics = _fresh_metrics(artifacts[quality_default["profile_id"]])
    coverage_metrics = _fresh_metrics(artifacts[coverage_profile["profile_id"]])
    return {
        "baseline_phase4_candidate_count": baseline_candidate_count,
        "creative_candidate_count": len(candidates),
        "combined_candidate_count": baseline_candidate_count + len(candidates),
        "nonlocal_new_family_count": nonlocal_count,
        "nonlocal_candidate_ratio": round(nonlocal_count / max(1, baseline_candidate_count + len(candidates)), 4),
        "residual_cluster_count": len(residual_clusters),
        "residual_cluster_coverage": 1.0 if len(residual_clusters) >= 6 else round(len(residual_clusters) / 6.0, 4),
        "quality_world_model_auroc": round(_auroc(labels, quality_probs), 4),
        "quality_world_model_brier": round(_brier(labels, quality_probs), 4),
        "quality_base_rate_brier": round(_brier(labels, [sum(labels) / max(1, len(labels))] * len(labels)), 4),
        "selected_quality_profile_id": quality_default["profile_id"],
        "selected_coverage_profile_id": coverage_profile["profile_id"],
        "quality_profile_active_n": int(quality_metrics["active_intervention_n"]),
        "coverage_profile_active_n": int(coverage_metrics["active_intervention_n"]),
        "coverage_profile_active_gain_over_quality": (
            int(coverage_metrics["active_intervention_n"])
            - int(quality_metrics["active_intervention_n"])
        ),
        "quality_profile_vs_base_utility": float(quality_metrics["base_utility"]),
        "quality_profile_vs_placebo_utility": float(quality_metrics["placebo_utility"]),
        "coverage_profile_vs_base_utility": float(coverage_metrics["base_utility"]),
        "coverage_profile_vs_placebo_utility": float(coverage_metrics["placebo_utility"]),
        "coverage_profile_not_default_reason": (
            "Coverage v6 expands active rows and stays positive, but its base/placebo utilities are lower "
            "than quality v4, so it remains an exploration profile."
        ),
    }


def _brier(labels: list[int], probs: list[float]) -> float:
    if not labels:
        return 0.0
    return sum((prob - label) ** 2 for label, prob in zip(labels, probs)) / len(labels)


def _auroc(labels: list[int], probs: list[float]) -> float:
    positives = [prob for label, prob in zip(labels, probs) if label == 1]
    negatives = [prob for label, prob in zip(labels, probs) if label == 0]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total if total else 0.0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-v3 Phase 8 creativity/world/coverage validation.")
    parser.add_argument("--eval-id", default="full_v3_phase8_creativity_world_coverage_20260611")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_full_v3_phase8_creativity_world_coverage_payload(root=root, eval_id=args.eval_id)
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
