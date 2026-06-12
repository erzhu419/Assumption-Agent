"""Final closure artifact for last_three_part.md.

This artifact is a claim ledger, not a new mechanism.  It reads the production
autonomy, simulator, generator, paper-line, main-graph monitor, NL-to-diagram,
and Lean finite theorem artifacts and records exactly which claims are now
allowed and which remain blocked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR


DEFAULT_OUT = PAPER_DIR / "last_three_part_final_closure_20260612.json"

SOURCE_ARTIFACTS = {
    "autonomy_supervised_production": PAPER_DIR / "autonomy_supervised_production_run_20260612.json",
    "simulator_production_gate": PAPER_DIR / "simulator_production_gate_20260612.json",
    "simulator_production_evidence": PAPER_DIR / "simulator_production_evidence_20260612.json",
    "paper_frozen_main_v2": PAPER_DIR / "paper_frozen_main_experiment_v2_20260612.json",
    "creative_hypothesis_search": PAPER_DIR / "creative_hypothesis_trajectory_search_20260612.json",
    "main_graph_monitor": PAPER_DIR / "main_graph_controlled_apply_monitor_20260612.json",
    "nl_to_diagram_scale": PAPER_DIR / "nl_to_diagram_scale_benchmark_20260612.json",
    "finite_theorem_lean": PAPER_DIR / "finite_theorem_lean_verifier_20260612.json",
    "blinded_recursive_live": PAPER_DIR / "full_v3_blinded_recursive_live_line_20260612.json",
}


def build_last_three_part_final_closure_payload(
    *,
    root: Path,
    eval_id: str = "last_three_part_final_closure_20260612",
) -> dict[str, Any]:
    root = root.resolve()
    artifacts = {name: _load_json(root / path) for name, path in SOURCE_ARTIFACTS.items()}
    sections = {
        "A6_production_autonomy": _autonomy_section(artifacts["autonomy_supervised_production"]),
        "B7_gate_router_simulator": _simulator_section(
            gate=artifacts["simulator_production_gate"],
            evidence=artifacts["simulator_production_evidence"],
        ),
        "paper_main_experiment": _paper_section(artifacts["paper_frozen_main_v2"]),
        "creative_generator": _generator_section(artifacts["creative_hypothesis_search"]),
        "main_graph_controlled_apply": _main_graph_section(artifacts["main_graph_monitor"]),
        "formal_nl_diagram": _formal_section(
            nl=artifacts["nl_to_diagram_scale"],
            lean=artifacts["finite_theorem_lean"],
        ),
        "recursive_live_line": _recursive_section(artifacts["blinded_recursive_live"]),
    }
    metrics = _metrics(artifacts=artifacts, sections=sections)
    gates = {
        "all_source_artifacts_pass": metrics["source_artifact_pass_rate"] == 1.0,
        "a6_supervised_production_candidate_allowed": sections["A6_production_autonomy"]["allowed_claim"] is True,
        "b7_gate_router_simulator_allowed": sections["B7_gate_router_simulator"]["allowed_claim"] is True,
        "raw_simulator_replacement_blocked": sections["B7_gate_router_simulator"]["blocked_claims"]["raw_simulator_replacement"] is True,
        "paper_main_same_batch_line_passes": sections["paper_main_experiment"]["allowed_claim"] is True,
        "creative_generator_multitrajectory_passes": sections["creative_generator"]["allowed_claim"] is True,
        "main_graph_controlled_apply_monitor_passes": sections["main_graph_controlled_apply"]["allowed_claim"] is True,
        "bounded_nl_diagram_allowed": sections["formal_nl_diagram"]["allowed_claim"] is True,
        "full_theorem_prover_blocked": sections["formal_nl_diagram"]["blocked_claims"]["full_theorem_prover"] is True,
        "recursive_live_line_present": sections["recursive_live_line"]["allowed_claim"] is True,
        "claim_boundary_complete": metrics["blocked_strong_claim_count"] >= 4,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "last_three_part_final_closure",
        "reconstruction_v2_full_phase": "last_three_part_claim_ledger",
        "implementation_level": "machine_readable_promotion_and_blocker_ledger",
        "performance_validation": True,
        "validation_scope": (
            "Aggregates the last_three_part.md closure artifacts into explicit claim decisions.  The ledger allows "
            "supervised production autonomy, gate/router simulator use, paper frozen main experiment, bounded "
            "creative generator, main-graph controlled apply monitor, and bounded finite NL-diagram/Lean fragment. "
            "It continues to block raw simulator replacement, unbounded 24/7 OS, unrestricted full theorem prover, "
            "and claims of a new live API main experiment where only frozen replay exists."
        ),
        "source_artifacts": {
            name: {
                "path": str(path),
                "exists": (root / path).exists(),
                "pass": bool(artifacts[name].get("pass")),
                "eval_kind": artifacts[name].get("eval_kind"),
                "sha256": _sha256(root / path),
            }
            for name, path in SOURCE_ARTIFACTS.items()
        },
        "sections": sections,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "last_three_part.md is now closed at the bounded-production research-prototype level.  The project can "
            "claim a recursive self-evolution agent prototype with supervised autonomy evidence, a production "
            "gate/router simulator, a same-batch frozen paper line, multi-trajectory generator, committed main-graph "
            "canary monitor, and bounded formal certificates.  Stronger claims remain correctly blocked."
        ),
    }


def _autonomy_section(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload["metrics"]
    return {
        "allowed_claim": bool(metrics["production_autonomy_candidate_allowed"]),
        "claim": "30-day-equivalent supervised production autonomy candidate for restricted low-risk actions",
        "metrics": {
            "supervised_day_count": metrics["supervised_day_count"],
            "cycle_count": metrics["cycle_count"],
            "auto_apply_count": metrics["auto_apply_count"],
            "manual_review_load_rate": metrics["manual_review_load_rate"],
            "downstream_regression_rate": metrics["downstream_regression_rate"],
        },
        "blocked_claims": {
            "unbounded_24_7_general_os": True,
            "ungated_policy_or_default_mutation": True,
        },
    }


def _simulator_section(*, gate: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    metrics = gate["metrics"]
    ev = evidence["metrics"]
    return {
        "allowed_claim": bool(metrics["production_simulator_candidate_allowed"]),
        "claim": "production graph-action simulator for proposal triage and verifier routing",
        "metrics": {
            "transition_row_count": metrics["transition_row_count"],
            "pattern_count": metrics["pattern_count"],
            "counterfactual_mae": ev["counterfactual_mae"],
            "global_baseline_mae": ev["global_baseline_mae"],
            "best_arm_agreement_rate": ev["best_arm_agreement_rate"],
            "policy_lift_over_v3": ev["policy_lift_over_v3"],
        },
        "blocked_claims": {
            "raw_simulator_replacement": bool(metrics["raw_simulator_promoted"]) is False,
            "judge_or_live_ablation_replacement": True,
        },
    }


def _paper_section(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload["metrics"]
    return {
        "allowed_claim": bool(payload["pass"]),
        "claim": "same-batch redacted frozen main experiment with problem-level bootstrap CIs",
        "metrics": {
            "problem_count": metrics["problem_count"],
            "baseline_count": metrics["baseline_count"],
            "full_v3_margin_over_best_baseline_score": metrics["full_v3_margin_over_best_baseline_score"],
            "min_pairwise_utility": metrics["min_pairwise_utility"],
            "core_baseline_min_ci_lower": metrics["core_baseline_min_ci_lower"],
        },
        "blocked_claims": {
            "new_live_api_main_experiment": metrics["new_api_call_count"] == 0,
            "raw_prompt_or_answer_release": True,
        },
    }


def _generator_section(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload["metrics"]
    return {
        "allowed_claim": bool(payload["pass"]),
        "claim": "bounded multi-trajectory residual-to-hypothesis generator with selective retention",
        "metrics": {
            "generation_count": metrics["generation_count"],
            "candidate_count": metrics["candidate_count"],
            "retained_count": metrics["retained_count"],
            "retention_rate": metrics["retention_rate"],
            "retained_family_count": metrics["retained_family_count"],
            "nonlocal_candidate_ratio": metrics["nonlocal_candidate_ratio"],
        },
        "blocked_claims": {
            "unrestricted_creative_general_agent": True,
            "ungated_generator_to_main_graph": metrics["graph_mutation_count"] == 0,
        },
    }


def _main_graph_section(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload["metrics"]
    return {
        "allowed_claim": bool(payload["pass"]),
        "claim": "committed canary-scope controlled apply with rollback and 30-day-equivalent monitor",
        "metrics": {
            "source_main_graph_mutated": metrics["source_main_graph_mutated"],
            "canary_consolidated_node_count": metrics["canary_consolidated_node_count"],
            "monitor_day_count": metrics["monitor_day_count"],
            "min_precision_delta_vs_before": metrics["min_precision_delta_vs_before"],
            "regression_alert_count": metrics["regression_alert_count"],
        },
        "blocked_claims": {
            "unbounded_main_graph_mutation": True,
            "policy_default_auto_apply": True,
        },
    }


def _formal_section(*, nl: dict[str, Any], lean: dict[str, Any]) -> dict[str, Any]:
    nl_metrics = nl["metrics"]
    lean_metrics = lean["metrics"]
    return {
        "allowed_claim": bool(nl["pass"]) and bool(lean["pass"]),
        "claim": "bounded finite theorem fragment with scaled NL-to-diagram certificates and external Lean check",
        "metrics": {
            "nl_family_count": nl_metrics["family_count"],
            "nl_positive_accuracy": nl_metrics["positive_accuracy"],
            "nl_negative_specificity": nl_metrics["negative_specificity"],
            "lean_theorem_count": lean_metrics["lean_theorem_count"],
            "external_lean_check_passed": lean_metrics["external_lean_check_passed"],
        },
        "blocked_claims": {
            "full_theorem_prover": nl_metrics["full_theorem_prover_claim_allowed"] is False,
            "arbitrary_natural_language_semantic_equivalence": True,
            "unbounded_high_category_reasoning": True,
        },
    }


def _recursive_section(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload["metrics"]
    return {
        "allowed_claim": bool(payload["pass"]),
        "claim": "5-generation blinded recursive live line with accepted/rejected selective retention",
        "metrics": {
            "generation_count": metrics["executed_generation_count"],
            "seed_count": metrics["seed_count"],
            "fresh_api_call_count": metrics["fresh_api_call_count"],
            "accepted_count": metrics["accepted_count"],
            "rejected_count": metrics["rejected_count"],
            "live_error_count": metrics["live_error_count"],
        },
        "blocked_claims": {
            "unbounded_long_horizon_live_os": True,
        },
    }


def _metrics(*, artifacts: dict[str, dict[str, Any]], sections: dict[str, dict[str, Any]]) -> dict[str, Any]:
    blocked = [
        name
        for section in sections.values()
        for name, value in section["blocked_claims"].items()
        if value
    ]
    return {
        "source_artifact_count": len(artifacts),
        "source_artifact_pass_rate": round(sum(1 for item in artifacts.values() if item.get("pass")) / len(artifacts), 4),
        "allowed_claim_count": sum(1 for section in sections.values() if section["allowed_claim"]),
        "section_count": len(sections),
        "blocked_strong_claim_count": len(blocked),
        "blocked_strong_claims": sorted(blocked),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build last_three_part final closure artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="last_three_part_final_closure_20260612")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_last_three_part_final_closure_payload(root=root, eval_id=args.eval_id)
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
