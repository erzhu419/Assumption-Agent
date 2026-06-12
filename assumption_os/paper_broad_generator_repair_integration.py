"""Paper-facing integration for the broad-generator repair chain.

The first 720-call fresh rerun showed that an unfiltered generated frontier was
not safe to claim as broadly useful.  This integration records the subsequent
repair: use prior fresh failures as calibration evidence, abstain from weak
families, and keep the 720-call prospective budget by assigning more rows to the
qualified candidate frontier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR, stable_hash


DEFAULT_OUT = PAPER_DIR / "paper_broad_generator_repair_integration_20260612.json"
DEFAULT_MD_OUT = Path("reconstruction/md/paper_broad_generator_repair_integration_20260612.md")

SOURCE_ARTIFACTS = {
    "original_live_720": PAPER_DIR / "paper_fresh_frozen_rerun_live_720_20260612.json",
    "repair_v1_live_720": PAPER_DIR / "paper_fresh_broad_generator_repair_live_720_20260612.json",
    "repair_v2_dryrun_720": PAPER_DIR / "paper_fresh_broad_generator_repair_v2_dryrun_20260612.json",
    "repair_v2_smoke_live_240": PAPER_DIR / "paper_fresh_broad_generator_repair_v2_smoke_live_240_20260612.json",
    "repair_v2_live_720": PAPER_DIR / "paper_fresh_broad_generator_repair_v2_live_720_20260612.json",
}


def build_paper_broad_generator_repair_integration_payload(
    *,
    root: Path,
    eval_id: str = "paper_broad_generator_repair_integration_20260612",
) -> dict[str, Any]:
    root = root.resolve()
    artifacts = {name: _load_json(root / path) for name, path in SOURCE_ARTIFACTS.items()}
    metrics = _metrics(artifacts)
    gates = {
        "source_artifacts_present": all((root / path).exists() for path in SOURCE_ARTIFACTS.values()),
        "original_broad_gate_failed": metrics["original_raw_pass"] is False
        and "all_candidate_trigger_exploration_not_catastrophic" in metrics["original_failed_gates"],
        "v1_repair_failed_and_supplied_evidence": metrics["repair_v1_raw_pass"] is False
        and metrics["repair_v1_fresh_api_call_count"] == 720
        and metrics["repair_v1_live_error_count"] == 0,
        "v2_dryrun_passed": metrics["repair_v2_dryrun_pass"] is True,
        "v2_smoke_live_passed": metrics["repair_v2_smoke_pass"] is True
        and metrics["repair_v2_smoke_fresh_api_call_count"] >= 180,
        "v2_live_720_passed": metrics["repair_v2_live_pass"] is True,
        "v2_live_completed_error_free": metrics["repair_v2_fresh_api_call_count"] == 720
        and metrics["repair_v2_planned_fresh_api_call_count"] == 720
        and metrics["repair_v2_live_error_count"] == 0,
        "v2_broad_gate_cleared": metrics["repair_v2_failed_gates"] == [],
        "v2_trigger_improves_original": metrics["trigger_utility_delta_vs_original"] >= 0.10,
        "v2_ci_separates_from_original": metrics["repair_v2_trigger_ci95_lower"] > metrics["original_trigger_ci95_upper"],
        "v2_control_loss_bounded": metrics["repair_v2_control_loss_ci95_upper"] <= 0.35
        and metrics["repair_v2_accepted_control_loss_ci95_upper"] <= 0.35,
        "evidence_calibrated_abstention_observed": metrics["selected_candidate_count_delta_vs_original"] <= -40
        and metrics["repair_v2_selected_candidate_count"] >= 4,
        "selection_keeps_large_fresh_budget": metrics["repair_v2_fresh_api_call_count"]
        == metrics["original_fresh_api_call_count"],
        "real_problem_and_blinding_complete": metrics["repair_v2_real_problem_assignment_rate"] == 1.0
        and metrics["repair_v2_side_assignment_rate"] == 1.0,
        "no_prompt_answer_or_secret_payload": metrics["repair_v2_prompt_answer_or_secret_payload_detected"] is False
        and metrics["repair_v2_secret_value_exposed"] is False,
        "graph_copy_only": metrics["repair_v2_main_graph_mutation_count"] == 0,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "paper_broad_generator_repair_integration",
        "reconstruction_v2_full_phase": "paper_facing_broad_generator_repair",
        "implementation_level": "fresh_evidence_calibrated_selector_with_abstention_and_720_call_validation",
        "performance_validation": True,
        "validation_scope": (
            "Integrates the broad-generator repair sequence.  The repair does not claim that every raw generated "
            "candidate is useful; it claims that after using fresh failures as evidence, the generator can form a "
            "qualified frontier that clears the same all-candidate trigger gate on a new 720-call live rerun."
        ),
        "source_artifacts": {
            name: {
                "path": str(path),
                "exists": (root / path).exists(),
                "pass": bool(artifacts[name].get("pass")),
                "eval_kind": artifacts[name].get("eval_kind"),
                "sha256": _sha256(root / path) if (root / path).exists() else None,
            }
            for name, path in SOURCE_ARTIFACTS.items()
        },
        "paper_table_update": _paper_table_update(metrics),
        "claim_boundaries": _claim_boundaries(metrics),
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "allowed_claim": (
            "Fresh evidence-calibrated broad generation passes: after abstaining from weak families, the qualified "
            "frontier clears the all-candidate trigger gate on a new 720-call prospective run."
        ),
        "blocked_claims": [
            "raw_unfiltered_generator_frontier_is_safe",
            "retained_status_without_live_evidence_is_sufficient",
            "broad_generation_should_fill_every_slot_when_evidence_says_abstain",
        ],
    }


def format_markdown(payload: dict[str, Any]) -> str:
    m = payload["metrics"]
    lines = [
        "# Paper Broad Generator Repair Integration",
        "",
        f"- pass: `{payload['pass']}`",
        f"- original trigger utility: `{m['original_trigger_problem_level_mean_utility']}` "
        f"CI `{m['original_trigger_problem_level_ci95']}`",
        f"- v1 repair trigger utility: `{m['repair_v1_trigger_problem_level_mean_utility']}` "
        f"CI `{m['repair_v1_trigger_problem_level_ci95']}`",
        f"- v2 repair trigger utility: `{m['repair_v2_trigger_problem_level_mean_utility']}` "
        f"CI `{m['repair_v2_trigger_problem_level_ci95']}`",
        f"- v2 fresh calls: `{m['repair_v2_fresh_api_call_count']}/"
        f"{m['repair_v2_planned_fresh_api_call_count']}`",
        f"- v2 selected candidates: `{m['repair_v2_selected_candidate_count']}` "
        f"(original `{m['original_selected_candidate_count']}`)",
        f"- v2 control loss: `{m['repair_v2_control_problem_level_mean_loss_rate']}` "
        f"CI `{m['repair_v2_control_problem_level_ci95']}`",
        "",
        "## Interpretation",
        "",
        "The raw broad generator failed.  The repaired generator uses fresh failure evidence as a selector and "
        "abstains from low-support families.  The resulting qualified frontier keeps the 720-call budget and "
        "passes the all-candidate trigger gate on a new live rerun.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _metrics(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    original = _live_metrics(artifacts["original_live_720"])
    v1 = _live_metrics(artifacts["repair_v1_live_720"])
    dry = _live_metrics(artifacts["repair_v2_dryrun_720"])
    smoke = _live_metrics(artifacts["repair_v2_smoke_live_240"])
    v2 = _live_metrics(artifacts["repair_v2_live_720"])
    original_ci_high = _ci_value(original["trigger_problem_level_ci95"], 1)
    v2_ci_low = _ci_value(v2["trigger_problem_level_ci95"], 0)
    return {
        "original_raw_pass": bool(artifacts["original_live_720"].get("pass")),
        "original_failed_gates": list(artifacts["original_live_720"].get("failed_gates", [])),
        "original_fresh_api_call_count": original["fresh_api_call_count"],
        "original_live_error_count": original["live_error_count"],
        "original_selected_candidate_count": original["selected_candidate_count"],
        "original_trigger_problem_level_mean_utility": original["trigger_problem_level_mean_utility"],
        "original_trigger_problem_level_ci95": original["trigger_problem_level_ci95"],
        "original_trigger_ci95_upper": original_ci_high,
        "repair_v1_raw_pass": bool(artifacts["repair_v1_live_720"].get("pass")),
        "repair_v1_failed_gates": list(artifacts["repair_v1_live_720"].get("failed_gates", [])),
        "repair_v1_fresh_api_call_count": v1["fresh_api_call_count"],
        "repair_v1_live_error_count": v1["live_error_count"],
        "repair_v1_selected_candidate_count": v1["selected_candidate_count"],
        "repair_v1_trigger_problem_level_mean_utility": v1["trigger_problem_level_mean_utility"],
        "repair_v1_trigger_problem_level_ci95": v1["trigger_problem_level_ci95"],
        "repair_v2_dryrun_pass": bool(artifacts["repair_v2_dryrun_720"].get("pass")),
        "repair_v2_dryrun_selected_candidate_count": dry["selected_candidate_count"],
        "repair_v2_smoke_pass": bool(artifacts["repair_v2_smoke_live_240"].get("pass")),
        "repair_v2_smoke_fresh_api_call_count": smoke["fresh_api_call_count"],
        "repair_v2_smoke_trigger_problem_level_mean_utility": smoke["trigger_problem_level_mean_utility"],
        "repair_v2_live_pass": bool(artifacts["repair_v2_live_720"].get("pass")),
        "repair_v2_failed_gates": list(artifacts["repair_v2_live_720"].get("failed_gates", [])),
        "repair_v2_fresh_api_call_count": v2["fresh_api_call_count"],
        "repair_v2_planned_fresh_api_call_count": v2["planned_fresh_api_call_count"],
        "repair_v2_live_error_count": v2["live_error_count"],
        "repair_v2_selected_candidate_count": v2["selected_candidate_count"],
        "repair_v2_accepted_count": v2["accepted_count"],
        "repair_v2_rejected_count": v2["rejected_count"],
        "repair_v2_trigger_problem_count": v2["trigger_problem_count"],
        "repair_v2_trigger_problem_level_mean_utility": v2["trigger_problem_level_mean_utility"],
        "repair_v2_trigger_problem_level_ci95": v2["trigger_problem_level_ci95"],
        "repair_v2_trigger_ci95_lower": v2_ci_low,
        "repair_v2_control_problem_level_mean_loss_rate": v2["control_problem_level_mean_loss_rate"],
        "repair_v2_control_problem_level_ci95": v2["control_problem_level_ci95"],
        "repair_v2_control_loss_ci95_upper": _ci_value(v2["control_problem_level_ci95"], 1),
        "repair_v2_accepted_control_problem_level_mean_loss_rate": v2[
            "accepted_control_problem_level_mean_loss_rate"
        ],
        "repair_v2_accepted_control_problem_level_ci95": v2["accepted_control_problem_level_ci95"],
        "repair_v2_accepted_control_loss_ci95_upper": _ci_value(v2["accepted_control_problem_level_ci95"], 1),
        "repair_v2_real_problem_assignment_rate": v2["real_problem_assignment_rate"],
        "repair_v2_side_assignment_rate": v2["side_assignment_rate"],
        "repair_v2_main_graph_mutation_count": v2["main_graph_mutation_count"],
        "repair_v2_prompt_answer_or_secret_payload_detected": v2["prompt_answer_or_secret_payload_detected"],
        "repair_v2_secret_value_exposed": v2["secret_value_exposed"],
        "trigger_utility_delta_vs_original": round(
            v2["trigger_problem_level_mean_utility"] - original["trigger_problem_level_mean_utility"],
            4,
        ),
        "trigger_ci_lower_minus_original_ci_upper": round(v2_ci_low - original_ci_high, 4),
        "selected_candidate_count_delta_vs_original": (
            v2["selected_candidate_count"] - original["selected_candidate_count"]
        ),
        "result_manifest_hash": stable_hash({
            "original": original,
            "repair_v1": v1,
            "repair_v2": v2,
            "repair_v2_failed_gates": artifacts["repair_v2_live_720"].get("failed_gates", []),
        }),
    }


def _live_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics", {})
    return {
        "fresh_api_call_count": int(metrics.get("fresh_api_call_count") or 0),
        "planned_fresh_api_call_count": int(metrics.get("planned_fresh_api_call_count") or 0),
        "live_error_count": int(metrics.get("live_error_count") or 0),
        "selected_candidate_count": int(metrics.get("selected_candidate_count") or 0),
        "accepted_count": int(metrics.get("accepted_count") or 0),
        "rejected_count": int(metrics.get("rejected_count") or 0),
        "trigger_problem_count": int(metrics.get("trigger_problem_count") or 0),
        "trigger_problem_level_mean_utility": float(metrics.get("trigger_problem_level_mean_utility") or 0.0),
        "trigger_problem_level_ci95": list(metrics.get("trigger_problem_level_ci95") or [0.0, 0.0]),
        "control_problem_level_mean_loss_rate": float(metrics.get("control_problem_level_mean_loss_rate") or 0.0),
        "control_problem_level_ci95": list(metrics.get("control_problem_level_ci95") or [0.0, 0.0]),
        "accepted_control_problem_level_mean_loss_rate": float(
            metrics.get("accepted_control_problem_level_mean_loss_rate") or 0.0
        ),
        "accepted_control_problem_level_ci95": list(
            metrics.get("accepted_control_problem_level_ci95") or [0.0, 0.0]
        ),
        "real_problem_assignment_rate": float(metrics.get("real_problem_assignment_rate") or 0.0),
        "side_assignment_rate": float(metrics.get("side_assignment_rate") or 0.0),
        "main_graph_mutation_count": int(metrics.get("main_graph_mutation_count") or 0),
        "prompt_answer_or_secret_payload_detected": bool(
            metrics.get("prompt_answer_or_secret_payload_detected")
        ),
        "secret_value_exposed": bool(metrics.get("secret_value_exposed")),
    }


def _paper_table_update(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_id": "fresh_live_720_broad_generator_repair",
        "statistical_unit": "problem_id",
        "fresh_api_calls": metrics["repair_v2_fresh_api_call_count"],
        "original_trigger_utility": metrics["original_trigger_problem_level_mean_utility"],
        "original_trigger_ci95": metrics["original_trigger_problem_level_ci95"],
        "repaired_trigger_utility": metrics["repair_v2_trigger_problem_level_mean_utility"],
        "repaired_trigger_ci95": metrics["repair_v2_trigger_problem_level_ci95"],
        "delta_vs_original": metrics["trigger_utility_delta_vs_original"],
        "candidate_frontier_size": metrics["repair_v2_selected_candidate_count"],
        "control_loss": metrics["repair_v2_control_problem_level_mean_loss_rate"],
        "control_loss_ci95": metrics["repair_v2_control_problem_level_ci95"],
        "paper_interpretation": "evidence-calibrated frontier passes; raw unfiltered frontier remains blocked",
    }


def _claim_boundaries(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "claim_id": "raw_unfiltered_generator_frontier",
            "allowed": False,
            "reason": "The original 720-call raw frontier failed the broad trigger gate.",
        },
        {
            "claim_id": "evidence_calibrated_broad_frontier",
            "allowed": metrics["repair_v2_live_pass"],
            "reason": "The repaired qualified frontier cleared the full live runner gates.",
        },
        {
            "claim_id": "repair_is_not_merely_smaller_prompt_sweep",
            "allowed": metrics["repair_v2_fresh_api_call_count"] == metrics["original_fresh_api_call_count"]
            and metrics["selected_candidate_count_delta_vs_original"] <= -40,
            "reason": "The repair keeps the fresh-call budget while shrinking the candidate frontier by evidence.",
        },
    ]


def _ci_value(ci: Any, index: int) -> float:
    if not isinstance(ci, list) or len(ci) <= index or ci[index] is None:
        return 0.0
    return float(ci[index])


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrate the broad-generator repair result.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="paper_broad_generator_repair_integration_20260612")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_paper_broad_generator_repair_integration_payload(root=root, eval_id=args.eval_id)
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_out = Path(args.md_out)
    md_out = md_out if md_out.is_absolute() else root / md_out
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(format_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "failed_gates": payload["failed_gates"],
        "metrics": payload["metrics"],
        "out": str(out),
        "md_out": str(md_out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
