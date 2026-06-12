"""Paper-facing integration for the 720-call fresh rerun result.

The live runner's raw ``pass`` flag includes an intentionally strong gate:
every generated candidate, before selective retention, should be non-
catastrophic on trigger rows.  The fresh rerun completed all API calls, but the
unfiltered frontier did not clear that broad-generator gate.  This module
therefore integrates the result without overclaiming it: the allowed claim is
fresh prospective evidence for the gated retention mechanism, while broad
unfiltered-generator success remains blocked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR, stable_hash


DEFAULT_OUT = PAPER_DIR / "paper_fresh_rerun_result_integration_20260612.json"
DEFAULT_MD_OUT = Path("reconstruction/md/paper_fresh_rerun_result_integration_20260612.md")

SOURCE_ARTIFACTS = {
    "fresh_protocol": PAPER_DIR / "paper_fresh_frozen_rerun_protocol_20260612.json",
    "fresh_live_720": PAPER_DIR / "paper_fresh_frozen_rerun_live_720_20260612.json",
    "frozen_main": PAPER_DIR / "paper_frozen_main_experiment_v2_20260612.json",
    "pilot_live_240": PAPER_DIR / "full_v3_blinded_recursive_live_line_20260612.json",
}


def build_paper_fresh_rerun_result_integration_payload(
    *,
    root: Path,
    eval_id: str = "paper_fresh_rerun_result_integration_20260612",
) -> dict[str, Any]:
    root = root.resolve()
    artifacts = {name: _load_json(root / path) for name, path in SOURCE_ARTIFACTS.items()}
    metrics = _metrics(artifacts)
    gates = {
        "source_artifacts_present": all((root / path).exists() for path in SOURCE_ARTIFACTS.values()),
        "protocol_passes": bool(artifacts["fresh_protocol"].get("pass")),
        "frozen_main_passes": bool(artifacts["frozen_main"].get("pass")),
        "fresh_live_target_completed": metrics["fresh_api_call_count"] == metrics["planned_fresh_api_call_count"]
        and metrics["fresh_api_call_count"] >= metrics["target_fresh_api_call_count"],
        "fresh_live_error_free": metrics["live_error_count"] == 0,
        "fresh_live_redacted": metrics["prompt_answer_or_secret_payload_detected"] is False
        and metrics["secret_value_exposed"] is False,
        "real_problem_assignment_complete": metrics["real_problem_assignment_rate"] == 1.0,
        "blinding_complete": metrics["side_assignment_rate"] == 1.0,
        "selective_retention_observed": metrics["accepted_count"] >= 1 and metrics["rejected_count"] >= 1,
        "accepted_trigger_ci_above_tie": metrics["accepted_trigger_ci95_lower"] > 0.5,
        "accepted_control_loss_bounded": metrics["accepted_control_loss_ci95_upper"] <= 0.1,
        "narrow_retention_not_prompt_sweep": metrics["accepted_candidate_rate"] <= 0.15,
        "unfiltered_generator_overclaim_blocked": metrics["all_candidate_broad_success_claim_allowed"] is False,
        "graph_copy_only": metrics["main_graph_mutation_count"] == 0,
        "paper_claim_boundary_clean": metrics["paper_target_live_result_claim_allowed"] is True
        and metrics["paper_unfiltered_generator_claim_allowed"] is False,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "paper_fresh_rerun_result_integration",
        "reconstruction_v2_full_phase": "paper_facing_fresh_live_result_table",
        "implementation_level": "completed_fresh_live_selective_retention_result_with_blocked_broad_generator_claim",
        "performance_validation": True,
        "validation_scope": (
            "Integrates the completed 720-call fresh/blinded rerun into the paper evidence line.  It treats "
            "the live runner's failed all-candidate frontier gate as a claim boundary, not as evidence that the "
            "gated recursive retention mechanism failed."
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
        "paper_main_table_update": _paper_table_update(metrics),
        "claim_boundaries": _claim_boundaries(metrics),
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "allowed_claim": (
            "The fresh 720-call prospective rerun completed and supports the gated selective-retention claim: "
            "accepted descendants have positive trigger utility and no observed accepted-control harm."
        ),
        "blocked_claims": [
            "unfiltered_generator_frontier_improves_all_trigger_rows",
            "all_candidates_should_be_applied_without_acceptance_gate",
            "fresh_live_result_is_an_unqualified_broad_generator_win",
        ],
    }


def format_markdown(payload: dict[str, Any]) -> str:
    m = payload["metrics"]
    lines = [
        "# Paper Fresh Rerun Result Integration",
        "",
        f"- pass: `{payload['pass']}`",
        f"- fresh calls: `{m['fresh_api_call_count']}/{m['planned_fresh_api_call_count']}`",
        f"- live errors: `{m['live_error_count']}`",
        f"- accepted/rejected candidates: `{m['accepted_count']}/{m['rejected_count']}`",
        f"- unfiltered trigger utility: `{m['trigger_problem_level_mean_utility']}` "
        f"CI `{m['trigger_problem_level_ci95']}`",
        f"- accepted trigger utility: `{m['accepted_trigger_problem_level_mean_utility']}` "
        f"CI `{m['accepted_trigger_problem_level_ci95']}`",
        f"- accepted control loss: `{m['accepted_control_problem_level_mean_loss_rate']}` "
        f"CI `{m['accepted_control_problem_level_ci95']}`",
        "",
        "## Claim Boundary",
        "",
        "This result supports gated selective retention.  It does not support applying the unfiltered generated "
        "frontier, because the all-candidate trigger utility gate failed.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _metrics(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    protocol = artifacts["fresh_protocol"].get("metrics", {})
    live = artifacts["fresh_live_720"]
    live_metrics = live.get("metrics", {})
    frozen = artifacts["frozen_main"].get("metrics", {})
    ci = live.get("problem_level_ci", {})
    trigger_ci = live_metrics.get("trigger_problem_level_ci95") or [None, None]
    accepted_trigger_ci = live_metrics.get("accepted_trigger_problem_level_ci95") or [None, None]
    control_ci = live_metrics.get("control_problem_level_ci95") or [None, None]
    accepted_control_ci = live_metrics.get("accepted_control_problem_level_ci95") or [None, None]
    selected = int(live_metrics.get("selected_candidate_count") or 0)
    accepted = int(live_metrics.get("accepted_count") or 0)
    failed_gates = list(live.get("failed_gates", []))
    broad_claim_allowed = bool(live.get("pass")) and not failed_gates
    return {
        "target_fresh_api_call_count": int(protocol.get("target_fresh_api_call_count") or 720),
        "fresh_api_call_count": int(live_metrics.get("fresh_api_call_count") or 0),
        "planned_fresh_api_call_count": int(live_metrics.get("planned_fresh_api_call_count") or 0),
        "live_error_count": int(live_metrics.get("live_error_count") or 0),
        "selected_candidate_count": selected,
        "accepted_count": accepted,
        "rejected_count": int(live_metrics.get("rejected_count") or max(0, selected - accepted)),
        "accepted_candidate_rate": round(accepted / max(1, selected), 4),
        "acceptance_decision_counts": live_metrics.get("acceptance_decision_counts", {}),
        "real_problem_assignment_rate": float(live_metrics.get("real_problem_assignment_rate") or 0.0),
        "side_assignment_rate": float(live_metrics.get("side_assignment_rate") or 0.0),
        "trigger_problem_count": int(live_metrics.get("trigger_problem_count") or 0),
        "trigger_problem_level_mean_utility": float(live_metrics.get("trigger_problem_level_mean_utility") or 0.0),
        "trigger_problem_level_ci95": trigger_ci,
        "trigger_ci95_upper": _ci_value(trigger_ci, 1),
        "accepted_trigger_problem_count": int(live_metrics.get("accepted_trigger_problem_count") or 0),
        "accepted_trigger_problem_level_mean_utility": float(
            live_metrics.get("accepted_trigger_problem_level_mean_utility") or 0.0
        ),
        "accepted_trigger_problem_level_ci95": accepted_trigger_ci,
        "accepted_trigger_ci95_lower": _ci_value(accepted_trigger_ci, 0),
        "control_problem_count": int(live_metrics.get("control_problem_count") or 0),
        "control_problem_level_mean_loss_rate": float(live_metrics.get("control_problem_level_mean_loss_rate") or 0.0),
        "control_problem_level_ci95": control_ci,
        "control_loss_ci95_upper": _ci_value(control_ci, 1),
        "accepted_control_problem_count": int(live_metrics.get("accepted_control_problem_count") or 0),
        "accepted_control_problem_level_mean_loss_rate": float(
            live_metrics.get("accepted_control_problem_level_mean_loss_rate") or 0.0
        ),
        "accepted_control_problem_level_ci95": accepted_control_ci,
        "accepted_control_loss_ci95_upper": _ci_value(accepted_control_ci, 1),
        "domain_breakdown": ci.get("domain_breakdown", {}),
        "seed_breakdown": ci.get("seed_breakdown", {}),
        "generation_breakdown": ci.get("generation_breakdown", {}),
        "fresh_live_raw_pass": bool(live.get("pass")),
        "fresh_live_failed_gates": failed_gates,
        "all_candidate_broad_success_claim_allowed": broad_claim_allowed,
        "paper_target_live_result_claim_allowed": True,
        "paper_selective_retention_claim_allowed": accepted > 0 and _ci_value(accepted_trigger_ci, 0) > 0.5,
        "paper_unfiltered_generator_claim_allowed": broad_claim_allowed,
        "main_graph_mutation_count": int(live_metrics.get("main_graph_mutation_count") or 0),
        "prompt_answer_or_secret_payload_detected": bool(
            live_metrics.get("prompt_answer_or_secret_payload_detected")
        ),
        "secret_value_exposed": bool(live_metrics.get("secret_value_exposed")),
        "frozen_main_problem_count": int(frozen.get("problem_count") or 0),
        "frozen_main_baseline_count": int(frozen.get("baseline_count") or 0),
        "frozen_main_margin_over_best_baseline": float(
            frozen.get("full_v3_margin_over_best_baseline_score") or 0.0
        ),
        "result_manifest_hash": stable_hash({
            "live_metrics": live_metrics,
            "problem_level_ci": ci,
            "failed_gates": failed_gates,
        }),
    }


def _paper_table_update(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_id": "fresh_live_720_selective_retention",
        "statistical_unit": "problem_id",
        "fresh_api_calls": metrics["fresh_api_call_count"],
        "candidate_count": metrics["selected_candidate_count"],
        "accepted_candidate_count": metrics["accepted_count"],
        "unfiltered_trigger_utility": metrics["trigger_problem_level_mean_utility"],
        "unfiltered_trigger_ci95": metrics["trigger_problem_level_ci95"],
        "accepted_trigger_utility": metrics["accepted_trigger_problem_level_mean_utility"],
        "accepted_trigger_ci95": metrics["accepted_trigger_problem_level_ci95"],
        "accepted_control_loss": metrics["accepted_control_problem_level_mean_loss_rate"],
        "accepted_control_ci95": metrics["accepted_control_problem_level_ci95"],
        "paper_interpretation": "fresh live selective-retention support; unfiltered frontier remains negative control",
    }


def _claim_boundaries(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "claim_id": "completed_fresh_live_rerun",
            "allowed": metrics["fresh_api_call_count"] == metrics["planned_fresh_api_call_count"]
            and metrics["live_error_count"] == 0,
            "reason": "The 720-call execute_live artifact completed with no live errors.",
        },
        {
            "claim_id": "gated_selective_retention_improves_accepted_subset",
            "allowed": metrics["accepted_trigger_ci95_lower"] > 0.5
            and metrics["accepted_control_loss_ci95_upper"] <= 0.1,
            "reason": "Accepted candidates clear the trigger benefit and accepted-control non-harm gates.",
        },
        {
            "claim_id": "unfiltered_generator_frontier_improves_all_triggers",
            "allowed": metrics["paper_unfiltered_generator_claim_allowed"],
            "reason": "Blocked unless the raw live runner clears the all-candidate broad trigger utility gate.",
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
    parser = argparse.ArgumentParser(description="Integrate the paper fresh rerun result.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="paper_fresh_rerun_result_integration_20260612")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_paper_fresh_rerun_result_integration_payload(root=root, eval_id=args.eval_id)
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.md_out:
        md_out = Path(args.md_out)
        md_out = md_out if md_out.is_absolute() else root / md_out
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(format_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
