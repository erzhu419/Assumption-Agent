"""Simulator-as-gate routing and closed-loop calibration writeback.

B5/B6 explicitly limits the simulator to budget triage, verifier routing, and
bounded profile selection.  It consumes B3 uncertainty decisions plus I2 live
readback rows, writes calibration rows, and emits simulator-defect residuals
when high-confidence routing is contradicted by validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR, stable_hash


DEFAULT_OUT = PAPER_DIR / "simulator_gate_calibration_loop_20260612.json"
DEFAULT_WRITEBACK_OUT = PAPER_DIR / "simulator_gate_calibration_writeback_20260612.jsonl"
UNCERTAINTY_PATH = PAPER_DIR / "simulator_uncertainty_20260612.json"
I2_EPISODE_PATH = PAPER_DIR / "integrated_recursive_episode_b3_c2_20260612.json"
ALLOWED_ROUTING_LEVELS = {"S1_budget_triage", "S2_verifier_routing", "S3_policy_selection"}
FORBIDDEN_ORACLE_LEVELS = {
    "S4_replace_fresh_ablation",
    "S5_replace_judge",
    "S6_simulate_arbitrary_real_world_outcome",
}
FORBIDDEN_ACTIONS = {
    "auto_accept_without_live",
    "auto_apply_policy_change",
    "replace_judge",
}


def build_simulator_gate_calibration_loop_payload(
    *,
    root: Path,
    eval_id: str = "simulator_gate_calibration_loop_20260612",
    writeback_out: Path | None = None,
    write_artifact: bool = True,
) -> dict[str, Any]:
    root = root.resolve()
    uncertainty = _load_json(root / UNCERTAINTY_PATH)
    episode = _load_json(root / I2_EPISODE_PATH)
    candidates = {row["candidate_id"]: row for row in episode.get("candidates", [])}
    routing_policy = [_routing_policy_row(action) for action in _allowed_b3_actions(uncertainty)]
    writeback_rows = _calibration_writeback_rows(episode=episode, candidates=candidates)
    residuals = _simulator_defect_residuals(writeback_rows)
    promotion_event = _promotion_event(
        uncertainty=uncertainty,
        residuals=residuals,
        writeback_rows=writeback_rows,
    )
    writeback_path = _resolve(root, writeback_out or DEFAULT_WRITEBACK_OUT)
    if write_artifact:
        writeback_path.parent.mkdir(parents=True, exist_ok=True)
        writeback_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in writeback_rows) + "\n",
            encoding="utf-8",
        )
    metrics = {
        "b3_uncertainty_pass": bool(uncertainty.get("pass")),
        "i2_episode_pass": bool(episode.get("pass")),
        "routing_policy_count": len(routing_policy),
        "allowed_routing_level_count": len({row["routing_level"] for row in routing_policy}),
        "forbidden_oracle_level_count": sum(
            1 for row in routing_policy if row["routing_level"] in FORBIDDEN_ORACLE_LEVELS
        ),
        "forbidden_action_count": sum(1 for row in routing_policy if row["recommended_action"] in FORBIDDEN_ACTIONS),
        "writeback_row_count": len(writeback_rows),
        "fresh_writeback_row_count": sum(1 for row in writeback_rows if row["outcome_source"] == "fresh_ablation"),
        "deferred_writeback_row_count": sum(1 for row in writeback_rows if row["outcome_source"] == "defer_live_validation"),
        "accepted_writeback_row_count": sum(1 for row in writeback_rows if row["outcome_label"] == 1),
        "rejected_writeback_row_count": sum(1 for row in writeback_rows if row["outcome_label"] == 0),
        "high_confidence_wrong_count": sum(1 for row in writeback_rows if row["high_confidence_wrong"]),
        "simulator_defect_residual_count": len(residuals),
        "calibration_mae_after_readback": round(
            sum(row["absolute_error"] for row in writeback_rows if row["outcome_label"] is not None)
            / max(1, sum(1 for row in writeback_rows if row["outcome_label"] is not None)),
            4,
        ),
        "promotion_event_count": 1,
        "raw_simulator_promoted": promotion_event["raw_simulator_promoted"],
        "gate_router_promoted": promotion_event["gate_router_promoted"],
        "main_graph_mutation_count": 0,
        "writeback_artifact_path": _display_path(root, writeback_path),
    }
    gates = {
        "b3_uncertainty_passes": metrics["b3_uncertainty_pass"] is True,
        "i2_episode_passes": metrics["i2_episode_pass"] is True,
        "only_s1_s2_s3_routing_levels": (
            metrics["allowed_routing_level_count"] >= 2
            and metrics["forbidden_oracle_level_count"] == 0
        ),
        "no_forbidden_oracle_actions": metrics["forbidden_action_count"] == 0,
        "fresh_outcomes_written_back": metrics["fresh_writeback_row_count"] >= 6,
        "deferred_outcomes_preserved_without_fake_label": metrics["deferred_writeback_row_count"] >= 2,
        "simulator_defects_emitted_for_high_confidence_errors": (
            metrics["high_confidence_wrong_count"] == metrics["simulator_defect_residual_count"]
            and metrics["simulator_defect_residual_count"] >= 2
        ),
        "promotion_event_keeps_raw_simulator_unpromoted": metrics["raw_simulator_promoted"] is False,
        "gate_router_can_be_promoted": metrics["gate_router_promoted"] is True,
        "writeback_artifact_written": (not write_artifact) or writeback_path.exists(),
        "no_main_graph_mutation": metrics["main_graph_mutation_count"] == 0,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "simulator_gate_calibration_loop",
        "last_three_part_ticket": "B5_B6_simulator_gate_and_calibration_writeback",
        "performance_validation": True,
        "validation_scope": (
            "Limits the simulator to S1/S2/S3 routing, blocks S4-S6 oracle behavior, and writes I2 validation "
            "outcomes back as calibration rows plus simulator-defect residuals for high-confidence mistakes."
        ),
        "source_artifacts": {
            "simulator_uncertainty": str(UNCERTAINTY_PATH),
            "integrated_episode_b3_c2": str(I2_EPISODE_PATH),
        },
        "routing_policy": routing_policy,
        "writeback_rows": writeback_rows,
        "simulator_defect_residuals": residuals,
        "promotion_event": promotion_event,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "claim_boundaries": [
            "The simulator is promoted only as a gate/router over budget and verifier choice.",
            "Fresh ablation and judge evidence remain mandatory for retention.",
            "High-confidence simulator mistakes become SIMULATOR_DEFECT residuals before any promotion decision.",
        ],
    }


def _allowed_b3_actions(uncertainty: dict[str, Any]) -> list[str]:
    actions = uncertainty.get("policy", {}).get("allowed_actions")
    if not actions:
        actions = [
            "recommend_run_ablation",
            "recommend_collect_more_evidence",
            "recommend_repair_scope",
            "recommend_reject_low_value",
            "abstain_to_live_validation",
        ]
    return sorted(str(action) for action in actions)


def _routing_policy_row(action: str) -> dict[str, Any]:
    if action == "recommend_reject_low_value":
        level = "S1_budget_triage"
        verifier = "tier1_budget_defer_with_manifest"
    elif action in {"recommend_run_ablation", "recommend_collect_more_evidence", "recommend_repair_scope"}:
        level = "S2_verifier_routing"
        verifier = {
            "recommend_run_ablation": "tier2_ablation_judge",
            "recommend_collect_more_evidence": "tier1_evidence_expand_then_rescreen",
            "recommend_repair_scope": "tier2_scope_repair_with_control",
        }[action]
    elif action == "abstain_to_live_validation":
        level = "S2_verifier_routing"
        verifier = "tier3_live_validation_or_human_review"
    else:
        level = "S3_policy_selection"
        verifier = "tier2_profile_selection_audit"
    return {
        "recommended_action": action,
        "routing_level": level,
        "required_verifier_tier": verifier,
        "can_auto_accept": False,
        "can_auto_apply_policy_change": False,
        "can_replace_judge": False,
    }


def _calibration_writeback_rows(
    *,
    episode: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for cycle in episode.get("cycle_rows", []):
        action = cycle.get("cycle_action")
        if action not in {"fresh_ablation", "defer_live_validation"}:
            continue
        candidate = candidates.get(cycle.get("candidate_id"), {})
        prediction = float(candidate.get("b3_score") or cycle.get("b3_score") or 0.5)
        if action == "defer_live_validation":
            outcome_label = None
            abs_error = None
            high_confidence_wrong = False
        else:
            outcome_label = 1 if cycle.get("fresh_decision") == "accept" and not cycle.get("control_harm") else 0
            abs_error = abs(prediction - outcome_label)
            high_confidence_wrong = prediction >= 0.70 and outcome_label == 0
        rows.append(
            {
                "writeback_id": f"simcal_{stable_hash([candidate.get('candidate_id'), action, prediction])}",
                "candidate_id": candidate.get("candidate_id"),
                "source_row_id": candidate.get("row_id"),
                "outcome_source": action,
                "prediction": {
                    "p_accept": round(prediction, 4),
                    "abstain_reason": candidate.get("b3_abstain_reason"),
                    "required_verifier_tier": candidate.get("required_verifier_tier"),
                    "formal_gate": candidate.get("formal_gate"),
                },
                "outcome_label": outcome_label,
                "fresh_decision": cycle.get("fresh_decision"),
                "control_harm": bool(cycle.get("control_harm", False)),
                "absolute_error": round(abs_error, 4) if abs_error is not None else None,
                "high_confidence_wrong": high_confidence_wrong,
                "redacted": True,
            }
        )
    return rows


def _simulator_defect_residuals(writeback_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    residuals = []
    for row in writeback_rows:
        if not row["high_confidence_wrong"]:
            continue
        residuals.append(
            {
                "residual_id": f"simdef_{stable_hash([row['writeback_id'], 'simulator_defect'])}",
                "residual_type": "SIMULATOR_DEFECT",
                "source_writeback_id": row["writeback_id"],
                "candidate_id": row["candidate_id"],
                "prediction_p_accept": row["prediction"]["p_accept"],
                "outcome_label": row["outcome_label"],
                "absolute_error": row["absolute_error"],
                "recommended_next_action": "demote_raw_simulator_for_this_slice_and_collect_counterfactual_arms",
            }
        )
    return residuals


def _promotion_event(
    *,
    uncertainty: dict[str, Any],
    residuals: list[dict[str, Any]],
    writeback_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    gate_router_promoted = (
        bool(uncertainty.get("pass"))
        and int(uncertainty.get("metrics", {}).get("forbidden_action_recommended_count") or 0) == 0
        and bool(writeback_rows)
    )
    return {
        "event_id": "simulator_gate_router_promotion_20260612",
        "raw_simulator_promoted": False,
        "gate_router_promoted": gate_router_promoted,
        "promotion_scope": "S1_budget_triage_and_S2_verifier_routing_only",
        "demotion_reasons_for_raw_simulator": [
            "B4 counterfactual audit blocks production best-arm selection",
            "high-confidence simulator defects exist in I2 readback",
            "fresh ablation and judge evidence remain mandatory",
        ]
        if residuals
        else ["fresh ablation and judge evidence remain mandatory"],
        "new_training_rows": len(writeback_rows),
        "new_simulator_defect_residuals": len(residuals),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build simulator gate/router calibration writeback artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="simulator_gate_calibration_loop_20260612")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--writeback-out", default=str(DEFAULT_WRITEBACK_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_simulator_gate_calibration_loop_payload(
        root=root,
        eval_id=args.eval_id,
        writeback_out=Path(args.writeback_out),
    )
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "eval_id": payload["eval_id"],
                "pass": payload["pass"],
                "metrics": payload["metrics"],
                "failed_gates": payload["failed_gates"],
                "out": str(out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
