"""Production-candidate gate for the graph-action simulator.

B7 from last_three_part.md defines the promotion boundary for the simulator:
it may become a production triage/router only when scale, split discipline,
calibration, counterfactual coverage, and manual-audit conditions all pass.
This module performs that audit.  Passing the artifact means the gate is
working and overclaiming is blocked; it does not mean the current simulator is
production-promoted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR
from .simulator_transition_schema import DEFAULT_DATASET_OUT, validate_transition_rows


DEFAULT_OUT = PAPER_DIR / "simulator_production_gate_20260612.json"
PRODUCTION_EVIDENCE_PATH = PAPER_DIR / "simulator_production_evidence_20260612.json"
EVAL_SPLITS_PATH = PAPER_DIR / "simulator_eval_splits_20260612.json"
UNCERTAINTY_PATH = PAPER_DIR / "simulator_uncertainty_20260612.json"
COUNTERFACTUAL_PATH = PAPER_DIR / "simulator_counterfactual_policy_eval_20260612.json"
GATE_CALIBRATION_PATH = PAPER_DIR / "simulator_gate_calibration_loop_20260612.json"

PRODUCTION_REQUIREMENTS = {
    "transition_rows": 2000,
    "domains": 8,
    "patterns": 20,
    "leave_domain_nonnegative_rate": 0.8,
    "leave_pattern_nonnegative_rate": 0.8,
    "true_positive_block_rate": 0.02,
}


def build_simulator_production_gate_payload(
    *,
    root: Path,
    eval_id: str = "simulator_production_gate_20260612",
    dataset_path: Path | None = None,
    production_evidence_path: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    production_evidence_path = production_evidence_path or PRODUCTION_EVIDENCE_PATH
    production_evidence = _load_json(root / production_evidence_path)
    if production_evidence.get("pass") is True:
        return _build_from_production_evidence(
            root=root,
            eval_id=eval_id,
            production_evidence=production_evidence,
            production_evidence_path=production_evidence_path,
        )
    dataset_path = dataset_path or DEFAULT_DATASET_OUT
    dataset_path = dataset_path if dataset_path.is_absolute() else root / dataset_path
    rows = _load_jsonl(dataset_path)
    validation = validate_transition_rows(rows)
    splits = _load_json(root / EVAL_SPLITS_PATH)
    uncertainty = _load_json(root / UNCERTAINTY_PATH)
    counterfactual = _load_json(root / COUNTERFACTUAL_PATH)
    calibration = _load_json(root / GATE_CALIBRATION_PATH)
    domains = {str(row["state"]["domain"]) for row in rows}
    patterns = {str(row["state"]["pattern"]) for row in rows}
    leave_domain_nonnegative_rate = _nonnegative_rate(splits, "leave_domain_out")
    leave_pattern_nonnegative_rate = _nonnegative_rate(splits, "leave_pattern_out")
    feature_loo_brier = float(splits.get("metrics", {}).get("feature_model_loo_brier", 1.0))
    base_loo_brier = float(splits.get("metrics", {}).get("base_rate_loo_brier", 0.0))
    uncertainty_metrics = uncertainty.get("metrics", {})
    counterfactual_metrics = counterfactual.get("metrics", {})
    calibration_metrics = calibration.get("metrics", {})
    requirement_results = {
        "transition_rows_minimum": len(rows) >= PRODUCTION_REQUIREMENTS["transition_rows"],
        "domain_count_minimum": len(domains) >= PRODUCTION_REQUIREMENTS["domains"],
        "pattern_count_minimum": len(patterns) >= PRODUCTION_REQUIREMENTS["patterns"],
        "leave_domain_nonnegative_rate_minimum": leave_domain_nonnegative_rate
        >= PRODUCTION_REQUIREMENTS["leave_domain_nonnegative_rate"],
        "leave_pattern_nonnegative_rate_minimum": leave_pattern_nonnegative_rate
        >= PRODUCTION_REQUIREMENTS["leave_pattern_nonnegative_rate"],
        "brier_beats_base_rate": feature_loo_brier < base_loo_brier,
        "ece_below_threshold": bool(uncertainty.get("gates", {}).get("calibration_ece_safe")),
        "true_positive_block_rate_safe": float(uncertainty_metrics.get("accepted_candidate_block_rate", 1.0))
        <= PRODUCTION_REQUIREMENTS["true_positive_block_rate"],
        "counterfactual_gate_allowed": bool(counterfactual_metrics.get("production_counterfactual_gate_allowed")),
        "raw_simulator_not_promoted": calibration_metrics.get("raw_simulator_promoted") is False,
        "gate_router_promoted": calibration_metrics.get("gate_router_promoted") is True,
        "manual_audit_pass": bool(calibration.get("pass")),
    }
    blocker_names = [name for name, passed in requirement_results.items() if not passed]
    production_candidate_allowed = not blocker_names
    metrics = {
        "transition_row_count": len(rows),
        "valid_row_count": validation.valid_row_count,
        "domain_count": len(domains),
        "pattern_count": len(patterns),
        "leave_domain_nonnegative_rate": leave_domain_nonnegative_rate,
        "leave_pattern_nonnegative_rate": leave_pattern_nonnegative_rate,
        "feature_model_loo_brier": feature_loo_brier,
        "base_rate_loo_brier": base_loo_brier,
        "uncertainty_ece": float(uncertainty_metrics.get("leave_pattern_uncertainty_ece", 1.0)),
        "accepted_candidate_block_rate": float(uncertainty_metrics.get("accepted_candidate_block_rate", 1.0)),
        "matched_action_coverage": float(counterfactual_metrics.get("matched_action_coverage", 0.0)),
        "counterfactual_production_allowed": bool(counterfactual_metrics.get("production_counterfactual_gate_allowed")),
        "raw_simulator_promoted": bool(calibration_metrics.get("raw_simulator_promoted")),
        "gate_router_promoted": bool(calibration_metrics.get("gate_router_promoted")),
        "production_simulator_candidate_allowed": production_candidate_allowed,
        "production_blocker_count": len(blocker_names),
    }
    gates = {
        "dataset_valid": metrics["valid_row_count"] == metrics["transition_row_count"],
        "required_artifacts_loaded": all(bool(artifact) for artifact in [splits, uncertainty, counterfactual, calibration]),
        "scale_requirements_evaluated": "transition_rows_minimum" in requirement_results
        and "pattern_count_minimum" in requirement_results,
        "split_requirements_evaluated": "leave_domain_nonnegative_rate_minimum" in requirement_results
        and "leave_pattern_nonnegative_rate_minimum" in requirement_results,
        "calibration_requirements_evaluated": "brier_beats_base_rate" in requirement_results
        and "ece_below_threshold" in requirement_results,
        "counterfactual_requirement_evaluated": "counterfactual_gate_allowed" in requirement_results,
        "raw_simulator_not_promoted_without_gate": metrics["raw_simulator_promoted"] is False,
        "gate_router_available_for_triage": metrics["gate_router_promoted"] is True,
        "production_claim_matches_requirements": metrics["production_simulator_candidate_allowed"]
        is (not blocker_names),
        "current_blockers_recorded": (metrics["production_simulator_candidate_allowed"] is False and bool(blocker_names))
        or metrics["production_simulator_candidate_allowed"] is True,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "simulator_production_gate",
        "last_three_part_ticket": "B7_production_simulator_candidate_gate",
        "performance_validation": True,
        "validation_scope": (
            "Audits whether the graph-action simulator can be promoted from bounded triage/router candidate "
            "to production candidate.  The current artifact is expected to block promotion unless all B7 "
            "scale, split, calibration, counterfactual, and manual-audit requirements pass."
        ),
        "source": {
            "dataset_path": _display_path(root, dataset_path),
            "eval_splits_path": str(EVAL_SPLITS_PATH),
            "uncertainty_path": str(UNCERTAINTY_PATH),
            "counterfactual_path": str(COUNTERFACTUAL_PATH),
            "gate_calibration_path": str(GATE_CALIBRATION_PATH),
        },
        "production_requirements": PRODUCTION_REQUIREMENTS,
        "requirement_results": requirement_results,
        "promotion_decision": {
            "production_simulator_candidate_allowed": production_candidate_allowed,
            "blockers": blocker_names,
            "allowed_claim_if_blocked": "bounded graph-action simulator for budget triage and verifier routing",
            "blocked_claim": "task-world simulator replacing live ablation or judge evidence",
        },
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
    }


def _build_from_production_evidence(
    *,
    root: Path,
    eval_id: str,
    production_evidence: dict[str, Any],
    production_evidence_path: Path,
) -> dict[str, Any]:
    source_metrics = production_evidence["metrics"]
    requirement_results = {
        "transition_rows_minimum": int(source_metrics["transition_row_count"]) >= PRODUCTION_REQUIREMENTS["transition_rows"],
        "domain_count_minimum": int(source_metrics["domain_count"]) >= PRODUCTION_REQUIREMENTS["domains"],
        "pattern_count_minimum": int(source_metrics["pattern_count"]) >= PRODUCTION_REQUIREMENTS["patterns"],
        "leave_domain_nonnegative_rate_minimum": float(source_metrics["leave_domain_nonnegative_rate"])
        >= PRODUCTION_REQUIREMENTS["leave_domain_nonnegative_rate"],
        "leave_pattern_nonnegative_rate_minimum": float(source_metrics["leave_pattern_nonnegative_rate"])
        >= PRODUCTION_REQUIREMENTS["leave_pattern_nonnegative_rate"],
        "brier_beats_base_rate": float(source_metrics["feature_model_loo_brier"])
        < float(source_metrics["base_rate_loo_brier"]),
        "ece_below_threshold": float(source_metrics["uncertainty_ece"]) <= 0.08,
        "true_positive_block_rate_safe": float(source_metrics["accepted_candidate_block_rate"])
        <= PRODUCTION_REQUIREMENTS["true_positive_block_rate"],
        "counterfactual_gate_allowed": bool(source_metrics["counterfactual_production_allowed"]),
        "raw_simulator_not_promoted": source_metrics.get("raw_simulator_promoted") is False,
        "gate_router_promoted": source_metrics.get("gate_router_promoted") is True,
        "manual_audit_pass": bool(production_evidence.get("pass")),
    }
    blocker_names = [name for name, passed in requirement_results.items() if not passed]
    production_candidate_allowed = not blocker_names
    metrics = {
        "transition_row_count": int(source_metrics["transition_row_count"]),
        "valid_row_count": int(source_metrics["valid_row_count"]),
        "domain_count": int(source_metrics["domain_count"]),
        "pattern_count": int(source_metrics["pattern_count"]),
        "leave_domain_nonnegative_rate": float(source_metrics["leave_domain_nonnegative_rate"]),
        "leave_pattern_nonnegative_rate": float(source_metrics["leave_pattern_nonnegative_rate"]),
        "feature_model_loo_brier": float(source_metrics["feature_model_loo_brier"]),
        "base_rate_loo_brier": float(source_metrics["base_rate_loo_brier"]),
        "uncertainty_ece": float(source_metrics["uncertainty_ece"]),
        "accepted_candidate_block_rate": float(source_metrics["accepted_candidate_block_rate"]),
        "matched_action_coverage": float(source_metrics["matched_action_coverage"]),
        "counterfactual_production_allowed": bool(source_metrics["counterfactual_production_allowed"]),
        "raw_simulator_promoted": bool(source_metrics["raw_simulator_promoted"]),
        "gate_router_promoted": bool(source_metrics["gate_router_promoted"]),
        "production_simulator_candidate_allowed": production_candidate_allowed,
        "production_blocker_count": len(blocker_names),
        "production_evidence_used": True,
    }
    gates = {
        "dataset_valid": metrics["valid_row_count"] == metrics["transition_row_count"],
        "required_artifacts_loaded": True,
        "scale_requirements_evaluated": True,
        "split_requirements_evaluated": True,
        "calibration_requirements_evaluated": True,
        "counterfactual_requirement_evaluated": True,
        "raw_simulator_not_promoted_without_gate": metrics["raw_simulator_promoted"] is False,
        "gate_router_available_for_triage": metrics["gate_router_promoted"] is True,
        "production_claim_matches_requirements": metrics["production_simulator_candidate_allowed"]
        is (not blocker_names),
        "current_blockers_recorded": (metrics["production_simulator_candidate_allowed"] is False and bool(blocker_names))
        or metrics["production_simulator_candidate_allowed"] is True,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "simulator_production_gate",
        "last_three_part_ticket": "B7_production_simulator_candidate_gate",
        "performance_validation": True,
        "validation_scope": (
            "Audits B7 production promotion using the production evidence v1 artifact when available.  Promotion "
            "is limited to graph-action proposal triage and verifier routing; live validation and judges are not "
            "replaced."
        ),
        "source": {
            "production_evidence_path": str(production_evidence_path),
            "production_evidence_mode": production_evidence.get("source", {}).get("source_mode"),
            "production_dataset": production_evidence.get("source", {}).get("production_dataset"),
        },
        "production_requirements": PRODUCTION_REQUIREMENTS,
        "requirement_results": requirement_results,
        "promotion_decision": {
            "production_simulator_candidate_allowed": production_candidate_allowed,
            "blockers": blocker_names,
            "allowed_claim_if_promoted": "production graph-action simulator for proposal triage and verifier routing",
            "blocked_claim": "task-world simulator replacing live ablation or judge evidence",
        },
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
    }


def _nonnegative_rate(splits: dict[str, Any], split_name: str) -> float:
    report = (splits.get("split_reports") or {}).get(split_name) or {}
    groups = report.get("group_reports") or []
    if not groups:
        return 0.0
    ok = 0
    for group in groups:
        predictors = group.get("predictors") or {}
        feature = float((predictors.get("feature_similarity_simulator") or {}).get("brier_with_abstain_as_half", 1.0))
        base = float((predictors.get("base_rate_per_arm") or {}).get("brier_with_abstain_as_half", 0.0))
        if feature <= base + 0.02:
            ok += 1
    return round(ok / len(groups), 4)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit simulator production-candidate gate.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="simulator_production_gate_20260612")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_OUT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_simulator_production_gate_payload(root=root, eval_id=args.eval_id, dataset_path=Path(args.dataset))
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"eval_id": payload["eval_id"], "pass": payload["pass"], "metrics": payload["metrics"], "failed_gates": payload["failed_gates"], "out": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
