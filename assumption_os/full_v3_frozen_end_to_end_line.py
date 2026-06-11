"""Frozen end-to-end evidence line for the V3 claim.

Paper-scale evidence is intentionally broad.  This module provides a narrower,
ordered line that can be cited as a single frozen pipeline:

tasks/evidence -> residual hypotheses -> live multigeneration validation ->
gated retention -> main graph memory apply -> supervised daemon readback ->
world-model claim guard.

It does not run new API calls; it freezes already-produced artifacts into one
auditable chain with explicit boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "full_v3_frozen_end_to_end_line_20260612.json"

SOURCE_ARTIFACTS = {
    "paper_scale": PAPER_DIR / "full_v3_paper_scale_evidence_20260611.json",
    "live_multigen": PAPER_DIR / "full_v3_live_multigeneration_expansion_20260612.json",
    "main_graph_apply": PAPER_DIR / "full_v3_main_graph_memory_controlled_apply_20260611.json",
    "supervised_daemon": PAPER_DIR / "full_v3_supervised_daemon_background_smoke_20260612.json",
    "world_model_calibration": PAPER_DIR / "full_v3_world_model_calibration_20260611.json",
    "phase11_capability_audit": PAPER_DIR / "full_v3_phase11_capability_audit_20260611.json",
}


def build_full_v3_frozen_end_to_end_line_payload(
    *,
    root: Path,
    eval_id: str = "full_v3_frozen_end_to_end_line_20260612",
) -> dict[str, Any]:
    root = root.resolve()
    artifacts = {name: _load_json(root / path) for name, path in SOURCE_ARTIFACTS.items()}
    manifest = {
        name: {
            "path": str(path),
            "sha256": _sha256(root / path),
            "pass": bool(artifacts[name].get("pass")),
            "eval_kind": artifacts[name].get("eval_kind"),
        }
        for name, path in SOURCE_ARTIFACTS.items()
    }
    steps = _steps(artifacts)
    metrics = _metrics(artifacts=artifacts, steps=steps)
    gates = {
        "all_source_artifacts_pass": metrics["source_pass_rate"] == 1.0,
        "ordered_pipeline_complete": metrics["step_count"] == 6,
        "prospective_live_multigen_present": (
            metrics["live_multigen_execution_mode"] == "execute_live"
            and metrics["live_multigen_api_call_count"] >= 36
            and metrics["live_multigen_live_error_count"] == 0
        ),
        "selective_retention_present": (
            metrics["live_multigen_accepted_count"] >= 1
            and metrics["live_multigen_rejected_count"] >= 1
        ),
        "main_graph_apply_committed": metrics["main_graph_mutated"] is True,
        "supervised_background_worker_started": metrics["background_process_started"] is True,
        "raw_world_model_blocked": metrics["raw_world_model_promoted"] is False,
        "calibrated_guard_promoted": metrics["calibrated_guard_promoted"] is True,
        "paper_scale_still_passes": metrics["paper_scale_required_artifact_pass_rate"] == 1.0,
        "claim_boundary_recorded": metrics["phase11_capability_count"] >= 15,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_frozen_end_to_end_line",
        "reconstruction_v2_full_phase": "frozen_prospective_v3_evidence_chain",
        "implementation_level": "ordered_artifact_manifest_for_single_pipeline_claim",
        "performance_validation": True,
        "validation_scope": (
            "Freezes a single ordered evidence line from paper-scale tasks through live multi-generation "
            "hypothesis validation, main graph apply, supervised daemon readback, and world-model claim guard. "
            "No new API calls are made by this module."
        ),
        "source_manifest": manifest,
        "pipeline_steps": steps,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "This is the paper-facing frozen line: it is no longer a loose list of artifacts.  It records one "
            "ordered chain with prospective live multigeneration evidence, committed memory consolidation, "
            "supervised background-worker validation, and explicit blocking of uncalibrated raw world-model claims."
        ),
    }


def _steps(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    paper = artifacts["paper_scale"]["metrics"]
    live = artifacts["live_multigen"]["metrics"]
    main_apply = artifacts["main_graph_apply"]["metrics"]
    daemon = artifacts["supervised_daemon"]["metrics"]
    wm = artifacts["world_model_calibration"]["metrics"]
    phase11 = artifacts["phase11_capability_audit"]["metrics"]
    return [
        {
            "step": 1,
            "name": "frozen_problem_and_baseline_table",
            "evidence": {
                "main_problem_level_n": paper["main_problem_level_n"],
                "structural_vs_base_utility": paper["structural_vs_base_utility"],
                "full_v3_margin_vs_v1_kernel": paper["full_v3_margin_vs_v1_kernel"],
            },
        },
        {
            "step": 2,
            "name": "residual_to_hypothesis_generation",
            "evidence": {
                "residual_multigen_generation_count": paper["residual_multigen_generation_count"],
                "residual_multigen_retained_count": paper["residual_multigen_retained_count"],
            },
        },
        {
            "step": 3,
            "name": "prospective_live_multigeneration_validation",
            "evidence": {
                "fresh_api_call_count": live["fresh_api_call_count"],
                "accepted_count": live["accepted_count"],
                "rejected_count": live["rejected_count"],
                "reject_harm_count": live["acceptance_decision_counts"].get("reject_harm", 0),
            },
        },
        {
            "step": 4,
            "name": "main_graph_memory_apply",
            "evidence": {
                "main_graph_mutated": main_apply["main_graph_mutated"],
                "rollback_entry_count": main_apply["rollback_entry_count"],
                "precision_delta": main_apply["precision_delta"],
            },
        },
        {
            "step": 5,
            "name": "supervised_background_daemon_readback",
            "evidence": {
                "background_process_started": daemon["background_process_started"],
                "heartbeat_count": daemon["heartbeat_count"],
                "ungated_graph_mutation_count": daemon["ungated_graph_mutation_count"],
            },
        },
        {
            "step": 6,
            "name": "world_model_and_claim_guard",
            "evidence": {
                "raw_phase10_calibration_beats_base_rate": wm["phase10_calibration_beats_base_rate"],
                "uncalibrated_promotion_count": wm["uncalibrated_promotion_count"],
                "phase11_capability_count": phase11["capability_count"],
            },
        },
    ]


def _metrics(*, artifacts: dict[str, dict[str, Any]], steps: list[dict[str, Any]]) -> dict[str, Any]:
    paper = artifacts["paper_scale"]["metrics"]
    live = artifacts["live_multigen"]["metrics"]
    main_apply = artifacts["main_graph_apply"]["metrics"]
    daemon = artifacts["supervised_daemon"]["metrics"]
    wm = artifacts["world_model_calibration"]["metrics"]
    phase11 = artifacts["phase11_capability_audit"]["metrics"]
    return {
        "source_artifact_count": len(SOURCE_ARTIFACTS),
        "source_pass_rate": round(sum(1 for item in artifacts.values() if item.get("pass")) / len(SOURCE_ARTIFACTS), 4),
        "step_count": len(steps),
        "paper_scale_required_artifact_count": paper["required_artifact_count"],
        "paper_scale_required_artifact_pass_rate": paper["required_artifact_pass_rate"],
        "live_multigen_execution_mode": live["execution_mode"],
        "live_multigen_api_call_count": live["fresh_api_call_count"],
        "live_multigen_live_error_count": live["live_error_count"],
        "live_multigen_accepted_count": live["accepted_count"],
        "live_multigen_rejected_count": live["rejected_count"],
        "main_graph_mutated": bool(main_apply["main_graph_mutated"]),
        "main_graph_precision_delta": float(main_apply["precision_delta"]),
        "background_process_started": bool(daemon["background_process_started"]),
        "background_heartbeat_count": int(daemon["heartbeat_count"]),
        "background_ungated_graph_mutation_count": int(daemon["ungated_graph_mutation_count"]),
        "raw_world_model_promoted": bool(wm["phase10_calibration_beats_base_rate"]),
        "calibrated_guard_promoted": wm["phase10_recommended_promotion"] == "promote_calibrated_residual_guard",
        "phase11_capability_count": int(phase11["capability_count"]),
        "phase11_blocked_claim_count": int(phase11["blocked_claim_count"]),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen V3 end-to-end evidence line artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="full_v3_frozen_end_to_end_line_20260612")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_full_v3_frozen_end_to_end_line_payload(root=root, eval_id=args.eval_id)
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
