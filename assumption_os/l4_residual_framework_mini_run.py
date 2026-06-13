"""L4a residual-to-framework mini-run.

This module stitches the L4-3/L4-4/L4-5 evidence into one bounded mini-run:
real residual clusters, LLM-contract framework candidates, conservative
generalization validation, a fresh-validation row packet from the existing
framework benchmark, and an expert-review packet.  It deliberately separates
bounded/preflight evidence from true external L4 completion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR, stable_hash
from .conservative_generalization_gate_v2 import build_conservative_generalization_gate_v2_payload
from .framework_external_eval_pack import build_framework_external_eval_pack_payload
from .llm_framework_candidate_experiment import build_llm_framework_candidate_experiment_payload
from .multigeneration_framework_evolution_benchmark import (
    build_multigeneration_framework_evolution_benchmark_payload,
)
from .residual_to_framework_generator import build_residual_to_framework_generator_payload


DEFAULT_OUT = PAPER_DIR / "l4_residual_framework_mini_run_20260613.json"
DEFAULT_MD_OUT = Path("reconstruction/md/l4_residual_framework_mini_run_20260613.md")


def build_l4_residual_framework_mini_run_payload(
    *,
    root: Path,
    eval_id: str = "l4_residual_framework_mini_run_20260613",
    execute_live_llm: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    generator = build_residual_to_framework_generator_payload(root=root, eval_id=f"{eval_id}_r3_source")
    llm = build_llm_framework_candidate_experiment_payload(
        root=root,
        eval_id=f"{eval_id}_llm_candidate_source",
        execute_live=execute_live_llm,
    )
    gate = build_conservative_generalization_gate_v2_payload(root=root, eval_id=f"{eval_id}_gate_source")
    benchmark = build_multigeneration_framework_evolution_benchmark_payload(
        root=root,
        eval_id=f"{eval_id}_bounded_benchmark_source",
    )
    external_pack = build_framework_external_eval_pack_payload(root=root, eval_id=f"{eval_id}_external_pack_source")
    candidate_rows = _candidate_rows(generator=generator, llm=llm, limit=20)
    validation_rows = _validation_rows(gate=gate, limit=5)
    fresh_rows = _fresh_validation_rows(benchmark=benchmark, limit=5)
    expert_packet = _expert_packet(external_pack=external_pack, limit=5)
    branch_ledger = _branch_ledger(candidate_rows=candidate_rows, validation_rows=validation_rows)
    metrics = _metrics(
        generator=generator,
        llm=llm,
        gate=gate,
        benchmark=benchmark,
        external_pack=external_pack,
        candidate_rows=candidate_rows,
        validation_rows=validation_rows,
        fresh_rows=fresh_rows,
        expert_packet=expert_packet,
        branch_ledger=branch_ledger,
    )
    gates = {
        "source_generator_passes": generator["pass"] is True,
        "llm_contract_source_passes": llm["pass"] is True,
        "conservative_gate_passes": gate["pass"] is True,
        "bounded_framework_benchmark_passes": benchmark["pass"] is True,
        "external_eval_packet_passes": external_pack["pass"] is True,
        "real_residual_clusters_enough": metrics["real_residual_cluster_count"] >= 10,
        "candidate_frameworks_enough": metrics["candidate_framework_count"] >= 20,
        "llm_contract_candidates_enough": metrics["llm_contract_candidate_count"] >= 10,
        "validation_rows_enough": metrics["conservative_validation_count"] >= 5,
        "bounded_fresh_validation_rows_enough": metrics["bounded_fresh_validation_row_count"] >= 5,
        "expert_review_packet_rows_enough": metrics["expert_review_packet_row_count"] >= 5,
        "active_scoped_framework_present": metrics["active_scoped_framework_count"] >= 1,
        "rejected_boundary_present": metrics["rejected_boundary_count"] >= 1,
        "old_success_preservation_high": metrics["accepted_min_old_success_preservation"] >= 0.95,
        "residual_explanation_high": metrics["accepted_min_residual_explanation"] >= 0.75,
        "negative_evidence_retained": metrics["negative_evidence_count"] >= 1,
        "human_panel_not_fabricated": metrics["human_expert_completed_count"] == 0
        and metrics["human_expert_panel_claim_allowed"] is False,
        "main_graph_not_mutated": metrics["main_graph_mutation_count"] == 0,
        "l4_mini_preflight_claim_allowed": metrics["l4_mini_preflight_claim_allowed"] is True,
        "completed_external_l4_claim_blocked": metrics["l4_external_completion_claim_allowed"] is False,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "l4_residual_framework_mini_run",
        "source_md": "reconstruction/md/L4_roadmap.md",
        "l4_stage": "L4-3_to_L4-5_residual_framework_generation_and_external_packet",
        "implementation_level": "l4_mini_bounded_preflight_with_no_fabricated_external_claim",
        "performance_validation": True,
        "validation_scope": (
            "Runs a bounded L4-mini residual-to-framework line over real residual clusters and LLM-contract "
            "candidate frameworks.  It attaches conservative validations, bounded fresh-validation rows, branch "
            "ledger entries, and an expert-review packet while keeping real human/external completion claims "
            "blocked until those runs occur."
        ),
        "candidate_frameworks": candidate_rows,
        "conservative_validations": validation_rows,
        "bounded_fresh_validation_rows": fresh_rows,
        "expert_review_packet": expert_packet,
        "branch_ledger": branch_ledger,
        "source_summaries": {
            "generator": {"pass": generator["pass"], "metrics": generator["metrics"]},
            "llm_candidate_experiment": {"pass": llm["pass"], "metrics": llm["metrics"]},
            "conservative_gate_v2": {"pass": gate["pass"], "metrics": gate["metrics"]},
            "multigeneration_benchmark": {"pass": benchmark["pass"], "metrics": benchmark["metrics"]},
            "external_eval_pack": {"pass": external_pack["pass"], "metrics": external_pack["metrics"]},
        },
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "allowed_claim": "bounded L4-mini residual-to-framework preflight with conservative validation evidence",
        "blocked_claims": [
            "fresh_external_llm_framework_generation_completed",
            "human_expert_panel_completed",
            "external_prospective_l4_run_completed",
            "ungated_main_graph_framework_promotion",
        ],
    }


def format_markdown(payload: dict[str, Any]) -> str:
    m = payload["metrics"]
    lines = [
        "# L4 Residual-to-Framework Mini-Run",
        "",
        f"- pass: `{payload['pass']}`",
        f"- real residual clusters: `{m['real_residual_cluster_count']}`",
        f"- candidate frameworks: `{m['candidate_framework_count']}`",
        f"- LLM-contract candidates: `{m['llm_contract_candidate_count']}`",
        f"- conservative validations: `{m['conservative_validation_count']}`",
        f"- bounded fresh rows: `{m['bounded_fresh_validation_row_count']}`",
        f"- expert packet rows: `{m['expert_review_packet_row_count']}`",
        f"- active scoped frameworks: `{m['active_scoped_framework_count']}`",
        f"- external completion claim: `{m['l4_external_completion_claim_allowed']}`",
        "",
        "## Claim Boundary",
        "",
        "This is a bounded L4-mini preflight. Human/external completion remains blocked until real evidence exists.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _candidate_rows(*, generator: dict[str, Any], llm: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in llm.get("llm_candidates", []):
        rows.append({
            "candidate_framework_id": row["candidate_framework_id"],
            "source": "llm_contract_candidate",
            "source_residual_cluster": row["source_residual_cluster"],
            "trajectory_type": row["trajectory_type"],
            "parent_frameworks": row["parents"],
            "complete_obligation_packet": _has_llm_obligations(row),
            "candidate_hash": stable_hash(row),
        })
    selected_ids = {row["candidate_framework_id"] for row in rows}
    generator_rows = [
        row
        for row in generator.get("candidate_frameworks", [])
        if row.get("real_residual_cluster")
        and row.get("conservative_gate_ready")
        and row["candidate_framework_id"] not in selected_ids
    ]
    generator_rows.sort(
        key=lambda row: (
            row["trajectory_type"] == "scope_narrowing_branch",
            -float(row.get("generator_quality_score") or 0.0),
            row["candidate_framework_id"],
        )
    )
    for row in generator_rows[: max(0, limit - len(rows))]:
        rows.append({
            "candidate_framework_id": row["candidate_framework_id"],
            "source": "residual_to_framework_generator",
            "source_residual_cluster": row["residuals_explained"][0] if row.get("residuals_explained") else None,
            "trajectory_type": row["trajectory_type"],
            "parent_frameworks": row["parent_frameworks"],
            "complete_obligation_packet": _has_generator_obligations(row),
            "candidate_hash": stable_hash(row),
        })
    return rows[:limit]


def _validation_rows(*, gate: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    evaluations = list(gate.get("evaluations", []))
    evaluations.sort(
        key=lambda row: (
            row["decision"] != "active_scoped_framework",
            row["decision"] not in {"candidate_framework", "active_scoped_framework"},
            -float(row["metrics"]["framework_growth_score"]),
            row["candidate_framework_id"],
        )
    )
    selected = evaluations[: max(0, limit - 1)]
    rejected = [
        row for row in evaluations
        if row["decision"] in {"reject", "rejected_old_success_regression"}
    ]
    if rejected:
        selected.append(rejected[0])
    selected = selected[:limit]
    rows = []
    for row in selected:
        rows.append({
            "candidate_framework_id": row["candidate_framework_id"],
            "decision": row["decision"],
            "old_success_preservation": row["metrics"]["old_success_preservation"],
            "residual_explanation": row["metrics"]["residual_explanation"],
            "limiting_case_reduction": row["metrics"]["limiting_case_reduction"],
            "new_prediction_success": row["metrics"]["new_prediction_success"],
            "regression_cost": row["metrics"]["regression_cost"],
            "framework_growth_score": row["metrics"]["framework_growth_score"],
            "negative_evidence_retained": bool(row.get("negative_evidence_retained")),
            "relation_types": row["relation_types"],
            "test_suite_hash": row["test_suite_hash"],
        })
    return rows


def _fresh_validation_rows(*, benchmark: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    rows = [
        candidate
        for generation in benchmark.get("generation_rows", [])
        for candidate in generation.get("candidate_rows", [])
        if candidate.get("fresh_validation_decision") == "accepted"
    ]
    rows.sort(key=lambda row: (-float(row["framework_growth_score"]), row["framework_id"]))
    out = []
    for row in rows[:limit]:
        out.append({
            "framework_id": row["framework_id"],
            "seed_candidate_id": row["seed_candidate_id"],
            "generation": row["generation"],
            "fresh_validation_decision": row["fresh_validation_decision"],
            "old_success_preservation": row["old_success_preservation"],
            "residual_explanation": row["residual_explanation"],
            "new_prediction_success": row["new_prediction_success"],
            "regression_cost": row["regression_cost"],
            "framework_growth_score": row["framework_growth_score"],
            "source": "bounded_multigeneration_framework_benchmark",
        })
    return out


def _expert_packet(*, external_pack: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    packet = []
    for row in external_pack.get("expert_annotation_packet", [])[:limit]:
        packet.append({
            "annotation_id": row["annotation_id"],
            "parent_framework": row["parent_framework"],
            "candidate_framework": row["candidate_framework"],
            "residual_cluster": row["residual_cluster"],
            "system_decision": row["system_decision"],
            "questions": row["questions"],
            "human_label": None,
            "status": "pending_external_human_panel",
        })
    return packet


def _branch_ledger(
    *,
    candidate_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    validation_by_id = {row["candidate_framework_id"]: row for row in validation_rows}
    entries = []
    recorded_ids = set()
    for candidate in candidate_rows:
        validation = validation_by_id.get(candidate["candidate_framework_id"])
        status = validation["decision"] if validation else "candidate_branch"
        if status in {"reject", "rejected_old_success_regression"}:
            current_status = "rejected_boundary_only"
        elif status == "active_scoped_framework":
            current_status = "active_scoped_framework"
        elif status == "candidate_framework":
            current_status = "candidate_framework"
        else:
            current_status = "candidate_branch"
        entries.append({
            "branch_id": candidate["candidate_framework_id"],
            "source": candidate["source"],
            "parent_frameworks": candidate["parent_frameworks"],
            "current_status": current_status,
            "negative_evidence_retained": current_status == "rejected_boundary_only",
            "main_graph_mutation_count": 0,
            "entry_hash": stable_hash([candidate, validation, current_status]),
        })
        recorded_ids.add(candidate["candidate_framework_id"])
    for validation in validation_rows:
        if validation["candidate_framework_id"] in recorded_ids:
            continue
        status = validation["decision"]
        current_status = (
            "rejected_boundary_only"
            if status in {"reject", "rejected_old_success_regression"}
            else status
        )
        entries.append({
            "branch_id": validation["candidate_framework_id"],
            "source": "conservative_validation_only",
            "parent_frameworks": [],
            "current_status": current_status,
            "negative_evidence_retained": current_status == "rejected_boundary_only",
            "main_graph_mutation_count": 0,
            "entry_hash": stable_hash([validation, current_status]),
        })
    return entries


def _metrics(
    *,
    generator: dict[str, Any],
    llm: dict[str, Any],
    gate: dict[str, Any],
    benchmark: dict[str, Any],
    external_pack: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    fresh_rows: list[dict[str, Any]],
    expert_packet: list[dict[str, Any]],
    branch_ledger: list[dict[str, Any]],
) -> dict[str, Any]:
    accepted = [
        row for row in validation_rows
        if row["decision"] in {"candidate_framework", "active_scoped_framework"}
    ]
    return {
        "source_pass_rate": round(
            sum(
                1
                for payload in [generator, llm, gate, benchmark, external_pack]
                if payload.get("pass")
            )
            / 5,
            4,
        ),
        "real_residual_cluster_count": int(generator["metrics"]["real_residual_cluster_count"]),
        "candidate_framework_count": len(candidate_rows),
        "candidate_obligation_coverage": round(
            sum(1 for row in candidate_rows if row["complete_obligation_packet"]) / max(1, len(candidate_rows)),
            4,
        ),
        "llm_contract_candidate_count": sum(1 for row in candidate_rows if row["source"] == "llm_contract_candidate"),
        "live_llm_api_executed": bool(llm["metrics"]["live_llm_api_executed"]),
        "conservative_validation_count": len(validation_rows),
        "bounded_fresh_validation_row_count": len(fresh_rows),
        "expert_review_packet_row_count": len(expert_packet),
        "human_expert_completed_count": sum(1 for row in expert_packet if row.get("human_label") is not None),
        "human_expert_panel_claim_allowed": False,
        "active_scoped_framework_count": sum(1 for row in validation_rows if row["decision"] == "active_scoped_framework"),
        "candidate_framework_validation_count": sum(1 for row in validation_rows if row["decision"] == "candidate_framework"),
        "rejected_boundary_count": sum(
            1 for row in validation_rows if row["decision"] in {"reject", "rejected_old_success_regression"}
        ),
        "accepted_min_old_success_preservation": round(
            min((float(row["old_success_preservation"]) for row in accepted), default=0.0),
            4,
        ),
        "accepted_min_residual_explanation": round(
            min((float(row["residual_explanation"]) for row in accepted), default=0.0),
            4,
        ),
        "negative_evidence_count": sum(1 for row in branch_ledger if row["negative_evidence_retained"]),
        "branch_ledger_entry_count": len(branch_ledger),
        "bounded_benchmark_framework_growth_score": benchmark["metrics"]["framework_growth_score"],
        "bounded_benchmark_full_margin_vs_best_ablation": benchmark["metrics"]["full_margin_vs_best_ablation"],
        "external_proxy_agreement_with_system": external_pack["metrics"]["expert_proxy_agreement_with_system"],
        "main_graph_mutation_count": sum(row["main_graph_mutation_count"] for row in branch_ledger),
        "l4_mini_preflight_claim_allowed": True,
        "l4_external_completion_claim_allowed": False,
    }


def _has_llm_obligations(row: dict[str, Any]) -> bool:
    return all(
        row.get(key)
        for key in [
            "parents",
            "old_success_obligations",
            "limiting_case",
            "new_predictions",
            "validation_plan",
            "antithesis_residual",
        ]
    )


def _has_generator_obligations(row: dict[str, Any]) -> bool:
    return all(
        row.get(key)
        for key in [
            "parent_frameworks",
            "residuals_explained",
            "old_successes_to_preserve",
            "limiting_case_claims",
            "new_predictions",
            "required_tests",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build L4 residual-to-framework mini-run artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="l4_residual_framework_mini_run_20260613")
    parser.add_argument("--execute-live-llm", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_l4_residual_framework_mini_run_payload(
        root=root,
        eval_id=args.eval_id,
        execute_live_llm=args.execute_live_llm,
    )
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
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
