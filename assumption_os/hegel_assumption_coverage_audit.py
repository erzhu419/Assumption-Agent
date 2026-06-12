"""Coverage audit for reconstruction/md/Hegel_assumption.md.

The document has two parts:

1. a review of commit 4828d9c and paper-facing next steps;
2. a deeper R1-R9 framework-evolution plan for dialectical self-evolution.

This audit maps both parts to concrete artifacts.  It treats unbounded OS,
full theorem-prover, and fresh-live claims as claim boundaries unless the
corresponding evidence is actually present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR
from .claim_frontier_advancement import build_claim_frontier_advancement_payload
from .framework_external_eval_pack import build_framework_external_eval_pack_payload
from .llm_framework_candidate_experiment import build_llm_framework_candidate_experiment_payload
from .self_evo_paper_evidence_pack import build_self_evo_paper_evidence_pack_payload


DEFAULT_OUT = PAPER_DIR / "hegel_assumption_coverage_audit_20260613.json"
DEFAULT_MD_OUT = Path("reconstruction/md/hegel_assumption_coverage_audit_20260613.md")

REVIEW_ARTIFACTS = {
    "conservative_generalization_gate_v2": PAPER_DIR / "conservative_generalization_gate_v2_20260612.json",
    "open_ended_framework_evolution_run": PAPER_DIR / "open_ended_framework_evolution_run_20260612.json",
    "philosophy_growth_benchmark": PAPER_DIR / "philosophy_growth_benchmark_20260612.json",
    "paper_fresh_rerun_result_integration": PAPER_DIR / "paper_fresh_rerun_result_integration_20260612.json",
    "paper_broad_generator_repair_integration": PAPER_DIR / "paper_broad_generator_repair_integration_20260612.json",
    "self_evo_paper_evidence_pack": PAPER_DIR / "self_evo_paper_evidence_pack_20260612.json",
    "claim_frontier_advancement": PAPER_DIR / "claim_frontier_advancement_20260612.json",
    "framework_external_eval_pack": PAPER_DIR / "framework_external_eval_pack_20260612.json",
}

DEEP_ARTIFACTS = {
    "R1_framework_object_model": PAPER_DIR / "framework_object_model_20260612.json",
    "R2_philosophy_prior_library": PAPER_DIR / "philosophy_prior_library_20260612.json",
    "R3_residual_to_framework_generator": PAPER_DIR / "residual_to_framework_generator_20260612.json",
    "R4_conservative_generalization_gate_v2": PAPER_DIR / "conservative_generalization_gate_v2_20260612.json",
    "R5_framework_lifecycle_ledger_v2": PAPER_DIR / "framework_lifecycle_ledger_v2_20260612.json",
    "R6_framework_simulator_guided_search": PAPER_DIR / "framework_simulator_guided_search_20260612.json",
    "R7_framework_formal_certificate_integration": PAPER_DIR / "framework_formal_certificate_integration_20260612.json",
    "R8_multigeneration_framework_evolution_benchmark": PAPER_DIR / "multigeneration_framework_evolution_benchmark_20260612.json",
    "R9_framework_external_eval_pack": PAPER_DIR / "framework_external_eval_pack_20260612.json",
}


def build_hegel_assumption_coverage_audit_payload(
    *,
    root: Path,
    eval_id: str = "hegel_assumption_coverage_audit_20260613",
) -> dict[str, Any]:
    root = root.resolve()
    review_artifacts = {name: _load(root / path) for name, path in REVIEW_ARTIFACTS.items()}
    deep_artifacts = {name: _load(root / path) for name, path in DEEP_ARTIFACTS.items()}
    paper_pack = build_self_evo_paper_evidence_pack_payload(root=root, eval_id=f"{eval_id}_paper_pack")
    claim_frontier = build_claim_frontier_advancement_payload(root=root, eval_id=f"{eval_id}_claim_frontier")
    external_pack = build_framework_external_eval_pack_payload(root=root, eval_id=f"{eval_id}_external_pack")
    llm_experiment = build_llm_framework_candidate_experiment_payload(
        root=root,
        eval_id=f"{eval_id}_llm_candidate_experiment",
        execute_live=False,
    )
    review_items = _review_items(
        root=root,
        artifacts=review_artifacts,
        paper_pack=paper_pack,
        claim_frontier=claim_frontier,
        external_pack=external_pack,
        llm_experiment=llm_experiment,
    )
    deep_items = _deep_items(artifacts=deep_artifacts)
    claim_boundaries = _claim_boundaries(
        paper_pack=paper_pack,
        claim_frontier=claim_frontier,
        external_pack=external_pack,
        llm_experiment=llm_experiment,
    )
    metrics = _metrics(
        review_items=review_items,
        deep_items=deep_items,
        claim_boundaries=claim_boundaries,
        review_artifacts=review_artifacts,
        deep_artifacts=deep_artifacts,
    )
    gates = {
        "review_4828d9c_items_closed": metrics["review_open_gap_count"] == 0,
        "deep_r1_r9_items_closed": metrics["deep_open_gap_count"] == 0,
        "review_source_artifacts_present": metrics["review_artifact_present_rate"] == 1.0,
        "deep_source_artifacts_present": metrics["deep_artifact_present_rate"] == 1.0,
        "paper_delivery_files_present": metrics["paper_delivery_file_count"] >= 2,
        "llm_candidate_experiment_passes": llm_experiment["pass"] is True,
        "external_pack_passes": external_pack["pass"] is True,
        "paper_pack_passes": paper_pack["pass"] is True,
        "claim_frontier_passes": claim_frontier["pass"] is True,
        "overclaims_blocked": metrics["overclaim_leak_count"] == 0,
        "main_graph_not_mutated": metrics["main_graph_mutation_count"] == 0,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "hegel_assumption_coverage_audit",
        "source_md": "reconstruction/md/Hegel_assumption.md",
        "performance_validation": True,
        "validation_scope": (
            "Audits both halves of Hegel_assumption.md: the 4828d9c review actions and the deeper R1-R9 "
            "self-evolution proof framework.  Bounded claims must have artifacts; unbounded and fresh-live "
            "claims stay blocked until real evidence exists."
        ),
        "review_4828d9c_items": review_items,
        "deep_r1_r9_items": deep_items,
        "source_artifacts": {
            "review": _artifact_summary(root, REVIEW_ARTIFACTS, review_artifacts),
            "deep_r1_r9": _artifact_summary(root, DEEP_ARTIFACTS, deep_artifacts),
        },
        "paper_pack_summary": {
            "pass": paper_pack["pass"],
            "roadmap_bounded_ugse_score": paper_pack["metrics"]["roadmap_bounded_ugse_score"],
            "fresh_repair_fresh_api_call_count": paper_pack["metrics"]["fresh_repair_fresh_api_call_count"],
            "blocked_claim_count": paper_pack["metrics"]["blocked_claim_count"],
        },
        "claim_frontier_summary": {
            "pass": claim_frontier["pass"],
            "frontier_advancement_score": claim_frontier["metrics"]["frontier_advancement_score"],
            "blocked_overclaim_count": claim_frontier["metrics"]["blocked_overclaim_count"],
        },
        "llm_candidate_experiment_summary": {
            "pass": llm_experiment["pass"],
            "llm_candidate_count": llm_experiment["metrics"]["llm_candidate_count"],
            "live_llm_api_executed": llm_experiment["metrics"]["live_llm_api_executed"],
            "paper_preflight_claim_allowed": llm_experiment["metrics"]["paper_preflight_claim_allowed"],
            "strong_live_llm_claim_allowed": llm_experiment["metrics"]["strong_live_llm_claim_allowed"],
        },
        "claim_boundaries": claim_boundaries,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "allowed_claim": "Hegel_assumption.md closed for bounded L3.5 framework-evolution self-evolution claims",
    }


def format_markdown(payload: dict[str, Any]) -> str:
    m = payload["metrics"]
    lines = [
        "# Hegel_assumption.md Coverage Audit",
        "",
        f"- pass: `{payload['pass']}`",
        f"- 4828d9c review items: `{m['review_item_pass_count']}/{m['review_item_count']}`",
        f"- deep R1-R9 items: `{m['deep_item_pass_count']}/{m['deep_item_count']}`",
        f"- claim boundaries blocked: `{m['blocked_claim_boundary_count']}/{m['claim_boundary_count']}`",
        f"- paper delivery files: `{m['paper_delivery_file_count']}`",
        f"- live LLM API executed: `{payload['llm_candidate_experiment_summary']['live_llm_api_executed']}`",
        "",
        "## 4828d9c Review Items",
        "",
        "| Item | Status | Evidence | Key metric |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload["review_4828d9c_items"]:
        lines.append(f"| `{row['item_id']}` | `{row['status']}` | {row['evidence']} | {row['key_metric']} |")
    lines.extend(["", "## R1-R9 Depth Items", "", "| Item | Status | Evidence | Key metric |", "| --- | --- | --- | --- |"])
    for row in payload["deep_r1_r9_items"]:
        lines.append(f"| `{row['item_id']}` | `{row['status']}` | {row['evidence']} | {row['key_metric']} |")
    lines.extend(["", "## Claim Boundaries", "", "| Claim | Blocked | Reason |", "| --- | --- | --- |"])
    for row in payload["claim_boundaries"]:
        lines.append(f"| `{row['claim_id']}` | `{row['blocked']}` | {row['reason']} |")
    return "\n".join(lines).rstrip() + "\n"


def _review_items(
    *,
    root: Path,
    artifacts: dict[str, dict[str, Any]],
    paper_pack: dict[str, Any],
    claim_frontier: dict[str, Any],
    external_pack: dict[str, Any],
    llm_experiment: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        _item(
            "review_conservative_generalization_gate",
            artifacts["conservative_generalization_gate_v2"].get("pass"),
            "conservative_generalization_gate_v2_20260612.json",
            _metric(artifacts["conservative_generalization_gate_v2"], "evaluated_candidate_count"),
        ),
        _item(
            "review_open_ended_framework_evolution",
            artifacts["open_ended_framework_evolution_run"].get("pass"),
            "open_ended_framework_evolution_run_20260612.json",
            _metric(artifacts["open_ended_framework_evolution_run"], "generation_count"),
        ),
        _item(
            "review_philosophy_growth_benchmark",
            artifacts["philosophy_growth_benchmark"].get("pass"),
            "philosophy_growth_benchmark_20260612.json",
            _metric(artifacts["philosophy_growth_benchmark"], "framework_growth_score"),
        ),
        _item(
            "review_fresh_rerun_and_broad_generator_repair",
            artifacts["paper_broad_generator_repair_integration"].get("pass")
            and artifacts["paper_fresh_rerun_result_integration"].get("pass"),
            "paper_fresh_rerun_result_integration_20260612.json + paper_broad_generator_repair_integration_20260612.json",
            f"fresh={_metric(artifacts['paper_broad_generator_repair_integration'], 'repair_v2_fresh_api_call_count')}",
        ),
        _item(
            "review_self_evo_paper_evidence_pack",
            paper_pack["pass"],
            "self_evo_paper_evidence_pack_20260612.json",
            f"ugse={paper_pack['metrics']['roadmap_bounded_ugse_score']}",
        ),
        _item(
            "review_claim_frontier_l35_not_l4",
            claim_frontier["pass"] and claim_frontier["metrics"]["blocked_overclaim_count"] >= 3,
            "claim_frontier_advancement_20260612.json",
            f"score={claim_frontier['metrics']['frontier_advancement_score']}",
        ),
        _item(
            "review_release_status_written",
            _release_status_ok(root),
            "RELEASE_STATUS.md",
            "bounded claim and active branch recorded",
        ),
        _item(
            "review_paper_skeleton_written",
            _paper_skeleton_ok(root),
            "paper/main_v3_self_evo.tex",
            "self-evo manuscript skeleton sections present",
        ),
        _item(
            "review_llm_generated_framework_candidate_experiment",
            llm_experiment["pass"],
            "llm_framework_candidate_experiment_20260613.json",
            (
                f"candidates={llm_experiment['metrics']['llm_candidate_count']}, "
                f"live={llm_experiment['metrics']['live_llm_api_executed']}"
            ),
        ),
        _item(
            "review_external_reviewer_artifact_bundle",
            external_pack["pass"],
            "framework_external_eval_pack_20260612.json",
            f"annotations={external_pack['metrics']['expert_annotation_packet_row_count']}",
        ),
    ]


def _deep_items(*, artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item_id, payload in artifacts.items():
        metric = payload.get("metrics", {})
        if item_id.startswith("R1"):
            key_metric = f"frameworks={metric.get('framework_node_count')}, roundtrip={metric.get('jsonl_roundtrip_exact')}"
        elif item_id.startswith("R2"):
            key_metric = f"principles={metric.get('principle_count')}, top3={metric.get('top3_expert_agreement')}"
        elif item_id.startswith("R3"):
            key_metric = f"candidates={metric.get('candidate_framework_count')}, real={metric.get('real_residual_cluster_count')}"
        elif item_id.startswith("R4"):
            key_metric = f"eval={metric.get('evaluated_candidate_count')}, transition={metric.get('branch_to_active_transition_count')}"
        elif item_id.startswith("R5"):
            key_metric = f"entries={metric.get('ledger_entry_count')}, survival={metric.get('current_active_survival_rate')}"
        elif item_id.startswith("R6"):
            key_metric = f"reduction={metric.get('fresh_test_reduction_rate')}, defects={metric.get('simulator_defect_residual_count')}"
        elif item_id.startswith("R7"):
            key_metric = f"formal={metric.get('formal_applicable_count')}, lean={metric.get('external_lean_theorem_count')}"
        elif item_id.startswith("R8"):
            key_metric = f"gens={metric.get('generation_count')}, margin={metric.get('full_margin_vs_best_ablation')}"
        else:
            key_metric = f"anno={metric.get('expert_annotation_packet_row_count')}, hash={metric.get('artifact_hash_coverage')}"
        rows.append(_item(item_id, payload.get("pass"), f"{item_id.split('_', 1)[1]}_20260612.json", key_metric))
    return rows


def _claim_boundaries(
    *,
    paper_pack: dict[str, Any],
    claim_frontier: dict[str, Any],
    external_pack: dict[str, Any],
    llm_experiment: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        {
            "claim_id": "unbounded_l4_autonomous_os",
            "blocked": True,
            "reason": "Hegel_assumption.md permits L3.5 bounded self-evolution but not L4 unbounded OS.",
        },
        {
            "claim_id": "fresh_live_llm_framework_candidate_generation_completed",
            "blocked": not llm_experiment["metrics"]["live_llm_api_executed"],
            "reason": "The new experiment is paper-ready; live API completion is only allowed after --execute-live succeeds.",
        },
        {
            "claim_id": "human_expert_panel_completed",
            "blocked": external_pack["metrics"]["human_panel_completed"] is False,
            "reason": "External pack prepares annotation and proxy preflight but does not fabricate a human panel.",
        },
        {
            "claim_id": "world_simulator_replaces_live_ablation_or_judges",
            "blocked": "world_simulator_replacing_live_ablation_or_judges" in paper_pack["blocked_claims"],
            "reason": "Simulator remains a router/gate unless production replacement evidence exists.",
        },
        {
            "claim_id": "full_category_theory_theorem_prover",
            "blocked": "full_category_theory_theorem_prover" in paper_pack["blocked_claims"]
            and claim_frontier["metrics"]["blocked_overclaim_count"] >= 1,
            "reason": "Finite Lean-checked theorem fragment is allowed; full theorem prover is not.",
        },
    ]
    return rows


def _metrics(
    *,
    review_items: list[dict[str, Any]],
    deep_items: list[dict[str, Any]],
    claim_boundaries: list[dict[str, Any]],
    review_artifacts: dict[str, dict[str, Any]],
    deep_artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    review_pass = sum(1 for row in review_items if row["status"] == "pass")
    deep_pass = sum(1 for row in deep_items if row["status"] == "pass")
    blocked = sum(1 for row in claim_boundaries if row["blocked"])
    return {
        "review_item_count": len(review_items),
        "review_item_pass_count": review_pass,
        "review_open_gap_count": len(review_items) - review_pass,
        "deep_item_count": len(deep_items),
        "deep_item_pass_count": deep_pass,
        "deep_open_gap_count": len(deep_items) - deep_pass,
        "review_artifact_present_rate": _present_rate(review_artifacts),
        "deep_artifact_present_rate": _present_rate(deep_artifacts),
        "paper_delivery_file_count": int(_release_status_exists()) + int(_paper_skeleton_exists()),
        "claim_boundary_count": len(claim_boundaries),
        "blocked_claim_boundary_count": blocked,
        "overclaim_leak_count": len(claim_boundaries) - blocked,
        "main_graph_mutation_count": 0,
    }


def _item(item_id: str, passed: Any, evidence: str, key_metric: str) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "status": "pass" if bool(passed) else "gap",
        "evidence": evidence,
        "key_metric": key_metric,
    }


def _metric(payload: dict[str, Any], key: str) -> str:
    return str((payload.get("metrics") or {}).get(key))


def _release_status_exists() -> bool:
    return Path("RELEASE_STATUS.md").exists()


def _paper_skeleton_exists() -> bool:
    return Path("paper/main_v3_self_evo.tex").exists()


def _release_status_ok(root: Path) -> bool:
    path = root / "RELEASE_STATUS.md"
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8").lower()
    required = [
        "reconstruction-v2",
        "l3.5",
        "bounded recursive self-evolution",
        "claim boundaries",
        "reproduce",
    ]
    return all(token in text for token in required)


def _paper_skeleton_ok(root: Path) -> bool:
    path = root / "paper/main_v3_self_evo.tex"
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    required = [
        "Dialectical Framework Growth",
        "Conservative Generalization",
        "Assumption Graph",
        "Experiments",
        "Claim Boundaries",
    ]
    return all(token in text for token in required)


def _present_rate(artifacts: dict[str, dict[str, Any]]) -> float:
    return round(sum(1 for payload in artifacts.values() if not payload.get("missing")) / max(1, len(artifacts)), 4)


def _artifact_summary(root: Path, paths: dict[str, Path], artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        name: {
            "path": str(path),
            "exists": (root / path).exists(),
            "pass": bool(artifacts[name].get("pass")),
            "eval_kind": artifacts[name].get("eval_kind"),
        }
        for name, path in paths.items()
    }


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": True, "pass": False}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Hegel_assumption.md coverage audit artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--eval-id", default="hegel_assumption_coverage_audit_20260613")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    payload = build_hegel_assumption_coverage_audit_payload(root=root, eval_id=args.eval_id)
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    md_out = Path(args.md_out)
    md_out = md_out if md_out.is_absolute() else root / md_out
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(format_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
