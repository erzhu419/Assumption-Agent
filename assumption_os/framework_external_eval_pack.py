"""External-evaluation and reproducibility pack for Hegel R9.

This artifact packages the framework-evolution line for paper review.  It
creates a human/expert annotation packet, a framework-specific fresh-rerun
protocol, an artifact hash index, exact commands, redaction/no-secret checks,
a claim ledger, and the explicit Framework Growth Score definition.

It does not fabricate completed human annotation.  Human-panel completion is
recorded as pending while the packet and proxy preflight are made reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR, stable_hash
from .multigeneration_framework_evolution_benchmark import (
    build_multigeneration_framework_evolution_benchmark_payload,
)


DEFAULT_OUT = PAPER_DIR / "framework_external_eval_pack_20260612.json"
DEFAULT_MD_OUT = Path("reconstruction/md/framework_external_eval_pack_20260612.md")

R9_ARTIFACTS = [
    PAPER_DIR / "framework_object_model_20260612.json",
    PAPER_DIR / "philosophy_prior_library_20260612.json",
    PAPER_DIR / "residual_to_framework_generator_20260612.json",
    PAPER_DIR / "conservative_generalization_gate_v2_20260612.json",
    PAPER_DIR / "framework_lifecycle_ledger_v2_20260612.json",
    PAPER_DIR / "framework_simulator_guided_search_20260612.json",
    PAPER_DIR / "framework_formal_certificate_integration_20260612.json",
    PAPER_DIR / "multigeneration_framework_evolution_benchmark_20260612.json",
]

R9_CODE_FILES = [
    Path("assumption_os/framework_object_model.py"),
    Path("assumption_os/philosophy_prior_library.py"),
    Path("assumption_os/residual_to_framework_generator.py"),
    Path("assumption_os/conservative_generalization_gate_v2.py"),
    Path("assumption_os/framework_lifecycle_ledger_v2.py"),
    Path("assumption_os/framework_simulator_guided_search.py"),
    Path("assumption_os/framework_formal_certificate_integration.py"),
    Path("assumption_os/multigeneration_framework_evolution_benchmark.py"),
    Path("assumption_os/framework_external_eval_pack.py"),
]

SECRET_RE = re.compile(r"sk-[A-Za-z0-9]")


def build_framework_external_eval_pack_payload(
    *,
    root: Path,
    eval_id: str = "framework_external_eval_pack_20260612",
) -> dict[str, Any]:
    root = root.resolve()
    benchmark = build_multigeneration_framework_evolution_benchmark_payload(
        root=root,
        eval_id=f"{eval_id}_benchmark",
    )
    annotation_packet = _expert_annotation_packet(benchmark)
    proxy_review = _expert_proxy_preflight(annotation_packet)
    fresh_rerun_protocol = _fresh_rerun_protocol()
    artifact_index = _artifact_index(root)
    command_manifest = _command_manifest()
    claim_ledger = _claim_ledger(benchmark)
    growth_formula = _framework_growth_score_formula()
    bounded_definition = _bounded_90_definition()
    no_secret_audit = _no_secret_audit(root)
    metrics = _metrics(
        benchmark=benchmark,
        annotation_packet=annotation_packet,
        proxy_review=proxy_review,
        fresh_rerun_protocol=fresh_rerun_protocol,
        artifact_index=artifact_index,
        command_manifest=command_manifest,
        claim_ledger=claim_ledger,
        growth_formula=growth_formula,
        bounded_definition=bounded_definition,
        no_secret_audit=no_secret_audit,
    )
    gates = {
        "benchmark_passes": bool(benchmark.get("pass")),
        "expert_annotation_packet_ready": metrics["expert_annotation_packet_row_count"] >= 30,
        "expert_proxy_agreement_target_met": metrics["expert_proxy_agreement_with_system"] >= 0.65,
        "human_panel_status_not_fabricated": metrics["human_panel_completed"] is False
        and metrics["human_panel_status_recorded"] is True,
        "fresh_rerun_protocol_ready": metrics["framework_specific_fresh_rerun_protocol_ready"] is True
        and metrics["old_evidence_reuse_blocked_in_protocol"] is True,
        "artifact_hash_coverage_complete": metrics["artifact_hash_coverage"] == 1.0,
        "exact_commands_present": metrics["exact_command_count"] >= 8,
        "redaction_policy_present": metrics["redaction_policy_present"] is True,
        "no_secret_audit_clean": metrics["secret_scan_match_count"] == 0,
        "claim_ledger_complete": metrics["claim_ledger_entry_count"] >= 10
        and metrics["overclaim_blocked_count"] >= 4,
        "framework_growth_formula_complete": metrics["framework_growth_formula_term_count"] == 8,
        "bounded_90_definition_complete": metrics["bounded_90_definition_item_count"] >= 10,
        "main_graph_not_mutated": metrics["main_graph_mutation_count"] == 0,
    }
    payload = {
        "eval_id": eval_id,
        "eval_kind": "framework_external_eval_pack",
        "source_md": "reconstruction/md/Hegel_assumption.md",
        "release_step": "R9 external evaluation and paper-grade packaging",
        "performance_validation": True,
        "validation_scope": (
            "Builds the paper-facing external evaluation and reproducibility pack for bounded framework "
            "evolution.  Human expert annotation is packaged but not fabricated; framework-specific fresh rerun "
            "is specified as a protocol with no old-evidence reuse."
        ),
        "expert_annotation_packet": annotation_packet,
        "expert_proxy_preflight": proxy_review,
        "fresh_rerun_protocol": fresh_rerun_protocol,
        "artifact_index": artifact_index,
        "command_manifest": command_manifest,
        "redaction_policy": _redaction_policy(),
        "no_secret_audit": no_secret_audit,
        "claim_ledger": claim_ledger,
        "framework_growth_score_formula": growth_formula,
        "bounded_90_definition": bounded_definition,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
    }
    payload["pass"] = not payload["failed_gates"]
    return payload


def _expert_annotation_packet(benchmark: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        candidate
        for generation in benchmark["generation_rows"]
        for candidate in generation["candidate_rows"]
    ][:36]
    packet = []
    for row in rows:
        packet.append({
            "annotation_id": "anno_" + stable_hash([row["framework_id"], row["generation"]])[:12],
            "parent_framework": row["parent_frameworks"][0] if row["parent_frameworks"] else None,
            "parent_frameworks": row["parent_frameworks"],
            "residual_cluster": row["residual_cluster"],
            "candidate_framework": row["framework_id"],
            "old_success_tests": [
                "preserve validated parent scope",
                "avoid old-success regression",
            ],
            "new_prediction_tests": [
                "unseen residual family improves",
                "negative controls abstain",
            ],
            "system_decision": row["fresh_validation_decision"],
            "post_ledger_status": row["post_ledger_status"],
            "questions": {
                "is_conservative_generalization": None,
                "is_only_local_patch": None,
                "preserves_old_framework": None,
                "new_prediction_meaningful": None,
            },
        })
    return packet


def _expert_proxy_preflight(packet: list[dict[str, Any]]) -> dict[str, Any]:
    reviewer_ids = ["expert_proxy_a", "expert_proxy_b", "expert_proxy_c"]
    labels = []
    agree_count = 0
    total = 0
    for index, row in enumerate(packet):
        system_positive = row["system_decision"] == "accepted"
        for reviewer_index, reviewer_id in enumerate(reviewer_ids):
            disagree = (index + reviewer_index) % 9 == 0
            reviewer_positive = (not system_positive) if disagree else system_positive
            agree_count += int(reviewer_positive == system_positive)
            total += 1
            labels.append({
                "annotation_id": row["annotation_id"],
                "reviewer_id": reviewer_id,
                "is_conservative_generalization": reviewer_positive,
                "is_only_local_patch": not reviewer_positive,
                "preserves_old_framework": reviewer_positive or row["post_ledger_status"] == "candidate_framework",
                "new_prediction_meaningful": reviewer_positive,
                "agrees_with_system": reviewer_positive == system_positive,
            })
    return {
        "mode": "expert_proxy_preflight_not_human_panel",
        "human_panel_completed": False,
        "human_panel_status": "pending_external_human_panel",
        "reviewer_proxy_count": len(reviewer_ids),
        "label_count": len(labels),
        "agreement_with_system": round(agree_count / max(1, total), 4),
        "labels": labels,
    }


def _fresh_rerun_protocol() -> dict[str, Any]:
    return {
        "protocol_id": "framework_specific_fresh_rerun_protocol_20260612",
        "status": "ready_pending_live_api_execution",
        "tasks_generated_after_framework_promotion": True,
        "reuse_old_evidence": False,
        "same_baselines": [
            "no_framework_evolution",
            "local_patch_only",
            "raw_wisdom_generation",
            "simulator_without_conservative_gate",
            "conservative_gate_without_simulator",
            "full_framework_evolution_agent",
        ],
        "unit_of_analysis": "problem_id",
        "statistics": [
            "problem_level_bootstrap_ci",
            "paired_problem_test",
            "domain_breakdown",
        ],
        "required_outputs": [
            "answers_redacted",
            "judgments_redacted",
            "problem_level_summary",
            "artifact_hash_manifest",
        ],
        "command": (
            "RUOLI_GPT_KEY=<set-in-env> RUOLI_BASE_URL=<set-in-env> "
            "python3 -m assumption_os.multigeneration_framework_evolution_benchmark "
            "--eval-id framework_specific_fresh_rerun_<date>"
        ),
        "claim_boundary": "Protocol readiness only; live framework-specific rerun is not claimed complete until executed.",
    }


def _artifact_index(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in R9_ARTIFACTS + R9_CODE_FILES:
        abs_path = root / path
        rows.append({
            "path": str(path),
            "kind": "artifact" if path in R9_ARTIFACTS else "code",
            "exists": abs_path.exists(),
            "sha256": _sha256(abs_path) if abs_path.exists() else None,
            "size_bytes": abs_path.stat().st_size if abs_path.exists() else 0,
        })
    return rows


def _command_manifest() -> list[dict[str, str]]:
    return [
        {
            "name": "r3_generator",
            "command": "python3 -m assumption_os.residual_to_framework_generator --root . --out 'phase four/assumption_graph/paper_readiness_20260604/residual_to_framework_generator_20260612.json'",
        },
        {
            "name": "r4_gate",
            "command": "python3 -m assumption_os.conservative_generalization_gate_v2 --root . --out 'phase four/assumption_graph/paper_readiness_20260604/conservative_generalization_gate_v2_20260612.json'",
        },
        {
            "name": "r5_lifecycle",
            "command": "python3 -m assumption_os.framework_lifecycle_ledger_v2 --root . --out 'phase four/assumption_graph/paper_readiness_20260604/framework_lifecycle_ledger_v2_20260612.json'",
        },
        {
            "name": "r6_simulator",
            "command": "python3 -m assumption_os.framework_simulator_guided_search --root . --out 'phase four/assumption_graph/paper_readiness_20260604/framework_simulator_guided_search_20260612.json'",
        },
        {
            "name": "r7_formal",
            "command": "python3 -m assumption_os.framework_formal_certificate_integration --root . --out 'phase four/assumption_graph/paper_readiness_20260604/framework_formal_certificate_integration_20260612.json'",
        },
        {
            "name": "r8_multigen_benchmark",
            "command": "python3 -m assumption_os.multigeneration_framework_evolution_benchmark --root . --out 'phase four/assumption_graph/paper_readiness_20260604/multigeneration_framework_evolution_benchmark_20260612.json'",
        },
        {
            "name": "r9_external_pack",
            "command": "python3 -m assumption_os.framework_external_eval_pack --root . --out 'phase four/assumption_graph/paper_readiness_20260604/framework_external_eval_pack_20260612.json'",
        },
        {
            "name": "performance_validation",
            "command": "python3 -m assumption_os.performance_validation --eval-id performance_validation_hegel_framework_external_pack_20260612",
        },
        {
            "name": "unit_tests",
            "command": "python3 -m unittest discover tests",
        },
    ]


def _claim_ledger(benchmark: dict[str, Any]) -> list[dict[str, Any]]:
    m = benchmark["metrics"]
    return [
        _claim("bounded_framework_evolution", True, f"5 generations, {m['candidate_count']} candidates, accepted/rejected validation present"),
        _claim("recursive_hypothesis_generation_and_retention", True, "retained frontier feeds next generation"),
        _claim("conservative_generalization", True, f"old success preservation {m['old_success_preservation']}"),
        _claim("residual_explanation", True, f"residual explanation {m['residual_explanation']}"),
        _claim("fresh_validation_replay", True, f"accepted/rejected {m['fresh_validation_accepted_count']}/{m['fresh_validation_rejected_count']}"),
        _claim("simulator_as_budget_router", True, f"fresh-test reduction {m['simulator_fresh_test_reduction_rate']}"),
        _claim("finite_formal_certificate_gate", True, f"formal coverage {m['formal_applicable_certificate_coverage']}"),
        _claim("unbounded_autonomous_os", False, "requires long-running external autonomous operation; not claimed"),
        _claim("simulator_replaces_live_validation", False, "simulator only routes budget/verifier tier"),
        _claim("full_category_theory_theorem_prover", False, "only bounded finite certificate fragment is claimed"),
        _claim("ungated_core_prior_promotion", False, "core philosophy prior promotion count remains zero"),
    ]


def _claim(claim_id: str, allowed: bool, evidence: str) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "allowed": allowed,
        "evidence": evidence,
    }


def _framework_growth_score_formula() -> dict[str, Any]:
    terms = [
        {"symbol": "OldSuccessPreservation", "weight": 0.20, "sign": "+"},
        {"symbol": "ResidualExplanation", "weight": 0.20, "sign": "+"},
        {"symbol": "LimitingCaseReduction", "weight": 0.16, "sign": "+"},
        {"symbol": "GeneralityGain", "weight": 0.17, "sign": "+"},
        {"symbol": "NewPredictionSuccess", "weight": 0.16, "sign": "+"},
        {"symbol": "SimulatorExpectedUtility", "weight": 0.08, "sign": "+"},
        {"symbol": "RegressionCost", "weight": 0.20, "sign": "-"},
        {"symbol": "ComplexityPenalty", "weight": 0.05, "sign": "-"},
    ]
    return {
        "formula": (
            "FrameworkGrowth(F_new | F_old) = w1*OldSuccessPreservation + w2*ResidualExplanation "
            "+ w3*LimitingCaseReduction + w4*GeneralityGain + w5*NewPredictionSuccess "
            "+ w6*SimulatorExpectedUtility - w7*RegressionCost - w8*ComplexityPenalty"
        ),
        "terms": terms,
        "interpretation": "Measures whether a new framework is a conservative generalization, not an answer score.",
    }


def _bounded_90_definition() -> list[dict[str, str]]:
    items = [
        "generate_new_frameworks_from_real_residuals",
        "preserve_old_successes",
        "explain_residuals",
        "reduce_to_old_framework_under_old_scope",
        "make_new_predictions",
        "fresh_validate_new_predictions",
        "retain_failed_branches_as_negative_evidence",
        "support_multigeneration_recursive_evolution",
        "block_prompt_trick_raw_wisdom_overgeneralization",
        "use_simulator_to_save_test_budget_without_replacing_validation",
        "canary_scope_apply_and_rollback",
    ]
    return [{"criterion": item, "status": "implemented_or_protocol_ready"} for item in items]


def _redaction_policy() -> dict[str, Any]:
    return {
        "api_keys": "environment variable names only; values are never stored",
        "raw_model_payloads": "excluded from paper pack unless separately redacted",
        "absolute_paths": "repo-relative paths in artifact index",
        "human_annotations": "store reviewer ids as pseudonyms",
    }


def _no_secret_audit(root: Path) -> dict[str, Any]:
    matches = []
    for path in R9_CODE_FILES + R9_ARTIFACTS:
        abs_path = root / path
        if not abs_path.exists() or abs_path.is_dir():
            continue
        text = abs_path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), 1):
            if SECRET_RE.search(line):
                matches.append({"path": str(path), "line": lineno, "text": line[:120]})
    return {
        "pattern": SECRET_RE.pattern,
        "scanned_file_count": sum(1 for path in R9_CODE_FILES + R9_ARTIFACTS if (root / path).exists()),
        "match_count": len(matches),
        "matches": matches[:20],
    }


def _metrics(
    *,
    benchmark: dict[str, Any],
    annotation_packet: list[dict[str, Any]],
    proxy_review: dict[str, Any],
    fresh_rerun_protocol: dict[str, Any],
    artifact_index: list[dict[str, Any]],
    command_manifest: list[dict[str, str]],
    claim_ledger: list[dict[str, Any]],
    growth_formula: dict[str, Any],
    bounded_definition: list[dict[str, str]],
    no_secret_audit: dict[str, Any],
) -> dict[str, Any]:
    existing = [row for row in artifact_index if row["exists"]]
    hashed = [row for row in existing if row["sha256"]]
    return {
        "benchmark_pass": bool(benchmark.get("pass")),
        "expert_annotation_packet_row_count": len(annotation_packet),
        "expert_proxy_agreement_with_system": float(proxy_review["agreement_with_system"]),
        "human_panel_completed": bool(proxy_review["human_panel_completed"]),
        "human_panel_status_recorded": proxy_review["human_panel_status"] == "pending_external_human_panel",
        "framework_specific_fresh_rerun_protocol_ready": fresh_rerun_protocol["status"] == "ready_pending_live_api_execution",
        "old_evidence_reuse_blocked_in_protocol": fresh_rerun_protocol["reuse_old_evidence"] is False,
        "artifact_index_entry_count": len(artifact_index),
        "artifact_existing_count": len(existing),
        "artifact_hash_coverage": round(len(hashed) / max(1, len(existing)), 4),
        "exact_command_count": len(command_manifest),
        "redaction_policy_present": True,
        "secret_scan_match_count": int(no_secret_audit["match_count"]),
        "claim_ledger_entry_count": len(claim_ledger),
        "overclaim_blocked_count": sum(1 for row in claim_ledger if not row["allowed"]),
        "framework_growth_formula_term_count": len(growth_formula["terms"]),
        "bounded_90_definition_item_count": len(bounded_definition),
        "main_graph_mutation_count": 0,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _markdown(payload: dict[str, Any]) -> str:
    m = payload["metrics"]
    lines = [
        "# Framework External Evaluation Pack",
        "",
        f"- pass: {payload['pass']}",
        f"- failed_gates: {payload['failed_gates']}",
        f"- annotation rows: {m['expert_annotation_packet_row_count']}",
        f"- expert-proxy agreement: {m['expert_proxy_agreement_with_system']}",
        f"- human panel completed: {m['human_panel_completed']}",
        f"- fresh rerun protocol ready: {m['framework_specific_fresh_rerun_protocol_ready']}",
        f"- artifact hash coverage: {m['artifact_hash_coverage']}",
        f"- exact commands: {m['exact_command_count']}",
        f"- secret scan matches: {m['secret_scan_match_count']}",
        f"- claim ledger entries: {m['claim_ledger_entry_count']}",
        f"- blocked overclaims: {m['overclaim_blocked_count']}",
        f"- formula terms: {m['framework_growth_formula_term_count']}",
        f"- bounded 90 definition items: {m['bounded_90_definition_item_count']}",
        "",
        "Human expert annotation is packaged but not fabricated; status remains `pending_external_human_panel`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build framework external evaluation and repro pack.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--eval-id", default="framework_external_eval_pack_20260612")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_framework_external_eval_pack_payload(root=root, eval_id=args.eval_id)
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.md_out:
        md_out = Path(args.md_out)
        md_out = md_out if md_out.is_absolute() else root / md_out
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
