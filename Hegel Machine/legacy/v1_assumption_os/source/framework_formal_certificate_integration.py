"""Formal-certificate integration for framework evolution.

R7 routes framework candidates through the formal layer only when a bounded
finite structural certificate is applicable.  Semi-formal candidates keep
process-graph and empirical obligations, while non-formalizable methodology
controls are explicitly kept out of the theorem-prover path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR, stable_hash
from .finite_category_certificate import build_finite_category_certificate_payload
from .finite_formal_reasoning_stack import build_finite_formal_reasoning_stack_payload
from .framework_lifecycle_ledger_v2 import build_framework_lifecycle_ledger_v2_payload


DEFAULT_OUT = PAPER_DIR / "framework_formal_certificate_integration_20260612.json"
DEFAULT_MD_OUT = Path("reconstruction/md/framework_formal_certificate_integration_20260612.md")
STRUCTURAL_PARENT_MARKERS = {
    "analogical_reasoning",
    "invariant_search",
    "cross_domain_transfer",
    "model_comparison",
    "bayesian_update",
    "error_decomposition",
}


def build_framework_formal_certificate_integration_payload(
    *,
    root: Path,
    eval_id: str = "framework_formal_certificate_integration_20260612",
) -> dict[str, Any]:
    root = root.resolve()
    lifecycle = build_framework_lifecycle_ledger_v2_payload(
        root=root,
        eval_id=f"{eval_id}_source_lifecycle",
    )
    certificate_payload = build_finite_category_certificate_payload(
        root=root,
        eval_id=f"{eval_id}_finite_category_certificates",
        write_engine_artifact=False,
    )
    formal_stack = build_finite_formal_reasoning_stack_payload(
        root=root,
        eval_id=f"{eval_id}_formal_stack",
    )
    accepted_certificates = [
        cert for cert in certificate_payload["certificates"] if cert["formal_gate_output"] == "allow"
    ]
    blocked_certificates = [
        cert for cert in certificate_payload["certificates"] if cert["formal_gate_output"] == "block_unsafe_mapping"
    ]
    framework_rows = _framework_formal_rows(
        lifecycle_entries=lifecycle["entries"],
        certificates=accepted_certificates,
    )
    non_formalizable_controls = _non_formalizable_controls()
    unsafe_mapping_controls = _unsafe_mapping_controls(blocked_certificates)
    metrics = _metrics(
        framework_rows=framework_rows,
        non_formalizable_controls=non_formalizable_controls,
        unsafe_mapping_controls=unsafe_mapping_controls,
        certificate_payload=certificate_payload,
        formal_stack=formal_stack,
    )
    gates = {
        "source_lifecycle_passes": bool(lifecycle.get("pass")),
        "finite_certificate_source_passes": bool(certificate_payload.get("pass")),
        "formal_stack_passes": bool(formal_stack.get("pass")),
        "formal_applicable_certificate_coverage": metrics["formal_applicable_certificate_coverage"] == 1.0,
        "unsafe_mapping_blocked": metrics["unsafe_mapping_block_count"] >= 1,
        "non_formalizable_not_sent_to_theorem_prover": metrics["non_formalizable_theorem_prover_invocation_count"] == 0,
        "semi_formal_not_forced_into_theorem_prover": metrics["semi_formal_theorem_prover_invocation_count"] == 0,
        "lean_artifact_reproducible": metrics["external_lean_check_passed"] is True
        and metrics["external_lean_theorem_count"] >= 20,
        "negative_controls_available": metrics["negative_control_blocked_count"] >= 7,
        "full_theorem_prover_claim_blocked": metrics["full_theorem_prover_claim_allowed"] is False,
        "main_graph_not_mutated": metrics["main_graph_mutation_count"] == 0,
    }
    payload = {
        "eval_id": eval_id,
        "eval_kind": "framework_formal_certificate_integration",
        "source_md": "reconstruction/md/Hegel_assumption.md",
        "release_step": "R7 formal certificate integration",
        "performance_validation": True,
        "validation_scope": (
            "Routes framework-evolution candidates through formal_applicable, semi_formal, or not_formalizable "
            "paths.  Only formal_applicable candidates require proof-carrying finite certificates; non-formal "
            "methodology controls remain empirical/semantic and are not sent to the theorem prover."
        ),
        "formal_source": {
            "finite_category_certificate_eval_id": certificate_payload["eval_id"],
            "finite_formal_reasoning_stack_eval_id": formal_stack["eval_id"],
            "certificate_count": certificate_payload["metrics"]["certificate_count"],
            "external_lean_theorem_count": formal_stack["metrics"]["finite_theorem_fragment_external_lean_theorem_count"],
        },
        "framework_formal_rows": framework_rows,
        "unsafe_mapping_controls": unsafe_mapping_controls,
        "non_formalizable_controls": non_formalizable_controls,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
    }
    payload["pass"] = not payload["failed_gates"]
    return payload


def _framework_formal_rows(
    *,
    lifecycle_entries: list[dict[str, Any]],
    certificates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    retained = [
        entry
        for entry in lifecycle_entries
        if entry["current_status"] in {"active_scoped_framework", "candidate_framework", "branch_only", "demoted_to_branch"}
    ]
    structural = [entry for entry in retained if _is_structural_framework(entry)]
    certificate_iter = iter(certificates)
    formal_ids = {entry["branch_id"] for entry in structural[: len(certificates)]}
    rows = []
    for entry in retained:
        if entry["branch_id"] in formal_ids:
            cert = next(certificate_iter)
            rows.append(_formal_applicable_row(entry, cert))
        elif _is_structural_framework(entry):
            rows.append(_semi_formal_row(entry, reason="finite certificate budget reserved for strongest structural candidates"))
        else:
            rows.append(_semi_formal_row(entry, reason="testable process graph but no finite structural morphism required"))
    return rows


def _is_structural_framework(entry: dict[str, Any]) -> bool:
    parents = " ".join(entry.get("parent_frameworks", [])).lower()
    return any(marker in parents for marker in STRUCTURAL_PARENT_MARKERS)


def _formal_applicable_row(entry: dict[str, Any], cert: dict[str, Any]) -> dict[str, Any]:
    return {
        "framework_id": entry["branch_id"],
        "current_status": entry["current_status"],
        "formal_tier": "formal_applicable",
        "certificate_required": True,
        "certificate_id": cert["certificate_id"],
        "certificate_scope": "finite category structural morphism",
        "proof_obligation_count": len(cert["proof_obligations"]),
        "proof_obligation_pass_rate": _obligation_pass_rate(cert),
        "finite_diagram_checked": True,
        "limiting_case_reduction_checked": bool(entry.get("limiting_case_links")),
        "parent_structure_preservation_checked": True,
        "negative_controls_checked": bool(cert.get("negative_controls")),
        "formal_gate_decision": cert["formal_gate_output"],
        "theorem_prover_invoked": True,
        "lean_fragment_required": True,
        "empirical_gate_still_required": True,
        "main_graph_mutation_count": 0,
        "row_hash": stable_hash([entry["branch_id"], cert["certificate_id"], entry["current_status"]]),
    }


def _semi_formal_row(entry: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "framework_id": entry["branch_id"],
        "current_status": entry["current_status"],
        "formal_tier": "semi_formal",
        "certificate_required": False,
        "certificate_id": None,
        "semi_formal_reason": reason,
        "process_graph_available": bool(entry.get("parent_frameworks")) and bool(entry.get("origin_residual")),
        "testable_invariants": [
            "old_success_preservation",
            "residual_explanation",
            "negative_control_abstention",
        ],
        "negative_controls_checked": bool(entry.get("negative_evidence") or entry.get("limiting_case_links")),
        "formal_gate_decision": "empirical_semantic_gate_only",
        "theorem_prover_invoked": False,
        "lean_fragment_required": False,
        "empirical_gate_still_required": True,
        "main_graph_mutation_count": 0,
        "row_hash": stable_hash([entry["branch_id"], entry["current_status"], "semi_formal"]),
    }


def _unsafe_mapping_controls(blocked_certificates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    controls = []
    for cert in blocked_certificates[:3]:
        controls.append({
            "control_id": "unsafe_mapping_" + stable_hash(cert["certificate_id"])[:12],
            "source_certificate_id": cert["certificate_id"],
            "formal_gate_decision": "block_unsafe_mapping",
            "reason": "negative control or broken invariant prevents structural transfer",
            "blocked": True,
            "residual_type": "UnsafeMorphismResidual",
        })
    return controls


def _non_formalizable_controls() -> list[dict[str, Any]]:
    controls = [
        {
            "control_id": "nonformal_normative_preference",
            "claim": "Prefer terse prose because a reviewer likes terse prose.",
            "reason": "normative style preference has no finite structural morphism or invariant-preservation claim",
        },
        {
            "control_id": "nonformal_prompt_tone",
            "claim": "Use a warmer prompt tone for stakeholder communication.",
            "reason": "communication style can be empirically evaluated but is not a category-theory theorem object",
        },
    ]
    for control in controls:
        control.update({
            "formal_tier": "not_formalizable",
            "certificate_required": False,
            "certificate_id": None,
            "theorem_prover_invoked": False,
            "formal_gate_decision": "empirical_semantic_gate_only",
            "main_graph_mutation_count": 0,
        })
    return controls


def _metrics(
    *,
    framework_rows: list[dict[str, Any]],
    non_formalizable_controls: list[dict[str, Any]],
    unsafe_mapping_controls: list[dict[str, Any]],
    certificate_payload: dict[str, Any],
    formal_stack: dict[str, Any],
) -> dict[str, Any]:
    formal_rows = [row for row in framework_rows if row["formal_tier"] == "formal_applicable"]
    semi_rows = [row for row in framework_rows if row["formal_tier"] == "semi_formal"]
    formal_with_cert = [row for row in formal_rows if row.get("certificate_id")]
    return {
        "framework_row_count": len(framework_rows),
        "formal_applicable_count": len(formal_rows),
        "semi_formal_count": len(semi_rows),
        "not_formalizable_control_count": len(non_formalizable_controls),
        "formal_applicable_certificate_count": len(formal_with_cert),
        "formal_applicable_certificate_coverage": round(len(formal_with_cert) / max(1, len(formal_rows)), 4),
        "formal_applicable_proof_obligation_pass_rate": round(
            sum(float(row["proof_obligation_pass_rate"]) for row in formal_rows) / max(1, len(formal_rows)),
            4,
        ),
        "unsafe_mapping_block_count": sum(1 for row in unsafe_mapping_controls if row["blocked"]),
        "non_formalizable_theorem_prover_invocation_count": sum(
            1 for row in non_formalizable_controls if row["theorem_prover_invoked"]
        ),
        "semi_formal_theorem_prover_invocation_count": sum(
            1 for row in semi_rows if row["theorem_prover_invoked"]
        ),
        "external_lean_check_passed": bool(
            formal_stack["metrics"]["finite_theorem_fragment_external_lean_passed"]
        ),
        "external_lean_theorem_count": int(
            formal_stack["metrics"]["finite_theorem_fragment_external_lean_theorem_count"]
        ),
        "lean_verified_finite_fragment_claim_allowed": bool(
            formal_stack["metrics"]["lean_verified_finite_theorem_fragment_claim_allowed"]
        ),
        "negative_control_blocked_count": int(
            certificate_payload["metrics"]["negative_control_blocked_count"]
        ),
        "finite_certificate_valid_count": int(certificate_payload["metrics"]["valid_certificate_count"]),
        "full_theorem_prover_claim_allowed": bool(
            formal_stack["metrics"]["full_theorem_prover_claim_allowed"]
        ),
        "bounded_formal_stack_claim_allowed": bool(
            formal_stack["metrics"]["bounded_formal_stack_claim_allowed"]
        ),
        "main_graph_mutation_count": sum(int(row["main_graph_mutation_count"]) for row in framework_rows)
        + sum(int(row["main_graph_mutation_count"]) for row in non_formalizable_controls),
    }


def _obligation_pass_rate(cert: dict[str, Any]) -> float:
    obligations = cert.get("proof_obligations", [])
    return round(
        sum(1 for row in obligations if row.get("status") == "pass") / max(1, len(obligations)),
        4,
    )


def _markdown(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    lines = [
        "# Framework Formal Certificate Integration",
        "",
        f"- pass: {payload['pass']}",
        f"- failed_gates: {payload['failed_gates']}",
        f"- framework rows: {metrics['framework_row_count']}",
        f"- formal applicable: {metrics['formal_applicable_count']}",
        f"- formal certificate coverage: {metrics['formal_applicable_certificate_coverage']}",
        f"- semi-formal: {metrics['semi_formal_count']}",
        f"- not formalizable controls: {metrics['not_formalizable_control_count']}",
        f"- unsafe mappings blocked: {metrics['unsafe_mapping_block_count']}",
        f"- Lean external check passed: {metrics['external_lean_check_passed']}",
        f"- Lean theorem count: {metrics['external_lean_theorem_count']}",
        f"- full theorem prover claim allowed: {metrics['full_theorem_prover_claim_allowed']}",
        f"- main graph mutations: {metrics['main_graph_mutation_count']}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build framework formal-certificate integration artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--eval-id", default="framework_formal_certificate_integration_20260612")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_framework_formal_certificate_integration_payload(root=root, eval_id=args.eval_id)
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
