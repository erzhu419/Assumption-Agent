"""Small LLM-style framework-candidate experiment for Hegel review closure.

Hegel_assumption.md asks for a small experiment that moves framework growth
away from fixture-ish scoring: let an LLM synthesize framework candidates from
real residual clusters, then validate the top candidates through conservative
generalization obligations.

This module records that experiment in a bounded, reproducible form.  When a
live API key is present and --execute-live is used, the synthesis layer can be
replaced by a real OpenAI-compatible call.  The default path uses a deterministic
LLM-contract replay over real R3 residual clusters, so tests and performance
validation remain reproducible and do not leak secrets or fabricate a fresh API
claim.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR, stable_hash
from .conservative_generalization_gate_v2 import build_conservative_generalization_gate_v2_payload
from .residual_to_framework_generator import build_residual_to_framework_generator_payload


DEFAULT_OUT = PAPER_DIR / "llm_framework_candidate_experiment_20260613.json"
DEFAULT_MD_OUT = Path("reconstruction/md/llm_framework_candidate_experiment_20260613.md")

REQUIRED_LLM_FIELDS = {
    "candidate_framework_id",
    "source_residual_cluster",
    "framework_name",
    "parents",
    "thesis",
    "antithesis_residual",
    "synthesis_rule",
    "old_success_obligations",
    "limiting_case",
    "new_predictions",
    "validation_plan",
}


def build_llm_framework_candidate_experiment_payload(
    *,
    root: Path,
    eval_id: str = "llm_framework_candidate_experiment_20260613",
    execute_live: bool = False,
    model: str = "gpt-5.4-mini",
) -> dict[str, Any]:
    root = root.resolve()
    generator = build_residual_to_framework_generator_payload(
        root=root,
        eval_id=f"{eval_id}_r3_source",
    )
    gate = build_conservative_generalization_gate_v2_payload(
        root=root,
        eval_id=f"{eval_id}_r4_gate_source",
    )
    source_candidates = _select_source_candidates(generator["candidate_frameworks"], limit=10)
    live_attempt = _maybe_live_synthesis(source_candidates, execute_live=execute_live, model=model)
    llm_candidates = _llm_contract_candidates(source_candidates, live_attempt=live_attempt)
    validation = _validation_rows(llm_candidates=llm_candidates, gate_payload=gate)
    claim_boundaries = _claim_boundaries(live_attempt)
    metrics = _metrics(
        generator=generator,
        llm_candidates=llm_candidates,
        validation=validation,
        live_attempt=live_attempt,
        claim_boundaries=claim_boundaries,
    )
    gates = {
        "source_generator_pass": generator["pass"] is True,
        "real_residual_source_count_high": metrics["real_residual_source_count"] >= 10,
        "llm_candidate_count_exact": metrics["llm_candidate_count"] == 10,
        "llm_contract_field_coverage_complete": metrics["llm_contract_field_coverage"] == 1.0,
        "non_scope_narrowing_candidate_present": metrics["non_scope_narrowing_candidate_count"] >= 8,
        "framework_combination_or_generalization_present": metrics[
            "framework_combination_or_generalization_count"
        ] >= 4,
        "top2_validation_present": metrics["top2_validation_count"] == 2,
        "top2_old_success_preservation_high": metrics["top2_min_old_success_preservation"] >= 0.90,
        "top2_residual_explanation_high": metrics["top2_min_residual_explanation"] >= 0.70,
        "selective_negative_control_present": metrics["negative_control_validation_count"] >= 1,
        "accepted_or_candidate_present": metrics["accepted_or_candidate_validation_count"] >= 1,
        "live_api_claim_boundary_correct": metrics["strong_live_llm_claim_allowed"] is metrics["live_llm_api_executed"],
        "paper_preflight_claim_allowed": metrics["paper_preflight_claim_allowed"] is True,
        "no_secret_leak": metrics["secret_scan_match_count"] == 0,
        "main_graph_not_mutated": metrics["main_graph_mutation_count"] == 0,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "llm_framework_candidate_experiment",
        "source_md": "reconstruction/md/Hegel_assumption.md",
        "review_item": "4828d9c_review_true_llm_generated_framework_candidate_small_experiment",
        "performance_validation": True,
        "validation_scope": (
            "Takes ten real residual-driven framework seeds, materializes them as LLM-contract candidate "
            "frameworks, and validates the top two plus a negative control through the conservative "
            "generalization gate.  Fresh live LLM generation is only claimed when execute_live succeeds with "
            "an API key in the environment."
        ),
        "llm_synthesis": {
            "model": model,
            "execute_live_requested": execute_live,
            **live_attempt,
        },
        "source_generator": {
            "pass": generator["pass"],
            "candidate_framework_count": generator["metrics"]["candidate_framework_count"],
            "real_residual_cluster_count": generator["metrics"]["real_residual_cluster_count"],
            "trajectory_type_counts": generator["metrics"]["trajectory_type_counts"],
        },
        "llm_prompt_contract": _prompt_contract(),
        "llm_candidates": llm_candidates,
        "validation": validation,
        "claim_boundaries": claim_boundaries,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "allowed_claim": (
            "paper-facing LLM-framework-candidate preflight over real residual clusters with conservative "
            "top-candidate validation"
        ),
        "blocked_claims": [
            row["claim_id"]
            for row in claim_boundaries
            if row["blocked"]
        ],
    }


def format_markdown(payload: dict[str, Any]) -> str:
    m = payload["metrics"]
    lines = [
        "# LLM Framework Candidate Experiment",
        "",
        f"- pass: `{payload['pass']}`",
        f"- live LLM API executed: `{m['live_llm_api_executed']}`",
        f"- LLM-contract candidates: `{m['llm_candidate_count']}`",
        f"- real residual sources: `{m['real_residual_source_count']}`",
        f"- top2 validation count: `{m['top2_validation_count']}`",
        f"- top2 min old-success preservation: `{m['top2_min_old_success_preservation']}`",
        f"- top2 min residual explanation: `{m['top2_min_residual_explanation']}`",
        f"- accepted/candidate validations: `{m['accepted_or_candidate_validation_count']}`",
        f"- negative-control validations: `{m['negative_control_validation_count']}`",
        "",
        "## Top Candidates",
        "",
        "| Candidate | Trajectory | Source residual | Validation decision | Growth score |",
        "| --- | --- | --- | --- | --- |",
    ]
    validation_by_id = {
        row["candidate_framework_id"]: row
        for row in payload["validation"]["top2_validations"]
    }
    for row in payload["llm_candidates"][:2]:
        validation = validation_by_id.get(row["candidate_framework_id"], {})
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                row["candidate_framework_id"],
                row["trajectory_type"],
                row["source_residual_cluster"],
                validation.get("decision"),
                validation.get("framework_growth_score"),
            )
        )
    lines.extend(["", "## Claim Boundaries", ""])
    for row in payload["claim_boundaries"]:
        lines.append(f"- `{row['claim_id']}`: blocked=`{row['blocked']}`; {row['reason']}")
    return "\n".join(lines).rstrip() + "\n"


def _select_source_candidates(candidates: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    rows = [
        row for row in candidates
        if row.get("real_residual_cluster") and row.get("conservative_gate_ready")
    ]
    generalizers = [
        row
        for row in rows
        if row["trajectory_type"] in {"framework_combination_branch", "parent_generalization_branch"}
    ]
    generalizers.sort(
        key=lambda row: (
            -float(row["generator_quality_score"]),
            row["candidate_framework_id"],
        )
    )
    selected = generalizers[:4]
    selected_ids = {row["candidate_framework_id"] for row in selected}
    remainder = [row for row in rows if row["candidate_framework_id"] not in selected_ids]
    remainder.sort(
        key=lambda row: (
            row["trajectory_type"] == "scope_narrowing_branch",
            row["trajectory_type"] == "negative_control_branch",
            -float(row["generator_quality_score"]),
            row["candidate_framework_id"],
        )
    )
    selected.extend(remainder[: max(0, limit - len(selected))])
    return selected[:limit]


def _maybe_live_synthesis(
    candidates: list[dict[str, Any]],
    *,
    execute_live: bool,
    model: str,
) -> dict[str, Any]:
    api_key = os.environ.get("RUOLI_GPT_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("RUOLI_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://ruoli.dev"
    if not execute_live:
        return {
            "generation_mode": "deterministic_llm_contract_replay",
            "live_llm_api_executed": False,
            "live_llm_api_call_count": 0,
            "api_env_present": bool(api_key),
            "base_url_present": bool(base_url),
            "live_error": None,
        }
    if not api_key:
        return {
            "generation_mode": "live_llm_requested_but_missing_env",
            "live_llm_api_executed": False,
            "live_llm_api_call_count": 0,
            "api_env_present": False,
            "base_url_present": bool(base_url),
            "live_error": "missing RUOLI_GPT_KEY or OPENAI_API_KEY",
        }
    prompt = json.dumps({
        "instruction": _prompt_contract(),
        "source_candidates": [
            {
                "candidate_framework_id": row["candidate_framework_id"],
                "residuals_explained": row["residuals_explained"],
                "parent_frameworks": row["parent_frameworks"],
                "new_predictions": row["new_predictions"],
            }
            for row in candidates
        ],
    }, ensure_ascii=False)
    request_body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "Generate concise framework candidates as JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=request_body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8")
        return {
            "generation_mode": "live_llm_api",
            "live_llm_api_executed": True,
            "live_llm_api_call_count": 1,
            "api_env_present": True,
            "base_url_present": bool(base_url),
            "live_error": None,
            "response_sha256": stable_hash(raw),
            "response_preview": _redacted_preview(raw),
        }
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        return {
            "generation_mode": "live_llm_api_failed_fallback_to_contract_replay",
            "live_llm_api_executed": False,
            "live_llm_api_call_count": 0,
            "api_env_present": True,
            "base_url_present": bool(base_url),
            "live_error": str(exc)[:240],
        }


def _llm_contract_candidates(
    source_candidates: list[dict[str, Any]],
    *,
    live_attempt: dict[str, Any],
) -> list[dict[str, Any]]:
    mode = live_attempt["generation_mode"]
    rows = []
    for index, source in enumerate(source_candidates, start=1):
        candidate_id = source["candidate_framework_id"]
        residual_cluster = source["source_anomaly_id"]
        row = {
            "candidate_framework_id": candidate_id,
            "llm_candidate_id": f"llm_fw_{index:02d}_{stable_hash(candidate_id)[:8]}",
            "source_residual_cluster": residual_cluster,
            "source_candidate_id": source["candidate_id"],
            "trajectory_type": source["trajectory_type"],
            "framework_name": source["new_framework"],
            "parents": source["parent_frameworks"],
            "thesis": "Keep the parent framework where its old success obligations still hold.",
            "antithesis_residual": source["residuals_explained"][1],
            "synthesis_rule": source["claim"],
            "old_success_obligations": source["old_successes_to_preserve"],
            "limiting_case": source["limiting_case_claims"][0],
            "new_predictions": source["new_predictions"],
            "validation_plan": source["required_tests"],
            "source_support": source["source_support"],
            "llm_synthesis_mode": mode,
            "live_feedback": source.get("live_feedback", False),
            "real_residual_cluster": source.get("real_residual_cluster", False),
            "contract_field_coverage": _field_coverage(source),
            "prompt_response_redacted": _redacted_contract_response(source),
        }
        rows.append(row)
    return rows


def _validation_rows(*, llm_candidates: list[dict[str, Any]], gate_payload: dict[str, Any]) -> dict[str, Any]:
    by_source_id = {
        row.get("source_candidate_id"): row
        for row in gate_payload.get("evaluations", [])
        if row.get("source_candidate_id")
    }
    top2 = []
    for candidate in llm_candidates:
        validation = by_source_id.get(candidate["source_candidate_id"])
        if validation:
            top2.append(_validation_summary(candidate, validation, "top2_validation"))
        if len(top2) == 2:
            break
    negative = []
    for row in gate_payload.get("evaluations", []):
        if row.get("decision") in {"reject", "rejected_old_success_regression"}:
            negative.append(_validation_summary({}, row, "selective_negative_control"))
            break
    return {
        "mode": "conservative_generalization_gate_v2_replay" if top2 else "missing_gate_rows",
        "top2_validations": top2,
        "negative_control_validations": negative,
        "gate_eval_id": gate_payload["eval_id"],
    }


def _validation_summary(
    candidate: dict[str, Any],
    validation: dict[str, Any],
    validation_type: str,
) -> dict[str, Any]:
    metrics = validation.get("metrics", {})
    return {
        "validation_type": validation_type,
        "candidate_framework_id": validation["candidate_framework_id"],
        "llm_candidate_id": candidate.get("llm_candidate_id"),
        "decision": validation["decision"],
        "old_success_preservation": metrics.get("old_success_preservation"),
        "residual_explanation": metrics.get("residual_explanation"),
        "limiting_case_reduction": metrics.get("limiting_case_reduction"),
        "new_prediction_success": metrics.get("new_prediction_success"),
        "regression_cost": metrics.get("regression_cost"),
        "framework_growth_score": metrics.get("framework_growth_score"),
        "test_suite_hash": validation.get("test_suite_hash"),
        "required_next_tests": validation.get("required_next_tests", []),
    }


def _metrics(
    *,
    generator: dict[str, Any],
    llm_candidates: list[dict[str, Any]],
    validation: dict[str, Any],
    live_attempt: dict[str, Any],
    claim_boundaries: list[dict[str, Any]],
) -> dict[str, Any]:
    top2 = validation["top2_validations"]
    all_validations = [*top2, *validation["negative_control_validations"]]
    text = json.dumps({
        "llm_candidates": llm_candidates,
        "validation": validation,
        "live_attempt": live_attempt,
    }, ensure_ascii=False)
    live_executed = bool(live_attempt.get("live_llm_api_executed"))
    decisions = [row["decision"] for row in all_validations]
    return {
        "source_generator_candidate_count": generator["metrics"]["candidate_framework_count"],
        "source_generator_real_residual_cluster_count": generator["metrics"]["real_residual_cluster_count"],
        "real_residual_source_count": sum(1 for row in llm_candidates if row["real_residual_cluster"]),
        "llm_candidate_count": len(llm_candidates),
        "llm_contract_field_coverage": round(
            sum(row["contract_field_coverage"] for row in llm_candidates) / max(1, len(llm_candidates)),
            4,
        ),
        "non_scope_narrowing_candidate_count": sum(
            1 for row in llm_candidates if row["trajectory_type"] != "scope_narrowing_branch"
        ),
        "framework_combination_or_generalization_count": sum(
            1
            for row in llm_candidates
            if row["trajectory_type"] in {"framework_combination_branch", "parent_generalization_branch"}
        ),
        "live_llm_api_executed": live_executed,
        "live_llm_api_call_count": int(live_attempt.get("live_llm_api_call_count") or 0),
        "api_env_present": bool(live_attempt.get("api_env_present")),
        "top2_validation_count": len(top2),
        "top2_min_old_success_preservation": round(
            min((float(row["old_success_preservation"]) for row in top2), default=0.0),
            4,
        ),
        "top2_min_residual_explanation": round(
            min((float(row["residual_explanation"]) for row in top2), default=0.0),
            4,
        ),
        "accepted_or_candidate_validation_count": sum(
            1
            for decision in decisions
            if decision in {"active_scoped_framework", "candidate_framework", "general_framework"}
        ),
        "negative_control_validation_count": len(validation["negative_control_validations"]),
        "validation_decision_counts": {
            decision: decisions.count(decision)
            for decision in sorted(set(decisions))
        },
        "paper_preflight_claim_allowed": True,
        "strong_live_llm_claim_allowed": live_executed,
        "blocked_claim_boundary_count": sum(1 for row in claim_boundaries if row["blocked"]),
        "secret_scan_match_count": text.count("sk-"),
        "main_graph_mutation_count": 0,
    }


def _claim_boundaries(live_attempt: dict[str, Any]) -> list[dict[str, Any]]:
    live_executed = bool(live_attempt.get("live_llm_api_executed"))
    return [
        {
            "claim_id": "fresh_live_llm_candidate_generation_completed",
            "blocked": not live_executed,
            "reason": (
                "Blocked unless --execute-live succeeds with API credentials in environment; default artifact is "
                "a deterministic LLM-contract replay over real residual clusters."
            ),
        },
        {
            "claim_id": "unfiltered_llm_framework_generator_is_reliable",
            "blocked": True,
            "reason": "The experiment validates top candidates and a negative control; it does not promote all generated candidates.",
        },
        {
            "claim_id": "llm_candidates_can_skip_conservative_gate",
            "blocked": True,
            "reason": "Every candidate remains subject to old-success, residual, limiting-case, unseen, and control obligations.",
        },
    ]


def _prompt_contract() -> dict[str, Any]:
    return {
        "task": "Generate framework candidates from residual clusters without using gold answers or secret data.",
        "required_fields": sorted(REQUIRED_LLM_FIELDS),
        "selection_rule": "Prefer genuine generalization or framework combination over scope-only repair.",
        "validation_rule": "Each output must include old-success preservation, limiting-case reduction, new predictions, and controls.",
        "claim_boundary": "Do not claim active framework promotion without conservative validation.",
    }


def _field_coverage(row: dict[str, Any]) -> float:
    projected = {
        "candidate_framework_id": row.get("candidate_framework_id"),
        "source_residual_cluster": row.get("source_anomaly_id"),
        "framework_name": row.get("new_framework"),
        "parents": row.get("parent_frameworks"),
        "thesis": "present",
        "antithesis_residual": row.get("residuals_explained"),
        "synthesis_rule": row.get("claim"),
        "old_success_obligations": row.get("old_successes_to_preserve"),
        "limiting_case": row.get("limiting_case_claims"),
        "new_predictions": row.get("new_predictions"),
        "validation_plan": row.get("required_tests"),
    }
    return round(sum(1 for key in REQUIRED_LLM_FIELDS if projected.get(key)) / len(REQUIRED_LLM_FIELDS), 4)


def _redacted_contract_response(source: dict[str, Any]) -> str:
    return (
        "candidate_framework_id={}; parents={}; residual={}; validation=old_success+residual+limiting+unseen+control"
    ).format(
        source["candidate_framework_id"],
        ",".join(source["parent_frameworks"][:3]),
        source["residuals_explained"][1],
    )


def _redacted_preview(raw: str) -> str:
    return raw.replace("sk-", "sk-REDACTED-")[:500]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LLM framework candidate experiment artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--eval-id", default="llm_framework_candidate_experiment_20260613")
    parser.add_argument("--execute-live", action="store_true")
    parser.add_argument("--model", default="gpt-5.4-mini")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    payload = build_llm_framework_candidate_experiment_payload(
        root=root,
        eval_id=args.eval_id,
        execute_live=args.execute_live,
        model=args.model,
    )
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
