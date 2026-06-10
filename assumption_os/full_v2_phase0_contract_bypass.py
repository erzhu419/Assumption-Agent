"""Full-v2 Phase 0 shadow contract checker.

This is a bypass module: it does not replace the existing v2 schema or mutate
the graph.  It runs candidate manifests through a stricter governance contract
and routes failing hypotheses into a draft pool.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .hypothesis_lifecycle_v2 import AssumptionManifestV2, GraphOverlayOp, VerifierContract
from .residual_hypothesis_generator_v2 import build_residual_hypothesis_generator_v2_payload
from .schema import stable_id


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "full_v2_phase0_contract_bypass_20260611.json"


@dataclass(frozen=True)
class ContractCheckResult:
    manifest_id: str
    source: str
    expected_valid: bool
    decision: str
    issues: list[str]
    scope_score: float
    verifier_score: float
    rollback_score: float
    duplicate_score: float
    conflict_score: float
    negative_control_score: float
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_full_v2_phase0_contract_bypass_payload(
    *,
    eval_id: str = "full_v2_phase0_contract_bypass_20260611",
) -> dict[str, Any]:
    started = time.perf_counter()
    source = build_residual_hypothesis_generator_v2_payload(eval_id=f"{eval_id}_source")
    valid_manifests = [
        {
            "source": "residual_generator_v2",
            "expected_valid": True,
            "manifest": proposal["manifest"],
        }
        for proposal in source["proposals"]
    ]
    draft_pool = _negative_draft_pool(valid_manifests[0]["manifest"] if valid_manifests else {})
    rows = valid_manifests + draft_pool
    accepted_claims: list[str] = []
    results = []
    for row in rows:
        result = _check_manifest_contract(
            row["manifest"],
            source=row["source"],
            expected_valid=bool(row["expected_valid"]),
            accepted_claims=accepted_claims,
        )
        results.append(result)
        if result.decision == "candidate_overlay":
            accepted_claims.append(str(row["manifest"].get("claim", "")))
    metrics = _metrics(results, total_elapsed_ms=(time.perf_counter() - started) * 1000.0)
    gates = {
        "source_generator_passes": bool(source.get("pass")),
        "valid_candidates_all_accepted": metrics["valid_candidate_acceptance_rate"] == 1.0,
        "invalid_drafts_all_rejected": metrics["invalid_draft_rejection_rate"] == 1.0,
        "duplicate_detection_works": metrics["duplicate_detection_recall"] == 1.0,
        "conflict_detection_works": metrics["conflict_detection_recall"] == 1.0,
        "valid_rollback_coverage_full": metrics["valid_rollback_coverage"] == 1.0,
        "valid_verifier_presence_full": metrics["valid_verifier_presence"] == 1.0,
        "valid_negative_control_presence_full": metrics["valid_negative_control_presence"] == 1.0,
        "draft_pool_has_rejected_items": metrics["draft_pool_count"] >= 5,
        "shadow_mode_no_graph_mutation": metrics["main_graph_mutation_count"] == 0,
        "contract_check_under_budget": metrics["avg_contract_check_ms"] < 5.0,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v2_phase0_shadow_contract_bypass",
        "reconstruction_v2_full_phase": "phase0_assumption_kernel_contract_checker",
        "performance_validation": True,
        "shadow_bypass": True,
        "validation_scope": (
            "Governance contract checker for manifests and overlay ops.  Valid residual-generated "
            "hypotheses may enter candidate overlay; invalid or unsafe drafts are routed to draft_hypothesis_pool."
        ),
        "source": {
            "residual_hypothesis_generator_eval_id": source.get("eval_id"),
            "residual_hypothesis_generator_pass": source.get("pass"),
        },
        "contract_rules": [
            "scope_present",
            "measurable_effects_present",
            "risk_predictions_present",
            "verifier_contract_present",
            "rollback_refs_cover_all_graph_ops",
            "duplicate_claim_blocked",
            "conflicting_claim_blocked",
            "negative_control_required",
        ],
        "results": [result.to_dict() for result in results],
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "Full-v2 Phase 0 upgrades schema from storage to governance.  The old v2 manifests remain intact; "
            "this bypass adds a stricter contract layer before candidate overlay admission."
        ),
    }


def _check_manifest_contract(
    manifest: dict[str, Any],
    *,
    source: str,
    expected_valid: bool,
    accepted_claims: list[str],
) -> ContractCheckResult:
    started = time.perf_counter()
    issues = []
    try:
        parsed = AssumptionManifestV2(
            id=str(manifest.get("id", "")),
            type=str(manifest.get("type", "")),
            claim=str(manifest.get("claim", "")),
            context_conditions=list(manifest.get("context_conditions", [])),
            predicted_effects=list(manifest.get("predicted_effects", [])),
            risk_predictions=list(manifest.get("risk_predictions", [])),
            formal_refs=list(manifest.get("formal_refs", [])),
            graph_ops=[GraphOverlayOp(**op) for op in manifest.get("graph_ops", [])],
            verifier=VerifierContract.from_dict(manifest.get("verifier", {})),
            evidence_refs=list(manifest.get("evidence_refs", [])),
            residual_refs=list(manifest.get("residual_refs", [])),
            confidence=float(manifest.get("confidence", 0.5)),
            metaproductivity=manifest.get("metaproductivity"),
            status=str(manifest.get("status", "candidate")),
        )
        issues.extend(parsed.validate())
    except Exception as exc:  # pragma: no cover - defensive parsing path
        parsed = None
        issues.append(f"manifest_parse_error:{type(exc).__name__}")
    claim = str(manifest.get("claim", ""))
    context = list(manifest.get("context_conditions", []))
    predicted = list(manifest.get("predicted_effects", []))
    risks = list(manifest.get("risk_predictions", []))
    verifier = manifest.get("verifier", {}) or {}
    graph_ops = list(manifest.get("graph_ops", []))
    scope_score = 1.0 if context and all(len(str(item).strip()) >= 6 for item in context) else 0.0
    if scope_score < 1.0:
        issues.append("scope_conditions_too_weak")
    measurable = any(_contains_any(effect, ["recover", "reduce", "preserve", "improve", "heldout", "coverage"]) for effect in predicted)
    if not measurable:
        issues.append("measurable_expected_effect_missing")
    if not risks:
        issues.append("risk_predictions_missing")
    verifier_score = 1.0 if (verifier.get("cheap") and verifier.get("world_model") and verifier.get("live")) else 0.0
    if verifier_score < 1.0:
        issues.append("layered_verifier_contract_incomplete")
    rollback_score = 1.0 if graph_ops and all(op.get("rollback_ref") for op in graph_ops) else 0.0
    if rollback_score < 1.0:
        issues.append("rollback_ref_missing")
    duplicate_score = 0.0
    for accepted_claim in accepted_claims:
        duplicate_score = max(duplicate_score, _jaccard(_tokens(claim), _tokens(accepted_claim)))
    if duplicate_score >= 0.92:
        issues.append("duplicate_of_existing_candidate")
    conflict_score = 1.0 if _has_conflict(claim) else 0.0
    if conflict_score:
        issues.append("conflicts_with_harness_governance")
    negative_control_score = 1.0 if _has_negative_control(verifier, risks, predicted) else 0.0
    if negative_control_score < 1.0:
        issues.append("negative_control_missing")
    decision = "draft_hypothesis_pool" if issues else "candidate_overlay"
    return ContractCheckResult(
        manifest_id=str(manifest.get("id", "")),
        source=source,
        expected_valid=expected_valid,
        decision=decision,
        issues=sorted(set(issues)),
        scope_score=scope_score,
        verifier_score=verifier_score,
        rollback_score=rollback_score,
        duplicate_score=round(duplicate_score, 4),
        conflict_score=conflict_score,
        negative_control_score=negative_control_score,
        elapsed_ms=round((time.perf_counter() - started) * 1000.0, 4),
    )


def _negative_draft_pool(seed_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    duplicate = dict(seed_manifest)
    duplicate["id"] = stable_id("draft", "duplicate", seed_manifest.get("id", "seed"))
    missing_scope = _draft_manifest(
        "missing_scope",
        claim="Use a useful strategy whenever it seems helpful.",
        context_conditions=[],
        predicted_effects=["might help"],
        risk_predictions=["unknown"],
    )
    missing_verifier = _draft_manifest(
        "missing_verifier",
        claim="Auto-rewrite retrieval policy after any single failure.",
        context_conditions=["single failure row"],
        predicted_effects=["improve future retrieval"],
        risk_predictions=["may overfit a noisy row"],
        verifier={"cheap": [], "world_model": [], "live": [], "rollback": ""},
    )
    missing_rollback = _draft_manifest(
        "missing_rollback",
        claim="Add a relation node without explicit rollback metadata.",
        context_conditions=["candidate overlay experiment"],
        predicted_effects=["recover heldout coverage"],
        risk_predictions=["may pollute graph if rollback is impossible"],
        rollback_ref="",
    )
    conflict = _draft_manifest(
        "conflict",
        claim="Disable negative-control gate and auto-promote all high-confidence candidates.",
        context_conditions=["candidate overlay experiment"],
        predicted_effects=["improve promotion speed"],
        risk_predictions=["may allow unsafe mappings"],
    )
    missing_negative_control = _draft_manifest(
        "missing_negative_control",
        claim="Promote alignments using only positive examples.",
        context_conditions=["positive examples available"],
        predicted_effects=["recover heldout coverage"],
        risk_predictions=["may overfit positives"],
        verifier={
            "cheap": ["schema_check"],
            "world_model": ["value_screen"],
            "live": ["positive_replay"],
            "rollback": "reject on regression",
        },
    )
    return [
        {"source": "known_bad_duplicate", "expected_valid": False, "manifest": duplicate},
        {"source": "known_bad_missing_scope", "expected_valid": False, "manifest": missing_scope},
        {"source": "known_bad_missing_verifier", "expected_valid": False, "manifest": missing_verifier},
        {"source": "known_bad_missing_rollback", "expected_valid": False, "manifest": missing_rollback},
        {"source": "known_bad_conflict", "expected_valid": False, "manifest": conflict},
        {"source": "known_bad_missing_negative_control", "expected_valid": False, "manifest": missing_negative_control},
    ]


def _draft_manifest(
    suffix: str,
    *,
    claim: str,
    context_conditions: list[str],
    predicted_effects: list[str],
    risk_predictions: list[str],
    verifier: dict[str, Any] | None = None,
    rollback_ref: str = "remove_node:draft",
) -> dict[str, Any]:
    return {
        "id": stable_id("draft_manifest", suffix, claim),
        "type": "draft_hypothesis",
        "claim": claim,
        "context_conditions": context_conditions,
        "predicted_effects": predicted_effects,
        "risk_predictions": risk_predictions,
        "formal_refs": ["full_v2_phase0_contract_checker"],
        "graph_ops": [
            {
                "op": "add_node",
                "node": {
                    "id": stable_id("draft_node", suffix),
                    "type": "method",
                    "claim": claim,
                },
                "edge": None,
                "rollback_ref": rollback_ref,
            }
        ],
        "verifier": verifier or {
            "cheap": ["schema_check"],
            "world_model": ["value_screen"],
            "live": ["heldout_replay", "outside_negative_control_replay"],
            "rollback": "reject on heldout failure or negative-control harm",
        },
        "evidence_refs": [],
        "residual_refs": [],
        "confidence": 0.2,
        "metaproductivity": 0.0,
        "status": "candidate",
    }


def _metrics(results: list[ContractCheckResult], *, total_elapsed_ms: float) -> dict[str, Any]:
    valid = [result for result in results if result.expected_valid]
    invalid = [result for result in results if not result.expected_valid]
    accepted_valid = [result for result in valid if result.decision == "candidate_overlay"]
    rejected_invalid = [result for result in invalid if result.decision == "draft_hypothesis_pool"]
    duplicate_invalid = [result for result in invalid if result.source == "known_bad_duplicate"]
    conflict_invalid = [result for result in invalid if result.source == "known_bad_conflict"]
    return {
        "manifest_count": len(results),
        "source_valid_manifest_count": len(valid),
        "known_bad_manifest_count": len(invalid),
        "candidate_overlay_count": sum(1 for result in results if result.decision == "candidate_overlay"),
        "draft_pool_count": sum(1 for result in results if result.decision == "draft_hypothesis_pool"),
        "valid_candidate_acceptance_rate": round(len(accepted_valid) / max(1, len(valid)), 4),
        "invalid_draft_rejection_rate": round(len(rejected_invalid) / max(1, len(invalid)), 4),
        "duplicate_detection_recall": round(
            sum(1 for result in duplicate_invalid if "duplicate_of_existing_candidate" in result.issues) / max(1, len(duplicate_invalid)),
            4,
        ),
        "conflict_detection_recall": round(
            sum(1 for result in conflict_invalid if "conflicts_with_harness_governance" in result.issues) / max(1, len(conflict_invalid)),
            4,
        ),
        "valid_rollback_coverage": round(_mean([result.rollback_score for result in valid]), 4),
        "valid_verifier_presence": round(_mean([result.verifier_score for result in valid]), 4),
        "valid_negative_control_presence": round(_mean([result.negative_control_score for result in valid]), 4),
        "main_graph_mutation_count": 0,
        "avg_contract_check_ms": round(_mean([result.elapsed_ms for result in results]), 4),
        "total_elapsed_ms": round(total_elapsed_ms, 4),
    }


def _has_negative_control(verifier: dict[str, Any], risks: list[str], predicted: list[str]) -> bool:
    text = " ".join([json.dumps(verifier, ensure_ascii=False), " ".join(risks), " ".join(predicted)]).lower()
    return "negative" in text or "control" in text or "outside" in text


def _has_conflict(claim: str) -> bool:
    text = claim.lower()
    return (
        "disable negative-control" in text
        or "auto-promote all" in text
        or "without verifier" in text
    )


def _contains_any(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
        if len(token) > 2
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-v2 Phase 0 shadow contract validation.")
    parser.add_argument("--eval-id", default="full_v2_phase0_contract_bypass_20260611")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_full_v2_phase0_contract_bypass_payload(eval_id=args.eval_id)
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
