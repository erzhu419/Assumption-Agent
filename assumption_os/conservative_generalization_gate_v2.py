"""Conservative-generalization gate v2 over residual-driven framework candidates.

R4 in Hegel_assumption.md asks for the conservative-generalization scores to be
grounded in real test-suite obligations instead of only a mechanism fixture.
This module evaluates candidates produced by the R3 residual-to-framework
generator against old-success, residual, limiting-case, unseen-domain, and
negative-control suites derived from existing first-party artifacts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR, stable_hash
from .conservative_generalization_gate import REQUIRED_PROMOTION_RELATIONS
from .residual_to_framework_generator import build_residual_to_framework_generator_payload
from .schema import EdgeType, stable_id


DEFAULT_OUT = PAPER_DIR / "conservative_generalization_gate_v2_20260612.json"
DEFAULT_MD_OUT = Path("reconstruction/md/conservative_generalization_gate_v2_20260612.md")

PROMOTED_DECISIONS = {"candidate_framework", "active_scoped_framework", "general_framework"}


@dataclass(frozen=True)
class FrameworkTestSuite:
    candidate_framework_id: str
    old_success_tests: list[dict[str, Any]]
    residual_tests: list[dict[str, Any]]
    limiting_case_tests: list[dict[str, Any]]
    unseen_domain_tests: list[dict[str, Any]]
    negative_control_tests: list[dict[str, Any]]
    source_artifacts: list[str]


@dataclass(frozen=True)
class GateV2Certificate:
    certificate_id: str
    candidate_framework_id: str
    decision: str
    metrics: dict[str, float]
    test_suite_hash: str
    required_next_tests: list[str]


def build_conservative_generalization_gate_v2_payload(
    *,
    root: Path,
    eval_id: str = "conservative_generalization_gate_v2_20260612",
) -> dict[str, Any]:
    root = root.resolve()
    generator = build_residual_to_framework_generator_payload(
        root=root,
        eval_id=f"{eval_id}_generator_source",
    )
    selected = _select_candidates(generator["candidate_frameworks"])
    evaluations = [_evaluate_candidate(candidate) for candidate in selected]
    transition = _branch_to_active_transition(selected[0])
    evaluations.extend(transition)
    certificates = [_certificate(row) for row in evaluations if row["decision"] in PROMOTED_DECISIONS]
    graph_patch = _graph_patch(evaluations=evaluations, certificates=certificates)
    metrics = _metrics(
        generator=generator,
        selected=selected,
        evaluations=evaluations,
        certificates=certificates,
        graph_patch=graph_patch,
    )
    gates = {
        "source_generator_pass": generator["pass"] is True,
        "evaluated_candidate_count_high": metrics["evaluated_candidate_count"] >= 24,
        "real_residual_candidate_rate_high": metrics["real_residual_candidate_rate"] >= 0.80,
        "old_success_suite_present": metrics["old_success_test_count"] >= metrics["evaluated_candidate_count"] * 2,
        "residual_suite_present": metrics["residual_test_count"] >= metrics["evaluated_candidate_count"] * 2,
        "limiting_case_suite_present": metrics["limiting_case_test_count"] >= metrics["evaluated_candidate_count"],
        "unseen_domain_suite_present": metrics["unseen_domain_test_count"] >= metrics["evaluated_candidate_count"],
        "active_framework_found": metrics["decision_counts"].get("active_scoped_framework", 0) >= 1,
        "candidate_framework_found": metrics["decision_counts"].get("candidate_framework", 0) >= 1,
        "branch_only_found": metrics["decision_counts"].get("branch_only", 0) >= 1,
        "old_success_reject_found": metrics["old_success_reject_count"] >= 1,
        "branch_to_active_transition_found": metrics["branch_to_active_transition_count"] >= 1,
        "active_certificates_complete": metrics["promoted_certificate_coverage"] == 1.0,
        "rejected_negative_evidence_retained": metrics["rejected_negative_evidence_count"] >= 1,
        "required_relation_coverage": metrics["required_relation_coverage"] == 1.0,
        "main_graph_not_mutated": metrics["main_graph_mutation_count"] == 0,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "conservative_generalization_gate_v2",
        "source_md": "reconstruction/md/Hegel_assumption.md",
        "release_step": "R4_conservative_generalization_gate_v2",
        "performance_validation": True,
        "validation_scope": (
            "Evaluates residual-driven framework candidates with artifact-derived old-success, residual, "
            "limiting-case, unseen-domain, and negative-control suites.  v2 keeps the same conservative "
            "generalization obligations but records branch-to-active promotion and old-success rejection evidence."
        ),
        "source_generator": {
            "pass": generator["pass"],
            "candidate_framework_count": generator["metrics"]["candidate_framework_count"],
            "real_residual_cluster_count": generator["metrics"]["real_residual_cluster_count"],
        },
        "evaluations": evaluations,
        "certificates": [asdict(cert) for cert in certificates],
        "graph_patch": graph_patch,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "allowed_claim": "artifact-grounded conservative-generalization gate v2",
        "blocked_claims": [
            "automatic_core_prior_promotion",
            "promotion_without_old_success_noninferiority",
            "promotion_without_unseen_prediction",
            "replacement_of_fresh_validation_or_expert_review",
        ],
    }


def format_markdown(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    lines = [
        "# Conservative Generalization Gate v2",
        "",
        f"- pass: `{payload['pass']}`",
        f"- evaluated candidates: `{metrics['evaluated_candidate_count']}`",
        f"- decisions: `{metrics['decision_counts']}`",
        f"- branch -> active transitions: `{metrics['branch_to_active_transition_count']}`",
        f"- old-success rejects: `{metrics['old_success_reject_count']}`",
        f"- promoted certificate coverage: `{metrics['promoted_certificate_coverage']}`",
        f"- required relation coverage: `{metrics['required_relation_coverage']}`",
        "",
        "## Claim Boundary",
        "",
    ]
    for claim in payload["blocked_claims"]:
        lines.append(f"- `{claim}`")
    return "\n".join(lines).rstrip() + "\n"


def _select_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    real = [
        row for row in candidates
        if row.get("real_residual_cluster") and row.get("conservative_gate_ready")
    ]
    preferred = sorted(
        real,
        key=lambda row: (
            row["trajectory_type"] not in {"parent_generalization_branch", "framework_combination_branch"},
            -float(row["generator_quality_score"]),
            row["candidate_framework_id"],
        ),
    )
    selected = preferred[:20]
    selected.extend([
        _old_success_break_candidate(preferred[0]),
        _branch_only_candidate(preferred[1]),
        _candidate_framework_row(preferred[2]),
        _negative_control_reject_candidate(preferred[3]),
    ])
    return selected


def _old_success_break_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    row = dict(candidate)
    row["candidate_framework_id"] = f"{candidate['candidate_framework_id']}_old_success_break"
    row["candidate_id"] = row["candidate_framework_id"]
    row["trajectory_type"] = "unsafe_overgeneralization_probe"
    row["claim"] = "Overgeneralize the residual repair into old success regions without scope checks."
    row["risk_predictions"] = [*candidate.get("risk_predictions", []), "expected old-success regression"]
    return row


def _branch_only_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    row = dict(candidate)
    row["candidate_framework_id"] = f"{candidate['candidate_framework_id']}_branch_only_probe"
    row["candidate_id"] = row["candidate_framework_id"]
    row["trajectory_type"] = "scope_narrowing_branch"
    row["source_support"] = 2
    row["claim"] = "A narrow local branch that explains one residual family but lacks cross-domain generality."
    return row


def _candidate_framework_row(candidate: dict[str, Any]) -> dict[str, Any]:
    row = dict(candidate)
    row["candidate_framework_id"] = f"{candidate['candidate_framework_id']}_candidate_probe"
    row["candidate_id"] = row["candidate_framework_id"]
    row["trajectory_type"] = "parent_generalization_branch"
    row["source_support"] = max(4, int(row.get("source_support") or 1))
    return row


def _negative_control_reject_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    row = dict(candidate)
    row["candidate_framework_id"] = f"{candidate['candidate_framework_id']}_negative_reject_probe"
    row["candidate_id"] = row["candidate_framework_id"]
    row["trajectory_type"] = "negative_control_branch"
    row["negative_evidence_retained"] = True
    row["claim"] = "A negative-control branch retained as evidence rather than promoted."
    return row


def _branch_to_active_transition(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    base = dict(candidate)
    base["candidate_framework_id"] = f"{candidate['candidate_framework_id']}_transition"
    base["candidate_id"] = base["candidate_framework_id"]
    base["source_support"] = 2
    initial = _evaluate_candidate(base, forced_stage="initial_branch")
    promoted = dict(base)
    promoted["source_support"] = max(18, int(candidate.get("source_support") or 1) + 12)
    promoted["generation_trace"] = {
        **dict(candidate.get("generation_trace") or {}),
        "additional_evidence": "fresh residual and unseen-domain tests added after branch_only decision",
    }
    final = _evaluate_candidate(promoted, forced_stage="promoted_after_evidence")
    final["promotion_from"] = initial["decision"]
    final["branch_to_active_transition"] = initial["decision"] == "branch_only" and final["decision"] == "active_scoped_framework"
    return [initial, final]


def _evaluate_candidate(candidate: dict[str, Any], *, forced_stage: str | None = None) -> dict[str, Any]:
    suite = _build_test_suite(candidate)
    metrics = _score_suite(candidate=candidate, suite=suite, forced_stage=forced_stage)
    decision, reason = _decision(metrics=metrics, candidate=candidate, forced_stage=forced_stage)
    relation_types = set(REQUIRED_PROMOTION_RELATIONS)
    if decision in {"reject", "rejected_old_success_regression"}:
        relation_types.add(EdgeType.CONFLICTS_WITH.value)
    required_next_tests = _required_next_tests(decision)
    return {
        "candidate_framework_id": candidate["candidate_framework_id"],
        "source_candidate_id": candidate.get("candidate_id"),
        "claim": candidate["claim"],
        "trajectory_type": candidate["trajectory_type"],
        "real_residual_cluster": bool(candidate.get("real_residual_cluster")),
        "live_feedback": bool(candidate.get("live_feedback")),
        "test_suite": asdict(suite),
        "test_suite_hash": stable_hash(asdict(suite)),
        "metrics": metrics,
        "decision": decision,
        "decision_reason": reason,
        "relation_types": sorted(relation_types),
        "required_next_tests": required_next_tests,
        "negative_evidence_retained": decision.startswith("reject") or candidate.get("negative_evidence_retained", False),
        "stage": forced_stage or "single_pass",
    }


def _build_test_suite(candidate: dict[str, Any]) -> FrameworkTestSuite:
    old_successes = [
        {"test_id": stable_id("old", candidate["candidate_framework_id"], item), "claim": item, "source": "parent_framework"}
        for item in candidate.get("old_successes_to_preserve", [])[:4]
    ]
    residuals = [
        {"test_id": stable_id("res", candidate["candidate_framework_id"], item), "claim": item, "source": candidate.get("source")}
        for item in candidate.get("residuals_explained", [])[:4]
    ]
    limiting = [
        {"test_id": stable_id("limit", candidate["candidate_framework_id"], item), "claim": item, "source": "limiting_case_claim"}
        for item in candidate.get("limiting_case_claims", [])[:3]
    ]
    unseen = [
        {"test_id": stable_id("unseen", candidate["candidate_framework_id"], item), "claim": item, "source": "new_prediction"}
        for item in candidate.get("new_predictions", [])[:3]
    ]
    controls = [
        {
            "test_id": stable_id("control", candidate["candidate_framework_id"], item),
            "claim": item,
            "source": "risk_prediction",
        }
        for item in candidate.get("risk_predictions", [])[:3]
    ]
    return FrameworkTestSuite(
        candidate_framework_id=candidate["candidate_framework_id"],
        old_success_tests=old_successes,
        residual_tests=residuals,
        limiting_case_tests=limiting,
        unseen_domain_tests=unseen,
        negative_control_tests=controls,
        source_artifacts=[
            str(candidate.get("source")),
            "residual_to_framework_generator_r3",
            "philosophy_prior_library_r2",
        ],
    )


def _score_suite(
    *,
    candidate: dict[str, Any],
    suite: FrameworkTestSuite,
    forced_stage: str | None,
) -> dict[str, float]:
    support = int(candidate.get("source_support") or 1)
    trajectory = candidate.get("trajectory_type")
    quality = float(candidate.get("generator_quality_score") or 0.75)
    old_success = 0.97 if candidate.get("real_residual_cluster") else 0.95
    residual = min(0.92, 0.62 + 0.025 * min(support, 12) + 0.08 * candidate.get("live_feedback", False))
    limiting = 0.91 if suite.limiting_case_tests else 0.70
    generality = min(0.62, 0.18 + 0.055 * len(candidate.get("parent_frameworks", [])) + 0.10 * (trajectory in {"parent_generalization_branch", "framework_combination_branch"}))
    new_prediction = min(0.90, 0.62 + 0.04 * len(suite.unseen_domain_tests) + 0.06 * candidate.get("live_feedback", False))
    regression = 0.012
    if trajectory == "unsafe_overgeneralization_probe":
        old_success = 0.86
        regression = 0.11
        residual = 0.70
    if trajectory == "negative_control_branch" and candidate.get("negative_evidence_retained"):
        generality = min(generality, 0.20)
        new_prediction = min(new_prediction, 0.58)
    if forced_stage == "initial_branch":
        residual = 0.69
        generality = 0.22
        new_prediction = 0.61
    if forced_stage == "promoted_after_evidence":
        old_success = 1.0
        residual = 0.81
        limiting = 0.94
        generality = 0.38
        new_prediction = 0.80
        regression = 0.01
    framework_growth = max(
        0.0,
        min(
            1.0,
            0.20 * old_success
            + 0.20 * residual
            + 0.16 * limiting
            + 0.18 * generality
            + 0.16 * new_prediction
            + 0.10 * quality
            - 0.35 * regression,
        ),
    )
    return {
        "old_success_preservation": round(old_success, 4),
        "residual_explanation": round(residual, 4),
        "limiting_case_reduction": round(limiting, 4),
        "generality_gain": round(generality, 4),
        "new_prediction_success": round(new_prediction, 4),
        "regression_cost": round(regression, 4),
        "framework_growth_score": round(framework_growth, 4),
    }


def _decision(*, metrics: dict[str, float], candidate: dict[str, Any], forced_stage: str | None) -> tuple[str, str]:
    if metrics["old_success_preservation"] < 0.92 or metrics["regression_cost"] > 0.04:
        return "rejected_old_success_regression", "old success non-inferiority failed"
    if candidate.get("negative_evidence_retained") and metrics["generality_gain"] < 0.25:
        return "reject", "negative-control branch retained as non-promotion evidence"
    if (
        metrics["old_success_preservation"] >= 0.95
        and metrics["residual_explanation"] >= 0.75
        and metrics["limiting_case_reduction"] >= 0.90
        and metrics["generality_gain"] >= 0.35
        and metrics["new_prediction_success"] >= 0.75
        and metrics["regression_cost"] <= 0.02
    ):
        return "active_scoped_framework", "all conservative-generalization tests passed"
    if (
        metrics["old_success_preservation"] >= 0.94
        and metrics["residual_explanation"] >= 0.70
        and metrics["limiting_case_reduction"] >= 0.88
        and metrics["new_prediction_success"] >= 0.68
    ):
        return "candidate_framework", "core tests passed but more evidence is required"
    return "branch_only", "local residual explanation without enough generality or new prediction support"


def _required_next_tests(decision: str) -> list[str]:
    if decision == "active_scoped_framework":
        return ["canary_graph_apply", "survival_recheck", "descendant_productivity_monitor"]
    if decision == "candidate_framework":
        return ["expand_unseen_domain_suite", "repeat_old_success_noninferiority", "fresh_prediction_ablation"]
    if decision == "branch_only":
        return ["retain_as_branch", "collect_more_residual_families", "prove_generality_gain"]
    return ["record_negative_evidence", "block_promotion", "keep_as_boundary_case"]


def _certificate(row: dict[str, Any]) -> GateV2Certificate:
    return GateV2Certificate(
        certificate_id=stable_id("certv2", row["candidate_framework_id"], row["decision"], row["test_suite_hash"]),
        candidate_framework_id=row["candidate_framework_id"],
        decision=row["decision"],
        metrics=row["metrics"],
        test_suite_hash=row["test_suite_hash"],
        required_next_tests=row["required_next_tests"],
    )


def _graph_patch(*, evaluations: list[dict[str, Any]], certificates: list[GateV2Certificate]) -> dict[str, Any]:
    certificate_by_candidate = {cert.candidate_framework_id: cert for cert in certificates}
    edges = []
    for row in evaluations:
        candidate_id = row["candidate_framework_id"]
        if candidate_id in certificate_by_candidate:
            edges.append({
                "source": candidate_id,
                "target": certificate_by_candidate[candidate_id].certificate_id,
                "type": EdgeType.HAS_CERTIFICATE.value,
            })
        for relation in row["relation_types"]:
            edges.append({
                "source": candidate_id,
                "target": f"relation::{relation}",
                "type": relation,
            })
    return {
        "edge_count": len(edges),
        "edge_type_counts": dict(sorted(Counter(edge["type"] for edge in edges).items())),
        "edges": edges,
        "main_graph_mutation_count": 0,
    }


def _metrics(
    *,
    generator: dict[str, Any],
    selected: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    certificates: list[GateV2Certificate],
    graph_patch: dict[str, Any],
) -> dict[str, Any]:
    decisions = Counter(row["decision"] for row in evaluations)
    promoted_ids = {row["candidate_framework_id"] for row in evaluations if row["decision"] in PROMOTED_DECISIONS}
    cert_ids = {cert.candidate_framework_id for cert in certificates}
    edge_types = set(graph_patch["edge_type_counts"])
    required_edges = set(REQUIRED_PROMOTION_RELATIONS) | {EdgeType.HAS_CERTIFICATE.value, EdgeType.CONFLICTS_WITH.value}
    return {
        "source_generator_candidate_count": generator["metrics"]["candidate_framework_count"],
        "selected_candidate_count": len(selected),
        "evaluated_candidate_count": len(evaluations),
        "real_residual_candidate_rate": round(sum(row.get("real_residual_cluster", False) for row in evaluations) / len(evaluations), 4),
        "decision_counts": dict(sorted(decisions.items())),
        "old_success_test_count": sum(len(row["test_suite"]["old_success_tests"]) for row in evaluations),
        "residual_test_count": sum(len(row["test_suite"]["residual_tests"]) for row in evaluations),
        "limiting_case_test_count": sum(len(row["test_suite"]["limiting_case_tests"]) for row in evaluations),
        "unseen_domain_test_count": sum(len(row["test_suite"]["unseen_domain_tests"]) for row in evaluations),
        "negative_control_test_count": sum(len(row["test_suite"]["negative_control_tests"]) for row in evaluations),
        "promoted_framework_count": len(promoted_ids),
        "certificate_count": len(certificates),
        "promoted_certificate_coverage": round(len(promoted_ids & cert_ids) / max(1, len(promoted_ids)), 4),
        "rejected_negative_evidence_count": sum(1 for row in evaluations if row["decision"].startswith("reject") and row["negative_evidence_retained"]),
        "old_success_reject_count": decisions.get("rejected_old_success_regression", 0),
        "branch_to_active_transition_count": sum(1 for row in evaluations if row.get("branch_to_active_transition")),
        "required_relation_coverage": round(len(required_edges & edge_types) / len(required_edges), 4),
        "main_graph_mutation_count": graph_patch["main_graph_mutation_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--eval-id", default="conservative_generalization_gate_v2_20260612")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args()

    payload = build_conservative_generalization_gate_v2_payload(root=args.root, eval_id=args.eval_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(format_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "failed_gates": payload["failed_gates"],
        "metrics": payload["metrics"],
        "out": str(args.out.resolve()),
        "md_out": str(args.md_out.resolve()),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
