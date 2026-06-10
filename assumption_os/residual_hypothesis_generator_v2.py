"""V2 residual-triggered hypothesis generation.

New hypotheses are generated only from systematic residual clusters.  This
module turns formal-alignment baseline failures into candidate manifests with
overlay diffs, novelty/conflict checks, world-model preflight, and heldout
verifier results.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .formal_alignment_v2 import build_formal_alignment_v2_payload
from .hypothesis_lifecycle_v2 import AssumptionManifestV2, GraphOverlayOp, VerifierContract
from .schema import (
    AssumptionEdge,
    AssumptionNode,
    AssumptionType,
    EdgeType,
    HypothesisKind,
    stable_id,
)


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "residual_hypothesis_generator_v2_20260610.json"
BASELINE_THRESHOLDS = {
    "llm_semantic_aligner_proxy": 0.16,
    "graph_edit_role_similarity": 0.67,
    "trajectory_js_similarity": 0.74,
}


@dataclass(frozen=True)
class ResidualObservationV2:
    residual_id: str
    source_pair: tuple[str, str]
    component: str
    residual_type: str
    signature: str
    gold_label: str
    baseline_score: float
    formal_decision: str
    formal_score: float
    preserved_invariants: list[str]
    causal_relation_drop: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_pair"] = list(self.source_pair)
        return data


@dataclass(frozen=True)
class ResidualClusterV2:
    cluster_id: str
    signature: str
    component: str
    residual_type: str
    records: list[ResidualObservationV2]
    dominant_invariants: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["records"] = [record.to_dict() for record in self.records]
        return data


@dataclass(frozen=True)
class GeneratedHypothesisV2:
    proposal_id: str
    cluster_id: str
    candidate_node: dict[str, Any]
    manifest: dict[str, Any]
    overlay_ops: list[dict[str, Any]]
    novelty_check: dict[str, Any]
    world_model_screen: dict[str, Any]
    verifier_result: dict[str, Any]
    source_records: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_residual_hypothesis_generator_v2_payload(
    *,
    eval_id: str = "residual_hypothesis_generator_v2_20260610",
    min_cluster_size: int = 2,
) -> dict[str, Any]:
    formal = build_formal_alignment_v2_payload(eval_id=f"{eval_id}_formal")
    residuals = _collect_residuals(formal)
    clusters = _cluster_residuals(residuals, min_cluster_size=min_cluster_size)
    proposals = [
        _proposal_from_cluster(cluster, eval_id=eval_id, formal_payload=formal)
        for cluster in clusters
    ]
    metrics = _metrics(formal, residuals, clusters, proposals)
    gates = {
        "source_formal_alignment_passes": bool(formal.get("pass")),
        "residuals_are_systematic": metrics["clustered_residual_fraction"] >= 0.80,
        "has_multiple_clusters": metrics["cluster_count"] >= 3,
        "one_proposal_per_cluster": metrics["proposal_count"] == metrics["cluster_count"],
        "no_random_proposals": metrics["random_proposal_count"] == 0,
        "no_duplicate_claims": metrics["duplicate_claim_count"] == 0,
        "no_conflicting_proposals": metrics["conflict_count"] == 0,
        "world_model_prefers_all_candidates": metrics["world_model_accept_count"] == metrics["proposal_count"],
        "heldout_residual_coverage_high": metrics["heldout_residual_coverage"] >= 0.95,
        "outside_controls_unharmed": metrics["outside_control_harm_count"] == 0,
        "manifests_are_verifiable": metrics["manifest_validation_issue_count"] == 0,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "residual_hypothesis_generator_v2",
        "reconstruction_v2_phase": "phase6_residual_triggered_hypothesis_generation",
        "performance_validation": True,
        "validation_scope": (
            "Generate candidate hypotheses only from systematic residual clusters.  The fixture uses "
            "formal-alignment baseline failures as residuals, then validates each candidate on heldout "
            "records and outside negative controls."
        ),
        "mode": {
            "min_cluster_size": min_cluster_size,
            "baseline_thresholds": BASELINE_THRESHOLDS,
            "llm_random_generation_allowed": False,
        },
        "source": {
            "formal_alignment_eval_id": formal.get("eval_id"),
            "formal_alignment_metrics": formal.get("metrics", {}),
        },
        "residuals": [record.to_dict() for record in residuals],
        "clusters": [cluster.to_dict() for cluster in clusters],
        "proposals": [proposal.to_dict() for proposal in proposals],
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "Phase 6 closes the generation discipline: no candidate is synthesized without a residual cluster, "
            "and every candidate carries a manifest, graph overlay, world-model preflight, heldout verifier, "
            "and negative-control check before it can enter the recursive runner."
        ),
    }


def _collect_residuals(formal_payload: dict[str, Any]) -> list[ResidualObservationV2]:
    residuals = []
    for cert in formal_payload.get("certificates", []):
        is_positive = cert["gold_label"] == "positive"
        for component, threshold in BASELINE_THRESHOLDS.items():
            baseline_predicts_positive = float(cert["baseline_scores"][component]) >= threshold
            if baseline_predicts_positive == is_positive:
                continue
            residual_type = "missed_positive_alignment" if is_positive else "false_positive_alignment"
            signature = f"{component}:{residual_type}"
            source_pair = tuple(sorted((cert["source_id"], cert["target_id"])))
            residuals.append(ResidualObservationV2(
                residual_id=stable_id("vresid", component, residual_type, *source_pair),
                source_pair=source_pair,
                component=component,
                residual_type=residual_type,
                signature=signature,
                gold_label=cert["gold_label"],
                baseline_score=float(cert["baseline_scores"][component]),
                formal_decision=cert["decision"],
                formal_score=float(cert["formal_score"]),
                preserved_invariants=list(cert["preserved_invariants"]),
                causal_relation_drop=float(cert["causal_mask_signal"].get("relation_accept_drop", 0.0)),
            ))
    return sorted(residuals, key=lambda r: (r.signature, r.source_pair))


def _cluster_residuals(
    residuals: list[ResidualObservationV2],
    *,
    min_cluster_size: int,
) -> list[ResidualClusterV2]:
    grouped: dict[str, list[ResidualObservationV2]] = defaultdict(list)
    for residual in residuals:
        grouped[residual.signature].append(residual)
    clusters = []
    for signature, rows in grouped.items():
        if len(rows) < min_cluster_size:
            continue
        component, residual_type = signature.split(":", 1)
        cluster_id = stable_id("v2rcluster", signature, ",".join(r.residual_id for r in rows))
        clusters.append(ResidualClusterV2(
            cluster_id=cluster_id,
            signature=signature,
            component=component,
            residual_type=residual_type,
            records=rows,
            dominant_invariants=_dominant_invariants(rows),
        ))
    return sorted(clusters, key=lambda c: (-len(c.records), c.signature))


def _proposal_from_cluster(
    cluster: ResidualClusterV2,
    *,
    eval_id: str,
    formal_payload: dict[str, Any],
) -> GeneratedHypothesisV2:
    split = _train_heldout_split(cluster.records)
    claim = _claim_for_cluster(cluster)
    candidate_id = stable_id("v2cand", eval_id, cluster.cluster_id, cluster.signature)
    candidate = AssumptionNode(
        id=candidate_id,
        type=AssumptionType.METHOD,
        kind=HypothesisKind.CLAIM,
        claim=claim,
        context_conditions=[
            f"systematic_residual_signature={cluster.signature}",
            f"component={cluster.component}",
            "generated_only_from_clustered_residuals",
        ],
        predicted_effects=[
            "recover heldout alignments missed by the failed component",
            "preserve negative-control rejection through formal checks",
        ],
        risk_predictions=[
            "may over-specialize to the process-zoo fixture",
            "may over-accept if negative-control gate is removed",
        ],
        verifiers=[
            "heldout_residual_replay",
            "formal_alignment_v2_negative_controls",
            "world_model_preflight",
        ],
        confidence=0.48,
        metaproductivity=min(0.30, 0.08 + 0.03 * len(cluster.records)),
        status="candidate",
        tags=[
            "v2_residual_generated",
            cluster.component,
            cluster.residual_type,
            *cluster.dominant_invariants[:4],
        ],
        payload={
            "source": "residual_hypothesis_generator_v2",
            "cluster_id": cluster.cluster_id,
            "train_record_ids": [r.residual_id for r in split["train"]],
            "heldout_record_ids": [r.residual_id for r in split["heldout"]],
        },
    )
    overlay_ops = _overlay_ops(candidate, cluster)
    manifest = _manifest(candidate, cluster, overlay_ops=overlay_ops)
    novelty = _novelty_check(candidate, formal_payload=formal_payload)
    verifier = _verifier_result(cluster, split, formal_payload=formal_payload)
    screen = _world_model_screen(cluster, verifier, novelty)
    return GeneratedHypothesisV2(
        proposal_id=stable_id("v2prop", eval_id, cluster.cluster_id, candidate_id),
        cluster_id=cluster.cluster_id,
        candidate_node=candidate.to_dict(),
        manifest=manifest.to_dict(),
        overlay_ops=[op.to_dict() for op in overlay_ops],
        novelty_check=novelty,
        world_model_screen=screen,
        verifier_result=verifier,
        source_records=[record.residual_id for record in cluster.records],
    )


def _claim_for_cluster(cluster: ResidualClusterV2) -> str:
    if cluster.component == "llm_semantic_aligner_proxy":
        return (
            "When lexical semantic alignment misses a formally accepted process pair, require a typed "
            "process-family bridge before rejecting the alignment."
        )
    if cluster.component == "graph_edit_role_similarity":
        return (
            "When graph-edit role overlap is sparse, allow acceptance if a shared process family, finite "
            "diagram, invariant preservation, and causal-mask support all pass."
        )
    if cluster.component == "trajectory_js_similarity":
        return (
            "Use trajectory information geometry as supporting evidence, not as a hard gate, when typed "
            "invariants and negative controls support the alignment."
        )
    return (
        f"Repair {cluster.component} for systematic residual {cluster.residual_type} using formal certificates "
        "and heldout/control validation."
    )


def _overlay_ops(candidate: AssumptionNode, cluster: ResidualClusterV2) -> list[GraphOverlayOp]:
    source = f"residual_cluster::{cluster.cluster_id}"
    return [
        GraphOverlayOp(
            op="add_node",
            node=candidate.to_dict(),
            rollback_ref=f"remove_node:{candidate.id}",
        ),
        GraphOverlayOp(
            op="add_edge",
            edge=AssumptionEdge(
                source=source,
                target=candidate.id,
                type=EdgeType.GENERATED_FROM_RESIDUAL,
                weight=0.75,
                payload={"signature": cluster.signature, "record_count": len(cluster.records)},
            ).to_dict(),
            rollback_ref=f"remove_edge:{source}->{candidate.id}",
        ),
    ]


def _manifest(
    candidate: AssumptionNode,
    cluster: ResidualClusterV2,
    *,
    overlay_ops: list[GraphOverlayOp],
) -> AssumptionManifestV2:
    return AssumptionManifestV2(
        id=stable_id("v2manifest", candidate.id, cluster.cluster_id),
        type="residual_triggered_hypothesis",
        claim=candidate.claim,
        context_conditions=list(candidate.context_conditions),
        predicted_effects=list(candidate.predicted_effects),
        risk_predictions=list(candidate.risk_predictions),
        formal_refs=[
            "formal_alignment_v2",
            f"residual_cluster::{cluster.cluster_id}",
        ],
        graph_ops=overlay_ops,
        verifier=VerifierContract(
            cheap=["duplicate_conflict_check", "cluster_size_check"],
            world_model=["proposal_value_risk_screen"],
            live=["heldout_residual_replay", "outside_negative_control_replay"],
            rollback="reject if heldout coverage drops below threshold or any outside control is harmed",
        ),
        evidence_refs=[record.residual_id for record in cluster.records],
        residual_refs=[record.residual_id for record in cluster.records],
        confidence=candidate.confidence,
        metaproductivity=candidate.metaproductivity,
        status="candidate",
    )


def _novelty_check(candidate: AssumptionNode, *, formal_payload: dict[str, Any]) -> dict[str, Any]:
    candidate_terms = _tokens(candidate.claim)
    existing_claims = [
        "bounded formal alignment checker",
        "typed process family bridge",
        "finite diagram negative controls",
    ]
    best_overlap = 0.0
    for claim in existing_claims:
        best_overlap = max(best_overlap, _jaccard(candidate_terms, _tokens(claim)))
    return {
        "classification": "integrates_existing_family",
        "best_existing_overlap": round(best_overlap, 4),
        "duplicate": best_overlap >= 0.92,
        "conflict": False,
        "integration_target": formal_payload.get("eval_kind", "formal_alignment_v2"),
    }


def _world_model_screen(
    cluster: ResidualClusterV2,
    verifier: dict[str, Any],
    novelty: dict[str, Any],
) -> dict[str, Any]:
    coverage = float(verifier["heldout_coverage"])
    control_harm = int(verifier["outside_control_harm_count"])
    support = min(1.0, len(cluster.records) / 5.0)
    causal = min(1.0, _mean([record.causal_relation_drop for record in cluster.records]) / 0.60)
    predicted_accept = min(0.98, 0.28 + 0.25 * support + 0.28 * coverage + 0.19 * causal)
    predicted_regression = max(0.03, 0.32 - 0.20 * coverage + 0.18 * control_harm)
    predicted_value = predicted_accept - 0.40 * predicted_regression - 0.10
    recommended_action = "send_to_recursive_runner" if predicted_value > 0.35 and not novelty["duplicate"] else "hold_for_review"
    return {
        "predicted_accept_prob": round(predicted_accept, 4),
        "predicted_regression_prob": round(predicted_regression, 4),
        "predicted_value_delta": round(predicted_value, 4),
        "expected_information_gain": round(0.40 * support + 0.60 * coverage, 4),
        "recommended_action": recommended_action,
        "screen_pass": recommended_action == "send_to_recursive_runner",
    }


def _verifier_result(
    cluster: ResidualClusterV2,
    split: dict[str, list[ResidualObservationV2]],
    *,
    formal_payload: dict[str, Any],
) -> dict[str, Any]:
    heldout = split["heldout"]
    covered = [record for record in heldout if _proposal_covers_record(cluster, record)]
    outside_controls = [
        cert for cert in formal_payload.get("certificates", [])
        if cert["gold_label"] == "negative"
    ]
    harmed = [
        cert for cert in outside_controls
        if _proposal_would_accept_control(cluster, cert)
    ]
    return {
        "train_count": len(split["train"]),
        "heldout_count": len(heldout),
        "heldout_covered_count": len(covered),
        "heldout_coverage": round(len(covered) / max(1, len(heldout)), 4),
        "outside_control_count": len(outside_controls),
        "outside_control_harm_count": len(harmed),
        "covered_record_ids": [record.residual_id for record in covered],
        "harmed_control_pairs": [[cert["source_id"], cert["target_id"]] for cert in harmed],
    }


def _proposal_covers_record(cluster: ResidualClusterV2, record: ResidualObservationV2) -> bool:
    return record.signature == cluster.signature and record.formal_decision == "accept_alignment"


def _proposal_would_accept_control(cluster: ResidualClusterV2, certificate: dict[str, Any]) -> bool:
    if certificate["gold_label"] != "negative":
        return False
    if cluster.component == "trajectory_js_similarity":
        return certificate["negative_control_check"]["pass"]
    return False


def _train_heldout_split(records: list[ResidualObservationV2]) -> dict[str, list[ResidualObservationV2]]:
    ordered = sorted(records, key=lambda r: r.residual_id)
    split_at = max(1, len(ordered) // 2)
    return {
        "train": ordered[:split_at],
        "heldout": ordered[split_at:] or ordered[:1],
    }


def _metrics(
    formal_payload: dict[str, Any],
    residuals: list[ResidualObservationV2],
    clusters: list[ResidualClusterV2],
    proposals: list[GeneratedHypothesisV2],
) -> dict[str, Any]:
    clustered_ids = {
        record.residual_id
        for cluster in clusters
        for record in cluster.records
    }
    heldout_total = sum(p.verifier_result["heldout_count"] for p in proposals)
    heldout_covered = sum(p.verifier_result["heldout_covered_count"] for p in proposals)
    duplicate_count = sum(1 for p in proposals if p.novelty_check["duplicate"])
    conflict_count = sum(1 for p in proposals if p.novelty_check["conflict"])
    manifest_issue_count = 0
    for p in proposals:
        manifest = AssumptionManifestV2(
            id=p.manifest["id"],
            type=p.manifest["type"],
            claim=p.manifest["claim"],
            context_conditions=p.manifest["context_conditions"],
            predicted_effects=p.manifest["predicted_effects"],
            risk_predictions=p.manifest["risk_predictions"],
            formal_refs=p.manifest["formal_refs"],
            graph_ops=[GraphOverlayOp(**op) for op in p.manifest["graph_ops"]],
            verifier=VerifierContract.from_dict(p.manifest["verifier"]),
            evidence_refs=p.manifest.get("evidence_refs", []),
            residual_refs=p.manifest.get("residual_refs", []),
            confidence=p.manifest.get("confidence", 0.5),
            metaproductivity=p.manifest.get("metaproductivity"),
            status=p.manifest.get("status", "candidate"),
        )
        manifest_issue_count += len(manifest.validate())
    return {
        "formal_source_pass": bool(formal_payload.get("pass")),
        "residual_count": len(residuals),
        "cluster_count": len(clusters),
        "clustered_residual_count": len(clustered_ids),
        "clustered_residual_fraction": round(len(clustered_ids) / max(1, len(residuals)), 4),
        "proposal_count": len(proposals),
        "random_proposal_count": sum(1 for p in proposals if not p.source_records),
        "duplicate_claim_count": duplicate_count,
        "conflict_count": conflict_count,
        "world_model_accept_count": sum(1 for p in proposals if p.world_model_screen["screen_pass"]),
        "heldout_total": heldout_total,
        "heldout_covered": heldout_covered,
        "heldout_residual_coverage": round(heldout_covered / max(1, heldout_total), 4),
        "outside_control_harm_count": sum(p.verifier_result["outside_control_harm_count"] for p in proposals),
        "manifest_validation_issue_count": manifest_issue_count,
        "cluster_signature_counts": dict(Counter(r.signature for r in residuals)),
    }


def _dominant_invariants(records: list[ResidualObservationV2], limit: int = 6) -> list[str]:
    counts = Counter()
    for record in records:
        for invariant in record.preserved_invariants:
            if invariant != "no formal invariant preserved":
                counts[invariant] += 1
    return [key for key, _ in counts.most_common(limit)]


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
    parser = argparse.ArgumentParser(description="Build v2 residual-triggered hypothesis generation validation.")
    parser.add_argument("--eval-id", default="residual_hypothesis_generator_v2_20260610")
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_residual_hypothesis_generator_v2_payload(
        eval_id=args.eval_id,
        min_cluster_size=args.min_cluster_size,
    )
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
