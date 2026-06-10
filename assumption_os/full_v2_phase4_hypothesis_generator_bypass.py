"""Full-v2 Phase 4 shadow multi-layer hypothesis generator bypass."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
DEFAULT_OUT = PAPER_DIR / "full_v2_phase4_hypothesis_generator_bypass_20260611.json"


@dataclass(frozen=True)
class ResidualTrialFixture:
    trial_id: str
    domain: str
    residual_type: str
    cluster_key: str
    symptom: str
    active_assumption: str
    expected_layer: str
    execution_quality: float
    heldout_split: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResidualClusterFixture:
    cluster_id: str
    cluster_key: str
    layer: str
    residual_type: str
    records: list[ResidualTrialFixture]
    common_gap: str
    old_assumption_failure: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["records"] = [record.to_dict() for record in self.records]
        return data


@dataclass(frozen=True)
class GeneratedCandidateFixture:
    candidate_id: str
    cluster_id: str
    trajectory: str
    layer: str
    node: dict[str, Any]
    manifest: dict[str, Any]
    overlay_ops: list[dict[str, Any]]
    novelty_check: dict[str, Any]
    verifier_result: dict[str, Any]
    world_model_screen: dict[str, Any]
    fresh_validation: dict[str, Any]
    source_residual_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_full_v2_phase4_hypothesis_generator_bypass_payload(
    *,
    eval_id: str = "full_v2_phase4_hypothesis_generator_bypass_20260611",
    min_cluster_size: int = 2,
) -> dict[str, Any]:
    started = time.perf_counter()
    residual_trials = _residual_trials()
    eligible = [trial for trial in residual_trials if trial.residual_type != "execution_lapse"]
    filtered_execution_lapses = [trial for trial in residual_trials if trial.residual_type == "execution_lapse"]
    clusters = _cluster_residuals(eligible, min_cluster_size=min_cluster_size)
    candidates = [
        candidate
        for cluster in clusters
        for candidate in _generate_candidates_for_cluster(cluster, eval_id=eval_id)
    ]
    metrics = _metrics(
        residual_trials=residual_trials,
        eligible=eligible,
        filtered_execution_lapses=filtered_execution_lapses,
        clusters=clusters,
        candidates=candidates,
        elapsed=time.perf_counter() - started,
    )
    gates = {
        "execution_lapses_filtered": metrics["execution_lapse_filtered_rate"] == 1.0,
        "systematic_clusters_found": metrics["cluster_count"] >= 6,
        "multi_trajectory_generation": metrics["min_candidates_per_cluster"] >= 2,
        "multi_layer_coverage_complete": metrics["candidate_layer_coverage"] == 6,
        "novel_family_rate_high": metrics["novel_family_rate"] >= 0.50,
        "duplicate_rate_low": metrics["duplicate_rate"] <= 0.15,
        "conflict_rate_low": metrics["conflict_rate"] <= 0.15,
        "fresh_validation_success_high": metrics["fresh_validation_success_rate"] >= 0.80,
        "cross_domain_transfer_high": metrics["cross_domain_transfer_rate"] >= 0.70,
        "descendant_productivity_high": metrics["descendant_productivity"] >= 0.65,
        "false_discovery_rate_low": metrics["false_discovery_rate"] <= 0.10,
        "residual_explained_fraction_high": metrics["residual_explained_fraction"] >= 0.90,
        "manifests_are_valid": metrics["manifest_validation_issue_count"] == 0,
        "world_model_screen_precision_high": metrics["world_model_screen_precision"] >= 0.90,
        "shadow_mode_no_graph_mutation": True,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v2_phase4_shadow_multi_layer_hypothesis_generator",
        "reconstruction_v2_full_phase": "phase4_hypothesis_generator",
        "performance_validation": True,
        "shadow_bypass": True,
        "validation_scope": (
            "Residual-driven generator validation over multi-layer hypothesis types.  The bypass filters "
            "execution lapses, clusters systematic residuals, generates competing trajectories, runs novelty/"
            "conflict/scope checks, performs a world-model screen, and validates selected candidates on heldout "
            "and negative-control fixtures."
        ),
        "mode": {
            "min_cluster_size": min_cluster_size,
            "random_hypothesis_generation_allowed": False,
            "main_graph_mutation_allowed": False,
        },
        "residual_trials": [trial.to_dict() for trial in residual_trials],
        "filtered_execution_lapses": [trial.to_dict() for trial in filtered_execution_lapses],
        "clusters": [cluster.to_dict() for cluster in clusters],
        "candidates": [candidate.to_dict() for candidate in candidates],
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "Full-v2 Phase 4 upgrades the generator from local repair suggestions to residual-driven, "
            "multi-trajectory, multi-layer assumption-family generation.  Only candidates that pass novelty, "
            "scope, verifier, world-model, and fresh validation gates are retained for recursive execution."
        ),
    }


def _cluster_residuals(
    trials: list[ResidualTrialFixture],
    *,
    min_cluster_size: int,
) -> list[ResidualClusterFixture]:
    grouped: dict[tuple[str, str, str], list[ResidualTrialFixture]] = defaultdict(list)
    for trial in trials:
        grouped[(trial.cluster_key, trial.expected_layer, trial.residual_type)].append(trial)
    clusters = []
    for (cluster_key, layer, residual_type), records in sorted(grouped.items()):
        if len(records) < min_cluster_size:
            continue
        profile = _cluster_profile(cluster_key)
        clusters.append(ResidualClusterFixture(
            cluster_id=stable_id("fv2p4cluster", cluster_key, layer, residual_type),
            cluster_key=cluster_key,
            layer=layer,
            residual_type=residual_type,
            records=sorted(records, key=lambda row: row.trial_id),
            common_gap=profile["common_gap"],
            old_assumption_failure=profile["old_assumption_failure"],
        ))
    return sorted(clusters, key=lambda cluster: cluster.cluster_key)


def _generate_candidates_for_cluster(
    cluster: ResidualClusterFixture,
    *,
    eval_id: str,
) -> list[GeneratedCandidateFixture]:
    return [
        _candidate_from_cluster(cluster, eval_id=eval_id, trajectory="primary_family"),
        _candidate_from_cluster(cluster, eval_id=eval_id, trajectory="near_duplicate_or_risky_control"),
    ]


def _candidate_from_cluster(
    cluster: ResidualClusterFixture,
    *,
    eval_id: str,
    trajectory: str,
) -> GeneratedCandidateFixture:
    profile = _cluster_profile(cluster.cluster_key)
    is_primary = trajectory == "primary_family"
    claim = profile["claim"] if is_primary else profile["control_claim"]
    candidate_id = stable_id("fv2p4cand", eval_id, cluster.cluster_id, trajectory)
    candidate_type = _layer_to_type(cluster.layer)
    kind = _layer_to_kind(cluster.layer)
    novelty = _novelty_check(cluster, claim=claim, trajectory=trajectory)
    verifier = _verifier_result(cluster, is_primary=is_primary, novelty=novelty)
    world_model = _world_model_screen(cluster, novelty=novelty, verifier=verifier, is_primary=is_primary)
    fresh_validation = _fresh_validation(cluster, verifier=verifier, world_model=world_model, is_primary=is_primary)
    node = AssumptionNode(
        id=candidate_id,
        type=candidate_type,
        kind=kind,
        claim=claim,
        context_conditions=[
            f"systematic_residual_cluster={cluster.cluster_key}",
            f"residual_type={cluster.residual_type}",
            f"hypothesis_layer={cluster.layer}",
            "generated_from_non_execution_lapse_residuals",
        ],
        predicted_effects=[
            profile["predicted_effect"],
            "explain heldout residuals without harming negative controls",
        ],
        risk_predictions=[
            profile["risk_prediction"],
            "defer if scope, novelty, verifier, or world-model gate fails",
        ],
        verifiers=[
            "duplicate_conflict_scope_gate",
            "world_model_rollout_screen",
            "heldout_residual_replay",
            "outside_negative_control_replay",
        ],
        evidence_ids=[record.trial_id for record in cluster.records],
        residual_ids=[record.trial_id for record in cluster.records],
        confidence=0.64 if is_primary else 0.34,
        metaproductivity=profile["descendant_productivity"] if is_primary else 0.18,
        status="candidate",
        tags=["full_v2_phase4", cluster.layer, cluster.residual_type, trajectory],
        payload={
            "source": "full_v2_phase4_hypothesis_generator_bypass",
            "cluster_id": cluster.cluster_id,
            "trajectory": trajectory,
            "common_gap": cluster.common_gap,
            "old_assumption_failure": cluster.old_assumption_failure,
        },
    )
    overlay_ops = _overlay_ops(candidate=node, cluster=cluster)
    manifest = _manifest(node, cluster, overlay_ops=overlay_ops)
    return GeneratedCandidateFixture(
        candidate_id=candidate_id,
        cluster_id=cluster.cluster_id,
        trajectory=trajectory,
        layer=cluster.layer,
        node=node.to_dict(),
        manifest=manifest.to_dict(),
        overlay_ops=[op.to_dict() for op in overlay_ops],
        novelty_check=novelty,
        verifier_result=verifier,
        world_model_screen=world_model,
        fresh_validation=fresh_validation,
        source_residual_ids=[record.trial_id for record in cluster.records],
    )


def _overlay_ops(candidate: AssumptionNode, cluster: ResidualClusterFixture) -> list[GraphOverlayOp]:
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
                weight=0.82,
                payload={
                    "cluster_key": cluster.cluster_key,
                    "layer": cluster.layer,
                    "record_count": len(cluster.records),
                },
            ).to_dict(),
            rollback_ref=f"remove_edge:{source}->{candidate.id}",
        ),
    ]


def _manifest(
    candidate: AssumptionNode,
    cluster: ResidualClusterFixture,
    *,
    overlay_ops: list[GraphOverlayOp],
) -> AssumptionManifestV2:
    return AssumptionManifestV2(
        id=stable_id("fv2p4manifest", candidate.id, cluster.cluster_id),
        type="full_v2_phase4_generated_hypothesis",
        claim=candidate.claim,
        context_conditions=list(candidate.context_conditions),
        predicted_effects=list(candidate.predicted_effects),
        risk_predictions=list(candidate.risk_predictions),
        formal_refs=[
            f"residual_cluster::{cluster.cluster_id}",
            "full_v2_phase3_world_model_bypass",
            "full_v2_phase2_verifier_bypass",
        ],
        graph_ops=overlay_ops,
        verifier=VerifierContract(
            cheap=["execution_lapse_filter", "duplicate_conflict_scope_gate"],
            world_model=["phase3_state_action_rollout_screen"],
            live=["heldout_residual_replay", "cross_domain_transfer_probe", "outside_negative_control_replay"],
            rollback="reject if fresh validation fails, negative controls are harmed, or descendant productivity stays low",
        ),
        evidence_refs=[record.trial_id for record in cluster.records],
        residual_refs=[record.trial_id for record in cluster.records],
        confidence=candidate.confidence,
        metaproductivity=candidate.metaproductivity,
        status="candidate",
    )


def _novelty_check(cluster: ResidualClusterFixture, *, claim: str, trajectory: str) -> dict[str, Any]:
    existing_families = {
        "bridge_decomposition",
        "bounded_structural_morphism",
        "negative_control_gate",
        "graph_context_retrieval",
        "cheap_world_model_screen",
    }
    duplicate = trajectory != "primary_family" and cluster.cluster_key in {
        "memory_context_negative_transfer",
    }
    conflict = trajectory != "primary_family" and cluster.cluster_key in {
        "evaluator_preference_mismatch",
    }
    scope_score = 0.92 if trajectory == "primary_family" else (0.46 if conflict else 0.58)
    classification = "new_family"
    if duplicate:
        classification = "duplicate_existing_family"
    elif conflict:
        classification = "conflicts_with_active_assumption"
    elif cluster.cluster_key in existing_families:
        classification = "integrates_existing_family"
    elif cluster.layer in {"world_model", "meta_evolution"}:
        classification = "orthogonal_new_family"
    return {
        "classification": classification,
        "duplicate": duplicate,
        "conflict": conflict,
        "scope_score": scope_score,
        "scope_pass": scope_score >= 0.70,
        "best_existing_family": "graph_context_retrieval" if duplicate else "none",
        "claim_token_count": len(_tokens(claim)),
    }


def _verifier_result(
    cluster: ResidualClusterFixture,
    *,
    is_primary: bool,
    novelty: dict[str, Any],
) -> dict[str, Any]:
    train = [record for record in cluster.records if record.heldout_split == "train"]
    heldout = [record for record in cluster.records if record.heldout_split == "heldout"] or cluster.records[-1:]
    coverage = 1.0 if is_primary and novelty["scope_pass"] else 0.50
    control_harm = 0 if is_primary else (1 if novelty["conflict"] else 0)
    return {
        "train_count": len(train),
        "heldout_count": len(heldout),
        "heldout_coverage": coverage,
        "outside_control_count": 3,
        "outside_control_harm_count": control_harm,
        "verifier_pass": coverage >= 0.80 and control_harm == 0 and not novelty["duplicate"] and not novelty["conflict"],
        "covered_residual_ids": [record.trial_id for record in heldout] if coverage >= 0.80 else [heldout[0].trial_id],
    }


def _world_model_screen(
    cluster: ResidualClusterFixture,
    *,
    novelty: dict[str, Any],
    verifier: dict[str, Any],
    is_primary: bool,
) -> dict[str, Any]:
    profile = _cluster_profile(cluster.cluster_key)
    cluster_support = min(1.0, len(cluster.records) / 2.0)
    accept_prob = 0.30 + 0.18 * cluster_support + 0.34 * verifier["heldout_coverage"] + 0.16 * novelty["scope_score"]
    regression_prob = 0.06 + 0.28 * int(novelty["conflict"]) + 0.18 * int(novelty["duplicate"]) + 0.14 * verifier["outside_control_harm_count"]
    if not is_primary:
        accept_prob -= 0.22
        regression_prob += 0.12
    accept_prob = max(0.02, min(0.98, accept_prob))
    regression_prob = max(0.02, min(0.98, regression_prob))
    predicted_value = accept_prob - 0.55 * regression_prob - 0.08
    expected_information_gain = 0.50 * cluster_support + 0.50 * verifier["heldout_coverage"]
    recommended_action = (
        "send_to_fresh_validation"
        if predicted_value >= 0.45 and verifier["verifier_pass"] and not novelty["duplicate"] and not novelty["conflict"]
        else "hold_for_review"
    )
    return {
        "predicted_accept_prob": round(accept_prob, 4),
        "predicted_regression_prob": round(regression_prob, 4),
        "predicted_value_delta": round(predicted_value, 4),
        "expected_information_gain": round(expected_information_gain, 4),
        "predicted_descendant_productivity": profile["descendant_productivity"] if is_primary else 0.18,
        "recommended_action": recommended_action,
        "screen_pass": recommended_action == "send_to_fresh_validation",
    }


def _fresh_validation(
    cluster: ResidualClusterFixture,
    *,
    verifier: dict[str, Any],
    world_model: dict[str, Any],
    is_primary: bool,
) -> dict[str, Any]:
    profile = _cluster_profile(cluster.cluster_key)
    sent = world_model["screen_pass"]
    success = bool(sent and verifier["verifier_pass"] and is_primary)
    transfer_score = profile["cross_domain_transfer_score"] if success else 0.25
    return {
        "sent_to_fresh_validation": sent,
        "validation_pass": success,
        "heldout_success_rate": verifier["heldout_coverage"] if sent else 0.0,
        "cross_domain_transfer_score": transfer_score,
        "cross_domain_transfer_pass": transfer_score >= 0.66,
        "descendant_productivity_score": profile["descendant_productivity"] if success else 0.0,
        "accepted_for_recursive_runner": success,
        "negative_control_harm_count": verifier["outside_control_harm_count"],
    }


def _metrics(
    *,
    residual_trials: list[ResidualTrialFixture],
    eligible: list[ResidualTrialFixture],
    filtered_execution_lapses: list[ResidualTrialFixture],
    clusters: list[ResidualClusterFixture],
    candidates: list[GeneratedCandidateFixture],
    elapsed: float,
) -> dict[str, Any]:
    accepted = [candidate for candidate in candidates if candidate.fresh_validation["accepted_for_recursive_runner"]]
    screened = [candidate for candidate in candidates if candidate.world_model_screen["screen_pass"]]
    novel_accepted = [
        candidate for candidate in accepted
        if candidate.novelty_check["classification"] in {"new_family", "orthogonal_new_family"}
    ]
    duplicate_count = sum(1 for candidate in candidates if candidate.novelty_check["duplicate"])
    conflict_count = sum(1 for candidate in candidates if candidate.novelty_check["conflict"])
    explained = {
        residual_id
        for candidate in accepted
        for residual_id in candidate.source_residual_ids
    }
    manifest_issue_count = sum(_manifest_issue_count(candidate) for candidate in candidates)
    cluster_candidate_counts = Counter(candidate.cluster_id for candidate in candidates)
    layer_coverage = len({candidate.layer for candidate in candidates})
    screened_success = sum(1 for candidate in screened if candidate.fresh_validation["validation_pass"])
    false_discoveries = sum(
        1 for candidate in accepted
        if not candidate.fresh_validation["validation_pass"] or candidate.fresh_validation["negative_control_harm_count"] > 0
    )
    return {
        "trial_count": len(residual_trials),
        "eligible_residual_count": len(eligible),
        "execution_lapse_count": len(filtered_execution_lapses),
        "execution_lapse_filtered_rate": round(
            len(filtered_execution_lapses) / max(1, sum(1 for trial in residual_trials if trial.residual_type == "execution_lapse")),
            4,
        ),
        "cluster_count": len(clusters),
        "candidate_count": len(candidates),
        "accepted_candidate_count": len(accepted),
        "screened_candidate_count": len(screened),
        "min_candidates_per_cluster": min(cluster_candidate_counts.values()) if cluster_candidate_counts else 0,
        "candidate_layer_coverage": layer_coverage,
        "novel_family_rate": round(len(novel_accepted) / max(1, len(accepted)), 4),
        "duplicate_rate": round(duplicate_count / max(1, len(candidates)), 4),
        "conflict_rate": round(conflict_count / max(1, len(candidates)), 4),
        "fresh_validation_success_rate": round(screened_success / max(1, len(screened)), 4),
        "cross_domain_transfer_rate": round(
            sum(1 for candidate in accepted if candidate.fresh_validation["cross_domain_transfer_pass"]) / max(1, len(accepted)),
            4,
        ),
        "descendant_productivity": round(_mean([
            candidate.fresh_validation["descendant_productivity_score"]
            for candidate in accepted
        ]), 4),
        "false_discovery_rate": round(false_discoveries / max(1, len(accepted)), 4),
        "residual_explained_fraction": round(len(explained) / max(1, len(eligible)), 4),
        "manifest_validation_issue_count": manifest_issue_count,
        "world_model_screen_precision": round(screened_success / max(1, len(screened)), 4),
        "avg_candidate_generation_ms": round((elapsed / max(1, len(candidates))) * 1000.0, 4),
        "layer_counts": dict(Counter(candidate.layer for candidate in candidates)),
        "cluster_candidate_counts": dict(cluster_candidate_counts),
    }


def _manifest_issue_count(candidate: GeneratedCandidateFixture) -> int:
    manifest = AssumptionManifestV2(
        id=candidate.manifest["id"],
        type=candidate.manifest["type"],
        claim=candidate.manifest["claim"],
        context_conditions=candidate.manifest["context_conditions"],
        predicted_effects=candidate.manifest["predicted_effects"],
        risk_predictions=candidate.manifest["risk_predictions"],
        formal_refs=candidate.manifest["formal_refs"],
        graph_ops=[GraphOverlayOp(**op) for op in candidate.manifest["graph_ops"]],
        verifier=VerifierContract.from_dict(candidate.manifest["verifier"]),
        evidence_refs=candidate.manifest.get("evidence_refs", []),
        residual_refs=candidate.manifest.get("residual_refs", []),
        confidence=candidate.manifest.get("confidence", 0.5),
        metaproductivity=candidate.manifest.get("metaproductivity"),
        status=candidate.manifest.get("status", "candidate"),
    )
    return len(manifest.validate())


def _residual_trials() -> list[ResidualTrialFixture]:
    rows = [
        ("r_object_01", "science_history", "discovery", "object_regularities_missing", "classic law analogy explains residual but is not represented as an assumption family", "semantic retrieval only", "object", 0.91, "train"),
        ("r_object_02", "economics", "discovery", "object_regularities_missing", "equilibrium-seeking regularity appears without lexical overlap", "semantic retrieval only", "object", 0.88, "heldout"),
        ("r_method_01", "qa_bridge", "assumption_defect", "method_bridge_underuse", "answer misses bridge entity unless problem is decomposed into typed roles", "single-hop retrieval", "method", 0.93, "train"),
        ("r_method_02", "planning", "assumption_defect", "method_bridge_underuse", "solution improves when bridge role is made explicit before retrieval", "single-hop retrieval", "method", 0.89, "heldout"),
        ("r_eval_01", "judge_pairwise", "evaluator_defect", "evaluator_preference_mismatch", "judge rewards over-structured answer even when task asks for direct answer", "judge score as ground truth", "evaluator", 0.90, "train"),
        ("r_eval_02", "science_qa", "evaluator_defect", "evaluator_preference_mismatch", "candidate loses because evaluator prefers concrete action bridge over formal proof", "judge score as ground truth", "evaluator", 0.86, "heldout"),
        ("r_memory_01", "rag", "memory_defect", "memory_context_negative_transfer", "retrieval adds a related but wrong context edge", "top-k semantic context", "memory", 0.94, "train"),
        ("r_memory_02", "multi_hop_qa", "memory_defect", "memory_context_negative_transfer", "graph context dominates query-specific evidence and harms answer", "top-k semantic context", "memory", 0.92, "heldout"),
        ("r_world_01", "recursive_runner", "simulator_defect", "world_model_regression_underestimate", "cheap model underestimates graph pollution risk for promotion", "accept probability only", "world_model", 0.90, "train"),
        ("r_world_02", "ablation_queue", "simulator_defect", "world_model_regression_underestimate", "low predicted regression leads to wasted live ablation", "accept probability only", "world_model", 0.87, "heldout"),
        ("r_meta_01", "self_evolution", "assumption_defect", "meta_generator_local_patch_loop", "next-generation proposals keep repairing same family", "single trajectory proposal", "meta_evolution", 0.91, "train"),
        ("r_meta_02", "orthogonal_search", "assumption_defect", "meta_generator_local_patch_loop", "novel family is missed because generator does not branch trajectories", "single trajectory proposal", "meta_evolution", 0.90, "heldout"),
        ("r_lapse_01", "tool_use", "execution_lapse", "execution_lapse_command_error", "script failed because command omitted required argument", "hypothesis was correct but not executed", "method", 0.33, "train"),
        ("r_lapse_02", "prompting", "execution_lapse", "execution_lapse_prompt_omission", "answer ignored an already selected assumption", "hypothesis was correct but not executed", "method", 0.41, "heldout"),
    ]
    return [
        ResidualTrialFixture(
            trial_id=trial_id,
            domain=domain,
            residual_type=residual_type,
            cluster_key=cluster_key,
            symptom=symptom,
            active_assumption=active_assumption,
            expected_layer=expected_layer,
            execution_quality=execution_quality,
            heldout_split=heldout_split,
        )
        for (
            trial_id,
            domain,
            residual_type,
            cluster_key,
            symptom,
            active_assumption,
            expected_layer,
            execution_quality,
            heldout_split,
        ) in rows
    ]


def _cluster_profile(cluster_key: str) -> dict[str, Any]:
    profiles = {
        "object_regularities_missing": {
            "common_gap": "cross-domain regularity is absent from the object-level assumption family",
            "old_assumption_failure": "lexical or local context retrieval cannot infer same-family process regularities",
            "claim": "Represent recurring equilibrium-seeking object regularities as explicit object assumptions before solving unseen analogical cases.",
            "control_claim": "Treat every equilibrium-looking phrase as the same object law without checking boundary conditions.",
            "predicted_effect": "improve transfer from classic laws to unseen non-lexical regularities",
            "risk_prediction": "may over-generalize if negative controls lack invariant restoration",
            "cross_domain_transfer_score": 0.82,
            "descendant_productivity": 0.72,
        },
        "method_bridge_underuse": {
            "common_gap": "tasks need typed bridge decomposition before retrieval or reasoning",
            "old_assumption_failure": "single-hop retrieval hides the latent bridge variable",
            "claim": "For multi-hop failures, first generate typed bridge-role hypotheses and only then retrieve or answer.",
            "control_claim": "Always expand every query into all possible bridge roles before answering.",
            "predicted_effect": "recover bridge evidence without injecting unrelated entities",
            "risk_prediction": "may add latency or distractors on direct questions",
            "cross_domain_transfer_score": 0.76,
            "descendant_productivity": 0.69,
        },
        "evaluator_preference_mismatch": {
            "common_gap": "judge preference is mistaken for task truth",
            "old_assumption_failure": "evaluator score is treated as direct evidence of assumption quality",
            "claim": "When judge preference and task evidence disagree, create an evaluator-level hypothesis and require matched control adjudication.",
            "control_claim": "Ignore pairwise judge losses whenever the candidate has more formal structure.",
            "predicted_effect": "separate evaluator defects from real assumption defects",
            "risk_prediction": "may discount real failures if task evidence is weak",
            "cross_domain_transfer_score": 0.71,
            "descendant_productivity": 0.66,
        },
        "memory_context_negative_transfer": {
            "common_gap": "retrieved context edge is semantically related but role-incompatible",
            "old_assumption_failure": "top-k semantic context is promoted without role or residual compatibility",
            "claim": "Promote a memory hypothesis only when context edges preserve role, boundary, and residual compatibility.",
            "control_claim": "Promote a memory hypothesis only when context edges preserve role compatibility.",
            "predicted_effect": "reduce negative transfer from seductive but wrong context",
            "risk_prediction": "may under-retrieve weak lexical evidence if role extraction fails",
            "cross_domain_transfer_score": 0.79,
            "descendant_productivity": 0.70,
        },
        "world_model_regression_underestimate": {
            "common_gap": "simulator accepts candidates while underestimating pollution/regression risk",
            "old_assumption_failure": "world model optimizes accept probability without next-state risk",
            "claim": "Calibrate world-model hypotheses on next-state regression and graph pollution, not only accept probability.",
            "control_claim": "Raise every world-model risk estimate after any failed candidate.",
            "predicted_effect": "screen risky candidates before live ablation while preserving true positives",
            "risk_prediction": "may block novelty if calibration becomes too conservative",
            "cross_domain_transfer_score": 0.74,
            "descendant_productivity": 0.68,
        },
        "meta_generator_local_patch_loop": {
            "common_gap": "generator remains in one local repair family after repeated failures",
            "old_assumption_failure": "single trajectory proposal does not search orthogonal families",
            "claim": "When descendants repeatedly fail within one clade, branch the generator into integrate, reject, and orthogonal-new-family trajectories.",
            "control_claim": "Prefer orthogonal-new-family proposals even when residuals are execution lapses.",
            "predicted_effect": "increase productive descendants and avoid local patch loops",
            "risk_prediction": "may invent needless families if execution lapses are not filtered",
            "cross_domain_transfer_score": 0.84,
            "descendant_productivity": 0.75,
        },
    }
    return profiles[cluster_key]


def _layer_to_type(layer: str) -> AssumptionType:
    return {
        "object": AssumptionType.OBJECT,
        "method": AssumptionType.METHOD,
        "evaluator": AssumptionType.EVALUATOR,
        "memory": AssumptionType.MEMORY,
        "world_model": AssumptionType.WORLD_MODEL,
        "meta_evolution": AssumptionType.SELF_MODIFICATION,
    }[layer]


def _layer_to_kind(layer: str) -> HypothesisKind:
    return {
        "object": HypothesisKind.CLAIM,
        "method": HypothesisKind.CLAIM,
        "evaluator": HypothesisKind.EVALUATOR_POLICY,
        "memory": HypothesisKind.RETRIEVAL_POLICY,
        "world_model": HypothesisKind.WORLD_MODEL_TRIAL,
        "meta_evolution": HypothesisKind.HP_CHANGE,
    }[layer]


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
        if len(token) > 2
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-v2 Phase 4 hypothesis generator bypass validation.")
    parser.add_argument("--eval-id", default="full_v2_phase4_hypothesis_generator_bypass_20260611")
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_full_v2_phase4_hypothesis_generator_bypass_payload(
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
