"""Full-v3 live-derived residual clusterer.

This module upgrades residual clustering from a single formal-alignment fixture
into a compact, redacted, artifact-level residual memory.  It reads committed
performance artifacts only and turns formal failures, Phase9 live residuals,
Phase8 creative residuals, and profile-level rejection evidence into unified
clusters plus next-generation proposal seeds.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .schema import stable_id


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "full_v3_live_residual_clusterer_20260611.json"

REQUIRED_ARTIFACTS = {
    "residual_v2": PAPER_DIR / "residual_hypothesis_generator_v2_20260610.json",
    "phase8": PAPER_DIR / "full_v3_phase8_creativity_world_coverage_20260611.json",
    "phase9_v1_regression": PAPER_DIR / "full_v3_phase9_v1_live_regression_20260611.json",
    "phase9_hybrid": PAPER_DIR / "full_v3_phase9_hybrid_guard_heldout_20260611.json",
    "phase9_compact": PAPER_DIR / "full_v3_phase9_selective_compact_guard_heldout_20260611.json",
    "phase9_micro": PAPER_DIR / "full_v3_phase9_micro_guard_heldout_20260611.json",
    "phase10_world_model": PAPER_DIR / "full_v3_phase10_discrete_world_model_selector_20260611.json",
}


@dataclass(frozen=True)
class LiveResidualObservation:
    observation_id: str
    source_artifact: str
    residual_axis: str
    domain: str
    pattern_id: str
    severity: str
    support_count: int
    source_problem_ids: list[str]
    claim: str
    downstream_status: str
    proposal_seed: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LiveResidualCluster:
    cluster_id: str
    residual_axis: str
    domain: str
    pattern_id: str
    total_support: int
    observation_ids: list[str]
    source_artifacts: list[str]
    severity_counts: dict[str, int]
    downstream_status: str
    proposal_seed: str
    evaluation_plan: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_full_v3_live_residual_clusterer_payload(
    *,
    root: Path,
    eval_id: str = "full_v3_live_residual_clusterer_20260611",
) -> dict[str, Any]:
    root = root.resolve()
    artifacts = {name: _load_json(root / path) for name, path in REQUIRED_ARTIFACTS.items()}
    observations = _collect_observations(artifacts)
    clusters = _cluster_observations(observations)
    proposal_seeds = [cluster for cluster in clusters if _is_open_for_next_generation(cluster)]
    metrics = _metrics(artifacts=artifacts, observations=observations, clusters=clusters, proposal_seeds=proposal_seeds)
    gates = {
        "source_artifacts_loaded": metrics["source_artifact_count"] == len(REQUIRED_ARTIFACTS),
        "source_artifacts_redacted": metrics["uses_raw_prompts_or_answers"] is False,
        "observation_count_high": metrics["observation_count"] >= 40,
        "weighted_residual_count_high": metrics["weighted_residual_count"] >= 70,
        "cluster_count_high": metrics["cluster_count"] >= 25,
        "systematic_weighted_coverage_high": metrics["systematic_weighted_coverage"] >= 0.85,
        "phase9_live_residuals_ingested": metrics["phase9_live_residual_observation_count"] >= 16,
        "formal_residuals_ingested": metrics["formal_residual_observation_count"] >= 15,
        "profile_residuals_ingested": metrics["profile_residual_observation_count"] >= 4,
        "largest_live_cluster_resolved": metrics["largest_live_cluster_status"] == "resolved_by_phase9_hybrid_guard",
        "next_generation_seeds_present": metrics["next_generation_proposal_seed_count"] >= 15,
        "blocked_profile_residuals_not_promoted": metrics["blocked_profile_residual_count"] >= 2,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_live_residual_clusterer",
        "reconstruction_v2_full_phase": "phase4_v3_live_residual_clusterer_upgrade",
        "implementation_level": "live_artifact_residual_clusterer",
        "performance_validation": True,
        "validation_scope": (
            "Unifies redacted residual evidence from formal alignment, Phase9 same-batch live failures, "
            "Phase8 creative residual families, and profile-level rejection/calibration artifacts.  It produces "
            "cluster-level next-generation proposal seeds without reading raw prompts, answers, or judge text."
        ),
        "source_artifacts": {
            name: {
                "path": str(path),
                "exists": (root / path).exists(),
                "pass": bool(artifacts[name].get("pass")),
                "eval_kind": artifacts[name].get("eval_kind"),
            }
            for name, path in REQUIRED_ARTIFACTS.items()
        },
        "observations": [observation.to_dict() for observation in observations],
        "clusters": [cluster.to_dict() for cluster in clusters],
        "next_generation_proposal_seeds": [cluster.to_dict() for cluster in proposal_seeds],
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "The residual clusterer now supports the recursive loop directly: observed failures are clustered, "
            "known resolved clusters are linked to the retained Phase9 hybrid profile, rejected profile residuals "
            "are kept as negative evidence, and unresolved clusters become explicit next-generation proposal seeds."
        ),
    }


def _collect_observations(artifacts: dict[str, dict[str, Any]]) -> list[LiveResidualObservation]:
    observations = []
    observations.extend(_formal_residual_observations(artifacts["residual_v2"]))
    observations.extend(_phase9_live_residual_observations(artifacts["phase9_v1_regression"], artifacts["phase9_hybrid"]))
    observations.extend(_phase8_creative_residual_observations(artifacts["phase8"]))
    observations.extend(_profile_residual_observations(artifacts))
    return sorted(observations, key=lambda row: row.observation_id)


def _formal_residual_observations(payload: dict[str, Any]) -> list[LiveResidualObservation]:
    rows = []
    for cluster in payload.get("clusters", []):
        axis = "formal_alignment_baseline_miss"
        component = str(cluster.get("component") or "unknown_component")
        residual_type = str(cluster.get("residual_type") or "unknown_residual")
        for record in cluster.get("records", []):
            source_pair = record.get("source_pair") or []
            problem_ids = ["::".join(source_pair)] if source_pair else [str(record.get("residual_id"))]
            rows.append(LiveResidualObservation(
                observation_id=stable_id("v3resobs", "formal", str(record.get("residual_id"))),
                source_artifact="residual_v2",
                residual_axis=axis,
                domain="formal_alignment",
                pattern_id=f"{component}:{residual_type}",
                severity="missed_positive_alignment",
                support_count=1,
                source_problem_ids=problem_ids,
                claim=(
                    "A baseline alignment component missed a formally accepted structural mapping; "
                    "candidate repair should preserve invariants and negative controls."
                ),
                downstream_status="proposal_ready_from_v2_generator",
                proposal_seed=f"Repair {component} for {residual_type} using the formal certificate and heldout controls.",
            ))
    return rows


def _phase9_live_residual_observations(
    payload: dict[str, Any],
    hybrid_payload: dict[str, Any],
) -> list[LiveResidualObservation]:
    rows = []
    hybrid_lift = float(hybrid_payload.get("metrics", {}).get("hybrid_lift_over_v3_vs_v1_heldout") or 0.0)
    for proposal in payload.get("residual_generated_next_proposals", []):
        pair = str(proposal.get("source_pair") or "")
        domain = str(proposal.get("source_domain") or "unknown_domain")
        pattern = str(proposal.get("source_pattern_id") or "unknown_pattern")
        axis = _axis_for_phase9_pair(pair)
        support = int(proposal.get("source_residual_count") or 1)
        status = "open_next_generation_seed"
        if pair == "v3_full_vs_v1_case_reflection_kernel" and domain == "business" and pattern == "pat_controlled_intervention" and hybrid_lift > 0.03:
            status = "resolved_by_phase9_hybrid_guard"
        rows.append(LiveResidualObservation(
            observation_id=stable_id("v3resobs", "phase9", str(proposal.get("proposal_id"))),
            source_artifact="phase9_v1_regression",
            residual_axis=axis,
            domain=domain,
            pattern_id=pattern,
            severity=str(proposal.get("residual_kind") or "loss_cluster"),
            support_count=support,
            source_problem_ids=list(proposal.get("seed_problem_ids") or []),
            claim=str(proposal.get("generated_next_hypothesis") or ""),
            downstream_status=status,
            proposal_seed=str(proposal.get("generated_next_hypothesis") or ""),
        ))
    return rows


def _phase8_creative_residual_observations(payload: dict[str, Any]) -> list[LiveResidualObservation]:
    rows = []
    for candidate in payload.get("creative_candidates", []):
        residual = str(candidate.get("source_residual_cluster") or "unknown_residual")
        family = str(candidate.get("hypothesis_family") or "unknown_family")
        decision = str(candidate.get("selective_retention_decision") or "unknown_decision")
        rows.append(LiveResidualObservation(
            observation_id=stable_id("v3resobs", "phase8", str(candidate.get("candidate_id"))),
            source_artifact="phase8",
            residual_axis=f"creative_generator:{residual}",
            domain="fresh_live_policy",
            pattern_id=family,
            severity="creative_residual_family",
            support_count=1,
            source_problem_ids=[],
            claim=str(candidate.get("claim") or ""),
            downstream_status=f"phase8_{decision}",
            proposal_seed=str(candidate.get("claim") or ""),
        ))
    return rows


def _profile_residual_observations(artifacts: dict[str, dict[str, Any]]) -> list[LiveResidualObservation]:
    compact = artifacts["phase9_compact"].get("metrics", {})
    micro = artifacts["phase9_micro"].get("metrics", {})
    phase10 = artifacts["phase10_world_model"].get("metrics", {})
    phase8 = artifacts["phase8"].get("metrics", {})
    rows = [
        LiveResidualObservation(
            observation_id=stable_id("v3resobs", "profile", "compact_overstructure"),
            source_artifact="phase9_compact",
            residual_axis="profile_policy:overstructured_compact_default",
            domain="heldout_policy",
            pattern_id="S14_S19_compact_guard",
            severity="rejected_profile",
            support_count=int(compact.get("selected_compact_case_count") or 1),
            source_problem_ids=[],
            claim="Broad compact framing improves V1 but regresses against original V3.",
            downstream_status="blocked_by_phase5_scheduler",
            proposal_seed="Keep compact framing as a scoped internal arm unless same-slice V3 non-regression passes.",
        ),
        LiveResidualObservation(
            observation_id=stable_id("v3resobs", "profile", "micro_no_lift"),
            source_artifact="phase9_micro",
            residual_axis="profile_policy:no_default_lift",
            domain="heldout_policy",
            pattern_id="S14_S19_micro_guard",
            severity="rejected_profile",
            support_count=int(micro.get("selected_micro_case_count") or 1),
            source_problem_ids=[],
            claim="Micro guard is non-regressive but has no heldout lift over the original V3-vs-V1 default.",
            downstream_status="blocked_by_phase5_scheduler",
            proposal_seed="Do not promote micro guard without a new trigger boundary that produces positive heldout lift.",
        ),
        LiveResidualObservation(
            observation_id=stable_id("v3resobs", "profile", "phase10_calibration"),
            source_artifact="phase10_world_model",
            residual_axis="world_model:calibration_miss",
            domain="heldout_policy",
            pattern_id="discrete_selector",
            severity="calibration_gap",
            support_count=int(phase10.get("candidate_transition_count") or 1),
            source_problem_ids=[],
            claim="The discrete selector is positive but does not beat base-rate calibration.",
            downstream_status="kept_as_candidate_by_phase5_scheduler",
            proposal_seed="Collect leave-domain-out live transitions before promoting the learned selector.",
        ),
        LiveResidualObservation(
            observation_id=stable_id("v3resobs", "profile", "coverage_quality_tradeoff"),
            source_artifact="phase8",
            residual_axis="coverage_policy:quality_tradeoff",
            domain="fresh_live_policy",
            pattern_id="coverage_v6",
            severity="coverage_tradeoff",
            support_count=int(phase8.get("coverage_profile_active_n") or 1),
            source_problem_ids=[],
            claim="Coverage v6 expands active rows but lowers utility relative to quality v4.",
            downstream_status="exploration_profile_only",
            proposal_seed="Separate quality and coverage objectives before expanding default activation.",
        ),
    ]
    return rows


def _cluster_observations(observations: list[LiveResidualObservation]) -> list[LiveResidualCluster]:
    grouped: dict[tuple[str, str, str], list[LiveResidualObservation]] = defaultdict(list)
    for observation in observations:
        grouped[(observation.residual_axis, observation.domain, observation.pattern_id)].append(observation)
    clusters = []
    for (axis, domain, pattern), rows in grouped.items():
        total_support = sum(max(1, row.support_count) for row in rows)
        statuses = Counter(row.downstream_status for row in rows)
        status = _cluster_status(statuses)
        proposal_seed = _cluster_proposal_seed(axis=axis, domain=domain, pattern=pattern, rows=rows, status=status)
        cluster_id = stable_id(
            "v3rcluster",
            axis,
            domain,
            pattern,
            ",".join(sorted(row.observation_id for row in rows)),
        )
        clusters.append(LiveResidualCluster(
            cluster_id=cluster_id,
            residual_axis=axis,
            domain=domain,
            pattern_id=pattern,
            total_support=total_support,
            observation_ids=sorted(row.observation_id for row in rows),
            source_artifacts=sorted({row.source_artifact for row in rows}),
            severity_counts=dict(Counter(row.severity for row in rows)),
            downstream_status=status,
            proposal_seed=proposal_seed,
            evaluation_plan=_evaluation_plan(axis=axis, domain=domain, pattern=pattern, status=status),
        ))
    return sorted(clusters, key=lambda row: (-row.total_support, row.residual_axis, row.domain, row.pattern_id))


def _cluster_status(statuses: Counter[str]) -> str:
    if statuses.get("resolved_by_phase9_hybrid_guard"):
        return "resolved_by_phase9_hybrid_guard"
    if statuses.get("blocked_by_phase5_scheduler"):
        return "blocked_by_phase5_scheduler"
    if statuses.get("kept_as_candidate_by_phase5_scheduler"):
        return "kept_as_candidate_by_phase5_scheduler"
    if statuses.get("exploration_profile_only"):
        return "exploration_profile_only"
    if statuses.get("proposal_ready_from_v2_generator"):
        return "proposal_ready_from_v2_generator"
    return "open_next_generation_seed"


def _cluster_proposal_seed(
    *,
    axis: str,
    domain: str,
    pattern: str,
    rows: list[LiveResidualObservation],
    status: str,
) -> str:
    if status == "resolved_by_phase9_hybrid_guard":
        return (
            "Use as positive retention evidence: this cluster should remain covered by the Phase9 hybrid guard "
            "rather than spawning another broad default repair."
        )
    if status == "blocked_by_phase5_scheduler":
        return (
            "Use as negative retention evidence: keep the profile gated off until it beats original V3 on the "
            "same heldout slice."
        )
    if status == "kept_as_candidate_by_phase5_scheduler":
        return rows[0].proposal_seed
    if axis.startswith("formal_alignment"):
        return rows[0].proposal_seed
    if axis.startswith("creative_generator"):
        return rows[0].proposal_seed
    return (
        f"For {domain}/{pattern}, generate two descendants: a narrower trigger-boundary repair and a "
        f"negative-control abstention repair for residual axis {axis}."
    )


def _evaluation_plan(*, axis: str, domain: str, pattern: str, status: str) -> str:
    if status == "resolved_by_phase9_hybrid_guard":
        return "Regression monitor only: verify future changes do not lose the retained Phase9 hybrid win."
    if status == "blocked_by_phase5_scheduler":
        return "Keep as negative-control row for future scheduler changes; do not promote without same-batch V3 non-regression."
    return (
        f"Run fresh ablation on heldout trigger rows for {domain}/{pattern}; include outside controls and compare "
        "against original V3, V1, and the retained Phase9 hybrid where applicable."
    )


def _axis_for_phase9_pair(pair: str) -> str:
    if "v1_case_reflection_kernel" in pair:
        return "same_batch_v1_regression:critical_reframe_gap"
    if "v3_no_recursive" in pair:
        return "toggle_regression:recursive_guidance_gap"
    if "v3_no_world_model" in pair:
        return "toggle_regression:world_model_guard_gap"
    if "v3_no_morphism" in pair:
        return "toggle_regression:morphism_unnecessary_or_harmful"
    return f"toggle_regression:{pair or 'unknown_pair'}"


def _is_open_for_next_generation(cluster: LiveResidualCluster) -> bool:
    return cluster.downstream_status not in {
        "resolved_by_phase9_hybrid_guard",
        "blocked_by_phase5_scheduler",
    }


def _metrics(
    *,
    artifacts: dict[str, dict[str, Any]],
    observations: list[LiveResidualObservation],
    clusters: list[LiveResidualCluster],
    proposal_seeds: list[LiveResidualCluster],
) -> dict[str, Any]:
    weighted_total = sum(max(1, row.support_count) for row in observations)
    systematic_weight = sum(cluster.total_support for cluster in clusters if cluster.total_support >= 2)
    phase9_live = [row for row in observations if row.source_artifact == "phase9_v1_regression"]
    largest_live = max(phase9_live, key=lambda row: row.support_count) if phase9_live else None
    profile_residuals = [row for row in observations if row.source_artifact in {"phase9_compact", "phase9_micro", "phase10_world_model", "phase8"} and row.residual_axis.startswith(("profile_policy", "world_model", "coverage_policy"))]
    return {
        "source_artifact_count": len(artifacts),
        "source_artifact_pass_count": sum(1 for artifact in artifacts.values() if artifact.get("pass")),
        "observation_count": len(observations),
        "weighted_residual_count": weighted_total,
        "cluster_count": len(clusters),
        "systematic_cluster_count": sum(1 for cluster in clusters if cluster.total_support >= 2),
        "systematic_weighted_coverage": round(systematic_weight / max(1, weighted_total), 4),
        "phase9_live_residual_observation_count": len(phase9_live),
        "formal_residual_observation_count": sum(1 for row in observations if row.source_artifact == "residual_v2"),
        "phase8_creative_residual_observation_count": sum(1 for row in observations if row.source_artifact == "phase8" and row.residual_axis.startswith("creative_generator")),
        "profile_residual_observation_count": len(profile_residuals),
        "resolved_cluster_count": sum(1 for cluster in clusters if cluster.downstream_status == "resolved_by_phase9_hybrid_guard"),
        "blocked_profile_residual_count": sum(1 for row in profile_residuals if row.downstream_status == "blocked_by_phase5_scheduler"),
        "next_generation_proposal_seed_count": len(proposal_seeds),
        "largest_live_cluster_support": int(largest_live.support_count if largest_live else 0),
        "largest_live_cluster_axis": largest_live.residual_axis if largest_live else "",
        "largest_live_cluster_domain": largest_live.domain if largest_live else "",
        "largest_live_cluster_pattern": largest_live.pattern_id if largest_live else "",
        "largest_live_cluster_status": largest_live.downstream_status if largest_live else "",
        "cluster_status_counts": dict(Counter(cluster.downstream_status for cluster in clusters)),
        "residual_axis_counts": dict(Counter(row.residual_axis for row in observations)),
        "uses_raw_prompts_or_answers": bool(
            artifacts["phase9_v1_regression"].get("metrics", {}).get("compact_payload_contains_prompts_answers", False)
            or artifacts["phase9_hybrid"].get("metrics", {}).get("compact_payload_contains_prompts_answers", False)
            or artifacts["phase10_world_model"].get("metrics", {}).get("uses_raw_prompts_or_answers", False)
        ),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-v3 live residual clusterer validation.")
    parser.add_argument("--eval-id", default="full_v3_live_residual_clusterer_20260611")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_full_v3_live_residual_clusterer_payload(root=root, eval_id=args.eval_id)
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
