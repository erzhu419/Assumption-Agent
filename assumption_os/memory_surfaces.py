"""Runtime memory-surface index for Assumption Graph.

The graph should remember not only domain methods and residuals, but also the
runtime mechanisms that govern self-evolution: retrieval policy, verifier
stack, world model, recursive runner, evaluator policy, formal mapping,
manifest logging, and harness governance.  This module materializes those
mechanisms as first-class assumption nodes with typed edges and evidence.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .graph_memory import JsonlGraphStore
from .schema import (
    AssumptionEdge,
    AssumptionNode,
    AssumptionType,
    EdgeType,
    EvidenceRecord,
    HypothesisKind,
    TrialManifest,
    TrialStatus,
    stable_id,
)


@dataclass(frozen=True)
class RuntimeSurfaceSpec:
    key: str
    type: AssumptionType
    claim: str
    tags: list[str]
    context_conditions: list[str] = field(default_factory=list)
    predicted_effects: list[str] = field(default_factory=list)
    risk_predictions: list[str] = field(default_factory=list)
    verifiers: list[str] = field(default_factory=list)
    source_section: str = ""

    @property
    def node_id(self) -> str:
        return stable_id("surface", self.key)


DEFAULT_SURFACES = [
    RuntimeSurfaceSpec(
        key="assumption_graph_memory",
        type=AssumptionType.MEMORY,
        claim="Assumption Graph memory should store assumptions, cases, residuals, verifiers, and runtime mechanisms as typed graph nodes.",
        tags=["memory_surface", "assumption_graph", "hipporag_style"],
        context_conditions=["long-term memory", "cross-run transfer", "graph retrieval"],
        predicted_effects=["increase retrievable lifecycle context", "support multi-hop assumption transfer"],
        risk_predictions=["over-broad memory nodes may dominate retrieval if not typed and gated"],
        verifiers=["memory_surface_audit", "assumption_bench_memory_transfer"],
        source_section="assumption_bench",
    ),
    RuntimeSurfaceSpec(
        key="domain_retrieval_policy",
        type=AssumptionType.RETRIEVAL,
        claim="Graph retrieval should be domain-aware and route-conditioned, not a raw top-k context dump.",
        tags=["retrieval_policy", "route_conditioned", "negative_transfer_guard"],
        context_conditions=["Phase2 graph injection", "software-engineering negative transfer", "candidate preflight"],
        predicted_effects=["improve trigger exposure", "reduce no-fire retrieval harm"],
        risk_predictions=["too narrow retrieval can suppress useful analogies"],
        verifiers=["candidate_preflight", "conditioned_eval_gate"],
        source_section="verifier_stack",
    ),
    RuntimeSurfaceSpec(
        key="verifier_stack",
        type=AssumptionType.VERIFIER,
        claim="Candidate assumptions require an ordered verifier stack before graph mutation.",
        tags=["verifier_stack", "V0_V4", "falsification_protocol"],
        context_conditions=["candidate promotion", "gated apply", "fresh ablation"],
        predicted_effects=["prevent weak or harmful candidates from contaminating graph memory"],
        risk_predictions=["over-strict gates can delay useful but underpowered hypotheses"],
        verifiers=["V0_preflight", "V1_world_model", "V3_falsification", "V4_acceptance"],
        source_section="verifier_stack",
    ),
    RuntimeSurfaceSpec(
        key="world_model_screen",
        type=AssumptionType.WORLD_MODEL,
        claim="A calibrated cheap world model should predict candidate acceptance and regression risk before expensive ablation.",
        tags=["world_model", "cheap_verifier", "calibration"],
        context_conditions=["candidate screening", "trajectory search", "budget allocation"],
        predicted_effects=["prioritize candidates worth fresh judgments", "surface simulator defects when predictions fail"],
        risk_predictions=["calibration can overfit if trained on too few accepted examples"],
        verifiers=["brier_score", "AUC", "leave_one_out_calibration"],
        source_section="world_model",
    ),
    RuntimeSurfaceSpec(
        key="evaluator_policy",
        type=AssumptionType.EVALUATOR,
        claim="Evaluator and judge behavior is itself an assumption that must be cross-checked against trigger/control evidence.",
        tags=["evaluator_policy", "judge_assumption", "cross_judge"],
        context_conditions=["pairwise judging", "fresh ablation", "placebo context"],
        predicted_effects=["reduce judge-style overfitting and side-bias"],
        risk_predictions=["single-judge wins can reward style rather than assumption validity"],
        verifiers=["bidirectional_judge", "placebo_context_control", "fresh_cross_judge_replay"],
        source_section="verifier_stack",
    ),
    RuntimeSurfaceSpec(
        key="formal_alignment_layer",
        type=AssumptionType.ALIGNMENT,
        claim="Formal mapping should act as a bounded alignment/search layer over typed finite kernels, not a universal theorem prover.",
        tags=["formal_mapping", "alignment", "finite_kernel"],
        context_conditions=["typed formal bundles", "cross-domain transfer", "promotion-sensitive policies"],
        predicted_effects=["block unsafe formal promotions", "support structural transfer when mappings are complete"],
        risk_predictions=["formal audit may not apply to non-formal candidates"],
        verifiers=["formal_mapping_gate", "finite_kernel_metrics"],
        source_section="formal_metrics",
    ),
    RuntimeSurfaceSpec(
        key="recursive_assumption_runner",
        type=AssumptionType.SELF_MODIFICATION,
        claim="Recursive self-evolution should expose parent hypotheses, child evidence gaps, return updates, and actionable frontier steps.",
        tags=["recursive_runner", "self_modification", "return_update"],
        context_conditions=["recursive hypothesis tree", "daemon resume", "candidate acceptance"],
        predicted_effects=["make recursive self-argument auditable instead of a flat proposal queue"],
        risk_predictions=["open child frames can stall without explicit frontier auditing"],
        verifiers=["recursive_audit", "recursive_daemon"],
        source_section="recursive_audit",
    ),
    RuntimeSurfaceSpec(
        key="manifest_logger",
        type=AssumptionType.HARNESS,
        claim="Every LLM, retrieval, judge, tool, simulator, and daemon event should become a redacted TrialManifest.",
        tags=["manifest_logger", "observability", "AHE"],
        context_conditions=["component observability", "experience logging", "decision audit"],
        predicted_effects=["turn hidden harness actions into falsifiable contracts"],
        risk_predictions=["logging can leak secrets if redaction fails"],
        verifiers=["secret_scan", "manifest_throughput", "real_log_ingest"],
        source_section="manifest_logger",
    ),
    RuntimeSurfaceSpec(
        key="evolution_context_gate",
        type=AssumptionType.HARNESS,
        claim="The evolution procedure itself must be governed by explicit responsibilities, permission boundaries, and rollback contracts.",
        tags=["evolution_context", "harness_responsibility", "permission_gate"],
        context_conditions=["writeback", "apply accepted", "execute commands", "autonomous apply"],
        predicted_effects=["block silent graph mutation and unbounded self-modification"],
        risk_predictions=["too much friction can slow low-risk learning loops"],
        verifiers=["evolution_context_responsibility_gate", "permission_violation_check"],
        source_section="evolution_context",
    ),
    RuntimeSurfaceSpec(
        key="assumption_lifecycle_scoreboard",
        type=AssumptionType.EVALUATOR,
        claim="Assumption OS progress should be scored by lifecycle capabilities, not only answer win-rate.",
        tags=["assumption_bench", "capability_scoreboard", "evaluation"],
        context_conditions=["system evaluation", "reconstruction gap validation", "capability regression"],
        predicted_effects=["surface weak lifecycle components even when pooled answer metrics look good"],
        risk_predictions=["score thresholds can be gamed if not tied to real artifacts"],
        verifiers=["assumption_bench", "performance_validation"],
        source_section="assumption_bench",
    ),
]


DEFAULT_EDGES = [
    ("domain_retrieval_policy", "assumption_graph_memory", EdgeType.DEPENDS_ON),
    ("verifier_stack", "domain_retrieval_policy", EdgeType.HAS_VERIFIER),
    ("verifier_stack", "world_model_screen", EdgeType.HAS_VERIFIER),
    ("verifier_stack", "evaluator_policy", EdgeType.USES_EVALUATOR),
    ("evaluator_policy", "verifier_stack", EdgeType.SPECIALIZES),
    ("formal_alignment_layer", "verifier_stack", EdgeType.SUPPORTS),
    ("formal_alignment_layer", "domain_retrieval_policy", EdgeType.IS_FORMAL_ISOMORPHISM_OF),
    ("recursive_assumption_runner", "verifier_stack", EdgeType.DEPENDS_ON),
    ("recursive_assumption_runner", "evolution_context_gate", EdgeType.SUPPORTS),
    ("manifest_logger", "evolution_context_gate", EdgeType.SUPPORTS),
    ("manifest_logger", "assumption_graph_memory", EdgeType.DERIVED_FROM),
    ("evolution_context_gate", "verifier_stack", EdgeType.DEPENDS_ON),
    ("evolution_context_gate", "recursive_assumption_runner", EdgeType.HAS_VERIFIER),
    ("assumption_lifecycle_scoreboard", "evolution_context_gate", EdgeType.HAS_VERIFIER),
    ("assumption_lifecycle_scoreboard", "world_model_screen", EdgeType.HAS_VERIFIER),
    ("assumption_lifecycle_scoreboard", "assumption_graph_memory", EdgeType.GENERALIZES),
]


def build_memory_surface_payload(
    *,
    graph_dir: Path,
    eval_id: str,
    performance_payload: dict | None = None,
    writeback: bool = False,
) -> dict:
    store = JsonlGraphStore(graph_dir)
    before = _graph_stats(store)
    nodes = [_surface_node(spec, performance_payload=performance_payload, eval_id=eval_id) for spec in DEFAULT_SURFACES]
    node_by_key = {spec.key: node for spec, node in zip(DEFAULT_SURFACES, nodes)}
    edges = [
        AssumptionEdge(
            source=node_by_key[source_key].id,
            target=node_by_key[target_key].id,
            type=edge_type,
            weight=0.85,
            payload={
                "eval_id": eval_id,
                "runtime_surface_edge": True,
                "source_key": source_key,
                "target_key": target_key,
            },
        )
        for source_key, target_key, edge_type in DEFAULT_EDGES
    ]
    evidence = [
        _surface_evidence(node, spec, performance_payload=performance_payload, eval_id=eval_id)
        for spec, node in zip(DEFAULT_SURFACES, nodes)
    ]
    manifest = _surface_manifest(eval_id=eval_id, nodes=nodes, edges=edges, before=before)
    existing_node_ids = set(store.nodes)
    existing_edge_keys = {edge.key for edge in store.edges}

    if writeback:
        for node in nodes:
            store.upsert_node(node)
        for edge in edges:
            store.add_edge(edge)
        for record in evidence:
            store.add_evidence(record)
        store.append_trial(manifest)
        store.flush()
        store.load()

    after = _graph_stats(store if writeback else _virtual_store(store, nodes, edges))
    return {
        "eval_id": eval_id,
        "writeback": writeback,
        "surface_count": len(nodes),
        "edge_count": len(edges),
        "evidence_count": len(evidence),
        "new_node_count": sum(1 for node in nodes if node.id not in existing_node_ids),
        "new_edge_count": sum(1 for edge in edges if edge.key not in existing_edge_keys),
        "surface_node_ids": [node.id for node in nodes],
        "surface_nodes": [node.to_dict() for node in nodes],
        "surface_edges": [edge.to_dict() for edge in edges],
        "evidence": [record.to_dict() for record in evidence],
        "manifest": manifest.to_dict(),
        "before_graph": before,
        "after_graph": after,
        "memory_transfer_ready": (
            after["node_type_count"] >= 8
            and after["edge_type_count"] >= 8
            and "world_model" in after["node_type_counts"]
            and "verifier" in after["node_type_counts"]
            and "retrieval" in after["node_type_counts"]
            and "evaluator" in after["node_type_counts"]
            and "alignment" in after["node_type_counts"]
            and "self_modification" in after["node_type_counts"]
        ),
    }


def _surface_node(
    spec: RuntimeSurfaceSpec,
    *,
    performance_payload: dict | None,
    eval_id: str,
) -> AssumptionNode:
    section = (performance_payload or {}).get("sections", {}).get(spec.source_section, {})
    score = _surface_confidence(spec, section)
    payload = {
        "runtime_surface": True,
        "surface_key": spec.key,
        "source_section": spec.source_section,
        "source_eval_id": (performance_payload or {}).get("eval_id"),
        "section_summary": _compact_section(section),
    }
    return AssumptionNode(
        id=spec.node_id,
        type=spec.type,
        kind=HypothesisKind.CLAIM,
        claim=spec.claim,
        context_conditions=spec.context_conditions,
        predicted_effects=spec.predicted_effects,
        risk_predictions=spec.risk_predictions,
        verifiers=spec.verifiers,
        confidence=score,
        metaproductivity=0.35 if spec.type in {AssumptionType.MEMORY, AssumptionType.SELF_MODIFICATION, AssumptionType.HARNESS} else 0.2,
        tags=spec.tags,
        source_refs=[spec.source_section, eval_id],
        payload=payload,
    )


def _surface_evidence(
    node: AssumptionNode,
    spec: RuntimeSurfaceSpec,
    *,
    performance_payload: dict | None,
    eval_id: str,
) -> EvidenceRecord:
    section = (performance_payload or {}).get("sections", {}).get(spec.source_section, {})
    score = _surface_confidence(spec, section)
    return EvidenceRecord(
        node_id=node.id,
        source=f"memory_surface::{eval_id}",
        outcome="observed" if section.get("pass", True) else "needs_repair",
        metric="source_section_score",
        value=score,
        split=eval_id,
        details={
            "surface_key": spec.key,
            "source_section": spec.source_section,
            "source_eval_id": (performance_payload or {}).get("eval_id"),
            "section_summary": _compact_section(section),
        },
        evidence_id=stable_id("ev", eval_id, node.id, spec.source_section),
    )


def _surface_manifest(
    *,
    eval_id: str,
    nodes: list[AssumptionNode],
    edges: list[AssumptionEdge],
    before: dict,
) -> TrialManifest:
    return TrialManifest(
        problem_id=f"memory_surface::{eval_id}",
        action_type="memory_surface_index",
        component="memory_surfaces",
        assumption="Runtime mechanisms should be materialized as typed Assumption Graph memory surfaces.",
        why_selected="AssumptionBench identified memory transfer as the weakest lifecycle capability.",
        expected_effect="Increase typed node and edge surfaces so future retrieval can access runtime self-evolution mechanisms.",
        assumption_ids=[node.id for node in nodes],
        predicted_regressions=[
            "runtime surface nodes could over-influence retrieval if not typed",
            "stale mechanism evidence could misrepresent current harness quality",
        ],
        verifier="assumption_bench_memory_transfer",
        verification_plan="Re-run AssumptionBench and confirm memory_transfer reaches full typed-surface coverage.",
        rollback_condition="Remove surface nodes/edges if retrieval starts selecting runtime mechanisms as irrelevant task context.",
        status=TrialStatus.OBSERVED,
        observed_effect=(
            f"indexed {len(nodes)} runtime surface nodes and {len(edges)} typed edges; "
            f"before node_type_count={before.get('node_type_count')}, edge_type_count={before.get('edge_type_count')}"
        ),
        artifacts={
            "node_ids": [node.id for node in nodes],
            "edge_count": len(edges),
            "before_graph": before,
        },
        metadata={"eval_id": eval_id},
        trial_id=stable_id("trial", eval_id, "memory_surface_index"),
    )


def _surface_confidence(spec: RuntimeSurfaceSpec, section: dict) -> float:
    if not section:
        return 0.62
    if section.get("pass") is False:
        return 0.42
    if spec.source_section == "world_model":
        calibration = section.get("post_calibration", {})
        brier = calibration.get("brier_score")
        if brier is not None:
            return max(0.55, min(0.95, 1.0 - float(brier)))
    if spec.source_section == "assumption_bench":
        return max(0.55, min(0.95, float(section.get("overall_score", 0.8) or 0.8)))
    if section.get("pass"):
        return 0.86
    return 0.68


def _compact_section(section: dict) -> dict:
    keep = [
        "pass",
        "overall_score",
        "min_score",
        "accepted_count",
        "rejected_count",
        "falsification_experiment_count",
        "matched_label_count",
        "post_calibration",
        "responsibility_status_counts",
        "dry_policy_decision",
        "apply_policy_decision",
        "blocked_policy_decision",
        "score_by_capability",
    ]
    return {key: section[key] for key in keep if key in section}


def _graph_stats(store: JsonlGraphStore) -> dict:
    node_type_counts = Counter(str(getattr(node.type, "value", node.type)) for node in store.nodes.values())
    edge_type_counts = Counter(str(getattr(edge.type, "value", edge.type)) for edge in store.edges)
    return {
        "node_count": len(store.nodes),
        "edge_count": len(store.edges),
        "evidence_count": len(store.evidence),
        "trial_count": len(store.trials),
        "node_type_count": len(node_type_counts),
        "edge_type_count": len(edge_type_counts),
        "node_type_counts": dict(node_type_counts),
        "edge_type_counts": dict(edge_type_counts),
    }


def _virtual_store(store: JsonlGraphStore, nodes: list[AssumptionNode], edges: list[AssumptionEdge]) -> JsonlGraphStore:
    class _Virtual:
        pass

    virtual = _Virtual()
    virtual.nodes = dict(store.nodes)
    for node in nodes:
        virtual.nodes[node.id] = node
    virtual.edges = list(store.edges)
    existing = {edge.key for edge in virtual.edges}
    for edge in edges:
        if edge.key not in existing:
            virtual.edges.append(edge)
            existing.add(edge.key)
    virtual.evidence = dict(store.evidence)
    virtual.trials = dict(store.trials)
    return virtual  # type: ignore[return-value]


def _load_json(path: Path | None) -> dict | None:
    if not path:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else root / p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--graph-dir", default="phase four/assumption_graph")
    ap.add_argument("--eval-id", required=True)
    ap.add_argument("--performance-payload", default=None)
    ap.add_argument("--writeback", action="store_true")
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    payload = build_memory_surface_payload(
        graph_dir=_resolve(root, args.graph_dir) or root,
        eval_id=args.eval_id,
        performance_payload=_load_json(_resolve(root, args.performance_payload)),
        writeback=args.writeback,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
