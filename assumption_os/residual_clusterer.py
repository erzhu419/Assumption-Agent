"""Residual clustering and method-hypothesis synthesis.

Failures should not immediately mutate the graph.  This module first clusters
systematic residuals, then produces reviewable candidate method hypotheses and
heldout validation plans.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .graph_memory import JsonlGraphStore, tokenize
from .proposals import CandidateProposal, ProposalType
from .schema import (
    AssumptionEdge,
    AssumptionNode,
    AssumptionType,
    EdgeType,
    HypothesisKind,
    ResidualType,
    TrialManifest,
    TrialStatus,
    stable_id,
)


@dataclass(frozen=True)
class ResidualRecord:
    record_id: str
    problem_id: str
    residual_type: str
    residual: str
    action_type: str
    component: str
    assumption_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ResidualCluster:
    cluster_id: str
    residual_type: str
    signature: str
    records: list[ResidualRecord]
    top_terms: list[str]
    parent_node_id: str | None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["records"] = [r.to_dict() for r in self.records]
        return d


def build_residual_cluster_payload(
    *,
    store: JsonlGraphStore,
    eval_id: str,
    min_cluster_size: int = 2,
    max_clusters: int = 8,
    llm_synthesizer: Callable[[str], str] | None = None,
    writeback_manifests: bool = False,
) -> dict:
    """Cluster residuals and synthesize candidate method hypotheses."""

    records = collect_residual_records(store)
    clusters = cluster_residual_records(
        records,
        min_cluster_size=min_cluster_size,
        max_clusters=max_clusters,
    )
    proposals = [
        synthesize_cluster_proposal(
            cluster=cluster,
            store=store,
            eval_id=eval_id,
            llm_synthesizer=llm_synthesizer,
        )
        for cluster in clusters
        if cluster.parent_node_id
    ]
    if writeback_manifests:
        for proposal in proposals:
            if proposal.manifest:
                store.append_trial(TrialManifest.from_dict(proposal.manifest))
        store.flush()
    return {
        "eval_id": eval_id,
        "mode": {
            "min_cluster_size": min_cluster_size,
            "max_clusters": max_clusters,
            "writeback_manifests": writeback_manifests,
            "llm_synthesizer": llm_synthesizer is not None,
        },
        "record_count": len(records),
        "cluster_count": len(clusters),
        "proposal_count": len(proposals),
        "residual_type_counts": dict(Counter(r.residual_type for r in records)),
        "cluster_signature_counts": dict(Counter(c.signature for c in clusters)),
        "clusters": [c.to_dict() for c in clusters],
        "proposal_counts": dict(Counter(p.proposal_type.value for p in proposals)),
        "proposals": [p.to_dict() for p in proposals],
    }


def collect_residual_records(store: JsonlGraphStore) -> list[ResidualRecord]:
    records = []
    for trial in store.trials.values():
        rtype = getattr(trial.residual_type, "value", trial.residual_type) or ""
        if not trial.residual or rtype in {"", ResidualType.NO_RESIDUAL.value}:
            continue
        records.append(ResidualRecord(
            record_id=trial.trial_id,
            problem_id=trial.problem_id,
            residual_type=str(rtype),
            residual=trial.residual,
            action_type=trial.action_type,
            component=trial.component or "",
            assumption_ids=list(trial.assumption_ids),
            metadata=dict(trial.metadata),
        ))
    for node in store.nodes.values():
        node_type = getattr(node.type, "value", node.type)
        if node_type != AssumptionType.RESIDUAL.value:
            continue
        rtype = str((node.payload or {}).get("residual_type") or ResidualType.UNKNOWN.value)
        records.append(ResidualRecord(
            record_id=node.id,
            problem_id=";".join(node.context_conditions[:2]) or node.id,
            residual_type=rtype,
            residual=node.claim,
            action_type="residual_node",
            component="graph_memory",
            assumption_ids=[],
            metadata=dict(node.payload or {}),
        ))
    return sorted(records, key=lambda r: (r.residual_type, r.problem_id, r.record_id))


def cluster_residual_records(
    records: list[ResidualRecord],
    *,
    min_cluster_size: int = 2,
    max_clusters: int = 8,
) -> list[ResidualCluster]:
    grouped: dict[tuple[str, str], list[ResidualRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.residual_type, _signature(record))].append(record)
    clusters = []
    for (rtype, signature), rows in grouped.items():
        if len(rows) < min_cluster_size:
            continue
        cluster_id = stable_id("rcluster", rtype, signature, ",".join(r.record_id for r in rows))
        clusters.append(ResidualCluster(
            cluster_id=cluster_id,
            residual_type=rtype,
            signature=signature,
            records=rows,
            top_terms=_top_terms(rows),
            parent_node_id=_dominant_parent(rows),
        ))
    return sorted(clusters, key=lambda c: (-len(c.records), c.residual_type, c.signature))[:max_clusters]


def synthesize_cluster_proposal(
    *,
    cluster: ResidualCluster,
    store: JsonlGraphStore,
    eval_id: str,
    llm_synthesizer: Callable[[str], str] | None = None,
) -> CandidateProposal:
    parent = store.nodes.get(cluster.parent_node_id or "")
    parent_claim = parent.claim if parent else cluster.parent_node_id or "unknown parent"
    synthesis_prompt = _synthesis_prompt(cluster, parent_claim=parent_claim)
    llm_claim = llm_synthesizer(synthesis_prompt).strip() if llm_synthesizer else ""
    claim = llm_claim or _deterministic_claim(cluster, parent_claim=parent_claim)
    cid = stable_id("cand", eval_id, cluster.cluster_id, cluster.parent_node_id, "residual_cluster")
    candidate = AssumptionNode(
        id=cid,
        type=AssumptionType.METHOD,
        kind=HypothesisKind.CLAIM,
        claim=claim,
        context_conditions=[
            f"systematic_residual_type={cluster.residual_type}",
            f"cluster_signature={cluster.signature}",
            f"parent={cluster.parent_node_id}",
        ],
        predicted_effects=[
            "reduce the clustered residual on heldout trigger rows",
            "avoid outside-control harm before graph promotion",
        ],
        risk_predictions=[
            "may overfit residual wording instead of solving the causal defect",
            "must pass heldout validation before promotion",
        ],
        verifiers=[
            "residual_cluster_heldout_validation",
            "candidate_acceptance_gate",
            "outside_control_harm_check",
        ],
        confidence=0.42,
        metaproductivity=0.06 + min(0.12, len(cluster.records) * 0.01),
        status="candidate",
        tags=[
            "candidate",
            "residual_cluster",
            cluster.residual_type,
            *(cluster.top_terms[:5]),
        ],
        payload={
            "source": "residual_clusterer",
            "cluster": cluster.to_dict(),
            "llm_synthesis_prompt": synthesis_prompt,
            "llm_synthesis_used": bool(llm_claim),
            "validation_plan": _validation_plan(cluster, candidate_id=cid),
        },
    )
    edge = AssumptionEdge(
        source=cluster.parent_node_id or "",
        target=cid,
        type=EdgeType.GENERATED_FROM_RESIDUAL,
        weight=0.65,
        payload={"cluster_id": cluster.cluster_id},
    ).to_dict()
    manifest = TrialManifest(
        problem_id=f"residual_cluster::{cluster.cluster_id}",
        action_type="residual_cluster_synthesis",
        component="residual_clusterer",
        assumption=claim,
        why_selected=f"{len(cluster.records)} residuals share {cluster.signature}.",
        expected_effect="Create a falsifiable method candidate only from systematic residual evidence.",
        assumption_ids=[x for x in [cluster.parent_node_id, cid] if x],
        verifier="heldout_residual_validation",
        verification_plan=json.dumps(_validation_plan(cluster, candidate_id=cid), ensure_ascii=False, sort_keys=True),
        rollback_condition="Reject if heldout trigger rows fail or outside controls show harm.",
        status=TrialStatus.PENDING,
        artifacts={"cluster": cluster.to_dict(), "candidate_node": candidate.to_dict()},
        metadata={"eval_id": eval_id, "cluster_id": cluster.cluster_id},
        trial_id=stable_id("trial", eval_id, cluster.cluster_id, "residual_cluster_synthesis"),
    ).to_dict()
    return CandidateProposal(
        proposal_id=stable_id("prop", eval_id, cluster.cluster_id, cid),
        proposal_type=ProposalType.FAILURE_HYPOTHESIS,
        parent_node_id=cluster.parent_node_id or "",
        candidate_node=candidate.to_dict(),
        edges=[edge] if cluster.parent_node_id else [],
        manifest=manifest,
        rationale=f"Systematic residual cluster {cluster.cluster_id} has {len(cluster.records)} records.",
        priority=min(1.0, 0.35 + 0.08 * len(cluster.records)),
        source_action={
            "action_type": "synthesize_from_residual_cluster",
            "cluster_id": cluster.cluster_id,
            "residual_type": cluster.residual_type,
            "signature": cluster.signature,
        },
    )


def _signature(record: ResidualRecord) -> str:
    component = record.component or record.action_type or "unknown_component"
    if record.residual_type == ResidualType.MEMORY_DEFECT.value:
        return "memory_retrieval"
    if record.residual_type == ResidualType.EXECUTION_LAPSE.value:
        return "execution_followthrough"
    if record.residual_type == ResidualType.SIMULATOR_DEFECT.value:
        return "simulator_calibration"
    if record.residual_type == ResidualType.EVALUATOR_DEFECT.value:
        return "evaluator_alignment"
    terms = _top_terms([record])
    term = terms[0] if terms else component
    return f"{component}:{term}"


def _top_terms(records: list[ResidualRecord], limit: int = 8) -> list[str]:
    counts = Counter()
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "into", "should",
        "would", "could", "residual", "candidate", "answer",
    }
    for record in records:
        counts.update({
            tok for tok in tokenize(record.residual)
            if len(tok) > 2 and tok not in stop
        })
    return [term for term, _ in counts.most_common(limit)]


def _dominant_parent(records: list[ResidualRecord]) -> str | None:
    counts = Counter(
        aid
        for record in records
        for aid in record.assumption_ids
        if aid
    )
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _deterministic_claim(cluster: ResidualCluster, *, parent_claim: str) -> str:
    terms = ", ".join(cluster.top_terms[:4]) or cluster.signature
    if cluster.residual_type == ResidualType.MEMORY_DEFECT.value:
        return (
            f"Before using the parent method ({parent_claim}), require retrieval evidence that names the "
            f"trigger concepts ({terms}) and suppresses unrelated memories."
        )
    if cluster.residual_type == ResidualType.EXECUTION_LAPSE.value:
        return (
            f"Turn the parent method ({parent_claim}) into an executable checklist with a final answer audit "
            f"that confirms each required step was actually used."
        )
    if cluster.residual_type == ResidualType.EVALUATOR_DEFECT.value:
        return (
            f"Validate the parent method ({parent_claim}) with cross-judge objective criteria before changing "
            f"the graph when residuals mention {terms}."
        )
    if cluster.residual_type == ResidualType.SIMULATOR_DEFECT.value:
        return (
            f"Downweight simulator rollouts for the parent method ({parent_claim}) until calibration examples "
            f"cover residual pattern {terms}."
        )
    return (
        f"Narrow and operationalize the parent method ({parent_claim}) for residual pattern {terms}, then "
        f"validate it on heldout trigger/control rows before promotion."
    )


def _synthesis_prompt(cluster: ResidualCluster, *, parent_claim: str) -> str:
    examples = "\n".join(
        f"- {record.problem_id}: {record.residual}"
        for record in cluster.records[:6]
    )
    return (
        "Synthesize one falsifiable method hypothesis from this systematic residual cluster.\n"
        f"Parent assumption: {parent_claim}\n"
        f"Residual type: {cluster.residual_type}\n"
        f"Signature: {cluster.signature}\n"
        f"Top terms: {', '.join(cluster.top_terms)}\n"
        "Residual examples:\n"
        f"{examples}\n"
        "Return one concrete claim that can be tested on heldout trigger/control rows."
    )


def _validation_plan(cluster: ResidualCluster, *, candidate_id: str) -> dict:
    trigger_ids = sorted({record.problem_id for record in cluster.records})[:12]
    return {
        "candidate_node_id": candidate_id,
        "trigger_problem_ids": trigger_ids,
        "control_selection": "sample outside the residual cluster but sharing the parent assumption/domain",
        "min_trigger_judgments": min(3, max(1, len(trigger_ids))),
        "acceptance_gate": {
            "trigger_benefit_lcb90": 0.54,
            "control_loss_ucb90": 0.35,
        },
        "command_hint": (
            "python3 -m assumption_os.candidate_acceptance "
            "--proposals <cluster_proposals.json> --preflight <cluster_preflight.json> "
            "--judgments <heldout_judgments.json> --candidate-variant <candidate> --baseline-variant <baseline>"
        ),
    }


def _resolve(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--graph-dir", default="phase four/assumption_graph")
    ap.add_argument("--eval-id", required=True)
    ap.add_argument("--min-cluster-size", type=int, default=2)
    ap.add_argument("--max-clusters", type=int, default=8)
    ap.add_argument("--writeback-manifests", action="store_true")
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    store = JsonlGraphStore(_resolve(root, args.graph_dir))
    payload = build_residual_cluster_payload(
        store=store,
        eval_id=args.eval_id,
        min_cluster_size=args.min_cluster_size,
        max_clusters=args.max_clusters,
        writeback_manifests=args.writeback_manifests,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
