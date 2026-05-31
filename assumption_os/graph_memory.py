"""JSONL Assumption Graph memory with HippoRAG-style spreading retrieval.

This intentionally starts with plain files instead of a graph database.  The
project already has many JSON artifacts; a JSONL graph can be inspected, diffed,
committed, reverted, and later migrated to NetworkX/Neo4j if needed.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Iterable

from .schema import (
    ActivatedSubgraph,
    AssumptionEdge,
    AssumptionNode,
    AssumptionType,
    EdgeType,
    EvidenceRecord,
    ResidualType,
    TrialManifest,
    TrialStatus,
    stable_id,
    utc_timestamp,
)


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]")


def tokenize(text: str) -> Counter[str]:
    toks = [t.lower() for t in TOKEN_RE.findall(text or "")]
    return Counter(t for t in toks if len(t) > 1 or "\u4e00" <= t <= "\u9fff")


def cosine_counter(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    shared = set(a) & set(b)
    num = sum(a[t] * b[t] for t in shared)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    return num / (da * db) if da and db else 0.0


class JsonlGraphStore:
    """Append-inspectable storage for nodes, edges, evidence, and trials."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.nodes_path = self.root / "nodes.jsonl"
        self.edges_path = self.root / "edges.jsonl"
        self.evidence_path = self.root / "evidence.jsonl"
        self.trials_path = self.root / "trials.jsonl"
        self.nodes: dict[str, AssumptionNode] = {}
        self.edges: list[AssumptionEdge] = []
        self.evidence: dict[str, EvidenceRecord] = {}
        self.trials: dict[str, TrialManifest] = {}
        self.load()

    def load(self) -> None:
        self.nodes = {}
        self.edges = []
        self.evidence = {}
        self.trials = {}
        for row in _read_jsonl(self.nodes_path):
            node = AssumptionNode.from_dict(row)
            self.nodes[node.id] = node
        for row in _read_jsonl(self.edges_path):
            self.edges.append(AssumptionEdge.from_dict(row))
        for row in _read_jsonl(self.evidence_path):
            ev = EvidenceRecord.from_dict(row)
            self.evidence[ev.evidence_id] = ev
        for row in _read_jsonl(self.trials_path):
            tr = TrialManifest.from_dict(row)
            self.trials[tr.trial_id] = tr

    def flush(self) -> None:
        _write_jsonl(self.nodes_path, [n.to_dict() for n in self.nodes.values()])
        _write_jsonl(self.edges_path, [e.to_dict() for e in self.edges])
        _write_jsonl(self.evidence_path, [e.to_dict() for e in self.evidence.values()])
        _write_jsonl(self.trials_path, [t.to_dict() for t in self.trials.values()])

    def upsert_node(self, node: AssumptionNode) -> None:
        existing = self.nodes.get(node.id)
        if existing:
            merged = _merge_node(existing, node)
            merged.updated_at = utc_timestamp()
            self.nodes[node.id] = merged
        else:
            self.nodes[node.id] = node

    def add_edge(self, edge: AssumptionEdge) -> None:
        for existing in self.edges:
            if existing.key == edge.key:
                existing.weight = max(existing.weight, edge.weight)
                existing.payload.update(edge.payload)
                if edge.evidence:
                    existing.evidence = edge.evidence
                return
        self.edges.append(edge)

    def add_evidence(self, evidence: EvidenceRecord) -> None:
        self.evidence[evidence.evidence_id] = evidence
        node = self.nodes.get(evidence.node_id)
        if node and evidence.evidence_id not in node.evidence_ids:
            node.evidence_ids.append(evidence.evidence_id)
            node.updated_at = utc_timestamp()

    def append_trial(self, trial: TrialManifest) -> None:
        self.trials[trial.trial_id] = trial


class SimpleAssumptionGraph:
    """Retrieval and lifecycle updates over a JsonlGraphStore."""

    def __init__(self, store: JsonlGraphStore):
        self.store = store
        self._node_tokens: dict[str, Counter[str]] = {}
        self._adj: dict[str, list[tuple[str, float]]] = {}
        self.reindex()

    def reindex(self) -> None:
        self._node_tokens = {
            nid: tokenize(_node_text(node))
            for nid, node in self.store.nodes.items()
        }
        adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for edge in self.store.edges:
            if edge.source in self.store.nodes and edge.target in self.store.nodes:
                adj[edge.source].append((edge.target, edge.weight))
                adj[edge.target].append((edge.source, edge.weight * 0.35))
        self._adj = dict(adj)

    def retrieve(
        self,
        query: str,
        *,
        seeds: Iterable[str] | None = None,
        top_k: int = 8,
        alpha: float = 0.85,
        iterations: int = 24,
        candidate_types: set[AssumptionType | str] | None = None,
    ) -> ActivatedSubgraph:
        """Activate a relevant assumption subgraph.

        The seed vector combines lexical match with explicit seed ids/tags, then
        spreads through the graph with a PPR-like update.  This is deliberately
        transparent and deterministic so failed retrievals can be audited.
        """

        seed_list = list(seeds or [])
        q_tokens = tokenize(" ".join([query, *[s for s in seed_list if s not in self.store.nodes]]))
        lexical = {
            nid: cosine_counter(q_tokens, toks)
            for nid, toks in self._node_tokens.items()
        }
        explicit_seed_ids = _resolve_seed_ids(seed_list, self.store.nodes.values())
        for nid in explicit_seed_ids:
            lexical[nid] = max(lexical.get(nid, 0.0), 1.0)

        if candidate_types:
            allowed = {str(t.value if isinstance(t, AssumptionType) else t) for t in candidate_types}
            lexical = {
                nid: score
                for nid, score in lexical.items()
                if str(self.store.nodes[nid].type.value) in allowed
            }

        seed_scores = _normalize({nid: s for nid, s in lexical.items() if s > 0.0})
        if not seed_scores:
            # Cold start: use confidence/metaproductivity instead of returning nothing.
            seed_scores = _normalize({
                nid: max(0.01, node.confidence + node.metaproductivity)
                for nid, node in self.store.nodes.items()
            })

        ppr = dict(seed_scores)
        for _ in range(iterations):
            nxt = {nid: (1.0 - alpha) * seed_scores.get(nid, 0.0) for nid in self.store.nodes}
            for src, val in ppr.items():
                outs = self._adj.get(src, [])
                if not outs:
                    nxt[src] = nxt.get(src, 0.0) + alpha * val
                    continue
                total_w = sum(max(w, 0.0) for _, w in outs) or 1.0
                for dst, weight in outs:
                    nxt[dst] = nxt.get(dst, 0.0) + alpha * val * (max(weight, 0.0) / total_w)
            ppr = nxt

        scores = {}
        for nid, node in self.store.nodes.items():
            if candidate_types and nid not in lexical:
                continue
            scores[nid] = (
                ppr.get(nid, 0.0)
                + 0.55 * lexical.get(nid, 0.0)
                + 0.08 * node.confidence
                + 0.12 * node.metaproductivity
            )
        ranked = sorted(scores, key=lambda nid: scores[nid], reverse=True)[:top_k]
        node_set = set(ranked)
        edges = [e for e in self.store.edges if e.source in node_set and e.target in node_set]
        nodes = [self.store.nodes[nid] for nid in ranked]
        cases = [n for n in nodes if n.type == AssumptionType.CASE]
        residuals = [n for n in nodes if n.type == AssumptionType.RESIDUAL]
        verifiers = [n for n in nodes if n.type == AssumptionType.VERIFIER]
        return ActivatedSubgraph(
            query=query,
            seed_ids=explicit_seed_ids,
            nodes=nodes,
            edges=edges,
            scores={nid: scores[nid] for nid in ranked},
            cases=cases,
            residuals=residuals,
            verifiers=verifiers,
        )

    def update_from_trial(
        self,
        manifest: TrialManifest,
        *,
        residual_type: ResidualType | str | None = None,
        persist: bool = True,
    ) -> list[str]:
        """Write a trial manifest back into node confidence/residual links."""

        rtype = ResidualType(residual_type or manifest.residual_type or ResidualType.UNKNOWN)
        manifest.residual_type = rtype
        self.store.append_trial(manifest)

        residual_node_ids: list[str] = []
        if manifest.residual:
            rid = stable_id("res", manifest.problem_id, manifest.residual, rtype.value)
            residual_node = AssumptionNode(
                id=rid,
                type=AssumptionType.RESIDUAL,
                claim=manifest.residual,
                context_conditions=[manifest.problem_id],
                payload={
                    "trial_id": manifest.trial_id,
                    "residual_type": rtype.value,
                    "observed_effect": manifest.observed_effect,
                },
                tags=[rtype.value, manifest.action_type],
                confidence=0.5,
            )
            self.store.upsert_node(residual_node)
            residual_node_ids.append(rid)

        for nid in manifest.assumption_ids:
            node = self.store.nodes.get(nid)
            if not node:
                continue
            node.confidence = _bounded(node.confidence + _confidence_delta(manifest, rtype))
            node.updated_at = utc_timestamp()
            if residual_node_ids:
                for rid in residual_node_ids:
                    if rid not in node.residual_ids:
                        node.residual_ids.append(rid)
                    edge_type = (
                        EdgeType.EXECUTION_LAPSE_OF
                        if rtype == ResidualType.EXECUTION_LAPSE
                        else EdgeType.FAILED_BECAUSE
                    )
                    self.store.add_edge(AssumptionEdge(source=nid, target=rid, type=edge_type, weight=0.8))

        if persist:
            self.store.flush()
        self.reindex()
        return residual_node_ids

    def clade_metaproductivity(self, node_id: str, max_depth: int = 4) -> float:
        descendants = self._descendants(node_id, max_depth=max_depth)
        if not descendants:
            return self.store.nodes.get(node_id, AssumptionNode(node_id, AssumptionType.METHOD, "")).metaproductivity
        values = []
        for nid in descendants:
            node = self.store.nodes[nid]
            ev_values = [
                ev.value
                for eid in node.evidence_ids
                for ev in [self.store.evidence.get(eid)]
                if ev and ev.value is not None and ev.outcome in {"accepted", "success", "improved"}
            ]
            if ev_values:
                values.append(sum(ev_values) / len(ev_values))
            else:
                values.append(node.confidence * 0.5 + node.metaproductivity)
        return sum(values) / len(values) if values else 0.0

    def _descendants(self, node_id: str, max_depth: int = 4) -> set[str]:
        forward = defaultdict(list)
        productive_edges = {
            EdgeType.SPECIALIZES,
            EdgeType.GENERATED_FROM_RESIDUAL,
            EdgeType.DERIVED_FROM,
            EdgeType.REPLACES,
        }
        for edge in self.store.edges:
            if edge.type in productive_edges:
                forward[edge.source].append(edge.target)
        seen: set[str] = set()
        q = deque([(node_id, 0)])
        while q:
            cur, depth = q.popleft()
            if depth >= max_depth:
                continue
            for nxt in forward.get(cur, []):
                if nxt in seen:
                    continue
                seen.add(nxt)
                q.append((nxt, depth + 1))
        seen.discard(node_id)
        return seen


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _node_text(node: AssumptionNode) -> str:
    parts = [
        node.claim,
        " ".join(node.context_conditions),
        " ".join(node.predicted_effects),
        " ".join(node.risk_predictions),
        " ".join(node.verifiers),
        " ".join(node.tags),
        json.dumps(node.payload, ensure_ascii=False, sort_keys=True),
    ]
    return "\n".join(p for p in parts if p)


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    total = sum(max(v, 0.0) for v in scores.values())
    if total <= 0:
        return {}
    return {k: max(v, 0.0) / total for k, v in scores.items()}


def _resolve_seed_ids(seeds: list[str], nodes: Iterable[AssumptionNode]) -> list[str]:
    node_list = list(nodes)
    by_id = {n.id: n.id for n in node_list}
    out = []
    for seed in seeds:
        if seed in by_id:
            out.append(seed)
            continue
        seed_l = seed.lower()
        for node in node_list:
            if seed_l in {t.lower() for t in node.tags}:
                out.append(node.id)
    return sorted(set(out))


def _merge_node(old: AssumptionNode, new: AssumptionNode) -> AssumptionNode:
    d = old.to_dict()
    nd = new.to_dict()
    for key in ("claim", "type", "kind", "formal_form", "status"):
        if nd.get(key):
            d[key] = nd[key]
    for key in (
        "context_conditions",
        "predicted_effects",
        "risk_predictions",
        "verifiers",
        "evidence_ids",
        "residual_ids",
        "tags",
        "source_refs",
    ):
        d[key] = _unique([*d.get(key, []), *nd.get(key, [])])
    d["payload"] = {**d.get("payload", {}), **nd.get("payload", {})}
    d["confidence"] = max(float(d.get("confidence", 0.5)), float(nd.get("confidence", 0.5)))
    d["metaproductivity"] = max(float(d.get("metaproductivity", 0.0)), float(nd.get("metaproductivity", 0.0)))
    return AssumptionNode.from_dict(d)


def _unique(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _confidence_delta(manifest: TrialManifest, rtype: ResidualType) -> float:
    status = TrialStatus(manifest.status)
    if status == TrialStatus.ACCEPTED:
        return 0.06
    if status == TrialStatus.REJECTED:
        return -0.08
    if rtype == ResidualType.NO_RESIDUAL:
        return 0.03
    if rtype == ResidualType.EXECUTION_LAPSE:
        return 0.0
    if rtype == ResidualType.OPTIMIZATION:
        return -0.02
    if rtype in {ResidualType.ASSUMPTION_DEFECT, ResidualType.MEMORY_DEFECT, ResidualType.EVALUATOR_DEFECT}:
        return -0.06
    if rtype == ResidualType.SIMULATOR_DEFECT:
        return -0.04
    if status == TrialStatus.FAILED:
        return -0.06
    return 0.0


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))
