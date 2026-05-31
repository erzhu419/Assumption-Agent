"""Adapters from existing project artifacts into Assumption Graph nodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import (
    AssumptionEdge,
    AssumptionNode,
    AssumptionType,
    EdgeType,
    EvidenceRecord,
    HypothesisKind,
    stable_id,
)


def load_strategy_nodes(kb_dir: str | Path) -> tuple[list[AssumptionNode], list[AssumptionEdge], list[EvidenceRecord]]:
    kb_path = Path(kb_dir)
    nodes: list[AssumptionNode] = []
    edges: list[AssumptionEdge] = []
    evidence: list[EvidenceRecord] = []
    for path in sorted(kb_path.glob("S*.json")):
        data = _load_json(path)
        sid = data.get("id") or path.stem
        desc = data.get("description", {})
        claim = desc.get("one_sentence") or desc.get("detailed") or data.get("name", {}).get("zh", sid)
        favorable, unfavorable, failure_modes = _strategy_conditions(data)
        node = AssumptionNode(
            id=f"strategy_{sid}",
            type=AssumptionType.METHOD,
            kind=HypothesisKind.CLAIM,
            claim=claim,
            context_conditions=favorable,
            risk_predictions=unfavorable + failure_modes,
            predicted_effects=[desc.get("detailed", "")[:240]] if desc.get("detailed") else [],
            confidence=_avg_condition_confidence(data, default=0.72),
            metaproductivity=_strategy_metaproductivity(data),
            tags=[
                "strategy",
                sid,
                data.get("category", ""),
                data.get("name", {}).get("zh", ""),
                data.get("name", {}).get("en", ""),
                *data.get("aliases", [])[:4],
            ],
            source_refs=[_format_source_ref(ref) for ref in data.get("source_references", [])],
            payload={
                "source_path": str(path),
                "name": data.get("name", {}),
                "operational_steps": data.get("operational_steps", []),
                "applicability_conditions": data.get("applicability_conditions", {}),
            },
        )
        nodes.append(node)
    return nodes, edges, evidence


def load_wisdom_nodes(path: str | Path) -> tuple[list[AssumptionNode], list[AssumptionEdge], list[EvidenceRecord]]:
    wisdom_path = Path(path)
    rows = _load_json(wisdom_path)
    nodes: list[AssumptionNode] = []
    edges: list[AssumptionEdge] = []
    evidence: list[EvidenceRecord] = []
    for row in rows:
        wid = row.get("id") or stable_id("wisdom", row.get("aphorism", ""))
        node_id = f"wisdom_{wid}"
        claim = row.get("unpacked_for_llm") or row.get("aphorism", "")
        node = AssumptionNode(
            id=node_id,
            type=AssumptionType.METHOD,
            kind=HypothesisKind.CLAIM,
            claim=claim,
            context_conditions=[row.get("signal", "")] if row.get("signal") else [],
            predicted_effects=["Should improve solver framing when its trigger signal matches the problem."],
            risk_predictions=[
                "May become generic prompt injection if not routed by trigger fit.",
                "May regress concise answers by adding verbosity.",
            ],
            verifiers=["cross-judge", "placebo-generic-warning", "fresh-split", "outside-trigger-regression"],
            confidence=_wisdom_confidence(row),
            metaproductivity=0.12 if row.get("cross_domain_examples") else 0.05,
            tags=["wisdom", wid, row.get("cluster", ""), row.get("aphorism", "")],
            source_refs=[row.get("source", "")] if row.get("source") else [],
            payload={**row, "source_path": str(wisdom_path)},
        )
        nodes.append(node)
        for i, case in enumerate(row.get("cross_domain_examples", []) or []):
            cid = stable_id("case", wid, i, case.get("domain", ""), case.get("scenario", ""))
            cnode = AssumptionNode(
                id=cid,
                type=AssumptionType.CASE,
                claim=case.get("scenario", ""),
                context_conditions=[case.get("domain", "")],
                tags=["case", "cross_domain", wid, case.get("domain", "")],
                confidence=0.55,
                payload={"wisdom_id": wid, "case_index": i, **case},
            )
            nodes.append(cnode)
            edges.append(AssumptionEdge(source=node_id, target=cid, type=EdgeType.HAS_CASE, weight=0.55))
    return nodes, edges, evidence


def load_residual_nodes(path: str | Path) -> tuple[list[AssumptionNode], list[AssumptionEdge], list[EvidenceRecord]]:
    residual_path = Path(path)
    if not residual_path.exists():
        return [], [], []
    rows = _load_json(residual_path)
    nodes: list[AssumptionNode] = []
    edges: list[AssumptionEdge] = []
    evidence: list[EvidenceRecord] = []
    for pid, row in _iter_dict_rows(rows):
        claim = row.get("what_v16_missed") or row.get("novel_orientation_needed") or row.get("cluster_tag", "")
        rid = stable_id("res", pid, claim, row.get("cluster_tag", ""))
        node = AssumptionNode(
            id=rid,
            type=AssumptionType.RESIDUAL,
            kind=HypothesisKind.CLAIM,
            claim=claim,
            context_conditions=[pid, row.get("domain", ""), row.get("difficulty", "")],
            predicted_effects=[row.get("proposed_refinement") or row.get("novel_orientation_needed") or ""],
            tags=["residual", row.get("cluster_tag", ""), row.get("domain", ""), row.get("primary_opponent", "")],
            confidence=0.5,
            payload={**row, "source_path": str(residual_path)},
        )
        nodes.append(node)
        nearest = row.get("nearest_existing_wisdom")
        if nearest and nearest != "null":
            wid = f"wisdom_{nearest}"
            weight = max(0.1, min(1.0, float(row.get("wisdom_applicability", 0)) / 10.0))
            edges.append(AssumptionEdge(source=wid, target=rid, type=EdgeType.HAS_RESIDUAL, weight=weight))
        if row.get("novel_orientation_needed"):
            hid = stable_id("candidate", pid, row["novel_orientation_needed"])
            cand = AssumptionNode(
                id=hid,
                type=AssumptionType.METHOD,
                claim=row["novel_orientation_needed"],
                context_conditions=[row.get("cluster_tag", "")],
                predicted_effects=[row.get("proposed_refinement", "")],
                tags=["candidate", "generated_from_residual", row.get("cluster_tag", "")],
                confidence=0.35,
                metaproductivity=0.18,
                payload={"source_residual": rid, "source_path": str(residual_path)},
            )
            nodes.append(cand)
            edges.append(AssumptionEdge(source=rid, target=hid, type=EdgeType.GENERATED_FROM_RESIDUAL, weight=0.8))
    return nodes, edges, evidence


def load_exp82_hypotheses(path: str | Path) -> tuple[list[AssumptionNode], list[AssumptionEdge], list[EvidenceRecord]]:
    hypo_path = Path(path)
    if not hypo_path.exists():
        return [], [], []
    nodes: list[AssumptionNode] = []
    edges: list[AssumptionEdge] = []
    evidence: list[EvidenceRecord] = []
    for row in _read_jsonl(hypo_path):
        hid = row.get("hid") or stable_id("hyp", row.get("claim", ""))
        node_id = f"hyp_{hid}"
        kind = _kind_from_exp82(row.get("kind", "claim"))
        ev = row.get("evidence", {}) or {}
        decision = row.get("decision", "deferred")
        node = AssumptionNode(
            id=node_id,
            type=AssumptionType.HARNESS,
            kind=kind,
            claim=row.get("claim", ""),
            formal_form={"expr": row.get("expr"), "kind": row.get("kind")},
            context_conditions=[
                f"trigger_subset:{len(row.get('trigger_subset', []))}",
                f"outside_subset:{len(row.get('outside_subset', []))}",
                row.get("seed_cid", ""),
            ],
            predicted_effects=[
                f"{row.get('expected_metric', 'metric')} should {row.get('expected_direction', 'increase')} by at least {row.get('expected_min_delta', 0)}"
            ],
            risk_predictions=["May be router-confounded; compare against gated GENERIC and outside subset."],
            verifiers=["objective-gold-grade", "cross-judge-strict", "gated-generic-control"],
            confidence=_hypothesis_confidence(row),
            metaproductivity=_hypothesis_metaproductivity(row),
            tags=["exp82", row.get("kind", ""), row.get("seed_cid", ""), decision, row.get("failure_reason") or ""],
            payload={**row, "source_path": str(hypo_path)},
        )
        nodes.append(node)
        if row.get("seed_cid"):
            seed_id = f"candidate_seed_{row['seed_cid']}"
            seed_node = AssumptionNode(
                id=seed_id,
                type=AssumptionType.METHOD,
                claim=f"Original v1 candidate wisdom seed {row['seed_cid']}",
                tags=["candidate_seed", row["seed_cid"]],
                confidence=0.4,
                payload={"seed_cid": row["seed_cid"]},
            )
            nodes.append(seed_node)
            edges.append(AssumptionEdge(source=seed_id, target=node_id, type=EdgeType.DERIVED_FROM, weight=0.75))
        if ev:
            record = EvidenceRecord(
                node_id=node_id,
                source="exp82",
                outcome=_decision_to_outcome(decision),
                metric=row.get("expected_metric", "correctness"),
                value=_primary_delta(ev),
                split="trigger_subset",
                details=ev,
            )
            evidence.append(record)
    return nodes, edges, evidence


def ingest_artifacts(store, artifact_groups: Iterable[tuple[list[AssumptionNode], list[AssumptionEdge], list[EvidenceRecord]]]) -> None:
    for nodes, edges, evidence in artifact_groups:
        for node in nodes:
            clean_tags(node)
            store.upsert_node(node)
        for edge in edges:
            store.add_edge(edge)
        for ev in evidence:
            store.add_evidence(ev)


def clean_tags(node: AssumptionNode) -> None:
    node.tags = [str(t) for t in node.tags if t not in (None, "")]
    node.context_conditions = [str(c) for c in node.context_conditions if c not in (None, "")]
    node.predicted_effects = [str(p) for p in node.predicted_effects if p not in (None, "")]
    node.risk_predictions = [str(r) for r in node.risk_predictions if r not in (None, "")]
    node.source_refs = [str(s) for s in node.source_refs if s not in (None, "")]


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _iter_dict_rows(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict):
                yield key, value
    elif isinstance(obj, list):
        for value in obj:
            if isinstance(value, dict):
                yield value.get("problem_id") or stable_id("row", value), value


def _strategy_conditions(data: dict) -> tuple[list[str], list[str], list[str]]:
    app = data.get("applicability_conditions", {}) or {}
    favorable = [x.get("condition", "") for x in app.get("favorable", []) if isinstance(x, dict)]
    unfavorable = [x.get("condition", "") for x in app.get("unfavorable", []) if isinstance(x, dict)]
    failure_modes = [x.get("description", "") for x in app.get("failure_modes", []) if isinstance(x, dict)]
    return favorable[:10], unfavorable[:8], failure_modes[:8]


def _avg_condition_confidence(data: dict, default: float) -> float:
    vals = []
    app = data.get("applicability_conditions", {}) or {}
    for key in ("favorable", "unfavorable", "failure_modes"):
        for item in app.get(key, []) or []:
            if isinstance(item, dict) and isinstance(item.get("confidence"), (int, float)):
                vals.append(float(item["confidence"]))
    return sum(vals) / len(vals) if vals else default


def _strategy_metaproductivity(data: dict) -> float:
    steps = len(data.get("operational_steps", []) or [])
    aliases = len(data.get("aliases", []) or [])
    return min(0.35, 0.04 + 0.035 * steps + 0.01 * aliases)


def _wisdom_confidence(row: dict) -> float:
    source = row.get("source", "")
    if row.get("status") == "deprecated":
        return 0.25
    if source and source != "original":
        return 0.58
    return 0.5


def _format_source_ref(ref: dict) -> str:
    author = ref.get("author", "")
    work = ref.get("work", "")
    year = ref.get("year", "")
    return " ".join(str(x) for x in (author, work, year) if x)


def _kind_from_exp82(kind: str) -> HypothesisKind:
    return {
        "feature": HypothesisKind.FEATURE,
        "constraint": HypothesisKind.CONSTRAINT,
        "decomposition": HypothesisKind.DECOMPOSITION,
        "verification": HypothesisKind.VERIFICATION,
        "hp_change": HypothesisKind.HP_CHANGE,
    }.get(kind, HypothesisKind.CLAIM)


def _hypothesis_confidence(row: dict) -> float:
    decision = row.get("decision", "deferred")
    ev = row.get("evidence", {}) or {}
    delta = _primary_delta(ev) or 0.0
    if decision == "accepted":
        return min(0.9, 0.62 + max(0.0, delta))
    if decision == "rejected":
        return max(0.12, 0.35 + min(0.0, delta))
    return 0.45 + max(-0.1, min(0.1, delta))


def _hypothesis_metaproductivity(row: dict) -> float:
    kind = row.get("kind")
    if kind == "feature":
        return 0.18
    if kind in {"constraint", "decomposition", "verification", "hp_change"}:
        return 0.14
    return 0.08


def _primary_delta(ev: dict) -> float | None:
    for key in ("delta_ext_base", "trigger_delta", "delta", "delta_ext_generic"):
        if isinstance(ev.get(key), (int, float)):
            return float(ev[key])
    return None


def _decision_to_outcome(decision: str) -> str:
    if decision == "accepted":
        return "accepted"
    if decision == "rejected":
        return "rejected"
    return "deferred"
