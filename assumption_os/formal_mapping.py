"""Formal mapping audit for executable assumption morphisms.

Exp82-style hypotheses already store typed formal forms: feature, constraint,
decomposition, verification, and hp_change.  This module groups those pieces by
their source seed and checks whether they form an executable mapping:

problem signal -> answer transformation -> verification -> runtime policy.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from .graph_memory import JsonlGraphStore
from .schema import AssumptionNode


class FormalRole(str, Enum):
    FEATURE = "feature"
    CONSTRAINT = "constraint"
    DECOMPOSITION = "decomposition"
    VERIFICATION = "verification"
    HP_CHANGE = "hp_change"
    FORMAL_MAPPING = "formal_mapping"
    UNKNOWN = "unknown"


class FormalMappingStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNSAFE = "unsafe"


class FormalMappingGateDecision(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    ALLOW = "allow"
    REPAIR_BEFORE_PROMOTION = "repair_before_promotion"
    BLOCK_UNSAFE_MAPPING = "block_unsafe_mapping"


@dataclass(frozen=True)
class FormalNodeView:
    node_id: str
    role: FormalRole
    claim: str
    invariants: dict

    def to_dict(self) -> dict:
        d = asdict(self)
        d["role"] = self.role.value
        return d


@dataclass(frozen=True)
class FormalMappingSummary:
    mapping_id: str
    source_key: str
    status: FormalMappingStatus
    roles: dict[str, list[str]]
    invariants: dict
    warnings: list[str] = field(default_factory=list)
    nodes: list[FormalNodeView] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["nodes"] = [n.to_dict() for n in self.nodes]
        return d


def build_formal_mapping_payload(store: JsonlGraphStore, *, node_ids: list[str] | None = None) -> dict:
    nodes = [
        node
        for node in store.nodes.values()
        if (not node_ids or node.id in node_ids) and _formal_role(node) != FormalRole.UNKNOWN
    ]
    grouped: dict[str, list[AssumptionNode]] = defaultdict(list)
    for node in nodes:
        grouped[_source_key(node)].append(node)
    summaries = [_summarize_group(key, group) for key, group in grouped.items()]
    summaries = sorted(summaries, key=lambda s: (s.status.value, s.source_key))
    return {
        "nodes_considered": len(nodes),
        "mapping_count": len(summaries),
        "status_counts": dict(Counter(s.status.value for s in summaries)),
        "role_counts": dict(Counter(role for s in summaries for role, ids in s.roles.items() for _ in ids)),
        "summaries": [s.to_dict() for s in summaries],
    }


def build_formal_mapping_gate_payload(*, proposal_payload: dict, formal_mapping_payload: dict) -> dict:
    gates = [
        _proposal_gate(proposal, formal_mapping_payload)
        for proposal in proposal_payload.get("proposals", [])
    ]
    return {
        "source_formal_mapping_count": formal_mapping_payload.get("mapping_count", 0),
        "decision_counts": dict(Counter(g["decision"] for g in gates)),
        "blocked_proposal_ids": sorted(
            g["proposal_id"] for g in gates if g.get("blocks_policy_update")
        ),
        "gates": gates,
    }


def _summarize_group(source_key: str, nodes: list[AssumptionNode]) -> FormalMappingSummary:
    views = [_node_view(node) for node in nodes]
    roles: dict[str, list[str]] = defaultdict(list)
    for view in views:
        roles[view.role.value].append(view.node_id)
    invariants = _mapping_invariants(views)
    warnings = _mapping_warnings(invariants)
    if warnings and "missing trigger detector" in warnings:
        status = FormalMappingStatus.UNSAFE
    elif warnings:
        status = FormalMappingStatus.PARTIAL
    else:
        status = FormalMappingStatus.COMPLETE
    return FormalMappingSummary(
        mapping_id=f"formal_map::{source_key}",
        source_key=source_key,
        status=status,
        roles={role: sorted(ids) for role, ids in sorted(roles.items())},
        invariants=invariants,
        warnings=warnings,
        nodes=sorted(views, key=lambda v: (v.role.value, v.node_id)),
    )


def _node_view(node: AssumptionNode) -> FormalNodeView:
    return FormalNodeView(
        node_id=node.id,
        role=_formal_role(node),
        claim=node.claim,
        invariants=_node_invariants(node),
    )


def _formal_role(node: AssumptionNode) -> FormalRole:
    formal = node.formal_form or {}
    kind = formal.get("kind") if isinstance(formal, dict) else None
    if not kind and isinstance(node.kind, str):
        kind = node.kind
    value = getattr(kind, "value", kind)
    try:
        return FormalRole(value)
    except Exception:
        return FormalRole.UNKNOWN


def _source_key(node: AssumptionNode) -> str:
    payload = node.payload or {}
    if payload.get("seed_cid"):
        return str(payload["seed_cid"])
    for tag in node.tags:
        if isinstance(tag, str) and tag.startswith("WC"):
            return tag
    return node.id


def _node_invariants(node: AssumptionNode) -> dict:
    formal = node.formal_form or {}
    expr = formal.get("expr", {}) if isinstance(formal, dict) else {}
    role = _formal_role(node)
    if role == FormalRole.FEATURE:
        return {
            "keywords_zh": sorted(expr.get("keywords_zh", [])),
            "keywords_en": sorted(expr.get("keywords_en", [])),
            "regex": sorted(expr.get("regex", [])),
            "has_trigger": bool(expr.get("keywords_zh") or expr.get("keywords_en") or expr.get("regex")),
        }
    if role == FormalRole.CONSTRAINT:
        return {
            "required_substrings": sorted(expr.get("required_substrings", [])),
            "forbidden_substrings": sorted(expr.get("forbidden_substrings", [])),
            "has_constraint": bool(expr.get("required_substrings") or expr.get("forbidden_substrings")),
        }
    if role == FormalRole.DECOMPOSITION:
        steps = expr.get("steps", [])
        return {
            "step_count": len(steps),
            "has_ordered_operator": bool(steps),
        }
    if role == FormalRole.VERIFICATION:
        return {
            "instruction": expr.get("instruction", ""),
            "has_verifier": bool(expr.get("instruction")),
        }
    if role == FormalRole.HP_CHANGE:
        return {
            "temperature": expr.get("temperature"),
            "top_p": expr.get("top_p"),
            "max_tokens": expr.get("max_tokens"),
            "has_runtime_policy": any(key in expr for key in ("temperature", "top_p", "max_tokens")),
        }
    return {}


def _mapping_invariants(views: list[FormalNodeView]) -> dict:
    by_role = defaultdict(list)
    for view in views:
        by_role[view.role].append(view)
    return {
        "trigger_detector": any(v.invariants.get("has_trigger") for v in by_role[FormalRole.FEATURE]),
        "constraint_operator": any(v.invariants.get("has_constraint") for v in by_role[FormalRole.CONSTRAINT]),
        "decomposition_operator": any(v.invariants.get("has_ordered_operator") for v in by_role[FormalRole.DECOMPOSITION]),
        "verification_operator": any(v.invariants.get("has_verifier") for v in by_role[FormalRole.VERIFICATION]),
        "runtime_policy": any(v.invariants.get("has_runtime_policy") for v in by_role[FormalRole.HP_CHANGE]),
    }


def _mapping_warnings(invariants: dict) -> list[str]:
    warnings = []
    if not invariants.get("trigger_detector"):
        warnings.append("missing trigger detector")
    if not (invariants.get("constraint_operator") or invariants.get("decomposition_operator")):
        warnings.append("missing answer transformation operator")
    if not invariants.get("verification_operator"):
        warnings.append("missing verifier")
    if not invariants.get("runtime_policy"):
        warnings.append("missing runtime policy")
    return warnings


def _proposal_gate(proposal: dict, formal_mapping_payload: dict) -> dict:
    matches = _proposal_mapping_matches(proposal, formal_mapping_payload)
    statuses = sorted({m["status"] for m in matches})
    if not matches:
        decision = FormalMappingGateDecision.NOT_APPLICABLE
    elif FormalMappingStatus.UNSAFE.value in statuses:
        decision = FormalMappingGateDecision.BLOCK_UNSAFE_MAPPING
    elif FormalMappingStatus.PARTIAL.value in statuses:
        decision = FormalMappingGateDecision.REPAIR_BEFORE_PROMOTION
    else:
        decision = FormalMappingGateDecision.ALLOW
    return {
        "proposal_id": proposal.get("proposal_id"),
        "proposal_type": proposal.get("proposal_type"),
        "parent_node_id": proposal.get("parent_node_id"),
        "decision": decision.value,
        "blocks_policy_update": decision in {
            FormalMappingGateDecision.REPAIR_BEFORE_PROMOTION,
            FormalMappingGateDecision.BLOCK_UNSAFE_MAPPING,
        },
        "mapping_ids": sorted({m["mapping_id"] for m in matches}),
        "source_keys": sorted({m["source_key"] for m in matches}),
        "mapping_statuses": statuses,
        "warnings": sorted({w for m in matches for w in m.get("warnings", [])}),
    }


def _proposal_mapping_matches(proposal: dict, formal_mapping_payload: dict) -> list[dict]:
    by_node_id, by_source_key = _mapping_indexes(formal_mapping_payload)
    node_ids = {proposal.get("parent_node_id")}
    source_keys: set[str] = set()
    candidate = proposal.get("candidate_node") or {}
    if candidate:
        node_ids.add(candidate.get("id"))
        source_keys.update(_node_source_keys(candidate))
    source_action = proposal.get("source_action") or {}
    source_keys.update(_node_source_keys(source_action))

    matches = []
    seen = set()
    for node_id in sorted(x for x in node_ids if x):
        for match in by_node_id.get(str(node_id), []):
            key = (match["mapping_id"], match["status"])
            if key not in seen:
                seen.add(key)
                matches.append(match)
    for source_key in sorted(source_keys):
        for match in by_source_key.get(source_key, []):
            key = (match["mapping_id"], match["status"])
            if key not in seen:
                seen.add(key)
                matches.append(match)
    return matches


def _mapping_indexes(formal_mapping_payload: dict) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    by_node_id: dict[str, list[dict]] = defaultdict(list)
    by_source_key: dict[str, list[dict]] = defaultdict(list)
    for summary in formal_mapping_payload.get("summaries", []):
        view = {
            "mapping_id": summary.get("mapping_id", ""),
            "source_key": summary.get("source_key", ""),
            "status": summary.get("status", ""),
            "warnings": summary.get("warnings", []),
        }
        if view["source_key"]:
            by_source_key[view["source_key"]].append(view)
        for node in summary.get("nodes", []):
            node_id = node.get("node_id")
            if node_id:
                by_node_id[node_id].append(view)
    return by_node_id, by_source_key


def _node_source_keys(node: dict) -> set[str]:
    keys = set()
    payload = node.get("payload") if isinstance(node, dict) else None
    if isinstance(payload, dict) and payload.get("seed_cid"):
        keys.add(str(payload["seed_cid"]))
    for tag in node.get("tags", []) if isinstance(node, dict) else []:
        if isinstance(tag, str) and tag.startswith("WC"):
            keys.add(tag)
    return keys


def _resolve(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--graph-dir", default="phase four/assumption_graph")
    ap.add_argument("--node-ids", nargs="*", default=None)
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    payload = build_formal_mapping_payload(
        JsonlGraphStore(_resolve(root, args.graph_dir)),
        node_ids=args.node_ids,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
