"""Formal mapping audit for executable assumption morphisms.

Exp82-style hypotheses already store typed formal forms: feature, constraint,
decomposition, verification, and hp_change.  This module groups those pieces by
their source seed and checks whether they form an executable mapping:

problem signal -> answer transformation -> verification -> runtime policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
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


def search_formal_mappings(
    formal_mapping_payload: dict,
    query: str,
    *,
    top_n: int = 3,
    min_score: float = 1.0,
) -> list[dict]:
    """Search complete formal mappings and return executable applications."""

    text = query.lower()
    applications = []
    for summary in formal_mapping_payload.get("summaries", []):
        if summary.get("status") != FormalMappingStatus.COMPLETE.value:
            continue
        score, matched_keywords, matched_regex = _mapping_trigger_score(summary, text)
        if score < min_score:
            continue
        applications.append({
            "mapping_id": summary.get("mapping_id"),
            "source_key": summary.get("source_key"),
            "score": score,
            "matched_keywords": matched_keywords,
            "matched_regex": matched_regex,
            "constraint_operator": _collect_role_invariants(summary, FormalRole.CONSTRAINT.value),
            "decomposition_operator": _collect_role_invariants(summary, FormalRole.DECOMPOSITION.value),
            "verification_operator": _collect_role_invariants(summary, FormalRole.VERIFICATION.value),
            "runtime_policy": _collect_role_invariants(summary, FormalRole.HP_CHANGE.value),
        })
    applications = sorted(applications, key=lambda a: (-a["score"], a["source_key"]))
    return applications[:top_n]


def format_formal_mapping_applications(applications: list[dict], *, max_items: int = 2) -> str:
    if not applications:
        return ""
    lines = [
        "## Formal Mapping Reasoning",
        "Use these only when their trigger signal fits the current problem.",
    ]
    for app in applications[:max_items]:
        lines.append(f"\n- {app['source_key']} ({app['mapping_id']}, score={app['score']:.1f})")
        if app.get("matched_keywords"):
            lines.append("  Trigger hits: " + ", ".join(app["matched_keywords"][:8]))
        constraints = _first(app.get("constraint_operator", []))
        if constraints:
            required = constraints.get("required_substrings", [])
            if required:
                lines.append("  Preserve constraints: " + "; ".join(required[:6]))
        decomp = _first(app.get("decomposition_operator", []))
        if decomp:
            steps = decomp.get("steps", [])
            if steps:
                lines.append("  Apply steps: " + " -> ".join(str(s) for s in steps[:5]))
        verifier = _first(app.get("verification_operator", []))
        if verifier and verifier.get("instruction"):
            lines.append("  Verify: " + str(verifier["instruction"]))
        runtime = _first(app.get("runtime_policy", []))
        if runtime and runtime.get("has_runtime_policy"):
            knobs = [
                f"{key}={runtime[key]}"
                for key in ("temperature", "top_p", "max_tokens")
                if runtime.get(key) is not None
            ]
            if knobs:
                lines.append("  Runtime hint: " + ", ".join(knobs))
    return "\n".join(lines).strip()


def finite_kernel_metrics(source_kernel: list[list[float]], target_kernel: list[list[float]]) -> dict:
    """Compare two finite stochastic kernels with information-geometry metrics.

    This is deliberately finite and executable: a kernel is a row-stochastic
    matrix over the same object set.  It gives the formal layer a real metric
    substrate without pretending to be a general theorem prover.
    """

    source = _normalize_kernel(source_kernel)
    target = _normalize_kernel(target_kernel)
    same_shape = _same_shape(source, target)
    if not same_shape:
        return {
            "same_shape": False,
            "row_kl_divergence": None,
            "total_variation": None,
            "frobenius_distance": None,
            "blackwell_dominance_proxy": None,
            "warnings": ["kernel shapes differ"],
        }
    rows = len(source)
    cols = len(source[0]) if source else 0
    row_kls = []
    tvs = []
    frob_sum = 0.0
    dominance_rows = 0
    for i in range(rows):
        kl = 0.0
        tv = 0.0
        source_entropy = _entropy(source[i])
        target_entropy = _entropy(target[i])
        if source_entropy <= target_entropy + 1e-9:
            dominance_rows += 1
        for j in range(cols):
            p = source[i][j]
            q = target[i][j]
            if p > 0:
                kl += p * math.log(p / max(q, 1e-12))
            tv += abs(p - q)
            frob_sum += (p - q) ** 2
        row_kls.append(kl)
        tvs.append(0.5 * tv)
    return {
        "same_shape": True,
        "row_kl_divergence": round(sum(row_kls) / rows, 6) if rows else 0.0,
        "max_row_kl_divergence": round(max(row_kls), 6) if row_kls else 0.0,
        "total_variation": round(sum(tvs) / rows, 6) if rows else 0.0,
        "max_total_variation": round(max(tvs), 6) if tvs else 0.0,
        "frobenius_distance": round(math.sqrt(frob_sum), 6),
        "blackwell_dominance_proxy": round(dominance_rows / rows, 6) if rows else 0.0,
        "warnings": [],
    }


def build_categorical_info_geometry_payload(
    formal_mapping_payload: dict,
    *,
    reference_kernel: list[list[float]] | None = None,
) -> dict:
    """Build finite category objects/morphisms and metric summaries."""

    objects = [
        FormalRole.FEATURE.value,
        FormalRole.CONSTRAINT.value,
        FormalRole.DECOMPOSITION.value,
        FormalRole.VERIFICATION.value,
        FormalRole.HP_CHANGE.value,
    ]
    reference = reference_kernel or _ideal_mapping_kernel()
    summaries = []
    for summary in formal_mapping_payload.get("summaries", []):
        kernel = _summary_kernel(summary, objects)
        metrics = finite_kernel_metrics(kernel, reference)
        morphisms = _summary_morphisms(summary, objects)
        warnings = list(summary.get("warnings", []))
        if metrics.get("frobenius_distance") is not None and metrics["frobenius_distance"] > 1.5:
            warnings.append("mapping kernel is far from the reference verifier pipeline")
        summaries.append({
            "mapping_id": summary.get("mapping_id"),
            "source_key": summary.get("source_key"),
            "status": summary.get("status"),
            "objects": objects,
            "morphisms": morphisms,
            "kernel": kernel,
            "reference_kernel": reference,
            "metrics": metrics,
            "warnings": sorted(set(warnings)),
        })
    return {
        "mapping_count": len(summaries),
        "object_count": len(objects),
        "objects": objects,
        "reference_kernel": reference,
        "metric_summary": _metric_summary(summaries),
        "summaries": summaries,
    }


def build_formal_dedup_payload(formal_mapping_payload: dict) -> dict:
    """Find formally equivalent complete mappings and recommend safe merges.

    This pass intentionally recommends merges instead of mutating the graph.  A
    mapping must already be complete before it participates, and equivalence is
    exact over normalized formal invariants.  That keeps the layer useful for
    deduplication without letting vague semantic similarity collapse unrelated
    hypothesis families.
    """

    complete = [
        summary
        for summary in formal_mapping_payload.get("summaries", [])
        if summary.get("status") == FormalMappingStatus.COMPLETE.value
    ]
    grouped: dict[str, list[dict]] = defaultdict(list)
    signature_payloads: dict[str, dict] = {}
    for summary in complete:
        signature = _formal_equivalence_signature(summary)
        signature_id = _signature_id(signature)
        grouped[signature_id].append(summary)
        signature_payloads[signature_id] = signature

    clusters = []
    unique_signatures = []
    for signature_id, rows in sorted(grouped.items(), key=lambda item: item[0]):
        rows = sorted(rows, key=lambda row: (row.get("source_key", ""), row.get("mapping_id", "")))
        canonical = rows[0]
        row = {
            "signature_id": signature_id,
            "formal_signature": signature_payloads[signature_id],
            "mapping_ids": [r.get("mapping_id") for r in rows],
            "source_keys": [r.get("source_key") for r in rows],
            "canonical_mapping_id": canonical.get("mapping_id"),
            "canonical_source_key": canonical.get("source_key"),
            "duplicate_mapping_ids": [r.get("mapping_id") for r in rows[1:]],
            "merge_recommendation_count": max(0, len(rows) - 1),
            "merge_action": "merge_complete_formal_equivalent" if len(rows) > 1 else "keep_unique",
        }
        if len(rows) > 1:
            clusters.append(row)
        else:
            unique_signatures.append(row)

    incomplete_count = formal_mapping_payload.get("mapping_count", 0) - len(complete)
    return {
        "mapping_count": formal_mapping_payload.get("mapping_count", 0),
        "complete_mapping_count": len(complete),
        "incomplete_mapping_excluded_count": max(0, incomplete_count),
        "unique_signature_count": len(grouped),
        "duplicate_cluster_count": len(clusters),
        "merge_recommendation_count": sum(row["merge_recommendation_count"] for row in clusters),
        "clusters": clusters,
        "unique_signatures": unique_signatures,
    }


def _normalize_kernel(kernel: list[list[float]]) -> list[list[float]]:
    normalized = []
    width = max((len(row) for row in kernel), default=0)
    for row in kernel:
        padded = [max(0.0, float(v)) for v in row] + [0.0] * (width - len(row))
        total = sum(padded)
        if total <= 0.0 and width:
            padded = [1.0 / width for _ in range(width)]
        elif total > 0.0:
            padded = [v / total for v in padded]
        normalized.append(padded)
    return normalized


def _same_shape(a: list[list[float]], b: list[list[float]]) -> bool:
    return len(a) == len(b) and all(len(ra) == len(rb) for ra, rb in zip(a, b))


def _entropy(row: list[float]) -> float:
    return -sum(p * math.log(max(p, 1e-12)) for p in row if p > 0)


def _ideal_mapping_kernel() -> list[list[float]]:
    # feature -> constraint/decomposition -> verification -> runtime policy
    return [
        [0.05, 0.45, 0.35, 0.10, 0.05],
        [0.00, 0.10, 0.35, 0.45, 0.10],
        [0.00, 0.10, 0.15, 0.60, 0.15],
        [0.00, 0.05, 0.05, 0.15, 0.75],
        [0.05, 0.05, 0.05, 0.70, 0.15],
    ]


def _summary_kernel(summary: dict, objects: list[str]) -> list[list[float]]:
    present = {
        node.get("role")
        for node in summary.get("nodes", [])
        if node.get("role") in objects
    }
    idx = {role: i for i, role in enumerate(objects)}
    matrix = [[0.0 for _ in objects] for _ in objects]

    def add(src: str, dst: str, weight: float) -> None:
        if src in present and dst in present:
            matrix[idx[src]][idx[dst]] += weight

    add(FormalRole.FEATURE.value, FormalRole.CONSTRAINT.value, 0.45)
    add(FormalRole.FEATURE.value, FormalRole.DECOMPOSITION.value, 0.35)
    add(FormalRole.FEATURE.value, FormalRole.VERIFICATION.value, 0.10)
    add(FormalRole.CONSTRAINT.value, FormalRole.DECOMPOSITION.value, 0.35)
    add(FormalRole.CONSTRAINT.value, FormalRole.VERIFICATION.value, 0.45)
    add(FormalRole.DECOMPOSITION.value, FormalRole.VERIFICATION.value, 0.60)
    add(FormalRole.VERIFICATION.value, FormalRole.HP_CHANGE.value, 0.75)
    add(FormalRole.HP_CHANGE.value, FormalRole.VERIFICATION.value, 0.70)
    for role in present:
        matrix[idx[role]][idx[role]] += 0.1
    return _normalize_kernel(matrix)


def _summary_morphisms(summary: dict, objects: list[str]) -> list[dict]:
    kernel = _summary_kernel(summary, objects)
    morphisms = []
    for i, src in enumerate(objects):
        for j, dst in enumerate(objects):
            weight = kernel[i][j]
            if weight > 0.0:
                morphisms.append({"source": src, "target": dst, "weight": round(weight, 6)})
    return morphisms


def _metric_summary(summaries: list[dict]) -> dict:
    distances = [
        row["metrics"].get("frobenius_distance")
        for row in summaries
        if row["metrics"].get("frobenius_distance") is not None
    ]
    tvs = [
        row["metrics"].get("total_variation")
        for row in summaries
        if row["metrics"].get("total_variation") is not None
    ]
    dominance = [
        row["metrics"].get("blackwell_dominance_proxy")
        for row in summaries
        if row["metrics"].get("blackwell_dominance_proxy") is not None
    ]
    return {
        "mean_frobenius_distance": round(sum(distances) / len(distances), 6) if distances else None,
        "mean_total_variation": round(sum(tvs) / len(tvs), 6) if tvs else None,
        "mean_blackwell_dominance_proxy": round(sum(dominance) / len(dominance), 6) if dominance else None,
    }


def _formal_equivalence_signature(summary: dict) -> dict:
    roles = [
        FormalRole.FEATURE.value,
        FormalRole.CONSTRAINT.value,
        FormalRole.DECOMPOSITION.value,
        FormalRole.VERIFICATION.value,
        FormalRole.HP_CHANGE.value,
    ]
    by_role = {}
    for role in roles:
        invariants = [
            _normalize_signature_value(node.get("invariants", {}))
            for node in summary.get("nodes", [])
            if node.get("role") == role
        ]
        by_role[role] = sorted(invariants, key=lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True))
    return {
        "status": FormalMappingStatus.COMPLETE.value,
        "roles": by_role,
    }


def _normalize_signature_value(value):
    if isinstance(value, dict):
        return {
            str(key): _normalize_signature_value(subvalue)
            for key, subvalue in sorted(value.items())
            if subvalue not in (None, "", [], {})
        }
    if isinstance(value, list):
        return [_normalize_signature_value(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value.strip().lower())
    if isinstance(value, float):
        return round(value, 6)
    return value


def _signature_id(signature: dict) -> str:
    raw = json.dumps(signature, ensure_ascii=False, sort_keys=True)
    return "formal_sig_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


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
            "steps": list(steps),
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


def _mapping_trigger_score(summary: dict, text: str) -> tuple[float, list[str], list[str]]:
    matched_keywords: list[str] = []
    matched_regex: list[str] = []
    for node in summary.get("nodes", []):
        if node.get("role") != FormalRole.FEATURE.value:
            continue
        invariants = node.get("invariants", {})
        for keyword in [*invariants.get("keywords_zh", []), *invariants.get("keywords_en", [])]:
            keyword_s = str(keyword).strip()
            if keyword_s and keyword_s.lower() in text:
                matched_keywords.append(keyword_s)
        for pattern in invariants.get("regex", []):
            try:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    matched_regex.append(pattern)
            except re.error:
                continue
    score = float(len(set(matched_keywords)) + 2 * len(set(matched_regex)))
    return score, sorted(set(matched_keywords)), sorted(set(matched_regex))


def _collect_role_invariants(summary: dict, role: str) -> list[dict]:
    return [
        node.get("invariants", {})
        for node in summary.get("nodes", [])
        if node.get("role") == role
    ]


def _first(items: list[dict]) -> dict:
    return items[0] if items else {}


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
    ap.add_argument("--query", default=None)
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--formal-metrics", action="store_true")
    ap.add_argument("--formal-dedup", action="store_true")
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    payload = build_formal_mapping_payload(
        JsonlGraphStore(_resolve(root, args.graph_dir)),
        node_ids=args.node_ids,
    )
    if args.query:
        payload["search"] = search_formal_mappings(payload, args.query, top_n=args.top_n)
        payload["formatted_search"] = format_formal_mapping_applications(payload["search"])
    if args.formal_metrics:
        payload["categorical_info_geometry"] = build_categorical_info_geometry_payload(payload)
    if args.formal_dedup:
        payload["formal_dedup"] = build_formal_dedup_payload(payload)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
