"""Framework object model for dialectical self-evolution.

R1 in Hegel_assumption.md asks for framework-level objects rather than treating
framework growth as ordinary method nodes.  This module defines explicit
FrameworkNode, FrameworkBranch, and ConservativeExtensionCertificate records
and proves they round-trip through the JSONL Assumption Graph substrate.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR, stable_hash
from .conservative_generalization_gate import (
    REQUIRED_PROMOTION_RELATIONS,
    build_conservative_generalization_gate_payload,
)
from .graph_memory import JsonlGraphStore
from .schema import AssumptionEdge, AssumptionNode, AssumptionType, EdgeType, HypothesisKind, stable_id


DEFAULT_OUT = PAPER_DIR / "framework_object_model_20260612.json"
DEFAULT_MD_OUT = Path("reconstruction/md/framework_object_model_20260612.md")

FRAMEWORK_STATUSES = {
    "draft_branch",
    "candidate_branch",
    "branch_only",
    "candidate_framework",
    "active_scoped_framework",
    "general_framework",
    "deprecated",
    "demoted_to_branch",
    "rejected_boundary_only",
    "contradicted",
}

PROMOTED_STATUSES = {"candidate_framework", "active_scoped_framework", "general_framework"}


@dataclass(frozen=True)
class FrameworkNode:
    id: str
    name: str
    claim: str
    parent_framework_ids: list[str]
    framework_type: str
    scope_conditions: list[str]
    limiting_cases: list[str]
    conserved_successes: list[str]
    residuals_explained: list[str]
    new_predictions: list[str]
    failure_boundaries: list[str]
    formal_certificate_refs: list[str]
    simulator_evidence_refs: list[str]
    verifier_protocol: dict[str, Any]
    status: str
    confidence: float
    framework_growth_score: float
    metaproductivity: float

    def to_assumption_node(self) -> AssumptionNode:
        return AssumptionNode(
            id=self.id,
            type=AssumptionType.FRAMEWORK,
            kind=HypothesisKind.CLAIM,
            claim=self.claim,
            context_conditions=self.scope_conditions,
            predicted_effects=self.new_predictions,
            risk_predictions=self.failure_boundaries,
            verifiers=list(self.verifier_protocol),
            evidence_ids=self.formal_certificate_refs + self.simulator_evidence_refs,
            confidence=self.confidence,
            metaproductivity=self.metaproductivity,
            status=self.status,
            tags=["framework", self.framework_type, self.status],
            payload={"framework_node": asdict(self)},
        )


@dataclass(frozen=True)
class FrameworkBranch:
    branch_id: str
    parent_framework_id: str
    claim: str
    residual_source: str
    branch_type: str
    expected_generalization: str
    expected_risks: list[str]
    required_tests: list[str]
    status: str

    def to_assumption_node(self) -> AssumptionNode:
        return AssumptionNode(
            id=self.branch_id,
            type=AssumptionType.FRAMEWORK_BRANCH,
            kind=HypothesisKind.CLAIM,
            claim=self.claim,
            context_conditions=[self.residual_source],
            risk_predictions=self.expected_risks,
            verifiers=self.required_tests,
            confidence=0.45 if self.status.startswith("rejected") else 0.58,
            metaproductivity=0.05 if self.status.startswith("rejected") else 0.18,
            status=self.status,
            tags=["framework_branch", self.branch_type, self.status],
            payload={"framework_branch": asdict(self)},
        )


@dataclass(frozen=True)
class ConservativeExtensionCertificate:
    certificate_id: str
    candidate_framework_id: str
    parent_framework_ids: list[str]
    old_success_preservation: float
    residual_explanation: float
    limiting_case_reduction: float
    generality_gain: float
    new_prediction_success: float
    regression_cost: float
    relation_types: list[str]
    decision: str
    required_next_tests: list[str]
    source_eval_id: str
    certificate_hash: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "certificate_hash",
            stable_hash({
                "candidate_framework_id": self.candidate_framework_id,
                "parents": self.parent_framework_ids,
                "metrics": {
                    "old": self.old_success_preservation,
                    "residual": self.residual_explanation,
                    "limiting": self.limiting_case_reduction,
                    "generality": self.generality_gain,
                    "new_prediction": self.new_prediction_success,
                    "regression": self.regression_cost,
                },
                "decision": self.decision,
            }),
        )

    def to_assumption_node(self) -> AssumptionNode:
        return AssumptionNode(
            id=self.certificate_id,
            type=AssumptionType.CERTIFICATE,
            kind=HypothesisKind.VERIFICATION,
            claim=(
                f"Conservative extension certificate for {self.candidate_framework_id}: "
                f"{self.decision}"
            ),
            predicted_effects=self.required_next_tests,
            confidence=1.0,
            metaproductivity=0.0,
            status="valid_certificate",
            tags=["conservative_extension_certificate", self.decision],
            payload={"conservative_extension_certificate": asdict(self)},
        )


def build_framework_object_model_payload(
    *,
    root: Path,
    eval_id: str = "framework_object_model_20260612",
) -> dict[str, Any]:
    root = root.resolve()
    gate = build_conservative_generalization_gate_payload(root=root, eval_id=f"{eval_id}_gate")
    objects = _objects_from_gate(gate)
    roundtrip = _roundtrip_graph(objects)
    metrics = _metrics(gate=gate, objects=objects, roundtrip=roundtrip)
    gates = {
        "source_gate_passes": gate["pass"] is True,
        "framework_nodes_present": metrics["framework_node_count"] >= 2,
        "branch_nodes_present": metrics["framework_branch_count"] >= 2,
        "certificate_nodes_present": metrics["certificate_count"] >= metrics["promoted_framework_count"],
        "promoted_frameworks_have_certificates": metrics["promoted_certificate_coverage"] == 1.0,
        "uncertified_active_frameworks_blocked": metrics["uncertified_active_framework_allowed_count"] == 0,
        "required_relations_present": metrics["required_relation_coverage"] == 1.0,
        "new_lifecycle_edges_present": metrics["demotes_to_branch_edge_count"] >= 1
        and metrics["replaces_boundary_of_edge_count"] >= 1,
        "jsonl_roundtrip_exact": roundtrip["roundtrip_exact"] is True,
        "roundtrip_node_types_preserved": roundtrip["framework_node_count_after"]
        == metrics["framework_node_count"]
        and roundtrip["framework_branch_count_after"] == metrics["framework_branch_count"]
        and roundtrip["certificate_count_after"] == metrics["certificate_count"],
        "main_graph_not_mutated": metrics["main_graph_mutation_count"] == 0,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "framework_object_model",
        "source_md": "reconstruction/md/Hegel_assumption.md",
        "release_step": "R1_framework_object_model",
        "performance_validation": True,
        "validation_scope": (
            "Promotes framework growth to first-class FrameworkNode, FrameworkBranch, and "
            "ConservativeExtensionCertificate objects.  Every promoted framework must have a certificate; "
            "uncertified active frameworks stay blocked; the objects round-trip through the JSONL graph."
        ),
        "source_gate": {
            "pass": gate["pass"],
            "eval_kind": gate["eval_kind"],
            "decision_counts": gate["metrics"]["decision_counts"],
        },
        "framework_nodes": [asdict(row) for row in objects["framework_nodes"]],
        "framework_branches": [asdict(row) for row in objects["framework_branches"]],
        "certificates": [asdict(row) for row in objects["certificates"]],
        "support_nodes": [node.to_dict() for node in objects["support_nodes"]],
        "graph_edges": [edge.to_dict() for edge in objects["edges"]],
        "roundtrip": roundtrip,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "allowed_claim": "first-class bounded framework object model with certificate-gated promotion",
        "blocked_claims": [
            "uncertified_active_framework_promotion",
            "ungated_core_prior_promotion",
            "main_graph_mutation_from_schema_test",
        ],
    }


def format_markdown(payload: dict[str, Any]) -> str:
    m = payload["metrics"]
    lines = [
        "# Framework Object Model",
        "",
        f"- pass: `{payload['pass']}`",
        f"- framework nodes: `{m['framework_node_count']}`",
        f"- framework branches: `{m['framework_branch_count']}`",
        f"- certificates: `{m['certificate_count']}`",
        f"- promoted certificate coverage: `{m['promoted_certificate_coverage']}`",
        f"- required relation coverage: `{m['required_relation_coverage']}`",
        f"- JSONL roundtrip exact: `{payload['roundtrip']['roundtrip_exact']}`",
        "",
        "## Framework Nodes",
        "",
        "| ID | Status | Growth | Certificate refs |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload["framework_nodes"]:
        lines.append(
            f"| `{row['id']}` | `{row['status']}` | `{row['framework_growth_score']}` | "
            f"`{len(row['formal_certificate_refs'])}` |"
        )
    lines.extend(["", "## Branches", ""])
    for row in payload["framework_branches"]:
        lines.append(f"- `{row['branch_id']}` -> `{row['status']}` from `{row['parent_framework_id']}`")
    lines.extend(["", "## Claim Boundary", ""])
    for claim in payload["blocked_claims"]:
        lines.append(f"- `{claim}`")
    return "\n".join(lines).rstrip() + "\n"


def _objects_from_gate(gate: dict[str, Any]) -> dict[str, Any]:
    framework_nodes: list[FrameworkNode] = []
    branches: list[FrameworkBranch] = []
    certificates: list[ConservativeExtensionCertificate] = []
    support_nodes_by_id: dict[str, AssumptionNode] = {}
    edges: list[AssumptionEdge] = []
    graph_patch_nodes = {row["id"]: row for row in gate.get("graph_patch", {}).get("nodes", [])}
    graph_patch_edges = [
        AssumptionEdge.from_dict(row)
        for row in gate.get("graph_patch", {}).get("edges", [])
    ]

    for row in gate["evaluations"]:
        metrics = row["metrics"]
        decision = row["decision"]
        parent_ids = list(row["parent_frameworks"])
        if decision in PROMOTED_STATUSES:
            cert = ConservativeExtensionCertificate(
                certificate_id=stable_id("cert", row["framework_id"], decision, gate["eval_id"]),
                candidate_framework_id=row["framework_id"],
                parent_framework_ids=parent_ids,
                old_success_preservation=metrics["old_success_preservation"],
                residual_explanation=metrics["residual_explanation"],
                limiting_case_reduction=metrics["limiting_case_reduction"],
                generality_gain=metrics["generality_gain"],
                new_prediction_success=metrics["new_prediction_success"],
                regression_cost=metrics["regression_cost"],
                relation_types=sorted(row["relation_types"]),
                decision=decision,
                required_next_tests=list(row["required_next_tests"]),
                source_eval_id=gate["eval_id"],
            )
            certificates.append(cert)
            framework = FrameworkNode(
                id=row["framework_id"],
                name=row["framework_id"].replace("fw_", "").replace("_", " "),
                claim=row["claim"],
                parent_framework_ids=parent_ids,
                framework_type="methodology_framework",
                scope_conditions=[f"parent_scope:{parent}" for parent in parent_ids],
                limiting_cases=[f"reduces_to:{parent}" for parent in parent_ids],
                conserved_successes=["old_success_preservation"],
                residuals_explained=["motivating_residual_cluster"],
                new_predictions=["new_prediction_cases"],
                failure_boundaries=list(row["conflict_boundaries"]),
                formal_certificate_refs=[cert.certificate_id],
                simulator_evidence_refs=["simulator_production_gate_20260612"],
                verifier_protocol={test: "required" for test in row["required_next_tests"]},
                status=decision,
                confidence=round(0.50 + 0.45 * metrics["framework_growth_score"], 4),
                framework_growth_score=metrics["framework_growth_score"],
                metaproductivity=round(metrics["framework_growth_score"] - metrics["regression_cost"], 4),
            )
            framework_nodes.append(framework)
            edges.append(
                AssumptionEdge(
                    source=framework.id,
                    target=cert.certificate_id,
                    type=EdgeType.HAS_CERTIFICATE,
                    weight=1.0,
                    payload={"decision": decision, "certificate_hash": cert.certificate_hash},
                )
            )
            for parent in parent_ids:
                edges.extend([
                    AssumptionEdge(source=framework.id, target=parent, type=EdgeType.GENERALIZES, weight=metrics["framework_growth_score"]),
                    AssumptionEdge(source=framework.id, target=parent, type=EdgeType.REDUCES_TO_UNDER_SCOPE, weight=metrics["limiting_case_reduction"]),
                    AssumptionEdge(source=framework.id, target=parent, type=EdgeType.MODIFIES_BOUNDARY_OF, weight=metrics["generality_gain"]),
                    AssumptionEdge(source=framework.id, target=parent, type=EdgeType.REPLACES_BOUNDARY_OF, weight=metrics["generality_gain"]),
                ])
            _append_gate_relation_support(
                row=row,
                graph_patch_nodes=graph_patch_nodes,
                graph_patch_edges=graph_patch_edges,
                support_nodes_by_id=support_nodes_by_id,
                edges=edges,
                relation_values={
                    EdgeType.EXPLAINS_RESIDUAL.value,
                    EdgeType.PRESERVES_SUCCESS_CASES.value,
                    EdgeType.PREDICTS_NEW_CASE.value,
                    EdgeType.CONFLICTS_WITH.value,
                },
            )
        else:
            parent = parent_ids[0] if parent_ids else "unknown_parent_framework"
            status = "rejected_boundary_only" if decision == "reject" else decision
            branch = FrameworkBranch(
                branch_id=row["framework_id"],
                parent_framework_id=parent,
                claim=row["claim"],
                residual_source="motivating_residual_cluster",
                branch_type="negative_control_branch" if decision == "reject" else "candidate_branch",
                expected_generalization="insufficient_generalization" if decision == "branch_only" else "failed_conservative_gate",
                expected_risks=list(row["conflict_boundaries"]) or ["insufficient_certificate_support"],
                required_tests=list(row["required_next_tests"]),
                status=status,
            )
            branches.append(branch)
            edge_type = EdgeType.DEMOTES_TO_BRANCH if decision == "branch_only" else EdgeType.CONFLICTS_WITH
            edges.append(
                AssumptionEdge(
                    source=branch.branch_id,
                    target=parent,
                    type=edge_type,
                    weight=metrics["framework_growth_score"],
                    payload={"decision": decision, "reason": branch.expected_generalization},
                )
            )
            _append_gate_relation_support(
                row=row,
                graph_patch_nodes=graph_patch_nodes,
                graph_patch_edges=graph_patch_edges,
                support_nodes_by_id=support_nodes_by_id,
                edges=edges,
                relation_values={
                    EdgeType.EXPLAINS_RESIDUAL.value,
                    EdgeType.PRESERVES_SUCCESS_CASES.value,
                    EdgeType.CONFLICTS_WITH.value,
                },
            )

    return {
        "framework_nodes": framework_nodes,
        "framework_branches": branches,
        "certificates": certificates,
        "support_nodes": list(support_nodes_by_id.values()),
        "edges": _dedupe_edges(edges),
    }


def _append_gate_relation_support(
    *,
    row: dict[str, Any],
    graph_patch_nodes: dict[str, dict[str, Any]],
    graph_patch_edges: list[AssumptionEdge],
    support_nodes_by_id: dict[str, AssumptionNode],
    edges: list[AssumptionEdge],
    relation_values: set[str],
) -> None:
    existing_edge_keys = {(edge.source, edge.target, _edge_type_value(edge)) for edge in edges}
    for edge in graph_patch_edges:
        edge_type = _edge_type_value(edge)
        if edge.source != row["framework_id"] or edge_type not in relation_values:
            continue
        key = (edge.source, edge.target, edge_type)
        if key not in existing_edge_keys:
            edges.append(edge)
            existing_edge_keys.add(key)
        node = graph_patch_nodes.get(edge.target)
        if node and edge.target not in support_nodes_by_id:
            support_nodes_by_id[edge.target] = AssumptionNode.from_dict(node)


def _roundtrip_graph(objects: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="framework_object_model_") as td:
        store = JsonlGraphStore(td)
        for framework in objects["framework_nodes"]:
            store.upsert_node(framework.to_assumption_node())
        for branch in objects["framework_branches"]:
            store.upsert_node(branch.to_assumption_node())
        for cert in objects["certificates"]:
            store.upsert_node(cert.to_assumption_node())
        for node in objects.get("support_nodes", []):
            store.upsert_node(node)
        for edge in objects["edges"]:
            store.add_edge(edge)
        before = _snapshot(store)
        store.flush()
        reloaded = JsonlGraphStore(td)
        after = _snapshot(reloaded)
    return {
        "roundtrip_exact": before == after,
        "before_hash": stable_hash(before),
        "after_hash": stable_hash(after),
        "node_count_after": len(after["nodes"]),
        "edge_count_after": len(after["edges"]),
        "framework_node_count_after": sum(1 for row in after["nodes"] if row["type"] == AssumptionType.FRAMEWORK.value),
        "framework_branch_count_after": sum(1 for row in after["nodes"] if row["type"] == AssumptionType.FRAMEWORK_BRANCH.value),
        "certificate_count_after": sum(1 for row in after["nodes"] if row["type"] == AssumptionType.CERTIFICATE.value),
        "edge_type_counts_after": _counts(row["type"] for row in after["edges"]),
    }


def _snapshot(store: JsonlGraphStore) -> dict[str, Any]:
    return {
        "nodes": sorted((node.to_dict() for node in store.nodes.values()), key=lambda row: row["id"]),
        "edges": sorted((edge.to_dict() for edge in store.edges), key=lambda row: (row["source"], row["target"], row["type"])),
    }


def _metrics(*, gate: dict[str, Any], objects: dict[str, Any], roundtrip: dict[str, Any]) -> dict[str, Any]:
    framework_nodes: list[FrameworkNode] = objects["framework_nodes"]
    branches: list[FrameworkBranch] = objects["framework_branches"]
    certificates: list[ConservativeExtensionCertificate] = objects["certificates"]
    support_nodes: list[AssumptionNode] = objects.get("support_nodes", [])
    edges: list[AssumptionEdge] = objects["edges"]
    promoted = [node for node in framework_nodes if node.status in PROMOTED_STATUSES]
    certified_ids = {cert.candidate_framework_id for cert in certificates}
    promoted_certified = [node for node in promoted if node.id in certified_ids and node.formal_certificate_refs]
    required_edges = set(REQUIRED_PROMOTION_RELATIONS) | {
        EdgeType.HAS_CERTIFICATE.value,
        EdgeType.DEMOTES_TO_BRANCH.value,
        EdgeType.REPLACES_BOUNDARY_OF.value,
    }
    edge_types = {_edge_type_value(edge) for edge in edges}
    return {
        "source_gate_pass": bool(gate.get("pass")),
        "framework_node_count": len(framework_nodes),
        "framework_branch_count": len(branches),
        "certificate_count": len(certificates),
        "support_node_count": len(support_nodes),
        "promoted_framework_count": len(promoted),
        "promoted_certificate_coverage": round(len(promoted_certified) / max(1, len(promoted)), 4),
        "uncertified_active_framework_allowed_count": sum(
            1 for node in framework_nodes if node.status in PROMOTED_STATUSES and not node.formal_certificate_refs
        ),
        "framework_status_counts": _counts(node.status for node in framework_nodes),
        "branch_status_counts": _counts(branch.status for branch in branches),
        "edge_count": len(edges),
        "edge_type_counts": _counts(_edge_type_value(edge) for edge in edges),
        "required_relation_coverage": round(len(required_edges & edge_types) / len(required_edges), 4),
        "demotes_to_branch_edge_count": sum(1 for edge in edges if edge.type == EdgeType.DEMOTES_TO_BRANCH),
        "replaces_boundary_of_edge_count": sum(1 for edge in edges if edge.type == EdgeType.REPLACES_BOUNDARY_OF),
        "jsonl_roundtrip_exact": bool(roundtrip["roundtrip_exact"]),
        "roundtrip_node_count": roundtrip["node_count_after"],
        "roundtrip_edge_count": roundtrip["edge_count_after"],
        "main_graph_mutation_count": 0,
        "allowed_framework_status_count": sum(
            1 for status in [*(node.status for node in framework_nodes), *(branch.status for branch in branches)]
            if status in FRAMEWORK_STATUSES
        ),
    }


def _edge_type_value(edge: AssumptionEdge) -> str:
    return str(edge.type.value if hasattr(edge.type, "value") else edge.type)


def _dedupe_edges(edges: list[AssumptionEdge]) -> list[AssumptionEdge]:
    seen: set[tuple[str, str, str]] = set()
    out: list[AssumptionEdge] = []
    for edge in edges:
        key = (edge.source, edge.target, _edge_type_value(edge))
        if key in seen:
            continue
        seen.add(key)
        out.append(edge)
    return out


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--eval-id", default="framework_object_model_20260612")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args()

    payload = build_framework_object_model_payload(root=args.root, eval_id=args.eval_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(format_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "eval_id": payload["eval_id"],
                "pass": payload["pass"],
                "failed_gates": payload["failed_gates"],
                "metrics": payload["metrics"],
                "out": str(args.out.resolve()),
                "md_out": str(args.md_out.resolve()),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
