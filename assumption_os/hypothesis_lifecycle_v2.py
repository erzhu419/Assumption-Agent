"""V2 hypothesis lifecycle schema.

This module makes the reconstruction-v2 distinction explicit:

* a hypothesis is first a manifest contract;
* the graph only stores a projection of that contract;
* process/alignment/world-model payloads are optional formal payloads;
* relation hypotheses are represented as nodes, not overloaded bare edges.

The objects here are deliberately compatible with the existing AssumptionNode /
AssumptionEdge / TrialManifest substrate, so v2 can be introduced without
rewriting the old recursive runner.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .schema import (
    AssumptionEdge,
    AssumptionNode,
    AssumptionType,
    EdgeType,
    HypothesisKind,
    TrialManifest,
    TrialStatus,
    stable_id,
)


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "hypothesis_lifecycle_v2_schema_20260610.json"


@dataclass(frozen=True)
class VerifierContract:
    cheap: list[str] = field(default_factory=list)
    world_model: list[str] = field(default_factory=list)
    live: list[str] = field(default_factory=list)
    rollback: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerifierContract":
        return cls(
            cheap=list(data.get("cheap", [])),
            world_model=list(data.get("world_model", [])),
            live=list(data.get("live", [])),
            rollback=str(data.get("rollback", "")),
        )


@dataclass(frozen=True)
class GraphOverlayOp:
    op: str
    node: dict[str, Any] | None = None
    edge: dict[str, Any] | None = None
    rollback_ref: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AssumptionManifestV2:
    id: str
    type: str
    claim: str
    context_conditions: list[str]
    predicted_effects: list[str]
    risk_predictions: list[str]
    formal_refs: list[str]
    graph_ops: list[GraphOverlayOp]
    verifier: VerifierContract
    evidence_refs: list[str] = field(default_factory=list)
    residual_refs: list[str] = field(default_factory=list)
    confidence: float = 0.5
    metaproductivity: float | None = None
    status: str = "candidate"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["graph_ops"] = [op.to_dict() for op in self.graph_ops]
        data["verifier"] = self.verifier.to_dict()
        return data

    def validate(self) -> list[str]:
        issues = []
        if not self.id:
            issues.append("manifest_id_missing")
        if not self.claim:
            issues.append("claim_missing")
        if not self.context_conditions:
            issues.append("context_conditions_missing")
        if not self.predicted_effects:
            issues.append("predicted_effects_missing")
        if not self.risk_predictions:
            issues.append("risk_predictions_missing")
        if not self.graph_ops:
            issues.append("graph_ops_missing")
        if not (self.verifier.cheap or self.verifier.world_model or self.verifier.live):
            issues.append("verifier_tests_missing")
        if self.status not in {"candidate", "active", "rejected", "deprecated", "contradicted"}:
            issues.append("invalid_status")
        return issues


@dataclass(frozen=True)
class ProcessModel:
    id: str
    domain: str
    state_variables: list[str]
    parameters: list[str]
    interventions: list[str]
    perturbation: str
    response: str
    dynamics: str | dict[str, Any]
    observation_map: str | dict[str, Any]
    invariants: list[str]
    failure_cases: list[str]
    traces: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> list[str]:
        issues = []
        required = {
            "id": self.id,
            "domain": self.domain,
            "perturbation": self.perturbation,
            "response": self.response,
        }
        for name, value in required.items():
            if not value:
                issues.append(f"{name}_missing")
        if not self.state_variables:
            issues.append("state_variables_missing")
        if not self.interventions:
            issues.append("interventions_missing")
        if not self.invariants:
            issues.append("invariants_missing")
        if not self.failure_cases:
            issues.append("failure_cases_missing")
        return issues

    def to_node(self) -> AssumptionNode:
        return AssumptionNode(
            id=self.id,
            type=AssumptionType.PROCESS,
            kind=HypothesisKind.PROCESS_MODEL,
            claim=f"Process model for {self.domain}: {self.perturbation} -> {self.response}",
            formal_form={"formal_kind": "process_model_v2", **self.to_dict()},
            context_conditions=[self.domain],
            predicted_effects=[
                "makes mechanism-level alignment testable beyond node-level analogy"
            ],
            risk_predictions=list(self.failure_cases),
            verifiers=["process_model_schema", "invariant_check"],
            tags=["process_model", self.domain],
            payload={"v2_object": "ProcessModel"},
        )


@dataclass(frozen=True)
class AlignmentHypothesis:
    id: str
    source_process: str
    target_process: str
    mapping: dict[str, str]
    preserved_structure: list[str]
    broken_structure: list[str]
    metric_scores: dict[str, float]
    verifier_tests: list[str]
    status: str = "candidate"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> list[str]:
        issues = []
        if not self.id:
            issues.append("alignment_id_missing")
        if not self.source_process or not self.target_process:
            issues.append("process_endpoint_missing")
        if len(self.mapping) < 2:
            issues.append("mapping_too_small")
        if not self.preserved_structure:
            issues.append("preserved_structure_missing")
        if not self.broken_structure:
            issues.append("broken_structure_missing")
        if not self.verifier_tests:
            issues.append("verifier_tests_missing")
        if self.status not in {"candidate", "active", "rejected", "deprecated", "contradicted"}:
            issues.append("invalid_status")
        return issues

    def to_relation_node(self, *, claim: str) -> AssumptionNode:
        return AssumptionNode(
            id=self.id,
            type=AssumptionType.ALIGNMENT,
            kind=HypothesisKind.ALIGNMENT_HYPOTHESIS,
            claim=claim,
            formal_form={"formal_kind": "alignment_hypothesis_v2", **self.to_dict()},
            context_conditions=[
                "source and target process models have typed perturbation-response structure"
            ],
            predicted_effects=[
                "supports transfer when preserved structure is activated",
                "allows bounded negative controls through broken_structure fields",
            ],
            risk_predictions=[
                "may over-transfer if broken structures are ignored",
                "may be only surface analogy without downstream verifier pass",
            ],
            verifiers=list(self.verifier_tests),
            tags=["alignment_hypothesis", "relation_node"],
            payload={"v2_object": "AlignmentHypothesis"},
        )


@dataclass(frozen=True)
class WorldModelTrial:
    id: str
    state_summary: dict[str, Any]
    action: dict[str, Any]
    predicted_accept_prob: float
    predicted_regression_prob: float
    predicted_failure_type: str
    predicted_value_delta: float
    actual_outcome: dict[str, Any] | None = None
    calibration_error: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> list[str]:
        issues = []
        if not self.id:
            issues.append("trial_id_missing")
        if not self.state_summary:
            issues.append("state_summary_missing")
        if not self.action:
            issues.append("action_missing")
        for name, value in [
            ("predicted_accept_prob", self.predicted_accept_prob),
            ("predicted_regression_prob", self.predicted_regression_prob),
        ]:
            if value < 0.0 or value > 1.0:
                issues.append(f"{name}_out_of_range")
        if not self.predicted_failure_type:
            issues.append("predicted_failure_type_missing")
        return issues


def build_hypothesis_lifecycle_v2_payload(*, eval_id: str = "hypothesis_lifecycle_v2_schema_20260610") -> dict[str, Any]:
    """Build a small v2 fixture around Le Chatelier / Lenz process alignment."""

    le_chatelier = _le_chatelier_process()
    lenz = _lenz_process()
    alignment = _negative_feedback_alignment(le_chatelier, lenz)
    verifier = VerifierContract(
        cheap=[
            "check typed mapping includes perturbation, response, and opposition relation",
            "check preserved_structure does not claim thermodynamic/electromagnetic equation identity",
        ],
        world_model=[
            "predict downstream analogy benefit and regression risk before live validation",
            "estimate whether adding the relation node is likely to be a low-benefit tie",
        ],
        live=[
            "fresh analogy/explanation tasks with placebo relation controls",
            "negative-control tasks where equilibrium restoration is absent",
        ],
        rollback="Do not merge the relation node if verifier fails or live transfer regresses.",
    )
    overlay_ops = build_alignment_overlay_ops(
        source_process=le_chatelier,
        target_process=lenz,
        alignment=alignment,
        verifier=verifier,
    )
    manifest = AssumptionManifestV2(
        id=stable_id("h2", alignment.id, "manifest"),
        type="alignment",
        claim=(
            "Le Chatelier principle and Lenz's law share a bounded perturbation-opposition "
            "process schema, while preserving explicit domain-specific breaks."
        ),
        context_conditions=[
            "both processes expose perturbation and compensatory response roles",
            "transfer task asks for mechanism-level analogy rather than equation identity",
        ],
        predicted_effects=[
            "improves cross-domain negative-feedback analogy retrieval",
            "reduces hallucinated identity claims by storing broken structures",
        ],
        risk_predictions=[
            "may over-generalize equilibrium restoration to open-circuit or irreversible cases",
            "may add prompt length without downstream answer benefit",
        ],
        formal_refs=[le_chatelier.id, lenz.id, alignment.id],
        graph_ops=overlay_ops,
        verifier=verifier,
        confidence=0.62,
        metaproductivity=0.0,
    )
    trial = WorldModelTrial(
        id=stable_id("wmtrial", alignment.id, "add_relation_node"),
        state_summary={
            "active_process_nodes": 2,
            "active_alignment_nodes": 0,
            "residual_cluster": "cross_domain_negative_feedback_analogy",
            "budget_state": "pre_live_screen_required",
        },
        action={
            "type": "add_alignment_hypothesis",
            "graph_ops": [op.to_dict() for op in overlay_ops],
            "counterfactual_masks": [
                {"type": "mask_alignment_node", "target": alignment.id},
                {"type": "mask_process_payload", "target": le_chatelier.id},
                {"type": "mask_process_payload", "target": lenz.id},
            ],
        },
        predicted_accept_prob=0.72,
        predicted_regression_prob=0.18,
        predicted_failure_type="analogy_overreach",
        predicted_value_delta=0.11,
        actual_outcome=None,
        calibration_error=None,
    )
    validation_rows = _validation_rows([le_chatelier, lenz], alignment, manifest, trial)
    graph_projection = graph_projection_from_overlay(overlay_ops)
    metrics = {
        "process_model_count": 2,
        "alignment_relation_node_count": sum(
            1 for node in graph_projection["nodes"]
            if node.get("kind") == HypothesisKind.ALIGNMENT_HYPOTHESIS.value
        ),
        "bare_alignment_edge_count": sum(
            1 for edge in graph_projection["edges"]
            if edge.get("type") in {EdgeType.IS_ANALOGY_OF.value, EdgeType.IS_FORMAL_ISOMORPHISM_OF.value}
        ),
        "participates_in_edge_count": sum(
            1 for edge in graph_projection["edges"]
            if edge.get("type") == EdgeType.PARTICIPATES_IN.value
        ),
        "counterfactual_mask_count": len(trial.action["counterfactual_masks"]),
        "validation_issue_count": sum(len(row["issues"]) for row in validation_rows),
        "mapping_score": round(process_alignment_score(le_chatelier, lenz, alignment), 4),
    }
    gates = {
        "manifest_contract_complete": not manifest.validate(),
        "process_models_are_typed": all(not process.validate() for process in [le_chatelier, lenz]),
        "alignment_is_relation_node_not_bare_edge": (
            metrics["alignment_relation_node_count"] == 1
            and metrics["bare_alignment_edge_count"] == 0
            and metrics["participates_in_edge_count"] == 2
        ),
        "mapping_preserves_perturbation_response_roles": _mapping_has_roles(alignment),
        "broken_structure_boundaries_recorded": len(alignment.broken_structure) >= 2,
        "world_model_trial_is_graph_action": not trial.validate() and trial.action.get("type") == "add_alignment_hypothesis",
        "counterfactual_masks_are_explicit": metrics["counterfactual_mask_count"] >= 3,
        "no_main_graph_mutation": True,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "hypothesis_lifecycle_v2_schema_fixture",
        "reconstruction_v2_phase": "phase0_schema_freeze",
        "performance_validation": False,
        "behavior_validation": True,
        "validation_scope": (
            "Schema-level v2 validation: represent a cross-domain process-alignment hypothesis as a "
            "manifest plus graph overlay relation node, process payloads, verifier contract, and world-model trial."
        ),
        "objects": {
            "manifest": manifest.to_dict(),
            "process_models": [le_chatelier.to_dict(), lenz.to_dict()],
            "alignment_hypothesis": alignment.to_dict(),
            "world_model_trial": trial.to_dict(),
        },
        "graph_projection": graph_projection,
        "validation_rows": validation_rows,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "V2 keeps graph as the lifecycle substrate, but a relation such as Le Chatelier <-> Lenz is no "
            "longer a naked edge.  It is an alignment hypothesis node with process payloads, preserved/broken "
            "structure, verifier tests, world-model predictions, and counterfactual masks."
        ),
    }


def build_alignment_overlay_ops(
    *,
    source_process: ProcessModel,
    target_process: ProcessModel,
    alignment: AlignmentHypothesis,
    verifier: VerifierContract,
) -> list[GraphOverlayOp]:
    source_node = source_process.to_node()
    target_node = target_process.to_node()
    relation_node = alignment.to_relation_node(
        claim=(
            f"{source_process.domain} and {target_process.domain} share a bounded "
            "perturbation-opposition process schema."
        )
    )
    manifest_stub = {
        "verifier_contract": verifier.to_dict(),
        "rollback_ref": f"rollback::{alignment.id}",
    }
    return [
        GraphOverlayOp(op="add_node", node=source_node.to_dict(), rollback_ref=f"remove::{source_node.id}"),
        GraphOverlayOp(op="add_node", node=target_node.to_dict(), rollback_ref=f"remove::{target_node.id}"),
        GraphOverlayOp(op="add_relation_node", node=relation_node.to_dict(), rollback_ref=f"remove::{relation_node.id}"),
        GraphOverlayOp(
            op="add_edge",
            edge=AssumptionEdge(
                source=source_process.id,
                target=alignment.id,
                type=EdgeType.PARTICIPATES_IN,
                payload={"role": "source_process", **manifest_stub},
            ).to_dict(),
            rollback_ref=f"remove_edge::{source_process.id}->{alignment.id}",
        ),
        GraphOverlayOp(
            op="add_edge",
            edge=AssumptionEdge(
                source=alignment.id,
                target=target_process.id,
                type=EdgeType.PARTICIPATES_IN,
                payload={"role": "target_process", **manifest_stub},
            ).to_dict(),
            rollback_ref=f"remove_edge::{alignment.id}->{target_process.id}",
        ),
    ]


def graph_projection_from_overlay(overlay_ops: Iterable[GraphOverlayOp]) -> dict[str, Any]:
    nodes = []
    edges = []
    op_counts = Counter()
    for op in overlay_ops:
        op_counts[op.op] += 1
        if op.node:
            nodes.append(op.node)
        if op.edge:
            edges.append(op.edge)
    return {
        "op_counts": dict(op_counts),
        "nodes": nodes,
        "edges": edges,
        "rollback_refs": [op.rollback_ref for op in overlay_ops if op.rollback_ref],
    }


def process_alignment_score(source: ProcessModel, target: ProcessModel, alignment: AlignmentHypothesis) -> float:
    """A transparent first-pass score for typed process alignment."""

    required_roles = {"perturbation", "response", "opposition_relation"}
    role_score = len(required_roles & set(alignment.mapping)) / len(required_roles)
    source_text = _token_set(" ".join(source.invariants + [source.perturbation, source.response]))
    target_text = _token_set(" ".join(target.invariants + [target.perturbation, target.response]))
    preserved_text = _token_set(" ".join(alignment.preserved_structure))
    source_overlap = _jaccard(source_text, preserved_text)
    target_overlap = _jaccard(target_text, preserved_text)
    boundary_penalty = 0.05 if alignment.broken_structure else 0.18
    return max(0.0, min(1.0, 0.55 * role_score + 0.25 * max(source_overlap, target_overlap) + 0.25 - boundary_penalty))


def _validation_rows(
    processes: list[ProcessModel],
    alignment: AlignmentHypothesis,
    manifest: AssumptionManifestV2,
    trial: WorldModelTrial,
) -> list[dict[str, Any]]:
    rows = [
        {"object_id": manifest.id, "object_type": "AssumptionManifestV2", "issues": manifest.validate()},
        {"object_id": alignment.id, "object_type": "AlignmentHypothesis", "issues": alignment.validate()},
        {"object_id": trial.id, "object_type": "WorldModelTrial", "issues": trial.validate()},
    ]
    rows.extend(
        {"object_id": process.id, "object_type": "ProcessModel", "issues": process.validate()}
        for process in processes
    )
    return rows


def _mapping_has_roles(alignment: AlignmentHypothesis) -> bool:
    return {"perturbation", "response", "opposition_relation"}.issubset(set(alignment.mapping))


def _token_set(text: str) -> set[str]:
    return {
        token
        for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
        if len(token) > 2
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _le_chatelier_process() -> ProcessModel:
    return ProcessModel(
        id="process_le_chatelier_v1",
        domain="chemical_thermodynamic_equilibrium",
        state_variables=[
            "reactant_concentration",
            "product_concentration",
            "temperature",
            "pressure",
            "reaction_quotient",
        ],
        parameters=["equilibrium_constant", "stoichiometric_coefficients"],
        interventions=[
            "external_concentration_change",
            "external_temperature_change",
            "external_pressure_change",
        ],
        perturbation="imposed external condition change",
        response="equilibrium shift that partially counteracts the imposed change",
        dynamics={
            "rule": "local relaxation toward constrained equilibrium",
            "sign_relation": "response direction reduces the externally imposed deviation",
        },
        observation_map="observe shift in reaction quotient and equilibrium composition",
        invariants=[
            "response partially opposes imposed perturbation",
            "system moves toward constrained equilibrium",
            "domain-specific thermodynamic variables remain explicit",
        ],
        failure_cases=[
            "far from equilibrium kinetics",
            "irreversible reaction regime",
            "ambiguous reaction coordinate",
        ],
    )


def _lenz_process() -> ProcessModel:
    return ProcessModel(
        id="process_lenz_law_v1",
        domain="electromagnetic_induction",
        state_variables=[
            "magnetic_flux",
            "flux_time_derivative",
            "induced_emf",
            "induced_current",
            "induced_magnetic_field",
        ],
        parameters=["circuit_resistance", "loop_geometry", "magnetic_permeability"],
        interventions=["external_flux_change"],
        perturbation="change in magnetic flux over time",
        response="induced current creates magnetic field opposing the flux change",
        dynamics={
            "rule": "Faraday-Lenz sign relation",
            "sign_relation": "induced response has sign opposing flux change",
        },
        observation_map="observe induced emf/current and induced magnetic-field direction",
        invariants=[
            "induced response opposes magnetic flux change",
            "opposition relation is local and sign-sensitive",
            "domain-specific electromagnetic variables remain explicit",
        ],
        failure_cases=[
            "open circuit with no current path",
            "nonlinear magnetic material regime",
            "high-frequency radiative regime",
        ],
    )


def _negative_feedback_alignment(source: ProcessModel, target: ProcessModel) -> AlignmentHypothesis:
    return AlignmentHypothesis(
        id="align_le_chatelier_lenz_v1",
        source_process=source.id,
        target_process=target.id,
        mapping={
            "perturbation": "imposed external condition change -> change in magnetic flux over time",
            "response": "equilibrium shift -> induced current / induced magnetic field",
            "opposition_relation": "partial counteraction -> opposing flux sign",
            "stability_intuition": "constrained equilibrium restoration -> local negative response",
        },
        preserved_structure=[
            "typed perturbation-response-opposition schema",
            "response direction reduces or opposes the imposed change",
            "local negative feedback intuition",
        ],
        broken_structure=[
            "thermodynamic equilibrium equations are not electromagnetic induction equations",
            "free-energy relaxation is not circuit dynamics",
            "chemical composition variables are not magnetic-flux variables",
        ],
        metric_scores={
            "role_mapping_coverage": 1.0,
            "invariant_overlap": 0.72,
            "broken_structure_penalty": 0.08,
        },
        verifier_tests=[
            "typed_mapping_check",
            "invariant_preservation_check",
            "negative_control_boundary_check",
            "fresh_cross_domain_transfer_ablation",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the hypothesis lifecycle v2 schema fixture.")
    parser.add_argument("--eval-id", default="hypothesis_lifecycle_v2_schema_20260610")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_hypothesis_lifecycle_v2_payload(eval_id=args.eval_id)
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
