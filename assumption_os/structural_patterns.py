"""Bounded structural morphism layer for assumption transfer.

This module is intentionally category-inspired, not a category-theory solver.
It represents reusable ideas as typed diagrams with roles, morphisms,
composition hints, invariants, and negative controls.  A candidate problem or
proposal is matched against these diagrams to decide whether it is a plausible
structure-preserving extension of an older pattern.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .formal_mapping import finite_kernel_metrics
from .graph_memory import JsonlGraphStore, tokenize
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


STRUCTURAL_PATTERN_KIND = "structural_pattern"
STRUCTURAL_MORPHISM_KIND = "structural_morphism_candidate"


@dataclass(frozen=True)
class StructuralSignature:
    source_text: str
    terms: list[str]
    role_hits: dict[str, list[str]] = field(default_factory=dict)
    invariant_hits: dict[str, list[str]] = field(default_factory=dict)
    negation_hits: list[str] = field(default_factory=list)
    pattern_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StructuralDiagram:
    source_text: str
    objects: list[dict] = field(default_factory=list)
    morphisms: list[dict] = field(default_factory=list)
    composition_laws: list[dict] = field(default_factory=list)
    invariants: list[dict] = field(default_factory=list)
    negation_hits: list[str] = field(default_factory=list)
    pattern_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "formal_kind": "structural_diagram",
            **asdict(self),
            "object_count": len(self.objects),
            "morphism_count": len(self.morphisms),
            "composition_law_count": len(self.composition_laws),
            "invariant_count": len(self.invariants),
        }


@dataclass(frozen=True)
class StructuralMorphismScore:
    pattern_id: str
    score: float
    object_role_coverage: float
    morphism_role_coverage: float
    composition_preservation: float
    invariant_preservation: float
    negative_control_score: float
    negative_control_margin: float
    matched_terms: list[str]
    preserved_invariants: list[str]
    broken_or_uncertain_invariants: list[str]
    negative_control_hits: list[str]
    decision: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


ROLE_MARKERS = {
    "baseline_path": [
        "baseline",
        "fallback",
        "identity",
        "skip",
        "preserve",
        "old path",
        "working path",
        "verified path",
    ],
    "delta_update": [
        "delta",
        "residual",
        "correction",
        "deviation",
        "local update",
        "minimal patch",
        "lora",
        "adapter",
    ],
    "control_row": [
        "control",
        "controls",
        "placebo",
        "ablation",
        "a/b",
        "one variable",
        "single variable",
    ],
    "perturbation": [
        "perturbation",
        "disturbance",
        "shock",
        "external change",
        "imposed change",
    ],
    "opposing_response": [
        "opposes",
        "compensates",
        "cancels",
        "resists",
        "negative feedback",
        "restore",
    ],
    "stable_signal": [
        "stable signal",
        "predictable",
        "correlated",
        "invariant signal",
        "latent state",
        "world state",
    ],
    "nuisance_noise": [
        "noise",
        "nuisance",
        "uncorrelated",
        "random",
        "gaussian",
        "irrelevant detail",
        "stochastic",
    ],
    "module_boundary": [
        "module",
        "adapter",
        "component",
        "boundary",
        "pipeline",
        "replace one",
    ],
    "root_problem": [
        "root problem",
        "whole problem",
        "goal",
        "overall task",
        "parent problem",
        "compose solution",
    ],
    "subproblem": [
        "subproblem",
        "subtask",
        "decompose",
        "split",
        "factor",
        "independent part",
    ],
    "interface_contract": [
        "interface",
        "contract",
        "schema",
        "boundary condition",
        "input output",
        "io contract",
    ],
    "bottleneck_resource": [
        "bottleneck",
        "capacity",
        "throughput",
        "rate limit",
        "scarce resource",
    ],
    "flow_item": [
        "flow",
        "queue",
        "traffic",
        "throughput",
        "token budget",
    ],
    "counterexample": [
        "counterexample",
        "adversarial",
        "edge case",
        "failure case",
        "falsify",
        "breaks",
    ],
    "refined_claim": [
        "refine",
        "patch",
        "narrow",
        "weaken claim",
        "guardrail",
        "revised claim",
    ],
    "conserved_quantity": [
        "invariant quantity",
        "mass balance",
        "energy balance",
        "budget balance",
        "probability mass",
        "budget",
    ],
    "transformation": [
        "transform",
        "transition",
        "state change",
        "mapping",
        "conversion",
        "update step",
    ],
    "ordered_state": [
        "ordered",
        "ranked",
        "monotonic",
        "partial order",
        "non-decreasing",
        "dominance",
    ],
    "objective_measure": [
        "objective",
        "score",
        "loss",
        "utility",
        "progress",
    ],
}


INVARIANT_MARKERS = {
    "identity_path_preserved": ["identity", "baseline", "fallback", "preserve", "old path", "working path"],
    "learned_part_models_deviation": ["delta", "residual", "correction", "deviation", "local update"],
    "zero_delta_recovers_baseline": ["zero", "fallback", "recover", "old behavior", "rollback"],
    "single_intervention_isolated": ["one variable", "single variable", "ablation", "control", "controls"],
    "matched_control_required": ["control", "controls", "placebo", "matched control", "matched"],
    "response_opposes_perturbation": ["opposes", "compensates", "resists", "cancels", "negative feedback"],
    "constraint_explains_response": ["constraint", "conservation", "free energy", "equilibrium", "law"],
    "predictable_structure_separated": ["predictable", "stable signal", "correlated", "latent state", "world state"],
    "stochastic_nuisance_suppressed": ["noise", "nuisance", "uncorrelated", "random", "gaussian", "irrelevant detail"],
    "module_boundary_preserved": ["module", "boundary", "pipeline", "component"],
    "rollback_path_available": ["rollback", "fallback", "old path", "working path", "revert"],
    "interface_contract_preserved": ["interface", "contract", "schema", "boundary condition", "input output"],
    "subproblem_independence_preserved": ["independent", "subproblem", "subtask", "factor", "separable"],
    "composed_solution_recovers_goal": ["compose solution", "recover goal", "overall task", "root problem", "join result"],
    "bottleneck_controls_throughput": ["bottleneck", "throughput", "capacity", "rate limit", "scarce resource"],
    "capacity_constraint_explicit": ["capacity constraint", "rate limit", "scarce resource", "resource budget"],
    "counterexample_targets_claim": ["counterexample", "falsify", "breaks", "edge case", "adversarial"],
    "refinement_handles_counterexample": ["refine", "patch", "narrow", "guardrail", "revised claim"],
    "conserved_quantity_preserved": ["conserved quantity", "invariant quantity", "mass balance", "energy balance", "budget balance", "probability mass"],
    "balance_check_closes": ["balance", "accounting", "check", "closed", "sum"],
    "order_preserved": ["ordered", "monotonic", "non-decreasing", "partial order", "dominance"],
    "progress_non_decreasing": ["progress", "objective", "score", "utility", "non-decreasing", "improve"],
}


NEGATION_PATTERNS = [
    r"\bno\s+(?:baseline|fallback|identity|control|residual|rollback|invariant)\b",
    r"\bno\s+(?:roles|morphisms|invariants|transfer prediction|predictable signal)\b",
    r"\bwithout\s+(?:baseline|fallback|identity|control|residual|rollback|invariant)\b",
    r"\bnot\s+(?:preserve|controlled|opposing|predictable)\b",
]


ROLE_OBJECT_HINTS = {
    "baseline_path": "preserved baseline or identity path",
    "delta_update": "local delta or residual update",
    "control_row": "matched control or intervention row",
    "perturbation": "external perturbation or imposed change",
    "opposing_response": "induced opposing response",
    "stable_signal": "predictable stable signal",
    "nuisance_noise": "stochastic nuisance variation",
    "module_boundary": "bounded module or adapter boundary",
    "root_problem": "whole goal or parent problem",
    "subproblem": "bounded subproblem or subtask",
    "interface_contract": "interface contract that composes subsolutions",
    "bottleneck_resource": "scarce capacity or rate-limiting resource",
    "flow_item": "items or work flowing through a constrained path",
    "counterexample": "adversarial counterexample or edge failure",
    "refined_claim": "narrowed claim or patched hypothesis",
    "conserved_quantity": "quantity that must balance through a transformation",
    "transformation": "state transition or conversion step",
    "ordered_state": "ordered state under a monotone relation",
    "objective_measure": "objective, metric, or progress measure",
}


MORPHISM_ENDPOINT_HINTS = {
    "identity_path": ("input_state", "output_state"),
    "learn_delta": ("input_state", "delta_update"),
    "compose_add": ("delta_update", "output_state"),
    "change_one_factor": ("baseline_case", "intervention_case"),
    "compare_outcomes": ("intervention_case", "outcome_measure"),
    "preserve_pipeline": ("working_pipeline", "working_pipeline"),
    "swap_one_module": ("working_pipeline", "replacement_delta"),
    "rollback_if_failed": ("replacement_delta", "working_pipeline"),
    "perturb_state": ("system_state", "external_perturbation"),
    "induce_response": ("external_perturbation", "induced_response"),
    "oppose_change": ("induced_response", "system_state"),
    "suppress_noise": ("nuisance_noise", "projection_operator"),
    "recover_signal": ("stable_signal", "projection_operator"),
    "split_problem": ("root_problem", "subproblem"),
    "solve_subproblem": ("subproblem", "interface_contract"),
    "compose_solution": ("interface_contract", "root_problem"),
    "route_flow": ("flow_input", "bottleneck_resource"),
    "constrain_capacity": ("bottleneck_resource", "flow_output"),
    "relieve_bottleneck": ("bottleneck_resource", "flow_output"),
    "generate_counterexample": ("claim_under_test", "counterexample_case"),
    "falsify_claim": ("counterexample_case", "claim_under_test"),
    "patch_claim": ("claim_under_test", "refined_claim"),
    "transform_state": ("source_state", "target_state"),
    "conserve_quantity": ("conserved_quantity", "target_state"),
    "check_balance": ("target_state", "conserved_quantity"),
    "apply_monotone_step": ("ordered_state", "ordered_state"),
    "preserve_order": ("ordered_state", "objective_measure"),
    "measure_progress": ("objective_measure", "objective_measure"),
}


ROLE_MORPHISM_RULES = [
    {
        "id": "identity_path",
        "role": "baseline_path",
        "source_role": "baseline_path",
        "target_role": "baseline_path",
        "requires": ["baseline_path"],
        "terms": ["identity", "baseline", "fallback", "preserve", "recover", "old behavior"],
    },
    {
        "id": "learn_delta",
        "role": "delta_update",
        "source_role": "baseline_path",
        "target_role": "delta_update",
        "requires": ["baseline_path", "delta_update"],
        "terms": ["delta", "residual", "correction", "deviation", "local update"],
    },
    {
        "id": "compose_add",
        "role": "delta_update",
        "source_role": "delta_update",
        "target_role": "baseline_path",
        "requires": ["baseline_path", "delta_update"],
        "terms": ["add", "compose", "local patch", "zero", "recover"],
    },
    {
        "id": "change_one_factor",
        "role": "control_row",
        "source_role": "control_row",
        "target_role": "control_row",
        "requires": ["control_row"],
        "terms": ["one variable", "single variable", "one intervention", "ablation"],
    },
    {
        "id": "compare_outcomes",
        "role": "control_row",
        "source_role": "control_row",
        "target_role": "control_row",
        "requires": ["control_row"],
        "terms": ["compare", "control", "baseline", "outcome", "acceptance"],
    },
    {
        "id": "preserve_pipeline",
        "role": "baseline_path",
        "source_role": "baseline_path",
        "target_role": "baseline_path",
        "requires": ["baseline_path", "module_boundary"],
        "terms": ["preserve", "keep", "pipeline", "fallback"],
    },
    {
        "id": "swap_one_module",
        "role": "module_boundary",
        "source_role": "baseline_path",
        "target_role": "delta_update",
        "requires": ["baseline_path", "module_boundary"],
        "terms": ["replace one", "single module", "component", "boundary"],
    },
    {
        "id": "rollback_if_failed",
        "role": "baseline_path",
        "source_role": "delta_update",
        "target_role": "baseline_path",
        "requires": ["baseline_path", "module_boundary"],
        "terms": ["rollback", "revert", "fallback"],
    },
    {
        "id": "perturb_state",
        "role": "perturbation",
        "source_role": "perturbation",
        "target_role": "perturbation",
        "requires": ["perturbation"],
        "terms": ["perturb", "disturbance", "external change", "imposed change"],
    },
    {
        "id": "induce_response",
        "role": "opposing_response",
        "source_role": "perturbation",
        "target_role": "opposing_response",
        "requires": ["perturbation", "opposing_response"],
        "terms": ["induce", "response", "reaction"],
    },
    {
        "id": "oppose_change",
        "role": "opposing_response",
        "source_role": "opposing_response",
        "target_role": "perturbation",
        "requires": ["perturbation", "opposing_response"],
        "terms": ["opposes", "compensates", "resists", "cancels"],
    },
    {
        "id": "suppress_noise",
        "role": "nuisance_noise",
        "source_role": "nuisance_noise",
        "target_role": "stable_signal",
        "requires": ["stable_signal", "nuisance_noise"],
        "terms": ["suppress", "ignore", "denoise", "not reconstruct"],
    },
    {
        "id": "recover_signal",
        "role": "stable_signal",
        "source_role": "stable_signal",
        "target_role": "stable_signal",
        "requires": ["stable_signal"],
        "terms": ["recover", "predict", "latent", "stable"],
    },
    {
        "id": "split_problem",
        "role": "subproblem",
        "source_role": "root_problem",
        "target_role": "subproblem",
        "requires": ["root_problem", "subproblem"],
        "terms": ["split", "decompose", "factor", "subproblem", "subtask"],
    },
    {
        "id": "solve_subproblem",
        "role": "subproblem",
        "source_role": "subproblem",
        "target_role": "interface_contract",
        "requires": ["subproblem", "interface_contract"],
        "terms": ["solve subproblem", "independent part", "interface", "contract"],
    },
    {
        "id": "compose_solution",
        "role": "interface_contract",
        "source_role": "interface_contract",
        "target_role": "root_problem",
        "requires": ["root_problem", "subproblem", "interface_contract"],
        "terms": ["compose solution", "join result", "recover goal", "overall task"],
    },
    {
        "id": "route_flow",
        "role": "flow_item",
        "source_role": "flow_item",
        "target_role": "bottleneck_resource",
        "requires": ["flow_item", "bottleneck_resource"],
        "terms": ["route", "flow", "queue", "traffic", "throughput"],
    },
    {
        "id": "constrain_capacity",
        "role": "bottleneck_resource",
        "source_role": "bottleneck_resource",
        "target_role": "flow_item",
        "requires": ["flow_item", "bottleneck_resource"],
        "terms": ["capacity", "constraint", "rate limit", "scarce resource"],
    },
    {
        "id": "relieve_bottleneck",
        "role": "bottleneck_resource",
        "source_role": "bottleneck_resource",
        "target_role": "flow_item",
        "requires": ["flow_item", "bottleneck_resource"],
        "terms": ["relieve", "widen", "remove bottleneck", "increase throughput"],
    },
    {
        "id": "generate_counterexample",
        "role": "counterexample",
        "source_role": "refined_claim",
        "target_role": "counterexample",
        "requires": ["counterexample"],
        "terms": ["generate counterexample", "adversarial", "edge case", "failure case"],
    },
    {
        "id": "falsify_claim",
        "role": "counterexample",
        "source_role": "counterexample",
        "target_role": "refined_claim",
        "requires": ["counterexample", "refined_claim"],
        "terms": ["falsify", "breaks", "disprove", "violates"],
    },
    {
        "id": "patch_claim",
        "role": "refined_claim",
        "source_role": "counterexample",
        "target_role": "refined_claim",
        "requires": ["counterexample", "refined_claim"],
        "terms": ["patch", "refine", "narrow", "guardrail", "revised claim"],
    },
    {
        "id": "transform_state",
        "role": "transformation",
        "source_role": "transformation",
        "target_role": "conserved_quantity",
        "requires": ["transformation", "conserved_quantity"],
        "terms": ["transform", "transition", "state change", "conversion"],
    },
    {
        "id": "conserve_quantity",
        "role": "conserved_quantity",
        "source_role": "conserved_quantity",
        "target_role": "transformation",
        "requires": ["transformation", "conserved_quantity"],
        "terms": ["conserve", "conservation", "balance", "invariant quantity"],
    },
    {
        "id": "check_balance",
        "role": "conserved_quantity",
        "source_role": "transformation",
        "target_role": "conserved_quantity",
        "requires": ["transformation", "conserved_quantity"],
        "terms": ["check balance", "accounting", "closed", "sum"],
    },
    {
        "id": "apply_monotone_step",
        "role": "ordered_state",
        "source_role": "ordered_state",
        "target_role": "ordered_state",
        "requires": ["ordered_state"],
        "terms": ["monotonic", "monotone step", "non-decreasing", "ordered"],
    },
    {
        "id": "preserve_order",
        "role": "ordered_state",
        "source_role": "ordered_state",
        "target_role": "objective_measure",
        "requires": ["ordered_state", "objective_measure"],
        "terms": ["preserve order", "dominance", "partial order"],
    },
    {
        "id": "measure_progress",
        "role": "objective_measure",
        "source_role": "objective_measure",
        "target_role": "objective_measure",
        "requires": ["ordered_state", "objective_measure"],
        "terms": ["measure progress", "score", "objective", "utility", "improve"],
    },
]


def default_structural_pattern_nodes() -> list[AssumptionNode]:
    return [_pattern_node(spec) for spec in DEFAULT_STRUCTURAL_PATTERNS]


def seed_structural_patterns(store: JsonlGraphStore, *, persist: bool = True) -> list[str]:
    """Upsert the default structural patterns into an Assumption Graph store."""

    node_ids = []
    for node in default_structural_pattern_nodes():
        store.upsert_node(node)
        node_ids.append(node.id)
    if persist:
        store.flush()
    return node_ids


def load_structural_patterns(
    store: JsonlGraphStore | None = None,
    *,
    include_defaults: bool = True,
) -> list[dict]:
    patterns: dict[str, dict] = {}
    if include_defaults:
        for node in default_structural_pattern_nodes():
            pattern = _node_to_pattern(node)
            patterns[pattern["pattern_id"]] = pattern
    if store:
        for node in store.nodes.values():
            pattern = _node_to_pattern(node)
            if pattern:
                patterns[pattern["pattern_id"]] = pattern
    return sorted(patterns.values(), key=lambda row: row["pattern_id"])


def extract_structural_signature(source: str | dict) -> StructuralSignature:
    """Extract a small deterministic structural signature.

    The first implementation is deliberately deterministic so extraction can be
    audited separately from LLM generation.  LLM extraction can later fill the
    same fields, but it should be tested against this payload shape.
    """

    text = _source_text(source)
    low = text.lower()
    terms = sorted(tokenize(text).keys())
    role_hits = {
        role: hits
        for role, hits in (
            (role, _term_hits(low, markers))
            for role, markers in ROLE_MARKERS.items()
        )
        if hits
    }
    invariant_hits = {
        inv: hits
        for inv, hits in (
            (inv, _term_hits(low, markers))
            for inv, markers in INVARIANT_MARKERS.items()
        )
        if hits
    }
    negation_hits = sorted({
        match.group(0)
        for pattern in NEGATION_PATTERNS
        for match in re.finditer(pattern, low)
    })
    pattern_hints = sorted({
        pattern["pattern_id"]
        for pattern in DEFAULT_STRUCTURAL_PATTERNS
        if _term_hits(low, pattern.get("trigger_terms", []))
    })
    return StructuralSignature(
        source_text=text,
        terms=terms,
        role_hits=role_hits,
        invariant_hits=invariant_hits,
        negation_hits=negation_hits,
        pattern_hints=pattern_hints,
    )


def extract_structural_diagram(source: str | dict | StructuralSignature | StructuralDiagram) -> StructuralDiagram:
    """Extract an auditable object/morphism/composition diagram.

    This is still deliberately bounded: it is a deterministic diagram extractor
    for the structural motifs the project actually uses.  The important step is
    that downstream gates reason over typed objects, morphisms, and composition
    rows instead of only over surface overlap.
    """

    if isinstance(source, StructuralDiagram):
        return source
    signature = source if isinstance(source, StructuralSignature) else extract_structural_signature(source)
    roles = set(signature.role_hits)
    objects = [
        {
            "id": f"obj_{role}",
            "role": role,
            "label": ROLE_OBJECT_HINTS.get(role, role.replace("_", " ")),
            "matched_terms": signature.role_hits.get(role, []),
        }
        for role in sorted(roles)
    ]
    morphisms = []
    for rule in ROLE_MORPHISM_RULES:
        required = set(rule.get("requires", []))
        if required and not required <= roles:
            continue
        matched = _term_hits(signature.source_text.lower(), rule.get("terms", []))
        if not matched and not signature.role_hits.get(rule["role"]):
            continue
        morphisms.append({
            "id": rule["id"],
            "role": rule["role"],
            "source_role": rule["source_role"],
            "target_role": rule["target_role"],
            "source": f"obj_{rule['source_role']}",
            "target": f"obj_{rule['target_role']}",
            "matched_terms": matched or signature.role_hits.get(rule["role"], []),
        })
    invariants = [
        {"id": inv_id, "matched_terms": hits}
        for inv_id, hits in sorted(signature.invariant_hits.items())
    ]
    composition_laws = _infer_composition_laws(roles, {m["id"] for m in morphisms}, signature)
    return StructuralDiagram(
        source_text=signature.source_text,
        objects=objects,
        morphisms=morphisms,
        composition_laws=composition_laws,
        invariants=invariants,
        negation_hits=signature.negation_hits,
        pattern_hints=signature.pattern_hints,
    )


def score_pattern_match(query_diagram: str | dict | StructuralSignature, pattern: dict) -> StructuralMorphismScore:
    signature = query_diagram if isinstance(query_diagram, StructuralSignature) else extract_structural_signature(query_diagram)
    text = signature.source_text.lower()
    objects = pattern.get("objects", [])
    morphisms = pattern.get("morphisms", [])
    invariants = pattern.get("invariants", [])
    composition_laws = pattern.get("composition_laws", [])

    object_hits = _role_rows_hit(text, objects)
    morphism_hits = _role_rows_hit(text, morphisms)
    invariant_hits = _invariants_hit(text, invariants)
    composition_hits = _composition_hits(text, composition_laws)
    negative_hits = _negative_hits(text, signature, pattern)
    trigger_hits = _term_hits(text, pattern.get("trigger_terms", []))
    realization_hits = _term_hits(text, pattern.get("good_realizations", []))

    object_cov = _ratio(len(object_hits), len(objects))
    morphism_cov = _ratio(len(morphism_hits), len(morphisms))
    invariant_cov = _ratio(len(invariant_hits), len(invariants))
    composition_cov = _ratio(composition_hits, len(composition_laws)) if composition_laws else 0.5
    negative_score = min(1.0, len(negative_hits) / max(1, min(3, len(pattern.get("negative_controls", [])))))
    severe_negative = _has_severe_negative(negative_hits)

    positive_signal = (
        0.25 * object_cov
        + 0.25 * morphism_cov
        + 0.25 * invariant_cov
        + 0.15 * composition_cov
        + 0.07 * min(1.0, len(trigger_hits) / 3)
        + 0.03 * min(1.0, len(realization_hits) / 2)
    )
    score = max(0.0, positive_signal - 0.65 * negative_score - 0.15 * len(signature.negation_hits))
    margin = positive_signal - negative_score
    preserved = [row["id"] for row in invariant_hits]
    broken = [
        row["id"]
        for row in invariants
        if row.get("id") not in set(preserved)
    ]
    if signature.negation_hits:
        broken.extend(f"explicit_negation::{hit}" for hit in signature.negation_hits)
    decision, reason = _gate_decision(
        object_cov=object_cov,
        morphism_cov=morphism_cov,
        invariant_cov=invariant_cov,
        composition_cov=composition_cov,
        negative_score=negative_score,
        margin=margin,
        severe_negative=severe_negative,
        transfer_predictions=pattern.get("transfer_predictions", []),
    )
    prediction_check = assess_transfer_prediction_testability(pattern.get("transfer_predictions", []))
    if decision not in {"block_negative_control"} and not prediction_check.get("pass"):
        decision = "repair_missing_testable_transfer_prediction"
        reason = prediction_check.get("reason", "Transfer prediction is not testable enough for promotion.")
    return StructuralMorphismScore(
        pattern_id=pattern["pattern_id"],
        score=round(score, 4),
        object_role_coverage=round(object_cov, 4),
        morphism_role_coverage=round(morphism_cov, 4),
        composition_preservation=round(composition_cov, 4),
        invariant_preservation=round(invariant_cov, 4),
        negative_control_score=round(negative_score, 4),
        negative_control_margin=round(margin, 4),
        matched_terms=sorted(set(
            trigger_hits
            + realization_hits
            + [hit for row in object_hits + morphism_hits + invariant_hits for hit in row.get("matched_terms", [])]
        )),
        preserved_invariants=preserved,
        broken_or_uncertain_invariants=sorted(set(broken)),
        negative_control_hits=sorted(set(negative_hits)),
        decision=decision,
        reason=reason,
    )


def check_structural_functor(
    query_diagram: str | dict | StructuralSignature | StructuralDiagram,
    pattern: dict,
) -> dict:
    """Check whether a target diagram preserves source-pattern structure.

    The check is intentionally finite and inspectable: every source object and
    morphism gets an explicit mapped/unmapped row, and every composition law is
    checked against target-side composition evidence.
    """

    diagram = _as_structural_diagram(query_diagram)
    text = diagram.source_text.lower()
    object_roles = {row.get("role") for row in diagram.objects}
    target_morphisms_by_id = {row.get("id"): row for row in diagram.morphisms}
    target_roles = {row.get("role") for row in diagram.morphisms}
    pattern = _pattern_with_endpoint_hints(pattern)
    pattern_objects = {row.get("id"): row for row in pattern.get("objects", [])}

    object_checks = []
    object_map = {}
    for source_obj in pattern.get("objects", []):
        role = source_obj.get("role")
        hits = _term_hits(text, source_obj.get("terms", []))
        mapped = bool(role in object_roles or hits)
        target_id = f"obj_{role}" if mapped else None
        if mapped:
            object_map[source_obj["id"]] = target_id
        object_checks.append({
            "source_object": source_obj["id"],
            "source_role": role,
            "target_object": target_id,
            "mapped": mapped,
            "matched_terms": hits,
        })

    morphism_checks = []
    morphism_map = {}
    for source_morphism in pattern.get("morphisms", []):
        src, dst = _morphism_endpoints(source_morphism)
        src_role = pattern_objects.get(src, {}).get("role")
        dst_role = pattern_objects.get(dst, {}).get("role")
        endpoint_preserved = (
            (not src_role or src_role in object_roles or src in object_map)
            and (not dst_role or dst_role in object_roles or dst in object_map)
        )
        direct = target_morphisms_by_id.get(source_morphism.get("id"))
        role_match = source_morphism.get("role") in target_roles
        term_hits = _term_hits(text, source_morphism.get("terms", []))
        mapped = bool(endpoint_preserved and (direct or role_match or term_hits))
        target_id = direct.get("id") if direct else (f"morphism_{source_morphism.get('role')}" if mapped else None)
        if mapped:
            morphism_map[source_morphism["id"]] = target_id
        morphism_checks.append({
            "source_morphism": source_morphism["id"],
            "source": src,
            "target": dst,
            "source_role": src_role,
            "target_role": dst_role,
            "target_morphism": target_id,
            "endpoint_preserved": endpoint_preserved,
            "mapped": mapped,
            "matched_terms": term_hits,
        })

    invariant_checks = []
    target_invariants = {row.get("id") for row in diagram.invariants}
    for invariant in pattern.get("invariants", []):
        hits = _term_hits(text, invariant.get("terms", []))
        preserved = invariant.get("id") in target_invariants or bool(hits)
        invariant_checks.append({
            "source_invariant": invariant.get("id"),
            "preserved": preserved,
            "matched_terms": hits,
        })

    composition_checks = []
    for idx, law in enumerate(pattern.get("composition_laws", [])):
        text_hit = _composition_hits(text, [str(law)]) > 0
        law_morphisms = _morphisms_for_law(str(law), pattern.get("morphisms", []))
        mapped_morphisms = [mid for mid in law_morphisms if mid in morphism_map]
        target_law_hit = _target_composition_law_hit(diagram, str(law), mapped_morphisms)
        preserved = bool(text_hit or target_law_hit or (law_morphisms and set(law_morphisms) <= set(mapped_morphisms)))
        composition_checks.append({
            "source_law": str(law),
            "law_index": idx,
            "source_morphisms": law_morphisms,
            "mapped_morphisms": mapped_morphisms,
            "text_hit": text_hit,
            "target_law_hit": target_law_hit,
            "preserved": preserved,
        })

    object_rate = _ratio(sum(1 for row in object_checks if row["mapped"]), len(object_checks))
    morphism_rate = _ratio(sum(1 for row in morphism_checks if row["mapped"]), len(morphism_checks))
    invariant_rate = _ratio(sum(1 for row in invariant_checks if row["preserved"]), len(invariant_checks))
    composition_rate = _ratio(sum(1 for row in composition_checks if row["preserved"]), len(composition_checks))
    negative_hits = _negative_hits(text, extract_structural_signature(diagram.source_text), pattern)
    severe_negative = _has_severe_negative(negative_hits)
    passed = (
        object_rate >= 0.67
        and morphism_rate >= 0.55
        and invariant_rate >= 0.5
        and composition_rate >= 0.5
        and not severe_negative
    )
    return {
        "formal_kind": "finite_structural_functor_check",
        "source_pattern_id": pattern.get("pattern_id"),
        "object_map": object_map,
        "morphism_map": morphism_map,
        "object_map_rate": round(object_rate, 4),
        "morphism_map_rate": round(morphism_rate, 4),
        "invariant_map_rate": round(invariant_rate, 4),
        "composition_preservation_rate": round(composition_rate, 4),
        "negative_control_hits": negative_hits,
        "severe_negative_control": severe_negative,
        "pass": passed,
        "object_checks": object_checks,
        "morphism_checks": morphism_checks,
        "invariant_checks": invariant_checks,
        "composition_checks": composition_checks,
    }


def propose_structural_morphism(query_diagram: str | dict | StructuralSignature | StructuralDiagram, pattern: dict) -> dict:
    signature = query_diagram if isinstance(query_diagram, StructuralSignature) else extract_structural_signature(query_diagram)
    diagram = extract_structural_diagram(signature)
    pattern = _pattern_with_endpoint_hints(pattern)
    score = score_pattern_match(signature, pattern)
    text = signature.source_text.lower()
    object_map = {
        row["id"]: _term_hits(text, row.get("terms", [])) or signature.role_hits.get(row.get("role", ""), [])
        for row in pattern.get("objects", [])
    }
    morphism_map = {
        row["id"]: _term_hits(text, row.get("terms", [])) or signature.role_hits.get(row.get("role", ""), [])
        for row in pattern.get("morphisms", [])
    }
    return {
        "formal_kind": STRUCTURAL_MORPHISM_KIND,
        "source_pattern_id": pattern["pattern_id"],
        "source_pattern_name": pattern.get("name", pattern["pattern_id"]),
        "object_map": {k: v for k, v in object_map.items() if v},
        "morphism_map": {k: v for k, v in morphism_map.items() if v},
        "source_diagram": _pattern_diagram(pattern),
        "target_diagram": diagram.to_dict(),
        "functor_check": check_structural_functor(diagram, pattern),
        "preserved_invariants": score.preserved_invariants,
        "broken_or_uncertain_invariants": score.broken_or_uncertain_invariants,
        "negative_control_hits": score.negative_control_hits,
        "transfer_predictions": pattern.get("transfer_predictions", []),
        "transfer_prediction_check": assess_transfer_prediction_testability(pattern.get("transfer_predictions", [])),
        "score": score.to_dict(),
        "status": "candidate" if score.decision != "allow" else "gate_passed_shadow",
    }


def score_structural_morphism(candidate: dict) -> dict:
    score = candidate.get("score") if isinstance(candidate, dict) else {}
    if isinstance(score, StructuralMorphismScore):
        score = score.to_dict()
    if not score:
        return {
            "decision": "repair_under_specified",
            "blocks_policy_update": True,
            "reason": "Structural morphism candidate has no score payload.",
        }
    decision = score.get("decision", "repair_under_specified")
    functor = candidate.get("functor_check") if isinstance(candidate, dict) else {}
    prediction_check = (
        candidate.get("transfer_prediction_check")
        if isinstance(candidate, dict)
        else {}
    )
    if isinstance(prediction_check, dict) and prediction_check and not prediction_check.get("pass"):
        decision = "repair_missing_testable_transfer_prediction"
    if (
        isinstance(functor, dict)
        and functor
        and not functor.get("pass")
        and decision not in {"block_negative_control", "repair_missing_transfer_prediction"}
    ):
        decision = "repair_functor_not_preserved"
    return {
        "decision": decision,
        "blocks_policy_update": decision in {
            "block_negative_control",
            "repair_under_specified",
            "repair_missing_transfer_prediction",
            "repair_missing_testable_transfer_prediction",
            "repair_functor_not_preserved",
        },
        "reason": (
            "Finite functor check did not preserve object, morphism, invariant, or composition structure."
            if decision == "repair_functor_not_preserved"
            else prediction_check.get("reason")
            if decision == "repair_missing_testable_transfer_prediction" and isinstance(prediction_check, dict)
            else score.get("reason")
        ),
        "score": score,
        "functor_check": functor if isinstance(functor, dict) else {},
        "transfer_prediction_check": prediction_check if isinstance(prediction_check, dict) else {},
    }


def build_structural_morphism_gate_payload(
    *,
    proposal_payload: dict,
    eval_id: str | None = None,
) -> dict:
    """Gate structural morphism candidates before promotion-sensitive use."""

    gates = [_proposal_structural_gate(proposal) for proposal in proposal_payload.get("proposals", [])]
    return {
        "eval_id": eval_id,
        "source_proposal_eval_id": proposal_payload.get("eval_id"),
        "gate_count": len(gates),
        "decision_counts": dict(Counter(g["decision"] for g in gates)),
        "blocked_proposal_ids": sorted(g["proposal_id"] for g in gates if g.get("blocks_policy_update")),
        "gates": gates,
    }


def build_structural_transfer_proposal_payload(
    store: JsonlGraphStore | None,
    *,
    problem: str,
    eval_id: str | None = None,
    parent_node_id: str | None = None,
    top_n: int = 2,
    min_score: float = 0.22,
) -> dict:
    """Create proposal payload rows from structural morphism retrieval."""

    applications = search_structural_patterns(store, problem, top_n=top_n, min_score=min_score)
    proposals = []
    for app in applications:
        pattern_node_id = app.get("node_id") or f"struct_{app['pattern_id']}"
        parent_id = parent_node_id or pattern_node_id
        candidate_id = stable_id("struct_cand", eval_id, problem, app["pattern_id"])
        proposal_id = stable_id("prop", eval_id, problem, app["pattern_id"], STRUCTURAL_MORPHISM_KIND)
        claim = (
            f"Structural transfer from {app['pattern_name']} should guide this problem while preserving "
            f"{', '.join(app.get('preserved_invariants', [])[:3]) or 'its finite diagram invariants'}."
        )
        candidate = AssumptionNode(
            id=candidate_id,
            type=AssumptionType.ALIGNMENT,
            kind=HypothesisKind.FORMAL_MAPPING,
            claim=claim,
            formal_form=app["candidate"],
            context_conditions=["structural_transfer_hypothesis", problem],
            predicted_effects=app.get("transfer_predictions", []),
            risk_predictions=[
                "may over-transfer if target task lacks mapped objects, morphisms, or composition laws",
            ],
            verifiers=["structural_morphism_gate", "structural_context_effect_probe", "candidate_acceptance_gate"],
            confidence=min(0.82, max(0.35, 0.35 + 0.45 * app.get("score", 0.0))),
            metaproductivity=0.12,
            status="candidate",
            tags=["structural_transfer", "structural_morphism", app["pattern_id"]],
            source_refs=["reconstruction/md/category_structural_morphism_layer_plan_20260602.md"],
            payload={"structural_application": app},
        )
        manifest = TrialManifest(
            problem_id=f"structural_transfer::{proposal_id}",
            action_type="structural_transfer_hypothesis",
            component="structural_patterns",
            assumption=claim,
            why_selected=f"Top structural pattern match {app['pattern_id']} with score={app.get('score')}.",
            expected_effect="The transferred pattern should improve target reasoning only when the functor check and context-effect probe pass.",
            assumption_ids=[parent_id, candidate_id],
            verifier="structural_morphism_gate",
            verification_plan="Run finite functor check, then controlled structural context-effect validation before graph writeback.",
            rollback_condition="Reject or keep shadow-only if functor, negative controls, or behavior validation fail.",
            status=TrialStatus.PENDING,
            metadata={"eval_id": eval_id, "proposal_id": proposal_id, "source_pattern_id": app["pattern_id"]},
            trial_id=stable_id("trial", eval_id, proposal_id, "structural_transfer"),
        )
        proposals.append({
            "proposal_id": proposal_id,
            "proposal_type": "structural_transfer_hypothesis",
            "parent_node_id": parent_id,
            "candidate_node": candidate.to_dict(),
            "edges": [
                AssumptionEdge(
                    source=candidate_id,
                    target=pattern_node_id,
                    type=EdgeType.IS_ANALOGY_OF,
                    weight=0.7,
                    payload={
                        "source": "structural_transfer_proposal",
                        "source_pattern_id": app["pattern_id"],
                        "structural_score": app.get("score"),
                    },
                ).to_dict(),
                AssumptionEdge(
                    source=candidate_id,
                    target=parent_id,
                    type=EdgeType.DERIVED_FROM,
                    weight=0.5,
                    payload={"source": "structural_transfer_proposal"},
                ).to_dict(),
            ],
            "manifest": manifest.to_dict(),
            "priority": app.get("score", 0.0),
            "rationale": f"Structural morphism candidate from {app['pattern_id']}: {app.get('reason')}",
        })
    return {
        "eval_id": eval_id,
        "proposal_source": "structural_morphism_retrieval",
        "problem": problem,
        "proposal_count": len(proposals),
        "proposals": proposals,
        "applications": applications,
    }


def apply_accepted_structural_morphisms(
    store: JsonlGraphStore,
    proposal_payload: dict,
    structural_gate_payload: dict,
    acceptance_payload: dict | None = None,
    *,
    require_acceptance: bool = True,
    persist: bool = True,
) -> list[str]:
    """Write accepted structural morphisms and lineage edges into the graph.

    This deliberately does not bypass acceptance.  By default a candidate must
    pass both the structural gate and the fresh acceptance gate before it can be
    added as a graph node.
    """

    seed_structural_patterns(store, persist=False)
    gate_by_id = {row.get("proposal_id"): row for row in structural_gate_payload.get("gates", [])}
    accepted_ids = set((acceptance_payload or {}).get("accepted_proposal_ids", []))
    acceptance_by_id = {row.get("proposal_id"): row for row in (acceptance_payload or {}).get("summaries", [])}
    applied: list[str] = []
    for proposal in proposal_payload.get("proposals", []):
        proposal_id = proposal.get("proposal_id", "")
        gate = gate_by_id.get(proposal_id, {})
        if gate.get("decision") != "allow" or gate.get("blocks_policy_update"):
            continue
        if require_acceptance and proposal_id not in accepted_ids:
            continue
        node = _structural_candidate_node(proposal)
        if not node:
            continue
        node.status = "active"
        node.payload.setdefault("structural_morphism_gate", gate)
        if acceptance_payload:
            node.payload.setdefault("acceptance_summary", acceptance_by_id.get(proposal_id, {}))
        store.upsert_node(node)
        score = gate.get("score", {})
        evidence = EvidenceRecord(
            node_id=node.id,
            source="structural_morphism_gate",
            outcome="accepted",
            metric="structural_morphism_score",
            value=score.get("score"),
            details={
                "proposal_id": proposal_id,
                "gate": gate,
                "acceptance": acceptance_by_id.get(proposal_id, {}),
            },
            evidence_id=stable_id("ev", proposal_id, node.id, "structural_morphism_gate"),
        )
        store.add_evidence(evidence)
        pattern_id = gate.get("source_pattern_id") or (node.formal_form or {}).get("source_pattern_id")
        pattern_node_id = f"struct_{pattern_id}" if pattern_id else proposal.get("parent_node_id")
        edge_type = (
            EdgeType.IS_FORMAL_ISOMORPHISM_OF
            if (gate.get("functor_check") or {}).get("pass")
            else EdgeType.IS_ANALOGY_OF
        )
        if pattern_node_id:
            store.add_edge(AssumptionEdge(
                source=node.id,
                target=pattern_node_id,
                type=edge_type,
                weight=max(0.5, min(1.0, float(score.get("score", 0.5) or 0.5))),
                evidence=evidence.evidence_id,
                payload={
                    "source": "accepted_structural_morphism",
                    "source_pattern_id": pattern_id,
                    "functor_check": gate.get("functor_check", {}),
                },
            ))
        parent_id = proposal.get("parent_node_id")
        if parent_id and parent_id != pattern_node_id:
            store.add_edge(AssumptionEdge(
                source=node.id,
                target=parent_id,
                type=EdgeType.DERIVED_FROM,
                weight=0.65,
                evidence=evidence.evidence_id,
                payload={"source": "accepted_structural_morphism"},
            ))
        if proposal.get("manifest"):
            manifest = TrialManifest.from_dict(proposal["manifest"])
            manifest.observe(
                "Accepted structural morphism was written to the graph lineage.",
                status=TrialStatus.ACCEPTED,
            )
            manifest.metadata["structural_gate"] = gate
            manifest.metadata["acceptance_summary"] = acceptance_by_id.get(proposal_id, {})
            store.append_trial(manifest)
        applied.append(node.id)
    if applied and persist:
        store.flush()
    return applied


def build_structural_lineage_payload(
    store: JsonlGraphStore,
    *,
    eval_id: str | None = None,
) -> dict:
    structural_nodes = [
        node
        for node in store.nodes.values()
        if isinstance(node.formal_form, dict)
        and node.formal_form.get("formal_kind") in {STRUCTURAL_PATTERN_KIND, STRUCTURAL_MORPHISM_KIND}
    ]
    lineage_edges = [
        edge
        for edge in store.edges
        if _edge_type_value(edge) in {
            EdgeType.IS_FORMAL_ISOMORPHISM_OF.value,
            EdgeType.IS_ANALOGY_OF.value,
            EdgeType.DERIVED_FROM.value,
        }
        and (
            edge.source in {node.id for node in structural_nodes}
            or edge.target in {node.id for node in structural_nodes}
        )
    ]
    by_pattern = Counter()
    for node in structural_nodes:
        formal = node.formal_form or {}
        if formal.get("formal_kind") == STRUCTURAL_MORPHISM_KIND:
            by_pattern[formal.get("source_pattern_id", "unknown")] += 1
    return {
        "eval_id": eval_id,
        "eval_kind": "structural_morphism_lineage",
        "structural_node_count": len(structural_nodes),
        "structural_morphism_count": sum(
            1 for node in structural_nodes if (node.formal_form or {}).get("formal_kind") == STRUCTURAL_MORPHISM_KIND
        ),
        "lineage_edge_count": len(lineage_edges),
        "morphism_count_by_source_pattern": dict(by_pattern),
        "pass": bool(structural_nodes) and bool(lineage_edges),
        "nodes": [
            {
                "id": node.id,
                "claim": node.claim,
                "formal_kind": (node.formal_form or {}).get("formal_kind"),
                "source_pattern_id": (node.formal_form or {}).get("source_pattern_id"),
                "status": node.status,
            }
            for node in sorted(structural_nodes, key=lambda n: n.id)
        ],
        "edges": [edge.to_dict() for edge in lineage_edges],
    }


def search_structural_patterns(
    store: JsonlGraphStore | None,
    query: str | dict,
    *,
    top_n: int = 3,
    min_score: float = 0.22,
    include_defaults: bool = True,
) -> list[dict]:
    signature = extract_structural_signature(query)
    rows = []
    for pattern in load_structural_patterns(store, include_defaults=include_defaults):
        score = score_pattern_match(signature, pattern)
        if score.score < min_score:
            continue
        candidate = propose_structural_morphism(signature, pattern)
        rows.append({
            "pattern_id": pattern["pattern_id"],
            "pattern_name": pattern.get("name", pattern["pattern_id"]),
            "node_id": pattern.get("node_id"),
            "score": score.score,
            "decision": score.decision,
            "reason": score.reason,
            "matched_terms": score.matched_terms,
            "preserved_invariants": score.preserved_invariants,
            "broken_or_uncertain_invariants": score.broken_or_uncertain_invariants,
            "negative_control_hits": score.negative_control_hits,
            "transfer_predictions": pattern.get("transfer_predictions", []),
            "candidate": candidate,
            "metrics": score.to_dict(),
        })
    return sorted(rows, key=lambda row: (-row["score"], row["pattern_id"]))[:top_n]


def _proposal_structural_gate(proposal: dict) -> dict:
    proposal_id = proposal.get("proposal_id", "")
    candidate = proposal.get("candidate_node") or {}
    formal = candidate.get("formal_form") or {}
    if not isinstance(formal, dict) or formal.get("formal_kind") != STRUCTURAL_MORPHISM_KIND:
        return {
            "proposal_id": proposal_id,
            "decision": "not_applicable",
            "blocks_policy_update": False,
            "reason": "Candidate is not a structural morphism proposal.",
        }
    gate = score_structural_morphism(formal)
    decision = gate["decision"]
    blocks = bool(gate["blocks_policy_update"])
    return {
        "proposal_id": proposal_id,
        "candidate_node_id": candidate.get("id"),
        "source_pattern_id": formal.get("source_pattern_id"),
        "decision": decision,
        "blocks_policy_update": blocks,
        "reason": gate.get("reason"),
        "score": gate.get("score", {}),
        "functor_check": gate.get("functor_check", {}),
        "transfer_prediction_check": gate.get("transfer_prediction_check", {}),
        "preserved_invariants": formal.get("preserved_invariants", []),
        "broken_or_uncertain_invariants": formal.get("broken_or_uncertain_invariants", []),
        "negative_control_hits": formal.get("negative_control_hits", []),
        "transfer_predictions": formal.get("transfer_predictions", []),
    }


def format_structural_morphism_applications(applications: list[dict], *, max_items: int = 2) -> str:
    if not applications:
        return ""
    lines = [
        "## Structural Morphism Reasoning",
        "Shadow-mode structural hints. Use only when the current problem preserves the listed invariants.",
    ]
    for app in applications[:max_items]:
        if not app.get("pattern_id"):
            continue
        lines.append(
            f"\n- {app.get('pattern_name', app['pattern_id'])} "
            f"({app['pattern_id']}, score={float(app.get('score', 0.0)):.2f}, gate={app.get('decision', 'unknown')})"
        )
        if app.get("matched_terms"):
            lines.append("  Matched terms: " + ", ".join(app["matched_terms"][:8]))
        if app.get("preserved_invariants"):
            lines.append("  Preserved invariants: " + "; ".join(app["preserved_invariants"][:5]))
        if app.get("broken_or_uncertain_invariants"):
            lines.append("  Broken/uncertain: " + "; ".join(app["broken_or_uncertain_invariants"][:4]))
        if app.get("negative_control_hits"):
            lines.append("  Negative-control hits: " + "; ".join(app["negative_control_hits"][:4]))
        predictions = app.get("transfer_predictions") or []
        if predictions:
            lines.append("  Transfer prediction: " + str(predictions[0]))
    return "\n".join(lines).strip()


def build_structural_pattern_payload(
    store: JsonlGraphStore | None = None,
    *,
    include_defaults: bool = True,
    eval_id: str | None = None,
) -> dict:
    patterns = load_structural_patterns(store, include_defaults=include_defaults)
    return {
        "eval_id": eval_id,
        "pattern_count": len(patterns),
        "pattern_ids": [p["pattern_id"] for p in patterns],
        "patterns": patterns,
    }


def build_structural_extraction_audit_payload(*, eval_id: str | None = None) -> dict:
    rows = []
    role_tp = role_fp = role_fn = 0
    inv_tp = inv_fp = inv_fn = 0
    broken_hits = 0
    for case in _default_extraction_audit_cases():
        sig = extract_structural_signature(case["text"])
        diagram = extract_structural_diagram(sig)
        role_pred = set(sig.role_hits)
        role_expected = set(case.get("expected_roles", []))
        inv_pred = set(sig.invariant_hits)
        inv_expected = set(case.get("expected_invariants", []))
        role_tp += len(role_pred & role_expected)
        role_fp += len(role_pred - role_expected)
        role_fn += len(role_expected - role_pred)
        inv_tp += len(inv_pred & inv_expected)
        inv_fp += len(inv_pred - inv_expected)
        inv_fn += len(inv_expected - inv_pred)
        broken_ok = bool(sig.negation_hits) == bool(case.get("expected_broken_invariant"))
        broken_hits += int(broken_ok)
        rows.append({
            "id": case["id"],
            "text": case["text"],
            "expected_roles": sorted(role_expected),
            "predicted_roles": sorted(role_pred),
            "expected_invariants": sorted(inv_expected),
            "predicted_invariants": sorted(inv_pred),
            "expected_broken_invariant": bool(case.get("expected_broken_invariant")),
            "predicted_negation_hits": sig.negation_hits,
            "predicted_diagram": diagram.to_dict(),
            "passed": role_expected <= role_pred and inv_expected <= inv_pred and broken_ok,
        })
    role_precision = _precision(role_tp, role_fp)
    role_recall = _recall(role_tp, role_fn)
    inv_precision = _precision(inv_tp, inv_fp)
    inv_recall = _recall(inv_tp, inv_fn)
    broken_accuracy = round(broken_hits / len(rows), 4) if rows else 0.0
    return {
        "eval_id": eval_id,
        "eval_kind": "structural_diagram_extraction_audit",
        "case_count": len(rows),
        "object_role_precision": role_precision,
        "object_role_recall": role_recall,
        "morphism_role_precision": role_precision,
        "morphism_role_recall": role_recall,
        "invariant_precision": inv_precision,
        "invariant_recall": inv_recall,
        "broken_invariant_detection": broken_accuracy,
        "pass": (
            len(rows) >= 6
            and role_precision >= 0.78
            and role_recall >= 0.78
            and inv_precision >= 0.72
            and inv_recall >= 0.72
            and broken_accuracy >= 0.8
        ),
        "rows": rows,
    }


def build_structural_pair_eval_payload(
    store: JsonlGraphStore | None = None,
    *,
    eval_id: str | None = None,
) -> dict:
    positive_rows = []
    positive_hits = 0
    for case in _default_positive_pair_cases():
        apps = search_structural_patterns(store, case["query"], top_n=3)
        top = apps[0] if apps else {}
        passed = top.get("pattern_id") == case["expected"]
        positive_hits += int(passed)
        positive_rows.append({
            **case,
            "top_pattern_id": top.get("pattern_id"),
            "top_score": top.get("score", 0.0),
            "passed": passed,
            "applications": apps,
        })

    negative_rows = []
    negative_rejections = 0
    for case in _default_negative_pair_cases():
        apps = search_structural_patterns(store, case["query"], top_n=3)
        top = apps[0] if apps else {}
        rejected = (not apps) or top.get("score", 0.0) < 0.22 or top.get("decision") == "block_negative_control"
        negative_rejections += int(rejected)
        negative_rows.append({
            **case,
            "top_pattern_id": top.get("pattern_id"),
            "top_score": top.get("score", 0.0),
            "top_decision": top.get("decision"),
            "rejected": rejected,
            "applications": apps,
        })
    pos_rate = round(positive_hits / len(positive_rows), 4) if positive_rows else 0.0
    neg_rate = round(negative_rejections / len(negative_rows), 4) if negative_rows else 0.0
    return {
        "eval_id": eval_id,
        "eval_kind": "structural_pair_suite",
        "positive_count": len(positive_rows),
        "negative_count": len(negative_rows),
        "positive_top1_rate": pos_rate,
        "negative_rejection_rate": neg_rate,
        "pass": len(positive_rows) >= 5 and len(negative_rows) >= 3 and pos_rate >= 0.8 and neg_rate >= 0.8,
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
    }


def build_nonlexical_structural_retrieval_probe_payload(
    store: JsonlGraphStore | None = None,
    *,
    eval_id: str | None = None,
) -> dict:
    rows = []
    hits = 0
    for case in _default_nonlexical_queries():
        apps = search_structural_patterns(store, case["query"], top_n=3)
        top = apps[0] if apps else {}
        passed = top.get("pattern_id") == case["expected"]
        hits += int(passed)
        rows.append({
            **case,
            "top_pattern_id": top.get("pattern_id"),
            "top_score": top.get("score", 0.0),
            "passed": passed,
            "applications": apps,
        })
    hit_rate = round(hits / len(rows), 4) if rows else 0.0
    return {
        "eval_id": eval_id,
        "eval_kind": "nonlexical_structural_retrieval_probe",
        "query_count": len(rows),
        "top1_hit_rate": hit_rate,
        "pass": len(rows) >= 5 and hit_rate >= 0.8,
        "rows": rows,
    }


def build_structural_behavior_probe_payload(
    store: JsonlGraphStore | None = None,
    *,
    eval_id: str | None = None,
) -> dict:
    rows = []
    wins = 0
    baseline_scores = []
    guided_scores = []
    for case in _default_behavior_tasks():
        apps = search_structural_patterns(store, case["query"], top_n=2)
        top = apps[0] if apps else {}
        pattern = _pattern_by_id(top.get("pattern_id"), store)
        baseline_answer = _generic_structural_baseline(case)
        guided_answer = _guided_structural_answer(top, pattern)
        baseline_quality = _structural_answer_quality(baseline_answer, case, pattern)
        guided_quality = _structural_answer_quality(guided_answer, case, pattern)
        baseline_scores.append(baseline_quality["score"])
        guided_scores.append(guided_quality["score"])
        win = guided_quality["score"] > baseline_quality["score"]
        wins += int(win)
        rows.append({
            **case,
            "top_pattern_id": top.get("pattern_id"),
            "top_score": top.get("score", 0.0),
            "baseline_score": baseline_quality["score"],
            "guided_score": guided_quality["score"],
            "delta": round(guided_quality["score"] - baseline_quality["score"], 4),
            "guided_wins": win,
            "baseline_quality": baseline_quality,
            "guided_quality": guided_quality,
        })
    count = len(rows)
    baseline_mean = round(sum(baseline_scores) / count, 4) if count else 0.0
    guided_mean = round(sum(guided_scores) / count, 4) if count else 0.0
    win_rate = round(wins / count, 4) if count else 0.0
    return {
        "eval_id": eval_id,
        "eval_kind": "structural_behavior_probe",
        "task_count": count,
        "baseline_mean_score": baseline_mean,
        "guided_mean_score": guided_mean,
        "mean_delta": round(guided_mean - baseline_mean, 4),
        "guided_win_rate": win_rate,
        "pass": count >= 4 and guided_mean >= 0.72 and win_rate >= 0.8 and guided_mean > baseline_mean + 0.25,
        "rows": rows,
    }


def build_structural_functor_eval_payload(
    store: JsonlGraphStore | None = None,
    *,
    eval_id: str | None = None,
) -> dict:
    positive_rows = []
    positive_pass = 0
    for case in _default_positive_pair_cases():
        apps = search_structural_patterns(store, case["query"], top_n=1, min_score=0.0)
        top = apps[0] if apps else {}
        functor = (top.get("candidate") or {}).get("functor_check", {})
        passed = top.get("pattern_id") == case["expected"] and bool(functor.get("pass"))
        positive_pass += int(passed)
        positive_rows.append({
            **case,
            "top_pattern_id": top.get("pattern_id"),
            "functor_pass": bool(functor.get("pass")),
            "functor_check": functor,
            "passed": passed,
        })

    negative_rows = []
    negative_reject = 0
    for case in _default_negative_pair_cases():
        apps = search_structural_patterns(store, case["query"], top_n=1, min_score=0.0)
        top = apps[0] if apps else {}
        functor = (top.get("candidate") or {}).get("functor_check", {})
        rejected = (not apps) or not functor.get("pass") or top.get("decision") == "block_negative_control"
        negative_reject += int(rejected)
        negative_rows.append({
            **case,
            "top_pattern_id": top.get("pattern_id"),
            "functor_pass": bool(functor.get("pass")),
            "top_decision": top.get("decision"),
            "functor_check": functor,
            "rejected": rejected,
        })

    positive_rate = round(_ratio(positive_pass, len(positive_rows)), 4)
    negative_rate = round(_ratio(negative_reject, len(negative_rows)), 4)
    return {
        "eval_id": eval_id,
        "eval_kind": "finite_structural_functor_eval",
        "positive_count": len(positive_rows),
        "negative_count": len(negative_rows),
        "positive_functor_pass_rate": positive_rate,
        "negative_functor_rejection_rate": negative_rate,
        "pass": len(positive_rows) >= 5 and len(negative_rows) >= 3 and positive_rate >= 0.8 and negative_rate >= 0.8,
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
    }


def build_structural_context_effect_payload(
    store: JsonlGraphStore | None = None,
    *,
    eval_id: str | None = None,
) -> dict:
    """Controlled offline behavior validation for structural context.

    This is not an LLM benchmark.  It checks that the generated structural
    context is discriminative: expected-pattern context beats both a generic
    baseline and a wrong-pattern placebo on the controlled task rubric.
    """

    rows = []
    guided_wins = 0
    placebo_wins = 0
    for case in _default_behavior_tasks():
        apps = search_structural_patterns(store, case["query"], top_n=3)
        guided = apps[0] if apps else {}
        guided_pattern = _pattern_by_id(guided.get("pattern_id"), store)
        placebo = next((app for app in apps[1:] if app.get("pattern_id") != case["expected_pattern"]), {})
        if not placebo:
            placebo = next(
                ({
                    "pattern_id": pattern["pattern_id"],
                    "pattern_name": pattern.get("name", pattern["pattern_id"]),
                    "score": 0.0,
                    "decision": "placebo",
                    "matched_terms": [],
                    "preserved_invariants": [],
                    "broken_or_uncertain_invariants": [],
                    "negative_control_hits": [],
                    "transfer_predictions": pattern.get("transfer_predictions", []),
                }
                 for pattern in load_structural_patterns(store)
                 if pattern["pattern_id"] != case["expected_pattern"]),
                {},
            )
        placebo_pattern = _pattern_by_id(placebo.get("pattern_id"), store)
        baseline_quality = _structural_answer_quality(_generic_structural_baseline(case), case, {})
        guided_context = format_structural_morphism_applications([guided], max_items=1)
        placebo_context = format_structural_morphism_applications([placebo], max_items=1)
        guided_quality = _structural_answer_quality(
            " ".join([guided_context, _guided_structural_answer(guided, guided_pattern)]),
            case,
            guided_pattern,
        )
        placebo_quality = _structural_answer_quality(
            " ".join([placebo_context, _guided_structural_answer(placebo, placebo_pattern)]),
            case,
            placebo_pattern,
        )
        guided_win = guided_quality["score"] > baseline_quality["score"]
        placebo_margin_ok = guided_quality["score"] > placebo_quality["score"]
        guided_wins += int(guided_win)
        placebo_wins += int(placebo_margin_ok)
        rows.append({
            **case,
            "guided_pattern_id": guided.get("pattern_id"),
            "placebo_pattern_id": placebo.get("pattern_id"),
            "baseline_score": baseline_quality["score"],
            "guided_score": guided_quality["score"],
            "placebo_score": placebo_quality["score"],
            "guided_delta": round(guided_quality["score"] - baseline_quality["score"], 4),
            "placebo_delta": round(guided_quality["score"] - placebo_quality["score"], 4),
            "guided_beats_baseline": guided_win,
            "guided_beats_placebo": placebo_margin_ok,
        })
    count = len(rows)
    guided_rate = round(_ratio(guided_wins, count), 4)
    placebo_rate = round(_ratio(placebo_wins, count), 4)
    mean_guided_delta = round(sum(row["guided_delta"] for row in rows) / count, 4) if count else 0.0
    mean_placebo_delta = round(sum(row["placebo_delta"] for row in rows) / count, 4) if count else 0.0
    return {
        "eval_id": eval_id,
        "eval_kind": "structural_context_effect_probe",
        "task_count": count,
        "guided_win_rate": guided_rate,
        "placebo_discrimination_rate": placebo_rate,
        "mean_guided_delta": mean_guided_delta,
        "mean_placebo_delta": mean_placebo_delta,
        "pass": count >= 4 and guided_rate >= 0.8 and placebo_rate >= 0.75 and mean_guided_delta > 0.25,
        "rows": rows,
    }


def build_transfer_prediction_testability_eval_payload(
    *,
    eval_id: str | None = None,
) -> dict:
    pattern_rows = []
    pattern_pass = 0
    for pattern in load_structural_patterns(None):
        check = assess_transfer_prediction_testability(pattern.get("transfer_predictions", []))
        pattern_pass += int(check.get("pass"))
        pattern_rows.append({
            "pattern_id": pattern["pattern_id"],
            "transfer_predictions": pattern.get("transfer_predictions", []),
            "check": check,
            "passed": bool(check.get("pass")),
        })

    negative_cases = [
        {
            "id": "empty_prediction",
            "predictions": [],
        },
        {
            "id": "inspirational_prediction",
            "predictions": ["This mapping is probably useful and elegant."],
        },
        {
            "id": "no_observable_outcome",
            "predictions": ["Prefer this structural idea when the domain feels similar."],
        },
    ]
    negative_rows = []
    negative_reject = 0
    for case in negative_cases:
        check = assess_transfer_prediction_testability(case["predictions"])
        rejected = not check.get("pass")
        negative_reject += int(rejected)
        negative_rows.append({**case, "check": check, "rejected": rejected})

    # Gate-level positive and negative controls.
    good = search_structural_patterns(
        None,
        "Keep the baseline identity path, apply a residual delta correction, and keep fallback recovery.",
        top_n=1,
    )[0]
    good_payload = {
        "eval_id": eval_id,
        "proposals": [{
            "proposal_id": "prop_testable_prediction_good",
            "proposal_type": "structural_transfer_hypothesis",
            "parent_node_id": "parent",
            "candidate_node": {"id": "cand_testable_prediction_good", "formal_form": good["candidate"]},
        }],
    }
    bad_formal = dict(good["candidate"])
    bad_formal["transfer_predictions"] = ["This analogy is conceptually interesting."]
    bad_formal["transfer_prediction_check"] = assess_transfer_prediction_testability(bad_formal["transfer_predictions"])
    bad_payload = {
        "eval_id": eval_id,
        "proposals": [{
            "proposal_id": "prop_testable_prediction_bad",
            "proposal_type": "structural_transfer_hypothesis",
            "parent_node_id": "parent",
            "candidate_node": {"id": "cand_testable_prediction_bad", "formal_form": bad_formal},
        }],
    }
    good_gate = build_structural_morphism_gate_payload(proposal_payload=good_payload, eval_id=eval_id)
    bad_gate = build_structural_morphism_gate_payload(proposal_payload=bad_payload, eval_id=eval_id)
    pattern_rate = round(_ratio(pattern_pass, len(pattern_rows)), 4)
    negative_rate = round(_ratio(negative_reject, len(negative_rows)), 4)
    return {
        "eval_id": eval_id,
        "eval_kind": "transfer_prediction_testability_eval",
        "pattern_count": len(pattern_rows),
        "pattern_pass_rate": pattern_rate,
        "negative_count": len(negative_rows),
        "negative_rejection_rate": negative_rate,
        "gate_positive_decision": good_gate["gates"][0]["decision"],
        "gate_negative_decision": bad_gate["gates"][0]["decision"],
        "pass": (
            len(pattern_rows) >= 10
            and pattern_rate >= 1.0
            and negative_rate >= 1.0
            and good_gate["gates"][0]["decision"] == "allow"
            and bad_gate["gates"][0]["decision"] == "repair_missing_testable_transfer_prediction"
            and bad_gate["gates"][0]["blocks_policy_update"]
        ),
        "pattern_rows": pattern_rows,
        "negative_rows": negative_rows,
        "good_gate": good_gate,
        "bad_gate": bad_gate,
    }


def build_structural_writeback_eval_payload(*, eval_id: str | None = None) -> dict:
    with tempfile.TemporaryDirectory() as td:
        store = JsonlGraphStore(Path(td) / "graph")
        seed_structural_patterns(store, persist=False)
        proposal_payload = build_structural_transfer_proposal_payload(
            store,
            problem=(
                "Keep the verified baseline identity path, add a residual delta correction, "
                "and recover old behavior through fallback when the delta is zero."
            ),
            eval_id=eval_id,
            top_n=1,
        )
        gate_payload = build_structural_morphism_gate_payload(
            proposal_payload=proposal_payload,
            eval_id=f"{eval_id}_gate" if eval_id else None,
        )
        proposal_id = proposal_payload["proposals"][0]["proposal_id"] if proposal_payload["proposals"] else ""
        acceptance_payload = {
            "eval_id": f"{eval_id}_accept" if eval_id else None,
            "accepted_proposal_ids": [proposal_id],
            "summaries": [{
                "proposal_id": proposal_id,
                "decision": "accept",
                "trigger_utility": 1.0,
                "trigger_lcb90": 0.9,
                "control_loss_ucb90": 0.0,
                "rationale": "synthetic positive-control acceptance for structural writeback validation",
            }],
        }
        applied = apply_accepted_structural_morphisms(
            store,
            proposal_payload,
            gate_payload,
            acceptance_payload,
            persist=False,
        )
        lineage = build_structural_lineage_payload(store, eval_id=eval_id)
        reloaded = JsonlGraphStore(Path(td) / "graph")
        store.flush()
        reloaded.load()
        persisted = all(node_id in reloaded.nodes for node_id in applied)
        edge_types = {_edge_type_value(edge) for edge in store.edges if edge.source in set(applied)}
    return {
        "eval_id": eval_id,
        "eval_kind": "structural_writeback_eval",
        "proposal_count": proposal_payload.get("proposal_count", 0),
        "gate_decision_counts": gate_payload.get("decision_counts", {}),
        "applied_node_ids": applied,
        "lineage": lineage,
        "persisted": persisted,
        "edge_types_from_applied": sorted(edge_types),
        "pass": (
            len(applied) == 1
            and gate_payload.get("decision_counts", {}).get("allow", 0) == 1
            and lineage.get("pass")
            and persisted
            and EdgeType.IS_FORMAL_ISOMORPHISM_OF.value in edge_types
        ),
    }


def build_structural_recursive_runner_eval_payload(*, eval_id: str | None = None) -> dict:
    from .recursive_runner import build_recursive_assumption_run

    with tempfile.TemporaryDirectory() as td:
        graph_dir = Path(td) / "graph"
        store = JsonlGraphStore(graph_dir)
        seed_structural_patterns(store, persist=True)
        problem = (
            "Transfer the residual-correction idea to a risky evaluator rewrite: preserve the "
            "baseline identity path, apply a local delta, and keep fallback recovery."
        )
        proposal_payload = build_structural_transfer_proposal_payload(
            store,
            problem=problem,
            eval_id=eval_id,
            top_n=1,
        )
        gate_payload = build_structural_morphism_gate_payload(
            proposal_payload=proposal_payload,
            eval_id=f"{eval_id}_gate" if eval_id else None,
        )
        evolution_payload = {
            "eval_id": eval_id,
            "proposals": proposal_payload,
            "structural_morphism_gate": gate_payload,
        }
        recursive = build_recursive_assumption_run(
            graph_dir=graph_dir,
            problem=problem,
            goal="Validate the structural transfer hypothesis recursively before graph mutation.",
            eval_id=eval_id or "structural_recursive_eval",
            evolution_payload=evolution_payload,
            top_k=3,
            max_children=2,
            max_depth=2,
            writeback=False,
        )
    structural_children = [
        frame
        for frame in recursive.get("frames", [])
        if frame.get("verifier") == "structural_morphism_gate"
        and frame.get("frame_type") == "verification_subproblem"
    ]
    return {
        "eval_id": eval_id,
        "eval_kind": "structural_recursive_runner_eval",
        "frame_counts": recursive.get("frame_counts", {}),
        "status_counts": recursive.get("status_counts", {}),
        "structural_child_count": len(structural_children),
        "structural_child_next_actions": [frame.get("next_action") for frame in structural_children],
        "pass": bool(structural_children) and any(
            action in {"run_structural_context_effect_validation", "return_structural_gate_to_parent"}
            for action in [frame.get("next_action") for frame in structural_children]
        ),
        "recursive_payload": recursive,
    }


def build_structural_morphism_performance_payload(
    store: JsonlGraphStore | None = None,
    *,
    eval_id: str | None = None,
) -> dict:
    components = {
        "diagram_extraction": build_structural_extraction_audit_payload(eval_id=eval_id),
        "pair_suite": build_structural_pair_eval_payload(store, eval_id=eval_id),
        "nonlexical_retrieval": build_nonlexical_structural_retrieval_probe_payload(store, eval_id=eval_id),
        "functor_eval": build_structural_functor_eval_payload(store, eval_id=eval_id),
        "transfer_prediction_testability": build_transfer_prediction_testability_eval_payload(eval_id=eval_id),
        "context_effect": build_structural_context_effect_payload(store, eval_id=eval_id),
        "behavior_probe": build_structural_behavior_probe_payload(store, eval_id=eval_id),
        "writeback_eval": build_structural_writeback_eval_payload(eval_id=eval_id),
        "recursive_runner_eval": build_structural_recursive_runner_eval_payload(eval_id=eval_id),
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "structural_morphism_performance_validation",
        "component_count": len(components),
        "pass": all(component.get("pass") for component in components.values()),
        "component_pass": {name: bool(component.get("pass")) for name, component in components.items()},
        "components": components,
    }


DEFAULT_STRUCTURAL_PATTERNS = [
    {
        "pattern_id": "pat_residual_correction",
        "name": "Residual Correction / Identity-Preserving Update",
        "claim": "Preserve a verified baseline or identity path while learning only the local delta.",
        "trigger_terms": [
            "residual",
            "skip connection",
            "identity",
            "baseline",
            "delta",
            "fallback",
            "lora",
            "adapter",
            "overwrite",
            "rewrite",
            "destructive overwrite",
        ],
        "objects": [
            {"id": "input_state", "role": "baseline_path", "terms": ["baseline", "identity", "input", "old path", "verified path"]},
            {"id": "delta_update", "role": "delta_update", "terms": ["delta", "residual", "correction", "deviation", "local update"]},
            {"id": "output_state", "role": "baseline_path", "terms": ["output", "fallback", "recover", "old behavior"]},
        ],
        "morphisms": [
            {"id": "identity_path", "role": "baseline_path", "terms": ["identity", "skip", "preserve", "baseline", "fallback"]},
            {"id": "learn_delta", "role": "delta_update", "terms": ["learn delta", "residual", "correction", "deviation"]},
            {"id": "compose_add", "role": "delta_update", "terms": ["add", "plus", "compose", "x + f", "local patch"]},
        ],
        "composition_laws": ["output = identity(input) + delta(input)", "zero delta recovers baseline"],
        "invariants": [
            {"id": "identity_path_preserved", "terms": ["identity", "baseline", "fallback", "preserve", "old path"]},
            {"id": "learned_part_models_deviation", "terms": ["delta", "residual", "correction", "deviation", "local update"]},
            {"id": "zero_delta_recovers_baseline", "terms": ["zero", "fallback", "recover", "rollback", "old behavior"]},
        ],
        "negative_controls": ["plain stack", "uncontrolled overwrite", "no fallback", "without identity", "delete baseline"],
        "good_realizations": ["resnet", "transformer residual", "lora", "adapter", "iterative refinement"],
        "bad_realizations": ["plain feedforward stack", "uncontrolled rewrite", "delete working path"],
        "transfer_predictions": [
            "When a plan risks destructive overwrite, structural context should preserve the old path and apply only a local delta.",
        ],
    },
    {
        "pattern_id": "pat_controlled_intervention",
        "name": "Controlled Intervention / A-B Falsification",
        "claim": "Test one intervention against a matched control before promoting a new assumption.",
        "trigger_terms": ["control", "controls", "controlled variable", "ablation", "a/b", "baseline", "placebo", "falsification"],
        "objects": [
            {"id": "baseline_case", "role": "control_row", "terms": ["baseline", "control", "controls", "placebo", "matched"]},
            {"id": "intervention_case", "role": "control_row", "terms": ["intervention", "variant", "candidate", "ablation"]},
            {"id": "outcome_measure", "role": "control_row", "terms": ["metric", "outcome", "judge", "acceptance"]},
        ],
        "morphisms": [
            {"id": "change_one_factor", "role": "control_row", "terms": ["one variable", "single variable", "one intervention"]},
            {"id": "compare_outcomes", "role": "control_row", "terms": ["compare", "baseline", "control", "ablation"]},
        ],
        "composition_laws": ["one intervention plus matched control identifies causal effect"],
        "invariants": [
            {"id": "single_intervention_isolated", "terms": ["one variable", "single variable", "one intervention", "ablation"]},
            {"id": "matched_control_required", "terms": ["matched control", "control", "baseline", "placebo"]},
        ],
        "negative_controls": ["multiple changes", "no control", "unmatched baseline", "post-hoc metric"],
        "good_realizations": ["controlled variable", "a/b test", "fresh ablation", "trigger control"],
        "bad_realizations": ["bundle many changes", "judge without baseline"],
        "transfer_predictions": [
            "A candidate with control-variable context should declare trigger and control rows before acceptance.",
        ],
    },
    {
        "pattern_id": "pat_incremental_replacement",
        "name": "Incremental Replacement / Module Boundary Preservation",
        "claim": "Keep a working pipeline and replace one bounded module at a time with rollback.",
        "trigger_terms": ["incremental", "replace one", "module", "pipeline", "rollback", "mvp", "adapter boundary"],
        "objects": [
            {"id": "working_pipeline", "role": "baseline_path", "terms": ["working pipeline", "old path", "baseline", "verified path"]},
            {"id": "module_boundary", "role": "module_boundary", "terms": ["module", "component", "boundary", "adapter"]},
            {"id": "replacement_delta", "role": "delta_update", "terms": ["replace one", "local update", "minimal replacement"]},
        ],
        "morphisms": [
            {"id": "preserve_pipeline", "role": "baseline_path", "terms": ["preserve", "keep", "fallback", "old path"]},
            {"id": "swap_one_module", "role": "module_boundary", "terms": ["replace one", "single module", "component"]},
            {"id": "rollback_if_failed", "role": "baseline_path", "terms": ["rollback", "revert", "fallback"]},
        ],
        "composition_laws": ["preserved pipeline + one bounded replacement + rollback supports safe iteration"],
        "invariants": [
            {"id": "module_boundary_preserved", "terms": ["module", "boundary", "component", "adapter"]},
            {"id": "rollback_path_available", "terms": ["rollback", "fallback", "revert", "old path"]},
            {"id": "identity_path_preserved", "terms": ["baseline", "preserve", "working path", "old path"]},
        ],
        "negative_controls": ["rewrite whole system", "many modules at once", "no rollback", "delete working path"],
        "good_realizations": ["mvp", "adapter boundary", "incremental replacement", "strangler fig"],
        "bad_realizations": ["big bang rewrite", "unbounded migration"],
        "transfer_predictions": [
            "For high-risk system changes, one-module replacement should preserve the rollback path and reduce failure versus whole-system rewrite.",
        ],
    },
    {
        "pattern_id": "pat_negative_feedback",
        "name": "Negative Feedback / Equilibrium Restoration",
        "claim": "An induced response opposes an external perturbation under a constraint or stability principle.",
        "trigger_terms": ["negative feedback", "perturbation", "disturbance", "opposes", "equilibrium", "lenz", "le chatelier"],
        "objects": [
            {"id": "system_state", "role": "perturbation", "terms": ["state", "equilibrium", "system"]},
            {"id": "external_perturbation", "role": "perturbation", "terms": ["perturbation", "disturbance", "imposed change"]},
            {"id": "induced_response", "role": "opposing_response", "terms": ["response", "opposes", "compensates", "resists"]},
        ],
        "morphisms": [
            {"id": "perturb_state", "role": "perturbation", "terms": ["perturb", "disturbance", "external change"]},
            {"id": "induce_response", "role": "opposing_response", "terms": ["induce", "response", "reaction"]},
            {"id": "oppose_change", "role": "opposing_response", "terms": ["opposes", "compensates", "resists", "cancels"]},
        ],
        "composition_laws": [
            "perturbation induces response; response opposes perturbation",
            "disturbance creates compensating reaction that cancels change",
        ],
        "invariants": [
            {"id": "response_opposes_perturbation", "terms": ["opposes", "compensates", "resists", "cancels", "negative feedback"]},
            {"id": "constraint_explains_response", "terms": ["constraint", "conservation", "free energy", "equilibrium", "law"]},
        ],
        "negative_controls": ["positive feedback", "random response", "no constraint", "amplifies disturbance"],
        "good_realizations": ["lenz law", "le chatelier", "control feedback", "homeostasis"],
        "bad_realizations": ["positive feedback loop", "runaway amplification"],
        "transfer_predictions": [
            "If the mapping is valid, an answer should identify the perturbation, induced response, and preserved constraint.",
        ],
    },
    {
        "pattern_id": "pat_signal_nuisance_separation",
        "name": "Signal vs Stochastic Nuisance Separation",
        "claim": "Bias estimation toward predictable structure while suppressing stochastic nuisance variation.",
        "trigger_terms": [
            "noise",
            "uncorrelated",
            "gaussian",
            "latent",
            "predictable",
            "jepa",
            "denoise",
            "autocorrelation",
        ],
        "objects": [
            {"id": "stable_signal", "role": "stable_signal", "terms": ["stable signal", "predictable", "correlated", "latent state"]},
            {"id": "nuisance_noise", "role": "nuisance_noise", "terms": ["noise", "nuisance", "uncorrelated", "random", "gaussian"]},
            {"id": "projection_operator", "role": "stable_signal", "terms": ["projection", "prediction", "correlation", "regularization"]},
        ],
        "morphisms": [
            {"id": "suppress_noise", "role": "nuisance_noise", "terms": ["suppress", "ignore", "denoise", "not reconstruct"]},
            {"id": "recover_signal", "role": "stable_signal", "terms": ["recover", "predict", "latent", "stable"]},
        ],
        "composition_laws": ["predictable structure is retained while uncorrelated nuisance is suppressed"],
        "invariants": [
            {"id": "predictable_structure_separated", "terms": ["predictable", "stable signal", "correlated", "latent state", "world state"]},
            {"id": "stochastic_nuisance_suppressed", "terms": ["noise", "nuisance", "uncorrelated", "random", "gaussian", "irrelevant detail"]},
        ],
        "negative_controls": ["any gaussian", "style noise only", "no predictable signal", "memorize noise", "memorizes noise"],
        "good_realizations": ["seismic denoising", "blind spot denoising", "jepa", "latent prediction"],
        "bad_realizations": ["arbitrary gaussian prior", "cosmetic smoothing"],
        "transfer_predictions": [
            "A valid transfer should improve stable-state prediction or denoising while avoiding reconstruction of nuisance details.",
        ],
    },
    {
        "pattern_id": "pat_decomposition_composition",
        "name": "Decomposition / Interface-Preserving Composition",
        "claim": "Split a root problem into bounded subproblems whose interface contracts compose back into the whole goal.",
        "trigger_terms": [
            "decompose",
            "decomposition",
            "split",
            "subproblem",
            "subtask",
            "interface",
            "contract",
            "compose solution",
        ],
        "objects": [
            {"id": "root_problem", "role": "root_problem", "terms": ["root problem", "whole problem", "overall task", "goal"]},
            {"id": "subproblem_node", "role": "subproblem", "terms": ["subproblem", "subtask", "decompose", "split", "factor"]},
            {"id": "interface_contract", "role": "interface_contract", "terms": ["interface", "contract", "schema", "input output", "boundary condition"]},
        ],
        "morphisms": [
            {"id": "split_problem", "role": "subproblem", "terms": ["split", "decompose", "factor", "subproblem", "subtask"]},
            {"id": "solve_subproblem", "role": "subproblem", "terms": ["solve subproblem", "independent part", "interface", "contract"]},
            {"id": "compose_solution", "role": "interface_contract", "terms": ["compose solution", "join result", "recover goal", "overall task"]},
        ],
        "composition_laws": [
            "split problem then solve subproblems through interface contracts; composed solution recovers the root problem",
        ],
        "invariants": [
            {"id": "interface_contract_preserved", "terms": ["interface", "contract", "schema", "input output", "boundary condition"]},
            {"id": "subproblem_independence_preserved", "terms": ["independent", "subproblem", "subtask", "factor", "separable"]},
            {"id": "composed_solution_recovers_goal", "terms": ["compose solution", "recover goal", "overall task", "root problem", "join result"]},
        ],
        "negative_controls": ["entangled subproblems", "no interface", "cannot compose", "hidden dependency", "lost root goal"],
        "good_realizations": ["divide and conquer", "map reduce", "modular proof", "interface contract"],
        "bad_realizations": ["arbitrary checklist", "split without join", "ambiguous handoff"],
        "transfer_predictions": [
            "A valid decomposition transfer should name subproblem interfaces and a composition check that recovers the root goal.",
        ],
    },
    {
        "pattern_id": "pat_bottleneck_capacity",
        "name": "Bottleneck / Capacity-Limited Flow",
        "claim": "System throughput is controlled by a scarce capacity constraint, so interventions should target the bottleneck rather than non-limiting parts.",
        "trigger_terms": [
            "bottleneck",
            "capacity",
            "throughput",
            "queue",
            "rate limit",
            "traffic",
            "scarce resource",
            "token budget",
        ],
        "objects": [
            {"id": "flow_input", "role": "flow_item", "terms": ["flow", "queue", "traffic", "token budget", "work item"]},
            {"id": "bottleneck_resource", "role": "bottleneck_resource", "terms": ["bottleneck", "capacity", "rate limit", "scarce resource"]},
            {"id": "flow_output", "role": "flow_item", "terms": ["throughput", "queue drained", "latency", "output flow"]},
        ],
        "morphisms": [
            {"id": "route_flow", "role": "flow_item", "terms": ["route", "flow", "queue", "traffic", "throughput"]},
            {"id": "constrain_capacity", "role": "bottleneck_resource", "terms": ["capacity", "capacity constraint", "rate limit", "scarce resource"]},
            {"id": "relieve_bottleneck", "role": "bottleneck_resource", "terms": ["relieve", "widen", "remove bottleneck", "increase throughput"]},
        ],
        "composition_laws": ["flow routed through bottleneck capacity determines output throughput"],
        "invariants": [
            {"id": "bottleneck_controls_throughput", "terms": ["bottleneck", "throughput", "capacity", "rate limit", "scarce resource"]},
            {"id": "capacity_constraint_explicit", "terms": ["capacity constraint", "rate limit", "scarce resource", "resource budget"]},
        ],
        "negative_controls": ["optimize non bottleneck", "no bottleneck", "unlimited capacity", "shift bottleneck without measuring"],
        "good_realizations": ["queueing bottleneck", "critical path", "rate limiter", "resource contention"],
        "bad_realizations": ["local optimization away from bottleneck", "throughput claim without capacity"],
        "transfer_predictions": [
            "A valid bottleneck transfer should identify the limiting resource and predict that relieving it improves throughput more than optimizing non-bottleneck steps.",
        ],
    },
    {
        "pattern_id": "pat_counterexample_refinement",
        "name": "Counterexample-Guided Claim Refinement",
        "claim": "Use an adversarial counterexample to falsify an overbroad claim, then narrow or patch the claim with a guardrail.",
        "trigger_terms": [
            "counterexample",
            "adversarial",
            "edge case",
            "failure case",
            "falsify",
            "refine",
            "patch",
            "guardrail",
        ],
        "objects": [
            {"id": "claim_under_test", "role": "refined_claim", "terms": ["claim", "hypothesis", "overbroad claim", "assumption"]},
            {"id": "counterexample_case", "role": "counterexample", "terms": ["counterexample", "adversarial", "edge case", "failure case"]},
            {"id": "refined_claim", "role": "refined_claim", "terms": ["refine", "patch", "narrow", "guardrail", "revised claim"]},
        ],
        "morphisms": [
            {"id": "generate_counterexample", "role": "counterexample", "terms": ["generate counterexample", "adversarial", "edge case", "failure case"]},
            {"id": "falsify_claim", "role": "counterexample", "terms": ["falsify", "breaks", "disprove", "violates"]},
            {"id": "patch_claim", "role": "refined_claim", "terms": ["patch", "refine", "narrow", "guardrail", "revised claim"]},
        ],
        "composition_laws": ["counterexample falsifies claim; refined claim handles the counterexample with an explicit guardrail"],
        "invariants": [
            {"id": "counterexample_targets_claim", "terms": ["counterexample", "falsify", "breaks", "edge case", "adversarial"]},
            {"id": "refinement_handles_counterexample", "terms": ["refine", "patch", "narrow", "guardrail", "revised claim"]},
        ],
        "negative_controls": ["ignore counterexample", "patch unrelated claim", "no falsification", "broaden claim after failure"],
        "good_realizations": ["CEGIS", "red-team repair", "property-based testing", "refinement loop"],
        "bad_realizations": ["post-hoc excuse", "unrelated guardrail", "counterexample discarded"],
        "transfer_predictions": [
            "A valid counterexample transfer should produce a specific failing case and a narrower claim that the same case no longer breaks.",
        ],
    },
    {
        "pattern_id": "pat_conservation_balance",
        "name": "Conservation / Balance-Preserving Transformation",
        "claim": "A transformation is valid only when a conserved quantity or budget remains balanced before and after the state change.",
        "trigger_terms": [
            "conservation",
            "conserved quantity",
            "mass balance",
            "energy balance",
            "budget balance",
            "probability mass",
            "accounting",
            "invariant quantity",
        ],
        "objects": [
            {"id": "source_state", "role": "transformation", "terms": ["source state", "before state", "input state", "state change"]},
            {"id": "conserved_quantity", "role": "conserved_quantity", "terms": ["conserved quantity", "invariant quantity", "mass balance", "energy balance", "budget balance", "probability mass", "budget"]},
            {"id": "target_state", "role": "transformation", "terms": ["target state", "after state", "output state", "transition"]},
        ],
        "morphisms": [
            {"id": "transform_state", "role": "transformation", "terms": ["transform", "transition", "state change", "conversion"]},
            {"id": "conserve_quantity", "role": "conserved_quantity", "terms": ["conserve", "conserved quantity", "balance", "invariant quantity"]},
            {"id": "check_balance", "role": "conserved_quantity", "terms": ["check balance", "accounting", "closed", "sum"]},
        ],
        "composition_laws": ["state transformation plus balance check preserves the conserved quantity"],
        "invariants": [
            {"id": "conserved_quantity_preserved", "terms": ["conserved quantity", "invariant quantity", "mass balance", "energy balance", "budget balance", "probability mass"]},
            {"id": "balance_check_closes", "terms": ["balance", "accounting", "check", "closed", "sum"]},
        ],
        "negative_controls": ["leaks budget", "unbalanced accounting", "creates quantity", "destroys quantity", "no balance check"],
        "good_realizations": ["mass balance", "energy conservation", "budget accounting", "probability mass preservation"],
        "bad_realizations": ["untracked loss", "magic resource creation", "normalization ignored"],
        "transfer_predictions": [
            "A valid conservation transfer should expose the conserved quantity and a before/after balance check.",
        ],
    },
    {
        "pattern_id": "pat_monotone_progress",
        "name": "Monotone Progress / Order-Preserving Improvement",
        "claim": "A safe iterative update should preserve an ordering or objective measure so progress does not regress.",
        "trigger_terms": [
            "monotonic",
            "monotone",
            "non-decreasing",
            "ordered",
            "partial order",
            "dominance",
            "progress",
            "objective",
        ],
        "objects": [
            {"id": "ordered_state", "role": "ordered_state", "terms": ["ordered state", "partial order", "ranked state", "dominance"]},
            {"id": "objective_measure", "role": "objective_measure", "terms": ["objective", "score", "loss", "utility", "progress"]},
            {"id": "updated_state", "role": "ordered_state", "terms": ["updated state", "non-decreasing", "monotonic", "improve"]},
        ],
        "morphisms": [
            {"id": "apply_monotone_step", "role": "ordered_state", "terms": ["monotonic", "monotone step", "non-decreasing", "ordered"]},
            {"id": "preserve_order", "role": "ordered_state", "terms": ["preserve order", "dominance", "partial order"]},
            {"id": "measure_progress", "role": "objective_measure", "terms": ["measure progress", "score", "objective", "utility", "improve"]},
        ],
        "composition_laws": ["monotone update preserves order and produces non-decreasing objective progress"],
        "invariants": [
            {"id": "order_preserved", "terms": ["ordered", "monotonic", "non-decreasing", "partial order", "dominance"]},
            {"id": "progress_non_decreasing", "terms": ["progress", "objective", "score", "utility", "non-decreasing", "improve"]},
        ],
        "negative_controls": ["regression allowed", "unmeasured progress", "order reversal", "improves metric by breaking invariant"],
        "good_realizations": ["policy improvement", "coordinate ascent", "curriculum progress", "monotone optimization"],
        "bad_realizations": ["oscillating update", "reward hacking", "metric-only improvement with broken order"],
        "transfer_predictions": [
            "A valid monotone-progress transfer should define the order/objective and detect any update that regresses it.",
        ],
    },
]


def _pattern_node(spec: dict) -> AssumptionNode:
    pattern_id = spec["pattern_id"]
    enriched = _pattern_with_endpoint_hints(spec)
    return AssumptionNode(
        id=f"struct_{pattern_id}",
        type=AssumptionType.ALIGNMENT,
        kind=HypothesisKind.FORMAL_MAPPING,
        claim=enriched["claim"],
        formal_form={"formal_kind": STRUCTURAL_PATTERN_KIND, **enriched},
        tags=["structural_pattern", "structural_morphism", pattern_id, *spec.get("trigger_terms", [])[:6]],
        context_conditions=["structural transfer", "cross-domain analogy", "recursive assumption validation"],
        predicted_effects=enriched.get("transfer_predictions", []),
        risk_predictions=[
            "structural analogy can overfit if broken invariants or negative controls are ignored",
        ],
        verifiers=["structural_pair_suite", "nonlexical_structural_retrieval_probe", "behavior_probe"],
        confidence=0.62,
        metaproductivity=0.2,
        source_refs=["reconstruction/md/category_structural_morphism_layer_plan_20260602.md"],
    )


def _node_to_pattern(node: AssumptionNode) -> dict | None:
    formal = node.formal_form or {}
    if not isinstance(formal, dict) or formal.get("formal_kind") != STRUCTURAL_PATTERN_KIND:
        return None
    pattern = dict(formal)
    pattern["node_id"] = node.id
    return _pattern_with_endpoint_hints(pattern)


def _pattern_with_endpoint_hints(pattern: dict) -> dict:
    enriched = dict(pattern)
    morphisms = []
    for row in pattern.get("morphisms", []):
        m = dict(row)
        src, dst = _morphism_endpoints(m)
        if src:
            m.setdefault("source", src)
        if dst:
            m.setdefault("target", dst)
        morphisms.append(m)
    enriched["morphisms"] = morphisms
    return enriched


def _morphism_endpoints(morphism: dict) -> tuple[str | None, str | None]:
    if morphism.get("source") or morphism.get("target"):
        return morphism.get("source"), morphism.get("target")
    return MORPHISM_ENDPOINT_HINTS.get(morphism.get("id"), (None, None))


def _as_structural_diagram(source: str | dict | StructuralSignature | StructuralDiagram) -> StructuralDiagram:
    if isinstance(source, StructuralDiagram):
        return source
    if isinstance(source, dict) and source.get("formal_kind") == "structural_diagram":
        return StructuralDiagram(
            source_text=source.get("source_text", ""),
            objects=list(source.get("objects", [])),
            morphisms=list(source.get("morphisms", [])),
            composition_laws=list(source.get("composition_laws", [])),
            invariants=list(source.get("invariants", [])),
            negation_hits=list(source.get("negation_hits", [])),
            pattern_hints=list(source.get("pattern_hints", [])),
        )
    return extract_structural_diagram(source)


def _pattern_diagram(pattern: dict) -> dict:
    pattern = _pattern_with_endpoint_hints(pattern)
    return {
        "formal_kind": "source_structural_diagram",
        "pattern_id": pattern.get("pattern_id"),
        "name": pattern.get("name"),
        "objects": pattern.get("objects", []),
        "morphisms": pattern.get("morphisms", []),
        "composition_laws": pattern.get("composition_laws", []),
        "invariants": pattern.get("invariants", []),
        "negative_controls": pattern.get("negative_controls", []),
    }


def _infer_composition_laws(
    roles: set[str],
    morphism_ids: set[str],
    signature: StructuralSignature,
) -> list[dict]:
    laws = []
    if {"baseline_path", "delta_update"} <= roles and {"identity_path", "learn_delta"} & morphism_ids:
        laws.append({
            "id": "compose_identity_plus_delta",
            "law": "baseline path composed with local delta yields updated output while zero delta recovers baseline",
            "morphisms": ["identity_path", "learn_delta", "compose_add"],
        })
    if "control_row" in roles:
        laws.append({
            "id": "compose_intervention_control_compare",
            "law": "single intervention and matched control compose into causal outcome comparison",
            "morphisms": ["change_one_factor", "compare_outcomes"],
        })
    if {"baseline_path", "module_boundary"} <= roles:
        laws.append({
            "id": "compose_preserve_swap_rollback",
            "law": "preserve pipeline, swap one module, and rollback compose into safe incremental replacement",
            "morphisms": ["preserve_pipeline", "swap_one_module", "rollback_if_failed"],
        })
    if {"perturbation", "opposing_response"} <= roles:
        laws.append({
            "id": "compose_perturb_induce_oppose",
            "law": "perturbation induces response and response opposes the perturbation",
            "morphisms": ["perturb_state", "induce_response", "oppose_change"],
        })
    if {"stable_signal", "nuisance_noise"} <= roles:
        laws.append({
            "id": "compose_suppress_noise_recover_signal",
            "law": "suppress nuisance variation while recovering predictable stable signal",
            "morphisms": ["suppress_noise", "recover_signal"],
        })
    if signature.negation_hits:
        laws.append({
            "id": "broken_explicit_negation",
            "law": "explicit negation breaks at least one claimed structural invariant",
            "morphisms": [],
            "broken": True,
            "negation_hits": signature.negation_hits,
        })
    return laws


def _morphisms_for_law(law: str, morphisms: list[dict]) -> list[str]:
    low = law.lower()
    out = []
    for morphism in morphisms:
        mid = str(morphism.get("id", ""))
        if mid and mid.lower() in low:
            out.append(mid)
            continue
        if _term_hits(low, morphism.get("terms", [])):
            out.append(mid)
    if out:
        return sorted(set(out))
    # Many seed laws are prose.  If no morphism name is mentioned, require the
    # whole pattern's morphism set as the finite composition support.
    return [str(m.get("id")) for m in morphisms if m.get("id")]


def _target_composition_law_hit(diagram: StructuralDiagram, source_law: str, mapped_morphisms: list[str]) -> bool:
    source_tokens = set(tokenize(source_law))
    for law in diagram.composition_laws:
        if law.get("broken"):
            continue
        if set(law.get("morphisms", [])) & set(mapped_morphisms):
            return True
        if source_tokens & set(tokenize(str(law.get("law", "")))):
            return True
    return False


def _structural_candidate_node(proposal: dict) -> AssumptionNode | None:
    candidate = proposal.get("candidate_node") or {}
    if not candidate:
        return None
    try:
        if {"type", "claim"}.issubset(candidate):
            return AssumptionNode.from_dict(candidate)
    except Exception:
        pass
    formal = candidate.get("formal_form") or {}
    if not isinstance(formal, dict) or formal.get("formal_kind") != STRUCTURAL_MORPHISM_KIND:
        return None
    node_id = candidate.get("id") or stable_id("struct_cand", proposal.get("proposal_id"), formal.get("source_pattern_id"))
    return AssumptionNode(
        id=node_id,
        type=AssumptionType.ALIGNMENT,
        kind=HypothesisKind.FORMAL_MAPPING,
        claim=candidate.get("claim") or f"Accepted structural morphism from {formal.get('source_pattern_id')}",
        formal_form=formal,
        context_conditions=candidate.get("context_conditions", ["structural_transfer_hypothesis"]),
        predicted_effects=formal.get("transfer_predictions", []),
        risk_predictions=["may over-transfer if target-side composition is not preserved"],
        verifiers=["structural_morphism_gate", "candidate_acceptance_gate"],
        confidence=float(candidate.get("confidence", 0.62) or 0.62),
        metaproductivity=float(candidate.get("metaproductivity", 0.1) or 0.1),
        status=candidate.get("status", "candidate"),
        tags=["structural_transfer", "structural_morphism", str(formal.get("source_pattern_id", ""))],
        payload=candidate.get("payload", {}),
    )


def _edge_type_value(edge: AssumptionEdge) -> str:
    return edge.type.value if isinstance(edge.type, EdgeType) else str(edge.type)


def _source_text(source: str | dict) -> str:
    if isinstance(source, StructuralDiagram):
        return source.source_text
    if isinstance(source, StructuralSignature):
        return source.source_text
    if isinstance(source, str):
        return source
    if not isinstance(source, dict):
        return str(source)
    parts = []
    for key in ("claim", "description", "query", "problem", "source_text"):
        if source.get(key):
            parts.append(str(source[key]))
    formal = source.get("formal_form") if isinstance(source.get("formal_form"), dict) else source
    for key in ("name", "claim", "composition_laws", "invariants", "objects", "morphisms", "trigger_terms"):
        value = formal.get(key) if isinstance(formal, dict) else None
        if value:
            parts.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return " ".join(parts)


def _term_hits(text: str, terms: Iterable[str]) -> list[str]:
    hits = []
    for term in terms or []:
        t = str(term).strip().lower()
        if t and _contains_term(text, t):
            hits.append(str(term))
    return sorted(set(hits), key=lambda x: x.lower())


def _contains_term(text: str, term: str) -> bool:
    if re.fullmatch(r"[a-z0-9_+-]+(?:\s+[a-z0-9_+-]+)*", term):
        phrase = r"\s+".join(re.escape(part) for part in term.split())
        return re.search(rf"(?<![a-z0-9_]){phrase}(?![a-z0-9_])", text, flags=re.IGNORECASE) is not None
    return term in text


def _role_rows_hit(text: str, rows: list[dict]) -> list[dict]:
    hits = []
    for row in rows:
        matched = _term_hits(text, row.get("terms", []))
        marker_hits = _term_hits(text, ROLE_MARKERS.get(row.get("role"), []))
        all_hits = sorted(set(matched + marker_hits), key=lambda x: x.lower())
        if all_hits:
            hits.append({**row, "matched_terms": all_hits})
    return hits


def _invariants_hit(text: str, rows: list[dict]) -> list[dict]:
    hits = []
    for row in rows:
        matched = _term_hits(text, row.get("terms", []))
        marker_hits = _term_hits(text, INVARIANT_MARKERS.get(row.get("id"), []))
        all_hits = sorted(set(matched + marker_hits), key=lambda x: x.lower())
        if all_hits:
            hits.append({**row, "matched_terms": all_hits})
    return hits


def _composition_hits(text: str, composition_laws: list[str]) -> int:
    hits = 0
    for law in composition_laws:
        terms = [term for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{3,}", str(law).lower()) if term not in {"input", "output"}]
        if any(term in text for term in terms):
            hits += 1
    return hits


def _negative_hits(text: str, signature: StructuralSignature, pattern: dict) -> list[str]:
    hits = _term_hits(text, pattern.get("negative_controls", [])) + _term_hits(text, pattern.get("bad_realizations", []))
    hits.extend(signature.negation_hits)
    return sorted(set(hits), key=lambda x: x.lower())


def _has_severe_negative(hits: list[str]) -> bool:
    severe_terms = ("no ", "without ", "delete ", "memorize", "plain stack", "rewrite everything")
    return any(str(hit).lower().startswith(severe_terms) or f" no " in f" {str(hit).lower()} " for hit in hits)


def _gate_decision(
    *,
    object_cov: float,
    morphism_cov: float,
    invariant_cov: float,
    composition_cov: float,
    negative_score: float,
    margin: float,
    severe_negative: bool,
    transfer_predictions: list[str],
) -> tuple[str, str]:
    if not transfer_predictions:
        return "repair_missing_transfer_prediction", "No falsifiable transfer prediction is attached."
    if severe_negative:
        return "block_negative_control", "A severe negative-control or explicit missing-structure signal is present."
    if negative_score >= 0.5 and margin <= 0.25:
        return "block_negative_control", "Negative-control evidence is too close to or stronger than the structural match."
    if object_cov < 0.45 or morphism_cov < 0.45 or invariant_cov < 0.45:
        return "repair_under_specified", "The diagram lacks enough object, morphism, or invariant coverage."
    if composition_cov < 0.35:
        return "repair_under_specified", "The proposed mapping does not preserve enough composition structure."
    if invariant_cov >= 0.68 and margin > 0.0:
        return "allow", "Structural mapping preserves enough roles and invariants for shadow-mode transfer."
    return "candidate_shadow_only", "Structural mapping is plausible but should remain shadow-only until stronger evidence arrives."


def assess_transfer_prediction_testability(predictions: list[str]) -> dict:
    """Check whether transfer predictions can become falsification work.

    This is a bounded lexical audit, not a judge.  It blocks empty or purely
    inspirational predictions before they can affect graph policy.
    """

    text = " ".join(str(p) for p in predictions or []).lower()
    action_hits = _term_hits(text, [
        "predict",
        "improve",
        "identify",
        "declare",
        "name",
        "produce",
        "define",
        "detect",
        "expose",
        "preserve",
        "apply",
        "prefer",
        "check",
        "verify",
        "compare",
        "relieve",
        "reduce",
    ])
    observable_hits = _term_hits(text, [
        "throughput",
        "control rows",
        "trigger",
        "acceptance",
        "before",
        "after",
        "balance",
        "objective",
        "progress",
        "regress",
        "failing case",
        "same case",
        "subproblem",
        "interface",
        "composition check",
        "stable-state prediction",
        "denoising",
        "old path",
        "local delta",
        "rollback",
        "failure",
        "perturbation",
        "response",
        "constraint",
    ])
    control_hits = _term_hits(text, [
        "control",
        "negative",
        "regress",
        "before",
        "after",
        "same case",
        "non-bottleneck",
        "without",
        "check",
        "compare",
        "versus",
        "whole-system rewrite",
    ])
    score = (
        0.45 * min(1.0, len(action_hits) / 2)
        + 0.45 * min(1.0, len(observable_hits) / 2)
        + 0.10 * min(1.0, len(control_hits))
    )
    passed = bool(predictions) and score >= 0.55 and bool(action_hits) and bool(observable_hits)
    missing = []
    if not predictions:
        missing.append("prediction_text")
    if not action_hits:
        missing.append("action_or_claim")
    if not observable_hits:
        missing.append("observable_outcome")
    return {
        "formal_kind": "transfer_prediction_testability",
        "prediction_count": len(predictions or []),
        "score": round(score, 4),
        "pass": passed,
        "action_hits": action_hits,
        "observable_hits": observable_hits,
        "control_hits": control_hits,
        "missing": missing,
        "reason": (
            "Transfer prediction is testable."
            if passed
            else "Transfer prediction lacks " + ", ".join(missing or ["enough falsifiable detail"])
        ),
    }


def _ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def _precision(tp: int, fp: int) -> float:
    return round(tp / (tp + fp), 4) if tp + fp else 0.0


def _recall(tp: int, fn: int) -> float:
    return round(tp / (tp + fn), 4) if tp + fn else 0.0


def _pattern_by_id(pattern_id: str | None, store: JsonlGraphStore | None = None) -> dict:
    for pattern in load_structural_patterns(store):
        if pattern.get("pattern_id") == pattern_id:
            return pattern
    return {}


def _default_extraction_audit_cases() -> list[dict]:
    return [
        {
            "id": "extract_residual",
            "text": "Keep the verified baseline path, add a residual delta correction, and recover old behavior when the delta is zero.",
            "expected_roles": ["baseline_path", "delta_update"],
            "expected_invariants": [
                "identity_path_preserved",
                "learned_part_models_deviation",
                "zero_delta_recovers_baseline",
            ],
        },
        {
            "id": "extract_control",
            "text": "Run one intervention against a matched control baseline before accepting the candidate.",
            "expected_roles": ["control_row"],
            "expected_invariants": ["single_intervention_isolated", "matched_control_required"],
        },
        {
            "id": "extract_incremental",
            "text": "Preserve the working pipeline, replace one module at a component boundary, and keep rollback available.",
            "expected_roles": ["baseline_path", "module_boundary"],
            "expected_invariants": ["module_boundary_preserved", "rollback_path_available", "identity_path_preserved"],
        },
        {
            "id": "extract_feedback",
            "text": "An external disturbance induces a response that opposes the imposed change because a conservation constraint must hold.",
            "expected_roles": ["perturbation", "opposing_response"],
            "expected_invariants": ["response_opposes_perturbation", "constraint_explains_response"],
        },
        {
            "id": "extract_signal_noise",
            "text": "The latent world state is predictable and correlated, while Gaussian uncorrelated noise is nuisance detail to suppress.",
            "expected_roles": ["stable_signal", "nuisance_noise"],
            "expected_invariants": ["predictable_structure_separated", "stochastic_nuisance_suppressed"],
        },
        {
            "id": "extract_broken",
            "text": "The rewrite has no baseline, no fallback, and no control row even though it mentions a candidate metric.",
            "expected_roles": ["control_row", "baseline_path"],
            "expected_invariants": ["matched_control_required", "identity_path_preserved"],
            "expected_broken_invariant": True,
        },
    ]


def _default_positive_pair_cases() -> list[dict]:
    return [
        {
            "id": "pair_resnet_to_runner",
            "query": "A solver keeps the baseline identity path, applies a residual delta correction, and can rollback to old behavior if the local update fails.",
            "expected": "pat_residual_correction",
        },
        {
            "id": "pair_control_ablation",
            "query": "The candidate should change one variable, compare against a matched control baseline, and use ablation outcome before acceptance.",
            "expected": "pat_controlled_intervention",
        },
        {
            "id": "pair_incremental_replacement",
            "query": "Keep the working pipeline, replace one module behind an adapter boundary, and rollback rather than rewrite the whole system.",
            "expected": "pat_incremental_replacement",
        },
        {
            "id": "pair_feedback",
            "query": "A disturbance perturbs equilibrium and induces a response that opposes the imposed change under a conservation law.",
            "expected": "pat_negative_feedback",
        },
        {
            "id": "pair_signal_noise",
            "query": "A latent world-state predictor should keep predictable correlated signal and suppress Gaussian uncorrelated nuisance noise instead of reconstructing irrelevant detail.",
            "expected": "pat_signal_nuisance_separation",
        },
        {
            "id": "pair_decomposition_composition",
            "query": "Decompose the root problem into independent subproblems, preserve interface contracts, and compose solution outputs back to the overall goal.",
            "expected": "pat_decomposition_composition",
        },
        {
            "id": "pair_bottleneck_capacity",
            "query": "The queue throughput is limited by a bottleneck capacity constraint, so relieve the scarce resource instead of optimizing non-bottleneck steps.",
            "expected": "pat_bottleneck_capacity",
        },
        {
            "id": "pair_counterexample_refinement",
            "query": "Generate an adversarial counterexample that falsifies the overbroad claim, then patch and narrow the revised claim with a guardrail.",
            "expected": "pat_counterexample_refinement",
        },
        {
            "id": "pair_conservation_balance",
            "query": "A state transformation must preserve a conserved quantity with before and after mass balance accounting checks.",
            "expected": "pat_conservation_balance",
        },
        {
            "id": "pair_monotone_progress",
            "query": "Each monotone update should preserve the partial order and show non-decreasing objective progress without regression.",
            "expected": "pat_monotone_progress",
        },
    ]


def _default_negative_pair_cases() -> list[dict]:
    return [
        {
            "id": "neg_synonym_only",
            "query": "Two papers use similar names and neighboring words, but no roles, morphisms, invariant, control, or transfer prediction are specified.",
        },
        {
            "id": "neg_residual_control",
            "query": "A plain stack rewrites every layer without identity, no fallback, and no residual path.",
        },
        {
            "id": "neg_signal_noise",
            "query": "A Gaussian style prior is mentioned, but there is no predictable signal and the method memorizes noise.",
        },
    ]


def _default_nonlexical_queries() -> list[dict]:
    return [
        {
            "id": "probe_delta_path",
            "query": "The new method should keep a verified path and only alter the part that differs, so the old behavior is recoverable.",
            "expected": "pat_residual_correction",
        },
        {
            "id": "probe_single_factor",
            "query": "Before promotion, isolate the candidate effect by comparing one changed factor with a matched baseline row.",
            "expected": "pat_controlled_intervention",
        },
        {
            "id": "probe_safe_swap",
            "query": "Use a component boundary and swap a single part while retaining a revert path for the rest of the pipeline.",
            "expected": "pat_incremental_replacement",
        },
        {
            "id": "probe_opposition",
            "query": "A system change creates a compensating reaction that cancels the disturbance because a constraint must remain valid.",
            "expected": "pat_negative_feedback",
        },
        {
            "id": "probe_predictable_structure",
            "query": "The representation should model stable predictable structure and ignore random uncorrelated nuisance variation.",
            "expected": "pat_signal_nuisance_separation",
        },
        {
            "id": "probe_interface_join",
            "query": "Factor the parent goal into separable parts, give each part a clear handoff contract, and verify the joined result still solves the parent goal.",
            "expected": "pat_decomposition_composition",
        },
        {
            "id": "probe_capacity_limiter",
            "query": "Work piles up at one scarce capacity point; improving unrelated stages will not increase end-to-end output.",
            "expected": "pat_bottleneck_capacity",
        },
        {
            "id": "probe_failure_refines_claim",
            "query": "Find a concrete edge failure that breaks the assumption, then narrow the assumption so that same failure is handled.",
            "expected": "pat_counterexample_refinement",
        },
        {
            "id": "probe_balance_accounting",
            "query": "Track an invariant quantity through the transition and close the before-after accounting balance.",
            "expected": "pat_conservation_balance",
        },
        {
            "id": "probe_ordered_improvement",
            "query": "Accept only updates that keep the dominance relation and make measured progress non-decreasing.",
            "expected": "pat_monotone_progress",
        },
    ]


def _default_behavior_tasks() -> list[dict]:
    return [
        {
            "id": "behavior_overwrite_repair",
            "query": "A plan wants to rewrite the whole evaluator and risks destructive overwrite; keep a baseline fallback and apply only a local delta.",
            "expected_pattern": "pat_residual_correction",
            "required_terms": ["baseline", "delta", "fallback"],
            "forbidden_terms": ["rewrite everything", "delete baseline"],
        },
        {
            "id": "behavior_candidate_gate",
            "query": "A new route policy looks promising but has not been tested against controls.",
            "expected_pattern": "pat_controlled_intervention",
            "required_terms": ["control", "baseline", "one intervention"],
            "forbidden_terms": ["accept immediately"],
        },
        {
            "id": "behavior_module_swap",
            "query": "A world-model pipeline should improve one component without losing the current working path.",
            "expected_pattern": "pat_incremental_replacement",
            "required_terms": ["module", "pipeline", "rollback"],
            "forbidden_terms": ["big bang rewrite"],
        },
        {
            "id": "behavior_latent_noise",
            "query": "A latent predictor should avoid reconstructing random detail and focus on stable world-state features.",
            "expected_pattern": "pat_signal_nuisance_separation",
            "required_terms": ["predictable", "noise", "suppress"],
            "forbidden_terms": ["memorize noise"],
        },
        {
            "id": "behavior_decomposition_contract",
            "query": "A complex task should be split into separable parts without losing the parent goal.",
            "expected_pattern": "pat_decomposition_composition",
            "required_terms": ["interface", "subproblem", "goal"],
            "forbidden_terms": ["arbitrary checklist"],
        },
        {
            "id": "behavior_bottleneck_capacity",
            "query": "A pipeline has low throughput because one stage is capacity limited.",
            "expected_pattern": "pat_bottleneck_capacity",
            "required_terms": ["bottleneck", "capacity", "throughput"],
            "forbidden_terms": ["optimize non bottleneck"],
        },
        {
            "id": "behavior_counterexample_refinement",
            "query": "A broad assumption failed on an edge case and needs a precise repair.",
            "expected_pattern": "pat_counterexample_refinement",
            "required_terms": ["counterexample", "refine", "patch"],
            "forbidden_terms": ["ignore counterexample"],
        },
        {
            "id": "behavior_conservation_balance",
            "query": "A transformation changes state but should not leak budget or probability mass.",
            "expected_pattern": "pat_conservation_balance",
            "required_terms": ["conserved quantity", "balance", "check"],
            "forbidden_terms": ["leaks budget"],
        },
        {
            "id": "behavior_monotone_progress",
            "query": "An iterative policy update should improve without regressing the ordering constraint.",
            "expected_pattern": "pat_monotone_progress",
            "required_terms": ["monotonic", "progress", "objective"],
            "forbidden_terms": ["regression allowed"],
        },
    ]


def _generic_structural_baseline(case: dict) -> str:
    return f"Analyze the task and propose a reasonable method for {case['id']}."


def _guided_structural_answer(app: dict, pattern: dict) -> str:
    if not app or not pattern:
        return ""
    terms = []
    for inv in app.get("preserved_invariants", []):
        terms.append(inv.replace("_", " "))
    for prediction in app.get("transfer_predictions", [])[:1]:
        terms.append(str(prediction))
    for row in pattern.get("objects", [])[:3]:
        terms.extend(str(term) for term in row.get("terms", [])[:2])
    return " ".join(terms)


def _structural_answer_quality(answer: str, case: dict, pattern: dict) -> dict:
    text = answer.lower()
    required = [str(term).lower() for term in case.get("required_terms", [])]
    forbidden = [str(term).lower() for term in case.get("forbidden_terms", [])]
    required_hits = [term for term in required if term in text]
    forbidden_hits = [term for term in forbidden if term in text]
    pattern_bonus = 0.0
    if pattern and pattern.get("pattern_id") == case.get("expected_pattern"):
        pattern_bonus = 0.15
    score = min(1.0, 0.2 + 0.65 * _ratio(len(required_hits), len(required)) + pattern_bonus)
    score = max(0.0, score - 0.25 * len(forbidden_hits))
    return {
        "score": round(score, 4),
        "required_hits": required_hits,
        "forbidden_hits": forbidden_hits,
        "pattern_id": pattern.get("pattern_id") if pattern else None,
    }


def _resolve(root: Path, path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else root / p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--graph-dir", default=None)
    ap.add_argument("--query", default=None)
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--seed-defaults", action="store_true")
    ap.add_argument("--extraction-audit", action="store_true")
    ap.add_argument("--pair-eval", action="store_true")
    ap.add_argument("--retrieval-probe", action="store_true")
    ap.add_argument("--behavior-probe", action="store_true")
    ap.add_argument("--functor-eval", action="store_true")
    ap.add_argument("--prediction-testability-eval", action="store_true")
    ap.add_argument("--context-effect", action="store_true")
    ap.add_argument("--writeback-eval", action="store_true")
    ap.add_argument("--recursive-runner-eval", action="store_true")
    ap.add_argument("--performance-validation", action="store_true")
    ap.add_argument("--eval-id", default=None)
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    store = JsonlGraphStore(_resolve(root, args.graph_dir)) if args.graph_dir else None
    payload = build_structural_pattern_payload(store, eval_id=args.eval_id)
    if args.seed_defaults:
        if not store:
            raise SystemExit("--seed-defaults requires --graph-dir")
        payload["seeded_node_ids"] = seed_structural_patterns(store, persist=True)
    if args.query:
        payload["search"] = search_structural_patterns(store, args.query, top_n=args.top_n)
        payload["formatted_search"] = format_structural_morphism_applications(payload["search"])
    if args.extraction_audit:
        payload["extraction_audit"] = build_structural_extraction_audit_payload(eval_id=args.eval_id)
    if args.pair_eval:
        payload["pair_eval"] = build_structural_pair_eval_payload(store, eval_id=args.eval_id)
    if args.retrieval_probe:
        payload["retrieval_probe"] = build_nonlexical_structural_retrieval_probe_payload(store, eval_id=args.eval_id)
    if args.behavior_probe:
        payload["behavior_probe"] = build_structural_behavior_probe_payload(store, eval_id=args.eval_id)
    if args.functor_eval:
        payload["functor_eval"] = build_structural_functor_eval_payload(store, eval_id=args.eval_id)
    if args.prediction_testability_eval:
        payload["prediction_testability_eval"] = build_transfer_prediction_testability_eval_payload(eval_id=args.eval_id)
    if args.context_effect:
        payload["context_effect"] = build_structural_context_effect_payload(store, eval_id=args.eval_id)
    if args.writeback_eval:
        payload["writeback_eval"] = build_structural_writeback_eval_payload(eval_id=args.eval_id)
    if args.recursive_runner_eval:
        payload["recursive_runner_eval"] = build_structural_recursive_runner_eval_payload(eval_id=args.eval_id)
    if args.performance_validation:
        payload["performance_validation"] = build_structural_morphism_performance_payload(store, eval_id=args.eval_id)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
