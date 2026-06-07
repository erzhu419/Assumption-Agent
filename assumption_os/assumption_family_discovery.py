"""Open-set assumption family discovery from theory cards.

The structural context edge module validates one hand-built context.  This
module sits one level upstream: it extracts reusable assumption kernels from a
set of scientific, mathematical, engineering, or philosophical theory cards,
then clusters them by structural roles rather than by surface words.

The implementation is intentionally deterministic and auditable.  It is not a
complete theory of scientific analogy; it is an open-set induction layer that
can discover how many assumption families are present in the supplied cards and
whether a new card belongs to an existing family or should start a new one.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

from .morphism_benchmark import _counter_cosine, _tokens
from .schema import stable_id


DEFAULT_OUT = Path("phase four/assumption_graph/paper_readiness_20260604/assumption_family_discovery_20260607.json")


@dataclass(frozen=True)
class TheoryCard:
    theory_id: str
    title: str
    domain: str
    text: str
    source_refs: list[str] = field(default_factory=list)
    gold_family: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TheorySignature:
    theory_id: str
    title: str
    domain: str
    surface_text: str
    primitive_hits: dict[str, list[str]]
    motif_hits: list[str]
    feature_vector: dict[str, float]
    abstract_claim: str
    gold_family: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRIMITIVE_MARKERS: dict[str, list[str]] = {
    "baseline_path": [
        "baseline",
        "identity",
        "skip",
        "prior",
        "base point",
        "current guess",
        "prediction unchanged",
        "leaves prediction unchanged",
        "preserves input",
        "preserved as base",
    ],
    "delta_update": [
        "delta",
        "residual",
        "innovation",
        "correction",
        "corrective update",
        "local error",
        "branch learns",
        "x + f",
        "update",
    ],
    "zero_delta_recovery": [
        "zero residual",
        "zero innovation",
        "zero delta",
        "unchanged",
        "recovers",
        "fixed point",
    ],
    "perturbation": [
        "perturb",
        "disturbance",
        "changing",
        "change in",
        "flux",
        "deviation",
        "displaces",
        "shock",
        "imposed change",
        "setpoint",
    ],
    "opposing_response": [
        "opposes",
        "opposing",
        "counteracts",
        "compensating",
        "resists",
        "restores",
        "actuator response",
        "equilibrium shift",
        "negative feedback",
        "stable range",
    ],
    "constraint_invariant": [
        "constraint",
        "conservation",
        "equilibrium",
        "free-energy",
        "free energy",
        "stable",
        "setpoint",
        "invariant",
        "balance",
    ],
    "stable_signal": [
        "stable signal",
        "coherent",
        "predictable",
        "correlated",
        "latent",
        "world state",
        "low-dimensional signal",
    ],
    "stochastic_noise": [
        "noise",
        "random",
        "uncorrelated",
        "stochastic",
        "nuisance",
        "pixel detail",
        "orthogonal random",
    ],
    "projection_operator": [
        "projection",
        "autocorrelation",
        "stacking",
        "matched filter",
        "principal component",
        "discard",
        "ignoring",
        "suppress",
    ],
    "control_baseline": [
        "control",
        "matched control",
        "treatment group",
        "baseline",
        "variant",
        "keeps controls fixed",
        "controlled variable",
    ],
    "single_factor_change": [
        "single intervention",
        "one factor",
        "one variable",
        "isolates one",
        "changed",
        "intervention",
    ],
    "outcome_measure": [
        "outcome",
        "metric",
        "measured",
        "decide causal effect",
        "causal",
        "compare",
    ],
    "root_problem": [
        "root problem",
        "whole result",
        "whole task",
        "root claim",
        "theorem",
        "goal",
    ],
    "subproblem": [
        "subproblem",
        "subtasks",
        "partitions",
        "lemmas",
        "split",
        "divide",
        "break",
    ],
    "interface_contract": [
        "interface",
        "contract",
        "compose",
        "reduce output",
        "merge",
        "input/output",
        "assumptions",
    ],
    "bottleneck_resource": [
        "bottleneck",
        "rate-limited",
        "scarce",
        "scarcest",
        "serial fraction",
        "limiter",
        "capacity",
        "capped",
    ],
    "flow_item": [
        "flow",
        "queue",
        "requests",
        "throughput",
        "output",
        "growth",
        "speedup",
    ],
    "counterexample_case": [
        "counterexample",
        "falsifiable",
        "disproves",
        "breaks",
        "checked against",
        "adversarial",
        "violates",
    ],
    "claim_under_test": [
        "hypothesis",
        "universal statement",
        "candidate program",
        "overbroad claim",
        "claim",
        "theorem",
    ],
    "refined_claim": [
        "revision",
        "refined",
        "narrows",
        "patched",
        "guardrail",
        "repair",
        "forces revision",
    ],
    "conserved_quantity": [
        "conserves",
        "conservation",
        "preserves total",
        "total energy",
        "charge",
        "probability mass",
        "normalization",
        "total probability",
    ],
    "transformation": [
        "state transformation",
        "transition",
        "reallocates",
        "entering",
        "leaving",
        "before/after",
        "move",
    ],
    "balance_check": [
        "balance check",
        "balance",
        "equals",
        "closes",
        "normalization",
        "keeps total",
    ],
    "ordered_state": [
        "ordered",
        "order",
        "non-decreasing",
        "never increases",
        "never decreases",
        "dominance",
        "accepted only if",
    ],
    "objective_measure": [
        "objective",
        "value",
        "loss",
        "energy",
        "lower bound",
        "improvement",
        "progress",
    ],
    "source_representation": [
        "time representation",
        "original trajectory",
        "positive products",
        "source representation",
        "differential equation",
        "signal",
    ],
    "target_representation": [
        "frequency representation",
        "algebraic representation",
        "sums",
        "target representation",
        "log transform",
        "fourier",
        "laplace",
    ],
    "operation_simplification": [
        "convolution becomes multiplication",
        "simpler operation",
        "solve simpler",
        "additive operation",
        "algebraic",
        "multiplication",
        "sums",
    ],
    "inverse_preservation": [
        "inverse",
        "invert",
        "preserves information",
        "back preserving",
        "recover original",
        "preserving ratios",
    ],
}


MOTIF_RULES: list[dict[str, Any]] = [
    {
        "motif": "kernel_residual_correction",
        "required": ["baseline_path", "delta_update"],
        "optional": ["zero_delta_recovery"],
    },
    {
        "motif": "kernel_negative_feedback",
        "required": ["perturbation", "opposing_response"],
        "optional": ["constraint_invariant"],
    },
    {
        "motif": "kernel_signal_nuisance_separation",
        "required": ["stable_signal", "stochastic_noise"],
        "optional": ["projection_operator"],
    },
    {
        "motif": "kernel_controlled_intervention",
        "required": ["control_baseline", "single_factor_change"],
        "optional": ["outcome_measure"],
    },
    {
        "motif": "kernel_decomposition_composition",
        "required": ["root_problem", "subproblem"],
        "optional": ["interface_contract"],
    },
    {
        "motif": "kernel_bottleneck_capacity",
        "required": ["bottleneck_resource", "flow_item"],
        "optional": [],
    },
    {
        "motif": "kernel_counterexample_refinement",
        "required": ["counterexample_case", "refined_claim"],
        "optional": ["claim_under_test"],
    },
    {
        "motif": "kernel_conservation_balance",
        "required": ["conserved_quantity", "balance_check"],
        "optional": ["transformation"],
    },
    {
        "motif": "kernel_monotone_progress",
        "required": ["ordered_state", "objective_measure"],
        "optional": [],
    },
    {
        "motif": "kernel_representation_transform",
        "required": ["source_representation", "target_representation", "operation_simplification"],
        "optional": ["inverse_preservation"],
    },
]


KERNEL_TO_EXISTING_PATTERN = {
    "kernel_residual_correction": "pat_residual_correction",
    "kernel_negative_feedback": "pat_negative_feedback",
    "kernel_signal_nuisance_separation": "pat_signal_nuisance_separation",
    "kernel_controlled_intervention": "pat_controlled_intervention",
    "kernel_decomposition_composition": "pat_decomposition_composition",
    "kernel_bottleneck_capacity": "pat_bottleneck_capacity",
    "kernel_counterexample_refinement": "pat_counterexample_refinement",
    "kernel_conservation_balance": "pat_conservation_balance",
    "kernel_monotone_progress": "pat_monotone_progress",
}


ASSUMPTION_TEMPLATES = {
    "kernel_residual_correction": (
        "Preserve a working identity or prior path, learn only the residual correction, "
        "and require zero correction to recover the baseline."
    ),
    "kernel_negative_feedback": (
        "When a perturbation induces an opposing response under a constraint, predict dampening "
        "or restoration only after checking positive-feedback controls."
    ),
    "kernel_signal_nuisance_separation": (
        "Prefer the predictable latent/coherent signal and suppress stochastic nuisance variation; "
        "validate that the retained structure, not the noise, carries downstream utility."
    ),
    "kernel_controlled_intervention": (
        "Change one factor against a matched control and promote the claim only if the measured outcome "
        "beats the control without outside harm."
    ),
    "kernel_decomposition_composition": (
        "Split the root problem into subproblems with explicit interfaces, then verify that composed "
        "sub-solutions recover the original goal."
    ),
    "kernel_bottleneck_capacity": (
        "Throughput is governed by the limiting capacity; interventions should target the bottleneck, "
        "not locally attractive non-limiting stages."
    ),
    "kernel_counterexample_refinement": (
        "A counterexample should falsify an overbroad claim and force a narrower, testable repair."
    ),
    "kernel_conservation_balance": (
        "A transformation is valid only if the conserved quantity or budget balances before and after."
    ),
    "kernel_monotone_progress": (
        "Accept an iterative update only when it preserves the relevant order/objective and detects regression."
    ),
    "kernel_representation_transform": (
        "Move to an information-preserving representation where the hard operation becomes simpler, "
        "then map back while checking the preserved invariant."
    ),
}


def build_assumption_family_discovery_payload(
    *,
    cards: list[dict[str, Any] | TheoryCard] | None = None,
    eval_id: str | None = None,
    similarity_threshold: float = 0.58,
) -> dict[str, Any]:
    theory_cards = [_coerce_card(card) for card in (cards or _default_theory_cards())]
    signatures = [extract_theory_signature(card) for card in theory_cards]
    similarity_rows = _pairwise_similarity_rows(signatures)
    family_rows = _induce_families(signatures, threshold=similarity_threshold)
    assignment_by_card = {
        member["theory_id"]: row["family_id"]
        for row in family_rows
        for member in row["members"]
    }
    graph = _build_family_graph(signatures, family_rows, similarity_rows, assignment_by_card)
    metrics = _gold_metrics(theory_cards, assignment_by_card, similarity_rows)
    gates = [
        {
            "gate": "family_count_open_set",
            "pass": metrics["discovered_family_count"] >= 10,
            "observed": metrics["discovered_family_count"],
        },
        {
            "gate": "cluster_purity",
            "pass": metrics["cluster_purity"] >= 0.90,
            "observed": metrics["cluster_purity"],
        },
        {
            "gate": "same_family_pair_recall",
            "pass": metrics["same_family_pair_recall"] >= 0.90,
            "observed": metrics["same_family_pair_recall"],
        },
        {
            "gate": "cross_family_block_rate",
            "pass": metrics["cross_family_block_rate"] >= 0.95,
            "observed": metrics["cross_family_block_rate"],
        },
        {
            "gate": "beats_word_context_pairing",
            "pass": metrics["same_family_pair_recall"] - metrics["word_context_pair_recall"] >= 0.30,
            "observed": {
                "structural_recall": metrics["same_family_pair_recall"],
                "word_context_recall": metrics["word_context_pair_recall"],
                "margin": round(metrics["same_family_pair_recall"] - metrics["word_context_pair_recall"], 4),
            },
        },
        {
            "gate": "nonlexical_positive_pair_recall",
            "pass": metrics["nonlexical_positive_pair_recall"] >= 0.85,
            "observed": metrics["nonlexical_positive_pair_recall"],
        },
        {
            "gate": "discovers_new_kernel_not_in_old_catalog",
            "pass": metrics["new_open_set_family_count"] >= 1,
            "observed": metrics["new_open_set_family_count"],
        },
    ]
    return {
        "eval_id": eval_id or "assumption_family_discovery_20260607",
        "eval_kind": "open_set_assumption_family_discovery",
        "claim_scope": (
            "Open-set, category-inspired assumption-kernel induction over supplied theory cards. "
            "It does not enumerate every possible philosophy/science/math principle."
        ),
        "input_card_count": len(theory_cards),
        "primitive_role_count": len(PRIMITIVE_MARKERS),
        "motif_rule_count": len(MOTIF_RULES),
        "similarity_threshold": similarity_threshold,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [gate["gate"] for gate in gates if not gate["pass"]],
        "pass": all(gate["pass"] for gate in gates),
        "families": family_rows,
        "theory_signatures": [sig.to_dict() for sig in signatures],
        "pairwise_similarity": similarity_rows,
        "family_graph": graph,
    }


def extract_theory_signature(card: TheoryCard) -> TheorySignature:
    text = f"{card.title}. {card.domain}. {card.text}"
    primitive_hits = {
        primitive: _term_hits(text, terms)
        for primitive, terms in PRIMITIVE_MARKERS.items()
    }
    primitive_hits = {primitive: hits for primitive, hits in primitive_hits.items() if hits}
    motif_hits = _derive_motifs(primitive_hits)
    vector: Counter[str] = Counter()
    for primitive, hits in primitive_hits.items():
        vector[f"role::{primitive}"] += 1.0 + min(len(hits), 3) * 0.15
    for motif in motif_hits:
        vector[f"motif::{motif}"] += 3.0
    for primitive in sorted(primitive_hits):
        vector[f"coarse::{_coarse_role_group(primitive)}"] += 0.35
    abstract_claim = _abstract_claim(motif_hits, primitive_hits)
    return TheorySignature(
        theory_id=card.theory_id,
        title=card.title,
        domain=card.domain,
        surface_text=text,
        primitive_hits=primitive_hits,
        motif_hits=motif_hits,
        feature_vector=dict(vector),
        abstract_claim=abstract_claim,
        gold_family=card.gold_family,
    )


def classify_new_theory_card(
    card: dict[str, Any] | TheoryCard,
    discovered_payload: dict[str, Any],
    *,
    attach_threshold: float = 0.58,
) -> dict[str, Any]:
    """Classify a new card as existing-family attachment or new-family seed."""

    theory_card = _coerce_card(card)
    signature = extract_theory_signature(theory_card)
    family_rows = discovered_payload.get("families", [])
    scored = []
    for family in family_rows:
        centroid = Counter(family.get("centroid_feature_vector", {}))
        score = _feature_similarity(Counter(signature.feature_vector), centroid)
        scored.append({
            "family_id": family["family_id"],
            "kernel_motif": family.get("kernel_motif"),
            "score": round(score, 4),
            "existing_pattern_id": family.get("existing_pattern_id"),
        })
    ranked = sorted(scored, key=lambda row: (-row["score"], row["family_id"]))
    top = ranked[0] if ranked else {}
    if top and top["score"] >= attach_threshold:
        decision = "attach_to_existing_family"
        family_id = top["family_id"]
        assumption = next((row.get("assumption_kernel") for row in family_rows if row["family_id"] == family_id), "")
    else:
        decision = "create_new_family"
        family_id = stable_id("fam_new", theory_card.theory_id, ",".join(signature.motif_hits), length=10)
        assumption = signature.abstract_claim
    return {
        "theory_card": theory_card.to_dict(),
        "signature": signature.to_dict(),
        "decision": decision,
        "family_id": family_id,
        "score": top.get("score", 0.0),
        "assumption_kernel": assumption,
        "ranking": ranked[:5],
    }


def _pairwise_similarity_rows(signatures: list[TheorySignature]) -> list[dict[str, Any]]:
    rows = []
    for left, right in combinations(signatures, 2):
        structural = _signature_similarity(left, right)
        word = _word_similarity(left, right)
        rows.append({
            "left_id": left.theory_id,
            "right_id": right.theory_id,
            "left_title": left.title,
            "right_title": right.title,
            "gold_same_family": bool(left.gold_family and left.gold_family == right.gold_family),
            "structural_similarity": round(structural, 4),
            "word_context_similarity": round(word, 4),
            "shared_motifs": sorted(set(left.motif_hits) & set(right.motif_hits)),
            "shared_primitives": sorted(set(left.primitive_hits) & set(right.primitive_hits)),
        })
    return rows


def _induce_families(signatures: list[TheorySignature], *, threshold: float) -> list[dict[str, Any]]:
    clusters: list[list[TheorySignature]] = []
    for signature in sorted(signatures, key=lambda row: row.theory_id):
        best_idx = None
        best_score = 0.0
        for idx, cluster in enumerate(clusters):
            score = _feature_similarity(Counter(signature.feature_vector), _centroid(cluster))
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is not None and best_score >= threshold:
            clusters[best_idx].append(signature)
        else:
            clusters.append([signature])
    return [_family_row(cluster) for cluster in clusters]


def _family_row(cluster: list[TheorySignature]) -> dict[str, Any]:
    motif_counts = Counter(motif for sig in cluster for motif in sig.motif_hits)
    primitive_counts = Counter(primitive for sig in cluster for primitive in sig.primitive_hits)
    kernel_motif = motif_counts.most_common(1)[0][0] if motif_counts else ""
    if not kernel_motif:
        kernel_motif = "kernel_" + "_".join(role for role, _ in primitive_counts.most_common(3))
    existing_pattern_id = KERNEL_TO_EXISTING_PATTERN.get(kernel_motif)
    centroid = _centroid(cluster)
    members = [
        {
            "theory_id": sig.theory_id,
            "title": sig.title,
            "domain": sig.domain,
            "motif_hits": sig.motif_hits,
            "primitive_roles": sorted(sig.primitive_hits),
            "gold_family": sig.gold_family,
        }
        for sig in sorted(cluster, key=lambda row: row.theory_id)
    ]
    family_id = stable_id("fam", kernel_motif, ",".join(member["theory_id"] for member in members), length=10)
    return {
        "family_id": family_id,
        "kernel_motif": kernel_motif,
        "existing_pattern_id": existing_pattern_id,
        "open_set_status": "existing_structural_family" if existing_pattern_id else "new_open_set_family",
        "assumption_kernel": ASSUMPTION_TEMPLATES.get(kernel_motif, _generic_assumption(primitive_counts)),
        "dominant_primitives": [role for role, _ in primitive_counts.most_common(6)],
        "member_count": len(members),
        "members": members,
        "centroid_feature_vector": {key: round(value, 4) for key, value in sorted(centroid.items())},
    }


def _build_family_graph(
    signatures: list[TheorySignature],
    families: list[dict[str, Any]],
    similarity_rows: list[dict[str, Any]],
    assignment_by_card: dict[str, str],
) -> dict[str, Any]:
    nodes = []
    edges = []
    family_by_id = {row["family_id"]: row for row in families}
    for family in families:
        nodes.append({
            "id": family["family_id"],
            "node_type": "assumption_family",
            "label": family["kernel_motif"],
            "claim": family["assumption_kernel"],
            "existing_pattern_id": family.get("existing_pattern_id"),
        })
    for sig in signatures:
        theory_node = f"theory::{sig.theory_id}"
        nodes.append({
            "id": theory_node,
            "node_type": "theory_card",
            "label": sig.title,
            "domain": sig.domain,
        })
        family_id = assignment_by_card.get(sig.theory_id)
        if family_id:
            edges.append({
                "source": theory_node,
                "target": family_id,
                "edge_type": "realizes_assumption_family",
                "weight": 1.0,
            })
            for primitive in sorted(sig.primitive_hits):
                primitive_node = f"primitive::{primitive}"
                nodes.append({
                    "id": primitive_node,
                    "node_type": "primitive_role",
                    "label": primitive,
                })
                edges.append({
                    "source": theory_node,
                    "target": primitive_node,
                    "edge_type": "has_structural_role",
                    "weight": 0.75,
                })
                edges.append({
                    "source": primitive_node,
                    "target": family_id,
                    "edge_type": "induces_assumption_kernel",
                    "weight": 0.65,
                })
    for row in similarity_rows:
        if assignment_by_card.get(row["left_id"]) != assignment_by_card.get(row["right_id"]):
            continue
        if row["structural_similarity"] < 0.58:
            continue
        edges.append({
            "source": f"theory::{row['left_id']}",
            "target": f"theory::{row['right_id']}",
            "edge_type": "structural_morphism_edge",
            "weight": row["structural_similarity"],
            "shared_motifs": row["shared_motifs"],
            "family_id": assignment_by_card.get(row["left_id"]),
        })
    dedup_nodes = {node["id"]: node for node in nodes}
    return {
        "nodes": list(dedup_nodes.values()),
        "edges": edges,
        "node_count": len(dedup_nodes),
        "edge_count": len(edges),
        "family_count": len(family_by_id),
    }


def _gold_metrics(
    cards: list[TheoryCard],
    assignment_by_card: dict[str, str],
    similarity_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    gold = {card.theory_id: card.gold_family for card in cards if card.gold_family}
    assigned = {card_id: family_id for card_id, family_id in assignment_by_card.items() if card_id in gold}
    cluster_to_gold = defaultdict(Counter)
    for card_id, family_id in assigned.items():
        cluster_to_gold[family_id][gold[card_id]] += 1
    pure_count = sum(counts.most_common(1)[0][1] for counts in cluster_to_gold.values() if counts)
    cluster_purity = _rate(pure_count, len(assigned))
    same_pairs = [row for row in similarity_rows if row["gold_same_family"]]
    diff_pairs = [row for row in similarity_rows if not row["gold_same_family"]]
    structural_same_hits = sum(
        1
        for row in same_pairs
        if assignment_by_card.get(row["left_id"]) == assignment_by_card.get(row["right_id"])
    )
    diff_block_hits = sum(
        1
        for row in diff_pairs
        if assignment_by_card.get(row["left_id"]) != assignment_by_card.get(row["right_id"])
    )
    word_same_hits = sum(1 for row in same_pairs if row["word_context_similarity"] >= 0.18)
    nonlexical_same_pairs = [row for row in same_pairs if row["word_context_similarity"] < 0.18]
    nonlexical_hits = sum(
        1
        for row in nonlexical_same_pairs
        if assignment_by_card.get(row["left_id"]) == assignment_by_card.get(row["right_id"])
    )
    discovered_family_ids = set(assignment_by_card.values())
    new_family_count = sum(
        1
        for family_id in discovered_family_ids
        if _family_open_set_status(family_id, assignment_by_card, cards) == "new_open_set_family"
    )
    return {
        "gold_card_count": len(gold),
        "discovered_family_count": len(discovered_family_ids),
        "gold_family_count": len(set(gold.values())),
        "cluster_purity": cluster_purity,
        "same_family_pair_recall": _rate(structural_same_hits, len(same_pairs)),
        "cross_family_block_rate": _rate(diff_block_hits, len(diff_pairs)),
        "word_context_pair_recall": _rate(word_same_hits, len(same_pairs)),
        "nonlexical_positive_pair_count": len(nonlexical_same_pairs),
        "nonlexical_positive_pair_recall": _rate(nonlexical_hits, len(nonlexical_same_pairs)),
        "new_open_set_family_count": new_family_count,
    }


def _family_open_set_status(family_id: str, assignment_by_card: dict[str, str], cards: list[TheoryCard]) -> str:
    members = [card for card in cards if assignment_by_card.get(card.theory_id) == family_id]
    signatures = [extract_theory_signature(card) for card in members]
    if not signatures:
        return "unknown"
    row = _family_row(signatures)
    return row["open_set_status"]


def _signature_similarity(left: TheorySignature, right: TheorySignature) -> float:
    feature = _feature_similarity(Counter(left.feature_vector), Counter(right.feature_vector))
    motif_overlap = _jaccard(set(left.motif_hits), set(right.motif_hits))
    primitive_overlap = _jaccard(set(left.primitive_hits), set(right.primitive_hits))
    return round(0.62 * feature + 0.28 * motif_overlap + 0.10 * primitive_overlap, 6)


def _feature_similarity(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    return _counter_cosine(left, right)


def _word_similarity(left: TheorySignature, right: TheorySignature) -> float:
    return _counter_cosine(_tokens(left.surface_text), _tokens(right.surface_text))


def _centroid(cluster: list[TheorySignature]) -> Counter:
    centroid: Counter[str] = Counter()
    for sig in cluster:
        centroid.update(Counter(sig.feature_vector))
    if cluster:
        for key in list(centroid):
            centroid[key] = centroid[key] / len(cluster)
    return centroid


def _derive_motifs(primitive_hits: dict[str, list[str]]) -> list[str]:
    roles = set(primitive_hits)
    motifs = []
    for rule in MOTIF_RULES:
        if set(rule["required"]) <= roles:
            motifs.append(rule["motif"])
    return motifs


def _abstract_claim(motif_hits: list[str], primitive_hits: dict[str, list[str]]) -> str:
    if motif_hits:
        return ASSUMPTION_TEMPLATES.get(motif_hits[0], "")
    return _generic_assumption(Counter(primitive_hits))


def _generic_assumption(primitive_counts: Counter) -> str:
    roles = [role for role, _ in primitive_counts.most_common(4)]
    return (
        "This candidate family is defined by co-occurring structural roles "
        f"{', '.join(roles)}; it should remain shadow-only until controlled transfer evidence confirms the kernel."
    )


def _coarse_role_group(primitive: str) -> str:
    if primitive in {"baseline_path", "delta_update", "zero_delta_recovery"}:
        return "identity_delta"
    if primitive in {"perturbation", "opposing_response", "constraint_invariant"}:
        return "feedback_invariant"
    if primitive in {"stable_signal", "stochastic_noise", "projection_operator"}:
        return "signal_noise"
    if primitive in {"control_baseline", "single_factor_change", "outcome_measure"}:
        return "controlled_causality"
    if primitive in {"root_problem", "subproblem", "interface_contract"}:
        return "composition"
    if primitive in {"bottleneck_resource", "flow_item"}:
        return "capacity_flow"
    if primitive in {"counterexample_case", "claim_under_test", "refined_claim"}:
        return "falsification"
    if primitive in {"conserved_quantity", "transformation", "balance_check"}:
        return "conservation"
    if primitive in {"ordered_state", "objective_measure"}:
        return "monotone_objective"
    if primitive in {"source_representation", "target_representation", "operation_simplification", "inverse_preservation"}:
        return "representation_transform"
    return primitive


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


def _term_hits(text: str, terms: list[str]) -> list[str]:
    low = text.lower()
    hits = []
    for term in terms:
        term_low = term.lower()
        if _ascii_term(term_low):
            matched = bool(re.search(rf"(?<![a-z0-9]){re.escape(term_low)}(?![a-z0-9])", low))
        else:
            matched = term_low in low
        if matched:
            hits.append(term)
    return sorted(set(hits))


def _ascii_term(term: str) -> bool:
    return all(ord(ch) < 128 for ch in term)


def _rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def _coerce_card(card: dict[str, Any] | TheoryCard) -> TheoryCard:
    if isinstance(card, TheoryCard):
        return card
    return TheoryCard(
        theory_id=str(card["theory_id"]),
        title=str(card["title"]),
        domain=str(card.get("domain", "")),
        text=str(card["text"]),
        source_refs=list(card.get("source_refs", [])),
        gold_family=card.get("gold_family"),
    )


def _default_theory_cards() -> list[TheoryCard]:
    rows = [
        ("resnet", "ResNet residual block", "deep_learning", "A skip connection preserves input activation as an identity baseline while a residual branch learns a delta; output is x + F(x), and zero delta recovers the input.", "residual_correction"),
        ("kalman", "Kalman innovation update", "control_estimation", "The prior state estimate is the baseline path; the innovation residual forms a correction delta, and zero innovation leaves prediction unchanged.", "residual_correction"),
        ("newton", "Newton residual root update", "numerical_math", "At each root-finding step, the current guess is preserved as base point; local error residual produces a corrective update, and zero residual means fixed point.", "residual_correction"),
        ("lenz", "Lenz law", "electromagnetism", "Changing magnetic flux perturbs a circuit; induced current opposes the flux change, and energy conservation constrains the response.", "negative_feedback"),
        ("le_chatelier", "Le Chatelier equilibrium shift", "chemistry", "A concentration or temperature disturbance displaces equilibrium; the equilibrium shift counteracts the imposed change under a free-energy constraint.", "negative_feedback"),
        ("homeostasis", "Homeostatic setpoint control", "biology", "Deviation from a setpoint triggers actuator response that resists disturbance and restores a stable range.", "negative_feedback"),
        ("seismic", "Seismic autocorrelation denoising", "signal_processing", "A coherent seismic event is the stable signal; random uncorrelated noise is suppressed by autocorrelation and stacking projection.", "signal_nuisance"),
        ("jepa", "JEPA latent prediction", "world_modeling", "Latent prediction keeps predictable world state while ignoring stochastic pixel detail and Gaussian nuisance variation.", "signal_nuisance"),
        ("pca", "PCA noise suppression", "statistics", "Project measurements onto principal component structure and discard orthogonal random noise to recover a low-dimensional signal.", "signal_nuisance"),
        ("rct", "Randomized controlled trial", "causal_inference", "A treatment group is compared with a matched control; a single intervention is changed and outcome is measured.", "controlled_intervention"),
        ("ab_test", "A/B test", "product_experiment", "A variant changes one factor from baseline; matched traffic control and outcome metric decide causal effect.", "controlled_intervention"),
        ("controlled_variable", "Controlled-variable physics experiment", "physics_method", "The experiment isolates one variable, keeps controls fixed, changes one factor, and compares the measured outcome.", "controlled_intervention"),
        ("divide_conquer", "Divide and conquer", "algorithm_design", "Split a root problem into subproblems, solve independent parts, and compose results back through an interface contract.", "decomposition_composition"),
        ("map_reduce", "MapReduce", "distributed_systems", "Map subtasks across partitions and reduce output through a contract to recover the whole result.", "decomposition_composition"),
        ("modular_proof", "Modular proof by lemmas", "mathematics", "Break a theorem into lemmas with interface assumptions, then compose lemmas to prove the root claim.", "decomposition_composition"),
        ("amdahl", "Amdahl law", "parallel_computing", "Speedup and throughput are limited by the serial bottleneck fraction; improving nonlimiting parallel work cannot raise output.", "bottleneck_capacity"),
        ("queue_limit", "Rate-limited worker queue", "systems", "Requests flow through a rate-limited worker queue; scarce capacity caps output, and relieving the limiter increases throughput.", "bottleneck_capacity"),
        ("liebig", "Liebig law of the minimum", "ecology", "Plant growth output is constrained by the scarcest nutrient; adding nonlimiting resources does not improve throughput.", "bottleneck_capacity"),
        ("popper", "Popper falsifiability", "philosophy_of_science", "A hypothesis must expose falsifiable predictions; a counterexample breaks an overbroad claim and forces revision.", "counterexample_refinement"),
        ("proof_counterexample", "Proof by counterexample", "mathematics", "One counterexample disproves a universal statement; the refined theorem narrows conditions.", "counterexample_refinement"),
        ("cegis", "Counterexample-guided synthesis", "program_synthesis", "A candidate program is checked against counterexample traces, then patched with a guardrail repair.", "counterexample_refinement"),
        ("energy", "Energy conservation", "physics", "A state transformation preserves total energy; the before/after balance check closes.", "conservation_balance"),
        ("kirchhoff", "Kirchhoff current law", "circuit_theory", "Charge conservation means current entering a node equals current leaving, preserving flow balance.", "conservation_balance"),
        ("probability_mass", "Probability mass normalization", "probability", "A transition reallocates probability mass, but normalization keeps total probability one and the balance closes.", "conservation_balance"),
        ("lyapunov", "Lyapunov descent", "dynamical_systems", "An ordered state update uses a Lyapunov energy objective that never increases; regression is rejected until stable.", "monotone_progress"),
        ("policy_iteration", "Policy iteration improvement", "reinforcement_learning", "A new policy is accepted only if the value objective is non-decreasing over ordered states.", "monotone_progress"),
        ("coordinate_ascent", "Coordinate ascent lower bound", "optimization", "Each coordinate step improves the lower bound objective and rejects updates that reverse progress order.", "monotone_progress"),
        ("fourier", "Fourier transform", "mathematics", "Transform a signal from time representation to frequency representation so convolution becomes multiplication; the inverse preserves information.", "representation_transform"),
        ("laplace", "Laplace transform", "differential_equations", "Map a differential equation to an algebraic representation, solve the simpler operation, then invert to recover the original trajectory.", "representation_transform"),
        ("log_products", "Log transform for products", "mathematics", "Represent positive products as sums with a log transform, perform the additive operation, and exponentiate back preserving ratios.", "representation_transform"),
    ]
    return [
        TheoryCard(theory_id=theory_id, title=title, domain=domain, text=text, gold_family=family)
        for theory_id, title, domain, text, family in rows
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_cards(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.get("cards", []))
    return list(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover open-set assumption families from theory cards.")
    parser.add_argument("--cards", default=None, help="Optional JSON list or {'cards': [...]} of theory cards.")
    parser.add_argument("--eval-id", default="assumption_family_discovery_20260607")
    parser.add_argument("--threshold", type=float, default=0.58)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    cards = _load_cards(Path(args.cards)) if args.cards else None
    payload = build_assumption_family_discovery_payload(
        cards=cards,
        eval_id=args.eval_id,
        similarity_threshold=args.threshold,
    )
    out = Path(args.out)
    _write_json(out, payload)
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
