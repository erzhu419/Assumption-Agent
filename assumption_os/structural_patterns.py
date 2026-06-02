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
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .graph_memory import JsonlGraphStore, tokenize
from .schema import AssumptionNode, AssumptionType, HypothesisKind, stable_id


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
}


NEGATION_PATTERNS = [
    r"\bno\s+(?:baseline|fallback|identity|control|residual|rollback|invariant)\b",
    r"\bno\s+(?:roles|morphisms|invariants|transfer prediction|predictable signal)\b",
    r"\bwithout\s+(?:baseline|fallback|identity|control|residual|rollback|invariant)\b",
    r"\bnot\s+(?:preserve|controlled|opposing|predictable)\b",
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


def propose_structural_morphism(query_diagram: str | dict | StructuralSignature, pattern: dict) -> dict:
    signature = query_diagram if isinstance(query_diagram, StructuralSignature) else extract_structural_signature(query_diagram)
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
        "preserved_invariants": score.preserved_invariants,
        "broken_or_uncertain_invariants": score.broken_or_uncertain_invariants,
        "negative_control_hits": score.negative_control_hits,
        "transfer_predictions": pattern.get("transfer_predictions", []),
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
    return {
        "decision": decision,
        "blocks_policy_update": decision in {
            "block_negative_control",
            "repair_under_specified",
            "repair_missing_transfer_prediction",
        },
        "reason": score.get("reason"),
        "score": score,
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
        lines.append(f"\n- {app['pattern_name']} ({app['pattern_id']}, score={app['score']:.2f}, gate={app['decision']})")
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
            "For high-risk system changes, structural context should prefer one-module replacement with rollback over whole-system rewrite.",
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
]


def _pattern_node(spec: dict) -> AssumptionNode:
    pattern_id = spec["pattern_id"]
    return AssumptionNode(
        id=f"struct_{pattern_id}",
        type=AssumptionType.ALIGNMENT,
        kind=HypothesisKind.FORMAL_MAPPING,
        claim=spec["claim"],
        formal_form={"formal_kind": STRUCTURAL_PATTERN_KIND, **spec},
        tags=["structural_pattern", "structural_morphism", pattern_id, *spec.get("trigger_terms", [])[:6]],
        context_conditions=["structural transfer", "cross-domain analogy", "recursive assumption validation"],
        predicted_effects=spec.get("transfer_predictions", []),
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
    return pattern


def _source_text(source: str | dict) -> str:
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
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
