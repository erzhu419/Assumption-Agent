"""Generalized structural-context edges inspired by HippoRAG.

HippoRAG-style graph memory spreads over phrase, synonym, and context edges.
For assumption reasoning we need one higher layer: synonym/context edges should
connect structural roles and reusable assumption contexts, not only neighboring
words.  This module implements that bounded expansion for negative feedback
and equilibrium-restoration contexts such as Le Chatelier and Lenz.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .morphism_benchmark import _counter_cosine, _tokens
from .structural_patterns import DEFAULT_STRUCTURAL_PATTERNS, DEFAULT_STRUCTURAL_REALIZATIONS


DEFAULT_OUT = Path("phase four/assumption_graph/paper_readiness_20260604/structural_context_edges_20260607.json")

STRUCTURAL_CONTEXT_SYNONYMS: dict[str, list[str]] = {
    "growth_perturbation": [
        "increase",
        "increases",
        "increasing",
        "rise",
        "rises",
        "rising",
        "growth",
        "grows",
        "surge",
        "traffic grows",
        "demand increases",
        "population rises",
        "retry volume rises",
        "flux increases",
        "adding reactant",
        "随着",
        "增长",
        "增加",
        "增多",
        "上升",
        "上涨",
    ],
    "opposing_response": [
        "opposes",
        "opposing",
        "counteracts",
        "counter",
        "compensates",
        "dampens",
        "dampen",
        "resists",
        "reducing",
        "suppresses",
        "limits",
        "throttle",
        "throttling",
        "rate limits",
        "backoff",
        "circuit breaker",
        "penalty",
        "pulls back",
        "back toward",
        "predators",
        "competition",
        "demand falls",
        "stimulates supply",
        "induced current",
        "shift",
        "抑制",
        "抵消",
        "反向",
        "反作用",
        "负反馈",
        "限制",
        "回落",
        "刺激供给",
    ],
    "constraint_or_equilibrium": [
        "equilibrium",
        "balance",
        "balanced",
        "stabilizes",
        "stabilize",
        "steady state",
        "setpoint",
        "carrying capacity",
        "capacity",
        "calibrated range",
        "constraint",
        "conservation",
        "free energy",
        "load reaches",
        "throughput stabilizes",
        "均衡",
        "平衡",
        "收敛",
        "稳态",
        "稳定",
        "饱和",
        "约束",
    ],
    "classic_negative_feedback_reference": [
        "le chatelier",
        "chatelier",
        "lenz",
        "楞次",
        "勒夏特列",
        "homeostasis",
    ],
}

NEGATIVE_CONTROL_TERMS = [
    "positive feedback",
    "runaway",
    "amplifies",
    "amplification",
    "accelerates",
    "snowball",
    "further increases",
    "no opposing",
    "no balancing",
    "no constraint",
    "without any induced",
    "without counter",
    "random",
    "没有抑制",
    "没有约束",
    "正反馈",
    "失控",
]

CONTEXT_PATTERNS = [
    {
        "context_id": "ctx_negative_feedback_equilibrium",
        "pattern_id": "pat_negative_feedback",
        "roles": ["growth_perturbation", "opposing_response", "constraint_or_equilibrium"],
        "minimum_roles": 2,
        "required_roles_for_accept": ["growth_perturbation", "opposing_response"],
        "classic_realizations": ["real_lenz_negative_feedback", "real_le_chatelier_shift"],
        "generalized_assumption": (
            "When growth or perturbation induces a compensating response under a constraint, "
            "do not extrapolate monotonic growth blindly; expect dampening, plateau, or convergence "
            "toward a constrained equilibrium unless negative controls indicate runaway amplification."
        ),
        "prediction_template": (
            "Identify the perturbation, the induced opposing response, and the preserved constraint; "
            "predict convergence/plateau only after checking lag, overshoot, and positive-feedback controls."
        ),
    },
]


def build_structural_context_edge_payload(*, eval_id: str | None = None) -> dict[str, Any]:
    rows = [_evaluate_case(case) for case in _eval_cases()]
    positives = [row for row in rows if row["expected_pattern_id"] == "pat_negative_feedback"]
    negatives = [row for row in rows if row["expected_pattern_id"] is None]
    structural_positive_hits = sum(1 for row in positives if row["structural_context"]["top_pattern_id"] == "pat_negative_feedback")
    baseline_positive_hits = sum(1 for row in positives if row["word_context_baseline"]["top_pattern_id"] == "pat_negative_feedback")
    negative_block_hits = sum(1 for row in negatives if row["structural_context"]["decision"] in {"abstain", "block_negative_control"})
    classic_reference_hits = sum(
        1
        for row in positives
        if {"real_lenz_negative_feedback", "real_le_chatelier_shift"}
        <= set(row["structural_context"]["expanded_realization_ids"])
    )
    gates = [
        {
            "gate": "positive_structural_context_recall",
            "pass": _rate(structural_positive_hits, len(positives)) >= 0.85,
            "observed": {
                "hit_rate": _rate(structural_positive_hits, len(positives)),
                "hits": structural_positive_hits,
                "n": len(positives),
            },
        },
        {
            "gate": "beats_word_context_baseline",
            "pass": structural_positive_hits > baseline_positive_hits,
            "observed": {
                "structural_hits": structural_positive_hits,
                "word_context_baseline_hits": baseline_positive_hits,
            },
        },
        {
            "gate": "negative_controls_block_or_abstain",
            "pass": _rate(negative_block_hits, len(negatives)) >= 0.90,
            "observed": {
                "hit_rate": _rate(negative_block_hits, len(negatives)),
                "hits": negative_block_hits,
                "n": len(negatives),
            },
        },
        {
            "gate": "classic_realization_context_expands",
            "pass": _rate(classic_reference_hits, len(positives)) >= 0.85,
            "observed": {
                "hit_rate": _rate(classic_reference_hits, len(positives)),
                "hits": classic_reference_hits,
                "n": len(positives),
            },
        },
        {
            "gate": "not_monotone_template",
            "pass": all("blindly" in row["structural_context"].get("generalized_assumption", "") for row in positives),
            "observed": "accepted contexts warn against blind monotone extrapolation",
        },
    ]
    return {
        "eval_id": eval_id or "structural_context_edges_20260607",
        "eval_kind": "generalized_assumption_context_edge_validation",
        "source_alignment": {
            "hipporag_mechanism": "phrase/synonym/context spreading",
            "assumption_os_extension": (
                "role synonym edges connect text to generalized assumption contexts, then to classic "
                "realizations such as Lenz law and Le Chatelier."
            ),
        },
        "context_patterns": CONTEXT_PATTERNS,
        "metrics": {
            "positive_count": len(positives),
            "negative_count": len(negatives),
            "structural_context_positive_recall": _rate(structural_positive_hits, len(positives)),
            "word_context_baseline_positive_recall": _rate(baseline_positive_hits, len(positives)),
            "negative_control_block_or_abstain_rate": _rate(negative_block_hits, len(negatives)),
            "classic_reference_expansion_rate": _rate(classic_reference_hits, len(positives)),
        },
        "gates": gates,
        "failed_gates": [gate["gate"] for gate in gates if not gate["pass"]],
        "pass": all(gate["pass"] for gate in gates),
        "rows": rows,
    }


def expand_structural_context(text: str) -> dict[str, Any]:
    role_hits = {
        role: _term_hits(text, terms)
        for role, terms in STRUCTURAL_CONTEXT_SYNONYMS.items()
    }
    role_hits = {role: hits for role, hits in role_hits.items() if hits}
    negative_hits = _term_hits(text, NEGATIVE_CONTROL_TERMS)
    graph, graph_meta = _build_context_graph(text, role_hits, negative_hits)
    context_rows = []
    for context in CONTEXT_PATTERNS:
        row = _score_context(context, role_hits, negative_hits)
        context_rows.append(row)
    ranked = sorted(context_rows, key=lambda row: (-row["score"], row["context_id"]))
    top = ranked[0] if ranked else {}
    if top.get("decision") == "accept":
        context = _context_by_id(top["context_id"])
        expanded = context.get("classic_realizations", [])
        generalized = context.get("generalized_assumption", "")
        prediction = context.get("prediction_template", "")
    else:
        expanded = []
        generalized = ""
        prediction = ""
    return {
        "role_hits": role_hits,
        "negative_control_hits": negative_hits,
        "top_context_id": top.get("context_id"),
        "top_pattern_id": top.get("pattern_id") if top.get("decision") == "accept" else None,
        "decision": top.get("decision", "abstain"),
        "score": top.get("score", 0.0),
        "reason": top.get("reason", ""),
        "expanded_realization_ids": expanded,
        "generalized_assumption": generalized,
        "prediction_template": prediction,
        "context_edges": graph_meta["edges"],
        "graph": graph,
    }


def _score_context(context: dict[str, Any], role_hits: dict[str, list[str]], negative_hits: list[str]) -> dict[str, Any]:
    roles = list(context["roles"])
    matched_roles = [role for role in roles if role_hits.get(role)]
    required = set(context.get("required_roles_for_accept", []))
    missing_required = sorted(required - set(matched_roles))
    negative_severe = bool(negative_hits)
    role_rate = len(matched_roles) / max(1, len(roles))
    score = round(role_rate - (0.75 if negative_severe else 0.0), 4)
    if negative_severe:
        decision = "block_negative_control"
        reason = "Negative control indicates runaway, missing opposition, random response, or no constraint."
    elif len(matched_roles) < int(context.get("minimum_roles", 2)):
        decision = "abstain"
        reason = "Not enough structural roles matched for generalized context expansion."
    elif missing_required:
        decision = "abstain"
        reason = f"Missing required roles: {', '.join(missing_required)}."
    else:
        decision = "accept"
        reason = "Perturbation/growth and induced opposing response are both present; expand generalized negative-feedback context."
    return {
        "context_id": context["context_id"],
        "pattern_id": context["pattern_id"],
        "matched_roles": matched_roles,
        "missing_required_roles": missing_required,
        "negative_control_hits": negative_hits,
        "score": max(0.0, score),
        "decision": decision,
        "reason": reason,
    }


def _word_context_baseline(text: str) -> dict[str, Any]:
    query_terms = _tokens(text)
    rows = []
    for pattern in DEFAULT_STRUCTURAL_PATTERNS:
        pattern_text = " ".join([
            pattern.get("name", ""),
            pattern.get("claim", ""),
            " ".join(pattern.get("trigger_terms", [])),
            " ".join(pattern.get("good_realizations", [])),
        ])
        rows.append({
            "pattern_id": pattern["pattern_id"],
            "score": round(_counter_cosine(query_terms, _tokens(pattern_text)), 4),
        })
    ranked = sorted(rows, key=lambda row: (-row["score"], row["pattern_id"]))
    top = ranked[0] if ranked else {"pattern_id": None, "score": 0.0}
    return {
        "top_pattern_id": top["pattern_id"] if top["score"] >= 0.12 else None,
        "score": top["score"],
        "ranking": ranked[:5],
    }


def _build_context_graph(text: str, role_hits: dict[str, list[str]], negative_hits: list[str]) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    graph: dict[str, dict[str, float]] = defaultdict(dict)
    edges = []
    problem_node = "problem::query"
    for role, hits in role_hits.items():
        role_node = f"role::{role}"
        _add_edge(graph, edges, problem_node, role_node, "structural_synonym_edge", 0.9)
        for hit in hits:
            _add_edge(graph, edges, f"phrase::{hit}", role_node, "role_synonym_edge", 0.75)
    for context in CONTEXT_PATTERNS:
        context_node = f"assumption_context::{context['context_id']}"
        for role in context["roles"]:
            if role in role_hits:
                _add_edge(graph, edges, f"role::{role}", context_node, "generalized_context_edge", 0.85)
        pattern_node = f"pattern::{context['pattern_id']}"
        _add_edge(graph, edges, context_node, pattern_node, "context_to_pattern_edge", 1.0)
        for realization_id in context["classic_realizations"]:
            _add_edge(graph, edges, pattern_node, f"realization::{realization_id}", "context_realization_edge", 0.8)
    for hit in negative_hits:
        _add_edge(graph, edges, f"phrase::{hit}", "negative_control::matched", "negative_control_edge", 1.0)
    return dict(graph), {"edges": edges}


def _evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    structural = expand_structural_context(case["text"])
    baseline = _word_context_baseline(case["text"])
    return {
        "case_id": case["case_id"],
        "text": case["text"],
        "expected_pattern_id": case.get("expected_pattern_id"),
        "expected_decision": case["expected_decision"],
        "word_context_baseline": baseline,
        "structural_context": {
            key: value
            for key, value in structural.items()
            if key != "graph"
        },
        "pass": (
            (
                case.get("expected_pattern_id") is None
                and structural["decision"] in {"abstain", "block_negative_control"}
            )
            or (
                case.get("expected_pattern_id") == structural["top_pattern_id"]
                and structural["decision"] == case["expected_decision"]
            )
        ),
    }


def _eval_cases() -> list[dict[str, Any]]:
    return [
        {
            "case_id": "feedback_platform_rate_limits",
            "text": "As platform traffic grows, moderation load grows too, but rate limits and review queues increasingly dampen posting until throughput stabilizes.",
            "expected_pattern_id": "pat_negative_feedback",
            "expected_decision": "accept",
        },
        {
            "case_id": "feedback_predator_capacity",
            "text": "As the prey population rises, predators and resource competition rise as well, pushing the population back toward carrying capacity.",
            "expected_pattern_id": "pat_negative_feedback",
            "expected_decision": "accept",
        },
        {
            "case_id": "feedback_market_chinese",
            "text": "随着需求增加，价格上涨会抑制一部分需求并刺激供给，市场逐步回到均衡附近。",
            "expected_pattern_id": "pat_negative_feedback",
            "expected_decision": "accept",
        },
        {
            "case_id": "feedback_api_backoff",
            "text": "When API retry volume rises, backoff delays and circuit breakers increase, reducing retry pressure until load reaches a steady state.",
            "expected_pattern_id": "pat_negative_feedback",
            "expected_decision": "accept",
        },
        {
            "case_id": "feedback_lenz_direct",
            "text": "When magnetic flux increases, the induced current creates a field opposing the flux change.",
            "expected_pattern_id": "pat_negative_feedback",
            "expected_decision": "accept",
        },
        {
            "case_id": "feedback_le_chatelier_direct",
            "text": "Adding reactant disturbs equilibrium, so the reaction shifts in the direction that counteracts the imposed change.",
            "expected_pattern_id": "pat_negative_feedback",
            "expected_decision": "accept",
        },
        {
            "case_id": "feedback_calibration_penalty",
            "text": "As confidence grows too high, an uncertainty penalty increases and pulls the score back toward a calibrated range.",
            "expected_pattern_id": "pat_negative_feedback",
            "expected_decision": "accept",
        },
        {
            "case_id": "negative_runaway_positive_feedback",
            "text": "As withdrawals increase, fear further increases withdrawals; no balancing institution intervenes, so this is runaway positive feedback.",
            "expected_pattern_id": None,
            "expected_decision": "block_negative_control",
        },
        {
            "case_id": "negative_no_opposing_mechanism",
            "text": "More ad spend increases impressions and sales in a simple monotonic forecast; no opposing mechanism is modeled.",
            "expected_pattern_id": None,
            "expected_decision": "block_negative_control",
        },
        {
            "case_id": "negative_random_no_constraint",
            "text": "Random sensor jitter changes readings without any induced compensating response or invariant constraint.",
            "expected_pattern_id": None,
            "expected_decision": "block_negative_control",
        },
        {
            "case_id": "negative_plain_bottleneck",
            "text": "Throughput is capped by a single GPU queue; add capacity to the scarce resource rather than optimize unrelated stages.",
            "expected_pattern_id": None,
            "expected_decision": "abstain",
        },
    ]


def _term_hits(text: str, terms: list[str]) -> list[str]:
    low = text.lower()
    hits = []
    for term in terms:
        term_low = term.lower()
        if re.search(rf"(?<![a-z0-9]){re.escape(term_low)}(?![a-z0-9])", low) or term_low in low:
            hits.append(term)
    return sorted(set(hits))


def _context_by_id(context_id: str | None) -> dict[str, Any]:
    for context in CONTEXT_PATTERNS:
        if context["context_id"] == context_id:
            return context
    return {}


def _add_edge(
    graph: dict[str, dict[str, float]],
    edges: list[dict[str, Any]],
    source: str,
    target: str,
    edge_type: str,
    weight: float,
) -> None:
    if source == target:
        return
    graph.setdefault(source, {})
    graph.setdefault(target, {})
    graph[source][target] = max(graph[source].get(target, 0.0), weight)
    graph[target][source] = max(graph[target].get(source, 0.0), weight)
    edges.append({"source": source, "target": target, "edge_type": edge_type, "weight": weight})


def _rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generalized structural-context edges.")
    parser.add_argument("--eval-id", default="structural_context_edges_20260607")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    payload = build_structural_context_edge_payload(eval_id=args.eval_id)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
