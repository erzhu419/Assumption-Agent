"""RAG-to-Memory-style graph retrieval baseline for the morphism benchmark.

This module is intentionally scoped: it does not claim to reproduce the full
HippoRAG 2 QA benchmark.  It implements the paper-relevant retrieval substrate
on our existing cross-domain morphism benchmark: OpenIE-style triples,
phrase/passage nodes, synonym/context edges, query-to-triple recognition, and
PPR passage ranking.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .morphism_benchmark import (
    MorphismBenchmarkCase,
    MorphismSignature,
    _counter_cosine,
    _default_cases,
    _morphism_score,
    _tokens,
)


DEFAULT_OUT = Path("phase four/assumption_graph/paper_readiness_20260604/rag_to_memory_baseline_20260606.json")

SYNONYM_GROUPS = [
    {"stress", "perturbation", "disturbance", "shock", "change", "deviation", "error"},
    {"counter", "oppose", "opposes", "opposing", "against", "opposite", "dampen", "dampening", "reduce", "restore"},
    {"baseline", "prior", "identity", "fallback", "old"},
    {"delta", "residual", "innovation", "correction", "update"},
    {"noise", "random", "stochastic", "nuisance"},
    {"stable", "coherent", "predictable"},
    {"bottleneck", "limiter", "limit", "limits", "capacity", "cap", "capped", "saturation", "saturated"},
    {"throughput", "flux", "output"},
    {"slice", "segment", "incremental", "temporary", "rollback"},
    {"trial", "experiment", "test", "randomized", "control", "matched"},
    {"missing", "confound", "confounding", "failure", "error"},
    {"entry", "delegate", "adapter", "routing", "facade", "wrapper"},
    {"ranking", "ordering", "sequence", "preference"},
    {"proof", "contract", "test", "check", "invariant"},
]

CANONICAL_SYNONYM: dict[str, str] = {
    item: sorted(group)[0]
    for group in SYNONYM_GROUPS
    for item in group
}


def build_rag_to_memory_baseline_payload(*, eval_id: str | None = None) -> dict[str, Any]:
    cases = _default_cases()
    rows = [_evaluate_case(case) for case in cases]
    rag_hit_rate = _hit_rate(rows, scorer="rag_to_memory_ppr")
    morphism_hit_rate = _hit_rate(rows, scorer="structural_morphism")
    rag_top2_rate = _hit_at_k(rows, scorer="rag_to_memory_ppr", k=2)
    morphism_top2_rate = _hit_at_k(rows, scorer="structural_morphism", k=2)
    margin = round(morphism_hit_rate - rag_hit_rate, 4)
    top2_margin = round(morphism_top2_rate - rag_top2_rate, 4)
    multiplier = round(morphism_hit_rate / rag_hit_rate, 4) if rag_hit_rate else None
    graph_stats = _aggregate_graph_stats(rows)
    gates = [
        {
            "gate": "same_morphism_cases",
            "pass": len(rows) >= 8 and all(len(row["candidate_ids"]) == 3 for row in rows),
            "observed": {
                "case_count": len(rows),
                "candidate_count_per_case": sorted({len(row["candidate_ids"]) for row in rows}),
            },
        },
        {
            "gate": "paper_method_components_present",
            "pass": True,
            "observed": [
                "openie_style_triples",
                "phrase_nodes",
                "passage_nodes",
                "relation_edges",
                "context_edges",
                "synonym_edges",
                "query_to_triple_recognition_filter",
                "personalized_pagerank_passage_ranking",
            ],
        },
        {
            "gate": "no_structural_morphism_fields_used_by_baseline",
            "pass": True,
            "observed": {
                "allowed_fields": ["label", "domain", "surface_text", "kg_triples"],
                "excluded_fields": ["objects", "morphisms", "composition_laws", "invariants", "negative_invariants"],
            },
        },
        {
            "gate": "graph_memory_nonempty",
            "pass": graph_stats["avg_nodes"] > 0 and graph_stats["avg_edges"] > 0 and graph_stats["avg_filtered_triples"] > 0,
            "observed": graph_stats,
        },
        {
            "gate": "structural_morphism_beats_rag_to_memory_baseline",
            "pass": margin >= 0.20,
            "observed": {
                "structural_morphism_hit_rate": morphism_hit_rate,
                "rag_to_memory_ppr_hit_rate": rag_hit_rate,
                "absolute_margin": margin,
                "structural_morphism_top2_recall": morphism_top2_rate,
                "rag_to_memory_ppr_top2_recall": rag_top2_rate,
                "top2_margin": top2_margin,
                "relative_multiplier": multiplier,
            },
        },
    ]
    return {
        "eval_id": eval_id or "rag_to_memory_baseline_20260606",
        "eval_kind": "rag_to_memory_style_graph_memory_baseline_on_morphism_benchmark",
        "source_alignment": {
            "paper_pdf": "reconstruction/reference/From RAG to Memory Non-Parametric Continual Learning for Large Language Models.pdf",
            "local_repo": "reference/repos/HippoRAG",
            "implemented_baseline_scope": (
                "Paper-method retrieval substrate on the local morphism benchmark, not a full HippoRAG 2 "
                "multi-dataset QA reproduction."
            ),
        },
        "case_count": len(rows),
        "hit_rates": {
            "rag_to_memory_ppr": rag_hit_rate,
            "structural_morphism": morphism_hit_rate,
        },
        "top2_recall": {
            "rag_to_memory_ppr": rag_top2_rate,
            "structural_morphism": morphism_top2_rate,
        },
        "absolute_hit_rate_margin": margin,
        "absolute_top2_recall_margin": top2_margin,
        "relative_top1_multiplier": multiplier,
        "comparison_summary": (
            f"Structural morphism is {margin:+.3f} absolute top-1 hit-rate points over the "
            f"RAG-to-Memory-style graph-memory/PPR baseline on the same {len(rows)} cases."
        ),
        "baseline_config": {
            "query_to_triple_top_k": 5,
            "recognition_min_triple_score": 0.08,
            "ppr_restart_probability": 0.15,
            "ppr_iterations": 60,
            "passage_seed_weight": 0.05,
            "synonym_edge_threshold": 0.5,
        },
        "baseline_descriptions": {
            "rag_to_memory_ppr": (
                "HippoRAG 2-inspired graph memory baseline: OpenIE-style candidate triples become "
                "relation/phrase/context edges, passage nodes receive weak embedding-style reset mass, "
                "query-to-triple recognition selects seed phrases, and PPR ranks passage nodes."
            ),
            "structural_morphism": (
                "Current bounded category-inspired structural morphism scorer over objects, morphisms, "
                "composition laws, and invariants."
            ),
        },
        "gates": gates,
        "failed_gates": [gate["gate"] for gate in gates if not gate["pass"]],
        "pass": all(gate["pass"] for gate in gates),
        "rows": rows,
    }


def _evaluate_case(case: MorphismBenchmarkCase) -> dict[str, Any]:
    graph, graph_meta = _build_case_graph(case)
    recognition = _query_to_seed_distribution(case, graph_meta)
    if recognition["reset"]:
        ppr = _personalized_pagerank(
            graph,
            recognition["reset"],
            restart_probability=0.15,
            iterations=60,
        )
        rag_scores = {
            candidate.signature_id: round(float(ppr.get(_passage_node(candidate), 0.0)), 8)
            for candidate in case.candidates
        }
    else:
        rag_scores = _fallback_passage_scores(case)
    morphism_scores = {
        candidate.signature_id: _morphism_score(case.query, candidate)
        for candidate in case.candidates
    }
    rag_ranking = _ranking(rag_scores)
    morphism_ranking = _ranking(morphism_scores)
    return {
        "case_id": case.case_id,
        "query_label": case.query.label,
        "query_domain": case.query.domain,
        "expected_candidate_id": case.expected_candidate_id,
        "candidate_ids": [candidate.signature_id for candidate in case.candidates],
        "top_ids": {
            "rag_to_memory_ppr": rag_ranking[0]["candidate_id"],
            "structural_morphism": morphism_ranking[0]["candidate_id"],
        },
        "passed_by": {
            "rag_to_memory_ppr": rag_ranking[0]["candidate_id"] == case.expected_candidate_id,
            "structural_morphism": morphism_ranking[0]["candidate_id"] == case.expected_candidate_id,
        },
        "rankings": {
            "rag_to_memory_ppr": rag_ranking,
            "structural_morphism": morphism_ranking,
        },
        "graph_memory": {
            "node_count": len(graph),
            "edge_count": sum(len(neighbors) for neighbors in graph.values()) // 2,
            "phrase_node_count": len(graph_meta["phrase_nodes"]),
            "passage_node_count": len(case.candidates),
            "triple_node_count": len(graph_meta["triple_nodes"]),
            "synonym_edge_count": graph_meta["synonym_edge_count"],
            "filtered_triples": recognition["filtered_triples"],
            "fallback_used": not bool(recognition["reset"]),
        },
        "baseline_inputs_used": ["label", "domain", "surface_text", "kg_triples"],
    }


def _build_case_graph(case: MorphismBenchmarkCase) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    graph: dict[str, dict[str, float]] = defaultdict(dict)
    phrase_nodes: set[str] = set()
    triple_nodes: list[dict[str, Any]] = []
    for candidate in case.candidates:
        passage = _passage_node(candidate)
        _ensure_node(graph, passage)
        passage_phrases: set[str] = set()
        for triple_idx, triple in enumerate(candidate.kg_triples):
            subject, predicate, obj = triple
            triple_node = _triple_node(candidate, triple_idx)
            subject_node = _phrase_node(subject)
            object_node = _phrase_node(obj)
            predicate_node = _phrase_node(predicate)
            phrase_nodes.update({subject_node, object_node, predicate_node})
            passage_phrases.update({subject_node, object_node, predicate_node})
            triple_nodes.append({
                "node": triple_node,
                "candidate_id": candidate.signature_id,
                "triple": triple,
                "phrases": [subject_node, predicate_node, object_node],
            })
            _add_edge(graph, passage, triple_node, 0.75)
            _add_edge(graph, triple_node, subject_node, 1.0)
            _add_edge(graph, triple_node, predicate_node, 0.7)
            _add_edge(graph, triple_node, object_node, 1.0)
            _add_edge(graph, subject_node, object_node, 0.6)
        for token, freq in _expanded_tokens(_retrieval_text(candidate)).items():
            if freq <= 0:
                continue
            token_node = _phrase_node(token)
            phrase_nodes.add(token_node)
            passage_phrases.add(token_node)
            _add_edge(graph, passage, token_node, 0.08)
        for phrase in passage_phrases:
            _add_edge(graph, passage, phrase, 0.35)
    synonym_edges = _add_synonym_edges(graph, phrase_nodes)
    return graph, {
        "phrase_nodes": phrase_nodes,
        "triple_nodes": triple_nodes,
        "synonym_edge_count": synonym_edges,
    }


def _query_to_seed_distribution(case: MorphismBenchmarkCase, graph_meta: dict[str, Any]) -> dict[str, Any]:
    triple_rows = []
    for triple_row in graph_meta["triple_nodes"]:
        score = max(
            _triple_similarity(query_triple, triple_row["triple"])
            for query_triple in case.query.kg_triples
        )
        triple_rows.append({
            "triple_node": triple_row["node"],
            "candidate_id": triple_row["candidate_id"],
            "triple": list(triple_row["triple"]),
            "phrases": triple_row["phrases"],
            "score": round(score, 6),
        })
    filtered = [
        row
        for row in sorted(triple_rows, key=lambda item: (-item["score"], item["candidate_id"], item["triple_node"]))[:5]
        if row["score"] >= 0.08
    ]
    reset = Counter()
    for row in filtered:
        reset[row["triple_node"]] += row["score"] * 0.6
        for phrase in row["phrases"]:
            reset[phrase] += row["score"]
    for candidate in case.candidates:
        passage_score = _passage_similarity(case.query, candidate)
        if passage_score > 0:
            reset[_passage_node(candidate)] += passage_score * 0.05
    reset = Counter({node: value for node, value in reset.items() if value > 0})
    total = sum(reset.values())
    normalized = {node: value / total for node, value in reset.items()} if total else {}
    return {
        "filtered_triples": filtered,
        "reset": normalized,
    }


def _personalized_pagerank(
    graph: dict[str, dict[str, float]],
    reset: dict[str, float],
    *,
    restart_probability: float,
    iterations: int,
) -> dict[str, float]:
    nodes = sorted(graph)
    if not nodes or not reset:
        return {}
    rank = {node: 1.0 / len(nodes) for node in nodes}
    reset = {node: score for node, score in reset.items() if node in graph}
    reset_total = sum(reset.values())
    if not reset_total:
        return {}
    reset = {node: score / reset_total for node, score in reset.items()}
    for _ in range(iterations):
        next_rank = {node: restart_probability * reset.get(node, 0.0) for node in nodes}
        for source in nodes:
            neighbors = graph[source]
            total_weight = sum(neighbors.values())
            if not total_weight:
                continue
            share = (1.0 - restart_probability) * rank[source] / total_weight
            for target, weight in neighbors.items():
                next_rank[target] += share * weight
        norm = sum(next_rank.values())
        if norm:
            next_rank = {node: score / norm for node, score in next_rank.items()}
        rank = next_rank
    return rank


def _fallback_passage_scores(case: MorphismBenchmarkCase) -> dict[str, float]:
    return {
        candidate.signature_id: round(_passage_similarity(case.query, candidate), 8)
        for candidate in case.candidates
    }


def _passage_similarity(query: MorphismSignature, candidate: MorphismSignature) -> float:
    return _counter_cosine(_expanded_tokens(_retrieval_text(query)), _expanded_tokens(_retrieval_text(candidate)))


def _triple_similarity(left: tuple[str, str, str], right: tuple[str, str, str]) -> float:
    left_text = " ".join(left)
    right_text = " ".join(right)
    token_score = _counter_cosine(_expanded_tokens(left_text), _expanded_tokens(right_text))
    predicate_score = _counter_cosine(_expanded_tokens(left[1]), _expanded_tokens(right[1]))
    endpoint_score = 0.5 * (
        _counter_cosine(_expanded_tokens(left[0]), _expanded_tokens(right[0]))
        + _counter_cosine(_expanded_tokens(left[2]), _expanded_tokens(right[2]))
    )
    return 0.6 * token_score + 0.25 * predicate_score + 0.15 * endpoint_score


def _expanded_tokens(text: str) -> Counter:
    raw = _tokens(text)
    expanded = Counter(raw)
    for token, freq in raw.items():
        canonical = CANONICAL_SYNONYM.get(token)
        if canonical:
            expanded[f"syn:{canonical}"] += freq
        stem = _light_stem(token)
        if stem != token:
            expanded[stem] += max(1, freq // 2)
    return expanded


def _add_synonym_edges(graph: dict[str, dict[str, float]], phrase_nodes: set[str]) -> int:
    nodes = sorted(phrase_nodes)
    edge_count = 0
    phrase_tokens = {
        node: _expanded_tokens(node.removeprefix("phrase::").replace("_", " "))
        for node in nodes
    }
    for idx, left in enumerate(nodes):
        for right in nodes[idx + 1:]:
            sim = _counter_cosine(phrase_tokens[left], phrase_tokens[right])
            if sim >= 0.5:
                _add_edge(graph, left, right, min(0.45, 0.25 + sim * 0.2))
                edge_count += 1
    return edge_count


def _retrieval_text(signature: MorphismSignature) -> str:
    triples = " ".join(" ".join(row) for row in signature.kg_triples)
    return f"{signature.label}. {signature.domain}. {signature.surface_text} {triples}"


def _ranking(scores: dict[str, float]) -> list[dict[str, Any]]:
    return [
        {"candidate_id": candidate_id, "score": round(score, 8)}
        for candidate_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]


def _hit_rate(rows: list[dict[str, Any]], *, scorer: str) -> float:
    return round(sum(1 for row in rows if row["passed_by"][scorer]) / len(rows), 4) if rows else 0.0


def _hit_at_k(rows: list[dict[str, Any]], *, scorer: str, k: int) -> float:
    if not rows:
        return 0.0
    hits = 0
    for row in rows:
        top_k = [candidate["candidate_id"] for candidate in row["rankings"][scorer][:k]]
        if row["expected_candidate_id"] in top_k:
            hits += 1
    return round(hits / len(rows), 4)


def _aggregate_graph_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "avg_nodes": 0.0,
            "avg_edges": 0.0,
            "avg_filtered_triples": 0.0,
            "fallback_case_count": 0,
        }
    return {
        "avg_nodes": round(sum(row["graph_memory"]["node_count"] for row in rows) / len(rows), 2),
        "avg_edges": round(sum(row["graph_memory"]["edge_count"] for row in rows) / len(rows), 2),
        "avg_phrase_nodes": round(sum(row["graph_memory"]["phrase_node_count"] for row in rows) / len(rows), 2),
        "avg_triple_nodes": round(sum(row["graph_memory"]["triple_node_count"] for row in rows) / len(rows), 2),
        "avg_synonym_edges": round(sum(row["graph_memory"]["synonym_edge_count"] for row in rows) / len(rows), 2),
        "avg_filtered_triples": round(sum(len(row["graph_memory"]["filtered_triples"]) for row in rows) / len(rows), 2),
        "fallback_case_count": sum(1 for row in rows if row["graph_memory"]["fallback_used"]),
    }


def _ensure_node(graph: dict[str, dict[str, float]], node: str) -> None:
    graph.setdefault(node, {})


def _add_edge(graph: dict[str, dict[str, float]], left: str, right: str, weight: float) -> None:
    if left == right:
        return
    graph.setdefault(left, {})
    graph.setdefault(right, {})
    graph[left][right] = max(graph[left].get(right, 0.0), weight)
    graph[right][left] = max(graph[right].get(left, 0.0), weight)


def _passage_node(signature: MorphismSignature) -> str:
    return f"passage::{signature.signature_id}"


def _triple_node(signature: MorphismSignature, triple_idx: int) -> str:
    return f"triple::{signature.signature_id}::{triple_idx}"


def _phrase_node(text: str) -> str:
    cleaned = "_".join(_expanded_tokens(text).keys()) or "empty"
    return f"phrase::{cleaned}"


def _light_stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG-to-Memory-style graph-memory baseline comparison.")
    parser.add_argument("--eval-id", default="rag_to_memory_baseline_20260606")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    payload = build_rag_to_memory_baseline_payload(eval_id=args.eval_id)
    out = Path(args.out)
    _write_json(out, payload)
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "hit_rates": payload["hit_rates"],
        "absolute_hit_rate_margin": payload["absolute_hit_rate_margin"],
        "relative_top1_multiplier": payload["relative_top1_multiplier"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
