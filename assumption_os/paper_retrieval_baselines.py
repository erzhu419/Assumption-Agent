"""Paper-facing full-text RAG/vector retrieval baselines."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .morphism_benchmark import (
    _counter_cosine,
    _default_cases,
    _morphism_score,
    _tokens,
    build_morphism_independent_benchmark_payload,
)


DEFAULT_OUT = Path("phase four/assumption_graph/paper_readiness_20260604/paper_retrieval_baselines_20260605.json")


def build_paper_retrieval_baselines_payload(
    *,
    eval_id: str | None = None,
    include_neural: bool = False,
    neural_model: str | None = None,
) -> dict[str, Any]:
    cases = _default_cases()
    tfidf_rows = [_evaluate_full_text_case(case, scorer="tfidf") for case in cases]
    bm25_rows = [_evaluate_full_text_case(case, scorer="bm25") for case in cases]
    morphism_rows = [_evaluate_morphism_case(case) for case in cases]
    rows_by_scorer = {
        "ordinary_rag_bm25_full_text": bm25_rows,
        "full_text_tfidf_vector_retrieval": tfidf_rows,
        "structural_morphism": morphism_rows,
    }
    neural_payload = None
    if include_neural:
        try:
            neural_payload = build_morphism_independent_benchmark_payload(
                eval_id=f"{eval_id or 'paper_retrieval_baselines'}_neural",
                neural_embedding_backend="sentence_transformer",
                neural_embedding_model=neural_model or "sentence-transformers/all-MiniLM-L6-v2",
            )
            rows_by_scorer["sentence_transformer_embedding"] = _rows_from_morphism_payload(
                neural_payload,
                scorer="neural_embedding",
            )
        except RuntimeError as exc:
            neural_payload = {
                "enabled": False,
                "status": "unavailable",
                "error": str(exc),
                "model": neural_model or "sentence-transformers/all-MiniLM-L6-v2",
            }

    hit_rates = {
        scorer: _hit_rate(rows)
        for scorer, rows in rows_by_scorer.items()
    }
    baseline_scorers = [scorer for scorer in hit_rates if scorer != "structural_morphism"]
    best_baseline = max(hit_rates[scorer] for scorer in baseline_scorers)
    gates = [
        {
            "gate": "full_text_rag_baselines_present",
            "pass": all(name in hit_rates for name in [
                "ordinary_rag_bm25_full_text",
                "full_text_tfidf_vector_retrieval",
            ]),
            "observed": sorted(hit_rates),
        },
        {
            "gate": "morphism_beats_full_text_retrieval",
            "pass": hit_rates["structural_morphism"] - best_baseline >= 0.20,
            "observed": {
                "hit_rates": hit_rates,
                "best_full_text_baseline": best_baseline,
                "margin": round(hit_rates["structural_morphism"] - best_baseline, 4),
            },
        },
        {
            "gate": "ordinary_rag_is_real_retrieval_not_prompt_length",
            "pass": all(
                row.get("retrieved_candidate_id")
                for rows in (tfidf_rows, bm25_rows)
                for row in rows
            ),
            "observed": {
                "bm25_row_count": len(bm25_rows),
                "tfidf_row_count": len(tfidf_rows),
                "retrieval_unit": "candidate full text, not answer prompt length",
            },
        },
    ]
    if include_neural:
        neural_enabled = bool(neural_payload and neural_payload.get("enabled", True) is not False)
        gates.append({
            "gate": "neural_embedding_baseline_recorded",
            "pass": neural_enabled,
            "observed": {
                "enabled": neural_enabled,
                "model": (neural_payload or {}).get("model"),
                "status": (neural_payload or {}).get("status", "ok" if neural_enabled else "unavailable"),
                "hit_rate": hit_rates.get("sentence_transformer_embedding"),
            },
        })
        if neural_enabled:
            gates.append({
                "gate": "morphism_beats_neural_embedding",
                "pass": hit_rates["structural_morphism"] - hit_rates.get("sentence_transformer_embedding", 1.0) >= 0.20,
                "observed": {
                    "morphism": hit_rates["structural_morphism"],
                    "sentence_transformer_embedding": hit_rates.get("sentence_transformer_embedding"),
                    "margin": round(
                        hit_rates["structural_morphism"] - hit_rates.get("sentence_transformer_embedding", 0.0),
                        4,
                    ),
                },
            })
    return {
        "eval_id": eval_id or "paper_retrieval_baselines_20260605",
        "eval_kind": "paper_full_text_rag_and_vector_baselines",
        "case_count": len(cases),
        "pass": all(gate["pass"] for gate in gates),
        "hit_rates": hit_rates,
        "best_full_text_baseline_hit_rate": best_baseline,
        "morphism_margin_over_best_retrieval": round(hit_rates["structural_morphism"] - best_baseline, 4),
        "baseline_descriptions": {
            "ordinary_rag_bm25_full_text": "BM25 retrieval over candidate full text, used as ordinary RAG context selection.",
            "full_text_tfidf_vector_retrieval": "TF-IDF vector cosine over candidate full text.",
            "sentence_transformer_embedding": "Optional sentence-transformer embedding retrieval over the same surface text.",
            "structural_morphism": "Bounded structural morphism score over objects, morphisms, composition laws, and invariants.",
        },
        "neural_embedding_baseline": {
            key: value
            for key, value in (neural_payload or {"enabled": False, "status": "not_requested"}).items()
            if key != "rows"
        },
        "rows_by_scorer": rows_by_scorer,
        "gates": gates,
        "failed_gates": [gate["gate"] for gate in gates if not gate["pass"]],
    }


def _evaluate_full_text_case(case, *, scorer: str) -> dict[str, Any]:
    docs = [_retrieval_text(candidate) for candidate in case.candidates]
    query = _retrieval_text(case.query)
    if scorer == "tfidf":
        all_texts = [query, *docs]
        scores = _tfidf_scores(query, docs, all_texts)
    elif scorer == "bm25":
        scores = _bm25_scores(query, docs)
    else:
        raise ValueError(f"unknown scorer: {scorer}")
    scored = [
        {
            "candidate_id": candidate.signature_id,
            "score": round(scores[idx], 6),
            "domain": candidate.domain,
        }
        for idx, candidate in enumerate(case.candidates)
    ]
    top = sorted(scored, key=lambda row: (-row["score"], row["candidate_id"]))[0]
    return {
        "case_id": case.case_id,
        "query_domain": case.query.domain,
        "expected_candidate_id": case.expected_candidate_id,
        "retrieved_candidate_id": top["candidate_id"],
        "hit": top["candidate_id"] == case.expected_candidate_id,
        "scores": scored,
    }


def _evaluate_morphism_case(case) -> dict[str, Any]:
    scored = [
        {
            "candidate_id": candidate.signature_id,
            "score": _morphism_score(case.query, candidate),
            "domain": candidate.domain,
        }
        for candidate in case.candidates
    ]
    top = sorted(scored, key=lambda row: (-row["score"], row["candidate_id"]))[0]
    return {
        "case_id": case.case_id,
        "query_domain": case.query.domain,
        "expected_candidate_id": case.expected_candidate_id,
        "retrieved_candidate_id": top["candidate_id"],
        "hit": top["candidate_id"] == case.expected_candidate_id,
        "scores": scored,
    }


def _rows_from_morphism_payload(payload: dict[str, Any], *, scorer: str) -> list[dict[str, Any]]:
    rows = []
    for row in payload.get("rows", []):
        ranked = row.get("rankings", {}).get(scorer, [])
        top = ranked[0]["candidate_id"] if ranked else None
        rows.append({
            "case_id": row.get("case_id"),
            "query_domain": row.get("query_domain"),
            "expected_candidate_id": row.get("expected_candidate_id"),
            "retrieved_candidate_id": top,
            "hit": top == row.get("expected_candidate_id"),
            "scores": [
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "score": candidate.get("scores", {}).get(scorer),
                    "domain": candidate.get("domain"),
                }
                for candidate in row.get("candidate_scores", [])
            ],
        })
    return rows


def _retrieval_text(signature) -> str:
    triples = " ".join(" ".join(row) for row in signature.kg_triples)
    return f"{signature.label}. {signature.domain}. {signature.surface_text} {triples}"


def _tfidf_scores(query: str, docs: list[str], all_texts: list[str]) -> list[float]:
    df = Counter()
    for text in all_texts:
        for token in set(_tokens(text)):
            df[token] += 1
    total = len(all_texts)
    return [
        _counter_cosine(_tfidf_vector(query, df=df, total=total), _tfidf_vector(doc, df=df, total=total))
        for doc in docs
    ]


def _tfidf_vector(text: str, *, df: Counter, total: int) -> Counter:
    terms = _tokens(text)
    return Counter({
        token: freq * (math.log((total + 1) / (df[token] + 1)) + 1.0)
        for token, freq in terms.items()
    })


def _bm25_scores(query: str, docs: list[str]) -> list[float]:
    tokenized = [_tokens(doc) for doc in docs]
    avgdl = sum(sum(doc.values()) for doc in tokenized) / len(tokenized)
    query_terms = list(_tokens(query))
    scores = []
    for doc in tokenized:
        dl = sum(doc.values())
        score = 0.0
        for term in query_terms:
            containing = sum(1 for row in tokenized if term in row)
            freq = doc.get(term, 0)
            if not freq:
                continue
            idf = math.log((len(docs) - containing + 0.5) / (containing + 0.5) + 1.0)
            score += idf * (freq * 2.5) / (freq + 1.5 * (0.25 + 0.75 * dl / avgdl))
        scores.append(score)
    return scores


def _hit_rate(rows: list[dict[str, Any]]) -> float:
    return round(sum(1 for row in rows if row.get("hit")) / len(rows), 4) if rows else 0.0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build paper full-text RAG/vector retrieval baselines.")
    ap.add_argument("--eval-id", default="paper_retrieval_baselines_20260605")
    ap.add_argument("--include-neural", action="store_true")
    ap.add_argument("--neural-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    payload = build_paper_retrieval_baselines_payload(
        eval_id=args.eval_id,
        include_neural=args.include_neural,
        neural_model=args.neural_model,
    )
    out = Path(args.out)
    _write_json(out, payload)
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "hit_rates": payload["hit_rates"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
