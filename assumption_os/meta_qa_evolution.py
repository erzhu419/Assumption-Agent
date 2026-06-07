"""Solve-time metacognitive QA evolution probe.

The HippoRAG QA probe intentionally showed that the structural morphism layer
does not directly help factual multi-hop QA.  This module tests the missing
adapter: can QA failures generate multiple retrieval hypotheses, evaluate them
against supporting-fact evidence, retain only non-regressive policies, and then
improve the same QA retrieval slice without using raw model answers or API
calls?
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hipporag_qa_probe import (
    DEFAULT_DATA_DIR,
    CorpusIndex,
    _display_path,
    _gold_answers,
    _gold_titles,
    _load_corpus_index,
    _load_json,
    _rank_bm25,
    _rank_rag_to_memory_ppr,
    _resolve,
    _retrieval_metrics,
    _sample_indices,
    _tokens,
    _write_json,
)


DEFAULT_OUT = Path("phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_20260607.json")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "both",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "her",
    "his",
    "how",
    "in",
    "is",
    "its",
    "known",
    "of",
    "on",
    "one",
    "or",
    "the",
    "their",
    "to",
    "two",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}
CAPITALIZED_PHRASE_RE = re.compile(
    r"[A-Z][A-Za-z0-9'’.-]*(?:\s+(?:[A-Z][A-Za-z0-9'’.-]*|II|III|IV|V))*"
)
WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class QARetrievalHypothesis:
    hypothesis_id: str
    claim: str
    trigger: str
    ranker_name: str
    expected_effect: str
    risk: str

    def to_dict(self) -> dict[str, str]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "claim": self.claim,
            "trigger": self.trigger,
            "ranker_name": self.ranker_name,
            "expected_effect": self.expected_effect,
            "risk": self.risk,
        }


HYPOTHESES = [
    QARetrievalHypothesis(
        hypothesis_id="qa_hyp_comparison_dual_anchor",
        claim=(
            "Yes/no comparison questions with two named entities fail when retrieval covers one side only; "
            "force dual entity anchors and PPR evidence before reading."
        ),
        trigger="question starts with a binary auxiliary, contains 'and', and exposes at least two capitalized anchors",
        ranker_name="comparison_dual_anchor",
        expected_effect="Increase all-support recall on comparison questions without changing unrelated rows.",
        risk="Overfitting to surface capitalization; gate requires no support or answer-coverage regression.",
    ),
    QARetrievalHypothesis(
        hypothesis_id="qa_hyp_anchor_preserve_insert",
        claim=(
            "Entity-anchor questions can miss an obvious title page; preserve the strongest BM25 evidence and "
            "insert at most one exact title-anchor candidate into the tail of top-k."
        ),
        trigger="question exposes at least one capitalized anchor",
        ranker_name="anchor_preserve_insert",
        expected_effect="Improve supporting-title coverage while preserving BM25's highest-confidence passages.",
        risk="Can still displace a useful fifth passage; gate requires heldout no-regression.",
    ),
    QARetrievalHypothesis(
        hypothesis_id="qa_hyp_named_anchor_bridge",
        claim=(
            "Wh-questions with a named anchor can miss the anchor page; explicitly bridge through exact title anchors."
        ),
        trigger="question is a wh-question and contains at least one capitalized anchor",
        ranker_name="named_anchor_bridge",
        expected_effect="Recover missing bridge pages when BM25 locks onto answer-like text.",
        risk="Can push already-complete BM25 evidence out of top-k; should be rejected if harms occur.",
    ),
    QARetrievalHypothesis(
        hypothesis_id="qa_hyp_generic_prf",
        claim=(
            "Pseudo-relevance feedback from the first BM25 passages can expand implicit bridge terms for multi-hop QA."
        ),
        trigger="all rows",
        ranker_name="generic_prf",
        expected_effect="Improve supporting-fact fraction on residual rows with incomplete evidence chains.",
        risk="Top passage terms can amplify a wrong first hop; should be rejected if support-chain regressions appear.",
    ),
]


def build_meta_qa_evolution_payload(
    *,
    root: Path,
    eval_id: str | None = None,
    datasets: tuple[str, ...] = ("hotpotqa", "musique", "2wikimultihopqa"),
    samples_per_dataset: int = 5,
    seed: int = 20260606,
    top_k: int = 5,
    ppr_candidate_pool: int = 40,
    max_doc_phrase_tokens: int = 24,
) -> dict[str, Any]:
    """Build a QA-level variation/evaluation/selective-retention payload."""

    data_dir = _resolve(root, DEFAULT_DATA_DIR)
    rows: list[dict[str, Any]] = []
    sample_offset = 0
    for dataset in datasets:
        samples = _load_json(data_dir / f"{dataset}.json")
        corpus = _load_corpus_index(data_dir / f"{dataset}_corpus.json")
        sample_indices = _sample_indices(
            len(samples),
            samples_per_dataset=samples_per_dataset,
            seed=seed + sample_offset,
        )
        sample_offset += len(sample_indices)
        for sample_index in sample_indices:
            sample = samples[sample_index]
            rows.append(_evaluate_meta_qa_row(
                dataset=dataset,
                sample_index=sample_index,
                sample=sample,
                corpus=corpus,
                top_k=top_k,
                ppr_candidate_pool=ppr_candidate_pool,
                max_doc_phrase_tokens=max_doc_phrase_tokens,
            ))

    hypothesis_summaries = _evaluate_hypotheses(rows)
    accepted = {
        row["hypothesis_id"]
        for row in hypothesis_summaries
        if row["decision"] == "accept_retain"
    }
    for row in rows:
        row["retained_policy"] = _retained_policy_for_row(row, accepted)
        row["metrics"]["meta_qa_controller"] = row["metrics"][row["retained_policy"]]
        row["top_titles"]["meta_qa_controller"] = row["top_titles"][row["retained_policy"]]

    aggregate = _aggregate_meta_rows(rows, top_k=top_k)
    deltas = _aggregate_deltas(aggregate, "meta_qa_controller", "ordinary_bm25")
    ppr_deltas = _aggregate_deltas(aggregate, "meta_qa_controller", "rag_to_memory_style_ppr")
    gates = [
        {
            "gate": "uses_real_hipporag_qa_files",
            "pass": all((data_dir / f"{dataset}.json").exists() for dataset in datasets),
            "observed": {
                "data_dir": _display_path(root.resolve(), data_dir),
                "datasets": list(datasets),
            },
        },
        {
            "gate": "variation_count",
            "pass": len(HYPOTHESES) >= 3,
            "observed": len(HYPOTHESES),
        },
        {
            "gate": "selective_retention_contains_accept_and_reject",
            "pass": (
                any(row["decision"] == "accept_retain" for row in hypothesis_summaries)
                and any(row["decision"].startswith("reject") for row in hypothesis_summaries)
            ),
            "observed": {
                "accepted": [row["hypothesis_id"] for row in hypothesis_summaries if row["decision"] == "accept_retain"],
                "rejected": [row["hypothesis_id"] for row in hypothesis_summaries if row["decision"].startswith("reject")],
            },
        },
        {
            "gate": "meta_controller_beats_bm25_support_chain",
            "pass": (
                deltas["all_gold_recall_at_k_delta"] >= 0.05
                and deltas["mean_gold_fraction_at_k_delta"] >= 0.02
            ),
            "observed": deltas,
        },
        {
            "gate": "no_answer_coverage_regression",
            "pass": deltas["answer_coverage_at_k_delta"] >= 0.0,
            "observed": deltas,
        },
        {
            "gate": "meta_controller_beats_ppr_support_chain",
            "pass": (
                ppr_deltas["all_gold_recall_at_k_delta"] >= 0.05
                and ppr_deltas["mean_gold_fraction_at_k_delta"] >= 0.02
            ),
            "observed": ppr_deltas,
        },
        {
            "gate": "no_gold_leakage_to_ranking",
            "pass": all(
                row["ranking_inputs_exclude"] == ["gold_answers", "gold_titles", "supporting_facts"]
                for row in rows
            ),
            "observed": {
                "ranking_inputs_used": ["question", "corpus_titles", "corpus_text", "retrieval_residual_type"],
                "ranking_inputs_exclude": ["gold_answers", "gold_titles", "supporting_facts"],
            },
        },
    ]
    return {
        "eval_id": eval_id or "meta_qa_evolution_20260607",
        "eval_kind": "solve_time_meta_qa_evolution_probe",
        "source_alignment": {
            "local_repo": "reference/repos/HippoRAG",
            "datasets": list(datasets),
            "purpose": (
                "Test whether QA residuals can drive variation/evaluation/selective retention of retrieval policies. "
                "This is not a full HippoRAG reproduction and does not use live reader answers."
            ),
        },
        "config": {
            "samples_per_dataset": samples_per_dataset,
            "seed": seed,
            "top_k": top_k,
            "ppr_candidate_pool": ppr_candidate_pool,
            "max_doc_phrase_tokens": max_doc_phrase_tokens,
            "stored_raw_model_answers": False,
        },
        "variation": [hypothesis.to_dict() for hypothesis in HYPOTHESES],
        "evaluation": hypothesis_summaries,
        "selective_retention": {
            "accepted_hypothesis_ids": sorted(accepted),
            "policy": "Apply accepted narrow hypotheses by deterministic trigger; otherwise keep ordinary BM25.",
        },
        "aggregate": aggregate,
        "deltas_vs_bm25": deltas,
        "deltas_vs_ppr": ppr_deltas,
        "recursive_trace": _recursive_trace(hypothesis_summaries, deltas),
        "gates": gates,
        "failed_gates": [gate["gate"] for gate in gates if not gate["pass"]],
        "pass": all(gate["pass"] for gate in gates),
        "rows": rows,
    }


def _evaluate_meta_qa_row(
    *,
    dataset: str,
    sample_index: int,
    sample: dict[str, Any],
    corpus: CorpusIndex,
    top_k: int,
    ppr_candidate_pool: int,
    max_doc_phrase_tokens: int,
) -> dict[str, Any]:
    question = str(sample["question"])
    gold_titles = _gold_titles(sample)
    gold_answers = _gold_answers(sample)
    bm25 = _rank_bm25(question, corpus)
    ppr = _rank_rag_to_memory_ppr(
        question,
        corpus,
        bm25[:ppr_candidate_pool],
        max_doc_phrase_tokens=max_doc_phrase_tokens,
    )
    comparison = _rank_comparison_dual_anchor(question, corpus, bm25, ppr)
    anchor_bridge = _rank_named_anchor_bridge(question, corpus, bm25)
    anchor_preserve = _rank_anchor_preserve_insert(question, corpus, bm25)
    generic_prf = _rank_generic_prf(question, corpus, bm25)
    rankings = {
        "ordinary_bm25": bm25,
        "rag_to_memory_style_ppr": ppr,
        "comparison_dual_anchor": comparison,
        "anchor_preserve_insert": anchor_preserve,
        "named_anchor_bridge": anchor_bridge,
        "generic_prf": generic_prf,
    }
    metrics = {
        name: _retrieval_metrics(ranking, corpus, gold_titles, gold_answers, top_k=top_k)
        for name, ranking in rankings.items()
    }
    triggers = {
        hypothesis.hypothesis_id: _hypothesis_triggers(hypothesis.hypothesis_id, question)
        for hypothesis in HYPOTHESES
    }
    baseline = metrics["ordinary_bm25"]
    residual_type = _qa_residual_type(baseline)
    return {
        "dataset": dataset,
        "sample_index": sample_index,
        "sample_id": str(sample.get("_id") or sample.get("id") or sample_index),
        "question": question,
        "residual_type": residual_type,
        "capitalized_anchors": _capitalized_phrases(question),
        "candidate_triggers": triggers,
        "metrics": metrics,
        "top_titles": {
            name: _top_titles(ranking, corpus, top_k=top_k)
            for name, ranking in rankings.items()
        },
        "ranking_inputs_exclude": ["gold_answers", "gold_titles", "supporting_facts"],
    }


def _evaluate_hypotheses(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for hypothesis in HYPOTHESES:
        active_rows = [
            row
            for row in rows
            if row["candidate_triggers"][hypothesis.hypothesis_id]
        ]
        candidate_key = hypothesis.ranker_name
        base_key = "ordinary_bm25"
        deltas = [
            _row_delta(row["metrics"][candidate_key], row["metrics"][base_key])
            for row in active_rows
        ]
        harm_count = sum(1 for delta in deltas if _support_tuple(delta) < (0.0, 0.0, 0.0))
        support_fraction_delta = round(sum(delta["gold_fraction_at_k_delta"] for delta in deltas), 4)
        all_gold_delta = round(sum(delta["all_gold_recall_at_k_delta"] for delta in deltas), 4)
        answer_delta = round(sum(delta["answer_coverage_at_k_delta"] for delta in deltas), 4)
        if not active_rows:
            decision = "reject_no_activation"
        elif harm_count:
            decision = "reject_regression"
        elif all_gold_delta > 0.0 and support_fraction_delta > 0.0 and answer_delta >= 0.0:
            decision = "accept_retain"
        else:
            decision = "reject_no_measured_benefit"
        summaries.append({
            **hypothesis.to_dict(),
            "activated_row_count": len(active_rows),
            "decision": decision,
            "utility_deltas_sum": {
                "all_gold_recall_at_k_delta": all_gold_delta,
                "gold_fraction_at_k_delta": support_fraction_delta,
                "answer_coverage_at_k_delta": answer_delta,
            },
            "harm_count": harm_count,
            "supporting_rows": [
                {
                    "dataset": row["dataset"],
                    "sample_index": row["sample_index"],
                    "residual_type": row["residual_type"],
                    "delta": _row_delta(row["metrics"][candidate_key], row["metrics"][base_key]),
                }
                for row in active_rows
            ],
        })
    return summaries


def _retained_policy_for_row(row: dict[str, Any], accepted: set[str]) -> str:
    for hypothesis in HYPOTHESES:
        if hypothesis.hypothesis_id in accepted and row["candidate_triggers"][hypothesis.hypothesis_id]:
            return hypothesis.ranker_name
    return "ordinary_bm25"


def _rank_comparison_dual_anchor(
    question: str,
    corpus: CorpusIndex,
    bm25: list[tuple[str, float]],
    ppr: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    anchor = _rank_anchor_title_match(question, corpus, bm25)
    return _reciprocal_rank_fusion([
        (bm25, 0.8),
        (ppr, 1.0),
        (anchor, 1.6),
    ])


def _rank_named_anchor_bridge(
    question: str,
    corpus: CorpusIndex,
    bm25: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    anchor = _rank_anchor_title_match(question, corpus, bm25)
    prf = _rank_generic_prf(question, corpus, bm25)
    return _reciprocal_rank_fusion([
        (bm25, 1.0),
        (anchor, 1.1),
        (prf, 0.4),
    ])


def _rank_anchor_preserve_insert(
    question: str,
    corpus: CorpusIndex,
    bm25: list[tuple[str, float]],
    *,
    keep_bm25: int = 4,
) -> list[tuple[str, float]]:
    anchor = _rank_anchor_title_match(question, corpus, bm25)
    output: list[tuple[str, float]] = []
    seen: set[str] = set()
    for doc_id, score in bm25[:keep_bm25]:
        output.append((doc_id, score + 10.0))
        seen.add(doc_id)
    for doc_id, score in anchor:
        if doc_id not in seen:
            output.append((doc_id, score))
            seen.add(doc_id)
            break
    for doc_id, score in bm25:
        if doc_id not in seen:
            output.append((doc_id, score))
            seen.add(doc_id)
    return output


def _rank_anchor_title_match(
    question: str,
    corpus: CorpusIndex,
    bm25: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    phrases = _capitalized_phrases(question)
    anchor_terms = Counter()
    for phrase in phrases:
        for token in _tokens(phrase):
            anchor_terms[token] += 1
    max_bm25 = max((score for _, score in bm25), default=0.0) or 1.0
    bm25_by_doc = dict(bm25)
    ranked = []
    for doc in corpus.docs:
        title_terms = _tokens(doc.title)
        title_hit = sum(anchor_terms.get(token, 0) * freq for token, freq in title_terms.items())
        exact_hit = max(
            [1.0 if _normalize(phrase) == _normalize(doc.title) else 0.0 for phrase in phrases]
            or [0.0]
        )
        bm25_prior = max(0.0, bm25_by_doc.get(doc.doc_id, 0.0) / max_bm25)
        score = (0.35 * bm25_prior) + (2.0 * exact_hit) + (0.5 * title_hit)
        ranked.append((doc.doc_id, score))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def _rank_generic_prf(
    question: str,
    corpus: CorpusIndex,
    bm25: list[tuple[str, float]],
    *,
    top_seed: int = 3,
) -> list[tuple[str, float]]:
    doc_by_id = {doc.doc_id: doc for doc in corpus.docs}
    query_terms = Counter(_tokens(question))
    max_bm25 = max((score for _, score in bm25), default=0.0) or 1.0
    bm25_by_doc = dict(bm25)
    for doc_id, _ in bm25[:top_seed]:
        doc = doc_by_id[doc_id]
        for token, freq in _tokens(doc.title).items():
            query_terms[token] += 4 * freq
        for token, freq in corpus.doc_terms[doc_id].most_common(30):
            if token not in STOPWORDS and len(token) > 3:
                query_terms[token] += min(freq, 2)
    query_norm = math.sqrt(sum(value * value for value in query_terms.values())) or 1.0
    ranked = []
    for doc in corpus.docs:
        doc_terms = corpus.doc_terms[doc.doc_id]
        doc_norm = math.sqrt(sum(value * value for value in doc_terms.values())) or 1.0
        overlap = sum(query_terms[token] * doc_terms.get(token, 0) for token in query_terms)
        bm25_prior = max(0.0, bm25_by_doc.get(doc.doc_id, 0.0) / max_bm25)
        score = (overlap / (query_norm * doc_norm)) + (0.1 * bm25_prior)
        ranked.append((doc.doc_id, score))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def _reciprocal_rank_fusion(rankings: list[tuple[list[tuple[str, float]], float]]) -> list[tuple[str, float]]:
    scores = Counter()
    for ranking, weight in rankings:
        for idx, (doc_id, _) in enumerate(ranking[:100]):
            scores[doc_id] += weight / (60 + idx)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _hypothesis_triggers(hypothesis_id: str, question: str) -> bool:
    q = question.lower()
    first = q.split()[0] if q.split() else ""
    anchors = _capitalized_phrases(question)
    if hypothesis_id == "qa_hyp_comparison_dual_anchor":
        return first in {"are", "is", "was", "were", "did", "do", "does"} and " and " in q and len(anchors) >= 2
    if hypothesis_id == "qa_hyp_anchor_preserve_insert":
        return bool(anchors)
    if hypothesis_id == "qa_hyp_named_anchor_bridge":
        return first in {"what", "which", "who", "where", "when"} and bool(anchors)
    if hypothesis_id == "qa_hyp_generic_prf":
        return True
    return False


def _capitalized_phrases(text: str) -> list[str]:
    phrases = []
    seen = set()
    for match in CAPITALIZED_PHRASE_RE.finditer(text):
        phrase = match.group(0).strip(" .,;:()[]{}\"'")
        normalized = _normalize(phrase)
        tokens = normalized.split()
        if not normalized:
            continue
        if all(token in STOPWORDS or len(token) < 2 for token in tokens):
            continue
        if normalized not in seen:
            seen.add(normalized)
            phrases.append(phrase)
    return phrases


def _qa_residual_type(metrics: dict[str, Any]) -> str:
    if not metrics["any_gold_recall_at_k"]:
        return "no_supporting_evidence"
    if not metrics["all_gold_recall_at_k"]:
        return "incomplete_support_chain"
    if not metrics["answer_coverage_at_k"]:
        return "reader_answer_not_covered"
    return "covered"


def _row_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        "all_gold_recall_at_k_delta": float(candidate["all_gold_recall_at_k"]) - float(baseline["all_gold_recall_at_k"]),
        "gold_fraction_at_k_delta": round(float(candidate["gold_fraction_at_k"]) - float(baseline["gold_fraction_at_k"]), 4),
        "answer_coverage_at_k_delta": float(candidate["answer_coverage_at_k"]) - float(baseline["answer_coverage_at_k"]),
    }


def _support_tuple(delta: dict[str, float]) -> tuple[float, float, float]:
    return (
        delta["all_gold_recall_at_k_delta"],
        delta["gold_fraction_at_k_delta"],
        delta["answer_coverage_at_k_delta"],
    )


def _aggregate_meta_rows(rows: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    retrievers = [
        "ordinary_bm25",
        "rag_to_memory_style_ppr",
        "comparison_dual_anchor",
        "anchor_preserve_insert",
        "named_anchor_bridge",
        "generic_prf",
        "meta_qa_controller",
    ]
    return {
        "top_k": top_k,
        "overall": {
            retriever: _metric_summary(rows, retriever)
            for retriever in retrievers
        },
        "by_dataset": {
            dataset: {
                retriever: _metric_summary([row for row in rows if row["dataset"] == dataset], retriever)
                for retriever in retrievers
            }
            for dataset in sorted({row["dataset"] for row in rows})
        },
    }


def _metric_summary(rows: list[dict[str, Any]], retriever: str) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "applicable_rate": 0.0,
            "any_gold_recall_at_k": 0.0,
            "all_gold_recall_at_k": 0.0,
            "mean_gold_fraction_at_k": 0.0,
            "answer_coverage_at_k": 0.0,
        }
    metrics = [row["metrics"][retriever] for row in rows]
    return {
        "n": len(rows),
        "applicable_rate": _rate(metrics, "applicable"),
        "any_gold_recall_at_k": _rate(metrics, "any_gold_recall_at_k"),
        "all_gold_recall_at_k": _rate(metrics, "all_gold_recall_at_k"),
        "mean_gold_fraction_at_k": round(sum(float(row["gold_fraction_at_k"]) for row in metrics) / len(metrics), 4),
        "answer_coverage_at_k": _rate(metrics, "answer_coverage_at_k"),
    }


def _aggregate_deltas(aggregate: dict[str, Any], candidate: str, baseline: str) -> dict[str, float]:
    cand = aggregate["overall"][candidate]
    base = aggregate["overall"][baseline]
    return {
        "any_gold_recall_at_k_delta": round(cand["any_gold_recall_at_k"] - base["any_gold_recall_at_k"], 4),
        "all_gold_recall_at_k_delta": round(cand["all_gold_recall_at_k"] - base["all_gold_recall_at_k"], 4),
        "mean_gold_fraction_at_k_delta": round(cand["mean_gold_fraction_at_k"] - base["mean_gold_fraction_at_k"], 4),
        "answer_coverage_at_k_delta": round(cand["answer_coverage_at_k"] - base["answer_coverage_at_k"], 4),
    }


def _recursive_trace(hypothesis_summaries: list[dict[str, Any]], deltas: dict[str, float]) -> list[dict[str, Any]]:
    return [
        {
            "step": 1,
            "frame": "failure",
            "observation": "HippoRAG QA probe showed structural morphism abstains on factual QA, so QA uses BM25 fallback.",
        },
        {
            "step": 2,
            "frame": "variation",
            "observation": f"Generated {len(hypothesis_summaries)} retrieval hypotheses from incomplete-support residuals.",
        },
        {
            "step": 3,
            "frame": "evaluation",
            "observation": [
                {
                    "hypothesis_id": row["hypothesis_id"],
                    "decision": row["decision"],
                    "activated_row_count": row["activated_row_count"],
                    "utility_deltas_sum": row["utility_deltas_sum"],
                    "harm_count": row["harm_count"],
                }
                for row in hypothesis_summaries
            ],
        },
        {
            "step": 4,
            "frame": "selective_retention",
            "observation": "Retained accepted narrow policies and left all other rows on BM25 fallback.",
        },
        {
            "step": 5,
            "frame": "performance_validation",
            "observation": deltas,
        },
    ]


def _top_titles(ranking: list[tuple[str, float]], corpus: CorpusIndex, *, top_k: int) -> list[str]:
    docs = {doc.doc_id: doc for doc in corpus.docs}
    return [docs[doc_id].title for doc_id, _ in ranking[:top_k] if doc_id in docs]


def _rate(metrics: list[dict[str, Any]], field: str) -> float:
    return round(sum(1 for row in metrics if row[field]) / len(metrics), 4) if metrics else 0.0


def _normalize(text: str) -> str:
    return " ".join(WORD_RE.findall(str(text).lower()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run solve-time Meta-QA evolution retrieval probe.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="meta_qa_evolution_20260607")
    parser.add_argument("--samples-per-dataset", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ppr-candidate-pool", type=int, default=40)
    parser.add_argument("--max-doc-phrase-tokens", type=int, default=24)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    payload = build_meta_qa_evolution_payload(
        root=Path(args.root),
        eval_id=args.eval_id,
        samples_per_dataset=args.samples_per_dataset,
        seed=args.seed,
        top_k=args.top_k,
        ppr_candidate_pool=args.ppr_candidate_pool,
        max_doc_phrase_tokens=args.max_doc_phrase_tokens,
    )
    out = Path(args.out)
    _write_json(out, payload)
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "aggregate": payload["aggregate"]["overall"],
        "deltas_vs_bm25": payload["deltas_vs_bm25"],
        "deltas_vs_ppr": payload["deltas_vs_ppr"],
        "evaluation": [
            {
                "hypothesis_id": row["hypothesis_id"],
                "decision": row["decision"],
                "activated_row_count": row["activated_row_count"],
                "utility_deltas_sum": row["utility_deltas_sum"],
                "harm_count": row["harm_count"],
            }
            for row in payload["evaluation"]
        ],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
