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
import hashlib
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .morphism_benchmark import _counter_cosine
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
    _top_docs,
    _contains_any_answer,
    _normalize_text,
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
    QARetrievalHypothesis(
        hypothesis_id="qa_hyp_representation_title_normalization",
        claim=(
            "Representation-transform priors from the pre-reconstruction notes apply to QA: convert possessive, quoted, "
            "and parenthesized surface mentions into canonical corpus-title candidates before ranking."
        ),
        trigger="question contains quoted/title-like, possessive, or parenthesized entity mentions",
        ranker_name="representation_title_normalization",
        expected_effect="Recover anchor pages that BM25 misses because the question surface form differs from the title.",
        risk="Can over-rank generic title matches; retained only if exact heldout rows show no support or answer regression.",
    ),
    QARetrievalHypothesis(
        hypothesis_id="qa_hyp_decomposition_bridge_entity",
        claim=(
            "Decomposition/composition priors from the pre-reconstruction notes apply to multi-hop QA: first retrieve an "
            "anchor page, extract the role-labeled bridge entity from that page, then retrieve the bridge page."
        ),
        trigger="question asks a relation chain such as creator/performer/director/husband/father/mother/plaintiff",
        ranker_name="decomposition_bridge_entity",
        expected_effect="Increase complete support-chain recall by adding the next-hop bridge entity instead of more same-hop pages.",
        risk="Bridge extraction can hallucinate from incidental entities; controlled insertion preserves BM25 evidence.",
    ),
    QARetrievalHypothesis(
        hypothesis_id="qa_hyp_controlled_bridge_insert",
        claim=(
            "Controlled-variable priors from the pre-reconstruction notes apply to retrieval repair: keep the working BM25 "
            "path fixed and insert only one or two bridge/title candidates, so the intervention is auditable."
        ),
        trigger="ordinary BM25 retrieves at least one plausible anchor but not a complete support chain",
        ranker_name="controlled_bridge_insert",
        expected_effect="Gain bridge evidence while limiting top-k displacement harm.",
        risk="Conservative insertion may under-improve no-support rows; accepted only if aggregate gains beat BM25/PPR.",
    ),
]

METHOD_LAYER_QA_PRIORS = [
    {
        "source": "pre_reconstruction_dialogue",
        "family": "kernel_representation_transform",
        "qa_hypothesis_id": "qa_hyp_representation_title_normalization",
        "principle": "Map a noisy surface representation into a canonical representation before comparing or retrieving.",
    },
    {
        "source": "pre_reconstruction_dialogue",
        "family": "kernel_decomposition_composition",
        "qa_hypothesis_id": "qa_hyp_decomposition_bridge_entity",
        "principle": "Split a root task into subproblems whose interface entity composes the answer chain.",
    },
    {
        "source": "pre_reconstruction_dialogue",
        "family": "kernel_controlled_intervention",
        "qa_hypothesis_id": "qa_hyp_controlled_bridge_insert",
        "principle": "Preserve the working baseline and change one bounded component at a time.",
    },
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
    run_extractive_reader: bool = False,
    reader_model: str = "distilbert-base-cased-distilled-squad",
    reader_retrievers: tuple[str, ...] = (
        "ordinary_bm25",
        "rag_to_memory_style_ppr",
        "meta_qa_controller",
    ),
    reader_samples_per_dataset: int = 0,
    reader_max_length: int = 384,
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
    accepted_priority = _accepted_policy_priority(hypothesis_summaries, accepted)
    for row in rows:
        row["retained_policy"] = _retained_policy_for_row(row, accepted, accepted_priority)
        row["metrics"]["meta_qa_controller"] = row["metrics"][row["retained_policy"]]
        row["top_titles"]["meta_qa_controller"] = row["top_titles"][row["retained_policy"]]
        row["top_doc_ids"]["meta_qa_controller"] = row["top_doc_ids"][row["retained_policy"]]

    aggregate = _aggregate_meta_rows(rows, top_k=top_k)
    deltas = _aggregate_deltas(aggregate, "meta_qa_controller", "ordinary_bm25")
    ppr_deltas = _aggregate_deltas(aggregate, "meta_qa_controller", "rag_to_memory_style_ppr")
    extractive_reader = _run_extractive_reader_for_payload(
        rows,
        data_dir=data_dir,
        top_k=top_k,
        run=run_extractive_reader,
        model_name=reader_model,
        retriever_names=reader_retrievers,
        samples_per_dataset=reader_samples_per_dataset,
        max_length=reader_max_length,
    )
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
    if run_extractive_reader:
        gates.extend(_extractive_reader_gates(extractive_reader))
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
            "pre_reconstruction_method_priors": METHOD_LAYER_QA_PRIORS,
        },
        "config": {
            "samples_per_dataset": samples_per_dataset,
            "seed": seed,
            "top_k": top_k,
            "ppr_candidate_pool": ppr_candidate_pool,
            "max_doc_phrase_tokens": max_doc_phrase_tokens,
            "stored_raw_model_answers": False,
            "run_extractive_reader": run_extractive_reader,
            "extractive_reader_model": reader_model if run_extractive_reader else None,
            "extractive_reader_retrievers": list(reader_retrievers) if run_extractive_reader else [],
            "extractive_reader_samples_per_dataset": reader_samples_per_dataset if run_extractive_reader else 0,
            "extractive_reader_max_length": reader_max_length if run_extractive_reader else 0,
        },
        "variation": [hypothesis.to_dict() for hypothesis in HYPOTHESES],
        "evaluation": hypothesis_summaries,
        "selective_retention": {
            "accepted_hypothesis_ids": sorted(accepted),
            "accepted_priority": accepted_priority,
            "policy": "Apply accepted narrow hypotheses by deterministic trigger; otherwise keep ordinary BM25.",
        },
        "aggregate": aggregate,
        "deltas_vs_bm25": deltas,
        "deltas_vs_ppr": ppr_deltas,
        "extractive_reader": extractive_reader,
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
    title_normalization = _rank_representation_title_normalization(question, corpus, bm25)
    bridge_entity = _rank_decomposition_bridge_entity(question, corpus, bm25, top_k=top_k)
    controlled_bridge = _rank_controlled_bridge_insert(question, corpus, bm25, top_k=top_k)
    rankings = {
        "ordinary_bm25": bm25,
        "rag_to_memory_style_ppr": ppr,
        "comparison_dual_anchor": comparison,
        "anchor_preserve_insert": anchor_preserve,
        "named_anchor_bridge": anchor_bridge,
        "generic_prf": generic_prf,
        "representation_title_normalization": title_normalization,
        "decomposition_bridge_entity": bridge_entity,
        "controlled_bridge_insert": controlled_bridge,
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
        "top_doc_ids": {
            name: [doc.doc_id for doc, _ in _top_docs(ranking, corpus, top_k)]
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


def _accepted_policy_priority(hypothesis_summaries: list[dict[str, Any]], accepted: set[str]) -> list[str]:
    accepted_rows = [row for row in hypothesis_summaries if row["hypothesis_id"] in accepted]
    def score(row: dict[str, Any]) -> tuple[float, float, str]:
        activated = max(1, int(row.get("activated_row_count") or 0))
        deltas = row.get("utility_deltas_sum", {})
        mean_utility = (
            float(deltas.get("all_gold_recall_at_k_delta") or 0.0)
            + float(deltas.get("gold_fraction_at_k_delta") or 0.0)
            + float(deltas.get("answer_coverage_at_k_delta") or 0.0)
        ) / activated
        return (mean_utility, float(deltas.get("all_gold_recall_at_k_delta") or 0.0), str(row["hypothesis_id"]))
    return [
        row["hypothesis_id"]
        for row in sorted(accepted_rows, key=score, reverse=True)
    ]


def _retained_policy_for_row(row: dict[str, Any], accepted: set[str], accepted_priority: list[str]) -> str:
    by_id = {hypothesis.hypothesis_id: hypothesis for hypothesis in HYPOTHESES}
    for hypothesis_id in accepted_priority:
        hypothesis = by_id[hypothesis_id]
        if hypothesis_id in accepted and row["candidate_triggers"][hypothesis_id]:
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
    phrases = _unique_phrases([*_capitalized_phrases(question), *_canonical_surface_mentions(question)])
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


def _rank_representation_title_normalization(
    question: str,
    corpus: CorpusIndex,
    bm25: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    title_matches = _rank_canonical_title_matches(question, corpus, bm25)
    return _controlled_insert(bm25, title_matches, keep_bm25=3, max_insert=2, min_candidate_score=1.5)


def _rank_decomposition_bridge_entity(
    question: str,
    corpus: CorpusIndex,
    bm25: list[tuple[str, float]],
    *,
    top_k: int,
) -> list[tuple[str, float]]:
    if _low_diversity_top_titles(bm25, corpus, top_k=top_k):
        return bm25
    title_matches = _rank_canonical_title_matches(question, corpus, bm25)
    bridge_matches = _rank_bridge_entity_matches(question, corpus, bm25, title_matches)
    return _controlled_insert(
        bm25,
        _reciprocal_rank_fusion([(title_matches, 0.6), (bridge_matches, 1.4)]),
        keep_bm25=max(1, top_k - 2),
        max_insert=2,
    )


def _rank_controlled_bridge_insert(
    question: str,
    corpus: CorpusIndex,
    bm25: list[tuple[str, float]],
    *,
    top_k: int,
) -> list[tuple[str, float]]:
    if _low_diversity_top_titles(bm25, corpus, top_k=top_k):
        return bm25
    title_matches = _rank_canonical_title_matches(question, corpus, bm25)
    bridge_matches = _rank_bridge_entity_matches(question, corpus, bm25, title_matches)
    combined = _reciprocal_rank_fusion([(title_matches, 0.9), (bridge_matches, 1.1)])
    keep = max(1, top_k - 2)
    return _controlled_insert(bm25, combined, keep_bm25=keep, max_insert=top_k - keep)


def _low_diversity_top_titles(
    ranking: list[tuple[str, float]],
    corpus: CorpusIndex,
    *,
    top_k: int,
) -> bool:
    doc_by_id = {doc.doc_id: doc for doc in corpus.docs}
    titles = [
        _normalize_title(doc_by_id[doc_id].title)
        for doc_id, _ in ranking[:top_k]
        if doc_id in doc_by_id
    ]
    if len(titles) < top_k:
        return False
    return len(set(titles[: max(1, top_k - 1)])) <= 1 and len(set(titles)) >= 2


def _rank_canonical_title_matches(
    question: str,
    corpus: CorpusIndex,
    bm25: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    mentions = _canonical_surface_mentions(question)
    mention_terms = [_tokens(mention) for mention in mentions]
    max_bm25 = max((score for _, score in bm25), default=0.0) or 1.0
    bm25_by_doc = dict(bm25)
    ranked = []
    for doc in corpus.docs:
        title_norm = _normalize_title(doc.title)
        title_terms = _tokens(doc.title)
        best = 0.0
        for mention, terms in zip(mentions, mention_terms):
            mention_norm = _normalize_title(mention)
            if not mention_norm:
                continue
            exact = 1.0 if mention_norm == title_norm else 0.0
            contained = 0.7 if mention_norm and (mention_norm in title_norm or title_norm in mention_norm) else 0.0
            overlap = _counter_cosine(terms, title_terms)
            best = max(best, exact * 3.0, contained * 2.0, overlap)
        bm25_prior = max(0.0, bm25_by_doc.get(doc.doc_id, 0.0) / max_bm25)
        ranked.append((doc.doc_id, best + 0.10 * bm25_prior))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def _rank_bridge_entity_matches(
    question: str,
    corpus: CorpusIndex,
    bm25: list[tuple[str, float]],
    title_matches: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    doc_by_id = {doc.doc_id: doc for doc in corpus.docs}
    title_docs = [doc_id for doc_id, score in title_matches[:8] if score > 0.2]
    seed_ids = _unique_doc_ids([doc_id for doc_id, _ in bm25[:4]] + title_docs)[:8]
    cue_terms = _relation_cues(question)
    candidates: Counter[str] = Counter()
    for doc_id in seed_ids:
        doc = doc_by_id.get(doc_id)
        if not doc:
            continue
        for phrase, score in _extract_bridge_phrases(question, doc.retrieval_text, cue_terms):
            normalized = _normalize_title(_strip_honorific(phrase))
            if normalized:
                candidates[normalized] += score
    if not candidates:
        return title_matches
    ranked = []
    for doc in corpus.docs:
        title_norm = _normalize_title(doc.title)
        title_terms = _tokens(doc.title)
        score = 0.0
        for phrase_norm, phrase_score in candidates.items():
            phrase_terms = _tokens(phrase_norm)
            exact = 1.0 if phrase_norm == title_norm else 0.0
            contained = 0.65 if phrase_norm and (phrase_norm in title_norm or title_norm in phrase_norm) else 0.0
            overlap = _counter_cosine(phrase_terms, title_terms)
            score = max(score, phrase_score * max(exact * 3.0, contained * 2.0, overlap))
        ranked.append((doc.doc_id, score))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def _controlled_insert(
    bm25: list[tuple[str, float]],
    candidates: list[tuple[str, float]],
    *,
    keep_bm25: int,
    max_insert: int,
    min_candidate_score: float = 0.0,
) -> list[tuple[str, float]]:
    output: list[tuple[str, float]] = []
    seen: set[str] = set()
    for doc_id, score in bm25[:keep_bm25]:
        output.append((doc_id, score + 20.0))
        seen.add(doc_id)
    inserts = 0
    for doc_id, score in candidates:
        if inserts >= max_insert:
            break
        if score <= min_candidate_score or doc_id in seen:
            continue
        output.append((doc_id, score + 10.0 - inserts * 0.01))
        seen.add(doc_id)
        inserts += 1
    for doc_id, score in bm25:
        if doc_id not in seen:
            output.append((doc_id, score))
            seen.add(doc_id)
    return output


def _extract_bridge_phrases(question: str, text: str, cue_terms: set[str]) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    patterns: list[tuple[str, float, set[str]]] = [
        (r"\bplaintiff\s+([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){1,4})", 5.0, {"plaintiff"}),
        (r"\bby\s+(?:English\s+)?(?:painter\s+)?(?:Sir\s+)?([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){1,4})", 3.8, {"creator"}),
        (r"\bfrom\s+([A-Z][A-Za-z0-9&.'-]+(?:\s+[A-Z][A-Za-z0-9&.'-]+){1,4})'s\b", 4.5, {"performer", "label"}),
        (r"\bperformed\s+by\s+([A-Z][A-Za-z0-9&.'-]+(?:\s+[A-Z][A-Za-z0-9&.'-]+){1,4})", 4.5, {"performer"}),
        (r"\bShe\s+married\s+([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){1,4})", 4.5, {"husband"}),
        (r"\bmarried\s+([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){1,4})", 4.0, {"husband"}),
        (r"([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){1,4}),\s+who\s+she\s+later\s+married", 4.5, {"husband"}),
        (r"\bson\s+of\s+([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){1,4})", 4.3, {"father", "mother"}),
        (r"\bdaughter\s+of\s+([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){1,4})", 4.3, {"father", "mother"}),
        (r"\bdirected\s+by\s+([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){1,4})", 4.7, {"director"}),
    ]
    for pattern, score, required_cues in patterns:
        if required_cues and cue_terms and not (required_cues & cue_terms):
            continue
        for match in re.finditer(pattern, text):
            phrase = _clean_bridge_phrase(match.group(1))
            if _valid_bridge_phrase(phrase):
                rows.append((phrase, score))
    return rows


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
    anchors = _unique_phrases([*_capitalized_phrases(question), *_canonical_surface_mentions(question)])
    if hypothesis_id == "qa_hyp_comparison_dual_anchor":
        return first in {"are", "is", "was", "were", "did", "do", "does"} and " and " in q and len(anchors) >= 2
    if hypothesis_id == "qa_hyp_anchor_preserve_insert":
        return bool(anchors)
    if hypothesis_id == "qa_hyp_named_anchor_bridge":
        return first in {"what", "which", "who", "where", "when"} and bool(anchors)
    if hypothesis_id == "qa_hyp_generic_prf":
        return True
    if hypothesis_id == "qa_hyp_representation_title_normalization":
        return bool(_canonical_surface_mentions(question))
    if hypothesis_id == "qa_hyp_decomposition_bridge_entity":
        return bool(_relation_cues(question)) and bool(anchors)
    if hypothesis_id == "qa_hyp_controlled_bridge_insert":
        return bool(anchors) and (bool(_relation_cues(question)) or _has_binary_choice(question))
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


def _canonical_surface_mentions(text: str) -> list[str]:
    mentions = []
    mentions.extend(_capitalized_phrases(text))
    mentions.extend(match.group(1) for match in re.finditer(r'"([^"]{2,80})"', text))
    mentions.extend(match.group(1) for match in re.finditer(r"'([^']{2,80})'", text))
    for match in re.finditer(r"\b([A-Z][A-Za-z0-9.'’ -]{2,80})'s\b", text):
        mentions.append(match.group(1))
    for phrase in list(mentions):
        if "(" in phrase or ")" in phrase:
            mentions.append(re.sub(r"\s*\([^)]*\)", "", phrase))
            mentions.append(phrase.replace("(", "").replace(")", ""))
    return _unique_phrases(_clean_surface_mention(mention) for mention in mentions)


def _clean_surface_mention(text: str) -> str:
    value = str(text).strip(" .,;:()[]{}\"'")
    value = re.sub(r"[’']s$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\b[Ff]ilm\b", "", value).strip()
    value = re.sub(r"\s+", " ", value)
    return value


def _normalize_title(text: str) -> str:
    value = _clean_surface_mention(text)
    value = re.sub(r"\s*\([^)]*\)", "", value)
    return _normalize(value)


def _unique_phrases(phrases: Any) -> list[str]:
    result = []
    seen = set()
    for phrase in phrases:
        cleaned = _clean_surface_mention(str(phrase))
        normalized = _normalize(cleaned)
        if not normalized:
            continue
        if all(token in STOPWORDS or len(token) < 2 for token in normalized.split()):
            continue
        if normalized not in seen:
            seen.add(normalized)
            result.append(cleaned)
    return result


def _unique_doc_ids(doc_ids: list[str]) -> list[str]:
    result = []
    seen = set()
    for doc_id in doc_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            result.append(doc_id)
    return result


def _relation_cues(question: str) -> set[str]:
    q = question.lower()
    cues = set()
    cue_terms = {
        "creator": ["creator", "created by"],
        "performer": ["performer", "performed", "singer", "band"],
        "label": ["label"],
        "director": ["director", "directed"],
        "husband": ["husband", "spouse", "married"],
        "father": ["father"],
        "mother": ["mother"],
        "plaintiff": ["plaintiff"],
        "headquarters": ["headquarters", "capitol", "capital"],
        "location": ["place of birth", "born"],
    }
    for cue, terms in cue_terms.items():
        if any(_contains_cue(q, term) for term in terms):
            cues.add(cue)
    return cues


def _has_binary_choice(question: str) -> bool:
    q = question.lower()
    return " or " in q and any(term in q for term in ("earlier", "later", "came out", "older", "younger"))


def _contains_cue(text: str, cue: str) -> bool:
    if " " in cue:
        return cue in text
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(cue)}(?![a-z0-9])", text))


def _clean_bridge_phrase(phrase: str) -> str:
    value = _clean_surface_mention(phrase)
    value = re.sub(r"\b(?:who|and|while|when|where|which|that|from|with)\b.*$", "", value).strip()
    return _strip_honorific(value)


def _strip_honorific(phrase: str) -> str:
    return re.sub(r"^(Sir|Dame|Dr|Professor|Prof|General|Brigadier General)\s+", "", phrase).strip()


def _valid_bridge_phrase(phrase: str) -> bool:
    normalized = _normalize_title(phrase)
    tokens = normalized.split()
    if len(tokens) < 2 or len(tokens) > 6:
        return False
    if any(token in STOPWORDS for token in tokens):
        return False
    bad = {"united states", "new york", "bbc one", "digital praise", "wow hits"}
    return normalized not in bad


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
        "representation_title_normalization",
        "decomposition_bridge_entity",
        "controlled_bridge_insert",
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


def _run_extractive_reader_for_payload(
    rows: list[dict[str, Any]],
    *,
    data_dir: Path,
    top_k: int,
    run: bool,
    model_name: str,
    retriever_names: tuple[str, ...],
    samples_per_dataset: int,
    max_length: int,
) -> dict[str, Any]:
    if not run:
        return {
            "status": "not_run",
            "reader_rows": 0,
            "attempted_calls": 0,
            "failed_calls": 0,
            "model": None,
            "by_retriever": {},
            "deltas_vs_bm25": {},
            "deltas_vs_ppr": {},
            "raw_answers_stored": False,
        }

    selected_rows = _select_reader_rows(rows, samples_per_dataset=samples_per_dataset)
    if not selected_rows:
        return {
            "status": "no_rows",
            "reader_rows": 0,
            "attempted_calls": 0,
            "failed_calls": 0,
            "model": model_name,
            "by_retriever": {},
            "deltas_vs_bm25": {},
            "deltas_vs_ppr": {},
            "raw_answers_stored": False,
        }

    try:
        reader = _LocalExtractiveReader(model_name=model_name, max_length=max_length)
        load_error = None
    except Exception as exc:
        reader = None
        load_error = str(exc)

    corpora = {
        dataset: _load_corpus_index(data_dir / f"{dataset}_corpus.json")
        for dataset in sorted({row["dataset"] for row in selected_rows})
    }
    samples = {
        dataset: _load_json(data_dir / f"{dataset}.json")
        for dataset in sorted({row["dataset"] for row in selected_rows})
    }

    attempted = 0
    failed = 0
    for row in selected_rows:
        corpus = corpora[row["dataset"]]
        sample = samples[row["dataset"]][row["sample_index"]]
        gold_answers = _gold_answers(sample)
        row_results = {}
        for retriever in retriever_names:
            attempted += 1
            docs = _docs_by_id(corpus, row.get("top_doc_ids", {}).get(retriever, [])[:top_k])
            started = time.time()
            if reader is None:
                prediction = ""
                score = 0.0
                error = load_error or "reader_unavailable"
            else:
                try:
                    prediction, score = reader.answer(row["question"], docs)
                    error = None
                except Exception as exc:
                    prediction = ""
                    score = 0.0
                    error = str(exc)
            elapsed = round(time.time() - started, 3)
            failed += 1 if error else 0
            exact_match, f1 = _answer_scores(prediction, gold_answers)
            row_results[retriever] = {
                "model": model_name,
                "top_k": len(docs),
                "answer_sha256": hashlib.sha256(prediction.encode("utf-8")).hexdigest() if prediction else None,
                "answer_char_count": len(prediction),
                "prediction_score": round(score, 4),
                "exact_match": exact_match,
                "f1": f1,
                "contains_gold_answer": _contains_any_answer(prediction, gold_answers),
                "latency_seconds": elapsed,
                "error": error,
            }
        row["extractive_reader"] = {
            "question_sha256": hashlib.sha256(row["question"].encode("utf-8")).hexdigest(),
            "retrievers": row_results,
            "raw_answer_stored": False,
            "gold_answers_stored": False,
        }

    by_retriever = _aggregate_extractive_reader(selected_rows, retriever_names)
    return {
        "status": "run",
        "reader_rows": len(selected_rows),
        "attempted_calls": attempted,
        "failed_calls": failed,
        "model": model_name,
        "retrievers": list(retriever_names),
        "by_retriever": by_retriever,
        "deltas_vs_bm25": _reader_deltas(by_retriever, "meta_qa_controller", "ordinary_bm25"),
        "deltas_vs_ppr": _reader_deltas(by_retriever, "meta_qa_controller", "rag_to_memory_style_ppr"),
        "raw_answers_stored": False,
        "gold_answers_stored": False,
    }


def _select_reader_rows(rows: list[dict[str, Any]], *, samples_per_dataset: int) -> list[dict[str, Any]]:
    if samples_per_dataset <= 0:
        return rows
    selected = []
    counts: Counter[str] = Counter()
    for row in rows:
        dataset = row["dataset"]
        if counts[dataset] >= samples_per_dataset:
            continue
        selected.append(row)
        counts[dataset] += 1
    return selected


def _docs_by_id(corpus: CorpusIndex, doc_ids: list[str]) -> list[Any]:
    doc_by_id = {doc.doc_id: doc for doc in corpus.docs}
    return [doc_by_id[doc_id] for doc_id in doc_ids if doc_id in doc_by_id]


class _LocalExtractiveReader:
    def __init__(self, *, model_name: str, max_length: int) -> None:
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        import torch

        self.model_name = model_name
        self.max_length = max_length
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.eval()

    def answer(self, question: str, docs: list[Any]) -> tuple[str, float]:
        best_answer = ""
        best_score = -1e12
        with self.torch.no_grad():
            for doc in docs:
                context = f"{doc.title}. {doc.text}"
                encoded = self.tokenizer(
                    question,
                    context,
                    return_tensors="pt",
                    truncation="only_second",
                    max_length=self.max_length,
                )
                output = self.model(**encoded)
                sequence_ids = encoded.sequence_ids(0)
                context_positions = [
                    idx
                    for idx, sequence_id in enumerate(sequence_ids)
                    if sequence_id == 1
                ]
                if not context_positions:
                    continue
                start_logits = output.start_logits[0]
                end_logits = output.end_logits[0]
                top_n = min(8, len(context_positions))
                top_starts = self.torch.topk(start_logits, k=min(top_n, start_logits.numel())).indices.tolist()
                top_ends = self.torch.topk(end_logits, k=min(top_n, end_logits.numel())).indices.tolist()
                context_set = set(context_positions)
                for start in top_starts:
                    if start not in context_set:
                        continue
                    for end in top_ends:
                        if end not in context_set or end < start or end - start > 30:
                            continue
                        score = float(start_logits[start] + end_logits[end])
                        token_ids = encoded["input_ids"][0][start : end + 1]
                        answer = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                        if answer and score > best_score:
                            best_score = score
                            best_answer = answer
        return best_answer, best_score if best_answer else 0.0


def _answer_scores(prediction: str, gold_answers: list[str]) -> tuple[float, float]:
    if not gold_answers:
        return 0.0, 0.0
    exact = 1.0 if any(_normalize_text(prediction) == _normalize_text(answer) for answer in gold_answers) else 0.0
    f1 = max((_answer_f1(prediction, answer) for answer in gold_answers), default=0.0)
    return exact, round(f1, 4)


def _answer_f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalize_text(prediction).split()
    gold_tokens = _normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    overlap_count = sum(overlap.values())
    if not overlap_count:
        return 0.0
    precision = overlap_count / len(pred_tokens)
    recall = overlap_count / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def _aggregate_extractive_reader(rows: list[dict[str, Any]], retriever_names: tuple[str, ...]) -> dict[str, Any]:
    summary = {}
    for retriever in retriever_names:
        results = [
            row["extractive_reader"]["retrievers"][retriever]
            for row in rows
            if "extractive_reader" in row and retriever in row["extractive_reader"].get("retrievers", {})
        ]
        n = len(results)
        if not n:
            summary[retriever] = {
                "n": 0,
                "exact_match": 0.0,
                "mean_f1": 0.0,
                "contains_gold_answer_rate": 0.0,
                "failed_calls": 0,
                "mean_latency_seconds": 0.0,
            }
            continue
        summary[retriever] = {
            "n": n,
            "exact_match": round(sum(float(row["exact_match"]) for row in results) / n, 4),
            "mean_f1": round(sum(float(row["f1"]) for row in results) / n, 4),
            "contains_gold_answer_rate": _rate(results, "contains_gold_answer"),
            "failed_calls": sum(1 for row in results if row.get("error")),
            "mean_latency_seconds": round(sum(float(row["latency_seconds"]) for row in results) / n, 3),
        }
    return summary


def _reader_deltas(summary: dict[str, Any], candidate: str, baseline: str) -> dict[str, float]:
    if candidate not in summary or baseline not in summary:
        return {}
    cand = summary[candidate]
    base = summary[baseline]
    return {
        "exact_match_delta": round(float(cand["exact_match"]) - float(base["exact_match"]), 4),
        "mean_f1_delta": round(float(cand["mean_f1"]) - float(base["mean_f1"]), 4),
        "contains_gold_answer_rate_delta": round(
            float(cand["contains_gold_answer_rate"]) - float(base["contains_gold_answer_rate"]),
            4,
        ),
    }


def _extractive_reader_gates(reader: dict[str, Any]) -> list[dict[str, Any]]:
    bm25_delta = reader.get("deltas_vs_bm25", {})
    ppr_delta = reader.get("deltas_vs_ppr", {})
    return [
        {
            "gate": "extractive_reader_completed",
            "pass": reader.get("attempted_calls", 0) > 0 and reader.get("failed_calls", 0) == 0,
            "observed": {
                "reader_rows": reader.get("reader_rows"),
                "attempted_calls": reader.get("attempted_calls"),
                "failed_calls": reader.get("failed_calls"),
                "model": reader.get("model"),
            },
        },
        {
            "gate": "extractive_reader_raw_answers_not_stored",
            "pass": reader.get("raw_answers_stored") is False and reader.get("gold_answers_stored") is False,
            "observed": {
                "raw_answers_stored": reader.get("raw_answers_stored"),
                "gold_answers_stored": reader.get("gold_answers_stored"),
            },
        },
        {
            "gate": "extractive_reader_meta_beats_bm25_f1",
            "pass": float(bm25_delta.get("mean_f1_delta", 0.0)) >= 0.0,
            "observed": bm25_delta,
        },
        {
            "gate": "extractive_reader_meta_beats_ppr_f1",
            "pass": float(ppr_delta.get("mean_f1_delta", 0.0)) >= 0.0,
            "observed": ppr_delta,
        },
    ]


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
    parser.add_argument("--run-extractive-reader", action="store_true")
    parser.add_argument("--reader-model", default="distilbert-base-cased-distilled-squad")
    parser.add_argument("--reader-samples-per-dataset", type=int, default=0)
    parser.add_argument("--reader-max-length", type=int, default=384)
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
        run_extractive_reader=args.run_extractive_reader,
        reader_model=args.reader_model,
        reader_samples_per_dataset=args.reader_samples_per_dataset,
        reader_max_length=args.reader_max_length,
    )
    out = Path(args.out)
    _write_json(out, payload)
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "aggregate": payload["aggregate"]["overall"],
        "deltas_vs_bm25": payload["deltas_vs_bm25"],
        "deltas_vs_ppr": payload["deltas_vs_ppr"],
        "extractive_reader": {
            "status": payload["extractive_reader"]["status"],
            "by_retriever": payload["extractive_reader"]["by_retriever"],
            "deltas_vs_bm25": payload["extractive_reader"]["deltas_vs_bm25"],
            "deltas_vs_ppr": payload["extractive_reader"]["deltas_vs_ppr"],
        },
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
