"""Small QA-benchmark probe against HippoRAG reproduction datasets.

The morphism benchmark is intentionally narrow.  This probe samples actual
HotpotQA, MuSiQue, and 2WikiMultihopQA files from the local HippoRAG repo and
checks retrieval/answer-coverage behavior before we over-claim that morphism
results transfer to full QA.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .morphism_benchmark import _counter_cosine, _tokens


DEFAULT_DATA_DIR = Path("reference/repos/HippoRAG/reproduce/dataset")
DEFAULT_OUT = Path("phase four/assumption_graph/paper_readiness_20260604/hipporag_qa_probe_20260606.json")
WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class CorpusDoc:
    doc_id: str
    title: str
    text: str

    @property
    def retrieval_text(self) -> str:
        return f"{self.title}\n{self.text}"


@dataclass(frozen=True)
class CorpusIndex:
    docs: list[CorpusDoc]
    doc_terms: dict[str, Counter]
    document_frequency: Counter
    avg_doc_len: float


def build_hipporag_qa_probe_payload(
    *,
    root: Path,
    eval_id: str | None = None,
    datasets: tuple[str, ...] = ("hotpotqa", "musique", "2wikimultihopqa"),
    samples_per_dataset: int = 5,
    seed: int = 20260606,
    top_k: int = 5,
    ppr_candidate_pool: int = 40,
    max_doc_phrase_tokens: int = 24,
    run_reader: bool = False,
    reader_samples_per_dataset: int = 1,
    reader_retrievers: tuple[str, ...] = ("ordinary_bm25", "rag_to_memory_style_ppr"),
) -> dict[str, Any]:
    data_dir = _resolve(root, DEFAULT_DATA_DIR)
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        samples = _load_json(data_dir / f"{dataset}.json")
        corpus = _load_corpus_index(data_dir / f"{dataset}_corpus.json")
        sample_indices = _sample_indices(len(samples), samples_per_dataset=samples_per_dataset, seed=seed + len(rows))
        for sample_index in sample_indices:
            sample = samples[sample_index]
            row = _evaluate_sample(
                dataset=dataset,
                sample_index=sample_index,
                sample=sample,
                corpus=corpus,
                top_k=top_k,
                ppr_candidate_pool=ppr_candidate_pool,
                max_doc_phrase_tokens=max_doc_phrase_tokens,
            )
            if run_reader and _reader_row_enabled(rows, row, reader_samples_per_dataset=reader_samples_per_dataset):
                row["reader_qa"] = _run_reader_for_row(
                    row,
                    corpus=corpus,
                    retriever_names=reader_retrievers,
                    top_k=top_k,
                )
            rows.append(row)
    aggregate = _aggregate(rows, top_k=top_k)
    reader_summary = _aggregate_reader(rows)
    risk = _qa_transfer_risk_summary(aggregate)
    gates = [
        {
            "gate": "uses_real_hipporag_qa_files",
            "pass": all((_resolve(root, DEFAULT_DATA_DIR) / f"{dataset}.json").exists() for dataset in datasets),
            "observed": {
                "data_dir": _display_path(root.resolve(), data_dir),
                "datasets": list(datasets),
            },
        },
        {
            "gate": "sample_count",
            "pass": len(rows) >= len(datasets) * samples_per_dataset,
            "observed": len(rows),
        },
        {
            "gate": "risk_probe_not_morphism_claim_reuse",
            "pass": aggregate["overall"]["structural_morphism_direct"]["applicable_rate"] <= 0.20,
            "observed": aggregate["overall"]["structural_morphism_direct"],
        },
        {
            "gate": "fallback_retrieval_has_nonzero_signal",
            "pass": aggregate["overall"]["ordinary_bm25"]["any_gold_recall_at_k"] > 0.0,
            "observed": aggregate["overall"]["ordinary_bm25"],
        },
        {
            "gate": "qa_probe_records_negative_or_mixed_transfer_risk",
            "pass": risk["risk_level"] in {"medium", "high"},
            "observed": risk,
        },
    ]
    if run_reader:
        gates.append({
            "gate": "reader_qa_completed",
            "pass": reader_summary["attempted_calls"] > 0 and reader_summary["failed_calls"] == 0,
            "observed": reader_summary,
        })
    return {
        "eval_id": eval_id or "hipporag_qa_probe_20260606",
        "eval_kind": "hipporag_full_qa_benchmark_small_probe",
        "source_alignment": {
            "local_repo": "reference/repos/HippoRAG",
            "datasets": list(datasets),
            "metric_alignment": (
                "HippoRAG 2 reports passage recall for retrieval and QA F1/EM for reader output. "
                "This cheap probe measures supporting-passage recall@k plus gold-answer string coverage@k."
            ),
            "reader_qa_status": "run" if run_reader else "not_run",
        },
        "config": {
            "samples_per_dataset": samples_per_dataset,
            "seed": seed,
            "top_k": top_k,
            "ppr_candidate_pool": ppr_candidate_pool,
            "max_doc_phrase_tokens": max_doc_phrase_tokens,
            "run_reader": run_reader,
            "reader_samples_per_dataset": reader_samples_per_dataset if run_reader else 0,
            "reader_retrievers": list(reader_retrievers) if run_reader else [],
            "retrievers": [
                "ordinary_bm25",
                "rag_to_memory_style_ppr",
                "current_safe_policy_bm25_fallback",
                "structural_morphism_direct",
            ],
        },
        "aggregate": aggregate,
        "reader_qa_summary": reader_summary,
        "qa_transfer_risk": risk,
        "gates": gates,
        "failed_gates": [gate["gate"] for gate in gates if not gate["pass"]],
        "pass": all(gate["pass"] for gate in gates),
        "rows": rows,
    }


def _evaluate_sample(
    *,
    dataset: str,
    sample_index: int,
    sample: dict[str, Any],
    corpus: CorpusIndex,
    top_k: int,
    ppr_candidate_pool: int,
    max_doc_phrase_tokens: int,
) -> dict[str, Any]:
    query = sample["question"]
    gold_titles = _gold_titles(sample)
    gold_answers = _gold_answers(sample)
    bm25_ranking = _rank_bm25(query, corpus)
    ppr_ranking = _rank_rag_to_memory_ppr(
        query,
        corpus,
        bm25_ranking[:ppr_candidate_pool],
        max_doc_phrase_tokens=max_doc_phrase_tokens,
    )
    current_safe_ranking = bm25_ranking
    structural_direct_ranking: list[tuple[str, float]] = []
    retrievers = {
        "ordinary_bm25": bm25_ranking,
        "rag_to_memory_style_ppr": ppr_ranking,
        "current_safe_policy_bm25_fallback": current_safe_ranking,
        "structural_morphism_direct": structural_direct_ranking,
    }
    metrics = {
        name: _retrieval_metrics(ranking, corpus, gold_titles, gold_answers, top_k=top_k)
        for name, ranking in retrievers.items()
    }
    metrics["structural_morphism_direct"]["applicable"] = False
    return {
        "dataset": dataset,
        "sample_index": sample_index,
        "sample_id": str(sample.get("_id") or sample.get("id") or sample_index),
        "question": query,
        "answer_alias_count": len(gold_answers),
        "gold_answers": gold_answers,
        "gold_titles": sorted(gold_titles),
        "metrics": metrics,
        "top_titles": {
            name: [doc.title for doc, _ in _top_docs(ranking, corpus, top_k)]
            for name, ranking in retrievers.items()
        },
        "top_doc_ids": {
            name: [doc.doc_id for doc, _ in _top_docs(ranking, corpus, top_k)]
            for name, ranking in retrievers.items()
        },
    }


def _rank_bm25(query: str, corpus: CorpusIndex) -> list[tuple[str, float]]:
    query_terms = list(_tokens(query))
    ranked = []
    total_docs = len(corpus.docs)
    for doc in corpus.docs:
        terms = corpus.doc_terms[doc.doc_id]
        doc_len = sum(terms.values()) or 1
        score = 0.0
        for term in query_terms:
            freq = terms.get(term, 0)
            if not freq:
                continue
            containing = corpus.document_frequency.get(term, 0)
            idf = math.log((total_docs - containing + 0.5) / (containing + 0.5) + 1.0)
            score += idf * (freq * 2.5) / (freq + 1.5 * (0.25 + 0.75 * doc_len / corpus.avg_doc_len))
        ranked.append((doc.doc_id, score))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def _rank_rag_to_memory_ppr(
    query: str,
    corpus: CorpusIndex,
    candidate_ranking: list[tuple[str, float]],
    *,
    max_doc_phrase_tokens: int,
) -> list[tuple[str, float]]:
    doc_by_id = {doc.doc_id: doc for doc in corpus.docs}
    graph: dict[str, dict[str, float]] = defaultdict(dict)
    reset = Counter()
    query_terms = _expanded_tokens(query)
    max_bm25 = max((score for _, score in candidate_ranking), default=0.0) or 1.0
    for doc_id, bm25_score in candidate_ranking:
        doc = doc_by_id[doc_id]
        passage = _passage_node(doc_id)
        _ensure_node(graph, passage)
        reset[passage] += max(0.0, bm25_score / max_bm25) * 0.05
        title_terms = _expanded_tokens(doc.title)
        text_terms = _limited_counter_terms(corpus.doc_terms[doc.doc_id], query_terms, max_terms=max_doc_phrase_tokens)
        for token, freq in title_terms.items():
            node = _phrase_node(token)
            _add_edge(graph, passage, node, min(1.5, 0.8 + 0.1 * freq))
            if token in query_terms:
                reset[node] += query_terms[token] * 1.2
        for token, freq in text_terms.items():
            node = _phrase_node(token)
            _add_edge(graph, passage, node, min(0.35, 0.04 + 0.03 * math.log1p(freq)))
            if token in query_terms:
                reset[node] += query_terms[token]
    if not reset:
        return candidate_ranking
    ppr = _personalized_pagerank(graph, dict(reset), restart_probability=0.15, iterations=45)
    ranked = [
        (doc_id, float(ppr.get(_passage_node(doc_id), 0.0)))
        for doc_id, _ in candidate_ranking
    ]
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def _personalized_pagerank(
    graph: dict[str, dict[str, float]],
    reset: dict[str, float],
    *,
    restart_probability: float,
    iterations: int,
) -> dict[str, float]:
    nodes = sorted(graph)
    reset = {node: score for node, score in reset.items() if node in graph and score > 0}
    total_reset = sum(reset.values())
    if not nodes or not total_reset:
        return {}
    reset = {node: score / total_reset for node, score in reset.items()}
    rank = {node: 1.0 / len(nodes) for node in nodes}
    for _ in range(iterations):
        next_rank = {node: restart_probability * reset.get(node, 0.0) for node in nodes}
        for source in nodes:
            total_weight = sum(graph[source].values())
            if not total_weight:
                continue
            share = (1.0 - restart_probability) * rank[source] / total_weight
            for target, weight in graph[source].items():
                next_rank[target] += share * weight
        norm = sum(next_rank.values())
        if norm:
            next_rank = {node: score / norm for node, score in next_rank.items()}
        rank = next_rank
    return rank


def _retrieval_metrics(
    ranking: list[tuple[str, float]],
    corpus: CorpusIndex,
    gold_titles: set[str],
    gold_answers: list[str],
    *,
    top_k: int,
) -> dict[str, Any]:
    top_docs = [doc for doc, _ in _top_docs(ranking, corpus, top_k)]
    top_titles = {doc.title for doc in top_docs}
    matched_titles = top_titles & gold_titles
    joined_text = "\n".join(doc.retrieval_text for doc in top_docs)
    return {
        "applicable": bool(ranking),
        "any_gold_recall_at_k": bool(matched_titles),
        "all_gold_recall_at_k": bool(gold_titles) and gold_titles <= top_titles,
        "gold_fraction_at_k": round(len(matched_titles) / len(gold_titles), 4) if gold_titles else 0.0,
        "answer_coverage_at_k": _contains_any_answer(joined_text, gold_answers),
        "matched_gold_titles": sorted(matched_titles),
    }


def _aggregate(rows: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    retriever_names = sorted(rows[0]["metrics"]) if rows else []
    result = {
        "top_k": top_k,
        "overall": {},
        "by_dataset": {},
    }
    for retriever in retriever_names:
        result["overall"][retriever] = _metric_summary(rows, retriever)
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        result["by_dataset"][dataset] = {
            retriever: _metric_summary(dataset_rows, retriever)
            for retriever in retriever_names
        }
    return result


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


def _qa_transfer_risk_summary(aggregate: dict[str, Any]) -> dict[str, Any]:
    overall = aggregate["overall"]
    bm25 = overall["ordinary_bm25"]
    ppr = overall["rag_to_memory_style_ppr"]
    structural_direct = overall["structural_morphism_direct"]
    current_safe = overall["current_safe_policy_bm25_fallback"]
    ppr_minus_bm25 = round(
        float(ppr["mean_gold_fraction_at_k"]) - float(bm25["mean_gold_fraction_at_k"]),
        4,
    )
    if structural_direct["applicable_rate"] == 0.0 and current_safe["mean_gold_fraction_at_k"] <= 0.5:
        level = "high"
    elif ppr_minus_bm25 < 0.05:
        level = "medium"
    else:
        level = "low"
    return {
        "risk_level": level,
        "why": (
            "The morphism layer is not a direct factual-QA retriever; on full QA samples the current safe policy "
            "falls back to ordinary retrieval.  Strong morphism benchmark margins should not be projected to QA "
            "without a dedicated reader/retrieval experiment."
        ),
        "rag_to_memory_ppr_minus_bm25_mean_gold_fraction_at_k": ppr_minus_bm25,
        "current_safe_policy_mean_gold_fraction_at_k": current_safe["mean_gold_fraction_at_k"],
        "structural_morphism_direct_applicable_rate": structural_direct["applicable_rate"],
    }


def _reader_row_enabled(rows: list[dict[str, Any]], row: dict[str, Any], *, reader_samples_per_dataset: int) -> bool:
    if reader_samples_per_dataset <= 0:
        return False
    existing = sum(1 for existing_row in rows if existing_row["dataset"] == row["dataset"] and "reader_qa" in existing_row)
    return existing < reader_samples_per_dataset


def _run_reader_for_row(
    row: dict[str, Any],
    *,
    corpus: CorpusIndex,
    retriever_names: tuple[str, ...],
    top_k: int,
) -> dict[str, Any]:
    client = _ReaderClient.from_env()
    results = {}
    for retriever in retriever_names:
        doc_ids = row.get("top_doc_ids", {}).get(retriever, [])[:top_k]
        docs = _docs_by_id(corpus, doc_ids)
        started = time.time()
        try:
            answer = client.answer(row["question"], docs)
            error = None
        except Exception as exc:
            answer = ""
            error = str(exc)
        elapsed = round(time.time() - started, 3)
        results[retriever] = {
            "model": client.model,
            "base_url_configured": bool(client.base_url),
            "top_k": len(docs),
            "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest() if answer else None,
            "answer_char_count": len(answer),
            "contains_gold_answer": _contains_any_answer(answer, _gold_answer_list_from_row(row)),
            "latency_seconds": elapsed,
            "error": error,
        }
    return {
        "question_sha256": hashlib.sha256(row["question"].encode("utf-8")).hexdigest(),
        "retrievers": results,
    }


def _aggregate_reader(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reader_rows = [row for row in rows if "reader_qa" in row]
    retriever_counts: dict[str, dict[str, Any]] = {}
    attempted = 0
    failed = 0
    for row in reader_rows:
        for retriever, result in row["reader_qa"].get("retrievers", {}).items():
            attempted += 1
            failed += 1 if result.get("error") else 0
            bucket = retriever_counts.setdefault(retriever, {
                "n": 0,
                "contains_gold_answer_rate": 0.0,
                "failed_calls": 0,
                "mean_latency_seconds": 0.0,
            })
            bucket["n"] += 1
            bucket["contains_gold_answer_rate"] += 1.0 if result.get("contains_gold_answer") else 0.0
            bucket["failed_calls"] += 1 if result.get("error") else 0
            bucket["mean_latency_seconds"] += float(result.get("latency_seconds") or 0.0)
    for bucket in retriever_counts.values():
        n = max(1, int(bucket["n"]))
        bucket["contains_gold_answer_rate"] = round(bucket["contains_gold_answer_rate"] / n, 4)
        bucket["mean_latency_seconds"] = round(bucket["mean_latency_seconds"] / n, 3)
    return {
        "reader_rows": len(reader_rows),
        "attempted_calls": attempted,
        "failed_calls": failed,
        "by_retriever": retriever_counts,
        "raw_answers_stored": False,
    }


def _docs_by_id(corpus: CorpusIndex, doc_ids: list[str]) -> list[CorpusDoc]:
    doc_by_id = {doc.doc_id: doc for doc in corpus.docs}
    return [doc_by_id[doc_id] for doc_id in doc_ids if doc_id in doc_by_id]


def _gold_answer_list_from_row(row: dict[str, Any]) -> list[str]:
    return [str(item) for item in row.get("gold_answers", [])]


class _ReaderClient:
    def __init__(self, *, model: str, base_url: str, api_key: str, timeout: float = 90.0) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "_ReaderClient":
        key = os.environ.get("RUOLI_GPT_KEY") or os.environ.get("GPT5_API_KEY")
        if not key:
            raise RuntimeError("RUOLI_GPT_KEY or GPT5_API_KEY is required for --run-reader")
        base_url = os.environ.get("GPT5_BASE_URL") or os.environ.get("RUOLI_BASE_URL") or "https://ruoli.dev"
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        model = os.environ.get("GPT55_MODEL") or "gpt-5.5"
        return cls(model=model, base_url=base_url, api_key=key)

    def answer(self, question: str, docs: list[CorpusDoc]) -> str:
        import requests

        context = "\n\n".join(
            f"[{idx + 1}] {doc.title}\n{doc.text[:1400]}"
            for idx, doc in enumerate(docs)
        )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer the question using only the provided context. "
                        "Return the shortest exact answer phrase if possible. "
                        "If the context is insufficient, answer 'insufficient context'."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
                },
            ],
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"]).strip()


def _rate(metrics: list[dict[str, Any]], field: str) -> float:
    return round(sum(1 for row in metrics if row[field]) / len(metrics), 4) if metrics else 0.0


def _top_docs(ranking: list[tuple[str, float]], corpus: CorpusIndex, top_k: int) -> list[tuple[CorpusDoc, float]]:
    doc_by_id = {doc.doc_id: doc for doc in corpus.docs}
    return [(doc_by_id[doc_id], score) for doc_id, score in ranking[:top_k] if doc_id in doc_by_id]


def _load_corpus_index(path: Path) -> CorpusIndex:
    data = _load_json(path)
    docs = []
    for idx, row in enumerate(data):
        docs.append(CorpusDoc(
            doc_id=str(row.get("idx", idx)),
            title=str(row["title"]),
            text=str(row["text"]),
        ))
    doc_terms = {doc.doc_id: _tokens(doc.retrieval_text) for doc in docs}
    document_frequency = Counter()
    for terms in doc_terms.values():
        for token in terms:
            document_frequency[token] += 1
    avg_doc_len = sum(sum(terms.values()) for terms in doc_terms.values()) / max(1, len(doc_terms))
    return CorpusIndex(
        docs=docs,
        doc_terms=doc_terms,
        document_frequency=document_frequency,
        avg_doc_len=avg_doc_len or 1.0,
    )


def _gold_titles(sample: dict[str, Any]) -> set[str]:
    if "supporting_facts" in sample:
        return {str(item[0]) for item in sample["supporting_facts"]}
    if "paragraphs" in sample:
        return {
            str(item["title"])
            for item in sample["paragraphs"]
            if item.get("is_supporting") is not False
        }
    if "contexts" in sample:
        return {
            str(item["title"])
            for item in sample["contexts"]
            if item.get("is_supporting")
        }
    return set()


def _gold_answers(sample: dict[str, Any]) -> list[str]:
    answers = []
    answer = sample.get("answer", sample.get("gold_ans", sample.get("reference")))
    if isinstance(answer, list):
        answers.extend(str(item) for item in answer)
    elif answer is not None:
        answers.append(str(answer))
    aliases = sample.get("answer_aliases") or []
    if isinstance(aliases, list):
        answers.extend(str(item) for item in aliases)
    deduped = []
    seen = set()
    for item in answers:
        norm = _normalize_text(item)
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(item)
    return deduped


def _contains_any_answer(text: str, answers: list[str]) -> bool:
    normalized_text = _normalize_text(text)
    for answer in answers:
        normalized_answer = _normalize_text(answer)
        if normalized_answer in {"yes", "no"}:
            if normalized_text == normalized_answer:
                return True
            continue
        if len(normalized_answer) < 3:
            continue
        if normalized_answer in normalized_text:
            return True
    return False


def _normalize_text(text: str) -> str:
    return " ".join(WORD_RE.findall(str(text).lower()))


def _expanded_tokens(text: str) -> Counter:
    terms = _tokens(text)
    expanded = Counter(terms)
    for token, freq in terms.items():
        stem = _light_stem(token)
        if stem != token:
            expanded[stem] += max(1, freq // 2)
    return expanded


def _limited_doc_terms(text: str, query_terms: Counter, *, max_terms: int) -> Counter:
    terms = _expanded_tokens(text)
    return _limited_counter_terms(terms, query_terms, max_terms=max_terms)


def _limited_counter_terms(terms: Counter, query_terms: Counter, *, max_terms: int) -> Counter:
    if len(terms) <= max_terms:
        return terms
    scored = []
    for token, freq in terms.items():
        query_bonus = 8.0 if token in query_terms else 0.0
        scored.append((query_bonus + math.log1p(freq), token, freq))
    return Counter({
        token: freq
        for _, token, freq in sorted(scored, key=lambda item: (-item[0], item[1]))[:max_terms]
    })


def _light_stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _ensure_node(graph: dict[str, dict[str, float]], node: str) -> None:
    graph.setdefault(node, {})


def _add_edge(graph: dict[str, dict[str, float]], left: str, right: str, weight: float) -> None:
    if left == right:
        return
    graph.setdefault(left, {})
    graph.setdefault(right, {})
    graph[left][right] = max(graph[left].get(right, 0.0), weight)
    graph[right][left] = max(graph[right].get(left, 0.0), weight)


def _passage_node(doc_id: str) -> str:
    return f"passage::{doc_id}"


def _phrase_node(token: str) -> str:
    return f"phrase::{token}"


def _sample_indices(total: int, *, samples_per_dataset: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    if total <= samples_per_dataset:
        return list(range(total))
    return sorted(rng.sample(range(total), samples_per_dataset))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small HippoRAG QA benchmark risk probe.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="hipporag_qa_probe_20260606")
    parser.add_argument("--samples-per-dataset", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ppr-candidate-pool", type=int, default=40)
    parser.add_argument("--max-doc-phrase-tokens", type=int, default=24)
    parser.add_argument("--run-reader", action="store_true")
    parser.add_argument("--reader-samples-per-dataset", type=int, default=1)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    payload = build_hipporag_qa_probe_payload(
        root=Path(args.root),
        eval_id=args.eval_id,
        samples_per_dataset=args.samples_per_dataset,
        seed=args.seed,
        top_k=args.top_k,
        ppr_candidate_pool=args.ppr_candidate_pool,
        max_doc_phrase_tokens=args.max_doc_phrase_tokens,
        run_reader=args.run_reader,
        reader_samples_per_dataset=args.reader_samples_per_dataset,
    )
    out = Path(args.out)
    _write_json(out, payload)
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "overall": payload["aggregate"]["overall"],
        "reader_qa_summary": payload["reader_qa_summary"],
        "qa_transfer_risk": payload["qa_transfer_risk"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
