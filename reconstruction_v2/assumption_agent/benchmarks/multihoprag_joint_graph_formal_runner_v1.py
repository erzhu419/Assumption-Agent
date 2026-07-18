"""Offline formal runner for the frozen MultiHopRAG joint graph v2 study.

The runner keeps acquisition, action execution, and late labels as separate
capabilities.  Corpus NER and MiniLM features are compiled once; query MiniLM
features are encoded in one batch per newly authorized stage; official
HippoRAG retrieval overlaps the eager Agent process wave; RAW reuses the exact
dense vector consumed by all six Agent actions.  No answer, gold document, or
source question type is accepted by a gold-free execution function.

Formal lifecycle helpers consume durable, exclusive markers and seals.  A_form
labels are descriptive only, F_search has no label input, A_hold labels are
opened only after its action seal, and M_search is inaccessible until a valid
A_hold promotion capability exists.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import importlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from typing import Any, Protocol

import numpy as np

from assumption_agent.benchmarks import multihoprag_direct_acquisition_v1 as acquisition
from assumption_agent.benchmarks.multihoprag_typed_operator_v2 import (
    ACTION_IDS,
    CAPABILITIES,
    INTEGER_SCALE,
    ActionTrace,
    ArticleRecord,
    EvaluationObservation,
    FrozenMapping,
    PolicySelection,
    QueryPlan,
    TypedCorpusGraph,
    build_typed_corpus_graph,
    compile_query_plan,
    exact_magnitude_signflip_p,
    item_utility,
    make_entity_key,
    normalize_text,
    paired_utility_summary,
    parse_date_ordinals,
    policies_identifiable,
    recompute_action_trace_sha256,
    recompute_policy_selection_sha256,
    run_all_actions,
    select_global_policy,
)
from replication_runtime.multihoprag_minilm_v1 import (
    ArticleText,
    CorpusEmbeddingIndex,
    QueryFeatures,
    build_corpus_embedding_index,
    canonical_text,
    encoder_receipt_sha256,
    reciprocal_topic_neighbors,
    recompute_query_feature_sha256,
    validate_corpus_embedding_index,
    validate_query_features,
)
from replication_runtime.multihoprag_minilm_v1.adapter import (
    CAPABILITY_ORDER,
    CAPABILITY_PROTOTYPES,
)
from replication_runtime.multihoprag_ner_v1 import (
    EntitySpan,
    decode_response,
    encode_request,
    verify_runtime_binding as verify_ner_runtime_binding,
)
from replication_runtime.multihoprag_official_hipporag_v1 import (
    RetrievalBatch,
    build_official_hipporag_global_index_v1,
    retrieve_official_hipporag_global_index_v1,
)
from replication_runtime.musique_official_hipporag_v1.runtime_attestation_v3 import (
    verify_formal_runtime_attestation_v3,
)
from replication_runtime.qasper_minilm_v1.binding import (
    EMBEDDING_DIMENSION,
    OfflineMiniLMEncoder,
    QUANTIZATION_SCALE,
    quantized_cosine_similarity,
)


VERSION = "multihoprag_joint_graph_formal_runner_v1"
RUNNER_MARKER_SCHEMA = f"{VERSION}_one_shot_marker"
RUNNER_FAILURE_SCHEMA = f"{VERSION}_terminal_failure"
RESULT_SCHEMA = f"{VERSION}_result"
TERMINAL_RESULT_RELATIVE = (
    "artifacts/multihoprag_joint_graph_formal_v1/formal_result.json"
)
RUNNER_MARKER_RELATIVE = (
    "artifacts/multihoprag_joint_graph_formal_v1/runner.one_shot_marker.json"
)
RUNNER_FAILURE_RELATIVE = (
    "artifacts/multihoprag_joint_graph_formal_v1/runner.terminal_failure.json"
)
A_FORM_DESCRIPTIVE_RELATIVE = (
    "artifacts/multihoprag_joint_graph_formal_v1/A_form.descriptive.json"
)
_SYNTHETIC_SENTINEL = ".multihoprag_synthetic_lifecycle_test_root"
_SYNTHETIC_SENTINEL_CONTENT = "offline_synthetic_no_formal_capabilities_v1\n"

LOCAL_CONCURRENCY_CAP = 64
DEFAULT_NER_BATCH_SIZE = 32
NER_PROCESS_COUNT = 1
OFFICIAL_HIPPO_QUERY_BATCH_CAP = 8
EXPECTED_FORMAL_ARTICLES = 609
EXPECTED_FORMAL_SOURCES = 49
E0_ID = "E0_INDEPENDENT_V2"
E1_ID = "E1_CAUSAL_NECESSITY_V2"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ACTION_WORKER_GRAPH: TypedCorpusGraph | None = None

_FROZEN_PYTHON_ROLE_MODULES = {
    "typed_core": "assumption_agent.benchmarks.multihoprag_typed_operator_v2",
    "acquisition": "assumption_agent.benchmarks.multihoprag_direct_acquisition_v1",
    "minilm_base_runtime_binding": "replication_runtime.qasper_minilm_v1.binding",
    "minilm_runtime_binding": "replication_runtime.multihoprag_minilm_v1.adapter",
    "ner_contract": "replication_runtime.multihoprag_ner_v1.contract",
    "ner_runtime_binding": "replication_runtime.multihoprag_ner_v1.binding",
    "ner_worker": "replication_runtime.multihoprag_ner_v1.worker",
    "global_hipporag_contract": (
        "replication_runtime.multihoprag_official_hipporag_v1.contract"
    ),
    "global_hipporag_adapter": (
        "replication_runtime.multihoprag_official_hipporag_v1.adapter"
    ),
    "global_hipporag_worker": (
        "replication_runtime.multihoprag_official_hipporag_v1.worker"
    ),
    "formal_runner": __name__,
}


class MultiHopRAGFormalRunnerError(RuntimeError):
    """Raised when a formal lifecycle, receipt, or offline invariant drifts."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise MultiHopRAGFormalRunnerError("value is not canonical JSON") from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise MultiHopRAGFormalRunnerError(f"{field} is not a SHA256")
    return value


def fraction_payload(value: Fraction) -> list[int]:
    if not isinstance(value, Fraction):
        raise MultiHopRAGFormalRunnerError("exact statistic is not a Fraction")
    return [value.numerator, value.denominator]


def _self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in body:
        raise MultiHopRAGFormalRunnerError("self-hash field already exists")
    payload = dict(body)
    payload[field] = stable_hash(body)
    return payload


def verify_self_hash(payload: Mapping[str, Any], field: str) -> str:
    declared = _require_sha256(payload.get(field), field)
    body = dict(payload)
    del body[field]
    if stable_hash(body) != declared:
        raise MultiHopRAGFormalRunnerError(f"{field} self-hash mismatch")
    return declared


@dataclass(frozen=True)
class LabelFreeArticle:
    article_i: int
    title: str
    body: str
    source: str
    category: str
    published_at: str

    def hippo_payload(self) -> dict[str, object]:
        return {"idx": self.article_i, "title": self.title, "body": self.body}

    def ner_payload(self) -> dict[str, object]:
        return {"kind": "article", "title": self.title, "body": self.body}


@dataclass(frozen=True)
class PreparedCorpus:
    articles: tuple[LabelFreeArticle, ...]
    corpus_view_sha256: str
    graph: TypedCorpusGraph
    embedding_index: CorpusEmbeddingIndex
    hippo_build_receipt: Mapping[str, Any]
    ner_runtime_receipt_sha256: str
    ner_entity_matrix_sha256: str
    preparation_sha256: str


@dataclass(frozen=True)
class StageItem:
    ordinal: int
    query_sha256: str
    query_feature: QueryFeatures
    plan: QueryPlan
    raw_top5: tuple[int, int, int, int, int]
    hippo_top5: tuple[int, int, int, int, int]
    traces: tuple[ActionTrace, ...]

    def observation(self) -> EvaluationObservation:
        return EvaluationObservation(
            FrozenMapping(tuple((trace.action_id, trace) for trace in self.traces))
        )


@dataclass(frozen=True)
class StageExecution:
    block: str
    view: Mapping[str, Any]
    view_sha256: str
    items: tuple[StageItem, ...]
    graph_sha256: str
    embedding_index_sha256: str
    hippo_build_receipt_sha256: str
    hippo_retrieval_receipt_sha256: str
    execution_matrix_sha256: str
    formal_shape: bool

    def observations(self) -> tuple[EvaluationObservation, ...]:
        return tuple(item.observation() for item in self.items)


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    e0_policy: PolicySelection
    e1_policy: PolicySelection
    delta_total: Fraction
    signflip_p: Fraction
    family_delta_totals: tuple[tuple[str, Fraction], ...]
    e0_utilities: tuple[Fraction, ...]
    e1_utilities: tuple[Fraction, ...]


@dataclass(frozen=True)
class MSearchAssessment:
    l5_delta_total: Fraction
    l5_signflip_p: Fraction
    l5_passed: bool
    agent_minus_hippo_delta_total: Fraction
    agent_minus_hippo_signflip_p: Fraction
    agent_minus_hippo_family_deltas: tuple[tuple[str, Fraction], ...]
    cross_family_agent_over_hippo_passed: bool
    agent_minus_raw_delta_total: Fraction
    agent_minus_raw_signflip_p: Fraction
    agent_complete_count: int
    raw_complete_count: int
    agent_minus_raw_complete_delta: int
    raw_complete_advantage_overcome: bool


@dataclass(frozen=True)
class FormalRuntimeConfig:
    """Path-only frozen local runtime configuration for the one-shot controller."""

    project: Path
    hippo_runtime_python: Path
    hippo_llm_model: Path
    hippo_embedding_model: Path
    hippo_base_binding_receipt: Path
    hippo_attestation_receipt: Path
    hippo_stage_root: Path
    hippo_work_root: Path
    minilm_asset_manifest: Path
    minilm_model_root: Path
    ner_asset_manifest: Path
    ner_model_root: Path
    local_worker_cap: int = LOCAL_CONCURRENCY_CAP
    ner_batch_size: int = DEFAULT_NER_BATCH_SIZE


@dataclass(frozen=True)
class _LifecycleOutputPaths:
    marker: str
    failure: str
    result: str
    a_form_descriptive: str


class BatchNER(Protocol):
    runtime_binding: Mapping[str, object]
    canary_receipt: Mapping[str, object]

    def extract_inputs(
        self, values: Sequence[Mapping[str, object]]
    ) -> tuple[tuple[EntitySpan, ...], ...]: ...


class Encoder(Protocol):
    runtime_receipt: Mapping[str, object]
    canary_receipt: Mapping[str, object]

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class HippoGateway(Protocol):
    def build(self, articles: Sequence[Mapping[str, object]]) -> Mapping[str, Any]: ...

    def retrieve(self, *, block: str, queries: Sequence[str]) -> RetrievalBatch: ...


class ExecutorLike(Protocol):
    def submit(self, fn: Callable[..., Any], /, *args: Any) -> Future[Any]: ...

    def __enter__(self) -> "ExecutorLike": ...

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...


ExecutorFactory = Callable[..., AbstractContextManager[ExecutorLike]]


def spawn_process_pool_executor(**kwargs: Any) -> ProcessPoolExecutor:
    """Create the frozen spawn pool, safe after MiniLM/Torch and live threads."""

    return ProcessPoolExecutor(
        mp_context=multiprocessing.get_context("spawn"),
        **kwargs,
    )


@dataclass(frozen=True)
class OfficialHippoGateway:
    """Path-bound official global-index adapter used by the formal entrypoint."""

    runtime_python: Path
    local_llm_model: Path
    local_embedding_model: Path
    base_binding_receipt_path: Path
    attestation_receipt_path: Path
    stage_root: Path
    work_root: Path

    def build(self, articles: Sequence[Mapping[str, object]]) -> Mapping[str, Any]:
        return build_official_hipporag_global_index_v1(
            articles=articles,
            runtime_python=self.runtime_python,
            local_llm_model=self.local_llm_model,
            local_embedding_model=self.local_embedding_model,
            base_binding_receipt_path=self.base_binding_receipt_path,
            attestation_receipt_path=self.attestation_receipt_path,
            stage_root=self.stage_root,
        )

    def retrieve(self, *, block: str, queries: Sequence[str]) -> RetrievalBatch:
        if block not in acquisition.BLOCK_ORDER:
            raise MultiHopRAGFormalRunnerError("Hippo stage is invalid")
        return retrieve_official_hipporag_global_index_v1(
            queries=queries,
            runtime_python=self.runtime_python,
            local_llm_model=self.local_llm_model,
            local_embedding_model=self.local_embedding_model,
            base_binding_receipt_path=self.base_binding_receipt_path,
            attestation_receipt_path=self.attestation_receipt_path,
            stage_root=self.stage_root,
            work_root=self.work_root / block,
        )


class OfflineNERJSONLClient:
    """One fixed persistent local NER worker, reused across bounded requests."""

    def __init__(
        self,
        *,
        project_root: Path,
        asset_manifest_path: Path,
        model_root: Path,
    ) -> None:
        if NER_PROCESS_COUNT != 1:
            raise MultiHopRAGFormalRunnerError("NER process contract drifted")
        self.runtime_binding = verify_ner_runtime_binding(
            asset_manifest_path=asset_manifest_path,
            model_root=model_root,
        )
        self.canary_receipt: dict[str, object] = {
            "status": "worker_startup_canary_pending"
        }
        environment = dict(os.environ)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "HF_HUB_OFFLINE": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": str(project_root),
                "TOKENIZERS_PARALLELISM": "false",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-B",
                "-m",
                "replication_runtime.multihoprag_ner_v1.worker",
                "--asset-manifest",
                str(asset_manifest_path),
                "--model-root",
                str(model_root),
                "--serve-jsonl",
            ],
            cwd=project_root,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def extract_inputs(
        self, values: Sequence[Mapping[str, object]]
    ) -> tuple[tuple[EntitySpan, ...], ...]:
        if self._process.stdin is None or self._process.stdout is None:
            raise MultiHopRAGFormalRunnerError("NER worker pipes are unavailable")
        raw = encode_request(values)
        self._process.stdin.write(raw)
        self._process.stdin.flush()
        response = self._process.stdout.readline()
        if not response:
            raise MultiHopRAGFormalRunnerError("NER worker terminated without output")
        # The worker cannot enter its JSONL serving loop until FrozenNERExtractor
        # has recomputed the exact manifest-bound synthetic startup canary.
        self.canary_receipt = {
            "multihoprag_rows_or_archives_accessed": False,
            "output_sha256": self.runtime_binding["canary_output_sha256"],
            "status": "passed_exact_row_free_synthetic_canary",
            "worker_serve_loop_reached": True,
        }
        canonical_texts = [
            str(row["query"])
            if row.get("kind") == "query"
            else str(row["title"]) + "\n\n" + str(row["body"])
            for row in values
        ]
        return decode_response(response, canonical_texts=canonical_texts)

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        try:
            returncode = self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=30)
            raise MultiHopRAGFormalRunnerError("NER worker did not terminate") from None
        if returncode != 0:
            stderr = b"" if self._process.stderr is None else self._process.stderr.read()
            raise MultiHopRAGFormalRunnerError(
                f"NER worker failed; stderr_sha256={hashlib.sha256(stderr).hexdigest()}"
            )

    def __enter__(self) -> "OfflineNERJSONLClient":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def _bounded_ner_extract(
    runtime: BatchNER,
    values: Sequence[Mapping[str, object]],
    *,
    batch_size: int,
) -> tuple[tuple[EntitySpan, ...], ...]:
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or not 1 <= batch_size <= 256
    ):
        raise MultiHopRAGFormalRunnerError("NER batch size is outside the frozen bound")
    rows = tuple(values)
    if not rows:
        raise MultiHopRAGFormalRunnerError("NER input batch is empty")
    output: list[tuple[EntitySpan, ...]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        result = runtime.extract_inputs(batch)
        if not isinstance(result, tuple) or len(result) != len(batch):
            raise MultiHopRAGFormalRunnerError("NER result cardinality drifted")
        output.extend(result)
    return tuple(output)


def _ner_receipt_sha256(runtime: BatchNER) -> str:
    runtime_binding = getattr(runtime, "runtime_binding", None)
    canary = getattr(runtime, "canary_receipt", None)
    if not isinstance(runtime_binding, Mapping) or not isinstance(canary, Mapping):
        raise MultiHopRAGFormalRunnerError("NER runtime receipts are unavailable")
    if canary.get("status") != "passed_exact_row_free_synthetic_canary":
        raise MultiHopRAGFormalRunnerError("NER startup canary is not terminal")
    return stable_hash(
        {"canary_receipt": dict(canary), "runtime_binding": dict(runtime_binding)}
    )


def _validate_corpus_view(
    corpus_view: Mapping[str, Any], *, formal_shape: bool
) -> tuple[LabelFreeArticle, ...]:
    acquisition.verify_self_hash(
        corpus_view,
        hash_field="corpus_view_sha256",
        schema=acquisition.CORPUS_VIEW_SCHEMA,
    )
    raw_articles = corpus_view.get("articles")
    if (
        set(corpus_view)
        != {
            "schema",
            "version",
            "article_count",
            "corpus_locator_fields_included",
            "articles",
            "corpus_view_sha256",
        }
        or corpus_view.get("version") != acquisition.VERSION
        or corpus_view.get("corpus_locator_fields_included") is not False
        or not isinstance(raw_articles, list)
        or not raw_articles
        or corpus_view.get("article_count") != len(raw_articles)
    ):
        raise MultiHopRAGFormalRunnerError("corpus view is empty")
    expected_keys = {
        "article_id",
        "title",
        "author",
        "source",
        "published_at",
        "category",
        "body",
    }
    rows: list[LabelFreeArticle] = []
    for position, raw in enumerate(raw_articles):
        if (
            not isinstance(raw, Mapping)
            or set(raw) != expected_keys
            or type(raw.get("article_id")) is not int
            or raw.get("article_id") != position
        ):
            raise MultiHopRAGFormalRunnerError("corpus article identity drifted")
        text_values = {
            key: raw.get(key)
            for key in expected_keys
            if key not in {"article_id", "author"}
        }
        if any(
            not isinstance(value, str) or "\x00" in value
            for value in text_values.values()
        ):
            raise MultiHopRAGFormalRunnerError("corpus article text drifted")
        if not str(raw["title"]).strip() or not str(raw["body"]).strip():
            raise MultiHopRAGFormalRunnerError(
                "official HippoRAG requires nonempty title and body"
            )
        if not normalize_text(str(raw["source"])) or not normalize_text(
            str(raw["category"])
        ):
            raise MultiHopRAGFormalRunnerError("source/category is empty")
        rows.append(
            LabelFreeArticle(
                article_i=position,
                title=str(raw["title"]),
                body=str(raw["body"]),
                source=str(raw["source"]),
                category=str(raw["category"]),
                published_at=str(raw["published_at"]),
            )
        )
    source_count = len({normalize_text(row.source) for row in rows})
    if formal_shape and (
        len(rows) != EXPECTED_FORMAL_ARTICLES
        or source_count != EXPECTED_FORMAL_SOURCES
    ):
        raise MultiHopRAGFormalRunnerError("formal corpus postflight drifted")
    return tuple(rows)


def _article_date(value: str) -> int | None:
    dates = parse_date_ordinals(value)
    if len(dates) > 1:
        raise MultiHopRAGFormalRunnerError("publication date has multiple operands")
    return None if not dates else dates[0]


def prepare_offline_corpus(
    *,
    corpus_view: Mapping[str, Any],
    encoder: Encoder,
    ner: BatchNER,
    hippo: HippoGateway,
    ner_batch_size: int = DEFAULT_NER_BATCH_SIZE,
    formal_shape: bool = True,
) -> PreparedCorpus:
    """Compile corpus NER/MiniLM and build official HippoRAG concurrently once."""

    articles = _validate_corpus_view(corpus_view, formal_shape=formal_shape)
    hippo_rows = tuple(row.hippo_payload() for row in articles)
    ner_rows = tuple(row.ner_payload() for row in articles)
    article_texts = tuple(
        ArticleText(row.article_i, row.title, row.body) for row in articles
    )
    with ThreadPoolExecutor(max_workers=2) as threads:
        hippo_future = threads.submit(hippo.build, hippo_rows)
        ner_future = threads.submit(
            _bounded_ner_extract,
            ner,
            ner_rows,
            batch_size=ner_batch_size,
        )
        embedding_index = build_corpus_embedding_index(
            articles=article_texts, encoder=encoder
        )
        entity_rows = ner_future.result()
        hippo_build_receipt = hippo_future.result()
    validate_corpus_embedding_index(embedding_index)
    topic_neighbors = reciprocal_topic_neighbors(embedding_index)
    if len(entity_rows) != len(articles) or len(topic_neighbors) != len(articles):
        raise MultiHopRAGFormalRunnerError("compiled corpus feature count drifted")
    typed_articles: list[ArticleRecord] = []
    entity_receipt_rows: list[list[list[object]]] = []
    for article, spans, neighbors in zip(
        articles, entity_rows, topic_neighbors, strict=True
    ):
        entities = tuple(
            sorted({make_entity_key(span.entity_type, span.text) for span in spans})
        )
        entity_receipt_rows.append(
            [[entity.entity_type, entity.normalized_span] for entity in entities]
        )
        typed_articles.append(
            ArticleRecord(
                article_i=article.article_i,
                normalized_source=normalize_text(article.source),
                normalized_category=normalize_text(article.category),
                published_ordinal=_article_date(article.published_at),
                entities=entities,
                reciprocal_topic_neighbors=neighbors,
            )
        )
    graph = build_typed_corpus_graph(typed_articles)
    if formal_shape and (
        len(graph.articles) != EXPECTED_FORMAL_ARTICLES
        or len(graph.sources) != EXPECTED_FORMAL_SOURCES
    ):
        raise MultiHopRAGFormalRunnerError("formal graph postflight drifted")
    if not isinstance(hippo_build_receipt, Mapping):
        raise MultiHopRAGFormalRunnerError("Hippo build receipt is absent")
    ner_receipt = _ner_receipt_sha256(ner)
    body = {
        "corpus_view_sha256": _require_sha256(
            corpus_view.get("corpus_view_sha256"), "corpus view"
        ),
        "embedding_index_sha256": embedding_index.index_sha256,
        "graph_sha256": graph.graph_sha256,
        "hippo_build_receipt_sha256": stable_hash(dict(hippo_build_receipt)),
        "ner_entity_matrix_sha256": stable_hash(entity_receipt_rows),
        "ner_runtime_receipt_sha256": ner_receipt,
        "offline_network_calls": 0,
        "online_evaluator_calls": 0,
        "version": VERSION,
    }
    return PreparedCorpus(
        articles=articles,
        corpus_view_sha256=str(corpus_view["corpus_view_sha256"]),
        graph=graph,
        embedding_index=embedding_index,
        hippo_build_receipt=dict(hippo_build_receipt),
        ner_runtime_receipt_sha256=ner_receipt,
        ner_entity_matrix_sha256=body["ner_entity_matrix_sha256"],
        preparation_sha256=stable_hash(body),
    )


def _validated_embedding_matrix(matrix: object, rows: int) -> np.ndarray:
    if (
        not isinstance(matrix, np.ndarray)
        or matrix.shape != (rows, EMBEDDING_DIMENSION)
        or matrix.dtype != np.float32
        or not np.isfinite(matrix).all()
    ):
        raise MultiHopRAGFormalRunnerError("batched MiniLM output drifted")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-5):
        raise MultiHopRAGFormalRunnerError("batched MiniLM output is not normalized")
    return np.ascontiguousarray(matrix, dtype=np.float32)


def compile_query_features_batched(
    *,
    queries: Sequence[str],
    index: CorpusEmbeddingIndex,
    encoder: Encoder,
) -> tuple[QueryFeatures, ...]:
    """Encode all authorized queries and the three prototypes in one model call."""

    index = validate_corpus_embedding_index(index)
    if encoder_receipt_sha256(encoder) != index.encoder_receipt_sha256:
        raise MultiHopRAGFormalRunnerError("query encoder differs from corpus encoder")
    if isinstance(queries, (str, bytes)) or not isinstance(queries, Sequence):
        raise MultiHopRAGFormalRunnerError("queries must be a sequence")
    normalized = tuple(canonical_text(query, field="query") for query in queries)
    if not normalized:
        raise MultiHopRAGFormalRunnerError("query stage is empty")
    texts = (
        *normalized,
        *(CAPABILITY_PROTOTYPES[name] for name in CAPABILITY_ORDER),
    )
    matrix = _validated_embedding_matrix(encoder.encode(texts), len(texts))
    prototypes = matrix[len(normalized) :]
    output: list[QueryFeatures] = []
    for query_text, query_vector in zip(normalized, matrix[: len(normalized)], strict=True):
        capability_scores = tuple(
            quantized_cosine_similarity(query_vector, prototypes[offset])
            for offset in range(len(CAPABILITY_ORDER))
        )
        predicted = min(
            CAPABILITY_ORDER,
            key=lambda name: (
                -capability_scores[CAPABILITY_ORDER.index(name)],
                CAPABILITY_ORDER.index(name),
            ),
        )
        relevance = tuple(
            max(
                quantized_cosine_similarity(query_vector, index.chunk_vectors[row])
                for row in range(start, end)
            )
            for start, end in index.article_chunk_ranges
        )
        provisional = QueryFeatures(
            embedding_index_sha256=index.index_sha256,
            normalized_query_sha256=hashlib.sha256(
                query_text.casefold().encode("utf-8")
            ).hexdigest(),
            capability_similarity_ints=capability_scores,
            predicted_capability=predicted,
            dense_relevance_ints=relevance,
            feature_sha256="0" * 64,
        )
        feature = replace(
            provisional,
            feature_sha256=recompute_query_feature_sha256(
                provisional, index=index
            ),
        )
        output.append(validate_query_features(feature, index=index))
    return tuple(output)


def raw_top5(relevance_ints: Sequence[int]) -> tuple[int, int, int, int, int]:
    rows = tuple(relevance_ints)
    if len(rows) < 5 or any(type(value) is not int for value in rows):
        raise MultiHopRAGFormalRunnerError("RAW relevance vector is invalid")
    selected = tuple(sorted(range(len(rows)), key=lambda i: (-rows[i], i))[:5])
    return selected  # type: ignore[return-value]


def _init_action_worker(graph: TypedCorpusGraph) -> None:
    global _ACTION_WORKER_GRAPH
    _ACTION_WORKER_GRAPH = graph


def _run_item_actions(
    ordinal: int, plan: QueryPlan, relevance: tuple[int, ...]
) -> tuple[int, tuple[ActionTrace, ...]]:
    if _ACTION_WORKER_GRAPH is None:
        raise MultiHopRAGFormalRunnerError("action worker graph is uninitialized")
    return ordinal, run_all_actions(
        graph=_ACTION_WORKER_GRAPH,
        plan=plan,
        relevance_ints=relevance,
    )


def execute_agent_actions_eager(
    *,
    graph: TypedCorpusGraph,
    plans: Sequence[QueryPlan],
    relevance_vectors: Sequence[Sequence[int]],
    local_worker_cap: int = LOCAL_CONCURRENCY_CAP,
    executor_factory: ExecutorFactory = spawn_process_pool_executor,
) -> tuple[tuple[ActionTrace, ...], ...]:
    """Eagerly submit every item before the first join, sharing graph by initializer."""

    if (
        isinstance(local_worker_cap, bool)
        or not isinstance(local_worker_cap, int)
        or not 1 <= local_worker_cap <= LOCAL_CONCURRENCY_CAP
    ):
        raise MultiHopRAGFormalRunnerError("local worker cap is outside 1..64")
    plan_rows = tuple(plans)
    relevance_rows = tuple(tuple(row) for row in relevance_vectors)
    if not plan_rows or len(plan_rows) != len(relevance_rows):
        raise MultiHopRAGFormalRunnerError("Agent input cardinality drifted")
    max_workers = min(local_worker_cap, len(plan_rows))
    with executor_factory(
        max_workers=max_workers,
        initializer=_init_action_worker,
        initargs=(graph,),
    ) as executor:
        futures = [
            executor.submit(_run_item_actions, ordinal, plan, relevance)
            for ordinal, (plan, relevance) in enumerate(
                zip(plan_rows, relevance_rows, strict=True)
            )
        ]
        results = [future.result() for future in futures]
    if [ordinal for ordinal, _traces in results] != list(range(len(plan_rows))):
        raise MultiHopRAGFormalRunnerError("Agent result order drifted")
    traces_by_item = tuple(traces for _ordinal, traces in results)
    for traces in traces_by_item:
        if tuple(trace.action_id for trace in traces) != ACTION_IDS or any(
            trace.trace_sha256 != recompute_action_trace_sha256(trace)
            for trace in traces
        ):
            raise MultiHopRAGFormalRunnerError("Agent action receipt drifted")
    return traces_by_item


def _validate_block_view(
    view: Mapping[str, Any], *, block: str, formal_shape: bool
) -> tuple[Mapping[str, Any], ...]:
    acquisition.verify_self_hash(
        view,
        hash_field="block_view_sha256",
        schema=acquisition.BLOCK_VIEW_SCHEMA,
    )
    items = view.get("items")
    if (
        set(view)
        != {
            "schema",
            "version",
            "block",
            "item_count",
            "late_label_fields_included",
            "items",
            "block_view_sha256",
        }
        or view.get("version") != acquisition.VERSION
        or block not in acquisition.BLOCK_ORDER
        or view.get("block") != block
        or view.get("late_label_fields_included") is not False
        or not isinstance(items, list)
        or not items
        or view.get("item_count") != len(items)
    ):
        raise MultiHopRAGFormalRunnerError("block view drifted")
    if formal_shape and len(items) != acquisition.BLOCK_COUNTS[block]:
        raise MultiHopRAGFormalRunnerError("formal block item count drifted")
    for ordinal, item in enumerate(items):
        if (
            not isinstance(item, Mapping)
            or set(item) != {"schema", "block", "ordinal", "query"}
            or item.get("schema") != acquisition.VIEW_ITEM_SCHEMA
            or item.get("block") != block
            or type(item.get("ordinal")) is not int
            or item.get("ordinal") != ordinal
            or not isinstance(item.get("query"), str)
            or not str(item["query"]).strip()
        ):
            raise MultiHopRAGFormalRunnerError("block view item drifted")
    return tuple(items)


def _validate_hippo_indices(
    batch: RetrievalBatch, *, count: int, corpus_count: int
) -> tuple[tuple[int, int, int, int, int], ...]:
    if not isinstance(batch, RetrievalBatch) or len(batch.indices) != count:
        raise MultiHopRAGFormalRunnerError("Hippo retrieval cardinality drifted")
    output: list[tuple[int, int, int, int, int]] = []
    for row in batch.indices:
        if (
            not isinstance(row, tuple)
            or len(row) != 5
            or len(set(row)) != 5
            or any(type(value) is not int or not 0 <= value < corpus_count for value in row)
        ):
            raise MultiHopRAGFormalRunnerError("Hippo top5 drifted")
        output.append(row)  # type: ignore[arg-type]
    if not isinstance(batch.receipt, Mapping):
        raise MultiHopRAGFormalRunnerError("Hippo retrieval receipt is absent")
    batch_sizes = batch.receipt.get("batch_sizes")
    if (
        not isinstance(batch_sizes, list)
        or any(type(value) is not int or not 1 <= value <= OFFICIAL_HIPPO_QUERY_BATCH_CAP for value in batch_sizes)
        or sum(batch_sizes) != count
    ):
        raise MultiHopRAGFormalRunnerError("Hippo query batch cap drifted")
    return tuple(output)


def execute_gold_free_stage(
    *,
    block: str,
    view: Mapping[str, Any],
    prepared: PreparedCorpus,
    encoder: Encoder,
    ner: BatchNER,
    hippo: HippoGateway,
    ner_batch_size: int = DEFAULT_NER_BATCH_SIZE,
    local_worker_cap: int = LOCAL_CONCURRENCY_CAP,
    formal_shape: bool = True,
    executor_factory: ExecutorFactory = spawn_process_pool_executor,
) -> StageExecution:
    """Run NER/MiniLM/RAW/Hippo/Agent without accepting a label capability."""

    items = _validate_block_view(view, block=block, formal_shape=formal_shape)
    queries = tuple(str(item["query"]) for item in items)
    ner_inputs = tuple({"kind": "query", "query": query} for query in queries)
    with ThreadPoolExecutor(max_workers=2) as threads:
        hippo_future = threads.submit(hippo.retrieve, block=block, queries=queries)
        ner_future = threads.submit(
            _bounded_ner_extract,
            ner,
            ner_inputs,
            batch_size=ner_batch_size,
        )
        features = compile_query_features_batched(
            queries=queries,
            index=prepared.embedding_index,
            encoder=encoder,
        )
        query_entities = ner_future.result()
        plans: list[QueryPlan] = []
        for query, feature, spans in zip(
            queries, features, query_entities, strict=True
        ):
            entities = tuple(
                sorted({make_entity_key(span.entity_type, span.text) for span in spans})
            )
            plan = compile_query_plan(
                graph=prepared.graph,
                query=query,
                capability_similarity_ints={
                    name: feature.capability_similarity_ints[offset]
                    for offset, name in enumerate(CAPABILITIES)
                },
                query_entities=entities,
            )
            if (
                plan.capability != feature.predicted_capability
                or plan.query_sha256 != feature.normalized_query_sha256
            ):
                raise MultiHopRAGFormalRunnerError(
                    "MiniLM feature and typed query plan disagree"
                )
            plans.append(plan)
        traces_by_item = execute_agent_actions_eager(
            graph=prepared.graph,
            plans=plans,
            relevance_vectors=[feature.dense_relevance_ints for feature in features],
            local_worker_cap=local_worker_cap,
            executor_factory=executor_factory,
        )
        hippo_batch = hippo_future.result()
    hippo_indices = _validate_hippo_indices(
        hippo_batch, count=len(items), corpus_count=len(prepared.articles)
    )
    stage_items = tuple(
        StageItem(
            ordinal=ordinal,
            query_sha256=plan.query_sha256,
            query_feature=feature,
            plan=plan,
            raw_top5=raw_top5(feature.dense_relevance_ints),
            hippo_top5=hippo_indices[ordinal],
            traces=traces_by_item[ordinal],
        )
        for ordinal, (feature, plan) in enumerate(zip(features, plans, strict=True))
    )
    return StageExecution(
        block=block,
        view=view,
        view_sha256=_require_sha256(view.get("block_view_sha256"), "block view"),
        items=stage_items,
        graph_sha256=prepared.graph.graph_sha256,
        embedding_index_sha256=prepared.embedding_index.index_sha256,
        hippo_build_receipt_sha256=stable_hash(dict(prepared.hippo_build_receipt)),
        hippo_retrieval_receipt_sha256=stable_hash(dict(hippo_batch.receipt)),
        execution_matrix_sha256=stage_execution_matrix_sha256(
            stage_items,
            expected_embedding_index_sha256=prepared.embedding_index.index_sha256,
            index=prepared.embedding_index,
        ),
        formal_shape=formal_shape,
    )


def stage_execution_matrix_sha256(
    items: Sequence[StageItem],
    *,
    expected_embedding_index_sha256: str,
    index: CorpusEmbeddingIndex | None = None,
) -> str:
    """Recompute the complete gold-free item/feature/output/trace commitment."""

    rows = tuple(items)
    if not rows:
        raise MultiHopRAGFormalRunnerError("stage execution is empty")
    body: list[dict[str, object]] = []
    for ordinal, item in enumerate(rows):
        if not isinstance(item, StageItem) or item.ordinal != ordinal:
            raise MultiHopRAGFormalRunnerError("stage item identity drifted")
        if item.query_sha256 != item.plan.query_sha256:
            raise MultiHopRAGFormalRunnerError("stage query/plan binding drifted")
        if item.query_feature.normalized_query_sha256 != item.query_sha256:
            raise MultiHopRAGFormalRunnerError("stage query/feature binding drifted")
        if (
            item.query_feature.embedding_index_sha256
            != expected_embedding_index_sha256
            or _SHA256.fullmatch(item.query_feature.feature_sha256) is None
        ):
            raise MultiHopRAGFormalRunnerError("stage query feature binding drifted")
        if index is not None:
            validate_query_features(item.query_feature, index=index)
        if item.raw_top5 != raw_top5(item.query_feature.dense_relevance_ints):
            raise MultiHopRAGFormalRunnerError(
                "RAW output differs from the shared dense relevance vector"
            )
        if tuple(trace.action_id for trace in item.traces) != ACTION_IDS:
            raise MultiHopRAGFormalRunnerError("stage action registry drifted")
        expected_relevance_sha256 = stable_hash(
            {
                "integer_scale": INTEGER_SCALE,
                "values": list(item.query_feature.dense_relevance_ints),
            }
        )
        expected_action_input = (
            item.plan.graph_sha256,
            item.plan.plan_sha256,
            item.query_sha256,
            expected_relevance_sha256,
        )
        observed_action_inputs = {
            (
                trace.graph_sha256,
                trace.plan_sha256,
                trace.query_sha256,
                trace.relevance_sha256,
            )
            for trace in item.traces
        }
        if observed_action_inputs != {expected_action_input}:
            raise MultiHopRAGFormalRunnerError(
                "six actions do not share the exact query/plan/graph/relevance input"
            )
        if any(
            trace.trace_sha256 != recompute_action_trace_sha256(trace)
            for trace in item.traces
        ):
            raise MultiHopRAGFormalRunnerError("stage action receipt binding drifted")
        for method, output in (
            ("RAW", item.raw_top5),
            ("HippoRAG", item.hippo_top5),
        ):
            if (
                len(output) != 5
                or len(set(output)) != 5
                or any(type(value) is not int or value < 0 for value in output)
            ):
                raise MultiHopRAGFormalRunnerError(
                    f"{method} stage output drifted"
                )
        body.append(
            {
                "feature_sha256": item.query_feature.feature_sha256,
                "hippo_top5_sha256": stable_hash(list(item.hippo_top5)),
                "ordinal": item.ordinal,
                "plan_sha256": item.plan.plan_sha256,
                "query_sha256": item.query_sha256,
                "raw_top5_sha256": stable_hash(list(item.raw_top5)),
                "trace_sha256s": [
                    trace.trace_sha256 for trace in item.traces
                ],
            }
        )
    return stable_hash(body)


def build_canonical_stage_records(
    stage: StageExecution,
) -> tuple[dict[str, Any], ...]:
    """Encode the only canonical full-typed-trace archive records."""

    expected_matrix = stage_execution_matrix_sha256(
        stage.items,
        expected_embedding_index_sha256=stage.embedding_index_sha256,
    )
    if expected_matrix != stage.execution_matrix_sha256:
        raise MultiHopRAGFormalRunnerError("stage execution matrix drifted")
    view_items = stage.view.get("items")
    if not isinstance(view_items, list) or len(view_items) != len(stage.items):
        raise MultiHopRAGFormalRunnerError("stage/view cardinality drifted")
    return tuple(
        acquisition.build_stage_output_record(
            block=stage.block,
            ordinal=item.ordinal,
            view_sha256=stable_hash(view_items[item.ordinal]),
            dense_relevance_ints=item.query_feature.dense_relevance_ints,
            raw_top5=item.raw_top5,
            hipporag_top5=item.hippo_top5,
            action_traces=item.traces,
        )
        for item in stage.items
    )


def build_stage_runtime_binding(
    *, prepared: PreparedCorpus, stage: StageExecution
) -> dict[str, Any]:
    """Bind every offline runtime/input receipt consumed by one stage."""

    expected_preparation = stable_hash(
        {
            "corpus_view_sha256": prepared.corpus_view_sha256,
            "embedding_index_sha256": prepared.embedding_index.index_sha256,
            "graph_sha256": prepared.graph.graph_sha256,
            "hippo_build_receipt_sha256": stable_hash(
                dict(prepared.hippo_build_receipt)
            ),
            "ner_entity_matrix_sha256": prepared.ner_entity_matrix_sha256,
            "ner_runtime_receipt_sha256": prepared.ner_runtime_receipt_sha256,
            "offline_network_calls": 0,
            "online_evaluator_calls": 0,
            "version": VERSION,
        }
    )
    expected_matrix = stage_execution_matrix_sha256(
        stage.items,
        expected_embedding_index_sha256=prepared.embedding_index.index_sha256,
        index=prepared.embedding_index,
    )
    if (
        prepared.preparation_sha256 != expected_preparation
        or stage.execution_matrix_sha256 != expected_matrix
        or stage.graph_sha256 != prepared.graph.graph_sha256
        or stage.embedding_index_sha256 != prepared.embedding_index.index_sha256
        or stage.hippo_build_receipt_sha256
        != stable_hash(dict(prepared.hippo_build_receipt))
        or stage.view_sha256 != stage.view.get("block_view_sha256")
    ):
        raise MultiHopRAGFormalRunnerError("stage runtime binding input drifted")
    return {
        "preparation_sha256": prepared.preparation_sha256,
        "graph_sha256": stage.graph_sha256,
        "embedding_index_sha256": stage.embedding_index_sha256,
        "ner_runtime_receipt_sha256": prepared.ner_runtime_receipt_sha256,
        "ner_entity_matrix_sha256": prepared.ner_entity_matrix_sha256,
        "hippo_build_receipt_sha256": stage.hippo_build_receipt_sha256,
        "hippo_retrieval_receipt_sha256": (
            stage.hippo_retrieval_receipt_sha256
        ),
        "execution_matrix_sha256": stage.execution_matrix_sha256,
    }


def persist_canonical_stage_archive(
    *, project: Path, prepared: PreparedCorpus, stage: StageExecution
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Delegate the unique O_EXCL private archive write to acquisition."""

    if not stage.formal_shape:
        raise MultiHopRAGFormalRunnerError(
            "synthetic execution cannot write a formal stage archive"
        )
    runtime_binding = build_stage_runtime_binding(
        prepared=prepared, stage=stage
    )
    records = build_canonical_stage_records(stage)
    archive, binding = acquisition.create_stage_output_archive_once(
        project=project,
        block=stage.block,
        records=records,
        stage_runtime_binding=runtime_binding,
    )
    if archive.get("records") != list(records):
        raise MultiHopRAGFormalRunnerError("canonical archive changed stage bytes")
    # The current acquisition API will accept and persist this exact object in
    # its next schema revision.  Until then, recomputing it here prevents a
    # caller from presenting a stage whose runtime receipts do not cohere.
    return archive, binding, runtime_binding


def select_f_policies(
    *, f_stage: StageExecution
) -> tuple[PolicySelection, PolicySelection, bool]:
    """Select both fixed evaluators from the complete label-free F matrix."""

    if f_stage.block != "F_search":
        raise MultiHopRAGFormalRunnerError("policy selection requires F_search")
    observations = f_stage.observations()
    e0 = select_global_policy(evaluator_id=E0_ID, observations=observations)
    e1 = select_global_policy(evaluator_id=E1_ID, observations=observations)
    return e0, e1, policies_identifiable(e0, e1, observations)


def _joined_labels(
    stage: StageExecution, labels: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    """Validate and join late labels; synthetic shapes use the same row schema."""

    verify_self_hash(labels, "block_labels_sha256")
    expected_envelope = {
        "schema",
        "version",
        "block",
        "item_count",
        "source_locator_payload_included",
        "items",
        "block_labels_sha256",
    }
    label_items = labels.get("items")
    if (
        set(labels) != expected_envelope
        or labels.get("schema") != acquisition.BLOCK_LABEL_SCHEMA
        or labels.get("version") != acquisition.VERSION
        or labels.get("block") != stage.block
        or labels.get("source_locator_payload_included") is not False
        or not isinstance(label_items, list)
        or labels.get("item_count") != len(label_items)
        or len(label_items) != len(stage.items)
    ):
        raise MultiHopRAGFormalRunnerError("late label envelope drifted")
    expected_row_keys = {
        "schema",
        "block",
        "ordinal",
        "view_sha256",
        "identity_commitment_sha256",
        "source_record_commitment_sha256",
        "question_type",
        "answer",
        "gold_article_ids",
    }
    label_map: dict[str, Mapping[str, Any]] = {}
    identities: set[str] = set()
    source_records: set[str] = set()
    family_counts: Counter[str] = Counter()
    for ordinal, label in enumerate(label_items):
        if (
            not isinstance(label, Mapping)
            or set(label) != expected_row_keys
            or label.get("schema") != acquisition.LABEL_ITEM_SCHEMA
            or label.get("block") != stage.block
            or label.get("ordinal") != ordinal
            or label.get("question_type") not in acquisition.FAMILIES
            or not isinstance(label.get("answer"), str)
            or not str(label["answer"]).strip()
            or not isinstance(label.get("gold_article_ids"), list)
            or not 2 <= len(label["gold_article_ids"]) <= 4
            or any(
                type(value) is not int or value < 0
                for value in label["gold_article_ids"]
            )
            or label["gold_article_ids"]
            != sorted(set(label["gold_article_ids"]))
        ):
            raise MultiHopRAGFormalRunnerError("late label item drifted")
        view_sha = _require_sha256(label.get("view_sha256"), "label view")
        identity = _require_sha256(
            label.get("identity_commitment_sha256"), "label identity"
        )
        source_record = _require_sha256(
            label.get("source_record_commitment_sha256"), "label source record"
        )
        if view_sha in label_map or identity in identities or source_record in source_records:
            raise MultiHopRAGFormalRunnerError("late label commitment overlaps")
        label_map[view_sha] = label
        identities.add(identity)
        source_records.add(source_record)
        family_counts[str(label["question_type"])] += 1
    if stage.formal_shape and family_counts != Counter(
        {
            family: acquisition.FAMILY_QUOTAS[stage.block]
            for family in acquisition.FAMILIES
        }
    ):
        raise MultiHopRAGFormalRunnerError("formal late label family quotas drifted")
    joined: list[Mapping[str, Any]] = []
    view_items = stage.view.get("items")
    if not isinstance(view_items, list) or len(view_items) != len(stage.items):
        raise MultiHopRAGFormalRunnerError("late label view cardinality drifted")
    for ordinal, view_item in enumerate(view_items):
        label = label_map.get(stable_hash(view_item))
        if label is None or label.get("ordinal") != ordinal:
            raise MultiHopRAGFormalRunnerError("late label join is incomplete")
        joined.append(label)
    return tuple(joined)


def _trace_for_policy(item: StageItem, policy: PolicySelection) -> ActionTrace:
    if policy.selection_sha256 != recompute_policy_selection_sha256(policy):
        raise MultiHopRAGFormalRunnerError("policy receipt drifted")
    traces = {trace.action_id: trace for trace in item.traces}
    if set(traces) != set(ACTION_IDS) or policy.action_id not in traces:
        raise MultiHopRAGFormalRunnerError("policy action is absent from item")
    return traces[policy.action_id]


def descriptive_a_form(
    *,
    stage: StageExecution,
    policy_freeze: Mapping[str, Any],
    labels: Mapping[str, Any],
) -> dict[str, Any]:
    """Open A_form labels after sealing and report without changing a policy."""

    if stage.block != "A_form":
        raise MultiHopRAGFormalRunnerError("descriptive labels require A_form")
    freeze_sha256 = verify_self_hash(
        policy_freeze, "a_form_policy_freeze_sha256"
    )
    observations = stage.observations()
    e0 = select_global_policy(evaluator_id=E0_ID, observations=observations)
    e1 = select_global_policy(evaluator_id=E1_ID, observations=observations)
    identifiable = policies_identifiable(e0, e1, observations)
    expected_trace_matrix = stable_hash(
        [
            [trace.trace_sha256 for trace in item.traces]
            for item in stage.items
        ]
    )
    if (
        policy_freeze.get("schema") != acquisition.A_FORM_POLICY_FREEZE_SCHEMA
        or policy_freeze.get("status")
        != "A_form_prelabel_descriptive_policies_frozen"
        or policy_freeze.get("a_form_item_count") != len(stage.items)
        or policy_freeze.get("complete_a_form_trace_matrix_receipt_sha256")
        != expected_trace_matrix
        or policy_freeze.get("e0_action_id") != e0.action_id
        or policy_freeze.get("e0_policy_sha256") != e0.selection_sha256
        or policy_freeze.get("e1_action_id") != e1.action_id
        or policy_freeze.get("e1_policy_sha256") != e1.selection_sha256
        or policy_freeze.get("policies_identifiable") is not identifiable
        or policy_freeze.get("selection_purpose")
        != "prelabel_descriptive_only_not_F_policy"
        or policy_freeze.get("A_form_gold_opened_before_policy_freeze")
        is not False
    ):
        raise MultiHopRAGFormalRunnerError(
            "A_form policy freeze does not bind the sealed action evidence"
        )
    label_rows = _joined_labels(stage, labels)
    arms: dict[str, list[Fraction]] = {"RAW": [], "HippoRAG": []}
    arms.update({action_id: [] for action_id in ACTION_IDS})
    families: Counter[str] = Counter()
    for item, label in zip(stage.items, label_rows, strict=True):
        gold = label["gold_article_ids"]
        family = str(label["question_type"])
        families[family] += 1
        arms["RAW"].append(item_utility(item.raw_top5, gold))
        arms["HippoRAG"].append(item_utility(item.hippo_top5, gold))
        for trace in item.traces:
            arms[trace.action_id].append(item_utility(trace.output_top5, gold))
    return _self_hashed(
        {
            "schema": f"{VERSION}_A_form_descriptive",
            "version": VERSION,
            "status": "descriptive_only_no_policy_or_threshold_change",
            "block_view_sha256": stage.view_sha256,
            "a_form_policy_freeze_sha256": freeze_sha256,
            "e0_action_id": e0.action_id,
            "e0_policy_sha256": e0.selection_sha256,
            "e1_action_id": e1.action_id,
            "e1_policy_sha256": e1.selection_sha256,
            "policies_identifiable": identifiable,
            "item_count": len(stage.items),
            "exact_family_counts": dict(sorted(families.items())),
            "arm_utility_totals": {
                arm: fraction_payload(sum(values, Fraction(0)))
                for arm, values in sorted(arms.items())
            },
            "labels_opened_after_action_seal": True,
            "outcome_used_to_change_action_evaluator_or_threshold": False,
        },
        "descriptive_sha256",
    )


def decide_a_hold_promotion(
    *,
    stage: StageExecution,
    labels: Mapping[str, Any],
    f_stage: StageExecution,
    e0_policy: PolicySelection,
    e1_policy: PolicySelection,
) -> PromotionDecision:
    """Apply the sole exact promotion rule after A_hold actions are sealed."""

    if stage.block != "A_hold" or f_stage.block != "F_search":
        raise MultiHopRAGFormalRunnerError("promotion stage identity drifted")
    if not policies_identifiable(e0_policy, e1_policy, f_stage.observations()):
        raise MultiHopRAGFormalRunnerError("unidentifiable policies cannot open A_hold")
    label_rows = _joined_labels(stage, labels)
    e0_values: list[Fraction] = []
    e1_values: list[Fraction] = []
    family_deltas: dict[str, Fraction] = {
        family: Fraction(0) for family in acquisition.FAMILIES
    }
    for item, label in zip(stage.items, label_rows, strict=True):
        gold = label["gold_article_ids"]
        family = str(label["question_type"])
        if family not in family_deltas:
            raise MultiHopRAGFormalRunnerError("A_hold family drifted")
        e0_value = item_utility(
            _trace_for_policy(item, e0_policy).output_top5, gold
        )
        e1_value = item_utility(
            _trace_for_policy(item, e1_policy).output_top5, gold
        )
        e0_values.append(e0_value)
        e1_values.append(e1_value)
        family_deltas[family] += e1_value - e0_value
    summary = paired_utility_summary(e1_values, e0_values)
    promoted = summary.delta_total > 0 and summary.exact_one_sided_p <= Fraction(1, 10)
    return PromotionDecision(
        promoted=promoted,
        e0_policy=e0_policy,
        e1_policy=e1_policy,
        delta_total=summary.delta_total,
        signflip_p=summary.exact_one_sided_p,
        family_delta_totals=tuple(
            (family, family_deltas[family]) for family in acquisition.FAMILIES
        ),
        e0_utilities=tuple(e0_values),
        e1_utilities=tuple(e1_values),
    )


def assess_m_search(
    *,
    stage: StageExecution,
    labels: Mapping[str, Any],
    f_stage: StageExecution,
    e0_policy: PolicySelection,
    e1_policy: PolicySelection,
) -> MSearchAssessment:
    """Compute frozen L5, Agent-Hippo cross-family, and Agent-RAW boundaries."""

    if stage.block != "M_search":
        raise MultiHopRAGFormalRunnerError("M assessment requires M_search")
    if not policies_identifiable(e0_policy, e1_policy, f_stage.observations()):
        raise MultiHopRAGFormalRunnerError("M policies differ from frozen F")
    labels_by_item = _joined_labels(stage, labels)
    e0_values: list[Fraction] = []
    e1_values: list[Fraction] = []
    hippo_values: list[Fraction] = []
    raw_values: list[Fraction] = []
    family_agent_hippo: dict[str, Fraction] = {
        family: Fraction(0) for family in acquisition.FAMILIES
    }
    agent_complete = 0
    raw_complete = 0
    for item, label in zip(stage.items, labels_by_item, strict=True):
        gold = tuple(label["gold_article_ids"])
        family = str(label["question_type"])
        if family not in family_agent_hippo:
            raise MultiHopRAGFormalRunnerError("M_search family drifted")
        e0_output = _trace_for_policy(item, e0_policy).output_top5
        e1_output = _trace_for_policy(item, e1_policy).output_top5
        e0_value = item_utility(e0_output, gold)
        e1_value = item_utility(e1_output, gold)
        hippo_value = item_utility(item.hippo_top5, gold)
        raw_value = item_utility(item.raw_top5, gold)
        e0_values.append(e0_value)
        e1_values.append(e1_value)
        hippo_values.append(hippo_value)
        raw_values.append(raw_value)
        family_agent_hippo[family] += e1_value - hippo_value
        agent_complete += int(set(gold) <= set(e1_output))
        raw_complete += int(set(gold) <= set(item.raw_top5))
    l5 = paired_utility_summary(e1_values, e0_values)
    agent_hippo = paired_utility_summary(e1_values, hippo_values)
    agent_raw = paired_utility_summary(e1_values, raw_values)
    cross_family = (
        agent_hippo.delta_total > 0
        and agent_hippo.exact_one_sided_p <= Fraction(1, 10)
        and all(value > 0 for value in family_agent_hippo.values())
    )
    complete_delta = agent_complete - raw_complete
    return MSearchAssessment(
        l5_delta_total=l5.delta_total,
        l5_signflip_p=l5.exact_one_sided_p,
        l5_passed=l5.delta_total > 0
        and l5.exact_one_sided_p <= Fraction(1, 10),
        agent_minus_hippo_delta_total=agent_hippo.delta_total,
        agent_minus_hippo_signflip_p=agent_hippo.exact_one_sided_p,
        agent_minus_hippo_family_deltas=tuple(
            (family, family_agent_hippo[family])
            for family in acquisition.FAMILIES
        ),
        cross_family_agent_over_hippo_passed=cross_family,
        agent_minus_raw_delta_total=agent_raw.delta_total,
        agent_minus_raw_signflip_p=agent_raw.exact_one_sided_p,
        agent_complete_count=agent_complete,
        raw_complete_count=raw_complete,
        agent_minus_raw_complete_delta=complete_delta,
        raw_complete_advantage_overcome=complete_delta >= 0,
    )


def make_result_receipt(
    *,
    assessment: Mapping[str, Any],
    promotion: Mapping[str, Any],
    m_action_seal: Mapping[str, Any],
    runner_marker: Mapping[str, Any],
) -> dict[str, Any]:
    promotion_sha256 = _require_sha256(
        promotion.get("promotion_sha256"), "promotion"
    )
    seal_sha256 = _require_sha256(
        m_action_seal.get("action_seal_sha256"), "M action seal"
    )
    fraction_fields = (
        "l5_delta_total",
        "l5_signflip_p",
        "agent_minus_hippo_delta_total",
        "agent_minus_hippo_signflip_p",
        "agent_minus_raw_delta_total",
        "agent_minus_raw_signflip_p",
    )
    if any(not isinstance(assessment.get(field), Fraction) for field in fraction_fields):
        raise MultiHopRAGFormalRunnerError(
            "authoritative M exact fractions drifted"
        )
    family_deltas = assessment.get("agent_minus_hippo_family_deltas")
    if (
        assessment.get("status") != "M_search_authoritatively_assessed"
        or assessment.get("promotion_sha256") != promotion_sha256
        or assessment.get("m_search_action_seal_sha256") != seal_sha256
        or not isinstance(family_deltas, Mapping)
        or set(family_deltas) != set(acquisition.FAMILIES)
        or any(not isinstance(value, Fraction) for value in family_deltas.values())
        or any(
            type(assessment.get(field)) is not int
            for field in (
                "agent_complete_count",
                "raw_complete_count",
                "agent_minus_raw_complete_delta",
            )
        )
        or any(
            type(assessment.get(field)) is not bool
            for field in (
                "l5_passed",
                "cross_family_agent_over_hippo_passed",
                "raw_complete_advantage_overcome",
            )
        )
    ):
        raise MultiHopRAGFormalRunnerError(
            "authoritative M assessment binding drifted"
        )
    marker_sha256 = verify_self_hash(runner_marker, "marker_sha256")
    if (
        runner_marker.get("schema") != RUNNER_MARKER_SCHEMA
        or runner_marker.get("version") != VERSION
        or runner_marker.get("phase")
        != "formal_A_form_F_A_hold_M_one_shot"
        or runner_marker.get("replay_retry_resample_replacement_authorized")
        is not False
    ):
        raise MultiHopRAGFormalRunnerError("runner marker contract drifted")
    body = {
        "schema": RESULT_SCHEMA,
        "version": VERSION,
        "status": "formal_M_search_complete",
        "promotion_sha256": promotion_sha256,
        "m_action_seal_sha256": seal_sha256,
        "m_search_output_archive_file_sha256": _require_sha256(
            assessment.get("m_search_output_archive_file_sha256"),
            "M archive file",
        ),
        "m_search_output_archive_semantic_sha256": _require_sha256(
            assessment.get("m_search_output_archive_semantic_sha256"),
            "M archive semantic",
        ),
        "runner_marker_sha256": marker_sha256,
        "L5": {
            "delta_total": fraction_payload(assessment["l5_delta_total"]),
            "signflip_p": fraction_payload(assessment["l5_signflip_p"]),
            "passed": assessment["l5_passed"],
        },
        "Agent_minus_HippoRAG": {
            "delta_total": fraction_payload(
                assessment["agent_minus_hippo_delta_total"]
            ),
            "signflip_p": fraction_payload(
                assessment["agent_minus_hippo_signflip_p"]
            ),
            "family_delta_totals": {
                family: fraction_payload(family_deltas[family])
                for family in acquisition.FAMILIES
            },
            "cross_family_passed": (
                assessment["cross_family_agent_over_hippo_passed"]
            ),
        },
        "Agent_minus_RAW": {
            "delta_total": fraction_payload(
                assessment["agent_minus_raw_delta_total"]
            ),
            "signflip_p": fraction_payload(
                assessment["agent_minus_raw_signflip_p"]
            ),
            "agent_complete_count": assessment["agent_complete_count"],
            "raw_complete_count": assessment["raw_complete_count"],
            "complete_delta": assessment["agent_minus_raw_complete_delta"],
            "raw_complete_advantage_overcome": (
                assessment["raw_complete_advantage_overcome"]
            ),
        },
        "external_network_calls": 0,
        "online_evaluator_calls": 0,
        "result_used_to_change_epoch": False,
        "same_source_replay_authorized": False,
    }
    return _self_hashed(body, "result_sha256")


def default_formal_runtime_config(project: Path) -> FormalRuntimeConfig:
    """Resolve the already-qualified local runtimes; no discovery or download."""

    root = project.resolve(strict=True)
    home = Path.home()
    return FormalRuntimeConfig(
        project=root,
        hippo_runtime_python=home / ".hr5/venv/bin/python",
        hippo_llm_model=home / ".hr5/models/smollm2-135m-instruct",
        hippo_embedding_model=(
            home
            / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
        ),
        hippo_base_binding_receipt=(
            root / "manifests/musique_official_hipporag_retrieve_only_binding_v1.json"
        ),
        hippo_attestation_receipt=(
            root / "manifests/musique_official_hipporag_runtime_attestation_v3.json"
        ),
        hippo_stage_root=(
            root
            / "artifacts/multihoprag_joint_graph_formal_v1/official_hipporag_stage"
        ),
        hippo_work_root=(
            root
            / "artifacts/multihoprag_joint_graph_formal_v1/hipporag_query_work"
        ),
        minilm_asset_manifest=(
            root / "manifests/qasper_minilm_runtime_asset_v1.json"
        ),
        minilm_model_root=(
            root / "artifacts/qasper_minilm_runtime_v1/model"
        ),
        ner_asset_manifest=(
            root / "manifests/multihoprag_ner_runtime_asset_v1.json"
        ),
        ner_model_root=(
            root / "artifacts/multihoprag_ner_runtime_v1/model"
        ),
    )


def preflight_formal_runtime_config(
    config: FormalRuntimeConfig,
) -> Mapping[str, Any]:
    """Verify the exact committed v3 Hippo runtime before burning the cohort."""

    if (
        not isinstance(config, FormalRuntimeConfig)
        or config.local_worker_cap != LOCAL_CONCURRENCY_CAP
        or config.ner_batch_size != DEFAULT_NER_BATCH_SIZE
    ):
        raise MultiHopRAGFormalRunnerError("formal runtime config is invalid")
    project = config.project.resolve(strict=True)
    expected_attestation = (
        project / "manifests/musique_official_hipporag_runtime_attestation_v3.json"
    )
    if config.hippo_attestation_receipt.absolute() != expected_attestation:
        raise MultiHopRAGFormalRunnerError(
            "Hippo input must be the committed v3 attestation manifest"
        )
    if config != default_formal_runtime_config(project):
        raise MultiHopRAGFormalRunnerError(
            "formal runtime config must equal the exact default path binding"
        )
    return verify_formal_runtime_attestation_v3(
        project_root=project,
        attestation_receipt_path=config.hippo_attestation_receipt,
        base_binding_receipt_path=config.hippo_base_binding_receipt,
        runtime_python=config.hippo_runtime_python,
        local_llm_model=config.hippo_llm_model,
        local_embedding_model=config.hippo_embedding_model,
    )


def _verify_loaded_frozen_module_origins(
    *,
    project: Path,
    implementation_receipt: Mapping[str, Any],
) -> dict[str, str]:
    """Close verified working/HEAD bytes to the modules actually in memory."""

    root = project.resolve(strict=True)
    if (
        implementation_receipt.get("all_bindings_byte_match_committed_HEAD")
        is not True
        or implementation_receipt.get("required_role_count")
        != len(acquisition.REQUIRED_FREEZE_ROLES)
    ):
        raise MultiHopRAGFormalRunnerError(
            "implementation freeze verification is not closed"
        )
    observed: dict[str, str] = {}
    for role, module_name in _FROZEN_PYTHON_ROLE_MODULES.items():
        relative = acquisition.FIXED_FREEZE_ROLE_PATHS.get(role)
        if not isinstance(relative, str) or not relative.endswith(".py"):
            raise MultiHopRAGFormalRunnerError(
                "frozen Python role path drifted"
            )
        expected = (root / relative).resolve(strict=True)
        module = importlib.import_module(module_name)
        module_file = getattr(module, "__file__", None)
        spec = getattr(module, "__spec__", None)
        origin = None if spec is None else getattr(spec, "origin", None)
        if not isinstance(module_file, str) or not isinstance(origin, str):
            raise MultiHopRAGFormalRunnerError(
                f"{role} loaded module origin is unavailable"
            )
        loaded_file = Path(module_file).resolve(strict=True)
        loaded_origin = Path(origin).resolve(strict=True)
        if loaded_file != expected or loaded_origin != expected:
            raise MultiHopRAGFormalRunnerError(
                f"{role} loaded module is outside the frozen project role path"
            )
        observed[role] = str(expected)
    return observed


def _nonpromotion_receipt(
    *,
    marker_sha256: str,
    assessment: Mapping[str, Any],
    f_policy_freeze: Mapping[str, Any],
    a_hold_seal: Mapping[str, Any],
    a_hold_archive_binding: Mapping[str, Any],
) -> dict[str, Any]:
    delta = assessment.get("family_balanced_delta_total")
    p_value = assessment.get("one_sided_magnitude_signflip_p")
    if (
        assessment.get("status") != "valid_nonpromotion"
        or assessment.get("challenger_promoted") is not False
        or not isinstance(delta, Fraction)
        or not isinstance(p_value, Fraction)
    ):
        raise MultiHopRAGFormalRunnerError(
            "authoritative A_hold nonpromotion assessment drifted"
        )
    freeze_sha256 = verify_self_hash(
        f_policy_freeze, "policy_freeze_sha256"
    )
    seal_sha256 = _require_sha256(
        a_hold_seal.get("action_seal_sha256"), "A_hold action seal"
    )
    archive_file_sha256 = _require_sha256(
        a_hold_archive_binding.get("file_sha256"), "A_hold archive file"
    )
    archive_semantic_sha256 = _require_sha256(
        a_hold_archive_binding.get("semantic_sha256"),
        "A_hold archive semantic",
    )
    if (
        assessment.get("f_search_policy_freeze_sha256") != freeze_sha256
        or assessment.get("a_hold_action_seal_sha256") != seal_sha256
        or assessment.get("a_hold_output_archive_file_sha256")
        != archive_file_sha256
        or assessment.get("a_hold_output_archive_semantic_sha256")
        != archive_semantic_sha256
        or assessment.get("e0_policy_sha256")
        != f_policy_freeze.get("e0_policy_sha256")
        or assessment.get("e1_policy_sha256")
        != f_policy_freeze.get("e1_policy_sha256")
    ):
        raise MultiHopRAGFormalRunnerError(
            "A_hold nonpromotion evidence binding drifted"
        )
    return _self_hashed(
        {
            "schema": RESULT_SCHEMA,
            "version": VERSION,
            "status": "valid_A_hold_nonpromotion_M_unopened",
            "runner_marker_sha256": _require_sha256(
                marker_sha256, "runner marker"
            ),
            "f_search_policy_freeze_sha256": freeze_sha256,
            "a_hold_action_seal_sha256": seal_sha256,
            "a_hold_output_archive_file_sha256": archive_file_sha256,
            "a_hold_output_archive_semantic_sha256": (
                archive_semantic_sha256
            ),
            "e0_policy_sha256": _require_sha256(
                assessment.get("e0_policy_sha256"), "A_hold E0 policy"
            ),
            "e1_policy_sha256": _require_sha256(
                assessment.get("e1_policy_sha256"), "A_hold E1 policy"
            ),
            "family_balanced_delta_total": fraction_payload(delta),
            "one_sided_magnitude_signflip_p": fraction_payload(
                p_value
            ),
            "M_search_view_or_labels_opened": False,
            "external_network_calls": 0,
            "online_evaluator_calls": 0,
            "same_source_replay_authorized": False,
        },
        "result_sha256",
    )


def _nonidentifiable_receipt(
    *,
    marker_sha256: str,
    archive: Mapping[str, Any],
    archive_binding: Mapping[str, Any],
    e0_policy: PolicySelection,
    e1_policy: PolicySelection,
    identifiable: bool,
) -> dict[str, Any]:
    runtime_binding = archive.get("stage_runtime_binding")
    if (
        identifiable is not False
        or not isinstance(runtime_binding, Mapping)
        or set(runtime_binding) != set(acquisition.STAGE_RUNTIME_BINDING_KEYS)
        or e0_policy.input_receipt_sha256
        != e1_policy.input_receipt_sha256
        or e0_policy.selection_sha256
        != recompute_policy_selection_sha256(e0_policy)
        or e1_policy.selection_sha256
        != recompute_policy_selection_sha256(e1_policy)
    ):
        raise MultiHopRAGFormalRunnerError(
            "authoritative F nonidentifiability evidence drifted"
        )
    archive_file_sha256 = _require_sha256(
        archive_binding.get("file_sha256"), "F archive file"
    )
    archive_semantic_sha256 = _require_sha256(
        archive_binding.get("semantic_sha256"), "F archive semantic"
    )
    trace_matrix_sha256 = _require_sha256(
        archive.get("agent_complete_six_action_trace_matrix_sha256"),
        "F complete trace matrix",
    )
    return _self_hashed(
        {
            "schema": RESULT_SCHEMA,
            "version": VERSION,
            "status": "valid_F_search_nonidentifiable_A_hold_and_M_unopened",
            "runner_marker_sha256": _require_sha256(
                marker_sha256, "runner marker"
            ),
            "F_search_policy_freeze_created": False,
            "f_search_output_archive_file_sha256": archive_file_sha256,
            "f_search_output_archive_semantic_sha256": (
                archive_semantic_sha256
            ),
            "complete_f_trace_matrix_receipt_sha256": trace_matrix_sha256,
            "stage_runtime_binding_sha256": stable_hash(
                dict(runtime_binding)
            ),
            "e0_action_id": e0_policy.action_id,
            "e0_policy_sha256": e0_policy.selection_sha256,
            "e1_action_id": e1_policy.action_id,
            "e1_policy_sha256": e1_policy.selection_sha256,
            "policy_input_receipt_sha256": e0_policy.input_receipt_sha256,
            "policies_identifiable": False,
            "A_hold_view_or_labels_opened": False,
            "M_search_view_or_labels_opened": False,
            "external_network_calls": 0,
            "online_evaluator_calls": 0,
            "same_source_replay_authorized": False,
        },
        "result_sha256",
    )


def _verify_archive_capability_binding(
    *,
    block: str,
    archive_binding: Mapping[str, Any],
    capability: Mapping[str, Any],
) -> None:
    file_sha256 = _require_sha256(
        archive_binding.get("file_sha256"), f"{block} archive file"
    )
    semantic_sha256 = _require_sha256(
        archive_binding.get("semantic_sha256"), f"{block} archive semantic"
    )
    if block == "F_search":
        file_field = "f_search_output_archive_file_sha256"
        semantic_field = "f_search_output_archive_semantic_sha256"
    else:
        file_field = "stage_output_archive_file_sha256"
        semantic_field = "stage_output_archive_semantic_sha256"
    if (
        capability.get(file_field) != file_sha256
        or capability.get(semantic_field) != semantic_sha256
    ):
        raise MultiHopRAGFormalRunnerError(
            f"{block} capability does not bind its canonical archive"
        )


def _run_lifecycle_core(
    config: FormalRuntimeConfig,
    *,
    output_paths: _LifecycleOutputPaths,
    formal: bool,
    encoder_factory: Callable[[FormalRuntimeConfig], Encoder] | None = None,
    ner_factory: Callable[[FormalRuntimeConfig], AbstractContextManager[BatchNER]]
    | None = None,
    hippo_factory: Callable[[FormalRuntimeConfig], HippoGateway] | None = None,
    prepare_corpus_fn: Callable[..., PreparedCorpus] = prepare_offline_corpus,
    execute_stage_fn: Callable[..., StageExecution] = execute_gold_free_stage,
    executor_factory: ExecutorFactory = spawn_process_pool_executor,
) -> dict[str, Any]:
    """Private common core; only the public wrapper may select formal outputs."""

    if not isinstance(config, FormalRuntimeConfig):
        raise MultiHopRAGFormalRunnerError("formal runtime config is invalid")
    project = config.project.resolve(strict=True)
    if formal:
        if any(
            value is not None
            for value in (encoder_factory, ner_factory, hippo_factory)
        ) or prepare_corpus_fn is not prepare_offline_corpus or execute_stage_fn is not execute_gold_free_stage or executor_factory is not spawn_process_pool_executor:
            raise MultiHopRAGFormalRunnerError(
                "formal lifecycle runtime overrides are forbidden"
            )
        preflight_formal_runtime_config(config)
    acquisition_receipt, _binding = acquisition.load_committed_acquisition_receipt(
        project
    )
    implementation = acquisition.verify_committed_implementation_freeze(project)
    if formal:
        _verify_loaded_frozen_module_origins(
            project=project,
            implementation_receipt=implementation,
        )
    marker = consume_one_shot_marker(
        path=project / output_paths.marker,
        phase="formal_A_form_F_A_hold_M_one_shot",
        bindings={
            "acquisition_sha256": acquisition_receipt["acquisition_sha256"],
            "implementation_freeze_sha256": implementation[
                "implementation_freeze_sha256"
            ],
        },
    )
    marker_sha = str(marker["marker_sha256"])
    failure_stage = "runtime_initialization"
    try:
        encoder = (
            OfflineMiniLMEncoder(
                asset_manifest_path=config.minilm_asset_manifest,
                model_root=config.minilm_model_root,
            )
            if encoder_factory is None
            else encoder_factory(config)
        )
        hippo = (
            OfficialHippoGateway(
                runtime_python=config.hippo_runtime_python,
                local_llm_model=config.hippo_llm_model,
                local_embedding_model=config.hippo_embedding_model,
                base_binding_receipt_path=config.hippo_base_binding_receipt,
                attestation_receipt_path=config.hippo_attestation_receipt,
                stage_root=config.hippo_stage_root,
                work_root=config.hippo_work_root,
            )
            if hippo_factory is None
            else hippo_factory(config)
        )
        if config.hippo_work_root.exists():
            raise MultiHopRAGFormalRunnerError("Hippo work root is not fresh")
        config.hippo_work_root.mkdir(parents=True, mode=0o700)
        ner_context = (
            OfflineNERJSONLClient(
                project_root=project,
                asset_manifest_path=config.ner_asset_manifest,
                model_root=config.ner_model_root,
            )
            if ner_factory is None
            else ner_factory(config)
        )
        with ner_context as ner:
            failure_stage = "corpus_preparation"
            prepared = prepare_corpus_fn(
                corpus_view=acquisition.load_corpus_view(project=project),
                encoder=encoder,
                ner=ner,
                hippo=hippo,
                ner_batch_size=config.ner_batch_size,
                formal_shape=True,
            )

            def execute_and_archive(
                block: str,
            ) -> tuple[StageExecution, dict[str, Any], dict[str, Any]]:
                nonlocal failure_stage
                failure_stage = f"{block}_gold_free_execution"
                stage = execute_stage_fn(
                    block=block,
                    view=acquisition.load_block_view(
                        project=project, expected_block=block
                    ),
                    prepared=prepared,
                    encoder=encoder,
                    ner=ner,
                    hippo=hippo,
                    ner_batch_size=config.ner_batch_size,
                    local_worker_cap=config.local_worker_cap,
                    formal_shape=True,
                    executor_factory=executor_factory,
                )
                failure_stage = f"{block}_canonical_archive"
                archive, archive_binding, _runtime_binding = (
                    persist_canonical_stage_archive(
                    project=project, prepared=prepared, stage=stage
                    )
                )
                return stage, archive, archive_binding

            a_form, _a_form_archive, a_form_archive_binding = (
                execute_and_archive("A_form")
            )
            failure_stage = "A_form_action_seal"
            a_form_seal = acquisition.create_action_seal_once(
                project=project, block="A_form"
            )
            _verify_archive_capability_binding(
                block="A_form",
                archive_binding=a_form_archive_binding,
                capability=a_form_seal,
            )
            failure_stage = "A_form_prelabel_policy_freeze"
            a_form_policy_freeze = (
                acquisition.create_a_form_policy_freeze_once(project=project)
            )
            if (
                a_form_policy_freeze.get(
                    "a_form_output_archive_file_sha256"
                )
                != a_form_archive_binding["file_sha256"]
                or a_form_policy_freeze.get(
                    "a_form_output_archive_semantic_sha256"
                )
                != a_form_archive_binding["semantic_sha256"]
                or a_form_policy_freeze.get("a_form_action_seal_sha256")
                != a_form_seal["action_seal_sha256"]
                or a_form_policy_freeze.get(
                    "complete_a_form_trace_matrix_receipt_sha256"
                )
                != _a_form_archive[
                    "agent_complete_six_action_trace_matrix_sha256"
                ]
            ):
                raise MultiHopRAGFormalRunnerError(
                    "A_form policy freeze capability binding drifted"
                )
            failure_stage = "A_form_descriptive_labels"
            a_form_report = descriptive_a_form(
                stage=a_form,
                policy_freeze=a_form_policy_freeze,
                labels=acquisition.load_block_labels(
                    project=project, expected_block="A_form"
                ),
            )
            write_json_exclusive(
                project / output_paths.a_form_descriptive,
                a_form_report,
                mode=0o600,
            )

            _f_stage, _f_archive, f_archive_binding = execute_and_archive(
                "F_search"
            )
            failure_stage = "F_search_canonical_policy_freeze"
            f_e0, f_e1, f_identifiable = (
                acquisition._recompute_f_search_policy_selections(_f_archive)
            )
            try:
                f_freeze = acquisition.create_f_search_policy_freeze_once(
                    project=project
                )
            except acquisition.MultiHopRAGAcquisitionError as exc:
                if str(exc) != "F policies are not identifiable":
                    raise
                result = _nonidentifiable_receipt(
                    marker_sha256=marker_sha,
                    archive=_f_archive,
                    archive_binding=f_archive_binding,
                    e0_policy=f_e0,
                    e1_policy=f_e1,
                    identifiable=f_identifiable,
                )
                write_json_exclusive(
                    project / output_paths.result, result, mode=0o644
                )
                return result
            if (
                f_identifiable is not True
                or f_freeze.get("e0_action_id") != f_e0.action_id
                or f_freeze.get("e0_policy_sha256")
                != f_e0.selection_sha256
                or f_freeze.get("e1_action_id") != f_e1.action_id
                or f_freeze.get("e1_policy_sha256")
                != f_e1.selection_sha256
            ):
                raise MultiHopRAGFormalRunnerError(
                    "F policy freeze does not bind authoritative selections"
                )
            _verify_archive_capability_binding(
                block="F_search",
                archive_binding=f_archive_binding,
                capability=f_freeze,
            )

            _a_hold, _a_hold_archive, a_hold_archive_binding = (
                execute_and_archive("A_hold")
            )
            failure_stage = "A_hold_action_seal"
            a_hold_seal = acquisition.create_action_seal_once(
                project=project, block="A_hold"
            )
            _verify_archive_capability_binding(
                block="A_hold",
                archive_binding=a_hold_archive_binding,
                capability=a_hold_seal,
            )
            failure_stage = "A_hold_late_labels_and_promotion"
            a_hold_assessment = acquisition.assess_a_hold_promotion(
                project=project
            )
            if a_hold_assessment.get("challenger_promoted") is not True:
                result = _nonpromotion_receipt(
                    marker_sha256=marker_sha,
                    assessment=a_hold_assessment,
                    f_policy_freeze=f_freeze,
                    a_hold_seal=a_hold_seal,
                    a_hold_archive_binding=a_hold_archive_binding,
                )
                write_json_exclusive(
                    project / output_paths.result, result, mode=0o644
                )
                return result
            promotion = acquisition.create_a_hold_promotion_once(
                project=project
            )

            _m_stage, _m_archive, m_archive_binding = execute_and_archive(
                "M_search"
            )
            failure_stage = "M_search_action_seal"
            m_seal = acquisition.create_action_seal_once(
                project=project, block="M_search"
            )
            _verify_archive_capability_binding(
                block="M_search",
                archive_binding=m_archive_binding,
                capability=m_seal,
            )
            failure_stage = "M_search_late_labels_and_exact_assessment"
            assessment = acquisition.assess_m_search(project=project)
            result = make_result_receipt(
                assessment=assessment,
                promotion=promotion,
                m_action_seal=m_seal,
                runner_marker=marker,
            )
            write_json_exclusive(
                project / output_paths.result, result, mode=0o644
            )
            return result
    except BaseException as exc:
        write_terminal_failure(
            path=project / output_paths.failure,
            marker_sha256=marker_sha,
            stage=failure_stage,
            exc=exc,
        )
        raise


def run_formal_lifecycle(config: FormalRuntimeConfig) -> dict[str, Any]:
    """Run the formal lifecycle with no injectable runtime or execution hook."""

    return _run_lifecycle_core(
        config,
        output_paths=_LifecycleOutputPaths(
            marker=RUNNER_MARKER_RELATIVE,
            failure=RUNNER_FAILURE_RELATIVE,
            result=TERMINAL_RESULT_RELATIVE,
            a_form_descriptive=A_FORM_DESCRIPTIVE_RELATIVE,
        ),
        formal=True,
    )


def _run_synthetic_lifecycle_core(
    config: FormalRuntimeConfig,
    *,
    encoder_factory: Callable[[FormalRuntimeConfig], Encoder],
    ner_factory: Callable[[FormalRuntimeConfig], AbstractContextManager[BatchNER]],
    hippo_factory: Callable[[FormalRuntimeConfig], HippoGateway],
    prepare_corpus_fn: Callable[..., PreparedCorpus],
    execute_stage_fn: Callable[..., StageExecution],
    executor_factory: ExecutorFactory,
) -> dict[str, Any]:
    """Private test core that cannot consume or write formal capabilities."""

    project = config.project.resolve(strict=True)
    sentinel = project / _SYNTHETIC_SENTINEL
    if (
        sentinel.is_symlink()
        or not sentinel.is_file()
        or sentinel.read_text(encoding="ascii") != _SYNTHETIC_SENTINEL_CONTENT
    ):
        raise MultiHopRAGFormalRunnerError(
            "synthetic lifecycle sentinel is absent"
        )
    formal_paths = (
        RUNNER_MARKER_RELATIVE,
        RUNNER_FAILURE_RELATIVE,
        TERMINAL_RESULT_RELATIVE,
        A_FORM_DESCRIPTIVE_RELATIVE,
        *acquisition.STAGE_OUTPUT_ARCHIVE_RELATIVES.values(),
        *acquisition.ACTION_SEAL_RELATIVES.values(),
        acquisition.F_POLICY_FREEZE_RELATIVE,
        acquisition.PROMOTION_RELATIVE,
    )
    if any(
        (project / relative).exists() or (project / relative).is_symlink()
        for relative in formal_paths
    ):
        raise MultiHopRAGFormalRunnerError(
            "synthetic lifecycle cannot coexist with formal canonical paths"
        )
    prefix = "artifacts/multihoprag_joint_graph_synthetic_test_only"
    return _run_lifecycle_core(
        config,
        output_paths=_LifecycleOutputPaths(
            marker=f"{prefix}/runner.marker.json",
            failure=f"{prefix}/runner.failure.json",
            result=f"{prefix}/result.json",
            a_form_descriptive=f"{prefix}/A_form.descriptive.json",
        ),
        formal=False,
        encoder_factory=encoder_factory,
        ner_factory=ner_factory,
        hippo_factory=hippo_factory,
        prepare_corpus_fn=prepare_corpus_fn,
        execute_stage_fn=execute_stage_fn,
        executor_factory=executor_factory,
    )


def write_json_exclusive(
    path: Path, payload: Mapping[str, Any], *, mode: int
) -> str:
    """O_EXCL write through nofollow ancestor dir_fds, then fsync file and dir."""

    if mode not in {0o600, 0o644}:
        raise MultiHopRAGFormalRunnerError("output mode is invalid")
    absolute = Path(os.path.abspath(os.fspath(path)))
    if (
        not absolute.is_absolute()
        or absolute.name in {"", ".", ".."}
        or "/" in absolute.name
        or "\x00" in absolute.name
    ):
        raise MultiHopRAGFormalRunnerError("exclusive output path is unsafe")
    raw = _canonical_bytes(payload) + b"\n"
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    parent_fd = os.open(os.path.sep, directory_flags)
    try:
        for component in absolute.parts[1:-1]:
            try:
                child_fd = os.open(
                    component, directory_flags, dir_fd=parent_fd
                )
            except FileNotFoundError:
                os.mkdir(component, 0o700, dir_fd=parent_fd)
                os.fsync(parent_fd)
                child_fd = os.open(
                    component, directory_flags, dir_fd=parent_fd
                )
            except OSError as exc:
                raise MultiHopRAGFormalRunnerError(
                    "exclusive output ancestor is unsafe"
                ) from exc
            os.close(parent_fd)
            parent_fd = child_fd
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(
                absolute.name, flags, mode, dir_fd=parent_fd
            )
        except OSError as exc:
            raise MultiHopRAGFormalRunnerError(
                "exclusive output already exists or is unsafe"
            ) from exc
        try:
            os.fchmod(descriptor, mode)
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            info = os.stat(
                absolute.name, dir_fd=parent_fd, follow_symlinks=False
            )
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != mode:
                raise MultiHopRAGFormalRunnerError(
                    "exclusive output type or mode drifted"
                )
            os.fsync(parent_fd)
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
    finally:
        os.close(parent_fd)
    return hashlib.sha256(raw).hexdigest()


def consume_one_shot_marker(
    *, path: Path, phase: str, bindings: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(phase, str) or not phase or "\x00" in phase:
        raise MultiHopRAGFormalRunnerError("marker phase is invalid")
    body = {
        "schema": RUNNER_MARKER_SCHEMA,
        "version": VERSION,
        "phase": phase,
        "bindings": dict(bindings),
        "replay_retry_resample_replacement_authorized": False,
    }
    marker = _self_hashed(body, "marker_sha256")
    write_json_exclusive(path, marker, mode=0o600)
    return marker


def write_terminal_failure(
    *,
    path: Path,
    marker_sha256: str,
    stage: str,
    exc: BaseException,
) -> None:
    body = {
        "schema": RUNNER_FAILURE_SCHEMA,
        "version": VERSION,
        "status": "terminal_cohort_burned_no_replay",
        "marker_sha256": _require_sha256(marker_sha256, "runner marker"),
        "failure_stage": stage,
        "exception_type_sha256": hashlib.sha256(
            f"{type(exc).__module__}.{type(exc).__qualname__}".encode("utf-8")
        ).hexdigest(),
        "private_item_or_label_content_included": False,
        "replay_retry_resample_replacement_authorized": False,
    }
    try:
        write_json_exclusive(
            path,
            _self_hashed(body, "failure_sha256"),
            mode=0o644,
        )
    except BaseException:
        pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = default_formal_runtime_config(args.project)
    result = run_formal_lifecycle(config)
    print(_canonical_bytes(result).decode("ascii"))
    return 0


__all__ = [
    "DEFAULT_NER_BATCH_SIZE",
    "FormalRuntimeConfig",
    "HippoGateway",
    "LabelFreeArticle",
    "LOCAL_CONCURRENCY_CAP",
    "MSearchAssessment",
    "MultiHopRAGFormalRunnerError",
    "NER_PROCESS_COUNT",
    "OfflineNERJSONLClient",
    "OfficialHippoGateway",
    "PreparedCorpus",
    "PromotionDecision",
    "StageExecution",
    "StageItem",
    "VERSION",
    "assess_m_search",
    "build_canonical_stage_records",
    "build_stage_runtime_binding",
    "compile_query_features_batched",
    "consume_one_shot_marker",
    "decide_a_hold_promotion",
    "descriptive_a_form",
    "default_formal_runtime_config",
    "execute_agent_actions_eager",
    "execute_gold_free_stage",
    "fraction_payload",
    "make_result_receipt",
    "persist_canonical_stage_archive",
    "preflight_formal_runtime_config",
    "prepare_offline_corpus",
    "raw_top5",
    "run_formal_lifecycle",
    "select_f_policies",
    "spawn_process_pool_executor",
    "stable_hash",
    "stage_execution_matrix_sha256",
    "verify_self_hash",
    "write_json_exclusive",
    "write_terminal_failure",
]


if __name__ == "__main__":
    raise SystemExit(main())
