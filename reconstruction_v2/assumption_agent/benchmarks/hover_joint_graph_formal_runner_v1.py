"""Gold-isolated offline core for the HoVer joint-graph study.

This module deliberately contains no official-source reader and no formal
entrypoint.  It compiles one fixed corpus, runs RAW, official HippoRAG, and all
six typed Agent actions, and applies already-frozen policies to late labels.
The small :class:`HoverAcquisitionAdapter` is the sole coupling point to the
eventual acquisition envelopes; the execution core never accepts gold titles,
hop labels, support sentences, or verdicts.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import json
import multiprocessing
import re
from typing import Any, Protocol

from assumption_agent.benchmarks import hover_direct_acquisition_v1 as acquisition
from assumption_agent.benchmarks.multihoprag_joint_graph_formal_runner_v1 import (
    compile_query_features_batched,
    raw_top5,
)
from assumption_agent.benchmarks.multihoprag_typed_operator_v2 import (
    ACTION_IDS,
    CAPABILITIES,
    SAME_SOURCE,
    ActionTrace,
    ArticleRecord,
    EvaluationObservation,
    FrozenMapping,
    PolicySelection,
    QueryPlan,
    TypedCorpusGraph,
    build_typed_corpus_graph,
    compile_query_plan,
    item_utility,
    make_entity_key,
    normalize_text,
    paired_utility_summary,
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
    reciprocal_topic_neighbors,
    validate_corpus_embedding_index,
)
from replication_runtime.multihoprag_ner_v1 import EntitySpan
from replication_runtime.multihoprag_official_hipporag_v1 import RetrievalBatch


VERSION = "hover_joint_graph_formal_runner_v1"
ACQUISITION_VERSION = acquisition.VERSION

CORPUS_COUNT = acquisition.CORPUS_SIZE
TOP_K = 5
LOCAL_CONCURRENCY_CAP = 32
DEFAULT_NER_BATCH_SIZE = 32
OFFICIAL_HIPPO_QUERY_BATCH_CAP = 8

BLOCK_ORDER = acquisition.BLOCK_ORDER
BLOCK_COUNTS = dict(acquisition.BLOCK_COUNTS)
HOP_STRATA = acquisition.HOP_STRATA
HOP_QUOTAS = dict(acquisition.HOP_QUOTAS)
E0_ID = "E0_INDEPENDENT_V2"
E1_ID = "E1_CAUSAL_NECESSITY_V2"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ACTION_WORKER_GRAPH: TypedCorpusGraph | None = None


class HoVerFormalRunnerError(RuntimeError):
    """A HoVer runner, acquisition-envelope, or scoring invariant drifted."""


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
        raise HoVerFormalRunnerError("value is not canonical JSON") from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise HoVerFormalRunnerError(f"{field} is not a SHA256")
    return value


def _self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in body:
        raise HoVerFormalRunnerError("self-hash field already exists")
    return {**body, field: stable_hash(body)}


def verify_self_hash(payload: Mapping[str, Any], field: str) -> str:
    declared = _require_sha256(payload.get(field), field)
    body = dict(payload)
    del body[field]
    if stable_hash(body) != declared:
        raise HoVerFormalRunnerError(f"{field} self-hash mismatch")
    return declared


def fraction_payload(value: Fraction) -> list[int]:
    if not isinstance(value, Fraction):
        raise HoVerFormalRunnerError("exact statistic is not a Fraction")
    return [value.numerator, value.denominator]


@dataclass(frozen=True)
class HoverArticle:
    article_i: int
    title: str
    body: str

    def hippo_payload(self) -> dict[str, object]:
        return {"idx": self.article_i, "title": self.title, "body": self.body}

    def ner_payload(self) -> dict[str, object]:
        return {"kind": "article", "title": self.title, "body": self.body}


@dataclass(frozen=True)
class PreparedCorpus:
    articles: tuple[HoverArticle, ...]
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
class LateLabel:
    ordinal: int
    view_sha256: str
    identity_commitment_sha256: str
    source_record_commitment_sha256: str
    hop_stratum: str
    gold_article_ids: tuple[int, ...]


@dataclass(frozen=True)
class AHoldAssessment:
    promoted: bool
    primary_passed: bool
    e0_policy: PolicySelection
    e1_policy: PolicySelection
    promotion_delta_total: Fraction
    promotion_signflip_p: Fraction
    e0_minus_hippo_delta_total: Fraction
    e0_minus_hippo_signflip_p: Fraction
    e0_minus_hippo_stratum_deltas: tuple[tuple[str, Fraction], ...]
    e0_minus_raw_delta_total: Fraction
    e0_minus_raw_signflip_p: Fraction
    e0_complete_count: int
    raw_complete_count: int


@dataclass(frozen=True)
class MSearchAssessment:
    l5_delta_total: Fraction
    l5_signflip_p: Fraction
    l5_passed: bool
    e1_minus_hippo_delta_total: Fraction
    e1_minus_hippo_signflip_p: Fraction
    e1_minus_hippo_stratum_deltas: tuple[tuple[str, Fraction], ...]
    e1_minus_raw_delta_total: Fraction
    e1_minus_raw_signflip_p: Fraction
    e1_complete_count: int
    raw_complete_count: int


class BatchNER(Protocol):
    runtime_binding: Mapping[str, object]
    canary_receipt: Mapping[str, object]

    def extract_inputs(
        self, values: Sequence[Mapping[str, object]]
    ) -> tuple[tuple[EntitySpan, ...], ...]: ...


class Encoder(Protocol):
    runtime_receipt: Mapping[str, object]
    canary_receipt: Mapping[str, object]

    def encode(self, texts: Sequence[str]) -> Any: ...


class HippoGateway(Protocol):
    def build(self, articles: Sequence[Mapping[str, object]]) -> Mapping[str, Any]: ...

    def retrieve(self, *, block: str, queries: Sequence[str]) -> RetrievalBatch: ...


class ExecutorLike(Protocol):
    def submit(self, fn: Callable[..., Any], /, *args: Any) -> Future[Any]: ...

    def __enter__(self) -> "ExecutorLike": ...

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...


ExecutorFactory = Callable[..., AbstractContextManager[ExecutorLike]]


def spawn_process_pool_executor(**kwargs: Any) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        mp_context=multiprocessing.get_context("spawn"),
        **kwargs,
    )


@dataclass(frozen=True)
class HoverAcquisitionAdapter:
    """All envelope/schema assumptions live in this replaceable adapter."""

    version: str = ACQUISITION_VERSION
    corpus_view_schema: str = acquisition.CORPUS_VIEW_SCHEMA
    block_view_schema: str = acquisition.BLOCK_VIEW_SCHEMA
    view_item_schema: str = acquisition.VIEW_ITEM_SCHEMA
    block_label_schema: str = acquisition.BLOCK_LABEL_SCHEMA
    label_item_schema: str = acquisition.LABEL_ITEM_SCHEMA

    def validate_corpus_view(
        self, corpus_view: Mapping[str, Any], *, formal_shape: bool
    ) -> tuple[HoverArticle, ...]:
        verify_self_hash(corpus_view, "corpus_view_sha256")
        articles = corpus_view.get("articles")
        if (
            set(corpus_view)
            != {
                "schema",
                "version",
                "article_count",
                "origin_or_gold_membership_included",
                "articles",
                "corpus_view_sha256",
            }
            or corpus_view.get("schema") != self.corpus_view_schema
            or corpus_view.get("version") != self.version
            or corpus_view.get("origin_or_gold_membership_included") is not False
            or not isinstance(articles, list)
            or len(articles) < TOP_K
            or corpus_view.get("article_count") != len(articles)
            or (formal_shape and len(articles) != CORPUS_COUNT)
        ):
            raise HoVerFormalRunnerError("HoVer corpus view envelope drifted")
        output: list[HoverArticle] = []
        for article_i, row in enumerate(articles):
            if (
                not isinstance(row, Mapping)
                or set(row) != {"article_id", "title", "body"}
                or type(row.get("article_id")) is not int
                or row.get("article_id") != article_i
                or not isinstance(row.get("title"), str)
                or not str(row["title"]).strip()
                or "\x00" in str(row["title"])
                or not isinstance(row.get("body"), str)
                or not str(row["body"]).strip()
                or "\x00" in str(row["body"])
            ):
                raise HoVerFormalRunnerError("HoVer corpus article drifted")
            output.append(
                HoverArticle(
                    article_i=article_i,
                    title=str(row["title"]),
                    body=str(row["body"]),
                )
            )
        return tuple(output)

    def validate_block_view(
        self,
        view: Mapping[str, Any],
        *,
        block: str,
        formal_shape: bool,
    ) -> tuple[Mapping[str, Any], ...]:
        verify_self_hash(view, "block_view_sha256")
        items = view.get("items")
        if (
            set(view)
            != {
                "schema",
                "version",
                "block",
                "item_count",
                "late_utility_fields_included",
                "items",
                "block_view_sha256",
            }
            or view.get("schema") != self.block_view_schema
            or view.get("version") != self.version
            or block not in BLOCK_ORDER
            or view.get("block") != block
            or view.get("late_utility_fields_included") is not False
            or not isinstance(items, list)
            or not items
            or view.get("item_count") != len(items)
            or (formal_shape and len(items) != BLOCK_COUNTS[block])
        ):
            raise HoVerFormalRunnerError("HoVer block view envelope drifted")
        for ordinal, item in enumerate(items):
            if (
                not isinstance(item, Mapping)
                or set(item) != {"schema", "block", "ordinal", "claim"}
                or item.get("schema") != self.view_item_schema
                or item.get("block") != block
                or type(item.get("ordinal")) is not int
                or item.get("ordinal") != ordinal
                or not isinstance(item.get("claim"), str)
                or not str(item["claim"]).strip()
                or "\x00" in str(item["claim"])
            ):
                raise HoVerFormalRunnerError("HoVer block view item drifted")
        return tuple(items)

    def join_late_labels(
        self, stage: StageExecution, labels: Mapping[str, Any]
    ) -> tuple[LateLabel, ...]:
        verify_self_hash(labels, "block_labels_sha256")
        items = labels.get("items")
        if (
            set(labels)
            != {
                "schema",
                "version",
                "block",
                "item_count",
                "source_or_verdict_payload_included",
                "items",
                "block_labels_sha256",
            }
            or labels.get("schema") != self.block_label_schema
            or labels.get("version") != self.version
            or labels.get("block") != stage.block
            or labels.get("source_or_verdict_payload_included") is not False
            or not isinstance(items, list)
            or labels.get("item_count") != len(items)
            or len(items) != len(stage.items)
        ):
            raise HoVerFormalRunnerError("HoVer late-label envelope drifted")
        expected_keys = {
            "schema",
            "block",
            "ordinal",
            "view_sha256",
            "identity_commitment_sha256",
            "source_record_commitment_sha256",
            "hop_stratum",
            "gold_article_ids",
        }
        by_view: dict[str, LateLabel] = {}
        identities: set[str] = set()
        source_records: set[str] = set()
        strata: Counter[str] = Counter()
        for ordinal, item in enumerate(items):
            if not isinstance(item, Mapping) or set(item) != expected_keys:
                raise HoVerFormalRunnerError("HoVer late-label row schema drifted")
            stratum = item.get("hop_stratum")
            gold = item.get("gold_article_ids")
            expected_gold_count = (
                int(str(stratum)[0]) if stratum in HOP_STRATA else -1
            )
            if (
                item.get("schema") != self.label_item_schema
                or item.get("block") != stage.block
                or item.get("ordinal") != ordinal
                or stratum not in HOP_STRATA
                or not isinstance(gold, list)
                or len(gold) != expected_gold_count
                or not 2 <= len(gold) <= 4
                or gold != sorted(set(gold))
                or any(type(value) is not int or not 0 <= value < CORPUS_COUNT for value in gold)
            ):
                raise HoVerFormalRunnerError("HoVer late-label row drifted")
            view_sha = _require_sha256(item.get("view_sha256"), "label view")
            identity = _require_sha256(
                item.get("identity_commitment_sha256"), "label identity"
            )
            source_record = _require_sha256(
                item.get("source_record_commitment_sha256"), "label source record"
            )
            if view_sha in by_view or identity in identities or source_record in source_records:
                raise HoVerFormalRunnerError("HoVer late-label commitment overlaps")
            row = LateLabel(
                ordinal=ordinal,
                view_sha256=view_sha,
                identity_commitment_sha256=identity,
                source_record_commitment_sha256=source_record,
                hop_stratum=str(stratum),
                gold_article_ids=tuple(gold),
            )
            by_view[view_sha] = row
            identities.add(identity)
            source_records.add(source_record)
            strata[row.hop_stratum] += 1
        if stage.formal_shape and strata != Counter(
            {stratum: HOP_QUOTAS[stage.block] for stratum in HOP_STRATA}
        ):
            raise HoVerFormalRunnerError("formal HoVer hop quotas drifted")
        view_items = stage.view.get("items")
        if not isinstance(view_items, list) or len(view_items) != len(stage.items):
            raise HoVerFormalRunnerError("late-label view cardinality drifted")
        joined: list[LateLabel] = []
        for ordinal, view_item in enumerate(view_items):
            row = by_view.get(stable_hash(view_item))
            if row is None or row.ordinal != ordinal:
                raise HoVerFormalRunnerError("HoVer late-label join is incomplete")
            joined.append(row)
        return tuple(joined)


DEFAULT_ACQUISITION_ADAPTER = HoverAcquisitionAdapter()


def _bounded_ner_extract(
    runtime: BatchNER,
    values: Sequence[Mapping[str, object]],
    *,
    batch_size: int,
) -> tuple[tuple[EntitySpan, ...], ...]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 256:
        raise HoVerFormalRunnerError("NER batch size is outside 1..256")
    rows = tuple(values)
    if not rows:
        raise HoVerFormalRunnerError("NER input is empty")
    output: list[tuple[EntitySpan, ...]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        extracted = runtime.extract_inputs(batch)
        if not isinstance(extracted, tuple) or len(extracted) != len(batch):
            raise HoVerFormalRunnerError("NER output cardinality drifted")
        output.extend(extracted)
    return tuple(output)


def _ner_receipt_sha256(runtime: BatchNER) -> str:
    binding = getattr(runtime, "runtime_binding", None)
    canary = getattr(runtime, "canary_receipt", None)
    if (
        not isinstance(binding, Mapping)
        or not isinstance(canary, Mapping)
        or canary.get("status") != "passed_exact_row_free_synthetic_canary"
    ):
        raise HoVerFormalRunnerError("NER runtime receipts are unavailable")
    return stable_hash({"runtime_binding": dict(binding), "canary_receipt": dict(canary)})


def _missing_source(article_i: int) -> str:
    return normalize_text(f"__hover_reserved_missing_source_{article_i:06d}__")


def _missing_category(article_i: int) -> str:
    return normalize_text(f"__hover_reserved_missing_category_{article_i:06d}__")


def prepare_offline_corpus(
    *,
    corpus_view: Mapping[str, Any],
    encoder: Encoder,
    ner: BatchNER,
    hippo: HippoGateway,
    ner_batch_size: int = DEFAULT_NER_BATCH_SIZE,
    formal_shape: bool = True,
    acquisition_adapter: HoverAcquisitionAdapter = DEFAULT_ACQUISITION_ADAPTER,
) -> PreparedCorpus:
    """Compile corpus features once without inventing shared metadata edges."""

    articles = acquisition_adapter.validate_corpus_view(
        corpus_view, formal_shape=formal_shape
    )
    hippo_rows = tuple(article.hippo_payload() for article in articles)
    ner_rows = tuple(article.ner_payload() for article in articles)
    article_texts = tuple(
        ArticleText(article.article_i, article.title, article.body)
        for article in articles
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
        hippo_receipt = hippo_future.result()
    validate_corpus_embedding_index(embedding_index)
    topic_neighbors = reciprocal_topic_neighbors(embedding_index)
    if len(entity_rows) != len(articles) or len(topic_neighbors) != len(articles):
        raise HoVerFormalRunnerError("compiled corpus cardinality drifted")
    typed: list[ArticleRecord] = []
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
        typed.append(
            ArticleRecord(
                article_i=article.article_i,
                normalized_source=_missing_source(article.article_i),
                normalized_category=_missing_category(article.article_i),
                published_ordinal=None,
                entities=entities,
                reciprocal_topic_neighbors=neighbors,
            )
        )
    graph = build_typed_corpus_graph(typed)
    if len(graph.sources) != len(articles) or any(graph.neighbors[SAME_SOURCE]):
        raise HoVerFormalRunnerError("reserved missing-source codec created false edges")
    if any(
        article.published_ordinal is not None
        or article.normalized_source != _missing_source(article.article_i)
        or article.normalized_category != _missing_category(article.article_i)
        for article in graph.articles
    ):
        raise HoVerFormalRunnerError("reserved missing metadata codec drifted")
    if not isinstance(hippo_receipt, Mapping):
        raise HoVerFormalRunnerError("Hippo build receipt is absent")
    ner_receipt = _ner_receipt_sha256(ner)
    body = {
        "version": VERSION,
        "corpus_view_sha256": _require_sha256(
            corpus_view.get("corpus_view_sha256"), "corpus view"
        ),
        "embedding_index_sha256": embedding_index.index_sha256,
        "graph_sha256": graph.graph_sha256,
        "hippo_build_receipt_sha256": stable_hash(dict(hippo_receipt)),
        "ner_runtime_receipt_sha256": ner_receipt,
        "ner_entity_matrix_sha256": stable_hash(entity_receipt_rows),
        "missing_metadata_codec": "distinct_reserved_per_article_and_date_None",
        "offline_network_calls": 0,
        "online_evaluator_calls": 0,
    }
    return PreparedCorpus(
        articles=articles,
        corpus_view_sha256=str(corpus_view["corpus_view_sha256"]),
        graph=graph,
        embedding_index=embedding_index,
        hippo_build_receipt=dict(hippo_receipt),
        ner_runtime_receipt_sha256=ner_receipt,
        ner_entity_matrix_sha256=body["ner_entity_matrix_sha256"],
        preparation_sha256=stable_hash(body),
    )


def _init_action_worker(graph: TypedCorpusGraph) -> None:
    global _ACTION_WORKER_GRAPH
    _ACTION_WORKER_GRAPH = graph


def _run_item_actions(
    ordinal: int, plan: QueryPlan, relevance: tuple[int, ...]
) -> tuple[int, tuple[ActionTrace, ...]]:
    if _ACTION_WORKER_GRAPH is None:
        raise HoVerFormalRunnerError("Agent worker graph is uninitialized")
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
    """Submit all item futures before joining the first, capped at 32."""

    if (
        isinstance(local_worker_cap, bool)
        or not isinstance(local_worker_cap, int)
        or not 1 <= local_worker_cap <= LOCAL_CONCURRENCY_CAP
    ):
        raise HoVerFormalRunnerError("local worker cap is outside 1..32")
    plan_rows = tuple(plans)
    relevance_rows = tuple(tuple(row) for row in relevance_vectors)
    if not plan_rows or len(plan_rows) != len(relevance_rows):
        raise HoVerFormalRunnerError("Agent input cardinality drifted")
    with executor_factory(
        max_workers=min(local_worker_cap, len(plan_rows)),
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
        raise HoVerFormalRunnerError("Agent result order drifted")
    output = tuple(traces for _ordinal, traces in results)
    for traces in output:
        if tuple(trace.action_id for trace in traces) != ACTION_IDS or any(
            trace.trace_sha256 != recompute_action_trace_sha256(trace)
            for trace in traces
        ):
            raise HoVerFormalRunnerError("Agent trace matrix drifted")
    return output


def _validate_hippo_indices(
    batch: RetrievalBatch, *, count: int, corpus_count: int
) -> tuple[tuple[int, int, int, int, int], ...]:
    if not isinstance(batch, RetrievalBatch) or len(batch.indices) != count:
        raise HoVerFormalRunnerError("Hippo result cardinality drifted")
    output: list[tuple[int, int, int, int, int]] = []
    for row in batch.indices:
        if (
            not isinstance(row, tuple)
            or len(row) != TOP_K
            or len(set(row)) != TOP_K
            or any(type(value) is not int or not 0 <= value < corpus_count for value in row)
        ):
            raise HoVerFormalRunnerError("Hippo top5 drifted")
        output.append(row)  # type: ignore[arg-type]
    receipt = batch.receipt
    batch_sizes = receipt.get("batch_sizes") if isinstance(receipt, Mapping) else None
    if (
        not isinstance(batch_sizes, list)
        or any(type(value) is not int or not 1 <= value <= OFFICIAL_HIPPO_QUERY_BATCH_CAP for value in batch_sizes)
        or sum(batch_sizes) != count
    ):
        raise HoVerFormalRunnerError("Hippo query batch contract drifted")
    return tuple(output)


def stage_execution_matrix_sha256(
    items: Sequence[StageItem], *, expected_embedding_index_sha256: str
) -> str:
    rows = tuple(items)
    if not rows:
        raise HoVerFormalRunnerError("stage execution is empty")
    body: list[dict[str, object]] = []
    for ordinal, item in enumerate(rows):
        if (
            not isinstance(item, StageItem)
            or item.ordinal != ordinal
            or item.query_sha256 != item.plan.query_sha256
            or item.query_feature.normalized_query_sha256 != item.query_sha256
            or item.query_feature.embedding_index_sha256
            != expected_embedding_index_sha256
            or tuple(trace.action_id for trace in item.traces) != ACTION_IDS
            or any(
                trace.trace_sha256 != recompute_action_trace_sha256(trace)
                for trace in item.traces
            )
        ):
            raise HoVerFormalRunnerError("stage item binding drifted")
        for output in (item.raw_top5, item.hippo_top5):
            if (
                len(output) != TOP_K
                or len(set(output)) != TOP_K
                or any(type(value) is not int or not 0 <= value < CORPUS_COUNT for value in output)
            ):
                raise HoVerFormalRunnerError("stage output drifted")
        body.append(
            {
                "ordinal": ordinal,
                "query_sha256": item.query_sha256,
                "feature_sha256": item.query_feature.feature_sha256,
                "plan_sha256": item.plan.plan_sha256,
                "raw_top5_sha256": stable_hash(list(item.raw_top5)),
                "hippo_top5_sha256": stable_hash(list(item.hippo_top5)),
                "trace_sha256s": [trace.trace_sha256 for trace in item.traces],
            }
        )
    return stable_hash(body)


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
    acquisition_adapter: HoverAcquisitionAdapter = DEFAULT_ACQUISITION_ADAPTER,
) -> StageExecution:
    """Execute the three retrieval arms from claim-only views."""

    items = acquisition_adapter.validate_block_view(
        view, block=block, formal_shape=formal_shape
    )
    queries = tuple(str(item["claim"]) for item in items)
    with ThreadPoolExecutor(max_workers=1) as threads:
        hippo_future = threads.submit(hippo.retrieve, block=block, queries=queries)
        features = compile_query_features_batched(
            queries=queries,
            index=prepared.embedding_index,
            encoder=encoder,
        )
        query_entities = _bounded_ner_extract(
            ner,
            tuple({"kind": "query", "query": query} for query in queries),
            batch_size=ner_batch_size,
        )
        plans: list[QueryPlan] = []
        for query, feature, spans in zip(
            queries, features, query_entities, strict=True
        ):
            plan = compile_query_plan(
                graph=prepared.graph,
                query=query,
                capability_similarity_ints={
                    capability: feature.capability_similarity_ints[offset]
                    for offset, capability in enumerate(CAPABILITIES)
                },
                query_entities=tuple(
                    sorted(
                        {
                            make_entity_key(span.entity_type, span.text)
                            for span in spans
                        }
                    )
                ),
            )
            if (
                plan.capability != feature.predicted_capability
                or plan.query_sha256 != feature.normalized_query_sha256
                or plan.normalized_sources
            ):
                raise HoVerFormalRunnerError(
                    "query plan consumed reserved missing metadata"
            )
            plans.append(plan)
        raw_rows = tuple(
            raw_top5(feature.dense_relevance_ints) for feature in features
        )
        traces = execute_agent_actions_eager(
            graph=prepared.graph,
            plans=plans,
            relevance_vectors=[feature.dense_relevance_ints for feature in features],
            local_worker_cap=local_worker_cap,
            executor_factory=executor_factory,
        )
        hippo_batch = hippo_future.result()
    hippo_rows = _validate_hippo_indices(
        hippo_batch, count=len(items), corpus_count=len(prepared.articles)
    )
    stage_items = tuple(
        StageItem(
            ordinal=ordinal,
            query_sha256=plan.query_sha256,
            query_feature=feature,
            plan=plan,
            raw_top5=raw_rows[ordinal],
            hippo_top5=hippo_rows[ordinal],
            traces=traces[ordinal],
        )
        for ordinal, (feature, plan) in enumerate(
            zip(features, plans, strict=True)
        )
    )
    matrix_sha = stage_execution_matrix_sha256(
        stage_items,
        expected_embedding_index_sha256=prepared.embedding_index.index_sha256,
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
        execution_matrix_sha256=matrix_sha,
        formal_shape=formal_shape,
    )


def _trace_for_policy(item: StageItem, policy: PolicySelection) -> ActionTrace:
    if policy.selection_sha256 != recompute_policy_selection_sha256(policy):
        raise HoVerFormalRunnerError("policy receipt drifted")
    traces = {trace.action_id: trace for trace in item.traces}
    if set(traces) != set(ACTION_IDS) or policy.action_id not in traces:
        raise HoVerFormalRunnerError("policy action is absent")
    return traces[policy.action_id]


def select_f_policies(
    *, f_stage: StageExecution
) -> tuple[PolicySelection, PolicySelection, bool]:
    if f_stage.block != "F_search":
        raise HoVerFormalRunnerError("policy selection requires F_search")
    observations = f_stage.observations()
    e0 = select_global_policy(evaluator_id=E0_ID, observations=observations)
    e1 = select_global_policy(evaluator_id=E1_ID, observations=observations)
    return e0, e1, policies_identifiable(e0, e1, observations)


def _utility(output: Sequence[int], gold: Sequence[int]) -> Fraction:
    values = tuple(gold)
    if (
        not 2 <= len(values) <= 4
        or len(set(values)) != len(values)
        or any(type(value) is not int or not 0 <= value < CORPUS_COUNT for value in values)
    ):
        raise HoVerFormalRunnerError("HoVer gold article set drifted")
    try:
        return item_utility(output, values)
    except Exception as exc:
        raise HoVerFormalRunnerError("HoVer utility input drifted") from exc


def _complete(output: Sequence[int], gold: Sequence[int]) -> bool:
    return set(gold) <= set(output)


def descriptive_stage_scores(
    *,
    stage: StageExecution,
    labels: Mapping[str, Any],
    acquisition_adapter: HoverAcquisitionAdapter = DEFAULT_ACQUISITION_ADAPTER,
) -> dict[str, Any]:
    """Return exact arm totals without selecting or changing a policy."""

    joined = acquisition_adapter.join_late_labels(stage, labels)
    arms: dict[str, list[Fraction]] = {"RAW": [], "HippoRAG": []}
    arms.update({action_id: [] for action_id in ACTION_IDS})
    strata: Counter[str] = Counter()
    for item, label in zip(stage.items, joined, strict=True):
        strata[label.hop_stratum] += 1
        arms["RAW"].append(_utility(item.raw_top5, label.gold_article_ids))
        arms["HippoRAG"].append(_utility(item.hippo_top5, label.gold_article_ids))
        for trace in item.traces:
            arms[trace.action_id].append(
                _utility(trace.output_top5, label.gold_article_ids)
            )
    return _self_hashed(
        {
            "schema": f"{VERSION}_descriptive_stage_scores",
            "version": VERSION,
            "block": stage.block,
            "item_count": len(stage.items),
            "exact_hop_stratum_counts": dict(sorted(strata.items())),
            "arm_utility_totals": {
                arm: fraction_payload(sum(values, Fraction(0)))
                for arm, values in sorted(arms.items())
            },
            "policy_or_threshold_changed": False,
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
    acquisition_adapter: HoverAcquisitionAdapter = DEFAULT_ACQUISITION_ADAPTER,
) -> AHoldAssessment:
    if stage.block != "A_hold" or f_stage.block != "F_search":
        raise HoVerFormalRunnerError("A_hold stage identity drifted")
    if not policies_identifiable(e0_policy, e1_policy, f_stage.observations()):
        raise HoVerFormalRunnerError("unidentifiable F policies cannot open A_hold")
    joined = acquisition_adapter.join_late_labels(stage, labels)
    e0_values: list[Fraction] = []
    e1_values: list[Fraction] = []
    hippo_values: list[Fraction] = []
    raw_values: list[Fraction] = []
    stratum_deltas = {stratum: Fraction(0) for stratum in HOP_STRATA}
    e0_complete = 0
    raw_complete = 0
    for item, label in zip(stage.items, joined, strict=True):
        e0_output = _trace_for_policy(item, e0_policy).output_top5
        e1_output = _trace_for_policy(item, e1_policy).output_top5
        e0_value = _utility(e0_output, label.gold_article_ids)
        e1_value = _utility(e1_output, label.gold_article_ids)
        hippo_value = _utility(item.hippo_top5, label.gold_article_ids)
        raw_value = _utility(item.raw_top5, label.gold_article_ids)
        e0_values.append(e0_value)
        e1_values.append(e1_value)
        hippo_values.append(hippo_value)
        raw_values.append(raw_value)
        stratum_deltas[label.hop_stratum] += e0_value - hippo_value
        e0_complete += int(_complete(e0_output, label.gold_article_ids))
        raw_complete += int(_complete(item.raw_top5, label.gold_article_ids))
    promotion = paired_utility_summary(e1_values, e0_values)
    primary = paired_utility_summary(e0_values, hippo_values)
    raw_boundary = paired_utility_summary(e0_values, raw_values)
    return AHoldAssessment(
        promoted=promotion.delta_total > 0
        and promotion.exact_one_sided_p <= Fraction(1, 10),
        primary_passed=primary.delta_total > 0
        and primary.exact_one_sided_p <= Fraction(1, 10)
        and all(value > 0 for value in stratum_deltas.values()),
        e0_policy=e0_policy,
        e1_policy=e1_policy,
        promotion_delta_total=promotion.delta_total,
        promotion_signflip_p=promotion.exact_one_sided_p,
        e0_minus_hippo_delta_total=primary.delta_total,
        e0_minus_hippo_signflip_p=primary.exact_one_sided_p,
        e0_minus_hippo_stratum_deltas=tuple(
            (stratum, stratum_deltas[stratum]) for stratum in HOP_STRATA
        ),
        e0_minus_raw_delta_total=raw_boundary.delta_total,
        e0_minus_raw_signflip_p=raw_boundary.exact_one_sided_p,
        e0_complete_count=e0_complete,
        raw_complete_count=raw_complete,
    )


def assess_m_search(
    *,
    stage: StageExecution,
    labels: Mapping[str, Any],
    f_stage: StageExecution,
    e0_policy: PolicySelection,
    e1_policy: PolicySelection,
    acquisition_adapter: HoverAcquisitionAdapter = DEFAULT_ACQUISITION_ADAPTER,
) -> MSearchAssessment:
    if stage.block != "M_search" or f_stage.block != "F_search":
        raise HoVerFormalRunnerError("M_search stage identity drifted")
    if not policies_identifiable(e0_policy, e1_policy, f_stage.observations()):
        raise HoVerFormalRunnerError("M policies differ from frozen F")
    joined = acquisition_adapter.join_late_labels(stage, labels)
    e0_values: list[Fraction] = []
    e1_values: list[Fraction] = []
    hippo_values: list[Fraction] = []
    raw_values: list[Fraction] = []
    stratum_deltas = {stratum: Fraction(0) for stratum in HOP_STRATA}
    e1_complete = 0
    raw_complete = 0
    for item, label in zip(stage.items, joined, strict=True):
        e0_value = _utility(
            _trace_for_policy(item, e0_policy).output_top5,
            label.gold_article_ids,
        )
        e1_output = _trace_for_policy(item, e1_policy).output_top5
        e1_value = _utility(e1_output, label.gold_article_ids)
        hippo_value = _utility(item.hippo_top5, label.gold_article_ids)
        raw_value = _utility(item.raw_top5, label.gold_article_ids)
        e0_values.append(e0_value)
        e1_values.append(e1_value)
        hippo_values.append(hippo_value)
        raw_values.append(raw_value)
        stratum_deltas[label.hop_stratum] += e1_value - hippo_value
        e1_complete += int(_complete(e1_output, label.gold_article_ids))
        raw_complete += int(_complete(item.raw_top5, label.gold_article_ids))
    l5 = paired_utility_summary(e1_values, e0_values)
    hippo_summary = paired_utility_summary(e1_values, hippo_values)
    raw_summary = paired_utility_summary(e1_values, raw_values)
    return MSearchAssessment(
        l5_delta_total=l5.delta_total,
        l5_signflip_p=l5.exact_one_sided_p,
        l5_passed=l5.delta_total > 0
        and l5.exact_one_sided_p <= Fraction(1, 10),
        e1_minus_hippo_delta_total=hippo_summary.delta_total,
        e1_minus_hippo_signflip_p=hippo_summary.exact_one_sided_p,
        e1_minus_hippo_stratum_deltas=tuple(
            (stratum, stratum_deltas[stratum]) for stratum in HOP_STRATA
        ),
        e1_minus_raw_delta_total=raw_summary.delta_total,
        e1_minus_raw_signflip_p=raw_summary.exact_one_sided_p,
        e1_complete_count=e1_complete,
        raw_complete_count=raw_complete,
    )


__all__ = [
    "ACQUISITION_VERSION",
    "ACTION_IDS",
    "AHoldAssessment",
    "BLOCK_COUNTS",
    "BLOCK_ORDER",
    "CORPUS_COUNT",
    "DEFAULT_ACQUISITION_ADAPTER",
    "HOP_QUOTAS",
    "HOP_STRATA",
    "HoVerFormalRunnerError",
    "HoverAcquisitionAdapter",
    "LateLabel",
    "LOCAL_CONCURRENCY_CAP",
    "MSearchAssessment",
    "PreparedCorpus",
    "StageExecution",
    "StageItem",
    "assess_m_search",
    "decide_a_hold_promotion",
    "descriptive_stage_scores",
    "execute_agent_actions_eager",
    "execute_gold_free_stage",
    "fraction_payload",
    "prepare_offline_corpus",
    "select_f_policies",
    "stable_hash",
    "stage_execution_matrix_sha256",
    "verify_self_hash",
]
