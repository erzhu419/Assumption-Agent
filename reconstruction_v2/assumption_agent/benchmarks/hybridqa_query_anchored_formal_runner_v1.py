"""Offline execution and evaluator core for the frozen HybridQA P6/E2 study.

The module keeps source acquisition, official HippoRAG, action formation and
late scoring as separate capabilities.  Its pure functions are intentionally
usable with synthetic graphs and injected encoders; a separate formal
controller can bind private packs and local runtimes.  No online evaluator or
model API exists here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
import hashlib
import hmac
from itertools import combinations
import json
import math
import re
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from assumption_agent.benchmarks import (
    feverous_e2_evaluator_v1 as evaluator_math,
)
from assumption_agent.benchmarks import (
    hybridqa_query_anchored_operator_v1 as operator,
)
from replication_runtime.multihoprag_minilm_v1 import adapter as minilm_adapter
from replication_runtime.qasper_minilm_v1.binding import (
    QUANTIZATION_SCALE,
    quantized_cosine_similarity,
)


VERSION = "hybridqa_query_anchored_formal_runner_v1"
BLOCK_ORDER = ("A_form", "F_search", "A_hold", "M_search")
BLOCK_COUNTS = {"A_form": 48, "F_search": 36, "A_hold": 30, "M_search": 30}
FAMILIES = ("PASSAGE_ONLY", "TABLE_ONLY", "DUAL_TABLE_PASSAGE")
BLOCK_FAMILY_COUNTS = {
    "A_hold": {family: 10 for family in FAMILIES},
    "M_search": {family: 10 for family in FAMILIES},
}
RECIPE_IDS = (
    "R0_DENSE5",
    "R1_P6_DIRECT_B2",
    "R2_P6_PATH1_B2",
    "R3_P6_PATH2_B2",
)
FEATURE_ORDER = (
    "direct_facet_coverage",
    "residual_facet_coverage",
    "deletion_mean_coverage_drop",
    "deletion_minimum_coverage_drop",
    "same_type_replacement_mean_coverage_drop",
    "query_anchored_path_coverage",
    "dense_relevance_mass",
    "negative_pairwise_redundancy",
)
TOP_K = operator.TOP_K
CORPUS_UNIT_COUNT = operator.CORPUS_UNIT_COUNT
DIRECT_ANCHOR_K = 8
PROMOTION_ALPHA = Fraction(1, 10)
ENTITY_FACET_LIMIT = 4
NUMERIC_FACET_LIMIT = 2
RELATION_FACET_LIMIT = 1
FACET_LIMIT = ENTITY_FACET_LIMIT + NUMERIC_FACET_LIMIT + RELATION_FACET_LIMIT
FACET_EMBEDDING_NORM_ATOL = 2e-5
BULK_SEMANTIC_BATCH_LIMIT = 16_384

_CONTENT_TAG_PREFIXES = ("NN", "JJ")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class HybridQaFormalRunnerError(RuntimeError):
    """A frozen runtime, pack, action, evaluator or stage contract drifted."""


def _verify_inherited_evaluator_contract() -> None:
    """Fail at import if the private evaluator mathematics drifts underneath us."""

    expected = {
        "RECIPE_IDS": RECIPE_IDS,
        "FEATURE_ORDER": FEATURE_ORDER,
        "FOLD_COUNT": 4,
        "RIDGE_LAMBDA": Decimal(1),
        "PAIR_WEIGHT": Fraction(1, 6),
        "DECIMAL_PRECISION": 80,
        "PROMOTION_ALPHA": PROMOTION_ALPHA,
    }
    observed = {
        "RECIPE_IDS": evaluator_math.RECIPE_IDS,
        "FEATURE_ORDER": evaluator_math.FEATURE_ORDER,
        "FOLD_COUNT": evaluator_math.FOLD_COUNT,
        "RIDGE_LAMBDA": evaluator_math.RIDGE_LAMBDA,
        "PAIR_WEIGHT": evaluator_math.PAIR_WEIGHT,
        "DECIMAL_PRECISION": evaluator_math.DECIMAL_PRECISION,
        "PROMOTION_ALPHA": evaluator_math.PROMOTION_ALPHA,
    }
    if observed != expected or operator.RECIPE_IDS != RECIPE_IDS:
        raise HybridQaFormalRunnerError("inherited evaluator mathematics drifted")


_verify_inherited_evaluator_contract()


class Encoder(Protocol):
    runtime_receipt: Mapping[str, object]
    canary_receipt: Mapping[str, object]

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


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
        raise HybridQaFormalRunnerError("value is not canonical JSON") from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in body:
        raise HybridQaFormalRunnerError("self-hash field already exists")
    return {**dict(body), field: stable_hash(body)}


def _verify_self_hashed(
    receipt: Mapping[str, Any], *, schema: str, field: str
) -> str:
    if not isinstance(receipt, Mapping):
        raise HybridQaFormalRunnerError(f"{schema} is not a mapping")
    body = dict(receipt)
    declared = body.pop(field, None)
    if (
        not isinstance(declared, str)
        or _HEX64.fullmatch(declared) is None
        or body.get("schema") != schema
        or stable_hash(body) != declared
    ):
        raise HybridQaFormalRunnerError(f"{schema} self-hash drifted")
    return declared


def _canonical_json_text(value: object) -> str:
    return _canonical_bytes(value).decode("ascii")


def _mapping_from_canonical_json(value: str, *, field: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise HybridQaFormalRunnerError(f"{field} is not canonical JSON text")
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, TypeError) as exc:
        raise HybridQaFormalRunnerError(f"{field} is not valid JSON") from exc
    if not isinstance(decoded, dict) or _canonical_json_text(decoded) != value:
        raise HybridQaFormalRunnerError(f"{field} is not canonical JSON text")
    return decoded


def _fraction_payload(value: Fraction) -> list[int]:
    if not isinstance(value, Fraction):
        raise HybridQaFormalRunnerError("value is not an exact Fraction")
    return [value.numerator, value.denominator]


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _normalized_text(value: str, *, field: str) -> str:
    try:
        return minilm_adapter.canonical_text(value, field=field)
    except Exception as exc:
        raise HybridQaFormalRunnerError(f"{field} is invalid") from exc


def _maximal_tag_spans(
    tokens: Sequence[str],
    tags: Sequence[str],
    *,
    predicate: Any,
) -> tuple[str, ...]:
    spans: list[str] = []
    start: int | None = None
    for index in range(len(tokens) + 1):
        accepted = index < len(tokens) and bool(predicate(tags[index]))
        if accepted and start is None:
            start = index
        elif not accepted and start is not None:
            value = operator.normalize_key(" ".join(tokens[start:index]))
            if value:
                spans.append(value)
            start = None
    return tuple(spans)


def extract_query_facets(question: str, question_postag: str) -> tuple[operator.QueryFacet, ...]:
    """Apply the frozen aligned-POS facet grammar without labels or IDs."""

    normalized_question = _normalized_text(question, field="question")
    if not isinstance(question_postag, str) or "\x00" in question_postag:
        raise HybridQaFormalRunnerError("question POS tags are invalid")
    tokens = normalized_question.split()
    tags = question_postag.split()
    if not tokens or len(tokens) != len(tags) or any(not tag for tag in tags):
        raise HybridQaFormalRunnerError("question/POS alignment drifted")
    content_candidates = _maximal_tag_spans(
        tokens,
        tags,
        predicate=lambda tag: tag.startswith(_CONTENT_TAG_PREFIXES) or tag == "CD",
    )
    numeric_candidates = _maximal_tag_spans(
        tokens,
        tags,
        predicate=lambda tag: tag == "CD",
    )
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()

    # Apply each type quota *after* exact-text deduplication.  In particular,
    # duplicate early CD spans may not consume both frozen numeric slots and
    # suppress a later distinct numeric/date facet.
    content = tuple(dict.fromkeys(content_candidates))[:ENTITY_FACET_LIMIT]
    for value in content:
        rows.append(("entity", value))
        seen.add(value)
    numeric = tuple(
        value
        for value in dict.fromkeys(numeric_candidates)
        if value not in seen
    )[:NUMERIC_FACET_LIMIT]
    for value in numeric:
        rows.append(("numeric_or_date", value))
        seen.add(value)
    relation = operator.normalize_key(normalized_question)
    if relation not in seen:
        rows.append(("relation_clause", relation))
        seen.add(relation)
    if not rows:
        rows.append(("relation_clause", relation))
    counts = Counter(facet_type for facet_type, _value in rows)
    if (
        len(rows) > FACET_LIMIT
        or counts["entity"] > ENTITY_FACET_LIMIT
        or counts["numeric_or_date"] > NUMERIC_FACET_LIMIT
        or counts["relation_clause"] > RELATION_FACET_LIMIT
    ):
        raise HybridQaFormalRunnerError("query facet quota drifted")
    try:
        return tuple(
            operator.make_query_facet(index, facet_type, text)
            for index, (facet_type, text) in enumerate(rows)
        )
    except operator.HybridQaOperatorError as exc:
        raise HybridQaFormalRunnerError("query facet formation failed") from exc


def _facet_coverage_rows(
    *,
    facets: Sequence[operator.QueryFacet],
    index: minilm_adapter.CorpusEmbeddingIndex,
    encoder: Encoder,
) -> tuple[tuple[int, ...], ...]:
    validated = minilm_adapter.validate_corpus_embedding_index(index)
    try:
        matrix = encoder.encode(tuple(facet.normalized_text for facet in facets))
    except Exception as exc:
        raise HybridQaFormalRunnerError("facet encoding failed") from exc
    if (
        not isinstance(matrix, np.ndarray)
        or matrix.dtype != np.float32
        or matrix.shape != (len(facets), index.chunk_vectors.shape[1])
        or not np.isfinite(matrix).all()
    ):
        raise HybridQaFormalRunnerError("facet encoder output drifted")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=FACET_EMBEDDING_NORM_ATOL):
        raise HybridQaFormalRunnerError("facet encoder output is not L2 normalized")
    output: list[tuple[int, ...]] = []
    for facet_i in range(len(facets)):
        row: list[int] = []
        for start, stop in validated.article_chunk_ranges:
            row.append(
                max(
                    quantized_cosine_similarity(matrix[facet_i], validated.chunk_vectors[i])
                    for i in range(start, stop)
                )
            )
        output.append(tuple(row))
    return tuple(output)


def build_query_semantic_tensor(
    *,
    question: str,
    question_postag: str,
    index: minilm_adapter.CorpusEmbeddingIndex,
    encoder: Encoder,
) -> operator.QuerySemanticTensor:
    """Build the complete facet-by-609 tensor and frozen top-eight anchors."""

    if index.article_count != CORPUS_UNIT_COUNT:
        raise HybridQaFormalRunnerError("MiniLM corpus is not exactly 609 units")
    facets = extract_query_facets(question, question_postag)
    coverage = _facet_coverage_rows(facets=facets, index=index, encoder=encoder)
    try:
        query_features = minilm_adapter.compile_query_features(
            query=question,
            index=index,
            encoder=encoder,
        )
    except Exception as exc:
        raise HybridQaFormalRunnerError("dense query features failed") from exc
    anchors: list[tuple[int, ...]] = []
    for row in coverage:
        selected = {
            ordinal
            for ordinal in sorted(
                (i for i, score in enumerate(row) if score > 0),
                key=lambda i: (-row[i], i),
            )[:DIRECT_ANCHOR_K]
        }
        anchors.append(
            tuple(row[i] if i in selected else 0 for i in range(CORPUS_UNIT_COUNT))
        )
    try:
        return operator.make_query_semantic_tensor(
            query_sha256=query_features.normalized_query_sha256,
            facets=facets,
            semantic_coverage_ints=coverage,
            direct_anchor_strength_ints=anchors,
            dense_relevance_ints=query_features.dense_relevance_ints,
        )
    except operator.HybridQaOperatorError as exc:
        raise HybridQaFormalRunnerError("semantic tensor formation failed") from exc


@dataclass(frozen=True)
class BulkQueryInput:
    """Label-free semantic input bound only to an opaque item commitment."""

    item_commitment_sha256: str
    question: str
    question_postag: str

    def __post_init__(self) -> None:
        if not _is_sha256(self.item_commitment_sha256):
            raise HybridQaFormalRunnerError("bulk query commitment is invalid")
        _normalized_text(self.question, field="question")
        if not isinstance(self.question_postag, str) or "\x00" in self.question_postag:
            raise HybridQaFormalRunnerError("bulk query POS tags are invalid")


def build_query_semantic_tensors_bulk(
    *,
    rows: Sequence[BulkQueryInput],
    index: minilm_adapter.CorpusEmbeddingIndex,
    encoder: Encoder,
) -> dict[str, operator.QuerySemanticTensor]:
    """Build many tensors with one frozen, deterministic MiniLM encode call.

    The batch schedule is commitment-sorted facets, then commitment-sorted
    questions, then the three frozen capability prototypes.  No label, gold
    unit, family, recipe output, RAW output, or HippoRAG output is accepted.
    """

    if not rows or any(not isinstance(row, BulkQueryInput) for row in rows):
        raise HybridQaFormalRunnerError("bulk semantic input rows are invalid")
    canonical_rows = tuple(sorted(rows, key=lambda row: row.item_commitment_sha256))
    if len({row.item_commitment_sha256 for row in canonical_rows}) != len(
        canonical_rows
    ):
        raise HybridQaFormalRunnerError("bulk query commitment duplicated")
    try:
        validated = minilm_adapter.validate_corpus_embedding_index(index)
        encoder_sha = minilm_adapter.encoder_receipt_sha256(encoder)
    except Exception as exc:
        raise HybridQaFormalRunnerError("bulk MiniLM dependency validation failed") from exc
    if (
        validated.article_count != CORPUS_UNIT_COUNT
        or encoder_sha != validated.encoder_receipt_sha256
    ):
        raise HybridQaFormalRunnerError("bulk MiniLM corpus/runtime binding drifted")

    facets_by_item: dict[str, tuple[operator.QueryFacet, ...]] = {}
    questions_by_item: dict[str, str] = {}
    facet_slices: dict[str, tuple[int, int]] = {}
    texts: list[str] = []
    for row in canonical_rows:
        facets = extract_query_facets(row.question, row.question_postag)
        facets_by_item[row.item_commitment_sha256] = facets
        questions_by_item[row.item_commitment_sha256] = _normalized_text(
            row.question, field="question"
        )
        start = len(texts)
        texts.extend(facet.normalized_text for facet in facets)
        facet_slices[row.item_commitment_sha256] = (start, len(texts))
    question_offsets: dict[str, int] = {}
    for row in canonical_rows:
        question_offsets[row.item_commitment_sha256] = len(texts)
        texts.append(questions_by_item[row.item_commitment_sha256])
    prototype_start = len(texts)
    texts.extend(
        minilm_adapter.CAPABILITY_PROTOTYPES[name]
        for name in minilm_adapter.CAPABILITY_ORDER
    )
    if len(texts) > BULK_SEMANTIC_BATCH_LIMIT:
        raise HybridQaFormalRunnerError("bulk semantic batch exceeds 16384 texts")
    try:
        matrix = encoder.encode(tuple(texts))
    except Exception as exc:
        raise HybridQaFormalRunnerError("bulk semantic encoding failed") from exc
    if (
        not isinstance(matrix, np.ndarray)
        or matrix.dtype != np.float32
        or matrix.shape != (len(texts), validated.chunk_vectors.shape[1])
        or not np.isfinite(matrix).all()
    ):
        raise HybridQaFormalRunnerError("bulk semantic encoder output drifted")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=FACET_EMBEDDING_NORM_ATOL):
        raise HybridQaFormalRunnerError("bulk semantic embeddings are not L2 normalized")

    prototype_vectors = matrix[prototype_start : prototype_start + 3]
    output: dict[str, operator.QuerySemanticTensor] = {}
    for row in canonical_rows:
        commitment = row.item_commitment_sha256
        facets = facets_by_item[commitment]
        start, stop = facet_slices[commitment]
        facet_vectors = matrix[start:stop]
        coverage: list[tuple[int, ...]] = []
        for facet_vector in facet_vectors:
            coverage.append(
                tuple(
                    max(
                        quantized_cosine_similarity(
                            facet_vector, validated.chunk_vectors[chunk_i]
                        )
                        for chunk_i in range(chunk_start, chunk_stop)
                    )
                    for chunk_start, chunk_stop in validated.article_chunk_ranges
                )
            )
        question_vector = matrix[question_offsets[commitment]]
        # Compute the frozen capability similarities even though only dense
        # relevance enters this study's tensor.  This retains exact parity with
        # the single-query compiler's fixed semantic schedule.
        _capability_scores = tuple(
            quantized_cosine_similarity(question_vector, prototype_vector)
            for prototype_vector in prototype_vectors
        )
        dense = tuple(
            max(
                quantized_cosine_similarity(
                    question_vector, validated.chunk_vectors[chunk_i]
                )
                for chunk_i in range(chunk_start, chunk_stop)
            )
            for chunk_start, chunk_stop in validated.article_chunk_ranges
        )
        anchors: list[tuple[int, ...]] = []
        for coverage_row in coverage:
            selected = {
                ordinal
                for ordinal in sorted(
                    (
                        index_i
                        for index_i, score in enumerate(coverage_row)
                        if score > 0
                    ),
                    key=lambda index_i: (-coverage_row[index_i], index_i),
                )[:DIRECT_ANCHOR_K]
            }
            anchors.append(
                tuple(
                    coverage_row[index_i] if index_i in selected else 0
                    for index_i in range(CORPUS_UNIT_COUNT)
                )
            )
        query_sha = hashlib.sha256(
            questions_by_item[commitment].casefold().encode("utf-8")
        ).hexdigest()
        try:
            output[commitment] = operator.make_query_semantic_tensor(
                query_sha256=query_sha,
                facets=facets,
                semantic_coverage_ints=coverage,
                direct_anchor_strength_ints=anchors,
                dense_relevance_ints=dense,
            )
        except operator.HybridQaOperatorError as exc:
            raise HybridQaFormalRunnerError(
                "bulk semantic tensor formation failed"
            ) from exc
    return output


def _coverage(maxima: Sequence[int]) -> Fraction:
    rows = tuple(maxima)
    if not rows or any(type(value) is not int for value in rows):
        raise HybridQaFormalRunnerError("facet maxima are invalid")
    return Fraction(sum(rows), len(rows) * QUANTIZATION_SCALE)


def _reachable_within_two(
    graph: operator.TypedCorpusGraph,
    tensor: operator.QuerySemanticTensor,
) -> frozenset[int]:
    direct = {
        ordinal
        for ordinal in range(CORPUS_UNIT_COUNT)
        if any(row.direct_anchor_strength_ints[ordinal] > 0 for row in tensor.rows)
    }
    reached = set(direct)
    frontier = set(direct)
    for _depth in range(2):
        following = {
            neighbor.neighbor_ordinal
            for ordinal in frontier
            for neighbor in graph.neighbors[ordinal]
        }
        following.difference_update(reached)
        reached.update(following)
        frontier = following
    return frozenset(reached)


def exact_action_features(
    *,
    graph: operator.TypedCorpusGraph,
    tensor: operator.QuerySemanticTensor,
    trace: operator.ActionTrace,
) -> dict[str, Fraction]:
    """Compute all eight frozen features with complete 609-unit interventions."""

    try:
        operator.verify_typed_graph(graph)
        operator.verify_query_semantic_tensor(tensor)
        operator.verify_action_trace(trace)
    except operator.HybridQaOperatorError as exc:
        raise HybridQaFormalRunnerError("feature input verification failed") from exc
    if (
        trace.graph_sha256 != graph.graph_sha256
        or trace.semantic_tensor_sha256 != tensor.tensor_sha256
    ):
        raise HybridQaFormalRunnerError("action trace is not bound to graph/tensor")
    selected = trace.output_top5

    # Inputs were exhaustively verified above.  Keep the intervention loop a
    # literal all-609 scan without rehashing the same graph/tensor thousands
    # of times per item.
    def maxima(ordinals: Sequence[int]) -> tuple[int, ...]:
        return tuple(
            max(row.semantic_coverage_ints[ordinal] for ordinal in ordinals)
            for row in tensor.rows
        )

    full_maxima = maxima(selected)
    retained_maxima = maxima(trace.retained_raw_top3)
    full = _coverage(full_maxima)
    residual = Fraction(
        sum(max(0, a - b) for a, b in zip(full_maxima, retained_maxima)),
        len(full_maxima) * QUANTIZATION_SCALE,
    )
    deletion_drops = tuple(
        full - _coverage(maxima(selected[:slot] + selected[slot + 1 :]))
        for slot in range(TOP_K)
    )
    replacement_drops: list[Fraction] = []
    replacement_candidate_scan_count = 0
    selected_set = set(selected)
    for slot in range(TOP_K):
        removed_type = graph.units[selected[slot]].unit_type
        candidates: list[int] = []
        # This is deliberately an explicit complete-corpus scan for every
        # intervention slot.  The counter makes a future shortcut detectable.
        for ordinal in range(CORPUS_UNIT_COUNT):
            replacement_candidate_scan_count += 1
            if (
                ordinal not in selected_set
                and graph.units[ordinal].unit_type == removed_type
            ):
                candidates.append(ordinal)
        if not candidates:
            replacement_drops.append(Fraction(0))
            continue
        best: Fraction | None = None
        for candidate in candidates:
            action = (*selected[:slot], candidate, *selected[slot + 1 :])
            value = _coverage(maxima(action))
            if best is None or value > best:
                best = value
        assert best is not None
        replacement_drops.append(full - best)
    if replacement_candidate_scan_count != TOP_K * CORPUS_UNIT_COUNT:
        raise HybridQaFormalRunnerError("replacement intervention scan drifted")
    reachable = _reachable_within_two(graph, tensor)
    redundancy_sum = 0
    for left, right in combinations(selected, 2):
        for row in tensor.rows:
            redundancy_sum += min(
                max(0, row.semantic_coverage_ints[left]),
                max(0, row.semantic_coverage_ints[right]),
            )
    values = {
        "direct_facet_coverage": full,
        "residual_facet_coverage": residual,
        "deletion_mean_coverage_drop": sum(deletion_drops, Fraction(0)) / TOP_K,
        "deletion_minimum_coverage_drop": min(deletion_drops),
        "same_type_replacement_mean_coverage_drop": (
            sum(replacement_drops, Fraction(0)) / TOP_K
        ),
        "query_anchored_path_coverage": Fraction(
            sum(ordinal in reachable for ordinal in selected), TOP_K
        ),
        "dense_relevance_mass": Fraction(
            sum(tensor.dense_relevance_ints[ordinal] for ordinal in selected),
            QUANTIZATION_SCALE,
        ),
        "negative_pairwise_redundancy": -Fraction(
            redundancy_sum,
            math.comb(TOP_K, 2) * len(tensor.rows) * QUANTIZATION_SCALE,
        ),
    }
    if tuple(values) != FEATURE_ORDER:
        raise HybridQaFormalRunnerError("feature order drifted")
    return values


def recipe_trace_from_action(
    *,
    item_commitment_sha256: str,
    graph: operator.TypedCorpusGraph,
    tensor: operator.QuerySemanticTensor,
    trace: operator.ActionTrace,
) -> evaluator_math.RecipeTrace:
    if (
        not _is_sha256(item_commitment_sha256)
    ):
        raise HybridQaFormalRunnerError("item commitment is invalid")
    features = exact_action_features(graph=graph, tensor=tensor, trace=trace)
    behavior = stable_hash(
        {
            "graph_sha256": graph.graph_sha256,
            "ordered_top5": list(trace.output_top5),
            "query_sha256": tensor.query_sha256,
            "semantic_tensor_sha256": tensor.tensor_sha256,
            "version": VERSION,
        }
    )
    try:
        return evaluator_math.RecipeTrace.from_mapping(
            item_commitment_sha256=item_commitment_sha256,
            recipe_id=trace.recipe_id,
            behavior_sha256=behavior,
            features=features,
        )
    except evaluator_math.FeverousEvaluatorError as exc:
        raise HybridQaFormalRunnerError("evaluator trace formation failed") from exc


@dataclass(frozen=True)
class ItemExecution:
    item_commitment_sha256: str
    action_traces: tuple[operator.ActionTrace, ...]
    recipe_traces: tuple[evaluator_math.RecipeTrace, ...]

    def __post_init__(self) -> None:
        if not _is_sha256(self.item_commitment_sha256):
            raise HybridQaFormalRunnerError("item execution commitment is invalid")
        if (
            not isinstance(self.action_traces, tuple)
            or not isinstance(self.recipe_traces, tuple)
            or any(
                not isinstance(trace, operator.ActionTrace)
                for trace in self.action_traces
            )
            or any(
                not isinstance(trace, evaluator_math.RecipeTrace)
                for trace in self.recipe_traces
            )
            or tuple(trace.recipe_id for trace in self.action_traces) != RECIPE_IDS
            or tuple(trace.recipe_id for trace in self.recipe_traces) != RECIPE_IDS
        ):
            raise HybridQaFormalRunnerError("item execution recipe matrix drifted")
        for action, recipe_trace in zip(
            self.action_traces, self.recipe_traces, strict=True
        ):
            try:
                operator.verify_action_trace(action)
            except operator.HybridQaOperatorError as exc:
                raise HybridQaFormalRunnerError(
                    "item execution contains an invalid action trace"
                ) from exc
            expected_behavior = stable_hash(
                {
                    "graph_sha256": action.graph_sha256,
                    "ordered_top5": list(action.output_top5),
                    "query_sha256": action.query_sha256,
                    "semantic_tensor_sha256": action.semantic_tensor_sha256,
                    "version": VERSION,
                }
            )
            if (
                recipe_trace.item_commitment_sha256
                != self.item_commitment_sha256
                or recipe_trace.recipe_id != action.recipe_id
                or recipe_trace.behavior_sha256 != expected_behavior
            ):
                raise HybridQaFormalRunnerError(
                    "item execution action/feature binding drifted"
                )

    @property
    def outputs(self) -> Mapping[str, tuple[int, ...]]:
        return {trace.recipe_id: tuple(trace.output_top5) for trace in self.action_traces}


@dataclass(frozen=True)
class AnchorLabel:
    """Late-opened anchor label bound to an opaque item commitment."""

    item_commitment_sha256: str
    gold_ordinals: tuple[int, ...]
    family: str

    def __post_init__(self) -> None:
        if not _is_sha256(self.item_commitment_sha256):
            raise HybridQaFormalRunnerError("anchor label commitment is invalid")
        if (
            not isinstance(self.gold_ordinals, tuple)
            or not 1 <= len(self.gold_ordinals) <= 3
            or len(set(self.gold_ordinals)) != len(self.gold_ordinals)
            or any(
                type(value) is not int or not 0 <= value < CORPUS_UNIT_COUNT
                for value in self.gold_ordinals
            )
        ):
            raise HybridQaFormalRunnerError("anchor gold ordinals are invalid")
        if self.family not in FAMILIES:
            raise HybridQaFormalRunnerError("anchor label family drifted")


@dataclass(frozen=True)
class HippoRetrieval:
    """One official HippoRAG top-five bound to an opaque item commitment."""

    item_commitment_sha256: str
    top5: tuple[int, ...]

    def __post_init__(self) -> None:
        if not _is_sha256(self.item_commitment_sha256):
            raise HybridQaFormalRunnerError("HippoRAG commitment is invalid")
        if (
            not isinstance(self.top5, tuple)
            or len(self.top5) != TOP_K
            or len(set(self.top5)) != TOP_K
            or any(
                type(value) is not int or not 0 <= value < CORPUS_UNIT_COUNT
                for value in self.top5
            )
        ):
            raise HybridQaFormalRunnerError("HippoRAG result is not an exact top five")

    def payload(self) -> list[object]:
        return [self.item_commitment_sha256, list(self.top5)]


def execute_item(
    *,
    item_commitment_sha256: str,
    graph: operator.TypedCorpusGraph,
    tensor: operator.QuerySemanticTensor,
) -> ItemExecution:
    try:
        actions = operator.run_all_recipes(graph=graph, semantic_tensor=tensor)
    except operator.HybridQaOperatorError as exc:
        raise HybridQaFormalRunnerError("operator execution failed") from exc
    traces = tuple(
        recipe_trace_from_action(
            item_commitment_sha256=item_commitment_sha256,
            graph=graph,
            tensor=tensor,
            trace=trace,
        )
        for trace in actions
    )
    return ItemExecution(item_commitment_sha256, actions, traces)


def item_utility(selected: Sequence[int], gold: Sequence[int]) -> tuple[Fraction, bool]:
    output = tuple(selected)
    target = tuple(gold)
    if (
        len(output) != TOP_K
        or len(set(output)) != TOP_K
        or not 1 <= len(target) <= 3
        or len(set(target)) != len(target)
        or any(
            type(value) is not int or not 0 <= value < CORPUS_UNIT_COUNT
            for value in (*output, *target)
        )
    ):
        raise HybridQaFormalRunnerError("utility input is invalid")
    overlap = len(set(output).intersection(target))
    complete = overlap == len(target)
    return Fraction(overlap, len(target)) + int(complete), complete


def build_feature_receipt(
    *, block: str, traces: Sequence[evaluator_math.RecipeTrace]
) -> dict[str, Any]:
    if block not in BLOCK_COUNTS:
        raise HybridQaFormalRunnerError("feature block is invalid")
    try:
        matrix = evaluator_math._normalize_matrix(traces)
    except evaluator_math.FeverousEvaluatorError as exc:
        raise HybridQaFormalRunnerError("trace matrix is invalid") from exc
    if len(matrix) != BLOCK_COUNTS[block]:
        raise HybridQaFormalRunnerError("feature block item count drifted")
    canonical_traces = tuple(trace for _, rows in matrix for trace in rows)
    commitments = tuple(item for item, _rows in matrix)
    body = {
        "schema": f"{VERSION}_feature_receipt",
        "version": VERSION,
        "block": block,
        "item_count": len(matrix),
        "trace_count": len(canonical_traces),
        "recipe_registry": list(RECIPE_IDS),
        "fixed_feature_order": list(FEATURE_ORDER),
        "inherited_evaluator_contract": {
            "fold_count": 4,
            "ridge_lambda": "1",
            "pair_weight": [1, 6],
            "decimal_precision": 80,
            "promotion_alpha": [1, 10],
        },
        "facet_contract": {
            "entity_limit": ENTITY_FACET_LIMIT,
            "numeric_or_date_limit": NUMERIC_FACET_LIMIT,
            "relation_clause_limit": RELATION_FACET_LIMIT,
            "total_limit": FACET_LIMIT,
            "embedding_L2_norm_absolute_tolerance": "0.00002",
        },
        "same_type_replacement_intervention": {
            "corpus_units_scanned_per_output_slot": CORPUS_UNIT_COUNT,
            "output_slot_count": TOP_K,
            "total_corpus_unit_visits_per_action": TOP_K * CORPUS_UNIT_COUNT,
        },
        "trace_matrix_sha256": stable_hash(
            [trace.payload() for trace in canonical_traces]
        ),
        "item_commitment_set_sha256": stable_hash(list(commitments)),
        "labels_or_utility_accessed": False,
        "raw_content_persisted": False,
    }
    return _self_hashed(body, "feature_receipt_sha256")


@dataclass(frozen=True)
class FeatureSeal:
    """Immutable proof that one complete block feature matrix was sealed."""

    block: str
    traces: tuple[evaluator_math.RecipeTrace, ...]
    feature_receipt_sha256: str
    trace_matrix_sha256: str
    item_commitment_set_sha256: str

    def __post_init__(self) -> None:
        if self.block not in BLOCK_COUNTS or not isinstance(self.traces, tuple):
            raise HybridQaFormalRunnerError("feature seal block/traces are invalid")
        try:
            matrix = evaluator_math._normalize_matrix(self.traces)
        except evaluator_math.FeverousEvaluatorError as exc:
            raise HybridQaFormalRunnerError(
                "feature seal trace matrix is invalid"
            ) from exc
        canonical = tuple(trace for _item, rows in matrix for trace in rows)
        if canonical != self.traces:
            raise HybridQaFormalRunnerError("feature seal traces are not canonical")
        receipt = build_feature_receipt(block=self.block, traces=self.traces)
        expected = (
            receipt["feature_receipt_sha256"],
            receipt["trace_matrix_sha256"],
            receipt["item_commitment_set_sha256"],
        )
        observed = (
            self.feature_receipt_sha256,
            self.trace_matrix_sha256,
            self.item_commitment_set_sha256,
        )
        if observed != expected:
            raise HybridQaFormalRunnerError("feature seal binding drifted")

    @property
    def receipt(self) -> dict[str, Any]:
        return build_feature_receipt(block=self.block, traces=self.traces)

    @property
    def item_commitments(self) -> tuple[str, ...]:
        return tuple(
            sorted({trace.item_commitment_sha256 for trace in self.traces})
        )


def seal_feature_matrix(
    *, block: str, traces: Sequence[evaluator_math.RecipeTrace]
) -> FeatureSeal:
    try:
        matrix = evaluator_math._normalize_matrix(traces)
    except evaluator_math.FeverousEvaluatorError as exc:
        raise HybridQaFormalRunnerError("trace matrix is invalid") from exc
    canonical = tuple(trace for _item, rows in matrix for trace in rows)
    receipt = build_feature_receipt(block=block, traces=canonical)
    return FeatureSeal(
        block=block,
        traces=canonical,
        feature_receipt_sha256=receipt["feature_receipt_sha256"],
        trace_matrix_sha256=receipt["trace_matrix_sha256"],
        item_commitment_set_sha256=receipt["item_commitment_set_sha256"],
    )


@dataclass(frozen=True)
class HippoRetrievalSeal:
    """Immutable, label-free official-HippoRAG retrieval matrix seal."""

    block: str
    rows: tuple[HippoRetrieval, ...]
    retrieval_matrix_sha256: str
    item_commitment_set_sha256: str

    def __post_init__(self) -> None:
        if self.block not in {"A_hold", "M_search"} or not isinstance(
            self.rows, tuple
        ):
            raise HybridQaFormalRunnerError("HippoRAG seal block/rows are invalid")
        if (
            len(self.rows) != BLOCK_COUNTS[self.block]
            or any(not isinstance(row, HippoRetrieval) for row in self.rows)
            or self.rows
            != tuple(sorted(self.rows, key=lambda row: row.item_commitment_sha256))
            or len({row.item_commitment_sha256 for row in self.rows}) != len(self.rows)
        ):
            raise HybridQaFormalRunnerError("HippoRAG seal matrix drifted")
        payload = [row.payload() for row in self.rows]
        commitments = [row.item_commitment_sha256 for row in self.rows]
        if (
            self.retrieval_matrix_sha256 != stable_hash(payload)
            or self.item_commitment_set_sha256 != stable_hash(commitments)
        ):
            raise HybridQaFormalRunnerError("HippoRAG seal binding drifted")

    @property
    def by_item(self) -> Mapping[str, tuple[int, ...]]:
        return {row.item_commitment_sha256: row.top5 for row in self.rows}


def seal_hippo_retrievals(
    *, block: str, rows: Sequence[HippoRetrieval]
) -> HippoRetrievalSeal:
    if any(not isinstance(row, HippoRetrieval) for row in rows):
        raise HybridQaFormalRunnerError("HippoRAG retrieval rows contain a foreign type")
    canonical = tuple(sorted(rows, key=lambda row: row.item_commitment_sha256))
    payload = [row.payload() for row in canonical]
    commitments = [row.item_commitment_sha256 for row in canonical]
    return HippoRetrievalSeal(
        block=block,
        rows=canonical,
        retrieval_matrix_sha256=stable_hash(payload),
        item_commitment_set_sha256=stable_hash(commitments),
    )


def _balanced_fold_assignment(items: Sequence[str], secret: bytes) -> dict[str, int]:
    if not isinstance(secret, bytes) or len(secret) != 32:
        raise HybridQaFormalRunnerError("fold secret must contain exactly 32 bytes")
    rows = tuple(items)
    if len(rows) != len(set(rows)) or any(
        not _is_sha256(item) for item in rows
    ):
        raise HybridQaFormalRunnerError("fold item commitments are invalid")
    ordered = sorted(
        rows,
        key=lambda item: (
            hmac.new(
                secret,
                f"{VERSION}:A_form:balanced_fold:{item}".encode("ascii"),
                hashlib.sha256,
            ).digest(),
            item,
        ),
    )
    return {item: rank % 4 for rank, item in enumerate(ordered)}


@dataclass(frozen=True)
class E2FitSeal:
    """Frozen E2 model and receipt bound to the sealed A_form features."""

    a_form_features: FeatureSeal
    model: evaluator_math.E2Model
    receipt_json: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.a_form_features, FeatureSeal)
            or self.a_form_features.block != "A_form"
            or not isinstance(self.model, evaluator_math.E2Model)
        ):
            raise HybridQaFormalRunnerError("E2 fit seal dependencies are invalid")
        receipt = _mapping_from_canonical_json(
            self.receipt_json, field="E2 fit receipt"
        )
        _verify_self_hashed(
            receipt,
            schema=f"{VERSION}_e2_fit_receipt",
            field="fit_receipt_sha256",
        )
        required = {
            "version": VERSION,
            "block": "A_form",
            "feature_receipt_sha256": (
                self.a_form_features.feature_receipt_sha256
            ),
            "trace_matrix_sha256": self.a_form_features.trace_matrix_sha256,
            "item_commitment_set_sha256": (
                self.a_form_features.item_commitment_set_sha256
            ),
            "item_count": BLOCK_COUNTS["A_form"],
            "pair_count": BLOCK_COUNTS["A_form"] * 6,
            "pair_count_per_item": 6,
            "pair_weight": [1, 6],
            "ridge_lambda": "1",
            "intercept": False,
            "decimal_contract": {
                "precision": 80,
                "rounding": "ROUND_HALF_EVEN",
                "binary_float_inputs": False,
            },
            "fold_count": 4,
            "fold_policy": (
                "private_HMAC_SHA256_order_then_balanced_rank_mod_4_v1"
            ),
            "crossfit_descriptive_only": True,
            "final_fit_count": 1,
            "model": self.model.payload(),
            "utility_values_persisted": False,
            "F_search_accessed": False,
            "A_hold_accessed": False,
            "online_evaluator_calls": 0,
            "raw_content_persisted": False,
        }
        if any(receipt.get(key) != value for key, value in required.items()):
            raise HybridQaFormalRunnerError("E2 fit seal semantics drifted")
        for field in ("utility_matrix_sha256", "fold_assignment_sha256"):
            value = receipt.get(field)
            if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
                raise HybridQaFormalRunnerError("E2 fit seal hash field drifted")
        diagnostics = receipt.get("crossfit")
        expected_crossfit_counts = [
            (fold, 36, 12, 72) for fold in range(4)
        ]
        observed_crossfit_counts = (
            [
                (
                    row.get("fold"),
                    row.get("fit_item_count"),
                    row.get("held_item_count"),
                    row.get("held_pair_count"),
                )
                for row in diagnostics
            ]
            if isinstance(diagnostics, list)
            and len(diagnostics) == 4
            and all(isinstance(row, dict) for row in diagnostics)
            else None
        )
        if observed_crossfit_counts != expected_crossfit_counts:
            raise HybridQaFormalRunnerError("E2 fit cross-fit diagnostics drifted")

    @property
    def receipt(self) -> dict[str, Any]:
        return _mapping_from_canonical_json(
            self.receipt_json, field="E2 fit receipt"
        )

    @property
    def fit_receipt_sha256(self) -> str:
        return str(self.receipt["fit_receipt_sha256"])


def fit_e2(
    *,
    feature_seal: FeatureSeal,
    utilities: Mapping[tuple[str, str], Fraction | int],
    fold_secret: bytes,
) -> E2FitSeal:
    """Run descriptive four-fold diagnostics and one final 48-item fit."""

    if not isinstance(feature_seal, FeatureSeal) or feature_seal.block != "A_form":
        raise HybridQaFormalRunnerError("A_form must be feature-sealed before fit")
    try:
        matrix = evaluator_math._normalize_matrix(feature_seal.traces)
        normalized = evaluator_math._normalize_utilities(matrix, utilities)
    except evaluator_math.FeverousEvaluatorError as exc:
        raise HybridQaFormalRunnerError("A_form evaluator inputs are invalid") from exc
    if len(matrix) != BLOCK_COUNTS["A_form"]:
        raise HybridQaFormalRunnerError("A_form evaluator count drifted")
    fold_by_item = _balanced_fold_assignment(
        [item for item, _rows in matrix], fold_secret
    )
    if Counter(fold_by_item.values()) != Counter({fold: 12 for fold in range(4)}):
        raise HybridQaFormalRunnerError("balanced four-fold counts drifted")
    diagnostics: list[dict[str, Any]] = []
    for fold in range(4):
        fit_matrix = tuple(row for row in matrix if fold_by_item[row[0]] != fold)
        held_matrix = tuple(row for row in matrix if fold_by_item[row[0]] == fold)
        if not fit_matrix or not held_matrix:
            raise HybridQaFormalRunnerError("HMAC cross-fit has an empty partition")
        fit_keys = {
            (item, recipe)
            for item, _rows in fit_matrix
            for recipe in RECIPE_IDS
        }
        fold_utilities = {
            key: value for key, value in normalized.items() if key in fit_keys
        }
        model, _pairs = evaluator_math._fit_model(fit_matrix, fold_utilities)
        correct, non_tie, mse = evaluator_math._prediction_error(
            model, held_matrix, normalized
        )
        diagnostics.append(
            {
                "fold": fold,
                "fit_item_count": len(fit_matrix),
                "held_item_count": len(held_matrix),
                "held_pair_count": len(held_matrix) * 6,
                "held_non_tie_pair_count": non_tie,
                "held_preference_correct_count": correct,
                "held_pair_mean_squared_error": _decimal_text(mse),
                "fit_model_sha256": stable_hash(model.payload()),
            }
        )
    model, pairs = evaluator_math._fit_model(matrix, normalized)
    utility_payload = [
        [item, recipe, _fraction_payload(normalized[(item, recipe)])]
        for item, _rows in matrix
        for recipe in RECIPE_IDS
    ]
    assignment_payload = [[item, fold_by_item[item]] for item, _rows in matrix]
    body = {
        "schema": f"{VERSION}_e2_fit_receipt",
        "version": VERSION,
        "block": "A_form",
        "feature_receipt_sha256": feature_seal.feature_receipt_sha256,
        "trace_matrix_sha256": feature_seal.trace_matrix_sha256,
        "item_commitment_set_sha256": feature_seal.item_commitment_set_sha256,
        "utility_matrix_sha256": stable_hash(utility_payload),
        "item_count": len(matrix),
        "pair_count": len(pairs),
        "pair_count_per_item": 6,
        "pair_weight": [1, 6],
        "ridge_lambda": "1",
        "intercept": False,
        "decimal_contract": {
            "precision": 80,
            "rounding": "ROUND_HALF_EVEN",
            "binary_float_inputs": False,
        },
        "fold_count": 4,
        "fold_policy": "private_HMAC_SHA256_order_then_balanced_rank_mod_4_v1",
        "crossfit_descriptive_only": True,
        "crossfit": diagnostics,
        "fold_assignment_sha256": stable_hash(assignment_payload),
        "final_fit_count": 1,
        "model": model.payload(),
        "utility_values_persisted": False,
        "F_search_accessed": False,
        "A_hold_accessed": False,
        "online_evaluator_calls": 0,
        "raw_content_persisted": False,
    }
    receipt = _self_hashed(body, "fit_receipt_sha256")
    return E2FitSeal(
        a_form_features=feature_seal,
        model=model,
        receipt_json=_canonical_json_text(receipt),
    )


@dataclass(frozen=True)
class PolicySeal:
    """Frozen global E0/E2 policies bound to A_form fit and F_search features."""

    f_search_features: FeatureSeal
    fit: E2FitSeal
    receipt_json: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.f_search_features, FeatureSeal)
            or self.f_search_features.block != "F_search"
            or not isinstance(self.fit, E2FitSeal)
        ):
            raise HybridQaFormalRunnerError("policy seal dependencies are invalid")
        receipt = _mapping_from_canonical_json(
            self.receipt_json, field="policy receipt"
        )
        _verify_self_hashed(
            receipt,
            schema=f"{VERSION}_policy_receipt",
            field="policy_receipt_sha256",
        )
        required = {
            "version": VERSION,
            "block": "F_search",
            "F_feature_receipt_sha256": (
                self.f_search_features.feature_receipt_sha256
            ),
            "A_form_feature_receipt_sha256": (
                self.fit.a_form_features.feature_receipt_sha256
            ),
            "fit_receipt_sha256": self.fit.fit_receipt_sha256,
            "trace_matrix_sha256": self.f_search_features.trace_matrix_sha256,
            "item_commitment_set_sha256": (
                self.f_search_features.item_commitment_set_sha256
            ),
            "item_count": BLOCK_COUNTS["F_search"],
            "recipe_registry": list(RECIPE_IDS),
            "labels_gold_utility_or_family_accessed": False,
            "A_hold_authorized": True,
            "M_search_authorized_before_A_hold_promotion": False,
            "online_evaluator_calls": 0,
            "raw_content_persisted": False,
        }
        if any(receipt.get(key) != value for key, value in required.items()):
            raise HybridQaFormalRunnerError("policy seal semantics drifted")
        e0_recipe = receipt.get("E0_selected_recipe_id")
        e2_recipe = receipt.get("E2_selected_recipe_id")
        same = e0_recipe == e2_recipe
        identical = receipt.get("identical_all_F_ordered_top5")
        if (
            e0_recipe not in RECIPE_IDS
            or e2_recipe not in RECIPE_IDS
            or type(identical) is not bool
            or receipt.get("same_recipe") is not same
            or receipt.get("evaluator_comparison_identifiable")
            is not (not same and not identical)
        ):
            raise HybridQaFormalRunnerError("policy seal selection drifted")

    @property
    def receipt(self) -> dict[str, Any]:
        return _mapping_from_canonical_json(
            self.receipt_json, field="policy receipt"
        )

    @property
    def policy_receipt_sha256(self) -> str:
        return str(self.receipt["policy_receipt_sha256"])

    @property
    def e0_recipe_id(self) -> str:
        return str(self.receipt["E0_selected_recipe_id"])

    @property
    def e2_recipe_id(self) -> str:
        return str(self.receipt["E2_selected_recipe_id"])

    @property
    def identifiable(self) -> bool:
        return bool(self.receipt["evaluator_comparison_identifiable"])


def freeze_f_policies(
    *, feature_seal: FeatureSeal, fit_seal: E2FitSeal
) -> PolicySeal:
    if (
        not isinstance(feature_seal, FeatureSeal)
        or feature_seal.block != "F_search"
        or not isinstance(fit_seal, E2FitSeal)
    ):
        raise HybridQaFormalRunnerError("F policy inputs are not sealed")
    try:
        matrix = evaluator_math._normalize_matrix(feature_seal.traces)
        e0_by_item = evaluator_math.e0_item_scores(feature_seal.traces)
    except evaluator_math.FeverousEvaluatorError as exc:
        raise HybridQaFormalRunnerError("F_search trace matrix is invalid") from exc
    if len(matrix) != BLOCK_COUNTS["F_search"]:
        raise HybridQaFormalRunnerError("F_search item count drifted")
    e0 = {
        recipe: sum(
            (e0_by_item[item][recipe] for item, _rows in matrix), Fraction(0)
        )
        / len(matrix)
        for recipe in RECIPE_IDS
    }
    with localcontext() as context:
        context.prec = 80
        context.rounding = ROUND_HALF_EVEN
        e2 = {
            recipe: sum(
                (
                    fit_seal.model.predict(
                        next(
                            row for row in rows if row.recipe_id == recipe
                        ).features
                    )
                    for _item, rows in matrix
                ),
                Decimal(0),
            )
            / Decimal(len(matrix))
            for recipe in RECIPE_IDS
        }
    e0_recipe = min(RECIPE_IDS, key=lambda recipe: (-e0[recipe], recipe))
    e2_recipe = min(RECIPE_IDS, key=lambda recipe: (-e2[recipe], recipe))
    identical = all(
        next(row for row in rows if row.recipe_id == e0_recipe).behavior_sha256
        == next(row for row in rows if row.recipe_id == e2_recipe).behavior_sha256
        for _item, rows in matrix
    )
    body = {
        "schema": f"{VERSION}_policy_receipt",
        "version": VERSION,
        "block": "F_search",
        "F_feature_receipt_sha256": feature_seal.feature_receipt_sha256,
        "A_form_feature_receipt_sha256": (
            fit_seal.a_form_features.feature_receipt_sha256
        ),
        "fit_receipt_sha256": fit_seal.fit_receipt_sha256,
        "trace_matrix_sha256": feature_seal.trace_matrix_sha256,
        "item_commitment_set_sha256": feature_seal.item_commitment_set_sha256,
        "item_count": len(matrix),
        "recipe_registry": list(RECIPE_IDS),
        "E0_recipe_scores": {
            recipe: _fraction_payload(e0[recipe]) for recipe in RECIPE_IDS
        },
        "E2_recipe_scores": {
            recipe: _decimal_text(e2[recipe]) for recipe in RECIPE_IDS
        },
        "E0_selected_recipe_id": e0_recipe,
        "E2_selected_recipe_id": e2_recipe,
        "same_recipe": e0_recipe == e2_recipe,
        "identical_all_F_ordered_top5": identical,
        "evaluator_comparison_identifiable": e0_recipe != e2_recipe and not identical,
        "labels_gold_utility_or_family_accessed": False,
        "A_hold_authorized": True,
        "M_search_authorized_before_A_hold_promotion": False,
        "online_evaluator_calls": 0,
        "raw_content_persisted": False,
    }
    receipt = _self_hashed(body, "policy_receipt_sha256")
    return PolicySeal(
        f_search_features=feature_seal,
        fit=fit_seal,
        receipt_json=_canonical_json_text(receipt),
    )


def _sign_flip_payload(deltas: Sequence[Fraction]) -> dict[str, Any]:
    try:
        result = evaluator_math.exact_magnitude_preserving_sign_flip(deltas)
    except evaluator_math.FeverousEvaluatorError as exc:
        raise HybridQaFormalRunnerError("exact sign-flip test failed") from exc
    body = result.payload()
    # The inherited implementation fixes the same alpha but carries its trust
    # root's historical schema label; this explicit field records this study.
    body["test"] = "hybridqa_one_sided_exact_magnitude_preserving_sign_flip_v1"
    body["consumer"] = VERSION
    return body


@dataclass(frozen=True)
class AnchorScoreSeal:
    """Terminal A_hold/M score bound to all pre-label sealed dependencies."""

    block: str
    anchor_features: FeatureSeal
    hippo_retrievals: HippoRetrievalSeal
    policies: PolicySeal
    a_hold_authorization: "AnchorScoreSeal | None"
    receipt_json: str

    def __post_init__(self) -> None:
        if (
            self.block not in {"A_hold", "M_search"}
            or not isinstance(self.anchor_features, FeatureSeal)
            or self.anchor_features.block != self.block
            or not isinstance(self.hippo_retrievals, HippoRetrievalSeal)
            or self.hippo_retrievals.block != self.block
            or not isinstance(self.policies, PolicySeal)
        ):
            raise HybridQaFormalRunnerError("anchor score seal dependencies are invalid")
        if self.block == "A_hold":
            if self.a_hold_authorization is not None:
                raise HybridQaFormalRunnerError("A_hold cannot carry M authorization")
            authorization_sha: str | None = None
        else:
            if (
                not isinstance(self.a_hold_authorization, AnchorScoreSeal)
                or self.a_hold_authorization.block != "A_hold"
                or not self.a_hold_authorization.evaluator_promoted
                or self.a_hold_authorization.policies.policy_receipt_sha256
                != self.policies.policy_receipt_sha256
            ):
                raise HybridQaFormalRunnerError(
                    "M_search lacks a promoted, policy-matched A_hold seal"
                )
            authorization_sha = self.a_hold_authorization.score_receipt_sha256
        receipt = _mapping_from_canonical_json(
            self.receipt_json, field=f"{self.block} score receipt"
        )
        _verify_self_hashed(
            receipt,
            schema=f"{VERSION}_{self.block}_score_receipt",
            field="score_receipt_sha256",
        )
        required = {
            "version": VERSION,
            "block": self.block,
            "item_count": BLOCK_COUNTS[self.block],
            "anchor_feature_receipt_sha256": (
                self.anchor_features.feature_receipt_sha256
            ),
            "policy_receipt_sha256": self.policies.policy_receipt_sha256,
            "hipporag_retrieval_matrix_sha256": (
                self.hippo_retrievals.retrieval_matrix_sha256
            ),
            "item_commitment_set_sha256": (
                self.anchor_features.item_commitment_set_sha256
            ),
            "A_hold_authorization_score_receipt_sha256": authorization_sha,
            "E0_recipe_id": self.policies.e0_recipe_id,
            "E2_recipe_id": self.policies.e2_recipe_id,
            "evaluator_comparison_identifiable": self.policies.identifiable,
            "family_item_counts": BLOCK_FAMILY_COUNTS[self.block],
            "item_level_utility_values_persisted": False,
            "online_evaluator_calls": 0,
            "raw_content_persisted": False,
        }
        if any(receipt.get(key) != value for key, value in required.items()):
            raise HybridQaFormalRunnerError("anchor score seal semantics drifted")

    @property
    def receipt(self) -> dict[str, Any]:
        return _mapping_from_canonical_json(
            self.receipt_json, field=f"{self.block} score receipt"
        )

    @property
    def score_receipt_sha256(self) -> str:
        return str(self.receipt["score_receipt_sha256"])

    @property
    def evaluator_promoted(self) -> bool:
        return bool(self.receipt.get("evaluator_promoted"))


def score_anchor(
    *,
    block: str,
    items: Sequence[ItemExecution],
    labels: Sequence[AnchorLabel],
    anchor_feature_seal: FeatureSeal,
    hippo_retrieval_seal: HippoRetrievalSeal,
    policy_seal: PolicySeal,
    a_hold_authorization: AnchorScoreSeal | None = None,
) -> AnchorScoreSeal:
    """Score A_hold or M only after all three logical arms are terminal."""

    if block not in {"A_hold", "M_search"} or len(items) != BLOCK_COUNTS[block]:
        raise HybridQaFormalRunnerError("anchor block identity drifted")
    if (
        not isinstance(anchor_feature_seal, FeatureSeal)
        or anchor_feature_seal.block != block
        or not isinstance(hippo_retrieval_seal, HippoRetrievalSeal)
        or hippo_retrieval_seal.block != block
        or not isinstance(policy_seal, PolicySeal)
    ):
        raise HybridQaFormalRunnerError("anchor inputs are not sealed")
    if block == "A_hold" and a_hold_authorization is not None:
        raise HybridQaFormalRunnerError("A_hold cannot use an earlier score gate")
    if block == "M_search" and (
        not isinstance(a_hold_authorization, AnchorScoreSeal)
        or not a_hold_authorization.evaluator_promoted
        or a_hold_authorization.policies.policy_receipt_sha256
        != policy_seal.policy_receipt_sha256
    ):
        raise HybridQaFormalRunnerError("M_search is not authorized by A_hold")

    if any(not isinstance(item, ItemExecution) for item in items):
        raise HybridQaFormalRunnerError("anchor items contain a foreign type")
    items_by_commitment = {item.item_commitment_sha256: item for item in items}
    if len(items_by_commitment) != len(items):
        raise HybridQaFormalRunnerError("anchor item commitment duplicated")
    commitments = tuple(sorted(items_by_commitment))
    expected_traces = tuple(
        trace
        for commitment in commitments
        for trace in items_by_commitment[commitment].recipe_traces
    )
    if (
        anchor_feature_seal.traces != expected_traces
        or anchor_feature_seal.item_commitments != commitments
        or anchor_feature_seal.item_commitment_set_sha256
        != stable_hash(list(commitments))
    ):
        raise HybridQaFormalRunnerError("anchor items are outside the feature seal")

    if any(not isinstance(label, AnchorLabel) for label in labels):
        raise HybridQaFormalRunnerError("anchor labels contain a foreign type")
    labels_by_commitment = {
        label.item_commitment_sha256: label for label in labels
    }
    if (
        len(labels_by_commitment) != len(labels)
        or set(labels_by_commitment) != set(commitments)
        or set(hippo_retrieval_seal.by_item) != set(commitments)
        or hippo_retrieval_seal.item_commitment_set_sha256
        != anchor_feature_seal.item_commitment_set_sha256
    ):
        raise HybridQaFormalRunnerError("anchor commitment-keyed alignment drifted")
    family_counts = Counter(label.family for label in labels_by_commitment.values())
    if dict(family_counts) != BLOCK_FAMILY_COUNTS[block]:
        raise HybridQaFormalRunnerError("anchor per-family counts drifted")

    e0_recipe_id = policy_seal.e0_recipe_id
    e2_recipe_id = policy_seal.e2_recipe_id
    identifiable = policy_seal.identifiable
    deltas_e0: list[Fraction] = []
    deltas_hippo: list[Fraction] = []
    deltas_raw: list[Fraction] = []
    family_deltas: dict[str, list[Fraction]] = defaultdict(list)
    complete = {arm: 0 for arm in ("E0", "E2", "HippoRAG", "RAW")}
    hippo_by_commitment = hippo_retrieval_seal.by_item
    for commitment in commitments:
        item = items_by_commitment[commitment]
        label = labels_by_commitment[commitment]
        gold = label.gold_ordinals
        family = label.family
        hippo = hippo_by_commitment[commitment]
        outputs = item.outputs
        scored: dict[str, tuple[Fraction, bool]] = {
            "E0": item_utility(outputs[e0_recipe_id], gold),
            "E2": item_utility(outputs[e2_recipe_id], gold),
            "HippoRAG": item_utility(hippo, gold),
            "RAW": item_utility(outputs["R0_DENSE5"], gold),
        }
        for arm, (_value, is_complete) in scored.items():
            complete[arm] += int(is_complete)
        d0 = scored["E2"][0] - scored["E0"][0]
        dh = scored["E2"][0] - scored["HippoRAG"][0]
        dr = scored["E2"][0] - scored["RAW"][0]
        deltas_e0.append(d0)
        deltas_hippo.append(dh)
        deltas_raw.append(dr)
        family_deltas[family].append(dh)
    e0_test = _sign_flip_payload(deltas_e0)
    hippo_test = _sign_flip_payload(deltas_hippo)
    raw_test = _sign_flip_payload(deltas_raw)
    family_sums = {
        family: sum(family_deltas[family], Fraction(0))
        for family in FAMILIES
    }
    evaluator_promoted = bool(e0_test["promoted"]) and identifiable
    primary = bool(hippo_test["promoted"]) and all(
        value > 0 for value in family_sums.values()
    )
    raw_overcome = bool(raw_test["promoted"]) and complete["E2"] >= complete["RAW"]
    authorization_sha = (
        a_hold_authorization.score_receipt_sha256
        if a_hold_authorization is not None
        else None
    )
    body = {
        "schema": f"{VERSION}_{block}_score_receipt",
        "version": VERSION,
        "block": block,
        "item_count": len(items),
        "logical_RAW_HippoRAG_Agent_work_units": 3 * len(items),
        "anchor_feature_receipt_sha256": (
            anchor_feature_seal.feature_receipt_sha256
        ),
        "policy_receipt_sha256": policy_seal.policy_receipt_sha256,
        "hipporag_retrieval_matrix_sha256": (
            hippo_retrieval_seal.retrieval_matrix_sha256
        ),
        "item_commitment_set_sha256": (
            anchor_feature_seal.item_commitment_set_sha256
        ),
        "late_opened_label_matrix_sha256": stable_hash(
            [
                [
                    commitment,
                    list(labels_by_commitment[commitment].gold_ordinals),
                    labels_by_commitment[commitment].family,
                ]
                for commitment in commitments
            ]
        ),
        "A_hold_authorization_score_receipt_sha256": authorization_sha,
        "E0_recipe_id": e0_recipe_id,
        "E2_recipe_id": e2_recipe_id,
        "evaluator_comparison_identifiable": identifiable,
        "E2_minus_E0": e0_test,
        "E2_minus_HippoRAG": hippo_test,
        "E2_minus_RAW": raw_test,
        "E2_minus_HippoRAG_family_sums": {
            family: _fraction_payload(value) for family, value in family_sums.items()
        },
        "family_item_counts": dict(family_counts),
        "complete_counts": complete,
        "A_hold_real_domain_primary_passed": primary if block == "A_hold" else None,
        "evaluator_promoted": evaluator_promoted if block == "A_hold" else None,
        "M_L5_passed": evaluator_promoted if block == "M_search" else None,
        "RAW_complete_advantage_overcome": raw_overcome,
        "item_level_utility_values_persisted": False,
        "online_evaluator_calls": 0,
        "raw_content_persisted": False,
    }
    receipt = _self_hashed(body, "score_receipt_sha256")
    return AnchorScoreSeal(
        block=block,
        anchor_features=anchor_feature_seal,
        hippo_retrievals=hippo_retrieval_seal,
        policies=policy_seal,
        a_hold_authorization=a_hold_authorization,
        receipt_json=_canonical_json_text(receipt),
    )


__all__ = [
    "AnchorLabel",
    "AnchorScoreSeal",
    "BLOCK_COUNTS",
    "BLOCK_FAMILY_COUNTS",
    "BLOCK_ORDER",
    "BULK_SEMANTIC_BATCH_LIMIT",
    "BulkQueryInput",
    "CORPUS_UNIT_COUNT",
    "DIRECT_ANCHOR_K",
    "E2FitSeal",
    "FACET_LIMIT",
    "FAMILIES",
    "FEATURE_ORDER",
    "FeatureSeal",
    "HippoRetrieval",
    "HippoRetrievalSeal",
    "HybridQaFormalRunnerError",
    "ItemExecution",
    "PolicySeal",
    "RECIPE_IDS",
    "VERSION",
    "build_feature_receipt",
    "build_query_semantic_tensor",
    "build_query_semantic_tensors_bulk",
    "exact_action_features",
    "execute_item",
    "extract_query_facets",
    "fit_e2",
    "freeze_f_policies",
    "item_utility",
    "recipe_trace_from_action",
    "score_anchor",
    "seal_feature_matrix",
    "seal_hippo_retrievals",
    "stable_hash",
]
