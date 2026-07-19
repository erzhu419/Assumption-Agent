"""Frozen, label-free contract for one global HybridQA HippoRAG index.

The adapter accepts exactly 609 logical corpus units and query text.  It has no
fields for answers, evidence, relation families, or scores.  HippoRAG addresses
documents by exact text, so byte-identical ``title + "\n\n" + body`` values are
indexed once and the official score is deterministically expanded back to all
logical corpus ordinals before the stable top-five is selected.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence


ADAPTER_VERSION = "hybridqa_official_hipporag_global_retrieve_only_v1"
OFFICIAL_HIPPORAG_COMMIT = "ef2f14c4f254f11ac29f9395f262466ad1bb4d10"
RUNTIME_TRUST_ROOT = "musique_official_hipporag_filesystem_attestation_v3"

CORPUS_SIZE = 609
TOP_K = 5
MAX_QUERY_BATCH = 8
MAX_QUERY_COUNT = 4096
MAX_TEXT_CHARACTERS = 1_000_000

ARTICLE_KEYS = frozenset({"idx", "title", "body"})
CORPUS_INPUT_SCHEMA = "hybridqa_official_hipporag_corpus_input_v1"
QUERY_INPUT_SCHEMA = "hybridqa_official_hipporag_query_input_v1"
BUILD_RECEIPT_SCHEMA = "hybridqa_official_hipporag_global_build_receipt_v1"
RETRIEVAL_RECEIPT_SCHEMA = (
    "hybridqa_official_hipporag_global_retrieval_receipt_v1"
)
RETRIEVAL_OUTPUT_SCHEMA = "hybridqa_official_hipporag_idx_receipt_output_v1"
SERIALIZATION = "title_utf8_then_two_lf_then_body_utf8_v1"
DUPLICATE_EXPANSION_POLICY = (
    "official_exact_text_dedup_then_equal_score_expand_to_all_logical_idx_v1"
)

FROZEN_CORE_CONFIG: dict[str, Any] = {
    "config_class": "hipporag.utils.config_utils.BaseConfig",
    "core_class": "hipporag.HippoRAG",
    "llm_backend": "Transformers/local_asset",
    "embedding_backend": "Transformers/local_asset",
    "openie_mode": "online",
    "max_new_tokens": 4,
    "retrieval_top_k": CORPUS_SIZE,
    "qa_top_k": TOP_K,
    "save_openie": True,
    "build_force_index_from_scratch": True,
    "reopen_force_index_from_scratch": False,
    "official_retrieve_num_to_retrieve": CORPUS_SIZE,
    "official_content_addressing": "exact_text_hash_deduplicated_v1",
    "logical_duplicate_expansion": DUPLICATE_EXPANSION_POLICY,
    "query_batch_size_upper_bound": MAX_QUERY_BATCH,
    "query_concurrency_upper_bound": MAX_QUERY_BATCH,
    "adapter_top_k_selection": "negative_official_score_then_logical_idx_v1",
    "network_namespace": "isolated_no_transport",
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_BUILD_RECEIPT_KEYS = frozenset(
    {
        "adapter_version",
        "corpus_count",
        "corpus_sha256",
        "duplicate_expansion_policy",
        "duplicate_text_group_count",
        "duplicate_text_unit_count",
        "force_index_from_scratch",
        "index_call_count",
        "index_file_count",
        "index_total_bytes",
        "index_tree_sha256",
        "official_hipporag_commit",
        "official_unique_text_count",
        "receipt_sha256",
        "runtime_attestation_receipt_sha256",
        "runtime_trust_root",
        "schema",
        "serialization",
        "status",
    }
)
_RETRIEVAL_RECEIPT_KEYS = frozenset(
    {
        "adapter_version",
        "batch_sizes",
        "build_receipt_sha256",
        "corpus_count",
        "corpus_sha256",
        "duplicate_expansion_policy",
        "duplicate_text_group_count",
        "duplicate_text_unit_count",
        "force_index_from_scratch",
        "index_call_count",
        "index_changed_during_retrieve",
        "index_file_count",
        "index_post_file_count",
        "index_post_total_bytes",
        "index_post_tree_sha256",
        "index_total_bytes",
        "index_tree_sha256",
        "official_hipporag_commit",
        "official_unique_text_count",
        "query_sha256",
        "query_count",
        "receipt_sha256",
        "result_idx_sha256",
        "retrieval_call_count",
        "runtime_attestation_receipt_sha256",
        "runtime_trust_root",
        "schema",
        "serialization",
        "status",
    }
)


class HybridQAOfficialHippoRAGError(RuntimeError):
    """Raised when the global retrieve-only contract cannot be proven."""


@dataclass(frozen=True)
class CorpusArticle:
    idx: int
    title: str
    body: str


@dataclass(frozen=True)
class RetrievalBatch:
    """The complete public retrieval return value: indices and a receipt."""

    indices: tuple[tuple[int, ...], ...]
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class IndexTreeSnapshot:
    """Canonical, content-complete identity of one persisted official index."""

    file_count: int
    total_bytes: int
    tree_sha256: str


def canonical_json_bytes(value: object) -> bytes:
    """Encode one exact JSON value for private IPC and public receipts."""

    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _stable_hash(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_index_tree(index_root: Path) -> IndexTreeSnapshot:
    """Hash every directory and regular file; reject links and special files.

    The canonical row set includes relative paths, entry kinds, permission bits,
    file sizes, and complete file hashes.  This makes the build receipt an
    identity for the actual persisted index rather than merely for its corpus.
    """

    root = index_root.absolute()
    if root.is_symlink() or not root.is_dir():
        raise HybridQAOfficialHippoRAGError(
            "persistent official index root is unavailable"
        )
    rows: list[dict[str, object]] = []
    file_count = 0
    total_bytes = 0
    try:
        entries = sorted(root.rglob("*"), key=lambda row: row.relative_to(root).as_posix())
        for entry in entries:
            relative = entry.relative_to(root).as_posix()
            metadata = entry.lstat()
            mode = stat.S_IMODE(metadata.st_mode)
            if stat.S_ISLNK(metadata.st_mode):
                raise HybridQAOfficialHippoRAGError(
                    "persistent official index contains a symbolic link"
                )
            if stat.S_ISDIR(metadata.st_mode):
                rows.append({"kind": "directory", "mode": mode, "path": relative})
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise HybridQAOfficialHippoRAGError(
                    "persistent official index contains a special file"
                )
            size = metadata.st_size
            rows.append(
                {
                    "kind": "file",
                    "mode": mode,
                    "path": relative,
                    "sha256": _sha256_file(entry),
                    "size": size,
                }
            )
            file_count += 1
            total_bytes += size
    except OSError as exc:
        raise HybridQAOfficialHippoRAGError(
            "persistent official index cannot be snapshotted"
        ) from exc
    if file_count == 0:
        raise HybridQAOfficialHippoRAGError(
            "persistent official index contains no regular files"
        )
    return IndexTreeSnapshot(
        file_count=file_count,
        total_bytes=total_bytes,
        tree_sha256=_stable_hash(rows),
    )


def _validate_index_snapshot(
    snapshot: IndexTreeSnapshot, field: str
) -> IndexTreeSnapshot:
    if (
        not isinstance(snapshot, IndexTreeSnapshot)
        or isinstance(snapshot.file_count, bool)
        or not isinstance(snapshot.file_count, int)
        or snapshot.file_count < 1
        or isinstance(snapshot.total_bytes, bool)
        or not isinstance(snapshot.total_bytes, int)
        or snapshot.total_bytes < 0
        or not isinstance(snapshot.tree_sha256, str)
        or _SHA256_RE.fullmatch(snapshot.tree_sha256) is None
    ):
        raise HybridQAOfficialHippoRAGError(f"{field} is invalid")
    return snapshot


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HybridQAOfficialHippoRAGError(f"{field} must be non-empty text")
    if "\x00" in value:
        raise HybridQAOfficialHippoRAGError(f"{field} contains a NUL character")
    if len(value) > MAX_TEXT_CHARACTERS:
        raise HybridQAOfficialHippoRAGError(
            f"{field} exceeds the frozen character bound"
        )
    return value


def validate_corpus(
    articles: Sequence[Mapping[str, object]],
) -> tuple[CorpusArticle, ...]:
    """Validate the exact global corpus without accepting benchmark labels."""

    if isinstance(articles, (str, bytes)) or not isinstance(articles, Sequence):
        raise HybridQAOfficialHippoRAGError("articles must be a sequence")
    if len(articles) != CORPUS_SIZE:
        raise HybridQAOfficialHippoRAGError(
            f"global corpus must contain exactly {CORPUS_SIZE} articles"
        )
    rows: list[CorpusArticle] = []
    for position, raw in enumerate(articles):
        if not isinstance(raw, Mapping) or set(raw) != ARTICLE_KEYS:
            raise HybridQAOfficialHippoRAGError(
                "article must contain only idx, title, and body"
            )
        idx = raw.get("idx")
        if isinstance(idx, bool) or not isinstance(idx, int) or idx != position:
            raise HybridQAOfficialHippoRAGError(
                "article idx must be canonical contiguous zero-based order"
            )
        rows.append(
            CorpusArticle(
                idx=idx,
                title=_required_text(raw.get("title"), f"articles[{position}].title"),
                body=_required_text(raw.get("body"), f"articles[{position}].body"),
            )
        )
    return tuple(rows)


def serialize_article(article: CorpusArticle) -> str:
    """Apply the exact official-document serialization: title, blank, body."""

    return article.title + "\n\n" + article.body


def serialize_corpus(articles: Sequence[CorpusArticle]) -> tuple[str, ...]:
    return tuple(serialize_article(article) for article in articles)


def corpus_sha256(documents: Sequence[str]) -> str:
    if (
        isinstance(documents, (str, bytes))
        or not isinstance(documents, Sequence)
        or len(documents) != CORPUS_SIZE
        or any(not isinstance(document, str) or not document for document in documents)
    ):
        raise HybridQAOfficialHippoRAGError("serialized global corpus is invalid")
    return _stable_hash(list(documents))


def corpus_text_multiplicity(documents: Sequence[str]) -> dict[str, int]:
    """Return content-free aggregates for the exact-text quotient corpus."""

    corpus_sha256(documents)
    counts: dict[str, int] = {}
    # HippoRAG's pinned passage store addresses text with MD5.  Prove that no
    # two distinct strings alias before either build or receipt formation.
    content_hash_to_text: dict[str, str] = {}
    for document in documents:
        content_hash = hashlib.md5(document.encode("utf-8")).hexdigest()
        prior = content_hash_to_text.setdefault(content_hash, document)
        if prior != document:
            raise HybridQAOfficialHippoRAGError(
                "distinct corpus texts collide under the official content hash"
            )
        counts[document] = counts.get(document, 0) + 1
    return {
        "duplicate_text_group_count": sum(count > 1 for count in counts.values()),
        "duplicate_text_unit_count": sum(
            count for count in counts.values() if count > 1
        ),
        "official_unique_text_count": len(counts),
    }


def validate_queries(queries: Sequence[str]) -> tuple[str, ...]:
    """Validate query-only input; no answer/evidence/type channel exists."""

    if isinstance(queries, (str, bytes)) or not isinstance(queries, Sequence):
        raise HybridQAOfficialHippoRAGError("queries must be a sequence")
    if not 1 <= len(queries) <= MAX_QUERY_COUNT:
        raise HybridQAOfficialHippoRAGError("query count is outside the frozen bound")
    return tuple(
        _required_text(query, f"queries[{position}]")
        for position, query in enumerate(queries)
    )


def stable_top_five_from_official_result(
    *,
    retrieved_documents: Sequence[object],
    retrieved_scores: Sequence[object],
    document_to_indices: Mapping[str, Sequence[int]],
) -> tuple[int, ...]:
    """Expand every official unique-text score to stable logical ordinals."""

    if isinstance(retrieved_documents, (str, bytes)) or isinstance(
        retrieved_scores, (str, bytes)
    ):
        raise HybridQAOfficialHippoRAGError("official result rows are malformed")
    try:
        documents = list(retrieved_documents)
        scores = list(retrieved_scores)
    except TypeError as exc:
        raise HybridQAOfficialHippoRAGError(
            "official result rows are not iterable"
        ) from exc
    canonical_mapping: dict[str, tuple[int, ...]] = {}
    all_indices: list[int] = []
    for document, raw_indices in document_to_indices.items():
        if not isinstance(document, str) or not document:
            raise HybridQAOfficialHippoRAGError(
                "document-to-indices mapping contains an invalid document"
            )
        if isinstance(raw_indices, (str, bytes)):
            raise HybridQAOfficialHippoRAGError(
                "document-to-indices mapping contains malformed indices"
            )
        try:
            indices = tuple(raw_indices)
        except TypeError as exc:
            raise HybridQAOfficialHippoRAGError(
                "document-to-indices mapping contains malformed indices"
            ) from exc
        if not indices or any(type(idx) is not int for idx in indices):
            raise HybridQAOfficialHippoRAGError(
                "document-to-indices mapping contains malformed indices"
            )
        canonical_mapping[document] = indices
        all_indices.extend(indices)
    if (
        sorted(all_indices) != list(range(CORPUS_SIZE))
        or len(documents) != len(canonical_mapping)
        or len(scores) != len(canonical_mapping)
    ):
        raise HybridQAOfficialHippoRAGError(
            "official retrieve must return every unique global corpus text exactly once"
        )

    ranked: list[tuple[float, int]] = []
    seen: set[str] = set()
    for document, score in zip(documents, scores):
        if not isinstance(document, str) or document not in canonical_mapping:
            raise HybridQAOfficialHippoRAGError(
                "official result contains an unknown corpus document"
            )
        if document in seen:
            raise HybridQAOfficialHippoRAGError(
                "official result contains a duplicate corpus document"
            )
        seen.add(document)
        if isinstance(score, bool) or not isinstance(score, Real):
            raise HybridQAOfficialHippoRAGError(
                "official result score is not numeric"
            )
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise HybridQAOfficialHippoRAGError(
                "official result score is not finite"
            )
        ranked.extend((numeric_score, idx) for idx in canonical_mapping[document])
    if seen != set(canonical_mapping):
        raise HybridQAOfficialHippoRAGError(
            "official result omitted a global corpus article"
        )
    ranked.sort(key=lambda row: (-row[0], row[1]))
    result = tuple(idx for _score, idx in ranked[:TOP_K])
    if len(result) != TOP_K or len(set(result)) != TOP_K:
        raise HybridQAOfficialHippoRAGError(
            "adapter did not produce five unique article indices"
        )
    return result


def make_build_receipt(
    documents: Sequence[str],
    *,
    index_snapshot: IndexTreeSnapshot,
    runtime_attestation_receipt_sha256: str,
) -> dict[str, Any]:
    snapshot = _validate_index_snapshot(index_snapshot, "build index snapshot")
    if (
        not isinstance(runtime_attestation_receipt_sha256, str)
        or _SHA256_RE.fullmatch(runtime_attestation_receipt_sha256) is None
    ):
        raise HybridQAOfficialHippoRAGError(
            "runtime attestation receipt hash is malformed"
        )
    multiplicity = corpus_text_multiplicity(documents)
    receipt: dict[str, Any] = {
        "adapter_version": ADAPTER_VERSION,
        "corpus_count": CORPUS_SIZE,
        "corpus_sha256": corpus_sha256(documents),
        "duplicate_expansion_policy": DUPLICATE_EXPANSION_POLICY,
        **multiplicity,
        "force_index_from_scratch": True,
        "index_call_count": 1,
        "index_file_count": snapshot.file_count,
        "index_total_bytes": snapshot.total_bytes,
        "index_tree_sha256": snapshot.tree_sha256,
        "official_hipporag_commit": OFFICIAL_HIPPORAG_COMMIT,
        "runtime_attestation_receipt_sha256": runtime_attestation_receipt_sha256,
        "runtime_trust_root": RUNTIME_TRUST_ROOT,
        "schema": BUILD_RECEIPT_SCHEMA,
        "serialization": SERIALIZATION,
        "status": "passed_build_once",
    }
    receipt["receipt_sha256"] = _stable_hash(receipt)
    return receipt


def validate_build_receipt(
    receipt: Mapping[str, object],
    *,
    expected_documents: Sequence[str],
    expected_index_snapshot: IndexTreeSnapshot,
    expected_runtime_attestation_receipt_sha256: str,
) -> dict[str, Any]:
    if set(receipt) != _BUILD_RECEIPT_KEYS:
        raise HybridQAOfficialHippoRAGError("build receipt key set mismatch")
    payload = dict(receipt)
    self_hash = payload.pop("receipt_sha256", None)
    if not isinstance(self_hash, str) or _SHA256_RE.fullmatch(self_hash) is None:
        raise HybridQAOfficialHippoRAGError("build receipt self-hash is malformed")
    if self_hash != _stable_hash(payload):
        raise HybridQAOfficialHippoRAGError("build receipt self-hash mismatch")
    snapshot = _validate_index_snapshot(
        expected_index_snapshot, "expected build index snapshot"
    )
    if (
        not isinstance(expected_runtime_attestation_receipt_sha256, str)
        or _SHA256_RE.fullmatch(expected_runtime_attestation_receipt_sha256) is None
    ):
        raise HybridQAOfficialHippoRAGError(
            "expected runtime attestation receipt hash is malformed"
        )
    documents = tuple(expected_documents)
    expected = {
        "adapter_version": ADAPTER_VERSION,
        "corpus_count": CORPUS_SIZE,
        "corpus_sha256": corpus_sha256(documents),
        "duplicate_expansion_policy": DUPLICATE_EXPANSION_POLICY,
        **corpus_text_multiplicity(documents),
        "force_index_from_scratch": True,
        "index_call_count": 1,
        "index_file_count": snapshot.file_count,
        "index_total_bytes": snapshot.total_bytes,
        "index_tree_sha256": snapshot.tree_sha256,
        "official_hipporag_commit": OFFICIAL_HIPPORAG_COMMIT,
        "runtime_attestation_receipt_sha256": (
            expected_runtime_attestation_receipt_sha256
        ),
        "runtime_trust_root": RUNTIME_TRUST_ROOT,
        "schema": BUILD_RECEIPT_SCHEMA,
        "serialization": SERIALIZATION,
        "status": "passed_build_once",
    }
    if payload != expected:
        raise HybridQAOfficialHippoRAGError("build receipt contract drifted")
    payload["receipt_sha256"] = self_hash
    return payload


def make_retrieval_receipt(
    *,
    documents: Sequence[str],
    queries: Sequence[str],
    indices: Sequence[Sequence[int]],
    batch_sizes: Sequence[int],
    build_receipt: Mapping[str, object],
    index_snapshot_before: IndexTreeSnapshot,
    index_snapshot_after: IndexTreeSnapshot,
) -> dict[str, Any]:
    validated_queries = validate_queries(queries)
    before = _validate_index_snapshot(
        index_snapshot_before, "retrieval index snapshot before"
    )
    after = _validate_index_snapshot(
        index_snapshot_after, "retrieval index snapshot after"
    )
    build_self_hash = build_receipt.get("receipt_sha256")
    runtime_receipt_hash = build_receipt.get(
        "runtime_attestation_receipt_sha256"
    )
    if (
        not isinstance(build_self_hash, str)
        or _SHA256_RE.fullmatch(build_self_hash) is None
        or not isinstance(runtime_receipt_hash, str)
        or _SHA256_RE.fullmatch(runtime_receipt_hash) is None
        or build_receipt.get("index_tree_sha256") != before.tree_sha256
        or build_receipt.get("index_file_count") != before.file_count
        or build_receipt.get("index_total_bytes") != before.total_bytes
        or build_receipt.get("corpus_count") != CORPUS_SIZE
        or build_receipt.get("corpus_sha256") != corpus_sha256(documents)
    ):
        raise HybridQAOfficialHippoRAGError(
            "retrieval is not bound to the validated build receipt"
        )
    canonical_indices = [list(row) for row in indices]
    _validate_indices(canonical_indices, query_count=len(validated_queries))
    if (
        not isinstance(batch_sizes, Sequence)
        or isinstance(batch_sizes, (str, bytes))
        or not batch_sizes
        or any(
            isinstance(size, bool)
            or not isinstance(size, int)
            or not 1 <= size <= MAX_QUERY_BATCH
            for size in batch_sizes
        )
        or sum(batch_sizes) != len(validated_queries)
    ):
        raise HybridQAOfficialHippoRAGError("retrieval batch sizes are invalid")
    multiplicity = corpus_text_multiplicity(documents)
    if any(build_receipt.get(key) != value for key, value in multiplicity.items()):
        raise HybridQAOfficialHippoRAGError(
            "retrieval duplicate-text aggregate is not bound to the corpus"
        )
    if build_receipt.get("duplicate_expansion_policy") != DUPLICATE_EXPANSION_POLICY:
        raise HybridQAOfficialHippoRAGError(
            "retrieval duplicate expansion policy drifted"
        )
    receipt: dict[str, Any] = {
        "adapter_version": ADAPTER_VERSION,
        "batch_sizes": list(batch_sizes),
        "build_receipt_sha256": build_self_hash,
        "corpus_count": CORPUS_SIZE,
        "corpus_sha256": corpus_sha256(documents),
        "duplicate_expansion_policy": DUPLICATE_EXPANSION_POLICY,
        **multiplicity,
        "force_index_from_scratch": False,
        "index_call_count": 0,
        "index_changed_during_retrieve": before != after,
        "index_file_count": before.file_count,
        "index_post_file_count": after.file_count,
        "index_post_total_bytes": after.total_bytes,
        "index_post_tree_sha256": after.tree_sha256,
        "index_total_bytes": before.total_bytes,
        "index_tree_sha256": before.tree_sha256,
        "official_hipporag_commit": OFFICIAL_HIPPORAG_COMMIT,
        "query_sha256": _stable_hash(list(validated_queries)),
        "query_count": len(canonical_indices),
        "result_idx_sha256": _stable_hash(canonical_indices),
        "retrieval_call_count": len(batch_sizes),
        "runtime_attestation_receipt_sha256": runtime_receipt_hash,
        "runtime_trust_root": RUNTIME_TRUST_ROOT,
        "schema": RETRIEVAL_RECEIPT_SCHEMA,
        "serialization": SERIALIZATION,
        "status": "passed_reopen_retrieve_only",
    }
    receipt["receipt_sha256"] = _stable_hash(receipt)
    return receipt


def _validate_indices(value: object, *, query_count: int) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list) or len(value) != query_count:
        raise HybridQAOfficialHippoRAGError("retrieval index row count mismatch")
    rows: list[tuple[int, ...]] = []
    for raw in value:
        if not isinstance(raw, list) or len(raw) != TOP_K:
            raise HybridQAOfficialHippoRAGError(
                "each retrieval output must contain exactly five indices"
            )
        row: list[int] = []
        for idx in raw:
            if (
                isinstance(idx, bool)
                or not isinstance(idx, int)
                or not 0 <= idx < CORPUS_SIZE
            ):
                raise HybridQAOfficialHippoRAGError(
                    "retrieved article index is outside the global corpus"
                )
            row.append(idx)
        if len(set(row)) != TOP_K:
            raise HybridQAOfficialHippoRAGError(
                "retrieval output contains duplicate article indices"
            )
        rows.append(tuple(row))
    return tuple(rows)


def parse_retrieval_output(
    raw: bytes,
    *,
    queries: Sequence[str],
    expected_build_receipt: Mapping[str, object],
    expected_index_snapshot_after: IndexTreeSnapshot,
) -> RetrievalBatch:
    """Parse an entire worker output, rejecting any document/score side channel."""

    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HybridQAOfficialHippoRAGError(
            "retrieval worker output is invalid JSON"
        ) from exc
    if raw != canonical_json_bytes(value):
        raise HybridQAOfficialHippoRAGError(
            "retrieval worker output is not canonical JSON"
        )
    if not isinstance(value, dict) or set(value) != {
        "receipt",
        "retrieved_idx",
        "schema",
    }:
        raise HybridQAOfficialHippoRAGError(
            "retrieval worker may output only indices and a receipt"
        )
    if value.get("schema") != RETRIEVAL_OUTPUT_SCHEMA:
        raise HybridQAOfficialHippoRAGError("retrieval output schema mismatch")
    validated_queries = validate_queries(queries)
    query_count = len(validated_queries)
    post_snapshot = _validate_index_snapshot(
        expected_index_snapshot_after, "expected retrieval index snapshot after"
    )
    before_snapshot = IndexTreeSnapshot(
        file_count=expected_build_receipt.get("index_file_count"),  # type: ignore[arg-type]
        total_bytes=expected_build_receipt.get("index_total_bytes"),  # type: ignore[arg-type]
        tree_sha256=expected_build_receipt.get("index_tree_sha256"),  # type: ignore[arg-type]
    )
    before_snapshot = _validate_index_snapshot(
        before_snapshot, "expected build index snapshot"
    )
    indices = _validate_indices(value.get("retrieved_idx"), query_count=query_count)
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping) or set(receipt) != _RETRIEVAL_RECEIPT_KEYS:
        raise HybridQAOfficialHippoRAGError("retrieval receipt key set mismatch")
    normalized = dict(receipt)
    self_hash = normalized.pop("receipt_sha256", None)
    if not isinstance(self_hash, str) or _SHA256_RE.fullmatch(self_hash) is None:
        raise HybridQAOfficialHippoRAGError(
            "retrieval receipt self-hash is malformed"
        )
    if self_hash != _stable_hash(normalized):
        raise HybridQAOfficialHippoRAGError("retrieval receipt self-hash mismatch")
    batch_sizes = normalized.get("batch_sizes")
    if (
        not isinstance(batch_sizes, list)
        or not batch_sizes
        or any(
            isinstance(size, bool)
            or not isinstance(size, int)
            or not 1 <= size <= MAX_QUERY_BATCH
            for size in batch_sizes
        )
        or sum(batch_sizes) != query_count
    ):
        raise HybridQAOfficialHippoRAGError("retrieval batch receipt is invalid")
    expected = {
        "adapter_version": ADAPTER_VERSION,
        "batch_sizes": batch_sizes,
        "build_receipt_sha256": expected_build_receipt.get("receipt_sha256"),
        "corpus_count": CORPUS_SIZE,
        "corpus_sha256": expected_build_receipt.get("corpus_sha256"),
        "duplicate_expansion_policy": expected_build_receipt.get(
            "duplicate_expansion_policy"
        ),
        "duplicate_text_group_count": expected_build_receipt.get(
            "duplicate_text_group_count"
        ),
        "duplicate_text_unit_count": expected_build_receipt.get(
            "duplicate_text_unit_count"
        ),
        "force_index_from_scratch": False,
        "index_call_count": 0,
        "index_changed_during_retrieve": before_snapshot != post_snapshot,
        "index_file_count": expected_build_receipt.get("index_file_count"),
        "index_post_file_count": post_snapshot.file_count,
        "index_post_total_bytes": post_snapshot.total_bytes,
        "index_post_tree_sha256": post_snapshot.tree_sha256,
        "index_total_bytes": expected_build_receipt.get("index_total_bytes"),
        "index_tree_sha256": expected_build_receipt.get("index_tree_sha256"),
        "official_hipporag_commit": OFFICIAL_HIPPORAG_COMMIT,
        "official_unique_text_count": expected_build_receipt.get(
            "official_unique_text_count"
        ),
        "query_sha256": _stable_hash(list(validated_queries)),
        "query_count": query_count,
        "result_idx_sha256": _stable_hash([list(row) for row in indices]),
        "retrieval_call_count": len(batch_sizes),
        "runtime_attestation_receipt_sha256": expected_build_receipt.get(
            "runtime_attestation_receipt_sha256"
        ),
        "runtime_trust_root": RUNTIME_TRUST_ROOT,
        "schema": RETRIEVAL_RECEIPT_SCHEMA,
        "serialization": SERIALIZATION,
        "status": "passed_reopen_retrieve_only",
    }
    if (
        normalized != expected
        or not isinstance(normalized.get("corpus_sha256"), str)
        or _SHA256_RE.fullmatch(normalized["corpus_sha256"]) is None
        or not isinstance(normalized.get("index_changed_during_retrieve"), bool)
        or isinstance(normalized.get("index_post_file_count"), bool)
        or not isinstance(normalized.get("index_post_file_count"), int)
        or normalized["index_post_file_count"] < 1
        or isinstance(normalized.get("index_post_total_bytes"), bool)
        or not isinstance(normalized.get("index_post_total_bytes"), int)
        or normalized["index_post_total_bytes"] < 0
        or not isinstance(normalized.get("index_post_tree_sha256"), str)
        or _SHA256_RE.fullmatch(normalized["index_post_tree_sha256"]) is None
    ):
        raise HybridQAOfficialHippoRAGError("retrieval receipt contract drifted")
    normalized["receipt_sha256"] = self_hash
    return RetrievalBatch(indices=indices, receipt=normalized)


__all__ = [
    "ADAPTER_VERSION",
    "CORPUS_SIZE",
    "DUPLICATE_EXPANSION_POLICY",
    "FROZEN_CORE_CONFIG",
    "MAX_QUERY_BATCH",
    "HybridQAOfficialHippoRAGError",
    "RetrievalBatch",
    "canonical_json_bytes",
    "corpus_sha256",
    "corpus_text_multiplicity",
    "make_build_receipt",
    "make_retrieval_receipt",
    "parse_retrieval_output",
    "serialize_article",
    "serialize_corpus",
    "stable_top_five_from_official_result",
    "validate_build_receipt",
    "validate_corpus",
    "validate_queries",
]
