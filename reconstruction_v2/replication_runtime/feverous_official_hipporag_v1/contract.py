"""Frozen, label-free contract for one global FEVEROUS HippoRAG index.

The adapter accepts the frozen 8,192-unit FEVEROUS corpus and query text.  It has
no fields for labels, answers, gold evidence, relation families, or scores.  The
only retrieval payload allowed to leave the worker is a five-element unit-index
list for each query plus a content-free, FEVEROUS-specific execution receipt.
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


ADAPTER_VERSION = "feverous_official_hipporag_global_retrieve_only_v1"
BENCHMARK = "FEVEROUS"
OFFICIAL_HIPPORAG_COMMIT = "ef2f14c4f254f11ac29f9395f262466ad1bb4d10"
RUNTIME_TRUST_ROOT = "musique_official_hipporag_filesystem_attestation_v3"

TOP_K = 5
CORPUS_SIZE = 8192
MIN_CORPUS_SIZE = CORPUS_SIZE
MAX_CORPUS_SIZE = CORPUS_SIZE
MAX_QUERY_BATCH = 8
# The frozen formal cohort contains at most 288 queries.  Qualification tooling
# may use the same label-free runtime for a predeclared expansion of at most
# 4,096 queries; neither bound changes the exact 8,192-unit corpus contract.
FORMAL_QUERY_COUNT_UPPER_BOUND = 288
MAX_QUERY_COUNT = 4096
MAX_TEXT_CHARACTERS = 1_000_000

UNIT_KEYS = frozenset({"idx", "text"})
CORPUS_INPUT_SCHEMA = "feverous_official_hipporag_corpus_input_v1"
QUERY_INPUT_SCHEMA = "feverous_official_hipporag_query_input_v1"
BUILD_RECEIPT_SCHEMA = "feverous_official_hipporag_global_build_receipt_v1"
RETRIEVAL_RECEIPT_SCHEMA = (
    "feverous_official_hipporag_global_retrieval_receipt_v1"
)
RETRIEVAL_OUTPUT_SCHEMA = "feverous_official_hipporag_idx_receipt_output_v1"
SERIALIZATION = "exact_linearized_text_utf8_v1"
DUPLICATE_EXPANSION_POLICY = (
    "official_content_hash_dedup_then_equal_score_expand_to_all_unit_idx_v1"
)
TRANSPORT = "systemd_run_user_transient_service_v1"
SYSTEMD_NETWORK_PROPERTIES = (
    "IPAddressDeny=any",
    "RestrictAddressFamilies=AF_UNIX",
)
# The user service manager may hold arbitrary session variables.  The adapter
# launches the worker through ``env --ignore-environment`` and supplies exactly
# this allowlist; the DBus/XDG variables needed to contact the manager belong
# only to the outer launcher and are deliberately absent here.
WORKER_ENVIRONMENT_KEYS = frozenset(
    {
        "PATH",
        "HOME",
        "LANG",
        "HF_HOME",
        "TMPDIR",
        "TMP",
        "TEMP",
        "PYTHONPATH",
        "PYTHONNOUSERSITE",
        "PYTHONDONTWRITEBYTECODE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
    }
)
WORKER_FIXED_ENVIRONMENT_VALUES = {
    "LANG": "C.UTF-8",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

FROZEN_CORE_CONFIG: dict[str, Any] = {
    "config_class": "hipporag.utils.config_utils.BaseConfig",
    "core_class": "hipporag.HippoRAG",
    "llm_backend": "Transformers/local_asset",
    "embedding_backend": "Transformers/local_asset",
    "openie_mode": "online",
    "max_new_tokens": 4,
    "corpus_count": CORPUS_SIZE,
    "retrieval_top_k": CORPUS_SIZE,
    "qa_top_k": TOP_K,
    "save_openie": True,
    "build_force_index_from_scratch": True,
    "reopen_force_index_from_scratch": False,
    "official_retrieve_num_to_retrieve": CORPUS_SIZE,
    "official_content_addressing": "exact_text_hash_deduplicated_v1",
    "logical_duplicate_expansion": DUPLICATE_EXPANSION_POLICY,
    "formal_query_count_upper_bound": FORMAL_QUERY_COUNT_UPPER_BOUND,
    "absolute_query_count_upper_bound": MAX_QUERY_COUNT,
    "query_batch_size_upper_bound": MAX_QUERY_BATCH,
    "query_concurrency_upper_bound": MAX_QUERY_BATCH,
    "adapter_top_k_selection": "negative_official_score_then_unit_idx_v1",
    "transport": TRANSPORT,
    "network_properties": SYSTEMD_NETWORK_PROPERTIES,
    "cuda_visibility_override": None,
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_BUILD_RECEIPT_KEYS = frozenset(
    {
        "adapter_version",
        "benchmark",
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
        "benchmark",
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


class FeverousOfficialHippoRAGError(RuntimeError):
    """Raised when the global retrieve-only contract cannot be proven."""


@dataclass(frozen=True)
class CorpusUnit:
    idx: int
    text: str


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
        raise FeverousOfficialHippoRAGError(
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
                raise FeverousOfficialHippoRAGError(
                    "persistent official index contains a symbolic link"
                )
            if stat.S_ISDIR(metadata.st_mode):
                rows.append({"kind": "directory", "mode": mode, "path": relative})
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise FeverousOfficialHippoRAGError(
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
        raise FeverousOfficialHippoRAGError(
            "persistent official index cannot be snapshotted"
        ) from exc
    if file_count == 0:
        raise FeverousOfficialHippoRAGError(
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
        raise FeverousOfficialHippoRAGError(f"{field} is invalid")
    return snapshot


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FeverousOfficialHippoRAGError(f"{field} must be non-empty text")
    if "\x00" in value:
        raise FeverousOfficialHippoRAGError(f"{field} contains a NUL character")
    if len(value) > MAX_TEXT_CHARACTERS:
        raise FeverousOfficialHippoRAGError(
            f"{field} exceeds the frozen character bound"
        )
    return value


def validate_corpus(
    units: Sequence[Mapping[str, object]],
) -> tuple[CorpusUnit, ...]:
    """Validate the exact global corpus without accepting benchmark labels."""

    if isinstance(units, (str, bytes)) or not isinstance(units, Sequence):
        raise FeverousOfficialHippoRAGError("units must be a sequence")
    if not MIN_CORPUS_SIZE <= len(units) <= MAX_CORPUS_SIZE:
        raise FeverousOfficialHippoRAGError(
            "FEVEROUS global corpus must contain exactly 8192 units"
        )
    rows: list[CorpusUnit] = []
    for position, raw in enumerate(units):
        if not isinstance(raw, Mapping) or set(raw) != UNIT_KEYS:
            raise FeverousOfficialHippoRAGError(
                "unit must contain only idx and exact linearized text"
            )
        idx = raw.get("idx")
        if isinstance(idx, bool) or not isinstance(idx, int) or idx != position:
            raise FeverousOfficialHippoRAGError(
                "unit idx must be canonical contiguous zero-based order"
            )
        rows.append(
            CorpusUnit(
                idx=idx,
                text=_required_text(raw.get("text"), f"units[{position}].text"),
            )
        )
    return tuple(rows)


def serialize_unit(unit: CorpusUnit) -> str:
    """Return the byte-identical FEVEROUS ``linearized_text`` string."""

    return unit.text


def serialize_corpus(units: Sequence[CorpusUnit]) -> tuple[str, ...]:
    return tuple(serialize_unit(unit) for unit in units)


def corpus_sha256(documents: Sequence[str]) -> str:
    if (
        not MIN_CORPUS_SIZE <= len(documents) <= MAX_CORPUS_SIZE
        or any(not isinstance(document, str) or not document for document in documents)
    ):
        raise FeverousOfficialHippoRAGError("serialized global corpus is invalid")
    return _stable_hash(list(documents))


def corpus_text_multiplicity(documents: Sequence[str]) -> dict[str, int]:
    """Return content-free duplicate aggregates for the logical corpus.

    The pinned official embedding store keys passages by a content hash, so
    byte-identical strings are indexed once.  They nevertheless remain
    distinct FEVEROUS atomic units.  The adapter therefore expands the one
    official score for a repeated string back to every logical unit index.
    """

    corpus_sha256(documents)
    counts: dict[str, int] = {}
    content_hash_to_text: dict[str, str] = {}
    for document in documents:
        content_hash = hashlib.md5(document.encode("utf-8")).hexdigest()
        prior = content_hash_to_text.setdefault(content_hash, document)
        if prior != document:
            raise FeverousOfficialHippoRAGError(
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
        raise FeverousOfficialHippoRAGError("queries must be a sequence")
    if not 1 <= len(queries) <= MAX_QUERY_COUNT:
        raise FeverousOfficialHippoRAGError("query count is outside the frozen bound")
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
    """Expand all official unique-text scores to a stable logical top five."""

    if isinstance(retrieved_documents, (str, bytes)) or isinstance(
        retrieved_scores, (str, bytes)
    ):
        raise FeverousOfficialHippoRAGError("official result rows are malformed")
    try:
        documents = list(retrieved_documents)
        scores = list(retrieved_scores)
    except TypeError as exc:
        raise FeverousOfficialHippoRAGError(
            "official result rows are not iterable"
        ) from exc
    canonical_mapping: dict[str, tuple[int, ...]] = {}
    all_indices: list[int] = []
    for document, raw_indices in document_to_indices.items():
        if not isinstance(document, str) or not document:
            raise FeverousOfficialHippoRAGError(
                "document-to-indices mapping contains an invalid document"
            )
        if isinstance(raw_indices, (str, bytes)):
            raise FeverousOfficialHippoRAGError(
                "document-to-indices mapping contains malformed indices"
            )
        try:
            indices = tuple(raw_indices)
        except TypeError as exc:
            raise FeverousOfficialHippoRAGError(
                "document-to-indices mapping contains malformed indices"
            ) from exc
        if not indices or any(type(idx) is not int for idx in indices):
            raise FeverousOfficialHippoRAGError(
                "document-to-indices mapping contains malformed indices"
            )
        canonical_mapping[document] = indices
        all_indices.extend(indices)
    corpus_count = len(all_indices)
    if (
        not MIN_CORPUS_SIZE <= corpus_count <= MAX_CORPUS_SIZE
        or sorted(all_indices) != list(range(corpus_count))
        or len(documents) != len(canonical_mapping)
        or len(scores) != len(canonical_mapping)
    ):
        raise FeverousOfficialHippoRAGError(
            "official retrieve must return every unique global corpus text exactly once"
        )

    ranked: list[tuple[float, int]] = []
    seen: set[str] = set()
    for document, score in zip(documents, scores):
        if not isinstance(document, str) or document not in canonical_mapping:
            raise FeverousOfficialHippoRAGError(
                "official result contains an unknown corpus document"
            )
        if document in seen:
            raise FeverousOfficialHippoRAGError(
                "official result contains a duplicate corpus document"
            )
        seen.add(document)
        if isinstance(score, bool) or not isinstance(score, Real):
            raise FeverousOfficialHippoRAGError(
                "official result score is not numeric"
            )
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise FeverousOfficialHippoRAGError(
                "official result score is not finite"
            )
        ranked.extend(
            (numeric_score, idx) for idx in canonical_mapping[document]
        )
    if seen != set(canonical_mapping):
        raise FeverousOfficialHippoRAGError(
            "official result omitted a unique global corpus text"
        )
    ranked.sort(key=lambda row: (-row[0], row[1]))
    result = tuple(idx for _score, idx in ranked[:TOP_K])
    if len(result) != TOP_K or len(set(result)) != TOP_K:
        raise FeverousOfficialHippoRAGError(
            "adapter did not produce five unique unit indices"
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
        raise FeverousOfficialHippoRAGError(
            "runtime attestation receipt hash is malformed"
        )
    multiplicity = corpus_text_multiplicity(documents)
    receipt: dict[str, Any] = {
        "adapter_version": ADAPTER_VERSION,
        "benchmark": BENCHMARK,
        "corpus_count": len(documents),
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
        raise FeverousOfficialHippoRAGError("build receipt key set mismatch")
    payload = dict(receipt)
    self_hash = payload.pop("receipt_sha256", None)
    if not isinstance(self_hash, str) or _SHA256_RE.fullmatch(self_hash) is None:
        raise FeverousOfficialHippoRAGError("build receipt self-hash is malformed")
    if self_hash != _stable_hash(payload):
        raise FeverousOfficialHippoRAGError("build receipt self-hash mismatch")
    snapshot = _validate_index_snapshot(
        expected_index_snapshot, "expected build index snapshot"
    )
    if (
        not isinstance(expected_runtime_attestation_receipt_sha256, str)
        or _SHA256_RE.fullmatch(expected_runtime_attestation_receipt_sha256) is None
    ):
        raise FeverousOfficialHippoRAGError(
            "expected runtime attestation receipt hash is malformed"
        )
    documents = tuple(expected_documents)
    expected_corpus_count = len(documents)
    expected_corpus_sha256 = corpus_sha256(documents)
    multiplicity = corpus_text_multiplicity(documents)
    expected = {
        "adapter_version": ADAPTER_VERSION,
        "benchmark": BENCHMARK,
        "corpus_count": expected_corpus_count,
        "corpus_sha256": expected_corpus_sha256,
        "duplicate_expansion_policy": DUPLICATE_EXPANSION_POLICY,
        **multiplicity,
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
        raise FeverousOfficialHippoRAGError("build receipt contract drifted")
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
        or build_receipt.get("corpus_count") != len(documents)
        or build_receipt.get("corpus_sha256") != corpus_sha256(documents)
    ):
        raise FeverousOfficialHippoRAGError(
            "retrieval is not bound to the validated build receipt"
        )
    canonical_indices = [list(row) for row in indices]
    _validate_indices(
        canonical_indices,
        query_count=len(validated_queries),
        corpus_count=len(documents),
    )
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
        raise FeverousOfficialHippoRAGError("retrieval batch sizes are invalid")
    multiplicity = corpus_text_multiplicity(documents)
    if any(build_receipt.get(key) != value for key, value in multiplicity.items()):
        raise FeverousOfficialHippoRAGError(
            "retrieval duplicate-text aggregate is not bound to the corpus"
        )
    if build_receipt.get("duplicate_expansion_policy") != DUPLICATE_EXPANSION_POLICY:
        raise FeverousOfficialHippoRAGError(
            "retrieval duplicate expansion policy drifted"
        )
    receipt: dict[str, Any] = {
        "adapter_version": ADAPTER_VERSION,
        "benchmark": BENCHMARK,
        "batch_sizes": list(batch_sizes),
        "build_receipt_sha256": build_self_hash,
        "corpus_count": len(documents),
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


def _validate_indices(
    value: object, *, query_count: int, corpus_count: int
) -> tuple[tuple[int, ...], ...]:
    if (
        isinstance(corpus_count, bool)
        or not isinstance(corpus_count, int)
        or not MIN_CORPUS_SIZE <= corpus_count <= MAX_CORPUS_SIZE
    ):
        raise FeverousOfficialHippoRAGError("retrieval corpus count is invalid")
    if not isinstance(value, list) or len(value) != query_count:
        raise FeverousOfficialHippoRAGError("retrieval index row count mismatch")
    rows: list[tuple[int, ...]] = []
    for raw in value:
        if not isinstance(raw, list) or len(raw) != TOP_K:
            raise FeverousOfficialHippoRAGError(
                "each retrieval output must contain exactly five indices"
            )
        row: list[int] = []
        for idx in raw:
            if (
                isinstance(idx, bool)
                or not isinstance(idx, int)
                or not 0 <= idx < corpus_count
            ):
                raise FeverousOfficialHippoRAGError(
                    "retrieved unit index is outside the global corpus"
                )
            row.append(idx)
        if len(set(row)) != TOP_K:
            raise FeverousOfficialHippoRAGError(
                "retrieval output contains duplicate unit indices"
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
        raise FeverousOfficialHippoRAGError(
            "retrieval worker output is invalid JSON"
        ) from exc
    if raw != canonical_json_bytes(value):
        raise FeverousOfficialHippoRAGError(
            "retrieval worker output is not canonical JSON"
        )
    if not isinstance(value, dict) or set(value) != {
        "receipt",
        "retrieved_idx",
        "schema",
    }:
        raise FeverousOfficialHippoRAGError(
            "retrieval worker may output only indices and a receipt"
        )
    if value.get("schema") != RETRIEVAL_OUTPUT_SCHEMA:
        raise FeverousOfficialHippoRAGError("retrieval output schema mismatch")
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
    corpus_count = expected_build_receipt.get("corpus_count")
    indices = _validate_indices(
        value.get("retrieved_idx"),
        query_count=query_count,
        corpus_count=corpus_count,  # type: ignore[arg-type]
    )
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping) or set(receipt) != _RETRIEVAL_RECEIPT_KEYS:
        raise FeverousOfficialHippoRAGError("retrieval receipt key set mismatch")
    normalized = dict(receipt)
    self_hash = normalized.pop("receipt_sha256", None)
    if not isinstance(self_hash, str) or _SHA256_RE.fullmatch(self_hash) is None:
        raise FeverousOfficialHippoRAGError(
            "retrieval receipt self-hash is malformed"
        )
    if self_hash != _stable_hash(normalized):
        raise FeverousOfficialHippoRAGError("retrieval receipt self-hash mismatch")
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
        raise FeverousOfficialHippoRAGError("retrieval batch receipt is invalid")
    expected = {
        "adapter_version": ADAPTER_VERSION,
        "benchmark": BENCHMARK,
        "batch_sizes": batch_sizes,
        "build_receipt_sha256": expected_build_receipt.get("receipt_sha256"),
        "corpus_count": corpus_count,
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
        raise FeverousOfficialHippoRAGError("retrieval receipt contract drifted")
    normalized["receipt_sha256"] = self_hash
    return RetrievalBatch(indices=indices, receipt=normalized)


__all__ = [
    "ADAPTER_VERSION",
    "BENCHMARK",
    "CORPUS_SIZE",
    "CorpusUnit",
    "DUPLICATE_EXPANSION_POLICY",
    "FROZEN_CORE_CONFIG",
    "FORMAL_QUERY_COUNT_UPPER_BOUND",
    "MAX_CORPUS_SIZE",
    "MAX_QUERY_BATCH",
    "MAX_QUERY_COUNT",
    "MIN_CORPUS_SIZE",
    "OFFICIAL_HIPPORAG_COMMIT",
    "RUNTIME_TRUST_ROOT",
    "TOP_K",
    "FeverousOfficialHippoRAGError",
    "IndexTreeSnapshot",
    "RetrievalBatch",
    "SERIALIZATION",
    "SYSTEMD_NETWORK_PROPERTIES",
    "TRANSPORT",
    "WORKER_ENVIRONMENT_KEYS",
    "WORKER_FIXED_ENVIRONMENT_VALUES",
    "canonical_json_bytes",
    "corpus_sha256",
    "corpus_text_multiplicity",
    "make_build_receipt",
    "make_retrieval_receipt",
    "parse_retrieval_output",
    "serialize_unit",
    "serialize_corpus",
    "snapshot_index_tree",
    "stable_top_five_from_official_result",
    "validate_build_receipt",
    "validate_corpus",
    "validate_queries",
]
