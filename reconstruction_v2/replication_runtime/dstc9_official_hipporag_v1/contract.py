"""Source-free contract for one global DSTC9 official HippoRAG index.

The only content-bearing inputs are an exact 2,900-row serialized corpus and
at most 256 exact query strings.  Corpus rows have only a contiguous ordinal
and text.  Query rows have only a contiguous ordinal, opaque work id, and
query text.  Domain/entity/document ids, families, qrels, answers, labels,
scores, and evaluator values have no input channel.

The only content-bearing retrieval result is five corpus ordinals per query.
Receipts contain counts, hashes, and fixed execution-policy values only.
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


ADAPTER_VERSION = "dstc9_official_hipporag_global_retrieve_only_v1"
BENCHMARK = "DSTC9_TRACK1"
OFFICIAL_HIPPORAG_COMMIT = "ef2f14c4f254f11ac29f9395f262466ad1bb4d10"
RUNTIME_TRUST_ROOT = (
    "dstc9_p17_reused_closure_plus_current_hardware_binding_v3"
)

TOP_K = 5
CORPUS_SIZE = 2900
MIN_CORPUS_SIZE = CORPUS_SIZE
MAX_CORPUS_SIZE = CORPUS_SIZE
MAX_QUERY_BATCH = 8
MAX_QUERY_COUNT = 256
FORMAL_QUERY_COUNT_UPPER_BOUND = MAX_QUERY_COUNT
MAX_TEXT_CHARACTERS = 1_000_000
MAX_WORK_ID_CHARACTERS = 512
MAX_STUDY_ID_CHARACTERS = 256

UNIT_KEYS = frozenset({"ordinal", "text"})
QUERY_KEYS = frozenset({"ordinal", "query_text", "work_id"})
CORPUS_INPUT_KEYS = frozenset({"schema", "self_sha256", "study_id", "units"})
QUERY_INPUT_KEYS = frozenset({"queries", "schema", "self_sha256", "study_id"})
CORPUS_INPUT_SCHEMA = "dstc9_official_hipporag_corpus_input_v1"
QUERY_INPUT_SCHEMA = "dstc9_official_hipporag_query_input_v1"
BUILD_RECEIPT_SCHEMA = "dstc9_official_hipporag_global_build_receipt_v1"
RETRIEVAL_RECEIPT_SCHEMA = (
    "dstc9_official_hipporag_global_retrieval_receipt_v1"
)
RETRIEVAL_OUTPUT_SCHEMA = "dstc9_official_hipporag_ordinal_receipt_output_v1"
SERIALIZATION = "exact_typed_serialized_text_utf8_v1"
QUERY_SERIALIZATION = "exact_query_text_utf8_v1"
DUPLICATE_EXPANSION_POLICY = (
    "official_content_hash_dedup_then_equal_score_expand_to_all_ordinals_v1"
)
FORMAL_QUERY_COUNT_POLICY = (
    "exact_count_must_be_bound_by_formal_freeze_before_source_access_v1"
)
TRANSPORT = "systemd_run_user_transient_service_v1"
SYSTEMD_NETWORK_PROPERTIES = (
    "IPAddressDeny=any",
    "RestrictAddressFamilies=AF_UNIX",
)
CUDA_VISIBLE_DEVICES = "0"
LOGICAL_CUDA_DEVICE = "cuda:0"

# The adapter executes ``env --ignore-environment`` inside the transient
# service and supplies exactly this allowlist.
WORKER_ENVIRONMENT_KEYS = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "HOME",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "LANG",
        "PATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPYCACHEPREFIX",
        "PYTHONNOUSERSITE",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TOKENIZERS_PARALLELISM",
        "TRANSFORMERS_OFFLINE",
    }
)
WORKER_FIXED_ENVIRONMENT_VALUES = {
    "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES,
    "HF_HUB_OFFLINE": "1",
    "LANG": "C.UTF-8",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
}

FROZEN_CORE_CONFIG: dict[str, Any] = {
    "absolute_query_count_upper_bound": MAX_QUERY_COUNT,
    "adapter_top_k_selection": "negative_official_score_then_ordinal_v1",
    "build_force_index_from_scratch": True,
    "core_class": "hipporag.HippoRAG",
    "corpus_count": CORPUS_SIZE,
    "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
    "dynamic_resize_allowed": False,
    "embedding_backend": "Transformers/local_asset",
    "formal_query_count_policy": FORMAL_QUERY_COUNT_POLICY,
    "formal_query_count_upper_bound": FORMAL_QUERY_COUNT_UPPER_BOUND,
    "llm_backend": "Transformers/local_asset",
    "logical_cuda_device": LOGICAL_CUDA_DEVICE,
    "logical_duplicate_expansion": DUPLICATE_EXPANSION_POLICY,
    "max_new_tokens": 4,
    "max_retry_attempts": 0,
    "network_access": "denied_by_transient_unit",
    "network_properties": SYSTEMD_NETWORK_PROPERTIES,
    "official_content_addressing": "exact_text_hash_deduplicated_v1",
    "official_retrieve_num_to_retrieve": CORPUS_SIZE,
    "openie_mode": "online",
    "qa_top_k": TOP_K,
    "query_batch_size_upper_bound": MAX_QUERY_BATCH,
    "query_concurrency_upper_bound": MAX_QUERY_BATCH,
    "reopen_force_index_from_scratch": False,
    "retrieval_top_k": CORPUS_SIZE,
    "save_openie": True,
    "transport": TRANSPORT,
}

FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "answer",
        "answer_id",
        "answers",
        "category",
        "doc",
        "doc_id",
        "document",
        "document_id",
        "domain",
        "domain_id",
        "entity",
        "entity_id",
        "evidence",
        "evaluator",
        "families",
        "family",
        "gold",
        "gold_document",
        "gold_id",
        "label",
        "labels",
        "metric",
        "qrel",
        "qrels",
        "relevance",
        "response",
        "score",
        "scores",
        "snippet_id",
        "split",
        "target",
        "utility",
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_OPAQUE_ID_RE = re.compile(r"[^\x00\r\n]{1,512}\Z")
_STUDY_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,255}\Z")
_BUILD_RECEIPT_KEYS = frozenset(
    {
        "adapter_version",
        "benchmark",
        "corpus_count",
        "corpus_input_self_sha256",
        "corpus_sha256",
        "cuda_visible_devices",
        "duplicate_expansion_policy",
        "duplicate_text_group_count",
        "duplicate_text_unit_count",
        "dynamic_resize_count",
        "force_index_from_scratch",
        "index_call_count",
        "index_file_count",
        "index_total_bytes",
        "index_tree_sha256",
        "logical_cuda_device",
        "network_access",
        "official_hipporag_commit",
        "official_unique_text_count",
        "query_count_policy",
        "receipt_sha256",
        "retry_count",
        "runtime_attestation_receipt_sha256",
        "runtime_trust_root",
        "schema",
        "serialization",
        "status",
        "study_id",
    }
)
_RETRIEVAL_RECEIPT_KEYS = frozenset(
    {
        "adapter_version",
        "batch_sizes",
        "benchmark",
        "build_receipt_sha256",
        "corpus_count",
        "corpus_input_self_sha256",
        "corpus_sha256",
        "cuda_visible_devices",
        "duplicate_expansion_policy",
        "duplicate_text_group_count",
        "duplicate_text_unit_count",
        "dynamic_resize_count",
        "force_index_from_scratch",
        "index_call_count",
        "index_changed_during_retrieve",
        "index_file_count",
        "index_post_file_count",
        "index_post_total_bytes",
        "index_post_tree_sha256",
        "index_total_bytes",
        "index_tree_sha256",
        "logical_cuda_device",
        "network_access",
        "official_hipporag_commit",
        "official_unique_text_count",
        "query_count",
        "query_count_policy",
        "query_input_self_sha256",
        "query_serialization",
        "query_sha256",
        "receipt_sha256",
        "result_ordinal_sha256",
        "retrieval_call_count",
        "retry_count",
        "runtime_attestation_receipt_sha256",
        "runtime_trust_root",
        "schema",
        "serialization",
        "status",
        "study_id",
    }
)


class Dstc9OfficialHippoRAGError(RuntimeError):
    """The source-free global retrieval contract could not be proven."""


@dataclass(frozen=True, slots=True)
class CorpusUnit:
    ordinal: int
    text: str


@dataclass(frozen=True, slots=True)
class QueryRow:
    ordinal: int
    work_id: str
    query_text: str


@dataclass(frozen=True, slots=True)
class CorpusInput:
    study_id: str
    units: tuple[CorpusUnit, ...]
    self_sha256: str


@dataclass(frozen=True, slots=True)
class QueryInput:
    study_id: str
    queries: tuple[QueryRow, ...]
    self_sha256: str


@dataclass(frozen=True)
class RetrievalBatch:
    """The complete public retrieval value: ordinals and a safe receipt."""

    indices: tuple[tuple[int, ...], ...]
    receipt: Mapping[str, Any]

    @property
    def ordinals(self) -> tuple[tuple[int, ...], ...]:
        return self.indices


@dataclass(frozen=True)
class IndexTreeSnapshot:
    file_count: int
    total_bytes: int
    tree_sha256: str


def canonical_json_bytes(value: object, *, newline: bool = True) -> bytes:
    """Encode exact canonical JSON used by private IPC and safe receipts."""

    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise Dstc9OfficialHippoRAGError("value is not canonical JSON") from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value, newline=False)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Dstc9OfficialHippoRAGError(f"{field} is not a lowercase SHA-256")
    return value


def _required_study_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) > MAX_STUDY_ID_CHARACTERS
        or _STUDY_ID_RE.fullmatch(value) is None
    ):
        raise Dstc9OfficialHippoRAGError("study_id is not an opaque study token")
    return value


def _required_text(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value) > MAX_TEXT_CHARACTERS
    ):
        raise Dstc9OfficialHippoRAGError(f"{field} is invalid")
    return value


def _required_work_id(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > MAX_WORK_ID_CHARACTERS
        or _OPAQUE_ID_RE.fullmatch(value) is None
        or not value.strip()
    ):
        raise Dstc9OfficialHippoRAGError(f"{field} is not an opaque work id")
    return value


def _reject_forbidden_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if (
                not isinstance(key, str)
                or key.casefold() in FORBIDDEN_INPUT_KEYS
            ):
                raise Dstc9OfficialHippoRAGError(
                    "source-free input contains a forbidden source/label field"
                )
            _reject_forbidden_keys(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            _reject_forbidden_keys(nested)


def _with_self_hash(body: Mapping[str, object]) -> dict[str, object]:
    if "self_sha256" in body:
        raise Dstc9OfficialHippoRAGError("self hash was supplied twice")
    payload = dict(body)
    payload["self_sha256"] = stable_hash(payload)
    return payload


def _verify_self_hash(value: Mapping[str, object], field: str) -> str:
    body = dict(value)
    claimed = _required_sha256(body.pop("self_sha256", None), f"{field} self hash")
    if stable_hash(body) != claimed:
        raise Dstc9OfficialHippoRAGError(f"{field} self hash drifted")
    return claimed


def validate_corpus(
    units: Sequence[Mapping[str, object]],
) -> tuple[CorpusUnit, ...]:
    """Validate exactly 2,900 ordinal/text rows and no other channel."""

    if isinstance(units, (str, bytes)) or not isinstance(units, Sequence):
        raise Dstc9OfficialHippoRAGError("units must be a sequence")
    if len(units) != CORPUS_SIZE:
        raise Dstc9OfficialHippoRAGError(
            "DSTC9 global corpus must contain exactly 2900 units"
        )
    rows: list[CorpusUnit] = []
    for position, raw in enumerate(units):
        if not isinstance(raw, Mapping) or set(raw) != UNIT_KEYS:
            raise Dstc9OfficialHippoRAGError(
                "unit must contain only ordinal and typed serialized text"
            )
        ordinal = raw.get("ordinal")
        if type(ordinal) is not int or ordinal != position:
            raise Dstc9OfficialHippoRAGError(
                "unit ordinals must be contiguous zero-based order"
            )
        rows.append(
            CorpusUnit(
                ordinal=ordinal,
                text=_required_text(raw.get("text"), f"units[{position}].text"),
            )
        )
    return tuple(rows)


def validate_queries(
    queries: Sequence[Mapping[str, object]],
) -> tuple[QueryRow, ...]:
    """Validate at most 256 ordinal/work-id/query-text rows."""

    if isinstance(queries, (str, bytes)) or not isinstance(queries, Sequence):
        raise Dstc9OfficialHippoRAGError("queries must be a sequence")
    if not 1 <= len(queries) <= MAX_QUERY_COUNT:
        raise Dstc9OfficialHippoRAGError(
            "query count is outside the frozen 1..256 bound"
        )
    rows: list[QueryRow] = []
    seen_work_ids: set[str] = set()
    for position, raw in enumerate(queries):
        if not isinstance(raw, Mapping) or set(raw) != QUERY_KEYS:
            raise Dstc9OfficialHippoRAGError(
                "query must contain only ordinal, opaque work_id, and query_text"
            )
        ordinal = raw.get("ordinal")
        if type(ordinal) is not int or ordinal != position:
            raise Dstc9OfficialHippoRAGError(
                "query ordinals must be contiguous zero-based order"
            )
        work_id = _required_work_id(
            raw.get("work_id"), f"queries[{position}].work_id"
        )
        if work_id in seen_work_ids:
            raise Dstc9OfficialHippoRAGError("query work_id values must be unique")
        seen_work_ids.add(work_id)
        rows.append(
            QueryRow(
                ordinal=ordinal,
                work_id=work_id,
                query_text=_required_text(
                    raw.get("query_text"), f"queries[{position}].query_text"
                ),
            )
        )
    return tuple(rows)


def make_corpus_input(
    *, study_id: str, units: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    rows = validate_corpus(units)
    return _with_self_hash(
        {
            "schema": CORPUS_INPUT_SCHEMA,
            "study_id": _required_study_id(study_id),
            "units": [
                {"ordinal": row.ordinal, "text": row.text} for row in rows
            ],
        }
    )


def validate_corpus_input(value: object) -> CorpusInput:
    _reject_forbidden_keys(value)
    if not isinstance(value, Mapping) or set(value) != CORPUS_INPUT_KEYS:
        raise Dstc9OfficialHippoRAGError("corpus input schema drifted")
    if value.get("schema") != CORPUS_INPUT_SCHEMA:
        raise Dstc9OfficialHippoRAGError("corpus input schema identifier drifted")
    study_id = _required_study_id(value.get("study_id"))
    raw_units = value.get("units")
    if not isinstance(raw_units, list):
        raise Dstc9OfficialHippoRAGError("corpus input units are malformed")
    units = validate_corpus(raw_units)
    self_sha256 = _verify_self_hash(value, "corpus input")
    return CorpusInput(study_id=study_id, units=units, self_sha256=self_sha256)


def make_query_input(
    *, study_id: str, queries: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    rows = validate_queries(queries)
    return _with_self_hash(
        {
            "queries": [
                {
                    "ordinal": row.ordinal,
                    "query_text": row.query_text,
                    "work_id": row.work_id,
                }
                for row in rows
            ],
            "schema": QUERY_INPUT_SCHEMA,
            "study_id": _required_study_id(study_id),
        }
    )


def validate_query_input(
    value: object, *, expected_study_id: str | None = None
) -> QueryInput:
    _reject_forbidden_keys(value)
    if not isinstance(value, Mapping) or set(value) != QUERY_INPUT_KEYS:
        raise Dstc9OfficialHippoRAGError("query input schema drifted")
    if value.get("schema") != QUERY_INPUT_SCHEMA:
        raise Dstc9OfficialHippoRAGError("query input schema identifier drifted")
    study_id = _required_study_id(value.get("study_id"))
    if expected_study_id is not None and study_id != _required_study_id(
        expected_study_id
    ):
        raise Dstc9OfficialHippoRAGError(
            "query input is bound to a different study"
        )
    raw_queries = value.get("queries")
    if not isinstance(raw_queries, list):
        raise Dstc9OfficialHippoRAGError("query input rows are malformed")
    queries = validate_queries(raw_queries)
    self_sha256 = _verify_self_hash(value, "query input")
    return QueryInput(
        study_id=study_id,
        queries=queries,
        self_sha256=self_sha256,
    )


def corpus_input_projection(value: CorpusInput) -> dict[str, object]:
    if not isinstance(value, CorpusInput):
        raise Dstc9OfficialHippoRAGError("validated corpus input is invalid")
    return {
        "schema": CORPUS_INPUT_SCHEMA,
        "self_sha256": value.self_sha256,
        "study_id": value.study_id,
        "units": [
            {"ordinal": row.ordinal, "text": row.text} for row in value.units
        ],
    }


def query_input_projection(value: QueryInput) -> dict[str, object]:
    if not isinstance(value, QueryInput):
        raise Dstc9OfficialHippoRAGError("validated query input is invalid")
    return {
        "queries": [
            {
                "ordinal": row.ordinal,
                "query_text": row.query_text,
                "work_id": row.work_id,
            }
            for row in value.queries
        ],
        "schema": QUERY_INPUT_SCHEMA,
        "self_sha256": value.self_sha256,
        "study_id": value.study_id,
    }


def serialize_unit(unit: CorpusUnit) -> str:
    if not isinstance(unit, CorpusUnit):
        raise Dstc9OfficialHippoRAGError("corpus unit is invalid")
    return unit.text


def serialize_corpus(units: Sequence[CorpusUnit]) -> tuple[str, ...]:
    return tuple(serialize_unit(unit) for unit in units)


def serialize_queries(queries: Sequence[QueryRow]) -> tuple[str, ...]:
    return tuple(row.query_text for row in queries)


def corpus_sha256(documents: Sequence[str]) -> str:
    if (
        len(documents) != CORPUS_SIZE
        or any(not isinstance(document, str) or not document for document in documents)
    ):
        raise Dstc9OfficialHippoRAGError("serialized global corpus is invalid")
    return stable_hash(list(documents))


def query_sha256(queries: Sequence[QueryRow]) -> str:
    if not 1 <= len(queries) <= MAX_QUERY_COUNT:
        raise Dstc9OfficialHippoRAGError("validated query rows are invalid")
    return stable_hash(
        [
            {
                "ordinal": row.ordinal,
                "query_text": row.query_text,
                "work_id": row.work_id,
            }
            for row in queries
        ]
    )


def corpus_text_multiplicity(documents: Sequence[str]) -> dict[str, int]:
    """Check official MD5 addressing and return content-free aggregates."""

    corpus_sha256(documents)
    counts: dict[str, int] = {}
    hash_to_text: dict[str, str] = {}
    for document in documents:
        content_hash = hashlib.md5(document.encode("utf-8")).hexdigest()
        prior = hash_to_text.setdefault(content_hash, document)
        if prior != document:
            raise Dstc9OfficialHippoRAGError(
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


def snapshot_index_tree(index_root: Path) -> IndexTreeSnapshot:
    """Content-completely bind a regular-file-only persisted index tree."""

    root = index_root.absolute()
    if root.is_symlink() or not root.is_dir():
        raise Dstc9OfficialHippoRAGError(
            "persistent official index root is unavailable"
        )
    rows: list[dict[str, object]] = []
    file_count = 0
    total_bytes = 0
    try:
        entries = sorted(
            root.rglob("*"), key=lambda row: row.relative_to(root).as_posix()
        )
        for entry in entries:
            relative = entry.relative_to(root).as_posix()
            metadata = entry.lstat()
            mode = stat.S_IMODE(metadata.st_mode)
            if stat.S_ISLNK(metadata.st_mode):
                raise Dstc9OfficialHippoRAGError(
                    "persistent official index contains a symbolic link"
                )
            if stat.S_ISDIR(metadata.st_mode):
                rows.append({"kind": "directory", "mode": mode, "path": relative})
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise Dstc9OfficialHippoRAGError(
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
        raise Dstc9OfficialHippoRAGError(
            "persistent official index cannot be snapshotted"
        ) from exc
    if file_count == 0:
        raise Dstc9OfficialHippoRAGError(
            "persistent official index contains no regular files"
        )
    return IndexTreeSnapshot(
        file_count=file_count,
        total_bytes=total_bytes,
        tree_sha256=stable_hash(rows),
    )


def _validate_index_snapshot(
    snapshot: IndexTreeSnapshot, field: str
) -> IndexTreeSnapshot:
    if (
        not isinstance(snapshot, IndexTreeSnapshot)
        or type(snapshot.file_count) is not int
        or snapshot.file_count < 1
        or type(snapshot.total_bytes) is not int
        or snapshot.total_bytes < 0
        or not isinstance(snapshot.tree_sha256, str)
        or _SHA256_RE.fullmatch(snapshot.tree_sha256) is None
    ):
        raise Dstc9OfficialHippoRAGError(f"{field} is invalid")
    return snapshot


def stable_top_five_from_official_result(
    *,
    retrieved_documents: Sequence[object],
    retrieved_scores: Sequence[object],
    document_to_ordinals: Mapping[str, Sequence[int]],
) -> tuple[int, ...]:
    """Expand unique-text official scores and rank by ``(-score, ordinal)``."""

    if isinstance(retrieved_documents, (str, bytes)) or isinstance(
        retrieved_scores, (str, bytes)
    ):
        raise Dstc9OfficialHippoRAGError("official result rows are malformed")
    try:
        documents = list(retrieved_documents)
        scores = list(retrieved_scores)
    except TypeError as exc:
        raise Dstc9OfficialHippoRAGError(
            "official result rows are not iterable"
        ) from exc
    canonical_mapping: dict[str, tuple[int, ...]] = {}
    all_ordinals: list[int] = []
    for document, raw_ordinals in document_to_ordinals.items():
        if not isinstance(document, str) or not document:
            raise Dstc9OfficialHippoRAGError(
                "document mapping contains an invalid document"
            )
        if isinstance(raw_ordinals, (str, bytes)):
            raise Dstc9OfficialHippoRAGError(
                "document mapping contains malformed ordinals"
            )
        try:
            ordinals = tuple(raw_ordinals)
        except TypeError as exc:
            raise Dstc9OfficialHippoRAGError(
                "document mapping contains malformed ordinals"
            ) from exc
        if not ordinals or any(type(value) is not int for value in ordinals):
            raise Dstc9OfficialHippoRAGError(
                "document mapping contains malformed ordinals"
            )
        canonical_mapping[document] = ordinals
        all_ordinals.extend(ordinals)
    if (
        len(all_ordinals) != CORPUS_SIZE
        or sorted(all_ordinals) != list(range(CORPUS_SIZE))
        or len(documents) != len(canonical_mapping)
        or len(scores) != len(canonical_mapping)
    ):
        raise Dstc9OfficialHippoRAGError(
            "official retrieve must return every unique global corpus text once"
        )

    ranked: list[tuple[float, int]] = []
    seen: set[str] = set()
    for document, score in zip(documents, scores):
        if not isinstance(document, str) or document not in canonical_mapping:
            raise Dstc9OfficialHippoRAGError(
                "official result contains an unknown corpus document"
            )
        if document in seen:
            raise Dstc9OfficialHippoRAGError(
                "official result contains a duplicate corpus document"
            )
        seen.add(document)
        if isinstance(score, bool) or not isinstance(score, Real):
            raise Dstc9OfficialHippoRAGError(
                "official result score is not numeric"
            )
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise Dstc9OfficialHippoRAGError(
                "official result score is not finite"
            )
        ranked.extend(
            (numeric_score, ordinal)
            for ordinal in canonical_mapping[document]
        )
    if seen != set(canonical_mapping):
        raise Dstc9OfficialHippoRAGError(
            "official result omitted a unique global corpus text"
        )
    ranked.sort(key=lambda row: (-row[0], row[1]))
    result = tuple(ordinal for _score, ordinal in ranked[:TOP_K])
    if len(result) != TOP_K or len(set(result)) != TOP_K:
        raise Dstc9OfficialHippoRAGError(
            "adapter did not produce five unique corpus ordinals"
        )
    return result


def make_build_receipt(
    corpus_input: CorpusInput,
    *,
    index_snapshot: IndexTreeSnapshot,
    runtime_attestation_receipt_sha256: str,
) -> dict[str, Any]:
    if not isinstance(corpus_input, CorpusInput):
        raise Dstc9OfficialHippoRAGError("validated corpus input is invalid")
    corpus_input = validate_corpus_input(corpus_input_projection(corpus_input))
    snapshot = _validate_index_snapshot(index_snapshot, "build index snapshot")
    runtime_hash = _required_sha256(
        runtime_attestation_receipt_sha256,
        "runtime binding receipt hash",
    )
    documents = serialize_corpus(corpus_input.units)
    multiplicity = corpus_text_multiplicity(documents)
    receipt: dict[str, Any] = {
        "adapter_version": ADAPTER_VERSION,
        "benchmark": BENCHMARK,
        "corpus_count": len(documents),
        "corpus_input_self_sha256": corpus_input.self_sha256,
        "corpus_sha256": corpus_sha256(documents),
        "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
        "duplicate_expansion_policy": DUPLICATE_EXPANSION_POLICY,
        **multiplicity,
        "dynamic_resize_count": 0,
        "force_index_from_scratch": True,
        "index_call_count": 1,
        "index_file_count": snapshot.file_count,
        "index_total_bytes": snapshot.total_bytes,
        "index_tree_sha256": snapshot.tree_sha256,
        "logical_cuda_device": LOGICAL_CUDA_DEVICE,
        "network_access": "denied",
        "official_hipporag_commit": OFFICIAL_HIPPORAG_COMMIT,
        "query_count_policy": FORMAL_QUERY_COUNT_POLICY,
        "retry_count": 0,
        "runtime_attestation_receipt_sha256": runtime_hash,
        "runtime_trust_root": RUNTIME_TRUST_ROOT,
        "schema": BUILD_RECEIPT_SCHEMA,
        "serialization": SERIALIZATION,
        "status": "passed_build_once",
        "study_id": corpus_input.study_id,
    }
    receipt["receipt_sha256"] = stable_hash(receipt)
    return receipt


def validate_build_receipt(
    receipt: Mapping[str, object],
    *,
    expected_corpus_input: CorpusInput,
    expected_index_snapshot: IndexTreeSnapshot,
    expected_runtime_attestation_receipt_sha256: str,
) -> dict[str, Any]:
    if not isinstance(receipt, Mapping) or set(receipt) != _BUILD_RECEIPT_KEYS:
        raise Dstc9OfficialHippoRAGError("build receipt key set mismatch")
    payload = dict(receipt)
    self_hash = _required_sha256(
        payload.pop("receipt_sha256", None), "build receipt self hash"
    )
    if self_hash != stable_hash(payload):
        raise Dstc9OfficialHippoRAGError("build receipt self hash mismatch")
    expected = make_build_receipt(
        expected_corpus_input,
        index_snapshot=_validate_index_snapshot(
            expected_index_snapshot, "expected build index snapshot"
        ),
        runtime_attestation_receipt_sha256=(
            expected_runtime_attestation_receipt_sha256
        ),
    )
    expected.pop("receipt_sha256")
    if payload != expected:
        raise Dstc9OfficialHippoRAGError("build receipt contract drifted")
    payload["receipt_sha256"] = self_hash
    return payload


def _validate_indices(
    value: object, *, query_count: int, corpus_count: int
) -> tuple[tuple[int, ...], ...]:
    if type(corpus_count) is not int or corpus_count != CORPUS_SIZE:
        raise Dstc9OfficialHippoRAGError("retrieval corpus count is invalid")
    if not isinstance(value, list) or len(value) != query_count:
        raise Dstc9OfficialHippoRAGError("retrieval ordinal row count mismatch")
    rows: list[tuple[int, ...]] = []
    for raw in value:
        if not isinstance(raw, list) or len(raw) != TOP_K:
            raise Dstc9OfficialHippoRAGError(
                "each retrieval output must contain exactly five ordinals"
            )
        if any(type(item) is not int or not 0 <= item < corpus_count for item in raw):
            raise Dstc9OfficialHippoRAGError(
                "retrieved ordinal is outside the global corpus"
            )
        if len(set(raw)) != TOP_K:
            raise Dstc9OfficialHippoRAGError(
                "retrieval output contains duplicate ordinals"
            )
        rows.append(tuple(raw))
    return tuple(rows)


def make_retrieval_receipt(
    *,
    corpus_input: CorpusInput,
    query_input: QueryInput,
    indices: Sequence[Sequence[int]],
    batch_sizes: Sequence[int],
    build_receipt: Mapping[str, object],
    index_snapshot_before: IndexTreeSnapshot,
    index_snapshot_after: IndexTreeSnapshot,
) -> dict[str, Any]:
    if (
        not isinstance(corpus_input, CorpusInput)
        or not isinstance(query_input, QueryInput)
        or corpus_input.study_id != query_input.study_id
    ):
        raise Dstc9OfficialHippoRAGError(
            "corpus and queries are not bound to one study"
        )
    corpus_input = validate_corpus_input(corpus_input_projection(corpus_input))
    query_input = validate_query_input(
        query_input_projection(query_input),
        expected_study_id=corpus_input.study_id,
    )
    before = _validate_index_snapshot(
        index_snapshot_before, "retrieval index snapshot before"
    )
    after = _validate_index_snapshot(
        index_snapshot_after, "retrieval index snapshot after"
    )
    documents = serialize_corpus(corpus_input.units)
    build_hash = _required_sha256(
        build_receipt.get("receipt_sha256"), "build receipt hash"
    )
    runtime_hash = _required_sha256(
        build_receipt.get("runtime_attestation_receipt_sha256"),
        "runtime binding receipt hash",
    )
    if (
        build_receipt.get("study_id") != corpus_input.study_id
        or build_receipt.get("corpus_input_self_sha256")
        != corpus_input.self_sha256
        or build_receipt.get("corpus_count") != CORPUS_SIZE
        or build_receipt.get("corpus_sha256") != corpus_sha256(documents)
        or build_receipt.get("index_tree_sha256") != before.tree_sha256
        or build_receipt.get("index_file_count") != before.file_count
        or build_receipt.get("index_total_bytes") != before.total_bytes
    ):
        raise Dstc9OfficialHippoRAGError(
            "retrieval is not bound to the validated build receipt"
        )
    canonical_indices = [list(row) for row in indices]
    _validate_indices(
        canonical_indices,
        query_count=len(query_input.queries),
        corpus_count=CORPUS_SIZE,
    )
    if (
        isinstance(batch_sizes, (str, bytes))
        or not isinstance(batch_sizes, Sequence)
        or not batch_sizes
        or any(
            type(size) is not int or not 1 <= size <= MAX_QUERY_BATCH
            for size in batch_sizes
        )
        or sum(batch_sizes) != len(query_input.queries)
    ):
        raise Dstc9OfficialHippoRAGError("retrieval batch sizes are invalid")
    multiplicity = corpus_text_multiplicity(documents)
    if any(build_receipt.get(key) != value for key, value in multiplicity.items()):
        raise Dstc9OfficialHippoRAGError(
            "duplicate aggregate is not bound to the corpus"
        )
    receipt: dict[str, Any] = {
        "adapter_version": ADAPTER_VERSION,
        "batch_sizes": list(batch_sizes),
        "benchmark": BENCHMARK,
        "build_receipt_sha256": build_hash,
        "corpus_count": CORPUS_SIZE,
        "corpus_input_self_sha256": corpus_input.self_sha256,
        "corpus_sha256": corpus_sha256(documents),
        "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
        "duplicate_expansion_policy": DUPLICATE_EXPANSION_POLICY,
        **multiplicity,
        "dynamic_resize_count": 0,
        "force_index_from_scratch": False,
        "index_call_count": 0,
        "index_changed_during_retrieve": before != after,
        "index_file_count": before.file_count,
        "index_post_file_count": after.file_count,
        "index_post_total_bytes": after.total_bytes,
        "index_post_tree_sha256": after.tree_sha256,
        "index_total_bytes": before.total_bytes,
        "index_tree_sha256": before.tree_sha256,
        "logical_cuda_device": LOGICAL_CUDA_DEVICE,
        "network_access": "denied",
        "official_hipporag_commit": OFFICIAL_HIPPORAG_COMMIT,
        "query_count": len(query_input.queries),
        "query_count_policy": FORMAL_QUERY_COUNT_POLICY,
        "query_input_self_sha256": query_input.self_sha256,
        "query_serialization": QUERY_SERIALIZATION,
        "query_sha256": query_sha256(query_input.queries),
        "result_ordinal_sha256": stable_hash(canonical_indices),
        "retrieval_call_count": len(batch_sizes),
        "retry_count": 0,
        "runtime_attestation_receipt_sha256": runtime_hash,
        "runtime_trust_root": RUNTIME_TRUST_ROOT,
        "schema": RETRIEVAL_RECEIPT_SCHEMA,
        "serialization": SERIALIZATION,
        "status": "passed_reopen_retrieve_only",
        "study_id": corpus_input.study_id,
    }
    receipt["receipt_sha256"] = stable_hash(receipt)
    return receipt


def parse_retrieval_output(
    raw: bytes,
    *,
    query_input: QueryInput,
    expected_build_receipt: Mapping[str, object],
    expected_index_snapshot_after: IndexTreeSnapshot,
) -> RetrievalBatch:
    """Reject every output channel other than ordinal rows and a receipt."""

    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Dstc9OfficialHippoRAGError(
            "retrieval worker output is invalid JSON"
        ) from exc
    if raw != canonical_json_bytes(value):
        raise Dstc9OfficialHippoRAGError(
            "retrieval worker output is not canonical JSON"
        )
    if not isinstance(value, dict) or set(value) != {
        "receipt",
        "retrieved_ordinals",
        "schema",
    }:
        raise Dstc9OfficialHippoRAGError(
            "retrieval worker may output only ordinals and a receipt"
        )
    if value.get("schema") != RETRIEVAL_OUTPUT_SCHEMA:
        raise Dstc9OfficialHippoRAGError("retrieval output schema mismatch")
    if not isinstance(query_input, QueryInput):
        raise Dstc9OfficialHippoRAGError("validated query input is invalid")
    query_input = validate_query_input(query_input_projection(query_input))
    if expected_build_receipt.get("study_id") != query_input.study_id:
        raise Dstc9OfficialHippoRAGError(
            "query input is not bound to the expected build study"
        )
    post_snapshot = _validate_index_snapshot(
        expected_index_snapshot_after,
        "expected retrieval index snapshot after",
    )
    before_snapshot = _validate_index_snapshot(
        IndexTreeSnapshot(
            file_count=expected_build_receipt.get("index_file_count"),  # type: ignore[arg-type]
            total_bytes=expected_build_receipt.get("index_total_bytes"),  # type: ignore[arg-type]
            tree_sha256=expected_build_receipt.get("index_tree_sha256"),  # type: ignore[arg-type]
        ),
        "expected build index snapshot",
    )
    indices = _validate_indices(
        value.get("retrieved_ordinals"),
        query_count=len(query_input.queries),
        corpus_count=expected_build_receipt.get("corpus_count"),  # type: ignore[arg-type]
    )
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping) or set(receipt) != _RETRIEVAL_RECEIPT_KEYS:
        raise Dstc9OfficialHippoRAGError("retrieval receipt key set mismatch")
    normalized = dict(receipt)
    self_hash = _required_sha256(
        normalized.pop("receipt_sha256", None),
        "retrieval receipt self hash",
    )
    if self_hash != stable_hash(normalized):
        raise Dstc9OfficialHippoRAGError("retrieval receipt self hash mismatch")
    batch_sizes = normalized.get("batch_sizes")
    if (
        not isinstance(batch_sizes, list)
        or not batch_sizes
        or any(
            type(size) is not int or not 1 <= size <= MAX_QUERY_BATCH
            for size in batch_sizes
        )
        or sum(batch_sizes) != len(query_input.queries)
    ):
        raise Dstc9OfficialHippoRAGError("retrieval batch receipt is invalid")
    expected = {
        "adapter_version": ADAPTER_VERSION,
        "batch_sizes": batch_sizes,
        "benchmark": BENCHMARK,
        "build_receipt_sha256": expected_build_receipt.get("receipt_sha256"),
        "corpus_count": CORPUS_SIZE,
        "corpus_input_self_sha256": expected_build_receipt.get(
            "corpus_input_self_sha256"
        ),
        "corpus_sha256": expected_build_receipt.get("corpus_sha256"),
        "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
        "duplicate_expansion_policy": DUPLICATE_EXPANSION_POLICY,
        "duplicate_text_group_count": expected_build_receipt.get(
            "duplicate_text_group_count"
        ),
        "duplicate_text_unit_count": expected_build_receipt.get(
            "duplicate_text_unit_count"
        ),
        "dynamic_resize_count": 0,
        "force_index_from_scratch": False,
        "index_call_count": 0,
        "index_changed_during_retrieve": before_snapshot != post_snapshot,
        "index_file_count": before_snapshot.file_count,
        "index_post_file_count": post_snapshot.file_count,
        "index_post_total_bytes": post_snapshot.total_bytes,
        "index_post_tree_sha256": post_snapshot.tree_sha256,
        "index_total_bytes": before_snapshot.total_bytes,
        "index_tree_sha256": before_snapshot.tree_sha256,
        "logical_cuda_device": LOGICAL_CUDA_DEVICE,
        "network_access": "denied",
        "official_hipporag_commit": OFFICIAL_HIPPORAG_COMMIT,
        "official_unique_text_count": expected_build_receipt.get(
            "official_unique_text_count"
        ),
        "query_count": len(query_input.queries),
        "query_count_policy": FORMAL_QUERY_COUNT_POLICY,
        "query_input_self_sha256": query_input.self_sha256,
        "query_serialization": QUERY_SERIALIZATION,
        "query_sha256": query_sha256(query_input.queries),
        "result_ordinal_sha256": stable_hash([list(row) for row in indices]),
        "retrieval_call_count": len(batch_sizes),
        "retry_count": 0,
        "runtime_attestation_receipt_sha256": expected_build_receipt.get(
            "runtime_attestation_receipt_sha256"
        ),
        "runtime_trust_root": RUNTIME_TRUST_ROOT,
        "schema": RETRIEVAL_RECEIPT_SCHEMA,
        "serialization": SERIALIZATION,
        "status": "passed_reopen_retrieve_only",
        "study_id": query_input.study_id,
    }
    if normalized != expected:
        raise Dstc9OfficialHippoRAGError("retrieval receipt contract drifted")
    normalized["receipt_sha256"] = self_hash
    return RetrievalBatch(indices=indices, receipt=normalized)


__all__ = [
    "ADAPTER_VERSION",
    "BENCHMARK",
    "BUILD_RECEIPT_SCHEMA",
    "CORPUS_INPUT_SCHEMA",
    "CORPUS_SIZE",
    "CUDA_VISIBLE_DEVICES",
    "CorpusInput",
    "CorpusUnit",
    "DUPLICATE_EXPANSION_POLICY",
    "Dstc9OfficialHippoRAGError",
    "FORMAL_QUERY_COUNT_POLICY",
    "FORMAL_QUERY_COUNT_UPPER_BOUND",
    "FROZEN_CORE_CONFIG",
    "IndexTreeSnapshot",
    "LOGICAL_CUDA_DEVICE",
    "MAX_CORPUS_SIZE",
    "MAX_QUERY_BATCH",
    "MAX_QUERY_COUNT",
    "MIN_CORPUS_SIZE",
    "OFFICIAL_HIPPORAG_COMMIT",
    "QUERY_INPUT_SCHEMA",
    "QUERY_SERIALIZATION",
    "QueryInput",
    "QueryRow",
    "RETRIEVAL_OUTPUT_SCHEMA",
    "RUNTIME_TRUST_ROOT",
    "RetrievalBatch",
    "SERIALIZATION",
    "SYSTEMD_NETWORK_PROPERTIES",
    "TOP_K",
    "TRANSPORT",
    "WORKER_ENVIRONMENT_KEYS",
    "WORKER_FIXED_ENVIRONMENT_VALUES",
    "canonical_json_bytes",
    "corpus_input_projection",
    "corpus_sha256",
    "corpus_text_multiplicity",
    "make_build_receipt",
    "make_corpus_input",
    "make_query_input",
    "make_retrieval_receipt",
    "parse_retrieval_output",
    "query_input_projection",
    "query_sha256",
    "serialize_corpus",
    "serialize_queries",
    "serialize_unit",
    "snapshot_index_tree",
    "stable_hash",
    "stable_top_five_from_official_result",
    "validate_build_receipt",
    "validate_corpus",
    "validate_corpus_input",
    "validate_queries",
    "validate_query_input",
]
