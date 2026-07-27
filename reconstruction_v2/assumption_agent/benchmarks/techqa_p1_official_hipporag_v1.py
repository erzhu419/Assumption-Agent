"""Source-free TechQA cluster adapter for the official HippoRAG worker.

The outer contract accepts only a public cluster projection: one audit-only
stage and cluster ordinal, exact question title/text pairs, and exact document
title/text pairs.  Source IDs, question IDs, qrels, families, gold documents,
answers, spans, labels, and evaluator values have no input field.

Every invocation creates one durable attempt marker and one fresh index root.
There is deliberately no retry, replay, resampling, online evaluator, API, or
network path.  The already-qualified AVeriTeC official worker is reused as an
inner implementation detail with its frozen ``block="A_hold"`` value.  The
outer stage and cluster ordinal are never passed to that worker and therefore
cannot influence retrieval.

The public output contains only audit values, counts, SHA-256 bindings, and
top-five public ordinals.  Query and document text stays in the private input
and the inner worker's private work directory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Callable, Mapping, Sequence

from replication_runtime.averitec_p1_official_v1 import worker as inner_worker
from replication_runtime.morehopqa_official_hipporag_v1 import (
    contract as inner_contract,
)

from . import techqa_p1_typed_core_v1 as typed_core


VERSION = "techqa_p1_official_hipporag_cluster_v1"
STUDY_ID = typed_core.STUDY_ID
INPUT_SCHEMA = f"{VERSION}_public_cluster_input_v1"
OUTPUT_SCHEMA = f"{VERSION}_safe_output_v1"
ATTEMPT_SCHEMA = f"{VERSION}_attempt_marker_v1"
OUTER_BINDING_SCHEMA = f"{VERSION}_outer_binding_v1"
INNER_QUERY_ID_SCHEMA = f"{VERSION}_inner_query_id_v1"
INDEX_LIFECYCLE = "one_fresh_index_per_outer_cluster_destroy_never_reused_v1"
INNER_BLOCK = inner_worker.FORMAL_BLOCK
ALLOWED_STAGES = ("A_hold", "M_search")
CLUSTER_COUNT_PER_STAGE = 4
MIN_DOCUMENT_COUNT = inner_contract.MIN_CORPUS_SIZE
MAX_DOCUMENT_COUNT = inner_contract.MAX_CORPUS_SIZE
MAX_QUERY_COUNT = inner_worker.MAX_QUERY_COUNT
QUERY_SERIALIZATION = "question_title_utf8_then_lf_then_question_text_utf8_v1"
DOCUMENT_SERIALIZATION = "title_utf8_then_lf_then_text_utf8_v1"
INNER_SERIALIZATION = inner_contract.SERIALIZATION
TOP_K = inner_contract.TOP_K

QUERY_KEYS = frozenset({"ordinal", "question_text", "question_title"})
DOCUMENT_KEYS = frozenset({"ordinal", "text", "title"})
INPUT_KEYS = frozenset(
    {
        "cluster_ordinal",
        "document_serialized_sha256",
        "documents",
        "queries",
        "query_serialized_sha256",
        "schema",
        "self_sha256",
        "stage",
        "study_id",
    }
)
ROW_KEYS = frozenset({"query_ordinal", "top5_document_ordinals"})
OUTPUT_KEYS = frozenset(
    {
        "attempt_marker_file_sha256",
        "attempt_marker_self_sha256",
        "cluster_ordinal",
        "document_count",
        "document_serialized_sha256",
        "fresh_index_create_count",
        "index_file_count",
        "index_lifecycle",
        "index_total_bytes",
        "index_tree_sha256",
        "inner_block",
        "inner_build_index_call_count",
        "inner_input_sha256",
        "inner_output_sha256",
        "inner_receipt_sha256",
        "inner_retrieval_index_call_count",
        "inner_serialization",
        "online_or_API_evaluator_call_count",
        "outer_binding_sha256",
        "outer_input_self_sha256",
        "query_count",
        "query_serialization",
        "query_serialized_sha256",
        "retry_replay_resample_count",
        "rows",
        "schema",
        "self_sha256",
        "stage",
        "status",
        "study_id",
        "document_serialization",
    }
)
ATTEMPT_KEYS = frozenset(
    {
        "attempt_count",
        "cluster_ordinal",
        "online_or_API_evaluator_call_count",
        "outer_input_self_sha256",
        "retry_replay_resample_count",
        "schema",
        "self_sha256",
        "stage",
        "study_id",
    }
)
FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "answer",
        "answerable",
        "answers",
        "category",
        "doc_id",
        "document_id",
        "end_offset",
        "evidence",
        "families",
        "family",
        "gold",
        "gold_document",
        "gold_document_id",
        "gold_id",
        "label",
        "labels",
        "product",
        "qrel",
        "qrels",
        "question_id",
        "source_id",
        "source_ids",
        "span",
        "split",
        "start_offset",
        "utility",
    }
)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class TechqaP1OfficialHippoRAGError(RuntimeError):
    """The public cluster, one-shot lifecycle, or inner result drifted."""


@dataclass(frozen=True, slots=True)
class PublicQuery:
    ordinal: int
    question_title: str
    question_text: str


@dataclass(frozen=True, slots=True)
class PublicCluster:
    stage: str
    cluster_ordinal: int
    query_serialized_sha256: str
    document_serialized_sha256: str
    queries: tuple[PublicQuery, ...]
    documents: tuple[typed_core.Document, ...]
    self_sha256: str


InnerRunner = Callable[..., Mapping[str, object]]


def canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    """Return the exact ASCII JSON encoding used by every outer receipt."""

    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise TechqaP1OfficialHippoRAGError(
            "value is not canonical JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise TechqaP1OfficialHippoRAGError(
            f"{field} is not a lowercase SHA-256"
        )
    return value


def _strict_int(
    value: object,
    field: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if (
        type(value) is not int
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise TechqaP1OfficialHippoRAGError(f"{field} is invalid")
    return value


def _self_hashed(body: Mapping[str, object]) -> dict[str, object]:
    if "self_sha256" in body:
        raise TechqaP1OfficialHippoRAGError("self hash was supplied twice")
    value = dict(body)
    value["self_sha256"] = stable_hash(value)
    return value


def _verify_self(value: Mapping[str, object], field: str) -> str:
    body = dict(value)
    claimed = body.pop("self_sha256", None)
    if (
        not isinstance(claimed, str)
        or _HEX64.fullmatch(claimed) is None
        or stable_hash(body) != claimed
    ):
        raise TechqaP1OfficialHippoRAGError(f"{field} self hash drifted")
    return claimed


def _reject_forbidden_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if (
                not isinstance(key, str)
                or key.casefold() in FORBIDDEN_INPUT_KEYS
            ):
                raise TechqaP1OfficialHippoRAGError(
                    "public cluster contains a forbidden label/source field"
                )
            _reject_forbidden_keys(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            _reject_forbidden_keys(nested)


def _text(
    value: object,
    field: str,
    *,
    maximum: int,
    allow_empty: bool = False,
) -> str:
    if (
        not isinstance(value, str)
        or "\x00" in value
        or len(value) > maximum
        or (not allow_empty and not value.strip())
    ):
        raise TechqaP1OfficialHippoRAGError(f"{field} is invalid")
    return value


def _query_projection(queries: Sequence[PublicQuery]) -> list[dict[str, object]]:
    return [
        {
            "ordinal": row.ordinal,
            "question_text": row.question_text,
            "question_title": row.question_title,
        }
        for row in queries
    ]


def _document_projection(
    documents: Sequence[typed_core.Document],
) -> list[dict[str, object]]:
    return [
        {
            "ordinal": row.ordinal,
            "text": row.text,
            "title": row.title,
        }
        for row in documents
    ]


def serialize_query(row: PublicQuery) -> str:
    if not isinstance(row, PublicQuery):
        raise TechqaP1OfficialHippoRAGError("query row is invalid")
    try:
        value = typed_core.serialize_query_text(
            row.question_title, row.question_text
        )
    except typed_core.TechqaP1TypedCoreError as exc:
        raise TechqaP1OfficialHippoRAGError(
            "query serialization failed"
        ) from exc
    if len(value) > inner_worker.MAX_QUERY_CHARACTERS:
        raise TechqaP1OfficialHippoRAGError(
            "serialized query exceeds the official worker bound"
        )
    return value


def serialize_document(row: typed_core.Document) -> str:
    try:
        return typed_core.serialize_document_text(row)
    except typed_core.TechqaP1TypedCoreError as exc:
        raise TechqaP1OfficialHippoRAGError(
            "document serialization failed"
        ) from exc


def query_serialized_sha256(queries: Sequence[PublicQuery]) -> str:
    return stable_hash([serialize_query(row) for row in queries])


def document_serialized_sha256(
    documents: Sequence[typed_core.Document],
) -> str:
    return stable_hash([serialize_document(row) for row in documents])


def input_payload(
    *,
    stage: str,
    cluster_ordinal: int,
    queries: Sequence[Mapping[str, object]],
    documents: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Construct and validate one exact public cluster input."""

    provisional = {
        "cluster_ordinal": cluster_ordinal,
        "document_serialized_sha256": "0" * 64,
        "documents": [dict(row) for row in documents],
        "queries": [dict(row) for row in queries],
        "query_serialized_sha256": "0" * 64,
        "schema": INPUT_SCHEMA,
        "stage": stage,
        "study_id": STUDY_ID,
    }
    query_rows, document_rows = _validate_public_rows(provisional)
    provisional["query_serialized_sha256"] = query_serialized_sha256(
        query_rows
    )
    provisional["document_serialized_sha256"] = (
        document_serialized_sha256(document_rows)
    )
    payload = _self_hashed(provisional)
    validate_input(payload)
    return payload


def _validate_public_rows(
    value: Mapping[str, object],
) -> tuple[tuple[PublicQuery, ...], tuple[typed_core.Document, ...]]:
    raw_queries = value.get("queries")
    raw_documents = value.get("documents")
    if (
        isinstance(raw_queries, (str, bytes))
        or not isinstance(raw_queries, Sequence)
        or not 1 <= len(raw_queries) <= MAX_QUERY_COUNT
    ):
        raise TechqaP1OfficialHippoRAGError("query count is invalid")
    if (
        isinstance(raw_documents, (str, bytes))
        or not isinstance(raw_documents, Sequence)
        or not MIN_DOCUMENT_COUNT
        <= len(raw_documents)
        <= MAX_DOCUMENT_COUNT
    ):
        raise TechqaP1OfficialHippoRAGError("document count is invalid")

    queries: list[PublicQuery] = []
    for position, raw in enumerate(raw_queries):
        if not isinstance(raw, Mapping) or set(raw) != QUERY_KEYS:
            raise TechqaP1OfficialHippoRAGError("query schema drifted")
        ordinal = _strict_int(raw.get("ordinal"), "query ordinal")
        if ordinal != position:
            raise TechqaP1OfficialHippoRAGError(
                "query ordinals are not contiguous"
            )
        row = PublicQuery(
            ordinal=ordinal,
            question_title=_text(
                raw.get("question_title"),
                "question title",
                maximum=20_000,
            ),
            question_text=_text(
                raw.get("question_text"),
                "question text",
                maximum=100_000,
                allow_empty=True,
            ),
        )
        serialize_query(row)
        queries.append(row)

    documents: list[typed_core.Document] = []
    for position, raw in enumerate(raw_documents):
        if not isinstance(raw, Mapping) or set(raw) != DOCUMENT_KEYS:
            raise TechqaP1OfficialHippoRAGError("document schema drifted")
        try:
            row = typed_core.document_from_public_fields(raw)
        except typed_core.TechqaP1TypedCoreError as exc:
            raise TechqaP1OfficialHippoRAGError(
                "document projection is invalid"
            ) from exc
        if row.ordinal != position:
            raise TechqaP1OfficialHippoRAGError(
                "document ordinals are not contiguous"
            )
        documents.append(row)
    serialized_documents = [serialize_document(row) for row in documents]
    if len(set(serialized_documents)) != len(serialized_documents):
        raise TechqaP1OfficialHippoRAGError(
            "document serialization is duplicated"
        )
    return tuple(queries), tuple(documents)


def validate_input(value: object) -> PublicCluster:
    """Validate the strict public-only input and both supplied content hashes."""

    _reject_forbidden_keys(value)
    if not isinstance(value, Mapping) or set(value) != INPUT_KEYS:
        raise TechqaP1OfficialHippoRAGError("public cluster schema drifted")
    if (
        value.get("schema") != INPUT_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("stage") not in ALLOWED_STAGES
    ):
        raise TechqaP1OfficialHippoRAGError(
            "public cluster identity drifted"
        )
    stage = str(value["stage"])
    cluster_ordinal = _strict_int(
        value.get("cluster_ordinal"),
        "cluster ordinal",
        maximum=CLUSTER_COUNT_PER_STAGE - 1,
    )
    supplied_query_hash = _sha256(
        value.get("query_serialized_sha256"),
        "query serialized SHA-256",
    )
    supplied_document_hash = _sha256(
        value.get("document_serialized_sha256"),
        "document serialized SHA-256",
    )
    queries, documents = _validate_public_rows(value)
    if supplied_query_hash != query_serialized_sha256(queries):
        raise TechqaP1OfficialHippoRAGError(
            "query serialized SHA-256 mismatch"
        )
    if supplied_document_hash != document_serialized_sha256(documents):
        raise TechqaP1OfficialHippoRAGError(
            "document serialized SHA-256 mismatch"
        )
    self_hash = _verify_self(value, "public cluster")
    return PublicCluster(
        stage=stage,
        cluster_ordinal=cluster_ordinal,
        query_serialized_sha256=supplied_query_hash,
        document_serialized_sha256=supplied_document_hash,
        queries=queries,
        documents=documents,
        self_sha256=self_hash,
    )


def _inner_query_id(row: PublicQuery) -> str:
    return stable_hash(
        {
            "query_ordinal": row.ordinal,
            "query_utf8_sha256": hashlib.sha256(
                serialize_query(row).encode("utf-8")
            ).hexdigest(),
            "schema": INNER_QUERY_ID_SCHEMA,
        }
    )


def inner_payload(cluster: PublicCluster) -> dict[str, object]:
    """Create the stage-invariant inner input.

    Neither ``cluster.stage`` nor ``cluster.cluster_ordinal`` is consulted.
    The frozen AVeriTeC block value is an inner lineage binding, not an outer
    TechQA stage.
    """

    if not isinstance(cluster, PublicCluster):
        raise TechqaP1OfficialHippoRAGError("public cluster is invalid")
    payload = inner_worker.input_payload(
        block=INNER_BLOCK,
        articles=[
            {
                "body": row.text,
                "idx": row.ordinal,
                "title": row.title,
            }
            for row in cluster.documents
        ],
        queries=[
            (_inner_query_id(row), serialize_query(row))
            for row in cluster.queries
        ],
    )
    inner_worker.validate_input(payload)
    return payload


def outer_binding(cluster: PublicCluster) -> str:
    return stable_hash(
        {
            "cluster_ordinal": cluster.cluster_ordinal,
            "document_serialized_sha256": (
                cluster.document_serialized_sha256
            ),
            "outer_input_self_sha256": cluster.self_sha256,
            "query_serialized_sha256": cluster.query_serialized_sha256,
            "schema": OUTER_BINDING_SCHEMA,
            "stage": cluster.stage,
            "study_id": STUDY_ID,
        }
    )


def _absolute_path(value: Path, field: str) -> Path:
    if (
        not isinstance(value, Path)
        or not value.is_absolute()
        or "\x00" in str(value)
        or ".." in value.parts
    ):
        raise TechqaP1OfficialHippoRAGError(
            f"{field} must be a normalized absolute path"
        )
    return value


def _write_exclusive(path: Path, value: Mapping[str, object]) -> bytes:
    raw = canonical_bytes(value, newline=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise TechqaP1OfficialHippoRAGError(
            "exclusive output creation failed"
        ) from exc
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise TechqaP1OfficialHippoRAGError(
                    "exclusive output write stalled"
                )
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    info = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o600
        or path.read_bytes() != raw
    ):
        raise TechqaP1OfficialHippoRAGError(
            "exclusive output metadata drifted"
        )
    return raw


def _attempt_marker(cluster: PublicCluster) -> dict[str, object]:
    return _self_hashed(
        {
            "attempt_count": 1,
            "cluster_ordinal": cluster.cluster_ordinal,
            "online_or_API_evaluator_call_count": 0,
            "outer_input_self_sha256": cluster.self_sha256,
            "retry_replay_resample_count": 0,
            "schema": ATTEMPT_SCHEMA,
            "stage": cluster.stage,
            "study_id": STUDY_ID,
        }
    )


def _validate_attempt(value: object, cluster: PublicCluster) -> str:
    if not isinstance(value, Mapping) or set(value) != ATTEMPT_KEYS:
        raise TechqaP1OfficialHippoRAGError("attempt marker schema drifted")
    if (
        value.get("schema") != ATTEMPT_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("stage") != cluster.stage
        or value.get("cluster_ordinal") != cluster.cluster_ordinal
        or value.get("outer_input_self_sha256") != cluster.self_sha256
        or value.get("attempt_count") != 1
        or value.get("retry_replay_resample_count") != 0
        or value.get("online_or_API_evaluator_call_count") != 0
    ):
        raise TechqaP1OfficialHippoRAGError(
            "attempt marker binding drifted"
        )
    return _verify_self(value, "attempt marker")


def _rows_from_inner(
    *,
    cluster: PublicCluster,
    inner_input: Mapping[str, object],
    inner_output: object,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not isinstance(inner_output, Mapping):
        raise TechqaP1OfficialHippoRAGError(
            "inner worker returned no mapping"
        )
    try:
        checked = inner_worker.validate_output(
            inner_output, expected_input=inner_input
        )
    except inner_worker.AveritecP1OfficialError as exc:
        raise TechqaP1OfficialHippoRAGError(
            "inner official output validation failed"
        ) from exc
    expected_ids = [_inner_query_id(row) for row in cluster.queries]
    output_rows = checked.get("rows")
    if not isinstance(output_rows, list) or len(output_rows) != len(
        expected_ids
    ):
        raise TechqaP1OfficialHippoRAGError(
            "inner official row count drifted"
        )
    rows: list[dict[str, object]] = []
    for ordinal, (expected_id, row) in enumerate(
        zip(expected_ids, output_rows)
    ):
        if (
            not isinstance(row, Mapping)
            or row.get("item_id") != expected_id
        ):
            raise TechqaP1OfficialHippoRAGError(
                "inner query binding drifted"
            )
        top5 = row.get("top5_document_ordinals")
        if (
            not isinstance(top5, list)
            or len(top5) != TOP_K
            or len(set(top5)) != TOP_K
            or any(
                type(value) is not int
                or not 0 <= value < len(cluster.documents)
                for value in top5
            )
        ):
            raise TechqaP1OfficialHippoRAGError(
                "inner top-five ordinals drifted"
            )
        rows.append(
            {
                "query_ordinal": ordinal,
                "top5_document_ordinals": list(top5),
            }
        )
    return rows, checked


def execute_cluster_once(
    value: object,
    *,
    work_root: Path,
    inner_runner: InnerRunner,
) -> dict[str, object]:
    """Execute one outer cluster exactly once with an injected inner worker."""

    cluster = validate_input(value)
    root = _absolute_path(work_root, "work root")
    if not callable(inner_runner):
        raise TechqaP1OfficialHippoRAGError("inner runner is not callable")
    try:
        root.mkdir(mode=0o700)
    except OSError as exc:
        raise TechqaP1OfficialHippoRAGError(
            "fresh work root cannot be created exactly once"
        ) from exc
    if root.is_symlink() or stat.S_IMODE(root.stat().st_mode) != 0o700:
        raise TechqaP1OfficialHippoRAGError("fresh work root drifted")

    marker = _attempt_marker(cluster)
    marker_self_hash = _validate_attempt(marker, cluster)
    marker_raw = _write_exclusive(root / "attempt.json", marker)
    marker_file_hash = hashlib.sha256(marker_raw).hexdigest()

    index_root = root / "fresh_index"
    if index_root.exists() or index_root.is_symlink():
        raise TechqaP1OfficialHippoRAGError(
            "fresh index root already exists"
        )
    private_inner_input = inner_payload(cluster)
    try:
        raw_inner_output = inner_runner(
            private_input=private_inner_input,
            index_root=index_root,
        )
    except BaseException as exc:
        raise TechqaP1OfficialHippoRAGError(
            "inner official worker failed; retry is forbidden"
        ) from exc
    rows, checked_inner = _rows_from_inner(
        cluster=cluster,
        inner_input=private_inner_input,
        inner_output=raw_inner_output,
    )
    try:
        snapshot = inner_contract.snapshot_index_tree(index_root)
    except inner_contract.MoreHopQAOfficialHippoRAGError as exc:
        raise TechqaP1OfficialHippoRAGError(
            "fresh official index evidence drifted"
        ) from exc
    receipt = checked_inner.get("receipt")
    if not isinstance(receipt, Mapping):
        raise TechqaP1OfficialHippoRAGError(
            "inner official receipt disappeared"
        )
    receipt_self_hash = _sha256(
        receipt.get("self_sha256"), "inner receipt self SHA-256"
    )
    if (
        receipt.get("index_post_tree_sha256") != snapshot.tree_sha256
        or receipt.get("index_post_file_count") != snapshot.file_count
        or receipt.get("index_post_total_bytes") != snapshot.total_bytes
    ):
        raise TechqaP1OfficialHippoRAGError(
            "fresh index snapshot is not bound to the inner receipt"
        )
    output = _self_hashed(
        {
            "attempt_marker_file_sha256": marker_file_hash,
            "attempt_marker_self_sha256": marker_self_hash,
            "cluster_ordinal": cluster.cluster_ordinal,
            "document_count": len(cluster.documents),
            "document_serialization": DOCUMENT_SERIALIZATION,
            "document_serialized_sha256": (
                cluster.document_serialized_sha256
            ),
            "fresh_index_create_count": 1,
            "index_file_count": snapshot.file_count,
            "index_lifecycle": INDEX_LIFECYCLE,
            "index_total_bytes": snapshot.total_bytes,
            "index_tree_sha256": snapshot.tree_sha256,
            "inner_block": INNER_BLOCK,
            "inner_build_index_call_count": receipt.get(
                "build_index_call_count"
            ),
            "inner_input_sha256": inner_worker.stable_hash(
                private_inner_input
            ),
            "inner_output_sha256": _sha256(
                checked_inner.get("self_sha256"),
                "inner output self SHA-256",
            ),
            "inner_receipt_sha256": receipt_self_hash,
            "inner_retrieval_index_call_count": receipt.get(
                "retrieval_index_call_count"
            ),
            "inner_serialization": INNER_SERIALIZATION,
            "online_or_API_evaluator_call_count": 0,
            "outer_binding_sha256": outer_binding(cluster),
            "outer_input_self_sha256": cluster.self_sha256,
            "query_count": len(cluster.queries),
            "query_serialization": QUERY_SERIALIZATION,
            "query_serialized_sha256": cluster.query_serialized_sha256,
            "retry_replay_resample_count": 0,
            "rows": rows,
            "schema": OUTPUT_SCHEMA,
            "stage": cluster.stage,
            "status": "passed_once",
            "study_id": STUDY_ID,
        }
    )
    validate_output(output, expected_input=value)
    return output


def validate_output(
    value: object,
    *,
    expected_input: object,
) -> dict[str, object]:
    """Validate the safe, text-free outer result against its public input."""

    cluster = validate_input(expected_input)
    if not isinstance(value, Mapping) or set(value) != OUTPUT_KEYS:
        raise TechqaP1OfficialHippoRAGError("safe output schema drifted")
    output = dict(value)
    self_hash = _verify_self(output, "safe output")
    expected_marker = _attempt_marker(cluster)
    expected_marker_self_hash = _validate_attempt(
        expected_marker, cluster
    )
    expected_marker_file_hash = hashlib.sha256(
        canonical_bytes(expected_marker, newline=True)
    ).hexdigest()
    expected_inner_input_hash = inner_worker.stable_hash(
        inner_payload(cluster)
    )
    if (
        output.get("schema") != OUTPUT_SCHEMA
        or output.get("study_id") != STUDY_ID
        or output.get("stage") != cluster.stage
        or output.get("cluster_ordinal") != cluster.cluster_ordinal
        or output.get("outer_input_self_sha256") != cluster.self_sha256
        or output.get("outer_binding_sha256") != outer_binding(cluster)
        or output.get("query_serialized_sha256")
        != cluster.query_serialized_sha256
        or output.get("document_serialized_sha256")
        != cluster.document_serialized_sha256
        or output.get("query_count") != len(cluster.queries)
        or output.get("document_count") != len(cluster.documents)
        or output.get("query_serialization") != QUERY_SERIALIZATION
        or output.get("document_serialization")
        != DOCUMENT_SERIALIZATION
        or output.get("inner_serialization") != INNER_SERIALIZATION
        or output.get("inner_block") != INNER_BLOCK
        or output.get("index_lifecycle") != INDEX_LIFECYCLE
        or output.get("fresh_index_create_count") != 1
        or output.get("inner_build_index_call_count") != 1
        or output.get("inner_retrieval_index_call_count") != 0
        or output.get("retry_replay_resample_count") != 0
        or output.get("online_or_API_evaluator_call_count") != 0
        or output.get("status") != "passed_once"
        or output.get("attempt_marker_self_sha256")
        != expected_marker_self_hash
        or output.get("attempt_marker_file_sha256")
        != expected_marker_file_hash
        or output.get("inner_input_sha256") != expected_inner_input_hash
    ):
        raise TechqaP1OfficialHippoRAGError(
            "safe output binding drifted"
        )
    for field in (
        "attempt_marker_file_sha256",
        "attempt_marker_self_sha256",
        "index_tree_sha256",
        "inner_input_sha256",
        "inner_output_sha256",
        "inner_receipt_sha256",
    ):
        _sha256(output.get(field), field)
    _strict_int(output.get("index_file_count"), "index file count", minimum=1)
    _strict_int(output.get("index_total_bytes"), "index total bytes")
    rows = output.get("rows")
    if not isinstance(rows, list) or len(rows) != len(cluster.queries):
        raise TechqaP1OfficialHippoRAGError("safe output row count drifted")
    for position, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != ROW_KEYS
            or row.get("query_ordinal") != position
        ):
            raise TechqaP1OfficialHippoRAGError(
                "safe output row schema drifted"
            )
        top5 = row.get("top5_document_ordinals")
        if (
            not isinstance(top5, list)
            or len(top5) != TOP_K
            or len(set(top5)) != TOP_K
            or any(
                type(ordinal) is not int
                or not 0 <= ordinal < len(cluster.documents)
                for ordinal in top5
            )
        ):
            raise TechqaP1OfficialHippoRAGError(
                "safe output top-five drifted"
            )
    output["self_sha256"] = self_hash
    return output


def _read_canonical_private(path: Path) -> dict[str, object]:
    path = _absolute_path(path, "input path")
    try:
        info = path.lstat()
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TechqaP1OfficialHippoRAGError(
            "canonical public cluster input is unavailable"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o600
        or not isinstance(value, dict)
        or raw != canonical_bytes(value, newline=True)
    ):
        raise TechqaP1OfficialHippoRAGError(
            "canonical public cluster input metadata drifted"
        )
    validate_input(value)
    return value


def production_inner_runner(
    *,
    private_input: Mapping[str, object],
    index_root: Path,
    llm_model: str,
    embedding_model: str,
    hipporag_source_root: Path,
) -> Mapping[str, object]:
    """Invoke the qualified worker once; this function contains no retry."""

    return inner_worker.run_once(
        private_input=private_input,
        output_path=index_root.parent / "inner.output.json",
        index_root=index_root,
        llm_model=llm_model,
        embedding_model=embedding_model,
        hipporag_source_root=hipporag_source_root,
    )


def run_from_files(
    *,
    input_path: Path,
    output_path: Path,
    work_root: Path,
    inner_runner: InnerRunner,
) -> dict[str, object]:
    """Read, execute, and exclusively persist one safe cluster result."""

    output_path = _absolute_path(output_path, "output path")
    if output_path.exists() or output_path.is_symlink():
        raise TechqaP1OfficialHippoRAGError(
            "safe output path already exists"
        )
    try:
        output_parent = output_path.parent
        output_parent_info = output_parent.lstat()
    except OSError as exc:
        raise TechqaP1OfficialHippoRAGError(
            "safe output parent is unavailable"
        ) from exc
    if (
        output_parent.is_symlink()
        or not stat.S_ISDIR(output_parent_info.st_mode)
    ):
        raise TechqaP1OfficialHippoRAGError(
            "safe output parent is not a direct directory"
        )
    payload = _read_canonical_private(input_path)
    result = execute_cluster_once(
        payload,
        work_root=work_root,
        inner_runner=inner_runner,
    )
    _write_exclusive(output_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--hipporag-source-root", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    arguments = parser.parse_args(argv)

    # Reuse the qualified worker's exact offline/thread and import-origin
    # preconditions before consuming the outer attempt.
    inner_worker._require_environment()
    inner_worker._require_project_origins(arguments.project_root)
    runner = partial(
        production_inner_runner,
        llm_model=inner_worker._model_alias(
            arguments.llm_model, "LLM model"
        ),
        embedding_model=inner_worker._model_alias(
            arguments.embedding_model, "embedding model"
        ),
        hipporag_source_root=arguments.hipporag_source_root,
    )
    result = run_from_files(
        input_path=arguments.input,
        output_path=arguments.output,
        work_root=arguments.work_root,
        inner_runner=runner,
    )
    print(
        json.dumps(
            {
                "cluster_ordinal": result["cluster_ordinal"],
                "document_count": result["document_count"],
                "query_count": result["query_count"],
                "stage": result["stage"],
                "status": result["status"],
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ALLOWED_STAGES",
    "CLUSTER_COUNT_PER_STAGE",
    "DOCUMENT_SERIALIZATION",
    "INDEX_LIFECYCLE",
    "INNER_BLOCK",
    "INPUT_SCHEMA",
    "OUTPUT_SCHEMA",
    "PublicCluster",
    "PublicQuery",
    "QUERY_SERIALIZATION",
    "STUDY_ID",
    "TOP_K",
    "TechqaP1OfficialHippoRAGError",
    "canonical_bytes",
    "document_serialized_sha256",
    "execute_cluster_once",
    "inner_payload",
    "input_payload",
    "main",
    "outer_binding",
    "production_inner_runner",
    "query_serialized_sha256",
    "run_from_files",
    "serialize_document",
    "serialize_query",
    "stable_hash",
    "validate_input",
    "validate_output",
]


if __name__ == "__main__":
    raise SystemExit(main())
