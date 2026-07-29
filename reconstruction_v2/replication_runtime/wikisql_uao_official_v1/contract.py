"""Frozen label-free contract for the WikiSQL UAO official-HippoRAG arm.

The worker receives only opaque item IDs, questions, table headers/types, and
complete item-local table rows.  SQL, answer, relation-family, gold, utility,
score, qrel, and evaluator fields are deliberately outside the representable
input language.  Every item is indexed independently.  Native rankings are
validated privately, then emitted only as the same view-bound common action
pack used by RAW and Agent; a separate safe receipt contains aggregate
commitments but no item, query, document, or action details.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import re
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks import (
    wikisql_uao_action_runtime_v1 as action_runtime,
)
from assumption_agent.benchmarks import wikisql_uao_reality_v1 as reality


VERSION = "wikisql_uao_candidate_restricted_official_hipporag_v1"
INPUT_SCHEMA = action_runtime.VIEW_PACK_SCHEMA
OUTPUT_SCHEMA = f"{VERSION}_private_native_output"
SAFE_RECEIPT_SCHEMA = f"{VERSION}_safe_aggregate_receipt"
ROW_DOCUMENT_SCHEMA = f"{VERSION}_row_document"
INDEX_RECEIPT_SCHEMA = f"{VERSION}_index_receipt"
RANKING_ROW_SCHEMA = f"{VERSION}_ranking_row"

OFFICIAL_HIPPORAG_COMMIT = "ef2f14c4f254f11ac29f9395f262466ad1bb4d10"
TOP_K = 5
MIN_ITEM_COUNT = 1
MAX_ITEM_COUNT = 72
MIN_ROW_COUNT = 11
MAX_ROW_COUNT = 80
MIN_COLUMN_COUNT = 1
MAX_COLUMN_COUNT = action_runtime.MAX_COLUMNS
MAX_QUESTION_CHARACTERS = action_runtime.MAX_QUESTION_CHARACTERS
MAX_HEADER_CHARACTERS = action_runtime.MAX_HEADER_CHARACTERS
MAX_CELL_CHARACTERS = action_runtime.MAX_CELL_CHARACTERS
MAX_SERIALIZED_ROW_CHARACTERS = (
    action_runtime.MAX_SERIALIZED_ROW_CHARACTERS
)
WIKISQL_COLUMN_TYPES = frozenset({"real", "text"})

FROZEN_CORE_CONFIG = {
    "force_index_from_scratch": True,
    "max_new_tokens": 96,
    "openie_mode": "online",
    "qa_top_k": TOP_K,
    "save_openie": True,
    "seed": 0,
    "temperature": 0,
}

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_ITEM_BASE_KEYS = frozenset(
    {"headers", "item_id", "question", "rows", "types"}
)
_ITEM_KEYS = frozenset(
    {
        *_ITEM_BASE_KEYS,
        "item_sha256",
        "row_corpus_sha256",
    }
)
_RANKING_KEYS = frozenset(
    {
        "item_id",
        "item_ordinal",
        "row_sha256",
        "schema",
        "top5_row_ordinals",
    }
)
_INDEX_RECEIPT_KEYS = frozenset(
    {
        "byte_count",
        "file_count",
        "fresh_index",
        "graph_edge_count",
        "graph_node_count",
        "index_call_count",
        "index_root_binding_sha256",
        "index_tree_sha256",
        "item_id",
        "item_ordinal",
        "item_sha256",
        "retrieve_call_count",
        "row_corpus_sha256",
        "row_count",
        "schema",
        "self_sha256",
    }
)
_OUTPUT_KEYS = frozenset(
    {
        "action_view_pack_sha256",
        "index_receipts",
        "index_receipts_sha256",
        "item_count",
        "official_hipporag_commit",
        "rankings",
        "rankings_sha256",
        "runtime",
        "schema",
        "self_sha256",
    }
)
_SAFE_RECEIPT_KEYS = frozenset(
    {
        "action_pack_sha256",
        "action_view_pack_sha256",
        "arm",
        "block",
        "index_receipts_sha256",
        "item_count",
        "native_output_sha256",
        "official_hipporag_commit",
        "rankings_sha256",
        "runtime",
        "schema",
        "self_sha256",
        "status",
        "study_id",
    }
)
_RUNTIME_KEYS = frozenset(
    {
        "core_config_sha256",
        "evaluator_call_count",
        "fresh_index_per_item",
        "index_call_count",
        "network_call_count",
        "official_hipporag_commit",
        "replay_count",
        "retrieve_call_count",
        "retry_count",
        "sequential_item_execution",
        "single_gpu_lane_count",
        "top_k",
    }
)
_FORBIDDEN_FIELD_FRAGMENTS = (
    "answer",
    "evaluator",
    "family",
    "gold",
    "label",
    "qrel",
    "score",
    "sql",
    "utility",
)


class WikiSQLUAOOfficialHippoRAGError(RuntimeError):
    """The source-free contract or pinned official-core result failed closed."""


Cell = str | int | float


@dataclass(frozen=True)
class WikiSQLItem:
    """One validated, label-free, item-local table retrieval problem."""

    item_id: str
    question: str
    headers: tuple[str, ...]
    types: tuple[str, ...]
    rows: tuple[tuple[Cell, ...], ...]
    item_sha256: str
    row_corpus_sha256: str


def canonical_json_bytes(value: object, *, newline: bool = True) -> bytes:
    """Encode one exact ASCII JSON value."""

    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise WikiSQLUAOOfficialHippoRAGError(
            "value is not canonical JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def semantic_sha256(value: object) -> str:
    """Hash an exact semantic value without a line terminator."""

    return hashlib.sha256(
        canonical_json_bytes(value, newline=False)
    ).hexdigest()


CORE_CONFIG_SHA256 = semantic_sha256(FROZEN_CORE_CONFIG)


def _hex64(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise WikiSQLUAOOfficialHippoRAGError(
            f"{field} must be a lowercase SHA-256 value"
        )
    return value


def _integer(
    value: object, field: str, *, minimum: int = 0
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise WikiSQLUAOOfficialHippoRAGError(f"{field} drifted")
    return value


def _text(
    value: object,
    field: str,
    *,
    maximum: int,
    allow_empty: bool = False,
) -> str:
    if (
        not isinstance(value, str)
        or (not allow_empty and not value.strip())
        or "\x00" in value
        or len(value) > maximum
    ):
        raise WikiSQLUAOOfficialHippoRAGError(f"{field} is invalid")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise WikiSQLUAOOfficialHippoRAGError(
            f"{field} contains invalid Unicode"
        ) from exc
    return value


def reject_forbidden_fields(value: object, *, path: str = "$") -> None:
    """Reject label/effect fields at every mapping depth.

    Data strings are not inspected: a legitimate header or cell may contain a
    word such as ``label``.  Only JSON field names define capabilities.
    """

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise WikiSQLUAOOfficialHippoRAGError(
                    f"{path} contains a non-string field name"
                )
            lowered = raw_key.casefold()
            if lowered == "contains_labels" and child is False:
                continue
            if any(
                fragment in lowered
                for fragment in _FORBIDDEN_FIELD_FRAGMENTS
            ):
                raise WikiSQLUAOOfficialHippoRAGError(
                    f"forbidden field at {path}.{raw_key}"
                )
            reject_forbidden_fields(child, path=f"{path}.{raw_key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for ordinal, child in enumerate(value):
            reject_forbidden_fields(child, path=f"{path}[{ordinal}]")


def _cell(value: object, field: str) -> Cell:
    if isinstance(value, bool):
        raise WikiSQLUAOOfficialHippoRAGError(f"{field} is invalid")
    if isinstance(value, str):
        return _text(
            value,
            field,
            maximum=MAX_CELL_CHARACTERS,
            allow_empty=True,
        )
    if isinstance(value, int):
        if abs(value) > 10**18:
            raise WikiSQLUAOOfficialHippoRAGError(f"{field} is invalid")
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise WikiSQLUAOOfficialHippoRAGError(f"{field} is invalid")


def _sequence(value: object, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise WikiSQLUAOOfficialHippoRAGError(
            f"{field} must be an array"
        )
    return value


def _semantic_item_payload(item: WikiSQLItem) -> dict[str, object]:
    return {
        "headers": list(item.headers),
        "item_id": item.item_id,
        "question": item.question,
        "rows": [list(row) for row in item.rows],
        "types": list(item.types),
    }


def _parse_semantic_item(
    value: object, *, item_ordinal: int, signed: bool
) -> WikiSQLItem:
    expected_keys = _ITEM_KEYS if signed else _ITEM_BASE_KEYS
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise WikiSQLUAOOfficialHippoRAGError(
            f"items[{item_ordinal}] shape drifted"
        )
    reject_forbidden_fields(value, path=f"$.items[{item_ordinal}]")
    item_id = _hex64(
        value.get("item_id"), f"items[{item_ordinal}].item_id"
    )
    question = _text(
        value.get("question"),
        f"items[{item_ordinal}].question",
        maximum=MAX_QUESTION_CHARACTERS,
    )
    raw_headers = _sequence(
        value.get("headers"), f"items[{item_ordinal}].headers"
    )
    raw_types = _sequence(
        value.get("types"), f"items[{item_ordinal}].types"
    )
    if (
        not MIN_COLUMN_COUNT <= len(raw_headers) <= MAX_COLUMN_COUNT
        or len(raw_headers) != len(raw_types)
    ):
        raise WikiSQLUAOOfficialHippoRAGError(
            f"items[{item_ordinal}] column contract drifted"
        )
    headers = tuple(
        _text(
            raw,
            f"items[{item_ordinal}].headers[{column_ordinal}]",
            maximum=MAX_HEADER_CHARACTERS,
        )
        for column_ordinal, raw in enumerate(raw_headers)
    )
    types = tuple(
        _text(
            raw,
            f"items[{item_ordinal}].types[{column_ordinal}]",
            maximum=16,
        )
        for column_ordinal, raw in enumerate(raw_types)
    )
    if any(value not in WIKISQL_COLUMN_TYPES for value in types):
        raise WikiSQLUAOOfficialHippoRAGError(
            f"items[{item_ordinal}] column type drifted"
        )
    raw_rows = _sequence(
        value.get("rows"), f"items[{item_ordinal}].rows"
    )
    if not MIN_ROW_COUNT <= len(raw_rows) <= MAX_ROW_COUNT:
        raise WikiSQLUAOOfficialHippoRAGError(
            f"items[{item_ordinal}] row count drifted"
        )
    rows: list[tuple[Cell, ...]] = []
    for row_ordinal, raw_row in enumerate(raw_rows):
        cells = _sequence(
            raw_row,
            f"items[{item_ordinal}].rows[{row_ordinal}]",
        )
        if len(cells) != len(headers):
            raise WikiSQLUAOOfficialHippoRAGError(
                f"items[{item_ordinal}].rows[{row_ordinal}] width drifted"
            )
        rows.append(
            tuple(
                _cell(
                    raw_cell,
                    (
                        f"items[{item_ordinal}].rows[{row_ordinal}]"
                        f"[{column_ordinal}]"
                    ),
                )
                for column_ordinal, raw_cell in enumerate(cells)
            )
        )
    provisional = WikiSQLItem(
        item_id=item_id,
        question=question,
        headers=headers,
        types=types,
        rows=tuple(rows),
        item_sha256="",
        row_corpus_sha256="",
    )
    item_sha256 = semantic_sha256(_semantic_item_payload(provisional))
    row_corpus_sha256 = semantic_sha256(
        list(serialize_rows(provisional))
    )
    if signed:
        if (
            value.get("item_sha256") != item_sha256
            or value.get("row_corpus_sha256") != row_corpus_sha256
        ):
            raise WikiSQLUAOOfficialHippoRAGError(
                f"items[{item_ordinal}] commitment drifted"
            )
    return WikiSQLItem(
        item_id=item_id,
        question=question,
        headers=headers,
        types=types,
        rows=tuple(rows),
        item_sha256=item_sha256,
        row_corpus_sha256=row_corpus_sha256,
    )


def serialize_row(item: WikiSQLItem, row_ordinal: int) -> str:
    """Return the exact cross-arm reality-core row serialization.

    Opaque item identity and physical ordinal remain only in contract metadata;
    neither is injected into the document text seen by HippoRAG.
    """

    if (
        isinstance(row_ordinal, bool)
        or not isinstance(row_ordinal, int)
        or not 0 <= row_ordinal < len(item.rows)
    ):
        raise WikiSQLUAOOfficialHippoRAGError(
            "row ordinal drifted"
        )
    try:
        table = reality.WikiSQLTable(
            table_id=item.item_id,
            header=item.headers,
            types=item.types,
            rows=item.rows,
        )
        return reality.serialize_table_row(table, row_ordinal)
    except reality.WikiSQLUAORealityError as exc:
        raise WikiSQLUAOOfficialHippoRAGError(
            "cross-arm row serialization drifted"
        ) from exc


def serialize_rows(item: WikiSQLItem) -> tuple[str, ...]:
    try:
        table = reality.WikiSQLTable(
            table_id=item.item_id,
            header=item.headers,
            types=item.types,
            rows=item.rows,
        )
        return reality.validated_retrieval_documents(table)
    except reality.WikiSQLUAORealityError as exc:
        raise WikiSQLUAOOfficialHippoRAGError(
            "shared row-document contract drifted"
        ) from exc


def input_payload(*, items: object) -> dict[str, object]:
    """Build the shared A_hold action-view pack used by all three arms.

    This helper exists for source-free qualification tests.  Formal execution
    consumes the source compiler's view pack directly, so no adapter-specific
    input envelope or controller-side translation exists.
    """

    rows = _sequence(items, "items")
    if not MIN_ITEM_COUNT <= len(rows) <= MAX_ITEM_COUNT:
        raise WikiSQLUAOOfficialHippoRAGError(
            "item count is outside the frozen bounds"
        )
    checked = tuple(
        _parse_semantic_item(raw, item_ordinal=ordinal, signed=False)
        for ordinal, raw in enumerate(rows)
    )
    identifiers = tuple(item.item_id for item in checked)
    if len(set(identifiers)) != len(identifiers):
        raise WikiSQLUAOOfficialHippoRAGError(
            "opaque item IDs must be unique"
        )
    try:
        return action_runtime.build_view_pack(
            block="A_hold",
            items=[
                {
                    "opaque_item_id": item.item_id,
                    "physical_rows": [list(row) for row in item.rows],
                    "question": item.question,
                    "table_header": list(item.headers),
                    "table_types": list(item.types),
                }
                for item in checked
            ],
        )
    except action_runtime.WikiSQLUAOActionRuntimeError as exc:
        raise WikiSQLUAOOfficialHippoRAGError(
            "shared action-view pack construction failed"
        ) from exc


def validate_input(value: object) -> tuple[WikiSQLItem, ...]:
    """Validate the exact shared A_hold view and derive private metadata."""

    if not isinstance(value, Mapping):
        raise WikiSQLUAOOfficialHippoRAGError(
            "input envelope drifted"
        )
    reject_forbidden_fields(value)
    try:
        views = action_runtime.decode_view_pack(
            value,
            expected_block="A_hold",
        )
    except action_runtime.WikiSQLUAOActionRuntimeError as exc:
        raise WikiSQLUAOOfficialHippoRAGError(
            "shared action-view pack drifted"
        ) from exc
    if not MIN_ITEM_COUNT <= len(views) <= MAX_ITEM_COUNT:
        raise WikiSQLUAOOfficialHippoRAGError(
            "item count is outside the frozen bounds"
        )
    checked: list[WikiSQLItem] = []
    for item_ordinal, view in enumerate(views):
        if (
            not MIN_COLUMN_COUNT
            <= len(view.header)
            <= MAX_COLUMN_COUNT
            or len(view.header) != len(view.types)
        ):
            raise WikiSQLUAOOfficialHippoRAGError(
                f"items[{item_ordinal}] column contract drifted"
            )
        headers = tuple(
            _text(
                value,
                f"items[{item_ordinal}].headers[{column_ordinal}]",
                maximum=MAX_HEADER_CHARACTERS,
            )
            for column_ordinal, value in enumerate(view.header)
        )
        types = tuple(
            _text(
                value,
                f"items[{item_ordinal}].types[{column_ordinal}]",
                maximum=16,
            )
            for column_ordinal, value in enumerate(view.types)
        )
        if any(value not in WIKISQL_COLUMN_TYPES for value in types):
            raise WikiSQLUAOOfficialHippoRAGError(
                f"items[{item_ordinal}] column type drifted"
            )
        rows = tuple(
            tuple(
                _cell(
                    value,
                    (
                        f"items[{item_ordinal}].rows[{row_ordinal}]"
                        f"[{column_ordinal}]"
                    ),
                )
                for column_ordinal, value in enumerate(row)
            )
            for row_ordinal, row in enumerate(view.rows)
        )
        provisional = WikiSQLItem(
            item_id=view.item_id,
            question=_text(
                view.question,
                f"items[{item_ordinal}].question",
                maximum=MAX_QUESTION_CHARACTERS,
            ),
            headers=headers,
            types=types,
            rows=rows,
            item_sha256="",
            row_corpus_sha256="",
        )
        item = WikiSQLItem(
            item_id=provisional.item_id,
            question=provisional.question,
            headers=provisional.headers,
            types=provisional.types,
            rows=provisional.rows,
            item_sha256=semantic_sha256(
                _semantic_item_payload(provisional)
            ),
            row_corpus_sha256=semantic_sha256(
                list(serialize_rows(provisional))
            ),
        )
        checked.append(item)
    return tuple(checked)


def parse_input(raw: bytes) -> tuple[dict[str, object], tuple[WikiSQLItem, ...]]:
    """Parse exact canonical private input bytes."""

    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WikiSQLUAOOfficialHippoRAGError(
            "worker input is invalid JSON"
        ) from exc
    if (
        not isinstance(value, dict)
        or canonical_json_bytes(value) != raw
    ):
        raise WikiSQLUAOOfficialHippoRAGError(
            "worker input is not canonical"
        )
    return value, validate_input(value)


def stable_top_k(
    *,
    retrieved_documents: object,
    retrieved_scores: object,
    document_to_ordinal: Mapping[str, int],
) -> tuple[int, ...]:
    """Return score-descending top five with stable row-ordinal ties."""

    if (
        isinstance(retrieved_documents, (str, bytes))
        or isinstance(retrieved_scores, (str, bytes))
    ):
        raise WikiSQLUAOOfficialHippoRAGError(
            "official result is malformed"
        )
    try:
        documents = list(retrieved_documents)  # type: ignore[arg-type]
        scores = list(retrieved_scores)  # type: ignore[arg-type]
    except TypeError as exc:
        raise WikiSQLUAOOfficialHippoRAGError(
            "official result is not iterable"
        ) from exc
    if (
        len(documents) != len(document_to_ordinal)
        or len(scores) != len(document_to_ordinal)
    ):
        raise WikiSQLUAOOfficialHippoRAGError(
            "official result omitted item-local rows"
        )
    seen: set[str] = set()
    ranked: list[tuple[float, int]] = []
    for document, score in zip(documents, scores):
        if (
            not isinstance(document, str)
            or document not in document_to_ordinal
            or document in seen
            or isinstance(score, bool)
            or not isinstance(score, Real)
            or not math.isfinite(float(score))
        ):
            raise WikiSQLUAOOfficialHippoRAGError(
                "official result row drifted"
            )
        seen.add(document)
        ranked.append((float(score), document_to_ordinal[document]))
    if seen != set(document_to_ordinal):
        raise WikiSQLUAOOfficialHippoRAGError(
            "official result candidate set drifted"
        )
    ranked.sort(key=lambda row: (-row[0], row[1]))
    result = tuple(row[1] for row in ranked[:TOP_K])
    if len(result) != TOP_K or len(set(result)) != TOP_K:
        raise WikiSQLUAOOfficialHippoRAGError(
            "official top-five rows drifted"
        )
    return result


def make_ranking_row(
    *,
    item: WikiSQLItem,
    item_ordinal: int,
    top5_row_ordinals: Sequence[int],
) -> dict[str, object]:
    values = tuple(top5_row_ordinals)
    if (
        len(values) != TOP_K
        or len(set(values)) != TOP_K
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < len(item.rows)
            for value in values
        )
    ):
        raise WikiSQLUAOOfficialHippoRAGError(
            "ranking row ordinals drifted"
        )
    base = {
        "item_id": item.item_id,
        "item_ordinal": _integer(
            item_ordinal, "ranking item ordinal"
        ),
        "schema": RANKING_ROW_SCHEMA,
        "top5_row_ordinals": list(values),
    }
    return {**base, "row_sha256": semantic_sha256(base)}


def make_index_receipt(
    *,
    item: WikiSQLItem,
    item_ordinal: int,
    index_tree_sha256: str,
    index_root_binding_sha256: str,
    file_count: int,
    byte_count: int,
    graph_node_count: int,
    graph_edge_count: int,
) -> dict[str, object]:
    base = {
        "byte_count": _integer(byte_count, "index byte count"),
        "file_count": _integer(file_count, "index file count"),
        "fresh_index": True,
        "graph_edge_count": _integer(
            graph_edge_count, "graph edge count"
        ),
        "graph_node_count": _integer(
            graph_node_count, "graph node count"
        ),
        "index_call_count": 1,
        "index_root_binding_sha256": _hex64(
            index_root_binding_sha256,
            "index root binding",
        ),
        "index_tree_sha256": _hex64(
            index_tree_sha256, "index tree SHA-256"
        ),
        "item_id": item.item_id,
        "item_ordinal": _integer(
            item_ordinal, "index item ordinal"
        ),
        "item_sha256": item.item_sha256,
        "retrieve_call_count": 1,
        "row_corpus_sha256": item.row_corpus_sha256,
        "row_count": len(item.rows),
        "schema": INDEX_RECEIPT_SCHEMA,
    }
    return {**base, "self_sha256": semantic_sha256(base)}


def _validate_ranking_row(
    value: object, *, item: WikiSQLItem, item_ordinal: int
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _RANKING_KEYS:
        raise WikiSQLUAOOfficialHippoRAGError(
            "ranking row shape drifted"
        )
    expected = make_ranking_row(
        item=item,
        item_ordinal=item_ordinal,
        top5_row_ordinals=value.get("top5_row_ordinals", ()),  # type: ignore[arg-type]
    )
    if dict(value) != expected:
        raise WikiSQLUAOOfficialHippoRAGError(
            "ranking row commitment drifted"
        )
    return expected


def _validate_index_receipt(
    value: object, *, item: WikiSQLItem, item_ordinal: int
) -> dict[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != _INDEX_RECEIPT_KEYS
        or value.get("schema") != INDEX_RECEIPT_SCHEMA
        or value.get("fresh_index") is not True
        or value.get("index_call_count") != 1
        or value.get("retrieve_call_count") != 1
    ):
        raise WikiSQLUAOOfficialHippoRAGError(
            "index receipt shape drifted"
        )
    expected = make_index_receipt(
        item=item,
        item_ordinal=item_ordinal,
        index_tree_sha256=value.get("index_tree_sha256"),  # type: ignore[arg-type]
        index_root_binding_sha256=value.get(  # type: ignore[arg-type]
            "index_root_binding_sha256"
        ),
        file_count=value.get("file_count"),  # type: ignore[arg-type]
        byte_count=value.get("byte_count"),  # type: ignore[arg-type]
        graph_node_count=value.get("graph_node_count"),  # type: ignore[arg-type]
        graph_edge_count=value.get("graph_edge_count"),  # type: ignore[arg-type]
    )
    if dict(value) != expected:
        raise WikiSQLUAOOfficialHippoRAGError(
            "index receipt commitment drifted"
        )
    return expected


def _runtime(item_count: int) -> dict[str, object]:
    return {
        "core_config_sha256": CORE_CONFIG_SHA256,
        "evaluator_call_count": 0,
        "fresh_index_per_item": True,
        "index_call_count": item_count,
        "network_call_count": 0,
        "official_hipporag_commit": OFFICIAL_HIPPORAG_COMMIT,
        "replay_count": 0,
        "retrieve_call_count": item_count,
        "retry_count": 0,
        "sequential_item_execution": True,
        "single_gpu_lane_count": 1,
        "top_k": TOP_K,
    }


def build_output(
    *,
    input_value: Mapping[str, object],
    rankings: Sequence[Mapping[str, object]],
    index_receipts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build the content-free, fully committed official-arm output."""

    items = validate_input(input_value)
    if len(rankings) != len(items) or len(index_receipts) != len(items):
        raise WikiSQLUAOOfficialHippoRAGError(
            "output item coverage drifted"
        )
    checked_rankings = [
        _validate_ranking_row(
            row, item=item, item_ordinal=ordinal
        )
        for ordinal, (row, item) in enumerate(zip(rankings, items))
    ]
    checked_receipts = [
        _validate_index_receipt(
            row, item=item, item_ordinal=ordinal
        )
        for ordinal, (row, item) in enumerate(
            zip(index_receipts, items)
        )
    ]
    base = {
        "action_view_pack_sha256": input_value["self_sha256"],
        "index_receipts": checked_receipts,
        "index_receipts_sha256": semantic_sha256(checked_receipts),
        "item_count": len(items),
        "official_hipporag_commit": OFFICIAL_HIPPORAG_COMMIT,
        "rankings": checked_rankings,
        "rankings_sha256": semantic_sha256(checked_rankings),
        "runtime": _runtime(len(items)),
        "schema": OUTPUT_SCHEMA,
    }
    return {**base, "self_sha256": semantic_sha256(base)}


def validate_output(
    value: object, *, expected_input: Mapping[str, object]
) -> dict[str, object]:
    """Validate a safe output against its exact private input commitment."""

    if (
        not isinstance(value, Mapping)
        or set(value) != _OUTPUT_KEYS
        or value.get("schema") != OUTPUT_SCHEMA
        or value.get("official_hipporag_commit")
        != OFFICIAL_HIPPORAG_COMMIT
    ):
        raise WikiSQLUAOOfficialHippoRAGError(
            "output envelope drifted"
        )
    rankings = _sequence(value.get("rankings"), "rankings")
    receipts = _sequence(
        value.get("index_receipts"), "index receipts"
    )
    expected = build_output(
        input_value=expected_input,
        rankings=rankings,  # type: ignore[arg-type]
        index_receipts=receipts,  # type: ignore[arg-type]
    )
    if dict(value) != expected:
        raise WikiSQLUAOOfficialHippoRAGError(
            "output self commitment drifted"
        )
    if (
        not isinstance(value.get("runtime"), Mapping)
        or set(value["runtime"]) != _RUNTIME_KEYS  # type: ignore[index]
    ):
        raise WikiSQLUAOOfficialHippoRAGError(
            "runtime receipt drifted"
        )
    return expected


def build_common_action_pack(
    *,
    expected_input: Mapping[str, object],
    native_output: Mapping[str, object],
) -> dict[str, object]:
    """Convert a fully validated native result to the shared three-arm pack."""

    validated = validate_output(
        native_output,
        expected_input=expected_input,
    )
    rankings = _sequence(validated.get("rankings"), "rankings")
    try:
        action_pack = action_runtime.build_action_pack(
            block="A_hold",
            arm="HippoRAG",
            action_view_pack_sha256=_hex64(
                expected_input.get("self_sha256"),
                "action view pack",
            ),
            items=[
                {
                    "opaque_item_id": row["item_id"],
                    "top5_row_ids": row["top5_row_ordinals"],
                }
                for row in rankings
            ],
        )
        action_runtime.decode_action_pack(
            action_pack,
            expected_block="A_hold",
            expected_arm="HippoRAG",
            expected_action_view_pack_sha256=expected_input[
                "self_sha256"
            ],  # type: ignore[arg-type]
        )
    except (
        KeyError,
        TypeError,
        action_runtime.WikiSQLUAOActionRuntimeError,
    ) as exc:
        raise WikiSQLUAOOfficialHippoRAGError(
            "common HippoRAG action pack construction failed"
        ) from exc
    return action_pack


def build_safe_receipt(
    *,
    expected_input: Mapping[str, object],
    native_output: Mapping[str, object],
    action_pack: Mapping[str, object],
) -> dict[str, object]:
    """Commit to private native evidence without disclosing item details."""

    validated = validate_output(
        native_output,
        expected_input=expected_input,
    )
    expected_action = build_common_action_pack(
        expected_input=expected_input,
        native_output=validated,
    )
    if dict(action_pack) != expected_action:
        raise WikiSQLUAOOfficialHippoRAGError(
            "safe receipt action pack binding drifted"
        )
    base = {
        "action_pack_sha256": action_pack["self_sha256"],
        "action_view_pack_sha256": expected_input["self_sha256"],
        "arm": "HippoRAG",
        "block": "A_hold",
        "index_receipts_sha256": validated[
            "index_receipts_sha256"
        ],
        "item_count": validated["item_count"],
        "native_output_sha256": validated["self_sha256"],
        "official_hipporag_commit": OFFICIAL_HIPPORAG_COMMIT,
        "rankings_sha256": validated["rankings_sha256"],
        "runtime": validated["runtime"],
        "schema": SAFE_RECEIPT_SCHEMA,
        "status": "passed",
        "study_id": action_runtime.STUDY_ID,
    }
    return {**base, "self_sha256": semantic_sha256(base)}


def validate_safe_receipt(
    value: object,
    *,
    expected_input: Mapping[str, object],
    native_output: Mapping[str, object],
    action_pack: Mapping[str, object],
) -> dict[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != _SAFE_RECEIPT_KEYS
        or value.get("schema") != SAFE_RECEIPT_SCHEMA
    ):
        raise WikiSQLUAOOfficialHippoRAGError(
            "safe aggregate receipt envelope drifted"
        )
    expected = build_safe_receipt(
        expected_input=expected_input,
        native_output=native_output,
        action_pack=action_pack,
    )
    if dict(value) != expected:
        raise WikiSQLUAOOfficialHippoRAGError(
            "safe aggregate receipt commitment drifted"
        )
    return expected


def parse_output(
    raw: bytes, *, expected_input: Mapping[str, object]
) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WikiSQLUAOOfficialHippoRAGError(
            "worker output is invalid JSON"
        ) from exc
    if (
        not isinstance(value, dict)
        or canonical_json_bytes(value) != raw
    ):
        raise WikiSQLUAOOfficialHippoRAGError(
            "worker output is not canonical"
        )
    return validate_output(value, expected_input=expected_input)


__all__ = [
    "CORE_CONFIG_SHA256",
    "FROZEN_CORE_CONFIG",
    "INDEX_RECEIPT_SCHEMA",
    "INPUT_SCHEMA",
    "MAX_ITEM_COUNT",
    "MAX_ROW_COUNT",
    "MIN_ITEM_COUNT",
    "MIN_ROW_COUNT",
    "OFFICIAL_HIPPORAG_COMMIT",
    "OUTPUT_SCHEMA",
    "ROW_DOCUMENT_SCHEMA",
    "SAFE_RECEIPT_SCHEMA",
    "TOP_K",
    "VERSION",
    "WIKISQL_COLUMN_TYPES",
    "WikiSQLItem",
    "WikiSQLUAOOfficialHippoRAGError",
    "build_common_action_pack",
    "build_output",
    "build_safe_receipt",
    "canonical_json_bytes",
    "input_payload",
    "make_index_receipt",
    "make_ranking_row",
    "parse_input",
    "parse_output",
    "reject_forbidden_fields",
    "semantic_sha256",
    "serialize_row",
    "serialize_rows",
    "stable_top_k",
    "validate_input",
    "validate_output",
    "validate_safe_receipt",
]
