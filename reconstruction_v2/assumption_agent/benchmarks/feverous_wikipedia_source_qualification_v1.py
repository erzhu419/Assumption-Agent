"""Exact, aggregate-only qualification of the FEVEROUS Wikipedia source.

This module deliberately separates two concerns:

* :class:`FeverousWikiResolver` resolves an annotation id against one exact
  ``wiki.id`` row and the official FEVEROUS page topology; and
* :func:`qualify_evidence_sets` emits only aggregate counts and hashes.

The resolver never derives a page from the content id alone.  For each
evidence reference the page authority is the single exact ``PAGE_title``
entry in that reference's annotation context.  Content/page drift and
context-member/page drift are counted separately.  No case folding,
normalisation, prefix lookup, or other fuzzy fallback is available.

The formal opener uses SQLite's immutable URI and enables ``query_only``.
Synthetic callers may supply their own connection, but qualification refuses
to run unless ``PRAGMA query_only`` is already enabled.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
from typing import Any


VERSION = "feverous_wikipedia_source_qualification_v1"
SCHEMA = VERSION

CONTENT_KINDS = (
    "sentence",
    "cell",
    "header_cell",
    "item",
    "table_caption",
)
CONTEXT_KINDS = CONTENT_KINDS + ("section", "title")

# Longest token first is a correctness property: ``header_cell`` must be
# attempted before ``cell`` and ``table_caption`` before any future suffix
# called ``caption``.
_KIND_ARITY = {
    "header_cell": 3,
    "table_caption": 1,
    "sentence": 1,
    "section": 1,
    "title": 0,
    "cell": 3,
    "item": 2,
}
_KIND_PARSE_ORDER = tuple(
    sorted(_KIND_ARITY, key=lambda value: (-len(value), value))
)
_TOP_LEVEL_RE = re.compile(r"(sentence|section|table|list)_([0-9]+)\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class FeverousWikipediaQualificationError(RuntimeError):
    """The source, annotation topology, or read-only boundary is invalid."""


@dataclass(frozen=True)
class ElementId:
    """A strictly parsed FEVEROUS annotation element identifier."""

    page: str
    kind: str
    indices: tuple[int, ...]
    local_id: str


@dataclass(frozen=True)
class WikiElement:
    """One resolved element; values stay private and never enter receipts."""

    page: str
    local_id: str
    kind: str
    indices: tuple[int, ...]
    value: str
    is_header: bool | None = None
    row_span: int | None = None
    column_span: int | None = None


@dataclass(frozen=True)
class ElementResolution:
    status: str
    element: WikiElement | None = None


@dataclass(frozen=True)
class _PageIndex:
    elements: Mapping[str, tuple[WikiElement, ...]]
    row_span_gt_one_count: int
    column_span_gt_one_count: int
    cell_count: int


@dataclass(frozen=True)
class _PageLookup:
    status: str
    page: _PageIndex | None = None


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise FeverousWikipediaQualificationError(
            "value is not canonical JSON"
        ) from exc


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _decode_strict_json(raw: str | bytes | bytearray | memoryview) -> Any:
    """Decode UTF-8 JSON while rejecting duplicates and non-finite values."""

    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, (bytes, bytearray, memoryview)):
        try:
            text = bytes(raw).decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise FeverousWikipediaQualificationError(
                "SQLite data is not strict UTF-8"
            ) from exc
    else:
        raise FeverousWikipediaQualificationError(
            "SQLite data is not JSON text"
        )

    def object_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise FeverousWikipediaQualificationError(
                    "duplicate JSON object key"
                )
            output[key] = value
        return output

    def reject_constant(_value: str) -> None:
        raise FeverousWikipediaQualificationError(
            "JSON contains a non-finite constant"
        )

    try:
        return json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except FeverousWikipediaQualificationError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise FeverousWikipediaQualificationError(
            "SQLite data is not strict JSON"
        ) from exc


def parse_element_id(value: str) -> ElementId:
    """Split ``PAGE_KIND_INDICES`` by the longest valid suffix.

    Page identifiers may contain arbitrary underscores and kind-looking text.
    Each kind has a fixed arity; a shorter suffix is never accepted as a
    recovery for a malformed longer suffix.
    """

    if not isinstance(value, str) or not value or "\x00" in value:
        raise FeverousWikipediaQualificationError(
            "element id must be a nonempty safe string"
        )
    for kind in _KIND_PARSE_ORDER:
        arity = _KIND_ARITY[kind]
        if arity == 0:
            marker = f"_{kind}"
            if not value.endswith(marker):
                continue
            page = value[: -len(marker)]
            if not page:
                break
            return ElementId(page=page, kind=kind, indices=(), local_id=kind)

        marker = f"_{kind}_"
        start = value.rfind(marker)
        if start < 1:
            continue
        suffix = value[start + len(marker) :]
        tokens = suffix.split("_")
        if len(tokens) != arity or any(
            not token or not token.isascii() or not token.isdecimal()
            for token in tokens
        ):
            # This marker may occur in a page title.  Keep checking longer or
            # different valid suffixes, but never relax this kind's arity.
            continue
        page = value[:start]
        indices = tuple(int(token) for token in tokens)
        local_id = f"{kind}_{'_'.join(tokens)}"
        return ElementId(
            page=page,
            kind=kind,
            indices=indices,
            local_id=local_id,
        )
    raise FeverousWikipediaQualificationError(
        "element id has no fixed-arity FEVEROUS suffix"
    )


def _parse_local_id(value: str) -> ElementId:
    parsed = parse_element_id(f"__local_page___{value}")
    if parsed.page != "__local_page__":
        raise FeverousWikipediaQualificationError(
            "page element id is not a canonical local id"
        )
    return parsed


def _require_string(value: Any) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise FeverousWikipediaQualificationError(
            "Wikipedia text field has the wrong type"
        )
    return value


def _require_positive_int(value: Any) -> int:
    """Decode the official dump's decimal-string table span fields."""

    if type(value) is int:
        decoded = value
    elif (
        isinstance(value, str)
        and value
        and value.isascii()
        and value.isdecimal()
    ):
        decoded = int(value)
    else:
        raise FeverousWikipediaQualificationError(
            "Wikipedia span field must be a positive decimal integer"
        )
    if decoded < 1:
        raise FeverousWikipediaQualificationError(
            "Wikipedia span field must be a positive decimal integer"
        )
    return decoded


def _parse_page(page_id: str, raw: Any) -> _PageIndex:
    payload = _decode_strict_json(raw)
    if not isinstance(payload, Mapping):
        raise FeverousWikipediaQualificationError(
            "Wikipedia page root must be an object"
        )
    title = _require_string(payload.get("title"))
    if title != page_id:
        raise FeverousWikipediaQualificationError(
            "Wikipedia page title does not exactly match wiki.id"
        )
    order = payload.get("order")
    if not isinstance(order, list) or any(
        not isinstance(value, str) for value in order
    ):
        raise FeverousWikipediaQualificationError(
            "Wikipedia page order must be a string list"
        )
    if len(order) != len(set(order)):
        raise FeverousWikipediaQualificationError(
            "Wikipedia page order contains a duplicate element"
        )

    elements: dict[str, list[WikiElement]] = defaultdict(list)
    elements["title"].append(
        WikiElement(
            page=page_id,
            local_id="title",
            kind="title",
            indices=(),
            value=title,
        )
    )
    row_span_gt_one_count = 0
    column_span_gt_one_count = 0
    cell_count = 0

    for top_level_id in order:
        match = _TOP_LEVEL_RE.fullmatch(top_level_id)
        if match is None or top_level_id not in payload:
            raise FeverousWikipediaQualificationError(
                "Wikipedia order references an invalid element"
            )
        kind = match.group(1)
        top_index = int(match.group(2))
        value = payload[top_level_id]

        if kind == "sentence":
            sentence = _require_string(value)
            elements[top_level_id].append(
                WikiElement(
                    page=page_id,
                    local_id=top_level_id,
                    kind="sentence",
                    indices=(top_index,),
                    value=sentence,
                )
            )
            continue

        if kind == "section":
            if not isinstance(value, Mapping):
                raise FeverousWikipediaQualificationError(
                    "Wikipedia section must be an object"
                )
            section = _require_string(value.get("value"))
            level = value.get("level")
            if type(level) is not int or level < 0:
                raise FeverousWikipediaQualificationError(
                    "Wikipedia section level is invalid"
                )
            elements[top_level_id].append(
                WikiElement(
                    page=page_id,
                    local_id=top_level_id,
                    kind="section",
                    indices=(top_index,),
                    value=section,
                )
            )
            continue

        if not isinstance(value, Mapping):
            raise FeverousWikipediaQualificationError(
                "Wikipedia structured element must be an object"
            )

        if kind == "table":
            if not isinstance(value.get("type"), str):
                raise FeverousWikipediaQualificationError(
                    "Wikipedia table type is invalid"
                )
            rows = value.get("table")
            if not isinstance(rows, list):
                raise FeverousWikipediaQualificationError(
                    "Wikipedia table must contain a row list"
                )
            if "caption" in value:
                caption = _require_string(value["caption"])
                caption_id = f"table_caption_{top_index}"
                elements[caption_id].append(
                    WikiElement(
                        page=page_id,
                        local_id=caption_id,
                        kind="table_caption",
                        indices=(top_index,),
                        value=caption,
                    )
                )
            for row in rows:
                if not isinstance(row, list):
                    raise FeverousWikipediaQualificationError(
                        "Wikipedia table row must be a list"
                    )
                for cell in row:
                    if not isinstance(cell, Mapping):
                        raise FeverousWikipediaQualificationError(
                            "Wikipedia table cell must be an object"
                        )
                    local_id = cell.get("id")
                    if not isinstance(local_id, str):
                        raise FeverousWikipediaQualificationError(
                            "Wikipedia table cell id is invalid"
                        )
                    parsed = _parse_local_id(local_id)
                    if parsed.kind not in {"cell", "header_cell"}:
                        raise FeverousWikipediaQualificationError(
                            "Wikipedia table cell id has the wrong kind"
                        )
                    if parsed.indices[0] != top_index:
                        raise FeverousWikipediaQualificationError(
                            "Wikipedia table cell id has the wrong table index"
                        )
                    cell_value = _require_string(cell.get("value"))
                    is_header = cell.get("is_header")
                    if type(is_header) is not bool:
                        raise FeverousWikipediaQualificationError(
                            "Wikipedia table header flag is invalid"
                        )
                    row_span = _require_positive_int(cell.get("row_span"))
                    column_span = _require_positive_int(
                        cell.get("column_span")
                    )
                    elements[local_id].append(
                        WikiElement(
                            page=page_id,
                            local_id=local_id,
                            kind=parsed.kind,
                            indices=parsed.indices,
                            value=cell_value,
                            is_header=is_header,
                            row_span=row_span,
                            column_span=column_span,
                        )
                    )
                    cell_count += 1
                    row_span_gt_one_count += int(row_span > 1)
                    column_span_gt_one_count += int(column_span > 1)
            continue

        if not isinstance(value.get("type"), str):
            raise FeverousWikipediaQualificationError(
                "Wikipedia list type is invalid"
            )
        items = value.get("list")
        if not isinstance(items, list):
            raise FeverousWikipediaQualificationError(
                "Wikipedia list must contain an item list"
            )
        for item in items:
            if not isinstance(item, Mapping):
                raise FeverousWikipediaQualificationError(
                    "Wikipedia list item must be an object"
                )
            local_id = item.get("id")
            if not isinstance(local_id, str):
                raise FeverousWikipediaQualificationError(
                    "Wikipedia list item id is invalid"
                )
            parsed = _parse_local_id(local_id)
            if parsed.kind != "item" or parsed.indices[0] != top_index:
                raise FeverousWikipediaQualificationError(
                    "Wikipedia list item id has the wrong topology"
                )
            item_value = _require_string(item.get("value"))
            level = item.get("level")
            if type(level) is not int or level < 0:
                raise FeverousWikipediaQualificationError(
                    "Wikipedia list item level is invalid"
                )
            if "type" in item and not isinstance(item["type"], str):
                raise FeverousWikipediaQualificationError(
                    "Wikipedia nested-list type is invalid"
                )
            elements[local_id].append(
                WikiElement(
                    page=page_id,
                    local_id=local_id,
                    kind="item",
                    indices=parsed.indices,
                    value=item_value,
                )
            )

    return _PageIndex(
        elements={key: tuple(values) for key, values in elements.items()},
        row_span_gt_one_count=row_span_gt_one_count,
        column_span_gt_one_count=column_span_gt_one_count,
        cell_count=cell_count,
    )


def _require_query_only(connection: sqlite3.Connection) -> None:
    try:
        observed = connection.execute("PRAGMA query_only").fetchone()
    except sqlite3.Error as exc:
        raise FeverousWikipediaQualificationError(
            "SQLite query_only boundary is unavailable"
        ) from exc
    if observed != (1,):
        raise FeverousWikipediaQualificationError(
            "SQLite connection is not query_only"
        )


def _require_wiki_schema(connection: sqlite3.Connection) -> dict[str, Any]:
    try:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
        columns = connection.execute("PRAGMA table_info(wiki)").fetchall()
    except sqlite3.Error as exc:
        raise FeverousWikipediaQualificationError(
            "SQLite wiki schema is unavailable"
        ) from exc
    table_names = [row[0] for row in tables if len(row) == 1]
    column_names = [row[1] for row in columns if len(row) >= 2]
    if "wiki" not in table_names or not {"id", "data"}.issubset(column_names):
        raise FeverousWikipediaQualificationError(
            "SQLite wiki(id, data) schema is absent"
        )
    return {
        "table_count": len(table_names),
        "table_name_set_sha256": _stable_hash(sorted(table_names)),
        "wiki_column_count": len(column_names),
        "wiki_column_name_set_sha256": _stable_hash(sorted(column_names)),
    }


def open_immutable_wiki_db(path: str | os.PathLike[str]) -> sqlite3.Connection:
    """Open a regular SQLite file through ``mode=ro&immutable=1``."""

    source = Path(path)
    try:
        source_stat = source.lstat()
    except OSError as exc:
        raise FeverousWikipediaQualificationError(
            "SQLite source cannot be stated"
        ) from exc
    if stat.S_ISLNK(source_stat.st_mode) or not stat.S_ISREG(source_stat.st_mode):
        raise FeverousWikipediaQualificationError(
            "SQLite source must be a regular non-symlink file"
        )
    uri = f"{source.resolve().as_uri()}?mode=ro&immutable=1"
    try:
        connection = sqlite3.connect(uri, uri=True, isolation_level=None)
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA trusted_schema = OFF")
        _require_query_only(connection)
    except (sqlite3.Error, FeverousWikipediaQualificationError) as exc:
        try:
            connection.close()
        except (UnboundLocalError, sqlite3.Error):
            pass
        raise FeverousWikipediaQualificationError(
            "immutable SQLite open failed"
        ) from exc
    return connection


class FeverousWikiResolver:
    """Exact page and element resolver over a query-only FEVEROUS database."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        _require_query_only(connection)
        self.connection = connection
        self.schema_receipt = _require_wiki_schema(connection)
        self._pages: dict[str, _PageLookup] = {}
        self.cache_hit_count = 0

    def page(self, page_id: str) -> _PageLookup:
        if page_id in self._pages:
            self.cache_hit_count += 1
            return self._pages[page_id]
        try:
            rows = self.connection.execute(
                "SELECT id, data FROM wiki "
                "WHERE id COLLATE BINARY = ? COLLATE BINARY LIMIT 2",
                (page_id,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise FeverousWikipediaQualificationError(
                "exact SQLite page lookup failed"
            ) from exc
        exact_rows = [row for row in rows if len(row) == 2 and row[0] == page_id]
        if not exact_rows:
            result = _PageLookup("missing")
        elif len(exact_rows) != 1:
            result = _PageLookup("ambiguous")
        else:
            try:
                index = _parse_page(page_id, exact_rows[0][1])
            except FeverousWikipediaQualificationError as exc:
                message = str(exc)
                status = (
                    "invalid_json"
                    if "JSON" in message or "UTF-8" in message
                    else "invalid_topology"
                )
                result = _PageLookup(status)
            else:
                result = _PageLookup("resolved", index)
        self._pages[page_id] = result
        return result

    @property
    def lookup_count(self) -> int:
        return len(self._pages)

    def resolve_exact(
        self,
        full_id: str,
        *,
        context_page: str,
    ) -> ElementResolution:
        """Resolve one full id only within its exact context-title page."""

        try:
            parsed = parse_element_id(full_id)
        except FeverousWikipediaQualificationError:
            return ElementResolution("invalid_id")
        if parsed.page != context_page:
            return ElementResolution("wrong_page")
        lookup = self.page(context_page)
        if lookup.status != "resolved" or lookup.page is None:
            return ElementResolution(f"page_{lookup.status}")
        candidates = lookup.page.elements.get(parsed.local_id, ())
        if not candidates:
            return ElementResolution("missing")
        if len(candidates) != 1:
            return ElementResolution("ambiguous")
        element = candidates[0]
        if parsed.kind == "header_cell" and element.is_header is not True:
            return ElementResolution("wrong_header", element)
        if parsed.kind == "cell" and element.is_header is not False:
            return ElementResolution("wrong_cell_header_flag", element)
        return ElementResolution("resolved", element)


def _all_zero_counter(keys: Sequence[str]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for key in keys:
        counter[key] = 0
    return counter


def _counter_payload(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _safe_binding(
    *,
    database_size_bytes: int | None,
    database_sha256: str | None,
    archive_size_bytes: int | None,
    archive_sha256: str | None,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, value in (
        ("database_size_bytes", database_size_bytes),
        ("archive_size_bytes", archive_size_bytes),
    ):
        if value is not None:
            if type(value) is not int or value < 0:
                raise FeverousWikipediaQualificationError(
                    "source size binding is invalid"
                )
            output[name] = value
    for name, value in (
        ("database_sha256", database_sha256),
        ("archive_sha256", archive_sha256),
    ):
        if value is not None:
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise FeverousWikipediaQualificationError(
                    "source SHA-256 binding is invalid"
                )
            output[name] = value
    return output


def qualify_evidence_sets(
    evidence_sets: Sequence[Mapping[str, Any]],
    connection: sqlite3.Connection,
    *,
    database_size_bytes: int | None = None,
    database_sha256: str | None = None,
    archive_size_bytes: int | None = None,
    archive_sha256: str | None = None,
) -> dict[str, Any]:
    """Resolve evidence/context topology and return a content-free receipt."""

    if not isinstance(evidence_sets, Sequence) or isinstance(
        evidence_sets, (str, bytes, bytearray)
    ):
        raise FeverousWikipediaQualificationError(
            "evidence_sets must be a sequence"
        )
    resolver = FeverousWikiResolver(connection)

    content_kinds = _all_zero_counter(CONTENT_KINDS)
    context_kinds = _all_zero_counter(CONTEXT_KINDS)
    content_statuses: Counter[str] = Counter()
    context_statuses: Counter[str] = Counter()
    page_ids: set[str] = set()
    reference_ids: set[str] = set()

    evidence_set_count = 0
    content_reference_count = 0
    context_reference_count = 0
    duplicate_content_reference_count = 0
    exact_context_key_count = 0
    missing_content_context_key_count = 0
    orphan_context_key_count = 0
    missing_title_context_count = 0
    ambiguous_title_context_count = 0
    content_title_page_drift_count = 0
    context_member_title_page_drift_count = 0
    invalid_content_id_count = 0
    invalid_context_id_count = 0

    for evidence_set in evidence_sets:
        evidence_set_count += 1
        if not isinstance(evidence_set, Mapping):
            raise FeverousWikipediaQualificationError(
                "evidence set must be an object"
            )
        content = evidence_set.get("content")
        context = evidence_set.get("context")
        if not isinstance(content, list) or any(
            not isinstance(value, str) for value in content
        ):
            raise FeverousWikipediaQualificationError(
                "evidence content must be a string list"
            )
        if not isinstance(context, Mapping) or any(
            not isinstance(key, str) for key in context
        ):
            raise FeverousWikipediaQualificationError(
                "evidence context must be an object"
            )
        duplicate_content_reference_count += len(content) - len(set(content))
        content_keys = set(content)
        orphan_context_key_count += sum(key not in content_keys for key in context)

        for content_id in content:
            content_reference_count += 1
            reference_ids.add(content_id)
            try:
                parsed_content = parse_element_id(content_id)
            except FeverousWikipediaQualificationError:
                invalid_content_id_count += 1
                content_statuses["invalid_id"] += 1
                parsed_content = None
            else:
                if parsed_content.kind not in CONTENT_KINDS:
                    invalid_content_id_count += 1
                    content_statuses["invalid_content_kind"] += 1
                    parsed_content = None
                else:
                    content_kinds[parsed_content.kind] += 1

            if content_id not in context:
                missing_content_context_key_count += 1
                missing_title_context_count += 1
                if parsed_content is not None:
                    content_statuses["no_context"] += 1
                continue
            exact_context_key_count += 1
            context_values = context[content_id]
            if not isinstance(context_values, list) or any(
                not isinstance(value, str) for value in context_values
            ):
                raise FeverousWikipediaQualificationError(
                    "context value must be a string list"
                )

            parsed_context: list[tuple[str, ElementId | None]] = []
            title_ids: list[ElementId] = []
            for context_id in context_values:
                context_reference_count += 1
                reference_ids.add(context_id)
                try:
                    parsed = parse_element_id(context_id)
                except FeverousWikipediaQualificationError:
                    invalid_context_id_count += 1
                    context_statuses["invalid_id"] += 1
                    parsed_context.append((context_id, None))
                    continue
                if parsed.kind not in CONTEXT_KINDS:
                    invalid_context_id_count += 1
                    context_statuses["invalid_context_kind"] += 1
                    parsed_context.append((context_id, None))
                    continue
                context_kinds[parsed.kind] += 1
                parsed_context.append((context_id, parsed))
                if parsed.kind == "title":
                    title_ids.append(parsed)

            if not title_ids:
                missing_title_context_count += 1
                if parsed_content is not None:
                    content_statuses["no_title_context"] += 1
                for _context_id, parsed in parsed_context:
                    if parsed is not None:
                        context_statuses["no_title_context"] += 1
                continue
            if len(title_ids) != 1:
                ambiguous_title_context_count += 1
                if parsed_content is not None:
                    content_statuses["ambiguous_title_context"] += 1
                for _context_id, parsed in parsed_context:
                    if parsed is not None:
                        context_statuses["ambiguous_title_context"] += 1
                continue

            context_page = title_ids[0].page
            page_ids.add(context_page)
            if parsed_content is not None:
                if parsed_content.page != context_page:
                    content_title_page_drift_count += 1
                resolution = resolver.resolve_exact(
                    content_id,
                    context_page=context_page,
                )
                content_statuses[resolution.status] += 1

            for context_id, parsed in parsed_context:
                if parsed is None:
                    continue
                if parsed.page != context_page:
                    context_member_title_page_drift_count += 1
                resolution = resolver.resolve_exact(
                    context_id,
                    context_page=context_page,
                )
                context_statuses[resolution.status] += 1

    page_statuses: Counter[str] = Counter()
    row_span_gt_one_count = 0
    column_span_gt_one_count = 0
    parsed_cell_count = 0
    for lookup in resolver._pages.values():
        page_statuses[lookup.status] += 1
        if lookup.page is not None:
            row_span_gt_one_count += lookup.page.row_span_gt_one_count
            column_span_gt_one_count += lookup.page.column_span_gt_one_count
            parsed_cell_count += lookup.page.cell_count

    invalid_total = sum(
        (
            duplicate_content_reference_count,
            missing_content_context_key_count,
            orphan_context_key_count,
            missing_title_context_count,
            ambiguous_title_context_count,
            content_title_page_drift_count,
            context_member_title_page_drift_count,
            invalid_content_id_count,
            invalid_context_id_count,
            sum(
                count
                for status, count in content_statuses.items()
                if status != "resolved"
            ),
            sum(
                count
                for status, count in context_statuses.items()
                if status != "resolved"
            ),
            sum(
                count
                for status, count in page_statuses.items()
                if status != "resolved"
            ),
        )
    )

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "version": "v1",
        "status": (
            "passed_exact_source_qualification_no_selection"
            if invalid_total == 0
            else "failed_exact_source_qualification_no_selection"
        ),
        "source_binding": _safe_binding(
            database_size_bytes=database_size_bytes,
            database_sha256=database_sha256,
            archive_size_bytes=archive_size_bytes,
            archive_sha256=archive_sha256,
        ),
        "sqlite_boundary": {
            "formal_opener_immutable_uri": True,
            "query_only_observed": True,
            **resolver.schema_receipt,
        },
        "evidence_aggregate": {
            "evidence_set_count": evidence_set_count,
            "content_reference_count": content_reference_count,
            "context_reference_count": context_reference_count,
            "duplicate_content_reference_count": (
                duplicate_content_reference_count
            ),
            "content_kind_counts": _counter_payload(content_kinds),
            "context_kind_counts": _counter_payload(context_kinds),
            "referenced_page_set_count": len(page_ids),
            "referenced_page_set_sha256": _stable_hash(sorted(page_ids)),
            "reference_id_set_count": len(reference_ids),
            "reference_id_set_sha256": _stable_hash(sorted(reference_ids)),
        },
        "context_exactness": {
            "exact_context_key_count": exact_context_key_count,
            "missing_content_context_key_count": (
                missing_content_context_key_count
            ),
            "orphan_context_key_count": orphan_context_key_count,
            "missing_title_context_count": missing_title_context_count,
            "ambiguous_title_context_count": ambiguous_title_context_count,
            "content_title_page_drift_count": (
                content_title_page_drift_count
            ),
            "context_member_title_page_drift_count": (
                context_member_title_page_drift_count
            ),
            "fuzzy_lookup_or_repair_count": 0,
        },
        "page_resolution": {
            "exact_lookup_count": resolver.lookup_count,
            "cache_hit_count": resolver.cache_hit_count,
            "status_counts": _counter_payload(page_statuses),
        },
        "element_resolution": {
            "content_status_counts": _counter_payload(content_statuses),
            "context_status_counts": _counter_payload(context_statuses),
            "content_missing_count": content_statuses["missing"],
            "content_ambiguous_count": content_statuses["ambiguous"],
            "content_wrong_header_count": content_statuses["wrong_header"],
            "context_missing_count": context_statuses["missing"],
            "context_ambiguous_count": context_statuses["ambiguous"],
            "context_wrong_header_count": context_statuses["wrong_header"],
            "invalid_content_id_count": invalid_content_id_count,
            "invalid_context_id_count": invalid_context_id_count,
            "parsed_cell_count": parsed_cell_count,
            "row_span_gt_one_cell_count": row_span_gt_one_count,
            "column_span_gt_one_cell_count": column_span_gt_one_count,
            "cell_resolution_uses_exact_cell_id_not_coordinates": True,
        },
        "claim_boundary": {
            "claim_label_challenge_or_utility_read": False,
            "cohort_or_candidate_selected": False,
            "identifiers_titles_or_text_serialized": False,
            "per_page_or_per_item_digest_serialized": False,
            "aggregate_counts_and_hashes_only": True,
        },
    }
    receipt["qualification_sha256"] = _stable_hash(receipt)
    return receipt


def verify_receipt(receipt: Mapping[str, Any]) -> bool:
    """Verify the receipt self-hash without mutating the caller's mapping."""

    if not isinstance(receipt, Mapping):
        return False
    body = dict(receipt)
    declared = body.pop("qualification_sha256", None)
    return (
        isinstance(declared, str)
        and _SHA256_RE.fullmatch(declared) is not None
        and declared == _stable_hash(body)
    )


__all__ = [
    "CONTENT_KINDS",
    "CONTEXT_KINDS",
    "ElementId",
    "ElementResolution",
    "FeverousWikiResolver",
    "FeverousWikipediaQualificationError",
    "SCHEMA",
    "VERSION",
    "WikiElement",
    "open_immutable_wiki_db",
    "parse_element_id",
    "qualify_evidence_sets",
    "verify_receipt",
]
