"""Deterministic FEVEROUS atomic-corpus and claim-facet compilation.

This module is deliberately source-local and outcome-blind.  It accepts one
official FEVEROUS Wikipedia page at a time and produces the exact text shared
by RAW, HippoRAG, and Agent plus a separate typed sidecar.  It never accepts a
claim record: :func:`compile_claim_facets` accepts only the claim string and
NER offsets over its normalized form, so labels, challenge names, evidence
maps, block identities, and gold data cannot enter the facet view.

The serialization is target first.  A tokenizer may therefore apply the
frozen tail-truncation rule without deleting context before the atomic target.
No tokenizer or model is imported here, which keeps this compiler suitable for
synthetic, non-scoring implementation qualification.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import re
import unicodedata
from typing import Any

from assumption_agent.benchmarks.feverous_wikipedia_source_qualification_v1 import (
    FeverousWikipediaQualificationError,
    parse_element_id,
)


VERSION = "feverous_atomic_corpus_v1"
IDENTITY_ENUMERATOR_VERSION = "feverous_atomic_identity_enumerator_v1"
IDENTITY_COMMITMENT_SCHEMA = "feverous_atomic_identity_commitment_v1"

ATOMIC_UNIT_TYPES = (
    "sentence",
    "item",
    "cell",
    "header_cell",
    "table_caption",
)
ARM_IDS = ("RAW", "HippoRAG", "Agent")
MAXIMUM_MODEL_TOKENS = 256
ENTITY_FACET_LIMIT = 4
NUMERIC_OR_DATE_FACET_LIMIT = 2
RELATION_CLAUSE_FACET_LIMIT = 2

_TOP_LEVEL_RE = re.compile(r"(sentence|section|table|list)_([0-9]+)\Z")
_CLAUSE_DELIMITER_RE = re.compile(r"[,;:.!?\u060c\u061b\u3002\uff0c\uff1b\uff1a\uff01\uff1f]+")
_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
_NUMERIC_OR_DATE_RE = re.compile(
    rf"(?<!\w)(?:{_MONTH}\s+\d{{4}})(?!\w)"
    rf"|(?<!\w)(?:{_MONTH}\s+\d{{1,2}}(?:\s*,\s*\d{{4}})?)(?!\w)"
    rf"|(?<!\w)(?:\d{{1,2}}\s+{_MONTH}(?:\s+\d{{4}})?)(?!\w)"
    r"|(?<!\w)(?:\d{4}[-/]\d{1,2}[-/]\d{1,2})(?!\w)"
    r"|(?<!\w)(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})(?!\w)"
    r"|(?<!\w)(?:[$\u00a3\u20ac]\s*)?[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:\.\d+)?(?:%|st|nd|rd|th)?(?!\w)",
    flags=re.IGNORECASE,
)


class FeverousAtomicCorpusError(ValueError):
    """The page, atomic target, arm view, or claim view is invalid."""


@dataclass(frozen=True)
class NerSpan:
    """One NER character span over the normalized claim.

    No entity label or source-record field is accepted.  The fixed NER model's
    only permitted output at this boundary is a half-open character interval.
    """

    start: int
    end: int


@dataclass(frozen=True)
class ClaimFacet:
    kind: str
    text: str
    source_start: int
    source_end: int


@dataclass(frozen=True)
class CompiledClaimFacets:
    normalized_claim: str
    facets: tuple[ClaimFacet, ...]

    def of_kind(self, kind: str) -> tuple[ClaimFacet, ...]:
        return tuple(facet for facet in self.facets if facet.kind == kind)


@dataclass(frozen=True)
class AtomicSidecar:
    """Agent-only topology for one byte-identical shared corpus unit."""

    page: str
    local_id: str
    unit_type: str
    coordinates: tuple[int, ...]
    section_ids: tuple[str, ...]
    section_path: tuple[str, ...]
    official_ordinal: int
    previous_atomic_local_id: str | None = None
    next_atomic_local_id: str | None = None
    table_id: str | None = None
    table_kind: str | None = None
    table_caption: str | None = None
    row_span: int | None = None
    column_span: int | None = None
    applicable_row_header_ids: tuple[str, ...] = ()
    applicable_column_header_ids: tuple[str, ...] = ()
    list_id: str | None = None
    list_ancestor_ids: tuple[str, ...] = ()
    linearizer_version: str = VERSION


@dataclass(frozen=True)
class AtomicUnit:
    target: str
    text: str
    text_utf8: bytes
    sidecar: AtomicSidecar

    def bytes_for_arm(self, arm_id: str) -> bytes:
        """Return the single shared byte string; sidecars are not embedded."""

        if arm_id not in ARM_IDS:
            raise FeverousAtomicCorpusError(f"unknown corpus arm: {arm_id!r}")
        return self.text_utf8


@dataclass(frozen=True)
class AtomicIdentity:
    """Lightweight source-order identity available before full linearization."""

    page: str
    local_id: str
    unit_type: str
    official_ordinal: int
    normalized_target: str
    target_sha256: str


@dataclass(frozen=True)
class PageIdentityEnumeration:
    """One page's lightweight identities and a content-free commitment seam."""

    page: str
    identities: tuple[AtomicIdentity, ...]
    excluded_empty_local_ids: tuple[str, ...]

    def commitment(self) -> dict[str, int | str]:
        payload = {
            "enumerator_version": IDENTITY_ENUMERATOR_VERSION,
            "excluded_empty_local_ids": list(self.excluded_empty_local_ids),
            "identities": [
                {
                    "local_id": row.local_id,
                    "official_ordinal": row.official_ordinal,
                    "page": row.page,
                    "target_sha256": row.target_sha256,
                    "unit_type": row.unit_type,
                }
                for row in self.identities
            ],
        }
        raw = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            "enumerator_version": IDENTITY_ENUMERATOR_VERSION,
            "excluded_empty_count": len(self.excluded_empty_local_ids),
            "identity_count": len(self.identities),
            "identity_enumeration_sha256": hashlib.sha256(raw).hexdigest(),
            "schema": IDENTITY_COMMITMENT_SCHEMA,
        }


@dataclass(frozen=True)
class PageCompilation:
    page: str
    units: tuple[AtomicUnit, ...]
    excluded_empty_local_ids: tuple[str, ...]


@dataclass(frozen=True)
class _Section:
    local_id: str
    value: str
    level: int


@dataclass(frozen=True)
class _TableCell:
    local_id: str
    kind: str
    coordinates: tuple[int, int, int]
    value: str
    is_header: bool
    row_span: int
    column_span: int
    source_ordinal: int

    @property
    def row(self) -> int:
        return self.coordinates[1]

    @property
    def column(self) -> int:
        return self.coordinates[2]


@dataclass(frozen=True)
class _DraftUnit:
    target: str
    parts: tuple[tuple[str, str], ...]
    sidecar: AtomicSidecar


def normalize_surface(value: str) -> str:
    """Apply NFKC, collapse all Unicode whitespace, and preserve case."""

    if not isinstance(value, str) or "\x00" in value:
        raise FeverousAtomicCorpusError("text must be a safe string")
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _require_nonempty_surface(value: str, *, field: str) -> str:
    normalized = normalize_surface(value)
    if not normalized:
        raise FeverousAtomicCorpusError(f"{field} is empty after normalization")
    return normalized


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FeverousAtomicCorpusError(f"{field} must be an object")
    return value


def _require_nonnegative_int(value: Any, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise FeverousAtomicCorpusError(f"{field} must be a nonnegative integer")
    return value


def _require_positive_int(value: Any, *, field: str) -> int:
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
        raise FeverousAtomicCorpusError(f"{field} must be a positive integer")
    if decoded < 1:
        raise FeverousAtomicCorpusError(f"{field} must be a positive integer")
    return decoded


def _decode_page_payload(raw_page: Any) -> Mapping[str, Any]:
    if isinstance(raw_page, Mapping):
        return raw_page
    if isinstance(raw_page, (bytes, bytearray, memoryview)):
        try:
            text = bytes(raw_page).decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise FeverousAtomicCorpusError("page is not strict UTF-8") from exc
    elif isinstance(raw_page, str):
        text = raw_page
    else:
        raise FeverousAtomicCorpusError("page must be an object or JSON text")

    def no_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise FeverousAtomicCorpusError("page JSON has a duplicate key")
            output[key] = value
        return output

    def no_nonfinite(_value: str) -> None:
        raise FeverousAtomicCorpusError("page JSON has a non-finite value")

    try:
        decoded = json.loads(
            text,
            object_pairs_hook=no_duplicates,
            parse_constant=no_nonfinite,
        )
    except FeverousAtomicCorpusError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise FeverousAtomicCorpusError("page is not strict JSON") from exc
    return _require_mapping(decoded, field="page")


def _parse_local_id(page: str, local_id: str, *, allowed: set[str]) -> tuple[str, tuple[int, ...]]:
    if not isinstance(local_id, str):
        raise FeverousAtomicCorpusError("local id must be a string")
    try:
        parsed = parse_element_id(f"{page}_{local_id}")
    except FeverousWikipediaQualificationError as exc:
        raise FeverousAtomicCorpusError("local id is not canonical") from exc
    if parsed.page != page or parsed.local_id != local_id or parsed.kind not in allowed:
        raise FeverousAtomicCorpusError("local id has the wrong topology")
    return parsed.kind, parsed.indices


def _section_values(sections: Sequence[_Section]) -> tuple[str, ...]:
    return tuple(section.value for section in sections)


def _section_ids(sections: Sequence[_Section]) -> tuple[str, ...]:
    return tuple(section.local_id for section in sections)


def _common_parts(
    *, target: str, title: str, sections: Sequence[_Section], unit_type: str
) -> list[tuple[str, str]]:
    section_path = " > ".join(_section_values(sections)) or "<ROOT>"
    return [
        ("TARGET", target),
        ("TITLE", title),
        ("SECTION_PATH", section_path),
        ("TYPE", unit_type),
    ]


def _render_parts(parts: Sequence[tuple[str, str]]) -> str:
    if not parts or parts[0][0] != "TARGET" or not parts[0][1]:
        raise FeverousAtomicCorpusError("serialization must begin with a nonempty target")
    rendered = "\n".join(f"{label}: {value}" for label, value in parts)
    if not rendered.startswith(f"TARGET: {parts[0][1]}\n"):
        raise FeverousAtomicCorpusError("target-first serialization invariant failed")
    return rendered


def _new_draft(
    *,
    page: str,
    local_id: str,
    unit_type: str,
    coordinates: tuple[int, ...],
    target: str,
    title: str,
    sections: Sequence[_Section],
    official_ordinal: int,
    extra_parts: Sequence[tuple[str, str]] = (),
    **sidecar_fields: Any,
) -> _DraftUnit:
    normalized_target = _require_nonempty_surface(target, field="atomic target")
    parts = _common_parts(
        target=normalized_target,
        title=title,
        sections=sections,
        unit_type=unit_type,
    )
    parts.extend(extra_parts)
    sidecar = AtomicSidecar(
        page=page,
        local_id=local_id,
        unit_type=unit_type,
        coordinates=coordinates,
        section_ids=_section_ids(sections),
        section_path=_section_values(sections),
        official_ordinal=official_ordinal,
        **sidecar_fields,
    )
    return _DraftUnit(
        target=normalized_target,
        parts=tuple(parts),
        sidecar=sidecar,
    )


def _header_context(
    target: _TableCell,
    grid: Sequence[Sequence[_TableCell]],
) -> tuple[tuple[_TableCell, ...], tuple[_TableCell, ...]]:
    if not 0 <= target.row < len(grid) or not 0 <= target.column < len(
        grid[target.row]
    ):
        raise FeverousAtomicCorpusError("cell is outside the normalized table grid")

    def nearest_header_run(candidates: Sequence[_TableCell]) -> tuple[_TableCell, ...]:
        output: list[_TableCell] = []
        seen: set[str] = set()
        encountered_header = False
        for cell in candidates:
            if cell.local_id == target.local_id:
                continue
            if cell.is_header:
                encountered_header = True
                if cell.local_id not in seen:
                    seen.add(cell.local_id)
                    output.append(cell)
            elif encountered_header:
                break
        return tuple(output)

    # This is the official FEVEROUS 0.54 context traversal made deterministic:
    # inspect the span-normalized grid nearest-first and keep the contiguous
    # header run.  Duplicate span-expanded ids collapse on first observation.
    row_candidates = tuple(reversed(grid[target.row][: target.column]))
    column_candidates = tuple(
        reversed(tuple(grid[row][target.column] for row in range(target.row)))
    )
    return nearest_header_run(row_candidates), nearest_header_run(column_candidates)


def _context_values(cells: Sequence[_TableCell]) -> str:
    values = [normalize_surface(cell.value) for cell in cells]
    values = [value for value in values if value]
    return " | ".join(values) or "<NONE>"


def _marked_row(
    target: _TableCell, grid: Sequence[Sequence[_TableCell]]
) -> str:
    if not 0 <= target.row < len(grid):
        raise FeverousAtomicCorpusError("target row is outside the normalized grid")
    row_cells = tuple(grid[target.row])
    rendered: list[str] = []
    target_seen = False
    for column, cell in enumerate(row_cells):
        value = normalize_surface(cell.value) or "<EMPTY>"
        if column == target.column and cell.local_id == target.local_id:
            value = f"<<TARGET>> {value} <</TARGET>>"
            target_seen = True
        rendered.append(value)
    if not target_seen:
        raise FeverousAtomicCorpusError("target is absent from its exact marked row")
    return " | ".join(rendered)


def _compile_table(
    *,
    page: str,
    top_index: int,
    value: Mapping[str, Any],
    title: str,
    sections: Sequence[_Section],
    start_ordinal: int,
) -> tuple[list[_DraftUnit], list[str], int]:
    table_id = f"table_{top_index}"
    table_kind_raw = value.get("type")
    if not isinstance(table_kind_raw, str) or "\x00" in table_kind_raw:
        raise FeverousAtomicCorpusError("table type must be a safe string")
    table_kind = normalize_surface(table_kind_raw)
    rows = value.get("table")
    if not isinstance(rows, list):
        raise FeverousAtomicCorpusError("table rows must be a list")

    raw_rows: list[list[_TableCell]] = []
    seen_ids: set[str] = set()
    source_ordinal = 0
    for raw_row_i, row in enumerate(rows):
        if not isinstance(row, list):
            raise FeverousAtomicCorpusError("table row must be a list")
        decoded_row: list[_TableCell] = []
        for raw_cell in row:
            cell = _require_mapping(raw_cell, field="table cell")
            local_id = cell.get("id")
            kind, indices = _parse_local_id(
                page,
                local_id,
                allowed={"cell", "header_cell"},
            )
            if len(indices) != 3 or indices[0] != top_index:
                raise FeverousAtomicCorpusError("cell coordinates have the wrong table")
            if local_id in seen_ids:
                raise FeverousAtomicCorpusError("page contains a duplicate atomic id")
            seen_ids.add(local_id)
            is_header = cell.get("is_header")
            if type(is_header) is not bool or is_header != (kind == "header_cell"):
                raise FeverousAtomicCorpusError("cell type and header flag disagree")
            raw_value = cell.get("value")
            if not isinstance(raw_value, str) or "\x00" in raw_value:
                raise FeverousAtomicCorpusError("cell value must be a safe string")
            decoded_row.append(
                _TableCell(
                    local_id=local_id,
                    kind=kind,
                    # Element-id coordinates remain identity only.  Physical
                    # table coordinates are assigned from the official 0.54
                    # span-normalized grid below.
                    coordinates=(top_index, raw_row_i, len(decoded_row)),
                    value=raw_value,
                    is_header=is_header,
                    row_span=_require_positive_int(
                        cell.get("row_span"), field="cell row_span"
                    ),
                    column_span=_require_positive_int(
                        cell.get("column_span"), field="cell column_span"
                    ),
                    source_ordinal=source_ordinal,
                )
            )
            source_ordinal += 1
        raw_rows.append(decoded_row)

    if not raw_rows or not raw_rows[0]:
        raise FeverousAtomicCorpusError("table has no normalized-grid width")
    column_count = sum(cell.column_span for cell in raw_rows[0])
    if column_count < 1:
        raise FeverousAtomicCorpusError("table has no normalized-grid width")
    normalized: list[list[_TableCell | None]] = [
        [None for _ in range(column_count)] for _ in raw_rows
    ]
    for row_i, raw_row in enumerate(raw_rows):
        for cell in raw_row:
            lowest_column = 0
            while (
                lowest_column < column_count
                and normalized[row_i][lowest_column] is not None
            ):
                lowest_column += 1
            if lowest_column >= column_count:
                raise FeverousAtomicCorpusError(
                    "table row exceeds the official normalized-grid width"
                )
            for offset in range(
                min(cell.column_span, column_count - lowest_column)
            ):
                normalized[row_i][lowest_column + offset] = cell
            for offset in range(min(cell.row_span, len(raw_rows) - row_i)):
                normalized[row_i + offset][lowest_column] = cell
    if any(cell is None for row in normalized for cell in row):
        raise FeverousAtomicCorpusError(
            "table cannot form the complete official normalized grid"
        )

    grid_rows: list[tuple[_TableCell, ...]] = []
    first_id_order: list[str] = []
    seen_grid_ids: set[str] = set()
    last_physical_by_id: dict[str, _TableCell] = {}
    for row_i, row in enumerate(normalized):
        physical_row: list[_TableCell] = []
        for column_i, raw_cell in enumerate(row):
            assert raw_cell is not None
            physical = replace(
                raw_cell,
                coordinates=(top_index, row_i, column_i),
            )
            physical_row.append(physical)
            if physical.local_id not in seen_grid_ids:
                seen_grid_ids.add(physical.local_id)
                first_id_order.append(physical.local_id)
            # FEVEROUS 0.54 all_cells keeps the final row-major occurrence of
            # a span-expanded id; bind the exact same target coordinate.
            last_physical_by_id[physical.local_id] = physical
        grid_rows.append(tuple(physical_row))
    grid = tuple(grid_rows)
    cells = tuple(last_physical_by_id[local_id] for local_id in first_id_order)

    caption = ""
    if "caption" in value:
        raw_caption = value["caption"]
        if not isinstance(raw_caption, str) or "\x00" in raw_caption:
            raise FeverousAtomicCorpusError("table caption must be a safe string")
        caption = normalize_surface(raw_caption)

    drafts: list[_DraftUnit] = []
    excluded: list[str] = []
    for cell_i, cell in enumerate(cells):
        target = normalize_surface(cell.value)
        if not target:
            excluded.append(cell.local_id)
            continue
        row_headers, column_headers = _header_context(cell, grid)
        headers = (
            f"ROW[{_context_values(row_headers)}] "
            f"COLUMN[{_context_values(column_headers)}]"
        )
        extra = (
            ("TABLE_CAPTION", caption or "<NONE>"),
            ("APPLICABLE_HEADERS", headers),
            ("ROW_WITH_TARGET_MARKED", _marked_row(cell, grid)),
        )
        drafts.append(
            _new_draft(
                page=page,
                local_id=cell.local_id,
                unit_type=cell.kind,
                coordinates=cell.coordinates,
                target=target,
                title=title,
                sections=sections,
                official_ordinal=start_ordinal + cell_i,
                extra_parts=extra,
                table_id=table_id,
                table_kind=table_kind,
                table_caption=caption or None,
                row_span=cell.row_span,
                column_span=cell.column_span,
                applicable_row_header_ids=tuple(
                    header.local_id for header in row_headers
                ),
                applicable_column_header_ids=tuple(
                    header.local_id for header in column_headers
                ),
            )
        )

    consumed = len(cells)
    if "caption" in value:
        caption_id = f"table_caption_{top_index}"
        if caption:
            drafts.append(
                _new_draft(
                    page=page,
                    local_id=caption_id,
                    unit_type="table_caption",
                    coordinates=(top_index,),
                    target=caption,
                    title=title,
                    sections=sections,
                    official_ordinal=start_ordinal + consumed,
                    extra_parts=(("TABLE_KIND", table_kind or "<EMPTY>"),),
                    table_id=table_id,
                    table_kind=table_kind,
                    table_caption=caption,
                )
            )
        else:
            excluded.append(caption_id)
        consumed += 1
    return drafts, excluded, consumed


def _compile_list(
    *,
    page: str,
    top_index: int,
    value: Mapping[str, Any],
    title: str,
    sections: Sequence[_Section],
    start_ordinal: int,
) -> tuple[list[_DraftUnit], list[str], int]:
    list_id = f"list_{top_index}"
    list_type = value.get("type")
    if not isinstance(list_type, str) or "\x00" in list_type:
        raise FeverousAtomicCorpusError("list type must be a safe string")
    items = value.get("list")
    if not isinstance(items, list):
        raise FeverousAtomicCorpusError("list items must be a list")
    level_stack: dict[int, tuple[str, str]] = {}
    drafts: list[_DraftUnit] = []
    excluded: list[str] = []
    seen_ids: set[str] = set()
    for item_i, raw_item in enumerate(items):
        item = _require_mapping(raw_item, field="list item")
        if "type" in item and (
            not isinstance(item["type"], str) or "\x00" in item["type"]
        ):
            raise FeverousAtomicCorpusError(
                "nested-list type must be a safe string"
            )
        local_id = item.get("id")
        kind, indices = _parse_local_id(page, local_id, allowed={"item"})
        if kind != "item" or len(indices) != 2 or indices[0] != top_index:
            raise FeverousAtomicCorpusError("item coordinates have the wrong list")
        if local_id in seen_ids:
            raise FeverousAtomicCorpusError("page contains a duplicate atomic id")
        seen_ids.add(local_id)
        level = _require_nonnegative_int(item.get("level"), field="item level")
        raw_value = item.get("value")
        if not isinstance(raw_value, str) or "\x00" in raw_value:
            raise FeverousAtomicCorpusError("item value must be a safe string")
        target = normalize_surface(raw_value)
        ancestors = tuple(
            level_stack[key]
            for key in sorted(level_stack)
            if key < level
        )
        for key in tuple(level_stack):
            if key >= level:
                del level_stack[key]
        level_stack[level] = (local_id, target)
        if not target:
            excluded.append(local_id)
            continue
        ancestor_values = " > ".join(value for _, value in ancestors if value)
        drafts.append(
            _new_draft(
                page=page,
                local_id=local_id,
                unit_type="item",
                coordinates=(indices[0], indices[1]),
                target=target,
                title=title,
                sections=sections,
                official_ordinal=start_ordinal + item_i,
                extra_parts=(("LIST_ANCESTOR_PATH", ancestor_values or "<ROOT>"),),
                list_id=list_id,
                list_ancestor_ids=tuple(local for local, _ in ancestors),
            )
        )
    return drafts, excluded, len(items)


def _new_identity(
    *,
    page: str,
    local_id: str,
    unit_type: str,
    official_ordinal: int,
    normalized_target: str,
) -> AtomicIdentity:
    target = _require_nonempty_surface(normalized_target, field="atomic target")
    return AtomicIdentity(
        page=page,
        local_id=local_id,
        unit_type=unit_type,
        official_ordinal=official_ordinal,
        normalized_target=target,
        target_sha256=hashlib.sha256(target.encode("utf-8")).hexdigest(),
    )


def _enumerate_table_identities(
    *,
    page: str,
    top_index: int,
    value: Mapping[str, Any],
    start_ordinal: int,
) -> tuple[list[AtomicIdentity], list[str], int, set[str]]:
    """Enumerate table identities without headers, marked rows, or text rendering."""

    table_kind = value.get("type")
    if not isinstance(table_kind, str) or "\x00" in table_kind:
        raise FeverousAtomicCorpusError("table type must be a safe string")
    rows = value.get("table")
    if not isinstance(rows, list):
        raise FeverousAtomicCorpusError("table rows must be a list")

    raw_rows: list[list[_TableCell]] = []
    raw_by_id: dict[str, _TableCell] = {}
    source_ordinal = 0
    for raw_row_i, row in enumerate(rows):
        if not isinstance(row, list):
            raise FeverousAtomicCorpusError("table row must be a list")
        decoded_row: list[_TableCell] = []
        for raw_cell in row:
            cell = _require_mapping(raw_cell, field="table cell")
            local_id = cell.get("id")
            kind, indices = _parse_local_id(
                page,
                local_id,
                allowed={"cell", "header_cell"},
            )
            if len(indices) != 3 or indices[0] != top_index:
                raise FeverousAtomicCorpusError(
                    "cell coordinates have the wrong table"
                )
            if local_id in raw_by_id:
                raise FeverousAtomicCorpusError(
                    "page contains a duplicate atomic id"
                )
            is_header = cell.get("is_header")
            if type(is_header) is not bool or is_header != (kind == "header_cell"):
                raise FeverousAtomicCorpusError(
                    "cell type and header flag disagree"
                )
            raw_value = cell.get("value")
            if not isinstance(raw_value, str) or "\x00" in raw_value:
                raise FeverousAtomicCorpusError("cell value must be a safe string")
            decoded = _TableCell(
                local_id=local_id,
                kind=kind,
                coordinates=(top_index, raw_row_i, len(decoded_row)),
                value=raw_value,
                is_header=is_header,
                row_span=_require_positive_int(
                    cell.get("row_span"), field="cell row_span"
                ),
                column_span=_require_positive_int(
                    cell.get("column_span"), field="cell column span"
                ),
                source_ordinal=source_ordinal,
            )
            raw_by_id[local_id] = decoded
            decoded_row.append(decoded)
            source_ordinal += 1
        raw_rows.append(decoded_row)

    if not raw_rows or not raw_rows[0]:
        raise FeverousAtomicCorpusError("table has no normalized-grid width")
    column_count = sum(cell.column_span for cell in raw_rows[0])
    if column_count < 1:
        raise FeverousAtomicCorpusError("table has no normalized-grid width")
    normalized: list[list[_TableCell | None]] = [
        [None for _ in range(column_count)] for _ in raw_rows
    ]
    for row_i, raw_row in enumerate(raw_rows):
        for cell in raw_row:
            lowest_column = 0
            while (
                lowest_column < column_count
                and normalized[row_i][lowest_column] is not None
            ):
                lowest_column += 1
            if lowest_column >= column_count:
                raise FeverousAtomicCorpusError(
                    "table row exceeds the official normalized-grid width"
                )
            for offset in range(
                min(cell.column_span, column_count - lowest_column)
            ):
                normalized[row_i][lowest_column + offset] = cell
            for offset in range(min(cell.row_span, len(raw_rows) - row_i)):
                normalized[row_i + offset][lowest_column] = cell
    if any(cell is None for row in normalized for cell in row):
        raise FeverousAtomicCorpusError(
            "table cannot form the complete official normalized grid"
        )

    first_id_order: list[str] = []
    seen_grid_ids: set[str] = set()
    for row in normalized:
        for cell in row:
            assert cell is not None
            if cell.local_id not in seen_grid_ids:
                seen_grid_ids.add(cell.local_id)
                first_id_order.append(cell.local_id)
    if seen_grid_ids != set(raw_by_id):
        raise FeverousAtomicCorpusError(
            "table identity enumeration omitted a source cell"
        )

    identities: list[AtomicIdentity] = []
    excluded: list[str] = []
    for cell_i, local_id in enumerate(first_id_order):
        cell = raw_by_id[local_id]
        target = normalize_surface(cell.value)
        if target:
            identities.append(
                _new_identity(
                    page=page,
                    local_id=cell.local_id,
                    unit_type=cell.kind,
                    official_ordinal=start_ordinal + cell_i,
                    normalized_target=target,
                )
            )
        else:
            excluded.append(cell.local_id)

    consumed = len(first_id_order)
    if "caption" in value:
        raw_caption = value["caption"]
        if not isinstance(raw_caption, str) or "\x00" in raw_caption:
            raise FeverousAtomicCorpusError("table caption must be a safe string")
        caption = normalize_surface(raw_caption)
        caption_id = f"table_caption_{top_index}"
        if caption:
            identities.append(
                _new_identity(
                    page=page,
                    local_id=caption_id,
                    unit_type="table_caption",
                    official_ordinal=start_ordinal + consumed,
                    normalized_target=caption,
                )
            )
        else:
            excluded.append(caption_id)
        consumed += 1
    return identities, excluded, consumed, set(raw_by_id) | {
        local_id
        for local_id in excluded
        if local_id.startswith("table_caption_")
    } | {
        row.local_id
        for row in identities
        if row.unit_type == "table_caption"
    }


def _enumerate_list_identities(
    *,
    page: str,
    top_index: int,
    value: Mapping[str, Any],
    start_ordinal: int,
) -> tuple[list[AtomicIdentity], list[str], int, set[str]]:
    list_type = value.get("type")
    if not isinstance(list_type, str) or "\x00" in list_type:
        raise FeverousAtomicCorpusError("list type must be a safe string")
    items = value.get("list")
    if not isinstance(items, list):
        raise FeverousAtomicCorpusError("list items must be a list")
    identities: list[AtomicIdentity] = []
    excluded: list[str] = []
    seen_ids: set[str] = set()
    for item_i, raw_item in enumerate(items):
        item = _require_mapping(raw_item, field="list item")
        if "type" in item and (
            not isinstance(item["type"], str) or "\x00" in item["type"]
        ):
            raise FeverousAtomicCorpusError(
                "nested-list type must be a safe string"
            )
        local_id = item.get("id")
        kind, indices = _parse_local_id(page, local_id, allowed={"item"})
        if kind != "item" or len(indices) != 2 or indices[0] != top_index:
            raise FeverousAtomicCorpusError("item coordinates have the wrong list")
        if local_id in seen_ids:
            raise FeverousAtomicCorpusError("page contains a duplicate atomic id")
        seen_ids.add(local_id)
        _require_nonnegative_int(item.get("level"), field="item level")
        raw_value = item.get("value")
        if not isinstance(raw_value, str) or "\x00" in raw_value:
            raise FeverousAtomicCorpusError("item value must be a safe string")
        target = normalize_surface(raw_value)
        if target:
            identities.append(
                _new_identity(
                    page=page,
                    local_id=local_id,
                    unit_type="item",
                    official_ordinal=start_ordinal + item_i,
                    normalized_target=target,
                )
            )
        else:
            excluded.append(local_id)
    return identities, excluded, len(items), seen_ids


def enumerate_official_page_atomic_identities(
    page_id: str, raw_page: Any
) -> PageIdentityEnumeration:
    """Enumerate atomic identities without constructing texts or sidecars.

    This first pass retains only normalized targets and their hashes.  Formal
    acquisition can therefore exhaust the official page stream before fully
    compiling only pages selected by gold membership or the bounded heap.
    """

    if not isinstance(page_id, str) or not page_id or "\x00" in page_id:
        raise FeverousAtomicCorpusError("page id must be a nonempty safe string")
    page = _decode_page_payload(raw_page)
    if page.get("title") != page_id:
        raise FeverousAtomicCorpusError("page title does not exactly match page id")
    _require_nonempty_surface(page_id, field="page title")
    order = page.get("order")
    if not isinstance(order, list) or any(not isinstance(item, str) for item in order):
        raise FeverousAtomicCorpusError("page order must be a string list")
    if len(order) != len(set(order)):
        raise FeverousAtomicCorpusError("page order contains a duplicate")

    identities: list[AtomicIdentity] = []
    excluded: list[str] = []
    observed_local_ids: set[str] = set()
    next_official_ordinal = 0
    for top_level_id in order:
        match = _TOP_LEVEL_RE.fullmatch(top_level_id)
        if match is None or top_level_id not in page:
            raise FeverousAtomicCorpusError(
                "page order references invalid topology"
            )
        kind = match.group(1)
        top_index = int(match.group(2))
        raw_value = page[top_level_id]
        if kind == "section":
            section = _require_mapping(raw_value, field="section")
            _require_nonnegative_int(section.get("level"), field="section level")
            section_value = section.get("value")
            if not isinstance(section_value, str) or "\x00" in section_value:
                raise FeverousAtomicCorpusError(
                    "section value must be a safe string"
                )
            normalize_surface(section_value)
            continue
        if kind == "sentence":
            if not isinstance(raw_value, str) or "\x00" in raw_value:
                raise FeverousAtomicCorpusError("sentence must be a safe string")
            target = normalize_surface(raw_value)
            if top_level_id in observed_local_ids:
                raise FeverousAtomicCorpusError(
                    "page contains a duplicate atomic id"
                )
            observed_local_ids.add(top_level_id)
            if target:
                identities.append(
                    _new_identity(
                        page=page_id,
                        local_id=top_level_id,
                        unit_type="sentence",
                        official_ordinal=next_official_ordinal,
                        normalized_target=target,
                    )
                )
            else:
                excluded.append(top_level_id)
            next_official_ordinal += 1
            continue

        structured = _require_mapping(raw_value, field=kind)
        if kind == "table":
            new_rows, new_excluded, consumed, new_ids = (
                _enumerate_table_identities(
                    page=page_id,
                    top_index=top_index,
                    value=structured,
                    start_ordinal=next_official_ordinal,
                )
            )
        else:
            new_rows, new_excluded, consumed, new_ids = (
                _enumerate_list_identities(
                    page=page_id,
                    top_index=top_index,
                    value=structured,
                    start_ordinal=next_official_ordinal,
                )
            )
        if observed_local_ids.intersection(new_ids):
            raise FeverousAtomicCorpusError(
                "page contains a duplicate atomic id"
            )
        observed_local_ids.update(new_ids)
        identities.extend(new_rows)
        excluded.extend(new_excluded)
        next_official_ordinal += consumed

    if len(excluded) != len(set(excluded)):
        raise FeverousAtomicCorpusError("page contains a duplicate atomic id")
    return PageIdentityEnumeration(
        page=page_id,
        identities=tuple(identities),
        excluded_empty_local_ids=tuple(excluded),
    )


def crosscheck_identity_enumeration(
    enumeration: PageIdentityEnumeration,
    compilation: PageCompilation,
) -> PageCompilation:
    """Require a selected page's full compilation to match its first pass."""

    if not isinstance(enumeration, PageIdentityEnumeration) or not isinstance(
        compilation, PageCompilation
    ):
        raise FeverousAtomicCorpusError(
            "identity/full-compilation crosscheck inputs are invalid"
        )
    if (
        enumeration.page != compilation.page
        or enumeration.excluded_empty_local_ids
        != compilation.excluded_empty_local_ids
        or len(enumeration.identities) != len(compilation.units)
    ):
        raise FeverousAtomicCorpusError(
            "identity enumeration and full compilation differ"
        )
    for identity, unit in zip(enumeration.identities, compilation.units):
        sidecar = unit.sidecar
        if (
            identity.page != sidecar.page
            or identity.local_id != sidecar.local_id
            or identity.unit_type != sidecar.unit_type
            or identity.official_ordinal != sidecar.official_ordinal
            or identity.normalized_target != unit.target
            or identity.target_sha256
            != hashlib.sha256(unit.target.encode("utf-8")).hexdigest()
        ):
            raise FeverousAtomicCorpusError(
                "identity enumeration and full compilation differ"
            )
    return compilation


def compile_official_page(page_id: str, raw_page: Any) -> PageCompilation:
    """Compile one exact official page without consulting claims or outcomes.

    Empty atomic targets are returned in ``excluded_empty_local_ids`` and are
    never materialized as corpus units.  The page compiler otherwise fails
    closed on malformed topology instead of repairing or fuzzily resolving it.
    """

    if not isinstance(page_id, str) or not page_id or "\x00" in page_id:
        raise FeverousAtomicCorpusError("page id must be a nonempty safe string")
    page = _decode_page_payload(raw_page)
    if page.get("title") != page_id:
        raise FeverousAtomicCorpusError("page title does not exactly match page id")
    title = _require_nonempty_surface(page_id, field="page title")
    order = page.get("order")
    if not isinstance(order, list) or any(not isinstance(item, str) for item in order):
        raise FeverousAtomicCorpusError("page order must be a string list")
    if len(order) != len(set(order)):
        raise FeverousAtomicCorpusError("page order contains a duplicate")

    sections_by_level: dict[int, _Section] = {}
    drafts: list[_DraftUnit] = []
    excluded: list[str] = []
    all_local_ids: set[str] = set()
    next_official_ordinal = 0
    for top_level_id in order:
        match = _TOP_LEVEL_RE.fullmatch(top_level_id)
        if match is None or top_level_id not in page:
            raise FeverousAtomicCorpusError("page order references invalid topology")
        kind = match.group(1)
        top_index = int(match.group(2))
        raw_value = page[top_level_id]
        if kind == "section":
            section = _require_mapping(raw_value, field="section")
            level = _require_nonnegative_int(section.get("level"), field="section level")
            raw_section_value = section.get("value")
            if not isinstance(raw_section_value, str) or "\x00" in raw_section_value:
                raise FeverousAtomicCorpusError("section value must be a safe string")
            value = normalize_surface(raw_section_value) or "<EMPTY_SECTION>"
            for key in tuple(sections_by_level):
                if key >= level:
                    del sections_by_level[key]
            sections_by_level[level] = _Section(top_level_id, value, level)
            continue

        sections = tuple(sections_by_level[key] for key in sorted(sections_by_level))
        if kind == "sentence":
            if not isinstance(raw_value, str) or "\x00" in raw_value:
                raise FeverousAtomicCorpusError("sentence must be a safe string")
            target = normalize_surface(raw_value)
            if target:
                draft = _new_draft(
                    page=page_id,
                    local_id=top_level_id,
                    unit_type="sentence",
                    coordinates=(top_index,),
                    target=target,
                    title=title,
                    sections=sections,
                    official_ordinal=next_official_ordinal,
                )
                drafts.append(draft)
            else:
                excluded.append(top_level_id)
            next_official_ordinal += 1
            continue

        structured = _require_mapping(raw_value, field=kind)
        if kind == "table":
            new_drafts, new_excluded, consumed = _compile_table(
                page=page_id,
                top_index=top_index,
                value=structured,
                title=title,
                sections=sections,
                start_ordinal=next_official_ordinal,
            )
        else:
            new_drafts, new_excluded, consumed = _compile_list(
                page=page_id,
                top_index=top_index,
                value=structured,
                title=title,
                sections=sections,
                start_ordinal=next_official_ordinal,
            )
        for draft in new_drafts:
            if draft.sidecar.local_id in all_local_ids:
                raise FeverousAtomicCorpusError("page contains a duplicate atomic id")
            all_local_ids.add(draft.sidecar.local_id)
        for local_id in new_excluded:
            if local_id in all_local_ids:
                raise FeverousAtomicCorpusError("page contains a duplicate atomic id")
            all_local_ids.add(local_id)
        drafts.extend(new_drafts)
        excluded.extend(new_excluded)
        next_official_ordinal += consumed

    # Sentences bypass the structured duplicate loop above.
    observed: set[str] = set()
    for draft in drafts:
        if draft.sidecar.local_id in observed:
            raise FeverousAtomicCorpusError("page contains a duplicate atomic id")
        observed.add(draft.sidecar.local_id)
    if observed.intersection(excluded) or len(excluded) != len(set(excluded)):
        raise FeverousAtomicCorpusError("page contains a duplicate atomic id")

    units: list[AtomicUnit] = []
    for index, draft in enumerate(drafts):
        previous = drafts[index - 1] if index else None
        following = drafts[index + 1] if index + 1 < len(drafts) else None
        previous_id = (
            previous.sidecar.local_id
            if previous is not None
            and previous.sidecar.official_ordinal + 1
            == draft.sidecar.official_ordinal
            and previous.sidecar.section_ids == draft.sidecar.section_ids
            else None
        )
        next_id = (
            following.sidecar.local_id
            if following is not None
            and draft.sidecar.official_ordinal + 1
            == following.sidecar.official_ordinal
            and following.sidecar.section_ids == draft.sidecar.section_ids
            else None
        )
        sidecar = replace(
            draft.sidecar,
            previous_atomic_local_id=previous_id,
            next_atomic_local_id=next_id,
        )
        text = _render_parts(draft.parts)
        units.append(
            AtomicUnit(
                target=draft.target,
                text=text,
                text_utf8=text.encode("utf-8", errors="strict"),
                sidecar=sidecar,
            )
        )
    return PageCompilation(
        page=page_id,
        units=tuple(units),
        excluded_empty_local_ids=tuple(excluded),
    )


def require_nonempty_atomic_target(value: str) -> str:
    """Public fail-closed target check used by acquisition eligibility."""

    return _require_nonempty_surface(value, field="atomic target")


def tail_truncate_token_ids(
    token_ids: Sequence[int], *, maximum: int = MAXIMUM_MODEL_TOKENS
) -> tuple[int, ...]:
    """Apply the design's prefix-preserving tail truncation to token ids."""

    if type(maximum) is not int or maximum < 1:
        raise FeverousAtomicCorpusError("maximum token count must be positive")
    if isinstance(token_ids, (str, bytes, bytearray)) or any(
        type(token) is not int for token in token_ids
    ):
        raise FeverousAtomicCorpusError("token ids must be an integer sequence")
    return tuple(token_ids[:maximum])


def _coerce_ner_spans(
    normalized_claim: str,
    spans: Sequence[NerSpan | tuple[int, int]],
) -> tuple[NerSpan, ...]:
    if isinstance(spans, (str, bytes, bytearray, Mapping)):
        raise FeverousAtomicCorpusError("NER spans must contain offsets only")
    output: list[NerSpan] = []
    for raw in spans:
        if isinstance(raw, NerSpan):
            span = raw
        elif (
            isinstance(raw, tuple)
            and len(raw) == 2
            and type(raw[0]) is int
            and type(raw[1]) is int
        ):
            span = NerSpan(raw[0], raw[1])
        else:
            raise FeverousAtomicCorpusError("NER spans must contain offsets only")
        if (
            type(span.start) is not int
            or type(span.end) is not int
            or span.start < 0
            or span.end <= span.start
            or span.end > len(normalized_claim)
        ):
            raise FeverousAtomicCorpusError("NER span is outside the normalized claim")
        if not normalize_surface(normalized_claim[span.start : span.end]):
            raise FeverousAtomicCorpusError("NER span is empty after normalization")
        output.append(span)
    output.sort(key=lambda span: (span.start, span.end))
    for left, right in zip(output, output[1:]):
        if right.start < left.end:
            raise FeverousAtomicCorpusError("NER spans overlap")
    return tuple(output)


def _masked_slice(
    claim: str,
    start: int,
    end: int,
    entity_spans: Sequence[NerSpan],
    numeric_spans: Sequence[NerSpan],
) -> str:
    intervals: list[tuple[int, int, str]] = [
        (span.start, span.end, "[ENTITY]") for span in entity_spans
    ]
    for span in numeric_spans:
        if any(span.start < other.end and other.start < span.end for other in entity_spans):
            continue
        intervals.append((span.start, span.end, "[NUMBER]"))
    intervals.sort(key=lambda value: (value[0], -(value[1] - value[0]), value[2]))
    cursor = start
    pieces: list[str] = []
    for interval_start, interval_end, marker in intervals:
        if interval_end <= start or interval_start >= end:
            continue
        clipped_start = max(interval_start, start)
        clipped_end = min(interval_end, end)
        if clipped_start < cursor:
            continue
        pieces.append(claim[cursor:clipped_start])
        pieces.append(marker)
        cursor = clipped_end
    pieces.append(claim[cursor:end])
    return normalize_surface("".join(pieces))


def _trimmed_bounds(claim: str, start: int, end: int) -> tuple[int, int]:
    while start < end and claim[start].isspace():
        start += 1
    while end > start and claim[end - 1].isspace():
        end -= 1
    return start, end


def _deduplicated_facets(
    candidates: Sequence[ClaimFacet],
    *,
    limit: int,
    seen: set[str],
) -> list[ClaimFacet]:
    output: list[ClaimFacet] = []
    for facet in candidates:
        identity = normalize_surface(facet.text).casefold()
        if not identity or identity in seen:
            continue
        seen.add(identity)
        output.append(facet)
        if len(output) == limit:
            break
    return output


def compile_claim_facets(
    claim: str,
    ner_spans: Sequence[NerSpan | tuple[int, int]] = (),
) -> CompiledClaimFacets:
    """Compile the frozen 4/2/2 claim-only facet view.

    ``ner_spans`` are offsets over ``normalized_claim`` (the NFKC and
    whitespace-collapsed claim returned here).  This small offset-only seam is
    the complete output accepted from the frozen offline NER model.  A mapping
    or a span carrying label/evidence metadata is rejected.
    """

    if not isinstance(claim, str):
        raise FeverousAtomicCorpusError(
            "claim facet compiler accepts a claim string, not a record"
        )
    normalized_claim = _require_nonempty_surface(claim, field="claim")
    entity_spans = _coerce_ner_spans(normalized_claim, ner_spans)

    entity_candidates = [
        ClaimFacet(
            kind="entity",
            text=normalize_surface(normalized_claim[span.start : span.end]),
            source_start=span.start,
            source_end=span.end,
        )
        for span in entity_spans
    ]
    numeric_spans = tuple(
        NerSpan(match.start(), match.end())
        for match in _NUMERIC_OR_DATE_RE.finditer(normalized_claim)
    )
    numeric_candidates = [
        ClaimFacet(
            kind="numeric_or_date",
            text=normalize_surface(normalized_claim[span.start : span.end]),
            source_start=span.start,
            source_end=span.end,
        )
        for span in numeric_spans
    ]

    relation_candidates: list[ClaimFacet] = []
    cursor = 0
    protected_spans = (*entity_spans, *numeric_spans)
    for delimiter in _CLAUSE_DELIMITER_RE.finditer(normalized_claim):
        if any(
            span.start < delimiter.end() and delimiter.start() < span.end
            for span in protected_spans
        ):
            continue
        clause_start, clause_end = _trimmed_bounds(
            normalized_claim, cursor, delimiter.start()
        )
        if clause_start < clause_end:
            masked = _masked_slice(
                normalized_claim,
                clause_start,
                clause_end,
                entity_spans,
                numeric_spans,
            )
            if masked:
                relation_candidates.append(
                    ClaimFacet(
                        kind="relation_clause",
                        text=masked,
                        source_start=clause_start,
                        source_end=clause_end,
                    )
                )
        cursor = delimiter.end()
    if cursor < len(normalized_claim):
        clause_start, clause_end = _trimmed_bounds(
            normalized_claim, cursor, len(normalized_claim)
        )
        masked = _masked_slice(
            normalized_claim,
            clause_start,
            clause_end,
            entity_spans,
            numeric_spans,
        )
        if masked:
            relation_candidates.append(
                ClaimFacet(
                    kind="relation_clause",
                    text=masked,
                    source_start=clause_start,
                    source_end=clause_end,
                )
            )
    if not relation_candidates:
        masked = _masked_slice(
            normalized_claim,
            0,
            len(normalized_claim),
            entity_spans,
            numeric_spans,
        )
        if masked:
            relation_candidates.append(
                ClaimFacet(
                    kind="relation_clause",
                    text=masked,
                    source_start=0,
                    source_end=len(normalized_claim),
                )
            )

    seen: set[str] = set()
    facets: list[ClaimFacet] = []
    facets.extend(
        _deduplicated_facets(
            entity_candidates,
            limit=ENTITY_FACET_LIMIT,
            seen=seen,
        )
    )
    facets.extend(
        _deduplicated_facets(
            numeric_candidates,
            limit=NUMERIC_OR_DATE_FACET_LIMIT,
            seen=seen,
        )
    )
    facets.extend(
        _deduplicated_facets(
            relation_candidates,
            limit=RELATION_CLAUSE_FACET_LIMIT,
            seen=seen,
        )
    )
    return CompiledClaimFacets(
        normalized_claim=normalized_claim,
        facets=tuple(facets),
    )


__all__ = [
    "ARM_IDS",
    "ATOMIC_UNIT_TYPES",
    "ENTITY_FACET_LIMIT",
    "IDENTITY_COMMITMENT_SCHEMA",
    "IDENTITY_ENUMERATOR_VERSION",
    "MAXIMUM_MODEL_TOKENS",
    "NUMERIC_OR_DATE_FACET_LIMIT",
    "RELATION_CLAUSE_FACET_LIMIT",
    "AtomicSidecar",
    "AtomicIdentity",
    "AtomicUnit",
    "ClaimFacet",
    "CompiledClaimFacets",
    "FeverousAtomicCorpusError",
    "NerSpan",
    "PageCompilation",
    "PageIdentityEnumeration",
    "VERSION",
    "compile_claim_facets",
    "compile_official_page",
    "crosscheck_identity_enumeration",
    "enumerate_official_page_atomic_identities",
    "normalize_surface",
    "require_nonempty_atomic_target",
    "tail_truncate_token_ids",
]
