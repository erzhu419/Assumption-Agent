"""Pure source-agnostic core for a derived WikiSQL UAO reality study.

This module deliberately has no file, archive, network, model, controller, or
official-HippoRAG entrypoint.  It implements only contracts that can be frozen
and qualified with synthetic examples from WikiSQL's documented release-1.1
schema:

* canonical commitments and private HMAC cohort ordering;
* the source-native single-WHERE EQ / GT / LT family partition;
* item-local table-row serialization and a deterministic RAW BM25 top five;
* derivation of physical gold row ordinals from documented ``sql.conds``;
* the integer ``U = hits + complete`` utility; and
* the preregistered paired exact sign-flip reference tail and primary
  intersection decision.

The gold-row helper reproduces the relevant *style* of the official WikiSQL
engine: text comparisons are lower-cased, numeric strings first use an
English-decimal parse and then the engine's digits-and-decimal-point fallback.
It does not claim to replace the official SQLite execution cross-check that a
future source-owning controller must perform before a formal study.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import hmac
import json
import math
import re
import unicodedata
from typing import Any, Mapping, Sequence


VERSION = "wikisql_uao_reality_v1"
STUDY_ID = "WIKISQL_UAO_REALITY_V1"
TOP_K = 5
BM25_K1 = 1.2
BM25_B = 0.75
BM25_INTEGER_SCALE = 1_000_000
PROMOTION_ALPHA = Fraction(1, 10)
HMAC_SECRET_BYTES = 32

AGGREGATION_NAMES = ("NONE", "MAX", "MIN", "COUNT", "SUM", "AVG")
CONDITION_OPERATORS = ("=", ">", "<")
CONDITION_FAMILIES = ("EQ", "GT", "LT")
FAMILY_ORDER = CONDITION_FAMILIES
TABLE_TYPES = ("text", "real")
MIN_TABLE_ROWS = 11
MAX_TABLE_ROWS = 80
MIN_GOLD_ROWS = 1
MAX_GOLD_ROWS = 5
# One row document is consumed by RAW tokenization, the Agent parser/encoder,
# and the official-HippoRAG OpenIE lane.  Freeze a shared whole-document bound
# before source selection so per-cell limits cannot admit a row that only one
# arm can represent.
MAX_SERIALIZED_ROW_CHARACTERS = 16_000
COHORT_QUOTAS = {
    "A_form": {family: 64 for family in FAMILY_ORDER},
    "A_hold": {family: 24 for family in FAMILY_ORDER},
}
OFFICIAL_ITEM_FIELDS = frozenset({"phase", "question", "sql", "table_id"})
OFFICIAL_TABLE_FIELDS = frozenset({"id", "header", "types", "rows"})
OFFICIAL_SQL_FIELDS = frozenset({"sel", "agg", "conds"})
MEASUREMENT_ARMS = ("agent", "raw", "hipporag")
BASELINE_ARMS = ("raw", "hipporag")

_TOKEN_RE = re.compile(r"[^\W_]+(?:[._:/+#%-][^\W_]+)*", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)
_ENGLISH_DECIMAL_RE = re.compile(
    r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?\Z"
)
_NUMERIC_FALLBACK_RE = re.compile(r"[-+]?\d*\.\d+|\d+")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_MAX_TEXT_CHARACTERS = 1_000_000
_HMAC_PREFIX = b"WIKISQL_UAO_REALITY_V1\0selection\0"


class WikiSQLUAORealityError(ValueError):
    """A pure-core input or frozen statistical contract drifted."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic ASCII JSON bytes, rejecting non-finite values."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise WikiSQLUAORealityError("value is not canonical JSON") from exc


def canonical_sha256(value: Any) -> str:
    """Hash the exact :func:`canonical_json_bytes` representation."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise WikiSQLUAORealityError(f"{field} must be an array")
    return value


def _text(
    value: object,
    *,
    field: str,
    allow_empty: bool = False,
    forbid_nul: bool = True,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) > _MAX_TEXT_CHARACTERS
        or (forbid_nul and "\x00" in value)
        or (not allow_empty and not value.strip())
    ):
        raise WikiSQLUAORealityError(f"{field} is invalid")
    return value


def _domain_text(value: object, *, field: str) -> str:
    result = _text(value, field=field)
    if "\x00" in result:
        raise WikiSQLUAORealityError(f"{field} contains a domain separator")
    return result


def _commitment(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise WikiSQLUAORealityError(f"{field} is not a SHA-256 commitment")
    return value


def _exact_nonnegative_integer(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise WikiSQLUAORealityError(f"{field} must be a nonnegative integer")
    return value


def aggregation_name(value: object) -> str:
    """Return the documented WikiSQL aggregation name without silent repair."""

    if type(value) is int and 0 <= value < len(AGGREGATION_NAMES):
        return AGGREGATION_NAMES[value]
    if isinstance(value, str) and value in AGGREGATION_NAMES:
        return value
    raise WikiSQLUAORealityError("aggregation is outside the WikiSQL registry")


def family_from_condition_operator(value: object) -> str:
    """Map the sole WHERE operator index 0/1/2 to EQ/GT/LT."""

    if type(value) is int and 0 <= value < len(CONDITION_FAMILIES):
        return CONDITION_FAMILIES[value]
    if isinstance(value, str) and value in CONDITION_FAMILIES:
        return value
    raise WikiSQLUAORealityError(
        "condition family is outside EQ/GT/LT; sentinel OP is forbidden"
    )


def item_identity_commitment(
    *,
    source_revision_sha256: str,
    split: str,
    line_number: int,
    raw_item: Mapping[str, object],
) -> str:
    """Commit an immutable JSONL identity without exposing question or SQL.

    ``line_number`` is one-based.  The full raw object is hashed, while the
    returned commitment binds only hashes and frozen lineage coordinates.
    """

    revision = _commitment(
        source_revision_sha256, field="source revision commitment"
    )
    if split not in {"train", "dev", "test"}:
        raise WikiSQLUAORealityError("split is outside the frozen registry")
    if type(line_number) is not int or line_number < 1:
        raise WikiSQLUAORealityError("line_number must be a positive integer")
    if not isinstance(raw_item, Mapping) or set(raw_item) != OFFICIAL_ITEM_FIELDS:
        raise WikiSQLUAORealityError("item does not use the documented exact schema")
    _text(raw_item["question"], field="question")
    table_id = _text(raw_item["table_id"], field="table_id")
    if type(raw_item["phase"]) is not int or raw_item["phase"] < 1:
        raise WikiSQLUAORealityError("phase must be a positive integer")
    raw_sql = raw_item["sql"]
    if not isinstance(raw_sql, Mapping) or set(raw_sql) != OFFICIAL_SQL_FIELDS:
        raise WikiSQLUAORealityError("sql does not use the documented exact schema")
    select_index = _exact_nonnegative_integer(raw_sql["sel"], field="select index")
    raw_conditions = _sequence(raw_sql["conds"], field="sql conditions")
    condition_columns: list[int] = []
    for condition_index, condition in enumerate(raw_conditions):
        fields = _sequence(
            condition, field=f"sql condition[{condition_index}]"
        )
        if len(fields) != 3:
            raise WikiSQLUAORealityError("each sql condition must have three fields")
        condition_columns.append(
            _exact_nonnegative_integer(
                fields[0], field=f"condition[{condition_index}] column"
            )
        )
    query_from_documented_sql(
        raw_sql,
        column_count=max((select_index, *condition_columns), default=select_index) + 1,
    )
    return canonical_sha256(
        {
            "schema": f"{VERSION}_item_identity_v1",
            "source_revision_sha256": revision,
            "split": split,
            "line_number": line_number,
            "raw_item_sha256": canonical_sha256(raw_item),
            "table_id_sha256": canonical_sha256(table_id),
        }
    )


def hmac_secret_commitment(secret: bytes) -> str:
    """Commit, but never serialize, an exact 32-byte private selection key."""

    if type(secret) is not bytes or len(secret) != HMAC_SECRET_BYTES:
        raise WikiSQLUAORealityError("HMAC secret must contain exactly 32 bytes")
    return hashlib.sha256(secret).hexdigest()


def hmac_selection_digest(
    secret: bytes,
    *,
    block: str,
    family: str,
    item_commitment_sha256: str,
) -> bytes:
    """Return the frozen domain-separated private ordering digest."""

    hmac_secret_commitment(secret)
    block_text = _domain_text(block, field="selection block")
    if family not in FAMILY_ORDER:
        raise WikiSQLUAORealityError("selection family is invalid")
    item_commitment = _commitment(
        item_commitment_sha256, field="item identity commitment"
    )
    message = (
        _HMAC_PREFIX
        + block_text.encode("utf-8")
        + b"\0"
        + family.encode("ascii")
        + b"\0"
        + item_commitment.encode("ascii")
    )
    return hmac.new(secret, message, hashlib.sha256).digest()


@dataclass(frozen=True, slots=True)
class SelectionCandidate:
    """Private selector projection; no question, SQL, table, or qrel value."""

    item_commitment_sha256: str
    table_commitment_sha256: str
    family: str
    table_row_count: int
    gold_row_count: int

    def __post_init__(self) -> None:
        _commitment(
            self.item_commitment_sha256, field="candidate item commitment"
        )
        _commitment(
            self.table_commitment_sha256, field="candidate table commitment"
        )
        if self.family not in FAMILY_ORDER:
            raise WikiSQLUAORealityError("candidate family is invalid")
        if (
            type(self.table_row_count) is not int
            or not MIN_TABLE_ROWS <= self.table_row_count <= MAX_TABLE_ROWS
        ):
            raise WikiSQLUAORealityError(
                "candidate table row count is outside 11-through-80 eligibility"
            )
        if (
            type(self.gold_row_count) is not int
            or not MIN_GOLD_ROWS <= self.gold_row_count <= MAX_GOLD_ROWS
        ):
            raise WikiSQLUAORealityError(
                "candidate gold row count is outside one-through-five eligibility"
            )


def hmac_order(
    secret: bytes,
    *,
    block: str,
    candidates: Sequence[SelectionCandidate],
) -> tuple[SelectionCandidate, ...]:
    """Order candidates by private digest with a commitment collision tie-break."""

    rows = tuple(_sequence(candidates, field="selection candidates"))
    if not rows or any(not isinstance(row, SelectionCandidate) for row in rows):
        raise WikiSQLUAORealityError("selection candidates are empty or malformed")
    if len({row.item_commitment_sha256 for row in rows}) != len(rows):
        raise WikiSQLUAORealityError("selection candidates repeat an item")
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                hmac_selection_digest(
                    secret,
                    block=block,
                    family=row.family,
                    item_commitment_sha256=row.item_commitment_sha256,
                ),
                row.item_commitment_sha256,
            ),
        )
    )


def select_hmac_cohort(
    secret: bytes,
    *,
    block: str,
    candidates: Sequence[SelectionCandidate],
) -> tuple[SelectionCandidate, ...]:
    """Select one block from its already split-specific candidate pool.

    This is the exact pure-core analogue of the production compiler: choose
    one private-HMAC winner per table using ``{block}:table``, then choose the
    frozen family quotas using ``block``.  The caller must supply only the
    official TRAIN pool for A_form or only the official TEST pool for A_hold;
    the source-owning compiler enforces that split and cross-split table
    disjointness before calling the same ordering primitives.
    """

    rows = tuple(_sequence(candidates, field="selection candidates"))
    if not rows or any(not isinstance(row, SelectionCandidate) for row in rows):
        raise WikiSQLUAORealityError("selection candidates are empty or malformed")
    if len({row.item_commitment_sha256 for row in rows}) != len(rows):
        raise WikiSQLUAORealityError("selection candidates repeat an item")
    block_text = _domain_text(block, field="selection block")
    if block_text not in COHORT_QUOTAS:
        raise WikiSQLUAORealityError(
            "selection block must be frozen A_form or A_hold"
        )

    by_table: dict[str, list[SelectionCandidate]] = defaultdict(list)
    for row in rows:
        by_table[row.table_commitment_sha256].append(row)
    table_winners = tuple(
        hmac_order(
            secret,
            block=f"{block_text}:table",
            candidates=table_rows,
        )[0]
        for _, table_rows in sorted(by_table.items())
    )
    selected: list[SelectionCandidate] = []
    for family in FAMILY_ORDER:
        family_rows = tuple(
            row for row in table_winners if row.family == family
        )
        quota = COHORT_QUOTAS[block_text][family]
        if len(family_rows) < quota:
            raise WikiSQLUAORealityError(
                f"family {family} lacks the frozen post-deduplication quota"
            )
        ordered = hmac_order(
            secret,
            block=block_text,
            candidates=family_rows,
        )
        selected.extend(ordered[:quota])
    selected_tuple = tuple(selected)
    if (
        len({row.item_commitment_sha256 for row in selected_tuple})
        != len(selected_tuple)
        or len({row.table_commitment_sha256 for row in selected_tuple})
        != len(selected_tuple)
    ):
        raise WikiSQLUAORealityError(
            "selected cohort violates item/table uniqueness"
        )
    return selected_tuple


def _cell(value: object, *, field: str) -> str | int | float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise WikiSQLUAORealityError(f"{field} is not a documented scalar cell")
    if isinstance(value, str):
        _text(value, field=field, allow_empty=True)
    if isinstance(value, float) and not math.isfinite(value):
        raise WikiSQLUAORealityError(f"{field} is non-finite")
    return value


@dataclass(frozen=True, slots=True)
class WikiSQLTable:
    """Validated exact projection of a documented WikiSQL table object."""

    table_id: str
    header: tuple[str, ...]
    types: tuple[str, ...]
    rows: tuple[tuple[str | int | float, ...], ...]

    def __post_init__(self) -> None:
        _text(self.table_id, field="table id")
        if not isinstance(self.header, tuple) or not self.header:
            raise WikiSQLUAORealityError("table header must be a nonempty tuple")
        for index, value in enumerate(self.header):
            _text(value, field=f"header[{index}]")
        if (
            not isinstance(self.types, tuple)
            or len(self.types) != len(self.header)
            or any(value not in TABLE_TYPES for value in self.types)
        ):
            raise WikiSQLUAORealityError("table types do not match the header")
        if not isinstance(self.rows, tuple) or not self.rows:
            raise WikiSQLUAORealityError("table must contain at least one row")
        normalized_rows: list[tuple[str | int | float, ...]] = []
        for row_index, row in enumerate(self.rows):
            if not isinstance(row, tuple) or len(row) != len(self.header):
                raise WikiSQLUAORealityError(
                    f"row[{row_index}] width does not match the header"
                )
            normalized_rows.append(
                tuple(
                    _cell(value, field=f"rows[{row_index}][{column_index}]")
                    for column_index, value in enumerate(row)
                )
            )
        object.__setattr__(self, "rows", tuple(normalized_rows))


def table_from_documented_schema(value: Mapping[str, object]) -> WikiSQLTable:
    """Validate the exact ``id/header/types/rows`` release table schema."""

    if not isinstance(value, Mapping) or set(value) != OFFICIAL_TABLE_FIELDS:
        raise WikiSQLUAORealityError("table does not use the documented exact schema")
    header = tuple(_sequence(value["header"], field="table header"))
    types = tuple(_sequence(value["types"], field="table types"))
    raw_rows = _sequence(value["rows"], field="table rows")
    rows = tuple(
        tuple(_sequence(row, field=f"table row {index}"))
        for index, row in enumerate(raw_rows)
    )
    return WikiSQLTable(
        table_id=value["id"],  # type: ignore[arg-type]
        header=header,  # type: ignore[arg-type]
        types=types,  # type: ignore[arg-type]
        rows=rows,  # type: ignore[arg-type]
    )


def _display_text(value: str) -> str:
    return _WHITESPACE_RE.sub(
        " ", unicodedata.normalize("NFKC", value)
    ).strip()


def _display_cell(value: str | int | float) -> str:
    if isinstance(value, str):
        normalized = _display_text(value)
        return normalized if normalized else "<EMPTY>"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def serialize_table_row(table: WikiSQLTable, row_index: int) -> str:
    """Serialize all row fields identically for every retrieval arm."""

    checked = _checked_table(table)
    ordinal = _exact_nonnegative_integer(row_index, field="row_index")
    if ordinal >= len(checked.rows):
        raise WikiSQLUAORealityError("row_index is outside the table")
    row = checked.rows[ordinal]
    return "\n".join(
        f'column[{column_index}] "{_display_text(header)}" '
        f"({column_type}) = {_display_cell(cell)}"
        for column_index, (header, column_type, cell) in enumerate(
            zip(checked.header, checked.types, row, strict=True)
        )
    )


def serialize_table_rows(table: WikiSQLTable) -> tuple[str, ...]:
    """Serialize every physical row in stable ordinal order."""

    checked = _checked_table(table)
    return tuple(serialize_table_row(checked, index) for index in range(len(checked.rows)))


def parse_serialized_table_row_values(
    serialized: str,
    headers: Sequence[str],
    types: Sequence[str],
) -> tuple[str, ...]:
    """Parse one frozen row document using its exact known schema prefix.

    The parser never searches for a delimiter inside the document.  This is
    important because legitimate header or text-cell content may itself
    contain strings such as ``" (text) = ``.
    """

    if (
        not isinstance(serialized, str)
        or "\x00" in serialized
        or not serialized.strip()
        or len(serialized) > MAX_SERIALIZED_ROW_CHARACTERS
    ):
        raise WikiSQLUAORealityError(
            "serialized row is empty, contains NUL, or exceeds the frozen bound"
        )
    checked_headers = tuple(
        _text(value, field=f"serialized header[{index}]")
        for index, value in enumerate(
            _sequence(headers, field="serialized headers")
        )
    )
    checked_types = tuple(
        _text(value, field=f"serialized type[{index}]")
        for index, value in enumerate(
            _sequence(types, field="serialized types")
        )
    )
    if (
        not checked_headers
        or len(checked_headers) != len(checked_types)
        or any(value not in TABLE_TYPES for value in checked_types)
    ):
        raise WikiSQLUAORealityError(
            "serialized row schema does not match WikiSQL columns"
        )
    text = unicodedata.normalize("NFKC", serialized).strip()
    lines = text.split("\n")
    if len(lines) != len(checked_headers):
        raise WikiSQLUAORealityError(
            "serialized row line count does not match the schema"
        )
    values: list[str] = []
    for index, (line, header, column_type) in enumerate(
        zip(lines, checked_headers, checked_types, strict=True)
    ):
        prefix = (
            f'column[{index}] "{_display_text(header)}" '
            f"({column_type}) = "
        )
        if not line.startswith(prefix):
            raise WikiSQLUAORealityError(
                "serialized row does not use the exact schema prefix"
            )
        values.append(line[len(prefix) :])
    return tuple(values)


def validated_retrieval_documents(
    table: WikiSQLTable,
) -> tuple[str, ...]:
    """Return rows only when every retrieval arm can represent them exactly."""

    checked = _checked_table(table)
    documents = serialize_table_rows(checked)
    if len(set(documents)) != len(documents):
        raise WikiSQLUAORealityError(
            "canonical row documents must be unique"
        )
    for row, document in zip(checked.rows, documents, strict=True):
        observed = parse_serialized_table_row_values(
            document,
            checked.header,
            checked.types,
        )
        expected = tuple(_display_cell(value) for value in row)
        if observed != expected:
            raise WikiSQLUAORealityError(
                "serialized row failed the frozen round-trip contract"
            )
    return documents


def _checked_table(table: object) -> WikiSQLTable:
    if not isinstance(table, WikiSQLTable):
        raise WikiSQLUAORealityError("table is not a validated WikiSQLTable")
    return WikiSQLTable(
        table_id=table.table_id,
        header=table.header,
        types=table.types,
        rows=table.rows,
    )


def lexical_tokens(value: str) -> tuple[str, ...]:
    """Frozen NFKC/casefold Unicode tokens used only by RAW BM25."""

    text = _text(value, field="lexical text")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens = tuple(_TOKEN_RE.findall(normalized))
    if not tokens:
        raise WikiSQLUAORealityError("lexical text contains no token")
    return tokens


def bm25_scores(question: str, row_texts: Sequence[str]) -> tuple[int, ...]:
    """Return quantized Okapi BM25 scores over a complete item-local table."""

    query_terms = lexical_tokens(question)
    texts = tuple(_sequence(row_texts, field="BM25 row texts"))
    if not texts or any(not isinstance(value, str) for value in texts):
        raise WikiSQLUAORealityError("BM25 row texts are empty or malformed")
    documents = tuple(lexical_tokens(value) for value in texts)
    average_length = sum(len(row) for row in documents) / len(documents)
    document_frequency: Counter[str] = Counter()
    for row in documents:
        document_frequency.update(set(row))
    query_frequency = Counter(query_terms)

    result: list[int] = []
    for row in documents:
        term_frequency = Counter(row)
        score = 0.0
        for term, query_count in query_frequency.items():
            frequency = term_frequency.get(term, 0)
            if frequency == 0:
                continue
            frequency_across_documents = document_frequency[term]
            inverse_document_frequency = math.log(
                1.0
                + (
                    len(documents)
                    - frequency_across_documents
                    + 0.5
                )
                / (frequency_across_documents + 0.5)
            )
            denominator = frequency + BM25_K1 * (
                1.0 - BM25_B + BM25_B * len(row) / average_length
            )
            score += (
                query_count
                * inverse_document_frequency
                * frequency
                * (BM25_K1 + 1.0)
                / denominator
            )
        result.append(int(round(score * BM25_INTEGER_SCALE)))
    return tuple(result)


def raw_bm25_top5(
    question: str, table: WikiSQLTable
) -> tuple[int | None, ...]:
    """Return the frozen RAW top five for an eligible 11--80 row table."""

    checked = _checked_table(table)
    if not MIN_TABLE_ROWS <= len(checked.rows) <= MAX_TABLE_ROWS:
        raise WikiSQLUAORealityError(
            "RAW table is outside 11-through-80 row eligibility"
        )
    scores = bm25_scores(question, serialize_table_rows(checked))
    ranked = sorted(
        range(len(scores)),
        key=lambda index: (-scores[index], index),
    )
    result: list[int | None] = list(ranked[:TOP_K])
    result.extend([None] * (TOP_K - len(result)))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class WikiSQLCondition:
    column_index: int
    operator_index: int
    value: str | int | float

    def __post_init__(self) -> None:
        _exact_nonnegative_integer(self.column_index, field="condition column")
        if (
            type(self.operator_index) is not int
            or not 0 <= self.operator_index < len(CONDITION_OPERATORS)
        ):
            raise WikiSQLUAORealityError("condition operator is unsupported")
        _cell(self.value, field="condition value")


@dataclass(frozen=True, slots=True)
class WikiSQLQuery:
    select_index: int
    aggregation_index: int
    conditions: tuple[WikiSQLCondition, ...]

    def __post_init__(self) -> None:
        _exact_nonnegative_integer(self.select_index, field="select index")
        aggregation_name(self.aggregation_index)
        if (
            not isinstance(self.conditions, tuple)
            or len(self.conditions) != 1
            or any(
                not isinstance(row, WikiSQLCondition)
                for row in self.conditions
            )
        ):
            raise WikiSQLUAORealityError(
                "query must contain exactly one WHERE condition"
            )

    @property
    def family(self) -> str:
        return family_from_condition_operator(self.conditions[0].operator_index)


def query_from_documented_sql(
    value: Mapping[str, object], *, column_count: int
) -> WikiSQLQuery:
    """Validate the exact ``sel/agg/conds`` documented SQL object."""

    if not isinstance(value, Mapping) or set(value) != OFFICIAL_SQL_FIELDS:
        raise WikiSQLUAORealityError("sql does not use the documented exact schema")
    width = _exact_nonnegative_integer(column_count, field="column_count")
    if width == 0:
        raise WikiSQLUAORealityError("column_count must be positive")
    select_index = _exact_nonnegative_integer(value["sel"], field="select index")
    if select_index >= width:
        raise WikiSQLUAORealityError("select index is outside the table")
    aggregation_index = value["agg"]
    aggregation_name(aggregation_index)
    raw_conditions = _sequence(value["conds"], field="sql conditions")
    if len(raw_conditions) != 1:
        raise WikiSQLUAORealityError(
            "sql must contain exactly one WHERE condition"
        )
    conditions: list[WikiSQLCondition] = []
    for condition_index, raw_condition in enumerate(raw_conditions):
        fields = _sequence(
            raw_condition, field=f"sql condition[{condition_index}]"
        )
        if len(fields) != 3:
            raise WikiSQLUAORealityError("each sql condition must have three fields")
        column_index = _exact_nonnegative_integer(
            fields[0], field=f"condition[{condition_index}] column"
        )
        if column_index >= width:
            raise WikiSQLUAORealityError("condition column is outside the table")
        conditions.append(
            WikiSQLCondition(
                column_index=column_index,
                operator_index=fields[1],  # type: ignore[arg-type]
                value=fields[2],  # type: ignore[arg-type]
            )
        )
    return WikiSQLQuery(
        select_index=select_index,
        aggregation_index=aggregation_index,  # type: ignore[arg-type]
        conditions=tuple(conditions),
    )


def _coerce_text(value: str | int | float) -> str:
    # Official WikiSQL uses lower(), not casefold(), for its execution query.
    return str(value).lower()


def _coerce_real(value: str | int | float) -> float:
    if isinstance(value, bool):
        raise WikiSQLUAORealityError("boolean cannot be coerced to WikiSQL real")
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        text = unicodedata.normalize("NFKC", value).strip()
        if not text:
            raise WikiSQLUAORealityError("empty real value cannot be coerced")
        if _ENGLISH_DECIMAL_RE.fullmatch(text):
            candidate = text.replace(",", "")
        else:
            # Exact relevant WikiSQL 1.1 DBEngine fallback: take the first
            # ``[-+]?\d*\.\d+|\d+`` match after decimal parsing fails.
            matches = _NUMERIC_FALLBACK_RE.findall(text)
            if not matches:
                raise WikiSQLUAORealityError("real value cannot be coerced")
            candidate = matches[0]
        if (
            not candidate
            or candidate == "."
            or candidate.count(".") > 1
        ):
            raise WikiSQLUAORealityError("real value cannot be coerced")
        try:
            result = float(candidate)
        except ValueError as exc:
            raise WikiSQLUAORealityError("real value cannot be coerced") from exc
    if not math.isfinite(result):
        raise WikiSQLUAORealityError("coerced real value is non-finite")
    return result


def _condition_matches(
    *,
    row_value: str | int | float,
    condition_value: str | int | float,
    column_type: str,
    operator_index: int,
) -> bool:
    if column_type == "text":
        left: str | float = _coerce_text(row_value)
        right: str | float = _coerce_text(condition_value)
    elif column_type == "real":
        left = _coerce_real(row_value)
        right = _coerce_real(condition_value)
    else:  # defensive even though WikiSQLTable already validates this.
        raise WikiSQLUAORealityError("condition column type is unsupported")
    if operator_index == 0:
        return left == right
    if operator_index == 1:
        return left > right
    if operator_index == 2:
        return left < right
    raise WikiSQLUAORealityError("condition operator is unsupported")


def derive_gold_row_ids(
    table: WikiSQLTable,
    sql: Mapping[str, object] | WikiSQLQuery,
) -> tuple[int, ...]:
    """Return all physical rows satisfying every gold WHERE condition."""

    checked = _checked_table(table)
    query = (
        query_from_documented_sql(sql, column_count=len(checked.header))
        if isinstance(sql, Mapping)
        else sql
    )
    if not isinstance(query, WikiSQLQuery):
        raise WikiSQLUAORealityError("sql is not a mapping or WikiSQLQuery")
    # Revalidate against this table even if a caller supplied a constructed
    # query that was originally checked against another width.
    if query.select_index >= len(checked.header) or any(
        condition.column_index >= len(checked.header)
        for condition in query.conditions
    ):
        raise WikiSQLUAORealityError("query column is outside the supplied table")
    return tuple(
        row_index
        for row_index, row in enumerate(checked.rows)
        if all(
            _condition_matches(
                row_value=row[condition.column_index],
                condition_value=condition.value,
                column_type=checked.types[condition.column_index],
                operator_index=condition.operator_index,
            )
            for condition in query.conditions
        )
    )


@dataclass(frozen=True, slots=True)
class EligibleGoldLabel:
    """Late-label projection for a fixed eligible single-WHERE item."""

    family: str
    table_row_count: int
    gold_row_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.family not in FAMILY_ORDER:
            raise WikiSQLUAORealityError("eligible label family is invalid")
        if (
            type(self.table_row_count) is not int
            or not MIN_TABLE_ROWS <= self.table_row_count <= MAX_TABLE_ROWS
        ):
            raise WikiSQLUAORealityError(
                "eligible label table must contain 11 through 80 rows"
            )
        object.__setattr__(
            self, "gold_row_ids", _validated_gold(self.gold_row_ids)
        )


def derive_eligible_gold_label(
    table: WikiSQLTable,
    sql: Mapping[str, object] | WikiSQLQuery,
) -> EligibleGoldLabel:
    """Derive and enforce the frozen table/gold-cardinality eligibility."""

    checked = _checked_table(table)
    if not MIN_TABLE_ROWS <= len(checked.rows) <= MAX_TABLE_ROWS:
        raise WikiSQLUAORealityError(
            "table must contain 11 through 80 physical rows"
        )
    query = (
        query_from_documented_sql(sql, column_count=len(checked.header))
        if isinstance(sql, Mapping)
        else sql
    )
    if not isinstance(query, WikiSQLQuery):
        raise WikiSQLUAORealityError("sql is not a mapping or WikiSQLQuery")
    gold = derive_gold_row_ids(checked, query)
    if not MIN_GOLD_ROWS <= len(gold) <= MAX_GOLD_ROWS:
        raise WikiSQLUAORealityError(
            "gold must contain one through five physical rows"
        )
    return EligibleGoldLabel(
        family=query.family,
        table_row_count=len(checked.rows),
        gold_row_ids=gold,
    )


def _validated_top5(value: Sequence[int | None], *, field: str) -> tuple[int | None, ...]:
    rows = tuple(_sequence(value, field=field))
    if len(rows) != TOP_K:
        raise WikiSQLUAORealityError(f"{field} must contain exactly five slots")
    seen_null = False
    selected: list[int] = []
    for row in rows:
        if row is None:
            seen_null = True
            continue
        if seen_null:
            raise WikiSQLUAORealityError(f"{field} null padding must be trailing")
        selected.append(_exact_nonnegative_integer(row, field=field))
    if len(set(selected)) != len(selected):
        raise WikiSQLUAORealityError(f"{field} repeats a physical row")
    return rows  # type: ignore[return-value]


def _validated_gold(value: Sequence[int]) -> tuple[int, ...]:
    rows = tuple(_sequence(value, field="gold row ids"))
    if (
        not 1 <= len(rows) <= TOP_K
        or any(type(row) is not int or row < 0 for row in rows)
        or tuple(sorted(set(rows))) != rows
    ):
        raise WikiSQLUAORealityError(
            "gold row ids must be one-through-five sorted distinct ordinals"
        )
    return rows  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ItemUtility:
    hits: int
    complete: bool
    utility: int

    def __post_init__(self) -> None:
        if (
            type(self.hits) is not int
            or not 0 <= self.hits <= TOP_K
            or type(self.complete) is not bool
            or type(self.utility) is not int
            or self.utility != self.hits + int(self.complete)
        ):
            raise WikiSQLUAORealityError("item utility is malformed")


def score_item(
    selected_top5: Sequence[int | None],
    gold_row_ids: Sequence[int],
) -> ItemUtility:
    """Score the frozen integer ``U = hits + complete`` item utility."""

    selected = _validated_top5(selected_top5, field="selected top5")
    gold = _validated_gold(gold_row_ids)
    selected_set = {row for row in selected if row is not None}
    hits = len(selected_set.intersection(gold))
    complete = set(gold).issubset(selected_set)
    return ItemUtility(hits=hits, complete=complete, utility=hits + int(complete))


def item_utility(
    selected_top5: Sequence[int | None],
    gold_row_ids: Sequence[int],
) -> int:
    """Return only the exact integer item utility."""

    return score_item(selected_top5, gold_row_ids).utility


@dataclass(frozen=True, slots=True)
class ExactSignFlipResult:
    observed_net_u: int
    nonzero_pair_count: int
    p_value: Fraction

    def __post_init__(self) -> None:
        if (
            type(self.observed_net_u) is not int
            or type(self.nonzero_pair_count) is not int
            or self.nonzero_pair_count < 0
            or not isinstance(self.p_value, Fraction)
            or not 0 <= self.p_value <= 1
            or (1 << self.nonzero_pair_count) % self.p_value.denominator != 0
        ):
            raise WikiSQLUAORealityError("exact sign-flip result is malformed")

    @property
    def positive_at_alpha(self) -> bool:
        return self.observed_net_u > 0 and self.p_value <= PROMOTION_ALPHA


def exact_magnitude_preserving_sign_flip(
    deltas: Sequence[int],
) -> ExactSignFlipResult:
    """Compute the one-sided exact magnitude-preserving reference tail."""

    rows = tuple(_sequence(deltas, field="paired utility deltas"))
    if (
        not rows
        or any(type(value) is not int or not -6 <= value <= 6 for value in rows)
    ):
        raise WikiSQLUAORealityError(
            "paired utility deltas must be nonempty exact integers in [-6, 6]"
        )
    observed = sum(rows)
    magnitudes = tuple(abs(value) for value in rows if value)
    distribution: Counter[int] = Counter({0: 1})
    for magnitude in magnitudes:
        updated: Counter[int] = Counter()
        for subtotal, count in distribution.items():
            updated[subtotal + magnitude] += count
            updated[subtotal - magnitude] += count
        distribution = updated
    denominator = 1 << len(magnitudes)
    if sum(distribution.values()) != denominator:
        raise WikiSQLUAORealityError("exact sign-flip probability mass drifted")
    numerator = sum(
        count for subtotal, count in distribution.items() if subtotal >= observed
    )
    return ExactSignFlipResult(
        observed_net_u=observed,
        nonzero_pair_count=len(magnitudes),
        p_value=Fraction(numerator, denominator),
    )


@dataclass(frozen=True, slots=True)
class ItemMeasurement:
    """Already-sealed three-arm actions joined to a late-opened gold label."""

    item_commitment_sha256: str
    family: str
    gold_row_ids: tuple[int, ...]
    agent_top5: tuple[int | None, ...]
    raw_top5: tuple[int | None, ...]
    hipporag_top5: tuple[int | None, ...]

    def __post_init__(self) -> None:
        _commitment(self.item_commitment_sha256, field="measurement item commitment")
        if self.family not in FAMILY_ORDER:
            raise WikiSQLUAORealityError("measurement family is invalid")
        object.__setattr__(self, "gold_row_ids", _validated_gold(self.gold_row_ids))
        for field in ("agent_top5", "raw_top5", "hipporag_top5"):
            object.__setattr__(
                self,
                field,
                _validated_top5(getattr(self, field), field=field),
            )

    def utility(self, arm: str) -> int:
        if arm not in MEASUREMENT_ARMS:
            raise WikiSQLUAORealityError("measurement arm is invalid")
        return item_utility(getattr(self, f"{arm}_top5"), self.gold_row_ids)


@dataclass(frozen=True, slots=True)
class BaselineComparison:
    baseline: str
    observed_net_u: int
    family_net_u: tuple[tuple[str, int], ...]
    sign_flip: ExactSignFlipResult
    passed: bool

    def __post_init__(self) -> None:
        if self.baseline not in BASELINE_ARMS:
            raise WikiSQLUAORealityError("comparison baseline is invalid")
        if type(self.observed_net_u) is not int:
            raise WikiSQLUAORealityError("comparison net utility is invalid")
        if (
            tuple(family for family, _ in self.family_net_u) != FAMILY_ORDER
            or any(type(value) is not int for _, value in self.family_net_u)
            or not isinstance(self.sign_flip, ExactSignFlipResult)
        ):
            raise WikiSQLUAORealityError("comparison family/sign-flip fields drifted")
        expected = (
            self.observed_net_u > 0
            and self.sign_flip.observed_net_u == self.observed_net_u
            and self.sign_flip.p_value <= PROMOTION_ALPHA
            and all(value > 0 for _, value in self.family_net_u)
        )
        if self.passed is not expected:
            raise WikiSQLUAORealityError("comparison decision drifted")


@dataclass(frozen=True, slots=True)
class PrimaryAggregation:
    item_count: int
    family_counts: tuple[tuple[str, int], ...]
    agent_vs_raw: BaselineComparison
    agent_vs_hipporag: BaselineComparison
    passed: bool

    def __post_init__(self) -> None:
        if (
            type(self.item_count) is not int
            or self.item_count <= 0
            or tuple(family for family, _ in self.family_counts) != FAMILY_ORDER
            or any(type(count) is not int or count <= 0 for _, count in self.family_counts)
            or sum(count for _, count in self.family_counts) != self.item_count
            or self.agent_vs_raw.baseline != "raw"
            or self.agent_vs_hipporag.baseline != "hipporag"
        ):
            raise WikiSQLUAORealityError("primary aggregation fields drifted")
        expected = self.agent_vs_raw.passed and self.agent_vs_hipporag.passed
        if self.passed is not expected:
            raise WikiSQLUAORealityError("primary intersection decision drifted")


def _comparison(
    measurements: tuple[ItemMeasurement, ...], baseline: str
) -> BaselineComparison:
    deltas = tuple(
        row.utility("agent") - row.utility(baseline) for row in measurements
    )
    family_nets = tuple(
        (
            family,
            sum(
                delta
                for row, delta in zip(measurements, deltas, strict=True)
                if row.family == family
            ),
        )
        for family in FAMILY_ORDER
    )
    exact = exact_magnitude_preserving_sign_flip(deltas)
    passed = (
        exact.observed_net_u > 0
        and exact.p_value <= PROMOTION_ALPHA
        and all(value > 0 for _, value in family_nets)
    )
    return BaselineComparison(
        baseline=baseline,
        observed_net_u=sum(deltas),
        family_net_u=family_nets,
        sign_flip=exact,
        passed=passed,
    )


def aggregate_primary(
    measurements: Sequence[ItemMeasurement],
) -> PrimaryAggregation:
    """Evaluate the frozen Agent-vs-both-baselines intersection claim."""

    rows = tuple(_sequence(measurements, field="item measurements"))
    if not rows or any(not isinstance(row, ItemMeasurement) for row in rows):
        raise WikiSQLUAORealityError("item measurements are empty or malformed")
    if len({row.item_commitment_sha256 for row in rows}) != len(rows):
        raise WikiSQLUAORealityError("item measurements repeat a commitment")
    counts = Counter(row.family for row in rows)
    if set(counts) != set(FAMILY_ORDER):
        raise WikiSQLUAORealityError("all three frozen families must be present")
    family_counts = tuple((family, counts[family]) for family in FAMILY_ORDER)
    raw = _comparison(rows, "raw")
    hipporag = _comparison(rows, "hipporag")
    return PrimaryAggregation(
        item_count=len(rows),
        family_counts=family_counts,
        agent_vs_raw=raw,
        agent_vs_hipporag=hipporag,
        passed=raw.passed and hipporag.passed,
    )


__all__ = [
    "AGGREGATION_NAMES",
    "BASELINE_ARMS",
    "BM25_B",
    "BM25_INTEGER_SCALE",
    "BM25_K1",
    "BaselineComparison",
    "COHORT_QUOTAS",
    "CONDITION_FAMILIES",
    "CONDITION_OPERATORS",
    "EligibleGoldLabel",
    "ExactSignFlipResult",
    "FAMILY_ORDER",
    "HMAC_SECRET_BYTES",
    "ItemMeasurement",
    "ItemUtility",
    "MAX_GOLD_ROWS",
    "MAX_SERIALIZED_ROW_CHARACTERS",
    "MAX_TABLE_ROWS",
    "MEASUREMENT_ARMS",
    "MIN_GOLD_ROWS",
    "MIN_TABLE_ROWS",
    "PROMOTION_ALPHA",
    "PrimaryAggregation",
    "STUDY_ID",
    "SelectionCandidate",
    "TABLE_TYPES",
    "TOP_K",
    "VERSION",
    "WikiSQLCondition",
    "WikiSQLQuery",
    "WikiSQLTable",
    "WikiSQLUAORealityError",
    "aggregate_primary",
    "aggregation_name",
    "bm25_scores",
    "canonical_json_bytes",
    "canonical_sha256",
    "derive_eligible_gold_label",
    "derive_gold_row_ids",
    "exact_magnitude_preserving_sign_flip",
    "family_from_condition_operator",
    "hmac_order",
    "hmac_secret_commitment",
    "hmac_selection_digest",
    "item_identity_commitment",
    "item_utility",
    "lexical_tokens",
    "query_from_documented_sql",
    "raw_bm25_top5",
    "parse_serialized_table_row_values",
    "score_item",
    "select_hmac_cohort",
    "serialize_table_row",
    "serialize_table_rows",
    "table_from_documented_schema",
    "validated_retrieval_documents",
]
