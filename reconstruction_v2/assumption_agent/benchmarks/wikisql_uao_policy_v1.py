"""Pure, offline UAO policy formation for the frozen WikiSQL reality study.

The module is deliberately source agnostic.  It has no filesystem, network,
dataset, SQL, model-download, HippoRAG, or online-evaluator entrypoint.  Both
formation and application operate on caller-supplied, already serialized table
rows.  Gold rows and source-family labels exist only on :class:`TrainingItem`;
the held-out :func:`apply_uao_policy` signature cannot accept either.

Four fixed claim recipes instantiate the source-free UAO operator registry:

* ``T02`` -- sparse typed anchor action;
* ``T05`` -- low-order header/value and comparator interaction;
* ``T08`` -- item-local neighbourhood expansion around RAW anchors; and
* ``T18`` -- sparse-contamination suppression.

Formation consumes sealed four-fold TRAIN assignments, selects exactly two
action-vector-distinct claims with a fixed MDL/redundancy score, then calibrates
one no-op margin on union-policy out-of-fold predictions over a frozen finite
integer grid.  The final ridge-logistic union changes RAW only when its learned,
quantized expected-utility margin is *strictly greater* than that TRAIN-only
threshold.  Consequently an abstention returns the caller's RAW top-five tuple
byte-for-byte, including its order and trailing ``None`` padding.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
import hashlib
import json
import math
import re
import unicodedata
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from assumption_agent.benchmarks import wikisql_uao_reality_v1 as reality


VERSION = "wikisql_uao_policy_v1"
TOP_K = 5
CROSS_FOLDS = 4
SELECTED_CLAIM_COUNT = 2
RIDGE_LAMBDA = 1.0
IRLS_STEPS = 32
PROBABILITY_SCALE = 1_000_000
UTILITY_MARGIN_SCALE = 1_000_000
# Frozen before any source payload.  The maximum item utility is six
# (five hits plus complete), so the final point is an explicit all-no-op
# option.  A_hold can neither extend nor select this grid.
NO_OP_MARGIN_GRID = (
    0,
    10_000,
    25_000,
    50_000,
    100_000,
    200_000,
    400_000,
    800_000,
    1_200_000,
    2_000_000,
    3_000_000,
    4_000_000,
    6_000_000,
)
# Source-native measurement strata are the single WHERE-condition operators,
# never the SELECT aggregation: equality, greater-than, and less-than.
FAMILY_ORDER = ("EQ", "GT", "LT")
A_FORM_QUOTA_PER_FAMILY = 64
A_HOLD_QUOTA_PER_FAMILY = 24
A_FORM_QUOTA_PER_FOLD_FAMILY = A_FORM_QUOTA_PER_FAMILY // CROSS_FOLDS
ACTION_VECTOR_LENGTH = A_FORM_QUOTA_PER_FAMILY * len(FAMILY_ORDER)
ACTION_VECTOR_MIN_HAMMING = 16
ACTION_REDUNDANCY_PENALTY_PER_MATCH = 10

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_NUMBER_RE = re.compile(
    r"(?<![\w.])(?:[$€£¥]\s*)?[+-]?"
    r"(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?%?(?![\w.])"
)
_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)
_GT_PHRASES = (
    "greater than",
    "more than",
    "higher than",
    "above",
    "over",
    "after",
    "later than",
    "exceed",
)
_LT_PHRASES = (
    "less than",
    "lower than",
    "below",
    "under",
    "before",
    "earlier than",
)
_EQ_PHRASES = (
    "equal to",
    "equals",
    "exactly",
    "same as",
)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    }
)


class WikiSQLUAOPolicyError(ValueError):
    """A pure-policy input or frozen formation contract drifted."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return a deterministic finite JSON representation."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise WikiSQLUAOPolicyError("value is not canonical JSON") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _content_addressed(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    if "self_sha256" in result:
        raise WikiSQLUAOPolicyError("receipt payload already contains self_sha256")
    result["self_sha256"] = canonical_sha256(result)
    return result


def _text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise WikiSQLUAOPolicyError(f"{field} must be NUL-free text")
    normalized = _WHITESPACE_RE.sub(
        " ", unicodedata.normalize("NFKC", value)
    ).strip()
    if not allow_empty and not normalized:
        raise WikiSQLUAOPolicyError(f"{field} must be nonempty")
    return normalized


def normalize_text(value: str) -> str:
    """NFKC/casefold text normalization shared by all four claims."""

    return _text(value, field="text", allow_empty=True).casefold()


def text_tokens(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(normalize_text(value)))


def normalize_number(value: object) -> float | None:
    """Totalized English-decimal normalization for typed numeric anchors.

    Currency marks, grouping commas, accounting parentheses, and percentages
    are recognized.  Percentages remain in their displayed scale (``12%`` is
    ``12.0``), matching lexical question/table comparisons.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if not isinstance(value, str):
        return None
    candidate = unicodedata.normalize("NFKC", value).strip()
    if not candidate:
        return None
    negative = candidate.startswith("(") and candidate.endswith(")")
    if negative:
        candidate = candidate[1:-1].strip()
    candidate = candidate.replace(",", "")
    candidate = re.sub(r"^[$€£¥]\s*", "", candidate)
    if candidate.endswith("%"):
        candidate = candidate[:-1].strip()
    try:
        result = float(Decimal(candidate))
    except (InvalidOperation, ValueError, OverflowError):
        return None
    if negative:
        result = -result
    return result if math.isfinite(result) else None


@dataclass(frozen=True, slots=True)
class QuestionAnchors:
    tokens: tuple[str, ...]
    content_tokens: tuple[str, ...]
    numbers: tuple[float, ...]
    comparator: str

    def __post_init__(self) -> None:
        if self.comparator not in {"EQ", "GT", "LT"}:
            raise WikiSQLUAOPolicyError("question comparator is invalid")
        if any(not math.isfinite(row) for row in self.numbers):
            raise WikiSQLUAOPolicyError("question number is non-finite")


def extract_anchors(question: str) -> QuestionAnchors:
    """Extract fixed lexical, numeric, and EQ/GT/LT question anchors."""

    normalized = normalize_text(question)
    tokens = text_tokens(normalized)
    content = tuple(token for token in tokens if token not in _STOPWORDS)
    numbers: list[float] = []
    for match in _NUMBER_RE.finditer(normalized):
        candidate = normalize_number(match.group(0))
        if candidate is not None and candidate not in numbers:
            numbers.append(candidate)
    if any(phrase in normalized for phrase in _GT_PHRASES):
        comparator = "GT"
    elif any(phrase in normalized for phrase in _LT_PHRASES):
        comparator = "LT"
    elif any(phrase in normalized for phrase in _EQ_PHRASES):
        comparator = "EQ"
    else:
        # WikiSQL equality conditions are commonly phrased without the word
        # "equals"; equality is therefore the frozen default.
        comparator = "EQ"
    return QuestionAnchors(
        tokens=tokens,
        content_tokens=content,
        numbers=tuple(numbers),
        comparator=comparator,
    )


@dataclass(frozen=True, slots=True)
class PrecomputedEmbeddings:
    """Caller-owned deterministic dense features.

    The model commitment is documentary and prevents silently combining
    vectors from different frozen encoders.  There is one query vector and one
    row vector per serialized row.
    """

    model_sha256: str
    question: tuple[float, ...]
    rows: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model_sha256, str)
            or _SHA256_RE.fullmatch(self.model_sha256) is None
        ):
            raise WikiSQLUAOPolicyError(
                "embedding model_sha256 is not a SHA-256 commitment"
            )
        if not self.question or any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in self.question
        ):
            raise WikiSQLUAOPolicyError("question embedding is malformed")
        width = len(self.question)
        if any(
            not isinstance(row, tuple)
            or len(row) != width
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in row
            )
            for row in self.rows
        ):
            raise WikiSQLUAOPolicyError("row embeddings are malformed")

    def private_payload(self) -> dict[str, object]:
        return {
            "schema": f"{VERSION}_precomputed_embeddings_private_v1",
            "model_sha256": self.model_sha256,
            "question": list(self.question),
            "rows": [list(row) for row in self.rows],
        }

    @property
    def embeddings_sha256(self) -> str:
        return canonical_sha256(self.private_payload())

    def content_addressed_payload(self) -> dict[str, object]:
        return _content_addressed(self.private_payload())


def _array(value: object, *, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise WikiSQLUAOPolicyError(f"{field} must be an array")
    return value


def _verify_content_addressed_payload(
    value: Mapping[str, object],
    *,
    exact_fields: frozenset[str],
    field: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != exact_fields | {"self_sha256"}:
        raise WikiSQLUAOPolicyError(f"{field} schema contains missing or extra fields")
    supplied_hash = value["self_sha256"]
    if not isinstance(supplied_hash, str) or _SHA256_RE.fullmatch(supplied_hash) is None:
        raise WikiSQLUAOPolicyError(f"{field} self_sha256 is malformed")
    payload = {key: row for key, row in value.items() if key != "self_sha256"}
    if canonical_sha256(payload) != supplied_hash:
        raise WikiSQLUAOPolicyError(f"{field} content hash mismatch")
    return payload


def _finite_numeric(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WikiSQLUAOPolicyError(f"{field} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise WikiSQLUAOPolicyError(f"{field} is non-finite")
    return result


def precomputed_embeddings_from_payload(
    value: Mapping[str, object],
) -> PrecomputedEmbeddings:
    """Reconstruct a content-addressed embedding bundle without pickle."""

    fields = frozenset({"schema", "model_sha256", "question", "rows"})
    payload = _verify_content_addressed_payload(
        value,
        exact_fields=fields,
        field="precomputed embeddings",
    )
    if payload["schema"] != f"{VERSION}_precomputed_embeddings_private_v1":
        raise WikiSQLUAOPolicyError("precomputed embeddings schema drifted")
    question_values = _array(
        payload["question"], field="precomputed embedding question"
    )
    row_values = _array(payload["rows"], field="precomputed embedding rows")
    question = tuple(
        _finite_numeric(row, field="question embedding value")
        for row in question_values
    )
    rows = tuple(
        tuple(
            _finite_numeric(cell, field="row embedding value")
            for cell in _array(row, field="precomputed embedding row")
        )
        for row in row_values
    )
    result = PrecomputedEmbeddings(
        model_sha256=payload["model_sha256"],  # type: ignore[arg-type]
        question=question,
        rows=rows,
    )
    if result.embeddings_sha256 != value["self_sha256"]:
        raise WikiSQLUAOPolicyError("precomputed embedding reconstruction drifted")
    return result


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


@dataclass(frozen=True, slots=True)
class ParsedRow:
    values: tuple[str, ...]
    value_tokens: tuple[tuple[str, ...], ...]
    numbers: tuple[float | None, ...]


def _display_header(value: str) -> str:
    return _WHITESPACE_RE.sub(
        " ", unicodedata.normalize("NFKC", value)
    ).strip()


def _parse_serialized_row(
    serialized: str,
    headers: tuple[str, ...],
    types: tuple[str, ...],
) -> ParsedRow:
    try:
        values = reality.parse_serialized_table_row_values(
            serialized,
            headers,
            types,
        )
    except reality.WikiSQLUAORealityError as exc:
        raise WikiSQLUAOPolicyError(
            "serialized row is outside the shared retrieval contract"
        ) from exc
    token_rows: list[tuple[str, ...]] = []
    numbers: list[float | None] = []
    for value, column_type in zip(values, types, strict=True):
        token_rows.append(text_tokens("" if value == "<EMPTY>" else value))
        numbers.append(normalize_number(value) if column_type == "real" else None)
    return ParsedRow(
        values=values,
        value_tokens=tuple(token_rows),
        numbers=tuple(numbers),
    )


def _validated_top5(
    value: object,
    *,
    row_count: int,
) -> tuple[int | None, ...]:
    if not isinstance(value, tuple) or len(value) != TOP_K:
        raise WikiSQLUAOPolicyError("RAW top5 must be an exact five-slot tuple")
    seen: set[int] = set()
    null_seen = False
    for row in value:
        if row is None:
            null_seen = True
            continue
        if (
            null_seen
            or type(row) is not int
            or not 0 <= row < row_count
            or row in seen
        ):
            raise WikiSQLUAOPolicyError("RAW top5 is malformed")
        seen.add(row)
    if len(seen) != min(TOP_K, row_count):
        raise WikiSQLUAOPolicyError("RAW top5 does not fill its available slots")
    return value


@dataclass(frozen=True, slots=True)
class LabelFreeItem:
    question: str
    headers: tuple[str, ...]
    types: tuple[str, ...]
    serialized_rows: tuple[str, ...]
    raw_top5: tuple[int | None, ...]
    embeddings: PrecomputedEmbeddings | None = None

    def __post_init__(self) -> None:
        question = _text(self.question, field="question")
        if (
            not isinstance(self.headers, tuple)
            or not self.headers
            or any(not isinstance(row, str) for row in self.headers)
        ):
            raise WikiSQLUAOPolicyError("headers must be a nonempty text tuple")
        headers = tuple(
            _text(row, field=f"header[{index}]")
            for index, row in enumerate(self.headers)
        )
        if (
            not isinstance(self.types, tuple)
            or len(self.types) != len(headers)
            or any(row not in {"text", "real"} for row in self.types)
        ):
            raise WikiSQLUAOPolicyError("types must match headers")
        if (
            not isinstance(self.serialized_rows, tuple)
            or not self.serialized_rows
            or any(not isinstance(row, str) for row in self.serialized_rows)
        ):
            raise WikiSQLUAOPolicyError("serialized_rows must be nonempty")
        raw = _validated_top5(self.raw_top5, row_count=len(self.serialized_rows))
        parsed = tuple(
            _parse_serialized_row(row, headers, self.types)
            for row in self.serialized_rows
        )
        if self.embeddings is not None:
            if not isinstance(self.embeddings, PrecomputedEmbeddings):
                raise WikiSQLUAOPolicyError(
                    "embeddings must be PrecomputedEmbeddings or None"
                )
            if len(self.embeddings.rows) != len(parsed):
                raise WikiSQLUAOPolicyError(
                    "embedding rows do not match serialized rows"
                )
        object.__setattr__(self, "question", question)
        object.__setattr__(self, "headers", headers)
        object.__setattr__(self, "raw_top5", raw)

    @property
    def commitment_sha256(self) -> str:
        embedding_payload: object = None
        if self.embeddings is not None:
            embedding_payload = {
                "model_sha256": self.embeddings.model_sha256,
                "question": list(self.embeddings.question),
                "rows": [list(row) for row in self.embeddings.rows],
            }
        return canonical_sha256(
            {
                "schema": f"{VERSION}_label_free_item_v1",
                "question": self.question,
                "headers": list(self.headers),
                "types": list(self.types),
                "serialized_rows": list(self.serialized_rows),
                "raw_top5": list(self.raw_top5),
                "embeddings": embedding_payload,
            }
        )


@dataclass(frozen=True, slots=True)
class TrainingItem:
    item: LabelFreeItem
    gold_row_ids: tuple[int, ...]
    family: str
    fold_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.item, LabelFreeItem):
            raise WikiSQLUAOPolicyError("training item lacks LabelFreeItem")
        if (
            not isinstance(self.gold_row_ids, tuple)
            or not 1 <= len(self.gold_row_ids) <= TOP_K
            or any(
                type(row) is not int
                or not 0 <= row < len(self.item.serialized_rows)
                for row in self.gold_row_ids
            )
            or tuple(sorted(set(self.gold_row_ids))) != self.gold_row_ids
        ):
            raise WikiSQLUAOPolicyError(
                "TRAIN gold rows must be sorted, distinct, and one-through-five"
            )
        if self.family not in FAMILY_ORDER:
            raise WikiSQLUAOPolicyError("TRAIN family is outside the frozen registry")
        if type(self.fold_index) is not int or not 0 <= self.fold_index < CROSS_FOLDS:
            raise WikiSQLUAOPolicyError(
                "TRAIN fold_index must be the sealed integer 0..3"
            )


@dataclass(frozen=True, slots=True)
class _ItemFeatures:
    item: LabelFreeItem
    anchors: QuestionAnchors
    header_tokens: tuple[tuple[str, ...], ...]
    parsed_rows: tuple[ParsedRow, ...]
    raw_rank: tuple[int, ...]


def _prepare(item: LabelFreeItem) -> _ItemFeatures:
    header_tokens = tuple(text_tokens(row) for row in item.headers)
    parsed = tuple(
        _parse_serialized_row(row, item.headers, item.types)
        for row in item.serialized_rows
    )
    ranks = [TOP_K + 1] * len(parsed)
    for rank, row in enumerate(item.raw_top5):
        if row is not None:
            ranks[row] = rank
    return _ItemFeatures(
        item=item,
        anchors=extract_anchors(item.question),
        header_tokens=header_tokens,
        parsed_rows=parsed,
        raw_rank=tuple(ranks),
    )


def _set_overlap(left: Sequence[str], right: Sequence[str]) -> int:
    return len(set(left).intersection(right))


def _value_token_set(row: ParsedRow) -> set[str]:
    return {token for cell in row.value_tokens for token in cell}


def _numeric_relations(
    prepared: _ItemFeatures, row_index: int
) -> tuple[float, float, float]:
    """Return equality, comparator satisfaction, and violation indicators."""

    anchors = prepared.anchors
    row = prepared.parsed_rows[row_index]
    values = tuple(value for value in row.numbers if value is not None)
    if not anchors.numbers or not values:
        return 0.0, 0.0, 0.0
    equal = float(
        any(math.isclose(value, anchor, rel_tol=1.0e-9, abs_tol=1.0e-9)
            for value in values for anchor in anchors.numbers)
    )
    if anchors.comparator == "EQ":
        satisfaction = equal
    elif anchors.comparator == "GT":
        satisfaction = float(
            any(value > anchor for value in values for anchor in anchors.numbers)
        )
    else:
        satisfaction = float(
            any(value < anchor for value in values for anchor in anchors.numbers)
        )
    return equal, satisfaction, float(not satisfaction)


def _embedding_feature(prepared: _ItemFeatures, row_index: int) -> float:
    embeddings = prepared.item.embeddings
    if embeddings is None:
        return 0.0
    return _cosine(embeddings.question, embeddings.rows[row_index])


def _raw_prior(prepared: _ItemFeatures, row_index: int) -> float:
    rank = prepared.raw_rank[row_index]
    return 1.0 / (rank + 1.0) if rank <= TOP_K else 0.0


def _t02_features(prepared: _ItemFeatures, row_index: int) -> tuple[float, ...]:
    row = prepared.parsed_rows[row_index]
    values = _value_token_set(row)
    content = set(prepared.anchors.content_tokens)
    hits = len(values.intersection(content))
    equal, _, _ = _numeric_relations(prepared, row_index)
    return (
        float(hits),
        hits / max(1, len(content)),
        equal,
        float(
            any(
                set(cell) and set(cell).issubset(content)
                for cell in row.value_tokens
            )
        ),
        _embedding_feature(prepared, row_index),
        _raw_prior(prepared, row_index),
    )


def _t05_features(prepared: _ItemFeatures, row_index: int) -> tuple[float, ...]:
    row = prepared.parsed_rows[row_index]
    question_tokens = set(prepared.anchors.tokens)
    header_matches = tuple(
        bool(set(tokens).intersection(question_tokens))
        for tokens in prepared.header_tokens
    )
    anchored_value_hits = sum(
        _set_overlap(cell, prepared.anchors.content_tokens)
        for matched, cell in zip(header_matches, row.value_tokens, strict=True)
        if matched
    )
    equal, satisfied, violated = _numeric_relations(prepared, row_index)
    real_header_matches = sum(
        int(matched and column_type == "real")
        for matched, column_type in zip(
            header_matches, prepared.item.types, strict=True
        )
    )
    return (
        float(sum(header_matches)),
        float(anchored_value_hits),
        satisfied,
        violated,
        equal,
        float(real_header_matches) * satisfied,
        _embedding_feature(prepared, row_index),
    )


def _t08_features(prepared: _ItemFeatures, row_index: int) -> tuple[float, ...]:
    raw_rows = tuple(row for row in prepared.item.raw_top5 if row is not None)
    distance = min(abs(row_index - row) for row in raw_rows)
    selected = float(row_index in raw_rows)
    adjacent = float(distance == 1)
    values = _value_token_set(prepared.parsed_rows[row_index])
    anchor_hits = len(values.intersection(prepared.anchors.content_tokens))
    return (
        selected,
        adjacent,
        1.0 / (distance + 1.0),
        float(anchor_hits) * (1.0 + adjacent),
        float(
            sum(
                bool(set(tokens).intersection(prepared.anchors.tokens))
                for tokens in prepared.header_tokens
            )
        ),
        _embedding_feature(prepared, row_index),
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left.union(right)
    return len(left.intersection(right)) / len(union) if union else 1.0


def _t18_features(prepared: _ItemFeatures, row_index: int) -> tuple[float, ...]:
    row_tokens = _value_token_set(prepared.parsed_rows[row_index])
    question = set(prepared.anchors.content_tokens)
    strong_hits = len(row_tokens.intersection(question))
    unmatched = len(row_tokens - question) / max(1, len(row_tokens))
    _, satisfied, violated = _numeric_relations(prepared, row_index)
    raw_rows = tuple(row for row in prepared.item.raw_top5 if row is not None)
    duplicate = max(
        (
            _jaccard(
                row_tokens,
                _value_token_set(prepared.parsed_rows[other]),
            )
            for other in raw_rows
            if other != row_index
        ),
        default=0.0,
    )
    return (
        float(strong_hits),
        unmatched,
        violated,
        satisfied,
        duplicate,
        math.log1p(len(row_tokens)),
        _raw_prior(prepared, row_index),
    )


@dataclass(frozen=True, slots=True)
class ClaimRecipe:
    claim_id: str
    operator_template: str
    description: str
    feature_names: tuple[str, ...]
    extractor: Callable[[_ItemFeatures, int], tuple[float, ...]]

    def __post_init__(self) -> None:
        if self.operator_template not in {"T02", "T05", "T08", "T18"}:
            raise WikiSQLUAOPolicyError("claim operator template is invalid")
        if not self.claim_id or not self.feature_names:
            raise WikiSQLUAOPolicyError("claim recipe is incomplete")


CLAIM_RECIPES = (
    ClaimRecipe(
        claim_id="C_T02_SPARSE_TYPED_ANCHORS",
        operator_template="T02",
        description="sparse typed lexical/numeric anchor action",
        feature_names=(
            "anchor_hits",
            "anchor_coverage",
            "numeric_equal",
            "whole_cell_anchor",
            "dense_cosine",
            "raw_rank_prior",
        ),
        extractor=_t02_features,
    ),
    ClaimRecipe(
        claim_id="C_T05_LOW_ORDER_INTERACTION",
        operator_template="T05",
        description="header/value and comparator low-order interaction",
        feature_names=(
            "header_hits",
            "header_value_interaction",
            "comparator_satisfied",
            "comparator_violated",
            "numeric_equal",
            "typed_numeric_interaction",
            "dense_cosine",
        ),
        extractor=_t05_features,
    ),
    ClaimRecipe(
        claim_id="C_T08_LOCALITY_EXPANSION",
        operator_template="T08",
        description="physical-row locality around RAW anchors",
        feature_names=(
            "raw_selected",
            "adjacent_to_raw",
            "inverse_raw_distance",
            "anchor_locality_interaction",
            "question_header_coverage",
            "dense_cosine",
        ),
        extractor=_t08_features,
    ),
    ClaimRecipe(
        claim_id="C_T18_SPARSE_CONTAMINATION",
        operator_template="T18",
        description="sparse contamination and duplicate suppression",
        feature_names=(
            "strong_anchor_hits",
            "unmatched_value_ratio",
            "numeric_conflict",
            "numeric_satisfaction",
            "raw_duplicate_similarity",
            "row_token_complexity",
            "raw_rank_prior",
        ),
        extractor=_t18_features,
    ),
)
_CLAIM_BY_ID = {row.claim_id: row for row in CLAIM_RECIPES}


@dataclass(frozen=True, slots=True)
class LogisticModel:
    feature_names: tuple[str, ...]
    population_mean: tuple[float, ...]
    population_std: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        width = len(self.feature_names)
        if (
            width == 0
            or not isinstance(self.feature_names, tuple)
            or any(
                not isinstance(row, str) or not row
                for row in self.feature_names
            )
            or len(set(self.feature_names)) != width
            or len(self.population_mean) != width
            or len(self.population_std) != width
            or len(self.coefficients) != width
        ):
            raise WikiSQLUAOPolicyError("logistic model dimensions drifted")
        values = (
            *self.population_mean,
            *self.population_std,
            self.intercept,
            *self.coefficients,
        )
        if any(not math.isfinite(float(row)) for row in values):
            raise WikiSQLUAOPolicyError("logistic model is non-finite")
        if any(row < 0.0 for row in self.population_std):
            raise WikiSQLUAOPolicyError("logistic population std is negative")

    def linear_score(self, feature_values: Sequence[float]) -> float:
        if len(feature_values) != len(self.feature_names):
            raise WikiSQLUAOPolicyError("prediction feature width drifted")
        score = self.intercept
        for value, mean, std, coefficient in zip(
            feature_values,
            self.population_mean,
            self.population_std,
            self.coefficients,
            strict=True,
        ):
            standardized = 0.0 if std == 0.0 else (float(value) - mean) / std
            score += coefficient * standardized
        return score

    def probability(self, feature_values: Sequence[float]) -> float:
        score = max(-40.0, min(40.0, self.linear_score(feature_values)))
        return 1.0 / (1.0 + math.exp(-score))

    def private_payload(self) -> dict[str, object]:
        return {
            "schema": f"{VERSION}_logistic_model_v1",
            "feature_names": list(self.feature_names),
            "population_mean": list(self.population_mean),
            "population_std": list(self.population_std),
            "intercept": self.intercept,
            "coefficients": list(self.coefficients),
        }

    @property
    def model_sha256(self) -> str:
        return canonical_sha256(self.private_payload())


def _logistic_model_from_payload(value: Mapping[str, object]) -> LogisticModel:
    fields = frozenset(
        {
            "schema",
            "feature_names",
            "population_mean",
            "population_std",
            "intercept",
            "coefficients",
        }
    )
    if not isinstance(value, Mapping) or set(value) != fields:
        raise WikiSQLUAOPolicyError(
            "logistic model schema contains missing or extra fields"
        )
    if value["schema"] != f"{VERSION}_logistic_model_v1":
        raise WikiSQLUAOPolicyError("logistic model schema drifted")
    feature_values = _array(value["feature_names"], field="model feature_names")
    if any(not isinstance(row, str) for row in feature_values):
        raise WikiSQLUAOPolicyError("model feature name is malformed")
    return LogisticModel(
        feature_names=tuple(feature_values),  # type: ignore[arg-type]
        population_mean=tuple(
            _finite_numeric(row, field="model population mean")
            for row in _array(value["population_mean"], field="model population_mean")
        ),
        population_std=tuple(
            _finite_numeric(row, field="model population std")
            for row in _array(value["population_std"], field="model population std")
        ),
        intercept=_finite_numeric(value["intercept"], field="model intercept"),
        coefficients=tuple(
            _finite_numeric(row, field="model coefficient")
            for row in _array(value["coefficients"], field="model coefficients")
        ),
    )


def _finite_float(value: float) -> float:
    result = float(round(float(value), 15))
    return 0.0 if abs(result) < 5.0e-16 else result


def _fit_logistic(
    feature_names: tuple[str, ...],
    rows: Sequence[Sequence[float]],
    targets: Sequence[int],
) -> LogisticModel:
    if (
        not rows
        or len(rows) != len(targets)
        or any(len(row) != len(feature_names) for row in rows)
        or any(value not in {0, 1} for value in targets)
    ):
        raise WikiSQLUAOPolicyError("logistic TRAIN matrix is malformed")
    matrix = np.asarray(rows, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise WikiSQLUAOPolicyError("logistic TRAIN matrix is non-finite")
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    standardized = np.divide(
        matrix - mean,
        std,
        out=np.zeros_like(matrix),
        where=std != 0.0,
    )
    design = np.column_stack(
        (np.ones(standardized.shape[0], dtype=np.float64), standardized)
    )
    positive = max(1, int(target.sum()))
    negative = max(1, len(target) - int(target.sum()))
    base_weight = np.where(
        target == 1.0,
        len(target) / (2.0 * positive),
        len(target) / (2.0 * negative),
    )
    beta = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.diag(
        np.asarray([0.0] + [RIDGE_LAMBDA] * len(feature_names), dtype=np.float64)
    )
    for _ in range(IRLS_STEPS):
        linear = np.clip(design @ beta, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        variance = np.maximum(probability * (1.0 - probability), 1.0e-8)
        weighted_variance = base_weight * variance
        gradient = (
            design.T @ (base_weight * (target - probability))
            - penalty @ beta
        )
        hessian = (design.T * weighted_variance) @ design + penalty
        try:
            delta = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(hessian, rcond=1.0e-12) @ gradient
        beta = np.clip(beta + delta, -30.0, 30.0)
        if float(np.max(np.abs(delta))) < 1.0e-10:
            break
    return LogisticModel(
        feature_names=feature_names,
        population_mean=tuple(_finite_float(row) for row in mean),
        population_std=tuple(_finite_float(row) for row in std),
        intercept=_finite_float(beta[0]),
        coefficients=tuple(_finite_float(row) for row in beta[1:]),
    )


def _claim_matrix(
    examples: Sequence[TrainingItem],
    recipe: ClaimRecipe,
) -> tuple[list[tuple[float, ...]], list[int]]:
    rows: list[tuple[float, ...]] = []
    targets: list[int] = []
    for example in examples:
        prepared = _prepare(example.item)
        gold = set(example.gold_row_ids)
        for row_index in range(len(prepared.parsed_rows)):
            features = recipe.extractor(prepared, row_index)
            if len(features) != len(recipe.feature_names):
                raise WikiSQLUAOPolicyError("claim feature recipe width drifted")
            rows.append(features)
            targets.append(int(row_index in gold))
    return rows, targets


def _union_features(
    prepared: _ItemFeatures,
    row_index: int,
    selected_claim_ids: Sequence[str],
) -> tuple[float, ...]:
    result: list[float] = []
    for claim_id in selected_claim_ids:
        try:
            recipe = _CLAIM_BY_ID[claim_id]
        except KeyError as exc:
            raise WikiSQLUAOPolicyError("selected claim is unknown") from exc
        result.extend(recipe.extractor(prepared, row_index))
    return tuple(result)


def _union_feature_names(selected_claim_ids: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        f"{claim_id}:{name}"
        for claim_id in selected_claim_ids
        for name in _CLAIM_BY_ID[claim_id].feature_names
    )


def _probability_scores(
    model: LogisticModel,
    prepared: _ItemFeatures,
    feature_function: Callable[[_ItemFeatures, int], tuple[float, ...]],
) -> tuple[int, ...]:
    return tuple(
        int(
            round(
                model.probability(feature_function(prepared, row_index))
                * PROBABILITY_SCALE
            )
        )
        for row_index in range(len(prepared.parsed_rows))
    )


def _candidate_and_margin(
    raw_top5: tuple[int | None, ...],
    scores: tuple[int, ...],
) -> tuple[tuple[int | None, ...], int]:
    raw_rows = tuple(row for row in raw_top5 if row is not None)
    raw_rank = {row: rank for rank, row in enumerate(raw_rows)}
    ranked = sorted(
        range(len(scores)),
        key=lambda row: (-scores[row], raw_rank.get(row, TOP_K + 1), row),
    )
    candidate_rows = tuple(ranked[: min(TOP_K, len(ranked))])
    candidate = candidate_rows + (None,) * (TOP_K - len(candidate_rows))
    if set(candidate_rows) == set(raw_rows):
        return raw_top5, 0
    raw_expected = _learned_expected_utility(raw_rows, scores)
    candidate_expected = _learned_expected_utility(candidate_rows, scores)
    margin = int(
        (
            (candidate_expected - raw_expected) * Decimal(UTILITY_MARGIN_SCALE)
        ).to_integral_value(rounding=ROUND_HALF_EVEN)
    )
    # A top-k maximizer must not have negative expected margin.  Fail closed if
    # a future expected-utility refactor breaks that invariant.
    if margin < 0:
        raise WikiSQLUAOPolicyError("candidate expected-utility margin is negative")
    return candidate, margin


def _learned_action(
    raw_top5: tuple[int | None, ...],
    scores: tuple[int, ...],
    *,
    margin_threshold: int,
) -> tuple[int | None, ...]:
    if margin_threshold not in NO_OP_MARGIN_GRID:
        raise WikiSQLUAOPolicyError("no-op margin threshold is outside frozen grid")
    candidate, margin = _candidate_and_margin(raw_top5, scores)
    if candidate is raw_top5 or margin <= margin_threshold:
        return raw_top5
    return candidate


def _learned_expected_utility(
    selected_rows: Sequence[int],
    scores: tuple[int, ...],
) -> Decimal:
    """Independent-Bernoulli expectation of ``hits + complete``.

    Row probabilities are quantized before this calculation.  Because every
    TRAIN item has at least one gold row, the complete probability is
    conditioned on at least one positive row.  Decimal arithmetic makes the
    strict no-op comparison deterministic across process orderings.
    """

    scale = Decimal(PROBABILITY_SCALE)
    probabilities = tuple(Decimal(score) / scale for score in scores)
    selected = set(selected_rows)
    unconditional_expected_hits = sum(
        (probabilities[row] for row in selected),
        start=Decimal(0),
    )
    all_none = Decimal(1)
    outside_none = Decimal(1)
    for row_index, probability in enumerate(probabilities):
        absence = Decimal(1) - probability
        all_none *= absence
        if row_index not in selected:
            outside_none *= absence
    conditioned_mass = Decimal(1) - all_none
    if conditioned_mass == 0:
        return Decimal(0)
    # Both terms are conditioned on at least one relevant row.  The complete
    # event contributes P(no positives outside S AND at least one anywhere)
    # = outside_none - all_none.  Each selected-row hit already implies the
    # conditioning event, so its unconditional expectation belongs in the
    # same numerator before the single division.
    return (
        unconditional_expected_hits + outside_none - all_none
    ) / conditioned_mass


def _utility(
    selected: tuple[int | None, ...],
    gold: tuple[int, ...],
) -> int:
    selected_set = {row for row in selected if row is not None}
    gold_set = set(gold)
    hits = len(selected_set.intersection(gold_set))
    return hits + int(gold_set.issubset(selected_set))


@dataclass(frozen=True, slots=True)
class FoldProbeReceipt:
    fold_index: int
    heldout_count: int
    support_count: int
    counter_count: int
    neutral_count: int
    utility_delta: int
    prediction_commitments: tuple[str, ...]
    family_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if (
            type(self.fold_index) is not int
            or not 0 <= self.fold_index < CROSS_FOLDS
            or any(
                type(value) is not int or value < 0
                for value in (
                    self.heldout_count,
                    self.support_count,
                    self.counter_count,
                    self.neutral_count,
                )
            )
            or self.support_count + self.counter_count + self.neutral_count
            != self.heldout_count
            or len(self.prediction_commitments) != self.heldout_count
            or any(_SHA256_RE.fullmatch(row) is None for row in self.prediction_commitments)
            or tuple(name for name, _ in self.family_counts) != FAMILY_ORDER
            or any(
                type(value) is not int
                or value != A_FORM_QUOTA_PER_FOLD_FAMILY
                for _, value in self.family_counts
            )
            or sum(value for _, value in self.family_counts) != self.heldout_count
        ):
            raise WikiSQLUAOPolicyError("fold probe receipt is malformed")

    def payload(self) -> dict[str, object]:
        return {
            "schema": f"{VERSION}_fold_probe_receipt_v1",
            "fold_index": self.fold_index,
            "heldout_count": self.heldout_count,
            "support_count": self.support_count,
            "counter_count": self.counter_count,
            "neutral_count": self.neutral_count,
            "utility_delta": self.utility_delta,
            "prediction_commitments": list(self.prediction_commitments),
            "family_counts": {
                name: value for name, value in self.family_counts
            },
        }

    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class ClaimProbeReceipt:
    claim_id: str
    operator_template: str
    fold_receipts: tuple[FoldProbeReceipt, ...]
    support_count: int
    counter_count: int
    neutral_count: int
    utility_delta: int
    mdl_units: int
    selection_score: int
    prediction_vector: tuple[str, ...]
    family_utility_delta: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if self.claim_id not in _CLAIM_BY_ID:
            raise WikiSQLUAOPolicyError("probe claim is unknown")
        if self.operator_template != _CLAIM_BY_ID[self.claim_id].operator_template:
            raise WikiSQLUAOPolicyError("probe operator template drifted")
        if (
            len(self.fold_receipts) != CROSS_FOLDS
            or tuple(row.fold_index for row in self.fold_receipts)
            != tuple(range(CROSS_FOLDS))
        ):
            raise WikiSQLUAOPolicyError("claim lacks four ordered fold receipts")
        heldout = sum(row.heldout_count for row in self.fold_receipts)
        if (
            self.support_count != sum(row.support_count for row in self.fold_receipts)
            or self.counter_count != sum(row.counter_count for row in self.fold_receipts)
            or self.neutral_count != sum(row.neutral_count for row in self.fold_receipts)
            or self.utility_delta != sum(row.utility_delta for row in self.fold_receipts)
            or heldout != len(self.prediction_vector)
            or any(_SHA256_RE.fullmatch(row) is None for row in self.prediction_vector)
            or type(self.mdl_units) is not int
            or self.mdl_units <= 0
        ):
            raise WikiSQLUAOPolicyError("claim aggregate probe receipt drifted")
        expected_score = (
            10_000 * self.utility_delta
            + 100 * (self.support_count - self.counter_count)
            - self.mdl_units
        )
        if self.selection_score != expected_score:
            raise WikiSQLUAOPolicyError("claim selection score drifted")
        if tuple(name for name, _ in self.family_utility_delta) != FAMILY_ORDER:
            raise WikiSQLUAOPolicyError("claim family aggregates drifted")

    def payload(self) -> dict[str, object]:
        return {
            "schema": f"{VERSION}_claim_probe_receipt_v1",
            "claim_id": self.claim_id,
            "operator_template": self.operator_template,
            "fold_receipt_sha256": [
                row.receipt_sha256 for row in self.fold_receipts
            ],
            "support_count": self.support_count,
            "counter_count": self.counter_count,
            "neutral_count": self.neutral_count,
            "utility_delta": self.utility_delta,
            "mdl_units": self.mdl_units,
            "selection_score": self.selection_score,
            "prediction_vector": list(self.prediction_vector),
            "prediction_vector_semantics": (
                "canonical_oof_selected_top5_plus_raw_no_op_per_item"
            ),
            "prediction_distance_metric": "ordered_item_hamming_count",
            "family_utility_delta": {
                name: value for name, value in self.family_utility_delta
            },
            "train_only": True,
            "heldout_access_count": 0,
            "online_evaluation_count": 0,
        }

    def safe_receipt(self) -> dict[str, object]:
        return _content_addressed(self.payload())

    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class ClaimSelectionCandidate:
    claim_id: str
    is_first: bool
    base_score: int
    hamming_from_first: int
    redundant_action_count: int
    redundancy_penalty: int
    adjusted_score: int
    eligible: bool

    def __post_init__(self) -> None:
        common_invalid = (
            self.claim_id not in _CLAIM_BY_ID
            or type(self.is_first) is not bool
            or type(self.base_score) is not int
            or type(self.hamming_from_first) is not int
            or not 0 <= self.hamming_from_first <= ACTION_VECTOR_LENGTH
        )
        if self.is_first:
            specific_invalid = (
                self.hamming_from_first != 0
                or self.redundant_action_count != 0
                or self.redundancy_penalty != 0
                or self.adjusted_score != self.base_score
                or self.eligible is not True
            )
        else:
            specific_invalid = (
                self.redundant_action_count
                != ACTION_VECTOR_LENGTH - self.hamming_from_first
                or self.redundancy_penalty
                != (
                    ACTION_REDUNDANCY_PENALTY_PER_MATCH
                    * self.redundant_action_count
                )
                or self.adjusted_score
                != self.base_score - self.redundancy_penalty
                or self.eligible
                != (self.hamming_from_first >= ACTION_VECTOR_MIN_HAMMING)
            )
        if common_invalid or specific_invalid:
            raise WikiSQLUAOPolicyError("claim selection candidate drifted")

    def payload(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "is_first": self.is_first,
            "base_score": self.base_score,
            "hamming_from_first": self.hamming_from_first,
            "redundant_action_count": self.redundant_action_count,
            "redundancy_penalty": self.redundancy_penalty,
            "adjusted_score": self.adjusted_score,
            "eligible": self.eligible,
        }


@dataclass(frozen=True, slots=True)
class ClaimSelectionReceipt:
    first_claim_id: str
    selected_claim_ids: tuple[str, ...]
    candidates: tuple[ClaimSelectionCandidate, ...]

    def __post_init__(self) -> None:
        if (
            self.first_claim_id not in _CLAIM_BY_ID
            or len(self.selected_claim_ids) != SELECTED_CLAIM_COUNT
            or self.selected_claim_ids[0] != self.first_claim_id
            or len(set(self.selected_claim_ids)) != SELECTED_CLAIM_COUNT
            or tuple(row.claim_id for row in self.candidates)
            != tuple(sorted(_CLAIM_BY_ID))
        ):
            raise WikiSQLUAOPolicyError("claim selection receipt is malformed")
        first = next(
            row for row in self.candidates
            if row.claim_id == self.first_claim_id
        )
        if (
            first.hamming_from_first != 0
            or first.redundant_action_count != 0
            or not first.is_first
            or not first.eligible
            or sum(row.is_first for row in self.candidates) != 1
        ):
            raise WikiSQLUAOPolicyError("first claim selection record drifted")
        eligible_seconds = tuple(
            row
            for row in self.candidates
            if row.claim_id != self.first_claim_id and row.eligible
        )
        if not eligible_seconds:
            raise WikiSQLUAOPolicyError(
                "no claim meets frozen action Hamming distance"
            )
        expected_second = sorted(
            eligible_seconds,
            key=lambda row: (
                -row.adjusted_score,
                -row.hamming_from_first,
                -row.base_score,
                row.claim_id,
            ),
        )[0]
        if self.selected_claim_ids[1] != expected_second.claim_id:
            raise WikiSQLUAOPolicyError("redundancy-adjusted claim choice drifted")

    def payload(self) -> dict[str, object]:
        return {
            "schema": f"{VERSION}_claim_selection_receipt_v1",
            "first_claim_id": self.first_claim_id,
            "selected_claim_ids": list(self.selected_claim_ids),
            "prediction_vector_semantics": (
                "canonical_oof_selected_top5_plus_raw_no_op_per_item"
            ),
            "distance_metric": "ordered_item_hamming_count",
            "action_vector_length": ACTION_VECTOR_LENGTH,
            "minimum_hamming_count": ACTION_VECTOR_MIN_HAMMING,
            "redundancy_penalty_per_matching_action": (
                ACTION_REDUNDANCY_PENALTY_PER_MATCH
            ),
            "first_selection_rule": "max_base_score_then_lexical_claim_id",
            "second_selection_rule": (
                "eligible_hamming_then_max_redundancy_adjusted_score_"
                "then_higher_hamming_then_base_score_then_lexical_claim_id"
            ),
            "candidates": [row.payload() for row in self.candidates],
            "train_only": True,
            "heldout_access_count": 0,
            "online_evaluation_count": 0,
        }

    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def safe_receipt(self) -> dict[str, object]:
        return _content_addressed(self.payload())


@dataclass(frozen=True, slots=True)
class ThresholdEvaluation:
    margin_threshold: int
    raw_total_utility: int
    total_true_utility: int
    net_utility: int
    action_count: int
    support_count: int
    counter_count: int
    neutral_count: int

    def __post_init__(self) -> None:
        if (
            type(self.margin_threshold) is not int
            or self.margin_threshold not in NO_OP_MARGIN_GRID
            or type(self.net_utility) is not int
            or type(self.raw_total_utility) is not int
            or self.raw_total_utility < 0
            or type(self.total_true_utility) is not int
            or self.total_true_utility < 0
            or self.net_utility
            != self.total_true_utility - self.raw_total_utility
            or any(
                type(value) is not int or value < 0
                for value in (
                    self.action_count,
                    self.support_count,
                    self.counter_count,
                    self.neutral_count,
                )
            )
            or self.support_count + self.counter_count + self.neutral_count
            != A_FORM_QUOTA_PER_FAMILY * len(FAMILY_ORDER)
            or self.action_count
            > A_FORM_QUOTA_PER_FAMILY * len(FAMILY_ORDER)
        ):
            raise WikiSQLUAOPolicyError("threshold evaluation is malformed")

    def payload(self) -> dict[str, object]:
        return {
            "margin_threshold": self.margin_threshold,
            "raw_total_utility": self.raw_total_utility,
            "total_true_utility": self.total_true_utility,
            "net_utility": self.net_utility,
            "action_count": self.action_count,
            "support_count": self.support_count,
            "counter_count": self.counter_count,
            "neutral_count": self.neutral_count,
        }


@dataclass(frozen=True, slots=True)
class NoOpCalibrationReceipt:
    selected_claim_ids: tuple[str, ...]
    threshold_evaluations: tuple[ThresholdEvaluation, ...]
    selected_margin_threshold: int
    fold_model_sha256: tuple[str, ...]
    fold_prediction_commitments: tuple[tuple[str, ...], ...]
    fold_family_counts: tuple[tuple[tuple[str, int], ...], ...]
    family_utility_delta: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if (
            len(self.selected_claim_ids) != SELECTED_CLAIM_COUNT
            or len(set(self.selected_claim_ids)) != SELECTED_CLAIM_COUNT
            or any(row not in _CLAIM_BY_ID for row in self.selected_claim_ids)
            or tuple(row.margin_threshold for row in self.threshold_evaluations)
            != NO_OP_MARGIN_GRID
            or len(self.fold_model_sha256) != CROSS_FOLDS
            or any(
                not isinstance(row, str) or _SHA256_RE.fullmatch(row) is None
                for row in self.fold_model_sha256
            )
            or len(self.fold_prediction_commitments) != CROSS_FOLDS
            or any(
                len(rows)
                != (
                    A_FORM_QUOTA_PER_FAMILY
                    * len(FAMILY_ORDER)
                    // CROSS_FOLDS
                )
                or any(
                    not isinstance(row, str) or _SHA256_RE.fullmatch(row) is None
                    for row in rows
                )
                for rows in self.fold_prediction_commitments
            )
            or len(self.fold_family_counts) != CROSS_FOLDS
            or any(
                tuple(name for name, _ in rows) != FAMILY_ORDER
                or any(
                    type(value) is not int
                    or value != A_FORM_QUOTA_PER_FAMILY // CROSS_FOLDS
                    for _, value in rows
                )
                for rows in self.fold_family_counts
            )
            or tuple(name for name, _ in self.family_utility_delta) != FAMILY_ORDER
            or any(
                type(value) is not int
                for _, value in self.family_utility_delta
            )
        ):
            raise WikiSQLUAOPolicyError("no-op calibration receipt is malformed")
        selected = max(
            self.threshold_evaluations,
            key=lambda row: (
                row.net_utility,
                row.margin_threshold,
                -row.action_count,
            ),
        )
        if self.selected_margin_threshold != selected.margin_threshold:
            raise WikiSQLUAOPolicyError("no-op threshold selection drifted")

    def payload(self) -> dict[str, object]:
        return {
            "schema": f"{VERSION}_no_op_calibration_receipt_v1",
            "selected_claim_ids": list(self.selected_claim_ids),
            "margin_scale": UTILITY_MARGIN_SCALE,
            "fixed_margin_grid": list(NO_OP_MARGIN_GRID),
            "threshold_evaluations": [
                row.payload() for row in self.threshold_evaluations
            ],
            "selection_rule": (
                "max_net_true_utility_then_higher_threshold_then_fewer_actions"
            ),
            "selected_margin_threshold": self.selected_margin_threshold,
            "fold_count": CROSS_FOLDS,
            "fold_model_sha256": list(self.fold_model_sha256),
            "fold_prediction_commitments": [
                list(rows) for rows in self.fold_prediction_commitments
            ],
            "fold_family_counts": [
                {name: value for name, value in rows}
                for rows in self.fold_family_counts
            ],
            "fold_assignment_source": (
                "sealed_training_item_fold_index_hmac_16_per_family_per_fold"
            ),
            "family_utility_delta": {
                name: value for name, value in self.family_utility_delta
            },
            "train_item_count": A_FORM_QUOTA_PER_FAMILY * len(FAMILY_ORDER),
            "train_only": True,
            "heldout_access_count": 0,
            "online_evaluation_count": 0,
        }

    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def safe_receipt(self) -> dict[str, object]:
        return _content_addressed(self.payload())


@dataclass(frozen=True, slots=True)
class CompiledPolicy:
    selected_claim_ids: tuple[str, ...]
    model: LogisticModel
    probe_receipt_sha256: tuple[str, ...]
    claim_selection_receipt_sha256: str
    no_op_calibration_receipt_sha256: str
    margin_threshold: int
    train_item_count: int

    def __post_init__(self) -> None:
        if (
            len(self.selected_claim_ids) != SELECTED_CLAIM_COUNT
            or len(set(self.selected_claim_ids)) != SELECTED_CLAIM_COUNT
            or any(row not in _CLAIM_BY_ID for row in self.selected_claim_ids)
        ):
            raise WikiSQLUAOPolicyError("compiled policy does not contain K=2 claims")
        if self.model.feature_names != _union_feature_names(self.selected_claim_ids):
            raise WikiSQLUAOPolicyError("compiled feature union drifted")
        if (
            len(self.probe_receipt_sha256) != len(CLAIM_RECIPES)
            or any(
                not isinstance(row, str) or _SHA256_RE.fullmatch(row) is None
                for row in self.probe_receipt_sha256
            )
            or not isinstance(self.claim_selection_receipt_sha256, str)
            or _SHA256_RE.fullmatch(
                self.claim_selection_receipt_sha256
            ) is None
            or not isinstance(
                self.no_op_calibration_receipt_sha256, str
            )
            or _SHA256_RE.fullmatch(
                self.no_op_calibration_receipt_sha256
            ) is None
            or type(self.margin_threshold) is not int
            or self.margin_threshold not in NO_OP_MARGIN_GRID
            or type(self.train_item_count) is not int
            or self.train_item_count
            != A_FORM_QUOTA_PER_FAMILY * len(FAMILY_ORDER)
        ):
            raise WikiSQLUAOPolicyError("compiled policy lineage is malformed")

    def private_payload(self) -> dict[str, object]:
        return {
            "schema": f"{VERSION}_compiled_policy_private_v1",
            "selected_claim_ids": list(self.selected_claim_ids),
            "model": self.model.private_payload(),
            "probe_receipt_sha256": list(self.probe_receipt_sha256),
            "claim_selection_receipt_sha256": (
                self.claim_selection_receipt_sha256
            ),
            "action_vector_length": ACTION_VECTOR_LENGTH,
            "action_vector_min_hamming": ACTION_VECTOR_MIN_HAMMING,
            "action_redundancy_penalty_per_match": (
                ACTION_REDUNDANCY_PENALTY_PER_MATCH
            ),
            "no_op_calibration_receipt_sha256": (
                self.no_op_calibration_receipt_sha256
            ),
            "margin_threshold": self.margin_threshold,
            "fixed_margin_grid": list(NO_OP_MARGIN_GRID),
            "margin_scale": UTILITY_MARGIN_SCALE,
            "train_item_count": self.train_item_count,
            "a_form_quota_per_family": A_FORM_QUOTA_PER_FAMILY,
            "a_form_quota_per_fold_family": A_FORM_QUOTA_PER_FOLD_FAMILY,
            "a_hold_quota_per_family": A_HOLD_QUOTA_PER_FAMILY,
            "family_order": list(FAMILY_ORDER),
            "fold_count": CROSS_FOLDS,
            "fold_assignment_source": (
                "sealed_training_item_fold_index_hmac_16_per_family_per_fold"
            ),
            "top_k": TOP_K,
            "no_op_rule": "raw_byte_exact_unless_margin_strictly_greater_than_train_oof_threshold",
        }

    @property
    def policy_sha256(self) -> str:
        return canonical_sha256(self.private_payload())

    def content_addressed_private_payload(self) -> dict[str, object]:
        """Transport payload for a process-isolated A_hold worker."""

        return _content_addressed(self.private_payload())

    def safe_receipt(self) -> dict[str, object]:
        return _content_addressed(
            {
                "schema": f"{VERSION}_compiled_policy_safe_receipt_v1",
                "policy_sha256": self.policy_sha256,
                "selected_claim_ids": list(self.selected_claim_ids),
                "selected_operator_templates": [
                    _CLAIM_BY_ID[row].operator_template
                    for row in self.selected_claim_ids
                ],
                "feature_union": list(self.model.feature_names),
                "model_sha256": self.model.model_sha256,
                "probe_receipt_sha256": list(self.probe_receipt_sha256),
                "claim_selection_receipt_sha256": (
                    self.claim_selection_receipt_sha256
                ),
                "action_vector_length": ACTION_VECTOR_LENGTH,
                "action_vector_min_hamming": ACTION_VECTOR_MIN_HAMMING,
                "action_redundancy_penalty_per_match": (
                    ACTION_REDUNDANCY_PENALTY_PER_MATCH
                ),
                "no_op_calibration_receipt_sha256": (
                    self.no_op_calibration_receipt_sha256
                ),
                "margin_threshold": self.margin_threshold,
                "fixed_margin_grid": list(NO_OP_MARGIN_GRID),
                "margin_scale": UTILITY_MARGIN_SCALE,
                "train_item_count": self.train_item_count,
                "a_form_quota_per_family": A_FORM_QUOTA_PER_FAMILY,
                "a_form_quota_per_fold_family": A_FORM_QUOTA_PER_FOLD_FAMILY,
                "a_hold_quota_per_family": A_HOLD_QUOTA_PER_FAMILY,
                "family_order": list(FAMILY_ORDER),
                "fold_count": CROSS_FOLDS,
                "fold_assignment_source": (
                    "sealed_training_item_fold_index_hmac_16_per_family_per_fold"
                ),
                "train_only": True,
                "heldout_access_count": 0,
                "online_evaluation_count": 0,
            }
        )


def compiled_policy_from_private_payload(
    value: Mapping[str, object],
) -> CompiledPolicy:
    """Strictly reconstruct a content-addressed policy without pickle.

    The exact frozen constants are repeated in the private payload so a held-
    out process cannot silently reinterpret a TRAIN artifact under a different
    family, quota, top-k, or abstention contract.
    """

    fields = frozenset(
        {
            "schema",
            "selected_claim_ids",
            "model",
            "probe_receipt_sha256",
            "claim_selection_receipt_sha256",
            "action_vector_length",
            "action_vector_min_hamming",
            "action_redundancy_penalty_per_match",
            "no_op_calibration_receipt_sha256",
            "margin_threshold",
            "fixed_margin_grid",
            "margin_scale",
            "train_item_count",
            "a_form_quota_per_family",
            "a_form_quota_per_fold_family",
            "a_hold_quota_per_family",
            "family_order",
            "fold_count",
            "fold_assignment_source",
            "top_k",
            "no_op_rule",
        }
    )
    payload = _verify_content_addressed_payload(
        value,
        exact_fields=fields,
        field="compiled policy",
    )
    expected_scalars = {
        "schema": f"{VERSION}_compiled_policy_private_v1",
        "train_item_count": A_FORM_QUOTA_PER_FAMILY * len(FAMILY_ORDER),
        "a_form_quota_per_family": A_FORM_QUOTA_PER_FAMILY,
        "a_form_quota_per_fold_family": A_FORM_QUOTA_PER_FOLD_FAMILY,
        "a_hold_quota_per_family": A_HOLD_QUOTA_PER_FAMILY,
        "family_order": list(FAMILY_ORDER),
        "action_vector_length": ACTION_VECTOR_LENGTH,
        "action_vector_min_hamming": ACTION_VECTOR_MIN_HAMMING,
        "action_redundancy_penalty_per_match": (
            ACTION_REDUNDANCY_PENALTY_PER_MATCH
        ),
        "fold_count": CROSS_FOLDS,
        "fold_assignment_source": (
            "sealed_training_item_fold_index_hmac_16_per_family_per_fold"
        ),
        "fixed_margin_grid": list(NO_OP_MARGIN_GRID),
        "margin_scale": UTILITY_MARGIN_SCALE,
        "top_k": TOP_K,
        "no_op_rule": (
            "raw_byte_exact_unless_margin_strictly_greater_than_train_oof_threshold"
        ),
    }
    for field, expected in expected_scalars.items():
        if payload[field] != expected:
            raise WikiSQLUAOPolicyError(f"compiled policy {field} drifted")
    selected_values = _array(
        payload["selected_claim_ids"], field="selected claim IDs"
    )
    probe_values = _array(
        payload["probe_receipt_sha256"], field="probe receipt commitments"
    )
    if any(not isinstance(row, str) for row in selected_values):
        raise WikiSQLUAOPolicyError("selected claim ID is malformed")
    if any(
        not isinstance(row, str) or _SHA256_RE.fullmatch(row) is None
        for row in probe_values
    ):
        raise WikiSQLUAOPolicyError("probe receipt commitment is malformed")
    selection_hash = payload["claim_selection_receipt_sha256"]
    if (
        not isinstance(selection_hash, str)
        or _SHA256_RE.fullmatch(selection_hash) is None
    ):
        raise WikiSQLUAOPolicyError(
            "claim selection receipt commitment is malformed"
        )
    calibration_hash = payload["no_op_calibration_receipt_sha256"]
    if (
        not isinstance(calibration_hash, str)
        or _SHA256_RE.fullmatch(calibration_hash) is None
    ):
        raise WikiSQLUAOPolicyError(
            "no-op calibration receipt commitment is malformed"
        )
    margin_threshold = payload["margin_threshold"]
    if type(margin_threshold) is not int or margin_threshold not in NO_OP_MARGIN_GRID:
        raise WikiSQLUAOPolicyError("compiled policy margin threshold drifted")
    if not isinstance(payload["model"], Mapping):
        raise WikiSQLUAOPolicyError("compiled policy model is not an object")
    result = CompiledPolicy(
        selected_claim_ids=tuple(selected_values),  # type: ignore[arg-type]
        model=_logistic_model_from_payload(payload["model"]),
        probe_receipt_sha256=tuple(probe_values),  # type: ignore[arg-type]
        claim_selection_receipt_sha256=selection_hash,
        no_op_calibration_receipt_sha256=calibration_hash,
        margin_threshold=margin_threshold,
        train_item_count=payload["train_item_count"],  # type: ignore[arg-type]
    )
    if result.policy_sha256 != value["self_sha256"]:
        raise WikiSQLUAOPolicyError("compiled policy reconstruction drifted")
    return result


@dataclass(frozen=True, slots=True)
class PolicyFormation:
    policy: CompiledPolicy
    probe_receipts: tuple[ClaimProbeReceipt, ...]
    claim_selection_receipt: ClaimSelectionReceipt
    no_op_calibration_receipt: NoOpCalibrationReceipt

    def __post_init__(self) -> None:
        if (
            tuple(row.claim_id for row in self.probe_receipts)
            != tuple(row.claim_id for row in CLAIM_RECIPES)
            or tuple(row.receipt_sha256 for row in self.probe_receipts)
            != self.policy.probe_receipt_sha256
            or not isinstance(
                self.claim_selection_receipt, ClaimSelectionReceipt
            )
            or self.claim_selection_receipt.receipt_sha256
            != self.policy.claim_selection_receipt_sha256
            or self.claim_selection_receipt.selected_claim_ids
            != self.policy.selected_claim_ids
            or not isinstance(
                self.no_op_calibration_receipt, NoOpCalibrationReceipt
            )
            or self.no_op_calibration_receipt.receipt_sha256
            != self.policy.no_op_calibration_receipt_sha256
            or self.no_op_calibration_receipt.selected_margin_threshold
            != self.policy.margin_threshold
        ):
            raise WikiSQLUAOPolicyError("formation lineage drifted")

    def safe_receipt(self) -> dict[str, object]:
        return _content_addressed(
            {
                "schema": f"{VERSION}_formation_safe_receipt_v1",
                "policy_receipt_sha256": canonical_sha256(
                    {
                        key: value
                        for key, value in self.policy.safe_receipt().items()
                        if key != "self_sha256"
                    }
                ),
                "probe_receipt_sha256": [
                    row.receipt_sha256 for row in self.probe_receipts
                ],
                "claim_selection_receipt_sha256": (
                    self.claim_selection_receipt.receipt_sha256
                ),
                "action_vector_length": ACTION_VECTOR_LENGTH,
                "action_vector_min_hamming": ACTION_VECTOR_MIN_HAMMING,
                "action_redundancy_penalty_per_match": (
                    ACTION_REDUNDANCY_PENALTY_PER_MATCH
                ),
                "no_op_calibration_receipt_sha256": (
                    self.no_op_calibration_receipt.receipt_sha256
                ),
                "margin_threshold": self.policy.margin_threshold,
                "fixed_margin_grid": list(NO_OP_MARGIN_GRID),
                "selected_claim_ids": list(self.policy.selected_claim_ids),
                "claim_count": len(self.probe_receipts),
                "cross_folds": CROSS_FOLDS,
                "selected_claim_count": SELECTED_CLAIM_COUNT,
                "a_form_quota_per_family": A_FORM_QUOTA_PER_FAMILY,
                "a_form_quota_per_fold_family": A_FORM_QUOTA_PER_FOLD_FAMILY,
                "a_hold_quota_per_family": A_HOLD_QUOTA_PER_FAMILY,
                "family_order": list(FAMILY_ORDER),
                "fold_assignment_source": (
                    "sealed_training_item_fold_index_hmac_16_per_family_per_fold"
                ),
                "train_only": True,
                "heldout_access_count": 0,
                "online_evaluation_count": 0,
            }
        )


def _canonical_training_items(
    train_items: Sequence[TrainingItem],
) -> tuple[TrainingItem, ...]:
    if (
        isinstance(train_items, (str, bytes, bytearray))
        or not isinstance(train_items, Sequence)
        or len(train_items) < CROSS_FOLDS
        or any(not isinstance(row, TrainingItem) for row in train_items)
    ):
        raise WikiSQLUAOPolicyError("at least four TRAIN items are required")
    ordered = tuple(
        sorted(
            train_items,
            key=lambda row: (row.item.commitment_sha256, row.family),
        )
    )
    commitments = tuple(row.item.commitment_sha256 for row in ordered)
    if len(set(commitments)) != len(commitments):
        raise WikiSQLUAOPolicyError("TRAIN repeats a label-free item")
    family_counts = Counter(row.family for row in ordered)
    if any(
        family_counts[family] != A_FORM_QUOTA_PER_FAMILY
        for family in FAMILY_ORDER
    ):
        raise WikiSQLUAOPolicyError(
            "TRAIN requires the frozen 64-item quota in every EQ/GT/LT family"
        )
    fold_family_counts = Counter(
        (row.fold_index, row.family) for row in ordered
    )
    expected_per_fold_family = A_FORM_QUOTA_PER_FAMILY // CROSS_FOLDS
    if any(
        fold_family_counts[(fold_index, family)] != expected_per_fold_family
        for fold_index in range(CROSS_FOLDS)
        for family in FAMILY_ORDER
    ):
        raise WikiSQLUAOPolicyError(
            "sealed TRAIN folds require exactly 16 items per family per fold"
        )
    return ordered


def _fit_claim(
    train_items: tuple[TrainingItem, ...],
    recipe: ClaimRecipe,
) -> ClaimProbeReceipt:
    fold_receipts: list[FoldProbeReceipt] = []
    prediction_by_commitment: dict[str, str] = {}
    family_delta: Counter[str] = Counter()
    for fold_index in range(CROSS_FOLDS):
        heldout = tuple(
            row
            for row in train_items
            if row.fold_index == fold_index
        )
        formation = tuple(
            row
            for row in train_items
            if row.fold_index != fold_index
        )
        matrix, targets = _claim_matrix(formation, recipe)
        model = _fit_logistic(recipe.feature_names, matrix, targets)
        support = counter = neutral = utility_delta = 0
        commitments: list[str] = []
        for example in heldout:
            prepared = _prepare(example.item)
            scores = _probability_scores(model, prepared, recipe.extractor)
            action = _learned_action(
                example.item.raw_top5,
                scores,
                margin_threshold=0,
            )
            delta = _utility(action, example.gold_row_ids) - _utility(
                example.item.raw_top5, example.gold_row_ids
            )
            support += int(delta > 0)
            counter += int(delta < 0)
            neutral += int(delta == 0)
            utility_delta += delta
            family_delta[example.family] += delta
            prediction = canonical_sha256(
                {
                    "schema": f"{VERSION}_crossfit_action_prediction_v1",
                    "fold_index": example.fold_index,
                    "item_sha256": example.item.commitment_sha256,
                    "selected_top5": list(action),
                    "raw_no_op": action == example.item.raw_top5,
                }
            )
            commitments.append(prediction)
            prediction_by_commitment[example.item.commitment_sha256] = prediction
        fold_receipts.append(
            FoldProbeReceipt(
                fold_index=fold_index,
                heldout_count=len(heldout),
                support_count=support,
                counter_count=counter,
                neutral_count=neutral,
                utility_delta=utility_delta,
                prediction_commitments=tuple(commitments),
                family_counts=tuple(
                    (
                        family,
                        sum(example.family == family for example in heldout),
                    )
                    for family in FAMILY_ORDER
                ),
            )
        )
    support_total = sum(row.support_count for row in fold_receipts)
    counter_total = sum(row.counter_count for row in fold_receipts)
    neutral_total = sum(row.neutral_count for row in fold_receipts)
    delta_total = sum(row.utility_delta for row in fold_receipts)
    # Fixed two-part code: recipe identifier/feature presence plus one unit per
    # fitted coefficient.  It is deliberately outcome-independent.
    mdl_units = 8 + 4 * len(recipe.feature_names) + len(recipe.feature_names) + 1
    score = (
        10_000 * delta_total
        + 100 * (support_total - counter_total)
        - mdl_units
    )
    vector = tuple(
        prediction_by_commitment[row.item.commitment_sha256]
        for row in train_items
    )
    return ClaimProbeReceipt(
        claim_id=recipe.claim_id,
        operator_template=recipe.operator_template,
        fold_receipts=tuple(fold_receipts),
        support_count=support_total,
        counter_count=counter_total,
        neutral_count=neutral_total,
        utility_delta=delta_total,
        mdl_units=mdl_units,
        selection_score=score,
        prediction_vector=vector,
        family_utility_delta=tuple(
            (family, family_delta[family]) for family in FAMILY_ORDER
        ),
    )


def _select_claims(
    receipts: Sequence[ClaimProbeReceipt],
) -> tuple[tuple[str, ...], ClaimSelectionReceipt]:
    if (
        len(receipts) != len(CLAIM_RECIPES)
        or {row.claim_id for row in receipts} != set(_CLAIM_BY_ID)
        or any(len(row.prediction_vector) != ACTION_VECTOR_LENGTH for row in receipts)
    ):
        raise WikiSQLUAOPolicyError("claim selection inputs are malformed")
    ordered = sorted(
        receipts,
        key=lambda row: (-row.selection_score, row.claim_id),
    )
    first = ordered[0]
    candidates: list[ClaimSelectionCandidate] = []
    for receipt in sorted(receipts, key=lambda row: row.claim_id):
        is_first = receipt.claim_id == first.claim_id
        distance = (
            0
            if is_first
            else sum(
                left != right
                for left, right in zip(
                    first.prediction_vector,
                    receipt.prediction_vector,
                    strict=True,
                )
            )
        )
        redundant = 0 if is_first else ACTION_VECTOR_LENGTH - distance
        penalty = ACTION_REDUNDANCY_PENALTY_PER_MATCH * redundant
        candidates.append(
            ClaimSelectionCandidate(
                claim_id=receipt.claim_id,
                is_first=is_first,
                base_score=receipt.selection_score,
                hamming_from_first=distance,
                redundant_action_count=redundant,
                redundancy_penalty=penalty,
                adjusted_score=receipt.selection_score - penalty,
                eligible=(
                    True
                    if is_first
                    else distance >= ACTION_VECTOR_MIN_HAMMING
                ),
            )
        )
    eligible_seconds = tuple(
        row
        for row in candidates
        if row.claim_id != first.claim_id and row.eligible
    )
    if not eligible_seconds:
        raise WikiSQLUAOPolicyError(
            "TRAIN does not support the frozen action-vector Hamming distance"
        )
    second = sorted(
        eligible_seconds,
        key=lambda row: (
            -row.adjusted_score,
            -row.hamming_from_first,
            -row.base_score,
            row.claim_id,
        ),
    )[0]
    selected = (first.claim_id, second.claim_id)
    selection_receipt = ClaimSelectionReceipt(
        first_claim_id=first.claim_id,
        selected_claim_ids=selected,
        candidates=tuple(candidates),
    )
    return selected, selection_receipt


def _fit_union_model(
    examples: Sequence[TrainingItem],
    selected_claim_ids: tuple[str, ...],
) -> LogisticModel:
    feature_names = _union_feature_names(selected_claim_ids)
    matrix: list[tuple[float, ...]] = []
    targets: list[int] = []
    for example in examples:
        prepared = _prepare(example.item)
        gold = set(example.gold_row_ids)
        for row_index in range(len(prepared.parsed_rows)):
            matrix.append(
                _union_features(prepared, row_index, selected_claim_ids)
            )
            targets.append(int(row_index in gold))
    return _fit_logistic(feature_names, matrix, targets)


@dataclass(frozen=True, slots=True)
class _CalibrationObservation:
    fold_index: int
    item_commitment_sha256: str
    family: str
    raw_top5: tuple[int | None, ...]
    candidate_top5: tuple[int | None, ...]
    quantized_margin: int
    raw_true_utility: int
    candidate_true_utility_delta: int
    prediction_commitment_sha256: str


def _calibrate_no_op_threshold(
    train_items: tuple[TrainingItem, ...],
    selected_claim_ids: tuple[str, ...],
) -> NoOpCalibrationReceipt:
    """Calibrate abstention once from union-policy A_form OOF predictions."""

    observations: list[_CalibrationObservation] = []
    fold_models: list[str] = []
    fold_predictions: list[tuple[str, ...]] = []
    for fold_index in range(CROSS_FOLDS):
        heldout = tuple(
            row
            for row in train_items
            if row.fold_index == fold_index
        )
        formation = tuple(
            row
            for row in train_items
            if row.fold_index != fold_index
        )
        model = _fit_union_model(formation, selected_claim_ids)
        fold_models.append(model.model_sha256)
        prediction_rows: list[str] = []
        for example in heldout:
            prepared = _prepare(example.item)
            scores = _probability_scores(
                model,
                prepared,
                lambda value, row_index: _union_features(
                    value, row_index, selected_claim_ids
                ),
            )
            candidate, margin = _candidate_and_margin(
                example.item.raw_top5,
                scores,
            )
            delta = _utility(candidate, example.gold_row_ids) - _utility(
                example.item.raw_top5,
                example.gold_row_ids,
            )
            raw_true_utility = _utility(
                example.item.raw_top5,
                example.gold_row_ids,
            )
            prediction_commitment = canonical_sha256(
                {
                    "schema": f"{VERSION}_union_oof_prediction_v1",
                    "fold_index": fold_index,
                    "item_sha256": example.item.commitment_sha256,
                    "quantized_row_probabilities": list(scores),
                    "candidate_top5": list(candidate),
                    "quantized_margin": margin,
                }
            )
            prediction_rows.append(prediction_commitment)
            observations.append(
                _CalibrationObservation(
                    fold_index=fold_index,
                    item_commitment_sha256=example.item.commitment_sha256,
                    family=example.family,
                    raw_top5=example.item.raw_top5,
                    candidate_top5=candidate,
                    quantized_margin=margin,
                    raw_true_utility=raw_true_utility,
                    candidate_true_utility_delta=delta,
                    prediction_commitment_sha256=prediction_commitment,
                )
            )
        fold_predictions.append(tuple(prediction_rows))
    if len(observations) != A_FORM_QUOTA_PER_FAMILY * len(FAMILY_ORDER):
        raise WikiSQLUAOPolicyError("union OOF calibration coverage drifted")

    evaluations: list[ThresholdEvaluation] = []
    raw_total_utility = sum(row.raw_true_utility for row in observations)
    for threshold in NO_OP_MARGIN_GRID:
        deltas: list[int] = []
        action_count = 0
        for row in observations:
            acted = (
                row.candidate_top5 != row.raw_top5
                and row.quantized_margin > threshold
            )
            action_count += int(acted)
            deltas.append(row.candidate_true_utility_delta if acted else 0)
        evaluations.append(
            ThresholdEvaluation(
                margin_threshold=threshold,
                raw_total_utility=raw_total_utility,
                total_true_utility=raw_total_utility + sum(deltas),
                net_utility=sum(deltas),
                action_count=action_count,
                support_count=sum(delta > 0 for delta in deltas),
                counter_count=sum(delta < 0 for delta in deltas),
                neutral_count=sum(delta == 0 for delta in deltas),
            )
        )
    selected = max(
        evaluations,
        key=lambda row: (
            row.net_utility,
            row.margin_threshold,
            -row.action_count,
        ),
    )
    family_delta = tuple(
        (
            family,
            sum(
                row.candidate_true_utility_delta
                for row in observations
                if row.family == family
                and row.candidate_top5 != row.raw_top5
                and row.quantized_margin > selected.margin_threshold
            ),
        )
        for family in FAMILY_ORDER
    )
    return NoOpCalibrationReceipt(
        selected_claim_ids=selected_claim_ids,
        threshold_evaluations=tuple(evaluations),
        selected_margin_threshold=selected.margin_threshold,
        fold_model_sha256=tuple(fold_models),
        fold_prediction_commitments=tuple(fold_predictions),
        fold_family_counts=tuple(
            tuple(
                (
                    family,
                    sum(
                        row.family == family and row.fold_index == fold_index
                        for row in train_items
                    ),
                )
                for family in FAMILY_ORDER
            )
            for fold_index in range(CROSS_FOLDS)
        ),
        family_utility_delta=family_delta,
    )


def fit_uao_policy(train_items: Sequence[TrainingItem]) -> PolicyFormation:
    """Cross-fit the four fixed claims and compile exactly two on TRAIN only."""

    ordered = _canonical_training_items(train_items)
    receipts = tuple(_fit_claim(ordered, recipe) for recipe in CLAIM_RECIPES)
    selected, selection_receipt = _select_claims(receipts)
    calibration = _calibrate_no_op_threshold(ordered, selected)
    model = _fit_union_model(ordered, selected)
    policy = CompiledPolicy(
        selected_claim_ids=selected,
        model=model,
        probe_receipt_sha256=tuple(
            row.receipt_sha256 for row in receipts
        ),
        claim_selection_receipt_sha256=selection_receipt.receipt_sha256,
        no_op_calibration_receipt_sha256=calibration.receipt_sha256,
        margin_threshold=calibration.selected_margin_threshold,
        train_item_count=len(ordered),
    )
    return PolicyFormation(
        policy=policy,
        probe_receipts=receipts,
        claim_selection_receipt=selection_receipt,
        no_op_calibration_receipt=calibration,
    )


def apply_uao_policy(
    policy: CompiledPolicy,
    *,
    question: str,
    headers: tuple[str, ...],
    types: tuple[str, ...],
    serialized_rows: tuple[str, ...],
    raw_top5: tuple[int | None, ...],
    embeddings: PrecomputedEmbeddings | None = None,
) -> tuple[int | None, ...]:
    """Apply a frozen policy to one label-free held-out item.

    The exact public signature is part of the leakage boundary: it contains no
    gold rows, family, answer, aggregation, conditions, or SQL.
    """

    if not isinstance(policy, CompiledPolicy):
        raise WikiSQLUAOPolicyError("policy must be a CompiledPolicy")
    # Reconstruction revalidates every caller-owned field and prevents forged
    # dataclass instances from bypassing LabelFreeItem.__post_init__.
    item = LabelFreeItem(
        question=question,
        headers=headers,
        types=types,
        serialized_rows=serialized_rows,
        raw_top5=raw_top5,
        embeddings=embeddings,
    )
    prepared = _prepare(item)
    scores = _probability_scores(
        policy.model,
        prepared,
        lambda value, row_index: _union_features(
            value, row_index, policy.selected_claim_ids
        ),
    )
    action = _learned_action(
        raw_top5,
        scores,
        margin_threshold=policy.margin_threshold,
    )
    # _learned_action returns the original tuple on abstention.  Keep this
    # explicit so future refactors do not silently rebuild/reorder RAW.
    return raw_top5 if action is raw_top5 else action


# Short aliases used by controllers that already carry the UAO study namespace.
fit_policy = fit_uao_policy
apply_policy = apply_uao_policy


__all__ = [
    "A_FORM_QUOTA_PER_FAMILY",
    "A_FORM_QUOTA_PER_FOLD_FAMILY",
    "A_HOLD_QUOTA_PER_FAMILY",
    "ACTION_REDUNDANCY_PENALTY_PER_MATCH",
    "ACTION_VECTOR_LENGTH",
    "ACTION_VECTOR_MIN_HAMMING",
    "CLAIM_RECIPES",
    "CROSS_FOLDS",
    "ClaimProbeReceipt",
    "ClaimRecipe",
    "ClaimSelectionCandidate",
    "ClaimSelectionReceipt",
    "CompiledPolicy",
    "FAMILY_ORDER",
    "FoldProbeReceipt",
    "LabelFreeItem",
    "LogisticModel",
    "NO_OP_MARGIN_GRID",
    "NoOpCalibrationReceipt",
    "PolicyFormation",
    "PrecomputedEmbeddings",
    "SELECTED_CLAIM_COUNT",
    "TOP_K",
    "ThresholdEvaluation",
    "TrainingItem",
    "VERSION",
    "UTILITY_MARGIN_SCALE",
    "WikiSQLUAOPolicyError",
    "apply_policy",
    "apply_uao_policy",
    "canonical_json_bytes",
    "canonical_sha256",
    "compiled_policy_from_private_payload",
    "extract_anchors",
    "fit_policy",
    "fit_uao_policy",
    "normalize_number",
    "normalize_text",
    "precomputed_embeddings_from_payload",
    "text_tokens",
]
