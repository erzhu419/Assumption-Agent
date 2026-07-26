"""Pure offline typed-recipe and evaluator core for MAUD extraction P2.

The module deliberately has no file, network, model, API, credential, retry,
or formal-source entrypoint.  Callers provide one exact contract context,
questions, and already-computed local model coordinates.  This core then
implements the source-independent parts frozen by
``MAUD_EXTRACTION_P2_CGROUP_BOUNDED_EVALUATOR_V1``:

* deterministic overlapping passage windows and canonical passage bytes;
* exact local BM25 plus million-scale half-even model coordinates;
* four regex/section-metadata typed edge families and nine fixed recipes;
* the fixed E0 selector and the one A_form-only lambda-one ridge challenger;
* exact character-union evidence coverage; and
* equal-contract cluster aggregation with a complete sign-flip reference tail.

Local passage ordinals and the public query family are structural coordinates.
Contract identity, item identity, answerability, answers, spans, qrels, and
block identity are not accepted by either evaluator.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from fractions import Fraction
import hashlib
import json
import math
from numbers import Real
import re
import unicodedata
from typing import Mapping, Sequence

import numpy as np


STUDY_ID = "MAUD_EXTRACTION_P2_CGROUP_BOUNDED_EVALUATOR_V1"
VERSION = "maud_extraction_p2_typed_core_v1"

INTEGER_SCALE = 1_000_000
TOP_K = 5
TARGET_CODE_POINTS = 1_200
MINIMUM_PREFERRED_CODE_POINTS = 800
HARD_MAXIMUM_CODE_POINTS = 1_400
OVERLAP_TARGET_CODE_POINTS = 240
BM25_K1 = Fraction(6, 5)
BM25_B = Fraction(3, 4)
RIDGE_L2 = 1.0
PROMOTION_ALPHA = Fraction(1, 10)

DEFINITION_REFERENCE = "DEFINITION_REFERENCE"
CONDITION_OBLIGATION = "CONDITION_OBLIGATION"
EXCEPTION_REMEDY = "EXCEPTION_REMEDY"
SECTION_XREF = "SECTION_XREF"
EDGE_FAMILIES = (
    DEFINITION_REFERENCE,
    CONDITION_OBLIGATION,
    EXCEPTION_REMEDY,
    SECTION_XREF,
)

FAMILY_DEFINITION_REFERENCE = "definition_reference"
FAMILY_CONDITION_OBLIGATION = "condition_obligation"
FAMILY_PROTECTION_EXCEPTION_REMEDY = "protection_exception_remedy"
QUERY_FAMILIES = (
    FAMILY_DEFINITION_REFERENCE,
    FAMILY_CONDITION_OBLIGATION,
    FAMILY_PROTECTION_EXCEPTION_REMEDY,
)

FEATURE_ORDER = (
    "mean_cross_encoder_sigmoid",
    "minimum_cross_encoder_sigmoid",
    "mean_MiniLM_cosine_unit_interval",
    "minimum_MiniLM_cosine_unit_interval",
    "mean_normalized_BM25",
    "query_lexical_coverage",
    "pairwise_MiniLM_diversity",
    "raw_top5_churn",
    "definition_reference_closure",
    "condition_obligation_closure",
    "exception_remedy_closure",
    "section_cross_reference_closure",
)
BASE_FEATURE_COUNT = len(FEATURE_ORDER)
E1_FEATURE_COUNT = BASE_FEATURE_COUNT * (1 + len(QUERY_FAMILIES))

R0_CE_XREF_1SWAP = "R0_CE_XREF_1SWAP"
R1_FUSED_DEFINITION_1SWAP = "R1_FUSED_DEFINITION_1SWAP"
R2_FUSED_CONDITION_1SWAP = "R2_FUSED_CONDITION_1SWAP"
R3_FUSED_EXCEPTION_1SWAP = "R3_FUSED_EXCEPTION_1SWAP"
R4_FUSED_XREF_1SWAP = "R4_FUSED_XREF_1SWAP"
R5_FUSED_DEFINITION_XREF_2SWAP = "R5_FUSED_DEFINITION_XREF_2SWAP"
R6_FUSED_CONDITION_EXCEPTION_2SWAP = (
    "R6_FUSED_CONDITION_EXCEPTION_2SWAP"
)
R7_FUSED_DEFINITION_CONDITION_2SWAP = (
    "R7_FUSED_DEFINITION_CONDITION_2SWAP"
)
R8_FUSED_ALL_TYPED_3SWAP = "R8_FUSED_ALL_TYPED_3SWAP"
RECIPE_IDS = (
    R0_CE_XREF_1SWAP,
    R1_FUSED_DEFINITION_1SWAP,
    R2_FUSED_CONDITION_1SWAP,
    R3_FUSED_EXCEPTION_1SWAP,
    R4_FUSED_XREF_1SWAP,
    R5_FUSED_DEFINITION_XREF_2SWAP,
    R6_FUSED_CONDITION_EXCEPTION_2SWAP,
    R7_FUSED_DEFINITION_CONDITION_2SWAP,
    R8_FUSED_ALL_TYPED_3SWAP,
)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_REFERENCE_RE = re.compile(
    r"\b(section|clause|article)\s+"
    r"([0-9]+(?:\.[0-9A-Za-z]+)*|[IVXLC]+)\b",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(
    r"(?im)^\s*(section|clause|article)\s+"
    r"([0-9]+(?:\.[0-9A-Za-z]+)*|[IVXLC]+)\b"
)
_QUOTED_DEFINITION_RE = re.compile(
    r"[\"“](?P<term>[^\"”\n]{1,120})[\"”]\s+"
    r"(?:means|shall\s+mean|has\s+the\s+meaning)",
    re.IGNORECASE,
)
_CAPITALIZED_DEFINITION_RE = re.compile(
    r"\b(?P<term>[A-Z][A-Za-z0-9&'’.-]*"
    r"(?:\s+[A-Z][A-Za-z0-9&'’.-]*){0,5})\s+"
    r"(?:means|shall\s+mean|has\s+the\s+meaning)\b"
)
_CONDITION_TRIGGER_RE = re.compile(
    r"\b(?:condition(?:s)?\s+(?:precedent|to\s+closing)|closing\s+condition|"
    r"provided\s+that|subject\s+to|shall|must|covenant(?:s|ed|ing)?|"
    r"obligat(?:e|es|ed|ion|ions|ory)|required\s+to)\b",
    re.IGNORECASE,
)
_EXCEPTION_TRIGGER_RE = re.compile(
    r"\b(?:except(?:ion|ions)?|unless|provided\s*,?\s*however|"
    r"remed(?:y|ies|ial)|specific\s+performance|injunct(?:ion|ive)|"
    r"breach|terminat(?:e|es|ed|ion|ions))\b",
    re.IGNORECASE,
)


class MaudExtractionP2CoreError(ValueError):
    """The frozen typed, evaluator, coverage, or cluster contract drifted."""


def _strict_int(
    value: object,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise MaudExtractionP2CoreError(f"{field} must be an exact integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise MaudExtractionP2CoreError(f"{field} is below its lower bound")
    if maximum is not None and result > maximum:
        raise MaudExtractionP2CoreError(f"{field} exceeds its upper bound")
    return result


def _finite_real(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise MaudExtractionP2CoreError(f"{field} must be a finite real")
    result = float(value)
    if not math.isfinite(result):
        raise MaudExtractionP2CoreError(f"{field} must be a finite real")
    return 0.0 if result == 0.0 else result


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise MaudExtractionP2CoreError(
            "value cannot be represented as canonical JSON"
        ) from exc


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _round_fraction_half_even(value: Fraction) -> int:
    """Round one exact rational to the nearest integer, ties to even."""

    numerator = value.numerator
    denominator = value.denominator
    sign = -1 if numerator < 0 else 1
    absolute = abs(numerator)
    quotient, remainder = divmod(absolute, denominator)
    doubled = remainder * 2
    if doubled > denominator or (doubled == denominator and quotient % 2):
        quotient += 1
    return sign * quotient


def quantize_half_even(value: object, *, field: str = "coordinate") -> int:
    """Quantize a finite real to signed million scale with decimal half-even."""

    numeric = _finite_real(value, field)
    decimal = Decimal(str(numeric)) * Decimal(INTEGER_SCALE)
    return int(decimal.quantize(Decimal("1"), rounding=ROUND_HALF_EVEN))


def _mean_integer(values: Sequence[int], field: str) -> int:
    if not values or any(type(value) is not int for value in values):
        raise MaudExtractionP2CoreError(f"{field} integer vector is malformed")
    return _round_fraction_half_even(Fraction(sum(values), len(values)))


def _normalized_text(value: str) -> str:
    return _SPACE_RE.sub(
        " ", unicodedata.normalize("NFKC", value).casefold()
    ).strip()


def lexical_tokens(value: str) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise MaudExtractionP2CoreError("lexical input must be text")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(_TOKEN_RE.findall(normalized))


@dataclass(frozen=True)
class Passage:
    """One exact, overlapping passage in Python code-point coordinates."""

    ordinal: int
    context_sha256: str
    start: int
    end: int
    text: str
    exact_substring_sha256: str

    def __post_init__(self) -> None:
        ordinal = _strict_int(self.ordinal, "passage ordinal", minimum=0)
        start = _strict_int(self.start, "passage start", minimum=0)
        end = _strict_int(self.end, "passage end", minimum=start + 1)
        if (
            not isinstance(self.text, str)
            or not self.text
            or not self.text.strip()
            or len(self.text) != end - start
        ):
            raise MaudExtractionP2CoreError("passage exact text is malformed")
        expected = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if (
            not isinstance(self.context_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.context_sha256) is None
            or self.exact_substring_sha256 != expected
        ):
            raise MaudExtractionP2CoreError("passage identity drifted")
        object.__setattr__(self, "ordinal", ordinal)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def identity(self) -> tuple[str, int, int, str]:
        return (
            self.context_sha256,
            self.start,
            self.end,
            self.exact_substring_sha256,
        )

    @property
    def serialized_title(self) -> str:
        return f"MAUD passage {self.ordinal:06d}"

    def serialized_bytes(self) -> bytes:
        return _canonical_json_bytes(
            {"text": self.text, "title": self.serialized_title}
        )


def _whitespace_boundaries(context: str) -> tuple[int, ...]:
    # A boundary is immediately after a Unicode whitespace code point.  This
    # keeps each selected substring exact and makes every boundary a Python
    # code-point offset.
    return tuple(
        [0]
        + [index + 1 for index, character in enumerate(context) if character.isspace()]
        + [len(context)]
    )


def build_passages(context: str) -> tuple[Passage, ...]:
    """Apply the frozen 1200/800/1400 window and 240-overlap rules."""

    if not isinstance(context, str) or not context:
        raise MaudExtractionP2CoreError("contract context must be nonempty")
    context_sha256 = hashlib.sha256(context.encode("utf-8")).hexdigest()
    if len(context) <= TARGET_CODE_POINTS:
        if not context.strip():
            raise MaudExtractionP2CoreError(
                "contract context contains no non-whitespace passage"
            )
        return (
            Passage(
                ordinal=0,
                context_sha256=context_sha256,
                start=0,
                end=len(context),
                text=context,
                exact_substring_sha256=hashlib.sha256(
                    context.encode("utf-8")
                ).hexdigest(),
            ),
        )

    boundaries = _whitespace_boundaries(context)
    retained: list[tuple[int, int, str]] = []
    start = 0
    while start < len(context):
        target = min(start + TARGET_CODE_POINTS, len(context))
        if target == len(context):
            end = len(context)
        else:
            earlier = [
                boundary
                for boundary in boundaries
                if start + MINIMUM_PREFERRED_CODE_POINTS
                <= boundary
                <= target
            ]
            if earlier:
                end = earlier[-1]
            else:
                later = [
                    boundary
                    for boundary in boundaries
                    if target
                    < boundary
                    <= min(start + HARD_MAXIMUM_CODE_POINTS, len(context))
                ]
                end = later[0] if later else target
        if end <= start or end - start > HARD_MAXIMUM_CODE_POINTS:
            raise MaudExtractionP2CoreError("passage boundary failed to progress")
        text = context[start:end]
        if text.strip():
            retained.append((start, end, text))
        if end == len(context):
            break
        overlap_boundary = end - OVERLAP_TARGET_CODE_POINTS
        candidates = [
            boundary
            for boundary in boundaries
            if start < boundary <= overlap_boundary
        ]
        next_start = candidates[-1] if candidates else end
        if next_start <= start:
            raise MaudExtractionP2CoreError(
                "next passage start failed to progress"
            )
        start = next_start

    if not retained:
        raise MaudExtractionP2CoreError(
            "contract context contains no non-whitespace passage"
        )
    passages = tuple(
        Passage(
            ordinal=ordinal,
            context_sha256=context_sha256,
            start=start_value,
            end=end_value,
            text=text,
            exact_substring_sha256=hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest(),
        )
        for ordinal, (start_value, end_value, text) in enumerate(retained)
    )
    if tuple(row.ordinal for row in passages) != tuple(range(len(passages))):
        raise MaudExtractionP2CoreError("passage ordinal sequence drifted")
    if any(
        context[row.start : row.end] != row.text
        or row.end - row.start > HARD_MAXIMUM_CODE_POINTS
        for row in passages
    ):
        raise MaudExtractionP2CoreError("passage exact substring drifted")
    return passages


def serialized_passage_corpus(passages: Sequence[Passage]) -> tuple[bytes, ...]:
    checked = _validated_passages(passages)
    return tuple(row.serialized_bytes() for row in checked)


def _validated_passages(passages: Sequence[Passage]) -> tuple[Passage, ...]:
    if (
        isinstance(passages, (str, bytes))
        or not isinstance(passages, Sequence)
        or not passages
        or not all(isinstance(row, Passage) for row in passages)
    ):
        raise MaudExtractionP2CoreError(
            "passage corpus requires at least one Passage row"
        )
    checked = tuple(passages)
    if tuple(row.ordinal for row in checked) != tuple(range(len(checked))):
        raise MaudExtractionP2CoreError(
            "passages must be in contiguous ordinal order"
        )
    contexts = {row.context_sha256 for row in checked}
    if len(contexts) != 1:
        raise MaudExtractionP2CoreError(
            "passages do not share one contract context identity"
        )
    if any(
        (left.start, left.end) >= (right.start, right.end)
        for left, right in zip(checked, checked[1:])
    ):
        raise MaudExtractionP2CoreError(
            "passages are not strictly ordered by start then end"
        )
    return checked


def bm25_scores(
    query: str, serialized_documents: Sequence[bytes]
) -> tuple[float, ...]:
    """Return exact local Okapi BM25(k1=1.2,b=.75) scores."""

    if (
        not isinstance(query, str)
        or not query.strip()
        or isinstance(serialized_documents, (str, bytes))
        or not isinstance(serialized_documents, Sequence)
        or not serialized_documents
        or any(not isinstance(row, bytes) for row in serialized_documents)
    ):
        raise MaudExtractionP2CoreError("BM25 query or corpus is malformed")
    query_terms = lexical_tokens(query)
    documents: list[tuple[str, ...]] = []
    for raw in serialized_documents:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MaudExtractionP2CoreError(
                "serialized passage is not UTF-8"
            ) from exc
        documents.append(lexical_tokens(text))
    lengths = [len(row) for row in documents]
    average_length = sum(lengths) / len(lengths)
    if average_length <= 0:
        return tuple(0.0 for _ in documents)
    document_frequencies = Counter(
        term
        for term in set(query_terms)
        for row in documents
        if term in set(row)
    )
    # The Counter expression above counts one occurrence per document.
    total = len(documents)
    scores: list[float] = []
    for terms in documents:
        frequencies = Counter(terms)
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            df = document_frequencies.get(term, 0)
            inverse = math.log1p((total - df + 0.5) / (df + 0.5))
            denominator = frequency + float(BM25_K1) * (
                1.0
                - float(BM25_B)
                + float(BM25_B) * len(terms) / average_length
            )
            score += inverse * (
                frequency * (float(BM25_K1) + 1.0) / denominator
            )
        scores.append(score)
    return tuple(scores)


def normalized_bm25_coordinates(
    query: str, passages: Sequence[Passage]
) -> tuple[int, ...]:
    documents = serialized_passage_corpus(passages)
    scores = bm25_scores(query, documents)
    maximum = max(scores, default=0.0)
    if maximum <= 0.0:
        return tuple(0 for _ in scores)
    return tuple(
        _strict_int(
            quantize_half_even(score / maximum, field="normalized BM25"),
            "normalized BM25 coordinate",
            minimum=0,
            maximum=INTEGER_SCALE,
        )
        for score in scores
    )


@dataclass(frozen=True)
class CoordinateTable:
    """All item-local integer coordinates; model values arrive from callers."""

    cross_encoder: tuple[int, ...]
    minilm: tuple[int, ...]
    bm25: tuple[int, ...]
    fused: tuple[int, ...]
    pairwise_minilm: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        vectors = (
            self.cross_encoder,
            self.minilm,
            self.bm25,
            self.fused,
        )
        lengths = {len(row) for row in vectors}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 1:
            raise MaudExtractionP2CoreError(
                "coordinate vectors have inconsistent width"
            )
        width = next(iter(lengths))
        for name, vector in zip(
            ("cross encoder", "MiniLM", "BM25", "fused"), vectors
        ):
            if any(
                type(value) is not int
                or not 0 <= value <= INTEGER_SCALE
                for value in vector
            ):
                raise MaudExtractionP2CoreError(
                    f"{name} coordinates must be million-scale integers"
                )
        if (
            len(self.pairwise_minilm) != width
            or any(len(row) != width for row in self.pairwise_minilm)
        ):
            raise MaudExtractionP2CoreError(
                "pairwise MiniLM matrix shape drifted"
            )
        for left in range(width):
            for right in range(width):
                value = self.pairwise_minilm[left][right]
                if (
                    type(value) is not int
                    or not 0 <= value <= INTEGER_SCALE
                    or value != self.pairwise_minilm[right][left]
                    or (left == right and value != INTEGER_SCALE)
                ):
                    raise MaudExtractionP2CoreError(
                        "pairwise MiniLM matrix is not symmetric unit scale"
                    )

    @property
    def passage_count(self) -> int:
        return len(self.cross_encoder)

    def vector(self, name: str) -> tuple[int, ...]:
        if name == "CE":
            return self.cross_encoder
        if name == "FUSED":
            return self.fused
        raise MaudExtractionP2CoreError("unknown frozen base coordinate")


def build_coordinate_table(
    *,
    query: str,
    passages: Sequence[Passage],
    cross_encoder_logits: Sequence[object],
    minilm_cosines: Sequence[object],
    pairwise_minilm_cosines: Sequence[Sequence[object]],
) -> CoordinateTable:
    """Quantize external CE/MiniLM coordinates and compute local BM25/FUSED."""

    checked = _validated_passages(passages)
    width = len(checked)
    if (
        isinstance(cross_encoder_logits, (str, bytes))
        or isinstance(minilm_cosines, (str, bytes))
        or len(cross_encoder_logits) != width
        or len(minilm_cosines) != width
        or len(pairwise_minilm_cosines) != width
        or any(len(row) != width for row in pairwise_minilm_cosines)
    ):
        raise MaudExtractionP2CoreError(
            "external model coordinate shape drifted"
        )
    ce: list[int] = []
    minilm: list[int] = []
    for index, raw in enumerate(cross_encoder_logits):
        logit = _finite_real(raw, f"cross-encoder logit {index}")
        if logit >= 0:
            probability = 1.0 / (1.0 + math.exp(-logit))
        else:
            exponential = math.exp(logit)
            probability = exponential / (1.0 + exponential)
        ce.append(
            _strict_int(
                quantize_half_even(
                    probability, field="cross-encoder sigmoid"
                ),
                "cross-encoder coordinate",
                minimum=0,
                maximum=INTEGER_SCALE,
            )
        )
    for index, raw in enumerate(minilm_cosines):
        cosine = _finite_real(raw, f"MiniLM cosine {index}")
        if not -1.0 <= cosine <= 1.0:
            raise MaudExtractionP2CoreError(
                "MiniLM cosine lies outside [-1, 1]"
            )
        minilm.append(
            _strict_int(
                quantize_half_even(
                    (cosine + 1.0) / 2.0,
                    field="MiniLM unit coordinate",
                ),
                "MiniLM coordinate",
                minimum=0,
                maximum=INTEGER_SCALE,
            )
        )
    pairwise_rows: list[tuple[int, ...]] = []
    for left, row in enumerate(pairwise_minilm_cosines):
        converted: list[int] = []
        for right, raw in enumerate(row):
            cosine = _finite_real(
                raw, f"pairwise MiniLM cosine {left},{right}"
            )
            if not -1.0 <= cosine <= 1.0:
                raise MaudExtractionP2CoreError(
                    "pairwise MiniLM cosine lies outside [-1, 1]"
                )
            converted.append(
                _strict_int(
                    quantize_half_even(
                        (cosine + 1.0) / 2.0,
                        field="pairwise MiniLM unit coordinate",
                    ),
                    "pairwise MiniLM coordinate",
                    minimum=0,
                    maximum=INTEGER_SCALE,
                )
            )
        pairwise_rows.append(tuple(converted))
    bm25 = normalized_bm25_coordinates(query, checked)
    fused = tuple(
        _round_fraction_half_even(
            Fraction(1, 2) * ce[index]
            + Fraction(3, 10) * minilm[index]
            + Fraction(1, 5) * bm25[index]
        )
        for index in range(width)
    )
    return CoordinateTable(
        cross_encoder=tuple(ce),
        minilm=tuple(minilm),
        bm25=bm25,
        fused=fused,
        pairwise_minilm=tuple(pairwise_rows),
    )


def build_coordinate_table_from_quantized(
    *,
    query: str,
    passages: Sequence[Passage],
    cross_encoder_sigmoid: Sequence[int],
    minilm_unit_interval: Sequence[int],
    pairwise_minilm_unit_interval: Sequence[Sequence[int]],
) -> CoordinateTable:
    """Join pre-quantized model outputs with local BM25 and exact FUSED.

    Production model workers may quantize their outputs at the model boundary.
    This entrypoint accepts only exact (non-boolean) million-scale integers, so
    no model coordinate is converted through floating point a second time.
    """

    checked = _validated_passages(passages)
    width = len(checked)
    vectors = (cross_encoder_sigmoid, minilm_unit_interval)
    if any(
        isinstance(vector, (str, bytes))
        or not isinstance(vector, Sequence)
        or len(vector) != width
        for vector in vectors
    ):
        raise MaudExtractionP2CoreError(
            "quantized model coordinate shape drifted"
        )
    if (
        isinstance(pairwise_minilm_unit_interval, (str, bytes))
        or not isinstance(pairwise_minilm_unit_interval, Sequence)
        or len(pairwise_minilm_unit_interval) != width
        or any(
            isinstance(row, (str, bytes))
            or not isinstance(row, Sequence)
            or len(row) != width
            for row in pairwise_minilm_unit_interval
        )
    ):
        raise MaudExtractionP2CoreError(
            "quantized pairwise coordinate shape drifted"
        )
    ce = tuple(
        _strict_int(
            value,
            f"quantized cross-encoder coordinate {index}",
            minimum=0,
            maximum=INTEGER_SCALE,
        )
        for index, value in enumerate(cross_encoder_sigmoid)
    )
    minilm = tuple(
        _strict_int(
            value,
            f"quantized MiniLM coordinate {index}",
            minimum=0,
            maximum=INTEGER_SCALE,
        )
        for index, value in enumerate(minilm_unit_interval)
    )
    pairwise = tuple(
        tuple(
            _strict_int(
                value,
                f"quantized pairwise MiniLM coordinate {left},{right}",
                minimum=0,
                maximum=INTEGER_SCALE,
            )
            for right, value in enumerate(row)
        )
        for left, row in enumerate(pairwise_minilm_unit_interval)
    )
    bm25 = normalized_bm25_coordinates(query, checked)
    fused = tuple(
        _round_fraction_half_even(
            Fraction(1, 2) * ce[index]
            + Fraction(3, 10) * minilm[index]
            + Fraction(1, 5) * bm25[index]
        )
        for index in range(width)
    )
    return CoordinateTable(
        cross_encoder=ce,
        minilm=minilm,
        bm25=bm25,
        fused=fused,
        pairwise_minilm=pairwise,
    )


@dataclass(frozen=True, order=True)
class SectionHeading:
    reference: str
    passage_ordinal: int

    def __post_init__(self) -> None:
        reference = normalize_section_reference(self.reference)
        ordinal = _strict_int(
            self.passage_ordinal, "heading passage ordinal", minimum=0
        )
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "passage_ordinal", ordinal)


@dataclass(frozen=True, order=True)
class TypedEdge:
    source_ordinal: int
    target_ordinal: int
    edge_family: str

    def __post_init__(self) -> None:
        source = _strict_int(self.source_ordinal, "edge source", minimum=0)
        target = _strict_int(self.target_ordinal, "edge target", minimum=0)
        if source == target or self.edge_family not in EDGE_FAMILIES:
            raise MaudExtractionP2CoreError("typed edge is malformed")
        object.__setattr__(self, "source_ordinal", source)
        object.__setattr__(self, "target_ordinal", target)


def normalize_section_reference(value: str) -> str:
    if not isinstance(value, str):
        raise MaudExtractionP2CoreError("section reference must be text")
    normalized = _normalized_text(value).rstrip(".")
    match = re.fullmatch(
        r"(section|clause|article)\s+"
        r"([0-9]+(?:\.[0-9a-z]+)*|[ivxlc]+)",
        normalized,
    )
    if match is None:
        raise MaudExtractionP2CoreError("section reference is malformed")
    return f"{match.group(1)} {match.group(2)}"


def _references(text: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                normalize_section_reference(
                    f"{match.group(1)} {match.group(2)}"
                )
                for match in _REFERENCE_RE.finditer(text)
            }
        )
    )


def _definition_terms(text: str) -> tuple[str, ...]:
    terms = {
        _normalized_text(match.group("term")).strip("\"“”'‘’ ")
        for expression in (_QUOTED_DEFINITION_RE, _CAPITALIZED_DEFINITION_RE)
        for match in expression.finditer(text)
    }
    return tuple(sorted(term for term in terms if term))


def _whole_normalized_term_mention(text: str, term: str) -> bool:
    normalized = _normalized_text(text)
    return (
        re.search(
            rf"(?<!\w){re.escape(term)}(?!\w)",
            normalized,
            flags=re.UNICODE,
        )
        is not None
    )


def build_typed_edges(
    passages: Sequence[Passage],
    *,
    section_headings: Sequence[SectionHeading] = (),
) -> tuple[TypedEdge, ...]:
    """Build only the four frozen regex/section-metadata edge families."""

    checked = _validated_passages(passages)
    width = len(checked)
    if isinstance(section_headings, (str, bytes)) or not isinstance(
        section_headings, Sequence
    ):
        raise MaudExtractionP2CoreError("section heading registry is malformed")
    explicit = tuple(section_headings)
    if any(not isinstance(row, SectionHeading) for row in explicit):
        raise MaudExtractionP2CoreError("section heading registry is malformed")
    derived: list[SectionHeading] = []
    for passage in checked:
        for match in _HEADING_RE.finditer(passage.text):
            derived.append(
                SectionHeading(
                    reference=f"{match.group(1)} {match.group(2)}",
                    passage_ordinal=passage.ordinal,
                )
            )
    headings = tuple(sorted(set((*explicit, *derived))))
    if any(row.passage_ordinal >= width for row in headings):
        raise MaudExtractionP2CoreError(
            "section heading points outside the passage corpus"
        )
    heading_targets: dict[str, int] = {}
    grouped: dict[str, set[int]] = {}
    for row in headings:
        grouped.setdefault(row.reference, set()).add(row.passage_ordinal)
    for reference, ordinals in grouped.items():
        if len(ordinals) == 1:
            heading_targets[reference] = next(iter(ordinals))

    edges: set[TypedEdge] = set()
    for source in checked:
        references = _references(source.text)
        referenced_targets = {
            heading_targets[reference]
            for reference in references
            if reference in heading_targets
            and heading_targets[reference] != source.ordinal
        }
        for target in sorted(referenced_targets):
            edges.add(
                TypedEdge(source.ordinal, target, SECTION_XREF)
            )

        for term in _definition_terms(source.text):
            for target in checked:
                if (
                    target.ordinal != source.ordinal
                    and _whole_normalized_term_mention(target.text, term)
                ):
                    edges.add(
                        TypedEdge(
                            source.ordinal,
                            target.ordinal,
                            DEFINITION_REFERENCE,
                        )
                    )

        if _CONDITION_TRIGGER_RE.search(source.text):
            adjacent = (
                source.ordinal + 1
                if source.ordinal + 1 < width
                else None
            )
            targets = set(referenced_targets)
            if adjacent is not None:
                targets.add(adjacent)
            for target in sorted(targets):
                if target != source.ordinal:
                    edges.add(
                        TypedEdge(
                            source.ordinal,
                            target,
                            CONDITION_OBLIGATION,
                        )
                    )

        if _EXCEPTION_TRIGGER_RE.search(source.text):
            targets = set(referenced_targets)
            if source.ordinal > 0:
                targets.add(source.ordinal - 1)
            for target in sorted(targets):
                if target != source.ordinal:
                    edges.add(
                        TypedEdge(
                            source.ordinal,
                            target,
                            EXCEPTION_REMEDY,
                        )
                    )
    return tuple(
        sorted(
            edges,
            key=lambda row: (
                EDGE_FAMILIES.index(row.edge_family),
                row.source_ordinal,
                row.target_ordinal,
            ),
        )
    )


@dataclass(frozen=True)
class RecipeSpec:
    recipe_id: str
    base_coordinate: str
    edge_families: tuple[str, ...]
    maximum_swaps: int

    def __post_init__(self) -> None:
        if (
            self.recipe_id not in RECIPE_IDS
            or self.base_coordinate not in {"CE", "FUSED"}
            or not self.edge_families
            or any(row not in EDGE_FAMILIES for row in self.edge_families)
            or len(set(self.edge_families)) != len(self.edge_families)
        ):
            raise MaudExtractionP2CoreError("recipe specification drifted")
        _strict_int(
            self.maximum_swaps,
            "recipe maximum swaps",
            minimum=1,
            maximum=3,
        )


RECIPE_REGISTRY = (
    RecipeSpec(R0_CE_XREF_1SWAP, "CE", (SECTION_XREF,), 1),
    RecipeSpec(
        R1_FUSED_DEFINITION_1SWAP,
        "FUSED",
        (DEFINITION_REFERENCE,),
        1,
    ),
    RecipeSpec(
        R2_FUSED_CONDITION_1SWAP,
        "FUSED",
        (CONDITION_OBLIGATION,),
        1,
    ),
    RecipeSpec(
        R3_FUSED_EXCEPTION_1SWAP,
        "FUSED",
        (EXCEPTION_REMEDY,),
        1,
    ),
    RecipeSpec(R4_FUSED_XREF_1SWAP, "FUSED", (SECTION_XREF,), 1),
    RecipeSpec(
        R5_FUSED_DEFINITION_XREF_2SWAP,
        "FUSED",
        (DEFINITION_REFERENCE, SECTION_XREF),
        2,
    ),
    RecipeSpec(
        R6_FUSED_CONDITION_EXCEPTION_2SWAP,
        "FUSED",
        (CONDITION_OBLIGATION, EXCEPTION_REMEDY),
        2,
    ),
    RecipeSpec(
        R7_FUSED_DEFINITION_CONDITION_2SWAP,
        "FUSED",
        (DEFINITION_REFERENCE, CONDITION_OBLIGATION),
        2,
    ),
    RecipeSpec(
        R8_FUSED_ALL_TYPED_3SWAP,
        "FUSED",
        EDGE_FAMILIES,
        3,
    ),
)


@dataclass(frozen=True)
class RecipeAction:
    recipe_id: str
    passage_ordinals: tuple[int, ...]
    accepted_edges: tuple[TypedEdge, ...]
    behavior_sha256: str

    def __post_init__(self) -> None:
        if (
            self.recipe_id not in RECIPE_IDS
            or len(self.passage_ordinals) != TOP_K
            or len(set(self.passage_ordinals)) != TOP_K
            or any(type(row) is not int or row < 0 for row in self.passage_ordinals)
            or any(not isinstance(row, TypedEdge) for row in self.accepted_edges)
            or re.fullmatch(r"[0-9a-f]{64}", self.behavior_sha256) is None
        ):
            raise MaudExtractionP2CoreError("recipe action is malformed")
        expected = _semantic_hash(
            {
                "accepted_edges": [
                    {
                        "edge_family": row.edge_family,
                        "source_ordinal": row.source_ordinal,
                        "target_ordinal": row.target_ordinal,
                    }
                    for row in self.accepted_edges
                ],
                "passage_ordinals": list(self.passage_ordinals),
                "recipe_id": self.recipe_id,
            }
        )
        if self.behavior_sha256 != expected:
            raise MaudExtractionP2CoreError("recipe behavior hash drifted")


def _rank_ordinals(
    scores: Sequence[int], passages: Sequence[Passage]
) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(len(passages)),
            key=lambda ordinal: (
                -scores[ordinal],
                passages[ordinal].start,
                passages[ordinal].end,
                ordinal,
            ),
        )
    )


def _candidate_table(
    *,
    edges: Sequence[TypedEdge],
    seed_top5: Sequence[int],
    base_scores: Sequence[int],
    passages: Sequence[Passage],
) -> tuple[TypedEdge, ...]:
    seed_positions = {ordinal: rank for rank, ordinal in enumerate(seed_top5)}
    rows = [
        edge
        for edge in edges
        if edge.source_ordinal in seed_positions
        and edge.target_ordinal not in seed_positions
    ]
    return tuple(
        sorted(
            rows,
            key=lambda edge: (
                EDGE_FAMILIES.index(edge.edge_family),
                -(TOP_K - seed_positions[edge.source_ordinal]),
                -base_scores[edge.target_ordinal],
                passages[edge.target_ordinal].start,
                passages[edge.target_ordinal].end,
                edge.target_ordinal,
                edge.source_ordinal,
            ),
        )
    )


def materialize_recipe_actions(
    *,
    passages: Sequence[Passage],
    coordinates: CoordinateTable,
    edges: Sequence[TypedEdge],
) -> tuple[RecipeAction, ...]:
    """Execute all nine fixed swaps against one canonical candidate table."""

    checked = _validated_passages(passages)
    if len(checked) < TOP_K:
        raise MaudExtractionP2CoreError(
            "recipe corpus requires at least five Passage rows"
        )
    if coordinates.passage_count != len(checked):
        raise MaudExtractionP2CoreError(
            "coordinate and passage counts do not match"
        )
    if (
        isinstance(edges, (str, bytes))
        or not isinstance(edges, Sequence)
        or any(not isinstance(row, TypedEdge) for row in edges)
        or any(
            row.source_ordinal >= len(checked)
            or row.target_ordinal >= len(checked)
            for row in edges
        )
    ):
        raise MaudExtractionP2CoreError("typed edge registry is malformed")
    edge_rows = tuple(edges)
    if len(set(edge_rows)) != len(edge_rows):
        raise MaudExtractionP2CoreError("typed edges contain duplicates")

    actions: list[RecipeAction] = []
    for spec in RECIPE_REGISTRY:
        base_scores = coordinates.vector(spec.base_coordinate)
        full_ranking = _rank_ordinals(base_scores, checked)
        originals = tuple(full_ranking[:TOP_K])
        original_rank = {
            ordinal: rank for rank, ordinal in enumerate(originals)
        }
        selected = set(originals)
        protected: set[int] = set()
        accepted: list[TypedEdge] = []
        candidates = _candidate_table(
            edges=edge_rows,
            seed_top5=originals,
            base_scores=base_scores,
            passages=checked,
        )
        for edge in candidates:
            if (
                len(accepted) >= spec.maximum_swaps
                or edge.edge_family not in spec.edge_families
                or edge.source_ordinal not in selected
                or edge.target_ordinal in selected
            ):
                continue
            droppable = [
                ordinal
                for ordinal in originals
                if ordinal in selected
                and ordinal not in protected
                and ordinal != edge.source_ordinal
            ]
            if not droppable:
                continue
            dropped = min(
                droppable,
                key=lambda ordinal: (
                    base_scores[ordinal],
                    -original_rank[ordinal],
                    -ordinal,
                ),
            )
            selected.remove(dropped)
            selected.add(edge.target_ordinal)
            protected.add(edge.source_ordinal)
            protected.add(edge.target_ordinal)
            accepted.append(edge)
        ordered = tuple(
            sorted(
                selected,
                key=lambda ordinal: (
                    -base_scores[ordinal],
                    checked[ordinal].start,
                    checked[ordinal].end,
                    ordinal,
                ),
            )
        )
        body = {
            "accepted_edges": [
                {
                    "edge_family": row.edge_family,
                    "source_ordinal": row.source_ordinal,
                    "target_ordinal": row.target_ordinal,
                }
                for row in accepted
            ],
            "passage_ordinals": list(ordered),
            "recipe_id": spec.recipe_id,
        }
        actions.append(
            RecipeAction(
                recipe_id=spec.recipe_id,
                passage_ordinals=ordered,
                accepted_edges=tuple(accepted),
                behavior_sha256=_semantic_hash(body),
            )
        )
    return tuple(actions)


@dataclass(frozen=True)
class ActionFeatures:
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            len(self.values) != BASE_FEATURE_COUNT
            or any(
                type(value) is not int
                or not 0 <= value <= INTEGER_SCALE
                for value in self.values
            )
        ):
            raise MaudExtractionP2CoreError(
                "action features must be twelve unit-scale integers"
            )

    def as_mapping(self) -> dict[str, int]:
        return dict(zip(FEATURE_ORDER, self.values))

    @property
    def churn(self) -> int:
        return self.values[FEATURE_ORDER.index("raw_top5_churn")]

    @property
    def typed_closure(self) -> Fraction:
        # The singular E0 coordinate is the exact arithmetic mean of the four
        # registered typed-closure action features.  It is derived, not a
        # thirteenth learned feature.
        return Fraction(sum(self.values[-4:]), 4)


def compute_action_features(
    *,
    query: str,
    passages: Sequence[Passage],
    coordinates: CoordinateTable,
    edges: Sequence[TypedEdge],
    action: RecipeAction,
) -> ActionFeatures:
    checked = _validated_passages(passages)
    if coordinates.passage_count != len(checked):
        raise MaudExtractionP2CoreError(
            "feature coordinate and passage counts do not match"
        )
    selected = action.passage_ordinals
    if any(row >= len(checked) for row in selected):
        raise MaudExtractionP2CoreError(
            "recipe selected passage outside the corpus"
        )
    ce = [coordinates.cross_encoder[row] for row in selected]
    minilm = [coordinates.minilm[row] for row in selected]
    bm25 = [coordinates.bm25[row] for row in selected]
    query_terms = set(lexical_tokens(query))
    selected_terms = {
        term
        for ordinal in selected
        for term in lexical_tokens(checked[ordinal].text)
    }
    lexical = (
        _round_fraction_half_even(
            Fraction(
                len(query_terms.intersection(selected_terms)) * INTEGER_SCALE,
                len(query_terms),
            )
        )
        if query_terms
        else 0
    )
    pairwise_diversities = [
        INTEGER_SCALE - coordinates.pairwise_minilm[left][right]
        for position, left in enumerate(selected)
        for right in selected[position + 1 :]
    ]
    diversity = _mean_integer(
        pairwise_diversities, "pairwise MiniLM diversity"
    )
    raw_top5 = set(
        _rank_ordinals(coordinates.cross_encoder, checked)[:TOP_K]
    )
    churn = _round_fraction_half_even(
        Fraction(
            (TOP_K - len(raw_top5.intersection(selected))) * INTEGER_SCALE,
            TOP_K,
        )
    )
    selected_set = set(selected)
    closures: list[int] = []
    for family in EDGE_FAMILIES:
        incident = {
            ordinal
            for edge in edges
            if edge.edge_family == family
            and edge.source_ordinal in selected_set
            and edge.target_ordinal in selected_set
            for ordinal in (edge.source_ordinal, edge.target_ordinal)
        }
        closures.append(
            _round_fraction_half_even(
                Fraction(len(incident) * INTEGER_SCALE, TOP_K)
            )
        )
    return ActionFeatures(
        (
            _mean_integer(ce, "mean cross-encoder"),
            min(ce),
            _mean_integer(minilm, "mean MiniLM"),
            min(minilm),
            _mean_integer(bm25, "mean BM25"),
            lexical,
            diversity,
            churn,
            *closures,
        )
    )


@dataclass(frozen=True)
class RecipeSlate:
    actions: tuple[RecipeAction, ...]
    features: tuple[ActionFeatures, ...]

    def __post_init__(self) -> None:
        if (
            tuple(row.recipe_id for row in self.actions) != RECIPE_IDS
            or len(self.features) != len(RECIPE_IDS)
            or any(not isinstance(row, ActionFeatures) for row in self.features)
        ):
            raise MaudExtractionP2CoreError(
                "recipe slate must contain the complete registry"
            )


def build_recipe_slate(
    *,
    query: str,
    passages: Sequence[Passage],
    coordinates: CoordinateTable,
    edges: Sequence[TypedEdge],
) -> RecipeSlate:
    actions = materialize_recipe_actions(
        passages=passages, coordinates=coordinates, edges=edges
    )
    features = tuple(
        compute_action_features(
            query=query,
            passages=passages,
            coordinates=coordinates,
            edges=edges,
            action=action,
        )
        for action in actions
    )
    return RecipeSlate(actions=actions, features=features)


@dataclass(frozen=True)
class EvaluatorSelection:
    evaluator_id: str
    recipe_id: str
    registry_ordinal: int
    score: Fraction | float


def e0_score(features: ActionFeatures) -> Fraction:
    values = features.values
    return (
        Fraction(11, 20) * values[0]
        + Fraction(1, 5) * values[2]
        + Fraction(1, 10) * values[4]
        + Fraction(1, 10) * features.typed_closure
        + Fraction(1, 20) * values[6]
        - Fraction(1, 20) * values[7]
    )


def select_e0(slate: RecipeSlate) -> EvaluatorSelection:
    ranked = max(
        range(len(RECIPE_IDS)),
        key=lambda index: (
            e0_score(slate.features[index]),
            -slate.features[index].churn,
            -index,
        ),
    )
    return EvaluatorSelection(
        evaluator_id="E0_FIXED_GENERAL_COVERAGE",
        recipe_id=RECIPE_IDS[ranked],
        registry_ordinal=ranked,
        score=e0_score(slate.features[ranked]),
    )


def expanded_family_features(
    features: ActionFeatures, family: str
) -> tuple[float, ...]:
    if family not in QUERY_FAMILIES:
        raise MaudExtractionP2CoreError("query family is outside the registry")
    base = tuple(value / INTEGER_SCALE for value in features.values)
    interactions = tuple(
        value if current == family else 0.0
        for current in QUERY_FAMILIES
        for value in base
    )
    result = (*base, *interactions)
    if len(result) != E1_FEATURE_COUNT:
        raise MaudExtractionP2CoreError("family interaction width drifted")
    return result


@dataclass(frozen=True)
class AFormSlate:
    """One anonymous A_form item with all nine actions and utilities."""

    family: str
    slate: RecipeSlate
    recipe_utilities: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.family not in QUERY_FAMILIES
            or not isinstance(self.slate, RecipeSlate)
            or len(self.recipe_utilities) != len(RECIPE_IDS)
            or any(
                type(value) is not int
                or not 0 <= value <= INTEGER_SCALE
                for value in self.recipe_utilities
            )
        ):
            raise MaudExtractionP2CoreError("A_form slate is malformed")


@dataclass(frozen=True)
class E1RidgeModel:
    means: tuple[float, ...]
    population_standard_deviations: tuple[float, ...]
    weights: tuple[float, ...]
    zero_variance_columns: tuple[bool, ...]
    training_row_count: int
    model_sha256: str

    def __post_init__(self) -> None:
        vectors = (
            self.means,
            self.population_standard_deviations,
            self.weights,
            self.zero_variance_columns,
        )
        if any(len(row) != E1_FEATURE_COUNT for row in vectors):
            raise MaudExtractionP2CoreError("E1 ridge width drifted")
        if (
            any(not math.isfinite(value) for value in self.means)
            or any(
                not math.isfinite(value) or value < 0
                for value in self.population_standard_deviations
            )
            or any(not math.isfinite(value) for value in self.weights)
            or any(type(value) is not bool for value in self.zero_variance_columns)
            or type(self.training_row_count) is not int
            or self.training_row_count <= 0
        ):
            raise MaudExtractionP2CoreError("E1 ridge model is nonfinite")
        expected = _semantic_hash(
            {
                "identifier": "E1_AFORM_CENTERED_RIDGE_L2_1",
                "l2": 1.0,
                "means_hex": [value.hex() for value in self.means],
                "population_standard_deviations_hex": [
                    value.hex()
                    for value in self.population_standard_deviations
                ],
                "training_row_count": self.training_row_count,
                "weights_hex": [value.hex() for value in self.weights],
                "zero_variance_columns": list(self.zero_variance_columns),
            }
        )
        if self.model_sha256 != expected:
            raise MaudExtractionP2CoreError("E1 ridge identity drifted")

    def predict(self, features: ActionFeatures, family: str) -> float:
        raw = expanded_family_features(features, family)
        standardized = tuple(
            0.0
            if self.zero_variance_columns[index]
            else (
                raw[index] - self.means[index]
            )
            / self.population_standard_deviations[index]
            for index in range(E1_FEATURE_COUNT)
        )
        score = float(
            np.dot(
                np.asarray(standardized, dtype=np.float64),
                np.asarray(self.weights, dtype=np.float64),
            )
        )
        if not math.isfinite(score):
            raise MaudExtractionP2CoreError("E1 prediction is nonfinite")
        return score


def fit_e1_ridge(slates: Sequence[AFormSlate]) -> E1RidgeModel:
    """Fit the sole no-intercept, A_form-only, lambda-one ridge challenger."""

    if (
        isinstance(slates, (str, bytes))
        or not isinstance(slates, Sequence)
        or not slates
        or any(not isinstance(row, AFormSlate) for row in slates)
    ):
        raise MaudExtractionP2CoreError(
            "E1 fitting requires anonymous A_form slates"
        )
    features: list[tuple[float, ...]] = []
    targets: list[float] = []
    for item in slates:
        incumbent = select_e0(item.slate).registry_ordinal
        baseline = item.recipe_utilities[incumbent]
        for index in range(len(RECIPE_IDS)):
            features.append(
                expanded_family_features(
                    item.slate.features[index], item.family
                )
            )
            targets.append(
                (item.recipe_utilities[index] - baseline) / INTEGER_SCALE
            )
    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if (
        matrix.shape != (len(slates) * len(RECIPE_IDS), E1_FEATURE_COUNT)
        or target.shape != (matrix.shape[0],)
        or not np.isfinite(matrix).all()
        or not np.isfinite(target).all()
    ):
        raise MaudExtractionP2CoreError("E1 training matrix drifted")
    means = matrix.mean(axis=0)
    deviations = matrix.std(axis=0, ddof=0)
    zero = deviations == 0.0
    standardized = np.zeros_like(matrix)
    standardized[:, ~zero] = (
        matrix[:, ~zero] - means[~zero]
    ) / deviations[~zero]
    gram = standardized.T @ standardized
    system = gram + np.eye(E1_FEATURE_COUNT, dtype=np.float64) * RIDGE_L2
    right = standardized.T @ target
    try:
        weights = np.linalg.solve(system, right)
    except np.linalg.LinAlgError as exc:
        raise MaudExtractionP2CoreError("E1 ridge solve failed") from exc
    if not np.isfinite(weights).all():
        raise MaudExtractionP2CoreError("E1 ridge weights are nonfinite")
    body = {
        "identifier": "E1_AFORM_CENTERED_RIDGE_L2_1",
        "l2": 1.0,
        "means_hex": [float(value).hex() for value in means],
        "population_standard_deviations_hex": [
            float(value).hex() for value in deviations
        ],
        "training_row_count": int(matrix.shape[0]),
        "weights_hex": [float(value).hex() for value in weights],
        "zero_variance_columns": [bool(value) for value in zero],
    }
    return E1RidgeModel(
        means=tuple(float(value) for value in means),
        population_standard_deviations=tuple(
            float(value) for value in deviations
        ),
        weights=tuple(float(value) for value in weights),
        zero_variance_columns=tuple(bool(value) for value in zero),
        training_row_count=int(matrix.shape[0]),
        model_sha256=_semantic_hash(body),
    )


def select_e1(
    model: E1RidgeModel, slate: RecipeSlate, family: str
) -> EvaluatorSelection:
    if not isinstance(model, E1RidgeModel):
        raise MaudExtractionP2CoreError("E1 model is malformed")
    scores = tuple(
        model.predict(features, family) for features in slate.features
    )
    ranked = max(
        range(len(RECIPE_IDS)),
        key=lambda index: (
            scores[index],
            -slate.features[index].churn,
            -index,
        ),
    )
    return EvaluatorSelection(
        evaluator_id="E1_AFORM_CENTERED_RIDGE_L2_1",
        recipe_id=RECIPE_IDS[ranked],
        registry_ordinal=ranked,
        score=scores[ranked],
    )


@dataclass(frozen=True, order=True)
class CharacterInterval:
    start: int
    end: int

    def __post_init__(self) -> None:
        start = _strict_int(self.start, "interval start", minimum=0)
        end = _strict_int(self.end, "interval end", minimum=start + 1)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class GoldAnswer:
    answer_start: int
    text: str

    def __post_init__(self) -> None:
        start = _strict_int(self.answer_start, "answer start", minimum=0)
        if not isinstance(self.text, str) or not self.text:
            raise MaudExtractionP2CoreError("gold answer text must be nonempty")
        object.__setattr__(self, "answer_start", start)


def merge_intervals(
    intervals: Sequence[CharacterInterval],
) -> tuple[CharacterInterval, ...]:
    if isinstance(intervals, (str, bytes)) or not isinstance(
        intervals, Sequence
    ):
        raise MaudExtractionP2CoreError("interval registry is malformed")
    if any(not isinstance(row, CharacterInterval) for row in intervals):
        raise MaudExtractionP2CoreError("interval registry is malformed")
    ordered = sorted(set(intervals), key=lambda row: (row.start, row.end))
    merged: list[CharacterInterval] = []
    for row in ordered:
        if not merged or row.start > merged[-1].end:
            merged.append(row)
        else:
            previous = merged.pop()
            merged.append(
                CharacterInterval(previous.start, max(previous.end, row.end))
            )
    return tuple(merged)


def validate_gold_intervals(
    context: str, answers: Sequence[GoldAnswer]
) -> tuple[CharacterInterval, ...]:
    """Consume every raw answer and return deduplicated/merged exact intervals."""

    if (
        not isinstance(context, str)
        or not context
        or isinstance(answers, (str, bytes))
        or not isinstance(answers, Sequence)
        or any(not isinstance(row, GoldAnswer) for row in answers)
    ):
        raise MaudExtractionP2CoreError("gold answer registry is malformed")
    intervals: list[CharacterInterval] = []
    for answer in answers:
        end = answer.answer_start + len(answer.text)
        if (
            end > len(context)
            or context[answer.answer_start : end] != answer.text
        ):
            raise MaudExtractionP2CoreError(
                "gold answer does not exactly match the raw context"
            )
        intervals.append(CharacterInterval(answer.answer_start, end))
    return merge_intervals(intervals)


def _intersection_length(
    left: Sequence[CharacterInterval],
    right: Sequence[CharacterInterval],
) -> int:
    total = 0
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        first = left[left_index]
        second = right[right_index]
        total += max(0, min(first.end, second.end) - max(first.start, second.start))
        if first.end <= second.end:
            left_index += 1
        else:
            right_index += 1
    return total


@dataclass(frozen=True)
class CoverageScore:
    answerable: bool
    primary_utility: int | None
    complete_at_5: int | None
    coverage_at_least_half: int | None
    rank_discounted_incremental_utility: int | None
    merged_gold_length: int

    def __post_init__(self) -> None:
        if self.answerable:
            if (
                type(self.primary_utility) is not int
                or not 0 <= self.primary_utility <= INTEGER_SCALE
                or self.complete_at_5 not in (0, 1)
                or self.coverage_at_least_half not in (0, 1)
                or type(self.rank_discounted_incremental_utility) is not int
                or not 0
                <= self.rank_discounted_incremental_utility
                <= INTEGER_SCALE
                or self.merged_gold_length <= 0
            ):
                raise MaudExtractionP2CoreError(
                    "answerable coverage score is malformed"
                )
        elif (
            self.primary_utility is not None
            or self.complete_at_5 is not None
            or self.coverage_at_least_half is not None
            or self.rank_discounted_incremental_utility is not None
            or self.merged_gold_length != 0
        ):
            raise MaudExtractionP2CoreError(
                "unanswerable coverage score must be undefined"
            )


def score_evidence_coverage(
    *,
    passages: Sequence[Passage],
    selected_ordinals: Sequence[int],
    merged_gold_intervals: Sequence[CharacterInterval],
) -> CoverageScore:
    checked = _validated_passages(passages)
    if (
        isinstance(selected_ordinals, (str, bytes))
        or not isinstance(selected_ordinals, Sequence)
        or len(selected_ordinals) != TOP_K
        or len(set(selected_ordinals)) != TOP_K
        or any(
            type(value) is not int or not 0 <= value < len(checked)
            for value in selected_ordinals
        )
    ):
        raise MaudExtractionP2CoreError("selected top five is malformed")
    gold = merge_intervals(merged_gold_intervals)
    if not gold:
        return CoverageScore(False, None, None, None, None, 0)
    selected_intervals = merge_intervals(
        [
            CharacterInterval(checked[ordinal].start, checked[ordinal].end)
            for ordinal in selected_ordinals
        ]
    )
    total_gold = sum(row.length for row in gold)
    covered = _intersection_length(selected_intervals, gold)
    primary = (INTEGER_SCALE * covered) // total_gold
    already: tuple[CharacterInterval, ...] = ()
    discounted = 0.0
    for rank, ordinal in enumerate(selected_ordinals):
        passage_interval = CharacterInterval(
            checked[ordinal].start, checked[ordinal].end
        )
        before = _intersection_length(already, gold)
        already = merge_intervals((*already, passage_interval))
        after = _intersection_length(already, gold)
        discounted += (after - before) / math.log2(rank + 2)
    discounted_utility = min(
        INTEGER_SCALE,
        int(math.floor(INTEGER_SCALE * discounted / total_gold)),
    )
    return CoverageScore(
        answerable=True,
        primary_utility=primary,
        complete_at_5=int(covered == total_gold),
        coverage_at_least_half=int(covered * 2 >= total_gold),
        rank_discounted_incremental_utility=discounted_utility,
        merged_gold_length=total_gold,
    )


@dataclass(frozen=True)
class ClusterItem:
    """One query's public family and arm utilities within one contract."""

    family: str
    arm_utilities: Mapping[str, int | None]

    def __post_init__(self) -> None:
        if (
            self.family not in QUERY_FAMILIES
            or not isinstance(self.arm_utilities, Mapping)
            or not self.arm_utilities
            or any(
                not isinstance(arm, str)
                or not arm
                or (
                    value is not None
                    and (
                        type(value) is not int
                        or not 0 <= value <= INTEGER_SCALE
                    )
                )
                for arm, value in self.arm_utilities.items()
            )
        ):
            raise MaudExtractionP2CoreError("cluster item is malformed")
        if len({value is None for value in self.arm_utilities.values()}) != 1:
            raise MaudExtractionP2CoreError(
                "cluster item arm answerability drifted"
            )
        object.__setattr__(self, "arm_utilities", dict(self.arm_utilities))


@dataclass(frozen=True)
class ContractCluster:
    items: tuple[ClusterItem, ...]

    def __post_init__(self) -> None:
        if (
            not self.items
            or any(not isinstance(row, ClusterItem) for row in self.items)
        ):
            raise MaudExtractionP2CoreError("contract cluster is empty")
        arm_sets = {tuple(sorted(row.arm_utilities)) for row in self.items}
        if len(arm_sets) != 1:
            raise MaudExtractionP2CoreError(
                "contract cluster arm registry drifted"
            )

    @property
    def arms(self) -> tuple[str, ...]:
        return tuple(sorted(self.items[0].arm_utilities))

    def family_means(self, arm: str) -> dict[str, Fraction]:
        if arm not in self.arms:
            raise MaudExtractionP2CoreError("unknown cluster arm")
        result: dict[str, Fraction] = {}
        for family in QUERY_FAMILIES:
            values = [
                row.arm_utilities[arm]
                for row in self.items
                if row.family == family
                and row.arm_utilities[arm] is not None
            ]
            if values:
                result[family] = Fraction(
                    sum(int(value) for value in values), len(values)
                )
        return result

    def utility(self, arm: str) -> Fraction | None:
        families = self.family_means(arm)
        if not families:
            return None
        return sum(families.values(), Fraction(0)) / len(families)


@dataclass(frozen=True)
class ExactSignFlipResult:
    observed_net: Fraction
    nonzero_contract_count: int
    reference_tail: Fraction

    @property
    def positive(self) -> bool:
        return self.observed_net > 0

    @property
    def at_or_below_alpha(self) -> bool:
        return self.reference_tail <= PROMOTION_ALPHA


def exact_contract_sign_flip(
    deltas: Sequence[Fraction | int],
) -> ExactSignFlipResult:
    """Complete one-sided magnitude-preserving contract sign enumeration."""

    if (
        isinstance(deltas, (str, bytes))
        or not isinstance(deltas, Sequence)
        or not deltas
    ):
        raise MaudExtractionP2CoreError(
            "paired contract delta vector is empty"
        )
    normalized: list[Fraction] = []
    for value in deltas:
        if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
            raise MaudExtractionP2CoreError(
                "paired contract delta must be exact"
            )
        normalized.append(Fraction(value))
    common_denominator = 1
    for value in normalized:
        common_denominator = math.lcm(
            common_denominator, value.denominator
        )
    integers = tuple(
        value.numerator * (common_denominator // value.denominator)
        for value in normalized
    )
    observed = sum(integers)
    magnitudes = tuple(abs(value) for value in integers if value)
    distribution: Counter[int] = Counter({0: 1})
    for magnitude in magnitudes:
        updated: Counter[int] = Counter()
        for subtotal, count in distribution.items():
            updated[subtotal + magnitude] += count
            updated[subtotal - magnitude] += count
        distribution = updated
    tail = Fraction(
        sum(
            count
            for subtotal, count in distribution.items()
            if subtotal >= observed
        ),
        1 << len(magnitudes),
    )
    return ExactSignFlipResult(
        observed_net=sum(normalized, Fraction(0)),
        nonzero_contract_count=len(magnitudes),
        reference_tail=tail,
    )


@dataclass(frozen=True)
class ContractClusterComparison:
    left_arm: str
    right_arm: str
    paired_contract_deltas: tuple[Fraction, ...]
    equal_weight_contract_mean_delta: Fraction
    family_deltas: Mapping[str, Fraction]
    zero_answerable_contract_count: int
    sign_flip: ExactSignFlipResult

    @property
    def promoted(self) -> bool:
        return (
            self.equal_weight_contract_mean_delta > 0
            and self.sign_flip.reference_tail <= PROMOTION_ALPHA
        )


def compare_contract_clusters(
    clusters: Sequence[ContractCluster],
    *,
    left_arm: str,
    right_arm: str,
) -> ContractClusterComparison:
    """Compare arms after within-family then equal-contract aggregation."""

    if (
        isinstance(clusters, (str, bytes))
        or not isinstance(clusters, Sequence)
        or not clusters
        or any(not isinstance(row, ContractCluster) for row in clusters)
        or not isinstance(left_arm, str)
        or not isinstance(right_arm, str)
        or left_arm == right_arm
    ):
        raise MaudExtractionP2CoreError(
            "contract cluster comparison is malformed"
        )
    deltas: list[Fraction] = []
    zero_answerable = 0
    family_rows: dict[str, list[Fraction]] = {
        family: [] for family in QUERY_FAMILIES
    }
    for cluster in clusters:
        left = cluster.utility(left_arm)
        right = cluster.utility(right_arm)
        if left is None or right is None:
            if left is not None or right is not None:
                raise MaudExtractionP2CoreError(
                    "paired arms disagree on answerability"
                )
            zero_answerable += 1
            continue
        deltas.append(left - right)
        left_families = cluster.family_means(left_arm)
        right_families = cluster.family_means(right_arm)
        if set(left_families) != set(right_families):
            raise MaudExtractionP2CoreError(
                "paired arms disagree on answerable families"
            )
        for family in left_families:
            family_rows[family].append(
                left_families[family] - right_families[family]
            )
    if not deltas:
        raise MaudExtractionP2CoreError(
            "comparison has no answerable paired contract"
        )
    family_deltas = {
        family: sum(values, Fraction(0)) / len(values)
        for family, values in family_rows.items()
        if values
    }
    return ContractClusterComparison(
        left_arm=left_arm,
        right_arm=right_arm,
        paired_contract_deltas=tuple(deltas),
        equal_weight_contract_mean_delta=(
            sum(deltas, Fraction(0)) / len(deltas)
        ),
        family_deltas=family_deltas,
        zero_answerable_contract_count=zero_answerable,
        sign_flip=exact_contract_sign_flip(deltas),
    )


__all__ = [
    "AFormSlate",
    "ActionFeatures",
    "BASE_FEATURE_COUNT",
    "BM25_B",
    "BM25_K1",
    "CharacterInterval",
    "ClusterItem",
    "CONDITION_OBLIGATION",
    "ContractCluster",
    "ContractClusterComparison",
    "CoordinateTable",
    "CoverageScore",
    "DEFINITION_REFERENCE",
    "EDGE_FAMILIES",
    "E1_FEATURE_COUNT",
    "E1RidgeModel",
    "EvaluatorSelection",
    "EXCEPTION_REMEDY",
    "ExactSignFlipResult",
    "FAMILY_CONDITION_OBLIGATION",
    "FAMILY_DEFINITION_REFERENCE",
    "FAMILY_PROTECTION_EXCEPTION_REMEDY",
    "FEATURE_ORDER",
    "GoldAnswer",
    "HARD_MAXIMUM_CODE_POINTS",
    "INTEGER_SCALE",
    "MINIMUM_PREFERRED_CODE_POINTS",
    "MaudExtractionP2CoreError",
    "OVERLAP_TARGET_CODE_POINTS",
    "Passage",
    "PROMOTION_ALPHA",
    "QUERY_FAMILIES",
    "RECIPE_IDS",
    "RECIPE_REGISTRY",
    "RecipeAction",
    "RecipeSlate",
    "SECTION_XREF",
    "STUDY_ID",
    "SectionHeading",
    "TARGET_CODE_POINTS",
    "TOP_K",
    "TypedEdge",
    "VERSION",
    "build_coordinate_table",
    "build_coordinate_table_from_quantized",
    "build_passages",
    "build_recipe_slate",
    "build_typed_edges",
    "bm25_scores",
    "compare_contract_clusters",
    "compute_action_features",
    "e0_score",
    "exact_contract_sign_flip",
    "expanded_family_features",
    "fit_e1_ridge",
    "lexical_tokens",
    "materialize_recipe_actions",
    "merge_intervals",
    "normalize_section_reference",
    "normalized_bm25_coordinates",
    "quantize_half_even",
    "score_evidence_coverage",
    "select_e0",
    "select_e1",
    "serialized_passage_corpus",
    "validate_gold_intervals",
]
