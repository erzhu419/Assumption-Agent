"""Pure, source-agnostic typed document retrieval core for TechQA P1.

Only the public question title, public question text, and the exact public
``ordinal/title/text`` projection of each candidate document may enter this
module.  It has no source loader, file, network, model, answerability, family,
answer-span, relevance-label, split, cluster, or document-identity entrypoint.

RAW is deterministic BM25 over the complete, unchanged
``question_title + "\\n" + question_text`` string and the identical serialized
``document.title + "\\n" + document.text`` bytes used by every recipe.  Five
additional label-free recipes derive typed coordinates from those same public
bytes.  E0 is the frozen typed cascade.  E1 is a conservative pairwise
structural-signature table fitted only from exact A_form utilities; unseen,
under-supported, tied, or contradicted signatures fall back to E0.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from fractions import Fraction
import hashlib
import json
import math
import re
import unicodedata
from typing import Mapping, Sequence


VERSION = "techqa_p1_typed_core_v1"
STUDY_ID = "TECHQA_P1_TYPED_DOCUMENT_RETRIEVAL_V1"
TOP_K = 5
SCALE = 1_000_000
BM25_K1 = 1.2
BM25_B = 0.75
MIN_SIGNATURE_SUPPORT = 3
SERIALIZATION_SEPARATOR = "\n"

R0_RAW_BM25 = "R0_RAW_BM25"
R1_TITLE_FOCUSED = "R1_TITLE_FOCUSED"
R2_LITERAL_SIGNATURE_ANCHOR = "R2_LITERAL_SIGNATURE_ANCHOR"
R3_FIELD_AWARE_COVERAGE = "R3_FIELD_AWARE_COVERAGE"
R4_MULTI_SEED_MARGINAL = "R4_MULTI_SEED_MARGINAL"
R5_TYPED_CASCADE = "R5_TYPED_CASCADE"
RECIPE_IDS = (
    R0_RAW_BM25,
    R1_TITLE_FOCUSED,
    R2_LITERAL_SIGNATURE_ANCHOR,
    R3_FIELD_AWARE_COVERAGE,
    R4_MULTI_SEED_MARGINAL,
    R5_TYPED_CASCADE,
)
E0_RECIPE_ID = R5_TYPED_CASCADE
POLICY_STAGES = ("F_search", "A_hold", "M_search")
PUBLIC_DOCUMENT_FIELDS = ("ordinal", "title", "text")

FEATURE_NAMES = (
    "raw_overlap_count",
    "title_match_document_count",
    "literal_document_count",
    "error_document_count",
    "version_document_count",
    "mean_title_coverage_micros",
    "mean_body_coverage_micros",
    "mean_field_coverage_micros",
    "mean_raw_bm25_micros",
    "minimum_raw_bm25_micros",
    "mean_selected_score_micros",
    "lexical_diversity_micros",
)
SIGNATURE_FEATURE_NAMES = FEATURE_NAMES

_TOKEN_RE = re.compile(
    r"[^\W_]+(?:[._:/+#-][^\W_]+)*",
    re.UNICODE,
)
_VERSION_RE = re.compile(
    r"(?:v(?:ersion)?[-_ ]*)?\d+(?:\.\d+){1,4}"
    r"(?:[-_.]?(?:alpha|beta|rc|sp|p)\d*)?\Z",
    re.IGNORECASE,
)
_QUOTED_RE = re.compile(r"[\"'`](.{2,120}?)[\"'`]", re.DOTALL)
_ERROR_PREFIXES = (
    "code",
    "cve",
    "e",
    "err",
    "errno",
    "error",
    "http",
    "ora",
    "sql",
    "sqlstate",
    "status",
)
_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "may",
        "my",
        "of",
        "on",
        "or",
        "should",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)


class TechqaP1TypedCoreError(ValueError):
    """A public projection, frozen action, or A_form evaluator drifted."""


def canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise TechqaP1TypedCoreError("value is not canonical JSON") from exc
    return encoded + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _checked_text(
    value: object,
    *,
    field: str,
    maximum_length: int,
    allow_empty: bool = False,
) -> str:
    if (
        not isinstance(value, str)
        or "\x00" in value
        or len(value) > maximum_length
        or (not allow_empty and not value.strip())
    ):
        raise TechqaP1TypedCoreError(f"{field} is invalid")
    return value


def normalize_text(
    value: str,
    *,
    field: str = "text",
    allow_empty: bool = False,
) -> str:
    _checked_text(
        value,
        field=field,
        maximum_length=200_000,
        allow_empty=allow_empty,
    )
    normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    if not allow_empty and not normalized:
        raise TechqaP1TypedCoreError(f"{field} tokenization is empty")
    return normalized


def lexical_tokens(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(normalize_text(value)))


def serialize_query_text(question_title: str, question_text: str) -> str:
    """Return the frozen, unchanged title/body query concatenation."""

    _checked_text(
        question_title,
        field="question_title",
        maximum_length=20_000,
    )
    _checked_text(
        question_text,
        field="question_text",
        maximum_length=100_000,
        allow_empty=True,
    )
    combined = question_title + SERIALIZATION_SEPARATOR + question_text
    if not _TOKEN_RE.search(normalize_text(combined)):
        raise TechqaP1TypedCoreError("serialized question has no lexical token")
    return combined


@dataclass(frozen=True, slots=True)
class Document:
    ordinal: int
    title: str
    text: str

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise TechqaP1TypedCoreError("document ordinal is invalid")
        _checked_text(self.title, field="document title", maximum_length=20_000)
        _checked_text(self.text, field="document text", maximum_length=200_000)
        lexical_tokens(self.title + SERIALIZATION_SEPARATOR + self.text)


def document_from_public_fields(value: object) -> Document:
    if (
        not isinstance(value, Mapping)
        or set(value) != set(PUBLIC_DOCUMENT_FIELDS)
    ):
        raise TechqaP1TypedCoreError(
            "document projection is not the exact public field set"
        )
    try:
        return Document(
            ordinal=value["ordinal"],  # type: ignore[arg-type]
            title=value["title"],  # type: ignore[arg-type]
            text=value["text"],  # type: ignore[arg-type]
        )
    except KeyError as exc:
        raise TechqaP1TypedCoreError("document projection is incomplete") from exc


def document_public_payload(document: Document) -> dict[str, object]:
    if not isinstance(document, Document):
        raise TechqaP1TypedCoreError("document is not a public Document")
    return {
        field.name: getattr(document, field.name)
        for field in fields(Document)
    }


def serialize_document_text(document: Document) -> str:
    """Return the exact text whose UTF-8 bytes are shared by all recipes."""

    if not isinstance(document, Document):
        raise TechqaP1TypedCoreError("document is not a public Document")
    return document.title + SERIALIZATION_SEPARATOR + document.text


def serialize_document_bytes(document: Document) -> bytes:
    return serialize_document_text(document).encode("utf-8")


def _checked_documents(
    documents: Sequence[Document],
) -> tuple[Document, ...]:
    if isinstance(documents, (str, bytes)) or len(documents) < TOP_K:
        raise TechqaP1TypedCoreError("corpus has fewer than five documents")
    checked = tuple(documents)
    if any(not isinstance(document, Document) for document in checked):
        raise TechqaP1TypedCoreError("corpus contains a non-Document value")
    if len({document.ordinal for document in checked}) != len(checked):
        raise TechqaP1TypedCoreError("document ordinals are not unique")
    return tuple(sorted(checked, key=lambda document: document.ordinal))


def bm25_scores(
    query_text: str,
    document_texts: Sequence[str],
) -> tuple[int, ...]:
    """Return quantized Okapi BM25 over the complete supplied strings."""

    query_terms = lexical_tokens(query_text)
    if not query_terms:
        raise TechqaP1TypedCoreError("BM25 query tokenization is empty")
    if (
        isinstance(document_texts, (str, bytes))
        or len(document_texts) < TOP_K
        or any(not isinstance(text, str) for text in document_texts)
    ):
        raise TechqaP1TypedCoreError("BM25 corpus is malformed")
    documents = [lexical_tokens(text) for text in document_texts]
    if any(not terms for terms in documents):
        raise TechqaP1TypedCoreError("BM25 document tokenization is empty")
    document_count = len(documents)
    average_length = sum(map(len, documents)) / document_count
    document_frequency: Counter[str] = Counter()
    for terms in documents:
        document_frequency.update(set(terms))
    query_frequency = Counter(query_terms)
    scores: list[int] = []
    for terms in documents:
        term_frequency = Counter(terms)
        score = 0.0
        for term, query_count in query_frequency.items():
            frequency = term_frequency.get(term, 0)
            if frequency == 0:
                continue
            df = document_frequency[term]
            inverse_document_frequency = math.log(
                1.0 + (document_count - df + 0.5) / (df + 0.5)
            )
            denominator = frequency + BM25_K1 * (
                1.0 - BM25_B + BM25_B * len(terms) / average_length
            )
            score += (
                query_count
                * inverse_document_frequency
                * frequency
                * (BM25_K1 + 1.0)
                / denominator
            )
        scores.append(int(round(score * SCALE)))
    return tuple(scores)


def _content_tokens(value: str) -> frozenset[str]:
    normalized = normalize_text(value, allow_empty=True)
    return frozenset(
        token
        for token in _TOKEN_RE.findall(normalized)
        if token not in _STOPWORDS and len(token) >= 2
    )


def _canonical_technical_token(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().strip(".,;:!?()[]{}")


def _looks_like_version(token: str) -> bool:
    return _VERSION_RE.fullmatch(token) is not None


def _looks_like_error(token: str, previous: str | None = None) -> bool:
    compact = token.replace("-", "").replace("_", "").replace(".", "")
    if not any(character.isdigit() for character in compact):
        return False
    if previous in _ERROR_PREFIXES:
        return True
    return any(
        compact.startswith(prefix)
        and len(compact) > len(prefix)
        for prefix in _ERROR_PREFIXES
    )


def _technical_anchors(
    value: str,
) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    normalized = normalize_text(value, allow_empty=True)
    raw_tokens = [
        _canonical_technical_token(token)
        for token in _TOKEN_RE.findall(normalized)
    ]
    raw_tokens = [token for token in raw_tokens if token]
    versions: set[str] = set()
    errors: set[str] = set()
    literals: set[str] = set()
    previous: str | None = None
    for token in raw_tokens:
        if _looks_like_version(token):
            versions.add(token)
        if _looks_like_error(token, previous):
            errors.add(token)
        if (
            any(character.isdigit() for character in token)
            or any(character in "._:/+#-" for character in token)
        ):
            literals.add(token)
        previous = token
    for quoted in _QUOTED_RE.findall(normalized):
        canonical = normalize_text(quoted, allow_empty=True)
        if canonical:
            literals.add(canonical)
    literals.update(versions)
    literals.update(errors)
    return frozenset(literals), frozenset(errors), frozenset(versions)


@dataclass(frozen=True, slots=True)
class QueryStructure:
    title_anchors: tuple[str, ...]
    body_anchors: tuple[str, ...]
    literal_anchors: tuple[str, ...]
    error_anchors: tuple[str, ...]
    version_anchors: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "title_anchors",
            "body_anchors",
            "literal_anchors",
            "error_anchors",
            "version_anchors",
        ):
            values = getattr(self, field_name)
            if values != tuple(sorted(set(values))):
                raise TechqaP1TypedCoreError(
                    f"{field_name} is not a sorted set"
                )

    def payload(self) -> dict[str, object]:
        return {
            "body_anchors": list(self.body_anchors),
            "error_anchors": list(self.error_anchors),
            "literal_anchors": list(self.literal_anchors),
            "title_anchors": list(self.title_anchors),
            "version_anchors": list(self.version_anchors),
        }


def query_structure(
    question_title: str,
    question_text: str,
) -> QueryStructure:
    raw_query = serialize_query_text(question_title, question_text)
    literal, errors, versions = _technical_anchors(raw_query)
    return QueryStructure(
        title_anchors=tuple(sorted(_content_tokens(question_title))),
        body_anchors=tuple(sorted(_content_tokens(question_text))),
        literal_anchors=tuple(sorted(literal)),
        error_anchors=tuple(sorted(errors)),
        version_anchors=tuple(sorted(versions)),
    )


@dataclass(frozen=True, slots=True)
class DocumentStructure:
    document_ordinal: int
    serialized_sha256: str
    raw_bm25: int
    title_bm25: int
    body_bm25: int
    title_hits: int
    body_hits: int
    literal_hits: int
    error_hits: int
    version_hits: int
    title_coverage: int
    body_coverage: int
    field_coverage: int

    def __post_init__(self) -> None:
        if type(self.document_ordinal) is not int or self.document_ordinal < 0:
            raise TechqaP1TypedCoreError("document feature ordinal is invalid")
        if re.fullmatch(r"[0-9a-f]{64}", self.serialized_sha256) is None:
            raise TechqaP1TypedCoreError("serialized document hash is invalid")
        for field_name in (
            "raw_bm25",
            "title_bm25",
            "body_bm25",
            "title_hits",
            "body_hits",
            "literal_hits",
            "error_hits",
            "version_hits",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise TechqaP1TypedCoreError(f"{field_name} is invalid")
        for field_name in (
            "title_coverage",
            "body_coverage",
            "field_coverage",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or not 0 <= value <= SCALE:
                raise TechqaP1TypedCoreError(f"{field_name} is invalid")

    def payload(self) -> dict[str, object]:
        return {
            "body_bm25": self.body_bm25,
            "body_coverage": self.body_coverage,
            "body_hits": self.body_hits,
            "document_ordinal": self.document_ordinal,
            "error_hits": self.error_hits,
            "field_coverage": self.field_coverage,
            "literal_hits": self.literal_hits,
            "raw_bm25": self.raw_bm25,
            "serialized_sha256": self.serialized_sha256,
            "title_bm25": self.title_bm25,
            "title_coverage": self.title_coverage,
            "title_hits": self.title_hits,
            "version_hits": self.version_hits,
        }


def _coverage(hits: int, anchor_count: int) -> int:
    if anchor_count == 0:
        return 0
    return min(SCALE, hits * SCALE // anchor_count)


def _document_structures(
    *,
    question_title: str,
    question_text: str,
    documents: Sequence[Document],
    structure: QueryStructure,
) -> tuple[DocumentStructure, ...]:
    raw_query = serialize_query_text(question_title, question_text)
    serialized_texts = tuple(serialize_document_text(document) for document in documents)
    raw_scores = bm25_scores(raw_query, serialized_texts)
    title_scores = bm25_scores(raw_query, [document.title for document in documents])
    body_scores = bm25_scores(raw_query, [document.text for document in documents])
    title_anchors = set(structure.title_anchors)
    body_anchors = set(structure.body_anchors)
    all_field_anchors = title_anchors | body_anchors
    literal_anchors = set(structure.literal_anchors)
    error_anchors = set(structure.error_anchors)
    version_anchors = set(structure.version_anchors)
    rows: list[DocumentStructure] = []
    for index, document in enumerate(documents):
        title_tokens = set(lexical_tokens(document.title))
        body_tokens = set(lexical_tokens(document.text))
        serialized = serialized_texts[index]
        document_literals, document_errors, document_versions = (
            _technical_anchors(serialized)
        )
        title_hits = len(title_anchors & title_tokens)
        body_hits = len(body_anchors & body_tokens)
        field_hits = len(
            all_field_anchors & (title_tokens | body_tokens)
        )
        rows.append(
            DocumentStructure(
                document_ordinal=document.ordinal,
                serialized_sha256=hashlib.sha256(
                    serialize_document_bytes(document)
                ).hexdigest(),
                raw_bm25=raw_scores[index],
                title_bm25=title_scores[index],
                body_bm25=body_scores[index],
                title_hits=title_hits,
                body_hits=body_hits,
                literal_hits=len(literal_anchors & set(document_literals)),
                error_hits=len(error_anchors & set(document_errors)),
                version_hits=len(version_anchors & set(document_versions)),
                title_coverage=_coverage(title_hits, len(title_anchors)),
                body_coverage=_coverage(body_hits, len(body_anchors)),
                field_coverage=_coverage(field_hits, len(all_field_anchors)),
            )
        )
    return tuple(rows)


def _rank(
    scores: Sequence[int],
    documents: Sequence[Document],
) -> tuple[int, ...]:
    if len(scores) != len(documents):
        raise TechqaP1TypedCoreError("score vector width drifted")
    if any(type(score) is not int or score < 0 for score in scores):
        raise TechqaP1TypedCoreError("score vector is invalid")
    return tuple(
        sorted(
            range(len(documents)),
            key=lambda index: (-scores[index], documents[index].ordinal),
        )
    )


def _jaccard_distance(left: set[str], right: set[str]) -> int:
    union = left | right
    if not union:
        return 0
    return len(union - (left & right)) * SCALE // len(union)


def _multi_seed_selection(
    *,
    documents: Sequence[Document],
    raw_scores: Sequence[int],
    title_scores: Sequence[int],
    literal_scores: Sequence[int],
    field_scores: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    selected: list[int] = []
    selected_scores: list[int] = []
    for scores in (title_scores, literal_scores, field_scores, raw_scores):
        candidate = _rank(scores, documents)[0]
        if candidate not in selected:
            selected.append(candidate)
            selected_scores.append(scores[candidate])
        if len(selected) == TOP_K:
            return tuple(selected), tuple(selected_scores)

    token_sets = [
        set(lexical_tokens(serialize_document_text(document)))
        for document in documents
    ]
    while len(selected) < TOP_K:
        candidates: list[tuple[int, int, int]] = []
        for index, document in enumerate(documents):
            if index in selected:
                continue
            novelty = min(
                _jaccard_distance(token_sets[index], token_sets[other])
                for other in selected
            )
            base = max(
                raw_scores[index],
                title_scores[index],
                literal_scores[index],
                field_scores[index],
            )
            marginal = (
                base
                + raw_scores[index]
                + novelty
            )
            candidates.append((-marginal, document.ordinal, index))
        negative_score, _ordinal, chosen = min(candidates)
        selected.append(chosen)
        selected_scores.append(-negative_score)
    return tuple(selected), tuple(selected_scores)


def _typed_cascade_selection(
    *,
    documents: Sequence[Document],
    raw_scores: Sequence[int],
    title_scores: Sequence[int],
    literal_scores: Sequence[int],
    field_scores: Sequence[int],
    composite_scores: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    selected: list[int] = []
    selected_scores: list[int] = []

    def add(index: int, score: int) -> None:
        if index not in selected and len(selected) < TOP_K:
            selected.append(index)
            selected_scores.append(score)

    for scores in (
        composite_scores,
        literal_scores,
        title_scores,
        field_scores,
        raw_scores,
    ):
        for index in _rank(scores, documents):
            if index not in selected:
                add(index, scores[index])
                break
    for index in _rank(composite_scores, documents):
        add(index, composite_scores[index])
    if len(selected) != TOP_K:
        raise TechqaP1TypedCoreError("typed cascade did not totalize top5")
    return tuple(selected), tuple(selected_scores)


@dataclass(frozen=True, slots=True)
class RecipeAction:
    recipe_id: str
    top5_document_ordinals: tuple[int, ...]
    selected_scores: tuple[int, ...]
    raw_top5_document_ordinals: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.recipe_id not in RECIPE_IDS:
            raise TechqaP1TypedCoreError("recipe is not frozen")
        if (
            len(self.top5_document_ordinals) != TOP_K
            or len(set(self.top5_document_ordinals)) != TOP_K
            or any(
                type(value) is not int or value < 0
                for value in self.top5_document_ordinals
            )
        ):
            raise TechqaP1TypedCoreError("recipe top5 is malformed")
        if (
            len(self.selected_scores) != TOP_K
            or any(
                type(value) is not int or value < 0
                for value in self.selected_scores
            )
        ):
            raise TechqaP1TypedCoreError("recipe scores are malformed")
        if (
            len(self.raw_top5_document_ordinals) != TOP_K
            or len(set(self.raw_top5_document_ordinals)) != TOP_K
        ):
            raise TechqaP1TypedCoreError("RAW top5 is malformed")

    def payload(self) -> dict[str, object]:
        return {
            "raw_top5_document_ordinals": list(
                self.raw_top5_document_ordinals
            ),
            "recipe_id": self.recipe_id,
            "selected_scores": list(self.selected_scores),
            "top5_document_ordinals": list(self.top5_document_ordinals),
        }


@dataclass(frozen=True, slots=True)
class ActionFeatures:
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(FEATURE_NAMES) or any(
            type(value) is not int or not 0 <= value <= SCALE
            for value in self.values
        ):
            raise TechqaP1TypedCoreError("action structural features drifted")

    def payload(self) -> dict[str, int]:
        return dict(zip(FEATURE_NAMES, self.values, strict=True))


def _action_features(
    *,
    action: RecipeAction,
    documents: Sequence[Document],
    structures: Sequence[DocumentStructure],
    raw_maximum: int,
    selected_score_maximum: int,
) -> ActionFeatures:
    by_ordinal = {
        document.ordinal: index
        for index, document in enumerate(documents)
    }
    indices = tuple(
        by_ordinal[ordinal]
        for ordinal in action.top5_document_ordinals
    )
    raw_values = [structures[index].raw_bm25 for index in indices]
    raw_normalizer = raw_maximum or 1
    normalized_raw = [
        min(SCALE, value * SCALE // raw_normalizer)
        for value in raw_values
    ]
    score_normalizer = selected_score_maximum or 1
    normalized_selected = [
        min(SCALE, score * SCALE // score_normalizer)
        for score in action.selected_scores
    ]
    token_sets = [
        set(lexical_tokens(serialize_document_text(documents[index])))
        for index in indices
    ]
    distances = [
        _jaccard_distance(token_sets[left], token_sets[right])
        for left in range(TOP_K)
        for right in range(left + 1, TOP_K)
    ]
    values = (
        len(
            set(action.top5_document_ordinals)
            & set(action.raw_top5_document_ordinals)
        ),
        sum(structures[index].title_hits > 0 for index in indices),
        sum(structures[index].literal_hits > 0 for index in indices),
        sum(structures[index].error_hits > 0 for index in indices),
        sum(structures[index].version_hits > 0 for index in indices),
        sum(structures[index].title_coverage for index in indices) // TOP_K,
        sum(structures[index].body_coverage for index in indices) // TOP_K,
        sum(structures[index].field_coverage for index in indices) // TOP_K,
        sum(normalized_raw) // TOP_K,
        min(normalized_raw),
        sum(normalized_selected) // TOP_K,
        sum(distances) // len(distances),
    )
    return ActionFeatures(values)


@dataclass(frozen=True, slots=True)
class ActionSlate:
    question_title_sha256: str
    question_text_sha256: str
    raw_query_bytes_sha256: str
    public_document_projection_sha256: str
    serialized_document_set_sha256: str
    query: QueryStructure
    document_structures: tuple[DocumentStructure, ...]
    actions: tuple[RecipeAction, ...]
    features: tuple[ActionFeatures, ...]

    def __post_init__(self) -> None:
        hashes = (
            self.question_title_sha256,
            self.question_text_sha256,
            self.raw_query_bytes_sha256,
            self.public_document_projection_sha256,
            self.serialized_document_set_sha256,
        )
        action_ordinals = {
            ordinal
            for action in self.actions
            for ordinal in action.top5_document_ordinals
        }
        structure_ordinals = {
            row.document_ordinal
            for row in self.document_structures
        }
        if (
            any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes)
            or tuple(action.recipe_id for action in self.actions) != RECIPE_IDS
            or len(self.features) != len(RECIPE_IDS)
            or len(structure_ordinals) != len(self.document_structures)
            or not action_ordinals <= structure_ordinals
        ):
            raise TechqaP1TypedCoreError("action slate drifted")

    def action(self, recipe_id: str) -> RecipeAction:
        if recipe_id not in RECIPE_IDS:
            raise TechqaP1TypedCoreError("requested recipe is not frozen")
        return self.actions[RECIPE_IDS.index(recipe_id)]

    def feature_row(self, recipe_id: str) -> ActionFeatures:
        if recipe_id not in RECIPE_IDS:
            raise TechqaP1TypedCoreError("requested recipe is not frozen")
        return self.features[RECIPE_IDS.index(recipe_id)]

    def audit_payload(self) -> dict[str, object]:
        body = {
            "actions": [action.payload() for action in self.actions],
            "document_structures": [
                row.payload() for row in self.document_structures
            ],
            "feature_names": list(FEATURE_NAMES),
            "features": [feature.payload() for feature in self.features],
            "forbidden_action_inputs": [
                "answer_span",
                "answerable",
                "cluster_identity",
                "document_identity",
                "family",
                "qrel",
                "source_identity",
                "stage_identity",
            ],
            "public_document_fields": list(PUBLIC_DOCUMENT_FIELDS),
            "public_document_projection_sha256": (
                self.public_document_projection_sha256
            ),
            "query_structure": self.query.payload(),
            "question_text_sha256": self.question_text_sha256,
            "question_title_sha256": self.question_title_sha256,
            "raw_query_bytes_sha256": self.raw_query_bytes_sha256,
            "recipe_ids": list(RECIPE_IDS),
            "schema": f"{VERSION}_action_slate",
            "serialization_separator": SERIALIZATION_SEPARATOR,
            "serialized_document_set_sha256": (
                self.serialized_document_set_sha256
            ),
        }
        body["self_sha256"] = stable_hash(body)
        return body


def build_action_slate(
    question_title: str,
    question_text: str,
    documents: Sequence[Document],
) -> ActionSlate:
    """Materialize all six label-free actions from the strict public view."""

    raw_query = serialize_query_text(question_title, question_text)
    checked_documents = _checked_documents(documents)
    structure = query_structure(question_title, question_text)
    rows = _document_structures(
        question_title=question_title,
        question_text=question_text,
        documents=checked_documents,
        structure=structure,
    )
    raw_scores = tuple(row.raw_bm25 for row in rows)
    title_scores = tuple(
        row.raw_bm25
        + 2 * row.title_bm25
        + 8 * row.title_coverage
        for row in rows
    )
    literal_scores = tuple(
        row.raw_bm25
        + row.title_bm25
        + 4 * row.title_coverage
        + 10_000_000 * row.literal_hits
        + 15_000_000 * row.error_hits
        + 12_000_000 * row.version_hits
        for row in rows
    )
    field_scores = tuple(
        row.raw_bm25
        + row.title_bm25
        + row.body_bm25
        + 8 * row.title_coverage
        + 5 * row.body_coverage
        + 5 * row.field_coverage
        for row in rows
    )
    composite_scores = tuple(
        row.raw_bm25
        + 2 * row.title_bm25
        + row.body_bm25
        + 6 * row.title_coverage
        + 4 * row.body_coverage
        + 5 * row.field_coverage
        + 8_000_000 * row.literal_hits
        + 12_000_000 * row.error_hits
        + 10_000_000 * row.version_hits
        for row in rows
    )
    raw_indices = _rank(raw_scores, checked_documents)[:TOP_K]
    raw_ordinals = tuple(
        checked_documents[index].ordinal for index in raw_indices
    )
    indices_and_scores: dict[
        str, tuple[tuple[int, ...], tuple[int, ...]]
    ] = {}
    for recipe_id, scores in (
        (R0_RAW_BM25, raw_scores),
        (R1_TITLE_FOCUSED, title_scores),
        (R2_LITERAL_SIGNATURE_ANCHOR, literal_scores),
        (R3_FIELD_AWARE_COVERAGE, field_scores),
    ):
        indices = _rank(scores, checked_documents)[:TOP_K]
        indices_and_scores[recipe_id] = (
            indices,
            tuple(scores[index] for index in indices),
        )
    indices_and_scores[R4_MULTI_SEED_MARGINAL] = _multi_seed_selection(
        documents=checked_documents,
        raw_scores=raw_scores,
        title_scores=title_scores,
        literal_scores=literal_scores,
        field_scores=field_scores,
    )
    indices_and_scores[R5_TYPED_CASCADE] = _typed_cascade_selection(
        documents=checked_documents,
        raw_scores=raw_scores,
        title_scores=title_scores,
        literal_scores=literal_scores,
        field_scores=field_scores,
        composite_scores=composite_scores,
    )
    actions = tuple(
        RecipeAction(
            recipe_id=recipe_id,
            top5_document_ordinals=tuple(
                checked_documents[index].ordinal
                for index in indices_and_scores[recipe_id][0]
            ),
            selected_scores=indices_and_scores[recipe_id][1],
            raw_top5_document_ordinals=raw_ordinals,
        )
        for recipe_id in RECIPE_IDS
    )
    raw_maximum = max(raw_scores)
    selected_score_maximum = max(
        score
        for action in actions
        for score in action.selected_scores
    )
    action_features = tuple(
        _action_features(
            action=action,
            documents=checked_documents,
            structures=rows,
            raw_maximum=raw_maximum,
            selected_score_maximum=selected_score_maximum,
        )
        for action in actions
    )
    public_payload = [
        document_public_payload(document)
        for document in checked_documents
    ]
    serialized_hash_rows = [
        {
            "ordinal": document.ordinal,
            "serialized_sha256": hashlib.sha256(
                serialize_document_bytes(document)
            ).hexdigest(),
        }
        for document in checked_documents
    ]
    return ActionSlate(
        question_title_sha256=hashlib.sha256(
            question_title.encode("utf-8")
        ).hexdigest(),
        question_text_sha256=hashlib.sha256(
            question_text.encode("utf-8")
        ).hexdigest(),
        raw_query_bytes_sha256=hashlib.sha256(
            raw_query.encode("utf-8")
        ).hexdigest(),
        public_document_projection_sha256=stable_hash(public_payload),
        serialized_document_set_sha256=stable_hash(serialized_hash_rows),
        query=structure,
        document_structures=rows,
        actions=actions,
        features=action_features,
    )


def pairwise_signature(
    candidate: ActionFeatures,
    reference: ActionFeatures,
) -> tuple[int, ...]:
    if not isinstance(candidate, ActionFeatures) or not isinstance(
        reference, ActionFeatures
    ):
        raise TechqaP1TypedCoreError("pairwise signature inputs drifted")
    return tuple(
        (left > right) - (left < right)
        for left, right in zip(
            candidate.values,
            reference.values,
            strict=True,
        )
    )


@dataclass(frozen=True, slots=True)
class AFormExample:
    features: tuple[ActionFeatures, ...]
    utilities: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        if len(self.features) != len(RECIPE_IDS) or len(self.utilities) != len(
            RECIPE_IDS
        ):
            raise TechqaP1TypedCoreError("A_form example width drifted")
        if any(not isinstance(value, Fraction) for value in self.utilities):
            raise TechqaP1TypedCoreError("A_form utility is not exact")
        if any(value < 0 or value > 1 for value in self.utilities):
            raise TechqaP1TypedCoreError("A_form utility is outside [0,1]")


def make_aform_example(
    slate: ActionSlate,
    utility_by_recipe: Mapping[str, Fraction],
) -> AFormExample:
    if not isinstance(slate, ActionSlate) or set(utility_by_recipe) != set(
        RECIPE_IDS
    ):
        raise TechqaP1TypedCoreError("A_form utility projection drifted")
    return AFormExample(
        features=slate.features,
        utilities=tuple(
            utility_by_recipe[recipe_id]
            for recipe_id in RECIPE_IDS
        ),
    )


@dataclass(frozen=True, slots=True)
class SignatureRule:
    recipe_id: str
    signature: tuple[int, ...]
    support_count: int
    positive_count: int
    negative_count: int
    tie_count: int
    net_utility: Fraction
    minimum_delta: Fraction
    one_sided_reference_tail: Fraction
    qualified: bool

    def __post_init__(self) -> None:
        if self.recipe_id not in RECIPE_IDS or self.recipe_id == E0_RECIPE_ID:
            raise TechqaP1TypedCoreError("signature rule recipe drifted")
        if (
            len(self.signature) != len(SIGNATURE_FEATURE_NAMES)
            or any(value not in (-1, 0, 1) for value in self.signature)
            or self.support_count <= 0
            or self.positive_count + self.negative_count + self.tie_count
            != self.support_count
            or self.minimum_delta < -1
            or self.minimum_delta > 1
        ):
            raise TechqaP1TypedCoreError("signature rule statistics drifted")
        expected = (
            self.support_count >= MIN_SIGNATURE_SUPPORT
            and self.positive_count == self.support_count
            and self.negative_count == 0
            and self.tie_count == 0
            and self.minimum_delta > 0
            and self.one_sided_reference_tail
            <= Fraction(1, 2**MIN_SIGNATURE_SUPPORT)
        )
        if self.qualified is not expected:
            raise TechqaP1TypedCoreError("signature qualification drifted")

    def payload(self) -> dict[str, object]:
        return {
            "minimum_delta": [
                self.minimum_delta.numerator,
                self.minimum_delta.denominator,
            ],
            "negative_count": self.negative_count,
            "net_utility": [
                self.net_utility.numerator,
                self.net_utility.denominator,
            ],
            "one_sided_reference_tail": [
                self.one_sided_reference_tail.numerator,
                self.one_sided_reference_tail.denominator,
            ],
            "positive_count": self.positive_count,
            "qualified": self.qualified,
            "recipe_id": self.recipe_id,
            "signature": list(self.signature),
            "support_count": self.support_count,
            "tie_count": self.tie_count,
        }


@dataclass(frozen=True, slots=True)
class E1Model:
    rules: tuple[SignatureRule, ...]
    training_item_count: int
    pair_observation_count: int
    training_stage: str = "A_form"
    reference_recipe_id: str = E0_RECIPE_ID

    def __post_init__(self) -> None:
        if (
            self.training_item_count <= 0
            or self.pair_observation_count
            != self.training_item_count * (len(RECIPE_IDS) - 1)
            or self.training_stage != "A_form"
            or self.reference_recipe_id != E0_RECIPE_ID
        ):
            raise TechqaP1TypedCoreError("E1 model identity drifted")
        keys = [
            (rule.recipe_id, rule.signature)
            for rule in self.rules
        ]
        if keys != sorted(
            keys,
            key=lambda row: (RECIPE_IDS.index(row[0]), row[1]),
        ) or len(set(keys)) != len(keys):
            raise TechqaP1TypedCoreError("E1 signature registry drifted")

    def payload(self) -> dict[str, object]:
        body = {
            "forbidden_model_inputs": [
                "answer_span",
                "answerable",
                "cluster_identity",
                "document_or_qrel_identity",
                "family",
                "source_identity",
                "split_or_stage_identity_as_feature",
            ],
            "minimum_signature_support": MIN_SIGNATURE_SUPPORT,
            "pair_observation_count": self.pair_observation_count,
            "reference_recipe_id": self.reference_recipe_id,
            "rules": [rule.payload() for rule in self.rules],
            "schema": f"{VERSION}_E1_pairwise_signature_model",
            "signature_feature_names": list(SIGNATURE_FEATURE_NAMES),
            "study_id": STUDY_ID,
            "training_item_count": self.training_item_count,
            "training_stage": self.training_stage,
        }
        body["self_sha256"] = stable_hash(body)
        return body


def fit_e1(examples: Sequence[AFormExample]) -> E1Model:
    """Fit exact candidate-vs-E0 rules from sealed A_form examples only."""

    if isinstance(examples, (str, bytes)) or not examples:
        raise TechqaP1TypedCoreError("A_form examples are empty")
    checked = tuple(examples)
    if any(not isinstance(example, AFormExample) for example in checked):
        raise TechqaP1TypedCoreError("A_form contains an invalid example")
    reference_index = RECIPE_IDS.index(E0_RECIPE_ID)
    grouped: defaultdict[
        tuple[str, tuple[int, ...]],
        list[Fraction],
    ] = defaultdict(list)
    for example in checked:
        reference_features = example.features[reference_index]
        reference_utility = example.utilities[reference_index]
        for recipe_index, recipe_id in enumerate(RECIPE_IDS):
            if recipe_id == E0_RECIPE_ID:
                continue
            signature = pairwise_signature(
                example.features[recipe_index],
                reference_features,
            )
            grouped[(recipe_id, signature)].append(
                example.utilities[recipe_index] - reference_utility
            )
    rules: list[SignatureRule] = []
    for recipe_id, signature in sorted(
        grouped,
        key=lambda row: (RECIPE_IDS.index(row[0]), row[1]),
    ):
        deltas = grouped[(recipe_id, signature)]
        positive = sum(delta > 0 for delta in deltas)
        negative = sum(delta < 0 for delta in deltas)
        ties = len(deltas) - positive - negative
        nonzero_count = positive + negative
        tail = (
            Fraction(1, 1)
            if nonzero_count == 0
            else Fraction(
                sum(
                    math.comb(nonzero_count, successes)
                    for successes in range(positive, nonzero_count + 1)
                ),
                2**nonzero_count,
            )
        )
        minimum = min(deltas)
        qualified = (
            len(deltas) >= MIN_SIGNATURE_SUPPORT
            and positive == len(deltas)
            and negative == 0
            and ties == 0
            and minimum > 0
            and tail <= Fraction(1, 2**MIN_SIGNATURE_SUPPORT)
        )
        rules.append(
            SignatureRule(
                recipe_id=recipe_id,
                signature=signature,
                support_count=len(deltas),
                positive_count=positive,
                negative_count=negative,
                tie_count=ties,
                net_utility=sum(deltas, Fraction(0, 1)),
                minimum_delta=minimum,
                one_sided_reference_tail=tail,
                qualified=qualified,
            )
        )
    return E1Model(
        rules=tuple(rules),
        training_item_count=len(checked),
        pair_observation_count=len(checked) * (len(RECIPE_IDS) - 1),
    )


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    stage: str
    evaluator_id: str
    selected_recipe_id: str
    e0_recipe_id: str
    top5_document_ordinals: tuple[int, ...]
    fallback_to_e0: bool
    matched_signature: tuple[int, ...] | None
    conservative_minimum_delta: Fraction

    def __post_init__(self) -> None:
        if (
            self.stage not in POLICY_STAGES
            or self.evaluator_id not in {"E0", "E1"}
            or self.selected_recipe_id not in RECIPE_IDS
            or self.e0_recipe_id != E0_RECIPE_ID
            or len(self.top5_document_ordinals) != TOP_K
            or len(set(self.top5_document_ordinals)) != TOP_K
        ):
            raise TechqaP1TypedCoreError("policy decision drifted")
        if self.evaluator_id == "E0" and (
            self.selected_recipe_id != E0_RECIPE_ID
            or not self.fallback_to_e0
            or self.matched_signature is not None
            or self.conservative_minimum_delta != 0
        ):
            raise TechqaP1TypedCoreError("E0 decision drifted")


def apply_e0(
    slate: ActionSlate,
    *,
    stage: str,
) -> PolicyDecision:
    if not isinstance(slate, ActionSlate) or stage not in POLICY_STAGES:
        raise TechqaP1TypedCoreError("E0 application stage drifted")
    action = slate.action(E0_RECIPE_ID)
    return PolicyDecision(
        stage=stage,
        evaluator_id="E0",
        selected_recipe_id=E0_RECIPE_ID,
        e0_recipe_id=E0_RECIPE_ID,
        top5_document_ordinals=action.top5_document_ordinals,
        fallback_to_e0=True,
        matched_signature=None,
        conservative_minimum_delta=Fraction(0, 1),
    )


def apply_e1(
    model: E1Model,
    slate: ActionSlate,
    *,
    stage: str,
) -> PolicyDecision:
    """Apply one frozen A_form model identically on F, A_hold, or M."""

    if (
        not isinstance(model, E1Model)
        or not isinstance(slate, ActionSlate)
        or stage not in POLICY_STAGES
    ):
        raise TechqaP1TypedCoreError("E1 application inputs drifted")
    reference = slate.feature_row(E0_RECIPE_ID)
    qualified_rules = {
        (rule.recipe_id, rule.signature): rule
        for rule in model.rules
        if rule.qualified
    }
    candidates: list[
        tuple[Fraction, Fraction, int, int, SignatureRule]
    ] = []
    for recipe_id in RECIPE_IDS:
        if recipe_id == E0_RECIPE_ID:
            continue
        signature = pairwise_signature(
            slate.feature_row(recipe_id),
            reference,
        )
        rule = qualified_rules.get((recipe_id, signature))
        if rule is None:
            continue
        candidates.append(
            (
                rule.minimum_delta,
                rule.net_utility,
                rule.support_count,
                -RECIPE_IDS.index(recipe_id),
                rule,
            )
        )
    if not candidates:
        baseline = apply_e0(slate, stage=stage)
        return PolicyDecision(
            stage=stage,
            evaluator_id="E1",
            selected_recipe_id=baseline.selected_recipe_id,
            e0_recipe_id=E0_RECIPE_ID,
            top5_document_ordinals=baseline.top5_document_ordinals,
            fallback_to_e0=True,
            matched_signature=None,
            conservative_minimum_delta=Fraction(0, 1),
        )
    selected_rule = max(candidates, key=lambda row: row[:4])[4]
    action = slate.action(selected_rule.recipe_id)
    return PolicyDecision(
        stage=stage,
        evaluator_id="E1",
        selected_recipe_id=selected_rule.recipe_id,
        e0_recipe_id=E0_RECIPE_ID,
        top5_document_ordinals=action.top5_document_ordinals,
        fallback_to_e0=False,
        matched_signature=selected_rule.signature,
        conservative_minimum_delta=selected_rule.minimum_delta,
    )


__all__ = [
    "ActionFeatures",
    "ActionSlate",
    "AFormExample",
    "Document",
    "DocumentStructure",
    "E0_RECIPE_ID",
    "E1Model",
    "FEATURE_NAMES",
    "MIN_SIGNATURE_SUPPORT",
    "POLICY_STAGES",
    "PUBLIC_DOCUMENT_FIELDS",
    "PolicyDecision",
    "QueryStructure",
    "R0_RAW_BM25",
    "R1_TITLE_FOCUSED",
    "R2_LITERAL_SIGNATURE_ANCHOR",
    "R3_FIELD_AWARE_COVERAGE",
    "R4_MULTI_SEED_MARGINAL",
    "R5_TYPED_CASCADE",
    "RECIPE_IDS",
    "RecipeAction",
    "SCALE",
    "SERIALIZATION_SEPARATOR",
    "SIGNATURE_FEATURE_NAMES",
    "SignatureRule",
    "STUDY_ID",
    "TOP_K",
    "TechqaP1TypedCoreError",
    "VERSION",
    "apply_e0",
    "apply_e1",
    "bm25_scores",
    "build_action_slate",
    "canonical_bytes",
    "document_from_public_fields",
    "document_public_payload",
    "fit_e1",
    "lexical_tokens",
    "make_aform_example",
    "normalize_text",
    "pairwise_signature",
    "query_structure",
    "serialize_document_bytes",
    "serialize_document_text",
    "serialize_query_text",
    "stable_hash",
]
