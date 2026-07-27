"""Source-free typed evidence-set selection for BioASQ P1.

The action boundary is deliberately narrow.  It accepts only one public
question string, the exact public ``ordinal/text`` projection of the frozen
2,900-passage corpus, and six integer, label-free score vectors.  It has no
source loader, file, network, gold question type, family, document identity,
snippet label, qrel, answer, split, or outcome entrypoint.

A frozen query-only structural predictor chooses one of four *predicted*
buckets.  Four typed evidence-set recipes and a frozen global RRF E0 produce
complete rankings over the same corpus.  A_form utilities may enter only
after a slate is sealed; they fit one immutable E1 recipe rule per predicted
bucket.  The resulting program is applied unchanged on A_hold and M_search.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import re
import unicodedata
from typing import Mapping, Sequence


VERSION = "bioasq_p1_typed_core_v1"
STUDY_ID = "BIOASQ_P1_TYPED_QUESTION_EVIDENCE_EVALUATOR_L5_V1"
CORPUS_SIZE = 2_900
TOP_K = 5
SCALE = 300_000
MAX_UTILITY = 600_000

MAX_QUERY_CHARACTERS = 100_000
MAX_PASSAGE_CHARACTERS = 2_000_000
MAX_SCORE_ABS = 10**15

B0_CLAIM = 0
B1_ENTITY = 1
B2_LIST = 2
B3_ASPECT = 3
PREDICTED_BUCKETS = (B0_CLAIM, B1_ENTITY, B2_LIST, B3_ASPECT)
BUCKET_NAMES = ("claim", "entity", "list", "aspect")
POLICY_STAGES = ("F_search", "A_hold", "M_search")

R1_CLAIM_BALANCE = "claim_polarity_balanced_evidence_set"
R2_ENTITY_FOCUS = "entity_focused_evidence_set"
R3_LIST_DIVERSITY = "list_redundancy_controlled_evidence_set"
R4_ASPECT_COVERAGE = "multi_aspect_coverage_evidence_set"
R0_GLOBAL_RAW_DENSE_RRF = (
    "global_raw_dense_reciprocal_rank_fusion"
)
TYPED_RECIPE_IDS = (
    R1_CLAIM_BALANCE,
    R2_ENTITY_FOCUS,
    R3_LIST_DIVERSITY,
    R4_ASPECT_COVERAGE,
)
E0_RECIPE_ID = R0_GLOBAL_RAW_DENSE_RRF
RECIPE_IDS = TYPED_RECIPE_IDS + (E0_RECIPE_ID,)

PUBLIC_PASSAGE_FIELDS = ("ordinal", "text")
SCORE_NAMES = (
    "raw_ce",
    "focus_ce",
    "dense_base",
    "dense_support",
    "dense_contrast",
    "dense_coverage",
)

MIN_BUCKET_SUPPORT = 6
MIN_NET_POSITIVE_MARGIN_COUNT = 2
SHRINKAGE_PSEUDOCOUNT = 4
RRF_K = 60
LIST_DIVERSITY_CANDIDATE_PREFIX = 128
LIST_DIVERSITY_RELEVANCE_WEIGHT = 1
LIST_DIVERSITY_NOVELTY_WEIGHT = 1
_HEX_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
_YES_NO_AUXILIARIES = frozenset(
    {
        "am",
        "are",
        "can",
        "could",
        "did",
        "do",
        "does",
        "had",
        "has",
        "have",
        "is",
        "may",
        "might",
        "must",
        "should",
        "was",
        "were",
        "will",
        "would",
    }
)
_WH_HEADS = frozenset(
    {"what", "which", "who", "where", "when", "why", "how"}
)
_LIST_LEADS = frozenset({"enumerate", "list", "name"})
_LIST_NOUNS = frozenset(
    {
        "agents",
        "causes",
        "complications",
        "diseases",
        "drugs",
        "effects",
        "examples",
        "factors",
        "genes",
        "interventions",
        "markers",
        "mechanisms",
        "methods",
        "mutations",
        "pathways",
        "proteins",
        "receptors",
        "risks",
        "symptoms",
        "therapies",
        "treatments",
        "types",
    }
)
_SUMMARY_LEADS = frozenset(
    {
        "characterize",
        "compare",
        "define",
        "describe",
        "discuss",
        "explain",
        "outline",
        "summarize",
    }
)
_SUMMARY_PHRASES = (
    ("how", "does"),
    ("how", "do"),
    ("how", "is"),
    ("what", "is", "known"),
    ("what", "role"),
    ("what", "mechanism"),
)

_FOCUS_INSTRUCTIONS = (
    "find evidence that directly supports or challenges the claim",
    "find evidence that identifies the requested biomedical entity",
    "find distinct evidence instances that enumerate the requested set",
    "find complementary evidence covering the requested explanation",
)


class BioasqP1TypedCoreError(ValueError):
    """A public projection, frozen action, or evaluator contract drifted."""


def canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    """Encode strict deterministic JSON for hashes and sealed artifacts."""

    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise BioasqP1TypedCoreError("value is not canonical JSON") from exc
    return encoded + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _validate_characters(value: str, *, field: str) -> None:
    for character in value:
        category = unicodedata.category(character)
        if category.startswith("C") and not character.isspace():
            raise BioasqP1TypedCoreError(
                f"{field} contains a forbidden control character"
            )


def normalize_text(
    value: object,
    *,
    field: str = "text",
    maximum_length: int = MAX_PASSAGE_CHARACTERS,
) -> str:
    """NFKC-normalize and collapse every whitespace run to one ASCII space."""

    if (
        type(maximum_length) is not int
        or maximum_length <= 0
        or not isinstance(value, str)
        or len(value) > maximum_length
    ):
        raise BioasqP1TypedCoreError(f"{field} is invalid")
    _validate_characters(value, field=field)
    normalized = " ".join(unicodedata.normalize("NFKC", value).split())
    _validate_characters(normalized, field=field)
    if not normalized:
        raise BioasqP1TypedCoreError(f"{field} is empty")
    if len(normalized) > maximum_length:
        raise BioasqP1TypedCoreError(
            f"{field} is too long after normalization"
        )
    return normalized


def _query_tokens(query_text: str) -> tuple[str, ...]:
    return tuple(
        match.group(0).casefold()
        for match in _TOKEN_RE.finditer(query_text)
    )


def validate_query_text(query_text: str) -> str:
    """Return the sole canonical public query accepted by every scorer."""

    normalized = normalize_text(
        query_text,
        field="query text",
        maximum_length=MAX_QUERY_CHARACTERS,
    )
    if not _query_tokens(normalized):
        raise BioasqP1TypedCoreError("query text has no lexical token")
    return normalized


def _starts_with(tokens: Sequence[str], phrase: Sequence[str]) -> bool:
    return tuple(tokens[: len(phrase)]) == tuple(phrase)


def _bucket_from_signals(
    *,
    yes_no_cue: bool,
    list_cue: bool,
    summary_cue: bool,
) -> int:
    if yes_no_cue:
        return B0_CLAIM
    if list_cue:
        return B2_LIST
    if summary_cue:
        return B3_ASPECT
    return B1_ENTITY


@dataclass(frozen=True, slots=True)
class QuestionStructure:
    predicted_bucket: int
    bucket_name: str
    yes_no_cue: bool
    list_cue: bool
    summary_cue: bool
    wh_head: str | None
    query_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.yes_no_cue) is not bool
            or type(self.list_cue) is not bool
            or type(self.summary_cue) is not bool
            or (
                self.wh_head is not None
                and self.wh_head not in _WH_HEADS
            )
            or _HEX_SHA256_RE.fullmatch(self.query_sha256) is None
        ):
            raise BioasqP1TypedCoreError(
                "question structure is malformed"
            )
        expected_bucket = _bucket_from_signals(
            yes_no_cue=self.yes_no_cue,
            list_cue=self.list_cue,
            summary_cue=self.summary_cue,
        )
        if (
            self.predicted_bucket != expected_bucket
            or self.bucket_name != BUCKET_NAMES[expected_bucket]
        ):
            raise BioasqP1TypedCoreError(
                "question structure bucket drifted"
            )

    def body_payload(self) -> dict[str, object]:
        return {
            "bucket_name": self.bucket_name,
            "list_cue": self.list_cue,
            "predicted_bucket": self.predicted_bucket,
            "query_sha256": self.query_sha256,
            "schema": f"{VERSION}_question_structure",
            "summary_cue": self.summary_cue,
            "version": VERSION,
            "wh_head": self.wh_head,
            "yes_no_cue": self.yes_no_cue,
        }

    @property
    def structure_sha256(self) -> str:
        return stable_hash(self.body_payload())

    def payload(self) -> dict[str, object]:
        body = self.body_payload()
        return {**body, "self_sha256": stable_hash(body)}


def predict_question_structure(query_text: str) -> QuestionStructure:
    """Predict a four-way structural bucket from the public query only."""

    normalized = validate_query_text(query_text)
    tokens = _query_tokens(normalized)
    yes_no_cue = (
        tokens[0] in _YES_NO_AUXILIARIES
        or "whether" in tokens[:8]
    )
    list_cue = (
        tokens[0] in _LIST_LEADS
        or _starts_with(tokens, ("what", "are"))
        or _starts_with(tokens, ("which", "are"))
        or (
            tokens[0] in {"what", "which"}
            and any(token in _LIST_NOUNS for token in tokens[1:8])
        )
    )
    summary_cue = (
        tokens[0] in _SUMMARY_LEADS
        or any(_starts_with(tokens, phrase) for phrase in _SUMMARY_PHRASES)
    )
    wh_head = tokens[0] if tokens[0] in _WH_HEADS else None
    bucket = _bucket_from_signals(
        yes_no_cue=yes_no_cue,
        list_cue=list_cue,
        summary_cue=summary_cue,
    )
    return QuestionStructure(
        predicted_bucket=bucket,
        bucket_name=BUCKET_NAMES[bucket],
        yes_no_cue=yes_no_cue,
        list_cue=list_cue,
        summary_cue=summary_cue,
        wh_head=wh_head,
        query_sha256=hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class ScoreQueryBundle:
    raw_ce: str
    focus_ce: str
    dense_base: str
    dense_support: str
    dense_contrast: str
    dense_coverage: str
    predicted_bucket: int
    question_structure_sha256: str

    def __post_init__(self) -> None:
        for name in SCORE_NAMES:
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise BioasqP1TypedCoreError(
                    "score-query serialization is malformed"
                )
        if (
            type(self.predicted_bucket) is not int
            or self.predicted_bucket not in PREDICTED_BUCKETS
            or _HEX_SHA256_RE.fullmatch(
                self.question_structure_sha256
            )
            is None
        ):
            raise BioasqP1TypedCoreError(
                "score-query bundle identity is malformed"
            )

    def body_payload(self) -> dict[str, object]:
        return {
            "predicted_bucket": self.predicted_bucket,
            "queries": {
                name: getattr(self, name) for name in SCORE_NAMES
            },
            "question_structure_sha256": (
                self.question_structure_sha256
            ),
            "schema": f"{VERSION}_score_query_bundle",
            "score_names": list(SCORE_NAMES),
            "version": VERSION,
        }

    @property
    def bundle_sha256(self) -> str:
        return stable_hash(self.body_payload())

    def payload(self) -> dict[str, object]:
        body = self.body_payload()
        return {**body, "self_sha256": stable_hash(body)}


def serialize_score_queries(query_text: str) -> ScoreQueryBundle:
    """Return all frozen, query-only scorer serializations."""

    normalized = validate_query_text(query_text)
    structure = predict_question_structure(normalized)
    return ScoreQueryBundle(
        raw_ce=normalized,
        focus_ce=(
            f"EVIDENCE FOCUS: "
            f"{_FOCUS_INSTRUCTIONS[structure.predicted_bucket]}\n"
            f"QUESTION: {normalized}"
        ),
        dense_base=normalized,
        dense_support=(
            f"supporting biomedical evidence for: {normalized}"
        ),
        dense_contrast=(
            f"contrasting or qualifying biomedical evidence for: "
            f"{normalized}"
        ),
        dense_coverage=(
            f"distinct complementary biomedical evidence aspects for: "
            f"{normalized}"
        ),
        predicted_bucket=structure.predicted_bucket,
        question_structure_sha256=structure.structure_sha256,
    )


def serialize_query_for_score(query_text: str, score_name: str) -> str:
    if score_name not in SCORE_NAMES:
        raise BioasqP1TypedCoreError("score name is not frozen")
    return getattr(serialize_score_queries(query_text), score_name)


@dataclass(frozen=True, slots=True)
class Passage:
    ordinal: int
    text: str

    def __post_init__(self) -> None:
        if (
            type(self.ordinal) is not int
            or not 0 <= self.ordinal < CORPUS_SIZE
        ):
            raise BioasqP1TypedCoreError("passage ordinal is invalid")
        canonical = normalize_text(
            self.text,
            field="passage text",
            maximum_length=MAX_PASSAGE_CHARACTERS,
        )
        object.__setattr__(self, "text", canonical)


def passage_from_public_fields(value: object) -> Passage:
    if (
        not isinstance(value, Mapping)
        or set(value) != set(PUBLIC_PASSAGE_FIELDS)
    ):
        raise BioasqP1TypedCoreError(
            "passage projection is not the exact public field set"
        )
    return Passage(
        ordinal=value["ordinal"],  # type: ignore[arg-type]
        text=value["text"],  # type: ignore[arg-type]
    )


def passage_public_payload(passage: Passage) -> dict[str, object]:
    if not isinstance(passage, Passage):
        raise BioasqP1TypedCoreError("passage is not a Passage")
    return {"ordinal": passage.ordinal, "text": passage.text}


def serialize_passage(passage: Passage) -> str:
    if not isinstance(passage, Passage):
        raise BioasqP1TypedCoreError("passage is not a Passage")
    return passage.text


def serialize_passage_bytes(passage: Passage) -> bytes:
    return serialize_passage(passage).encode("utf-8")


def _checked_action_inputs(
    passages: Sequence[Passage],
    score_vectors: Mapping[str, Sequence[int]],
) -> tuple[tuple[Passage, ...], dict[str, tuple[int, ...]]]:
    if isinstance(passages, (str, bytes)) or len(passages) != CORPUS_SIZE:
        raise BioasqP1TypedCoreError(
            "passage corpus must contain exactly 2900 rows"
        )
    checked_passages = tuple(passages)
    if any(not isinstance(row, Passage) for row in checked_passages):
        raise BioasqP1TypedCoreError(
            "passage corpus contains a non-passage"
        )
    ordinals = [row.ordinal for row in checked_passages]
    if set(ordinals) != set(range(CORPUS_SIZE)):
        raise BioasqP1TypedCoreError(
            "passage ordinals are not the frozen corpus universe"
        )
    if set(score_vectors) != set(SCORE_NAMES):
        raise BioasqP1TypedCoreError("score vector registry drifted")
    checked_vectors: dict[str, tuple[int, ...]] = {}
    for name in SCORE_NAMES:
        vector = score_vectors[name]
        if (
            isinstance(vector, (str, bytes))
            or len(vector) != CORPUS_SIZE
        ):
            raise BioasqP1TypedCoreError(
                f"{name} score vector width drifted"
            )
        values = tuple(vector)
        if any(
            type(score) is not int or abs(score) > MAX_SCORE_ABS
            for score in values
        ):
            raise BioasqP1TypedCoreError(
                f"{name} scores are not bounded integers"
            )
        checked_vectors[name] = values

    permutation = tuple(
        sorted(
            range(CORPUS_SIZE),
            key=lambda index: checked_passages[index].ordinal,
        )
    )
    return (
        tuple(checked_passages[index] for index in permutation),
        {
            name: tuple(
                checked_vectors[name][index] for index in permutation
            )
            for name in SCORE_NAMES
        },
    )


def _rank(
    scores: Sequence[int],
    passages: Sequence[Passage],
) -> tuple[int, ...]:
    if len(scores) != len(passages):
        raise BioasqP1TypedCoreError("rank width drifted")
    return tuple(
        sorted(
            range(len(passages)),
            key=lambda index: (-scores[index], passages[index].ordinal),
        )
    )


def _rank_points(
    scores: Sequence[int],
    passages: Sequence[Passage],
) -> tuple[int, ...]:
    order = _rank(scores, passages)
    result = [0] * len(passages)
    for rank, index in enumerate(order):
        result[index] = (
            (len(passages) - rank) * SCALE // len(passages)
        )
    return tuple(result)


def _weighted_order(
    *,
    passages: Sequence[Passage],
    points: Mapping[str, Sequence[int]],
    weights: Sequence[tuple[str, int]],
) -> tuple[int, ...]:
    if any(
        name not in SCORE_NAMES or type(weight) is not int or weight <= 0
        for name, weight in weights
    ):
        raise BioasqP1TypedCoreError("recipe weight registry drifted")
    fused = tuple(
        sum(weight * points[name][index] for name, weight in weights)
        for index in range(len(passages))
    )
    return _rank(fused, passages)


def _round_robin_order(
    *,
    view_orders: Mapping[str, Sequence[int]],
    view_names: Sequence[str],
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    if (
        not view_names
        or len(set(view_names)) != len(view_names)
        or any(name not in view_orders for name in view_names)
    ):
        raise BioasqP1TypedCoreError(
            "round-robin view registry drifted"
        )
    width = len(view_orders[view_names[0]])
    if any(len(view_orders[name]) != width for name in view_names):
        raise BioasqP1TypedCoreError("round-robin view width drifted")
    selected: list[int] = []
    traces: list[str] = []
    seen: set[int] = set()
    for depth in range(width):
        for name in view_names:
            index = view_orders[name][depth]
            if index in seen:
                continue
            selected.append(index)
            traces.append(f"view:{name}:rank{depth}")
            seen.add(index)
    if len(selected) != width:
        raise BioasqP1TypedCoreError(
            "round-robin recipe did not totalize the corpus"
        )
    return tuple(selected), tuple(traces)


def _token_jaccard(
    left: frozenset[str],
    right: frozenset[str],
) -> Fraction:
    union = left | right
    if not union:
        return Fraction(1, 1)
    return Fraction(len(left & right), len(union))


def _list_redundancy_controlled_order(
    *,
    base_order: Sequence[int],
    passages: Sequence[Passage],
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Greedily rerank a frozen prefix by relevance plus lexical novelty."""

    if (
        len(base_order) != CORPUS_SIZE
        or set(base_order) != set(range(CORPUS_SIZE))
        or len(passages) != CORPUS_SIZE
        or not TOP_K <= LIST_DIVERSITY_CANDIDATE_PREFIX <= CORPUS_SIZE
    ):
        raise BioasqP1TypedCoreError(
            "list-diversity candidate registry drifted"
        )
    prefix = tuple(base_order[:LIST_DIVERSITY_CANDIDATE_PREFIX])
    base_rank = {
        candidate: rank for rank, candidate in enumerate(base_order)
    }
    token_sets = {
        candidate: frozenset(_query_tokens(passages[candidate].text))
        for candidate in prefix
    }
    selected: list[int] = []
    top_traces: list[str] = []
    selected_set: set[int] = set()

    while len(selected) < TOP_K:
        winner: int | None = None
        winner_key: tuple[Fraction, int, int] | None = None
        winner_similarity = Fraction(0, 1)
        for candidate in prefix:
            if candidate in selected_set:
                continue
            maximum_similarity = max(
                (
                    _token_jaccard(
                        token_sets[candidate],
                        token_sets[chosen],
                    )
                    for chosen in selected
                ),
                default=Fraction(0, 1),
            )
            relevance = Fraction(
                LIST_DIVERSITY_CANDIDATE_PREFIX
                - base_rank[candidate],
                LIST_DIVERSITY_CANDIDATE_PREFIX,
            )
            novelty = Fraction(1, 1) - maximum_similarity
            greedy_score = (
                LIST_DIVERSITY_RELEVANCE_WEIGHT * relevance
                + LIST_DIVERSITY_NOVELTY_WEIGHT * novelty
            )
            key = (
                greedy_score,
                -base_rank[candidate],
                -passages[candidate].ordinal,
            )
            if winner_key is None or key > winner_key:
                winner = candidate
                winner_key = key
                winner_similarity = maximum_similarity
        if winner is None:
            raise BioasqP1TypedCoreError(
                "list-diversity prefix was exhausted"
            )
        selected.append(winner)
        selected_set.add(winner)
        top_traces.append(
            (
                f"greedy:base_rank{base_rank[winner]}:"
                f"max_jaccard{winner_similarity.numerator}_"
                f"{winner_similarity.denominator}"
            )
        )

    full_order = tuple(
        [*selected]
        + [
            candidate
            for candidate in base_order
            if candidate not in selected_set
        ]
    )
    traces = tuple(
        [*top_traces]
        + [
            f"base:list_multiview:rank{base_rank[candidate]}"
            for candidate in full_order[TOP_K:]
        ]
    )
    if (
        len(full_order) != CORPUS_SIZE
        or set(full_order) != set(range(CORPUS_SIZE))
        or len(traces) != CORPUS_SIZE
    ):
        raise BioasqP1TypedCoreError(
            "list-diversity recipe did not totalize the corpus"
        )
    return full_order, traces


def _global_raw_dense_rrf(
    *,
    view_orders: Mapping[str, Sequence[int]],
    passages: Sequence[Passage],
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    names = ("raw_ce", "dense_base")
    if (
        any(name not in view_orders for name in names)
        or any(len(view_orders[name]) != len(passages) for name in names)
    ):
        raise BioasqP1TypedCoreError("E0 RRF view registry drifted")
    positions: dict[str, list[int]] = {
        name: [0] * len(passages) for name in names
    }
    for name in names:
        for rank, index in enumerate(view_orders[name], start=1):
            positions[name][index] = rank
    fused = tuple(
        sum(
            (
                Fraction(1, RRF_K + positions[name][index])
                for name in names
            ),
            Fraction(0, 1),
        )
        for index in range(len(passages))
    )
    order = tuple(
        sorted(
            range(len(passages)),
            key=lambda index: (
                -fused[index],
                passages[index].ordinal,
            ),
        )
    )
    traces = tuple(
        (
            f"rrf:raw_ce_rank{positions['raw_ce'][index]}:"
            f"dense_base_rank{positions['dense_base'][index]}"
        )
        for index in order
    )
    return order, traces


@dataclass(frozen=True, slots=True)
class EvidenceAction:
    recipe_id: str
    ranked_ordinals: tuple[int, ...]
    top5_trace: tuple[str, ...]
    behavior_digest: str

    def __post_init__(self) -> None:
        if self.recipe_id not in RECIPE_IDS:
            raise BioasqP1TypedCoreError("recipe id is not frozen")
        if (
            len(self.ranked_ordinals) != CORPUS_SIZE
            or set(self.ranked_ordinals) != set(range(CORPUS_SIZE))
            or len(self.top5_trace) != TOP_K
            or any(
                not isinstance(value, str) or not value
                for value in self.top5_trace
            )
            or _HEX_SHA256_RE.fullmatch(self.behavior_digest) is None
        ):
            raise BioasqP1TypedCoreError(
                "evidence action is malformed"
            )

    @property
    def top5_ordinals(self) -> tuple[int, ...]:
        return self.ranked_ordinals[:TOP_K]

    def payload(self) -> dict[str, object]:
        return {
            "behavior_digest": self.behavior_digest,
            "ranking_sha256": stable_hash(list(self.ranked_ordinals)),
            "recipe_id": self.recipe_id,
            "top5_ordinals": list(self.top5_ordinals),
            "top5_trace": list(self.top5_trace),
        }


def _make_action(
    *,
    recipe_id: str,
    order: Sequence[int],
    traces: Sequence[str],
    passages: Sequence[Passage],
    query_sha256: str,
    passage_sha256: str,
    score_sha256: str,
    predicted_bucket: int,
) -> EvidenceAction:
    if (
        len(order) != CORPUS_SIZE
        or set(order) != set(range(CORPUS_SIZE))
        or len(traces) != CORPUS_SIZE
    ):
        raise BioasqP1TypedCoreError(
            "recipe did not return a full candidate permutation"
        )
    ordinals = tuple(passages[index].ordinal for index in order)
    top5_trace = tuple(traces[:TOP_K])
    behavior_digest = stable_hash(
        {
            "passage_sha256": passage_sha256,
            "predicted_bucket": predicted_bucket,
            "query_sha256": query_sha256,
            "ranked_ordinals": list(ordinals),
            "recipe_id": recipe_id,
            "score_sha256": score_sha256,
            "study_id": STUDY_ID,
            "top5_trace": list(top5_trace),
            "version": VERSION,
        }
    )
    return EvidenceAction(
        recipe_id=recipe_id,
        ranked_ordinals=ordinals,
        top5_trace=top5_trace,
        behavior_digest=behavior_digest,
    )


@dataclass(frozen=True, slots=True)
class ActionSlate:
    predicted_bucket: int
    question_structure_sha256: str
    normalized_query_sha256: str
    score_query_bundle_sha256: str
    passage_projection_sha256: str
    passage_serialization_sha256: str
    score_bundle_sha256: str
    actions: tuple[EvidenceAction, ...]

    def __post_init__(self) -> None:
        if (
            type(self.predicted_bucket) is not int
            or self.predicted_bucket not in PREDICTED_BUCKETS
        ):
            raise BioasqP1TypedCoreError("slate bucket is invalid")
        for digest in (
            self.question_structure_sha256,
            self.normalized_query_sha256,
            self.score_query_bundle_sha256,
            self.passage_projection_sha256,
            self.passage_serialization_sha256,
            self.score_bundle_sha256,
        ):
            if _HEX_SHA256_RE.fullmatch(digest) is None:
                raise BioasqP1TypedCoreError("slate digest is malformed")
        if tuple(action.recipe_id for action in self.actions) != RECIPE_IDS:
            raise BioasqP1TypedCoreError("slate recipe order drifted")
        if any(
            set(action.ranked_ordinals) != set(range(CORPUS_SIZE))
            for action in self.actions
        ):
            raise BioasqP1TypedCoreError(
                "slate candidate universe drifted"
            )

    def action(self, recipe_id: str) -> EvidenceAction:
        if recipe_id not in RECIPE_IDS:
            raise BioasqP1TypedCoreError("recipe id is not frozen")
        return self.actions[RECIPE_IDS.index(recipe_id)]

    def audit_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "actions": [action.payload() for action in self.actions],
            "corpus_size": CORPUS_SIZE,
            "e0_recipe_id": E0_RECIPE_ID,
            "label_bearing_action_inputs": False,
            "list_diversity": {
                "candidate_prefix": LIST_DIVERSITY_CANDIDATE_PREFIX,
                "novelty_weight": LIST_DIVERSITY_NOVELTY_WEIGHT,
                "passage_similarity": "token_set_jaccard",
                "relevance_weight": (
                    LIST_DIVERSITY_RELEVANCE_WEIGHT
                ),
            },
            "normalized_query_sha256": self.normalized_query_sha256,
            "passage_projection_sha256": (
                self.passage_projection_sha256
            ),
            "passage_serialization_sha256": (
                self.passage_serialization_sha256
            ),
            "policy_stages": list(POLICY_STAGES),
            "predicted_bucket": self.predicted_bucket,
            "public_passage_fields": list(PUBLIC_PASSAGE_FIELDS),
            "question_structure_sha256": (
                self.question_structure_sha256
            ),
            "recipe_ids": list(RECIPE_IDS),
            "score_bundle_sha256": self.score_bundle_sha256,
            "score_names": list(SCORE_NAMES),
            "score_query_bundle_sha256": (
                self.score_query_bundle_sha256
            ),
            "study_id": STUDY_ID,
            "top_k": TOP_K,
            "typed_recipe_ids": list(TYPED_RECIPE_IDS),
            "version": VERSION,
        }
        return {**payload, "self_sha256": stable_hash(payload)}


def build_action_slate(
    query_text: str,
    passages: Sequence[Passage],
    raw_ce_scores: Sequence[int],
    focus_ce_scores: Sequence[int],
    dense_base_scores: Sequence[int],
    dense_support_scores: Sequence[int],
    dense_contrast_scores: Sequence[int],
    dense_coverage_scores: Sequence[int],
) -> ActionSlate:
    """Build every frozen ranking without accepting gold information."""

    normalized_query = validate_query_text(query_text)
    structure = predict_question_structure(normalized_query)
    score_queries = serialize_score_queries(normalized_query)
    checked_passages, score_vectors = _checked_action_inputs(
        passages,
        {
            "raw_ce": raw_ce_scores,
            "focus_ce": focus_ce_scores,
            "dense_base": dense_base_scores,
            "dense_support": dense_support_scores,
            "dense_contrast": dense_contrast_scores,
            "dense_coverage": dense_coverage_scores,
        },
    )
    query_sha = hashlib.sha256(
        normalized_query.encode("utf-8")
    ).hexdigest()
    passage_payload = [
        passage_public_payload(passage)
        for passage in checked_passages
    ]
    passage_sha = stable_hash(passage_payload)
    passage_serialization_sha = stable_hash(
        [
            hashlib.sha256(
                serialize_passage_bytes(passage)
            ).hexdigest()
            for passage in checked_passages
        ]
    )
    score_sha = stable_hash(
        {
            "ordinals": [
                passage.ordinal for passage in checked_passages
            ],
            "scores": {
                name: list(score_vectors[name]) for name in SCORE_NAMES
            },
        }
    )

    view_orders = {
        name: _rank(score_vectors[name], checked_passages)
        for name in SCORE_NAMES
    }
    points = {
        name: _rank_points(score_vectors[name], checked_passages)
        for name in SCORE_NAMES
    }

    claim_order, claim_trace = _round_robin_order(
        view_orders=view_orders,
        view_names=(
            "dense_support",
            "dense_contrast",
            "raw_ce",
            "focus_ce",
            "dense_coverage",
            "dense_base",
        ),
    )
    entity_order = _weighted_order(
        passages=checked_passages,
        points=points,
        weights=(
            ("focus_ce", 14),
            ("raw_ce", 11),
            ("dense_base", 6),
            ("dense_support", 2),
        ),
    )
    entity_trace = tuple(
        f"weighted:{R2_ENTITY_FOCUS}:rank{rank}"
        for rank in range(CORPUS_SIZE)
    )
    list_base_order, _list_base_trace = _round_robin_order(
        view_orders=view_orders,
        view_names=(
            "dense_coverage",
            "dense_base",
            "focus_ce",
            "dense_support",
            "raw_ce",
            "dense_contrast",
        ),
    )
    list_order, list_trace = _list_redundancy_controlled_order(
        base_order=list_base_order,
        passages=checked_passages,
    )
    aspect_order = _weighted_order(
        passages=checked_passages,
        points=points,
        weights=(
            ("dense_coverage", 13),
            ("dense_support", 8),
            ("dense_contrast", 7),
            ("dense_base", 4),
            ("focus_ce", 2),
            ("raw_ce", 1),
        ),
    )
    aspect_trace = tuple(
        f"weighted:{R4_ASPECT_COVERAGE}:rank{rank}"
        for rank in range(CORPUS_SIZE)
    )
    typed_orders: dict[str, tuple[int, ...]] = {
        R1_CLAIM_BALANCE: claim_order,
        R2_ENTITY_FOCUS: entity_order,
        R3_LIST_DIVERSITY: list_order,
        R4_ASPECT_COVERAGE: aspect_order,
    }
    typed_traces: dict[str, tuple[str, ...]] = {
        R1_CLAIM_BALANCE: claim_trace,
        R2_ENTITY_FOCUS: entity_trace,
        R3_LIST_DIVERSITY: list_trace,
        R4_ASPECT_COVERAGE: aspect_trace,
    }
    e0_order, e0_trace = _global_raw_dense_rrf(
        view_orders=view_orders,
        passages=checked_passages,
    )
    orders = {**typed_orders, E0_RECIPE_ID: e0_order}
    traces = {**typed_traces, E0_RECIPE_ID: e0_trace}
    actions = tuple(
        _make_action(
            recipe_id=recipe_id,
            order=orders[recipe_id],
            traces=traces[recipe_id],
            passages=checked_passages,
            query_sha256=query_sha,
            passage_sha256=passage_sha,
            score_sha256=score_sha,
            predicted_bucket=structure.predicted_bucket,
        )
        for recipe_id in RECIPE_IDS
    )
    return ActionSlate(
        predicted_bucket=structure.predicted_bucket,
        question_structure_sha256=structure.structure_sha256,
        normalized_query_sha256=query_sha,
        score_query_bundle_sha256=score_queries.bundle_sha256,
        passage_projection_sha256=passage_sha,
        passage_serialization_sha256=passage_serialization_sha,
        score_bundle_sha256=score_sha,
        actions=actions,
    )


@dataclass(frozen=True, slots=True)
class AFormExample:
    predicted_bucket: int
    utility_vector: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            type(self.predicted_bucket) is not int
            or self.predicted_bucket not in PREDICTED_BUCKETS
            or len(self.utility_vector) != len(RECIPE_IDS)
            or any(
                type(value) is not int
                or not 0 <= value <= MAX_UTILITY
                for value in self.utility_vector
            )
        ):
            raise BioasqP1TypedCoreError(
                "A_form utility vector is invalid"
            )


def make_aform_example(
    slate: ActionSlate,
    utility_vector: Sequence[int],
) -> AFormExample:
    if not isinstance(slate, ActionSlate):
        raise BioasqP1TypedCoreError("slate is invalid")
    return AFormExample(
        predicted_bucket=slate.predicted_bucket,
        utility_vector=tuple(utility_vector),
    )


@dataclass(frozen=True, slots=True)
class RecipeEvidence:
    predicted_bucket: int
    recipe_id: str
    support_count: int
    positive_count: int
    negative_count: int
    total_delta: int
    shrunken_mean_delta: Fraction
    qualified: bool

    def __post_init__(self) -> None:
        if (
            type(self.predicted_bucket) is not int
            or self.predicted_bucket not in PREDICTED_BUCKETS
            or self.recipe_id not in TYPED_RECIPE_IDS
            or type(self.support_count) is not int
            or self.support_count < 0
            or type(self.positive_count) is not int
            or not 0 <= self.positive_count <= self.support_count
            or type(self.negative_count) is not int
            or not 0 <= self.negative_count <= self.support_count
            or self.positive_count + self.negative_count
            > self.support_count
            or type(self.total_delta) is not int
            or not isinstance(self.shrunken_mean_delta, Fraction)
        ):
            raise BioasqP1TypedCoreError("E1 evidence is malformed")
        expected_shrinkage = Fraction(
            self.total_delta,
            self.support_count + SHRINKAGE_PSEUDOCOUNT,
        )
        expected_qualified = (
            self.support_count >= MIN_BUCKET_SUPPORT
            and self.positive_count - self.negative_count
            >= MIN_NET_POSITIVE_MARGIN_COUNT
            and self.total_delta > 0
            and expected_shrinkage > 0
        )
        if (
            self.shrunken_mean_delta != expected_shrinkage
            or self.qualified != expected_qualified
        ):
            raise BioasqP1TypedCoreError(
                "E1 evidence rule drifted"
            )

    def payload(self) -> dict[str, object]:
        return {
            "negative_count": self.negative_count,
            "net_positive_margin_count": (
                self.positive_count - self.negative_count
            ),
            "positive_count": self.positive_count,
            "predicted_bucket": self.predicted_bucket,
            "qualified": self.qualified,
            "recipe_id": self.recipe_id,
            "shrunken_mean_delta": [
                self.shrunken_mean_delta.numerator,
                self.shrunken_mean_delta.denominator,
            ],
            "support_count": self.support_count,
            "total_delta": self.total_delta,
        }


@dataclass(frozen=True, slots=True)
class BucketRule:
    predicted_bucket: int
    selected_recipe_id: str
    fallback_reason: str
    evidence: tuple[RecipeEvidence, ...]

    def __post_init__(self) -> None:
        if (
            type(self.predicted_bucket) is not int
            or self.predicted_bucket not in PREDICTED_BUCKETS
            or tuple(
                (row.predicted_bucket, row.recipe_id)
                for row in self.evidence
            )
            != tuple(
                (self.predicted_bucket, recipe_id)
                for recipe_id in TYPED_RECIPE_IDS
            )
        ):
            raise BioasqP1TypedCoreError(
                "bucket evidence registry drifted"
            )
        qualified = [row for row in self.evidence if row.qualified]
        if not qualified:
            expected_recipe = E0_RECIPE_ID
            expected_reason = "no_qualified_recipe"
        else:
            maximum = max(
                row.shrunken_mean_delta for row in qualified
            )
            winners = [
                row for row in qualified
                if row.shrunken_mean_delta == maximum
            ]
            if len(winners) == 1:
                expected_recipe = winners[0].recipe_id
                expected_reason = "selected"
            else:
                expected_recipe = E0_RECIPE_ID
                expected_reason = "tie_to_e0"
        if (
            self.selected_recipe_id != expected_recipe
            or self.fallback_reason != expected_reason
        ):
            raise BioasqP1TypedCoreError(
                "bucket selection rule drifted"
            )

    @property
    def support_count(self) -> int:
        return self.evidence[0].support_count

    def payload(self) -> dict[str, object]:
        return {
            "evidence": [row.payload() for row in self.evidence],
            "fallback_reason": self.fallback_reason,
            "predicted_bucket": self.predicted_bucket,
            "selected_recipe_id": self.selected_recipe_id,
            "support_count": self.support_count,
        }


@dataclass(frozen=True, slots=True)
class E1Program:
    rules: tuple[BucketRule, ...]
    training_item_count: int
    training_stage: str = "A_form"

    def __post_init__(self) -> None:
        if (
            tuple(rule.predicted_bucket for rule in self.rules)
            != PREDICTED_BUCKETS
            or type(self.training_item_count) is not int
            or self.training_item_count < 0
            or self.training_stage != "A_form"
            or sum(rule.support_count for rule in self.rules)
            != self.training_item_count
        ):
            raise BioasqP1TypedCoreError("E1 program is malformed")

    def rule(self, predicted_bucket: int) -> BucketRule:
        if (
            type(predicted_bucket) is not int
            or predicted_bucket not in PREDICTED_BUCKETS
        ):
            raise BioasqP1TypedCoreError("predicted bucket is invalid")
        return self.rules[PREDICTED_BUCKETS.index(predicted_bucket)]

    def body_payload(self) -> dict[str, object]:
        return {
            "e0_recipe_id": E0_RECIPE_ID,
            "minimum_bucket_support": MIN_BUCKET_SUPPORT,
            "minimum_net_positive_margin_count": (
                MIN_NET_POSITIVE_MARGIN_COUNT
            ),
            "predicted_buckets": list(PREDICTED_BUCKETS),
            "recipe_ids": list(RECIPE_IDS),
            "rules": [rule.payload() for rule in self.rules],
            "schema": f"{VERSION}_E1_bucket_program",
            "shrinkage_pseudocount": SHRINKAGE_PSEUDOCOUNT,
            "study_id": STUDY_ID,
            "training_item_count": self.training_item_count,
            "training_stage": self.training_stage,
            "typed_recipe_ids": list(TYPED_RECIPE_IDS),
            "version": VERSION,
        }

    @property
    def program_sha256(self) -> str:
        return stable_hash(self.body_payload())

    def payload(self) -> dict[str, object]:
        body = self.body_payload()
        return {**body, "self_sha256": stable_hash(body)}


def fit_e1(examples: Sequence[AFormExample]) -> E1Program:
    """Fit one immutable typed-recipe rule per predicted A_form bucket."""

    if isinstance(examples, (str, bytes)):
        raise BioasqP1TypedCoreError("A_form examples are invalid")
    checked = tuple(examples)
    if any(not isinstance(row, AFormExample) for row in checked):
        raise BioasqP1TypedCoreError(
            "A_form contains a non-example"
        )
    grouped: defaultdict[int, list[AFormExample]] = defaultdict(list)
    for row in checked:
        grouped[row.predicted_bucket].append(row)
    e0_index = RECIPE_IDS.index(E0_RECIPE_ID)
    rules: list[BucketRule] = []
    for bucket in PREDICTED_BUCKETS:
        rows = grouped[bucket]
        evidence: list[RecipeEvidence] = []
        for recipe_id in TYPED_RECIPE_IDS:
            recipe_index = RECIPE_IDS.index(recipe_id)
            deltas = [
                row.utility_vector[recipe_index]
                - row.utility_vector[e0_index]
                for row in rows
            ]
            support = len(rows)
            positive = sum(delta > 0 for delta in deltas)
            negative = sum(delta < 0 for delta in deltas)
            total = sum(deltas)
            shrunken = Fraction(
                total,
                support + SHRINKAGE_PSEUDOCOUNT,
            )
            qualified = (
                support >= MIN_BUCKET_SUPPORT
                and positive - negative
                >= MIN_NET_POSITIVE_MARGIN_COUNT
                and total > 0
                and shrunken > 0
            )
            evidence.append(
                RecipeEvidence(
                    predicted_bucket=bucket,
                    recipe_id=recipe_id,
                    support_count=support,
                    positive_count=positive,
                    negative_count=negative,
                    total_delta=total,
                    shrunken_mean_delta=shrunken,
                    qualified=qualified,
                )
            )
        qualified_rows = [row for row in evidence if row.qualified]
        if not qualified_rows:
            selected_recipe = E0_RECIPE_ID
            fallback_reason = "no_qualified_recipe"
        else:
            maximum = max(
                row.shrunken_mean_delta for row in qualified_rows
            )
            winners = [
                row for row in qualified_rows
                if row.shrunken_mean_delta == maximum
            ]
            if len(winners) == 1:
                selected_recipe = winners[0].recipe_id
                fallback_reason = "selected"
            else:
                selected_recipe = E0_RECIPE_ID
                fallback_reason = "tie_to_e0"
        rules.append(
            BucketRule(
                predicted_bucket=bucket,
                selected_recipe_id=selected_recipe,
                fallback_reason=fallback_reason,
                evidence=tuple(evidence),
            )
        )
    return E1Program(
        rules=tuple(rules),
        training_item_count=len(checked),
    )


_E0_PROGRAM_SHA256 = stable_hash(
    {
        "e0_recipe_id": E0_RECIPE_ID,
        "rrf_k": RRF_K,
        "rrf_score_names": ["raw_ce", "dense_base"],
        "study_id": STUDY_ID,
        "version": VERSION,
    }
)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    evaluator_id: str
    stage: str
    predicted_bucket: int
    selected_recipe_id: str
    top5_ordinals: tuple[int, ...]
    fallback_to_e0: bool
    program_sha256: str
    action_behavior_digest: str
    decision_digest: str

    def __post_init__(self) -> None:
        if (
            self.evaluator_id not in {"E0", "E1"}
            or self.stage not in POLICY_STAGES
            or type(self.predicted_bucket) is not int
            or self.predicted_bucket not in PREDICTED_BUCKETS
            or self.selected_recipe_id not in RECIPE_IDS
            or len(self.top5_ordinals) != TOP_K
            or len(set(self.top5_ordinals)) != TOP_K
        ):
            raise BioasqP1TypedCoreError(
                "policy decision is malformed"
            )
        if self.evaluator_id == "E0":
            if (
                self.selected_recipe_id != E0_RECIPE_ID
                or self.fallback_to_e0
                or self.program_sha256 != _E0_PROGRAM_SHA256
            ):
                raise BioasqP1TypedCoreError("E0 decision drifted")
        elif self.fallback_to_e0 != (
            self.selected_recipe_id == E0_RECIPE_ID
        ):
            raise BioasqP1TypedCoreError("E1 fallback flag drifted")
        for digest in (
            self.program_sha256,
            self.action_behavior_digest,
            self.decision_digest,
        ):
            if _HEX_SHA256_RE.fullmatch(digest) is None:
                raise BioasqP1TypedCoreError(
                    "policy decision digest is malformed"
                )
        expected = stable_hash(
            {
                "action_behavior_digest": self.action_behavior_digest,
                "evaluator_id": self.evaluator_id,
                "fallback_to_e0": self.fallback_to_e0,
                "predicted_bucket": self.predicted_bucket,
                "program_sha256": self.program_sha256,
                "selected_recipe_id": self.selected_recipe_id,
                "stage": self.stage,
                "study_id": STUDY_ID,
                "top5_ordinals": list(self.top5_ordinals),
                "version": VERSION,
            }
        )
        if self.decision_digest != expected:
            raise BioasqP1TypedCoreError(
                "policy decision digest drifted"
            )


def _make_decision(
    *,
    evaluator_id: str,
    stage: str,
    slate: ActionSlate,
    selected_recipe_id: str,
    fallback_to_e0: bool,
    program_sha256: str,
) -> PolicyDecision:
    if stage not in POLICY_STAGES:
        raise BioasqP1TypedCoreError("policy stage is invalid")
    action = slate.action(selected_recipe_id)
    payload = {
        "action_behavior_digest": action.behavior_digest,
        "evaluator_id": evaluator_id,
        "fallback_to_e0": fallback_to_e0,
        "predicted_bucket": slate.predicted_bucket,
        "program_sha256": program_sha256,
        "selected_recipe_id": selected_recipe_id,
        "stage": stage,
        "study_id": STUDY_ID,
        "top5_ordinals": list(action.top5_ordinals),
        "version": VERSION,
    }
    return PolicyDecision(
        evaluator_id=evaluator_id,
        stage=stage,
        predicted_bucket=slate.predicted_bucket,
        selected_recipe_id=selected_recipe_id,
        top5_ordinals=action.top5_ordinals,
        fallback_to_e0=fallback_to_e0,
        program_sha256=program_sha256,
        action_behavior_digest=action.behavior_digest,
        decision_digest=stable_hash(payload),
    )


def apply_e0(slate: ActionSlate, *, stage: str) -> PolicyDecision:
    if not isinstance(slate, ActionSlate):
        raise BioasqP1TypedCoreError("slate is invalid")
    return _make_decision(
        evaluator_id="E0",
        stage=stage,
        slate=slate,
        selected_recipe_id=E0_RECIPE_ID,
        fallback_to_e0=False,
        program_sha256=_E0_PROGRAM_SHA256,
    )


def apply_e1(
    program: E1Program,
    slate: ActionSlate,
    *,
    stage: str,
) -> PolicyDecision:
    if not isinstance(program, E1Program) or not isinstance(
        slate, ActionSlate
    ):
        raise BioasqP1TypedCoreError(
            "E1 application input is invalid"
        )
    rule = program.rule(slate.predicted_bucket)
    return _make_decision(
        evaluator_id="E1",
        stage=stage,
        slate=slate,
        selected_recipe_id=rule.selected_recipe_id,
        fallback_to_e0=(
            rule.selected_recipe_id == E0_RECIPE_ID
        ),
        program_sha256=program.program_sha256,
    )


@dataclass(frozen=True, slots=True)
class BehaviorSummary:
    evaluator_id: str
    stage: str
    program_sha256: str
    item_count: int
    fallback_count: int
    bucket_recipe_counts: tuple[tuple[int, str, int], ...]
    decision_set_sha256: str

    def __post_init__(self) -> None:
        identities = tuple(
            (bucket, recipe_id)
            for bucket, recipe_id, _count in self.bucket_recipe_counts
        )
        if (
            self.evaluator_id != "E1"
            or self.stage not in POLICY_STAGES
            or _HEX_SHA256_RE.fullmatch(self.program_sha256) is None
            or _HEX_SHA256_RE.fullmatch(
                self.decision_set_sha256
            )
            is None
            or type(self.item_count) is not int
            or self.item_count <= 0
            or type(self.fallback_count) is not int
            or not 0 <= self.fallback_count <= self.item_count
            or identities != tuple(sorted(identities))
            or any(
                bucket not in PREDICTED_BUCKETS
                or recipe_id not in RECIPE_IDS
                or type(count) is not int
                or count <= 0
                for bucket, recipe_id, count
                in self.bucket_recipe_counts
            )
            or sum(
                count for _bucket, _recipe_id, count
                in self.bucket_recipe_counts
            )
            != self.item_count
        ):
            raise BioasqP1TypedCoreError(
                "behavior summary is malformed"
            )

    def body_payload(self) -> dict[str, object]:
        return {
            "bucket_recipe_counts": [
                {
                    "count": count,
                    "predicted_bucket": bucket,
                    "recipe_id": recipe_id,
                }
                for bucket, recipe_id, count
                in self.bucket_recipe_counts
            ],
            "decision_set_sha256": self.decision_set_sha256,
            "evaluator_id": self.evaluator_id,
            "fallback_count": self.fallback_count,
            "item_count": self.item_count,
            "program_sha256": self.program_sha256,
            "schema": f"{VERSION}_behavior_summary",
            "stage": self.stage,
            "study_id": STUDY_ID,
            "version": VERSION,
        }

    def payload(self) -> dict[str, object]:
        body = self.body_payload()
        return {**body, "self_sha256": stable_hash(body)}


def summarize_e1_behavior(
    program: E1Program,
    decisions: Sequence[PolicyDecision],
    *,
    stage: str,
) -> BehaviorSummary:
    if not isinstance(program, E1Program) or stage not in POLICY_STAGES:
        raise BioasqP1TypedCoreError(
            "behavior summary input is invalid"
        )
    if isinstance(decisions, (str, bytes)) or not decisions:
        raise BioasqP1TypedCoreError("behavior decisions are invalid")
    checked = tuple(decisions)
    if any(
        not isinstance(decision, PolicyDecision)
        or decision.evaluator_id != "E1"
        or decision.stage != stage
        or decision.program_sha256 != program.program_sha256
        for decision in checked
    ):
        raise BioasqP1TypedCoreError(
            "behavior decisions do not share the frozen program"
        )
    counts = Counter(
        (decision.predicted_bucket, decision.selected_recipe_id)
        for decision in checked
    )
    return BehaviorSummary(
        evaluator_id="E1",
        stage=stage,
        program_sha256=program.program_sha256,
        item_count=len(checked),
        fallback_count=sum(
            decision.fallback_to_e0 for decision in checked
        ),
        bucket_recipe_counts=tuple(
            (bucket, recipe_id, count)
            for (bucket, recipe_id), count in sorted(counts.items())
        ),
        decision_set_sha256=stable_hash(
            sorted(decision.decision_digest for decision in checked)
        ),
    )


__all__ = [
    "ActionSlate",
    "AFormExample",
    "B0_CLAIM",
    "B1_ENTITY",
    "B2_LIST",
    "B3_ASPECT",
    "BUCKET_NAMES",
    "BehaviorSummary",
    "BioasqP1TypedCoreError",
    "BucketRule",
    "CORPUS_SIZE",
    "E0_RECIPE_ID",
    "E1Program",
    "EvidenceAction",
    "LIST_DIVERSITY_CANDIDATE_PREFIX",
    "LIST_DIVERSITY_NOVELTY_WEIGHT",
    "LIST_DIVERSITY_RELEVANCE_WEIGHT",
    "MAX_PASSAGE_CHARACTERS",
    "MAX_QUERY_CHARACTERS",
    "MAX_SCORE_ABS",
    "MAX_UTILITY",
    "MIN_BUCKET_SUPPORT",
    "MIN_NET_POSITIVE_MARGIN_COUNT",
    "POLICY_STAGES",
    "PREDICTED_BUCKETS",
    "PUBLIC_PASSAGE_FIELDS",
    "Passage",
    "PolicyDecision",
    "QuestionStructure",
    "R0_GLOBAL_RAW_DENSE_RRF",
    "R1_CLAIM_BALANCE",
    "R2_ENTITY_FOCUS",
    "R3_LIST_DIVERSITY",
    "R4_ASPECT_COVERAGE",
    "RECIPE_IDS",
    "RecipeEvidence",
    "SCALE",
    "SCORE_NAMES",
    "SHRINKAGE_PSEUDOCOUNT",
    "STUDY_ID",
    "ScoreQueryBundle",
    "TOP_K",
    "TYPED_RECIPE_IDS",
    "VERSION",
    "apply_e0",
    "apply_e1",
    "build_action_slate",
    "canonical_bytes",
    "fit_e1",
    "make_aform_example",
    "normalize_text",
    "passage_from_public_fields",
    "passage_public_payload",
    "predict_question_structure",
    "serialize_passage",
    "serialize_passage_bytes",
    "serialize_query_for_score",
    "serialize_score_queries",
    "stable_hash",
    "summarize_e1_behavior",
    "validate_query_text",
]
