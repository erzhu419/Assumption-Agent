"""Label-free QuAC dialogue-to-evidence action adapter.

The adapter is intentionally downstream of source acquisition and MiniLM
inference.  It accepts only an anonymized block of canonical evidence windows,
the current question followed by at most three preceding questions, and
embeddings bound to this module's exact text serializations.  It cannot accept
an item identity, source split, dialogue family, answer, qrel, utility, or a
HippoRAG output.

The resulting :class:`~quac_rjmc_evaluator_v1.RelationalGraph` is the complete
fixed action domain for RJMC-V1:

* RAW is full-dialogue-query dense top five over the complete block;
* one direct top-one anchor is formed for every available dialogue slot;
* at most two typed frontier edges add one outside endpoint each; and
* every one- and two-replacement state over the retained graph is therefore
  expressible by the frozen evaluator.

All ranking decisions are made after ties-to-even microquantization.  Vector
coordinates are first converted to IEEE-754 float32, then L2-normalized, and
cosines are accumulated deterministically with :func:`math.fsum`.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from numbers import Real
import re
import struct
import unicodedata
from typing import Any, Mapping, Sequence

from . import quac_rjmc_evaluator_v1 as evaluator


VERSION = "quac_p1_action_adapter_v1"
SCHEMA = VERSION
MINILM_EMBEDDING_DIMENSION = 384
MICRO_SCALE = 1_000_000
MAX_DIALOGUE_TURNS = 4
MAX_TYPED_EXPANSIONS = 2
MAX_GRAPH_UNITS = 11
MAX_REPLACEMENT_CANDIDATES = 6
MAX_COMPLETE_STATES = 181
ENTITY_MIN_DOCUMENT_FREQUENCY = 2
ENTITY_MAX_DOCUMENT_FREQUENCY = 16
ENTITY_MIN_CODEPOINTS = 3
TURN_MARKERS = (
    "TURN_0_CURRENT:",
    "TURN_1_PREVIOUS:",
    "TURN_2_PREVIOUS:",
    "TURN_3_PREVIOUS:",
)
TURN_RECENCY_MICRO = (
    MICRO_SCALE,
    750_000,
    500_000,
    250_000,
)

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_UNICODE_WORD = re.compile(r"\w+", flags=re.UNICODE)


class QuacP1ActionAdapterError(ValueError):
    """The label-free action input or frozen construction contract drifted."""


def _strict_text(value: object, *, field: str, nonempty: bool) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise QuacP1ActionAdapterError(f"{field} must be exact text")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise QuacP1ActionAdapterError(
            f"{field} contains invalid Unicode"
        ) from exc
    if nonempty and not value:
        raise QuacP1ActionAdapterError(f"{field} must be nonempty")
    return value


def _opaque_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise QuacP1ActionAdapterError(
            f"{field} must be an opaque lowercase SHA-256 token"
        )
    return value


def _rows(value: object, *, field: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise QuacP1ActionAdapterError(f"{field} must be a sequence")
    return tuple(value)


def _float32(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise QuacP1ActionAdapterError(f"{field} must be a finite real")
    result = float(value)
    if not math.isfinite(result):
        raise QuacP1ActionAdapterError(f"{field} must be a finite real")
    try:
        result = struct.unpack("!f", struct.pack("!f", result))[0]
    except OverflowError as exc:
        raise QuacP1ActionAdapterError(
            f"{field} is outside the float32 range"
        ) from exc
    if not math.isfinite(result):
        raise QuacP1ActionAdapterError(
            f"{field} is outside the float32 range"
        )
    return 0.0 if result == 0.0 else result


def _normalized_float32_vector(value: object) -> tuple[float, ...]:
    coordinates = _rows(value, field="MiniLM embedding")
    if len(coordinates) != MINILM_EMBEDDING_DIMENSION:
        raise QuacP1ActionAdapterError(
            "MiniLM embedding dimension drifted"
        )
    vector = tuple(
        _float32(coordinate, field="MiniLM embedding coordinate")
        for coordinate in coordinates
    )
    squared_norm = math.fsum(coordinate * coordinate for coordinate in vector)
    if not math.isfinite(squared_norm) or squared_norm <= 0.0:
        raise QuacP1ActionAdapterError(
            "MiniLM embedding has zero or nonfinite norm"
        )
    norm = math.sqrt(squared_norm)
    normalized = tuple(
        _float32(coordinate / norm, field="normalized MiniLM coordinate")
        for coordinate in vector
    )
    normalized_norm = math.fsum(
        coordinate * coordinate for coordinate in normalized
    )
    if not math.isfinite(normalized_norm) or normalized_norm <= 0.0:
        raise QuacP1ActionAdapterError(
            "normalized MiniLM embedding is invalid"
        )
    return normalized


def microquantize(value: object) -> int:
    """Quantize one cosine/strength with Python ties-to-even rounding."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise QuacP1ActionAdapterError("microquantized value must be finite")
    exact = float(value)
    if not math.isfinite(exact):
        raise QuacP1ActionAdapterError("microquantized value must be finite")
    if exact < -1.0 - 1.0e-6 or exact > 1.0 + 1.0e-6:
        raise QuacP1ActionAdapterError(
            "microquantized cosine/strength escaped [-1, 1]"
        )
    exact = min(1.0, max(-1.0, exact))
    return int(round(exact * MICRO_SCALE))


def _cosine_micro(
    left: Sequence[float],
    right: Sequence[float],
) -> int:
    if len(left) != len(right):
        raise QuacP1ActionAdapterError("MiniLM vector widths drifted")
    cosine = math.fsum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )
    return microquantize(cosine)


@dataclass(frozen=True)
class BlockDocument:
    """One anonymized canonical evidence window in a fixed block corpus."""

    unit_id: str
    context_id: str
    title: str
    section_title: str
    context_window_ordinal: int
    text: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "unit_id",
            _opaque_id(self.unit_id, field="unit ID"),
        )
        object.__setattr__(
            self,
            "context_id",
            _opaque_id(self.context_id, field="context ID"),
        )
        object.__setattr__(
            self,
            "title",
            _strict_text(self.title, field="title", nonempty=False),
        )
        object.__setattr__(
            self,
            "section_title",
            _strict_text(
                self.section_title,
                field="section title",
                nonempty=False,
            ),
        )
        if (
            type(self.context_window_ordinal) is not int
            or not 0 <= self.context_window_ordinal < 2**31
        ):
            raise QuacP1ActionAdapterError(
                "context window ordinal must be an integer in [0, 2^31)"
            )
        object.__setattr__(
            self,
            "text",
            _strict_text(self.text, field="window text", nonempty=True),
        )


@dataclass(frozen=True)
class QuestionTurn:
    """One question, positioned by recent-first tuple order.

    Slot zero is current; slots one through three are previous turns.  No
    source turn coordinate or native question-to-context association exists.
    """

    question_text: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "question_text",
            _strict_text(
                self.question_text,
                field="question text",
                nonempty=True,
            ),
        )


@dataclass(frozen=True)
class MiniLmEmbedding:
    """One normalized embedding bound only to an exact serialization hash."""

    serialization_sha256: str
    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.serialization_sha256, str)
            or _HEX64.fullmatch(self.serialization_sha256) is None
        ):
            raise QuacP1ActionAdapterError(
                "embedding serialization SHA-256 is invalid"
            )
        object.__setattr__(
            self,
            "vector",
            _normalized_float32_vector(self.vector),
        )


def official_inner_unit_text(document: BlockDocument) -> str:
    """Return the exact inner text supplied to the official block contract."""

    if not isinstance(document, BlockDocument):
        raise QuacP1ActionAdapterError(
            "evidence serialization requires BlockDocument"
        )
    return (
        f"TITLE:{document.title}\n"
        f"SECTION:{document.section_title}\n"
        f"TEXT:{document.text}"
    )


def serialize_evidence_unit(document: BlockDocument) -> str:
    """Return the unified canonical-JSON MiniLM/HippoRAG document."""

    inner = official_inner_unit_text(document)
    try:
        raw = json.dumps(
            {
                "text": inner,
                "title": f"QUAC_EVIDENCE_UNIT_{document.unit_id}",
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise QuacP1ActionAdapterError(
            "evidence unit is not canonical-JSON serializable"
        ) from exc
    return (raw + b"\n").decode("ascii")


def serialize_turn_query(turn: QuestionTurn, *, slot: int) -> str:
    """Return one fixed per-turn Agent anchor query."""

    if not isinstance(turn, QuestionTurn):
        raise QuacP1ActionAdapterError(
            "turn serialization requires QuestionTurn"
        )
    if type(slot) is not int or not 0 <= slot < MAX_DIALOGUE_TURNS:
        raise QuacP1ActionAdapterError("dialogue slot is outside [0, 4)")
    return f"{TURN_MARKERS[slot]}\n{turn.question_text}"


def serialize_full_query(turns: Sequence[QuestionTurn]) -> str:
    """Return current, previous-1, previous-2, previous-3 marker order."""

    rows = _rows(turns, field="question turns")
    if not 1 <= len(rows) <= MAX_DIALOGUE_TURNS:
        raise QuacP1ActionAdapterError(
            "question turns must contain current plus at most three previous"
        )
    if any(not isinstance(row, QuestionTurn) for row in rows):
        raise QuacP1ActionAdapterError(
            "question turns must contain QuestionTurn values"
        )
    return "\n".join(
        serialize_turn_query(row, slot=slot)
        for slot, row in enumerate(rows)
    )


def _serialization_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def required_embedding_serializations(
    documents: Sequence[BlockDocument],
    question_turns: Sequence[QuestionTurn],
) -> tuple[tuple[str, str], ...]:
    """Return unique ``(sha256, text)`` MiniLM requests in frozen order.

    Evidence windows use canonical unit-ID order, followed by the full query
    and each available per-turn query.  With only a current turn, the full and
    per-turn queries are identical and are encoded once.
    """

    document_rows = _rows(documents, field="block documents")
    turn_rows = _rows(question_turns, field="question turns")
    if len(document_rows) < evaluator.TOP_K:
        raise QuacP1ActionAdapterError(
            "block corpus must contain at least five evidence windows"
        )
    if any(not isinstance(row, BlockDocument) for row in document_rows):
        raise QuacP1ActionAdapterError(
            "block documents must contain BlockDocument values"
        )
    if len({row.unit_id for row in document_rows}) != len(document_rows):
        raise QuacP1ActionAdapterError("block corpus contains duplicate unit IDs")
    if len(
        {
            (row.context_id, row.context_window_ordinal)
            for row in document_rows
        }
    ) != len(document_rows):
        raise QuacP1ActionAdapterError(
            "block corpus contains duplicate context-window coordinates"
        )
    full_query = serialize_full_query(turn_rows)
    serialized = [
        serialize_evidence_unit(row)
        for row in sorted(document_rows, key=lambda row: row.unit_id)
    ]
    serialized.append(full_query)
    serialized.extend(
        serialize_turn_query(row, slot=slot)
        for slot, row in enumerate(turn_rows)
    )
    output: list[tuple[str, str]] = []
    observed: dict[str, str] = {}
    for text in serialized:
        digest = _serialization_sha256(text)
        if digest in observed:
            if observed[digest] != text:
                raise QuacP1ActionAdapterError(
                    "embedding serialization SHA-256 collision"
                )
            continue
        observed[digest] = text
        output.append((digest, text))
    return tuple(output)


@dataclass(frozen=True)
class ActionAdapterInput:
    """Complete label-free input for one fixed action construction."""

    documents: tuple[BlockDocument, ...]
    question_turns: tuple[QuestionTurn, ...]
    minilm_embeddings: tuple[MiniLmEmbedding, ...]

    def __post_init__(self) -> None:
        documents = _rows(self.documents, field="block documents")
        turns = _rows(self.question_turns, field="question turns")
        embeddings = _rows(self.minilm_embeddings, field="MiniLM embeddings")
        if any(not isinstance(row, BlockDocument) for row in documents):
            raise QuacP1ActionAdapterError(
                "block documents must contain BlockDocument values"
            )
        if any(not isinstance(row, QuestionTurn) for row in turns):
            raise QuacP1ActionAdapterError(
                "question turns must contain QuestionTurn values"
            )
        if any(not isinstance(row, MiniLmEmbedding) for row in embeddings):
            raise QuacP1ActionAdapterError(
                "MiniLM embeddings must contain MiniLmEmbedding values"
            )
        requests = required_embedding_serializations(documents, turns)
        expected_hashes = {digest for digest, _text in requests}
        supplied_hashes = tuple(row.serialization_sha256 for row in embeddings)
        if len(set(supplied_hashes)) != len(supplied_hashes):
            raise QuacP1ActionAdapterError(
                "MiniLM embedding serialization hashes are duplicated"
            )
        if set(supplied_hashes) != expected_hashes:
            raise QuacP1ActionAdapterError(
                "MiniLM embeddings do not exactly bind every frozen serialization"
            )
        object.__setattr__(self, "documents", tuple(documents))
        object.__setattr__(self, "question_turns", tuple(turns))
        object.__setattr__(self, "minilm_embeddings", tuple(embeddings))


def proper_name_keys(text: str) -> tuple[str, ...]:
    """Extract frozen NFKC proper-name keys from one evidence window.

    A token qualifies when its first Unicode cased character is uppercase.
    Qualifying tokens form one maximal span only when separated by Unicode
    whitespace; punctuation or a nonqualifying word closes the span.  Keys are
    casefolded, whitespace-collapsed, and must contain at least three
    codepoints.
    """

    exact = _strict_text(text, field="entity parser text", nonempty=False)
    normalized = unicodedata.normalize("NFKC", exact)
    tokens = tuple(_UNICODE_WORD.finditer(normalized))

    def qualifies(token: str) -> bool:
        for character in token:
            if character.lower() != character.upper():
                return character.isupper()
        return False

    spans: list[tuple[str, ...]] = []
    active: list[str] = []
    prior_end: int | None = None
    for match in tokens:
        token = match.group(0)
        whitespace_adjacent = (
            prior_end is None
            or normalized[prior_end : match.start()].isspace()
        )
        if qualifies(token):
            if active and not whitespace_adjacent:
                spans.append(tuple(active))
                active = []
            active.append(token)
        else:
            if active:
                spans.append(tuple(active))
                active = []
        prior_end = match.end()
    if active:
        spans.append(tuple(active))

    keys = {
        " ".join(span).casefold()
        for span in spans
        if len(" ".join(span).casefold()) >= ENTITY_MIN_CODEPOINTS
    }
    return tuple(sorted(keys))


@dataclass(frozen=True)
class _MicroEdge:
    left: str
    right: str
    relation: str
    strength_micro: int

    def __post_init__(self) -> None:
        if not self.left < self.right:
            raise QuacP1ActionAdapterError(
                "internal typed edge endpoints are not canonical"
            )
        if self.relation not in evaluator.RELATION_TYPES:
            raise QuacP1ActionAdapterError(
                "internal typed edge relation drifted"
            )
        if (
            type(self.strength_micro) is not int
            or not 0 < self.strength_micro <= MICRO_SCALE
        ):
            raise QuacP1ActionAdapterError(
                "internal typed edge strength drifted"
            )


def _canonical_pair(left: str, right: str) -> tuple[str, str]:
    if left == right:
        raise QuacP1ActionAdapterError("typed edge cannot be a self-loop")
    return (left, right) if left < right else (right, left)


def _frozen_typed_edges(
    documents: Sequence[BlockDocument],
) -> tuple[_MicroEdge, ...]:
    by_id = {row.unit_id: row for row in documents}
    edges: dict[tuple[str, str, str], int] = {}

    def add(
        left: str,
        right: str,
        relation: str,
        strength_micro: int,
    ) -> None:
        canonical_left, canonical_right = _canonical_pair(left, right)
        key = (canonical_left, canonical_right, relation)
        if key in edges:
            raise QuacP1ActionAdapterError(
                "typed topology generated a duplicate relation edge"
            )
        edges[key] = strength_micro

    by_context: dict[str, dict[int, str]] = defaultdict(dict)
    for row in documents:
        by_context[row.context_id][row.context_window_ordinal] = row.unit_id
    for context_id in sorted(by_context):
        ordinal_map = by_context[context_id]
        for ordinal in sorted(ordinal_map):
            if ordinal + 1 in ordinal_map:
                add(
                    ordinal_map[ordinal],
                    ordinal_map[ordinal + 1],
                    "adjacent_window",
                    MICRO_SCALE,
                )

    by_section: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in documents:
        by_section[(row.title, row.section_title)].append(row.unit_id)
    for section_key in sorted(by_section):
        members = sorted(by_section[section_key])
        for left, right in itertools.combinations(members, 2):
            distance = abs(
                by_id[left].context_window_ordinal
                - by_id[right].context_window_ordinal
            )
            add(
                left,
                right,
                "same_section",
                microquantize(1.0 / (1.0 + distance)),
            )

    keys_by_unit = {
        row.unit_id: set(proper_name_keys(row.text))
        for row in documents
    }
    document_frequency = Counter(
        key
        for unit_keys in keys_by_unit.values()
        for key in unit_keys
    )
    units_by_key: dict[str, list[str]] = defaultdict(list)
    for unit_id in sorted(keys_by_unit):
        for key in sorted(keys_by_unit[unit_id]):
            if (
                ENTITY_MIN_DOCUMENT_FREQUENCY
                <= document_frequency[key]
                <= ENTITY_MAX_DOCUMENT_FREQUENCY
            ):
                units_by_key[key].append(unit_id)
    shared_key_count: Counter[tuple[str, str]] = Counter()
    for key in sorted(units_by_key):
        for left, right in itertools.combinations(
            sorted(units_by_key[key]),
            2,
        ):
            shared_key_count[(left, right)] += 1
    for (left, right), count in sorted(shared_key_count.items()):
        add(
            left,
            right,
            "entity_chain",
            microquantize(min(1.0, count / 4.0)),
        )

    return tuple(
        _MicroEdge(left, right, relation, strength)
        for (left, right, relation), strength in sorted(
            edges.items(),
            key=lambda row: (
                row[0][0],
                row[0][1],
                evaluator.RELATION_TYPES.index(row[0][2]),
            ),
        )
    )


def _distance_to_direct_anchor(
    unit_ids: Sequence[str],
    edges: Sequence[_MicroEdge],
    direct_anchors: Sequence[str],
) -> Mapping[str, int]:
    adjacency = {unit_id: set() for unit_id in unit_ids}
    for edge in edges:
        adjacency[edge.left].add(edge.right)
        adjacency[edge.right].add(edge.left)
    distance: dict[str, int] = {}
    queue: deque[str] = deque()
    for unit_id in direct_anchors:
        if unit_id not in distance:
            distance[unit_id] = 0
            queue.append(unit_id)
    while queue:
        current = queue.popleft()
        if distance[current] >= 2:
            continue
        for neighbor in sorted(adjacency[current]):
            if neighbor not in distance:
                distance[neighbor] = distance[current] + 1
                queue.append(neighbor)
    return distance


@dataclass(frozen=True)
class ActionAdapterResult:
    """Content-free fixed graph and RAW set accepted by RJMC-V1.

    ``direct_anchor_unit_ids`` is per-turn, recent-first, and deliberately
    preserves duplicates.  Only the construction seed deduplicates unit IDs;
    preserving this tuple is what keeps the four facet-bit positions exact
    when two dialogue turns share one direct anchor.
    """

    graph: evaluator.RelationalGraph
    raw_top5: tuple[str, ...]
    direct_anchor_unit_ids: tuple[str, ...]
    input_serialization_set_sha256: str
    complete_state_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.graph, evaluator.RelationalGraph):
            raise QuacP1ActionAdapterError(
                "adapter result graph is not an RJMC graph"
            )
        if any(
            _HEX64.fullmatch(unit_id) is None
            for unit_id in self.graph.unit_ids
        ):
            raise QuacP1ActionAdapterError(
                "adapter result graph exposed a nonopaque unit ID"
            )
        if (
            len(self.raw_top5) != evaluator.TOP_K
            or len(set(self.raw_top5)) != evaluator.TOP_K
            or any(row not in self.graph.unit_ids for row in self.raw_top5)
        ):
            raise QuacP1ActionAdapterError(
                "adapter result RAW top five is invalid"
            )
        if (
            not 1 <= len(self.direct_anchor_unit_ids) <= MAX_DIALOGUE_TURNS
            or any(
                row not in self.graph.unit_ids
                for row in self.direct_anchor_unit_ids
            )
        ):
            raise QuacP1ActionAdapterError(
                "adapter result direct anchors are invalid"
            )
        if (
            not isinstance(self.input_serialization_set_sha256, str)
            or _HEX64.fullmatch(self.input_serialization_set_sha256) is None
        ):
            raise QuacP1ActionAdapterError(
                "adapter result input serialization commitment is invalid"
            )
        candidate_count = len(
            set(self.graph.unit_ids).difference(self.raw_top5)
        )
        exact_state_count = evaluator.complete_state_count(candidate_count)
        if (
            self.complete_state_count != exact_state_count
            or self.complete_state_count > MAX_COMPLETE_STATES
        ):
            raise QuacP1ActionAdapterError(
                "adapter result complete state count drifted"
            )


def _embedding_lookup(
    action_input: ActionAdapterInput,
) -> Mapping[str, tuple[float, ...]]:
    return {
        row.serialization_sha256: row.vector
        for row in action_input.minilm_embeddings
    }


def build_action_graph(
    action_input: ActionAdapterInput,
) -> ActionAdapterResult:
    """Build the one frozen, label-free QuAC action graph."""

    if not isinstance(action_input, ActionAdapterInput):
        raise QuacP1ActionAdapterError(
            "action graph requires ActionAdapterInput"
        )
    documents = tuple(
        sorted(action_input.documents, key=lambda row: row.unit_id)
    )
    turns = action_input.question_turns
    embeddings = _embedding_lookup(action_input)

    def embedded(serialized: str) -> tuple[float, ...]:
        digest = _serialization_sha256(serialized)
        try:
            return embeddings[digest]
        except KeyError as exc:
            raise QuacP1ActionAdapterError(
                "required MiniLM serialization is unbound"
            ) from exc

    unit_vectors = {
        row.unit_id: embedded(serialize_evidence_unit(row))
        for row in documents
    }
    full_query_vector = embedded(serialize_full_query(turns))
    turn_vectors = tuple(
        embedded(serialize_turn_query(row, slot=slot))
        for slot, row in enumerate(turns)
    )
    full_query_micro = {
        unit_id: _cosine_micro(full_query_vector, vector)
        for unit_id, vector in unit_vectors.items()
    }
    per_turn_micro = tuple(
        {
            unit_id: _cosine_micro(turn_vector, vector)
            for unit_id, vector in unit_vectors.items()
        }
        for turn_vector in turn_vectors
    )

    ranked_full = tuple(
        sorted(
            unit_vectors,
            key=lambda unit_id: (
                -full_query_micro[unit_id],
                unit_id,
            ),
        )
    )
    raw_top5 = ranked_full[: evaluator.TOP_K]
    direct_anchors = tuple(
        min(
            unit_vectors,
            key=lambda unit_id: (
                -turn_scores[unit_id],
                unit_id,
            ),
        )
        for turn_scores in per_turn_micro
    )
    retained = set(raw_top5)
    retained.update(direct_anchors)

    typed_edges = _frozen_typed_edges(documents)
    for _step in range(MAX_TYPED_EXPANSIONS):
        frontier: list[
            tuple[tuple[int, int, int, str, str, str], _MicroEdge, str]
        ] = []
        for edge in typed_edges:
            left_retained = edge.left in retained
            right_retained = edge.right in retained
            if left_retained == right_retained:
                continue
            outside = edge.right if left_retained else edge.left
            frontier.append(
                (
                    (
                        -edge.strength_micro,
                        -full_query_micro[outside],
                        evaluator.RELATION_TYPES.index(edge.relation),
                        edge.left,
                        edge.right,
                        outside,
                    ),
                    edge,
                    outside,
                )
            )
        if not frontier:
            break
        _key, _selected_edge, outside = min(frontier, key=lambda row: row[0])
        retained.add(outside)

    if len(retained) > MAX_GRAPH_UNITS:
        raise QuacP1ActionAdapterError(
            "frozen candidate graph exceeded eleven units"
        )
    candidate_count = len(retained.difference(raw_top5))
    if candidate_count > MAX_REPLACEMENT_CANDIDATES:
        raise QuacP1ActionAdapterError(
            "frozen candidate graph exceeded six replacements"
        )
    state_count = evaluator.complete_state_count(candidate_count)
    if state_count > MAX_COMPLETE_STATES:
        raise QuacP1ActionAdapterError(
            "frozen complete state space exceeded 181 states"
        )

    distance = _distance_to_direct_anchor(
        tuple(unit_vectors),
        typed_edges,
        direct_anchors,
    )
    units: list[evaluator.EvidenceUnit] = []
    for row in documents:
        if row.unit_id not in retained:
            continue
        turn_scores = tuple(
            slot_scores[row.unit_id] for slot_scores in per_turn_micro
        )
        best_turn_micro = max(turn_scores)
        best_slot = next(
            slot
            for slot, score in enumerate(turn_scores)
            if score == best_turn_micro
        )
        topology_distance = distance.get(row.unit_id)
        topology_micro = (
            0
            if topology_distance is None
            else microquantize(1.0 / (1.0 + topology_distance))
        )
        units.append(
            evaluator.EvidenceUnit(
                unit_id=row.unit_id,
                node_features=(
                    full_query_micro[row.unit_id] / MICRO_SCALE,
                    best_turn_micro / MICRO_SCALE,
                    TURN_RECENCY_MICRO[best_slot] / MICRO_SCALE,
                    topology_micro / MICRO_SCALE,
                ),
                dialogue_facets=tuple(
                    int(row.unit_id == anchor)
                    for anchor in direct_anchors
                )
                + (0,) * (MAX_DIALOGUE_TURNS - len(direct_anchors)),
            )
        )
    graph_edges = tuple(
        evaluator.TypedEdge(
            left=edge.left,
            right=edge.right,
            relation=edge.relation,
            strength=edge.strength_micro / MICRO_SCALE,
        )
        for edge in typed_edges
        if edge.left in retained and edge.right in retained
    )
    graph = evaluator.RelationalGraph(
        units=tuple(units),
        edges=graph_edges,
    )
    serialization_hashes = tuple(
        digest
        for digest, _text in required_embedding_serializations(
            documents,
            turns,
        )
    )
    serialization_set_sha256 = evaluator.stable_hash(
        sorted(serialization_hashes)
    )
    return ActionAdapterResult(
        graph=graph,
        raw_top5=raw_top5,
        direct_anchor_unit_ids=direct_anchors,
        input_serialization_set_sha256=serialization_set_sha256,
        complete_state_count=state_count,
    )


def canonical_action_payload(
    result: ActionAdapterResult,
) -> dict[str, Any]:
    """Return the content-free integer-micro canonical action payload."""

    if not isinstance(result, ActionAdapterResult):
        raise QuacP1ActionAdapterError(
            "canonical action serialization requires ActionAdapterResult"
        )
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "input_serialization_set_sha256": (
            result.input_serialization_set_sha256
        ),
        "raw_top5": list(result.raw_top5),
        "direct_anchor_unit_ids": list(result.direct_anchor_unit_ids),
        "complete_state_count": result.complete_state_count,
        "graph": {
            "units": [
                {
                    "unit_id": unit.unit_id,
                    "node_features_micro": [
                        microquantize(value)
                        for value in unit.node_features
                    ],
                    "dialogue_facets": list(unit.dialogue_facets),
                }
                for unit in result.graph.units
            ],
            "edges": [
                {
                    "left": edge.left,
                    "right": edge.right,
                    "relation": edge.relation,
                    "strength_micro": microquantize(edge.strength),
                }
                for edge in result.graph.edges
            ],
        },
    }


def canonical_action_bytes(result: ActionAdapterResult) -> bytes:
    """Return canonical ASCII JSON bytes for a sealed private action."""

    return evaluator.canonical_bytes(canonical_action_payload(result))


def action_sha256(result: ActionAdapterResult) -> str:
    """Commit the exact content-free graph and RAW/anchor decisions."""

    return hashlib.sha256(canonical_action_bytes(result)).hexdigest()


__all__ = [
    "ActionAdapterInput",
    "ActionAdapterResult",
    "BlockDocument",
    "ENTITY_MAX_DOCUMENT_FREQUENCY",
    "ENTITY_MIN_CODEPOINTS",
    "ENTITY_MIN_DOCUMENT_FREQUENCY",
    "MAX_COMPLETE_STATES",
    "MAX_DIALOGUE_TURNS",
    "MAX_GRAPH_UNITS",
    "MAX_REPLACEMENT_CANDIDATES",
    "MAX_TYPED_EXPANSIONS",
    "MICRO_SCALE",
    "MINILM_EMBEDDING_DIMENSION",
    "MiniLmEmbedding",
    "QuacP1ActionAdapterError",
    "QuestionTurn",
    "SCHEMA",
    "TURN_MARKERS",
    "TURN_RECENCY_MICRO",
    "VERSION",
    "action_sha256",
    "build_action_graph",
    "canonical_action_bytes",
    "canonical_action_payload",
    "microquantize",
    "official_inner_unit_text",
    "proper_name_keys",
    "required_embedding_serializations",
    "serialize_evidence_unit",
    "serialize_full_query",
    "serialize_turn_query",
]
