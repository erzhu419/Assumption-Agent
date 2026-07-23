"""Canonical, source-free GPT action contract for BIRCO P1.

The module does not know source, query, document, family, block, or qrel IDs.
It exposes only anonymous ordinals and the text projection explicitly allowed
by the preregistration.  Malformed model completions are deterministically
totalized; they are never repaired with another model request.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence


VERSION = "birco_p1_gpt54_semantic_contract_v1"
MODEL_ID = "gpt-5.4-mini"
PLAN_INPUT_SCHEMA = f"{VERSION}_plan_input"
MATRIX_INPUT_SCHEMA = f"{VERSION}_matrix_input"
RAW_INPUT_SCHEMA = f"{VERSION}_raw_input"
TERMINAL_OUTPUT_SCHEMA = f"{VERSION}_terminal_output"

FACET_TYPES = frozenset(
    {"REQUIRED", "EXCLUDED", "PREFERRED", "ELIGIBILITY", "TEMPORAL", "RELATIONAL"}
)
EDGE_TYPES = frozenset({"REQUIRES", "REFINES", "CONTRASTS_WITH"})

MINIMUM_FACETS = 2
MAXIMUM_FACETS = 12
MAXIMUM_EDGES = 36
MAXIMUM_FACET_CHARACTERS = 384
MAXIMUM_QUERY_CHARACTERS = 250_000
MAXIMUM_OBJECTIVE_CHARACTERS = 8_192
MAXIMUM_DOCUMENT_CHARACTERS = 2_000_000
MAXIMUM_EVIDENCE_UNITS = 6
MAXIMUM_EVIDENCE_UNIT_CHARACTERS = 192
MAXIMUM_CANDIDATES_PER_BATCH = 24
MINIMUM_POOL_CANDIDATES = 10
MAXIMUM_POOL_CANDIDATES = 256
MAXIMUM_WORK_ID_CHARACTERS = 1_024
MAXIMUM_COMPLETION_BYTES = 2 * 1024 * 1024
MAXIMUM_OUTPUT_TOKENS = 8_192

_BOUNDARY_CHARACTERS = frozenset(".!?;\n\r。！？；")
_CLAUSE_SPLIT = re.compile(r"(?:\r?\n)+|(?<=[.!?;。！？；])\s+|\s*[,，:]\s+")
_WORD = re.compile(r"\w+", flags=re.UNICODE)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class BircoP1GptContractError(RuntimeError):
    """A public semantic envelope or deterministic projection drifted."""


def canonical_json_bytes(value: object, *, newline: bool = True) -> bytes:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise BircoP1GptContractError("value is not canonical JSON") from exc
    return raw + (b"\n" if newline else b"")


def semantic_hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value, newline=False)).hexdigest()


def _text(value: object, *, label: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise BircoP1GptContractError(f"{label} is not canonical text")
    return value


def _integer(value: object, *, label: str, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise BircoP1GptContractError(f"{label} is outside its integer domain")
    return value


@dataclass(frozen=True)
class Facet:
    ordinal: int
    facet_type: str
    text: str
    weight: int

    def __post_init__(self) -> None:
        _integer(self.ordinal, label="facet ordinal", minimum=0, maximum=MAXIMUM_FACETS - 1)
        if self.facet_type not in FACET_TYPES:
            raise BircoP1GptContractError("facet type drifted")
        _text(self.text, label="facet text", maximum=MAXIMUM_FACET_CHARACTERS)
        _integer(self.weight, label="facet weight", minimum=1, maximum=4)

    def payload(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "text": self.text,
            "type": self.facet_type,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class PlanEdge:
    source: int
    target: int
    edge_type: str

    def __post_init__(self) -> None:
        _integer(self.source, label="edge source", minimum=0, maximum=MAXIMUM_FACETS - 1)
        _integer(self.target, label="edge target", minimum=0, maximum=MAXIMUM_FACETS - 1)
        if self.source == self.target or self.edge_type not in EDGE_TYPES:
            raise BircoP1GptContractError("typed edge drifted")

    def payload(self) -> dict[str, object]:
        return {"source": self.source, "target": self.target, "type": self.edge_type}


@dataclass(frozen=True)
class Plan:
    facets: tuple[Facet, ...]
    edges: tuple[PlanEdge, ...]
    generation_valid: bool

    def __post_init__(self) -> None:
        if not isinstance(self.generation_valid, bool):
            raise BircoP1GptContractError("plan validity flag drifted")
        if not MINIMUM_FACETS <= len(self.facets) <= MAXIMUM_FACETS:
            raise BircoP1GptContractError("plan facet count drifted")
        if tuple(row.ordinal for row in self.facets) != tuple(range(len(self.facets))):
            raise BircoP1GptContractError("plan facet ordinals are not contiguous")
        if len(self.edges) > MAXIMUM_EDGES or len(set(self.edges)) != len(self.edges):
            raise BircoP1GptContractError("plan edge set drifted")
        if any(row.source >= len(self.facets) or row.target >= len(self.facets) for row in self.edges):
            raise BircoP1GptContractError("plan edge endpoint is absent")
        _require_acyclic(len(self.facets), self.edges)

    def payload(self) -> dict[str, object]:
        return {
            "edges": [row.payload() for row in self.edges],
            "facets": [row.payload() for row in self.facets],
            "generation_valid": self.generation_valid,
        }

    @property
    def plan_sha256(self) -> str:
        return semantic_hash(self.payload())


def _require_acyclic(facet_count: int, edges: Sequence[PlanEdge]) -> None:
    adjacency: list[list[int]] = [[] for _ in range(facet_count)]
    indegree = [0] * facet_count
    for edge in edges:
        adjacency[edge.source].append(edge.target)
        indegree[edge.target] += 1
    frontier = [index for index, degree in enumerate(indegree) if degree == 0]
    visited = 0
    while frontier:
        node = frontier.pop()
        visited += 1
        for target in adjacency[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                frontier.append(target)
    if visited != facet_count:
        raise BircoP1GptContractError("plan dependency graph is cyclic")


@dataclass(frozen=True)
class EvidenceUnit:
    ordinal: int
    byte_start: int
    byte_end: int
    text: str

    def __post_init__(self) -> None:
        _integer(self.ordinal, label="evidence ordinal", minimum=0, maximum=1_000_000)
        _integer(self.byte_start, label="evidence byte start", minimum=0, maximum=100_000_000)
        _integer(self.byte_end, label="evidence byte end", minimum=1, maximum=100_000_000)
        if self.byte_end <= self.byte_start:
            raise BircoP1GptContractError("evidence byte interval is empty")
        _text(self.text, label="evidence text", maximum=MAXIMUM_EVIDENCE_UNIT_CHARACTERS)

    def payload(self) -> dict[str, object]:
        return {
            "byte_end": self.byte_end,
            "byte_start": self.byte_start,
            "ordinal": self.ordinal,
            "text": self.text,
        }


@dataclass(frozen=True)
class CandidateProjection:
    ordinal: int
    evidence_units: tuple[EvidenceUnit, ...]

    def __post_init__(self) -> None:
        _integer(
            self.ordinal,
            label="candidate ordinal",
            minimum=0,
            maximum=10_000_000,
        )
        if not 1 <= len(self.evidence_units) <= MAXIMUM_EVIDENCE_UNITS:
            raise BircoP1GptContractError("candidate evidence-unit count drifted")
        if tuple(row.ordinal for row in self.evidence_units) != tuple(range(len(self.evidence_units))):
            raise BircoP1GptContractError("projected evidence ordinals drifted")
        previous_end = -1
        for row in self.evidence_units:
            if row.byte_start < previous_end:
                raise BircoP1GptContractError("projected evidence intervals overlap or reverse")
            if len(row.text.encode("utf-8")) != row.byte_end - row.byte_start:
                raise BircoP1GptContractError("projected evidence text does not match its byte span")
            previous_end = row.byte_end

    @property
    def projection_text(self) -> str:
        """The exact candidate-text value shared by Agent, RAW, and HippoRAG."""

        return json.dumps(
            {"evidence_units": [row.payload() for row in self.evidence_units]},
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def payload(self) -> dict[str, object]:
        return {
            "evidence_units": [row.payload() for row in self.evidence_units],
            "ordinal": self.ordinal,
            "text": self.projection_text,
        }


@dataclass(frozen=True)
class FacetEvidence:
    support: int
    contradiction: int
    evidence_unit_ordinal: int | None

    def __post_init__(self) -> None:
        _integer(self.support, label="support", minimum=0, maximum=4)
        _integer(self.contradiction, label="contradiction", minimum=0, maximum=4)
        if self.evidence_unit_ordinal is not None:
            _integer(
                self.evidence_unit_ordinal,
                label="evidence-unit reference",
                minimum=0,
                maximum=MAXIMUM_EVIDENCE_UNITS - 1,
            )

    def payload(self) -> list[int | None]:
        return [self.support, self.contradiction, self.evidence_unit_ordinal]


def _char_byte_offsets(text: str) -> tuple[int, ...]:
    offsets = [0]
    total = 0
    for character in text:
        total += len(character.encode("utf-8"))
        offsets.append(total)
    return tuple(offsets)


def _bounded_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Split at sentence/clause boundaries, then cap exceptionally long spans."""

    spans: list[tuple[int, int]] = []
    start = 0
    for index, character in enumerate(text):
        if character in _BOUNDARY_CHARACTERS:
            end = index + 1
            if text[start:end].strip():
                spans.append((start, end))
            start = end
    if text[start:].strip():
        spans.append((start, len(text)))
    if not spans:
        spans.append((0, len(text)))

    bounded: list[tuple[int, int]] = []
    for original_start, original_end in spans:
        cursor = original_start
        while cursor < original_end:
            hard_end = min(original_end, cursor + MAXIMUM_EVIDENCE_UNIT_CHARACTERS)
            end = hard_end
            if hard_end < original_end:
                window = text[cursor:hard_end]
                split = max(window.rfind(" "), window.rfind("\t"))
                if split >= MAXIMUM_EVIDENCE_UNIT_CHARACTERS // 2:
                    end = cursor + split
            while cursor < end and text[cursor].isspace():
                cursor += 1
            while end > cursor and text[end - 1].isspace():
                end -= 1
            if cursor >= end:
                cursor = end
                continue
            bounded.append((cursor, end))
            cursor = end
    return tuple(bounded)


def _uniform_indices(length: int, count: int) -> tuple[int, ...]:
    if length <= count:
        return tuple(range(length))
    if count == 1:
        return (0,)
    # Query-independent uniform coverage prevents relevance leakage while
    # retaining head, body, and tail evidence from long documents.
    indices = {(position * (length - 1)) // (count - 1) for position in range(count)}
    if len(indices) != count:
        for index in range(length):
            indices.add(index)
            if len(indices) == count:
                break
    return tuple(sorted(indices))


def project_candidate_text(text: str, *, candidate_ordinal: int) -> CandidateProjection:
    _integer(candidate_ordinal, label="candidate ordinal", minimum=0, maximum=10_000_000)
    _text(text, label="candidate text", maximum=MAXIMUM_DOCUMENT_CHARACTERS)
    offsets = _char_byte_offsets(text)
    all_spans = _bounded_spans(text)
    selected = _uniform_indices(len(all_spans), MAXIMUM_EVIDENCE_UNITS)
    rows: list[EvidenceUnit] = []
    for ordinal, span_index in enumerate(selected):
        start, end = all_spans[span_index]
        visible = text[start:end]
        if not visible:
            raise BircoP1GptContractError("projected evidence became empty")
        rows.append(
            EvidenceUnit(
                ordinal=ordinal,
                byte_start=offsets[start],
                byte_end=offsets[end],
                text=visible,
            )
        )
    return CandidateProjection(candidate_ordinal, tuple(rows))


def candidate_projection_from_text(
    text: str, *, candidate_ordinal: int
) -> CandidateProjection:
    """Reopen one selector-sealed common projection without source text."""

    _integer(
        candidate_ordinal,
        label="candidate ordinal",
        minimum=0,
        maximum=10_000_000,
    )
    if not isinstance(text, str) or not text or "\x00" in text:
        raise BircoP1GptContractError("candidate projection text is invalid")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise BircoP1GptContractError(
            "candidate projection text is invalid JSON"
        ) from exc
    if not isinstance(value, Mapping) or set(value) != {"evidence_units"}:
        raise BircoP1GptContractError("candidate projection shape drifted")
    raw_units = value.get("evidence_units")
    if isinstance(raw_units, (str, bytes)) or not isinstance(raw_units, Sequence):
        raise BircoP1GptContractError("candidate projection rows drifted")
    try:
        units = tuple(
            EvidenceUnit(
                ordinal=row["ordinal"],
                byte_start=row["byte_start"],
                byte_end=row["byte_end"],
                text=row["text"],
            )
            for row in raw_units
            if isinstance(row, Mapping)
            and set(row) == {"byte_end", "byte_start", "ordinal", "text"}
        )
    except (KeyError, TypeError, BircoP1GptContractError) as exc:
        raise BircoP1GptContractError(
            "candidate projection evidence is invalid"
        ) from exc
    if len(units) != len(raw_units):
        raise BircoP1GptContractError("candidate projection evidence shape drifted")
    result = CandidateProjection(candidate_ordinal, units)
    if result.projection_text != text:
        raise BircoP1GptContractError("candidate projection is not canonical")
    return result


def deterministic_plan_totalizer(query: str) -> Plan:
    query = _text(query, label="query", maximum=MAXIMUM_QUERY_CHARACTERS)
    clauses = [part.strip() for part in _CLAUSE_SPLIT.split(query) if part.strip()]
    if len(clauses) < MINIMUM_FACETS:
        words = _WORD.findall(query)
        if len(words) >= 2:
            midpoint = max(1, len(words) // 2)
            clauses = [" ".join(words[:midpoint]), " ".join(words[midpoint:])]
        else:
            clauses = [query, f"Direct evidence for {query}"]
    clauses = clauses[:MAXIMUM_FACETS]
    facets = tuple(
        Facet(index, "REQUIRED", clause[:MAXIMUM_FACET_CHARACTERS].rstrip(), 4)
        for index, clause in enumerate(clauses)
    )
    return Plan(facets=facets, edges=(), generation_valid=False)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError("non-finite JSON constant")


def parse_completion_object(content: str) -> Mapping[str, Any] | None:
    if not isinstance(content, str) or "\x00" in content:
        return None
    stripped = content.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()
    if len(stripped.encode("utf-8")) > MAXIMUM_COMPLETION_BYTES:
        return None
    try:
        value = json.loads(
            stripped,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def parse_plan_completion(content: str, *, query: str) -> Plan:
    fallback = deterministic_plan_totalizer(query)
    value = parse_completion_object(content)
    if value is None or set(value) != {"edges", "facets"}:
        return fallback
    raw_facets = value.get("facets")
    raw_edges = value.get("edges")
    if (
        isinstance(raw_facets, (str, bytes))
        or not isinstance(raw_facets, Sequence)
        or isinstance(raw_edges, (str, bytes))
        or not isinstance(raw_edges, Sequence)
    ):
        return fallback
    try:
        facets = tuple(
            Facet(
                ordinal=_integer(row.get("ordinal"), label="facet ordinal", minimum=0, maximum=MAXIMUM_FACETS - 1),
                facet_type=str(row.get("type")),
                text=_text(row.get("text"), label="facet text", maximum=MAXIMUM_FACET_CHARACTERS),
                weight=_integer(row.get("weight"), label="facet weight", minimum=1, maximum=4),
            )
            for row in raw_facets
            if isinstance(row, Mapping) and set(row) == {"ordinal", "text", "type", "weight"}
        )
        if len(facets) != len(raw_facets):
            return fallback
        edges = tuple(
            PlanEdge(
                source=_integer(row.get("source"), label="edge source", minimum=0, maximum=MAXIMUM_FACETS - 1),
                target=_integer(row.get("target"), label="edge target", minimum=0, maximum=MAXIMUM_FACETS - 1),
                edge_type=str(row.get("type")),
            )
            for row in raw_edges
            if isinstance(row, Mapping) and set(row) == {"source", "target", "type"}
        )
        if len(edges) != len(raw_edges):
            return fallback
        return Plan(facets=facets, edges=edges, generation_valid=True)
    except (BircoP1GptContractError, TypeError, ValueError):
        return fallback


def _validate_batch(candidates: Sequence[CandidateProjection]) -> tuple[CandidateProjection, ...]:
    rows = tuple(candidates)
    if not 1 <= len(rows) <= MAXIMUM_CANDIDATES_PER_BATCH:
        raise BircoP1GptContractError("candidate batch count drifted")
    if len({row.ordinal for row in rows}) != len(rows):
        raise BircoP1GptContractError("candidate batch ordinals repeat")
    return rows


def _work_id(value: object) -> str:
    return _text(value, label="work ID", maximum=MAXIMUM_WORK_ID_CHARACTERS)


def common_projection_payload(
    *, objective: str, query: str, candidates: Sequence[CandidateProjection]
) -> dict[str, object]:
    rows = _validate_batch(candidates)
    return {
        "documents": [
            {"ordinal": row.ordinal, "text": row.projection_text} for row in rows
        ],
        "objective": _text(
            objective, label="objective", maximum=MAXIMUM_OBJECTIVE_CHARACTERS
        ),
        "query": _text(query, label="query", maximum=MAXIMUM_QUERY_CHARACTERS),
    }


def common_projection_sha256(
    *, objective: str, query: str, candidates: Sequence[CandidateProjection]
) -> str:
    return semantic_hash(
        common_projection_payload(
            objective=objective, query=query, candidates=candidates
        )
    )


def _batch_metadata(
    *,
    candidates: Sequence[CandidateProjection],
    batch_ordinal: int,
    batch_count: int,
    pool_candidate_count: int,
) -> tuple[CandidateProjection, ...]:
    rows = _validate_batch(candidates)
    _integer(
        pool_candidate_count,
        label="pool candidate count",
        minimum=MINIMUM_POOL_CANDIDATES,
        maximum=MAXIMUM_POOL_CANDIDATES,
    )
    expected_batch_count = (
        pool_candidate_count + MAXIMUM_CANDIDATES_PER_BATCH - 1
    ) // MAXIMUM_CANDIDATES_PER_BATCH
    if batch_count != expected_batch_count:
        raise BircoP1GptContractError("batch count is not the frozen ceiling division")
    _integer(
        batch_ordinal,
        label="batch ordinal",
        minimum=0,
        maximum=batch_count - 1,
    )
    start = batch_ordinal * MAXIMUM_CANDIDATES_PER_BATCH
    stop = min(pool_candidate_count, start + MAXIMUM_CANDIDATES_PER_BATCH)
    if tuple(row.ordinal for row in rows) != tuple(range(start, stop)):
        raise BircoP1GptContractError("batch is not the canonical full-pool slice")
    return rows


def planner_input(*, work_id: str, objective: str, query: str) -> dict[str, object]:
    return {
        "objective": _text(objective, label="objective", maximum=MAXIMUM_OBJECTIVE_CHARACTERS),
        "query": _text(query, label="query", maximum=MAXIMUM_QUERY_CHARACTERS),
        "schema": PLAN_INPUT_SCHEMA,
        "work_id": _work_id(work_id),
    }


def matrix_input(
    *,
    work_id: str,
    objective: str,
    query: str,
    plan: Plan,
    candidates: Sequence[CandidateProjection],
    batch_ordinal: int,
    batch_count: int,
    pool_candidate_count: int,
    pool_common_projection_sha256: str,
) -> dict[str, object]:
    rows = _batch_metadata(
        candidates=candidates,
        batch_ordinal=batch_ordinal,
        batch_count=batch_count,
        pool_candidate_count=pool_candidate_count,
    )
    batch_common = common_projection_sha256(
        objective=objective, query=query, candidates=rows
    )
    return {
        "batch_count": batch_count,
        "batch_common_projection_sha256": batch_common,
        "batch_ordinal": batch_ordinal,
        "candidates": [row.payload() for row in rows],
        "objective": _text(objective, label="objective", maximum=MAXIMUM_OBJECTIVE_CHARACTERS),
        "plan": plan.payload(),
        "pool_candidate_count": pool_candidate_count,
        "pool_common_projection_sha256": validate_sha256(
            pool_common_projection_sha256,
            label="pool common projection SHA-256",
        ),
        "query": _text(query, label="query", maximum=MAXIMUM_QUERY_CHARACTERS),
        "schema": MATRIX_INPUT_SCHEMA,
        "work_id": _work_id(work_id),
    }


def raw_input(
    *,
    work_id: str,
    objective: str,
    query: str,
    candidates: Sequence[CandidateProjection],
    batch_ordinal: int,
    batch_count: int,
    pool_candidate_count: int,
    pool_common_projection_sha256: str,
) -> dict[str, object]:
    rows = _batch_metadata(
        candidates=candidates,
        batch_ordinal=batch_ordinal,
        batch_count=batch_count,
        pool_candidate_count=pool_candidate_count,
    )
    batch_common = common_projection_sha256(
        objective=objective, query=query, candidates=rows
    )
    return {
        "batch_count": batch_count,
        "batch_common_projection_sha256": batch_common,
        "batch_ordinal": batch_ordinal,
        "candidates": [row.payload() for row in rows],
        "objective": _text(objective, label="objective", maximum=MAXIMUM_OBJECTIVE_CHARACTERS),
        "pool_candidate_count": pool_candidate_count,
        "pool_common_projection_sha256": validate_sha256(
            pool_common_projection_sha256,
            label="pool common projection SHA-256",
        ),
        "query": _text(query, label="query", maximum=MAXIMUM_QUERY_CHARACTERS),
        "schema": RAW_INPUT_SCHEMA,
        "work_id": _work_id(work_id),
    }


def totalized_matrix(
    *, plan: Plan, candidates: Sequence[CandidateProjection]
) -> dict[int, tuple[FacetEvidence, ...]]:
    return {
        candidate.ordinal: tuple(FacetEvidence(0, 0, None) for _ in plan.facets)
        for candidate in _validate_batch(candidates)
    }


def parse_matrix_completion(
    content: str, *, plan: Plan, candidates: Sequence[CandidateProjection]
) -> tuple[dict[int, tuple[FacetEvidence, ...]], bool]:
    fallback = totalized_matrix(plan=plan, candidates=candidates)
    candidate_rows = _validate_batch(candidates)
    value = parse_completion_object(content)
    if value is None or set(value) != {"candidates"}:
        return fallback, False
    raw = value.get("candidates")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or len(raw) != len(candidate_rows):
        return fallback, False
    expected = {row.ordinal: row for row in candidate_rows}
    parsed: dict[int, tuple[FacetEvidence, ...]] = {}
    try:
        for row in raw:
            if not isinstance(row, Mapping) or set(row) != {"ordinal", "rows"}:
                return fallback, False
            ordinal = row.get("ordinal")
            if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal not in expected or ordinal in parsed:
                return fallback, False
            values = row.get("rows")
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) != len(plan.facets):
                return fallback, False
            evidence_count = len(expected[ordinal].evidence_units)
            cells: list[FacetEvidence] = []
            for cell in values:
                if isinstance(cell, (str, bytes)) or not isinstance(cell, Sequence) or len(cell) != 3:
                    return fallback, False
                evidence = cell[2]
                if evidence is not None and (
                    isinstance(evidence, bool)
                    or not isinstance(evidence, int)
                    or not 0 <= evidence < evidence_count
                ):
                    return fallback, False
                cells.append(
                    FacetEvidence(
                        support=_integer(cell[0], label="support", minimum=0, maximum=4),
                        contradiction=_integer(cell[1], label="contradiction", minimum=0, maximum=4),
                        evidence_unit_ordinal=evidence,
                    )
                )
            parsed[ordinal] = tuple(cells)
    except (BircoP1GptContractError, TypeError, ValueError):
        return fallback, False
    if set(parsed) != set(expected):
        return fallback, False
    return parsed, True


def totalized_raw(candidates: Sequence[CandidateProjection]) -> dict[int, int]:
    return {row.ordinal: 0 for row in _validate_batch(candidates)}


def parse_raw_completion(
    content: str, *, candidates: Sequence[CandidateProjection]
) -> tuple[dict[int, int], bool]:
    fallback = totalized_raw(candidates)
    candidate_rows = _validate_batch(candidates)
    value = parse_completion_object(content)
    if value is None or set(value) != {"scores"}:
        return fallback, False
    raw = value.get("scores")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or len(raw) != len(candidate_rows):
        return fallback, False
    expected = {row.ordinal for row in candidate_rows}
    parsed: dict[int, int] = {}
    try:
        for row in raw:
            if not isinstance(row, Mapping) or set(row) != {"ordinal", "score"}:
                return fallback, False
            ordinal = row.get("ordinal")
            if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal not in expected or ordinal in parsed:
                return fallback, False
            parsed[ordinal] = _integer(row.get("score"), label="RAW score", minimum=0, maximum=100)
    except (BircoP1GptContractError, TypeError, ValueError):
        return fallback, False
    if set(parsed) != expected:
        return fallback, False
    return parsed, True


def matrix_payload(value: Mapping[int, Sequence[FacetEvidence]]) -> list[dict[str, object]]:
    return [
        {"ordinal": ordinal, "rows": [cell.payload() for cell in value[ordinal]]}
        for ordinal in sorted(value)
    ]


def raw_payload(value: Mapping[int, int]) -> list[dict[str, int]]:
    return [{"ordinal": ordinal, "score": value[ordinal]} for ordinal in sorted(value)]


def validate_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise BircoP1GptContractError(f"{label} is not SHA-256")
    return value


def finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BircoP1GptContractError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise BircoP1GptContractError(f"{label} is not finite")
    return result
