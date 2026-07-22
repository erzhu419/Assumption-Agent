"""Frozen label-free semantic feature compiler for TAT-QA P18.

This module is the boundary between the immutable local MiniLM encoder and the
pure typed-action algebra in :mod:`tatqa_p18_typed_evaluator_core_v1`.  It has
no source-loader, label, answer, family, baseline, filesystem, or network
interface.  A caller supplies one already-canonical label-free item and either
an encoder or a complete normalized embedding matrix.

The compiler deliberately uses no tuned cosine threshold.  Each typed-plan
facet assigns deterministic query authority to its single best canonical unit,
its single best table unit, and (when present) its single best paragraph unit.
Those assignments induce the five frozen typed-edge coordinates.  This makes
the candidate grammar semantic while avoiding another data-dependent gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Mapping, Protocol, Sequence
import unicodedata

import numpy as np

from assumption_agent.benchmarks import tatqa_p18_typed_evaluator_core_v1 as core


VERSION = "tatqa_p18_label_free_runtime_v1"
QUANTIZATION_SCALE = 1_000_000
EMBEDDING_DIMENSION = 384
MAXIMUM_UNIT_COUNT = 96
MAXIMUM_TEXT_CHARACTERS = 24_000
MAXIMUM_ITEM_CHARACTERS = 250_000
MAXIMUM_NUMERIC_OPERANDS_PER_UNIT = 8

_UNIT_ID = re.compile(r"(?P<kind>T|P):(?P<ordinal>0|[1-9][0-9]*)\Z")
_WHITESPACE = re.compile(r"\s+", flags=re.UNICODE)
_NUMBER = re.compile(
    r"(?<![\w.])(?:[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?)(?![\w.])",
    flags=re.UNICODE,
)


class TatqaP18LabelFreeRuntimeError(RuntimeError):
    """A label-free runtime input or deterministic feature tensor drifted."""


class TextEncoder(Protocol):
    """The only model capability accepted by the compiler."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return one finite, L2-normalized float matrix in input order."""


def _canonical_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise TatqaP18LabelFreeRuntimeError(f"{field} must be NUL-free text")
    try:
        normalized = unicodedata.normalize("NFKC", value)
    except UnicodeError as exc:
        raise TatqaP18LabelFreeRuntimeError(f"{field} is invalid Unicode") from exc
    normalized = _WHITESPACE.sub(" ", normalized).strip()
    if not normalized or len(normalized) > MAXIMUM_TEXT_CHARACTERS:
        raise TatqaP18LabelFreeRuntimeError(f"{field} is empty or oversized")
    return normalized


def _unit_key(unit_id: object) -> tuple[int, int]:
    if not isinstance(unit_id, str):
        raise TatqaP18LabelFreeRuntimeError("canonical unit ID must be text")
    match = _UNIT_ID.fullmatch(unit_id)
    if match is None:
        raise TatqaP18LabelFreeRuntimeError("canonical unit ID drifted")
    kind = 0 if match.group("kind") == "T" else 1
    ordinal = int(match.group("ordinal"))
    if kind == 1 and ordinal == 0:
        raise TatqaP18LabelFreeRuntimeError(
            "paragraph unit order must be positive"
        )
    return kind, ordinal


@dataclass(frozen=True)
class RuntimeUnit:
    """One canonical label-free evidence unit."""

    unit_id: str
    text: str

    def __post_init__(self) -> None:
        _unit_key(self.unit_id)
        object.__setattr__(
            self,
            "text",
            _canonical_text(self.text, field=f"unit {self.unit_id} text"),
        )


@dataclass(frozen=True)
class LabelFreeRuntimeItem:
    """The complete item payload visible to an action worker.

    ``item_id`` is an acquisition-generated opaque commitment.  It is carried
    only for archive binding and never enters any score or tie break.
    """

    item_id: str
    question: str
    units: tuple[RuntimeUnit, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.item_id, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.item_id) is None
        ):
            raise TatqaP18LabelFreeRuntimeError("opaque item ID drifted")
        object.__setattr__(
            self, "question", _canonical_text(self.question, field="question")
        )
        if not isinstance(self.units, tuple):
            raise TatqaP18LabelFreeRuntimeError("units must be a canonical tuple")
        if not 5 <= len(self.units) <= MAXIMUM_UNIT_COUNT:
            raise TatqaP18LabelFreeRuntimeError("unit count is outside the frozen bound")
        if any(not isinstance(row, RuntimeUnit) for row in self.units):
            raise TatqaP18LabelFreeRuntimeError("unit row type drifted")
        identifiers = tuple(row.unit_id for row in self.units)
        if len(set(identifiers)) != len(identifiers):
            raise TatqaP18LabelFreeRuntimeError("unit IDs are duplicated")
        if identifiers != tuple(sorted(identifiers, key=_unit_key)):
            raise TatqaP18LabelFreeRuntimeError("unit order is not canonical")
        if "T:0" not in identifiers:
            raise TatqaP18LabelFreeRuntimeError("canonical table header T:0 is absent")
        if sum(len(row.text) for row in self.units) + len(self.question) > MAXIMUM_ITEM_CHARACTERS:
            raise TatqaP18LabelFreeRuntimeError("item text exceeds the frozen bound")


@dataclass(frozen=True)
class CompiledLabelFreeItem:
    """Content-minimized tensor consumed by the action/evaluator core."""

    item_id: str
    plan: core.TypedPlan
    units: tuple[core.CanonicalUnit, ...]
    raw_top5: tuple[str, ...]
    redundancy_features: Mapping[tuple[str, str], int]
    tensor_sha256: str


def plan_facets(plan: core.TypedPlan | Mapping[str, object]) -> tuple[str, ...]:
    """Return the exact entity/metric/time/relation feature order."""

    checked = core.validate_typed_plan(plan)
    return (
        *checked.entity_facets,
        *checked.metric_facets,
        *checked.time_facets,
        checked.relation_query,
    )


def embedding_texts(
    item: LabelFreeRuntimeItem,
    plan: core.TypedPlan | Mapping[str, object],
) -> tuple[str, ...]:
    """Freeze the one-shot MiniLM input order."""

    if not isinstance(item, LabelFreeRuntimeItem):
        raise TatqaP18LabelFreeRuntimeError("runtime item type drifted")
    return (item.question, *plan_facets(plan), *(row.text for row in item.units))


def _quantized_dot(left: np.ndarray, right: np.ndarray) -> int:
    value = np.asarray(left @ right, dtype=np.float32).item()
    return int(np.rint(float(value) * QUANTIZATION_SCALE))


def _validated_embeddings(value: object, *, rows: int) -> np.ndarray:
    try:
        matrix = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TatqaP18LabelFreeRuntimeError("embedding tensor is not an array") from exc
    if matrix.dtype != np.dtype(np.float32):
        raise TatqaP18LabelFreeRuntimeError("embedding tensor is not exact float32")
    if matrix.shape != (rows, EMBEDDING_DIMENSION):
        raise TatqaP18LabelFreeRuntimeError("embedding tensor shape drifted")
    if not np.isfinite(matrix).all():
        raise TatqaP18LabelFreeRuntimeError("embedding tensor contains nonfinite values")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-6):
        raise TatqaP18LabelFreeRuntimeError("embeddings are not L2 normalized")
    return np.ascontiguousarray(matrix)


def _best_unit(
    unit_indices: Sequence[int],
    similarities: Sequence[int],
    units: Sequence[RuntimeUnit],
) -> int | None:
    if not unit_indices:
        return None
    return min(
        unit_indices,
        key=lambda index: (-similarities[index], _unit_key(units[index].unit_id)),
    )


def _operand_count(text: str, time_facets: Sequence[str]) -> int:
    numeric = {match.group(0).casefold() for match in _NUMBER.finditer(text)}
    folded = text.casefold()
    time_hits = {
        facet.casefold()
        for facet in time_facets
        if facet.casefold() in folded
    }
    return min(MAXIMUM_NUMERIC_OPERANDS_PER_UNIT, len(numeric | time_hits))


def _stable_hash(value: object) -> str:
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def compile_from_embeddings(
    item: LabelFreeRuntimeItem,
    plan: core.TypedPlan | Mapping[str, object],
    embeddings: object,
) -> CompiledLabelFreeItem:
    """Compile a deterministic typed tensor from one complete embedding matrix."""

    if not isinstance(item, LabelFreeRuntimeItem):
        raise TatqaP18LabelFreeRuntimeError("runtime item type drifted")
    checked_plan = core.validate_typed_plan(plan)
    facets = plan_facets(checked_plan)
    matrix = _validated_embeddings(
        embeddings, rows=1 + len(facets) + len(item.units)
    )
    facet_matrix = matrix[1 : 1 + len(facets)]
    unit_matrix = matrix[1 + len(facets) :]
    query_vector = matrix[0]

    query_scores = tuple(_quantized_dot(row, query_vector) for row in unit_matrix)
    facet_scores: list[tuple[int, ...]] = []
    for facet_vector in facet_matrix:
        facet_scores.append(
            tuple(_quantized_dot(row, facet_vector) for row in unit_matrix)
        )

    all_indices = tuple(range(len(item.units)))
    table_indices = tuple(
        index for index, row in enumerate(item.units) if row.unit_id.startswith("T:")
    )
    paragraph_indices = tuple(
        index for index, row in enumerate(item.units) if row.unit_id.startswith("P:")
    )
    global_winners = tuple(
        _best_unit(all_indices, scores, item.units) for scores in facet_scores
    )
    table_winners = tuple(
        _best_unit(table_indices, scores, item.units) for scores in facet_scores
    )
    paragraph_winners = tuple(
        _best_unit(paragraph_indices, scores, item.units) for scores in facet_scores
    )

    coverage = [[0] * len(facets) for _row in item.units]
    edges = [[0] * len(core.TYPED_EDGE_ORDER) for _row in item.units]
    for facet_index, winner in enumerate(global_winners):
        if winner is not None:
            coverage[winner][facet_index] = 1
            # Connectivity is the nonnegative, quantized semantic mass of the
            # winning facet-to-unit edge.  It is therefore not a duplicate of
            # the binary coverage count used by P0's first ranking key.
            edges[winner][0] += max(0, facet_scores[facet_index][winner])

    metric_start = len(checked_plan.entity_facets)
    metric_stop = metric_start + len(checked_plan.metric_facets)
    time_stop = metric_stop + len(checked_plan.time_facets)
    metric_or_time = frozenset(range(metric_start, time_stop))
    header_index = next(
        index for index, row in enumerate(item.units) if row.unit_id == "T:0"
    )
    table_by_ordinal = {
        _unit_key(item.units[index].unit_id)[1]: index for index in table_indices
    }
    operand_counts = tuple(
        _operand_count(row.text, checked_plan.time_facets) for row in item.units
    )

    for facet_index, (table_winner, paragraph_winner) in enumerate(
        zip(table_winners, paragraph_winners)
    ):
        table_mass = (
            max(0, facet_scores[facet_index][table_winner])
            if table_winner is not None
            else 0
        )
        paragraph_mass = (
            max(0, facet_scores[facet_index][paragraph_winner])
            if paragraph_winner is not None
            else 0
        )
        if table_winner is not None:
            table_ordinal = _unit_key(item.units[table_winner].unit_id)[1]
            if table_ordinal > 0:
                edges[table_winner][1] += table_mass
                edges[header_index][1] += table_mass
                if facet_index in metric_or_time and operand_counts[table_winner] > 0:
                    edges[table_winner][2] += table_mass
                for neighbor_ordinal in (table_ordinal - 1, table_ordinal + 1):
                    neighbor = table_by_ordinal.get(neighbor_ordinal)
                    if neighbor is not None and neighbor != header_index:
                        edges[neighbor][4] += table_mass
        if (
            facet_index < time_stop
            and table_winner is not None
            and paragraph_winner is not None
        ):
            cross_modal_mass = min(table_mass, paragraph_mass)
            edges[table_winner][3] += cross_modal_mass
            edges[paragraph_winner][3] += cross_modal_mass

    compiled_units = tuple(
        core.CanonicalUnit(
            unit_id=row.unit_id,
            facet_coverage=tuple(coverage[index]),
            typed_edge_features=tuple(edges[index]),
            numeric_or_time_operand_coverage=operand_counts[index],
            full_question_similarity=query_scores[index],
        )
        for index, row in enumerate(item.units)
    )
    raw_top5 = tuple(
        item.units[index].unit_id
        for index in sorted(
            all_indices,
            key=lambda index: (
                -query_scores[index],
                _unit_key(item.units[index].unit_id),
            ),
        )[: core.TOP_K]
    )
    redundancy: dict[tuple[str, str], int] = {}
    for left in range(len(item.units)):
        for right in range(left + 1, len(item.units)):
            redundancy[(item.units[left].unit_id, item.units[right].unit_id)] = max(
                0, _quantized_dot(unit_matrix[left], unit_matrix[right])
            )

    payload = {
        "compiler": VERSION,
        "facet_order": list(facets),
        "item_id": item.item_id,
        "query_scores": list(query_scores),
        "raw_top5": list(raw_top5),
        "units": [row.payload() for row in compiled_units],
        "redundancy": [
            [left, right, value]
            for (left, right), value in sorted(
                redundancy.items(), key=lambda row: (_unit_key(row[0][0]), _unit_key(row[0][1]))
            )
        ],
    }
    return CompiledLabelFreeItem(
        item_id=item.item_id,
        plan=checked_plan,
        units=compiled_units,
        raw_top5=raw_top5,
        redundancy_features=redundancy,
        tensor_sha256=_stable_hash(payload),
    )


def compile_with_encoder(
    item: LabelFreeRuntimeItem,
    plan: core.TypedPlan | Mapping[str, object],
    encoder: TextEncoder,
) -> CompiledLabelFreeItem:
    """Encode once in frozen order, then call :func:`compile_from_embeddings`."""

    if not hasattr(encoder, "encode") or not callable(encoder.encode):
        raise TatqaP18LabelFreeRuntimeError("encoder capability is unavailable")
    texts = embedding_texts(item, plan)
    try:
        matrix = encoder.encode(texts)
    except Exception as exc:
        raise TatqaP18LabelFreeRuntimeError("offline MiniLM encoding failed") from exc
    return compile_from_embeddings(item, plan, matrix)


__all__ = [
    "CompiledLabelFreeItem",
    "EMBEDDING_DIMENSION",
    "LabelFreeRuntimeItem",
    "MAXIMUM_UNIT_COUNT",
    "QUANTIZATION_SCALE",
    "RuntimeUnit",
    "TatqaP18LabelFreeRuntimeError",
    "TextEncoder",
    "VERSION",
    "compile_from_embeddings",
    "compile_with_encoder",
    "embedding_texts",
    "plan_facets",
]
