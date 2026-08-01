"""Label-blind, within-side grounding for bounded GSCL relation sets.

This module converts one independently extracted narrative relation set into
one finite slot graph.  It is intentionally narrower than an entity linker:
each mention endpoint is compared only with the unordered slot surfaces from
the *same* system, cosine scores are quantized once, and a quantized tie
abstains.  No threshold, cross-system text, gold mapping, label pack, scorer,
filesystem, or network capability is accepted.

The resulting graph records noisy within-side grounding.  It does not assert
that the selected slot is semantically correct, that ``slot0``/``slot1`` are
causal directions, or that any formal law has been discovered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence
import unicodedata

import numpy as np

from assumption_agent.generalized_structural_correspondence_v1 import (
    strict_canonical_bytes,
    strict_content_hash,
)
from assumption_agent.gscl_slot_set_mapping_v1 import (
    SemanticSlotScoreMatrixV1,
    SlotGraphV1,
    SlotRelationInputV1,
    build_slot_graph_v1,
)
from replication_runtime.gscl_narrative_extractor_v2 import (
    bounded_set_consumer as _bounded,
)


VERSION = "scar.within_side_minilm_slot_graph_binder.v1"
BINDING_SCHEMA = f"{VERSION}.safe_receipt.v1"
SEMANTIC_MATRIX_SCHEMA = f"{VERSION}.semantic_matrix.safe_receipt.v1"
COSINE_QUANTIZATION_SCALE = 1_000_000
MAXIMUM_EMBEDDING_DIMENSION = 16_384

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}\Z")
_RESULT_MARKER = object()


class TextEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> object: ...


class SlotGraphBinderError(ValueError):
    """Stable content-free binder failure."""

    _KNOWN = frozenset(
        {
            "SCAR_BINDER_ENCODER_INVALID",
            "SCAR_BINDER_INPUT_INVALID",
            "SCAR_BINDER_RECEIPT_INVALID",
            "SCAR_BINDER_RESOURCE_BOUND_EXCEEDED",
            "SCAR_BINDER_RESULT_INVALID",
            "SCAR_BINDER_SLOT_INVALID",
        }
    )

    def __init__(self, issue_id: str) -> None:
        if issue_id not in self._KNOWN:
            raise ValueError("scar_slot_graph_binder_issue_unknown")
        self.issue_id = issue_id
        super().__init__(issue_id)


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _seal(body: Mapping[str, object]) -> bytes:
    payload = dict(body)
    payload["self_sha256"] = strict_content_hash(body)
    return strict_canonical_bytes(payload)


def _read_receipt(raw: bytes, *, schema: str) -> Mapping[str, object]:
    try:
        value = json.loads(raw.decode("ascii"))
    except Exception as exc:
        raise SlotGraphBinderError("SCAR_BINDER_RECEIPT_INVALID") from exc
    if type(value) is not dict or value.get("schema") != schema:
        raise SlotGraphBinderError("SCAR_BINDER_RECEIPT_INVALID")
    body = dict(value)
    claimed = body.pop("self_sha256", None)
    if not _is_hash(claimed) or strict_content_hash(body) != claimed:
        raise SlotGraphBinderError("SCAR_BINDER_RECEIPT_INVALID")
    return MappingProxyType(value)


def _matrix(value: object, *, row_count: int) -> np.ndarray:
    try:
        matrix = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise SlotGraphBinderError("SCAR_BINDER_ENCODER_INVALID") from exc
    if (
        matrix.ndim != 2
        or matrix.shape[0] != row_count
        or matrix.shape[1] < 1
        or matrix.shape[1] > MAXIMUM_EMBEDDING_DIMENSION
        or matrix.dtype.kind != "f"
        or not np.isfinite(matrix).all()
    ):
        raise SlotGraphBinderError("SCAR_BINDER_ENCODER_INVALID")
    as64 = matrix.astype(np.float64, copy=False)
    norms = np.linalg.norm(as64, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise SlotGraphBinderError("SCAR_BINDER_ENCODER_INVALID")
    return as64 / norms[:, None]


def _encode(encoder: TextEncoder, texts: tuple[str, ...]) -> np.ndarray:
    encode = getattr(encoder, "encode", None)
    if not callable(encode) or not texts:
        raise SlotGraphBinderError("SCAR_BINDER_ENCODER_INVALID")
    try:
        value = encode(texts)
    except Exception as exc:
        raise SlotGraphBinderError("SCAR_BINDER_ENCODER_INVALID") from exc
    return _matrix(value, row_count=len(texts))


def _quantize_cosine(value: float) -> int:
    if not math.isfinite(value):
        raise SlotGraphBinderError("SCAR_BINDER_ENCODER_INVALID")
    clipped = max(-1.0, min(1.0, value))
    # Python round is ties-to-even.  The integer is the sole downstream score.
    return int(round(clipped * COSINE_QUANTIZATION_SCALE))


def _validated_relation_set(
    result: _bounded.BoundedNarrativeRelationSetV1,
) -> _bounded.BoundedNarrativeRelationSetV1:
    if type(result) is not _bounded.BoundedNarrativeRelationSetV1:
        raise SlotGraphBinderError("SCAR_BINDER_INPUT_INVALID")
    try:
        result.__post_init__()
        if (
            result.disposition
            is _bounded.SetConsumerDisposition.TYPED_FAILURE_BLOCKED
        ):
            raise SlotGraphBinderError("SCAR_BINDER_INPUT_INVALID")
        source = result.upstream_envelope.source_text.encode(
            "utf-8", errors="strict"
        )
        if (
            result.structural_episode is not None
            and result.structural_episode.verify_source_bytes(source)
        ):
            raise SlotGraphBinderError("SCAR_BINDER_INPUT_INVALID")
    except SlotGraphBinderError:
        raise
    except Exception as exc:
        raise SlotGraphBinderError("SCAR_BINDER_INPUT_INVALID") from exc
    return result


def _validated_slots(slot_labels: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    if (
        not isinstance(slot_labels, Mapping)
        or not slot_labels
        or len(slot_labels) > 16
    ):
        raise SlotGraphBinderError("SCAR_BINDER_SLOT_INVALID")
    rows: list[tuple[str, str]] = []
    normalized_seen: set[str] = set()
    for slot_id, label in slot_labels.items():
        if not isinstance(slot_id, str) or _IDENTIFIER.fullmatch(slot_id) is None:
            raise SlotGraphBinderError("SCAR_BINDER_SLOT_INVALID")
        if not isinstance(label, str) or not label or "\x00" in label:
            raise SlotGraphBinderError("SCAR_BINDER_SLOT_INVALID")
        try:
            encoded = label.encode("utf-8", errors="strict")
            normalized = unicodedata.normalize("NFKC", label).casefold()
            normalized_bytes = normalized.encode("utf-8", errors="strict")
        except (UnicodeError, ValueError) as exc:
            raise SlotGraphBinderError("SCAR_BINDER_SLOT_INVALID") from exc
        if (
            len(encoded) > 4_096
            or len(normalized_bytes) > 4_096
            or not normalized_bytes
            or normalized in normalized_seen
        ):
            raise SlotGraphBinderError("SCAR_BINDER_SLOT_INVALID")
        normalized_seen.add(normalized)
        rows.append((slot_id, label))
    rows.sort(key=lambda row: row[0])
    if len({row[0] for row in rows}) != len(rows):
        raise SlotGraphBinderError("SCAR_BINDER_SLOT_INVALID")
    return tuple(rows)


def _span_quotes(
    result: _bounded.BoundedNarrativeRelationSetV1,
) -> Mapping[str, str]:
    episode = result.structural_episode
    if episode is None:
        if result.units:
            raise SlotGraphBinderError("SCAR_BINDER_INPUT_INVALID")
        return MappingProxyType({})
    raw = result.upstream_envelope.source_text.encode("utf-8", errors="strict")
    needed = {
        span_id
        for unit in result.units
        for span_id in (unit.slot0_span_id, unit.slot1_span_id)
    }
    spans = {row.span_id: row for row in episode.evidence_spans}
    if not needed.issubset(spans):
        raise SlotGraphBinderError("SCAR_BINDER_INPUT_INVALID")
    output: dict[str, str] = {}
    try:
        for span_id in sorted(needed):
            try:
                span = spans[span_id]
            except KeyError as exc:
                raise SlotGraphBinderError(
                    "SCAR_BINDER_INPUT_INVALID"
                ) from exc
            quote = raw[span.start_byte : span.end_byte]
            if hashlib.sha256(quote).hexdigest() != span.span_sha256:
                raise SlotGraphBinderError("SCAR_BINDER_INPUT_INVALID")
            output[span_id] = quote.decode("utf-8", errors="strict")
    except (UnicodeError, IndexError) as exc:
        raise SlotGraphBinderError("SCAR_BINDER_INPUT_INVALID") from exc
    return MappingProxyType(output)


@dataclass(frozen=True, slots=True)
class EndpointBindingV1:
    span_id: str
    selected_slot_id: str | None
    maximum_quantized_cosine: int
    tied_maximum_count: int
    score_vector_commitment: str

    def private_payload(self) -> dict[str, object]:
        return {
            "maximum_quantized_cosine": self.maximum_quantized_cosine,
            "score_vector_commitment": self.score_vector_commitment,
            "selected_slot_id": self.selected_slot_id,
            "span_id": self.span_id,
            "tied_maximum_count": self.tied_maximum_count,
        }


@dataclass(frozen=True, slots=True)
class BoundSlotGraphV1:
    graph: SlotGraphV1
    endpoint_bindings: tuple[EndpointBindingV1, ...] = field(
        repr=False, compare=False
    )
    receipt_bytes: bytes
    _marker: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self._marker is not _RESULT_MARKER
            or type(self.graph) is not SlotGraphV1
            or type(self.endpoint_bindings) is not tuple
            or tuple(sorted(self.endpoint_bindings, key=lambda row: row.span_id))
            != self.endpoint_bindings
        ):
            raise SlotGraphBinderError("SCAR_BINDER_RESULT_INVALID")
        receipt = _read_receipt(self.receipt_bytes, schema=BINDING_SCHEMA)
        if (
            receipt.get("graph_evidence_binding_sha256")
            != self.graph.graph_evidence_binding_sha256
            or receipt.get("endpoint_binding_commitment")
            != strict_content_hash(
                [row.private_payload() for row in self.endpoint_bindings]
            )
        ):
            raise SlotGraphBinderError("SCAR_BINDER_RESULT_INVALID")

    @property
    def receipt(self) -> Mapping[str, object]:
        return _read_receipt(self.receipt_bytes, schema=BINDING_SCHEMA)


def bind_relation_set_to_slots_v1(
    relation_set: _bounded.BoundedNarrativeRelationSetV1,
    *,
    slot_labels: Mapping[str, str],
    encoder: TextEncoder,
    encoder_binding_sha256: str,
) -> BoundSlotGraphV1:
    """Bind one side without seeing the other side or any gold mapping."""

    result = _validated_relation_set(relation_set)
    slots = _validated_slots(slot_labels)
    if not _is_hash(encoder_binding_sha256):
        raise SlotGraphBinderError("SCAR_BINDER_ENCODER_INVALID")
    quotes = _span_quotes(result)
    span_ids = tuple(sorted(quotes))
    texts = tuple(quotes[row] for row in span_ids) + tuple(
        label for _, label in slots
    )
    vectors = _encode(encoder, texts)
    mention_vectors = vectors[: len(span_ids)]
    slot_vectors = vectors[len(span_ids) :]
    bindings: list[EndpointBindingV1] = []
    selected_by_span: dict[str, str | None] = {}
    for row_index, span_id in enumerate(span_ids):
        scores = tuple(
            _quantize_cosine(float(mention_vectors[row_index] @ vector))
            for vector in slot_vectors
        )
        maximum = max(scores)
        winners = tuple(
            slots[index][0]
            for index, score in enumerate(scores)
            if score == maximum
        )
        selected = winners[0] if len(winners) == 1 else None
        selected_by_span[span_id] = selected
        bindings.append(
            EndpointBindingV1(
                span_id=span_id,
                selected_slot_id=selected,
                maximum_quantized_cosine=maximum,
                tied_maximum_count=len(winners),
                score_vector_commitment=strict_content_hash(list(scores)),
            )
        )

    relations: list[SlotRelationInputV1] = []
    dropped = 0
    for unit in result.units:
        slot0 = selected_by_span[unit.slot0_span_id]
        slot1 = selected_by_span[unit.slot1_span_id]
        if slot0 is None or slot1 is None:
            dropped += 1
            continue
        relations.append(
            SlotRelationInputV1(
                relation_id=f"r.{unit.unit_id.split('.', 1)[-1]}",
                slot0_id=slot0,
                slot1_id=slot1,
                generator_kind=unit.generator_kind,
                polarity=unit.polarity,
                temporal_orientation=unit.temporal_orientation,
                causal_orientation=unit.causal_orientation,
                evidence_binding_sha256=strict_content_hash(
                    {
                        "binder_version": VERSION,
                        "slot0_binding": next(
                            row.private_payload()
                            for row in bindings
                            if row.span_id == unit.slot0_span_id
                        ),
                        "slot1_binding": next(
                            row.private_payload()
                            for row in bindings
                            if row.span_id == unit.slot1_span_id
                        ),
                        "unit_evidence_binding_sha256": (
                            unit.evidence_binding_sha256
                        ),
                    }
                ),
            )
        )

    slot_evidence: dict[str, str] = {}
    for slot_id, label in slots:
        attached = [
            row.private_payload()
            for row in bindings
            if row.selected_slot_id == slot_id
        ]
        slot_evidence[slot_id] = strict_content_hash(
            {
                "attached_endpoint_bindings": attached,
                "normalized_surface_sha256": hashlib.sha256(
                    unicodedata.normalize("NFKC", label)
                    .casefold()
                    .encode("utf-8")
                ).hexdigest(),
                "slot_id": slot_id,
            }
        )
    extractor_binding = strict_content_hash(
        {
            "binder_version": VERSION,
            "bounded_consumer_contract_sha256": (
                _bounded.CONSUMER_CONTRACT_SHA256
            ),
            "encoder_binding_sha256": encoder_binding_sha256,
            "relation_set_evidence_binding_sha256": (
                result.evidence_binding_sha256
            ),
            "relation_set_receipt_sha256": hashlib.sha256(
                result.receipt_bytes
            ).hexdigest(),
        }
    )
    # The upstream envelope explicitly disclaims relation-recall totality, so
    # this binder must never upgrade the graph to complete coverage.
    graph = build_slot_graph_v1(
        slot_labels=dict(slots),
        slot_evidence_bindings=slot_evidence,
        relations=relations,
        extractor_binding_sha256=extractor_binding,
        coverage_complete=False,
    )
    endpoint_bindings = tuple(sorted(bindings, key=lambda row: row.span_id))
    body = {
        "claim_scope": "noisy_within_side_slot_grounding_only",
        "cross_side_text_access_count": 0,
        "dropped_relation_count": dropped,
        "effect_authority_established": False,
        "encoder_binding_sha256": encoder_binding_sha256,
        "endpoint_binding_commitment": strict_content_hash(
            [row.private_payload() for row in endpoint_bindings]
        ),
        "endpoint_count": len(endpoint_bindings),
        "formal_law_binding_count": 0,
        "gold_mapping_access_count": 0,
        "graph_evidence_binding_sha256": graph.graph_evidence_binding_sha256,
        "gold_pair_or_scorer_access_count": 0,
        "quantization_rounding": "python_ties_to_even",
        "quantization_scale": COSINE_QUANTIZATION_SCALE,
        "retained_relation_count": len(relations),
        "relation_set_disposition": result.disposition.value,
        "schema": BINDING_SCHEMA,
        "slot_count": len(slots),
        "slot_surface_access_count": len(slots),
        "threshold_applied": False,
        "tie_policy": "quantized_maximum_tie_is_unbound",
        "unbound_endpoint_count": sum(
            row.selected_slot_id is None for row in endpoint_bindings
        ),
        "version": VERSION,
        "zero_degree_slots_retained": True,
    }
    return BoundSlotGraphV1(
        graph=graph,
        endpoint_bindings=endpoint_bindings,
        receipt_bytes=_seal(body),
        _marker=_RESULT_MARKER,
    )


@dataclass(frozen=True, slots=True)
class SemanticMatrixResultV1:
    matrix: SemanticSlotScoreMatrixV1
    receipt_bytes: bytes

    @property
    def receipt(self) -> Mapping[str, object]:
        receipt = _read_receipt(
            self.receipt_bytes, schema=SEMANTIC_MATRIX_SCHEMA
        )
        if receipt.get("matrix_commitment") != self.matrix.commitment:
            raise SlotGraphBinderError("SCAR_BINDER_RESULT_INVALID")
        return receipt


def semantic_slot_score_matrix_v1(
    *,
    source_slot_labels: Mapping[str, str],
    target_slot_labels: Mapping[str, str],
    encoder: TextEncoder,
    encoder_binding_sha256: str,
) -> SemanticMatrixResultV1:
    """Create the common semantic proposal pool from slot surfaces only."""

    source = _validated_slots(source_slot_labels)
    target = _validated_slots(target_slot_labels)
    if len(source) != len(target):
        raise SlotGraphBinderError("SCAR_BINDER_SLOT_INVALID")
    if not _is_hash(encoder_binding_sha256):
        raise SlotGraphBinderError("SCAR_BINDER_ENCODER_INVALID")
    vectors = _encode(
        encoder,
        tuple(row[1] for row in source) + tuple(row[1] for row in target),
    )
    source_vectors = vectors[: len(source)]
    target_vectors = vectors[len(source) :]
    rows: dict[tuple[str, str], int] = {}
    for source_index, (source_id, _) in enumerate(source):
        for target_index, (target_id, _) in enumerate(target):
            rows[(source_id, target_id)] = _quantize_cosine(
                float(source_vectors[source_index] @ target_vectors[target_index])
            )
    matrix = SemanticSlotScoreMatrixV1.from_mapping(rows)
    body = {
        "cross_system_background_access_count": 0,
        "encoder_binding_sha256": encoder_binding_sha256,
        "gold_mapping_access_count": 0,
        "gold_pair_or_scorer_access_count": 0,
        "matrix_commitment": matrix.commitment,
        "quantization_rounding": "python_ties_to_even",
        "quantization_scale": COSINE_QUANTIZATION_SCALE,
        "schema": SEMANTIC_MATRIX_SCHEMA,
        "source_slot_count": len(source),
        "slot_surface_access_count": len(source) + len(target),
        "target_slot_count": len(target),
        "text_fields": ["unordered_source_slot_surfaces", "unordered_target_slot_surfaces"],
        "version": VERSION,
    }
    return SemanticMatrixResultV1(matrix=matrix, receipt_bytes=_seal(body))


__all__ = [
    "BINDING_SCHEMA",
    "BoundSlotGraphV1",
    "COSINE_QUANTIZATION_SCALE",
    "EndpointBindingV1",
    "SEMANTIC_MATRIX_SCHEMA",
    "SemanticMatrixResultV1",
    "SlotGraphBinderError",
    "TextEncoder",
    "VERSION",
    "bind_relation_set_to_slots_v1",
    "semantic_slot_score_matrix_v1",
]
