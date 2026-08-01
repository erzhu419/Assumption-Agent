"""Source-free categorical slot-set correspondence for SCAR/GSCL.

The module deliberately separates two authorities which must not be
confused:

* :func:`qualify_exact_bounded_slot_ownership` can match mention-local
  endpoints in a ``BoundedNarrativeRelationSetV1`` to caller supplied labels
  by exact UTF-8 span ownership followed by NFKC+casefold normalization.  It
  is only a *partial compatibility qualification*.  It never creates an
  authoritative slot graph and is never accepted by the mapping core.
* :func:`build_slot_graph_v1` is the bounded in-memory trust boundary for a
  future slot-aware extractor.  It seals independently formed slots,
  categorical relations, and their evidence commitments.  Only exact
  ``SlotGraphV1`` instances are accepted by :func:`map_slot_graphs_v1`.

Proposal search is finite and non-tunable.  It unions semantic k-best
injective assignments with genuinely structure-only k-best assignments for
a fixed eight-operator closure.  Both pools use the already audited
rectangular Hungarian plus Lawler/Murty implementation.  The verifier never
receives pair labels, gold answers, laws, source files, or scorer callbacks.
It checks injectivity, mention-independent typed incidence, and ordered
length-two composition.  ``slot0`` and ``slot1`` remain positional names;
they do not acquire causal source/target semantics here.

No function in this module performs filesystem, network, model, source-
archive, label, or scorer I/O.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import unicodedata

from assumption_agent import gscl_unit_mapping_v2 as _assignment
from assumption_agent.generalized_structural_correspondence_v1 import (
    strict_canonical_bytes,
    strict_content_hash,
)
from replication_runtime.gscl_narrative_extractor_v2 import (
    bounded_set_consumer as _bounded,
)


VERSION = "scar.categorical_slot_set_mapping.v1"
SLOT_GRAPH_SCHEMA = f"{VERSION}.slot_graph.safe_receipt.v1"
EXACT_OWNERSHIP_SCHEMA = f"{VERSION}.exact_ownership.safe_receipt.v1"
MAPPING_RECEIPT_SCHEMA = f"{VERSION}.mapping.safe_receipt.v1"

MAXIMUM_SLOTS_PER_SIDE = 16
MAXIMUM_RELATIONS_PER_SIDE = 64
K_BEST_PER_POOL = 4
MAXIMUM_SCORE_ABS = 1_000_000_000
MAXIMUM_PROPOSALS = 2 * 8 * K_BEST_PER_POOL
MAXIMUM_ASSIGNMENT_SUBPROBLEMS = 9 * (
    1 + (K_BEST_PER_POOL - 1) * MAXIMUM_SLOTS_PER_SIDE
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}\Z")
_GRAPH_MARKER = object()
_QUALIFICATION_MARKER = object()
_RESULT_MARKER = object()

_KINDS = frozenset({"relation", "state_change", "temporal", "causal"})
_POLARITIES = frozenset({"negative", "neutral", "positive"})
_ORIENTATIONS = frozenset({"none", "forward", "reverse"})


class SlotSetMappingError(ValueError):
    """Stable, content-free typed failure."""

    _KNOWN = frozenset(
        {
            "SCAR_ASSIGNMENT_SOLVER_INVALID",
            "SCAR_BOUNDED_SET_INVALID",
            "SCAR_EXACT_OWNERSHIP_INVALID",
            "SCAR_GRAPH_AUTHORITY_INVALID",
            "SCAR_GRAPH_EVIDENCE_INVALID",
            "SCAR_GRAPH_RELATION_INVALID",
            "SCAR_GRAPH_SLOT_INVALID",
            "SCAR_MAPPING_AUTHORITY_INVALID",
            "SCAR_MAPPING_RECEIPT_INVALID",
            "SCAR_NORMALIZED_SLOT_AMBIGUOUS",
            "SCAR_RESOURCE_BOUND_EXCEEDED",
            "SCAR_SCORE_MATRIX_INCOMPLETE",
            "SCAR_SCORE_MATRIX_INVALID",
        }
    )

    def __init__(self, issue_id: str) -> None:
        if issue_id not in self._KNOWN:
            raise ValueError("scar_slot_set_mapping_issue_unknown")
        self.issue_id = issue_id
        super().__init__(issue_id)


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _identifier(value: object, issue_id: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise SlotSetMappingError(issue_id)
    return value


def _normal(value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise SlotSetMappingError("SCAR_GRAPH_SLOT_INVALID")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    if not normalized or len(normalized.encode("utf-8")) > 4096:
        raise SlotSetMappingError("SCAR_GRAPH_SLOT_INVALID")
    return normalized


def _self_seal(payload: Mapping[str, Any]) -> bytes:
    body = dict(payload)
    body["self_sha256"] = strict_content_hash(payload)
    return strict_canonical_bytes(body)


def _read_receipt(payload: bytes, *, schema: str) -> Mapping[str, object]:
    try:
        value = json.loads(payload.decode("ascii"))
    except Exception as exc:
        raise SlotSetMappingError("SCAR_MAPPING_RECEIPT_INVALID") from exc
    if type(value) is not dict or value.get("schema") != schema:
        raise SlotSetMappingError("SCAR_MAPPING_RECEIPT_INVALID")
    body = dict(value)
    self_hash = body.pop("self_sha256", None)
    if not _is_hash(self_hash) or self_hash != strict_content_hash(body):
        raise SlotSetMappingError("SCAR_MAPPING_RECEIPT_INVALID")
    return MappingProxyType(value)


@dataclass(frozen=True, slots=True)
class SlotRelationInputV1:
    relation_id: str
    slot0_id: str
    slot1_id: str
    generator_kind: str
    polarity: str
    temporal_orientation: str
    causal_orientation: str
    evidence_binding_sha256: str

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        for value in (self.relation_id, self.slot0_id, self.slot1_id):
            if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
                issues.append("relation_identifier_invalid")
        if self.generator_kind not in _KINDS:
            issues.append("relation_kind_invalid")
        if self.polarity not in _POLARITIES:
            issues.append("relation_polarity_invalid")
        if self.temporal_orientation not in _ORIENTATIONS:
            issues.append("relation_temporal_invalid")
        if self.causal_orientation not in _ORIENTATIONS:
            issues.append("relation_causal_invalid")
        if not _is_hash(self.evidence_binding_sha256):
            issues.append("relation_evidence_invalid")
        return tuple(sorted(set(issues)))

    def private_payload(self) -> dict[str, object]:
        return {
            "causal_orientation": self.causal_orientation,
            "evidence_binding_sha256": self.evidence_binding_sha256,
            "generator_kind": self.generator_kind,
            "polarity": self.polarity,
            "relation_id": self.relation_id,
            "slot0_id": self.slot0_id,
            "slot1_id": self.slot1_id,
            "temporal_orientation": self.temporal_orientation,
        }


@dataclass(frozen=True, slots=True)
class SlotNodeV1:
    slot_id: str
    normalized_label: str = field(repr=False)
    evidence_binding_sha256: str

    @property
    def normalized_label_sha256(self) -> str:
        return hashlib.sha256(self.normalized_label.encode("utf-8")).hexdigest()

    def private_payload(self) -> dict[str, object]:
        return {
            "evidence_binding_sha256": self.evidence_binding_sha256,
            "normalized_label_sha256": self.normalized_label_sha256,
            "slot_id": self.slot_id,
        }


@dataclass(frozen=True, slots=True)
class SlotGraphV1:
    """Internally authoritative finite categorical slot graph.

    Authority means that this in-memory object is a self-consistent result of
    the controlled factory below.  It does not certify an external
    extractor's semantic truth or any benchmark effect.
    """

    slots: tuple[SlotNodeV1, ...]
    relations: tuple[SlotRelationInputV1, ...]
    extractor_binding_sha256: str
    graph_evidence_binding_sha256: str
    coverage_complete: bool
    receipt_bytes: bytes
    _marker: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._marker is not _GRAPH_MARKER:
            raise SlotSetMappingError("SCAR_GRAPH_AUTHORITY_INVALID")
        expected = _derive_graph(
            self.slots,
            self.relations,
            extractor_binding_sha256=self.extractor_binding_sha256,
            coverage_complete=self.coverage_complete,
        )
        if (
            self.graph_evidence_binding_sha256
            != expected["graph_evidence_binding_sha256"]
            or self.receipt_bytes != expected["receipt_bytes"]
        ):
            raise SlotSetMappingError("SCAR_GRAPH_AUTHORITY_INVALID")

    @property
    def receipt(self) -> Mapping[str, object]:
        return _read_receipt(self.receipt_bytes, schema=SLOT_GRAPH_SCHEMA)


def _edge_color(row: SlotRelationInputV1) -> tuple[str, str, str, str]:
    return (
        row.generator_kind,
        row.polarity,
        row.temporal_orientation,
        row.causal_orientation,
    )


def _rename_invariant_shape(
    slots: tuple[SlotNodeV1, ...],
    relations: tuple[SlotRelationInputV1, ...],
) -> str:
    """A bounded WL-style categorical graph commitment without slot names."""

    colors: dict[str, str] = {}
    for slot in slots:
        incidences: list[object] = []
        for edge in relations:
            color = list(_edge_color(edge))
            if edge.slot0_id == slot.slot_id:
                incidences.append(["slot0", color])
            if edge.slot1_id == slot.slot_id:
                incidences.append(["slot1", color])
        colors[slot.slot_id] = strict_content_hash(sorted(incidences))
    for _ in range(min(len(slots), MAXIMUM_SLOTS_PER_SIDE)):
        updated: dict[str, str] = {}
        for slot in slots:
            neighborhood: list[object] = []
            for edge in relations:
                color = list(_edge_color(edge))
                if edge.slot0_id == slot.slot_id:
                    neighborhood.append(
                        ["slot0", color, colors[edge.slot1_id]]
                    )
                if edge.slot1_id == slot.slot_id:
                    neighborhood.append(
                        ["slot1", color, colors[edge.slot0_id]]
                    )
            updated[slot.slot_id] = strict_content_hash(
                {"prior": colors[slot.slot_id], "neighbors": sorted(neighborhood)}
            )
        colors = updated
    return strict_content_hash(
        {
            "edge_color_multiset": sorted(
                [list(_edge_color(row)) for row in relations]
            ),
            "node_color_multiset": sorted(colors.values()),
            "relation_count": len(relations),
            "slot_count": len(slots),
        }
    )


def _derive_graph(
    slots: tuple[SlotNodeV1, ...],
    relations: tuple[SlotRelationInputV1, ...],
    *,
    extractor_binding_sha256: str,
    coverage_complete: bool,
) -> dict[str, object]:
    if (
        type(slots) is not tuple
        or type(relations) is not tuple
        or not slots
        or len(slots) > MAXIMUM_SLOTS_PER_SIDE
        or len(relations) > MAXIMUM_RELATIONS_PER_SIDE
    ):
        raise SlotSetMappingError("SCAR_RESOURCE_BOUND_EXCEEDED")
    if type(coverage_complete) is not bool:
        raise SlotSetMappingError("SCAR_GRAPH_AUTHORITY_INVALID")
    if not _is_hash(extractor_binding_sha256):
        raise SlotSetMappingError("SCAR_GRAPH_EVIDENCE_INVALID")
    if any(type(row) is not SlotNodeV1 for row in slots):
        raise SlotSetMappingError("SCAR_GRAPH_SLOT_INVALID")
    if any(type(row) is not SlotRelationInputV1 for row in relations):
        raise SlotSetMappingError("SCAR_GRAPH_RELATION_INVALID")
    if tuple(sorted(slots, key=lambda row: row.slot_id)) != slots:
        raise SlotSetMappingError("SCAR_GRAPH_SLOT_INVALID")
    if tuple(sorted(relations, key=lambda row: row.relation_id)) != relations:
        raise SlotSetMappingError("SCAR_GRAPH_RELATION_INVALID")
    slot_ids = [row.slot_id for row in slots]
    normalized = [row.normalized_label for row in slots]
    if (
        len(set(slot_ids)) != len(slot_ids)
        or len(set(normalized)) != len(normalized)
        or any(
            _IDENTIFIER.fullmatch(row.slot_id) is None
            or _normal(row.normalized_label) != row.normalized_label
            or not _is_hash(row.evidence_binding_sha256)
            for row in slots
        )
    ):
        issue = (
            "SCAR_NORMALIZED_SLOT_AMBIGUOUS"
            if len(set(normalized)) != len(normalized)
            else "SCAR_GRAPH_SLOT_INVALID"
        )
        raise SlotSetMappingError(issue)
    relation_ids = [row.relation_id for row in relations]
    known_slots = set(slot_ids)
    if (
        len(set(relation_ids)) != len(relation_ids)
        or any(row.validate() for row in relations)
        or any(
            row.slot0_id not in known_slots or row.slot1_id not in known_slots
            for row in relations
        )
    ):
        raise SlotSetMappingError("SCAR_GRAPH_RELATION_INVALID")

    graph_binding = strict_content_hash(
        {
            "coverage_complete": coverage_complete,
            "extractor_binding_sha256": extractor_binding_sha256,
            "relations": [row.private_payload() for row in relations],
            "slots": [row.private_payload() for row in slots],
        }
    )
    body = {
        "categorical_relation_count": len(relations),
        "claim_scope": "internally_sealed_source_free_categorical_slot_graph",
        "coverage_complete": coverage_complete,
        "directed_endpoint_semantics_established": False,
        "effect_authority_established": False,
        "external_extractor_semantic_truth_established": False,
        "extractor_binding_sha256": extractor_binding_sha256,
        "formal_law_binding_count": 0,
        "graph_evidence_binding_sha256": graph_binding,
        "internal_graph_authority_established": True,
        "normalized_slot_labels_disclosed": False,
        "positional_slot0_slot1_only": True,
        "relation_commitment": strict_content_hash(
            [row.private_payload() for row in relations]
        ),
        "rename_invariant_shape_sha256": _rename_invariant_shape(
            slots, relations
        ),
        "schema": SLOT_GRAPH_SCHEMA,
        "slot_commitment": strict_content_hash(
            [row.private_payload() for row in slots]
        ),
        "slot_count": len(slots),
        "source_archive_access_count": 0,
        "version": VERSION,
    }
    return {
        "graph_evidence_binding_sha256": graph_binding,
        "receipt_bytes": _self_seal(body),
    }


def build_slot_graph_v1(
    *,
    slot_labels: Mapping[str, str],
    slot_evidence_bindings: Mapping[str, str],
    relations: Sequence[SlotRelationInputV1],
    extractor_binding_sha256: str,
    coverage_complete: bool,
) -> SlotGraphV1:
    """Seal a bounded graph produced independently on one side.

    ``slot_labels`` are used only for collision detection and categorical
    ownership.  Safe receipts retain their commitments, never their text.
    """

    if not isinstance(slot_labels, Mapping) or not isinstance(
        slot_evidence_bindings, Mapping
    ):
        raise SlotSetMappingError("SCAR_GRAPH_SLOT_INVALID")
    if len(slot_labels) != len(slot_evidence_bindings):
        raise SlotSetMappingError("SCAR_GRAPH_EVIDENCE_INVALID")
    nodes: list[SlotNodeV1] = []
    normalized_seen: set[str] = set()
    for slot_id, label in slot_labels.items():
        slot_id = _identifier(slot_id, "SCAR_GRAPH_SLOT_INVALID")
        normalized = _normal(label)
        if normalized in normalized_seen:
            raise SlotSetMappingError("SCAR_NORMALIZED_SLOT_AMBIGUOUS")
        normalized_seen.add(normalized)
        evidence = slot_evidence_bindings.get(slot_id)
        if not _is_hash(evidence):
            raise SlotSetMappingError("SCAR_GRAPH_EVIDENCE_INVALID")
        nodes.append(
            SlotNodeV1(
                slot_id=slot_id,
                normalized_label=normalized,
                evidence_binding_sha256=evidence,
            )
        )
    if set(slot_evidence_bindings) != set(slot_labels):
        raise SlotSetMappingError("SCAR_GRAPH_EVIDENCE_INVALID")
    canonical_slots = tuple(sorted(nodes, key=lambda row: row.slot_id))
    canonical_relations = tuple(sorted(tuple(relations), key=lambda row: row.relation_id))
    derived = _derive_graph(
        canonical_slots,
        canonical_relations,
        extractor_binding_sha256=extractor_binding_sha256,
        coverage_complete=coverage_complete,
    )
    return SlotGraphV1(
        slots=canonical_slots,
        relations=canonical_relations,
        extractor_binding_sha256=extractor_binding_sha256,
        graph_evidence_binding_sha256=str(
            derived["graph_evidence_binding_sha256"]
        ),
        coverage_complete=coverage_complete,
        receipt_bytes=derived["receipt_bytes"],  # type: ignore[arg-type]
        _marker=_GRAPH_MARKER,
    )


@dataclass(frozen=True, slots=True)
class ExactOwnershipQualificationV1:
    """A partial exact-match diagnostic, never a mapping graph."""

    receipt_bytes: bytes
    _relation_set: _bounded.BoundedNarrativeRelationSetV1 = field(
        repr=False, compare=False
    )
    _slot_labels: tuple[str, ...] = field(repr=False, compare=False)
    _marker: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._marker is not _QUALIFICATION_MARKER:
            raise SlotSetMappingError("SCAR_EXACT_OWNERSHIP_INVALID")
        if self.receipt_bytes != _derive_exact_ownership(
            self._relation_set, self._slot_labels
        ):
            raise SlotSetMappingError("SCAR_EXACT_OWNERSHIP_INVALID")

    @property
    def receipt(self) -> Mapping[str, object]:
        return _read_receipt(self.receipt_bytes, schema=EXACT_OWNERSHIP_SCHEMA)


def _validated_bounded(
    value: _bounded.BoundedNarrativeRelationSetV1,
) -> None:
    if type(value) is not _bounded.BoundedNarrativeRelationSetV1:
        raise SlotSetMappingError("SCAR_BOUNDED_SET_INVALID")
    try:
        value.__post_init__()
    except Exception as exc:
        raise SlotSetMappingError("SCAR_BOUNDED_SET_INVALID") from exc


def _canonical_unordered_labels(labels: Sequence[str]) -> tuple[str, ...]:
    if isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence):
        raise SlotSetMappingError("SCAR_GRAPH_SLOT_INVALID")
    normalized = tuple(_normal(row) for row in labels)
    if len(normalized) > MAXIMUM_SLOTS_PER_SIDE:
        raise SlotSetMappingError("SCAR_RESOURCE_BOUND_EXCEEDED")
    if len(set(normalized)) != len(normalized):
        raise SlotSetMappingError("SCAR_NORMALIZED_SLOT_AMBIGUOUS")
    return tuple(sorted(normalized))


def _derive_exact_ownership(
    relation_set: _bounded.BoundedNarrativeRelationSetV1,
    labels: tuple[str, ...],
) -> bytes:
    _validated_bounded(relation_set)
    normalized_labels = _canonical_unordered_labels(labels)
    label_set = set(normalized_labels)
    episode = relation_set.structural_episode
    spans = {} if episode is None else {row.span_id: row for row in episode.evidence_spans}
    source_bytes = relation_set.upstream_envelope.source_text.encode("utf-8")
    matched_endpoint_count = 0
    endpoint_count = 2 * len(relation_set.units)
    partial_edges: list[object] = []
    covered_slots: set[str] = set()
    for unit in relation_set.units:
        endpoint_norms: list[str | None] = []
        for span_id in (unit.slot0_span_id, unit.slot1_span_id):
            span = spans.get(span_id)
            if span is None:
                raise SlotSetMappingError("SCAR_EXACT_OWNERSHIP_INVALID")
            raw = source_bytes[span.start_byte : span.end_byte]
            if (
                hashlib.sha256(raw).hexdigest() != span.span_sha256
                or hashlib.sha256(source_bytes).hexdigest() != span.source_sha256
            ):
                raise SlotSetMappingError("SCAR_EXACT_OWNERSHIP_INVALID")
            try:
                normalized = unicodedata.normalize(
                    "NFKC", raw.decode("utf-8", errors="strict")
                ).casefold()
            except UnicodeError as exc:
                raise SlotSetMappingError("SCAR_EXACT_OWNERSHIP_INVALID") from exc
            if normalized in label_set:
                matched_endpoint_count += 1
                covered_slots.add(normalized)
                endpoint_norms.append(normalized)
            else:
                endpoint_norms.append(None)
        if all(row is not None for row in endpoint_norms):
            partial_edges.append(
                {
                    "color": list(_edge_color_from_unit(unit)),
                    "slot0_sha256": hashlib.sha256(
                        str(endpoint_norms[0]).encode("utf-8")
                    ).hexdigest(),
                    "slot1_sha256": hashlib.sha256(
                        str(endpoint_norms[1]).encode("utf-8")
                    ).hexdigest(),
                }
            )
    body = {
        "claim_scope": "partial_exact_nfkc_casefold_ownership_qualification_only",
        "covered_slot_count": len(covered_slots),
        "disposition": "PARTIAL_EXACT_OWNERSHIP_ONLY",
        "effect_authority_established": False,
        "endpoint_count": endpoint_count,
        "exact_endpoint_coverage_complete": (
            endpoint_count > 0 and matched_endpoint_count == endpoint_count
        ),
        "formal_law_binding_count": 0,
        "mapping_eligible": False,
        "matched_endpoint_count": matched_endpoint_count,
        "missing_endpoint_count": endpoint_count - matched_endpoint_count,
        "normalized_slot_collision_checked": True,
        "partial_edge_count": len(partial_edges),
        "partial_graph_authority_established": False,
        "partial_graph_commitment": strict_content_hash(partial_edges),
        "positional_slot0_slot1_only": True,
        "relation_count": len(relation_set.units),
        "relation_set_evidence_binding_sha256": relation_set.evidence_binding_sha256,
        "schema": EXACT_OWNERSHIP_SCHEMA,
        "slot_label_count": len(normalized_labels),
        "slot_label_set_commitment": strict_content_hash(
            [hashlib.sha256(row.encode("utf-8")).hexdigest() for row in normalized_labels]
        ),
        "source_archive_access_count": 0,
        "version": VERSION,
    }
    return _self_seal(body)


def _edge_color_from_unit(
    row: _bounded.NarrativeRelationUnitV1,
) -> tuple[str, str, str, str]:
    return (
        row.generator_kind,
        row.polarity,
        row.temporal_orientation,
        row.causal_orientation,
    )


def qualify_exact_bounded_slot_ownership(
    relation_set: _bounded.BoundedNarrativeRelationSetV1,
    slot_labels: Sequence[str],
) -> ExactOwnershipQualificationV1:
    canonical = _canonical_unordered_labels(slot_labels)
    receipt = _derive_exact_ownership(relation_set, canonical)
    return ExactOwnershipQualificationV1(
        receipt_bytes=receipt,
        _relation_set=relation_set,
        _slot_labels=canonical,
        _marker=_QUALIFICATION_MARKER,
    )


@dataclass(frozen=True, slots=True)
class SemanticSlotScoreMatrixV1:
    rows: tuple[tuple[str, str, int], ...]

    def __post_init__(self) -> None:
        if type(self.rows) is not tuple or tuple(sorted(self.rows)) != self.rows:
            raise SlotSetMappingError("SCAR_SCORE_MATRIX_INVALID")
        seen: set[tuple[str, str]] = set()
        for row in self.rows:
            if type(row) is not tuple or len(row) != 3:
                raise SlotSetMappingError("SCAR_SCORE_MATRIX_INVALID")
            source_id, target_id, score = row
            _identifier(source_id, "SCAR_SCORE_MATRIX_INVALID")
            _identifier(target_id, "SCAR_SCORE_MATRIX_INVALID")
            if (
                type(score) is not int
                or abs(score) > MAXIMUM_SCORE_ABS
                or (source_id, target_id) in seen
            ):
                raise SlotSetMappingError("SCAR_SCORE_MATRIX_INVALID")
            seen.add((source_id, target_id))

    @classmethod
    def from_mapping(
        cls, rows: Mapping[tuple[str, str], int]
    ) -> "SemanticSlotScoreMatrixV1":
        if not isinstance(rows, Mapping):
            raise SlotSetMappingError("SCAR_SCORE_MATRIX_INVALID")
        canonical: list[tuple[str, str, int]] = []
        for key, score in rows.items():
            if type(key) is not tuple or len(key) != 2:
                raise SlotSetMappingError("SCAR_SCORE_MATRIX_INVALID")
            canonical.append((key[0], key[1], score))
        return cls(rows=tuple(sorted(canonical)))

    @property
    def commitment(self) -> str:
        return strict_content_hash([list(row) for row in self.rows])


@dataclass(frozen=True, slots=True)
class CategoricalOperatorV1:
    orientation_inverting: bool
    invert_polarity: bool
    reverse_positional_slots: bool

    @property
    def operator_id(self) -> str:
        return (
            f"ori_{'inv' if self.orientation_inverting else 'keep'}."
            f"pol_{'inv' if self.invert_polarity else 'keep'}."
            f"slots_{'reverse' if self.reverse_positional_slots else 'identity'}"
        )

    def safe_payload(self) -> dict[str, object]:
        return {
            "invert_polarity": self.invert_polarity,
            "operator_id": self.operator_id,
            "orientation_inverting": self.orientation_inverting,
            "reverse_positional_slots": self.reverse_positional_slots,
        }


OPERATOR_CLOSURE = tuple(
    CategoricalOperatorV1(orientation, polarity, reverse)
    for orientation in (False, True)
    for polarity in (False, True)
    for reverse in (False, True)
)


class MappingArm(str, Enum):
    SEMANTIC_ONLY = "semantic_only"
    FLAT_STRUCTURAL = "flat_structural"
    FULL_NO_COMPOSITION = "full_no_composition"
    FULL_WITH_LENGTH2_COMPOSITION = "full_with_length2_composition"


class ChoiceDisposition(str, Enum):
    SELECTED = "SELECTED"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True, slots=True)
class MappingProposalV1:
    operator_id: str
    target_indices: tuple[int, ...]
    origins: tuple[str, ...]
    semantic_score: int
    flat_structural_score: int
    typed_incidence_matched: int
    typed_incidence_total: int
    length2_path_matched: int
    length2_path_total: int
    injective_verified: bool
    typed_incidence_verified: bool
    length2_composition_verified: bool
    proposal_hash: str

    def _body(self) -> dict[str, object]:
        return {
            "flat_structural_score": self.flat_structural_score,
            "injective_verified": self.injective_verified,
            "length2_composition_verified": self.length2_composition_verified,
            "length2_path_matched": self.length2_path_matched,
            "length2_path_total": self.length2_path_total,
            "operator_id": self.operator_id,
            "origins": list(self.origins),
            "semantic_score": self.semantic_score,
            "target_indices": list(self.target_indices),
            "typed_incidence_matched": self.typed_incidence_matched,
            "typed_incidence_total": self.typed_incidence_total,
            "typed_incidence_verified": self.typed_incidence_verified,
        }

    def __post_init__(self) -> None:
        if self.proposal_hash != strict_content_hash(self._body()):
            raise SlotSetMappingError("SCAR_MAPPING_AUTHORITY_INVALID")


@dataclass(frozen=True, slots=True)
class ArmChoiceV1:
    arm: MappingArm
    disposition: ChoiceDisposition
    proposal_hash: str | None
    reason_ids: tuple[str, ...]

    def safe_payload(self) -> dict[str, object]:
        return {
            "arm": self.arm.value,
            "disposition": self.disposition.value,
            "proposal_hash": self.proposal_hash,
            "reason_ids": list(self.reason_ids),
        }


@dataclass(frozen=True, slots=True)
class _DerivedMapping:
    proposals: tuple[MappingProposalV1, ...]
    choices: tuple[ArmChoiceV1, ...]
    assignment_subproblems_solved: int
    target_color_shuffle_effective: bool
    receipt_bytes: bytes


@dataclass(frozen=True, slots=True)
class SlotSetMappingResultV1:
    proposals: tuple[MappingProposalV1, ...]
    choices: tuple[ArmChoiceV1, ...]
    assignment_subproblems_solved: int
    target_color_shuffle_effective: bool
    receipt_bytes: bytes
    _source: SlotGraphV1 = field(repr=False, compare=False)
    _target: SlotGraphV1 = field(repr=False, compare=False)
    _scores: SemanticSlotScoreMatrixV1 = field(repr=False, compare=False)
    _target_color_shuffle: bool = field(repr=False, compare=False)
    _marker: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._marker is not _RESULT_MARKER:
            raise SlotSetMappingError("SCAR_MAPPING_AUTHORITY_INVALID")
        expected = _derive_mapping(
            self._source,
            self._target,
            self._scores,
            target_color_shuffle=self._target_color_shuffle,
        )
        if (
            self.proposals != expected.proposals
            or self.choices != expected.choices
            or self.assignment_subproblems_solved
            != expected.assignment_subproblems_solved
            or self.target_color_shuffle_effective
            is not expected.target_color_shuffle_effective
            or self.receipt_bytes != expected.receipt_bytes
        ):
            raise SlotSetMappingError("SCAR_MAPPING_AUTHORITY_INVALID")

    @property
    def receipt(self) -> Mapping[str, object]:
        return _read_receipt(self.receipt_bytes, schema=MAPPING_RECEIPT_SCHEMA)

    def choice(self, arm: MappingArm) -> ArmChoiceV1:
        return next(row for row in self.choices if row.arm is arm)


@dataclass(frozen=True, slots=True)
class _Edge:
    slot0: int
    slot1: int
    color: tuple[str, str, str, str]


def _invert_orientation(value: str) -> str:
    return {"none": "none", "forward": "reverse", "reverse": "forward"}[value]


def _transform_edges(
    graph: SlotGraphV1, operator: CategoricalOperatorV1
) -> tuple[_Edge, ...]:
    indices = {row.slot_id: index for index, row in enumerate(graph.slots)}
    edges: list[_Edge] = []
    for relation in graph.relations:
        slot0 = indices[relation.slot0_id]
        slot1 = indices[relation.slot1_id]
        kind, polarity, temporal, causal = _edge_color(relation)
        if operator.reverse_positional_slots:
            slot0, slot1 = slot1, slot0
        if operator.invert_polarity:
            polarity = {"negative": "positive", "neutral": "neutral", "positive": "negative"}[
                polarity
            ]
        if operator.orientation_inverting:
            temporal = _invert_orientation(temporal)
            causal = _invert_orientation(causal)
        edges.append(_Edge(slot0, slot1, (kind, polarity, temporal, causal)))
    return tuple(edges)


def _target_edges(
    graph: SlotGraphV1, *, shuffle: bool
) -> tuple[tuple[_Edge, ...], bool]:
    identity = OPERATOR_CLOSURE[0]
    edges = _transform_edges(graph, identity)
    if not shuffle or len(edges) < 2:
        return edges, False
    colors = tuple(row.color for row in edges)
    rotated = colors[1:] + colors[:1]
    effective = rotated != colors
    return (
        tuple(
            _Edge(edge.slot0, edge.slot1, rotated[index])
            for index, edge in enumerate(edges)
        ),
        effective,
    )


def _profiles(
    slot_count: int, edges: tuple[_Edge, ...]
) -> tuple[Counter[tuple[object, ...]], ...]:
    rows = [Counter() for _ in range(slot_count)]
    for edge in edges:
        rows[edge.slot0][("slot0", *edge.color)] += 1
        rows[edge.slot1][("slot1", *edge.color)] += 1
    return tuple(rows)


def _profile_score(
    left: Counter[tuple[object, ...]],
    right: Counter[tuple[object, ...]],
) -> int:
    overlap = sum((left & right).values())
    return 2 * overlap - sum(left.values()) - sum(right.values())


def _paths(edges: tuple[_Edge, ...]) -> Counter[tuple[object, ...]]:
    rows: Counter[tuple[object, ...]] = Counter()
    for first in edges:
        for second in edges:
            if first.slot1 == second.slot0:
                rows[(
                    first.slot0,
                    first.slot1,
                    second.slot1,
                    *first.color,
                    *second.color,
                )] += 1
    return rows


def _proposal(
    *,
    operator: CategoricalOperatorV1,
    assignment: tuple[int, ...],
    origins: tuple[str, ...],
    source_edges: tuple[_Edge, ...],
    target_edges: tuple[_Edge, ...],
    source_profiles: tuple[Counter[tuple[object, ...]], ...],
    target_profiles: tuple[Counter[tuple[object, ...]], ...],
    semantic_weights: tuple[tuple[int, ...], ...],
) -> MappingProposalV1:
    injective = (
        len(assignment) == len(source_profiles)
        and len(set(assignment)) == len(assignment)
        and all(0 <= row < len(target_profiles) for row in assignment)
    )
    if not injective:
        raise SlotSetMappingError("SCAR_ASSIGNMENT_SOLVER_INVALID")
    semantic = sum(semantic_weights[row][column] for row, column in enumerate(assignment))
    flat = sum(
        _profile_score(source_profiles[row], target_profiles[column])
        for row, column in enumerate(assignment)
    )
    incidence_matched = 0
    incidence_total = 0
    # This verifies only the finite categorical graph supplied to the core.
    # Global relation recall is deliberately outside this certificate: an
    # independently selected partial graph can still have internally valid
    # typed incidence and composition without being a complete world model.
    incidence_ok = True
    for source_index, target_index in enumerate(assignment):
        source_profile = source_profiles[source_index]
        target_profile = target_profiles[target_index]
        incidence_matched += sum((source_profile & target_profile).values())
        incidence_total += sum(source_profile.values())
        if source_profile - target_profile:
            incidence_ok = False
    incidence_ok = incidence_ok and incidence_total > 0

    source_paths = _paths(source_edges)
    target_paths = _paths(target_edges)
    mapped_paths: Counter[tuple[object, ...]] = Counter()
    for path, count in source_paths.items():
        mapped_paths[(
            assignment[int(path[0])],
            assignment[int(path[1])],
            assignment[int(path[2])],
            *path[3:],
        )] += count
    path_matched = sum((mapped_paths & target_paths).values())
    path_total = sum(source_paths.values())
    path_ok = (
        incidence_ok
        and path_total > 0
        and not (mapped_paths - target_paths)
    )
    body = {
        "flat_structural_score": flat,
        "injective_verified": injective,
        "length2_composition_verified": path_ok,
        "length2_path_matched": path_matched,
        "length2_path_total": path_total,
        "operator_id": operator.operator_id,
        "origins": list(origins),
        "semantic_score": semantic,
        "target_indices": list(assignment),
        "typed_incidence_matched": incidence_matched,
        "typed_incidence_total": incidence_total,
        "typed_incidence_verified": incidence_ok,
    }
    return MappingProposalV1(
        operator_id=operator.operator_id,
        target_indices=assignment,
        origins=origins,
        semantic_score=semantic,
        flat_structural_score=flat,
        typed_incidence_matched=incidence_matched,
        typed_incidence_total=incidence_total,
        length2_path_matched=path_matched,
        length2_path_total=path_total,
        injective_verified=injective,
        typed_incidence_verified=incidence_ok,
        length2_composition_verified=path_ok,
        proposal_hash=strict_content_hash(body),
    )


def _choose(
    arm: MappingArm, proposals: tuple[MappingProposalV1, ...]
) -> ArmChoiceV1:
    if arm is MappingArm.SEMANTIC_ONLY:
        identity_operator_id = OPERATOR_CLOSURE[0].operator_id
        eligible = tuple(
            row
            for row in proposals
            if row.operator_id == identity_operator_id
            and "semantic_kbest" in row.origins
        )
        # The secondary key is only the semantic assignment vector.  Neither
        # a structural score, structural verification, proposal hash (which
        # binds structural fields), nor a structure-only proposal can affect
        # this arm.
        key = lambda row: (-row.semantic_score, row.target_indices)
        reason = "no_identity_semantic_kbest_proposal"
    elif arm is MappingArm.FLAT_STRUCTURAL:
        eligible = tuple(row for row in proposals if row.typed_incidence_total > 0)
        key = lambda row: (
            -row.flat_structural_score,
            -row.semantic_score,
            row.proposal_hash,
        )
        reason = "no_categorical_incidence"
    elif arm is MappingArm.FULL_NO_COMPOSITION:
        eligible = tuple(row for row in proposals if row.typed_incidence_verified)
        key = lambda row: (
            -row.typed_incidence_matched,
            -row.flat_structural_score,
            -row.semantic_score,
            row.proposal_hash,
        )
        reason = "no_typed_incidence_verified_proposal"
    else:
        eligible = tuple(
            row
            for row in proposals
            if row.typed_incidence_verified
            and row.length2_composition_verified
        )
        key = lambda row: (
            -row.length2_path_matched,
            -row.typed_incidence_matched,
            -row.flat_structural_score,
            -row.semantic_score,
            row.proposal_hash,
        )
        reason = "no_length2_composition_verified_proposal"
    if not eligible:
        return ArmChoiceV1(
            arm=arm,
            disposition=ChoiceDisposition.ABSTAIN,
            proposal_hash=None,
            reason_ids=(reason,),
        )
    selected = min(eligible, key=key)
    return ArmChoiceV1(
        arm=arm,
        disposition=ChoiceDisposition.SELECTED,
        proposal_hash=selected.proposal_hash,
        reason_ids=(),
    )


def _derive_mapping(
    source: SlotGraphV1,
    target: SlotGraphV1,
    scores: SemanticSlotScoreMatrixV1,
    *,
    target_color_shuffle: bool,
) -> _DerivedMapping:
    if (
        type(source) is not SlotGraphV1
        or type(target) is not SlotGraphV1
        or type(scores) is not SemanticSlotScoreMatrixV1
        or type(target_color_shuffle) is not bool
    ):
        raise SlotSetMappingError("SCAR_MAPPING_AUTHORITY_INVALID")
    source.__post_init__()
    target.__post_init__()
    scores.__post_init__()
    if len(source.slots) > len(target.slots):
        raise SlotSetMappingError("SCAR_RESOURCE_BOUND_EXCEEDED")
    expected_keys = {
        (left.slot_id, right.slot_id)
        for left in source.slots
        for right in target.slots
    }
    score_lookup = {(left, right): value for left, right, value in scores.rows}
    if set(score_lookup) != expected_keys:
        raise SlotSetMappingError("SCAR_SCORE_MATRIX_INCOMPLETE")
    semantic_weights = tuple(
        tuple(score_lookup[(left.slot_id, right.slot_id)] for right in target.slots)
        for left in source.slots
    )
    try:
        semantic_assignments, solved = _assignment._k_best_maximum_injections(
            semantic_weights, k=K_BEST_PER_POOL
        )
    except Exception as exc:
        raise SlotSetMappingError("SCAR_ASSIGNMENT_SOLVER_INVALID") from exc
    target_edges, shuffle_effective = _target_edges(
        target, shuffle=target_color_shuffle
    )
    target_profiles = _profiles(len(target.slots), target_edges)

    origin_map: dict[tuple[str, tuple[int, ...]], set[str]] = {}
    operator_payload: dict[str, tuple[CategoricalOperatorV1, tuple[_Edge, ...], tuple[Counter[tuple[object, ...]], ...]]] = {}
    for operator in OPERATOR_CLOSURE:
        source_edges = _transform_edges(source, operator)
        source_profiles = _profiles(len(source.slots), source_edges)
        operator_payload[operator.operator_id] = (
            operator,
            source_edges,
            source_profiles,
        )
        for assignment in semantic_assignments:
            origin_map.setdefault(
                (operator.operator_id, assignment.target_indices), set()
            ).add("semantic_kbest")
        structure_weights = tuple(
            tuple(
                _profile_score(left, right) for right in target_profiles
            )
            for left in source_profiles
        )
        try:
            structural_assignments, count = _assignment._k_best_maximum_injections(
                structure_weights, k=K_BEST_PER_POOL
            )
        except Exception as exc:
            raise SlotSetMappingError("SCAR_ASSIGNMENT_SOLVER_INVALID") from exc
        solved += count
        for assignment in structural_assignments:
            origin_map.setdefault(
                (operator.operator_id, assignment.target_indices), set()
            ).add("structure_kbest")

    if solved > MAXIMUM_ASSIGNMENT_SUBPROBLEMS or len(origin_map) > MAXIMUM_PROPOSALS:
        raise SlotSetMappingError("SCAR_RESOURCE_BOUND_EXCEEDED")
    proposals: list[MappingProposalV1] = []
    for (operator_id, assignment), origins in origin_map.items():
        operator, source_edges, source_profiles = operator_payload[operator_id]
        proposals.append(
            _proposal(
                operator=operator,
                assignment=assignment,
                origins=tuple(sorted(origins)),
                source_edges=source_edges,
                target_edges=target_edges,
                source_profiles=source_profiles,
                target_profiles=target_profiles,
                semantic_weights=semantic_weights,
            )
        )
    canonical_proposals = tuple(sorted(proposals, key=lambda row: row.proposal_hash))
    choices = tuple(_choose(arm, canonical_proposals) for arm in MappingArm)
    body = {
        "assignment_algorithm": _assignment.UNIT_MAPPING_ALGORITHM,
        "assignment_subproblems_solved": solved,
        "candidate_pool_union": ["semantic_kbest", "structure_kbest"],
        "claim_scope": "source_free_categorical_slot_graph_proposal_consistency",
        "effect_authority_established": False,
        "external_pair_label_access_count": 0,
        "formal_law_binding_count": 0,
        "full_verifier_checks": [
            "injectivity",
            "typed_local_incidence",
            "ordered_length2_path_composition",
        ],
        "k_best_per_pool": K_BEST_PER_POOL,
        "mapping_choices": [row.safe_payload() for row in choices],
        "maximum_assignment_subproblems": MAXIMUM_ASSIGNMENT_SUBPROBLEMS,
        "maximum_proposals": MAXIMUM_PROPOSALS,
        "online_evaluator_access_count": 0,
        "operator_closure": [row.safe_payload() for row in OPERATOR_CLOSURE],
        "pair_label_or_gold_access_count": 0,
        "positional_slot0_slot1_only": True,
        "proposal_count": len(canonical_proposals),
        "proposal_set_commitment": strict_content_hash(
            [row._body() | {"proposal_hash": row.proposal_hash} for row in canonical_proposals]
        ),
        "schema": MAPPING_RECEIPT_SCHEMA,
        "score_matrix_commitment": scores.commitment,
        "scorer_access_count": 0,
        "semantic_only_candidate_origin": "semantic_kbest",
        "semantic_only_operator_id": OPERATOR_CLOSURE[0].operator_id,
        "semantic_only_tie_break": "lexicographic_target_assignment_only",
        "semantic_only_uses_structural_scores": False,
        "semantic_only_uses_structure_only_proposals": False,
        "selected_graph_verification_requires_global_coverage": False,
        "selected_graph_verification_scope": (
            "finite_supplied_categorical_graph_only"
        ),
        "source_input_coverage_complete": source.coverage_complete,
        "source_graph_evidence_binding_sha256": source.graph_evidence_binding_sha256,
        "source_graph_shape_sha256": source.receipt["rename_invariant_shape_sha256"],
        "source_archive_access_count": 0,
        "structure_only_pool_uses_semantic_scores": False,
        "target_color_shuffle_effective": shuffle_effective,
        "target_color_shuffle_requested": target_color_shuffle,
        "target_input_coverage_complete": target.coverage_complete,
        "target_graph_evidence_binding_sha256": target.graph_evidence_binding_sha256,
        "target_graph_shape_sha256": target.receipt["rename_invariant_shape_sha256"],
        "relation_recall_total": False,
        "typed_incidence_verified_proposal_count": sum(
            row.typed_incidence_verified for row in canonical_proposals
        ),
        "version": VERSION,
        "length2_composition_verified_proposal_count": sum(
            row.length2_composition_verified for row in canonical_proposals
        ),
    }
    return _DerivedMapping(
        proposals=canonical_proposals,
        choices=choices,
        assignment_subproblems_solved=solved,
        target_color_shuffle_effective=shuffle_effective,
        receipt_bytes=_self_seal(body),
    )


def map_slot_graphs_v1(
    source: SlotGraphV1,
    target: SlotGraphV1,
    semantic_scores: SemanticSlotScoreMatrixV1,
    *,
    target_color_shuffle: bool = False,
) -> SlotSetMappingResultV1:
    derived = _derive_mapping(
        source,
        target,
        semantic_scores,
        target_color_shuffle=target_color_shuffle,
    )
    return SlotSetMappingResultV1(
        proposals=derived.proposals,
        choices=derived.choices,
        assignment_subproblems_solved=derived.assignment_subproblems_solved,
        target_color_shuffle_effective=derived.target_color_shuffle_effective,
        receipt_bytes=derived.receipt_bytes,
        _source=source,
        _target=target,
        _scores=semantic_scores,
        _target_color_shuffle=target_color_shuffle,
        _marker=_RESULT_MARKER,
    )


__all__ = [
    "ArmChoiceV1",
    "CategoricalOperatorV1",
    "ChoiceDisposition",
    "EXACT_OWNERSHIP_SCHEMA",
    "ExactOwnershipQualificationV1",
    "K_BEST_PER_POOL",
    "MAPPING_RECEIPT_SCHEMA",
    "MAXIMUM_ASSIGNMENT_SUBPROBLEMS",
    "MAXIMUM_PROPOSALS",
    "MAXIMUM_RELATIONS_PER_SIDE",
    "MAXIMUM_SLOTS_PER_SIDE",
    "MappingArm",
    "MappingProposalV1",
    "OPERATOR_CLOSURE",
    "SLOT_GRAPH_SCHEMA",
    "SemanticSlotScoreMatrixV1",
    "SlotGraphV1",
    "SlotNodeV1",
    "SlotRelationInputV1",
    "SlotSetMappingError",
    "SlotSetMappingResultV1",
    "VERSION",
    "build_slot_graph_v1",
    "map_slot_graphs_v1",
    "qualify_exact_bounded_slot_ownership",
]
