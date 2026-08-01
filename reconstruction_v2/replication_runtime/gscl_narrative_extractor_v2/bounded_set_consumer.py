"""Bounded, law-neutral consumer for the v2 document-envelope ABI.

The consumer is deliberately additive.  It never converts a document result
back into the old single-source ``NarrativeExtraction`` ABI and it never
manufactures quantities, constraints, law bindings, or numeric observables.
It projects only already-grounded ``EXTRACTED`` relations into a finite set of
mention-local structural units while accounting for every upstream segment.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import re
from types import MappingProxyType
from typing import Any, Mapping

from assumption_agent.generalized_structural_correspondence_v1 import (
    EvidenceSpanRef,
    GSCLSchemaRegistry,
    InferenceProvenance,
    ObservationStatus,
    RoleTargetKind,
    StructuralEpisode,
    StructuralObject,
    StructuralRelation,
    strict_canonical_bytes,
    strict_content_hash,
)

from . import document_envelope


VERSION = "gscl_bounded_narrative_relation_set_consumer_v1"
RECEIPT_SCHEMA = f"{VERSION}.safe_receipt.v1"
SIGNATURE_SCHEMA = f"{VERSION}.categorical_set_signature.v1"
MAXIMUM_RELATION_UNITS = document_envelope.MAXIMUM_PROJECTED_RELATIONS
MAXIMUM_STRUCTURAL_OBJECTS = 2 * MAXIMUM_RELATION_UNITS
MAXIMUM_EVIDENCE_SPANS = document_envelope.MAXIMUM_PROJECTED_MENTIONS

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RESULT_MARKER = object()

_KIND_SYMBOL = MappingProxyType(
    {
        "relation": "NarrativeOrderedSlotsRelation",
        "state_change": "NarrativeOrderedSlotsStateChange",
        "temporal": "NarrativeOrderedSlotsTemporal",
        "causal": "NarrativeOrderedSlotsCausal",
    }
)
_POLARITY_SYMBOL = MappingProxyType(
    {"positive": "Pos", "negative": "Neg", "neutral": "Neu"}
)
_ORIENTATION_SYMBOL = MappingProxyType(
    {"none": "None", "forward": "Fwd", "reverse": "Rev"}
)

_CONSUMER_CONTRACT = {
    "version": VERSION,
    "input_abi": document_envelope.VERSION,
    "output_abi": "BoundedNarrativeRelationSetV1",
    "maximum_relation_units": MAXIMUM_RELATION_UNITS,
    "maximum_structural_objects": MAXIMUM_STRUCTURAL_OBJECTS,
    "maximum_evidence_spans": MAXIMUM_EVIDENCE_SPANS,
    "object_identity": "mention_local_no_quote_coreference",
    "relation_projection": "one_extracted_relation_to_one_ordered_slot_pair_unit",
    "endpoint_semantics": (
        "slot0_and_slot1_are_positional_only_not_source_target_direction"
    ),
    "evidence_projection": "anchor_and_two_endpoint_utf8_byte_spans",
    "semantic_signature": "rename_invariant_categorical_multiset",
    "typed_failure_policy": "block_all_partial_projection",
    "no_relation_policy": "coverage_only_not_negative_edge",
    "short_context_policy": "coverage_only_not_synthetic_relation",
    "numeric_policy": "never_invent_quantity_constraint_or_observable",
    "unit_authority": "only_enclosing_result_full_recomputation_is_authoritative",
}
CONSUMER_CONTRACT_SHA256 = strict_content_hash(_CONSUMER_CONTRACT)


class BoundedSetConsumerError(RuntimeError):
    """A stable, content-free consumer failure."""

    _KNOWN = frozenset(
        {
            "SET_CONSUMER_AUTHORITY_INVALID",
            "SET_CONSUMER_EPISODE_INVALID",
            "SET_CONSUMER_OWNERSHIP_INVALID",
            "SET_CONSUMER_RECEIPT_INVALID",
            "SET_CONSUMER_REGISTRY_INVALID",
            "SET_CONSUMER_RESOURCE_BOUND_EXCEEDED",
            "SET_CONSUMER_SIGNATURE_UNAVAILABLE",
            "SET_CONSUMER_UPSTREAM_INVALID",
        }
    )

    def __init__(self, issue_id: str) -> None:
        if issue_id not in self._KNOWN:
            raise ValueError("bounded_set_consumer_issue_unknown")
        self.issue_id = issue_id
        super().__init__(issue_id)


class SetConsumerDisposition(str, Enum):
    COMPLETE_SELECTED_SET = "COMPLETE_SELECTED_SET"
    PARTIAL_SELECTED_SET = "PARTIAL_SELECTED_SET"
    EMPTY_ABSTENTION = "EMPTY_ABSTENTION"
    TYPED_FAILURE_BLOCKED = "TYPED_FAILURE_BLOCKED"


class LawReadinessDisposition(str, Enum):
    INCONCLUSIVE_MISSING_EVIDENCE = "INCONCLUSIVE_MISSING_EVIDENCE"
    INCONCLUSIVE_PARTIAL_COVERAGE = "INCONCLUSIVE_PARTIAL_COVERAGE"
    INCONCLUSIVE_NO_RELATION_UNITS = "INCONCLUSIVE_NO_RELATION_UNITS"
    BLOCKED_UPSTREAM_TYPED_FAILURE = "BLOCKED_UPSTREAM_TYPED_FAILURE"


def _is_hex64(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _strict_int(value: object, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _stable_id(prefix: str, payload: object) -> str:
    return f"{prefix}.{strict_content_hash(payload)[:24]}"


def _relation_type(
    *, kind: str, polarity: str, temporal: str, causal: str
) -> str:
    try:
        return ".".join(
            (
                _KIND_SYMBOL[kind],
                _POLARITY_SYMBOL[polarity],
                f"T{_ORIENTATION_SYMBOL[temporal]}",
                f"C{_ORIENTATION_SYMBOL[causal]}",
            )
        )
    except KeyError as exc:
        raise BoundedSetConsumerError(
            "SET_CONSUMER_OWNERSHIP_INVALID"
        ) from exc


@dataclass(frozen=True, slots=True)
class SegmentCoverageV1:
    segment_id: str
    parent_sentence_id: str
    core_start_byte: int
    core_end_byte: int
    lexical_token_count: int
    chunk_index: int
    chunk_count: int
    leaf_eligible: bool
    leaf_called: bool
    disposition: str
    error_code: str | None
    relation_count: int

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not isinstance(self.segment_id, str) or not self.segment_id:
            issues.append("coverage_segment_id_invalid")
        if (
            not isinstance(self.parent_sentence_id, str)
            or not self.parent_sentence_id
        ):
            issues.append("coverage_parent_id_invalid")
        if not _strict_int(self.core_start_byte):
            issues.append("coverage_start_invalid")
        if (
            not _strict_int(self.core_end_byte, minimum=1)
            or self.core_end_byte <= self.core_start_byte
        ):
            issues.append("coverage_end_invalid")
        for value, issue in (
            (self.lexical_token_count, "coverage_token_count_invalid"),
            (self.chunk_index, "coverage_chunk_index_invalid"),
            (self.chunk_count, "coverage_chunk_count_invalid"),
            (self.relation_count, "coverage_relation_count_invalid"),
        ):
            if not _strict_int(value):
                issues.append(issue)
        if self.chunk_count < 1 or self.chunk_index >= self.chunk_count:
            issues.append("coverage_chunk_topology_invalid")
        if type(self.leaf_eligible) is not bool:
            issues.append("coverage_leaf_eligibility_invalid")
        if type(self.leaf_called) is not bool:
            issues.append("coverage_leaf_called_invalid")
        if self.disposition not in {
            row.value for row in document_envelope.SegmentDisposition
        }:
            issues.append("coverage_disposition_invalid")
        if self.error_code is not None and not isinstance(
            self.error_code, str
        ):
            issues.append("coverage_error_invalid")
        return tuple(sorted(set(issues)))

    def private_payload(self) -> dict[str, object]:
        return {
            "chunk_count": self.chunk_count,
            "chunk_index": self.chunk_index,
            "core_end_byte": self.core_end_byte,
            "core_start_byte": self.core_start_byte,
            "disposition": self.disposition,
            "error_code": self.error_code,
            "leaf_called": self.leaf_called,
            "leaf_eligible": self.leaf_eligible,
            "lexical_token_count": self.lexical_token_count,
            "parent_sentence_id": self.parent_sentence_id,
            "relation_count": self.relation_count,
            "segment_id": self.segment_id,
        }

    def signature_payload(self) -> dict[str, object]:
        return {
            "disposition": self.disposition,
            "leaf_called": self.leaf_called,
            "leaf_eligible": self.leaf_eligible,
        }


@dataclass(frozen=True, slots=True)
class NarrativeRelationUnitV1:
    """Shape-valid unit; authority comes only from its enclosing result."""

    unit_id: str
    segment_id: str
    parent_sentence_id: str
    slot0_object_id: str
    slot1_object_id: str
    structural_relation_id: str
    anchor_span_id: str
    slot0_span_id: str
    slot1_span_id: str
    generator_kind: str
    polarity: str
    temporal_orientation: str
    causal_orientation: str
    relation_type: str
    semantic_signature_sha256: str
    evidence_binding_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "causal_orientation": self.causal_orientation,
            "endpoint_roles": ["slot0", "slot1"],
            "generator_kind": self.generator_kind,
            "polarity": self.polarity,
            "relation_type": self.relation_type,
            "temporal_orientation": self.temporal_orientation,
        }

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        for value, issue in (
            (self.unit_id, "relation_unit_id_invalid"),
            (self.segment_id, "relation_unit_segment_invalid"),
            (self.parent_sentence_id, "relation_unit_parent_invalid"),
            (self.slot0_object_id, "relation_unit_slot0_invalid"),
            (self.slot1_object_id, "relation_unit_slot1_invalid"),
            (self.structural_relation_id, "relation_unit_relation_invalid"),
            (self.anchor_span_id, "relation_unit_anchor_span_invalid"),
            (self.slot0_span_id, "relation_unit_slot0_span_invalid"),
            (self.slot1_span_id, "relation_unit_slot1_span_invalid"),
        ):
            if not isinstance(value, str) or not value:
                issues.append(issue)
        if len(
            {
                self.anchor_span_id,
                self.slot0_span_id,
                self.slot1_span_id,
            }
        ) != 3:
            issues.append("relation_unit_span_ownership_invalid")
        try:
            expected_type = _relation_type(
                kind=self.generator_kind,
                polarity=self.polarity,
                temporal=self.temporal_orientation,
                causal=self.causal_orientation,
            )
        except BoundedSetConsumerError:
            expected_type = None
        if self.relation_type != expected_type:
            issues.append("relation_unit_type_invalid")
        if self.semantic_signature_sha256 != strict_content_hash(
            self.semantic_payload()
        ):
            issues.append("relation_unit_signature_invalid")
        if not _is_hex64(self.evidence_binding_sha256):
            issues.append("relation_unit_evidence_binding_invalid")
        return tuple(sorted(set(issues)))

    def safe_payload(self) -> dict[str, object]:
        return {
            "evidence_binding_sha256": self.evidence_binding_sha256,
            "parent_sentence_id": self.parent_sentence_id,
            "segment_id": self.segment_id,
            "semantic_signature_sha256": self.semantic_signature_sha256,
            "structural_relation_id": self.structural_relation_id,
            "unit_id": self.unit_id,
        }


@dataclass(frozen=True, slots=True)
class LawReadinessV1:
    law_id: str
    disposition: LawReadinessDisposition
    missing_role_ids: tuple[str, ...]
    missing_observable_ids: tuple[str, ...]
    reasons: tuple[str, ...]

    def safe_payload(self) -> dict[str, object]:
        return {
            "disposition": self.disposition.value,
            "law_id": self.law_id,
            "missing_observable_ids": list(self.missing_observable_ids),
            "missing_role_ids": list(self.missing_role_ids),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class _Derived:
    disposition: SetConsumerDisposition
    coverage: tuple[SegmentCoverageV1, ...]
    units: tuple[NarrativeRelationUnitV1, ...]
    structural_episode: StructuralEpisode | None
    signature_bytes: bytes | None
    signature_sha256: str | None
    evidence_binding_sha256: str


@dataclass(frozen=True, slots=True)
class BoundedNarrativeRelationSetV1:
    """Validated result; source text and leaf decisions remain upstream."""

    upstream_envelope: document_envelope.NarrativeDocumentEnvelopeV1 = field(
        repr=False, compare=False
    )
    disposition: SetConsumerDisposition
    coverage: tuple[SegmentCoverageV1, ...]
    units: tuple[NarrativeRelationUnitV1, ...]
    structural_episode: StructuralEpisode | None = field(repr=False)
    relation_set_signature_bytes: bytes | None
    relation_set_signature_sha256: str | None
    evidence_binding_sha256: str
    receipt_bytes: bytes
    _marker: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._marker is not _RESULT_MARKER:
            raise BoundedSetConsumerError(
                "SET_CONSUMER_AUTHORITY_INVALID"
            )
        expected = _derive(self.upstream_envelope)
        if (
            self.disposition is not expected.disposition
            or self.coverage != expected.coverage
            or self.units != expected.units
            or self.structural_episode != expected.structural_episode
            or self.relation_set_signature_bytes
            != expected.signature_bytes
            or self.relation_set_signature_sha256
            != expected.signature_sha256
            or self.evidence_binding_sha256
            != expected.evidence_binding_sha256
        ):
            raise BoundedSetConsumerError(
                "SET_CONSUMER_OWNERSHIP_INVALID"
            )
        expected_receipt = _receipt_bytes(expected, self.upstream_envelope)
        if self.receipt_bytes != expected_receipt:
            raise BoundedSetConsumerError(
                "SET_CONSUMER_RECEIPT_INVALID"
            )

    @property
    def receipt(self) -> Mapping[str, object]:
        try:
            value = json.loads(self.receipt_bytes.decode("ascii"))
        except Exception as exc:
            raise BoundedSetConsumerError(
                "SET_CONSUMER_RECEIPT_INVALID"
            ) from exc
        if type(value) is not dict:
            raise BoundedSetConsumerError(
                "SET_CONSUMER_RECEIPT_INVALID"
            )
        return MappingProxyType(value)

    @property
    def selected_set_available(self) -> bool:
        return self.structural_episode is not None

    def safe_payload(self) -> Mapping[str, object]:
        return self.receipt


def _coverage(
    envelope: document_envelope.NarrativeDocumentEnvelopeV1,
) -> tuple[SegmentCoverageV1, ...]:
    rows = tuple(
        SegmentCoverageV1(
            segment_id=outcome.plan.segment_id,
            parent_sentence_id=outcome.plan.parent_sentence_id,
            core_start_byte=outcome.plan.core_start_byte,
            core_end_byte=outcome.plan.core_end_byte,
            lexical_token_count=outcome.plan.lexical_token_count,
            chunk_index=outcome.plan.chunk_index,
            chunk_count=outcome.plan.chunk_count,
            leaf_eligible=outcome.plan.leaf_eligible,
            leaf_called=outcome.leaf_called,
            disposition=outcome.disposition.value,
            error_code=outcome.error_code,
            relation_count=len(outcome.relation_ids),
        )
        for outcome in envelope.segments
    )
    if any(row.validate() for row in rows):
        raise BoundedSetConsumerError(
            "SET_CONSUMER_UPSTREAM_INVALID"
        )
    return rows


def _span_for_mention(
    source_sha256: str,
    mention: document_envelope.ProjectedMention,
) -> EvidenceSpanRef:
    return EvidenceSpanRef(
        span_id=_stable_id(
            "span",
            {
                "mention_id": mention.mention_id,
                "source_sha256": source_sha256,
            },
        ),
        source_sha256=source_sha256,
        start_byte=mention.start_byte,
        end_byte=mention.end_byte,
        span_sha256=mention.quote_sha256,
    )


def _input_evidence_hash(spans: tuple[EvidenceSpanRef, ...]) -> str:
    return strict_content_hash(
        [
            row.private_payload()
            for row in sorted(spans, key=lambda item: item.span_id)
        ]
    )


def _provenance(
    *,
    spans: tuple[EvidenceSpanRef, ...],
    outcome: document_envelope.SegmentOutcome,
) -> InferenceProvenance:
    implementation_hash = strict_content_hash(
        {
            "consumer_contract_sha256": CONSUMER_CONTRACT_SHA256,
            "leaf_decision_sha256": outcome.leaf_decision_sha256,
            "leaf_parser_provenance_hash": (
                outcome.leaf_parser_provenance_hash
            ),
            "leaf_receipt_sha256": outcome.leaf_receipt_sha256,
        }
    )
    return InferenceProvenance(
        extractor_id="gscl.document_envelope_leaf",
        extractor_version=VERSION,
        extractor_implementation_hash=implementation_hash,
        input_evidence_hash=_input_evidence_hash(spans),
        calibration_bucket="qualification.unscored",
        alternative_binding_hashes=(),
    )


def _build_units_and_episode(
    envelope: document_envelope.NarrativeDocumentEnvelopeV1,
) -> tuple[
    tuple[NarrativeRelationUnitV1, ...],
    StructuralEpisode | None,
]:
    source_sha = str(envelope.receipt["root_source_sha256"])
    mentions = {row.mention_id: row for row in envelope.mentions}
    outcomes = {row.plan.segment_id: row for row in envelope.segments}
    spans_by_mention = {
        mention_id: _span_for_mention(source_sha, mention)
        for mention_id, mention in mentions.items()
    }
    units: list[NarrativeRelationUnitV1] = []
    objects: list[StructuralObject] = []
    relations: list[StructuralRelation] = []
    used_mentions: set[str] = set()
    used_spans: dict[str, EvidenceSpanRef] = {}

    for projected in envelope.relations:
        outcome = outcomes.get(projected.segment_id)
        identifiers = (
            projected.anchor_mention_id,
            projected.slot_mention_ids[0],
            projected.slot_mention_ids[1],
        )
        if (
            outcome is None
            or outcome.disposition
            is not document_envelope.SegmentDisposition.EXTRACTED
            or outcome.leaf_decision is None
            or any(row not in mentions for row in identifiers)
            or any(row in used_mentions for row in identifiers)
        ):
            raise BoundedSetConsumerError(
                "SET_CONSUMER_OWNERSHIP_INVALID"
            )
        anchor_span, slot0_span, slot1_span = tuple(
            spans_by_mention[row] for row in identifiers
        )
        for span in (anchor_span, slot0_span, slot1_span):
            used_spans[span.span_id] = span
        slot0_id = _stable_id(
            "object",
            {"mention_id": identifiers[1], "role": "slot0"},
        )
        slot1_id = _stable_id(
            "object",
            {"mention_id": identifiers[2], "role": "slot1"},
        )
        structural_relation_id = _stable_id(
            "relation", {"projected_relation_id": projected.relation_id}
        )
        unit_id = _stable_id(
            "unit",
            {
                "projected_relation_id": projected.relation_id,
                "segment_id": projected.segment_id,
            },
        )
        relation_type = _relation_type(
            kind=projected.generator_kind,
            polarity=projected.polarity,
            temporal=projected.temporal_orientation,
            causal=projected.causal_orientation,
        )
        slot0_provenance = _provenance(
            spans=(slot0_span,), outcome=outcome
        )
        slot1_provenance = _provenance(
            spans=(slot1_span,), outcome=outcome
        )
        relation_spans = (anchor_span, slot0_span, slot1_span)
        relation_provenance = _provenance(
            spans=relation_spans, outcome=outcome
        )
        objects.extend(
            (
                StructuralObject(
                    object_id=slot0_id,
                    object_type="NarrativeMention",
                    evidence_span_ids=(slot0_span.span_id,),
                    observation_status=ObservationStatus.INFERRED,
                    inference_provenance=slot0_provenance,
                ),
                StructuralObject(
                    object_id=slot1_id,
                    object_type="NarrativeMention",
                    evidence_span_ids=(slot1_span.span_id,),
                    observation_status=ObservationStatus.INFERRED,
                    inference_provenance=slot1_provenance,
                ),
            )
        )
        relations.append(
            StructuralRelation(
                relation_id=structural_relation_id,
                relation_type=relation_type,
                # StructuralRelation calls these fields source/target.  This
                # consumer uses them only as storage for upstream slot0/slot1;
                # neither the unit ABI nor its receipt grants direction.
                source_object_id=slot0_id,
                target_object_id=slot1_id,
                evidence_span_ids=tuple(
                    sorted(row.span_id for row in relation_spans)
                ),
                observation_status=ObservationStatus.INFERRED,
                inference_provenance=relation_provenance,
                order_index=None,
            )
        )
        semantic_payload = {
            "causal_orientation": projected.causal_orientation,
            "endpoint_roles": ["slot0", "slot1"],
            "generator_kind": projected.generator_kind,
            "polarity": projected.polarity,
            "relation_type": relation_type,
            "temporal_orientation": projected.temporal_orientation,
        }
        evidence_binding = strict_content_hash(
            {
                "anchor_span": anchor_span.private_payload(),
                "leaf_decision_sha256": outcome.leaf_decision_sha256,
                "leaf_parser_provenance_hash": (
                    outcome.leaf_parser_provenance_hash
                ),
                "leaf_receipt_sha256": outcome.leaf_receipt_sha256,
                "parent_sentence_id": projected.parent_sentence_id,
                "projected_relation_id": projected.relation_id,
                "segment_id": projected.segment_id,
                "slot0_span": slot0_span.private_payload(),
                "slot1_span": slot1_span.private_payload(),
            }
        )
        units.append(
            NarrativeRelationUnitV1(
                unit_id=unit_id,
                segment_id=projected.segment_id,
                parent_sentence_id=projected.parent_sentence_id,
                slot0_object_id=slot0_id,
                slot1_object_id=slot1_id,
                structural_relation_id=structural_relation_id,
                anchor_span_id=anchor_span.span_id,
                slot0_span_id=slot0_span.span_id,
                slot1_span_id=slot1_span.span_id,
                generator_kind=projected.generator_kind,
                polarity=projected.polarity,
                temporal_orientation=projected.temporal_orientation,
                causal_orientation=projected.causal_orientation,
                relation_type=relation_type,
                semantic_signature_sha256=strict_content_hash(
                    semantic_payload
                ),
                evidence_binding_sha256=evidence_binding,
            )
        )
        used_mentions.update(identifiers)

    if len(used_mentions) != len(envelope.mentions):
        raise BoundedSetConsumerError(
            "SET_CONSUMER_OWNERSHIP_INVALID"
        )
    units_tuple = tuple(units)
    if not units_tuple:
        return (), None
    episode = StructuralEpisode(
        episode_id=_stable_id(
            "episode",
            {
                "envelope_receipt_sha256": hashlib.sha256(
                    envelope.receipt_bytes
                ).hexdigest(),
                "source_sha256": source_sha,
            },
        ),
        source_sha256=source_sha,
        evidence_spans=tuple(
            sorted(used_spans.values(), key=lambda row: row.span_id)
        ),
        objects=tuple(sorted(objects, key=lambda row: row.object_id)),
        relations=tuple(
            sorted(relations, key=lambda row: row.relation_id)
        ),
        quantities=(),
        hyperrelations=(),
        constraints=(),
        observables=(),
        declared_boundary_object_id=None,
        missing_observables=(),
    )
    if episode.validate() or episode.verify_source_bytes(
        envelope.source_text.encode("utf-8")
    ):
        raise BoundedSetConsumerError(
            "SET_CONSUMER_EPISODE_INVALID"
        )
    return units_tuple, episode


def _signature_payload(
    *,
    disposition: SetConsumerDisposition,
    coverage: tuple[SegmentCoverageV1, ...],
    units: tuple[NarrativeRelationUnitV1, ...],
) -> dict[str, object]:
    unit_counts = Counter(
        row.semantic_signature_sha256 for row in units
    )
    disposition_counts = Counter(row.disposition for row in coverage)
    coverage_shapes = Counter(
        strict_content_hash(row.signature_payload()) for row in coverage
    )
    return {
        "consumer_contract_sha256": CONSUMER_CONTRACT_SHA256,
        "coverage_class": disposition.value,
        "coverage_disposition_counts": {
            key: disposition_counts.get(key, 0)
            for key in sorted(
                row.value
                for row in document_envelope.SegmentDisposition
            )
        },
        "coverage_shape_multiset": [
            {"count": count, "shape_sha256": digest}
            for digest, count in sorted(coverage_shapes.items())
        ],
        "schema": SIGNATURE_SCHEMA,
        "segment_count": len(coverage),
        "unit_count": len(units),
        "unit_signature_multiset": [
            {"count": count, "unit_signature_sha256": digest}
            for digest, count in sorted(unit_counts.items())
        ],
        "version": VERSION,
    }


def _derive(
    envelope: document_envelope.NarrativeDocumentEnvelopeV1,
) -> _Derived:
    if type(envelope) is not document_envelope.NarrativeDocumentEnvelopeV1:
        raise BoundedSetConsumerError(
            "SET_CONSUMER_UPSTREAM_INVALID"
        )
    try:
        document_envelope._validate_envelope(envelope)
        upstream_receipt = dict(envelope.receipt)
    except Exception as exc:
        raise BoundedSetConsumerError(
            "SET_CONSUMER_UPSTREAM_INVALID"
        ) from exc
    if (
        upstream_receipt.get("schema") != document_envelope.RECEIPT_SCHEMA
        or upstream_receipt.get("version") != document_envelope.VERSION
        or upstream_receipt.get("segmentation_policy_sha256")
        != document_envelope.SEGMENTATION_POLICY_SHA256
        or upstream_receipt.get("byte_outcome_coverage_complete") is not True
        or upstream_receipt.get("formal_leaf_authority_established") is not False
        or upstream_receipt.get("downstream_eligible") is not False
        or upstream_receipt.get("relation_recall_total") is not False
    ):
        raise BoundedSetConsumerError(
            "SET_CONSUMER_UPSTREAM_INVALID"
        )
    coverage = _coverage(envelope)
    typed_failure = any(
        row.disposition
        == document_envelope.SegmentDisposition.TYPED_FAILURE.value
        for row in coverage
    )
    if typed_failure:
        units: tuple[NarrativeRelationUnitV1, ...] = ()
        episode = None
        disposition = SetConsumerDisposition.TYPED_FAILURE_BLOCKED
    else:
        units, episode = _build_units_and_episode(envelope)
        if not units:
            disposition = SetConsumerDisposition.EMPTY_ABSTENTION
        elif all(
            row.disposition
            == document_envelope.SegmentDisposition.EXTRACTED.value
            for row in coverage
        ):
            disposition = SetConsumerDisposition.COMPLETE_SELECTED_SET
        else:
            disposition = SetConsumerDisposition.PARTIAL_SELECTED_SET
    if (
        len(units) > MAXIMUM_RELATION_UNITS
        or (episode is not None and len(episode.objects) > MAXIMUM_STRUCTURAL_OBJECTS)
        or (episode is not None and len(episode.evidence_spans) > MAXIMUM_EVIDENCE_SPANS)
        or len(units) != (0 if episode is None else len(episode.relations))
    ):
        raise BoundedSetConsumerError(
            "SET_CONSUMER_RESOURCE_BOUND_EXCEEDED"
        )
    if episode is None:
        signature_bytes = None
        signature_sha = None
    else:
        signature_bytes = strict_canonical_bytes(
            _signature_payload(
                disposition=disposition,
                coverage=coverage,
                units=units,
            )
        )
        signature_sha = hashlib.sha256(signature_bytes).hexdigest()
    evidence_binding = strict_content_hash(
        {
            "consumer_contract_sha256": CONSUMER_CONTRACT_SHA256,
            "coverage": [row.private_payload() for row in coverage],
            "episode_hash": None if episode is None else episode.episode_hash,
            "relation_units": [row.safe_payload() for row in units],
            "upstream_envelope_receipt_sha256": hashlib.sha256(
                envelope.receipt_bytes
            ).hexdigest(),
            "upstream_root_source_sha256": upstream_receipt[
                "root_source_sha256"
            ],
        }
    )
    return _Derived(
        disposition=disposition,
        coverage=coverage,
        units=units,
        structural_episode=episode,
        signature_bytes=signature_bytes,
        signature_sha256=signature_sha,
        evidence_binding_sha256=evidence_binding,
    )


def _receipt_bytes(
    derived: _Derived,
    envelope: document_envelope.NarrativeDocumentEnvelopeV1,
) -> bytes:
    upstream = envelope.receipt
    counts = Counter(row.disposition for row in derived.coverage)
    body: dict[str, Any] = {
        "claim_scope": "bounded_grounded_categorical_selected_set_only",
        "consumer_contract_sha256": CONSUMER_CONTRACT_SHA256,
        "correspondence_acceptance_established": False,
        "directed_endpoint_semantics_established": False,
        "coverage_commitment": strict_content_hash(
            [row.private_payload() for row in derived.coverage]
        ),
        "coverage_disposition_counts": {
            key: counts.get(key, 0)
            for key in sorted(
                row.value
                for row in document_envelope.SegmentDisposition
            )
        },
        "disposition": derived.disposition.value,
        "effect_or_quality_evidence": False,
        "evidence_binding_sha256": derived.evidence_binding_sha256,
        "free_form_generation_count": 0,
        "law_binding_count": 0,
        "law_evaluation_eligible": False,
        "numeric_observable_count": 0,
        "quantity_count": 0,
        "relation_recall_total": False,
        "structural_source_target_fields_are_positional_slots": True,
        "relation_set_signature_sha256": derived.signature_sha256,
        "schema": RECEIPT_SCHEMA,
        "segment_count": len(derived.coverage),
        "selected_set_available": derived.structural_episode is not None,
        "standalone_unit_authority_established": False,
        "structural_episode_hash": (
            None
            if derived.structural_episode is None
            else derived.structural_episode.episode_hash
        ),
        "structural_object_count": (
            0
            if derived.structural_episode is None
            else len(derived.structural_episode.objects)
        ),
        "structural_relation_count": len(derived.units),
        "upstream_downstream_eligible": upstream[
            "downstream_eligible"
        ],
        "upstream_envelope_receipt_sha256": hashlib.sha256(
            envelope.receipt_bytes
        ).hexdigest(),
        "upstream_envelope_self_sha256": upstream["self_sha256"],
        "upstream_envelope_version": upstream["version"],
        "upstream_root_source_sha256": upstream["root_source_sha256"],
        "version": VERSION,
    }
    return strict_canonical_bytes(
        {**body, "self_sha256": strict_content_hash(body)}
    )


def consume_document_envelope(
    envelope: document_envelope.NarrativeDocumentEnvelopeV1,
) -> BoundedNarrativeRelationSetV1:
    """Consume one exact envelope under the immutable set policy."""

    derived = _derive(envelope)
    receipt = _receipt_bytes(derived, envelope)
    return BoundedNarrativeRelationSetV1(
        upstream_envelope=envelope,
        disposition=derived.disposition,
        coverage=derived.coverage,
        units=derived.units,
        structural_episode=derived.structural_episode,
        relation_set_signature_bytes=derived.signature_bytes,
        relation_set_signature_sha256=derived.signature_sha256,
        evidence_binding_sha256=derived.evidence_binding_sha256,
        receipt_bytes=receipt,
        _marker=_RESULT_MARKER,
    )


def canonical_relation_set_signature(
    result: BoundedNarrativeRelationSetV1,
) -> Mapping[str, object]:
    """Return the rename-invariant categorical multiset signature."""

    if (
        type(result) is not BoundedNarrativeRelationSetV1
        or result._marker is not _RESULT_MARKER
        or result.relation_set_signature_bytes is None
    ):
        raise BoundedSetConsumerError(
            "SET_CONSUMER_SIGNATURE_UNAVAILABLE"
        )
    try:
        value = json.loads(
            result.relation_set_signature_bytes.decode("ascii")
        )
    except Exception as exc:
        raise BoundedSetConsumerError(
            "SET_CONSUMER_SIGNATURE_UNAVAILABLE"
        ) from exc
    if (
        type(value) is not dict
        or strict_canonical_bytes(value)
        != result.relation_set_signature_bytes
        or hashlib.sha256(result.relation_set_signature_bytes).hexdigest()
        != result.relation_set_signature_sha256
    ):
        raise BoundedSetConsumerError(
            "SET_CONSUMER_SIGNATURE_UNAVAILABLE"
        )
    return MappingProxyType(value)


def _targets_for_kind(
    episode: StructuralEpisode | None, kind: RoleTargetKind
) -> tuple[object, ...]:
    if episode is None:
        return ()
    return {
        RoleTargetKind.OBJECT: episode.objects,
        RoleTargetKind.RELATION: episode.relations,
        RoleTargetKind.QUANTITY: episode.quantities,
        RoleTargetKind.HYPERRELATION: episode.hyperrelations,
        RoleTargetKind.CONSTRAINT: episode.constraints,
    }[kind]


def _target_type(target: object) -> str | None:
    if isinstance(target, StructuralObject):
        return target.object_type
    if isinstance(target, StructuralRelation):
        return target.relation_type
    return None


def assess_law_readiness(
    result: BoundedNarrativeRelationSetV1,
    registry: GSCLSchemaRegistry,
) -> tuple[LawReadinessV1, ...]:
    """Report missing law evidence without creating bindings or values."""

    if (
        type(result) is not BoundedNarrativeRelationSetV1
        or result._marker is not _RESULT_MARKER
        or type(registry) is not GSCLSchemaRegistry
        or registry.validate_frozen_contract()
    ):
        raise BoundedSetConsumerError(
            "SET_CONSUMER_REGISTRY_INVALID"
        )
    if result.disposition is SetConsumerDisposition.TYPED_FAILURE_BLOCKED:
        disposition = LawReadinessDisposition.BLOCKED_UPSTREAM_TYPED_FAILURE
        coverage_reason = "upstream_typed_failure_blocks_partial_projection"
    elif result.disposition is SetConsumerDisposition.EMPTY_ABSTENTION:
        disposition = LawReadinessDisposition.INCONCLUSIVE_NO_RELATION_UNITS
        coverage_reason = "no_relation_units_without_synthetic_objects"
    elif result.disposition is SetConsumerDisposition.PARTIAL_SELECTED_SET:
        disposition = LawReadinessDisposition.INCONCLUSIVE_PARTIAL_COVERAGE
        coverage_reason = "context_or_no_relation_coverage_is_not_semantic_recall"
    else:
        disposition = LawReadinessDisposition.INCONCLUSIVE_MISSING_EVIDENCE
        coverage_reason = "categorical_units_are_not_law_observables"

    rows: list[LawReadinessV1] = []
    episode = result.structural_episode
    for schema in sorted(registry.schemas, key=lambda row: row.law_id):
        missing_roles: list[str] = []
        for role in schema.roles:
            candidates = _targets_for_kind(episode, role.target_kind)
            if not any(
                _target_type(target) in role.allowed_target_types
                for target in candidates
            ):
                missing_roles.append(role.role_id)
        available_observables = (
            set()
            if episode is None
            else {row.observable_id for row in episode.observables}
        )
        missing_observables = tuple(
            sorted(
                row.observable_id
                for row in schema.required_observables
                if row.observable_id not in available_observables
            )
        )
        reasons = {
            coverage_reason,
            "required_role_types_not_grounded",
            "required_observables_not_grounded",
            "no_law_binding_or_residual_was_constructed",
            "numeric_or_typed_values_were_not_invented",
        }
        rows.append(
            LawReadinessV1(
                law_id=schema.law_id,
                disposition=disposition,
                missing_role_ids=tuple(sorted(missing_roles)),
                missing_observable_ids=missing_observables,
                reasons=tuple(sorted(reasons)),
            )
        )
    return tuple(rows)


__all__ = [
    "BoundedNarrativeRelationSetV1",
    "BoundedSetConsumerError",
    "CONSUMER_CONTRACT_SHA256",
    "LawReadinessDisposition",
    "LawReadinessV1",
    "MAXIMUM_EVIDENCE_SPANS",
    "MAXIMUM_RELATION_UNITS",
    "MAXIMUM_STRUCTURAL_OBJECTS",
    "NarrativeRelationUnitV1",
    "RECEIPT_SCHEMA",
    "SIGNATURE_SCHEMA",
    "SegmentCoverageV1",
    "SetConsumerDisposition",
    "VERSION",
    "assess_law_readiness",
    "canonical_relation_set_signature",
    "consume_document_envelope",
]
