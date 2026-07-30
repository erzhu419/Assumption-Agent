"""Evidence-grounded generalized structural correspondence contracts.

This additive sidecar represents finite typed diagrams ``D=(O,M,H,C)`` and
content-addressed law inputs.  It does not mutate UAO v1, historical receipts,
or treatment compilers.  Text/model extraction and downstream efficacy remain
outside Phase 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from .meta_assumption import UniversalAssumptionOntology
from .universal_assumption_ontology_v1 import (
    T05,
    T09,
    T14,
    T15,
    T17,
    build_universal_assumption_ontology_v1,
)


GSCL_SCHEMA_VERSION = "gscl_schema_v1"
GSCL_RESIDUAL_KERNEL_VERSION = "gscl_residual_kernel_v1"
FROZEN_UAO_V1_ONTOLOGY_HASH = (
    build_universal_assumption_ontology_v1().ontology_hash
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_.-]{2,127}\Z")
_SYMBOL = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}\Z")
_UNIT = re.compile(r"[A-Za-z0-9][A-Za-z0-9*/^_.-]{0,63}\Z")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _valid_identifier(value: object) -> bool:
    return isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None


def _valid_symbol(value: object) -> bool:
    return isinstance(value, str) and _SYMBOL.fullmatch(value) is not None


def _strict_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _unique_strings(
    values: object, *, allow_empty: bool = False
) -> bool:
    return (
        isinstance(values, tuple)
        and (allow_empty or bool(values))
        and len(set(values)) == len(values)
        and all(
            isinstance(value, str)
            and bool(value)
            and value.strip() == value
            for value in values
        )
    )


def _require_strict_json(value: Any, *, path: str = "$") -> None:
    """Reject implicit coercions, floats, sets, tuples and non-finite values."""

    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise TypeError(f"{path}: non-finite float is forbidden")
        raise TypeError(f"{path}: floats are forbidden; use ExactRational")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_strict_json(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path}: JSON object key must be a string")
            _require_strict_json(item, path=f"{path}.{key}")
        return
    raise TypeError(f"{path}: unsupported JSON value {type(value).__name__}")


def strict_canonical_bytes(value: Any) -> bytes:
    _require_strict_json(value)
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def strict_content_hash(value: Any) -> str:
    return hashlib.sha256(strict_canonical_bytes(value)).hexdigest()


class ObservationStatus(str, Enum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class RoleTargetKind(str, Enum):
    OBJECT = "object"
    RELATION = "relation"
    QUANTITY = "quantity"
    HYPERRELATION = "hyperrelation"
    CONSTRAINT = "constraint"


class ObservableValueType(str, Enum):
    EXACT_VECTOR = "exact_vector"
    SIGNED_PERMUTATION = "signed_permutation"
    COMPARABLE_PAIRS = "comparable_pairs"
    DIRECTION = "direction"
    QUANTITY_LEDGER = "quantity_ledger"
    BOUNDARY_DECLARATION = "boundary_declaration"
    FINITE_DOMAIN = "finite_domain"
    FINITE_MAP = "finite_map"
    COMPONENT_SET = "component_set"
    DESIGNATED_PAIR = "designated_pair"
    INTERACTION_EXPECTATION = "interaction_expectation"
    SUBSET_UTILITY_FOLDS = "subset_utility_folds"


class LawKind(str, Enum):
    EQUIVARIANCE = "equivariance"
    MONOTONE_ORDER = "monotone_order"
    CLOSED_BALANCE = "closed_balance"
    PATH_COMPOSITION = "path_composition"
    LOW_ORDER_INTERACTION = "low_order_interaction"


class ResidualDisposition(str, Enum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    NOT_APPLICABLE = "not_applicable"
    INCONCLUSIVE = "inconclusive"


class CorrespondenceDisposition(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class ExactRational:
    numerator: int
    denominator: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.numerator, int)
            or isinstance(self.numerator, bool)
            or not isinstance(self.denominator, int)
            or isinstance(self.denominator, bool)
            or self.denominator == 0
        ):
            raise TypeError("ExactRational requires integer numerator/denominator")
        normalized = Fraction(self.numerator, self.denominator)
        object.__setattr__(self, "numerator", normalized.numerator)
        object.__setattr__(self, "denominator", normalized.denominator)

    @classmethod
    def from_value(
        cls, value: "ExactRational | Fraction | int"
    ) -> "ExactRational":
        if isinstance(value, cls):
            return value
        if isinstance(value, Fraction):
            return cls(value.numerator, value.denominator)
        if isinstance(value, int) and not isinstance(value, bool):
            return cls(value, 1)
        raise TypeError("exact values must be int, Fraction, or ExactRational")

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def safe_payload(self) -> dict[str, int]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
        }


@dataclass(frozen=True)
class InferenceProvenance:
    extractor_id: str
    extractor_version: str
    extractor_implementation_hash: str
    input_evidence_hash: str
    calibration_bucket: str
    alternative_binding_hashes: tuple[str, ...] = ()

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        for value, issue in (
            (self.extractor_id, "inference_extractor_id_invalid"),
            (self.extractor_version, "inference_extractor_version_invalid"),
            (self.calibration_bucket, "inference_calibration_bucket_invalid"),
        ):
            if not _valid_identifier(value):
                issues.append(issue)
        if not _is_sha256(self.extractor_implementation_hash):
            issues.append("inference_extractor_hash_invalid")
        if not _is_sha256(self.input_evidence_hash):
            issues.append("inference_input_evidence_hash_invalid")
        if (
            not isinstance(self.alternative_binding_hashes, tuple)
            or any(
                not _is_sha256(value)
                for value in self.alternative_binding_hashes
            )
            or len(set(self.alternative_binding_hashes))
            != len(self.alternative_binding_hashes)
        ):
            issues.append("inference_alternatives_invalid")
        return tuple(sorted(set(issues)))

    def safe_payload(self) -> dict[str, Any]:
        return {
            "extractor_id": self.extractor_id,
            "extractor_version": self.extractor_version,
            "extractor_implementation_hash": (
                self.extractor_implementation_hash
            ),
            "input_evidence_hash": self.input_evidence_hash,
            "calibration_bucket": self.calibration_bucket,
            "alternative_binding_hashes": list(
                sorted(self.alternative_binding_hashes)
            ),
        }


@dataclass(frozen=True)
class EvidenceSpanRef:
    span_id: str
    source_sha256: str
    start_byte: int
    end_byte: int
    span_sha256: str

    @property
    def span_commitment(self) -> str:
        return strict_content_hash(self.private_payload())

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not _valid_identifier(self.span_id):
            issues.append("evidence_span_id_invalid")
        if not _is_sha256(self.source_sha256):
            issues.append("evidence_span_source_hash_invalid")
        if not _strict_nonnegative_int(self.start_byte):
            issues.append("evidence_span_start_invalid")
        if (
            not _strict_nonnegative_int(self.end_byte)
            or (
                _strict_nonnegative_int(self.start_byte)
                and self.end_byte <= self.start_byte
            )
        ):
            issues.append("evidence_span_end_invalid")
        if not _is_sha256(self.span_sha256):
            issues.append("evidence_span_hash_invalid")
        return tuple(sorted(set(issues)))

    def verify_against(self, source_bytes: bytes) -> tuple[str, ...]:
        issues = list(self.validate())
        if not isinstance(source_bytes, bytes):
            return tuple(sorted(set((*issues, "evidence_source_bytes_invalid"))))
        if hashlib.sha256(source_bytes).hexdigest() != self.source_sha256:
            issues.append("evidence_source_digest_mismatch")
        if (
            _strict_nonnegative_int(self.start_byte)
            and _strict_nonnegative_int(self.end_byte)
            and self.end_byte <= len(source_bytes)
            and hashlib.sha256(
                source_bytes[self.start_byte : self.end_byte]
            ).hexdigest()
            != self.span_sha256
        ):
            issues.append("evidence_span_digest_mismatch")
        if (
            _strict_nonnegative_int(self.end_byte)
            and self.end_byte > len(source_bytes)
        ):
            issues.append("evidence_span_out_of_bounds")
        return tuple(sorted(set(issues)))

    def private_payload(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "source_sha256": self.source_sha256,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "span_sha256": self.span_sha256,
        }

    def safe_payload(self) -> dict[str, Any]:
        return {
            "source_sha256": self.source_sha256,
            "span_commitment": self.span_commitment,
        }


def _evidence_input_hash(
    span_ids: tuple[str, ...],
    spans_by_id: Mapping[str, EvidenceSpanRef],
) -> str | None:
    if any(span_id not in spans_by_id for span_id in span_ids):
        return None
    return strict_content_hash(
        [
            spans_by_id[span_id].private_payload()
            for span_id in sorted(span_ids)
        ]
    )


def _validate_observation(
    *,
    status: object,
    evidence_span_ids: object,
    inference_provenance: object,
    spans_by_id: Mapping[str, EvidenceSpanRef] | None = None,
) -> tuple[str, ...]:
    issues: list[str] = []
    if not isinstance(status, ObservationStatus):
        return ("observation_status_invalid",)
    if not _unique_strings(
        evidence_span_ids, allow_empty=(status is ObservationStatus.UNKNOWN)
    ):
        issues.append("observation_evidence_invalid")
    if status is ObservationStatus.OBSERVED:
        if inference_provenance is not None:
            issues.append("observed_field_has_inference_provenance")
    elif status is ObservationStatus.INFERRED:
        if not isinstance(inference_provenance, InferenceProvenance):
            issues.append("inferred_field_provenance_missing")
        else:
            issues.extend(inference_provenance.validate())
            if (
                spans_by_id is not None
                and isinstance(evidence_span_ids, tuple)
            ):
                expected = _evidence_input_hash(
                    evidence_span_ids, spans_by_id
                )
                if (
                    expected is not None
                    and inference_provenance.input_evidence_hash != expected
                ):
                    issues.append("inferred_field_input_hash_mismatch")
    else:
        if evidence_span_ids:
            issues.append("unknown_field_has_evidence")
        if inference_provenance is not None:
            issues.append("unknown_field_has_inference_provenance")
    return tuple(sorted(set(issues)))


@dataclass(frozen=True)
class StructuralObject:
    object_id: str
    object_type: str
    evidence_span_ids: tuple[str, ...]
    observation_status: ObservationStatus = ObservationStatus.OBSERVED
    inference_provenance: InferenceProvenance | None = None

    def validate(
        self,
        spans_by_id: Mapping[str, EvidenceSpanRef] | None = None,
    ) -> tuple[str, ...]:
        issues = list(
            _validate_observation(
                status=self.observation_status,
                evidence_span_ids=self.evidence_span_ids,
                inference_provenance=self.inference_provenance,
                spans_by_id=spans_by_id,
            )
        )
        if not _valid_identifier(self.object_id):
            issues.append("structural_object_id_invalid")
        if not _valid_symbol(self.object_type):
            issues.append("structural_object_type_invalid")
        return tuple(sorted(set(issues)))

    def private_payload(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "object_type": self.object_type,
            "evidence_span_ids": list(sorted(self.evidence_span_ids)),
            "observation_status": (
                self.observation_status.value
                if isinstance(self.observation_status, ObservationStatus)
                else None
            ),
            "inference_provenance": (
                None
                if self.inference_provenance is None
                else self.inference_provenance.safe_payload()
            ),
        }


@dataclass(frozen=True)
class StructuralRelation:
    relation_id: str
    relation_type: str
    source_object_id: str
    target_object_id: str
    evidence_span_ids: tuple[str, ...]
    observation_status: ObservationStatus = ObservationStatus.OBSERVED
    inference_provenance: InferenceProvenance | None = None
    order_index: int | None = None

    def validate(
        self,
        spans_by_id: Mapping[str, EvidenceSpanRef] | None = None,
    ) -> tuple[str, ...]:
        issues = list(
            _validate_observation(
                status=self.observation_status,
                evidence_span_ids=self.evidence_span_ids,
                inference_provenance=self.inference_provenance,
                spans_by_id=spans_by_id,
            )
        )
        for value, issue in (
            (self.relation_id, "structural_relation_id_invalid"),
            (self.source_object_id, "structural_relation_source_invalid"),
            (self.target_object_id, "structural_relation_target_invalid"),
        ):
            if not _valid_identifier(value):
                issues.append(issue)
        if not _valid_symbol(self.relation_type):
            issues.append("structural_relation_type_invalid")
        if self.order_index is not None and not _strict_nonnegative_int(
            self.order_index
        ):
            issues.append("structural_relation_order_invalid")
        return tuple(sorted(set(issues)))

    def private_payload(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "relation_type": self.relation_type,
            "source_object_id": self.source_object_id,
            "target_object_id": self.target_object_id,
            "evidence_span_ids": list(sorted(self.evidence_span_ids)),
            "observation_status": (
                self.observation_status.value
                if isinstance(self.observation_status, ObservationStatus)
                else None
            ),
            "inference_provenance": (
                None
                if self.inference_provenance is None
                else self.inference_provenance.safe_payload()
            ),
            "order_index": self.order_index,
        }


@dataclass(frozen=True)
class StructuralQuantity:
    quantity_id: str
    owner_object_id: str
    dimension: str
    unit: str
    value: ExactRational | None
    evidence_span_ids: tuple[str, ...]
    observation_status: ObservationStatus = ObservationStatus.OBSERVED
    inference_provenance: InferenceProvenance | None = None

    def validate(
        self,
        spans_by_id: Mapping[str, EvidenceSpanRef] | None = None,
    ) -> tuple[str, ...]:
        issues = list(
            _validate_observation(
                status=self.observation_status,
                evidence_span_ids=self.evidence_span_ids,
                inference_provenance=self.inference_provenance,
                spans_by_id=spans_by_id,
            )
        )
        if not _valid_identifier(self.quantity_id):
            issues.append("structural_quantity_id_invalid")
        if not _valid_identifier(self.owner_object_id):
            issues.append("structural_quantity_owner_invalid")
        if not _valid_symbol(self.dimension):
            issues.append("structural_quantity_dimension_invalid")
        if not isinstance(self.unit, str) or _UNIT.fullmatch(self.unit) is None:
            issues.append("structural_quantity_unit_invalid")
        if self.observation_status is ObservationStatus.UNKNOWN:
            if self.value is not None:
                issues.append("unknown_structural_quantity_has_value")
        elif not isinstance(self.value, ExactRational):
            issues.append("structural_quantity_value_missing")
        return tuple(sorted(set(issues)))

    def private_payload(self) -> dict[str, Any]:
        return {
            "quantity_id": self.quantity_id,
            "owner_object_id": self.owner_object_id,
            "dimension": self.dimension,
            "unit": self.unit,
            "value": (
                None
                if not isinstance(self.value, ExactRational)
                else self.value.safe_payload()
            ),
            "evidence_span_ids": list(sorted(self.evidence_span_ids)),
            "observation_status": (
                self.observation_status.value
                if isinstance(self.observation_status, ObservationStatus)
                else None
            ),
            "inference_provenance": (
                None
                if self.inference_provenance is None
                else self.inference_provenance.safe_payload()
            ),
        }


@dataclass(frozen=True)
class HyperRoleEndpoint:
    endpoint_role: str
    object_id: str

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not _valid_identifier(self.endpoint_role):
            issues.append("hyper_endpoint_role_invalid")
        if not _valid_identifier(self.object_id):
            issues.append("hyper_endpoint_object_invalid")
        return tuple(sorted(set(issues)))

    def private_payload(self) -> dict[str, str]:
        return {
            "endpoint_role": self.endpoint_role,
            "object_id": self.object_id,
        }


@dataclass(frozen=True)
class StructuralHyperrelation:
    hyperrelation_id: str
    hyperrelation_type: str
    endpoints: tuple[HyperRoleEndpoint, ...]
    evidence_span_ids: tuple[str, ...]
    observation_status: ObservationStatus = ObservationStatus.OBSERVED
    inference_provenance: InferenceProvenance | None = None

    def validate(
        self,
        spans_by_id: Mapping[str, EvidenceSpanRef] | None = None,
    ) -> tuple[str, ...]:
        issues = list(
            _validate_observation(
                status=self.observation_status,
                evidence_span_ids=self.evidence_span_ids,
                inference_provenance=self.inference_provenance,
                spans_by_id=spans_by_id,
            )
        )
        if not _valid_identifier(self.hyperrelation_id):
            issues.append("structural_hyperrelation_id_invalid")
        if not _valid_symbol(self.hyperrelation_type):
            issues.append("structural_hyperrelation_type_invalid")
        if (
            not isinstance(self.endpoints, tuple)
            or len(self.endpoints) < 2
        ):
            issues.append("structural_hyperrelation_arity_invalid")
        else:
            endpoint_roles = tuple(
                endpoint.endpoint_role for endpoint in self.endpoints
            )
            if len(endpoint_roles) != len(set(endpoint_roles)):
                issues.append("structural_hyperrelation_roles_duplicate")
            issues.extend(
                issue
                for endpoint in self.endpoints
                for issue in endpoint.validate()
            )
        return tuple(sorted(set(issues)))

    def private_payload(self) -> dict[str, Any]:
        return {
            "hyperrelation_id": self.hyperrelation_id,
            "hyperrelation_type": self.hyperrelation_type,
            "endpoints": [
                endpoint.private_payload()
                for endpoint in sorted(
                    self.endpoints, key=lambda row: row.endpoint_role
                )
            ],
            "evidence_span_ids": list(sorted(self.evidence_span_ids)),
            "observation_status": (
                self.observation_status.value
                if isinstance(self.observation_status, ObservationStatus)
                else None
            ),
            "inference_provenance": (
                None
                if self.inference_provenance is None
                else self.inference_provenance.safe_payload()
            ),
        }


@dataclass(frozen=True)
class ConstraintParticipant:
    participant_role: str
    target_kind: RoleTargetKind
    target_id: str

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not _valid_identifier(self.participant_role):
            issues.append("constraint_participant_role_invalid")
        if not isinstance(self.target_kind, RoleTargetKind):
            issues.append("constraint_participant_kind_invalid")
        if not _valid_identifier(self.target_id):
            issues.append("constraint_participant_target_invalid")
        return tuple(sorted(set(issues)))

    def private_payload(self) -> dict[str, str | None]:
        return {
            "participant_role": self.participant_role,
            "target_kind": (
                self.target_kind.value
                if isinstance(self.target_kind, RoleTargetKind)
                else None
            ),
            "target_id": self.target_id,
        }


@dataclass(frozen=True)
class StructuralConstraint:
    constraint_id: str
    constraint_type: str
    participants: tuple[ConstraintParticipant, ...]
    observable_ids: tuple[str, ...]
    evidence_span_ids: tuple[str, ...]
    observation_status: ObservationStatus = ObservationStatus.OBSERVED
    inference_provenance: InferenceProvenance | None = None

    def validate(
        self,
        spans_by_id: Mapping[str, EvidenceSpanRef] | None = None,
    ) -> tuple[str, ...]:
        issues = list(
            _validate_observation(
                status=self.observation_status,
                evidence_span_ids=self.evidence_span_ids,
                inference_provenance=self.inference_provenance,
                spans_by_id=spans_by_id,
            )
        )
        if not _valid_identifier(self.constraint_id):
            issues.append("structural_constraint_id_invalid")
        if not _valid_symbol(self.constraint_type):
            issues.append("structural_constraint_type_invalid")
        if not isinstance(self.participants, tuple):
            issues.append("structural_constraint_participants_invalid")
        else:
            roles = tuple(
                participant.participant_role
                for participant in self.participants
            )
            if len(roles) != len(set(roles)):
                issues.append("structural_constraint_roles_duplicate")
            issues.extend(
                issue
                for participant in self.participants
                for issue in participant.validate()
            )
        if not _unique_strings(self.observable_ids):
            issues.append("structural_constraint_observables_invalid")
        return tuple(sorted(set(issues)))

    def private_payload(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "constraint_type": self.constraint_type,
            "participants": [
                participant.private_payload()
                for participant in sorted(
                    self.participants,
                    key=lambda row: row.participant_role,
                )
            ],
            "observable_ids": list(sorted(self.observable_ids)),
            "evidence_span_ids": list(sorted(self.evidence_span_ids)),
            "observation_status": (
                self.observation_status.value
                if isinstance(self.observation_status, ObservationStatus)
                else None
            ),
            "inference_provenance": (
                None
                if self.inference_provenance is None
                else self.inference_provenance.safe_payload()
            ),
        }


@dataclass(frozen=True)
class TypedObservable:
    observable_id: str
    value_type: ObservableValueType
    value_payload: Any
    evidence_span_ids: tuple[str, ...]
    observation_status: ObservationStatus = ObservationStatus.OBSERVED
    inference_provenance: InferenceProvenance | None = None
    dimension: str | None = None
    unit: str | None = None

    @property
    def observable_hash(self) -> str:
        return strict_content_hash(self.private_payload())

    def validate(
        self,
        spans_by_id: Mapping[str, EvidenceSpanRef] | None = None,
    ) -> tuple[str, ...]:
        issues = list(
            _validate_observation(
                status=self.observation_status,
                evidence_span_ids=self.evidence_span_ids,
                inference_provenance=self.inference_provenance,
                spans_by_id=spans_by_id,
            )
        )
        if not _valid_identifier(self.observable_id):
            issues.append("typed_observable_id_invalid")
        if not isinstance(self.value_type, ObservableValueType):
            issues.append("typed_observable_value_type_invalid")
        if self.observation_status is ObservationStatus.UNKNOWN:
            if self.value_payload is not None:
                issues.append("unknown_observable_has_value")
        else:
            try:
                strict_canonical_bytes(self.value_payload)
            except TypeError:
                issues.append("typed_observable_payload_not_strict_json")
        if (self.dimension is None) != (self.unit is None):
            issues.append("typed_observable_unit_pair_incomplete")
        if self.dimension is not None and not _valid_symbol(self.dimension):
            issues.append("typed_observable_dimension_invalid")
        if (
            self.unit is not None
            and (
                not isinstance(self.unit, str)
                or _UNIT.fullmatch(self.unit) is None
            )
        ):
            issues.append("typed_observable_unit_invalid")
        return tuple(sorted(set(issues)))

    def private_payload(self) -> dict[str, Any]:
        return {
            "observable_id": self.observable_id,
            "value_type": (
                self.value_type.value
                if isinstance(self.value_type, ObservableValueType)
                else None
            ),
            "value_payload": self.value_payload,
            "evidence_span_ids": list(sorted(self.evidence_span_ids)),
            "observation_status": (
                self.observation_status.value
                if isinstance(self.observation_status, ObservationStatus)
                else None
            ),
            "inference_provenance": (
                None
                if self.inference_provenance is None
                else self.inference_provenance.safe_payload()
            ),
            "dimension": self.dimension,
            "unit": self.unit,
        }

    def safe_payload(self) -> dict[str, Any]:
        return {
            "observable_id": self.observable_id,
            "observable_hash": self.observable_hash,
            "value_type": self.value_type.value,
            "observation_status": self.observation_status.value,
            "dimension": self.dimension,
            "unit": self.unit,
            "evidence_span_count": len(self.evidence_span_ids),
            "evidence_commitment": strict_content_hash(
                list(sorted(self.evidence_span_ids))
            ),
        }


StructuralTarget = (
    StructuralObject
    | StructuralRelation
    | StructuralQuantity
    | StructuralHyperrelation
    | StructuralConstraint
)


@dataclass(frozen=True)
class StructuralEpisode:
    episode_id: str
    source_sha256: str
    evidence_spans: tuple[EvidenceSpanRef, ...]
    objects: tuple[StructuralObject, ...]
    relations: tuple[StructuralRelation, ...]
    quantities: tuple[StructuralQuantity, ...]
    hyperrelations: tuple[StructuralHyperrelation, ...]
    constraints: tuple[StructuralConstraint, ...]
    observables: tuple[TypedObservable, ...]
    declared_boundary_object_id: str | None = None
    missing_observables: tuple[str, ...] = ()

    @property
    def episode_hash(self) -> str:
        return strict_content_hash(self.private_payload())

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not _valid_identifier(self.episode_id):
            issues.append("structural_episode_id_invalid")
        if not _is_sha256(self.source_sha256):
            issues.append("structural_episode_source_hash_invalid")
        typed_collections = (
            (
                self.evidence_spans,
                EvidenceSpanRef,
                "structural_episode_evidence_invalid",
            ),
            (
                self.objects,
                StructuralObject,
                "structural_episode_objects_invalid",
            ),
            (
                self.relations,
                StructuralRelation,
                "structural_episode_relations_invalid",
            ),
            (
                self.quantities,
                StructuralQuantity,
                "structural_episode_quantities_invalid",
            ),
            (
                self.hyperrelations,
                StructuralHyperrelation,
                "structural_episode_hyperrelations_invalid",
            ),
            (
                self.constraints,
                StructuralConstraint,
                "structural_episode_constraints_invalid",
            ),
            (
                self.observables,
                TypedObservable,
                "structural_episode_observables_invalid",
            ),
        )
        collection_type_invalid = False
        for collection, expected_type, issue in typed_collections:
            if (
                not isinstance(collection, tuple)
                or any(
                    not isinstance(item, expected_type)
                    for item in collection
                )
            ):
                issues.append(issue)
                collection_type_invalid = True
        if collection_type_invalid:
            issues.append("structural_episode_collection_type_invalid")
            return tuple(sorted(set(issues)))
        if not self.evidence_spans:
            issues.append("structural_episode_evidence_missing")
        if not self.objects:
            issues.append("structural_episode_objects_missing")

        span_ids = tuple(span.span_id for span in self.evidence_spans)
        spans_by_id = {span.span_id: span for span in self.evidence_spans}
        object_ids = tuple(item.object_id for item in self.objects)
        relation_ids = tuple(item.relation_id for item in self.relations)
        quantity_ids = tuple(item.quantity_id for item in self.quantities)
        hyperrelation_ids = tuple(
            item.hyperrelation_id for item in self.hyperrelations
        )
        constraint_ids = tuple(
            item.constraint_id for item in self.constraints
        )
        observable_ids = tuple(
            item.observable_id for item in self.observables
        )
        for values, issue in (
            (span_ids, "structural_episode_span_ids_duplicate"),
            (object_ids, "structural_episode_object_ids_duplicate"),
            (relation_ids, "structural_episode_relation_ids_duplicate"),
            (quantity_ids, "structural_episode_quantity_ids_duplicate"),
            (
                hyperrelation_ids,
                "structural_episode_hyperrelation_ids_duplicate",
            ),
            (
                constraint_ids,
                "structural_episode_constraint_ids_duplicate",
            ),
            (
                observable_ids,
                "structural_episode_observable_ids_duplicate",
            ),
        ):
            if len(values) != len(set(values)):
                issues.append(issue)

        issues.extend(
            issue for span in self.evidence_spans for issue in span.validate()
        )
        issues.extend(
            issue
            for item in self.objects
            for issue in item.validate(spans_by_id)
        )
        issues.extend(
            issue
            for item in self.relations
            for issue in item.validate(spans_by_id)
        )
        issues.extend(
            issue
            for item in self.quantities
            for issue in item.validate(spans_by_id)
        )
        issues.extend(
            issue
            for item in self.hyperrelations
            for issue in item.validate(spans_by_id)
        )
        issues.extend(
            issue
            for item in self.constraints
            for issue in item.validate(spans_by_id)
        )
        issues.extend(
            issue
            for item in self.observables
            for issue in item.validate(spans_by_id)
        )

        span_set = frozenset(span_ids)
        object_set = frozenset(object_ids)
        observable_set = frozenset(observable_ids)
        if any(
            span.source_sha256 != self.source_sha256
            for span in self.evidence_spans
        ):
            issues.append("structural_episode_mixed_source_hash")
        observed_items: tuple[Any, ...] = (
            *self.objects,
            *self.relations,
            *self.quantities,
            *self.hyperrelations,
            *self.constraints,
            *self.observables,
        )
        if any(
            span_id not in span_set
            for item in observed_items
            for span_id in (
                item.evidence_span_ids
                if isinstance(item.evidence_span_ids, tuple)
                else ()
            )
        ):
            issues.append("structural_episode_unknown_span_reference")
        if any(
            relation.source_object_id not in object_set
            or relation.target_object_id not in object_set
            for relation in self.relations
        ):
            issues.append("structural_episode_relation_object_unknown")
        if any(
            quantity.owner_object_id not in object_set
            for quantity in self.quantities
        ):
            issues.append("structural_episode_quantity_owner_unknown")
        if any(
            endpoint.object_id not in object_set
            for hyperrelation in self.hyperrelations
            for endpoint in hyperrelation.endpoints
        ):
            issues.append("structural_episode_hyperrelation_object_unknown")
        target_sets: Mapping[RoleTargetKind, frozenset[str]] = {
            RoleTargetKind.OBJECT: object_set,
            RoleTargetKind.RELATION: frozenset(relation_ids),
            RoleTargetKind.QUANTITY: frozenset(quantity_ids),
            RoleTargetKind.HYPERRELATION: frozenset(hyperrelation_ids),
            RoleTargetKind.CONSTRAINT: frozenset(constraint_ids),
        }
        for constraint in self.constraints:
            if any(
                not isinstance(participant.target_kind, RoleTargetKind)
                or participant.target_id
                not in target_sets.get(
                    participant.target_kind, frozenset()
                )
                for participant in constraint.participants
            ):
                issues.append(
                    "structural_episode_constraint_target_unknown"
                )
            if any(
                observable_id not in observable_set
                for observable_id in constraint.observable_ids
            ):
                issues.append(
                    "structural_episode_constraint_observable_unknown"
                )
        if (
            self.declared_boundary_object_id is not None
            and self.declared_boundary_object_id not in object_set
        ):
            issues.append("structural_episode_boundary_unknown")
        if not _unique_strings(
            self.missing_observables, allow_empty=True
        ):
            issues.append("structural_episode_missing_observables_invalid")
        unknown_observable_ids = {
            observable.observable_id
            for observable in self.observables
            if observable.observation_status is ObservationStatus.UNKNOWN
        }
        if set(self.missing_observables) != unknown_observable_ids:
            issues.append(
                "structural_episode_missing_observables_mismatch"
            )
        try:
            strict_canonical_bytes(self.private_payload())
        except (AttributeError, TypeError):
            issues.append("structural_episode_payload_not_strict_json")
        return tuple(sorted(set(issues)))

    def verify_source_bytes(self, source_bytes: bytes) -> tuple[str, ...]:
        issues = list(self.validate())
        if not isinstance(source_bytes, bytes):
            issues.append("structural_episode_source_bytes_invalid")
            return tuple(sorted(set(issues)))
        if hashlib.sha256(source_bytes).hexdigest() != self.source_sha256:
            issues.append("structural_episode_source_digest_mismatch")
        issues.extend(
            issue
            for span in self.evidence_spans
            for issue in span.verify_against(source_bytes)
        )
        return tuple(sorted(set(issues)))

    def private_payload(self) -> dict[str, Any]:
        return {
            "schema_version": GSCL_SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "source_sha256": self.source_sha256,
            "evidence_spans": [
                span.private_payload()
                for span in sorted(
                    self.evidence_spans, key=lambda row: row.span_id
                )
            ],
            "objects": [
                item.private_payload()
                for item in sorted(
                    self.objects, key=lambda row: row.object_id
                )
            ],
            "relations": [
                item.private_payload()
                for item in sorted(
                    self.relations, key=lambda row: row.relation_id
                )
            ],
            "quantities": [
                item.private_payload()
                for item in sorted(
                    self.quantities, key=lambda row: row.quantity_id
                )
            ],
            "hyperrelations": [
                item.private_payload()
                for item in sorted(
                    self.hyperrelations,
                    key=lambda row: row.hyperrelation_id,
                )
            ],
            "constraints": [
                item.private_payload()
                for item in sorted(
                    self.constraints, key=lambda row: row.constraint_id
                )
            ],
            "observables": [
                item.private_payload()
                for item in sorted(
                    self.observables, key=lambda row: row.observable_id
                )
            ],
            "declared_boundary_object_id": (
                self.declared_boundary_object_id
            ),
            "missing_observables": list(
                sorted(self.missing_observables)
            ),
        }

    def safe_payload(self) -> dict[str, Any]:
        return {
            "schema_version": GSCL_SCHEMA_VERSION,
            "episode_hash": self.episode_hash,
            "source_sha256": self.source_sha256,
            "evidence_span_count": len(self.evidence_spans),
            "object_count": len(self.objects),
            "relation_count": len(self.relations),
            "quantity_count": len(self.quantities),
            "hyperrelation_count": len(self.hyperrelations),
            "constraint_count": len(self.constraints),
            "observable_count": len(self.observables),
            "boundary_declared": self.declared_boundary_object_id is not None,
            "missing_observable_count": len(self.missing_observables),
            "missing_observable_commitment": strict_content_hash(
                list(sorted(self.missing_observables))
            ),
        }

    def require_target(
        self, kind: RoleTargetKind, target_id: str
    ) -> StructuralTarget:
        collections: Mapping[
            RoleTargetKind, Sequence[StructuralTarget]
        ] = {
            RoleTargetKind.OBJECT: self.objects,
            RoleTargetKind.RELATION: self.relations,
            RoleTargetKind.QUANTITY: self.quantities,
            RoleTargetKind.HYPERRELATION: self.hyperrelations,
            RoleTargetKind.CONSTRAINT: self.constraints,
        }
        attributes = {
            RoleTargetKind.OBJECT: "object_id",
            RoleTargetKind.RELATION: "relation_id",
            RoleTargetKind.QUANTITY: "quantity_id",
            RoleTargetKind.HYPERRELATION: "hyperrelation_id",
            RoleTargetKind.CONSTRAINT: "constraint_id",
        }
        if not isinstance(kind, RoleTargetKind):
            raise KeyError(f"unknown structural target kind: {kind!r}")
        matches = [
            item
            for item in collections[kind]
            if getattr(item, attributes[kind]) == target_id
        ]
        if len(matches) != 1:
            raise KeyError(f"unknown {kind.value} target: {target_id}")
        return matches[0]

    def require_observable(self, observable_id: str) -> TypedObservable:
        matches = [
            observable
            for observable in self.observables
            if observable.observable_id == observable_id
        ]
        if len(matches) != 1:
            raise KeyError(f"unknown observable: {observable_id}")
        return matches[0]


_HARD_NEGATIVE_OPERATOR_CONTRACTS = {
    "output_sign_flip": {
        "law_kind": "equivariance",
        "operator": "negate_first_output_action_sign",
    },
    "role_swap_input_output": {
        "law_kind": "equivariance",
        "operator": "replace_after_vector_with_before_vector",
    },
    "direction_flip": {
        "law_kind": "monotone_order",
        "operator": "negate_declared_direction",
    },
    "lower_upper_role_swap": {
        "law_kind": "monotone_order",
        "operator": "swap_every_comparable_pair",
    },
    "delete_boundary_flow": {
        "law_kind": "closed_balance",
        "operator": "delete_first_inflow_else_first_source",
    },
    "flow_sign_flip": {
        "law_kind": "closed_balance",
        "operator": "swap_inflow_outflow_and_source_sink_ledgers",
    },
    "intermediate_map_substitution": {
        "law_kind": "path_composition",
        "operator": "replace_first_nonanchor_second_map_target",
    },
    "path_order_reversal": {
        "law_kind": "path_composition",
        "operator": "swap_first_map_targets_for_first_two_domain_rows",
    },
    "interaction_sign_flip": {
        "law_kind": "low_order_interaction",
        "operator": "negate_designated_pair_mobius_coefficient",
    },
    "unmodeled_third_order_term": {
        "law_kind": "low_order_interaction",
        "operator": "add_unit_full_set_mobius_coefficient",
    },
}
HARD_NEGATIVE_OPERATOR_CONTRACT_HASHES = {
    transformation_id: strict_content_hash(
        {
            "version": "gscl_hard_negative_operator_v1",
            "transformation_id": transformation_id,
            **contract,
        }
    )
    for transformation_id, contract in (
        _HARD_NEGATIVE_OPERATOR_CONTRACTS.items()
    )
}


_RESIDUAL_KERNEL_CONTRACT = {
    "version": GSCL_RESIDUAL_KERNEL_VERSION,
    "numeric_domain": "exact_rational",
    "reference_namespaces": {
        "bound_role": "role:<law_role_id>",
        "observable_local": "local:<bundle_scoped_token>",
    },
    "binding_semantics": {
        "constraint_bridge": "law_specific_exact_role_and_observable_coverage",
        "equivariance": (
            "phase0_single_coordinate_perm0_sign_pm1_and_quantity"
        ),
        "monotone_order": (
            "phase0_single_bound_pair_and_direction_minus1_zero_plus1"
        ),
        "closed_balance": (
            "declared_boundary_and_storage_quantities_bound_to_ledger"
        ),
        "path_composition": (
            "role_anchored_endpoints_with_bundle_local_intermediates"
        ),
        "low_order_interaction": (
            "three_unique_bound_components_pair_and_complete_subset_lattice"
        ),
    },
    "receipt_construction": (
        "trusted_internal_recomputation_of_primary_and_contrastives"
    ),
    "hard_negative_operators": {
        transformation_id: {
            **contract,
            "operator_contract_hash": (
                HARD_NEGATIVE_OPERATOR_CONTRACT_HASHES[
                    transformation_id
                ]
            ),
        }
        for transformation_id, contract in sorted(
            _HARD_NEGATIVE_OPERATOR_CONTRACTS.items()
        )
    },
    "disposition_rule": (
        "decided iff every nonnegative residual component is within its "
        "policy-bound tolerance"
    ),
    "law_function_ids": {
        "equivariance": "gscl.residual.v1.evaluate_equivariance",
        "monotone_order": "gscl.residual.v1.evaluate_monotone_order",
        "closed_balance": "gscl.residual.v1.evaluate_closed_balance",
        "path_composition": "gscl.residual.v1.evaluate_path_composition",
        "low_order_interaction": (
            "gscl.residual.v1.evaluate_low_order_interaction"
        ),
    },
}
RESIDUAL_KERNEL_CONTRACT_HASH = strict_content_hash(
    _RESIDUAL_KERNEL_CONTRACT
)


@dataclass(frozen=True)
class LawRoleSpec:
    role_id: str
    target_kind: RoleTargetKind
    allowed_target_types: tuple[str, ...]
    required: bool = True

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not _valid_identifier(self.role_id):
            issues.append("law_role_id_invalid")
        if not isinstance(self.target_kind, RoleTargetKind):
            issues.append("law_role_target_kind_invalid")
        if (
            not _unique_strings(self.allowed_target_types)
            or any(
                not _valid_symbol(value)
                for value in self.allowed_target_types
            )
        ):
            issues.append("law_role_target_types_invalid")
        if not isinstance(self.required, bool):
            issues.append("law_role_required_invalid")
        return tuple(sorted(set(issues)))

    def safe_payload(self) -> dict[str, Any]:
        return {
            "role_id": self.role_id,
            "target_kind": self.target_kind.value,
            "allowed_target_types": list(
                sorted(self.allowed_target_types)
            ),
            "required": self.required,
        }


@dataclass(frozen=True)
class ObservableSpec:
    observable_id: str
    value_type: ObservableValueType
    unit_required: bool = False

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not _valid_identifier(self.observable_id):
            issues.append("observable_spec_id_invalid")
        if not isinstance(self.value_type, ObservableValueType):
            issues.append("observable_spec_value_type_invalid")
        if not isinstance(self.unit_required, bool):
            issues.append("observable_spec_unit_required_invalid")
        return tuple(sorted(set(issues)))

    def safe_payload(self) -> dict[str, Any]:
        return {
            "observable_id": self.observable_id,
            "value_type": self.value_type.value,
            "unit_required": self.unit_required,
        }


@dataclass(frozen=True)
class ExecutableLawSchema:
    law_id: str
    ontology_template_id: str
    law_kind: LawKind
    arity: int
    roles: tuple[LawRoleSpec, ...]
    required_observables: tuple[ObservableSpec, ...]
    applicability_preconditions: tuple[str, ...]
    residual_function_id: str
    verifier_version: str
    verifier_contract_hash: str
    expected_component_ids: tuple[str, ...]
    legal_transformations: tuple[str, ...]
    hard_negative_transformations: tuple[str, ...]
    missing_evidence_policy: str
    complexity_bits: int

    @property
    def schema_hash(self) -> str:
        return strict_content_hash(self.safe_payload())

    def validate(
        self, ontology: UniversalAssumptionOntology
    ) -> tuple[str, ...]:
        issues: list[str] = []
        if not _valid_identifier(self.law_id):
            issues.append("executable_law_id_invalid")
        try:
            ontology.require_template(self.ontology_template_id)
        except (KeyError, PermissionError):
            issues.append("executable_law_template_unknown")
        if not isinstance(self.law_kind, LawKind):
            issues.append("executable_law_kind_invalid")
        if (
            not isinstance(self.arity, int)
            or isinstance(self.arity, bool)
            or self.arity <= 0
            or self.arity != len(self.roles)
        ):
            issues.append("executable_law_arity_invalid")
        role_ids = tuple(role.role_id for role in self.roles)
        if not role_ids or len(role_ids) != len(set(role_ids)):
            issues.append("executable_law_roles_invalid")
        issues.extend(
            issue for role in self.roles for issue in role.validate()
        )
        observable_ids = tuple(
            observable.observable_id
            for observable in self.required_observables
        )
        if (
            not observable_ids
            or len(observable_ids) != len(set(observable_ids))
        ):
            issues.append("executable_law_observables_invalid")
        issues.extend(
            issue
            for observable in self.required_observables
            for issue in observable.validate()
        )
        for values, issue in (
            (
                self.applicability_preconditions,
                "executable_law_preconditions_invalid",
            ),
            (
                self.expected_component_ids,
                "executable_law_component_ids_invalid",
            ),
            (
                self.legal_transformations,
                "executable_law_legal_transforms_invalid",
            ),
            (
                self.hard_negative_transformations,
                "executable_law_hard_negatives_invalid",
            ),
        ):
            if not _unique_strings(values):
                issues.append(issue)
        for value, issue in (
            (
                self.residual_function_id,
                "executable_law_residual_function_invalid",
            ),
            (
                self.verifier_version,
                "executable_law_verifier_version_invalid",
            ),
            (
                self.missing_evidence_policy,
                "executable_law_missing_policy_invalid",
            ),
        ):
            if not _valid_identifier(value):
                issues.append(issue)
        if not _is_sha256(self.verifier_contract_hash):
            issues.append("executable_law_verifier_hash_invalid")
        if (
            not isinstance(self.complexity_bits, int)
            or isinstance(self.complexity_bits, bool)
            or self.complexity_bits <= 0
        ):
            issues.append("executable_law_complexity_invalid")
        return tuple(sorted(set(issues)))

    def safe_payload(self) -> dict[str, Any]:
        return {
            "schema_version": GSCL_SCHEMA_VERSION,
            "law_id": self.law_id,
            "ontology_template_id": self.ontology_template_id,
            "law_kind": self.law_kind.value,
            "arity": self.arity,
            "roles": [
                role.safe_payload()
                for role in sorted(
                    self.roles, key=lambda row: row.role_id
                )
            ],
            "required_observables": [
                observable.safe_payload()
                for observable in sorted(
                    self.required_observables,
                    key=lambda row: row.observable_id,
                )
            ],
            "applicability_preconditions": list(
                sorted(self.applicability_preconditions)
            ),
            "residual_function_id": self.residual_function_id,
            "verifier_version": self.verifier_version,
            "verifier_contract_hash": self.verifier_contract_hash,
            "expected_component_ids": list(
                sorted(self.expected_component_ids)
            ),
            "legal_transformations": list(
                sorted(self.legal_transformations)
            ),
            "hard_negative_transformations": list(
                sorted(self.hard_negative_transformations)
            ),
            "missing_evidence_policy": self.missing_evidence_policy,
            "complexity_bits": self.complexity_bits,
        }


def _role(
    role_id: str,
    kind: RoleTargetKind,
    *allowed_types: str,
) -> LawRoleSpec:
    return LawRoleSpec(
        role_id=role_id,
        target_kind=kind,
        allowed_target_types=tuple(allowed_types),
    )


def _observable(
    observable_id: str,
    value_type: ObservableValueType,
    *,
    unit_required: bool = False,
) -> ObservableSpec:
    return ObservableSpec(
        observable_id=observable_id,
        value_type=value_type,
        unit_required=unit_required,
    )


def _build_schema_definitions_v1() -> tuple[ExecutableLawSchema, ...]:
    obj = RoleTargetKind.OBJECT
    rel = RoleTargetKind.RELATION
    qty = RoleTargetKind.QUANTITY
    con = RoleTargetKind.CONSTRAINT
    missing_policy = "gscl.missing.abstain_inconclusive"
    version = GSCL_RESIDUAL_KERNEL_VERSION
    contract_hash = RESIDUAL_KERNEL_CONTRACT_HASH
    return (
        ExecutableLawSchema(
            law_id="gscl.v1.t14_finite_equivariance",
            ontology_template_id=T14,
            law_kind=LawKind.EQUIVARIANCE,
            arity=6,
            roles=(
                _role("input_before", obj, "State"),
                _role("input_after", obj, "State"),
                _role("transformation", rel, "Transformation"),
                _role("output_before", qty, "Quantity"),
                _role("output_after", qty, "Quantity"),
                _role("equivariance_constraint", con, "Equivariance"),
            ),
            required_observables=(
                _observable(
                    "input_action", ObservableValueType.FINITE_MAP
                ),
                _observable(
                    "output_action",
                    ObservableValueType.SIGNED_PERMUTATION,
                ),
                _observable(
                    "outputs_after",
                    ObservableValueType.EXACT_VECTOR,
                    unit_required=True,
                ),
                _observable(
                    "outputs_before",
                    ObservableValueType.EXACT_VECTOR,
                    unit_required=True,
                ),
            ),
            applicability_preconditions=(
                "finite_action_is_declared",
                "output_coordinates_are_comparable",
            ),
            residual_function_id=(
                "gscl.residual.v1.evaluate_equivariance"
            ),
            verifier_version=version,
            verifier_contract_hash=contract_hash,
            expected_component_ids=(
                "equivariance_involution_failure_count",
                "equivariance_max_abs_residual",
            ),
            legal_transformations=(
                "entity_renaming",
                "evidence_serialization_order",
            ),
            hard_negative_transformations=(
                "output_sign_flip",
                "role_swap_input_output",
            ),
            missing_evidence_policy=missing_policy,
            complexity_bits=40,
        ),
        ExecutableLawSchema(
            law_id="gscl.v1.t17_monotone_order",
            ontology_template_id=T17,
            law_kind=LawKind.MONOTONE_ORDER,
            arity=6,
            roles=(
                _role("lower_state", obj, "State"),
                _role("upper_state", obj, "State"),
                _role("order_relation", rel, "PartialOrder"),
                _role("lower_value", qty, "Quantity"),
                _role("upper_value", qty, "Quantity"),
                _role("monotone_constraint", con, "Monotonicity"),
            ),
            required_observables=(
                _observable(
                    "comparable_output_pairs",
                    ObservableValueType.COMPARABLE_PAIRS,
                    unit_required=True,
                ),
                _observable(
                    "declared_direction",
                    ObservableValueType.DIRECTION,
                ),
            ),
            applicability_preconditions=(
                "at_least_one_pair_is_comparable",
                "output_units_match",
            ),
            residual_function_id=(
                "gscl.residual.v1.evaluate_monotone_order"
            ),
            verifier_version=version,
            verifier_contract_hash=contract_hash,
            expected_component_ids=(
                "monotone_max_order_residual",
                "monotone_violation_count",
            ),
            legal_transformations=(
                "entity_renaming",
                "order_preserving_reparameterization",
            ),
            hard_negative_transformations=(
                "direction_flip",
                "lower_upper_role_swap",
            ),
            missing_evidence_policy=missing_policy,
            complexity_bits=36,
        ),
        ExecutableLawSchema(
            law_id="gscl.v1.t15_closed_balance",
            ontology_template_id=T15,
            law_kind=LawKind.CLOSED_BALANCE,
            arity=5,
            roles=(
                _role("system_boundary", obj, "SystemBoundary"),
                _role("storage_before", qty, "Quantity"),
                _role("storage_after", qty, "Quantity"),
                _role("flow_ledger", obj, "FlowLedger"),
                _role("balance_constraint", con, "Balance"),
            ),
            required_observables=(
                _observable(
                    "boundary_declaration",
                    ObservableValueType.BOUNDARY_DECLARATION,
                ),
                _observable(
                    "quantity_ledger",
                    ObservableValueType.QUANTITY_LEDGER,
                    unit_required=True,
                ),
            ),
            applicability_preconditions=(
                "system_boundary_is_explicit",
                "unobserved_boundary_flow_is_absent",
            ),
            residual_function_id=(
                "gscl.residual.v1.evaluate_closed_balance"
            ),
            verifier_version=version,
            verifier_contract_hash=contract_hash,
            expected_component_ids=("closed_balance_abs_residual",),
            legal_transformations=(
                "entity_renaming",
                "accounting_partition_refinement",
            ),
            hard_negative_transformations=(
                "delete_boundary_flow",
                "flow_sign_flip",
            ),
            missing_evidence_policy=missing_policy,
            complexity_bits=44,
        ),
        ExecutableLawSchema(
            law_id="gscl.v1.t09_path_composition",
            ontology_template_id=T09,
            law_kind=LawKind.PATH_COMPOSITION,
            arity=5,
            roles=(
                _role("source_state", obj, "State"),
                _role("target_state", obj, "State"),
                _role("composed_path", obj, "TypedPath"),
                _role("direct_path", obj, "TypedPath"),
                _role("path_constraint", con, "PathEquality"),
            ),
            required_observables=(
                _observable(
                    "direct_map", ObservableValueType.FINITE_MAP
                ),
                _observable(
                    "finite_domain", ObservableValueType.FINITE_DOMAIN
                ),
                _observable(
                    "first_map", ObservableValueType.FINITE_MAP
                ),
                _observable(
                    "second_map", ObservableValueType.FINITE_MAP
                ),
            ),
            applicability_preconditions=(
                "map_evaluation_is_deterministic",
                "path_sources_and_targets_match",
            ),
            residual_function_id=(
                "gscl.residual.v1.evaluate_path_composition"
            ),
            verifier_version=version,
            verifier_contract_hash=contract_hash,
            expected_component_ids=(
                "path_composition_mismatch_rate",
            ),
            legal_transformations=(
                "entity_renaming",
                "finite_domain_permutation",
            ),
            hard_negative_transformations=(
                "intermediate_map_substitution",
                "path_order_reversal",
            ),
            missing_evidence_policy=missing_policy,
            complexity_bits=42,
        ),
        ExecutableLawSchema(
            law_id="gscl.v1.t05_pair_interaction",
            ontology_template_id=T05,
            law_kind=LawKind.LOW_ORDER_INTERACTION,
            arity=5,
            roles=(
                _role("component_a", obj, "Component"),
                _role("component_b", obj, "Component"),
                _role("component_c", obj, "Component"),
                _role("utility_ledger", obj, "UtilityLedger"),
                _role("interaction_constraint", con, "Interaction"),
            ),
            required_observables=(
                _observable(
                    "components", ObservableValueType.COMPONENT_SET
                ),
                _observable(
                    "designated_pair",
                    ObservableValueType.DESIGNATED_PAIR,
                ),
                _observable(
                    "held_fold_utilities",
                    ObservableValueType.SUBSET_UTILITY_FOLDS,
                    unit_required=True,
                ),
                _observable(
                    "interaction_expectation",
                    ObservableValueType.INTERACTION_EXPECTATION,
                ),
            ),
            applicability_preconditions=(
                "at_least_two_components",
                "common_utility_scale",
            ),
            residual_function_id=(
                "gscl.residual.v1.evaluate_low_order_interaction"
            ),
            verifier_version=version,
            verifier_contract_hash=contract_hash,
            expected_component_ids=(
                "interaction_high_order_excess",
                "interaction_pair_relation_residual",
            ),
            legal_transformations=(
                "component_order_permutation",
                "component_renaming",
            ),
            hard_negative_transformations=(
                "interaction_sign_flip",
                "unmodeled_third_order_term",
            ),
            missing_evidence_policy=missing_policy,
            complexity_bits=46,
        ),
    )


@dataclass(frozen=True)
class GSCLSchemaRegistry:
    ontology_hash: str
    schemas: tuple[ExecutableLawSchema, ...]

    @property
    def registry_hash(self) -> str:
        return strict_content_hash(self.safe_payload())

    def validate_frozen_contract(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.ontology_hash != FROZEN_UAO_V1_ONTOLOGY_HASH:
            issues.append("gscl_registry_frozen_ontology_mismatch")
        if not isinstance(self.schemas, tuple):
            return tuple(
                sorted(
                    set((*issues, "gscl_registry_schemas_invalid"))
                )
            )
        law_ids = tuple(schema.law_id for schema in self.schemas)
        expected = {
            schema.law_id: schema.schema_hash
            for schema in _build_schema_definitions_v1()
        }
        actual = {
            schema.law_id: schema.schema_hash for schema in self.schemas
        }
        if (
            len(law_ids) != 5
            or len(set(law_ids)) != 5
            or set(actual) != set(expected)
        ):
            issues.append("gscl_registry_laws_invalid")
        if actual != expected:
            issues.append("gscl_registry_frozen_contract_mismatch")
        return tuple(sorted(set(issues)))

    def validate(
        self, ontology: UniversalAssumptionOntology
    ) -> tuple[str, ...]:
        issues: list[str] = list(self.validate_frozen_contract())
        if self.ontology_hash != ontology.ontology_hash:
            issues.append("gscl_registry_ontology_mismatch")
        if not isinstance(self.schemas, tuple):
            return tuple(
                sorted(set((*issues, "gscl_registry_schemas_invalid")))
            )
        issues.extend(
            issue
            for schema in self.schemas
            for issue in schema.validate(ontology)
        )
        return tuple(sorted(set(issues)))

    def require_law(self, law_id: str) -> ExecutableLawSchema:
        frozen_issues = self.validate_frozen_contract()
        if frozen_issues:
            raise PermissionError(
                "cannot read a non-frozen GSCL registry: "
                + ",".join(frozen_issues)
            )
        matches = [
            schema for schema in self.schemas if schema.law_id == law_id
        ]
        if len(matches) != 1:
            raise KeyError(f"unknown executable law: {law_id}")
        return matches[0]

    def safe_payload(self) -> dict[str, Any]:
        return {
            "schema_version": GSCL_SCHEMA_VERSION,
            "ontology_hash": self.ontology_hash,
            "schemas": [
                {
                    **schema.safe_payload(),
                    "schema_hash": schema.schema_hash,
                }
                for schema in sorted(
                    self.schemas, key=lambda row: row.law_id
                )
            ],
            "schema_count": len(self.schemas),
        }


def build_gscl_schema_registry_v1(
    ontology: UniversalAssumptionOntology,
) -> GSCLSchemaRegistry:
    if ontology.validate():
        raise PermissionError("cannot bind an invalid UAO ontology")
    registry = GSCLSchemaRegistry(
        ontology_hash=ontology.ontology_hash,
        schemas=_build_schema_definitions_v1(),
    )
    issues = registry.validate(ontology)
    if issues:
        raise PermissionError(
            "invalid GSCL schema registry: " + ",".join(issues)
        )
    return registry


@dataclass(frozen=True)
class RoleBinding:
    role_id: str
    target_id: str
    evidence_span_ids: tuple[str, ...]

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not _valid_identifier(self.role_id):
            issues.append("role_binding_role_id_invalid")
        if not _valid_identifier(self.target_id):
            issues.append("role_binding_target_id_invalid")
        if not _unique_strings(
            self.evidence_span_ids, allow_empty=True
        ):
            issues.append("role_binding_evidence_invalid")
        return tuple(sorted(set(issues)))

    def private_payload(self) -> dict[str, Any]:
        return {
            "role_id": self.role_id,
            "target_id": self.target_id,
            "evidence_span_ids": list(sorted(self.evidence_span_ids)),
        }


@dataclass(frozen=True)
class ObservableBinding:
    observable_id: str
    observable_hash: str

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not _valid_identifier(self.observable_id):
            issues.append("observable_binding_id_invalid")
        if not _is_sha256(self.observable_hash):
            issues.append("observable_binding_hash_invalid")
        return tuple(sorted(set(issues)))

    def safe_payload(self) -> dict[str, str]:
        return {
            "observable_id": self.observable_id,
            "observable_hash": self.observable_hash,
        }


@dataclass(frozen=True)
class LawBinding:
    binding_id: str
    law_id: str
    registry_hash: str
    schema_hash: str
    episode_hash: str
    role_bindings: tuple[RoleBinding, ...]
    observable_bindings: tuple[ObservableBinding, ...]

    @property
    def binding_hash(self) -> str:
        return strict_content_hash(self.private_payload())

    @property
    def evaluation_input_hash(self) -> str:
        return strict_content_hash(
            {
                "law_id": self.law_id,
                "registry_hash": self.registry_hash,
                "schema_hash": self.schema_hash,
                "episode_hash": self.episode_hash,
                "role_bindings": [
                    row.private_payload()
                    for row in sorted(
                        self.role_bindings,
                        key=lambda item: item.role_id,
                    )
                ],
                "observable_bindings": [
                    row.safe_payload()
                    for row in sorted(
                        self.observable_bindings,
                        key=lambda item: item.observable_id,
                    )
                ],
            }
        )

    def private_payload(self) -> dict[str, Any]:
        return {
            "schema_version": GSCL_SCHEMA_VERSION,
            "binding_id": self.binding_id,
            "law_id": self.law_id,
            "registry_hash": self.registry_hash,
            "schema_hash": self.schema_hash,
            "episode_hash": self.episode_hash,
            "role_bindings": [
                row.private_payload()
                for row in sorted(
                    self.role_bindings, key=lambda item: item.role_id
                )
            ],
            "observable_bindings": [
                row.safe_payload()
                for row in sorted(
                    self.observable_bindings,
                    key=lambda item: item.observable_id,
                )
            ],
        }

    def safe_payload(self) -> dict[str, Any]:
        return {
            "schema_version": GSCL_SCHEMA_VERSION,
            "binding_hash": self.binding_hash,
            "law_id": self.law_id,
            "registry_hash": self.registry_hash,
            "schema_hash": self.schema_hash,
            "episode_hash": self.episode_hash,
            "role_count": len(self.role_bindings),
            "observable_count": len(self.observable_bindings),
            "evaluation_input_hash": self.evaluation_input_hash,
        }


def _target_type(target: StructuralTarget) -> str:
    if isinstance(target, StructuralObject):
        return target.object_type
    if isinstance(target, StructuralRelation):
        return target.relation_type
    if isinstance(target, StructuralQuantity):
        return "Quantity"
    if isinstance(target, StructuralHyperrelation):
        return target.hyperrelation_type
    if isinstance(target, StructuralConstraint):
        return target.constraint_type
    raise TypeError(f"unknown structural target {type(target).__name__}")


def _target_status(target: StructuralTarget) -> ObservationStatus:
    return target.observation_status


def validate_law_binding(
    registry: GSCLSchemaRegistry,
    schema: ExecutableLawSchema,
    episode: StructuralEpisode,
    binding: LawBinding,
) -> tuple[str, ...]:
    issues: list[str] = list(registry.validate_frozen_contract())
    issues.extend(episode.validate())
    if not _valid_identifier(binding.binding_id):
        issues.append("law_binding_id_invalid")
    if binding.law_id != schema.law_id:
        issues.append("law_binding_law_mismatch")
    if binding.registry_hash != registry.registry_hash:
        issues.append("law_binding_registry_mismatch")
    if binding.schema_hash != schema.schema_hash:
        issues.append("law_binding_schema_mismatch")
    if binding.episode_hash != episode.episode_hash:
        issues.append("law_binding_episode_mismatch")
    try:
        if registry.require_law(schema.law_id).schema_hash != schema.schema_hash:
            issues.append("law_binding_schema_not_in_registry")
    except (KeyError, PermissionError):
        issues.append("law_binding_schema_not_in_registry")
    if not isinstance(binding.role_bindings, tuple):
        issues.append("law_binding_roles_invalid")
        return tuple(sorted(set(issues)))
    if not isinstance(binding.observable_bindings, tuple):
        issues.append("law_binding_observables_invalid")
        return tuple(sorted(set(issues)))

    role_ids = tuple(item.role_id for item in binding.role_bindings)
    if len(role_ids) != len(set(role_ids)):
        issues.append("law_binding_roles_duplicate")
    issues.extend(
        issue for item in binding.role_bindings for issue in item.validate()
    )
    schema_roles = {item.role_id: item for item in schema.roles}
    if set(role_ids) != set(schema_roles):
        issues.append("law_binding_role_coverage_mismatch")
    target_keys: list[tuple[RoleTargetKind, str]] = []
    episode_span_ids = frozenset(
        item.span_id for item in episode.evidence_spans
    )
    constraint_targets: list[StructuralConstraint] = []
    for item in binding.role_bindings:
        role = schema_roles.get(item.role_id)
        if role is None:
            continue
        try:
            target = episode.require_target(
                role.target_kind, item.target_id
            )
        except KeyError:
            issues.append("law_binding_target_unknown")
            continue
        target_keys.append((role.target_kind, item.target_id))
        if _target_type(target) not in role.allowed_target_types:
            issues.append("law_binding_target_type_invalid")
        target_span_ids = frozenset(target.evidence_span_ids)
        if any(
            span_id not in episode_span_ids
            or span_id not in target_span_ids
            for span_id in item.evidence_span_ids
        ):
            issues.append("law_binding_evidence_not_grounded")
        if _target_status(target) is ObservationStatus.UNKNOWN:
            if item.evidence_span_ids:
                issues.append(
                    "law_binding_unknown_target_has_evidence"
                )
        elif not item.evidence_span_ids:
            issues.append(
                "law_binding_observed_target_evidence_missing"
            )
        if isinstance(target, StructuralConstraint):
            constraint_targets.append(target)
    if len(target_keys) != len(set(target_keys)):
        issues.append("law_binding_target_reused")

    observable_ids = tuple(
        item.observable_id for item in binding.observable_bindings
    )
    if len(observable_ids) != len(set(observable_ids)):
        issues.append("law_binding_observables_duplicate")
    issues.extend(
        issue
        for item in binding.observable_bindings
        for issue in item.validate()
    )
    specs = {
        observable.observable_id: observable
        for observable in schema.required_observables
    }
    if set(observable_ids) != set(specs):
        issues.append("law_binding_observable_coverage_mismatch")
    for observable_binding in binding.observable_bindings:
        spec = specs.get(observable_binding.observable_id)
        if spec is None:
            continue
        try:
            observable = episode.require_observable(
                observable_binding.observable_id
            )
        except KeyError:
            issues.append("law_binding_observable_unknown")
            continue
        if observable.observable_hash != observable_binding.observable_hash:
            issues.append("law_binding_observable_hash_mismatch")
        if observable.value_type is not spec.value_type:
            issues.append("law_binding_observable_type_mismatch")
        if spec.unit_required and (
            observable.dimension is None or observable.unit is None
        ):
            issues.append("law_binding_observable_unit_missing")
    if constraint_targets and any(
        set(constraint.observable_ids) != set(specs)
        for constraint in constraint_targets
    ):
        issues.append("law_binding_constraint_observables_mismatch")
    try:
        strict_canonical_bytes(binding.private_payload())
    except (AttributeError, TypeError):
        issues.append("law_binding_payload_not_strict_json")
    return tuple(sorted(set(issues)))


@dataclass(frozen=True)
class ResidualComponent:
    component_id: str
    value: ExactRational
    tolerance: ExactRational

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not _valid_identifier(self.component_id):
            issues.append("residual_component_id_invalid")
        if not isinstance(self.value, ExactRational):
            issues.append("residual_component_value_invalid")
        elif self.value.fraction < 0:
            issues.append("residual_component_value_negative")
        if not isinstance(self.tolerance, ExactRational):
            issues.append("residual_component_tolerance_invalid")
        elif self.tolerance.fraction < 0:
            issues.append("residual_component_tolerance_negative")
        return tuple(sorted(set(issues)))

    def private_payload(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "value": (
                self.value.safe_payload()
                if isinstance(self.value, ExactRational)
                else None
            ),
            "tolerance": (
                self.tolerance.safe_payload()
                if isinstance(self.tolerance, ExactRational)
                else None
            ),
        }


@dataclass(frozen=True)
class ContrastiveResidual:
    transformation_id: str
    operator_contract_hash: str
    transformed_input_hash: str
    policy_hash: str
    disposition: ResidualDisposition
    components: tuple[ResidualComponent, ...]

    @property
    def component_commitment(self) -> str:
        return strict_content_hash(
            [
                component.private_payload()
                for component in sorted(
                    self.components, key=lambda row: row.component_id
                )
            ]
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not _valid_identifier(self.transformation_id):
            issues.append("contrastive_transformation_id_invalid")
        if not _is_sha256(self.operator_contract_hash):
            issues.append("contrastive_operator_contract_hash_invalid")
        if not _is_sha256(self.transformed_input_hash):
            issues.append("contrastive_input_hash_invalid")
        if not _is_sha256(self.policy_hash):
            issues.append("contrastive_policy_hash_invalid")
        if not isinstance(self.disposition, ResidualDisposition):
            issues.append("contrastive_disposition_invalid")
        if not isinstance(self.components, tuple) or not self.components:
            issues.append("contrastive_components_invalid")
        else:
            component_ids = tuple(
                component.component_id for component in self.components
            )
            if len(component_ids) != len(set(component_ids)):
                issues.append("contrastive_components_duplicate")
            issues.extend(
                issue
                for component in self.components
                for issue in component.validate()
            )
            if (
                self.disposition
                in {
                    ResidualDisposition.SATISFIED,
                    ResidualDisposition.VIOLATED,
                }
                and (
                    all(
                        component.value.fraction
                        <= component.tolerance.fraction
                        for component in self.components
                    )
                    != (
                        self.disposition
                        is ResidualDisposition.SATISFIED
                    )
                )
            ):
                issues.append("contrastive_disposition_component_mismatch")
        return tuple(sorted(set(issues)))

    def private_payload(self) -> dict[str, Any]:
        return {
            "transformation_id": self.transformation_id,
            "operator_contract_hash": self.operator_contract_hash,
            "transformed_input_hash": self.transformed_input_hash,
            "policy_hash": self.policy_hash,
            "disposition": (
                self.disposition.value
                if isinstance(self.disposition, ResidualDisposition)
                else None
            ),
            "components": [
                component.private_payload()
                for component in sorted(
                    self.components, key=lambda row: row.component_id
                )
            ],
        }

    def safe_payload(self) -> dict[str, Any]:
        return {
            "transformation_id": self.transformation_id,
            "operator_contract_hash": self.operator_contract_hash,
            "transformed_input_hash": self.transformed_input_hash,
            "policy_hash": self.policy_hash,
            "disposition": self.disposition.value,
            "component_count": len(self.components),
            "component_commitment": self.component_commitment,
        }


@dataclass(frozen=True)
class LawResidualReceipt:
    receipt_id: str
    law_id: str
    registry_hash: str
    schema_hash: str
    episode_hash: str
    binding_hash: str
    evaluation_input_hash: str
    policy_hash: str
    verifier_id: str
    verifier_version: str
    verifier_contract_hash: str
    disposition: ResidualDisposition
    components: tuple[ResidualComponent, ...]
    missing_observables: tuple[str, ...]
    applicability_failures: tuple[str, ...]
    contrastive_residuals: tuple[ContrastiveResidual, ...]
    evidence_span_ids: tuple[str, ...]

    @property
    def receipt_hash(self) -> str:
        return strict_content_hash(self.private_payload())

    @property
    def component_commitment(self) -> str:
        return strict_content_hash(
            [
                item.private_payload()
                for item in sorted(
                    self.components, key=lambda row: row.component_id
                )
            ]
        )

    @property
    def contrastive_commitment(self) -> str:
        return strict_content_hash(
            [
                item.private_payload()
                for item in sorted(
                    self.contrastive_residuals,
                    key=lambda row: row.transformation_id,
                )
            ]
        )

    def validate(
        self,
        registry: GSCLSchemaRegistry,
        schema: ExecutableLawSchema,
        episode: StructuralEpisode,
        binding: LawBinding,
        *,
        expected_policy_hash: str | None = None,
    ) -> tuple[str, ...]:
        issues: list[str] = []
        if not _valid_identifier(self.receipt_id):
            issues.append("law_residual_receipt_id_invalid")
        if self.law_id != schema.law_id:
            issues.append("law_residual_receipt_law_mismatch")
        if self.registry_hash != registry.registry_hash:
            issues.append("law_residual_receipt_registry_mismatch")
        if self.schema_hash != schema.schema_hash:
            issues.append("law_residual_receipt_schema_mismatch")
        if self.episode_hash != episode.episode_hash:
            issues.append("law_residual_receipt_episode_mismatch")
        if self.binding_hash != binding.binding_hash:
            issues.append("law_residual_receipt_binding_mismatch")
        if self.evaluation_input_hash != binding.evaluation_input_hash:
            issues.append("law_residual_evaluation_input_mismatch")
        if not _is_sha256(self.policy_hash):
            issues.append("law_residual_policy_hash_invalid")
        if (
            expected_policy_hash is not None
            and self.policy_hash != expected_policy_hash
        ):
            issues.append("law_residual_policy_hash_mismatch")
        if self.verifier_id != schema.residual_function_id:
            issues.append("law_residual_verifier_id_mismatch")
        if self.verifier_version != schema.verifier_version:
            issues.append("law_residual_verifier_version_mismatch")
        if (
            self.verifier_contract_hash
            != schema.verifier_contract_hash
        ):
            issues.append("law_residual_verifier_hash_mismatch")
        if not isinstance(self.disposition, ResidualDisposition):
            issues.append("law_residual_disposition_invalid")

        if not isinstance(self.components, tuple):
            issues.append("law_residual_components_invalid")
            component_ids: tuple[str, ...] = ()
        else:
            component_ids = tuple(
                item.component_id for item in self.components
            )
            if len(component_ids) != len(set(component_ids)):
                issues.append("law_residual_components_duplicate")
            issues.extend(
                issue
                for item in self.components
                for issue in item.validate()
            )
        for values, issue in (
            (
                self.missing_observables,
                "law_residual_missing_observables_invalid",
            ),
            (
                self.applicability_failures,
                "law_residual_applicability_failures_invalid",
            ),
            (
                self.evidence_span_ids,
                "law_residual_evidence_invalid",
            ),
        ):
            if not _unique_strings(
                values,
                allow_empty=(
                    issue
                    != "law_residual_evidence_invalid"
                ),
            ):
                issues.append(issue)

        if (
            self.disposition
            in {
                ResidualDisposition.SATISFIED,
                ResidualDisposition.VIOLATED,
            }
        ):
            if set(component_ids) != set(schema.expected_component_ids):
                issues.append("law_residual_component_contract_mismatch")
            if self.missing_observables or self.applicability_failures:
                issues.append("law_residual_decided_with_abstention")
            if self.components and all(
                component.value.fraction
                <= component.tolerance.fraction
                for component in self.components
            ) != (
                self.disposition is ResidualDisposition.SATISFIED
            ):
                issues.append("law_residual_disposition_component_mismatch")
        elif self.disposition is ResidualDisposition.INCONCLUSIVE:
            if (
                not self.missing_observables
                or self.applicability_failures
                or self.components
            ):
                issues.append("law_residual_inconclusive_contract_invalid")
        elif self.disposition is ResidualDisposition.NOT_APPLICABLE:
            if (
                not self.applicability_failures
                or self.missing_observables
                or self.components
            ):
                issues.append(
                    "law_residual_not_applicable_contract_invalid"
                )

        binding_issues = validate_law_binding(
            registry, schema, episode, binding
        )
        if binding_issues:
            issues.append("law_residual_binding_invalid")
        episode_span_ids = frozenset(
            item.span_id for item in episode.evidence_spans
        )
        bound_span_ids = set(
            (
                span_id
                for row in binding.role_bindings
                for span_id in row.evidence_span_ids
            )
        )
        for row in binding.observable_bindings:
            try:
                observable = episode.require_observable(
                    row.observable_id
                )
            except KeyError:
                issues.append("law_residual_bound_observable_unknown")
                continue
            bound_span_ids.update(observable.evidence_span_ids)
        if isinstance(self.evidence_span_ids, tuple) and any(
            span_id not in episode_span_ids
            or span_id not in bound_span_ids
            for span_id in self.evidence_span_ids
        ):
            issues.append("law_residual_evidence_not_bound")

        if not isinstance(self.contrastive_residuals, tuple):
            issues.append("law_residual_contrastives_invalid")
        else:
            transformation_ids = tuple(
                row.transformation_id
                for row in self.contrastive_residuals
            )
            if len(transformation_ids) != len(set(transformation_ids)):
                issues.append("law_residual_contrastives_duplicate")
            expected_transformations = (
                set(schema.hard_negative_transformations)
                if self.disposition
                in {
                    ResidualDisposition.SATISFIED,
                    ResidualDisposition.VIOLATED,
                }
                else set()
            )
            if set(transformation_ids) != expected_transformations:
                issues.append(
                    "law_residual_contrastive_coverage_mismatch"
                )
            if any(
                row.operator_contract_hash
                != HARD_NEGATIVE_OPERATOR_CONTRACT_HASHES.get(
                    row.transformation_id
                )
                for row in self.contrastive_residuals
            ):
                issues.append(
                    "law_residual_contrastive_operator_contract_mismatch"
                )
            if any(
                row.policy_hash != self.policy_hash
                for row in self.contrastive_residuals
            ):
                issues.append("law_residual_contrastive_policy_mismatch")
            issues.extend(
                issue
                for row in self.contrastive_residuals
                for issue in row.validate()
            )
        return tuple(sorted(set(issues)))

    def private_payload(self) -> dict[str, Any]:
        return {
            "schema_version": GSCL_SCHEMA_VERSION,
            "receipt_id": self.receipt_id,
            "law_id": self.law_id,
            "registry_hash": self.registry_hash,
            "schema_hash": self.schema_hash,
            "episode_hash": self.episode_hash,
            "binding_hash": self.binding_hash,
            "evaluation_input_hash": self.evaluation_input_hash,
            "policy_hash": self.policy_hash,
            "verifier_id": self.verifier_id,
            "verifier_version": self.verifier_version,
            "verifier_contract_hash": self.verifier_contract_hash,
            "disposition": (
                self.disposition.value
                if isinstance(self.disposition, ResidualDisposition)
                else None
            ),
            "components": [
                item.private_payload()
                for item in sorted(
                    self.components, key=lambda row: row.component_id
                )
            ],
            "missing_observables": list(
                sorted(self.missing_observables)
            ),
            "applicability_failures": list(
                sorted(self.applicability_failures)
            ),
            "contrastive_residuals": [
                item.private_payload()
                for item in sorted(
                    self.contrastive_residuals,
                    key=lambda row: row.transformation_id,
                )
            ],
            "evidence_span_ids": list(
                sorted(self.evidence_span_ids)
            ),
        }

    def safe_payload(self) -> dict[str, Any]:
        return {
            "schema_version": GSCL_SCHEMA_VERSION,
            "receipt_hash": self.receipt_hash,
            "law_id": self.law_id,
            "registry_hash": self.registry_hash,
            "schema_hash": self.schema_hash,
            "episode_hash": self.episode_hash,
            "binding_hash": self.binding_hash,
            "evaluation_input_hash": self.evaluation_input_hash,
            "policy_hash": self.policy_hash,
            "verifier_id": self.verifier_id,
            "verifier_version": self.verifier_version,
            "verifier_contract_hash": self.verifier_contract_hash,
            "disposition": self.disposition.value,
            "component_count": len(self.components),
            "component_commitment": self.component_commitment,
            "missing_observable_count": len(self.missing_observables),
            "missing_observable_commitment": strict_content_hash(
                list(sorted(self.missing_observables))
            ),
            "applicability_failure_count": len(
                self.applicability_failures
            ),
            "applicability_failure_commitment": strict_content_hash(
                list(sorted(self.applicability_failures))
            ),
            "contrastive_count": len(self.contrastive_residuals),
            "contrastive_commitment": self.contrastive_commitment,
            "evidence_span_count": len(self.evidence_span_ids),
            "evidence_span_commitment": strict_content_hash(
                list(sorted(self.evidence_span_ids))
            ),
        }


def _binding_role_maps(
    schema: ExecutableLawSchema,
    binding: LawBinding,
) -> tuple[
    dict[str, LawRoleSpec],
    dict[tuple[RoleTargetKind, str], str],
]:
    role_specs = {row.role_id: row for row in schema.roles}
    target_to_role = {
        (role_specs[row.role_id].target_kind, row.target_id): row.role_id
        for row in binding.role_bindings
        if row.role_id in role_specs
    }
    return role_specs, target_to_role


def canonical_structural_signature(
    registry: GSCLSchemaRegistry,
    schema: ExecutableLawSchema,
    episode: StructuralEpisode,
    binding: LawBinding,
) -> dict[str, Any]:
    """Canonicalize the role-induced typed ``O/M/H/C`` subdiagram."""

    issues = validate_law_binding(
        registry, schema, episode, binding
    )
    if issues:
        raise PermissionError(
            "cannot sign invalid law binding: " + ",".join(issues)
        )
    role_specs, target_to_role = _binding_role_maps(schema, binding)
    role_rows: list[dict[str, Any]] = []
    for row in sorted(
        binding.role_bindings, key=lambda item: item.role_id
    ):
        spec = role_specs[row.role_id]
        target = episode.require_target(
            spec.target_kind, row.target_id
        )
        role_row: dict[str, Any] = {
            "role_id": row.role_id,
            "target_kind": spec.target_kind.value,
            "target_type": _target_type(target),
        }
        if isinstance(target, StructuralQuantity):
            role_row.update(
                {
                    "dimension": target.dimension,
                    "unit": target.unit,
                    "owner_role": target_to_role.get(
                        (
                            RoleTargetKind.OBJECT,
                            target.owner_object_id,
                        )
                    ),
                }
            )
        role_rows.append(role_row)

    relation_rows: list[dict[str, Any]] = []
    for relation in episode.relations:
        relation_role = target_to_role.get(
            (RoleTargetKind.RELATION, relation.relation_id)
        )
        source_role = target_to_role.get(
            (RoleTargetKind.OBJECT, relation.source_object_id)
        )
        target_role = target_to_role.get(
            (RoleTargetKind.OBJECT, relation.target_object_id)
        )
        if not any((relation_role, source_role, target_role)):
            continue
        relation_rows.append(
            {
                "relation_role": relation_role,
                "relation_type": relation.relation_type,
                "source_role": source_role,
                "target_role": target_role,
                "relative_order": relation.order_index,
            }
        )

    quantity_rows: list[dict[str, Any]] = []
    for quantity in episode.quantities:
        quantity_role = target_to_role.get(
            (RoleTargetKind.QUANTITY, quantity.quantity_id)
        )
        owner_role = target_to_role.get(
            (RoleTargetKind.OBJECT, quantity.owner_object_id)
        )
        if not any((quantity_role, owner_role)):
            continue
        quantity_rows.append(
            {
                "quantity_role": quantity_role,
                "owner_role": owner_role,
                "dimension": quantity.dimension,
                "unit": quantity.unit,
            }
        )

    hyperrelation_rows: list[dict[str, Any]] = []
    for hyperrelation in episode.hyperrelations:
        hyperrelation_role = target_to_role.get(
            (
                RoleTargetKind.HYPERRELATION,
                hyperrelation.hyperrelation_id,
            )
        )
        endpoints = [
            {
                "endpoint_role": endpoint.endpoint_role,
                "object_role": target_to_role.get(
                    (RoleTargetKind.OBJECT, endpoint.object_id)
                ),
            }
            for endpoint in sorted(
                hyperrelation.endpoints,
                key=lambda row: row.endpoint_role,
            )
        ]
        if hyperrelation_role is None and not any(
            row["object_role"] for row in endpoints
        ):
            continue
        hyperrelation_rows.append(
            {
                "hyperrelation_role": hyperrelation_role,
                "hyperrelation_type": hyperrelation.hyperrelation_type,
                "endpoints": endpoints,
            }
        )

    constraint_rows: list[dict[str, Any]] = []
    for constraint in episode.constraints:
        constraint_role = target_to_role.get(
            (RoleTargetKind.CONSTRAINT, constraint.constraint_id)
        )
        participants = [
            {
                "participant_role": participant.participant_role,
                "target_role": target_to_role.get(
                    (
                        participant.target_kind,
                        participant.target_id,
                    )
                ),
            }
            for participant in sorted(
                constraint.participants,
                key=lambda row: row.participant_role,
            )
        ]
        if constraint_role is None and not any(
            row["target_role"] for row in participants
        ):
            continue
        constraint_rows.append(
            {
                "constraint_role": constraint_role,
                "constraint_type": constraint.constraint_type,
                "participants": participants,
                "observable_ids": list(
                    sorted(constraint.observable_ids)
                ),
            }
        )

    observables = {
        row.observable_id: episode.require_observable(
            row.observable_id
        )
        for row in binding.observable_bindings
    }
    observable_rows = [
        {
            "observable_id": observable_id,
            "value_type": observable.value_type.value,
            "observation_status": observable.observation_status.value,
            "dimension": observable.dimension,
            "unit": observable.unit,
        }
        for observable_id, observable in sorted(observables.items())
    ]
    boundary_role = (
        None
        if episode.declared_boundary_object_id is None
        else target_to_role.get(
            (
                RoleTargetKind.OBJECT,
                episode.declared_boundary_object_id,
            )
        )
    )
    return {
        "schema_version": GSCL_SCHEMA_VERSION,
        "law_id": schema.law_id,
        "law_kind": schema.law_kind.value,
        "residual_function_id": schema.residual_function_id,
        "verifier_version": schema.verifier_version,
        "verifier_contract_hash": schema.verifier_contract_hash,
        "role_rows": role_rows,
        "relation_rows": sorted(
            relation_rows,
            key=lambda row: strict_canonical_bytes(row),
        ),
        "quantity_rows": sorted(
            quantity_rows,
            key=lambda row: strict_canonical_bytes(row),
        ),
        "hyperrelation_rows": sorted(
            hyperrelation_rows,
            key=lambda row: strict_canonical_bytes(row),
        ),
        "constraint_rows": sorted(
            constraint_rows,
            key=lambda row: strict_canonical_bytes(row),
        ),
        "observable_rows": observable_rows,
        "boundary_role": boundary_role,
    }


@dataclass(frozen=True)
class CorrespondencePair:
    role_id: str
    source_target_id: str
    target_target_id: str

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        for value, issue in (
            (self.role_id, "correspondence_pair_role_invalid"),
            (
                self.source_target_id,
                "correspondence_pair_source_invalid",
            ),
            (
                self.target_target_id,
                "correspondence_pair_target_invalid",
            ),
        ):
            if not _valid_identifier(value):
                issues.append(issue)
        return tuple(sorted(set(issues)))

    def private_payload(self) -> dict[str, str]:
        return {
            "role_id": self.role_id,
            "source_target_id": self.source_target_id,
            "target_target_id": self.target_target_id,
        }


@dataclass(frozen=True)
class StructuralCorrespondence:
    correspondence_id: str
    law_id: str
    registry_hash: str
    schema_hash: str
    source_episode_hash: str
    target_episode_hash: str
    source_binding_hash: str
    target_binding_hash: str
    source_receipt_hash: str
    target_receipt_hash: str
    policy_hash: str
    pairs: tuple[CorrespondencePair, ...]
    preserved_constraints: tuple[str, ...]
    broken_constraints: tuple[str, ...]
    unresolved_constraints: tuple[str, ...]
    disposition: CorrespondenceDisposition

    @property
    def correspondence_hash(self) -> str:
        return strict_content_hash(self.private_payload())

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not _valid_identifier(self.correspondence_id):
            issues.append("structural_correspondence_id_invalid")
        if not _valid_identifier(self.law_id):
            issues.append("structural_correspondence_law_invalid")
        for value, issue in (
            (self.registry_hash, "structural_correspondence_registry_invalid"),
            (self.schema_hash, "structural_correspondence_schema_invalid"),
            (
                self.source_episode_hash,
                "structural_correspondence_source_episode_invalid",
            ),
            (
                self.target_episode_hash,
                "structural_correspondence_target_episode_invalid",
            ),
            (
                self.source_binding_hash,
                "structural_correspondence_source_binding_invalid",
            ),
            (
                self.target_binding_hash,
                "structural_correspondence_target_binding_invalid",
            ),
            (
                self.source_receipt_hash,
                "structural_correspondence_source_receipt_invalid",
            ),
            (
                self.target_receipt_hash,
                "structural_correspondence_target_receipt_invalid",
            ),
            (self.policy_hash, "structural_correspondence_policy_invalid"),
        ):
            if not _is_sha256(value):
                issues.append(issue)
        if not isinstance(self.pairs, tuple) or not self.pairs:
            issues.append("structural_correspondence_pairs_invalid")
        else:
            role_ids = tuple(pair.role_id for pair in self.pairs)
            if len(role_ids) != len(set(role_ids)):
                issues.append("structural_correspondence_pairs_duplicate")
            issues.extend(
                issue
                for pair in self.pairs
                for issue in pair.validate()
            )
        for values, issue in (
            (
                self.preserved_constraints,
                "structural_correspondence_preserved_invalid",
            ),
            (
                self.broken_constraints,
                "structural_correspondence_broken_invalid",
            ),
            (
                self.unresolved_constraints,
                "structural_correspondence_unresolved_invalid",
            ),
        ):
            if not _unique_strings(values, allow_empty=True):
                issues.append(issue)
        if not isinstance(
            self.disposition, CorrespondenceDisposition
        ):
            issues.append("structural_correspondence_disposition_invalid")
        elif (
            self.disposition is CorrespondenceDisposition.ACCEPTED
            and (
                self.broken_constraints
                or self.unresolved_constraints
            )
        ):
            issues.append(
                "structural_correspondence_accepted_with_failures"
            )
        elif (
            self.disposition is CorrespondenceDisposition.REJECTED
            and not self.broken_constraints
        ):
            issues.append(
                "structural_correspondence_rejected_without_break"
            )
        elif (
            self.disposition is CorrespondenceDisposition.INCONCLUSIVE
            and not self.unresolved_constraints
        ):
            issues.append(
                "structural_correspondence_inconclusive_without_unresolved"
            )
        return tuple(sorted(set(issues)))

    def private_payload(self) -> dict[str, Any]:
        return {
            "schema_version": GSCL_SCHEMA_VERSION,
            "correspondence_id": self.correspondence_id,
            "law_id": self.law_id,
            "registry_hash": self.registry_hash,
            "schema_hash": self.schema_hash,
            "source_episode_hash": self.source_episode_hash,
            "target_episode_hash": self.target_episode_hash,
            "source_binding_hash": self.source_binding_hash,
            "target_binding_hash": self.target_binding_hash,
            "source_receipt_hash": self.source_receipt_hash,
            "target_receipt_hash": self.target_receipt_hash,
            "policy_hash": self.policy_hash,
            "pairs": [
                pair.private_payload()
                for pair in sorted(
                    self.pairs, key=lambda row: row.role_id
                )
            ],
            "preserved_constraints": list(
                sorted(self.preserved_constraints)
            ),
            "broken_constraints": list(
                sorted(self.broken_constraints)
            ),
            "unresolved_constraints": list(
                sorted(self.unresolved_constraints)
            ),
            "disposition": (
                self.disposition.value
                if isinstance(
                    self.disposition, CorrespondenceDisposition
                )
                else None
            ),
        }

    def safe_payload(self) -> dict[str, Any]:
        return {
            "schema_version": GSCL_SCHEMA_VERSION,
            "correspondence_hash": self.correspondence_hash,
            "law_id": self.law_id,
            "registry_hash": self.registry_hash,
            "schema_hash": self.schema_hash,
            "source_episode_hash": self.source_episode_hash,
            "target_episode_hash": self.target_episode_hash,
            "source_binding_hash": self.source_binding_hash,
            "target_binding_hash": self.target_binding_hash,
            "source_receipt_hash": self.source_receipt_hash,
            "target_receipt_hash": self.target_receipt_hash,
            "policy_hash": self.policy_hash,
            "pair_count": len(self.pairs),
            "pair_commitment": strict_content_hash(
                [
                    pair.private_payload()
                    for pair in sorted(
                        self.pairs, key=lambda row: row.role_id
                    )
                ]
            ),
            "preserved_constraints": list(
                sorted(self.preserved_constraints)
            ),
            "broken_constraints": list(
                sorted(self.broken_constraints)
            ),
            "unresolved_constraints": list(
                sorted(self.unresolved_constraints)
            ),
            "disposition": self.disposition.value,
        }


def compare_structural_bindings(
    registry: GSCLSchemaRegistry,
    schema: ExecutableLawSchema,
    source_episode: StructuralEpisode,
    source_binding: LawBinding,
    source_receipt: LawResidualReceipt,
    target_episode: StructuralEpisode,
    target_binding: LawBinding,
    target_receipt: LawResidualReceipt,
    *,
    correspondence_id: str,
    source_policy: Any | None = None,
    target_policy: Any | None = None,
) -> StructuralCorrespondence:
    """Compare two trusted-recomputed bindings without collapsing abstention."""

    broken: list[str] = []
    unresolved: list[str] = []
    source_binding_issues = validate_law_binding(
        registry, schema, source_episode, source_binding
    )
    target_binding_issues = validate_law_binding(
        registry, schema, target_episode, target_binding
    )
    source_receipt_issues = source_receipt.validate(
        registry, schema, source_episode, source_binding
    )
    target_receipt_issues = target_receipt.validate(
        registry, schema, target_episode, target_binding
    )
    if source_binding_issues:
        unresolved.append("source_binding_invalid")
    if target_binding_issues:
        unresolved.append("target_binding_invalid")
    if source_receipt_issues:
        unresolved.append("source_receipt_invalid")
    if target_receipt_issues:
        unresolved.append("target_receipt_invalid")
    if source_policy is None or target_policy is None:
        unresolved.append("trusted_recomputation_not_supplied")
    else:
        # Imported lazily to preserve the schema→kernel dependency direction.
        from .structural_law_residuals_v1 import (  # noqa: PLC0415
            verify_law_residual_receipt_trusted,
        )

        source_trusted_issues = verify_law_residual_receipt_trusted(
            source_receipt,
            registry,
            schema,
            source_episode,
            source_binding,
            source_policy,
        )
        target_trusted_issues = verify_law_residual_receipt_trusted(
            target_receipt,
            registry,
            schema,
            target_episode,
            target_binding,
            target_policy,
        )
        if source_trusted_issues:
            unresolved.append(
                "source_receipt_trusted_recomputation_failed"
            )
        if target_trusted_issues:
            unresolved.append(
                "target_receipt_trusted_recomputation_failed"
            )

    structural_match = False
    if not unresolved:
        source_signature = canonical_structural_signature(
            registry, schema, source_episode, source_binding
        )
        target_signature = canonical_structural_signature(
            registry, schema, target_episode, target_binding
        )
        structural_match = source_signature == target_signature
        if not structural_match:
            broken.append("role_normalized_typed_structure")

    dispositions = {
        source_receipt.disposition,
        target_receipt.disposition,
    }
    if dispositions & {
        ResidualDisposition.NOT_APPLICABLE,
        ResidualDisposition.INCONCLUSIVE,
    }:
        unresolved.append("law_residual_abstention")
    elif ResidualDisposition.VIOLATED in dispositions:
        broken.append("law_residual_satisfaction")
    if source_receipt.policy_hash != target_receipt.policy_hash:
        broken.append("law_residual_policy")

    source_contrastive = {
        row.transformation_id: row.disposition
        for row in source_receipt.contrastive_residuals
    }
    target_contrastive = {
        row.transformation_id: row.disposition
        for row in target_receipt.contrastive_residuals
    }
    required_hard_negatives = set(
        schema.hard_negative_transformations
    )
    if (
        set(source_contrastive) != required_hard_negatives
        or set(target_contrastive) != required_hard_negatives
    ):
        unresolved.append("hard_negative_coverage")
    elif (
        source_contrastive != target_contrastive
        or any(
            disposition is not ResidualDisposition.VIOLATED
            for disposition in source_contrastive.values()
        )
    ):
        broken.append("hard_negative_rejection_behavior")

    source_by_role = {
        row.role_id: row.target_id for row in source_binding.role_bindings
    }
    target_by_role = {
        row.role_id: row.target_id for row in target_binding.role_bindings
    }
    common_roles = sorted(set(source_by_role) & set(target_by_role))
    pairs = tuple(
        CorrespondencePair(
            role_id=role_id,
            source_target_id=source_by_role[role_id],
            target_target_id=target_by_role[role_id],
        )
        for role_id in common_roles
    )
    if unresolved:
        disposition = CorrespondenceDisposition.INCONCLUSIVE
        preserved: tuple[str, ...] = ()
    elif broken:
        disposition = CorrespondenceDisposition.REJECTED
        preserved = ()
    elif structural_match:
        disposition = CorrespondenceDisposition.ACCEPTED
        preserved = (
            "hard_negative_rejection_behavior",
            "law_residual_policy_and_sign",
            "object_relation_hyperrelation_constraint_incidence",
            "role_types",
            "source_target_and_quantity_owner",
        )
    else:
        disposition = CorrespondenceDisposition.INCONCLUSIVE
        unresolved.append("structural_comparison_unavailable")
        preserved = ()
    correspondence = StructuralCorrespondence(
        correspondence_id=correspondence_id,
        law_id=schema.law_id,
        registry_hash=registry.registry_hash,
        schema_hash=schema.schema_hash,
        source_episode_hash=source_episode.episode_hash,
        target_episode_hash=target_episode.episode_hash,
        source_binding_hash=source_binding.binding_hash,
        target_binding_hash=target_binding.binding_hash,
        source_receipt_hash=source_receipt.receipt_hash,
        target_receipt_hash=target_receipt.receipt_hash,
        policy_hash=source_receipt.policy_hash,
        pairs=pairs,
        preserved_constraints=preserved,
        broken_constraints=tuple(sorted(set(broken))),
        unresolved_constraints=tuple(sorted(set(unresolved))),
        disposition=disposition,
    )
    issues = correspondence.validate()
    if issues:
        raise PermissionError(
            "invalid structural correspondence: " + ",".join(issues)
        )
    return correspondence


__all__ = [
    "ConstraintParticipant",
    "ContrastiveResidual",
    "CorrespondenceDisposition",
    "CorrespondencePair",
    "EvidenceSpanRef",
    "ExactRational",
    "ExecutableLawSchema",
    "GSCL_SCHEMA_VERSION",
    "GSCL_RESIDUAL_KERNEL_VERSION",
    "GSCLSchemaRegistry",
    "HyperRoleEndpoint",
    "InferenceProvenance",
    "LawBinding",
    "LawKind",
    "LawResidualReceipt",
    "LawRoleSpec",
    "ObservableBinding",
    "ObservableSpec",
    "ObservableValueType",
    "ObservationStatus",
    "FROZEN_UAO_V1_ONTOLOGY_HASH",
    "HARD_NEGATIVE_OPERATOR_CONTRACT_HASHES",
    "RESIDUAL_KERNEL_CONTRACT_HASH",
    "ResidualComponent",
    "ResidualDisposition",
    "RoleBinding",
    "RoleTargetKind",
    "StructuralConstraint",
    "StructuralCorrespondence",
    "StructuralEpisode",
    "StructuralHyperrelation",
    "StructuralObject",
    "StructuralQuantity",
    "StructuralRelation",
    "TypedObservable",
    "build_gscl_schema_registry_v1",
    "canonical_structural_signature",
    "compare_structural_bindings",
    "strict_canonical_bytes",
    "strict_content_hash",
    "validate_law_binding",
]
