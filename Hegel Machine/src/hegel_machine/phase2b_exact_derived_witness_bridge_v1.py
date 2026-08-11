"""Exact law witnesses rebuilt from content-rooted transformed observations.

This bridge never treats the legacy base-observation footprint as evidence at
every scale.  Its only public entrypoint accepts the v2 transform authority,
the frozen theory, and the frozen wire registry.  It internally reruns exact
transform compilation, inventories every derived observation/component
receipt, derives exact support slices, and commits the complete
law-by-binding-by-scale-slice candidate grid before evaluating any law.

The v1 task scope is intentionally strict: task entities and quantities must
exactly cover the public authority/registry, and every transformed observation
must occur in at least one candidate witness slot.  A future relevance
certificate may relax that rule; this module does not silently ignore inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from itertools import product
from typing import Final

from .hashing import stable_hash
from .phase2b_adapter import (
    LawWireBinding,
    Phase2BAdapterRegistry,
)
from .phase2b_exact_bridge_v1 import (
    DEFAULT_EXACT_BRIDGE_POLICY,
    DEFAULT_EXACT_SELECTION_POLICY,
    ExactBridgeDisposition,
    ExactCandidateStatus,
    ExactFractionAtom,
    ExactInterval,
    ExactObservableKind,
    ExactSelectionDisposition,
    _ArithmeticBudget,
    _EXACT_VERIFIERS,
    _ResourceLimit,
    _bounded_bundle_preflight,
    _divide,
    _observable_registry,
    _require_exact_authority_tree,
    _shallow_authority_size_preflight,
    _tolerance,
    _validate_registry,
)
from .phase2b_exact_transform_semantics_v1 import (
    EXACT_TRANSFORM_SEMANTICS_VERSION,
    ComponentValueKind,
    DerivedObservationDescriptor,
    ExactSpatialSupport,
    ExactTemporalSupport,
    ExactTransformCompilation,
    ExactTransformPreflightRejection,
    ExactTransformedComponent,
    PublicTransformEvidenceBundleV2,
    TransformCompilationDisposition,
    _AuthorityTreeBudget as _TransformAuthorityTreeBudget,
    _DEFAULT_POLICY as _TRANSFORM_POLICY,
    _detailed_resource_preflight as _transform_detailed_preflight,
    _forest_plan as _transform_forest_plan,
    _contract_commitment as _transform_contract_commitment,
    _graph_commitment as _transform_graph_commitment,
    _shallow_preflight as _transform_shallow_preflight,
    _validate_contract_index as _transform_contract_preflight,
    _validate_metadata as _transform_metadata_preflight,
    run_exact_transform_semantics,
)
from .phase2b_uncertainty_compiler import compile_bundle_uncertainty
from .phase2b_wire import PublicEvidenceBundle, RoleBinding
from .schema import LawKind, RelationLaw, TheoryState, require_tuple


EXACT_DERIVED_WITNESS_BRIDGE_VERSION: Final = (
    "hegel-machine-phase2b-exact-derived-witness-bridge/1"
)
EXACT_DERIVED_WITNESS_MATCHER_VERSION: Final = (
    "hegel-machine-phase2b-exact-derived-witness-matcher/1"
)


@dataclass(frozen=True, slots=True)
class _DerivedBridgePolicy:
    version: str = EXACT_DERIVED_WITNESS_BRIDGE_VERSION
    matcher_version: str = EXACT_DERIVED_WITNESS_MATCHER_VERSION
    transform_semantics_version: str = EXACT_TRANSFORM_SEMANTICS_VERSION
    verifier_semantics_id: str = (
        DEFAULT_EXACT_BRIDGE_POLICY.verifier_semantics_id
    )
    maximum_candidate_count: int = 50_000
    maximum_support_slice_count: int = 65_536
    maximum_inventory_observations: int = 262_144
    maximum_inventory_components: int = 262_144
    maximum_match_scan_work: int = 1_000_000
    maximum_slot_match_count: int = 4_096
    maximum_aggregate_replay_work: int = 100_000

    def __post_init__(self) -> None:
        if (
            self.version,
            self.matcher_version,
            self.transform_semantics_version,
            self.verifier_semantics_id,
        ) != (
            EXACT_DERIVED_WITNESS_BRIDGE_VERSION,
            EXACT_DERIVED_WITNESS_MATCHER_VERSION,
            EXACT_TRANSFORM_SEMANTICS_VERSION,
            DEFAULT_EXACT_BRIDGE_POLICY.verifier_semantics_id,
        ):
            raise ValueError("derived witness semantic identity drift")
        if (
            self.maximum_candidate_count,
            self.maximum_support_slice_count,
            self.maximum_inventory_observations,
            self.maximum_inventory_components,
            self.maximum_match_scan_work,
            self.maximum_slot_match_count,
            self.maximum_aggregate_replay_work,
        ) != (
            50_000,
            65_536,
            262_144,
            262_144,
            1_000_000,
            4_096,
            100_000,
        ):
            raise ValueError("derived witness resource budget drift")

    @property
    def matcher_semantics_id(self) -> str:
        return stable_hash(
            (
                self.matcher_version,
                "exact_scale_temporal_spatial_support_slice",
                "exact_quantity_entity_and_required_wire_role_match",
                "zero_one_many_matches_are_committed",
                "strict_task_scope_and_no_unused_inventory_observation",
                "complete_law_binding_scale_slice_grid",
                "transformed_values_only_no_legacy_footprint",
                "scale_normalized_hull_min_lower_max_upper",
                "ambiguous_matches_do_not_consume_inventory",
                "registry_candidate_budget_must_equal_frozen_limit",
                "every_slot_costs_one_plus_match_pool_size",
                "at_least_one_explicit_temporal_or_spatial_support_required",
                (
                    "aggregate_replay_is_two_bounded_scans_plus_canonical_"
                    "group_sort"
                ),
                self.verifier_semantics_id,
            ),
            prefix="phase2b_exact_derived_matcher_",
        )

    @property
    def policy_id(self) -> str:
        return stable_hash(self, prefix="phase2b_exact_derived_bridge_policy_")


_DEFAULT_POLICY: Final = _DerivedBridgePolicy()
_DEFAULT_POLICY_ID: Final = _DEFAULT_POLICY.policy_id
_DEFAULT_MATCHER_SEMANTICS_ID: Final = _DEFAULT_POLICY.matcher_semantics_id
_DEFAULT_SELECTION_POLICY_ID: Final = DEFAULT_EXACT_SELECTION_POLICY.policy_id


@dataclass(frozen=True, slots=True)
class ExactSupportSlice:
    scale_id: str
    temporal_support: ExactTemporalSupport | None
    spatial_support: ExactSpatialSupport | None

    def __post_init__(self) -> None:
        if type(self.scale_id) is not str or not self.scale_id:
            raise ValueError("exact support slice needs a scale ID")
        if self.temporal_support is not None and type(
            self.temporal_support
        ) is not ExactTemporalSupport:
            raise TypeError("exact support slice temporal support has wrong type")
        if self.spatial_support is not None and type(
            self.spatial_support
        ) is not ExactSpatialSupport:
            raise TypeError("exact support slice spatial support has wrong type")

    @property
    def support_slice_id(self) -> str:
        return stable_hash(self, prefix="phase2b_exact_support_slice_")


@dataclass(frozen=True, slots=True)
class DerivedInventoryObservation:
    descriptor: DerivedObservationDescriptor
    observation_descriptor_id: str
    support_slice: ExactSupportSlice
    ordered_component_receipt_ids: tuple[str, ...]
    source_observation_ids: tuple[str, ...]
    used_observation_compilation_ids: tuple[str, ...]
    ordered_transform_path_ids: tuple[str, ...]
    ordered_contract_semantics_ids: tuple[str, ...]
    transform_result_id: str
    wrapper_content_id: str
    base_bundle_content_id: str
    uncertainty_result_id: str
    transform_policy_id: str
    transform_semantics_id: str
    contract_commitment_id: str
    graph_commitment_id: str

    def __post_init__(self) -> None:
        if type(self.descriptor) is not DerivedObservationDescriptor:
            raise TypeError("inventory descriptor has wrong type")
        for name in (
            "ordered_component_receipt_ids",
            "source_observation_ids",
            "used_observation_compilation_ids",
            "ordered_transform_path_ids",
            "ordered_contract_semantics_ids",
        ):
            require_tuple(getattr(self, name), f"inventory {name}")
        # The authoritative inventory builder computes this root exactly once
        # before construction; recomputing it here for every downstream use
        # would make large vector inventories quadratic.
        if not self.observation_descriptor_id:
            raise ValueError("inventory descriptor root is missing")
        if self.support_slice.scale_id != self.descriptor.scale_id:
            raise ValueError("inventory support slice leaves descriptor scale")
        if (
            self.support_slice.temporal_support
            != self.descriptor.temporal_support
            or self.support_slice.spatial_support
            != self.descriptor.spatial_support
        ):
            raise ValueError("inventory support slice metadata drift")
        if len(self.ordered_component_receipt_ids) != len(
            self.descriptor.component_refs
        ):
            raise ValueError("inventory component receipt coverage mismatch")
        if self.source_observation_ids != self.descriptor.source_observation_ids:
            raise ValueError("inventory source observation lineage drift")
        if self.used_observation_compilation_ids != tuple(
            sorted(self.used_observation_compilation_ids)
        ):
            raise ValueError("inventory compilation lineage is not canonical")
        if len(self.ordered_transform_path_ids) != len(
            self.ordered_contract_semantics_ids
        ):
            raise ValueError("inventory transform path roots disagree")
        roots = (
            self.transform_result_id,
            self.wrapper_content_id,
            self.base_bundle_content_id,
            self.uncertainty_result_id,
            self.transform_policy_id,
            self.transform_semantics_id,
            self.contract_commitment_id,
            self.graph_commitment_id,
        )
        if not all(type(item) is str and item for item in roots):
            raise ValueError("inventory provenance roots are incomplete")

    @property
    def inventory_observation_id(self) -> str:
        return stable_hash(self, prefix="phase2b_derived_inventory_observation_")


@dataclass(frozen=True, slots=True)
class DerivedScaleInventory:
    wrapper_content_id: str
    transform_result_id: str
    task_target_id: str
    matcher_semantics_id: str
    observations: tuple[DerivedInventoryObservation, ...]
    inventory_id: str

    def __post_init__(self) -> None:
        require_tuple(self.observations, "derived inventory observations")
        if not self.observations:
            raise ValueError("derived inventory cannot be empty")
        if self.observations != tuple(
            sorted(
                self.observations,
                key=lambda item: item.observation_descriptor_id,
            )
        ):
            raise ValueError("derived inventory is not canonical")
        if len({item.observation_descriptor_id for item in self.observations}) != len(
            self.observations
        ):
            raise ValueError("derived inventory repeats an observation")
        if not all(
            type(item) is str and item
            for item in (
                self.wrapper_content_id,
                self.transform_result_id,
                self.task_target_id,
                self.matcher_semantics_id,
                self.inventory_id,
            )
        ):
            raise ValueError("derived inventory identity is incomplete")
        if any(
            item.wrapper_content_id != self.wrapper_content_id
            or item.transform_result_id != self.transform_result_id
            for item in self.observations
        ):
            raise ValueError("derived inventory observation root drift")

@dataclass(frozen=True, slots=True)
class DerivedWitnessMatch:
    inventory_observation_id: str
    observation_descriptor_id: str

    def __post_init__(self) -> None:
        if not self.inventory_observation_id or not self.observation_descriptor_id:
            raise ValueError("derived witness match identity is incomplete")

    @classmethod
    def from_inventory(
        cls,
        item: DerivedInventoryObservation,
        *,
        inventory_observation_id: str | None = None,
    ) -> "DerivedWitnessMatch":
        return cls(
            inventory_observation_id=(
                item.inventory_observation_id
                if inventory_observation_id is None
                else inventory_observation_id
            ),
            observation_descriptor_id=item.observation_descriptor_id,
        )


@dataclass(frozen=True, slots=True)
class DerivedObservableSlot:
    observable_id: str
    quantity_id: str
    expected_entity_ids: tuple[str, ...]
    expected_wire_role_ids: tuple[str, ...]
    support_slice_id: str
    matches: tuple[DerivedWitnessMatch, ...]
    slot_id: str

    def __post_init__(self) -> None:
        for name in ("expected_entity_ids", "expected_wire_role_ids", "matches"):
            require_tuple(getattr(self, name), f"derived observable slot {name}")
        if not self.observable_id or not self.quantity_id or not self.support_slice_id:
            raise ValueError("derived observable slot identity is incomplete")
        if type(self.slot_id) is not str or not self.slot_id:
            raise ValueError("derived observable slot commitment is missing")
        if self.matches != tuple(
            sorted(self.matches, key=lambda item: item.observation_descriptor_id)
        ):
            raise ValueError("derived observable matches are not canonical")

    @property
    def cardinality(self) -> "DerivedWitnessCardinality":
        if not self.matches:
            return DerivedWitnessCardinality.MISSING
        if len(self.matches) == 1:
            return DerivedWitnessCardinality.UNIQUE
        return DerivedWitnessCardinality.AMBIGUOUS


@dataclass(frozen=True, slots=True)
class DerivedCandidateHypothesis:
    law_id: str
    law_kind: LawKind
    family_id: str
    role_binding: tuple[tuple[str, str], ...]
    public_binding: tuple[RoleBinding, ...]
    support_slice: ExactSupportSlice
    required_observable_ids: tuple[str, ...]
    slots: tuple[DerivedObservableSlot, ...]
    task_target_id: str
    inventory_id: str
    registry_id: str
    matcher_semantics_id: str
    candidate_id: str
    footprint_id: str

    def __post_init__(self) -> None:
        for name in (
            "role_binding",
            "public_binding",
            "required_observable_ids",
            "slots",
        ):
            require_tuple(getattr(self, name), f"derived candidate {name}")
        if self.role_binding != tuple(sorted(self.role_binding)):
            raise ValueError("derived candidate role binding is not canonical")
        if self.public_binding != tuple(
            sorted(self.public_binding, key=lambda item: item.role_id)
        ):
            raise ValueError("derived public binding is not canonical")
        if self.slots != tuple(sorted(self.slots, key=lambda item: item.observable_id)):
            raise ValueError("derived candidate slots are not canonical")
        if tuple(item.observable_id for item in self.slots) != (
            self.required_observable_ids
        ):
            raise ValueError("derived candidate slot coverage mismatch")
        support_slice_id = self.support_slice.support_slice_id
        if any(
            item.support_slice_id != support_slice_id for item in self.slots
        ):
            raise ValueError("derived candidate slot support slice drift")
        identities = (
            self.law_id,
            self.family_id,
            self.task_target_id,
            self.inventory_id,
            self.registry_id,
            self.matcher_semantics_id,
            self.candidate_id,
            self.footprint_id,
        )
        if not all(type(item) is str and item for item in identities):
            raise ValueError("derived candidate identity is incomplete")

@dataclass(frozen=True, slots=True)
class DerivedCandidateGridCommitment:
    wrapper_content_id: str
    transform_result_id: str
    inventory_id: str
    task_target_id: str
    theory_version_id: str
    registry_id: str
    bridge_policy_id: str
    matcher_semantics_id: str
    candidates: tuple[DerivedCandidateHypothesis, ...]
    candidate_grid_commitment_id: str

    def __post_init__(self) -> None:
        require_tuple(self.candidates, "derived candidate grid")
        if not self.candidates:
            raise ValueError("derived candidate grid cannot be empty")
        if self.candidates != tuple(
            sorted(self.candidates, key=lambda item: item.candidate_id)
        ):
            raise ValueError("derived candidate grid is not canonical")
        if len({item.candidate_id for item in self.candidates}) != len(
            self.candidates
        ):
            raise ValueError("derived candidate grid repeats a candidate")
        if (
            type(self.candidate_grid_commitment_id) is not str
            or not self.candidate_grid_commitment_id
        ):
            raise ValueError("derived candidate grid commitment is missing")


class DerivedWitnessCardinality(str, Enum):
    MISSING = "missing"
    UNIQUE = "unique"
    AMBIGUOUS = "ambiguous"


def _slot_cardinality(slot: DerivedObservableSlot) -> DerivedWitnessCardinality:
    if not slot.matches:
        return DerivedWitnessCardinality.MISSING
    if len(slot.matches) == 1:
        return DerivedWitnessCardinality.UNIQUE
    return DerivedWitnessCardinality.AMBIGUOUS


@dataclass(frozen=True, slots=True)
class ExactDerivedCandidateEvaluation:
    candidate_id: str
    footprint_id: str
    law_id: str
    law_kind: LawKind
    violation_functional_id: str
    role_binding: tuple[tuple[str, str], ...]
    scale_id: str
    support_slice_id: str
    slot_ids: tuple[str, ...]
    used_inventory_observation_ids: tuple[str, ...]
    wrapper_content_id: str
    transform_result_id: str
    inventory_id: str
    candidate_grid_commitment_id: str
    task_target_id: str
    theory_version_id: str
    registry_id: str
    bridge_policy_id: str
    matcher_semantics_id: str
    verifier_semantics_id: str
    residual: ExactInterval | None
    tolerance: ExactInterval | None
    normalized: ExactInterval | None
    completed: bool
    error_code: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "role_binding",
            "slot_ids",
            "used_inventory_observation_ids",
        ):
            value = getattr(self, name)
            require_tuple(value, f"derived evaluation {name}")
            if value != tuple(sorted(value)):
                raise ValueError(f"derived evaluation {name} is not canonical")
        if type(self.completed) is not bool:
            raise TypeError("derived evaluation completed flag must be Boolean")
        if self.completed:
            if (
                self.error_code is not None
                or self.residual is None
                or self.tolerance is None
                or self.normalized is None
            ):
                raise ValueError("completed derived evaluation needs exact intervals")
            if self.residual.lower_fraction < 0:
                raise ValueError("derived residual cannot be negative")
            if self.tolerance.lower_fraction <= 0:
                raise ValueError("derived tolerance must be positive")
        elif (
            self.error_code is None
            or self.residual is not None
            or self.tolerance is not None
            or self.normalized is not None
        ):
            raise ValueError("failed derived evaluation needs only an error code")

    @property
    def status(self) -> ExactCandidateStatus:
        if self.normalized is None:
            return ExactCandidateStatus.ERROR
        if self.normalized.upper_fraction <= 1:
            return ExactCandidateStatus.PASS
        if self.normalized.lower_fraction > 1:
            return ExactCandidateStatus.FAIL
        return ExactCandidateStatus.INCONCLUSIVE

    @property
    def evaluation_id(self) -> str:
        return stable_hash(self, prefix="phase2b_exact_derived_evaluation_")


@dataclass(frozen=True, slots=True)
class ExactDerivedScaleAggregate:
    law_id: str
    law_kind: LawKind
    role_binding: tuple[tuple[str, str], ...]
    scale_id: str
    slice_evaluation_ids: tuple[str, ...]
    normalized_hull: ExactInterval | None
    error_code: str | None
    candidate_grid_commitment_id: str
    matcher_semantics_id: str

    def __post_init__(self) -> None:
        require_tuple(self.role_binding, "derived scale aggregate role binding")
        require_tuple(
            self.slice_evaluation_ids,
            "derived scale aggregate slice evaluations",
        )
        if not self.slice_evaluation_ids or self.slice_evaluation_ids != tuple(
            sorted(self.slice_evaluation_ids)
        ):
            raise ValueError("derived scale aggregate slice coverage is not canonical")
        if len(set(self.slice_evaluation_ids)) != len(self.slice_evaluation_ids):
            raise ValueError("derived scale aggregate repeats a slice evaluation")
        if self.role_binding != tuple(sorted(self.role_binding)):
            raise ValueError("derived scale aggregate role binding is not canonical")
        if (self.normalized_hull is None) is (self.error_code is None):
            raise ValueError("derived scale aggregate payload is ambiguous")

    @property
    def status(self) -> ExactCandidateStatus:
        if self.normalized_hull is None:
            return ExactCandidateStatus.ERROR
        if self.normalized_hull.upper_fraction <= 1:
            return ExactCandidateStatus.PASS
        if self.normalized_hull.lower_fraction > 1:
            return ExactCandidateStatus.FAIL
        return ExactCandidateStatus.INCONCLUSIVE

    @property
    def scale_aggregate_id(self) -> str:
        return stable_hash(self, prefix="phase2b_exact_derived_scale_aggregate_")


_AggregateGroupKey = tuple[
    str,
    LawKind,
    tuple[tuple[str, str], ...],
    str,
]
_AggregateReplayPayload = tuple[
    tuple[str, ...],
    ExactInterval | None,
    str | None,
]


def _linear_aggregate_replay_index(
    evaluations: tuple[ExactDerivedCandidateEvaluation, ...],
) -> tuple[
    dict[str, ExactDerivedCandidateEvaluation],
    dict[_AggregateGroupKey, _AggregateReplayPayload],
    int,
]:
    """Index and replay with two bounded scans plus canonical group sorting."""

    evaluations_by_id: dict[str, ExactDerivedCandidateEvaluation] = {}
    ids_by_group: dict[_AggregateGroupKey, list[str]] = {}
    work = 0
    for evaluation in evaluations:
        work += 1
        if work > _DEFAULT_POLICY.maximum_aggregate_replay_work:
            raise ValueError("RESOURCE_LIMIT:aggregate_replay_work")
        evaluation_id = evaluation.evaluation_id
        if evaluation_id in evaluations_by_id:
            raise ValueError("duplicate_candidate_evaluation")
        evaluations_by_id[evaluation_id] = evaluation
        ids_by_group.setdefault(
            (
                evaluation.law_id,
                evaluation.law_kind,
                evaluation.role_binding,
                evaluation.scale_id,
            ),
            [],
        ).append(evaluation_id)
    payloads: dict[_AggregateGroupKey, _AggregateReplayPayload] = {}
    for key, group_ids in ids_by_group.items():
        work += len(group_ids)
        if work > _DEFAULT_POLICY.maximum_aggregate_replay_work:
            raise ValueError("RESOURCE_LIMIT:aggregate_replay_work")
        expected_ids = tuple(sorted(group_ids))
        group_evaluations = tuple(
            evaluations_by_id[evaluation_id]
            for evaluation_id in expected_ids
        )
        errors = tuple(
            sorted(
                item.error_code
                for item in group_evaluations
                if item.error_code is not None
            )
        )
        if errors:
            expected_hull = None
            expected_error = "slice_evaluation_error:" + errors[0]
        else:
            normalized = tuple(
                item.normalized
                for item in group_evaluations
                if item.normalized is not None
            )
            if len(normalized) != len(group_evaluations):
                expected_hull = None
                expected_error = "slice_normalized_interval_missing"
            else:
                expected_hull = ExactInterval.from_fractions(
                    min(item.lower_fraction for item in normalized),
                    max(item.upper_fraction for item in normalized),
                )
                expected_error = None
        payloads[key] = (expected_ids, expected_hull, expected_error)
    return evaluations_by_id, payloads, work


@dataclass(frozen=True, slots=True)
class ExactDerivedBridgeCompilation:
    disposition: ExactBridgeDisposition
    reason: str
    wrapper_content_id: str
    transform_result_id: str
    inventory_id: str | None
    candidate_grid: DerivedCandidateGridCommitment | None
    candidate_grid_commitment_id: str | None
    scale_aggregate_commitment_id: str | None
    task_target_id: str
    theory_version_id: str
    registry_id: str
    bridge_policy_id: str
    matcher_semantics_id: str
    verifier_semantics_id: str
    evaluations: tuple[ExactDerivedCandidateEvaluation, ...]
    scale_aggregates: tuple[ExactDerivedScaleAggregate, ...]

    def __post_init__(self) -> None:
        require_tuple(self.evaluations, "derived bridge evaluations")
        require_tuple(self.scale_aggregates, "derived bridge scale aggregates")
        if type(self.disposition) is not ExactBridgeDisposition:
            raise TypeError("derived bridge disposition has wrong type")
        if self.disposition is ExactBridgeDisposition.COMPLETE:
            if (
                self.inventory_id is None
                or self.candidate_grid is None
                or self.candidate_grid_commitment_id is None
                or self.scale_aggregate_commitment_id is None
                or not self.evaluations
                or not self.scale_aggregates
            ):
                raise ValueError("complete derived bridge needs a full grid")
            if type(self.candidate_grid) is not DerivedCandidateGridCommitment:
                raise TypeError("derived bridge candidate grid has wrong type")
            if self.candidate_grid.inventory_id != self.inventory_id:
                raise ValueError("derived bridge inventory root drift")
            if (
                self.candidate_grid.wrapper_content_id != self.wrapper_content_id
                or self.candidate_grid.transform_result_id
                != self.transform_result_id
                or self.candidate_grid.task_target_id != self.task_target_id
                or self.candidate_grid.theory_version_id
                != self.theory_version_id
                or self.candidate_grid.registry_id != self.registry_id
                or self.candidate_grid.bridge_policy_id != self.bridge_policy_id
                or self.candidate_grid.matcher_semantics_id
                != self.matcher_semantics_id
            ):
                raise ValueError("derived bridge nested candidate grid root drift")
            if (
                self.candidate_grid.candidate_grid_commitment_id
                != self.candidate_grid_commitment_id
            ):
                raise ValueError("derived bridge candidate grid root drift")
            if self.evaluations != tuple(
                sorted(self.evaluations, key=lambda item: item.candidate_id)
            ):
                raise ValueError("derived bridge evaluations are not canonical")
            expected_candidate_ids = tuple(
                item.candidate_id for item in self.candidate_grid.candidates
            )
            if tuple(
                item.candidate_id for item in self.evaluations
            ) != expected_candidate_ids:
                raise ValueError("derived bridge candidate coverage mismatch")
            if any(
                item.wrapper_content_id != self.wrapper_content_id
                or item.transform_result_id != self.transform_result_id
                or item.inventory_id != self.inventory_id
                or item.candidate_grid_commitment_id
                != self.candidate_grid_commitment_id
                or item.task_target_id != self.task_target_id
                or item.theory_version_id != self.theory_version_id
                or item.registry_id != self.registry_id
                or item.bridge_policy_id != self.bridge_policy_id
                or item.matcher_semantics_id != self.matcher_semantics_id
                or item.verifier_semantics_id != self.verifier_semantics_id
                for item in self.evaluations
            ):
                raise ValueError("derived bridge evaluation provenance drift")
            if self.scale_aggregates != tuple(
                sorted(
                    self.scale_aggregates,
                    key=lambda item: item.scale_aggregate_id,
                )
            ):
                raise ValueError("derived bridge scale aggregates are not canonical")
            if any(
                item.candidate_grid_commitment_id
                != self.candidate_grid_commitment_id
                or item.matcher_semantics_id != self.matcher_semantics_id
                for item in self.scale_aggregates
            ):
                raise ValueError("derived bridge aggregate provenance drift")
            if _scale_aggregate_commitment_id(self.scale_aggregates) != (
                self.scale_aggregate_commitment_id
            ):
                raise ValueError("derived bridge scale aggregate root drift")
            evaluation_ids, expected_payload_by_group, _ = (
                _linear_aggregate_replay_index(self.evaluations)
            )
            covered_ids = tuple(
                evaluation_id
                for aggregate in self.scale_aggregates
                for evaluation_id in aggregate.slice_evaluation_ids
            )
            if (
                len(evaluation_ids) != len(self.evaluations)
                or len(covered_ids) != len(set(covered_ids))
                or set(covered_ids) != set(evaluation_ids)
            ):
                raise ValueError("derived bridge aggregate coverage mismatch")
            aggregate_keys = set()
            for aggregate in self.scale_aggregates:
                key = (
                    aggregate.law_id,
                    aggregate.law_kind,
                    aggregate.role_binding,
                    aggregate.scale_id,
                )
                if key in aggregate_keys:
                    raise ValueError("derived bridge repeats a scale aggregate")
                aggregate_keys.add(key)
                expected_payload = expected_payload_by_group.get(key)
                if expected_payload is None:
                    raise ValueError("derived bridge aggregate group drift")
                expected_ids, expected_hull, expected_error = expected_payload
                if (
                    aggregate.slice_evaluation_ids != expected_ids
                    or aggregate.normalized_hull != expected_hull
                    or aggregate.error_code != expected_error
                ):
                    raise ValueError("derived bridge aggregate payload drift")
            if aggregate_keys != set(expected_payload_by_group):
                raise ValueError("derived bridge aggregate key coverage drift")
        elif (
            self.inventory_id is not None
            or self.candidate_grid is not None
            or self.candidate_grid_commitment_id is not None
            or self.scale_aggregate_commitment_id is not None
            or self.evaluations
            or self.scale_aggregates
        ):
            raise ValueError("abstaining derived bridge cannot return a partial grid")

    @property
    def result_id(self) -> str:
        return stable_hash(self, prefix="phase2b_exact_derived_bridge_result_")


@dataclass(frozen=True, slots=True)
class ExactDerivedSelectorDecision:
    disposition: ExactSelectionDisposition
    reason: str
    bridge_result_id: str
    candidate_grid_commitment_id: str | None
    scale_aggregate_commitment_id: str | None
    selection_policy_id: str
    evaluated_candidate_ids: tuple[str, ...]
    consumed_scale_aggregate_ids: tuple[str, ...]
    selected_law_kind: LawKind | None = None
    selected_role_binding: tuple[tuple[str, str], ...] = ()
    admissible_scale_ids: tuple[str, ...] = ()
    normalized_structural_margin: ExactFractionAtom | None = None

    def __post_init__(self) -> None:
        for name in (
            "evaluated_candidate_ids",
            "consumed_scale_aggregate_ids",
            "selected_role_binding",
            "admissible_scale_ids",
        ):
            require_tuple(getattr(self, name), f"derived decision {name}")
        if type(self.disposition) is not ExactSelectionDisposition:
            raise TypeError("derived selector disposition has wrong type")
        if self.evaluated_candidate_ids != tuple(
            sorted(self.evaluated_candidate_ids)
        ):
            raise ValueError("derived selector candidate IDs are not canonical")
        if self.consumed_scale_aggregate_ids != tuple(
            sorted(self.consumed_scale_aggregate_ids)
        ):
            raise ValueError("derived selector aggregate IDs are not canonical")
        if self.selected_role_binding != tuple(sorted(self.selected_role_binding)):
            raise ValueError("derived selector role binding is not canonical")
        if self.admissible_scale_ids != tuple(sorted(set(self.admissible_scale_ids))):
            raise ValueError("derived selector scale IDs are not canonical")
        if (
            (self.candidate_grid_commitment_id is None)
            != (self.scale_aggregate_commitment_id is None)
        ):
            raise ValueError("derived selector commitment roots are partial")
        if self.candidate_grid_commitment_id is None and (
            self.evaluated_candidate_ids or self.consumed_scale_aggregate_ids
        ):
            raise ValueError("uncommitted derived selector cannot consume a grid")
        if self.disposition is ExactSelectionDisposition.ABSTAIN:
            if any(
                (
                    self.selected_law_kind is not None,
                    bool(self.selected_role_binding),
                    bool(self.admissible_scale_ids),
                    self.normalized_structural_margin is not None,
                )
            ):
                raise ValueError("derived abstention cannot carry selection")
        elif (
            self.candidate_grid_commitment_id is None
            or self.scale_aggregate_commitment_id is None
            or self.selected_law_kind is None
            or not self.selected_role_binding
            or not self.admissible_scale_ids
            or self.normalized_structural_margin is None
        ):
            raise ValueError("derived selector identification is incomplete")

    @property
    def decision_id(self) -> str:
        return stable_hash(self, prefix="phase2b_exact_derived_decision_")


@dataclass(frozen=True, slots=True)
class ExactDerivedBridgeRun:
    disposition: ExactBridgeDisposition
    reason: str
    wrapper_content_id: str
    transform_result_id: str
    task_target_id: str
    theory_version_id: str
    registry_id: str
    bridge_policy_id: str
    matcher_semantics_id: str
    transform_result: ExactTransformCompilation
    inventory: DerivedScaleInventory | None
    compilation: ExactDerivedBridgeCompilation
    decision: ExactDerivedSelectorDecision

    def __post_init__(self) -> None:
        if type(self.transform_result) is not ExactTransformCompilation:
            raise TypeError("derived run transform result has wrong type")
        if type(self.compilation) is not ExactDerivedBridgeCompilation:
            raise TypeError("derived run compilation has wrong type")
        if type(self.decision) is not ExactDerivedSelectorDecision:
            raise TypeError("derived run decision has wrong type")
        if self.transform_result.result_id != self.transform_result_id:
            raise ValueError("derived run transform result root drift")
        if (
            self.wrapper_content_id != self.compilation.wrapper_content_id
            or self.transform_result_id != self.compilation.transform_result_id
            or self.task_target_id != self.compilation.task_target_id
            or self.theory_version_id != self.compilation.theory_version_id
            or self.registry_id != self.compilation.registry_id
            or self.bridge_policy_id != self.compilation.bridge_policy_id
            or self.matcher_semantics_id != self.compilation.matcher_semantics_id
            or self.reason != self.compilation.reason
        ):
            raise ValueError("derived run root chain drift")
        if (
            self.wrapper_content_id != self.transform_result.wrapper_content_id
            and self.reason != "transform_authority_root_drift"
        ):
            raise ValueError("derived run transform authority root drift")
        if self.compilation.result_id != self.decision.bridge_result_id:
            raise ValueError("derived run selector bridge root drift")
        if (
            self.decision.candidate_grid_commitment_id
            != self.compilation.candidate_grid_commitment_id
            or self.decision.scale_aggregate_commitment_id
            != self.compilation.scale_aggregate_commitment_id
        ):
            raise ValueError("derived run selector grid root drift")
        if self.decision.selection_policy_id != _DEFAULT_SELECTION_POLICY_ID:
            raise ValueError("derived run selector policy drift")
        if self.decision.evaluated_candidate_ids != tuple(
            sorted(item.candidate_id for item in self.compilation.evaluations)
        ) or self.decision.consumed_scale_aggregate_ids != tuple(
            sorted(
                item.scale_aggregate_id
                for item in self.compilation.scale_aggregates
            )
        ):
            raise ValueError("derived run selector consumption drift")
        if self.disposition is not self.compilation.disposition:
            raise ValueError("derived run disposition drift")
        if self.disposition is ExactBridgeDisposition.ABSTAIN:
            if self.inventory is not None:
                raise ValueError("abstaining derived run cannot expose inventory")
        else:
            if type(self.inventory) is not DerivedScaleInventory:
                raise TypeError("complete derived run needs exact inventory")
            if (
                self.transform_result.disposition
                is not TransformCompilationDisposition.COMPLETE
                or self.inventory.wrapper_content_id != self.wrapper_content_id
                or self.inventory.transform_result_id != self.transform_result_id
                or self.inventory.task_target_id != self.task_target_id
                or self.inventory.matcher_semantics_id
                != self.matcher_semantics_id
                or self.inventory.inventory_id != self.compilation.inventory_id
            ):
                raise ValueError("derived run inventory root drift")

    @property
    def run_id(self) -> str:
        return stable_hash(self, prefix="phase2b_exact_derived_run_")


@dataclass(frozen=True, slots=True)
class ExactDerivedBridgePreflightRejection:
    disposition: ExactBridgeDisposition
    reason: str
    bundle_id: str
    wrapper_schema_version: str
    theory_schema_version: str
    registry_theory_version_id: str
    bridge_policy_id: str
    matcher_semantics_id: str
    wrapper_content_id: None = None
    transform_result: None = None
    inventory: None = None
    compilation: None = None
    decision: None = None

    def __post_init__(self) -> None:
        if self.disposition is not ExactBridgeDisposition.ABSTAIN:
            raise ValueError("derived preflight rejection must abstain")
        if any(
            item is not None
            for item in (
                self.wrapper_content_id,
                self.transform_result,
                self.inventory,
                self.compilation,
                self.decision,
            )
        ):
            raise ValueError("derived preflight rejection cannot carry receipts")


def _task_target_id(bundle: PublicEvidenceBundle) -> str:
    return stable_hash(bundle.task_target, prefix="phase2b_exact_task_target_")


def _strict_task_scope_error(
    bundle: PublicEvidenceBundle,
    registry: Phase2BAdapterRegistry,
) -> str | None:
    entity_ids = tuple(item.entity_id for item in bundle.entity_candidates)
    registry_quantities = tuple(
        sorted(item.quantity_id for item in registry.observable_channels)
    )
    registry_roles = {
        wire_role
        for binding in registry.law_bindings
        for _, wire_role in binding.role_ids
    }
    if set(bundle.task_target.entity_ids) != set(entity_ids):
        return "strict_task_entity_scope_mismatch"
    if (
        set(bundle.task_target.quantity_ids) != set(bundle.quantity_ids)
        or set(bundle.quantity_ids) != set(registry_quantities)
    ):
        return "strict_task_quantity_scope_mismatch"
    if set(bundle.role_ids) != registry_roles:
        return "strict_task_role_scope_mismatch"
    if any(
        not set(item.role_candidate_ids).issubset(registry_roles)
        or not item.role_candidate_ids
        for item in bundle.entity_candidates
    ):
        return "strict_task_entity_role_scope_mismatch"
    return None


def _preflight_rejection(
    reason: str,
    authority: PublicTransformEvidenceBundleV2,
    theory: TheoryState,
    registry: Phase2BAdapterRegistry,
) -> ExactDerivedBridgePreflightRejection:
    return ExactDerivedBridgePreflightRejection(
        disposition=ExactBridgeDisposition.ABSTAIN,
        reason=reason,
        bundle_id=authority.base_bundle.bundle_id,
        wrapper_schema_version=authority.schema_version,
        theory_schema_version=theory.schema_version,
        registry_theory_version_id=registry.theory_version_id,
        bridge_policy_id=_DEFAULT_POLICY_ID,
        matcher_semantics_id=_DEFAULT_MATCHER_SEMANTICS_ID,
    )


def _build_inventory(
    authority: PublicTransformEvidenceBundleV2,
    transform_result: ExactTransformCompilation,
    *,
    transform_result_id: str,
    task_target_id: str,
) -> tuple[DerivedScaleInventory | None, str | None]:
    policy = _DEFAULT_POLICY
    if len(transform_result.observations) > policy.maximum_inventory_observations:
        return None, "RESOURCE_LIMIT:inventory_observation_count"
    if len(transform_result.components) > policy.maximum_inventory_components:
        return None, "RESOURCE_LIMIT:inventory_component_count"
    descriptors: dict[str, DerivedObservationDescriptor] = {}
    for descriptor in transform_result.observations:
        descriptor_id = descriptor.descriptor_id
        if descriptor_id in descriptors:
            return None, "duplicate_inventory_observation_descriptor"
        descriptors[descriptor_id] = descriptor
    components: dict[str, list[ExactTransformedComponent]] = {
        descriptor_id: [] for descriptor_id in descriptors
    }
    for component in transform_result.components:
        bucket = components.get(component.observation_descriptor_id)
        if bucket is None:
            return None, "component_references_unknown_inventory_observation"
        bucket.append(component)
    task = authority.base_bundle.task_target
    inventory_items: list[DerivedInventoryObservation] = []
    consumed_component_receipts: set[str] = set()
    for descriptor_id, descriptor in descriptors.items():
        if (
            descriptor.temporal_support is None
            and descriptor.spatial_support is None
        ):
            return None, "explicit_support_required"
        if (
            not set(descriptor.entity_ids).issubset(task.entity_ids)
            or descriptor.quantity_id not in task.quantity_ids
        ):
            return None, "derived_observation_outside_task_scope"
        cells = tuple(
            sorted(
                components[descriptor_id],
                key=lambda item: item.descriptor.ref,
            )
        )
        if tuple(item.descriptor.ref for item in cells) != descriptor.component_refs:
            return None, "inventory_component_exact_coverage_mismatch"
        if any(item.value_kind is not descriptor.value_kind for item in cells):
            return None, "inventory_component_value_kind_drift"
        if not cells:
            return None, "inventory_observation_without_components"
        first = cells[0]
        if any(
            item.ordered_transform_path_ids
            != first.ordered_transform_path_ids
            or item.ordered_contract_semantics_ids
            != first.ordered_contract_semantics_ids
            or item.wrapper_content_id != transform_result.wrapper_content_id
            or item.base_bundle_content_id
            != transform_result.base_bundle_content_id
            or item.uncertainty_result_id
            != transform_result.uncertainty_result_id
            or item.transform_policy_id != transform_result.transform_policy_id
            or item.transform_semantics_id
            != transform_result.transform_semantics_id
            or item.contract_commitment_id
            != transform_result.contract_commitment_id
            or item.graph_commitment_id != transform_result.graph_commitment_id
            for item in cells
        ):
            return None, "inventory_component_provenance_root_drift"
        receipt_ids = tuple(item.component_receipt_id for item in cells)
        if any(item in consumed_component_receipts for item in receipt_ids):
            return None, "inventory_component_receipt_reused"
        consumed_component_receipts.update(receipt_ids)
        compilation_ids = tuple(
            sorted(
                {
                    compilation_id
                    for item in cells
                    for compilation_id in item.uncertainty_compilation_ids
                }
            )
        )
        inventory_items.append(
            DerivedInventoryObservation(
                descriptor=descriptor,
                observation_descriptor_id=descriptor_id,
                support_slice=ExactSupportSlice(
                    descriptor.scale_id,
                    descriptor.temporal_support,
                    descriptor.spatial_support,
                ),
                ordered_component_receipt_ids=receipt_ids,
                source_observation_ids=descriptor.source_observation_ids,
                used_observation_compilation_ids=compilation_ids,
                ordered_transform_path_ids=first.ordered_transform_path_ids,
                ordered_contract_semantics_ids=(
                    first.ordered_contract_semantics_ids
                ),
                transform_result_id=transform_result_id,
                wrapper_content_id=transform_result.wrapper_content_id,
                base_bundle_content_id=transform_result.base_bundle_content_id,
                uncertainty_result_id=transform_result.uncertainty_result_id,
                transform_policy_id=transform_result.transform_policy_id,
                transform_semantics_id=transform_result.transform_semantics_id,
                contract_commitment_id=transform_result.contract_commitment_id,
                graph_commitment_id=transform_result.graph_commitment_id,
            )
        )
    if len(consumed_component_receipts) != len(transform_result.components):
        return None, "inventory_component_receipt_coverage_mismatch"
    inventory_quantities = {
        item.descriptor.quantity_id for item in inventory_items
    }
    inventory_entities = {
        entity_id
        for item in inventory_items
        for entity_id in item.descriptor.entity_ids
    }
    if inventory_quantities != set(task.quantity_ids):
        return None, "inventory_quantity_scope_mismatch"
    if inventory_entities != set(task.entity_ids):
        return None, "inventory_entity_scope_mismatch"
    canonical_inventory = tuple(
        sorted(
            inventory_items,
            key=lambda item: item.observation_descriptor_id,
        )
    )
    inventory_id = stable_hash(
        {
            "wrapper_content_id": transform_result.wrapper_content_id,
            "transform_result_id": transform_result_id,
            "task_target_id": task_target_id,
            "matcher_semantics_id": _DEFAULT_MATCHER_SEMANTICS_ID,
            "observations": canonical_inventory,
        },
        prefix="phase2b_exact_derived_inventory_",
    )
    return (
        DerivedScaleInventory(
            wrapper_content_id=transform_result.wrapper_content_id,
            transform_result_id=transform_result_id,
            task_target_id=task_target_id,
            matcher_semantics_id=_DEFAULT_MATCHER_SEMANTICS_ID,
            observations=canonical_inventory,
            inventory_id=inventory_id,
        ),
        None,
    )


def _law_bindings(
    binding: LawWireBinding,
    bundle: PublicEvidenceBundle,
    *,
    slice_count: int,
    candidate_limit: int,
) -> tuple[tuple[tuple[str, str, str], ...], ...] | str:
    role_options: list[tuple[str, str, tuple[str, ...]]] = []
    raw_product = 1
    for semantic_role, wire_role in binding.role_ids:
        options = tuple(
            entity.entity_id
            for entity in bundle.entity_candidates
            if wire_role in entity.role_candidate_ids
        )
        if not options:
            return "incomplete_role_candidate_coverage"
        if raw_product > candidate_limit // len(options):
            return "RESOURCE_LIMIT:raw_role_binding_product"
        raw_product *= len(options)
        role_options.append((semantic_role, wire_role, options))
    if slice_count and raw_product > (
        candidate_limit // slice_count
    ):
        return "RESOURCE_LIMIT:role_binding_slice_product"
    rows = []
    for chosen in product(*(item[2] for item in role_options)):
        if len(chosen) != len(set(chosen)):
            continue
        rows.append(
            tuple(
                (semantic_role, wire_role, entity_id)
                for (semantic_role, wire_role, _), entity_id in zip(
                    role_options,
                    chosen,
                    strict=True,
                )
            )
        )
    if not rows:
        return "no_injective_role_binding"
    return tuple(rows)


def _candidate_grid(
    authority: PublicTransformEvidenceBundleV2,
    theory: TheoryState,
    registry: Phase2BAdapterRegistry,
    inventory: DerivedScaleInventory,
    *,
    theory_version_id: str,
    registry_id: str,
) -> tuple[DerivedCandidateGridCommitment | None, str | None]:
    support_slice_ids: dict[ExactSupportSlice, str] = {}
    inventory_observation_ids: dict[str, str] = {}
    for item in inventory.observations:
        if item.support_slice not in support_slice_ids:
            support_slice_ids[item.support_slice] = (
                item.support_slice.support_slice_id
            )
        inventory_observation_ids[item.observation_descriptor_id] = (
            item.inventory_observation_id
        )
    slices_by_id = {
        support_slice_id: support_slice
        for support_slice, support_slice_id in support_slice_ids.items()
    }
    if len(slices_by_id) > _DEFAULT_POLICY.maximum_support_slice_count:
        return None, "RESOURCE_LIMIT:support_slice_count"
    slices = tuple(item for _, item in sorted(slices_by_id.items()))
    inventory_id = inventory.inventory_id
    candidate_limit = _DEFAULT_POLICY.maximum_candidate_count
    inventory_index: dict[
        tuple[str, str, str],
        list[DerivedInventoryObservation],
    ] = {}
    for item in inventory.observations:
        key = (
            item.descriptor.scale_id,
            support_slice_ids[item.support_slice],
            item.descriptor.quantity_id,
        )
        inventory_index.setdefault(key, []).append(item)
    laws = {law.law_id: law for law in theory.relation_laws}
    quantity_by_observable = {
        item.observable_id: item.quantity_id for item in registry.observable_channels
    }
    candidates: list[DerivedCandidateHypothesis] = []
    used_inventory_ids: set[str] = set()
    match_scan_work = 0
    total_slots = 0
    for binding in registry.law_bindings:
        law = laws[binding.law_id]
        bindings_or_error = _law_bindings(
            binding,
            authority.base_bundle,
            slice_count=len(slices),
            candidate_limit=candidate_limit,
        )
        if isinstance(bindings_or_error, str):
            return None, bindings_or_error
        projected = len(bindings_or_error) * len(slices)
        if len(candidates) > candidate_limit - projected:
            return None, "RESOURCE_LIMIT:candidate_count"
        required_count = len(binding.required_observable_ids)
        if required_count and projected > (
            _DEFAULT_POLICY.maximum_match_scan_work - total_slots
        ) // required_count:
            return None, "RESOURCE_LIMIT:total_slot_count"
        total_slots += projected * required_count
        for role_row in bindings_or_error:
            entities_by_role = {
                semantic_role: entity_id
                for semantic_role, _, entity_id in role_row
            }
            wire_roles_by_role = {
                semantic_role: wire_role
                for semantic_role, wire_role, _ in role_row
            }
            for support_slice in slices:
                slots: list[DerivedObservableSlot] = []
                for observable_id in binding.required_observable_ids:
                    witness_roles = tuple(
                        role
                        for role, observable_names in law.role_observable_requirements
                        if observable_id in observable_names
                    )
                    if not witness_roles:
                        witness_roles = law.roles
                    expected_entities = tuple(
                        sorted(entities_by_role[role] for role in witness_roles)
                    )
                    expected_wire_roles = tuple(
                        sorted(wire_roles_by_role[role] for role in witness_roles)
                    )
                    pool = inventory_index.get(
                        (
                            support_slice.scale_id,
                            support_slice_ids[support_slice],
                            quantity_by_observable[observable_id],
                        ),
                        [],
                    )
                    expected_wire_role_set = set(expected_wire_roles)
                    for item in pool:
                        match_scan_work += (
                            1
                            + len(item.descriptor.entity_ids)
                            + len(item.descriptor.role_candidate_ids)
                        )
                        if (
                            match_scan_work
                            > _DEFAULT_POLICY.maximum_match_scan_work
                        ):
                            return None, "RESOURCE_LIMIT:match_scan_work"
                    match_scan_work += 1
                    if match_scan_work > _DEFAULT_POLICY.maximum_match_scan_work:
                        return None, "RESOURCE_LIMIT:match_scan_work"
                    matched = tuple(
                        item
                        for item in pool
                        if item.descriptor.entity_ids == expected_entities
                        and expected_wire_role_set.issubset(
                            item.descriptor.role_candidate_ids
                        )
                    )
                    if len(matched) > _DEFAULT_POLICY.maximum_slot_match_count:
                        return None, "RESOURCE_LIMIT:slot_match_count"
                    matches = tuple(
                        sorted(
                            (
                                DerivedWitnessMatch.from_inventory(
                                    item,
                                    inventory_observation_id=(
                                        inventory_observation_ids[
                                            item.observation_descriptor_id
                                        ]
                                    ),
                                )
                                for item in matched
                            ),
                            key=lambda item: item.observation_descriptor_id,
                        )
                    )
                    if len(matched) == 1:
                        used_inventory_ids.add(
                            inventory_observation_ids[
                                matched[0].observation_descriptor_id
                            ]
                        )
                    support_slice_id = support_slice_ids[support_slice]
                    quantity_id = quantity_by_observable[observable_id]
                    slot_id = stable_hash(
                        {
                            "observable_id": observable_id,
                            "quantity_id": quantity_id,
                            "expected_entity_ids": expected_entities,
                            "expected_wire_role_ids": expected_wire_roles,
                            "support_slice_id": support_slice_id,
                            "matches": matches,
                        },
                        prefix="phase2b_exact_derived_slot_",
                    )
                    slots.append(
                        DerivedObservableSlot(
                            observable_id=observable_id,
                            quantity_id=quantity_id,
                            expected_entity_ids=expected_entities,
                            expected_wire_role_ids=expected_wire_roles,
                            support_slice_id=support_slice_id,
                            matches=matches,
                            slot_id=slot_id,
                        )
                    )
                role_binding = tuple(
                    sorted(
                        (semantic_role, entity_id)
                        for semantic_role, _, entity_id in role_row
                    )
                )
                public_binding = tuple(
                    sorted(
                        (
                            RoleBinding(wire_role, entity_id)
                            for _, wire_role, entity_id in role_row
                        ),
                        key=lambda item: item.role_id,
                    )
                )
                canonical_slots = tuple(
                    sorted(slots, key=lambda item: item.observable_id)
                )
                candidate_payload = {
                    "law_id": law.law_id,
                    "law_kind": law.kind,
                    "family_id": binding.family_id,
                    "role_binding": role_binding,
                    "public_binding": public_binding,
                    "support_slice": support_slice,
                    "required_observable_ids": binding.required_observable_ids,
                    "task_target_id": inventory.task_target_id,
                    "inventory_id": inventory_id,
                    "registry_id": registry_id,
                    "matcher_semantics_id": inventory.matcher_semantics_id,
                }
                candidate_id = stable_hash(
                    candidate_payload,
                    prefix="phase2b_exact_derived_candidate_",
                )
                footprint_id = stable_hash(
                    (candidate_id, tuple(item.slot_id for item in canonical_slots)),
                    prefix="phase2b_exact_derived_footprint_",
                )
                candidates.append(
                    DerivedCandidateHypothesis(
                        law_id=law.law_id,
                        law_kind=law.kind,
                        family_id=binding.family_id,
                        role_binding=role_binding,
                        public_binding=public_binding,
                        support_slice=support_slice,
                        required_observable_ids=binding.required_observable_ids,
                        slots=canonical_slots,
                        task_target_id=inventory.task_target_id,
                        inventory_id=inventory_id,
                        registry_id=registry_id,
                        matcher_semantics_id=inventory.matcher_semantics_id,
                        candidate_id=candidate_id,
                        footprint_id=footprint_id,
                    )
                )
    expected_inventory_ids = {
        inventory_observation_ids[item.observation_descriptor_id]
        for item in inventory.observations
    }
    if used_inventory_ids != expected_inventory_ids:
        return None, "unused_or_ambiguously_consumed_derived_observation"
    canonical = tuple(sorted(candidates, key=lambda item: item.candidate_id))
    if not canonical:
        return None, "empty_derived_candidate_grid"
    grid_payload = {
        "wrapper_content_id": inventory.wrapper_content_id,
        "transform_result_id": inventory.transform_result_id,
        "inventory_id": inventory_id,
        "task_target_id": inventory.task_target_id,
        "theory_version_id": theory_version_id,
        "registry_id": registry_id,
        "bridge_policy_id": _DEFAULT_POLICY_ID,
        "matcher_semantics_id": _DEFAULT_MATCHER_SEMANTICS_ID,
        "candidates": canonical,
    }
    grid_id = stable_hash(
        grid_payload,
        prefix="phase2b_exact_derived_grid_",
    )
    return (
        DerivedCandidateGridCommitment(
            wrapper_content_id=inventory.wrapper_content_id,
            transform_result_id=inventory.transform_result_id,
            inventory_id=inventory_id,
            task_target_id=inventory.task_target_id,
            theory_version_id=theory_version_id,
            registry_id=registry_id,
            bridge_policy_id=_DEFAULT_POLICY_ID,
            matcher_semantics_id=_DEFAULT_MATCHER_SEMANTICS_ID,
            candidates=canonical,
            candidate_grid_commitment_id=grid_id,
        ),
        None,
    )


def _evaluation_usage(
    candidate: DerivedCandidateHypothesis,
    slot_ids: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    matches = tuple(
        match for slot in candidate.slots for match in slot.matches
    )
    return {
        "slot_ids": slot_ids,
        "used_inventory_observation_ids": tuple(
            sorted({item.inventory_observation_id for item in matches})
        ),
    }


def _evaluation_common(
    candidate: DerivedCandidateHypothesis,
    law: RelationLaw,
    functional_id: str,
    grid: DerivedCandidateGridCommitment,
    grid_id: str,
    candidate_id: str,
) -> dict[str, object]:
    slot_ids = tuple(sorted(slot.slot_id for slot in candidate.slots))
    return {
        "candidate_id": candidate_id,
        "footprint_id": candidate.footprint_id,
        "law_id": candidate.law_id,
        "law_kind": candidate.law_kind,
        "violation_functional_id": functional_id,
        "role_binding": candidate.role_binding,
        "scale_id": candidate.support_slice.scale_id,
        "support_slice_id": candidate.slots[0].support_slice_id,
        **_evaluation_usage(candidate, slot_ids),
        "wrapper_content_id": grid.wrapper_content_id,
        "transform_result_id": grid.transform_result_id,
        "inventory_id": grid.inventory_id,
        "candidate_grid_commitment_id": grid_id,
        "task_target_id": grid.task_target_id,
        "theory_version_id": grid.theory_version_id,
        "registry_id": grid.registry_id,
        "bridge_policy_id": grid.bridge_policy_id,
        "matcher_semantics_id": grid.matcher_semantics_id,
        "verifier_semantics_id": _DEFAULT_POLICY.verifier_semantics_id,
    }


def _error_evaluation(
    common: dict[str, object],
    error_code: str,
) -> ExactDerivedCandidateEvaluation:
    return ExactDerivedCandidateEvaluation(
        **common,  # type: ignore[arg-type]
        residual=None,
        tolerance=None,
        normalized=None,
        completed=False,
        error_code=error_code,
    )


def _derived_observable(
    budget: _ArithmeticBudget,
    inventory_item: DerivedInventoryObservation,
    components_by_receipt: dict[str, ExactTransformedComponent],
    kind: ExactObservableKind,
) -> object | str:
    descriptor = inventory_item.descriptor
    if descriptor.si_exponents != (0,) * 7:
        return "nondimensionless_unit_semantics_not_implemented"
    cells = tuple(
        components_by_receipt[item]
        for item in inventory_item.ordered_component_receipt_ids
    )
    if descriptor.value_kind is ComponentValueKind.MISSING:
        return "missing_observation"
    if kind is ExactObservableKind.BOOLEAN:
        if (
            descriptor.value_kind is not ComponentValueKind.BOOLEAN
            or len(cells) != 1
            or type(cells[0].boolean_value) is not bool
        ):
            return "observable_shape_mismatch"
        return cells[0].boolean_value
    if descriptor.value_kind is not ComponentValueKind.NUMERIC_INTERVAL:
        return "observable_shape_mismatch"
    intervals = []
    for cell in cells:
        value = cell.numeric_interval
        if value is None:
            return "observable_shape_mismatch"
        intervals.append(
            budget.interval(value.lower_fraction, value.upper_fraction)
        )
    if kind is ExactObservableKind.SCALAR:
        return intervals[0] if len(intervals) == 1 else "observable_shape_mismatch"
    return tuple(intervals)


def _compile_candidate(
    candidate: DerivedCandidateHypothesis,
    *,
    grid: DerivedCandidateGridCommitment,
    theory: TheoryState,
    inventory_by_id: dict[str, DerivedInventoryObservation],
    components_by_receipt: dict[str, ExactTransformedComponent],
    arithmetic: _ArithmeticBudget,
    grid_id: str,
    candidate_id: str,
) -> ExactDerivedCandidateEvaluation:
    laws = {law.law_id: law for law in theory.relation_laws}
    law = laws[candidate.law_id]
    tolerance_or_error = _tolerance(arithmetic, theory, law)
    functional_id = (
        law.violation_functional_id
        if isinstance(tolerance_or_error, str)
        else tolerance_or_error[0]
    )
    common = _evaluation_common(
        candidate,
        law,
        functional_id,
        grid,
        grid_id,
        candidate_id,
    )
    if isinstance(tolerance_or_error, str):
        return _error_evaluation(common, tolerance_or_error)
    for slot in candidate.slots:
        if slot.cardinality is DerivedWitnessCardinality.MISSING:
            return _error_evaluation(
                common,
                "missing_observable_witness:" + slot.observable_id,
            )
        if slot.cardinality is DerivedWitnessCardinality.AMBIGUOUS:
            return _error_evaluation(
                common,
                "ambiguous_observable_witness:" + slot.observable_id,
            )
    kinds = _observable_registry()[law.kind]
    observables: dict[str, object] = {}
    for slot in candidate.slots:
        match = slot.matches[0]
        inventory_item = inventory_by_id[match.inventory_observation_id]
        if (
            inventory_item.support_slice != candidate.support_slice
        ):
            return _error_evaluation(common, "support_slice_witness_drift")
        value_or_error = _derived_observable(
            arithmetic,
            inventory_item,
            components_by_receipt,
            kinds[slot.observable_id],
        )
        if isinstance(value_or_error, str):
            return _error_evaluation(common, value_or_error)
        observables[slot.observable_id] = value_or_error
    try:
        residual_or_error = _EXACT_VERIFIERS[law.kind](arithmetic, observables)
    except _ResourceLimit:
        raise
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return _error_evaluation(common, "exact_verifier_internal_error")
    if isinstance(residual_or_error, str):
        return _error_evaluation(common, residual_or_error)
    tolerance = tolerance_or_error[1]
    normalized = _divide(arithmetic, residual_or_error, tolerance)
    return ExactDerivedCandidateEvaluation(
        **common,  # type: ignore[arg-type]
        residual=residual_or_error,
        tolerance=tolerance,
        normalized=normalized,
        completed=True,
    )


def _scale_aggregates(
    evaluations: tuple[ExactDerivedCandidateEvaluation, ...],
    grid: DerivedCandidateGridCommitment,
    grid_id: str,
) -> tuple[ExactDerivedScaleAggregate, ...]:
    groups: dict[
        tuple[str, LawKind, tuple[tuple[str, str], ...], str],
        list[ExactDerivedCandidateEvaluation],
    ] = {}
    for item in evaluations:
        groups.setdefault(
            (item.law_id, item.law_kind, item.role_binding, item.scale_id),
            [],
        ).append(item)
    result = []
    for (law_id, law_kind, role_binding, scale_id), items in groups.items():
        evaluation_ids = tuple(sorted(item.evaluation_id for item in items))
        errors = tuple(
            sorted(item.error_code for item in items if item.error_code is not None)
        )
        if errors:
            normalized_hull = None
            error_code = "slice_evaluation_error:" + errors[0]
        else:
            normalized_values = tuple(
                item.normalized for item in items if item.normalized is not None
            )
            if len(normalized_values) != len(items):
                normalized_hull = None
                error_code = "slice_normalized_interval_missing"
            else:
                normalized_hull = ExactInterval.from_fractions(
                    min(item.lower_fraction for item in normalized_values),
                    max(item.upper_fraction for item in normalized_values),
                )
                error_code = None
        result.append(
            ExactDerivedScaleAggregate(
                law_id=law_id,
                law_kind=law_kind,
                role_binding=role_binding,
                scale_id=scale_id,
                slice_evaluation_ids=evaluation_ids,
                normalized_hull=normalized_hull,
                error_code=error_code,
                candidate_grid_commitment_id=grid_id,
                matcher_semantics_id=grid.matcher_semantics_id,
            )
        )
    return tuple(sorted(result, key=lambda item: item.scale_aggregate_id))


def _compile_grid(
    grid: DerivedCandidateGridCommitment,
    *,
    theory: TheoryState,
    inventory: DerivedScaleInventory,
    transform_result: ExactTransformCompilation,
) -> tuple[
    tuple[ExactDerivedCandidateEvaluation, ...] | None,
    tuple[ExactDerivedScaleAggregate, ...] | None,
    str | None,
]:
    inventory_by_descriptor = {
        item.observation_descriptor_id: item for item in inventory.observations
    }
    inventory_by_id: dict[str, DerivedInventoryObservation] = {}
    for candidate in grid.candidates:
        for slot in candidate.slots:
            for match in slot.matches:
                inventory_by_id[match.inventory_observation_id] = (
                    inventory_by_descriptor[match.observation_descriptor_id]
                )
    components_by_ref = {
        item.descriptor.ref: item for item in transform_result.components
    }
    components_by_receipt: dict[str, ExactTransformedComponent] = {}
    for item in inventory.observations:
        for receipt_id, ref in zip(
            item.ordered_component_receipt_ids,
            item.descriptor.component_refs,
            strict=True,
        ):
            components_by_receipt[receipt_id] = components_by_ref[ref]
    arithmetic = _ArithmeticBudget(DEFAULT_EXACT_BRIDGE_POLICY)
    grid_id = grid.candidate_grid_commitment_id
    candidate_rows = tuple(
        (candidate, candidate.candidate_id) for candidate in grid.candidates
    )
    evaluations = []
    try:
        for candidate, candidate_id in candidate_rows:
            arithmetic.start_candidate()
            evaluations.append(
                _compile_candidate(
                    candidate,
                    grid=grid,
                    theory=theory,
                    inventory_by_id=inventory_by_id,
                    components_by_receipt=components_by_receipt,
                    arithmetic=arithmetic,
                    grid_id=grid_id,
                    candidate_id=candidate_id,
                )
            )
    except _ResourceLimit as exc:
        return None, None, str(exc)
    canonical = tuple(sorted(evaluations, key=lambda item: item.candidate_id))
    if tuple(item.candidate_id for item in canonical) != tuple(
        candidate_id for _, candidate_id in candidate_rows
    ):
        return None, None, "candidate_evaluation_grid_drift"
    aggregates = _scale_aggregates(canonical, grid, grid_id)
    return canonical, aggregates, None


def _scale_aggregate_commitment_id(
    aggregates: tuple[ExactDerivedScaleAggregate, ...],
) -> str:
    return stable_hash(
        aggregates,
        prefix="phase2b_exact_derived_scale_aggregate_grid_",
    )


def _abstaining_compilation(
    *,
    reason: str,
    wrapper_content_id: str,
    transform_result_id: str,
    task_target_id: str,
    theory_version_id: str,
    registry_id: str,
) -> ExactDerivedBridgeCompilation:
    return ExactDerivedBridgeCompilation(
        disposition=ExactBridgeDisposition.ABSTAIN,
        reason=reason,
        wrapper_content_id=wrapper_content_id,
        transform_result_id=transform_result_id,
        inventory_id=None,
        candidate_grid=None,
        candidate_grid_commitment_id=None,
        scale_aggregate_commitment_id=None,
        task_target_id=task_target_id,
        theory_version_id=theory_version_id,
        registry_id=registry_id,
        bridge_policy_id=_DEFAULT_POLICY_ID,
        matcher_semantics_id=_DEFAULT_MATCHER_SEMANTICS_ID,
        verifier_semantics_id=_DEFAULT_POLICY.verifier_semantics_id,
        evaluations=(),
        scale_aggregates=(),
    )


def _decision_payload_ids(
    compilation: ExactDerivedBridgeCompilation,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(sorted(item.candidate_id for item in compilation.evaluations)),
        tuple(
            sorted(
                item.scale_aggregate_id
                for item in compilation.scale_aggregates
            )
        ),
    )


def _abstaining_decision(
    reason: str,
    compilation: ExactDerivedBridgeCompilation,
    *,
    compilation_id: str,
    candidate_ids: tuple[str, ...],
    aggregate_ids: tuple[str, ...],
) -> ExactDerivedSelectorDecision:
    return ExactDerivedSelectorDecision(
        disposition=ExactSelectionDisposition.ABSTAIN,
        reason=reason,
        bridge_result_id=compilation_id,
        candidate_grid_commitment_id=(
            compilation.candidate_grid_commitment_id
        ),
        scale_aggregate_commitment_id=(
            compilation.scale_aggregate_commitment_id
        ),
        selection_policy_id=_DEFAULT_SELECTION_POLICY_ID,
        evaluated_candidate_ids=candidate_ids,
        consumed_scale_aggregate_ids=aggregate_ids,
    )


def _select_scale_aggregates(
    compilation: ExactDerivedBridgeCompilation,
    *,
    compilation_id: str,
) -> ExactDerivedSelectorDecision:
    candidate_ids, aggregate_ids = _decision_payload_ids(compilation)
    abstain = lambda reason: _abstaining_decision(
        reason,
        compilation,
        compilation_id=compilation_id,
        candidate_ids=candidate_ids,
        aggregate_ids=aggregate_ids,
    )
    if compilation.disposition is ExactBridgeDisposition.ABSTAIN:
        return abstain("bridge_" + compilation.reason)
    grid = compilation.candidate_grid
    if grid is None:
        return abstain("candidate_grid_missing")
    expected_candidate_ids = tuple(
        sorted(item.candidate_id for item in grid.candidates)
    )
    if candidate_ids != expected_candidate_ids:
        return abstain("incomplete_candidate_grid")
    candidates_by_id = {item.candidate_id: item for item in grid.candidates}
    for evaluation in compilation.evaluations:
        candidate = candidates_by_id.get(evaluation.candidate_id)
        if candidate is None or (
            evaluation.footprint_id != candidate.footprint_id
            or evaluation.law_id != candidate.law_id
            or evaluation.law_kind is not candidate.law_kind
            or evaluation.role_binding != candidate.role_binding
            or evaluation.scale_id != candidate.support_slice.scale_id
            or evaluation.support_slice_id
            != candidate.slots[0].support_slice_id
            or evaluation.slot_ids
            != tuple(sorted(item.slot_id for item in candidate.slots))
            or evaluation.used_inventory_observation_ids
            != tuple(
                sorted(
                    {
                        match.inventory_observation_id
                        for slot in candidate.slots
                        for match in slot.matches
                    }
                )
            )
            or evaluation.wrapper_content_id != grid.wrapper_content_id
            or evaluation.transform_result_id != grid.transform_result_id
            or evaluation.inventory_id != grid.inventory_id
            or evaluation.candidate_grid_commitment_id
            != compilation.candidate_grid_commitment_id
            or evaluation.task_target_id != grid.task_target_id
            or evaluation.theory_version_id != grid.theory_version_id
            or evaluation.registry_id != grid.registry_id
            or evaluation.bridge_policy_id != grid.bridge_policy_id
            or evaluation.matcher_semantics_id != grid.matcher_semantics_id
            or evaluation.verifier_semantics_id
            != compilation.verifier_semantics_id
        ):
            return abstain("candidate_evaluation_payload_drift")
    try:
        evaluations_by_id, expected_aggregate_payloads, _ = (
            _linear_aggregate_replay_index(compilation.evaluations)
        )
    except ValueError as exc:
        return abstain(str(exc))
    covered_ids = tuple(
        evaluation_id
        for aggregate in compilation.scale_aggregates
        for evaluation_id in aggregate.slice_evaluation_ids
    )
    if (
        len(covered_ids) != len(set(covered_ids))
        or set(covered_ids) != set(evaluations_by_id)
    ):
        return abstain("scale_aggregate_slice_coverage_mismatch")
    group_keys: set[
        tuple[str, LawKind, tuple[tuple[str, str], ...], str]
    ] = set()
    for aggregate in compilation.scale_aggregates:
        key = (
            aggregate.law_id,
            aggregate.law_kind,
            aggregate.role_binding,
            aggregate.scale_id,
        )
        if key in group_keys:
            return abstain("duplicate_scale_aggregate")
        group_keys.add(key)
        expected_payload = expected_aggregate_payloads.get(key)
        if expected_payload is None:
            return abstain("scale_aggregate_provenance_drift")
        expected_ids, expected_hull, expected_error = expected_payload
        if (
            aggregate.slice_evaluation_ids != expected_ids
            or aggregate.candidate_grid_commitment_id
            != compilation.candidate_grid_commitment_id
            or aggregate.matcher_semantics_id
            != compilation.matcher_semantics_id
        ):
            return abstain("scale_aggregate_provenance_drift")
        if (
            aggregate.normalized_hull != expected_hull
            or aggregate.error_code != expected_error
        ):
            return abstain("scale_aggregate_value_drift")
    if group_keys != set(expected_aggregate_payloads):
        return abstain("scale_aggregate_key_coverage_mismatch")
    if any(
        item.status is ExactCandidateStatus.ERROR
        for item in compilation.scale_aggregates
    ):
        return abstain("candidate_evaluation_error")
    if (
        DEFAULT_EXACT_SELECTION_POLICY.require_complete_family_coverage
        and {item.law_kind for item in compilation.scale_aggregates}
        != set(LawKind)
    ):
        return abstain("incomplete_family_coverage")
    groups: dict[
        tuple[LawKind, tuple[tuple[str, str], ...]],
        list[ExactDerivedScaleAggregate],
    ] = {}
    for item in compilation.scale_aggregates:
        groups.setdefault((item.law_kind, item.role_binding), []).append(item)
    passing_groups = tuple(
        key
        for key, items in groups.items()
        if any(item.status is ExactCandidateStatus.PASS for item in items)
    )
    if not passing_groups:
        return abstain(
            "nonidentifiable_interval_overlap"
            if any(
                item.status is ExactCandidateStatus.INCONCLUSIVE
                for item in compilation.scale_aggregates
            )
            else "no_passing_structure"
        )
    if len(passing_groups) != 1:
        return abstain("multiple_passing_structures")
    selected_key = passing_groups[0]
    selected_items = tuple(groups[selected_key])
    if any(
        item.status is ExactCandidateStatus.INCONCLUSIVE
        for item in selected_items
    ):
        return abstain("selected_structure_has_inconclusive_scale")
    scale_ids = {item.scale_id for item in selected_items}
    if (
        DEFAULT_EXACT_SELECTION_POLICY.require_scale_competitor
        and len(scale_ids) < 2
    ):
        return abstain("missing_scale_competitor")
    if (
        DEFAULT_EXACT_SELECTION_POLICY.require_binding_competitor
        and not any(
            item.law_kind is selected_key[0]
            and item.role_binding != selected_key[1]
            for item in compilation.scale_aggregates
        )
    ):
        return abstain("missing_binding_competitor")
    passing_scales = tuple(
        sorted(
            item.scale_id
            for item in selected_items
            if item.status is ExactCandidateStatus.PASS
        )
    )
    selected_upper = min(
        item.normalized_hull.upper_fraction
        for item in selected_items
        if item.status is ExactCandidateStatus.PASS
        and item.normalized_hull is not None
    )
    competitors = tuple(
        item
        for key, items in groups.items()
        if key != selected_key
        for item in items
    )
    if not competitors:
        return abstain("missing_structural_competitor")
    if any(
        item.status is ExactCandidateStatus.INCONCLUSIVE
        for item in competitors
    ):
        return abstain("inconclusive_structural_competitor")
    competitor_lower = min(
        item.normalized_hull.lower_fraction
        for item in competitors
        if item.normalized_hull is not None
    )
    margin = competitor_lower - selected_upper
    if (
        margin.numerator.bit_length()
        > DEFAULT_EXACT_BRIDGE_POLICY.maximum_fraction_bit_length
        or margin.denominator.bit_length()
        > DEFAULT_EXACT_BRIDGE_POLICY.maximum_fraction_bit_length
    ):
        return abstain("RESOURCE_LIMIT:selection_margin_bit_length")
    if margin < DEFAULT_EXACT_SELECTION_POLICY.minimum_structural_margin.as_fraction():
        return abstain("insufficient_structural_margin")
    return ExactDerivedSelectorDecision(
        disposition=(
            ExactSelectionDisposition.UNIQUE_IDENTIFICATION
            if len(passing_scales) == 1
            else ExactSelectionDisposition.ADMISSIBLE_SCALE_SET
        ),
        reason="unique_structure_with_exact_derived_admissible_scales",
        bridge_result_id=compilation_id,
        candidate_grid_commitment_id=(
            compilation.candidate_grid_commitment_id
        ),
        scale_aggregate_commitment_id=(
            compilation.scale_aggregate_commitment_id
        ),
        selection_policy_id=_DEFAULT_SELECTION_POLICY_ID,
        evaluated_candidate_ids=candidate_ids,
        consumed_scale_aggregate_ids=aggregate_ids,
        selected_law_kind=selected_key[0],
        selected_role_binding=selected_key[1],
        admissible_scale_ids=passing_scales,
        normalized_structural_margin=ExactFractionAtom.from_fraction(margin),
    )


def _expected_semantic_paths(
    authority: PublicTransformEvidenceBundleV2,
    ordered_edges: tuple[object, ...],
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    semantics_ids = {
        contract.transform_id: contract.semantics_id
        for contract in authority.transform_contracts
    }
    paths: dict[str, tuple[str, ...]] = {
        scale_id: ()
        for scale_id in authority.base_bundle.aggregation_graph.root_scale_ids
    }
    for edge in ordered_edges:
        paths[edge.target_scale_id] = (
            *paths[edge.source_scale_id],
            semantics_ids[edge.transform_id],
        )
    return semantics_ids, paths


def _transform_receipt_error(
    *,
    authority: PublicTransformEvidenceBundleV2,
    transform_result: ExactTransformCompilation,
    expected_wrapper_content_id: str,
    expected_base_content_id: str,
    expected_uncertainty_receipt: object,
    expected_uncertainty_result_id: str,
    expected_graph_commitment_id: str,
    expected_contract_commitment_id: str,
    transform_paths: dict[str, tuple[str, ...]],
    semantic_paths: dict[str, tuple[str, ...]],
) -> str | None:
    if transform_result.wrapper_content_id != expected_wrapper_content_id:
        return "transform_authority_root_drift"
    if transform_result.base_bundle_content_id != expected_base_content_id:
        return "transform_base_bundle_content_root_drift"
    if (
        transform_result.uncertainty_result_id
        != expected_uncertainty_result_id
        or transform_result.uncertainty_receipt != expected_uncertainty_receipt
        or transform_result.uncertainty_policy_id
        != transform_result.uncertainty_receipt.compiler_policy_id
        or transform_result.phase2b_exact_freeze_id
        != transform_result.uncertainty_receipt.phase2b_exact_freeze_id
        or transform_result.rational_grid_id
        != transform_result.uncertainty_receipt.rational_grid_id
        or transform_result.transform_policy_id != _TRANSFORM_POLICY.policy_id
        or transform_result.transform_semantics_id != _TRANSFORM_POLICY.semantics_id
        or transform_result.graph_commitment_id
        != expected_graph_commitment_id
        or transform_result.contract_commitment_id
        != expected_contract_commitment_id
    ):
        return "transform_receipt_root_drift"
    receipt_compilation_by_observation = {
        item.observation_id: item.compilation_id
        for item in transform_result.uncertainty_receipt.observations
    }
    receipt_compilation_ids = set(receipt_compilation_by_observation.values())
    used_compilation_ids: set[str] = set()
    components_by_descriptor: dict[str, list[ExactTransformedComponent]] = {}
    for component in transform_result.components:
        scale_id = component.descriptor.ref.scale_id
        if (
            component.ordered_transform_path_ids
            != transform_paths.get(scale_id)
            or component.ordered_contract_semantics_ids
            != semantic_paths.get(scale_id)
        ):
            return "inventory_transform_path_drift"
        if not set(component.uncertainty_compilation_ids).issubset(
            receipt_compilation_ids
        ):
            return "inventory_uncertainty_lineage_drift"
        used_compilation_ids.update(component.uncertainty_compilation_ids)
        components_by_descriptor.setdefault(
            component.observation_descriptor_id,
            [],
        ).append(component)
    for descriptor in transform_result.observations:
        try:
            expected_ids = {
                receipt_compilation_by_observation[observation_id]
                for observation_id in descriptor.source_observation_ids
            }
        except KeyError:
            return "inventory_uncertainty_observation_lineage_drift"
        actual_ids = {
            compilation_id
            for component in components_by_descriptor.get(
                descriptor.descriptor_id,
                (),
            )
            for compilation_id in component.uncertainty_compilation_ids
        }
        if actual_ids != expected_ids:
            return "inventory_uncertainty_observation_lineage_drift"
    if (
        transform_result.disposition is TransformCompilationDisposition.COMPLETE
        and used_compilation_ids != receipt_compilation_ids
    ):
        return "inventory_uncertainty_lineage_coverage_mismatch"
    return None


def _committed_run(
    *,
    transform_result: ExactTransformCompilation,
    transform_result_id: str,
    inventory: DerivedScaleInventory | None,
    compilation: ExactDerivedBridgeCompilation,
    decision: ExactDerivedSelectorDecision,
    task_target_id: str,
    theory_version_id: str,
    registry_id: str,
) -> ExactDerivedBridgeRun:
    return ExactDerivedBridgeRun(
        disposition=compilation.disposition,
        reason=compilation.reason,
        wrapper_content_id=compilation.wrapper_content_id,
        transform_result_id=transform_result_id,
        task_target_id=task_target_id,
        theory_version_id=theory_version_id,
        registry_id=registry_id,
        bridge_policy_id=_DEFAULT_POLICY_ID,
        matcher_semantics_id=_DEFAULT_MATCHER_SEMANTICS_ID,
        transform_result=transform_result,
        inventory=inventory,
        compilation=compilation,
        decision=decision,
    )


def run_exact_derived_witness_bridge(
    *,
    authority: PublicTransformEvidenceBundleV2,
    theory: TheoryState,
    registry: Phase2BAdapterRegistry,
) -> ExactDerivedBridgeRun | ExactDerivedBridgePreflightRejection:
    """Rebuild transformed exact witnesses from the three raw authorities."""

    if type(authority) is not PublicTransformEvidenceBundleV2:
        raise TypeError("derived witness bridge requires exact v2 authority type")
    if type(theory) is not TheoryState:
        raise TypeError("derived witness bridge requires exact theory type")
    if type(registry) is not Phase2BAdapterRegistry:
        raise TypeError("derived witness bridge requires exact registry type")
    if type(authority.base_bundle) is not PublicEvidenceBundle:
        raise TypeError("derived witness bridge requires exact base bundle type")
    if any(
        type(value) is not str
        for value in (
            authority.schema_version,
            authority.base_bundle.bundle_id,
            theory.schema_version,
            registry.theory_version_id,
        )
    ) or type(registry.maximum_candidate_count) is not int:
        raise TypeError("derived witness preflight receipt scalars are not exact")
    if registry.maximum_candidate_count != _DEFAULT_POLICY.maximum_candidate_count:
        return _preflight_rejection(
            "registry_candidate_budget_drift",
            authority,
            theory,
            registry,
        )
    try:
        shallow_error = _transform_shallow_preflight(
            authority,
            _TRANSFORM_POLICY,
        )
        if shallow_error is None:
            shallow_error = _shallow_authority_size_preflight(
                authority.base_bundle,
                theory,
                registry,
                DEFAULT_EXACT_BRIDGE_POLICY,
            )
    except (AttributeError, TypeError, ValueError):
        raise TypeError("derived witness authority shallow schema is invalid")
    if shallow_error is not None:
        return _preflight_rejection(
            shallow_error,
            authority,
            theory,
            registry,
        )
    try:
        _TransformAuthorityTreeBudget(_TRANSFORM_POLICY).visit(authority)
        _require_exact_authority_tree(
            authority.base_bundle,
            theory,
            registry,
            DEFAULT_EXACT_BRIDGE_POLICY,
        )
    except _ResourceLimit as exc:
        return _preflight_rejection(str(exc), authority, theory, registry)
    except (AttributeError, TypeError, ValueError):
        return _preflight_rejection(
            "authority_exact_tree_validation_failed",
            authority,
            theory,
            registry,
        )
    for error in (
        _transform_detailed_preflight(authority, _TRANSFORM_POLICY),
        _transform_metadata_preflight(authority),
        _transform_contract_preflight(authority, _TRANSFORM_POLICY),
        _bounded_bundle_preflight(
            authority.base_bundle,
            registry,
            DEFAULT_EXACT_BRIDGE_POLICY,
        ),
        _strict_task_scope_error(authority.base_bundle, registry),
    ):
        if error is not None:
            return _preflight_rejection(
                error,
                authority,
                theory,
                registry,
            )
    forest_error, ordered_edges, transform_paths = _transform_forest_plan(
        authority,
        _TRANSFORM_POLICY,
    )
    if forest_error is not None:
        return _preflight_rejection(
            forest_error,
            authority,
            theory,
            registry,
        )
    theory_version_id = theory.version_id
    if registry.theory_version_id != theory_version_id:
        return _preflight_rejection(
            "theory_registry_version_mismatch",
            authority,
            theory,
            registry,
        )
    try:
        _validate_registry(theory, registry)
    except (KeyError, TypeError, ValueError):
        return _preflight_rejection(
            "registry_semantics_validation_failed",
            authority,
            theory,
            registry,
        )
    registry_id = registry.registry_id
    task_target_id = _task_target_id(authority.base_bundle)
    expected_wrapper_content_id = authority.content_id
    expected_base_content_id = authority.base_bundle.content_id
    expected_graph_commitment_id = _transform_graph_commitment(
        authority.base_bundle.aggregation_graph
    )
    semantics_ids, semantic_paths = _expected_semantic_paths(
        authority,
        ordered_edges,
    )
    expected_contract_commitment_id = _transform_contract_commitment(
        authority.transform_contracts,
        semantics_ids,
    )
    expected_uncertainty_receipt = compile_bundle_uncertainty(
        authority.base_bundle
    )
    expected_uncertainty_result_id = expected_uncertainty_receipt.result_id
    transform_result_or_rejection = run_exact_transform_semantics(authority)
    if type(transform_result_or_rejection) is ExactTransformPreflightRejection:
        return _preflight_rejection(
            "transform_" + transform_result_or_rejection.reason,
            authority,
            theory,
            registry,
        )
    if type(transform_result_or_rejection) is not ExactTransformCompilation:
        return _preflight_rejection(
            "transform_result_type_invalid",
            authority,
            theory,
            registry,
        )
    transform_result = transform_result_or_rejection
    if (
        len(transform_result.observations)
        > _DEFAULT_POLICY.maximum_inventory_observations
    ):
        return _preflight_rejection(
            "RESOURCE_LIMIT:inventory_observation_count",
            authority,
            theory,
            registry,
        )
    if (
        len(transform_result.components)
        > _DEFAULT_POLICY.maximum_inventory_components
    ):
        return _preflight_rejection(
            "RESOURCE_LIMIT:inventory_component_count",
            authority,
            theory,
            registry,
        )
    transform_result_id = transform_result.result_id
    transform_error = _transform_receipt_error(
        authority=authority,
        transform_result=transform_result,
        expected_wrapper_content_id=expected_wrapper_content_id,
        expected_base_content_id=expected_base_content_id,
        expected_uncertainty_receipt=expected_uncertainty_receipt,
        expected_uncertainty_result_id=expected_uncertainty_result_id,
        expected_graph_commitment_id=expected_graph_commitment_id,
        expected_contract_commitment_id=expected_contract_commitment_id,
        transform_paths=transform_paths,
        semantic_paths=semantic_paths,
    )
    if transform_error is not None:
        compilation = _abstaining_compilation(
            reason=transform_error,
            wrapper_content_id=expected_wrapper_content_id,
            transform_result_id=transform_result_id,
            task_target_id=task_target_id,
            theory_version_id=theory_version_id,
            registry_id=registry_id,
        )
        compilation_id = compilation.result_id
        decision = _select_scale_aggregates(
            compilation,
            compilation_id=compilation_id,
        )
        return _committed_run(
            transform_result=transform_result,
            transform_result_id=transform_result_id,
            inventory=None,
            compilation=compilation,
            decision=decision,
            task_target_id=task_target_id,
            theory_version_id=theory_version_id,
            registry_id=registry_id,
        )
    if transform_result.disposition is TransformCompilationDisposition.ABSTAIN:
        compilation = _abstaining_compilation(
            reason="transform_" + transform_result.reason,
            wrapper_content_id=transform_result.wrapper_content_id,
            transform_result_id=transform_result_id,
            task_target_id=task_target_id,
            theory_version_id=theory_version_id,
            registry_id=registry_id,
        )
        compilation_id = compilation.result_id
        decision = _select_scale_aggregates(
            compilation,
            compilation_id=compilation_id,
        )
        return _committed_run(
            transform_result=transform_result,
            transform_result_id=transform_result_id,
            inventory=None,
            compilation=compilation,
            decision=decision,
            task_target_id=task_target_id,
            theory_version_id=theory_version_id,
            registry_id=registry_id,
        )
    inventory, error = _build_inventory(
        authority,
        transform_result,
        transform_result_id=transform_result_id,
        task_target_id=task_target_id,
    )
    if error is None and inventory is not None:
        grid, error = _candidate_grid(
            authority,
            theory,
            registry,
            inventory,
            theory_version_id=theory_version_id,
            registry_id=registry_id,
        )
    else:
        grid = None
    if error is None and inventory is not None and grid is not None:
        evaluations, aggregates, error = _compile_grid(
            grid,
            theory=theory,
            inventory=inventory,
            transform_result=transform_result,
        )
    else:
        evaluations = None
        aggregates = None
    if (
        error is not None
        or inventory is None
        or grid is None
        or evaluations is None
        or aggregates is None
    ):
        compilation = _abstaining_compilation(
            reason=error or "derived_grid_compilation_failed",
            wrapper_content_id=transform_result.wrapper_content_id,
            transform_result_id=transform_result_id,
            task_target_id=task_target_id,
            theory_version_id=theory_version_id,
            registry_id=registry_id,
        )
        compilation_id = compilation.result_id
        decision = _select_scale_aggregates(
            compilation,
            compilation_id=compilation_id,
        )
        return _committed_run(
            transform_result=transform_result,
            transform_result_id=transform_result_id,
            inventory=None,
            compilation=compilation,
            decision=decision,
            task_target_id=task_target_id,
            theory_version_id=theory_version_id,
            registry_id=registry_id,
        )
    evaluations = tuple(sorted(evaluations, key=lambda item: item.candidate_id))
    aggregates = tuple(
        sorted(aggregates, key=lambda item: item.scale_aggregate_id)
    )
    grid_id = evaluations[0].candidate_grid_commitment_id
    aggregate_commitment_id = _scale_aggregate_commitment_id(aggregates)
    try:
        compilation = ExactDerivedBridgeCompilation(
            disposition=ExactBridgeDisposition.COMPLETE,
            reason="complete_exact_derived_witness_candidate_grid",
            wrapper_content_id=transform_result.wrapper_content_id,
            transform_result_id=transform_result_id,
            inventory_id=grid.inventory_id,
            candidate_grid=grid,
            candidate_grid_commitment_id=grid_id,
            scale_aggregate_commitment_id=aggregate_commitment_id,
            task_target_id=task_target_id,
            theory_version_id=theory_version_id,
            registry_id=registry_id,
            bridge_policy_id=_DEFAULT_POLICY_ID,
            matcher_semantics_id=_DEFAULT_MATCHER_SEMANTICS_ID,
            verifier_semantics_id=_DEFAULT_POLICY.verifier_semantics_id,
            evaluations=evaluations,
            scale_aggregates=aggregates,
        )
    except (TypeError, ValueError) as exc:
        reason = (
            "scale_aggregate_value_drift"
            if "aggregate payload drift" in str(exc)
            else "derived_compilation_integrity_drift"
        )
        compilation = _abstaining_compilation(
            reason=reason,
            wrapper_content_id=transform_result.wrapper_content_id,
            transform_result_id=transform_result_id,
            task_target_id=task_target_id,
            theory_version_id=theory_version_id,
            registry_id=registry_id,
        )
        compilation_id = compilation.result_id
        decision = _select_scale_aggregates(
            compilation,
            compilation_id=compilation_id,
        )
        return _committed_run(
            transform_result=transform_result,
            transform_result_id=transform_result_id,
            inventory=None,
            compilation=compilation,
            decision=decision,
            task_target_id=task_target_id,
            theory_version_id=theory_version_id,
            registry_id=registry_id,
        )
    compilation_id = compilation.result_id
    decision = _select_scale_aggregates(
        compilation,
        compilation_id=compilation_id,
    )
    return _committed_run(
        transform_result=transform_result,
        transform_result_id=transform_result_id,
        inventory=inventory,
        compilation=compilation,
        decision=decision,
        task_target_id=task_target_id,
        theory_version_id=theory_version_id,
        registry_id=registry_id,
    )


__all__ = ("run_exact_derived_witness_bridge",)
