"""Frozen public protocol and preregistered lifecycle model for Phase-2B.

This module contains no holdout generator, answer key, scorer feedback loop, or
recognizer implementation.  It freezes the public statistical contract and the
host-side state transitions that must occur around an externally enforced,
untrusted recognizer run.  An in-repository object cannot prove that an
independent custodian or an OS sandbox exists, so those remain explicit external
prerequisites and the generated report never claims a sealed qualification.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, fields, replace
from enum import Enum
import hashlib
from math import isfinite
from pathlib import Path
import re
from statistics import NormalDist
from threading import Lock
from typing import Final

from .hashing import stable_hash
from .milestones import PHASE2B, PHASE2B_FORMAL_CLAIM_NAME
from .phase2b_freeze_v1 import (
    PHASE2B_EXACT_FREEZE_VERSION,
    PreservationTransform,
    frozen_phase2b_exact_freeze,
)
from .schema import LawKind, require_tuple


PHASE2B_PROTOCOL_SCHEMA: Final = "hegel-machine-phase2b-preregistration/1"
PHASE2B_PROTOCOL_VERSION: Final = "phase2b_typed_evidence_preregistration_v1"
PHASE2B_REPORT_NAME: Final = "phase2b_preregistration_readiness_v1"
ONE_SIDED_CONFIDENCE: Final = 0.95
ONE_SIDED_Z_95: Final = NormalDist().inv_cdf(ONE_SIDED_CONFIDENCE)


class Phase2BCaseType(str, Enum):
    UNIQUE_SCALE_ANSWERABLE = "unique_scale_answerable"
    ADMISSIBLE_SCALE_SET_ANSWERABLE = "admissible_scale_set_answerable"
    WRONG_FAMILY_HARD_NEGATIVE = "wrong_family_hard_negative"
    BINDING_COUNTERFACTUAL = "binding_counterfactual"
    SCALE_COUNTERFACTUAL = "scale_counterfactual"
    SIGN_OR_INVARIANT_BREAK = "sign_or_invariant_break"
    INSUFFICIENT_OR_NONIDENTIFIABLE = "insufficient_or_nonidentifiable"


class MarginStratum(str, Enum):
    CLEAR_INTERIOR = "clear_interior"
    MODERATE = "moderate"
    NEAR_BOUNDARY_IDENTIFIABLE = "near_boundary_identifiable"
    NONUNIQUE_OR_INSUFFICIENT = "nonunique_or_insufficient"


class BaselineKind(str, Enum):
    EMBEDDING_NEAREST_PROTOTYPE = "embedding_nearest_prototype"
    FROZEN_LLM_SEMANTIC_ONLY = "frozen_llm_semantic_only"
    FLAT_LEARNED_TYPED = "flat_learned_typed"


class HoldoutLifecycle(str, Enum):
    PREREGISTERED = "preregistered"
    GENERATED_SEALED = "generated_sealed"
    PREDICTIONS_COMMITTED = "predictions_committed"
    CONSUMED = "consumed"
    INVALIDATED = "invalidated"


class PreservationMode(str, Enum):
    EXACT_INVARIANCE = "exact_invariance"
    APPROXIMATE_EQUIVARIANCE = "approximate_equivariance"
    INVALID_TRANSFORM_CONTROL = "invalid_transform_control"


@dataclass(frozen=True, slots=True)
class CaseAllocation:
    case_type: Phase2BCaseType
    per_family_scale_cell: int

    def __post_init__(self) -> None:
        if isinstance(self.per_family_scale_cell, bool) or (
            not isinstance(self.per_family_scale_cell, int)
        ):
            raise TypeError("case allocation must be an integer")
        if self.per_family_scale_cell <= 0:
            raise ValueError("case allocation must be positive")


@dataclass(frozen=True, slots=True)
class MarginAllocation:
    stratum: MarginStratum
    share_numerator: int
    share_denominator: int
    lower_inclusive: float | None
    upper_exclusive: float | None
    oracle_policy: str

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.share_numerator, self.share_denominator)
        ):
            raise TypeError("margin shares must use integer fractions")
        if self.share_numerator <= 0 or self.share_denominator <= 0:
            raise ValueError("margin shares must be positive")
        if self.share_numerator > self.share_denominator:
            raise ValueError("margin share cannot exceed one")
        for value in (self.lower_inclusive, self.upper_exclusive):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise ValueError("margin boundary must be finite")
        if (
            self.lower_inclusive is not None
            and self.upper_exclusive is not None
            and self.lower_inclusive >= self.upper_exclusive
        ):
            raise ValueError("margin stratum has an empty interval")
        if not self.oracle_policy:
            raise ValueError("margin stratum needs an oracle policy")


@dataclass(frozen=True, slots=True)
class BinaryGateThreshold:
    metric: str
    minimum_point_estimate: float
    minimum_one_sided_wilson_lcb: float | None
    scope: str = "overall"

    def __post_init__(self) -> None:
        if not self.metric or not self.scope:
            raise ValueError("metric and scope are required")
        for item in fields(self):
            if not item.name.startswith("minimum_"):
                continue
            value = getattr(self, item.name)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{item.name} must be a finite probability")


@dataclass(frozen=True, slots=True)
class ScalarUpperGate:
    metric: str
    maximum_point_estimate: float
    maximum_bootstrap_upper_bound: float

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("scalar gate metric is required")
        for value in (
            self.maximum_point_estimate,
            self.maximum_bootstrap_upper_bound,
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < 0
            ):
                raise ValueError("scalar upper gate must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class PreservationRequirement:
    transform: PreservationTransform
    applicability: str
    minimum_pairs_per_family_scale: int = 0
    minimum_pairs_per_family: int = 0
    mode: PreservationMode = PreservationMode.EXACT_INVARIANCE

    def __post_init__(self) -> None:
        if not isinstance(self.transform, PreservationTransform):
            raise TypeError("preservation transform must use PreservationTransform")
        if not self.applicability:
            raise ValueError("preservation transform and applicability are required")
        counts = (
            self.minimum_pairs_per_family_scale,
            self.minimum_pairs_per_family,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise TypeError("preservation pair counts must be integers")
        if any(value < 0 for value in counts) or not any(counts):
            raise ValueError("preservation needs one positive minimum pair count")


@dataclass(frozen=True, slots=True)
class IsolationProfile:
    profile_id: str
    read_only_input: bool
    read_only_root_filesystem: bool
    network_disabled: bool
    repository_not_mounted: bool
    generator_source_not_mounted: bool
    answer_manifest_not_mounted: bool
    ephemeral_working_filesystem: bool
    fixed_image_digest_required: bool
    environment_scrubbed: bool
    capabilities_dropped: bool
    no_new_privileges: bool
    resource_limits_required: bool
    output_schema_only: bool
    external_enforcement_required: bool = True

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("isolation profile ID is required")
        for item in fields(self):
            if item.name == "profile_id":
                continue
            if type(getattr(self, item.name)) is not bool:
                raise TypeError("isolation controls must be booleans")

    @property
    def missing_controls(self) -> tuple[str, ...]:
        return tuple(
            item.name
            for item in fields(self)
            if item.name != "profile_id" and not getattr(self, item.name)
        )

    @property
    def contract_complete(self) -> bool:
        return not self.missing_controls

    @property
    def proves_external_enforcement(self) -> bool:
        # A local declaration is not an OS/container attestation.
        return False


@dataclass(frozen=True, slots=True)
class BaselineRegistration:
    kind: BaselineKind
    baseline_spec_id: str
    implementation_id: str
    artifact_sha256: str
    frozen_before_holdout_generation: bool

    def __post_init__(self) -> None:
        if not isinstance(self.kind, BaselineKind):
            raise TypeError("baseline kind must use BaselineKind")
        if re.fullmatch(
            r"phase2b_baseline_spec_[0-9a-f]{64}",
            self.baseline_spec_id,
        ) is None:
            raise ValueError("baseline registration needs a BaselineSpec content ID")
        if not self.implementation_id:
            raise ValueError("baseline implementation ID is required")
        _require_sha256(self.artifact_sha256, "baseline artifact SHA-256")
        if type(self.frozen_before_holdout_generation) is not bool:
            raise TypeError("baseline frozen flag must be boolean")


@dataclass(frozen=True, slots=True)
class Phase2BProtocol:
    schema_version: str
    protocol_version: str
    exact_freeze_version: str
    milestone_id: str
    milestone_name: str
    formal_claim_name: str
    law_families: tuple[LawKind, ...]
    scale_cell_count: int
    case_allocations: tuple[CaseAllocation, ...]
    margin_allocations: tuple[MarginAllocation, ...]
    overall_gates: tuple[BinaryGateThreshold, ...]
    slice_gates: tuple[BinaryGateThreshold, ...]
    scale_regret_gate: ScalarUpperGate
    preservation_requirements: tuple[PreservationRequirement, ...]
    isolation_profile: IsolationProfile
    public_fixture_policy: str
    scale_hypothesis_generation_required: bool
    independent_custodian_required: bool
    separate_preservation_denominator: bool
    shadow_only: bool
    active_promotion_enabled: bool
    holdout_run_limit: int
    validation_attempts_per_version: int
    maximum_validation_versions_before_no_go: int
    unresolved_freeze_questions: tuple[str, ...]
    implementation_blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "law_families",
            "case_allocations",
            "margin_allocations",
            "overall_gates",
            "slice_gates",
            "preservation_requirements",
            "unresolved_freeze_questions",
            "implementation_blockers",
        ):
            require_tuple(getattr(self, name), f"Phase-2B {name}")
        if (self.schema_version, self.protocol_version) != (
            PHASE2B_PROTOCOL_SCHEMA,
            PHASE2B_PROTOCOL_VERSION,
        ):
            raise ValueError("Phase-2B protocol version drift")
        if self.exact_freeze_version != PHASE2B_EXACT_FREEZE_VERSION:
            raise ValueError("Phase-2B exact freeze version drift")
        if (self.milestone_id, self.milestone_name) != (
            PHASE2B.machine_id,
            PHASE2B.name,
        ):
            raise ValueError("Phase-2B milestone identity drift")
        if self.formal_claim_name != PHASE2B_FORMAL_CLAIM_NAME:
            raise ValueError("Phase-2B formal claim name drift")
        if tuple(self.law_families) != tuple(LawKind):
            raise ValueError("Phase-2B must freeze all six registered law families")
        if self.scale_cell_count != 2:
            raise ValueError("Phase-2B v1 requires two scale cells")
        if len({item.case_type for item in self.case_allocations}) != len(
            Phase2BCaseType
        ):
            raise ValueError("Phase-2B case allocation is incomplete or duplicated")
        if self.cases_per_family_scale_cell != 60:
            raise ValueError("Phase-2B requires 60 independent cases per cell")
        if self.independent_latent_case_count != 720:
            raise ValueError("Phase-2B requires exactly 720 independent latent cases")
        if len({item.stratum for item in self.margin_allocations}) != len(
            MarginStratum
        ):
            raise ValueError("Phase-2B margin allocation is incomplete or duplicated")
        numerator = sum(
            item.share_numerator
            * _least_common_share_denominator(self.margin_allocations)
            // item.share_denominator
            for item in self.margin_allocations
        )
        if numerator != _least_common_share_denominator(self.margin_allocations):
            raise ValueError("Phase-2B margin shares must sum exactly to one")
        if any(
            self.cases_per_family_scale_cell * item.share_numerator
            % item.share_denominator
            for item in self.margin_allocations
        ):
            raise ValueError("Phase-2B margin shares must yield integer per-cell counts")
        exact_freeze = frozen_phase2b_exact_freeze()
        if dict(self.case_type_totals) != dict(exact_freeze.case_type_totals):
            raise ValueError("Phase-2B case quotas drift from the exact freeze")
        if dict(self.margin_stratum_totals) != dict(
            exact_freeze.margin_stratum_totals
        ):
            raise ValueError("Phase-2B margin quotas drift from the exact freeze")
        expected_preservation_counts = tuple(
            (
                rule.transform,
                rule.legal_pairs_per_family_scale,
                rule.legal_pairs_per_family,
            )
            for rule in exact_freeze.preservation_rules
        )
        actual_preservation_counts = tuple(
            (
                requirement.transform,
                requirement.minimum_pairs_per_family_scale,
                requirement.minimum_pairs_per_family,
            )
            for requirement in self.preservation_requirements
        )
        if actual_preservation_counts != expected_preservation_counts:
            raise ValueError("Phase-2B preservation IDs or pair quotas drift")
        if not self.isolation_profile.contract_complete:
            raise ValueError("Phase-2B isolation profile omits a required control")
        for name in (
            "scale_hypothesis_generation_required",
            "independent_custodian_required",
            "separate_preservation_denominator",
            "shadow_only",
            "active_promotion_enabled",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be boolean")
        if not all(
            (
                self.scale_hypothesis_generation_required,
                self.independent_custodian_required,
                self.separate_preservation_denominator,
                self.shadow_only,
            )
        ):
            raise ValueError("Phase-2B v1 removed a mandatory claim boundary")
        if self.active_promotion_enabled:
            raise ValueError("Phase 2-3 must remain shadow-only")
        if self.holdout_run_limit != 1:
            raise ValueError("the sealed holdout is one-shot")
        if (
            self.validation_attempts_per_version,
            self.maximum_validation_versions_before_no_go,
        ) != (2, 2):
            raise ValueError("validation attempts and version limits drift")

    @property
    def cases_per_family_scale_cell(self) -> int:
        return sum(item.per_family_scale_cell for item in self.case_allocations)

    @property
    def independent_latent_case_count(self) -> int:
        return (
            len(self.law_families)
            * self.scale_cell_count
            * self.cases_per_family_scale_cell
        )

    @property
    def case_type_totals(self) -> tuple[tuple[str, int], ...]:
        multiplier = len(self.law_families) * self.scale_cell_count
        return tuple(
            (
                allocation.case_type.value,
                allocation.per_family_scale_cell * multiplier,
            )
            for allocation in self.case_allocations
        )

    @property
    def margin_stratum_totals(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (
                allocation.stratum.value,
                self.independent_latent_case_count
                * allocation.share_numerator
                // allocation.share_denominator,
            )
            for allocation in self.margin_allocations
        )

    @property
    def margin_stratum_per_cell(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (
                allocation.stratum.value,
                self.cases_per_family_scale_cell
                * allocation.share_numerator
                // allocation.share_denominator,
            )
            for allocation in self.margin_allocations
        )

    @property
    def ready_for_holdout_generation(self) -> bool:
        return not self.unresolved_freeze_questions and not self.implementation_blockers

    @property
    def protocol_id(self) -> str:
        return stable_hash(self, prefix="phase2b_protocol_")


def _least_common_share_denominator(
    allocations: tuple[MarginAllocation, ...],
) -> int:
    # The frozen shares use denominator 20.  Multiplication avoids importing a
    # mutable numerical stack and remains exact for this tiny registry.
    product = 1
    for allocation in allocations:
        product *= allocation.share_denominator
    for candidate in range(1, product + 1):
        if all(candidate % item.share_denominator == 0 for item in allocations):
            return candidate
    raise AssertionError("no common denominator")


def _require_sha256(value: str, name: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _require_prefixed_sha256(value: str, name: str) -> None:
    if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must use sha256:<digest>")


def frozen_phase2b_protocol() -> Phase2BProtocol:
    """Return the immutable public preregistration candidate.

    All normative parameter questions are frozen.  Unimplemented external
    prerequisites remain machine-visible so an external custodian cannot
    generate a formal holdout prematurely.
    """

    allocations = (
        CaseAllocation(Phase2BCaseType.UNIQUE_SCALE_ANSWERABLE, 19),
        CaseAllocation(Phase2BCaseType.ADMISSIBLE_SCALE_SET_ANSWERABLE, 1),
        CaseAllocation(Phase2BCaseType.WRONG_FAMILY_HARD_NEGATIVE, 8),
        CaseAllocation(Phase2BCaseType.BINDING_COUNTERFACTUAL, 8),
        CaseAllocation(Phase2BCaseType.SCALE_COUNTERFACTUAL, 8),
        CaseAllocation(Phase2BCaseType.SIGN_OR_INVARIANT_BREAK, 8),
        CaseAllocation(Phase2BCaseType.INSUFFICIENT_OR_NONIDENTIFIABLE, 8),
    )
    margins = (
        MarginAllocation(
            MarginStratum.CLEAR_INTERIOR,
            7,
            20,
            3.0,
            None,
            "unique_or_admissible_set",
        ),
        MarginAllocation(
            MarginStratum.MODERATE,
            3,
            10,
            1.0,
            3.0,
            "unique_or_admissible_set",
        ),
        MarginAllocation(
            MarginStratum.NEAR_BOUNDARY_IDENTIFIABLE,
            1,
            5,
            0.25,
            1.0,
            "identifiable_without_forced_top1",
        ),
        MarginAllocation(
            MarginStratum.NONUNIQUE_OR_INSUFFICIENT,
            3,
            20,
            None,
            0.25,
            "admissible_set_or_abstain_only",
        ),
    )
    overall = (
        BinaryGateThreshold("family_exact", 0.90, 0.86),
        BinaryGateThreshold("binding_exact", 0.90, 0.86),
        BinaryGateThreshold("scale_set_accuracy", 0.87, 0.82),
        BinaryGateThreshold("joint_exact", 0.85, 0.80),
        BinaryGateThreshold("hard_negative_rejection", 0.95, 0.90),
        BinaryGateThreshold("binding_counterfactual_rejection", 0.95, 0.90),
        BinaryGateThreshold("scale_counterfactual_rejection", 0.93, 0.88),
        BinaryGateThreshold("sign_or_invariant_break_rejection", 0.95, 0.90),
        BinaryGateThreshold("abstention_specificity", 0.95, 0.90),
        BinaryGateThreshold("fail_closed_rate", 1.0, None),
        BinaryGateThreshold("preservation_consistency", 0.97, 0.94),
        BinaryGateThreshold("nonidentifiable_scale_abstention", 0.95, 0.90),
    )
    slices = (
        BinaryGateThreshold("answerable_joint_exact", 0.80, 0.70, "family"),
        BinaryGateThreshold("all_control_rejection", 0.88, 0.78, "family"),
        BinaryGateThreshold("abstention_specificity", 0.85, 0.75, "family"),
        BinaryGateThreshold("answerable_joint_exact", 0.80, 0.70, "scale"),
        BinaryGateThreshold("all_control_rejection", 0.88, 0.78, "scale"),
        BinaryGateThreshold("abstention_specificity", 0.85, 0.75, "scale"),
    )
    preservation = (
        PreservationRequirement(
            PreservationTransform.ENTITY_ALPHA_RENAMING,
            "all",
            6,
        ),
        PreservationRequirement(
            PreservationTransform.OBSERVATION_REORDER,
            "order_invariant",
            6,
        ),
        PreservationRequirement(
            PreservationTransform.IRRELEVANT_ENTITY_AUGMENTATION,
            "scoped",
            6,
        ),
        PreservationRequirement(
            PreservationTransform.UNIT_CONVERSION,
            "numeric",
            8,
        ),
        PreservationRequirement(
            PreservationTransform.COORDINATE_AFFINE_TRANSFORM,
            "invariant_or_equivariant",
            8,
            mode=PreservationMode.APPROXIMATE_EQUIVARIANCE,
        ),
        PreservationRequirement(
            PreservationTransform.EQUIVALENT_AGGREGATION_SPLIT_MERGE,
            "conservation_additivity_coverage",
            8,
        ),
        PreservationRequirement(
            PreservationTransform.NONTRIVIAL_SCALE_MAP,
            "cross_scale_stable",
            minimum_pairs_per_family=10,
            mode=PreservationMode.APPROXIMATE_EQUIVARIANCE,
        ),
        PreservationRequirement(
            PreservationTransform.SIGN_CONVENTION_REPARAMETERIZATION,
            "direction_or_sign",
            6,
        ),
    )
    isolation = IsolationProfile(
        profile_id="phase2b_untrusted_recognizer_oci_v1",
        read_only_input=True,
        read_only_root_filesystem=True,
        network_disabled=True,
        repository_not_mounted=True,
        generator_source_not_mounted=True,
        answer_manifest_not_mounted=True,
        ephemeral_working_filesystem=True,
        fixed_image_digest_required=True,
        environment_scrubbed=True,
        capabilities_dropped=True,
        no_new_privileges=True,
        resource_limits_required=True,
        output_schema_only=True,
        external_enforcement_required=True,
    )
    return Phase2BProtocol(
        schema_version=PHASE2B_PROTOCOL_SCHEMA,
        protocol_version=PHASE2B_PROTOCOL_VERSION,
        exact_freeze_version=PHASE2B_EXACT_FREEZE_VERSION,
        milestone_id=PHASE2B.machine_id,
        milestone_name=PHASE2B.name,
        formal_claim_name=PHASE2B_FORMAL_CLAIM_NAME,
        law_families=tuple(LawKind),
        scale_cell_count=2,
        case_allocations=allocations,
        margin_allocations=margins,
        overall_gates=overall,
        slice_gates=slices,
        scale_regret_gate=ScalarUpperGate(
            "normalized_scale_decision_regret",
            maximum_point_estimate=0.05,
            maximum_bootstrap_upper_bound=0.08,
        ),
        preservation_requirements=preservation,
        isolation_profile=isolation,
        public_fixture_policy="phase2a_development_only_excluded_from_phase2b",
        scale_hypothesis_generation_required=True,
        independent_custodian_required=True,
        separate_preservation_denominator=True,
        shadow_only=True,
        active_promotion_enabled=False,
        holdout_run_limit=1,
        validation_attempts_per_version=2,
        maximum_validation_versions_before_no_go=2,
        unresolved_freeze_questions=(),
        implementation_blockers=(
            (
                "trusted_rfc8785_wire_builder_and_namespace_aware_formal_"
                "covert_auditor_not_implemented"
            ),
            (
                "formal_preservation_pair_generator_evaluator_and_complete_"
                "transform_to_verifier_coverage_not_implemented"
            ),
            "exact_baseline_revisions_and_artifact_hashes_not_registered",
            "independent_holdout_generator_and_validation_artifacts_not_implemented",
            (
                "functional_recognizer_cli_signed_minimal_image_and_formal_"
                "scoring_evaluator_not_implemented"
            ),
            "durable_signed_custodian_cas_ledger_not_implemented",
        ),
    )


@dataclass(frozen=True, slots=True)
class BinaryGateResult:
    metric: str
    successes: int
    total: int
    point_estimate: float
    one_sided_wilson_lcb: float
    passed: bool


def one_sided_wilson_lower_bound(
    successes: int,
    total: int,
    *,
    confidence: float = ONE_SIDED_CONFIDENCE,
) -> float:
    """Compute the preregistered one-sided Wilson lower confidence bound."""

    if any(isinstance(value, bool) or not isinstance(value, int) for value in (successes, total)):
        raise TypeError("Wilson counts must be integers")
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Wilson counts are outside their valid range")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not isfinite(confidence)
        or not 0.5 < confidence < 1.0
    ):
        raise ValueError("Wilson confidence must lie strictly between 0.5 and 1")
    proportion = successes / total
    z = NormalDist().inv_cdf(confidence)
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = proportion + z_squared / (2.0 * total)
    radius = z * (
        (proportion * (1.0 - proportion) / total)
        + z_squared / (4.0 * total * total)
    ) ** 0.5
    return max(0.0, (center - radius) / denominator)


def evaluate_binary_gate(
    threshold: BinaryGateThreshold,
    *,
    successes: int,
    total: int,
) -> BinaryGateResult:
    point = successes / total if total else 0.0
    lower = one_sided_wilson_lower_bound(successes, total)
    passed = point >= threshold.minimum_point_estimate and (
        threshold.minimum_one_sided_wilson_lcb is None
        or lower >= threshold.minimum_one_sided_wilson_lcb
    )
    return BinaryGateResult(
        metric=threshold.metric,
        successes=successes,
        total=total,
        point_estimate=point,
        one_sided_wilson_lcb=lower,
        passed=passed,
    )


@dataclass(frozen=True, slots=True)
class MeasurementUse:
    measurement_id: str
    witness_kind: str
    nonconstant: bool = True
    candidate_private: bool = False

    def __post_init__(self) -> None:
        if not self.measurement_id or not self.witness_kind:
            raise ValueError("measurement use needs identity and witness kind")
        if type(self.nonconstant) is not bool or type(self.candidate_private) is not bool:
            raise TypeError("measurement use flags must be booleans")


@dataclass(frozen=True, slots=True)
class CandidateFootprint:
    candidate_id: str
    measurements: tuple[MeasurementUse, ...]

    def __post_init__(self) -> None:
        require_tuple(self.measurements, "candidate measurements")
        if not self.candidate_id or not self.measurements:
            raise ValueError("candidate footprint cannot be empty")
        if len({item.measurement_id for item in self.measurements}) != len(
            self.measurements
        ):
            raise ValueError("candidate footprint repeats a measurement")


@dataclass(frozen=True, slots=True)
class SharedFootprintResult:
    shared_measurement_count: int
    correct_shared_fraction: float
    competitor_shared_fraction: float
    footprint_size_ratio: float
    shared_structural_witness: bool
    candidate_private_measurement_count: int
    passed: bool


def evaluate_shared_footprint(
    correct: CandidateFootprint,
    strongest_competitor: CandidateFootprint,
) -> SharedFootprintResult:
    """Apply the preregistered case-level shared-evidence gate."""

    correct_public = {
        item.measurement_id: item
        for item in correct.measurements
        if item.nonconstant and not item.candidate_private
    }
    competitor_public = {
        item.measurement_id: item
        for item in strongest_competitor.measurements
        if item.nonconstant and not item.candidate_private
    }
    shared_ids = {
        measurement_id
        for measurement_id in set(correct_public).intersection(competitor_public)
        if (
            correct_public[measurement_id].witness_kind
            == competitor_public[measurement_id].witness_kind
        )
    }
    correct_fraction = (
        len(shared_ids) / len(correct_public) if correct_public else 0.0
    )
    competitor_fraction = (
        len(shared_ids) / len(competitor_public) if competitor_public else 0.0
    )
    smaller = min(len(correct_public), len(competitor_public))
    larger = max(len(correct_public), len(competitor_public))
    ratio = larger / smaller if smaller else float("inf")
    shared_kinds = {
        correct_public[item_id].witness_kind for item_id in shared_ids
    }
    structural = bool(shared_kinds.intersection({"numeric", "order", "sign"}))
    private_count = sum(
        item.candidate_private
        for footprint in (correct, strongest_competitor)
        for item in footprint.measurements
    )
    passed = bool(
        len(shared_ids) >= 2
        and correct_fraction >= 0.60
        and competitor_fraction >= 0.60
        and ratio <= 3.0
        and structural
        and private_count == 0
    )
    return SharedFootprintResult(
        shared_measurement_count=len(shared_ids),
        correct_shared_fraction=correct_fraction,
        competitor_shared_fraction=competitor_fraction,
        footprint_size_ratio=ratio,
        shared_structural_witness=structural,
        candidate_private_measurement_count=private_count,
        passed=passed,
    )


@dataclass(frozen=True, slots=True)
class ExecutionFreezeManifest:
    protocol_id: str
    exact_freeze_id: str
    git_commit: str
    recognizer_image_digest: str
    configuration_sha256: str
    theory_version_id: str
    adapter_implementation_sha256: str
    selector_implementation_sha256: str
    verifier_registry_sha256: str
    baseline_registrations: tuple[BaselineRegistration, ...]
    isolation_profile_id: str

    def __post_init__(self) -> None:
        require_tuple(self.baseline_registrations, "baseline registrations")
        protocol = frozen_phase2b_protocol()
        exact_freeze = frozen_phase2b_exact_freeze()
        if self.protocol_id != protocol.protocol_id:
            raise ValueError("execution manifest does not bind the frozen protocol")
        if self.exact_freeze_id != exact_freeze.freeze_id:
            raise ValueError("execution manifest does not bind the exact freeze")
        if re.fullmatch(r"[0-9a-f]{40}", self.git_commit) is None:
            raise ValueError("execution manifest needs a full Git commit")
        _require_prefixed_sha256(
            self.recognizer_image_digest,
            "recognizer image digest",
        )
        for name in (
            "configuration_sha256",
            "adapter_implementation_sha256",
            "selector_implementation_sha256",
            "verifier_registry_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if not self.theory_version_id:
            raise ValueError("execution manifest needs a theory version")
        if self.isolation_profile_id != protocol.isolation_profile.profile_id:
            raise ValueError("execution manifest isolation profile drift")
        if (
            len(self.baseline_registrations) != len(BaselineKind)
            or {item.kind for item in self.baseline_registrations}
            != set(BaselineKind)
        ):
            raise ValueError("all three baseline classes must be registered")
        expected_baseline_spec_ids = {
            BaselineKind(spec.baseline_id): spec.content_id
            for spec in exact_freeze.baselines
        }
        for registration in self.baseline_registrations:
            if (
                registration.baseline_spec_id
                != expected_baseline_spec_ids[registration.kind]
            ):
                raise ValueError(
                    "baseline registration does not bind its frozen BaselineSpec"
                )
        if not all(
            item.frozen_before_holdout_generation
            for item in self.baseline_registrations
        ):
            raise ValueError("baselines must be frozen before holdout generation")

    @property
    def manifest_id(self) -> str:
        return stable_hash(self, prefix="phase2b_execution_freeze_")


_PROCESS_TRANSITION_LOCK = Lock()
_PROCESS_TRANSITIONED_LEDGER_IDS: set[str] = set()
_PROCESS_REGISTERED_RUN_KEYS: set[tuple[str, str, str]] = set()
_PROCESS_TRANSITION_AUTHORITY = object()


def salted_answer_commitment_sha256(
    revealed_answer_manifest_sha256: str,
    salt: str,
) -> str:
    """Commit to an answer-manifest digest with a reveal-time salt."""

    _require_sha256(
        revealed_answer_manifest_sha256,
        "revealed answer manifest SHA-256",
    )
    if not isinstance(salt, str) or len(salt.encode("utf-8")) < 32:
        raise ValueError("answer commitment salt must contain at least 32 bytes")
    return hashlib.sha256(
        (salt + ":" + revealed_answer_manifest_sha256).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class SealedRunLedger:
    """Content-addressed state record with a process-local fork guard.

    The local atomic guard prevents accidental or concurrent reuse inside one
    custodian process.  It is not durable across process restarts and therefore
    does not constitute the independent one-shot authority required for a
    formal sealed run.
    """
    run_id: str
    protocol_id: str
    freeze_manifest_id: str
    independent_custodian_id: str
    lifecycle: HoldoutLifecycle = HoldoutLifecycle.PREREGISTERED
    holdout_input_commitment_sha256: str | None = None
    salted_answer_commitment_sha256: str | None = None
    prediction_archive_sha256: str | None = None
    audit_archive_sha256: str | None = None
    revealed_answer_manifest_sha256: str | None = None
    score_report_sha256: str | None = None
    invalidation_reason: str | None = None
    prior_ledger_id: str | None = None
    _transition_authority: InitVar[object | None] = None

    def __post_init__(self, _transition_authority: object | None) -> None:
        if not all(
            (
                self.run_id,
                self.protocol_id,
                self.freeze_manifest_id,
                self.independent_custodian_id,
            )
        ):
            raise ValueError("sealed run ledger identity is incomplete")
        protocol = frozen_phase2b_protocol()
        if self.protocol_id != protocol.protocol_id:
            raise ValueError("sealed run ledger protocol drift")
        if re.fullmatch(
            r"phase2b_execution_freeze_[0-9a-f]{64}",
            self.freeze_manifest_id,
        ) is None:
            raise ValueError("sealed run ledger freeze manifest ID is malformed")
        run_key: tuple[str, str, str] | None = None
        if self.lifecycle is HoldoutLifecycle.PREREGISTERED:
            if _transition_authority is not None:
                raise ValueError("initial ledger cannot carry transition authority")
            if self.prior_ledger_id is not None:
                raise ValueError("initial ledger cannot name a predecessor")
            run_key = (self.run_id, self.protocol_id, self.freeze_manifest_id)
        else:
            if _transition_authority is not _PROCESS_TRANSITION_AUTHORITY:
                raise ValueError(
                    "noninitial ledger states require the transition authority"
                )
            if self.prior_ledger_id is None or re.fullmatch(
                r"phase2b_run_ledger_[0-9a-f]{64}",
                self.prior_ledger_id,
            ) is None:
                raise ValueError("transitioned ledger needs a valid predecessor ID")
        for item in fields(self):
            if not item.name.endswith("_sha256"):
                continue
            value = getattr(self, item.name)
            if value is not None:
                _require_sha256(value, item.name)
        required_by_state = {
            HoldoutLifecycle.PREREGISTERED: (),
            HoldoutLifecycle.GENERATED_SEALED: (
                "holdout_input_commitment_sha256",
                "salted_answer_commitment_sha256",
            ),
            HoldoutLifecycle.PREDICTIONS_COMMITTED: (
                "holdout_input_commitment_sha256",
                "salted_answer_commitment_sha256",
                "prediction_archive_sha256",
                "audit_archive_sha256",
            ),
            HoldoutLifecycle.CONSUMED: (
                "holdout_input_commitment_sha256",
                "salted_answer_commitment_sha256",
                "prediction_archive_sha256",
                "audit_archive_sha256",
                "revealed_answer_manifest_sha256",
                "score_report_sha256",
            ),
            HoldoutLifecycle.INVALIDATED: (),
        }
        if any(getattr(self, name) is None for name in required_by_state[self.lifecycle]):
            raise ValueError("sealed run ledger omits a required state commitment")
        allowed_by_state = {
            HoldoutLifecycle.PREREGISTERED: frozenset(),
            HoldoutLifecycle.GENERATED_SEALED: frozenset(
                {
                    "holdout_input_commitment_sha256",
                    "salted_answer_commitment_sha256",
                }
            ),
            HoldoutLifecycle.PREDICTIONS_COMMITTED: frozenset(
                {
                    "holdout_input_commitment_sha256",
                    "salted_answer_commitment_sha256",
                    "prediction_archive_sha256",
                    "audit_archive_sha256",
                }
            ),
            HoldoutLifecycle.CONSUMED: frozenset(
                item.name for item in fields(self) if item.name.endswith("_sha256")
            ),
            HoldoutLifecycle.INVALIDATED: frozenset(
                {
                    "holdout_input_commitment_sha256",
                    "salted_answer_commitment_sha256",
                    "prediction_archive_sha256",
                    "audit_archive_sha256",
                }
            ),
        }
        present_commitments = {
            item.name
            for item in fields(self)
            if item.name.endswith("_sha256") and getattr(self, item.name) is not None
        }
        if not present_commitments.issubset(allowed_by_state[self.lifecycle]):
            raise ValueError("sealed run ledger carries a future-state commitment")
        if self.lifecycle is HoldoutLifecycle.INVALIDATED and not self.invalidation_reason:
            raise ValueError("invalidated sealed run needs a reason")
        if self.lifecycle is not HoldoutLifecycle.INVALIDATED and self.invalidation_reason:
            raise ValueError("only an invalidated sealed run may carry a reason")
        if run_key is not None:
            with _PROCESS_TRANSITION_LOCK:
                if run_key in _PROCESS_REGISTERED_RUN_KEYS:
                    raise ValueError("sealed run key already registered in this process")
                _PROCESS_REGISTERED_RUN_KEYS.add(run_key)

    @property
    def ledger_id(self) -> str:
        return stable_hash(self, prefix="phase2b_run_ledger_")

    def record_generated_holdout(
        self,
        *,
        input_commitment_sha256: str,
        salted_answer_commitment_sha256: str,
    ) -> "SealedRunLedger":
        self._require_state(HoldoutLifecycle.PREREGISTERED)
        self._claim_process_local_transition()
        return replace(
            self,
            lifecycle=HoldoutLifecycle.GENERATED_SEALED,
            holdout_input_commitment_sha256=input_commitment_sha256,
            salted_answer_commitment_sha256=salted_answer_commitment_sha256,
            prior_ledger_id=self.ledger_id,
            _transition_authority=_PROCESS_TRANSITION_AUTHORITY,
        )

    def commit_predictions(
        self,
        *,
        prediction_archive_sha256: str,
        audit_archive_sha256: str,
    ) -> "SealedRunLedger":
        self._require_state(HoldoutLifecycle.GENERATED_SEALED)
        self._claim_process_local_transition()
        return replace(
            self,
            lifecycle=HoldoutLifecycle.PREDICTIONS_COMMITTED,
            prediction_archive_sha256=prediction_archive_sha256,
            audit_archive_sha256=audit_archive_sha256,
            prior_ledger_id=self.ledger_id,
            _transition_authority=_PROCESS_TRANSITION_AUTHORITY,
        )

    def consume(
        self,
        *,
        revealed_answer_manifest_sha256: str,
        score_report_sha256: str,
        answer_commitment_salt: str,
    ) -> "SealedRunLedger":
        self._require_state(HoldoutLifecycle.PREDICTIONS_COMMITTED)
        expected_commitment = salted_answer_commitment_sha256(
            revealed_answer_manifest_sha256,
            answer_commitment_salt,
        )
        if expected_commitment != self.salted_answer_commitment_sha256:
            raise ValueError("revealed answer manifest does not open the commitment")
        self._claim_process_local_transition()
        return replace(
            self,
            lifecycle=HoldoutLifecycle.CONSUMED,
            revealed_answer_manifest_sha256=revealed_answer_manifest_sha256,
            score_report_sha256=score_report_sha256,
            prior_ledger_id=self.ledger_id,
            _transition_authority=_PROCESS_TRANSITION_AUTHORITY,
        )

    def invalidate(self, reason: str) -> "SealedRunLedger":
        if self.lifecycle in {HoldoutLifecycle.CONSUMED, HoldoutLifecycle.INVALIDATED}:
            raise ValueError("a terminal sealed run cannot transition again")
        if not reason:
            raise ValueError("invalidation reason is required")
        self._claim_process_local_transition()
        return replace(
            self,
            lifecycle=HoldoutLifecycle.INVALIDATED,
            invalidation_reason=reason,
            prior_ledger_id=self.ledger_id,
            _transition_authority=_PROCESS_TRANSITION_AUTHORITY,
        )

    def _require_state(self, expected: HoldoutLifecycle) -> None:
        if self.lifecycle is not expected:
            raise ValueError(
                f"sealed run transition requires {expected.value}, "
                f"found {self.lifecycle.value}"
            )

    def _claim_process_local_transition(self) -> None:
        predecessor_id = self.ledger_id
        with _PROCESS_TRANSITION_LOCK:
            if predecessor_id in _PROCESS_TRANSITIONED_LEDGER_IDS:
                raise ValueError(
                    "sealed ledger predecessor already transitioned in this process"
                )
            _PROCESS_TRANSITIONED_LEDGER_IDS.add(predecessor_id)


def phase2b_preregistration_report() -> dict[str, object]:
    """Emit the checked-in, non-qualification preregistration artifact."""

    from . import (
        phase2b_adapter,
        phase2b_covert_audit_v1,
        phase2b_exact_derived_witness_bridge_v1,
        phase2b_exact_bridge_v1,
        phase2b_exact_transform_semantics_v1,
        phase2b_projection_compiler,
        phase2b_recognizer_prediction_archive_v1,
        phase2b_recognizer_prediction_archive_v2,
        phase2b_recognizer_prediction_v2,
        phase2b_runner,
        phase2b_selector,
        phase2b_strict_recognizer_cli_v2,
        phase2b_recognizer_input_archive_v1,
        phase2b_recognizer_input_archive_v2,
        phase2b_trusted_wire_batch_v1,
        phase2b_trusted_wire_batch_v2,
        phase2b_trusted_wire_typed_authority_v1,
        phase2b_trusted_wire_typed_authority_v2,
        phase2b_trusted_wire_typed_replay_v1,
        phase2b_trusted_wire_typed_replay_v2,
        phase2b_trusted_wire_v1,
        phase2b_uncertainty_compiler,
        phase2b_unsealed_prediction_evaluator_v1,
        phase2b_unsealed_prediction_evaluator_v2,
        phase2b_wire,
    )

    protocol = frozen_phase2b_protocol()
    exact_freeze = frozen_phase2b_exact_freeze()
    implementation_id = (
        "phase2b_protocol_source_sha256_"
        + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    )
    payload: dict[str, object] = {
        "artifact": PHASE2B_REPORT_NAME,
        "schema_version": protocol.schema_version,
        "protocol_version": protocol.protocol_version,
        "exact_freeze_version": protocol.exact_freeze_version,
        "exact_freeze_id": exact_freeze.freeze_id,
        "protocol_id": protocol.protocol_id,
        "implementation_id": implementation_id,
        "milestone_id": protocol.milestone_id,
        "milestone_name": protocol.milestone_name,
        "formal_claim_name_reserved_until_exit": protocol.formal_claim_name,
        "status": "exact_parameter_freeze_with_implementation_blockers",
        "normative_parameter_freeze_complete": True,
        "formal_holdout_generation_authorized": False,
        "formal_phase2b_exit_claim": False,
        "sealed_holdout_generated": False,
        "sealed_holdout_consumed": False,
        "process_local_ledger_fork_guard_implemented": True,
        "durable_external_one_shot_ledger_implemented": False,
        "answer_commitment_opening_validation_implemented": True,
        "independent_custodian_attested": False,
        "external_isolation_attested": False,
        "recognizer_image_built_from_allowlist": False,
        "recognizer_entrypoint_implemented": False,
        "formal_recognizer_run_runnable": False,
        "signed_sbom_validation_implemented": False,
        "runtime_attestation_signature_verifier_implemented": False,
        "unsealed_pipeline_validation_run": False,
        "typed_evidence_to_prediction_pipeline_complete": False,
        "projection_compiler_implemented": False,
        (
            "bounded_binary64_dimensionless_point_root_identity_"
            "projection_mechanics_implemented"
        ): True,
        "binary64_absolute_bound_envelope_mechanics_implemented": True,
        "formal_rational_grid_uncertainty_compiler_implemented": True,
        "absolute_bound_uncertainty_semantics_compiler_implemented": True,
        "standard_error_uncertainty_semantics_compiler_implemented": False,
        "bundle_atomic_exact_uncertainty_receipt_implemented": True,
        "exact_uncertainty_compiler_policy_id": (
            phase2b_uncertainty_compiler.DEFAULT_EXACT_UNCERTAINTY_POLICY.policy_id
        ),
        "formal_rational_grid_id": (
            phase2b_uncertainty_compiler.FROZEN_RATIONAL_GRID_ID
        ),
        "exact_uncertainty_receipt_consumed_by_projection_compiler": False,
        "exact_rational_residual_interval_semantics_implemented": False,
        (
            "root_identity_six_law_exact_rational_residual_interval_"
            "semantics_implemented"
        ): True,
        "exact_rational_selector_bridge_implemented": True,
        (
            "authoritative_exact_bridge_recomputes_uncertainty_and_"
            "adapter_internally"
        ): True,
        (
            "oversized_bundle_theory_or_registry_rejected_before_content_hash"
        ): True,
        "nested_authority_exact_type_enforced_before_content_hash": True,
        "exact_uncertainty_receipt_consumed_by_root_identity_bridge": True,
        "exact_bridge_policy_id": (
            phase2b_exact_bridge_v1.DEFAULT_EXACT_BRIDGE_POLICY.policy_id
        ),
        "exact_selector_policy_id": (
            phase2b_exact_bridge_v1.DEFAULT_EXACT_SELECTION_POLICY.policy_id
        ),
        "exact_verifier_semantics_id": (
            phase2b_exact_bridge_v1.DEFAULT_EXACT_BRIDGE_POLICY.verifier_semantics_id
        ),
        "public_transform_evidence_v2_authority_implemented": True,
        "public_transform_evidence_schema_version": (
            phase2b_exact_transform_semantics_v1.PUBLIC_TRANSFORM_EVIDENCE_SCHEMA_VERSION
        ),
        "exact_transform_semantics_version": (
            phase2b_exact_transform_semantics_v1.EXACT_TRANSFORM_SEMANTICS_VERSION
        ),
        "exact_transform_policy_id": (
            phase2b_exact_transform_semantics_v1.EXACT_TRANSFORM_POLICY_ID
        ),
        "eight_wire_transform_operation_exact_kernel_mechanics_implemented": True,
        "bundle_atomic_exact_transform_receipt_implemented": True,
        "exact_transform_recomputes_uncertainty_internally": True,
        "complete_transform_semantics_implemented": False,
        "formal_preservation_transform_suite_implemented": False,
        "exact_derived_observation_witness_bridge_implemented": True,
        "exact_derived_witness_bridge_version": (
            phase2b_exact_derived_witness_bridge_v1.EXACT_DERIVED_WITNESS_BRIDGE_VERSION
        ),
        "exact_derived_witness_matcher_version": (
            phase2b_exact_derived_witness_bridge_v1.EXACT_DERIVED_WITNESS_MATCHER_VERSION
        ),
        (
            "authoritative_derived_witness_bridge_recomputes_transform_"
            "internally"
        ): True,
        (
            "strict_scope_complete_law_binding_scale_support_slice_grid_"
            "implemented"
        ): True,
        "scale_selector_aggregates_exact_support_slices_before_selection": True,
        "exact_transform_receipt_consumed_by_derived_witness_bridge": True,
        "all_eight_transform_operations_covered_by_derived_six_law_bridge": False,
        "nondimensionless_derived_verifier_semantics_implemented": False,
        "prediction_archive_evaluator_implemented": False,
        "public_wire_contract_implemented": True,
        "public_wire_is_family_neutral_shaped_only": True,
        "semantic_family_neutrality_audited": False,
        "allowed_field_answer_correlation_audit_implemented": False,
        "schema_closed_accepted_jcs_profile_mechanics_implemented": True,
        "accepted_jcs_profile_id": phase2b_trusted_wire_v1.JCS_PROFILE_ID,
        "explicit_v2_uuid_namespace_path_manifest_mechanics_implemented": True,
        "uuid_namespace_path_manifest_id": (
            phase2b_trusted_wire_v1.FIELD_MANIFEST_ID
        ),
        "fixed_65536_public_padding_envelope_mechanics_implemented": True,
        "trusted_wire_profile_claim_level": (
            phase2b_trusted_wire_v1.NON_AUTHORITATIVE_CLAIM_LEVEL
        ),
        "trusted_wire_profile_transform_policy_id": (
            phase2b_exact_transform_semantics_v1.EXACT_TRANSFORM_POLICY_ID
        ),
        "keyed_trusted_wire_batch_mechanics_implemented": True,
        "trusted_wire_batch_policy_id": (
            phase2b_trusted_wire_batch_v1.TRUSTED_WIRE_BATCH_POLICY_ID
        ),
        "trusted_wire_key_schedule_version": (
            phase2b_trusted_wire_batch_v1.TRUSTED_WIRE_KEY_SCHEDULE_VERSION
        ),
        "trusted_wire_public_provenance_version": (
            phase2b_trusted_wire_batch_v1.TRUSTED_WIRE_PUBLIC_PROVENANCE_VERSION
        ),
        "trusted_wire_exact_transform_validator_policy_id": (
            phase2b_trusted_wire_batch_v1.EXACT_TRANSFORM_VALIDATOR_POLICY_ID
        ),
        "typed_authority_codec_version": (
            phase2b_trusted_wire_typed_authority_v1.TYPED_AUTHORITY_CODEC_VERSION
        ),
        "typed_authority_schema_id": (
            phase2b_trusted_wire_typed_authority_v1.TYPED_AUTHORITY_SCHEMA_ID
        ),
        "typed_authority_codec_policy_id": (
            phase2b_trusted_wire_typed_authority_v1.TYPED_AUTHORITY_CODEC_POLICY_ID
        ),
        "strict_closed_typed_authority_codec_mechanics_implemented": True,
        "exact_transform_provenance_compiler_version": (
            phase2b_exact_transform_semantics_v1.EXACT_TRANSFORM_PROVENANCE_COMPILER_VERSION
        ),
        "exact_transform_provenance_compiler_policy_id": (
            phase2b_exact_transform_semantics_v1.EXACT_TRANSFORM_PROVENANCE_COMPILER_POLICY_ID
        ),
        "native_v2_provenance_compile_before_framing_implemented": True,
        "typed_trusted_wire_replay_version": (
            phase2b_trusted_wire_typed_replay_v1.TYPED_TRUSTED_WIRE_REPLAY_VERSION
        ),
        "typed_trusted_wire_replay_policy_id": (
            phase2b_trusted_wire_typed_replay_v1.TYPED_TRUSTED_WIRE_REPLAY_POLICY_ID
        ),
        "direct_payload_authority_exact_transform_complete_replay_implemented": True,
        "whole_batch_atomic_typed_replay_mechanics_implemented": True,
        "source_order_bound_stage_b_secret_replay_receipt_implemented": True,
        "public_recognizer_registry_schema_version": (
            phase2b_recognizer_input_archive_v1.PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_VERSION
        ),
        "public_recognizer_registry_schema_id": (
            phase2b_recognizer_input_archive_v1.PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_ID
        ),
        "public_recognizer_family_alias_policy_id": (
            phase2b_recognizer_input_archive_v1.PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID
        ),
        "trusted_recognizer_input_archive_version": (
            phase2b_recognizer_input_archive_v1.TRUSTED_RECOGNIZER_INPUT_ARCHIVE_VERSION
        ),
        "recognizer_input_archive_policy_id": (
            phase2b_recognizer_input_archive_v1.RECOGNIZER_INPUT_ARCHIVE_POLICY_ID
        ),
        "recognizer_input_archive_claim_level": (
            phase2b_trusted_wire_v1.NON_AUTHORITATIVE_CLAIM_LEVEL
        ),
        "strict_public_recognizer_registry_codec_mechanics_implemented": True,
        "live_post_hmac_recognizer_registry_projection_mechanics_implemented": True,
        "registry_envelope_exact_scope_bijection_replay_implemented": True,
        "global_source_public_uuid_disjointness_gate_implemented": True,
        (
            "whole_batch_atomic_custodian_gated_recognizer_input_archive_"
            "issuer_mechanics_implemented"
        ): True,
        "public_recognizer_input_archive_structural_decode_replay_implemented": True,
        "recognizer_input_archive_success_is_false_claim_public_decode": True,
        "durable_trusted_recognizer_input_archive_receipt_implemented": False,
        "recognizer_input_archive_batch_policy_membership_verified": False,
        "recognizer_input_archive_source_registry_projection_verified": False,
        "recognizer_input_archive_secret_custodian_replay_verified": False,
        "recognizer_input_archive_origin_authenticated": False,
        "recognizer_input_archive_formal_covert_audit": False,
        "recognizer_input_archive_sealed_holdout_eligible": False,
        "recognizer_input_archive_recognizer_executed": False,
        "recognizer_input_archive_prediction_archive_evaluated": False,
        "recognizer_input_archive_c1_exit_evidence": False,
        "public_run_context_schema_version": (
            phase2b_recognizer_prediction_archive_v1.PUBLIC_RUN_CONTEXT_SCHEMA_VERSION
        ),
        "public_run_context_schema_id": (
            phase2b_recognizer_prediction_archive_v1.PUBLIC_RUN_CONTEXT_SCHEMA_ID
        ),
        "public_recognizer_prediction_record_schema_version": (
            phase2b_recognizer_prediction_archive_v1.PUBLIC_RECOGNIZER_PREDICTION_RECORD_SCHEMA_VERSION
        ),
        "public_recognizer_prediction_record_schema_id": (
            phase2b_recognizer_prediction_archive_v1.PUBLIC_RECOGNIZER_PREDICTION_RECORD_SCHEMA_ID
        ),
        "recognizer_prediction_archive_schema_version": (
            phase2b_recognizer_prediction_archive_v1.RECOGNIZER_PREDICTION_ARCHIVE_SCHEMA_VERSION
        ),
        "recognizer_prediction_archive_policy_id": (
            phase2b_recognizer_prediction_archive_v1.RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID
        ),
        "recognizer_prediction_archive_claim_level": (
            phase2b_trusted_wire_v1.NON_AUTHORITATIVE_CLAIM_LEVEL
        ),
        "public_run_context_structural_schema_mechanics_implemented": True,
        "closed_public_prediction_record_schema_mechanics_implemented": True,
        (
            "record_framed_exact_960_prediction_archive_structural_codec_"
            "mechanics_implemented"
        ): True,
        "internal_derived_to_prediction_mapping_gate_mechanics_implemented": True,
        "decoded_prediction_semantic_fields_exclude_split_gold_index_labels": True,
        "unsealed_prediction_evaluator_version": (
            phase2b_unsealed_prediction_evaluator_v1.UNSEALED_PREDICTION_EVALUATOR_VERSION
        ),
        "unsealed_prediction_evaluator_policy_id": (
            phase2b_unsealed_prediction_evaluator_v1.UNSEALED_PREDICTION_EVALUATOR_POLICY_ID
        ),
        (
            "unsealed_720_240_sorted_disjoint_exhaustive_structural_"
            "evaluator_implemented"
        ): True,
        "recognizer_runner_total_case_count": (
            phase2b_runner.TOTAL_RECOGNIZER_CASE_COUNT
        ),
        "recognizer_runner_total_960_contract_implemented": True,
        "minimum_constructed_positive_typed_profile_bytes": 125_582,
        "trusted_wire_maximum_payload_bytes": (
            phase2b_trusted_wire_v1.MAXIMUM_PAYLOAD_BYTES
        ),
        "real_positive_typed_profile_fits_trusted_wire": False,
        "compact_typed_authority_codec_v2_version": (
            phase2b_trusted_wire_typed_authority_v2.COMPACT_TYPED_AUTHORITY_CODEC_VERSION
        ),
        "compact_typed_authority_schema_id_v2": (
            phase2b_trusted_wire_typed_authority_v2.COMPACT_TYPED_AUTHORITY_SCHEMA_ID
        ),
        "compact_typed_authority_codec_policy_id_v2": (
            phase2b_trusted_wire_typed_authority_v2.COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID
        ),
        "lossless_compact_typed_authority_codec_v2_mechanics_implemented": True,
        "trusted_wire_batch_v2_schema_version": (
            phase2b_trusted_wire_batch_v2.TRUSTED_WIRE_BATCH_V2_SCHEMA_VERSION
        ),
        "trusted_wire_batch_v2_payload_schema_version": (
            phase2b_trusted_wire_batch_v2.TRUSTED_WIRE_BATCH_V2_PAYLOAD_SCHEMA_VERSION
        ),
        "trusted_wire_envelope_v2_version": (
            phase2b_trusted_wire_batch_v2.TRUSTED_WIRE_ENVELOPE_V2_VERSION
        ),
        "trusted_wire_envelope_v2_magic_hex": (
            phase2b_trusted_wire_batch_v2.TRUSTED_WIRE_ENVELOPE_V2_MAGIC.hex()
        ),
        "trusted_wire_batch_v2_policy_id": (
            phase2b_trusted_wire_batch_v2.TRUSTED_WIRE_BATCH_V2_POLICY_ID
        ),
        "compact_v2_fixed_65536_envelope_mechanics_implemented": True,
        "typed_trusted_wire_replay_v2_version": (
            phase2b_trusted_wire_typed_replay_v2.TYPED_TRUSTED_WIRE_REPLAY_V2_VERSION
        ),
        "typed_trusted_wire_replay_v2_policy_id": (
            phase2b_trusted_wire_typed_replay_v2.TYPED_TRUSTED_WIRE_REPLAY_V2_POLICY_ID
        ),
        "public_typed_trusted_wire_v2_replay_mechanics_implemented": True,
        "typed_trusted_wire_replay_v2_batch_policy_membership_mechanics_implemented": True,
        "typed_trusted_wire_replay_v2_whole_batch_atomic_mechanics_implemented": True,
        "typed_trusted_wire_replay_v2_compact_authority_canonical_mechanics_implemented": True,
        "typed_trusted_wire_replay_v2_public_provenance_mechanics_implemented": True,
        "typed_trusted_wire_replay_v2_direct_exact_transform_mechanics_implemented": True,
        "typed_trusted_wire_replay_v2_secret_custodian_replay_verified": False,
        "typed_trusted_wire_replay_v2_whole_batch_shuffle_publicly_verified": False,
        "typed_trusted_wire_replay_v2_purpose_separated_keys_publicly_verified": False,
        "typed_trusted_wire_replay_v2_post_shuffle_hmac_uuidv4_publicly_verified": False,
        "typed_trusted_wire_replay_v2_secret_hmac_padding_publicly_verified": False,
        "typed_trusted_wire_replay_v2_source_authority_binding_verified": False,
        "typed_trusted_wire_replay_v2_live_allocation_schedule_verified": False,
        "typed_trusted_wire_replay_v2_recognizer_capacity_evidence": False,
        "typed_trusted_wire_replay_v2_origin_authenticated": False,
        "typed_trusted_wire_replay_v2_formal_uuid_audit": False,
        "typed_trusted_wire_replay_v2_formal_covert_audit": False,
        "typed_trusted_wire_replay_v2_sealed_holdout_eligible": False,
        "typed_trusted_wire_replay_v2_c1_exit_evidence": False,
        "public_recognizer_registry_v2_schema_version": (
            phase2b_recognizer_input_archive_v2.PUBLIC_RECOGNIZER_REGISTRY_V2_SCHEMA_VERSION
        ),
        "public_recognizer_registry_v2_schema_id": (
            phase2b_recognizer_input_archive_v2.PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_ID_V2
        ),
        "public_recognizer_family_alias_policy_id_v2": (
            phase2b_recognizer_input_archive_v2.PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID_V2
        ),
        "trusted_recognizer_input_archive_v2_version": (
            phase2b_recognizer_input_archive_v2.TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION
        ),
        "recognizer_input_archive_v2_policy_id": (
            phase2b_recognizer_input_archive_v2.RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2
        ),
        "recognizer_input_archive_v2_claim_level": (
            phase2b_trusted_wire_v1.NON_AUTHORITATIVE_CLAIM_LEVEL
        ),
        "recognizer_input_archive_v2_structural_archive_mechanics_implemented": True,
        "recognizer_input_archive_v2_row_bijection_mechanics_implemented": True,
        "recognizer_input_archive_v2_registry_schema_mechanics_implemented": True,
        "recognizer_input_archive_v2_registry_authority_exact_scope_mechanics_implemented": True,
        "recognizer_input_archive_v2_compact_typed_replay_mechanics_implemented": True,
        "recognizer_input_archive_v2_direct_payload_transform_replay_mechanics_implemented": True,
        "recognizer_input_archive_v2_cross_row_unlinkable_public_uuid_disjoint_mechanics_implemented": True,
        "recognizer_input_archive_v2_private_single_live_allocation_gate_mechanics_implemented": True,
        "recognizer_input_archive_v2_private_source_public_uuid_disjointness_gate_mechanics_implemented": True,
        "real_positive_expanded_typed_profile_bytes": 125_582,
        "real_positive_compact_v2_payload_bytes": 50_255,
        "real_positive_compact_v2_payload_cap_headroom_bytes": 15_169,
        "real_positive_compact_v2_secret_padding_bytes": 15_201,
        "real_positive_compact_v2_fixed_envelope_bytes": 65_536,
        "real_positive_compact_v2_payload_fits_trusted_wire": True,
        "single_constructed_positive_compact_v2_mechanics_verified": True,
        "real_positive_compact_v2_exact_transform_replay_implemented": True,
        "real_positive_compact_v2_recognizer_input_archive_replay_implemented": True,
        "real_positive_compact_v2_derived_bridge_compilation_parity_implemented": True,
        "real_positive_compact_v2_derived_bridge_decision_parity_implemented": True,
        "recognizer_input_archive_v2_batch_policy_membership_verified": False,
        "recognizer_input_archive_v2_source_registry_projection_verified": False,
        "recognizer_input_archive_v2_source_public_disjoint_verified": False,
        "recognizer_input_archive_v2_single_live_allocation_verified": False,
        "recognizer_input_archive_v2_secret_custodian_replay_verified": False,
        "recognizer_input_archive_v2_origin_authenticated": False,
        "recognizer_input_archive_v2_formal_uuid_audit": False,
        "recognizer_input_archive_v2_formal_covert_audit": False,
        "recognizer_input_archive_v2_sealed_holdout_eligible": False,
        "recognizer_input_archive_v2_recognizer_executed": False,
        "recognizer_input_archive_v2_prediction_archive_evaluated": False,
        "recognizer_input_archive_v2_capacity_evidence": False,
        "recognizer_input_archive_v2_c1_exit_evidence": False,
        "public_recognizer_prediction_outcome_v2_schema_version": (
            phase2b_recognizer_prediction_v2.PUBLIC_RECOGNIZER_PREDICTION_OUTCOME_V2_SCHEMA_VERSION
        ),
        "public_recognizer_prediction_outcome_v2_schema_id": (
            phase2b_recognizer_prediction_v2.PUBLIC_RECOGNIZER_PREDICTION_OUTCOME_V2_SCHEMA_ID
        ),
        "recognizer_prediction_row_v2_policy_id": (
            phase2b_recognizer_prediction_v2.RECOGNIZER_PREDICTION_ROW_POLICY_ID_V2
        ),
        "recognizer_prediction_row_v2_claim_level": (
            phase2b_trusted_wire_v1.NON_AUTHORITATIVE_CLAIM_LEVEL
        ),
        "public_recognizer_prediction_outcome_v2_ephemeral_schema_mechanics_implemented": True,
        "v2_single_row_prediction_mapping_mechanics_implemented": True,
        "recognizer_prediction_row_v2_exact_input_and_freeze_binding_mechanics_implemented": True,
        "recognizer_prediction_row_v2_compact_typed_replay_mechanics_implemented": True,
        "recognizer_prediction_row_v2_public_registry_adapter_mechanics_implemented": True,
        "recognizer_prediction_row_v2_exact_derived_bridge_mechanics_implemented": True,
        "recognizer_prediction_row_v2_closed_decision_reason_mapping_mechanics_implemented": True,
        "recognizer_prediction_row_v2_cross_version_rejection_mechanics_implemented": True,
        "recognizer_prediction_row_v2_private_ephemeral_issue_mechanics_implemented": True,
        "real_positive_compact_v2_single_row_prediction_mapping_mechanics_verified": True,
        "real_positive_compact_v2_prediction_decision_parity_implemented": True,
        "real_positive_compact_v2_prediction_bundle_identity_parity_implemented": True,
        "real_positive_compact_v2_prediction_family_binding_scale_parity_implemented": True,
        "real_positive_compact_v2_prediction_input_protocol_freeze_root_parity_implemented": True,
        "recognizer_prediction_row_v2_durable_receipt_implemented": False,
        "recognizer_prediction_row_v2_input_archive_membership_verified": False,
        "recognizer_prediction_row_v2_batch_policy_membership_verified": False,
        "recognizer_prediction_row_v2_execution_manifest_authority_verified": False,
        "recognizer_prediction_row_v2_recognizer_executed": False,
        "recognizer_prediction_row_v2_runtime_executed": False,
        "recognizer_prediction_row_v2_capacity_evidence": False,
        "recognizer_prediction_row_v2_prediction_scoring_implemented": False,
        "recognizer_prediction_row_v2_effect_evidence": False,
        "recognizer_prediction_row_v2_origin_authenticated": False,
        "recognizer_prediction_row_v2_formal_uuid_audit": False,
        "recognizer_prediction_row_v2_formal_covert_audit": False,
        "recognizer_prediction_row_v2_sealed_holdout_eligible": False,
        "recognizer_prediction_row_v2_c1_exit_evidence": False,
        "public_prediction_run_context_v2_schema_version": (
            phase2b_recognizer_prediction_archive_v2.PUBLIC_PREDICTION_RUN_CONTEXT_V2_SCHEMA_VERSION
        ),
        "public_prediction_run_context_v2_schema_id": (
            phase2b_recognizer_prediction_archive_v2.PUBLIC_PREDICTION_RUN_CONTEXT_V2_SCHEMA_ID
        ),
        "public_recognizer_prediction_record_v2_schema_version": (
            phase2b_recognizer_prediction_archive_v2.PUBLIC_RECOGNIZER_PREDICTION_RECORD_V2_SCHEMA_VERSION
        ),
        "public_recognizer_prediction_record_v2_schema_id": (
            phase2b_recognizer_prediction_archive_v2.PUBLIC_RECOGNIZER_PREDICTION_RECORD_V2_SCHEMA_ID
        ),
        "recognizer_prediction_archive_v2_version": (
            phase2b_recognizer_prediction_archive_v2.RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION
        ),
        "recognizer_prediction_archive_v2_policy_id": (
            phase2b_recognizer_prediction_archive_v2.RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2
        ),
        "recognizer_prediction_archive_v2_wire_version": (
            phase2b_recognizer_prediction_archive_v2.PREDICTION_ARCHIVE_WIRE_VERSION_V2
        ),
        "recognizer_prediction_archive_v2_magic_hex": (
            phase2b_recognizer_prediction_archive_v2.PREDICTION_ARCHIVE_MAGIC_V2.hex()
        ),
        "recognizer_prediction_archive_v2_claim_level": (
            phase2b_trusted_wire_v1.NON_AUTHORITATIVE_CLAIM_LEVEL
        ),
        "v2_full_960_prediction_archive_structural_codec_implemented": True,
        "recognizer_prediction_archive_v2_independent_identity_mechanics_implemented": True,
        "recognizer_prediction_archive_v2_closed_context_schema_mechanics_implemented": True,
        "recognizer_prediction_archive_v2_closed_record_schema_mechanics_implemented": True,
        "recognizer_prediction_archive_v2_bounded_canonical_framing_mechanics_implemented": True,
        "recognizer_prediction_archive_v2_ordered_row_root_coverage_mechanics_implemented": True,
        "recognizer_prediction_archive_v2_exact_960_count_gate_mechanics_implemented": True,
        "recognizer_prediction_archive_v2_atomic_fail_closed_builder_mechanics_implemented": True,
        "recognizer_prediction_archive_v2_cross_version_rejection_mechanics_implemented": True,
        "synthetic_exact_960_prediction_archive_v2_structural_mechanics_verified": True,
        "recognizer_prediction_archive_v2_input_archive_membership_verified": False,
        "recognizer_prediction_archive_v2_batch_policy_membership_verified": False,
        "recognizer_prediction_archive_v2_source_registry_projection_verified": False,
        "recognizer_prediction_archive_v2_source_public_disjoint_verified": False,
        "recognizer_prediction_archive_v2_single_live_allocation_verified": False,
        "recognizer_prediction_archive_v2_secret_custodian_replay_verified": False,
        "recognizer_prediction_archive_v2_execution_manifest_authority_verified": False,
        "recognizer_prediction_archive_v2_derived_mapping_verified": False,
        "recognizer_prediction_archive_v2_recognizer_executed": False,
        "recognizer_prediction_archive_v2_runtime_executed": False,
        "recognizer_prediction_archive_v2_actual_960_case_run_verified": False,
        "recognizer_prediction_archive_v2_recognizer_capacity_evidence": False,
        "recognizer_prediction_archive_v2_origin_authenticated": False,
        "recognizer_prediction_archive_v2_formal_uuid_audit": False,
        "recognizer_prediction_archive_v2_formal_covert_audit": False,
        "recognizer_prediction_archive_v2_sealed_holdout_eligible": False,
        "recognizer_prediction_archive_v2_prediction_scored": False,
        "recognizer_prediction_archive_v2_effect_evidence": False,
        "recognizer_prediction_archive_v2_c1_exit_evidence": False,
        "unsealed_prediction_evaluator_v2_version": (
            phase2b_unsealed_prediction_evaluator_v2.UNSEALED_PREDICTION_EVALUATOR_V2_VERSION
        ),
        "unsealed_prediction_evaluator_v2_policy_id": (
            phase2b_unsealed_prediction_evaluator_v2.UNSEALED_PREDICTION_EVALUATOR_POLICY_ID_V2
        ),
        "unsealed_prediction_evaluator_v2_claim_level": (
            phase2b_trusted_wire_v1.NON_AUTHORITATIVE_CLAIM_LEVEL
        ),
        "v2_unsealed_prediction_evaluator_implemented": True,
        "unsealed_prediction_evaluator_v2_independent_identity_mechanics_implemented": True,
        "unsealed_prediction_evaluator_v2_evaluator_side_partition_label_separation_mechanics_implemented": True,
        "unsealed_prediction_evaluator_v2_exact_720_240_count_gate_mechanics_implemented": True,
        "unsealed_prediction_evaluator_v2_sorted_unique_partition_mechanics_implemented": True,
        "unsealed_prediction_evaluator_v2_disjoint_set_exhaustive_partition_mechanics_implemented": True,
        "unsealed_prediction_evaluator_v2_partition_root_binding_mechanics_implemented": True,
        "unsealed_prediction_evaluator_v2_ordered_archive_row_root_binding_mechanics_implemented": True,
        "unsealed_prediction_evaluator_v2_single_public_v2_archive_replay_mechanics_implemented": True,
        "unsealed_prediction_evaluator_v2_atomic_fail_closed_mechanics_implemented": True,
        "unsealed_prediction_evaluator_v2_cross_version_rejection_mechanics_implemented": True,
        "synthetic_exact_720_240_unsealed_prediction_evaluator_v2_structural_replay_verified": True,
        "unsealed_prediction_evaluator_v2_challenge_in_main_denominator": False,
        "unsealed_prediction_evaluator_v2_input_archive_membership_verified": False,
        "unsealed_prediction_evaluator_v2_batch_policy_membership_verified": False,
        "unsealed_prediction_evaluator_v2_source_registry_projection_verified": False,
        "unsealed_prediction_evaluator_v2_source_public_disjoint_verified": False,
        "unsealed_prediction_evaluator_v2_single_live_allocation_verified": False,
        "unsealed_prediction_evaluator_v2_secret_custodian_replay_verified": False,
        "unsealed_prediction_evaluator_v2_execution_manifest_authority_verified": False,
        "unsealed_prediction_evaluator_v2_partition_manifest_authority_verified": False,
        "unsealed_prediction_evaluator_v2_derived_mapping_verified": False,
        "unsealed_prediction_evaluator_v2_recognizer_executed": False,
        "unsealed_prediction_evaluator_v2_runtime_executed": False,
        "unsealed_prediction_evaluator_v2_actual_960_case_run_verified": False,
        "unsealed_prediction_evaluator_v2_recognizer_capacity_evidence": False,
        "unsealed_prediction_evaluator_v2_origin_authenticated": False,
        "unsealed_prediction_evaluator_v2_formal_uuid_audit": False,
        "unsealed_prediction_evaluator_v2_formal_covert_audit": False,
        "unsealed_prediction_evaluator_v2_sealed_holdout_eligible": False,
        "unsealed_prediction_evaluator_v2_scoring_performed": False,
        "unsealed_prediction_evaluator_v2_prediction_scored": False,
        "unsealed_prediction_evaluator_v2_effect_evidence": False,
        "unsealed_prediction_evaluator_v2_c1_exit_evidence": False,
        "strict_recognizer_cli_v2_command": (
            phase2b_strict_recognizer_cli_v2.STRICT_RECOGNIZER_CLI_V2_COMMAND
        ),
        "strict_recognizer_cli_v2_schema_version": (
            phase2b_strict_recognizer_cli_v2.STRICT_RECOGNIZER_CLI_V2_SCHEMA_VERSION
        ),
        "strict_recognizer_cli_v2_schema_id": (
            phase2b_strict_recognizer_cli_v2.STRICT_RECOGNIZER_CLI_V2_SCHEMA_ID
        ),
        "strict_recognizer_cli_v2_policy_id": (
            phase2b_strict_recognizer_cli_v2.STRICT_RECOGNIZER_CLI_V2_POLICY_ID
        ),
        "strict_recognizer_cli_v2_claim_level": (
            phase2b_trusted_wire_v1.NON_AUTHORITATIVE_CLAIM_LEVEL
        ),
        "strict_recognizer_cli_v2_structural_input_output_contract_implemented": True,
        "strict_recognizer_cli_v2_read_only_no_output_artifact_mechanics_implemented": True,
        "strict_recognizer_cli_v2_canonical_absolute_nofollow_single_link_regular_file_mechanics_implemented": True,
        "strict_recognizer_cli_v2_bounded_stable_fd_read_before_decode_mechanics_implemented": True,
        "strict_recognizer_cli_v2_single_public_v2_input_archive_replay_mechanics_implemented": True,
        "strict_recognizer_cli_v2_single_public_v2_prediction_archive_replay_mechanics_implemented": True,
        "strict_recognizer_cli_v2_cross_archive_context_binding_mechanics_implemented": True,
        "strict_recognizer_cli_v2_ordered_row_identity_binding_mechanics_implemented": True,
        "strict_recognizer_cli_v2_seven_input_root_columns_positional_binding_mechanics_implemented": True,
        "strict_recognizer_cli_v2_generic_atomic_fail_closed_json_mechanics_implemented": True,
        "strict_recognizer_cli_v2_input_archive_membership_verified": False,
        "strict_recognizer_cli_v2_batch_policy_membership_verified": False,
        "strict_recognizer_cli_v2_source_registry_projection_verified": False,
        "strict_recognizer_cli_v2_source_public_disjoint_verified": False,
        "strict_recognizer_cli_v2_single_live_allocation_verified": False,
        "strict_recognizer_cli_v2_secret_custodian_replay_verified": False,
        "strict_recognizer_cli_v2_execution_manifest_authority_verified": False,
        "strict_recognizer_cli_v2_partition_manifest_authority_verified": False,
        "strict_recognizer_cli_v2_derived_mapping_verified": False,
        "strict_recognizer_cli_v2_recognizer_executed": False,
        "strict_recognizer_cli_v2_runtime_executed": False,
        "strict_recognizer_cli_v2_actual_960_case_run_verified": False,
        "strict_recognizer_cli_v2_recognizer_capacity_evidence": False,
        "strict_recognizer_cli_v2_origin_authenticated": False,
        "strict_recognizer_cli_v2_formal_uuid_audit": False,
        "strict_recognizer_cli_v2_formal_covert_audit": False,
        "strict_recognizer_cli_v2_sealed_holdout_eligible": False,
        "strict_recognizer_cli_v2_scoring_performed": False,
        "strict_recognizer_cli_v2_prediction_scored": False,
        "strict_recognizer_cli_v2_effect_evidence": False,
        "strict_recognizer_cli_v2_c1_exit_evidence": False,
        "next_phase2b_construction_slice": (
            "formal_unsealed_prediction_scoring_contract_v2"
        ),
        "real_positive_prediction_end_to_end_replay_implemented": False,
        "recognizer_prediction_capacity_evidence": False,
        "prediction_scoring_implemented": False,
        "prediction_effect_evidence": False,
        "actual_960_case_prediction_archive_run": False,
        "recognizer_runtime_executed": False,
        "prediction_archive_input_membership_verified": False,
        "prediction_archive_execution_manifest_authority_verified": False,
        "prediction_archive_derived_mapping_verified_by_public_decode": False,
        "prediction_archive_origin_authenticated": False,
        "prediction_archive_formal_covert_audit": False,
        "prediction_archive_sealed_holdout_eligible": False,
        "prediction_archive_c1_exit_evidence": False,
        "pairwise_distinct_key_source_contract_implemented": True,
        "key_source_statistical_independence_attested": False,
        "whole_batch_unbiased_fisher_yates_mechanics_implemented": True,
        "post_shuffle_namespace_hmac_uuidv4_mechanics_implemented": True,
        "case_local_latent_id_anti_link_allocation_implemented": True,
        "renamed_authority_schema_recanonicalization_implemented": True,
        "wire_only_public_provenance_rebinding_mechanics_implemented": True,
        "secret_hmac_padding_custodian_replay_mechanics_implemented": True,
        "batch_atomic_keyed_trusted_wire_mechanics_implemented": True,
        "uuid_collision_retry_warning_mechanics_implemented": True,
        "trusted_wire_custodian_secret_replay_mechanics_implemented": True,
        "trusted_wire_1024_authority_capacity_qualified": False,
        "global_batch_shuffle_implemented": False,
        "post_shuffle_hmac_uuidv4_assignment_implemented": False,
        "provenance_rebound_to_public_payload_implemented": False,
        "secret_padding_replay_implemented": False,
        "batch_atomic_trusted_wire_builder_implemented": False,
        "typed_trusted_wire_authority_decode_replay_implemented": True,
        "typed_trusted_wire_authority_decode_replay_claim_level": (
            phase2b_trusted_wire_v1.NON_AUTHORITATIVE_CLAIM_LEVEL
        ),
        "trusted_wire_origin_authenticated": False,
        "trusted_rfc8785_wire_builder_implemented": False,
        "formal_uuid_namespace_field_audit_implemented": False,
        "randomized_identifier_assignment_attested": False,
        "uncertainty_semantics_compiler_implemented": False,
        "internal_candidate_enumeration_implemented": True,
        "interval_selector_core_implemented": True,
        "public_selector_reenumerates_adapter_grid_from_bundle": True,
        "inconclusive_structural_competitor_forces_abstention": True,
        "admissible_scale_set_output_supported": True,
        "oci_isolation_launch_contract_implemented": True,
        "component_source_ids": {
            name: "sha256:"
            + hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for name, module in (
                ("adapter", phase2b_adapter),
                ("covert_audit_mechanics", phase2b_covert_audit_v1),
                (
                    "exact_derived_witness_bridge",
                    phase2b_exact_derived_witness_bridge_v1,
                ),
                ("exact_bridge", phase2b_exact_bridge_v1),
                (
                    "exact_transform_semantics",
                    phase2b_exact_transform_semantics_v1,
                ),
                ("projection_compiler", phase2b_projection_compiler),
                ("runner", phase2b_runner),
                ("selector", phase2b_selector),
                (
                    "recognizer_input_archive_mechanics",
                    phase2b_recognizer_input_archive_v1,
                ),
                (
                    "recognizer_input_archive_mechanics_v2",
                    phase2b_recognizer_input_archive_v2,
                ),
                (
                    "recognizer_prediction_archive_mechanics",
                    phase2b_recognizer_prediction_archive_v1,
                ),
                (
                    "recognizer_prediction_archive_mechanics_v2",
                    phase2b_recognizer_prediction_archive_v2,
                ),
                (
                    "recognizer_prediction_row_mapping_mechanics_v2",
                    phase2b_recognizer_prediction_v2,
                ),
                (
                    "strict_recognizer_cli_v2_structural_verifier",
                    phase2b_strict_recognizer_cli_v2,
                ),
                (
                    "trusted_wire_keyed_batch_mechanics",
                    phase2b_trusted_wire_batch_v1,
                ),
                (
                    "trusted_wire_keyed_batch_mechanics_v2",
                    phase2b_trusted_wire_batch_v2,
                ),
                ("trusted_wire_profile_mechanics", phase2b_trusted_wire_v1),
                (
                    "trusted_wire_typed_authority_codec",
                    phase2b_trusted_wire_typed_authority_v1,
                ),
                (
                    "trusted_wire_compact_typed_authority_codec_v2",
                    phase2b_trusted_wire_typed_authority_v2,
                ),
                (
                    "trusted_wire_typed_replay_mechanics",
                    phase2b_trusted_wire_typed_replay_v1,
                ),
                (
                    "trusted_wire_typed_replay_mechanics_v2",
                    phase2b_trusted_wire_typed_replay_v2,
                ),
                ("uncertainty_compiler", phase2b_uncertainty_compiler),
                (
                    "unsealed_prediction_structural_evaluator",
                    phase2b_unsealed_prediction_evaluator_v1,
                ),
                (
                    "unsealed_prediction_structural_evaluator_v2",
                    phase2b_unsealed_prediction_evaluator_v2,
                ),
                ("wire", phase2b_wire),
            )
        },
        "phase2a_fixtures_in_formal_protocol": False,
        "scale_hypothesis_generation_required": (
            protocol.scale_hypothesis_generation_required
        ),
        "current_explicit_scale_selection_counts_as_scale_inference": False,
        "independent_latent_case_count": protocol.independent_latent_case_count,
        "preservation_pairs_counted_in_720": False,
        "law_family_count": len(protocol.law_families),
        "canonical_family_mapping": {
            law_kind.value: family_id.value
            for law_kind, family_id in exact_freeze.family_mapping
        },
        "scale_cell_count": protocol.scale_cell_count,
        "cases_per_family_scale_cell": protocol.cases_per_family_scale_cell,
        "case_type_totals": dict(protocol.case_type_totals),
        "case_type_quota_per_cell": dict(exact_freeze.holdout.case_quota_per_cell),
        "margin_stratum_totals": dict(protocol.margin_stratum_totals),
        "margin_quota_per_cell": dict(protocol.margin_stratum_per_cell),
        "margin_case_joint_quota_per_cell": [
            {
                "margin_stratum": margin,
                "case_type": case_type,
                "count": count,
            }
            for margin, case_type, count in (
                exact_freeze.holdout.margin_case_joint_quota_per_cell
            )
        ],
        "metric_denominators": {
            item.metric: {
                "case_types": list(item.included_case_types),
                "count": item.expected_count,
                "separately_reported": item.separately_reported,
            }
            for item in exact_freeze.holdout.metric_denominators
        },
        "semantic_conflict_challenge_case_count": (
            exact_freeze.semantic_conflict.case_count
        ),
        "semantic_conflict_in_main_accuracy_denominator": False,
        "legal_preservation_pair_count": (
            exact_freeze.legal_preservation_pair_count
        ),
        "invalid_transform_control_count": (
            exact_freeze.invalid_transform_control_count
        ),
        "total_preservation_sensitivity_pair_count": (
            exact_freeze.total_preservation_sensitivity_pair_count
        ),
        "bootstrap": {
            "method": exact_freeze.bootstrap.method,
            "replicates": exact_freeze.bootstrap.replicates,
            "master_seed": exact_freeze.bootstrap.seed,
            "uint32_derivation_id": exact_freeze.bootstrap.seed_derivation_id,
            "derived_uint32_seed": exact_freeze.bootstrap.derived_uint32_seed,
            "resampling_unit": exact_freeze.bootstrap.resampling_unit,
            "interval": exact_freeze.bootstrap.interval,
        },
        "baseline_specs_frozen": True,
        "baseline_spec_ids": {
            item.baseline_id: item.content_id for item in exact_freeze.baselines
        },
        "exact_baseline_revisions_registered": False,
        "rerun_policy_frozen": True,
        "maximum_reexecutions_before_answer_reveal": (
            exact_freeze.rerun_policy.maximum_reexecutions
        ),
        "validation_version_policy_frozen": True,
        "covert_channel_audit_frozen": True,
        "fixed_envelope_covert_audit_mechanics_implemented": True,
        "fixed_envelope_covert_statistics_mechanics_implemented": True,
        "fixed_envelope_covert_audit_semantics_id": (
            phase2b_covert_audit_v1.SEMANTICS_ID
        ),
        "fixed_envelope_covert_audit_policy_id": (
            phase2b_covert_audit_v1.DEFAULT_COVERT_AUDIT_POLICY.policy_id
        ),
        "fixed_envelope_covert_audit_claim_level": (
            phase2b_covert_audit_v1.NON_AUTHORITATIVE_CLAIM_LEVEL
        ),
        "covert_channel_audit_implemented": False,
        "formal_covert_channel_audit_passed": False,
        "covert_channel_audit_executed": False,
        "global_consistent_renamings_required": (
            exact_freeze.covert_channel_audit.global_consistent_renamings
        ),
        "formal_uncertainty_models_allowed": [
            item.value for item in exact_freeze.uncertainty_policy.allowed_kinds
        ],
        "standard_error_formal_selector_status": (
            exact_freeze.uncertainty_policy.standard_error_status
        ),
        "holdout_run_limit": protocol.holdout_run_limit,
        "validation_attempts_per_version": protocol.validation_attempts_per_version,
        "maximum_validation_versions_before_no_go": (
            protocol.maximum_validation_versions_before_no_go
        ),
        "shadow_only": protocol.shadow_only,
        "active_promotion_enabled": protocol.active_promotion_enabled,
        "isolation_profile": {
            "profile_id": protocol.isolation_profile.profile_id,
            "contract_complete": protocol.isolation_profile.contract_complete,
            "local_contract_proves_external_enforcement": (
                protocol.isolation_profile.proves_external_enforcement
            ),
            "required_controls": [
                item.name
                for item in fields(protocol.isolation_profile)
                if item.name != "profile_id"
            ],
        },
        "overall_gates": [
            {
                "metric": item.metric,
                "minimum_point_estimate": item.minimum_point_estimate,
                "minimum_one_sided_wilson_lcb": (
                    item.minimum_one_sided_wilson_lcb
                ),
            }
            for item in protocol.overall_gates
        ],
        "slice_gates": [
            {
                "scope": item.scope,
                "metric": item.metric,
                "minimum_point_estimate": item.minimum_point_estimate,
                "minimum_one_sided_wilson_lcb": (
                    item.minimum_one_sided_wilson_lcb
                ),
            }
            for item in protocol.slice_gates
        ],
        "unresolved_freeze_questions": list(protocol.unresolved_freeze_questions),
        "implementation_blockers": list(protocol.implementation_blockers),
        "ready_for_holdout_generation": protocol.ready_for_holdout_generation,
        "claim_boundary": (
            "Public protocol, immutable lifecycle records, and a process-local "
            "fork guard only; no durable one-shot custodian authority, secret "
            "holdout, external attestation, isolated run, baseline result, or "
            "formal Phase-2B qualification is present."
        ),
    }
    payload["report_id"] = stable_hash(payload, prefix="phase2b_prereg_report_")
    return payload


__all__ = (
    "BaselineKind",
    "BaselineRegistration",
    "BinaryGateResult",
    "BinaryGateThreshold",
    "CandidateFootprint",
    "CaseAllocation",
    "ExecutionFreezeManifest",
    "HoldoutLifecycle",
    "IsolationProfile",
    "MarginAllocation",
    "MarginStratum",
    "MeasurementUse",
    "ONE_SIDED_CONFIDENCE",
    "PHASE2B_PROTOCOL_SCHEMA",
    "PHASE2B_PROTOCOL_VERSION",
    "PHASE2B_REPORT_NAME",
    "Phase2BCaseType",
    "Phase2BProtocol",
    "PreservationMode",
    "PreservationRequirement",
    "ScalarUpperGate",
    "SealedRunLedger",
    "SharedFootprintResult",
    "evaluate_binary_gate",
    "evaluate_shared_footprint",
    "frozen_phase2b_protocol",
    "one_sided_wilson_lower_bound",
    "phase2b_preregistration_report",
    "salted_answer_commitment_sha256",
)
