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
from .schema import LawKind, require_tuple


PHASE2B_PROTOCOL_SCHEMA: Final = "hegel-machine-phase2b-preregistration/1"
PHASE2B_PROTOCOL_VERSION: Final = "phase2b_typed_evidence_preregistration_v1"
PHASE2B_REPORT_NAME: Final = "phase2b_preregistration_readiness_v1"
ONE_SIDED_CONFIDENCE: Final = 0.95
ONE_SIDED_Z_95: Final = NormalDist().inv_cdf(ONE_SIDED_CONFIDENCE)


class Phase2BCaseType(str, Enum):
    ANSWERABLE_POSITIVE = "answerable_positive"
    WRONG_FAMILY_HARD_NEGATIVE = "wrong_family_hard_negative"
    BINDING_COUNTERFACTUAL = "binding_counterfactual"
    SCALE_COUNTERFACTUAL = "scale_counterfactual"
    SIGN_OR_INVARIANT_BREAK = "sign_or_invariant_break"
    INSUFFICIENT_OR_AMBIGUOUS = "insufficient_or_ambiguous"


class MarginStratum(str, Enum):
    CLEAR_INTERIOR = "clear_interior"
    MODERATE = "moderate"
    NEAR_BOUNDARY_IDENTIFIABLE = "near_boundary_identifiable"
    AMBIGUOUS_OR_INSUFFICIENT = "ambiguous_or_insufficient"


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
    transform: str
    applicability: str
    minimum_pairs_per_family_scale: int = 0
    minimum_pairs_per_family: int = 0
    mode: PreservationMode = PreservationMode.EXACT_INVARIANCE

    def __post_init__(self) -> None:
        if not self.transform or not self.applicability:
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
    implementation_id: str
    artifact_sha256: str
    frozen_before_holdout_generation: bool

    def __post_init__(self) -> None:
        if not self.implementation_id:
            raise ValueError("baseline implementation ID is required")
        _require_sha256(self.artifact_sha256, "baseline artifact SHA-256")
        if type(self.frozen_before_holdout_generation) is not bool:
            raise TypeError("baseline frozen flag must be boolean")


@dataclass(frozen=True, slots=True)
class Phase2BProtocol:
    schema_version: str
    protocol_version: str
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
    maximum_validation_protocol_runs: int
    unresolved_freeze_questions: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "law_families",
            "case_allocations",
            "margin_allocations",
            "overall_gates",
            "slice_gates",
            "preservation_requirements",
            "unresolved_freeze_questions",
        ):
            require_tuple(getattr(self, name), f"Phase-2B {name}")
        if (self.schema_version, self.protocol_version) != (
            PHASE2B_PROTOCOL_SCHEMA,
            PHASE2B_PROTOCOL_VERSION,
        ):
            raise ValueError("Phase-2B protocol version drift")
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
            self.independent_latent_case_count * item.share_numerator
            % item.share_denominator
            for item in self.margin_allocations
        ):
            raise ValueError("Phase-2B margin shares must yield integer case counts")
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
        if self.maximum_validation_protocol_runs != 2:
            raise ValueError("validation permits at most two full protocol runs")

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
    def ready_for_holdout_generation(self) -> bool:
        return not self.unresolved_freeze_questions

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

    Decided fields are frozen.  The remaining questions are machine-visible so
    an external custodian cannot generate a formal holdout prematurely.
    """

    allocations = (
        CaseAllocation(Phase2BCaseType.ANSWERABLE_POSITIVE, 20),
        CaseAllocation(Phase2BCaseType.WRONG_FAMILY_HARD_NEGATIVE, 8),
        CaseAllocation(Phase2BCaseType.BINDING_COUNTERFACTUAL, 8),
        CaseAllocation(Phase2BCaseType.SCALE_COUNTERFACTUAL, 8),
        CaseAllocation(Phase2BCaseType.SIGN_OR_INVARIANT_BREAK, 8),
        CaseAllocation(Phase2BCaseType.INSUFFICIENT_OR_AMBIGUOUS, 8),
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
            MarginStratum.AMBIGUOUS_OR_INSUFFICIENT,
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
        PreservationRequirement("entity_alpha_renaming", "all", 6),
        PreservationRequirement("observation_reorder", "order_invariant", 6),
        PreservationRequirement("irrelevant_entity_augmentation", "scoped", 6),
        PreservationRequirement("unit_conversion", "numeric", 8),
        PreservationRequirement(
            "coordinate_translation_or_scaling",
            "invariant_or_equivariant",
            8,
            mode=PreservationMode.APPROXIMATE_EQUIVARIANCE,
        ),
        PreservationRequirement(
            "equivalent_aggregation_split_merge",
            "conservation_additivity_coverage",
            8,
        ),
        PreservationRequirement(
            "nontrivial_scale_map",
            "cross_scale_stable",
            minimum_pairs_per_family=10,
            mode=PreservationMode.APPROXIMATE_EQUIVARIANCE,
        ),
        PreservationRequirement(
            "sign_convention_reparameterization",
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
        maximum_validation_protocol_runs=2,
        unresolved_freeze_questions=(
            "margin_strata_require_108_ambiguous_or_admissible_cases_but_case_table_allocates_96",
            "preservation_applicability_matrix_and_total_pair_count_not_frozen",
            "embedding_llm_and_flat_typed_baseline_artifacts_not_pinned",
            "bootstrap_resampling_unit_seed_and_iteration_count_not_frozen",
            "infrastructure_failure_retry_policy_before_answer_reveal_not_frozen",
            "shared_footprint_cell_taxonomy_and_family_discrimination_statistic_not_frozen",
            "semantic_conflict_subset_membership_in_or_beyond_720_not_frozen",
            "allowed_field_side_channel_and_identifier_randomization_audit_not_frozen",
            "uncertainty_model_to_interval_semantics_not_frozen",
            "functional_recognizer_cli_signed_minimal_image_and_archive_evaluator_not_implemented",
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
    shared_ids = set(correct_public).intersection(competitor_public)
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
    }.intersection(
        competitor_public[item_id].witness_kind for item_id in shared_ids
    )
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
        if self.protocol_id != protocol.protocol_id:
            raise ValueError("execution manifest does not bind the frozen protocol")
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
        if {item.kind for item in self.baseline_registrations} != set(BaselineKind):
            raise ValueError("all three baseline classes must be registered")
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

    from . import phase2b_adapter, phase2b_runner, phase2b_selector, phase2b_wire

    protocol = frozen_phase2b_protocol()
    implementation_id = (
        "phase2b_protocol_source_sha256_"
        + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    )
    payload: dict[str, object] = {
        "artifact": PHASE2B_REPORT_NAME,
        "schema_version": protocol.schema_version,
        "protocol_version": protocol.protocol_version,
        "protocol_id": protocol.protocol_id,
        "implementation_id": implementation_id,
        "milestone_id": protocol.milestone_id,
        "milestone_name": protocol.milestone_name,
        "formal_claim_name_reserved_until_exit": protocol.formal_claim_name,
        "status": "preregistration_candidate_with_open_freeze_questions",
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
        "prediction_archive_evaluator_implemented": False,
        "public_wire_contract_implemented": True,
        "public_wire_is_family_neutral_shaped_only": True,
        "semantic_family_neutrality_audited": False,
        "allowed_field_answer_correlation_audit_implemented": False,
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
                ("runner", phase2b_runner),
                ("selector", phase2b_selector),
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
        "scale_cell_count": protocol.scale_cell_count,
        "cases_per_family_scale_cell": protocol.cases_per_family_scale_cell,
        "case_type_totals": dict(protocol.case_type_totals),
        "margin_stratum_totals": dict(protocol.margin_stratum_totals),
        "holdout_run_limit": protocol.holdout_run_limit,
        "maximum_validation_protocol_runs": (
            protocol.maximum_validation_protocol_runs
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
