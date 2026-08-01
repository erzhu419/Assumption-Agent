"""Exact, immutable Phase-2B statistical and anti-leak freeze.

This module records the normative decisions in
``Hegel_Machine_Phase2B_Phase3_Exact_Freeze_Decisions.md``.  It is a contract,
not a generator, auditor, baseline implementation, or sealed-run result.  In
particular, ``formal_holdout_generation_authorized`` remains false until the
independent implementations named by ``implementation_blockers`` exist and
are externally attested.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
from typing import Final

from .hashing import stable_hash
from .schema import LawKind


PHASE2B_EXACT_FREEZE_VERSION: Final = "hegel-freeze-p2b-p3-v1.0.1"
# The source decision freezes this 64-bit value.  Consumers such as sklearn
# accept only a uint32 random_state, so the conversion is itself part of the
# freeze rather than an implementation-defined truncation.
BOOTSTRAP_SEED: Final = 411876909552964556
BOOTSTRAP_MASTER_SEED: Final = BOOTSTRAP_SEED
BOOTSTRAP_UINT32_DERIVATION_ID: Final = (
    "sha256_domain_separated_uint64_be_first32_v1"
)
_BOOTSTRAP_UINT32_DERIVATION_DOMAIN: Final = (
    b"hegel-machine/phase2b/bootstrap-and-flat-baseline/uint32/v1\x00"
)


def derive_bootstrap_uint32_seed(master_seed: int) -> int:
    """Derive the frozen executable uint32 seed from a uint64 master seed."""

    if type(master_seed) is not int:
        raise TypeError("bootstrap master seed must be an integer")
    if not 0 <= master_seed < (1 << 64):
        raise ValueError("bootstrap master seed must fit unsigned 64-bit")
    digest = hashlib.sha256(
        _BOOTSTRAP_UINT32_DERIVATION_DOMAIN
        + master_seed.to_bytes(8, byteorder="big", signed=False)
    ).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


BOOTSTRAP_DERIVED_UINT32_SEED: Final = derive_bootstrap_uint32_seed(
    BOOTSTRAP_MASTER_SEED
)


def _exact_frozen_value_equal(actual: object, expected: object) -> bool:
    """Compare a frozen field recursively without bool/int or Enum/str aliases."""

    if type(actual) is not type(expected):
        return False
    if is_dataclass(expected):
        return all(
            _exact_frozen_value_equal(
                getattr(actual, item.name),
                getattr(expected, item.name),
            )
            for item in fields(expected)
        )
    if isinstance(expected, tuple):
        return len(actual) == len(expected) and all(  # type: ignore[arg-type]
            _exact_frozen_value_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(  # type: ignore[arg-type]
                actual,
                expected,
                strict=True,
            )
        )
    return bool(actual == expected)


class CanonicalFamilyId(str, Enum):
    F01 = "F01_symmetry_equivariance"
    F02 = "F02_monotonicity_order"
    F03 = "F03_conservation_balance"
    F04 = "F04_additivity_complementarity"
    F05 = "F05_locality_composition"
    F06 = "F06_negative_feedback_stability"


class PreservationTransform(str, Enum):
    ENTITY_ALPHA_RENAMING = "entity_alpha_renaming"
    OBSERVATION_REORDER = "observation_reorder"
    IRRELEVANT_ENTITY_AUGMENTATION = "irrelevant_entity_augmentation"
    UNIT_CONVERSION = "unit_conversion"
    COORDINATE_AFFINE_TRANSFORM = "coordinate_affine_transform"
    EQUIVALENT_AGGREGATION_SPLIT_MERGE = "equivalent_aggregation_split_merge"
    NONTRIVIAL_SCALE_MAP = "nontrivial_scale_map"
    SIGN_CONVENTION_REPARAMETERIZATION = (
        "sign_convention_reparameterization"
    )


class FootprintClass(str, Enum):
    P2_PAIR = "P2_PAIR"
    P3_CHAIN = "P3_CHAIN"
    P4_STAR = "P4_STAR"
    PSET_AGGREGATE = "PSET_AGGREGATE"


class FormalUncertaintyKind(str, Enum):
    ABSOLUTE_BOUND = "absolute_bound"
    STANDARD_ERROR = "standard_error"


@dataclass(frozen=True, slots=True)
class MetricDenominator:
    metric: str
    included_case_types: tuple[str, ...]
    expected_count: int
    separately_reported: bool = False

    def __post_init__(self) -> None:
        _require_nonempty_tuple(self.included_case_types, "metric case types")
        if not self.metric:
            raise ValueError("metric name is required")
        if isinstance(self.expected_count, bool) or self.expected_count <= 0:
            raise ValueError("metric denominator must be positive")
        if type(self.separately_reported) is not bool:
            raise TypeError("separately_reported must be boolean")


@dataclass(frozen=True, slots=True)
class HoldoutAllocationFreeze:
    family_count: int
    scale_count: int
    cases_per_cell: int
    case_quota_per_cell: tuple[tuple[str, int], ...]
    margin_quota_per_cell: tuple[tuple[str, int], ...]
    margin_case_joint_quota_per_cell: tuple[tuple[str, str, int], ...]
    metric_denominators: tuple[MetricDenominator, ...]
    set_valued_joint_rule: str

    def __post_init__(self) -> None:
        _require_nonempty_tuple(self.case_quota_per_cell, "case quotas")
        _require_nonempty_tuple(self.margin_quota_per_cell, "margin quotas")
        _require_nonempty_tuple(
            self.margin_case_joint_quota_per_cell,
            "margin-by-case joint quotas",
        )
        _require_nonempty_tuple(self.metric_denominators, "metric denominators")
        if (self.family_count, self.scale_count, self.cases_per_cell) != (6, 2, 60):
            raise ValueError("Phase-2B v1 cell axes drift")
        if sum(count for _, count in self.case_quota_per_cell) != 60:
            raise ValueError("case quotas must sum to 60 in every cell")
        if sum(count for _, count in self.margin_quota_per_cell) != 60:
            raise ValueError("margin quotas must sum to 60 in every cell")
        if self.case_quota_per_cell != (
            ("unique_scale_answerable", 19),
            ("admissible_scale_set_answerable", 1),
            ("wrong_family_hard_negative", 8),
            ("binding_counterfactual", 8),
            ("scale_counterfactual", 8),
            ("sign_or_invariant_break", 8),
            ("insufficient_or_nonidentifiable", 8),
        ):
            raise ValueError("case quotas drift from the exact 19+1 freeze")
        if self.margin_quota_per_cell != (
            ("clear_interior", 21),
            ("moderate", 18),
            ("near_boundary_identifiable", 12),
            ("nonunique_or_insufficient", 9),
        ):
            raise ValueError("margin quotas drift from the 21/18/12/9 freeze")
        if self.margin_case_joint_quota_per_cell != (
            (
                "nonunique_or_insufficient",
                "insufficient_or_nonidentifiable",
                8,
            ),
            (
                "nonunique_or_insufficient",
                "admissible_scale_set_answerable",
                1,
            ),
        ):
            raise ValueError(
                "nonunique-or-insufficient joint row must be exactly 8+1"
            )
        if sum(
            count
            for margin, _, count in self.margin_case_joint_quota_per_cell
            if margin == "nonunique_or_insufficient"
        ) != 9:
            raise ValueError("nonunique-or-insufficient joint row must sum to nine")
        if len({name for name, _ in self.case_quota_per_cell}) != len(
            self.case_quota_per_cell
        ):
            raise ValueError("case quota names must be unique")
        if len({name for name, _ in self.margin_quota_per_cell}) != len(
            self.margin_quota_per_cell
        ):
            raise ValueError("margin quota names must be unique")
        expected_denominators = {
            "answerable_count": 240,
            "family_exact_accuracy": 240,
            "binding_exact_accuracy": 240,
            "scale_set_accuracy": 240,
            "unique_scale_accuracy": 228,
            "joint_exact_accuracy": 240,
            "abstention_specificity": 228,
            "nonidentifiability_abstention_accuracy": 96,
            "set_valued_answer_accuracy": 12,
        }
        actual_denominators = {
            item.metric: item.expected_count for item in self.metric_denominators
        }
        if actual_denominators != expected_denominators:
            raise ValueError("metric denominators drift from the exact freeze")
        metric_case_types = {
            item.metric: item.included_case_types for item in self.metric_denominators
        }
        answerable = (
            "unique_scale_answerable",
            "admissible_scale_set_answerable",
        )
        if metric_case_types != {
            "answerable_count": answerable,
            "family_exact_accuracy": answerable,
            "binding_exact_accuracy": answerable,
            "scale_set_accuracy": answerable,
            "unique_scale_accuracy": ("unique_scale_answerable",),
            "joint_exact_accuracy": answerable,
            "abstention_specificity": ("unique_scale_answerable",),
            "nonidentifiability_abstention_accuracy": (
                "insufficient_or_nonidentifiable",
            ),
            "set_valued_answer_accuracy": (
                "admissible_scale_set_answerable",
            ),
        }:
            raise ValueError("metric case-type inclusion drift")
        if self.set_valued_joint_rule != (
            "family_exact_and_binding_exact_and_scale_set_exact_and_ANSWER_SET"
        ):
            raise ValueError("set-valued joint-exact rule drift")

    @property
    def cell_count(self) -> int:
        return self.family_count * self.scale_count

    @property
    def independent_latent_case_count(self) -> int:
        return self.cell_count * self.cases_per_cell

    @property
    def case_type_totals(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (name, per_cell * self.cell_count)
            for name, per_cell in self.case_quota_per_cell
        )

    @property
    def margin_stratum_totals(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (name, per_cell * self.cell_count)
            for name, per_cell in self.margin_quota_per_cell
        )


@dataclass(frozen=True, slots=True)
class PreservationRule:
    transform: PreservationTransform
    applicable_families: tuple[CanonicalFamilyId, ...]
    legal_pairs_per_family_scale: int = 0
    legal_pairs_per_family: int = 0
    invalid_controls_per_applicable_family: int = 2
    invalid_control_scale_coverage: str = "one_from_each_scale_when_scale_specific"

    def __post_init__(self) -> None:
        if not isinstance(self.transform, PreservationTransform):
            raise TypeError("preservation transform must use PreservationTransform")
        _require_nonempty_tuple(self.applicable_families, "applicable families")
        if len(set(self.applicable_families)) != len(self.applicable_families):
            raise ValueError("applicable families must be unique")
        counts = (self.legal_pairs_per_family_scale, self.legal_pairs_per_family)
        if any(isinstance(count, bool) or count < 0 for count in counts):
            raise ValueError("preservation counts must be nonnegative integers")
        if sum(count > 0 for count in counts) != 1:
            raise ValueError("a preservation rule needs exactly one counting axis")
        if self.invalid_controls_per_applicable_family != 2:
            raise ValueError("v1 requires two invalid controls per applicable family")
        if self.invalid_control_scale_coverage != (
            "one_from_each_scale_when_scale_specific"
        ):
            raise ValueError("invalid-control scale coverage drift")

    @property
    def legal_pair_count(self) -> int:
        family_count = len(self.applicable_families)
        return (
            family_count * 2 * self.legal_pairs_per_family_scale
            + family_count * self.legal_pairs_per_family
        )

    @property
    def invalid_control_count(self) -> int:
        return len(self.applicable_families) * self.invalid_controls_per_applicable_family


@dataclass(frozen=True, slots=True)
class BaselineSpec:
    baseline_id: str
    implementation: str
    revision_policy: str
    parameters: tuple[tuple[str, object], ...]
    output_heads: tuple[str, ...] = ()
    prompt: str | None = None

    def __post_init__(self) -> None:
        if not self.baseline_id or not self.implementation:
            raise ValueError("baseline identity is required")
        if self.revision_policy not in {
            "exact_40_hex_commit_required",
            "repository_dependency_lock_required",
        }:
            raise ValueError("baseline revision policy is not frozen")
        if not isinstance(self.parameters, tuple):
            raise TypeError("baseline parameters must be an immutable tuple")
        if not isinstance(self.output_heads, tuple):
            raise TypeError("baseline output heads must be an immutable tuple")
        if any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            for item in self.parameters
        ):
            raise TypeError("baseline parameters must be nonempty-name pairs")
        parameter_names = tuple(name for name, _ in self.parameters)
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("baseline parameter names must be unique")
        if self.prompt is not None and (
            not isinstance(self.prompt, str) or not self.prompt
        ):
            raise TypeError("baseline prompt must be nonempty text when present")

    @property
    def content_id(self) -> str:
        return stable_hash(self, prefix="phase2b_baseline_spec_")


@dataclass(frozen=True, slots=True)
class BootstrapSpec:
    method: str
    replicates: int
    seed: int
    seed_derivation_id: str
    derived_uint32_seed: int
    resampling_unit: str
    cluster_members: tuple[str, ...]
    interval: str

    def __post_init__(self) -> None:
        _require_nonempty_tuple(self.cluster_members, "bootstrap cluster members")
        if (
            self.method,
            self.replicates,
            self.seed,
            self.seed_derivation_id,
            self.derived_uint32_seed,
            self.resampling_unit,
            self.interval,
        ) != (
            "paired_cluster_bootstrap",
            10_000,
            BOOTSTRAP_MASTER_SEED,
            BOOTSTRAP_UINT32_DERIVATION_ID,
            BOOTSTRAP_DERIVED_UINT32_SEED,
            "latent_base_case",
            "one_sided_95_percent_percentile",
        ):
            raise ValueError("bootstrap configuration drift")
        if not 0 <= self.derived_uint32_seed < (1 << 32):
            raise ValueError("derived bootstrap seed must fit unsigned 32-bit")
        if self.derived_uint32_seed != derive_bootstrap_uint32_seed(self.seed):
            raise ValueError("derived bootstrap seed does not match the frozen algorithm")


@dataclass(frozen=True, slots=True)
class SemanticConflictChallenge:
    family_scale_cells: int = 12
    low_overlap_structural_positives_per_cell: int = 10
    high_overlap_structural_negatives_per_cell: int = 10
    included_in_main_accuracy_denominator: bool = False
    threshold_tuning_allowed: bool = False
    same_freeze_and_reveal_as_main: bool = True

    def __post_init__(self) -> None:
        if self.case_count != 240:
            raise ValueError("semantic-conflict challenge must contain 240 cases")
        for name in (
            "included_in_main_accuracy_denominator",
            "threshold_tuning_allowed",
        ):
            if getattr(self, name):
                raise ValueError(f"{name} must remain false")
        if not self.same_freeze_and_reveal_as_main:
            raise ValueError("challenge must share the main freeze and reveal")

    @property
    def case_count(self) -> int:
        return self.family_scale_cells * (
            self.low_overlap_structural_positives_per_cell
            + self.high_overlap_structural_negatives_per_cell
        )


@dataclass(frozen=True, slots=True)
class FootprintAuditSpec:
    classes: tuple[FootprintClass, ...]
    class_definitions: tuple[tuple[FootprintClass, str], ...]
    minimum_classes_per_family_scale_cell: int
    minimum_shared_nonconstant_measurements: int
    minimum_correct_shared_fraction: float
    minimum_competitor_shared_fraction: float
    maximum_footprint_size_ratio: float
    shared_structural_witness_required: bool
    candidate_private_measurement_count_maximum: int
    grouped_permutation_replicates: int
    grouped_permutation_strata: tuple[str, ...]
    maximum_single_group_share: float
    maximum_best_single_measurement_balanced_accuracy: float

    def __post_init__(self) -> None:
        if self.classes != tuple(FootprintClass):
            raise ValueError("shared-footprint taxonomy drift")
        if tuple(item[0] for item in self.class_definitions) != self.classes:
            raise ValueError("every footprint class needs one frozen definition")
        if self.minimum_classes_per_family_scale_cell != 3:
            raise ValueError("each family-scale cell needs at least three classes")
        if self.grouped_permutation_replicates != 1_000:
            raise ValueError("grouped permutation replicate count drift")
        if self.grouped_permutation_strata != ("case_type", "scale"):
            raise ValueError("grouped permutation strata drift")
        if (
            self.minimum_shared_nonconstant_measurements,
            self.minimum_correct_shared_fraction,
            self.minimum_competitor_shared_fraction,
            self.maximum_footprint_size_ratio,
            self.shared_structural_witness_required,
            self.candidate_private_measurement_count_maximum,
        ) != (2, 0.60, 0.60, 3.0, True, 0):
            raise ValueError("shared-footprint candidate gate drift")
        if (
            self.maximum_single_group_share,
            self.maximum_best_single_measurement_balanced_accuracy,
        ) != (0.50, 0.50):
            raise ValueError("single-measurement dominance gates drift")


@dataclass(frozen=True, slots=True)
class RerunPolicy:
    allowed_reexecution_reasons: tuple[str, ...]
    upload_only_retry_reason: str
    forbidden_reexecution_reasons: tuple[str, ...]
    maximum_reexecutions: int
    every_attempt_permanently_recorded: bool
    any_valid_prediction_byte_makes_formal_attempt: bool

    def __post_init__(self) -> None:
        _require_nonempty_tuple(
            self.allowed_reexecution_reasons, "allowed rerun reasons"
        )
        _require_nonempty_tuple(
            self.forbidden_reexecution_reasons, "forbidden rerun reasons"
        )
        if set(self.allowed_reexecution_reasons).intersection(
            self.forbidden_reexecution_reasons
        ):
            raise ValueError("rerun reason cannot be both allowed and forbidden")
        if self.maximum_reexecutions != 2:
            raise ValueError("v1 allows at most two reexecutions")
        if not (
            self.every_attempt_permanently_recorded
            and self.any_valid_prediction_byte_makes_formal_attempt
        ):
            raise ValueError("attempt recording and prediction-byte boundary drift")

    def permits_reexecution(
        self,
        reason: str,
        *,
        any_valid_prediction_byte_produced: bool,
    ) -> bool:
        if type(any_valid_prediction_byte_produced) is not bool:
            raise TypeError("prediction-byte flag must be boolean")
        return bool(
            not any_valid_prediction_byte_produced
            and reason in self.allowed_reexecution_reasons
        )

    def retry_action(
        self,
        reason: str,
        *,
        any_valid_prediction_byte_produced: bool,
    ) -> str:
        """Return the only preregistered action for a failed attempt."""

        if self.permits_reexecution(
            reason,
            any_valid_prediction_byte_produced=any_valid_prediction_byte_produced,
        ):
            return "REEXECUTE"
        if reason == self.upload_only_retry_reason:
            return "REUPLOAD_COMMITTED_OUTPUT"
        return "FORBIDDEN"


@dataclass(frozen=True, slots=True)
class ValidationVersionPolicy:
    attempts_per_version: int
    maximum_validation_versions_before_no_go: int
    failed_validation_becomes_public_development_evidence: bool
    version_fields_required_after_change: tuple[str, ...]
    sealed_holdout_only_after_validation_pass: bool
    fresh_independent_seed_required_for_v2: bool
    v3_forbidden_after_v2_failure_without_external_review: bool

    def __post_init__(self) -> None:
        if (
            self.attempts_per_version,
            self.maximum_validation_versions_before_no_go,
        ) != (2, 2):
            raise ValueError("validation retry/version limits drift")
        if not (
            self.failed_validation_becomes_public_development_evidence
            and self.sealed_holdout_only_after_validation_pass
            and self.fresh_independent_seed_required_for_v2
            and self.v3_forbidden_after_v2_failure_without_external_review
        ):
            raise ValueError("validation lifecycle must fail closed")
        if self.version_fields_required_after_change != (
            "protocol_version",
            "selector_version",
            "validation_version",
        ):
            raise ValueError("validation version bump set drift")


@dataclass(frozen=True, slots=True)
class CovertChannelAuditSpec:
    independent_secret_keys: tuple[str, ...]
    keys_separate_from: tuple[str, ...]
    id_assignment_after_global_shuffle: bool
    id_algorithm: str
    id_bits: int
    collision_max_retries: int
    collision_warning_threshold: int
    provenance_commits_public_payload_only: bool
    field_classes: tuple[str, ...]
    channel_targets: tuple[str, ...]
    audit_tests: tuple[str, ...]
    permutation_strata: tuple[tuple[str, tuple[str, ...]], ...]
    label_permutations: int
    multiple_testing: str
    family_wise_alpha: float
    minimum_adjusted_p: float
    maximum_normalized_mutual_information: float
    maximum_balanced_accuracy_advantage: float
    unique_id_feature_family: tuple[str, ...]
    renaming_namespaces: tuple[str, ...]
    global_consistent_renamings: int
    renaming_invariants: tuple[str, ...]
    global_case_order_permutations: int
    within_case_observation_order_permutations: int
    canonical_json: str
    envelope_bytes: int
    missingness_representation: str
    per_case_unused_transform_list_allowed: bool
    recognizer_receives_stdin_only: bool

    def __post_init__(self) -> None:
        _require_nonempty_tuple(self.independent_secret_keys, "audit keys")
        _require_nonempty_tuple(self.keys_separate_from, "key separation sources")
        _require_nonempty_tuple(self.channel_targets, "channel targets")
        _require_nonempty_tuple(self.audit_tests, "channel audit tests")
        _require_nonempty_tuple(self.permutation_strata, "permutation strata")
        _require_nonempty_tuple(self.renaming_namespaces, "renaming namespaces")
        if self.independent_secret_keys != ("K_shuffle", "K_id", "K_padding"):
            raise ValueError("wire keys drift")
        if set(self.independent_secret_keys).intersection(self.keys_separate_from):
            raise ValueError("wire and answer/generator keys must be independent")
        if not self.id_assignment_after_global_shuffle:
            raise ValueError("public IDs must be assigned after global shuffle")
        if self.collision_warning_threshold != 0:
            raise ValueError("any public-ID collision must trigger an audit warning")
        if not self.provenance_commits_public_payload_only:
            raise ValueError("public provenance cannot commit to private answers")
        if self.field_classes != ("EVIDENCE_BEARING", "CHANNEL_ONLY"):
            raise ValueError("public field classification drift")
        expected_scalars = (
            self.id_algorithm,
            self.id_bits,
            self.collision_max_retries,
            self.label_permutations,
            self.multiple_testing,
            self.family_wise_alpha,
            self.minimum_adjusted_p,
            self.maximum_normalized_mutual_information,
            self.maximum_balanced_accuracy_advantage,
            self.global_consistent_renamings,
            self.global_case_order_permutations,
            self.within_case_observation_order_permutations,
            self.canonical_json,
            self.envelope_bytes,
        )
        if expected_scalars != (
            "HMAC-SHA256_truncate_128_set_UUIDv4_bits",
            128,
            10,
            10_000,
            "Holm-Bonferroni",
            0.01,
            0.01,
            0.02,
            0.05,
            32,
            32,
            16,
            "RFC-8785",
            65_536,
        ):
            raise ValueError("covert-channel audit configuration drift")
        if self.missingness_representation != (
            '{"available":false,"value":null}'
        ):
            raise ValueError("missingness representation drift")
        if self.per_case_unused_transform_list_allowed:
            raise ValueError("per-case unused transforms are a forbidden channel")
        if not self.recognizer_receives_stdin_only:
            raise ValueError("recognizer must not see filenames or case indices")
        if tuple(target for target, _ in self.permutation_strata) != (
            "family",
            "scale",
            "binding",
            "answerable",
            "joint_class",
        ):
            raise ValueError("channel permutation strata target order drift")
        if self.renaming_invariants != (
            "decision_identical",
            "family_identical",
            "binding_identical",
            "scale_set_identical",
            "candidate_residual_vector_bit_identical",
        ):
            raise ValueError("consistent-renaming invariants drift")
        if self.audit_tests != (
            "byte_prefix_suffix_features",
            "normalized_mutual_information",
            "one_field_predictive_auditor",
            "label_permutation_test",
        ):
            raise ValueError("channel-only audit test set drift")
        if self.permutation_strata != (
            ("family", ("case_type", "scale")),
            ("scale", ("case_type", "family")),
            ("binding", ("case_type", "family", "scale")),
            ("answerable", ("family", "scale")),
            ("joint_class", ("case_type",)),
        ):
            raise ValueError("channel permutation strata drift")


@dataclass(frozen=True, slots=True)
class FormalUncertaintyPolicy:
    allowed_kinds: tuple[FormalUncertaintyKind, ...]
    standard_error_status: str
    standard_error_required_semantics: tuple[str, ...]
    simultaneous_interval_family_wise_coverage: float
    endpoint_rounding: str

    def __post_init__(self) -> None:
        if self.allowed_kinds != (FormalUncertaintyKind.ABSOLUTE_BOUND,):
            raise ValueError("formal v1 selector must be absolute-bound only")
        if self.standard_error_status != "STANDARD_ERROR_UNSUPPORTED":
            raise ValueError("standard-error status must fail closed")
        if self.standard_error_required_semantics != (
            "sample_count_n_gte_3",
            "sampling_unit_id",
            "estimator_sample_mean",
            "independent_replicates_true",
            "finite_variance_assumption_true",
            "distribution_model_student_t_iid",
        ):
            raise ValueError("standard-error semantic prerequisites drift")
        if self.simultaneous_interval_family_wise_coverage != 0.99:
            raise ValueError("standard-error interval coverage drift")
        if self.endpoint_rounding != "outward_to_frozen_RationalValue_grid":
            raise ValueError("uncertainty endpoint rounding drift")


@dataclass(frozen=True, slots=True)
class Phase2BExactFreeze:
    freeze_version: str
    family_mapping: tuple[tuple[LawKind, CanonicalFamilyId], ...]
    holdout: HoldoutAllocationFreeze
    preservation_rules: tuple[PreservationRule, ...]
    baselines: tuple[BaselineSpec, ...]
    bootstrap: BootstrapSpec
    semantic_conflict: SemanticConflictChallenge
    footprint_audit: FootprintAuditSpec
    rerun_policy: RerunPolicy
    validation_policy: ValidationVersionPolicy
    covert_channel_audit: CovertChannelAuditSpec
    uncertainty_policy: FormalUncertaintyPolicy
    formal_holdout_generated: bool
    formal_holdout_consumed: bool
    formal_holdout_generation_authorized: bool
    shadow_only: bool
    implementation_blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.freeze_version != PHASE2B_EXACT_FREEZE_VERSION:
            raise ValueError("exact freeze version drift")
        _require_nonempty_tuple(self.family_mapping, "family mapping")
        _require_nonempty_tuple(self.preservation_rules, "preservation rules")
        _require_nonempty_tuple(self.baselines, "baselines")
        _require_nonempty_tuple(self.implementation_blockers, "implementation blockers")
        if tuple(kind for kind, _ in self.family_mapping) != tuple(LawKind):
            raise ValueError("family mapping must cover every LawKind once")
        if tuple(family for _, family in self.family_mapping) != (
            CanonicalFamilyId.F01,
            CanonicalFamilyId.F02,
            CanonicalFamilyId.F03,
            CanonicalFamilyId.F04,
            CanonicalFamilyId.F06,
            CanonicalFamilyId.F05,
        ):
            raise ValueError("LawKind-to-canonical-family mapping drift")
        if self.holdout.independent_latent_case_count != 720:
            raise ValueError("main holdout must contain 720 independent cases")
        expected_applicability = {
            PreservationTransform.ENTITY_ALPHA_RENAMING: tuple(CanonicalFamilyId),
            PreservationTransform.OBSERVATION_REORDER: (
                CanonicalFamilyId.F01,
                CanonicalFamilyId.F03,
                CanonicalFamilyId.F04,
                CanonicalFamilyId.F05,
            ),
            PreservationTransform.IRRELEVANT_ENTITY_AUGMENTATION: tuple(
                CanonicalFamilyId
            ),
            PreservationTransform.UNIT_CONVERSION: tuple(CanonicalFamilyId),
            PreservationTransform.COORDINATE_AFFINE_TRANSFORM: (
                CanonicalFamilyId.F01,
                CanonicalFamilyId.F02,
                CanonicalFamilyId.F05,
                CanonicalFamilyId.F06,
            ),
            PreservationTransform.EQUIVALENT_AGGREGATION_SPLIT_MERGE: (
                CanonicalFamilyId.F03,
                CanonicalFamilyId.F04,
                CanonicalFamilyId.F05,
            ),
            PreservationTransform.NONTRIVIAL_SCALE_MAP: tuple(CanonicalFamilyId),
            PreservationTransform.SIGN_CONVENTION_REPARAMETERIZATION: (
                CanonicalFamilyId.F02,
                CanonicalFamilyId.F03,
                CanonicalFamilyId.F06,
            ),
        }
        if {
            rule.transform: rule.applicable_families
            for rule in self.preservation_rules
        } != expected_applicability:
            raise ValueError("preservation applicability matrix drift")
        if self.legal_preservation_pair_count != 496:
            raise ValueError("legal preservation matrix must derive 496 pairs")
        if self.invalid_transform_control_count != 76:
            raise ValueError("invalid-transform matrix must derive 76 controls")
        if self.total_preservation_sensitivity_pair_count != 572:
            raise ValueError("preservation/sensitivity total must be 572")
        if self.semantic_conflict.case_count != 240:
            raise ValueError("semantic challenge count drift")
        if any(
            (
                self.formal_holdout_generated,
                self.formal_holdout_consumed,
                self.formal_holdout_generation_authorized,
            )
        ):
            raise ValueError("this freeze is not a formal holdout run or authorization")
        if not self.shadow_only:
            raise ValueError("Phase 2-3 must remain shadow-only")
        expected_components = _canonical_phase2b_exact_freeze_components()
        for field_name, expected_value in expected_components.items():
            if not _exact_frozen_value_equal(
                getattr(self, field_name),
                expected_value,
            ):
                raise ValueError(
                    f"{field_name} differs from the exact Phase-2B decision"
                )

    @property
    def case_type_totals(self) -> tuple[tuple[str, int], ...]:
        return self.holdout.case_type_totals

    @property
    def margin_stratum_totals(self) -> tuple[tuple[str, int], ...]:
        return self.holdout.margin_stratum_totals

    @property
    def legal_preservation_pair_count(self) -> int:
        return sum(rule.legal_pair_count for rule in self.preservation_rules)

    @property
    def invalid_transform_control_count(self) -> int:
        return sum(rule.invalid_control_count for rule in self.preservation_rules)

    @property
    def total_preservation_sensitivity_pair_count(self) -> int:
        return self.legal_preservation_pair_count + self.invalid_transform_control_count

    @property
    def freeze_id(self) -> str:
        return stable_hash(self, prefix="phase2b_exact_freeze_")


def _canonical_phase2b_exact_freeze_components() -> dict[str, object]:
    """Build the sole accepted field values for the exact Phase-2B freeze."""

    all_families = tuple(CanonicalFamilyId)
    holdout = HoldoutAllocationFreeze(
        family_count=6,
        scale_count=2,
        cases_per_cell=60,
        case_quota_per_cell=(
            ("unique_scale_answerable", 19),
            ("admissible_scale_set_answerable", 1),
            ("wrong_family_hard_negative", 8),
            ("binding_counterfactual", 8),
            ("scale_counterfactual", 8),
            ("sign_or_invariant_break", 8),
            ("insufficient_or_nonidentifiable", 8),
        ),
        margin_quota_per_cell=(
            ("clear_interior", 21),
            ("moderate", 18),
            ("near_boundary_identifiable", 12),
            ("nonunique_or_insufficient", 9),
        ),
        margin_case_joint_quota_per_cell=(
            (
                "nonunique_or_insufficient",
                "insufficient_or_nonidentifiable",
                8,
            ),
            (
                "nonunique_or_insufficient",
                "admissible_scale_set_answerable",
                1,
            ),
        ),
        metric_denominators=(
            MetricDenominator(
                "answerable_count",
                ("unique_scale_answerable", "admissible_scale_set_answerable"),
                240,
            ),
            MetricDenominator(
                "family_exact_accuracy",
                ("unique_scale_answerable", "admissible_scale_set_answerable"),
                240,
            ),
            MetricDenominator(
                "binding_exact_accuracy",
                ("unique_scale_answerable", "admissible_scale_set_answerable"),
                240,
            ),
            MetricDenominator(
                "scale_set_accuracy",
                ("unique_scale_answerable", "admissible_scale_set_answerable"),
                240,
            ),
            MetricDenominator(
                "unique_scale_accuracy", ("unique_scale_answerable",), 228
            ),
            MetricDenominator(
                "joint_exact_accuracy",
                ("unique_scale_answerable", "admissible_scale_set_answerable"),
                240,
            ),
            MetricDenominator(
                "abstention_specificity", ("unique_scale_answerable",), 228
            ),
            MetricDenominator(
                "nonidentifiability_abstention_accuracy",
                ("insufficient_or_nonidentifiable",),
                96,
            ),
            MetricDenominator(
                "set_valued_answer_accuracy",
                ("admissible_scale_set_answerable",),
                12,
                separately_reported=True,
            ),
        ),
        set_valued_joint_rule=(
            "family_exact_and_binding_exact_and_scale_set_exact_and_ANSWER_SET"
        ),
    )
    preservation = (
        PreservationRule(
            PreservationTransform.ENTITY_ALPHA_RENAMING,
            all_families,
            legal_pairs_per_family_scale=6,
        ),
        PreservationRule(
            PreservationTransform.OBSERVATION_REORDER,
            (
                CanonicalFamilyId.F01,
                CanonicalFamilyId.F03,
                CanonicalFamilyId.F04,
                CanonicalFamilyId.F05,
            ),
            legal_pairs_per_family_scale=6,
        ),
        PreservationRule(
            PreservationTransform.IRRELEVANT_ENTITY_AUGMENTATION,
            all_families,
            legal_pairs_per_family_scale=6,
        ),
        PreservationRule(
            PreservationTransform.UNIT_CONVERSION,
            all_families,
            legal_pairs_per_family_scale=8,
        ),
        PreservationRule(
            PreservationTransform.COORDINATE_AFFINE_TRANSFORM,
            (
                CanonicalFamilyId.F01,
                CanonicalFamilyId.F02,
                CanonicalFamilyId.F05,
                CanonicalFamilyId.F06,
            ),
            legal_pairs_per_family_scale=8,
        ),
        PreservationRule(
            PreservationTransform.EQUIVALENT_AGGREGATION_SPLIT_MERGE,
            (
                CanonicalFamilyId.F03,
                CanonicalFamilyId.F04,
                CanonicalFamilyId.F05,
            ),
            legal_pairs_per_family_scale=8,
        ),
        PreservationRule(
            PreservationTransform.NONTRIVIAL_SCALE_MAP,
            all_families,
            legal_pairs_per_family=10,
        ),
        PreservationRule(
            PreservationTransform.SIGN_CONVENTION_REPARAMETERIZATION,
            (
                CanonicalFamilyId.F02,
                CanonicalFamilyId.F03,
                CanonicalFamilyId.F06,
            ),
            legal_pairs_per_family_scale=6,
        ),
    )
    semantic_prompt = """You are a semantic-only structural-label baseline.
You receive one canonical evidence description.
Do not execute equations, call a verifier, enumerate candidate programs,
or use hidden metadata.
Return exactly one JSON object:
{
  "family": "<one allowed family id or ABSTAIN>",
  "binding": "<one candidate binding id or ABSTAIN>",
  "scale": ["<zero, one, or multiple allowed scale ids>"],
  "decision": "ANSWER | ANSWER_SET | ABSTAIN"
}
Use only the visible wording and surface associations in the evidence."""
    baselines = (
        BaselineSpec(
            "embedding_nearest_prototype",
            "sentence-transformers/all-mpnet-base-v2",
            "exact_40_hex_commit_required",
            (
                ("pooling", "model_default_mean_pooling"),
                ("normalization", "l2"),
                ("similarity", "cosine"),
                ("input", "canonical_public_evidence_text"),
                ("prototype", "development-family-centroid"),
                ("no_verifier_access", True),
            ),
        ),
        BaselineSpec(
            "frozen_llm_semantic_only",
            "Qwen/Qwen2.5-7B-Instruct",
            "exact_40_hex_commit_required",
            (
                ("do_sample", False),
                ("temperature", 0.0),
                ("top_p", 1.0),
                ("max_new_tokens", 128),
                ("seed", 0),
                ("tool_access", False),
                ("verifier_access", False),
            ),
            prompt=semantic_prompt,
        ),
        BaselineSpec(
            "flat_learned_typed",
            "sklearn.ensemble.HistGradientBoostingClassifier",
            "repository_dependency_lock_required",
            (
                ("learning_rate", 0.05),
                ("max_iter", 200),
                ("max_leaf_nodes", 15),
                ("max_depth", 3),
                ("min_samples_leaf", 20),
                ("l2_regularization", 1.0),
                ("early_stopping", False),
                ("random_state", BOOTSTRAP_DERIVED_UINT32_SEED),
                ("holdout_adjustment_allowed", False),
            ),
            output_heads=(
                "family",
                "binding",
                "scale_set_class",
                "answer_vs_abstain",
            ),
        ),
    )
    covert_audit = CovertChannelAuditSpec(
        independent_secret_keys=("K_shuffle", "K_id", "K_padding"),
        keys_separate_from=("latent_generator_seed", "answer_seed"),
        id_assignment_after_global_shuffle=True,
        id_algorithm="HMAC-SHA256_truncate_128_set_UUIDv4_bits",
        id_bits=128,
        collision_max_retries=10,
        collision_warning_threshold=0,
        provenance_commits_public_payload_only=True,
        field_classes=("EVIDENCE_BEARING", "CHANNEL_ONLY"),
        channel_targets=(
            "family",
            "binding",
            "scale",
            "answerable_vs_abstain",
            "joint_decision_class",
        ),
        audit_tests=(
            "byte_prefix_suffix_features",
            "normalized_mutual_information",
            "one_field_predictive_auditor",
            "label_permutation_test",
        ),
        permutation_strata=(
            ("family", ("case_type", "scale")),
            ("scale", ("case_type", "family")),
            ("binding", ("case_type", "family", "scale")),
            ("answerable", ("family", "scale")),
            ("joint_class", ("case_type",)),
        ),
        label_permutations=10_000,
        multiple_testing="Holm-Bonferroni",
        family_wise_alpha=0.01,
        minimum_adjusted_p=0.01,
        maximum_normalized_mutual_information=0.02,
        maximum_balanced_accuracy_advantage=0.05,
        unique_id_feature_family=(
            "128_individual_bits",
            "first_8_16_32_bits",
            "last_8_16_32_bits",
            "hamming_weight",
            "integer_mod_3_5_7_11_13",
            "hex_character_histogram",
        ),
        renaming_namespaces=(
            "bundle_id",
            "observation_id",
            "entity_id",
            "role_candidate_id",
            "quantity_id",
            "context_id",
            "task_id",
            "scale_id",
            "aggregate_map_id",
            "transform_id",
        ),
        global_consistent_renamings=32,
        renaming_invariants=(
            "decision_identical",
            "family_identical",
            "binding_identical",
            "scale_set_identical",
            "candidate_residual_vector_bit_identical",
        ),
        global_case_order_permutations=32,
        within_case_observation_order_permutations=16,
        canonical_json="RFC-8785",
        envelope_bytes=65_536,
        missingness_representation='{"available":false,"value":null}',
        per_case_unused_transform_list_allowed=False,
        recognizer_receives_stdin_only=True,
    )
    return {
        "freeze_version": PHASE2B_EXACT_FREEZE_VERSION,
        "family_mapping": (
            (LawKind.SYMMETRY, CanonicalFamilyId.F01),
            (LawKind.MONOTONICITY, CanonicalFamilyId.F02),
            (LawKind.CONSERVATION, CanonicalFamilyId.F03),
            (LawKind.COMPLEMENTARITY, CanonicalFamilyId.F04),
            (LawKind.NEGATIVE_FEEDBACK, CanonicalFamilyId.F06),
            (LawKind.LOCALITY, CanonicalFamilyId.F05),
        ),
        "holdout": holdout,
        "preservation_rules": preservation,
        "baselines": baselines,
        "bootstrap": BootstrapSpec(
            method="paired_cluster_bootstrap",
            replicates=10_000,
            seed=BOOTSTRAP_MASTER_SEED,
            seed_derivation_id=BOOTSTRAP_UINT32_DERIVATION_ID,
            derived_uint32_seed=BOOTSTRAP_DERIVED_UINT32_SEED,
            resampling_unit="latent_base_case",
            cluster_members=(
                "original_case",
                "all_preservation_variants",
                "all_baseline_predictions",
            ),
            interval="one_sided_95_percent_percentile",
        ),
        "semantic_conflict": SemanticConflictChallenge(),
        "footprint_audit": FootprintAuditSpec(
            classes=tuple(FootprintClass),
            class_definitions=(
                (FootprintClass.P2_PAIR, "2_shared_nonconstant_measurements"),
                (
                    FootprintClass.P3_CHAIN,
                    "3_measurements_form_directed_or_ordered_chain",
                ),
                (
                    FootprintClass.P4_STAR,
                    "4_or_more_measurements_share_central_entity_or_quantity",
                ),
                (
                    FootprintClass.PSET_AGGREGATE,
                    "5_to_8_set_measurements_used_by_aggregation",
                ),
            ),
            minimum_classes_per_family_scale_cell=3,
            minimum_shared_nonconstant_measurements=2,
            minimum_correct_shared_fraction=0.60,
            minimum_competitor_shared_fraction=0.60,
            maximum_footprint_size_ratio=3.0,
            shared_structural_witness_required=True,
            candidate_private_measurement_count_maximum=0,
            grouped_permutation_replicates=1_000,
            grouped_permutation_strata=("case_type", "scale"),
            maximum_single_group_share=0.50,
            maximum_best_single_measurement_balanced_accuracy=0.50,
        ),
        "rerun_policy": RerunPolicy(
            allowed_reexecution_reasons=(
                "CONTAINER_START_FAILURE",
                "HOST_OOM_BEFORE_FIRST_PREDICTION_BYTE",
                "HOST_TERMINATION_BEFORE_FIRST_PREDICTION_BYTE",
                "INPUT_TRANSFER_CHECKSUM_MISMATCH",
            ),
            upload_only_retry_reason="OUTPUT_UPLOAD_FAILURE_AFTER_OUTPUT_HASH_COMMITTED",
            forbidden_reexecution_reasons=(
                "MODEL_EXCEPTION",
                "PARSER_FAILURE",
                "MISSING_CASE_OUTPUT",
                "VERIFIER_EXCEPTION",
                "LOW_COVERAGE",
                "NONDETERMINISTIC_OUTPUT",
                "TIMEOUT_AFTER_ANY_PREDICTION_BYTE",
            ),
            maximum_reexecutions=2,
            every_attempt_permanently_recorded=True,
            any_valid_prediction_byte_makes_formal_attempt=True,
        ),
        "validation_policy": ValidationVersionPolicy(
            attempts_per_version=2,
            maximum_validation_versions_before_no_go=2,
            failed_validation_becomes_public_development_evidence=True,
            version_fields_required_after_change=(
                "protocol_version",
                "selector_version",
                "validation_version",
            ),
            sealed_holdout_only_after_validation_pass=True,
            fresh_independent_seed_required_for_v2=True,
            v3_forbidden_after_v2_failure_without_external_review=True,
        ),
        "covert_channel_audit": covert_audit,
        "uncertainty_policy": FormalUncertaintyPolicy(
            allowed_kinds=(FormalUncertaintyKind.ABSOLUTE_BOUND,),
            standard_error_status="STANDARD_ERROR_UNSUPPORTED",
            standard_error_required_semantics=(
                "sample_count_n_gte_3",
                "sampling_unit_id",
                "estimator_sample_mean",
                "independent_replicates_true",
                "finite_variance_assumption_true",
                "distribution_model_student_t_iid",
            ),
            simultaneous_interval_family_wise_coverage=0.99,
            endpoint_rounding="outward_to_frozen_RationalValue_grid",
        ),
        "formal_holdout_generated": False,
        "formal_holdout_consumed": False,
        "formal_holdout_generation_authorized": False,
        "shadow_only": True,
        "implementation_blockers": (
            "formal_wire_builder_and_covert_channel_auditor_not_implemented",
            "exact_baseline_revisions_and_artifact_hashes_not_registered",
            "independent_holdout_generator_and_validation_artifacts_not_implemented",
            "functional_recognizer_cli_signed_minimal_image_and_archive_evaluator_not_implemented",
            "durable_signed_custodian_cas_ledger_not_implemented",
        ),
    }


def frozen_phase2b_exact_freeze() -> Phase2BExactFreeze:
    """Return the exact Phase-2B v1 contract without claiming execution."""

    return Phase2BExactFreeze(**_canonical_phase2b_exact_freeze_components())


def canonical_family_id(law_kind: LawKind) -> CanonicalFamilyId:
    """Map an internal v1 ``LawKind`` to its frozen public family identity."""

    if not isinstance(law_kind, LawKind):
        raise TypeError("law_kind must be a LawKind")
    return dict(frozen_phase2b_exact_freeze().family_mapping)[law_kind]


def _require_nonempty_tuple(value: object, name: str) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be an immutable tuple")
    if not value:
        raise ValueError(f"{name} cannot be empty")


__all__ = (
    "BOOTSTRAP_DERIVED_UINT32_SEED",
    "BOOTSTRAP_MASTER_SEED",
    "BOOTSTRAP_SEED",
    "BOOTSTRAP_UINT32_DERIVATION_ID",
    "BaselineSpec",
    "BootstrapSpec",
    "CanonicalFamilyId",
    "CovertChannelAuditSpec",
    "FootprintAuditSpec",
    "FootprintClass",
    "FormalUncertaintyKind",
    "FormalUncertaintyPolicy",
    "HoldoutAllocationFreeze",
    "MetricDenominator",
    "PHASE2B_EXACT_FREEZE_VERSION",
    "Phase2BExactFreeze",
    "PreservationRule",
    "PreservationTransform",
    "RerunPolicy",
    "SemanticConflictChallenge",
    "ValidationVersionPolicy",
    "canonical_family_id",
    "derive_bootstrap_uint32_seed",
    "frozen_phase2b_exact_freeze",
)
