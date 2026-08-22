"""Pure candidate mechanics for Phase-2B C1 statistical validation.

This module closes a prospective statistical-validation profile and verifies
caller-supplied candidate graphs entirely in memory.  It implements exact
cohort/quota closure, the currently frozen twelve overall and twenty-four
slice gates, deterministic paired-cluster bootstrap indices, a two-attempt
lifecycle, six-purpose Ed25519 verification, candidate CAS ``0 -> 1``
receipts, and a contiguous declared-time chain.

The implementation intentionally has no final config or final Wire/R06 pin.
It does not read or write files, call a clock, use an RNG, generate keys or
signatures, invoke Docker or a network, persist CAS state, execute a system,
or establish R06/R07 authority.  Even a structurally complete PASS is only
``STATISTICAL_VALIDATION_CANDIDATE_MECHANICS_VERIFIED_NOT_R08``.
"""

from __future__ import annotations

import copy
from dataclasses import MISSING, dataclass, fields
from enum import Enum
from fractions import Fraction
import hashlib
import hmac
import math
from typing import Callable, Final, NoReturn

from .rfc8785_v1 import (
    RFC8785Error,
    canonicalize_rfc8785_v1,
    decode_and_require_canonical_rfc8785_v1,
)

try:  # Verification only.  Absence is a fail-closed candidate rejection.
    from cryptography.exceptions import InvalidSignature as _InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey as _Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - exercised by backend substitution.
    _InvalidSignature = Exception
    _Ed25519PublicKey = None


PHASE2B_C1_STATISTICAL_VALIDATION_PROFILE_V1_SCHEMA: Final = (
    "hegel-machine-phase2b-c1-statistical-validation-profile/1"
)
PHASE2B_C1_STATISTICAL_VALIDATION_MECHANICS_V1_VERSION: Final = (
    "hegel-machine-phase2b-c1-statistical-validation-candidate-mechanics/1"
)
PHASE2B_C1_STATISTICAL_VALIDATION_MECHANICS_V1_CLAIM_LEVEL: Final = (
    "STATISTICAL_VALIDATION_CANDIDATE_MECHANICS_VERIFIED_NOT_R08"
)
PHASE2B_C1_STATISTICAL_VALIDATION_V1_ED25519_BACKEND_AVAILABLE: Final = (
    _Ed25519PublicKey is not None
)

_PROFILE_DOMAIN: Final = b"HEGEL/PHASE2B/C1/STATISTICAL_VALIDATION/PROFILE/V1\x00"
_PROFILE_PREFIX: Final = "phase2b_c1_statistical_validation_profile_v1_"
_SEED_DOMAIN: Final = b"HEGEL/PHASE2B/C1/STATISTICAL_VALIDATION/SEED_ROOT/V1\x00"
_KEY_DOMAIN: Final = b"HEGEL/PHASE2B/C1/STATISTICAL_VALIDATION/KEY/V1\x00"
_KEY_PREFIX: Final = "phase2b_c1_statistical_validation_candidate_key_v1_"
_COHORT_DOMAIN: Final = b"HEGEL/PHASE2B/C1/STATISTICAL_VALIDATION/COHORT/V1\x00"
_SYSTEM_RUNS_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/STATISTICAL_VALIDATION/SYSTEM_RUNS/V1\x00"
)
_METRIC_BUNDLE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/STATISTICAL_VALIDATION/METRIC_BUNDLE/V1\x00"
)
_EVENT_DOMAIN: Final = b"HEGEL/PHASE2B/C1/STATISTICAL_VALIDATION/EVENT/V1\x00"
_EVENT_SIGNATURE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/STATISTICAL_VALIDATION/EVENT_SIGNATURE/V1\x00"
)
_EVENT_PREFIX: Final = "phase2b_c1_statistical_validation_event_v1_"
_CAS_DOMAIN: Final = b"HEGEL/PHASE2B/C1/STATISTICAL_VALIDATION/CAS/V1\x00"
_CAS_SIGNATURE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/STATISTICAL_VALIDATION/CAS_SIGNATURE/V1\x00"
)
_CAS_PREFIX: Final = "phase2b_c1_statistical_validation_cas_candidate_v1_"
_ATTEMPT_ARCHIVE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/STATISTICAL_VALIDATION/ATTEMPT_ARCHIVE/V1\x00"
)
_INFRA_DOMAIN: Final = b"HEGEL/PHASE2B/C1/STATISTICAL_VALIDATION/INFRA/V1\x00"
_PREREQUISITE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/STATISTICAL_VALIDATION/PREREQUISITE_GRAPH/V1\x00"
)
_VERSION_BUMP_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/STATISTICAL_VALIDATION/VERSION_BUMP/V1\x00"
)
_VERSION_BUMP_PREFIX: Final = (
    "phase2b_c1_statistical_validation_version_bump_candidate_v1_"
)
_VERSION_BUMP_CORE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/STATISTICAL_VALIDATION/VERSION_BUMP_CORE/V1\x00"
)
_VERSION_BUMP_CORE_PREFIX: Final = (
    "phase2b_c1_statistical_validation_version_bump_core_v1_"
)
_VERSION_GATE_ROOT_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/STATISTICAL_VALIDATION/VERSION_GATE_ROOT/V1\x00"
)
_VERSION_ATTEMPT_CHAIN_ROOT_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/STATISTICAL_VALIDATION/VERSION_ATTEMPT_CHAIN_ROOT/V1\x00"
)
_RESULT_DOMAIN: Final = b"HEGEL/PHASE2B/C1/STATISTICAL_VALIDATION/RESULT/V1\x00"
_RESULT_PREFIX: Final = (
    "phase2b_c1_statistical_validation_candidate_mechanics_result_v1_"
)
_SNAPSHOT_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/STATISTICAL_VALIDATION/INPUT_SNAPSHOT/V1\x00"
)
_BOOTSTRAP_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/STATISTICAL_VALIDATION/PAIRED_CLUSTER_BOOTSTRAP/V1\x00"
)

_PRIMARY_COUNT: Final = 960
_MAIN_COUNT: Final = 720
_CHALLENGE_COUNT: Final = 240
_LEGAL_VARIANT_COUNT: Final = 496
_INVALID_VARIANT_COUNT: Final = 76
_VARIANT_COUNT: Final = 572
_ALL_SYSTEM_ROW_COUNT: Final = 1532
_BOOTSTRAP_REPLICATES: Final = 10_000
_BOOTSTRAP_MASTER_SEED: Final = 411_876_909_552_964_556
_BOOTSTRAP_LCB_INDEX: Final = 499
_BOOTSTRAP_UCB_INDEX: Final = 9_499
_WILSON_Z: Final = float.fromhex("0x1.a515209676ab8p+0")
_MAX_SAFE_INTEGER: Final = (1 << 53) - 1
_MAX_PROFILE_BYTES: Final = 262_144
_MAX_PREREQUISITE_BYTES: Final = 4_194_304
_MAX_SNAPSHOT_BYTES: Final = 20_971_520
_MAX_VERSION_ARTIFACT_BYTES: Final = 262_144
_MAX_TEXT_BYTES: Final = 4_096
_MAX_SHORT_ID_BYTES: Final = 256
_RAW_DIGEST_BYTES: Final = 32
_RAW_PUBLIC_KEY_BYTES: Final = 32
_RAW_SIGNATURE_BYTES: Final = 64
_EVENTS_PER_ATTEMPT: Final = 6
_MAX_RATIONAL_DENOMINATOR: Final = 1_024
_MAX_DECLARED_SYSTEM_OUTPUT_BYTES: Final = 17_179_869_184
_MAX_DECLARED_ATTEMPT_OUTPUT_BYTES: Final = 68_719_476_736

_PROTOCOL_VERSION_IDS: Final = (
    "phase2b_c1_statistical_validation_protocol_v1",
    "phase2b_c1_statistical_validation_protocol_v2",
)
_SELECTOR_VERSION_IDS: Final = (
    "phase2b_c1_statistical_validation_selector_v1",
    "phase2b_c1_statistical_validation_selector_v2",
)
_VALIDATION_VERSION_IDS: Final = (
    "phase2b_c1_statistical_validation_version_v1",
    "phase2b_c1_statistical_validation_version_v2",
)
_PUBLIC_FAILURE_DEVELOPMENT_REPORT_SCHEMA: Final = (
    "hegel-machine-phase2b-c1-public-failure-development-report/1"
)
_VERSION_TWO_ISSUANCE_AUTHORITY_SCHEMA: Final = (
    "hegel-machine-phase2b-c1-version-two-issuance-authority/1"
)

_FAMILIES: Final = (
    "F01_symmetry_equivariance",
    "F02_monotonicity_order",
    "F03_conservation_balance",
    "F04_additivity_complementarity",
    "F05_locality_composition",
    "F06_negative_feedback_stability",
)
_SCALES: Final = ("S0_FINE_LOCAL", "S1_COARSE_AGGREGATE")


class ValidationSeedPurposeV1(str, Enum):
    LATENT_GENERATOR = "latent_generator"
    ANSWER = "answer"
    SHUFFLE = "shuffle"
    IDENTIFIER = "identifier"
    PADDING = "padding"


class ValidationSystemV1(str, Enum):
    HEGEL_RECOGNIZER = "hegel_recognizer"
    EMBEDDING_NEAREST_PROTOTYPE = "embedding_nearest_prototype"
    FROZEN_LLM_SEMANTIC_ONLY = "frozen_llm_semantic_only"
    FLAT_LEARNED_TYPED = "flat_learned_typed"


class ValidationKeyPurposeV1(str, Enum):
    CUSTODIAN = "custodian"
    RUNTIME = "runtime"
    OBSERVER = "observer"
    EVALUATOR = "evaluator"
    STORE = "store"
    TIME = "time"


class ValidationEventKindV1(str, Enum):
    CUSTODIAN_COHORT_BOUND = "CUSTODIAN_COHORT_BOUND"
    RUNTIME_EXECUTION_RECORDED = "RUNTIME_EXECUTION_RECORDED"
    OBSERVER_OUTPUTS_BOUND = "OBSERVER_OUTPUTS_BOUND"
    EVALUATOR_METRICS_BOUND = "EVALUATOR_METRICS_BOUND"
    STORE_IF_ABSENT_RECORDED = "STORE_IF_ABSENT_RECORDED"
    ATTEMPT_FINALIZED = "ATTEMPT_FINALIZED"


class ValidationAttemptTerminalCodeV1(str, Enum):
    COMPLETE_PASS = "COMPLETE_PASS"
    COMPLETE_FAIL = "COMPLETE_FAIL"
    CONTAINER_START_FAILURE = "CONTAINER_START_FAILURE"
    HOST_OOM_BEFORE_FIRST_PREDICTION_BYTE = (
        "HOST_OOM_BEFORE_FIRST_PREDICTION_BYTE"
    )
    HOST_TERMINATION_BEFORE_FIRST_PREDICTION_BYTE = (
        "HOST_TERMINATION_BEFORE_FIRST_PREDICTION_BYTE"
    )
    INPUT_TRANSFER_CHECKSUM_MISMATCH = "INPUT_TRANSFER_CHECKSUM_MISMATCH"


class ValidationExecutionRetryActionV1(str, Enum):
    NO_RETRY = "NO_RETRY"
    REEXECUTE = "REEXECUTE"


class ValidationExecutionRetryReasonV1(str, Enum):
    NONE = "NONE"
    CONTAINER_START_FAILURE = "CONTAINER_START_FAILURE"
    HOST_OOM_BEFORE_FIRST_PREDICTION_BYTE = (
        "HOST_OOM_BEFORE_FIRST_PREDICTION_BYTE"
    )
    HOST_TERMINATION_BEFORE_FIRST_PREDICTION_BYTE = (
        "HOST_TERMINATION_BEFORE_FIRST_PREDICTION_BYTE"
    )
    INPUT_TRANSFER_CHECKSUM_MISMATCH = "INPUT_TRANSFER_CHECKSUM_MISMATCH"


class ValidationUploadRetryActionV1(str, Enum):
    NO_RETRY = "NO_RETRY"
    REUPLOAD_COMMITTED_OUTPUT = "REUPLOAD_COMMITTED_OUTPUT"


class ValidationUploadRetryReasonV1(str, Enum):
    NONE = "NONE"
    OUTPUT_UPLOAD_FAILURE_AFTER_OUTPUT_HASH_COMMITTED = (
        "OUTPUT_UPLOAD_FAILURE_AFTER_OUTPUT_HASH_COMMITTED"
    )


class ValidationMainCaseTypeV1(str, Enum):
    UNIQUE_SCALE_ANSWERABLE = "unique_scale_answerable"
    ADMISSIBLE_SCALE_SET_ANSWERABLE = "admissible_scale_set_answerable"
    WRONG_FAMILY_HARD_NEGATIVE = "wrong_family_hard_negative"
    BINDING_COUNTERFACTUAL = "binding_counterfactual"
    SCALE_COUNTERFACTUAL = "scale_counterfactual"
    SIGN_OR_INVARIANT_BREAK = "sign_or_invariant_break"
    INSUFFICIENT_OR_NONIDENTIFIABLE = "insufficient_or_nonidentifiable"


class ValidationChallengeKindV1(str, Enum):
    LOW_OVERLAP_STRUCTURAL_POSITIVE = "LOW_OVERLAP_STRUCTURAL_POSITIVE"
    HIGH_OVERLAP_STRUCTURAL_NEGATIVE = "HIGH_OVERLAP_STRUCTURAL_NEGATIVE"


class ValidationPreservationKindV1(str, Enum):
    LEGAL = "LEGAL"
    INVALID_CONTROL = "INVALID_CONTROL"


class ValidationPredictionDecisionV1(str, Enum):
    ANSWER = "ANSWER"
    ANSWER_SET = "ANSWER_SET"
    ABSTAIN = "ABSTAIN"


class StatisticalValidationOutcomeV1(str, Enum):
    COMPLETE_PASS = "COMPLETE_PASS"
    COMPLETE_FAIL = "COMPLETE_FAIL"
    REJECTED = "REJECTED"


class StatisticalValidationCandidateReasonV1(str, Enum):
    CANDIDATE_MECHANICS_VERIFIED = "CANDIDATE_MECHANICS_VERIFIED"
    WRONG_INPUT_TYPE = "WRONG_INPUT_TYPE"
    PROFILE_INVALID = "PROFILE_INVALID"
    STRUCTURAL_CLOSURE_INVALID = "STRUCTURAL_CLOSURE_INVALID"
    SIGNATURE_BACKEND_UNAVAILABLE = "SIGNATURE_BACKEND_UNAVAILABLE"
    SIGNATURE_INVALID = "SIGNATURE_INVALID"
    INPUT_MUTATED_DURING_VERIFICATION = "INPUT_MUTATED_DURING_VERIFICATION"
    RESOURCE_BUDGET_EXCEEDED = "RESOURCE_BUDGET_EXCEEDED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


_SEED_PURPOSES: Final = tuple(ValidationSeedPurposeV1)
_SYSTEMS: Final = tuple(ValidationSystemV1)
_KEY_PURPOSES: Final = tuple(ValidationKeyPurposeV1)
_EVENT_KINDS: Final = tuple(ValidationEventKindV1)
_EVENT_PURPOSES: Final = (
    ValidationKeyPurposeV1.CUSTODIAN,
    ValidationKeyPurposeV1.RUNTIME,
    ValidationKeyPurposeV1.OBSERVER,
    ValidationKeyPurposeV1.EVALUATOR,
    ValidationKeyPurposeV1.STORE,
    ValidationKeyPurposeV1.TIME,
)
_RETRYABLE_INFRA_CODES: Final = (
    ValidationAttemptTerminalCodeV1.CONTAINER_START_FAILURE,
    ValidationAttemptTerminalCodeV1.HOST_OOM_BEFORE_FIRST_PREDICTION_BYTE,
    ValidationAttemptTerminalCodeV1.HOST_TERMINATION_BEFORE_FIRST_PREDICTION_BYTE,
    ValidationAttemptTerminalCodeV1.INPUT_TRANSFER_CHECKSUM_MISMATCH,
)
_CASE_TYPE_CELL_COUNTS: Final = (
    (ValidationMainCaseTypeV1.UNIQUE_SCALE_ANSWERABLE, 19),
    (ValidationMainCaseTypeV1.ADMISSIBLE_SCALE_SET_ANSWERABLE, 1),
    (ValidationMainCaseTypeV1.WRONG_FAMILY_HARD_NEGATIVE, 8),
    (ValidationMainCaseTypeV1.BINDING_COUNTERFACTUAL, 8),
    (ValidationMainCaseTypeV1.SCALE_COUNTERFACTUAL, 8),
    (ValidationMainCaseTypeV1.SIGN_OR_INVARIANT_BREAK, 8),
    (ValidationMainCaseTypeV1.INSUFFICIENT_OR_NONIDENTIFIABLE, 8),
)
_ANSWERABLE_CASE_TYPES: Final = (
    ValidationMainCaseTypeV1.UNIQUE_SCALE_ANSWERABLE,
    ValidationMainCaseTypeV1.ADMISSIBLE_SCALE_SET_ANSWERABLE,
)
_CONTROL_CASE_TYPES: Final = (
    ValidationMainCaseTypeV1.WRONG_FAMILY_HARD_NEGATIVE,
    ValidationMainCaseTypeV1.BINDING_COUNTERFACTUAL,
    ValidationMainCaseTypeV1.SCALE_COUNTERFACTUAL,
    ValidationMainCaseTypeV1.SIGN_OR_INVARIANT_BREAK,
)

_OVERALL_GATE_SPECS: Final = (
    ("family_exact", 240, 90, 100, 86, 100),
    ("binding_exact", 240, 90, 100, 86, 100),
    ("scale_set_accuracy", 240, 87, 100, 82, 100),
    ("joint_exact", 240, 85, 100, 80, 100),
    ("hard_negative_rejection", 96, 95, 100, 90, 100),
    ("binding_counterfactual_rejection", 96, 95, 100, 90, 100),
    ("scale_counterfactual_rejection", 96, 93, 100, 88, 100),
    ("sign_or_invariant_break_rejection", 96, 95, 100, 90, 100),
    ("abstention_specificity", 228, 95, 100, 90, 100),
    ("fail_closed_rate", None, 1, 1, None, None),
    ("preservation_consistency", 572, 97, 100, 94, 100),
    ("nonidentifiable_scale_abstention", 96, 95, 100, 90, 100),
)
_SLICE_GATE_BASE_SPECS: Final = (
    ("answerable_joint_exact", 80, 100, 70, 100),
    ("all_control_rejection", 88, 100, 78, 100),
    ("abstention_specificity", 85, 100, 75, 100),
)


def _expected_profile_body_v1() -> dict[str, object]:
    overall = []
    for name, denominator, point_n, point_d, lower_n, lower_d in _OVERALL_GATE_SPECS:
        overall.append(
            {
                "metric": name,
                "denominator": denominator,
                "minimum_point": [point_n, point_d],
                "minimum_one_sided_wilson_lower": (
                    None if lower_n is None else [lower_n, lower_d]
                ),
            }
        )
    return {
        "bootstrap": {
            "cluster_count": 12,
            "hmac_sampler": "HMAC_SHA256_UINT32_UNBIASED_REJECTION",
            "lower_rank_index_zero_based": _BOOTSTRAP_LCB_INDEX,
            "master_seed_decimal": str(_BOOTSTRAP_MASTER_SEED),
            "paired_cluster_resampling": True,
            "replicates": _BOOTSTRAP_REPLICATES,
            "upper_rank_index_zero_based": _BOOTSTRAP_UCB_INDEX,
        },
        "canonicalization": "RFC8785_EXACT_BYTES_NO_TRAILING_LF",
        "challenge_gates": {
            "high_overlap_structural_negative": {
                "correct_minimum": 114,
                "denominator": 120,
                "minimum_one_sided_wilson_lower": [90, 100],
                "minimum_point": [95, 100],
            },
            "low_overlap_structural_positive": {
                "correct_minimum": 109,
                "denominator": 120,
                "minimum_one_sided_wilson_lower": [85, 100],
                "minimum_point": [90, 100],
            },
            "semantic_advantage": {
                "baseline_rule": "REPLICATE_WISE_MAX_EMBEDDING_AND_FROZEN_LLM",
                "minimum_bootstrap_lower": [10, 100],
                "minimum_point": [15, 100],
            },
        },
        "claim_boundary": {
            "claim_level": PHASE2B_C1_STATISTICAL_VALIDATION_MECHANICS_V1_CLAIM_LEVEL,
            "false_claims": {
                "actual_execution_verified": False,
                "authoritative_time_verified": False,
                "authority_verified": False,
                "cas_durability_verified": False,
                "docker_invoked": False,
                "entropy_used": False,
                "filesystem_accessed": False,
                "keys_generated": False,
                "network_accessed": False,
                "r06_qualified": False,
                "r07_frozen": False,
                "r08_satisfied": False,
                "result_object_unforgeable": False,
                "signatures_generated": False,
            },
            "fresh_original_graph_and_result_reverification_required": True,
        },
        "cohort": {
            "batches_in_order": [
                {
                    "batch": "PRIMARY_960",
                    "challenge": _CHALLENGE_COUNT,
                    "main": _MAIN_COUNT,
                    "total": _PRIMARY_COUNT,
                },
                {
                    "batch": "PRESERVATION_VARIANTS_572",
                    "invalid": _INVALID_VARIANT_COUNT,
                    "legal": _LEGAL_VARIANT_COUNT,
                    "total": _VARIANT_COUNT,
                },
            ],
            "rows_per_system": _ALL_SYSTEM_ROW_COUNT,
            "systems_in_order": [item.value for item in _SYSTEMS],
        },
        "lifecycle": {
            "attempt_one_exact_reuse": [
                "cohort_canonical_bytes",
                "five_seed_root_canonical_bytes",
            ],
            "attempt_indices": [0, 1],
            "cas_transition": [0, 1],
            "execution_retry_reason_by_terminal_code": {
                code.value: ValidationExecutionRetryReasonV1(code.value).value
                for code in _RETRYABLE_INFRA_CODES
            },
            "execution_retry_actions": [
                ValidationExecutionRetryActionV1.NO_RETRY.value,
                ValidationExecutionRetryActionV1.REEXECUTE.value,
            ],
            "event_kinds_in_order": [item.value for item in _EVENT_KINDS],
            "key_purposes_in_order": [item.value for item in _KEY_PURPOSES],
            "logical_time_contiguous": True,
            "retry_requires_any_raw_prediction_byte_produced": False,
            "retry_requires_attempt_zero_raw_output_bytes": 0,
            "same_digest_upload_retry_action": ValidationUploadRetryActionV1.REUPLOAD_COMMITTED_OUTPUT.value,
            "same_digest_upload_retry_reason": ValidationUploadRetryReasonV1.OUTPUT_UPLOAD_FAILURE_AFTER_OUTPUT_HASH_COMMITTED.value,
            "safe_unix_seconds_and_nanoseconds": True,
            "retry_forbidden_categories": [
                "ANY_PREDICTION_BYTE_PRODUCED",
                "MODEL_FAILURE",
                "PARSER_FAILURE",
                "STATISTICAL_FAILURE",
                "VERIFIER_FAILURE",
            ],
            "statistical_or_byte_failure_retry_allowed": False,
        },
        "metrics": {
            "evaluator_manifests": {
                "challenge_rows": 240,
                "main_rows": 720,
                "preservation_rows": 572,
                "scale_regret_rows": 240,
                "reported_gate_counts_recomputed_from_rows": True,
                "row_ids_unique_and_primary_preservation_disjoint": True,
            },
            "exact_algebra": {
                "control_and_abstention_cross_sums": True,
                "family_and_scale_slices_sum_to_overall": True,
                "joint_is_component_conjunction": True,
            },
            "fail_closed": {
                "aggregate_graph_exact_one_required": True,
                "nonzero_denominator_required": True,
                "per_version_signed_event_denominator": True,
            },
            "overall_gates_in_order": overall,
            "preservation": {
                "invalid_correct_required": _INVALID_VARIANT_COUNT,
                "invalid_total": _INVALID_VARIANT_COUNT,
                "legal_correct_minimum": 482,
                "legal_total": _LEGAL_VARIANT_COUNT,
            },
            "scale_regret": {
                "denominator": 240,
                "maximum_bootstrap_upper": [8, 100],
                "maximum_point": [1, 20],
                "rational_denominator_maximum": _MAX_RATIONAL_DENOMINATOR,
                "values": "EXACT_NONNEGATIVE_RATIONALS",
            },
            "slice_axis_cardinalities": {"family": 6, "scale": 2},
            "slice_gate_bases_in_order": [
                {
                    "metric": name,
                    "minimum_one_sided_wilson_lower": [lower_n, lower_d],
                    "minimum_point": [point_n, point_d],
                }
                for name, point_n, point_d, lower_n, lower_d in _SLICE_GATE_BASE_SPECS
            ],
            "slice_gate_result_count": 24,
        },
        "prerequisite_binding": {
            "cohort_cross_binds_profile_and_prerequisite_root": True,
            "final_profile_pins_included_in_this_skeleton": False,
            "original_r06_candidate_and_result_bytes_required": True,
            "original_r07_candidate_and_result_bytes_required": True,
            "r06_or_r07_authority_inferred_from_bytes": False,
        },
        "resource_preflight": {
            "declared_attempt_output_bytes_maximum": _MAX_DECLARED_ATTEMPT_OUTPUT_BYTES,
            "declared_system_output_bytes_maximum": _MAX_DECLARED_SYSTEM_OUTPUT_BYTES,
            "system_run_records_before_snapshot": [0, 4],
        },
        "result_projection": {
            "caller_defined_equality_used": False,
            "content_address_covers_all_fields_except_self_id": True,
            "fresh_candidate_and_result_reverification_required": True,
        },
        "schema_version": PHASE2B_C1_STATISTICAL_VALIDATION_PROFILE_V1_SCHEMA,
        "seed_roots": {
            "domain_commitment_required": True,
            "purposes_in_order": [item.value for item in _SEED_PURPOSES],
            "raw_root_bytes": 32,
            "version_two_disjoint_from_version_one": True,
            "within_version_distinct": True,
        },
        "status": "PURE_CANDIDATE_MECHANICS_SKELETON_NOT_R08",
        "validation_versions_maximum": 2,
        "version_two_bump": {
            "exact_indices": [1, 2],
            "all_three_version_ids_bump_together": True,
            "protocol_version_ids": list(_PROTOCOL_VERSION_IDS),
            "selector_version_ids": list(_SELECTOR_VERSION_IDS),
            "validation_version_ids": list(_VALIDATION_VERSION_IDS),
            "closed_public_failure_development_report_schema": _PUBLIC_FAILURE_DEVELOPMENT_REPORT_SCHEMA,
            "closed_version_two_issuance_authority_schema": _VERSION_TWO_ISSUANCE_AUTHORITY_SCHEMA,
            "fresh_cohort_and_seed_roots_required": True,
            "immutable_version_one_failure_cas_required": True,
            "ordered_full_version_one_attempt_archive_cas_root_required": True,
            "public_failure_development_report_required": True,
            "version_two_issuance_authority_required": True,
            "version_two_failure_disposition": "NO_GO",
        },
        "wilson_z_binary64_hex": "0x1.a515209676ab8p+0",
    }


def _content_id(prefix: str, domain: bytes, value: object) -> str:
    return prefix + hashlib.sha256(domain + canonicalize_rfc8785_v1(value)).hexdigest()


def _profile_payload_v1() -> dict[str, object]:
    body = _expected_profile_body_v1()
    profile_id = _content_id(_PROFILE_PREFIX, _PROFILE_DOMAIN, body)
    return {"profile_body": body, "profile_id": profile_id}


@dataclass(frozen=True, slots=True)
class StatisticalValidationProfileVerificationV1:
    schema_version: str
    profile_id: str
    canonical_profile_sha256: str
    claim_level: str
    profile_candidate_mechanics_verified: bool = True
    final_wire_pin_verified: bool = False
    final_r06_pin_verified: bool = False
    r06_qualified: bool = False
    r07_frozen: bool = False
    r08_satisfied: bool = False
    result_object_unforgeable: bool = False
    is_execution_evidence: bool = False

    def __post_init__(self) -> None:
        if (
            self.schema_version != PHASE2B_C1_STATISTICAL_VALIDATION_PROFILE_V1_SCHEMA
            or type(self.profile_id) is not str
            or not self.profile_id.startswith(_PROFILE_PREFIX)
            or type(self.canonical_profile_sha256) is not str
            or len(self.canonical_profile_sha256) != 64
            or self.claim_level
            != PHASE2B_C1_STATISTICAL_VALIDATION_MECHANICS_V1_CLAIM_LEVEL
            or type(self.profile_candidate_mechanics_verified) is not bool
            or not self.profile_candidate_mechanics_verified
        ):
            raise ValueError("profile verification identity differs")
        false_claims = (
            self.final_wire_pin_verified,
            self.final_r06_pin_verified,
            self.r06_qualified,
            self.r07_frozen,
            self.r08_satisfied,
            self.result_object_unforgeable,
            self.is_execution_evidence,
        )
        if any(type(value) is not bool or value for value in false_claims):
            raise ValueError("profile verifier overclaims readiness or authority")


class StatisticalValidationProfileErrorV1(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _profile_fail(code: str, detail: str) -> NoReturn:
    raise StatisticalValidationProfileErrorV1(code, detail)


def verify_c1_statistical_validation_profile_bytes_v1(
    profile_bytes: bytes,
) -> StatisticalValidationProfileVerificationV1:
    """Verify the exact candidate profile bytes without consulting a file."""

    if type(profile_bytes) is not bytes:
        _profile_fail("WRONG_INPUT_TYPE", "profile must be exact bytes")
    if not profile_bytes or len(profile_bytes) > _MAX_PROFILE_BYTES:
        _profile_fail("PROFILE_SIZE_INVALID", "profile byte length is outside the bound")
    expected = canonicalize_rfc8785_v1(_profile_payload_v1())
    if profile_bytes != expected:
        try:
            decode_and_require_canonical_rfc8785_v1(profile_bytes)
        except (RFC8785Error, TypeError, ValueError) as exc:
            _profile_fail("PROFILE_NOT_CANONICAL", str(exc))
        _profile_fail("PROFILE_SCHEMA_INVALID", "profile differs from the closed candidate profile")
    payload = _profile_payload_v1()
    return StatisticalValidationProfileVerificationV1(
        schema_version=PHASE2B_C1_STATISTICAL_VALIDATION_PROFILE_V1_SCHEMA,
        profile_id=str(payload["profile_id"]),
        canonical_profile_sha256=hashlib.sha256(profile_bytes).hexdigest(),
        claim_level=PHASE2B_C1_STATISTICAL_VALIDATION_MECHANICS_V1_CLAIM_LEVEL,
    )


def _require_exact_int(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum or value > _MAX_SAFE_INTEGER:
        raise TypeError(f"{name} must be an exact safe integer >= {minimum}")
    return value


def _require_text(value: object, name: str) -> str:
    if type(value) is not str or not value or len(value.encode("utf-8")) > _MAX_TEXT_BYTES:
        raise TypeError(f"{name} must be a bounded nonempty exact string")
    return value


def _require_short_id(value: object, name: str) -> str:
    text = _require_text(value, name)
    if len(text.encode("ascii", errors="ignore")) != len(text.encode("utf-8")):
        raise TypeError(f"{name} must be an exact ASCII identifier")
    if len(text.encode("ascii")) > _MAX_SHORT_ID_BYTES or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:/-"
        for character in text
    ):
        raise TypeError(f"{name} exceeds the short identifier closure")
    return text


def _require_bytes(value: object, name: str, length: int) -> bytes:
    if type(value) is not bytes or len(value) != length:
        raise TypeError(f"{name} must be exact {length}-byte bytes")
    return value


def _digest_json(domain: bytes, value: object) -> bytes:
    return hashlib.sha256(domain + canonicalize_rfc8785_v1(value)).digest()


def _digest_id(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


@dataclass(frozen=True, slots=True)
class OriginalPrerequisiteGraphV1:
    r06_original_candidate_bytes: bytes
    r06_original_result_bytes: bytes
    r07_original_candidate_bytes: bytes
    r07_original_result_bytes: bytes

    def __post_init__(self) -> None:
        total_bytes = 0
        for name in (
            "r06_original_candidate_bytes",
            "r06_original_result_bytes",
            "r07_original_candidate_bytes",
            "r07_original_result_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not bytes or not value or len(value) > _MAX_PREREQUISITE_BYTES:
                raise TypeError(f"{name} must be bounded nonempty exact bytes")
            total_bytes += len(value)
        if total_bytes > _MAX_PREREQUISITE_BYTES:
            raise ValueError("combined original prerequisite bytes exceed the graph bound")


@dataclass(frozen=True, slots=True)
class ValidationSeedRootCandidateV1:
    purpose: ValidationSeedPurposeV1
    raw_seed_root: bytes
    domain_commitment_sha256: bytes

    def __post_init__(self) -> None:
        if type(self.purpose) is not ValidationSeedPurposeV1:
            raise TypeError("seed purpose must use ValidationSeedPurposeV1")
        _require_bytes(self.raw_seed_root, "raw_seed_root", _RAW_DIGEST_BYTES)
        _require_bytes(
            self.domain_commitment_sha256,
            "domain_commitment_sha256",
            _RAW_DIGEST_BYTES,
        )
        expected = hashlib.sha256(
            _SEED_DOMAIN + self.purpose.value.encode("ascii") + b"\x00" + self.raw_seed_root
        ).digest()
        if not hmac.compare_digest(self.domain_commitment_sha256, expected):
            raise ValueError("seed domain commitment differs")


@dataclass(frozen=True, slots=True)
class ValidationCandidatePublicKeyV1:
    purpose: ValidationKeyPurposeV1
    key_id: str
    raw_public_key: bytes

    def __post_init__(self) -> None:
        if type(self.purpose) is not ValidationKeyPurposeV1:
            raise TypeError("key purpose must use ValidationKeyPurposeV1")
        _require_short_id(self.key_id, "key_id")
        _require_bytes(self.raw_public_key, "raw_public_key", _RAW_PUBLIC_KEY_BYTES)
        expected = _content_id(
            _KEY_PREFIX,
            _KEY_DOMAIN,
            {"purpose": self.purpose.value, "raw_public_key_hex": self.raw_public_key.hex()},
        )
        if self.key_id != expected:
            raise ValueError("candidate key id differs")


@dataclass(frozen=True, slots=True)
class ValidationCohortBindingV1:
    statistical_validation_profile_id: str
    prerequisite_graph_root_sha256: bytes
    primary_original_bytes_sha256: bytes
    primary_batch_root_sha256: bytes
    preservation_original_bytes_sha256: bytes
    preservation_batch_root_sha256: bytes
    generation_receipt_root_sha256: bytes
    primary_count: int = _PRIMARY_COUNT
    main_count: int = _MAIN_COUNT
    challenge_count: int = _CHALLENGE_COUNT
    legal_variant_count: int = _LEGAL_VARIANT_COUNT
    invalid_variant_count: int = _INVALID_VARIANT_COUNT
    preservation_count: int = _VARIANT_COUNT

    def __post_init__(self) -> None:
        if (
            type(self.statistical_validation_profile_id) is not str
            or not self.statistical_validation_profile_id.startswith(_PROFILE_PREFIX)
        ):
            raise ValueError("cohort statistical-validation profile id differs")
        _require_short_id(
            self.statistical_validation_profile_id,
            "statistical_validation_profile_id",
        )
        _require_bytes(
            self.prerequisite_graph_root_sha256,
            "prerequisite_graph_root_sha256",
            _RAW_DIGEST_BYTES,
        )
        for name in (
            "primary_original_bytes_sha256",
            "primary_batch_root_sha256",
            "preservation_original_bytes_sha256",
            "preservation_batch_root_sha256",
            "generation_receipt_root_sha256",
        ):
            _require_bytes(getattr(self, name), name, _RAW_DIGEST_BYTES)
        counts = (
            self.primary_count,
            self.main_count,
            self.challenge_count,
            self.legal_variant_count,
            self.invalid_variant_count,
            self.preservation_count,
        )
        if any(type(value) is not int for value in counts) or counts != (
            _PRIMARY_COUNT,
            _MAIN_COUNT,
            _CHALLENGE_COUNT,
            _LEGAL_VARIANT_COUNT,
            _INVALID_VARIANT_COUNT,
            _VARIANT_COUNT,
        ):
            raise ValueError("cohort count closure differs")


@dataclass(frozen=True, slots=True)
class ValidationSystemRunCandidateV1:
    system: ValidationSystemV1
    primary_prediction_root_sha256: bytes
    preservation_prediction_root_sha256: bytes
    raw_output_byte_count: int
    primary_count: int = _PRIMARY_COUNT
    preservation_count: int = _VARIANT_COUNT
    total_count: int = _ALL_SYSTEM_ROW_COUNT

    def __post_init__(self) -> None:
        if type(self.system) is not ValidationSystemV1:
            raise TypeError("system must use ValidationSystemV1")
        _require_bytes(
            self.primary_prediction_root_sha256,
            "primary_prediction_root_sha256",
            _RAW_DIGEST_BYTES,
        )
        _require_bytes(
            self.preservation_prediction_root_sha256,
            "preservation_prediction_root_sha256",
            _RAW_DIGEST_BYTES,
        )
        _require_exact_int(self.raw_output_byte_count, "raw_output_byte_count", 1)
        if (
            type(self.primary_count) is not int
            or type(self.preservation_count) is not int
            or type(self.total_count) is not int
            or (self.primary_count, self.preservation_count, self.total_count)
            != (_PRIMARY_COUNT, _VARIANT_COUNT, _ALL_SYSTEM_ROW_COUNT)
        ):
            raise ValueError("system run count closure differs")


@dataclass(frozen=True, slots=True)
class BinaryGateCandidateV1:
    metric: str
    scope_axis: str
    scope_value: str
    correct_count: int
    total_count: int

    def __post_init__(self) -> None:
        _require_short_id(self.metric, "metric")
        _require_short_id(self.scope_axis, "scope_axis")
        if self.scope_axis not in {"overall", "family", "scale"}:
            raise ValueError("scope_axis is outside the closed set")
        _require_short_id(self.scope_value, "scope_value")
        _require_exact_int(self.correct_count, "correct_count")
        _require_exact_int(self.total_count, "total_count", 1)
        if self.correct_count > self.total_count:
            raise ValueError("correct_count exceeds total_count")


@dataclass(frozen=True, slots=True)
class ExactNonnegativeRationalV1:
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        _require_exact_int(self.numerator, "numerator")
        _require_exact_int(self.denominator, "denominator", 1)
        if self.denominator > _MAX_RATIONAL_DENOMINATOR:
            raise ValueError("exact rational denominator exceeds 1024")
        if math.gcd(self.numerator, self.denominator) != 1:
            raise ValueError("exact rational must be reduced")

    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)


@dataclass(frozen=True, slots=True)
class MainEvaluationRowV1:
    input_row_id: str
    family: str
    scale: str
    case_type: ValidationMainCaseTypeV1
    cell_ordinal: int
    predicted_decision: ValidationPredictionDecisionV1
    family_exact: bool | None
    binding_exact: bool | None
    scale_set_exact: bool | None
    decision_exact: bool
    joint_exact: bool

    def __post_init__(self) -> None:
        _require_short_id(self.input_row_id, "input_row_id")
        if (
            type(self.family) is not str
            or self.family not in _FAMILIES
            or type(self.scale) is not str
            or self.scale not in _SCALES
            or type(self.case_type) is not ValidationMainCaseTypeV1
            or type(self.cell_ordinal) is not int
            or self.cell_ordinal < 0
            or type(self.predicted_decision) is not ValidationPredictionDecisionV1
            or type(self.decision_exact) is not bool
            or type(self.joint_exact) is not bool
        ):
            raise TypeError("main evaluator row axis, ordinal, decision, or booleans differ")
        if self.case_type in _ANSWERABLE_CASE_TYPES:
            exact_values = (self.family_exact, self.binding_exact, self.scale_set_exact)
            if any(type(value) is not bool for value in exact_values):
                raise TypeError("answerable row exactness fields must be exact booleans")
            expected_decision = (
                ValidationPredictionDecisionV1.ANSWER_SET
                if self.case_type is ValidationMainCaseTypeV1.ADMISSIBLE_SCALE_SET_ANSWERABLE
                else ValidationPredictionDecisionV1.ANSWER
            )
            if self.decision_exact is not (self.predicted_decision is expected_decision):
                raise ValueError("answerable decision_exact differs from predicted decision")
            if self.joint_exact is not all((*exact_values, self.decision_exact)):
                raise ValueError("joint exactness is not the conjunction of component exactness")
        else:
            if any(value is not None for value in (self.family_exact, self.binding_exact, self.scale_set_exact)):
                raise ValueError("nonanswerable component exactness fields must be absent")
            if self.decision_exact is not (
                self.predicted_decision is ValidationPredictionDecisionV1.ABSTAIN
            ):
                raise ValueError("nonanswerable decision_exact differs from abstention")
            if self.joint_exact:
                raise ValueError("nonanswerable row cannot be joint exact")


@dataclass(frozen=True, slots=True)
class ScaleRegretEvaluationRowV1:
    input_row_id: str
    family: str
    scale: str
    regret: ExactNonnegativeRationalV1

    def __post_init__(self) -> None:
        _require_short_id(self.input_row_id, "input_row_id")
        if (
            type(self.family) is not str
            or self.family not in _FAMILIES
            or type(self.scale) is not str
            or self.scale not in _SCALES
            or type(self.regret) is not ExactNonnegativeRationalV1
        ):
            raise TypeError("scale-regret row closure differs")


@dataclass(frozen=True, slots=True)
class ChallengeEvaluationRowV1:
    input_row_id: str
    family: str
    scale: str
    challenge_kind: ValidationChallengeKindV1
    cell_ordinal: int
    system_correct: bool
    embedding_correct: bool
    frozen_llm_correct: bool

    def __post_init__(self) -> None:
        _require_short_id(self.input_row_id, "input_row_id")
        if (
            type(self.family) is not str
            or self.family not in _FAMILIES
            or type(self.scale) is not str
            or self.scale not in _SCALES
            or type(self.challenge_kind) is not ValidationChallengeKindV1
            or type(self.cell_ordinal) is not int
            or not 0 <= self.cell_ordinal < 10
        ):
            raise TypeError("challenge row axis, kind, or ordinal differs")
        if any(
            type(value) is not bool
            for value in (
                self.system_correct,
                self.embedding_correct,
                self.frozen_llm_correct,
            )
        ):
            raise TypeError("challenge correctness fields must be exact booleans")


@dataclass(frozen=True, slots=True)
class PreservationEvaluationRowV1:
    variant_row_id: str
    preservation_kind: ValidationPreservationKindV1
    kind_ordinal: int
    recognizer_correct: bool

    def __post_init__(self) -> None:
        _require_short_id(self.variant_row_id, "variant_row_id")
        if (
            type(self.preservation_kind) is not ValidationPreservationKindV1
            or type(self.kind_ordinal) is not int
            or self.kind_ordinal < 0
            or type(self.recognizer_correct) is not bool
        ):
            raise TypeError("preservation evaluator row closure differs")


@dataclass(frozen=True, slots=True)
class StatisticalMetricBundleCandidateV1:
    main_rows: tuple[MainEvaluationRowV1, ...]
    preservation_rows: tuple[PreservationEvaluationRowV1, ...]
    scale_regret_rows: tuple[ScaleRegretEvaluationRowV1, ...]
    challenge_rows: tuple[ChallengeEvaluationRowV1, ...]
    overall_gates: tuple[BinaryGateCandidateV1, ...]
    slice_gates: tuple[BinaryGateCandidateV1, ...]

    def __post_init__(self) -> None:
        if (
            type(self.main_rows) is not tuple
            or len(self.main_rows) != _MAIN_COUNT
            or any(type(item) is not MainEvaluationRowV1 for item in self.main_rows)
            or type(self.preservation_rows) is not tuple
            or len(self.preservation_rows) != _VARIANT_COUNT
            or any(
                type(item) is not PreservationEvaluationRowV1
                for item in self.preservation_rows
            )
            or type(self.scale_regret_rows) is not tuple
            or len(self.scale_regret_rows) != 240
            or any(
                type(item) is not ScaleRegretEvaluationRowV1
                for item in self.scale_regret_rows
            )
            or type(self.challenge_rows) is not tuple
            or len(self.challenge_rows) != _CHALLENGE_COUNT
            or any(
                type(item) is not ChallengeEvaluationRowV1
                for item in self.challenge_rows
            )
            or type(self.overall_gates) is not tuple
            or len(self.overall_gates) != 12
            or any(type(item) is not BinaryGateCandidateV1 for item in self.overall_gates)
            or type(self.slice_gates) is not tuple
            or len(self.slice_gates) != 24
            or any(type(item) is not BinaryGateCandidateV1 for item in self.slice_gates)
        ):
            raise TypeError("metric bundle tuple closure differs")


@dataclass(frozen=True, slots=True)
class ValidationCasReceiptCandidateV1:
    validation_version_index: int
    attempt_index: int
    object_key: str
    previous_generation: int
    new_generation: int
    value_digest_sha256: bytes
    upload_digest_attempts: tuple[bytes, ...]
    upload_retry_action: ValidationUploadRetryActionV1
    upload_retry_reason: ValidationUploadRetryReasonV1
    store_key_id: str
    signature: bytes
    receipt_id: str

    def __post_init__(self) -> None:
        if (
            type(self.validation_version_index) is not int
            or self.validation_version_index not in {1, 2}
            or type(self.attempt_index) is not int
            or self.attempt_index not in {0, 1}
        ):
            raise ValueError("CAS version or attempt index differs")
        _require_short_id(self.object_key, "object_key")
        expected_key = (
            "phase2b/c1/statistical-validation/v1/version-"
            f"{self.validation_version_index}/attempt-{self.attempt_index}"
        )
        if (
            self.object_key != expected_key
            or type(self.previous_generation) is not int
            or self.previous_generation != 0
            or type(self.new_generation) is not int
            or self.new_generation != 1
        ):
            raise ValueError("CAS fixed object key or 0 -> 1 transition differs")
        _require_bytes(self.value_digest_sha256, "value_digest_sha256", _RAW_DIGEST_BYTES)
        if (
            type(self.upload_digest_attempts) is not tuple
            or len(self.upload_digest_attempts) not in {1, 2}
            or any(
                type(value) is not bytes
                or len(value) != _RAW_DIGEST_BYTES
                or not hmac.compare_digest(value, self.value_digest_sha256)
                for value in self.upload_digest_attempts
            )
        ):
            raise ValueError("CAS upload retry must reuse the exact value digest")
        if (
            type(self.upload_retry_action) is not ValidationUploadRetryActionV1
            or type(self.upload_retry_reason) is not ValidationUploadRetryReasonV1
        ):
            raise TypeError("CAS upload retry action/reason enums differ")
        expected_retry = (
            (
                ValidationUploadRetryActionV1.NO_RETRY,
                ValidationUploadRetryReasonV1.NONE,
            )
            if len(self.upload_digest_attempts) == 1
            else (
                ValidationUploadRetryActionV1.REUPLOAD_COMMITTED_OUTPUT,
                ValidationUploadRetryReasonV1.OUTPUT_UPLOAD_FAILURE_AFTER_OUTPUT_HASH_COMMITTED,
            )
        )
        if (self.upload_retry_action, self.upload_retry_reason) != expected_retry:
            raise ValueError("CAS upload retry action/reason does not match digest attempts")
        _require_short_id(self.store_key_id, "store_key_id")
        _require_bytes(self.signature, "signature", _RAW_SIGNATURE_BYTES)
        _require_short_id(self.receipt_id, "receipt_id")


@dataclass(frozen=True, slots=True)
class ValidationSignedEventCandidateV1:
    validation_version_index: int
    attempt_index: int
    sequence: int
    event_kind: ValidationEventKindV1
    payload_root_sha256: bytes
    previous_event_id: str | None
    logical_time: int
    declared_unix_seconds: int
    declared_nanosecond: int
    signer_key_id: str
    signature: bytes
    event_id: str

    def __post_init__(self) -> None:
        if (
            type(self.validation_version_index) is not int
            or self.validation_version_index not in {1, 2}
            or type(self.attempt_index) is not int
            or self.attempt_index not in {0, 1}
        ):
            raise ValueError("event version or attempt index differs")
        if type(self.sequence) is not int or not 0 <= self.sequence < _EVENTS_PER_ATTEMPT:
            raise ValueError("event sequence differs")
        if type(self.event_kind) is not ValidationEventKindV1:
            raise TypeError("event kind must use ValidationEventKindV1")
        _require_bytes(self.payload_root_sha256, "payload_root_sha256", _RAW_DIGEST_BYTES)
        if self.previous_event_id is not None:
            _require_short_id(self.previous_event_id, "previous_event_id")
        _require_exact_int(self.logical_time, "logical_time")
        _require_exact_int(self.declared_unix_seconds, "declared_unix_seconds")
        if (
            type(self.declared_nanosecond) is not int
            or not 0 <= self.declared_nanosecond <= 999_999_999
        ):
            raise ValueError("declared_nanosecond is outside [0, 999999999]")
        _require_short_id(self.signer_key_id, "signer_key_id")
        _require_bytes(self.signature, "signature", _RAW_SIGNATURE_BYTES)
        _require_short_id(self.event_id, "event_id")


@dataclass(frozen=True, slots=True)
class StatisticalValidationAttemptCandidateV1:
    validation_version_index: int
    attempt_index: int
    terminal_code: ValidationAttemptTerminalCodeV1
    execution_retry_action: ValidationExecutionRetryActionV1
    execution_retry_reason: ValidationExecutionRetryReasonV1
    cohort: ValidationCohortBindingV1
    seed_roots: tuple[ValidationSeedRootCandidateV1, ...]
    system_runs: tuple[ValidationSystemRunCandidateV1, ...]
    metric_bundle: StatisticalMetricBundleCandidateV1 | None
    observer_transcript_root_sha256: bytes
    any_raw_prediction_byte_produced: bool
    raw_output_byte_count: int
    cas_receipt: ValidationCasReceiptCandidateV1
    events: tuple[ValidationSignedEventCandidateV1, ...]

    def __post_init__(self) -> None:
        if (
            type(self.validation_version_index) is not int
            or self.validation_version_index not in {1, 2}
            or type(self.attempt_index) is not int
            or self.attempt_index not in {0, 1}
        ):
            raise ValueError("attempt version or attempt index differs")
        if type(self.terminal_code) is not ValidationAttemptTerminalCodeV1:
            raise TypeError("terminal code must use ValidationAttemptTerminalCodeV1")
        if (
            type(self.execution_retry_action) is not ValidationExecutionRetryActionV1
            or type(self.execution_retry_reason) is not ValidationExecutionRetryReasonV1
        ):
            raise TypeError("execution retry action/reason enums differ")
        if type(self.cohort) is not ValidationCohortBindingV1:
            raise TypeError("cohort must use ValidationCohortBindingV1")
        if (
            type(self.seed_roots) is not tuple
            or len(self.seed_roots) != 5
            or any(type(item) is not ValidationSeedRootCandidateV1 for item in self.seed_roots)
        ):
            raise TypeError("attempt needs five seed roots")
        if (
            type(self.system_runs) is not tuple
            or len(self.system_runs) not in {0, 4}
            or any(type(item) is not ValidationSystemRunCandidateV1 for item in self.system_runs)
        ):
            raise TypeError("system_runs must contain exactly zero or four closed records")
        if self.metric_bundle is not None and type(self.metric_bundle) is not StatisticalMetricBundleCandidateV1:
            raise TypeError("metric_bundle has the wrong type")
        _require_bytes(
            self.observer_transcript_root_sha256,
            "observer_transcript_root_sha256",
            _RAW_DIGEST_BYTES,
        )
        if type(self.any_raw_prediction_byte_produced) is not bool:
            raise TypeError("any_raw_prediction_byte_produced must be an exact boolean")
        _require_exact_int(self.raw_output_byte_count, "raw_output_byte_count")
        if any(
            type(row.raw_output_byte_count) is not int
            or row.raw_output_byte_count < 1
            or row.raw_output_byte_count > _MAX_SAFE_INTEGER
            for row in self.system_runs
        ):
            raise TypeError("system output byte counts must be exact positive safe integers")
        declared_output_sum = sum(row.raw_output_byte_count for row in self.system_runs)
        if any(
            row.raw_output_byte_count > _MAX_DECLARED_SYSTEM_OUTPUT_BYTES
            for row in self.system_runs
        ) or declared_output_sum > _MAX_DECLARED_ATTEMPT_OUTPUT_BYTES:
            raise ValueError("declared system output byte count exceeds the preflight bound")
        complete = self.terminal_code in {
            ValidationAttemptTerminalCodeV1.COMPLETE_PASS,
            ValidationAttemptTerminalCodeV1.COMPLETE_FAIL,
        }
        if complete:
            if (
                len(self.system_runs) != 4
                or self.metric_bundle is None
                or self.raw_output_byte_count != declared_output_sum
                or self.raw_output_byte_count == 0
                or not self.any_raw_prediction_byte_produced
                or self.execution_retry_action is not ValidationExecutionRetryActionV1.NO_RETRY
                or self.execution_retry_reason is not ValidationExecutionRetryReasonV1.NONE
            ):
                raise ValueError("complete attempt output/retry preflight differs")
        else:
            expected_reason = (
                ValidationExecutionRetryReasonV1(self.terminal_code.value)
                if self.terminal_code in _RETRYABLE_INFRA_CODES
                else None
            )
            if (
                expected_reason is None
                or self.system_runs
                or self.metric_bundle is not None
                or self.raw_output_byte_count != 0
                or self.any_raw_prediction_byte_produced
                or self.execution_retry_action is not ValidationExecutionRetryActionV1.REEXECUTE
                or self.execution_retry_reason is not expected_reason
            ):
                raise ValueError("retryable attempt must be exact zero-byte REEXECUTE")
        if type(self.cas_receipt) is not ValidationCasReceiptCandidateV1:
            raise TypeError("cas_receipt has the wrong type")
        if (
            type(self.events) is not tuple
            or len(self.events) != _EVENTS_PER_ATTEMPT
            or any(type(item) is not ValidationSignedEventCandidateV1 for item in self.events)
        ):
            raise TypeError("attempt needs six signed lifecycle events")


@dataclass(frozen=True, slots=True)
class StatisticalValidationVersionCandidateV1:
    validation_version_index: int
    attempts: tuple[StatisticalValidationAttemptCandidateV1, ...]

    def __post_init__(self) -> None:
        if (
            type(self.validation_version_index) is not int
            or self.validation_version_index not in {1, 2}
        ):
            raise ValueError("validation version index differs")
        if (
            type(self.attempts) is not tuple
            or len(self.attempts) not in {1, 2}
            or any(type(item) is not StatisticalValidationAttemptCandidateV1 for item in self.attempts)
            or any(
                type(item.attempt_index) is not int
                or type(item.validation_version_index) is not int
                for item in self.attempts
            )
            or tuple(item.attempt_index for item in self.attempts) != tuple(range(len(self.attempts)))
            or any(item.validation_version_index != self.validation_version_index for item in self.attempts)
        ):
            raise ValueError("attempt sequence is not exact 0 or 0,1")
        if len(self.attempts) == 1:
            if self.attempts[0].terminal_code not in {
                ValidationAttemptTerminalCodeV1.COMPLETE_PASS,
                ValidationAttemptTerminalCodeV1.COMPLETE_FAIL,
            }:
                raise ValueError("a single attempt must be statistically terminal")
        else:
            first, second = self.attempts
            if (
                first.terminal_code not in _RETRYABLE_INFRA_CODES
                or first.execution_retry_action is not ValidationExecutionRetryActionV1.REEXECUTE
                or second.terminal_code not in {
                    ValidationAttemptTerminalCodeV1.COMPLETE_PASS,
                    ValidationAttemptTerminalCodeV1.COMPLETE_FAIL,
                }
            ):
                raise ValueError("two attempts must be zero-byte REEXECUTE then terminal")


@dataclass(frozen=True, slots=True)
class StatisticalValidationVersionBumpCandidateV1:
    from_validation_version_index: int
    to_validation_version_index: int
    old_protocol_version_id: str
    new_protocol_version_id: str
    old_selector_version_id: str
    new_selector_version_id: str
    old_validation_version_id: str
    new_validation_version_id: str
    immutable_version_one_failure_root_sha256: bytes
    version_one_failure_cas_receipt_id: str
    version_one_full_attempt_chain_root_sha256: bytes
    version_bump_core_id: str
    public_development_change_set_sha256: bytes
    public_failure_development_report_bytes: bytes
    version_two_issuance_authority_bytes: bytes
    version_two_cohort_root_sha256: bytes
    version_two_seed_roots: tuple[bytes, ...]
    version_bump_id: str

    def __post_init__(self) -> None:
        if (
            type(self.from_validation_version_index) is not int
            or self.from_validation_version_index != 1
            or type(self.to_validation_version_index) is not int
            or self.to_validation_version_index != 2
        ):
            raise ValueError("version bump must be exactly 1 -> 2")
        version_id_pairs = (
            (
                self.old_protocol_version_id,
                self.new_protocol_version_id,
                _PROTOCOL_VERSION_IDS,
            ),
            (
                self.old_selector_version_id,
                self.new_selector_version_id,
                _SELECTOR_VERSION_IDS,
            ),
            (
                self.old_validation_version_id,
                self.new_validation_version_id,
                _VALIDATION_VERSION_IDS,
            ),
        )
        for old_id, new_id, expected_pair in version_id_pairs:
            _require_short_id(old_id, "old version id")
            _require_short_id(new_id, "new version id")
            if (old_id, new_id) != expected_pair:
                raise ValueError("protocol, selector, and validation IDs must all bump 1 -> 2")
        _require_bytes(
            self.immutable_version_one_failure_root_sha256,
            "immutable_version_one_failure_root_sha256",
            _RAW_DIGEST_BYTES,
        )
        _require_short_id(
            self.version_one_failure_cas_receipt_id,
            "version_one_failure_cas_receipt_id",
        )
        _require_bytes(
            self.version_one_full_attempt_chain_root_sha256,
            "version_one_full_attempt_chain_root_sha256",
            _RAW_DIGEST_BYTES,
        )
        _require_short_id(self.version_bump_core_id, "version_bump_core_id")
        _require_bytes(
            self.public_development_change_set_sha256,
            "public_development_change_set_sha256",
            _RAW_DIGEST_BYTES,
        )
        for name in (
            "public_failure_development_report_bytes",
            "version_two_issuance_authority_bytes",
        ):
            value = getattr(self, name)
            if (
                type(value) is not bytes
                or not value
                or len(value) > _MAX_VERSION_ARTIFACT_BYTES
            ):
                raise TypeError(f"{name} must be bounded nonempty exact bytes")
        _require_bytes(
            self.version_two_cohort_root_sha256,
            "version_two_cohort_root_sha256",
            _RAW_DIGEST_BYTES,
        )
        if (
            type(self.version_two_seed_roots) is not tuple
            or len(self.version_two_seed_roots) != len(_SEED_PURPOSES)
            or any(
                type(value) is not bytes or len(value) != _RAW_DIGEST_BYTES
                for value in self.version_two_seed_roots
            )
            or len(set(self.version_two_seed_roots)) != len(_SEED_PURPOSES)
        ):
            raise TypeError("version-two bump needs five distinct exact raw seed roots")
        _require_short_id(self.version_bump_id, "version_bump_id")


@dataclass(frozen=True, slots=True)
class StatisticalValidationCandidateGraphV1:
    profile_bytes: bytes
    prerequisites: OriginalPrerequisiteGraphV1
    candidate_keys: tuple[ValidationCandidatePublicKeyV1, ...]
    versions: tuple[StatisticalValidationVersionCandidateV1, ...]
    version_bump: StatisticalValidationVersionBumpCandidateV1 | None

    def __post_init__(self) -> None:
        if (
            type(self.profile_bytes) is not bytes
            or not self.profile_bytes
            or len(self.profile_bytes) > _MAX_PROFILE_BYTES
        ):
            raise TypeError("profile_bytes must be bounded nonempty exact bytes")
        if type(self.prerequisites) is not OriginalPrerequisiteGraphV1:
            raise TypeError("prerequisites has the wrong type")
        if (
            type(self.candidate_keys) is not tuple
            or len(self.candidate_keys) != 6
            or any(type(item) is not ValidationCandidatePublicKeyV1 for item in self.candidate_keys)
        ):
            raise TypeError("candidate_keys must contain six closed key records")
        if (
            type(self.versions) is not tuple
            or len(self.versions) not in {1, 2}
            or any(type(item) is not StatisticalValidationVersionCandidateV1 for item in self.versions)
            or any(type(item.validation_version_index) is not int for item in self.versions)
            or tuple(item.validation_version_index for item in self.versions)
            != tuple(range(1, len(self.versions) + 1))
        ):
            raise ValueError("validation versions must be exactly 1 or 1,2")
        if len(self.versions) == 1:
            if self.version_bump is not None:
                raise ValueError("single-version candidate cannot carry a version bump")
        elif type(self.version_bump) is not StatisticalValidationVersionBumpCandidateV1:
            raise TypeError("two-version candidate requires a closed version-bump record")


@dataclass(frozen=True, slots=True)
class StatisticalValidationCandidateMechanicsResultV1:
    outcome: StatisticalValidationOutcomeV1
    reason: StatisticalValidationCandidateReasonV1
    version: str
    claim_level: str
    profile_id: str
    result_id: str
    input_snapshot_commitment_sha256: str
    prerequisite_graph_root_sha256: str
    validation_version_count: int
    final_validation_version_index: int
    final_attempt_index: int
    version_bump_id: str | None
    no_go_after_version_two_failure: bool
    verified_signature_count: int
    total_event_count: int
    fail_closed_denominators_by_version: tuple[int, ...]
    fail_closed_correct_by_version: tuple[int, ...]
    per_version_outcome_gate_root_vector: tuple[
        tuple[StatisticalValidationOutcomeV1, str], ...
    ]
    aggregate_fail_closed_correct_count: int
    aggregate_fail_closed_denominator: int
    aggregate_fail_closed_passed: bool
    evaluator_manifest_row_counts: tuple[int, int, int, int]
    final_metric_bundle_root_sha256: str
    overall_gate_passes: tuple[bool, ...]
    slice_gate_passes: tuple[bool, ...]
    challenge_gate_passes: tuple[bool, bool]
    scale_regret_point: tuple[int, int] | None
    scale_regret_bootstrap_upper: tuple[int, int] | None
    semantic_advantage_point: tuple[int, int] | None
    semantic_advantage_bootstrap_lower: tuple[int, int] | None
    final_validation_version_all_statistical_gates_passed: bool
    statistical_validation_candidate_mechanics_verified: bool = True
    original_r06_r07_bytes_canonically_bound: bool = True
    exact_cohort_and_quota_mechanics_verified: bool = True
    retry_lifecycle_mechanics_verified: bool = True
    six_purpose_signature_mechanics_verified: bool = True
    cas_zero_to_one_candidate_mechanics_verified: bool = True
    contiguous_declared_time_mechanics_verified: bool = True
    bootstrap_mechanics_verified: bool = True
    cooperative_computation_only: bool = True
    downstream_fresh_reverification_required: bool = True
    r06_qualified: bool = False
    r07_frozen: bool = False
    r08_satisfied: bool = False
    actual_execution_verified: bool = False
    authority_verified: bool = False
    origin_authenticated: bool = False
    durable_cas_verified: bool = False
    authoritative_time_verified: bool = False
    result_object_unforgeable: bool = False
    filesystem_accessed: bool = False
    network_accessed: bool = False
    docker_invoked: bool = False
    entropy_used: bool = False
    keys_generated: bool = False
    signatures_generated: bool = False
    clock_consulted: bool = False
    store_mutated: bool = False

    def __post_init__(self) -> None:
        if type(self.outcome) is not StatisticalValidationOutcomeV1 or self.outcome not in {
            StatisticalValidationOutcomeV1.COMPLETE_PASS,
            StatisticalValidationOutcomeV1.COMPLETE_FAIL,
        }:
            raise ValueError("successful mechanics result must be complete pass or fail")
        if (
            type(self.reason) is not StatisticalValidationCandidateReasonV1
            or self.reason
            is not StatisticalValidationCandidateReasonV1.CANDIDATE_MECHANICS_VERIFIED
            or type(self.version) is not str
            or self.version != PHASE2B_C1_STATISTICAL_VALIDATION_MECHANICS_V1_VERSION
            or type(self.claim_level) is not str
            or self.claim_level
            != PHASE2B_C1_STATISTICAL_VALIDATION_MECHANICS_V1_CLAIM_LEVEL
            or type(self.profile_id) is not str
            or len(self.profile_id) != len(_PROFILE_PREFIX) + 64
            or not self.profile_id.startswith(_PROFILE_PREFIX)
            or any(
                character not in "0123456789abcdef"
                for character in self.profile_id[len(_PROFILE_PREFIX) :]
            )
            or type(self.result_id) is not str
            or len(self.result_id) != len(_RESULT_PREFIX) + 64
            or not self.result_id.startswith(_RESULT_PREFIX)
            or any(
                character not in "0123456789abcdef"
                for character in self.result_id[len(_RESULT_PREFIX) :]
            )
            or type(self.input_snapshot_commitment_sha256) is not str
            or type(self.prerequisite_graph_root_sha256) is not str
            or type(self.final_metric_bundle_root_sha256) is not str
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in (
                    self.input_snapshot_commitment_sha256,
                    self.prerequisite_graph_root_sha256,
                    self.final_metric_bundle_root_sha256,
                )
            )
        ):
            raise ValueError("mechanics result identity differs")
        int_fields = (
            self.validation_version_count,
            self.final_validation_version_index,
            self.final_attempt_index,
            self.verified_signature_count,
            self.total_event_count,
            self.aggregate_fail_closed_correct_count,
            self.aggregate_fail_closed_denominator,
        )
        if any(
            type(value) is not int or value < 0 or value > _MAX_SAFE_INTEGER
            for value in int_fields
        ):
            raise TypeError("mechanics result integer fields must be exact nonnegative ints")
        if (
            self.validation_version_count not in {1, 2}
            or self.final_validation_version_index != self.validation_version_count
            or self.final_attempt_index not in {0, 1}
            or self.total_event_count <= 0
            or self.total_event_count % _EVENTS_PER_ATTEMPT != 0
        ):
            raise ValueError("result validation-version/attempt/event cardinality differs")
        if self.validation_version_count == 1:
            if self.version_bump_id is not None or self.no_go_after_version_two_failure:
                raise ValueError("single-version result cannot carry bump or version-two NO-GO")
        else:
            if (
                type(self.version_bump_id) is not str
                or len(self.version_bump_id) != len(_VERSION_BUMP_PREFIX) + 64
                or not self.version_bump_id.startswith(_VERSION_BUMP_PREFIX)
                or any(
                    character not in "0123456789abcdef"
                    for character in self.version_bump_id[len(_VERSION_BUMP_PREFIX) :]
                )
            ):
                raise ValueError("two-version result lacks exact version-bump id")
        if type(self.no_go_after_version_two_failure) is not bool:
            raise TypeError("NO-GO marker must be an exact boolean")
        if self.no_go_after_version_two_failure is not (
            self.validation_version_count == 2
            and self.outcome is StatisticalValidationOutcomeV1.COMPLETE_FAIL
        ):
            raise ValueError("version-two failure NO-GO marker differs")
        if (
            type(self.per_version_outcome_gate_root_vector) is not tuple
            or len(self.per_version_outcome_gate_root_vector)
            != self.validation_version_count
            or any(
                type(row) is not tuple
                or len(row) != 2
                or type(row[0]) is not StatisticalValidationOutcomeV1
                or row[0]
                not in {
                    StatisticalValidationOutcomeV1.COMPLETE_PASS,
                    StatisticalValidationOutcomeV1.COMPLETE_FAIL,
                }
                or type(row[1]) is not str
                or len(row[1]) != 64
                or any(
                    character not in "0123456789abcdef" for character in row[1]
                )
                for row in self.per_version_outcome_gate_root_vector
            )
            or self.per_version_outcome_gate_root_vector[-1][0] is not self.outcome
            or (
                self.validation_version_count == 2
                and self.per_version_outcome_gate_root_vector[0][0]
                is not StatisticalValidationOutcomeV1.COMPLETE_FAIL
            )
            or len(
                {row[1] for row in self.per_version_outcome_gate_root_vector}
            )
            != self.validation_version_count
        ):
            raise ValueError("per-version outcome/gate-root vector differs")
        for name, values in (
            ("fail_closed_denominators_by_version", self.fail_closed_denominators_by_version),
            ("fail_closed_correct_by_version", self.fail_closed_correct_by_version),
        ):
            if (
                type(values) is not tuple
                or len(values) != self.validation_version_count
                or any(
                    type(value) is not int
                    or value not in {_EVENTS_PER_ATTEMPT, 2 * _EVENTS_PER_ATTEMPT}
                    for value in values
                )
            ):
                raise ValueError(f"{name} cardinality or exact integer type differs")
        if (
            self.fail_closed_correct_by_version != self.fail_closed_denominators_by_version
            or sum(self.fail_closed_denominators_by_version) != self.total_event_count
            or self.final_attempt_index
            != self.fail_closed_denominators_by_version[-1] // _EVENTS_PER_ATTEMPT - 1
            or self.aggregate_fail_closed_correct_count != self.total_event_count
            or self.aggregate_fail_closed_denominator != self.total_event_count
            or type(self.aggregate_fail_closed_passed) is not bool
            or not self.aggregate_fail_closed_passed
        ):
            raise ValueError("per-version or aggregate fail-closed algebra differs")
        attempt_count = self.total_event_count // _EVENTS_PER_ATTEMPT
        if self.verified_signature_count != self.total_event_count + attempt_count:
            raise ValueError("verified signature count differs from six events plus one CAS")
        if (
            type(self.evaluator_manifest_row_counts) is not tuple
            or any(type(value) is not int for value in self.evaluator_manifest_row_counts)
            or self.evaluator_manifest_row_counts
            != (_MAIN_COUNT, _CHALLENGE_COUNT, _VARIANT_COUNT, 240)
        ):
            raise ValueError("evaluator manifest row counts differ")
        if (
            type(self.overall_gate_passes) is not tuple
            or len(self.overall_gate_passes) != 12
            or any(type(value) is not bool for value in self.overall_gate_passes)
            or type(self.slice_gate_passes) is not tuple
            or len(self.slice_gate_passes) != 24
            or any(type(value) is not bool for value in self.slice_gate_passes)
            or type(self.challenge_gate_passes) is not tuple
            or len(self.challenge_gate_passes) != 2
            or any(type(value) is not bool for value in self.challenge_gate_passes)
        ):
            raise TypeError("result gate-pass tuple cardinality/type differs")

        def validate_ratio(
            name: str,
            value: tuple[int, int] | None,
            *,
            nonnegative: bool,
        ) -> None:
            if (
                type(value) is not tuple
                or len(value) != 2
                or type(value[0]) is not int
                or type(value[1]) is not int
                or abs(value[0]) > _MAX_SAFE_INTEGER
                or value[1] > _MAX_SAFE_INTEGER
                or value[1] <= 0
                or (nonnegative and value[0] < 0)
                or math.gcd(value[0], value[1]) != 1
            ):
                raise ValueError(f"{name} must be an exact reduced rational pair")

        validate_ratio("scale_regret_point", self.scale_regret_point, nonnegative=True)
        validate_ratio(
            "scale_regret_bootstrap_upper",
            self.scale_regret_bootstrap_upper,
            nonnegative=True,
        )
        validate_ratio(
            "semantic_advantage_point",
            self.semantic_advantage_point,
            nonnegative=False,
        )
        validate_ratio(
            "semantic_advantage_bootstrap_lower",
            self.semantic_advantage_bootstrap_lower,
            nonnegative=False,
        )
        if type(self.final_validation_version_all_statistical_gates_passed) is not bool:
            raise TypeError(
                "final_validation_version_all_statistical_gates_passed must be exact bool"
            )
        regret_point = Fraction(*self.scale_regret_point)
        regret_upper = Fraction(*self.scale_regret_bootstrap_upper)
        semantic_point = Fraction(*self.semantic_advantage_point)
        semantic_lower = Fraction(*self.semantic_advantage_bootstrap_lower)
        recomputed_all_gates = (
            all(self.overall_gate_passes)
            and all(self.slice_gate_passes)
            and all(self.challenge_gate_passes)
            and regret_point <= Fraction(1, 20)
            and regret_upper <= Fraction(2, 25)
            and semantic_point >= Fraction(15, 100)
            and semantic_lower >= Fraction(10, 100)
        )
        if (
            self.final_validation_version_all_statistical_gates_passed
            is not recomputed_all_gates
            or self.final_validation_version_all_statistical_gates_passed
            is not (self.outcome is StatisticalValidationOutcomeV1.COMPLETE_PASS)
        ):
            raise ValueError(
                "terminal version outcome differs from its recomputed statistical gates"
            )
        true_claims = (
            self.statistical_validation_candidate_mechanics_verified,
            self.original_r06_r07_bytes_canonically_bound,
            self.exact_cohort_and_quota_mechanics_verified,
            self.retry_lifecycle_mechanics_verified,
            self.six_purpose_signature_mechanics_verified,
            self.cas_zero_to_one_candidate_mechanics_verified,
            self.contiguous_declared_time_mechanics_verified,
            self.bootstrap_mechanics_verified,
            self.cooperative_computation_only,
            self.downstream_fresh_reverification_required,
        )
        if any(type(value) is not bool or not value for value in true_claims):
            raise ValueError("a mechanics claim is not exact true")
        false_claims = (
            self.r06_qualified,
            self.r07_frozen,
            self.r08_satisfied,
            self.actual_execution_verified,
            self.authority_verified,
            self.origin_authenticated,
            self.durable_cas_verified,
            self.authoritative_time_verified,
            self.result_object_unforgeable,
            self.filesystem_accessed,
            self.network_accessed,
            self.docker_invoked,
            self.entropy_used,
            self.keys_generated,
            self.signatures_generated,
            self.clock_consulted,
            self.store_mutated,
        )
        if any(type(value) is not bool or value for value in false_claims):
            raise ValueError("a prohibited readiness, authority, or side-effect claim is not false")
        expected_result_id = _content_id(
            _RESULT_PREFIX,
            _RESULT_DOMAIN,
            _result_projection_from_object(self),
        )
        if self.result_id != expected_result_id:
            raise ValueError("result content address differs from full canonical projection")


@dataclass(frozen=True, slots=True)
class StatisticalValidationCandidateRejectionV1:
    outcome: StatisticalValidationOutcomeV1
    reason: StatisticalValidationCandidateReasonV1
    version: None = None
    claim_level: None = None
    profile_id: None = None
    result_id: None = None
    input_snapshot_commitment_sha256: None = None
    verified_signature_count: int = 0
    partial_output_published: bool = False

    def __post_init__(self) -> None:
        if self.outcome is not StatisticalValidationOutcomeV1.REJECTED:
            raise ValueError("rejection outcome must be REJECTED")
        if type(self.verified_signature_count) is not int or self.verified_signature_count < 0:
            raise ValueError("verified signature count differs")
        if type(self.partial_output_published) is not bool or self.partial_output_published:
            raise ValueError("atomic rejection cannot publish partial output")


def _validate_result_bounded_nonrecursive_shape_v1(
    result: StatisticalValidationCandidateMechanicsResultV1,
) -> None:
    """Preflight the exact, shallow result shape before recursive projection."""

    if type(result) is not StatisticalValidationCandidateMechanicsResultV1:
        raise TypeError("fresh result verification needs the exact closed DTO")

    def value(name: str) -> object:
        return object.__getattribute__(result, name)

    if type(value("outcome")) is not StatisticalValidationOutcomeV1:
        raise TypeError("result outcome must be the exact enum")
    if type(value("reason")) is not StatisticalValidationCandidateReasonV1:
        raise TypeError("result reason must be the exact enum")

    for name in (
        "version",
        "claim_level",
        "profile_id",
        "result_id",
        "input_snapshot_commitment_sha256",
        "prerequisite_graph_root_sha256",
        "final_metric_bundle_root_sha256",
    ):
        text_value = value(name)
        if type(text_value) is not str or len(text_value) > _MAX_SHORT_ID_BYTES:
            raise TypeError(f"{name} must be a bounded exact string")
    version_bump_id = value("version_bump_id")
    if version_bump_id is not None and (
        type(version_bump_id) is not str
        or len(version_bump_id) > _MAX_SHORT_ID_BYTES
    ):
        raise TypeError("version_bump_id must be absent or a bounded exact string")

    for name in (
        "validation_version_count",
        "final_validation_version_index",
        "final_attempt_index",
        "verified_signature_count",
        "total_event_count",
        "aggregate_fail_closed_correct_count",
        "aggregate_fail_closed_denominator",
    ):
        integer = value(name)
        if (
            type(integer) is not int
            or integer < 0
            or integer > _MAX_SAFE_INTEGER
        ):
            raise TypeError(f"{name} must be a bounded exact integer")
    version_count = value("validation_version_count")
    if version_count not in {1, 2}:
        raise ValueError("validation_version_count must be one or two")

    for name in (
        "no_go_after_version_two_failure",
        "aggregate_fail_closed_passed",
        "final_validation_version_all_statistical_gates_passed",
        "statistical_validation_candidate_mechanics_verified",
        "original_r06_r07_bytes_canonically_bound",
        "exact_cohort_and_quota_mechanics_verified",
        "retry_lifecycle_mechanics_verified",
        "six_purpose_signature_mechanics_verified",
        "cas_zero_to_one_candidate_mechanics_verified",
        "contiguous_declared_time_mechanics_verified",
        "bootstrap_mechanics_verified",
        "cooperative_computation_only",
        "downstream_fresh_reverification_required",
        "r06_qualified",
        "r07_frozen",
        "r08_satisfied",
        "actual_execution_verified",
        "authority_verified",
        "origin_authenticated",
        "durable_cas_verified",
        "authoritative_time_verified",
        "result_object_unforgeable",
        "filesystem_accessed",
        "network_accessed",
        "docker_invoked",
        "entropy_used",
        "keys_generated",
        "signatures_generated",
        "clock_consulted",
        "store_mutated",
    ):
        if type(value(name)) is not bool:
            raise TypeError(f"{name} must be an exact boolean")

    for name in (
        "fail_closed_denominators_by_version",
        "fail_closed_correct_by_version",
    ):
        integers = value(name)
        if (
            type(integers) is not tuple
            or len(integers) != version_count
            or any(type(item) is not int for item in integers)
        ):
            raise TypeError(f"{name} must be a bounded exact integer tuple")

    vector = value("per_version_outcome_gate_root_vector")
    if type(vector) is not tuple or len(vector) != version_count:
        raise TypeError("per-version outcome/root vector has an unbounded shape")
    for row in vector:
        if (
            type(row) is not tuple
            or len(row) != 2
            or type(row[0]) is not StatisticalValidationOutcomeV1
            or type(row[1]) is not str
            or len(row[1]) > _MAX_SHORT_ID_BYTES
        ):
            raise TypeError("per-version outcome/root row has a non-closed shape")

    evaluator_counts = value("evaluator_manifest_row_counts")
    if (
        type(evaluator_counts) is not tuple
        or len(evaluator_counts) != 4
        or any(type(item) is not int for item in evaluator_counts)
    ):
        raise TypeError("evaluator counts must be four exact integers")
    for name, expected_length in (
        ("overall_gate_passes", 12),
        ("slice_gate_passes", 24),
        ("challenge_gate_passes", 2),
    ):
        gates = value(name)
        if (
            type(gates) is not tuple
            or len(gates) != expected_length
            or any(type(item) is not bool for item in gates)
        ):
            raise TypeError(f"{name} must have its exact shallow boolean shape")

    for name in (
        "scale_regret_point",
        "scale_regret_bootstrap_upper",
        "semantic_advantage_point",
        "semantic_advantage_bootstrap_lower",
    ):
        ratio = value(name)
        if (
            type(ratio) is not tuple
            or len(ratio) != 2
            or type(ratio[0]) is not int
            or type(ratio[1]) is not int
        ):
            raise TypeError(f"{name} must be one exact shallow integer pair")


def _result_json_value(value: object) -> object:
    """Project one result value without invoking caller-defined equality/coercion."""

    if value is None:
        return None
    if type(value) is bool:
        return value
    if type(value) is int:
        return value
    if type(value) is str:
        return value
    if type(value) in {
        StatisticalValidationOutcomeV1,
        StatisticalValidationCandidateReasonV1,
    }:
        return value.value  # type: ignore[union-attr]
    if type(value) is tuple:
        return [_result_json_value(item) for item in value]
    raise TypeError("result projection contains a non-closed value type")


def _complete_result_values(values: dict[str, object]) -> dict[str, object]:
    """Fill declared boolean defaults while rejecting unknown/missing fields."""

    if type(values) is not dict:
        raise TypeError("result values must be an exact dict")
    result_fields = tuple(
        field
        for field in fields(StatisticalValidationCandidateMechanicsResultV1)
        if field.name != "result_id"
    )
    expected_names = {field.name for field in result_fields}
    if not set(values).issubset(expected_names):
        raise ValueError("result values contain an unknown or self-address field")
    completed: dict[str, object] = {}
    for field in result_fields:
        if field.name in values:
            completed[field.name] = values[field.name]
        elif field.default is not MISSING:
            completed[field.name] = field.default
        else:
            raise ValueError(f"result values omit required field {field.name}")
    return completed


def _result_projection_from_values(values: dict[str, object]) -> dict[str, object]:
    expected_names = tuple(
        field.name
        for field in fields(StatisticalValidationCandidateMechanicsResultV1)
        if field.name != "result_id"
    )
    if type(values) is not dict or set(values) != set(expected_names):
        raise ValueError("result projection field closure differs")
    return {name: _result_json_value(values[name]) for name in expected_names}


def _result_projection_from_object(
    result: StatisticalValidationCandidateMechanicsResultV1,
) -> dict[str, object]:
    if type(result) is not StatisticalValidationCandidateMechanicsResultV1:
        raise TypeError("result projection needs the exact closed DTO")
    values = {
        field.name: object.__getattribute__(result, field.name)
        for field in fields(StatisticalValidationCandidateMechanicsResultV1)
        if field.name != "result_id"
    }
    return _result_projection_from_values(values)


def _seed_json(item: ValidationSeedRootCandidateV1) -> dict[str, object]:
    return {
        "domain_commitment_sha256": item.domain_commitment_sha256.hex(),
        "purpose": item.purpose.value,
        "raw_seed_root": item.raw_seed_root.hex(),
    }


def _cohort_json(item: ValidationCohortBindingV1) -> dict[str, object]:
    return {
        "challenge_count": item.challenge_count,
        "generation_receipt_root_sha256": item.generation_receipt_root_sha256.hex(),
        "invalid_variant_count": item.invalid_variant_count,
        "legal_variant_count": item.legal_variant_count,
        "main_count": item.main_count,
        "preservation_batch_root_sha256": item.preservation_batch_root_sha256.hex(),
        "preservation_count": item.preservation_count,
        "preservation_original_bytes_sha256": item.preservation_original_bytes_sha256.hex(),
        "prerequisite_graph_root_sha256": item.prerequisite_graph_root_sha256.hex(),
        "primary_batch_root_sha256": item.primary_batch_root_sha256.hex(),
        "primary_count": item.primary_count,
        "primary_original_bytes_sha256": item.primary_original_bytes_sha256.hex(),
        "statistical_validation_profile_id": item.statistical_validation_profile_id,
    }


def _cohort_root(item: ValidationCohortBindingV1) -> bytes:
    return _digest_json(_COHORT_DOMAIN, _cohort_json(item))


def _system_run_json(item: ValidationSystemRunCandidateV1) -> dict[str, object]:
    return {
        "preservation_count": item.preservation_count,
        "preservation_prediction_root_sha256": item.preservation_prediction_root_sha256.hex(),
        "primary_count": item.primary_count,
        "primary_prediction_root_sha256": item.primary_prediction_root_sha256.hex(),
        "raw_output_byte_count": item.raw_output_byte_count,
        "system": item.system.value,
        "total_count": item.total_count,
    }


def _system_runs_root(items: tuple[ValidationSystemRunCandidateV1, ...]) -> bytes:
    return _digest_json(_SYSTEM_RUNS_DOMAIN, [_system_run_json(item) for item in items])


def _binary_gate_json(item: BinaryGateCandidateV1) -> dict[str, object]:
    return {
        "correct_count": item.correct_count,
        "metric": item.metric,
        "scope_axis": item.scope_axis,
        "scope_value": item.scope_value,
        "total_count": item.total_count,
    }


def _metric_bundle_json(item: StatisticalMetricBundleCandidateV1) -> dict[str, object]:
    return {
        "challenge_rows": [
            {
                "embedding_correct": row.embedding_correct,
                "family": row.family,
                "frozen_llm_correct": row.frozen_llm_correct,
                "challenge_kind": row.challenge_kind.value,
                "cell_ordinal": row.cell_ordinal,
                "input_row_id": row.input_row_id,
                "scale": row.scale,
                "system_correct": row.system_correct,
            }
            for row in item.challenge_rows
        ],
        "main_rows": [
            {
                "binding_exact": row.binding_exact,
                "case_type": row.case_type.value,
                "cell_ordinal": row.cell_ordinal,
                "decision_exact": row.decision_exact,
                "family": row.family,
                "family_exact": row.family_exact,
                "input_row_id": row.input_row_id,
                "joint_exact": row.joint_exact,
                "predicted_decision": row.predicted_decision.value,
                "scale": row.scale,
                "scale_set_exact": row.scale_set_exact,
            }
            for row in item.main_rows
        ],
        "overall_gates": [_binary_gate_json(row) for row in item.overall_gates],
        "preservation_rows": [
            {
                "kind_ordinal": row.kind_ordinal,
                "preservation_kind": row.preservation_kind.value,
                "recognizer_correct": row.recognizer_correct,
                "variant_row_id": row.variant_row_id,
            }
            for row in item.preservation_rows
        ],
        "scale_regret_rows": [
            {
                "family": row.family,
                "input_row_id": row.input_row_id,
                "regret": [row.regret.numerator, row.regret.denominator],
                "scale": row.scale,
            }
            for row in item.scale_regret_rows
        ],
        "slice_gates": [_binary_gate_json(row) for row in item.slice_gates],
    }


def _metric_bundle_root(item: StatisticalMetricBundleCandidateV1) -> bytes:
    return _digest_json(_METRIC_BUNDLE_DOMAIN, _metric_bundle_json(item))


def _cas_subject_json(item: ValidationCasReceiptCandidateV1) -> dict[str, object]:
    return {
        "attempt_index": item.attempt_index,
        "new_generation": item.new_generation,
        "object_key": item.object_key,
        "previous_generation": item.previous_generation,
        "store_key_id": item.store_key_id,
        "upload_digest_attempts": [value.hex() for value in item.upload_digest_attempts],
        "upload_retry_action": item.upload_retry_action.value,
        "upload_retry_reason": item.upload_retry_reason.value,
        "validation_version_index": item.validation_version_index,
        "value_digest_sha256": item.value_digest_sha256.hex(),
    }


def _cas_receipt_id(item: ValidationCasReceiptCandidateV1) -> str:
    return _content_id(_CAS_PREFIX, _CAS_DOMAIN, _cas_subject_json(item))


def _cas_signature_message(item: ValidationCasReceiptCandidateV1) -> bytes:
    return _CAS_SIGNATURE_DOMAIN + canonicalize_rfc8785_v1(_cas_subject_json(item))


def _event_subject_json(item: ValidationSignedEventCandidateV1) -> dict[str, object]:
    return {
        "attempt_index": item.attempt_index,
        "declared_nanosecond": item.declared_nanosecond,
        "declared_unix_seconds": item.declared_unix_seconds,
        "event_kind": item.event_kind.value,
        "logical_time": item.logical_time,
        "payload_root_sha256": item.payload_root_sha256.hex(),
        "previous_event_id": item.previous_event_id,
        "sequence": item.sequence,
        "signer_key_id": item.signer_key_id,
        "validation_version_index": item.validation_version_index,
    }


def _event_id(item: ValidationSignedEventCandidateV1) -> str:
    return _content_id(_EVENT_PREFIX, _EVENT_DOMAIN, _event_subject_json(item))


def _event_signature_message(item: ValidationSignedEventCandidateV1) -> bytes:
    return _EVENT_SIGNATURE_DOMAIN + canonicalize_rfc8785_v1(_event_subject_json(item))


def _prerequisite_root(item: OriginalPrerequisiteGraphV1) -> bytes:
    return _digest_json(
        _PREREQUISITE_DOMAIN,
        {
            "r06_original_candidate_sha256": hashlib.sha256(
                item.r06_original_candidate_bytes
            ).hexdigest(),
            "r06_original_result_sha256": hashlib.sha256(
                item.r06_original_result_bytes
            ).hexdigest(),
            "r07_original_candidate_sha256": hashlib.sha256(
                item.r07_original_candidate_bytes
            ).hexdigest(),
            "r07_original_result_sha256": hashlib.sha256(
                item.r07_original_result_bytes
            ).hexdigest(),
        },
    )


def _version_bump_core_subject_json(
    item: StatisticalValidationVersionBumpCandidateV1,
) -> dict[str, object]:
    return {
        "from_validation_version_index": item.from_validation_version_index,
        "immutable_version_one_failure_root_sha256": (
            item.immutable_version_one_failure_root_sha256.hex()
        ),
        "new_protocol_version_id": item.new_protocol_version_id,
        "new_selector_version_id": item.new_selector_version_id,
        "new_validation_version_id": item.new_validation_version_id,
        "old_protocol_version_id": item.old_protocol_version_id,
        "old_selector_version_id": item.old_selector_version_id,
        "old_validation_version_id": item.old_validation_version_id,
        "public_development_change_set_sha256": (
            item.public_development_change_set_sha256.hex()
        ),
        "to_validation_version_index": item.to_validation_version_index,
        "version_one_failure_cas_receipt_id": item.version_one_failure_cas_receipt_id,
        "version_one_full_attempt_chain_root_sha256": (
            item.version_one_full_attempt_chain_root_sha256.hex()
        ),
        "version_two_cohort_root_sha256": item.version_two_cohort_root_sha256.hex(),
        "version_two_seed_roots": [value.hex() for value in item.version_two_seed_roots],
    }


def _version_bump_core_id(item: StatisticalValidationVersionBumpCandidateV1) -> str:
    return _content_id(
        _VERSION_BUMP_CORE_PREFIX,
        _VERSION_BUMP_CORE_DOMAIN,
        _version_bump_core_subject_json(item),
    )


def _version_bump_internal_bindings_json(
    item: StatisticalValidationVersionBumpCandidateV1,
    *,
    version_one_final: StatisticalValidationAttemptCandidateV1,
    version_two_first: StatisticalValidationAttemptCandidateV1,
) -> dict[str, object]:
    return {
        "immutable_version_one_failure_root_sha256": (
            item.immutable_version_one_failure_root_sha256.hex()
        ),
        "new_version_ids": {
            "protocol_version_id": item.new_protocol_version_id,
            "selector_version_id": item.new_selector_version_id,
            "validation_version_id": item.new_validation_version_id,
        },
        "old_version_ids": {
            "protocol_version_id": item.old_protocol_version_id,
            "selector_version_id": item.old_selector_version_id,
            "validation_version_id": item.old_validation_version_id,
        },
        "version_bump_core_id": item.version_bump_core_id,
        "version_one_cohort_root_sha256": _cohort_root(version_one_final.cohort).hex(),
        "version_one_failure_cas_receipt_id": item.version_one_failure_cas_receipt_id,
        "version_one_full_attempt_chain_root_sha256": (
            item.version_one_full_attempt_chain_root_sha256.hex()
        ),
        "version_one_seed_roots": [
            row.raw_seed_root.hex() for row in version_one_final.seed_roots
        ],
        "version_two_cohort_root_sha256": _cohort_root(version_two_first.cohort).hex(),
        "version_two_seed_roots": [
            row.raw_seed_root.hex() for row in version_two_first.seed_roots
        ],
    }


def _public_failure_development_report_payload(
    item: StatisticalValidationVersionBumpCandidateV1,
    *,
    version_one_final: StatisticalValidationAttemptCandidateV1,
    version_two_first: StatisticalValidationAttemptCandidateV1,
) -> dict[str, object]:
    return {
        "bindings": _version_bump_internal_bindings_json(
            item,
            version_one_final=version_one_final,
            version_two_first=version_two_first,
        ),
        "development": {
            "change_set_sha256": item.public_development_change_set_sha256.hex(),
            "input_scope": "PUBLIC_OR_SYNTHETIC_ONLY",
            "private_validation_inputs_accessed": False,
        },
        "disposition": "VERSION_ONE_COMPLETE_FAIL_PUBLIC_DEVELOPMENT_RECORDED",
        "report_visibility": "PUBLIC",
        "schema_version": _PUBLIC_FAILURE_DEVELOPMENT_REPORT_SCHEMA,
    }


def _version_two_issuance_authority_payload(
    item: StatisticalValidationVersionBumpCandidateV1,
    *,
    version_one_final: StatisticalValidationAttemptCandidateV1,
    version_two_first: StatisticalValidationAttemptCandidateV1,
) -> dict[str, object]:
    return {
        "authority": {
            "authenticated_authority_claimed": False,
            "issuance_kind": "CANDIDATE_VERSION_TWO_ONLY",
            "permitted_validation_version_index": 2,
            "permitted_validation_version_id": item.new_validation_version_id,
        },
        "bindings": _version_bump_internal_bindings_json(
            item,
            version_one_final=version_one_final,
            version_two_first=version_two_first,
        ),
        "schema_version": _VERSION_TWO_ISSUANCE_AUTHORITY_SCHEMA,
    }


def _version_bump_subject_json(
    item: StatisticalValidationVersionBumpCandidateV1,
) -> dict[str, object]:
    return {
        **_version_bump_core_subject_json(item),
        "public_failure_development_report_sha256": hashlib.sha256(
            item.public_failure_development_report_bytes
        ).hexdigest(),
        "version_bump_core_id": item.version_bump_core_id,
        "version_two_issuance_authority_sha256": hashlib.sha256(
            item.version_two_issuance_authority_bytes
        ).hexdigest(),
    }


def _version_bump_id(item: StatisticalValidationVersionBumpCandidateV1) -> str:
    return _content_id(
        _VERSION_BUMP_PREFIX,
        _VERSION_BUMP_DOMAIN,
        _version_bump_subject_json(item),
    )


def _attempt_archive_root(
    item: StatisticalValidationAttemptCandidateV1,
) -> bytes:
    execution_root = (
        _system_runs_root(item.system_runs)
        if item.system_runs
        else _digest_json(
            _INFRA_DOMAIN,
            {
                "any_raw_prediction_byte_produced": item.any_raw_prediction_byte_produced,
                "attempt_index": item.attempt_index,
                "execution_retry_action": item.execution_retry_action.value,
                "execution_retry_reason": item.execution_retry_reason.value,
                "terminal_code": item.terminal_code.value,
                "validation_version_index": item.validation_version_index,
            },
        )
    )
    metric_root = (
        _metric_bundle_root(item.metric_bundle)
        if item.metric_bundle is not None
        else execution_root
    )
    return _digest_json(
        _ATTEMPT_ARCHIVE_DOMAIN,
        {
            "attempt_index": item.attempt_index,
            "cohort_root_sha256": _cohort_root(item.cohort).hex(),
            "any_raw_prediction_byte_produced": item.any_raw_prediction_byte_produced,
            "execution_retry_action": item.execution_retry_action.value,
            "execution_retry_reason": item.execution_retry_reason.value,
            "execution_root_sha256": execution_root.hex(),
            "metric_root_sha256": metric_root.hex(),
            "observer_transcript_root_sha256": item.observer_transcript_root_sha256.hex(),
            "raw_output_byte_count": item.raw_output_byte_count,
            "seed_commitments": [row.domain_commitment_sha256.hex() for row in item.seed_roots],
            "terminal_code": item.terminal_code.value,
            "validation_version_index": item.validation_version_index,
        },
    )


def _version_attempt_chain_root(
    item: StatisticalValidationVersionCandidateV1,
) -> bytes:
    """Bind every v1 attempt, archive, signed event, and CAS receipt in order."""

    item.__post_init__()
    return _digest_json(
        _VERSION_ATTEMPT_CHAIN_ROOT_DOMAIN,
        {
            "attempts_in_order": [
                {
                    "attempt_archive_root_sha256": _attempt_archive_root(
                        attempt
                    ).hex(),
                    "attempt_index": attempt.attempt_index,
                    "cas_receipt_id": attempt.cas_receipt.receipt_id,
                    "cas_signature_sha256": hashlib.sha256(
                        attempt.cas_receipt.signature
                    ).hexdigest(),
                    "cas_subject_sha256": hashlib.sha256(
                        canonicalize_rfc8785_v1(
                            _cas_subject_json(attempt.cas_receipt)
                        )
                    ).hexdigest(),
                    "event_ids_in_order": [
                        event.event_id for event in attempt.events
                    ],
                    "terminal_code": attempt.terminal_code.value,
                    "value_digest_sha256": (
                        attempt.cas_receipt.value_digest_sha256.hex()
                    ),
                }
                for attempt in item.attempts
            ],
            "validation_version_index": item.validation_version_index,
        },
    )


def _expected_event_payloads(
    item: StatisticalValidationAttemptCandidateV1,
) -> tuple[bytes, ...]:
    execution_root = (
        _system_runs_root(item.system_runs)
        if item.system_runs
        else _digest_json(
            _INFRA_DOMAIN,
            {
                "any_raw_prediction_byte_produced": item.any_raw_prediction_byte_produced,
                "attempt_index": item.attempt_index,
                "execution_retry_action": item.execution_retry_action.value,
                "execution_retry_reason": item.execution_retry_reason.value,
                "terminal_code": item.terminal_code.value,
                "validation_version_index": item.validation_version_index,
            },
        )
    )
    metric_root = (
        _metric_bundle_root(item.metric_bundle)
        if item.metric_bundle is not None
        else execution_root
    )
    return (
        _cohort_root(item.cohort),
        execution_root,
        item.observer_transcript_root_sha256,
        metric_root,
        _digest_id(item.cas_receipt.receipt_id),
        _attempt_archive_root(item),
    )


def paired_cluster_bootstrap_indices_v1(
    *,
    cohort_root_sha256: bytes,
    metric_name: str,
    replicate_index: int,
    cluster_count: int = 12,
) -> tuple[int, ...]:
    """Return one deterministic unbiased paired-cluster bootstrap sample.

    This is a pure HMAC-SHA256 sampler.  It performs rejection rather than
    modulo reduction on the uint32 stream, so non-power-of-two cluster counts
    are unbiased.  No process RNG or external state is consulted.
    """

    _require_bytes(cohort_root_sha256, "cohort_root_sha256", _RAW_DIGEST_BYTES)
    _require_text(metric_name, "metric_name")
    _require_exact_int(replicate_index, "replicate_index")
    if type(cluster_count) is not int or not 1 <= cluster_count <= 65_536:
        raise ValueError("cluster_count must be an exact integer in [1, 65536]")
    key = _BOOTSTRAP_MASTER_SEED.to_bytes(8, "big", signed=False)
    limit = (1 << 32) - ((1 << 32) % cluster_count)
    output: list[int] = []
    draw_index = 0
    while len(output) < cluster_count:
        rejection_counter = 0
        while True:
            message = (
                _BOOTSTRAP_DOMAIN
                + cohort_root_sha256
                + len(metric_name.encode("utf-8")).to_bytes(2, "big")
                + metric_name.encode("utf-8")
                + replicate_index.to_bytes(8, "big")
                + draw_index.to_bytes(4, "big")
                + rejection_counter.to_bytes(4, "big")
            )
            candidate = int.from_bytes(hmac.new(key, message, hashlib.sha256).digest()[:4], "big")
            if candidate < limit:
                output.append(candidate % cluster_count)
                break
            rejection_counter += 1
            if rejection_counter > 0xFFFF_FFFF:
                raise OverflowError("bootstrap rejection counter exhausted")
        draw_index += 1
    return tuple(output)


def _wilson_lower(correct: int, total: int) -> float:
    if total <= 0 or not 0 <= correct <= total:
        raise ValueError("Wilson counts are invalid")
    p = correct / total
    z2 = _WILSON_Z * _WILSON_Z
    denominator = 1.0 + z2 / total
    centre = p + z2 / (2.0 * total)
    radius = _WILSON_Z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)
    return (centre - radius) / denominator


def _binary_gate_pass(
    row: BinaryGateCandidateV1,
    *,
    point_n: int,
    point_d: int,
    lower_n: int | None,
    lower_d: int | None,
) -> bool:
    if row.correct_count * point_d < row.total_count * point_n:
        return False
    if lower_n is None:
        return True
    if lower_d is None:
        raise ValueError("Wilson lower denominator is absent")
    return _wilson_lower(row.correct_count, row.total_count) >= lower_n / lower_d


def _expected_slice_layout() -> tuple[tuple[str, str, str, int, int, int, int, int], ...]:
    rows: list[tuple[str, str, str, int, int, int, int, int]] = []
    for axis, values in (("family", _FAMILIES), ("scale", _SCALES)):
        for value in values:
            for metric, point_n, point_d, lower_n, lower_d in _SLICE_GATE_BASE_SPECS:
                denominator = {
                    ("family", "answerable_joint_exact"): 40,
                    ("family", "all_control_rejection"): 64,
                    ("family", "abstention_specificity"): 38,
                    ("scale", "answerable_joint_exact"): 120,
                    ("scale", "all_control_rejection"): 192,
                    ("scale", "abstention_specificity"): 114,
                }[(axis, metric)]
                rows.append(
                    (metric, axis, value, denominator, point_n, point_d, lower_n, lower_d)
                )
    return tuple(rows)


def _validate_metric_layout(
    bundle: StatisticalMetricBundleCandidateV1,
    *,
    total_event_count: int,
) -> tuple[
    tuple[BinaryGateCandidateV1, ...],
    tuple[BinaryGateCandidateV1, ...],
    int,
    int,
]:
    _require_exact_int(total_event_count, "total_event_count", 1)
    expected_main_layout = tuple(
        (family, scale, case_type, ordinal)
        for family in _FAMILIES
        for scale in _SCALES
        for case_type, count in _CASE_TYPE_CELL_COUNTS
        for ordinal in range(count)
    )
    if tuple(
        (row.family, row.scale, row.case_type, row.cell_ordinal)
        for row in bundle.main_rows
    ) != expected_main_layout:
        raise ValueError("main evaluator row quota/order differs")
    main_ids = tuple(row.input_row_id for row in bundle.main_rows)
    if len(set(main_ids)) != _MAIN_COUNT:
        raise ValueError("main evaluator input row ids are not unique")

    expected_challenge_layout = tuple(
        (family, scale, kind, ordinal)
        for family in _FAMILIES
        for scale in _SCALES
        for kind in ValidationChallengeKindV1
        for ordinal in range(10)
    )
    if tuple(
        (row.family, row.scale, row.challenge_kind, row.cell_ordinal)
        for row in bundle.challenge_rows
    ) != expected_challenge_layout:
        raise ValueError("challenge evaluator row quota/order differs")
    challenge_ids = tuple(row.input_row_id for row in bundle.challenge_rows)
    if (
        len(set(challenge_ids)) != _CHALLENGE_COUNT
        or set(main_ids) & set(challenge_ids)
    ):
        raise ValueError("primary evaluator row ids are replayed or overlap")

    expected_preservation_layout = tuple(
        (ValidationPreservationKindV1.LEGAL, ordinal)
        for ordinal in range(_LEGAL_VARIANT_COUNT)
    ) + tuple(
        (ValidationPreservationKindV1.INVALID_CONTROL, ordinal)
        for ordinal in range(_INVALID_VARIANT_COUNT)
    )
    if tuple(
        (row.preservation_kind, row.kind_ordinal)
        for row in bundle.preservation_rows
    ) != expected_preservation_layout:
        raise ValueError("preservation evaluator row quota/order differs")
    preservation_ids = tuple(row.variant_row_id for row in bundle.preservation_rows)
    if (
        len(set(preservation_ids)) != _VARIANT_COUNT
        or (set(main_ids) | set(challenge_ids)) & set(preservation_ids)
    ):
        raise ValueError("preservation evaluator row ids are replayed or overlap")

    answerable_rows = tuple(
        row for row in bundle.main_rows if row.case_type in _ANSWERABLE_CASE_TYPES
    )
    if len(answerable_rows) != 240:
        raise ValueError("answerable evaluator row denominator differs")
    if tuple(
        (row.input_row_id, row.family, row.scale)
        for row in bundle.scale_regret_rows
    ) != tuple(
        (row.input_row_id, row.family, row.scale) for row in answerable_rows
    ):
        raise ValueError("scale-regret rows are not the exact answerable row projection")

    def abstained(row: MainEvaluationRowV1) -> bool:
        return row.predicted_decision is ValidationPredictionDecisionV1.ABSTAIN

    main = bundle.main_rows
    overall_counts = {
        "family_exact": sum(row.family_exact is True for row in answerable_rows),
        "binding_exact": sum(row.binding_exact is True for row in answerable_rows),
        "scale_set_accuracy": sum(row.scale_set_exact is True for row in answerable_rows),
        "joint_exact": sum(row.joint_exact for row in answerable_rows),
        "hard_negative_rejection": sum(
            abstained(row)
            for row in main
            if row.case_type is ValidationMainCaseTypeV1.WRONG_FAMILY_HARD_NEGATIVE
        ),
        "binding_counterfactual_rejection": sum(
            abstained(row)
            for row in main
            if row.case_type is ValidationMainCaseTypeV1.BINDING_COUNTERFACTUAL
        ),
        "scale_counterfactual_rejection": sum(
            abstained(row)
            for row in main
            if row.case_type is ValidationMainCaseTypeV1.SCALE_COUNTERFACTUAL
        ),
        "sign_or_invariant_break_rejection": sum(
            abstained(row)
            for row in main
            if row.case_type is ValidationMainCaseTypeV1.SIGN_OR_INVARIANT_BREAK
        ),
        "abstention_specificity": sum(
            not abstained(row)
            for row in main
            if row.case_type is ValidationMainCaseTypeV1.UNIQUE_SCALE_ANSWERABLE
        ),
        "fail_closed_rate": total_event_count,
        "preservation_consistency": sum(
            row.recognizer_correct for row in bundle.preservation_rows
        ),
        "nonidentifiable_scale_abstention": sum(
            abstained(row)
            for row in main
            if row.case_type
            is ValidationMainCaseTypeV1.INSUFFICIENT_OR_NONIDENTIFIABLE
        ),
    }
    recomputed_overall = tuple(
        BinaryGateCandidateV1(
            metric=name,
            scope_axis="overall",
            scope_value="all",
            correct_count=overall_counts[name],
            total_count=(total_event_count if denominator is None else denominator),
        )
        for name, denominator, *_rest in _OVERALL_GATE_SPECS
    )

    recomputed_slices: list[BinaryGateCandidateV1] = []
    for metric, axis, value, denominator, *_thresholds in _expected_slice_layout():
        members = tuple(
            row
            for row in main
            if (row.family if axis == "family" else row.scale) == value
        )
        if metric == "answerable_joint_exact":
            correct = sum(
                row.joint_exact
                for row in members
                if row.case_type in _ANSWERABLE_CASE_TYPES
            )
        elif metric == "all_control_rejection":
            correct = sum(
                abstained(row)
                for row in members
                if row.case_type in _CONTROL_CASE_TYPES
            )
        elif metric == "abstention_specificity":
            correct = sum(
                not abstained(row)
                for row in members
                if row.case_type is ValidationMainCaseTypeV1.UNIQUE_SCALE_ANSWERABLE
            )
        else:  # pragma: no cover - closed by _expected_slice_layout.
            raise RuntimeError("unknown slice metric")
        recomputed_slices.append(
            BinaryGateCandidateV1(
                metric=metric,
                scope_axis=axis,
                scope_value=value,
                correct_count=correct,
                total_count=denominator,
            )
        )
    recomputed_slice_tuple = tuple(recomputed_slices)

    def gate_projection(rows: tuple[BinaryGateCandidateV1, ...]) -> bytes:
        return canonicalize_rfc8785_v1([_binary_gate_json(row) for row in rows])

    if gate_projection(bundle.overall_gates) != gate_projection(recomputed_overall):
        raise ValueError("reported overall gates differ from evaluator-row recomputation")
    if gate_projection(bundle.slice_gates) != gate_projection(recomputed_slice_tuple):
        raise ValueError("reported slice gates differ from evaluator-row recomputation")

    family_slices = tuple(row for row in recomputed_slice_tuple if row.scope_axis == "family")
    scale_slices = tuple(row for row in recomputed_slice_tuple if row.scope_axis == "scale")
    for metric, overall_name in (
        ("answerable_joint_exact", "joint_exact"),
        ("abstention_specificity", "abstention_specificity"),
    ):
        overall_value = overall_counts[overall_name]
        if (
            sum(row.correct_count for row in family_slices if row.metric == metric)
            != overall_value
            or sum(row.correct_count for row in scale_slices if row.metric == metric)
            != overall_value
        ):
            raise ValueError("family/scale slice sums do not close to overall")
    total_control_correct = sum(
        overall_counts[name]
        for name in (
            "hard_negative_rejection",
            "binding_counterfactual_rejection",
            "scale_counterfactual_rejection",
            "sign_or_invariant_break_rejection",
        )
    )
    if (
        sum(
            row.correct_count
            for row in family_slices
            if row.metric == "all_control_rejection"
        )
        != total_control_correct
        or sum(
            row.correct_count
            for row in scale_slices
            if row.metric == "all_control_rejection"
        )
        != total_control_correct
    ):
        raise ValueError("control rejection slice sums do not close")
    if not (
        overall_counts["joint_exact"] <= overall_counts["family_exact"]
        and overall_counts["joint_exact"] <= overall_counts["binding_exact"]
        and overall_counts["joint_exact"] <= overall_counts["scale_set_accuracy"]
    ):
        raise ValueError("joint exact count exceeds a component exact count")

    legal_correct = sum(
        row.recognizer_correct
        for row in bundle.preservation_rows
        if row.preservation_kind is ValidationPreservationKindV1.LEGAL
    )
    invalid_correct = sum(
        row.recognizer_correct
        for row in bundle.preservation_rows
        if row.preservation_kind is ValidationPreservationKindV1.INVALID_CONTROL
    )
    return recomputed_overall, recomputed_slice_tuple, legal_correct, invalid_correct


def _nearest_rank_value(values: list[Fraction], index: int) -> Fraction:
    if type(values) is not list or not values:
        raise TypeError("nearest-rank values must be a nonempty exact list")
    if (
        type(index) is not int
        or not 0 <= index < len(values)
        or any(type(value) is not Fraction for value in values)
    ):
        raise TypeError("nearest-rank index or value type differs")
    values.sort()
    return values[index]


def _evaluate_metric_bundle(
    bundle: StatisticalMetricBundleCandidateV1,
    *,
    cohort_root_sha256: bytes,
    total_event_count: int,
) -> tuple[
    tuple[bool, ...],
    tuple[bool, ...],
    tuple[bool, bool],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
    bool,
]:
    _, _, legal_correct, invalid_correct = _validate_metric_layout(
        bundle,
        total_event_count=total_event_count,
    )
    overall_passes = []
    for row, spec in zip(bundle.overall_gates, _OVERALL_GATE_SPECS):
        overall_passes.append(
            _binary_gate_pass(
                row,
                point_n=spec[2],
                point_d=spec[3],
                lower_n=spec[4],
                lower_d=spec[5],
            )
        )
    preservation_pass = (
        legal_correct >= 482
        and _wilson_lower(legal_correct, _LEGAL_VARIANT_COUNT) >= 0.94
        and invalid_correct == _INVALID_VARIANT_COUNT
    )
    overall_passes[10] = overall_passes[10] and preservation_pass

    slice_passes = []
    for row, spec in zip(bundle.slice_gates, _expected_slice_layout()):
        slice_passes.append(
            _binary_gate_pass(
                row,
                point_n=spec[4],
                point_d=spec[5],
                lower_n=spec[6],
                lower_d=spec[7],
            )
        )

    cluster_sums = tuple(
        sum(
            (
                row.regret.fraction()
                for row in bundle.scale_regret_rows
                if row.family == family and row.scale == scale
            ),
            Fraction(0, 1),
        )
        for family in _FAMILIES
        for scale in _SCALES
    )
    point_regret = sum(cluster_sums, Fraction(0, 1)) / 240
    regret_replicates: list[Fraction] = []
    for replicate in range(_BOOTSTRAP_REPLICATES):
        indices = paired_cluster_bootstrap_indices_v1(
            cohort_root_sha256=cohort_root_sha256,
            metric_name="normalized_scale_decision_regret",
            replicate_index=replicate,
        )
        regret_replicates.append(
            sum((cluster_sums[index] for index in indices), Fraction(0, 1)) / 240
        )
    regret_upper = _nearest_rank_value(regret_replicates, _BOOTSTRAP_UCB_INDEX)
    regret_pass = point_regret <= Fraction(1, 20) and regret_upper <= Fraction(2, 25)

    challenge = bundle.challenge_rows
    low_correct = sum(
        row.system_correct
        for row in challenge
        if row.challenge_kind
        is ValidationChallengeKindV1.LOW_OVERLAP_STRUCTURAL_POSITIVE
    )
    high_correct = sum(
        row.system_correct
        for row in challenge
        if row.challenge_kind
        is ValidationChallengeKindV1.HIGH_OVERLAP_STRUCTURAL_NEGATIVE
    )
    low_pass = (
        low_correct >= 109
        and low_correct * 100 >= 120 * 90
        and _wilson_lower(low_correct, 120) >= 0.85
    )
    high_pass = (
        high_correct >= 114
        and high_correct * 100 >= 120 * 95
        and _wilson_lower(high_correct, 120) >= 0.90
    )
    system_total = sum(row.system_correct for row in challenge)
    embedding_total = sum(row.embedding_correct for row in challenge)
    llm_total = sum(row.frozen_llm_correct for row in challenge)
    semantic_point = Fraction(system_total - max(embedding_total, llm_total), 240)
    challenge_cluster_counts = tuple(
        (
            sum(
                row.system_correct
                for row in challenge
                if row.family == family and row.scale == scale
            ),
            sum(
                row.embedding_correct
                for row in challenge
                if row.family == family and row.scale == scale
            ),
            sum(
                row.frozen_llm_correct
                for row in challenge
                if row.family == family and row.scale == scale
            ),
        )
        for family in _FAMILIES
        for scale in _SCALES
    )
    semantic_replicates: list[Fraction] = []
    for replicate in range(_BOOTSTRAP_REPLICATES):
        indices = paired_cluster_bootstrap_indices_v1(
            cohort_root_sha256=cohort_root_sha256,
            metric_name="semantic_only_advantage",
            replicate_index=replicate,
        )
        sampled_system = sum(challenge_cluster_counts[index][0] for index in indices)
        sampled_embedding = sum(challenge_cluster_counts[index][1] for index in indices)
        sampled_llm = sum(challenge_cluster_counts[index][2] for index in indices)
        semantic_replicates.append(
            Fraction(sampled_system - max(sampled_embedding, sampled_llm), 240)
        )
    semantic_lower = _nearest_rank_value(semantic_replicates, _BOOTSTRAP_LCB_INDEX)
    semantic_pass = semantic_point >= Fraction(15, 100) and semantic_lower >= Fraction(10, 100)

    all_pass = (
        all(overall_passes)
        and all(slice_passes)
        and regret_pass
        and low_pass
        and high_pass
        and semantic_pass
    )
    return (
        tuple(overall_passes),
        tuple(slice_passes),
        (low_pass, high_pass),
        (point_regret.numerator, point_regret.denominator),
        (regret_upper.numerator, regret_upper.denominator),
        (semantic_point.numerator, semantic_point.denominator),
        (semantic_lower.numerator, semantic_lower.denominator),
        all_pass,
    )


def _version_gate_evaluation_root(
    *,
    validation_version_index: int,
    final_attempt: StatisticalValidationAttemptCandidateV1,
    version_event_count: int,
    evaluation: tuple[
        tuple[bool, ...],
        tuple[bool, ...],
        tuple[bool, bool],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        bool,
    ],
) -> bytes:
    outcome = (
        StatisticalValidationOutcomeV1.COMPLETE_PASS
        if evaluation[7]
        else StatisticalValidationOutcomeV1.COMPLETE_FAIL
    )
    if (
        final_attempt.terminal_code is ValidationAttemptTerminalCodeV1.COMPLETE_PASS
    ) is not evaluation[7]:
        raise ValueError("version gate root terminal code differs")
    if type(final_attempt.metric_bundle) is not StatisticalMetricBundleCandidateV1:
        raise TypeError("version gate root requires the terminal metric bundle")
    return _digest_json(
        _VERSION_GATE_ROOT_DOMAIN,
        {
            "challenge_gate_passes": list(evaluation[2]),
            "fail_closed_correct_count": version_event_count,
            "fail_closed_denominator": version_event_count,
            "final_attempt_archive_root_sha256": _attempt_archive_root(
                final_attempt
            ).hex(),
            "final_attempt_index": final_attempt.attempt_index,
            "metric_bundle_root_sha256": _metric_bundle_root(
                final_attempt.metric_bundle
            ).hex(),
            "outcome": outcome.value,
            "overall_gate_passes": list(evaluation[0]),
            "scale_regret_bootstrap_upper": list(evaluation[4]),
            "scale_regret_point": list(evaluation[3]),
            "semantic_advantage_bootstrap_lower": list(evaluation[6]),
            "semantic_advantage_point": list(evaluation[5]),
            "slice_gate_passes": list(evaluation[1]),
            "validation_version_index": validation_version_index,
        },
    )


class _SignatureBackendUnavailable(Exception):
    pass


class _SignatureInvalid(Exception):
    pass


def _verify_ed25519_signature(
    raw_public_key: bytes,
    signature: bytes,
    message: bytes,
) -> None:
    if _Ed25519PublicKey is None:
        raise _SignatureBackendUnavailable
    try:
        _Ed25519PublicKey.from_public_bytes(raw_public_key).verify(signature, message)
    except (_InvalidSignature, TypeError, ValueError) as exc:
        raise _SignatureInvalid from exc


def _verify_canonical_original_bytes(item: OriginalPrerequisiteGraphV1) -> None:
    item.__post_init__()
    for value in (
        item.r06_original_candidate_bytes,
        item.r06_original_result_bytes,
        item.r07_original_candidate_bytes,
        item.r07_original_result_bytes,
    ):
        decoded = decode_and_require_canonical_rfc8785_v1(value)
        if type(decoded) is not dict or not decoded:
            raise ValueError("original prerequisite candidate/result must be a nonempty object")


def _verify_version_bump(
    candidate: StatisticalValidationCandidateGraphV1,
) -> None:
    if len(candidate.versions) == 1:
        if candidate.version_bump is not None:
            raise ValueError("single-version candidate has a version-bump record")
        return
    bump = candidate.version_bump
    if type(bump) is not StatisticalValidationVersionBumpCandidateV1:
        raise TypeError("two-version candidate lacks the closed version-bump record")
    bump.__post_init__()
    version_one_final = candidate.versions[0].attempts[-1]
    version_two_first = candidate.versions[1].attempts[0]
    if version_one_final.terminal_code is not ValidationAttemptTerminalCodeV1.COMPLETE_FAIL:
        raise ValueError("version two requires an immutable statistical COMPLETE_FAIL in v1")
    version_one_failure_root = _attempt_archive_root(version_one_final)
    version_one_attempt_chain_root = _version_attempt_chain_root(
        candidate.versions[0]
    )
    version_two_seed_roots = tuple(
        row.raw_seed_root for row in version_two_first.seed_roots
    )
    if (
        not hmac.compare_digest(
            bump.immutable_version_one_failure_root_sha256,
            version_one_failure_root,
        )
        or bump.version_one_failure_cas_receipt_id
        != version_one_final.cas_receipt.receipt_id
        or not hmac.compare_digest(
            bump.version_one_full_attempt_chain_root_sha256,
            version_one_attempt_chain_root,
        )
        or not hmac.compare_digest(
            bump.version_two_cohort_root_sha256,
            _cohort_root(version_two_first.cohort),
        )
        or any(
            not hmac.compare_digest(bound, actual)
            for bound, actual in zip(
                bump.version_two_seed_roots,
                version_two_seed_roots,
            )
        )
    ):
        raise ValueError(
            "version bump does not bind ordered v1 attempts/archive/CAS and "
            "v2 cohort/seeds"
        )
    if bump.version_bump_core_id != _version_bump_core_id(bump):
        raise ValueError("version-bump core content address differs")
    expected_report = canonicalize_rfc8785_v1(
        _public_failure_development_report_payload(
            bump,
            version_one_final=version_one_final,
            version_two_first=version_two_first,
        )
    )
    expected_issuance = canonicalize_rfc8785_v1(
        _version_two_issuance_authority_payload(
            bump,
            version_one_final=version_one_final,
            version_two_first=version_two_first,
        )
    )
    if (
        not hmac.compare_digest(
            bump.public_failure_development_report_bytes,
            expected_report,
        )
        or not hmac.compare_digest(
            bump.version_two_issuance_authority_bytes,
            expected_issuance,
        )
    ):
        raise ValueError("version bump report or issuance differs from the closed schema")
    if bump.version_bump_id != _version_bump_id(bump):
        raise ValueError("version-bump content address differs")
    cohort_digest_fields = (
        "primary_original_bytes_sha256",
        "primary_batch_root_sha256",
        "preservation_original_bytes_sha256",
        "preservation_batch_root_sha256",
        "generation_receipt_root_sha256",
    )
    version_one_digests = {
        getattr(version_one_final.cohort, name) for name in cohort_digest_fields
    }
    version_two_digests = {
        getattr(version_two_first.cohort, name) for name in cohort_digest_fields
    }
    if (
        len(version_one_digests) != len(cohort_digest_fields)
        or len(version_two_digests) != len(cohort_digest_fields)
        or version_one_digests & version_two_digests
    ):
        raise ValueError("version two cohort byte/root bindings are not fresh and disjoint")


def _candidate_json(item: StatisticalValidationCandidateGraphV1) -> dict[str, object]:
    def key_json(row: ValidationCandidatePublicKeyV1) -> dict[str, object]:
        return {
            "key_id": row.key_id,
            "purpose": row.purpose.value,
            "raw_public_key": row.raw_public_key.hex(),
        }

    def cas_json(row: ValidationCasReceiptCandidateV1) -> dict[str, object]:
        value = _cas_subject_json(row)
        value.update({"receipt_id": row.receipt_id, "signature": row.signature.hex()})
        return value

    def event_json(row: ValidationSignedEventCandidateV1) -> dict[str, object]:
        value = _event_subject_json(row)
        value.update({"event_id": row.event_id, "signature": row.signature.hex()})
        return value

    def attempt_json(row: StatisticalValidationAttemptCandidateV1) -> dict[str, object]:
        return {
            "attempt_index": row.attempt_index,
            "cas_receipt": cas_json(row.cas_receipt),
            "cohort": _cohort_json(row.cohort),
            "any_raw_prediction_byte_produced": row.any_raw_prediction_byte_produced,
            "execution_retry_action": row.execution_retry_action.value,
            "execution_retry_reason": row.execution_retry_reason.value,
            "events": [event_json(event) for event in row.events],
            "metric_bundle": (
                None if row.metric_bundle is None else _metric_bundle_json(row.metric_bundle)
            ),
            "observer_transcript_root_sha256": row.observer_transcript_root_sha256.hex(),
            "raw_output_byte_count": row.raw_output_byte_count,
            "seed_roots": [_seed_json(seed) for seed in row.seed_roots],
            "system_runs": [_system_run_json(run) for run in row.system_runs],
            "terminal_code": row.terminal_code.value,
            "validation_version_index": row.validation_version_index,
        }

    return {
        "candidate_keys": [key_json(key) for key in item.candidate_keys],
        "prerequisites": {
            "r06_original_candidate_bytes": item.prerequisites.r06_original_candidate_bytes.hex(),
            "r06_original_result_bytes": item.prerequisites.r06_original_result_bytes.hex(),
            "r07_original_candidate_bytes": item.prerequisites.r07_original_candidate_bytes.hex(),
            "r07_original_result_bytes": item.prerequisites.r07_original_result_bytes.hex(),
        },
        "profile_bytes": item.profile_bytes.hex(),
        "version_bump": (
            None
            if item.version_bump is None
            else {
                **_version_bump_subject_json(item.version_bump),
                "public_failure_development_report_bytes": (
                    item.version_bump.public_failure_development_report_bytes.hex()
                ),
                "version_bump_id": item.version_bump.version_bump_id,
                "version_two_issuance_authority_bytes": (
                    item.version_bump.version_two_issuance_authority_bytes.hex()
                ),
            }
        ),
        "versions": [
            {
                "attempts": [attempt_json(attempt) for attempt in version.attempts],
                "validation_version_index": version.validation_version_index,
            }
            for version in item.versions
        ],
    }


def _preflight_candidate_canonical_size(
    item: StatisticalValidationCandidateGraphV1,
) -> None:
    """Reject a conservative 20 MiB upper bound before canonical allocation."""

    binary_bytes = len(item.profile_bytes) + sum(
        len(value)
        for value in (
            item.prerequisites.r06_original_candidate_bytes,
            item.prerequisites.r06_original_result_bytes,
            item.prerequisites.r07_original_candidate_bytes,
            item.prerequisites.r07_original_result_bytes,
        )
    )
    text_bytes = sum(len(key.key_id) for key in item.candidate_keys)
    if item.version_bump is not None:
        binary_bytes += len(item.version_bump.public_failure_development_report_bytes)
        binary_bytes += len(item.version_bump.version_two_issuance_authority_bytes)
        text_bytes += sum(
            len(value)
            for value in (
                item.version_bump.old_protocol_version_id,
                item.version_bump.new_protocol_version_id,
                item.version_bump.old_selector_version_id,
                item.version_bump.new_selector_version_id,
                item.version_bump.old_validation_version_id,
                item.version_bump.new_validation_version_id,
                item.version_bump.version_one_failure_cas_receipt_id,
                item.version_bump.version_bump_core_id,
                item.version_bump.version_bump_id,
            )
        )
    for version in item.versions:
        for attempt in version.attempts:
            text_bytes += len(attempt.cohort.statistical_validation_profile_id)
            text_bytes += len(attempt.cas_receipt.object_key)
            text_bytes += len(attempt.cas_receipt.store_key_id)
            text_bytes += len(attempt.cas_receipt.receipt_id)
            text_bytes += sum(
                len(event.signer_key_id)
                + len(event.event_id)
                + (0 if event.previous_event_id is None else len(event.previous_event_id))
                for event in attempt.events
            )
            if attempt.metric_bundle is not None:
                text_bytes += sum(
                    len(row.input_row_id) for row in attempt.metric_bundle.main_rows
                )
                text_bytes += sum(
                    len(row.variant_row_id)
                    for row in attempt.metric_bundle.preservation_rows
                )
                text_bytes += sum(
                    len(row.input_row_id)
                    for row in attempt.metric_bundle.scale_regret_rows
                )
                text_bytes += sum(
                    len(row.input_row_id)
                    for row in attempt.metric_bundle.challenge_rows
                )
    conservative_upper = 2_097_152 + 2 * binary_bytes + 2 * text_bytes
    if conservative_upper > _MAX_SNAPSHOT_BYTES:
        raise MemoryError("candidate pre-canonical size exceeds the 20 MiB bound")


def _snapshot_bytes(item: StatisticalValidationCandidateGraphV1) -> bytes:
    _preflight_candidate_canonical_size(item)
    payload = canonicalize_rfc8785_v1(_candidate_json(item))
    if len(payload) > _MAX_SNAPSHOT_BYTES:
        raise MemoryError("candidate snapshot exceeds the bounded canonical size")
    return payload


def _validate_seed_roots(items: tuple[ValidationSeedRootCandidateV1, ...]) -> None:
    for item in items:
        item.__post_init__()
    if tuple(item.purpose for item in items) != _SEED_PURPOSES:
        raise ValueError("seed purpose order differs")
    if len({item.raw_seed_root for item in items}) != len(_SEED_PURPOSES):
        raise ValueError("seed roots are not distinct within the validation version")


def _validate_metric_bundle_records(
    item: StatisticalMetricBundleCandidateV1,
) -> None:
    item.__post_init__()
    for row in item.main_rows:
        row.__post_init__()
    for row in item.preservation_rows:
        row.__post_init__()
    for row in item.scale_regret_rows:
        row.__post_init__()
        row.regret.__post_init__()
    for row in item.challenge_rows:
        row.__post_init__()
    for row in item.overall_gates:
        row.__post_init__()
    for row in item.slice_gates:
        row.__post_init__()


def _validate_graph_records(item: StatisticalValidationCandidateGraphV1) -> None:
    """Freshly re-run every closed DTO invariant before snapshot allocation."""

    item.__post_init__()
    item.prerequisites.__post_init__()
    if item.version_bump is not None:
        item.version_bump.__post_init__()
    for key in item.candidate_keys:
        key.__post_init__()
    for version in item.versions:
        version.__post_init__()
        for attempt in version.attempts:
            attempt.__post_init__()
            attempt.cohort.__post_init__()
            _validate_seed_roots(attempt.seed_roots)
            for run in attempt.system_runs:
                run.__post_init__()
            if attempt.metric_bundle is not None:
                _validate_metric_bundle_records(attempt.metric_bundle)
            attempt.cas_receipt.__post_init__()
            for event in attempt.events:
                event.__post_init__()


def _validate_keys(
    items: tuple[ValidationCandidatePublicKeyV1, ...],
) -> dict[ValidationKeyPurposeV1, ValidationCandidatePublicKeyV1]:
    if tuple(item.purpose for item in items) != _KEY_PURPOSES:
        raise ValueError("candidate key purpose order differs")
    if len({item.raw_public_key for item in items}) != len(_KEY_PURPOSES):
        raise ValueError("candidate public keys are not distinct")
    if len({item.key_id for item in items}) != len(_KEY_PURPOSES):
        raise ValueError("candidate key ids are not distinct")
    for item in items:
        item.__post_init__()
    return {item.purpose: item for item in items}


def _verify_attempt(
    item: StatisticalValidationAttemptCandidateV1,
    *,
    key_by_purpose: dict[ValidationKeyPurposeV1, ValidationCandidatePublicKeyV1],
    expected_logical_time: int,
    previous_global_time: tuple[int, int] | None,
    total_event_count: int,
    after_signature: Callable[[], None],
) -> tuple[
    int,
    tuple[int, int],
    tuple[bool, ...],
    tuple[bool, ...],
    tuple[bool, bool],
    tuple[int, int] | None,
    tuple[int, int] | None,
    tuple[int, int] | None,
    tuple[int, int] | None,
    bool | None,
]:
    item.__post_init__()
    item.cohort.__post_init__()
    _validate_seed_roots(item.seed_roots)
    for run in item.system_runs:
        run.__post_init__()
    if item.metric_bundle is not None:
        _validate_metric_bundle_records(item.metric_bundle)
    item.cas_receipt.__post_init__()
    complete = item.terminal_code in {
        ValidationAttemptTerminalCodeV1.COMPLETE_PASS,
        ValidationAttemptTerminalCodeV1.COMPLETE_FAIL,
    }
    if complete:
        if (
            tuple(row.system for row in item.system_runs) != _SYSTEMS
            or item.metric_bundle is None
            or item.raw_output_byte_count <= 0
            or item.raw_output_byte_count
            != sum(row.raw_output_byte_count for row in item.system_runs)
        ):
            raise ValueError("complete attempt system/metric/raw-byte closure differs")
    else:
        if item.terminal_code not in _RETRYABLE_INFRA_CODES:
            raise ValueError("attempt terminal code is outside the closed set")
        if (
            item.system_runs
            or item.metric_bundle is not None
            or item.raw_output_byte_count != 0
            or item.any_raw_prediction_byte_produced
            or item.execution_retry_action
            is not ValidationExecutionRetryActionV1.REEXECUTE
        ):
            raise ValueError(
                "retryable infrastructure attempt must precede every prediction byte"
            )

    archive_root = _attempt_archive_root(item)
    if (
        item.cas_receipt.validation_version_index != item.validation_version_index
        or item.cas_receipt.attempt_index != item.attempt_index
        or not hmac.compare_digest(item.cas_receipt.value_digest_sha256, archive_root)
        or item.cas_receipt.receipt_id != _cas_receipt_id(item.cas_receipt)
    ):
        raise ValueError("CAS receipt does not bind the attempt archive")
    store_key = key_by_purpose[ValidationKeyPurposeV1.STORE]
    if item.cas_receipt.store_key_id != store_key.key_id:
        raise ValueError("CAS receipt store key differs")
    _verify_ed25519_signature(
        store_key.raw_public_key,
        item.cas_receipt.signature,
        _cas_signature_message(item.cas_receipt),
    )
    after_signature()

    expected_payloads = _expected_event_payloads(item)
    previous_event_id: str | None = None
    current_time = previous_global_time
    for sequence, (event, expected_kind, expected_purpose, expected_payload) in enumerate(
        zip(item.events, _EVENT_KINDS, _EVENT_PURPOSES, expected_payloads)
    ):
        event.__post_init__()
        if (
            event.validation_version_index != item.validation_version_index
            or event.attempt_index != item.attempt_index
            or event.sequence != sequence
            or event.event_kind is not expected_kind
            or event.previous_event_id != previous_event_id
            or event.logical_time != expected_logical_time
            or not hmac.compare_digest(event.payload_root_sha256, expected_payload)
            or event.event_id != _event_id(event)
        ):
            raise ValueError("signed lifecycle event chain differs")
        declared = (event.declared_unix_seconds, event.declared_nanosecond)
        if current_time is not None and declared <= current_time:
            raise ValueError("declared event time must be strictly increasing")
        key = key_by_purpose[expected_purpose]
        if event.signer_key_id != key.key_id:
            raise ValueError("event signer purpose differs")
        _verify_ed25519_signature(
            key.raw_public_key,
            event.signature,
            _event_signature_message(event),
        )
        after_signature()
        previous_event_id = event.event_id
        current_time = declared
        expected_logical_time += 1

    if current_time is None:
        raise ValueError("attempt produced no declared time")
    if item.metric_bundle is None:
        return (
            expected_logical_time,
            current_time,
            (),
            (),
            (),  # type: ignore[return-value]
            None,
            None,
            None,
            None,
            None,
        )
    evaluation = _evaluate_metric_bundle(
        item.metric_bundle,
        cohort_root_sha256=_cohort_root(item.cohort),
        total_event_count=total_event_count,
    )
    if (
        item.terminal_code is ValidationAttemptTerminalCodeV1.COMPLETE_PASS
    ) != evaluation[7]:
        raise ValueError("terminal COMPLETE_PASS/COMPLETE_FAIL differs from recomputed gates")
    return (
        expected_logical_time,
        current_time,
        evaluation[0],
        evaluation[1],
        evaluation[2],
        evaluation[3],
        evaluation[4],
        evaluation[5],
        evaluation[6],
        evaluation[7],
    )


def _rejection(
    reason: StatisticalValidationCandidateReasonV1,
    verified_signature_count: int = 0,
) -> StatisticalValidationCandidateRejectionV1:
    return StatisticalValidationCandidateRejectionV1(
        outcome=StatisticalValidationOutcomeV1.REJECTED,
        reason=reason,
        verified_signature_count=verified_signature_count,
    )


def verify_c1_statistical_validation_candidate_v1(
    candidate: object,
) -> StatisticalValidationCandidateMechanicsResultV1 | StatisticalValidationCandidateRejectionV1:
    """Fresh-verify a closed original candidate graph atomically in memory."""

    if type(candidate) is not StatisticalValidationCandidateGraphV1:
        return _rejection(StatisticalValidationCandidateReasonV1.WRONG_INPUT_TYPE)
    verified_signature_count = 0
    try:
        _validate_graph_records(candidate)
        _preflight_candidate_canonical_size(candidate)
        snapshot = _snapshot_bytes(candidate)
        try:
            detached = copy.deepcopy(candidate)
        except MemoryError:
            raise
        except Exception as exc:
            raise RuntimeError("INPUT_MUTATED") from exc
        _validate_graph_records(detached)

        def require_original_snapshot() -> None:
            try:
                _validate_graph_records(candidate)
                _preflight_candidate_canonical_size(candidate)
                current = _snapshot_bytes(candidate)
            except MemoryError:
                raise
            except Exception as exc:
                raise RuntimeError("INPUT_MUTATED") from exc
            if not hmac.compare_digest(current, snapshot):
                raise RuntimeError("INPUT_MUTATED")

        if not hmac.compare_digest(_snapshot_bytes(detached), snapshot):
            raise RuntimeError("INPUT_MUTATED")
        require_original_snapshot()
        profile = verify_c1_statistical_validation_profile_bytes_v1(detached.profile_bytes)
        _verify_canonical_original_bytes(detached.prerequisites)
        prerequisite_root = _prerequisite_root(detached.prerequisites)
        key_by_purpose = _validate_keys(detached.candidate_keys)
        for version in detached.versions:
            for attempt in version.attempts:
                if (
                    attempt.cohort.statistical_validation_profile_id != profile.profile_id
                    or not hmac.compare_digest(
                        attempt.cohort.prerequisite_graph_root_sha256,
                        prerequisite_root,
                    )
                ):
                    raise ValueError("cohort does not cross-bind profile and original prerequisite graph")
        _verify_version_bump(detached)

        total_event_count = sum(
            len(version.attempts) * _EVENTS_PER_ATTEMPT for version in detached.versions
        )
        expected_logical_time = 0
        previous_time: tuple[int, int] | None = None
        previous_version_final_pass: bool | None = None
        version_one_seed_roots: set[bytes] | None = None
        final_evaluation: tuple[
            tuple[bool, ...],
            tuple[bool, ...],
            tuple[bool, bool],
            tuple[int, int] | None,
            tuple[int, int] | None,
            tuple[int, int] | None,
            tuple[int, int] | None,
            bool | None,
        ] | None = None
        final_attempt_index = 0
        fail_closed_denominators_by_version: list[int] = []
        per_version_outcome_gate_root_vector: list[
            tuple[StatisticalValidationOutcomeV1, str]
        ] = []

        def after_signature() -> None:
            nonlocal verified_signature_count
            require_original_snapshot()
            verified_signature_count += 1

        for version in detached.versions:
            version.__post_init__()
            version_event_count = len(version.attempts) * _EVENTS_PER_ATTEMPT
            first = version.attempts[0]
            _validate_seed_roots(first.seed_roots)
            version_seed_roots = {row.raw_seed_root for row in first.seed_roots}
            if version.validation_version_index == 1:
                version_one_seed_roots = version_seed_roots
            else:
                if version_one_seed_roots is None or version_seed_roots & version_one_seed_roots:
                    raise ValueError("validation version two seed roots are not fresh and disjoint")
                if previous_version_final_pass is not False:
                    raise ValueError("validation version two requires version one COMPLETE_FAIL")

            if len(version.attempts) == 2:
                retry_source, retry = version.attempts
                if retry_source.terminal_code not in _RETRYABLE_INFRA_CODES:
                    raise ValueError("attempt one may follow only an exact retryable infrastructure code")
                if (
                    retry_source.raw_output_byte_count != 0
                    or retry_source.any_raw_prediction_byte_produced
                    or retry_source.execution_retry_action
                    is not ValidationExecutionRetryActionV1.REEXECUTE
                    or not hmac.compare_digest(
                        canonicalize_rfc8785_v1(_cohort_json(retry.cohort)),
                        canonicalize_rfc8785_v1(_cohort_json(retry_source.cohort)),
                    )
                    or not hmac.compare_digest(
                        canonicalize_rfc8785_v1(
                            [_seed_json(row) for row in retry.seed_roots]
                        ),
                        canonicalize_rfc8785_v1(
                            [_seed_json(row) for row in retry_source.seed_roots]
                        ),
                    )
                    or retry.terminal_code in _RETRYABLE_INFRA_CODES
                ):
                    raise ValueError("attempt one does not reuse exact cohort/seeds or is not terminal")

            version_terminal_evaluation: tuple[
                tuple[bool, ...],
                tuple[bool, ...],
                tuple[bool, bool],
                tuple[int, int],
                tuple[int, int],
                tuple[int, int],
                tuple[int, int],
                bool,
            ] | None = None
            for attempt in version.attempts:
                evaluation = _verify_attempt(
                    attempt,
                    key_by_purpose=key_by_purpose,
                    expected_logical_time=expected_logical_time,
                    previous_global_time=previous_time,
                    total_event_count=version_event_count,
                    after_signature=after_signature,
                )
                expected_logical_time = evaluation[0]
                previous_time = evaluation[1]
                final_evaluation = evaluation[2:]
                final_attempt_index = attempt.attempt_index
                if attempt.metric_bundle is not None:
                    version_terminal_evaluation = evaluation[2:]  # type: ignore[assignment]

            fail_closed_denominators_by_version.append(version_event_count)

            final_attempt = version.attempts[-1]
            if version_terminal_evaluation is None:
                raise ValueError("validation version lacks a terminal statistical evaluation")
            version_outcome = (
                StatisticalValidationOutcomeV1.COMPLETE_PASS
                if version_terminal_evaluation[7]
                else StatisticalValidationOutcomeV1.COMPLETE_FAIL
            )
            per_version_outcome_gate_root_vector.append(
                (
                    version_outcome,
                    _version_gate_evaluation_root(
                        validation_version_index=version.validation_version_index,
                        final_attempt=final_attempt,
                        version_event_count=version_event_count,
                        evaluation=version_terminal_evaluation,
                    ).hex(),
                )
            )
            previous_version_final_pass = (
                version_outcome is StatisticalValidationOutcomeV1.COMPLETE_PASS
            )
            if previous_version_final_pass and version is not detached.versions[-1]:
                raise ValueError("a passing validation version cannot be followed by another version")

        if final_evaluation is None:
            raise ValueError("candidate has no terminal evaluation")
        require_original_snapshot()
        outcome = (
            StatisticalValidationOutcomeV1.COMPLETE_PASS
            if previous_version_final_pass is True
            else StatisticalValidationOutcomeV1.COMPLETE_FAIL
        )
        (
            overall,
            slices,
            challenge,
            regret_point,
            regret_upper,
            semantic_point,
            semantic_lower,
            final_validation_version_all_statistical_gates_passed,
        ) = final_evaluation
        final_attempt = detached.versions[-1].attempts[-1]
        if type(final_attempt.metric_bundle) is not StatisticalMetricBundleCandidateV1:
            raise ValueError("terminal validation version lacks a metric bundle")
        result_values = _complete_result_values({
            "outcome": outcome,
            "reason": StatisticalValidationCandidateReasonV1.CANDIDATE_MECHANICS_VERIFIED,
            "version": PHASE2B_C1_STATISTICAL_VALIDATION_MECHANICS_V1_VERSION,
            "claim_level": PHASE2B_C1_STATISTICAL_VALIDATION_MECHANICS_V1_CLAIM_LEVEL,
            "profile_id": profile.profile_id,
            "input_snapshot_commitment_sha256": hashlib.sha256(
                _SNAPSHOT_DOMAIN + snapshot
            ).hexdigest(),
            "prerequisite_graph_root_sha256": prerequisite_root.hex(),
            "validation_version_count": len(detached.versions),
            "final_validation_version_index": detached.versions[-1].validation_version_index,
            "final_attempt_index": final_attempt_index,
            "version_bump_id": (
                None
                if detached.version_bump is None
                else detached.version_bump.version_bump_id
            ),
            "no_go_after_version_two_failure": (
                len(detached.versions) == 2
                and outcome is StatisticalValidationOutcomeV1.COMPLETE_FAIL
            ),
            "verified_signature_count": verified_signature_count,
            "total_event_count": total_event_count,
            "fail_closed_denominators_by_version": tuple(
                fail_closed_denominators_by_version
            ),
            "fail_closed_correct_by_version": tuple(
                fail_closed_denominators_by_version
            ),
            "per_version_outcome_gate_root_vector": tuple(
                per_version_outcome_gate_root_vector
            ),
            "aggregate_fail_closed_correct_count": total_event_count,
            "aggregate_fail_closed_denominator": total_event_count,
            "aggregate_fail_closed_passed": True,
            "evaluator_manifest_row_counts": (
                _MAIN_COUNT,
                _CHALLENGE_COUNT,
                _VARIANT_COUNT,
                240,
            ),
            "final_metric_bundle_root_sha256": _metric_bundle_root(
                final_attempt.metric_bundle
            ).hex(),
            "overall_gate_passes": overall,
            "slice_gate_passes": slices,
            "challenge_gate_passes": challenge,
            "scale_regret_point": regret_point,
            "scale_regret_bootstrap_upper": regret_upper,
            "semantic_advantage_point": semantic_point,
            "semantic_advantage_bootstrap_lower": semantic_lower,
            "final_validation_version_all_statistical_gates_passed": (
                final_validation_version_all_statistical_gates_passed
            ),
        })
        result_id = _content_id(
            _RESULT_PREFIX,
            _RESULT_DOMAIN,
            _result_projection_from_values(result_values),
        )
        return StatisticalValidationCandidateMechanicsResultV1(
            result_id=result_id,
            **result_values,
        )
    except StatisticalValidationProfileErrorV1:
        return _rejection(
            StatisticalValidationCandidateReasonV1.PROFILE_INVALID,
            verified_signature_count,
        )
    except RuntimeError as exc:
        if str(exc) == "INPUT_MUTATED":
            return _rejection(
                StatisticalValidationCandidateReasonV1.INPUT_MUTATED_DURING_VERIFICATION,
                verified_signature_count,
            )
        return _rejection(
            StatisticalValidationCandidateReasonV1.INTERNAL_ERROR,
            verified_signature_count,
        )
    except _SignatureBackendUnavailable:
        return _rejection(
            StatisticalValidationCandidateReasonV1.SIGNATURE_BACKEND_UNAVAILABLE,
            verified_signature_count,
        )
    except _SignatureInvalid:
        return _rejection(
            StatisticalValidationCandidateReasonV1.SIGNATURE_INVALID,
            verified_signature_count,
        )
    except MemoryError:
        return _rejection(
            StatisticalValidationCandidateReasonV1.RESOURCE_BUDGET_EXCEEDED,
            verified_signature_count,
        )
    except (AttributeError, KeyError, OverflowError, RFC8785Error, TypeError, ValueError):
        return _rejection(
            StatisticalValidationCandidateReasonV1.STRUCTURAL_CLOSURE_INVALID,
            verified_signature_count,
        )
    except Exception:
        return _rejection(
            StatisticalValidationCandidateReasonV1.INTERNAL_ERROR,
            verified_signature_count,
        )


def verify_c1_statistical_validation_candidate_result_v1(
    *,
    candidate: object,
    result: object,
) -> bool:
    """Fresh-reverify the original graph, then exact-compare a result."""

    if type(result) is not StatisticalValidationCandidateMechanicsResultV1:
        return False
    try:
        _validate_result_bounded_nonrecursive_shape_v1(result)
        result.__post_init__()
        result_before = canonicalize_rfc8785_v1(_result_projection_from_object(result))
        _validate_result_bounded_nonrecursive_shape_v1(result)
        result.__post_init__()
        result_after = canonicalize_rfc8785_v1(_result_projection_from_object(result))
        if not hmac.compare_digest(result_before, result_after):
            return False
        fresh = verify_c1_statistical_validation_candidate_v1(candidate)
        if type(fresh) is not StatisticalValidationCandidateMechanicsResultV1:
            return False
        _validate_result_bounded_nonrecursive_shape_v1(fresh)
        fresh.__post_init__()
        fresh_projection = canonicalize_rfc8785_v1(
            _result_projection_from_object(fresh)
        )
        _validate_result_bounded_nonrecursive_shape_v1(result)
        result.__post_init__()
        result_final = canonicalize_rfc8785_v1(
            _result_projection_from_object(result)
        )
        _validate_result_bounded_nonrecursive_shape_v1(result)
        result.__post_init__()
        result_final_after = canonicalize_rfc8785_v1(
            _result_projection_from_object(result)
        )
        if (
            not hmac.compare_digest(result_after, fresh_projection)
            or not hmac.compare_digest(result_after, result_final)
            or not hmac.compare_digest(result_final, result_final_after)
            or not hmac.compare_digest(
                object.__getattribute__(result, "result_id").encode("ascii"),
                object.__getattribute__(fresh, "result_id").encode("ascii"),
            )
        ):
            return False
    except (
        AttributeError,
        MemoryError,
        RecursionError,
        RFC8785Error,
        TypeError,
        UnicodeError,
        ValueError,
    ):
        return False
    return True


__all__ = [
    "PHASE2B_C1_STATISTICAL_VALIDATION_PROFILE_V1_SCHEMA",
    "PHASE2B_C1_STATISTICAL_VALIDATION_MECHANICS_V1_VERSION",
    "PHASE2B_C1_STATISTICAL_VALIDATION_MECHANICS_V1_CLAIM_LEVEL",
    "PHASE2B_C1_STATISTICAL_VALIDATION_V1_ED25519_BACKEND_AVAILABLE",
    "ValidationSeedPurposeV1",
    "ValidationSystemV1",
    "ValidationKeyPurposeV1",
    "ValidationEventKindV1",
    "ValidationAttemptTerminalCodeV1",
    "ValidationExecutionRetryActionV1",
    "ValidationExecutionRetryReasonV1",
    "ValidationUploadRetryActionV1",
    "ValidationUploadRetryReasonV1",
    "ValidationMainCaseTypeV1",
    "ValidationChallengeKindV1",
    "ValidationPreservationKindV1",
    "ValidationPredictionDecisionV1",
    "StatisticalValidationOutcomeV1",
    "StatisticalValidationCandidateReasonV1",
    "StatisticalValidationProfileVerificationV1",
    "StatisticalValidationProfileErrorV1",
    "OriginalPrerequisiteGraphV1",
    "ValidationSeedRootCandidateV1",
    "ValidationCandidatePublicKeyV1",
    "ValidationCohortBindingV1",
    "ValidationSystemRunCandidateV1",
    "BinaryGateCandidateV1",
    "ExactNonnegativeRationalV1",
    "MainEvaluationRowV1",
    "ScaleRegretEvaluationRowV1",
    "ChallengeEvaluationRowV1",
    "PreservationEvaluationRowV1",
    "StatisticalMetricBundleCandidateV1",
    "ValidationCasReceiptCandidateV1",
    "ValidationSignedEventCandidateV1",
    "StatisticalValidationAttemptCandidateV1",
    "StatisticalValidationVersionCandidateV1",
    "StatisticalValidationVersionBumpCandidateV1",
    "StatisticalValidationCandidateGraphV1",
    "StatisticalValidationCandidateMechanicsResultV1",
    "StatisticalValidationCandidateRejectionV1",
    "verify_c1_statistical_validation_profile_bytes_v1",
    "paired_cluster_bootstrap_indices_v1",
    "verify_c1_statistical_validation_candidate_v1",
    "verify_c1_statistical_validation_candidate_result_v1",
]
