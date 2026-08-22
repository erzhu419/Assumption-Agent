"""Executable schema catalog for a future authoritative C1 actual-960 run.

This module freezes the machine contract that a later raw evidence-bundle
verifier and the eighteen package verifiers must implement.  It does not
accept operational evidence, allocate campaign or attempt identifiers, sign
anything, touch the filesystem, or establish any R01--R16 gate.  Successful
catalog verification therefore means only ``RUNBOOK_FROZEN_NOT_ADMITTED``.
The separately frozen Package 0 raw ABI and private result contract remain an
exact ``NOT_IMPLEMENTED`` boundary: this module issues no Package 0 result and
does not turn presented ledger, custody, or signature bytes into R10 evidence.

The catalog deliberately replaces two unusable topology fragments in the
older gap inventory: an attempt is limited to index 0 and conditional index 1
(never index 2), and the only terminal stage is ``TERMINAL_CONSUMED``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Final, NoReturn

from .rfc8785_v1 import (
    canonicalize_rfc8785_v1,
)


SCHEMA_VERSION_V1: Final = (
    "hegel-machine-phase2b-actual-unsealed-960-authoritative-evidence-schema-catalog/1"
)
STATUS_V1: Final = "RUNBOOK_FROZEN_NOT_ADMITTED"
CLAIM_LEVEL_V1: Final = (
    "EXECUTABLE_SCHEMA_CATALOG_AND_RUNBOOK_MECHANICS_ONLY_NOT_EVIDENCE_"
    "NOT_AUTHORITY_NOT_ACTUAL_NOT_EXECUTION_NOT_C1"
)
CANONICALIZATION_V1: Final = "RFC8785_EXACT_BYTES_NO_TRAILING_LF"
REAL_BUNDLE_VERIFICATION_STATUS_V1: Final = "NOT_IMPLEMENTED"
REAL_PACKAGE0_VERIFICATION_STATUS_V1: Final = "NOT_IMPLEMENTED"
PACKAGE_OUTER_STATUS_V1: Final = "PRESENTED_UNVERIFIED"
PACKAGE_OUTER_CLAIM_LEVEL_V1: Final = (
    "PRESENTED_EXTERNAL_EVIDENCE_ONLY_NOT_VERIFIED_NOT_GATE_ADMISSION_"
    "NOT_AUTHORITY_NOT_CURRENTNESS_NOT_DURABILITY_NOT_C1"
)
SIGNATURE_ALGORITHM_V1: Final = "Ed25519"
MAXIMUM_CATALOG_PAYLOAD_BYTES_V1: Final = 2_097_152
CONTENT_ID_PREIMAGE_PROFILE_V1: Final = (
    "BODY_ONLY_DOMAIN_BYTES_PLUS_RFC8785_CANONICAL_BODY_SHA256_V1"
)
CONTENT_ID_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1: Final = ("body",)
CONTENT_ID_FORMULA_V1: Final = (
    "content_id_prefix || lowerhex(sha256(content_id_domain_ascii_bytes || "
    "rfc8785(body)))"
)
DETACHED_SIGNATURE_PREIMAGE_PROFILE_V1: Final = (
    "DOMAIN_BYTES_PLUS_RFC8785_EXACT_DETACHED_SIGNATURE_PREIMAGE_V1"
)
DETACHED_SIGNATURE_PREIMAGE_SCHEMA_VERSION_V1: Final = (
    "hegel-machine-phase2b-actual-unsealed-960-authoritative-evidence-"
    "detached-signature-preimage/1"
)
DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_V1: Final = (
    "hegel-machine-phase2b-actual-unsealed-960-detached-signature/1"
)
DETACHED_SIGNATURE_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1: Final = (
    "schema_version",
    "artifact_kind",
    "artifact_schema_version",
    "content_id",
    "body",
    "signature_ordinal",
    "role",
    "purpose",
    "key_epoch",
    "signer_key_id",
)
DETACHED_SIGNATURE_FORMULA_V1: Final = (
    "Ed25519.sign(private_key, signature_domain_ascii_bytes || "
    "rfc8785(detached_signature_preimage_object))"
)
DETACHED_SIGNATURE_EXCLUDED_FIELDS_IN_EXACT_ORDER_V1: Final = (
    "signature_hex",
    "signatures_in_exact_order",
)

INSPECTED_READINESS_DAG_CONFIG_SHA256_V1: Final = (
    "dd5c00785fccaf36bc307e21689d12f5049489971b3a47c9f253936cdb6037a0"
)
INSPECTED_PROGRESSION_PLAN_SHA256_V1: Final = (
    "8d5a88a52aae3236fe57e5d1599eb6508156f9ce0d2a135c7355a0263d15843b"
)
INSPECTED_GAP_INVENTORY_SOURCE_SHA256_V1: Final = (
    "b1dc11a0ac6e3dfafc9621c10957308b9d50c30e8806e78ff5b6a056a8ff1450"
)
INSPECTED_STATISTICAL_VALIDATION_SOURCE_SHA256_V1: Final = (
    "9bd4a7d26d9973a2b4dcb61ffa283aab93bcc6b7758b38a69ea484cccdc26339"
)
INSPECTED_READINESS_DAG_ID_V1: Final = (
    "phase2b_actual_unsealed_960_readiness_dag_v1_"
    "775b86cb499db3a991686634335ec444f7e44c7caea92b1c2b948036b585ea84"
)

INSPECTED_DEPENDENCIES_IN_EXACT_ORDER_V1: Final = (
    (
        "readiness_dag_config",
        "config/phase2b_actual_unsealed_960_readiness_dag_v1.json",
        INSPECTED_READINESS_DAG_CONFIG_SHA256_V1,
        INSPECTED_READINESS_DAG_ID_V1,
        "INSPECTED_PROVENANCE_ONLY_NOT_CURRENTNESS_NOT_AUTHORITY",
    ),
    (
        "c1_c4_progression_plan",
        "docs/Hegel_Machine_C1_C4_Authoritative_Progression_Plan_v1.md",
        INSPECTED_PROGRESSION_PLAN_SHA256_V1,
        "NONE",
        "INSPECTED_PROVENANCE_ONLY_NOT_EXECUTION_PLAN_ADMISSION",
    ),
    (
        "legacy_gap_inventory_source",
        "src/hegel_machine/phase2b_actual_unsealed_960_replay_admission_evidence_contract_v2.py",
        INSPECTED_GAP_INVENTORY_SOURCE_SHA256_V1,
        "NONE",
        "NAMES_AND_PURPOSES_ONLY_LEGACY_ATTEMPT_AND_TERMINAL_TOPOLOGY_NOT_INHERITED",
    ),
    (
        "statistical_validation_source",
        "src/hegel_machine/phase2b_c1_statistical_validation_v1.py",
        INSPECTED_STATISTICAL_VALIDATION_SOURCE_SHA256_V1,
        "NONE",
        "CANDIDATE_MECHANICS_ONLY_NOT_R08",
    ),
)


class CatalogDispositionV1(str, Enum):
    RUNBOOK_FROZEN_NOT_ADMITTED = "RUNBOOK_FROZEN_NOT_ADMITTED"
    REJECTED = "REJECTED"


class CatalogReasonV1(str, Enum):
    EXACT_EXECUTABLE_SCHEMA_CATALOG_FROZEN = (
        "EXACT_EXECUTABLE_SCHEMA_CATALOG_FROZEN"
    )
    WRONG_INPUT_TYPE = "WRONG_INPUT_TYPE"
    NONCANONICAL_INPUT = "NONCANONICAL_INPUT"
    SCHEMA_INVALID = "SCHEMA_INVALID"
    PIN_MISMATCH = "PIN_MISMATCH"
    CONTENT_ID_MISMATCH = "CONTENT_ID_MISMATCH"
    DAG_INVALID = "DAG_INVALID"
    CLAIM_BOUNDARY_INVALID = "CLAIM_BOUNDARY_INVALID"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class Package0DispositionV1(str, Enum):
    VERIFIED_PRESENTED_EXTERNAL_EVIDENCE_NOT_GATE_ADMISSION = (
        "VERIFIED_PRESENTED_EXTERNAL_EVIDENCE_NOT_GATE_ADMISSION"
    )
    REJECTED = "REJECTED"


class Package0ReasonV1(str, Enum):
    EXACT_PACKAGE0_RAW_CONTRACT_VERIFIED = (
        "EXACT_PACKAGE0_RAW_CONTRACT_VERIFIED"
    )
    WRONG_INPUT_TYPE = "WRONG_INPUT_TYPE"
    NONCANONICAL_INPUT = "NONCANONICAL_INPUT"
    PIN_MISMATCH = "PIN_MISMATCH"
    SCHEMA_INVALID = "SCHEMA_INVALID"
    CONTENT_ID_MISMATCH = "CONTENT_ID_MISMATCH"
    SIGNATURE_INVALID = "SIGNATURE_INVALID"
    LINEAGE_INVALID = "LINEAGE_INVALID"
    TERMINAL_BINDING_INVALID = "TERMINAL_BINDING_INVALID"
    CLAIM_BOUNDARY_INVALID = "CLAIM_BOUNDARY_INVALID"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AttemptDispositionV1(str, Enum):
    COMPLETE_PASS = "COMPLETE_PASS"
    COMPLETE_FAIL = "COMPLETE_FAIL"
    REJECTED = "REJECTED"
    INVALIDATED = "INVALIDATED"
    NOT_RUN = "NOT_RUN"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class _GateSpecV1:
    ordinal: int
    gate_id: str
    dependencies_in_exact_order: tuple[str, ...]
    evidence_requirement: str
    initial_satisfied: bool


@dataclass(frozen=True, slots=True)
class _GateSourceSpecV1:
    ordinal: int
    gate_id: str
    allowed_native_source_kind_orders_in_exact_order: tuple[
        tuple[str, ...], ...
    ]
    verification_availability: str
    verifier_dispatch_id: str
    verifier_source_sha256_or_none: str | None


@dataclass(frozen=True, slots=True)
class _StageSpecV1:
    ordinal: int
    stage: str
    predecessor_nodes_in_exact_order: tuple[str, ...]
    producer_role: str
    consumer_roles_in_exact_order: tuple[str, ...]
    occurs_only_after_r16: bool
    terminal: bool


@dataclass(frozen=True, slots=True)
class _SignatureSlotSpecV1:
    ordinal: int
    role: str
    purpose: str
    key_epoch: int
    signature_domain_ascii: str


@dataclass(frozen=True, slots=True)
class _PackageSpecV1:
    ordinal: int
    package_name: str
    purpose: str
    schema_version: str
    envelope_fields_in_exact_order: tuple[str, ...]
    body_fields_in_exact_order: tuple[str, ...]
    nested_field_manifests_in_exact_order: tuple[tuple[str, tuple[str, ...]], ...]
    content_id_domain_ascii: str
    content_id_prefix: str
    signature_slots_in_exact_order: tuple[_SignatureSlotSpecV1, ...]
    minimum_payload_bytes: int
    maximum_payload_bytes: int
    minimum_item_count: int
    maximum_item_count: int
    predecessor_package_ordinals_in_exact_order: tuple[int, ...]
    required_external_evidence_class: str
    catalog_evidence_disposition: str
    producer_evidence_disposition: str
    verifier_success_disposition: str
    verifier_failure_dispositions_in_exact_order: tuple[str, ...]
    frozen_threshold_ids_in_exact_order: tuple[str, ...]
    required_gate_ids_in_exact_order: tuple[str, ...]
    producer_stage: str
    finalization_stage: str
    consumer_stages_in_exact_order: tuple[str, ...]
    terminal_behavior: str
    negative_vector_families_in_exact_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ReferencedArtifactSpecV1:
    artifact_kind: str
    schema_version: str
    body_fields_in_exact_order: tuple[str, ...]
    content_id_domain_ascii: str
    content_id_prefix: str
    signature_slots_in_exact_order: tuple[_SignatureSlotSpecV1, ...]


_GATE_ROWS_V1: Final = (
    ("R01_AUTHORIZATION_VERIFIED", (), "EXACT_OWNER_AUTHORIZATION_CONTENT_ID_ACTIVE_AND_NOT_REVOKED"),
    ("R02_EXECUTION_BRANCH_BINDINGS_REPLAYED", ("R01_AUTHORIZATION_VERIFIED",), "SELECTED_EXECUTION_BRANCH_REPLAYS_PROTOCOL_EXACT_FREEZE_AND_AUTHORIZATION_BINDINGS"),
    ("R03_FORMAL_TRUSTED_WIRE_IMPLEMENTATION_QUALIFIED", ("R02_EXECUTION_BRANCH_BINDINGS_REPLAYED",), "FORMAL_RFC8785_TRUSTED_WIRE_BUILDER_IMPLEMENTATION_QUALIFICATION"),
    ("R04_FORMAL_COVERT_AUDITOR_IMPLEMENTATION_QUALIFIED", ("R03_FORMAL_TRUSTED_WIRE_IMPLEMENTATION_QUALIFIED",), "NAMESPACE_AWARE_COVERT_AUDITOR_IMPLEMENTATION_QUALIFICATION"),
    ("R05_SYNTHETIC_DEV_WIRE_COVERT_QUALIFICATION_VERIFIED", ("R04_FORMAL_COVERT_AUDITOR_IMPLEMENTATION_QUALIFIED",), "SYNTHETIC_DEV_ONLY_INTEGRATED_WIRE_COVERT_QUALIFICATION_RECEIPT"),
    ("R06_GENERATOR_IMPLEMENTATION_QUALIFIED", ("R05_SYNTHETIC_DEV_WIRE_COVERT_QUALIFICATION_VERIFIED",), "INDEPENDENT_HOLDOUT_GENERATOR_IMPLEMENTED_AND_QUALIFIED_WITHOUT_ACTUAL_COHORT_GENERATION"),
    ("R07_BASELINE_REGISTRATIONS_FROZEN", ("R05_SYNTHETIC_DEV_WIRE_COVERT_QUALIFICATION_VERIFIED",), "ALL_THREE_EXACT_BASELINE_REVISIONS_AND_ARTIFACTS_REGISTERED_BEFORE_GENERATION"),
    ("R08_STATISTICAL_VALIDATION_PASSED", ("R06_GENERATOR_IMPLEMENTATION_QUALIFIED", "R07_BASELINE_REGISTRATIONS_FROZEN"), "PERMITTED_STATISTICAL_VALIDATION_VERSION_PASSED"),
    ("R09_SIX_ROLE_LIVE_DOCKER_QUALIFIED", ("R08_STATISTICAL_VALIDATION_PASSED",), "SIX_ROLE_LIVE_DOCKER_QUALIFICATION_VERIFIED_UNDER_DISCLOSED_THREAT_MODEL"),
    ("R10_DURABLE_CUSTODY_AND_LEDGER_VERIFIED", ("R09_SIX_ROLE_LIVE_DOCKER_QUALIFIED",), "DURABLE_CUSTODY_LEDGER_CAS_ATTEMPT_DENOMINATOR_AND_CRASH_RETRY_BEHAVIOR_VERIFIED"),
    ("R11_EXECUTION_FREEZE_COMMITMENTS_BOUND", ("R10_DURABLE_CUSTODY_AND_LEDGER_VERIFIED",), "EXECUTION_CONFIGURATION_FREEZE_AND_PREGENERATION_COMMITMENTS_CROSS_BOUND"),
    ("R12_INDEPENDENT_LAUNCH_OVERLAY_VERIFIED", ("R11_EXECUTION_FREEZE_COMMITMENTS_BOUND",), "INDEPENDENT_LAUNCH_OVERLAY_AUTHORIZED_AND_VERIFIED_WITHOUT_ALLOCATING_IDS"),
    ("R13_FRESH_IDS_ALLOCATED_BY_VERIFIED_OVERLAY", ("R12_INDEPENDENT_LAUNCH_OVERLAY_VERIFIED",), "VERIFIED_LAUNCH_OVERLAY_ALLOCATES_FRESH_CAMPAIGN_AND_ATTEMPT_IDS"),
    ("R14_ACTUAL_UNSEALED_960_COHORT_GENERATED", ("R13_FRESH_IDS_ALLOCATED_BY_VERIFIED_OVERLAY",), "ACTUAL_UNSEALED_960_COHORT_GENERATED_AND_PUBLIC_PAYLOAD_COMMITMENTS_BOUND"),
    ("R15_ACTUAL_CORPUS_COVERT_AUDIT_PASSED", ("R14_ACTUAL_UNSEALED_960_COHORT_GENERATED",), "ACTUAL_CORPUS_NAMESPACE_METADATA_CAPACITY_AND_ATOMIC_COVERT_AUDIT_PASSED"),
    ("R16_INFERENCE_ADMISSION_UNLOCKED", ("R15_ACTUAL_CORPUS_COVERT_AUDIT_PASSED",), "RECOGNIZER_AND_THREE_BASELINE_INFERENCE_ADMISSION_ONLY_AFTER_ACTUAL_CORPUS_AUDIT"),
)
GATE_SPECS_IN_EXACT_ORDER_V1: Final = tuple(
    _GateSpecV1(index + 1, gate_id, dependencies, requirement, False)
    for index, (gate_id, dependencies, requirement) in enumerate(_GATE_ROWS_V1)
)

GATE_VERIFICATION_AVAILABILITIES_IN_EXACT_ORDER_V1: Final = (
    "SCHEMA_FROZEN_NOT_VERIFIABLE",
    "EXECUTABLE_PRODUCER_FREE",
)
CURRENT_GATE_VERIFICATION_AVAILABILITY_V1: Final = (
    "SCHEMA_FROZEN_NOT_VERIFIABLE"
)
GATE_VERIFIER_NOT_IMPLEMENTED_FAILURE_CODE_V1: Final = (
    "GATE_VERIFIER_NOT_IMPLEMENTED"
)

_GATE_NATIVE_SOURCE_KIND_ORDER_VARIANTS_V1: Final = (
    (
        (
            "SIGNED_OWNER_AUTHORIZATION",
            "STAGE_B_TRUST_LOCK",
            "AUTHORITY_PROFILE",
            "SIGNER_REGISTRY",
            "OWNER_STATUS_LEDGER_BUNDLE",
            "CURRENT_CUT_RESPONSE",
            "CURRENT_CUT_STATEMENT",
            "CURRENT_CUT_LOGICAL_TIME_CHAIN",
            "EXTERNAL_WITNESS_POLICY",
            "SIGNED_ANTI_FORK_ANCHOR",
            "EXTERNAL_ANCHOR_RECEIPT",
            "LIVE_RUNTIME_CLOSURE_REPORT_0",
            "LIVE_RUNTIME_CLOSURE_REPORT_1",
            "LIVE_RUNTIME_CLOSURE_REPORT_2",
            "LIVE_RUNTIME_CLOSURE_REPORT_3",
            "LIVE_RUNTIME_CLOSURE_REPORT_4",
            "LIVE_RUNTIME_CLOSURE_BUNDLE",
            "SERVICE_KEY_CUSTODY_ATTESTATION_0",
            "SERVICE_KEY_CUSTODY_ATTESTATION_1",
            "SERVICE_KEY_CUSTODY_ATTESTATION_2",
            "SERVICE_KEY_CUSTODY_ATTESTATION_3",
            "SERVICE_KEY_CUSTODY_ATTESTATION_4",
            "CAS_COMPLETE_SESSION",
            "DUAL_SIGNED_R01_RECEIPT",
        ),
    ),
    (
        (
            "EXECUTION_BRANCH_EVENT",
            "EXECUTION_BINDING_MANIFEST",
            "BRANCH_REPLAY_TRANSCRIPT",
            "DUAL_SIGNED_R02_RECEIPT",
        ),
    ),
    (
        (
            "WIRE_SOURCE_RUNTIME_CLOSURE_BUNDLE",
            "RFC8785_KAT_ARCHIVE",
            "TEN_NAMESPACE_WIRE_FIXTURE_ARCHIVE",
            "ENVELOPE_CAPACITY_KAT_ARCHIVE",
            "SYNTHETIC_WIRE_OUTPUT_ARCHIVE",
            "CAS_LOGICAL_TIME_BINDING_BUNDLE",
            "PURE_WIRE_REPLAY_REPORT",
            "R03_SIGNED_QUALIFICATION_RECEIPT",
        ),
    ),
    (
        (
            "AUDITOR_SOURCE_RUNTIME_CLOSURE_BUNDLE",
            "FULL_RESULT_MATRIX_ARCHIVE",
            "ORIGINAL_WORKER_ARTIFACT_0",
            "ORIGINAL_WORKER_ARTIFACT_1",
            "ORIGINAL_WORKER_ARTIFACT_2",
            "ORIGINAL_WORKER_ARTIFACT_3",
            "ORIGINAL_WORKER_ARTIFACT_4",
            "ORIGINAL_WORKER_ARTIFACT_5",
            "ORIGINAL_WORKER_ARTIFACT_6",
            "ORIGINAL_WORKER_ARTIFACT_7",
            "ORIGINAL_WORKER_ARTIFACT_8",
            "ORIGINAL_WORKER_ARTIFACT_9",
            "ORIGINAL_WORKER_ARTIFACT_10",
            "ORIGINAL_WORKER_ARTIFACT_11",
            "VARIANT_SCHEDULE_MANIFEST",
            "AUDITOR_STATISTICAL_REPLAY_REPORT",
            "CAS_LOGICAL_TIME_BINDING_BUNDLE",
            "PURE_AUDITOR_REPLAY_REPORT",
            "R04_SIGNED_QUALIFICATION_RECEIPT",
        ),
    ),
    (
        (
            "INTEGRATED_FIXTURE_ROOT_BINDING",
            "INTEGRATION_RERUN_ARCHIVE",
            "CAS_LOGICAL_TIME_BINDING_BUNDLE",
            "PURE_INTEGRATION_REPLAY_REPORT",
            "R05_SIGNED_QUALIFICATION_RECEIPT",
        ),
    ),
    (
        (
            "GENERATOR_SOURCE_RUNTIME_BUNDLE",
            "GENERATOR_SYNTHETIC_FIXTURE_ARCHIVE_0",
            "GENERATOR_SYNTHETIC_FIXTURE_ARCHIVE_1",
            "GENERATOR_SYNTHETIC_FIXTURE_ARCHIVE_2",
            "GENERATOR_SYNTHETIC_FIXTURE_ARCHIVE_3",
            "FRESH_PROCESS_REPLAY_REPORT_0",
            "FRESH_PROCESS_REPLAY_REPORT_1",
            "RESOURCE_OBSERVER_ATTESTATION",
            "GENERATOR_QUALIFIER_REGISTRY_SNAPSHOT",
            "R06_SIGNED_ADMISSION_RECEIPT",
        ),
    ),
    (
        (
            "BASELINE_SOURCE_ARTIFACT_ARCHIVE_0",
            "BASELINE_SOURCE_ARTIFACT_ARCHIVE_1",
            "BASELINE_SOURCE_ARTIFACT_ARCHIVE_2",
            "FINAL_BASELINE_REGISTRATION_BUNDLE_0",
            "FINAL_BASELINE_REGISTRATION_BUNDLE_1",
            "FINAL_BASELINE_REGISTRATION_BUNDLE_2",
            "THREE_BASELINE_REGISTRY_SNAPSHOT",
            "BASELINE_ARTIFACT_ROOT_REPLAY_REPORT",
            "R07_SIGNED_REGISTRATION_RECEIPT",
        ),
    ),
    (
        (
            "R06_ORIGINAL_CANDIDATE",
            "R06_ORIGINAL_RESULT",
            "R07_ORIGINAL_CANDIDATE",
            "R07_ORIGINAL_RESULT",
            "FINAL_STATISTICAL_VALIDATION_POLICY",
            "VALIDATION_VERSION_ARCHIVE_0",
            "R08_SIGNED_VALIDATION_RECEIPT",
        ),
        (
            "R06_ORIGINAL_CANDIDATE",
            "R06_ORIGINAL_RESULT",
            "R07_ORIGINAL_CANDIDATE",
            "R07_ORIGINAL_RESULT",
            "FINAL_STATISTICAL_VALIDATION_POLICY",
            "VALIDATION_VERSION_ARCHIVE_0",
            "VALIDATION_VERSION_ARCHIVE_1",
            "VERSION_BUMP_AUTHORITY",
            "R08_SIGNED_VALIDATION_RECEIPT",
        ),
    ),
    (
        (
            "C1_CONTAINER_ACTOR_PROFILE",
            "SIX_ROLE_SIGNER_REGISTRY",
            "ROLE_LIVE_RUNTIME_CLOSURE_ATTESTATION_0",
            "ROLE_LIVE_RUNTIME_CLOSURE_ATTESTATION_1",
            "ROLE_LIVE_RUNTIME_CLOSURE_ATTESTATION_2",
            "ROLE_LIVE_RUNTIME_CLOSURE_ATTESTATION_3",
            "ROLE_LIVE_RUNTIME_CLOSURE_ATTESTATION_4",
            "ROLE_LIVE_RUNTIME_CLOSURE_ATTESTATION_5",
            "CROSS_ROLE_ISOLATION_AND_THREAT_MODEL_REPORT",
            "STORE_TIME_GENESIS_DURABILITY_BUNDLE",
            "R09_SIGNED_LIVE_DOCKER_RECEIPT",
        ),
    ),
    (
        (
            "ROLE_DURABLE_CUSTODY_ATTESTATION_0",
            "ROLE_DURABLE_CUSTODY_ATTESTATION_1",
            "ROLE_DURABLE_CUSTODY_ATTESTATION_2",
            "ROLE_DURABLE_CUSTODY_ATTESTATION_3",
            "ROLE_DURABLE_CUSTODY_ATTESTATION_4",
            "ROLE_DURABLE_CUSTODY_ATTESTATION_5",
            "CAS_COMPLETE_SESSION",
            "SIGNED_ATTEMPT_LEDGER",
            "ATTEMPT_DENOMINATOR_REPORT",
            "CRASH_RETRY_FAULT_INJECTION_REPORT",
            "ROLLBACK_ANTI_FORK_REPORT",
            "PURE_DURABILITY_REPLAY_REPORT",
            "R10_SIGNED_QUALIFICATION_RECEIPT",
        ),
    ),
    (
        (
            "EXECUTION_CONFIGURATION_MANIFEST",
            "PACKAGE_MANIFEST_VECTOR",
            "FIVE_SECRET_COMMITMENT_BUNDLE",
            "INDEPENDENT_SECRET_SOURCE_ATTESTATION_0",
            "INDEPENDENT_SECRET_SOURCE_ATTESTATION_1",
            "INDEPENDENT_SECRET_SOURCE_ATTESTATION_2",
            "INDEPENDENT_SECRET_SOURCE_ATTESTATION_3",
            "INDEPENDENT_SECRET_SOURCE_ATTESTATION_4",
            "SECRET_CUSTODY_RECEIPT_0",
            "SECRET_CUSTODY_RECEIPT_1",
            "SECRET_CUSTODY_RECEIPT_2",
            "SECRET_CUSTODY_RECEIPT_3",
            "SECRET_CUSTODY_RECEIPT_4",
            "FREEZE_CROSS_BIND_RECEIPT",
            "R11_LOGICAL_TIME_TOKEN",
        ),
    ),
    (
        (
            "LATEST_OWNER_STATUS_CURRENT_CUT",
            "PREALLOCATION_LEDGER_HEAD_AND_EXTERNAL_ANCHOR",
            "LAUNCH_PREFLIGHT_REPLAY_REPORT",
            "R12_SIGNED_LAUNCH_ADMISSION_RECEIPT",
        ),
    ),
    (
        (
            "PREALLOCATION_LEDGER_HEAD",
            "FRESH_ID_ALLOCATION_REQUEST",
            "ID_ENTROPY_AND_COLLISION_CHECK_REPORT",
            "ATOMIC_ID_ALLOCATION_RECEIPT",
            "POSTALLOCATION_LEDGER_HEAD",
            "R13_LOGICAL_TIME_TOKEN",
        ),
    ),
    (
        (
            "GENERATOR_SOURCE_RUNTIME_BUNDLE",
            "R11_COMMITMENT_EVIDENCE_BUNDLE",
            "PRIVATE_SOURCE_MANIFEST",
            "ANSWER_MANIFEST",
            "SEED_MANIFEST",
            "PUBLIC_PRIMARY_WIRE_ARCHIVE",
            "PUBLIC_VARIANT_WIRE_ARCHIVE",
            "PRESERVATION_MANIFEST",
            "DISJOINTNESS_MANIFEST",
            "SCALE_REGRET_ORACLE_MANIFEST",
            "DETERMINISTIC_GENERATION_REPLAY_REPORT",
            "R14_PURPOSE_KEY_REGISTRY_AND_REVOCATION_SNAPSHOT",
            "ATOMIC_SEVEN_KEY_STORE_TRANSACTION_RECEIPT",
            "R11_R14_LOGICAL_TIME_CHAIN",
            "R14_SIGNED_GENERATION_RECEIPT",
        ),
    ),
    (
        (
            "PUBLIC_PRIMARY_WIRE_ARCHIVE",
            "PUBLIC_VARIANT_WIRE_ARCHIVE",
            "ACTUAL_AUDITOR_SOURCE_RUNTIME_BUNDLE",
            "ACTUAL_NAMESPACE_AUDIT_REPORT",
            "ACTUAL_METADATA_PREDICTIVITY_RESULT_ARCHIVE",
            "ACTUAL_CAPACITY_SUPERVISOR_REPORT",
            "FORMAL_UUID_RECEIPT",
            "FORMAL_COVERT_AUDIT_RECEIPT",
            "AUDIT_CAS_LOGICAL_TIME_BUNDLE",
            "PURE_ACTUAL_AUDIT_REPLAY_REPORT",
            "R15_ATOMIC_AUDIT_RECEIPT",
        ),
    ),
    (
        (
            "RECOGNIZER_EXECUTION_PACKAGE_MANIFEST",
            "FINAL_BASELINE_REGISTRATION_BUNDLE_0",
            "FINAL_BASELINE_REGISTRATION_BUNDLE_1",
            "FINAL_BASELINE_REGISTRATION_BUNDLE_2",
            "ACTUAL_1532_INPUT_ARCHIVE_COMMITMENT",
            "FOUR_SYSTEM_OUTPUT_ARCHIVE_POLICY",
            "EXECUTION_FREEZE_MANIFEST",
            "PRE_REVEAL_ZERO_OUTPUT_STATE_SNAPSHOT",
            "RUNNER_PREFLIGHT_ATTESTATION_0",
            "RUNNER_PREFLIGHT_ATTESTATION_1",
            "RUNNER_PREFLIGHT_ATTESTATION_2",
            "RUNNER_PREFLIGHT_ATTESTATION_3",
            "R16_SIGNED_ADMISSION_RECEIPT",
        ),
    ),
)

_GATE_VERIFIER_DISPATCH_IDS_IN_EXACT_ORDER_V1: Final = (
    "VERIFY_R01_REAL_CURRENT_CUT_ACTIVE_NONREVOKED_V1",
    "VERIFY_R02_EXACT_BRANCH_REPLAY_V1",
    "VERIFY_R03_WIRE_IMPLEMENTATION_QUALIFICATION_V1",
    "VERIFY_R04_AUDITOR_IMPLEMENTATION_QUALIFICATION_V1",
    "VERIFY_R05_SYNTHETIC_INTEGRATION_QUALIFICATION_V1",
    "VERIFY_R06_GENERATOR_IMPLEMENTATION_QUALIFICATION_V1",
    "VERIFY_R07_THREE_REGISTRATIONS_FROZEN_V1",
    "VERIFY_R08_STATISTICAL_VALIDATION_V1",
    "VERIFY_R09_SIX_ROLE_LIVE_DOCKER_V1",
    "VERIFY_R10_DURABLE_CUSTODY_LEDGER_V1",
    "VERIFY_R11_EXECUTION_FREEZE_COMMITMENTS_V1",
    "VERIFY_R12_INDEPENDENT_OVERLAY_NO_IDS_V1",
    "VERIFY_R13_FRESH_ID_ALLOCATION_TRANSITION_V1",
    "VERIFY_R14_ACTUAL_960_GENERATION_V1",
    "VERIFY_R15_ACTUAL_CORPUS_COVERT_AUDIT_V1",
    "VERIFY_R16_INFERENCE_ADMISSION_UNLOCK_V1",
)

GATE_SOURCE_SPECS_IN_EXACT_ORDER_V1: Final = tuple(
    _GateSourceSpecV1(
        ordinal=gate.ordinal,
        gate_id=gate.gate_id,
        allowed_native_source_kind_orders_in_exact_order=kind_orders,
        verification_availability=CURRENT_GATE_VERIFICATION_AVAILABILITY_V1,
        verifier_dispatch_id=dispatch_id,
        verifier_source_sha256_or_none=None,
    )
    for gate, kind_orders, dispatch_id in zip(
        GATE_SPECS_IN_EXACT_ORDER_V1,
        _GATE_NATIVE_SOURCE_KIND_ORDER_VARIANTS_V1,
        _GATE_VERIFIER_DISPATCH_IDS_IN_EXACT_ORDER_V1,
        strict=True,
    )
)


def _gate_lineage_scope_v1(ordinal: int) -> str:
    if ordinal <= 12:
        return "PREALLOCATION"
    if ordinal == 13:
        return "ALLOCATION_TRANSITION"
    return "ALLOCATED"


def _gate_campaign_attempt_policy_v1(ordinal: int) -> str:
    if ordinal <= 12:
        return (
            "CAMPAIGN_AND_ATTEMPT_EXACT_NULL_NO_FUTURE_ID_BACKFILL_OR_BACKSIGNING"
        )
    if ordinal == 13:
        return (
            "FIRST_AND_ONLY_ALLOCATION_TRANSITION_CONCRETE_FRESH_CAMPAIGN_AND_ATTEMPT_IDS_WHILE_PREDECESSOR_R12_IDS_REMAIN_NULL"
        )
    return (
        "CONCRETE_CAMPAIGN_AND_ATTEMPT_IDS_EXACT_EQUAL_R13_ALLOCATION_LINEAGE"
    )


def _gate_support_artifact_kind_v1(spec: _GateSpecV1) -> str:
    return f"{spec.gate_id}_AUTHORITATIVE_SUPPORT_V1"


def _gate_source_receipt_artifact_kind_v1(
    spec: _GateSourceSpecV1,
) -> str:
    return f"{spec.gate_id}_NATIVE_SOURCE_RECEIPT_V1"


def _gate_source_receipt_signature_slots_v1(
    spec: _GateSourceSpecV1,
) -> tuple[_SignatureSlotSpecV1, ...]:
    domain_prefix = (
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/GATE_SOURCE/"
        f"{spec.ordinal}/SIGN/"
    )
    purpose_suffix = f"{spec.gate_id}_NATIVE_SOURCE_RECEIPT"
    return (
        _SignatureSlotSpecV1(
            0,
            "GOVERNANCE_ANCHOR",
            f"ATTEST_{purpose_suffix}",
            1,
            domain_prefix + "GOVERNANCE/V1\x00",
        ),
        _SignatureSlotSpecV1(
            1,
            "CUSTODIAN",
            f"COUNTERSIGN_{purpose_suffix}_CUSTODY",
            1,
            domain_prefix + "CUSTODIAN/V1\x00",
        ),
        _SignatureSlotSpecV1(
            2,
            "STORE_RECEIPT",
            f"COUNTERSIGN_{purpose_suffix}_STORAGE",
            1,
            domain_prefix + "STORE_RECEIPT/V1\x00",
        ),
        _SignatureSlotSpecV1(
            3,
            "TIME_ATTESTOR",
            f"COUNTERSIGN_{purpose_suffix}_OBSERVATION_ORDER",
            1,
            domain_prefix + "TIME_ATTESTOR/V1\x00",
        ),
    )


_PREALLOCATION_LINEAGE_SIGNATURE_SLOTS_IN_EXACT_ORDER_V1: Final = (
    _SignatureSlotSpecV1(
        0,
        "GOVERNANCE_ANCHOR",
        "SIGN_PREALLOCATION_AUTHORITY_LINEAGE",
        1,
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PREALLOCATION_LINEAGE/"
        "SIGN/GOVERNANCE/V1\x00",
    ),
    _SignatureSlotSpecV1(
        1,
        "CUSTODIAN",
        "COUNTERSIGN_PREALLOCATION_CUSTODY",
        1,
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PREALLOCATION_LINEAGE/"
        "SIGN/CUSTODIAN/V1\x00",
    ),
    _SignatureSlotSpecV1(
        2,
        "STORE_RECEIPT",
        "COUNTERSIGN_PREALLOCATION_ANCHOR_STORAGE",
        1,
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PREALLOCATION_LINEAGE/"
        "SIGN/STORE_RECEIPT/V1\x00",
    ),
    _SignatureSlotSpecV1(
        3,
        "TIME_ATTESTOR",
        "COUNTERSIGN_PREALLOCATION_CURRENTNESS",
        1,
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PREALLOCATION_LINEAGE/"
        "SIGN/TIME_ATTESTOR/V1\x00",
    ),
)


def _gate_support_signature_slots_v1(
    spec: _GateSpecV1,
) -> tuple[_SignatureSlotSpecV1, ...]:
    domain_prefix = (
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/GATE_SUPPORT/"
        f"{spec.ordinal}/SIGN/"
    )
    purpose_suffix = f"{spec.gate_id}_SUPPORT"
    return (
        _SignatureSlotSpecV1(
            0,
            "GOVERNANCE_ANCHOR",
            f"ATTEST_{purpose_suffix}",
            1,
            domain_prefix + "GOVERNANCE/V1\x00",
        ),
        _SignatureSlotSpecV1(
            1,
            "CUSTODIAN",
            f"COUNTERSIGN_{purpose_suffix}_CUSTODY",
            1,
            domain_prefix + "CUSTODIAN/V1\x00",
        ),
        _SignatureSlotSpecV1(
            2,
            "STORE_RECEIPT",
            f"COUNTERSIGN_{purpose_suffix}_STORAGE",
            1,
            domain_prefix + "STORE_RECEIPT/V1\x00",
        ),
        _SignatureSlotSpecV1(
            3,
            "TIME_ATTESTOR",
            f"COUNTERSIGN_{purpose_suffix}_CURRENTNESS",
            1,
            domain_prefix + "TIME_ATTESTOR/V1\x00",
        ),
    )

STAGE_NAMES_IN_EXACT_ORDER_V1: Final = (
    "PREREGISTERED",
    "RAW_INPUT_COMMITTED",
    "PACKAGE_COMMITMENTS_FROZEN",
    "RUN_STARTED",
    "RUN_FINISHED",
    "PREDICTIONS_COMMITTED",
    "REVEALED",
    "SCORING_STARTED",
    "REPORTS_COMMITTED",
    "TERMINAL_CONSUMED",
)
_STAGE_PRODUCER_ROLES_V1: Final = (
    "GOVERNANCE_ANCHOR",
    "CUSTODIAN",
    "CUSTODIAN",
    "RECOGNIZER",
    "RECOGNIZER",
    "CUSTODIAN",
    "GOVERNANCE_ANCHOR",
    "SCORER_EVALUATOR",
    "SCORER_EVALUATOR",
    "CUSTODIAN",
)
_STAGE_CONSUMERS_V1: Final = (
    ("CUSTODIAN",),
    ("CUSTODIAN",),
    ("RECOGNIZER", "SCORER_EVALUATOR"),
    ("RECOGNIZER",),
    ("CUSTODIAN",),
    ("GOVERNANCE_ANCHOR",),
    ("SCORER_EVALUATOR",),
    ("SCORER_EVALUATOR",),
    ("CUSTODIAN", "GOVERNANCE_ANCHOR"),
    ("GOVERNANCE_ANCHOR",),
)


def _stage_predecessors_v1(index: int) -> tuple[str, ...]:
    if index == 0:
        return ("R13_FRESH_IDS_ALLOCATED_BY_VERIFIED_OVERLAY",)
    if index == 1:
        return ("PREREGISTERED", "R14_ACTUAL_UNSEALED_960_COHORT_GENERATED")
    if index == 2:
        return ("RAW_INPUT_COMMITTED", "R15_ACTUAL_CORPUS_COVERT_AUDIT_PASSED")
    if index == 3:
        return ("PACKAGE_COMMITMENTS_FROZEN", "R16_INFERENCE_ADMISSION_UNLOCKED")
    return (STAGE_NAMES_IN_EXACT_ORDER_V1[index - 1],)


STAGE_SPECS_IN_EXACT_ORDER_V1: Final = tuple(
    _StageSpecV1(
        index,
        stage,
        _stage_predecessors_v1(index),
        _STAGE_PRODUCER_ROLES_V1[index],
        _STAGE_CONSUMERS_V1[index],
        index >= 3,
        stage == "TERMINAL_CONSUMED",
    )
    for index, stage in enumerate(STAGE_NAMES_IN_EXACT_ORDER_V1)
)

INTERLEAVED_DAG_EDGES_IN_EXACT_ORDER_V1: Final = tuple(
    (dependency, gate.gate_id)
    for gate in GATE_SPECS_IN_EXACT_ORDER_V1[:13]
    for dependency in gate.dependencies_in_exact_order
) + (
    ("R13_FRESH_IDS_ALLOCATED_BY_VERIFIED_OVERLAY", "PREREGISTERED"),
    ("PREREGISTERED", "R14_ACTUAL_UNSEALED_960_COHORT_GENERATED"),
    ("R14_ACTUAL_UNSEALED_960_COHORT_GENERATED", "RAW_INPUT_COMMITTED"),
    ("RAW_INPUT_COMMITTED", "R15_ACTUAL_CORPUS_COVERT_AUDIT_PASSED"),
    ("R15_ACTUAL_CORPUS_COVERT_AUDIT_PASSED", "PACKAGE_COMMITMENTS_FROZEN"),
    ("PACKAGE_COMMITMENTS_FROZEN", "R16_INFERENCE_ADMISSION_UNLOCKED"),
    ("R16_INFERENCE_ADMISSION_UNLOCKED", "RUN_STARTED"),
) + tuple(
    (STAGE_NAMES_IN_EXACT_ORDER_V1[index - 1], STAGE_NAMES_IN_EXACT_ORDER_V1[index])
    for index in range(4, len(STAGE_NAMES_IN_EXACT_ORDER_V1))
)


_PACKAGE_ENVELOPE_FIELDS_V1: Final = (
    "schema_version",
    "body",
    "package_id",
    "signatures_in_exact_order",
)
_PACKAGE_COMMON_BODY_FIELDS_V1: Final = (
    "schema_version",
    "status",
    "claim_level",
    "catalog_id",
    "signer_registry_id",
    "signer_registry_payload_sha256",
    "signer_registry_entries_root_sha256",
    "package_ordinal",
    "package_name",
    "external_evidence_class",
    "external_evidence_disposition",
    "campaign_id",
    "attempt_id",
    "predecessor_package_instance_ids_in_exact_order",
    "predecessor_package_instance_payload_sha256s_in_exact_order",
    "interleaved_dag_root",
)
_DETACHED_SIGNATURE_FIELDS_V1: Final = (
    "schema_version",
    "algorithm",
    "role",
    "purpose",
    "key_epoch",
    "signer_key_id",
    "signature_hex",
)
_FIELD_TYPE_MANIFEST_FIELDS_V1: Final = (
    "field_name",
    "exact_wire_type",
)
_EXACT_BOOLEAN_FIELD_NAMES_V1: Final = frozenset(
    {
        "all_1532_rows_present",
        "all_bytes_canonical",
        "active",
        "any_raw_prediction_byte_produced",
        "append_applied",
        "complete",
        "current",
        "evidence_observed",
        "fail_closed_satisfied",
        "non_revoked",
        "passed",
        "point_passed",
        "readback_verified",
        "revoked",
        "satisfied",
        "stage_occurred",
        "superseded",
    }
)
_EXACT_INTEGER_FIELD_NAMES_V1: Final = frozenset(
    {
        "event_denominator",
        "event_count",
        "event_index",
        "event_numerator",
        "head_sequence",
        "gate_ordinal",
        "invalid_pair_count",
        "key_epoch",
        "lineage_sequence",
        "observation_sequence",
        "producer_observation_sequence",
        "legal_pair_count",
        "master_seed_uint32",
        "maximum_payload_bytes",
        "minimum_payload_bytes",
        "attempt_index",
        "anchor_sequence",
        "current_logical_time",
        "denominator",
        "overall_gate_count",
        "package_count",
        "package_ordinal",
        "point_mean_denominator",
        "point_mean_numerator",
        "numerator",
        "previous_logical_time",
        "raw_output_byte_count",
        "replicate_count",
        "signature_ordinal",
        "slice_gate_count",
        "stage_ordinal",
        "stored_artifact_size_bytes",
        "threshold_denominator",
        "threshold_numerator",
        "point_threshold_denominator",
        "point_threshold_numerator",
        "event_sequence",
    }
)
_EXACT_INTEGER_DECIMAL_FIELD_NAMES_V1: Final = frozenset({"master_seed_decimal"})
_EXACT_UNIT_INTERVAL_DECIMAL_FIELD_NAMES_V1: Final = frozenset(
    {"lower_bound_decimal", "upper_bound_decimal"}
)

DENOMINATOR_EVENT_KINDS_IN_EXACT_ORDER_V1: Final = (
    "ATTEMPT_OPEN",
    "ATTEMPT_PRE_SCORE_CLOSED",
    "PRE_SCORE_APPEND_RECEIPT",
    "MISSING_OUTPUT",
    "CRASH",
    "RETRY",
)

FROZEN_STATISTICAL_THRESHOLD_IDS_IN_EXACT_ORDER_V1: Final = (
    "PRESERVATION_LEGAL_POINT_MINIMUM",
    "PRESERVATION_LEGAL_WILSON_LOWER_MINIMUM",
    "PRESERVATION_INVALID_CONTROL_EXACT",
    "SCALE_REGRET_POINT_MEAN_MAXIMUM",
    "SCALE_REGRET_BOOTSTRAP_UPPER_MAXIMUM",
    "CHALLENGE_LOW_OVERLAP_POINT_MINIMUM",
    "CHALLENGE_LOW_OVERLAP_WILSON_LOWER_MINIMUM",
    "CHALLENGE_HIGH_OVERLAP_POINT_MINIMUM",
    "CHALLENGE_HIGH_OVERLAP_WILSON_LOWER_MINIMUM",
    "SEMANTIC_ADVANTAGE_POINT_MINIMUM",
    "SEMANTIC_ADVANTAGE_BOOTSTRAP_LOWER_MINIMUM",
)

VALIDATION_OVERALL_GATE_SPECS_IN_EXACT_ORDER_V1: Final = (
    ("OVERALL_FAMILY_EXACT", "family_exact", "overall", "all", 240, 90, 100, 86, 100),
    ("OVERALL_BINDING_EXACT", "binding_exact", "overall", "all", 240, 90, 100, 86, 100),
    ("OVERALL_SCALE_SET_ACCURACY", "scale_set_accuracy", "overall", "all", 240, 87, 100, 82, 100),
    ("OVERALL_JOINT_EXACT", "joint_exact", "overall", "all", 240, 85, 100, 80, 100),
    ("OVERALL_HARD_NEGATIVE_REJECTION", "hard_negative_rejection", "overall", "all", 96, 95, 100, 90, 100),
    ("OVERALL_BINDING_COUNTERFACTUAL_REJECTION", "binding_counterfactual_rejection", "overall", "all", 96, 95, 100, 90, 100),
    ("OVERALL_SCALE_COUNTERFACTUAL_REJECTION", "scale_counterfactual_rejection", "overall", "all", 96, 93, 100, 88, 100),
    ("OVERALL_SIGN_OR_INVARIANT_BREAK_REJECTION", "sign_or_invariant_break_rejection", "overall", "all", 96, 95, 100, 90, 100),
    ("OVERALL_ABSTENTION_SPECIFICITY", "abstention_specificity", "overall", "all", 228, 95, 100, 90, 100),
    ("OVERALL_FAIL_CLOSED_RATE", "fail_closed_rate", "overall", "all", None, 1, 1, None, None),
    ("OVERALL_PRESERVATION_CONSISTENCY", "preservation_consistency", "overall", "all", 572, 97, 100, 94, 100),
    ("OVERALL_NONIDENTIFIABLE_SCALE_ABSTENTION", "nonidentifiable_scale_abstention", "overall", "all", 96, 95, 100, 90, 100),
)
VALIDATION_FAMILIES_IN_EXACT_ORDER_V1: Final = (
    "F01_symmetry_equivariance",
    "F02_monotonicity_order",
    "F03_conservation_balance",
    "F04_additivity_complementarity",
    "F05_locality_composition",
    "F06_negative_feedback_stability",
)
VALIDATION_SCALES_IN_EXACT_ORDER_V1: Final = (
    "S0_FINE_LOCAL",
    "S1_COARSE_AGGREGATE",
)
VALIDATION_SLICE_GATE_BASE_SPECS_IN_EXACT_ORDER_V1: Final = (
    ("answerable_joint_exact", 80, 100, 70, 100),
    ("all_control_rejection", 88, 100, 78, 100),
    ("abstention_specificity", 85, 100, 75, 100),
)


def _validation_slice_gate_specs_v1() -> tuple[tuple[object, ...], ...]:
    rows: list[tuple[object, ...]] = []
    for scope_axis, scope_values in (
        ("family", VALIDATION_FAMILIES_IN_EXACT_ORDER_V1),
        ("scale", VALIDATION_SCALES_IN_EXACT_ORDER_V1),
    ):
        for scope_value in scope_values:
            for (
                metric,
                point_numerator,
                point_denominator,
                wilson_numerator,
                wilson_denominator,
            ) in VALIDATION_SLICE_GATE_BASE_SPECS_IN_EXACT_ORDER_V1:
                fixed_total_count = {
                    ("family", "answerable_joint_exact"): 40,
                    ("family", "all_control_rejection"): 64,
                    ("family", "abstention_specificity"): 38,
                    ("scale", "answerable_joint_exact"): 120,
                    ("scale", "all_control_rejection"): 192,
                    ("scale", "abstention_specificity"): 114,
                }[(scope_axis, metric)]
                statistical_gate_id = (
                    f"SLICE_{scope_axis}_{scope_value}_{metric}".upper()
                )
                rows.append(
                    (
                        statistical_gate_id,
                        metric,
                        scope_axis,
                        scope_value,
                        fixed_total_count,
                        point_numerator,
                        point_denominator,
                        wilson_numerator,
                        wilson_denominator,
                    )
                )
    return tuple(rows)


FROZEN_VALIDATION_GATE_RESULT_PLAN_IN_EXACT_ORDER_V1: Final = (
    VALIDATION_OVERALL_GATE_SPECS_IN_EXACT_ORDER_V1
    + _validation_slice_gate_specs_v1()
)
FROZEN_VALIDATION_STATISTICAL_GATE_IDS_IN_EXACT_ORDER_V1: Final = tuple(
    str(row[0]) for row in FROZEN_VALIDATION_GATE_RESULT_PLAN_IN_EXACT_ORDER_V1
)
FROZEN_VALIDATION_MINIMUM_PASSING_CORRECT_COUNTS_IN_EXACT_ORDER_V1: Final = (
    216,
    216,
    209,
    204,
    92,
    92,
    90,
    92,
    217,
    None,
    558,
    92,
    *((33, 57, 33) * 6),
    96,
    169,
    97,
    96,
    169,
    97,
)
WILSON_ONE_SIDED_95_Z_BINARY64_HEX_V1: Final = "0x1.a515209676ab8p+0"
WILSON_BINARY64_OPERATION_ORDER_IN_EXACT_ORDER_V1: Final = (
    "01_REQUIRE_CORRECT_AND_TOTAL_EXACT_SAFE_INTEGERS_BOOL_FORBIDDEN_ZERO_LE_CORRECT_LE_TOTAL_AND_TOTAL_GT_ZERO",
    "02_Z_EQUALS_FLOAT_FROMHEX_0X1_A515209676AB8P_PLUS_0",
    "03_P_EQUALS_BINARY64_TRUE_DIVIDE_CORRECT_BY_TOTAL",
    "04_Z2_EQUALS_BINARY64_MULTIPLY_Z_BY_Z",
    "05_DENOMINATOR_EQUALS_BINARY64_ADD_1_0_AND_BINARY64_TRUE_DIVIDE_Z2_BY_TOTAL",
    "06_CENTRE_EQUALS_BINARY64_ADD_P_AND_BINARY64_TRUE_DIVIDE_Z2_BY_BINARY64_MULTIPLY_2_0_BY_TOTAL",
    "07_ONE_MINUS_P_EQUALS_BINARY64_SUBTRACT_1_0_MINUS_P",
    "08_BERNOULLI_EQUALS_BINARY64_MULTIPLY_P_BY_ONE_MINUS_P",
    "09_QUARTER_TERM_EQUALS_BINARY64_TRUE_DIVIDE_Z2_BY_BINARY64_MULTIPLY_4_0_BY_TOTAL",
    "10_VARIANCE_NUMERATOR_EQUALS_BINARY64_ADD_BERNOULLI_AND_QUARTER_TERM",
    "11_VARIANCE_TERM_EQUALS_BINARY64_TRUE_DIVIDE_VARIANCE_NUMERATOR_BY_TOTAL",
    "12_RADIUS_EQUALS_BINARY64_MULTIPLY_Z_BY_MATH_SQRT_VARIANCE_TERM",
    "13_LOWER_EQUALS_BINARY64_TRUE_DIVIDE_BINARY64_SUBTRACT_CENTRE_MINUS_RADIUS_BY_DENOMINATOR",
    "14_SERIALIZE_LOWER_AS_PYTHON_FLOAT_HEX_EXACT_ASCII",
    "15_COMPARE_LOWER_AS_INTEGER_RATIO_TO_THRESHOLD_NUMERATOR_OVER_DENOMINATOR_BY_EXACT_INTEGER_CROSS_MULTIPLICATION_NO_DECIMAL_CONTEXT",
)
_VALIDATION_GATE_CROSS_RESULT_BINDING_RULES_IN_EXACT_ORDER_V1: Final = (
    (
        "FAIL_CLOSED_DYNAMIC_DENOMINATOR_BINDING",
        (
            "OVERALL_FAIL_CLOSED_RATE.correct_count",
            "OVERALL_FAIL_CLOSED_RATE.total_count",
            "package10.event_numerator",
            "package10.event_denominator",
            "package10.denominator_finalization_id",
        ),
        "FAIL_CLOSED_COUNTS_MUST_EXACT_EQUAL_THE_VERIFIED_FOUR_ROLE_SEALED_PRE_SCORE_ALL_ATTEMPT_EVENT_NUMERATOR_AND_DENOMINATOR_FROM_PREDECESSOR_PACKAGE10_WITH_RETRY_CAUSE_BIJECTION_VERIFIED",
    ),
    (
        "PRESERVATION_COMPOSITE_BINDING",
        (
            "OVERALL_PRESERVATION_CONSISTENCY",
            "package11.legal_correct_count",
            "package11.legal_pair_count",
            "package11.invalid_correct_count",
            "package11.invalid_pair_count",
        ),
        "IN_ADDITION_TO_555_OF_572_BASE_AND_94_OF_100_WILSON_REQUIRE_LEGAL_CORRECT_AT_LEAST_482_OF_496_WITH_94_OF_100_WILSON_AND_INVALID_CORRECT_EXACTLY_76_OF_76_SO_EFFECTIVE_MINIMUM_TOTAL_CORRECT_IS_558",
    ),
    (
        "FAMILY_AND_SCALE_ANSWERABLE_CROSS_SUM",
        (
            "SLICE_FAMILY_ANSWERABLE_JOINT_EXACT_ROWS",
            "SLICE_SCALE_ANSWERABLE_JOINT_EXACT_ROWS",
            "OVERALL_JOINT_EXACT",
        ),
        "SUM_OF_SIX_FAMILY_CORRECT_AND_SUM_OF_TWO_SCALE_CORRECT_MUST_EACH_EXACT_EQUAL_OVERALL_JOINT_EXACT_CORRECT_COUNT_AND_TOTALS_MUST_EACH_SUM_TO_240",
    ),
    (
        "FAMILY_AND_SCALE_CONTROL_CROSS_SUM",
        (
            "SLICE_FAMILY_ALL_CONTROL_REJECTION_ROWS",
            "SLICE_SCALE_ALL_CONTROL_REJECTION_ROWS",
            "OVERALL_FOUR_CONTROL_REJECTION_ROWS",
        ),
        "EACH_AXIS_CORRECT_SUM_MUST_EQUAL_THE_SUM_OF_HARD_NEGATIVE_BINDING_COUNTERFACTUAL_SCALE_COUNTERFACTUAL_AND_SIGN_OR_INVARIANT_BREAK_OVERALL_CORRECT_COUNTS_AND_TOTALS_MUST_EACH_SUM_TO_384",
    ),
    (
        "FAMILY_AND_SCALE_ABSTENTION_CROSS_SUM",
        (
            "SLICE_FAMILY_ABSTENTION_SPECIFICITY_ROWS",
            "SLICE_SCALE_ABSTENTION_SPECIFICITY_ROWS",
            "OVERALL_ABSTENTION_SPECIFICITY",
        ),
        "SUM_OF_SIX_FAMILY_CORRECT_AND_SUM_OF_TWO_SCALE_CORRECT_MUST_EACH_EXACT_EQUAL_OVERALL_ABSTENTION_SPECIFICITY_CORRECT_COUNT_AND_TOTALS_MUST_EACH_SUM_TO_228",
    ),
)

_GATE_ID_EXACT_WIRE_TYPE_V1: Final = (
    "EXACT_ASCII_ENUM_MEMBER_OF_GATE_IDS_IN_EXACT_ORDER"
)
GATE_LINEAGE_SCOPES_IN_EXACT_ORDER_V1: Final = (
    "PREALLOCATION",
    "ALLOCATION_TRANSITION",
    "ALLOCATED",
)
_GATE_LINEAGE_SCOPE_EXACT_WIRE_TYPE_V1: Final = (
    "EXACT_ASCII_ENUM_MEMBER_OF_GATE_LINEAGE_SCOPES_IN_EXACT_ORDER"
)
_NULL_OR_CAMPAIGN_ID_EXACT_WIRE_TYPE_V1: Final = (
    "EXACT_JSON_NULL_OR_PHASE2B_ACTUAL_UNSEALED_960_CAMPAIGN_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX"
)
_NULL_OR_ATTEMPT_ID_EXACT_WIRE_TYPE_V1: Final = (
    "EXACT_JSON_NULL_OR_PHASE2B_ACTUAL_UNSEALED_960_ATTEMPT_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX"
)
_GATE_VERIFICATION_AVAILABILITY_EXACT_WIRE_TYPE_V1: Final = (
    "EXACT_ASCII_ENUM_MEMBER_OF_GATE_VERIFICATION_AVAILABILITIES_IN_EXACT_ORDER"
)
_NULL_OR_SHA256_EXACT_WIRE_TYPE_V1: Final = (
    "EXACT_JSON_NULL_OR_ASCII_SHA256_COLON_64_LOWERHEX"
)
_INTEGER_DECIMAL_EXACT_WIRE_TYPE_V1: Final = (
    "EXACT_ASCII_CANONICAL_NONNEGATIVE_DECIMAL_INTEGER_NO_LEADING_ZERO"
)
_UNIT_INTERVAL_DECIMAL_EXACT_WIRE_TYPE_V1: Final = (
    "EXACT_ASCII_CANONICAL_UNIT_INTERVAL_DECIMAL"
)
_NULL_OR_UNIT_INTERVAL_DECIMAL_EXACT_WIRE_TYPE_V1: Final = (
    "EXACT_JSON_NULL_OR_ASCII_CANONICAL_UNIT_INTERVAL_DECIMAL"
)
_THRESHOLD_ID_EXACT_WIRE_TYPE_V1: Final = (
    "EXACT_ASCII_ENUM_MEMBER_OF_FROZEN_STATISTICAL_THRESHOLD_IDS_IN_EXACT_ORDER"
)
_STATISTICAL_GATE_ID_EXACT_WIRE_TYPE_V1: Final = (
    "EXACT_ASCII_ENUM_MEMBER_OF_FROZEN_VALIDATION_STATISTICAL_GATE_IDS_IN_EXACT_ORDER"
)
_DENOMINATOR_EVENT_KIND_EXACT_WIRE_TYPE_V1: Final = (
    "EXACT_ASCII_ENUM_MEMBER_OF_DENOMINATOR_EVENT_KINDS_IN_EXACT_ORDER"
)
_STAGE_NAME_EXACT_WIRE_TYPE_V1: Final = (
    "EXACT_ASCII_ENUM_MEMBER_OF_STAGE_NAMES_IN_EXACT_ORDER"
)
_NULL_OR_SAFE_INTEGER_EXACT_WIRE_TYPE_V1: Final = (
    "EXACT_JSON_NULL_OR_SAFE_INTEGER_MIN_0_BOOL_FORBIDDEN"
)
_NULL_OR_BOOLEAN_EXACT_WIRE_TYPE_V1: Final = "EXACT_JSON_NULL_OR_BOOLEAN"
_NULL_OR_BINARY64_HEX_EXACT_WIRE_TYPE_V1: Final = (
    "EXACT_JSON_NULL_OR_ASCII_IEEE754_BINARY64_HEX"
)
_NULL_OR_ATTEMPT_RECORD_ID_EXACT_WIRE_TYPE_V1: Final = (
    "EXACT_JSON_NULL_OR_ATTEMPT_RECORD_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX"
)
_NULL_OR_LEDGER_EVENT_ID_EXACT_WIRE_TYPE_V1: Final = (
    "EXACT_JSON_NULL_OR_SIGNED_STAGE_EVENT_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX"
)
_NULL_OR_EXTERNAL_ANCHOR_ID_EXACT_WIRE_TYPE_V1: Final = (
    "EXACT_JSON_NULL_OR_EXTERNAL_ANCHOR_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX"
)
_NULL_OR_DENOMINATOR_EVENT_ID_EXACT_WIRE_TYPE_V1: Final = (
    "EXACT_JSON_NULL_OR_DENOMINATOR_EVENT_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX"
)
_NULL_OR_ATTEMPT_UNIVERSE_HEAD_ID_EXACT_WIRE_TYPE_V1: Final = (
    "EXACT_JSON_NULL_OR_ATTEMPT_UNIVERSE_HEAD_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX"
)
_NULL_OR_PREALLOCATION_LINEAGE_ID_EXACT_WIRE_TYPE_V1: Final = (
    "EXACT_JSON_NULL_OR_PREALLOCATION_LINEAGE_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX"
)
TERMINAL_DISPOSITIONS_IN_EXACT_ORDER_V1: Final = (
    "NOT_RUN",
    "BLOCKED",
    "REJECTED",
    "INVALIDATED",
    "COMPLETE_FAIL",
    "COMPLETE_PASS",
)
COMPLETED_RUN_DISPOSITIONS_IN_EXACT_ORDER_V1: Final = (
    "COMPLETE_FAIL",
    "COMPLETE_PASS",
)
_TERMINAL_DISPOSITION_EXACT_WIRE_TYPE_V1: Final = (
    "EXACT_ASCII_ENUM_MEMBER_OF_TERMINAL_DISPOSITIONS_IN_EXACT_ORDER"
)
_NULL_OR_TERMINAL_DISPOSITION_EXACT_WIRE_TYPE_V1: Final = (
    "EXACT_JSON_NULL_OR_ENUM_MEMBER_OF_TERMINAL_DISPOSITIONS_IN_EXACT_ORDER"
)
_COMPLETED_RUN_DISPOSITION_EXACT_WIRE_TYPE_V1: Final = (
    "EXACT_ASCII_ENUM_MEMBER_OF_COMPLETED_RUN_DISPOSITIONS_IN_EXACT_ORDER"
)
EXACT_SPECIAL_WIRE_TYPE_PROFILES_IN_EXACT_ORDER_V1: Final = (
    (
        _GATE_ID_EXACT_WIRE_TYPE_V1,
        "EXACT_ENUM_MEMBERSHIP",
        "NONE",
        tuple(gate.gate_id for gate in GATE_SPECS_IN_EXACT_ORDER_V1),
    ),
    (
        _GATE_LINEAGE_SCOPE_EXACT_WIRE_TYPE_V1,
        "EXACT_ENUM_MEMBERSHIP",
        "NONE",
        GATE_LINEAGE_SCOPES_IN_EXACT_ORDER_V1,
    ),
    (
        _GATE_VERIFICATION_AVAILABILITY_EXACT_WIRE_TYPE_V1,
        "EXACT_ENUM_MEMBERSHIP",
        "NONE",
        GATE_VERIFICATION_AVAILABILITIES_IN_EXACT_ORDER_V1,
    ),
    (
        _NULL_OR_SHA256_EXACT_WIRE_TYPE_V1,
        "EXACT_JSON_NULL_OR_SHA256_COLON_64_LOWERHEX",
        "NONE",
        ("null", "sha256:<64_lowerhex>"),
    ),
    (
        _INTEGER_DECIMAL_EXACT_WIRE_TYPE_V1,
        "FULLMATCH_ASCII_REGEX",
        "(?:0|[1-9][0-9]*)",
        (),
    ),
    (
        _UNIT_INTERVAL_DECIMAL_EXACT_WIRE_TYPE_V1,
        "FULLMATCH_ASCII_REGEX_AND_EXACT_INTEGER_CROSS_MULTIPLICATION_RANGE_ZERO_TO_ONE_INCLUSIVE_NO_BINARY_FLOAT",
        "(?:0|1|0\\.[0-9]*[1-9])",
        ("0.05", "0.08", "0.1", "0.15", "0.85", "0.9", "0.94", "0.95"),
    ),
    (
        _NULL_OR_UNIT_INTERVAL_DECIMAL_EXACT_WIRE_TYPE_V1,
        "EXACT_JSON_NULL_OR_FULLMATCH_ASCII_REGEX_AND_EXACT_INTEGER_CROSS_MULTIPLICATION_RANGE_ZERO_TO_ONE_INCLUSIVE_NO_BINARY_FLOAT",
        "(?:0|1|0\\.[0-9]*[1-9])",
        ("null", "0.05", "0.08", "0.1", "0.15", "0.85", "0.9", "0.94", "0.95"),
    ),
    (
        _THRESHOLD_ID_EXACT_WIRE_TYPE_V1,
        "EXACT_ENUM_MEMBERSHIP",
        "NONE",
        FROZEN_STATISTICAL_THRESHOLD_IDS_IN_EXACT_ORDER_V1,
    ),
    (
        _STATISTICAL_GATE_ID_EXACT_WIRE_TYPE_V1,
        "EXACT_ENUM_MEMBERSHIP",
        "NONE",
        FROZEN_VALIDATION_STATISTICAL_GATE_IDS_IN_EXACT_ORDER_V1,
    ),
    (
        _DENOMINATOR_EVENT_KIND_EXACT_WIRE_TYPE_V1,
        "EXACT_ENUM_MEMBERSHIP",
        "NONE",
        DENOMINATOR_EVENT_KINDS_IN_EXACT_ORDER_V1,
    ),
    (
        _STAGE_NAME_EXACT_WIRE_TYPE_V1,
        "EXACT_ENUM_MEMBERSHIP",
        "NONE",
        STAGE_NAMES_IN_EXACT_ORDER_V1,
    ),
    (
        _NULL_OR_SAFE_INTEGER_EXACT_WIRE_TYPE_V1,
        "EXACT_JSON_NULL_OR_SAFE_INTEGER_MIN_0_BOOL_FORBIDDEN",
        "NONE",
        ("null", "0", "1", "240", "572"),
    ),
    (
        _NULL_OR_BOOLEAN_EXACT_WIRE_TYPE_V1,
        "EXACT_JSON_NULL_OR_BOOLEAN",
        "NONE",
        ("null", "false", "true"),
    ),
    (
        _NULL_OR_BINARY64_HEX_EXACT_WIRE_TYPE_V1,
        "EXACT_JSON_NULL_OR_PYTHON_FLOAT_HEX_OF_FINITE_IEEE754_BINARY64",
        "NONE",
        ("null", "0x0.0p+0", "0x1.0000000000000p+0"),
    ),
    (
        _NULL_OR_ATTEMPT_RECORD_ID_EXACT_WIRE_TYPE_V1,
        "EXACT_JSON_NULL_OR_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX",
        "NONE",
        ("null", "phase2b_actual_unsealed_960_attempt_record_v1_<64_lowerhex>"),
    ),
    (
        _NULL_OR_LEDGER_EVENT_ID_EXACT_WIRE_TYPE_V1,
        "EXACT_JSON_NULL_OR_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX",
        "NONE",
        ("null", "phase2b_actual_unsealed_960_signed_stage_event_v1_<64_lowerhex>"),
    ),
    (
        _NULL_OR_EXTERNAL_ANCHOR_ID_EXACT_WIRE_TYPE_V1,
        "EXACT_JSON_NULL_OR_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX",
        "NONE",
        ("null", "phase2b_actual_unsealed_960_external_anchor_v1_<64_lowerhex>"),
    ),
    (
        _NULL_OR_DENOMINATOR_EVENT_ID_EXACT_WIRE_TYPE_V1,
        "EXACT_JSON_NULL_OR_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX",
        "NONE",
        (
            "null",
            "phase2b_actual_unsealed_960_denominator_event_v1_<64_lowerhex>",
        ),
    ),
    (
        _NULL_OR_ATTEMPT_UNIVERSE_HEAD_ID_EXACT_WIRE_TYPE_V1,
        "EXACT_JSON_NULL_OR_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX",
        "NONE",
        (
            "null",
            "phase2b_actual_unsealed_960_attempt_universe_head_v1_<64_lowerhex>",
        ),
    ),
    (
        _NULL_OR_PREALLOCATION_LINEAGE_ID_EXACT_WIRE_TYPE_V1,
        "EXACT_JSON_NULL_OR_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX",
        "NONE",
        (
            "null",
            "phase2b_actual_unsealed_960_preallocation_lineage_v1_<64_lowerhex>",
        ),
    ),
    (
        _NULL_OR_CAMPAIGN_ID_EXACT_WIRE_TYPE_V1,
        "EXACT_JSON_NULL_OR_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX",
        "NONE",
        ("null", "phase2b_actual_unsealed_960_campaign_v1_<64_lowerhex>"),
    ),
    (
        _NULL_OR_ATTEMPT_ID_EXACT_WIRE_TYPE_V1,
        "EXACT_JSON_NULL_OR_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX",
        "NONE",
        ("null", "phase2b_actual_unsealed_960_attempt_v1_<64_lowerhex>"),
    ),
    (
        _TERMINAL_DISPOSITION_EXACT_WIRE_TYPE_V1,
        "EXACT_ENUM_MEMBERSHIP",
        "NONE",
        TERMINAL_DISPOSITIONS_IN_EXACT_ORDER_V1,
    ),
    (
        _NULL_OR_TERMINAL_DISPOSITION_EXACT_WIRE_TYPE_V1,
        "EXACT_JSON_NULL_OR_ENUM_MEMBERSHIP",
        "NONE",
        ("null",) + TERMINAL_DISPOSITIONS_IN_EXACT_ORDER_V1,
    ),
    (
        _COMPLETED_RUN_DISPOSITION_EXACT_WIRE_TYPE_V1,
        "EXACT_ENUM_MEMBERSHIP",
        "NONE",
        COMPLETED_RUN_DISPOSITIONS_IN_EXACT_ORDER_V1,
    ),
)


def _exact_wire_type_v1(field_name: str) -> str:
    if field_name == "body":
        return "EXACT_JSON_OBJECT_WITH_FROZEN_KEYSET"
    if field_name == "signatures_in_exact_order":
        return "EXACT_JSON_ARRAY_OF_DETACHED_SIGNATURE_OBJECTS"
    if field_name == "gate_id":
        return _GATE_ID_EXACT_WIRE_TYPE_V1
    if field_name == "lineage_scope":
        return _GATE_LINEAGE_SCOPE_EXACT_WIRE_TYPE_V1
    if field_name == "verification_availability":
        return _GATE_VERIFICATION_AVAILABILITY_EXACT_WIRE_TYPE_V1
    if field_name == "verifier_dispatch_id":
        return "EXACT_NONEMPTY_ASCII_ENUM_OR_IDENTIFIER_STRING"
    if field_name == "verifier_source_sha256_or_none":
        return _NULL_OR_SHA256_EXACT_WIRE_TYPE_V1
    if field_name == "native_source_payload_byte_counts_in_exact_order":
        return "EXACT_JSON_ARRAY_OF_SAFE_INTEGERS_MIN_0_BOOL_FORBIDDEN"
    if field_name == "threshold_id":
        return _THRESHOLD_ID_EXACT_WIRE_TYPE_V1
    if field_name == "statistical_gate_id":
        return _STATISTICAL_GATE_ID_EXACT_WIRE_TYPE_V1
    if field_name == "event_kind":
        return _DENOMINATOR_EVENT_KIND_EXACT_WIRE_TYPE_V1
    if field_name in {"stage", "sealed_stage"}:
        return _STAGE_NAME_EXACT_WIRE_TYPE_V1
    if field_name in {
        "fixed_total_count_or_none",
        "wilson_threshold_numerator_or_none",
        "wilson_threshold_denominator_or_none",
    }:
        return _NULL_OR_SAFE_INTEGER_EXACT_WIRE_TYPE_V1
    if field_name == "wilson_passed_or_none":
        return _NULL_OR_BOOLEAN_EXACT_WIRE_TYPE_V1
    if field_name == "wilson_lower_binary64_hex_or_none":
        return _NULL_OR_BINARY64_HEX_EXACT_WIRE_TYPE_V1
    if field_name == "threshold_decimal_or_none":
        return _NULL_OR_UNIT_INTERVAL_DECIMAL_EXACT_WIRE_TYPE_V1
    if field_name == "predecessor_attempt_record_id_or_none":
        return _NULL_OR_ATTEMPT_RECORD_ID_EXACT_WIRE_TYPE_V1
    if field_name == "predecessor_event_id_or_none":
        return _NULL_OR_LEDGER_EVENT_ID_EXACT_WIRE_TYPE_V1
    if field_name == "previous_external_anchor_id_or_none":
        return _NULL_OR_EXTERNAL_ANCHOR_ID_EXACT_WIRE_TYPE_V1
    if field_name == "predecessor_denominator_event_id_or_none":
        return _NULL_OR_DENOMINATOR_EVENT_ID_EXACT_WIRE_TYPE_V1
    if field_name == "previous_attempt_universe_head_id_or_none":
        return _NULL_OR_ATTEMPT_UNIVERSE_HEAD_ID_EXACT_WIRE_TYPE_V1
    if field_name == "previous_preallocation_lineage_id_or_none":
        return _NULL_OR_PREALLOCATION_LINEAGE_ID_EXACT_WIRE_TYPE_V1
    if field_name == "campaign_id_or_none":
        return _NULL_OR_CAMPAIGN_ID_EXACT_WIRE_TYPE_V1
    if field_name == "attempt_id_or_none":
        return _NULL_OR_ATTEMPT_ID_EXACT_WIRE_TYPE_V1
    if field_name in {"append_applied_or_none", "readback_verified_or_none"}:
        return _NULL_OR_BOOLEAN_EXACT_WIRE_TYPE_V1
    if field_name == "terminal_disposition":
        return _TERMINAL_DISPOSITION_EXACT_WIRE_TYPE_V1
    if field_name == "terminal_disposition_or_none":
        return _NULL_OR_TERMINAL_DISPOSITION_EXACT_WIRE_TYPE_V1
    if field_name == "final_disposition":
        return _COMPLETED_RUN_DISPOSITION_EXACT_WIRE_TYPE_V1
    if field_name == "anchor_service_id":
        return "EXACT_NONEMPTY_ASCII_SERVICE_IDENTIFIER"
    if field_name == "interleaved_dag_root":
        return "EXACT_ASCII_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX"
    if field_name in _EXACT_BOOLEAN_FIELD_NAMES_V1 or field_name.endswith(
        "_verified"
    ):
        return "EXACT_JSON_BOOLEAN"
    if field_name.endswith("_or_none"):
        return "EXACT_JSON_NULL_OR_NONEMPTY_ASCII_ENUM_STRING"
    if field_name in _EXACT_INTEGER_FIELD_NAMES_V1 or field_name.endswith(
        "_count"
    ):
        return "EXACT_JSON_SAFE_INTEGER_MIN_0_BOOL_FORBIDDEN"
    if field_name.endswith("_indices_in_exact_order") or field_name.endswith(
        "_ordinals_in_exact_order"
    ):
        return "EXACT_JSON_ARRAY_OF_SAFE_INTEGERS_BOOL_FORBIDDEN"
    if field_name.endswith("_payload_sha256s_in_exact_order"):
        return "EXACT_JSON_ARRAY_OF_SHA256_COLON_64_LOWERHEX"
    if field_name.endswith("_ids_in_exact_order"):
        return "EXACT_JSON_ARRAY_OF_CONTENT_IDENTIFIERS"
    if field_name.endswith("_in_exact_order"):
        return "EXACT_JSON_ARRAY_OF_ASCII_STRINGS"
    if field_name.endswith("_payload_sha256") or field_name.endswith(
        "_root_sha256"
    ) or field_name.endswith("_sha256"):
        return "EXACT_ASCII_SHA256_COLON_64_LOWERHEX"
    if field_name.endswith("_id"):
        return "EXACT_ASCII_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX"
    if field_name.endswith("_hex"):
        return "EXACT_ASCII_EVEN_LENGTH_LOWERHEX"
    if field_name in _EXACT_INTEGER_DECIMAL_FIELD_NAMES_V1:
        return _INTEGER_DECIMAL_EXACT_WIRE_TYPE_V1
    if field_name in _EXACT_UNIT_INTERVAL_DECIMAL_FIELD_NAMES_V1:
        return _UNIT_INTERVAL_DECIMAL_EXACT_WIRE_TYPE_V1
    if field_name.endswith("_decimal"):
        return "EXACT_ASCII_NONNEGATIVE_FIXED_POINT_DECIMAL_NO_SIGN_NO_EXPONENT"
    return "EXACT_NONEMPTY_ASCII_ENUM_OR_IDENTIFIER_STRING"


def _field_type_manifest_v1(
    field_names: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    return tuple((field_name, _exact_wire_type_v1(field_name)) for field_name in field_names)


_GATE_EVIDENCE_ENVELOPE_FIELDS_V1: Final = (
    "schema_version",
    "body",
    "gate_evidence_id",
    "signatures_in_exact_order",
)
_PREALLOCATION_LINEAGE_ENVELOPE_FIELDS_V1: Final = (
    "schema_version",
    "body",
    "preallocation_lineage_id",
    "signatures_in_exact_order",
)
_PREALLOCATION_LINEAGE_BODY_FIELDS_V1: Final = (
    "schema_version",
    "status",
    "claim_level",
    "catalog_id",
    "catalog_payload_sha256",
    "signer_registry_id",
    "signer_registry_payload_sha256",
    "signer_registry_entries_root_sha256",
    "readiness_dag_id",
    "readiness_dag_payload_sha256",
    "progression_plan_payload_sha256",
    "gap_inventory_source_sha256",
    "statistical_validation_source_sha256",
    "authority_anchor_id",
    "authority_anchor_payload_sha256",
    "currentness_token_id",
    "currentness_token_payload_sha256",
    "custody_checkpoint_id",
    "custody_checkpoint_payload_sha256",
    "lineage_sequence",
    "previous_preallocation_lineage_id_or_none",
    "active",
    "revoked",
    "complete",
    "first_failure_code",
)
_GATE_SOURCE_RECEIPT_ENVELOPE_FIELDS_V1: Final = (
    "schema_version",
    "body",
    "gate_source_receipt_id",
    "signatures_in_exact_order",
)
_GATE_SOURCE_RECEIPT_BODY_FIELDS_V1: Final = (
    "schema_version",
    "status",
    "claim_level",
    "catalog_id",
    "catalog_payload_sha256",
    "signer_registry_id",
    "signer_registry_payload_sha256",
    "signer_registry_entries_root_sha256",
    "preallocation_lineage_id",
    "preallocation_lineage_payload_sha256",
    "lineage_scope",
    "campaign_id_or_none",
    "attempt_id_or_none",
    "gate_ordinal",
    "gate_id",
    "gate_schema_id",
    "verification_availability",
    "verifier_dispatch_id",
    "verifier_source_sha256_or_none",
    "native_source_kinds_in_exact_order",
    "native_source_payload_sha256s_in_exact_order",
    "native_source_payload_byte_counts_in_exact_order",
    "source_evidence_root_sha256",
    "producer_observation_sequence",
    "predecessor_gate_source_receipt_ids_in_exact_order",
    "predecessor_gate_source_receipt_payload_sha256s_in_exact_order",
    "first_failure_code",
)
_GATE_SUPPORT_ARTIFACT_ENVELOPE_FIELDS_V1: Final = (
    "schema_version",
    "body",
    "gate_support_artifact_id",
    "signatures_in_exact_order",
)
_GATE_OPERATIONAL_EVIDENCE_FIELDS_V1: Final = (
    "schema_version",
    "status",
    "claim_level",
    "catalog_id",
    "preallocation_lineage_id",
    "preallocation_lineage_payload_sha256",
    "lineage_scope",
    "campaign_id_or_none",
    "attempt_id_or_none",
    "gate_ordinal",
    "gate_id",
    "gate_source_schema_id",
    "gate_source_receipt_id",
    "gate_source_receipt_payload_sha256",
    "verification_availability",
    "verifier_dispatch_id",
    "verifier_source_sha256_or_none",
    "evidence_requirement",
    "evidence_assertions_in_exact_order",
    "source_artifact_kinds_in_exact_order",
    "source_artifact_ids_in_exact_order",
    "source_artifact_payload_sha256s_in_exact_order",
    "source_evidence_root_sha256",
    "observation_sequence",
    "evidence_observed",
    "current",
    "non_revoked",
    "satisfied",
    "first_failure_code",
)
_GATE_SUPPORT_ARTIFACT_BODY_FIELDS_V1: Final = (
    "schema_version",
    "status",
    "claim_level",
    "catalog_id",
    "catalog_payload_sha256",
    "signer_registry_id",
    "signer_registry_payload_sha256",
    "signer_registry_entries_root_sha256",
    "preallocation_lineage_id",
    "preallocation_lineage_payload_sha256",
    "lineage_scope",
    "campaign_id_or_none",
    "attempt_id_or_none",
    "gate_ordinal",
    "gate_id",
    "gate_schema_id",
    "gate_source_schema_id",
    "gate_source_receipt_id",
    "gate_source_receipt_payload_sha256",
    "verification_availability",
    "verifier_dispatch_id",
    "verifier_source_sha256_or_none",
    "support_artifact_kind",
    "evidence_requirement",
    "evidence_assertions_in_exact_order",
    "evidence_payload_format",
    "evidence_payload_hex",
    "evidence_payload_sha256",
    "evidence_payload_byte_count",
    "evidence_root_sha256",
    "authority_anchor_id",
    "authority_anchor_payload_sha256",
    "currentness_token_id",
    "currentness_token_payload_sha256",
    "custody_checkpoint_id",
    "custody_checkpoint_payload_sha256",
    "evidence_observed",
    "current",
    "non_revoked",
    "satisfied",
    "first_failure_code",
)
_GATE_EVIDENCE_BODY_FIELDS_V1: Final = (
    "schema_version",
    "status",
    "claim_level",
    "catalog_id",
    "signer_registry_id",
    "signer_registry_payload_sha256",
    "signer_registry_entries_root_sha256",
    "preallocation_lineage_id",
    "preallocation_lineage_payload_sha256",
    "lineage_scope",
    "campaign_id_or_none",
    "attempt_id_or_none",
    "gate_ordinal",
    "gate_id",
    "gate_schema_id",
    "evidence_requirement",
    "predecessor_gate_evidence_ids_in_exact_order",
    "predecessor_gate_evidence_payload_sha256s_in_exact_order",
    "supporting_artifact_kinds_in_exact_order",
    "supporting_artifact_ids_in_exact_order",
    "supporting_artifact_payload_sha256s_in_exact_order",
    "evidence_root_sha256",
    "satisfied",
    "first_failure_code",
)
_STAGE_EVENT_ENVELOPE_FIELDS_V1: Final = (
    "schema_version",
    "body",
    "stage_event_id",
    "signatures_in_exact_order",
)
_STAGE_EVENT_BODY_FIELDS_V1: Final = (
    "schema_version",
    "status",
    "claim_level",
    "catalog_id",
    "signer_registry_id",
    "signer_registry_payload_sha256",
    "signer_registry_entries_root_sha256",
    "campaign_id",
    "attempt_id",
    "stage_ordinal",
    "stage",
    "predecessor_node_names_in_exact_order",
    "predecessor_evidence_or_event_ids_in_exact_order",
    "predecessor_evidence_or_event_payload_sha256s_in_exact_order",
    "subject_artifact_kinds_in_exact_order",
    "subject_artifact_ids_in_exact_order",
    "subject_artifact_payload_sha256s_in_exact_order",
    "event_sequence",
    "terminal_disposition_or_none",
    "stage_occurred",
    "first_failure_code",
)
_BUNDLE_INDEX_ENVELOPE_FIELDS_V1: Final = (
    "schema_version",
    "body",
    "bundle_index_id",
    "signatures_in_exact_order",
)
_BUNDLE_INDEX_BODY_FIELDS_V1: Final = (
    "schema_version",
    "status",
    "claim_level",
    "catalog_id",
    "signer_registry_id",
    "signer_registry_payload_sha256",
    "signer_registry_entries_root_sha256",
    "campaign_id",
    "attempt_id",
    "package_schema_ids_in_exact_order",
    "package_instance_ids_in_exact_order",
    "package_instance_payload_sha256s_in_exact_order",
    "package_count",
    "terminal_stage_event_id",
    "terminal_stage_event_payload_sha256",
    "final_disposition",
    "bundle_root_sha256",
    "interleaved_dag_root",
)

_NEGATIVE_VECTOR_FAMILIES_V1: Final = (
    "WRONG_EXACT_TYPE_OR_BOOL_AS_INT",
    "MISSING_EXTRA_OR_REORDERED_FIELD",
    "NONCANONICAL_RFC8785_BYTES",
    "WRONG_CONTENT_ID_DOMAIN_PREFIX_OR_PREIMAGE",
    "WRONG_PREDECESSOR_OR_FORWARD_REFERENCE",
    "EMPTY_REQUIRED_INSTANCE_LINEAGE",
    "WRONG_SCHEMA_INSTANCE_SPLICE",
    "PREDECESSOR_INSTANCE_ORDER_SWAP",
    "PREDECESSOR_RAW_DIGEST_MISMATCH",
    "WRONG_SIGNATURE_ROLE_PURPOSE_EPOCH_OR_CARDINALITY",
    "CROSS_CAMPAIGN_ATTEMPT_OR_PACKAGE_SPLICE",
    "OVERSIZE_OR_CARDINALITY_DRIFT",
    "CLAIM_ELEVATION_OR_TERMINAL_DRIFT",
)


def _signature_slot_v1(
    ordinal: int, role: str, purpose: str, package_ordinal: int
) -> _SignatureSlotSpecV1:
    return _SignatureSlotSpecV1(
        ordinal,
        role,
        purpose,
        1,
        (
            "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PACKAGE/"
            f"{package_ordinal}/SIGN/{purpose}/V1\x00"
        ),
    )


def _single_slot_v1(
    role: str, purpose: str, package_ordinal: int
) -> tuple[_SignatureSlotSpecV1, ...]:
    return (_signature_slot_v1(0, role, purpose, package_ordinal),)


_PACKAGE_ROWS_V1: Final = (
    (0, "durable_signed_custodian_ledger_and_signer_chain", "authenticate_one_shot_append_only_custody", ("signed_stage_event_ids_in_exact_order", "signed_stage_event_payload_sha256s_in_exact_order", "store_if_absent_receipt_id", "store_if_absent_receipt_payload_sha256", "logical_time_token_id", "logical_time_token_payload_sha256", "governance_checkpoint_id", "governance_checkpoint_payload_sha256", "custodian_checkpoint_id", "custodian_checkpoint_payload_sha256", "external_anchor_id", "external_anchor_payload_sha256", "chain_root_sha256"), (), ("R10_DURABLE_CUSTODY_AND_LEDGER_VERIFIED",), "PREREGISTERED", ("TERMINAL_CONSUMED",), "ONE_TERMINAL_APPEND_ONLY_CHAIN_NO_EVENT_AFTER_TERMINAL"),
    (1, "prediction_reveal_scoring_chronology", "prove_package_commitment_precedes_run_and_predictions_precede_reveal", ("stage_event_ids_in_exact_order", "stage_event_payload_sha256s_in_exact_order", "prediction_commitment_id", "prediction_commitment_payload_sha256", "reveal_receipt_id", "reveal_receipt_payload_sha256", "scoring_receipt_id", "scoring_receipt_payload_sha256", "chronology_root_sha256"), (), ("R10_DURABLE_CUSTODY_AND_LEDGER_VERIFIED", "R16_INFERENCE_ADMISSION_UNLOCKED"), "PREDICTIONS_COMMITTED", ("REVEALED", "SCORING_STARTED", "TERMINAL_CONSUMED"), "PREDICTIONS_MUST_PRECEDE_REVEAL_AND_SCORING"),
    (2, "raw_input_and_prediction_archive_cas", "bind_authoritative_input_and_output_bytes", ("input_archive_id", "input_archive_payload_sha256", "prediction_archive_ids_in_exact_order", "prediction_archive_payload_sha256s_in_exact_order", "cas_plan_id", "cas_plan_payload_sha256", "cas_complete_head_id", "cas_complete_head_payload_sha256", "readback_verified"), (), ("R10_DURABLE_CUSTODY_AND_LEDGER_VERIFIED", "R15_ACTUAL_CORPUS_COVERT_AUDIT_PASSED", "R16_INFERENCE_ADMISSION_UNLOCKED"), "RAW_INPUT_COMMITTED", ("PREDICTIONS_COMMITTED", "TERMINAL_CONSUMED"), "IF_ABSENT_READBACK_COMPLETE_HEAD_REQUIRED"),
    (3, "canonical_archive_replay_and_decoder_transcript", "prove_public_canonical_input_and_prediction_replay", ("decoder_manifest_id", "decoder_manifest_payload_sha256", "input_archive_id", "input_archive_payload_sha256", "prediction_archive_ids_in_exact_order", "prediction_archive_payload_sha256s_in_exact_order", "decoder_transcript_id", "decoder_transcript_payload_sha256", "replay_result_root_sha256", "all_bytes_canonical"), (2,), ("R03_FORMAL_TRUSTED_WIRE_IMPLEMENTATION_QUALIFIED", "R16_INFERENCE_ADMISSION_UNLOCKED"), "PREDICTIONS_COMMITTED", ("SCORING_STARTED", "TERMINAL_CONSUMED"), "PRODUCER_FREE_REPLAY_MUST_CLOSE_ALL_ARCHIVES"),
    (4, "source_image_sbom_config_runtime_execution_attestations", "authenticate_frozen_executable_environment", ("source_manifest_id", "source_manifest_payload_sha256", "image_manifest_id", "image_manifest_payload_sha256", "sbom_id", "sbom_payload_sha256", "configuration_id", "configuration_payload_sha256", "runtime_attestation_ids_in_exact_order", "runtime_attestation_payload_sha256s_in_exact_order", "environment_root_sha256"), (), ("R09_SIX_ROLE_LIVE_DOCKER_QUALIFIED", "R11_EXECUTION_FREEZE_COMMITMENTS_BOUND"), "PACKAGE_COMMITMENTS_FROZEN", ("RUN_STARTED", "TERMINAL_CONSUMED"), "FROZEN_ENVIRONMENT_MUST_MATCH_ENTRY_EXIT_ATTESTATIONS"),
    (5, "actual_recognizer_execution_capacity_and_output_attestation", "prove_recognizer_execution_capacity_and_output_origin", ("recognizer_manifest_id", "recognizer_manifest_payload_sha256", "run_identity_id", "run_identity_payload_sha256", "capacity_receipt_id", "capacity_receipt_payload_sha256", "exit_receipt_id", "exit_receipt_payload_sha256", "output_archive_id", "output_archive_payload_sha256", "output_origin_attestation_id", "output_origin_attestation_payload_sha256"), (2, 4), ("R16_INFERENCE_ADMISSION_UNLOCKED",), "RUN_STARTED", ("RUN_FINISHED", "PREDICTIONS_COMMITTED", "TERMINAL_CONSUMED"), "NO_OUTPUT_BEFORE_R16_AND_ALL_OUTPUT_ORIGIN_ATTESTED"),
    (6, "allocation_source_batch_partition_answer_authorities", "authenticate_membership_and_evaluator_inputs", ("allocation_authority_id", "allocation_authority_payload_sha256", "source_authority_id", "source_authority_payload_sha256", "batch_manifest_id", "batch_manifest_payload_sha256", "partition_manifest_id", "partition_manifest_payload_sha256", "answer_authority_id", "answer_authority_payload_sha256", "gate_authority_id", "gate_authority_payload_sha256"), (), ("R11_EXECUTION_FREEZE_COMMITMENTS_BOUND", "R13_FRESH_IDS_ALLOCATED_BY_VERIFIED_OVERLAY"), "PREREGISTERED", ("RAW_INPUT_COMMITTED", "REVEALED", "TERMINAL_CONSUMED"), "AUTHORITIES_FROZEN_BEFORE_IDS_INPUT_AND_REVEAL"),
    (7, "independent_720_240_generation_and_source_disjointness_receipt", "prove_post_validation_generation_source_public_disjointness_and_latent_case_independence", ("generator_manifest_id", "generator_manifest_payload_sha256", "generation_receipt_id", "generation_receipt_payload_sha256", "primary_720_manifest_id", "primary_720_manifest_payload_sha256", "challenge_240_manifest_id", "challenge_240_manifest_payload_sha256", "source_disjointness_receipt_id", "source_disjointness_receipt_payload_sha256", "latent_independence_receipt_id", "latent_independence_receipt_payload_sha256"), (6,), ("R06_GENERATOR_IMPLEMENTATION_QUALIFIED", "R08_STATISTICAL_VALIDATION_PASSED", "R14_ACTUAL_UNSEALED_960_COHORT_GENERATED"), "PREREGISTERED", ("RAW_INPUT_COMMITTED", "TERMINAL_CONSUMED"), "GENERATION_EXACTLY_ONCE_AFTER_VALIDATION_AND_FRESH_IDS"),
    (8, "audit_archive_formal_uuid_and_covert_receipts", "bind_formal_audit_outputs", ("audit_archive_id", "audit_archive_payload_sha256", "formal_receipt_id", "formal_receipt_payload_sha256", "uuid_receipt_id", "uuid_receipt_payload_sha256", "covert_receipt_id", "covert_receipt_payload_sha256", "audit_result_root_sha256"), (7,), ("R03_FORMAL_TRUSTED_WIRE_IMPLEMENTATION_QUALIFIED", "R04_FORMAL_COVERT_AUDITOR_IMPLEMENTATION_QUALIFIED", "R15_ACTUAL_CORPUS_COVERT_AUDIT_PASSED"), "RAW_INPUT_COMMITTED", ("PACKAGE_COMMITMENTS_FROZEN", "TERMINAL_CONSUMED"), "ATOMIC_AUDIT_NO_PARTIAL_PASS"),
    (9, "three_baseline_prediction_outputs", "support_embedding_nearest_prototype_frozen_llm_semantic_only_flat_learned_typed_comparisons", ("baseline_registration_ids_in_exact_order", "baseline_registration_payload_sha256s_in_exact_order", "baseline_prediction_archive_ids_in_exact_order", "baseline_prediction_archive_payload_sha256s_in_exact_order", "baseline_output_attestation_ids_in_exact_order", "baseline_output_attestation_payload_sha256s_in_exact_order", "all_1532_rows_present"), (2, 4, 6), ("R07_BASELINE_REGISTRATIONS_FROZEN", "R16_INFERENCE_ADMISSION_UNLOCKED"), "RUN_STARTED", ("PREDICTIONS_COMMITTED", "TERMINAL_CONSUMED"), "EXACTLY_THREE_BASELINES_ALL_1532_ROWS_NO_POST_REVEAL_OUTPUT"),
    (10, "durable_fail_closed_attempt_event_denominator", "define_fail_closed_rate_over_all_pre_score_attempt_events", ("denominator_event_ids_in_exact_order", "denominator_event_payload_sha256s_in_exact_order", "denominator_event_kinds_in_exact_order", "attempt_open_record_ids_in_exact_order", "attempt_pre_score_closed_record_ids_in_exact_order", "pre_score_append_receipt_ids_in_exact_order", "missing_output_record_ids_in_exact_order", "crash_record_ids_in_exact_order", "retry_record_ids_in_exact_order", "attempt_indices_in_exact_order", "pre_score_tail_event_id", "pre_score_tail_event_payload_sha256", "event_numerator", "event_denominator", "attempt_universe_head_id", "attempt_universe_head_payload_sha256", "denominator_finalization_id", "denominator_finalization_payload_sha256"), (2,), ("R10_DURABLE_CUSTODY_AND_LEDGER_VERIFIED",), "RUN_STARTED", ("SCORING_STARTED", "REPORTS_COMMITTED", "TERMINAL_CONSUMED"), "ALL_PRE_SCORE_ATTEMPT_EVENTS_FORM_ONE_MULTI_ROLE_SIGNED_CONTIGUOUS_PARTITIONED_CHAIN_AND_MUST_EXACT_MATCH_THE_OOB_PINNED_IMMUTABLE_ATTEMPT_UNIVERSE_HEAD_WITH_EXACT_CHAIN_TAIL_EVENT_TOTAL_ATTEMPT_TOPOLOGY_AND_FAILURES_MISSING_OUTPUTS_RETRIES_RETAINED_WITHOUT_ANY_SCIENTIFIC_TERMINAL_DISPOSITION"),
    (11, "preservation_496_legal_76_invalid_pair_results", "bind_original_variant_links_predictions_and_evaluator_results_for_496_legal_and_76_invalid_pairs", ("pair_manifest_id", "pair_manifest_payload_sha256", "legal_pair_count", "invalid_pair_count", "pre_reveal_prediction_archive_id", "pre_reveal_prediction_archive_payload_sha256", "post_reveal_result_id", "post_reveal_result_payload_sha256", "pair_consistency_root_sha256"), (2, 6, 8), ("R15_ACTUAL_CORPUS_COVERT_AUDIT_PASSED", "R16_INFERENCE_ADMISSION_UNLOCKED"), "PREDICTIONS_COMMITTED", ("SCORING_STARTED", "REPORTS_COMMITTED", "TERMINAL_CONSUMED"), "EXACT_496_LEGAL_AND_76_INVALID_ALL_RETAINED"),
    (12, "semantic_conflict_240_descriptor_audit_package", "support_separately_reported_challenge_audit_or_scoring_excluded_from_main_thresholds_and_tuning", ("descriptor_manifest_id", "descriptor_manifest_payload_sha256", "challenge_240_manifest_id", "challenge_240_manifest_payload_sha256", "prediction_archive_id", "prediction_archive_payload_sha256", "audit_report_id", "audit_report_payload_sha256", "main_threshold_exclusion_receipt_id", "main_threshold_exclusion_receipt_payload_sha256"), (6, 7, 8), ("R15_ACTUAL_CORPUS_COVERT_AUDIT_PASSED", "R16_INFERENCE_ADMISSION_UNLOCKED"), "PREDICTIONS_COMMITTED", ("SCORING_STARTED", "REPORTS_COMMITTED", "TERMINAL_CONSUMED"), "CHALLENGE_AUDIT_SEPARATE_NO_TUNING_NO_MAIN_THRESHOLD_CONTAMINATION"),
    (13, "scale_regret_oracle_loss_and_normalizer", "define_normalized_per_case_regret_point_max_0_05_and_bootstrap_upper_bound_0_08", ("oracle_manifest_id", "oracle_manifest_payload_sha256", "loss_manifest_id", "loss_manifest_payload_sha256", "normalizer_manifest_id", "normalizer_manifest_payload_sha256", "per_case_regret_report_id", "per_case_regret_report_payload_sha256", "point_mean_numerator", "point_mean_denominator"), (3, 5, 6), ("R08_STATISTICAL_VALIDATION_PASSED", "R16_INFERENCE_ADMISSION_UNLOCKED"), "SCORING_STARTED", ("REPORTS_COMMITTED", "TERMINAL_CONSUMED"), "EXACT_RATIONAL_REGRET_NO_FLOAT_THRESHOLD_DRIFT"),
    (14, "clustered_bootstrap_inputs_and_statistic", "define_10000_replicate_paired_latent_base_case_cluster_bootstrap_master_seed_411876909552964556_uint32_2611585425_original_and_preservation_variants_one_sided_95_percentile", ("cluster_manifest_id", "cluster_manifest_payload_sha256", "statistic_manifest_id", "statistic_manifest_payload_sha256", "replicate_count", "master_seed_decimal", "master_seed_uint32", "bootstrap_report_id", "bootstrap_report_payload_sha256", "recomputation_root_sha256"), (11, 13), ("R08_STATISTICAL_VALIDATION_PASSED", "R16_INFERENCE_ADMISSION_UNLOCKED"), "SCORING_STARTED", ("REPORTS_COMMITTED", "TERMINAL_CONSUMED"), "EXACT_10000_PAIRED_CLUSTER_REPLICATES_ONE_SIDED_95_PERCENTILE"),
    (15, "metric_wilson_overall_and_slice_gate_report", "materialize_metric_counts_frozen_thresholds_one_sided_95_percent_wilson_and_gate_results", ("scored_rows_root_sha256", "metric_counts_id", "metric_counts_payload_sha256", "wilson_report_id", "wilson_report_payload_sha256", "overall_gate_count", "slice_gate_count", "gate_result_ids_in_exact_order", "gate_result_payload_sha256s_in_exact_order", "all_gate_results_root_sha256"), (10, 11, 12, 13, 14), ("R08_STATISTICAL_VALIDATION_PASSED", "R16_INFERENCE_ADMISSION_UNLOCKED"), "SCORING_STARTED", ("REPORTS_COMMITTED", "TERMINAL_CONSUMED"), "EXACT_12_OVERALL_24_SLICE_GATES_AND_WILSON_THRESHOLDS"),
    (16, "attempt_rerun_permanent_record", "enforce_attempt_policy_and_permanent_attempt_history", ("attempt_record_ids_in_exact_order", "attempt_record_payload_sha256s_in_exact_order", "attempt_indices_in_exact_order", "execution_retry_action", "execution_retry_reason", "upload_retry_action", "upload_retry_reason", "any_raw_prediction_byte_produced", "raw_output_byte_count", "attempt_universe_head_id", "attempt_universe_head_payload_sha256", "permanent_history_root_sha256"), (10, 15), ("R10_DURABLE_CUSTODY_AND_LEDGER_VERIFIED",), "PREREGISTERED", ("REPORTS_COMMITTED", "TERMINAL_CONSUMED"), "ATTEMPT_0_AND_CONDITIONAL_1_ONLY_NO_ATTEMPT_2_AND_EXACT_OOB_PINNED_ATTEMPT_UNIVERSE_HEAD_MATCH"),
    (17, "formal_c1_consumed_sealed_report", "prove_all_overall_family_scale_gates_and_consumed_ledger_state", ("gate_evidence_ids_in_exact_order", "gate_evidence_payload_sha256s_in_exact_order", "all_gate_evidence_root_sha256", "terminal_stage_event_id", "terminal_stage_event_payload_sha256", "terminal_append_receipt_id", "terminal_append_receipt_payload_sha256", "denominator_finalization_id", "denominator_finalization_payload_sha256", "overall_gate_count", "slice_gate_count", "final_disposition", "sealed_report_root_sha256"), tuple(range(17)), tuple(g.gate_id for g in GATE_SPECS_IN_EXACT_ORDER_V1), "TERMINAL_CONSUMED", ("TERMINAL_CONSUMED",), "BACKWARD_BINDS_PACKAGES_0_THROUGH_16_AND_RECOMPUTES_16_READINESS_GATES_36_STATISTICAL_GATES_FINAL_ATTEMPT_TERMINAL_EVENT_RECEIPT_AND_DENOMINATOR_BEFORE_BUNDLE_INDEX_BINDS_0_THROUGH_17"),
)


def _package_signature_slots_v1(
    ordinal: int,
) -> tuple[_SignatureSlotSpecV1, ...]:
    if ordinal == 0:
        return (
            _signature_slot_v1(0, "STORE_RECEIPT", "COUNTERSIGN_STORE_IF_ABSENT_RECEIPT_BINDING", ordinal),
            _signature_slot_v1(1, "TIME_ATTESTOR", "COUNTERSIGN_DECLARED_TIME_TOKEN_BINDING", ordinal),
            _signature_slot_v1(2, "GOVERNANCE_ANCHOR", "SIGN_FINAL_GOVERNANCE_CHECKPOINT", ordinal),
            _signature_slot_v1(3, "CUSTODIAN", "COUNTERSIGN_FINAL_CUSTODIAN_CHECKPOINT", ordinal),
        )
    role_and_purpose = {
        1: ("CUSTODIAN", "SIGN_PREDICTION_REVEAL_SCORING_CHRONOLOGY"),
        2: ("STORE_RECEIPT", "SIGN_RAW_ARCHIVE_CAS_COMPLETE"),
        3: ("CUSTODIAN", "SIGN_CANONICAL_REPLAY_TRANSCRIPT"),
        4: ("GOVERNANCE_ANCHOR", "SIGN_FROZEN_RUNTIME_ENVIRONMENT"),
        5: ("RECOGNIZER", "SIGN_RECOGNIZER_OUTPUT_ATTESTATION"),
        6: ("GOVERNANCE_ANCHOR", "SIGN_ALLOCATION_AND_ANSWER_AUTHORITIES"),
        7: ("CUSTODIAN", "SIGN_INDEPENDENT_720_240_GENERATION_RECEIPT"),
        8: ("CUSTODIAN", "SIGN_FORMAL_UUID_COVERT_AUDIT_RECEIPTS"),
        9: ("CUSTODIAN", "SIGN_THREE_BASELINE_OUTPUTS"),
        10: ("CUSTODIAN", "SIGN_FAIL_CLOSED_ATTEMPT_DENOMINATOR"),
        11: ("SCORER_EVALUATOR", "SIGN_PRESERVATION_PAIR_RESULTS"),
        12: ("SCORER_EVALUATOR", "SIGN_SEMANTIC_CONFLICT_DESCRIPTOR_AUDIT"),
        13: ("SCORER_EVALUATOR", "SIGN_SCALE_REGRET_REPORT"),
        14: ("SCORER_EVALUATOR", "SIGN_CLUSTERED_BOOTSTRAP_REPORT"),
        15: ("SCORER_EVALUATOR", "SIGN_WILSON_GATE_REPORT"),
        16: ("GOVERNANCE_ANCHOR", "SIGN_PERMANENT_ATTEMPT_HISTORY"),
        17: ("GOVERNANCE_ANCHOR", "SIGN_FORMAL_C1_CONSUMED_SEALED_REPORT"),
    }[ordinal]
    slots = _single_slot_v1(role_and_purpose[0], role_and_purpose[1], ordinal)
    if ordinal in (10, 16):
        return slots + (
            _signature_slot_v1(
                1,
                "CUSTODIAN",
                {
                    10: "COUNTERSIGN_FAIL_CLOSED_ATTEMPT_DENOMINATOR",
                    16: "COUNTERSIGN_PERMANENT_ATTEMPT_HISTORY",
                    17: "COUNTERSIGN_FORMAL_C1_CONSUMED_SEALED_REPORT",
                }[ordinal],
                ordinal,
            ),
        )
    if ordinal == 17:
        return slots + (
            _signature_slot_v1(1, "CUSTODIAN", "COUNTERSIGN_FORMAL_C1_CONSUMED_SEALED_REPORT", ordinal),
            _signature_slot_v1(2, "SCORER_EVALUATOR", "COUNTERSIGN_FORMAL_C1_SCORE", ordinal),
        )
    return slots


_CONTENT_REFERENCE_FIELDS_V1: Final = (
    "artifact_kind",
    "artifact_id",
    "artifact_payload_sha256",
)
_SIGNER_BINDING_FIELDS_V1: Final = (
    "role",
    "purpose",
    "key_epoch",
    "key_id",
    "public_key_hex",
    "control_domain_id",
    "registry_id",
)
_PACKAGE0_SIGNED_STAGE_EVENT_ENVELOPE_SCHEMA_VERSION_V1: Final = (
    "hegel-machine-phase2b-actual-unsealed-960-signed-stage-event/1"
)
_PACKAGE0_CUSTODIAN_LEDGER_EVENT_BODY_SCHEMA_VERSION_V1: Final = (
    "hegel-machine-phase2b-actual-unsealed-960-custodian-ledger-event/1"
)
_SIGNED_STAGE_EVENT_ENVELOPE_FIELDS_V1: Final = (
    "schema_version",
    "body",
    "signed_stage_event_id",
    "signatures_in_exact_order",
)
_LEDGER_EVENT_FIELDS_V1: Final = (
    "schema_version",
    "catalog_id",
    "campaign_id",
    "attempt_id",
    "event_index",
    "stage",
    "predecessor_event_id_or_none",
    "payload_id",
    "payload_sha256",
    "signer_registry_id",
    "signer_key_id",
)
_STORE_RECEIPT_FIELDS_V1: Final = (
    "schema_version",
    "catalog_id",
    "campaign_id",
    "attempt_id",
    "transition_from",
    "transition_to",
    "mode",
    "stored_artifact_id",
    "stored_artifact_payload_sha256",
    "stored_artifact_size_bytes",
    "append_applied",
    "readback_verified",
)
_TIME_TOKEN_FIELDS_V1: Final = (
    "schema_version",
    "catalog_id",
    "campaign_id",
    "attempt_id",
    "previous_logical_time",
    "current_logical_time",
    "subject_id",
    "subject_payload_sha256",
)
_CHECKPOINT_FIELDS_V1: Final = (
    "schema_version",
    "catalog_id",
    "campaign_id",
    "attempt_id",
    "chain_id",
    "chain_root_sha256",
    "store_receipt_id",
    "store_receipt_payload_sha256",
    "time_token_id",
    "time_token_payload_sha256",
    "terminal_signed_ledger_event_id",
    "terminal_signed_ledger_event_payload_sha256",
)
_EXTERNAL_ANCHOR_FIELDS_V1: Final = (
    "schema_version",
    "catalog_id",
    "campaign_id",
    "attempt_id",
    "chain_id",
    "chain_root_sha256",
    "anchor_service_id",
    "anchor_locator",
    "governance_checkpoint_id",
    "governance_checkpoint_payload_sha256",
    "custodian_checkpoint_id",
    "custodian_checkpoint_payload_sha256",
    "previous_external_anchor_id_or_none",
    "anchor_sequence",
    "effective_state",
    "superseded",
    "revoked",
)
_REFERENCED_ARTIFACT_ENVELOPE_FIELDS_V1: Final = (
    "schema_version",
    "body",
    "artifact_id",
    "signatures_in_exact_order",
)
_SIGNER_REGISTRY_ENVELOPE_FIELDS_V1: Final = (
    "schema_version",
    "body",
    "signer_registry_id",
    "signatures_in_exact_order",
)
_SIGNER_REGISTRY_BODY_FIELDS_V1: Final = (
    "schema_version",
    "status",
    "claim_level",
    "catalog_id",
    "registry_epoch",
    "entries_in_exact_order",
    "entries_root_sha256",
    "authorized_signature_rows_in_exact_order",
    "authorized_signature_rows_root_sha256",
)
_SIGNER_REGISTRY_ENTRY_FIELDS_V1: Final = (
    "entry_ordinal",
    "role",
    "control_domain_id",
    "key_id",
    "key_epoch",
    "algorithm",
    "public_key_hex",
    "authorized_signature_purposes_in_exact_order",
)
_SIGNATURE_AUTHORIZATION_ROW_FIELDS_V1: Final = (
    "artifact_kind",
    "signature_ordinal",
    "role",
    "purpose",
    "key_epoch",
    "signature_domain_ascii",
)


def _referenced_signature_slot_v1(
    ordinal: int,
    role: str,
    purpose: str,
    artifact_kind: str,
) -> _SignatureSlotSpecV1:
    return _SignatureSlotSpecV1(
        ordinal,
        role,
        purpose,
        1,
        (
            "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PACKAGE/0/REFERENCED/"
            f"{artifact_kind.upper()}/SIGN/{purpose}/V1\x00"
        ),
    )


_PACKAGE0_REFERENCED_ARTIFACT_SPECS_IN_EXACT_ORDER_V1: Final = (
    _ReferencedArtifactSpecV1(
        "store_if_absent_receipt",
        "hegel-machine-phase2b-actual-unsealed-960-store-if-absent-receipt/1",
        _STORE_RECEIPT_FIELDS_V1,
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PACKAGE/0/STORE_RECEIPT/ID/V1\x00",
        "phase2b_actual_unsealed_960_store_if_absent_receipt_v1_",
        (
            _referenced_signature_slot_v1(
                0,
                "STORE_RECEIPT",
                "SIGN_STORE_IF_ABSENT_RECEIPT",
                "store_if_absent_receipt",
            ),
        ),
    ),
    _ReferencedArtifactSpecV1(
        "logical_time_token",
        "hegel-machine-phase2b-actual-unsealed-960-logical-time-token/1",
        _TIME_TOKEN_FIELDS_V1,
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PACKAGE/0/LOGICAL_TIME_TOKEN/ID/V1\x00",
        "phase2b_actual_unsealed_960_logical_time_token_v1_",
        (
            _referenced_signature_slot_v1(
                0,
                "TIME_ATTESTOR",
                "SIGN_LOGICAL_TIME_TOKEN",
                "logical_time_token",
            ),
        ),
    ),
    _ReferencedArtifactSpecV1(
        "governance_checkpoint",
        "hegel-machine-phase2b-actual-unsealed-960-governance-checkpoint/1",
        _CHECKPOINT_FIELDS_V1,
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PACKAGE/0/GOVERNANCE_CHECKPOINT/ID/V1\x00",
        "phase2b_actual_unsealed_960_governance_checkpoint_v1_",
        (
            _referenced_signature_slot_v1(
                0,
                "GOVERNANCE_ANCHOR",
                "SIGN_GOVERNANCE_CHECKPOINT",
                "governance_checkpoint",
            ),
        ),
    ),
    _ReferencedArtifactSpecV1(
        "custodian_checkpoint",
        "hegel-machine-phase2b-actual-unsealed-960-custodian-checkpoint/1",
        _CHECKPOINT_FIELDS_V1,
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PACKAGE/0/CUSTODIAN_CHECKPOINT/ID/V1\x00",
        "phase2b_actual_unsealed_960_custodian_checkpoint_v1_",
        (
            _referenced_signature_slot_v1(
                0,
                "CUSTODIAN",
                "SIGN_CUSTODIAN_CHECKPOINT",
                "custodian_checkpoint",
            ),
        ),
    ),
    _ReferencedArtifactSpecV1(
        "external_anchor",
        "hegel-machine-phase2b-actual-unsealed-960-external-anchor/1",
        _EXTERNAL_ANCHOR_FIELDS_V1,
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PACKAGE/0/EXTERNAL_ANCHOR/ID/V1\x00",
        "phase2b_actual_unsealed_960_external_anchor_v1_",
        (
            _referenced_signature_slot_v1(
                0,
                "STORE_RECEIPT",
                "SIGN_EXTERNAL_ANCHOR",
                "external_anchor",
            ),
            _referenced_signature_slot_v1(
                1,
                "GOVERNANCE_ANCHOR",
                "COUNTERSIGN_EXTERNAL_ANCHOR",
                "external_anchor",
            ),
        ),
    ),
)
_PACKAGE0_REFERENCED_SIGNATURE_CARDINALITY_V1: Final = (
    sum(
        len(spec.signature_slots_in_exact_order)
        for spec in _PACKAGE0_REFERENCED_ARTIFACT_SPECS_IN_EXACT_ORDER_V1
    )
)
_PACKAGE0_LEDGER_EVENT_SIGNATURE_CARDINALITY_V1: Final = len(
    STAGE_NAMES_IN_EXACT_ORDER_V1
)
_PACKAGE0_NESTED_SIGNATURE_CARDINALITY_V1: Final = (
    _PACKAGE0_LEDGER_EVENT_SIGNATURE_CARDINALITY_V1
    + _PACKAGE0_REFERENCED_SIGNATURE_CARDINALITY_V1
)
_PACKAGE10_EVENT_SIGNATURE_CARDINALITY_MINIMUM_V1: Final = 3
_PACKAGE10_EVENT_SIGNATURE_CARDINALITY_MAXIMUM_V1: Final = 9
_PACKAGE10_ATTEMPT_UNIVERSE_HEAD_SIGNATURE_CARDINALITY_V1: Final = 4
_PACKAGE10_FINALIZATION_SIGNATURE_CARDINALITY_V1: Final = 4
_PACKAGE10_NESTED_SIGNATURE_CARDINALITY_MINIMUM_V1: Final = (
    _PACKAGE10_EVENT_SIGNATURE_CARDINALITY_MINIMUM_V1
    + _PACKAGE10_ATTEMPT_UNIVERSE_HEAD_SIGNATURE_CARDINALITY_V1
    + _PACKAGE10_FINALIZATION_SIGNATURE_CARDINALITY_V1
)
_PACKAGE10_NESTED_SIGNATURE_CARDINALITY_MAXIMUM_V1: Final = (
    _PACKAGE10_EVENT_SIGNATURE_CARDINALITY_MAXIMUM_V1
    + _PACKAGE10_ATTEMPT_UNIVERSE_HEAD_SIGNATURE_CARDINALITY_V1
    + _PACKAGE10_FINALIZATION_SIGNATURE_CARDINALITY_V1
)
_ATTEMPT_RECORD_FIELDS_V1: Final = (
    "attempt_record_id",
    "attempt_index",
    "terminal_disposition",
    "execution_retry_action",
    "execution_retry_reason",
    "upload_retry_action",
    "upload_retry_reason",
    "any_raw_prediction_byte_produced",
    "raw_output_byte_count",
    "predecessor_attempt_record_id_or_none",
)
_GATE_RESULT_FIELDS_V1: Final = (
    "gate_result_id",
    "statistical_gate_id",
    "metric",
    "scope_axis",
    "scope_value",
    "correct_count",
    "total_count",
    "fixed_total_count_or_none",
    "point_threshold_numerator",
    "point_threshold_denominator",
    "wilson_threshold_numerator_or_none",
    "wilson_threshold_denominator_or_none",
    "wilson_lower_binary64_hex_or_none",
    "point_passed",
    "wilson_passed_or_none",
    "passed",
)
_DENOMINATOR_EVENT_ENVELOPE_FIELDS_V1: Final = (
    "schema_version",
    "body",
    "denominator_event_id",
    "signatures_in_exact_order",
)
_DENOMINATOR_EVENT_BODY_FIELDS_V1: Final = (
    "schema_version",
    "status",
    "claim_level",
    "catalog_id",
    "signer_registry_id",
    "signer_registry_payload_sha256",
    "signer_registry_entries_root_sha256",
    "campaign_id",
    "attempt_id",
    "event_index",
    "event_kind",
    "attempt_index",
    "stage",
    "predecessor_denominator_event_id_or_none",
    "execution_retry_action",
    "execution_retry_reason",
    "upload_retry_action",
    "upload_retry_reason",
    "any_raw_prediction_byte_produced",
    "raw_output_byte_count",
    "append_applied_or_none",
    "readback_verified_or_none",
    "fail_closed_satisfied",
    "first_failure_code",
)
_ATTEMPT_UNIVERSE_HEAD_ENVELOPE_FIELDS_V1: Final = (
    "schema_version",
    "body",
    "attempt_universe_head_id",
    "signatures_in_exact_order",
)
_ATTEMPT_UNIVERSE_HEAD_BODY_FIELDS_V1: Final = (
    "schema_version",
    "status",
    "claim_level",
    "catalog_id",
    "signer_registry_id",
    "signer_registry_payload_sha256",
    "signer_registry_entries_root_sha256",
    "campaign_id",
    "attempt_id",
    "head_sequence",
    "previous_attempt_universe_head_id_or_none",
    "attempt_indices_in_exact_order",
    "denominator_event_ids_in_exact_order",
    "denominator_event_payload_sha256s_in_exact_order",
    "denominator_event_kinds_in_exact_order",
    "pre_score_tail_event_id",
    "pre_score_tail_event_payload_sha256",
    "event_count",
    "event_universe_root_sha256",
    "sealed_stage",
    "complete",
)
_DENOMINATOR_FINALIZATION_FIELDS_V1: Final = (
    "denominator_finalization_id",
    "schema_version",
    "catalog_id",
    "signer_registry_id",
    "signer_registry_payload_sha256",
    "signer_registry_entries_root_sha256",
    "campaign_id",
    "attempt_id",
    "attempt_universe_head_id",
    "attempt_universe_head_payload_sha256",
    "denominator_event_ids_in_exact_order",
    "denominator_event_payload_sha256s_in_exact_order",
    "denominator_event_kinds_in_exact_order",
    "attempt_open_record_ids_in_exact_order",
    "attempt_pre_score_closed_record_ids_in_exact_order",
    "pre_score_append_receipt_ids_in_exact_order",
    "missing_output_record_ids_in_exact_order",
    "crash_record_ids_in_exact_order",
    "retry_record_ids_in_exact_order",
    "attempt_indices_in_exact_order",
    "pre_score_tail_event_id",
    "pre_score_tail_event_payload_sha256",
    "event_numerator",
    "event_denominator",
    "denominator_root_sha256",
    "complete",
    "signatures_in_exact_order",
)

_PREALLOCATION_LINEAGE_BINDING_RULES_IN_EXACT_ORDER_V1: Final = (
    (
        "RAW_CONTENT_ID_SHA256_AND_OOB_PIN",
        (
            "raw_preallocation_lineage_payload",
            "out_of_band_expected_preallocation_lineage_id_bytes",
            "out_of_band_expected_preallocation_lineage_payload_sha256_digest",
        ),
        "REQUIRE_EXACT_BUILTIN_BYTES_CANONICAL_ENVELOPE_RECOMPUTE_BODY_ONLY_CONTENT_ID_AND_REQUIRE_ID_AND_RAW_SHA256_EXACT_EQUAL_MANDATORY_OOB_PINS",
    ),
    (
        "FOUR_ROLE_SIGNATURE_AND_REGISTRY",
        (
            "raw_preallocation_lineage_payload",
            "raw_signer_registry_payload",
        ),
        "VERIFY_EXACT_GOVERNANCE_CUSTODIAN_STORE_RECEIPT_TIME_ATTESTOR_SIGNATURES_IN_FROZEN_ORDER_PURPOSE_EPOCH_DOMAIN_AGAINST_THE_OOB_PINNED_SIGNER_REGISTRY",
    ),
    (
        "CATALOG_AND_INSPECTED_ROOT_PROJECTION",
        (
            "catalog_id",
            "catalog_payload_sha256",
            "readiness_dag_id",
            "readiness_dag_payload_sha256",
            "progression_plan_payload_sha256",
            "gap_inventory_source_sha256",
            "statistical_validation_source_sha256",
        ),
        "ALL_CATALOG_AND_INSPECTED_DEPENDENCY_IDENTITIES_MUST_EXACT_EQUAL_THE_CURRENT_VERIFIER_INPUT_AND_FROZEN_CATALOG_CONSTANTS_WITHOUT_CLAIMING_RUNTIME_SOURCE_AUTHENTICATION",
    ),
    (
        "AUTHORITY_CURRENTNESS_CUSTODY_ROOTS_AND_STATE",
        (
            "authority_anchor_id",
            "authority_anchor_payload_sha256",
            "currentness_token_id",
            "currentness_token_payload_sha256",
            "custody_checkpoint_id",
            "custody_checkpoint_payload_sha256",
            "lineage_sequence",
            "previous_preallocation_lineage_id_or_none",
            "active",
            "revoked",
            "complete",
            "first_failure_code",
        ),
        "ALL_THREE_EXTERNAL_ROOT_PAIRS_MUST_BE_NONEMPTY_CONTENT_REFERENCES_LINEAGE_SEQUENCE_EXACT_ONE_PREVIOUS_EXACT_NULL_ACTIVE_TRUE_REVOKED_FALSE_COMPLETE_TRUE_FIRST_FAILURE_CODE_EXACT_NONE_BOOL_AS_INT_FORBIDDEN",
    ),
)

_GATE_SOURCE_ROOT_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1: Final = (
    "native_source_kinds_in_exact_order",
    "native_source_payload_sha256s_in_exact_order",
    "native_source_payload_byte_counts_in_exact_order",
)

_GATE_SOURCE_RECEIPT_BINDING_RULES_IN_EXACT_ORDER_V1: Final = (
    (
        "EXACT_GATE_SOURCE_SCHEMA_KIND_AND_FAIL_CLOSED_AVAILABILITY",
        (
            "descriptor.gate_source_schema_id",
            "descriptor.gate_source_receipt_artifact_kind",
            "gate_descriptor.gate_schema_id",
            "descriptor.verification_availability",
            "descriptor.verifier_dispatch_id",
            "raw_gate_source_receipt_payload",
        ),
        "DECODE_THE_EXACT_DESCRIPTOR_SPECIFIC_CANONICAL_GATE_SOURCE_RECEIPT_REJECT_UNKNOWN_CROSS_GATE_SCHEMA_KIND_DOMAIN_PREFIX_OR_FIELDSET_REQUIRE_BODY_GATE_SCHEMA_ID_EXACT_EQUAL_THE_MATCHED_GATE_DESCRIPTOR_AND_REQUIRE_CURRENT_V1_VERIFICATION_AVAILABILITY_EXACT_SCHEMA_FROZEN_NOT_VERIFIABLE_WITH_VERIFIER_SOURCE_SHA256_EXACT_NULL",
    ),
    (
        "CONTENT_ID_RAW_SHA256_AND_FOUR_ROLE_PROVENANCE_SIGNATURES",
        (
            "raw_gate_source_receipt_payload",
            "raw_signer_registry_payload",
        ),
        "RECOMPUTE_BODY_ONLY_CONTENT_ID_BIND_SHA256_OF_EXACT_RAW_RECEIPT_AND_VERIFY_EXACT_GOVERNANCE_CUSTODIAN_STORE_RECEIPT_TIME_ATTESTOR_SIGNATURES_IN_FROZEN_ORDER_PURPOSE_EPOCH_AND_GATE_SOURCE_DOMAIN_SIGNATURES_PROVE_PROVENANCE_ONLY_NOT_GATE_TRUTH",
    ),
    (
        "RAW_NATIVE_SOURCE_KIND_ORDER_CARDINALITY_SHA256_AND_BYTE_COUNT",
        (
            "descriptor.allowed_native_source_kind_orders_in_exact_order",
            "raw_gate_source_native_payloads_in_exact_order",
            "body.native_source_kinds_in_exact_order",
            "body.native_source_payload_sha256s_in_exact_order",
            "body.native_source_payload_byte_counts_in_exact_order",
        ),
        "REQUIRE_EXACT_BUILTIN_TUPLE_AND_EXACT_BUILTIN_BYTES_ITEMS_REQUIRE_KIND_ORDER_EQUAL_ONE_DESCRIPTOR_ALLOWED_ORDER_REQUIRE_ALL_FOUR_SEQUENCES_EQUAL_CARDINALITY_AND_RECOMPUTE_EACH_RAW_SHA256_AND_BYTE_COUNT_BOOL_AS_INT_FORBIDDEN_WITH_NO_MISSING_EXTRA_REORDERED_OR_OPAQUE_ID_SHA_ONLY_BUNDLE",
    ),
    (
        "SOURCE_ROOT_RECOMPUTATION",
        (
            "descriptor.source_root_domain_ascii",
            "descriptor.source_root_preimage_fields_in_exact_order",
            "body.native_source_kinds_in_exact_order",
            "body.native_source_payload_sha256s_in_exact_order",
            "body.native_source_payload_byte_counts_in_exact_order",
            "body.source_evidence_root_sha256",
        ),
        "RECOMPUTE_SHA256_COLON_LOWERHEX_FROM_THE_DESCRIPTOR_GATE_SPECIFIC_NUL_TERMINATED_DOMAIN_PLUS_RFC8785_EXACT_OBJECT_OF_KIND_SHA256_AND_BYTE_COUNT_ARRAYS_IN_FROZEN_FIELD_ORDER_AND_REJECT_ANY_PRODUCER_SELECTED_ROOT",
    ),
    (
        "PREDECESSOR_SOURCE_RECEIPT_AND_ALLOCATION_LINEAGE",
        (
            "descriptor.dependencies_in_exact_order",
            "raw_predecessor_gate_source_receipt_payloads_in_exact_order",
            "body.predecessor_gate_source_receipt_ids_in_exact_order",
            "body.predecessor_gate_source_receipt_payload_sha256s_in_exact_order",
            "body.lineage_scope",
            "body.campaign_id_or_none",
            "body.attempt_id_or_none",
        ),
        "REQUIRE_ONE_EARLIER_GATE_SOURCE_RECEIPT_PER_GATE_DEPENDENCY_IN_EXACT_ORDER_RECOMPUTE_EACH_ID_AND_RAW_SHA256_REQUIRE_SHARED_CATALOG_REGISTRY_PREALLOCATION_LINEAGE_AND_APPLY_R01_THROUGH_R12_NULL_R13_FIRST_CONCRETE_R14_THROUGH_R16_EQUAL_R13_ALLOCATION_CHRONOLOGY_WITH_NO_FUTURE_ID_BACKFILL",
    ),
    (
        "DISPATCH_IS_SCHEMA_ONLY_AND_CANNOT_ESTABLISH_GATE_TRUTH",
        (
            "body.verification_availability",
            "body.verifier_dispatch_id",
            "body.verifier_source_sha256_or_none",
            "body.first_failure_code",
        ),
        "CURRENT_V1_REQUIRES_SCHEMA_FROZEN_NOT_VERIFIABLE_A_DESCRIPTOR_FROZEN_FUTURE_DISPATCH_IDENTIFIER_NULL_VERIFIER_SOURCE_SHA256_AND_FIRST_FAILURE_CODE_GATE_VERIFIER_NOT_IMPLEMENTED_ANY_EXECUTABLE_OR_SUCCESSFUL_PROJECTION_REQUIRES_A_NEW_CATALOG_VERSION_WITH_A_PUBLIC_BYTES_ONLY_PRODUCER_FREE_DISPATCH_AND_OOB_SOURCE_PIN",
    ),
)

_GATE_SUPPORT_ARTIFACT_BINDING_RULES_IN_EXACT_ORDER_V1: Final = (
    (
        "EXACT_GATE_SUPPORT_SCHEMA_AND_KIND",
        (
            "descriptor.gate_support_schema_id",
            "descriptor.gate_support_artifact_kind",
            "raw_gate_support_artifact_payload",
        ),
        "DECODE_THE_EXACT_DESCRIPTOR_SPECIFIC_CANONICAL_ENVELOPE_AND_REJECT_ANY_UNKNOWN_CROSS_GATE_SCHEMA_KIND_DOMAIN_PREFIX_OR_FIELDSET",
    ),
    (
        "CONTENT_ID_RAW_SHA256_AND_FOUR_ROLE_SIGNATURES",
        (
            "raw_gate_support_artifact_payload",
            "raw_signer_registry_payload",
        ),
        "RECOMPUTE_BODY_ONLY_CONTENT_ID_BIND_SHA256_OF_EXACT_RAW_ENVELOPE_AND_VERIFY_EXACT_GOVERNANCE_CUSTODIAN_STORE_RECEIPT_TIME_ATTESTOR_SIGNATURES_IN_FROZEN_ORDER_PURPOSE_EPOCH_DOMAIN",
    ),
    (
        "RAW_GATE_SOURCE_RECEIPT_NATIVE_BYTES_AND_SOURCE_ROOT",
        (
            "raw_gate_source_receipt_payload",
            "raw_gate_source_native_payloads_in_exact_order",
            "descriptor.gate_source_schema_id",
            "descriptor.verification_availability",
            "descriptor.verifier_dispatch_id",
            "body.gate_source_receipt_id",
            "body.gate_source_receipt_payload_sha256",
            "body.source_evidence_root_sha256",
        ),
        "VERIFY_THE_DESCRIPTOR_SPECIFIC_RAW_GATE_SOURCE_RECEIPT_WITH_ITS_EXACT_RAW_NATIVE_BYTES_USING_THE_FULL_GATE_SOURCE_RECEIPT_BINDING_RULES_THEN_CROSS_BIND_RECEIPT_ID_RAW_SHA256_SOURCE_ROOT_AVAILABILITY_DISPATCH_AND_NULL_SOURCE_SHA256_INTO_OUTER_AND_INLINE_SUPPORT_OBJECTS",
    ),
    (
        "INLINE_OPERATIONAL_EVIDENCE_PAYLOAD",
        (
            "body.evidence_payload_format",
            "body.evidence_payload_hex",
            "body.evidence_payload_sha256",
            "body.evidence_payload_byte_count",
            "body.evidence_assertions_in_exact_order",
        ),
        "LOWERHEX_DECODE_THE_INLINE_EVIDENCE_BYTES_REQUIRE_BYTE_COUNT_AND_SHA256_EXACT_DECODE_AND_REENCODE_ONE_RFC8785_CANONICAL_OBJECT_WITH_THE_FROZEN_GATE_OPERATIONAL_EVIDENCE_KEYSET_AND_TYPES_CROSS_BIND_ALL_LINEAGE_GATE_SOURCE_RECEIPT_AVAILABILITY_DISPATCH_ASSERTION_SOURCE_ROOT_AND_FAIL_CLOSED_RESULT_FIELDS_AND_REJECT_EMPTY_OVERSIZE_NONCANONICAL_OR_UNKNOWN_FORMAT",
    ),
    (
        "OUTER_GATE_AND_PREALLOCATION_CROSS_BINDING",
        (
            "raw_preallocation_lineage_payload",
            "raw_gate_support_artifact_payload",
            "gate_evidence.body",
        ),
        "CATALOG_REGISTRY_PREALLOCATION_LINEAGE_SCOPE_NULL_OR_CONCRETE_CAMPAIGN_ATTEMPT_IDS_GATE_ORDINAL_GATE_ID_GATE_SCHEMA_ID_EVIDENCE_REQUIREMENT_SUPPORT_KIND_EVIDENCE_ROOT_AND_ALL_EXTERNAL_ANCHOR_PAIRS_MUST_EXACT_MATCH_THE_DESCRIPTOR_AND_OUTER_GATE_EVIDENCE",
    ),
    (
        "AUTHORITATIVE_SUCCESS_FORMULA",
        (
            "evidence_observed",
            "current",
            "non_revoked",
            "satisfied",
            "first_failure_code",
        ),
        "CURRENT_V1_REQUIRES_VERIFICATION_AVAILABILITY_SCHEMA_FROZEN_NOT_VERIFIABLE_EVIDENCE_OBSERVED_CURRENT_NON_REVOKED_AND_SATISFIED_ALL_EXACT_FALSE_FIRST_FAILURE_CODE_EXACT_GATE_VERIFIER_NOT_IMPLEMENTED_AND_OUTER_GATE_SATISFIED_EXACT_FALSE_FOUR_ROLE_SIGNATURES_PROVE_ONLY_PROVENANCE_EXECUTABLE_PRODUCER_FREE_SUCCESS_REQUIRES_A_NEW_VERSION_WITH_RAW_BYTES_DISPATCH_AND_PRIVATE_BRANDED_RESULT_REQUIREMENT",
    ),
)

_GATE_PREDECESSOR_BINDING_RULES_IN_EXACT_ORDER_V1: Final = (
    (
        "DEPENDENCY_CARDINALITY_AND_ORDER",
        (
            "descriptor.dependencies_in_exact_order",
            "body.predecessor_gate_evidence_ids_in_exact_order",
            "body.predecessor_gate_evidence_payload_sha256s_in_exact_order",
        ),
        "ALL_THREE_SEQUENCES_MUST_HAVE_EXACT_EQUAL_LENGTH_AND_INDEX_ORDER_WITH_R01_EXACTLY_ZERO",
    ),
    (
        "RAW_DEPENDENCY_EVIDENCE_REQUIRED",
        ("raw_predecessor_gate_evidence_payloads_in_exact_order",),
        "VERIFIER_MUST_RECEIVE_ONE_EXACT_BUILTIN_BYTES_CANONICAL_GATE_EVIDENCE_ENVELOPE_PER_DEPENDENCY",
    ),
    (
        "DEPENDENCY_CONTENT_ID_SIGNATURE_AND_SHA256",
        (
            "descriptor.dependencies_in_exact_order",
            "raw_predecessor_gate_evidence_payloads_in_exact_order",
            "body.predecessor_gate_evidence_ids_in_exact_order",
            "body.predecessor_gate_evidence_payload_sha256s_in_exact_order",
        ),
        "MATCH_EACH_DEPENDENCY_GATE_DESCRIPTOR_RECOMPUTE_ITS_EVIDENCE_CONTENT_ID_VERIFY_EXACT_SIGNATURE_AND_BIND_SHA256_OF_EXACT_RAW_BYTES",
    ),
    (
        "DEPENDENCY_BODY_CROSS_BINDING",
        (
            "raw_predecessor_gate_evidence_payloads_in_exact_order",
            "body.catalog_id",
            "body.preallocation_lineage_id",
            "body.preallocation_lineage_payload_sha256",
        ),
        "EACH_PREDECESSOR_GATE_ID_GATE_SCHEMA_ID_CATALOG_ID_PREALLOCATION_LINEAGE_ID_PREALLOCATION_LINEAGE_RAW_SHA256_AND_SATISFIED_TRUE_MUST_MATCH_THE_EXPECTED_DEPENDENCY",
    ),
    (
        "ALLOCATION_CHRONOLOGY_AND_NO_FUTURE_ID_BACKFILL",
        (
            "descriptor.ordinal",
            "descriptor.lineage_scope",
            "descriptor.campaign_attempt_id_policy",
            "raw_predecessor_gate_evidence_payloads_in_exact_order",
            "body.lineage_scope",
            "body.campaign_id_or_none",
            "body.attempt_id_or_none",
        ),
        "R01_THROUGH_R12_MUST_USE_PREALLOCATION_SCOPE_WITH_CAMPAIGN_AND_ATTEMPT_EXACT_NULL_R13_MUST_USE_ALLOCATION_TRANSITION_SCOPE_WITH_FIRST_CONCRETE_FRESH_IDS_WHILE_ITS_R12_PREDECESSOR_REMAINS_NULL_R14_THROUGH_R16_MUST_USE_ALLOCATED_SCOPE_WITH_CONCRETE_IDS_EXACT_EQUAL_THEIR_PREDECESSOR_AND_NO_EARLIER_GATE_MAY_BE_BACKFILLED_OR_RESIGNED_WITH_FUTURE_IDS",
    ),
    (
        "SELF_FORWARD_DUPLICATE_AND_PARTIAL_FORBIDDEN",
        (
            "descriptor.ordinal",
            "descriptor.dependencies_in_exact_order",
            "body.predecessor_gate_evidence_ids_in_exact_order",
        ),
        "DEPENDENCIES_MUST_BE_EARLIER_PAIRWISE_DISTINCT_AND_ANY_EMPTY_MISSING_EXTRA_REORDERED_SPLICED_OR_DIGEST_DRIFTED_REQUIRED_DEPENDENCY_REJECTS_ATOMICALLY",
    ),
)

_GATE_SUPPORTING_ARTIFACT_BINDING_RULES_IN_EXACT_ORDER_V1: Final = (
    (
        "PREALLOCATION_LINEAGE_ARTIFACT_AT_INDEX_ZERO",
        (
            "raw_preallocation_lineage_payload",
            "body.preallocation_lineage_id",
            "body.preallocation_lineage_payload_sha256",
            "body.supporting_artifact_kinds_in_exact_order[0]",
            "body.supporting_artifact_ids_in_exact_order[0]",
            "body.supporting_artifact_payload_sha256s_in_exact_order[0]",
            "raw_supporting_artifact_payloads_in_exact_order[0]",
        ),
        "INDEX_ZERO_KIND_MUST_EXACT_EQUAL_PREALLOCATION_AUTHORITY_LINEAGE_V1_ITS_RAW_BYTES_MUST_BE_BYTE_FOR_BYTE_EQUAL_THE_SEPARATELY_OOB_PINNED_RAW_PREALLOCATION_LINEAGE_PAYLOAD_ITS_ID_AND_RAW_SHA256_MUST_EXACT_EQUAL_THE_BODY_PAIR_AND_NO_LATER_GATE_MAY_REPLACE_OR_BACKFILL_THIS_LINEAGE",
    ),
    (
        "SUPPORTING_ARTIFACT_THREE_TUPLE",
        (
            "body.supporting_artifact_kinds_in_exact_order",
            "body.supporting_artifact_ids_in_exact_order",
            "body.supporting_artifact_payload_sha256s_in_exact_order",
        ),
        "ALL_THREE_SEQUENCES_AND_RAW_TUPLE_MUST_HAVE_EXACT_CARDINALITY_THREE_AND_RETAIN_INDEX_ORDER_PREALLOCATION_LINEAGE_THEN_DESCRIPTOR_SPECIFIC_GATE_SOURCE_RECEIPT_THEN_DESCRIPTOR_SPECIFIC_GATE_SUPPORT_ARTIFACT",
    ),
    (
        "RAW_SUPPORTING_ARTIFACTS_REQUIRED",
        ("raw_supporting_artifact_payloads_in_exact_order",),
        "VERIFIER_MUST_RECEIVE_EXACTLY_THREE_EXACT_BUILTIN_BYTES_PAYLOADS_VERIFY_INDEX_ZERO_WITH_THE_PREALLOCATION_LINEAGE_CONTRACT_INDEX_ONE_WITH_THE_DESCRIPTOR_SPECIFIC_GATE_SOURCE_RECEIPT_CONTRACT_AND_INDEX_TWO_WITH_THE_DESCRIPTOR_SPECIFIC_GATE_SUPPORT_CONTRACT_AND_BIND_EACH_DECLARED_CONTENT_ID_AND_EXACT_RAW_SHA256",
    ),
    (
        "GATE_SOURCE_RECEIPT_AT_INDEX_ONE",
        (
            "descriptor.gate_source_receipt_artifact_kind",
            "descriptor.gate_source_schema_id",
            "body.supporting_artifact_kinds_in_exact_order[1]",
            "body.supporting_artifact_ids_in_exact_order[1]",
            "body.supporting_artifact_payload_sha256s_in_exact_order[1]",
            "raw_supporting_artifact_payloads_in_exact_order[1]",
            "raw_gate_source_native_payloads_in_exact_order",
        ),
        "INDEX_ONE_KIND_MUST_EXACT_EQUAL_DESCRIPTOR_GATE_SOURCE_RECEIPT_ARTIFACT_KIND_RAW_BYTES_AND_NATIVE_PAYLOAD_TUPLE_MUST_VALIDATE_AGAINST_THE_DESCRIPTOR_GATE_SOURCE_SCHEMA_RECOMPUTED_ID_AND_RAW_SHA256_MUST_EQUAL_THE_OUTER_REFERENCES_AND_CURRENT_V1_AVAILABILITY_MUST_REMAIN_SCHEMA_FROZEN_NOT_VERIFIABLE",
    ),
    (
        "GATE_SUPPORT_ARTIFACT_AT_INDEX_TWO",
        (
            "descriptor.gate_support_artifact_kind",
            "descriptor.gate_support_schema_id",
            "body.supporting_artifact_kinds_in_exact_order[2]",
            "body.supporting_artifact_ids_in_exact_order[2]",
            "body.supporting_artifact_payload_sha256s_in_exact_order[2]",
            "raw_supporting_artifact_payloads_in_exact_order[2]",
        ),
        "INDEX_TWO_KIND_MUST_EXACT_EQUAL_DESCRIPTOR_GATE_SUPPORT_ARTIFACT_KIND_RAW_BYTES_MUST_VALIDATE_AGAINST_THE_DESCRIPTOR_GATE_SUPPORT_SCHEMA_RECOMPUTED_ID_AND_RAW_SHA256_MUST_EQUAL_THE_OUTER_REFERENCES_AND_CURRENT_V1_FAIL_CLOSED_FORMULA_MUST_HOLD_WITH_OUTER_GATE_SATISFIED_EXACT_FALSE",
    ),
)

_STAGE_PREDECESSOR_BINDING_RULES_IN_EXACT_ORDER_V1: Final = (
    (
        "PREDECESSOR_NODE_CARDINALITY_AND_ORDER",
        (
            "descriptor.predecessor_nodes_in_exact_order",
            "body.predecessor_node_names_in_exact_order",
            "body.predecessor_evidence_or_event_ids_in_exact_order",
            "body.predecessor_evidence_or_event_payload_sha256s_in_exact_order",
        ),
        "ALL_FOUR_SEQUENCES_MUST_HAVE_EXACT_EQUAL_LENGTH_AND_BODY_NODE_NAMES_MUST_EXACT_EQUAL_DESCRIPTOR_ORDER",
    ),
    (
        "RAW_PREDECESSOR_NODES_REQUIRED",
        ("raw_predecessor_evidence_or_event_payloads_in_exact_order",),
        "VERIFIER_MUST_RECEIVE_ONE_EXACT_BUILTIN_BYTES_CANONICAL_GATE_EVIDENCE_OR_STAGE_EVENT_ENVELOPE_PER_PREDECESSOR_NODE",
    ),
    (
        "NODE_KIND_CONTENT_ID_SIGNATURE_AND_SHA256",
        (
            "descriptor.predecessor_nodes_in_exact_order",
            "raw_predecessor_evidence_or_event_payloads_in_exact_order",
            "body.predecessor_evidence_or_event_ids_in_exact_order",
            "body.predecessor_evidence_or_event_payload_sha256s_in_exact_order",
        ),
        "SELECT_THE_EXACT_GATE_OR_STAGE_DESCRIPTOR_BY_NODE_NAME_RECOMPUTE_CONTENT_ID_VERIFY_SIGNATURE_AND_BIND_EXACT_RAW_SHA256_AT_THE_SAME_INDEX",
    ),
    (
        "PREDECESSOR_STATE_AND_LINEAGE",
        (
            "raw_predecessor_evidence_or_event_payloads_in_exact_order",
            "body.catalog_id",
            "body.campaign_id",
            "body.attempt_id",
        ),
        "GATE_PREDECESSOR_MUST_HAVE_SATISFIED_TRUE_STAGE_PREDECESSOR_MUST_HAVE_STAGE_OCCURRED_TRUE_AND_ALL_AVAILABLE_CATALOG_CAMPAIGN_ATTEMPT_LINEAGE_MUST_MATCH",
    ),
    (
        "SELF_FORWARD_DUPLICATE_AND_PARTIAL_FORBIDDEN",
        (
            "descriptor.ordinal",
            "descriptor.predecessor_nodes_in_exact_order",
            "body.predecessor_evidence_or_event_ids_in_exact_order",
        ),
        "PREDECESSOR_NODES_MUST_BE_EARLIER_PAIRWISE_DISTINCT_AND_ANY_MISSING_EXTRA_REORDERED_SPLICED_OR_DIGEST_DRIFTED_ITEM_REJECTS_ATOMICALLY",
    ),
)

_STAGE_SUBJECT_ARTIFACT_BINDING_RULES_IN_EXACT_ORDER_V1: Final = (
    (
        "SUBJECT_ARTIFACT_THREE_TUPLE",
        (
            "body.subject_artifact_kinds_in_exact_order",
            "body.subject_artifact_ids_in_exact_order",
            "body.subject_artifact_payload_sha256s_in_exact_order",
        ),
        "ALL_THREE_SEQUENCES_MUST_HAVE_EXACT_EQUAL_LENGTH_WITHIN_DESCRIPTOR_MINIMUM_AND_MAXIMUM_AND_RETAIN_INDEX_ORDER",
    ),
    (
        "RAW_SUBJECT_ARTIFACTS_REQUIRED",
        ("raw_subject_artifact_payloads_in_exact_order",),
        "VERIFIER_MUST_RECEIVE_ONE_EXACT_BUILTIN_BYTES_PAYLOAD_PER_SUBJECT_AND_BIND_EACH_DECLARED_CONTENT_ID_AND_EXACT_RAW_SHA256",
    ),
)

_STAGE_TERMINAL_DISPOSITION_BINDING_RULES_IN_EXACT_ORDER_V1: Final = (
    (
        "NONTERMINAL_STAGE_HAS_NULL_DISPOSITION",
        ("descriptor.terminal", "body.terminal_disposition_or_none"),
        "WHEN_DESCRIPTOR_TERMINAL_IS_FALSE_TERMINAL_DISPOSITION_MUST_BE_EXACT_JSON_NULL",
    ),
    (
        "TERMINAL_STAGE_HAS_POST_START_DISPOSITION",
        ("descriptor.terminal", "body.terminal_disposition_or_none"),
        "WHEN_DESCRIPTOR_TERMINAL_IS_TRUE_TERMINAL_DISPOSITION_MUST_BE_EXACTLY_ONE_OF_REJECTED_INVALIDATED_COMPLETE_FAIL_COMPLETE_PASS_AND_MUST_NEVER_BE_NOT_RUN_OR_BLOCKED",
    ),
    (
        "NOT_RUN_OR_BLOCKED_HAS_NO_ATTEMPT_OR_EVENT",
        (
            "terminal_disposition_policy",
            "body.attempt_id",
            "body.stage_occurred",
        ),
        "NOT_RUN_AND_BLOCKED_ARE_PREFLIGHT_OUTCOMES_WITH_NO_ATTEMPT_ID_NO_STAGE_EVENT_NO_PACKAGE17_AND_NO_BUNDLE_INSTANCE",
    ),
)

_SIGNER_REGISTRY_BINDING_RULES_IN_EXACT_ORDER_V1: Final = (
    (
        "EXACT_DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION",
        (
            "descriptor.detached_signature_schema_version",
            "signature.schema_version",
        ),
        "EVERY_DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_MUST_EXACT_EQUAL_"
        "HEGEL_MACHINE_PHASE2B_ACTUAL_UNSEALED_960_DETACHED_SIGNATURE_V1_"
        "BEFORE_ANY_ROLE_PURPOSE_EPOCH_DOMAIN_OR_ED25519_CHECK",
    ),
    (
        "RAW_SIGNER_REGISTRY_REQUIRED",
        ("raw_signer_registry_payload",),
        "VERIFIER_MUST_RECEIVE_ONE_EXACT_BUILTIN_BYTES_CANONICAL_SIGNER_REGISTRY_ENVELOPE_AND_MUST_NOT_ACCEPT_A_DECODED_MAPPING_OR_CALLER_KEY_LOOKUP",
    ),
    (
        "OUT_OF_BAND_REGISTRY_ID_AND_SHA256",
        (
            "raw_signer_registry_payload",
            "out_of_band_expected_signer_registry_id_bytes",
            "out_of_band_expected_signer_registry_payload_sha256_digest",
        ),
        "RECOMPUTE_THE_REGISTRY_CONTENT_ID_AND_EXACT_RAW_SHA256_AND_REQUIRE_BOTH_MANDATORY_OUT_OF_BAND_BYTE_PINS",
    ),
    (
        "OUT_OF_BAND_ROOT_SIGNATURES",
        (
            "raw_signer_registry_payload",
            "out_of_band_expected_governance_root_key_id_bytes",
            "out_of_band_expected_governance_root_public_key_bytes",
            "out_of_band_expected_custodian_root_key_id_bytes",
            "out_of_band_expected_custodian_root_public_key_bytes",
        ),
        "VERIFY_EXACT_GOVERNANCE_THEN_CUSTODIAN_REGISTRY_SIGNATURES_AGAINST_DISTINCT_OUT_OF_BAND_PINNED_ROOT_KEYS_NEVER_KEYS_DECLARED_ONLY_INSIDE_THE_REGISTRY",
    ),
    (
        "ENTRY_AND_AUTHORIZATION_ROOTS",
        (
            "body.entries_in_exact_order",
            "body.entries_root_sha256",
            "body.authorized_signature_rows_in_exact_order",
            "body.authorized_signature_rows_root_sha256",
        ),
        "RECOMPUTE_EACH_ROOT_AS_ITS_NUL_TERMINATED_DOMAIN_BYTES_CONCATENATED_WITH_RFC8785_OF_THE_EXACT_ORDERED_ARRAY_REQUIRE_EXACT_ROLE_ORDER_DISTINCT_KEY_IDS_DISTINCT_CONTROL_DOMAINS_ED25519_EPOCH_1_AND_THE_FULL_CATALOG_AUTHORIZATION_ROW_SET",
    ),
    (
        "OUT_OF_BAND_ENTRY_ROOT",
        (
            "body.entries_root_sha256",
            "out_of_band_expected_signer_registry_entries_root_sha256_digest",
        ),
        "REQUIRE_THE_RECOMPUTED_ENTRIES_ROOT_TO_EXACT_EQUAL_THE_MANDATORY_32_BYTE_OUT_OF_BAND_DIGEST_PIN",
    ),
    (
        "SIGNATURE_LOOKUP",
        (
            "signature.artifact_kind",
            "signature.signature_ordinal",
            "signature.role",
            "signature.purpose",
            "signature.key_epoch",
            "signature.signer_key_id",
            "signature.signature_domain_ascii",
            "body.entries_in_exact_order",
            "body.authorized_signature_rows_in_exact_order",
        ),
        "EVERY_GATE_STAGE_PACKAGE_NESTED_ARTIFACT_AND_BUNDLE_SIGNATURE_MUST_RESOLVE_EXACTLY_ONE_AUTHORIZATION_ROW_BY_ARTIFACT_KIND_SIGNATURE_ORDINAL_ROLE_PURPOSE_KEY_EPOCH_AND_DOMAIN_THEN_EXACTLY_ONE_REGISTRY_ENTRY_AND_SIGNER_KEY_ID_BEFORE_ED25519_VERIFICATION_BOOL_AS_INT_FORBIDDEN",
    ),
)

_PACKAGE0_STAGE_SUBJECT_BINDING_RULES_IN_EXACT_ORDER_V1: Final = (
    (
        "RAW_STAGE_SUBJECT_EVENTS_REQUIRED",
        ("raw_stage_subject_event_payloads_in_exact_order",),
        "VERIFIER_MUST_RECEIVE_EXACTLY_10_EXACT_BUILTIN_BYTES_CANONICAL_STAGE_EVENT_ENVELOPES_IN_THE_FROZEN_STAGE_ORDER",
    ),
    (
        "LEDGER_PAYLOAD_ID_AND_SHA256_BINDING",
        (
            "raw_ledger_event_payloads_in_exact_order",
            "raw_stage_subject_event_payloads_in_exact_order",
        ),
        "EACH_LEDGER_EVENT_BODY_PAYLOAD_ID_AND_PAYLOAD_SHA256_MUST_EQUAL_THE_RECOMPUTED_STAGE_EVENT_CONTENT_ID_AND_SHA256_OF_THE_EXACT_RAW_STAGE_EVENT_ENVELOPE_AT_THE_SAME_INDEX",
    ),
    (
        "STAGE_DESCRIPTOR_AND_SIGNATURE_BINDING",
        (
            "stage_schema_objects_in_exact_order",
            "raw_stage_subject_event_payloads_in_exact_order",
            "raw_signer_registry_payload",
        ),
        "VERIFY_EACH_STAGE_EVENT_AGAINST_ITS_EXACT_ORDINAL_SCHEMA_CONTENT_ID_SIGNATURE_SLOT_AND_SIGNER_REGISTRY_ENTRY_WITH_CATALOG_CAMPAIGN_ATTEMPT_STAGE_AND_EVENT_SEQUENCE_CROSS_BINDING",
    ),
    (
        "TERMINAL_SUBJECT_EXACT_P15_P16_NO_FORWARD_REFERENCE",
        (
            "raw_stage_subject_event_payloads_in_exact_order[9]",
            "raw_package15_envelope_payload",
            "raw_package16_envelope_payload",
        ),
        "TERMINAL_CONSUMED_STAGE_EVENT_MUST_BIND_EXACTLY_PACKAGE15_THEN_PACKAGE16_IDS_AND_RAW_SHA256S_AND_MUST_NOT_REFERENCE_PACKAGE0_PACKAGE17_OR_BUNDLE_INDEX",
    ),
)

_PACKAGE0_REFERENCED_ARTIFACT_BINDING_RULES_IN_EXACT_ORDER_V1: Final = (
    (
        "EXACT_DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION",
        (
            "artifact_contract.detached_signature_schema_version",
            "raw_referenced_artifact_payload.signatures_in_exact_order[*].schema_version",
        ),
        "EVERY_REFERENCED_ARTIFACT_DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_"
        "MUST_EXACT_EQUAL_THE_FROZEN_V1_LITERAL_BEFORE_SIGNATURE_VERIFICATION",
    ),
    (
        "RAW_REFERENCED_ARTIFACT_REQUIRED",
        ("raw_referenced_artifact_payload",),
        "VERIFIER_MUST_RECEIVE_ONE_EXACT_BUILTIN_BYTES_CANONICAL_ENVELOPE_FOR_THIS_REFERENCED_ARTIFACT",
    ),
    (
        "OUTER_ID_AND_RAW_SHA256_BINDING",
        (
            "raw_referenced_artifact_payload",
            "package0_body_reference_id",
            "package0_body_reference_payload_sha256",
        ),
        "RECOMPUTE_THE_EXACT_BODY_ONLY_CONTENT_ID_AND_SHA256_OF_THE_EXACT_RAW_ENVELOPE_AND_REQUIRE_BOTH_PACKAGE0_BODY_REFERENCES",
    ),
    (
        "EXACT_SIGNATURE_AND_REGISTRY_BINDING",
        (
            "raw_referenced_artifact_payload",
            "artifact_signature_slots_in_exact_order",
            "raw_signer_registry_payload",
        ),
        "VERIFY_EXACT_SIGNATURE_CARDINALITY_ROLE_PURPOSE_EPOCH_DOMAIN_PREIMAGE_AND_REGISTRY_LOOKUP_FOR_EVERY_REFERENCED_ARTIFACT_SIGNATURE",
    ),
    (
        "CATALOG_CAMPAIGN_ATTEMPT_CROSS_BINDING",
        (
            "raw_referenced_artifact_payload",
            "package0.body.catalog_id",
            "package0.body.campaign_id",
            "package0.body.attempt_id",
        ),
        "ALL_LINEAGE_FIELDS_PRESENT_IN_THE_ARTIFACT_BODY_MUST_EXACT_EQUAL_PACKAGE0_AND_ANY_MISSING_EXTRA_REORDERED_SPLICED_OR_DIGEST_DRIFTED_ARTIFACT_REJECTS_ATOMICALLY",
    ),
)

_PACKAGE0_REFERENCED_GRAPH_BINDING_RULES_IN_EXACT_ORDER_V1: Final = (
    (
        "TERMINAL_EVENT_STORE_RECEIPT",
        (
            "terminal_signed_stage_event",
            "store_if_absent_receipt",
        ),
        "STORE_RECEIPT_TRANSITION_MUST_BE_ABSENT_TO_PRESENT_MODE_IF_ABSENT_APPEND_APPLIED_AND_READBACK_VERIFIED_EXACT_TRUE_AND_STORED_ARTIFACT_MUST_BE_SIGNED_LEDGER_EVENT_INDEX_9",
    ),
    (
        "TERMINAL_EVENT_LOGICAL_TIME",
        (
            "terminal_signed_stage_event",
            "logical_time_token",
        ),
        "TIME_TOKEN_SUBJECT_MUST_BE_THE_TERMINAL_SIGNED_STAGE_EVENT_AND_CURRENT_LOGICAL_TIME_MUST_EQUAL_PREVIOUS_PLUS_ONE_USING_EXACT_INTEGERS_BOOL_FORBIDDEN_WITHOUT_CLAIMING_EXTERNAL_CURRENTNESS",
    ),
    (
        "DUAL_CHECKPOINT_PROJECTION",
        (
            "governance_checkpoint",
            "custodian_checkpoint",
            "store_if_absent_receipt",
            "logical_time_token",
            "terminal_signed_stage_event",
            "body.chain_root_sha256",
        ),
        "BOTH_CHECKPOINT_BODIES_MUST_EXACT_MATCH_CATALOG_CAMPAIGN_ATTEMPT_CHAIN_ROOT_RECEIPT_TIME_TOKEN_AND_TERMINAL_SIGNED_LEDGER_EVENT_REFERENCES",
    ),
    (
        "EXTERNAL_ANCHOR_PROJECTION",
        (
            "external_anchor",
            "governance_checkpoint",
            "custodian_checkpoint",
            "body.chain_root_sha256",
        ),
        "EXTERNAL_ANCHOR_MUST_BIND_AN_OPAQUE_ANCHOR_SERVICE_AND_LOCATOR_BOTH_CHECKPOINTS_AND_CHAIN_ROOT_WITH_PREVIOUS_ANCHOR_EXACT_NULL_ANCHOR_SEQUENCE_EXACT_ZERO_EFFECTIVE_STATE_ACTIVE_AND_SUPERSEDED_AND_REVOKED_EXACT_FALSE_WITHOUT_CLAIMING_CURRENTNESS_DURABILITY_NON_EQUIVOCATION_OR_GLOBAL_ANTI_FORK",
    ),
)

_FORMAL_C1_FINAL_DISPOSITION_BINDING_RULES_IN_EXACT_ORDER_V1: Final = (
    (
        "RAW_READINESS_GATE_EVIDENCE_EXACT_16",
        (
            "raw_gate_evidence_payloads_in_exact_order",
            "raw_preallocation_lineage_payload",
            "raw_gate_supporting_artifact_payload_tuples_in_exact_order",
            "raw_gate_source_native_payload_tuples_in_exact_order",
            "body.gate_evidence_ids_in_exact_order",
            "body.gate_evidence_payload_sha256s_in_exact_order",
            "body.all_gate_evidence_root_sha256",
            "body.campaign_id",
            "body.attempt_id",
        ),
        "VERIFY_EXACTLY_16_CANONICAL_GATE_EVIDENCE_ENVELOPES_IN_R01_THROUGH_R16_ORDER_WITH_THE_OOB_PINNED_PREALLOCATION_LINEAGE_EXACT_THREE_RAW_SUPPORT_ARTIFACTS_PER_GATE_AND_DESCRIPTOR_CARDINALITY_RAW_NATIVE_SOURCE_TUPLES_RECOMPUTE_EACH_SOURCE_RECEIPT_SCHEMA_ID_CONTENT_ID_RAW_SHA256_SOURCE_ROOT_PROVENANCE_SIGNATURE_SUPPORT_SIGNATURE_OUTER_GATE_SIGNATURE_AND_ORDERED_ROOT_CURRENT_CATALOG_V1_REQUIRES_ALL_16_VERIFICATION_AVAILABILITIES_SCHEMA_FROZEN_NOT_VERIFIABLE_AND_ALL_SUPPORT_AND_OUTER_SATISFIED_EXACT_FALSE_SO_NO_PACKAGE17_OR_BUNDLE_INSTANCE_CAN_BE_ACCEPTED_A_NEW_CATALOG_VERSION_WITH_16_EXECUTABLE_PRODUCER_FREE_DISPATCHES_IS_REQUIRED_BEFORE_THE_PROSPECTIVE_TRUE_DERIVATION_CAN_RUN_WITH_ONE_SHARED_LINEAGE_R01_THROUGH_R12_NULL_IDS_R13_FIRST_ALLOCATION_AND_R14_THROUGH_R16_EQUAL_R13_IDS",
    ),
    (
        "RAW_PACKAGE15_AND_36_GATE_RESULTS",
        (
            "raw_package15_envelope_payload",
            "raw_package15_gate_result_payloads_in_exact_order",
        ),
        "VERIFY_PACKAGE15_AND_ITS_EXACT_36_CANONICAL_GATE_RESULT_RECORDS_RECOMPUTE_EACH_ID_RAW_SHA256_PLAN_ROW_THRESHOLD_RESULT_AND_ORDERED_ROOT_WITH_12_OVERALL_THEN_24_SLICE_ROWS",
    ),
    (
        "RAW_PACKAGE16_AND_FINAL_ATTEMPT",
        (
            "raw_package16_envelope_payload",
            "raw_package16_attempt_record_payloads_in_exact_order",
        ),
        "VERIFY_PACKAGE16_AND_ITS_EXACT_ONE_OR_TWO_ATTEMPT_RECORDS_AND_SELECT_ONLY_THE_LAST_RECORD_AS_THE_FINAL_TERMINAL_ATTEMPT_WITH_NO_ATTEMPT_2",
    ),
    (
        "PACKAGE0_TERMINAL_STAGE_AND_APPEND_RECEIPT",
        (
            "raw_package0_envelope_payload",
            "raw_package0_stage_subject_event_payloads_in_exact_order[9]",
            "raw_package0_ledger_event_payloads_in_exact_order[9]",
            "raw_store_if_absent_receipt_payload",
            "body.terminal_stage_event_id",
            "body.terminal_stage_event_payload_sha256",
            "body.terminal_append_receipt_id",
            "body.terminal_append_receipt_payload_sha256",
        ),
        "VERIFY_PACKAGE0_THEN_REQUIRE_THE_P17_TERMINAL_STAGE_EVENT_PAIR_TO_EQUAL_AUTHORITATIVE_STAGE_SUBJECT_INDEX_9_AND_THE_APPEND_RECEIPT_PAIR_TO_EQUAL_PACKAGE0_STORE_RECEIPT_WHICH_STORES_SIGNED_LEDGER_EVENT_INDEX_9",
    ),
    (
        "PACKAGE10_DENOMINATOR_FINALIZATION",
        (
            "raw_package10_envelope_payload",
            "raw_package10_denominator_event_payloads_in_exact_order",
            "raw_attempt_universe_head_payload",
            "raw_package10_denominator_finalization_payload",
            "body.denominator_finalization_id",
            "body.denominator_finalization_payload_sha256",
        ),
        "VERIFY_PACKAGE10_ALL_RAW_SIGNED_DENOMINATOR_EVENTS_THE_OOB_PINNED_FOUR_ROLE_ATTEMPT_UNIVERSE_HEAD_AND_FOUR_ROLE_SIGNED_UNIVERSE_FINALIZATION_THEN_REQUIRE_THE_P17_DENOMINATOR_FINALIZATION_PAIR_TO_EXACT_EQUAL_ITS_VERIFIED_FINALIZATION_REFERENCE",
    ),
    (
        "PACKAGE10_PACKAGE16_ATTEMPT_UNIVERSE_BIJECTION",
        (
            "raw_package10_denominator_event_payloads_in_exact_order",
            "raw_attempt_universe_head_payload",
            "raw_package10_denominator_finalization_payload",
            "raw_package16_attempt_record_payloads_in_exact_order",
        ),
        "APPLY_ALL_PACKAGE16_DENOMINATOR_CROSS_BINDING_RULES_REQUIRE_EXACT_ATTEMPT_INDEX_BIJECTION_ONE_OPEN_PRE_SCORE_CLOSE_AND_PRE_SCORE_APPEND_PER_ATTEMPT_OPERATIONAL_OUTPUT_AND_RETRY_FIELD_PROJECTION_WITH_NO_OMITTED_EVENT_OR_ATTEMPT_AND_WITHOUT_PROJECTING_ANY_POST_SCORE_TERMINAL_DISPOSITION_BACKWARD_INTO_PACKAGE10",
    ),
    (
        "PRE_SCORE_DENOMINATOR_AND_POST_SCORE_TERMINAL_LEDGER_SEAL",
        (
            "raw_package0_stage_subject_event_payloads_in_exact_order[9]",
            "raw_package0_ledger_event_payloads_in_exact_order[9]",
            "raw_store_if_absent_receipt_payload",
            "raw_package10_denominator_finalization_payload",
            "raw_package16_envelope_payload",
        ),
        "PACKAGE10_PRE_SCORE_EVENT_UNIVERSE_MUST_ALREADY_BE_FOUR_ROLE_SEALED_BEFORE_PACKAGE15_SCORING_TERMINAL_STAGE_SUBJECT_MUST_THEN_EXACT_BIND_PACKAGE15_AND_PACKAGE16_AND_STORE_RECEIPT_MUST_IF_ABSENT_APPEND_AND_READBACK_THAT_POST_SCORE_TERMINAL_SIGNED_LEDGER_EVENT_BEFORE_PACKAGE17",
    ),
    (
        "COMPLETE_PASS_EQUIVALENCE",
        (
            "all_16_readiness_gates_satisfied",
            "all_36_statistical_gate_results_passed",
            "package10.finalization.complete",
            "package10.event_numerator",
            "package10.event_denominator",
            "package15.overall_fail_closed_rate",
            "package10_package16_attempt_universe_bijection_verified",
            "package16.final_attempt.terminal_disposition",
            "terminal_stage_event.body.terminal_disposition_or_none",
            "body.final_disposition",
        ),
        "FINAL_DISPOSITION_IS_COMPLETE_PASS_IF_AND_ONLY_IF_ALL_16_READINESS_GATES_ALL_36_STATISTICAL_GATES_SIGNED_PRE_SCORE_DENOMINATOR_PACKAGE15_FAIL_CLOSED_COUNT_EQUALITY_PACKAGE10_PACKAGE16_ATTEMPT_BIJECTION_AND_POST_SCORE_TERMINAL_LEDGER_SEAL_VERIFY_AND_PACKAGE16_FINAL_ATTEMPT_AND_TERMINAL_STAGE_EVENT_ARE_BOTH_COMPLETE_PASS",
    ),
    (
        "COMPLETE_FAIL_EQUIVALENCE",
        (
            "all_16_readiness_gates_satisfied",
            "any_of_36_statistical_gate_results_failed",
            "package10.finalization.complete",
            "package10_package16_attempt_universe_bijection_verified",
            "package16.final_attempt.terminal_disposition",
            "terminal_stage_event.body.terminal_disposition_or_none",
            "body.final_disposition",
        ),
        "FINAL_DISPOSITION_IS_COMPLETE_FAIL_IF_AND_ONLY_IF_ALL_16_READINESS_GATES_SIGNED_PRE_SCORE_DENOMINATOR_PACKAGE10_PACKAGE16_ATTEMPT_BIJECTION_AND_POST_SCORE_TERMINAL_LEDGER_SEAL_VERIFY_AT_LEAST_ONE_OF_36_STATISTICAL_GATES_IS_FALSE_AND_PACKAGE16_FINAL_ATTEMPT_AND_TERMINAL_STAGE_EVENT_ARE_BOTH_COMPLETE_FAIL",
    ),
    (
        "SEALED_REPORT_ROOT_REDERIVATION",
        (
            "body.predecessor_package_instance_ids_in_exact_order",
            "body.predecessor_package_instance_payload_sha256s_in_exact_order",
            "body.gate_evidence_ids_in_exact_order",
            "body.gate_evidence_payload_sha256s_in_exact_order",
            "body.terminal_stage_event_id",
            "body.terminal_stage_event_payload_sha256",
            "body.terminal_append_receipt_id",
            "body.terminal_append_receipt_payload_sha256",
            "body.denominator_finalization_id",
            "body.denominator_finalization_payload_sha256",
            "body.overall_gate_count",
            "body.slice_gate_count",
            "body.final_disposition",
            "body.sealed_report_root_sha256",
        ),
        "RECOMPUTE_THE_SEALED_REPORT_ROOT_FROM_THE_EXACT_RFC8785_OBJECT_CONTAINING_17_PREDECESSOR_PAIRS_16_GATE_EVIDENCE_PAIRS_TERMINAL_EVENT_RECEIPT_DENOMINATOR_COUNTS_AND_COMPLETED_DISPOSITION",
    ),
)

_BUNDLE_FINAL_DISPOSITION_BINDING_RULES_IN_EXACT_ORDER_V1: Final = (
    (
        "PACKAGE17_MUST_VERIFY_FIRST",
        (
            "raw_package_envelope_payloads_in_exact_order[17]",
            "raw_gate_evidence_payloads_in_exact_order",
            "raw_package_envelope_payloads_in_exact_order[0]",
            "raw_package0_ledger_event_payloads_in_exact_order",
            "raw_package0_stage_subject_event_payloads_in_exact_order",
            "raw_store_if_absent_receipt_payload",
            "raw_package_envelope_payloads_in_exact_order[10]",
            "raw_package10_denominator_event_payloads_in_exact_order",
            "raw_package10_denominator_finalization_payload",
            "raw_package_envelope_payloads_in_exact_order[15]",
            "raw_package15_gate_result_payloads_in_exact_order",
            "raw_package_envelope_payloads_in_exact_order[16]",
            "raw_package16_attempt_record_payloads_in_exact_order",
        ),
        "VERIFY_PACKAGE17_WITH_THE_FULL_FORMAL_C1_FINAL_DISPOSITION_BINDING_RULES_BEFORE_ACCEPTING_THE_BUNDLE_INDEX",
    ),
    (
        "BUNDLE_FINAL_DISPOSITION_AND_TERMINAL_PAIR",
        (
            "package17.body.final_disposition",
            "package17.body.terminal_stage_event_id",
            "package17.body.terminal_stage_event_payload_sha256",
            "bundle.body.final_disposition",
            "bundle.body.terminal_stage_event_id",
            "bundle.body.terminal_stage_event_payload_sha256",
        ),
        "BUNDLE_FINAL_DISPOSITION_AND_TERMINAL_STAGE_EVENT_PAIR_MUST_EXACT_EQUAL_VERIFIED_PACKAGE17_AND_DISPOSITION_MUST_BE_COMPLETE_FAIL_OR_COMPLETE_PASS_ONLY",
    ),
    (
        "NONCOMPLETED_OUTCOME_HAS_NO_ACCEPTED_BUNDLE",
        ("terminal_disposition_policy", "bundle.body.final_disposition"),
        "NOT_RUN_BLOCKED_REJECTED_OR_INVALIDATED_MUST_NOT_PRODUCE_AN_ACCEPTED_PACKAGE17_OR_BUNDLE_INDEX_INSTANCE",
    ),
)

_PACKAGE_OUTER_BINDING_RULES_IN_EXACT_ORDER_V1: Final = (
    (
        "EXACT_PACKAGE_SCHEMA_STATUS_CLAIM_AND_IDENTITY",
        (
            "descriptor.schema_version",
            "envelope.schema_version",
            "body.schema_version",
            "descriptor.ordinal",
            "body.package_ordinal",
            "descriptor.package_name",
            "body.package_name",
            "descriptor.body_required_literals.status",
            "body.status",
            "descriptor.body_required_literals.claim_level",
            "body.claim_level",
        ),
        "ENVELOPE_AND_BODY_SCHEMA_VERSION_MUST_EQUAL_THE_DESCRIPTOR_ORDINAL_AND_"
        "PACKAGE_NAME_MUST_EQUAL_THE_FROZEN_IDENTITY_STATUS_MUST_BE_PRESENTED_"
        "UNVERIFIED_AND_CLAIM_LEVEL_MUST_BE_PRESENTED_EXTERNAL_EVIDENCE_ONLY_"
        "NOT_VERIFIED_NOT_GATE_ADMISSION_NOT_AUTHORITY_NOT_CURRENTNESS_NOT_"
        "DURABILITY_NOT_C1",
    ),
    (
        "PACKAGE_NON_ELEVATION",
        (
            "descriptor.required_gate_ids_in_exact_order",
            "descriptor.verifier_success_disposition",
            "body.status",
            "body.claim_level",
            "verified_result",
        ),
        "PACKAGE_ACCEPTANCE_MAY_ONLY_RETURN_VERIFIED_PRESENTED_EXTERNAL_EVIDENCE_"
        "NOT_GATE_ADMISSION_AND_MUST_KEEP_GATE_AUTHORITY_CURRENTNESS_DURABILITY_"
        "NON_EQUIVOCATION_EXECUTION_AUTHORIZATION_AND_C1_EXACT_FALSE",
    ),
)

_PACKAGE_INSTANCE_LINEAGE_BINDING_RULES_IN_EXACT_ORDER_V1: Final = (
    (
        "EXACT_FOUR_TUPLE_CARDINALITY_AND_ORDER",
        (
            "descriptor.predecessor_package_ordinals_in_exact_order",
            "descriptor.predecessor_package_schema_ids_in_exact_order",
            "body.predecessor_package_instance_ids_in_exact_order",
            "body.predecessor_package_instance_payload_sha256s_in_exact_order",
        ),
        "ALL_FOUR_SEQUENCES_MUST_HAVE_EXACT_EQUAL_LENGTH_AND_INDEX_ORDER",
    ),
    (
        "SCHEMA_ID_FROM_PREDECESSOR_ORDINAL",
        (
            "descriptor.predecessor_package_ordinals_in_exact_order",
            "descriptor.predecessor_package_schema_ids_in_exact_order",
        ),
        "EACH_SCHEMA_ID_MUST_EXACT_EQUAL_CATALOG_PACKAGE_SCHEMA_ID_AT_THE_SAME_PREDECESSOR_ORDINAL",
    ),
    (
        "RAW_PREDECESSOR_REQUIRED",
        (
            "raw_predecessor_envelope_payloads_in_exact_order",
            "body.predecessor_package_instance_ids_in_exact_order",
        ),
        "VERIFIER_MUST_RECEIVE_ONE_EXACT_BUILTIN_BYTES_CANONICAL_PREDECESSOR_ENVELOPE_PER_INDEX",
    ),
    (
        "INSTANCE_CONTENT_ID_REDERIVATION",
        (
            "raw_predecessor_envelope_payloads_in_exact_order",
            "body.predecessor_package_instance_ids_in_exact_order",
        ),
        "RECOMPUTE_EACH_INSTANCE_CONTENT_ID_FROM_ITS_DESCRIPTOR_DOMAIN_PREFIX_AND_EXACT_BODY_AND_REQUIRE_INDEX_EQUALITY",
    ),
    (
        "RAW_PAYLOAD_SHA256_BINDING",
        (
            "raw_predecessor_envelope_payloads_in_exact_order",
            "body.predecessor_package_instance_payload_sha256s_in_exact_order",
        ),
        "SHA256_MUST_COVER_THE_EXACT_RAW_PREDECESSOR_ENVELOPE_BYTES_AT_THE_SAME_INDEX",
    ),
    (
        "PREDECESSOR_SIGNATURE_VERIFICATION",
        (
            "descriptor.predecessor_package_schema_ids_in_exact_order",
            "raw_predecessor_envelope_payloads_in_exact_order",
        ),
        "VERIFY_EACH_PREDECESSOR_EXACT_SIGNATURE_CARDINALITY_ROLE_PURPOSE_KEY_EPOCH_DOMAIN_AND_PREIMAGE_AGAINST_ITS_MATCHED_SCHEMA_DESCRIPTOR",
    ),
    (
        "PREDECESSOR_BODY_CROSS_BINDING",
        (
            "descriptor.predecessor_package_ordinals_in_exact_order",
            "raw_predecessor_envelope_payloads_in_exact_order",
            "body.catalog_id",
            "body.campaign_id",
            "body.attempt_id",
        ),
        "DECODED_PREDECESSOR_PACKAGE_ORDINAL_PACKAGE_NAME_SCHEMA_VERSION_CATALOG_ID_CAMPAIGN_ID_AND_ATTEMPT_ID_MUST_MATCH_THE_DESCRIPTOR_AND_CURRENT_BODY",
    ),
    (
        "SELF_FORWARD_AND_DUPLICATE_INSTANCE_FORBIDDEN",
        (
            "descriptor.ordinal",
            "descriptor.predecessor_package_ordinals_in_exact_order",
            "body.predecessor_package_instance_ids_in_exact_order",
        ),
        "EVERY_PREDECESSOR_ORDINAL_MUST_BE_STRICTLY_LESS_THAN_CURRENT_ORDINAL_AND_INSTANCE_IDS_MUST_BE_PAIRWISE_DISTINCT",
    ),
    (
        "ATOMIC_LINEAGE_REJECTION",
        (
            "raw_predecessor_envelope_payloads_in_exact_order",
            "body.predecessor_package_instance_ids_in_exact_order",
            "body.predecessor_package_instance_payload_sha256s_in_exact_order",
        ),
        "MISSING_EXTRA_EMPTY_REORDERED_SPLICED_OR_DIGEST_DRIFTED_LINEAGE_MUST_FAIL_BEFORE_PACKAGE_ACCEPTANCE",
    ),
)

_BUNDLE_INSTANCE_LINEAGE_BINDING_RULES_IN_EXACT_ORDER_V1: Final = (
    (
        "EXACT_FOUR_TUPLE_CARDINALITY_AND_ORDER",
        (
            "descriptor.predecessor_package_ordinals_in_exact_order",
            "body.package_schema_ids_in_exact_order",
            "body.package_instance_ids_in_exact_order",
            "body.package_instance_payload_sha256s_in_exact_order",
        ),
        "ALL_FOUR_SEQUENCES_MUST_HAVE_EXACT_CARDINALITY_18_AND_INDEX_ORDER_0_THROUGH_17",
    ),
    (
        "SCHEMA_ID_FROM_PACKAGE_ORDINAL",
        (
            "descriptor.predecessor_package_ordinals_in_exact_order",
            "body.package_schema_ids_in_exact_order",
        ),
        "EACH_SCHEMA_ID_MUST_EXACT_EQUAL_THE_CATALOG_PACKAGE_SCHEMA_ID_AT_THE_SAME_ORDINAL",
    ),
    (
        "RAW_PACKAGE_REQUIRED",
        (
            "raw_package_envelope_payloads_in_exact_order",
            "body.package_instance_ids_in_exact_order",
        ),
        "VERIFIER_MUST_RECEIVE_EXACTLY_18_EXACT_BUILTIN_BYTES_CANONICAL_PACKAGE_ENVELOPES",
    ),
    (
        "INSTANCE_ID_AND_RAW_SHA256",
        (
            "raw_package_envelope_payloads_in_exact_order",
            "body.package_instance_ids_in_exact_order",
            "body.package_instance_payload_sha256s_in_exact_order",
        ),
        "RECOMPUTE_EACH_PACKAGE_INSTANCE_ID_FROM_ITS_OWN_DESCRIPTOR_AND_BIND_SHA256_OF_THE_EXACT_RAW_ENVELOPE_AT_THE_SAME_INDEX",
    ),
    (
        "PACKAGE_SIGNATURE_VERIFICATION",
        (
            "descriptor.predecessor_package_schema_ids_in_exact_order",
            "raw_package_envelope_payloads_in_exact_order",
        ),
        "VERIFY_EACH_PACKAGE_EXACT_SIGNATURE_CARDINALITY_ROLE_PURPOSE_KEY_EPOCH_DOMAIN_SIGNER_REGISTRY_AND_PREIMAGE_AGAINST_ITS_ORDINAL_DESCRIPTOR",
    ),
    (
        "PACKAGE_BODY_CROSS_BINDING",
        (
            "raw_package_envelope_payloads_in_exact_order",
            "body.catalog_id",
            "body.campaign_id",
            "body.attempt_id",
        ),
        "DECODED_PACKAGE_ORDINAL_PACKAGE_NAME_SCHEMA_VERSION_CATALOG_ID_CAMPAIGN_ID_AND_ATTEMPT_ID_MUST_MATCH_INDEX_DESCRIPTOR_AND_BUNDLE_BODY",
    ),
    (
        "PAIRWISE_DISTINCT_AND_ATOMIC",
        (
            "body.package_instance_ids_in_exact_order",
            "body.package_instance_payload_sha256s_in_exact_order",
        ),
        "INSTANCE_IDS_MUST_BE_PAIRWISE_DISTINCT_AND_ANY_MISSING_EXTRA_REORDERED_SPLICED_OR_DIGEST_DRIFTED_ITEM_REJECTS_THE_ENTIRE_BUNDLE",
    ),
)

_LEDGER_EVENT_ID_DOMAIN_ASCII_V1: Final = (
    "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/SIGNED_STAGE_EVENT/ID/V1\x00"
)
_LEDGER_EVENT_ID_PREFIX_V1: Final = (
    "phase2b_actual_unsealed_960_signed_stage_event_v1_"
)
_LEDGER_EVENT_ID_FORMULA_V1: Final = (
    "signed_stage_event_id_prefix || lowerhex(sha256(signed_stage_event_id_domain_ascii_bytes || "
    "rfc8785({schema_version, body, signatures_in_exact_order})))"
)
_LEDGER_EVENT_ID_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1: Final = (
    "schema_version",
    "body",
    "signatures_in_exact_order",
)
_LEDGER_EVENT_SIGNATURE_PREIMAGE_SCHEMA_VERSION_V1: Final = (
    "hegel-machine-phase2b-actual-unsealed-960-custodian-ledger-event-"
    "signature-preimage/1"
)
_LEDGER_EVENT_SIGNATURE_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1: Final = (
    "schema_version",
    "artifact_kind",
    "artifact_schema_version",
    "algorithm",
    "body",
    "signature_ordinal",
    "role",
    "purpose",
    "key_epoch",
    "signer_key_id",
)
_LEDGER_EVENT_SIGNATURE_FORMULA_V1: Final = (
    "Ed25519.sign(custodian_private_key, event_signature_domain_ascii_bytes || "
    "rfc8785(event_signature_preimage_object_without_event_id_or_signature_hex))"
)
_LEDGER_EVENT_CHAIN_ROOT_DOMAIN_ASCII_V1: Final = (
    "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/SIGNED_STAGE_EVENT_CHAIN/ROOT/V1\x00"
)
_LEDGER_EVENT_CHAIN_ROOT_FORMULA_V1: Final = (
    "sha256_colon_lowerhex(sha256(chain_root_domain_ascii_bytes || rfc8785("
    "ordered_10_pairs_of_signed_stage_event_id_and_exact_raw_payload_sha256)))"
)


def _ledger_event_signature_slots_v1() -> tuple[_SignatureSlotSpecV1, ...]:
    return tuple(
        _SignatureSlotSpecV1(
            0,
            "CUSTODIAN",
            f"SIGN_{stage}_LEDGER_EVENT",
            1,
            (
                "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/SIGNED_STAGE_EVENT/"
                f"SIGN/{stage}/CUSTODIAN/V1\x00"
            ),
        )
        for stage in STAGE_NAMES_IN_EXACT_ORDER_V1
    )


_LEDGER_EVENT_SIGNATURE_SLOTS_IN_EXACT_ORDER_V1: Final = (
    _ledger_event_signature_slots_v1()
)
_LEDGER_EVENT_BINDING_RULES_IN_EXACT_ORDER_V1: Final = (
    (
        "EXACT_LEDGER_ENVELOPE_BODY_AND_SIGNATURE_RECORD_SCHEMA_VERSIONS",
        (
            "descriptor.record_schema_version",
            "descriptor.event_body_schema_version",
            "descriptor.detached_signature_schema_version",
            "signed_stage_event_envelope.schema_version",
            "signed_stage_event_envelope.body.schema_version",
            "signed_stage_event_envelope.signatures_in_exact_order[0].schema_version",
        ),
        "ENVELOPE_SCHEMA_VERSION_MUST_EXACT_EQUAL_DESCRIPTOR_RECORD_SCHEMA_VERSION_"
        "BODY_SCHEMA_VERSION_MUST_EXACT_EQUAL_DESCRIPTOR_EVENT_BODY_SCHEMA_VERSION_"
        "AND_THE_DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_MUST_EXACT_EQUAL_THE_"
        "FROZEN_V1_LITERAL",
    ),
    (
        "LEDGER_EVENT_CONTENT_ID",
        ("signed_stage_event_id",) + _LEDGER_EVENT_ID_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1,
        "RECOMPUTE_SIGNED_EVENT_ID_FROM_SCHEMA_VERSION_BODY_AND_EXACT_ONE_DETACHED_SIGNATURE_WITH_SIGNED_STAGE_EVENT_ID_EXCLUDED_FROM_THE_PREIMAGE",
    ),
    (
        "RAW_LEDGER_EVENT_PAYLOADS_REQUIRED",
        ("raw_ledger_event_payloads_in_exact_order",),
        "VERIFIER_MUST_RECEIVE_EXACTLY_10_EXACT_BUILTIN_BYTES_CANONICAL_SIGNED_STAGE_EVENT_ENVELOPES",
    ),
    (
        "OUTER_LEDGER_EVENT_THREE_TUPLE_BINDING",
        (
            "raw_ledger_event_payloads_in_exact_order",
            "body.signed_stage_event_ids_in_exact_order",
            "body.signed_stage_event_payload_sha256s_in_exact_order",
        ),
        "ALL_THREE_SEQUENCES_MUST_HAVE_EXACT_CARDINALITY_10_IDENTICAL_STAGE_ORDER_AND_PAIRWISE_DISTINCT_EVENT_IDS",
    ),
    (
        "LEDGER_EVENT_ID_SHA_AND_BODY_CROSS_BINDING",
        (
            "raw_ledger_event_payloads_in_exact_order",
            "body.signed_stage_event_ids_in_exact_order",
            "body.signed_stage_event_payload_sha256s_in_exact_order",
            "body.catalog_id",
            "body.campaign_id",
            "body.attempt_id",
        ),
        "FOR_EACH_RAW_EVENT_RECOMPUTE_CONTENT_ID_AND_SHA256_OF_EXACT_RAW_BYTES_AND_REQUIRE_CATALOG_CAMPAIGN_ATTEMPT_MATCH_THE_PACKAGE_BODY",
    ),
    (
        "LEDGER_EVENT_INDEX_STAGE_AND_PREDECESSOR_CHAIN",
        (
            "raw_ledger_event_payloads_in_exact_order",
            "stage_names_in_exact_order",
        ),
        "EVENT_INDICES_MUST_BE_EXACT_0_THROUGH_9_STAGES_MUST_EQUAL_THE_10_FROZEN_STAGE_NAMES_EVENT0_PREDECESSOR_MUST_BE_JSON_NULL_AND_EACH_LATER_PREDECESSOR_MUST_EQUAL_THE_RECOMPUTED_PRIOR_EVENT_ID",
    ),
    (
        "LEDGER_EVENT_LEVEL_SIGNATURE_BINDING",
        (
            "event_signature_slots_in_exact_order",
            "raw_ledger_event_payloads_in_exact_order",
        ),
        "EACH_EVENT_ENVELOPE_MUST_HAVE_EXACTLY_ONE_CUSTODIAN_SIGNATURE_WITH_STAGE_SPECIFIC_PURPOSE_DOMAIN_EPOCH_AND_PREIMAGE_EXCLUDING_EVENT_ID_AND_SIGNATURE_HEX_AND_BODY_SIGNER_KEY_ID_MUST_EQUAL_SIGNATURE_SIGNER_KEY_ID",
    ),
    (
        "LEDGER_EVENT_CHAIN_ROOT",
        (
            "body.signed_stage_event_ids_in_exact_order",
            "body.signed_stage_event_payload_sha256s_in_exact_order",
            "body.chain_root_sha256",
        ),
        "RECOMPUTE_CHAIN_ROOT_FROM_THE_LENGTH_ORDERED_10_ID_AND_EXACT_RAW_PAYLOAD_SHA256_PAIRS_UNDER_THE_NUL_TERMINATED_CHAIN_ROOT_DOMAIN",
    ),
)

_ATTEMPT_RECORD_ID_DOMAIN_ASCII_V1: Final = (
    "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/ATTEMPT_RECORD/ID/V1\x00"
)
_ATTEMPT_RECORD_ID_PREFIX_V1: Final = (
    "phase2b_actual_unsealed_960_attempt_record_v1_"
)
_ATTEMPT_RECORD_ID_FORMULA_V1: Final = (
    "attempt_record_id_prefix || lowerhex(sha256(attempt_record_id_domain_ascii_bytes || "
    "rfc8785(exact_record_object_with_attempt_record_id_excluded)))"
)
_ATTEMPT_RECORD_ID_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1: Final = tuple(
    field_name
    for field_name in _ATTEMPT_RECORD_FIELDS_V1
    if field_name != "attempt_record_id"
)
_ATTEMPT_RECORD_BINDING_RULES_IN_EXACT_ORDER_V1: Final = (
    (
        "ATTEMPT_RECORD_CONTENT_ID",
        ("attempt_record_id",) + _ATTEMPT_RECORD_ID_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1,
        "RECOMPUTE_RECORD_ID_AS_PREFIX_PLUS_SHA256_OF_NUL_TERMINATED_DOMAIN_PLUS_RFC8785_OF_EXACT_PREIMAGE_FIELDS_EXCLUDING_ATTEMPT_RECORD_ID",
    ),
    (
        "ATTEMPT_INDEX_SEQUENCE",
        ("attempt_index",),
        "PACKAGE16_RECORD_SEQUENCE_MUST_BE_EXACTLY_INDEX_0_OR_INDICES_0_THEN_1_BOOL_FORBIDDEN_AND_ATTEMPT_2_FORBIDDEN",
    ),
    (
        "ATTEMPT_ZERO_PREDECESSOR",
        ("attempt_index", "predecessor_attempt_record_id_or_none"),
        "ATTEMPT_INDEX_0_MUST_HAVE_EXACT_JSON_NULL_PREDECESSOR",
    ),
    (
        "ATTEMPT_ONE_PREDECESSOR",
        ("attempt_index", "predecessor_attempt_record_id_or_none"),
        "ATTEMPT_INDEX_1_MUST_EXACT_EQUAL_THE_RECOMPUTED_ATTEMPT_RECORD_ID_OF_INDEX_0",
    ),
    (
        "ATTEMPT_ONE_REEXECUTE_ELIGIBILITY",
        (
            "attempt_records_in_exact_order[0].execution_retry_action",
            "attempt_records_in_exact_order[0].execution_retry_reason",
            "attempt_records_in_exact_order[0].any_raw_prediction_byte_produced",
            "attempt_records_in_exact_order[0].raw_output_byte_count",
        ),
        "THE_EXISTENCE_OF_ATTEMPT_INDEX_1_REQUIRES_ATTEMPT_INDEX_0_TO_DECLARE_REEXECUTE_ONE_OF_FOUR_FROZEN_PRE_OUTPUT_REASONS_FALSE_AND_ZERO",
    ),
    (
        "ATTEMPT_REEXECUTE_BIDIRECTIONAL_CARDINALITY",
        (
            "attempt_records_in_exact_order",
            "attempt_records_in_exact_order[0].execution_retry_action",
            "attempt_records_in_exact_order[-1].execution_retry_action",
        ),
        "ATTEMPT_RECORD_INDICES_MUST_BE_EXACTLY_ZERO_THEN_ONE_IF_AND_ONLY_IF_ATTEMPT_ZERO_EXECUTION_RETRY_ACTION_IS_REEXECUTE_OTHERWISE_THE_SEQUENCE_MUST_BE_EXACTLY_ZERO",
    ),
    (
        "FINAL_ATTEMPT_HAS_NO_OUTSTANDING_REEXECUTE",
        (
            "attempt_records_in_exact_order[-1].execution_retry_action",
            "attempt_records_in_exact_order[-1].execution_retry_reason",
        ),
        "THE_FINAL_ATTEMPT_RECORD_MUST_HAVE_EXECUTION_RETRY_ACTION_AND_REASON_EXACT_NONE_SO_A_COMPLETE_HISTORY_CAN_NEVER_END_WITH_AN_UNCONSUMED_REEXECUTE",
    ),
    (
        "UPLOAD_RETRY_NOT_NEW_ATTEMPT",
        ("upload_retry_action", "upload_retry_reason"),
        "REUPLOAD_COMMITTED_OUTPUT_IS_BYTE_IDENTICAL_AND_NEVER_CREATES_ANOTHER_ATTEMPT_RECORD",
    ),
    (
        "RAW_ATTEMPT_RECORD_PAYLOADS_REQUIRED",
        ("raw_attempt_record_payloads_in_exact_order",),
        "VERIFIER_MUST_RECEIVE_EXACTLY_ONE_OR_TWO_EXACT_BUILTIN_BYTES_CANONICAL_ATTEMPT_RECORD_OBJECTS",
    ),
    (
        "OUTER_ATTEMPT_RECORD_FOUR_TUPLE_BINDING",
        (
            "raw_attempt_record_payloads_in_exact_order",
            "body.attempt_record_ids_in_exact_order",
            "body.attempt_record_payload_sha256s_in_exact_order",
            "body.attempt_indices_in_exact_order",
        ),
        "ALL_FOUR_SEQUENCES_MUST_HAVE_EXACT_EQUAL_CARDINALITY_ONE_OR_TWO_AND_IDENTICAL_INDEX_ORDER",
    ),
    (
        "OUTER_ATTEMPT_RECORD_ID_SHA_INDEX_REDERIVATION",
        (
            "raw_attempt_record_payloads_in_exact_order",
            "body.attempt_record_ids_in_exact_order",
            "body.attempt_record_payload_sha256s_in_exact_order",
            "body.attempt_indices_in_exact_order",
        ),
        "FOR_EACH_RAW_RECORD_RECOMPUTE_CONTENT_ID_AND_SHA256_OF_EXACT_RAW_BYTES_AND_REQUIRE_RECORD_ATTEMPT_INDEX_EQUALS_THE_OUTER_INDEX_VALUE",
    ),
    (
        "OUTER_ATTEMPT_TERMINAL_SUMMARY_PROJECTION",
        (
            "raw_attempt_record_payloads_in_exact_order[-1]",
            "body.execution_retry_action",
            "body.execution_retry_reason",
            "body.upload_retry_action",
            "body.upload_retry_reason",
            "body.any_raw_prediction_byte_produced",
            "body.raw_output_byte_count",
        ),
        "OUTER_TERMINAL_SUMMARY_FIELDS_MUST_EXACT_EQUAL_THE_FINAL_ATTEMPT_RECORD_AND_RECORD_IDS_MUST_BE_PAIRWISE_DISTINCT",
    ),
)

_ATTEMPT_DENOMINATOR_CROSS_BINDING_RULES_IN_EXACT_ORDER_V1: Final = (
    (
        "RAW_PACKAGE10_DENOMINATOR_CLOSURE_REQUIRED",
        (
            "raw_package_envelope_payloads_in_exact_order[10]",
            "raw_package10_denominator_event_payloads_in_exact_order",
            "raw_attempt_universe_head_payload",
            "raw_package10_denominator_finalization_payload",
            "raw_package16_attempt_record_payloads_in_exact_order",
        ),
        "VERIFY_PACKAGE10_ENVELOPE_ALL_SIGNED_DENOMINATOR_EVENTS_THE_OOB_PINNED_FOUR_ROLE_ATTEMPT_UNIVERSE_HEAD_AND_FOUR_ROLE_FINALIZATION_BEFORE_ACCEPTING_PACKAGE16_ATTEMPT_HISTORY",
    ),
    (
        "ATTEMPT_UNIVERSE_HEAD_EXACT_CROSS_BINDING",
        (
            "raw_attempt_universe_head_payload",
            "package10.body.attempt_universe_head_id",
            "package10.body.attempt_universe_head_payload_sha256",
            "package16.body.attempt_universe_head_id",
            "package16.body.attempt_universe_head_payload_sha256",
        ),
        "PACKAGE10_AND_PACKAGE16_MUST_EXACT_BIND_THE_SAME_OOB_PINNED_RECOMPUTED_ATTEMPT_UNIVERSE_HEAD_ID_AND_RAW_SHA256_BEFORE_ANY_ATTEMPT_INDEX_OR_TERMINAL_SUMMARY_PROJECTION",
    ),
    (
        "ATTEMPT_INDEX_BIJECTION",
        (
            "package10.body.attempt_indices_in_exact_order",
            "package16.body.attempt_indices_in_exact_order",
            "raw_package16_attempt_record_payloads_in_exact_order",
        ),
        "PACKAGE10_SIGNED_ATTEMPT_INDEX_SEQUENCE_MUST_EXACT_EQUAL_PACKAGE16_ATTEMPT_INDEX_SEQUENCE_AND_RECOMPUTED_RAW_ATTEMPT_RECORD_SEQUENCE_WITH_EXACTLY_ZERO_OR_ZERO_THEN_ONE_AND_NO_OMISSION_DUPLICATE_OR_ATTEMPT_TWO",
    ),
    (
        "REEXECUTE_ATTEMPT_CARDINALITY_IFF_AND_FINAL_NONE",
        (
            "raw_package10_denominator_event_payloads_in_exact_order",
            "raw_package16_attempt_record_payloads_in_exact_order",
            "package10.body.attempt_indices_in_exact_order",
            "package16.body.attempt_indices_in_exact_order",
        ),
        "BOTH_SIGNED_CHAINS_MUST_HAVE_ATTEMPT_INDICES_ZERO_THEN_ONE_IF_AND_ONLY_IF_ATTEMPT_ZERO_HAS_EXACTLY_ONE_ELIGIBLE_REEXECUTE_RETRY_AND_THE_FINAL_ATTEMPT_PRE_SCORE_CLOSE_AND_FINAL_ATTEMPT_RECORD_MUST_BOTH_HAVE_EXECUTION_RETRY_ACTION_AND_REASON_EXACT_NONE",
    ),
    (
        "ATTEMPT_PRE_SCORE_FACT_AND_RETRY_PROJECTION",
        (
            "raw_package10_denominator_event_payloads_in_exact_order",
            "raw_package16_attempt_record_payloads_in_exact_order",
        ),
        "FOR_EACH_ATTEMPT_REQUIRE_EXACTLY_ONE_OPEN_ONE_PRE_SCORE_CLOSE_ONE_PRE_SCORE_APPEND_RECEIPT_REQUIRE_THE_PRE_SCORE_CLOSE_OUTPUT_BYTE_FACTS_EXECUTION_RETRY_ACTION_AND_REASON_UPLOAD_RETRY_ACTION_AND_REASON_TO_EXACT_EQUAL_THE_CORRESPONDING_ATTEMPT_RECORD_REQUIRE_EACH_NON_NONE_CLOSE_RETRY_SUMMARY_TO_EXACT_EQUAL_ONE_PRECEDING_RETRY_EVENT_OF_THE_SAME_ATTEMPT_REQUIRE_THE_APPEND_OUTPUT_FACTS_TO_EXACT_EQUAL_THE_CLOSE_REQUIRE_ATTEMPT_ONE_ONLY_AFTER_ELIGIBLE_ATTEMPT_ZERO_REEXECUTE_EVENT_AND_FORBID_ANY_BACKWARD_PROJECTION_OF_PACKAGE16_POST_SCORE_TERMINAL_DISPOSITION",
    ),
    (
        "PACKAGE10_CAUSE_RETRY_CLOSURE",
        ("raw_package10_denominator_event_payloads_in_exact_order",),
        "APPLY_EXECUTION_RETRY_CAUSE_BIJECTION_AND_UPLOAD_RETRY_EXACTNESS_BEFORE_PROJECTING_ANY_PACKAGE10_FACT_INTO_PACKAGE16_SO_A_RETRY_REASON_CANNOT_SURVIVE_WITHOUT_ITS_REQUIRED_SIGNED_CAUSE_EVENT_AND_NO_CALLER_SUPPLIED_MASTER_LIST_MAY_DEFINE_COMPLETENESS_BY_CARDINALITY_ALONE",
    ),
    (
        "PRE_SCORE_TAIL_AND_POST_SCORE_FINAL_ATTEMPT_COMPLETENESS",
        (
            "package10.finalization.complete",
            "package10.finalization.pre_score_tail_event_id",
            "package16.final_attempt_record",
        ),
        "FOUR_ROLE_PRE_SCORE_FINALIZATION_COMPLETE_MUST_BE_EXACT_TRUE_PRE_SCORE_TAIL_EVENT_MUST_BE_THE_FINAL_ATTEMPT_PRE_SCORE_APPEND_RECEIPT_AND_ONLY_AFTER_PACKAGE15_SCORING_MAY_PACKAGE16_FINAL_ATTEMPT_DISPOSITION_BE_COMPLETE_FAIL_OR_COMPLETE_PASS_BEFORE_PACKAGE17_CAN_EXIST",
    ),
)

_GATE_RESULT_ID_DOMAIN_ASCII_V1: Final = (
    "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/GATE_RESULT/ID/V1\x00"
)
_GATE_RESULT_ID_PREFIX_V1: Final = (
    "phase2b_actual_unsealed_960_gate_result_v1_"
)
_GATE_RESULT_ID_FORMULA_V1: Final = (
    "gate_result_id_prefix || lowerhex(sha256(gate_result_id_domain_ascii_bytes || "
    "rfc8785(exact_record_object_with_gate_result_id_excluded)))"
)
_GATE_RESULT_ID_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1: Final = tuple(
    field_name for field_name in _GATE_RESULT_FIELDS_V1 if field_name != "gate_result_id"
)
_DENOMINATOR_EVENT_ID_DOMAIN_ASCII_V1: Final = (
    "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PACKAGE/10/DENOMINATOR_EVENT/ID/V1\x00"
)
_DENOMINATOR_EVENT_ID_PREFIX_V1: Final = (
    "phase2b_actual_unsealed_960_denominator_event_v1_"
)
_DENOMINATOR_EVENT_SIGNATURE_SLOT_V1: Final = _SignatureSlotSpecV1(
    0,
    "CUSTODIAN",
    "SIGN_DENOMINATOR_EVENT_RECORD",
    1,
    "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PACKAGE/10/"
    "DENOMINATOR_EVENT/SIGN/SIGN_DENOMINATOR_EVENT_RECORD/V1\x00",
)
_ATTEMPT_UNIVERSE_HEAD_ID_DOMAIN_ASCII_V1: Final = (
    "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PACKAGE/10/ATTEMPT_UNIVERSE_HEAD/ID/V1\x00"
)
_ATTEMPT_UNIVERSE_HEAD_ID_PREFIX_V1: Final = (
    "phase2b_actual_unsealed_960_attempt_universe_head_v1_"
)
_ATTEMPT_UNIVERSE_ROOT_DOMAIN_ASCII_V1: Final = (
    "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PACKAGE/10/ATTEMPT_UNIVERSE_HEAD/ROOT/V1\x00"
)
_ATTEMPT_UNIVERSE_ROOT_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1: Final = (
    "catalog_id",
    "signer_registry_id",
    "signer_registry_payload_sha256",
    "signer_registry_entries_root_sha256",
    "campaign_id",
    "attempt_id",
    "attempt_indices_in_exact_order",
    "denominator_event_ids_in_exact_order",
    "denominator_event_payload_sha256s_in_exact_order",
    "denominator_event_kinds_in_exact_order",
    "pre_score_tail_event_id",
    "pre_score_tail_event_payload_sha256",
    "event_count",
    "sealed_stage",
    "complete",
)
_ATTEMPT_UNIVERSE_HEAD_SIGNATURE_SLOTS_IN_EXACT_ORDER_V1: Final = (
    _SignatureSlotSpecV1(
        0,
        "CUSTODIAN",
        "SIGN_ATTEMPT_UNIVERSE_HEAD",
        1,
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PACKAGE/10/"
        "ATTEMPT_UNIVERSE_HEAD/SIGN/CUSTODIAN/V1\x00",
    ),
    _SignatureSlotSpecV1(
        1,
        "STORE_RECEIPT",
        "COUNTERSIGN_ATTEMPT_UNIVERSE_STORAGE_HEAD",
        1,
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PACKAGE/10/"
        "ATTEMPT_UNIVERSE_HEAD/SIGN/STORE_RECEIPT/V1\x00",
    ),
    _SignatureSlotSpecV1(
        2,
        "TIME_ATTESTOR",
        "COUNTERSIGN_ATTEMPT_UNIVERSE_TIME_HEAD",
        1,
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PACKAGE/10/"
        "ATTEMPT_UNIVERSE_HEAD/SIGN/TIME_ATTESTOR/V1\x00",
    ),
    _SignatureSlotSpecV1(
        3,
        "GOVERNANCE_ANCHOR",
        "COUNTERSIGN_ATTEMPT_UNIVERSE_HIGH_WATER",
        1,
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PACKAGE/10/"
        "ATTEMPT_UNIVERSE_HEAD/SIGN/GOVERNANCE/V1\x00",
    ),
)
_DENOMINATOR_FINALIZATION_SIGNATURE_SLOTS_IN_EXACT_ORDER_V1: Final = (
    _SignatureSlotSpecV1(
        0,
        "CUSTODIAN",
        "SIGN_DENOMINATOR_UNIVERSE_FINALIZATION",
        1,
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PACKAGE/10/"
        "DENOMINATOR_FINALIZATION/SIGN/CUSTODIAN/V1\x00",
    ),
    _SignatureSlotSpecV1(
        1,
        "STORE_RECEIPT",
        "COUNTERSIGN_DENOMINATOR_UNIVERSE_STORAGE",
        1,
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PACKAGE/10/"
        "DENOMINATOR_FINALIZATION/SIGN/STORE_RECEIPT/V1\x00",
    ),
    _SignatureSlotSpecV1(
        2,
        "TIME_ATTESTOR",
        "COUNTERSIGN_DENOMINATOR_UNIVERSE_TIME",
        1,
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PACKAGE/10/"
        "DENOMINATOR_FINALIZATION/SIGN/TIME_ATTESTOR/V1\x00",
    ),
    _SignatureSlotSpecV1(
        3,
        "GOVERNANCE_ANCHOR",
        "COUNTERSIGN_DENOMINATOR_UNIVERSE_FINALIZATION",
        1,
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PACKAGE/10/"
        "DENOMINATOR_FINALIZATION/SIGN/GOVERNANCE/V1\x00",
    ),
)
_DENOMINATOR_FINALIZATION_ID_DOMAIN_ASCII_V1: Final = (
    "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PACKAGE/10/DENOMINATOR_FINALIZATION/ID/V1\x00"
)
_DENOMINATOR_FINALIZATION_ID_PREFIX_V1: Final = (
    "phase2b_actual_unsealed_960_denominator_finalization_v1_"
)
_DENOMINATOR_FINALIZATION_ID_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1: Final = tuple(
    field_name
    for field_name in _DENOMINATOR_FINALIZATION_FIELDS_V1
    if field_name not in ("denominator_finalization_id", "signatures_in_exact_order")
)
_DENOMINATOR_FINALIZATION_ID_FORMULA_V1: Final = (
    "denominator_finalization_id_prefix || lowerhex(sha256("
    "denominator_finalization_id_domain_ascii_bytes || rfc8785("
    "exact_record_object_with_denominator_finalization_id_and_signatures_excluded)))"
)
_DENOMINATOR_ROOT_DOMAIN_ASCII_V1: Final = (
    "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PACKAGE/10/DENOMINATOR/ROOT/V1\x00"
)
_DENOMINATOR_ROOT_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1: Final = tuple(
    field_name
    for field_name in _DENOMINATOR_FINALIZATION_FIELDS_V1
    if field_name
    not in (
        "denominator_finalization_id",
        "denominator_root_sha256",
        "signatures_in_exact_order",
    )
)
_GATE_RESULT_BINDING_RULES_IN_EXACT_ORDER_V1: Final = (
    (
        "GATE_RESULT_CONTENT_ID",
        ("gate_result_id",) + _GATE_RESULT_ID_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1,
        "RECOMPUTE_RECORD_ID_AS_PREFIX_PLUS_SHA256_OF_NUL_TERMINATED_DOMAIN_PLUS_RFC8785_OF_EXACT_PREIMAGE_FIELDS_EXCLUDING_GATE_RESULT_ID",
    ),
    (
        "EXACT_36_GATE_PLAN_ROW",
        (
            "statistical_gate_id",
            "metric",
            "scope_axis",
            "scope_value",
            "fixed_total_count_or_none",
            "point_threshold_numerator",
            "point_threshold_denominator",
            "wilson_threshold_numerator_or_none",
            "wilson_threshold_denominator_or_none",
        ),
        "RECORD_AT_EACH_INDEX_MUST_EXACT_EQUAL_THE_CORRESPONDING_UNIQUE_FROZEN_VALIDATION_GATE_RESULT_PLAN_ROW_WITH_12_OVERALL_THEN_18_FAMILY_THEN_6_SCALE_ROWS",
    ),
    (
        "RAW_GATE_RESULT_PAYLOADS_REQUIRED",
        ("raw_gate_result_payloads_in_exact_order",),
        "VERIFIER_MUST_RECEIVE_EXACTLY_36_EXACT_BUILTIN_BYTES_CANONICAL_GATE_RESULT_OBJECTS",
    ),
    (
        "OUTER_GATE_RESULT_THREE_TUPLE_BINDING",
        (
            "raw_gate_result_payloads_in_exact_order",
            "body.overall_gate_count",
            "body.slice_gate_count",
            "body.gate_result_ids_in_exact_order",
            "body.gate_result_payload_sha256s_in_exact_order",
        ),
        "OVERALL_GATE_COUNT_MUST_BE_12_SLICE_GATE_COUNT_MUST_BE_24_AND_ALL_THREE_SEQUENCES_MUST_HAVE_EXACT_CARDINALITY_36_IDENTICAL_PLAN_ORDER_AND_PAIRWISE_DISTINCT_IDS",
    ),
    (
        "OUTER_GATE_RESULT_ID_SHA_AND_ROOT_REDERIVATION",
        (
            "raw_gate_result_payloads_in_exact_order",
            "body.gate_result_ids_in_exact_order",
            "body.gate_result_payload_sha256s_in_exact_order",
            "body.all_gate_results_root_sha256",
        ),
        "RECOMPUTE_EACH_CONTENT_ID_AND_SHA256_OF_EXACT_RAW_BYTES_THEN_RECOMPUTE_THE_LENGTH_FRAMED_ORDERED_36_RECORD_ROOT",
    ),
    (
        "COUNT_AND_POINT_THRESHOLD_EXACTNESS",
        (
            "correct_count",
            "total_count",
            "fixed_total_count_or_none",
            "point_threshold_numerator",
            "point_threshold_denominator",
            "point_passed",
        ),
        "REQUIRE_ZERO_LE_CORRECT_LE_TOTAL_TOTAL_EQUALS_FIXED_TOTAL_WHEN_NON_NULL_AND_POINT_PASSED_EQUALS_EXACT_INTEGER_CROSS_MULTIPLICATION_WITH_BOOL_AS_INT_FORBIDDEN",
    ),
    (
        "WILSON_THRESHOLD_AND_BINARY64_RECOMPUTATION",
        (
            "correct_count",
            "total_count",
            "wilson_threshold_numerator_or_none",
            "wilson_threshold_denominator_or_none",
            "wilson_lower_binary64_hex_or_none",
            "wilson_passed_or_none",
        ),
        "WHEN_WILSON_THRESHOLD_IS_NULL_BOTH_WILSON_OUTPUTS_MUST_BE_NULL_OTHERWISE_RECOMPUTE_THE_FROZEN_ONE_SIDED_95_PERCENT_WILSON_FORMULA_WITH_Z_FLOAT_HEX_0X1_A515209676AB8P_PLUS_0_AND_EXACT_OPERATION_ORDER_AND_COMPARE_TO_THE_EXACT_RATIONAL_THRESHOLD",
    ),
    (
        "FINAL_GATE_PASS_CONJUNCTION",
        ("point_passed", "wilson_passed_or_none", "passed"),
        "PASSED_MUST_BE_POINT_PASSED_AND_TRUE_WHEN_WILSON_PASSED_IS_NULL_OTHERWISE_POINT_PASSED_AND_WILSON_PASSED",
    ),
)

_DENOMINATOR_EVENT_BINDING_RULES_IN_EXACT_ORDER_V1: Final = (
    (
        "RAW_EVENT_ENVELOPES_AND_MASTER_THREE_TUPLE",
        (
            "raw_package10_denominator_event_payloads_in_exact_order",
            "body.denominator_event_ids_in_exact_order",
            "body.denominator_event_payload_sha256s_in_exact_order",
            "body.denominator_event_kinds_in_exact_order",
        ),
        "REQUIRE_EXACTLY_3_TO_9_EXACT_BUILTIN_BYTES_CANONICAL_DENOMINATOR_EVENT_ENVELOPES_AND_ALL_FOUR_SEQUENCES_HAVE_IDENTICAL_CARDINALITY_AND_INDEX_ORDER",
    ),
    (
        "EVENT_CONTENT_ID_RAW_SHA256_SIGNATURE_AND_REGISTRY",
        (
            "raw_package10_denominator_event_payloads_in_exact_order",
            "body.denominator_event_ids_in_exact_order",
            "body.denominator_event_payload_sha256s_in_exact_order",
            "raw_signer_registry_payload",
        ),
        "FOR_EACH_EVENT_RECOMPUTE_BODY_ONLY_CONTENT_ID_BIND_SHA256_OF_EXACT_RAW_ENVELOPE_AND_VERIFY_EXACT_ONE_CUSTODIAN_SIGNATURE_WITH_PURPOSE_SIGN_DENOMINATOR_EVENT_RECORD_EPOCH_1_AND_FROZEN_DOMAIN_AGAINST_THE_OOB_PINNED_SIGNER_REGISTRY",
    ),
    (
        "EVENT_LINEAGE_AND_CONTIGUOUS_CHAIN",
        (
            "raw_package10_denominator_event_payloads_in_exact_order",
            "package10.body.catalog_id",
            "package10.body.campaign_id",
            "package10.body.attempt_id",
        ),
        "EVERY_EVENT_MUST_EXACT_MATCH_CATALOG_REGISTRY_CAMPAIGN_ATTEMPT_EVENT_INDEX_ZERO_THROUGH_N_MINUS_ONE_FIRST_PREDECESSOR_EXACT_NULL_LATER_PREDECESSOR_EXACT_PRIOR_EVENT_ID_AND_ALL_EVENT_IDS_PAIRWISE_DISTINCT",
    ),
    (
        "EVENT_KIND_LOCAL_SEMANTICS",
        (
            "event_kind",
            "attempt_index",
            "stage",
            "execution_retry_action",
            "execution_retry_reason",
            "upload_retry_action",
            "upload_retry_reason",
            "any_raw_prediction_byte_produced",
            "raw_output_byte_count",
            "append_applied_or_none",
            "readback_verified_or_none",
            "fail_closed_satisfied",
            "first_failure_code",
        ),
        "APPLY_THE_EXACT_PRE_SCORE_EVENT_KIND_CONTRACT_ROW_TO_EVERY_FIELD_REQUIRE_ALL_NONAPPLICABLE_NULL_OR_NONE_FIELDS_EXACT_RECOMPUTE_FAIL_CLOSED_SATISFIED_FROM_THE_FROZEN_FORMULA_WITH_BOOL_AS_INT_FORBIDDEN_AND_FORBID_ANY_TERMINAL_OR_SCIENTIFIC_DISPOSITION_FIELD_IN_A_DENOMINATOR_EVENT",
    ),
    (
        "ATTEMPT_ZERO_OPTIONAL_ONE_TOPOLOGY",
        (
            "raw_package10_denominator_event_payloads_in_exact_order",
            "attempt_policy",
        ),
        "ATTEMPT_INDICES_MUST_BE_EXACTLY_ZERO_OR_ZERO_THEN_ONE_FOR_EACH_ATTEMPT_THE_STABLE_SUBSEQUENCE_MUST_START_WITH_EXACTLY_ONE_OPEN_THEN_CONTAIN_ZERO_OR_MORE_FACT_EVENTS_AND_AT_MOST_ONE_RETRY_THEN_EXACTLY_ONE_PRE_SCORE_CLOSE_THEN_EXACTLY_ONE_PRE_SCORE_APPEND_RECEIPT_ATTEMPT_INDICES_ARE_ZERO_THEN_ONE_IF_AND_ONLY_IF_ATTEMPT_ZERO_HAS_EXACTLY_ONE_ELIGIBLE_REEXECUTE_RETRY_ATTEMPT_ONE_OPEN_MUST_IMMEDIATELY_FOLLOW_ATTEMPT_ZERO_PRE_SCORE_APPEND_THE_FINAL_ATTEMPT_MUST_HAVE_NO_REEXECUTE_RETRY_AND_NO_ATTEMPT_TWO_OR_EVENT_AFTER_FINAL_ATTEMPT_PRE_SCORE_APPEND_EXISTS",
    ),
    (
        "EXECUTION_RETRY_CAUSE_BIJECTION_AND_UPLOAD_RETRY_EXACTNESS",
        (
            "raw_package10_denominator_event_payloads_in_exact_order",
            "attempt_policy",
        ),
        "EACH_REEXECUTE_RETRY_EVENT_MUST_HAVE_EXACTLY_ONE_EARLIER_SAME_ATTEMPT_CAUSE_EVENT_WITH_THE_SAME_REASON_CONTAINER_START_FAILURE_HOST_OOM_OR_HOST_TERMINATION_REQUIRE_CRASH_INPUT_TRANSFER_CHECKSUM_MISMATCH_REQUIRES_MISSING_OUTPUT_EACH_SUCH_CAUSE_EVENT_MUST_HAVE_EXACTLY_ONE_LATER_SAME_ATTEMPT_REEXECUTE_RETRY_AND_THE_PRE_SCORE_CLOSE_MUST_REPEAT_THE_EXACT_ACTION_REASON_SUMMARY_EACH_REUPLOAD_RETRY_REQUIRES_EXACT_REASON_OUTPUT_UPLOAD_FAILURE_AFTER_OUTPUT_HASH_COMMITTED_NO_CAUSE_EVENT_AND_AN_EXACT_MATCHING_CLOSE_SUMMARY_EACH_RETRY_EVENT_MUST_HAVE_EXACTLY_ONE_ACTIVE_ACTION_REEXECUTE_XOR_REUPLOAD_AND_NO_ORPHAN_CAUSE_RETRY_OR_NON_NONE_CLOSE_SUMMARY_IS_ALLOWED",
    ),
    (
        "SIX_KIND_PARTITION_AND_ORDER",
        (
            "body.denominator_event_ids_in_exact_order",
            "body.denominator_event_kinds_in_exact_order",
            "body.attempt_open_record_ids_in_exact_order",
            "body.attempt_pre_score_closed_record_ids_in_exact_order",
            "body.pre_score_append_receipt_ids_in_exact_order",
            "body.missing_output_record_ids_in_exact_order",
            "body.crash_record_ids_in_exact_order",
            "body.retry_record_ids_in_exact_order",
        ),
        "EACH_KIND_ARRAY_MUST_EXACT_EQUAL_THE_STABLE_FILTER_OF_THE_MASTER_EVENT_ID_SEQUENCE_FOR_THAT_EVENT_KIND_ALL_SIX_ARRAYS_ARE_PAIRWISE_DISJOINT_AND_THEIR_UNION_EXACTLY_COVERS_THE_MASTER_SEQUENCE",
    ),
    (
        "NUMERATOR_DENOMINATOR_DERIVATION",
        (
            "raw_package10_denominator_event_payloads_in_exact_order",
            "body.event_numerator",
            "body.event_denominator",
        ),
        "EVENT_DENOMINATOR_MUST_EXACT_EQUAL_RAW_EVENT_CARDINALITY_AND_EVENT_NUMERATOR_MUST_EXACT_EQUAL_THE_COUNT_OF_RECOMPUTED_FAIL_CLOSED_SATISFIED_EXACT_TRUE_VALUES_USING_SAFE_INTEGERS_WITH_BOOL_AS_INT_FORBIDDEN",
    ),
    (
        "ATOMIC_COMPLETENESS_REJECTION",
        (
            "raw_package10_denominator_event_payloads_in_exact_order",
            "body.denominator_event_ids_in_exact_order",
            "body.denominator_event_payload_sha256s_in_exact_order",
            "body.denominator_event_kinds_in_exact_order",
        ),
        "ANY_MISSING_EXTRA_DUPLICATE_REORDERED_CROSS_ATTEMPT_CROSS_CAMPAIGN_WRONG_KIND_BROKEN_PREDECESSOR_FORGED_SIGNATURE_OR_INCOMPLETE_PARTITION_REJECTS_THE_DENOMINATOR_ATOMICALLY",
    ),
)

_ATTEMPT_UNIVERSE_HEAD_BINDING_RULES_IN_EXACT_ORDER_V1: Final = (
    (
        "RAW_HEAD_AND_OOB_HIGH_WATER_REQUIRED",
        (
            "raw_attempt_universe_head_payload",
            "out_of_band_expected_attempt_universe_head_id_bytes",
            "out_of_band_expected_attempt_universe_head_payload_sha256_digest",
        ),
        "REQUIRE_EXACT_BUILTIN_BYTES_RAW_HEAD_AND_MANDATORY_OOB_ID_AND_32_BYTE_RAW_SHA256_DIGEST_PINS_FROM_THE_INDEPENDENT_ATTEMPT_LEDGER_HIGH_WATER",
    ),
    (
        "HEAD_CONTENT_ID_RAW_SHA256_AND_FOUR_ROLE_SIGNATURES",
        (
            "raw_attempt_universe_head_payload",
            "raw_signer_registry_payload",
            "out_of_band_expected_attempt_universe_head_id_bytes",
            "out_of_band_expected_attempt_universe_head_payload_sha256_digest",
        ),
        "DECODE_EXACT_CANONICAL_ENVELOPE_RECOMPUTE_BODY_ONLY_CONTENT_ID_REQUIRE_ID_AND_RAW_SHA256_EQUAL_OOB_PINS_AND_VERIFY_EXACT_CUSTODIAN_STORE_RECEIPT_TIME_ATTESTOR_GOVERNANCE_SIGNATURES_IN_FROZEN_ORDER_DOMAIN_PURPOSE_EPOCH_AGAINST_THE_OOB_PINNED_REGISTRY",
    ),
    (
        "HEAD_EVENT_UNIVERSE_EXACT_PROJECTION",
        (
            "raw_attempt_universe_head_payload",
            "raw_package10_denominator_event_payloads_in_exact_order",
            "body.denominator_event_ids_in_exact_order",
            "body.denominator_event_payload_sha256s_in_exact_order",
            "body.denominator_event_kinds_in_exact_order",
            "body.attempt_indices_in_exact_order",
            "body.pre_score_tail_event_id",
            "body.pre_score_tail_event_payload_sha256",
        ),
        "HEAD_CATALOG_REGISTRY_CAMPAIGN_ATTEMPT_EVENT_THREE_TUPLE_ATTEMPT_INDICES_TAIL_ID_TAIL_RAW_SHA256_EVENT_COUNT_AND_INDEPENDENT_EVENT_UNIVERSE_ROOT_MUST_EXACT_EQUAL_THE_FULL_VERIFIED_RAW_EVENT_CHAIN_AND_PACKAGE10_BODY_WITH_NO_PREFIX_OR_SUFFIX_OMISSION",
    ),
    (
        "FIRST_SEALED_HIGH_WATER_AND_STAGE",
        (
            "head_sequence",
            "previous_attempt_universe_head_id_or_none",
            "sealed_stage",
            "complete",
        ),
        "HEAD_SEQUENCE_MUST_EXACT_EQUAL_ONE_PREVIOUS_HEAD_MUST_EXACT_NULL_SEALED_STAGE_MUST_EXACT_EQUAL_PREDICTIONS_COMMITTED_COMPLETE_MUST_BE_EXACT_TRUE_BOOL_AS_INT_IS_FORBIDDEN_ATTEMPT_INDICES_MUST_BE_ZERO_THEN_ONE_IF_AND_ONLY_IF_ATTEMPT_ZERO_HAS_AN_ELIGIBLE_REEXECUTE_AND_THE_FINAL_ATTEMPT_MUST_HAVE_NO_OUTSTANDING_REEXECUTE",
    ),
    (
        "EVENT_UNIVERSE_ROOT_RECOMPUTATION",
        (
            "event_universe_root_sha256",
            "attempt_indices_in_exact_order",
            "denominator_event_ids_in_exact_order",
            "denominator_event_payload_sha256s_in_exact_order",
            "denominator_event_kinds_in_exact_order",
            "pre_score_tail_event_id",
            "pre_score_tail_event_payload_sha256",
            "event_count",
        ),
        "RECOMPUTE_EVENT_UNIVERSE_ROOT_FROM_THE_FROZEN_NUL_TERMINATED_DOMAIN_AND_EXACT_PREIMAGE_FIELDS_THAT_EXCLUDE_THE_HEAD_ID_SIGNATURES_AND_ANY_LATER_DENOMINATOR_FINALIZATION_REFERENCE",
    ),
    (
        "NO_CALLER_DEFINED_SHORTER_UNIVERSE",
        (
            "raw_attempt_universe_head_payload",
            "out_of_band_expected_attempt_universe_head_id_bytes",
            "raw_package10_denominator_event_payloads_in_exact_order",
        ),
        "THE_VERIFIED_OOB_PINNED_HEAD_IS_THE_AUTHORITATIVE_EVENT_COUNT_AND_TAIL_ANY_CALLER_EVENT_TUPLE_WITH_FEWER_MORE_REORDERED_OR_DIFFERENT_EVENTS_OR_ATTEMPT_INDICES_REJECTS_ATOMICALLY",
    ),
)

_DENOMINATOR_FINALIZATION_BINDING_RULES_IN_EXACT_ORDER_V1: Final = (
    (
        "RAW_DENOMINATOR_FINALIZATION_REQUIRED",
        ("raw_package10_denominator_finalization_payload",),
        "VERIFIER_MUST_RECEIVE_ONE_EXACT_BUILTIN_BYTES_CANONICAL_DENOMINATOR_FINALIZATION_OBJECT",
    ),
    (
        "RAW_DENOMINATOR_EVENTS_REQUIRED",
        (
            "raw_package10_denominator_event_payloads_in_exact_order",
            "raw_attempt_universe_head_payload",
            "raw_signer_registry_payload",
        ),
        "VERIFY_THE_COMPLETE_SIGNED_DENOMINATOR_EVENT_CHAIN_AND_OOB_PINNED_ATTEMPT_UNIVERSE_HEAD_WITH_ALL_FROZEN_BINDING_RULES_BEFORE_ACCEPTING_THE_FINALIZATION_RECORD",
    ),
    (
        "CONTENT_ID_RAW_SHA256_AND_MULTI_ROLE_SIGNATURES",
        (
            "raw_package10_denominator_finalization_payload",
            "body.denominator_finalization_id",
            "body.denominator_finalization_payload_sha256",
            "raw_signer_registry_payload",
        ),
        "RECOMPUTE_THE_DENOMINATOR_FINALIZATION_CONTENT_ID_EXCLUDING_OWN_ID_AND_SIGNATURES_BIND_SHA256_OF_EXACT_RAW_BYTES_REQUIRE_BOTH_PACKAGE10_REFERENCES_AND_VERIFY_EXACT_CUSTODIAN_STORE_RECEIPT_TIME_ATTESTOR_GOVERNANCE_SIGNATURES_IN_FROZEN_ORDER_PURPOSE_EPOCH_DOMAIN_AGAINST_THE_OOB_PINNED_REGISTRY",
    ),
    (
        "PACKAGE10_FULL_BODY_PROJECTION",
        (
            "raw_package10_denominator_finalization_payload",
            "package10.body.catalog_id",
            "package10.body.signer_registry_id",
            "package10.body.signer_registry_payload_sha256",
            "package10.body.signer_registry_entries_root_sha256",
            "package10.body.campaign_id",
            "package10.body.attempt_id",
            "package10.body.attempt_universe_head_id",
            "package10.body.attempt_universe_head_payload_sha256",
            "package10.body.denominator_event_ids_in_exact_order",
            "package10.body.denominator_event_payload_sha256s_in_exact_order",
            "package10.body.denominator_event_kinds_in_exact_order",
            "package10.body.attempt_open_record_ids_in_exact_order",
            "package10.body.attempt_pre_score_closed_record_ids_in_exact_order",
            "package10.body.pre_score_append_receipt_ids_in_exact_order",
            "package10.body.missing_output_record_ids_in_exact_order",
            "package10.body.crash_record_ids_in_exact_order",
            "package10.body.retry_record_ids_in_exact_order",
            "package10.body.attempt_indices_in_exact_order",
            "package10.body.pre_score_tail_event_id",
            "package10.body.pre_score_tail_event_payload_sha256",
            "package10.body.event_numerator",
            "package10.body.event_denominator",
        ),
        "CATALOG_REGISTRY_CAMPAIGN_ATTEMPT_OOB_PINNED_ATTEMPT_UNIVERSE_HEAD_PAIR_MASTER_EVENT_THREE_TUPLE_ALL_SIX_EVENT_ID_ARRAYS_ATTEMPT_INDICES_PRE_SCORE_TAIL_EVENT_PAIR_EVENT_NUMERATOR_AND_EVENT_DENOMINATOR_MUST_EXACT_EQUAL_PACKAGE10_BODY",
    ),
    (
        "SIGNED_EVENT_UNIVERSE_TAIL_AND_TOTAL_SEAL",
        (
            "raw_package10_denominator_event_payloads_in_exact_order",
            "raw_attempt_universe_head_payload",
            "raw_package10_denominator_finalization_payload",
            "attempt_indices_in_exact_order",
            "pre_score_tail_event_id",
            "pre_score_tail_event_payload_sha256",
            "event_denominator",
            "signatures_in_exact_order",
        ),
        "ATTEMPT_INDICES_MUST_EXACT_EQUAL_THE_STABLE_DISTINCT_ATTEMPT_INDEX_SEQUENCE_FROM_ALL_RAW_EVENTS_PRE_SCORE_TAIL_EVENT_ID_AND_RAW_SHA256_MUST_EXACT_EQUAL_THE_LAST_RAW_EVENT_EVENT_DENOMINATOR_MUST_EXACT_EQUAL_THE_OOB_PINNED_HEAD_EVENT_COUNT_AND_THE_FOUR_ROLE_FINALIZATION_SEALS_THE_SAME_COMPLETE_PRE_SCORE_EVENT_UNIVERSE_AT_PREDICTIONS_COMMITTED_BEFORE_PACKAGE15_SCORING",
    ),
    (
        "DENOMINATOR_EXACTNESS_AND_COMPLETENESS",
        (
            "event_numerator",
            "event_denominator",
            "denominator_root_sha256",
            "complete",
        ),
        "EVENT_NUMERATOR_AND_DENOMINATOR_MUST_BE_EXACT_SAFE_INTEGERS_BOOL_FORBIDDEN_ZERO_LE_NUMERATOR_LE_DENOMINATOR_RECOMPUTE_THE_DENOMINATOR_ROOT_FROM_THE_FROZEN_NUL_TERMINATED_DOMAIN_AND_EXACT_PREIMAGE_FIELDS_AND_REQUIRE_COMPLETE_EXACT_TRUE",
    ),
)


def _nested_manifests_v1(
    ordinal: int,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    common = (
        ("content_reference", _CONTENT_REFERENCE_FIELDS_V1),
        ("detached_signature", _DETACHED_SIGNATURE_FIELDS_V1),
    )
    if ordinal == 0:
        return common + (
            ("custodian_signer_binding", _SIGNER_BINDING_FIELDS_V1),
            ("signed_stage_event_envelope", _SIGNED_STAGE_EVENT_ENVELOPE_FIELDS_V1),
            ("custodian_ledger_event_body", _LEDGER_EVENT_FIELDS_V1),
            ("referenced_artifact_envelope", _REFERENCED_ARTIFACT_ENVELOPE_FIELDS_V1),
            ("store_if_absent_receipt_body", _STORE_RECEIPT_FIELDS_V1),
            ("declared_time_token_body", _TIME_TOKEN_FIELDS_V1),
            ("checkpoint_body", _CHECKPOINT_FIELDS_V1),
            ("external_anchor_body", _EXTERNAL_ANCHOR_FIELDS_V1),
        )
    if ordinal == 16:
        return common + (("attempt_record", _ATTEMPT_RECORD_FIELDS_V1),)
    if ordinal == 15:
        return common + (("gate_result", _GATE_RESULT_FIELDS_V1),)
    if ordinal == 10:
        return common + (
            ("denominator_event_envelope", _DENOMINATOR_EVENT_ENVELOPE_FIELDS_V1),
            ("denominator_event_body", _DENOMINATOR_EVENT_BODY_FIELDS_V1),
            (
                "attempt_universe_head_envelope",
                _ATTEMPT_UNIVERSE_HEAD_ENVELOPE_FIELDS_V1,
            ),
            (
                "attempt_universe_head_body",
                _ATTEMPT_UNIVERSE_HEAD_BODY_FIELDS_V1,
            ),
            ("denominator_finalization", _DENOMINATOR_FINALIZATION_FIELDS_V1),
        )
    return common


_ITEM_CARDINALITY_BY_PACKAGE_V1: Final = {
    0: (10, 10),
    1: (10, 12),
    2: (5, 16),
    3: (5, 16),
    4: (1, 64),
    5: (1532, 1532),
    6: (1, 64),
    7: (960, 960),
    8: (960, 1532),
    9: (3, 3),
    10: (3, 9),
    11: (572, 572),
    12: (240, 240),
    13: (960, 960),
    14: (10000, 10000),
    15: (36, 36),
    16: (1, 2),
    17: (17, 17),
}

PACKAGE_CATALOG_EVIDENCE_DISPOSITION_V1: Final = "NOT_SUPPLIED_NOT_VERIFIED"
PACKAGE_PRODUCER_EVIDENCE_DISPOSITION_V1: Final = "PRESENTED_UNVERIFIED"
PACKAGE_VERIFIER_SUCCESS_DISPOSITION_V1: Final = (
    "VERIFIED_PRESENTED_EXTERNAL_EVIDENCE_NOT_GATE_ADMISSION"
)
PACKAGE_VERIFIER_FAILURE_DISPOSITIONS_IN_EXACT_ORDER_V1: Final = (
    "REJECTED",
    "INVALIDATED",
)
PACKAGE_REQUIRED_EXTERNAL_EVIDENCE_CLASS_V1: Final = (
    "AUTHORITATIVE_C1_EVIDENCE_REQUIRING_EXTERNAL_LINEAGE"
)

FROZEN_STATISTICAL_THRESHOLDS_IN_EXACT_ORDER_V1: Final = (
    ("PRESERVATION_LEGAL_POINT_MINIMUM", 482, 496, "AT_LEAST", None),
    ("PRESERVATION_LEGAL_WILSON_LOWER_MINIMUM", 94, 100, "AT_LEAST", "0.94"),
    ("PRESERVATION_INVALID_CONTROL_EXACT", 76, 76, "EXACTLY", "1"),
    ("SCALE_REGRET_POINT_MEAN_MAXIMUM", 1, 20, "AT_MOST", "0.05"),
    ("SCALE_REGRET_BOOTSTRAP_UPPER_MAXIMUM", 8, 100, "AT_MOST", "0.08"),
    ("CHALLENGE_LOW_OVERLAP_POINT_MINIMUM", 109, 120, "AT_LEAST", None),
    ("CHALLENGE_LOW_OVERLAP_WILSON_LOWER_MINIMUM", 85, 100, "AT_LEAST", "0.85"),
    ("CHALLENGE_HIGH_OVERLAP_POINT_MINIMUM", 114, 120, "AT_LEAST", "0.95"),
    ("CHALLENGE_HIGH_OVERLAP_WILSON_LOWER_MINIMUM", 90, 100, "AT_LEAST", "0.9"),
    ("SEMANTIC_ADVANTAGE_POINT_MINIMUM", 15, 100, "AT_LEAST", "0.15"),
    ("SEMANTIC_ADVANTAGE_BOOTSTRAP_LOWER_MINIMUM", 10, 100, "AT_LEAST", "0.1"),
)
_PACKAGE_THRESHOLD_IDS_BY_ORDINAL_V1: Final = {
    11: (
        "PRESERVATION_LEGAL_POINT_MINIMUM",
        "PRESERVATION_LEGAL_WILSON_LOWER_MINIMUM",
        "PRESERVATION_INVALID_CONTROL_EXACT",
    ),
    12: (
        "CHALLENGE_LOW_OVERLAP_POINT_MINIMUM",
        "CHALLENGE_LOW_OVERLAP_WILSON_LOWER_MINIMUM",
        "CHALLENGE_HIGH_OVERLAP_POINT_MINIMUM",
        "CHALLENGE_HIGH_OVERLAP_WILSON_LOWER_MINIMUM",
        "SEMANTIC_ADVANTAGE_POINT_MINIMUM",
        "SEMANTIC_ADVANTAGE_BOOTSTRAP_LOWER_MINIMUM",
    ),
    13: (
        "SCALE_REGRET_POINT_MEAN_MAXIMUM",
        "SCALE_REGRET_BOOTSTRAP_UPPER_MAXIMUM",
    ),
}

_PACKAGE_FINALIZATION_STAGE_BY_ORDINAL_V1: Final = {
    0: "TERMINAL_CONSUMED",
    1: "TERMINAL_CONSUMED",
    2: "PREDICTIONS_COMMITTED",
    3: "PREDICTIONS_COMMITTED",
    4: "RUN_FINISHED",
    5: "PREDICTIONS_COMMITTED",
    6: "PREREGISTERED",
    7: "RAW_INPUT_COMMITTED",
    8: "PACKAGE_COMMITMENTS_FROZEN",
    9: "PREDICTIONS_COMMITTED",
    10: "PREDICTIONS_COMMITTED",
    11: "REPORTS_COMMITTED",
    12: "REPORTS_COMMITTED",
    13: "REPORTS_COMMITTED",
    14: "REPORTS_COMMITTED",
    15: "REPORTS_COMMITTED",
    16: "REPORTS_COMMITTED",
    17: "TERMINAL_CONSUMED",
}


PACKAGE_SPECS_IN_EXACT_ORDER_V1: Final = tuple(
    _PackageSpecV1(
        ordinal=ordinal,
        package_name=name,
        purpose=purpose,
        schema_version=(
            "hegel-machine-phase2b-actual-unsealed-960-authoritative-evidence-"
            f"package/{ordinal}/1"
        ),
        envelope_fields_in_exact_order=_PACKAGE_ENVELOPE_FIELDS_V1,
        body_fields_in_exact_order=_PACKAGE_COMMON_BODY_FIELDS_V1 + body_fields,
        nested_field_manifests_in_exact_order=_nested_manifests_v1(ordinal),
        content_id_domain_ascii=(
            "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PACKAGE/"
            f"{ordinal}/ID/V1\x00"
        ),
        content_id_prefix=(
            "phase2b_actual_unsealed_960_authoritative_evidence_package_"
            f"{ordinal}_v1_"
        ),
        signature_slots_in_exact_order=_package_signature_slots_v1(ordinal),
        minimum_payload_bytes=2,
        maximum_payload_bytes=16_777_216,
        minimum_item_count=_ITEM_CARDINALITY_BY_PACKAGE_V1[ordinal][0],
        maximum_item_count=_ITEM_CARDINALITY_BY_PACKAGE_V1[ordinal][1],
        predecessor_package_ordinals_in_exact_order=predecessors,
        required_external_evidence_class=PACKAGE_REQUIRED_EXTERNAL_EVIDENCE_CLASS_V1,
        catalog_evidence_disposition=PACKAGE_CATALOG_EVIDENCE_DISPOSITION_V1,
        producer_evidence_disposition=PACKAGE_PRODUCER_EVIDENCE_DISPOSITION_V1,
        verifier_success_disposition=PACKAGE_VERIFIER_SUCCESS_DISPOSITION_V1,
        verifier_failure_dispositions_in_exact_order=(
            PACKAGE_VERIFIER_FAILURE_DISPOSITIONS_IN_EXACT_ORDER_V1
        ),
        frozen_threshold_ids_in_exact_order=(
            _PACKAGE_THRESHOLD_IDS_BY_ORDINAL_V1.get(ordinal, ())
        ),
        required_gate_ids_in_exact_order=required_gates,
        producer_stage=producer_stage,
        finalization_stage=_PACKAGE_FINALIZATION_STAGE_BY_ORDINAL_V1[ordinal],
        consumer_stages_in_exact_order=consumers,
        terminal_behavior=terminal_behavior,
        negative_vector_families_in_exact_order=_NEGATIVE_VECTOR_FAMILIES_V1,
    )
    for (
        ordinal,
        name,
        purpose,
        body_fields,
        predecessors,
        required_gates,
        producer_stage,
        consumers,
        terminal_behavior,
    ) in _PACKAGE_ROWS_V1
)


ATTEMPT_POLICY_V1: Final = (
    ("allowed_attempt_index_sequences", ((0,), (0, 1))),
    ("attempt_2_exists", False),
    ("conditional_attempt_1_action", "REEXECUTE"),
    ("conditional_attempt_1_requires_any_raw_prediction_byte_produced", False),
    ("conditional_attempt_1_requires_raw_output_byte_count", 0),
    (
        "conditional_attempt_1_reasons_in_exact_order",
        (
            "CONTAINER_START_FAILURE",
            "HOST_OOM_BEFORE_FIRST_PREDICTION_BYTE",
            "HOST_TERMINATION_BEFORE_FIRST_PREDICTION_BYTE",
            "INPUT_TRANSFER_CHECKSUM_MISMATCH",
        ),
    ),
    (
        "attempt_1_terminal_dispositions_in_exact_order",
        ("COMPLETE_PASS", "COMPLETE_FAIL"),
    ),
    ("upload_retry_action", "REUPLOAD_COMMITTED_OUTPUT"),
    (
        "upload_retry_reason",
        "OUTPUT_UPLOAD_FAILURE_AFTER_OUTPUT_HASH_COMMITTED",
    ),
    ("upload_retry_requires_byte_identical_payload", True),
    ("upload_retry_creates_new_attempt", False),
)

TERMINAL_DISPOSITION_POLICY_V1: Final = (
    (
        "NOT_RUN",
        "NO_ATTEMPT_STARTED_NO_ATTEMPT_ID_NO_TERMINAL_EVENT_AND_NO_PACKAGE17_OR_BUNDLE_INSTANCE",
    ),
    (
        "BLOCKED",
        "UNMET_PREDECESSOR_PREVENTED_ATTEMPT_START_NO_ATTEMPT_ID_NO_TERMINAL_EVENT_AND_NO_PACKAGE17_OR_BUNDLE_INSTANCE",
    ),
    (
        "REJECTED",
        "AUTHORITY_CUSTODY_CHRONOLOGY_CANONICALITY_OR_SIGNATURE_FAILURE_REJECTS_BEFORE_SCIENTIFIC_SCORING",
    ),
    (
        "INVALIDATED",
        "POST_START_LEAKAGE_INTEGRITY_RETRY_ONE_SHOT_OR_CUSTODY_FAILURE_PERMANENTLY_INVALIDATES_THE_ATTEMPT",
    ),
    (
        "COMPLETE_FAIL",
        "COMPLETE_VALID_RUN_MISSED_FROZEN_SCIENTIFIC_OR_PERFORMANCE_GATE_RETAIN_RESULT",
    ),
    (
        "COMPLETE_PASS",
        "COMPLETE_VALID_RUN_PASSED_ALL_16_AUTHORITY_GATES_AND_ALL_36_FROZEN_STATISTICAL_GATES_AND_REACHED_TERMINAL_CONSUMED",
    ),
)

DENOMINATOR_EVENT_KIND_CONTRACTS_IN_EXACT_ORDER_V1: Final = (
    (
        "ATTEMPT_OPEN",
        ("RUN_STARTED",),
        "EXACT_NONE",
        (),
        "EXACT_NONE",
        (),
        "EXACT_FALSE_AND_ZERO",
        "EXACT_NULL",
        "EXACT_NULL",
        ("NONE",),
        "EXACT_NONE",
        "FAIL_CLOSED_SATISFIED_MUST_BE_EXACT_TRUE_AFTER_ALL_FIELDS_MATCH",
    ),
    (
        "ATTEMPT_PRE_SCORE_CLOSED",
        ("RUN_STARTED", "RUN_FINISHED", "PREDICTIONS_COMMITTED"),
        "EXACT_REEXECUTE_OR_NONE",
        (
            "CONTAINER_START_FAILURE",
            "HOST_OOM_BEFORE_FIRST_PREDICTION_BYTE",
            "HOST_TERMINATION_BEFORE_FIRST_PREDICTION_BYTE",
            "INPUT_TRANSFER_CHECKSUM_MISMATCH",
            "NONE",
        ),
        "EXACT_NONE_OR_REUPLOAD_COMMITTED_OUTPUT",
        ("NONE", "OUTPUT_UPLOAD_FAILURE_AFTER_OUTPUT_HASH_COMMITTED"),
        "ANY_RAW_PREDICTION_BYTE_PRODUCED_MUST_EXACT_EQUAL_RAW_OUTPUT_BYTE_COUNT_GT_ZERO",
        "EXACT_NULL",
        "EXACT_NULL",
        (
            "NONE",
            "CONTAINER_START_FAILURE",
            "HOST_OOM_BEFORE_FIRST_PREDICTION_BYTE",
            "HOST_TERMINATION_BEFORE_FIRST_PREDICTION_BYTE",
            "INPUT_TRANSFER_CHECKSUM_MISMATCH",
            "OUTPUT_UPLOAD_FAILURE_AFTER_OUTPUT_HASH_COMMITTED",
        ),
        "EXACT_NONE_WHEN_BOTH_RETRY_ACTIONS_ARE_NONE_OTHERWISE_EXACT_EQUAL_THE_ONE_ACTIVE_EXECUTION_OR_UPLOAD_RETRY_REASON",
        "FAIL_CLOSED_SATISFIED_MUST_BE_EXACT_TRUE_ONLY_FOR_A_COMPLETE_PRE_SCORE_CLOSE_WITH_VALID_OUTPUT_AND_RETRY_SUMMARY_STATE_AND_NO_TERMINAL_OR_SCIENTIFIC_DISPOSITION_FIELD",
    ),
    (
        "PRE_SCORE_APPEND_RECEIPT",
        ("RUN_STARTED", "RUN_FINISHED", "PREDICTIONS_COMMITTED"),
        "EXACT_NONE",
        (),
        "EXACT_NONE",
        (),
        "ANY_RAW_PREDICTION_BYTE_PRODUCED_MUST_EXACT_EQUAL_RAW_OUTPUT_BYTE_COUNT_GT_ZERO",
        "EXACT_BOOLEAN",
        "EXACT_BOOLEAN",
        (
            "NONE",
            "STORE_APPEND_FAILURE",
            "STORE_READBACK_FAILURE",
            "STORE_APPEND_AND_READBACK_FAILURE",
        ),
        "EXACT_NONE_IFF_APPEND_APPLIED_AND_READBACK_VERIFIED_ARE_TRUE_OTHERWISE_EXACT_FAILURE_CODE_SELECTED_BY_THE_TWO_BOOLEAN_FAILURE_PATTERN",
        "FAIL_CLOSED_SATISFIED_MUST_EXACT_EQUAL_APPEND_APPLIED_AND_READBACK_VERIFIED",
    ),
    (
        "MISSING_OUTPUT",
        ("RUN_STARTED", "RUN_FINISHED"),
        "EXACT_NONE",
        ("INPUT_TRANSFER_CHECKSUM_MISMATCH",),
        "EXACT_NONE",
        (),
        "EXACT_FALSE_AND_ZERO",
        "EXACT_NULL",
        "EXACT_NULL",
        ("INPUT_TRANSFER_CHECKSUM_MISMATCH",),
        "EXACT_INPUT_TRANSFER_CHECKSUM_MISMATCH",
        "FAIL_CLOSED_SATISFIED_MUST_BE_EXACT_TRUE_ONLY_WHEN_MISSING_OUTPUT_IS_RETAINED_BEFORE_ANY_RAW_BYTE_AND_ANY_INPUT_TRANSFER_CHECKSUM_MISMATCH_CAUSE_IS_BIJECTIVELY_BOUND_TO_A_REEXECUTE_RETRY",
    ),
    (
        "CRASH",
        ("RUN_STARTED", "RUN_FINISHED"),
        "EXACT_NONE",
        (
            "CONTAINER_START_FAILURE",
            "HOST_OOM_BEFORE_FIRST_PREDICTION_BYTE",
            "HOST_TERMINATION_BEFORE_FIRST_PREDICTION_BYTE",
        ),
        "EXACT_NONE",
        (),
        "EXACT_FALSE_AND_ZERO",
        "EXACT_NULL",
        "EXACT_NULL",
        (
            "CONTAINER_START_FAILURE",
            "HOST_OOM_BEFORE_FIRST_PREDICTION_BYTE",
            "HOST_TERMINATION_BEFORE_FIRST_PREDICTION_BYTE",
        ),
        "EXACT_EQUAL_THE_SINGLE_EXECUTION_RETRY_REASON",
        "FAIL_CLOSED_SATISFIED_MUST_BE_EXACT_TRUE_ONLY_FOR_FROZEN_PREOUTPUT_CRASH_REASON_FALSE_AND_ZERO",
    ),
    (
        "RETRY",
        ("RUN_STARTED", "RUN_FINISHED", "PREDICTIONS_COMMITTED"),
        "EXACT_REEXECUTE_OR_NONE",
        (
            "CONTAINER_START_FAILURE",
            "HOST_OOM_BEFORE_FIRST_PREDICTION_BYTE",
            "HOST_TERMINATION_BEFORE_FIRST_PREDICTION_BYTE",
            "INPUT_TRANSFER_CHECKSUM_MISMATCH",
            "NONE",
        ),
        "EXACT_NONE_OR_REUPLOAD_COMMITTED_OUTPUT",
        ("NONE", "OUTPUT_UPLOAD_FAILURE_AFTER_OUTPUT_HASH_COMMITTED"),
        "REEXECUTE_REQUIRES_FALSE_AND_ZERO_WHILE_REUPLOAD_REQUIRES_BYTE_IDENTICAL_COMMITTED_OUTPUT_AND_CREATES_NO_ATTEMPT",
        "EXACT_NULL",
        "EXACT_NULL",
        (
            "CONTAINER_START_FAILURE",
            "HOST_OOM_BEFORE_FIRST_PREDICTION_BYTE",
            "HOST_TERMINATION_BEFORE_FIRST_PREDICTION_BYTE",
            "INPUT_TRANSFER_CHECKSUM_MISMATCH",
            "OUTPUT_UPLOAD_FAILURE_AFTER_OUTPUT_HASH_COMMITTED",
        ),
        "EXACT_EQUAL_THE_ONE_ACTIVE_EXECUTION_OR_UPLOAD_RETRY_REASON",
        "FAIL_CLOSED_SATISFIED_MUST_EXACT_EQUAL_THE_FROZEN_EXECUTION_XOR_UPLOAD_RETRY_PREDICATE_EACH_REEXECUTE_MUST_HAVE_ITS_BIJECTIVE_SIGNED_CAUSE_EVENT_EACH_REUPLOAD_HAS_NO_CAUSE_EVENT_AND_ONLY_REEXECUTE_MAY_ENABLE_ATTEMPT_ONE",
    ),
)

DENOMINATOR_EXECUTION_RETRY_CAUSE_ROWS_IN_EXACT_ORDER_V1: Final = (
    ("CONTAINER_START_FAILURE", "CRASH"),
    ("HOST_OOM_BEFORE_FIRST_PREDICTION_BYTE", "CRASH"),
    ("HOST_TERMINATION_BEFORE_FIRST_PREDICTION_BYTE", "CRASH"),
    ("INPUT_TRANSFER_CHECKSUM_MISMATCH", "MISSING_OUTPUT"),
)

_SIGNER_REGISTRY_SIGNATURE_SLOTS_IN_EXACT_ORDER_V1: Final = (
    _SignatureSlotSpecV1(
        0,
        "GOVERNANCE_ANCHOR",
        "SIGN_C1_SIGNER_REGISTRY",
        1,
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/SIGNER_REGISTRY/SIGN/GOVERNANCE/V1\x00",
    ),
    _SignatureSlotSpecV1(
        1,
        "CUSTODIAN",
        "COUNTERSIGN_C1_SIGNER_REGISTRY",
        1,
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/SIGNER_REGISTRY/SIGN/CUSTODIAN/V1\x00",
    ),
)
_BUNDLE_INDEX_SIGNATURE_SLOTS_IN_EXACT_ORDER_V1: Final = (
    _SignatureSlotSpecV1(
        0,
        "GOVERNANCE_ANCHOR",
        "SIGN_AUTHORITATIVE_EVIDENCE_BUNDLE_INDEX",
        1,
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/BUNDLE_INDEX/SIGN/GOVERNANCE/V1\x00",
    ),
    _SignatureSlotSpecV1(
        1,
        "CUSTODIAN",
        "COUNTERSIGN_AUTHORITATIVE_EVIDENCE_BUNDLE_INDEX",
        1,
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/BUNDLE_INDEX/SIGN/CUSTODIAN/V1\x00",
    ),
)

PERMANENT_NONCLAIMS_IN_EXACT_ORDER_V1: Final = (
    "evidence_supplied",
    "evidence_verified",
    "r01_satisfied",
    "r02_satisfied",
    "r03_satisfied",
    "r04_satisfied",
    "r05_satisfied",
    "r06_satisfied",
    "r07_satisfied",
    "r08_satisfied",
    "r09_satisfied",
    "r10_satisfied",
    "r11_satisfied",
    "r12_satisfied",
    "r13_satisfied",
    "r14_satisfied",
    "r15_satisfied",
    "r16_satisfied",
    "authority_currentness_verified",
    "owner_non_revocation_verified",
    "six_live_roles_verified",
    "external_key_custody_verified",
    "external_durability_verified",
    "external_non_equivocation_verified",
    "campaign_id_allocated",
    "attempt_id_allocated",
    "actual_960_generated",
    "actual_corpus_audit_executed",
    "inference_performed",
    "predictions_committed",
    "answers_revealed",
    "scoring_performed",
    "terminal_report_consumed",
    "execution_authorized",
    "c1_satisfied",
)
_PERMANENT_NONCLAIMS_ORACLE_V1: Final = {
    "evidence_supplied": False,
    "evidence_verified": False,
    "r01_satisfied": False,
    "r02_satisfied": False,
    "r03_satisfied": False,
    "r04_satisfied": False,
    "r05_satisfied": False,
    "r06_satisfied": False,
    "r07_satisfied": False,
    "r08_satisfied": False,
    "r09_satisfied": False,
    "r10_satisfied": False,
    "r11_satisfied": False,
    "r12_satisfied": False,
    "r13_satisfied": False,
    "r14_satisfied": False,
    "r15_satisfied": False,
    "r16_satisfied": False,
    "authority_currentness_verified": False,
    "owner_non_revocation_verified": False,
    "six_live_roles_verified": False,
    "external_key_custody_verified": False,
    "external_durability_verified": False,
    "external_non_equivocation_verified": False,
    "campaign_id_allocated": False,
    "attempt_id_allocated": False,
    "actual_960_generated": False,
    "actual_corpus_audit_executed": False,
    "inference_performed": False,
    "predictions_committed": False,
    "answers_revealed": False,
    "scoring_performed": False,
    "terminal_report_consumed": False,
    "execution_authorized": False,
    "c1_satisfied": False,
}


_GATE_SCHEMA_ID_DOMAIN_PREFIX_V1: Final = (
    "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/CATALOG/GATE_SCHEMA/"
)
_STAGE_SCHEMA_ID_DOMAIN_PREFIX_V1: Final = (
    "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/CATALOG/STAGE_SCHEMA/"
)
_PACKAGE_SCHEMA_ID_DOMAIN_PREFIX_V1: Final = (
    "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/CATALOG/PACKAGE_SCHEMA/"
)
_GATE_SCHEMA_ID_PREFIX_V1: Final = (
    "phase2b_actual_unsealed_960_authoritative_gate_schema_"
)
_PREALLOCATION_LINEAGE_SCHEMA_ID_DOMAIN_V1: Final = (
    b"HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/CATALOG/PREALLOCATION_LINEAGE_SCHEMA/ID/V1\x00"
)
_PREALLOCATION_LINEAGE_SCHEMA_ID_PREFIX_V1: Final = (
    "phase2b_actual_unsealed_960_preallocation_lineage_schema_v1_"
)
_PREALLOCATION_LINEAGE_CONTENT_ID_DOMAIN_V1: Final = (
    "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PREALLOCATION_LINEAGE/ID/V1\x00"
)
_PREALLOCATION_LINEAGE_CONTENT_ID_PREFIX_V1: Final = (
    "phase2b_actual_unsealed_960_preallocation_lineage_v1_"
)
_GATE_SUPPORT_SCHEMA_ID_DOMAIN_PREFIX_V1: Final = (
    "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/CATALOG/GATE_SUPPORT_SCHEMA/"
)
_GATE_SOURCE_SCHEMA_ID_DOMAIN_PREFIX_V1: Final = (
    "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/CATALOG/GATE_SOURCE_SCHEMA/"
)
_GATE_SOURCE_SCHEMA_ID_PREFIX_V1: Final = (
    "phase2b_actual_unsealed_960_gate_source_schema_"
)
_GATE_SOURCE_SCHEMA_ROOT_DOMAIN_V1: Final = (
    b"HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/CATALOG/GATE_SOURCE_SCHEMA_ROOT/V1\x00"
)
_GATE_SUPPORT_SCHEMA_ID_PREFIX_V1: Final = (
    "phase2b_actual_unsealed_960_gate_support_schema_"
)
_GATE_SUPPORT_SCHEMA_ROOT_DOMAIN_V1: Final = (
    b"HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/CATALOG/GATE_SUPPORT_SCHEMA_ROOT/V1\x00"
)
_STAGE_SCHEMA_ID_PREFIX_V1: Final = (
    "phase2b_actual_unsealed_960_authoritative_stage_schema_"
)
_PACKAGE_SCHEMA_ID_PREFIX_V1: Final = (
    "phase2b_actual_unsealed_960_authoritative_package_schema_"
)
_GATE_ROOT_DOMAIN_V1: Final = (
    b"HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/CATALOG/GATE_ROOT/V1\x00"
)
_STAGE_ROOT_DOMAIN_V1: Final = (
    b"HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/CATALOG/STAGE_ROOT/V1\x00"
)
_PACKAGE_ROOT_DOMAIN_V1: Final = (
    b"HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/CATALOG/PACKAGE_ROOT/V1\x00"
)
_PACKAGE_SCHEMA_DAG_ROOT_DOMAIN_V1: Final = (
    b"HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/CATALOG/PACKAGE_SCHEMA_DAG_ROOT/V1\x00"
)
_FINALIZATION_OPERATIONAL_DAG_ROOT_DOMAIN_V1: Final = (
    b"HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/CATALOG/FINALIZATION_OPERATIONAL_DAG_ROOT/V1\x00"
)
_BUNDLE_INDEX_SCHEMA_ID_DOMAIN_V1: Final = (
    b"HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/CATALOG/BUNDLE_INDEX_SCHEMA/ID/V1\x00"
)
_SIGNER_REGISTRY_SCHEMA_ID_DOMAIN_V1: Final = (
    b"HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/SIGNER_REGISTRY/SCHEMA/ID/V1\x00"
)
_SIGNER_REGISTRY_SCHEMA_ID_PREFIX_V1: Final = (
    "phase2b_actual_unsealed_960_signer_registry_schema_v1_"
)
_SIGNER_REGISTRY_CONTENT_ID_DOMAIN_V1: Final = (
    "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/SIGNER_REGISTRY/ID/V1\x00"
)
_SIGNER_REGISTRY_CONTENT_ID_PREFIX_V1: Final = (
    "phase2b_actual_unsealed_960_signer_registry_v1_"
)
_SIGNER_REGISTRY_ENTRY_ROOT_DOMAIN_V1: Final = (
    "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/SIGNER_REGISTRY/ENTRY_ROOT/V1\x00"
)
_SIGNER_REGISTRY_AUTHORIZATION_ROOT_DOMAIN_V1: Final = (
    "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/SIGNER_REGISTRY/AUTHORIZATION_ROOT/V1\x00"
)
_BUNDLE_INDEX_SCHEMA_ID_PREFIX_V1: Final = (
    "phase2b_actual_unsealed_960_authoritative_bundle_index_schema_v1_"
)
_BUNDLE_INDEX_CONTENT_ID_DOMAIN_V1: Final = (
    "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/BUNDLE_INDEX/ID/V1\x00"
)
_BUNDLE_INDEX_CONTENT_ID_PREFIX_V1: Final = (
    "phase2b_actual_unsealed_960_authoritative_evidence_bundle_index_v1_"
)
_P17_GATE_EVIDENCE_ROOT_DOMAIN_ASCII_V1: Final = (
    "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PACKAGE/17/GATE_EVIDENCE_ROOT/V1\x00"
)
_P17_SEALED_REPORT_ROOT_DOMAIN_ASCII_V1: Final = (
    "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/PACKAGE/17/SEALED_REPORT_ROOT/V1\x00"
)
_BUNDLE_ROOT_DOMAIN_ASCII_V1: Final = (
    "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/BUNDLE_INDEX/BUNDLE_ROOT/V1\x00"
)
_P17_GATE_EVIDENCE_ROOT_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1: Final = (
    "gate_evidence_ids_in_exact_order",
    "gate_evidence_payload_sha256s_in_exact_order",
)
_P17_SEALED_REPORT_ROOT_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1: Final = (
    "predecessor_package_instance_ids_in_exact_order",
    "predecessor_package_instance_payload_sha256s_in_exact_order",
    "gate_evidence_ids_in_exact_order",
    "gate_evidence_payload_sha256s_in_exact_order",
    "terminal_stage_event_id",
    "terminal_stage_event_payload_sha256",
    "terminal_append_receipt_id",
    "terminal_append_receipt_payload_sha256",
    "denominator_finalization_id",
    "denominator_finalization_payload_sha256",
    "overall_gate_count",
    "slice_gate_count",
    "final_disposition",
)
_BUNDLE_ROOT_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1: Final = (
    "package_schema_ids_in_exact_order",
    "package_instance_ids_in_exact_order",
    "package_instance_payload_sha256s_in_exact_order",
    "terminal_stage_event_id",
    "terminal_stage_event_payload_sha256",
    "final_disposition",
    "interleaved_dag_root",
)
_ORDERED_ROOT_FORMULA_V1: Final = (
    "sha256_colon_lowerhex(sha256(nul_terminated_domain_ascii_bytes || "
    "rfc8785(exact_object_of_preimage_fields_in_exact_order)))"
)
_DAG_ROOT_DOMAIN_V1: Final = (
    b"HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/CATALOG/DAG_ROOT/V1\x00"
)
_CATALOG_ID_DOMAIN_V1: Final = (
    b"HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/CATALOG/ID/V1\x00"
)
_CATALOG_ID_PREFIX_V1: Final = (
    "phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1_"
)


def _content_id_v1(domain: bytes, prefix: str, value: object) -> str:
    return prefix + hashlib.sha256(
        domain + canonicalize_rfc8785_v1(value)  # type: ignore[arg-type]
    ).hexdigest()


def _gate_signature_slot_for_registry_v1(
    spec: _GateSpecV1,
) -> tuple[str, _SignatureSlotSpecV1]:
    return (
        f"gate_evidence_{spec.ordinal}_v1",
        _SignatureSlotSpecV1(
            0,
            "GOVERNANCE_ANCHOR",
            f"SIGN_{spec.gate_id}_EVIDENCE",
            1,
            (
                "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/GATE/"
                f"{spec.ordinal}/SIGN/SIGN_{spec.gate_id}_EVIDENCE/V1\x00"
            ),
        ),
    )


def _stage_signature_slot_for_registry_v1(
    spec: _StageSpecV1,
) -> tuple[str, _SignatureSlotSpecV1]:
    return (
        f"stage_event_{spec.ordinal}_v1",
        _SignatureSlotSpecV1(
            0,
            spec.producer_role,
            f"SIGN_{spec.stage}_STAGE_EVENT",
            1,
            (
                "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/STAGE/"
                f"{spec.ordinal}/SIGN/SIGN_{spec.stage}_STAGE_EVENT/V1\x00"
            ),
        ),
    )


def _signature_authorization_rows_in_exact_order_v1(
) -> tuple[tuple[str, int, str, str, int, str], ...]:
    rows: list[tuple[str, int, str, str, int, str]] = []
    rows.extend(
        (
            "preallocation_authority_lineage_v1",
            slot.ordinal,
            slot.role,
            slot.purpose,
            slot.key_epoch,
            slot.signature_domain_ascii,
        )
        for slot in _PREALLOCATION_LINEAGE_SIGNATURE_SLOTS_IN_EXACT_ORDER_V1
    )
    for spec in GATE_SPECS_IN_EXACT_ORDER_V1:
        rows.extend(
            (
                f"gate_support_artifact_{spec.ordinal}_v1",
                slot.ordinal,
                slot.role,
                slot.purpose,
                slot.key_epoch,
                slot.signature_domain_ascii,
            )
            for slot in _gate_support_signature_slots_v1(spec)
        )
    for spec in GATE_SOURCE_SPECS_IN_EXACT_ORDER_V1:
        rows.extend(
            (
                f"gate_source_receipt_{spec.ordinal}_v1",
                slot.ordinal,
                slot.role,
                slot.purpose,
                slot.key_epoch,
                slot.signature_domain_ascii,
            )
            for slot in _gate_source_receipt_signature_slots_v1(spec)
        )
    for spec in GATE_SPECS_IN_EXACT_ORDER_V1:
        artifact_kind, slot = _gate_signature_slot_for_registry_v1(spec)
        rows.append(
            (
                artifact_kind,
                slot.ordinal,
                slot.role,
                slot.purpose,
                slot.key_epoch,
                slot.signature_domain_ascii,
            )
        )
    for spec in STAGE_SPECS_IN_EXACT_ORDER_V1:
        artifact_kind, slot = _stage_signature_slot_for_registry_v1(spec)
        rows.append(
            (
                artifact_kind,
                slot.ordinal,
                slot.role,
                slot.purpose,
                slot.key_epoch,
                slot.signature_domain_ascii,
            )
        )
    for spec in PACKAGE_SPECS_IN_EXACT_ORDER_V1:
        rows.extend(
            (
                f"authoritative_evidence_package_{spec.ordinal}_v1",
                slot.ordinal,
                slot.role,
                slot.purpose,
                slot.key_epoch,
                slot.signature_domain_ascii,
            )
            for slot in spec.signature_slots_in_exact_order
        )
    rows.extend(
        (
            f"signed_custodian_stage_event_{index}_v1",
            slot.ordinal,
            slot.role,
            slot.purpose,
            slot.key_epoch,
            slot.signature_domain_ascii,
        )
        for index, slot in enumerate(
            _LEDGER_EVENT_SIGNATURE_SLOTS_IN_EXACT_ORDER_V1
        )
    )
    for artifact_spec in _PACKAGE0_REFERENCED_ARTIFACT_SPECS_IN_EXACT_ORDER_V1:
        rows.extend(
            (
                f"package0_referenced_{artifact_spec.artifact_kind}_v1",
                slot.ordinal,
                slot.role,
                slot.purpose,
                slot.key_epoch,
                slot.signature_domain_ascii,
            )
            for slot in artifact_spec.signature_slots_in_exact_order
        )
    rows.append(
        (
            "package10_denominator_event_v1",
            _DENOMINATOR_EVENT_SIGNATURE_SLOT_V1.ordinal,
            _DENOMINATOR_EVENT_SIGNATURE_SLOT_V1.role,
            _DENOMINATOR_EVENT_SIGNATURE_SLOT_V1.purpose,
            _DENOMINATOR_EVENT_SIGNATURE_SLOT_V1.key_epoch,
            _DENOMINATOR_EVENT_SIGNATURE_SLOT_V1.signature_domain_ascii,
        )
    )
    rows.extend(
        (
            "package10_attempt_universe_head_v1",
            slot.ordinal,
            slot.role,
            slot.purpose,
            slot.key_epoch,
            slot.signature_domain_ascii,
        )
        for slot in _ATTEMPT_UNIVERSE_HEAD_SIGNATURE_SLOTS_IN_EXACT_ORDER_V1
    )
    rows.extend(
        (
            "package10_denominator_finalization_v1",
            slot.ordinal,
            slot.role,
            slot.purpose,
            slot.key_epoch,
            slot.signature_domain_ascii,
        )
        for slot in _DENOMINATOR_FINALIZATION_SIGNATURE_SLOTS_IN_EXACT_ORDER_V1
    )
    rows.extend(
        (
            "authoritative_evidence_bundle_index_v1",
            slot.ordinal,
            slot.role,
            slot.purpose,
            slot.key_epoch,
            slot.signature_domain_ascii,
        )
        for slot in _BUNDLE_INDEX_SIGNATURE_SLOTS_IN_EXACT_ORDER_V1
    )
    return tuple(rows)


SIGNATURE_AUTHORIZATION_ROWS_IN_EXACT_ORDER_V1: Final = (
    _signature_authorization_rows_in_exact_order_v1()
)
SIGNER_REGISTRY_ROLES_IN_EXACT_ORDER_V1: Final = (
    "GOVERNANCE_ANCHOR",
    "CUSTODIAN",
    "STORE_RECEIPT",
    "TIME_ATTESTOR",
    "RECOGNIZER",
    "SCORER_EVALUATOR",
)


def _signer_registry_schema_json_v1() -> dict[str, object]:
    body_field_types = _field_type_manifest_json_v1(
        _SIGNER_REGISTRY_BODY_FIELDS_V1
    )
    for row in body_field_types:
        if row["field_name"] == "entries_in_exact_order":
            row["exact_wire_type"] = (
                "EXACT_JSON_ARRAY_OF_SIGNER_REGISTRY_ENTRY_OBJECTS"
            )
        if row["field_name"] == "authorized_signature_rows_in_exact_order":
            row["exact_wire_type"] = (
                "EXACT_JSON_ARRAY_OF_SIGNATURE_AUTHORIZATION_ROW_OBJECTS"
            )
    core: dict[str, object] = {
        "schema_version": (
            "hegel-machine-phase2b-actual-unsealed-960-signer-registry/1"
        ),
        "envelope_fields_in_exact_order": list(
            _SIGNER_REGISTRY_ENVELOPE_FIELDS_V1
        ),
        "envelope_field_types_in_exact_order": _field_type_manifest_json_v1(
            _SIGNER_REGISTRY_ENVELOPE_FIELDS_V1
        ),
        "body_fields_in_exact_order": list(_SIGNER_REGISTRY_BODY_FIELDS_V1),
        "body_field_types_in_exact_order": body_field_types,
        "nested_field_manifests_in_exact_order": [
            {
                "record_name": "signer_registry_entry",
                "fields_in_exact_order": list(_SIGNER_REGISTRY_ENTRY_FIELDS_V1),
                "field_types_in_exact_order": _field_type_manifest_json_v1(
                    _SIGNER_REGISTRY_ENTRY_FIELDS_V1
                ),
            },
            {
                "record_name": "signature_authorization_row",
                "fields_in_exact_order": list(
                    _SIGNATURE_AUTHORIZATION_ROW_FIELDS_V1
                ),
                "field_types_in_exact_order": _field_type_manifest_json_v1(
                    _SIGNATURE_AUTHORIZATION_ROW_FIELDS_V1
                ),
            },
            {
                "record_name": "detached_signature",
                "fields_in_exact_order": list(_DETACHED_SIGNATURE_FIELDS_V1),
                "field_types_in_exact_order": _field_type_manifest_json_v1(
                    _DETACHED_SIGNATURE_FIELDS_V1
                ),
            },
        ],
        "schema_content_id_domain_ascii": (
            _SIGNER_REGISTRY_SCHEMA_ID_DOMAIN_V1.decode("ascii")
        ),
        "schema_content_id_prefix": _SIGNER_REGISTRY_SCHEMA_ID_PREFIX_V1,
        "content_id_domain_ascii": _SIGNER_REGISTRY_CONTENT_ID_DOMAIN_V1,
        "content_id_prefix": _SIGNER_REGISTRY_CONTENT_ID_PREFIX_V1,
        "content_id_preimage_profile": CONTENT_ID_PREIMAGE_PROFILE_V1,
        "content_id_preimage_fields_in_exact_order": list(
            CONTENT_ID_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
        ),
        "signature_preimage_profile": DETACHED_SIGNATURE_PREIMAGE_PROFILE_V1,
        "detached_signature_schema_version": (
            DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_V1
        ),
        "signature_preimage_schema_version": (
            DETACHED_SIGNATURE_PREIMAGE_SCHEMA_VERSION_V1
        ),
        "signature_preimage_artifact_kind": "c1_signer_registry_v1",
        "signature_algorithm": SIGNATURE_ALGORITHM_V1,
        "signature_preimage_fields_in_exact_order": list(
            DETACHED_SIGNATURE_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
        ),
        "signature_slots_in_exact_order": [
            _signature_slot_json_v1(slot)
            for slot in _SIGNER_REGISTRY_SIGNATURE_SLOTS_IN_EXACT_ORDER_V1
        ],
        "signature_cardinality": 2,
        "root_signature_key_source": (
            "MANDATORY_OUT_OF_BAND_GOVERNANCE_AND_CUSTODIAN_ROOT_KEY_IDS_AND_PUBLIC_KEYS_NOT_REGISTRY_SELF_DECLARATIONS"
        ),
        "required_roles_in_exact_order": list(
            SIGNER_REGISTRY_ROLES_IN_EXACT_ORDER_V1
        ),
        "required_entry_cardinality": len(SIGNER_REGISTRY_ROLES_IN_EXACT_ORDER_V1),
        "required_authorization_rows_in_exact_order": [
            dict(zip(_SIGNATURE_AUTHORIZATION_ROW_FIELDS_V1, row, strict=True))
            for row in SIGNATURE_AUTHORIZATION_ROWS_IN_EXACT_ORDER_V1
        ],
        "entry_root_domain_ascii": _SIGNER_REGISTRY_ENTRY_ROOT_DOMAIN_V1,
        "entry_root_formula": (
            "sha256_colon_lowerhex(sha256(entry_root_domain_ascii_bytes || rfc8785(entries_in_exact_order)))"
        ),
        "authorization_root_domain_ascii": (
            _SIGNER_REGISTRY_AUTHORIZATION_ROOT_DOMAIN_V1
        ),
        "authorization_root_formula": (
            "sha256_colon_lowerhex(sha256(authorization_root_domain_ascii_bytes || rfc8785(authorized_signature_rows_in_exact_order)))"
        ),
        "raw_signer_registry_payload_required": True,
        "signer_registry_binding_rules_in_exact_order": _binding_rules_json_v1(
            _SIGNER_REGISTRY_BINDING_RULES_IN_EXACT_ORDER_V1
        ),
        "minimum_payload_bytes": 2,
        "maximum_payload_bytes": 16_777_216,
        "negative_vector_families_in_exact_order": list(
            _NEGATIVE_VECTOR_FAMILIES_V1
        ),
    }
    return {
        **core,
        "signer_registry_schema_id": _content_id_v1(
            _SIGNER_REGISTRY_SCHEMA_ID_DOMAIN_V1,
            _SIGNER_REGISTRY_SCHEMA_ID_PREFIX_V1,
            core,
        ),
    }


def _canonical_unit_interval_decimal_from_rational_or_none_v1(
    numerator: int,
    denominator: int,
) -> str | None:
    if (
        type(numerator) is not int
        or type(denominator) is not int
        or numerator < 0
        or denominator <= 0
        or numerator > denominator
    ):
        raise RuntimeError("catalog unit-interval rational differs")
    left = numerator
    right = denominator
    while right:
        left, right = right, left % right
    divisor = left
    reduced_numerator = numerator // divisor
    reduced_denominator = denominator // divisor
    if reduced_numerator == 0:
        return "0"
    if reduced_numerator == reduced_denominator:
        return "1"
    twos = 0
    fives = 0
    remaining = reduced_denominator
    while remaining % 2 == 0:
        twos += 1
        remaining //= 2
    while remaining % 5 == 0:
        fives += 1
        remaining //= 5
    if remaining != 1:
        return None
    scale = max(twos, fives)
    scaled_numerator = (
        reduced_numerator
        * (2 ** (scale - twos))
        * (5 ** (scale - fives))
    )
    digits = str(scaled_numerator).rjust(scale + 1, "0")
    integer_part = digits[:-scale] if scale else digits
    fractional_part = digits[-scale:] if scale else ""
    fractional_part = fractional_part.rstrip("0")
    if not fractional_part:
        return integer_part
    return integer_part + "." + fractional_part


DAG_ROOT_V1: Final = _content_id_v1(
    _DAG_ROOT_DOMAIN_V1,
    "phase2b_actual_unsealed_960_authoritative_runbook_dag_root_v1_",
    [list(edge) for edge in INTERLEAVED_DAG_EDGES_IN_EXACT_ORDER_V1],
)

FINALIZATION_OPERATIONAL_EDGES_IN_EXACT_ORDER_V1: Final = (
    ("package10", "package15"),
    ("package10", "package16"),
    ("package15", "package16"),
    ("package15", "terminal_stage_event"),
    ("package16", "terminal_stage_event"),
    ("terminal_stage_event", "package0"),
    ("package0", "package17"),
    ("package17", "bundle_index"),
)
FINALIZATION_OPERATIONAL_DAG_ROOT_V1: Final = _content_id_v1(
    _FINALIZATION_OPERATIONAL_DAG_ROOT_DOMAIN_V1,
    "phase2b_actual_unsealed_960_finalization_operational_dag_root_v1_",
    [list(edge) for edge in FINALIZATION_OPERATIONAL_EDGES_IN_EXACT_ORDER_V1],
)


def _identifier_sequence_root_v1(
    domain: bytes, prefix: str, identifiers: tuple[str, ...]
) -> str:
    framed = bytearray(domain)
    framed.extend(len(identifiers).to_bytes(4, "big"))
    for identifier in identifiers:
        raw = identifier.encode("ascii")
        if len(raw) > 65_535:
            raise RuntimeError("catalog identifier exceeds u16 frame")
        framed.extend(len(raw).to_bytes(2, "big"))
        framed.extend(raw)
    return prefix + hashlib.sha256(bytes(framed)).hexdigest()


def _field_type_manifest_json_v1(
    field_names: tuple[str, ...],
) -> list[dict[str, str]]:
    return [
        {"field_name": field_name, "exact_wire_type": exact_wire_type}
        for field_name, exact_wire_type in _field_type_manifest_v1(field_names)
    ]


def _preallocation_lineage_schema_json_v1() -> dict[str, object]:
    core: dict[str, object] = {
        "schema_version": (
            "hegel-machine-phase2b-actual-unsealed-960-preallocation-"
            "authority-lineage-schema/1"
        ),
        "envelope_fields_in_exact_order": list(
            _PREALLOCATION_LINEAGE_ENVELOPE_FIELDS_V1
        ),
        "envelope_field_types_in_exact_order": _field_type_manifest_json_v1(
            _PREALLOCATION_LINEAGE_ENVELOPE_FIELDS_V1
        ),
        "body_fields_in_exact_order": list(
            _PREALLOCATION_LINEAGE_BODY_FIELDS_V1
        ),
        "body_field_types_in_exact_order": _field_type_manifest_json_v1(
            _PREALLOCATION_LINEAGE_BODY_FIELDS_V1
        ),
        "schema_content_id_domain_ascii": (
            _PREALLOCATION_LINEAGE_SCHEMA_ID_DOMAIN_V1.decode("ascii")
        ),
        "schema_content_id_prefix": (
            _PREALLOCATION_LINEAGE_SCHEMA_ID_PREFIX_V1
        ),
        "content_id_domain_ascii": _PREALLOCATION_LINEAGE_CONTENT_ID_DOMAIN_V1,
        "content_id_prefix": _PREALLOCATION_LINEAGE_CONTENT_ID_PREFIX_V1,
        "content_id_preimage_profile": CONTENT_ID_PREIMAGE_PROFILE_V1,
        "content_id_preimage_fields_in_exact_order": list(
            CONTENT_ID_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
        ),
        "content_id_formula": CONTENT_ID_FORMULA_V1,
        "signature_preimage_profile": DETACHED_SIGNATURE_PREIMAGE_PROFILE_V1,
        "detached_signature_schema_version": (
            DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_V1
        ),
        "signature_preimage_schema_version": (
            DETACHED_SIGNATURE_PREIMAGE_SCHEMA_VERSION_V1
        ),
        "signature_algorithm": SIGNATURE_ALGORITHM_V1,
        "signature_preimage_artifact_kind": "preallocation_authority_lineage_v1",
        "signature_preimage_fields_in_exact_order": list(
            DETACHED_SIGNATURE_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
        ),
        "signature_formula": DETACHED_SIGNATURE_FORMULA_V1,
        "signature_slots_in_exact_order": [
            _signature_slot_json_v1(slot)
            for slot in _PREALLOCATION_LINEAGE_SIGNATURE_SLOTS_IN_EXACT_ORDER_V1
        ],
        "signature_cardinality": 4,
        "signer_registry_schema_id": SIGNER_REGISTRY_SCHEMA_OBJECT_V1[
            "signer_registry_schema_id"
        ],
        "raw_signer_registry_payload_required": True,
        "binding_rules_in_exact_order": _binding_rules_json_v1(
            _PREALLOCATION_LINEAGE_BINDING_RULES_IN_EXACT_ORDER_V1
        ),
        "minimum_payload_bytes": 2,
        "maximum_payload_bytes": 16_777_216,
        "claim_boundary": (
            "RAW_MULTI_ROLE_PREALLOCATION_LINEAGE_MECHANICS_NOT_CURRENTNESS_"
            "NOT_AUTHORITY_UNTIL_EXTERNAL_KEYS_AND_ROOTS_ARE_ACTUALLY_PROVISIONED"
        ),
    }
    return {
        **core,
        "preallocation_lineage_schema_id": _content_id_v1(
            _PREALLOCATION_LINEAGE_SCHEMA_ID_DOMAIN_V1,
            _PREALLOCATION_LINEAGE_SCHEMA_ID_PREFIX_V1,
            core,
        ),
    }


def _gate_source_receipt_schema_json_v1(
    spec: _GateSourceSpecV1,
) -> dict[str, object]:
    gate_spec = GATE_SPECS_IN_EXACT_ORDER_V1[spec.ordinal - 1]
    schema_domain_ascii = (
        _GATE_SOURCE_SCHEMA_ID_DOMAIN_PREFIX_V1
        + f"{spec.ordinal}/ID/V1\x00"
    )
    schema_prefix = _GATE_SOURCE_SCHEMA_ID_PREFIX_V1 + f"{spec.ordinal}_v1_"
    content_domain_ascii = (
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/GATE_SOURCE/"
        f"{spec.ordinal}/ID/V1\x00"
    )
    content_prefix = (
        "phase2b_actual_unsealed_960_gate_source_receipt_"
        f"{spec.ordinal}_v1_"
    )
    source_root_domain_ascii = (
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/GATE/"
        f"{spec.ordinal}/SOURCE_ROOT/V1\x00"
    )
    slots = _gate_source_receipt_signature_slots_v1(spec)
    allowed_cardinalities = tuple(
        len(order)
        for order in spec.allowed_native_source_kind_orders_in_exact_order
    )
    core: dict[str, object] = {
        "schema_version": (
            "hegel-machine-phase2b-actual-unsealed-960-authoritative-gate-"
            f"source-receipt-schema/{spec.ordinal}/1"
        ),
        "gate_ordinal": spec.ordinal,
        "gate_id": spec.gate_id,
        "dependencies_in_exact_order": list(
            gate_spec.dependencies_in_exact_order
        ),
        "lineage_scope": _gate_lineage_scope_v1(spec.ordinal),
        "campaign_attempt_id_policy": _gate_campaign_attempt_policy_v1(
            spec.ordinal
        ),
        "gate_source_receipt_artifact_kind": (
            _gate_source_receipt_artifact_kind_v1(spec)
        ),
        "verification_availability": spec.verification_availability,
        "verifier_dispatch_id": spec.verifier_dispatch_id,
        "verifier_source_sha256_or_none": (
            spec.verifier_source_sha256_or_none
        ),
        "allowed_native_source_kind_orders_in_exact_order": [
            list(order)
            for order in spec.allowed_native_source_kind_orders_in_exact_order
        ],
        "allowed_native_source_cardinalities_in_exact_order": list(
            allowed_cardinalities
        ),
        "native_payload_format": (
            "EXACT_BUILTIN_BYTES_PASSED_OUT_OF_LINE_IN_DESCRIPTOR_ORDER"
        ),
        "envelope_fields_in_exact_order": list(
            _GATE_SOURCE_RECEIPT_ENVELOPE_FIELDS_V1
        ),
        "envelope_field_types_in_exact_order": _field_type_manifest_json_v1(
            _GATE_SOURCE_RECEIPT_ENVELOPE_FIELDS_V1
        ),
        "body_fields_in_exact_order": list(
            _GATE_SOURCE_RECEIPT_BODY_FIELDS_V1
        ),
        "body_field_types_in_exact_order": _field_type_manifest_json_v1(
            _GATE_SOURCE_RECEIPT_BODY_FIELDS_V1
        ),
        "nested_field_manifests_in_exact_order": [
            {
                "record_name": "detached_signature",
                "fields_in_exact_order": list(_DETACHED_SIGNATURE_FIELDS_V1),
                "field_types_in_exact_order": _field_type_manifest_json_v1(
                    _DETACHED_SIGNATURE_FIELDS_V1
                ),
            }
        ],
        "schema_content_id_domain_ascii": schema_domain_ascii,
        "schema_content_id_prefix": schema_prefix,
        "content_id_domain_ascii": content_domain_ascii,
        "content_id_prefix": content_prefix,
        "content_id_preimage_profile": CONTENT_ID_PREIMAGE_PROFILE_V1,
        "content_id_preimage_fields_in_exact_order": list(
            CONTENT_ID_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
        ),
        "content_id_formula": CONTENT_ID_FORMULA_V1,
        "signature_preimage_profile": DETACHED_SIGNATURE_PREIMAGE_PROFILE_V1,
        "detached_signature_schema_version": (
            DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_V1
        ),
        "signature_preimage_schema_version": (
            DETACHED_SIGNATURE_PREIMAGE_SCHEMA_VERSION_V1
        ),
        "signature_algorithm": SIGNATURE_ALGORITHM_V1,
        "signature_preimage_artifact_kind": (
            f"gate_source_receipt_{spec.ordinal}_v1"
        ),
        "signature_preimage_fields_in_exact_order": list(
            DETACHED_SIGNATURE_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
        ),
        "signature_formula": DETACHED_SIGNATURE_FORMULA_V1,
        "signature_slots_in_exact_order": [
            _signature_slot_json_v1(slot) for slot in slots
        ],
        "signature_cardinality": 4,
        "signer_registry_schema_id": SIGNER_REGISTRY_SCHEMA_OBJECT_V1[
            "signer_registry_schema_id"
        ],
        "raw_signer_registry_payload_required": True,
        "preallocation_lineage_schema_id": (
            PREALLOCATION_LINEAGE_SCHEMA_OBJECT_V1[
                "preallocation_lineage_schema_id"
            ]
        ),
        "raw_preallocation_lineage_payload_required": True,
        "raw_native_source_payload_tuple_required": True,
        "raw_predecessor_gate_source_receipt_payloads_required": True,
        "predecessor_gate_source_receipt_cardinality": len(
            gate_spec.dependencies_in_exact_order
        ),
        "source_root_domain_ascii": source_root_domain_ascii,
        "source_root_preimage_fields_in_exact_order": list(
            _GATE_SOURCE_ROOT_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
        ),
        "source_root_formula": _ORDERED_ROOT_FORMULA_V1,
        "binding_rules_in_exact_order": _binding_rules_json_v1(
            _GATE_SOURCE_RECEIPT_BINDING_RULES_IN_EXACT_ORDER_V1
        ),
        "body_required_literals": {
            "status": "RAW_NATIVE_SOURCE_RECEIPT_PRESENTED",
            "claim_level": "RAW_PROVENANCE_ONLY_NOT_GATE_DECISION",
            "lineage_scope": _gate_lineage_scope_v1(spec.ordinal),
            "gate_ordinal": spec.ordinal,
            "gate_id": spec.gate_id,
            "verification_availability": spec.verification_availability,
            "verifier_dispatch_id": spec.verifier_dispatch_id,
            "verifier_source_sha256_or_none": None,
            "producer_observation_sequence": spec.ordinal,
            "first_failure_code": (
                GATE_VERIFIER_NOT_IMPLEMENTED_FAILURE_CODE_V1
            ),
        },
        "minimum_payload_bytes": 2,
        "maximum_payload_bytes": 16_777_216,
        "claim_boundary": (
            "SIGNED_NATIVE_SOURCE_PROVENANCE_AND_RAW_BYTE_BINDING_ONLY_"
            "NOT_GATE_TRUTH_NOT_CURRENTNESS_NOT_AUTHORITY_CURRENT_V1_"
            "HAS_NO_EXECUTABLE_PRODUCER_FREE_DISPATCH"
        ),
    }
    return {
        **core,
        "gate_source_schema_id": _content_id_v1(
            schema_domain_ascii.encode("ascii"), schema_prefix, core
        ),
    }


def _gate_support_schema_json_v1(spec: _GateSpecV1) -> dict[str, object]:
    source_spec = GATE_SOURCE_SPECS_IN_EXACT_ORDER_V1[spec.ordinal - 1]
    source_schema = GATE_SOURCE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[
        spec.ordinal - 1
    ]
    schema_domain_ascii = (
        _GATE_SUPPORT_SCHEMA_ID_DOMAIN_PREFIX_V1
        + f"{spec.ordinal}/ID/V1\x00"
    )
    schema_prefix = _GATE_SUPPORT_SCHEMA_ID_PREFIX_V1 + f"{spec.ordinal}_v1_"
    content_domain_ascii = (
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/GATE_SUPPORT/"
        f"{spec.ordinal}/ID/V1\x00"
    )
    content_prefix = (
        "phase2b_actual_unsealed_960_gate_support_artifact_"
        f"{spec.ordinal}_v1_"
    )
    slots = _gate_support_signature_slots_v1(spec)
    core: dict[str, object] = {
        "schema_version": (
            "hegel-machine-phase2b-actual-unsealed-960-authoritative-gate-"
            f"support-schema/{spec.ordinal}/1"
        ),
        "gate_ordinal": spec.ordinal,
        "gate_id": spec.gate_id,
        "lineage_scope": _gate_lineage_scope_v1(spec.ordinal),
        "campaign_attempt_id_policy": _gate_campaign_attempt_policy_v1(
            spec.ordinal
        ),
        "gate_support_artifact_kind": _gate_support_artifact_kind_v1(spec),
        "gate_source_schema_id": source_schema["gate_source_schema_id"],
        "gate_source_receipt_artifact_kind": (
            _gate_source_receipt_artifact_kind_v1(source_spec)
        ),
        "verification_availability": source_spec.verification_availability,
        "verifier_dispatch_id": source_spec.verifier_dispatch_id,
        "verifier_source_sha256_or_none": (
            source_spec.verifier_source_sha256_or_none
        ),
        "allowed_native_source_kind_orders_in_exact_order": [
            list(order)
            for order in (
                source_spec.allowed_native_source_kind_orders_in_exact_order
            )
        ],
        "evidence_requirement": spec.evidence_requirement,
        "required_evidence_assertions_in_exact_order": [
            spec.evidence_requirement
        ],
        "evidence_payload_format": (
            "CANONICAL_RFC8785_EXACT_GATE_OPERATIONAL_EVIDENCE_OBJECT_HEX_V1"
        ),
        "envelope_fields_in_exact_order": list(
            _GATE_SUPPORT_ARTIFACT_ENVELOPE_FIELDS_V1
        ),
        "envelope_field_types_in_exact_order": _field_type_manifest_json_v1(
            _GATE_SUPPORT_ARTIFACT_ENVELOPE_FIELDS_V1
        ),
        "body_fields_in_exact_order": list(
            _GATE_SUPPORT_ARTIFACT_BODY_FIELDS_V1
        ),
        "body_field_types_in_exact_order": _field_type_manifest_json_v1(
            _GATE_SUPPORT_ARTIFACT_BODY_FIELDS_V1
        ),
        "inline_operational_evidence_schema_version": (
            "hegel-machine-phase2b-actual-unsealed-960-gate-operational-"
            f"evidence/{spec.ordinal}/1"
        ),
        "inline_operational_evidence_fields_in_exact_order": list(
            _GATE_OPERATIONAL_EVIDENCE_FIELDS_V1
        ),
        "inline_operational_evidence_field_types_in_exact_order": (
            _field_type_manifest_json_v1(
                _GATE_OPERATIONAL_EVIDENCE_FIELDS_V1
            )
        ),
        "schema_content_id_domain_ascii": schema_domain_ascii,
        "schema_content_id_prefix": schema_prefix,
        "content_id_domain_ascii": content_domain_ascii,
        "content_id_prefix": content_prefix,
        "content_id_preimage_profile": CONTENT_ID_PREIMAGE_PROFILE_V1,
        "content_id_preimage_fields_in_exact_order": list(
            CONTENT_ID_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
        ),
        "content_id_formula": CONTENT_ID_FORMULA_V1,
        "signature_preimage_profile": DETACHED_SIGNATURE_PREIMAGE_PROFILE_V1,
        "detached_signature_schema_version": (
            DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_V1
        ),
        "signature_preimage_schema_version": (
            DETACHED_SIGNATURE_PREIMAGE_SCHEMA_VERSION_V1
        ),
        "signature_algorithm": SIGNATURE_ALGORITHM_V1,
        "signature_preimage_artifact_kind": (
            f"gate_support_artifact_{spec.ordinal}_v1"
        ),
        "signature_preimage_fields_in_exact_order": list(
            DETACHED_SIGNATURE_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
        ),
        "signature_formula": DETACHED_SIGNATURE_FORMULA_V1,
        "signature_slots_in_exact_order": [
            _signature_slot_json_v1(slot) for slot in slots
        ],
        "signature_cardinality": 4,
        "signer_registry_schema_id": SIGNER_REGISTRY_SCHEMA_OBJECT_V1[
            "signer_registry_schema_id"
        ],
        "raw_signer_registry_payload_required": True,
        "raw_gate_source_receipt_payload_required": True,
        "raw_gate_source_native_payload_tuple_required": True,
        "preallocation_lineage_schema_id": (
            PREALLOCATION_LINEAGE_SCHEMA_OBJECT_V1[
                "preallocation_lineage_schema_id"
            ]
        ),
        "binding_rules_in_exact_order": _binding_rules_json_v1(
            _GATE_SUPPORT_ARTIFACT_BINDING_RULES_IN_EXACT_ORDER_V1
        ),
        "minimum_evidence_payload_bytes": 1,
        "maximum_evidence_payload_bytes": 16_777_216,
        "success_required_literals": {
            "status": "SCHEMA_FROZEN_GATE_SUPPORT_NOT_VERIFIED",
            "claim_level": "PRESENTED_SOURCE_PROVENANCE_ONLY_NOT_GATE_DECISION",
            "support_artifact_kind": _gate_support_artifact_kind_v1(spec),
            "lineage_scope": _gate_lineage_scope_v1(spec.ordinal),
            "verification_availability": (
                CURRENT_GATE_VERIFICATION_AVAILABILITY_V1
            ),
            "verifier_dispatch_id": source_spec.verifier_dispatch_id,
            "verifier_source_sha256_or_none": None,
            "evidence_requirement": spec.evidence_requirement,
            "evidence_assertions_in_exact_order": [spec.evidence_requirement],
            "evidence_payload_format": (
                "CANONICAL_RFC8785_EXACT_GATE_OPERATIONAL_EVIDENCE_OBJECT_HEX_V1"
            ),
            "evidence_observed": False,
            "current": False,
            "non_revoked": False,
            "satisfied": False,
            "first_failure_code": (
                GATE_VERIFIER_NOT_IMPLEMENTED_FAILURE_CODE_V1
            ),
        },
        "claim_boundary": (
            "CURRENT_V1_GATE_SUPPORT_IS_SCHEMA_ONLY_AND_CANNOT_ESTABLISH_"
            "ANY_GATE_FOUR_ROLE_SIGNATURES_PROVE_PROVENANCE_ONLY_A_NEW_"
            "CATALOG_VERSION_IS_REQUIRED_FOR_AN_EXECUTABLE_BYTES_ONLY_DISPATCH"
        ),
    }
    return {
        **core,
        "gate_support_schema_id": _content_id_v1(
            schema_domain_ascii.encode("ascii"), schema_prefix, core
        ),
    }


def _gate_json_v1(spec: _GateSpecV1) -> dict[str, object]:
    source_spec = GATE_SOURCE_SPECS_IN_EXACT_ORDER_V1[spec.ordinal - 1]
    source_schema = GATE_SOURCE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[
        spec.ordinal - 1
    ]
    gate_support_schema = GATE_SUPPORT_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[
        spec.ordinal - 1
    ]
    schema_domain_ascii = (
        _GATE_SCHEMA_ID_DOMAIN_PREFIX_V1 + f"{spec.ordinal}/ID/V1\x00"
    )
    schema_content_id_prefix = (
        _GATE_SCHEMA_ID_PREFIX_V1 + f"{spec.ordinal}_v1_"
    )
    evidence_schema_version = (
        "hegel-machine-phase2b-actual-unsealed-960-authoritative-gate-"
        f"evidence/{spec.ordinal}/1"
    )
    evidence_domain_ascii = (
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/GATE/"
        f"{spec.ordinal}/EVIDENCE/ID/V1\x00"
    )
    evidence_content_id_prefix = (
        "phase2b_actual_unsealed_960_authoritative_gate_evidence_"
        f"{spec.ordinal}_v1_"
    )
    signature_slot = _SignatureSlotSpecV1(
        0,
        "GOVERNANCE_ANCHOR",
        f"SIGN_{spec.gate_id}_EVIDENCE",
        1,
        (
            "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/GATE/"
            f"{spec.ordinal}/SIGN/SIGN_{spec.gate_id}_EVIDENCE/V1\x00"
        ),
    )
    core: dict[str, object] = {
        "ordinal": spec.ordinal,
        "gate_id": spec.gate_id,
        "gate_id_wire_type": _GATE_ID_EXACT_WIRE_TYPE_V1,
        "preallocation_lineage_schema_id": (
            PREALLOCATION_LINEAGE_SCHEMA_OBJECT_V1[
                "preallocation_lineage_schema_id"
            ]
        ),
        "gate_support_schema_id": gate_support_schema[
            "gate_support_schema_id"
        ],
        "gate_source_schema_id": source_schema["gate_source_schema_id"],
        "gate_source_receipt_artifact_kind": (
            _gate_source_receipt_artifact_kind_v1(source_spec)
        ),
        "verification_availability": source_spec.verification_availability,
        "verifier_dispatch_id": source_spec.verifier_dispatch_id,
        "verifier_source_sha256_or_none": (
            source_spec.verifier_source_sha256_or_none
        ),
        "allowed_native_source_kind_orders_in_exact_order": [
            list(order)
            for order in source_spec.allowed_native_source_kind_orders_in_exact_order
        ],
        "gate_support_artifact_kind": _gate_support_artifact_kind_v1(spec),
        "required_supporting_artifact_kinds_in_exact_order": [
            "PREALLOCATION_AUTHORITY_LINEAGE_V1",
            _gate_source_receipt_artifact_kind_v1(source_spec),
            _gate_support_artifact_kind_v1(spec),
        ],
        "lineage_scope": _gate_lineage_scope_v1(spec.ordinal),
        "campaign_attempt_id_policy": _gate_campaign_attempt_policy_v1(
            spec.ordinal
        ),
        "dependencies_in_exact_order": list(spec.dependencies_in_exact_order),
        "dependency_gate_id_wire_type": _GATE_ID_EXACT_WIRE_TYPE_V1,
        "evidence_requirement": spec.evidence_requirement,
        "initial_satisfied": spec.initial_satisfied,
        "evidence_schema_version": evidence_schema_version,
        "evidence_envelope_fields_in_exact_order": list(
            _GATE_EVIDENCE_ENVELOPE_FIELDS_V1
        ),
        "evidence_envelope_field_types_in_exact_order": (
            _field_type_manifest_json_v1(_GATE_EVIDENCE_ENVELOPE_FIELDS_V1)
        ),
        "evidence_body_fields_in_exact_order": list(
            _GATE_EVIDENCE_BODY_FIELDS_V1
        ),
        "evidence_body_field_types_in_exact_order": (
            _field_type_manifest_json_v1(_GATE_EVIDENCE_BODY_FIELDS_V1)
        ),
        "nested_field_manifests_in_exact_order": [
            {
                "record_name": "detached_signature",
                "fields_in_exact_order": list(_DETACHED_SIGNATURE_FIELDS_V1),
                "field_types_in_exact_order": _field_type_manifest_json_v1(
                    _DETACHED_SIGNATURE_FIELDS_V1
                ),
            }
        ],
        "schema_content_id_domain_ascii": schema_domain_ascii,
        "schema_content_id_prefix": schema_content_id_prefix,
        "content_id_domain_ascii": evidence_domain_ascii,
        "content_id_prefix": evidence_content_id_prefix,
        "content_id_preimage_profile": CONTENT_ID_PREIMAGE_PROFILE_V1,
        "content_id_preimage_fields_in_exact_order": list(
            CONTENT_ID_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
        ),
        "signature_preimage_profile": DETACHED_SIGNATURE_PREIMAGE_PROFILE_V1,
        "detached_signature_schema_version": (
            DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_V1
        ),
        "signature_preimage_schema_version": (
            DETACHED_SIGNATURE_PREIMAGE_SCHEMA_VERSION_V1
        ),
        "signature_preimage_artifact_kind": f"gate_evidence_{spec.ordinal}_v1",
        "signature_algorithm": SIGNATURE_ALGORITHM_V1,
        "signature_preimage_fields_in_exact_order": list(
            DETACHED_SIGNATURE_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
        ),
        "signature_slots_in_exact_order": [
            _signature_slot_json_v1(signature_slot)
        ],
        "signature_cardinality": 1,
        "signer_registry_schema_id": SIGNER_REGISTRY_SCHEMA_OBJECT_V1[
            "signer_registry_schema_id"
        ],
        "raw_signer_registry_payload_required": True,
        "signer_registry_binding_rules_in_exact_order": _binding_rules_json_v1(
            _SIGNER_REGISTRY_BINDING_RULES_IN_EXACT_ORDER_V1
        ),
        "minimum_payload_bytes": 2,
        "maximum_payload_bytes": 16_777_216,
        "minimum_supporting_artifact_count": 3,
        "maximum_supporting_artifact_count": 3,
        "predecessor_gate_evidence_cardinality": len(
            spec.dependencies_in_exact_order
        ),
        "raw_predecessor_gate_evidence_payloads_required": True,
        "predecessor_gate_evidence_binding_rules_in_exact_order": (
            _binding_rules_json_v1(
                _GATE_PREDECESSOR_BINDING_RULES_IN_EXACT_ORDER_V1
            )
        ),
        "raw_supporting_artifact_payloads_required": True,
        "raw_preallocation_lineage_parameter_name": (
            "raw_preallocation_lineage_payload"
        ),
        "raw_supporting_artifact_payload_tuple_parameter_name": (
            "raw_gate_supporting_artifact_payload_tuples_in_exact_order"
            f"[{spec.ordinal - 1}]"
        ),
        "raw_gate_source_native_payload_tuple_parameter_name": (
            "raw_gate_source_native_payload_tuples_in_exact_order"
            f"[{spec.ordinal - 1}]"
        ),
        "supporting_artifact_binding_rules_in_exact_order": (
            _binding_rules_json_v1(
                _GATE_SUPPORTING_ARTIFACT_BINDING_RULES_IN_EXACT_ORDER_V1
            )
        ),
        "evidence_body_binding_rules_in_exact_order": [
            [
                "gate_ordinal",
                "MUST_EXACT_EQUAL_DESCRIPTOR_ORDINAL_SAFE_INT_BOOL_FORBIDDEN",
            ],
            ["gate_id", "MUST_EXACT_EQUAL_DESCRIPTOR_GATE_ID_ENUM"],
            [
                "gate_schema_id",
                "MUST_EXACT_EQUAL_THIS_DESCRIPTOR_GATE_SCHEMA_ID",
            ],
            [
                "preallocation_lineage_id_and_payload_sha256",
                "MUST_BE_NONEMPTY_CONTENT_ADDRESSED_RAW_LINEAGE_AND_MUST_REMAIN_EXACT_EQUAL_ACROSS_R01_THROUGH_R16",
            ],
            [
                "lineage_scope",
                "MUST_EXACT_EQUAL_DESCRIPTOR_LINEAGE_SCOPE",
            ],
            [
                "campaign_id_or_none_and_attempt_id_or_none",
                _gate_campaign_attempt_policy_v1(spec.ordinal),
            ],
        ],
        "negative_vector_families_in_exact_order": list(
            _NEGATIVE_VECTOR_FAMILIES_V1
        ),
        "current_catalog_gate_success_allowed": False,
        "current_catalog_gate_success_failure_code": (
            GATE_VERIFIER_NOT_IMPLEMENTED_FAILURE_CODE_V1
        ),
    }
    domain = schema_domain_ascii.encode("ascii")
    return {
        **core,
        "gate_schema_id": _content_id_v1(
            domain,
            schema_content_id_prefix,
            core,
        ),
    }


def _stage_json_v1(spec: _StageSpecV1) -> dict[str, object]:
    schema_domain_ascii = (
        _STAGE_SCHEMA_ID_DOMAIN_PREFIX_V1 + f"{spec.ordinal}/ID/V1\x00"
    )
    schema_content_id_prefix = (
        _STAGE_SCHEMA_ID_PREFIX_V1 + f"{spec.ordinal}_v1_"
    )
    event_schema_version = (
        "hegel-machine-phase2b-actual-unsealed-960-authoritative-stage-"
        f"event/{spec.ordinal}/1"
    )
    event_domain_ascii = (
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/STAGE/"
        f"{spec.ordinal}/EVENT/ID/V1\x00"
    )
    event_content_id_prefix = (
        "phase2b_actual_unsealed_960_authoritative_stage_event_"
        f"{spec.ordinal}_v1_"
    )
    signature_slot = _SignatureSlotSpecV1(
        0,
        spec.producer_role,
        f"SIGN_{spec.stage}_STAGE_EVENT",
        1,
        (
            "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/STAGE/"
            f"{spec.ordinal}/SIGN/SIGN_{spec.stage}_STAGE_EVENT/V1\x00"
        ),
    )
    core: dict[str, object] = {
        "ordinal": spec.ordinal,
        "stage": spec.stage,
        "predecessor_nodes_in_exact_order": list(
            spec.predecessor_nodes_in_exact_order
        ),
        "producer_role": spec.producer_role,
        "consumer_roles_in_exact_order": list(
            spec.consumer_roles_in_exact_order
        ),
        "occurs_only_after_r16": spec.occurs_only_after_r16,
        "terminal": spec.terminal,
        "event_schema_version": event_schema_version,
        "event_envelope_fields_in_exact_order": list(
            _STAGE_EVENT_ENVELOPE_FIELDS_V1
        ),
        "event_envelope_field_types_in_exact_order": (
            _field_type_manifest_json_v1(_STAGE_EVENT_ENVELOPE_FIELDS_V1)
        ),
        "event_body_fields_in_exact_order": list(_STAGE_EVENT_BODY_FIELDS_V1),
        "event_body_field_types_in_exact_order": _field_type_manifest_json_v1(
            _STAGE_EVENT_BODY_FIELDS_V1
        ),
        "nested_field_manifests_in_exact_order": [
            {
                "record_name": "detached_signature",
                "fields_in_exact_order": list(_DETACHED_SIGNATURE_FIELDS_V1),
                "field_types_in_exact_order": _field_type_manifest_json_v1(
                    _DETACHED_SIGNATURE_FIELDS_V1
                ),
            }
        ],
        "schema_content_id_domain_ascii": schema_domain_ascii,
        "schema_content_id_prefix": schema_content_id_prefix,
        "content_id_domain_ascii": event_domain_ascii,
        "content_id_prefix": event_content_id_prefix,
        "content_id_preimage_profile": CONTENT_ID_PREIMAGE_PROFILE_V1,
        "content_id_preimage_fields_in_exact_order": list(
            CONTENT_ID_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
        ),
        "signature_preimage_profile": DETACHED_SIGNATURE_PREIMAGE_PROFILE_V1,
        "detached_signature_schema_version": (
            DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_V1
        ),
        "signature_preimage_schema_version": (
            DETACHED_SIGNATURE_PREIMAGE_SCHEMA_VERSION_V1
        ),
        "signature_preimage_artifact_kind": f"stage_event_{spec.ordinal}_v1",
        "signature_algorithm": SIGNATURE_ALGORITHM_V1,
        "signature_preimage_fields_in_exact_order": list(
            DETACHED_SIGNATURE_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
        ),
        "signature_slots_in_exact_order": [
            _signature_slot_json_v1(signature_slot)
        ],
        "signature_cardinality": 1,
        "signer_registry_schema_id": SIGNER_REGISTRY_SCHEMA_OBJECT_V1[
            "signer_registry_schema_id"
        ],
        "raw_signer_registry_payload_required": True,
        "signer_registry_binding_rules_in_exact_order": _binding_rules_json_v1(
            _SIGNER_REGISTRY_BINDING_RULES_IN_EXACT_ORDER_V1
        ),
        "minimum_payload_bytes": 2,
        "maximum_payload_bytes": 16_777_216,
        "minimum_subject_artifact_count": 1,
        "maximum_subject_artifact_count": 64,
        "predecessor_node_cardinality": len(
            spec.predecessor_nodes_in_exact_order
        ),
        "raw_predecessor_evidence_or_event_payloads_required": True,
        "predecessor_node_binding_rules_in_exact_order": (
            _binding_rules_json_v1(
                _STAGE_PREDECESSOR_BINDING_RULES_IN_EXACT_ORDER_V1
            )
        ),
        "raw_subject_artifact_payloads_required": True,
        "subject_artifact_binding_rules_in_exact_order": (
            _binding_rules_json_v1(
                _STAGE_SUBJECT_ARTIFACT_BINDING_RULES_IN_EXACT_ORDER_V1
            )
        ),
        "terminal_disposition_binding_rules_in_exact_order": (
            _binding_rules_json_v1(
                _STAGE_TERMINAL_DISPOSITION_BINDING_RULES_IN_EXACT_ORDER_V1
            )
        ),
        "event_body_binding_rules_in_exact_order": [
            [
                "stage_ordinal",
                "MUST_EXACT_EQUAL_DESCRIPTOR_ORDINAL_SAFE_INT_BOOL_FORBIDDEN",
            ],
            ["stage", "MUST_EXACT_EQUAL_DESCRIPTOR_STAGE_ENUM"],
            [
                "predecessor_node_names_in_exact_order",
                "MUST_EXACT_EQUAL_DESCRIPTOR_PREDECESSOR_NODES_IN_EXACT_ORDER",
            ],
        ],
        "negative_vector_families_in_exact_order": list(
            _NEGATIVE_VECTOR_FAMILIES_V1
        ),
    }
    domain = schema_domain_ascii.encode("ascii")
    return {
        **core,
        "stage_schema_id": _content_id_v1(
            domain,
            schema_content_id_prefix,
            core,
        ),
    }


def _signature_slot_json_v1(spec: _SignatureSlotSpecV1) -> dict[str, object]:
    return {
        "ordinal": spec.ordinal,
        "role": spec.role,
        "purpose": spec.purpose,
        "key_epoch": spec.key_epoch,
        "signature_domain_ascii": spec.signature_domain_ascii,
    }


def _binding_rules_json_v1(
    rules: tuple[tuple[str, tuple[str, ...], str], ...],
) -> list[dict[str, object]]:
    return [
        {
            "rule_id": rule_id,
            "subject_fields_in_exact_order": list(subject_fields),
            "requirement": requirement,
        }
        for rule_id, subject_fields, requirement in rules
    ]


def _package0_referenced_artifact_contract_json_v1(
    spec: _ReferencedArtifactSpecV1,
) -> dict[str, object]:
    return {
        "record_name": spec.artifact_kind,
        "record_schema_version": spec.schema_version,
        "record_envelope_fields_in_exact_order": list(
            _REFERENCED_ARTIFACT_ENVELOPE_FIELDS_V1
        ),
        "record_envelope_field_types_in_exact_order": _field_type_manifest_json_v1(
            _REFERENCED_ARTIFACT_ENVELOPE_FIELDS_V1
        ),
        "record_body_fields_in_exact_order": list(spec.body_fields_in_exact_order),
        "record_body_field_types_in_exact_order": _field_type_manifest_json_v1(
            spec.body_fields_in_exact_order
        ),
        "record_content_id_domain_ascii": spec.content_id_domain_ascii,
        "record_content_id_prefix": spec.content_id_prefix,
        "record_content_id_preimage_profile": CONTENT_ID_PREIMAGE_PROFILE_V1,
        "record_content_id_preimage_fields_in_exact_order": list(
            CONTENT_ID_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
        ),
        "record_content_id_formula": CONTENT_ID_FORMULA_V1,
        "signature_preimage_profile": DETACHED_SIGNATURE_PREIMAGE_PROFILE_V1,
        "detached_signature_schema_version": (
            DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_V1
        ),
        "signature_preimage_schema_version": (
            DETACHED_SIGNATURE_PREIMAGE_SCHEMA_VERSION_V1
        ),
        "signature_preimage_artifact_kind": (
            f"package0_referenced_{spec.artifact_kind}_v1"
        ),
        "signature_preimage_fields_in_exact_order": list(
            DETACHED_SIGNATURE_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
        ),
        "signature_slots_in_exact_order": [
            _signature_slot_json_v1(slot)
            for slot in spec.signature_slots_in_exact_order
        ],
        "signature_cardinality": len(spec.signature_slots_in_exact_order),
        "raw_record_payload_required": True,
        "raw_record_parameter_name": f"raw_{spec.artifact_kind}_payload",
        "outer_body_reference_fields_in_exact_order": [
            f"{spec.artifact_kind}_id",
            f"{spec.artifact_kind}_payload_sha256",
        ],
        "signer_registry_schema_id": SIGNER_REGISTRY_SCHEMA_OBJECT_V1[
            "signer_registry_schema_id"
        ],
        "binding_rules_in_exact_order": _binding_rules_json_v1(
            _PACKAGE0_REFERENCED_ARTIFACT_BINDING_RULES_IN_EXACT_ORDER_V1
        ),
    }


def _nested_record_binding_contracts_json_v1(
    spec: _PackageSpecV1,
) -> list[dict[str, object]]:
    contracts: list[dict[str, object]] = []
    if spec.ordinal == 0:
        contracts.append(
            {
                "record_name": "signed_stage_event_envelope",
                "event_body_record_name": "custodian_ledger_event_body",
                "record_schema_version": (
                    _PACKAGE0_SIGNED_STAGE_EVENT_ENVELOPE_SCHEMA_VERSION_V1
                ),
                "event_body_schema_version": (
                    _PACKAGE0_CUSTODIAN_LEDGER_EVENT_BODY_SCHEMA_VERSION_V1
                ),
                "record_content_id_domain_ascii": _LEDGER_EVENT_ID_DOMAIN_ASCII_V1,
                "record_content_id_prefix": _LEDGER_EVENT_ID_PREFIX_V1,
                "record_content_id_preimage_profile": (
                    "SIGNED_EVENT_ENVELOPE_SCHEMA_BODY_AND_SIGNATURES_EXCLUDING_OWN_CONTENT_ID_V1"
                ),
                "record_content_id_preimage_fields_in_exact_order": list(
                    _LEDGER_EVENT_ID_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
                ),
                "record_content_id_formula": _LEDGER_EVENT_ID_FORMULA_V1,
                "event_signature_preimage_schema_version": (
                    _LEDGER_EVENT_SIGNATURE_PREIMAGE_SCHEMA_VERSION_V1
                ),
                "detached_signature_schema_version": (
                    DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_V1
                ),
                "event_signature_preimage_fields_in_exact_order": list(
                    _LEDGER_EVENT_SIGNATURE_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
                ),
                "event_signature_formula": _LEDGER_EVENT_SIGNATURE_FORMULA_V1,
                "event_signature_slots_in_exact_stage_order": [
                    _signature_slot_json_v1(slot)
                    for slot in _LEDGER_EVENT_SIGNATURE_SLOTS_IN_EXACT_ORDER_V1
                ],
                "event_signature_cardinality_per_envelope": 1,
                "nested_signature_cardinality": 10,
                "required_record_cardinality": 10,
                "raw_record_payloads_required": True,
                "outer_body_reference_fields_in_exact_order": [
                    "signed_stage_event_ids_in_exact_order",
                    "signed_stage_event_payload_sha256s_in_exact_order",
                ],
                "stage_names_in_exact_order": list(STAGE_NAMES_IN_EXACT_ORDER_V1),
                "chain_root_domain_ascii": _LEDGER_EVENT_CHAIN_ROOT_DOMAIN_ASCII_V1,
                "chain_root_formula": _LEDGER_EVENT_CHAIN_ROOT_FORMULA_V1,
                "binding_rules_in_exact_order": _binding_rules_json_v1(
                    _LEDGER_EVENT_BINDING_RULES_IN_EXACT_ORDER_V1
                ),
                "raw_stage_subject_event_payloads_required": True,
                "stage_subject_binding_rules_in_exact_order": (
                    _binding_rules_json_v1(
                        _PACKAGE0_STAGE_SUBJECT_BINDING_RULES_IN_EXACT_ORDER_V1
                    )
                ),
            }
        )
        contracts.extend(
            _package0_referenced_artifact_contract_json_v1(artifact_spec)
            for artifact_spec in (
                _PACKAGE0_REFERENCED_ARTIFACT_SPECS_IN_EXACT_ORDER_V1
            )
        )
        contracts.append(
            {
                "record_name": "package0_referenced_artifact_graph",
                "raw_terminal_stage_subject_event_required": True,
                "raw_referenced_artifact_parameters_in_exact_order": [
                    f"raw_{artifact_spec.artifact_kind}_payload"
                    for artifact_spec in (
                        _PACKAGE0_REFERENCED_ARTIFACT_SPECS_IN_EXACT_ORDER_V1
                    )
                ],
                "binding_rules_in_exact_order": _binding_rules_json_v1(
                    _PACKAGE0_REFERENCED_GRAPH_BINDING_RULES_IN_EXACT_ORDER_V1
                ),
            }
        )
    if spec.ordinal == 10:
        contracts.append(
            {
                "record_name": "denominator_event_envelope",
                "event_body_record_name": "denominator_event_body",
                "record_schema_version": (
                    "hegel-machine-phase2b-actual-unsealed-960-denominator-event/1"
                ),
                "record_content_id_domain_ascii": (
                    _DENOMINATOR_EVENT_ID_DOMAIN_ASCII_V1
                ),
                "record_content_id_prefix": _DENOMINATOR_EVENT_ID_PREFIX_V1,
                "record_content_id_preimage_profile": (
                    CONTENT_ID_PREIMAGE_PROFILE_V1
                ),
                "record_content_id_preimage_fields_in_exact_order": list(
                    CONTENT_ID_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
                ),
                "record_content_id_formula": CONTENT_ID_FORMULA_V1,
                "signature_preimage_profile": (
                    DETACHED_SIGNATURE_PREIMAGE_PROFILE_V1
                ),
                "detached_signature_schema_version": (
                    DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_V1
                ),
                "signature_preimage_schema_version": (
                    DETACHED_SIGNATURE_PREIMAGE_SCHEMA_VERSION_V1
                ),
                "signature_preimage_artifact_kind": (
                    "package10_denominator_event_v1"
                ),
                "signature_preimage_fields_in_exact_order": list(
                    DETACHED_SIGNATURE_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
                ),
                "signature_formula": DETACHED_SIGNATURE_FORMULA_V1,
                "signature_slots_in_exact_order": [
                    _signature_slot_json_v1(
                        _DENOMINATOR_EVENT_SIGNATURE_SLOT_V1
                    )
                ],
                "signature_cardinality_per_envelope": 1,
                "minimum_record_cardinality": (
                    _PACKAGE10_EVENT_SIGNATURE_CARDINALITY_MINIMUM_V1
                ),
                "maximum_record_cardinality": (
                    _PACKAGE10_EVENT_SIGNATURE_CARDINALITY_MAXIMUM_V1
                ),
                "raw_record_payloads_required": True,
                "raw_record_parameter_name": (
                    "raw_package10_denominator_event_payloads_in_exact_order"
                ),
                "outer_body_reference_fields_in_exact_order": [
                    "denominator_event_ids_in_exact_order",
                    "denominator_event_payload_sha256s_in_exact_order",
                    "denominator_event_kinds_in_exact_order",
                ],
                "event_kinds_in_exact_order": list(
                    DENOMINATOR_EVENT_KINDS_IN_EXACT_ORDER_V1
                ),
                "semantic_boundary": (
                    "PRE_SCORE_OPERATIONAL_FACTS_ONLY_NO_TERMINAL_OR_"
                    "SCIENTIFIC_DISPOSITION"
                ),
                "denominator_finalization_stage": "PREDICTIONS_COMMITTED",
                "post_score_terminal_disposition_owner": (
                    "PACKAGE16_FINAL_ATTEMPT_AND_TERMINAL_CONSUMED_STAGE_EVENT"
                ),
                "event_kind_contract_fields_in_exact_order": [
                    "event_kind",
                    "allowed_stages_in_exact_order",
                    "execution_retry_action_policy",
                    "execution_retry_reasons_in_exact_order",
                    "upload_retry_action_policy",
                    "upload_retry_reasons_in_exact_order",
                    "raw_output_policy",
                    "append_applied_policy",
                    "readback_verified_policy",
                    "allowed_first_failure_codes_in_exact_order",
                    "first_failure_code_formula",
                    "fail_closed_formula",
                ],
                "event_kind_contracts_in_exact_order": [
                    {
                        "event_kind": row[0],
                        "allowed_stages_in_exact_order": list(row[1]),
                        "execution_retry_action_policy": row[2],
                        "execution_retry_reasons_in_exact_order": list(row[3]),
                        "upload_retry_action_policy": row[4],
                        "upload_retry_reasons_in_exact_order": list(row[5]),
                        "raw_output_policy": row[6],
                        "append_applied_policy": row[7],
                        "readback_verified_policy": row[8],
                        "allowed_first_failure_codes_in_exact_order": list(
                            row[9]
                        ),
                        "first_failure_code_formula": row[10],
                        "fail_closed_formula": row[11],
                    }
                    for row in DENOMINATOR_EVENT_KIND_CONTRACTS_IN_EXACT_ORDER_V1
                ],
                "execution_retry_cause_rows_in_exact_order": [
                    {
                        "execution_retry_reason": reason,
                        "required_cause_event_kind": event_kind,
                        "required_cause_cardinality": 1,
                        "required_retry_cardinality": 1,
                        "same_attempt_required": True,
                        "cause_before_retry_required": True,
                    }
                    for reason, event_kind in (
                        DENOMINATOR_EXECUTION_RETRY_CAUSE_ROWS_IN_EXACT_ORDER_V1
                    )
                ],
                "upload_retry_cause_policy": (
                    "NO_SEPARATE_CAUSE_EVENT_EXACT_OUTPUT_UPLOAD_FAILURE_REASON_"
                    "AND_BYTE_IDENTICAL_COMMITTED_OUTPUT_ONLY"
                ),
                "binding_rules_in_exact_order": _binding_rules_json_v1(
                    _DENOMINATOR_EVENT_BINDING_RULES_IN_EXACT_ORDER_V1
                ),
            }
        )
        contracts.append(
            {
                "record_name": "attempt_universe_head_envelope",
                "record_schema_version": (
                    "hegel-machine-phase2b-actual-unsealed-960-attempt-"
                    "universe-head/1"
                ),
                "envelope_fields_in_exact_order": list(
                    _ATTEMPT_UNIVERSE_HEAD_ENVELOPE_FIELDS_V1
                ),
                "body_fields_in_exact_order": list(
                    _ATTEMPT_UNIVERSE_HEAD_BODY_FIELDS_V1
                ),
                "record_content_id_domain_ascii": (
                    _ATTEMPT_UNIVERSE_HEAD_ID_DOMAIN_ASCII_V1
                ),
                "record_content_id_prefix": (
                    _ATTEMPT_UNIVERSE_HEAD_ID_PREFIX_V1
                ),
                "record_content_id_preimage_profile": (
                    CONTENT_ID_PREIMAGE_PROFILE_V1
                ),
                "record_content_id_preimage_fields_in_exact_order": list(
                    CONTENT_ID_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
                ),
                "record_content_id_formula": CONTENT_ID_FORMULA_V1,
                "signature_preimage_profile": (
                    DETACHED_SIGNATURE_PREIMAGE_PROFILE_V1
                ),
                "detached_signature_schema_version": (
                    DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_V1
                ),
                "signature_preimage_schema_version": (
                    DETACHED_SIGNATURE_PREIMAGE_SCHEMA_VERSION_V1
                ),
                "signature_preimage_artifact_kind": (
                    "package10_attempt_universe_head_v1"
                ),
                "signature_preimage_fields_in_exact_order": list(
                    DETACHED_SIGNATURE_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
                ),
                "signature_formula": DETACHED_SIGNATURE_FORMULA_V1,
                "signature_slots_in_exact_order": [
                    _signature_slot_json_v1(slot)
                    for slot in (
                        _ATTEMPT_UNIVERSE_HEAD_SIGNATURE_SLOTS_IN_EXACT_ORDER_V1
                    )
                ],
                "signature_cardinality": len(
                    _ATTEMPT_UNIVERSE_HEAD_SIGNATURE_SLOTS_IN_EXACT_ORDER_V1
                ),
                "required_record_cardinality": 1,
                "raw_record_payload_required": True,
                "raw_record_parameter_name": (
                    "raw_attempt_universe_head_payload"
                ),
                "outer_body_reference_fields_in_exact_order": [
                    "attempt_universe_head_id",
                    "attempt_universe_head_payload_sha256",
                ],
                "event_universe_root_domain_ascii": (
                    _ATTEMPT_UNIVERSE_ROOT_DOMAIN_ASCII_V1
                ),
                "event_universe_root_preimage_fields_in_exact_order": list(
                    _ATTEMPT_UNIVERSE_ROOT_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
                ),
                "event_universe_root_formula": _ORDERED_ROOT_FORMULA_V1,
                "binding_rules_in_exact_order": _binding_rules_json_v1(
                    _ATTEMPT_UNIVERSE_HEAD_BINDING_RULES_IN_EXACT_ORDER_V1
                ),
            }
        )
        contracts.append(
            {
                "record_name": "denominator_finalization",
                "record_content_id_domain_ascii": (
                    _DENOMINATOR_FINALIZATION_ID_DOMAIN_ASCII_V1
                ),
                "record_content_id_prefix": (
                    _DENOMINATOR_FINALIZATION_ID_PREFIX_V1
                ),
                "record_content_id_preimage_profile": (
                    "EXACT_NESTED_RECORD_FIELDS_EXCLUDING_OWN_CONTENT_ID_V1"
                ),
                "record_content_id_preimage_fields_in_exact_order": list(
                    _DENOMINATOR_FINALIZATION_ID_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
                ),
                "record_content_id_formula": (
                    _DENOMINATOR_FINALIZATION_ID_FORMULA_V1
                ),
                "signature_preimage_profile": (
                    DETACHED_SIGNATURE_PREIMAGE_PROFILE_V1
                ),
                "detached_signature_schema_version": (
                    DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_V1
                ),
                "signature_preimage_schema_version": (
                    DETACHED_SIGNATURE_PREIMAGE_SCHEMA_VERSION_V1
                ),
                "signature_preimage_artifact_kind": (
                    "package10_denominator_finalization_v1"
                ),
                "signature_preimage_fields_in_exact_order": list(
                    DETACHED_SIGNATURE_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
                ),
                "signature_body_fields_in_exact_order": list(
                    _DENOMINATOR_FINALIZATION_ID_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
                ),
                "signature_formula": DETACHED_SIGNATURE_FORMULA_V1,
                "signature_slots_in_exact_order": [
                    _signature_slot_json_v1(slot)
                    for slot in (
                        _DENOMINATOR_FINALIZATION_SIGNATURE_SLOTS_IN_EXACT_ORDER_V1
                    )
                ],
                "signature_cardinality": len(
                    _DENOMINATOR_FINALIZATION_SIGNATURE_SLOTS_IN_EXACT_ORDER_V1
                ),
                "required_record_cardinality": 1,
                "raw_record_payload_required": True,
                "raw_denominator_event_payloads_required": True,
                "raw_record_parameter_name": (
                    "raw_package10_denominator_finalization_payload"
                ),
                "outer_body_reference_fields_in_exact_order": [
                    "denominator_finalization_id",
                    "denominator_finalization_payload_sha256",
                ],
                "denominator_root_domain_ascii": (
                    _DENOMINATOR_ROOT_DOMAIN_ASCII_V1
                ),
                "denominator_root_preimage_fields_in_exact_order": list(
                    _DENOMINATOR_ROOT_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
                ),
                "denominator_root_formula": _ORDERED_ROOT_FORMULA_V1,
                "binding_rules_in_exact_order": _binding_rules_json_v1(
                    _DENOMINATOR_FINALIZATION_BINDING_RULES_IN_EXACT_ORDER_V1
                ),
            }
        )
    if spec.ordinal == 16:
        contracts.append(
            {
                "record_name": "attempt_record",
                "record_content_id_domain_ascii": (
                    _ATTEMPT_RECORD_ID_DOMAIN_ASCII_V1
                ),
                "record_content_id_prefix": _ATTEMPT_RECORD_ID_PREFIX_V1,
                "record_content_id_preimage_profile": (
                    "EXACT_NESTED_RECORD_FIELDS_EXCLUDING_OWN_CONTENT_ID_V1"
                ),
                "record_content_id_preimage_fields_in_exact_order": list(
                    _ATTEMPT_RECORD_ID_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
                ),
                "record_content_id_formula": _ATTEMPT_RECORD_ID_FORMULA_V1,
                "allowed_record_cardinalities_in_exact_order": [1, 2],
                "raw_record_payloads_required": True,
                "outer_body_reference_fields_in_exact_order": [
                    "attempt_record_ids_in_exact_order",
                    "attempt_record_payload_sha256s_in_exact_order",
                    "attempt_indices_in_exact_order",
                ],
                "raw_package10_envelope_payload_required": True,
                "raw_package10_denominator_event_payloads_required": True,
                "raw_package10_denominator_finalization_payload_required": True,
                "denominator_cross_binding_rules_in_exact_order": (
                    _binding_rules_json_v1(
                        _ATTEMPT_DENOMINATOR_CROSS_BINDING_RULES_IN_EXACT_ORDER_V1
                    )
                ),
                "binding_rules_in_exact_order": _binding_rules_json_v1(
                    _ATTEMPT_RECORD_BINDING_RULES_IN_EXACT_ORDER_V1
                ),
            }
        )
    if spec.ordinal == 15:
        contracts.append(
            {
                "record_name": "gate_result",
                "record_content_id_domain_ascii": _GATE_RESULT_ID_DOMAIN_ASCII_V1,
                "record_content_id_prefix": _GATE_RESULT_ID_PREFIX_V1,
                "record_content_id_preimage_profile": (
                    "EXACT_NESTED_RECORD_FIELDS_EXCLUDING_OWN_CONTENT_ID_V1"
                ),
                "record_content_id_preimage_fields_in_exact_order": list(
                    _GATE_RESULT_ID_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
                ),
                "record_content_id_formula": _GATE_RESULT_ID_FORMULA_V1,
                "required_record_cardinality": 36,
                "raw_record_payloads_required": True,
                "outer_body_reference_fields_in_exact_order": [
                    "gate_result_ids_in_exact_order",
                    "gate_result_payload_sha256s_in_exact_order",
                    "all_gate_results_root_sha256",
                ],
                "wilson_algorithm": (
                    "ONE_SIDED_95_PERCENT_WILSON_BINARY64_EXACT_OPERATION_ORDER_V1"
                ),
                "wilson_z_binary64_hex": (
                    WILSON_ONE_SIDED_95_Z_BINARY64_HEX_V1
                ),
                "wilson_binary64_operation_order_in_exact_order": list(
                    WILSON_BINARY64_OPERATION_ORDER_IN_EXACT_ORDER_V1
                ),
                "frozen_gate_result_plan_in_exact_order": [
                    {
                        "statistical_gate_id": statistical_gate_id,
                        "metric": metric,
                        "scope_axis": scope_axis,
                        "scope_value": scope_value,
                        "fixed_total_count_or_none": fixed_total_count_or_none,
                        "point_threshold_numerator": point_numerator,
                        "point_threshold_denominator": point_denominator,
                        "wilson_threshold_numerator_or_none": wilson_numerator_or_none,
                        "wilson_threshold_denominator_or_none": wilson_denominator_or_none,
                        "minimum_passing_correct_count_or_none": minimum_passing_correct_count_or_none,
                    }
                    for (
                        (
                            statistical_gate_id,
                            metric,
                            scope_axis,
                            scope_value,
                            fixed_total_count_or_none,
                            point_numerator,
                            point_denominator,
                            wilson_numerator_or_none,
                            wilson_denominator_or_none,
                        ),
                        minimum_passing_correct_count_or_none,
                    ) in zip(
                        FROZEN_VALIDATION_GATE_RESULT_PLAN_IN_EXACT_ORDER_V1,
                        FROZEN_VALIDATION_MINIMUM_PASSING_CORRECT_COUNTS_IN_EXACT_ORDER_V1,
                        strict=True,
                    )
                ],
                "cross_result_binding_rules_in_exact_order": _binding_rules_json_v1(
                    _VALIDATION_GATE_CROSS_RESULT_BINDING_RULES_IN_EXACT_ORDER_V1
                ),
                "binding_rules_in_exact_order": _binding_rules_json_v1(
                    _GATE_RESULT_BINDING_RULES_IN_EXACT_ORDER_V1
                ),
            }
        )
    if spec.ordinal == 17:
        contracts.append(
            {
                "record_name": "formal_c1_final_disposition_derivation",
                "accepted_final_dispositions_in_exact_order": list(
                    COMPLETED_RUN_DISPOSITIONS_IN_EXACT_ORDER_V1
                ),
                "raw_gate_evidence_payloads_required": True,
                "required_gate_evidence_cardinality": 16,
                "gate_evidence_ids_root_domain_ascii": (
                    _P17_GATE_EVIDENCE_ROOT_DOMAIN_ASCII_V1
                ),
                "gate_evidence_root_preimage_fields_in_exact_order": list(
                    _P17_GATE_EVIDENCE_ROOT_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
                ),
                "gate_evidence_root_formula": _ORDERED_ROOT_FORMULA_V1,
                "sealed_report_root_domain_ascii": (
                    _P17_SEALED_REPORT_ROOT_DOMAIN_ASCII_V1
                ),
                "sealed_report_root_preimage_fields_in_exact_order": list(
                    _P17_SEALED_REPORT_ROOT_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
                ),
                "sealed_report_root_formula": _ORDERED_ROOT_FORMULA_V1,
                "raw_package0_envelope_payload_required": True,
                "raw_package0_ledger_event_payloads_required": True,
                "raw_package0_stage_subject_event_payloads_required": True,
                "raw_store_if_absent_receipt_payload_required": True,
                "raw_package10_envelope_payload_required": True,
                "raw_package10_denominator_event_payloads_required": True,
                "raw_package10_denominator_finalization_payload_required": True,
                "raw_package15_envelope_payload_required": True,
                "raw_package15_gate_result_payloads_required": True,
                "required_gate_result_cardinality": 36,
                "raw_package16_envelope_payload_required": True,
                "raw_package16_attempt_record_payloads_required": True,
                "allowed_attempt_record_cardinalities_in_exact_order": [1, 2],
                "binding_rules_in_exact_order": _binding_rules_json_v1(
                    _FORMAL_C1_FINAL_DISPOSITION_BINDING_RULES_IN_EXACT_ORDER_V1
                ),
            }
        )
    return contracts


def _package_body_required_literals_v1(spec: _PackageSpecV1) -> dict[str, object]:
    literals: dict[str, object] = {
        "schema_version": spec.schema_version,
        "status": PACKAGE_OUTER_STATUS_V1,
        "claim_level": PACKAGE_OUTER_CLAIM_LEVEL_V1,
        "package_ordinal": spec.ordinal,
        "package_name": spec.package_name,
        "external_evidence_class": spec.required_external_evidence_class,
        "external_evidence_disposition": spec.producer_evidence_disposition,
        "interleaved_dag_root": DAG_ROOT_V1,
    }
    if spec.ordinal in (15, 17):
        literals["overall_gate_count"] = 12
        literals["slice_gate_count"] = 24
    return literals


def _package_json_v1(
    spec: _PackageSpecV1,
    predecessor_package_schema_ids_in_exact_order: tuple[str, ...],
) -> dict[str, object]:
    schema_domain_ascii = (
        _PACKAGE_SCHEMA_ID_DOMAIN_PREFIX_V1 + f"{spec.ordinal}/ID/V1\x00"
    )
    schema_content_id_prefix = (
        _PACKAGE_SCHEMA_ID_PREFIX_V1 + f"{spec.ordinal}_v1_"
    )
    if spec.ordinal == 0:
        nested_signature_minimum = _PACKAGE0_NESTED_SIGNATURE_CARDINALITY_V1
        nested_signature_maximum = _PACKAGE0_NESTED_SIGNATURE_CARDINALITY_V1
        referenced_signature_cardinality = (
            _PACKAGE0_REFERENCED_SIGNATURE_CARDINALITY_V1
        )
    elif spec.ordinal == 10:
        nested_signature_minimum = (
            _PACKAGE10_NESTED_SIGNATURE_CARDINALITY_MINIMUM_V1
        )
        nested_signature_maximum = (
            _PACKAGE10_NESTED_SIGNATURE_CARDINALITY_MAXIMUM_V1
        )
        referenced_signature_cardinality = 0
    else:
        nested_signature_minimum = 0
        nested_signature_maximum = 0
        referenced_signature_cardinality = 0
    nested_signature_cardinality: int | None = (
        nested_signature_minimum
        if nested_signature_minimum == nested_signature_maximum
        else None
    )
    total_signature_minimum = (
        len(spec.signature_slots_in_exact_order) + nested_signature_minimum
    )
    total_signature_maximum = (
        len(spec.signature_slots_in_exact_order) + nested_signature_maximum
    )
    total_signature_cardinality: int | None = (
        total_signature_minimum
        if total_signature_minimum == total_signature_maximum
        else None
    )
    core: dict[str, object] = {
        "ordinal": spec.ordinal,
        "package_name": spec.package_name,
        "purpose": spec.purpose,
        "schema_version": spec.schema_version,
        "envelope_fields_in_exact_order": list(
            spec.envelope_fields_in_exact_order
        ),
        "envelope_field_types_in_exact_order": _field_type_manifest_json_v1(
            spec.envelope_fields_in_exact_order
        ),
        "body_fields_in_exact_order": list(spec.body_fields_in_exact_order),
        "body_field_types_in_exact_order": _field_type_manifest_json_v1(
            spec.body_fields_in_exact_order
        ),
        "nested_field_manifests_in_exact_order": [
            {
                "record_name": record_name,
                "fields_in_exact_order": list(field_names),
                "field_types_in_exact_order": _field_type_manifest_json_v1(
                    field_names
                ),
            }
            for record_name, field_names in spec.nested_field_manifests_in_exact_order
        ],
        "schema_content_id_domain_ascii": schema_domain_ascii,
        "schema_content_id_prefix": schema_content_id_prefix,
        "content_id_domain_ascii": spec.content_id_domain_ascii,
        "content_id_prefix": spec.content_id_prefix,
        "content_id_preimage_profile": CONTENT_ID_PREIMAGE_PROFILE_V1,
        "content_id_preimage_fields_in_exact_order": list(
            CONTENT_ID_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
        ),
        "signature_preimage_profile": DETACHED_SIGNATURE_PREIMAGE_PROFILE_V1,
        "detached_signature_schema_version": (
            DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_V1
        ),
        "signature_preimage_schema_version": (
            DETACHED_SIGNATURE_PREIMAGE_SCHEMA_VERSION_V1
        ),
        "signature_preimage_artifact_kind": (
            f"authoritative_evidence_package_{spec.ordinal}_v1"
        ),
        "signature_algorithm": SIGNATURE_ALGORITHM_V1,
        "signature_preimage_fields_in_exact_order": list(
            DETACHED_SIGNATURE_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
        ),
        "signature_slots_in_exact_order": [
            _signature_slot_json_v1(slot)
            for slot in spec.signature_slots_in_exact_order
        ],
        "signature_cardinality": len(spec.signature_slots_in_exact_order),
        "signer_registry_schema_id": SIGNER_REGISTRY_SCHEMA_OBJECT_V1[
            "signer_registry_schema_id"
        ],
        "raw_signer_registry_payload_required": True,
        "signer_registry_binding_rules_in_exact_order": _binding_rules_json_v1(
            _SIGNER_REGISTRY_BINDING_RULES_IN_EXACT_ORDER_V1
        ),
        "nested_signature_cardinality": nested_signature_cardinality,
        "nested_signature_cardinality_minimum": nested_signature_minimum,
        "nested_signature_cardinality_maximum": nested_signature_maximum,
        "referenced_signature_cardinality": referenced_signature_cardinality,
        "total_evidence_signature_cardinality": total_signature_cardinality,
        "total_evidence_signature_cardinality_minimum": total_signature_minimum,
        "total_evidence_signature_cardinality_maximum": total_signature_maximum,
        "minimum_payload_bytes": spec.minimum_payload_bytes,
        "maximum_payload_bytes": spec.maximum_payload_bytes,
        "minimum_item_count": spec.minimum_item_count,
        "maximum_item_count": spec.maximum_item_count,
        "predecessor_package_ordinals_in_exact_order": list(
            spec.predecessor_package_ordinals_in_exact_order
        ),
        "predecessor_package_schema_ids_in_exact_order": list(
            predecessor_package_schema_ids_in_exact_order
        ),
        "predecessor_package_finalization_stages_in_exact_order": [
            _PACKAGE_FINALIZATION_STAGE_BY_ORDINAL_V1[ordinal]
            for ordinal in spec.predecessor_package_ordinals_in_exact_order
        ],
        "predecessor_runtime_chronology_rule": (
            "EVERY_COMPLETE_PREDECESSOR_PACKAGE_FINALIZATION_STAGE_MUST_NOT_"
            "FOLLOW_THE_CURRENT_PACKAGE_FINALIZATION_STAGE"
        ),
        "predecessor_package_instance_cardinality": len(
            spec.predecessor_package_ordinals_in_exact_order
        ),
        "raw_predecessor_envelope_payloads_required": True,
        "predecessor_instance_binding_rules_in_exact_order": (
            _binding_rules_json_v1(
                _PACKAGE_INSTANCE_LINEAGE_BINDING_RULES_IN_EXACT_ORDER_V1
            )
        ),
        "required_external_evidence_class": (
            spec.required_external_evidence_class
        ),
        "catalog_evidence_disposition": spec.catalog_evidence_disposition,
        "producer_evidence_disposition": spec.producer_evidence_disposition,
        "verifier_success_disposition": spec.verifier_success_disposition,
        "verifier_failure_dispositions_in_exact_order": list(
            spec.verifier_failure_dispositions_in_exact_order
        ),
        "frozen_threshold_ids_in_exact_order": list(
            spec.frozen_threshold_ids_in_exact_order
        ),
        "nested_record_binding_contracts_in_exact_order": (
            _nested_record_binding_contracts_json_v1(spec)
        ),
        "interleaved_dag_root": DAG_ROOT_V1,
        "body_required_literals": _package_body_required_literals_v1(spec),
        "package_outer_binding_rules_in_exact_order": (
            _binding_rules_json_v1(
                _PACKAGE_OUTER_BINDING_RULES_IN_EXACT_ORDER_V1
            )
        ),
        "required_gate_ids_in_exact_order": list(
            spec.required_gate_ids_in_exact_order
        ),
        "required_gate_id_wire_type": _GATE_ID_EXACT_WIRE_TYPE_V1,
        "producer_stage": spec.producer_stage,
        "finalization_stage": spec.finalization_stage,
        "consumer_stages_in_exact_order": list(
            spec.consumer_stages_in_exact_order
        ),
        "terminal_behavior": spec.terminal_behavior,
        "negative_vector_families_in_exact_order": list(
            spec.negative_vector_families_in_exact_order
        ),
        "initial_status": "PACKAGE_SCHEMA_FROZEN_EVIDENCE_NOT_SUPPLIED",
    }
    if spec.ordinal == 17:
        core.update(
            {
                "accepted_final_dispositions_in_exact_order": list(
                    COMPLETED_RUN_DISPOSITIONS_IN_EXACT_ORDER_V1
                ),
                "formal_c1_final_disposition_binding_rules_in_exact_order": (
                    _binding_rules_json_v1(
                        _FORMAL_C1_FINAL_DISPOSITION_BINDING_RULES_IN_EXACT_ORDER_V1
                    )
                ),
                "finalization_operational_dag_root": (
                    FINALIZATION_OPERATIONAL_DAG_ROOT_V1
                ),
            }
        )
    domain = schema_domain_ascii.encode("ascii")
    return {
        **core,
        "package_schema_id": _content_id_v1(
            domain,
            schema_content_id_prefix,
            core,
        ),
    }


def _build_package_schema_objects_v1(
    specs: tuple[_PackageSpecV1, ...],
) -> tuple[dict[str, object], ...]:
    built: list[dict[str, object]] = []
    for spec in specs:
        predecessor_schema_ids = tuple(
            str(built[ordinal]["package_schema_id"])
            for ordinal in spec.predecessor_package_ordinals_in_exact_order
        )
        built.append(_package_json_v1(spec, predecessor_schema_ids))
    return tuple(built)


def _bundle_index_json_v1() -> dict[str, object]:
    signature_slots = _BUNDLE_INDEX_SIGNATURE_SLOTS_IN_EXACT_ORDER_V1
    schema_domain_ascii = _BUNDLE_INDEX_SCHEMA_ID_DOMAIN_V1.decode("ascii")
    core: dict[str, object] = {
        "schema_version": (
            "hegel-machine-phase2b-actual-unsealed-960-authoritative-"
            "evidence-bundle-index/1"
        ),
        "envelope_fields_in_exact_order": list(_BUNDLE_INDEX_ENVELOPE_FIELDS_V1),
        "envelope_field_types_in_exact_order": _field_type_manifest_json_v1(
            _BUNDLE_INDEX_ENVELOPE_FIELDS_V1
        ),
        "body_fields_in_exact_order": list(_BUNDLE_INDEX_BODY_FIELDS_V1),
        "body_field_types_in_exact_order": _field_type_manifest_json_v1(
            _BUNDLE_INDEX_BODY_FIELDS_V1
        ),
        "nested_field_manifests_in_exact_order": [
            {
                "record_name": "detached_signature",
                "fields_in_exact_order": list(_DETACHED_SIGNATURE_FIELDS_V1),
                "field_types_in_exact_order": _field_type_manifest_json_v1(
                    _DETACHED_SIGNATURE_FIELDS_V1
                ),
            }
        ],
        "schema_content_id_domain_ascii": schema_domain_ascii,
        "schema_content_id_prefix": _BUNDLE_INDEX_SCHEMA_ID_PREFIX_V1,
        "content_id_domain_ascii": _BUNDLE_INDEX_CONTENT_ID_DOMAIN_V1,
        "content_id_prefix": _BUNDLE_INDEX_CONTENT_ID_PREFIX_V1,
        "content_id_preimage_profile": CONTENT_ID_PREIMAGE_PROFILE_V1,
        "content_id_preimage_fields_in_exact_order": list(
            CONTENT_ID_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
        ),
        "signature_preimage_profile": DETACHED_SIGNATURE_PREIMAGE_PROFILE_V1,
        "detached_signature_schema_version": (
            DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_V1
        ),
        "signature_preimage_schema_version": (
            DETACHED_SIGNATURE_PREIMAGE_SCHEMA_VERSION_V1
        ),
        "signature_preimage_artifact_kind": (
            "authoritative_evidence_bundle_index_v1"
        ),
        "signature_algorithm": SIGNATURE_ALGORITHM_V1,
        "signature_preimage_fields_in_exact_order": list(
            DETACHED_SIGNATURE_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
        ),
        "signature_slots_in_exact_order": [
            _signature_slot_json_v1(slot) for slot in signature_slots
        ],
        "signature_cardinality": 2,
        "signer_registry_schema_id": SIGNER_REGISTRY_SCHEMA_OBJECT_V1[
            "signer_registry_schema_id"
        ],
        "raw_signer_registry_payload_required": True,
        "signer_registry_binding_rules_in_exact_order": _binding_rules_json_v1(
            _SIGNER_REGISTRY_BINDING_RULES_IN_EXACT_ORDER_V1
        ),
        "predecessor_package_ordinals_in_exact_order": list(range(18)),
        "predecessor_package_schema_ids_in_exact_order": [
            str(item["package_schema_id"])
            for item in PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
        ],
        "package_instance_cardinality": 18,
        "raw_package_envelope_payloads_required": True,
        "package_instance_binding_rules_in_exact_order": _binding_rules_json_v1(
            _BUNDLE_INSTANCE_LINEAGE_BINDING_RULES_IN_EXACT_ORDER_V1
        ),
        "accepted_final_dispositions_in_exact_order": list(
            COMPLETED_RUN_DISPOSITIONS_IN_EXACT_ORDER_V1
        ),
        "final_disposition_binding_rules_in_exact_order": (
            _binding_rules_json_v1(
                _BUNDLE_FINAL_DISPOSITION_BINDING_RULES_IN_EXACT_ORDER_V1
            )
        ),
        "bundle_root_domain_ascii": _BUNDLE_ROOT_DOMAIN_ASCII_V1,
        "bundle_root_preimage_fields_in_exact_order": list(
            _BUNDLE_ROOT_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
        ),
        "bundle_root_formula": _ORDERED_ROOT_FORMULA_V1,
        "finalization_operational_dag_root": (
            FINALIZATION_OPERATIONAL_DAG_ROOT_V1
        ),
        "interleaved_dag_root": DAG_ROOT_V1,
        "body_required_literals": {"interleaved_dag_root": DAG_ROOT_V1},
        "package_cardinality": 18,
        "created_after_stage": "TERMINAL_CONSUMED",
        "terminal_behavior": (
            "POST_PACKAGE_INDEX_BACKWARD_BINDS_PACKAGES_0_THROUGH_17_"
            "WITHOUT_SELF_REFERENCE"
        ),
        "minimum_payload_bytes": 2,
        "maximum_payload_bytes": 16_777_216,
        "negative_vector_families_in_exact_order": list(
            _NEGATIVE_VECTOR_FAMILIES_V1
        ),
    }
    return {
        **core,
        "bundle_index_schema_id": _content_id_v1(
            _BUNDLE_INDEX_SCHEMA_ID_DOMAIN_V1,
            _BUNDLE_INDEX_SCHEMA_ID_PREFIX_V1,
            core,
        ),
    }


SIGNER_REGISTRY_SCHEMA_OBJECT_V1: Final = _signer_registry_schema_json_v1()
PREALLOCATION_LINEAGE_SCHEMA_OBJECT_V1: Final = (
    _preallocation_lineage_schema_json_v1()
)
GATE_SOURCE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1: Final = tuple(
    _gate_source_receipt_schema_json_v1(spec)
    for spec in GATE_SOURCE_SPECS_IN_EXACT_ORDER_V1
)
GATE_SUPPORT_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1: Final = tuple(
    _gate_support_schema_json_v1(spec) for spec in GATE_SPECS_IN_EXACT_ORDER_V1
)
GATE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1: Final = tuple(
    _gate_json_v1(spec) for spec in GATE_SPECS_IN_EXACT_ORDER_V1
)
STAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1: Final = tuple(
    _stage_json_v1(spec) for spec in STAGE_SPECS_IN_EXACT_ORDER_V1
)
PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1: Final = tuple(
    _build_package_schema_objects_v1(PACKAGE_SPECS_IN_EXACT_ORDER_V1)
)
PACKAGE_SCHEMA_DAG_ROOT_V1: Final = _content_id_v1(
    _PACKAGE_SCHEMA_DAG_ROOT_DOMAIN_V1,
    "phase2b_actual_unsealed_960_authoritative_package_schema_dag_root_v1_",
    [
        {
            "ordinal": spec.ordinal,
            "package_schema_id": item["package_schema_id"],
            "predecessor_package_ordinals_in_exact_order": list(
                spec.predecessor_package_ordinals_in_exact_order
            ),
            "predecessor_package_schema_ids_in_exact_order": item[
                "predecessor_package_schema_ids_in_exact_order"
            ],
        }
        for spec, item in zip(
            PACKAGE_SPECS_IN_EXACT_ORDER_V1,
            PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1,
            strict=True,
        )
    ],
)
BUNDLE_INDEX_SCHEMA_OBJECT_V1: Final = _bundle_index_json_v1()

_PACKAGE0_MECHANICS_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1: Final = (
    "exact_raw_bytes_verified",
    "rfc8785_canonical_bytes_verified",
    "content_ids_verified",
    "raw_payload_sha256s_verified",
    "signature_contracts_verified",
    "ledger_chain_verified",
    "stage_subject_binding_verified",
    "referenced_artifact_graph_verified",
    "package0_lineage_verified",
    "terminal_p15_p16_binding_verified",
    "predecessor_absence_verified",
    "private_lineage_verified",
)
_PACKAGE0_NONCLAIM_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1: Final = (
    "package15_verified",
    "package16_verified",
    "gate_admission_verified",
    "r10_satisfied",
    "authority_currentness_verified",
    "owner_non_revocation_verified",
    "external_key_custody_verified",
    "external_durability_verified",
    "external_non_equivocation_verified",
    "global_anti_fork_verified",
    "execution_authorized",
    "actual_960_verified",
    "c1_satisfied",
)
PACKAGE0_VERIFIED_RESULT_FIELDS_IN_EXACT_ORDER_V1: Final = (
    ("schema_version", "str"),
    ("status", "str"),
    ("claim_level", "str"),
    ("disposition", "Package0DispositionV1"),
    ("reason", "Package0ReasonV1"),
    ("catalog_id", "str"),
    ("catalog_payload_sha256", "str"),
    ("package_schema_id", "str"),
    ("package_id", "str"),
    ("package_payload_sha256", "str"),
    ("signer_registry_id", "str"),
    ("signer_registry_payload_sha256", "str"),
    ("signer_registry_entries_root_sha256", "str"),
    ("campaign_id", "str"),
    ("attempt_id", "str"),
    ("chain_root_sha256", "str"),
    ("terminal_signed_stage_event_id", "str"),
    ("terminal_signed_stage_event_payload_sha256", "str"),
    ("ledger_event_count", "int"),
    ("stage_subject_event_count", "int"),
    ("referenced_artifact_count", "int"),
    ("total_signature_count", "int"),
    *((field_name, "bool") for field_name in _PACKAGE0_MECHANICS_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1),
    *((field_name, "bool") for field_name in _PACKAGE0_NONCLAIM_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1),
)
FUTURE_REAL_PACKAGE0_VERIFIED_RESULT_CONTRACT_V1: Final = {
    "result_type_name": (
        "VerifiedPhase2BActualUnsealed960AuthoritativeEvidencePackage0V1"
    ),
    "require_callable_name": (
        "require_verified_phase2b_actual_unsealed_960_authoritative_evidence_package0_v1"
    ),
    "schema_version": (
        "hegel-machine-phase2b-actual-unsealed-960-authoritative-evidence-"
        "verified-package0/1"
    ),
    "status": "VERIFIED_PRESENTED_EXTERNAL_EVIDENCE_NOT_GATE_ADMISSION",
    "claim_level": (
        "PACKAGE0_RAW_CONTRACT_VERIFIED_NOT_GATE_ADMISSION_NOT_R10_NOT_AUTHORITY_"
        "NOT_CURRENTNESS_NOT_DURABILITY_NOT_NON_EQUIVOCATION_NOT_EXECUTION_NOT_C1"
    ),
    "success_disposition": (
        "VERIFIED_PRESENTED_EXTERNAL_EVIDENCE_NOT_GATE_ADMISSION"
    ),
    "success_reason": "EXACT_PACKAGE0_RAW_CONTRACT_VERIFIED",
    "fields_in_exact_order": [
        [field_name, field_type]
        for field_name, field_type in PACKAGE0_VERIFIED_RESULT_FIELDS_IN_EXACT_ORDER_V1
    ],
    "mechanics_boolean_fields_that_may_be_true_in_exact_order": list(
        _PACKAGE0_MECHANICS_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1
    ),
    "nonclaim_boolean_fields_required_exact_false_in_exact_order": list(
        _PACKAGE0_NONCLAIM_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1
    ),
    "private_issuer_only": True,
    "issuance_available_while_verifier_not_implemented": False,
    "copy_deepcopy_pickle_and_mutation_allowed": False,
}

FUTURE_REAL_PACKAGE0_VERIFIER_RAW_ABI_V1: Final = {
    "public_callable_name": (
        "verify_real_phase2b_actual_unsealed_960_authoritative_evidence_package0_v1"
    ),
    "verified_result_contract": FUTURE_REAL_PACKAGE0_VERIFIED_RESULT_CONTRACT_V1,
    "positional_parameters_in_exact_order": [
        ["catalog_payload", "EXACT_BUILTIN_BYTES"],
        ["raw_signer_registry_payload", "EXACT_BUILTIN_BYTES"],
        ["raw_package0_envelope_payload", "EXACT_BUILTIN_BYTES"],
        [
            "raw_package0_ledger_event_payloads_in_exact_order",
            "EXACT_BUILTIN_TUPLE_OF_10_EXACT_BUILTIN_BYTES",
        ],
        [
            "raw_package0_stage_subject_event_payloads_in_exact_order",
            "EXACT_BUILTIN_TUPLE_OF_10_EXACT_BUILTIN_BYTES",
        ],
        ["raw_store_if_absent_receipt_payload", "EXACT_BUILTIN_BYTES"],
        ["raw_logical_time_token_payload", "EXACT_BUILTIN_BYTES"],
        ["raw_governance_checkpoint_payload", "EXACT_BUILTIN_BYTES"],
        ["raw_custodian_checkpoint_payload", "EXACT_BUILTIN_BYTES"],
        ["raw_external_anchor_payload", "EXACT_BUILTIN_BYTES"],
        ["raw_package15_envelope_payload", "EXACT_BUILTIN_BYTES"],
        ["raw_package16_envelope_payload", "EXACT_BUILTIN_BYTES"],
    ],
    "mandatory_keyword_only_byte_parameters_in_exact_order": [
        "out_of_band_expected_catalog_id_bytes",
        "out_of_band_expected_catalog_payload_sha256_digest",
        "out_of_band_expected_signer_registry_id_bytes",
        "out_of_band_expected_signer_registry_payload_sha256_digest",
        "out_of_band_expected_signer_registry_entries_root_sha256_digest",
        "out_of_band_expected_governance_root_key_id_bytes",
        "out_of_band_expected_governance_root_public_key_bytes",
        "out_of_band_expected_custodian_root_key_id_bytes",
        "out_of_band_expected_custodian_root_public_key_bytes",
        "out_of_band_expected_readiness_dag_id_bytes",
        "out_of_band_expected_readiness_dag_payload_sha256_digest",
        "out_of_band_expected_progression_plan_payload_sha256_digest",
        "out_of_band_expected_gap_inventory_source_sha256_digest",
        "out_of_band_expected_statistical_validation_source_sha256_digest",
        "expected_campaign_id_bytes",
        "expected_attempt_id_bytes",
    ],
    "defaults_allowed": False,
    "varargs_or_kwargs_allowed": False,
    "decoded_mapping_path_key_callback_socket_clock_nonce_inputs_allowed": False,
    "implementation_status": REAL_PACKAGE0_VERIFICATION_STATUS_V1,
    "claim_boundary": (
        "FUTURE_RAW_PACKAGE0_VERIFIER_ABI_ONLY_NOT_IMPLEMENTED_PRESENTED_EXTERNAL_"
        "EVIDENCE_NOT_GATE_ADMISSION_NOT_R10_NOT_AUTHORITY_NOT_CURRENTNESS_NOT_"
        "DURABILITY_NOT_NON_EQUIVOCATION_NOT_EXECUTION_NOT_C1"
    ),
}

FUTURE_REAL_BUNDLE_VERIFIER_RAW_ABI_V1: Final = {
    "public_callable_name": (
        "verify_real_phase2b_actual_unsealed_960_authoritative_evidence_bundle_v1"
    ),
    "positional_parameters_in_exact_order": [
        ["catalog_payload", "EXACT_BUILTIN_BYTES"],
        ["raw_signer_registry_payload", "EXACT_BUILTIN_BYTES"],
        [
            "raw_gate_evidence_payloads_in_exact_order",
            "EXACT_BUILTIN_TUPLE_OF_16_EXACT_BUILTIN_BYTES",
        ],
        ["raw_preallocation_lineage_payload", "EXACT_BUILTIN_BYTES"],
        [
            "raw_gate_supporting_artifact_payload_tuples_in_exact_order",
            "EXACT_BUILTIN_TUPLE_OF_16_EXACT_BUILTIN_TUPLES_OF_3_EXACT_BUILTIN_BYTES",
        ],
        [
            "raw_gate_source_native_payload_tuples_in_exact_order",
            "EXACT_BUILTIN_TUPLE_OF_16_DESCRIPTOR_FROZEN_EXACT_BUILTIN_TUPLES_OF_24_4_8_19_5_10_9_7_OR_9_11_13_15_4_6_15_11_13_EXACT_BUILTIN_BYTES",
        ],
        [
            "raw_stage_event_payloads_in_exact_order",
            "EXACT_BUILTIN_TUPLE_OF_10_EXACT_BUILTIN_BYTES",
        ],
        [
            "raw_package_envelope_payloads_in_exact_order",
            "EXACT_BUILTIN_TUPLE_OF_18_EXACT_BUILTIN_BYTES",
        ],
        ["raw_bundle_index_payload", "EXACT_BUILTIN_BYTES"],
        [
            "raw_package0_ledger_event_payloads_in_exact_order",
            "EXACT_BUILTIN_TUPLE_OF_10_EXACT_BUILTIN_BYTES",
        ],
        [
            "raw_package0_stage_subject_event_payloads_in_exact_order",
            "EXACT_BUILTIN_TUPLE_OF_10_EXACT_BUILTIN_BYTES",
        ],
        ["raw_store_if_absent_receipt_payload", "EXACT_BUILTIN_BYTES"],
        ["raw_logical_time_token_payload", "EXACT_BUILTIN_BYTES"],
        ["raw_governance_checkpoint_payload", "EXACT_BUILTIN_BYTES"],
        ["raw_custodian_checkpoint_payload", "EXACT_BUILTIN_BYTES"],
        ["raw_external_anchor_payload", "EXACT_BUILTIN_BYTES"],
        [
            "raw_package10_denominator_event_payloads_in_exact_order",
            "EXACT_BUILTIN_TUPLE_OF_3_TO_9_EXACT_BUILTIN_BYTES",
        ],
        ["raw_attempt_universe_head_payload", "EXACT_BUILTIN_BYTES"],
        [
            "raw_package10_denominator_finalization_payload",
            "EXACT_BUILTIN_BYTES",
        ],
        [
            "raw_package15_gate_result_payloads_in_exact_order",
            "EXACT_BUILTIN_TUPLE_OF_36_EXACT_BUILTIN_BYTES",
        ],
        [
            "raw_package16_attempt_record_payloads_in_exact_order",
            "EXACT_BUILTIN_TUPLE_OF_1_OR_2_EXACT_BUILTIN_BYTES",
        ],
    ],
    "mandatory_keyword_only_byte_parameters_in_exact_order": [
        "out_of_band_expected_catalog_id_bytes",
        "out_of_band_expected_catalog_payload_sha256_digest",
        "out_of_band_expected_signer_registry_id_bytes",
        "out_of_band_expected_signer_registry_payload_sha256_digest",
        "out_of_band_expected_signer_registry_entries_root_sha256_digest",
        "out_of_band_expected_preallocation_lineage_id_bytes",
        "out_of_band_expected_preallocation_lineage_payload_sha256_digest",
        "out_of_band_expected_governance_root_key_id_bytes",
        "out_of_band_expected_governance_root_public_key_bytes",
        "out_of_band_expected_custodian_root_key_id_bytes",
        "out_of_band_expected_custodian_root_public_key_bytes",
        "out_of_band_expected_attempt_universe_head_id_bytes",
        "out_of_band_expected_attempt_universe_head_payload_sha256_digest",
        "out_of_band_expected_readiness_dag_id_bytes",
        "out_of_band_expected_readiness_dag_payload_sha256_digest",
        "out_of_band_expected_progression_plan_payload_sha256_digest",
        "out_of_band_expected_gap_inventory_source_sha256_digest",
        "out_of_band_expected_statistical_validation_source_sha256_digest",
        "expected_campaign_id_bytes",
        "expected_attempt_id_bytes",
    ],
    "duplicate_parameter_equalities_in_exact_order": [
        [
            "raw_stage_event_payloads_in_exact_order",
            "raw_package0_stage_subject_event_payloads_in_exact_order",
            "EXACT_BUILTIN_TUPLE_AND_EACH_INDEX_MUST_BE_BYTE_FOR_BYTE_EQUAL",
        ]
    ],
    "defaults_allowed": False,
    "varargs_or_kwargs_allowed": False,
    "decoded_mapping_path_key_callback_socket_clock_nonce_inputs_allowed": False,
    "claim_boundary": (
        "FUTURE_RAW_BUNDLE_VERIFIER_ABI_ONLY_NOT_IMPLEMENTED_NOT_AUTHORITY_NOT_ACTUAL_NOT_C1"
    ),
}

GATE_SCHEMA_IDS_ROOT_V1: Final = _identifier_sequence_root_v1(
    _GATE_ROOT_DOMAIN_V1,
    "phase2b_actual_unsealed_960_authoritative_gate_schema_root_v1_",
    tuple(str(item["gate_schema_id"]) for item in GATE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1),
)
GATE_SUPPORT_SCHEMA_IDS_ROOT_V1: Final = _identifier_sequence_root_v1(
    _GATE_SUPPORT_SCHEMA_ROOT_DOMAIN_V1,
    "phase2b_actual_unsealed_960_gate_support_schema_root_v1_",
    tuple(
        str(item["gate_support_schema_id"])
        for item in GATE_SUPPORT_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
    ),
)
GATE_SOURCE_SCHEMA_IDS_ROOT_V1: Final = _identifier_sequence_root_v1(
    _GATE_SOURCE_SCHEMA_ROOT_DOMAIN_V1,
    "phase2b_actual_unsealed_960_gate_source_schema_root_v1_",
    tuple(
        str(item["gate_source_schema_id"])
        for item in GATE_SOURCE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
    ),
)
STAGE_SCHEMA_IDS_ROOT_V1: Final = _identifier_sequence_root_v1(
    _STAGE_ROOT_DOMAIN_V1,
    "phase2b_actual_unsealed_960_authoritative_stage_schema_root_v1_",
    tuple(str(item["stage_schema_id"]) for item in STAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1),
)
PACKAGE_SCHEMA_IDS_ROOT_V1: Final = _identifier_sequence_root_v1(
    _PACKAGE_ROOT_DOMAIN_V1,
    "phase2b_actual_unsealed_960_authoritative_package_schema_root_v1_",
    tuple(
        str(item["package_schema_id"])
        for item in PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
    ),
)
def _attempt_policy_json_v1() -> dict[str, object]:
    return {
        name: [list(item) if type(item) is tuple else item for item in value]
        if type(value) is tuple
        else value
        for name, value in ATTEMPT_POLICY_V1
    }


def _catalog_body_v1() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION_V1,
        "status": STATUS_V1,
        "claim_level": CLAIM_LEVEL_V1,
        "canonicalization": CANONICALIZATION_V1,
        "wire_preimage_profiles": {
            "content_id": {
                "profile": CONTENT_ID_PREIMAGE_PROFILE_V1,
                "fields_in_exact_order": list(
                    CONTENT_ID_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
                ),
                "formula": CONTENT_ID_FORMULA_V1,
                "content_id_field_excluded_from_preimage": True,
                "signatures_excluded_from_preimage": True,
            },
            "detached_signature": {
                "profile": DETACHED_SIGNATURE_PREIMAGE_PROFILE_V1,
                "record_schema_version": (
                    DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_V1
                ),
                "schema_version": DETACHED_SIGNATURE_PREIMAGE_SCHEMA_VERSION_V1,
                "algorithm": SIGNATURE_ALGORITHM_V1,
                "fields_in_exact_order": list(
                    DETACHED_SIGNATURE_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
                ),
                "formula": DETACHED_SIGNATURE_FORMULA_V1,
                "excluded_fields_in_exact_order": list(
                    DETACHED_SIGNATURE_EXCLUDED_FIELDS_IN_EXACT_ORDER_V1
                ),
            },
        },
        "exact_special_wire_type_profiles_in_exact_order": [
            {
                "exact_wire_type": exact_wire_type,
                "validation_profile": validation_profile,
                "fullmatch_ascii_regex_or_none": (
                    None if regex == "NONE" else regex
                ),
                "reference_literals_in_exact_order": list(reference_literals),
            }
            for (
                exact_wire_type,
                validation_profile,
                regex,
                reference_literals,
            ) in EXACT_SPECIAL_WIRE_TYPE_PROFILES_IN_EXACT_ORDER_V1
        ],
        "frozen_statistical_thresholds_in_exact_order": [
            {
                "threshold_id": threshold_id,
                "exact_numerator": numerator,
                "exact_denominator": denominator,
                "comparison": comparison,
                "canonical_decimal_or_none": canonical_decimal,
            }
            for (
                threshold_id,
                numerator,
                denominator,
                comparison,
                canonical_decimal,
            ) in FROZEN_STATISTICAL_THRESHOLDS_IN_EXACT_ORDER_V1
        ],
        "frozen_validation_gate_result_plan_in_exact_order": [
            {
                "statistical_gate_id": statistical_gate_id,
                "metric": metric,
                "scope_axis": scope_axis,
                "scope_value": scope_value,
                "fixed_total_count_or_none": fixed_total_count_or_none,
                "point_threshold_numerator": point_numerator,
                "point_threshold_denominator": point_denominator,
                "wilson_threshold_numerator_or_none": wilson_numerator_or_none,
                "wilson_threshold_denominator_or_none": wilson_denominator_or_none,
                "minimum_passing_correct_count_or_none": minimum_passing_correct_count_or_none,
            }
            for (
                (
                    statistical_gate_id,
                    metric,
                    scope_axis,
                    scope_value,
                    fixed_total_count_or_none,
                    point_numerator,
                    point_denominator,
                    wilson_numerator_or_none,
                    wilson_denominator_or_none,
                ),
                minimum_passing_correct_count_or_none,
            ) in zip(
                FROZEN_VALIDATION_GATE_RESULT_PLAN_IN_EXACT_ORDER_V1,
                FROZEN_VALIDATION_MINIMUM_PASSING_CORRECT_COUNTS_IN_EXACT_ORDER_V1,
                strict=True,
            )
        ],
        "inspected_dependencies_in_exact_order": [
            {
                "dependency_name": name,
                "repository_path": path,
                "sha256": sha256,
                "content_id_or_none": content_id,
                "claim_boundary": claim_boundary,
            }
            for name, path, sha256, content_id, claim_boundary
            in INSPECTED_DEPENDENCIES_IN_EXACT_ORDER_V1
        ],
        "gate_schema_objects_in_exact_order": list(
            GATE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
        ),
        "preallocation_lineage_schema_object": (
            PREALLOCATION_LINEAGE_SCHEMA_OBJECT_V1
        ),
        "gate_source_schema_objects_in_exact_order": list(
            GATE_SOURCE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
        ),
        "gate_support_schema_objects_in_exact_order": list(
            GATE_SUPPORT_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
        ),
        "stage_schema_objects_in_exact_order": list(
            STAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
        ),
        "package_schema_objects_in_exact_order": list(
            PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
        ),
        "signer_registry_schema_object": SIGNER_REGISTRY_SCHEMA_OBJECT_V1,
        "signer_registry_schema_id": SIGNER_REGISTRY_SCHEMA_OBJECT_V1[
            "signer_registry_schema_id"
        ],
        "bundle_index_schema_object": BUNDLE_INDEX_SCHEMA_OBJECT_V1,
        "future_real_bundle_verifier_raw_abi": (
            FUTURE_REAL_BUNDLE_VERIFIER_RAW_ABI_V1
        ),
        "future_real_package0_verifier_raw_abi": (
            FUTURE_REAL_PACKAGE0_VERIFIER_RAW_ABI_V1
        ),
        "future_real_package0_verified_result_contract": (
            FUTURE_REAL_PACKAGE0_VERIFIED_RESULT_CONTRACT_V1
        ),
        "gate_schema_ids_root": GATE_SCHEMA_IDS_ROOT_V1,
        "gate_source_schema_ids_root": GATE_SOURCE_SCHEMA_IDS_ROOT_V1,
        "gate_support_schema_ids_root": GATE_SUPPORT_SCHEMA_IDS_ROOT_V1,
        "stage_schema_ids_root": STAGE_SCHEMA_IDS_ROOT_V1,
        "package_schema_ids_root": PACKAGE_SCHEMA_IDS_ROOT_V1,
        "package_schema_dag_root": PACKAGE_SCHEMA_DAG_ROOT_V1,
        "finalization_operational_edges_in_exact_order": [
            list(edge)
            for edge in FINALIZATION_OPERATIONAL_EDGES_IN_EXACT_ORDER_V1
        ],
        "finalization_operational_dag_root": (
            FINALIZATION_OPERATIONAL_DAG_ROOT_V1
        ),
        "bundle_index_schema_id": BUNDLE_INDEX_SCHEMA_OBJECT_V1[
            "bundle_index_schema_id"
        ],
        "interleaved_dag_edges_in_exact_order": [
            list(edge) for edge in INTERLEAVED_DAG_EDGES_IN_EXACT_ORDER_V1
        ],
        "interleaved_dag_root": DAG_ROOT_V1,
        "attempt_policy": _attempt_policy_json_v1(),
        "terminal_disposition_policy_in_exact_order": [
            {"disposition_family": family, "semantics": semantics}
            for family, semantics in TERMINAL_DISPOSITION_POLICY_V1
        ],
        "permanent_nonclaims": dict(_PERMANENT_NONCLAIMS_ORACLE_V1),
        "gate_count": 16,
        "satisfied_gate_count": 0,
        "package_count": 18,
        "stage_count": 10,
        "campaign_id": None,
        "attempt_id": None,
        "execution_authorized": False,
        "real_bundle_verification_status": REAL_BUNDLE_VERIFICATION_STATUS_V1,
        "real_package0_verification_status": (
            REAL_PACKAGE0_VERIFICATION_STATUS_V1
        ),
    }


CATALOG_BODY_V1: Final = _catalog_body_v1()
CATALOG_ID_V1: Final = _content_id_v1(
    _CATALOG_ID_DOMAIN_V1,
    _CATALOG_ID_PREFIX_V1,
    CATALOG_BODY_V1,
)
CATALOG_WRAPPER_V1: Final = {
    "schema_version": SCHEMA_VERSION_V1,
    "body": CATALOG_BODY_V1,
    "catalog_id": CATALOG_ID_V1,
}
CATALOG_PAYLOAD_V1: Final = canonicalize_rfc8785_v1(CATALOG_WRAPPER_V1)
CATALOG_PAYLOAD_SHA256_V1: Final = hashlib.sha256(CATALOG_PAYLOAD_V1).hexdigest()


def _assert_static_catalog_v1() -> None:
    if (
        type(MAXIMUM_CATALOG_PAYLOAD_BYTES_V1) is not int
        or MAXIMUM_CATALOG_PAYLOAD_BYTES_V1 != 2_097_152
        or len(CATALOG_PAYLOAD_V1) > MAXIMUM_CATALOG_PAYLOAD_BYTES_V1
    ):
        raise RuntimeError("catalog payload byte bound differs")
    if tuple(_SIGNATURE_AUTHORIZATION_ROW_FIELDS_V1) != (
        "artifact_kind",
        "signature_ordinal",
        "role",
        "purpose",
        "key_epoch",
        "signature_domain_ascii",
    ):
        raise RuntimeError("signature authorization row keyset drift")
    if len(SIGNATURE_AUTHORIZATION_ROWS_IN_EXACT_ORDER_V1) != 210 or len(
        set(SIGNATURE_AUTHORIZATION_ROWS_IN_EXACT_ORDER_V1)
    ) != 210:
        raise RuntimeError("signature authorization plan cardinality differs")
    for row in SIGNATURE_AUTHORIZATION_ROWS_IN_EXACT_ORDER_V1:
        artifact_kind, signature_ordinal, role, purpose, key_epoch, domain = row
        if (
            type(artifact_kind) is not str
            or not artifact_kind
            or type(signature_ordinal) is not int
            or signature_ordinal < 0
            or role not in SIGNER_REGISTRY_ROLES_IN_EXACT_ORDER_V1
            or type(purpose) is not str
            or not purpose
            or type(key_epoch) is not int
            or key_epoch != 1
            or type(domain) is not str
            or not domain.endswith("\x00")
        ):
            raise RuntimeError("signature authorization row differs")
    registry = SIGNER_REGISTRY_SCHEMA_OBJECT_V1
    if tuple(registry["envelope_fields_in_exact_order"]) != (
        _SIGNER_REGISTRY_ENVELOPE_FIELDS_V1
    ) or tuple(registry["body_fields_in_exact_order"]) != (
        _SIGNER_REGISTRY_BODY_FIELDS_V1
    ):
        raise RuntimeError("signer registry envelope or body keyset differs")
    if registry["required_authorization_rows_in_exact_order"] != [
        dict(zip(_SIGNATURE_AUTHORIZATION_ROW_FIELDS_V1, row, strict=True))
        for row in SIGNATURE_AUTHORIZATION_ROWS_IN_EXACT_ORDER_V1
    ]:
        raise RuntimeError("signer registry authorization plan differs")
    if registry["entry_root_formula"] != (
        "sha256_colon_lowerhex(sha256(entry_root_domain_ascii_bytes || "
        "rfc8785(entries_in_exact_order)))"
    ) or registry["authorization_root_formula"] != (
        "sha256_colon_lowerhex(sha256(authorization_root_domain_ascii_bytes || "
        "rfc8785(authorized_signature_rows_in_exact_order)))"
    ):
        raise RuntimeError("signer registry ordered root formula differs")
    if registry["signer_registry_binding_rules_in_exact_order"] != (
        _binding_rules_json_v1(_SIGNER_REGISTRY_BINDING_RULES_IN_EXACT_ORDER_V1)
    ):
        raise RuntimeError("signer registry raw binding rules differ")
    if registry["detached_signature_schema_version"] != (
        DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_V1
    ):
        raise RuntimeError("detached signature record schema version differs")
    if (
        "out_of_band_expected_signer_registry_entries_root_sha256_digest"
        not in FUTURE_REAL_BUNDLE_VERIFIER_RAW_ABI_V1[
            "mandatory_keyword_only_byte_parameters_in_exact_order"
        ]
    ):
        raise RuntimeError("signer registry entries-root OOB pin is absent")
    if FUTURE_REAL_BUNDLE_VERIFIER_RAW_ABI_V1["public_callable_name"] != (
        "verify_real_phase2b_actual_unsealed_960_authoritative_evidence_bundle_v1"
    ):
        raise RuntimeError("future real bundle verifier callable name differs")
    future_positional_parameters = tuple(
        row[0]
        for row in FUTURE_REAL_BUNDLE_VERIFIER_RAW_ABI_V1[
            "positional_parameters_in_exact_order"
        ]
    )
    if not {
        "raw_preallocation_lineage_payload",
        "raw_gate_supporting_artifact_payload_tuples_in_exact_order",
        "raw_gate_source_native_payload_tuples_in_exact_order",
    }.issubset(future_positional_parameters):
        raise RuntimeError("gate source/support raw evidence ABI input is absent")
    if len(future_positional_parameters) != 21:
        raise RuntimeError("future real bundle positional ABI cardinality differs")
    future_keyword_parameters = tuple(
        FUTURE_REAL_BUNDLE_VERIFIER_RAW_ABI_V1[
            "mandatory_keyword_only_byte_parameters_in_exact_order"
        ]
    )
    if not {
        "out_of_band_expected_preallocation_lineage_id_bytes",
        "out_of_band_expected_preallocation_lineage_payload_sha256_digest",
    }.issubset(future_keyword_parameters):
        raise RuntimeError("preallocation lineage OOB pin pair is absent")
    if not {
        "raw_package10_denominator_event_payloads_in_exact_order",
        "raw_attempt_universe_head_payload",
        "raw_package10_denominator_finalization_payload",
    }.issubset(future_positional_parameters):
        raise RuntimeError("package 10 denominator raw ABI input is absent")
    if not {
        "out_of_band_expected_attempt_universe_head_id_bytes",
        "out_of_band_expected_attempt_universe_head_payload_sha256_digest",
    }.issubset(future_keyword_parameters):
        raise RuntimeError("attempt-universe head OOB pin pair is absent")
    if FUTURE_REAL_BUNDLE_VERIFIER_RAW_ABI_V1[
        "duplicate_parameter_equalities_in_exact_order"
    ] != [
        [
            "raw_stage_event_payloads_in_exact_order",
            "raw_package0_stage_subject_event_payloads_in_exact_order",
            "EXACT_BUILTIN_TUPLE_AND_EACH_INDEX_MUST_BE_BYTE_FOR_BYTE_EQUAL",
        ]
    ]:
        raise RuntimeError("duplicated stage-subject ABI equality differs")
    package0_positional_parameters = tuple(
        row[0]
        for row in FUTURE_REAL_PACKAGE0_VERIFIER_RAW_ABI_V1[
            "positional_parameters_in_exact_order"
        ]
    )
    expected_package0_positional_parameters = (
        "catalog_payload",
        "raw_signer_registry_payload",
        "raw_package0_envelope_payload",
        "raw_package0_ledger_event_payloads_in_exact_order",
        "raw_package0_stage_subject_event_payloads_in_exact_order",
        "raw_store_if_absent_receipt_payload",
        "raw_logical_time_token_payload",
        "raw_governance_checkpoint_payload",
        "raw_custodian_checkpoint_payload",
        "raw_external_anchor_payload",
        "raw_package15_envelope_payload",
        "raw_package16_envelope_payload",
    )
    package0_keyword_parameters = tuple(
        FUTURE_REAL_PACKAGE0_VERIFIER_RAW_ABI_V1[
            "mandatory_keyword_only_byte_parameters_in_exact_order"
        ]
    )
    expected_package0_keyword_parameters = (
        "out_of_band_expected_catalog_id_bytes",
        "out_of_band_expected_catalog_payload_sha256_digest",
        "out_of_band_expected_signer_registry_id_bytes",
        "out_of_band_expected_signer_registry_payload_sha256_digest",
        "out_of_band_expected_signer_registry_entries_root_sha256_digest",
        "out_of_band_expected_governance_root_key_id_bytes",
        "out_of_band_expected_governance_root_public_key_bytes",
        "out_of_band_expected_custodian_root_key_id_bytes",
        "out_of_band_expected_custodian_root_public_key_bytes",
        "out_of_band_expected_readiness_dag_id_bytes",
        "out_of_band_expected_readiness_dag_payload_sha256_digest",
        "out_of_band_expected_progression_plan_payload_sha256_digest",
        "out_of_band_expected_gap_inventory_source_sha256_digest",
        "out_of_band_expected_statistical_validation_source_sha256_digest",
        "expected_campaign_id_bytes",
        "expected_attempt_id_bytes",
    )
    package0_parameter_types = tuple(
        row[1]
        for row in FUTURE_REAL_PACKAGE0_VERIFIER_RAW_ABI_V1[
            "positional_parameters_in_exact_order"
        ]
    )
    if (
        FUTURE_REAL_PACKAGE0_VERIFIER_RAW_ABI_V1["public_callable_name"]
        != "verify_real_phase2b_actual_unsealed_960_authoritative_evidence_package0_v1"
        or package0_positional_parameters
        != expected_package0_positional_parameters
        or package0_keyword_parameters != expected_package0_keyword_parameters
        or package0_parameter_types
        != (
            "EXACT_BUILTIN_BYTES",
            "EXACT_BUILTIN_BYTES",
            "EXACT_BUILTIN_BYTES",
            "EXACT_BUILTIN_TUPLE_OF_10_EXACT_BUILTIN_BYTES",
            "EXACT_BUILTIN_TUPLE_OF_10_EXACT_BUILTIN_BYTES",
            "EXACT_BUILTIN_BYTES",
            "EXACT_BUILTIN_BYTES",
            "EXACT_BUILTIN_BYTES",
            "EXACT_BUILTIN_BYTES",
            "EXACT_BUILTIN_BYTES",
            "EXACT_BUILTIN_BYTES",
            "EXACT_BUILTIN_BYTES",
        )
        or FUTURE_REAL_PACKAGE0_VERIFIER_RAW_ABI_V1["defaults_allowed"]
        is not False
        or FUTURE_REAL_PACKAGE0_VERIFIER_RAW_ABI_V1[
            "varargs_or_kwargs_allowed"
        ]
        is not False
        or FUTURE_REAL_PACKAGE0_VERIFIER_RAW_ABI_V1[
            "decoded_mapping_path_key_callback_socket_clock_nonce_inputs_allowed"
        ]
        is not False
        or FUTURE_REAL_PACKAGE0_VERIFIER_RAW_ABI_V1["implementation_status"]
        != REAL_PACKAGE0_VERIFICATION_STATUS_V1
    ):
        raise RuntimeError("future real package 0 verifier ABI differs")
    if FUTURE_REAL_PACKAGE0_VERIFIED_RESULT_CONTRACT_V1 != (
        FUTURE_REAL_PACKAGE0_VERIFIER_RAW_ABI_V1["verified_result_contract"]
    ) or FUTURE_REAL_PACKAGE0_VERIFIED_RESULT_CONTRACT_V1[
        "fields_in_exact_order"
    ] != [
        [field_name, field_type]
        for field_name, field_type in (
            PACKAGE0_VERIFIED_RESULT_FIELDS_IN_EXACT_ORDER_V1
        )
    ]:
        raise RuntimeError("future package 0 verified result contract differs")
    if tuple(
        FUTURE_REAL_PACKAGE0_VERIFIED_RESULT_CONTRACT_V1[
            "mechanics_boolean_fields_that_may_be_true_in_exact_order"
        ]
    ) != _PACKAGE0_MECHANICS_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1 or tuple(
        FUTURE_REAL_PACKAGE0_VERIFIED_RESULT_CONTRACT_V1[
            "nonclaim_boolean_fields_required_exact_false_in_exact_order"
        ]
    ) != _PACKAGE0_NONCLAIM_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1:
        raise RuntimeError("future package 0 result claim boundary differs")
    if (
        CATALOG_BODY_V1["future_real_package0_verifier_raw_abi"]
        != FUTURE_REAL_PACKAGE0_VERIFIER_RAW_ABI_V1
        or CATALOG_BODY_V1["future_real_package0_verified_result_contract"]
        != FUTURE_REAL_PACKAGE0_VERIFIED_RESULT_CONTRACT_V1
        or CATALOG_BODY_V1["real_package0_verification_status"]
        != REAL_PACKAGE0_VERIFICATION_STATUS_V1
        or CATALOG_BODY_V1["wire_preimage_profiles"]["detached_signature"][
            "record_schema_version"
        ]
        != DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_V1
    ):
        raise RuntimeError("package 0 or signature catalog projection differs")
    if (
        PREALLOCATION_LINEAGE_SCHEMA_OBJECT_V1[
            "signature_slots_in_exact_order"
        ]
        != [
            _signature_slot_json_v1(slot)
            for slot in (
                _PREALLOCATION_LINEAGE_SIGNATURE_SLOTS_IN_EXACT_ORDER_V1
            )
        ]
        or PREALLOCATION_LINEAGE_SCHEMA_OBJECT_V1[
            "binding_rules_in_exact_order"
        ]
        != _binding_rules_json_v1(
            _PREALLOCATION_LINEAGE_BINDING_RULES_IN_EXACT_ORDER_V1
        )
        or PREALLOCATION_LINEAGE_SCHEMA_OBJECT_V1["signature_cardinality"]
        != 4
    ):
        raise RuntimeError("preallocation lineage schema differs")
    if len(GATE_SOURCE_SPECS_IN_EXACT_ORDER_V1) != 16 or len(
        GATE_SOURCE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
    ) != 16 or len(
        {
            item["gate_source_schema_id"]
            for item in GATE_SOURCE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
        }
    ) != 16:
        raise RuntimeError("gate source schema cardinality differs")
    expected_source_cardinalities = (
        (24,),
        (4,),
        (8,),
        (19,),
        (5,),
        (10,),
        (9,),
        (7, 9),
        (11,),
        (13,),
        (15,),
        (4,),
        (6,),
        (15,),
        (11,),
        (13,),
    )
    forbidden_source_truth_fields = {
        "passed",
        "satisfied",
        "current",
        "non_revoked",
        "evidence_observed",
    }
    gate_ordinal_by_id = {
        spec.gate_id: spec.ordinal for spec in GATE_SPECS_IN_EXACT_ORDER_V1
    }
    if len(gate_ordinal_by_id) != 16:
        raise RuntimeError("gate ID to ordinal map differs")
    for source_spec, source_schema, expected_cardinalities in zip(
        GATE_SOURCE_SPECS_IN_EXACT_ORDER_V1,
        GATE_SOURCE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1,
        expected_source_cardinalities,
        strict=True,
    ):
        gate_spec = GATE_SPECS_IN_EXACT_ORDER_V1[source_spec.ordinal - 1]
        expected_dependencies = gate_spec.dependencies_in_exact_order
        source_body_fields = tuple(source_schema["body_fields_in_exact_order"])
        expected_source_literals = {
            "status": "RAW_NATIVE_SOURCE_RECEIPT_PRESENTED",
            "claim_level": "RAW_PROVENANCE_ONLY_NOT_GATE_DECISION",
            "lineage_scope": _gate_lineage_scope_v1(source_spec.ordinal),
            "gate_ordinal": source_spec.ordinal,
            "gate_id": source_spec.gate_id,
            "verification_availability": (
                CURRENT_GATE_VERIFICATION_AVAILABILITY_V1
            ),
            "verifier_dispatch_id": source_spec.verifier_dispatch_id,
            "verifier_source_sha256_or_none": None,
            "producer_observation_sequence": source_spec.ordinal,
            "first_failure_code": (
                GATE_VERIFIER_NOT_IMPLEMENTED_FAILURE_CODE_V1
            ),
        }
        if (
            source_spec.verification_availability
            != CURRENT_GATE_VERIFICATION_AVAILABILITY_V1
            or source_spec.verification_availability
            != "SCHEMA_FROZEN_NOT_VERIFIABLE"
            or source_spec.verifier_source_sha256_or_none is not None
            or source_schema["gate_ordinal"] != source_spec.ordinal
            or source_schema["gate_id"] != source_spec.gate_id
            or source_schema["verification_availability"]
            != source_spec.verification_availability
            or source_schema["verifier_dispatch_id"]
            != source_spec.verifier_dispatch_id
            or source_schema["verifier_source_sha256_or_none"] is not None
            or source_schema["dependencies_in_exact_order"]
            != list(expected_dependencies)
            or source_schema[
                "predecessor_gate_source_receipt_cardinality"
            ]
            != len(expected_dependencies)
            or source_schema[
                "raw_predecessor_gate_source_receipt_payloads_required"
            ]
            is not True
            or len(set(expected_dependencies)) != len(expected_dependencies)
            or any(
                dependency not in gate_ordinal_by_id
                for dependency in expected_dependencies
            )
            or any(
                gate_ordinal_by_id[dependency] >= source_spec.ordinal
                for dependency in expected_dependencies
            )
            or (source_spec.ordinal == 1 and expected_dependencies != ())
            or tuple(source_schema["envelope_fields_in_exact_order"])
            != _GATE_SOURCE_RECEIPT_ENVELOPE_FIELDS_V1
            or source_body_fields != _GATE_SOURCE_RECEIPT_BODY_FIELDS_V1
            or forbidden_source_truth_fields.intersection(source_body_fields)
            or tuple(source_schema["allowed_native_source_cardinalities_in_exact_order"])
            != expected_cardinalities
            or tuple(
                tuple(order)
                for order in source_schema[
                    "allowed_native_source_kind_orders_in_exact_order"
                ]
            )
            != source_spec.allowed_native_source_kind_orders_in_exact_order
            or source_schema["signature_slots_in_exact_order"]
            != [
                _signature_slot_json_v1(slot)
                for slot in _gate_source_receipt_signature_slots_v1(source_spec)
            ]
            or source_schema["signature_cardinality"] != 4
            or source_schema["source_root_preimage_fields_in_exact_order"]
            != list(_GATE_SOURCE_ROOT_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1)
            or source_schema["source_root_formula"] != _ORDERED_ROOT_FORMULA_V1
            or not source_schema["source_root_domain_ascii"].endswith("\x00")
            or source_schema["binding_rules_in_exact_order"]
            != _binding_rules_json_v1(
                _GATE_SOURCE_RECEIPT_BINDING_RULES_IN_EXACT_ORDER_V1
            )
            or source_schema["body_required_literals"]
            != expected_source_literals
        ):
            raise RuntimeError("gate source schema differs")
    source_root_domains = tuple(
        str(item["source_root_domain_ascii"])
        for item in GATE_SOURCE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
    )
    expected_source_root_domains = tuple(
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/GATE/"
        f"{ordinal}/SOURCE_ROOT/V1\x00"
        for ordinal in range(1, 17)
    )
    if source_root_domains != expected_source_root_domains or len(
        set(source_root_domains)
    ) != 16 or any(
        not domain.endswith("\x00") for domain in source_root_domains
    ):
        raise RuntimeError("gate source root domains differ")
    if GATE_SOURCE_SCHEMA_IDS_ROOT_V1 != _identifier_sequence_root_v1(
        _GATE_SOURCE_SCHEMA_ROOT_DOMAIN_V1,
        "phase2b_actual_unsealed_960_gate_source_schema_root_v1_",
        tuple(
            str(item["gate_source_schema_id"])
            for item in GATE_SOURCE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
        ),
    ) or CATALOG_BODY_V1["gate_source_schema_ids_root"] != (
        GATE_SOURCE_SCHEMA_IDS_ROOT_V1
    ) or CATALOG_BODY_V1["gate_source_schema_objects_in_exact_order"] != list(
        GATE_SOURCE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
    ):
        raise RuntimeError("gate source schema root or catalog projection differs")
    if len(GATE_SUPPORT_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1) != 16 or len(
        {
            item["gate_support_schema_id"]
            for item in GATE_SUPPORT_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
        }
    ) != 16:
        raise RuntimeError("gate support schema cardinality differs")
    for spec, source_spec, source_schema, support_schema in zip(
        GATE_SPECS_IN_EXACT_ORDER_V1,
        GATE_SOURCE_SPECS_IN_EXACT_ORDER_V1,
        GATE_SOURCE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1,
        GATE_SUPPORT_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1,
        strict=True,
    ):
        expected_support_literals = {
            "status": "SCHEMA_FROZEN_GATE_SUPPORT_NOT_VERIFIED",
            "claim_level": (
                "PRESENTED_SOURCE_PROVENANCE_ONLY_NOT_GATE_DECISION"
            ),
            "support_artifact_kind": _gate_support_artifact_kind_v1(spec),
            "lineage_scope": _gate_lineage_scope_v1(spec.ordinal),
            "verification_availability": (
                CURRENT_GATE_VERIFICATION_AVAILABILITY_V1
            ),
            "verifier_dispatch_id": source_spec.verifier_dispatch_id,
            "verifier_source_sha256_or_none": None,
            "evidence_requirement": spec.evidence_requirement,
            "evidence_assertions_in_exact_order": [
                spec.evidence_requirement
            ],
            "evidence_payload_format": (
                "CANONICAL_RFC8785_EXACT_GATE_OPERATIONAL_EVIDENCE_OBJECT_HEX_V1"
            ),
            "evidence_observed": False,
            "current": False,
            "non_revoked": False,
            "satisfied": False,
            "first_failure_code": (
                GATE_VERIFIER_NOT_IMPLEMENTED_FAILURE_CODE_V1
            ),
        }
        if (
            support_schema["gate_ordinal"] != spec.ordinal
            or support_schema["gate_id"] != spec.gate_id
            or support_schema["lineage_scope"]
            != _gate_lineage_scope_v1(spec.ordinal)
            or support_schema["campaign_attempt_id_policy"]
            != _gate_campaign_attempt_policy_v1(spec.ordinal)
            or support_schema["gate_support_artifact_kind"]
            != _gate_support_artifact_kind_v1(spec)
            or support_schema["gate_source_schema_id"]
            != source_schema["gate_source_schema_id"]
            or support_schema["gate_source_receipt_artifact_kind"]
            != _gate_source_receipt_artifact_kind_v1(source_spec)
            or support_schema["verification_availability"]
            != "SCHEMA_FROZEN_NOT_VERIFIABLE"
            or support_schema["verifier_dispatch_id"]
            != source_spec.verifier_dispatch_id
            or support_schema["verifier_source_sha256_or_none"] is not None
            or support_schema["evidence_requirement"]
            != spec.evidence_requirement
            or support_schema["required_evidence_assertions_in_exact_order"]
            != [spec.evidence_requirement]
            or support_schema["signature_slots_in_exact_order"]
            != [
                _signature_slot_json_v1(slot)
                for slot in _gate_support_signature_slots_v1(spec)
            ]
            or support_schema["signature_cardinality"] != 4
            or support_schema["evidence_payload_format"]
            != "CANONICAL_RFC8785_EXACT_GATE_OPERATIONAL_EVIDENCE_OBJECT_HEX_V1"
            or tuple(
                support_schema[
                    "inline_operational_evidence_fields_in_exact_order"
                ]
            )
            != _GATE_OPERATIONAL_EVIDENCE_FIELDS_V1
            or support_schema["binding_rules_in_exact_order"]
            != _binding_rules_json_v1(
                _GATE_SUPPORT_ARTIFACT_BINDING_RULES_IN_EXACT_ORDER_V1
            )
            or not support_schema["content_id_domain_ascii"].endswith("\x00")
            or support_schema["success_required_literals"]
            != expected_support_literals
        ):
            raise RuntimeError("gate support schema differs")
    if len(GATE_SPECS_IN_EXACT_ORDER_V1) != 16:
        raise RuntimeError("gate catalog cardinality drift")
    if TERMINAL_DISPOSITIONS_IN_EXACT_ORDER_V1 != (
        "NOT_RUN",
        "BLOCKED",
        "REJECTED",
        "INVALIDATED",
        "COMPLETE_FAIL",
        "COMPLETE_PASS",
    ) or COMPLETED_RUN_DISPOSITIONS_IN_EXACT_ORDER_V1 != (
        "COMPLETE_FAIL",
        "COMPLETE_PASS",
    ):
        raise RuntimeError("terminal disposition vocabulary differs")
    if "PASS" in TERMINAL_DISPOSITIONS_IN_EXACT_ORDER_V1:
        raise RuntimeError("legacy PASS disposition reappeared")
    if tuple(spec.ordinal for spec in GATE_SPECS_IN_EXACT_ORDER_V1) != tuple(
        range(1, 17)
    ):
        raise RuntimeError("gate ordinal drift")
    if any(spec.initial_satisfied for spec in GATE_SPECS_IN_EXACT_ORDER_V1):
        raise RuntimeError("gate catalog elevated a readiness claim")
    for spec, source_spec, source_schema, item in zip(
        GATE_SPECS_IN_EXACT_ORDER_V1,
        GATE_SOURCE_SPECS_IN_EXACT_ORDER_V1,
        GATE_SOURCE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1,
        GATE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1,
        strict=True,
    ):
        gate_body_types = {
            row["field_name"]: row["exact_wire_type"]
            for row in item["evidence_body_field_types_in_exact_order"]
        }
        if gate_body_types["gate_id"] != _GATE_ID_EXACT_WIRE_TYPE_V1:
            raise RuntimeError("gate ID enum wire type drift")
        if gate_body_types["gate_schema_id"] != (
            "EXACT_ASCII_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX"
        ):
            raise RuntimeError("gate schema content-ID wire type drift")
        required_gate_lineage_fields = {
            "preallocation_lineage_id",
            "preallocation_lineage_payload_sha256",
            "lineage_scope",
            "campaign_id_or_none",
            "attempt_id_or_none",
        }
        if not required_gate_lineage_fields.issubset(gate_body_types):
            raise RuntimeError("gate allocation chronology fields are absent")
        if gate_body_types["lineage_scope"] != (
            _GATE_LINEAGE_SCOPE_EXACT_WIRE_TYPE_V1
        ):
            raise RuntimeError("gate lineage-scope wire type differs")
        if gate_body_types["campaign_id_or_none"] != (
            _NULL_OR_CAMPAIGN_ID_EXACT_WIRE_TYPE_V1
        ) or gate_body_types["attempt_id_or_none"] != (
            _NULL_OR_ATTEMPT_ID_EXACT_WIRE_TYPE_V1
        ):
            raise RuntimeError("gate nullable campaign/attempt wire type differs")
        if item["lineage_scope"] != _gate_lineage_scope_v1(spec.ordinal):
            raise RuntimeError("gate lineage scope differs")
        if item["campaign_attempt_id_policy"] != (
            _gate_campaign_attempt_policy_v1(spec.ordinal)
        ):
            raise RuntimeError("gate campaign/attempt chronology policy differs")
        if item["preallocation_lineage_schema_id"] != (
            PREALLOCATION_LINEAGE_SCHEMA_OBJECT_V1[
                "preallocation_lineage_schema_id"
            ]
        ) or item["gate_source_schema_id"] != source_schema[
            "gate_source_schema_id"
        ] or item["gate_source_receipt_artifact_kind"] != (
            _gate_source_receipt_artifact_kind_v1(source_spec)
        ) or item["verification_availability"] != (
            "SCHEMA_FROZEN_NOT_VERIFIABLE"
        ) or item["verifier_dispatch_id"] != (
            source_spec.verifier_dispatch_id
        ) or item["verifier_source_sha256_or_none"] is not None or item[
            "allowed_native_source_kind_orders_in_exact_order"
        ] != [
            list(order)
            for order in source_spec.allowed_native_source_kind_orders_in_exact_order
        ] or item["current_catalog_gate_success_allowed"] is not False or item[
            "current_catalog_gate_success_failure_code"
        ] != GATE_VERIFIER_NOT_IMPLEMENTED_FAILURE_CODE_V1 or item[
            "gate_support_schema_id"
        ] != (
            GATE_SUPPORT_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[
                spec.ordinal - 1
            ]["gate_support_schema_id"]
        ):
            raise RuntimeError("gate support schema projection differs")
        if item["required_supporting_artifact_kinds_in_exact_order"] != [
            "PREALLOCATION_AUTHORITY_LINEAGE_V1",
            _gate_source_receipt_artifact_kind_v1(source_spec),
            _gate_support_artifact_kind_v1(spec),
        ] or item["minimum_supporting_artifact_count"] != 3 or item[
            "maximum_supporting_artifact_count"
        ] != 3:
            raise RuntimeError("gate support artifact cardinality differs")
        if spec.ordinal <= 12 and item["lineage_scope"] != "PREALLOCATION":
            raise RuntimeError("preallocation gate scope differs")
        if spec.ordinal == 13 and item["lineage_scope"] != (
            "ALLOCATION_TRANSITION"
        ):
            raise RuntimeError("R13 allocation transition scope differs")
        if spec.ordinal >= 14 and item["lineage_scope"] != "ALLOCATED":
            raise RuntimeError("post-allocation gate scope differs")
        if item["predecessor_gate_evidence_cardinality"] != len(
            spec.dependencies_in_exact_order
        ):
            raise RuntimeError("gate predecessor evidence cardinality differs")
        if item["raw_predecessor_gate_evidence_payloads_required"] is not True:
            raise RuntimeError("gate raw predecessor requirement differs")
        if item["predecessor_gate_evidence_binding_rules_in_exact_order"] != (
            _binding_rules_json_v1(
                _GATE_PREDECESSOR_BINDING_RULES_IN_EXACT_ORDER_V1
            )
        ):
            raise RuntimeError("gate predecessor evidence binding rules differ")
        if item["raw_supporting_artifact_payloads_required"] is not True:
            raise RuntimeError("gate raw supporting artifact requirement differs")
        if item["raw_preallocation_lineage_parameter_name"] != (
            "raw_preallocation_lineage_payload"
        ) or item[
            "raw_supporting_artifact_payload_tuple_parameter_name"
        ] != (
            "raw_gate_supporting_artifact_payload_tuples_in_exact_order"
            f"[{spec.ordinal - 1}]"
        ) or item[
            "raw_gate_source_native_payload_tuple_parameter_name"
        ] != (
            "raw_gate_source_native_payload_tuples_in_exact_order"
            f"[{spec.ordinal - 1}]"
        ):
            raise RuntimeError("gate raw supporting artifact ABI projection differs")
        if item["supporting_artifact_binding_rules_in_exact_order"] != (
            _binding_rules_json_v1(
                _GATE_SUPPORTING_ARTIFACT_BINDING_RULES_IN_EXACT_ORDER_V1
            )
        ):
            raise RuntimeError("gate supporting artifact binding rules differ")
    if len(STAGE_SPECS_IN_EXACT_ORDER_V1) != 10:
        raise RuntimeError("stage catalog cardinality drift")
    terminals = tuple(spec.stage for spec in STAGE_SPECS_IN_EXACT_ORDER_V1 if spec.terminal)
    if terminals != ("TERMINAL_CONSUMED",):
        raise RuntimeError("terminal stage topology drift")
    for spec, item in zip(
        STAGE_SPECS_IN_EXACT_ORDER_V1,
        STAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1,
        strict=True,
    ):
        if item["predecessor_node_cardinality"] != len(
            spec.predecessor_nodes_in_exact_order
        ):
            raise RuntimeError("stage predecessor node cardinality differs")
        if item["raw_predecessor_evidence_or_event_payloads_required"] is not True:
            raise RuntimeError("stage raw predecessor requirement differs")
        if item["predecessor_node_binding_rules_in_exact_order"] != (
            _binding_rules_json_v1(
                _STAGE_PREDECESSOR_BINDING_RULES_IN_EXACT_ORDER_V1
            )
        ):
            raise RuntimeError("stage predecessor binding rules differ")
        if item["raw_subject_artifact_payloads_required"] is not True:
            raise RuntimeError("stage raw subject requirement differs")
        if item["subject_artifact_binding_rules_in_exact_order"] != (
            _binding_rules_json_v1(
                _STAGE_SUBJECT_ARTIFACT_BINDING_RULES_IN_EXACT_ORDER_V1
            )
        ):
            raise RuntimeError("stage subject binding rules differ")
        if item["terminal_disposition_binding_rules_in_exact_order"] != (
            _binding_rules_json_v1(
                _STAGE_TERMINAL_DISPOSITION_BINDING_RULES_IN_EXACT_ORDER_V1
            )
        ):
            raise RuntimeError("stage terminal disposition rules differ")
    if len(PACKAGE_SPECS_IN_EXACT_ORDER_V1) != 18:
        raise RuntimeError("package catalog cardinality drift")
    if tuple(spec.ordinal for spec in PACKAGE_SPECS_IN_EXACT_ORDER_V1) != tuple(
        range(18)
    ):
        raise RuntimeError("package ordinal drift")
    threshold_ids = tuple(
        row[0] for row in FROZEN_STATISTICAL_THRESHOLDS_IN_EXACT_ORDER_V1
    )
    if threshold_ids != FROZEN_STATISTICAL_THRESHOLD_IDS_IN_EXACT_ORDER_V1:
        raise RuntimeError("frozen statistical threshold ID order differs")
    if len(set(threshold_ids)) != len(threshold_ids):
        raise RuntimeError("frozen statistical threshold ID duplicate")
    for row in FROZEN_STATISTICAL_THRESHOLDS_IN_EXACT_ORDER_V1:
        _, numerator, denominator, comparison, canonical_decimal = row
        if (
            type(numerator) is not int
            or type(denominator) is not int
            or numerator < 0
            or denominator <= 0
            or numerator > denominator
        ):
            raise RuntimeError("frozen statistical threshold rational differs")
        if comparison not in ("AT_LEAST", "AT_MOST", "EXACTLY"):
            raise RuntimeError("frozen statistical comparison differs")
        if canonical_decimal != (
            _canonical_unit_interval_decimal_from_rational_or_none_v1(
                numerator,
                denominator,
            )
        ):
            raise RuntimeError("frozen statistical canonical decimal differs")
    if len(FROZEN_VALIDATION_GATE_RESULT_PLAN_IN_EXACT_ORDER_V1) != 36:
        raise RuntimeError("validation gate-result plan cardinality differs")
    if len(
        FROZEN_VALIDATION_MINIMUM_PASSING_CORRECT_COUNTS_IN_EXACT_ORDER_V1
    ) != 36:
        raise RuntimeError("validation minimum passing-count cardinality differs")
    if len(set(FROZEN_VALIDATION_STATISTICAL_GATE_IDS_IN_EXACT_ORDER_V1)) != 36:
        raise RuntimeError("validation gate-result plan ID uniqueness differs")
    if tuple(
        str(row[2]) for row in FROZEN_VALIDATION_GATE_RESULT_PLAN_IN_EXACT_ORDER_V1[:12]
    ) != ("overall",) * 12:
        raise RuntimeError("validation overall gate-result plan order differs")
    if tuple(
        str(row[2]) for row in FROZEN_VALIDATION_GATE_RESULT_PLAN_IN_EXACT_ORDER_V1[12:30]
    ) != ("family",) * 18 or tuple(
        str(row[2]) for row in FROZEN_VALIDATION_GATE_RESULT_PLAN_IN_EXACT_ORDER_V1[30:]
    ) != ("scale",) * 6:
        raise RuntimeError("validation slice gate-result plan order differs")
    if FROZEN_VALIDATION_MINIMUM_PASSING_CORRECT_COUNTS_IN_EXACT_ORDER_V1[10] != 558:
        raise RuntimeError("preservation composite minimum correct count differs")
    for row in FROZEN_VALIDATION_GATE_RESULT_PLAN_IN_EXACT_ORDER_V1:
        (
            statistical_gate_id,
            metric,
            scope_axis,
            scope_value,
            fixed_total_count_or_none,
            point_numerator,
            point_denominator,
            wilson_numerator_or_none,
            wilson_denominator_or_none,
        ) = row
        if not all(
            type(value) is str and value
            for value in (statistical_gate_id, metric, scope_axis, scope_value)
        ):
            raise RuntimeError("validation gate-result plan string differs")
        if fixed_total_count_or_none is not None and (
            type(fixed_total_count_or_none) is not int
            or fixed_total_count_or_none <= 0
        ):
            raise RuntimeError("validation fixed total differs")
        if (
            type(point_numerator) is not int
            or type(point_denominator) is not int
            or not 0 <= point_numerator <= point_denominator
            or point_denominator <= 0
        ):
            raise RuntimeError("validation point threshold differs")
        if (wilson_numerator_or_none is None) != (
            wilson_denominator_or_none is None
        ):
            raise RuntimeError("validation Wilson threshold null pairing differs")
        if wilson_numerator_or_none is not None and (
            type(wilson_numerator_or_none) is not int
            or type(wilson_denominator_or_none) is not int
            or not 0 <= wilson_numerator_or_none <= wilson_denominator_or_none
            or wilson_denominator_or_none <= 0
        ):
            raise RuntimeError("validation Wilson threshold differs")
    lineage_body_fields = {
        "external_evidence_class",
        "external_evidence_disposition",
        "predecessor_package_instance_ids_in_exact_order",
        "predecessor_package_instance_payload_sha256s_in_exact_order",
        "interleaved_dag_root",
    }
    stage_index = {
        stage: index
        for index, stage in enumerate(STAGE_NAMES_IN_EXACT_ORDER_V1)
    }
    for spec, item in zip(
        PACKAGE_SPECS_IN_EXACT_ORDER_V1,
        PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1,
        strict=True,
    ):
        if any(value >= spec.ordinal for value in spec.predecessor_package_ordinals_in_exact_order):
            raise RuntimeError("package schema contains a self or forward reference")
        if not spec.content_id_domain_ascii.endswith("\x00"):
            raise RuntimeError("package content-ID domain is not NUL terminated")
        if tuple(slot.ordinal for slot in spec.signature_slots_in_exact_order) != tuple(
            range(len(spec.signature_slots_in_exact_order))
        ):
            raise RuntimeError("signature slot ordinal drift")
        if any(not slot.signature_domain_ascii.endswith("\x00") for slot in spec.signature_slots_in_exact_order):
            raise RuntimeError("signature domain is not NUL terminated")
        if spec.producer_stage not in STAGE_NAMES_IN_EXACT_ORDER_V1:
            raise RuntimeError("package producer stage differs")
        if spec.finalization_stage not in STAGE_NAMES_IN_EXACT_ORDER_V1:
            raise RuntimeError("package finalization stage differs")
        if STAGE_NAMES_IN_EXACT_ORDER_V1.index(spec.finalization_stage) < STAGE_NAMES_IN_EXACT_ORDER_V1.index(spec.producer_stage):
            raise RuntimeError("package finalization precedes production")
        expected_predecessor_finalization_stages = tuple(
            _PACKAGE_FINALIZATION_STAGE_BY_ORDINAL_V1[ordinal]
            for ordinal in spec.predecessor_package_ordinals_in_exact_order
        )
        if tuple(
            item["predecessor_package_finalization_stages_in_exact_order"]
        ) != expected_predecessor_finalization_stages:
            raise RuntimeError("package predecessor finalization-stage lineage differs")
        if any(
            stage_index[predecessor_stage]
            > stage_index[spec.finalization_stage]
            for predecessor_stage in expected_predecessor_finalization_stages
        ):
            raise RuntimeError(
                "complete predecessor package finalizes after its consumer"
            )
        if item["predecessor_runtime_chronology_rule"] != (
            "EVERY_COMPLETE_PREDECESSOR_PACKAGE_FINALIZATION_STAGE_MUST_NOT_"
            "FOLLOW_THE_CURRENT_PACKAGE_FINALIZATION_STAGE"
        ):
            raise RuntimeError("package predecessor runtime chronology rule differs")
        if len(set(spec.envelope_fields_in_exact_order)) != len(
            spec.envelope_fields_in_exact_order
        ):
            raise RuntimeError("package envelope field keyset contains a duplicate")
        if len(set(spec.body_fields_in_exact_order)) != len(
            spec.body_fields_in_exact_order
        ):
            raise RuntimeError("package body field keyset contains a duplicate")
        for _, nested_fields in spec.nested_field_manifests_in_exact_order:
            if len(set(nested_fields)) != len(nested_fields):
                raise RuntimeError("package nested field keyset contains a duplicate")
        expected_predecessor_schema_ids = tuple(
            str(PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[ordinal]["package_schema_id"])
            for ordinal in spec.predecessor_package_ordinals_in_exact_order
        )
        actual_predecessor_schema_ids = tuple(
            item["predecessor_package_schema_ids_in_exact_order"]
        )
        if actual_predecessor_schema_ids != expected_predecessor_schema_ids:
            raise RuntimeError("package predecessor schema-ID lineage differs")
        if len(actual_predecessor_schema_ids) != len(
            spec.predecessor_package_ordinals_in_exact_order
        ):
            raise RuntimeError("package predecessor schema-ID cardinality differs")
        expected_predecessor_cardinality = len(
            spec.predecessor_package_ordinals_in_exact_order
        )
        if item["predecessor_package_instance_cardinality"] != (
            expected_predecessor_cardinality
        ):
            raise RuntimeError("package predecessor instance cardinality differs")
        if item["raw_predecessor_envelope_payloads_required"] is not True:
            raise RuntimeError("package raw predecessor requirement differs")
        if item["predecessor_instance_binding_rules_in_exact_order"] != (
            _binding_rules_json_v1(
                _PACKAGE_INSTANCE_LINEAGE_BINDING_RULES_IN_EXACT_ORDER_V1
            )
        ):
            raise RuntimeError("package predecessor instance binding rules differ")
        if not lineage_body_fields.issubset(spec.body_fields_in_exact_order):
            raise RuntimeError("package instance lineage body keyset differs")
        if item["required_external_evidence_class"] != (
            PACKAGE_REQUIRED_EXTERNAL_EVIDENCE_CLASS_V1
        ):
            raise RuntimeError("package external evidence class differs")
        if item["catalog_evidence_disposition"] != (
            PACKAGE_CATALOG_EVIDENCE_DISPOSITION_V1
        ):
            raise RuntimeError("package catalog evidence disposition differs")
        if item["producer_evidence_disposition"] != (
            PACKAGE_PRODUCER_EVIDENCE_DISPOSITION_V1
        ):
            raise RuntimeError("package producer evidence disposition differs")
        if item["verifier_success_disposition"] != (
            PACKAGE_VERIFIER_SUCCESS_DISPOSITION_V1
        ) or tuple(item["verifier_failure_dispositions_in_exact_order"]) != (
            PACKAGE_VERIFIER_FAILURE_DISPOSITIONS_IN_EXACT_ORDER_V1
        ):
            raise RuntimeError("package verifier disposition vocabulary differs")
        if item["interleaved_dag_root"] != DAG_ROOT_V1:
            raise RuntimeError("package interleaved DAG root binding differs")
        if item["body_required_literals"] != _package_body_required_literals_v1(spec):
            raise RuntimeError("package required body literals differ")
        if item["package_outer_binding_rules_in_exact_order"] != (
            _binding_rules_json_v1(
                _PACKAGE_OUTER_BINDING_RULES_IN_EXACT_ORDER_V1
            )
        ):
            raise RuntimeError("package outer claim binding rules differ")
        if item["detached_signature_schema_version"] != (
            DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_V1
        ):
            raise RuntimeError("package detached signature schema differs")
        if any(
            threshold_id not in threshold_ids
            for threshold_id in spec.frozen_threshold_ids_in_exact_order
        ):
            raise RuntimeError("package references an unknown frozen threshold")
        if item["nested_record_binding_contracts_in_exact_order"] != (
            _nested_record_binding_contracts_json_v1(spec)
        ):
            raise RuntimeError("package nested record binding contract differs")
    ledger_contracts = PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[0][
        "nested_record_binding_contracts_in_exact_order"
    ]
    expected_package0_record_names = (
        "signed_stage_event_envelope",
        "store_if_absent_receipt",
        "logical_time_token",
        "governance_checkpoint",
        "custodian_checkpoint",
        "external_anchor",
        "package0_referenced_artifact_graph",
    )
    if tuple(contract["record_name"] for contract in ledger_contracts) != (
        expected_package0_record_names
    ):
        raise RuntimeError("package 0 nested binding contract order differs")
    ledger_contract = ledger_contracts[0]
    if ledger_contract["record_name"] != "signed_stage_event_envelope" or (
        ledger_contract["event_body_record_name"]
        != "custodian_ledger_event_body"
    ):
        raise RuntimeError("signed stage-event record naming differs")
    if (
        ledger_contract["record_schema_version"]
        != _PACKAGE0_SIGNED_STAGE_EVENT_ENVELOPE_SCHEMA_VERSION_V1
        or ledger_contract["event_body_schema_version"]
        != _PACKAGE0_CUSTODIAN_LEDGER_EVENT_BODY_SCHEMA_VERSION_V1
        or ledger_contract["detached_signature_schema_version"]
        != DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_V1
        or ledger_contract["binding_rules_in_exact_order"]
        != _binding_rules_json_v1(
            _LEDGER_EVENT_BINDING_RULES_IN_EXACT_ORDER_V1
        )
        or len(ledger_contract["binding_rules_in_exact_order"]) != 8
    ):
        raise RuntimeError("signed stage-event exact schema contract differs")
    if ledger_contract["record_content_id_domain_ascii"] != (
        _LEDGER_EVENT_ID_DOMAIN_ASCII_V1
    ) or ledger_contract["record_content_id_prefix"] != (
        _LEDGER_EVENT_ID_PREFIX_V1
    ):
        raise RuntimeError("custodian ledger event content-ID differs")
    if ledger_contract["required_record_cardinality"] != 10 or (
        ledger_contract["raw_record_payloads_required"] is not True
    ):
        raise RuntimeError("custodian ledger raw record cardinality differs")
    if tuple(ledger_contract["stage_names_in_exact_order"]) != (
        STAGE_NAMES_IN_EXACT_ORDER_V1
    ):
        raise RuntimeError("custodian ledger stage sequence differs")
    if ledger_contract["event_signature_cardinality_per_envelope"] != 1 or (
        ledger_contract["nested_signature_cardinality"] != 10
    ):
        raise RuntimeError("custodian ledger event signature cardinality differs")
    if tuple(
        slot["purpose"]
        for slot in ledger_contract[
            "event_signature_slots_in_exact_stage_order"
        ]
    ) != tuple(
        f"SIGN_{stage}_LEDGER_EVENT" for stage in STAGE_NAMES_IN_EXACT_ORDER_V1
    ):
        raise RuntimeError("custodian ledger event signature purpose order differs")
    ledger_manifest = {
        row["record_name"]: {
            field["field_name"]: field["exact_wire_type"]
            for field in row["field_types_in_exact_order"]
        }
        for row in PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[0][
            "nested_field_manifests_in_exact_order"
        ]
    }["custodian_ledger_event_body"]
    if ledger_manifest["predecessor_event_id_or_none"] != (
        _NULL_OR_LEDGER_EVENT_ID_EXACT_WIRE_TYPE_V1
    ) or "predecessor_event_id" in ledger_manifest:
        raise RuntimeError("custodian ledger predecessor nullable type differs")
    if len(PACKAGE_SPECS_IN_EXACT_ORDER_V1[0].signature_slots_in_exact_order) != 4:
        raise RuntimeError("package 0 aggregate signature cardinality drift")
    if PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[0][
        "nested_signature_cardinality"
    ] != 16 or PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[0][
        "referenced_signature_cardinality"
    ] != 6:
        raise RuntimeError("package 0 nested signature cardinality drift")
    if (
        PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[0][
            "nested_signature_cardinality_minimum"
        ]
        != 16
        or PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[0][
            "nested_signature_cardinality_maximum"
        ]
        != 16
    ):
        raise RuntimeError("package 0 nested signature range drift")
    if PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[0][
        "total_evidence_signature_cardinality"
    ] != 20:
        raise RuntimeError("package 0 total evidence signature cardinality drift")
    if (
        PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[0][
            "total_evidence_signature_cardinality_minimum"
        ]
        != 20
        or PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[0][
            "total_evidence_signature_cardinality_maximum"
        ]
        != 20
    ):
        raise RuntimeError("package 0 total evidence signature range drift")
    referenced_contracts = ledger_contracts[1:6]
    if tuple(
        contract["record_name"] for contract in referenced_contracts
    ) != tuple(
        spec.artifact_kind
        for spec in _PACKAGE0_REFERENCED_ARTIFACT_SPECS_IN_EXACT_ORDER_V1
    ):
        raise RuntimeError("package 0 referenced artifact order differs")
    if sum(
        contract["signature_cardinality"] for contract in referenced_contracts
    ) != 6:
        raise RuntimeError("package 0 referenced signature cardinality differs")
    package0_manifest = {
        row["record_name"]: tuple(
            field["field_name"] for field in row["field_types_in_exact_order"]
        )
        for row in PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[0][
            "nested_field_manifests_in_exact_order"
        ]
    }
    for record_name in ("store_if_absent_receipt_body", "declared_time_token_body"):
        if not {
            "catalog_id",
            "campaign_id",
            "attempt_id",
        }.issubset(package0_manifest[record_name]):
            raise RuntimeError("package 0 supporting artifact lineage differs")
    if "append_applied" not in package0_manifest[
        "store_if_absent_receipt_body"
    ]:
        raise RuntimeError("store receipt append-applied field absent")
    checkpoint_fields = package0_manifest["checkpoint_body"]
    if (
        "terminal_signed_ledger_event_id" not in checkpoint_fields
        or "terminal_signed_ledger_event_payload_sha256" not in checkpoint_fields
        or "terminal_event_id" in checkpoint_fields
    ):
        raise RuntimeError("checkpoint terminal ledger-event naming differs")
    external_anchor_fields = package0_manifest["external_anchor_body"]
    if not {"anchor_service_id", "anchor_locator"}.issubset(
        external_anchor_fields
    ):
        raise RuntimeError("external anchor service metadata absent")
    external_anchor_contract = ledger_contracts[5]
    if tuple(
        (slot["role"], slot["purpose"])
        for slot in external_anchor_contract["signature_slots_in_exact_order"]
    ) != (
        ("STORE_RECEIPT", "SIGN_EXTERNAL_ANCHOR"),
        ("GOVERNANCE_ANCHOR", "COUNTERSIGN_EXTERNAL_ANCHOR"),
    ):
        raise RuntimeError("external anchor signature slots differ")
    denominator_contracts = PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[10][
        "nested_record_binding_contracts_in_exact_order"
    ]
    if tuple(contract["record_name"] for contract in denominator_contracts) != (
        "denominator_event_envelope",
        "attempt_universe_head_envelope",
        "denominator_finalization",
    ):
        raise RuntimeError("package 10 denominator nested contract differs")
    (
        denominator_event_contract,
        attempt_universe_head_contract,
        denominator_contract,
    ) = denominator_contracts
    if denominator_event_contract["record_content_id_domain_ascii"] != (
        _DENOMINATOR_EVENT_ID_DOMAIN_ASCII_V1
    ) or denominator_event_contract["record_content_id_prefix"] != (
        _DENOMINATOR_EVENT_ID_PREFIX_V1
    ):
        raise RuntimeError("package 10 denominator event content-ID differs")
    if (
        denominator_event_contract["record_content_id_preimage_profile"]
        != CONTENT_ID_PREIMAGE_PROFILE_V1
        or tuple(
            denominator_event_contract[
                "record_content_id_preimage_fields_in_exact_order"
            ]
        )
        != CONTENT_ID_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
        or denominator_event_contract["record_content_id_formula"]
        != CONTENT_ID_FORMULA_V1
    ):
        raise RuntimeError("package 10 denominator event preimage differs")
    if denominator_event_contract["signature_slots_in_exact_order"] != [
        _signature_slot_json_v1(_DENOMINATOR_EVENT_SIGNATURE_SLOT_V1)
    ] or denominator_event_contract["signature_cardinality_per_envelope"] != 1:
        raise RuntimeError("package 10 denominator event signature differs")
    if (
        denominator_event_contract["minimum_record_cardinality"]
        != _PACKAGE10_EVENT_SIGNATURE_CARDINALITY_MINIMUM_V1
        or denominator_event_contract["maximum_record_cardinality"]
        != _PACKAGE10_EVENT_SIGNATURE_CARDINALITY_MAXIMUM_V1
        or denominator_event_contract["raw_record_payloads_required"] is not True
        or denominator_event_contract["raw_record_parameter_name"]
        != "raw_package10_denominator_event_payloads_in_exact_order"
    ):
        raise RuntimeError("package 10 denominator event raw cardinality differs")
    if tuple(
        denominator_event_contract[
            "outer_body_reference_fields_in_exact_order"
        ]
    ) != (
        "denominator_event_ids_in_exact_order",
        "denominator_event_payload_sha256s_in_exact_order",
        "denominator_event_kinds_in_exact_order",
    ) or tuple(denominator_event_contract["event_kinds_in_exact_order"]) != (
        DENOMINATOR_EVENT_KINDS_IN_EXACT_ORDER_V1
    ):
        raise RuntimeError("package 10 denominator event projection differs")
    expected_denominator_kind_contracts = [
        {
            "event_kind": row[0],
            "allowed_stages_in_exact_order": list(row[1]),
            "execution_retry_action_policy": row[2],
            "execution_retry_reasons_in_exact_order": list(row[3]),
            "upload_retry_action_policy": row[4],
            "upload_retry_reasons_in_exact_order": list(row[5]),
            "raw_output_policy": row[6],
            "append_applied_policy": row[7],
            "readback_verified_policy": row[8],
            "allowed_first_failure_codes_in_exact_order": list(row[9]),
            "first_failure_code_formula": row[10],
            "fail_closed_formula": row[11],
        }
        for row in DENOMINATOR_EVENT_KIND_CONTRACTS_IN_EXACT_ORDER_V1
    ]
    if denominator_event_contract["event_kind_contracts_in_exact_order"] != (
        expected_denominator_kind_contracts
    ) or tuple(
        row["event_kind"] for row in expected_denominator_kind_contracts
    ) != DENOMINATOR_EVENT_KINDS_IN_EXACT_ORDER_V1:
        raise RuntimeError("package 10 denominator kind contracts differ")
    if "terminal_disposition_or_none" in _DENOMINATOR_EVENT_BODY_FIELDS_V1:
        raise RuntimeError("package 10 pre-score event carries terminal disposition")
    if any(
        "terminal_disposition" in field_name
        for field_name in denominator_event_contract[
            "event_kind_contract_fields_in_exact_order"
        ]
    ) or "terminal_disposition_policy_in_exact_order" in denominator_event_contract:
        raise RuntimeError("package 10 pre-score event contract carries terminal policy")
    if (
        denominator_event_contract["semantic_boundary"]
        != "PRE_SCORE_OPERATIONAL_FACTS_ONLY_NO_TERMINAL_OR_SCIENTIFIC_DISPOSITION"
        or denominator_event_contract["denominator_finalization_stage"]
        != "PREDICTIONS_COMMITTED"
        or denominator_event_contract["post_score_terminal_disposition_owner"]
        != "PACKAGE16_FINAL_ATTEMPT_AND_TERMINAL_CONSUMED_STAGE_EVENT"
    ):
        raise RuntimeError("package 10 pre-score semantic boundary differs")
    if tuple(
        expected_denominator_kind_contracts[1]["allowed_stages_in_exact_order"]
    ) != ("RUN_STARTED", "RUN_FINISHED", "PREDICTIONS_COMMITTED") or tuple(
        expected_denominator_kind_contracts[2]["allowed_stages_in_exact_order"]
    ) != ("RUN_STARTED", "RUN_FINISHED", "PREDICTIONS_COMMITTED"):
        raise RuntimeError("package 10 pre-score close or append stage differs")
    denominator_kind_contract_by_name = {
        row["event_kind"]: row for row in expected_denominator_kind_contracts
    }
    if any(
        not row["allowed_first_failure_codes_in_exact_order"]
        or len(set(row["allowed_first_failure_codes_in_exact_order"]))
        != len(row["allowed_first_failure_codes_in_exact_order"])
        or type(row["first_failure_code_formula"]) is not str
        or not row["first_failure_code_formula"]
        for row in expected_denominator_kind_contracts
    ):
        raise RuntimeError("package 10 first-failure-code policy is incomplete")
    if (
        denominator_kind_contract_by_name["ATTEMPT_PRE_SCORE_CLOSED"][
            "execution_retry_action_policy"
        ]
        != "EXACT_REEXECUTE_OR_NONE"
        or tuple(
            denominator_kind_contract_by_name["MISSING_OUTPUT"][
                "execution_retry_reasons_in_exact_order"
            ]
        )
        != ("INPUT_TRANSFER_CHECKSUM_MISMATCH",)
        or "XOR" not in denominator_kind_contract_by_name["RETRY"][
            "fail_closed_formula"
        ]
        or denominator_kind_contract_by_name["ATTEMPT_OPEN"][
            "allowed_first_failure_codes_in_exact_order"
        ]
        != ["NONE"]
        or denominator_kind_contract_by_name["MISSING_OUTPUT"][
            "allowed_first_failure_codes_in_exact_order"
        ]
        != ["INPUT_TRANSFER_CHECKSUM_MISMATCH"]
    ):
        raise RuntimeError("package 10 retry summary or cause contract differs")
    denominator_binding_rule_names = tuple(
        row["rule_id"]
        for row in denominator_event_contract["binding_rules_in_exact_order"]
    )
    if "EXECUTION_RETRY_CAUSE_BIJECTION_AND_UPLOAD_RETRY_EXACTNESS" not in (
        denominator_binding_rule_names
    ) or "ATTEMPT_ZERO_OPTIONAL_ONE_TOPOLOGY" not in (
        denominator_binding_rule_names
    ):
        raise RuntimeError("package 10 retry or attempt topology is absent")
    expected_cause_rows = [
        {
            "execution_retry_reason": reason,
            "required_cause_event_kind": event_kind,
            "required_cause_cardinality": 1,
            "required_retry_cardinality": 1,
            "same_attempt_required": True,
            "cause_before_retry_required": True,
        }
        for reason, event_kind in (
            DENOMINATOR_EXECUTION_RETRY_CAUSE_ROWS_IN_EXACT_ORDER_V1
        )
    ]
    if denominator_event_contract[
        "execution_retry_cause_rows_in_exact_order"
    ] != expected_cause_rows or denominator_event_contract[
        "upload_retry_cause_policy"
    ] != (
        "NO_SEPARATE_CAUSE_EVENT_EXACT_OUTPUT_UPLOAD_FAILURE_REASON_"
        "AND_BYTE_IDENTICAL_COMMITTED_OUTPUT_ONLY"
    ):
        raise RuntimeError("package 10 retry cause table differs")
    if denominator_event_contract["binding_rules_in_exact_order"] != (
        _binding_rules_json_v1(_DENOMINATOR_EVENT_BINDING_RULES_IN_EXACT_ORDER_V1)
    ):
        raise RuntimeError("package 10 denominator event binding rules differ")
    if (
        attempt_universe_head_contract["record_content_id_domain_ascii"]
        != _ATTEMPT_UNIVERSE_HEAD_ID_DOMAIN_ASCII_V1
        or attempt_universe_head_contract["record_content_id_prefix"]
        != _ATTEMPT_UNIVERSE_HEAD_ID_PREFIX_V1
        or attempt_universe_head_contract["raw_record_parameter_name"]
        != "raw_attempt_universe_head_payload"
        or attempt_universe_head_contract["signature_slots_in_exact_order"]
        != [
            _signature_slot_json_v1(slot)
            for slot in (
                _ATTEMPT_UNIVERSE_HEAD_SIGNATURE_SLOTS_IN_EXACT_ORDER_V1
            )
        ]
        or attempt_universe_head_contract["binding_rules_in_exact_order"]
        != _binding_rules_json_v1(
            _ATTEMPT_UNIVERSE_HEAD_BINDING_RULES_IN_EXACT_ORDER_V1
        )
        or attempt_universe_head_contract[
            "event_universe_root_domain_ascii"
        ]
        != _ATTEMPT_UNIVERSE_ROOT_DOMAIN_ASCII_V1
        or tuple(
            attempt_universe_head_contract[
                "event_universe_root_preimage_fields_in_exact_order"
            ]
        )
        != _ATTEMPT_UNIVERSE_ROOT_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
        or attempt_universe_head_contract["event_universe_root_formula"]
        != _ORDERED_ROOT_FORMULA_V1
    ):
        raise RuntimeError("package 10 attempt-universe head contract differs")
    if denominator_contract["record_content_id_domain_ascii"] != (
        _DENOMINATOR_FINALIZATION_ID_DOMAIN_ASCII_V1
    ) or denominator_contract["record_content_id_prefix"] != (
        _DENOMINATOR_FINALIZATION_ID_PREFIX_V1
    ):
        raise RuntimeError("package 10 denominator content-ID contract differs")
    if (
        denominator_contract["required_record_cardinality"] != 1
        or denominator_contract["raw_record_payload_required"] is not True
        or denominator_contract["raw_denominator_event_payloads_required"]
        is not True
        or denominator_contract["raw_record_parameter_name"]
        != "raw_package10_denominator_finalization_payload"
    ):
        raise RuntimeError("package 10 denominator raw contract differs")
    if denominator_contract["signature_slots_in_exact_order"] != [
        _signature_slot_json_v1(slot)
        for slot in _DENOMINATOR_FINALIZATION_SIGNATURE_SLOTS_IN_EXACT_ORDER_V1
    ] or denominator_contract["signature_cardinality"] != (
        _PACKAGE10_FINALIZATION_SIGNATURE_CARDINALITY_V1
    ):
        raise RuntimeError("package 10 denominator finalization signatures differ")
    if tuple(
        denominator_contract["outer_body_reference_fields_in_exact_order"]
    ) != (
        "denominator_finalization_id",
        "denominator_finalization_payload_sha256",
    ):
        raise RuntimeError("package 10 denominator outer references differ")
    if denominator_contract["denominator_root_domain_ascii"] != (
        _DENOMINATOR_ROOT_DOMAIN_ASCII_V1
    ) or tuple(
        denominator_contract["denominator_root_preimage_fields_in_exact_order"]
    ) != _DENOMINATOR_ROOT_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1 or (
        denominator_contract["denominator_root_formula"]
        != _ORDERED_ROOT_FORMULA_V1
    ):
        raise RuntimeError("package 10 denominator root contract differs")
    if denominator_contract["binding_rules_in_exact_order"] != (
        _binding_rules_json_v1(
            _DENOMINATOR_FINALIZATION_BINDING_RULES_IN_EXACT_ORDER_V1
        )
    ):
        raise RuntimeError("package 10 denominator binding rules differ")
    package10 = PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[10]
    if _ITEM_CARDINALITY_BY_PACKAGE_V1[10] != (3, 9) or (
        len(PACKAGE_SPECS_IN_EXACT_ORDER_V1[10].signature_slots_in_exact_order)
        != 2
    ):
        raise RuntimeError("package 10 outer cardinality differs")
    if (
        package10["nested_signature_cardinality"] is not None
        or package10["nested_signature_cardinality_minimum"] != 11
        or package10["nested_signature_cardinality_maximum"] != 17
        or package10["referenced_signature_cardinality"] != 0
        or package10["total_evidence_signature_cardinality"] is not None
        or package10["total_evidence_signature_cardinality_minimum"] != 13
        or package10["total_evidence_signature_cardinality_maximum"] != 19
    ):
        raise RuntimeError("package 10 signature cardinality range differs")
    if (
        "package10_denominator_event_v1",
        _DENOMINATOR_EVENT_SIGNATURE_SLOT_V1.ordinal,
        _DENOMINATOR_EVENT_SIGNATURE_SLOT_V1.role,
        _DENOMINATOR_EVENT_SIGNATURE_SLOT_V1.purpose,
        _DENOMINATOR_EVENT_SIGNATURE_SLOT_V1.key_epoch,
        _DENOMINATOR_EVENT_SIGNATURE_SLOT_V1.signature_domain_ascii,
    ) not in SIGNATURE_AUTHORIZATION_ROWS_IN_EXACT_ORDER_V1:
        raise RuntimeError("package 10 denominator authorization row absent")
    for slot in _DENOMINATOR_FINALIZATION_SIGNATURE_SLOTS_IN_EXACT_ORDER_V1:
        if (
            "package10_denominator_finalization_v1",
            slot.ordinal,
            slot.role,
            slot.purpose,
            slot.key_epoch,
            slot.signature_domain_ascii,
        ) not in SIGNATURE_AUTHORIZATION_ROWS_IN_EXACT_ORDER_V1:
            raise RuntimeError(
                "package 10 denominator finalization authorization row absent"
            )
    if len(PACKAGE_SPECS_IN_EXACT_ORDER_V1[17].signature_slots_in_exact_order) != 3:
        raise RuntimeError("package 17 signature cardinality drift")
    if PACKAGE_SPECS_IN_EXACT_ORDER_V1[17].predecessor_package_ordinals_in_exact_order != tuple(
        range(17)
    ):
        raise RuntimeError("package 17 must backward-bind packages 0 through 16")
    if _ITEM_CARDINALITY_BY_PACKAGE_V1[17] != (17, 17):
        raise RuntimeError("package 17 must bind exactly 17 predecessor packages")
    if PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[0][
        "predecessor_package_instance_cardinality"
    ] != 0:
        raise RuntimeError("package 0 predecessor instance cardinality differs")
    if PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[17][
        "predecessor_package_instance_cardinality"
    ] != 17:
        raise RuntimeError("package 17 predecessor instance cardinality differs")
    if "package_instance_ids_in_exact_order" in (
        PACKAGE_SPECS_IN_EXACT_ORDER_V1[17].body_fields_in_exact_order
    ):
        raise RuntimeError("package 17 predecessor-only body keyset drift")
    package17 = PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[17]
    package17_body_types = {
        row["field_name"]: row["exact_wire_type"]
        for row in package17["body_field_types_in_exact_order"]
    }
    if not {
        "gate_evidence_ids_in_exact_order",
        "gate_evidence_payload_sha256s_in_exact_order",
        "all_gate_evidence_root_sha256",
    }.issubset(package17_body_types):
        raise RuntimeError("package 17 readiness gate evidence fields absent")
    if package17_body_types["final_disposition"] != (
        _COMPLETED_RUN_DISPOSITION_EXACT_WIRE_TYPE_V1
    ):
        raise RuntimeError("package 17 accepted disposition type differs")
    if tuple(package17["accepted_final_dispositions_in_exact_order"]) != (
        COMPLETED_RUN_DISPOSITIONS_IN_EXACT_ORDER_V1
    ) or package17[
        "formal_c1_final_disposition_binding_rules_in_exact_order"
    ] != _binding_rules_json_v1(
        _FORMAL_C1_FINAL_DISPOSITION_BINDING_RULES_IN_EXACT_ORDER_V1
    ):
        raise RuntimeError("package 17 final-disposition derivation differs")
    package17_contracts = package17[
        "nested_record_binding_contracts_in_exact_order"
    ]
    if len(package17_contracts) != 1 or package17_contracts[0][
        "record_name"
    ] != "formal_c1_final_disposition_derivation":
        raise RuntimeError("package 17 final derivation contract differs")
    if package17_contracts[0]["required_gate_evidence_cardinality"] != 16 or (
        package17_contracts[0]["required_gate_result_cardinality"] != 36
    ) or tuple(
        package17_contracts[0][
            "allowed_attempt_record_cardinalities_in_exact_order"
        ]
    ) != (1, 2):
        raise RuntimeError("package 17 nested evidence cardinality differs")
    if (
        package17_contracts[0][
            "raw_package10_denominator_event_payloads_required"
        ]
        is not True
        or package17_contracts[0][
            "raw_package10_denominator_finalization_payload_required"
        ]
        is not True
    ):
        raise RuntimeError("package 17 denominator raw binding differs")
    if (
        package17_contracts[0]["gate_evidence_root_formula"]
        != _ORDERED_ROOT_FORMULA_V1
        or package17_contracts[0]["sealed_report_root_formula"]
        != _ORDERED_ROOT_FORMULA_V1
        or tuple(
            package17_contracts[0][
                "gate_evidence_root_preimage_fields_in_exact_order"
            ]
        )
        != _P17_GATE_EVIDENCE_ROOT_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
        or tuple(
            package17_contracts[0][
                "sealed_report_root_preimage_fields_in_exact_order"
            ]
        )
        != _P17_SEALED_REPORT_ROOT_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
    ):
        raise RuntimeError("package 17 ordered root formula differs")
    if BUNDLE_INDEX_SCHEMA_OBJECT_V1[
        "predecessor_package_ordinals_in_exact_order"
    ] != list(range(18)):
        raise RuntimeError("bundle index must backward-bind packages 0 through 17")
    if BUNDLE_INDEX_SCHEMA_OBJECT_V1["package_cardinality"] != 18:
        raise RuntimeError("bundle index package cardinality drift")
    if BUNDLE_INDEX_SCHEMA_OBJECT_V1["package_instance_cardinality"] != 18:
        raise RuntimeError("bundle index instance cardinality drift")
    if BUNDLE_INDEX_SCHEMA_OBJECT_V1[
        "raw_package_envelope_payloads_required"
    ] is not True:
        raise RuntimeError("bundle index raw package requirement differs")
    if BUNDLE_INDEX_SCHEMA_OBJECT_V1[
        "package_instance_binding_rules_in_exact_order"
    ] != _binding_rules_json_v1(
        _BUNDLE_INSTANCE_LINEAGE_BINDING_RULES_IN_EXACT_ORDER_V1
    ):
        raise RuntimeError("bundle package instance binding rules differ")
    if tuple(
        BUNDLE_INDEX_SCHEMA_OBJECT_V1[
            "predecessor_package_schema_ids_in_exact_order"
        ]
    ) != tuple(
        str(item["package_schema_id"])
        for item in PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
    ):
        raise RuntimeError("bundle index package schema-ID lineage differs")
    if not {
        "package_schema_ids_in_exact_order",
        "package_instance_ids_in_exact_order",
        "package_instance_payload_sha256s_in_exact_order",
    }.issubset(BUNDLE_INDEX_SCHEMA_OBJECT_V1["body_fields_in_exact_order"]):
        raise RuntimeError("bundle index package lineage body keyset differs")
    if BUNDLE_INDEX_SCHEMA_OBJECT_V1["interleaved_dag_root"] != DAG_ROOT_V1:
        raise RuntimeError("bundle index interleaved DAG root differs")
    bundle_body_types = {
        row["field_name"]: row["exact_wire_type"]
        for row in BUNDLE_INDEX_SCHEMA_OBJECT_V1[
            "body_field_types_in_exact_order"
        ]
    }
    if bundle_body_types["final_disposition"] != (
        _COMPLETED_RUN_DISPOSITION_EXACT_WIRE_TYPE_V1
    ) or tuple(
        BUNDLE_INDEX_SCHEMA_OBJECT_V1[
            "accepted_final_dispositions_in_exact_order"
        ]
    ) != COMPLETED_RUN_DISPOSITIONS_IN_EXACT_ORDER_V1:
        raise RuntimeError("bundle accepted disposition type differs")
    if BUNDLE_INDEX_SCHEMA_OBJECT_V1[
        "final_disposition_binding_rules_in_exact_order"
    ] != _binding_rules_json_v1(
        _BUNDLE_FINAL_DISPOSITION_BINDING_RULES_IN_EXACT_ORDER_V1
    ):
        raise RuntimeError("bundle final disposition rules differ")
    if BUNDLE_INDEX_SCHEMA_OBJECT_V1["bundle_root_formula"] != (
        _ORDERED_ROOT_FORMULA_V1
    ) or tuple(
        BUNDLE_INDEX_SCHEMA_OBJECT_V1[
            "bundle_root_preimage_fields_in_exact_order"
        ]
    ) != _BUNDLE_ROOT_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1:
        raise RuntimeError("bundle ordered root formula differs")
    expected_finalization_edges = (
        ("package10", "package15"),
        ("package10", "package16"),
        ("package15", "package16"),
        ("package15", "terminal_stage_event"),
        ("package16", "terminal_stage_event"),
        ("terminal_stage_event", "package0"),
        ("package0", "package17"),
        ("package17", "bundle_index"),
    )
    if FINALIZATION_OPERATIONAL_EDGES_IN_EXACT_ORDER_V1 != (
        expected_finalization_edges
    ) or BUNDLE_INDEX_SCHEMA_OBJECT_V1[
        "finalization_operational_dag_root"
    ] != FINALIZATION_OPERATIONAL_DAG_ROOT_V1:
        raise RuntimeError("finalization operational DAG differs")
    finalization_order = {
        node: index
        for index, node in enumerate(
            (
                "package10",
                "package15",
                "package16",
                "terminal_stage_event",
                "package0",
                "package17",
                "bundle_index",
            )
        )
    }
    if any(
        finalization_order[source] >= finalization_order[target]
        for source, target in FINALIZATION_OPERATIONAL_EDGES_IN_EXACT_ORDER_V1
    ):
        raise RuntimeError("finalization operational DAG contains a cycle")
    package10_schema = PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[10]
    package15_schema = PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[15]
    package16_schema = PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[16]
    if (
        package10_schema["producer_stage"] != "RUN_STARTED"
        or package10_schema["finalization_stage"] != "PREDICTIONS_COMMITTED"
        or tuple(package10_schema["consumer_stages_in_exact_order"])
        != ("SCORING_STARTED", "REPORTS_COMMITTED", "TERMINAL_CONSUMED")
    ):
        raise RuntimeError("package 10 pre-score production chronology differs")
    if not (
        stage_index[package10_schema["finalization_stage"]]
        < stage_index[package15_schema["producer_stage"]]
        <= stage_index[package15_schema["finalization_stage"]]
        == stage_index[package16_schema["finalization_stage"]]
    ):
        raise RuntimeError("package 10 pre-score seal does not precede scoring")
    if "terminal_disposition_or_none" in package10_schema[
        "body_fields_in_exact_order"
    ] or "terminal_disposition" in package10_schema["terminal_behavior"]:
        raise RuntimeError("package 10 backward terminal-disposition edge exists")
    nested = {
        row["record_name"]: {
            field["field_name"]: field["exact_wire_type"]
            for field in row["field_types_in_exact_order"]
        }
        for row in PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[15][
            "nested_field_manifests_in_exact_order"
        ]
    }
    gate_result = nested["gate_result"]
    if gate_result["statistical_gate_id"] != (
        _STATISTICAL_GATE_ID_EXACT_WIRE_TYPE_V1
    ):
        raise RuntimeError("statistical gate-result ID enum type drift")
    for field_name in (
        "fixed_total_count_or_none",
        "wilson_threshold_numerator_or_none",
        "wilson_threshold_denominator_or_none",
    ):
        if gate_result[field_name] != _NULL_OR_SAFE_INTEGER_EXACT_WIRE_TYPE_V1:
            raise RuntimeError("gate-result nullable integer wire type drift")
    if gate_result["wilson_passed_or_none"] != _NULL_OR_BOOLEAN_EXACT_WIRE_TYPE_V1:
        raise RuntimeError("gate-result nullable boolean wire type drift")
    if gate_result["wilson_lower_binary64_hex_or_none"] != (
        _NULL_OR_BINARY64_HEX_EXACT_WIRE_TYPE_V1
    ):
        raise RuntimeError("gate-result nullable binary64 wire type drift")
    for field_name in (
        "correct_count",
        "total_count",
        "point_threshold_numerator",
        "point_threshold_denominator",
    ):
        if gate_result[field_name] != (
            "EXACT_JSON_SAFE_INTEGER_MIN_0_BOOL_FORBIDDEN"
        ):
            raise RuntimeError("gate-result exact integer wire type drift")
    if any(
        record_name == "gate_result"
        for record_name, _fields in PACKAGE_SPECS_IN_EXACT_ORDER_V1[
            17
        ].nested_field_manifests_in_exact_order
    ):
        raise RuntimeError("package 17 must not duplicate package 15 gate results")
    gate_result_contracts = PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[15][
        "nested_record_binding_contracts_in_exact_order"
    ]
    if len(gate_result_contracts) != 1:
        raise RuntimeError("gate-result binding contract cardinality differs")
    gate_result_contract = gate_result_contracts[0]
    if gate_result_contract["record_content_id_domain_ascii"] != (
        _GATE_RESULT_ID_DOMAIN_ASCII_V1
    ) or gate_result_contract["record_content_id_prefix"] != (
        _GATE_RESULT_ID_PREFIX_V1
    ):
        raise RuntimeError("gate-result content-ID domain or prefix differs")
    if gate_result_contract["required_record_cardinality"] != 36 or (
        gate_result_contract["raw_record_payloads_required"] is not True
    ):
        raise RuntimeError("gate-result raw record cardinality differs")
    if gate_result_contract["wilson_z_binary64_hex"] != (
        WILSON_ONE_SIDED_95_Z_BINARY64_HEX_V1
    ) or tuple(
        gate_result_contract[
            "wilson_binary64_operation_order_in_exact_order"
        ]
    ) != WILSON_BINARY64_OPERATION_ORDER_IN_EXACT_ORDER_V1:
        raise RuntimeError("Wilson binary64 operation order differs")
    if tuple(
        row["statistical_gate_id"]
        for row in gate_result_contract["frozen_gate_result_plan_in_exact_order"]
    ) != FROZEN_VALIDATION_STATISTICAL_GATE_IDS_IN_EXACT_ORDER_V1:
        raise RuntimeError("gate-result plan order differs")
    if tuple(
        row["minimum_passing_correct_count_or_none"]
        for row in gate_result_contract["frozen_gate_result_plan_in_exact_order"]
    ) != FROZEN_VALIDATION_MINIMUM_PASSING_CORRECT_COUNTS_IN_EXACT_ORDER_V1:
        raise RuntimeError("gate-result minimum passing counts differ")
    if gate_result_contract[
        "cross_result_binding_rules_in_exact_order"
    ] != _binding_rules_json_v1(
        _VALIDATION_GATE_CROSS_RESULT_BINDING_RULES_IN_EXACT_ORDER_V1
    ):
        raise RuntimeError("gate-result cross-result binding rules differ")
    if len(set(FROZEN_VALIDATION_STATISTICAL_GATE_IDS_IN_EXACT_ORDER_V1)) != 36:
        raise RuntimeError("gate-result plan IDs must be unique")
    attempt_nested = {
        row["record_name"]: {
            field["field_name"]: field["exact_wire_type"]
            for field in row["field_types_in_exact_order"]
        }
        for row in PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[16][
            "nested_field_manifests_in_exact_order"
        ]
    }
    attempt_record = attempt_nested["attempt_record"]
    if attempt_record["predecessor_attempt_record_id_or_none"] != (
        _NULL_OR_ATTEMPT_RECORD_ID_EXACT_WIRE_TYPE_V1
    ):
        raise RuntimeError("attempt predecessor nullable content-ID type drift")
    if "predecessor_attempt_record_id" in attempt_record:
        raise RuntimeError("non-null attempt predecessor field is forbidden")
    attempt_contracts = PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[16][
        "nested_record_binding_contracts_in_exact_order"
    ]
    if len(attempt_contracts) != 1 or attempt_contracts[0][
        "record_name"
    ] != "attempt_record":
        raise RuntimeError("attempt record binding contract differs")
    if attempt_contracts[0]["record_content_id_domain_ascii"] != (
        _ATTEMPT_RECORD_ID_DOMAIN_ASCII_V1
    ) or attempt_contracts[0]["record_content_id_prefix"] != (
        _ATTEMPT_RECORD_ID_PREFIX_V1
    ):
        raise RuntimeError("attempt record content-ID domain or prefix differs")
    if attempt_contracts[0]["raw_record_payloads_required"] is not True or tuple(
        attempt_contracts[0]["outer_body_reference_fields_in_exact_order"]
    ) != (
        "attempt_record_ids_in_exact_order",
        "attempt_record_payload_sha256s_in_exact_order",
        "attempt_indices_in_exact_order",
    ):
        raise RuntimeError("attempt record outer raw binding differs")
    if (
        attempt_contracts[0]["raw_package10_envelope_payload_required"]
        is not True
        or attempt_contracts[0][
            "raw_package10_denominator_event_payloads_required"
        ]
        is not True
        or attempt_contracts[0][
            "raw_package10_denominator_finalization_payload_required"
        ]
        is not True
        or attempt_contracts[0][
            "denominator_cross_binding_rules_in_exact_order"
        ]
        != _binding_rules_json_v1(
            _ATTEMPT_DENOMINATOR_CROSS_BINDING_RULES_IN_EXACT_ORDER_V1
        )
    ):
        raise RuntimeError("attempt denominator cross-binding differs")
    attempt_record_binding_rule_names = tuple(
        row["rule_id"]
        for row in attempt_contracts[0]["binding_rules_in_exact_order"]
    )
    if not {
        "ATTEMPT_REEXECUTE_BIDIRECTIONAL_CARDINALITY",
        "FINAL_ATTEMPT_HAS_NO_OUTSTANDING_REEXECUTE",
    }.issubset(attempt_record_binding_rule_names) or (
        "REEXECUTE_ATTEMPT_CARDINALITY_IFF_AND_FINAL_NONE"
        not in {
            row["rule_id"]
            for row in attempt_contracts[0][
                "denominator_cross_binding_rules_in_exact_order"
            ]
        }
    ):
        raise RuntimeError("attempt reexecute bidirectional closure is absent")
    if "attempt_terminal" in STAGE_NAMES_IN_EXACT_ORDER_V1:
        raise RuntimeError("legacy split terminal alias is forbidden")
    if dict(ATTEMPT_POLICY_V1)["attempt_2_exists"] is not False:
        raise RuntimeError("attempt 2 is forbidden")
    nonclaims = CATALOG_BODY_V1["permanent_nonclaims"]
    if type(nonclaims) is not dict:
        raise RuntimeError("catalog permanent nonclaim object type differs")
    if tuple(nonclaims) != PERMANENT_NONCLAIMS_IN_EXACT_ORDER_V1:
        raise RuntimeError("catalog permanent nonclaim keyset or order differs")
    if len(nonclaims) != len(_PERMANENT_NONCLAIMS_ORACLE_V1):
        raise RuntimeError("catalog permanent nonclaim cardinality differs")
    for name in PERMANENT_NONCLAIMS_IN_EXACT_ORDER_V1:
        value = nonclaims[name]
        if type(value) is not bool or value is not False:
            raise RuntimeError("catalog permanent nonclaim value differs")
    if canonicalize_rfc8785_v1(nonclaims) != canonicalize_rfc8785_v1(
        _PERMANENT_NONCLAIMS_ORACLE_V1
    ):
        raise RuntimeError("catalog permanent nonclaim canonical bytes differ")
    schema_domains_and_prefixes = tuple(
        (
            str(item["schema_content_id_domain_ascii"]),
            str(item["schema_content_id_prefix"]),
        )
        for item in (
            SIGNER_REGISTRY_SCHEMA_OBJECT_V1,
            PREALLOCATION_LINEAGE_SCHEMA_OBJECT_V1,
            *GATE_SOURCE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1,
            *GATE_SUPPORT_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1,
            *GATE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1,
            *STAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1,
            *PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1,
            BUNDLE_INDEX_SCHEMA_OBJECT_V1,
        )
    )
    if len(set(schema_domains_and_prefixes)) != 79:
        raise RuntimeError("schema content-ID domain/prefix pairs are not unique")
    if len({domain for domain, _ in schema_domains_and_prefixes}) != 79:
        raise RuntimeError("schema content-ID domains are not unique")
    if len({prefix for _, prefix in schema_domains_and_prefixes}) != 79:
        raise RuntimeError("schema content-ID prefixes are not unique")
    artifact_domains_and_prefixes = tuple(
        (str(item["content_id_domain_ascii"]), str(item["content_id_prefix"]))
        for item in (
            SIGNER_REGISTRY_SCHEMA_OBJECT_V1,
            PREALLOCATION_LINEAGE_SCHEMA_OBJECT_V1,
            *GATE_SOURCE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1,
            *GATE_SUPPORT_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1,
            *GATE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1,
            *STAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1,
            *PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1,
            BUNDLE_INDEX_SCHEMA_OBJECT_V1,
        )
    )
    if len(set(artifact_domains_and_prefixes)) != 79:
        raise RuntimeError("artifact content-ID domain/prefix pairs are not unique")
    top_level_content_id_domains = {
        domain for domain, _ in schema_domains_and_prefixes
    } | {domain for domain, _ in artifact_domains_and_prefixes}
    if any(
        domain in top_level_content_id_domains for domain in source_root_domains
    ):
        raise RuntimeError("gate source root domain collides with content-ID domain")
    nested_artifact_domains_and_prefixes = (
        (_LEDGER_EVENT_ID_DOMAIN_ASCII_V1, _LEDGER_EVENT_ID_PREFIX_V1),
        (_ATTEMPT_RECORD_ID_DOMAIN_ASCII_V1, _ATTEMPT_RECORD_ID_PREFIX_V1),
        (_GATE_RESULT_ID_DOMAIN_ASCII_V1, _GATE_RESULT_ID_PREFIX_V1),
        (
            _DENOMINATOR_EVENT_ID_DOMAIN_ASCII_V1,
            _DENOMINATOR_EVENT_ID_PREFIX_V1,
        ),
        (
            _ATTEMPT_UNIVERSE_HEAD_ID_DOMAIN_ASCII_V1,
            _ATTEMPT_UNIVERSE_HEAD_ID_PREFIX_V1,
        ),
        (
            _DENOMINATOR_FINALIZATION_ID_DOMAIN_ASCII_V1,
            _DENOMINATOR_FINALIZATION_ID_PREFIX_V1,
        ),
    )
    if any(not domain.endswith("\x00") for domain, _ in nested_artifact_domains_and_prefixes):
        raise RuntimeError("nested artifact content-ID domain is not NUL terminated")
    if len(set(nested_artifact_domains_and_prefixes)) != 6 or any(
        pair in artifact_domains_and_prefixes
        for pair in nested_artifact_domains_and_prefixes
    ):
        raise RuntimeError("nested artifact content-ID domain/prefix uniqueness differs")
    if any(
        domain in {pair[0] for pair in nested_artifact_domains_and_prefixes}
        for domain in source_root_domains
    ):
        raise RuntimeError("gate source root domain collides with nested artifact domain")
    signature_domains: list[str] = []
    signature_artifact_kinds: list[str] = []
    for item in (
        SIGNER_REGISTRY_SCHEMA_OBJECT_V1,
        PREALLOCATION_LINEAGE_SCHEMA_OBJECT_V1,
        *GATE_SOURCE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1,
        *GATE_SUPPORT_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1,
        *GATE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1,
        *STAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1,
        *PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1,
        BUNDLE_INDEX_SCHEMA_OBJECT_V1,
    ):
        if item["content_id_preimage_profile"] != CONTENT_ID_PREIMAGE_PROFILE_V1:
            raise RuntimeError("content-ID preimage profile drift")
        if item["signature_preimage_profile"] != DETACHED_SIGNATURE_PREIMAGE_PROFILE_V1:
            raise RuntimeError("signature preimage profile drift")
        if item["detached_signature_schema_version"] != (
            DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_V1
        ):
            raise RuntimeError("detached signature record schema version drift")
        if item["signature_preimage_schema_version"] != (
            DETACHED_SIGNATURE_PREIMAGE_SCHEMA_VERSION_V1
        ):
            raise RuntimeError("signature preimage schema version drift")
        if item["signature_algorithm"] != SIGNATURE_ALGORITHM_V1:
            raise RuntimeError("signature algorithm drift")
        signature_artifact_kinds.append(
            str(item["signature_preimage_artifact_kind"])
        )
        if item["content_id_preimage_fields_in_exact_order"] != ["body"]:
            raise RuntimeError("content-ID preimage fields drift")
        if item["signature_preimage_fields_in_exact_order"] != list(
            DETACHED_SIGNATURE_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
        ):
            raise RuntimeError("signature preimage fields drift")
        slots = item["signature_slots_in_exact_order"]
        if type(slots) is not list or item["signature_cardinality"] != len(slots):
            raise RuntimeError("signature slot cardinality drift")
        for slot in slots:
            if type(slot) is not dict or type(slot["key_epoch"]) is not int:
                raise RuntimeError("signature slot exact integer epoch drift")
            if slot["key_epoch"] != 1:
                raise RuntimeError("signature slot epoch drift")
            signature_domains.append(str(slot["signature_domain_ascii"]))
    for slot in _LEDGER_EVENT_SIGNATURE_SLOTS_IN_EXACT_ORDER_V1:
        if type(slot.key_epoch) is not int or slot.key_epoch != 1:
            raise RuntimeError("ledger event signature epoch differs")
        signature_domains.append(slot.signature_domain_ascii)
    if (
        type(_DENOMINATOR_EVENT_SIGNATURE_SLOT_V1.key_epoch) is not int
        or _DENOMINATOR_EVENT_SIGNATURE_SLOT_V1.key_epoch != 1
    ):
        raise RuntimeError("denominator event signature epoch differs")
    signature_domains.append(
        _DENOMINATOR_EVENT_SIGNATURE_SLOT_V1.signature_domain_ascii
    )
    for slot in _ATTEMPT_UNIVERSE_HEAD_SIGNATURE_SLOTS_IN_EXACT_ORDER_V1:
        if type(slot.key_epoch) is not int or slot.key_epoch != 1:
            raise RuntimeError("attempt-universe head signature epoch differs")
        signature_domains.append(slot.signature_domain_ascii)
    for slot in _DENOMINATOR_FINALIZATION_SIGNATURE_SLOTS_IN_EXACT_ORDER_V1:
        if type(slot.key_epoch) is not int or slot.key_epoch != 1:
            raise RuntimeError("denominator finalization signature epoch differs")
        signature_domains.append(slot.signature_domain_ascii)
    if len(set(signature_domains)) != len(signature_domains):
        raise RuntimeError("signature domains are not globally unique")
    if len(set(signature_artifact_kinds)) != len(signature_artifact_kinds):
        raise RuntimeError("signature artifact kinds are not globally unique")
    nodes = {
        *(gate.gate_id for gate in GATE_SPECS_IN_EXACT_ORDER_V1),
        *STAGE_NAMES_IN_EXACT_ORDER_V1,
    }
    indegree = {node: 0 for node in nodes}
    successors = {node: [] for node in nodes}
    for predecessor, successor in INTERLEAVED_DAG_EDGES_IN_EXACT_ORDER_V1:
        if predecessor not in nodes or successor not in nodes:
            raise RuntimeError("interleaved DAG references an unknown node")
        successors[predecessor].append(successor)
        indegree[successor] += 1
    ready = [node for node, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for successor in successors[node]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    if visited != len(nodes):
        raise RuntimeError("interleaved DAG contains a cycle or unreachable node")


_assert_static_catalog_v1()


def canonical_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_bytes_v1() -> bytes:
    """Return the deterministic candidate catalog bytes; no evidence is accepted."""

    return bytes(CATALOG_PAYLOAD_V1)


def _build_static_catalog_payload_getter_v1(payload: bytes):
    captured_payload = bytes(payload)
    bytes_builtin = bytes

    def canonical_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_bytes_v1() -> bytes:
        return bytes_builtin(captured_payload)

    return canonical_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_bytes_v1


canonical_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_bytes_v1 = (
    _build_static_catalog_payload_getter_v1(CATALOG_PAYLOAD_V1)
)
del _build_static_catalog_payload_getter_v1


class Phase2BActualUnsealed960AuthoritativeEvidenceSchemaCatalogErrorV1(
    ValueError
):
    """Stable failure type for the raw catalog boundary."""


class Phase2BActualUnsealed960AuthoritativeEvidencePackage0ErrorV1(
    ValueError
):
    """Stable failure type for the future raw Package 0 boundary."""


@dataclass(frozen=True, slots=True, init=False)
class VerifiedPhase2BActualUnsealed960AuthoritativeEvidenceSchemaCatalogV1:
    schema_version: str
    status: str
    claim_level: str
    disposition: CatalogDispositionV1
    reason: CatalogReasonV1
    catalog_id: str
    catalog_payload_sha256: str
    inspected_dependency_count: int
    gate_count: int
    satisfied_gate_count: int
    package_count: int
    stage_count: int
    gate_schema_ids_root: str
    stage_schema_ids_root: str
    package_schema_ids_root: str
    package_schema_dag_root: str
    bundle_index_schema_id: str
    interleaved_dag_root: str
    exact_raw_bytes_verified: bool
    rfc8785_canonical_bytes_verified: bool
    inspected_dependency_pins_verified: bool
    schema_catalog_exactly_verified: bool
    content_address_roots_verified: bool
    exact_preimage_profiles_verified: bool
    package_schema_dag_verified: bool
    bundle_index_schema_verified: bool
    interleaved_dag_verified: bool
    attempt_retry_policy_verified: bool
    terminal_policy_verified: bool
    private_lineage_verified: bool
    evidence_supplied: bool
    evidence_verified: bool
    r01_satisfied: bool
    r02_satisfied: bool
    r03_satisfied: bool
    r04_satisfied: bool
    r05_satisfied: bool
    r06_satisfied: bool
    r07_satisfied: bool
    r08_satisfied: bool
    r09_satisfied: bool
    r10_satisfied: bool
    r11_satisfied: bool
    r12_satisfied: bool
    r13_satisfied: bool
    r14_satisfied: bool
    r15_satisfied: bool
    r16_satisfied: bool
    authority_currentness_verified: bool
    owner_non_revocation_verified: bool
    six_live_roles_verified: bool
    external_key_custody_verified: bool
    external_durability_verified: bool
    external_non_equivocation_verified: bool
    campaign_id_allocated: bool
    attempt_id_allocated: bool
    actual_960_generated: bool
    actual_corpus_audit_executed: bool
    inference_performed: bool
    predictions_committed: bool
    answers_revealed: bool
    scoring_performed: bool
    terminal_report_consumed: bool
    execution_authorized: bool
    c1_satisfied: bool

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("verified catalog results are privately issued")

    def __copy__(self, _type_error: type[TypeError] = TypeError) -> NoReturn:
        raise _type_error("verified catalog results cannot be copied")

    def __deepcopy__(
        self,
        memo: object,
        _type_error: type[TypeError] = TypeError,
    ) -> NoReturn:
        del memo
        raise _type_error("verified catalog results cannot be deep-copied")

    def __reduce_ex__(
        self,
        protocol: object,
        _type_error: type[TypeError] = TypeError,
    ) -> NoReturn:
        del protocol
        raise _type_error("verified catalog results cannot be serialized")


@dataclass(frozen=True, slots=True, init=False)
class VerifiedPhase2BActualUnsealed960AuthoritativeEvidencePackage0V1:
    schema_version: str
    status: str
    claim_level: str
    disposition: Package0DispositionV1
    reason: Package0ReasonV1
    catalog_id: str
    catalog_payload_sha256: str
    package_schema_id: str
    package_id: str
    package_payload_sha256: str
    signer_registry_id: str
    signer_registry_payload_sha256: str
    signer_registry_entries_root_sha256: str
    campaign_id: str
    attempt_id: str
    chain_root_sha256: str
    terminal_signed_stage_event_id: str
    terminal_signed_stage_event_payload_sha256: str
    ledger_event_count: int
    stage_subject_event_count: int
    referenced_artifact_count: int
    total_signature_count: int
    exact_raw_bytes_verified: bool
    rfc8785_canonical_bytes_verified: bool
    content_ids_verified: bool
    raw_payload_sha256s_verified: bool
    signature_contracts_verified: bool
    ledger_chain_verified: bool
    stage_subject_binding_verified: bool
    referenced_artifact_graph_verified: bool
    package0_lineage_verified: bool
    terminal_p15_p16_binding_verified: bool
    predecessor_absence_verified: bool
    private_lineage_verified: bool
    package15_verified: bool
    package16_verified: bool
    gate_admission_verified: bool
    r10_satisfied: bool
    authority_currentness_verified: bool
    owner_non_revocation_verified: bool
    external_key_custody_verified: bool
    external_durability_verified: bool
    external_non_equivocation_verified: bool
    global_anti_fork_verified: bool
    execution_authorized: bool
    actual_960_verified: bool
    c1_satisfied: bool

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("verified package 0 results are privately issued")

    def __copy__(self, _type_error: type[TypeError] = TypeError) -> NoReturn:
        raise _type_error("verified package 0 results cannot be copied")

    def __deepcopy__(
        self,
        memo: object,
        _type_error: type[TypeError] = TypeError,
    ) -> NoReturn:
        del memo
        raise _type_error("verified package 0 results cannot be deep-copied")

    def __reduce_ex__(
        self,
        protocol: object,
        _type_error: type[TypeError] = TypeError,
    ) -> NoReturn:
        del protocol
        raise _type_error("verified package 0 results cannot be serialized")


_RESULT_FIELD_NAMES_V1: Final = tuple(
    VerifiedPhase2BActualUnsealed960AuthoritativeEvidenceSchemaCatalogV1.__dataclass_fields__
)
_RESULT_NONCLAIM_START_V1: Final = _RESULT_FIELD_NAMES_V1.index("evidence_supplied")
if _RESULT_FIELD_NAMES_V1[_RESULT_NONCLAIM_START_V1:] != (
    PERMANENT_NONCLAIMS_IN_EXACT_ORDER_V1
):
    raise RuntimeError("permanent nonclaim/result field set differs")

_PACKAGE0_RESULT_FIELD_NAMES_V1: Final = tuple(
    VerifiedPhase2BActualUnsealed960AuthoritativeEvidencePackage0V1.__dataclass_fields__
)
if _PACKAGE0_RESULT_FIELD_NAMES_V1 != tuple(
    field_name
    for field_name, _ in PACKAGE0_VERIFIED_RESULT_FIELDS_IN_EXACT_ORDER_V1
):
    raise RuntimeError("package 0 verified result field set differs")


def _build_private_catalog_boundary_v1():
    """Build a definition-time closed raw verifier and private lineage."""

    result_type = VerifiedPhase2BActualUnsealed960AuthoritativeEvidenceSchemaCatalogV1
    error_type = Phase2BActualUnsealed960AuthoritativeEvidenceSchemaCatalogErrorV1
    disposition_type = CatalogDispositionV1
    reason_type = CatalogReasonV1
    expected_payload = bytes(CATALOG_PAYLOAD_V1)
    expected_catalog_id = str(CATALOG_ID_V1)
    expected_schema = str(SCHEMA_VERSION_V1)
    expected_status = str(STATUS_V1)
    expected_claim_level = str(CLAIM_LEVEL_V1)
    maximum_catalog_payload_bytes = int(MAXIMUM_CATALOG_PAYLOAD_BYTES_V1)
    expected_readiness_id = INSPECTED_READINESS_DAG_ID_V1.encode("ascii")
    expected_readiness_sha = bytes.fromhex(INSPECTED_READINESS_DAG_CONFIG_SHA256_V1)
    expected_plan_sha = bytes.fromhex(INSPECTED_PROGRESSION_PLAN_SHA256_V1)
    expected_gap_sha = bytes.fromhex(INSPECTED_GAP_INVENTORY_SOURCE_SHA256_V1)
    expected_validation_sha = bytes.fromhex(
        INSPECTED_STATISTICAL_VALIDATION_SOURCE_SHA256_V1
    )
    expected_gate_root = str(GATE_SCHEMA_IDS_ROOT_V1)
    expected_stage_root = str(STAGE_SCHEMA_IDS_ROOT_V1)
    expected_package_root = str(PACKAGE_SCHEMA_IDS_ROOT_V1)
    expected_package_dag_root = str(PACKAGE_SCHEMA_DAG_ROOT_V1)
    expected_bundle_index_schema_id = str(
        BUNDLE_INDEX_SCHEMA_OBJECT_V1["bundle_index_schema_id"]
    )
    expected_dag_root = str(DAG_ROOT_V1)
    expected_result_fields = tuple(_RESULT_FIELD_NAMES_V1)
    sha256_constructor = hashlib.sha256
    object_type = object
    type_builtin = type
    len_builtin = len
    id_builtin = id
    getattr_builtin = getattr
    tuple_builtin = tuple
    zip_builtin = zip
    all_builtin = all
    bytes_type = bytes
    tuple_type = tuple
    str_type = str
    int_type = int
    bool_type = bool
    none_type = type(None)
    registry: dict[int, tuple[object, tuple[object, ...]]] = {}

    def fail(code: str, detail: str) -> NoReturn:
        raise error_type(f"{code}: {detail}")

    def exact_bytes(value: object, name: str, *, length: int | None = None) -> bytes:
        if type_builtin(value) is not bytes_type:
            fail("FAIL_C1_CATALOG_V1_TYPE", f"{name} must be exact bytes")
        raw = value
        if length is not None and len_builtin(raw) != length:
            fail("FAIL_C1_CATALOG_V1_PIN", f"{name} length differs")
        return raw

    def exact_equal(left: object, right: object) -> bool:
        if type_builtin(left) is not type_builtin(right):
            return False
        if type_builtin(left) is tuple_type:
            return len_builtin(left) == len_builtin(right) and all_builtin(
                exact_equal(a, b)
                for a, b in zip_builtin(left, right, strict=True)
            )
        if type_builtin(left) in (
            str_type,
            int_type,
            bool_type,
            bytes_type,
            none_type,
        ):
            return left == right
        if type_builtin(left) in (disposition_type, reason_type):
            return left is right
        return left is right

    def snapshot(result: object) -> tuple[object, ...]:
        return tuple_builtin(
            getattr_builtin(result, name) for name in expected_result_fields
        )

    def require_result(
        result: VerifiedPhase2BActualUnsealed960AuthoritativeEvidenceSchemaCatalogV1,
    ) -> VerifiedPhase2BActualUnsealed960AuthoritativeEvidenceSchemaCatalogV1:
        if type_builtin(result) is not result_type:
            fail("FAIL_C1_CATALOG_V1_LINEAGE", "result type differs")
        record = registry.get(id_builtin(result))
        if record is None or record[0] is not result:
            fail("FAIL_C1_CATALOG_V1_LINEAGE", "result was not privately issued")
        current = snapshot(result)
        if not exact_equal(current, record[1]):
            fail("FAIL_C1_CATALOG_V1_LINEAGE", "result was mutated")
        return result

    def issue_result(
        values: tuple[tuple[str, object], ...]
    ) -> VerifiedPhase2BActualUnsealed960AuthoritativeEvidenceSchemaCatalogV1:
        if tuple_builtin(name for name, _ in values) != expected_result_fields:
            fail("FAIL_C1_CATALOG_V1_INTERNAL", "result field order differs")
        result = object_type.__new__(result_type)
        for name, value in values:
            object_type.__setattr__(result, name, value)
        issued = result
        issued_snapshot = snapshot(issued)
        registry[id_builtin(issued)] = (issued, issued_snapshot)
        return require_result(issued)

    def verify_implementation(
        payload: bytes,
        *,
        out_of_band_expected_readiness_dag_id_bytes: bytes,
        out_of_band_expected_readiness_dag_payload_sha256_digest: bytes,
        out_of_band_expected_progression_plan_payload_sha256_digest: bytes,
        out_of_band_expected_gap_inventory_source_sha256_digest: bytes,
        out_of_band_expected_statistical_validation_source_sha256_digest: bytes,
    ) -> VerifiedPhase2BActualUnsealed960AuthoritativeEvidenceSchemaCatalogV1:
        raw = exact_bytes(payload, "payload")
        readiness_id = exact_bytes(
            out_of_band_expected_readiness_dag_id_bytes,
            "out_of_band_expected_readiness_dag_id_bytes",
        )
        readiness_sha = exact_bytes(
            out_of_band_expected_readiness_dag_payload_sha256_digest,
            "out_of_band_expected_readiness_dag_payload_sha256_digest",
            length=32,
        )
        plan_sha = exact_bytes(
            out_of_band_expected_progression_plan_payload_sha256_digest,
            "out_of_band_expected_progression_plan_payload_sha256_digest",
            length=32,
        )
        gap_sha = exact_bytes(
            out_of_band_expected_gap_inventory_source_sha256_digest,
            "out_of_band_expected_gap_inventory_source_sha256_digest",
            length=32,
        )
        validation_sha = exact_bytes(
            out_of_band_expected_statistical_validation_source_sha256_digest,
            "out_of_band_expected_statistical_validation_source_sha256_digest",
            length=32,
        )
        if (
            readiness_id != expected_readiness_id
            or readiness_sha != expected_readiness_sha
            or plan_sha != expected_plan_sha
            or gap_sha != expected_gap_sha
            or validation_sha != expected_validation_sha
        ):
            fail("FAIL_C1_CATALOG_V1_PIN", "one or more OOB inspected pins differ")
        if len_builtin(raw) > maximum_catalog_payload_bytes:
            fail(
                "FAIL_C1_CATALOG_V1_SCHEMA",
                "catalog exceeds the frozen two MiB byte bound",
            )
        if raw != expected_payload:
            fail(
                "FAIL_C1_CATALOG_V1_CANONICAL",
                "catalog bytes differ from the definition-time frozen canonical payload",
            )
        payload_sha = sha256_constructor(raw).hexdigest()
        values: tuple[tuple[str, object], ...] = (
            ("schema_version", expected_schema),
            ("status", expected_status),
            ("claim_level", expected_claim_level),
            ("disposition", disposition_type.RUNBOOK_FROZEN_NOT_ADMITTED),
            ("reason", reason_type.EXACT_EXECUTABLE_SCHEMA_CATALOG_FROZEN),
            ("catalog_id", expected_catalog_id),
            ("catalog_payload_sha256", payload_sha),
            ("inspected_dependency_count", 4),
            ("gate_count", 16),
            ("satisfied_gate_count", 0),
            ("package_count", 18),
            ("stage_count", 10),
            ("gate_schema_ids_root", expected_gate_root),
            ("stage_schema_ids_root", expected_stage_root),
            ("package_schema_ids_root", expected_package_root),
            ("package_schema_dag_root", expected_package_dag_root),
            ("bundle_index_schema_id", expected_bundle_index_schema_id),
            ("interleaved_dag_root", expected_dag_root),
            ("exact_raw_bytes_verified", True),
            ("rfc8785_canonical_bytes_verified", True),
            ("inspected_dependency_pins_verified", True),
            ("schema_catalog_exactly_verified", True),
            ("content_address_roots_verified", True),
            ("exact_preimage_profiles_verified", True),
            ("package_schema_dag_verified", True),
            ("bundle_index_schema_verified", True),
            ("interleaved_dag_verified", True),
            ("attempt_retry_policy_verified", True),
            ("terminal_policy_verified", True),
            ("private_lineage_verified", True),
            ("evidence_supplied", False),
            ("evidence_verified", False),
            ("r01_satisfied", False),
            ("r02_satisfied", False),
            ("r03_satisfied", False),
            ("r04_satisfied", False),
            ("r05_satisfied", False),
            ("r06_satisfied", False),
            ("r07_satisfied", False),
            ("r08_satisfied", False),
            ("r09_satisfied", False),
            ("r10_satisfied", False),
            ("r11_satisfied", False),
            ("r12_satisfied", False),
            ("r13_satisfied", False),
            ("r14_satisfied", False),
            ("r15_satisfied", False),
            ("r16_satisfied", False),
            ("authority_currentness_verified", False),
            ("owner_non_revocation_verified", False),
            ("six_live_roles_verified", False),
            ("external_key_custody_verified", False),
            ("external_durability_verified", False),
            ("external_non_equivocation_verified", False),
            ("campaign_id_allocated", False),
            ("attempt_id_allocated", False),
            ("actual_960_generated", False),
            ("actual_corpus_audit_executed", False),
            ("inference_performed", False),
            ("predictions_committed", False),
            ("answers_revealed", False),
            ("scoring_performed", False),
            ("terminal_report_consumed", False),
            ("execution_authorized", False),
            ("c1_satisfied", False),
        )
        return issue_result(values)

    def real_implementation(
        catalog_payload: bytes,
        raw_signer_registry_payload: bytes,
        raw_gate_evidence_payloads_in_exact_order: tuple[bytes, ...],
        raw_preallocation_lineage_payload: bytes,
        raw_gate_supporting_artifact_payload_tuples_in_exact_order: tuple[
            tuple[bytes, ...], ...
        ],
        raw_gate_source_native_payload_tuples_in_exact_order: tuple[
            tuple[bytes, ...], ...
        ],
        raw_stage_event_payloads_in_exact_order: tuple[bytes, ...],
        raw_package_envelope_payloads_in_exact_order: tuple[bytes, ...],
        raw_bundle_index_payload: bytes,
        raw_package0_ledger_event_payloads_in_exact_order: tuple[bytes, ...],
        raw_package0_stage_subject_event_payloads_in_exact_order: tuple[
            bytes, ...
        ],
        raw_store_if_absent_receipt_payload: bytes,
        raw_logical_time_token_payload: bytes,
        raw_governance_checkpoint_payload: bytes,
        raw_custodian_checkpoint_payload: bytes,
        raw_external_anchor_payload: bytes,
        raw_package10_denominator_event_payloads_in_exact_order: tuple[
            bytes, ...
        ],
        raw_attempt_universe_head_payload: bytes,
        raw_package10_denominator_finalization_payload: bytes,
        raw_package15_gate_result_payloads_in_exact_order: tuple[bytes, ...],
        raw_package16_attempt_record_payloads_in_exact_order: tuple[bytes, ...],
        *,
        out_of_band_expected_catalog_id_bytes: bytes,
        out_of_band_expected_catalog_payload_sha256_digest: bytes,
        out_of_band_expected_signer_registry_id_bytes: bytes,
        out_of_band_expected_signer_registry_payload_sha256_digest: bytes,
        out_of_band_expected_signer_registry_entries_root_sha256_digest: bytes,
        out_of_band_expected_preallocation_lineage_id_bytes: bytes,
        out_of_band_expected_preallocation_lineage_payload_sha256_digest: bytes,
        out_of_band_expected_governance_root_key_id_bytes: bytes,
        out_of_band_expected_governance_root_public_key_bytes: bytes,
        out_of_band_expected_custodian_root_key_id_bytes: bytes,
        out_of_band_expected_custodian_root_public_key_bytes: bytes,
        out_of_band_expected_attempt_universe_head_id_bytes: bytes,
        out_of_band_expected_attempt_universe_head_payload_sha256_digest: bytes,
        out_of_band_expected_readiness_dag_id_bytes: bytes,
        out_of_band_expected_readiness_dag_payload_sha256_digest: bytes,
        out_of_band_expected_progression_plan_payload_sha256_digest: bytes,
        out_of_band_expected_gap_inventory_source_sha256_digest: bytes,
        out_of_band_expected_statistical_validation_source_sha256_digest: bytes,
        expected_campaign_id_bytes: bytes,
        expected_attempt_id_bytes: bytes,
    ) -> NoReturn:
        del (
            catalog_payload,
            raw_signer_registry_payload,
            raw_gate_evidence_payloads_in_exact_order,
            raw_preallocation_lineage_payload,
            raw_gate_supporting_artifact_payload_tuples_in_exact_order,
            raw_gate_source_native_payload_tuples_in_exact_order,
            raw_stage_event_payloads_in_exact_order,
            raw_package_envelope_payloads_in_exact_order,
            raw_bundle_index_payload,
            raw_package0_ledger_event_payloads_in_exact_order,
            raw_package0_stage_subject_event_payloads_in_exact_order,
            raw_store_if_absent_receipt_payload,
            raw_logical_time_token_payload,
            raw_governance_checkpoint_payload,
            raw_custodian_checkpoint_payload,
            raw_external_anchor_payload,
            raw_package10_denominator_event_payloads_in_exact_order,
            raw_attempt_universe_head_payload,
            raw_package10_denominator_finalization_payload,
            raw_package15_gate_result_payloads_in_exact_order,
            raw_package16_attempt_record_payloads_in_exact_order,
            out_of_band_expected_catalog_id_bytes,
            out_of_band_expected_catalog_payload_sha256_digest,
            out_of_band_expected_signer_registry_id_bytes,
            out_of_band_expected_signer_registry_payload_sha256_digest,
            out_of_band_expected_signer_registry_entries_root_sha256_digest,
            out_of_band_expected_preallocation_lineage_id_bytes,
            out_of_band_expected_preallocation_lineage_payload_sha256_digest,
            out_of_band_expected_governance_root_key_id_bytes,
            out_of_band_expected_governance_root_public_key_bytes,
            out_of_band_expected_custodian_root_key_id_bytes,
            out_of_band_expected_custodian_root_public_key_bytes,
            out_of_band_expected_attempt_universe_head_id_bytes,
            out_of_band_expected_attempt_universe_head_payload_sha256_digest,
            out_of_band_expected_readiness_dag_id_bytes,
            out_of_band_expected_readiness_dag_payload_sha256_digest,
            out_of_band_expected_progression_plan_payload_sha256_digest,
            out_of_band_expected_gap_inventory_source_sha256_digest,
            out_of_band_expected_statistical_validation_source_sha256_digest,
            expected_campaign_id_bytes,
            expected_attempt_id_bytes,
        )
        fail(
            "FAIL_C1_CATALOG_V1_NOT_IMPLEMENTED",
            "real authoritative evidence-bundle verification is exactly NOT_IMPLEMENTED",
        )

    return verify_implementation, require_result, real_implementation


(
    _VERIFY_CATALOG_IMPLEMENTATION_V1,
    require_verified_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1,
    _VERIFY_REAL_BUNDLE_IMPLEMENTATION_V1,
) = _build_private_catalog_boundary_v1()


def _build_private_package0_boundary_v1():
    """Build a definition-time closed fail-closed Package 0 boundary."""

    result_type = VerifiedPhase2BActualUnsealed960AuthoritativeEvidencePackage0V1
    error_type = Phase2BActualUnsealed960AuthoritativeEvidencePackage0ErrorV1
    type_builtin = type
    id_builtin = id
    registry: dict[int, object] = {}

    def fail(code: str, detail: str) -> NoReturn:
        raise error_type(f"{code}: {detail}")

    def require_result(
        result: VerifiedPhase2BActualUnsealed960AuthoritativeEvidencePackage0V1,
    ) -> VerifiedPhase2BActualUnsealed960AuthoritativeEvidencePackage0V1:
        if type_builtin(result) is not result_type:
            fail("FAIL_C1_PACKAGE0_V1_LINEAGE", "result type differs")
        issued = registry.get(id_builtin(result))
        if issued is not result:
            fail(
                "FAIL_C1_PACKAGE0_V1_LINEAGE",
                "result was not privately issued by an implemented verifier",
            )
        return result

    def real_implementation(
        catalog_payload: bytes,
        raw_signer_registry_payload: bytes,
        raw_package0_envelope_payload: bytes,
        raw_package0_ledger_event_payloads_in_exact_order: tuple[bytes, ...],
        raw_package0_stage_subject_event_payloads_in_exact_order: tuple[
            bytes, ...
        ],
        raw_store_if_absent_receipt_payload: bytes,
        raw_logical_time_token_payload: bytes,
        raw_governance_checkpoint_payload: bytes,
        raw_custodian_checkpoint_payload: bytes,
        raw_external_anchor_payload: bytes,
        raw_package15_envelope_payload: bytes,
        raw_package16_envelope_payload: bytes,
        *,
        out_of_band_expected_catalog_id_bytes: bytes,
        out_of_band_expected_catalog_payload_sha256_digest: bytes,
        out_of_band_expected_signer_registry_id_bytes: bytes,
        out_of_band_expected_signer_registry_payload_sha256_digest: bytes,
        out_of_band_expected_signer_registry_entries_root_sha256_digest: bytes,
        out_of_band_expected_governance_root_key_id_bytes: bytes,
        out_of_band_expected_governance_root_public_key_bytes: bytes,
        out_of_band_expected_custodian_root_key_id_bytes: bytes,
        out_of_band_expected_custodian_root_public_key_bytes: bytes,
        out_of_band_expected_readiness_dag_id_bytes: bytes,
        out_of_band_expected_readiness_dag_payload_sha256_digest: bytes,
        out_of_band_expected_progression_plan_payload_sha256_digest: bytes,
        out_of_band_expected_gap_inventory_source_sha256_digest: bytes,
        out_of_band_expected_statistical_validation_source_sha256_digest: bytes,
        expected_campaign_id_bytes: bytes,
        expected_attempt_id_bytes: bytes,
    ) -> VerifiedPhase2BActualUnsealed960AuthoritativeEvidencePackage0V1:
        del (
            catalog_payload,
            raw_signer_registry_payload,
            raw_package0_envelope_payload,
            raw_package0_ledger_event_payloads_in_exact_order,
            raw_package0_stage_subject_event_payloads_in_exact_order,
            raw_store_if_absent_receipt_payload,
            raw_logical_time_token_payload,
            raw_governance_checkpoint_payload,
            raw_custodian_checkpoint_payload,
            raw_external_anchor_payload,
            raw_package15_envelope_payload,
            raw_package16_envelope_payload,
            out_of_band_expected_catalog_id_bytes,
            out_of_band_expected_catalog_payload_sha256_digest,
            out_of_band_expected_signer_registry_id_bytes,
            out_of_band_expected_signer_registry_payload_sha256_digest,
            out_of_band_expected_signer_registry_entries_root_sha256_digest,
            out_of_band_expected_governance_root_key_id_bytes,
            out_of_band_expected_governance_root_public_key_bytes,
            out_of_band_expected_custodian_root_key_id_bytes,
            out_of_band_expected_custodian_root_public_key_bytes,
            out_of_band_expected_readiness_dag_id_bytes,
            out_of_band_expected_readiness_dag_payload_sha256_digest,
            out_of_band_expected_progression_plan_payload_sha256_digest,
            out_of_band_expected_gap_inventory_source_sha256_digest,
            out_of_band_expected_statistical_validation_source_sha256_digest,
            expected_campaign_id_bytes,
            expected_attempt_id_bytes,
        )
        fail(
            "FAIL_C1_PACKAGE0_V1_NOT_IMPLEMENTED",
            "real package 0 evidence verification is exactly NOT_IMPLEMENTED",
        )

    return require_result, real_implementation


(
    require_verified_phase2b_actual_unsealed_960_authoritative_evidence_package0_v1,
    _VERIFY_REAL_PACKAGE0_IMPLEMENTATION_V1,
) = _build_private_package0_boundary_v1()


def _build_public_catalog_verifier_v1(implementation):
    def verify_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1(
        payload: bytes,
        *,
        out_of_band_expected_readiness_dag_id_bytes: bytes,
        out_of_band_expected_readiness_dag_payload_sha256_digest: bytes,
        out_of_band_expected_progression_plan_payload_sha256_digest: bytes,
        out_of_band_expected_gap_inventory_source_sha256_digest: bytes,
        out_of_band_expected_statistical_validation_source_sha256_digest: bytes,
    ) -> VerifiedPhase2BActualUnsealed960AuthoritativeEvidenceSchemaCatalogV1:
        return implementation(
            payload,
            out_of_band_expected_readiness_dag_id_bytes=(
                out_of_band_expected_readiness_dag_id_bytes
            ),
            out_of_band_expected_readiness_dag_payload_sha256_digest=(
                out_of_band_expected_readiness_dag_payload_sha256_digest
            ),
            out_of_band_expected_progression_plan_payload_sha256_digest=(
                out_of_band_expected_progression_plan_payload_sha256_digest
            ),
            out_of_band_expected_gap_inventory_source_sha256_digest=(
                out_of_band_expected_gap_inventory_source_sha256_digest
            ),
            out_of_band_expected_statistical_validation_source_sha256_digest=(
                out_of_band_expected_statistical_validation_source_sha256_digest
            ),
        )

    return verify_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1


def _build_public_real_package0_verifier_v1(implementation):
    def verify_real_phase2b_actual_unsealed_960_authoritative_evidence_package0_v1(
        catalog_payload: bytes,
        raw_signer_registry_payload: bytes,
        raw_package0_envelope_payload: bytes,
        raw_package0_ledger_event_payloads_in_exact_order: tuple[bytes, ...],
        raw_package0_stage_subject_event_payloads_in_exact_order: tuple[
            bytes, ...
        ],
        raw_store_if_absent_receipt_payload: bytes,
        raw_logical_time_token_payload: bytes,
        raw_governance_checkpoint_payload: bytes,
        raw_custodian_checkpoint_payload: bytes,
        raw_external_anchor_payload: bytes,
        raw_package15_envelope_payload: bytes,
        raw_package16_envelope_payload: bytes,
        *,
        out_of_band_expected_catalog_id_bytes: bytes,
        out_of_band_expected_catalog_payload_sha256_digest: bytes,
        out_of_band_expected_signer_registry_id_bytes: bytes,
        out_of_band_expected_signer_registry_payload_sha256_digest: bytes,
        out_of_band_expected_signer_registry_entries_root_sha256_digest: bytes,
        out_of_band_expected_governance_root_key_id_bytes: bytes,
        out_of_band_expected_governance_root_public_key_bytes: bytes,
        out_of_band_expected_custodian_root_key_id_bytes: bytes,
        out_of_band_expected_custodian_root_public_key_bytes: bytes,
        out_of_band_expected_readiness_dag_id_bytes: bytes,
        out_of_band_expected_readiness_dag_payload_sha256_digest: bytes,
        out_of_band_expected_progression_plan_payload_sha256_digest: bytes,
        out_of_band_expected_gap_inventory_source_sha256_digest: bytes,
        out_of_band_expected_statistical_validation_source_sha256_digest: bytes,
        expected_campaign_id_bytes: bytes,
        expected_attempt_id_bytes: bytes,
    ) -> VerifiedPhase2BActualUnsealed960AuthoritativeEvidencePackage0V1:
        return implementation(
            catalog_payload,
            raw_signer_registry_payload,
            raw_package0_envelope_payload,
            raw_package0_ledger_event_payloads_in_exact_order,
            raw_package0_stage_subject_event_payloads_in_exact_order,
            raw_store_if_absent_receipt_payload,
            raw_logical_time_token_payload,
            raw_governance_checkpoint_payload,
            raw_custodian_checkpoint_payload,
            raw_external_anchor_payload,
            raw_package15_envelope_payload,
            raw_package16_envelope_payload,
            out_of_band_expected_catalog_id_bytes=(
                out_of_band_expected_catalog_id_bytes
            ),
            out_of_band_expected_catalog_payload_sha256_digest=(
                out_of_band_expected_catalog_payload_sha256_digest
            ),
            out_of_band_expected_signer_registry_id_bytes=(
                out_of_band_expected_signer_registry_id_bytes
            ),
            out_of_band_expected_signer_registry_payload_sha256_digest=(
                out_of_band_expected_signer_registry_payload_sha256_digest
            ),
            out_of_band_expected_signer_registry_entries_root_sha256_digest=(
                out_of_band_expected_signer_registry_entries_root_sha256_digest
            ),
            out_of_band_expected_governance_root_key_id_bytes=(
                out_of_band_expected_governance_root_key_id_bytes
            ),
            out_of_band_expected_governance_root_public_key_bytes=(
                out_of_band_expected_governance_root_public_key_bytes
            ),
            out_of_band_expected_custodian_root_key_id_bytes=(
                out_of_band_expected_custodian_root_key_id_bytes
            ),
            out_of_band_expected_custodian_root_public_key_bytes=(
                out_of_band_expected_custodian_root_public_key_bytes
            ),
            out_of_band_expected_readiness_dag_id_bytes=(
                out_of_band_expected_readiness_dag_id_bytes
            ),
            out_of_band_expected_readiness_dag_payload_sha256_digest=(
                out_of_band_expected_readiness_dag_payload_sha256_digest
            ),
            out_of_band_expected_progression_plan_payload_sha256_digest=(
                out_of_band_expected_progression_plan_payload_sha256_digest
            ),
            out_of_band_expected_gap_inventory_source_sha256_digest=(
                out_of_band_expected_gap_inventory_source_sha256_digest
            ),
            out_of_band_expected_statistical_validation_source_sha256_digest=(
                out_of_band_expected_statistical_validation_source_sha256_digest
            ),
            expected_campaign_id_bytes=expected_campaign_id_bytes,
            expected_attempt_id_bytes=expected_attempt_id_bytes,
        )

    return verify_real_phase2b_actual_unsealed_960_authoritative_evidence_package0_v1


def _build_public_real_bundle_verifier_v1(implementation):
    def verify_real_phase2b_actual_unsealed_960_authoritative_evidence_bundle_v1(
        catalog_payload: bytes,
        raw_signer_registry_payload: bytes,
        raw_gate_evidence_payloads_in_exact_order: tuple[bytes, ...],
        raw_preallocation_lineage_payload: bytes,
        raw_gate_supporting_artifact_payload_tuples_in_exact_order: tuple[
            tuple[bytes, ...], ...
        ],
        raw_gate_source_native_payload_tuples_in_exact_order: tuple[
            tuple[bytes, ...], ...
        ],
        raw_stage_event_payloads_in_exact_order: tuple[bytes, ...],
        raw_package_envelope_payloads_in_exact_order: tuple[bytes, ...],
        raw_bundle_index_payload: bytes,
        raw_package0_ledger_event_payloads_in_exact_order: tuple[bytes, ...],
        raw_package0_stage_subject_event_payloads_in_exact_order: tuple[
            bytes, ...
        ],
        raw_store_if_absent_receipt_payload: bytes,
        raw_logical_time_token_payload: bytes,
        raw_governance_checkpoint_payload: bytes,
        raw_custodian_checkpoint_payload: bytes,
        raw_external_anchor_payload: bytes,
        raw_package10_denominator_event_payloads_in_exact_order: tuple[
            bytes, ...
        ],
        raw_attempt_universe_head_payload: bytes,
        raw_package10_denominator_finalization_payload: bytes,
        raw_package15_gate_result_payloads_in_exact_order: tuple[bytes, ...],
        raw_package16_attempt_record_payloads_in_exact_order: tuple[bytes, ...],
        *,
        out_of_band_expected_catalog_id_bytes: bytes,
        out_of_band_expected_catalog_payload_sha256_digest: bytes,
        out_of_band_expected_signer_registry_id_bytes: bytes,
        out_of_band_expected_signer_registry_payload_sha256_digest: bytes,
        out_of_band_expected_signer_registry_entries_root_sha256_digest: bytes,
        out_of_band_expected_preallocation_lineage_id_bytes: bytes,
        out_of_band_expected_preallocation_lineage_payload_sha256_digest: bytes,
        out_of_band_expected_governance_root_key_id_bytes: bytes,
        out_of_band_expected_governance_root_public_key_bytes: bytes,
        out_of_band_expected_custodian_root_key_id_bytes: bytes,
        out_of_band_expected_custodian_root_public_key_bytes: bytes,
        out_of_band_expected_attempt_universe_head_id_bytes: bytes,
        out_of_band_expected_attempt_universe_head_payload_sha256_digest: bytes,
        out_of_band_expected_readiness_dag_id_bytes: bytes,
        out_of_band_expected_readiness_dag_payload_sha256_digest: bytes,
        out_of_band_expected_progression_plan_payload_sha256_digest: bytes,
        out_of_band_expected_gap_inventory_source_sha256_digest: bytes,
        out_of_band_expected_statistical_validation_source_sha256_digest: bytes,
        expected_campaign_id_bytes: bytes,
        expected_attempt_id_bytes: bytes,
    ) -> NoReturn:
        return implementation(
            catalog_payload,
            raw_signer_registry_payload,
            raw_gate_evidence_payloads_in_exact_order,
            raw_preallocation_lineage_payload,
            raw_gate_supporting_artifact_payload_tuples_in_exact_order,
            raw_gate_source_native_payload_tuples_in_exact_order,
            raw_stage_event_payloads_in_exact_order,
            raw_package_envelope_payloads_in_exact_order,
            raw_bundle_index_payload,
            raw_package0_ledger_event_payloads_in_exact_order,
            raw_package0_stage_subject_event_payloads_in_exact_order,
            raw_store_if_absent_receipt_payload,
            raw_logical_time_token_payload,
            raw_governance_checkpoint_payload,
            raw_custodian_checkpoint_payload,
            raw_external_anchor_payload,
            raw_package10_denominator_event_payloads_in_exact_order,
            raw_attempt_universe_head_payload,
            raw_package10_denominator_finalization_payload,
            raw_package15_gate_result_payloads_in_exact_order,
            raw_package16_attempt_record_payloads_in_exact_order,
            out_of_band_expected_catalog_id_bytes=(
                out_of_band_expected_catalog_id_bytes
            ),
            out_of_band_expected_catalog_payload_sha256_digest=(
                out_of_band_expected_catalog_payload_sha256_digest
            ),
            out_of_band_expected_signer_registry_id_bytes=(
                out_of_band_expected_signer_registry_id_bytes
            ),
            out_of_band_expected_signer_registry_payload_sha256_digest=(
                out_of_band_expected_signer_registry_payload_sha256_digest
            ),
            out_of_band_expected_signer_registry_entries_root_sha256_digest=(
                out_of_band_expected_signer_registry_entries_root_sha256_digest
            ),
            out_of_band_expected_preallocation_lineage_id_bytes=(
                out_of_band_expected_preallocation_lineage_id_bytes
            ),
            out_of_band_expected_preallocation_lineage_payload_sha256_digest=(
                out_of_band_expected_preallocation_lineage_payload_sha256_digest
            ),
            out_of_band_expected_governance_root_key_id_bytes=(
                out_of_band_expected_governance_root_key_id_bytes
            ),
            out_of_band_expected_governance_root_public_key_bytes=(
                out_of_band_expected_governance_root_public_key_bytes
            ),
            out_of_band_expected_custodian_root_key_id_bytes=(
                out_of_band_expected_custodian_root_key_id_bytes
            ),
            out_of_band_expected_custodian_root_public_key_bytes=(
                out_of_band_expected_custodian_root_public_key_bytes
            ),
            out_of_band_expected_attempt_universe_head_id_bytes=(
                out_of_band_expected_attempt_universe_head_id_bytes
            ),
            out_of_band_expected_attempt_universe_head_payload_sha256_digest=(
                out_of_band_expected_attempt_universe_head_payload_sha256_digest
            ),
            out_of_band_expected_readiness_dag_id_bytes=(
                out_of_band_expected_readiness_dag_id_bytes
            ),
            out_of_band_expected_readiness_dag_payload_sha256_digest=(
                out_of_band_expected_readiness_dag_payload_sha256_digest
            ),
            out_of_band_expected_progression_plan_payload_sha256_digest=(
                out_of_band_expected_progression_plan_payload_sha256_digest
            ),
            out_of_band_expected_gap_inventory_source_sha256_digest=(
                out_of_band_expected_gap_inventory_source_sha256_digest
            ),
            out_of_band_expected_statistical_validation_source_sha256_digest=(
                out_of_band_expected_statistical_validation_source_sha256_digest
            ),
            expected_campaign_id_bytes=expected_campaign_id_bytes,
            expected_attempt_id_bytes=expected_attempt_id_bytes,
        )

    return verify_real_phase2b_actual_unsealed_960_authoritative_evidence_bundle_v1


verify_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1 = (
    _build_public_catalog_verifier_v1(_VERIFY_CATALOG_IMPLEMENTATION_V1)
)
verify_real_phase2b_actual_unsealed_960_authoritative_evidence_package0_v1 = (
    _build_public_real_package0_verifier_v1(
        _VERIFY_REAL_PACKAGE0_IMPLEMENTATION_V1
    )
)
verify_real_phase2b_actual_unsealed_960_authoritative_evidence_bundle_v1 = (
    _build_public_real_bundle_verifier_v1(_VERIFY_REAL_BUNDLE_IMPLEMENTATION_V1)
)

_REAL_PACKAGE0_CODE_V1 = (
    verify_real_phase2b_actual_unsealed_960_authoritative_evidence_package0_v1.__code__
)
_REAL_PACKAGE0_PRIVATE_CODE_V1 = _VERIFY_REAL_PACKAGE0_IMPLEMENTATION_V1.__code__
_REAL_PACKAGE0_POSITIONAL_NAMES_V1 = tuple(
    row[0]
    for row in FUTURE_REAL_PACKAGE0_VERIFIER_RAW_ABI_V1[
        "positional_parameters_in_exact_order"
    ]
)
_REAL_PACKAGE0_KEYWORD_ONLY_NAMES_V1 = tuple(
    FUTURE_REAL_PACKAGE0_VERIFIER_RAW_ABI_V1[
        "mandatory_keyword_only_byte_parameters_in_exact_order"
    ]
)
for _real_package0_code_v1 in (
    _REAL_PACKAGE0_PRIVATE_CODE_V1,
    _REAL_PACKAGE0_CODE_V1,
):
    if (
        _real_package0_code_v1.co_posonlyargcount != 0
        or _real_package0_code_v1.co_argcount
        != len(_REAL_PACKAGE0_POSITIONAL_NAMES_V1)
        or _real_package0_code_v1.co_kwonlyargcount
        != len(_REAL_PACKAGE0_KEYWORD_ONLY_NAMES_V1)
        or tuple(
            _real_package0_code_v1.co_varnames[
                : _real_package0_code_v1.co_argcount
            ]
        )
        != _REAL_PACKAGE0_POSITIONAL_NAMES_V1
        or tuple(
            _real_package0_code_v1.co_varnames[
                _real_package0_code_v1.co_argcount :
                _real_package0_code_v1.co_argcount
                + _real_package0_code_v1.co_kwonlyargcount
            ]
        )
        != _REAL_PACKAGE0_KEYWORD_ONLY_NAMES_V1
        or _real_package0_code_v1.co_flags & 0x04
        or _real_package0_code_v1.co_flags & 0x08
    ):
        raise RuntimeError("real package 0 NI stub ABI differs from frozen catalog")

_REAL_BUNDLE_CODE_V1 = (
    verify_real_phase2b_actual_unsealed_960_authoritative_evidence_bundle_v1.__code__
)
_REAL_BUNDLE_PRIVATE_CODE_V1 = _VERIFY_REAL_BUNDLE_IMPLEMENTATION_V1.__code__
_REAL_BUNDLE_POSITIONAL_NAMES_V1 = tuple(
    row[0]
    for row in FUTURE_REAL_BUNDLE_VERIFIER_RAW_ABI_V1[
        "positional_parameters_in_exact_order"
    ]
)
_REAL_BUNDLE_KEYWORD_ONLY_NAMES_V1 = tuple(
    FUTURE_REAL_BUNDLE_VERIFIER_RAW_ABI_V1[
        "mandatory_keyword_only_byte_parameters_in_exact_order"
    ]
)
for _real_bundle_code_v1 in (
    _REAL_BUNDLE_PRIVATE_CODE_V1,
    _REAL_BUNDLE_CODE_V1,
):
    if (
        _real_bundle_code_v1.co_posonlyargcount != 0
        or _real_bundle_code_v1.co_argcount
        != len(_REAL_BUNDLE_POSITIONAL_NAMES_V1)
        or _real_bundle_code_v1.co_kwonlyargcount
        != len(_REAL_BUNDLE_KEYWORD_ONLY_NAMES_V1)
        or tuple(
            _real_bundle_code_v1.co_varnames[
                : _real_bundle_code_v1.co_argcount
            ]
        )
        != _REAL_BUNDLE_POSITIONAL_NAMES_V1
        or tuple(
            _real_bundle_code_v1.co_varnames[
                _real_bundle_code_v1.co_argcount :
                _real_bundle_code_v1.co_argcount
                + _real_bundle_code_v1.co_kwonlyargcount
            ]
        )
        != _REAL_BUNDLE_KEYWORD_ONLY_NAMES_V1
        or _real_bundle_code_v1.co_flags & 0x04
        or _real_bundle_code_v1.co_flags & 0x08
    ):
        raise RuntimeError("real bundle NI stub ABI differs from frozen catalog")

del _VERIFY_CATALOG_IMPLEMENTATION_V1
del _VERIFY_REAL_PACKAGE0_IMPLEMENTATION_V1
del _VERIFY_REAL_BUNDLE_IMPLEMENTATION_V1
del _build_private_catalog_boundary_v1
del _build_private_package0_boundary_v1
del _build_public_catalog_verifier_v1
del _build_public_real_package0_verifier_v1
del _build_public_real_bundle_verifier_v1
del _REAL_PACKAGE0_CODE_V1
del _REAL_PACKAGE0_PRIVATE_CODE_V1
del _real_package0_code_v1
del _REAL_PACKAGE0_POSITIONAL_NAMES_V1
del _REAL_PACKAGE0_KEYWORD_ONLY_NAMES_V1
del _REAL_BUNDLE_CODE_V1
del _REAL_BUNDLE_PRIVATE_CODE_V1
del _real_bundle_code_v1
del _REAL_BUNDLE_POSITIONAL_NAMES_V1
del _REAL_BUNDLE_KEYWORD_ONLY_NAMES_V1


__all__ = (
    "SCHEMA_VERSION_V1",
    "STATUS_V1",
    "CLAIM_LEVEL_V1",
    "CANONICALIZATION_V1",
    "REAL_BUNDLE_VERIFICATION_STATUS_V1",
    "REAL_PACKAGE0_VERIFICATION_STATUS_V1",
    "PACKAGE_OUTER_STATUS_V1",
    "PACKAGE_OUTER_CLAIM_LEVEL_V1",
    "SIGNATURE_ALGORITHM_V1",
    "MAXIMUM_CATALOG_PAYLOAD_BYTES_V1",
    "CONTENT_ID_PREIMAGE_PROFILE_V1",
    "CONTENT_ID_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1",
    "CONTENT_ID_FORMULA_V1",
    "DETACHED_SIGNATURE_PREIMAGE_PROFILE_V1",
    "DETACHED_SIGNATURE_PREIMAGE_SCHEMA_VERSION_V1",
    "DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_V1",
    "DETACHED_SIGNATURE_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1",
    "DETACHED_SIGNATURE_FORMULA_V1",
    "DETACHED_SIGNATURE_EXCLUDED_FIELDS_IN_EXACT_ORDER_V1",
    "EXACT_SPECIAL_WIRE_TYPE_PROFILES_IN_EXACT_ORDER_V1",
    "INSPECTED_READINESS_DAG_CONFIG_SHA256_V1",
    "INSPECTED_PROGRESSION_PLAN_SHA256_V1",
    "INSPECTED_GAP_INVENTORY_SOURCE_SHA256_V1",
    "INSPECTED_STATISTICAL_VALIDATION_SOURCE_SHA256_V1",
    "INSPECTED_READINESS_DAG_ID_V1",
    "INSPECTED_DEPENDENCIES_IN_EXACT_ORDER_V1",
    "CatalogDispositionV1",
    "CatalogReasonV1",
    "Package0DispositionV1",
    "Package0ReasonV1",
    "AttemptDispositionV1",
    "Phase2BActualUnsealed960AuthoritativeEvidenceSchemaCatalogErrorV1",
    "VerifiedPhase2BActualUnsealed960AuthoritativeEvidenceSchemaCatalogV1",
    "Phase2BActualUnsealed960AuthoritativeEvidencePackage0ErrorV1",
    "VerifiedPhase2BActualUnsealed960AuthoritativeEvidencePackage0V1",
    "GATE_SPECS_IN_EXACT_ORDER_V1",
    "GATE_SOURCE_SPECS_IN_EXACT_ORDER_V1",
    "GATE_VERIFICATION_AVAILABILITIES_IN_EXACT_ORDER_V1",
    "CURRENT_GATE_VERIFICATION_AVAILABILITY_V1",
    "GATE_VERIFIER_NOT_IMPLEMENTED_FAILURE_CODE_V1",
    "GATE_LINEAGE_SCOPES_IN_EXACT_ORDER_V1",
    "PREALLOCATION_LINEAGE_SCHEMA_OBJECT_V1",
    "GATE_SOURCE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1",
    "GATE_SUPPORT_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1",
    "GATE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1",
    "STAGE_NAMES_IN_EXACT_ORDER_V1",
    "STAGE_SPECS_IN_EXACT_ORDER_V1",
    "STAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1",
    "INTERLEAVED_DAG_EDGES_IN_EXACT_ORDER_V1",
    "PACKAGE_SPECS_IN_EXACT_ORDER_V1",
    "PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1",
    "BUNDLE_INDEX_SCHEMA_OBJECT_V1",
    "ATTEMPT_POLICY_V1",
    "TERMINAL_DISPOSITION_POLICY_V1",
    "PACKAGE_CATALOG_EVIDENCE_DISPOSITION_V1",
    "PACKAGE_PRODUCER_EVIDENCE_DISPOSITION_V1",
    "PACKAGE_VERIFIER_SUCCESS_DISPOSITION_V1",
    "PACKAGE_VERIFIER_FAILURE_DISPOSITIONS_IN_EXACT_ORDER_V1",
    "PACKAGE_REQUIRED_EXTERNAL_EVIDENCE_CLASS_V1",
    "FROZEN_STATISTICAL_THRESHOLD_IDS_IN_EXACT_ORDER_V1",
    "FROZEN_STATISTICAL_THRESHOLDS_IN_EXACT_ORDER_V1",
    "VALIDATION_OVERALL_GATE_SPECS_IN_EXACT_ORDER_V1",
    "VALIDATION_FAMILIES_IN_EXACT_ORDER_V1",
    "VALIDATION_SCALES_IN_EXACT_ORDER_V1",
    "VALIDATION_SLICE_GATE_BASE_SPECS_IN_EXACT_ORDER_V1",
    "FROZEN_VALIDATION_GATE_RESULT_PLAN_IN_EXACT_ORDER_V1",
    "FROZEN_VALIDATION_STATISTICAL_GATE_IDS_IN_EXACT_ORDER_V1",
    "FROZEN_VALIDATION_MINIMUM_PASSING_CORRECT_COUNTS_IN_EXACT_ORDER_V1",
    "PERMANENT_NONCLAIMS_IN_EXACT_ORDER_V1",
    "GATE_SCHEMA_IDS_ROOT_V1",
    "GATE_SOURCE_SCHEMA_IDS_ROOT_V1",
    "GATE_SUPPORT_SCHEMA_IDS_ROOT_V1",
    "STAGE_SCHEMA_IDS_ROOT_V1",
    "PACKAGE_SCHEMA_IDS_ROOT_V1",
    "PACKAGE_SCHEMA_DAG_ROOT_V1",
    "DAG_ROOT_V1",
    "CATALOG_ID_V1",
    "CATALOG_PAYLOAD_V1",
    "CATALOG_PAYLOAD_SHA256_V1",
    "PACKAGE0_VERIFIED_RESULT_FIELDS_IN_EXACT_ORDER_V1",
    "FUTURE_REAL_PACKAGE0_VERIFIED_RESULT_CONTRACT_V1",
    "FUTURE_REAL_PACKAGE0_VERIFIER_RAW_ABI_V1",
    "FUTURE_REAL_BUNDLE_VERIFIER_RAW_ABI_V1",
    "canonical_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_bytes_v1",
    "verify_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1",
    "require_verified_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1",
    "verify_real_phase2b_actual_unsealed_960_authoritative_evidence_package0_v1",
    "require_verified_phase2b_actual_unsealed_960_authoritative_evidence_package0_v1",
    "verify_real_phase2b_actual_unsealed_960_authoritative_evidence_bundle_v1",
)
