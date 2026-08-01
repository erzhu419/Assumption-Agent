"""Phase-3A M2.5 deterministic formal-wire foundation.

This module implements the byte-level decisions that are unambiguous in
``hegel-freeze-p2b-p3-v1.1.2``.  Deterministic candidate roots are deliberately
separate from authoritative evidence: no function here generates a seed or
key, signs an object, advances a gate, or authorizes an M3 run.  The few
remaining normative conflicts fail with :data:`FAIL_M25_NORMATIVE_GAP` instead
of guessing a wire format.

The actual deterministic CBOR codec lives in :mod:`strict_cbor_v1`.  Keeping
the codec shared avoids a subtly different "M2.5 CBOR" implementation while
still requiring exact decode/re-encode at every formal-object boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import re
import time
from types import MappingProxyType
from typing import Final, Mapping, Sequence

from .strict_cbor_v1 import (
    StrictCborValue,
    canonical_cbor_decode,
    canonical_cbor_encode,
    content_hash,
    rfc6962_root,
)


FAIL_M25_NORMATIVE_GAP: Final = "FAIL_M25_NORMATIVE_GAP"
MACHINE_FREEZE_ID: Final = "hegel-freeze-p2b-p3-v1.1.2"
ID_DIGEST_DOMAIN: Final = b"HEGEL/ID_DIGEST/V1"
CANONICAL_INPUT_DOMAIN: Final = "HEGEL/CANONICAL_INPUT/V1"
AUTHORITATIVE_MIN_TIMESTAMP: Final = 1_704_067_200
MAX_TIMESTAMP: Final = 253_402_300_799
MAX_AUTHORITATIVE_FUTURE_SKEW_SECONDS: Final = 300
BRIDGE_ATTESTATION_SIGNATURE_DOMAIN: Final = "HEGEL/BRIDGE_ATTESTATION_SIGNATURE/V1"
PARENT_AUDITOR_SIGNATURE_DOMAIN: Final = "HEGEL/PARENT_ABSENCE_AUDITOR_SIGNATURE/V2"
M3_RUN_OUTPUT_SLOT_COUNT: Final = 15
AUDITED_PARENT_COMMIT_SHA1: Final = bytes.fromhex(
    "fb3a3ee4865a140c558821017ddd3e9a6a99de48"
)

CUSTODIAN_SIGNATURE_DOMAIN_BY_TAG: Final = MappingProxyType(
    {
        0x3103: "HEGEL/CUSTODIAN_SPLIT_SEED_COMMITMENT_SIGNATURE/V1",
        0x3105: "HEGEL/CUSTODIAN_BINDING_SIGNATURE/V1",
        0x3106: "HEGEL/CUSTODIAN_SEED_CONTINUITY_SIGNATURE/V1",
        0x3108: "HEGEL/CUSTODIAN_LEDGER_GENESIS_SIGNATURE/V1",
    }
)
EXTERNAL_INPUT_SIGNED_TAG_PURPOSES: Final = (
    (1, 0x3103),
    (1, 0x3105),
    (1, 0x3106),
    (1, 0x3108),
    (4, 0x3114),
)

_MACHINE_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")


def id_digest_v1(machine_id: str) -> bytes:
    """Return the frozen digest of one exact ASCII machine-ID string."""

    if not isinstance(machine_id, str):
        _fail("REJECT_MACHINE_ID_NON_ASCII", "machine ID must be ASCII text")
    try:
        raw = machine_id.encode("ascii")
    except UnicodeEncodeError:
        _fail("REJECT_MACHINE_ID_NON_ASCII", "machine ID contains non-ASCII text")
    if not 1 <= len(raw) <= 256:
        _fail("REJECT_MACHINE_ID_LENGTH", "machine ID length must be in [1, 256]")
    if _MACHINE_ID_RE.fullmatch(machine_id) is None:
        _fail("REJECT_MACHINE_ID_SYNTAX", "machine ID violates IdDigestV1 syntax")
    return hashlib.sha256(ID_DIGEST_DOMAIN + b"\x00" + raw).digest()


LEGACY_PARENT_SOURCE_IDS: Final = (
    "target_spec_b491c0a9719fb0279fe02798ede026e440c17a539965514145a7818b15387ac3",
    "sink_control_spec_7fd6f9a6e2b4c6eda0c7e1545ad42cb19666743ede8ed87f40d82c0ef46198a0",
)
_LEGACY_PARENT_SOURCE_EXPECTATIONS: Final = MappingProxyType(
    {
        1: (1, bytes.fromhex(LEGACY_PARENT_SOURCE_IDS[0].split("_", 2)[-1])),
        2: (2, bytes.fromhex(LEGACY_PARENT_SOURCE_IDS[1].rsplit("_", 1)[-1])),
    }
)


def validate_timestamp_v1(
    timestamp: int,
    *,
    authoritative: bool = False,
    verifier_unix_seconds: int | None = None,
) -> int:
    """Validate a synthetic or authoritative Unix-second timestamp."""

    if type(timestamp) is not int or not 0 <= timestamp <= MAX_TIMESTAMP:
        _fail("REJECT_TIMESTAMP_OUT_OF_RANGE", "timestamp is outside the wire range")
    if not authoritative:
        return timestamp
    if timestamp < AUTHORITATIVE_MIN_TIMESTAMP:
        _fail("FAIL_AUTHORITATIVE_TIMESTAMP_ZERO", "authoritative timestamp is too early")
    now = int(time.time()) if verifier_unix_seconds is None else verifier_unix_seconds
    if type(now) is not int or not 0 <= now <= MAX_TIMESTAMP:
        _fail("REJECT_TIMESTAMP_OUT_OF_RANGE", "verifier timestamp is outside the wire range")
    if timestamp > now + MAX_AUTHORITATIVE_FUTURE_SKEW_SECONDS:
        _fail("FAIL_TIMESTAMP_EXCESSIVELY_FUTURE", "timestamp exceeds allowed future skew")
    return timestamp


def validate_timestamp_ordering_v1(earlier: int, later: int) -> None:
    """Require nondecreasing timestamps after validating their wire ranges."""

    validate_timestamp_v1(earlier)
    validate_timestamp_v1(later)
    if earlier > later:
        _fail("FAIL_TIMESTAMP_ORDERING", "timestamps are not nondecreasing")


def validate_opaque_id128_v1(
    value: object,
    *,
    seen: set[bytes] | frozenset[bytes] | None = None,
) -> bytes:
    """Validate the wire identity and optional duplicate scope of an opaque ID."""

    if type(value) is not bytes or len(value) != 16:
        _fail("REJECT_M25_FIELD_TYPE", "opaque ID must be exactly 16 bytes")
    if value == bytes(16):
        _fail("FAIL_OPAQUE_ID_ALL_ZERO", "opaque ID may not be all zero")
    if seen is not None and value in seen:
        _fail("FAIL_OPAQUE_ID_ALREADY_USED", "opaque ID already exists in the trust scope")
    return value


@dataclass(frozen=True)
class NumericEnumRegistry:
    """One immutable numeric enum table with fail-closed lookup semantics."""

    name: str
    entries: Mapping[int, str]
    zero_is_valid: bool = False
    tombstones: frozenset[int] = frozenset()

    def validate(self, value: object, *, field: str) -> int:
        if type(value) is not int:
            _fail("REJECT_M25_FIELD_TYPE", f"{field} must be a numeric enum")
        assert isinstance(value, int)
        if value in self.tombstones:
            _fail("REJECT_TOMBSTONED_ENUM_VALUE", f"{field} uses a tombstoned enum value")
        if value == 0 and not self.zero_is_valid:
            _fail("REJECT_UNKNOWN_ENUM_VALUE", f"{field} may not use INVALID/UNSPECIFIED")
        if value not in self.entries:
            _fail("REJECT_UNKNOWN_ENUM_VALUE", f"{field} uses an unknown enum value")
        if 32768 <= value <= 65535:
            _fail("REJECT_UNKNOWN_ENUM_VALUE", f"{field} uses a private extension value")
        return value


def _enum(
    name: str,
    names: Sequence[str],
    *,
    start: int = 0,
    zero_is_valid: bool = False,
) -> NumericEnumRegistry:
    return NumericEnumRegistry(
        name=name,
        entries=MappingProxyType({index + start: item for index, item in enumerate(names)}),
        zero_is_valid=zero_is_valid,
    )


NUMERIC_ENUM_REGISTRIES: Final = MappingProxyType(
    {
        "InputSignatureId": _enum(
            "InputSignatureId",
            ("INVALID", "ODD_BITSET_ENTITY_SET_V1", "OBSERVED_OMITTED_SINK_TUPLE_V1"),
        ),
        "SortId": _enum(
            "SortId",
            (
                "INVALID", "BOOL", "BIT", "SIGN", "BOUNDED_INT", "RATIONAL_VALUE",
                "RATIONAL_PARAMETER", "TOLERANCE", "CLOSED_INTERVAL", "ENTITY_SLOT",
                "INDEX", "QUANTITY_ID", "CONTEXT_ID", "ROLE_ID", "SCALE_ID", "TASK_ID",
                "ENTITY_SET", "SCOPE_ID", "AGGREGATE_MAP_ID", "TRANSFORM_ID",
                "OBSERVATION", "EVENT",
            ),
        ),
        "RegistryKindId": _enum(
            "RegistryKindId",
            (
                "INVALID", "ENTITY_SLOT", "QUANTITY", "CONTEXT", "ROLE", "SCALE",
                "TASK", "SCOPE", "AGGREGATE_MAP", "TRANSFORM", "OPERATOR", "NEW_SYMBOL",
            ),
        ),
        "RegistryEntryStateId": _enum(
            "RegistryEntryStateId", ("INVALID", "ACTIVE", "TOMBSTONE")
        ),
        "OperatorClassId": _enum(
            "OperatorClassId",
            ("INVALID", "LEAF", "UNARY", "BINARY", "TERNARY", "CONJUNCTION", "AGGREGATE_MAP", "ADAPTER_TRANSFORM"),
        ),
        "OperatorAdmissionStateId": _enum(
            "OperatorAdmissionStateId",
            ("INVALID", "ACTIVE_DSL", "TOMBSTONE_REMOVED", "ADAPTER_ONLY", "RESERVED_NOT_IMPLEMENTED"),
        ),
        "UndefinedSemanticsId": _enum(
            "UndefinedSemanticsId",
            (
                "INVALID", "TOTAL_NO_BOTTOM", "STRICT_BOTTOM_PROPAGATION",
                "LEAF_INDEX_OUT_OF_RANGE_BOTTOM", "EMPTY_AGGREGATE_BOTTOM",
                "MISSING_TYPED_MEASUREMENT_BOTTOM", "RATIONAL_DOMAIN_OVERFLOW_BOTTOM",
            ),
        ),
        "ArtifactRoleId": _enum(
            "ArtifactRoleId",
            (
                "INVALID", "OUTSIDE_TARGET_SPEC", "OUTSIDE_TARGET_UNIVERSE",
                "OUTSIDE_TARGET_TRUTH", "IN_LANGUAGE_NULL_SPEC", "IN_LANGUAGE_NULL_UNIVERSE",
                "IN_LANGUAGE_NULL_TRUTH", "CHILD_DSL_SPEC", "OPERATOR_SEMANTICS",
                "IDENTIFIER_REGISTRY", "CANONICAL_AST_SCHEMA", "CANONICAL_CBOR_PROFILE",
                "SPLIT_CONTRACT", "DISCOVERY_SPLIT", "VALIDATION_SPLIT",
                "SEALED_PREDICTION_SPLIT", "SHRINK_TRANSITION", "M3_EXECUTION",
                "NORMATIVE_APPROVAL", "CUSTODIAN_BINDING", "PARENT_ABSENCE_ATTESTATION",
                "FALLBACK_REGISTRY",
            ),
        ),
        "DiagnosticNamespaceId": _enum(
            "DiagnosticNamespaceId",
            (
                "INVALID", "target_spec", "sink_control_spec", "bounded_universe",
                "target_truth_table", "dsl_spec", "operator_semantics", "identifier_registry",
                "canonical_ast_schema", "canonical_cbor_profile", "split_contract",
                "hidden_generator_spec", "publication", "replay", "freeze_document",
            ),
        ),
        "FormalObjectKindId": _enum(
            "FormalObjectKindId",
            ("INVALID", "CONTENT_HASH", "RFC6962_TREE_ROOT", "SIGNED_MANIFEST_ROOT", "ARCHIVE_ROOT", "RECEIPT_ROOT", "EXECUTION_MANIFEST_ROOT"),
        ),
        "DiagnosticProfileId": _enum(
            "DiagnosticProfileId", ("INVALID", "HEGEL_LEGACY_STABLE_JSON_V1", "RFC8785_JCS_V1")
        ),
        "FormalProfileId": _enum(
            "FormalProfileId", ("INVALID", "HEGEL_CBOR_CONTENT_HASH_V1", "HEGEL_RFC6962_ROW_TREE_V1", "HEGEL_RFC6962_ARCHIVE_TREE_V1")
        ),
        "StratumId": _enum(
            "StratumId",
            (
                "ODD_SIZE5_LABEL0", "ODD_SIZE5_LABEL1", "ODD_SIZE6_LABEL0", "ODD_SIZE6_LABEL1",
                "ODD_SIZE7_LABEL0", "ODD_SIZE7_LABEL1", "ODD_SIZE8_LABEL0", "ODD_SIZE8_LABEL1",
                "SINK_D0", "SINK_D1", "SINK_D2", "SINK_D3", "SINK_D4",
            ),
            start=1,
        ),
        "PartitionId": _enum(
            "PartitionId", ("INVALID", "DISCOVERY", "VALIDATION", "SEALED_PREDICTION")
        ),
        "EquivalenceModeId": _enum("EquivalenceModeId", ("INVALID", "EXACT_EXTENSIONAL")),
        "ImplementationId": _enum(
            "ImplementationId", ("INVALID", "PYTHON_REFERENCE", "RUST_INDEPENDENT", "CUSTODIAN", "AUDITOR")
        ),
        "ParentStatusId": _enum(
            "ParentStatusId", ("INVALID", "COMPLETE", "DSL_TOO_LARGE", "INCONCLUSIVE_BUDGET", "INCONCLUSIVE_SEMANTICS", "INCONCLUSIVE_EXECUTION")
        ),
        "ChildInitialStateId": _enum("ChildInitialStateId", ("INVALID", "NOT_RUN")),
        "M3TransitionReasonId": _enum(
            "M3TransitionReasonId",
            (
                "INVALID", "ENTRY_GATES_24_OF_24", "ENUMERATION_FRONTIER_EXHAUSTED",
                "CANONICAL_PROGRAM_50001_ACCEPTED", "RAW_OPERATOR_CAP_HIT",
                "WALL_CLOCK_BUDGET_HIT", "SEMANTICS_OR_DUAL_REPLAY_MISMATCH",
                "EXECUTION_FAILURE", "ROLE_EVALUATION_COMPLETE",
            ),
        ),
        "M3ClosureStatusId": _enum(
            "M3ClosureStatusId",
            ("NOT_RUN", "COMPLETE", "DSL_TOO_LARGE", "INCONCLUSIVE_BUDGET", "INCONCLUSIVE_SEMANTICS", "INCONCLUSIVE_EXECUTION"),
            zero_is_valid=True,
        ),
        "M3StateId": _enum(
            "M3StateId",
            ("NOT_RUN", "RUNNING", "COMPLETE", "DSL_TOO_LARGE", "INCONCLUSIVE_BUDGET", "INCONCLUSIVE_SEMANTICS", "INCONCLUSIVE_EXECUTION"),
            zero_is_valid=True,
        ),
        "M3RunningPhaseId": _enum(
            "M3RunningPhaseId", ("NONE", "CANONICAL_ENUMERATION", "ROLE_EVALUATION"), zero_is_valid=True
        ),
        "RoleAgreementStatusId": _enum(
            "RoleAgreementStatusId", ("NOT_APPLICABLE", "AGREED", "DISAGREED"), zero_is_valid=True
        ),
        "ActorPurposeId": _enum(
            "ActorPurposeId",
            (
                "CUSTODIAN_IDENTITY_AND_BRIDGE_ATTESTER",
                "PYTHON_BRIDGE_ATTESTER",
                "RUST_BRIDGE_ATTESTER",
                "PARENT_ABSENCE_AUDITOR",
                "FINAL_CERTIFICATE_SIGNER_RESERVED_FOR_M4",
            ),
            start=1,
        ),
        "ClaimLevelId": _enum(
            "ClaimLevelId",
            (
                "FALSE_INVENTION_NULL_ONLY",
                "MECHANISM_SPECIFIC_RECOVERY",
                "OUTSIDE_TARGET_CANDIDATE",
            ),
            start=1,
        ),
        "TargetRoleId": _enum(
            "TargetRoleId", ("OUTSIDE_TARGET", "IN_LANGUAGE_NULL"), start=1
        ),
        "MismatchKindId": _enum(
            "MismatchKindId",
            (
                "CANONICAL_PROGRAM_COUNT",
                "CANONICAL_PROGRAM_ARCHIVE_ROOT",
                "PROGRAM_OUTPUT_ARCHIVE_ROOT",
                "BUCKET_ACCOUNTING_ROOT",
                "FIRST_OUT_OF_BUDGET_WITNESS",
                "ROLE_MATCH_SET",
                "RECEIPT_FIELD_PRESENCE",
                "EXECUTION_ENVIRONMENT_BINDING",
            ),
            start=1,
        ),
        "AssignmentOrderingRuleId": _enum(
            "AssignmentOrderingRuleId",
            ("PER_STRATUM_RANK_THEN_QUOTA", "UNIVERSE_INDEX_WITHIN_PARTITION"),
            start=1,
        ),
        "FallbackSplitPolicyId": _enum(
            "FallbackSplitPolicyId",
            ("NEW_TARGET_NEW_SPLIT_FIRST_INSTANTIATION",),
            start=1,
        ),
        "RankTieBreakRuleId": _enum(
            "RankTieBreakRuleId", ("RANK_DIGEST_THEN_CANONICAL_INPUT_HASH",), start=1
        ),
        "TraversalFieldId": _enum(
            "TraversalFieldId",
            (
                "AST_DEPTH",
                "AST_NODE_COUNT",
                "OUTPUT_SORT_ID",
                "ROOT_OPERATOR_ID",
                "CANONICAL_AST_CBOR_BYTES",
            ),
            start=1,
        ),
        "BucketFieldId": _enum(
            "BucketFieldId", ("OUTPUT_SORT_ID", "AST_DEPTH", "AST_NODE_COUNT"), start=1
        ),
        "AccountingCounterFieldId": _enum(
            "AccountingCounterFieldId",
            (
                "RAW_OPERATOR_APPLICATIONS",
                "ACCEPTED_CANONICAL_PROGRAMS",
                "SYNTACTIC_DUPLICATES",
                "TYPE_REJECTIONS",
                "STRUCTURAL_LIMIT_REJECTIONS",
                "REWRITE_COLLAPSES",
            ),
            start=1,
        ),
        "AccountingInvariantId": _enum(
            "AccountingInvariantId",
            (
                "SUM_ACCEPTED_EQUALS_RECEIPT_CANONICAL_PROGRAM_COUNT",
                "SUM_RAW_APPLICATIONS_EQUALS_RECEIPT_RAW_APPLICATION_COUNT",
                "NONNULL_PROGRAM_INDEX_RANGE_SIZE_EQUALS_ACCEPTED_COUNT",
            ),
            start=1,
        ),
        "OpaqueIdKindId": _enum(
            "OpaqueIdKindId", ("RUN_ID", "LEDGER_ID"), start=1
        ),
        "NormativeDocumentRoleId": _enum(
            "NormativeDocumentRoleId",
            ("BASE_AMENDMENT", "ERRATA_RESOLUTION", "IMPLEMENTATION_CLOSURE_ADDENDUM"),
            start=1,
        ),
        "DependencyEcosystemId": _enum(
            "DependencyEcosystemId", ("PYTHON", "RUST", "SYSTEM"), start=1
        ),
        "GitObjectAlgorithmId": _enum(
            "GitObjectAlgorithmId", ("SHA1",), start=1
        ),
        "SeedStateId": _enum(
            "SeedStateId", ("SPEC_FROZEN_SEED_NOT_INSTANTIATED", "SEED_INSTANTIATED", "COMPROMISED_REQUIRES_NEW_VERSION"), start=1
        ),
        # Retained as a construction-API compatibility alias only.  Errata E8
        # maps every target execution role field to TargetRoleId.
        "DslRoleId": _enum("DslRoleId", ("OUTSIDE_TARGET", "IN_LANGUAGE_NULL"), start=1),
        "ApprovalStatusId": _enum("ApprovalStatusId", ("APPROVED", "REJECTED", "SUPERSEDED"), start=1),
        "ApprovalMethodId": _enum(
            "ApprovalMethodId", ("USER_DECISION_RECORDED_IN_COMMITTED_NORMATIVE_DOCUMENT", "EXTERNAL_DIGITAL_SIGNATURE"), start=1
        ),
        "SplitInstantiationStatusId": _enum(
            "SplitInstantiationStatusId", ("FIRST_INSTANTIATION", "VERIFIED_REUSE", "FRESH_AFTER_COMPROMISE"), start=1
        ),
        "SeedContinuityStatusId": _enum(
            "SeedContinuityStatusId", ("FIRST_INSTANTIATION_AFTER_SPEC_FREEZE", "VERIFIED_PARENT_SEED_REUSE", "FRESH_VERSION_AFTER_COMPROMISE"), start=1
        ),
        "HiddenAccessEventTypeId": _enum(
            "HiddenAccessEventTypeId", ("SPLIT_SEED_FIRST_INSTANTIATION", "HIDDEN_ARTIFACT_ACCESS_GRANTED", "HIDDEN_ARTIFACT_ACCESS_DENIED", "HIDDEN_ARTIFACT_REVEALED"), start=1
        ),
    }
)


class M25WireError(ValueError):
    """Stable fail-closed error raised by the non-authoritative foundation."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> "None":
    raise M25WireError(code, detail)


def _normative_gap(name: str, detail: str) -> "None":
    _fail(FAIL_M25_NORMATIVE_GAP, f"{name}: {detail}")


@dataclass(frozen=True)
class FormalSchema:
    """One numeric-array schema copied literally from the amendment."""

    name: str
    tag: int
    schema_id: bytes
    fields: tuple[str, ...]
    hash_domain: str | None = None
    rfc6962_records: bool = False
    ordering_fields: tuple[str, ...] = ()
    wire_gap: str | None = None

    @property
    def prefix(self) -> tuple[int, int, bytes]:
        return (1, self.tag, self.schema_id)


def _schema(
    name: str,
    tag: int,
    schema_id: str,
    fields: tuple[str, ...],
    *,
    hash_domain: str | None = None,
    rfc6962_records: bool = False,
    ordering_fields: tuple[str, ...] = (),
    wire_gap: str | None = None,
) -> FormalSchema:
    return FormalSchema(
        name=name,
        tag=tag,
        schema_id=schema_id.encode("ascii"),
        fields=fields,
        hash_domain=hash_domain,
        rfc6962_records=rfc6962_records,
        ordering_fields=ordering_fields,
        wire_gap=wire_gap,
    )


# The tag table includes inherited v1.1.1 objects and every v1.1.2 addition.
# A tag remains distinct from authority to publish an object under that tag.
OBJECT_TAGS: Final = MappingProxyType(
    {
        "NormativeDocumentBlobV1": 0x3001,
        "FreezeSpecV1": 0x3002,
        "DslSpecV1": 0x3003,
        "SplitContractV1": 0x3004,
        "TargetBundleV1": 0x3005,
        "ApprovalEvidenceBundleV1": 0x3006,
        "ReplacementPolicyV1": 0x3007,
        "SplitSpecFreezeV1": 0x3008,
        "TombstonePolicyV1": 0x3009,
        "CrossDslHashPolicyV1": 0x300A,
        "FallbackRegistryV1": 0x300B,
        "ImplementationBindingV1": 0x300C,
        "TraversalContractV1": 0x300D,
        "BucketAccountingContractV1": 0x300E,
        "ProgramArchiveContractV1": 0x300F,
        "OutputArchiveContractV1": 0x3010,
        "StateMachineContractV1": 0x3011,
        "RowTransformSpecV1": 0x3012,
        "InputSignatureSpecV1": 0x3013,
        "TargetSpecFormalV1": 0x3014,
        "SplitAlgorithmSpecV1": 0x3015,
        "ExecutionEnvironmentSpecV1": 0x3016,
        "NormativeDocumentBundleV1": 0x3018,
        "CanonicalAstProfileSpecV1": 0x3019,
        "CanonicalCborProfileSpecV1": 0x301A,
        "Phase2BContractSpecV1": 0x301B,
        "MdlCodeTableSpecV1": 0x301C,
        "StaticRoleMetadataV1": 0x301D,
        "HiddenArtifactScopeV1": 0x301E,
        "NormativeApprovalManifestV1": 0x3101,
        "DslRoleBindingManifestV1": 0x3102,
        "SplitSeedCommitmentManifestV1": 0x3103,
        "SplitBindingManifestV1": 0x3104,
        "CustodianBindingManifestV1": 0x3105,
        "SeedContinuityManifestV1": 0x3106,
        "ParentManifestAbsenceAttestationV1": 0x3107,
        "HiddenAccessLedgerRecordV1": 0x3108,
        "DslShrinkTransitionFormalV1": 0x3109,
        "M3ExecutionManifestV1": 0x310A,
        "CustodianBindingCoreV1": 0x310B,
        "ActorKeyManifestV1": 0x310C,
        "AttestationBundleV1": 0x310D,
        "BridgeReplayStatementV1": 0x310E,
        "M3ExecutionCandidateV1": 0x310F,
        "M3ExecutionManifestV2": 0x3110,
        "ActorTrustGenesisV1": 0x3111,
        "OpaqueIdRegistrySnapshotV1": 0x3112,
        "ParentAbsenceAuditBundleV1": 0x3113,
        "ParentManifestAbsenceAttestationV2": 0x3114,
        "OpaqueIdRegistrationIntentV1": 0x3115,
        "SignedManifestEnvelopeV1": 0x31FF,
        "BoundedUniverseRowV1": 0x3201,
        "TargetTruthRowV1": 0x3202,
        "SplitAssignmentRowV1": 0x3203,
        "IdentifierRegistryEntryV1": 0x3204,
        "OperatorSemanticsEntryV1": 0x3205,
        "DiagnosticFormalBridgeRecordV1": 0x3206,
        "CanonicalProgramRecordV2": 0x3207,
        "ProgramOutputRecordV2": 0x3208,
        "ProgramChunkManifestV2": 0x3209,
        "RoleOutputChunkManifestV2": 0x320A,
        "MatchRecordV2": 0x320B,
        "BucketAccountingRecordV1": 0x320C,
        "MismatchRecordV1": 0x320D,
        "PartialDiagnosticBundleV1": 0x320E,
        "AuditedPathBlobRecordV1": 0x3210,
        "AuditedHistoryRowV1": 0x3211,
        "LegacyParentSourceRowV1": 0x3212,
        "RepositoryPathAliasRecordV1": 0x3213,
        "SourceFileRecordV1": 0x3215,
        "DependencyLockRecordV1": 0x3216,
        "LegalTransitionRowV1": 0x3217,
        "OpaqueIdRegistryRecordV1": 0x3218,
        "M3RunGenesisV1": 0x3300,
        "M3RunStateRecordV1": 0x3301,
        "M3ImplementationEnumerationReceiptV1": 0x3302,
        "M3RoleEvaluationReceiptV1": 0x3303,
        "M3DualReplayAgreementV1": 0x3304,
        "M3RoleAgreementEntryV1": 0x3305,
        "OddInputV1": 0x3401,
        "SinkInputV1": 0x3402,
    }
)


_SCHEMAS = (
    _schema(
        "NormativeDocumentBlobV1",
        0x3001,
        "hegel-normative-document-blob/1",
        (
            "repository_relative_path_id_digest",
            "raw_git_blob_bytes",
            "repository_commit_id",
        ),
        hash_domain="HEGEL/NORMATIVE_DOCUMENT/V1",
    ),
    _schema(
        "FreezeSpecV1",
        0x3002,
        "hegel-freeze-spec/1",
        (
            "freeze_version_id_digest",
            "parent_freeze_root_or_null",
            "child_dsl_spec_root",
            "phase2b_contract_root",
            "canonical_ast_schema_root",
            "canonical_cbor_profile_root",
            "mdl_code_table_root",
            "amendment_document_root",
            "effective_repository_commit_id",
        ),
        hash_domain="HEGEL/FREEZE_SPEC/V1",
    ),
    _schema(
        "DslSpecV1",
        0x3003,
        "hegel-dsl-spec/1",
        (
            "dsl_version_id_digest",
            "parent_dsl_spec_root",
            "canonical_ast_schema_root",
            "canonical_cbor_profile_root",
            "identifier_registry_root",
            "operator_semantics_root",
            "equivalence_mode_id",
            "max_ast_depth",
            "max_ast_node_count",
            "max_top_level_clauses",
            "max_distinct_bit_slots",
            "max_aggregate_leaves",
            "max_scope_clauses",
            "max_composition_depth",
            "max_fitted_parameters",
            "max_entity_set_size",
            "canonical_program_budget",
            "raw_operator_application_cap",
            "shrink_step_id_digest",
        ),
        hash_domain="HEGEL/DSL_SPEC/V1",
    ),
    _schema(
        "SplitContractV1",
        0x3004,
        "hegel-split-contract/1",
        (
            "split_contract_version_id_digest",
            "split_algorithm_spec_root",
            "hkdf_profile_id_digest",
            "rank_hmac_profile_id_digest",
            "exhaustive_partition_required",
            "odd_stratum_quota_table",
            "sink_stratum_quota_table",
            "assignment_ordering_rule_id",
            "fallback_split_policy_id",
            "hidden_artifact_scope_root",
        ),
        hash_domain="HEGEL/SPLIT_CONTRACT/V1",
    ),
    _schema(
        "TargetBundleV1",
        0x3005,
        "hegel-target-bundle/1",
        (
            "outside_target_spec_root",
            "outside_target_universe_root",
            "outside_target_truth_root",
            "null_control_spec_root",
            "null_control_universe_root",
            "null_control_truth_root",
            "fallback_registry_root",
            "null_control_required_witness_ast_hash_or_null",
            "null_control_claim_level_id",
        ),
        hash_domain="HEGEL/TARGET_BUNDLE/V1",
    ),
    _schema(
        "ApprovalEvidenceBundleV1",
        0x3006,
        "hegel-approval-evidence-bundle/1",
        (
            "amendment_document_root",
            "approving_actor_id_digest",
            "approval_statement_id_digest",
            "parent_normative_decision_root",
            "approval_method_id",
            "approval_recorded_at_unix_seconds",
        ),
        hash_domain="HEGEL/APPROVAL_EVIDENCE_BUNDLE/V1",
    ),
    _schema(
        "ReplacementPolicyV1",
        0x3007,
        "hegel-replacement-policy/1",
        (
            "key_rotation_threshold",
            "key_revocation_threshold",
            "custodian_replacement_requires_new_seed_version",
            "actor_key_reuse_across_purposes_allowed",
            "secret_material_export_allowed",
        ),
        hash_domain="HEGEL/REPLACEMENT_POLICY/V1",
    ),
    _schema(
        "SplitSpecFreezeV1",
        0x3008,
        "hegel-split-spec-freeze/1",
        (
            "split_contract_root",
            "target_bundle_root",
            "child_freeze_root",
            "amendment_document_root",
            "seed_state_id",
            "frozen_at_unix_seconds",
            "repository_commit_id",
        ),
        hash_domain="HEGEL/SPLIT_SPEC_FREEZE/V1",
    ),
    _schema(
        "TombstonePolicyV1",
        0x3009,
        "hegel-tombstone-policy/1",
        (
            "registry_namespace_id_digest",
            "id_reuse_allowed",
            "removed_source_name_error_id_digest",
            "removed_numeric_id_error_id_digest",
            "unknown_numeric_id_error_id_digest",
        ),
        hash_domain="HEGEL/TOMBSTONE_POLICY/V1",
    ),
    _schema(
        "CrossDslHashPolicyV1",
        0x300A,
        "hegel-cross-dsl-hash-policy/1",
        (
            "ast_hash_domain_id_digest",
            "surviving_ast_bytes_stable",
            "surviving_ast_hash_stable",
            "semantic_identity_domain_id_digest",
            "required_binding_root_role_ids",
            "cross_version_archive_reuse_allowed",
            "cross_version_receipt_reuse_allowed",
            "cross_version_certificate_reuse_allowed",
        ),
        hash_domain="HEGEL/CROSS_DSL_HASH_POLICY/V1",
    ),
    _schema(
        "FallbackRegistryV1",
        0x300B,
        "hegel-fallback-registry/1",
        (
            "fallback_entries",
            "selection_rule_id_digest",
            "requires_new_target_version",
            "requires_new_split_first_instantiation",
        ),
        hash_domain="HEGEL/FALLBACK_REGISTRY/V1",
    ),
    _schema(
        "ImplementationBindingV1",
        0x300C,
        "hegel-implementation-binding/1",
        (
            "implementation_id",
            "source_root",
            "binary_digest",
            "execution_environment_spec_root",
            "compiler_or_interpreter_id_digest",
            "compiler_or_interpreter_version_digest",
            "dependency_lock_root",
            "build_profile_id_digest",
            "entrypoint_id_digest",
            "golden_vector_root",
            "repository_commit_id",
        ),
        hash_domain="HEGEL/IMPLEMENTATION_BINDING/V1",
    ),
    _schema(
        "TraversalContractV1",
        0x300D,
        "hegel-traversal-contract/1",
        (
            "bucket_key_field_ids",
            "canonical_sort_key_field_ids",
            "commutative_child_ordering_rule_id_digest",
            "maximum_canonical_programs",
            "maximum_raw_operator_applications",
            "frontier_exhaustion_definition_id_digest",
        ),
        hash_domain="HEGEL/TRAVERSAL_CONTRACT/V1",
    ),
    _schema(
        "BucketAccountingContractV1",
        0x300E,
        "hegel-bucket-accounting-contract/1",
        (
            "bucket_key_field_ids",
            "required_counter_field_ids",
            "bucket_ordering_rule_id_digest",
            "zero_count_bucket_emission_required",
            "accounting_sum_invariants",
        ),
        hash_domain="HEGEL/BUCKET_ACCOUNTING_CONTRACT/V1",
    ),
    _schema(
        "ProgramArchiveContractV1",
        0x300F,
        "hegel-program-archive-contract/1",
        (
            "program_record_schema_tag",
            "program_ordering_rule_id_digest",
            "records_per_chunk",
            "chunk_blob_codec_id",
            "chunk_blob_framing_rule_id_digest",
            "rfc6962_profile_id_digest",
            "target_independent",
        ),
        hash_domain="HEGEL/PROGRAM_ARCHIVE_CONTRACT/V1",
    ),
    _schema(
        "OutputArchiveContractV1",
        0x3010,
        "hegel-output-archive-contract/1",
        (
            "output_record_schema_tag",
            "output_ordering_rule_id_digest",
            "records_per_chunk",
            "chunk_blob_codec_id",
            "chunk_blob_framing_rule_id_digest",
            "undefined_bitmap_profile_id_digest",
            "role_specific",
        ),
        hash_domain="HEGEL/OUTPUT_ARCHIVE_CONTRACT/V1",
    ),
    _schema(
        "StateMachineContractV1",
        0x3011,
        "hegel-m3-state-machine-contract/1",
        (
            "m3_state_registry_root",
            "m3_phase_registry_root",
            "m3_transition_reason_registry_root",
            "legal_transition_table",
            "terminal_state_ids",
            "reopen_allowed",
        ),
        hash_domain="HEGEL/M3_STATE_MACHINE_CONTRACT/V1",
    ),
    _schema(
        "RowTransformSpecV1",
        0x3012,
        "hegel-row-transform-spec/1",
        (
            "source_diagnostic_profile_id",
            "source_namespace_id",
            "target_formal_profile_id",
            "target_object_tag",
            "transform_rule_id_digest",
            "ordering_rule_id_digest",
            "expected_row_count_or_null",
        ),
        hash_domain="HEGEL/ROW_TRANSFORM_SPEC/V1",
    ),
    _schema(
        "InputSignatureSpecV1",
        0x3013,
        "hegel-input-signature-spec/1",
        (
            "input_signature_id",
            "input_object_tag",
            "field_sort_ids",
            "static_role_metadata",
            "canonical_ordering_rule_id_digest",
        ),
        hash_domain="HEGEL/INPUT_SIGNATURE_SPEC/V1",
    ),
    _schema(
        "TargetSpecFormalV1",
        0x3014,
        "hegel-target-spec-formal/1",
        (
            "role_id",
            "target_machine_id_digest",
            "input_signature_spec_root",
            "output_sort_id",
            "target_rule_id_digest",
            "universe_row_count",
            "target_output_cardinality",
            "required_witness_ast_hash_or_null",
            "claim_level_id",
        ),
        hash_domain="HEGEL/TARGET_SPEC_FORMAL/V1",
    ),
    _schema(
        "SplitAlgorithmSpecV1",
        0x3015,
        "hegel-split-algorithm-spec/1",
        (
            "os_csprng_profile_id_digest",
            "hkdf_profile_id_digest",
            "rank_hmac_profile_id_digest",
            "rank_tie_break_rule_id_digest",
            "exhaustive_partition_required",
            "assignment_row_schema_tag",
        ),
        hash_domain="HEGEL/SPLIT_ALGORITHM_SPEC/V1",
    ),
    _schema(
        "ExecutionEnvironmentSpecV1",
        0x3016,
        "hegel-execution-environment-spec/1",
        (
            "os_id_digest",
            "architecture_id_digest",
            "runtime_id_digest",
            "runtime_version_id_digest",
            "dependency_lock_root",
            "locale_id_digest",
            "timezone_id_digest",
            "container_or_host_profile_id_digest",
            "oci_manifest_digest_or_null",
        ),
        hash_domain="HEGEL/EXECUTION_ENVIRONMENT_SPEC/V1",
    ),
    _schema(
        "NormativeDocumentBundleV1",
        0x3018,
        "hegel-normative-document-bundle/1",
        ("bundle_id_digest", "document_entries", "repository_commit_id"),
        hash_domain="HEGEL/NORMATIVE_DOCUMENT_BUNDLE/V1",
    ),
    _schema(
        "CanonicalAstProfileSpecV1",
        0x3019,
        "hegel-canonical-ast-profile/1",
        (
            "profile_id_digest",
            "governing_normative_document_root",
            "section_selector_id_digest",
            "section_blob_sha256",
            "section_byte_length",
            "repository_commit_id",
        ),
        hash_domain="HEGEL/CANONICAL_AST_PROFILE/V1",
    ),
    _schema(
        "CanonicalCborProfileSpecV1",
        0x301A,
        "hegel-canonical-cbor-profile/1",
        (
            "profile_id_digest",
            "governing_normative_document_root",
            "section_selector_id_digest",
            "section_blob_sha256",
            "section_byte_length",
            "repository_commit_id",
        ),
        hash_domain="HEGEL/CANONICAL_CBOR_PROFILE/V1",
    ),
    _schema(
        "Phase2BContractSpecV1",
        0x301B,
        "hegel-phase2b-contract/1",
        (
            "contract_id_digest",
            "governing_normative_document_root",
            "section_selector_id_digest",
            "section_blob_sha256",
            "section_byte_length",
            "repository_commit_id",
        ),
        hash_domain="HEGEL/PHASE2B_CONTRACT/V1",
    ),
    _schema(
        "MdlCodeTableSpecV1",
        0x301C,
        "hegel-mdl-code-table/1",
        (
            "table_id_digest",
            "governing_normative_document_root",
            "section_selector_id_digest",
            "section_blob_sha256",
            "section_byte_length",
            "repository_commit_id",
        ),
        hash_domain="HEGEL/MDL_CODE_TABLE/V1",
    ),
    _schema(
        "StaticRoleMetadataV1",
        0x301D,
        "hegel-static-role-metadata/1",
        (
            "input_signature_id",
            "role_ids",
            "quantity_ids",
            "scope_ids",
            "signed_orientations",
            "metadata_rule_id_digest",
        ),
        hash_domain="HEGEL/STATIC_ROLE_METADATA/V1",
    ),
    _schema(
        "HiddenArtifactScopeV1",
        0x301E,
        "hegel-hidden-artifact-scope/1",
        (
            "policy_id_digest",
            "governing_normative_document_root",
            "section_selector_id_digest",
            "section_blob_sha256",
            "section_byte_length",
            "repository_commit_id",
        ),
        hash_domain="HEGEL/HIDDEN_ARTIFACT_SCOPE/V1",
    ),
    _schema(
        "NormativeApprovalManifestV1",
        0x3101,
        "hegel-normative-approval-manifest/1",
        (
            "amendment_document_root",
            "parent_freeze_root",
            "child_freeze_root",
            "child_dsl_spec_root_or_null",
            "approval_status_id",
            "approval_method_id",
            "approval_evidence_root",
            "approving_actor_id_digest",
            "recorded_at_unix_seconds",
            "repository_commit_id",
        ),
        hash_domain="HEGEL/NORMATIVE_APPROVAL_MANIFEST/V1",
    ),
    _schema(
        "DslRoleBindingManifestV1",
        0x3102,
        "hegel-dsl-role-binding-manifest/1",
        (
            "role_id",
            "child_dsl_spec_root",
            "child_freeze_root",
            "operator_semantics_root",
            "identifier_registry_root",
            "canonical_ast_schema_root",
            "canonical_cbor_profile_root",
            "semantic_spec_diagnostic_id_digest",
            "semantic_spec_formal_root",
            "universe_diagnostic_id_digest",
            "truth_diagnostic_id_digest",
            "formal_universe_root",
            "formal_truth_root",
            "split_binding_manifest_root",
            "custodian_binding_manifest_root",
            "seed_continuity_manifest_root",
            "parent_binding_manifest_root_or_null",
            "legacy_parent_payload_source_id_digest_or_null",
            "parent_manifest_absence_attestation_root_or_null",
            "fallback_registry_root_or_null",
            "created_at_unix_seconds",
            "repository_commit_id",
        ),
        hash_domain="HEGEL/DSL_ROLE_BINDING_MANIFEST/V1",
    ),
    _schema(
        "SplitSeedCommitmentManifestV1",
        0x3103,
        "hegel-split-seed-commitment-manifest/1",
        (
            "split_contract_root",
            "target_bundle_root",
            "split_seed_commitment_digest",
            "seed_length_bytes",
            "rng_profile_id_digest",
            "kdf_profile_id_digest",
            "commitment_profile_id_digest",
            "custodian_key_id",
            "created_at_unix_seconds",
            "repository_commit_id",
        ),
        hash_domain="HEGEL/SPLIT_SEED_COMMITMENT_MANIFEST/V1",
    ),
    _schema(
        "SplitBindingManifestV1",
        0x3104,
        "hegel-split-binding-manifest/1",
        (
            "split_contract_root",
            "split_seed_commitment_manifest_root",
            "seed_continuity_manifest_root",
            "split_algorithm_id_digest",
            "outside_target_discovery_root",
            "outside_target_validation_root",
            "outside_target_sealed_root",
            "null_control_discovery_root",
            "null_control_validation_root",
            "null_control_sealed_root",
            "hidden_access_ledger_genesis_root",
            "hidden_access_ledger_head_root",
            "split_instantiation_status_id",
            "created_at_unix_seconds",
            "repository_commit_id",
        ),
        hash_domain="HEGEL/SPLIT_BINDING_MANIFEST/V1",
    ),
    _schema(
        "CustodianBindingManifestV1",
        0x3105,
        "hegel-custodian-binding-manifest/1",
        (
            "custodian_key_id",
            "custodian_public_key_32_bytes",
            "custodian_key_epoch",
            "responsibility_bitmask",
            "split_seed_commitment_manifest_root",
            "hidden_access_ledger_genesis_root",
            "seed_continuity_manifest_root",
            "valid_from_unix_seconds",
            "valid_until_unix_seconds_or_null",
            "replacement_policy_root",
            "repository_commit_id",
        ),
        hash_domain="HEGEL/CUSTODIAN_BINDING_MANIFEST/V1",
    ),
    _schema(
        "SeedContinuityManifestV1",
        0x3106,
        "hegel-seed-continuity-manifest/1",
        (
            "continuity_status_id",
            "split_spec_freeze_root",
            "parent_seed_commitment_manifest_root_or_null",
            "current_seed_commitment_manifest_root",
            "parent_manifest_absence_attestation_root",
            "hidden_access_ledger_genesis_root",
            "custodian_binding_core_root",
            "instantiated_at_unix_seconds",
            "repository_commit_id",
        ),
        hash_domain="HEGEL/SEED_CONTINUITY_MANIFEST/V1",
    ),
    _schema(
        "ParentManifestAbsenceAttestationV1",
        0x3107,
        "hegel-parent-manifest-absence-attestation/1",
        (
            "parent_dsl_version_digest",
            "parent_freeze_version_digest",
            "parent_repository_commit_id",
            "audited_source_tree_root",
            "audited_path_set_root",
            "legacy_parent_payload_source_id_digest",
            "absence_reason_bitmask",
            "auditor_key_id",
            "audited_at_unix_seconds",
        ),
        hash_domain="HEGEL/PARENT_MANIFEST_ABSENCE_ATTESTATION/V1",
    ),
    _schema(
        "HiddenAccessLedgerRecordV1",
        0x3108,
        "hegel-hidden-access-ledger-record/1",
        (
            "ledger_id",
            "sequence_number",
            "previous_record_root_or_null",
            "event_type_id",
            "actor_key_id",
            "subject_manifest_root",
            "revealed_artifact_root_or_null",
            "authorization_root_or_null",
            "created_at_unix_seconds",
            "repository_commit_id",
        ),
        hash_domain="HEGEL/HIDDEN_ACCESS_LEDGER_RECORD/V1",
    ),
    _schema(
        "DslShrinkTransitionFormalV1",
        0x3109,
        "hegel-dsl-shrink-transition-formal/1",
        (
            "parent_dsl_spec_root",
            "child_dsl_spec_root",
            "parent_freeze_root",
            "child_freeze_root",
            "parent_execution_evidence_root",
            "parent_status_id",
            "shrink_step_id_digest",
            "removed_registry_entry_root",
            "surviving_registry_entry_root",
            "tombstone_policy_root",
            "cross_dsl_hash_policy_root",
            "approval_manifest_root",
            "outside_target_binding_manifest_root",
            "null_control_binding_manifest_root",
            "split_binding_manifest_root",
            "custodian_binding_manifest_root",
            "seed_continuity_manifest_root",
            "shrink1_subset_replay_root",
            "child_initial_state_id",
            "created_at_unix_seconds",
            "repository_commit_id",
        ),
        hash_domain="HEGEL/DSL_SHRINK_TRANSITION/V1",
    ),
    _schema(
        "M3ExecutionManifestV1",
        0x310A,
        "hegel-m3-execution-manifest/1",
        (
            "run_id",
            "child_dsl_spec_root",
            "child_freeze_root",
            "approval_manifest_root",
            "shrink_transition_root",
            "operator_semantics_root",
            "identifier_registry_root",
            "canonical_ast_schema_root",
            "canonical_cbor_profile_root",
            "diagnostic_formal_bridge_root",
            "outside_target_binding_manifest_root",
            "null_control_binding_manifest_root",
            "split_binding_manifest_root",
            "custodian_binding_manifest_root",
            "seed_continuity_manifest_root",
            "hidden_access_ledger_genesis_root",
            "hidden_access_ledger_head_root",
            "outside_target_universe_root",
            "outside_target_truth_root",
            "null_control_universe_root",
            "null_control_truth_root",
            "outside_discovery_split_root",
            "outside_validation_split_root",
            "outside_sealed_split_root",
            "null_discovery_split_root",
            "null_validation_split_root",
            "null_sealed_split_root",
            "canonical_program_budget",
            "raw_operator_application_cap",
            "records_per_chunk",
            "equivalence_mode_id",
            "python_implementation_binding_root",
            "rust_implementation_binding_root",
            "traversal_contract_root",
            "bucket_accounting_contract_root",
            "program_archive_contract_root",
            "output_archive_contract_root",
            "state_machine_contract_root",
            "created_at_unix_seconds",
            "repository_commit_id",
        ),
        hash_domain="HEGEL/M3_EXECUTION_MANIFEST/V1",
    ),
    _schema(
        "CustodianBindingCoreV1",
        0x310B,
        "hegel-custodian-binding-core/1",
        (
            "custodian_key_id",
            "custodian_public_key_32_bytes",
            "custodian_key_epoch",
            "responsibility_bitmask",
            "valid_from_unix_seconds",
            "valid_until_unix_seconds_or_null",
            "replacement_policy_root",
            "repository_commit_id",
        ),
        hash_domain="HEGEL/CUSTODIAN_BINDING_CORE/V1",
    ),
    _schema(
        "ActorKeyManifestV1",
        0x310C,
        "hegel-actor-key-manifest/1",
        (
            "purpose_id",
            "key_id",
            "public_key_32_bytes",
            "key_epoch",
            "valid_from_unix_seconds",
            "valid_until_unix_seconds_or_null",
            "repository_commit_id",
        ),
        hash_domain="HEGEL/ACTOR_KEY_MANIFEST/V1",
    ),
    _schema(
        "AttestationBundleV1",
        0x310D,
        "hegel-attestation-bundle/1",
        ("attestations",),
        hash_domain="HEGEL/ATTESTATION_BUNDLE/V1",
    ),
    _schema(
        "BridgeReplayStatementV1",
        0x310E,
        "hegel-bridge-replay-statement/1",
        (
            "run_id",
            "diagnostic_formal_bridge_root",
            "m3_execution_candidate_root",
            "child_dsl_spec_root",
            "child_freeze_root",
            "actor_trust_genesis_root",
            "opaque_id_registry_snapshot_root",
        ),
        hash_domain="HEGEL/BRIDGE_REPLAY_STATEMENT/V1",
    ),
    _schema(
        "M3ExecutionCandidateV1",
        0x310F,
        "hegel-m3-execution-candidate/1",
        (
            "run_id",
            "child_dsl_spec_root",
            "child_freeze_root",
            "approval_manifest_root",
            "shrink_transition_root",
            "operator_semantics_root",
            "identifier_registry_root",
            "canonical_ast_schema_root",
            "canonical_cbor_profile_root",
            "diagnostic_formal_bridge_root",
            "outside_target_binding_manifest_root",
            "null_control_binding_manifest_root",
            "split_binding_manifest_root",
            "custodian_binding_manifest_root",
            "seed_continuity_manifest_root",
            "custodian_attestation_bundle_root",
            "parent_absence_attestation_root",
            "hidden_access_ledger_genesis_root",
            "hidden_access_ledger_head_root",
            "opaque_id_registry_snapshot_root",
            "actor_trust_genesis_root",
            "outside_target_universe_root",
            "outside_target_truth_root",
            "null_control_universe_root",
            "null_control_truth_root",
            "outside_discovery_split_root",
            "outside_validation_split_root",
            "outside_sealed_split_root",
            "null_discovery_split_root",
            "null_validation_split_root",
            "null_sealed_split_root",
            "canonical_program_budget",
            "raw_operator_application_cap",
            "records_per_chunk",
            "equivalence_mode_id",
            "python_implementation_binding_root",
            "rust_implementation_binding_root",
            "traversal_contract_root",
            "bucket_accounting_contract_root",
            "program_archive_contract_root",
            "output_archive_contract_root",
            "state_machine_contract_root",
            "created_at_unix_seconds",
            "repository_commit_id",
        ),
        hash_domain="HEGEL/M3_EXECUTION_CANDIDATE/V1",
    ),
    _schema(
        "M3ExecutionManifestV2",
        0x3110,
        "hegel-m3-execution-manifest/2",
        (
            "run_id",
            "m3_execution_candidate_root",
            "bridge_replay_statement_root",
            "bridge_attestation_bundle_root",
            "actor_trust_genesis_root",
            "opaque_id_registry_snapshot_root",
            "created_at_unix_seconds",
            "repository_commit_id",
        ),
        hash_domain="HEGEL/M3_EXECUTION_MANIFEST/V2",
    ),
    _schema(
        "ActorTrustGenesisV1",
        0x3111,
        "hegel-actor-trust-genesis/1",
        (
            "trust_genesis_id_16_bytes",
            "purpose_key_entries",
            "purpose_key_policy_root",
            "created_at_unix_seconds",
            "repository_commit_id",
        ),
        hash_domain="HEGEL/ACTOR_TRUST_GENESIS/V1",
    ),
    _schema(
        "OpaqueIdRegistrySnapshotV1",
        0x3112,
        "hegel-opaque-id-registry-snapshot/1",
        (
            "previous_snapshot_root_or_null",
            "registry_tree_root",
            "record_count",
            "added_record_root",
            "repository_commit_id",
        ),
        hash_domain="HEGEL/OPAQUE_ID_REGISTRY_SNAPSHOT/V1",
    ),
    _schema(
        "ParentAbsenceAuditBundleV1",
        0x3113,
        "hegel-parent-absence-audit-bundle/1",
        (
            "audited_parent_repository_commit_id",
            "audited_path_tree_root",
            "audited_history_tree_root",
            "legacy_source_tree_root",
            "audited_path_count",
            "audited_history_row_count",
            "legacy_source_count",
        ),
        hash_domain="HEGEL/PARENT_ABSENCE_AUDIT_BUNDLE/V1",
    ),
    _schema(
        "ParentManifestAbsenceAttestationV2",
        0x3114,
        "hegel-parent-manifest-absence-attestation/2",
        (
            "parent_dsl_version_digest",
            "parent_freeze_version_digest",
            "parent_repository_commit_id",
            "audit_bundle_root",
            "absence_reason_bitmask",
            "auditor_key_id",
            "audited_at_unix_seconds",
        ),
        hash_domain="HEGEL/PARENT_MANIFEST_ABSENCE_ATTESTATION/V2",
    ),
    _schema(
        "OpaqueIdRegistrationIntentV1",
        0x3115,
        "hegel-opaque-id-registration-intent/1",
        (
            "opaque_id_kind_id",
            "opaque_id_16_bytes",
            "registration_context_root",
            "created_at_unix_seconds",
            "repository_commit_id",
        ),
        hash_domain="HEGEL/OPAQUE_ID_REGISTRATION_INTENT/V1",
    ),
    _schema(
        "SignedManifestEnvelopeV1",
        0x31FF,
        "hegel-signed-manifest-envelope/1",
        (
            "enclosed_object_tag",
            "enclosed_manifest_root",
            "created_at_unix_seconds",
            "signer_key_epoch",
            "signatures",
        ),
        hash_domain="HEGEL/SIGNED_MANIFEST_ENVELOPE/V1",
    ),
    _schema(
        "OddInputV1",
        0x3401,
        "hegel-odd-input/1",
        ("set_size", "bits"),
        hash_domain=CANONICAL_INPUT_DOMAIN,
    ),
    _schema(
        "SinkInputV1",
        0x3402,
        "hegel-sink-input/1",
        ("a", "b", "c", "d"),
        hash_domain=CANONICAL_INPUT_DOMAIN,
    ),
    _schema(
        "BoundedUniverseRowV1",
        0x3201,
        "hegel-bounded-universe-row/1",
        ("universe_index", "input_signature_id", "canonical_input_object"),
        rfc6962_records=True,
        ordering_fields=("universe_index",),
    ),
    _schema(
        "TargetTruthRowV1",
        0x3202,
        "hegel-target-truth-row/1",
        ("universe_index", "canonical_input_hash", "target_output"),
        rfc6962_records=True,
        ordering_fields=("universe_index",),
    ),
    _schema(
        "SplitAssignmentRowV1",
        0x3203,
        "hegel-split-assignment-row/1",
        (
            "role_id",
            "universe_index",
            "canonical_input_hash",
            "stratum_id",
            "partition_id",
            "rank_digest",
        ),
        rfc6962_records=True,
        ordering_fields=("role_id", "partition_id", "universe_index"),
    ),
    _schema(
        "IdentifierRegistryEntryV1",
        0x3204,
        "hegel-identifier-registry-entry/1",
        (
            "registry_kind_id",
            "numeric_id",
            "entry_state_id",
            "canonical_name_digest",
            "semantics_digest_or_null",
            "introduced_dsl_version_digest",
            "removed_dsl_version_digest_or_null",
        ),
        rfc6962_records=True,
        ordering_fields=("registry_kind_id", "numeric_id"),
    ),
    _schema(
        "OperatorSemanticsEntryV1",
        0x3205,
        "hegel-operator-semantics-entry/1",
        (
            "operator_class_id",
            "operator_id",
            "admission_state_id",
            "input_sort_ids",
            "output_sort_id",
            "undefined_semantics_id",
            "normalization_rule_root_or_null",
            "executable_semantics_root",
        ),
        rfc6962_records=True,
        ordering_fields=("operator_class_id", "operator_id"),
    ),
    _schema(
        "DiagnosticFormalBridgeRecordV1",
        0x3206,
        "hegel-diagnostic-formal-bridge-record/1",
        (
            "artifact_role_id",
            "diagnostic_namespace_id",
            "diagnostic_digest",
            "formal_object_kind_id",
            "formal_digest_or_root",
            "row_count_or_null",
            "diagnostic_profile_id_digest",
            "formal_profile_id_digest",
            "row_transform_spec_root",
            "source_artifact_digest",
            "repository_commit_id",
        ),
        rfc6962_records=True,
        ordering_fields=(
            "artifact_role_id",
            "diagnostic_namespace_id",
            "diagnostic_digest",
        ),
    ),
    _schema(
        "CanonicalProgramRecordV2",
        0x3207,
        "hegel-canonical-program-record/2",
        (
            "program_index",
            "canonical_ast_cbor_bytes",
            "canonical_ast_hash",
            "output_sort_id",
            "ast_depth",
            "ast_node_count",
            "distinct_bit_slot_count",
            "program_mdl_length_q32",
            "child_dsl_spec_root",
            "operator_semantics_root",
            "identifier_registry_root",
        ),
        rfc6962_records=True,
        ordering_fields=("program_index",),
    ),
    _schema(
        "ProgramOutputRecordV2",
        0x3208,
        "hegel-program-output-record/2",
        (
            "role_id",
            "program_index",
            "canonical_ast_hash",
            "bounded_universe_root",
            "operator_semantics_root",
            "output_sort_id",
            "row_count",
            "output_blob_hash",
            "undefined_bitmap_hash",
        ),
        rfc6962_records=True,
        ordering_fields=("program_index",),
    ),
    _schema(
        "ProgramChunkManifestV2",
        0x3209,
        "hegel-program-chunk-manifest/2",
        (
            "chunk_index",
            "first_program_index",
            "last_program_index",
            "record_count",
            "canonical_program_record_subtree_root",
            "compressed_program_blob_hash",
            "uncompressed_program_byte_length",
        ),
        rfc6962_records=True,
        ordering_fields=("chunk_index",),
    ),
    _schema(
        "RoleOutputChunkManifestV2",
        0x320A,
        "hegel-role-output-chunk-manifest/2",
        (
            "role_id",
            "chunk_index",
            "first_program_index",
            "last_program_index",
            "record_count",
            "output_record_subtree_root",
            "compressed_output_blob_hash",
            "uncompressed_output_byte_length",
        ),
        rfc6962_records=True,
        ordering_fields=("role_id", "chunk_index"),
    ),
    _schema(
        "MatchRecordV2",
        0x320B,
        "hegel-match-record/2",
        (
            "role_id",
            "canonical_ast_hash",
            "output_blob_hash",
            "target_truth_table_root",
        ),
        rfc6962_records=True,
        ordering_fields=("canonical_ast_hash", "output_blob_hash"),
    ),
    _schema(
        "BucketAccountingRecordV1",
        0x320C,
        "hegel-bucket-accounting-record/1",
        (
            "bucket_index",
            "output_sort_id",
            "ast_depth",
            "ast_node_count",
            "raw_operator_applications",
            "accepted_canonical_programs",
            "syntactic_duplicates",
            "type_rejections",
            "structural_limit_rejections",
            "rewrite_collapses",
            "first_program_index_or_null",
            "last_program_index_or_null",
        ),
        rfc6962_records=True,
        ordering_fields=("bucket_index",),
    ),
    _schema(
        "MismatchRecordV1",
        0x320D,
        "hegel-mismatch-record/1",
        (
            "mismatch_index",
            "mismatch_kind_id",
            "python_object_root_or_null",
            "rust_object_root_or_null",
            "affected_program_index_or_null",
            "diagnostic_detail_digest",
        ),
        rfc6962_records=True,
        ordering_fields=("mismatch_index",),
    ),
    _schema(
        "PartialDiagnosticBundleV1",
        0x320E,
        "hegel-partial-diagnostic-bundle/1",
        (
            "run_id",
            "implementation_id",
            "terminal_failure_code_id_digest",
            "completed_bucket_count",
            "partial_bucket_accounting_root_or_null",
            "partial_log_digest",
            "authoritative_claim_allowed",
        ),
        hash_domain="HEGEL/PARTIAL_DIAGNOSTIC_BUNDLE/V1",
    ),
    _schema(
        "AuditedPathBlobRecordV1",
        0x3210,
        "hegel-audited-path-blob-record/1",
        (
            "repository_path_alias_id_digest",
            "raw_repository_path_utf8_bytes",
            "git_object_algorithm_id",
            "git_blob_digest",
            "file_mode",
            "byte_length",
        ),
        rfc6962_records=True,
        ordering_fields=(
            "raw_repository_path_utf8_bytes",
            "repository_path_alias_id_digest",
            "git_blob_digest",
        ),
    ),
    _schema(
        "AuditedHistoryRowV1",
        0x3211,
        "hegel-audited-history-row/1",
        (
            "commit_generation",
            "repository_commit_id",
            "ordered_parent_commit_ids",
            "touched_path_set_root",
        ),
        rfc6962_records=True,
        ordering_fields=("commit_generation", "repository_commit_id"),
    ),
    _schema(
        "LegacyParentSourceRowV1",
        0x3212,
        "hegel-legacy-parent-source-row/1",
        (
            "target_role_id",
            "legacy_parent_payload_source_id_digest",
            "diagnostic_namespace_id",
            "diagnostic_digest",
            "source_repository_commit_id",
        ),
        rfc6962_records=True,
        ordering_fields=("target_role_id",),
    ),
    _schema(
        "RepositoryPathAliasRecordV1",
        0x3213,
        "hegel-repository-path-alias-record/1",
        (
            "path_alias_id_digest",
            "raw_repository_path_utf8_bytes",
            "repository_commit_id",
        ),
        rfc6962_records=True,
        ordering_fields=("path_alias_id_digest",),
    ),
    _schema(
        "SourceFileRecordV1",
        0x3215,
        "hegel-source-file-record/1",
        (
            "path_alias_id_digest",
            "raw_path_bytes",
            "git_blob_algorithm_id",
            "git_blob_digest",
            "file_mode",
            "byte_length",
        ),
        rfc6962_records=True,
        ordering_fields=("raw_path_bytes",),
    ),
    _schema(
        "DependencyLockRecordV1",
        0x3216,
        "hegel-dependency-lock-record/1",
        (
            "ecosystem_id",
            "package_name_id_digest",
            "version_id_digest",
            "source_id_digest",
            "lock_entry_digest",
        ),
        rfc6962_records=True,
        ordering_fields=("ecosystem_id", "package_name_id_digest", "version_id_digest"),
    ),
    _schema(
        "LegalTransitionRowV1",
        0x3217,
        "hegel-legal-transition-row/1",
        (
            "from_state_id",
            "from_phase_id",
            "to_state_id",
            "to_phase_id",
            "allowed_reason_ids",
        ),
        rfc6962_records=True,
        ordering_fields=("from_state_id", "from_phase_id", "to_state_id", "to_phase_id"),
    ),
    _schema(
        "OpaqueIdRegistryRecordV1",
        0x3218,
        "hegel-opaque-id-registry-record/1",
        (
            "registry_sequence_number",
            "opaque_id_kind_id",
            "opaque_id_16_bytes",
            "first_seen_object_root",
            "first_seen_repository_commit_id",
            "created_at_unix_seconds",
        ),
        rfc6962_records=True,
        ordering_fields=("registry_sequence_number",),
    ),
    _schema(
        "M3RunGenesisV1",
        0x3300,
        "hegel-m3-run-genesis/1",
        (
            "run_id",
            "execution_manifest_root",
            "initial_state_id",
            "canonical_program_archive_root_or_null",
            "program_chunk_manifest_root_or_null",
            "bucket_accounting_root_or_null",
            "outside_program_output_archive_root_or_null",
            "outside_output_chunk_manifest_root_or_null",
            "outside_match_set_root_or_null",
            "outside_role_evaluation_receipt_root_or_null",
            "null_program_output_archive_root_or_null",
            "null_output_chunk_manifest_root_or_null",
            "null_match_set_root_or_null",
            "null_role_evaluation_receipt_root_or_null",
            "python_enumeration_receipt_root_or_null",
            "rust_enumeration_receipt_root_or_null",
            "dual_replay_agreement_root_or_null",
            "final_state_record_root_or_null",
            "created_at_unix_seconds",
            "repository_commit_id",
        ),
        hash_domain="HEGEL/M3_RUN_GENESIS/V1",
    ),
    _schema(
        "M3RunStateRecordV1",
        0x3301,
        "hegel-m3-run-state-record/1",
        (
            "run_id",
            "transition_index",
            "previous_state_record_root_or_null",
            "from_state_id",
            "from_phase_id",
            "to_state_id",
            "to_phase_id",
            "transition_reason_id",
            "execution_manifest_root",
            "triggering_receipt_root_or_null",
            "recorded_at_unix_seconds",
        ),
        hash_domain="HEGEL/M3_RUN_STATE_RECORD/V1",
    ),
    _schema(
        "M3ImplementationEnumerationReceiptV1",
        0x3302,
        "hegel-m3-implementation-enumeration-receipt/1",
        (
            "implementation_id",
            "run_id",
            "execution_manifest_root",
            "implementation_source_root",
            "implementation_binary_digest",
            "environment_image_digest",
            "child_dsl_spec_root",
            "operator_semantics_root",
            "identifier_registry_root",
            "canonical_ast_schema_root",
            "canonical_cbor_profile_root",
            "closure_status_id",
            "raw_operator_application_count",
            "canonical_program_count",
            "closure_cardinality_or_null",
            "frontier_exhausted",
            "all_type_buckets_closed",
            "raw_expansion_limit_hit",
            "wall_clock_abort_hit",
            "canonical_program_archive_root_or_null",
            "program_chunk_manifest_root_or_null",
            "bucket_accounting_root_or_null",
            "first_out_of_budget_program_hash_or_null",
            "partial_diagnostic_bundle_root_or_null",
            "started_at_unix_seconds",
            "finished_at_unix_seconds",
            "process_exit_code",
        ),
        hash_domain="HEGEL/M3_IMPLEMENTATION_ENUMERATION_RECEIPT/V1",
    ),
    _schema(
        "M3RoleEvaluationReceiptV1",
        0x3303,
        "hegel-m3-role-evaluation-receipt/1",
        (
            "implementation_id",
            "role_id",
            "run_id",
            "execution_manifest_root",
            "enumeration_receipt_root",
            "canonical_program_archive_root",
            "bounded_universe_root",
            "target_truth_table_root",
            "program_output_archive_root",
            "role_output_chunk_manifest_root",
            "match_set_count",
            "match_set_root",
            "undefined_program_count",
            "evaluation_complete",
            "started_at_unix_seconds",
            "finished_at_unix_seconds",
            "process_exit_code",
        ),
        hash_domain="HEGEL/M3_ROLE_EVALUATION_RECEIPT/V1",
    ),
    _schema(
        "M3DualReplayAgreementV1",
        0x3304,
        "hegel-m3-dual-replay-agreement/1",
        (
            "run_id",
            "execution_manifest_root",
            "python_enumeration_receipt_root",
            "rust_enumeration_receipt_root",
            "agreed_closure_status_id",
            "canonical_program_count_or_null",
            "closure_cardinality_or_null",
            "canonical_program_archive_root_or_null",
            "program_chunk_manifest_root_or_null",
            "bucket_accounting_root_or_null",
            "first_out_of_budget_program_hash_or_null",
            "role_agreement_entries",
            "enumeration_agreement",
            "role_agreement_status_id",
            "mismatch_record_root_or_null",
            "created_at_unix_seconds",
        ),
        hash_domain="HEGEL/M3_DUAL_REPLAY_AGREEMENT/V1",
    ),
    _schema(
        "M3RoleAgreementEntryV1",
        0x3305,
        "hegel-m3-role-agreement-entry/1",
        (
            "role_id",
            "python_role_receipt_root",
            "rust_role_receipt_root",
            "bounded_universe_root",
            "target_truth_table_root",
            "program_output_archive_root",
            "role_output_chunk_manifest_root",
            "match_set_count",
            "match_set_root",
            "agreement",
        ),
    ),
)


FORMAL_SCHEMA_REGISTRY: Final = MappingProxyType(
    {schema.name: schema for schema in _SCHEMAS}
)
_SCHEMA_BY_IDENTITY: Final = MappingProxyType(
    {(schema.tag, schema.schema_id): schema for schema in _SCHEMAS}
)


# These gaps prohibit authoritative M2.5 roots/gates.  They are intentionally
# data, not TODO comments, so callers can expose the exact fail-closed reason.
AUTHORITATIVE_BLOCKING_GAPS: Final = MappingProxyType(
    {
        "ExternalActorEvidence": (
            "authoritative custody, audit replay, append-only opaque-ID persistence, "
            "signatures, and first seed genesis are external"
        ),
    }
)


_BOOL_FIELDS: Final = frozenset(
    {
        "frontier_exhausted",
        "all_type_buckets_closed",
        "raw_expansion_limit_hit",
        "wall_clock_abort_hit",
        "evaluation_complete",
        "enumeration_agreement",
        "agreement",
        "exhaustive_partition_required",
        "custodian_replacement_requires_new_seed_version",
        "actor_key_reuse_across_purposes_allowed",
        "secret_material_export_allowed",
        "id_reuse_allowed",
        "surviving_ast_bytes_stable",
        "surviving_ast_hash_stable",
        "cross_version_archive_reuse_allowed",
        "cross_version_receipt_reuse_allowed",
        "cross_version_certificate_reuse_allowed",
        "requires_new_target_version",
        "requires_new_split_first_instantiation",
        "zero_count_bucket_emission_required",
        "target_independent",
        "role_specific",
        "reopen_allowed",
        "authoritative_claim_allowed",
    }
)
_ARRAY_FIELDS: Final = frozenset(
    {
        "input_sort_ids",
        "field_sort_ids",
        "signatures",
        "role_agreement_entries",
        "odd_stratum_quota_table",
        "sink_stratum_quota_table",
        "required_binding_root_role_ids",
        "fallback_entries",
        "bucket_key_field_ids",
        "canonical_sort_key_field_ids",
        "required_counter_field_ids",
        "accounting_sum_invariants",
        "legal_transition_table",
        "terminal_state_ids",
        "static_role_metadata",
        "attestations",
        "document_entries",
        "purpose_key_entries",
        "role_ids",
        "quantity_ids",
        "scope_ids",
        "signed_orientations",
        "ordered_parent_commit_ids",
        "allowed_reason_ids",
        "bits",
    }
)

_VARIABLE_BYTES_FIELDS: Final = frozenset(
    {
        "raw_git_blob_bytes",
        "canonical_ast_cbor_bytes",
        "raw_repository_path_utf8_bytes",
        "raw_path_bytes",
    }
)

_EXACT_ODD_QUOTAS: Final = (
    (1, 16, 6, 3, 7),
    (2, 16, 6, 3, 7),
    (3, 32, 13, 6, 13),
    (4, 32, 13, 6, 13),
    (5, 64, 26, 13, 25),
    (6, 64, 26, 13, 25),
    (7, 128, 51, 26, 51),
    (8, 128, 51, 26, 51),
)
_EXACT_SINK_QUOTAS: Final = (
    (9, 15, 7, 4, 4),
    (10, 18, 8, 4, 6),
    (11, 19, 9, 4, 6),
    (12, 18, 8, 4, 6),
    (13, 15, 7, 4, 4),
)

_FIELD_ENUM_REGISTRY: Final = MappingProxyType(
    {
        "input_signature_id": "InputSignatureId",
        "output_sort_id": "SortId",
        "registry_kind_id": "RegistryKindId",
        "entry_state_id": "RegistryEntryStateId",
        "operator_class_id": "OperatorClassId",
        "admission_state_id": "OperatorAdmissionStateId",
        "undefined_semantics_id": "UndefinedSemanticsId",
        "artifact_role_id": "ArtifactRoleId",
        "diagnostic_namespace_id": "DiagnosticNamespaceId",
        "source_namespace_id": "DiagnosticNamespaceId",
        "formal_object_kind_id": "FormalObjectKindId",
        "source_diagnostic_profile_id": "DiagnosticProfileId",
        "target_formal_profile_id": "FormalProfileId",
        "stratum_id": "StratumId",
        "partition_id": "PartitionId",
        "equivalence_mode_id": "EquivalenceModeId",
        "implementation_id": "ImplementationId",
        "parent_status_id": "ParentStatusId",
        "child_initial_state_id": "ChildInitialStateId",
        "transition_reason_id": "M3TransitionReasonId",
        "closure_status_id": "M3ClosureStatusId",
        "agreed_closure_status_id": "M3ClosureStatusId",
        "initial_state_id": "M3StateId",
        "from_state_id": "M3StateId",
        "to_state_id": "M3StateId",
        "from_phase_id": "M3RunningPhaseId",
        "to_phase_id": "M3RunningPhaseId",
        "role_agreement_status_id": "RoleAgreementStatusId",
        "purpose_id": "ActorPurposeId",
        "claim_level_id": "ClaimLevelId",
        "null_control_claim_level_id": "ClaimLevelId",
        "mismatch_kind_id": "MismatchKindId",
        "assignment_ordering_rule_id": "AssignmentOrderingRuleId",
        "fallback_split_policy_id": "FallbackSplitPolicyId",
        "opaque_id_kind_id": "OpaqueIdKindId",
        "target_role_id": "TargetRoleId",
        "ecosystem_id": "DependencyEcosystemId",
        "git_object_algorithm_id": "GitObjectAlgorithmId",
        "git_blob_algorithm_id": "GitObjectAlgorithmId",
        "seed_state_id": "SeedStateId",
        "role_id": "TargetRoleId",
        "approval_status_id": "ApprovalStatusId",
        "approval_method_id": "ApprovalMethodId",
        "split_instantiation_status_id": "SplitInstantiationStatusId",
        "continuity_status_id": "SeedContinuityStatusId",
        "event_type_id": "HiddenAccessEventTypeId",
    }
)


def git_sha1_commit_id(raw_digest: bytes) -> tuple[int, bytes]:
    """Build the frozen Git SHA-1 wire form ``[1, 20-byte digest]``."""

    if type(raw_digest) is not bytes or len(raw_digest) != 20:
        _fail("REJECT_GIT_COMMIT_ID", "Git SHA-1 digest must be exactly 20 bytes")
    return (1, raw_digest)


def _require_bytes(value: object, length: int, field: str) -> None:
    if type(value) is not bytes or len(value) != length:
        _fail("REJECT_M25_FIELD_TYPE", f"{field} must be exactly {length} bytes")


def _validate_repository_commit(value: object, field: str) -> None:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        _fail("REJECT_GIT_COMMIT_ID", f"{field} must be [1, 20-byte digest]")
    if type(value[0]) is not int or value[0] != 1:
        _fail("REJECT_GIT_COMMIT_ID", f"{field} algorithm ID must be 1 (SHA-1)")
    _require_bytes(value[1], 20, field)


def _repository_commit_digest(value: object, field: str) -> bytes:
    _validate_repository_commit(value, field)
    assert isinstance(value, (tuple, list)) and isinstance(value[1], bytes)
    return value[1]


def _validate_signature_records(value: object) -> None:
    if not isinstance(value, (tuple, list)):
        _fail("REJECT_M25_FIELD_TYPE", "signatures must be an array")
    for signature in value:
        if not isinstance(signature, (tuple, list)) or len(signature) != 2:
            _fail("REJECT_M25_FIELD_TYPE", "SignatureRecordV1 must have two fields")
        _require_bytes(signature[0], 16, "signature.key_id")
        _require_bytes(signature[1], 64, "signature.ed25519_signature")


def _validate_field(field: str, value: object) -> None:
    nullable = field.endswith("_or_null")
    if value is None:
        if nullable:
            return
        _fail("REJECT_M25_FIELD_NULL", f"{field} is non-nullable")
    base_field = field.removesuffix("_or_null")

    enum_name = _FIELD_ENUM_REGISTRY.get(base_field)
    if enum_name is not None:
        NUMERIC_ENUM_REGISTRIES[enum_name].validate(value, field=field)
        return
    if base_field in {"run_id", "ledger_id", "opaque_id_16_bytes", "trust_genesis_id_16_bytes"}:
        validate_opaque_id128_v1(value)
        return
    if base_field.endswith("key_id") or base_field in {
        "actor_key_id",
        "auditor_key_id",
    }:
        _require_bytes(value, 16, field)
        return
    if base_field in {"custodian_public_key_32_bytes", "public_key_32_bytes"}:
        _require_bytes(value, 32, field)
        return
    if base_field.endswith("repository_commit_id"):
        _validate_repository_commit(value, field)
        return
    if base_field.endswith("_at_unix_seconds") or base_field in {
        "valid_from_unix_seconds",
        "valid_until_unix_seconds",
        "recorded_at_unix_seconds",
        "created_at_unix_seconds",
        "started_at_unix_seconds",
        "finished_at_unix_seconds",
    }:
        validate_timestamp_v1(value)  # type: ignore[arg-type]
        return
    if base_field == "signatures":
        _validate_signature_records(value)
        return
    if base_field in _ARRAY_FIELDS:
        if not isinstance(value, (tuple, list)):
            _fail("REJECT_M25_FIELD_TYPE", f"{field} must be an array")
        if base_field in {"input_sort_ids", "field_sort_ids"}:
            for item in value:
                NUMERIC_ENUM_REGISTRIES["SortId"].validate(item, field=base_field)
        if base_field in {
            "bucket_key_field_ids",
            "canonical_sort_key_field_ids",
            "required_counter_field_ids",
        } and any(type(item) is not int or item < 0 for item in value):
            _fail("REJECT_M25_FIELD_TYPE", f"{field} must contain unsigned integer IDs")
        if base_field == "ordered_parent_commit_ids":
            for item in value:
                _validate_repository_commit(item, "ordered_parent_commit_ids item")
        return
    if base_field in _BOOL_FIELDS:
        if type(value) is not bool:
            _fail("REJECT_M25_FIELD_TYPE", f"{field} must be a CBOR boolean")
        return
    if base_field == "git_blob_digest":
        _require_bytes(value, 20, field)
        return
    if base_field == "section_blob_sha256":
        _require_bytes(value, 32, field)
        return
    if (
        base_field.endswith("_root")
        or base_field.endswith("_digest")
        or base_field.endswith("_hash")
        or base_field == "formal_digest_or_root"
    ):
        _require_bytes(value, 32, field)
        return
    if base_field in _VARIABLE_BYTES_FIELDS:
        if type(value) is not bytes:
            _fail("REJECT_M25_FIELD_TYPE", f"{field} must be a byte string")
        if base_field in {"raw_repository_path_utf8_bytes", "raw_path_bytes"}:
            if not value or b"\x00" in value:
                _fail("REJECT_M25_FIELD_VALUE", f"{field} must be a nonempty NUL-free path")
        if base_field == "raw_repository_path_utf8_bytes":
            try:
                value.decode("utf-8")
            except UnicodeDecodeError:
                _fail("REJECT_M25_FIELD_VALUE", f"{field} must contain exact UTF-8 bytes")
        return
    if base_field == "canonical_input_object":
        if not isinstance(value, (tuple, list)):
            _fail("REJECT_M25_FIELD_TYPE", "canonical_input_object must be a formal array")
        return
    if base_field == "target_output":
        if type(value) is not int or value not in {0, 1}:
            _fail("FAIL_TARGET_OUTPUT_TYPE", "target output must be CBOR uint Bit 0/1")
        return
    if base_field.endswith("_schema_tag") or base_field in {
        "enclosed_object_tag",
        "target_object_tag",
        "input_object_tag",
    }:
        if type(value) is not int or value not in set(OBJECT_TAGS.values()):
            _fail("REJECT_M25_FIELD_VALUE", f"{field} is not an allocated object tag")
        return
    if (
        base_field.endswith("_id")
        or base_field.endswith("_index")
        or base_field.endswith("_count")
        or base_field.endswith("_seconds")
        or base_field.endswith("_bitmask")
        or base_field.endswith("_length")
        or base_field.endswith("_length_bytes")
        or base_field.endswith("_cardinality")
        or base_field.endswith("_threshold")
        or base_field.endswith("_epoch")
        or base_field.startswith("max_")
        or base_field.startswith("maximum_")
        or base_field in {
            "numeric_id",
            "ast_depth",
            "ast_node_count",
            "distinct_bit_slot_count",
            "program_mdl_length_q32",
            "canonical_program_budget",
            "raw_operator_application_cap",
            "records_per_chunk",
            "process_exit_code",
            "custodian_key_epoch",
            "sequence_number",
            "set_size",
            "a",
            "b",
            "c",
            "d",
            "raw_operator_applications",
            "accepted_canonical_programs",
            "syntactic_duplicates",
            "type_rejections",
            "structural_limit_rejections",
            "rewrite_collapses",
            "file_mode",
            "commit_generation",
            "registry_sequence_number",
        }
    ):
        if type(value) is not int or value < 0:
            _fail(
                "REJECT_M25_FIELD_TYPE",
                f"{field} must be a nonnegative CBOR integer",
            )
        return
    _fail("REJECT_M25_FIELD_TYPE", f"no exact wire type is registered for {field}")


def _require_schema(name: str) -> FormalSchema:
    schema = FORMAL_SCHEMA_REGISTRY.get(name)
    if schema is None:
        if name in AUTHORITATIVE_BLOCKING_GAPS:
            _normative_gap(name, AUTHORITATIVE_BLOCKING_GAPS[name])
        _fail("REJECT_UNKNOWN_M25_SCHEMA", f"unknown formal object {name!r}")
    if schema.wire_gap is not None:
        _normative_gap(name, schema.wire_gap)
    return schema


def build_formal_object(name: str, fields: Mapping[str, object]) -> tuple[object, ...]:
    """Build one strict numeric-array object from a named-field mapping.

    The mapping is accepted only as a Python construction convenience.  Maps
    never enter formal CBOR; values are emitted in the schema's frozen order.
    """

    schema = _require_schema(name)
    if not isinstance(fields, Mapping):
        raise TypeError("formal object fields must be a mapping")
    expected = set(schema.fields)
    actual = set(fields)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        _fail(
            "REJECT_M25_FIELD_SET",
            f"{name} field mismatch; missing={missing}, extra={extra}",
        )
    ordered = tuple(fields[field] for field in schema.fields)
    for field, value in zip(schema.fields, ordered, strict=True):
        _validate_field(field, value)
    _validate_cross_field_guards(name, fields)
    result = schema.prefix + ordered
    # This also rejects text, maps, floats, tags, and unsupported nested types.
    canonical_cbor_encode(result)
    return result


def _require_array(value: object, field: str) -> tuple[object, ...]:
    if not isinstance(value, (tuple, list)):
        _fail("REJECT_M25_FIELD_TYPE", f"{field} must be an array")
    return tuple(value)


def _require_uint(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        _fail("REJECT_M25_FIELD_TYPE", f"{field} must be a nonnegative integer")
    return value


def _validate_exact_quota_table(
    value: object,
    expected: tuple[tuple[int, int, int, int, int], ...],
    field: str,
) -> None:
    rows = _require_array(value, field)
    normalized: list[tuple[int, int, int, int, int]] = []
    for index, raw_row in enumerate(rows):
        row = _require_array(raw_row, f"{field}[{index}]")
        if len(row) != 5:
            _fail("REJECT_M25_FIELD_VALUE", f"{field}[{index}] must have five values")
        normalized_row = tuple(
            _require_uint(item, f"{field}[{index}][{column}]")
            for column, item in enumerate(row)
        )
        assert len(normalized_row) == 5
        if sum(normalized_row[2:]) != normalized_row[1]:
            _fail("REJECT_M25_FIELD_VALUE", f"{field}[{index}] quotas do not exhaust its stratum")
        normalized.append(normalized_row)  # type: ignore[arg-type]
    if tuple(normalized) != expected:
        _fail("REJECT_M25_FIELD_VALUE", f"{field} differs from the frozen quota table")


def _validate_formal_nested_object(
    value: object,
    expected_name: str,
    field: str,
) -> Mapping[str, object]:
    nested = _require_array(value, field)
    schema = _require_schema(expected_name)
    if tuple(nested[:3]) != schema.prefix or len(nested) != 3 + len(schema.fields):
        _fail("REJECT_M25_FIELD_VALUE", f"{field} is not {expected_name}")
    nested_fields = dict(zip(schema.fields, nested[3:], strict=True))
    for nested_field, nested_value in nested_fields.items():
        _validate_field(nested_field, nested_value)
    _validate_cross_field_guards(expected_name, nested_fields)
    return MappingProxyType(nested_fields)


def _validate_sorted_unique_enum_array(
    value: object,
    enum_name: str,
    field: str,
) -> tuple[int, ...]:
    items = _require_array(value, field)
    normalized = tuple(
        NUMERIC_ENUM_REGISTRIES[enum_name].validate(item, field=field) for item in items
    )
    if normalized != tuple(sorted(set(normalized))):
        _fail("REJECT_M25_FIELD_VALUE", f"{field} must be unique and ascending")
    return normalized


def _validate_cross_field_guards(name: str, fields: Mapping[str, object]) -> None:
    """Apply schema-specific guards stated exactly by the frozen documents."""

    if name == "NormativeDocumentBundleV1":
        entries = _require_array(fields["document_entries"], "document_entries")
        normalized: list[tuple[int, bytes]] = []
        for index, raw_entry in enumerate(entries):
            entry = _require_array(raw_entry, f"document_entries[{index}]")
            if len(entry) != 2:
                _fail("REJECT_M25_FIELD_VALUE", "document bundle entry must have two fields")
            role_id = NUMERIC_ENUM_REGISTRIES["NormativeDocumentRoleId"].validate(
                entry[0], field="document role_id"
            )
            _require_bytes(entry[1], 32, "normative document root")
            normalized.append((role_id, entry[1]))  # type: ignore[arg-type]
        if tuple(role for role, _ in normalized) != (1, 2, 3):
            _fail(
                "REJECT_M25_FIELD_VALUE",
                "normative document bundle must contain ordered roles 1,2,3 exactly once",
            )
    elif name == "StaticRoleMetadataV1":
        role_ids = _require_array(fields["role_ids"], "role_ids")
        quantity_ids = _require_array(fields["quantity_ids"], "quantity_ids")
        scope_ids = _require_array(fields["scope_ids"], "scope_ids")
        orientations = _require_array(fields["signed_orientations"], "signed_orientations")
        profile = (tuple(role_ids), tuple(quantity_ids), tuple(scope_ids), tuple(orientations))
        expected = {
            1: ((), (), (), ()),
            2: ((0, 1, 2, 3), (0,), (3,), (1, 1, -1, -1)),
        }[fields["input_signature_id"]]  # type: ignore[index]
        if profile != expected:
            _fail("REJECT_M25_FIELD_VALUE", "static role metadata differs from the frozen profile")
        if any(type(item) is not int for array in profile for item in array):
            _fail("REJECT_M25_FIELD_TYPE", "static role metadata IDs/orientations must be integers")
    elif name == "NormativeApprovalManifestV1":
        if fields["child_dsl_spec_root_or_null"] is None:
            _fail("REJECT_M25_FIELD_NULL", "approved child DSL root must be non-null")
    elif name == "DslRoleBindingManifestV1":
        parent_present = fields["parent_binding_manifest_root_or_null"] is not None
        legacy_absence_present = (
            fields["legacy_parent_payload_source_id_digest_or_null"] is not None
            and fields["parent_manifest_absence_attestation_root_or_null"] is not None
        )
        if parent_present == legacy_absence_present:
            _fail(
                "FAIL_PARENT_BINDING_PROVENANCE_INCOMPLETE",
                "parent binding must XOR legacy payload plus absence attestation",
            )
        if not parent_present and (
            fields["legacy_parent_payload_source_id_digest_or_null"] is None
            or fields["parent_manifest_absence_attestation_root_or_null"] is None
        ):
            _fail(
                "FAIL_PARENT_BINDING_PROVENANCE_INCOMPLETE",
                "both legacy payload digest and absence attestation root are required",
            )
    elif name == "SplitSeedCommitmentManifestV1":
        if fields["seed_length_bytes"] != 32:
            _fail("REJECT_M25_FIELD_VALUE", "split seed length must equal 32")
    elif name == "SplitContractV1":
        if fields["exhaustive_partition_required"] is not True:
            _fail("REJECT_M25_FIELD_VALUE", "split partition must be exhaustive")
        _validate_exact_quota_table(
            fields["odd_stratum_quota_table"], _EXACT_ODD_QUOTAS, "odd_stratum_quota_table"
        )
        _validate_exact_quota_table(
            fields["sink_stratum_quota_table"], _EXACT_SINK_QUOTAS, "sink_stratum_quota_table"
        )
    elif name == "TargetBundleV1":
        if fields["null_control_claim_level_id"] == 1:
            if fields["null_control_required_witness_ast_hash_or_null"] is None:
                _fail(
                    "REJECT_M25_FIELD_NULL",
                    "false-invention null control requires its pre-registered witness hash",
                )
    elif name == "ReplacementPolicyV1":
        expected = {
            "key_rotation_threshold": 2,
            "key_revocation_threshold": 2,
            "custodian_replacement_requires_new_seed_version": True,
            "actor_key_reuse_across_purposes_allowed": False,
            "secret_material_export_allowed": False,
        }
        if any(fields[key] != value for key, value in expected.items()):
            _fail("REJECT_M25_FIELD_VALUE", "replacement policy differs from v1.1.2")
    elif name == "SplitSpecFreezeV1":
        if fields["seed_state_id"] != 1:
            _fail("REJECT_M25_FIELD_VALUE", "Commit-A split freeze must retain uninstantiated seed state")
    elif name == "TombstonePolicyV1":
        if fields["id_reuse_allowed"] is not False:
            _fail("REJECT_M25_FIELD_VALUE", "tombstoned numeric IDs may never be reused")
    elif name == "CrossDslHashPolicyV1":
        expected = {
            "surviving_ast_bytes_stable": True,
            "surviving_ast_hash_stable": True,
            "cross_version_archive_reuse_allowed": False,
            "cross_version_receipt_reuse_allowed": False,
            "cross_version_certificate_reuse_allowed": False,
        }
        if any(fields[key] != value for key, value in expected.items()):
            _fail("REJECT_M25_FIELD_VALUE", "cross-DSL hash policy differs from v1.1.2")
        role_ids = _require_array(
            fields["required_binding_root_role_ids"], "required_binding_root_role_ids"
        )
        for role_id in role_ids:
            NUMERIC_ENUM_REGISTRIES["ArtifactRoleId"].validate(
                role_id, field="required_binding_root_role_ids"
            )
        if tuple(role_ids) != tuple(sorted(set(role_ids))):
            _fail("REJECT_M25_FIELD_VALUE", "binding role IDs must be unique and sorted")
    elif name == "FallbackRegistryV1":
        entries = _require_array(fields["fallback_entries"], "fallback_entries")
        priorities: list[int] = []
        for index, raw_entry in enumerate(entries):
            entry = _require_array(raw_entry, f"fallback_entries[{index}]")
            if len(entry) != 3:
                _fail("REJECT_M25_FIELD_VALUE", "fallback entry must have three fields")
            priorities.append(_require_uint(entry[0], "fallback priority"))
            _require_bytes(entry[1], 32, "fallback target_machine_id_digest")
            if entry[2] is not None:
                _require_bytes(entry[2], 32, "fallback target_spec_root")
        if priorities != sorted(priorities) or len(priorities) != len(set(priorities)):
            _fail("REJECT_M25_FIELD_VALUE", "fallback priorities must be unique and ascending")
    elif name == "TraversalContractV1":
        _validate_sorted_unique_enum_array(
            fields["bucket_key_field_ids"], "TraversalFieldId", "bucket_key_field_ids"
        )
        _validate_sorted_unique_enum_array(
            fields["canonical_sort_key_field_ids"],
            "TraversalFieldId",
            "canonical_sort_key_field_ids",
        )
    elif name == "BucketAccountingContractV1":
        _validate_sorted_unique_enum_array(
            fields["bucket_key_field_ids"], "BucketFieldId", "bucket_key_field_ids"
        )
        _validate_sorted_unique_enum_array(
            fields["required_counter_field_ids"],
            "AccountingCounterFieldId",
            "required_counter_field_ids",
        )
        _validate_sorted_unique_enum_array(
            fields["accounting_sum_invariants"],
            "AccountingInvariantId",
            "accounting_sum_invariants",
        )
    elif name == "ProgramArchiveContractV1":
        if fields["chunk_blob_codec_id"] != 0 or fields["target_independent"] is not True:
            _fail("REJECT_M25_FIELD_VALUE", "program archive must use IDENTITY_V1 and be target-independent")
    elif name == "OutputArchiveContractV1":
        if fields["chunk_blob_codec_id"] != 0 or fields["role_specific"] is not True:
            _fail("REJECT_M25_FIELD_VALUE", "output archive must use IDENTITY_V1 and be role-specific")
    elif name == "StateMachineContractV1":
        if fields["reopen_allowed"] is not False:
            _fail("REJECT_M25_FIELD_VALUE", "terminal M3 states may not reopen")
        terminal = _require_array(fields["terminal_state_ids"], "terminal_state_ids")
        for state_id in terminal:
            NUMERIC_ENUM_REGISTRIES["M3StateId"].validate(state_id, field="terminal_state_ids")
        if tuple(terminal) != (2, 3, 4, 5, 6):
            _fail("REJECT_M25_FIELD_VALUE", "terminal state registry must equal 2..6")
        transitions: list[tuple[object, ...]] = []
        for index, raw_transition in enumerate(
            _require_array(fields["legal_transition_table"], "legal_transition_table")
        ):
            _validate_formal_nested_object(
                raw_transition, "LegalTransitionRowV1", f"legal_transition_table[{index}]"
            )
            transitions.append(tuple(_require_array(raw_transition, "legal transition row")))
        if transitions != sorted(transitions, key=lambda row: row[3:7]):
            _fail("FAIL_ROW_ORDERING", "legal transition rows are not canonically ordered")
    elif name == "InputSignatureSpecV1":
        expected_tag = {1: 0x3401, 2: 0x3402}[fields["input_signature_id"]]  # type: ignore[index]
        if fields["input_object_tag"] != expected_tag:
            _fail("FAIL_INPUT_SIGNATURE_MISMATCH", "input signature and input-object tag differ")
        metadata = _validate_formal_nested_object(
            fields["static_role_metadata"], "StaticRoleMetadataV1", "static_role_metadata"
        )
        if metadata["input_signature_id"] != fields["input_signature_id"]:
            _fail(
                "FAIL_INPUT_SIGNATURE_MISMATCH",
                "input signature and static-role metadata profile differ",
            )
    elif name == "TargetSpecFormalV1":
        if fields["output_sort_id"] != 2:
            _fail("REJECT_M25_FIELD_VALUE", "odd and sink targets must output SortId.BIT")
        if fields["role_id"] == 1:
            if fields["claim_level_id"] != 3:
                _fail("REJECT_M25_FIELD_VALUE", "outside target must use OUTSIDE_TARGET_CANDIDATE")
            if fields["required_witness_ast_hash_or_null"] is not None:
                _fail("REJECT_M25_FIELD_VALUE", "outside target has no designated null witness")
            if fields["universe_row_count"] != 480 or fields["target_output_cardinality"] != 2:
                _fail("REJECT_M25_FIELD_VALUE", "outside target requires 480 rows and two outputs")
        if fields["role_id"] == 2:
            if fields["claim_level_id"] != 1:
                _fail("REJECT_M25_FIELD_VALUE", "current sink control is false-invention null only")
            if fields["required_witness_ast_hash_or_null"] is None:
                _fail("REJECT_M25_FIELD_NULL", "false-invention sink control requires its witness hash")
            if fields["universe_row_count"] != 85 or fields["target_output_cardinality"] != 1:
                _fail("REJECT_M25_FIELD_VALUE", "sink null requires 85 rows and one output")
    elif name == "SplitAlgorithmSpecV1":
        if fields["exhaustive_partition_required"] is not True:
            _fail("REJECT_M25_FIELD_VALUE", "split algorithm must require exhaustive partitioning")
        if fields["assignment_row_schema_tag"] != 0x3203:
            _fail("REJECT_M25_FIELD_VALUE", "split algorithm must emit SplitAssignmentRowV1")
    elif name == "CustodianBindingCoreV1":
        if fields["responsibility_bitmask"] != 0b011111:
            _fail("REJECT_M25_FIELD_VALUE", "pre-M3 custodian responsibility mask must equal 0b011111")
        if fields["valid_until_unix_seconds_or_null"] is not None:
            validate_timestamp_ordering_v1(
                fields["valid_from_unix_seconds"],  # type: ignore[arg-type]
                fields["valid_until_unix_seconds_or_null"],  # type: ignore[arg-type]
            )
    elif name == "ActorKeyManifestV1":
        if fields["valid_until_unix_seconds_or_null"] is not None:
            validate_timestamp_ordering_v1(
                fields["valid_from_unix_seconds"],  # type: ignore[arg-type]
                fields["valid_until_unix_seconds_or_null"],  # type: ignore[arg-type]
            )
    elif name == "ActorTrustGenesisV1":
        entries = _require_array(fields["purpose_key_entries"], "purpose_key_entries")
        normalized: list[tuple[int, bytes]] = []
        for index, raw_entry in enumerate(entries):
            entry = _require_array(raw_entry, f"purpose_key_entries[{index}]")
            if len(entry) != 2:
                _fail("REJECT_M25_FIELD_VALUE", "purpose-key entry must have two fields")
            purpose_id = NUMERIC_ENUM_REGISTRIES["ActorPurposeId"].validate(
                entry[0], field="purpose-key purpose_id"
            )
            _require_bytes(entry[1], 32, "actor_key_manifest_root")
            normalized.append((purpose_id, entry[1]))  # type: ignore[arg-type]
        if tuple(purpose for purpose, _ in normalized) != (1, 2, 3, 4):
            _fail(
                "REJECT_M25_FIELD_VALUE",
                "pre-M4 trust genesis must bind ordered purposes 1,2,3,4 exactly once",
            )
    elif name == "OpaqueIdRegistrySnapshotV1":
        count = fields["record_count"]
        assert isinstance(count, int)
        if count < 1:
            _fail("REJECT_M25_FIELD_VALUE", "opaque-ID snapshot must contain at least one record")
        if fields["previous_snapshot_root_or_null"] is None and count != 1:
            _fail(
                "REJECT_M25_FIELD_VALUE",
                "opaque-ID genesis snapshot must have null predecessor and record_count 1",
            )
        if fields["previous_snapshot_root_or_null"] is not None and count < 2:
            _fail(
                "REJECT_M25_FIELD_VALUE",
                "non-genesis opaque-ID snapshot must have record_count at least 2",
            )
    elif name == "ParentAbsenceAuditBundleV1":
        if _repository_commit_digest(
            fields["audited_parent_repository_commit_id"],
            "audited_parent_repository_commit_id",
        ) != AUDITED_PARENT_COMMIT_SHA1:
            _fail("REJECT_M25_FIELD_VALUE", "parent audit binds the wrong frozen parent commit")
        if fields["legacy_source_count"] != 2:
            _fail("REJECT_M25_FIELD_VALUE", "parent audit must bind exactly two legacy sources")
    elif name == "ParentManifestAbsenceAttestationV2":
        if _repository_commit_digest(
            fields["parent_repository_commit_id"], "parent_repository_commit_id"
        ) != AUDITED_PARENT_COMMIT_SHA1:
            _fail("REJECT_M25_FIELD_VALUE", "parent attestation binds the wrong frozen parent commit")
        if fields["absence_reason_bitmask"] != 0b1111:
            _fail(
                "REJECT_M25_FIELD_VALUE",
                "parent-manifest absence reason bitmask must equal 0b1111",
            )
    elif name == "M3ExecutionCandidateV1":
        if fields["hidden_access_ledger_head_root"] != fields["hidden_access_ledger_genesis_root"]:
            _fail(
                "FAIL_M3_LEDGER_HEAD_NOT_GENESIS",
                "pre-M3 execution candidate requires ledger head equal to genesis",
            )
    elif name == "AttestationBundleV1":
        attestations = _require_array(fields["attestations"], "attestations")
        normalized: list[tuple[int, bytes, bytes]] = []
        for index, raw_attestation in enumerate(attestations):
            attestation = _require_array(raw_attestation, f"attestations[{index}]")
            if len(attestation) != 3:
                _fail("REJECT_M25_FIELD_VALUE", "attestation entry must have three fields")
            purpose_id = NUMERIC_ENUM_REGISTRIES["ActorPurposeId"].validate(
                attestation[0], field="attestation purpose_id"
            )
            _require_bytes(attestation[1], 32, "attestation enclosed_object_root")
            _require_bytes(attestation[2], 32, "attestation signed_envelope_root")
            normalized.append((purpose_id, attestation[1], attestation[2]))  # type: ignore[arg-type]
        if tuple(normalized) != tuple(sorted(normalized)) or len(normalized) != len(set(normalized)):
            _fail("REJECT_M25_FIELD_VALUE", "attestations must be unique and canonically sorted")
    elif name == "OddInputV1":
        set_size = fields["set_size"]
        bits = _require_array(fields["bits"], "bits")
        if set_size not in {5, 6, 7, 8} or len(bits) != set_size:
            _fail("REJECT_M25_FIELD_VALUE", "odd input size and bit count must agree in 5..8")
        if any(type(bit) is not int or bit not in {0, 1} for bit in bits):
            _fail("REJECT_M25_FIELD_TYPE", "odd input bits must be CBOR uint 0/1")
    elif name == "SinkInputV1":
        values = tuple(fields[field] for field in ("a", "b", "c", "d"))
        if any(type(value) is not int or not 0 <= value <= 4 for value in values):
            _fail("REJECT_M25_FIELD_VALUE", "sink input values must be in 0..4")
        if fields["d"] != fields["a"] + fields["b"] - fields["c"]:  # type: ignore[operator]
            _fail("REJECT_M25_FIELD_VALUE", "sink input must satisfy d = a + b - c")
    elif name == "BoundedUniverseRowV1":
        signature_id = fields["input_signature_id"]
        expected_name = {1: "OddInputV1", 2: "SinkInputV1"}[signature_id]  # type: ignore[index]
        _validate_formal_nested_object(
            fields["canonical_input_object"], expected_name, "canonical_input_object"
        )
    elif name == "SplitAssignmentRowV1":
        role_id = fields["role_id"]
        stratum_id = fields["stratum_id"]
        if (role_id == 1 and stratum_id not in range(1, 9)) or (
            role_id == 2 and stratum_id not in range(9, 14)
        ):
            _fail("REJECT_M25_FIELD_VALUE", "role and split stratum are incompatible")
    elif name == "AuditedHistoryRowV1":
        parents = _require_array(fields["ordered_parent_commit_ids"], "ordered_parent_commit_ids")
        parent_digests = tuple(
            _repository_commit_digest(parent, "ordered_parent_commit_ids item") for parent in parents
        )
        if len(parent_digests) != len(set(parent_digests)):
            _fail("REJECT_M25_FIELD_VALUE", "history row contains a duplicate parent commit")
    elif name == "LegacyParentSourceRowV1":
        role_id = fields["target_role_id"]
        assert isinstance(role_id, int)
        namespace_id, diagnostic_digest = _LEGACY_PARENT_SOURCE_EXPECTATIONS[role_id]
        expected_source_id = LEGACY_PARENT_SOURCE_IDS[role_id - 1]
        expected_values = {
            "legacy_parent_payload_source_id_digest": id_digest_v1(expected_source_id),
            "diagnostic_namespace_id": namespace_id,
            "diagnostic_digest": diagnostic_digest,
            "source_repository_commit_id": AUDITED_PARENT_COMMIT_SHA1,
        }
        if fields["legacy_parent_payload_source_id_digest"] != expected_values[
            "legacy_parent_payload_source_id_digest"
        ]:
            _fail("REJECT_M25_FIELD_VALUE", "legacy source ID does not match its target role")
        if fields["diagnostic_namespace_id"] != expected_values["diagnostic_namespace_id"]:
            _fail("REJECT_M25_FIELD_VALUE", "legacy diagnostic namespace does not match its role")
        if fields["diagnostic_digest"] != expected_values["diagnostic_digest"]:
            _fail("REJECT_M25_FIELD_VALUE", "legacy diagnostic digest does not match its source ID")
        if _repository_commit_digest(
            fields["source_repository_commit_id"], "source_repository_commit_id"
        ) != expected_values["source_repository_commit_id"]:
            _fail("REJECT_M25_FIELD_VALUE", "legacy source row binds the wrong parent commit")
    elif name == "LegalTransitionRowV1":
        reasons = _validate_sorted_unique_enum_array(
            fields["allowed_reason_ids"], "M3TransitionReasonId", "allowed_reason_ids"
        )
        if not reasons:
            _fail("REJECT_M25_FIELD_VALUE", "legal transition must allow at least one reason")
        validate_m3_state_transition(
            fields["from_state_id"],  # type: ignore[arg-type]
            fields["from_phase_id"],  # type: ignore[arg-type]
            fields["to_state_id"],  # type: ignore[arg-type]
            fields["to_phase_id"],  # type: ignore[arg-type]
        )
    elif name == "M3RunStateRecordV1":
        validate_m3_state_transition(
            fields["from_state_id"],  # type: ignore[arg-type]
            fields["from_phase_id"],  # type: ignore[arg-type]
            fields["to_state_id"],  # type: ignore[arg-type]
            fields["to_phase_id"],  # type: ignore[arg-type]
        )
        if fields["transition_index"] == 0:
            expected_start = {
                "previous_state_record_root_or_null": None,
                "from_state_id": 0,
                "from_phase_id": 0,
                "to_state_id": 1,
                "to_phase_id": 1,
                "transition_reason_id": 1,
                "triggering_receipt_root_or_null": None,
            }
            if any(fields[field] != value for field, value in expected_start.items()):
                _fail("FAIL_ILLEGAL_M3_STATE_TRANSITION", "transition index 0 is not the exact start record")
        elif fields["previous_state_record_root_or_null"] is None:
            _fail("FAIL_M3_STATE_CHAIN_BREAK", "noninitial transition requires a previous state root")
    elif name == "PartialDiagnosticBundleV1":
        if fields["authoritative_claim_allowed"] is not False:
            _fail("REJECT_M25_FIELD_VALUE", "partial diagnostic bundle cannot make an authoritative claim")
    elif name == "M3RunGenesisV1":
        if fields["initial_state_id"] != 0:
            _fail("REJECT_M25_FIELD_VALUE", "M3 genesis initial state must be NOT_RUN")
        output_fields = tuple(f"{output_name}_or_null" for output_name in M3_RUN_OUTPUT_ROOTS)
        if any(fields[field] is not None for field in output_fields):
            _fail("FAIL_M3_OUTPUT_ROOT_PREPOPULATED", "M3 genesis output root is prepopulated")
    elif name == "ParentManifestAbsenceAttestationV1":
        if fields["absence_reason_bitmask"] != 0b1111:
            _fail(
                "REJECT_M25_FIELD_VALUE",
                "parent-manifest absence reason bitmask must equal 0b1111",
            )
    elif name == "SignedManifestEnvelopeV1":
        signatures = fields["signatures"]
        assert isinstance(signatures, (tuple, list))
        key_ids = [signature[0] for signature in signatures]
        if key_ids != sorted(key_ids) or len(key_ids) != len(set(key_ids)):
            _fail(
                "REJECT_M25_SIGNATURE_ORDER",
                "signature records must have unique ascending key IDs",
            )
        if len(signatures) != 1:
            _fail(
                "REJECT_M25_SIGNATURE_COUNT",
                "every external SignedManifestEnvelopeV1 contains exactly one signature",
            )
    if name in {"M3ImplementationEnumerationReceiptV1", "M3RoleEvaluationReceiptV1"}:
        validate_timestamp_ordering_v1(
            fields["started_at_unix_seconds"],  # type: ignore[arg-type]
            fields["finished_at_unix_seconds"],  # type: ignore[arg-type]
        )


def encode_formal_object(name: str, fields: Mapping[str, object]) -> bytes:
    return canonical_cbor_encode(build_formal_object(name, fields))


@dataclass(frozen=True)
class DecodedFormalObject:
    schema: FormalSchema
    fields: Mapping[str, StrictCborValue]
    value: tuple[StrictCborValue, ...]


def decode_formal_object(
    payload: bytes,
    *,
    expected_name: str | None = None,
) -> DecodedFormalObject:
    """Strictly decode one object and require its exact numeric-array schema."""

    value = canonical_cbor_decode(payload)
    if not isinstance(value, tuple) or len(value) < 3:
        _fail("REJECT_M25_OBJECT_PREFIX", "formal object must be an array with a prefix")
    if type(value[0]) is not int or value[0] != 1:
        _fail("REJECT_M25_OBJECT_PREFIX", "formal object schema version must be 1")
    if type(value[1]) is not int or type(value[2]) is not bytes:
        _fail("REJECT_M25_OBJECT_PREFIX", "formal tag and schema ID have wrong types")
    schema = _SCHEMA_BY_IDENTITY.get((value[1], value[2]))
    if schema is None:
        _fail("REJECT_UNKNOWN_M25_SCHEMA", "unknown tag/schema-ID pair")
    if expected_name is not None and schema.name != expected_name:
        _fail(
            "REJECT_M25_SCHEMA_MISMATCH",
            f"expected {expected_name}, decoded {schema.name}",
        )
    if schema.wire_gap is not None:
        _normative_gap(schema.name, schema.wire_gap)
    if len(value) != 3 + len(schema.fields):
        _fail("REJECT_M25_FIELD_SET", f"wrong array length for {schema.name}")
    field_values = value[3:]
    for field, field_value in zip(schema.fields, field_values, strict=True):
        _validate_field(field, field_value)
    decoded_fields = dict(zip(schema.fields, field_values, strict=True))
    _validate_cross_field_guards(schema.name, decoded_fields)
    return DecodedFormalObject(
        schema=schema,
        fields=MappingProxyType(decoded_fields),
        value=value,
    )


def synthetic_content_root(name: str, fields: Mapping[str, object]) -> bytes:
    """Compute a non-authoritative ContentHash for golden-vector qualification."""

    schema = _require_schema(name)
    if schema.hash_domain is None:
        _normative_gap(name, "no ContentHash domain is frozen for this object")
    return content_hash(schema.hash_domain, build_formal_object(name, fields))


def synthetic_record_tree_root(
    name: str,
    records: Sequence[Mapping[str, object]],
) -> bytes:
    """RFC6962-root ordered synthetic records for cross-implementation tests.

    For ``BoundedUniverseRowV1`` and ``TargetTruthRowV1`` this validates the
    within-tree requirement that ``universe_index`` is unique, contiguous, and
    starts at zero.  A single-tree API cannot establish the cross-tree
    ``canonical_input_hash`` binding between a universe row and its truth row;
    callers must validate that relationship while both row collections are in
    scope before treating the two candidate roots as a qualified pair.
    """

    schema = _require_schema(name)
    if not schema.rfc6962_records:
        _normative_gap(name, "RFC6962 record-tree root rule is not frozen")
    objects = [build_formal_object(name, record) for record in records]
    if name in {"BoundedUniverseRowV1", "TargetTruthRowV1"}:
        universe_indices = [record["universe_index"] for record in records]
        if len(set(universe_indices)) != len(universe_indices):
            _fail(
                "FAIL_UNIVERSE_INDEX_DUPLICATE",
                f"{name} contains a duplicate universe_index",
            )
        expected_indices = list(range(len(universe_indices)))
        if universe_indices != expected_indices:
            if sorted(universe_indices) == expected_indices:
                _fail(
                    "FAIL_ROW_ORDERING",
                    f"{name} universe_index values are not in ascending order",
                )
            _fail(
                "FAIL_UNIVERSE_INDEX_GAP",
                f"{name} universe_index values must be contiguous from zero",
            )
    elif name == "OpaqueIdRegistryRecordV1":
        sequences = [record["registry_sequence_number"] for record in records]
        if sequences != list(range(len(sequences))):
            _fail(
                "FAIL_OPAQUE_ID_REGISTRY_SEQUENCE",
                "opaque-ID registry sequence numbers must be contiguous from zero",
            )
        raw_ids = [record["opaque_id_16_bytes"] for record in records]
        if len(raw_ids) != len(set(raw_ids)):
            _fail(
                "FAIL_OPAQUE_ID_ALREADY_USED",
                "raw opaque ID is reused across registry kinds",
            )
    elif name == "RepositoryPathAliasRecordV1":
        alias_ids = [record["path_alias_id_digest"] for record in records]
        raw_paths = [record["raw_repository_path_utf8_bytes"] for record in records]
        if len(alias_ids) != len(set(alias_ids)) or len(raw_paths) != len(set(raw_paths)):
            _fail(
                "FAIL_REPOSITORY_PATH_ALIAS_DUPLICATE",
                "path alias digests and raw path bytes must both be unique",
            )
    elif name == "LegacyParentSourceRowV1":
        if [record["target_role_id"] for record in records] != [1, 2]:
            _fail(
                "FAIL_PARENT_AUDIT_LEGACY_SOURCE_SET",
                "legacy source tree must contain target roles 1 and 2 exactly once",
            )
    if schema.ordering_fields:
        keys = [tuple(record[field] for field in schema.ordering_fields) for record in records]
        if any(left > right for left, right in zip(keys, keys[1:])):
            _fail(
                "REJECT_M25_RECORD_ORDER",
                f"{name} records violate ordering {schema.ordering_fields}",
            )
    return rfc6962_root(objects)


def candidate_content_root(name: str, fields: Mapping[str, object]) -> bytes:
    """Compute a deterministic v1.1.2 candidate root without an authority claim."""

    return synthetic_content_root(name, fields)


def candidate_record_tree_root(
    name: str,
    records: Sequence[Mapping[str, object]],
) -> bytes:
    """Compute one deterministic v1.1.2 candidate RFC6962 root.

    This delegates all within-tree validation to
    :func:`synthetic_record_tree_root`; it does not claim to validate
    relationships to a separate record tree.
    """

    return synthetic_record_tree_root(name, records)


def external_signature_preimage_v1(
    enclosed_object_tag: int,
    enclosed_object_root: bytes,
    signer_purpose_id: int,
    signer_key_epoch: int,
) -> bytes:
    """Build the owner-authorized purpose/epoch-bound external signature bytes."""

    if type(enclosed_object_tag) is not int:
        _fail("REJECT_M25_FIELD_TYPE", "enclosed object tag must be an integer")
    _require_bytes(enclosed_object_root, 32, "enclosed_object_root")
    purpose_id = NUMERIC_ENUM_REGISTRIES["ActorPurposeId"].validate(
        signer_purpose_id, field="signer_purpose_id"
    )
    if type(signer_key_epoch) is not int or not 0 <= signer_key_epoch <= (1 << 64) - 1:
        _fail("REJECT_M25_FIELD_TYPE", "signer_key_epoch must be a uint64")

    if enclosed_object_tag == 0x310E:
        if purpose_id not in {1, 2, 3}:
            _fail("REJECT_M25_FIELD_VALUE", "bridge statement signer purpose must be 1, 2, or 3")
        domain = BRIDGE_ATTESTATION_SIGNATURE_DOMAIN
    elif enclosed_object_tag in CUSTODIAN_SIGNATURE_DOMAIN_BY_TAG:
        if purpose_id != 1:
            _fail("REJECT_M25_FIELD_VALUE", "custodian object signer purpose must be 1")
        domain = CUSTODIAN_SIGNATURE_DOMAIN_BY_TAG[enclosed_object_tag]
    elif enclosed_object_tag == 0x3114:
        if purpose_id != 4:
            _fail("REJECT_M25_FIELD_VALUE", "parent absence signer purpose must be 4")
        domain = PARENT_AUDITOR_SIGNATURE_DOMAIN
    else:
        _fail("REJECT_M25_FIELD_VALUE", "object tag has no frozen external signature domain")

    return (
        domain.encode("utf-8")
        + b"\x00"
        + enclosed_object_root
        + purpose_id.to_bytes(2, "big")
        + signer_key_epoch.to_bytes(8, "big")
    )


def bridge_attestation_signature_preimage_v1(
    bridge_replay_statement_root: bytes,
    signer_purpose_id: int,
    signer_key_epoch: int,
) -> bytes:
    """Build the exact E2/E3 bridge signature preimage."""

    return external_signature_preimage_v1(
        0x310E,
        bridge_replay_statement_root,
        signer_purpose_id,
        signer_key_epoch,
    )


def validate_source_section_profile_v1(
    name: str,
    fields: Mapping[str, object],
    exact_section_bytes: bytes,
) -> bytes:
    """Recompute one addendum-defined source-section profile candidate root."""

    if name not in {
        "CanonicalAstProfileSpecV1",
        "CanonicalCborProfileSpecV1",
        "Phase2BContractSpecV1",
        "MdlCodeTableSpecV1",
        "HiddenArtifactScopeV1",
    }:
        _fail("REJECT_M25_FIELD_VALUE", "object is not a source-section profile")
    if type(exact_section_bytes) is not bytes:
        _fail("REJECT_M25_FIELD_TYPE", "exact_section_bytes must be bytes")
    if fields["section_blob_sha256"] != hashlib.sha256(exact_section_bytes).digest():
        _fail("FAIL_SOURCE_SECTION_HASH_MISMATCH", "section SHA-256 does not match exact bytes")
    if fields["section_byte_length"] != len(exact_section_bytes):
        _fail("FAIL_SOURCE_SECTION_LENGTH_MISMATCH", "section byte length does not match")
    return candidate_content_root(name, fields)


def validate_null_witness_binding_v1(
    target_spec_fields: Mapping[str, object],
    target_bundle_fields: Mapping[str, object],
) -> None:
    """Require the two E11 sink-witness fields to be non-null and byte-identical."""

    build_formal_object("TargetSpecFormalV1", target_spec_fields)
    build_formal_object("TargetBundleV1", target_bundle_fields)
    if target_spec_fields["role_id"] != 2:
        _fail("REJECT_M25_FIELD_VALUE", "witness binding validator requires the null-control spec")
    spec_witness = target_spec_fields["required_witness_ast_hash_or_null"]
    bundle_witness = target_bundle_fields["null_control_required_witness_ast_hash_or_null"]
    if spec_witness is None or bundle_witness is None or spec_witness != bundle_witness:
        _fail("FAIL_NULL_WITNESS_BINDING_MISMATCH", "sink witness hashes are absent or unequal")


def validate_actor_trust_bindings_v1(
    trust_genesis_fields: Mapping[str, object],
    actor_key_manifests: Sequence[Mapping[str, object]],
    replacement_policy_fields: Mapping[str, object],
) -> bytes:
    """Replay trust entries, policy root, and cross-purpose key separation."""

    if len(actor_key_manifests) != 4:
        _fail("FAIL_ACTOR_TRUST_PURPOSE_SET", "pre-M4 actor trust requires four key manifests")
    normalized: list[tuple[int, bytes]] = []
    key_ids: list[bytes] = []
    public_keys: list[bytes] = []
    for manifest in actor_key_manifests:
        build_formal_object("ActorKeyManifestV1", manifest)
        purpose_id = manifest["purpose_id"]
        assert isinstance(purpose_id, int)
        normalized.append((purpose_id, candidate_content_root("ActorKeyManifestV1", manifest)))
        key_ids.append(manifest["key_id"])  # type: ignore[arg-type]
        public_keys.append(manifest["public_key_32_bytes"])  # type: ignore[arg-type]
    normalized.sort()
    if tuple(purpose for purpose, _ in normalized) != (1, 2, 3, 4):
        _fail("FAIL_ACTOR_TRUST_PURPOSE_SET", "actor key purposes must be exactly 1,2,3,4")
    if len(key_ids) != len(set(key_ids)) or len(public_keys) != len(set(public_keys)):
        _fail("FAIL_ACTOR_KEY_CROSS_PURPOSE_REUSE", "key IDs and public keys must be pairwise distinct")
    expected_policy_root = candidate_content_root("ReplacementPolicyV1", replacement_policy_fields)
    if trust_genesis_fields["purpose_key_policy_root"] != expected_policy_root:
        _fail("FAIL_ACTOR_TRUST_POLICY_MISMATCH", "trust genesis does not bind ReplacementPolicyV1")
    if tuple(tuple(entry) for entry in trust_genesis_fields["purpose_key_entries"]) != tuple(normalized):  # type: ignore[union-attr]
        _fail("FAIL_ACTOR_TRUST_KEY_ROOT_MISMATCH", "trust entries do not match actor key manifests")
    return candidate_content_root("ActorTrustGenesisV1", trust_genesis_fields)


def _candidate_attestation_rows(
    purpose_envelopes: Sequence[tuple[int, Mapping[str, object]]],
) -> tuple[tuple[int, bytes, bytes], ...]:
    rows: list[tuple[int, bytes, bytes]] = []
    for purpose_id, envelope in purpose_envelopes:
        NUMERIC_ENUM_REGISTRIES["ActorPurposeId"].validate(
            purpose_id, field="attestation purpose_id"
        )
        build_formal_object("SignedManifestEnvelopeV1", envelope)
        rows.append(
            (
                purpose_id,
                envelope["enclosed_manifest_root"],  # type: ignore[arg-type]
                candidate_content_root("SignedManifestEnvelopeV1", envelope),
            )
        )
    return tuple(sorted(rows))


def validate_external_input_attestation_bundle_v1(
    bundle_fields: Mapping[str, object],
    purpose_envelopes: Sequence[tuple[int, Mapping[str, object]]],
) -> bytes:
    """Validate the addendum's four custodian plus one auditor envelopes."""

    actual_tag_purposes = sorted(
        (purpose, envelope["enclosed_object_tag"]) for purpose, envelope in purpose_envelopes
    )
    if actual_tag_purposes != sorted(EXTERNAL_INPUT_SIGNED_TAG_PURPOSES):
        _fail("FAIL_EXTERNAL_INPUT_ATTESTATION_COVERAGE", "external-input envelope coverage differs")
    expected_rows = _candidate_attestation_rows(purpose_envelopes)
    if tuple(tuple(row) for row in bundle_fields["attestations"]) != expected_rows:  # type: ignore[union-attr]
        _fail("FAIL_EXTERNAL_INPUT_ATTESTATION_COVERAGE", "bundle rows do not bind the supplied envelopes")
    return candidate_content_root("AttestationBundleV1", bundle_fields)


def validate_bridge_attestation_bundle_v1(
    bridge_statement_root: bytes,
    bundle_fields: Mapping[str, object],
    purpose_envelopes: Sequence[tuple[int, Mapping[str, object]]],
) -> bytes:
    """Validate three purpose-specific envelopes over one bridge statement root."""

    _require_bytes(bridge_statement_root, 32, "bridge_statement_root")
    if sorted(purpose for purpose, _ in purpose_envelopes) != [1, 2, 3]:
        _fail("FAIL_BRIDGE_ATTESTATION_PURPOSE_SET", "bridge purposes must be exactly 1,2,3")
    for _, envelope in purpose_envelopes:
        if envelope["enclosed_object_tag"] != 0x310E or envelope[
            "enclosed_manifest_root"
        ] != bridge_statement_root:
            _fail("FAIL_BRIDGE_ATTESTATION_BINDING", "bridge envelope binds the wrong statement")
    expected_rows = _candidate_attestation_rows(purpose_envelopes)
    if tuple(tuple(row) for row in bundle_fields["attestations"]) != expected_rows:  # type: ignore[union-attr]
        _fail("FAIL_BRIDGE_ATTESTATION_BINDING", "bridge bundle rows do not bind the envelopes")
    return candidate_content_root("AttestationBundleV1", bundle_fields)


def validate_opaque_id_registry_append_v1(
    registration_intents: Sequence[Mapping[str, object]],
    records: Sequence[Mapping[str, object]],
    snapshot_fields: Mapping[str, object],
    *,
    previous_snapshot_fields: Mapping[str, object] | None = None,
) -> bytes:
    """Replay one exact append-only opaque-ID snapshot transition."""

    if not records or len(registration_intents) != len(records):
        _fail(
            "FAIL_OPAQUE_ID_REGISTRY_SNAPSHOT",
            "one registration intent is required for every nonempty registry record set",
        )
    intent_roots: list[bytes] = []
    for intent, record in zip(registration_intents, records, strict=True):
        build_formal_object("OpaqueIdRegistrationIntentV1", intent)
        build_formal_object("OpaqueIdRegistryRecordV1", record)
        if intent["opaque_id_kind_id"] != record["opaque_id_kind_id"] or intent[
            "opaque_id_16_bytes"
        ] != record["opaque_id_16_bytes"]:
            _fail("FAIL_OPAQUE_ID_INTENT_MISMATCH", "registry record differs from its intent")
        intent_root = candidate_content_root("OpaqueIdRegistrationIntentV1", intent)
        if record["first_seen_object_root"] != intent_root:
            _fail("FAIL_OPAQUE_ID_INTENT_MISMATCH", "first_seen_object_root is not the intent root")
        if _repository_commit_digest(
            intent["repository_commit_id"], "intent repository_commit_id"
        ) != _repository_commit_digest(
            record["first_seen_repository_commit_id"], "record first_seen_repository_commit_id"
        ):
            _fail("FAIL_OPAQUE_ID_INTENT_MISMATCH", "intent and record commits differ")
        intent_roots.append(intent_root)

    registry_root = candidate_record_tree_root("OpaqueIdRegistryRecordV1", records)
    build_formal_object("OpaqueIdRegistrySnapshotV1", snapshot_fields)
    if snapshot_fields["record_count"] != len(records) or snapshot_fields[
        "registry_tree_root"
    ] != registry_root:
        _fail("FAIL_OPAQUE_ID_REGISTRY_SNAPSHOT", "snapshot count/tree root differs from records")
    added_record_root = rfc6962_root(
        [build_formal_object("OpaqueIdRegistryRecordV1", records[-1])]
    )
    if snapshot_fields["added_record_root"] != added_record_root:
        _fail("FAIL_OPAQUE_ID_REGISTRY_SNAPSHOT", "added_record_root is not the singleton record root")

    if previous_snapshot_fields is None:
        if snapshot_fields["previous_snapshot_root_or_null"] is not None or len(records) != 1:
            _fail("FAIL_OPAQUE_ID_REGISTRY_SNAPSHOT", "genesis snapshot shape is invalid")
    else:
        build_formal_object("OpaqueIdRegistrySnapshotV1", previous_snapshot_fields)
        previous_root = candidate_content_root(
            "OpaqueIdRegistrySnapshotV1", previous_snapshot_fields
        )
        if snapshot_fields["previous_snapshot_root_or_null"] != previous_root:
            _fail("FAIL_OPAQUE_ID_REGISTRY_SNAPSHOT", "snapshot predecessor root differs")
        if snapshot_fields["record_count"] != previous_snapshot_fields["record_count"] + 1:  # type: ignore[operator]
            _fail("FAIL_OPAQUE_ID_REGISTRY_SNAPSHOT", "snapshot count did not increase by one")
        expected_previous_tree = candidate_record_tree_root(
            "OpaqueIdRegistryRecordV1", records[:-1]
        )
        if previous_snapshot_fields["registry_tree_root"] != expected_previous_tree:
            _fail("FAIL_OPAQUE_ID_REGISTRY_SNAPSHOT", "previous snapshot tree is not the record prefix")
    return candidate_content_root("OpaqueIdRegistrySnapshotV1", snapshot_fields)


def validate_parent_absence_audit_bundle_v1(
    top_level_path_rows: Sequence[Mapping[str, object]],
    history_rows: Sequence[Mapping[str, object]],
    touched_path_rows_by_history_row: Sequence[Sequence[Mapping[str, object]]],
    legacy_source_rows: Sequence[Mapping[str, object]],
    audit_bundle_fields: Mapping[str, object],
) -> bytes:
    """Replay the addendum's reachable-history, touched-path, and legacy-source roots."""

    if len(history_rows) != len(touched_path_rows_by_history_row):
        _fail("FAIL_PARENT_AUDIT_HISTORY", "every history row needs its touched-path rows")
    history_by_digest: dict[bytes, Mapping[str, object]] = {}
    union_rows_by_cbor: dict[bytes, Mapping[str, object]] = {}
    for history_row, touched_rows in zip(
        history_rows, touched_path_rows_by_history_row, strict=True
    ):
        build_formal_object("AuditedHistoryRowV1", history_row)
        commit_digest = _repository_commit_digest(
            history_row["repository_commit_id"], "history repository_commit_id"
        )
        if commit_digest in history_by_digest:
            _fail("FAIL_PARENT_AUDIT_HISTORY", "history contains a duplicate commit")
        history_by_digest[commit_digest] = history_row
        touched_root = candidate_record_tree_root("AuditedPathBlobRecordV1", touched_rows)
        if history_row["touched_path_set_root"] != touched_root:
            _fail("FAIL_PARENT_AUDIT_TOUCHED_PATH_ROOT", "history touched-path root differs")
        for row in touched_rows:
            encoded = encode_formal_object("AuditedPathBlobRecordV1", row)
            union_rows_by_cbor[encoded] = row

    for commit_digest, history_row in history_by_digest.items():
        parent_digests = tuple(
            _repository_commit_digest(parent, "history parent commit")
            for parent in history_row["ordered_parent_commit_ids"]  # type: ignore[union-attr]
        )
        if any(parent not in history_by_digest for parent in parent_digests):
            _fail("FAIL_PARENT_AUDIT_HISTORY", "reachable parent commit is absent")
        expected_generation = (
            0
            if not parent_digests
            else 1
            + max(
                history_by_digest[parent]["commit_generation"]  # type: ignore[type-var]
                for parent in parent_digests
            )
        )
        if history_row["commit_generation"] != expected_generation:
            _fail("FAIL_PARENT_AUDIT_HISTORY", f"commit generation is wrong for {commit_digest.hex()}")
    if AUDITED_PARENT_COMMIT_SHA1 not in history_by_digest:
        _fail("FAIL_PARENT_AUDIT_HISTORY", "frozen audited parent commit is absent")

    expected_union = sorted(
        union_rows_by_cbor.values(),
        key=lambda row: (
            row["raw_repository_path_utf8_bytes"],
            row["repository_path_alias_id_digest"],
            row["git_blob_digest"],
        ),
    )
    if [encode_formal_object("AuditedPathBlobRecordV1", row) for row in top_level_path_rows] != [
        encode_formal_object("AuditedPathBlobRecordV1", row) for row in expected_union
    ]:
        _fail("FAIL_PARENT_AUDIT_PATH_UNION", "top-level path rows are not the deduplicated union")

    path_root = candidate_record_tree_root("AuditedPathBlobRecordV1", top_level_path_rows)
    history_root = candidate_record_tree_root("AuditedHistoryRowV1", history_rows)
    legacy_root = candidate_record_tree_root("LegacyParentSourceRowV1", legacy_source_rows)
    build_formal_object("ParentAbsenceAuditBundleV1", audit_bundle_fields)
    expected_fields = {
        "audited_path_tree_root": path_root,
        "audited_history_tree_root": history_root,
        "legacy_source_tree_root": legacy_root,
        "audited_path_count": len(top_level_path_rows),
        "audited_history_row_count": len(history_rows),
        "legacy_source_count": len(legacy_source_rows),
    }
    if any(audit_bundle_fields[field] != value for field, value in expected_fields.items()):
        _fail("FAIL_PARENT_AUDIT_BUNDLE_MISMATCH", "audit bundle fields differ from replayed rows")
    return candidate_content_root("ParentAbsenceAuditBundleV1", audit_bundle_fields)


def validate_execution_identity_linkage_v1(
    execution_candidate_fields: Mapping[str, object],
    bridge_statement_fields: Mapping[str, object],
    bridge_bundle_fields: Mapping[str, object],
    execution_manifest_fields: Mapping[str, object],
    run_genesis_fields: Mapping[str, object],
) -> tuple[bytes, bytes, bytes]:
    """Replay the acyclic candidate -> statement -> manifest -> genesis linkage."""

    candidate_root = candidate_content_root("M3ExecutionCandidateV1", execution_candidate_fields)
    statement_root = candidate_content_root("BridgeReplayStatementV1", bridge_statement_fields)
    bridge_bundle_root = candidate_content_root("AttestationBundleV1", bridge_bundle_fields)
    manifest_root = candidate_content_root("M3ExecutionManifestV2", execution_manifest_fields)
    build_formal_object("M3RunGenesisV1", run_genesis_fields)

    run_id = execution_candidate_fields["run_id"]
    if any(
        fields["run_id"] != run_id
        for fields in (bridge_statement_fields, execution_manifest_fields, run_genesis_fields)
    ):
        _fail("FAIL_M3_EXECUTION_IDENTITY_LINKAGE", "run IDs differ across execution objects")
    statement_links = {
        "m3_execution_candidate_root": candidate_root,
        "diagnostic_formal_bridge_root": execution_candidate_fields["diagnostic_formal_bridge_root"],
        "child_dsl_spec_root": execution_candidate_fields["child_dsl_spec_root"],
        "child_freeze_root": execution_candidate_fields["child_freeze_root"],
        "actor_trust_genesis_root": execution_candidate_fields["actor_trust_genesis_root"],
        "opaque_id_registry_snapshot_root": execution_candidate_fields[
            "opaque_id_registry_snapshot_root"
        ],
    }
    if any(bridge_statement_fields[field] != value for field, value in statement_links.items()):
        _fail("FAIL_M3_EXECUTION_IDENTITY_LINKAGE", "bridge statement links differ")
    manifest_links = {
        "m3_execution_candidate_root": candidate_root,
        "bridge_replay_statement_root": statement_root,
        "bridge_attestation_bundle_root": bridge_bundle_root,
        "actor_trust_genesis_root": execution_candidate_fields["actor_trust_genesis_root"],
        "opaque_id_registry_snapshot_root": execution_candidate_fields[
            "opaque_id_registry_snapshot_root"
        ],
    }
    if any(execution_manifest_fields[field] != value for field, value in manifest_links.items()):
        _fail("FAIL_M3_EXECUTION_IDENTITY_LINKAGE", "execution manifest links differ")
    if run_genesis_fields["execution_manifest_root"] != manifest_root:
        _fail("FAIL_M3_EXECUTION_IDENTITY_LINKAGE", "run genesis does not bind manifest V2")
    validate_timestamp_ordering_v1(
        execution_candidate_fields["created_at_unix_seconds"],  # type: ignore[arg-type]
        execution_manifest_fields["created_at_unix_seconds"],  # type: ignore[arg-type]
    )
    validate_timestamp_ordering_v1(
        execution_manifest_fields["created_at_unix_seconds"],  # type: ignore[arg-type]
        run_genesis_fields["created_at_unix_seconds"],  # type: ignore[arg-type]
    )
    candidate_commit = _repository_commit_digest(
        execution_candidate_fields["repository_commit_id"], "candidate repository_commit_id"
    )
    for fields in (execution_manifest_fields, run_genesis_fields):
        if _repository_commit_digest(fields["repository_commit_id"], "repository_commit_id") != candidate_commit:
            _fail("FAIL_M3_EXECUTION_IDENTITY_LINKAGE", "execution objects bind different commits")
    return candidate_root, statement_root, manifest_root


def formal_content_root(name: str, fields: Mapping[str, object]) -> bytes:
    """Refuse to label a candidate ContentHash as authoritative evidence."""

    del name, fields
    assert_authoritative_m25_ready()


def formal_record_tree_root(
    name: str,
    records: Sequence[Mapping[str, object]],
) -> bytes:
    """Refuse to label a candidate record root as authoritative evidence."""

    del name, records
    assert_authoritative_m25_ready()


M3_REQUIRED_INPUT_ROOTS: Final = (
    "approval_manifest_root",
    "shrink_transition_root",
    "child_dsl_spec_root",
    "child_freeze_root",
    "operator_semantics_root",
    "identifier_registry_root",
    "canonical_ast_schema_root",
    "canonical_cbor_profile_root",
    "outside_target_binding_manifest_root",
    "null_control_binding_manifest_root",
    "split_seed_commitment_manifest_root",
    "split_binding_manifest_root",
    "custodian_binding_manifest_root",
    "seed_continuity_manifest_root",
    "parent_manifest_absence_attestation_root",
    "hidden_access_ledger_genesis_root",
    "hidden_access_ledger_head_root",
    "diagnostic_formal_bridge_root",
    "outside_target_universe_root",
    "outside_target_truth_root",
    "null_control_universe_root",
    "null_control_truth_root",
    "outside_discovery_split_root",
    "outside_validation_split_root",
    "outside_sealed_split_root",
    "null_discovery_split_root",
    "null_validation_split_root",
    "null_sealed_split_root",
    "python_implementation_binding_root",
    "rust_implementation_binding_root",
    "traversal_contract_root",
    "bucket_accounting_contract_root",
    "program_archive_contract_root",
    "output_archive_contract_root",
    "state_machine_contract_root",
)

M3_RUN_OUTPUT_ROOTS: Final = (
    "canonical_program_archive_root",
    "program_chunk_manifest_root",
    "bucket_accounting_root",
    "outside_program_output_archive_root",
    "outside_output_chunk_manifest_root",
    "outside_match_set_root",
    "outside_role_evaluation_receipt_root",
    "null_program_output_archive_root",
    "null_output_chunk_manifest_root",
    "null_match_set_root",
    "null_role_evaluation_receipt_root",
    "python_enumeration_receipt_root",
    "rust_enumeration_receipt_root",
    "dual_replay_agreement_root",
    "final_state_record_root",
)

if len(M3_RUN_OUTPUT_ROOTS) != M3_RUN_OUTPUT_SLOT_COUNT:
    raise RuntimeError("M3RunGenesisV1 output-slot registry is not exactly fifteen fields")


def validate_m3_input_roots(input_roots: Mapping[str, object]) -> None:
    """Diagnostic guard for the amendment's complete required-input list."""

    if not isinstance(input_roots, Mapping):
        raise TypeError("M3 input roots must be a mapping")
    for name in M3_REQUIRED_INPUT_ROOTS:
        value = input_roots.get(name)
        if value is None:
            _fail("FAIL_M3_INPUT_ROOT_NULL", f"required input root {name} is null or absent")
        _require_bytes(value, 32, name)


def validate_m3_output_roots_null(output_roots: Mapping[str, object]) -> None:
    """Require an explicit null slot for every run-produced output.

    The check is useful for a diagnostic snapshot and ``M3RunGenesisV1``
    construction.  It does not resolve the amendment's 15-versus-16 slot
    contradiction, so authoritative genesis remains fail-closed.
    """

    if not isinstance(output_roots, Mapping):
        raise TypeError("M3 output roots must be a mapping")
    if set(output_roots) != set(M3_RUN_OUTPUT_ROOTS):
        missing = sorted(set(M3_RUN_OUTPUT_ROOTS) - set(output_roots))
        extra = sorted(set(output_roots) - set(M3_RUN_OUTPUT_ROOTS))
        if missing:
            _normative_gap("M3OutputSlotCarrier", f"diagnostic slots absent: {missing}")
        _fail("REJECT_M25_FIELD_SET", f"unexpected M3 output slots: {extra}")
    for name in M3_RUN_OUTPUT_ROOTS:
        if output_roots[name] is not None:
            _fail("FAIL_M3_OUTPUT_ROOT_PREPOPULATED", f"output root {name} is prepopulated")


def validate_hidden_access_ledger_genesis(
    fields: Mapping[str, object],
) -> bytes:
    """Validate the exact genesis event and return its synthetic ContentHash.

    The returned hash is useful for local/root-equality qualification only. It
    is not a signed custodian claim and cannot advance gate 16.
    """

    build_formal_object("HiddenAccessLedgerRecordV1", fields)
    expected_values = {
        "sequence_number": 0,
        "previous_record_root_or_null": None,
        "event_type_id": 1,
        "revealed_artifact_root_or_null": None,
        "authorization_root_or_null": None,
    }
    for field, expected in expected_values.items():
        if fields[field] != expected:
            _fail(
                "FAIL_M3_LEDGER_HEAD_NOT_GENESIS",
                f"ledger genesis requires {field}={expected!r}",
            )
    return synthetic_content_root("HiddenAccessLedgerRecordV1", fields)


def validate_m3_prerun_snapshot(
    input_roots: Mapping[str, object],
    output_roots: Mapping[str, object],
    *,
    ledger_record_count: int,
    ledger_genesis_fields: Mapping[str, object],
) -> None:
    """Apply input, output-null, and exact genesis-ledger diagnostic guards."""

    validate_m3_input_roots(input_roots)
    validate_m3_output_roots_null(output_roots)
    if type(ledger_record_count) is not int or ledger_record_count != 1:
        _fail("FAIL_M3_LEDGER_HEAD_NOT_GENESIS", "ledger record count must equal one")
    if input_roots["hidden_access_ledger_head_root"] != input_roots[
        "hidden_access_ledger_genesis_root"
    ]:
        _fail("FAIL_M3_LEDGER_HEAD_NOT_GENESIS", "ledger head must equal genesis root")
    genesis_root = validate_hidden_access_ledger_genesis(ledger_genesis_fields)
    if ledger_genesis_fields["subject_manifest_root"] != input_roots[
        "split_seed_commitment_manifest_root"
    ]:
        _fail(
            "FAIL_SPLIT_CUSTODIAN_BINDING_MISMATCH",
            "ledger genesis subject must be the split-seed commitment manifest root",
        )
    if genesis_root != input_roots["hidden_access_ledger_genesis_root"]:
        _fail(
            "FAIL_M3_LEDGER_HEAD_NOT_GENESIS",
            "ledger genesis fields do not hash to the bound genesis root",
        )


class M3State(IntEnum):
    NOT_RUN = 0
    RUNNING = 1
    COMPLETE = 2
    DSL_TOO_LARGE = 3
    INCONCLUSIVE_BUDGET = 4
    INCONCLUSIVE_SEMANTICS = 5
    INCONCLUSIVE_EXECUTION = 6


class M3RunningPhase(IntEnum):
    NONE = 0
    CANONICAL_ENUMERATION = 1
    ROLE_EVALUATION = 2


M3_TERMINAL_STATES: Final = frozenset(
    {
        M3State.COMPLETE,
        M3State.DSL_TOO_LARGE,
        M3State.INCONCLUSIVE_BUDGET,
        M3State.INCONCLUSIVE_SEMANTICS,
        M3State.INCONCLUSIVE_EXECUTION,
    }
)

LEGAL_M3_TRANSITIONS: Final = frozenset(
    {
        (
            M3State.NOT_RUN,
            M3RunningPhase.NONE,
            M3State.RUNNING,
            M3RunningPhase.CANONICAL_ENUMERATION,
        ),
        (
            M3State.RUNNING,
            M3RunningPhase.CANONICAL_ENUMERATION,
            M3State.RUNNING,
            M3RunningPhase.ROLE_EVALUATION,
        ),
        (
            M3State.RUNNING,
            M3RunningPhase.CANONICAL_ENUMERATION,
            M3State.DSL_TOO_LARGE,
            M3RunningPhase.NONE,
        ),
        (
            M3State.RUNNING,
            M3RunningPhase.CANONICAL_ENUMERATION,
            M3State.INCONCLUSIVE_BUDGET,
            M3RunningPhase.NONE,
        ),
        (
            M3State.RUNNING,
            M3RunningPhase.CANONICAL_ENUMERATION,
            M3State.INCONCLUSIVE_SEMANTICS,
            M3RunningPhase.NONE,
        ),
        (
            M3State.RUNNING,
            M3RunningPhase.CANONICAL_ENUMERATION,
            M3State.INCONCLUSIVE_EXECUTION,
            M3RunningPhase.NONE,
        ),
        (
            M3State.RUNNING,
            M3RunningPhase.ROLE_EVALUATION,
            M3State.COMPLETE,
            M3RunningPhase.NONE,
        ),
        (
            M3State.RUNNING,
            M3RunningPhase.ROLE_EVALUATION,
            M3State.INCONCLUSIVE_SEMANTICS,
            M3RunningPhase.NONE,
        ),
        (
            M3State.RUNNING,
            M3RunningPhase.ROLE_EVALUATION,
            M3State.INCONCLUSIVE_EXECUTION,
            M3RunningPhase.NONE,
        ),
    }
)


def validate_m3_state_transition(
    from_state: int | M3State,
    from_phase: int | M3RunningPhase,
    to_state: int | M3State,
    to_phase: int | M3RunningPhase,
) -> tuple[M3State, M3RunningPhase, M3State, M3RunningPhase]:
    """Validate the exact transition graph without emitting a state record."""

    try:
        normalized = (
            M3State(from_state),
            M3RunningPhase(from_phase),
            M3State(to_state),
            M3RunningPhase(to_phase),
        )
    except (TypeError, ValueError) as exc:
        _fail("FAIL_ILLEGAL_M3_STATE_TRANSITION", f"unknown state/phase ID: {exc}")
    if normalized[0] in M3_TERMINAL_STATES:
        _fail("FAIL_M3_TERMINAL_STATE_REOPEN", "terminal M3 state cannot transition")
    if normalized not in LEGAL_M3_TRANSITIONS:
        _fail("FAIL_ILLEGAL_M3_STATE_TRANSITION", f"illegal transition {normalized!r}")
    return normalized


def validate_m3_state_chain_link(
    previous_state_record_root: bytes | None,
    expected_previous_state_record_root: bytes | None,
) -> None:
    """Check a supplied chain link without assuming an unfrozen initial index."""

    for name, value in (
        ("previous_state_record_root", previous_state_record_root),
        ("expected_previous_state_record_root", expected_previous_state_record_root),
    ):
        if value is not None:
            _require_bytes(value, 32, name)
    if previous_state_record_root != expected_previous_state_record_root:
        _fail("FAIL_M3_STATE_CHAIN_BREAK", "previous state-record root does not match")


def assert_authoritative_m25_ready() -> "None":
    """Always fail while the enumerated normative gaps remain unresolved."""

    names = ", ".join(AUTHORITATIVE_BLOCKING_GAPS)
    _normative_gap("AuthoritativeM25", f"blocked by: {names}")


__all__ = [
    "AUDITED_PARENT_COMMIT_SHA1",
    "AUTHORITATIVE_MIN_TIMESTAMP",
    "AUTHORITATIVE_BLOCKING_GAPS",
    "BRIDGE_ATTESTATION_SIGNATURE_DOMAIN",
    "CANONICAL_INPUT_DOMAIN",
    "CUSTODIAN_SIGNATURE_DOMAIN_BY_TAG",
    "DecodedFormalObject",
    "EXTERNAL_INPUT_SIGNED_TAG_PURPOSES",
    "FAIL_M25_NORMATIVE_GAP",
    "FORMAL_SCHEMA_REGISTRY",
    "FormalSchema",
    "ID_DIGEST_DOMAIN",
    "LEGAL_M3_TRANSITIONS",
    "MACHINE_FREEZE_ID",
    "MAX_AUTHORITATIVE_FUTURE_SKEW_SECONDS",
    "MAX_TIMESTAMP",
    "M25WireError",
    "M3_REQUIRED_INPUT_ROOTS",
    "M3_RUN_OUTPUT_ROOTS",
    "M3_RUN_OUTPUT_SLOT_COUNT",
    "M3_TERMINAL_STATES",
    "M3RunningPhase",
    "M3State",
    "NUMERIC_ENUM_REGISTRIES",
    "NumericEnumRegistry",
    "OBJECT_TAGS",
    "PARENT_AUDITOR_SIGNATURE_DOMAIN",
    "LEGACY_PARENT_SOURCE_IDS",
    "assert_authoritative_m25_ready",
    "bridge_attestation_signature_preimage_v1",
    "build_formal_object",
    "candidate_content_root",
    "candidate_record_tree_root",
    "decode_formal_object",
    "encode_formal_object",
    "external_signature_preimage_v1",
    "formal_content_root",
    "formal_record_tree_root",
    "git_sha1_commit_id",
    "id_digest_v1",
    "synthetic_content_root",
    "synthetic_record_tree_root",
    "validate_actor_trust_bindings_v1",
    "validate_bridge_attestation_bundle_v1",
    "validate_execution_identity_linkage_v1",
    "validate_external_input_attestation_bundle_v1",
    "validate_null_witness_binding_v1",
    "validate_opaque_id_registry_append_v1",
    "validate_opaque_id128_v1",
    "validate_parent_absence_audit_bundle_v1",
    "validate_source_section_profile_v1",
    "validate_timestamp_ordering_v1",
    "validate_timestamp_v1",
    "validate_m3_input_roots",
    "validate_hidden_access_ledger_genesis",
    "validate_m3_output_roots_null",
    "validate_m3_prerun_snapshot",
    "validate_m3_state_chain_link",
    "validate_m3_state_transition",
]
