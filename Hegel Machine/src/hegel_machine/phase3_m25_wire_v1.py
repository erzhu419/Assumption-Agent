"""Non-authoritative Phase-3A M2.5 formal-wire foundation.

This module implements only byte-level decisions that are unambiguous in
``hegel-freeze-p2b-p3-v1.1.1``.  It deliberately does *not* turn synthetic
objects into M2.5 evidence, advance a gate, or authorize an M3 run.  Several
normative dependencies are still missing from that amendment; each such path
fails with :data:`FAIL_M25_NORMATIVE_GAP` instead of inventing a wire format.

The actual deterministic CBOR codec lives in :mod:`strict_cbor_v1`.  Keeping
the codec shared avoids a subtly different "M2.5 CBOR" implementation while
still requiring exact decode/re-encode at every formal-object boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
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


# The tag table is complete even where the corresponding schema is not.  A tag
# in this registry is therefore not evidence that the object can be emitted.
OBJECT_TAGS: Final = MappingProxyType(
    {
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
        "M3RunStateRecordV1": 0x3301,
        "M3ImplementationEnumerationReceiptV1": 0x3302,
        "M3RoleEvaluationReceiptV1": 0x3303,
        "M3DualReplayAgreementV1": 0x3304,
        "M3RoleAgreementEntryV1": 0x3305,
    }
)


_SCHEMAS = (
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
        "SignedManifestEnvelopeV1",
        0x31FF,
        "hegel-signed-manifest-envelope/1",
        (
            "enclosed_object_tag",
            "enclosed_manifest_root",
            "created_at_unix_seconds",
            "custodian_key_epoch",
            "signatures",
        ),
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
        ordering_fields=("canonical_ast_hash",),
    ),
    _schema(
        "BucketAccountingRecordV1",
        0x320C,
        "hegel-bucket-accounting-record/1",
        (),
        wire_gap="tag is frozen, but array fields, ordering, and root rules are absent",
    ),
    _schema(
        "M3RunStateRecordV1",
        0x3301,
        "hegel-m3-run-state-record/1",
        (),
        hash_domain="HEGEL/M3_RUN_STATE_RECORD/V1",
        wire_gap="the M3 run-state record prefix interpretation is unresolved",
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
        wire_gap="the ContentHash domain for M3DualReplayAgreementV1 is absent",
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
        "IdDigestV1": "machine-ID to 32-byte digest profile is not frozen",
        "FormalRootPreimages": "required contract/spec root objects are not frozen",
        "OddSinkRows": "typed odd/sink canonical input objects are not frozen",
        "SinkSplitContract": "85-row null-control strata and quotas are not frozen",
        "CustodianBindingCoreV1": "tag, schema, fields, and hash domain are absent",
        "CustodianGenesis": "independent actor and persistent custody procedure are absent",
        "AuditorAttestation": "auditor identity/signature and audited tree preimages are absent",
        "DiagnosticBridge": "legacy diagnostic digest profile and bridge signature boundary conflict",
        "BucketAccountingRecordV1": "tag exists but schema and root rules are absent",
        "M3RunStateRecordV1": "record-prefix interpretation remains unresolved",
        "M3DualReplayAgreementV1": "ContentHash domain is absent",
        "M3OutputSlotCarrier": "no formal object carries the required pre-run null output slots",
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
    }
)
_ARRAY_FIELDS: Final = frozenset({"input_sort_ids", "signatures", "role_agreement_entries"})


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

    if base_field in {"run_id", "ledger_id"}:
        _require_bytes(value, 16, field)
        return
    if base_field.endswith("key_id") or base_field in {
        "actor_key_id",
        "auditor_key_id",
    }:
        _require_bytes(value, 16, field)
        return
    if base_field == "custodian_public_key_32_bytes":
        _require_bytes(value, 32, field)
        return
    if base_field in {"repository_commit_id", "parent_repository_commit_id"}:
        _validate_repository_commit(value, field)
        return
    if base_field == "signatures":
        _validate_signature_records(value)
        return
    if base_field in _ARRAY_FIELDS:
        if not isinstance(value, (tuple, list)):
            _fail("REJECT_M25_FIELD_TYPE", f"{field} must be an array")
        if base_field == "input_sort_ids" and any(
            type(item) is not int or item < 0 for item in value
        ):
            _fail(
                "REJECT_M25_FIELD_TYPE",
                "input_sort_ids must contain only nonnegative integer IDs",
            )
        return
    if base_field in _BOOL_FIELDS:
        if type(value) is not bool:
            _fail("REJECT_M25_FIELD_TYPE", f"{field} must be a CBOR boolean")
        return
    if (
        base_field.endswith("_root")
        or base_field.endswith("_digest")
        or base_field.endswith("_hash")
        or base_field == "formal_digest_or_root"
    ):
        _require_bytes(value, 32, field)
        return
    if base_field == "canonical_ast_cbor_bytes":
        if type(value) is not bytes:
            _fail("REJECT_M25_FIELD_TYPE", f"{field} must be a byte string")
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
            "enclosed_object_tag",
            "sequence_number",
        }
    ):
        if type(value) is not int or value < 0:
            _fail(
                "REJECT_M25_FIELD_TYPE",
                f"{field} must be a nonnegative CBOR integer",
            )
        if base_field == "enclosed_object_tag" and value not in set(OBJECT_TAGS.values()):
            _fail("REJECT_M25_FIELD_VALUE", "enclosed object tag is not allocated")
        return


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


def _validate_cross_field_guards(name: str, fields: Mapping[str, object]) -> None:
    """Apply only cross-field guards stated exactly by the amendment."""

    if name == "DslRoleBindingManifestV1":
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
    elif name == "ParentManifestAbsenceAttestationV1":
        if fields["absence_reason_bitmask"] != 0b11:
            _fail(
                "REJECT_M25_FIELD_VALUE",
                "parent-manifest absence reason bitmask must equal 0b11",
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
        if fields["enclosed_object_tag"] in {0x3103, 0x3106, 0x3108} and len(
            signatures
        ) != 1:
            _fail(
                "REJECT_M25_SIGNATURE_COUNT",
                "seed commitment, seed continuity, and ledger genesis require one signature",
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
    """RFC6962-root ordered synthetic records for cross-implementation tests."""

    schema = _require_schema(name)
    if not schema.rfc6962_records:
        _normative_gap(name, "RFC6962 record-tree root rule is not frozen")
    objects = [build_formal_object(name, record) for record in records]
    if schema.ordering_fields:
        keys = [tuple(record[field] for field in schema.ordering_fields) for record in records]
        if any(left > right for left, right in zip(keys, keys[1:])):
            _fail(
                "REJECT_M25_RECORD_ORDER",
                f"{name} records violate ordering {schema.ordering_fields}",
            )
    return rfc6962_root(objects)


def formal_content_root(name: str, fields: Mapping[str, object]) -> bytes:
    """Refuse authoritative ContentHash generation until M2.5 is fully frozen."""

    del name, fields
    assert_authoritative_m25_ready()


def formal_record_tree_root(
    name: str,
    records: Sequence[Mapping[str, object]],
) -> bytes:
    """Refuse authoritative row/archive roots while their preimages are open."""

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

    The check is useful for a diagnostic snapshot.  It does not solve the
    normative gap that no formal object has yet been assigned these slots.
    """

    if not isinstance(output_roots, Mapping):
        raise TypeError("M3 output roots must be a mapping")
    for name in M3_RUN_OUTPUT_ROOTS:
        if name not in output_roots:
            _normative_gap("M3OutputSlotCarrier", f"diagnostic slot {name} is absent")
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
    "AUTHORITATIVE_BLOCKING_GAPS",
    "DecodedFormalObject",
    "FAIL_M25_NORMATIVE_GAP",
    "FORMAL_SCHEMA_REGISTRY",
    "FormalSchema",
    "LEGAL_M3_TRANSITIONS",
    "M25WireError",
    "M3_REQUIRED_INPUT_ROOTS",
    "M3_RUN_OUTPUT_ROOTS",
    "M3_TERMINAL_STATES",
    "M3RunningPhase",
    "M3State",
    "OBJECT_TAGS",
    "assert_authoritative_m25_ready",
    "build_formal_object",
    "decode_formal_object",
    "encode_formal_object",
    "formal_content_root",
    "formal_record_tree_root",
    "git_sha1_commit_id",
    "synthetic_content_root",
    "synthetic_record_tree_root",
    "validate_m3_input_roots",
    "validate_hidden_access_ledger_genesis",
    "validate_m3_output_roots_null",
    "validate_m3_prerun_snapshot",
    "validate_m3_state_chain_link",
    "validate_m3_state_transition",
]
