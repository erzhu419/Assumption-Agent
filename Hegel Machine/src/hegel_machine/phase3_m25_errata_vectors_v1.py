"""Independent Python vectors for the Phase-3A M2.5 exact-wire errata.

The values in this module are public, deterministic qualification material.
They are deliberately derived in Python from the frozen wire schemas and the
synthetic labels below; no Rust output, checked-in golden report, filesystem
state, random source, private key, or external signature is an input.

Candidate hashes produced here are cross-implementation test values only.
They are not formal roots, external-genesis evidence, signatures, or authority
to advance an M3 gate.
"""

from __future__ import annotations

import hashlib
from typing import Callable, Final, Mapping, Sequence

from .phase3_m25_wire_v1 import (
    AUDITED_PARENT_COMMIT_SHA1,
    BRIDGE_ATTESTATION_SIGNATURE_DOMAIN,
    FORMAL_SCHEMA_REGISTRY,
    LEGACY_PARENT_SOURCE_IDS,
    MACHINE_FREEZE_ID,
    M25WireError,
    M3_RUN_OUTPUT_ROOTS,
    OBJECT_TAGS,
    bridge_attestation_signature_preimage_v1,
    build_formal_object,
    candidate_content_root,
    candidate_record_tree_root,
    encode_formal_object,
    git_sha1_commit_id,
    id_digest_v1,
    validate_actor_trust_bindings_v1,
    validate_bridge_attestation_bundle_v1,
    validate_external_input_attestation_bundle_v1,
    validate_null_witness_binding_v1,
    validate_opaque_id_registry_append_v1,
)


VECTOR_SCHEMA: Final = "hegel-phase3-m25-exact-wire-errata-vectors/1"
VECTOR_OPERATION: Final = "errata_vectors"
BASE_TIMESTAMP: Final = 1_704_067_200
VECTOR_ROOT_DOMAIN: Final = b"HEGEL/M25/ERRATA/VECTOR/ROOT/V1\x00"
VECTOR_ID_DOMAIN: Final = b"HEGEL/M25/ERRATA/VECTOR/ID/V1\x00"
VECTOR_KEY_DOMAIN: Final = b"HEGEL/M25/ERRATA/VECTOR/KEY/V1\x00"
VECTOR_SIGNATURE_A_DOMAIN: Final = b"HEGEL/M25/ERRATA/VECTOR/SIG/A/V1\x00"
VECTOR_SIGNATURE_B_DOMAIN: Final = b"HEGEL/M25/ERRATA/VECTOR/SIG/B/V1\x00"
REPOSITORY_COMMIT: Final = git_sha1_commit_id(AUDITED_PARENT_COMMIT_SHA1)


def vector_root_v1(label: str) -> bytes:
    """Return the public 32-byte synthetic root for one ASCII label."""

    return hashlib.sha256(VECTOR_ROOT_DOMAIN + label.encode("ascii")).digest()


def vector_id_v1(label: str) -> bytes:
    """Return the public 16-byte synthetic opaque ID for one ASCII label."""

    return hashlib.sha256(VECTOR_ID_DOMAIN + label.encode("ascii")).digest()[:16]


def vector_key_v1(label: str) -> bytes:
    """Return deterministic public key-shaped bytes for qualification only."""

    return hashlib.sha256(VECTOR_KEY_DOMAIN + label.encode("ascii")).digest()


def vector_signature_v1(label: str) -> bytes:
    """Return deterministic signature-shaped bytes; this is not a signature."""

    raw = label.encode("ascii")
    return hashlib.sha256(VECTOR_SIGNATURE_A_DOMAIN + raw).digest() + hashlib.sha256(
        VECTOR_SIGNATURE_B_DOMAIN + raw
    ).digest()


def _source_profile(
    *,
    identity_field: str,
    identity_label: str,
    selector_label: str,
    blob_label: str,
    byte_length: int,
    governing_root: bytes,
) -> dict[str, object]:
    return {
        identity_field: vector_root_v1(identity_label),
        "governing_normative_document_root": governing_root,
        "section_selector_id_digest": vector_root_v1(selector_label),
        "section_blob_sha256": vector_root_v1(blob_label),
        "section_byte_length": byte_length,
        "repository_commit_id": REPOSITORY_COMMIT,
    }


def _record_rows() -> dict[str, list[dict[str, object]]]:
    raw_path_a = b"Hegel Machine/a.md"
    raw_path_b = b"Hegel Machine/b.md"
    alias_a = id_digest_v1("repo-path:errata-vector-a")
    alias_b = id_digest_v1("repo-path:errata-vector-b")
    blob_a = vector_root_v1("git_blob_a")[:20]
    blob_b = vector_root_v1("git_blob_b")[:20]

    audited_paths = [
        {
            "repository_path_alias_id_digest": alias_a,
            "raw_repository_path_utf8_bytes": raw_path_a,
            "git_object_algorithm_id": 1,
            "git_blob_digest": blob_a,
            "file_mode": 0o100644,
            "byte_length": 123,
        },
        {
            "repository_path_alias_id_digest": alias_b,
            "raw_repository_path_utf8_bytes": raw_path_b,
            "git_object_algorithm_id": 1,
            "git_blob_digest": blob_b,
            "file_mode": 0o100644,
            "byte_length": 456,
        },
    ]
    audited_path_root = candidate_record_tree_root(
        "AuditedPathBlobRecordV1", audited_paths
    )

    legacy_rows = [
        {
            "target_role_id": 1,
            "legacy_parent_payload_source_id_digest": id_digest_v1(
                LEGACY_PARENT_SOURCE_IDS[0]
            ),
            "diagnostic_namespace_id": 1,
            "diagnostic_digest": bytes.fromhex(
                LEGACY_PARENT_SOURCE_IDS[0].rsplit("_", 1)[-1]
            ),
            "source_repository_commit_id": REPOSITORY_COMMIT,
        },
        {
            "target_role_id": 2,
            "legacy_parent_payload_source_id_digest": id_digest_v1(
                LEGACY_PARENT_SOURCE_IDS[1]
            ),
            "diagnostic_namespace_id": 2,
            "diagnostic_digest": bytes.fromhex(
                LEGACY_PARENT_SOURCE_IDS[1].rsplit("_", 1)[-1]
            ),
            "source_repository_commit_id": REPOSITORY_COMMIT,
        },
    ]

    run_intent = {
        "opaque_id_kind_id": 1,
        "opaque_id_16_bytes": vector_id_v1("run_id"),
        "registration_context_root": vector_root_v1("run_registration_context"),
        "created_at_unix_seconds": BASE_TIMESTAMP,
        "repository_commit_id": REPOSITORY_COMMIT,
    }
    opaque_rows = [
        {
            "registry_sequence_number": 0,
            "opaque_id_kind_id": 1,
            "opaque_id_16_bytes": vector_id_v1("run_id"),
            "first_seen_object_root": candidate_content_root(
                "OpaqueIdRegistrationIntentV1", run_intent
            ),
            "first_seen_repository_commit_id": REPOSITORY_COMMIT,
            "created_at_unix_seconds": BASE_TIMESTAMP,
        },
        {
            "registry_sequence_number": 1,
            "opaque_id_kind_id": 2,
            "opaque_id_16_bytes": vector_id_v1("ledger_id"),
            "first_seen_object_root": vector_root_v1(
                "ledger_registration_intent_root"
            ),
            "first_seen_repository_commit_id": REPOSITORY_COMMIT,
            "created_at_unix_seconds": BASE_TIMESTAMP + 1,
        },
    ]

    rows: dict[str, list[dict[str, object]]] = {
        "AuditedPathBlobRecordV1": audited_paths,
        "AuditedHistoryRowV1": [
            {
                "commit_generation": 0,
                "repository_commit_id": REPOSITORY_COMMIT,
                "ordered_parent_commit_ids": (),
                "touched_path_set_root": audited_path_root,
            },
            {
                "commit_generation": 1,
                "repository_commit_id": REPOSITORY_COMMIT,
                "ordered_parent_commit_ids": (REPOSITORY_COMMIT,),
                "touched_path_set_root": audited_path_root,
            },
        ],
        "LegacyParentSourceRowV1": legacy_rows,
        "RepositoryPathAliasRecordV1": sorted(
            [
                {
                    "path_alias_id_digest": alias_a,
                    "raw_repository_path_utf8_bytes": raw_path_a,
                    "repository_commit_id": REPOSITORY_COMMIT,
                },
                {
                    "path_alias_id_digest": alias_b,
                    "raw_repository_path_utf8_bytes": raw_path_b,
                    "repository_commit_id": REPOSITORY_COMMIT,
                },
            ],
            key=lambda row: row["path_alias_id_digest"],
        ),
        "SourceFileRecordV1": [
            {
                "path_alias_id_digest": alias_a,
                "raw_path_bytes": raw_path_a,
                "git_blob_algorithm_id": 1,
                "git_blob_digest": blob_a,
                "file_mode": 0o100644,
                "byte_length": 123,
            },
            {
                "path_alias_id_digest": alias_b,
                "raw_path_bytes": raw_path_b,
                "git_blob_algorithm_id": 1,
                "git_blob_digest": blob_b,
                "file_mode": 0o100644,
                "byte_length": 456,
            },
        ],
        "DependencyLockRecordV1": [
            {
                "ecosystem_id": 1,
                "package_name_id_digest": vector_root_v1("dependency_package_python"),
                "version_id_digest": vector_root_v1("dependency_version_python"),
                "source_id_digest": vector_root_v1("dependency_source_python"),
                "lock_entry_digest": vector_root_v1("dependency_lock_python"),
            },
            {
                "ecosystem_id": 2,
                "package_name_id_digest": vector_root_v1("dependency_package_rust"),
                "version_id_digest": vector_root_v1("dependency_version_rust"),
                "source_id_digest": vector_root_v1("dependency_source_rust"),
                "lock_entry_digest": vector_root_v1("dependency_lock_rust"),
            },
        ],
        "LegalTransitionRowV1": [
            {
                "from_state_id": 0,
                "from_phase_id": 0,
                "to_state_id": 1,
                "to_phase_id": 1,
                "allowed_reason_ids": (1,),
            },
            {
                "from_state_id": 1,
                "from_phase_id": 1,
                "to_state_id": 1,
                "to_phase_id": 2,
                "allowed_reason_ids": (8,),
            },
        ],
        "OpaqueIdRegistryRecordV1": opaque_rows,
    }
    return rows


def _positive_objects(
    rows: Mapping[str, Sequence[Mapping[str, object]]],
) -> tuple[dict[str, tuple[str, Mapping[str, object]]], bytes]:
    normative_bundle = {
        "bundle_id_digest": vector_root_v1("normative_document_bundle_id"),
        "document_entries": (
            (1, vector_root_v1("base_amendment_document_root")),
            (2, vector_root_v1("errata_resolution_document_root")),
            (3, vector_root_v1("implementation_closure_addendum_document_root")),
        ),
        "repository_commit_id": REPOSITORY_COMMIT,
    }
    normative_bundle_root = candidate_content_root(
        "NormativeDocumentBundleV1", normative_bundle
    )

    canonical_ast = _source_profile(
        identity_field="profile_id_digest",
        identity_label="canonical_ast_profile_id",
        selector_label="canonical_ast_section_selector",
        blob_label="canonical_ast_section_blob",
        byte_length=4096,
        governing_root=normative_bundle_root,
    )
    canonical_cbor = _source_profile(
        identity_field="profile_id_digest",
        identity_label="canonical_cbor_profile_id",
        selector_label="canonical_cbor_section_selector",
        blob_label="canonical_cbor_section_blob",
        byte_length=2048,
        governing_root=normative_bundle_root,
    )
    phase2b = _source_profile(
        identity_field="contract_id_digest",
        identity_label="phase2b_contract_id",
        selector_label="phase2b_section_selector",
        blob_label="phase2b_section_blob",
        byte_length=8192,
        governing_root=normative_bundle_root,
    )
    mdl = _source_profile(
        identity_field="table_id_digest",
        identity_label="mdl_code_table_id",
        selector_label="mdl_section_selector",
        blob_label="mdl_section_blob",
        byte_length=1024,
        governing_root=normative_bundle_root,
    )
    hidden_scope = _source_profile(
        identity_field="policy_id_digest",
        identity_label="hidden_artifact_scope_policy_id",
        selector_label="hidden_scope_section_selector",
        blob_label="hidden_scope_section_blob",
        byte_length=512,
        governing_root=normative_bundle_root,
    )

    odd_metadata = {
        "input_signature_id": 1,
        "role_ids": (),
        "quantity_ids": (),
        "scope_ids": (),
        "signed_orientations": (),
        "metadata_rule_id_digest": vector_root_v1("static_role_metadata_rule"),
    }
    sink_metadata = {
        "input_signature_id": 2,
        "role_ids": (0, 1, 2, 3),
        "quantity_ids": (0,),
        "scope_ids": (3,),
        "signed_orientations": (1, 1, -1, -1),
        "metadata_rule_id_digest": vector_root_v1("static_role_metadata_rule"),
    }

    run_intent = {
        "opaque_id_kind_id": 1,
        "opaque_id_16_bytes": vector_id_v1("run_id"),
        "registration_context_root": vector_root_v1("run_registration_context"),
        "created_at_unix_seconds": BASE_TIMESTAMP,
        "repository_commit_id": REPOSITORY_COMMIT,
    }
    opaque_singleton_root = candidate_record_tree_root(
        "OpaqueIdRegistryRecordV1", rows["OpaqueIdRegistryRecordV1"][:1]
    )
    opaque_snapshot = {
        "previous_snapshot_root_or_null": None,
        "registry_tree_root": opaque_singleton_root,
        "record_count": 1,
        "added_record_root": opaque_singleton_root,
        "repository_commit_id": REPOSITORY_COMMIT,
    }

    parent_audit = {
        "audited_parent_repository_commit_id": REPOSITORY_COMMIT,
        "audited_path_tree_root": candidate_record_tree_root(
            "AuditedPathBlobRecordV1", rows["AuditedPathBlobRecordV1"]
        ),
        "audited_history_tree_root": candidate_record_tree_root(
            "AuditedHistoryRowV1", rows["AuditedHistoryRowV1"]
        ),
        "legacy_source_tree_root": candidate_record_tree_root(
            "LegacyParentSourceRowV1", rows["LegacyParentSourceRowV1"]
        ),
        "audited_path_count": 2,
        "audited_history_row_count": 2,
        "legacy_source_count": 2,
    }
    parent_attestation = {
        "parent_dsl_version_digest": vector_root_v1("parent_dsl_version_digest"),
        "parent_freeze_version_digest": vector_root_v1("parent_freeze_version_digest"),
        "parent_repository_commit_id": REPOSITORY_COMMIT,
        "audit_bundle_root": candidate_content_root(
            "ParentAbsenceAuditBundleV1", parent_audit
        ),
        "absence_reason_bitmask": 15,
        "auditor_key_id": vector_id_v1("parent_absence_auditor_key_id"),
        "audited_at_unix_seconds": BASE_TIMESTAMP,
    }
    parent_attestation_root = candidate_content_root(
        "ParentManifestAbsenceAttestationV2", parent_attestation
    )
    parent_envelope = {
        "enclosed_object_tag": OBJECT_TAGS["ParentManifestAbsenceAttestationV2"],
        "enclosed_manifest_root": parent_attestation_root,
        "created_at_unix_seconds": BASE_TIMESTAMP,
        "signer_key_epoch": 0,
        "signatures": (
            (
                vector_id_v1("parent_absence_auditor_key_id"),
                vector_signature_v1("parent_absence_auditor"),
            ),
        ),
    }

    actor_trust = {
        "trust_genesis_id_16_bytes": vector_id_v1("actor_trust_genesis_id"),
        "purpose_key_entries": tuple(
            (
                purpose_id,
                vector_root_v1(f"actor_key_manifest_purpose_{purpose_id}"),
            )
            for purpose_id in range(1, 5)
        ),
        "purpose_key_policy_root": vector_root_v1("replacement_policy_root"),
        "created_at_unix_seconds": BASE_TIMESTAMP,
        "repository_commit_id": REPOSITORY_COMMIT,
    }
    actor_trust_root = candidate_content_root("ActorTrustGenesisV1", actor_trust)
    opaque_snapshot_root = candidate_content_root(
        "OpaqueIdRegistrySnapshotV1", opaque_snapshot
    )
    canonical_ast_root = candidate_content_root("CanonicalAstProfileSpecV1", canonical_ast)
    canonical_cbor_root = candidate_content_root(
        "CanonicalCborProfileSpecV1", canonical_cbor
    )

    candidate_root_fields = (
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
    )
    execution_candidate: dict[str, object] = {
        "run_id": vector_id_v1("run_id"),
        **{field: vector_root_v1(field) for field in candidate_root_fields},
        "canonical_program_budget": 50_000,
        "raw_operator_application_cap": 1_000_000,
        "records_per_chunk": 4096,
        "equivalence_mode_id": 1,
        "python_implementation_binding_root": vector_root_v1(
            "python_implementation_binding_root"
        ),
        "rust_implementation_binding_root": vector_root_v1(
            "rust_implementation_binding_root"
        ),
        "traversal_contract_root": vector_root_v1("traversal_contract_root"),
        "bucket_accounting_contract_root": vector_root_v1(
            "bucket_accounting_contract_root"
        ),
        "program_archive_contract_root": vector_root_v1(
            "program_archive_contract_root"
        ),
        "output_archive_contract_root": vector_root_v1(
            "output_archive_contract_root"
        ),
        "state_machine_contract_root": vector_root_v1("state_machine_contract_root"),
        "created_at_unix_seconds": BASE_TIMESTAMP,
        "repository_commit_id": REPOSITORY_COMMIT,
    }
    execution_candidate.update(
        {
            "canonical_ast_schema_root": canonical_ast_root,
            "canonical_cbor_profile_root": canonical_cbor_root,
            "parent_absence_attestation_root": parent_attestation_root,
            "opaque_id_registry_snapshot_root": opaque_snapshot_root,
            "actor_trust_genesis_root": actor_trust_root,
            "hidden_access_ledger_head_root": execution_candidate[
                "hidden_access_ledger_genesis_root"
            ],
        }
    )
    if tuple(execution_candidate) != FORMAL_SCHEMA_REGISTRY["M3ExecutionCandidateV1"].fields:
        raise AssertionError("M3ExecutionCandidateV1 synthetic fields drifted from the schema")
    execution_candidate_root = candidate_content_root(
        "M3ExecutionCandidateV1", execution_candidate
    )

    bridge_statement = {
        "run_id": vector_id_v1("run_id"),
        "diagnostic_formal_bridge_root": vector_root_v1(
            "diagnostic_formal_bridge_root"
        ),
        "m3_execution_candidate_root": execution_candidate_root,
        "child_dsl_spec_root": vector_root_v1("child_dsl_spec_root"),
        "child_freeze_root": vector_root_v1("child_freeze_root"),
        "actor_trust_genesis_root": actor_trust_root,
        "opaque_id_registry_snapshot_root": opaque_snapshot_root,
    }
    bridge_statement_root = candidate_content_root(
        "BridgeReplayStatementV1", bridge_statement
    )
    execution_manifest = {
        "run_id": vector_id_v1("run_id"),
        "m3_execution_candidate_root": execution_candidate_root,
        "bridge_replay_statement_root": bridge_statement_root,
        "bridge_attestation_bundle_root": vector_root_v1(
            "bridge_attestation_bundle_root"
        ),
        "actor_trust_genesis_root": actor_trust_root,
        "opaque_id_registry_snapshot_root": opaque_snapshot_root,
        "created_at_unix_seconds": BASE_TIMESTAMP,
        "repository_commit_id": REPOSITORY_COMMIT,
    }
    execution_manifest_root = candidate_content_root(
        "M3ExecutionManifestV2", execution_manifest
    )
    run_genesis: dict[str, object] = {
        "run_id": vector_id_v1("run_id"),
        "execution_manifest_root": execution_manifest_root,
        "initial_state_id": 0,
        **{f"{name}_or_null": None for name in M3_RUN_OUTPUT_ROOTS},
        "created_at_unix_seconds": BASE_TIMESTAMP,
        "repository_commit_id": REPOSITORY_COMMIT,
    }
    if tuple(run_genesis) != FORMAL_SCHEMA_REGISTRY["M3RunGenesisV1"].fields:
        raise AssertionError("M3RunGenesisV1 synthetic fields drifted from the schema")
    start_state = {
        "run_id": vector_id_v1("run_id"),
        "transition_index": 0,
        "previous_state_record_root_or_null": None,
        "from_state_id": 0,
        "from_phase_id": 0,
        "to_state_id": 1,
        "to_phase_id": 1,
        "transition_reason_id": 1,
        "execution_manifest_root": execution_manifest_root,
        "triggering_receipt_root_or_null": None,
        "recorded_at_unix_seconds": BASE_TIMESTAMP,
    }
    dual_replay = {
        "run_id": vector_id_v1("run_id"),
        "execution_manifest_root": execution_manifest_root,
        "python_enumeration_receipt_root": vector_root_v1(
            "python_enumeration_receipt_root"
        ),
        "rust_enumeration_receipt_root": vector_root_v1(
            "rust_enumeration_receipt_root"
        ),
        "agreed_closure_status_id": 0,
        "canonical_program_count_or_null": None,
        "closure_cardinality_or_null": None,
        "canonical_program_archive_root_or_null": None,
        "program_chunk_manifest_root_or_null": None,
        "bucket_accounting_root_or_null": None,
        "first_out_of_budget_program_hash_or_null": None,
        "role_agreement_entries": (),
        "enumeration_agreement": True,
        "role_agreement_status_id": 0,
        "mismatch_record_root_or_null": None,
        "created_at_unix_seconds": BASE_TIMESTAMP,
    }

    objects: dict[str, tuple[str, Mapping[str, object]]] = {
        "ActorTrustGenesisV1": ("ActorTrustGenesisV1", actor_trust),
        "BridgeReplayStatementV1": ("BridgeReplayStatementV1", bridge_statement),
        "CanonicalAstProfileSpecV1": ("CanonicalAstProfileSpecV1", canonical_ast),
        "CanonicalCborProfileSpecV1": ("CanonicalCborProfileSpecV1", canonical_cbor),
        "HiddenArtifactScopeV1": ("HiddenArtifactScopeV1", hidden_scope),
        "M3DualReplayAgreementV1": ("M3DualReplayAgreementV1", dual_replay),
        "M3ExecutionCandidateV1": ("M3ExecutionCandidateV1", execution_candidate),
        "M3ExecutionManifestV2": ("M3ExecutionManifestV2", execution_manifest),
        "M3RunGenesisV1": ("M3RunGenesisV1", run_genesis),
        "M3RunStateRecordV1.synthetic_start_shape": ("M3RunStateRecordV1", start_state),
        "MdlCodeTableSpecV1": ("MdlCodeTableSpecV1", mdl),
        "NormativeDocumentBundleV1": ("NormativeDocumentBundleV1", normative_bundle),
        "OpaqueIdRegistrationIntentV1.run": ("OpaqueIdRegistrationIntentV1", run_intent),
        "OpaqueIdRegistrySnapshotV1.genesis": (
            "OpaqueIdRegistrySnapshotV1",
            opaque_snapshot,
        ),
        "ParentAbsenceAuditBundleV1": ("ParentAbsenceAuditBundleV1", parent_audit),
        "ParentManifestAbsenceAttestationV2": (
            "ParentManifestAbsenceAttestationV2",
            parent_attestation,
        ),
        "Phase2BContractSpecV1": ("Phase2BContractSpecV1", phase2b),
        "SignedManifestEnvelopeV1.parent_absence": (
            "SignedManifestEnvelopeV1",
            parent_envelope,
        ),
        "StaticRoleMetadataV1.odd": ("StaticRoleMetadataV1", odd_metadata),
        "StaticRoleMetadataV1.sink": ("StaticRoleMetadataV1", sink_metadata),
    }
    return objects, bridge_statement_root


def _object_item(
    name: str,
    schema_name: str,
    fields: Mapping[str, object],
) -> dict[str, object]:
    return {
        "name": name,
        "schema_name": schema_name,
        "tag": OBJECT_TAGS[schema_name],
        "status": "PASS_CANDIDATE_NON_AUTHORITATIVE",
        "bytes_hex": encode_formal_object(schema_name, fields).hex(),
        "candidate_root_hex": candidate_content_root(schema_name, fields).hex(),
        "error_code": None,
    }


def _record_tree_item(
    schema_name: str,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not rows:
        raise AssertionError(f"{schema_name} synthetic record tree must be nonempty")
    return {
        "name": schema_name,
        "schema_name": schema_name,
        "tag": OBJECT_TAGS[schema_name],
        "status": "PASS_CANDIDATE_NON_AUTHORITATIVE",
        "record_count": len(rows),
        "first_record_cbor_hex": encode_formal_object(schema_name, rows[0]).hex(),
        "root_hex": candidate_record_tree_root(schema_name, rows).hex(),
        "error_code": None,
    }


def _capture_guard_error(
    vector_id: str,
    action: Callable[[], object],
) -> dict[str, str]:
    try:
        action()
    except M25WireError as error:
        return {"vector_id": vector_id, "error_code": error.code}
    raise AssertionError(f"negative vector {vector_id!r} unexpectedly passed")


def _replacement_policy() -> dict[str, object]:
    return {
        "key_rotation_threshold": 2,
        "key_revocation_threshold": 2,
        "custodian_replacement_requires_new_seed_version": True,
        "actor_key_reuse_across_purposes_allowed": False,
        "secret_material_export_allowed": False,
    }


def _actor_key_manifest(purpose_id: int) -> dict[str, object]:
    return {
        "purpose_id": purpose_id,
        "key_id": vector_id_v1(f"guard_actor_key_id_{purpose_id}"),
        "public_key_32_bytes": vector_key_v1(f"guard_actor_public_key_{purpose_id}"),
        "key_epoch": 0,
        "valid_from_unix_seconds": BASE_TIMESTAMP,
        "valid_until_unix_seconds_or_null": None,
        "repository_commit_id": REPOSITORY_COMMIT,
    }


def _guard_actor_trust(
    keys: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "trust_genesis_id_16_bytes": vector_id_v1("guard_actor_trust_genesis"),
        "purpose_key_entries": tuple(
            (
                key["purpose_id"],
                candidate_content_root("ActorKeyManifestV1", key),
            )
            for key in keys
        ),
        "purpose_key_policy_root": candidate_content_root(
            "ReplacementPolicyV1", _replacement_policy()
        ),
        "created_at_unix_seconds": BASE_TIMESTAMP,
        "repository_commit_id": REPOSITORY_COMMIT,
    }


def _guard_envelope(tag: int, label: str) -> dict[str, object]:
    return {
        "enclosed_object_tag": tag,
        "enclosed_manifest_root": vector_root_v1(label),
        "created_at_unix_seconds": BASE_TIMESTAMP,
        "signer_key_epoch": 0,
        "signatures": ((vector_id_v1(f"guard_signer_{label}"), vector_signature_v1(label)),),
    }


def _negative_guard_errors(
    rows: Mapping[str, Sequence[Mapping[str, object]]],
    objects: Mapping[str, tuple[str, Mapping[str, object]]],
) -> list[dict[str, str]]:
    actor_keys = [_actor_key_manifest(purpose_id) for purpose_id in range(1, 5)]
    actor_trust = _guard_actor_trust(actor_keys)
    reused_public_keys = [dict(key) for key in actor_keys]
    reused_public_keys[3]["public_key_32_bytes"] = reused_public_keys[0][
        "public_key_32_bytes"
    ]
    reused_manifest_trust = dict(actor_trust)
    reused_manifest_entries = [list(entry) for entry in actor_trust["purpose_key_entries"]]  # type: ignore[arg-type]
    reused_manifest_entries[3][1] = reused_manifest_entries[0][1]
    reused_manifest_trust["purpose_key_entries"] = tuple(
        tuple(entry) for entry in reused_manifest_entries
    )

    bridge_root = vector_root_v1("guard_bridge_statement")
    bridge_envelopes = [
        (purpose_id, _guard_envelope(0x310E, "guard_bridge_statement"))
        for purpose_id in (1, 2)
    ]
    external_envelopes = [
        (purpose_id, _guard_envelope(tag, f"guard_external_{tag:04x}"))
        for purpose_id, tag in ((1, 0x3103), (1, 0x3105), (1, 0x3106), (1, 0x3108))
    ]

    normative_schema, normative = objects["NormativeDocumentBundleV1"]
    assert normative_schema == "NormativeDocumentBundleV1"
    wrong_documents = dict(normative)
    wrong_documents["document_entries"] = tuple(reversed(normative["document_entries"]))  # type: ignore[arg-type]

    genesis_schema, genesis = objects["M3RunGenesisV1"]
    assert genesis_schema == "M3RunGenesisV1"
    prepopulated_genesis = dict(genesis)
    prepopulated_genesis["canonical_program_archive_root_or_null"] = vector_root_v1(
        "guard_prepopulated_output"
    )

    state_schema, start_state = objects["M3RunStateRecordV1.synthetic_start_shape"]
    assert state_schema == "M3RunStateRecordV1"
    wrong_start = dict(start_state)
    wrong_start["transition_reason_id"] = 2

    sink_schema, sink_metadata = objects["StaticRoleMetadataV1.sink"]
    assert sink_schema == "StaticRoleMetadataV1"
    wrong_sink = dict(sink_metadata)
    wrong_sink["signed_orientations"] = (1, 1, 1, -1)

    parent_schema, parent_attestation = objects["ParentManifestAbsenceAttestationV2"]
    assert parent_schema == "ParentManifestAbsenceAttestationV2"
    wrong_parent = dict(parent_attestation)
    wrong_parent["absence_reason_bitmask"] = 7

    null_witness = vector_root_v1("guard_null_witness")
    null_target = {
        "role_id": 2,
        "target_machine_id_digest": vector_root_v1("guard_null_target_id"),
        "input_signature_spec_root": vector_root_v1("guard_null_input_signature"),
        "output_sort_id": 2,
        "target_rule_id_digest": vector_root_v1("guard_null_rule"),
        "universe_row_count": 85,
        "target_output_cardinality": 1,
        "required_witness_ast_hash_or_null": null_witness,
        "claim_level_id": 1,
    }
    null_bundle = {
        "outside_target_spec_root": vector_root_v1("guard_outside_spec"),
        "outside_target_universe_root": vector_root_v1("guard_outside_universe"),
        "outside_target_truth_root": vector_root_v1("guard_outside_truth"),
        "null_control_spec_root": vector_root_v1("guard_null_spec"),
        "null_control_universe_root": vector_root_v1("guard_null_universe"),
        "null_control_truth_root": vector_root_v1("guard_null_truth"),
        "fallback_registry_root": vector_root_v1("guard_fallback_registry"),
        "null_control_required_witness_ast_hash_or_null": vector_root_v1(
            "guard_wrong_null_witness"
        ),
        "null_control_claim_level_id": 1,
    }

    run_schema, run_intent = objects["OpaqueIdRegistrationIntentV1.run"]
    assert run_schema == "OpaqueIdRegistrationIntentV1"
    first_opaque_record = dict(rows["OpaqueIdRegistryRecordV1"][0])
    wrong_snapshot = {
        "previous_snapshot_root_or_null": None,
        "registry_tree_root": vector_root_v1("guard_wrong_registry_tree"),
        "record_count": 1,
        "added_record_root": candidate_record_tree_root(
            "OpaqueIdRegistryRecordV1", [first_opaque_record]
        ),
        "repository_commit_id": REPOSITORY_COMMIT,
    }
    duplicate_opaque_records = [
        first_opaque_record,
        {
            **dict(rows["OpaqueIdRegistryRecordV1"][1]),
            "opaque_id_16_bytes": first_opaque_record["opaque_id_16_bytes"],
        },
    ]
    sequence_gap_records = [
        first_opaque_record,
        {
            **dict(rows["OpaqueIdRegistryRecordV1"][1]),
            "registry_sequence_number": 2,
        },
    ]

    guards = [
        _capture_guard_error(
            "actor_public_key_reused_across_purposes",
            lambda: validate_actor_trust_bindings_v1(
                actor_trust, reused_public_keys, _replacement_policy()
            ),
        ),
        _capture_guard_error(
            "actor_trust_missing_purpose",
            lambda: validate_actor_trust_bindings_v1(
                actor_trust, actor_keys[:3], _replacement_policy()
            ),
        ),
        _capture_guard_error(
            "actor_trust_reused_manifest_root",
            lambda: validate_actor_trust_bindings_v1(
                reused_manifest_trust, actor_keys, _replacement_policy()
            ),
        ),
        _capture_guard_error(
            "audited_path_wrong_order",
            lambda: candidate_record_tree_root(
                "AuditedPathBlobRecordV1", list(reversed(rows["AuditedPathBlobRecordV1"]))
            ),
        ),
        _capture_guard_error(
            "bridge_attester_purpose_order",
            lambda: validate_bridge_attestation_bundle_v1(
                bridge_root, {"attestations": ()}, bridge_envelopes
            ),
        ),
        _capture_guard_error(
            "document_roles_wrong_order",
            lambda: build_formal_object("NormativeDocumentBundleV1", wrong_documents),
        ),
        _capture_guard_error(
            "external_attestation_missing_auditor",
            lambda: validate_external_input_attestation_bundle_v1(
                {"attestations": ()}, external_envelopes
            ),
        ),
        _capture_guard_error(
            "m3_genesis_output_prepopulated",
            lambda: build_formal_object("M3RunGenesisV1", prepopulated_genesis),
        ),
        _capture_guard_error(
            "m3_start_wrong_reason",
            lambda: build_formal_object("M3RunStateRecordV1", wrong_start),
        ),
        _capture_guard_error(
            "null_witness_mismatch",
            lambda: validate_null_witness_binding_v1(null_target, null_bundle),
        ),
        _capture_guard_error(
            "opaque_registry_raw_id_reuse_across_kinds",
            lambda: candidate_record_tree_root(
                "OpaqueIdRegistryRecordV1", duplicate_opaque_records
            ),
        ),
        _capture_guard_error(
            "opaque_registry_sequence_gap",
            lambda: candidate_record_tree_root(
                "OpaqueIdRegistryRecordV1", sequence_gap_records
            ),
        ),
        _capture_guard_error(
            "opaque_snapshot_tree_root_mismatch",
            lambda: validate_opaque_id_registry_append_v1(
                [run_intent], [first_opaque_record], wrong_snapshot
            ),
        ),
        _capture_guard_error(
            "parent_absence_bitmask_not_15",
            lambda: build_formal_object("ParentManifestAbsenceAttestationV2", wrong_parent),
        ),
        _capture_guard_error(
            "sink_static_orientation_mismatch",
            lambda: build_formal_object("StaticRoleMetadataV1", wrong_sink),
        ),
    ]
    return sorted(guards, key=lambda item: item["vector_id"])


def generate_errata_vector_report_v1() -> dict[str, object]:
    """Generate the complete non-authoritative Python errata-vector report."""

    rows = _record_rows()
    positive_objects, bridge_statement_root = _positive_objects(rows)

    object_items = [
        _object_item(name, schema_name, fields)
        for name, (schema_name, fields) in positive_objects.items()
    ]
    bridge_preimage = bridge_attestation_signature_preimage_v1(
        bridge_statement_root, 2, 7
    )
    object_items.append(
        {
            "name": "BridgeAttestationSignaturePreimageV1",
            "schema_name": "raw-signature-preimage",
            "tag": 0,
            "status": "PASS_CANDIDATE_NON_AUTHORITATIVE",
            "bytes_hex": bridge_preimage.hex(),
            "candidate_root_hex": None,
            "error_code": None,
        }
    )
    object_items.sort(key=lambda item: item["name"])

    record_tree_items = [
        _record_tree_item(schema_name, records)
        for schema_name, records in rows.items()
    ]
    record_tree_items.sort(key=lambda item: item["name"])

    return {
        "ok": True,
        "op": VECTOR_OPERATION,
        "machine_freeze_id": MACHINE_FREEZE_ID,
        "vector_schema": VECTOR_SCHEMA,
        "objects": object_items,
        "record_trees": record_tree_items,
        "guard_errors": _negative_guard_errors(rows, positive_objects),
    }


__all__ = [
    "BASE_TIMESTAMP",
    "REPOSITORY_COMMIT",
    "VECTOR_ID_DOMAIN",
    "VECTOR_KEY_DOMAIN",
    "VECTOR_OPERATION",
    "VECTOR_ROOT_DOMAIN",
    "VECTOR_SCHEMA",
    "VECTOR_SIGNATURE_A_DOMAIN",
    "VECTOR_SIGNATURE_B_DOMAIN",
    "generate_errata_vector_report_v1",
    "vector_id_v1",
    "vector_key_v1",
    "vector_root_v1",
    "vector_signature_v1",
]
