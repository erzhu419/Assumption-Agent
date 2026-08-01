"""Fail-closed publication bundle for Phase-3 shrink step 1.

This module records the user-approved child language, dual subset evidence,
diagnostic DSL bindings, and every still-closed M3 gate.  It does not fabricate
the absent historical split seed, parent binding manifests, custodian
attestation, or formal roots.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

from .hashing import stable_hash
from .phase3_dsl_v1 import (
    OBSERVED_OMITTED_SINK_CONTROL,
    ODD_REDUCTION_TARGET,
    ODD_REDUCTION_SPLITS,
)
from .phase3_shrink1_registry_v1 import (
    ACTIVE_AGGREGATE_IDS,
    AGGREGATE_REGISTRY_DIAGNOSTIC_ID,
    AGGREGATE_REGISTRY_POLICY,
    AST_SCHEMA_ID,
    CBOR_PROFILE_ID,
    DSL_VERSION,
    FORMAL_ROOT_NAMES,
    FREEZE_VERSION,
    HUMAN_AMENDMENT_ID,
    OPERATOR_ADMISSION_SEMANTICS_DIAGNOSTIC_ID,
    PARENT_DSL_VERSION,
    PARENT_FREEZE_VERSION,
    REMOVED_AGGREGATE_ERROR,
    SHRINK_STEP_ID,
    SHRUNK_DSL_SURFACE_DIAGNOSTIC_ID,
    TOMBSTONED_AGGREGATE_IDS,
    aggregate_registry_object,
    operator_admission_semantics_object,
    shrunk_dsl_surface_object,
)
from .phase3_shrink1_replay_v1 import (
    DEFAULT_RUST_BINARY,
    dual_shrink1_capacity_replay_report,
    dual_shrink1_strict_gate_report,
)


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
NORMATIVE_DOCUMENT_PATH: Final = (
    PROJECT_ROOT / "docs" / "Hegel_Machine_Phase3_Shrink_Step1_Freeze_Decisions.md"
)

PARENT_TRIGGER_COMMIT: Final = "fb3a3ee4865a140c558821017ddd3e9a6a99de48"
PARENT_GATE_REPORT_ID: Final = (
    "phase3_dual_strict_gate_"
    "06eae23f68536e3f7e80badb46a5b15e0665072f65477608a3f688e54adefad6"
)
PARENT_CAPACITY_REPLAY_REPORT_ID: Final = (
    "phase3_dual_strict_capacity_replay_"
    "f75214e75f5fc3812d7375463ba72c347c9c08bc7bae3b68c87a63b484c4e414"
)
PARENT_PYTHON_STRICT_SOURCE_ROOT: Final = (
    "sha256:bb3d9b3ee9b270165f66f0e0d8fcc3c364226b38290ea2bf3b09ebad34fe5c9a"
)
PARENT_PYTHON_CAPACITY_SOURCE_ROOT: Final = (
    "sha256:eb8a0b6f6425084c964ebb200ec6eeeb995f0ac4a8909e5d551ad0eb88c0d525"
)
PARENT_RUST_SOURCE_ROOT: Final = (
    "sha256:98fec63ea16d4e5ded2fc09ad8ed57b8cc2f599234c59fbe86868d445401e46f"
)
PARENT_CAPACITY_SET_COMMITMENT: Final = (
    "sha256:c1a02a66a8d6d8f75204cb3daf03ab0b01c2b3b8e486d0ab3d481ee3be43c930"
)

TARGET_ROLE_OUTSIDE: Final = "OUTSIDE_TARGET"
TARGET_ROLE_NULL: Final = "IN_LANGUAGE_NULL"
OLD_VALIDATION_DISPOSITION: Final = "HISTORICAL_PRECOMMITMENT_ONLY_SEALED"

M3_REQUIRED_GATES: Final = (
    "SHRINK1_NORMATIVE_AMENDMENT_APPROVED",
    "NEW_DSL_AND_FREEZE_IDS_COMMITTED",
    "SPARSE_AGGREGATE_REGISTRY_FROZEN",
    "TOMBSTONE_REJECTION_FROZEN",
    "CROSS_DSL_HASH_POLICY_FROZEN",
    "PYTHON_STRICT_IMPLEMENTATION_UPDATED",
    "RUST_STRICT_IMPLEMENTATION_UPDATED",
    "SHRINK1_GOLDEN_VECTORS_EQUAL",
    "REMOVED_MAP_VECTORS_REJECTED_IDENTICALLY",
    "SURVIVING_AST_HASH_STABILITY_VERIFIED",
    "SHRINK1_SOURCE_SUBSET_COUNT_25872",
    "SHRINK1_DUAL_ACCEPTED_SET_EQUAL",
    "SHRINK1_ACCEPTED_UNIQUE_COUNT_LE_50000",
    "SHRINK1_FIRST_OUT_OF_BUDGET_WITNESS_NULL",
    "TARGET_AND_CONTROL_BINDING_MANIFESTS_COMMITTED",
    "SPLIT_COMMITMENTS_PRECEDE_HIDDEN_ACCESS",
    "DIAGNOSTIC_FORMAL_BRIDGE_DUAL_REPLAY_EQUAL",
    "ALL_REQUIRED_FORMAL_SPEC_AND_TARGET_ROOTS_NON_NULL",
    "PYTHON_COMPLETE_ENUMERATOR_IMPLEMENTED",
    "RUST_COMPLETE_ENUMERATOR_IMPLEMENTED",
    "TRAVERSAL_AND_BUCKET_ACCOUNTING_FROZEN",
    "PROGRAM_OUTPUT_AND_CHUNK_ARCHIVE_EMITTERS_VERIFIED",
    "NEW_EXECUTION_MANIFEST_ROOT_NON_NULL",
    "NEW_RUN_ID_WITH_INITIAL_STATE_NOT_RUN",
)


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def shrink1_approval_manifest() -> dict[str, object]:
    payload: dict[str, object] = {
        "artifact": "phase3_shrink1_approval_manifest_v1",
        "schema_version": "hegel-shrink1-approval-manifest/1",
        "approval_status": "USER_APPROVED",
        "approval_basis": "USER_PROVIDED_NORMATIVE_DECISION_DOCUMENT",
        "normative_document_path": str(NORMATIVE_DOCUMENT_PATH.relative_to(PROJECT_ROOT)),
        "normative_document_sha256": _sha256_file(NORMATIVE_DOCUMENT_PATH),
        "human_amendment_id": HUMAN_AMENDMENT_ID,
        "parent_dsl_version": PARENT_DSL_VERSION,
        "parent_freeze_version": PARENT_FREEZE_VERSION,
        "child_dsl_version": DSL_VERSION,
        "child_freeze_version": FREEZE_VERSION,
        "authorized_shrink_step_id": SHRINK_STEP_ID,
        "authorized_surface_delta_only": ["mean_v1", "min_v1", "max_v1"],
        "phase2b_contract_changed": False,
        "formal_roots": None,
    }
    payload["approval_manifest_id"] = stable_hash(
        payload, prefix="phase3_shrink1_approval_"
    )
    return payload


def _split_contract_payload() -> dict[str, object]:
    return {
        "target_id": ODD_REDUCTION_TARGET.target_id,
        "rank_algorithm": ODD_REDUCTION_TARGET.split_rank_algorithm,
        "quotas": [
            {
                "set_size": item.set_size,
                "discovery_train": item.discovery_train,
                "validation": item.validation,
                "sealed_prediction": item.sealed_prediction,
                "discovery_per_label": item.discovery_per_label,
                "validation_per_label": item.validation_per_label,
                "sealed_per_label": item.sealed_per_label,
            }
            for item in ODD_REDUCTION_SPLITS
        ],
        "old_validation_disposition": OLD_VALIDATION_DISPOSITION,
        "old_validation_becomes_development": False,
        "fresh_split_seed_required": False,
        "split_seed_reuse_required_if_uncompromised": True,
    }


SPLIT_CONTRACT_PAYLOAD_CONTENT_ID: Final = stable_hash(
    _split_contract_payload(), prefix="split_contract_payload_"
)


def split_binding_manifest() -> dict[str, object]:
    """Record the frozen policy and the absence of a replayable old seed."""

    payload: dict[str, object] = {
        "artifact": "phase3_shrink1_split_binding_manifest_v1",
        "schema_version": "hegel-split-binding-manifest/1",
        "payload_content_id": SPLIT_CONTRACT_PAYLOAD_CONTENT_ID,
        "parent_binding_manifest_id": None,
        "new_dsl_version": DSL_VERSION,
        "new_freeze_version": FREEZE_VERSION,
        "target_role": "DUAL_ROLE_PRECOMMITMENT",
        "target_roles": [TARGET_ROLE_OUTSIDE, TARGET_ROLE_NULL],
        "split_seed_commitment": None,
        "parent_split_seed_commitment": None,
        "realized_discovery_split_payload_content_id": None,
        "realized_validation_split_payload_content_id": None,
        "realized_sealed_prediction_split_payload_content_id": None,
        "parent_split_material_present_in_repository": False,
        "seed_reuse_policy": "REQUIRED_IF_UNCOMPROMISED",
        "fresh_seed_authorized_without_compromise": False,
        "custodian_attestation_status": "MISSING",
        "hidden_access_ledger_root": None,
        "commitment_precedes_hidden_access": False,
        "m3_gate_satisfied": False,
        "failure_mode": "FAIL_CLOSED_PENDING_EXTERNAL_CUSTODIAN_EVIDENCE",
    }
    payload["split_binding_manifest_id"] = stable_hash(
        payload, prefix="phase3_shrink1_split_binding_"
    )
    return payload


def custodian_binding_manifest() -> dict[str, object]:
    split = split_binding_manifest()
    payload: dict[str, object] = {
        "artifact": "phase3_shrink1_custodian_binding_manifest_v1",
        "schema_version": "hegel-custodian-binding-manifest/1",
        "payload_content_id": stable_hash(
            {
                "seed_continuity_contract": "REUSE_IF_UNCOMPROMISED_ELSE_NEW_TARGET_VERSION",
                "hidden_access_count_required": 0,
                "synthesis_use_count_required": 0,
                "row_allocation_recoverable_required": False,
            },
            prefix="custodian_contract_payload_",
        ),
        "parent_binding_manifest_id": None,
        "new_dsl_version": DSL_VERSION,
        "new_freeze_version": FREEZE_VERSION,
        "target_role": "DUAL_ROLE_PRECOMMITMENT",
        "split_seed_commitment": None,
        "split_binding_manifest_id": split["split_binding_manifest_id"],
        "custodian_id": None,
        "custodian_key_id": None,
        "custodian_key_epoch": None,
        "signed_seed_continuity_attestation": None,
        "parent_lifecycle_status": "UNKNOWN_NOT_ATTESTED",
        "hidden_access_event_count": None,
        "synthesis_use_count": None,
        "row_allocation_recoverable": None,
        "reuse_authorized": False,
        "m3_gate_satisfied": False,
        "failure_mode": "FAIL_CLOSED",
    }
    payload["custodian_binding_manifest_id"] = stable_hash(
        payload, prefix="phase3_shrink1_custodian_binding_"
    )
    return payload


def _target_binding_manifest(*, target_role: str) -> dict[str, object]:
    split = split_binding_manifest()
    custodian = custodian_binding_manifest()
    if target_role == TARGET_ROLE_OUTSIDE:
        payload_content_id = ODD_REDUCTION_TARGET.content_id
        universe_content_id = ODD_REDUCTION_TARGET.diagnostic_universe_content_id
        truth_content_id = ODD_REDUCTION_TARGET.diagnostic_target_table_content_id
        source_id = ODD_REDUCTION_TARGET.target_id
        prefix = "phase3_shrink1_odd_target_binding_"
    elif target_role == TARGET_ROLE_NULL:
        payload_content_id = OBSERVED_OMITTED_SINK_CONTROL.content_id
        universe_content_id = (
            OBSERVED_OMITTED_SINK_CONTROL.diagnostic_universe_content_id
        )
        truth_content_id = (
            OBSERVED_OMITTED_SINK_CONTROL.diagnostic_target_table_content_id
        )
        source_id = OBSERVED_OMITTED_SINK_CONTROL.control_id
        prefix = "phase3_shrink1_hidden_sink_binding_"
    else:
        raise ValueError("unknown target role")
    payload: dict[str, object] = {
        "artifact": "phase3_shrink1_target_dsl_binding_manifest_v1",
        "schema_version": "hegel-target-dsl-binding-manifest/1",
        "payload_content_id": payload_content_id,
        "parent_binding_manifest_id": None,
        "legacy_parent_payload_source_id": source_id,
        "new_dsl_version": DSL_VERSION,
        "new_freeze_version": FREEZE_VERSION,
        "target_role": target_role,
        "split_seed_commitment": None,
        "diagnostic_universe_content_id": universe_content_id,
        "diagnostic_truth_table_content_id": truth_content_id,
        "payload_content_identity_retained": True,
        "row_order_changed": False,
        "artificial_salt_added": False,
        "formal_bounded_universe_root": None,
        "formal_target_truth_table_root": None,
        "split_binding_manifest_id": split["split_binding_manifest_id"],
        "custodian_binding_manifest_id": custodian["custodian_binding_manifest_id"],
        "source_binding_manifest_emitted": True,
        "m3_binding_commitment_complete": False,
        "incomplete_reason": "SPLIT_SEED_AND_CUSTODIAN_ATTESTATION_MISSING",
    }
    payload["target_binding_manifest_id"] = stable_hash(payload, prefix=prefix)
    return payload


def binding_manifests_report() -> dict[str, object]:
    odd = _target_binding_manifest(target_role=TARGET_ROLE_OUTSIDE)
    sink = _target_binding_manifest(target_role=TARGET_ROLE_NULL)
    split = split_binding_manifest()
    custodian = custodian_binding_manifest()
    payload: dict[str, object] = {
        "artifact": "phase3_shrink1_binding_manifests_v1",
        "schema_version": "hegel-shrink1-binding-manifest-bundle/1",
        "dsl_version": DSL_VERSION,
        "freeze_version": FREEZE_VERSION,
        "odd_target_binding_manifest": odd,
        "hidden_sink_binding_manifest": sink,
        "split_binding_manifest": split,
        "custodian_binding_manifest": custodian,
        "old_validation_disposition": OLD_VALIDATION_DISPOSITION,
        "old_validation_still_sealed": True,
        "old_validation_becomes_development": False,
        "old_dsl_scoring_reuse_allowed": False,
        "parent_binding_manifest_absent": True,
        "retrospective_parent_manifest_fabricated": False,
        "formal_roots": None,
        "m3_commitment_gate_satisfied": False,
        "blockers": [
            "PARENT_BINDING_MANIFEST_NOT_PRESENT",
            "PARENT_SPLIT_SEED_COMMITMENT_NOT_PRESENT",
            "CUSTODIAN_SEED_CONTINUITY_ATTESTATION_NOT_PRESENT",
            "HIDDEN_ACCESS_LEDGER_ROOT_NOT_PRESENT",
        ],
    }
    payload["binding_bundle_id"] = stable_hash(
        payload, prefix="phase3_shrink1_binding_bundle_"
    )
    return payload


def formal_root_state() -> dict[str, object]:
    """Expose every publication-time slot as null without forging roots."""

    return {
        "formal_root_state": "UNGENERATED",
        "formal_roots": None,
        "formal_root_generation_allowed": False,
        "binding_roots": {
            "dsl_spec_root": None,
            "operator_semantics_root": None,
            "identifier_registry_root": None,
            "canonical_ast_schema_root": None,
            "canonical_cbor_profile_root": None,
            "diagnostic_formal_bridge_root": None,
            "outside_target_universe_root": None,
            "outside_target_truth_root": None,
            "null_control_universe_root": None,
            "null_control_truth_root": None,
        },
        "run_output_roots": {
            "canonical_program_archive_root": None,
            "program_output_archive_roots": None,
            "chunk_manifest_roots": None,
            "match_set_roots": None,
            "replay_receipt_roots": None,
        },
        "m3_execution_manifest_root": None,
        "required_root_names_from_normative_freeze": list(FORMAL_ROOT_NAMES),
    }


def _m3_gate_statuses() -> dict[str, bool]:
    statuses = {gate: False for gate in M3_REQUIRED_GATES}
    for gate in M3_REQUIRED_GATES[:14]:
        statuses[gate] = True
    return statuses


def m3_entry_contract_report() -> dict[str, object]:
    capacity = dual_shrink1_capacity_replay_report(DEFAULT_RUST_BINARY)
    gates = _m3_gate_statuses()
    if not (
        capacity["dual_replay_equal"] is True
        and capacity["python"]["accepted_unique_count"] == 25_872
        and capacity["first_out_of_budget_witness"] is None
    ):
        raise AssertionError("shrink-1 replay cannot satisfy frozen M3 subset gates")
    return {
        "m3_entry_contract_id": "hegel-m3-entry-shrink1-v1",
        "normative": {
            "shrink1_amendment_approved": True,
            "dsl_version": DSL_VERSION,
            "freeze_version": FREEZE_VERSION,
            "ast_schema_id": AST_SCHEMA_ID,
            "cbor_profile_id": CBOR_PROFILE_ID,
            "aggregate_policy": AGGREGATE_REGISTRY_POLICY,
            "removed_map_error": REMOVED_AGGREGATE_ERROR,
            "hash_compatibility_policy_frozen": True,
        },
        "dual_strict_implementation": {
            "python_updated": True,
            "rust_updated": True,
            "shared_golden_vectors_frozen": True,
            "valid_vectors_equal": True,
            "invalid_vectors_rejected_identically": True,
            "surviving_ast_hash_stability_verified": True,
            "tombstone_rejection_verified": True,
        },
        "shrink1_subset": {
            "source_count": 25_872,
            "python_rust_accepted_set_equal": True,
            "accepted_unique_count_le_50000": True,
            "first_out_of_budget_witness": None,
            "semantic_disagreement": False,
            "execution_disagreement": False,
            "interpreted_as_complete_closure": False,
        },
        "commitments": {
            "odd_target_binding_committed": False,
            "hidden_sink_binding_committed": False,
            "split_binding_committed": False,
            "custodian_binding_committed": False,
            "commitments_precede_hidden_access": False,
        },
        "formal_bridge": {
            "python_rust_equal": False,
            "dsl_spec_root_non_null": False,
            "operator_semantics_root_non_null": False,
            "identifier_registry_root_non_null": False,
            "ast_schema_root_non_null": False,
            "cbor_profile_root_non_null": False,
            "odd_universe_root_non_null": False,
            "odd_target_root_non_null": False,
            "sink_universe_root_non_null": False,
            "sink_target_root_non_null": False,
            "diagnostic_formal_bridge_root_non_null": False,
        },
        "enumeration_and_archives": {
            "python_complete_enumerator_implemented": False,
            "rust_complete_enumerator_implemented": False,
            "canonical_traversal_frozen": False,
            "bucket_accounting_frozen": False,
            "program_archive_emitter_verified": False,
            "output_archive_emitter_verified": False,
            "chunk_manifest_emitter_verified": False,
            "records_per_chunk": 4096,
        },
        "execution": {
            "execution_manifest_root_non_null": False,
            "new_run_id": False,
            "initial_state": "NOT_RUN",
            "parent_run_state_reuse": False,
        },
        "required_gate_statuses": gates,
        "satisfied_gate_count": sum(gates.values()),
        "required_gate_count": len(gates),
        "allowed_terminal_states_after_m3_start": [
            "COMPLETE",
            "DSL_TOO_LARGE",
            "INCONCLUSIVE_BUDGET",
            "INCONCLUSIVE_SEMANTICS",
            "INCONCLUSIVE_EXECUTION",
        ],
        "certificate_signatures_are_m3_entry_gate": False,
        "key_status_chain_is_m3_entry_gate": False,
        "outside_certificate_is_m3_entry_gate": False,
        "mdl_replay_is_m3_entry_gate": False,
        "invention_synthesis_is_m3_entry_gate": False,
        "active_governance_is_m3_entry_gate": False,
        "m3_entry_allowed": False,
        "child_execution_state": "NOT_RUN",
        "complete_closure_enumerated": False,
    }


def shrink1_publication_report() -> dict[str, object]:
    approval = shrink1_approval_manifest()
    gate = dual_shrink1_strict_gate_report(DEFAULT_RUST_BINARY)
    capacity = dual_shrink1_capacity_replay_report(DEFAULT_RUST_BINARY)
    bindings = binding_manifests_report()
    roots = formal_root_state()
    m3 = m3_entry_contract_report()
    payload: dict[str, object] = {
        "artifact": "phase3_shrink1_publication_v1",
        "schema_version": "hegel-phase3-shrink1-publication/1",
        "status": "SHRINK1_SUBSET_QUALIFIED_M3_BLOCKED",
        "dsl_version": DSL_VERSION,
        "freeze_version": FREEZE_VERSION,
        "human_amendment_id": HUMAN_AMENDMENT_ID,
        "approval_manifest": approval,
        "phase2b_contract_inherited_from": PARENT_FREEZE_VERSION,
        "phase2b_contract_changed": False,
        "parent_trigger_evidence": {
            "repository_commit": PARENT_TRIGGER_COMMIT,
            "gate_report_id": PARENT_GATE_REPORT_ID,
            "capacity_replay_report_id": PARENT_CAPACITY_REPLAY_REPORT_ID,
            "parent_status": "DSL_TOO_LARGE",
            "python_strict_source_root": PARENT_PYTHON_STRICT_SOURCE_ROOT,
            "python_capacity_source_root": PARENT_PYTHON_CAPACITY_SOURCE_ROOT,
            "rust_source_root": PARENT_RUST_SOURCE_ROOT,
            "accepted_set_commitment": PARENT_CAPACITY_SET_COMMITMENT,
        },
        "shrunk_dsl_surface_object": shrunk_dsl_surface_object(),
        "shrunk_dsl_surface_diagnostic_id": SHRUNK_DSL_SURFACE_DIAGNOSTIC_ID,
        "aggregate_registry_object": aggregate_registry_object(),
        "aggregate_registry_diagnostic_id": AGGREGATE_REGISTRY_DIAGNOSTIC_ID,
        "operator_admission_semantics_object": operator_admission_semantics_object(),
        "operator_admission_semantics_diagnostic_id": (
            OPERATOR_ADMISSION_SEMANTICS_DIAGNOSTIC_ID
        ),
        "strict_gate_report_id": gate["gate_report_id"],
        "capacity_replay_report_id": capacity["capacity_replay_report_id"],
        "binding_manifests": bindings,
        "formal_root_state": roots,
        "formal_roots": None,
        "m3_entry": m3,
        "child_execution_state": "NOT_RUN",
        "complete_closure_enumerated": False,
        "closure_cardinality": None,
        "target_synthesis_allowed": False,
        "hidden_sink_formal_verdict_allowed": False,
        "outside_certificate_issued": False,
        "mdl_certificate_issued": False,
        "phase2b_formal_exit": False,
        "active_promotion_allowed": False,
        "next_gate": "CUSTODIAN_CONTINUITY_AND_DUAL_FORMAL_ROOT_GENERATION",
        "claim_boundary": (
            "Shrink step 1 is implemented and its 25,872-source subset is dual "
            "verified within budget. The full closure has not run. Formal roots, "
            "custodian continuity, M3 execution, target/control verdicts, "
            "certificates, Phase-2B exit, and ACTIVE remain closed."
        ),
    }
    payload["publication_report_id"] = stable_hash(
        payload, prefix="phase3_shrink1_publication_"
    )
    return payload


def shrink_transition_report() -> dict[str, object]:
    publication = shrink1_publication_report()
    bindings = publication["binding_manifests"]
    assert isinstance(bindings, dict)
    capacity_id = publication["capacity_replay_report_id"]
    compatibility_policy = {
        "syntax_identity": "STRICT_CANONICAL_AST_CBOR_BYTES",
        "ast_hash_domain": "HEGEL/AST/V1",
        "surviving_ast_bytes_stable": True,
        "surviving_ast_hash_stable": True,
        "semantic_binding_versioned": True,
        "cross_version_archive_root_reuse_allowed": False,
    }
    payload: dict[str, object] = {
        "artifact": "phase3_dsl_shrink_transition_v1",
        "schema_version": "hegel-dsl-shrink-transition/1",
        "parent_dsl_version": PARENT_DSL_VERSION,
        "child_dsl_version": DSL_VERSION,
        "parent_freeze_version": PARENT_FREEZE_VERSION,
        "child_freeze_version": FREEZE_VERSION,
        "triggering_parent_receipt_id": PARENT_CAPACITY_REPLAY_REPORT_ID,
        "triggering_parent_evidence_kind": "BOUNDED_CAPACITY_REPLAY_REPORT",
        "parent_status": "DSL_TOO_LARGE",
        "shrink_step_id": SHRINK_STEP_ID,
        "removed_registry_entries": list(TOMBSTONED_AGGREGATE_IDS),
        "surviving_registry_entries": list(ACTIVE_AGGREGATE_IDS),
        "tombstone_policy": "PERMANENT_NO_REUSE_IN_AggregateMapId/v1",
        "hash_compatibility_policy_id": stable_hash(
            compatibility_policy, prefix="cross_dsl_hash_policy_"
        ),
        "regenerated_binding_manifest_ids": [
            bindings["odd_target_binding_manifest"]["target_binding_manifest_id"],
            bindings["hidden_sink_binding_manifest"]["target_binding_manifest_id"],
            bindings["split_binding_manifest"]["split_binding_manifest_id"],
            bindings["custodian_binding_manifest"]["custodian_binding_manifest_id"],
        ],
        "retained_payload_content_ids": [
            ODD_REDUCTION_TARGET.content_id,
            ODD_REDUCTION_TARGET.diagnostic_universe_content_id,
            ODD_REDUCTION_TARGET.diagnostic_target_table_content_id,
            OBSERVED_OMITTED_SINK_CONTROL.content_id,
            OBSERVED_OMITTED_SINK_CONTROL.diagnostic_universe_content_id,
            OBSERVED_OMITTED_SINK_CONTROL.diagnostic_target_table_content_id,
        ],
        "new_capacity_replay_id": capacity_id,
        "child_initial_state": "NOT_RUN",
        "formal_roots": None,
        "transition_is_formal_certificate": False,
    }
    payload["transition_report_id"] = stable_hash(
        payload, prefix="phase3_dsl_shrink_transition_"
    )
    return payload


__all__ = [
    "M3_REQUIRED_GATES",
    "binding_manifests_report",
    "custodian_binding_manifest",
    "formal_root_state",
    "m3_entry_contract_report",
    "shrink1_approval_manifest",
    "shrink1_publication_report",
    "shrink_transition_report",
    "split_binding_manifest",
]
