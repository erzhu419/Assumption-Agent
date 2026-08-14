"""One-lifecycle live Docker bootstrap for the Phase-2B C1 V2 authority graph.

The lifecycle, compiled image/security pins, qualification, key generation,
ownership reconciliation, deadline, cleanup, and retained-volume proof are the
frozen V1 implementation's private primitives.  This module only extends the
in-container graph construction: all six live roles sign one V2 authorization
amendment, purpose 3 durably stores both the V1 and V2 registries under distinct
IF_ABSENT keys, and purpose 4 then signs the unchanged V1 genesis token.

Success proves same-admin local-Docker and cryptographic mechanics only.  It is
not an actual campaign/run, recognizer execution, scoring, a gate, effect, or
C1 completion.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
import re
from typing import Final, NoReturn

from . import phase2b_c1_live_docker_ceremony_v1 as live_v1
from . import phase2b_c1_logical_stage_authority_v2 as authority
from . import phase2b_c1_six_role_wire_v1 as six


PHASE2B_C1_LIVE_DOCKER_CEREMONY_V2_VERSION: Final = (
    "hegel-machine-phase2b-c1-live-docker-ceremony/2"
)
PHASE2B_C1_LIVE_DOCKER_CEREMONY_V2_CLAIM_LEVEL: Final = (
    "OWNER_ACCEPTED_SAME_ADMIN_LOCAL_DOCKER_V2_AUTHORITY_BOOTSTRAP_MECHANICS_ONLY"
)
PHASE2B_C1_LIVE_DOCKER_CEREMONY_V2_COMPLETE_REASON: Final = (
    "phase2b_c1_owner_accepted_local_docker_v2_authority_bootstrap_complete_not_actual_960"
)

FAIL_AUTHORITY: Final = "FAIL_PHASE2B_C1_LIVE_CEREMONY_V2_AUTHORITY"
FAIL_OPERATION_BINDING: Final = (
    "FAIL_PHASE2B_C1_LIVE_CEREMONY_V2_OPERATION_BINDING"
)

_RECEIPT_PREFIX: Final = "phase2b_c1_live_docker_ceremony_receipt_v2_"
_RECEIPT_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/LIVE_DOCKER_CEREMONY_RECEIPT/V2\x00"
)
_SIGNATURE_PATTERN: Final = re.compile(r"[0-9a-f]{128}")


def _fail(code: str, detail: str) -> NoReturn:
    raise live_v1.Phase2BC1LiveDockerCeremonyErrorV1(code, detail)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


_TRUE_RESULT_CLAIMS: Final = (
    "profile_v2_preflight_verified",
    "docker_invoked",
    "six_containers_simultaneously_live_verified",
    "actor_qualification_executed",
    "technical_role_independence_live_verified",
    "fresh_keys_generated",
    "v1_registry_signature_generated",
    "six_v2_authorization_signatures_generated_and_verified",
    "v1_registry_store_if_absent_zero_to_one_executed",
    "v2_registry_store_if_absent_zero_to_one_executed",
    "durable_store_file_and_directory_fsync_verified",
    "unchanged_v1_logical_genesis_signed_and_persisted",
    "v1_six_role_wire_graph_verified",
    "v2_authority_graph_verified",
    "exact_twenty_two_operation_bindings_verified",
    "containers_removed",
    "container_absence_proved",
    "purpose_private_volumes_retained",
)
_FALSE_RESULT_CLAIMS: Final = (
    "external_organizational_authority_verified",
    "independent_human_actors_verified",
    "malicious_host_root_resistance",
    "malicious_docker_daemon_resistance",
    "hostile_administrator_tamper_resistance_verified",
    "owner_deletion_resistance_verified",
    "owner_rollback_resistance_verified",
    "cas_persistence_executed",
    "crash_recovery_verified",
    "cross_host_global_one_shot_verified",
    "authoritative_utc_time_verified",
    "runtime_attestation_authority_verified",
    "actual_campaign_eligible",
    "actual_execution_claimed",
    "execution_authorized",
    "actual_960_case_run_verified",
    "recognizer_executed",
    "recognizer_operation_signature_verified",
    "scoring_performed",
    "prediction_scored",
    "actual_prediction_scoring_evidence",
    "formal_gate_evaluation_performed",
    "effect_evidence",
    "formal_c1_report_verified",
    "c1_exit_evidence",
)
_RECEIPT_TOKEN: Final = object()


@dataclass(frozen=True, slots=True, init=False)
class Phase2BC1LiveDockerCeremonyReceiptV2:
    ceremony_receipt_id: str
    version: str
    claim_level: str
    reason: str
    request: live_v1.Phase2BC1LiveDockerCeremonyRequestV1
    legacy_context: six.Phase2BC1CeremonyContextV1
    legacy_registry: six.Phase2BC1ExecutionRegistryCandidateV1
    legacy_store_receipt: six.Phase2BC1StoreIfAbsentReceiptCandidateV1
    legacy_logical_genesis_token: six.Phase2BC1LogicalGenesisTokenCandidateV1
    legacy_wire_verification: six.Phase2BC1SixRoleWireVerificationV1
    authority_context: authority.Phase2BC1LogicalStageAuthorityContextV2
    authority_registry: authority.Phase2BC1OwnerAcceptedExecutionRegistryV2
    authority_store_receipt: authority.Phase2BC1LogicalStageAuthorizationStoreReceiptCandidateV2
    authority_verification: authority.Phase2BC1LogicalStageAuthorityVerificationV2
    role_runtime_bindings: tuple[live_v1.Phase2BC1LiveRoleRuntimeBindingV1, ...]
    operation_bindings: tuple[live_v1.Phase2BC1LiveOperationBindingV1, ...]
    docker_control_plane_binding_sha256: str
    recognizer_image_inspect_sha256: str
    scorer_image_inspect_sha256: str
    openssl_sha256: str
    openssl_version: str
    stored_v1_registry_sha256: str
    stored_v2_registry_sha256: str
    logical_genesis_record_sha256: str
    profile_v2_preflight_verified: bool
    docker_invoked: bool
    six_containers_simultaneously_live_verified: bool
    actor_qualification_executed: bool
    technical_role_independence_live_verified: bool
    fresh_keys_generated: bool
    v1_registry_signature_generated: bool
    six_v2_authorization_signatures_generated_and_verified: bool
    v1_registry_store_if_absent_zero_to_one_executed: bool
    v2_registry_store_if_absent_zero_to_one_executed: bool
    durable_store_file_and_directory_fsync_verified: bool
    unchanged_v1_logical_genesis_signed_and_persisted: bool
    v1_six_role_wire_graph_verified: bool
    v2_authority_graph_verified: bool
    exact_twenty_two_operation_bindings_verified: bool
    containers_removed: bool
    container_absence_proved: bool
    purpose_private_volumes_retained: bool
    external_organizational_authority_verified: bool
    independent_human_actors_verified: bool
    malicious_host_root_resistance: bool
    malicious_docker_daemon_resistance: bool
    hostile_administrator_tamper_resistance_verified: bool
    owner_deletion_resistance_verified: bool
    owner_rollback_resistance_verified: bool
    cas_persistence_executed: bool
    crash_recovery_verified: bool
    cross_host_global_one_shot_verified: bool
    authoritative_utc_time_verified: bool
    runtime_attestation_authority_verified: bool
    actual_campaign_eligible: bool
    actual_execution_claimed: bool
    execution_authorized: bool
    actual_960_case_run_verified: bool
    recognizer_executed: bool
    recognizer_operation_signature_verified: bool
    scoring_performed: bool
    prediction_scored: bool
    actual_prediction_scoring_evidence: bool
    formal_gate_evaluation_performed: bool
    effect_evidence: bool
    formal_c1_report_verified: bool
    c1_exit_evidence: bool

    def __new__(cls, token: object = None) -> "Phase2BC1LiveDockerCeremonyReceiptV2":
        if token is not _RECEIPT_TOKEN:
            raise TypeError("live V2 ceremony receipts are privately issued")
        return object.__new__(cls)

    def __init__(self, token: object = None) -> None:
        if token is not _RECEIPT_TOKEN:
            raise TypeError("live V2 ceremony receipts are privately issued")

    def __post_init__(self) -> None:
        purpose_order = tuple(
            row.purpose_id for row in self.role_runtime_bindings
        ) if type(self.role_runtime_bindings) is tuple else ()
        container_names = tuple(
            row.container_name for row in self.role_runtime_bindings
        ) if type(self.role_runtime_bindings) is tuple else ()
        container_ids = tuple(
            row.container_id for row in self.role_runtime_bindings
        ) if type(self.role_runtime_bindings) is tuple else ()
        volume_names = tuple(
            name
            for row in self.role_runtime_bindings
            for _, name in row.purpose_private_volumes
        ) if type(self.role_runtime_bindings) is tuple else ()
        if (
            type(self.request) is not live_v1.Phase2BC1LiveDockerCeremonyRequestV1
            or type(self.legacy_context) is not six.Phase2BC1CeremonyContextV1
            or type(self.authority_context) is not authority.Phase2BC1LogicalStageAuthorityContextV2
            or type(self.role_runtime_bindings) is not tuple
            or len(self.role_runtime_bindings) != 6
            or any(
                type(row) is not live_v1.Phase2BC1LiveRoleRuntimeBindingV1
                for row in self.role_runtime_bindings
            )
            or purpose_order != (1, 2, 3, 4, 5, 6)
            or len(set(container_names)) != 6
            or len(set(container_ids)) != 6
            or len(volume_names) != 11
            or len(set(volume_names)) != 11
            or type(self.operation_bindings) is not tuple
            or len(self.operation_bindings) != 22
            or self.version != PHASE2B_C1_LIVE_DOCKER_CEREMONY_V2_VERSION
            or self.claim_level != PHASE2B_C1_LIVE_DOCKER_CEREMONY_V2_CLAIM_LEVEL
            or self.reason != PHASE2B_C1_LIVE_DOCKER_CEREMONY_V2_COMPLETE_REASON
            or not self.ceremony_receipt_id.startswith(_RECEIPT_PREFIX)
            or self.openssl_sha256 != live_v1.OPENSSL_SHA256_V1
            or self.openssl_version != live_v1.OPENSSL_VERSION_V1
        ):
            raise ValueError("live V2 ceremony receipt identity/shape drift")
        live_v1._require_sha(
            self.ceremony_receipt_id[len(_RECEIPT_PREFIX):], "receipt suffix"
        )
        for name in (
            "docker_control_plane_binding_sha256", "recognizer_image_inspect_sha256",
            "scorer_image_inspect_sha256", "stored_v1_registry_sha256",
            "stored_v2_registry_sha256", "logical_genesis_record_sha256",
        ):
            live_v1._require_sha(getattr(self, name), name)
        graph = (
            self.legacy_context, self.legacy_registry, self.legacy_store_receipt,
            self.legacy_logical_genesis_token, self.legacy_wire_verification,
            self.authority_context, self.authority_registry,
            self.authority_store_receipt, self.authority_verification,
            self.stored_v1_registry_sha256, self.stored_v2_registry_sha256,
            self.logical_genesis_record_sha256,
        )
        expected_receipt_id = _receipt_id(
            request=self.request, graph=graph,
            runtime_bindings=self.role_runtime_bindings,
            operation_bindings=self.operation_bindings,
            control_sha=self.docker_control_plane_binding_sha256,
            recognizer_inspect_sha=self.recognizer_image_inspect_sha256,
            scorer_inspect_sha=self.scorer_image_inspect_sha256,
        )
        if self.ceremony_receipt_id != expected_receipt_id:
            raise ValueError("live V2 ceremony receipt content address drift")
        if any(getattr(self, name) is not True for name in _TRUE_RESULT_CLAIMS):
            raise ValueError("live V2 ceremony true-claim drift")
        if any(getattr(self, name) is not False for name in _FALSE_RESULT_CLAIMS):
            raise ValueError("live V2 ceremony false-claim drift")


def _actor_signature(
    control: object,
    root: Path,
    role: live_v1._RoleStateV1,
    *,
    operation_name: str,
    preimage: bytes,
    operations: list[live_v1.Phase2BC1LiveOperationBindingV1],
) -> bytes:
    output = live_v1._record_actor_exec_v1(
        control, root, role, operation_name=operation_name,
        worker_operation="sign", stdin_payload=preimage, bindings=operations,
        failure_code=live_v1.FAIL_SIGNATURE,
    )
    if (
        set(output) != {
            "operation", "purpose_id", "role", "preimage_sha256", "signature_hex"
        }
        or output.get("operation") != "SIGNED"
        or output.get("purpose_id") != role.purpose_id
        or output.get("role") != role.role.value
        or output.get("preimage_sha256") != _sha256(preimage)
        or type(output.get("signature_hex")) is not str
        or _SIGNATURE_PATTERN.fullmatch(output["signature_hex"]) is None
    ):
        _fail(live_v1.FAIL_SIGNATURE, f"{operation_name} output drift")
    return bytes.fromhex(output["signature_hex"])


def _store_and_sign(
    control: object,
    root: Path,
    role: live_v1._RoleStateV1,
    *,
    operation_name: str,
    object_key: str,
    content: bytes,
    preimage: bytes,
    operations: list[live_v1.Phase2BC1LiveOperationBindingV1],
) -> bytes:
    content_sha = _sha256(content)
    payload = _canonical_json_bytes({
        "content_b64": base64.b64encode(content).decode("ascii"),
        "object_key": object_key,
        "preimage_b64": base64.b64encode(preimage).decode("ascii"),
        "stored_content_sha256": content_sha,
    })
    output = live_v1._record_actor_exec_v1(
        control, root, role, operation_name=operation_name,
        worker_operation="store-and-sign", stdin_payload=payload,
        bindings=operations, failure_code=live_v1.FAIL_STORE,
    )
    record_name = (
        "registry-" + _sha256(object_key.encode("ascii"))
        + ".generation-0000000001"
    )
    if (
        set(output) != {
            "operation", "purpose_id", "role", "condition",
            "previous_generation", "new_generation", "stored_content_sha256",
            "record_name_sha256", "preimage_sha256", "signature_hex",
            "file_fsync", "directory_fsync", "hardlink_no_replace",
        }
        or output.get("operation") != "STORE_IF_ABSENT_SIGNED"
        or output.get("purpose_id") != 3
        or output.get("role") != six.Phase2BC1SixRoleV1.STORE_RECEIPT.value
        or output.get("condition") != "IF_ABSENT"
        or output.get("previous_generation") != 0
        or output.get("new_generation") != 1
        or output.get("stored_content_sha256") != content_sha
        or output.get("record_name_sha256") != _sha256(record_name.encode("ascii"))
        or output.get("preimage_sha256") != _sha256(preimage)
        or type(output.get("signature_hex")) is not str
        or _SIGNATURE_PATTERN.fullmatch(output["signature_hex"]) is None
        or any(output.get(name) is not True for name in (
            "file_fsync", "directory_fsync", "hardlink_no_replace"
        ))
    ):
        _fail(live_v1.FAIL_STORE, f"{operation_name} durable output drift")
    return bytes.fromhex(output["signature_hex"])


def _build_live_graph_v2(
    control: object,
    root: Path,
    request: live_v1.Phase2BC1LiveDockerCeremonyRequestV1,
    roles: tuple[live_v1._RoleStateV1, ...],
    keys: tuple[six.Phase2BC1RoleKeyV1, ...],
    operations: list[live_v1.Phase2BC1LiveOperationBindingV1],
) -> tuple[object, ...]:
    legacy_context = live_v1._build_context_v1(request)
    old = legacy_context.body
    legacy_registry_body = six.Phase2BC1ExecutionRegistryBodyV1(
        version=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_VERSION,
        context_id=legacy_context.context_id,
        authority_class=old.authority_class,
        authority_requirement_id=old.authority_requirement_id,
        actor_profile_content_id=old.actor_profile_content_id,
        actor_profile_canonical_sha256=old.actor_profile_canonical_sha256,
        campaign_id=old.campaign_id, protocol_id=old.protocol_id,
        exact_freeze_id=old.exact_freeze_id,
        execution_freeze_manifest_id=old.execution_freeze_manifest_id,
        execution_freeze_manifest_sha256=old.execution_freeze_manifest_sha256,
        attempt_nonce_sha256=old.attempt_nonce_sha256, attempt_id=old.attempt_id,
        recognizer_image_digest=old.recognizer_image_digest,
        recognizer_source_root_sha256=old.recognizer_source_root_sha256,
        scorer_image_digest=old.scorer_image_digest,
        scorer_source_root_sha256=old.scorer_source_root_sha256,
        role_keys=keys,
    )
    governance_preimage = six.phase2b_c1_governance_registry_signature_preimage_v1(
        legacy_registry_body, signer_key_id=keys[0].key_id
    )
    governance_signature = six.Phase2BC1DetachedSignatureV1(
        purpose_id=1, role=six.Phase2BC1SixRoleV1.GOVERNANCE_ANCHOR,
        signature_purpose=six.Phase2BC1SignaturePurposeV1.GOVERNANCE_CHECKPOINT,
        algorithm="Ed25519", key_epoch=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_KEY_EPOCH,
        signer_key_id=keys[0].key_id,
        signature=_actor_signature(
            control, root, roles[0], operation_name="GOVERNANCE_SIGN",
            preimage=governance_preimage, operations=operations,
        ),
    )
    legacy_registry = six.Phase2BC1ExecutionRegistryCandidateV1(
        body=legacy_registry_body, governance_signature=governance_signature,
        execution_registry_id=six.phase2b_c1_execution_registry_id_v1(
            legacy_registry_body, governance_signature
        ),
    )

    authority_context_body = authority.Phase2BC1LogicalStageAuthorityContextBodyV2(
        version=authority.PHASE2B_C1_LOGICAL_STAGE_AUTHORITY_V2_VERSION,
        authority_class=old.authority_class,
        authority_requirement_id=old.authority_requirement_id,
        actor_profile_content_id=old.actor_profile_content_id,
        actor_profile_canonical_sha256=old.actor_profile_canonical_sha256,
        campaign_id=old.campaign_id, protocol_id=old.protocol_id,
        exact_freeze_id=old.exact_freeze_id,
        execution_freeze_manifest_id=old.execution_freeze_manifest_id,
        execution_freeze_manifest_sha256=old.execution_freeze_manifest_sha256,
        attempt_nonce_sha256=old.attempt_nonce_sha256, attempt_id=old.attempt_id,
        recognizer_image_digest=old.recognizer_image_digest,
        recognizer_source_root_sha256=old.recognizer_source_root_sha256,
        scorer_image_digest=old.scorer_image_digest,
        scorer_source_root_sha256=old.scorer_source_root_sha256,
        legacy_context_id=legacy_context.context_id,
        legacy_execution_registry_id=legacy_registry.execution_registry_id,
        logical_stage_wire_schema_id=authority.PHASE2B_C1_LOGICAL_STAGE_WIRE_SCHEMA_ID_V1,
        logical_stage_wire_policy_id=authority.PHASE2B_C1_LOGICAL_STAGE_WIRE_POLICY_ID_V2,
        stage_event_contract_manifest_id=authority.PHASE2B_C1_STAGE_EVENT_CONTRACT_MANIFEST_ID_V1,
    )
    authority_context = authority.Phase2BC1LogicalStageAuthorityContextV2(
        body=authority_context_body,
        authority_context_id=authority.phase2b_c1_logical_stage_authority_context_id_v2(
            authority_context_body
        ),
    )
    amendment = authority.Phase2BC1LogicalStageAuthorizationAmendmentBodyV2(
        version=authority.PHASE2B_C1_LOGICAL_STAGE_AUTHORITY_V2_VERSION,
        authority_context_id=authority_context.authority_context_id,
        legacy_context_id=legacy_context.context_id,
        legacy_execution_registry_id=legacy_registry.execution_registry_id,
        authority_class=old.authority_class,
        authority_requirement_id=old.authority_requirement_id,
        actor_profile_content_id=old.actor_profile_content_id,
        actor_profile_canonical_sha256=old.actor_profile_canonical_sha256,
        campaign_id=old.campaign_id, attempt_id=old.attempt_id,
        logical_stage_wire_schema_id=authority.PHASE2B_C1_LOGICAL_STAGE_WIRE_SCHEMA_ID_V1,
        logical_stage_wire_policy_id=authority.PHASE2B_C1_LOGICAL_STAGE_WIRE_POLICY_ID_V2,
        stage_event_contract_manifest_id=authority.PHASE2B_C1_STAGE_EVENT_CONTRACT_MANIFEST_ID_V1,
        authorized_stage_order=(
            six.Phase2BC1LogicalStageV1.RAW_INPUT_COMMITTED,
            six.Phase2BC1LogicalStageV1.PACKAGE_COMMITMENTS_FROZEN,
            six.Phase2BC1LogicalStageV1.RUN_STARTED,
            six.Phase2BC1LogicalStageV1.RUN_FINISHED,
        ),
        authorized_stage_indices=(1, 2, 3, 4),
        required_signer_roles=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_ROLE_ORDER,
        required_signer_key_ids=tuple(key.key_id for key in keys),
        authorization_semantics=authority.PHASE2B_C1_LOGICAL_STAGE_AUTHORIZATION_SEMANTICS_V2,
        legacy_token_authorization_field_must_remain_unchanged=True,
        legacy_actual_campaign_eligible_must_remain_false=True,
        actual_execution_claimed=False,
    )
    amendment_id = authority.phase2b_c1_logical_stage_authorization_amendment_id_v2(
        amendment
    )
    authorization_signatures = tuple(
        authority.Phase2BC1LogicalStageAuthorizationSignatureV2(
            purpose_id=key.purpose_id, role=key.role,
            signature_purpose=authority.PHASE2B_C1_LOGICAL_STAGE_AUTHORIZATION_SIGNATURE_PURPOSE_V2,
            algorithm="Ed25519", key_epoch=key.key_epoch,
            signer_key_id=key.key_id,
            signature=_actor_signature(
                control, root, role,
                operation_name="LOGICAL_STAGE_AUTHORIZATION_SIGN",
                preimage=authority.phase2b_c1_logical_stage_authorization_signature_preimage_v2(
                    amendment, purpose_id=key.purpose_id, role=key.role,
                    signer_key_id=key.key_id,
                ),
                operations=operations,
            ),
        )
        for role, key in zip(roles, keys, strict=True)
    )
    authority_registry = authority.Phase2BC1OwnerAcceptedExecutionRegistryV2(
        body=amendment, authorization_amendment_id=amendment_id,
        role_acceptance_signatures=authorization_signatures,
        execution_registry_id=authority.phase2b_c1_owner_accepted_execution_registry_id_v2(
            amendment, amendment_id, authorization_signatures
        ),
    )

    legacy_registry_bytes = six.canonical_phase2b_c1_execution_registry_bytes_v1(
        legacy_registry
    )
    legacy_registry_sha = _sha256(legacy_registry_bytes)
    legacy_object_key = (
        "phase2b/c1/actual-unsealed-960/"
        f"{request.campaign_id}/attempt-0/execution-registry"
    )
    legacy_store_body = six.Phase2BC1StoreIfAbsentReceiptBodyV1(
        version=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_VERSION,
        context_id=legacy_context.context_id,
        authority_requirement_id=old.authority_requirement_id,
        actor_profile_content_id=old.actor_profile_content_id,
        campaign_id=old.campaign_id, attempt_id=old.attempt_id,
        execution_registry_id=legacy_registry.execution_registry_id,
        object_key=legacy_object_key, previous_generation=0, new_generation=1,
        condition="IF_ABSENT", previous_content_sha256_or_null=None,
        stored_content_sha256=legacy_registry_sha,
    )
    legacy_store_preimage = six.phase2b_c1_store_receipt_signature_preimage_v1(
        legacy_store_body, signer_key_id=keys[2].key_id
    )
    legacy_store_signature = six.Phase2BC1DetachedSignatureV1(
        purpose_id=3, role=six.Phase2BC1SixRoleV1.STORE_RECEIPT,
        signature_purpose=six.Phase2BC1SignaturePurposeV1.STORE_IF_ABSENT_RECEIPT,
        algorithm="Ed25519", key_epoch=keys[2].key_epoch,
        signer_key_id=keys[2].key_id,
        signature=_store_and_sign(
            control, root, roles[2], operation_name="STORE_IF_ABSENT_AND_SIGN",
            object_key=legacy_object_key, content=legacy_registry_bytes,
            preimage=legacy_store_preimage, operations=operations,
        ),
    )
    legacy_store = six.Phase2BC1StoreIfAbsentReceiptCandidateV1(
        body=legacy_store_body, store_signature=legacy_store_signature,
        store_receipt_id=six.phase2b_c1_store_receipt_id_v1(
            legacy_store_body, legacy_store_signature
        ),
    )

    authority_registry_bytes = (
        authority.canonical_phase2b_c1_owner_accepted_execution_registry_bytes_v2(
            authority_registry
        )
    )
    authority_registry_sha = _sha256(authority_registry_bytes)
    authority_object_key = (
        "phase2b/c1/actual-unsealed-960/"
        f"{request.campaign_id}/attempt-0/logical-stage-authorization-registry-v2"
    )
    authority_store_body = authority.Phase2BC1LogicalStageAuthorizationStoreReceiptBodyV2(
        version=authority.PHASE2B_C1_LOGICAL_STAGE_AUTHORITY_V2_VERSION,
        authority_context_id=authority_context.authority_context_id,
        legacy_context_id=legacy_context.context_id,
        campaign_id=old.campaign_id, attempt_id=old.attempt_id,
        execution_registry_id=authority_registry.execution_registry_id,
        object_key=authority_object_key, previous_generation=0, new_generation=1,
        condition="IF_ABSENT", previous_content_sha256_or_null=None,
        stored_content_sha256=authority_registry_sha,
    )
    authority_store_preimage = (
        authority.phase2b_c1_logical_stage_authorization_store_signature_preimage_v2(
            authority_store_body, signer_key_id=keys[2].key_id
        )
    )
    authority_store_signature = six.Phase2BC1DetachedSignatureV1(
        purpose_id=3, role=six.Phase2BC1SixRoleV1.STORE_RECEIPT,
        signature_purpose=six.Phase2BC1SignaturePurposeV1.STORE_IF_ABSENT_RECEIPT,
        algorithm="Ed25519", key_epoch=keys[2].key_epoch,
        signer_key_id=keys[2].key_id,
        signature=_store_and_sign(
            control, root, roles[2],
            operation_name="LOGICAL_STAGE_AUTHORIZATION_STORE_IF_ABSENT_AND_SIGN",
            object_key=authority_object_key, content=authority_registry_bytes,
            preimage=authority_store_preimage, operations=operations,
        ),
    )
    authority_store = authority.Phase2BC1LogicalStageAuthorizationStoreReceiptCandidateV2(
        body=authority_store_body, store_signature=authority_store_signature,
        store_receipt_id=authority.phase2b_c1_logical_stage_authorization_store_receipt_id_v2(
            authority_store_body, authority_store_signature
        ),
    )

    genesis_body = six.Phase2BC1LogicalGenesisTokenBodyV1(
        version=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_VERSION,
        context_id=legacy_context.context_id,
        authority_requirement_id=old.authority_requirement_id,
        actor_profile_content_id=old.actor_profile_content_id,
        campaign_id=old.campaign_id, attempt_id=old.attempt_id,
        execution_registry_id=legacy_registry.execution_registry_id,
        store_receipt_id=legacy_store.store_receipt_id,
        time_semantics=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_TIME_SEMANTICS,
        previous_logical_sequence_or_null=None, logical_sequence=0,
        stage=six.Phase2BC1LogicalStageV1.PREREGISTERED, stage_index=0,
    )
    genesis_preimage = six.phase2b_c1_logical_genesis_signature_preimage_v1(
        genesis_body, signer_key_id=keys[3].key_id
    )
    genesis_payload = _canonical_json_bytes({
        "logical_sequence": 0,
        "preimage_b64": base64.b64encode(genesis_preimage).decode("ascii"),
        "previous_logical_sequence_or_null": None,
    })
    genesis_output = live_v1._record_actor_exec_v1(
        control, root, roles[3], operation_name="TIME_GENESIS_AND_SIGN",
        worker_operation="time-genesis-and-sign", stdin_payload=genesis_payload,
        bindings=operations, failure_code=live_v1.FAIL_TIME,
    )
    expected_record = _canonical_json_bytes({
        "logical_sequence": 0, "preimage_sha256": _sha256(genesis_preimage),
        "previous_logical_sequence_or_null": None,
    }) + b"\n"
    if (
        set(genesis_output) != {
            "operation", "purpose_id", "role", "previous_logical_sequence_or_null",
            "logical_sequence", "record_sha256", "preimage_sha256", "signature_hex",
            "file_fsync", "directory_fsync", "hardlink_no_replace",
        }
        or genesis_output.get("operation") != "TIME_GENESIS_SIGNED"
        or genesis_output.get("purpose_id") != 4
        or genesis_output.get("role") != six.Phase2BC1SixRoleV1.TIME_ATTESTOR.value
        or genesis_output.get("previous_logical_sequence_or_null") is not None
        or genesis_output.get("logical_sequence") != 0
        or genesis_output.get("preimage_sha256") != _sha256(genesis_preimage)
        or genesis_output.get("record_sha256") != _sha256(expected_record)
        or type(genesis_output.get("signature_hex")) is not str
        or _SIGNATURE_PATTERN.fullmatch(genesis_output["signature_hex"]) is None
        or any(genesis_output.get(name) is not True for name in (
            "file_fsync", "directory_fsync", "hardlink_no_replace"
        ))
    ):
        _fail(live_v1.FAIL_TIME, "TIME durable unchanged V1 genesis output drift")
    genesis_signature = six.Phase2BC1DetachedSignatureV1(
        purpose_id=4, role=six.Phase2BC1SixRoleV1.TIME_ATTESTOR,
        signature_purpose=six.Phase2BC1SignaturePurposeV1.DECLARED_TIME_TOKEN,
        algorithm="Ed25519", key_epoch=keys[3].key_epoch,
        signer_key_id=keys[3].key_id,
        signature=bytes.fromhex(genesis_output["signature_hex"]),
    )
    genesis = six.Phase2BC1LogicalGenesisTokenCandidateV1(
        body=genesis_body, time_signature=genesis_signature,
        logical_genesis_token_id=six.phase2b_c1_logical_genesis_token_id_v1(
            genesis_body, genesis_signature
        ),
    )

    legacy_wire = six.verify_phase2b_c1_six_role_wire_v1(
        context=legacy_context, registry=legacy_registry,
        store_receipt=legacy_store, logical_genesis_token=genesis,
    )
    if (
        legacy_wire.disposition
        is not six.Phase2BC1SixRoleWireDispositionV1.CALLER_SUPPLIED_SIX_ROLE_WIRE_GRAPH_VERIFIED_NOT_AUTHORITY
        or legacy_wire.result_id is None
        or legacy_wire.partial_output_published is not False
        or legacy_wire.actual_execution_claimed is not False
        or legacy_wire.execution_authorized is not False
    ):
        _fail(live_v1.FAIL_WIRE, "V1 six-role verifier rejected live V2 graph base")
    authority_verification = authority.verify_phase2b_c1_logical_stage_authority_graph_v2(
        legacy_context=legacy_context, legacy_registry=legacy_registry,
        legacy_store_receipt=legacy_store, legacy_logical_genesis_token=genesis,
        authority_context=authority_context, execution_registry=authority_registry,
        authorization_store_receipt=authority_store,
    )
    if (
        authority_verification.disposition
        is not authority.Phase2BC1LogicalStageAuthorityDispositionV2.CALLER_SUPPLIED_OWNER_ACCEPTED_V2_AUTHORITY_GRAPH_VERIFIED_NOT_EXECUTION
        or authority_verification.result_id is None
        or authority_verification.actual_campaign_eligible is not False
        or authority_verification.actual_execution_claimed is not False
        or authority_verification.execution_authorized is not False
    ):
        _fail(FAIL_AUTHORITY, "V2 authority verifier rejected live graph")
    return (
        legacy_context, legacy_registry, legacy_store, genesis, legacy_wire,
        authority_context, authority_registry, authority_store,
        authority_verification, legacy_registry_sha, authority_registry_sha,
        genesis_output["record_sha256"],
    )


def _expected_operations() -> tuple[tuple[int, six.Phase2BC1SixRoleV1, str], ...]:
    return (
        *((index, role, "QUALIFY") for index, role in enumerate(
            six.PHASE2B_C1_SIX_ROLE_WIRE_V1_ROLE_ORDER, 1
        )),
        *((index, role, "KEYGEN") for index, role in enumerate(
            six.PHASE2B_C1_SIX_ROLE_WIRE_V1_ROLE_ORDER, 1
        )),
        (1, six.Phase2BC1SixRoleV1.GOVERNANCE_ANCHOR, "GOVERNANCE_SIGN"),
        *((index, role, "LOGICAL_STAGE_AUTHORIZATION_SIGN")
          for index, role in enumerate(six.PHASE2B_C1_SIX_ROLE_WIRE_V1_ROLE_ORDER, 1)),
        (3, six.Phase2BC1SixRoleV1.STORE_RECEIPT, "STORE_IF_ABSENT_AND_SIGN"),
        (3, six.Phase2BC1SixRoleV1.STORE_RECEIPT,
         "LOGICAL_STAGE_AUTHORIZATION_STORE_IF_ABSENT_AND_SIGN"),
        (4, six.Phase2BC1SixRoleV1.TIME_ATTESTOR, "TIME_GENESIS_AND_SIGN"),
    )


def _runtime_binding_projection(
    bindings: tuple[live_v1.Phase2BC1LiveRoleRuntimeBindingV1, ...],
) -> tuple[tuple[tuple[str, object], ...], ...]:
    return tuple((
        ("purpose_id", row.purpose_id),
        ("role", row.role.value),
        ("image_digest", row.image_digest),
        ("container_name", row.container_name),
        ("container_id", row.container_id),
        ("purpose_private_volumes", row.purpose_private_volumes),
        ("qualification_sha256", row.qualification_sha256),
        ("namespace_identities", row.namespace_identities),
        ("openssl_sha256", row.openssl_sha256),
        ("openssl_version", row.openssl_version),
    ) for row in bindings)


def _operation_binding_projection(
    bindings: tuple[live_v1.Phase2BC1LiveOperationBindingV1, ...],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (row.purpose_id, row.role.value, row.operation, row.command_sha256,
         row.stdin_sha256, row.stdout_sha256, row.stderr_sha256,
         row.started_monotonic_ns, row.finished_monotonic_ns)
        for row in bindings
    )


def _receipt_identity(
    *, request: live_v1.Phase2BC1LiveDockerCeremonyRequestV1,
    graph: tuple[object, ...],
    runtime_bindings: tuple[live_v1.Phase2BC1LiveRoleRuntimeBindingV1, ...],
    operation_bindings: tuple[live_v1.Phase2BC1LiveOperationBindingV1, ...],
    control_sha: str, recognizer_inspect_sha: str, scorer_inspect_sha: str,
) -> dict[str, object]:
    (
        legacy_context, legacy_registry, legacy_store, genesis, legacy_wire,
        authority_context, authority_registry, authority_store,
        authority_verification, legacy_registry_sha, authority_registry_sha,
        genesis_record_sha,
    ) = graph
    return {
        "version": PHASE2B_C1_LIVE_DOCKER_CEREMONY_V2_VERSION,
        "campaign_id": request.campaign_id,
        "attempt_id": request.attempt_id,
        "legacy_context_id": legacy_context.context_id,
        "legacy_execution_registry_id": legacy_registry.execution_registry_id,
        "authority_context_id": authority_context.authority_context_id,
        "authority_execution_registry_id": authority_registry.execution_registry_id,
        "legacy_store_receipt_id": legacy_store.store_receipt_id,
        "authority_store_receipt_id": authority_store.store_receipt_id,
        "logical_genesis_token_id": genesis.logical_genesis_token_id,
        "legacy_wire_result_id": legacy_wire.result_id,
        "authority_result_id": authority_verification.result_id,
        "runtime_bindings": _runtime_binding_projection(runtime_bindings),
        "operation_bindings": _operation_binding_projection(operation_bindings),
        "control_sha256": control_sha,
        "recognizer_inspect_sha256": recognizer_inspect_sha,
        "scorer_inspect_sha256": scorer_inspect_sha,
        "stored_v1_registry_sha256": legacy_registry_sha,
        "stored_v2_registry_sha256": authority_registry_sha,
        "logical_genesis_record_sha256": genesis_record_sha,
    }


def _receipt_id(
    *, request: live_v1.Phase2BC1LiveDockerCeremonyRequestV1,
    graph: tuple[object, ...],
    runtime_bindings: tuple[live_v1.Phase2BC1LiveRoleRuntimeBindingV1, ...],
    operation_bindings: tuple[live_v1.Phase2BC1LiveOperationBindingV1, ...],
    control_sha: str, recognizer_inspect_sha: str, scorer_inspect_sha: str,
) -> str:
    identity = _receipt_identity(
        request=request, graph=graph, runtime_bindings=runtime_bindings,
        operation_bindings=operation_bindings, control_sha=control_sha,
        recognizer_inspect_sha=recognizer_inspect_sha,
        scorer_inspect_sha=scorer_inspect_sha,
    )
    return _RECEIPT_PREFIX + _sha256(
        _RECEIPT_DOMAIN + _canonical_json_bytes(identity)
    )


def _issue_receipt(
    *, request: live_v1.Phase2BC1LiveDockerCeremonyRequestV1,
    graph: tuple[object, ...],
    runtime_bindings: tuple[live_v1.Phase2BC1LiveRoleRuntimeBindingV1, ...],
    operation_bindings: tuple[live_v1.Phase2BC1LiveOperationBindingV1, ...],
    control_sha: str, recognizer_inspect_sha: str, scorer_inspect_sha: str,
) -> Phase2BC1LiveDockerCeremonyReceiptV2:
    (
        legacy_context, legacy_registry, legacy_store, genesis, legacy_wire,
        authority_context, authority_registry, authority_store,
        authority_verification, legacy_registry_sha, authority_registry_sha,
        genesis_record_sha,
    ) = graph
    receipt_id = _receipt_id(
        request=request, graph=graph, runtime_bindings=runtime_bindings,
        operation_bindings=operation_bindings, control_sha=control_sha,
        recognizer_inspect_sha=recognizer_inspect_sha,
        scorer_inspect_sha=scorer_inspect_sha,
    )
    values: list[tuple[str, object]] = [
        ("ceremony_receipt_id", receipt_id),
        ("version", PHASE2B_C1_LIVE_DOCKER_CEREMONY_V2_VERSION),
        ("claim_level", PHASE2B_C1_LIVE_DOCKER_CEREMONY_V2_CLAIM_LEVEL),
        ("reason", PHASE2B_C1_LIVE_DOCKER_CEREMONY_V2_COMPLETE_REASON),
        ("request", request), ("legacy_context", legacy_context),
        ("legacy_registry", legacy_registry),
        ("legacy_store_receipt", legacy_store),
        ("legacy_logical_genesis_token", genesis),
        ("legacy_wire_verification", legacy_wire),
        ("authority_context", authority_context),
        ("authority_registry", authority_registry),
        ("authority_store_receipt", authority_store),
        ("authority_verification", authority_verification),
        ("role_runtime_bindings", runtime_bindings),
        ("operation_bindings", operation_bindings),
        ("docker_control_plane_binding_sha256", control_sha),
        ("recognizer_image_inspect_sha256", recognizer_inspect_sha),
        ("scorer_image_inspect_sha256", scorer_inspect_sha),
        ("openssl_sha256", live_v1.OPENSSL_SHA256_V1),
        ("openssl_version", live_v1.OPENSSL_VERSION_V1),
        ("stored_v1_registry_sha256", legacy_registry_sha),
        ("stored_v2_registry_sha256", authority_registry_sha),
        ("logical_genesis_record_sha256", genesis_record_sha),
    ]
    values.extend((name, True) for name in _TRUE_RESULT_CLAIMS)
    values.extend((name, False) for name in _FALSE_RESULT_CLAIMS)
    receipt = Phase2BC1LiveDockerCeremonyReceiptV2(_RECEIPT_TOKEN)
    if tuple(name for name, _ in values) != tuple(
        item.name for item in fields(type(receipt))
    ):
        raise RuntimeError("live V2 receipt field drift")
    for name, value in values:
        object.__setattr__(receipt, name, value)
    receipt.__post_init__()
    return receipt


def execute_phase2b_c1_live_docker_ceremony_v2(
    request: live_v1.Phase2BC1LiveDockerCeremonyRequestV1,
    *, runtime_parent: Path = Path("/tmp"),
) -> Phase2BC1LiveDockerCeremonyReceiptV2:
    """Execute V1 bootstrap plus V2 authority in one six-container lifecycle."""

    if type(request) is not live_v1.Phase2BC1LiveDockerCeremonyRequestV1:
        _fail(live_v1.FAIL_PREFLIGHT, "executor requires exact V1 live request")
    if type(runtime_parent) is not type(Path("/")) or not runtime_parent.is_absolute():
        _fail(live_v1.FAIL_PREFLIGHT, "runtime parent must be exact absolute Path")
    if live_v1._CEREMONY_DEADLINES_V1.get() is None:
        with live_v1._ceremony_deadline_scope_v1():
            return execute_phase2b_c1_live_docker_ceremony_v2(
                request, runtime_parent=runtime_parent
            )
    request = live_v1._preflight_request_v1(request)
    source_seccomp = live_v1._load_seccomp_v1()

    with live_v1._ceremony_deadline_scope_v1():
        with live_v1._prepared_runtime_v1(runtime_parent) as (runtime_root, control):
            live_v1._validate_control_plane_v1(control, runtime_root)
            seccomp = live_v1._stage_seccomp_v1(source_seccomp, runtime_root)
            control_sha = _sha256(_canonical_json_bytes(dict(control.binding)))
            recognizer_image = live_v1._inspect_image_v1(
                control, runtime_root, image_digest=request.recognizer_image_digest,
                source_root_sha256=request.recognizer_source_root_sha256,
                role_kind="recognizer",
            )
            scorer_image = live_v1._inspect_image_v1(
                control, runtime_root, image_digest=request.scorer_image_digest,
                source_root_sha256=request.scorer_source_root_sha256,
                role_kind="scorer",
            )
            run_token = live_v1.secrets.token_hex(12)
            roles = live_v1._make_role_states_v1(
                request, run_token, recognizer_image=recognizer_image,
                scorer_image=scorer_image,
            )
            raw_volume_plan = sorted(
                ((role.purpose_id, path, name) for role in roles for path, name in role.volumes),
                key=lambda row: (0 if row[:2] == (3, "/ledger") else 1, row[0], row[1]),
            )
            volume_targets = tuple(
                (name, live_v1._volume_labels_v1(run_token, name))
                for _, _, name in raw_volume_plan
            )
            initializer_targets = tuple(
                live_v1._initializer_target_v1(
                    volume_name=name, image=scorer_image, run_token=run_token
                ) for _, _, name in raw_volume_plan
            )
            actor_targets = tuple(
                live_v1._actor_target_v1(role, run_token) for role in roles
            )
            all_targets = (*initializer_targets, *actor_targets)
            if len({target.name for target in all_targets}) != 17:
                raise RuntimeError("live V2 ceremony target collision")

            owned_volumes: dict[str, live_v1._OwnedVolumeV1] = {}
            owned_containers: dict[str, live_v1._OwnedContainerV1] = {}
            ambiguous_volumes: set[str] = set()
            ambiguous_containers: set[str] = set()
            primary_error: BaseException | None = None
            completed: tuple[object, ...] | None = None
            operations: list[live_v1.Phase2BC1LiveOperationBindingV1] = []
            try:
                for (_, _, name), initializer in zip(
                    raw_volume_plan, initializer_targets, strict=True
                ):
                    live_v1._provision_volume_v1(
                        control, runtime_root, seccomp, volume_name=name,
                        initializer_target=initializer, run_token=run_token,
                        owned_volumes=owned_volumes,
                        owned_containers=owned_containers,
                        ambiguous_volumes=ambiguous_volumes,
                        ambiguous_containers=ambiguous_containers,
                    )
                created = tuple(
                    live_v1._create_actor_v1(
                        control, runtime_root, seccomp, role, run_token,
                        owned_containers, ambiguous_containers,
                    ) for role in roles
                )
                if len({role.container_id for role in created}) != 6:
                    _fail(live_v1.FAIL_CONTAINER_LIFECYCLE, "actor full-ID drift")
                for role in created:
                    live_v1._start_actor_v1(control, runtime_root, role)
                for role in created:
                    live_v1._inspect_actor_v1(
                        control, runtime_root, seccomp, role, run_token
                    )
                qualified, runtime_bindings = live_v1._qualify_all_roles_v1(
                    control, runtime_root, seccomp, created, operations, run_token
                )
                keys = live_v1._generate_role_keys_v1(
                    control, runtime_root, qualified, operations
                )
                graph = _build_live_graph_v2(
                    control, runtime_root, request, qualified, keys, operations
                )
                observed = tuple(
                    (row.purpose_id, row.role, row.operation) for row in operations
                )
                if observed != _expected_operations() or len(operations) != 22:
                    _fail(FAIL_OPERATION_BINDING, "exact 22-operation graph drift")
                completed = (graph, runtime_bindings, tuple(operations))
            except BaseException as exc:
                primary_error = exc

            container_cleanup_error: BaseException | None = None
            try:
                live_v1._cleanup_containers_best_effort_v1(
                    control, runtime_root, seccomp, all_targets,
                    owned_containers, ambiguous_containers,
                )
            except BaseException as exc:
                container_cleanup_error = exc

            retention_error: BaseException | None = None
            if primary_error is None and container_cleanup_error is None:
                try:
                    live_v1._prove_volumes_retained_v1(
                        control, runtime_root, volume_targets, owned_volumes
                    )
                except BaseException as exc:
                    retention_error = exc
            if any(exc is not None for exc in (
                primary_error, container_cleanup_error, retention_error
            )):
                volume_cleanup_error: BaseException | None = None
                try:
                    live_v1._cleanup_volumes_best_effort_v1(
                        control, runtime_root, volume_targets, owned_volumes,
                        ambiguous_volumes,
                    )
                except BaseException as exc:
                    volume_cleanup_error = exc
                cleanup = tuple(exc for exc in (
                    container_cleanup_error, retention_error, volume_cleanup_error
                ) if exc is not None)
                if cleanup:
                    _fail(
                        live_v1.FAIL_CLEANUP,
                        "V2 cleanup failed after exhaustive pass: "
                        + ",".join(type(exc).__name__ for exc in cleanup),
                    )
                if primary_error is None:
                    raise RuntimeError("live V2 primary failure drift")
                raise primary_error

            if completed is None or len(owned_volumes) != 11:
                try:
                    live_v1._cleanup_volumes_best_effort_v1(
                        control, runtime_root, volume_targets, owned_volumes,
                        ambiguous_volumes,
                    )
                except BaseException as exc:
                    _fail(live_v1.FAIL_CLEANUP, f"incomplete V2 cleanup: {type(exc).__name__}")
                _fail(live_v1.FAIL_CLEANUP, "live V2 graph handoff incomplete")
            graph, runtime_bindings, operation_bindings = completed
            try:
                return _issue_receipt(
                    request=request, graph=graph,
                    runtime_bindings=runtime_bindings,
                    operation_bindings=operation_bindings,
                    control_sha=control_sha,
                    recognizer_inspect_sha=recognizer_image.inspect_sha256,
                    scorer_inspect_sha=scorer_image.inspect_sha256,
                )
            except BaseException as receipt_error:
                try:
                    live_v1._cleanup_volumes_best_effort_v1(
                        control, runtime_root, volume_targets, owned_volumes,
                        ambiguous_volumes,
                    )
                except BaseException as cleanup_error:
                    _fail(
                        live_v1.FAIL_CLEANUP,
                        "V2 receipt cleanup failed: " + type(cleanup_error).__name__,
                    )
                raise receipt_error


if len(_expected_operations()) != 22:
    raise RuntimeError("live V2 operation policy count drift")
if set(_TRUE_RESULT_CLAIMS) & set(_FALSE_RESULT_CLAIMS):
    raise RuntimeError("live V2 claim polarity overlap")


__all__ = (
    "FAIL_AUTHORITY", "FAIL_OPERATION_BINDING",
    "PHASE2B_C1_LIVE_DOCKER_CEREMONY_V2_CLAIM_LEVEL",
    "PHASE2B_C1_LIVE_DOCKER_CEREMONY_V2_COMPLETE_REASON",
    "PHASE2B_C1_LIVE_DOCKER_CEREMONY_V2_VERSION",
    "Phase2BC1LiveDockerCeremonyReceiptV2",
    "execute_phase2b_c1_live_docker_ceremony_v2",
)
