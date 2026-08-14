from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import Enum
import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from hegel_machine import phase2b_c1_logical_stage_authority_v2 as authority
from hegel_machine import phase2b_c1_logical_stage_wire_v1 as logical
from hegel_machine import phase2b_c1_six_role_wire_v1 as six


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/phase2b_c1_logical_stage_authorization_v2.json"
H = tuple(format(index, "x") * 64 for index in range(1, 10))
PRIVATE_SEED_HEX = tuple(f"{index:02x}" * 32 for index in range(1, 7))
PUBLIC_KEY_HEX = (
    "8a88e3dd7409f195fd52db2d3cba5d72ca6709bf1d94121bf3748801b40f6f5c",
    "8139770ea87d175f56a35466c34c7ecccb8d8a91b4ee37a25df60f5b8fc9b394",
    "ed4928c628d1c2c6eae90338905995612959273a5c63f93636c14614ac8737d1",
    "ca93ac1705187071d67b83c7ff0efe8108e8ec4530575d7726879333dbdabe7c",
    "6e7a1cdd29b0b78fd13af4c5598feff4ef2a97166e3ca6f2e4fbfccd80505bf1",
    "8a875fff1eb38451577acd5afee405456568dd7c89e090863a0557bc7af49f17",
)
AUTHORIZATION_SIGNATURE_HEX = (
    "b3e336e3bfa34527cf2f7a307f50c546bd53dae8217026bf256898f1b4896e3c891c4038ab8e457e075fc2bca0de19f682fb5cf53948ec6940697106b285e00f",
    "a1d6c71fc3cf2ad3913cd8b1dc107ac53896d36553ab65495a8e74380b7b01edd78cb2d31043db658244e95ab3541f3679e53178176a3caf93a3ae7a3defc80d",
    "bd0371781f241627e75e88863e771dfbe5a7757956a102ba62ee666ac61c74531f45315bb74c64638b3d02acf309ecbcd6dc6996934b105a17c5b46c3c48c90e",
    "9ab0eec4631dd8845dcacf38001760c07ed6941d453492f6c6ac4f8b2a5d8c8b5081eb77b9f42ed2c9daecede4e9d309173e51998454c175e65eb0885dc02301",
    "5909c1c7c3911eeb7a0ec2514162ce7fadf204072020bac72dd8a8db795ba6602d35bd63ec40c7bf6c4ea21cfd3ee1ca3ad81472044c41f5bd68dc89b4654200",
    "c2d35cefa141efcddfbd2b75324f0f1622b1b212e1c60964df03de467af825bf9a4c1ef2d0f5dea837d92bbc1fa29cd358f422cd93a84dd6a705220be9471b0b",
)
AUTHORIZATION_PREIMAGE_SHA256 = (
    "85b8b600f87c3ab4c8adb808cac1e2a9e5a3de47f87cb342d986f158440a7989",
    "3db993dd82da7a6e7d25b07a9f4914ab316662bb70793a2a9dcfe54a1a2b89c1",
    "ec6f3193fbb3b62dea9e4575bedbc580dc8438d6974a5667c445543c2eb0025f",
    "ab89ee64d95da8d5b5234d7ca6b769c675c998ffe9002c6604eb75bfe36eb9ce",
    "a804829527cdce91b2206b8485ffdf45325a9ec39bffb6a0dd5242b2644a548f",
    "70ee25c9e7a80406189ada3960f04630c53d5a03189592616a073deb66596f34",
)
STORE_SIGNATURE_HEX = (
    "32c76af4b14550356f9722cd98a056be9ae0c76f6e87cb463edee9f28308abca"
    "61ae6f6f94088ccc345483f26ccc271cf865452f1c6e1ad04809269097d22c00"
)
EXPECTED_PUBLIC_SURFACE = {
    "PHASE2B_C1_LOGICAL_STAGE_AUTHORITY_V2_VERSION",
    "PHASE2B_C1_LOGICAL_STAGE_AUTHORITY_V2_CLAIM_LEVEL",
    "PHASE2B_C1_LOGICAL_STAGE_WIRE_SCHEMA_ID_V1",
    "PHASE2B_C1_STAGE_EVENT_CONTRACT_MANIFEST_ID_V1",
    "PHASE2B_C1_LOGICAL_STAGE_WIRE_POLICY_ID_V2",
    "PHASE2B_C1_LOGICAL_STAGE_WIRE_SOURCE_COMMIT_V1",
    "PHASE2B_C1_LOGICAL_STAGE_WIRE_SOURCE_BLOB_V1",
    "PHASE2B_C1_LOGICAL_STAGE_WIRE_SOURCE_SHA256_V1",
    "PHASE2B_C1_LOGICAL_STAGE_AUTHORIZATION_SEMANTICS_V2",
    "PHASE2B_C1_LOGICAL_STAGE_AUTHORIZATION_SIGNATURE_PURPOSE_V2",
    "Phase2BC1LogicalStageAuthorityDispositionV2",
    "Phase2BC1LogicalStageAuthorityReasonV2",
    "Phase2BC1AuthorizedLogicalStagePrefixDispositionV2",
    "Phase2BC1AuthorizedLogicalStagePrefixReasonV2",
    "Phase2BC1LogicalStageAuthorityContextBodyV2",
    "Phase2BC1LogicalStageAuthorityContextV2",
    "Phase2BC1LogicalStageAuthorizationAmendmentBodyV2",
    "Phase2BC1LogicalStageAuthorizationSignatureV2",
    "Phase2BC1OwnerAcceptedExecutionRegistryV2",
    "Phase2BC1LogicalStageAuthorizationStoreReceiptBodyV2",
    "Phase2BC1LogicalStageAuthorizationStoreReceiptCandidateV2",
    "Phase2BC1LogicalStageAuthorityVerificationV2",
    "Phase2BC1AuthorizedLogicalStagePrefixVerificationV2",
    "phase2b_c1_logical_stage_authority_context_id_v2",
    "project_phase2b_c1_logical_stage_authority_context_to_v1",
    "phase2b_c1_logical_stage_authorization_amendment_id_v2",
    "phase2b_c1_logical_stage_authorization_signature_preimage_v2",
    "phase2b_c1_owner_accepted_execution_registry_id_v2",
    "canonical_phase2b_c1_owner_accepted_execution_registry_bytes_v2",
    "phase2b_c1_logical_stage_authorization_store_signature_preimage_v2",
    "phase2b_c1_logical_stage_authorization_store_receipt_id_v2",
    "verify_phase2b_c1_logical_stage_authority_graph_v2",
    "verify_authorized_phase2b_c1_logical_stage_prefix_v2",
}


@dataclass(frozen=True)
class LegacyGraph:
    private_keys: tuple[Ed25519PrivateKey, ...]
    context: six.Phase2BC1CeremonyContextV1
    registry: six.Phase2BC1ExecutionRegistryCandidateV1
    store: six.Phase2BC1StoreIfAbsentReceiptCandidateV1
    genesis: six.Phase2BC1LogicalGenesisTokenCandidateV1


@dataclass(frozen=True)
class AuthorityGraph:
    context: authority.Phase2BC1LogicalStageAuthorityContextV2
    registry: authority.Phase2BC1OwnerAcceptedExecutionRegistryV2
    store: authority.Phase2BC1LogicalStageAuthorizationStoreReceiptCandidateV2


def _public(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )


def _legacy_signature(
    *, key: Ed25519PrivateKey, key_id: str, purpose_id: int,
    role: six.Phase2BC1SixRoleV1,
    signature_purpose: six.Phase2BC1SignaturePurposeV1, preimage: bytes,
) -> six.Phase2BC1DetachedSignatureV1:
    return six.Phase2BC1DetachedSignatureV1(
        purpose_id=purpose_id, role=role,
        signature_purpose=signature_purpose, algorithm="Ed25519",
        key_epoch=1, signer_key_id=key_id, signature=key.sign(preimage),
    )


def _build_legacy(*, campaign_hash: str = H[0]) -> LegacyGraph:
    private_keys = tuple(
        Ed25519PrivateKey.from_private_bytes(bytes([index]) * 32)
        for index in range(1, 7)
    )
    campaign_id = "phase2b_actual_unsealed_960_campaign_v2_" + campaign_hash
    freeze_id = "phase2b_execution_freeze_" + H[1]
    attempt_id = six.phase2b_c1_attempt_id_v1(
        authority_requirement_id=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_REQUIREMENT_ID,
        actor_profile_content_id=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CONTENT_ID,
        actor_profile_canonical_sha256=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CANONICAL_SHA256,
        campaign_id=campaign_id,
        protocol_id=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_FROZEN_PROTOCOL_ID,
        exact_freeze_id=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_FROZEN_EXACT_FREEZE_ID,
        execution_freeze_manifest_id=freeze_id,
        execution_freeze_manifest_sha256=H[2],
        attempt_nonce_sha256=H[3],
    )
    body = six.Phase2BC1CeremonyContextBodyV1(
        version=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_VERSION,
        authority_class=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_CLASS,
        authority_requirement_id=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_REQUIREMENT_ID,
        actor_profile_content_id=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CONTENT_ID,
        actor_profile_canonical_sha256=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CANONICAL_SHA256,
        campaign_id=campaign_id,
        protocol_id=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_FROZEN_PROTOCOL_ID,
        exact_freeze_id=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_FROZEN_EXACT_FREEZE_ID,
        execution_freeze_manifest_id=freeze_id,
        execution_freeze_manifest_sha256=H[2],
        attempt_nonce_sha256=H[3], attempt_id=attempt_id,
        recognizer_image_digest="sha256:" + H[4],
        recognizer_source_root_sha256=H[5],
        scorer_image_digest="sha256:" + H[6],
        scorer_source_root_sha256=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_SCORER_SOURCE_ROOT_SHA256,
    )
    context = six.Phase2BC1CeremonyContextV1(
        body=body, context_id=six.phase2b_c1_ceremony_context_id_v1(body)
    )
    role_keys = tuple(
        six.Phase2BC1RoleKeyV1(
            purpose_id=index, role=role, algorithm="Ed25519", key_epoch=1,
            purpose_private=True, public_key=_public(private_keys[index - 1]),
            key_id=six.phase2b_c1_role_key_id_v1(
                purpose_id=index, role=role,
                public_key=_public(private_keys[index - 1]),
            ),
        )
        for index, role in enumerate(six.PHASE2B_C1_SIX_ROLE_WIRE_V1_ROLE_ORDER, 1)
    )
    registry_body = six.Phase2BC1ExecutionRegistryBodyV1(
        version=body.version, context_id=context.context_id,
        authority_class=body.authority_class,
        authority_requirement_id=body.authority_requirement_id,
        actor_profile_content_id=body.actor_profile_content_id,
        actor_profile_canonical_sha256=body.actor_profile_canonical_sha256,
        campaign_id=body.campaign_id, protocol_id=body.protocol_id,
        exact_freeze_id=body.exact_freeze_id,
        execution_freeze_manifest_id=body.execution_freeze_manifest_id,
        execution_freeze_manifest_sha256=body.execution_freeze_manifest_sha256,
        attempt_nonce_sha256=body.attempt_nonce_sha256, attempt_id=body.attempt_id,
        recognizer_image_digest=body.recognizer_image_digest,
        recognizer_source_root_sha256=body.recognizer_source_root_sha256,
        scorer_image_digest=body.scorer_image_digest,
        scorer_source_root_sha256=body.scorer_source_root_sha256,
        role_keys=role_keys,
    )
    governance = _legacy_signature(
        key=private_keys[0], key_id=role_keys[0].key_id, purpose_id=1,
        role=six.Phase2BC1SixRoleV1.GOVERNANCE_ANCHOR,
        signature_purpose=six.Phase2BC1SignaturePurposeV1.GOVERNANCE_CHECKPOINT,
        preimage=six.phase2b_c1_governance_registry_signature_preimage_v1(
            registry_body, signer_key_id=role_keys[0].key_id
        ),
    )
    registry = six.Phase2BC1ExecutionRegistryCandidateV1(
        body=registry_body, governance_signature=governance,
        execution_registry_id=six.phase2b_c1_execution_registry_id_v1(
            registry_body, governance
        ),
    )
    object_key = (
        f"phase2b/c1/actual-unsealed-960/{campaign_id}/attempt-0/execution-registry"
    )
    store_body = six.Phase2BC1StoreIfAbsentReceiptBodyV1(
        version=body.version, context_id=context.context_id,
        authority_requirement_id=body.authority_requirement_id,
        actor_profile_content_id=body.actor_profile_content_id,
        campaign_id=campaign_id, attempt_id=attempt_id,
        execution_registry_id=registry.execution_registry_id,
        object_key=object_key, previous_generation=0, new_generation=1,
        condition="IF_ABSENT", previous_content_sha256_or_null=None,
        stored_content_sha256=hashlib.sha256(
            six.canonical_phase2b_c1_execution_registry_bytes_v1(registry)
        ).hexdigest(),
    )
    store_signature = _legacy_signature(
        key=private_keys[2], key_id=role_keys[2].key_id, purpose_id=3,
        role=six.Phase2BC1SixRoleV1.STORE_RECEIPT,
        signature_purpose=six.Phase2BC1SignaturePurposeV1.STORE_IF_ABSENT_RECEIPT,
        preimage=six.phase2b_c1_store_receipt_signature_preimage_v1(
            store_body, signer_key_id=role_keys[2].key_id
        ),
    )
    store = six.Phase2BC1StoreIfAbsentReceiptCandidateV1(
        body=store_body, store_signature=store_signature,
        store_receipt_id=six.phase2b_c1_store_receipt_id_v1(
            store_body, store_signature
        ),
    )
    genesis_body = six.Phase2BC1LogicalGenesisTokenBodyV1(
        version=body.version, context_id=context.context_id,
        authority_requirement_id=body.authority_requirement_id,
        actor_profile_content_id=body.actor_profile_content_id,
        campaign_id=campaign_id, attempt_id=attempt_id,
        execution_registry_id=registry.execution_registry_id,
        store_receipt_id=store.store_receipt_id,
        time_semantics=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_TIME_SEMANTICS,
        previous_logical_sequence_or_null=None, logical_sequence=0,
        stage=six.Phase2BC1LogicalStageV1.PREREGISTERED, stage_index=0,
    )
    genesis_signature = _legacy_signature(
        key=private_keys[3], key_id=role_keys[3].key_id, purpose_id=4,
        role=six.Phase2BC1SixRoleV1.TIME_ATTESTOR,
        signature_purpose=six.Phase2BC1SignaturePurposeV1.DECLARED_TIME_TOKEN,
        preimage=six.phase2b_c1_logical_genesis_signature_preimage_v1(
            genesis_body, signer_key_id=role_keys[3].key_id
        ),
    )
    genesis = six.Phase2BC1LogicalGenesisTokenCandidateV1(
        body=genesis_body, time_signature=genesis_signature,
        logical_genesis_token_id=six.phase2b_c1_logical_genesis_token_id_v1(
            genesis_body, genesis_signature
        ),
    )
    return LegacyGraph(private_keys, context, registry, store, genesis)


def _build_authority(legacy: LegacyGraph) -> AuthorityGraph:
    old = legacy.context.body
    context_body = authority.Phase2BC1LogicalStageAuthorityContextBodyV2(
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
        legacy_context_id=legacy.context.context_id,
        legacy_execution_registry_id=legacy.registry.execution_registry_id,
        logical_stage_wire_schema_id=authority.PHASE2B_C1_LOGICAL_STAGE_WIRE_SCHEMA_ID_V1,
        logical_stage_wire_policy_id=authority.PHASE2B_C1_LOGICAL_STAGE_WIRE_POLICY_ID_V2,
        stage_event_contract_manifest_id=authority.PHASE2B_C1_STAGE_EVENT_CONTRACT_MANIFEST_ID_V1,
    )
    context = authority.Phase2BC1LogicalStageAuthorityContextV2(
        body=context_body,
        authority_context_id=authority.phase2b_c1_logical_stage_authority_context_id_v2(
            context_body
        ),
    )
    amendment = authority.Phase2BC1LogicalStageAuthorizationAmendmentBodyV2(
        version=authority.PHASE2B_C1_LOGICAL_STAGE_AUTHORITY_V2_VERSION,
        authority_context_id=context.authority_context_id,
        legacy_context_id=legacy.context.context_id,
        legacy_execution_registry_id=legacy.registry.execution_registry_id,
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
        required_signer_key_ids=tuple(
            key.key_id for key in legacy.registry.body.role_keys
        ),
        authorization_semantics=authority.PHASE2B_C1_LOGICAL_STAGE_AUTHORIZATION_SEMANTICS_V2,
        legacy_token_authorization_field_must_remain_unchanged=True,
        legacy_actual_campaign_eligible_must_remain_false=True,
        actual_execution_claimed=False,
    )
    amendment_id = authority.phase2b_c1_logical_stage_authorization_amendment_id_v2(
        amendment
    )
    signatures = tuple(
        authority.Phase2BC1LogicalStageAuthorizationSignatureV2(
            purpose_id=index, role=role,
            signature_purpose=authority.PHASE2B_C1_LOGICAL_STAGE_AUTHORIZATION_SIGNATURE_PURPOSE_V2,
            algorithm="Ed25519", key_epoch=1, signer_key_id=key.key_id,
            signature=legacy.private_keys[index - 1].sign(
                authority.phase2b_c1_logical_stage_authorization_signature_preimage_v2(
                    amendment, purpose_id=index, role=role,
                    signer_key_id=key.key_id,
                )
            ),
        )
        for index, (role, key) in enumerate(zip(
            six.PHASE2B_C1_SIX_ROLE_WIRE_V1_ROLE_ORDER,
            legacy.registry.body.role_keys,
            strict=True,
        ), start=1)
    )
    registry = authority.Phase2BC1OwnerAcceptedExecutionRegistryV2(
        body=amendment, authorization_amendment_id=amendment_id,
        role_acceptance_signatures=signatures,
        execution_registry_id=authority.phase2b_c1_owner_accepted_execution_registry_id_v2(
            amendment, amendment_id, signatures
        ),
    )
    registry_sha = hashlib.sha256(
        authority.canonical_phase2b_c1_owner_accepted_execution_registry_bytes_v2(
            registry
        )
    ).hexdigest()
    store_body = authority.Phase2BC1LogicalStageAuthorizationStoreReceiptBodyV2(
        version=authority.PHASE2B_C1_LOGICAL_STAGE_AUTHORITY_V2_VERSION,
        authority_context_id=context.authority_context_id,
        legacy_context_id=legacy.context.context_id,
        campaign_id=old.campaign_id, attempt_id=old.attempt_id,
        execution_registry_id=registry.execution_registry_id,
        object_key=(
            "phase2b/c1/actual-unsealed-960/"
            f"{old.campaign_id}/attempt-0/logical-stage-authorization-registry-v2"
        ),
        previous_generation=0, new_generation=1, condition="IF_ABSENT",
        previous_content_sha256_or_null=None,
        stored_content_sha256=registry_sha,
    )
    store_key = legacy.registry.body.role_keys[2]
    store_signature = _legacy_signature(
        key=legacy.private_keys[2], key_id=store_key.key_id, purpose_id=3,
        role=six.Phase2BC1SixRoleV1.STORE_RECEIPT,
        signature_purpose=six.Phase2BC1SignaturePurposeV1.STORE_IF_ABSENT_RECEIPT,
        preimage=authority.phase2b_c1_logical_stage_authorization_store_signature_preimage_v2(
            store_body, signer_key_id=store_key.key_id
        ),
    )
    store = authority.Phase2BC1LogicalStageAuthorizationStoreReceiptCandidateV2(
        body=store_body, store_signature=store_signature,
        store_receipt_id=authority.phase2b_c1_logical_stage_authorization_store_receipt_id_v2(
            store_body, store_signature
        ),
    )
    return AuthorityGraph(context, registry, store)


def _verify(legacy: LegacyGraph, graph: AuthorityGraph):
    return authority.verify_phase2b_c1_logical_stage_authority_graph_v2(
        legacy_context=legacy.context, legacy_registry=legacy.registry,
        legacy_store_receipt=legacy.store,
        legacy_logical_genesis_token=legacy.genesis,
        authority_context=graph.context,
        execution_registry=graph.registry,
        authorization_store_receipt=graph.store,
    )


def _reseal_store(
    legacy: LegacyGraph,
    graph: AuthorityGraph,
    registry: authority.Phase2BC1OwnerAcceptedExecutionRegistryV2,
) -> AuthorityGraph:
    body = replace(
        graph.store.body,
        execution_registry_id=registry.execution_registry_id,
        stored_content_sha256=hashlib.sha256(
            authority.canonical_phase2b_c1_owner_accepted_execution_registry_bytes_v2(
                registry
            )
        ).hexdigest(),
    )
    key = legacy.registry.body.role_keys[2]
    signature = _legacy_signature(
        key=legacy.private_keys[2], key_id=key.key_id, purpose_id=3,
        role=six.Phase2BC1SixRoleV1.STORE_RECEIPT,
        signature_purpose=six.Phase2BC1SignaturePurposeV1.STORE_IF_ABSENT_RECEIPT,
        preimage=authority.phase2b_c1_logical_stage_authorization_store_signature_preimage_v2(
            body, signer_key_id=key.key_id
        ),
    )
    store = authority.Phase2BC1LogicalStageAuthorizationStoreReceiptCandidateV2(
        body=body, store_signature=signature,
        store_receipt_id=authority.phase2b_c1_logical_stage_authorization_store_receipt_id_v2(
            body, signature
        ),
    )
    return replace(graph, registry=registry, store=store)


def _raw_prefix(legacy: LegacyGraph):
    old = legacy.context.body
    projection = logical.Phase2BC1RawInputCommittedEventProjectionV1(
        version=logical.PHASE2B_C1_LOGICAL_STAGE_WIRE_V1_VERSION,
        context_id=legacy.context.context_id,
        authority_requirement_id=old.authority_requirement_id,
        actor_profile_content_id=old.actor_profile_content_id,
        campaign_id=old.campaign_id, attempt_id=old.attempt_id,
        execution_registry_id=legacy.registry.execution_registry_id,
        input_archive_id="phase2b_recognizer_input_archive_v2_" + H[7],
        input_archive_sha256=H[8], input_archive_bytes=1234,
        raw_input_commitment_sha256=H[1],
        raw_input_commitment_receipt_id=(
            "phase2b_c1_raw_input_commitment_receipt_v1_" + H[2]
        ),
    )
    token_body = logical.Phase2BC1LogicalStageTokenBodyV1(
        version=logical.PHASE2B_C1_LOGICAL_STAGE_WIRE_V1_VERSION,
        context_id=legacy.context.context_id,
        authority_requirement_id=old.authority_requirement_id,
        actor_profile_content_id=old.actor_profile_content_id,
        actor_profile_canonical_sha256=old.actor_profile_canonical_sha256,
        campaign_id=old.campaign_id, attempt_id=old.attempt_id,
        execution_registry_id=legacy.registry.execution_registry_id,
        logical_genesis_token_id=legacy.genesis.logical_genesis_token_id,
        legacy_registry_policy_authorization=logical.PHASE2B_C1_LOGICAL_STAGE_WIRE_V1_LEGACY_POLICY_AUTHORIZATION,
        actual_campaign_eligible=False,
        time_semantics=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_TIME_SEMANTICS,
        predecessor_token_id=legacy.genesis.logical_genesis_token_id,
        previous_logical_sequence=0, logical_sequence=1,
        previous_stage=six.Phase2BC1LogicalStageV1.PREREGISTERED,
        stage=six.Phase2BC1LogicalStageV1.RAW_INPUT_COMMITTED,
        stage_index=1,
        event_projection_id=logical.phase2b_c1_logical_stage_event_projection_id_v1(
            projection
        ),
        event_root_sha256=logical.phase2b_c1_logical_stage_event_root_sha256_v1(
            projection
        ),
    )
    time_key = legacy.registry.body.role_keys[3]
    signature = _legacy_signature(
        key=legacy.private_keys[3], key_id=time_key.key_id, purpose_id=4,
        role=six.Phase2BC1SixRoleV1.TIME_ATTESTOR,
        signature_purpose=six.Phase2BC1SignaturePurposeV1.DECLARED_TIME_TOKEN,
        preimage=logical.phase2b_c1_logical_stage_signature_preimage_v1(
            token_body, signer_key_id=time_key.key_id
        ),
    )
    token = logical.Phase2BC1LogicalStageTokenCandidateV1(
        body=token_body, time_signature=signature,
        logical_stage_token_id=logical.phase2b_c1_logical_stage_token_id_v1(
            token_body, signature
        ),
    )
    return (projection,), (token,)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def _oracle_named_pairs(pairs: tuple[tuple[str, object], ...]) -> bytes:
    return json.dumps(
        pairs, ensure_ascii=True, separators=(",", ":"), allow_nan=False,
    ).encode("ascii")


def _oracle_object_pairs(
    value: object, names: tuple[str, ...],
) -> tuple[tuple[str, object], ...]:
    output: list[tuple[str, object]] = []
    for name in names:
        item = getattr(value, name)
        if isinstance(item, Enum):
            item = item.value
        elif type(item) is tuple:
            item = tuple(
                member.value if isinstance(member, Enum) else member
                for member in item
            )
        output.append((name, item))
    return tuple(output)


def _oracle_id(prefix: str, domain: bytes, pairs: tuple[tuple[str, object], ...]) -> str:
    return prefix + hashlib.sha256(domain + _oracle_named_pairs(pairs)).hexdigest()


CONTEXT_BODY_FIELDS = (
    "version", "authority_class", "authority_requirement_id",
    "actor_profile_content_id", "actor_profile_canonical_sha256", "campaign_id",
    "protocol_id", "exact_freeze_id", "execution_freeze_manifest_id",
    "execution_freeze_manifest_sha256", "attempt_nonce_sha256", "attempt_id",
    "recognizer_image_digest", "recognizer_source_root_sha256",
    "scorer_image_digest", "scorer_source_root_sha256", "legacy_context_id",
    "legacy_execution_registry_id", "logical_stage_wire_schema_id",
    "logical_stage_wire_policy_id", "stage_event_contract_manifest_id",
)
AMENDMENT_BODY_FIELDS = (
    "version", "authority_context_id", "legacy_context_id",
    "legacy_execution_registry_id", "authority_class", "authority_requirement_id",
    "actor_profile_content_id", "actor_profile_canonical_sha256", "campaign_id",
    "attempt_id", "logical_stage_wire_schema_id", "logical_stage_wire_policy_id",
    "stage_event_contract_manifest_id", "authorized_stage_order",
    "authorized_stage_indices", "required_signer_roles", "required_signer_key_ids",
    "authorization_semantics",
    "legacy_token_authorization_field_must_remain_unchanged",
    "legacy_actual_campaign_eligible_must_remain_false", "actual_execution_claimed",
)
STORE_BODY_FIELDS = (
    "version", "authority_context_id", "legacy_context_id", "campaign_id",
    "attempt_id", "execution_registry_id", "object_key", "previous_generation",
    "new_generation", "condition", "previous_content_sha256_or_null",
    "stored_content_sha256",
)


def test_config_three_preimages_compute_exact_ids_and_pin_committed_wire() -> None:
    value = json.loads(CONFIG.read_bytes())
    schema_id = "phase2b_c1_logical_stage_wire_schema_v1_" + hashlib.sha256(
        b"HEGEL/PHASE2B/C1/LOGICAL_STAGE_WIRE/SCHEMA/V1\0"
        + _canonical(value["logical_stage_wire_schema"])
    ).hexdigest()
    manifest_id = "phase2b_c1_stage_event_contract_manifest_v1_" + hashlib.sha256(
        b"HEGEL/PHASE2B/C1/STAGE_EVENT_CONTRACT_MANIFEST/V1\0"
        + _canonical(value["stage_event_contract_manifest"])
    ).hexdigest()
    policy_id = "phase2b_c1_logical_stage_wire_policy_v2_" + hashlib.sha256(
        b"HEGEL/PHASE2B/C1/LOGICAL_STAGE_AUTHORIZATION_POLICY/V2\0"
        + _canonical(value["logical_stage_wire_policy"])
    ).hexdigest()
    assert schema_id == value["logical_stage_wire_schema_id"] == authority.PHASE2B_C1_LOGICAL_STAGE_WIRE_SCHEMA_ID_V1
    assert manifest_id == value["stage_event_contract_manifest_id"] == authority.PHASE2B_C1_STAGE_EVENT_CONTRACT_MANIFEST_ID_V1
    assert policy_id == value["logical_stage_wire_policy_id"] == authority.PHASE2B_C1_LOGICAL_STAGE_WIRE_POLICY_ID_V2
    manifest = value["stage_event_contract_manifest"]
    assert manifest["normative_source_commit"] == authority.PHASE2B_C1_LOGICAL_STAGE_WIRE_SOURCE_COMMIT_V1
    assert manifest["normative_source_git_blob_oid"] == authority.PHASE2B_C1_LOGICAL_STAGE_WIRE_SOURCE_BLOB_V1
    assert manifest["normative_source_sha256"] == authority.PHASE2B_C1_LOGICAL_STAGE_WIRE_SOURCE_SHA256_V1
    assert hashlib.sha256((ROOT / manifest["normative_source_path"]).read_bytes()).hexdigest() == manifest["normative_source_sha256"]


def test_manifest_exact_domains_and_fields_match_committed_public_dtos() -> None:
    value = json.loads(CONFIG.read_bytes())
    schema = value["logical_stage_wire_schema"]
    types = (
        logical.Phase2BC1RawInputCommittedEventProjectionV1,
        logical.Phase2BC1PackageCommitmentsFrozenEventProjectionV1,
        logical.Phase2BC1RunStartedEventProjectionV1,
        logical.Phase2BC1RunFinishedEventProjectionV1,
    )
    assert tuple(row["class_name"] for row in schema["event_projection_types_in_order"]) == tuple(item.__name__ for item in types)
    assert tuple(tuple(row["fields_in_order"]) for row in schema["event_projection_types_in_order"]) == tuple(tuple(item.name for item in fields(cls)) for cls in types)
    assert tuple(schema["token_body_fields_in_order"]) == tuple(item.name for item in fields(logical.Phase2BC1LogicalStageTokenBodyV1))
    assert tuple(schema["token_candidate_fields_in_order"]) == tuple(item.name for item in fields(logical.Phase2BC1LogicalStageTokenCandidateV1))
    assert bytes.fromhex(schema["stage_signature_domain_hex"]) == logical._STAGE_SIGNATURE_DOMAIN
    assert bytes.fromhex(schema["stage_token_id_domain_hex"]) == logical._STAGE_TOKEN_ID_DOMAIN
    assert schema["stage_token_id_prefix"] == logical._STAGE_TOKEN_PREFIX
    assert tuple(bytes.fromhex(row["event_domain_hex"]) for row in schema["event_projection_types_in_order"]) == tuple(logical._EVENT_DOMAINS[stage] for stage in logical._PREFIX_STAGES)
    assert tuple(row["projection_id_prefix"] for row in schema["event_projection_types_in_order"]) == tuple(logical._PROJECTION_PREFIXES[stage] for stage in logical._PREFIX_STAGES)
    assert tuple(schema["signature_header_fields_in_order"]) == (
        "purpose_id", "role", "signature_purpose", "algorithm", "key_epoch",
        "signer_key_id",
    )


def test_authority_graph_projects_exact_v1_and_verifies_six_signatures_and_store() -> None:
    legacy = _build_legacy()
    graph = _build_authority(legacy)
    assert authority.project_phase2b_c1_logical_stage_authority_context_to_v1(
        graph.context
    ) == legacy.context
    result = _verify(legacy, graph)
    assert result.disposition is authority.Phase2BC1LogicalStageAuthorityDispositionV2.CALLER_SUPPLIED_OWNER_ACCEPTED_V2_AUTHORITY_GRAPH_VERIFIED_NOT_EXECUTION
    assert result.six_unanimous_role_acceptance_signatures_verified is True
    assert result.v2_store_if_absent_transition_verified is True
    assert result.v2_policy_admission_verified is True
    for name in authority._FALSE_CLAIMS:
        assert getattr(result, name) is False


def test_literal_authority_context_amendment_and_first_signature_kat() -> None:
    legacy = _build_legacy()
    graph = _build_authority(legacy)
    assert graph.context.authority_context_id == (
        "phase2b_c1_logical_stage_authority_context_v2_"
        "09631a94bc95c0ad0244c48f62c41bd1eb49480bd09f31bb224da110cc9bc3cd"
    )
    assert graph.registry.authorization_amendment_id == (
        "phase2b_c1_logical_stage_authorization_amendment_v2_"
        "fbc58d200646afbdec16608f32d1506979340d8e134bf5ed5548549ab0059cf3"
    )
    preimage = authority.phase2b_c1_logical_stage_authorization_signature_preimage_v2(
        graph.registry.body, purpose_id=1,
        role=six.Phase2BC1SixRoleV1.GOVERNANCE_ANCHOR,
        signer_key_id=legacy.registry.body.role_keys[0].key_id,
    )
    legacy.private_keys[0].public_key().verify(
        graph.registry.role_acceptance_signatures[0].signature, preimage
    )
    assert hashlib.sha256(preimage).hexdigest() == (
        "85b8b600f87c3ab4c8adb808cac1e2a9e5a3de47f87cb342d986f158440a7989"
    )
    assert graph.registry.role_acceptance_signatures[0].signature.hex() == (
        "b3e336e3bfa34527cf2f7a307f50c546bd53dae8217026bf256898f1b4896e3c"
        "891c4038ab8e457e075fc2bca0de19f682fb5cf53948ec6940697106b285e00f"
    )
    assert graph.registry.execution_registry_id == (
        "phase2b_c1_owner_accepted_execution_registry_v2_"
        "c15f6bf40ba754fec13153d250b1ee4badda5df584cee733950adb22f63b76ef"
    )
    assert graph.store.store_receipt_id == (
        "phase2b_c1_logical_stage_authorization_if_absent_receipt_v2_"
        "c804c5d84a524403677705cfaa7463f8f6f5f3886911416a070436471581595f"
    )


def test_source_independent_v2_oracle_and_public_helper_differential() -> None:
    legacy = _build_legacy()
    graph = _build_authority(legacy)
    roles = (
        "GOVERNANCE_ANCHOR", "CUSTODIAN", "STORE_RECEIPT", "TIME_ATTESTOR",
        "RECOGNIZER", "SCORER_EVALUATOR",
    )

    context_pairs = _oracle_object_pairs(graph.context.body, CONTEXT_BODY_FIELDS)
    oracle_context_id = _oracle_id(
        "phase2b_c1_logical_stage_authority_context_v2_",
        b"HEGEL/PHASE2B/C1/LOGICAL_STAGE_AUTHORITY/CONTEXT/V2\0",
        context_pairs,
    )
    assert oracle_context_id == (
        "phase2b_c1_logical_stage_authority_context_v2_"
        "09631a94bc95c0ad0244c48f62c41bd1eb49480bd09f31bb224da110cc9bc3cd"
    )
    assert oracle_context_id == graph.context.authority_context_id

    amendment_pairs = _oracle_object_pairs(
        graph.registry.body, AMENDMENT_BODY_FIELDS
    )
    assert dict(amendment_pairs)["authorized_stage_order"] == (
        "RAW_INPUT_COMMITTED", "PACKAGE_COMMITMENTS_FROZEN", "RUN_STARTED",
        "RUN_FINISHED",
    )
    assert dict(amendment_pairs)["required_signer_roles"] == roles
    oracle_amendment_id = _oracle_id(
        "phase2b_c1_logical_stage_authorization_amendment_v2_",
        b"HEGEL/PHASE2B/C1/LOGICAL_STAGE_AUTHORIZATION/AMENDMENT_ID/V2\0",
        amendment_pairs,
    )
    assert oracle_amendment_id == (
        "phase2b_c1_logical_stage_authorization_amendment_v2_"
        "fbc58d200646afbdec16608f32d1506979340d8e134bf5ed5548549ab0059cf3"
    )
    assert oracle_amendment_id == graph.registry.authorization_amendment_id

    oracle_preimages: list[bytes] = []
    oracle_signature_rows: list[tuple[tuple[str, object], ...]] = []
    for purpose_id, (seed_hex, public_hex, role, key, expected_signature) in enumerate(
        zip(
            PRIVATE_SEED_HEX, PUBLIC_KEY_HEX, roles,
            legacy.registry.body.role_keys, AUTHORIZATION_SIGNATURE_HEX,
            strict=True,
        ),
        start=1,
    ):
        private_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed_hex))
        raw_public = private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        assert raw_public.hex() == public_hex
        assert raw_public == key.public_key
        header = (
            ("purpose_id", purpose_id), ("role", role),
            ("signature_purpose", "LOGICAL_STAGE_AUTHORIZATION_AMENDMENT_ACCEPTANCE"),
            ("algorithm", "Ed25519"), ("key_epoch", 1),
            ("signer_key_id", key.key_id),
        )
        preimage = (
            b"HEGEL/PHASE2B/C1/LOGICAL_STAGE_AUTHORIZATION/ROLE_ACCEPTANCE/V2\0"
            + _oracle_named_pairs((
                ("signature_header", header), ("body", amendment_pairs),
            ))
        )
        signature = private_key.sign(preimage)
        assert hashlib.sha256(preimage).hexdigest() == (
            AUTHORIZATION_PREIMAGE_SHA256[purpose_id - 1]
        )
        assert signature.hex() == expected_signature
        assert signature == graph.registry.role_acceptance_signatures[purpose_id - 1].signature
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_hex)).verify(
            bytes.fromhex(expected_signature), preimage
        )
        oracle_preimages.append(preimage)
        oracle_signature_rows.append((
            ("purpose_id", purpose_id), ("role", role),
            ("signature_purpose", "LOGICAL_STAGE_AUTHORIZATION_AMENDMENT_ACCEPTANCE"),
            ("algorithm", "Ed25519"), ("key_epoch", 1),
            ("signer_key_id", key.key_id), ("signature_hex", expected_signature),
        ))

    registry_id_pairs = (
        ("body", amendment_pairs),
        ("authorization_amendment_id", oracle_amendment_id),
        ("role_acceptance_signatures", tuple(oracle_signature_rows)),
    )
    oracle_registry_id = _oracle_id(
        "phase2b_c1_owner_accepted_execution_registry_v2_",
        b"HEGEL/PHASE2B/C1/LOGICAL_STAGE_AUTHORIZATION/REGISTRY_ID/V2\0",
        registry_id_pairs,
    )
    assert oracle_registry_id == (
        "phase2b_c1_owner_accepted_execution_registry_v2_"
        "c15f6bf40ba754fec13153d250b1ee4badda5df584cee733950adb22f63b76ef"
    )
    oracle_registry_bytes = _oracle_named_pairs(
        registry_id_pairs + (("execution_registry_id", oracle_registry_id),)
    )
    assert hashlib.sha256(oracle_registry_bytes).hexdigest() == (
        "4ee214f485353525dc03fb462bfb60d75acada2f6ab2cf48fad8dd40d69a4a34"
    )
    assert graph.store.body.stored_content_sha256 == hashlib.sha256(
        oracle_registry_bytes
    ).hexdigest()

    store_pairs = _oracle_object_pairs(graph.store.body, STORE_BODY_FIELDS)
    store_key = legacy.registry.body.role_keys[2]
    store_header = (
        ("purpose_id", 3), ("role", "STORE_RECEIPT"),
        ("signature_purpose", "STORE_IF_ABSENT_RECEIPT"),
        ("algorithm", "Ed25519"), ("key_epoch", 1),
        ("signer_key_id", store_key.key_id),
    )
    oracle_store_preimage = (
        b"HEGEL/PHASE2B/C1/LOGICAL_STAGE_AUTHORIZATION/STORE_IF_ABSENT/V2\0"
        + _oracle_named_pairs((
            ("signature_header", store_header), ("body", store_pairs),
        ))
    )
    assert hashlib.sha256(oracle_store_preimage).hexdigest() == (
        "25934b5719d132e7448949fb2260a01103c8b15fe32a4fa819659040f79eb19c"
    )
    oracle_store_signature = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(PRIVATE_SEED_HEX[2])
    ).sign(oracle_store_preimage)
    assert oracle_store_signature.hex() == STORE_SIGNATURE_HEX
    assert oracle_store_signature == graph.store.store_signature.signature
    store_signature_pairs = (
        ("purpose_id", 3), ("role", "STORE_RECEIPT"),
        ("signature_purpose", "STORE_IF_ABSENT_RECEIPT"),
        ("algorithm", "Ed25519"), ("key_epoch", 1),
        ("signer_key_id", store_key.key_id),
        ("signature_hex", STORE_SIGNATURE_HEX),
    )
    oracle_store_id = _oracle_id(
        "phase2b_c1_logical_stage_authorization_if_absent_receipt_v2_",
        b"HEGEL/PHASE2B/C1/LOGICAL_STAGE_AUTHORIZATION/STORE_RECEIPT_ID/V2\0",
        (("body", store_pairs), ("store_signature", store_signature_pairs)),
    )
    assert oracle_store_id == (
        "phase2b_c1_logical_stage_authorization_if_absent_receipt_v2_"
        "c804c5d84a524403677705cfaa7463f8f6f5f3886911416a070436471581595f"
    )
    assert oracle_store_id == graph.store.store_receipt_id

    assert authority.phase2b_c1_logical_stage_authority_context_id_v2(
        graph.context.body
    ) == oracle_context_id
    assert authority.phase2b_c1_logical_stage_authorization_amendment_id_v2(
        graph.registry.body
    ) == oracle_amendment_id
    for purpose_id, (role, key, oracle_preimage) in enumerate(zip(
        six.PHASE2B_C1_SIX_ROLE_WIRE_V1_ROLE_ORDER,
        legacy.registry.body.role_keys, oracle_preimages, strict=True,
    ), start=1):
        assert authority.phase2b_c1_logical_stage_authorization_signature_preimage_v2(
            graph.registry.body, purpose_id=purpose_id, role=role,
            signer_key_id=key.key_id,
        ) == oracle_preimage
    assert authority.phase2b_c1_owner_accepted_execution_registry_id_v2(
        graph.registry.body, graph.registry.authorization_amendment_id,
        graph.registry.role_acceptance_signatures,
    ) == oracle_registry_id
    assert authority.canonical_phase2b_c1_owner_accepted_execution_registry_bytes_v2(
        graph.registry
    ) == oracle_registry_bytes
    assert authority.phase2b_c1_logical_stage_authorization_store_signature_preimage_v2(
        graph.store.body, signer_key_id=store_key.key_id
    ) == oracle_store_preimage
    assert authority.phase2b_c1_logical_stage_authorization_store_receipt_id_v2(
        graph.store.body, graph.store.store_signature
    ) == oracle_store_id


@pytest.mark.parametrize("mode", ("missing", "reordered", "bad_signature"))
def test_unanimous_six_signature_layout_is_fail_closed(mode: str) -> None:
    legacy = _build_legacy()
    graph = _build_authority(legacy)
    signatures = graph.registry.role_acceptance_signatures
    if mode == "missing":
        signatures = signatures[:-1]
    elif mode == "reordered":
        signatures = (signatures[1], signatures[0], *signatures[2:])
    else:
        signatures = (
            replace(signatures[0], signature=b"\x00" * 64), *signatures[1:]
        )
    registry = replace(
        graph.registry, role_acceptance_signatures=signatures,
        execution_registry_id=authority.phase2b_c1_owner_accepted_execution_registry_id_v2(
            graph.registry.body, graph.registry.authorization_amendment_id,
            signatures,
        ),
    )
    candidate = replace(graph, registry=registry)
    if mode == "bad_signature":
        candidate = _reseal_store(legacy, graph, registry)
    result = _verify(legacy, candidate)
    assert result.disposition is authority.Phase2BC1LogicalStageAuthorityDispositionV2.REJECTED
    if mode == "bad_signature":
        assert result.reason is authority.Phase2BC1LogicalStageAuthorityReasonV2.SIGNATURE_INVALID


def test_contract_identity_mutation_and_cross_campaign_splice_reject_atomically() -> None:
    legacy = _build_legacy()
    graph = _build_authority(legacy)
    changed_body = replace(
        graph.context.body,
        logical_stage_wire_policy_id=(
            "phase2b_c1_logical_stage_wire_policy_v2_" + "0" * 64
        ),
    )
    changed_context = replace(
        graph.context, body=changed_body,
        authority_context_id=authority.phase2b_c1_logical_stage_authority_context_id_v2(
            changed_body
        ),
    )
    rejected = _verify(legacy, replace(graph, context=changed_context))
    assert rejected.result_id is None
    assert rejected.partial_output_published is False
    other_legacy = _build_legacy(campaign_hash=H[8])
    assert _verify(other_legacy, graph).disposition is authority.Phase2BC1LogicalStageAuthorityDispositionV2.REJECTED

    other_graph = _build_authority(other_legacy)
    spliced_registry = replace(
        graph.registry,
        role_acceptance_signatures=other_graph.registry.role_acceptance_signatures,
        execution_registry_id=authority.phase2b_c1_owner_accepted_execution_registry_id_v2(
            graph.registry.body, graph.registry.authorization_amendment_id,
            other_graph.registry.role_acceptance_signatures,
        ),
    )
    spliced = _reseal_store(legacy, graph, spliced_registry)
    splice_result = _verify(legacy, spliced)
    assert splice_result.reason is authority.Phase2BC1LogicalStageAuthorityReasonV2.SIGNATURE_INVALID


def test_store_generation_object_key_and_content_mutations_reject() -> None:
    legacy = _build_legacy()
    graph = _build_authority(legacy)
    for changed in (
        replace(graph.store.body, new_generation=2),
        replace(graph.store.body, object_key=graph.store.body.object_key + "-other"),
        replace(graph.store.body, stored_content_sha256="0" * 64),
    ):
        signature = _legacy_signature(
            key=legacy.private_keys[2],
            key_id=legacy.registry.body.role_keys[2].key_id, purpose_id=3,
            role=six.Phase2BC1SixRoleV1.STORE_RECEIPT,
            signature_purpose=six.Phase2BC1SignaturePurposeV1.STORE_IF_ABSENT_RECEIPT,
            preimage=authority.phase2b_c1_logical_stage_authorization_store_signature_preimage_v2(
                changed, signer_key_id=legacy.registry.body.role_keys[2].key_id
            ),
        )
        store = authority.Phase2BC1LogicalStageAuthorizationStoreReceiptCandidateV2(
            body=changed, store_signature=signature,
            store_receipt_id=authority.phase2b_c1_logical_stage_authorization_store_receipt_id_v2(
                changed, signature
            ),
        )
        assert _verify(legacy, replace(graph, store=store)).disposition is authority.Phase2BC1LogicalStageAuthorityDispositionV2.REJECTED


def test_complete_snapshot_does_not_reread_original_after_legacy_verification(monkeypatch) -> None:
    legacy = _build_legacy()
    graph = _build_authority(legacy)
    original = six.verify_phase2b_c1_six_role_wire_v1

    def mutate_original_then_verify(**kwargs):
        object.__setattr__(
            graph.context.body, "logical_stage_wire_policy_id",
            "phase2b_c1_logical_stage_wire_policy_v2_" + "0" * 64,
        )
        return original(**kwargs)

    monkeypatch.setattr(six, "verify_phase2b_c1_six_role_wire_v1", mutate_original_then_verify)
    assert _verify(legacy, graph).disposition is authority.Phase2BC1LogicalStageAuthorityDispositionV2.CALLER_SUPPLIED_OWNER_ACCEPTED_V2_AUTHORITY_GRAPH_VERIFIED_NOT_EXECUTION


def test_backend_unavailable_and_signature_invalid_are_public_rejections(monkeypatch) -> None:
    legacy = _build_legacy()
    graph = _build_authority(legacy)
    monkeypatch.setattr(authority, "_Ed25519PublicKey", None)
    result = _verify(legacy, graph)
    assert result.reason is authority.Phase2BC1LogicalStageAuthorityReasonV2.SIGNATURE_BACKEND_UNAVAILABLE
    assert result.partial_output_published is False


def test_wrapper_internally_reverifies_authority_and_committed_prefix_without_promoting_legacy_flags(monkeypatch) -> None:
    legacy = _build_legacy()
    graph = _build_authority(legacy)
    projections, tokens = _raw_prefix(legacy)
    authority_calls = 0
    logical_calls = 0
    original_authority = authority.verify_phase2b_c1_logical_stage_authority_graph_v2
    original_logical = logical.verify_phase2b_c1_logical_stage_prefix_v1

    def count_authority(**kwargs):
        nonlocal authority_calls
        authority_calls += 1
        return original_authority(**kwargs)

    def count_logical(**kwargs):
        nonlocal logical_calls
        logical_calls += 1
        return original_logical(**kwargs)

    monkeypatch.setattr(authority, "verify_phase2b_c1_logical_stage_authority_graph_v2", count_authority)
    monkeypatch.setattr(logical, "verify_phase2b_c1_logical_stage_prefix_v1", count_logical)
    result = authority.verify_authorized_phase2b_c1_logical_stage_prefix_v2(
        legacy_context=legacy.context, legacy_registry=legacy.registry,
        legacy_store_receipt=legacy.store,
        legacy_logical_genesis_token=legacy.genesis,
        authority_context=graph.context, execution_registry=graph.registry,
        authorization_store_receipt=graph.store,
        event_projections=projections, stage_tokens=tokens,
    )
    assert authority_calls == logical_calls == 1
    assert result.disposition is authority.Phase2BC1AuthorizedLogicalStagePrefixDispositionV2.OWNER_ACCEPTED_V2_POLICY_AND_CALLER_SUPPLIED_V1_PREFIX_VERIFIED_NOT_EXECUTION
    assert result.v2_policy_admission_verified is True
    assert result.v1_legacy_false_fields_preserved is True
    for name in authority._FALSE_CLAIMS:
        assert getattr(result, name) is False
    legacy_result = original_logical(
        context=legacy.context, registry=legacy.registry,
        store_receipt=legacy.store, logical_genesis_token=legacy.genesis,
        event_projections=projections, stage_tokens=tokens,
    )
    assert legacy_result.legacy_registry_policy_authorized is False
    assert legacy_result.actual_campaign_eligible is False
    assert legacy_result.execution_authorized is False


def test_wrapper_rejects_cross_prefix_and_private_results_cannot_be_forged() -> None:
    legacy = _build_legacy()
    graph = _build_authority(legacy)
    other = _build_legacy(campaign_hash=H[8])
    projections, tokens = _raw_prefix(other)
    result = authority.verify_authorized_phase2b_c1_logical_stage_prefix_v2(
        legacy_context=legacy.context, legacy_registry=legacy.registry,
        legacy_store_receipt=legacy.store,
        legacy_logical_genesis_token=legacy.genesis,
        authority_context=graph.context, execution_registry=graph.registry,
        authorization_store_receipt=graph.store,
        event_projections=projections, stage_tokens=tokens,
    )
    assert result.disposition is authority.Phase2BC1AuthorizedLogicalStagePrefixDispositionV2.REJECTED
    assert result.stage_token_ids == ()
    with pytest.raises(TypeError, match="privately issued"):
        authority.Phase2BC1LogicalStageAuthorityVerificationV2()
    with pytest.raises(TypeError, match="privately issued"):
        authority.Phase2BC1AuthorizedLogicalStagePrefixVerificationV2()


def test_public_surface_is_narrow_and_no_operation_modules_are_imported() -> None:
    assert "subprocess" not in authority.__dict__
    assert "pathlib" not in authority.__dict__
    assert "os" not in authority.__dict__
    assert "time" not in authority.__dict__
    assert set(authority.__all__) == EXPECTED_PUBLIC_SURFACE
    assert len(authority.__all__) == len(EXPECTED_PUBLIC_SURFACE)
