from __future__ import annotations

import ast
from dataclasses import dataclass, fields, replace
import hashlib
import inspect
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from hegel_machine import phase2b_c1_six_role_wire_v1 as wire


SHA_A = "1" * 64
SHA_B = "2" * 64
SHA_C = "3" * 64
SHA_D = "4" * 64
SHA_E = "5" * 64
SHA_F = "6" * 64
SHA_G = "7" * 64
SHA_H = "8" * 64
LEGACY_SCORER_ROOT = (
    "3f28fcf1f5669859350d6297a8ab45e81740e5dea1a375f5257ea278f6ca0b6a"
)


EXPECTED_PUBLIC_SURFACE = (
    "PHASE2B_C1_SIX_ROLE_WIRE_V1_VERSION",
    "PHASE2B_C1_SIX_ROLE_WIRE_V1_CLAIM_LEVEL",
    "PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_CLASS",
    "PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_REQUIREMENT_ID",
    "PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CONTENT_ID",
    "PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CANONICAL_SHA256",
    "PHASE2B_C1_SIX_ROLE_WIRE_V1_FROZEN_PROTOCOL_ID",
    "PHASE2B_C1_SIX_ROLE_WIRE_V1_FROZEN_EXACT_FREEZE_ID",
    "PHASE2B_C1_SIX_ROLE_WIRE_V1_SCORER_SOURCE_ROOT_SHA256",
    "PHASE2B_C1_SIX_ROLE_WIRE_V1_ED25519_BACKEND_AVAILABLE",
    "PHASE2B_C1_SIX_ROLE_WIRE_V1_KEY_EPOCH",
    "PHASE2B_C1_SIX_ROLE_WIRE_V1_STORE_TRANSITION",
    "PHASE2B_C1_SIX_ROLE_WIRE_V1_TIME_SEMANTICS",
    "PHASE2B_C1_SIX_ROLE_WIRE_V1_ROLE_ORDER",
    "Phase2BC1SixRoleV1",
    "Phase2BC1SignaturePurposeV1",
    "Phase2BC1LogicalStageV1",
    "Phase2BC1SixRoleWireDispositionV1",
    "Phase2BC1SixRoleWireReasonV1",
    "Phase2BC1CeremonyContextBodyV1",
    "Phase2BC1CeremonyContextV1",
    "Phase2BC1RoleKeyV1",
    "Phase2BC1DetachedSignatureV1",
    "Phase2BC1ExecutionRegistryBodyV1",
    "Phase2BC1ExecutionRegistryCandidateV1",
    "Phase2BC1StoreIfAbsentReceiptBodyV1",
    "Phase2BC1StoreIfAbsentReceiptCandidateV1",
    "Phase2BC1LogicalGenesisTokenBodyV1",
    "Phase2BC1LogicalGenesisTokenCandidateV1",
    "Phase2BC1SixRoleWireVerificationV1",
    "phase2b_c1_attempt_id_v1",
    "phase2b_c1_role_key_id_v1",
    "phase2b_c1_ceremony_context_id_v1",
    "canonical_phase2b_c1_ceremony_context_bytes_v1",
    "phase2b_c1_governance_registry_signature_preimage_v1",
    "phase2b_c1_execution_registry_id_v1",
    "canonical_phase2b_c1_execution_registry_bytes_v1",
    "phase2b_c1_store_receipt_signature_preimage_v1",
    "phase2b_c1_store_receipt_id_v1",
    "canonical_phase2b_c1_store_receipt_bytes_v1",
    "phase2b_c1_logical_genesis_signature_preimage_v1",
    "phase2b_c1_logical_genesis_token_id_v1",
    "canonical_phase2b_c1_logical_genesis_token_bytes_v1",
    "verify_phase2b_c1_six_role_wire_v1",
)


TRUE_CLAIMS = (
    "complete_graph_snapshotted_before_hash_or_signature",
    "all_content_addresses_checked_before_signature_verification",
    "ceremony_context_content_address_verified",
    "ordered_six_role_key_registry_verified",
    "six_distinct_public_keys_verified",
    "declared_purpose_private_key_layout_verified",
    "governance_registry_signature_verified",
    "execution_registry_content_address_verified",
    "registry_image_source_context_binding_verified",
    "store_if_absent_zero_to_one_transition_verified",
    "store_receipt_signature_verified",
    "store_receipt_content_address_verified",
    "logical_genesis_shape_verified",
    "logical_genesis_signature_verified",
    "logical_genesis_content_address_verified",
    "graph_cross_bindings_verified",
    "verification_result_privately_issued",
)


FALSE_CLAIMS = (
    "actual_execution_claimed",
    "docker_invoked",
    "filesystem_read",
    "filesystem_written",
    "clock_read",
    "keys_generated",
    "signatures_generated",
    "fresh_keys_generated",
    "purpose_private_key_custody_verified",
    "caller_supplied_key_registry_authoritative",
    "governance_identity_verified",
    "registry_issuance_authority_verified",
    "image_runtime_identity_observed",
    "source_root_runtime_identity_observed",
    "actor_qualification_executed",
    "technical_role_independence_live_verified",
    "store_operation_executed",
    "store_durability_verified",
    "store_rollback_resistance_verified",
    "cross_process_no_replace_verified",
    "cross_host_global_one_shot_verified",
    "one_shot_policy_enforced",
    "time_operation_executed",
    "authoritative_time_verified",
    "utc_time_verified",
    "timeline_observed",
    "recognizer_executed",
    "runtime_executed",
    "actual_960_case_run_verified",
    "scoring_performed",
    "prediction_scored",
    "actual_prediction_scoring_evidence",
    "formal_gate_evaluation_performed",
    "effect_evidence",
    "formal_c1_report_verified",
    "c1_exit_evidence",
    "organizational_independence_verified",
    "independent_human_actors_verified",
    "malicious_host_root_resistance",
    "malicious_docker_daemon_resistance",
    "owner_rollback_deletion_resistance",
    "remote_attestation_verified",
    "hardware_key_nonexportability_verified",
    "execution_authorized",
)


@dataclass(frozen=True)
class SuppliedGraph:
    private_keys: tuple[Ed25519PrivateKey, ...]
    context: wire.Phase2BC1CeremonyContextV1
    registry: wire.Phase2BC1ExecutionRegistryCandidateV1
    store_receipt: wire.Phase2BC1StoreIfAbsentReceiptCandidateV1
    time_token: wire.Phase2BC1LogicalGenesisTokenCandidateV1


def _public_key(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )


def _signature(
    *, purpose_id: int, role: wire.Phase2BC1SixRoleV1,
    signature_purpose: wire.Phase2BC1SignaturePurposeV1,
    signer_key_id: str, signature: bytes,
) -> wire.Phase2BC1DetachedSignatureV1:
    return wire.Phase2BC1DetachedSignatureV1(
        purpose_id=purpose_id,
        role=role,
        signature_purpose=signature_purpose,
        algorithm="Ed25519",
        key_epoch=wire.PHASE2B_C1_SIX_ROLE_WIRE_V1_KEY_EPOCH,
        signer_key_id=signer_key_id,
        signature=signature,
    )


def _build_graph() -> SuppliedGraph:
    private_keys = tuple(
        Ed25519PrivateKey.from_private_bytes(bytes([index]) * 32)
        for index in range(1, 7)
    )
    campaign_id = "phase2b_actual_unsealed_960_campaign_v2_" + SHA_A
    execution_freeze_id = "phase2b_execution_freeze_" + SHA_B
    attempt_id = wire.phase2b_c1_attempt_id_v1(
        authority_requirement_id=(
            wire.PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_REQUIREMENT_ID
        ),
        actor_profile_content_id=(
            wire.PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CONTENT_ID
        ),
        actor_profile_canonical_sha256=(
            wire.PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CANONICAL_SHA256
        ),
        campaign_id=campaign_id,
        protocol_id=wire.PHASE2B_C1_SIX_ROLE_WIRE_V1_FROZEN_PROTOCOL_ID,
        exact_freeze_id=wire.PHASE2B_C1_SIX_ROLE_WIRE_V1_FROZEN_EXACT_FREEZE_ID,
        execution_freeze_manifest_id=execution_freeze_id,
        execution_freeze_manifest_sha256=SHA_C,
        attempt_nonce_sha256=SHA_D,
    )
    context_body = wire.Phase2BC1CeremonyContextBodyV1(
        version=wire.PHASE2B_C1_SIX_ROLE_WIRE_V1_VERSION,
        authority_class=wire.PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_CLASS,
        authority_requirement_id=(
            wire.PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_REQUIREMENT_ID
        ),
        actor_profile_content_id=(
            wire.PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CONTENT_ID
        ),
        actor_profile_canonical_sha256=(
            wire.PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CANONICAL_SHA256
        ),
        campaign_id=campaign_id,
        protocol_id=wire.PHASE2B_C1_SIX_ROLE_WIRE_V1_FROZEN_PROTOCOL_ID,
        exact_freeze_id=wire.PHASE2B_C1_SIX_ROLE_WIRE_V1_FROZEN_EXACT_FREEZE_ID,
        execution_freeze_manifest_id=execution_freeze_id,
        execution_freeze_manifest_sha256=SHA_C,
        attempt_nonce_sha256=SHA_D,
        attempt_id=attempt_id,
        recognizer_image_digest="sha256:" + SHA_E,
        recognizer_source_root_sha256=SHA_F,
        scorer_image_digest="sha256:" + SHA_G,
        scorer_source_root_sha256=(
            wire.PHASE2B_C1_SIX_ROLE_WIRE_V1_SCORER_SOURCE_ROOT_SHA256
        ),
    )
    context = wire.Phase2BC1CeremonyContextV1(
        body=context_body,
        context_id=wire.phase2b_c1_ceremony_context_id_v1(context_body),
    )
    role_keys = tuple(
        wire.Phase2BC1RoleKeyV1(
            purpose_id=index,
            role=role,
            algorithm="Ed25519",
            key_epoch=wire.PHASE2B_C1_SIX_ROLE_WIRE_V1_KEY_EPOCH,
            purpose_private=True,
            public_key=_public_key(private_keys[index - 1]),
            key_id=wire.phase2b_c1_role_key_id_v1(
                purpose_id=index,
                role=role,
                public_key=_public_key(private_keys[index - 1]),
            ),
        )
        for index, role in enumerate(
            wire.PHASE2B_C1_SIX_ROLE_WIRE_V1_ROLE_ORDER, start=1
        )
    )
    registry_body = wire.Phase2BC1ExecutionRegistryBodyV1(
        version=context_body.version,
        context_id=context.context_id,
        authority_class=context_body.authority_class,
        authority_requirement_id=context_body.authority_requirement_id,
        actor_profile_content_id=context_body.actor_profile_content_id,
        actor_profile_canonical_sha256=(
            context_body.actor_profile_canonical_sha256
        ),
        campaign_id=context_body.campaign_id,
        protocol_id=context_body.protocol_id,
        exact_freeze_id=context_body.exact_freeze_id,
        execution_freeze_manifest_id=(
            context_body.execution_freeze_manifest_id
        ),
        execution_freeze_manifest_sha256=(
            context_body.execution_freeze_manifest_sha256
        ),
        attempt_nonce_sha256=context_body.attempt_nonce_sha256,
        attempt_id=context_body.attempt_id,
        recognizer_image_digest=context_body.recognizer_image_digest,
        recognizer_source_root_sha256=(
            context_body.recognizer_source_root_sha256
        ),
        scorer_image_digest=context_body.scorer_image_digest,
        scorer_source_root_sha256=context_body.scorer_source_root_sha256,
        role_keys=role_keys,
    )
    governance_signature = _signature(
        purpose_id=1,
        role=wire.Phase2BC1SixRoleV1.GOVERNANCE_ANCHOR,
        signature_purpose=(
            wire.Phase2BC1SignaturePurposeV1.GOVERNANCE_CHECKPOINT
        ),
        signer_key_id=role_keys[0].key_id,
        signature=private_keys[0].sign(
            wire.phase2b_c1_governance_registry_signature_preimage_v1(
                registry_body, signer_key_id=role_keys[0].key_id
            )
        ),
    )
    registry = wire.Phase2BC1ExecutionRegistryCandidateV1(
        body=registry_body,
        governance_signature=governance_signature,
        execution_registry_id=wire.phase2b_c1_execution_registry_id_v1(
            registry_body, governance_signature
        ),
    )
    store_body = wire.Phase2BC1StoreIfAbsentReceiptBodyV1(
        version=context_body.version,
        context_id=context.context_id,
        authority_requirement_id=context_body.authority_requirement_id,
        actor_profile_content_id=context_body.actor_profile_content_id,
        campaign_id=context_body.campaign_id,
        attempt_id=context_body.attempt_id,
        execution_registry_id=registry.execution_registry_id,
        object_key=(
            "phase2b/c1/actual-unsealed-960/"
            f"{context_body.campaign_id}/attempt-0/execution-registry"
        ),
        previous_generation=0,
        new_generation=1,
        condition="IF_ABSENT",
        previous_content_sha256_or_null=None,
        stored_content_sha256=hashlib.sha256(
            wire.canonical_phase2b_c1_execution_registry_bytes_v1(registry)
        ).hexdigest(),
    )
    store_signature = _signature(
        purpose_id=3,
        role=wire.Phase2BC1SixRoleV1.STORE_RECEIPT,
        signature_purpose=(
            wire.Phase2BC1SignaturePurposeV1.STORE_IF_ABSENT_RECEIPT
        ),
        signer_key_id=role_keys[2].key_id,
        signature=private_keys[2].sign(
            wire.phase2b_c1_store_receipt_signature_preimage_v1(
                store_body, signer_key_id=role_keys[2].key_id
            )
        ),
    )
    store_receipt = wire.Phase2BC1StoreIfAbsentReceiptCandidateV1(
        body=store_body,
        store_signature=store_signature,
        store_receipt_id=wire.phase2b_c1_store_receipt_id_v1(
            store_body, store_signature
        ),
    )
    time_body = wire.Phase2BC1LogicalGenesisTokenBodyV1(
        version=context_body.version,
        context_id=context.context_id,
        authority_requirement_id=context_body.authority_requirement_id,
        actor_profile_content_id=context_body.actor_profile_content_id,
        campaign_id=context_body.campaign_id,
        attempt_id=context_body.attempt_id,
        execution_registry_id=registry.execution_registry_id,
        store_receipt_id=store_receipt.store_receipt_id,
        time_semantics=wire.PHASE2B_C1_SIX_ROLE_WIRE_V1_TIME_SEMANTICS,
        previous_logical_sequence_or_null=None,
        logical_sequence=0,
        stage=wire.Phase2BC1LogicalStageV1.PREREGISTERED,
        stage_index=0,
    )
    time_signature = _signature(
        purpose_id=4,
        role=wire.Phase2BC1SixRoleV1.TIME_ATTESTOR,
        signature_purpose=wire.Phase2BC1SignaturePurposeV1.DECLARED_TIME_TOKEN,
        signer_key_id=role_keys[3].key_id,
        signature=private_keys[3].sign(
            wire.phase2b_c1_logical_genesis_signature_preimage_v1(
                time_body, signer_key_id=role_keys[3].key_id
            )
        ),
    )
    time_token = wire.Phase2BC1LogicalGenesisTokenCandidateV1(
        body=time_body,
        time_signature=time_signature,
        logical_genesis_token_id=wire.phase2b_c1_logical_genesis_token_id_v1(
            time_body, time_signature
        ),
    )
    return SuppliedGraph(
        private_keys=private_keys,
        context=context,
        registry=registry,
        store_receipt=store_receipt,
        time_token=time_token,
    )


@pytest.fixture
def graph() -> SuppliedGraph:
    return _build_graph()


def _verify(graph: SuppliedGraph) -> wire.Phase2BC1SixRoleWireVerificationV1:
    return wire.verify_phase2b_c1_six_role_wire_v1(
        context=graph.context,
        registry=graph.registry,
        store_receipt=graph.store_receipt,
        logical_genesis_token=graph.time_token,
    )


def _rebind_store_and_time(
    graph: SuppliedGraph,
    registry: wire.Phase2BC1ExecutionRegistryCandidateV1,
) -> SuppliedGraph:
    keys = registry.body.role_keys
    store_body = replace(
        graph.store_receipt.body,
        execution_registry_id=registry.execution_registry_id,
        stored_content_sha256=hashlib.sha256(
            wire.canonical_phase2b_c1_execution_registry_bytes_v1(registry)
        ).hexdigest(),
    )
    store_signature = _signature(
        purpose_id=3,
        role=wire.Phase2BC1SixRoleV1.STORE_RECEIPT,
        signature_purpose=(
            wire.Phase2BC1SignaturePurposeV1.STORE_IF_ABSENT_RECEIPT
        ),
        signer_key_id=keys[2].key_id,
        signature=graph.private_keys[2].sign(
            wire.phase2b_c1_store_receipt_signature_preimage_v1(
                store_body, signer_key_id=keys[2].key_id
            )
        ),
    )
    store = wire.Phase2BC1StoreIfAbsentReceiptCandidateV1(
        body=store_body,
        store_signature=store_signature,
        store_receipt_id=wire.phase2b_c1_store_receipt_id_v1(
            store_body, store_signature
        ),
    )
    time_body = replace(
        graph.time_token.body,
        execution_registry_id=registry.execution_registry_id,
        store_receipt_id=store.store_receipt_id,
    )
    time_signature = _signature(
        purpose_id=4,
        role=wire.Phase2BC1SixRoleV1.TIME_ATTESTOR,
        signature_purpose=wire.Phase2BC1SignaturePurposeV1.DECLARED_TIME_TOKEN,
        signer_key_id=keys[3].key_id,
        signature=graph.private_keys[3].sign(
            wire.phase2b_c1_logical_genesis_signature_preimage_v1(
                time_body, signer_key_id=keys[3].key_id
            )
        ),
    )
    time_token = wire.Phase2BC1LogicalGenesisTokenCandidateV1(
        body=time_body,
        time_signature=time_signature,
        logical_genesis_token_id=wire.phase2b_c1_logical_genesis_token_id_v1(
            time_body, time_signature
        ),
    )
    return replace(
        graph, registry=registry, store_receipt=store, time_token=time_token
    )


def _rebind_time(
    graph: SuppliedGraph,
    store: wire.Phase2BC1StoreIfAbsentReceiptCandidateV1,
) -> SuppliedGraph:
    key = graph.registry.body.role_keys[3]
    time_body = replace(graph.time_token.body, store_receipt_id=store.store_receipt_id)
    signature = _signature(
        purpose_id=4,
        role=wire.Phase2BC1SixRoleV1.TIME_ATTESTOR,
        signature_purpose=wire.Phase2BC1SignaturePurposeV1.DECLARED_TIME_TOKEN,
        signer_key_id=key.key_id,
        signature=graph.private_keys[3].sign(
            wire.phase2b_c1_logical_genesis_signature_preimage_v1(
                time_body, signer_key_id=key.key_id
            )
        ),
    )
    time = wire.Phase2BC1LogicalGenesisTokenCandidateV1(
        body=time_body,
        time_signature=signature,
        logical_genesis_token_id=wire.phase2b_c1_logical_genesis_token_id_v1(
            time_body, signature
        ),
    )
    return replace(graph, store_receipt=store, time_token=time)


def test_public_surface_is_exact_and_frozen() -> None:
    assert wire.__all__ == EXPECTED_PUBLIC_SURFACE
    assert set(EXPECTED_PUBLIC_SURFACE) == {
        name for name in dir(wire) if name in EXPECTED_PUBLIC_SURFACE
    }


def test_exact_profile_requirement_protocol_freeze_and_scorer_root() -> None:
    assert wire.PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_REQUIREMENT_ID.endswith(
        "53631d563f8da8f26c7202b7dd0681be5423fe2ca5c15e9a9f5deaa2920d9940"
    )
    assert wire.PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CONTENT_ID.endswith(
        "b2f081771f3de8d2a531b08b58109d95c51815904e9612e66f1bbfe69babf906"
    )
    assert wire.PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CANONICAL_SHA256 == (
        "8ca1850f026dabb7a4b69add0f89b33a3fcfb789223b2fd95ef0379608adbeed"
    )
    assert wire.PHASE2B_C1_SIX_ROLE_WIRE_V1_SCORER_SOURCE_ROOT_SHA256 == (
        "34f4a2404d00d46cede832d34ca9713253da93f1becef4a1d302c1411d6bd5e7"
    )
    assert wire.PHASE2B_C1_SIX_ROLE_WIRE_V1_SCORER_SOURCE_ROOT_SHA256 != (
        LEGACY_SCORER_ROOT
    )


def test_six_role_order_key_epoch_store_and_time_state_machine_are_exact() -> None:
    assert tuple(role.value for role in wire.PHASE2B_C1_SIX_ROLE_WIRE_V1_ROLE_ORDER) == (
        "GOVERNANCE_ANCHOR",
        "CUSTODIAN",
        "STORE_RECEIPT",
        "TIME_ATTESTOR",
        "RECOGNIZER",
        "SCORER_EVALUATOR",
    )
    assert wire.PHASE2B_C1_SIX_ROLE_WIRE_V1_KEY_EPOCH == 1
    assert wire.PHASE2B_C1_SIX_ROLE_WIRE_V1_STORE_TRANSITION == (
        0,
        1,
        "IF_ABSENT",
    )
    assert tuple(stage.value for stage in wire.Phase2BC1LogicalStageV1) == (
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


def test_valid_caller_supplied_graph_verifies_only_narrow_mechanics(
    graph: SuppliedGraph,
) -> None:
    result = _verify(graph)
    assert result.disposition is (
        wire.Phase2BC1SixRoleWireDispositionV1.CALLER_SUPPLIED_SIX_ROLE_WIRE_GRAPH_VERIFIED_NOT_AUTHORITY
    )
    assert result.reason is (
        wire.Phase2BC1SixRoleWireReasonV1.CALLER_SUPPLIED_GRAPH_SIGNATURE_AND_CONTENT_MECHANICS_VERIFIED
    )
    assert result.context_id == graph.context.context_id
    assert result.attempt_id == graph.context.body.attempt_id
    assert result.execution_registry_id == graph.registry.execution_registry_id
    assert result.store_receipt_id == graph.store_receipt.store_receipt_id
    assert result.logical_genesis_token_id == (
        graph.time_token.logical_genesis_token_id
    )
    assert result.role_key_ids == tuple(
        key.key_id for key in graph.registry.body.role_keys
    )
    assert result.result_id is not None
    assert result.partial_output_published is False
    assert all(getattr(result, name) is True for name in TRUE_CLAIMS)
    assert all(getattr(result, name) is False for name in FALSE_CLAIMS)


def test_results_are_privately_issued() -> None:
    with pytest.raises(TypeError, match="privately issued"):
        wire.Phase2BC1SixRoleWireVerificationV1()


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("version", "cross-version"),
        ("authority_class", "UNTRUSTED_CALLER"),
        (
            "authority_requirement_id",
            "phase2b_c1_owner_accepted_container_authority_requirement_v1_" + SHA_H,
        ),
        (
            "actor_profile_content_id",
            "phase2b_c1_container_actor_profile_v2_" + SHA_H,
        ),
        ("actor_profile_canonical_sha256", SHA_H),
        ("protocol_id", "phase2b_protocol_" + SHA_H),
        ("exact_freeze_id", "phase2b_exact_freeze_" + SHA_H),
        ("scorer_source_root_sha256", LEGACY_SCORER_ROOT),
        ("scorer_source_root_sha256", SHA_H),
    ),
)
def test_context_frozen_identity_tamper_is_rejected_before_crypto(
    graph: SuppliedGraph, field: str, replacement: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = replace(graph.context.body, **{field: replacement})
    context = wire.Phase2BC1CeremonyContextV1(
        body=body,
        context_id=wire.phase2b_c1_ceremony_context_id_v1(body),
    )
    calls = 0

    class BombBackend:
        @classmethod
        def from_public_bytes(cls, payload: bytes) -> "BombBackend":
            nonlocal calls
            calls += 1
            raise AssertionError(payload)

    monkeypatch.setattr(wire, "_Ed25519PublicKey", BombBackend)
    result = wire.verify_phase2b_c1_six_role_wire_v1(
        context=context,
        registry=graph.registry,
        store_receipt=graph.store_receipt,
        logical_genesis_token=graph.time_token,
    )
    assert result.reason is wire.Phase2BC1SixRoleWireReasonV1.CONTEXT_INVALID
    assert calls == 0


def test_attempt_id_binds_requirement_profile_campaign_protocol_and_freezes() -> None:
    graph = _build_graph()
    body = graph.context.body
    baseline = body.attempt_id
    values = {
        "authority_requirement_id": body.authority_requirement_id,
        "actor_profile_content_id": body.actor_profile_content_id,
        "actor_profile_canonical_sha256": body.actor_profile_canonical_sha256,
        "campaign_id": body.campaign_id,
        "protocol_id": body.protocol_id,
        "exact_freeze_id": body.exact_freeze_id,
        "execution_freeze_manifest_id": body.execution_freeze_manifest_id,
        "execution_freeze_manifest_sha256": body.execution_freeze_manifest_sha256,
        "attempt_nonce_sha256": body.attempt_nonce_sha256,
    }
    replacements = {
        "authority_requirement_id": (
            "phase2b_c1_owner_accepted_container_authority_requirement_v1_" + SHA_H
        ),
        "actor_profile_content_id": (
            "phase2b_c1_container_actor_profile_v2_" + SHA_H
        ),
        "actor_profile_canonical_sha256": SHA_H,
        "campaign_id": "phase2b_actual_unsealed_960_campaign_v2_" + SHA_H,
        "protocol_id": "phase2b_protocol_" + SHA_H,
        "exact_freeze_id": "phase2b_exact_freeze_" + SHA_H,
        "execution_freeze_manifest_id": "phase2b_execution_freeze_" + SHA_H,
        "execution_freeze_manifest_sha256": SHA_H,
        "attempt_nonce_sha256": SHA_H,
    }
    for field, replacement in replacements.items():
        changed = {**values, field: replacement}
        assert wire.phase2b_c1_attempt_id_v1(**changed) != baseline


def test_context_content_id_tamper_is_rejected(graph: SuppliedGraph) -> None:
    context = replace(
        graph.context,
        context_id="phase2b_c1_six_role_ceremony_context_v1_" + SHA_H,
    )
    result = wire.verify_phase2b_c1_six_role_wire_v1(
        context=context,
        registry=graph.registry,
        store_receipt=graph.store_receipt,
        logical_genesis_token=graph.time_token,
    )
    assert result.reason is wire.Phase2BC1SixRoleWireReasonV1.CONTEXT_INVALID


@pytest.mark.parametrize(
    "field",
    (
        "authority_class",
        "authority_requirement_id",
        "actor_profile_content_id",
        "actor_profile_canonical_sha256",
        "campaign_id",
        "protocol_id",
        "exact_freeze_id",
        "execution_freeze_manifest_id",
        "execution_freeze_manifest_sha256",
        "attempt_nonce_sha256",
        "attempt_id",
        "recognizer_image_digest",
        "recognizer_source_root_sha256",
        "scorer_image_digest",
        "scorer_source_root_sha256",
    ),
)
def test_registry_must_repeat_every_explicit_context_binding(
    graph: SuppliedGraph, field: str
) -> None:
    current = getattr(graph.registry.body, field)
    if field.endswith("image_digest"):
        replacement = "sha256:" + SHA_H
    elif field == "authority_class":
        replacement = "CALLER_CONTEXT_DRIFT"
    elif field.endswith("_id"):
        replacement = current.rsplit("_", 1)[0] + "_" + SHA_H
    else:
        replacement = SHA_H
    body = replace(graph.registry.body, **{field: replacement})
    registry = replace(graph.registry, body=body)
    result = _verify(replace(graph, registry=registry))
    assert result.reason is wire.Phase2BC1SixRoleWireReasonV1.REGISTRY_INVALID


@pytest.mark.parametrize(
    "mutation",
    ("reordered", "duplicate", "wrong_algorithm", "wrong_epoch", "not_private", "wrong_key_id"),
)
def test_registry_requires_ordered_distinct_purpose_private_ed25519_keys(
    graph: SuppliedGraph, mutation: str
) -> None:
    keys = list(graph.registry.body.role_keys)
    if mutation == "reordered":
        keys[0], keys[1] = keys[1], keys[0]
    elif mutation == "duplicate":
        keys[1] = keys[0]
    elif mutation == "wrong_algorithm":
        keys[4] = replace(keys[4], algorithm="Ed448")
    elif mutation == "wrong_epoch":
        keys[4] = replace(keys[4], key_epoch=2)
    elif mutation == "not_private":
        keys[4] = replace(keys[4], purpose_private=False)
    else:
        keys[4] = replace(
            keys[4], key_id="phase2b_c1_six_role_key_v1_" + SHA_H
        )
    registry = replace(
        graph.registry, body=replace(graph.registry.body, role_keys=tuple(keys))
    )
    result = _verify(replace(graph, registry=registry))
    assert result.reason is wire.Phase2BC1SixRoleWireReasonV1.REGISTRY_INVALID


def test_governance_signature_wrong_domain_reaches_crypto_rejection(
    graph: SuppliedGraph,
) -> None:
    governance_key = graph.registry.body.role_keys[0]
    wrong_bytes = graph.private_keys[0].sign(
        wire.phase2b_c1_store_receipt_signature_preimage_v1(
            graph.store_receipt.body,
            signer_key_id=graph.registry.body.role_keys[2].key_id,
        )
    )
    signature = replace(graph.registry.governance_signature, signature=wrong_bytes)
    registry = replace(
        graph.registry,
        governance_signature=signature,
        execution_registry_id=wire.phase2b_c1_execution_registry_id_v1(
            graph.registry.body, signature
        ),
    )
    rebound = _rebind_store_and_time(graph, registry)
    result = _verify(rebound)
    assert governance_key.key_id == signature.signer_key_id
    assert result.reason is wire.Phase2BC1SixRoleWireReasonV1.SIGNATURE_INVALID


@pytest.mark.parametrize(
    "field",
    (
        "object_key",
        "previous_generation",
        "new_generation",
        "condition",
        "previous_content_sha256_or_null",
        "stored_content_sha256",
    ),
)
def test_store_receipt_requires_exact_if_absent_zero_to_one_and_registry_bytes(
    graph: SuppliedGraph, field: str
) -> None:
    replacements = {
        "object_key": "phase2b/c1/wrong-object",
        "previous_generation": 1,
        "new_generation": 2,
        "condition": "REPLACE",
        "previous_content_sha256_or_null": SHA_H,
        "stored_content_sha256": SHA_H,
    }
    store = replace(
        graph.store_receipt,
        body=replace(graph.store_receipt.body, **{field: replacements[field]}),
    )
    result = _verify(replace(graph, store_receipt=store))
    assert result.reason is wire.Phase2BC1SixRoleWireReasonV1.STORE_RECEIPT_INVALID


def test_store_signature_tamper_with_recomputed_content_id_is_crypto_rejected(
    graph: SuppliedGraph,
) -> None:
    broken = bytes([graph.store_receipt.store_signature.signature[0] ^ 1]) + (
        graph.store_receipt.store_signature.signature[1:]
    )
    signature = replace(graph.store_receipt.store_signature, signature=broken)
    store = replace(
        graph.store_receipt,
        store_signature=signature,
        store_receipt_id=wire.phase2b_c1_store_receipt_id_v1(
            graph.store_receipt.body, signature
        ),
    )
    result = _verify(_rebind_time(graph, store))
    assert result.reason is wire.Phase2BC1SixRoleWireReasonV1.SIGNATURE_INVALID


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("time_semantics", "UTC"),
        ("previous_logical_sequence_or_null", 0),
        ("logical_sequence", 1),
        ("stage", wire.Phase2BC1LogicalStageV1.RAW_INPUT_COMMITTED),
        ("stage_index", 1),
    ),
)
def test_time_token_is_only_signed_monotonic_logical_genesis(
    graph: SuppliedGraph, field: str, replacement: object
) -> None:
    time = replace(
        graph.time_token,
        body=replace(graph.time_token.body, **{field: replacement}),
    )
    result = _verify(replace(graph, time_token=time))
    assert result.reason is (
        wire.Phase2BC1SixRoleWireReasonV1.LOGICAL_GENESIS_INVALID
    )


def test_time_signature_tamper_with_recomputed_content_id_is_crypto_rejected(
    graph: SuppliedGraph,
) -> None:
    broken = bytes([graph.time_token.time_signature.signature[0] ^ 1]) + (
        graph.time_token.time_signature.signature[1:]
    )
    signature = replace(graph.time_token.time_signature, signature=broken)
    time = replace(
        graph.time_token,
        time_signature=signature,
        logical_genesis_token_id=wire.phase2b_c1_logical_genesis_token_id_v1(
            graph.time_token.body, signature
        ),
    )
    result = _verify(replace(graph, time_token=time))
    assert result.reason is wire.Phase2BC1SixRoleWireReasonV1.SIGNATURE_INVALID


@pytest.mark.parametrize(
    "target",
    ("registry_id", "store_context", "time_store", "time_registry"),
)
def test_graph_cross_bindings_fail_closed(graph: SuppliedGraph, target: str) -> None:
    if target == "registry_id":
        changed = replace(
            graph.registry,
            execution_registry_id=(
                "phase2b_c1_owner_accepted_execution_registry_v1_" + SHA_H
            ),
        )
        candidate = replace(graph, registry=changed)
    elif target == "store_context":
        changed = replace(
            graph.store_receipt,
            body=replace(
                graph.store_receipt.body,
                context_id="phase2b_c1_six_role_ceremony_context_v1_" + SHA_H,
            ),
        )
        candidate = replace(graph, store_receipt=changed)
    elif target == "time_store":
        changed = replace(
            graph.time_token,
            body=replace(
                graph.time_token.body,
                store_receipt_id=(
                    "phase2b_c1_execution_registry_if_absent_receipt_v1_" + SHA_H
                ),
            ),
        )
        candidate = replace(graph, time_token=changed)
    else:
        changed = replace(
            graph.time_token,
            body=replace(
                graph.time_token.body,
                execution_registry_id=(
                    "phase2b_c1_owner_accepted_execution_registry_v1_" + SHA_H
                ),
            ),
        )
        candidate = replace(graph, time_token=changed)
    result = _verify(candidate)
    assert result.disposition is wire.Phase2BC1SixRoleWireDispositionV1.REJECTED
    assert result.partial_output_published is False


def test_backend_unavailable_fails_closed(
    graph: SuppliedGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wire, "_Ed25519PublicKey", None)
    result = _verify(graph)
    assert result.reason is (
        wire.Phase2BC1SixRoleWireReasonV1.SIGNATURE_BACKEND_UNAVAILABLE
    )
    assert result.partial_output_published is False
    assert all(getattr(result, name) is False for name in (*TRUE_CLAIMS, *FALSE_CLAIMS))


def test_wrong_exact_input_types_fail_closed(graph: SuppliedGraph) -> None:
    result = wire.verify_phase2b_c1_six_role_wire_v1(
        context=object(),
        registry=graph.registry,
        store_receipt=graph.store_receipt,
        logical_genesis_token=graph.time_token,
    )
    assert result.reason is wire.Phase2BC1SixRoleWireReasonV1.WRONG_INPUT_TYPE
    assert result.partial_output_published is False


def test_canonical_wires_are_deterministic_ascii_without_newline(
    graph: SuppliedGraph,
) -> None:
    payloads = (
        wire.canonical_phase2b_c1_ceremony_context_bytes_v1(graph.context),
        wire.canonical_phase2b_c1_execution_registry_bytes_v1(graph.registry),
        wire.canonical_phase2b_c1_store_receipt_bytes_v1(graph.store_receipt),
        wire.canonical_phase2b_c1_logical_genesis_token_bytes_v1(
            graph.time_token
        ),
    )
    assert all(type(payload) is bytes for payload in payloads)
    assert all(payload.decode("ascii").encode("ascii") == payload for payload in payloads)
    assert all(not payload.endswith(b"\n") for payload in payloads)
    assert len(set(payloads)) == len(payloads)
    assert payloads == (
        wire.canonical_phase2b_c1_ceremony_context_bytes_v1(graph.context),
        wire.canonical_phase2b_c1_execution_registry_bytes_v1(graph.registry),
        wire.canonical_phase2b_c1_store_receipt_bytes_v1(graph.store_receipt),
        wire.canonical_phase2b_c1_logical_genesis_token_bytes_v1(
            graph.time_token
        ),
    )


def test_signature_preimages_are_domain_separated(graph: SuppliedGraph) -> None:
    governance = wire.phase2b_c1_governance_registry_signature_preimage_v1(
        graph.registry.body,
        signer_key_id=graph.registry.body.role_keys[0].key_id,
    )
    store = wire.phase2b_c1_store_receipt_signature_preimage_v1(
        graph.store_receipt.body,
        signer_key_id=graph.registry.body.role_keys[2].key_id,
    )
    logical_time = wire.phase2b_c1_logical_genesis_signature_preimage_v1(
        graph.time_token.body,
        signer_key_id=graph.registry.body.role_keys[3].key_id,
    )
    assert len({governance, store, logical_time}) == 3
    assert all(payload.endswith(b"]") for payload in (governance, store, logical_time))


def test_verifier_has_no_caller_supplied_verifier_or_authority_boolean() -> None:
    signature = inspect.signature(wire.verify_phase2b_c1_six_role_wire_v1)
    assert tuple(signature.parameters) == (
        "context",
        "registry",
        "store_receipt",
        "logical_genesis_token",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    assert not any(
        token in name
        for name in signature.parameters
        for token in ("verifier", "valid", "authority", "trusted")
    )


def test_production_module_has_no_operation_or_secret_key_capability() -> None:
    source_path = Path(wire.__file__).resolve()
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots.isdisjoint(
        {
            "ctypes",
            "datetime",
            "docker",
            "os",
            "pathlib",
            "random",
            "requests",
            "secrets",
            "shutil",
            "socket",
            "subprocess",
            "tempfile",
            "time",
            "urllib",
        }
    )
    assert "Ed25519PrivateKey" not in source
    assert "Callable" not in source
    calls = tuple(node for node in ast.walk(tree) if isinstance(node, ast.Call))
    assert not any(
        isinstance(node.func, ast.Attribute)
        and node.func.attr
        in {
            "generate",
            "from_private_bytes",
            "mkdir",
            "monotonic",
            "now",
            "open",
            "read_bytes",
            "read_text",
            "rename",
            "replace",
            "sign",
            "time",
            "token_bytes",
            "unlink",
            "urandom",
            "utcnow",
            "write_bytes",
            "write_text",
        }
        for node in calls
    )
    assert not any(
        isinstance(node.func, ast.Name)
        and node.func.id in {"open", "exec", "eval", "compile", "input"}
        for node in calls
    )


def test_explicit_context_and_registry_image_source_fields_are_not_authority(
    graph: SuppliedGraph,
) -> None:
    context_fields = {item.name for item in fields(wire.Phase2BC1CeremonyContextBodyV1)}
    registry_fields = {item.name for item in fields(wire.Phase2BC1ExecutionRegistryBodyV1)}
    required = {
        "recognizer_image_digest",
        "recognizer_source_root_sha256",
        "scorer_image_digest",
        "scorer_source_root_sha256",
    }
    assert required <= context_fields
    assert required <= registry_fields
    result = _verify(graph)
    assert result.registry_image_source_context_binding_verified is True
    assert result.image_runtime_identity_observed is False
    assert result.source_root_runtime_identity_observed is False
    assert result.registry_issuance_authority_verified is False


def test_result_claim_fields_cover_exact_declared_polarities() -> None:
    result_fields = {item.name for item in fields(wire.Phase2BC1SixRoleWireVerificationV1)}
    assert set(TRUE_CLAIMS).isdisjoint(FALSE_CLAIMS)
    assert set((*TRUE_CLAIMS, *FALSE_CLAIMS)) <= result_fields
    assert {
        "actual_960_case_run_verified",
        "scoring_performed",
        "formal_gate_evaluation_performed",
        "effect_evidence",
        "formal_c1_report_verified",
        "c1_exit_evidence",
    } <= set(FALSE_CLAIMS)


# Independent literal known-answer vector.  Nothing below derives wire bytes,
# hashes, preimages, or IDs through the production implementation.
KAT_TEST_ONLY_SEED_HEX = (
    "0101010101010101010101010101010101010101010101010101010101010101",
    "0202020202020202020202020202020202020202020202020202020202020202",
    "0303030303030303030303030303030303030303030303030303030303030303",
    "0404040404040404040404040404040404040404040404040404040404040404",
    "0505050505050505050505050505050505050505050505050505050505050505",
    "0606060606060606060606060606060606060606060606060606060606060606",
)
KAT_TEST_ONLY_SEED_SHA256 = (
    "72cd6e8422c407fb6d098690f1130b7ded7ec2f7f5e1d30bd9d521f015363793",
    "75877bb41d393b5fb8455ce60ecd8dda001d06316496b14dfa7f895656eeca4a",
    "648aa5c579fb30f38af744d97d6ec840c7a91277a499a0d780f3e7314eca090b",
    "9f4fb68f3e1dac82202f9aa581ce0bbf1f765df0e9ac3c8c57e20f685abab8ed",
    "f849d67325facf04177bc663b2dc544051831c589ef581d412f2eba44834e77c",
    "e802086ad6a1e16b78352ad7296d2aabd835b1b16dbe951e1135b97c68e29d81",
)
KAT_RAW_PUBLIC_KEY_HEX = (
    "8a88e3dd7409f195fd52db2d3cba5d72ca6709bf1d94121bf3748801b40f6f5c",
    "8139770ea87d175f56a35466c34c7ecccb8d8a91b4ee37a25df60f5b8fc9b394",
    "ed4928c628d1c2c6eae90338905995612959273a5c63f93636c14614ac8737d1",
    "ca93ac1705187071d67b83c7ff0efe8108e8ec4530575d7726879333dbdabe7c",
    "6e7a1cdd29b0b78fd13af4c5598feff4ef2a97166e3ca6f2e4fbfccd80505bf1",
    "8a875fff1eb38451577acd5afee405456568dd7c89e090863a0557bc7af49f17",
)
KAT_KEY_IDS = (
    "phase2b_c1_six_role_key_v1_85aed206830f55abb7261123ce287e3b4b6f8b2e6a4b5ae224f5558afbc0c0db",
    "phase2b_c1_six_role_key_v1_d8143041287cf1b7edafd19ba897e072e0b259bffe3ec870ebe0c80b96676f3b",
    "phase2b_c1_six_role_key_v1_ab35a39cb0fe05b75bbc478fb433401e78b3017f7f53b4f33d4d689a1471432b",
    "phase2b_c1_six_role_key_v1_9516c609a1c0eee576cec70e921d4630dfa33b76f983b8003270fa1455301802",
    "phase2b_c1_six_role_key_v1_67e765ea7cf4c108bad36a1ee1fa957505e85eef4f2b1310857f924c14a381e9",
    "phase2b_c1_six_role_key_v1_18d893bc7bbccbc65074983b931ba86d1e31e3e2865479ca2fec0a5d2ca419b7",
)
KAT_ATTEMPT_ID = (
    "phase2b_actual_unsealed_960_attempt_v2_"
    "94351fba48a7f13d6406fc004b3c465b0e313cf47d80960954321cb25aa5d2eb"
)
KAT_CONTEXT_ID = (
    "phase2b_c1_six_role_ceremony_context_v1_"
    "ec7a50eabb1c60d2d8fda6852d6217765b4c831aa44bbf86dd01d297629d5381"
)
KAT_GOVERNANCE_SIGNATURE_HEX = (
    "b343317144b0c84a5f354893371a905d347e5ce2171003601e500865f57c9fca"
    "734649a23d3ceb05c767c51b31f59d69aa8f8a3e7babd581dd28b86c1d55280b"
)
KAT_REGISTRY_ID = (
    "phase2b_c1_owner_accepted_execution_registry_v1_"
    "eecaf3709291c1142528a36a3e4b056181799e95bf839c47a298726575865a12"
)
KAT_STORE_SIGNATURE_HEX = (
    "5a42d7fbc5f4e329f48a97224662c8cd17df6abb547368cae2218a3f53b5307a"
    "296eb58b1b7920ea161dee7471f21671610ced0fe141abb4cbbe850042b6670d"
)
KAT_STORE_RECEIPT_ID = (
    "phase2b_c1_execution_registry_if_absent_receipt_v1_"
    "1974571add711cd6e4d95ebc74aad39fccc85890d54257c63855a825550348ac"
)
KAT_TIME_SIGNATURE_HEX = (
    "9f82a9952d005d2c13434845674b8d29efb9ba4a45c1032aed1ffdfc9e490805"
    "2e7a4e7abd89a3c28db78e84f874942b6257ae4a624bd47f96fe012e72dd310b"
)
KAT_TIME_TOKEN_ID = (
    "phase2b_c1_signed_logical_genesis_token_v1_"
    "4979954595ac86b2539d16d6f6a2127605b749d577a9008e656d8d31d415573a"
)
KAT_RESULT_ID = (
    "phase2b_c1_six_role_wire_verification_v1_"
    "cbea3b4c045ce6dd0dec7896c70f5a2753f7d3e4518fee5264a41abec8cb67c5"
)
KAT_CANONICAL_WIRE_SHA256 = {
    "context": "d8a4c93de0d33a49483a26f3581c3a625c0c3c643c57531e7ed20719bd402bf9",
    "registry": "2b27ec5ec1afb9915acb0fa9f43ecff2025b82b5aa07d5fcc382db69dec9e87d",
    "store": "759344dba3f1fe590bb32d4eaf88ec85c762f59f7ac15812c98a33b4195b5bba",
    "time": "d96bbd29c3234f2ebe1c4e514e65c9cf2a001671d6d27f09d1263b18bf5a8af7",
}
KAT_SIGNATURE_PREIMAGE_SHA256 = {
    "governance": "82d661d22f2cc66df51a5959d2470c54bfc072a7a90edbb864623c56cf7f556c",
    "store": "cb67e352190f0252831bc2a341033325625ff8fabba080956c631d1c2e0b1695",
    "time": "e67bf5b4911f9caa869df8c5b4561314568ceece139afeb89d54c0c683f7d073",
}

KAT_DOMAINS = {
    "attempt": b"HEGEL/PHASE2B/C1/SIX_ROLE_WIRE/ATTEMPT/V1\x00",
    "context": b"HEGEL/PHASE2B/C1/SIX_ROLE_WIRE/CONTEXT/V1\x00",
    "key": b"HEGEL/PHASE2B/C1/SIX_ROLE_WIRE/ROLE_KEY/V1\x00",
    "governance_signature": (
        b"HEGEL/PHASE2B/C1/SIX_ROLE_WIRE/"
        b"GOVERNANCE_REGISTRY_SIGNATURE/V1\x00"
    ),
    "registry_id": (
        b"HEGEL/PHASE2B/C1/SIX_ROLE_WIRE/EXECUTION_REGISTRY_ID/V1\x00"
    ),
    "store_signature": (
        b"HEGEL/PHASE2B/C1/SIX_ROLE_WIRE/STORE_IF_ABSENT_SIGNATURE/V1\x00"
    ),
    "store_id": (
        b"HEGEL/PHASE2B/C1/SIX_ROLE_WIRE/"
        b"STORE_IF_ABSENT_RECEIPT_ID/V1\x00"
    ),
    "time_signature": (
        b"HEGEL/PHASE2B/C1/SIX_ROLE_WIRE/LOGICAL_GENESIS_SIGNATURE/V1\x00"
    ),
    "time_id": (
        b"HEGEL/PHASE2B/C1/SIX_ROLE_WIRE/LOGICAL_GENESIS_TOKEN_ID/V1\x00"
    ),
    "result": b"HEGEL/PHASE2B/C1/SIX_ROLE_WIRE/RESULT/V1\x00",
}
KAT_ATTEMPT_FIELD_ORDER = (
    "authority_requirement_id",
    "actor_profile_content_id",
    "actor_profile_canonical_sha256",
    "campaign_id",
    "protocol_id",
    "exact_freeze_id",
    "execution_freeze_manifest_id",
    "execution_freeze_manifest_sha256",
    "attempt_nonce_sha256",
)
KAT_CONTEXT_FIELD_ORDER = (
    "version",
    "authority_class",
    "authority_requirement_id",
    "actor_profile_content_id",
    "actor_profile_canonical_sha256",
    "campaign_id",
    "protocol_id",
    "exact_freeze_id",
    "execution_freeze_manifest_id",
    "execution_freeze_manifest_sha256",
    "attempt_nonce_sha256",
    "attempt_id",
    "recognizer_image_digest",
    "recognizer_source_root_sha256",
    "scorer_image_digest",
    "scorer_source_root_sha256",
)
KAT_REGISTRY_FIELD_ORDER = (
    "version",
    "context_id",
    "authority_class",
    "authority_requirement_id",
    "actor_profile_content_id",
    "actor_profile_canonical_sha256",
    "campaign_id",
    "protocol_id",
    "exact_freeze_id",
    "execution_freeze_manifest_id",
    "execution_freeze_manifest_sha256",
    "attempt_nonce_sha256",
    "attempt_id",
    "recognizer_image_digest",
    "recognizer_source_root_sha256",
    "scorer_image_digest",
    "scorer_source_root_sha256",
    "role_keys",
)
KAT_STORE_FIELD_ORDER = (
    "version",
    "context_id",
    "authority_requirement_id",
    "actor_profile_content_id",
    "campaign_id",
    "attempt_id",
    "execution_registry_id",
    "object_key",
    "previous_generation",
    "new_generation",
    "condition",
    "previous_content_sha256_or_null",
    "stored_content_sha256",
)
KAT_TIME_FIELD_ORDER = (
    "version",
    "context_id",
    "authority_requirement_id",
    "actor_profile_content_id",
    "campaign_id",
    "attempt_id",
    "execution_registry_id",
    "store_receipt_id",
    "time_semantics",
    "previous_logical_sequence_or_null",
    "logical_sequence",
    "stage",
    "stage_index",
)
KAT_ROLE_VALUES = (
    "GOVERNANCE_ANCHOR",
    "CUSTODIAN",
    "STORE_RECEIPT",
    "TIME_ATTESTOR",
    "RECOGNIZER",
    "SCORER_EVALUATOR",
)


def _kat_canonical_named_bytes(
    pairs: tuple[tuple[str, object], ...],
) -> bytes:
    return json.dumps(
        pairs,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _kat_content_id(
    prefix: str, domain: bytes, pairs: tuple[tuple[str, object], ...]
) -> str:
    return prefix + hashlib.sha256(
        domain + _kat_canonical_named_bytes(pairs)
    ).hexdigest()


def _literal_kat_material() -> dict[str, object]:
    version = "hegel-machine-phase2b-c1-six-role-wire/1"
    authority_class = "OWNER_ACCEPTED_CONTAINER_TECHNICAL_ACTORS_V1"
    authority_requirement_id = (
        "phase2b_c1_owner_accepted_container_authority_requirement_v1_"
        "53631d563f8da8f26c7202b7dd0681be5423fe2ca5c15e9a9f5deaa2920d9940"
    )
    profile_id = (
        "phase2b_c1_container_actor_profile_v2_"
        "b2f081771f3de8d2a531b08b58109d95c51815904e9612e66f1bbfe69babf906"
    )
    profile_sha = (
        "8ca1850f026dabb7a4b69add0f89b33a3fcfb789223b2fd95ef0379608adbeed"
    )
    campaign_id = "phase2b_actual_unsealed_960_campaign_v2_" + SHA_A
    protocol_id = (
        "phase2b_protocol_"
        "62ad411b5b5a0f912626c54e4bd822a8c585a8f612f5fce6040cce500a11756a"
    )
    exact_freeze_id = (
        "phase2b_exact_freeze_"
        "ffa1fd4fed0b5c2c018803aa9f730b8c85c144efe7e4aa324256681d1c742cbe"
    )
    execution_freeze_id = "phase2b_execution_freeze_" + SHA_B
    attempt_values = {
        "authority_requirement_id": authority_requirement_id,
        "actor_profile_content_id": profile_id,
        "actor_profile_canonical_sha256": profile_sha,
        "campaign_id": campaign_id,
        "protocol_id": protocol_id,
        "exact_freeze_id": exact_freeze_id,
        "execution_freeze_manifest_id": execution_freeze_id,
        "execution_freeze_manifest_sha256": SHA_C,
        "attempt_nonce_sha256": SHA_D,
    }
    attempt_pairs = tuple(
        (name, attempt_values[name]) for name in KAT_ATTEMPT_FIELD_ORDER
    )
    attempt_id = _kat_content_id(
        "phase2b_actual_unsealed_960_attempt_v2_",
        KAT_DOMAINS["attempt"],
        attempt_pairs,
    )
    context_values = {
        "version": version,
        "authority_class": authority_class,
        **attempt_values,
        "attempt_id": attempt_id,
        "recognizer_image_digest": "sha256:" + SHA_E,
        "recognizer_source_root_sha256": SHA_F,
        "scorer_image_digest": "sha256:" + SHA_G,
        "scorer_source_root_sha256": (
            "34f4a2404d00d46cede832d34ca9713253da93f1becef4a1d302c1411d6bd5e7"
        ),
    }
    context_pairs = tuple(
        (name, context_values[name]) for name in KAT_CONTEXT_FIELD_ORDER
    )
    context_id = _kat_content_id(
        "phase2b_c1_six_role_ceremony_context_v1_",
        KAT_DOMAINS["context"],
        context_pairs,
    )
    key_pairs: list[tuple[tuple[str, object], ...]] = []
    independently_computed_key_ids: list[str] = []
    for purpose_id, (role, public_key_hex) in enumerate(
        zip(KAT_ROLE_VALUES, KAT_RAW_PUBLIC_KEY_HEX, strict=True), start=1
    ):
        unsigned_key_pairs = (
            ("purpose_id", purpose_id),
            ("role", role),
            ("algorithm", "Ed25519"),
            ("key_epoch", 1),
            ("purpose_private", True),
            ("public_key_hex", public_key_hex),
        )
        key_id = _kat_content_id(
            "phase2b_c1_six_role_key_v1_",
            KAT_DOMAINS["key"],
            unsigned_key_pairs,
        )
        independently_computed_key_ids.append(key_id)
        key_pairs.append((*unsigned_key_pairs, ("key_id", key_id)))
    registry_values = {
        "version": version,
        "context_id": context_id,
        "authority_class": authority_class,
        **attempt_values,
        "attempt_id": attempt_id,
        "recognizer_image_digest": context_values["recognizer_image_digest"],
        "recognizer_source_root_sha256": context_values[
            "recognizer_source_root_sha256"
        ],
        "scorer_image_digest": context_values["scorer_image_digest"],
        "scorer_source_root_sha256": context_values[
            "scorer_source_root_sha256"
        ],
        "role_keys": tuple(key_pairs),
    }
    registry_pairs = tuple(
        (name, registry_values[name]) for name in KAT_REGISTRY_FIELD_ORDER
    )
    governance_header = (
        ("purpose_id", 1),
        ("role", "GOVERNANCE_ANCHOR"),
        ("signature_purpose", "GOVERNANCE_CHECKPOINT"),
        ("algorithm", "Ed25519"),
        ("key_epoch", 1),
        ("signer_key_id", KAT_KEY_IDS[0]),
    )
    governance_preimage = KAT_DOMAINS[
        "governance_signature"
    ] + _kat_canonical_named_bytes(
        (("signature_header", governance_header), ("body", registry_pairs))
    )
    governance_signature_pairs = (
        *governance_header,
        ("signature_hex", KAT_GOVERNANCE_SIGNATURE_HEX),
    )
    registry_id = _kat_content_id(
        "phase2b_c1_owner_accepted_execution_registry_v1_",
        KAT_DOMAINS["registry_id"],
        (
            ("body", registry_pairs),
            ("governance_signature", governance_signature_pairs),
        ),
    )
    registry_wire = _kat_canonical_named_bytes(
        (
            ("body", registry_pairs),
            ("governance_signature", governance_signature_pairs),
            ("execution_registry_id", registry_id),
        )
    )
    store_values = {
        "version": version,
        "context_id": context_id,
        "authority_requirement_id": authority_requirement_id,
        "actor_profile_content_id": profile_id,
        "campaign_id": campaign_id,
        "attempt_id": attempt_id,
        "execution_registry_id": registry_id,
        "object_key": (
            "phase2b/c1/actual-unsealed-960/"
            f"{campaign_id}/attempt-0/execution-registry"
        ),
        "previous_generation": 0,
        "new_generation": 1,
        "condition": "IF_ABSENT",
        "previous_content_sha256_or_null": None,
        "stored_content_sha256": hashlib.sha256(registry_wire).hexdigest(),
    }
    store_pairs = tuple(
        (name, store_values[name]) for name in KAT_STORE_FIELD_ORDER
    )
    store_header = (
        ("purpose_id", 3),
        ("role", "STORE_RECEIPT"),
        ("signature_purpose", "STORE_IF_ABSENT_RECEIPT"),
        ("algorithm", "Ed25519"),
        ("key_epoch", 1),
        ("signer_key_id", KAT_KEY_IDS[2]),
    )
    store_preimage = KAT_DOMAINS[
        "store_signature"
    ] + _kat_canonical_named_bytes(
        (("signature_header", store_header), ("body", store_pairs))
    )
    store_signature_pairs = (
        *store_header,
        ("signature_hex", KAT_STORE_SIGNATURE_HEX),
    )
    store_receipt_id = _kat_content_id(
        "phase2b_c1_execution_registry_if_absent_receipt_v1_",
        KAT_DOMAINS["store_id"],
        (("body", store_pairs), ("store_signature", store_signature_pairs)),
    )
    store_wire = _kat_canonical_named_bytes(
        (
            ("body", store_pairs),
            ("store_signature", store_signature_pairs),
            ("store_receipt_id", store_receipt_id),
        )
    )
    time_values = {
        "version": version,
        "context_id": context_id,
        "authority_requirement_id": authority_requirement_id,
        "actor_profile_content_id": profile_id,
        "campaign_id": campaign_id,
        "attempt_id": attempt_id,
        "execution_registry_id": registry_id,
        "store_receipt_id": store_receipt_id,
        "time_semantics": "SIGNED_MONOTONIC_LOGICAL_ORDER_NOT_UTC",
        "previous_logical_sequence_or_null": None,
        "logical_sequence": 0,
        "stage": "PREREGISTERED",
        "stage_index": 0,
    }
    time_pairs = tuple(
        (name, time_values[name]) for name in KAT_TIME_FIELD_ORDER
    )
    time_header = (
        ("purpose_id", 4),
        ("role", "TIME_ATTESTOR"),
        ("signature_purpose", "DECLARED_TIME_TOKEN"),
        ("algorithm", "Ed25519"),
        ("key_epoch", 1),
        ("signer_key_id", KAT_KEY_IDS[3]),
    )
    time_preimage = KAT_DOMAINS[
        "time_signature"
    ] + _kat_canonical_named_bytes(
        (("signature_header", time_header), ("body", time_pairs))
    )
    time_signature_pairs = (
        *time_header,
        ("signature_hex", KAT_TIME_SIGNATURE_HEX),
    )
    time_token_id = _kat_content_id(
        "phase2b_c1_signed_logical_genesis_token_v1_",
        KAT_DOMAINS["time_id"],
        (("body", time_pairs), ("time_signature", time_signature_pairs)),
    )
    time_wire = _kat_canonical_named_bytes(
        (
            ("body", time_pairs),
            ("time_signature", time_signature_pairs),
            ("logical_genesis_token_id", time_token_id),
        )
    )
    result_id = _kat_content_id(
        "phase2b_c1_six_role_wire_verification_v1_",
        KAT_DOMAINS["result"],
        (
            ("version", version),
            (
                "claim_level",
                "CALLER_SUPPLIED_OWNER_ACCEPTED_CONTEXT_"
                "CRYPTOGRAPHIC_WIRE_MECHANICS_ONLY",
            ),
            ("context_id", context_id),
            ("attempt_id", attempt_id),
            ("execution_registry_id", registry_id),
            ("store_receipt_id", store_receipt_id),
            ("logical_genesis_token_id", time_token_id),
            ("role_key_ids", tuple(independently_computed_key_ids)),
        ),
    )
    return {
        "attempt_id": attempt_id,
        "context_id": context_id,
        "key_ids": tuple(independently_computed_key_ids),
        "registry_id": registry_id,
        "store_receipt_id": store_receipt_id,
        "time_token_id": time_token_id,
        "result_id": result_id,
        "context_pairs": context_pairs,
        "registry_pairs": registry_pairs,
        "store_pairs": store_pairs,
        "time_pairs": time_pairs,
        "governance_preimage": governance_preimage,
        "store_preimage": store_preimage,
        "time_preimage": time_preimage,
        "context_wire": _kat_canonical_named_bytes(
            (("body", context_pairs), ("context_id", context_id))
        ),
        "registry_wire": registry_wire,
        "store_wire": store_wire,
        "time_wire": time_wire,
    }


@dataclass(frozen=True)
class LiteralKatGraph:
    context: wire.Phase2BC1CeremonyContextV1
    registry: wire.Phase2BC1ExecutionRegistryCandidateV1
    store_receipt: wire.Phase2BC1StoreIfAbsentReceiptCandidateV1
    time_token: wire.Phase2BC1LogicalGenesisTokenCandidateV1


def _build_literal_kat_graph() -> LiteralKatGraph:
    material = _literal_kat_material()
    assert material["attempt_id"] == KAT_ATTEMPT_ID
    assert material["context_id"] == KAT_CONTEXT_ID
    assert material["key_ids"] == KAT_KEY_IDS
    assert material["registry_id"] == KAT_REGISTRY_ID
    assert material["store_receipt_id"] == KAT_STORE_RECEIPT_ID
    assert material["time_token_id"] == KAT_TIME_TOKEN_ID
    assert material["result_id"] == KAT_RESULT_ID
    for public_index, signature_hex, preimage_name in (
        (0, KAT_GOVERNANCE_SIGNATURE_HEX, "governance_preimage"),
        (2, KAT_STORE_SIGNATURE_HEX, "store_preimage"),
        (3, KAT_TIME_SIGNATURE_HEX, "time_preimage"),
    ):
        Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(KAT_RAW_PUBLIC_KEY_HEX[public_index])
        ).verify(bytes.fromhex(signature_hex), material[preimage_name])
    context_values = dict(material["context_pairs"])
    context_body = wire.Phase2BC1CeremonyContextBodyV1(**context_values)
    context = wire.Phase2BC1CeremonyContextV1(
        body=context_body,
        context_id=KAT_CONTEXT_ID,
    )
    role_keys = tuple(
        wire.Phase2BC1RoleKeyV1(
            purpose_id=index,
            role=wire.Phase2BC1SixRoleV1(role),
            algorithm="Ed25519",
            key_epoch=1,
            purpose_private=True,
            public_key=bytes.fromhex(public_key_hex),
            key_id=key_id,
        )
        for index, (role, public_key_hex, key_id) in enumerate(
            zip(
                KAT_ROLE_VALUES,
                KAT_RAW_PUBLIC_KEY_HEX,
                KAT_KEY_IDS,
                strict=True,
            ),
            start=1,
        )
    )
    registry_values = dict(material["registry_pairs"])
    del registry_values["role_keys"]
    registry_body = wire.Phase2BC1ExecutionRegistryBodyV1(
        **registry_values,
        role_keys=role_keys,
    )
    governance_signature = wire.Phase2BC1DetachedSignatureV1(
        purpose_id=1,
        role=wire.Phase2BC1SixRoleV1("GOVERNANCE_ANCHOR"),
        signature_purpose=wire.Phase2BC1SignaturePurposeV1(
            "GOVERNANCE_CHECKPOINT"
        ),
        algorithm="Ed25519",
        key_epoch=1,
        signer_key_id=KAT_KEY_IDS[0],
        signature=bytes.fromhex(KAT_GOVERNANCE_SIGNATURE_HEX),
    )
    registry = wire.Phase2BC1ExecutionRegistryCandidateV1(
        body=registry_body,
        governance_signature=governance_signature,
        execution_registry_id=KAT_REGISTRY_ID,
    )
    store_body = wire.Phase2BC1StoreIfAbsentReceiptBodyV1(
        **dict(material["store_pairs"])
    )
    store_signature = wire.Phase2BC1DetachedSignatureV1(
        purpose_id=3,
        role=wire.Phase2BC1SixRoleV1("STORE_RECEIPT"),
        signature_purpose=wire.Phase2BC1SignaturePurposeV1(
            "STORE_IF_ABSENT_RECEIPT"
        ),
        algorithm="Ed25519",
        key_epoch=1,
        signer_key_id=KAT_KEY_IDS[2],
        signature=bytes.fromhex(KAT_STORE_SIGNATURE_HEX),
    )
    store_receipt = wire.Phase2BC1StoreIfAbsentReceiptCandidateV1(
        body=store_body,
        store_signature=store_signature,
        store_receipt_id=KAT_STORE_RECEIPT_ID,
    )
    time_values = dict(material["time_pairs"])
    time_values["stage"] = wire.Phase2BC1LogicalStageV1(
        time_values["stage"]
    )
    time_body = wire.Phase2BC1LogicalGenesisTokenBodyV1(**time_values)
    time_signature = wire.Phase2BC1DetachedSignatureV1(
        purpose_id=4,
        role=wire.Phase2BC1SixRoleV1("TIME_ATTESTOR"),
        signature_purpose=wire.Phase2BC1SignaturePurposeV1(
            "DECLARED_TIME_TOKEN"
        ),
        algorithm="Ed25519",
        key_epoch=1,
        signer_key_id=KAT_KEY_IDS[3],
        signature=bytes.fromhex(KAT_TIME_SIGNATURE_HEX),
    )
    time_token = wire.Phase2BC1LogicalGenesisTokenCandidateV1(
        body=time_body,
        time_signature=time_signature,
        logical_genesis_token_id=KAT_TIME_TOKEN_ID,
    )
    return LiteralKatGraph(
        context=context,
        registry=registry,
        store_receipt=store_receipt,
        time_token=time_token,
    )


def test_literal_kat_independently_freezes_domains_and_field_order() -> None:
    assert len(set(KAT_DOMAINS.values())) == 10
    assert all(domain.endswith(b"\x00") for domain in KAT_DOMAINS.values())
    assert tuple(item.name for item in fields(wire.Phase2BC1CeremonyContextBodyV1)) == (
        KAT_CONTEXT_FIELD_ORDER
    )
    assert tuple(
        item.name for item in fields(wire.Phase2BC1ExecutionRegistryBodyV1)
    ) == KAT_REGISTRY_FIELD_ORDER
    assert tuple(
        item.name for item in fields(wire.Phase2BC1StoreIfAbsentReceiptBodyV1)
    ) == KAT_STORE_FIELD_ORDER
    assert tuple(
        item.name for item in fields(wire.Phase2BC1LogicalGenesisTokenBodyV1)
    ) == KAT_TIME_FIELD_ORDER
    source_tree = ast.parse(Path(wire.__file__).read_text(encoding="utf-8"))
    production_domains: dict[str, bytes] = {}
    source_names = {
        "_ATTEMPT_DOMAIN": "attempt",
        "_CONTEXT_DOMAIN": "context",
        "_KEY_DOMAIN": "key",
        "_GOVERNANCE_SIGNATURE_DOMAIN": "governance_signature",
        "_REGISTRY_ID_DOMAIN": "registry_id",
        "_STORE_SIGNATURE_DOMAIN": "store_signature",
        "_STORE_ID_DOMAIN": "store_id",
        "_TIME_SIGNATURE_DOMAIN": "time_signature",
        "_TIME_ID_DOMAIN": "time_id",
        "_RESULT_ID_DOMAIN": "result",
    }
    for node in source_tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in source_names
        ):
            production_domains[source_names[node.target.id]] = ast.literal_eval(
                node.value
            )
    assert production_domains == KAT_DOMAINS


def test_literal_kat_independent_ids_wires_and_preimages_match_answers() -> None:
    material = _literal_kat_material()
    assert material["key_ids"] == KAT_KEY_IDS
    assert material["attempt_id"] == KAT_ATTEMPT_ID
    assert material["context_id"] == KAT_CONTEXT_ID
    assert material["registry_id"] == KAT_REGISTRY_ID
    assert material["store_receipt_id"] == KAT_STORE_RECEIPT_ID
    assert material["time_token_id"] == KAT_TIME_TOKEN_ID
    assert material["result_id"] == KAT_RESULT_ID
    assert {
        name: hashlib.sha256(material[f"{name}_wire"]).hexdigest()
        for name in ("context", "registry", "store", "time")
    } == KAT_CANONICAL_WIRE_SHA256
    assert {
        name: hashlib.sha256(material[f"{name}_preimage"]).hexdigest()
        for name in ("governance", "store", "time")
    } == KAT_SIGNATURE_PREIMAGE_SHA256
    assert dict(material["store_pairs"])["stored_content_sha256"] == (
        KAT_CANONICAL_WIRE_SHA256["registry"]
    )


def test_literal_kat_cross_checks_every_production_public_wire_helper() -> None:
    graph = _build_literal_kat_graph()
    context_body = graph.context.body
    assert wire.phase2b_c1_attempt_id_v1(
        authority_requirement_id=context_body.authority_requirement_id,
        actor_profile_content_id=context_body.actor_profile_content_id,
        actor_profile_canonical_sha256=(
            context_body.actor_profile_canonical_sha256
        ),
        campaign_id=context_body.campaign_id,
        protocol_id=context_body.protocol_id,
        exact_freeze_id=context_body.exact_freeze_id,
        execution_freeze_manifest_id=(
            context_body.execution_freeze_manifest_id
        ),
        execution_freeze_manifest_sha256=(
            context_body.execution_freeze_manifest_sha256
        ),
        attempt_nonce_sha256=context_body.attempt_nonce_sha256,
    ) == KAT_ATTEMPT_ID
    assert tuple(
        wire.phase2b_c1_role_key_id_v1(
            purpose_id=key.purpose_id,
            role=key.role,
            public_key=key.public_key,
        )
        for key in graph.registry.body.role_keys
    ) == KAT_KEY_IDS
    assert wire.phase2b_c1_ceremony_context_id_v1(
        graph.context.body
    ) == KAT_CONTEXT_ID
    assert wire.phase2b_c1_execution_registry_id_v1(
        graph.registry.body,
        graph.registry.governance_signature,
    ) == KAT_REGISTRY_ID
    assert wire.phase2b_c1_store_receipt_id_v1(
        graph.store_receipt.body,
        graph.store_receipt.store_signature,
    ) == KAT_STORE_RECEIPT_ID
    assert wire.phase2b_c1_logical_genesis_token_id_v1(
        graph.time_token.body,
        graph.time_token.time_signature,
    ) == KAT_TIME_TOKEN_ID
    production_canonical_wire_sha256 = {
        "context": hashlib.sha256(
            wire.canonical_phase2b_c1_ceremony_context_bytes_v1(
                graph.context
            )
        ).hexdigest(),
        "registry": hashlib.sha256(
            wire.canonical_phase2b_c1_execution_registry_bytes_v1(
                graph.registry
            )
        ).hexdigest(),
        "store": hashlib.sha256(
            wire.canonical_phase2b_c1_store_receipt_bytes_v1(
                graph.store_receipt
            )
        ).hexdigest(),
        "time": hashlib.sha256(
            wire.canonical_phase2b_c1_logical_genesis_token_bytes_v1(
                graph.time_token
            )
        ).hexdigest(),
    }
    assert production_canonical_wire_sha256 == KAT_CANONICAL_WIRE_SHA256
    production_preimage_sha256 = {
        "governance": hashlib.sha256(
            wire.phase2b_c1_governance_registry_signature_preimage_v1(
                graph.registry.body,
                signer_key_id=graph.registry.body.role_keys[0].key_id,
            )
        ).hexdigest(),
        "store": hashlib.sha256(
            wire.phase2b_c1_store_receipt_signature_preimage_v1(
                graph.store_receipt.body,
                signer_key_id=graph.registry.body.role_keys[2].key_id,
            )
        ).hexdigest(),
        "time": hashlib.sha256(
            wire.phase2b_c1_logical_genesis_signature_preimage_v1(
                graph.time_token.body,
                signer_key_id=graph.registry.body.role_keys[3].key_id,
            )
        ).hexdigest(),
    }
    assert production_preimage_sha256 == KAT_SIGNATURE_PREIMAGE_SHA256


def test_independent_kat_oracle_helpers_cannot_call_production_wire() -> None:
    for helper in (
        _kat_canonical_named_bytes,
        _kat_content_id,
        _literal_kat_material,
    ):
        helper_tree = ast.parse(inspect.getsource(helper))
        assert not any(
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "wire"
            for node in ast.walk(helper_tree)
        ), helper.__name__


def test_literal_kat_fixed_seed_commitments_public_keys_and_signatures() -> None:
    assert tuple(
        hashlib.sha256(bytes.fromhex(seed_hex)).hexdigest()
        for seed_hex in KAT_TEST_ONLY_SEED_HEX
    ) == KAT_TEST_ONLY_SEED_SHA256
    seed_private_keys = tuple(
        Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed_hex))
        for seed_hex in KAT_TEST_ONLY_SEED_HEX
    )
    assert tuple(
        private_key.public_key()
        .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        .hex()
        for private_key in seed_private_keys
    ) == KAT_RAW_PUBLIC_KEY_HEX
    material = _literal_kat_material()
    for public_index, signature_hex, preimage_name in (
        (0, KAT_GOVERNANCE_SIGNATURE_HEX, "governance_preimage"),
        (2, KAT_STORE_SIGNATURE_HEX, "store_preimage"),
        (3, KAT_TIME_SIGNATURE_HEX, "time_preimage"),
    ):
        Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(KAT_RAW_PUBLIC_KEY_HEX[public_index])
        ).verify(bytes.fromhex(signature_hex), material[preimage_name])


def test_literal_kat_success_candidate_uses_no_production_wire_builders() -> None:
    graph = _build_literal_kat_graph()
    result = wire.verify_phase2b_c1_six_role_wire_v1(
        context=graph.context,
        registry=graph.registry,
        store_receipt=graph.store_receipt,
        logical_genesis_token=graph.time_token,
    )
    assert result.disposition is (
        wire.Phase2BC1SixRoleWireDispositionV1.CALLER_SUPPLIED_SIX_ROLE_WIRE_GRAPH_VERIFIED_NOT_AUTHORITY
    )
    assert result.result_id == KAT_RESULT_ID
    assert all(getattr(result, name) is True for name in TRUE_CLAIMS)
    assert all(getattr(result, name) is False for name in FALSE_CLAIMS)
    builder_tree = ast.parse(inspect.getsource(_build_literal_kat_graph))
    forbidden_production_calls = tuple(
        node.func.attr
        for node in ast.walk(builder_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "wire"
        and (
            node.func.attr.startswith("_")
            or node.func.attr.startswith("phase2b_c1_")
            or node.func.attr.startswith("canonical_phase2b_c1_")
        )
    )
    assert forbidden_production_calls == ()


@pytest.mark.parametrize(
    ("target", "expected_reason"),
    (
        ("registry", wire.Phase2BC1SixRoleWireReasonV1.REGISTRY_INVALID),
        ("store", wire.Phase2BC1SixRoleWireReasonV1.STORE_RECEIPT_INVALID),
        (
            "time",
            wire.Phase2BC1SixRoleWireReasonV1.LOGICAL_GENESIS_INVALID,
        ),
    ),
)
def test_registry_store_time_content_address_tamper_calls_crypto_zero_times(
    target: str,
    expected_reason: wire.Phase2BC1SixRoleWireReasonV1,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _build_literal_kat_graph()
    if target == "registry":
        graph = replace(
            graph,
            registry=replace(
                graph.registry,
                execution_registry_id=(
                    "phase2b_c1_owner_accepted_execution_registry_v1_" + SHA_H
                ),
            ),
        )
    elif target == "store":
        graph = replace(
            graph,
            store_receipt=replace(
                graph.store_receipt,
                store_receipt_id=(
                    "phase2b_c1_execution_registry_if_absent_receipt_v1_"
                    + SHA_H
                ),
            ),
        )
    else:
        graph = replace(
            graph,
            time_token=replace(
                graph.time_token,
                logical_genesis_token_id=(
                    "phase2b_c1_signed_logical_genesis_token_v1_" + SHA_H
                ),
            ),
        )
    calls = 0

    class NoCryptoBeforeContentAddress:
        @classmethod
        def from_public_bytes(
            cls, public_key: bytes
        ) -> "NoCryptoBeforeContentAddress":
            nonlocal calls
            calls += 1
            raise AssertionError(public_key)

    monkeypatch.setattr(wire, "_Ed25519PublicKey", NoCryptoBeforeContentAddress)
    result = wire.verify_phase2b_c1_six_role_wire_v1(
        context=graph.context,
        registry=graph.registry,
        store_receipt=graph.store_receipt,
        logical_genesis_token=graph.time_token,
    )
    assert result.reason is expected_reason
    assert calls == 0
    assert result.partial_output_published is False


def test_first_signature_verification_mutation_cannot_change_full_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _build_literal_kat_graph()
    calls = 0

    class MutatingBackend:
        def __init__(self, public_key: Ed25519PublicKey) -> None:
            self.public_key = public_key

        @classmethod
        def from_public_bytes(cls, public_key: bytes) -> "MutatingBackend":
            return cls(Ed25519PublicKey.from_public_bytes(public_key))

        def verify(self, signature: bytes, message: bytes) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                object.__setattr__(graph.registry.body, "role_keys", ())
                object.__setattr__(
                    graph.registry.governance_signature,
                    "signature",
                    b"\x00" * 64,
                )
                object.__setattr__(
                    graph.store_receipt.body,
                    "stored_content_sha256",
                    SHA_H,
                )
                object.__setattr__(graph.time_token.body, "logical_sequence", 9)
            self.public_key.verify(signature, message)

    monkeypatch.setattr(wire, "_Ed25519PublicKey", MutatingBackend)
    result = wire.verify_phase2b_c1_six_role_wire_v1(
        context=graph.context,
        registry=graph.registry,
        store_receipt=graph.store_receipt,
        logical_genesis_token=graph.time_token,
    )
    assert calls == 3
    assert result.disposition is (
        wire.Phase2BC1SixRoleWireDispositionV1.CALLER_SUPPLIED_SIX_ROLE_WIRE_GRAPH_VERIFIED_NOT_AUTHORITY
    )
    assert result.result_id == KAT_RESULT_ID
