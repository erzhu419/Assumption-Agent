from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hegel_machine import phase2b_c1_logical_stage_wire_v1 as logical
from hegel_machine import phase2b_c1_six_role_wire_v1 as six


H = tuple(str(index) * 64 for index in range(1, 10))


@dataclass(frozen=True)
class Graph:
    private_keys: tuple[Ed25519PrivateKey, ...]
    context: six.Phase2BC1CeremonyContextV1
    registry: six.Phase2BC1ExecutionRegistryCandidateV1
    store: six.Phase2BC1StoreIfAbsentReceiptCandidateV1
    genesis: six.Phase2BC1LogicalGenesisTokenCandidateV1


def _public(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )


def _signature(
    *, key: Ed25519PrivateKey, key_id: str, purpose_id: int,
    role: six.Phase2BC1SixRoleV1,
    signature_purpose: six.Phase2BC1SignaturePurposeV1, preimage: bytes,
) -> six.Phase2BC1DetachedSignatureV1:
    return six.Phase2BC1DetachedSignatureV1(
        purpose_id=purpose_id, role=role, signature_purpose=signature_purpose,
        algorithm="Ed25519", key_epoch=1, signer_key_id=key_id,
        signature=key.sign(preimage),
    )


def _build_six_role_graph(*, campaign_hash: str = H[0]) -> Graph:
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
        execution_freeze_manifest_sha256=H[2], attempt_nonce_sha256=H[3],
    )
    context_body = six.Phase2BC1CeremonyContextBodyV1(
        version=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_VERSION,
        authority_class=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_CLASS,
        authority_requirement_id=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_REQUIREMENT_ID,
        actor_profile_content_id=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CONTENT_ID,
        actor_profile_canonical_sha256=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CANONICAL_SHA256,
        campaign_id=campaign_id,
        protocol_id=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_FROZEN_PROTOCOL_ID,
        exact_freeze_id=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_FROZEN_EXACT_FREEZE_ID,
        execution_freeze_manifest_id=freeze_id,
        execution_freeze_manifest_sha256=H[2], attempt_nonce_sha256=H[3],
        attempt_id=attempt_id, recognizer_image_digest="sha256:" + H[4],
        recognizer_source_root_sha256=H[5], scorer_image_digest="sha256:" + H[6],
        scorer_source_root_sha256=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_SCORER_SOURCE_ROOT_SHA256,
    )
    context = six.Phase2BC1CeremonyContextV1(
        body=context_body,
        context_id=six.phase2b_c1_ceremony_context_id_v1(context_body),
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
        version=context_body.version, context_id=context.context_id,
        authority_class=context_body.authority_class,
        authority_requirement_id=context_body.authority_requirement_id,
        actor_profile_content_id=context_body.actor_profile_content_id,
        actor_profile_canonical_sha256=context_body.actor_profile_canonical_sha256,
        campaign_id=context_body.campaign_id, protocol_id=context_body.protocol_id,
        exact_freeze_id=context_body.exact_freeze_id,
        execution_freeze_manifest_id=context_body.execution_freeze_manifest_id,
        execution_freeze_manifest_sha256=context_body.execution_freeze_manifest_sha256,
        attempt_nonce_sha256=context_body.attempt_nonce_sha256,
        attempt_id=context_body.attempt_id,
        recognizer_image_digest=context_body.recognizer_image_digest,
        recognizer_source_root_sha256=context_body.recognizer_source_root_sha256,
        scorer_image_digest=context_body.scorer_image_digest,
        scorer_source_root_sha256=context_body.scorer_source_root_sha256,
        role_keys=role_keys,
    )
    governance = _signature(
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
    store_body = six.Phase2BC1StoreIfAbsentReceiptBodyV1(
        version=context_body.version, context_id=context.context_id,
        authority_requirement_id=context_body.authority_requirement_id,
        actor_profile_content_id=context_body.actor_profile_content_id,
        campaign_id=campaign_id, attempt_id=attempt_id,
        execution_registry_id=registry.execution_registry_id,
        object_key=f"phase2b/c1/actual-unsealed-960/{campaign_id}/attempt-0/execution-registry",
        previous_generation=0, new_generation=1, condition="IF_ABSENT",
        previous_content_sha256_or_null=None,
        stored_content_sha256=hashlib.sha256(
            six.canonical_phase2b_c1_execution_registry_bytes_v1(registry)
        ).hexdigest(),
    )
    store_signature = _signature(
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
        version=context_body.version, context_id=context.context_id,
        authority_requirement_id=context_body.authority_requirement_id,
        actor_profile_content_id=context_body.actor_profile_content_id,
        campaign_id=campaign_id, attempt_id=attempt_id,
        execution_registry_id=registry.execution_registry_id,
        store_receipt_id=store.store_receipt_id,
        time_semantics=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_TIME_SEMANTICS,
        previous_logical_sequence_or_null=None, logical_sequence=0,
        stage=six.Phase2BC1LogicalStageV1.PREREGISTERED, stage_index=0,
    )
    genesis_signature = _signature(
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
    return Graph(private_keys, context, registry, store, genesis)


def _projection_with_id(projection):
    return projection


def _token(graph: Graph, projection, index: int, previous_token_id: str):
    stages = (
        six.Phase2BC1LogicalStageV1.RAW_INPUT_COMMITTED,
        six.Phase2BC1LogicalStageV1.PACKAGE_COMMITMENTS_FROZEN,
        six.Phase2BC1LogicalStageV1.RUN_STARTED,
        six.Phase2BC1LogicalStageV1.RUN_FINISHED,
    )
    body = logical.Phase2BC1LogicalStageTokenBodyV1(
        version=logical.PHASE2B_C1_LOGICAL_STAGE_WIRE_V1_VERSION,
        context_id=graph.context.context_id,
        authority_requirement_id=graph.context.body.authority_requirement_id,
        actor_profile_content_id=graph.context.body.actor_profile_content_id,
        actor_profile_canonical_sha256=graph.context.body.actor_profile_canonical_sha256,
        campaign_id=graph.context.body.campaign_id,
        attempt_id=graph.context.body.attempt_id,
        execution_registry_id=graph.registry.execution_registry_id,
        logical_genesis_token_id=graph.genesis.logical_genesis_token_id,
        legacy_registry_policy_authorization=logical.PHASE2B_C1_LOGICAL_STAGE_WIRE_V1_LEGACY_POLICY_AUTHORIZATION,
        actual_campaign_eligible=False,
        time_semantics=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_TIME_SEMANTICS,
        predecessor_token_id=previous_token_id,
        previous_logical_sequence=index - 1, logical_sequence=index,
        previous_stage=(six.Phase2BC1LogicalStageV1.PREREGISTERED if index == 1 else stages[index - 2]),
        stage=stages[index - 1], stage_index=index,
        event_projection_id=logical.phase2b_c1_logical_stage_event_projection_id_v1(
            projection
        ),
        event_root_sha256=logical.phase2b_c1_logical_stage_event_root_sha256_v1(projection),
    )
    time_key = graph.registry.body.role_keys[3]
    signature = _signature(
        key=graph.private_keys[3], key_id=time_key.key_id, purpose_id=4,
        role=six.Phase2BC1SixRoleV1.TIME_ATTESTOR,
        signature_purpose=six.Phase2BC1SignaturePurposeV1.DECLARED_TIME_TOKEN,
        preimage=logical.phase2b_c1_logical_stage_signature_preimage_v1(
            body, signer_key_id=time_key.key_id
        ),
    )
    return logical.Phase2BC1LogicalStageTokenCandidateV1(
        body=body, time_signature=signature,
        logical_stage_token_id=logical.phase2b_c1_logical_stage_token_id_v1(
            body, signature
        ),
    )


def _build_prefix(graph: Graph):
    common = dict(
        version=logical.PHASE2B_C1_LOGICAL_STAGE_WIRE_V1_VERSION,
        context_id=graph.context.context_id,
        authority_requirement_id=graph.context.body.authority_requirement_id,
        actor_profile_content_id=graph.context.body.actor_profile_content_id,
        campaign_id=graph.context.body.campaign_id,
        attempt_id=graph.context.body.attempt_id,
        execution_registry_id=graph.registry.execution_registry_id,
    )
    raw = _projection_with_id(
        logical.Phase2BC1RawInputCommittedEventProjectionV1(
            **common, input_archive_id="phase2b_recognizer_input_archive_v2_" + H[0],
            input_archive_sha256=H[1], input_archive_bytes=1234,
            raw_input_commitment_sha256=H[2],
            raw_input_commitment_receipt_id="phase2b_c1_raw_input_commitment_receipt_v1_" + H[3],
        )
    )
    token1 = _token(graph, raw, 1, graph.genesis.logical_genesis_token_id)
    package = _projection_with_id(
        logical.Phase2BC1PackageCommitmentsFrozenEventProjectionV1(
            **common, raw_input_committed_token_id=token1.logical_stage_token_id,
            execution_freeze_manifest_id=graph.context.body.execution_freeze_manifest_id,
            execution_freeze_manifest_sha256=graph.context.body.execution_freeze_manifest_sha256,
            input_archive_id=raw.input_archive_id,
            input_archive_sha256=raw.input_archive_sha256,
            recognizer_run_spec_id="phase2b_oci_run_spec_" + H[4],
            answer_commitment_sha256=H[5], gate_commitment_sha256=H[6],
            public_partition_spec_id="phase2b_c1_public_partition_spec_v1_" + H[7],
            public_partition_spec_sha256=H[8],
            package_commitment_manifest_id="phase2b_c1_package_commitment_manifest_v1_" + H[0],
        )
    )
    token2 = _token(graph, package, 2, token1.logical_stage_token_id)
    started = _projection_with_id(
        logical.Phase2BC1RunStartedEventProjectionV1(
            **common, package_commitments_frozen_token_id=token2.logical_stage_token_id,
            recognizer_run_spec_id=package.recognizer_run_spec_id,
            execution_freeze_manifest_id=package.execution_freeze_manifest_id,
            input_archive_id=package.input_archive_id,
            start_operation_nonce_sha256=H[1],
        )
    )
    token3 = _token(graph, started, 3, token2.logical_stage_token_id)
    finished = _projection_with_id(
        logical.Phase2BC1RunFinishedEventProjectionV1(
            **common, run_started_token_id=token3.logical_stage_token_id,
            recognizer_run_spec_id=package.recognizer_run_spec_id,
            execution_freeze_manifest_id=graph.context.body.execution_freeze_manifest_id,
            execution_freeze_manifest_sha256=graph.context.body.execution_freeze_manifest_sha256,
            input_archive_id=raw.input_archive_id,
            input_archive_sha256=raw.input_archive_sha256,
            runtime_receipt_id="phase2b_recognizer_runtime_receipt_v2_" + H[2],
            local_execution_receipt_id="phase2b_recognizer_local_execution_receipt_v3_" + H[3],
            prediction_archive_id="phase2b_recognizer_prediction_archive_v2_" + H[4],
            prediction_archive_sha256=H[5], prediction_archive_bytes=5678,
        )
    )
    token4 = _token(graph, finished, 4, token3.logical_stage_token_id)
    return (raw, package, started, finished), (token1, token2, token3, token4)


def _verify(graph: Graph, projections, tokens):
    return logical.verify_phase2b_c1_logical_stage_prefix_v1(
        context=graph.context, registry=graph.registry, store_receipt=graph.store,
        logical_genesis_token=graph.genesis,
        event_projections=tuple(projections), stage_tokens=tuple(tokens),
    )


@pytest.mark.parametrize("length", (1, 2, 3, 4))
def test_each_contiguous_prefix_reverifies_genesis_and_purpose_4_signatures(length):
    graph = _build_six_role_graph()
    projections, tokens = _build_prefix(graph)
    result = _verify(graph, projections[:length], tokens[:length])
    assert result.disposition is logical.Phase2BC1LogicalStageWireDispositionV1.CALLER_SUPPLIED_CONTIGUOUS_PREFIX_VERIFIED_NOT_AUTHORITY
    assert result.terminal_logical_sequence == length
    assert result.terminal_stage is tokens[length - 1].body.stage
    assert result.stage_token_ids == tuple(item.logical_stage_token_id for item in tokens[:length])
    assert result.legacy_registry_policy_authorized is False
    assert result.actual_campaign_eligible is False
    assert result.actual_execution_claimed is False
    assert result.timeline_observed is False
    assert result.scoring_performed is False
    assert result.formal_gate_evaluation_performed is False
    assert result.formal_c1_report_verified is False


def _oracle_bytes(pairs) -> bytes:
    return json.dumps(pairs, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def test_literal_kat_uses_independent_json_sha256_and_ed25519_oracle():
    graph = _build_six_role_graph()
    projections, tokens = _build_prefix(graph)
    projection = projections[0]
    token = tokens[0]
    projection_pairs = tuple(
        (field.name, getattr(projection, field.name))
        for field in fields(type(projection))
    )
    projection_digest = hashlib.sha256(
        b"HEGEL/PHASE2B/C1/LOGICAL_STAGE_WIRE/EVENT/RAW_INPUT_COMMITTED/V1\x00"
        + _oracle_bytes(projection_pairs)
    ).hexdigest()
    assert logical.phase2b_c1_logical_stage_event_projection_id_v1(
        projection
    ) == "phase2b_c1_raw_input_committed_event_projection_v1_" + projection_digest
    assert projection_digest == (
        "fa05ec6275321b40203adf576e69ddef48075bf43015b96edb7f16a5c9a66bfa"
    )

    body_pairs = tuple(
        (field.name, getattr(token.body, field.name).value if field.name in ("previous_stage", "stage") else getattr(token.body, field.name))
        for field in fields(type(token.body))
    )
    key_id = graph.registry.body.role_keys[3].key_id
    header = (
        ("purpose_id", 4), ("role", "TIME_ATTESTOR"),
        ("signature_purpose", "DECLARED_TIME_TOKEN"), ("algorithm", "Ed25519"),
        ("key_epoch", 1), ("signer_key_id", key_id),
    )
    preimage = b"HEGEL/PHASE2B/C1/LOGICAL_STAGE_WIRE/STAGE_SIGNATURE/V1\x00" + _oracle_bytes((("signature_header", header), ("body", body_pairs)))
    graph.private_keys[3].public_key().verify(token.time_signature.signature, preimage)
    assert hashlib.sha256(preimage).hexdigest() == (
        "1406dd3363f4e885b6331fc4c0f143f19ce4cd937868e63e0b109efc2b381ba7"
    )
    signature_pairs = (
        ("purpose_id", 4), ("role", "TIME_ATTESTOR"),
        ("signature_purpose", "DECLARED_TIME_TOKEN"), ("algorithm", "Ed25519"),
        ("key_epoch", 1), ("signer_key_id", key_id),
        ("signature_hex", token.time_signature.signature.hex()),
    )
    token_digest = hashlib.sha256(
        b"HEGEL/PHASE2B/C1/LOGICAL_STAGE_WIRE/STAGE_TOKEN_ID/V1\x00"
        + _oracle_bytes((("body", body_pairs), ("time_signature", signature_pairs)))
    ).hexdigest()
    assert token.logical_stage_token_id == "phase2b_c1_signed_logical_stage_token_v1_" + token_digest
    assert token_digest == (
        "98493529dd34b4271614a55d167fd249b2ec86d0b419b9f5c929a20b7b85735f"
    )


@pytest.mark.parametrize("length", (0, 5))
def test_prefix_length_is_one_through_four_only(length):
    graph = _build_six_role_graph()
    projections, tokens = _build_prefix(graph)
    if length == 5:
        projections = projections + (projections[-1],)
        tokens = tokens + (tokens[-1],)
    result = _verify(graph, projections[:length], tokens[:length])
    assert result.reason is logical.Phase2BC1LogicalStageWireReasonV1.PREFIX_LENGTH_INVALID


def test_skip_reorder_and_exact_projection_type_are_rejected():
    graph = _build_six_role_graph()
    projections, tokens = _build_prefix(graph)
    assert _verify(graph, projections[1:2], tokens[1:2]).reason is logical.Phase2BC1LogicalStageWireReasonV1.GRAPH_BINDING_INVALID
    assert _verify(graph, (projections[1], projections[0]), (tokens[1], tokens[0])).reason is logical.Phase2BC1LogicalStageWireReasonV1.GRAPH_BINDING_INVALID
    assert logical.verify_phase2b_c1_logical_stage_prefix_v1(
        context=graph.context, registry=graph.registry, store_receipt=graph.store,
        logical_genesis_token=graph.genesis,
        event_projections=[projections[0]], stage_tokens=(tokens[0],),
    ).reason is logical.Phase2BC1LogicalStageWireReasonV1.WRONG_INPUT_TYPE


def test_cross_campaign_token_splice_is_rejected_even_when_resigned():
    graph = _build_six_role_graph()
    other = _build_six_role_graph(campaign_hash=H[8])
    projections, tokens = _build_prefix(graph)
    other_projections, other_tokens = _build_prefix(other)
    result = _verify(graph, (other_projections[0],), (other_tokens[0],))
    assert result.reason is logical.Phase2BC1LogicalStageWireReasonV1.GRAPH_BINDING_INVALID


def test_projection_and_token_content_address_mutations_reject():
    graph = _build_six_role_graph()
    projections, tokens = _build_prefix(graph)
    changed_projection = replace(projections[0], input_archive_bytes=999)
    assert _verify(graph, (changed_projection,), (tokens[0],)).reason is logical.Phase2BC1LogicalStageWireReasonV1.GRAPH_BINDING_INVALID
    changed_token = replace(tokens[0], logical_stage_token_id="phase2b_c1_signed_logical_stage_token_v1_" + H[8])
    assert _verify(graph, (projections[0],), (changed_token,)).reason is logical.Phase2BC1LogicalStageWireReasonV1.GRAPH_BINDING_INVALID


def test_valid_content_address_with_invalid_signature_rejects_as_signature_invalid():
    graph = _build_six_role_graph()
    projections, tokens = _build_prefix(graph)
    bad_signature = replace(tokens[0].time_signature, signature=b"\x00" * 64)
    bad_token = replace(
        tokens[0], time_signature=bad_signature,
        logical_stage_token_id=logical.phase2b_c1_logical_stage_token_id_v1(tokens[0].body, bad_signature),
    )
    result = _verify(graph, (projections[0],), (bad_token,))
    assert result.reason is logical.Phase2BC1LogicalStageWireReasonV1.SIGNATURE_INVALID


def test_backend_unavailable_fails_closed(monkeypatch):
    graph = _build_six_role_graph()
    projections, tokens = _build_prefix(graph)
    monkeypatch.setattr(logical, "_Ed25519PublicKey", None)
    result = _verify(graph, projections[:1], tokens[:1])
    assert result.reason is logical.Phase2BC1LogicalStageWireReasonV1.SIGNATURE_BACKEND_UNAVAILABLE
    assert result.partial_output_published is False


def test_complete_snapshot_is_not_reread_after_genesis_verification_starts(monkeypatch):
    graph = _build_six_role_graph()
    projections, tokens = _build_prefix(graph)
    original = six.verify_phase2b_c1_six_role_wire_v1

    def mutate_original_then_verify(**kwargs):
        object.__setattr__(tokens[0].body, "event_root_sha256", H[8])
        return original(**kwargs)

    monkeypatch.setattr(six, "verify_phase2b_c1_six_role_wire_v1", mutate_original_then_verify)
    result = _verify(graph, projections[:1], tokens[:1])
    assert result.disposition is logical.Phase2BC1LogicalStageWireDispositionV1.CALLER_SUPPLIED_CONTIGUOUS_PREFIX_VERIFIED_NOT_AUTHORITY


def test_old_local_execution_receipt_is_never_promoted_to_purpose_5_or_actual():
    graph = _build_six_role_graph()
    projections, tokens = _build_prefix(graph)
    result = _verify(graph, projections, tokens)
    assert projections[-1].local_execution_receipt_id.startswith("phase2b_recognizer_local_execution_receipt_v3_")
    assert result.local_execution_receipt_is_purpose_5_receipt is False
    assert result.run_finish_observed is False
    assert result.actual_execution_claimed is False


def test_result_is_private_and_rejections_publish_no_partial_ids():
    with pytest.raises(TypeError, match="privately issued"):
        logical.Phase2BC1LogicalStageWireVerificationV1()
    graph = _build_six_role_graph()
    projections, tokens = _build_prefix(graph)
    result = _verify(graph, (), ())
    assert result.result_id is None
    assert result.stage_token_ids == ()
    assert result.event_projection_ids == ()
    assert result.partial_output_published is False


def test_public_surface_and_no_operation_imports_are_narrow():
    expected = {
        "PHASE2B_C1_LOGICAL_STAGE_WIRE_V1_VERSION",
        "PHASE2B_C1_LOGICAL_STAGE_WIRE_V1_CLAIM_LEVEL",
        "PHASE2B_C1_LOGICAL_STAGE_WIRE_V1_LEGACY_POLICY_AUTHORIZATION",
        "PHASE2B_C1_LOGICAL_STAGE_WIRE_V1_ACTUAL_CAMPAIGN_ELIGIBLE",
        "PHASE2B_C1_LOGICAL_STAGE_WIRE_V1_ED25519_BACKEND_AVAILABLE",
        "Phase2BC1LogicalStageWireDispositionV1",
        "Phase2BC1LogicalStageWireReasonV1",
        "Phase2BC1RawInputCommittedEventProjectionV1",
        "Phase2BC1PackageCommitmentsFrozenEventProjectionV1",
        "Phase2BC1RunStartedEventProjectionV1",
        "Phase2BC1RunFinishedEventProjectionV1",
        "Phase2BC1LogicalStageEventProjectionV1",
        "Phase2BC1LogicalStageTokenBodyV1",
        "Phase2BC1LogicalStageTokenCandidateV1",
        "Phase2BC1LogicalStageWireVerificationV1",
        "phase2b_c1_logical_stage_event_projection_id_v1",
        "phase2b_c1_logical_stage_event_root_sha256_v1",
        "canonical_phase2b_c1_logical_stage_event_projection_bytes_v1",
        "phase2b_c1_logical_stage_signature_preimage_v1",
        "phase2b_c1_logical_stage_token_id_v1",
        "canonical_phase2b_c1_logical_stage_token_bytes_v1",
        "verify_phase2b_c1_logical_stage_prefix_v1",
    }
    assert set(logical.__all__) == expected
