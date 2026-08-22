from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hegel_machine.rfc8785_v1 import canonicalize_rfc8785_v1
import hegel_machine.phase2b_formal_wire_covert_step5_authority_v2 as authority_v2
import hegel_machine.phase2b_formal_wire_covert_step5_owner_status_v1 as status


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OWNER_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "phase2b_c1_owner_actual_unsealed_960_authorization_v1.json"
)
OWNER_SOURCE = (
    PROJECT_ROOT
    / "src"
    / "hegel_machine"
    / "phase2b_c1_owner_actual_unsealed_960_authority_v1.py"
)

ROLE_KEY_DOMAIN = b"HEGEL/PHASE2B/STEP5/ROLE_KEY/V1\x00"
REQUEST_ID_DOMAIN = b"HEGEL/PHASE2B/STEP5/LOGICAL_TIME/REQUEST_ID/V1\x00"
TOKEN_SIGNATURE_DOMAIN = b"HEGEL/PHASE2B/STEP5/SIGN/LOGICAL_TIME_TOKEN/V1\x00"
TOKEN_ID_DOMAIN = b"HEGEL/PHASE2B/STEP5/LOGICAL_TIME/TOKEN_ID/V1\x00"
EVENT_ID_DOMAIN = b"HEGEL/PHASE2B/STEP5/OWNER_STATUS/EVENT_ID/V1\x00"


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _address(payload: bytes) -> str:
    return "sha256:" + _digest(payload)


def _content_id(prefix: str, domain: bytes, subject: object) -> str:
    return prefix + _digest(domain + canonicalize_rfc8785_v1(subject))


def _public_bytes(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )


def _logical_key_id(public_key: bytes) -> str:
    subject = {
        "schema_version": "hegel-machine-phase2b-step5-role-key/1",
        "role": status.LOGICAL_TIME_ROLE_V1,
        "control_domain": status.LOGICAL_TIME_CONTROL_DOMAIN_V1,
        "authorized_signature_purposes": [
            status.LOGICAL_TIME_SIGNATURE_PURPOSE_V1
        ],
        "algorithm": status.ED25519_ALGORITHM_V1,
        "key_epoch": 1,
        "public_key_hex": public_key.hex(),
    }
    return _content_id(status.ROLE_KEY_PREFIX_V1, ROLE_KEY_DOMAIN, subject)


def _seed(label: str) -> bytes:
    return hashlib.sha256(label.encode("ascii")).digest()


def _identity(prefix: str, label: str) -> str:
    return prefix + hashlib.sha256(label.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class Context:
    root: Path
    owner_payload: bytes
    authority_profile_id: str
    signer_registry_id: str
    attempt_context_id: str
    governance_private_key: Ed25519PrivateKey
    governance_public_key: bytes
    governance_key_id: str
    logical_time_private_key: Ed25519PrivateKey
    logical_time_public_key: bytes
    logical_time_key_id: str
    cas_signer_key_id: str


@dataclass(frozen=True)
class AppendArtifacts:
    decision_payload: bytes
    cas_reference_payload: bytes
    logical_reference_payload: bytes
    logical_token_payload: bytes
    logical_token_id: str
    result: status.OwnerStatusServiceAppendV1


@dataclass(frozen=True)
class CompleteChain:
    context: Context
    genesis_head_payload: bytes
    active: AppendArtifacts
    revoked: AppendArtifacts


def _context(tmp_path: Path, label: str = "primary") -> Context:
    governance_private = Ed25519PrivateKey.from_private_bytes(
        _seed(f"governance-{label}")
    )
    governance_public = _public_bytes(governance_private)
    logical_private = Ed25519PrivateKey.from_private_bytes(
        _seed(f"logical-time-{label}")
    )
    logical_public = _public_bytes(logical_private)
    root = tmp_path / f"ledger-{label}"
    root.mkdir()
    return Context(
        root=root,
        owner_payload=OWNER_CONFIG.read_bytes(),
        authority_profile_id=_identity(
            status.AUTHORITY_PROFILE_PREFIX_V2, f"profile-{label}"
        ),
        signer_registry_id=_identity(
            status.SIGNER_REGISTRY_PREFIX_V2, f"registry-{label}"
        ),
        attempt_context_id=f"step5-attempt-{label}",
        governance_private_key=governance_private,
        governance_public_key=governance_public,
        governance_key_id=status.derive_required_governance_role_key_id_v1(
            governance_public
        ),
        logical_time_private_key=logical_private,
        logical_time_public_key=logical_public,
        logical_time_key_id=_logical_key_id(logical_public),
        cas_signer_key_id=_identity(status.ROLE_KEY_PREFIX_V1, f"cas-{label}"),
    )


def _open_ledger(
    context: Context, *, initialize_empty: bool
) -> status.OwnerStatusLedgerV1:
    return status.OwnerStatusLedgerV1(
        str(context.root),
        authority_profile_id=context.authority_profile_id,
        signer_registry_id=context.signer_registry_id,
        governance_key_id=context.governance_key_id,
        governance_key_epoch=1,
        attempt_context_id=context.attempt_context_id,
        governance_public_key=context.governance_public_key,
        logical_time_public_key=context.logical_time_public_key,
        owner_authorization_payload=context.owner_payload,
        initialize_empty=initialize_empty,
    )


def _logical_token(
    context: Context,
    decision_payload: bytes,
    *,
    sequence: int,
    prior_token_id: str,
    request_nonce: str,
    request_id_override: str | None = None,
    private_key: Ed25519PrivateKey | None = None,
) -> tuple[bytes, str]:
    decision = status.load_owner_status_decision_v1(decision_payload)
    signing_key = private_key or context.logical_time_private_key
    public_key = _public_bytes(signing_key)
    logical_key_id = _logical_key_id(public_key)
    request_subject = {
        "schema_version": status.LOGICAL_TIME_REQUEST_SCHEMA_V1,
        "authority_profile_id": decision.authority_profile_id,
        "signer_registry_id": decision.signer_registry_id,
        "logical_time_key_id": logical_key_id,
        "logical_time_key_epoch": 1,
        "attempt_context_id": decision.attempt_context_id,
        "request_nonce": request_nonce,
        "signature_purpose": status.LOGICAL_TIME_SIGNATURE_PURPOSE_V1,
        "purpose": status.LOGICAL_TIME_EVENT_PURPOSE_V1,
        "bound_artifact_id": decision.decision_id,
        "bound_artifact_sha256": _address(decision_payload),
        "prior_token_id": prior_token_id,
        "next_sequence": sequence,
    }
    request_id = request_id_override or _content_id(
        status.LOGICAL_TIME_REQUEST_ID_PREFIX_V1,
        REQUEST_ID_DOMAIN,
        request_subject,
    )
    unsigned = {
        "schema_version": status.LOGICAL_TIME_TOKEN_SCHEMA_V1,
        "authority_profile_id": decision.authority_profile_id,
        "signer_registry_id": decision.signer_registry_id,
        "logical_time_key_id": logical_key_id,
        "logical_time_key_epoch": 1,
        "attempt_context_id": decision.attempt_context_id,
        "request_id": request_id,
        "request_nonce": request_nonce,
        "signature_purpose": status.LOGICAL_TIME_SIGNATURE_PURPOSE_V1,
        "purpose": status.LOGICAL_TIME_EVENT_PURPOSE_V1,
        "bound_artifact_id": decision.decision_id,
        "bound_artifact_sha256": _address(decision_payload),
        "prior_token_id": prior_token_id,
        "sequence": sequence,
        "algorithm": status.ED25519_ALGORITHM_V1,
    }
    preimage = TOKEN_SIGNATURE_DOMAIN + canonicalize_rfc8785_v1(
        {
            "schema_version": "hegel-machine-phase2b-step5-signature-preimage/1",
            "artifact_kind": "logical_time",
            "role": status.LOGICAL_TIME_ROLE_V1,
            "purpose": status.LOGICAL_TIME_SIGNATURE_PURPOSE_V1,
            "key_epoch": 1,
            "signer_key_id": logical_key_id,
            "body": unsigned,
        }
    )
    signature_hex = signing_key.sign(preimage).hex()
    id_subject = {**unsigned, "signature_hex": signature_hex}
    token_id = _content_id(
        status.LOGICAL_TIME_TOKEN_ID_PREFIX_V1,
        TOKEN_ID_DOMAIN,
        id_subject,
    )
    return canonicalize_rfc8785_v1({**id_subject, "token_id": token_id}), token_id


def _logical_reference(
    context: Context,
    decision_payload: bytes,
    token_payload: bytes,
    *,
    public_key: bytes | None = None,
) -> bytes:
    return status.build_logical_time_dependency_reference_v1(
        decision_payload=decision_payload,
        logical_time_token_payload=token_payload,
        logical_time_public_key=public_key or context.logical_time_public_key,
    )


def _cas_reference(
    context: Context,
    decision_payload: bytes,
    *,
    checkpoint_index: int,
) -> bytes:
    receipt = _address(f"receipt-{checkpoint_index}".encode("ascii"))
    checkpoint = _address(f"checkpoint-{checkpoint_index}".encode("ascii"))
    head = _address(f"head-{checkpoint_index}".encode("ascii"))
    return status.build_cas_store_dependency_reference_v1(
        decision_payload=decision_payload,
        retention_session_id="owner-status-retention-session",
        chain_id="owner-status-cas-chain",
        signer_key_id=context.cas_signer_key_id,
        signed_receipt_schema_version=status.CAS_SIGNED_RECEIPT_SCHEMA_V1,
        signed_receipt_id=receipt,
        signed_receipt_canonical_sha256=receipt,
        checkpoint_schema_version=status.CAS_CHECKPOINT_SCHEMA_V1,
        checkpoint_id=checkpoint,
        checkpoint_canonical_sha256=checkpoint,
        checkpoint_index=checkpoint_index,
        head_schema_version=status.CAS_HEAD_SCHEMA_V1,
        head_id=head,
        head_canonical_sha256=head,
        head_last_checkpoint_id=checkpoint,
        head_checkpoint_count=checkpoint_index + 1,
    )


def _append(
    ledger: status.OwnerStatusLedgerV1,
    context: Context,
    *,
    prior_token_id: str,
    decision_nonce: str,
    request_nonce: str,
    token_sequence: int,
    cas_checkpoint_index: int,
    token_private_key: Ed25519PrivateKey | None = None,
) -> AppendArtifacts:
    previous_head = ledger.current_head_bytes()
    decision_payload = status.build_owner_status_decision_v1(
        previous_head,
        owner_authorization_payload=context.owner_payload,
        reason_code=(
            "INITIAL_OWNER_STATUS"
            if token_sequence == 0
            else "OWNER_AUTHORIZATION_REVOKED"
        ),
        decision_nonce=decision_nonce,
    )
    token_payload, token_id = _logical_token(
        context,
        decision_payload,
        sequence=token_sequence,
        prior_token_id=prior_token_id,
        request_nonce=request_nonce,
        private_key=token_private_key,
    )
    token_public = (
        _public_bytes(token_private_key)
        if token_private_key is not None
        else context.logical_time_public_key
    )
    logical_reference = _logical_reference(
        context, decision_payload, token_payload, public_key=token_public
    )
    cas_reference = _cas_reference(
        context,
        decision_payload,
        checkpoint_index=cas_checkpoint_index,
    )
    result = ledger.append_service_side_v1(
        decision_payload=decision_payload,
        cas_store_dependency_reference_payload=cas_reference,
        logical_time_dependency_reference_payload=logical_reference,
        logical_time_token_payload=token_payload,
        expected_previous_head_payload=previous_head,
        governance_private_key=context.governance_private_key,
    )
    return AppendArtifacts(
        decision_payload=decision_payload,
        cas_reference_payload=cas_reference,
        logical_reference_payload=logical_reference,
        logical_token_payload=token_payload,
        logical_token_id=token_id,
        result=result,
    )


def _complete_chain(tmp_path: Path, label: str = "complete") -> CompleteChain:
    context = _context(tmp_path, label)
    with _open_ledger(context, initialize_empty=True) as ledger:
        genesis = ledger.current_head_bytes()
        active = _append(
            ledger,
            context,
            prior_token_id=status.LOGICAL_TIME_GENESIS_TOKEN_ID_V1,
            decision_nonce="10" * 32,
            request_nonce="20" * 32,
            token_sequence=0,
            cas_checkpoint_index=0,
        )
        revoked = _append(
            ledger,
            context,
            prior_token_id=active.logical_token_id,
            decision_nonce="30" * 32,
            request_nonce="40" * 32,
            token_sequence=1,
            cas_checkpoint_index=1,
        )
    return CompleteChain(context, genesis, active, revoked)


def _replay(chain: CompleteChain) -> status.OwnerStatusProjectionV1:
    return status.verify_owner_status_ledger_replay_v1(
        chain.genesis_head_payload,
        (
            chain.active.result.checkpoint_payload,
            chain.revoked.result.checkpoint_payload,
        ),
        (chain.active.result.head_payload, chain.revoked.result.head_payload),
        (
            chain.active.logical_token_payload,
            chain.revoked.logical_token_payload,
        ),
        owner_authorization_payload=chain.context.owner_payload,
        governance_public_key=chain.context.governance_public_key,
        logical_time_public_key=chain.context.logical_time_public_key,
        expected_authority_profile_id=chain.context.authority_profile_id,
        expected_signer_registry_id=chain.context.signer_registry_id,
        expected_governance_key_id=chain.context.governance_key_id,
        expected_governance_key_epoch=1,
        expected_attempt_context_id=chain.context.attempt_context_id,
    )


@pytest.fixture
def chain(tmp_path: Path) -> CompleteChain:
    return _complete_chain(tmp_path)


def test_exact_owner_artifact_sha_id_and_full_fourteen_field_binding() -> None:
    payload = OWNER_CONFIG.read_bytes()
    assert _digest(payload) == status.OWNER_AUTHORIZATION_PAYLOAD_SHA256_V1
    assert _digest(OWNER_SOURCE.read_bytes()) == status.OWNER_AUTHORIZATION_SOURCE_SHA256_V1
    assert len(status.EXACT_OWNER_AUTHORIZATION_BINDING_V1) == 14
    wrapper = json.loads(payload)
    body = wrapper["authorization_body"]
    expected = dict(status.EXACT_OWNER_AUTHORIZATION_BINDING_V1)
    assert wrapper["authorization_id"] == expected["authorization_id"]
    assert body["schema_version"] == expected["authorization_schema"]
    assert body["exact_contract_bindings"] == {
        key: expected[key]
        for key in (
            "protocol_id",
            "exact_freeze_version",
            "exact_freeze_id",
            "authority_class",
            "authority_requirement_id",
            "actor_profile_content_id",
            "actor_profile_canonical_sha256",
            "logical_stage_wire_schema_id",
            "logical_stage_wire_policy_id",
            "stage_event_contract_manifest_id",
        )
    }
    status.verify_exact_owner_authorization_payload_v1(payload)


def test_owner_payload_raw_byte_drift_fails_closed() -> None:
    with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1):
        status.verify_exact_owner_authorization_payload_v1(
            OWNER_CONFIG.read_bytes() + b" "
        )


@pytest.mark.parametrize(
    "binding_field", tuple(status.EXACT_OWNER_AUTHORIZATION_BINDING_V1)
)
def test_each_of_fourteen_owner_binding_fields_is_exact(
    chain: CompleteChain, binding_field: str
) -> None:
    body = json.loads(chain.active.decision_payload)
    observed = body["owner_authorization_binding"][binding_field]
    body["owner_authorization_binding"][binding_field] = str(observed) + "-drift"
    with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1):
        status.load_owner_status_decision_v1(canonicalize_rfc8785_v1(body))


def test_governance_role_key_derivation_freezes_role_domain_two_purposes() -> None:
    private_key = Ed25519PrivateKey.from_private_bytes(_seed("governance-kat"))
    public_key = _public_bytes(private_key)
    subject = {
        "schema_version": "hegel-machine-phase2b-step5-role-key/1",
        "role": "GOVERNANCE_ANCHOR",
        "control_domain": "STEP5_GOVERNANCE_CONTROL",
        "authorized_signature_purposes": [
            "SIGN_SIGNER_REGISTRY",
            "SIGN_OWNER_AUTHORIZATION_STATUS_EVENT",
        ],
        "algorithm": "Ed25519",
        "key_epoch": 1,
        "public_key_hex": public_key.hex(),
    }
    expected = _content_id(status.ROLE_KEY_PREFIX_V1, ROLE_KEY_DOMAIN, subject)
    assert status.AUTHORITY_PROFILE_PREFIX_V2 == authority_v2.AUTHORITY_PROFILE_PREFIX_V2
    assert status.SIGNER_REGISTRY_PREFIX_V2 == authority_v2.SIGNER_REGISTRY_PREFIX_V2
    assert status.REQUIRED_GOVERNANCE_PURPOSES_V1 == tuple(
        subject["authorized_signature_purposes"]
    )
    assert status.REQUIRED_GOVERNANCE_PURPOSES_V1 == (
        authority_v2.ROLE_SIGNATURE_PURPOSES_V2["GOVERNANCE_ANCHOR"]
    )
    assert status.derive_required_governance_role_key_id_v1(public_key) == expected
    assert authority_v2.phase2b_step5_role_key_id_v2(subject) == expected
    with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1):
        status.derive_required_governance_role_key_id_v1(public_key, key_epoch=True)


def test_owner_status_schema_one_accepts_only_profile_v2_registry_v2_lineage(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, "v2-lineage")
    genesis = status.build_owner_status_genesis_head_v1(
        authority_profile_id=context.authority_profile_id,
        signer_registry_id=context.signer_registry_id,
        governance_key_id=context.governance_key_id,
        governance_key_epoch=1,
        attempt_context_id=context.attempt_context_id,
        governance_public_key=context.governance_public_key,
        owner_authorization_payload=context.owner_payload,
    )
    decoded = json.loads(genesis)
    assert decoded["schema_version"] == status.OWNER_STATUS_HEAD_SCHEMA_V1
    assert decoded["authority_profile_id"].startswith(
        authority_v2.AUTHORITY_PROFILE_PREFIX_V2
    )
    assert decoded["signer_registry_id"].startswith(
        authority_v2.SIGNER_REGISTRY_PREFIX_V2
    )

    old_profile = _identity("phase2b_step5_authority_profile_v1_", "legacy")
    old_registry = _identity("phase2b_step5_signer_registry_v1_", "legacy")
    for profile_id, registry_id in (
        (old_profile, old_registry),
        (old_profile, context.signer_registry_id),
        (context.authority_profile_id, old_registry),
    ):
        with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1):
            status.build_owner_status_genesis_head_v1(
                authority_profile_id=profile_id,
                signer_registry_id=registry_id,
                governance_key_id=context.governance_key_id,
                governance_key_epoch=1,
                attempt_context_id=context.attempt_context_id,
                governance_public_key=context.governance_public_key,
                owner_authorization_payload=context.owner_payload,
            )


def test_retargeted_logical_time_dependency_carries_only_v2_graph_ids(
    chain: CompleteChain,
) -> None:
    reference = status.load_logical_time_dependency_reference_v1(
        chain.active.logical_reference_payload
    )
    token = json.loads(chain.active.logical_token_payload)
    for observed in (reference.authority_profile_id, token["authority_profile_id"]):
        assert observed.startswith(status.AUTHORITY_PROFILE_PREFIX_V2)
    for observed in (reference.signer_registry_id, token["signer_registry_id"]):
        assert observed.startswith(status.SIGNER_REGISTRY_PREFIX_V2)


def test_positive_active_then_terminal_revoked_and_no_restore(
    chain: CompleteChain,
) -> None:
    active_head = status.load_owner_status_head_v1(chain.active.result.head_payload)
    revoked_head = status.load_owner_status_head_v1(chain.revoked.result.head_payload)
    assert (active_head.event_count, active_head.latest_effective_status) == (1, "ACTIVE")
    assert active_head.terminal is False
    assert (revoked_head.event_count, revoked_head.latest_effective_status) == (
        2,
        "REVOKED",
    )
    assert revoked_head.terminal is True
    with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1):
        status.build_owner_status_decision_v1(
            chain.revoked.result.head_payload,
            owner_authorization_payload=chain.context.owner_payload,
            reason_code="RESTORE_NOT_ALLOWED",
            decision_nonce="50" * 32,
        )


def test_positive_active_only_bytes_replay(chain: CompleteChain) -> None:
    projection = status.verify_owner_status_ledger_replay_v1(
        chain.genesis_head_payload,
        (chain.active.result.checkpoint_payload,),
        (chain.active.result.head_payload,),
        (chain.active.logical_token_payload,),
        owner_authorization_payload=chain.context.owner_payload,
        governance_public_key=chain.context.governance_public_key,
        logical_time_public_key=chain.context.logical_time_public_key,
        expected_authority_profile_id=chain.context.authority_profile_id,
        expected_signer_registry_id=chain.context.signer_registry_id,
        expected_governance_key_id=chain.context.governance_key_id,
        expected_governance_key_epoch=1,
        expected_attempt_context_id=chain.context.attempt_context_id,
    )
    assert projection.event_count == 1
    assert projection.latest_sequence == 0
    assert projection.latest_effective_status == status.STATUS_ACTIVE_V1
    assert projection.authority_evidence is False


def test_v2_cross_graph_expected_profile_registry_pairs_are_rejected(
    chain: CompleteChain,
) -> None:
    foreign_profile = _identity(status.AUTHORITY_PROFILE_PREFIX_V2, "cross-graph")
    foreign_registry = _identity(status.SIGNER_REGISTRY_PREFIX_V2, "cross-graph")
    for profile_id, registry_id in (
        (foreign_profile, chain.context.signer_registry_id),
        (chain.context.authority_profile_id, foreign_registry),
    ):
        with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1):
            status.verify_owner_status_ledger_replay_v1(
                chain.genesis_head_payload,
                (chain.active.result.checkpoint_payload,),
                (chain.active.result.head_payload,),
                (chain.active.logical_token_payload,),
                owner_authorization_payload=chain.context.owner_payload,
                governance_public_key=chain.context.governance_public_key,
                logical_time_public_key=chain.context.logical_time_public_key,
                expected_authority_profile_id=profile_id,
                expected_signer_registry_id=registry_id,
                expected_governance_key_id=chain.context.governance_key_id,
                expected_governance_key_epoch=1,
                expected_attempt_context_id=chain.context.attempt_context_id,
            )


def test_complete_event_set_root_commits_ordered_prior_event_and_checkpoint_chain(
    chain: CompleteChain,
) -> None:
    active_cp = status.load_owner_status_checkpoint_v1(
        chain.active.result.checkpoint_payload
    )
    revoked_cp = status.load_owner_status_checkpoint_v1(
        chain.revoked.result.checkpoint_payload
    )
    active_head = status.load_owner_status_head_v1(chain.active.result.head_payload)
    assert active_cp.previous_event_id == status.GENESIS_EVENT_ID_V1
    assert active_cp.previous_checkpoint_id == status.GENESIS_CHECKPOINT_ID_V1
    assert active_cp.previous_complete_event_set_root == (
        status.GENESIS_COMPLETE_EVENT_SET_ROOT_V1
    )
    assert revoked_cp.previous_event_id == active_cp.event_id
    assert revoked_cp.previous_checkpoint_id == active_cp.checkpoint_id
    assert revoked_cp.previous_head_id == active_head.head_id
    assert revoked_cp.previous_complete_event_set_root == active_cp.complete_event_set_root
    assert revoked_cp.complete_event_set_root != active_cp.complete_event_set_root


def test_projection_keeps_every_authority_r01_readiness_execution_claim_false(
    chain: CompleteChain,
) -> None:
    projection = _replay(chain)
    assert projection.governance_signatures_verified is True
    assert projection.logical_time_token_signatures_verified is True
    assert projection.store_dependency_candidate_references_exact is True
    assert projection.governance_status_purpose_authorized_by_pinned_registry is False
    assert projection.owner_signature_authenticated is False
    assert projection.store_dependencies_verified is False
    assert projection.latest_authority_status_verified is False
    assert projection.r01_satisfied is False
    assert projection.readiness_node_transition_issued is False
    assert projection.execution_authorized is False
    assert projection.authority_evidence is False
    assert status.KNOWN_INTEGRATION_BLOCKERS_V1


def test_cas_opaque_reference_cannot_claim_verified(chain: CompleteChain) -> None:
    mapping = json.loads(chain.active.cas_reference_payload)
    mapping["verification_status"] = "STORE_DEPENDENCY_VERIFIED"
    with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1):
        status.load_cas_store_dependency_reference_v1(
            canonicalize_rfc8785_v1(mapping)
        )


def test_original_signed_777_all_zero_request_arbitrary_prior_kat_is_rejected(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, "request-kat")
    with _open_ledger(context, initialize_empty=True) as ledger:
        decision = status.build_owner_status_decision_v1(
            ledger.current_head_bytes(),
            owner_authorization_payload=context.owner_payload,
            reason_code="INITIAL_OWNER_STATUS",
            decision_nonce="60" * 32,
        )
        token, _token_id = _logical_token(
            context,
            decision,
            sequence=777,
            prior_token_id=status.LOGICAL_TIME_TOKEN_ID_PREFIX_V1 + ("9" * 64),
            request_nonce="70" * 32,
            request_id_override=status.LOGICAL_TIME_REQUEST_ID_PREFIX_V1 + ("0" * 64),
        )
        with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1) as failure:
            _logical_reference(context, decision, token)
        assert failure.value.code == status.FAIL_CONTENT_ID_V1


def test_valid_request_but_sequence_777_fails_before_any_publication(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, "sequence-kat")
    with _open_ledger(context, initialize_empty=True) as ledger:
        previous_head = ledger.current_head_bytes()
        decision = status.build_owner_status_decision_v1(
            previous_head,
            owner_authorization_payload=context.owner_payload,
            reason_code="INITIAL_OWNER_STATUS",
            decision_nonce="80" * 32,
        )
        token, _token_id = _logical_token(
            context,
            decision,
            sequence=777,
            prior_token_id=status.LOGICAL_TIME_TOKEN_ID_PREFIX_V1 + ("9" * 64),
            request_nonce="90" * 32,
        )
        logical_reference = _logical_reference(context, decision, token)
        cas_reference = _cas_reference(context, decision, checkpoint_index=0)
        with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1):
            ledger.append_service_side_v1(
                decision_payload=decision,
                cas_store_dependency_reference_payload=cas_reference,
                logical_time_dependency_reference_payload=logical_reference,
                logical_time_token_payload=token,
                expected_previous_head_payload=previous_head,
                governance_private_key=context.governance_private_key,
            )
        assert len(tuple((context.root / "heads").iterdir())) == 1
        assert not tuple((context.root / "checkpoints").iterdir())
        assert not tuple((context.root / "logical-time-tokens").iterdir())


@pytest.mark.parametrize("splice", ("prior", "sequence", "request_nonce"))
def test_revoked_logical_time_prior_sequence_and_nonce_splices_fail_prepublish(
    tmp_path: Path, splice: str
) -> None:
    context = _context(tmp_path, f"splice-{splice}")
    with _open_ledger(context, initialize_empty=True) as ledger:
        active = _append(
            ledger,
            context,
            prior_token_id=status.LOGICAL_TIME_GENESIS_TOKEN_ID_V1,
            decision_nonce="a0" * 32,
            request_nonce="b0" * 32,
            token_sequence=0,
            cas_checkpoint_index=0,
        )
        previous_head = ledger.current_head_bytes()
        decision = status.build_owner_status_decision_v1(
            previous_head,
            owner_authorization_payload=context.owner_payload,
            reason_code="OWNER_AUTHORIZATION_REVOKED",
            decision_nonce="c0" * 32,
        )
        token_sequence = 2 if splice == "sequence" else 1
        prior = (
            status.LOGICAL_TIME_GENESIS_TOKEN_ID_V1
            if splice == "prior"
            else active.logical_token_id
        )
        request_nonce = "b0" * 32 if splice == "request_nonce" else "d0" * 32
        token, _token_id = _logical_token(
            context,
            decision,
            sequence=token_sequence,
            prior_token_id=prior,
            request_nonce=request_nonce,
        )
        logical_reference = _logical_reference(context, decision, token)
        cas_reference = _cas_reference(context, decision, checkpoint_index=1)
        with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1):
            ledger.append_service_side_v1(
                decision_payload=decision,
                cas_store_dependency_reference_payload=cas_reference,
                logical_time_dependency_reference_payload=logical_reference,
                logical_time_token_payload=token,
                expected_previous_head_payload=previous_head,
                governance_private_key=context.governance_private_key,
            )
        assert len(tuple((context.root / "heads").iterdir())) == 2
        assert len(tuple((context.root / "checkpoints").iterdir())) == 1
        assert len(tuple((context.root / "logical-time-tokens").iterdir())) == 1


def test_governance_and_logical_time_signatures_are_real_ed25519(
    chain: CompleteChain,
) -> None:
    event = status.verify_signed_owner_status_event_v1(
        chain.active.result.signed_event_payload,
        governance_public_key=chain.context.governance_public_key,
        logical_time_token_payload=chain.active.logical_token_payload,
        logical_time_public_key=chain.context.logical_time_public_key,
        expected_authority_profile_id=chain.context.authority_profile_id,
        expected_signer_registry_id=chain.context.signer_registry_id,
        expected_governance_key_id=chain.context.governance_key_id,
        expected_governance_key_epoch=1,
        expected_attempt_context_id=chain.context.attempt_context_id,
    )
    assert event.signature_preimage().startswith(
        b"HEGEL/PHASE2B/STEP5/SIGN/OWNER_AUTHORIZATION_STATUS_EVENT/"
        b"GOVERNANCE/V1\x00"
    )
    wrong_public = _public_bytes(
        Ed25519PrivateKey.from_private_bytes(_seed("wrong-governance"))
    )
    with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1):
        status.verify_signed_owner_status_event_v1(
            chain.active.result.signed_event_payload,
            governance_public_key=wrong_public,
            logical_time_token_payload=chain.active.logical_token_payload,
            logical_time_public_key=chain.context.logical_time_public_key,
            expected_authority_profile_id=chain.context.authority_profile_id,
            expected_signer_registry_id=chain.context.signer_registry_id,
            expected_governance_key_id=chain.context.governance_key_id,
            expected_governance_key_epoch=1,
            expected_attempt_context_id=chain.context.attempt_context_id,
        )


def test_governance_signature_splice_with_recomputed_event_id_still_fails(
    chain: CompleteChain,
) -> None:
    mapping = json.loads(chain.active.result.signed_event_payload)
    signature = mapping["signature_hex"]
    mapping["signature_hex"] = ("0" if signature[0] != "0" else "1") + signature[1:]
    id_subject = dict(mapping)
    id_subject.pop("event_id")
    mapping["event_id"] = _content_id(
        status.EVENT_ID_PREFIX_V1, EVENT_ID_DOMAIN, id_subject
    )
    payload = canonicalize_rfc8785_v1(mapping)
    with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1) as failure:
        status.verify_signed_owner_status_event_v1(
            payload,
            governance_public_key=chain.context.governance_public_key,
            logical_time_token_payload=chain.active.logical_token_payload,
            logical_time_public_key=chain.context.logical_time_public_key,
            expected_authority_profile_id=chain.context.authority_profile_id,
            expected_signer_registry_id=chain.context.signer_registry_id,
            expected_governance_key_id=chain.context.governance_key_id,
            expected_governance_key_epoch=1,
            expected_attempt_context_id=chain.context.attempt_context_id,
        )
    assert failure.value.code == status.FAIL_SIGNATURE_V1


def test_logical_time_signature_splice_with_recomputed_token_id_fails(
    chain: CompleteChain,
) -> None:
    token = json.loads(chain.active.logical_token_payload)
    signature = token["signature_hex"]
    token["signature_hex"] = ("0" if signature[0] != "0" else "1") + signature[1:]
    id_subject = dict(token)
    id_subject.pop("token_id")
    token["token_id"] = _content_id(
        status.LOGICAL_TIME_TOKEN_ID_PREFIX_V1, TOKEN_ID_DOMAIN, id_subject
    )
    with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1) as failure:
        _logical_reference(
            chain.context,
            chain.active.decision_payload,
            canonicalize_rfc8785_v1(token),
        )
    assert failure.value.code == status.FAIL_SIGNATURE_V1


@pytest.mark.parametrize(
    ("branch", "field"),
    (
        ("cas_store_dependency", "authority_profile_id"),
        ("cas_store_dependency", "signer_registry_id"),
        ("cas_store_dependency", "attempt_context_id"),
        ("logical_time_dependency", "authority_profile_id"),
        ("logical_time_dependency", "signer_registry_id"),
        ("logical_time_dependency", "attempt_context_id"),
    ),
)
def test_cross_graph_profile_registry_attempt_exchange_rejected(
    chain: CompleteChain, branch: str, field: str
) -> None:
    mapping = json.loads(chain.active.result.signed_event_payload)
    if field == "authority_profile_id":
        replacement = _identity(status.AUTHORITY_PROFILE_PREFIX_V2, "foreign")
    elif field == "signer_registry_id":
        replacement = _identity(status.SIGNER_REGISTRY_PREFIX_V2, "foreign")
    else:
        replacement = "foreign-attempt"
    mapping[branch][field] = replacement
    with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1):
        status.load_signed_owner_status_event_v1(canonicalize_rfc8785_v1(mapping))


def test_logical_time_key_exchange_rejected_before_publication(tmp_path: Path) -> None:
    context = _context(tmp_path, "key-exchange")
    foreign_private = Ed25519PrivateKey.from_private_bytes(_seed("foreign-logical"))
    with _open_ledger(context, initialize_empty=True) as ledger:
        previous_head = ledger.current_head_bytes()
        decision = status.build_owner_status_decision_v1(
            previous_head,
            owner_authorization_payload=context.owner_payload,
            reason_code="INITIAL_OWNER_STATUS",
            decision_nonce="e0" * 32,
        )
        token, _token_id = _logical_token(
            context,
            decision,
            sequence=0,
            prior_token_id=status.LOGICAL_TIME_GENESIS_TOKEN_ID_V1,
            request_nonce="f0" * 32,
            private_key=foreign_private,
        )
        reference = _logical_reference(
            context,
            decision,
            token,
            public_key=_public_bytes(foreign_private),
        )
        cas_reference = _cas_reference(context, decision, checkpoint_index=0)
        with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1):
            ledger.append_service_side_v1(
                decision_payload=decision,
                cas_store_dependency_reference_payload=cas_reference,
                logical_time_dependency_reference_payload=reference,
                logical_time_token_payload=token,
                expected_previous_head_payload=previous_head,
                governance_private_key=context.governance_private_key,
            )
        assert not tuple((context.root / "logical-time-tokens").iterdir())


class _StringSubclass(str):
    pass


class _EqualToEverything:
    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        (
            "schema_version",
            _StringSubclass(status.CAS_DEPENDENCY_REFERENCE_SCHEMA_V1),
        ),
        ("verification_status", _EqualToEverything()),
        ("signer_key_epoch", True),
    ),
)
def test_direct_cas_dto_rejects_string_subclass_eqall_and_bool_integer(
    chain: CompleteChain, field: str, replacement: object
) -> None:
    mapping = json.loads(chain.active.cas_reference_payload)
    mapping[field] = replacement
    with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1):
        status.CasStoreDependencyReferenceV1.from_mapping(mapping)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        (
            "schema_version",
            _StringSubclass(status.LOGICAL_TIME_DEPENDENCY_REFERENCE_SCHEMA_V1),
        ),
        ("verification_status", _EqualToEverything()),
        ("signer_key_epoch", True),
    ),
)
def test_direct_logical_time_dto_rejects_string_subclass_eqall_and_bool_integer(
    chain: CompleteChain, field: str, replacement: object
) -> None:
    mapping = json.loads(chain.active.logical_reference_payload)
    mapping[field] = replacement
    with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1):
        status.LogicalTimeDependencyReferenceV1.from_mapping(mapping)


def test_canonical_bytes_exact_field_set_fraction_and_bytes_subclass_rejected(
    chain: CompleteChain,
) -> None:
    payload = chain.active.decision_payload
    with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1):
        status.load_owner_status_decision_v1(type("BytesSubclass", (bytes,), {})(payload))
    with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1):
        status.load_owner_status_decision_v1(b" " + payload)
    unknown = json.loads(payload)
    unknown["unknown"] = "field"
    with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1):
        status.load_owner_status_decision_v1(canonicalize_rfc8785_v1(unknown))
    boolean_sequence = json.loads(payload)
    boolean_sequence["sequence"] = False
    with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1):
        status.load_owner_status_decision_v1(
            canonicalize_rfc8785_v1(boolean_sequence)
        )
    fractional = payload.replace(b'"sequence":0', b'"sequence":0.5')
    with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1):
        status.load_owner_status_decision_v1(fractional)


def test_clean_import_has_no_authority_store_or_logical_time_actor_imports() -> None:
    module = "hegel_machine.phase2b_formal_wire_covert_step5_owner_status_v1"
    forbidden = (
        "hegel_machine.phase2b_formal_wire_covert_step5_authority_v1",
        "hegel_machine.phase2b_formal_wire_covert_step5_store_v1",
        "hegel_machine.phase2b_formal_wire_covert_step5_logical_time_v1",
    )
    code = (
        "import importlib,sys;"
        f"sys.path.insert(0,{str(PROJECT_ROOT / 'src')!r});"
        f"importlib.import_module({module!r});"
        f"bad=[name for name in {forbidden!r} if name in sys.modules];"
        "assert not bad,bad"
    )
    subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_pure_replay_uses_bytes_and_public_keys_not_private_key_objects(
    chain: CompleteChain, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(status, "_Ed25519PrivateKey", None)
    projection = _replay(chain)
    assert projection.latest_effective_status == status.STATUS_REVOKED_V1
    assert projection.authority_evidence is False


@pytest.mark.parametrize("attack", ("duplicate", "gap", "reverse", "list"))
def test_replay_duplicate_gap_reverse_and_container_type_fail_closed(
    chain: CompleteChain, attack: str
) -> None:
    checkpoints: object = (
        chain.active.result.checkpoint_payload,
        chain.revoked.result.checkpoint_payload,
    )
    heads: object = (
        chain.active.result.head_payload,
        chain.revoked.result.head_payload,
    )
    tokens: object = (
        chain.active.logical_token_payload,
        chain.revoked.logical_token_payload,
    )
    if attack == "duplicate":
        checkpoints = (checkpoints[0], checkpoints[0])  # type: ignore[index]
        heads = (heads[0], heads[0])  # type: ignore[index]
        tokens = (tokens[0], tokens[0])  # type: ignore[index]
    elif attack == "gap":
        checkpoints = (checkpoints[1],)  # type: ignore[index]
        heads = (heads[1],)  # type: ignore[index]
        tokens = (tokens[1],)  # type: ignore[index]
    elif attack == "reverse":
        checkpoints = tuple(reversed(checkpoints))  # type: ignore[arg-type]
        heads = tuple(reversed(heads))  # type: ignore[arg-type]
        tokens = tuple(reversed(tokens))  # type: ignore[arg-type]
    else:
        checkpoints = list(checkpoints)  # type: ignore[arg-type]
    with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1):
        status.verify_owner_status_ledger_replay_v1(
            chain.genesis_head_payload,
            checkpoints,  # type: ignore[arg-type]
            heads,  # type: ignore[arg-type]
            tokens,  # type: ignore[arg-type]
            owner_authorization_payload=chain.context.owner_payload,
            governance_public_key=chain.context.governance_public_key,
            logical_time_public_key=chain.context.logical_time_public_key,
            expected_authority_profile_id=chain.context.authority_profile_id,
            expected_signer_registry_id=chain.context.signer_registry_id,
            expected_governance_key_id=chain.context.governance_key_id,
            expected_governance_key_epoch=1,
            expected_attempt_context_id=chain.context.attempt_context_id,
        )


def test_replay_expected_registry_and_attempt_exchange_fail(chain: CompleteChain) -> None:
    for registry, attempt in (
        (_identity(status.SIGNER_REGISTRY_PREFIX_V2, "foreign"), chain.context.attempt_context_id),
        (chain.context.signer_registry_id, "foreign-attempt"),
    ):
        with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1):
            status.verify_owner_status_ledger_replay_v1(
                chain.genesis_head_payload,
                (
                    chain.active.result.checkpoint_payload,
                    chain.revoked.result.checkpoint_payload,
                ),
                (chain.active.result.head_payload, chain.revoked.result.head_payload),
                (
                    chain.active.logical_token_payload,
                    chain.revoked.logical_token_payload,
                ),
                owner_authorization_payload=chain.context.owner_payload,
                governance_public_key=chain.context.governance_public_key,
                logical_time_public_key=chain.context.logical_time_public_key,
                expected_authority_profile_id=chain.context.authority_profile_id,
                expected_signer_registry_id=registry,
                expected_governance_key_id=chain.context.governance_key_id,
                expected_governance_key_epoch=1,
                expected_attempt_context_id=attempt,
            )


def test_symlink_root_is_rejected_by_no_follow(tmp_path: Path) -> None:
    context = _context(tmp_path, "symlink-root")
    target = context.root
    link = tmp_path / "ledger-link"
    link.symlink_to(target, target_is_directory=True)
    linked_context = Context(**{**context.__dict__, "root": link})
    with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1):
        _open_ledger(linked_context, initialize_empty=True)


def test_exclusive_publication_collision_preserves_existing_record(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, "exclusive-collision")
    with _open_ledger(context, initialize_empty=True) as ledger:
        name = "00000000000000000000.json"
        existing = (context.root / "heads" / name).read_bytes()
        with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1) as failure:
            ledger._publish_exclusive(
                ledger._heads_fd,
                name,
                existing,
                status.MAXIMUM_HEAD_BYTES,
                "head",
            )
        assert failure.value.code == status.FAIL_LEDGER_V1
        assert (context.root / "heads" / name).read_bytes() == existing
        assert not tuple((context.root / "heads").glob(".pending-*"))


def test_external_hardlink_to_record_is_detected(chain: CompleteChain, tmp_path: Path) -> None:
    head = chain.context.root / "heads" / "00000000000000000000.json"
    os.link(head, tmp_path / "outside-hardlink.json")
    with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1):
        _open_ledger(chain.context, initialize_empty=False)


def test_held_directory_rename_and_rebind_is_detected(tmp_path: Path) -> None:
    context = _context(tmp_path, "directory-rebind")
    with _open_ledger(context, initialize_empty=True) as ledger:
        displaced = tmp_path / "displaced-heads"
        os.rename(context.root / "heads", displaced)
        (context.root / "heads").mkdir()
        with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1):
            ledger.current_head_bytes()


def test_same_size_record_mutation_fails_readback_and_replay(
    chain: CompleteChain,
) -> None:
    head = chain.context.root / "heads" / "00000000000000000000.json"
    with head.open("r+b") as handle:
        handle.write(b"[")
        handle.flush()
        os.fsync(handle.fileno())
    with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1):
        _open_ledger(chain.context, initialize_empty=False)


def test_crash_after_final_token_creation_leaves_visible_poison_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path, "crash-gap")
    ledger = _open_ledger(context, initialize_empty=True)
    previous_head = ledger.current_head_bytes()
    decision = status.build_owner_status_decision_v1(
        previous_head,
        owner_authorization_payload=context.owner_payload,
        reason_code="INITIAL_OWNER_STATUS",
        decision_nonce="01" * 32,
    )
    token, _token_id = _logical_token(
        context,
        decision,
        sequence=0,
        prior_token_id=status.LOGICAL_TIME_GENESIS_TOKEN_ID_V1,
        request_nonce="02" * 32,
    )
    logical_reference = _logical_reference(context, decision, token)
    cas_reference = _cas_reference(context, decision, checkpoint_index=0)
    original_fsync = status.os.fsync
    calls = 0

    def fail_second_fsync(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected crash after final token creation")
        original_fsync(fd)

    monkeypatch.setattr(status.os, "fsync", fail_second_fsync)
    with pytest.raises(OSError):
        ledger.append_service_side_v1(
            decision_payload=decision,
            cas_store_dependency_reference_payload=cas_reference,
            logical_time_dependency_reference_payload=logical_reference,
            logical_time_token_payload=token,
            expected_previous_head_payload=previous_head,
            governance_private_key=context.governance_private_key,
        )
    ledger.close()
    monkeypatch.setattr(status.os, "fsync", original_fsync)
    assert len(tuple((context.root / "logical-time-tokens").iterdir())) == 1
    with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1):
        _open_ledger(context, initialize_empty=False)


def test_whole_ledger_rollback_is_locally_replayable_but_never_authority(
    chain: CompleteChain,
) -> None:
    (chain.context.root / "logical-time-tokens" / "00000000000000000001.json").unlink()
    (chain.context.root / "checkpoints" / "00000000000000000001.json").unlink()
    (chain.context.root / "heads" / "00000000000000000002.json").unlink()
    with _open_ledger(chain.context, initialize_empty=False) as rolled_back:
        projection = status.load_owner_status_projection_v1(
            rolled_back.latest_projection_bytes()
        )
    assert projection.latest_effective_status == status.STATUS_ACTIVE_V1
    assert projection.latest_authority_status_verified is False
    assert projection.r01_satisfied is False
    assert projection.authority_evidence is False
    assert "EXTERNAL_ANTI_ROLLBACK_ANCHOR_OF_COMPLETE_EVENT_SET_ROOT_VERIFIED" in (
        status.REQUIRED_EXTERNAL_AUTHORITY_CHECKS_V1
    )


def test_existing_stale_head_compare_and_exclusive_sequence_prevent_append_replay(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, "stale-head")
    with _open_ledger(context, initialize_empty=True) as ledger:
        genesis = ledger.current_head_bytes()
        active = _append(
            ledger,
            context,
            prior_token_id=status.LOGICAL_TIME_GENESIS_TOKEN_ID_V1,
            decision_nonce="03" * 32,
            request_nonce="04" * 32,
            token_sequence=0,
            cas_checkpoint_index=0,
        )
        with pytest.raises(status.Phase2BStep5OwnerStatusErrorV1):
            ledger.append_service_side_v1(
                decision_payload=active.decision_payload,
                cas_store_dependency_reference_payload=active.cas_reference_payload,
                logical_time_dependency_reference_payload=active.logical_reference_payload,
                logical_time_token_payload=active.logical_token_payload,
                expected_previous_head_payload=genesis,
                governance_private_key=context.governance_private_key,
            )


def test_source_declares_v2_purpose_static_freeze_but_no_authority_projection() -> None:
    assert "NOT_ROLLBACK_OR_CLONE_RESISTANT" in status.AUTHORITY_CLAIM_BOUNDARY_V1
    assert (
        "AUTHORITY_V2_GOVERNANCE_PURPOSE_IS_STATICALLY_FROZEN_BUT_RAW_REGISTRY_GRAPH_IS_NOT_VERIFIED_HERE"
        in status.KNOWN_INTEGRATION_BLOCKERS_V1
    )
    assert (
        "AUTHORITY_DEPENDENCY_REPLACED_BY_PROFILE_V2_AND_SIGNER_REGISTRY_V2"
        in status.REQUIRED_AUTHORITY_SCHEMA_CASCADE_V1
    )
    assert (
        "NO_REAL_R01_RECEIPT_OR_READINESS_OVERLAY_SCHEMA_IS_ISSUED_HERE"
        in status.KNOWN_INTEGRATION_BLOCKERS_V1
    )
