from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from hegel_machine import phase2b_formal_wire_covert_step5_authority_v1 as authority


ROOT = Path(__file__).resolve().parents[1]
OWNER_AUTHORIZATION = (
    ROOT / "config/phase2b_c1_owner_actual_unsealed_960_authorization_v1.json"
)
READINESS_DAG = ROOT / "config/phase2b_actual_unsealed_960_readiness_dag_v1.json"


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical(value: object) -> bytes:
    return authority.canonical_step5_json_bytes_v1(value)


def _public_hex(private_key: Ed25519PrivateKey) -> str:
    return private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()


def _qv2_binding() -> dict[str, object]:
    return {
        "schema_version": "hegel-machine-phase2b-formal-wire-covert-qualification/9",
        "profile_id": authority.QV2_PROFILE_ID_V1,
        "profile_payload_sha256": authority.QV2_PROFILE_PAYLOAD_SHA256_V1,
        "source_sha256": authority.QV2_SOURCE_SHA256_V1,
        "dependency_lock_sha256": authority.QV2_DEPENDENCY_LOCK_SHA256_V1,
    }


def _owner_binding() -> dict[str, object]:
    return {
        "authorization_schema": (
            "hegel-machine-phase2b-c1-owner-actual-unsealed-960-authorization/1"
        ),
        "authorization_id": authority.OWNER_AUTHORIZATION_ID_V1,
        "authorization_config_sha256": authority.OWNER_AUTHORIZATION_CONFIG_SHA256_V1,
        "authorization_source_sha256": authority.OWNER_AUTHORIZATION_SOURCE_SHA256_V1,
        "protocol_id": authority.PROTOCOL_ID_V1,
        "exact_freeze_version": authority.EXACT_FREEZE_VERSION_V1,
        "exact_freeze_id": authority.EXACT_FREEZE_ID_V1,
        "authority_class": authority.AUTHORITY_CLASS_V1,
        "authority_requirement_id": authority.AUTHORITY_REQUIREMENT_ID_V1,
        "actor_profile_content_id": authority.ACTOR_PROFILE_CONTENT_ID_V1,
        "actor_profile_canonical_sha256": authority.ACTOR_PROFILE_CANONICAL_SHA256_V1,
        "logical_stage_wire_schema_id": authority.LOGICAL_STAGE_WIRE_SCHEMA_ID_V1,
        "logical_stage_wire_policy_id": authority.LOGICAL_STAGE_WIRE_POLICY_ID_V1,
        "stage_event_contract_manifest_id": authority.STAGE_EVENT_CONTRACT_MANIFEST_ID_V1,
    }


def _threat_model() -> dict[str, object]:
    return {
        "same_admin_controller": True,
        "owner_accepted_technical_actor_model": True,
        "producer_may_be_malicious": True,
        "producer_outside_authority_roles": True,
        "producer_local_filesystem_claims_authoritative": False,
        "producer_local_filesystem_is_non_authoritative": True,
        "purpose_separated_keys_required": True,
        "distinct_control_domains_required": True,
        "organizational_independence": False,
        "independent_human_actors": False,
        "external_authority": False,
        "remote_attestation": False,
        "hardware_key_nonexportability": False,
        "malicious_admin_resistance": False,
        "malicious_host_root_resistance": False,
        "malicious_container_runtime_resistance": False,
    }


def _permanent_nonclaims() -> dict[str, object]:
    return {
        "actual_corpus_input": False,
        "actual_campaign_id_allocated": False,
        "actual_attempt_id_allocated": False,
        "actual_execution_authorized": False,
        "r01_satisfied": False,
        "r02_satisfied": False,
        "r03_satisfied": False,
        "r04_satisfied": False,
        "r05_satisfied": False,
        "r06_or_later_unlocked": False,
        "cas_store_verified": False,
        "logical_time_verified": False,
        "readiness_evidence_overlay_issued": False,
        "technical_control_domains_live_verified": False,
        "organizational_independence": False,
        "independent_human_actors": False,
        "external_authority": False,
        "remote_attestation": False,
        "hardware_key_nonexportability": False,
        "malicious_admin_resistance": False,
    }


def _role_key(
    role: str, private_key: Ed25519PrivateKey
) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": "hegel-machine-phase2b-step5-role-key/1",
        "role": role,
        "control_domain": authority.ROLE_CONTROL_DOMAINS_V1[role],
        "authorized_signature_purposes": list(
            authority.ROLE_SIGNATURE_PURPOSES_V1[role]
        ),
        "algorithm": "Ed25519",
        "key_epoch": 1,
        "public_key_hex": _public_hex(private_key),
    }
    row["key_id"] = authority.phase2b_step5_role_key_id_v1(row)
    return row


def _signature(
    *,
    artifact_kind: str,
    body: dict[str, object],
    role: str,
    purpose: str,
    row: dict[str, object],
    private_key: Ed25519PrivateKey,
) -> dict[str, object]:
    preimage = authority.phase2b_step5_signature_preimage_v1(
        artifact_kind=artifact_kind,
        body=body,
        role=role,
        purpose=purpose,
        signer_key_id=row["key_id"],
        key_epoch=1,
    )
    return {
        "schema_version": authority.DETACHED_SIGNATURE_SCHEMA_V1,
        "algorithm": "Ed25519",
        "role": role,
        "purpose": purpose,
        "key_epoch": 1,
        "signer_key_id": row["key_id"],
        "signature_hex": private_key.sign(preimage).hex(),
    }


def _profile_wrapper(body: dict[str, object]) -> dict[str, object]:
    return {
        "body": body,
        "id": authority.phase2b_step5_content_id_v1("authority_profile", body),
    }


def _registry_wrapper(
    body: dict[str, object], governance_key: Ed25519PrivateKey
) -> dict[str, object]:
    rows = body["role_keys_in_exact_order"]
    assert type(rows) is list and type(rows[0]) is dict
    return {
        "body": body,
        "id": authority.phase2b_step5_content_id_v1("signer_registry", body),
        "signatures": [
            _signature(
                artifact_kind="signer_registry",
                body=body,
                role="GOVERNANCE_ANCHOR",
                purpose="SIGN_SIGNER_REGISTRY",
                row=rows[0],
                private_key=governance_key,
            )
        ],
    }


def _stage_b_lock(
    profile: dict[str, object], registry: dict[str, object]
) -> tuple[dict[str, object], bytes]:
    profile_payload = _canonical(profile)
    registry_payload = _canonical(registry)
    body = {
        "schema_version": authority.STAGE_B_TRUST_LOCK_SCHEMA_V1,
        "status": "FROZEN_OUT_OF_BAND_STAGE_B_TRUST_LOCK",
        "canonicalization": "RFC8785_EXACT_BYTES_NO_TRAILING_LF",
        "authority_profile_id": profile["id"],
        "authority_profile_payload_sha256": hashlib.sha256(
            profile_payload
        ).hexdigest(),
        "signer_registry_id": registry["id"],
        "signer_registry_payload_sha256": hashlib.sha256(
            registry_payload
        ).hexdigest(),
        "qv2_binding": _qv2_binding(),
    }
    lock = {
        "body": body,
        "id": authority.phase2b_step5_content_id_v1("stage_b_trust_lock", body),
    }
    return lock, _canonical(lock)


def _load_lock(
    lock: dict[str, object], payload: bytes
) -> authority.Phase2BStep5VerifiedArtifactV1:
    return authority.load_step5_stage_b_trust_lock_v1(
        payload,
        out_of_band_expected_lock_id=lock["id"],
        out_of_band_expected_payload_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _dual_candidate_wrapper(
    *,
    kind: str,
    body: dict[str, object],
    purposes: tuple[str, str],
    rows: tuple[dict[str, object], ...],
    private_keys: tuple[Ed25519PrivateKey, ...],
) -> dict[str, object]:
    return {
        "body": body,
        "id": authority.phase2b_step5_content_id_v1(kind, body),
        "signatures": [
            _signature(
                artifact_kind=kind,
                body=body,
                role="SOURCE_RUNTIME_CUSTODIAN",
                purpose=purposes[0],
                row=rows[1],
                private_key=private_keys[1],
            ),
            _signature(
                artifact_kind=kind,
                body=body,
                role="PURE_REPLAYER",
                purpose=purposes[1],
                row=rows[7],
                private_key=private_keys[7],
            ),
        ],
    }


def _r01_claim_boundary() -> dict[str, object]:
    return {
        "owner_authorization_exact_bytes_verified": True,
        "owner_signature_authenticated_by_original_authorization": False,
        "governance_status_ledger_verified": False,
        "latest_revocation_status_verified": False,
        "cas_store_checkpoint_verified": False,
        "logical_time_verified": False,
        "r01_satisfied": False,
        "r02_or_later_satisfied": False,
        "execution_authorized": False,
        "actual_input_consumed": False,
        "organizational_independence": False,
        "independent_human_actors": False,
    }


def _r02_claim_boundary() -> dict[str, object]:
    return {
        "declared_binding_equality_verified": True,
        "independent_branch_replay_verified": False,
        "branch_stage_event_verified": False,
        "cas_store_checkpoint_verified": False,
        "logical_time_verified": False,
        "r01_satisfied": False,
        "r02_satisfied": False,
        "r03_satisfied": False,
        "r04_satisfied": False,
        "r05_satisfied": False,
        "actual_execution_authorized": False,
        "actual_corpus_qualified": False,
        "campaign_or_attempt_id_allocated": False,
        "organizational_independence": False,
        "independent_human_actors": False,
    }


def _replay_bindings() -> dict[str, object]:
    owner = _owner_binding()
    return {
        "owner_authorization_id": owner["authorization_id"],
        "readiness_dag_id": authority.READINESS_DAG_ID_V1,
        "protocol_id": owner["protocol_id"],
        "exact_freeze_version": owner["exact_freeze_version"],
        "exact_freeze_id": owner["exact_freeze_id"],
        "authority_class": owner["authority_class"],
        "authority_requirement_id": owner["authority_requirement_id"],
        "actor_profile_content_id": owner["actor_profile_content_id"],
        "actor_profile_canonical_sha256": owner[
            "actor_profile_canonical_sha256"
        ],
        "logical_stage_wire_schema_id": owner["logical_stage_wire_schema_id"],
        "logical_stage_wire_policy_id": owner["logical_stage_wire_policy_id"],
        "stage_event_contract_manifest_id": owner[
            "stage_event_contract_manifest_id"
        ],
    }


@dataclass(frozen=True)
class Graph:
    private_keys: tuple[Ed25519PrivateKey, ...]
    rows: tuple[dict[str, object], ...]
    profile: dict[str, object]
    profile_payload: bytes
    registry: dict[str, object]
    registry_payload: bytes
    lock: dict[str, object]
    lock_payload: bytes
    lock_token: authority.Phase2BStep5VerifiedArtifactV1
    profile_token: authority.Phase2BStep5VerifiedArtifactV1
    registry_token: authority.Phase2BStep5VerifiedArtifactV1
    owner_payload: bytes
    r01: dict[str, object]
    r01_payload: bytes
    r01_token: authority.Phase2BStep5VerifiedArtifactV1
    r02: dict[str, object]
    r02_payload: bytes
    r02_token: authority.Phase2BStep5VerifiedArtifactV1
    snapshot: authority.Phase2BStep5VerifiedArtifactV1


def _build_graph() -> Graph:
    private_keys = tuple(
        Ed25519PrivateKey.from_private_bytes(bytes([index]) * 32)
        for index in range(1, 9)
    )
    rows = tuple(
        _role_key(role, private_keys[index])
        for index, role in enumerate(authority.ROLE_ORDER_V1)
    )
    profile_body: dict[str, object] = {
        "schema_version": authority.AUTHORITY_PROFILE_SCHEMA_V1,
        "status": "FROZEN_OWNER_ACCEPTED_TECHNICAL_AUTHORITY_PROFILE",
        "claim_level": (
            "SAME_ADMIN_TECHNICAL_AUTHORITY_NOT_EXTERNAL_OR_ORGANIZATIONAL_INDEPENDENCE"
        ),
        "canonicalization": "RFC8785_EXACT_BYTES_NO_TRAILING_LF",
        "profile_epoch": 1,
        "owner_authorization_binding": _owner_binding(),
        "readiness_dag_binding": {
            "schema_version": (
                "hegel-machine-phase2b-actual-unsealed-960-readiness-dag/1"
            ),
            "readiness_dag_id": authority.READINESS_DAG_ID_V1,
        },
        "qv2_binding": _qv2_binding(),
        "source_binding": {
            "source_commit_sha": "a" * 40,
            "source_archive_sha256": _h("source-archive"),
            "dependency_lock_sha256": _h("step5-dependency-lock"),
        },
        "runtime_image_bindings": {
            "runner": "sha256:" + _h("runner-image"),
            "pure_replayer": "sha256:" + _h("pure-replayer-image"),
            "cas_store": "sha256:" + _h("cas-store-image"),
            "logical_time": "sha256:" + _h("logical-time-image"),
        },
        "role_policy": {
            "roles_in_exact_order": list(authority.ROLE_ORDER_V1),
            "role_count": 8,
            "distinct_public_keys": True,
            "distinct_key_ids": True,
            "distinct_control_domains": True,
            "key_epoch": 1,
        },
        "governance_root": copy.deepcopy(rows[0]),
        "threat_model": _threat_model(),
        "permanent_nonclaims": _permanent_nonclaims(),
    }
    profile = _profile_wrapper(profile_body)
    profile_payload = _canonical(profile)
    registry_body: dict[str, object] = {
        "schema_version": authority.SIGNER_REGISTRY_SCHEMA_V1,
        "authority_profile_id": profile["id"],
        "registry_epoch": 1,
        "status": "ACTIVE",
        "revocation_epoch": 0,
        "role_keys_in_exact_order": [copy.deepcopy(row) for row in rows],
    }
    registry = _registry_wrapper(registry_body, private_keys[0])
    registry_payload = _canonical(registry)
    lock, lock_payload = _stage_b_lock(profile, registry)
    lock_token = _load_lock(lock, lock_payload)
    profile_token = authority.load_step5_authority_profile_v1(
        profile_payload, lock_token
    )
    registry_token = authority.load_step5_signer_registry_v1(
        registry_payload, lock_token, profile_token
    )
    owner_payload = OWNER_AUTHORIZATION.read_bytes()
    r01_body: dict[str, object] = {
        "schema_version": authority.R01_AUTHORIZATION_STATUS_CANDIDATE_SCHEMA_V1,
        "authority_profile_id": profile["id"],
        "signer_registry_id": registry["id"],
        "owner_authorization_binding": _owner_binding(),
        "owner_authorization_payload_sha256": hashlib.sha256(
            owner_payload
        ).hexdigest(),
        "status_checkpoint_candidate": {
            "schema_version": (
                "hegel-machine-phase2b-step5-owner-authorization-status-checkpoint-candidate/1"
            ),
            "authorization_id": authority.OWNER_AUTHORIZATION_ID_V1,
            "sequence": 0,
            "event": "INITIAL_ACTIVE_STATUS_CANDIDATE_NOT_LATEST_STATUS_EVIDENCE",
            "effective_status": "ACTIVE_CANDIDATE_NOT_VERIFIED_CURRENT",
            "previous_status_head_id": None,
            "revocation_receipt_id": None,
            "revocation_count": 0,
        },
        "claim_boundary": _r01_claim_boundary(),
    }
    r01 = _dual_candidate_wrapper(
        kind="r01_candidate",
        body=r01_body,
        purposes=(
            "SIGN_R01_AUTHORIZATION_STATUS_CANDIDATE",
            "COUNTERSIGN_R01_AUTHORIZATION_STATUS_CANDIDATE",
        ),
        rows=rows,
        private_keys=private_keys,
    )
    r01_payload = _canonical(r01)
    r01_token = authority.load_step5_r01_authorization_status_candidate_v1(
        r01_payload,
        owner_payload,
        lock_token,
        profile_token,
        registry_token,
    )
    r02_body: dict[str, object] = {
        "schema_version": authority.R02_SYNTHETIC_BRANCH_CANDIDATE_SCHEMA_V1,
        "authority_profile_id": profile["id"],
        "signer_registry_id": registry["id"],
        "r01_candidate_id": r01["id"],
        "declared_expected_bindings": _replay_bindings(),
        "declared_observed_selected_branch_bindings": _replay_bindings(),
        "qv2_binding": _qv2_binding(),
        "source_binding": copy.deepcopy(profile_body["source_binding"]),
        "runtime_image_bindings": copy.deepcopy(
            profile_body["runtime_image_bindings"]
        ),
        "branch_selection": {
            "selected_branch": "SYNTHETIC_DEVELOPMENT_QUALIFICATION_ONLY",
            "input_class": "SYNTHETIC_DEV_ONLY",
            "synthetic_input": True,
            "actual_corpus_input": False,
            "actual_input_count": 0,
            "actual_campaign_id": None,
            "actual_attempt_id": None,
            "actual_material_consumed": False,
        },
        "claim_boundary": _r02_claim_boundary(),
    }
    r02 = _dual_candidate_wrapper(
        kind="r02_candidate",
        body=r02_body,
        purposes=(
            "SIGN_R02_SYNTHETIC_BRANCH_CANDIDATE",
            "COUNTERSIGN_R02_SYNTHETIC_BRANCH_CANDIDATE",
        ),
        rows=rows,
        private_keys=private_keys,
    )
    r02_payload = _canonical(r02)
    r02_token = authority.load_step5_r02_synthetic_branch_candidate_v1(
        r02_payload,
        lock_token,
        profile_token,
        registry_token,
        r01_token,
    )
    snapshot = authority.derive_step5_r01_r02_candidate_snapshot_v1(
        lock_token,
        profile_token,
        registry_token,
        r01_token,
        r02_token,
        READINESS_DAG.read_bytes(),
    )
    return Graph(
        private_keys=private_keys,
        rows=rows,
        profile=profile,
        profile_payload=profile_payload,
        registry=registry,
        registry_payload=registry_payload,
        lock=lock,
        lock_payload=lock_payload,
        lock_token=lock_token,
        profile_token=profile_token,
        registry_token=registry_token,
        owner_payload=owner_payload,
        r01=r01,
        r01_payload=r01_payload,
        r01_token=r01_token,
        r02=r02,
        r02_payload=r02_payload,
        r02_token=r02_token,
        snapshot=snapshot,
    )


@pytest.fixture(scope="module")
def graph() -> Graph:
    return _build_graph()


def _assert_rejected(
    operation: Callable[[], object], expected_codes: set[str] | None = None
) -> authority.Phase2BStep5AuthorityErrorV1:
    with pytest.raises(authority.Phase2BStep5AuthorityErrorV1) as captured:
        operation()
    if expected_codes is not None:
        assert captured.value.code in expected_codes
    return captured.value


def _load_profile_registry_for(
    profile: dict[str, object], registry: dict[str, object]
) -> tuple[
    authority.Phase2BStep5VerifiedArtifactV1,
    authority.Phase2BStep5VerifiedArtifactV1,
    authority.Phase2BStep5VerifiedArtifactV1,
]:
    lock, lock_payload = _stage_b_lock(profile, registry)
    lock_token = _load_lock(lock, lock_payload)
    profile_token = authority.load_step5_authority_profile_v1(
        _canonical(profile), lock_token
    )
    registry_token = authority.load_step5_signer_registry_v1(
        _canonical(registry), lock_token, profile_token
    )
    return lock_token, profile_token, registry_token


def _resigned_r01(graph: Graph, body: dict[str, object]) -> dict[str, object]:
    return _dual_candidate_wrapper(
        kind="r01_candidate",
        body=body,
        purposes=(
            "SIGN_R01_AUTHORIZATION_STATUS_CANDIDATE",
            "COUNTERSIGN_R01_AUTHORIZATION_STATUS_CANDIDATE",
        ),
        rows=graph.rows,
        private_keys=graph.private_keys,
    )


def _resigned_r02(graph: Graph, body: dict[str, object]) -> dict[str, object]:
    return _dual_candidate_wrapper(
        kind="r02_candidate",
        body=body,
        purposes=(
            "SIGN_R02_SYNTHETIC_BRANCH_CANDIDATE",
            "COUNTERSIGN_R02_SYNTHETIC_BRANCH_CANDIDATE",
        ),
        rows=graph.rows,
        private_keys=graph.private_keys,
    )


def test_positive_stage_b_to_all_false_candidate_snapshot(graph: Graph) -> None:
    assert graph.lock_token.kind == "stage_b_trust_lock"
    assert graph.profile_token.kind == "authority_profile"
    assert graph.registry_token.kind == "signer_registry"
    assert graph.registry_token.signature_backends == ("CRYPTOGRAPHY_ED25519",)
    assert graph.r01_token.kind == "r01_candidate"
    assert graph.r02_token.kind == "r02_candidate"
    assert graph.r01_token.signature_backends == (
        "CRYPTOGRAPHY_ED25519",
        "CRYPTOGRAPHY_ED25519",
    )
    assert graph.r02_token.signature_backends == graph.r01_token.signature_backends
    assert graph.r01_token.body["claim_boundary"]["r01_satisfied"] is False
    assert graph.r02_token.body["claim_boundary"]["r02_satisfied"] is False
    assert graph.snapshot.kind == "readiness_candidate"
    assert graph.snapshot.body["satisfied_node_count"] == 0
    assert graph.snapshot.body["highest_contiguous_satisfied_node"] is None
    assert graph.snapshot.body["all_nodes_satisfied"] is False
    rows = graph.snapshot.body["node_states_in_canonical_topological_order"]
    assert len(rows) == 16
    assert all(row["satisfied"] is False for row in rows)
    assert rows[0]["candidate_artifact_id"] == graph.r01_token.artifact_id
    assert rows[1]["candidate_artifact_id"] == graph.r02_token.artifact_id
    assert all(row["candidate_artifact_id"] is None for row in rows[2:])


@pytest.mark.parametrize("wrong_pin", ("id", "sha"))
def test_out_of_band_stage_b_pins_reject_substitution(
    graph: Graph, wrong_pin: str
) -> None:
    expected_id = graph.lock["id"]
    expected_sha = hashlib.sha256(graph.lock_payload).hexdigest()
    if wrong_pin == "id":
        expected_id = authority.STAGE_B_TRUST_LOCK_PREFIX_V1 + _h("wrong-lock")
    else:
        expected_sha = _h("wrong-lock-payload")
    _assert_rejected(
        lambda: authority.load_step5_stage_b_trust_lock_v1(
            graph.lock_payload,
            out_of_band_expected_lock_id=expected_id,
            out_of_band_expected_payload_sha256=expected_sha,
        ),
        {authority.FAIL_TRUST_ANCHOR_V1},
    )


def test_profile_substitution_is_rejected_by_stage_b_lock(graph: Graph) -> None:
    profile = copy.deepcopy(graph.profile)
    profile["body"]["source_binding"]["source_archive_sha256"] = _h(
        "substituted-source"
    )
    profile = _profile_wrapper(profile["body"])
    _assert_rejected(
        lambda: authority.load_step5_authority_profile_v1(
            _canonical(profile), graph.lock_token
        ),
        {authority.FAIL_TRUST_ANCHOR_V1},
    )


def test_governance_signed_late_registry_is_rejected_by_stage_b_lock(
    graph: Graph,
) -> None:
    body = copy.deepcopy(graph.registry["body"])
    alternate_key = Ed25519PrivateKey.from_private_bytes(bytes([99]) * 32)
    body["role_keys_in_exact_order"][7] = _role_key(
        "PURE_REPLAYER", alternate_key
    )
    late_registry = _registry_wrapper(body, graph.private_keys[0])
    _assert_rejected(
        lambda: authority.load_step5_signer_registry_v1(
            _canonical(late_registry), graph.lock_token, graph.profile_token
        ),
        {authority.FAIL_TRUST_ANCHOR_V1},
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        (
            "profile_id",
            "phase2b_formal_wire_covert_qualification_profile_v2_"
            + _h("wrong-qv2-id"),
        ),
        ("profile_payload_sha256", _h("wrong-qv2-config")),
        ("source_sha256", _h("wrong-qv2-source")),
        ("dependency_lock_sha256", _h("wrong-qv2-lock")),
    ),
)
def test_each_qv2_pin_is_exact(
    graph: Graph, field: str, replacement: str
) -> None:
    profile = copy.deepcopy(graph.profile)
    profile["body"]["qv2_binding"][field] = replacement
    profile = _profile_wrapper(profile["body"])
    _assert_rejected(
        lambda: authority.load_step5_authority_profile_v1(
            _canonical(profile), graph.lock_token
        ),
        {authority.FAIL_BINDING_V1},
    )


@pytest.mark.parametrize(
    ("field", "prefix"),
    (
        ("logical_stage_wire_schema_id", "phase2b_c1_logical_stage_wire_schema_v1_"),
        ("logical_stage_wire_policy_id", "phase2b_c1_logical_stage_wire_policy_v2_"),
        (
            "stage_event_contract_manifest_id",
            "phase2b_c1_stage_event_contract_manifest_v1_",
        ),
    ),
)
def test_owner_logical_and_stage_ids_are_exact(
    graph: Graph, field: str, prefix: str
) -> None:
    profile = copy.deepcopy(graph.profile)
    profile["body"]["owner_authorization_binding"][field] = prefix + _h(field)
    profile = _profile_wrapper(profile["body"])
    _assert_rejected(
        lambda: authority.load_step5_authority_profile_v1(
            _canonical(profile), graph.lock_token
        ),
        {authority.FAIL_BINDING_V1},
    )


def test_runtime_image_digests_must_be_pairwise_distinct(graph: Graph) -> None:
    profile = copy.deepcopy(graph.profile)
    images = profile["body"]["runtime_image_bindings"]
    images["pure_replayer"] = images["runner"]
    profile = _profile_wrapper(profile["body"])
    _assert_rejected(
        lambda: authority.load_step5_authority_profile_v1(
            _canonical(profile), graph.lock_token
        ),
        {authority.FAIL_SCHEMA_V1},
    )


def test_decoded_and_injected_legacy_apis_always_fail_closed(graph: Graph) -> None:
    _assert_rejected(
        lambda: authority.verify_step5_authority_profile_v1(graph.profile),
        {authority.FAIL_TRUST_ANCHOR_V1},
    )
    _assert_rejected(
        lambda: authority.verify_step5_signer_registry_v1(
            graph.profile, graph.registry, verifier=lambda *_: True
        ),
        {authority.FAIL_TRUST_ANCHOR_V1},
    )
    _assert_rejected(
        lambda: authority.verify_step5_r01_authorization_status_receipt_v1(
            graph.profile,
            graph.registry,
            graph.r01,
            verifier=lambda *_: True,
        ),
        {authority.FAIL_NOT_IMPLEMENTED_V1},
    )
    _assert_rejected(
        lambda: authority.verify_step5_r02_synthetic_branch_receipt_v1(
            graph.profile,
            graph.registry,
            graph.r01,
            graph.r02,
            verifier=lambda *_: True,
        ),
        {authority.FAIL_NOT_IMPLEMENTED_V1},
    )
    with pytest.raises(TypeError):
        authority.load_step5_signer_registry_v1(
            graph.registry_payload,
            graph.lock_token,
            graph.profile_token,
            verifier=lambda *_: True,
        )


def test_bad_registry_signature_fails_even_when_stage_b_pins_its_bytes(
    graph: Graph,
) -> None:
    registry = copy.deepcopy(graph.registry)
    registry["signatures"][0]["signature_hex"] = "00" * 64
    lock, lock_payload = _stage_b_lock(graph.profile, registry)
    lock_token = _load_lock(lock, lock_payload)
    profile_token = authority.load_step5_authority_profile_v1(
        graph.profile_payload, lock_token
    )
    _assert_rejected(
        lambda: authority.load_step5_signer_registry_v1(
            _canonical(registry), lock_token, profile_token
        ),
        {authority.FAIL_SIGNATURE_V1},
    )


@pytest.mark.parametrize(
    "mutation",
    ("signature", "role", "purpose", "key", "epoch", "order", "duplicate"),
)
def test_r01_signature_role_purpose_key_epoch_order_and_duplicate_fail(
    graph: Graph, mutation: str
) -> None:
    candidate = copy.deepcopy(graph.r01)
    signatures = candidate["signatures"]
    if mutation == "signature":
        signatures[0]["signature_hex"] = "00" * 64
    elif mutation == "role":
        signatures[0]["role"] = "PURE_REPLAYER"
    elif mutation == "purpose":
        signatures[0]["purpose"] = "SIGN_R02_SYNTHETIC_BRANCH_CANDIDATE"
    elif mutation == "key":
        signatures[0]["signer_key_id"] = graph.rows[7]["key_id"]
    elif mutation == "epoch":
        signatures[0]["key_epoch"] = 2
    elif mutation == "order":
        candidate["signatures"] = list(reversed(signatures))
    else:
        candidate["signatures"] = [
            copy.deepcopy(signatures[0]),
            copy.deepcopy(signatures[0]),
        ]
    _assert_rejected(
        lambda: authority.load_step5_r01_authorization_status_candidate_v1(
            _canonical(candidate),
            graph.owner_payload,
            graph.lock_token,
            graph.profile_token,
            graph.registry_token,
        ),
        {authority.FAIL_SIGNATURE_V1},
    )


@pytest.mark.parametrize("mutation", ("public_key", "key_id", "control_domain"))
def test_registry_duplicate_key_and_domain_mutants_fail(
    graph: Graph, mutation: str
) -> None:
    body = copy.deepcopy(graph.registry["body"])
    rows = body["role_keys_in_exact_order"]
    if mutation == "public_key":
        rows[2]["public_key_hex"] = rows[1]["public_key_hex"]
        rows[2]["key_id"] = authority.phase2b_step5_role_key_id_v1(rows[2])
    elif mutation == "key_id":
        rows[2]["key_id"] = rows[1]["key_id"]
    else:
        rows[2]["control_domain"] = rows[1]["control_domain"]
        rows[2]["key_id"] = authority.phase2b_step5_role_key_id_v1(rows[2])
    registry = _registry_wrapper(body, graph.private_keys[0])
    lock, lock_payload = _stage_b_lock(graph.profile, registry)
    lock_token = _load_lock(lock, lock_payload)
    profile_token = authority.load_step5_authority_profile_v1(
        graph.profile_payload, lock_token
    )
    _assert_rejected(
        lambda: authority.load_step5_signer_registry_v1(
            _canonical(registry), lock_token, profile_token
        ),
        {
            authority.FAIL_SCHEMA_V1,
            authority.FAIL_CONTENT_ID_V1,
        },
    )


def _duplicate_top_level_id(payload: bytes) -> bytes:
    value = json.loads(payload)
    members: list[bytes] = []
    for key in sorted(value):
        encoded = _canonical(key) + b":" + _canonical(value[key])
        members.append(encoded)
        if key == "id":
            members.append(encoded)
    return b"{" + b",".join(members) + b"}"


def _invoke_loader(graph: Graph, artifact: str, payload: bytes) -> object:
    if artifact == "lock":
        return authority.load_step5_stage_b_trust_lock_v1(
            payload,
            out_of_band_expected_lock_id=graph.lock["id"],
            out_of_band_expected_payload_sha256=hashlib.sha256(
                graph.lock_payload
            ).hexdigest(),
        )
    if artifact == "profile":
        return authority.load_step5_authority_profile_v1(payload, graph.lock_token)
    if artifact == "registry":
        return authority.load_step5_signer_registry_v1(
            payload, graph.lock_token, graph.profile_token
        )
    if artifact == "r01":
        return authority.load_step5_r01_authorization_status_candidate_v1(
            payload,
            graph.owner_payload,
            graph.lock_token,
            graph.profile_token,
            graph.registry_token,
        )
    return authority.load_step5_r02_synthetic_branch_candidate_v1(
        payload,
        graph.lock_token,
        graph.profile_token,
        graph.registry_token,
        graph.r01_token,
    )


@pytest.mark.parametrize("artifact", ("lock", "profile", "registry", "r01", "r02"))
@pytest.mark.parametrize("mutation", ("duplicate", "noncanonical", "lf"))
def test_every_production_loader_rejects_bad_transport(
    graph: Graph, artifact: str, mutation: str
) -> None:
    payload = getattr(graph, f"{artifact}_payload")
    if mutation == "duplicate":
        payload = _duplicate_top_level_id(payload)
    elif mutation == "noncanonical":
        payload = b" " + payload
    else:
        payload += b"\n"
    _assert_rejected(
        lambda: _invoke_loader(graph, artifact, payload),
        {
            authority.FAIL_CANONICAL_V1,
            authority.FAIL_INPUT_V1,
        },
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("key_epoch", True),
        ("distinct_public_keys", 1),
        ("distinct_key_ids", 1),
    ),
)
def test_profile_bool_integer_type_confusion_fails(
    graph: Graph, field: str, value: object
) -> None:
    profile = copy.deepcopy(graph.profile)
    profile["body"]["role_policy"][field] = value
    profile = _profile_wrapper(profile["body"])
    _assert_rejected(
        lambda: authority.load_step5_authority_profile_v1(
            _canonical(profile), graph.lock_token
        ),
        {authority.FAIL_SCHEMA_V1},
    )


@pytest.mark.parametrize(
    ("field", "value"), (("registry_epoch", True), ("revocation_epoch", False))
)
def test_registry_bool_integer_epoch_confusion_fails(
    graph: Graph, field: str, value: object
) -> None:
    body = copy.deepcopy(graph.registry["body"])
    body[field] = value
    registry = _registry_wrapper(body, graph.private_keys[0])
    lock, lock_payload = _stage_b_lock(graph.profile, registry)
    lock_token = _load_lock(lock, lock_payload)
    profile_token = authority.load_step5_authority_profile_v1(
        graph.profile_payload, lock_token
    )
    _assert_rejected(
        lambda: authority.load_step5_signer_registry_v1(
            _canonical(registry), lock_token, profile_token
        ),
        {authority.FAIL_SCHEMA_V1},
    )


@pytest.mark.parametrize("field", ("sequence", "revocation_count"))
def test_r01_bool_integer_status_confusion_fails(
    graph: Graph, field: str
) -> None:
    body = copy.deepcopy(graph.r01["body"])
    body["status_checkpoint_candidate"][field] = False
    candidate = _resigned_r01(graph, body)
    _assert_rejected(
        lambda: authority.load_step5_r01_authorization_status_candidate_v1(
            _canonical(candidate),
            graph.owner_payload,
            graph.lock_token,
            graph.profile_token,
            graph.registry_token,
        ),
        {authority.FAIL_CLAIM_BOUNDARY_V1},
    )


def test_authority_number_surface_rejects_float_values() -> None:
    _assert_rejected(
        lambda: authority.canonical_step5_json_bytes_v1({"value": 1.5}),
        {authority.FAIL_CANONICAL_V1},
    )
    _assert_rejected(
        lambda: authority.decode_canonical_step5_json_v1(b'{"value":1.5}'),
        {authority.FAIL_CANONICAL_V1},
    )


@pytest.mark.parametrize("mutation", ("stale", "revoked", "claim_true"))
def test_r01_stale_revoked_and_true_claims_fail(
    graph: Graph, mutation: str
) -> None:
    body = copy.deepcopy(graph.r01["body"])
    checkpoint = body["status_checkpoint_candidate"]
    if mutation == "stale":
        checkpoint["sequence"] = 1
        checkpoint["previous_status_head_id"] = "phase2b_step5_status_head_v1_" + _h(
            "old-head"
        )
    elif mutation == "revoked":
        checkpoint["effective_status"] = "REVOKED"
        checkpoint["revocation_receipt_id"] = "phase2b_step5_revocation_v1_" + _h(
            "revocation"
        )
        checkpoint["revocation_count"] = 1
    else:
        body["claim_boundary"]["r01_satisfied"] = True
    candidate = _resigned_r01(graph, body)
    _assert_rejected(
        lambda: authority.load_step5_r01_authorization_status_candidate_v1(
            _canonical(candidate),
            graph.owner_payload,
            graph.lock_token,
            graph.profile_token,
            graph.registry_token,
        ),
        {authority.FAIL_CLAIM_BOUNDARY_V1},
    )


@pytest.mark.parametrize(
    "mutation", ("actual", "cross_profile", "r01_exchange", "claim_true")
)
def test_r02_actual_cross_profile_r01_exchange_and_true_claims_fail(
    graph: Graph, mutation: str
) -> None:
    body = copy.deepcopy(graph.r02["body"])
    if mutation == "actual":
        branch = body["branch_selection"]
        branch["actual_corpus_input"] = True
        branch["actual_input_count"] = 1
        branch["actual_campaign_id"] = "phase2b_actual_unsealed_960_campaign_v2_" + _h(
            "forbidden-actual"
        )
        branch["actual_material_consumed"] = True
    elif mutation == "cross_profile":
        body["authority_profile_id"] = authority.AUTHORITY_PROFILE_PREFIX_V1 + _h(
            "other-profile"
        )
    elif mutation == "r01_exchange":
        body["r01_candidate_id"] = authority.R01_CANDIDATE_PREFIX_V1 + _h(
            "other-r01"
        )
    else:
        body["claim_boundary"]["r02_satisfied"] = True
    candidate = _resigned_r02(graph, body)
    _assert_rejected(
        lambda: authority.load_step5_r02_synthetic_branch_candidate_v1(
            _canonical(candidate),
            graph.lock_token,
            graph.profile_token,
            graph.registry_token,
            graph.r01_token,
        ),
        {
            authority.FAIL_BINDING_V1,
            authority.FAIL_CLAIM_BOUNDARY_V1,
        },
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("authority_class", "MUTATED_OWNER_AUTHORITY_CLASS"),
        (
            "authority_requirement_id",
            "phase2b_c1_owner_accepted_container_authority_requirement_v1_"
            + _h("mutated-authority-requirement"),
        ),
        (
            "actor_profile_content_id",
            "phase2b_c1_container_actor_profile_v2_"
            + _h("mutated-actor-profile"),
        ),
        ("actor_profile_canonical_sha256", _h("mutated-actor-profile-bytes")),
    ),
)
def test_r02_replays_every_owner_exact_contract_binding(
    graph: Graph, field: str, replacement: str
) -> None:
    body = copy.deepcopy(graph.r02["body"])
    body["declared_observed_selected_branch_bindings"][field] = replacement
    candidate = _resigned_r02(graph, body)
    _assert_rejected(
        lambda: authority.load_step5_r02_synthetic_branch_candidate_v1(
            _canonical(candidate),
            graph.lock_token,
            graph.profile_token,
            graph.registry_token,
            graph.r01_token,
        ),
        {authority.FAIL_BINDING_V1},
    )


def test_verified_cross_graph_r01_cannot_enter_r02_or_snapshot(graph: Graph) -> None:
    profile_b = copy.deepcopy(graph.profile)
    profile_b["body"]["source_binding"]["source_archive_sha256"] = _h(
        "graph-b-source-archive"
    )
    profile_b = _profile_wrapper(profile_b["body"])
    registry_b_body = copy.deepcopy(graph.registry["body"])
    registry_b_body["authority_profile_id"] = profile_b["id"]
    registry_b = _registry_wrapper(registry_b_body, graph.private_keys[0])
    lock_b_token, profile_b_token, registry_b_token = _load_profile_registry_for(
        profile_b, registry_b
    )

    r01_b_body = copy.deepcopy(graph.r01["body"])
    r01_b_body["authority_profile_id"] = profile_b["id"]
    r01_b_body["signer_registry_id"] = registry_b["id"]
    r01_b = _resigned_r01(graph, r01_b_body)
    r01_b_token = authority.load_step5_r01_authorization_status_candidate_v1(
        _canonical(r01_b),
        graph.owner_payload,
        lock_b_token,
        profile_b_token,
        registry_b_token,
    )

    r02_b_body = copy.deepcopy(graph.r02["body"])
    r02_b_body["authority_profile_id"] = profile_b["id"]
    r02_b_body["signer_registry_id"] = registry_b["id"]
    r02_b_body["r01_candidate_id"] = r01_b_token.artifact_id
    r02_b_body["source_binding"] = copy.deepcopy(
        profile_b["body"]["source_binding"]
    )
    r02_b = _resigned_r02(graph, r02_b_body)
    r02_b_token = authority.load_step5_r02_synthetic_branch_candidate_v1(
        _canonical(r02_b),
        lock_b_token,
        profile_b_token,
        registry_b_token,
        r01_b_token,
    )

    r02_a_cross_body = copy.deepcopy(graph.r02["body"])
    r02_a_cross_body["r01_candidate_id"] = r01_b_token.artifact_id
    r02_a_cross = _resigned_r02(graph, r02_a_cross_body)
    _assert_rejected(
        lambda: authority.load_step5_r02_synthetic_branch_candidate_v1(
            _canonical(r02_a_cross),
            graph.lock_token,
            graph.profile_token,
            graph.registry_token,
            r01_b_token,
        ),
        {authority.FAIL_TRUST_ANCHOR_V1},
    )
    _assert_rejected(
        lambda: authority.derive_step5_r01_r02_candidate_snapshot_v1(
            graph.lock_token,
            graph.profile_token,
            graph.registry_token,
            r01_b_token,
            r02_b_token,
            READINESS_DAG.read_bytes(),
        ),
        {authority.FAIL_TRUST_ANCHOR_V1},
    )


def test_verified_registry_cannot_be_laundered_through_hybrid_lock(graph: Graph) -> None:
    profile_b = copy.deepcopy(graph.profile)
    profile_b["body"]["source_binding"]["source_archive_sha256"] = _h(
        "hybrid-graph-b-source-archive"
    )
    profile_b = _profile_wrapper(profile_b["body"])
    registry_b_body = copy.deepcopy(graph.registry["body"])
    registry_b_body["authority_profile_id"] = profile_b["id"]
    registry_b = _registry_wrapper(registry_b_body, graph.private_keys[0])
    _, _, registry_b_token = _load_profile_registry_for(profile_b, registry_b)

    hybrid_lock, hybrid_lock_payload = _stage_b_lock(graph.profile, registry_b)
    hybrid_lock_token = _load_lock(hybrid_lock, hybrid_lock_payload)
    profile_a_hybrid_token = authority.load_step5_authority_profile_v1(
        graph.profile_payload, hybrid_lock_token
    )
    hybrid_r01_body = copy.deepcopy(graph.r01["body"])
    hybrid_r01_body["signer_registry_id"] = registry_b["id"]
    hybrid_r01 = _resigned_r01(graph, hybrid_r01_body)
    _assert_rejected(
        lambda: authority.load_step5_r01_authorization_status_candidate_v1(
            _canonical(hybrid_r01),
            graph.owner_payload,
            hybrid_lock_token,
            profile_a_hybrid_token,
            registry_b_token,
        ),
        {authority.FAIL_TRUST_ANCHOR_V1},
    )


def test_owner_payload_mutation_is_rejected(graph: Graph) -> None:
    mutated = graph.owner_payload[:-1] + b" " + graph.owner_payload[-1:]
    _assert_rejected(
        lambda: authority.load_step5_r01_authorization_status_candidate_v1(
            graph.r01_payload,
            mutated,
            graph.lock_token,
            graph.profile_token,
            graph.registry_token,
        ),
        {authority.FAIL_BINDING_V1},
    )


def test_old_readiness_overlay_derivation_is_permanently_fail_closed(
    graph: Graph,
) -> None:
    _assert_rejected(
        lambda: authority.derive_step5_r01_r02_readiness_overlay_v1(
            graph.lock_token,
            graph.profile_token,
            graph.registry_token,
            graph.r01_token,
            graph.r02_token,
            READINESS_DAG.read_bytes(),
        ),
        {authority.FAIL_NOT_IMPLEMENTED_V1},
    )
