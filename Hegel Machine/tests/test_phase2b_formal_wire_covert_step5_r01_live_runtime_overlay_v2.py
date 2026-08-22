from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import struct
from types import FunctionType
from typing import Callable, Mapping

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hegel_machine import phase2b_formal_wire_covert_step5_authority_v2 as authority
from hegel_machine import (
    phase2b_formal_wire_covert_step5_r01_external_witness_admission_contract_v1
    as admission,
)
from hegel_machine import (
    phase2b_formal_wire_covert_step5_r01_live_runtime_overlay_v2 as overlay,
)
from hegel_machine import phase2b_formal_wire_covert_step5_store_replay_core_v2 as replay
from hegel_machine import phase2b_formal_wire_covert_step5_store_v2 as store
from hegel_machine import phase2b_formal_wire_covert_step5_runtime_closure_v1 as closure
from hegel_machine.rfc8785_v1 import canonicalize_rfc8785_v1

import test_phase2b_formal_wire_covert_step5_r01_evidence_contract_v1 as evidence_fixture
import test_phase2b_formal_wire_covert_step5_r01_external_witness_admission_contract_v1 as admission_fixture
import test_phase2b_formal_wire_covert_step5_runtime_closure_v1 as closure_fixture


# Preserve the dependency entrypoint objects that the overlay captured when this
# test module was collected.  Earlier combined suites deliberately reload some
# dependency modules, so their later module attributes are not stable identity
# oracles for definition-time capture.
_COLLECTION_ADMISSION_VERIFY = (
    admission.verify_external_witness_admission_transcript_candidate_v1
)
_COLLECTION_ADMISSION_REQUIRE = (
    admission.require_verified_external_witness_admission_candidate_v1
)
_COLLECTION_CLOSURE_VERIFY = closure.verify_offline_step5_image_closure_mechanics_v1
_COLLECTION_CLOSURE_REQUIRE = (
    closure.require_verified_offline_step5_image_closure_mechanics_v1
)
_COLLECTION_REPLAY_VERIFY = replay.verify_cas_store_complete_session_replay_core_v2
_COLLECTION_REPLAY_REQUIRE = (
    replay.require_verified_cas_store_complete_session_replay_core_v2
)


@dataclass(frozen=True)
class _OverlayCorpus:
    call: dict[str, object]
    payloads: Mapping[str, bytes]
    roots: Mapping[str, Mapping[str, object]]
    admission_corpus: object
    runtime_graph: object


@dataclass(frozen=True)
class _CasEvidence:
    plan: bytes
    requests: tuple[bytes, ...]
    objects: tuple[bytes, ...]
    receipts: tuple[bytes, ...]
    checkpoints: tuple[bytes, ...]
    heads: tuple[bytes, ...]
    responses: tuple[bytes, ...]


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _store_sha(payload: bytes) -> str:
    return "sha256:" + _sha(payload)


def _json(payload: bytes) -> dict[str, object]:
    value = json.loads(payload)
    assert type(value) is dict
    return value


def _public_hex(private_key: Ed25519PrivateKey) -> str:
    return private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()


def _private_seed(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )


def _nonclaims() -> dict[str, object]:
    return {field: False for field in overlay.POLICY_ALONE_NONCLAIMS_FIELDS_V2}


def _authority_signature(
    *,
    kind: str,
    body: Mapping[str, object],
    row: Mapping[str, object],
    private_key: Ed25519PrivateKey,
    purpose: str,
) -> dict[str, object]:
    preimage = authority.phase2b_step5_signature_preimage_v2(
        artifact_kind=kind,
        body=dict(body),
        role=row["role"],
        purpose=purpose,
        signer_key_id=row["key_id"],
        key_epoch=row["key_epoch"],
    )
    return {
        "schema_version": authority.DETACHED_SIGNATURE_SCHEMA_V2,
        "algorithm": overlay.ALGORITHM_V2,
        "role": row["role"],
        "purpose": purpose,
        "key_epoch": row["key_epoch"],
        "signer_key_id": row["key_id"],
        "signature_hex": private_key.sign(preimage).hex(),
    }


def _external_signature(
    *,
    kind: str,
    body: Mapping[str, object],
    content_id: str,
    private_key: Ed25519PrivateKey,
    key_id: str,
    purpose: str,
) -> dict[str, object]:
    preimage = overlay._external_signature_preimage(
        kind=kind,
        body=body,
        content_id=content_id,
        signer_key_id=key_id,
        key_epoch=1,
        purpose=purpose,
    )
    return {
        "schema_version": (
            "hegel-machine-phase2b-step5-live-runtime-external-signature/2"
        ),
        "algorithm": overlay.ALGORITHM_V2,
        "signer_key_id": key_id,
        "key_epoch": 1,
        "purpose": purpose,
        "signature_hex": private_key.sign(preimage).hex(),
    }


def _signed_overlay_object(
    kind: str,
    body: Mapping[str, object],
    signers: tuple[tuple[str, Ed25519PrivateKey, Mapping[str, object] | str, str], ...],
) -> tuple[bytes, dict[str, object]]:
    exact_body = copy.deepcopy(dict(body))
    expected_fields = overlay._BODY_FIELDS_BY_KIND[kind]
    assert frozenset(exact_body) == expected_fields
    content_id = overlay._content_id(kind, exact_body)
    signatures: list[dict[str, object]] = []
    for mode, private_key, signer, purpose in signers:
        if mode == "authority":
            assert isinstance(signer, Mapping)
            signatures.append(
                _authority_signature(
                    kind=kind,
                    body=exact_body,
                    row=signer,
                    private_key=private_key,
                    purpose=purpose,
                )
            )
        else:
            assert type(signer) is str
            signatures.append(
                _external_signature(
                    kind=kind,
                    body=exact_body,
                    content_id=content_id,
                    private_key=private_key,
                    key_id=signer,
                    purpose=purpose,
                )
            )
    root = {
        "schema_version": overlay._SCHEMA_BY_KIND[kind],
        "body": exact_body,
        overlay._ID_FIELD_BY_KIND[kind]: content_id,
        "signatures_in_exact_order": signatures,
    }
    assert frozenset(root) == overlay._TOP_FIELDS_BY_KIND[kind]
    return overlay._canonical(root), root


def _unified_authority(runtime_graph: object) -> evidence_fixture._AuthorityGraph:
    arguments = runtime_graph.verify_arguments
    role_rows, private_keys = closure_fixture._role_keys()
    profile_payload = arguments["authority_profile_payload"]
    registry_payload = arguments["signer_registry_payload"]
    lock_payload = arguments["stage_b_trust_lock_payload"]
    assert type(profile_payload) is bytes
    assert type(registry_payload) is bytes
    assert type(lock_payload) is bytes
    profile = _json(profile_payload)
    return evidence_fixture._AuthorityGraph(
        private_keys={
            role: private_keys[index]
            for index, role in enumerate(authority.ROLE_ORDER_V2)
        },
        role_keys=copy.deepcopy(role_rows),
        service_bindings=copy.deepcopy(
            profile["body"]["service_static_bindings_in_exact_order"]
        ),
        profile_body=copy.deepcopy(profile["body"]),
        profile_payload=profile_payload,
        registry_payload=registry_payload,
        lock_id=runtime_graph.lock_id,
        lock_payload=lock_payload,
    )


def _custody_policy(
    base_graph: evidence_fixture._EvidenceGraph,
    *,
    online_private: Ed25519PrivateKey,
    custody_private: Ed25519PrivateKey,
    online_key_id: str,
    custody_key_id: str,
) -> tuple[bytes, dict[str, object]]:
    bootstrap = _json(base_graph.payloads["policy"])
    profile = _json(base_graph.authority.profile_payload)
    registry = _json(base_graph.authority.registry_payload)
    body = {
        "schema_version": overlay.CUSTODY_ATTESTATION_POLICY_SCHEMA_V2,
        "status": overlay.STATUS_V2,
        "claim_level": overlay.CLAIM_LEVEL_V2,
        "canonicalization": overlay.CANONICALIZATION_V2,
        "policy_epoch": 1,
        "stage_b_trust_lock_id": base_graph.authority.lock_id,
        "stage_b_trust_lock_payload_sha256": _sha(
            base_graph.authority.lock_payload
        ),
        "authority_profile_id": profile["id"],
        "authority_profile_payload_sha256": _sha(
            base_graph.authority.profile_payload
        ),
        "signer_registry_id": registry["id"],
        "signer_registry_payload_sha256": _sha(
            base_graph.authority.registry_payload
        ),
        "attempt_context_id": base_graph.attempt_id,
        "bootstrap_external_witness_policy_id": bootstrap[
            "external_witness_policy_id"
        ],
        "bootstrap_external_witness_policy_payload_sha256": _sha(
            base_graph.payloads["policy"]
        ),
        "online_witness_key_id": online_key_id,
        "online_witness_key_epoch": 1,
        "online_witness_public_key_hex": _public_hex(online_private),
        "custody_attestor_key_id": custody_key_id,
        "custody_attestor_key_epoch": 1,
        "custody_attestor_public_key_hex": _public_hex(custody_private),
        "custody_attestor_signature_purpose": (
            overlay.ATTEST_ONLINE_KEY_NONEXPORTABILITY_PURPOSE_V2
        ),
        "attestation_evidence_format": "TPM2_QUOTE_X509_CHAIN_V1",
        "trusted_certificate_root_ids_in_exact_order": ["test-root-ca-v1"],
        "allowed_certificate_signature_algorithms_in_exact_order": ["Ed25519"],
        "minimum_attestation_security_version": 1,
        "challenge_nonce_size_bytes": 32,
        "report_binding_required": True,
        "online_key_binding_required": True,
        "online_key_generated_in_attested_boundary_required": True,
        "online_key_nonexportable_required": True,
        "certificate_revocation_check_required": True,
        "maximum_attestation_evidence_bytes": 4096,
        "policy_alone_nonclaims": _nonclaims(),
    }
    return _signed_overlay_object(
        "custody_attestation_policy",
        body,
        ((
            "external",
            base_graph.witness_private_key,
            bootstrap["witness_key_id"],
            overlay.SIGN_ONLINE_WITNESS_POLICY_PURPOSE_V2,
        ),),
    )


def _policy_refs(
    admission_corpus: object,
    runtime_graph: object,
) -> dict[str, object]:
    call = admission_corpus.call
    roots = admission_corpus.roots
    profile = _json(call["authority_profile_payload"])
    registry = _json(call["signer_registry_payload"])
    closure_policy = _json(runtime_graph.policy_payload)
    closure_evidence = _json(runtime_graph.evidence_payload)
    return {
        "stage_b_trust_lock_id": admission_corpus.graph.authority.lock_id,
        "stage_b_trust_lock_payload_sha256": _sha(call["stage_b_trust_lock_payload"]),
        "authority_profile_id": profile["id"],
        "authority_profile_payload_sha256": _sha(call["authority_profile_payload"]),
        "signer_registry_id": registry["id"],
        "signer_registry_payload_sha256": _sha(call["signer_registry_payload"]),
        "attempt_context_id": admission_corpus.graph.attempt_id,
        "admission_candidate_result_id": roots["result"]["candidate_result_id"],
        "admission_candidate_result_payload_sha256": _sha(
            admission_corpus.payloads["result"]
        ),
        "bootstrap_external_witness_policy_id": roots["bootstrap_policy"][
            "external_witness_policy_id"
        ],
        "bootstrap_external_witness_policy_payload_sha256": _sha(
            admission_corpus.payloads["bootstrap_policy"]
        ),
        "online_witness_policy_id": roots["online_policy"][
            "online_witness_policy_id"
        ],
        "online_witness_policy_payload_sha256": _sha(
            admission_corpus.payloads["online_policy"]
        ),
        "durable_anchor_policy_id": roots["durable_policy"][
            "durable_anchor_policy_id"
        ],
        "durable_anchor_policy_payload_sha256": _sha(
            admission_corpus.payloads["durable_policy"]
        ),
        "runtime_closure_policy_id": closure_policy["id"],
        "runtime_closure_policy_payload_sha256": _sha(runtime_graph.policy_payload),
        "runtime_closure_evidence_id": closure_evidence["id"],
        "runtime_closure_evidence_payload_sha256": _sha(runtime_graph.evidence_payload),
    }


def _role(
    admission_corpus: object, role: str
) -> tuple[Mapping[str, object], Ed25519PrivateKey]:
    graph = admission_corpus.graph.authority
    return (
        graph.role_keys[authority.ROLE_ORDER_V2.index(role)],
        graph.private_keys[role],
    )


def _live_and_anchor_policies(
    admission_corpus: object,
    runtime_graph: object,
    custody_payload: bytes,
    custody_root: Mapping[str, object],
    *,
    live_mutator: Callable[[dict[str, object]], None] | None = None,
    anchor_mutator: Callable[[dict[str, object]], None] | None = None,
) -> tuple[bytes, dict[str, object], bytes, dict[str, object], dict[str, object]]:
    refs = _policy_refs(admission_corpus, runtime_graph)
    bootstrap = admission_corpus.roots["bootstrap_policy"]
    online = admission_corpus.roots["online_policy"]["body"]
    durable = admission_corpus.roots["durable_policy"]["body"]
    graph = admission_corpus.graph.authority
    services = {
        row["service_name"]: row for row in graph.service_bindings
    }
    source, source_private = _role(admission_corpus, "SOURCE_RUNTIME_CUSTODIAN")
    pure, _pure_private = _role(admission_corpus, "PURE_REPLAYER")
    cas, _cas_private = _role(admission_corpus, "CAS_STORE")
    runner = services["runner"]
    pure_service = services["pure_replayer"]
    cas_service = services["cas_store"]
    cgroup_sha = _sha(b"overlay-online-witness-cgroup-v2")
    body: dict[str, object] = {
        "schema_version": overlay.LIVE_RUNTIME_POLICY_SCHEMA_V2,
        "status": overlay.STATUS_V2,
        "claim_level": overlay.CLAIM_LEVEL_V2,
        "canonicalization": overlay.CANONICALIZATION_V2,
        "policy_epoch": 1,
        **refs,
        "runtime_closure_service_name": "runner",
        "runtime_identity": copy.deepcopy(runtime_graph.policy_body["runtime_identity"]),
        "custody_attestation_policy_id": custody_root[
            "custody_attestation_policy_id"
        ],
        "custody_attestation_policy_payload_sha256": _sha(custody_payload),
        "bootstrap_key_id": bootstrap["witness_key_id"],
        "bootstrap_key_epoch": bootstrap["witness_key_epoch"],
        "bootstrap_public_key_hex": bootstrap["public_key_hex"],
        "bootstrap_authorized_signature_purposes_in_exact_order": list(
            admission.BOOTSTRAP_PURPOSES_IN_EXACT_ORDER_V1
        ),
        "online_witness_key_id": online["online_key_id"],
        "online_witness_key_epoch": online["online_key_epoch"],
        "online_witness_public_key_hex": online["online_public_key_hex"],
        "online_witness_authorized_signature_purposes_in_exact_order": list(
            overlay.LIVE_OVERLAY_ONLINE_PURPOSES_IN_EXACT_ORDER_V2
        ),
        "online_witness_service_binding_id": online["service_binding_id"],
        "online_witness_control_domain_id": online["witness_control_domain_id"],
        "online_witness_service_uid": online["service_uid"],
        "online_witness_service_gid": online["service_gid"],
        "online_witness_service_user_namespace_inode": online[
            "service_user_namespace_inode"
        ],
        "online_witness_expected_cgroup_sha256": cgroup_sha,
        "source_runtime_custodian_role": "SOURCE_RUNTIME_CUSTODIAN",
        "source_runtime_custodian_control_domain": source["control_domain"],
        "source_runtime_custodian_service_binding_id": runner["binding_id"],
        "source_runtime_custodian_service_uid": runner["uid"],
        "source_runtime_custodian_service_gid": runner["gid"],
        "source_runtime_custodian_key_id": source["key_id"],
        "source_runtime_custodian_key_epoch": source["key_epoch"],
        "source_runtime_custodian_public_key_hex": source["public_key_hex"],
        "source_runtime_custodian_authorized_signature_purposes_in_exact_order": list(
            source["authorized_signature_purposes"]
        ),
        "pure_replayer_role": "PURE_REPLAYER",
        "pure_replayer_control_domain": pure["control_domain"],
        "pure_replayer_service_binding_id": pure_service["binding_id"],
        "pure_replayer_service_uid": pure_service["uid"],
        "pure_replayer_service_gid": pure_service["gid"],
        "pure_replayer_key_id": pure["key_id"],
        "pure_replayer_key_epoch": pure["key_epoch"],
        "pure_replayer_public_key_hex": pure["public_key_hex"],
        "pure_replayer_authorized_signature_purposes_in_exact_order": list(
            pure["authorized_signature_purposes"]
        ),
        "cas_store_role": "CAS_STORE",
        "cas_store_control_domain": cas["control_domain"],
        "cas_store_service_binding_id": cas_service["binding_id"],
        "cas_store_service_uid": cas_service["uid"],
        "cas_store_service_gid": cas_service["gid"],
        "cas_store_key_id": cas["key_id"],
        "cas_store_key_epoch": cas["key_epoch"],
        "cas_store_public_key_hex": cas["public_key_hex"],
        "cas_store_authorized_signature_purposes_in_exact_order": list(
            cas["authorized_signature_purposes"]
        ),
        "custody_attestor_key_id": online["custody_attestor_key_id"],
        "custody_attestor_key_epoch": online["custody_attestor_key_epoch"],
        "custody_attestor_public_key_hex": online["custody_attestor_public_key_hex"],
        "custody_attestor_authorized_signature_purposes_in_exact_order": list(
            overlay.CUSTODY_PURPOSES_IN_EXACT_ORDER_V2
        ),
        "custody_attestation_profile": "TPM2_QUOTE_X509_CHAIN_V1",
        "independent_custody_attestor_required": True,
        "online_key_nonexportable_required": True,
        "live_runtime_anchor_key_id": durable["anchor_key_id"],
        "live_runtime_anchor_key_epoch": durable["anchor_key_epoch"],
        "live_runtime_anchor_public_key_hex": durable["anchor_public_key_hex"],
        "live_runtime_anchor_authorized_signature_purposes_in_exact_order": list(
            overlay.ANCHOR_PURPOSES_IN_EXACT_ORDER_V2
        ),
        "live_runtime_anchor_control_domain_id": durable["anchor_control_domain_id"],
        "transport_address_family": "AF_UNIX",
        "transport_socket_type": "SOCK_SEQPACKET",
        "transport_peer_credential_option": "SO_PEERCRED",
        "transport_peer_credential_struct_format": "=iii",
        "transport_peer_credential_struct_size_bytes": 12,
        "exactly_one_challenge_packet_required": True,
        "exactly_one_response_packet_required": True,
        "scm_rights_allowed": False,
        "maximum_packet_bytes": 1_000_000,
        "pidfd_required": True,
        "procfs_identity_required": True,
        "entry_exit_identity_equal_required": True,
        "challenge_nonce_size_bytes": 32,
        "response_nonce_size_bytes": 32,
        "maximum_round_trip_nanoseconds": 1_000_000,
        "monotonic_clock": "CLOCK_MONOTONIC_RAW",
        "caller_nonce_forbidden": True,
        "caller_sequence_forbidden": True,
        "one_active_session_per_attempt_and_online_key_required": True,
        "cas_retention_plan_kind": overlay.CAS_PLAN_KIND_V2,
        "cas_chain_id_profile": "CONTENT_DERIVED_EXACT_LIVE_RESULT_V2",
        "cas_expected_object_count": 6,
        "cas_artifact_kinds_in_exact_order": list(
            overlay.CAS_ARTIFACT_KINDS_IN_EXACT_ORDER_V2
        ),
        "cas_complete_head_required": True,
        "cas_durable_before_return_required": True,
        "cas_readback_required": True,
        "anchor_slot_key_profile": durable["admission_slot_key_profile"],
        "anchor_compare_and_append_semantics": durable[
            "compare_and_append_semantics"
        ],
        "anchor_durable_before_return_required": True,
        "policy_alone_nonclaims": _nonclaims(),
    }
    if live_mutator is not None:
        live_mutator(body)
    live_payload, live_root = _signed_overlay_object(
        "live_runtime_policy",
        body,
        (
            (
                "external",
                admission_corpus.graph.witness_private_key,
                bootstrap["witness_key_id"],
                overlay.SIGN_ONLINE_WITNESS_POLICY_PURPOSE_V2,
            ),
            (
                "authority",
                source_private,
                source,
                overlay.SIGN_RUNTIME_CLOSURE_PURPOSE_V2,
            ),
        ),
    )
    genesis = overlay._genesis_mapping(
        attempt_id=refs["attempt_context_id"],
        admission_result_id=refs["admission_candidate_result_id"],
        admission_result_sha=refs["admission_candidate_result_payload_sha256"],
    )
    zeros = "0" * 64
    anchor_body: dict[str, object] = {
        "schema_version": overlay.LIVE_RUNTIME_ANCHOR_POLICY_SCHEMA_V2,
        "status": overlay.STATUS_V2,
        "claim_level": overlay.CLAIM_LEVEL_V2,
        "canonicalization": overlay.CANONICALIZATION_V2,
        "policy_epoch": 1,
        **{
            key: refs[key]
            for key in (
                "stage_b_trust_lock_id",
                "stage_b_trust_lock_payload_sha256",
                "authority_profile_id",
                "authority_profile_payload_sha256",
                "signer_registry_id",
                "signer_registry_payload_sha256",
                "attempt_context_id",
                "admission_candidate_result_id",
                "admission_candidate_result_payload_sha256",
                "bootstrap_external_witness_policy_id",
                "bootstrap_external_witness_policy_payload_sha256",
                "online_witness_policy_id",
                "online_witness_policy_payload_sha256",
                "durable_anchor_policy_id",
                "durable_anchor_policy_payload_sha256",
            )
        },
        "live_runtime_policy_id": live_root["live_runtime_policy_id"],
        "live_runtime_policy_payload_sha256": _sha(live_payload),
        "anchor_namespace": durable["anchor_namespace"],
        "anchor_control_domain_id": durable["anchor_control_domain_id"],
        "anchor_service_binding_id": "external-live-anchor-service-v2",
        "anchor_service_uid": 65000,
        "anchor_service_gid": 65001,
        "anchor_service_user_namespace_inode": 65002,
        "anchor_key_id": durable["anchor_key_id"],
        "anchor_key_epoch": durable["anchor_key_epoch"],
        "anchor_public_key_hex": durable["anchor_public_key_hex"],
        "algorithm": overlay.ALGORITHM_V2,
        "authorized_signature_purposes_in_exact_order": list(
            overlay.ANCHOR_PURPOSES_IN_EXACT_ORDER_V2
        ),
        "requester_role": "SOURCE_RUNTIME_CUSTODIAN",
        "requester_control_domain": source["control_domain"],
        "requester_key_id": source["key_id"],
        "requester_key_epoch": source["key_epoch"],
        "requester_public_key_hex": source["public_key_hex"],
        "requester_signature_purpose": overlay.SIGN_RUNTIME_CLOSURE_PURPOSE_V2,
        "genesis_checkpoint_id": (
            "phase2b_step5_live_runtime_anchor_checkpoint_v2_" + zeros
        ),
        "genesis_head_id": "phase2b_step5_live_runtime_anchor_head_v2_" + zeros,
        "genesis_mapping": copy.deepcopy(genesis),
        "oob_high_water_checkpoint_id": (
            "phase2b_step5_live_runtime_anchor_checkpoint_v2_" + zeros
        ),
        "oob_high_water_checkpoint_payload_sha256": zeros,
        "oob_high_water_checkpoint_sequence": 0,
        "oob_high_water_head_id": (
            "phase2b_step5_live_runtime_anchor_head_v2_" + zeros
        ),
        "oob_high_water_head_payload_sha256": zeros,
        "oob_high_water_head_sequence": 0,
        "oob_high_water_mapping": copy.deepcopy(genesis),
        "admission_slot_key_profile": durable["admission_slot_key_profile"],
        "compare_and_append_semantics": durable["compare_and_append_semantics"],
        "durable_before_return_required": True,
        "cas_complete_head_required": True,
    }
    if anchor_mutator is not None:
        anchor_mutator(anchor_body)
    anchor_payload, anchor_root = _signed_overlay_object(
        "live_runtime_anchor_policy",
        anchor_body,
        ((
            "external",
            admission_corpus.durable_private_key,
            durable["anchor_key_id"],
            overlay.SIGN_LIVE_RUNTIME_ANCHOR_POLICY_PURPOSE_V2,
        ),),
    )
    return live_payload, live_root, anchor_payload, anchor_root, body


def _session_common(
    *,
    refs: Mapping[str, object],
    custody_id: str,
    live_policy_id: str,
    anchor_policy_id: str,
    session_id: str,
    sequence: int,
) -> dict[str, object]:
    return {
        "status": overlay.STATUS_V2,
        "claim_level": overlay.CLAIM_LEVEL_V2,
        "stage_b_trust_lock_id": refs["stage_b_trust_lock_id"],
        "authority_profile_id": refs["authority_profile_id"],
        "signer_registry_id": refs["signer_registry_id"],
        "attempt_context_id": refs["attempt_context_id"],
        "bootstrap_external_witness_policy_id": refs[
            "bootstrap_external_witness_policy_id"
        ],
        "online_witness_policy_id": refs["online_witness_policy_id"],
        "durable_anchor_policy_id": refs["durable_anchor_policy_id"],
        "runtime_closure_policy_id": refs["runtime_closure_policy_id"],
        "runtime_closure_evidence_id": refs["runtime_closure_evidence_id"],
        "custody_attestation_policy_id": custody_id,
        "live_runtime_policy_id": live_policy_id,
        "live_runtime_anchor_policy_id": anchor_policy_id,
        "admission_candidate_result_id": refs["admission_candidate_result_id"],
        "admission_candidate_result_payload_sha256": refs[
            "admission_candidate_result_payload_sha256"
        ],
        "session_id": session_id,
        "session_sequence": sequence,
    }


def _process_snapshot(
    *, phase: str, uid: int, gid: int, cgroup_sha256: str
) -> dict[str, object]:
    pid = 4242
    body: dict[str, object] = {
        "schema_version": overlay.PROCESS_SNAPSHOT_SCHEMA_V2,
        "phase": phase,
        "observation_source": "KERNEL_SO_PEERCRED_PIDFD_PROCFS",
        "address_family": "AF_UNIX",
        "socket_type": "SOCK_SEQPACKET",
        "peer_credential_option": "SO_PEERCRED",
        "peer_credential_struct_format": "=iii",
        "peer_credential_struct_size_bytes": 12,
        "so_peercred_payload_sha256": _sha(struct.pack("=iii", pid, uid, gid)),
        "accepted_socket_inode": 70001,
        "server_socket_inode": 70002,
        "peer_pid": pid,
        "peer_uid": uid,
        "peer_gid": gid,
        "host_boot_id": "boot-id-overlay-v2",
        "process_start_ticks": 70003,
        "user_namespace_inode": 64000,
        "pid_namespace_inode": 70004,
        "mount_namespace_inode": 70005,
        "exe_device": 70006,
        "exe_inode": 70007,
        "exe_sha256": _sha(b"overlay-online-executable-v2"),
        "cgroup_sha256": cgroup_sha256,
        "pidfd_open_succeeded": True,
        "pidfd_alive": True,
        "procfs_read_succeeded": True,
        "observation_complete": True,
    }
    body["process_snapshot_id"] = (
        overlay._PREFIX_BY_KIND["process_snapshot"]
        + hashlib.sha256(
            overlay._CONTENT_ID_DOMAIN_BY_KIND["process_snapshot"]
            + overlay._canonical(body)
        ).hexdigest()
    )
    assert frozenset(body) == overlay.PROCESS_SNAPSHOT_FIELDS_V2
    return body


def _six_live_artifacts(
    admission_corpus: object,
    runtime_graph: object,
    custody_root: Mapping[str, object],
    live_payload: bytes,
    live_root: Mapping[str, object],
    live_body: Mapping[str, object],
    anchor_root: Mapping[str, object],
    *,
    online_private: Ed25519PrivateKey,
    custody_private: Ed25519PrivateKey,
    attestation_mutator: Callable[[dict[str, object]], None] | None = None,
) -> tuple[tuple[bytes, ...], tuple[Mapping[str, object], ...], dict[str, object]]:
    refs = _policy_refs(admission_corpus, runtime_graph)
    source, source_private = _role(admission_corpus, "SOURCE_RUNTIME_CUSTODIAN")
    pure, pure_private = _role(admission_corpus, "PURE_REPLAYER")
    online = admission_corpus.roots["online_policy"]["body"]
    custody = custody_root["body"]
    sequence = 1
    challenge_nonce = "11" * 32
    response_nonce = "22" * 32
    session_id = overlay._session_id(
        attempt_context_id=refs["attempt_context_id"],
        online_witness_key_id=online["online_key_id"],
        session_sequence=sequence,
        challenge_nonce=challenge_nonce,
    )
    common = _session_common(
        refs=refs,
        custody_id=custody_root["custody_attestation_policy_id"],
        live_policy_id=live_root["live_runtime_policy_id"],
        anchor_policy_id=anchor_root["live_runtime_anchor_policy_id"],
        session_id=session_id,
        sequence=sequence,
    )
    entry = _process_snapshot(
        phase="ENTRY",
        uid=online["service_uid"],
        gid=online["service_gid"],
        cgroup_sha256=live_body["online_witness_expected_cgroup_sha256"],
    )
    exit_snapshot = _process_snapshot(
        phase="EXIT",
        uid=online["service_uid"],
        gid=online["service_gid"],
        cgroup_sha256=live_body["online_witness_expected_cgroup_sha256"],
    )
    challenge_body: dict[str, object] = {
        "schema_version": overlay.LIVE_RUNTIME_CHALLENGE_SCHEMA_V2,
        **common,
        "challenge_nonce": challenge_nonce,
        "challenge_nonce_size_bytes": 32,
        "challenger_role": "SOURCE_RUNTIME_CUSTODIAN",
        "challenger_key_id": source["key_id"],
        "challenger_key_epoch": source["key_epoch"],
        "expected_responder_key_id": online["online_key_id"],
        "expected_responder_key_epoch": online["online_key_epoch"],
        "entry_process_snapshot": copy.deepcopy(entry),
        "entry_process_snapshot_payload_sha256": _sha(overlay._canonical(entry)),
        "expected_previous_anchor_mapping": copy.deepcopy(
            anchor_root["body"]["oob_high_water_mapping"]
        ),
        "challenge_monotonic_nanoseconds_decimal": "1000000",
        "maximum_round_trip_nanoseconds": 1_000_000,
        "one_shot_required": True,
        "challenge_outcome": "ISSUED",
    }
    challenge_payload, challenge_root = _signed_overlay_object(
        "live_runtime_challenge",
        challenge_body,
        ((
            "authority",
            source_private,
            source,
            overlay.SIGN_RUNTIME_CLOSURE_PURPOSE_V2,
        ),),
    )
    response_body: dict[str, object] = {
        "schema_version": overlay.LIVE_RUNTIME_RESPONSE_SCHEMA_V2,
        **common,
        "live_runtime_challenge_id": challenge_root["live_runtime_challenge_id"],
        "live_runtime_challenge_payload_sha256": _sha(challenge_payload),
        "challenge_nonce": challenge_nonce,
        "challenge_nonce_size_bytes": 32,
        "responder_key_id": online["online_key_id"],
        "responder_key_epoch": online["online_key_epoch"],
        "response_nonce": response_nonce,
        "response_nonce_size_bytes": 32,
        "entry_process_snapshot_id": entry["process_snapshot_id"],
        "entry_process_snapshot_payload_sha256": _sha(overlay._canonical(entry)),
        "response_outcome": "RESPONDED",
    }
    response_payload, response_root = _signed_overlay_object(
        "live_runtime_response",
        response_body,
        ((
            "external",
            online_private,
            online["online_key_id"],
            overlay.SIGN_LIVE_RUNTIME_CHALLENGE_RESPONSE_PURPOSE_V2,
        ),),
    )
    report_body: dict[str, object] = {
        "schema_version": overlay.LIVE_RUNTIME_REPORT_SCHEMA_V2,
        **common,
        "live_runtime_challenge_id": challenge_root["live_runtime_challenge_id"],
        "live_runtime_challenge_payload_sha256": _sha(challenge_payload),
        "live_runtime_response_id": response_root["live_runtime_response_id"],
        "live_runtime_response_payload_sha256": _sha(response_payload),
        "challenge_nonce": challenge_nonce,
        "response_nonce": response_nonce,
        "entry_process_snapshot": copy.deepcopy(entry),
        "entry_process_snapshot_payload_sha256": _sha(overlay._canonical(entry)),
        "exit_process_snapshot": copy.deepcopy(exit_snapshot),
        "exit_process_snapshot_payload_sha256": _sha(
            overlay._canonical(exit_snapshot)
        ),
        "entry_exit_identity_fields_in_exact_order": list(
            overlay.ENTRY_EXIT_IDENTITY_FIELDS_IN_EXACT_ORDER_V2
        ),
        "entry_exit_identity_equal": True,
        "challenge_sent_monotonic_nanoseconds_decimal": "1000000",
        "response_received_monotonic_nanoseconds_decimal": "1000100",
        "round_trip_nanoseconds": 100,
        "maximum_round_trip_nanoseconds": 1_000_000,
        "round_trip_within_policy": True,
        "exact_one_challenge_packet_sent": True,
        "exact_one_response_packet_received": True,
        "response_packet_size_bytes": len(response_payload),
        "response_message_flags": 0,
        "response_ancillary_item_count": 0,
        "message_truncation_absent": True,
        "control_truncation_absent": True,
        "unexpected_ancillary_absent": True,
        "so_peercred_verified": True,
        "pidfd_identity_verified": True,
        "procfs_identity_verified": True,
        "service_binding_verified": True,
        "runtime_closure_policy_binding_verified": True,
        "runtime_closure_evidence_binding_verified": True,
        "runtime_executable_binding_verified": True,
        "distinct_live_uid_gid_verified": True,
        "live_channel_observed": True,
        "report_outcome": "OBSERVED_ENTRY_EXIT_EQUAL",
    }
    report_payload, report_root = _signed_overlay_object(
        "live_runtime_report",
        report_body,
        ((
            "authority",
            source_private,
            source,
            overlay.SIGN_RUNTIME_CLOSURE_PURPOSE_V2,
        ),),
    )
    opaque_handle = _sha(b"opaque-online-key-handle-v2")
    attestation_body: dict[str, object] = {
        "schema_version": overlay.LIVE_RUNTIME_CUSTODY_ATTESTATION_SCHEMA_V2,
        **common,
        "live_runtime_challenge_id": challenge_root["live_runtime_challenge_id"],
        "live_runtime_challenge_payload_sha256": _sha(challenge_payload),
        "live_runtime_response_id": response_root["live_runtime_response_id"],
        "live_runtime_response_payload_sha256": _sha(response_payload),
        "live_runtime_report_id": report_root["live_runtime_report_id"],
        "live_runtime_report_payload_sha256": _sha(report_payload),
        "challenge_nonce": challenge_nonce,
        "response_nonce": response_nonce,
        "entry_process_snapshot_id": entry["process_snapshot_id"],
        "entry_process_snapshot_payload_sha256": _sha(overlay._canonical(entry)),
        "exit_process_snapshot_id": exit_snapshot["process_snapshot_id"],
        "exit_process_snapshot_payload_sha256": _sha(
            overlay._canonical(exit_snapshot)
        ),
        "attestation_sequence": sequence,
        "attestor_service_instance_id": "custody-attestor-instance-v2",
        "attestor_control_domain_id": "external-custody-control-v2",
        "attestor_key_id": custody["custody_attestor_key_id"],
        "attestor_key_epoch": custody["custody_attestor_key_epoch"],
        "attestation_purpose": (
            overlay.ATTEST_ONLINE_KEY_NONEXPORTABILITY_PURPOSE_V2
        ),
        "online_witness_key_id": online["online_key_id"],
        "online_witness_key_epoch": online["online_key_epoch"],
        "online_witness_public_key_hex": online["online_public_key_hex"],
        "opaque_key_handle_sha256": opaque_handle,
        "attestation_evidence_format": custody["attestation_evidence_format"],
        "attestation_evidence_hex": "ab" * 32,
        "certificate_chain_der_hex_in_exact_order": ["30"],
        "certificate_chain_root_id": "test-root-ca-v1",
        "endorsement_material_hex_in_exact_order": ["aa"],
        "revocation_material_hex_in_exact_order": ["bb"],
        "attested_challenge_nonce": challenge_nonce,
        "attested_report_id": report_root["live_runtime_report_id"],
        "attested_report_payload_sha256": _sha(report_payload),
        "attested_online_key_public_key_sha256": _sha(
            bytes.fromhex(online["online_public_key_hex"])
        ),
        "attested_key_handle_sha256": opaque_handle,
        "online_key_generated_in_attested_boundary": True,
        "online_key_nonexportable": True,
        "certificate_chain_verified": True,
        "quote_verified": True,
        "fresh_nonce_verified": True,
        "report_binding_verified": True,
        "online_key_binding_verified": True,
        "attestation_outcome": "ATTESTED_PRESENTED_EVIDENCE",
    }
    if attestation_mutator is not None:
        attestation_mutator(attestation_body)
    attestation_payload, attestation_root = _signed_overlay_object(
        "live_runtime_custody_attestation",
        attestation_body,
        ((
            "external",
            custody_private,
            custody["custody_attestor_key_id"],
            overlay.ATTEST_ONLINE_KEY_NONEXPORTABILITY_PURPOSE_V2,
        ),),
    )
    result_body: dict[str, object] = {
        "schema_version": overlay.LIVE_RUNTIME_RESULT_SCHEMA_V2,
        **common,
        "live_runtime_challenge_id": challenge_root["live_runtime_challenge_id"],
        "live_runtime_challenge_payload_sha256": _sha(challenge_payload),
        "live_runtime_response_id": response_root["live_runtime_response_id"],
        "live_runtime_response_payload_sha256": _sha(response_payload),
        "live_runtime_report_id": report_root["live_runtime_report_id"],
        "live_runtime_report_payload_sha256": _sha(report_payload),
        "live_runtime_custody_attestation_id": attestation_root[
            "live_runtime_custody_attestation_id"
        ],
        "live_runtime_custody_attestation_payload_sha256": _sha(
            attestation_payload
        ),
        "verified_signature_count": overlay.PRE_CAS_VERIFIED_SIGNATURE_COUNT_V2,
        "verified_signature_scope": overlay.PRE_CAS_VERIFIED_SIGNATURE_SCOPE_V2,
        "verified_protocol_object_count": (
            overlay.PRE_CAS_VERIFIED_PROTOCOL_OBJECT_COUNT_V2
        ),
        "raw_authority_graph_verified": True,
        "frozen_b480_candidate_verified": True,
        "nineteen_expected_pins_verified": True,
        "key_sets_pairwise_disjoint": True,
        "offline_static_image_graph_verified": True,
        "service_image_exists": False,
        "service_image_pulled": False,
        "service_image_executed": False,
        "runtime_closure_verified": False,
        "entry_exit_process_identity_equal": False,
        "socket_seqpacket_verified": False,
        "so_peercred_verified": False,
        "pidfd_identity_verified": False,
        "procfs_identity_verified": False,
        "actor_nonce_freshness_verified": False,
        "actor_sequence_high_water_verified": False,
        "one_shot_verified": False,
        "round_trip_freshness_verified": False,
        "key_custody_verified": False,
        "online_key_nonexportable_verified": False,
        "distinct_live_uid_gid_verified": False,
        "live_channel_verified": False,
        "external_currentness_verified": False,
        "durability_verified": False,
        "non_equivocation_verified": False,
        "global_anti_fork_verified": False,
        "malicious_root_resistance_verified": False,
        "authority_evidence": False,
        "r01_satisfied": False,
        "readiness_evidence": False,
        "execution_authorized": False,
        "presented_only": True,
        "private_key_present": False,
        "spool_path_present": False,
        "socket_present": False,
        "decoded_mapping_input_present": False,
        "callback_present": False,
        "caller_nonce_present": False,
        "caller_clock_present": False,
        "first_failure_code": "NONE",
    }
    result_payload, result_root = _signed_overlay_object(
        "live_runtime_result",
        result_body,
        (
            (
                "authority",
                source_private,
                source,
                overlay.SIGN_RUNTIME_CLOSURE_PURPOSE_V2,
            ),
            (
                "authority",
                pure_private,
                pure,
                overlay.SIGN_PURE_REPLAY_RESULT_PURPOSE_V2,
            ),
        ),
    )
    payloads = (
        live_payload,
        challenge_payload,
        response_payload,
        report_payload,
        attestation_payload,
        result_payload,
    )
    roots = (
        live_root,
        challenge_root,
        response_root,
        report_root,
        attestation_root,
        result_root,
    )
    context = {
        "lock_id": refs["stage_b_trust_lock_id"],
        "profile_id": refs["authority_profile_id"],
        "registry_id": refs["signer_registry_id"],
        "attempt_id": refs["attempt_context_id"],
        "bootstrap_id": refs["bootstrap_external_witness_policy_id"],
        "online_id": refs["online_witness_policy_id"],
        "durable_id": refs["durable_anchor_policy_id"],
        "live_policy_id": live_root["live_runtime_policy_id"],
        "anchor_policy_id": anchor_root["live_runtime_anchor_policy_id"],
        "admission_result_id": refs["admission_candidate_result_id"],
        "admission_result_sha": refs[
            "admission_candidate_result_payload_sha256"
        ],
        "session_id": session_id,
        "session_sequence": sequence,
    }
    return payloads, roots, context


def _complete_cas(
    tmp_path: Path,
    admission_corpus: object,
    payloads: tuple[bytes, ...],
    roots: tuple[Mapping[str, object], ...],
    context: Mapping[str, object],
) -> tuple[_CasEvidence, dict[str, object]]:
    result_payload = payloads[-1]
    result_root = roots[-1]
    chain_id = overlay._cas_chain_id(
        attempt_context_id=context["attempt_id"],
        session_id=context["session_id"],
        session_sequence=context["session_sequence"],
        live_runtime_result_id=result_root["live_runtime_result_id"],
        live_runtime_result_payload_sha256=_sha(result_payload),
    )
    plan = store.build_retention_plan_v1(
        authority_profile_id=context["profile_id"],
        signer_registry_id=context["registry_id"],
        attempt_context_id=context["attempt_id"],
        retention_session_id=context["session_id"],
        chain_id=chain_id,
        plan_kind=overlay.CAS_PLAN_KIND_V2,
        objects_in_exact_order=tuple(
            (artifact_kind, maximum_bytes)
            for _ordinal, artifact_kind, maximum_bytes
            in overlay.CAS_OBJECTS_IN_EXACT_ORDER_V2
        ),
    )
    call = admission_corpus.call
    _cas_row, cas_private = _role(admission_corpus, "CAS_STORE")
    id_fields = (
        "live_runtime_policy_id",
        "live_runtime_challenge_id",
        "live_runtime_response_id",
        "live_runtime_report_id",
        "live_runtime_custody_attestation_id",
        "live_runtime_result_id",
    )
    requests: list[bytes] = []
    receipts: list[bytes] = []
    checkpoints: list[bytes] = []
    heads: list[bytes] = []
    responses: list[bytes] = []
    with store.CasStoreV2(
        root_directory=str(tmp_path / "live-runtime-cas-v2"),
        retention_plan_payload=plan,
        private_key_seed=_private_seed(cas_private),
        allowed_peer_uids=frozenset({os.geteuid()}),
        stage_b_lock_payload=call["stage_b_trust_lock_payload"],
        authority_profile_payload=call["authority_profile_payload"],
        signer_registry_payload=call["signer_registry_payload"],
        out_of_band_expected_lock_id_bytes=call[
            "out_of_band_expected_lock_id_bytes"
        ],
        out_of_band_expected_lock_payload_sha256_digest=call[
            "out_of_band_expected_lock_payload_sha256_digest"
        ],
        create=True,
    ) as cas_store:
        for ordinal, (payload, root, id_field) in enumerate(
            zip(payloads, roots, id_fields, strict=True)
        ):
            request_payload = store.build_generic_store_request_v1(
                retention_plan_payload=plan,
                ordinal=ordinal,
                bound_artifact_id=root[id_field],
                bound_artifact_sha256=_store_sha(payload),
                reported_size_bytes=len(payload),
            )
            request = store.load_generic_store_request_v1(request_payload)
            object_path = tmp_path / f"live-runtime-object-{ordinal}.json"
            object_path.write_bytes(payload)
            descriptor = os.open(
                object_path,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            )
            try:
                response = cas_store.ingest_open_fd(
                    request,
                    descriptor,
                    store.PeerCredentialsV2(
                        os.getpid(), os.geteuid(), os.getegid()
                    ),
                )
            finally:
                os.close(descriptor)
            requests.append(request_payload)
            receipts.append(
                response.checkpoint_record.signed_receipt.canonical_bytes()
            )
            checkpoints.append(response.checkpoint_record.canonical_bytes())
            heads.append(response.head_record.canonical_bytes())
            responses.append(response.canonical_bytes())
        assert len(cas_store.audit_complete_session()) == 6
    evidence = _CasEvidence(
        plan=plan,
        requests=tuple(requests),
        objects=payloads,
        receipts=tuple(receipts),
        checkpoints=tuple(checkpoints),
        heads=tuple(heads),
        responses=tuple(responses),
    )
    replay_result = replay.require_verified_cas_store_complete_session_replay_core_v2(
        replay.verify_cas_store_complete_session_replay_core_v2(
            stage_b_lock_payload=call["stage_b_trust_lock_payload"],
            authority_profile_payload=call["authority_profile_payload"],
            signer_registry_payload=call["signer_registry_payload"],
            out_of_band_expected_lock_id_bytes=call[
                "out_of_band_expected_lock_id_bytes"
            ],
            out_of_band_expected_lock_payload_sha256_digest=call[
                "out_of_band_expected_lock_payload_sha256_digest"
            ],
            retention_plan_payload=evidence.plan,
            request_payloads_in_exact_order=evidence.requests,
            object_payloads_in_exact_order=evidence.objects,
            signed_receipt_payloads_in_exact_order=evidence.receipts,
            checkpoint_payloads_in_exact_order=evidence.checkpoints,
            head_payloads_in_exact_order=evidence.heads,
            response_payloads_in_exact_order=evidence.responses,
        )
    )
    final_request = _json(evidence.requests[-1])
    final_binding = replay_result.object_bindings_in_exact_order[-1]
    complete = {
        "schema_version": overlay.CAS_COMPLETE_BINDING_SCHEMA_V2,
        "authority_profile_id": replay_result.authority_profile_id,
        "signer_registry_id": replay_result.signer_registry_id,
        "attempt_context_id": replay_result.attempt_context_id,
        "session_id": context["session_id"],
        "session_sequence": context["session_sequence"],
        "retention_plan_id": replay_result.plan_id,
        "retention_plan_payload_sha256": _sha(evidence.plan),
        "retention_session_id": replay_result.retention_session_id,
        "chain_id": replay_result.chain_id,
        "plan_kind": replay_result.plan_kind,
        "expected_object_count": replay_result.expected_object_count,
        "artifact_kinds_in_exact_order": list(
            overlay.CAS_ARTIFACT_KINDS_IN_EXACT_ORDER_V2
        ),
        "final_ordinal": 5,
        "final_artifact_kind": "live_runtime_result_v2",
        "live_runtime_result_id": result_root["live_runtime_result_id"],
        "live_runtime_result_payload_sha256": _sha(result_payload),
        "final_store_request_id": final_request["request_id"],
        "final_store_request_payload_sha256": _store_sha(evidence.requests[-1]),
        "final_object_id": final_binding[4],
        "final_object_sha256": _store_sha(result_payload),
        "final_object_size_bytes": len(result_payload),
        "final_signed_receipt_payload_sha256": final_binding[6],
        "final_checkpoint_payload_sha256": (
            replay_result.complete_checkpoint_sha256
        ),
        "final_head_state_id": replay_result.complete_head_state_id,
        "final_head_payload_sha256": replay_result.complete_head_sha256,
        "final_response_payload_sha256": replay_result.complete_response_sha256,
        "observed_object_count": 6,
        "complete": True,
        "complete_head_required": True,
        "durable_before_return": True,
        "readback_verified": True,
    }
    assert frozenset(complete) == overlay.LIVE_RUNTIME_CAS_COMPLETE_BINDING_FIELDS_V2
    return evidence, complete


def _anchor_transcript(
    admission_corpus: object,
    anchor_root: Mapping[str, object],
    payloads: tuple[bytes, ...],
    roots: tuple[Mapping[str, object], ...],
    context: Mapping[str, object],
    complete: Mapping[str, object],
) -> tuple[bytes, bytes, bytes, bytes]:
    source, source_private = _role(admission_corpus, "SOURCE_RUNTIME_CUSTODIAN")
    anchor = anchor_root["body"]
    common = {
        "status": overlay.STATUS_V2,
        "claim_level": overlay.CLAIM_LEVEL_V2,
        "stage_b_trust_lock_id": context["lock_id"],
        "authority_profile_id": context["profile_id"],
        "signer_registry_id": context["registry_id"],
        "attempt_context_id": context["attempt_id"],
        "bootstrap_external_witness_policy_id": context["bootstrap_id"],
        "online_witness_policy_id": context["online_id"],
        "durable_anchor_policy_id": context["durable_id"],
        "live_runtime_policy_id": context["live_policy_id"],
        "live_runtime_anchor_policy_id": context["anchor_policy_id"],
        "admission_candidate_result_id": context["admission_result_id"],
        "admission_candidate_result_payload_sha256": context[
            "admission_result_sha"
        ],
        "session_id": context["session_id"],
        "session_sequence": context["session_sequence"],
    }
    previous = copy.deepcopy(anchor["oob_high_water_mapping"])
    next_sequence = anchor["oob_high_water_head_sequence"] + 1
    committed = overlay._committed_mapping(
        context=context,
        result_root=roots[-1],
        result_payload=payloads[-1],
        complete=complete,
        anchor_sequence=next_sequence,
    )
    request_body: dict[str, object] = {
        "schema_version": overlay.LIVE_RUNTIME_ANCHOR_REQUEST_SCHEMA_V2,
        **common,
        "live_runtime_policy_payload_sha256": _sha(payloads[0]),
        "live_runtime_challenge_id": roots[1]["live_runtime_challenge_id"],
        "live_runtime_challenge_payload_sha256": _sha(payloads[1]),
        "live_runtime_response_id": roots[2]["live_runtime_response_id"],
        "live_runtime_response_payload_sha256": _sha(payloads[2]),
        "live_runtime_report_id": roots[3]["live_runtime_report_id"],
        "live_runtime_report_payload_sha256": _sha(payloads[3]),
        "live_runtime_custody_attestation_id": roots[4][
            "live_runtime_custody_attestation_id"
        ],
        "live_runtime_custody_attestation_payload_sha256": _sha(payloads[4]),
        "live_runtime_result_id": roots[5]["live_runtime_result_id"],
        "live_runtime_result_payload_sha256": _sha(payloads[5]),
        "cas_complete_binding": copy.deepcopy(complete),
        "expected_previous_anchor_checkpoint_id": anchor[
            "oob_high_water_checkpoint_id"
        ],
        "expected_previous_anchor_checkpoint_payload_sha256": anchor[
            "oob_high_water_checkpoint_payload_sha256"
        ],
        "expected_previous_anchor_checkpoint_sequence": anchor[
            "oob_high_water_checkpoint_sequence"
        ],
        "expected_previous_anchor_head_id": anchor["oob_high_water_head_id"],
        "expected_previous_anchor_head_payload_sha256": anchor[
            "oob_high_water_head_payload_sha256"
        ],
        "expected_previous_anchor_head_sequence": anchor[
            "oob_high_water_head_sequence"
        ],
        "expected_current_mapping": copy.deepcopy(previous),
        "proposed_current_mapping": copy.deepcopy(committed),
        "compare_and_append_mode": anchor["compare_and_append_semantics"],
    }
    request_payload, request_root = _signed_overlay_object(
        "live_runtime_anchor_request",
        request_body,
        ((
            "authority",
            source_private,
            source,
            overlay.SIGN_RUNTIME_CLOSURE_PURPOSE_V2,
        ),),
    )
    anchor_key_id = anchor["anchor_key_id"]
    anchor_private = admission_corpus.durable_private_key
    checkpoint_body: dict[str, object] = {
        "schema_version": overlay.LIVE_RUNTIME_ANCHOR_CHECKPOINT_SCHEMA_V2,
        **common,
        "live_runtime_anchor_request_id": request_root[
            "live_runtime_anchor_request_id"
        ],
        "live_runtime_anchor_request_payload_sha256": _sha(request_payload),
        "previous_live_runtime_anchor_checkpoint_id": anchor[
            "oob_high_water_checkpoint_id"
        ],
        "checkpoint_sequence": next_sequence,
        "previous_mapping": copy.deepcopy(previous),
        "committed_mapping": copy.deepcopy(committed),
        "cas_complete_binding": copy.deepcopy(complete),
        "cas_outcome": "APPENDED",
    }
    checkpoint_payload, checkpoint_root = _signed_overlay_object(
        "live_runtime_anchor_checkpoint",
        checkpoint_body,
        ((
            "external",
            anchor_private,
            anchor_key_id,
            overlay.SIGN_LIVE_RUNTIME_ANCHOR_CHECKPOINT_PURPOSE_V2,
        ),),
    )
    head_body: dict[str, object] = {
        "schema_version": overlay.LIVE_RUNTIME_ANCHOR_HEAD_SCHEMA_V2,
        **common,
        "live_runtime_anchor_request_id": request_root[
            "live_runtime_anchor_request_id"
        ],
        "live_runtime_anchor_request_payload_sha256": _sha(request_payload),
        "live_runtime_anchor_checkpoint_id": checkpoint_root[
            "live_runtime_anchor_checkpoint_id"
        ],
        "live_runtime_anchor_checkpoint_payload_sha256": _sha(
            checkpoint_payload
        ),
        "previous_live_runtime_anchor_head_id": anchor[
            "oob_high_water_head_id"
        ],
        "head_sequence": next_sequence,
        "committed_mapping": copy.deepcopy(committed),
        "cas_complete_binding": copy.deepcopy(complete),
        "cas_outcome": "APPENDED",
    }
    head_payload, head_root = _signed_overlay_object(
        "live_runtime_anchor_head",
        head_body,
        ((
            "external",
            anchor_private,
            anchor_key_id,
            overlay.SIGN_LIVE_RUNTIME_ANCHOR_HEAD_PURPOSE_V2,
        ),),
    )
    receipt_body: dict[str, object] = {
        "schema_version": overlay.LIVE_RUNTIME_ANCHOR_RECEIPT_SCHEMA_V2,
        **common,
        "live_runtime_anchor_request_id": request_root[
            "live_runtime_anchor_request_id"
        ],
        "live_runtime_anchor_request_payload_sha256": _sha(request_payload),
        "live_runtime_anchor_checkpoint_id": checkpoint_root[
            "live_runtime_anchor_checkpoint_id"
        ],
        "live_runtime_anchor_checkpoint_payload_sha256": _sha(
            checkpoint_payload
        ),
        "live_runtime_anchor_head_id": head_root["live_runtime_anchor_head_id"],
        "live_runtime_anchor_head_payload_sha256": _sha(head_payload),
        "previous_mapping": copy.deepcopy(previous),
        "committed_mapping": copy.deepcopy(committed),
        "cas_complete_binding": copy.deepcopy(complete),
        "cas_outcome": "APPENDED",
        "compare_matched": True,
        "append_applied": True,
        "oob_high_water_checkpoint_id": anchor[
            "oob_high_water_checkpoint_id"
        ],
        "oob_high_water_head_id": anchor["oob_high_water_head_id"],
        "durable_before_return": True,
    }
    receipt_payload, _receipt_root = _signed_overlay_object(
        "live_runtime_anchor_receipt",
        receipt_body,
        ((
            "external",
            anchor_private,
            anchor_key_id,
            overlay.SIGN_LIVE_RUNTIME_ANCHOR_RECEIPT_PURPOSE_V2,
        ),),
    )
    return request_payload, checkpoint_payload, head_payload, receipt_payload


def _positive_overlay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    live_mutator: Callable[[dict[str, object]], None] | None = None,
    anchor_mutator: Callable[[dict[str, object]], None] | None = None,
    attestation_mutator: Callable[[dict[str, object]], None] | None = None,
) -> _OverlayCorpus:
    runtime_graph = closure_fixture._build_runtime_graph()
    unified = _unified_authority(runtime_graph)
    online_private = Ed25519PrivateKey.from_private_bytes(bytes([90]) * 32)
    custody_private = Ed25519PrivateKey.from_private_bytes(bytes([100]) * 32)
    online_key_id = admission_fixture._oid("external_online_witness_key", 3000)
    custody_key_id = admission_fixture._oid(
        "external_custody_attestor_key", 3001
    )
    base_graph = evidence_fixture._build_evidence(
        salt=0, authority_graph=unified
    )
    custody_payload, custody_root = _custody_policy(
        base_graph,
        online_private=online_private,
        custody_private=custody_private,
        online_key_id=online_key_id,
        custody_key_id=custody_key_id,
    )
    original_oid = admission_fixture._oid
    original_hex = admission_fixture._hex
    real_build_evidence = evidence_fixture._build_evidence

    def unified_build_evidence(
        *,
        salt: int = 0,
        mutators: Mapping[str, object] | None = None,
        authority_graph: object | None = None,
    ) -> object:
        return real_build_evidence(
            salt=salt,
            mutators=mutators,
            authority_graph=(unified if authority_graph is None else authority_graph),
        )

    def repinned_oid(label: str, value: int) -> str:
        if label == "external_custody_attestation_policy" and value == 3050:
            return custody_root["custody_attestation_policy_id"]
        return original_oid(label, value)

    def repinned_hex(value: int) -> str:
        if value == 3060:
            return _sha(custody_payload)
        return original_hex(value)

    monkeypatch.setattr(admission_fixture, "_build_evidence", unified_build_evidence)
    monkeypatch.setattr(admission_fixture, "_oid", repinned_oid)
    monkeypatch.setattr(admission_fixture, "_hex", repinned_hex)
    b480_root = tmp_path / "b480-positive"
    b480_root.mkdir()
    admission_corpus = admission_fixture._positive_corpus(b480_root, salt=0)
    assert admission_corpus.graph.authority.lock_id == runtime_graph.lock_id
    assert admission_corpus.roots["online_policy"]["body"][
        "custody_attestation_policy_id"
    ] == custody_root["custody_attestation_policy_id"]
    assert admission_corpus.roots["online_policy"]["body"][
        "custody_attestation_policy_payload_sha256"
    ] == _sha(custody_payload)
    live_payload, live_root, anchor_payload, anchor_root, live_body = (
        _live_and_anchor_policies(
            admission_corpus,
            runtime_graph,
            custody_payload,
            custody_root,
            live_mutator=live_mutator,
            anchor_mutator=anchor_mutator,
        )
    )
    payload_tuple, root_tuple, context = _six_live_artifacts(
        admission_corpus,
        runtime_graph,
        custody_root,
        live_payload,
        live_root,
        live_body,
        anchor_root,
        online_private=online_private,
        custody_private=custody_private,
        attestation_mutator=attestation_mutator,
    )
    cas_evidence, complete = _complete_cas(
        tmp_path, admission_corpus, payload_tuple, root_tuple, context
    )
    anchor_request, anchor_checkpoint, anchor_head, anchor_receipt = (
        _anchor_transcript(
            admission_corpus,
            anchor_root,
            payload_tuple,
            root_tuple,
            context,
            complete,
        )
    )
    runtime_args = runtime_graph.verify_arguments
    call = dict(admission_corpus.call)
    call.update(
        {
            "runtime_closure_policy_payload": runtime_graph.policy_payload,
            "runtime_closure_evidence_payload": runtime_graph.evidence_payload,
            "raw_oci_index_payload": runtime_args["raw_oci_index_payload"],
            "raw_oci_manifest_payload": runtime_args["raw_oci_manifest_payload"],
            "raw_oci_config_payload": runtime_args["raw_oci_config_payload"],
            "ordered_raw_layer_chunk_tuples": runtime_args[
                "ordered_raw_layer_chunk_tuples"
            ],
            "ordered_support_artifact_chunk_tuples": runtime_args[
                "ordered_support_artifact_chunk_tuples"
            ],
            "custody_attestation_policy_payload": custody_payload,
            "live_runtime_policy_payload": live_payload,
            "live_runtime_anchor_policy_payload": anchor_payload,
            "live_runtime_challenge_payload": payload_tuple[1],
            "live_runtime_response_payload": payload_tuple[2],
            "live_runtime_report_payload": payload_tuple[3],
            "live_runtime_custody_attestation_payload": payload_tuple[4],
            "live_runtime_result_payload": payload_tuple[5],
            "live_runtime_cas_retention_plan_payload": cas_evidence.plan,
            "live_runtime_cas_store_request_payloads_in_exact_order": (
                cas_evidence.requests
            ),
            "live_runtime_cas_object_payloads_in_exact_order": (
                cas_evidence.objects
            ),
            "live_runtime_cas_signed_receipt_payloads_in_exact_order": (
                cas_evidence.receipts
            ),
            "live_runtime_cas_checkpoint_payloads_in_exact_order": (
                cas_evidence.checkpoints
            ),
            "live_runtime_cas_head_payloads_in_exact_order": cas_evidence.heads,
            "live_runtime_cas_response_payloads_in_exact_order": (
                cas_evidence.responses
            ),
            "live_runtime_anchor_request_payload": anchor_request,
            "live_runtime_anchor_checkpoint_payload": anchor_checkpoint,
            "live_runtime_anchor_head_payload": anchor_head,
            "live_runtime_anchor_receipt_payload": anchor_receipt,
            "out_of_band_expected_runtime_closure_policy_id_bytes": (
                _json(runtime_graph.policy_payload)["id"].encode("ascii")
            ),
            "out_of_band_expected_runtime_closure_policy_payload_sha256_digest": (
                hashlib.sha256(runtime_graph.policy_payload).digest()
            ),
            "out_of_band_expected_runtime_closure_evidence_id_bytes": (
                _json(runtime_graph.evidence_payload)["id"].encode("ascii")
            ),
            "out_of_band_expected_runtime_closure_evidence_payload_sha256_digest": (
                hashlib.sha256(runtime_graph.evidence_payload).digest()
            ),
            "out_of_band_expected_custody_attestation_policy_id_bytes": (
                custody_root["custody_attestation_policy_id"].encode("ascii")
            ),
            "out_of_band_expected_custody_attestation_policy_payload_sha256_digest": (
                hashlib.sha256(custody_payload).digest()
            ),
            "out_of_band_expected_live_runtime_policy_id_bytes": (
                live_root["live_runtime_policy_id"].encode("ascii")
            ),
            "out_of_band_expected_live_runtime_policy_payload_sha256_digest": (
                hashlib.sha256(live_payload).digest()
            ),
            "out_of_band_expected_live_runtime_anchor_policy_id_bytes": (
                anchor_root["live_runtime_anchor_policy_id"].encode("ascii")
            ),
            "out_of_band_expected_live_runtime_anchor_policy_payload_sha256_digest": (
                hashlib.sha256(anchor_payload).digest()
            ),
        }
    )
    payload_map = {
        "custody_policy": custody_payload,
        "live_policy": live_payload,
        "anchor_policy": anchor_payload,
        "challenge": payload_tuple[1],
        "response": payload_tuple[2],
        "report": payload_tuple[3],
        "attestation": payload_tuple[4],
        "result": payload_tuple[5],
        "anchor_request": anchor_request,
        "anchor_checkpoint": anchor_checkpoint,
        "anchor_head": anchor_head,
        "anchor_receipt": anchor_receipt,
    }
    root_map = {
        "custody_policy": custody_root,
        "live_policy": live_root,
        "anchor_policy": anchor_root,
        "challenge": root_tuple[1],
        "response": root_tuple[2],
        "report": root_tuple[3],
        "attestation": root_tuple[4],
        "result": root_tuple[5],
    }
    return _OverlayCorpus(
        call=call,
        payloads=payload_map,
        roots=root_map,
        admission_corpus=admission_corpus,
        runtime_graph=runtime_graph,
    )


@pytest.fixture(scope="module")
def positive_corpus(tmp_path_factory: pytest.TempPathFactory) -> _OverlayCorpus:
    monkeypatch = pytest.MonkeyPatch()
    try:
        return _positive_overlay(
            tmp_path_factory.mktemp("live-runtime-overlay-v2"), monkeypatch
        )
    finally:
        monkeypatch.undo()


def test_unified_f836_b480_six_object_cas_a2_positive(
    positive_corpus: _OverlayCorpus,
) -> None:
    corpus = positive_corpus
    verified = overlay.verify_presented_live_runtime_overlay_v2(**corpus.call)
    required = overlay.require_verified_presented_live_runtime_overlay_v2(verified)
    assert required.verification_status == (
        "PRESENTED_RAW_LIVE_RUNTIME_OVERLAY_V2_VERIFIED_MECHANICS_ONLY"
    )
    assert required.raw_authority_graph_verified is True
    assert required.frozen_b480_candidate_verified is True
    assert required.six_object_cas_replay_verified is True
    assert required.presented_a2_compare_and_append_verified is True
    assert required.external_cas_verified is False
    assert required.external_anchor_verified is False
    assert required.private_key_present is False


def test_literal_content_id_domains_are_independent_oracles(
    positive_corpus: _OverlayCorpus,
) -> None:
    corpus = positive_corpus
    challenge_body = corpus.roots["challenge"]["body"]
    process_snapshot = copy.deepcopy(challenge_body["entry_process_snapshot"])
    process_snapshot_id = process_snapshot.pop("process_snapshot_id")
    cases = (
        (
            corpus.roots["live_policy"]["body"],
            corpus.roots["live_policy"]["live_runtime_policy_id"],
            "phase2b_step5_live_runtime_policy_v2_",
            b"HEGEL/PHASE2B/STEP5/LIVE_RUNTIME/POLICY/ID/V2\0",
        ),
        (
            corpus.roots["result"]["body"],
            corpus.roots["result"]["live_runtime_result_id"],
            "phase2b_step5_live_runtime_result_v2_",
            b"HEGEL/PHASE2B/STEP5/LIVE_RUNTIME/RESULT/ID/V2\0",
        ),
        (
            corpus.roots["custody_policy"]["body"],
            corpus.roots["custody_policy"]["custody_attestation_policy_id"],
            "phase2b_step5_live_runtime_custody_attestation_policy_v2_",
            (
                b"HEGEL/PHASE2B/STEP5/LIVE_RUNTIME/"
                b"CUSTODY_ATTESTATION_POLICY/ID/V2\0"
            ),
        ),
        (
            process_snapshot,
            process_snapshot_id,
            "phase2b_step5_live_runtime_process_snapshot_v2_",
            b"HEGEL/PHASE2B/STEP5/LIVE_RUNTIME/PROCESS_SNAPSHOT/ID/V2\0",
        ),
    )
    for body, observed, prefix, literal_domain in cases:
        expected = prefix + hashlib.sha256(
            literal_domain + canonicalize_rfc8785_v1(body)
        ).hexdigest()
        assert observed == expected


def test_public_signature_exact_abi_nineteen_pins_and_dependency_provenance() -> None:
    positional_names = tuple(
        """
        stage_b_trust_lock_payload authority_profile_payload signer_registry_payload
        bootstrap_external_witness_policy_payload online_witness_policy_payload
        durable_anchor_policy_payload candidate_anchor_payload candidate_receipt_payload
        candidate_ledger_payload candidate_checkpoint_payload candidate_head_payload
        q1_prepare_request_payload q1_prepare_response_payload
        a_compare_and_append_request_payload a_durable_checkpoint_payload
        a_durable_head_payload a_compare_and_append_receipt_payload
        q2_confirm_request_payload q2_confirm_response_payload admission_transcript_payload
        post_commit_logical_time_request_payloads_in_exact_order
        post_commit_signed_logical_time_token_payloads_in_exact_order
        post_commit_logical_time_checkpoint_payloads_in_exact_order
        post_commit_logical_time_head_payloads_in_exact_order
        admission_candidate_result_payload runtime_closure_policy_payload
        runtime_closure_evidence_payload raw_oci_index_payload raw_oci_manifest_payload
        raw_oci_config_payload ordered_raw_layer_chunk_tuples
        ordered_support_artifact_chunk_tuples custody_attestation_policy_payload
        live_runtime_policy_payload live_runtime_anchor_policy_payload
        live_runtime_challenge_payload live_runtime_response_payload
        live_runtime_report_payload live_runtime_custody_attestation_payload
        live_runtime_result_payload live_runtime_cas_retention_plan_payload
        live_runtime_cas_store_request_payloads_in_exact_order
        live_runtime_cas_object_payloads_in_exact_order
        live_runtime_cas_signed_receipt_payloads_in_exact_order
        live_runtime_cas_checkpoint_payloads_in_exact_order
        live_runtime_cas_head_payloads_in_exact_order
        live_runtime_cas_response_payloads_in_exact_order
        live_runtime_anchor_request_payload live_runtime_anchor_checkpoint_payload
        live_runtime_anchor_head_payload live_runtime_anchor_receipt_payload
        """.split()
    )
    signature = inspect.signature(overlay.verify_presented_live_runtime_overlay_v2)
    parameters = tuple(signature.parameters.values())
    assert len(parameters) == 70
    assert tuple(parameter.name for parameter in parameters[:51]) == positional_names
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        and parameter.default is inspect.Parameter.empty
        for parameter in parameters[:51]
    )
    assert tuple(parameter.name for parameter in parameters[51:]) == (
        overlay.EXPECTED_PIN_NAMES_IN_EXACT_ORDER_V2
    )
    assert len(overlay.EXPECTED_PIN_NAMES_IN_EXACT_ORDER_V2) == 19
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
        for parameter in parameters[51:]
    )
    assert signature.return_annotation == "VerifiedPresentedLiveRuntimeOverlayV2"
    provenance = (
        (
            authority,
            overlay.INSPECTED_AUTHORITY_V2_SOURCE_SHA256,
            "03d689327ca422f93e5a32db0a7f813fe84b1099f1bac1b31395c975d76d8e86",
        ),
        (
            admission,
            overlay.INSPECTED_ADMISSION_CONTRACT_V1_SOURCE_SHA256,
            "b480b0d4b286e1b0cfb43f88846bbbf635b753fffc64b5e89e9073264360e285",
        ),
        (
            closure,
            overlay.INSPECTED_RUNTIME_CLOSURE_V1_SOURCE_SHA256,
            "f836b3d00f332398214d3ccf7787109abb8f2de96b7b8281748bbb0ec695f2f9",
        ),
        (
            replay,
            overlay.INSPECTED_STORE_REPLAY_CORE_V2_SOURCE_SHA256,
            "e33bb8ae1f4c70668de3bfbdbd803a953513db94cb0a9c1ceb1082e97141de02",
        ),
    )
    for module, declared, frozen in provenance:
        assert declared == frozen
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == frozen


def test_each_of_nineteen_public_pins_is_mandatory(
    positive_corpus: _OverlayCorpus,
) -> None:
    for pin_name in overlay.EXPECTED_PIN_NAMES_IN_EXACT_ORDER_V2:
        call = dict(positive_corpus.call)
        original = call[pin_name]
        assert type(original) is bytes and original
        if pin_name.endswith("_digest"):
            call[pin_name] = bytes([original[0] ^ 1]) + original[1:]
        else:
            call[pin_name] = original + b"x"
        with pytest.raises(
            overlay.Phase2BStep5R01LiveRuntimeOverlayErrorV2
        ) as rejected:
            overlay.verify_presented_live_runtime_overlay_v2(**call)
        assert rejected.value.code in {
            overlay.FAIL_INPUT_V2,
            overlay.FAIL_BINDING_V2,
            overlay.FAIL_DEPENDENCY_V2,
        }


def test_result_has_no_same_session_cas_or_a2_refs_and_extra_raw_field_fails(
    positive_corpus: _OverlayCorpus,
) -> None:
    result_body = positive_corpus.roots["result"]["body"]
    forbidden = set(overlay.CAS_FORBIDDEN_IN_ALL_SIX_LIVE_BODY_FIELDS_V2) | {
        "cas_complete_binding",
        "live_runtime_anchor_request_id",
        "live_runtime_anchor_checkpoint_id",
        "live_runtime_anchor_head_id",
        "live_runtime_anchor_receipt_id",
    }
    assert forbidden.isdisjoint(result_body)
    mutated = _json(positive_corpus.payloads["result"])
    mutated["body"]["cas_complete_binding"] = {"complete": True}
    call = dict(positive_corpus.call)
    call["live_runtime_result_payload"] = canonicalize_rfc8785_v1(mutated)
    with pytest.raises(
        overlay.Phase2BStep5R01LiveRuntimeOverlayErrorV2
    ) as rejected:
        overlay.verify_presented_live_runtime_overlay_v2(**call)
    assert rejected.value.code == overlay.FAIL_SCHEMA_V2


def test_private_result_lineage_copy_mutation_claim_boundary_and_real_ni(
    positive_corpus: _OverlayCorpus,
) -> None:
    value = overlay.verify_presented_live_runtime_overlay_v2(**positive_corpus.call)
    assert overlay.require_verified_presented_live_runtime_overlay_v2(value) is value
    mechanics_true = (
        "raw_authority_graph_verified",
        "frozen_b480_candidate_verified",
        "offline_static_image_graph_verified",
        "nineteen_expected_pins_verified",
        "key_sets_pairwise_disjoint",
        "six_live_artifact_chain_verified",
        "presented_kernel_observation_report_verified",
        "presented_custody_attestation_verified",
        "six_object_cas_replay_verified",
        "presented_a2_compare_and_append_verified",
        "presented_only",
    )
    elevated_and_absent_false = (
        "private_key_present spool_path_present socket_present procfs_path_present "
        "pidfd_present decoded_mapping_input_present callback_present "
        "caller_nonce_present caller_clock_present service_image_exists "
        "service_image_pulled service_image_executed runtime_closure_signature_verified "
        "runtime_closure_verified live_runtime_closure_verified remote_attestation_verified "
        "entry_exit_process_identity_equal socket_seqpacket_verified so_peercred_verified "
        "pidfd_identity_verified procfs_identity_verified actor_nonce_freshness_verified "
        "actor_sequence_high_water_verified one_shot_verified round_trip_freshness_verified "
        "distinct_live_uid_gid_verified live_uid_gid_verified live_control_domain_verified "
        "key_custody_verified online_key_nonexportable_verified live_channel_verified "
        "external_cas_verified external_anchor_verified external_anti_rollback_verified "
        "external_currentness_verified durability_verified non_equivocation_verified "
        "global_anti_fork_verified malicious_root_resistance_verified authority_evidence "
        "r01_satisfied readiness_evidence execution_authorized"
    ).split()
    assert all(getattr(value, field) is True for field in mechanics_true)
    assert all(getattr(value, field) is False for field in elevated_and_absent_false)
    with pytest.raises(TypeError):
        overlay.VerifiedPresentedLiveRuntimeOverlayV2()
    forged = object.__new__(overlay.VerifiedPresentedLiveRuntimeOverlayV2)
    with pytest.raises(
        overlay.Phase2BStep5R01LiveRuntimeOverlayErrorV2
    ) as unissued:
        overlay.require_verified_presented_live_runtime_overlay_v2(forged)
    assert unissued.value.code == overlay.FAIL_RESULT_V2
    with pytest.raises(TypeError):
        copy.copy(value)
    with pytest.raises(TypeError):
        copy.deepcopy(value)
    object.__setattr__(value, "presented_only", False)
    with pytest.raises(
        overlay.Phase2BStep5R01LiveRuntimeOverlayErrorV2
    ) as mutated:
        overlay.require_verified_presented_live_runtime_overlay_v2(value)
    assert mutated.value.code == overlay.FAIL_RESULT_V2
    with pytest.raises(
        overlay.Phase2BStep5R01LiveRuntimeOverlayErrorV2
    ) as not_implemented:
        overlay.verify_real_live_runtime_overlay_v2()
    assert not_implemented.value.code == overlay.FAIL_NOT_IMPLEMENTED_V2


def _captured_implementation() -> FunctionType:
    cells = overlay.verify_presented_live_runtime_overlay_v2.__closure__
    assert cells is not None and len(cells) == 1
    implementation = cells[0].cell_contents
    assert type(implementation) is FunctionType
    assert implementation.__name__ == (
        "_verify_presented_live_runtime_overlay_implementation_v2"
    )
    return implementation


def test_definition_capture_ignores_evil_module_globals_and_upstream_public_bombs(
    positive_corpus: _OverlayCorpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    implementation = _captured_implementation()
    captured_cas = implementation.__globals__["_verify_cas_session"]
    assert type(captured_cas) is FunctionType
    assert implementation.__globals__["_canonical"] is not overlay._canonical
    assert captured_cas is not overlay._verify_cas_session
    kwdefaults = implementation.__kwdefaults__
    assert kwdefaults is not None
    assert kwdefaults["_verify_admission"] is _COLLECTION_ADMISSION_VERIFY
    assert kwdefaults["_require_admission"] is _COLLECTION_ADMISSION_REQUIRE
    assert kwdefaults["_verify_closure"] is _COLLECTION_CLOSURE_VERIFY
    assert kwdefaults["_require_closure"] is _COLLECTION_CLOSURE_REQUIRE
    cas_defaults = captured_cas.__kwdefaults__
    assert cas_defaults is not None
    assert cas_defaults["_verify"] is _COLLECTION_REPLAY_VERIFY
    assert cas_defaults["_require"] is _COLLECTION_REPLAY_REQUIRE

    def bomb(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("EVIL public/global replacement was reached")

    for name in (
        "_canonical",
        "_load_root",
        "_load_live_result",
        "_verify_cas_session",
        "_load_anchor_transcript",
    ):
        monkeypatch.setattr(overlay, name, bomb)
    monkeypatch.setattr(
        admission,
        "verify_external_witness_admission_transcript_candidate_v1",
        bomb,
    )
    monkeypatch.setattr(
        admission,
        "require_verified_external_witness_admission_candidate_v1",
        bomb,
    )
    monkeypatch.setattr(
        closure, "verify_offline_step5_image_closure_mechanics_v1", bomb
    )
    monkeypatch.setattr(
        closure, "require_verified_offline_step5_image_closure_mechanics_v1", bomb
    )
    monkeypatch.setattr(
        replay, "verify_cas_store_complete_session_replay_core_v2", bomb
    )
    monkeypatch.setattr(
        replay, "require_verified_cas_store_complete_session_replay_core_v2", bomb
    )
    value = overlay.verify_presented_live_runtime_overlay_v2(**positive_corpus.call)
    assert overlay.require_verified_presented_live_runtime_overlay_v2(value) is value


def test_captured_b480_f836_and_store_dependencies_all_require_private_results(
    positive_corpus: _OverlayCorpus,
) -> None:
    implementation = _captured_implementation()

    def untrusted_verify(*_args: object, **_kwargs: object) -> object:
        return object()

    def rejecting_require(_value: object) -> object:
        raise RuntimeError("dependency require sentinel")

    with pytest.raises(
        overlay.Phase2BStep5R01LiveRuntimeOverlayErrorV2
    ) as b480_rejected:
        implementation(
            **positive_corpus.call,
            _verify_admission=untrusted_verify,
            _require_admission=rejecting_require,
        )
    assert b480_rejected.value.code == overlay.FAIL_DEPENDENCY_V2
    assert "b480" in b480_rejected.value.detail

    with pytest.raises(
        overlay.Phase2BStep5R01LiveRuntimeOverlayErrorV2
    ) as f836_rejected:
        implementation(
            **positive_corpus.call,
            _verify_closure=untrusted_verify,
            _require_closure=rejecting_require,
        )
    assert f836_rejected.value.code == overlay.FAIL_DEPENDENCY_V2
    assert "closure" in f836_rejected.value.detail

    captured_cas = implementation.__globals__["_verify_cas_session"]
    result_body = positive_corpus.roots["result"]["body"]
    context = {
        "lock_id": result_body["stage_b_trust_lock_id"],
        "profile_id": result_body["authority_profile_id"],
        "registry_id": result_body["signer_registry_id"],
        "attempt_id": result_body["attempt_context_id"],
        "bootstrap_id": result_body["bootstrap_external_witness_policy_id"],
        "online_id": result_body["online_witness_policy_id"],
        "durable_id": result_body["durable_anchor_policy_id"],
        "closure_policy_id": result_body["runtime_closure_policy_id"],
        "closure_evidence_id": result_body["runtime_closure_evidence_id"],
        "custody_policy_id": result_body["custody_attestation_policy_id"],
        "live_policy_id": result_body["live_runtime_policy_id"],
        "anchor_policy_id": result_body["live_runtime_anchor_policy_id"],
        "admission_result_id": result_body["admission_candidate_result_id"],
        "admission_result_sha": result_body[
            "admission_candidate_result_payload_sha256"
        ],
        "session_id": result_body["session_id"],
        "session_sequence": result_body["session_sequence"],
    }
    live_roots = tuple(
        positive_corpus.roots[name]
        for name in (
            "live_policy",
            "challenge",
            "response",
            "report",
            "attestation",
            "result",
        )
    )
    call = positive_corpus.call
    with pytest.raises(
        overlay.Phase2BStep5R01LiveRuntimeOverlayErrorV2
    ) as store_rejected:
        captured_cas(
            stage_b_trust_lock_payload=call["stage_b_trust_lock_payload"],
            authority_profile_payload=call["authority_profile_payload"],
            signer_registry_payload=call["signer_registry_payload"],
            oob_lock_id=call["out_of_band_expected_lock_id_bytes"],
            oob_lock_digest=call[
                "out_of_band_expected_lock_payload_sha256_digest"
            ],
            retention_plan_payload=call[
                "live_runtime_cas_retention_plan_payload"
            ],
            request_payloads=call[
                "live_runtime_cas_store_request_payloads_in_exact_order"
            ],
            object_payloads=call[
                "live_runtime_cas_object_payloads_in_exact_order"
            ],
            signed_receipt_payloads=call[
                "live_runtime_cas_signed_receipt_payloads_in_exact_order"
            ],
            checkpoint_payloads=call[
                "live_runtime_cas_checkpoint_payloads_in_exact_order"
            ],
            head_payloads=call[
                "live_runtime_cas_head_payloads_in_exact_order"
            ],
            response_payloads=call[
                "live_runtime_cas_response_payloads_in_exact_order"
            ],
            context=context,
            live_roots=live_roots,
            live_payloads=call[
                "live_runtime_cas_object_payloads_in_exact_order"
            ],
            _verify=untrusted_verify,
            _require=rejecting_require,
        )
    assert store_rejected.value.code == overlay.FAIL_DEPENDENCY_V2
    assert "store replay-core" in store_rejected.value.detail


def test_legally_resigned_non_runner_closure_service_fails_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mutate_live(body: dict[str, object]) -> None:
        body["runtime_closure_service_name"] = "cas_store"

    corpus = _positive_overlay(
        tmp_path,
        monkeypatch,
        live_mutator=mutate_live,
    )
    with pytest.raises(
        overlay.Phase2BStep5R01LiveRuntimeOverlayErrorV2
    ) as rejected:
        overlay.verify_presented_live_runtime_overlay_v2(**corpus.call)
    assert rejected.value.code == overlay.FAIL_BINDING_V2
    assert "authority runner" in rejected.value.detail


@pytest.mark.parametrize(
    ("target", "field", "replacement", "detail"),
    (
        (
            "anchor",
            "anchor_namespace",
            "drifted-durable-anchor-namespace-v2",
            "key/ref binding",
        ),
        (
            "anchor",
            "admission_slot_key_profile",
            "DRIFTED_UNIQUE_ATTEMPT_AND_SEQUENCE_V2",
            "key/ref binding",
        ),
        (
            "anchor",
            "compare_and_append_semantics",
            "DRIFTED_ATOMIC_COMPARE_APPEND_V2",
            "key/ref binding",
        ),
        (
            "live",
            "anchor_slot_key_profile",
            "DRIFTED_UNIQUE_ATTEMPT_AND_SEQUENCE_V2",
            "live policy/A2 policy key binding",
        ),
        (
            "live",
            "anchor_compare_and_append_semantics",
            "DRIFTED_ATOMIC_COMPARE_APPEND_V2",
            "live policy/A2 policy key binding",
        ),
    ),
)
def test_legally_resigned_a2_namespace_slot_and_compare_drift_fail_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    field: str,
    replacement: str,
    detail: str,
) -> None:
    def mutate(body: dict[str, object]) -> None:
        body[field] = replacement

    corpus = _positive_overlay(
        tmp_path,
        monkeypatch,
        live_mutator=(mutate if target == "live" else None),
        anchor_mutator=(mutate if target == "anchor" else None),
    )
    with pytest.raises(
        overlay.Phase2BStep5R01LiveRuntimeOverlayErrorV2
    ) as rejected:
        overlay.verify_presented_live_runtime_overlay_v2(**corpus.call)
    assert rejected.value.code == overlay.FAIL_BINDING_V2
    assert detail in rejected.value.detail


@pytest.mark.parametrize(
    ("field", "item"),
    (
        ("endorsement_material_hex_in_exact_order", "NOT_HEX"),
        ("endorsement_material_hex_in_exact_order", ""),
        ("revocation_material_hex_in_exact_order", "NOT_HEX"),
        ("revocation_material_hex_in_exact_order", ""),
    ),
)
def test_legally_resigned_attestation_material_requires_nonempty_lower_hex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    item: str,
) -> None:
    def mutate_attestation(body: dict[str, object]) -> None:
        body[field] = [item]

    corpus = _positive_overlay(
        tmp_path,
        monkeypatch,
        attestation_mutator=mutate_attestation,
    )
    with pytest.raises(
        overlay.Phase2BStep5R01LiveRuntimeOverlayErrorV2
    ) as rejected:
        overlay.verify_presented_live_runtime_overlay_v2(**corpus.call)
    assert rejected.value.code == overlay.FAIL_SCHEMA_V2
    assert "hex encoding differs" in rejected.value.detail
