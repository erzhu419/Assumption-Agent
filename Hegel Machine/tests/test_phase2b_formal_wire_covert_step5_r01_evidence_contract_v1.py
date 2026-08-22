from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import importlib
import inspect
import json
from types import SimpleNamespace
from typing import Callable, Mapping

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import hegel_machine.phase2b_formal_wire_covert_step5_authority_v2 as authority
import hegel_machine.phase2b_formal_wire_covert_step5_r01_evidence_contract_v1 as contract
import hegel_machine.phase2b_formal_wire_covert_step5_owner_status_v1 as owner_status


Mutator = Callable[[dict[str, object]], None]


@dataclass(frozen=True)
class _AuthorityGraph:
    private_keys: Mapping[str, Ed25519PrivateKey]
    role_keys: list[dict[str, object]]
    service_bindings: list[dict[str, object]]
    profile_body: dict[str, object]
    profile_payload: bytes
    registry_payload: bytes
    lock_id: str
    lock_payload: bytes


@dataclass(frozen=True)
class _EvidenceGraph:
    authority: _AuthorityGraph
    witness_private_key: Ed25519PrivateKey
    attempt_id: str
    values: Mapping[str, object]
    payloads: Mapping[str, bytes]
    call: Mapping[str, object]


def _oid(label: str, value: int) -> str:
    return f"{label}_{value:064x}"


def _sha(value: int) -> str:
    return f"{value:064x}"


def _readiness() -> dict[str, object]:
    return {
        "node_states_in_canonical_topological_order": [
            {"node_id": node, "satisfied": False}
            for node in authority.READINESS_NODE_ORDER_V1
        ],
        "satisfied_node_count": 0,
        "highest_contiguous_satisfied_node": None,
        "all_nodes_satisfied": False,
        "readiness_evidence_overlay_issued": False,
    }


def _build_authority(*, salt: int = 0) -> _AuthorityGraph:
    private_keys: dict[str, Ed25519PrivateKey] = {}
    role_keys: list[dict[str, object]] = []
    for index, role in enumerate(authority.ROLE_ORDER_V2, start=1):
        private_key = Ed25519PrivateKey.from_private_bytes(bytes([index + salt]) * 32)
        private_keys[role] = private_key
        row: dict[str, object] = {
            "schema_version": authority.ROLE_KEY_SCHEMA_V1,
            "role": role,
            "control_domain": authority.ROLE_CONTROL_DOMAINS_V2[role],
            "authorized_signature_purposes": list(authority.ROLE_SIGNATURE_PURPOSES_V2[role]),
            "algorithm": "Ed25519",
            "key_epoch": 1,
            "public_key_hex": private_key.public_key().public_bytes(
                serialization.Encoding.Raw, serialization.PublicFormat.Raw
            ).hex(),
        }
        row["key_id"] = authority.phase2b_step5_role_key_id_v2(row)
        role_keys.append(row)

    service_bindings: list[dict[str, object]] = []
    for index, service in enumerate(authority.SERVICE_ORDER_V2, start=1):
        role = authority.SERVICE_ROLE_V2[service]
        role_key = role_keys[authority.ROLE_ORDER_V2.index(role)]
        uid, gid = authority.SERVICE_UID_GID_V2[service]

        def oci(offset: int) -> str:
            return "sha256:" + f"{salt * 1000 + index * 10 + offset:064x}"

        binding: dict[str, object] = {
            "schema_version": authority.SERVICE_STATIC_BINDING_SCHEMA_V2,
            "service_name": service,
            "service_type": authority.SERVICE_TYPES_V2[service],
            "binding_status": "STATIC_REQUESTED_BINDING_NOT_OBSERVED_LIVE",
            "role": role,
            "control_domain": authority.ROLE_CONTROL_DOMAINS_V2[role],
            "uid": uid,
            "gid": gid,
            "role_key_id": role_key["key_id"],
            "key_epoch": 1,
            "authorized_signature_purposes": list(authority.ROLE_SIGNATURE_PURPOSES_V2[role]),
            "oci_index_digest": oci(1),
            "oci_manifest_digest": oci(2),
            "oci_config_digest": oci(3),
            "ordered_layer_digests": [oci(4), oci(5)],
            "static_runtime_closure_policy_sha256": _sha(100 + index + salt),
        }
        binding["binding_id"] = authority.phase2b_step5_service_static_binding_id_v2(binding)
        service_bindings.append(binding)

    profile_body: dict[str, object] = {
        "schema_version": authority.AUTHORITY_PROFILE_SCHEMA_V2,
        "status": "CANDIDATE_STATIC_SERVICE_AUTHORITY_PROFILE_NOT_ADMITTED",
        "claim_level": "STATIC_CANDIDATE_FOUNDATION_NOT_IMAGE_EXISTENCE_NOT_LIVE_RUNTIME_CLOSURE_NOT_AUTHORITY_NOT_READINESS",
        "canonicalization": "RFC8785_EXACT_BYTES_NO_TRAILING_LF",
        "profile_epoch": 2,
        "owner_authorization_binding": dict(authority._EXACT_OWNER_BINDING_V2),
        "readiness_dag_binding": {
            "schema_version": "hegel-machine-phase2b-actual-unsealed-960-readiness-dag/1",
            "readiness_dag_id": authority.READINESS_DAG_ID_V1,
        },
        "readiness_state": _readiness(),
        "qv2_binding": dict(authority.EXACT_QV2_BINDING_V1),
        "source_binding": {
            "source_commit_sha": f"{salt + 1:040x}",
            "source_archive_sha256": _sha(salt + 2),
            "dependency_lock_sha256": _sha(salt + 3),
        },
        "service_static_bindings_in_exact_order": service_bindings,
        "role_policy": {
            "roles_in_exact_order": list(authority.ROLE_ORDER_V2),
            "role_count": 8,
            "distinct_public_keys": True,
            "distinct_key_ids": True,
            "distinct_control_domains": True,
            "key_epoch": 1,
            "purpose_policy_version": 2,
            "services_in_exact_order": list(authority.SERVICE_ORDER_V2),
            "service_binding_count": 5,
            "distinct_service_uids_gids": True,
            "static_service_identity_only": True,
        },
        "governance_root": role_keys[0],
        "threat_model": dict(authority._THREAT_MODEL_V2),
        "permanent_nonclaims": dict(authority._PERMANENT_NONCLAIMS_V2),
    }
    profile_id = authority.phase2b_step5_content_id_v2("authority_profile", profile_body)
    profile_payload = authority.canonical_step5_json_bytes_v2({"body": profile_body, "id": profile_id})

    registry_body: dict[str, object] = {
        "schema_version": authority.SIGNER_REGISTRY_SCHEMA_V2,
        "authority_profile_id": profile_id,
        "registry_epoch": 2,
        "status": "CANDIDATE_NOT_ADMITTED",
        "claim_level": "SIGNED_STATIC_KEY_REGISTRY_CANDIDATE_NOT_AUTHORITY_NOT_READINESS",
        "revocation_epoch": 0,
        "role_keys_in_exact_order": role_keys,
        "readiness_state": _readiness(),
    }
    registry_id = authority.phase2b_step5_content_id_v2("signer_registry", registry_body)
    registry_preimage = authority.phase2b_step5_signature_preimage_v2(
        artifact_kind="signer_registry", body=registry_body,
        role="GOVERNANCE_ANCHOR", purpose="SIGN_SIGNER_REGISTRY",
        signer_key_id=role_keys[0]["key_id"],
    )
    registry_signature = {
        "schema_version": authority.DETACHED_SIGNATURE_SCHEMA_V2,
        "algorithm": "Ed25519", "role": "GOVERNANCE_ANCHOR",
        "purpose": "SIGN_SIGNER_REGISTRY", "key_epoch": 1,
        "signer_key_id": role_keys[0]["key_id"],
        "signature_hex": private_keys["GOVERNANCE_ANCHOR"].sign(registry_preimage).hex(),
    }
    registry_payload = authority.canonical_step5_json_bytes_v2(
        {"body": registry_body, "id": registry_id, "signatures": [registry_signature]}
    )
    lock_body: dict[str, object] = {
        "schema_version": authority.STAGE_B_TRUST_LOCK_SCHEMA_V2,
        "status": "CANDIDATE_OUT_OF_BAND_STAGE_B_TRUST_LOCK",
        "claim_level": "CANDIDATE_GRAPH_PIN_ONLY_NOT_AUTHORITY_NOT_READINESS",
        "canonicalization": "RFC8785_EXACT_BYTES_NO_TRAILING_LF",
        "authority_profile_schema": authority.AUTHORITY_PROFILE_SCHEMA_V2,
        "authority_profile_id": profile_id,
        "authority_profile_payload_sha256": hashlib.sha256(profile_payload).hexdigest(),
        "signer_registry_schema": authority.SIGNER_REGISTRY_SCHEMA_V2,
        "signer_registry_id": registry_id,
        "signer_registry_payload_sha256": hashlib.sha256(registry_payload).hexdigest(),
        "qv2_binding": dict(authority.EXACT_QV2_BINDING_V1),
    }
    lock_id = authority.phase2b_step5_content_id_v2("stage_b_trust_lock", lock_body)
    lock_payload = authority.canonical_step5_json_bytes_v2({"body": lock_body, "id": lock_id})
    return _AuthorityGraph(private_keys, role_keys, service_bindings, profile_body,
                           profile_payload, registry_payload, lock_id, lock_payload)


def _authority_signature(graph: _AuthorityGraph, *, role: str, purpose: str,
                         artifact_kind: str, body: object) -> dict[str, object]:
    row = graph.role_keys[authority.ROLE_ORDER_V2.index(role)]
    preimage = contract.phase2b_step5_r01_authority_signature_preimage_v1(
        artifact_kind=artifact_kind, body=body, role=role, purpose=purpose,
        signer_key_id=row["key_id"], key_epoch=1,
    )
    return {
        "schema_version": authority.DETACHED_SIGNATURE_SCHEMA_V2,
        "algorithm": "Ed25519", "role": role, "purpose": purpose,
        "key_epoch": 1, "signer_key_id": row["key_id"],
        "signature_hex": graph.private_keys[role].sign(preimage).hex(),
    }


def _apply(mutators: Mapping[str, Mutator], name: str, value: dict[str, object]) -> None:
    if name in mutators:
        mutators[name](value)


def _build_evidence(*, salt: int = 0,
                    mutators: Mapping[str, Mutator] | None = None,
                    authority_graph: _AuthorityGraph | None = None) -> _EvidenceGraph:
    mutators = {} if mutators is None else mutators
    graph = _build_authority(salt=salt) if authority_graph is None else authority_graph
    authority_ids = (
        graph.lock_id,
        authority.phase2b_step5_content_id_v2("authority_profile", graph.profile_body),
        json.loads(graph.registry_payload)["id"],
    )
    attempt_id = _oid("attempt_context", 2000 + salt)
    values: dict[str, object] = {}
    payloads: dict[str, bytes] = {}

    response: dict[str, object] = {
        "schema_version": contract.CURRENT_CUT_RESPONSE_SCHEMA_V1,
        "status": contract.CANDIDATE_STATUS_V1,
        "stage_b_trust_lock_id": authority_ids[0], "authority_profile_id": authority_ids[1],
        "signer_registry_id": authority_ids[2], "attempt_context_id": attempt_id,
        "governance_service_binding_id": graph.service_bindings[4]["binding_id"],
        "cut_request_nonce": f"cut-{salt}", "owner_status_ledger_id": _oid("phase2b_step5_owner_status_ledger_v1", 1 + salt),
        "current_head_id": _oid("phase2b_step5_owner_status_head_v1", 2 + salt),
        "current_head_payload_sha256": "sha256:" + _sha(3 + salt), "complete_event_set_root": "sha256:" + _sha(4 + salt),
        "projection_id": _oid("phase2b_step5_owner_status_projection_v1", 5 + salt),
        "projection_payload_sha256": "sha256:" + _sha(6 + salt), "event_count": 1,
        "latest_sequence": 0, "latest_effective_status": "ACTIVE",
        "revocation_count": 0, "terminal": False,
    }
    _apply(mutators, "response", response)
    response["response_id"] = contract.phase2b_step5_r01_evidence_content_id_v1(
        "current_cut_response", response
    )
    payloads["response"] = contract.canonical_r01_evidence_contract_json_bytes_v1(response)
    values["response"] = response

    owner_binding = {
        "ledger_id": response["owner_status_ledger_id"], "current_head_id": response["current_head_id"],
        "current_head_payload_sha256": response["current_head_payload_sha256"],
        "complete_event_set_root": response["complete_event_set_root"],
        "projection_id": response["projection_id"],
        "projection_payload_sha256": response["projection_payload_sha256"],
        "event_count": response["event_count"], "latest_sequence": response["latest_sequence"],
        "latest_effective_status": response["latest_effective_status"],
        "revocation_count": response["revocation_count"], "terminal": response["terminal"],
    }
    statement: dict[str, object] = {
        "schema_version": contract.CURRENT_CUT_STATEMENT_SCHEMA_V1,
        "status": contract.CANDIDATE_STATUS_V1, "claim_level": contract.CANDIDATE_CLAIM_LEVEL_V1,
        "stage_b_trust_lock_id": authority_ids[0], "authority_profile_id": authority_ids[1],
        "signer_registry_id": authority_ids[2], "attempt_context_id": attempt_id,
        "current_cut_response_id": response["response_id"],
        "current_cut_response_payload_sha256": hashlib.sha256(payloads["response"]).hexdigest(),
        "governance_peer_observation": {
            "service_name": "governance_status", "service_binding_id": graph.service_bindings[4]["binding_id"],
            "socket_endpoint_id": _oid("socket_endpoint", 7 + salt), "request_nonce": response["cut_request_nonce"],
            "request_payload_sha256": _sha(8 + salt),
            "response_payload_sha256": hashlib.sha256(payloads["response"]).hexdigest(),
            "peer_pid": 101, "peer_uid": graph.service_bindings[4]["uid"],
            "peer_gid": graph.service_bindings[4]["gid"], "runtime_instance_id": _oid("runtime_governance", 9 + salt),
        },
        "owner_status_binding": owner_binding,
        "pre_cut_logical_time_binding": {
            "ledger_id": _oid("phase2b_step5_logical_time_ledger_v1", 10 + salt), "final_head_id": _oid("phase2b_step5_logical_time_head_v1", 11 + salt),
            "final_head_payload_sha256": "sha256:" + _sha(12 + salt), "last_token_id": _oid("phase2b_step5_logical_time_token_v1", 13 + salt),
            "next_sequence": 2,
        },
    }
    _apply(mutators, "statement", statement)
    statement["statement_id"] = contract.phase2b_step5_r01_evidence_content_id_v1(
        "current_cut_statement", statement
    )
    payloads["statement"] = contract.canonical_r01_evidence_contract_json_bytes_v1(statement)
    values["statement"] = statement

    report_values: list[dict[str, object]] = []
    report_payloads: list[bytes] = []
    for index, service in enumerate(graph.service_bindings):
        role_key = graph.role_keys[authority.ROLE_ORDER_V2.index(service["role"])]
        observation = {
            "pid": 200 + index, "process_start_ticks": 300 + index,
            "uid": service["uid"], "gid": service["gid"],
            "pid_namespace_inode": 400 + index, "user_namespace_inode": 500 + index,
            "mount_namespace_inode": 600 + index, "network_namespace_inode": 700 + index,
            "cgroup_v2_path": f"/hegel/{service['service_name']}",
            "rootfs_mount_id": f"rootfs-{index}", "executable_sha256": _sha(1000 + index),
            "argv_sha256": _sha(1010 + index), "environment_sha256": _sha(1020 + index),
            "fd_table_sha256": _sha(1030 + index), "python_import_manifest_sha256": _sha(1040 + index),
            "native_mapping_manifest_sha256": _sha(1050 + index), "closure_projection_sha256": _sha(1060 + index),
        }
        report_body: dict[str, object] = {
            "schema_version": contract.LIVE_RUNTIME_CLOSURE_REPORT_SCHEMA_V1,
            "status": contract.CANDIDATE_STATUS_V1, "claim_level": contract.CANDIDATE_CLAIM_LEVEL_V1,
            "stage_b_trust_lock_id": authority_ids[0], "authority_profile_id": authority_ids[1],
            "signer_registry_id": authority_ids[2], "attempt_context_id": attempt_id,
            "service_name": service["service_name"], "service_binding_id": service["binding_id"],
            "role": service["role"], "role_key_id": service["role_key_id"], "key_epoch": 1,
            "closure_policy_id": _oid("closure_policy", 1100 + index),
            "closure_policy_payload_sha256": service["static_runtime_closure_policy_sha256"],
            "image_closure_evidence_id": _oid("image_evidence", 1200 + index),
            "image_closure_evidence_payload_sha256": _sha(1210 + index),
            "oci_manifest_digest": service["oci_manifest_digest"],
            "runtime_instance_id": _oid("runtime_instance", 1300 + index),
            "challenge_nonce": f"challenge-{salt}-{index}",
            "entry_observation": copy.deepcopy(observation), "exit_observation": copy.deepcopy(observation),
            "live_protocol_evidence_binding": {
                "artifact_kind": f"candidate-{service['service_name']}",
                "artifact_id": _oid("protocol_artifact", 1400 + index),
                "artifact_payload_sha256": _sha(1410 + index),
                "signature_purpose": role_key["authorized_signature_purposes"][0],
                "signer_key_id": role_key["key_id"],
                "observed_runtime_instance_id": _oid("runtime_instance", 1300 + index),
            },
            "service_image_executed": False, "entry_exit_closure_equal": False,
            "uid_gid_verified": False, "control_domain_verified": False,
            "report_body_claim_boundary": contract.CANDIDATE_CLAIM_LEVEL_V1,
        }
        _apply(mutators, f"report_{index}", report_body)
        report = {
            "schema_version": contract.LIVE_RUNTIME_CLOSURE_REPORT_SCHEMA_V1,
            "body": report_body,
            "report_id": contract.phase2b_step5_r01_evidence_content_id_v1("live_runtime_closure_report", report_body),
            "signature": _authority_signature(
                graph, role="SOURCE_RUNTIME_CUSTODIAN", purpose="SIGN_RUNTIME_CLOSURE",
                artifact_kind="live_runtime_closure_report", body=report_body,
            ),
        }
        raw = contract.canonical_r01_evidence_contract_json_bytes_v1(report)
        report_values.append(report); report_payloads.append(raw)
    values["reports"] = report_values
    payloads["reports"] = tuple(report_payloads)  # type: ignore[assignment]

    bundle_body: dict[str, object] = {
        "schema_version": contract.LIVE_RUNTIME_CLOSURE_BUNDLE_SCHEMA_V1,
        "status": contract.CANDIDATE_STATUS_V1, "claim_level": contract.CANDIDATE_CLAIM_LEVEL_V1,
        "stage_b_trust_lock_id": authority_ids[0], "authority_profile_id": authority_ids[1],
        "signer_registry_id": authority_ids[2], "attempt_context_id": attempt_id,
        "service_names_in_exact_order": list(authority.SERVICE_ORDER_V2),
        "service_binding_ids_in_exact_order": [row["binding_id"] for row in graph.service_bindings],
        "report_ids_in_exact_order": [row["report_id"] for row in report_values],
        "report_payload_sha256s_in_exact_order": [hashlib.sha256(raw).hexdigest() for raw in report_payloads],
        "runtime_instance_ids_in_exact_order": [row["body"]["runtime_instance_id"] for row in report_values],
        "uids_in_exact_order": [row["body"]["entry_observation"]["uid"] for row in report_values],
        "gids_in_exact_order": [row["body"]["entry_observation"]["gid"] for row in report_values],
        "control_domains_in_exact_order": [row["control_domain"] for row in graph.service_bindings],
        "report_count": 5, "all_images_executed": False,
        "all_entry_exit_closures_equal": False, "all_uids_gids_pairwise_distinct": False,
        "all_control_domains_pairwise_distinct": False, "all_live_protocol_evidence_bound": False,
        "bundle_claim_boundary": contract.CANDIDATE_CLAIM_LEVEL_V1,
    }
    _apply(mutators, "bundle", bundle_body)
    bundle = {
        "schema_version": contract.LIVE_RUNTIME_CLOSURE_BUNDLE_SCHEMA_V1, "body": bundle_body,
        "bundle_id": contract.phase2b_step5_r01_evidence_content_id_v1("live_runtime_closure_bundle", bundle_body),
        "signature": _authority_signature(graph, role="SOURCE_RUNTIME_CUSTODIAN",
            purpose="SIGN_RUNTIME_CLOSURE", artifact_kind="live_runtime_closure_bundle", body=bundle_body),
    }
    payloads["bundle"] = contract.canonical_r01_evidence_contract_json_bytes_v1(bundle); values["bundle"] = bundle

    witness_private = Ed25519PrivateKey.from_private_bytes(bytes([80 + salt]) * 32)
    witness_public = witness_private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()
    policy: dict[str, object] = {
        "schema_version": contract.EXTERNAL_WITNESS_POLICY_SCHEMA_V1,
        "status": contract.CANDIDATE_STATUS_V1, "claim_level": contract.CANDIDATE_CLAIM_LEVEL_V1,
        "witness_namespace": f"independent-witness-{salt}",
        "witness_control_domain_id": _oid("external_control_domain", 1500 + salt),
        "witness_key_id": _oid("external_witness_key", 1501 + salt), "witness_key_epoch": 1,
        "algorithm": "Ed25519", "public_key_hex": witness_public,
    }
    policy["external_witness_policy_id"] = "phase2b_step5_external_witness_policy_v1_" + hashlib.sha256(
        b"HEGEL/PHASE2B/STEP5/EXTERNAL_WITNESS_POLICY/V1\0" +
        contract.canonical_r01_evidence_contract_json_bytes_v1(policy)
    ).hexdigest()
    _apply(mutators, "policy", policy)
    payloads["policy"] = contract.canonical_r01_evidence_contract_json_bytes_v1(policy); values["policy"] = policy

    anchor_body: dict[str, object] = {
        "schema_version": contract.SIGNED_OWNER_STATUS_ANTI_FORK_ANCHOR_SCHEMA_V1,
        "status": contract.CANDIDATE_STATUS_V1, "claim_level": contract.CANDIDATE_CLAIM_LEVEL_V1,
        "stage_b_trust_lock_id": authority_ids[0], "authority_profile_id": authority_ids[1],
        "signer_registry_id": authority_ids[2], "attempt_context_id": attempt_id,
        "current_cut_statement_id": statement["statement_id"],
        "current_cut_statement_payload_sha256": hashlib.sha256(payloads["statement"]).hexdigest(),
        "current_cut_token_id": _oid("phase2b_step5_logical_time_token_v1", 1600 + salt),
        "current_cut_token_payload_sha256": "sha256:" + _sha(1601 + salt),
        "current_cut_checkpoint_id": _oid("phase2b_step5_logical_time_checkpoint_v1", 1602 + salt),
        "current_cut_checkpoint_payload_sha256": "sha256:" + _sha(1603 + salt),
        "current_cut_head_id": _oid("phase2b_step5_logical_time_head_v1", 1604 + salt),
        "current_cut_head_payload_sha256": "sha256:" + _sha(1605 + salt),
        "owner_status_ledger_id": owner_binding["ledger_id"], "owner_current_head_id": owner_binding["current_head_id"],
        "complete_event_set_root": owner_binding["complete_event_set_root"], "projection_id": owner_binding["projection_id"],
        "anchor_sequence": 1, "previous_external_anchor_receipt_id": None,
        "external_witness_policy_id": policy["external_witness_policy_id"], "anchor_nonce": f"anchor-{salt}",
    }
    _apply(mutators, "anchor", anchor_body)
    anchor = {
        "schema_version": contract.SIGNED_OWNER_STATUS_ANTI_FORK_ANCHOR_SCHEMA_V1, "body": anchor_body,
        "anchor_id": contract.phase2b_step5_r01_evidence_content_id_v1("signed_owner_status_anti_fork_anchor", anchor_body),
        "signature": _authority_signature(graph, role="PURE_REPLAYER",
            purpose="SIGN_OWNER_STATUS_ANTI_FORK_ANCHOR", artifact_kind="owner_status_anti_fork_anchor", body=anchor_body),
    }
    payloads["anchor"] = contract.canonical_r01_evidence_contract_json_bytes_v1(anchor); values["anchor"] = anchor

    receipt: dict[str, object] = {
        "schema_version": contract.EXTERNAL_OWNER_STATUS_ANCHOR_RECEIPT_SCHEMA_V1,
        "external_witness_policy_id": policy["external_witness_policy_id"],
        "witness_namespace": policy["witness_namespace"], "witness_control_domain_id": policy["witness_control_domain_id"],
        "witness_key_id": policy["witness_key_id"], "witness_key_epoch": 1,
        "anchor_id": anchor["anchor_id"], "anchor_payload_sha256": hashlib.sha256(payloads["anchor"]).hexdigest(),
        "witness_sequence": anchor_body["anchor_sequence"],
        "previous_external_receipt_id": anchor_body["previous_external_anchor_receipt_id"],
        "leaf_index": 0, "tree_size": 1,
        "signed_tree_root_sha256": hashlib.sha256(
            b"HEGEL/PHASE2B/STEP5/EXTERNAL_WITNESS/MERKLE_LEAF/V1\0" + payloads["anchor"]
        ).hexdigest(),
        "inclusion_path_sha256_in_exact_order": [], "algorithm": "Ed25519",
    }
    _apply(mutators, "external_receipt", receipt)
    receipt["receipt_id"] = contract.phase2b_step5_r01_evidence_content_id_v1(
        "external_owner_status_anchor_receipt", dict(receipt)
    )
    receipt["signature_hex"] = witness_private.sign(
        contract.phase2b_step5_external_anchor_receipt_signature_preimage_v1(receipt)
    ).hex()
    payloads["external_receipt"] = contract.canonical_r01_evidence_contract_json_bytes_v1(receipt); values["external_receipt"] = receipt

    receipt_body: dict[str, object] = {
        "schema_version": contract.R01_RECEIPT_BODY_SCHEMA_V1, "status": contract.CANDIDATE_STATUS_V1,
        "claim_level": contract.CANDIDATE_CLAIM_LEVEL_V1,
        "stage_b_trust_lock_id": authority_ids[0], "authority_profile_id": authority_ids[1],
        "signer_registry_id": authority_ids[2], "readiness_dag_id": authority.READINESS_DAG_ID_V1,
        "attempt_context_id": attempt_id, "owner_authorization_binding": copy.deepcopy(graph.profile_body["owner_authorization_binding"]),
        "owner_status_binding": copy.deepcopy(owner_binding),
        "current_cut_binding": {
            "current_cut_response_id": response["response_id"],
            "current_cut_response_payload_sha256": hashlib.sha256(payloads["response"]).hexdigest(),
            "current_cut_statement_id": statement["statement_id"],
            "current_cut_statement_payload_sha256": hashlib.sha256(payloads["statement"]).hexdigest(),
            "current_cut_token_id": anchor_body["current_cut_token_id"],
            "current_cut_token_payload_sha256": anchor_body["current_cut_token_payload_sha256"],
            "current_cut_checkpoint_id": anchor_body["current_cut_checkpoint_id"],
            "current_cut_checkpoint_payload_sha256": anchor_body["current_cut_checkpoint_payload_sha256"],
            "current_cut_head_id": anchor_body["current_cut_head_id"],
            "current_cut_head_payload_sha256": anchor_body["current_cut_head_payload_sha256"],
        },
        "anti_fork_binding": {
            "anchor_id": anchor["anchor_id"], "anchor_payload_sha256": hashlib.sha256(payloads["anchor"]).hexdigest(),
            "external_witness_policy_id": policy["external_witness_policy_id"],
            "external_anchor_receipt_id": receipt["receipt_id"],
            "external_anchor_receipt_payload_sha256": hashlib.sha256(payloads["external_receipt"]).hexdigest(),
            "witness_sequence": receipt["witness_sequence"], "signed_tree_root_sha256": receipt["signed_tree_root_sha256"],
        },
        "runtime_closure_bundle_binding": {
            "bundle_id": bundle["bundle_id"], "bundle_payload_sha256": hashlib.sha256(payloads["bundle"]).hexdigest(),
            "report_ids_in_exact_order": bundle_body["report_ids_in_exact_order"],
            "report_payload_sha256s_in_exact_order": bundle_body["report_payload_sha256s_in_exact_order"],
        },
        "cas_complete_session_binding": {
            "session_id": _oid("cas_session", 1700 + salt), "publication_plan_id": _oid("cas_plan", 1701 + salt),
            "publication_plan_payload_sha256": "sha256:" + _sha(1702 + salt),
            "request_ids_in_exact_order": [_oid("cas_request", 1703 + salt + index) for index in range(29)],
            "receipt_ids_in_exact_order": [_oid("cas_receipt", 1803 + salt + index) for index in range(29)],
            "checkpoint_id": _oid("cas_checkpoint", 1705 + salt), "checkpoint_payload_sha256": "sha256:" + _sha(1706 + salt),
            "head_id": _oid("cas_head", 1707 + salt), "head_payload_sha256": "sha256:" + _sha(1708 + salt),
            "response_id": _oid("cas_response", 1709 + salt), "response_payload_sha256": "sha256:" + _sha(1710 + salt),
            "object_count": 29,
        },
        "service_key_custody_evidence_in_exact_order": [
            {
                "service_name": service["service_name"], "service_binding_id": service["binding_id"],
                "role": service["role"], "role_key_id": service["role_key_id"], "key_epoch": 1,
                "runtime_instance_id": report["body"]["runtime_instance_id"],
                "custody_artifact_id": _oid("custody_artifact", 1800 + index),
                "custody_artifact_payload_sha256": _sha(1810 + index), "custody_verified": False,
            }
            for index, (service, report) in enumerate(zip(graph.service_bindings, report_values, strict=True))
        ],
        "readiness_effect": {name: False for name in contract.RECEIPT_READINESS_EFFECT_FIELDS_V1},
        "r01_verdict": {name: False for name in contract.R01_VERDICT_FIELDS_V1},
        "overlay_retention_plan_id": _oid("overlay_retention_plan", 1900 + salt),
        "receipt_nonce": f"receipt-{salt}",
    }
    _apply(mutators, "receipt_body", receipt_body)
    receipt_body["receipt_body_id"] = contract.phase2b_step5_r01_evidence_content_id_v1("r01_receipt_body", receipt_body)
    payloads["receipt_body"] = contract.canonical_r01_evidence_contract_json_bytes_v1(receipt_body); values["receipt_body"] = receipt_body

    custodian = _authority_signature(graph, role="SOURCE_RUNTIME_CUSTODIAN",
        purpose="SIGN_R01_AUTHORIZATION_STATUS_RECEIPT", artifact_kind="r01_authorization_status_receipt", body=receipt_body)
    countersign_subject = contract.phase2b_step5_r01_pure_countersign_subject_v1(
        receipt_body_id=receipt_body["receipt_body_id"], receipt_body_payload=payloads["receipt_body"],
        custodian_signature=custodian,
    )
    dual: dict[str, object] = {
        "schema_version": contract.DUAL_SIGNED_R01_RECEIPT_SCHEMA_V1, "body": receipt_body,
        "receipt_body_id": receipt_body["receipt_body_id"], "custodian_signature": custodian,
        "pure_countersignature": _authority_signature(graph, role="PURE_REPLAYER",
            purpose="COUNTERSIGN_R01_AUTHORIZATION_STATUS_RECEIPT",
            artifact_kind="r01_authorization_status_receipt_countersignature", body=countersign_subject),
    }
    _apply(mutators, "dual", dual)
    dual["receipt_id"] = contract.phase2b_step5_r01_evidence_content_id_v1("dual_signed_r01_receipt", dual)
    payloads["dual"] = contract.canonical_r01_evidence_contract_json_bytes_v1(dual); values["dual"] = dual

    overlay: dict[str, object] = {
        "schema_version": contract.R01_READINESS_OVERLAY_SCHEMA_V1,
        "status": contract.CANDIDATE_STATUS_V1, "claim_level": contract.CANDIDATE_CLAIM_LEVEL_V1,
        "stage_b_trust_lock_id": authority_ids[0], "authority_profile_id": authority_ids[1],
        "signer_registry_id": authority_ids[2], "attempt_context_id": attempt_id,
        "readiness_effect": {
            "readiness_dag_id": authority.READINESS_DAG_ID_V1,
            "base_readiness_candidate_id": _oid("base_readiness_candidate", 2001 + salt),
            "overlay_sequence": 0, "previous_overlay_id": None, "transitioned_node_id": None,
            "node_states_in_canonical_topological_order": [
                {"node_id": node, "satisfied": False} for node in authority.READINESS_NODE_ORDER_V1
            ],
            "satisfied_node_count": 0, "highest_contiguous_satisfied_node": None,
            "all_nodes_satisfied": False, "r01_satisfied": False,
            "r02_or_later_satisfied": False, "execution_authorized": False,
        },
        "r01_receipt_id": dual["receipt_id"], "r01_receipt_payload_sha256": hashlib.sha256(payloads["dual"]).hexdigest(),
        "overlay_retention_plan_id": receipt_body["overlay_retention_plan_id"],
    }
    _apply(mutators, "overlay", overlay)
    overlay["overlay_id"] = contract.phase2b_step5_r01_evidence_content_id_v1("r01_readiness_overlay", overlay)
    payloads["overlay"] = contract.canonical_r01_evidence_contract_json_bytes_v1(overlay); values["overlay"] = overlay

    call: dict[str, object] = {
        "stage_b_trust_lock_payload": graph.lock_payload,
        "authority_profile_payload": graph.profile_payload,
        "signer_registry_payload": graph.registry_payload,
        "current_cut_response_payload": payloads["response"],
        "current_cut_statement_payload": payloads["statement"],
        "live_runtime_closure_report_payloads_in_exact_order": tuple(report_payloads),
        "live_runtime_closure_bundle_payload": payloads["bundle"],
        "signed_owner_status_anti_fork_anchor_payload": payloads["anchor"],
        "external_owner_status_anchor_receipt_payload": payloads["external_receipt"],
        "r01_receipt_body_payload": payloads["receipt_body"],
        "dual_signed_r01_receipt_payload": payloads["dual"],
        "r01_readiness_overlay_payload": payloads["overlay"],
        "external_witness_policy_payload": payloads["policy"],
        "out_of_band_expected_lock_id_bytes": graph.lock_id.encode("ascii"),
        "out_of_band_expected_lock_payload_sha256_digest": hashlib.sha256(graph.lock_payload).digest(),
        "out_of_band_expected_external_witness_policy_id_bytes": policy["external_witness_policy_id"].encode("ascii"),
        "out_of_band_expected_external_witness_policy_payload_sha256_digest": hashlib.sha256(payloads["policy"]).digest(),
        "expected_attempt_context_id": attempt_id,
    }
    return _EvidenceGraph(graph, witness_private, attempt_id, values, payloads, call)


def _verify(graph: _EvidenceGraph):
    return contract.verify_r01_evidence_contract_candidate_v1(**graph.call)


def _policy_epoch_mutator(epoch: object) -> Mutator:
    def mutate(policy: dict[str, object]) -> None:
        policy["witness_key_epoch"] = epoch
        policy.pop("external_witness_policy_id", None)
        policy["external_witness_policy_id"] = (
            "phase2b_step5_external_witness_policy_v1_"
            + hashlib.sha256(
                b"HEGEL/PHASE2B/STEP5/EXTERNAL_WITNESS_POLICY/V1\0"
                + contract.canonical_r01_evidence_contract_json_bytes_v1(policy)
            ).hexdigest()
        )

    return mutate


def _assert_evidence_error(code: str, operation: Callable[[], object]) -> None:
    with pytest.raises(contract.Phase2BStep5R01EvidenceContractErrorV1) as caught:
        operation()
    assert caught.value.code == code


def test_complete_raw_authority_and_nine_object_candidate_dag_is_mechanics_only() -> None:
    graph = _build_evidence()
    result = _verify(graph)
    assert contract.require_verified_r01_evidence_contract_candidate_v1(result) is result
    assert result.candidate_signature_mechanics_verified is True
    for field in (
        "external_currentness_verified", "live_runtime_closure_verified",
        "external_anti_fork_anchor_verified", "durable_cas_verified",
        "authority_established", "r01_satisfied", "readiness_overlay_admitted",
        "execution_authorized",
    ):
        assert getattr(result, field) is False
    assert len(result.artifact_ids["live_runtime_closure_report_ids_in_exact_order"]) == 5
    assert result.artifact_ids["external_witness_policy_id"] == graph.values["policy"]["external_witness_policy_id"]
    assert result.payload_sha256s["stage_b_trust_lock_payload_sha256"] == hashlib.sha256(graph.authority.lock_payload).hexdigest()
    assert result.payload_sha256s["external_witness_policy_payload_sha256"] == hashlib.sha256(graph.payloads["policy"]).hexdigest()


def test_real_owner_projection_active_sequence_and_sha256_addresses_are_accepted() -> None:
    graph = _build_authority()
    profile_id = authority.phase2b_step5_content_id_v2("authority_profile", graph.profile_body)
    registry_id = json.loads(graph.registry_payload)["id"]
    attempt_id = _oid("attempt_context", 2000)
    governance_key_id = graph.role_keys[0]["key_id"]
    common = owner_status._validate_common_bindings(
        authority_profile_id=profile_id, signer_registry_id=registry_id,
        governance_key_id=governance_key_id, governance_key_epoch=1,
        attempt_context_id=attempt_id,
    )

    head = SimpleNamespace(
        ledger_id=owner_status._ledger_id_from_common(common),
        authority_profile_id=profile_id, signer_registry_id=registry_id,
        governance_key_id=governance_key_id, governance_key_epoch=1,
        attempt_context_id=attempt_id, event_count=1,
        latest_effective_status="ACTIVE", revocation_count=0,
        last_event_id=_oid("phase2b_step5_owner_status_event_v1", 1),
        last_checkpoint_id=_oid("phase2b_step5_owner_status_checkpoint_v1", 2),
        head_id=_oid("phase2b_step5_owner_status_head_v1", 3),
        complete_event_set_root="sha256:" + _sha(4),
    )
    projection = owner_status._projection_from_head(head)
    projection_payload = projection.canonical_bytes()

    def from_real_projection(response: dict[str, object]) -> None:
        response.update({
            "owner_status_ledger_id": projection.ledger_id,
            "current_head_id": projection.head_id,
            "current_head_payload_sha256": "sha256:" + _sha(5),
            "complete_event_set_root": projection.complete_event_set_root,
            "projection_id": projection.projection_id,
            "projection_payload_sha256": "sha256:" + hashlib.sha256(projection_payload).hexdigest(),
            "event_count": projection.event_count,
            "latest_sequence": projection.latest_sequence,
            "latest_effective_status": projection.latest_effective_status,
            "revocation_count": projection.revocation_count,
            "terminal": False,
        })

    evidence = _build_evidence(authority_graph=graph, mutators={"response": from_real_projection})
    assert _verify(evidence).r01_satisfied is False


def test_revoked_owner_projection_shape_is_not_a_candidate_r01_current_cut() -> None:
    def revoked(response: dict[str, object]) -> None:
        response.update({
            "event_count": 2, "latest_sequence": 1,
            "latest_effective_status": "REVOKED", "revocation_count": 1,
            "terminal": True,
        })

    graph = _build_evidence(mutators={"response": revoked})
    with pytest.raises(contract.Phase2BStep5R01EvidenceContractErrorV1) as caught:
        _verify(graph)
    assert caught.value.code == contract.FAIL_CLAIM_BOUNDARY_V1


def test_real_admission_and_sequence_one_overlay_are_not_implemented() -> None:
    for function in (contract.verify_real_r01_admission_v1, contract.issue_admitted_r01_only_overlay_v1):
        with pytest.raises(contract.Phase2BStep5R01EvidenceContractErrorV1) as caught:
            function()
        assert caught.value.code == contract.FAIL_NOT_IMPLEMENTED_V1


@pytest.mark.parametrize("pin", ["lock-id", "lock-sha", "policy-id", "policy-sha"])
def test_both_authority_and_independent_witness_policy_require_exact_oob_bytes(pin: str) -> None:
    graph = _build_evidence()
    call = dict(graph.call)
    key = {
        "lock-id": "out_of_band_expected_lock_id_bytes",
        "lock-sha": "out_of_band_expected_lock_payload_sha256_digest",
        "policy-id": "out_of_band_expected_external_witness_policy_id_bytes",
        "policy-sha": "out_of_band_expected_external_witness_policy_payload_sha256_digest",
    }[pin]
    call[key] = "not-bytes"
    with pytest.raises((contract.Phase2BStep5R01EvidenceContractErrorV1, authority.Phase2BStep5AuthorityErrorV2)):
        contract.verify_r01_evidence_contract_candidate_v1(**call)


@pytest.mark.parametrize("collision", ["public-key", "key-id", "control-domain", "namespace"])
def test_external_witness_is_disjoint_from_all_eight_authority_roles(collision: str) -> None:
    base = _build_evidence()
    role = base.authority.role_keys[-1]

    def mutate(policy: dict[str, object]) -> None:
        if collision == "public-key": policy["public_key_hex"] = role["public_key_hex"]
        elif collision == "key-id": policy["witness_key_id"] = role["key_id"]
        elif collision == "control-domain": policy["witness_control_domain_id"] = role["control_domain"]
        else: policy["witness_namespace"] = role["role"]

    graph = _build_evidence(mutators={"policy": mutate})
    with pytest.raises(contract.Phase2BStep5R01EvidenceContractErrorV1):
        _verify(graph)


@pytest.mark.parametrize(
    ("name", "mutator"),
    [
        ("missing", lambda x: x.pop("terminal")),
        ("extra", lambda x: x.__setitem__("evil", False)),
        ("bool-int", lambda x: x.__setitem__("event_count", True)),
        ("float", lambda x: x.__setitem__("latest_sequence", 0.5)),
        ("true", lambda x: x.__setitem__("terminal", True)),
    ],
)
def test_response_exact_schema_types_and_false_claims(name: str, mutator: Mutator) -> None:
    graph = _build_evidence()
    response = copy.deepcopy(graph.values["response"])
    mutator(response)
    response.pop("response_id", None)
    response["response_id"] = contract.phase2b_step5_r01_evidence_content_id_v1(
        "current_cut_response", response
    )
    call = dict(graph.call)
    call["current_cut_response_payload"] = contract.canonical_r01_evidence_contract_json_bytes_v1(response)
    with pytest.raises(contract.Phase2BStep5R01EvidenceContractErrorV1):
        contract.verify_r01_evidence_contract_candidate_v1(**call)


def _target(root: object, path: tuple[object, ...]) -> dict[str, object]:
    value = root
    for part in path:
        value = value[part]  # type: ignore[index]
    assert type(value) is dict
    return value


@pytest.mark.parametrize(
    ("artifact", "path"),
    [
        ("response", ()), ("statement", ()),
        ("statement", ("governance_peer_observation",)),
        ("statement", ("owner_status_binding",)),
        ("statement", ("pre_cut_logical_time_binding",)),
        ("report", ("body",)), ("report", ("body", "entry_observation")),
        ("report", ("body", "exit_observation")),
        ("report", ("body", "live_protocol_evidence_binding")),
        ("bundle", ("body",)), ("anchor", ("body",)),
        ("external_receipt", ()), ("policy", ()),
        ("receipt_body", ()), ("receipt_body", ("owner_authorization_binding",)),
        ("receipt_body", ("owner_status_binding",)),
        ("receipt_body", ("current_cut_binding",)),
        ("receipt_body", ("anti_fork_binding",)),
        ("receipt_body", ("runtime_closure_bundle_binding",)),
        ("receipt_body", ("cas_complete_session_binding",)),
        ("receipt_body", ("service_key_custody_evidence_in_exact_order", 0)),
        ("receipt_body", ("readiness_effect",)),
        ("receipt_body", ("r01_verdict",)),
        ("dual", ()), ("dual", ("custodian_signature",)),
        ("dual", ("pure_countersignature",)),
        ("overlay", ()), ("overlay", ("readiness_effect",)),
        ("overlay", ("readiness_effect", "node_states_in_canonical_topological_order", 0)),
    ],
)
@pytest.mark.parametrize("operation", ["missing", "extra"])
def test_every_schema_and_nested_keyset_is_exact(
    artifact: str, path: tuple[object, ...], operation: str
) -> None:
    graph = _build_evidence()
    if artifact == "report":
        value = copy.deepcopy(graph.values["reports"][0])
    else:
        value = copy.deepcopy(graph.values[artifact])
    target = _target(value, path)
    if operation == "missing":
        target.pop(next(iter(target)))
    else:
        target["evil_extra_field"] = False
    raw = contract.canonical_r01_evidence_contract_json_bytes_v1(value)
    call = dict(graph.call)
    if artifact == "report":
        reports = list(call["live_runtime_closure_report_payloads_in_exact_order"])
        reports[0] = raw
        call["live_runtime_closure_report_payloads_in_exact_order"] = tuple(reports)
    else:
        call_key = {
            "response": "current_cut_response_payload",
            "statement": "current_cut_statement_payload",
            "bundle": "live_runtime_closure_bundle_payload",
            "anchor": "signed_owner_status_anti_fork_anchor_payload",
            "external_receipt": "external_owner_status_anchor_receipt_payload",
            "policy": "external_witness_policy_payload",
            "receipt_body": "r01_receipt_body_payload",
            "dual": "dual_signed_r01_receipt_payload",
            "overlay": "r01_readiness_overlay_payload",
        }[artifact]
        call[call_key] = raw
        if artifact == "policy":
            call["out_of_band_expected_external_witness_policy_payload_sha256_digest"] = hashlib.sha256(raw).digest()
    with pytest.raises(contract.Phase2BStep5R01EvidenceContractErrorV1):
        contract.verify_r01_evidence_contract_candidate_v1(**call)


@pytest.mark.parametrize(
    ("field", "evil"),
    [
        ("closure_policy_id", {"id": "evil"}),
        ("image_closure_evidence_id", False),
        ("image_closure_evidence_payload_sha256", ["00" * 32]),
        ("runtime_instance_id", {"runtime": "evil"}),
        ("challenge_nonce", True),
    ],
)
def test_fully_readdressed_and_resigned_report_rejects_untyped_fields(field: str, evil: object) -> None:
    graph = _build_evidence(mutators={"report_0": lambda body: body.__setitem__(field, evil)})
    with pytest.raises(contract.Phase2BStep5R01EvidenceContractErrorV1):
        _verify(graph)


@pytest.mark.parametrize("field", ["signature_purpose", "signer_key_id"])
def test_live_protocol_binding_requires_the_service_role_key_and_purpose(field: str) -> None:
    def mutate(body: dict[str, object]) -> None:
        body["live_protocol_evidence_binding"][field] = (
            "UNAUTHORIZED_PURPOSE" if field == "signature_purpose" else _oid("wrong_key", 1)
        )
    graph = _build_evidence(mutators={"report_0": mutate})
    with pytest.raises(contract.Phase2BStep5R01EvidenceContractErrorV1):
        _verify(graph)


def test_exact_string_types_and_eqall_objects_are_rejected() -> None:
    class StringSubclass(str):
        pass

    class EqAll:
        def __hash__(self) -> int:
            return hash("current_cut_response")

        def __eq__(self, other: object) -> bool:
            return True

    with pytest.raises(contract.Phase2BStep5R01EvidenceContractErrorV1):
        contract.phase2b_step5_r01_pure_countersign_subject_v1(
            receipt_body_id=StringSubclass(_oid("phase2b_step5_r01_receipt_body_v1", 1)),
            receipt_body_payload=b"{}", custodian_signature={},
        )
    with pytest.raises(contract.Phase2BStep5R01EvidenceContractErrorV1):
        contract.canonical_r01_evidence_contract_json_bytes_v1({"x": EqAll()})
    with pytest.raises(contract.Phase2BStep5R01EvidenceContractErrorV1):
        contract.canonical_r01_evidence_contract_json_bytes_v1(
            {"x": StringSubclass("x")}
        )
    for hostile_kind in (
        StringSubclass("current_cut_response"),
        EqAll(),
    ):
        _assert_evidence_error(
            contract.FAIL_INPUT_V1,
            lambda hostile_kind=hostile_kind: (
                contract.phase2b_step5_r01_evidence_content_id_v1(
                    hostile_kind, {}
                )
            ),
        )


def test_noncanonical_duplicate_and_wrong_id_are_rejected() -> None:
    graph = _build_evidence()
    for raw in (
        graph.payloads["response"] + b"\n",
        graph.payloads["response"].replace(b'"event_count":1', b'"event_count":1,"event_count":1'),
        graph.payloads["response"].replace(b'"event_count":1', b'"event_count":1.0'),
    ):
        call = dict(graph.call); call["current_cut_response_payload"] = raw
        with pytest.raises(contract.Phase2BStep5R01EvidenceContractErrorV1):
            contract.verify_r01_evidence_contract_candidate_v1(**call)
    parsed = json.loads(graph.payloads["response"]); parsed["response_id"] = _oid("wrong", 1)
    call = dict(graph.call); call["current_cut_response_payload"] = contract.canonical_r01_evidence_contract_json_bytes_v1(parsed)
    with pytest.raises(contract.Phase2BStep5R01EvidenceContractErrorV1):
        contract.verify_r01_evidence_contract_candidate_v1(**call)


def test_report_order_is_exactly_the_five_authority_service_order() -> None:
    graph = _build_evidence()
    call = dict(graph.call)
    reports = list(call["live_runtime_closure_report_payloads_in_exact_order"])
    reports[0], reports[1] = reports[1], reports[0]
    call["live_runtime_closure_report_payloads_in_exact_order"] = tuple(reports)
    with pytest.raises(contract.Phase2BStep5R01EvidenceContractErrorV1):
        contract.verify_r01_evidence_contract_candidate_v1(**call)


@pytest.mark.parametrize("field", ["last_token_id", "final_head_id", "final_head_payload_sha256"])
def test_current_cut_logical_time_cannot_reuse_pre_cut_state(field: str) -> None:
    def mutate(anchor: dict[str, object]) -> None:
        statement = {
            "last_token_id": _oid("phase2b_step5_logical_time_token_v1", 13),
            "final_head_id": _oid("phase2b_step5_logical_time_head_v1", 11),
            "final_head_payload_sha256": "sha256:" + _sha(12),
        }
        target = {
            "last_token_id": "current_cut_token_id",
            "final_head_id": "current_cut_head_id",
            "final_head_payload_sha256": "current_cut_head_payload_sha256",
        }[field]
        anchor[target] = statement[field]

    graph = _build_evidence(mutators={"anchor": mutate})
    with pytest.raises(contract.Phase2BStep5R01EvidenceContractErrorV1):
        _verify(graph)


@pytest.mark.parametrize("artifact", ["statement", "report_0", "bundle", "anchor", "receipt_body", "overlay"])
def test_every_authority_or_attempt_splice_is_rejected(artifact: str) -> None:
    def mutate(value: dict[str, object]) -> None:
        value["attempt_context_id"] = _oid("other_attempt", 999)
    graph = _build_evidence(mutators={artifact: mutate})
    with pytest.raises(contract.Phase2BStep5R01EvidenceContractErrorV1):
        _verify(graph)


@pytest.mark.parametrize(
    "mutators",
    [
        {"report_1": lambda x: (
            x.__setitem__("runtime_instance_id", _oid("runtime_instance", 1300)),
            x["live_protocol_evidence_binding"].__setitem__("observed_runtime_instance_id", _oid("runtime_instance", 1300)),
        )},
        {"report_1": lambda x: x["entry_observation"].__setitem__("uid", 42001)},
        {"receipt_body": lambda x: x["service_key_custody_evidence_in_exact_order"][0].__setitem__("runtime_instance_id", _oid("wrong_runtime", 1))},
    ],
)
def test_runtime_instance_uid_gid_or_custody_splice_is_rejected(mutators: Mapping[str, Mutator]) -> None:
    graph = _build_evidence(mutators=mutators)
    with pytest.raises(contract.Phase2BStep5R01EvidenceContractErrorV1):
        _verify(graph)


@pytest.mark.parametrize(
    ("artifact", "field"),
    [
        *[("report_0", field) for field in (
            "service_image_executed", "entry_exit_closure_equal",
            "uid_gid_verified", "control_domain_verified",
        )],
        *[("bundle", field) for field in (
            "all_images_executed", "all_entry_exit_closures_equal",
            "all_uids_gids_pairwise_distinct", "all_control_domains_pairwise_distinct",
            "all_live_protocol_evidence_bound",
        )],
        ("custody", "custody_verified"),
        *[("receipt_effect", field) for field in contract.RECEIPT_READINESS_EFFECT_FIELDS_V1],
        *[("verdict", field) for field in contract.R01_VERDICT_FIELDS_V1],
        ("overlay_row", "satisfied"),
        *[("overlay_effect", field) for field in (
            "all_nodes_satisfied", "r01_satisfied", "r02_or_later_satisfied",
            "execution_authorized",
        )],
    ],
)
def test_every_live_current_durable_authority_readiness_or_execution_true_is_rejected(
    artifact: str, field: str
) -> None:
    if artifact in {"report_0", "bundle"}:
        mutator = {artifact: lambda value: value.__setitem__(field, True)}
    elif artifact == "custody":
        mutator = {"receipt_body": lambda value: value["service_key_custody_evidence_in_exact_order"][0].__setitem__(field, True)}
    elif artifact == "receipt_effect":
        mutator = {"receipt_body": lambda value: value["readiness_effect"].__setitem__(field, True)}
    elif artifact == "verdict":
        mutator = {"receipt_body": lambda value: value["r01_verdict"].__setitem__(field, True)}
    elif artifact == "overlay_row":
        mutator = {"overlay": lambda value: value["readiness_effect"]["node_states_in_canonical_topological_order"][0].__setitem__(field, True)}
    else:
        mutator = {"overlay": lambda value: value["readiness_effect"].__setitem__(field, True)}
    graph = _build_evidence(mutators=mutator)
    with pytest.raises(contract.Phase2BStep5R01EvidenceContractErrorV1):
        _verify(graph)


@pytest.mark.parametrize("field", ["role", "purpose", "signer_key_id", "signature_hex"])
def test_signature_role_purpose_key_and_preimage_are_frozen(field: str) -> None:
    graph = _build_evidence()
    report = copy.deepcopy(graph.values["reports"][0])
    report["signature"][field] = {
        "role": "PURE_REPLAYER",
        "purpose": "COUNTERSIGN_R01_AUTHORIZATION_STATUS_RECEIPT",
        "signer_key_id": _oid("wrong_key", 1),
        "signature_hex": "00" * 64,
    }[field]
    raw = contract.canonical_r01_evidence_contract_json_bytes_v1(report)
    call = dict(graph.call)
    reports = list(call["live_runtime_closure_report_payloads_in_exact_order"])
    reports[0] = raw; call["live_runtime_closure_report_payloads_in_exact_order"] = tuple(reports)
    with pytest.raises(contract.Phase2BStep5R01EvidenceContractErrorV1) as caught:
        contract.verify_r01_evidence_contract_candidate_v1(**call)
    assert caught.value.code == contract.FAIL_SIGNATURE_V1


@pytest.mark.parametrize(
    "mutator",
    [
        lambda x: x.__setitem__("anchor_sequence", 2),
        lambda x: x.__setitem__("previous_external_anchor_receipt_id", _oid("prior_receipt", 1)),
    ],
)
def test_anchor_sequence_and_predecessor_are_consistent(mutator: Mutator) -> None:
    graph = _build_evidence(mutators={"anchor": mutator})
    with pytest.raises(contract.Phase2BStep5R01EvidenceContractErrorV1):
        _verify(graph)


def test_external_receipt_merkle_inclusion_proof_is_verified() -> None:
    graph = _build_evidence(mutators={"external_receipt": lambda x: x.__setitem__("signed_tree_root_sha256", _sha(9999))})
    with pytest.raises(contract.Phase2BStep5R01EvidenceContractErrorV1):
        _verify(graph)


@pytest.mark.parametrize(
    "epoch",
    [True, 1.5, 2, (1 << 53)],
    ids=["bool", "fractional-float", "wrong-int", "unsafe-int"],
)
def test_fully_readdressed_real_witness_policy_epoch_requires_exact_one_int(
    epoch: object,
) -> None:
    graph = _build_evidence(
        mutators={
            "policy": _policy_epoch_mutator(epoch),
            "external_receipt": lambda receipt: receipt.__setitem__(
                "witness_key_epoch", epoch
            ),
        }
    )
    _assert_evidence_error(contract.FAIL_SCHEMA_V1, lambda: _verify(graph))


def test_fully_readdressed_real_witness_policy_epoch_exact_int_one_accepts() -> None:
    graph = _build_evidence(
        mutators={
            "policy": _policy_epoch_mutator(1),
            "external_receipt": lambda receipt: receipt.__setitem__(
                "witness_key_epoch", 1
            ),
        }
    )
    assert _verify(graph).candidate_signature_mechanics_verified is True


@pytest.mark.parametrize(
    ("epoch", "code"),
    [
        (True, contract.FAIL_SCHEMA_V1),
        (1.5, contract.FAIL_SCHEMA_V1),
        (2, contract.FAIL_SCHEMA_V1),
        ((1 << 53), contract.FAIL_SCHEMA_V1),
    ],
    ids=["bool", "fractional-float", "wrong-int", "unsafe-int"],
)
def test_fully_readdressed_and_resigned_receipt_epoch_cannot_alias_policy_int(
    epoch: object, code: str
) -> None:
    graph = _build_evidence(
        mutators={
            "external_receipt": lambda receipt: receipt.__setitem__(
                "witness_key_epoch", epoch
            )
        }
    )
    _assert_evidence_error(code, lambda: _verify(graph))


@pytest.mark.parametrize(
    ("field", "evil"),
    [
        ("schema_version", False),
        ("schema_version", "wrong-schema"),
        ("external_witness_policy_id", []),
        ("witness_namespace", {}),
        ("witness_control_domain_id", False),
        ("witness_key_id", 1),
        ("algorithm", []),
        ("algorithm", "RSA"),
        ("anchor_id", {}),
        ("anchor_payload_sha256", False),
        ("previous_external_receipt_id", False),
        ("witness_sequence", True),
        ("leaf_index", False),
        ("tree_size", True),
        ("signed_tree_root_sha256", []),
        ("inclusion_path_sha256_in_exact_order", {}),
    ],
)
def test_fully_readdressed_and_resigned_receipt_header_types_are_exact(
    field: str, evil: object
) -> None:
    graph = _build_evidence(
        mutators={
            "external_receipt": lambda receipt: receipt.__setitem__(field, evil)
        }
    )
    _assert_evidence_error(contract.FAIL_SCHEMA_V1, lambda: _verify(graph))


def test_external_policy_payload_type_is_checked_before_raw_hashing() -> None:
    graph = _build_evidence()
    call = dict(graph.call)
    call["external_witness_policy_payload"] = graph.payloads["policy"].decode(
        "ascii"
    )
    _assert_evidence_error(
        contract.FAIL_INPUT_V1,
        lambda: contract.verify_r01_evidence_contract_candidate_v1(**call),
    )


def test_pure_countersignature_binds_custodian_signature_sha() -> None:
    graph = _build_evidence()
    dual = copy.deepcopy(graph.values["dual"])
    dual["custodian_signature"]["signature_hex"] = "00" * 64
    dual["receipt_id"] = contract.phase2b_step5_r01_evidence_content_id_v1(
        "dual_signed_r01_receipt", {k: v for k, v in dual.items() if k != "receipt_id"}
    )
    call = dict(graph.call)
    call["dual_signed_r01_receipt_payload"] = contract.canonical_r01_evidence_contract_json_bytes_v1(dual)
    with pytest.raises(contract.Phase2BStep5R01EvidenceContractErrorV1):
        contract.verify_r01_evidence_contract_candidate_v1(**call)


def test_private_result_cannot_be_forged_copied_or_mutated() -> None:
    graph = _build_evidence(); result = _verify(graph)
    with pytest.raises(TypeError):
        contract.VerifiedR01EvidenceContractCandidateV1()
    forged = object.__new__(contract.VerifiedR01EvidenceContractCandidateV1)
    with pytest.raises(contract.Phase2BStep5R01EvidenceContractErrorV1):
        contract.require_verified_r01_evidence_contract_candidate_v1(forged)
    with pytest.raises(TypeError):
        copy.copy(result)
    object.__setattr__(result, "r01_satisfied", True)
    with pytest.raises(contract.Phase2BStep5R01EvidenceContractErrorV1):
        contract.require_verified_r01_evidence_contract_candidate_v1(result)


def test_module_and_upstream_global_rebinding_does_not_change_acceptance() -> None:
    graph = _build_evidence()
    invalid_response = copy.deepcopy(graph.values["response"])
    invalid_response.pop("terminal")
    invalid_call = dict(graph.call)
    invalid_call["current_cut_response_payload"] = contract.canonical_r01_evidence_contract_json_bytes_v1(invalid_response)
    originals: list[tuple[object, str, object]] = []
    replacements = {
        "_fail": lambda *a, **k: None,
        "_canonical": lambda value: b"evil",
        "_decode": lambda payload, name: {},
        "_restore_wire_integers": lambda value: value,
        "_integer": lambda *a, **k: 0,
        "_sha_address": lambda *a, **k: "sha256:" + "0" * 64,
        "_same": lambda *a, **k: None,
        "_id": lambda *a, **k: _oid("evil", 1),
        "_validate_detached_signature": lambda *a, **k: {},
        "_load_external_policy": lambda *a, **k: {},
        "_verify_external_receipt_signature": lambda *a, **k: None,
        "_canonicalize": lambda value: b"evil",
        "_decode_canonical": lambda payload: {},
        "_DOMAINS": {name: b"EVIL\0" for name in contract._SCHEMAS},
        "_PREFIXES": {name: "evil_" for name in contract._SCHEMAS},
        "_SCHEMAS": {}, "_HEX64": None, "_HEX128": None, "_ID": None,
        "_authority_signature_preimage": lambda **kwargs: b"evil",
        "_Ed25519PublicKey": None,
    }
    upstream = {
        "load_step5_stage_b_trust_lock_v2": lambda *a, **k: object(),
        "load_step5_authority_profile_v2": lambda *a, **k: object(),
        "load_step5_signer_registry_v2": lambda *a, **k: object(),
        "require_locked_step5_authority_graph_v2": lambda *a, **k: (object(), object(), object()),
        "phase2b_step5_signature_preimage_v2": lambda **kwargs: b"evil",
    }
    try:
        for owner, mapping in ((contract, replacements), (authority, upstream)):
            for name, replacement in mapping.items():
                originals.append((owner, name, getattr(owner, name)))
                setattr(owner, name, replacement)
        assert contract.require_verified_r01_evidence_contract_candidate_v1(
            contract.verify_r01_evidence_contract_candidate_v1(**graph.call)
        ).r01_satisfied is False
        with pytest.raises(contract.Phase2BStep5R01EvidenceContractErrorV1):
            contract.verify_r01_evidence_contract_candidate_v1(**invalid_call)
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)


def test_public_accepting_boundary_requires_raw_graph_and_oob_pins() -> None:
    signature = inspect.signature(contract.verify_r01_evidence_contract_candidate_v1)
    assert "stage_b_trust_lock_payload" in signature.parameters
    assert "authority_profile_payload" in signature.parameters
    assert "signer_registry_payload" in signature.parameters
    assert "out_of_band_expected_lock_id_bytes" in signature.parameters
    assert "out_of_band_expected_lock_payload_sha256_digest" in signature.parameters
    assert not [
        name for name, value in vars(contract).items()
        if name.startswith("load_") and callable(value)
    ]
    for name in contract.__all__:
        value = getattr(contract, name)
        if callable(value) and name.startswith(("verify_", "issue_")):
            assert name in {
                "verify_r01_evidence_contract_candidate_v1",
                "verify_real_r01_admission_v1", "issue_admitted_r01_only_overlay_v1",
            }


@pytest.mark.parametrize("field", [
    "publication_plan_payload_sha256", "checkpoint_payload_sha256",
    "head_payload_sha256", "response_payload_sha256",
])
def test_cas_replay_payload_addresses_require_sha256_prefix(field: str) -> None:
    graph = _build_evidence()
    receipt_body = copy.deepcopy(graph.values["receipt_body"])
    receipt_body["cas_complete_session_binding"][field] = _sha(999)
    receipt_body.pop("receipt_body_id")
    receipt_body["receipt_body_id"] = contract.phase2b_step5_r01_evidence_content_id_v1(
        "r01_receipt_body", receipt_body
    )
    call = dict(graph.call)
    call["r01_receipt_body_payload"] = contract.canonical_r01_evidence_contract_json_bytes_v1(receipt_body)
    with pytest.raises(contract.Phase2BStep5R01EvidenceContractErrorV1):
        contract.verify_r01_evidence_contract_candidate_v1(**call)


def test_pure_composer_spool_cardinality_cannot_be_used_as_r01_main_cas() -> None:
    def pure_spool(body: dict[str, object]) -> None:
        cas = body["cas_complete_session_binding"]
        cas["object_count"] = 21
        cas["request_ids_in_exact_order"] = [_oid("pure_request", 3000 + index) for index in range(21)]
        cas["receipt_ids_in_exact_order"] = [_oid("pure_receipt", 4000 + index) for index in range(21)]

    graph = _build_evidence(mutators={"receipt_body": pure_spool})
    with pytest.raises(contract.Phase2BStep5R01EvidenceContractErrorV1) as caught:
        _verify(graph)
    assert caught.value.code == contract.FAIL_BINDING_V1


def test_clean_imports_and_source_contains_no_producer_io() -> None:
    source = inspect.getsource(contract)
    for forbidden in ("import socket", "import subprocess", "import pathlib", "import os", "phase2b_formal_wire_covert_step5_store_v", "phase2b_formal_wire_covert_step5_logical_time_v1", "phase2b_formal_wire_covert_step5_owner_status_v1", "phase2b_formal_wire_covert_step5_runtime_v1"):
        assert forbidden not in source
    assert importlib.reload(contract) is contract
