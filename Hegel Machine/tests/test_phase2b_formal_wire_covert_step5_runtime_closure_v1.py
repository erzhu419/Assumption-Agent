from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import importlib
import inspect
import json
import re
from types import SimpleNamespace
from typing import Callable

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import hegel_machine.phase2b_formal_wire_covert_step5_authority_v2 as authority
import hegel_machine.phase2b_formal_wire_covert_step5_runtime_closure_v1 as closure


PolicyMutator = Callable[[dict[str, object]], None]
EvidenceMutator = Callable[[dict[str, object]], None]
JsonMutator = Callable[[dict[str, object]], None]
StaticMutator = Callable[[dict[str, dict[str, object]]], None]


@dataclass(frozen=True)
class _RuntimeGraph:
    verify_arguments: dict[str, object]
    policy_body: dict[str, object]
    policy_payload: bytes
    evidence_body: dict[str, object]
    evidence_payload: bytes
    service_binding: dict[str, object]
    profile_id: str
    registry_id: str
    lock_id: str


class _EqAll:
    def __eq__(self, other: object) -> bool:
        return True

    def __hash__(self) -> int:
        return 1


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _oci_sha(payload: bytes) -> str:
    return "sha256:" + _sha(payload)


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _all_false_readiness() -> dict[str, object]:
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


def _role_keys() -> tuple[list[dict[str, object]], list[Ed25519PrivateKey]]:
    rows: list[dict[str, object]] = []
    private_keys: list[Ed25519PrivateKey] = []
    for ordinal, role in enumerate(authority.ROLE_ORDER_V2, start=1):
        private_key = Ed25519PrivateKey.from_private_bytes(bytes([ordinal]) * 32)
        private_keys.append(private_key)
        row: dict[str, object] = {
            "schema_version": authority.ROLE_KEY_SCHEMA_V1,
            "role": role,
            "control_domain": authority.ROLE_CONTROL_DOMAINS_V2[role],
            "authorized_signature_purposes": list(
                authority.ROLE_SIGNATURE_PURPOSES_V2[role]
            ),
            "algorithm": "Ed25519",
            "key_epoch": 1,
            "public_key_hex": private_key.public_key()
            .public_bytes(
                serialization.Encoding.Raw,
                serialization.PublicFormat.Raw,
            )
            .hex(),
        }
        row["key_id"] = authority.phase2b_step5_role_key_id_v2(row)
        rows.append(row)
    return rows, private_keys


def _python_entry(
    ordinal: int,
    module_name: str,
    classification: str,
    path: str,
) -> dict[str, object]:
    payload = ("python:" + module_name).encode("utf-8")
    return {
        "ordinal": ordinal,
        "module_name": module_name,
        "classification": classification,
        "origin_kind": "FILE",
        "path": path,
        "sha256": _sha(payload),
        "byte_size": len(payload),
    }


def _native_entry(
    ordinal: int,
    path: str,
    soname: str,
    marker: str,
) -> dict[str, object]:
    payload = ("elf:" + marker).encode("utf-8")
    return {
        "ordinal": ordinal,
        "path": path,
        "soname": soname,
        "sha256": _sha(payload),
        "byte_size": len(payload),
        "elf_build_id": _sha(("build:" + marker).encode("utf-8"))[:40],
    }


def _static_manifests() -> dict[str, dict[str, object]]:
    rows: dict[str, list[dict[str, object]]] = {
        "PYTHON_STDLIB": [
            _python_entry(
                0,
                "os",
                "STDLIB",
                "/usr/local/lib/python3.10/os.py",
            ),
            _python_entry(
                1,
                "json",
                "STDLIB",
                "/usr/local/lib/python3.10/json/__init__.py",
            ),
        ],
        "IMPORT_CLOSURE": [
            _python_entry(
                0,
                "hegel_machine",
                "PROJECT",
                "/opt/hegel/hegel_machine/__init__.py",
            ),
            _python_entry(
                1,
                "hegel_machine.worker",
                "PROJECT",
                "/opt/hegel/hegel_machine/worker.py",
            ),
        ],
        "NATIVE_LOADER": [
            _native_entry(
                0,
                "/lib64/ld-linux-x86-64.so.2",
                "ld-linux-x86-64.so.2",
                "loader",
            )
        ],
        "SHARED_LIBRARIES": [
            _native_entry(
                0,
                "/usr/lib/x86_64-linux-gnu/libc.so.6",
                "libc.so.6",
                "libc",
            ),
            _native_entry(
                1,
                "/usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0",
                "libpython3.10.so.1.0",
                "libpython",
            ),
        ],
    }
    result: dict[str, dict[str, object]] = {}
    for kind in closure.STATIC_MANIFEST_KINDS_V1:
        entries = rows[kind]
        result[kind] = {
            "schema_version": closure.STATIC_RUNTIME_MANIFEST_SCHEMA_V1,
            "manifest_kind": kind,
            "inventory_scope": "STATIC_IMAGE_DECLARATION_NOT_LIVE_OBSERVATION",
            "entries_in_exact_order": entries,
            "entry_count": len(entries),
            "static_inventory_complete_declared": True,
            "live_observation": False,
        }
    return result


def _build_runtime_graph(
    *,
    variant: str | None = None,
    policy_mutator: PolicyMutator | None = None,
    evidence_mutator: EvidenceMutator | None = None,
    index_mutator: JsonMutator | None = None,
    manifest_mutator: JsonMutator | None = None,
    config_mutator: JsonMutator | None = None,
    static_mutator: StaticMutator | None = None,
) -> _RuntimeGraph:
    role_keys, private_keys = _role_keys()
    service_name = "runner"
    service_role = authority.SERVICE_ROLE_V2[service_name]
    service_key = role_keys[authority.ROLE_ORDER_V2.index(service_role)]

    raw_support = (
        b"hegel-source-archive-v1",
        b"locked-dependencies-v1",
        b"spdx-sbom-v1",
        b"in-toto-provenance-v1",
        b"cpython-3.10.12-executable-v1",
    )
    source_binding = {
        "source_commit_sha": "12" * 20,
        "source_archive_sha256": _sha(raw_support[0]),
        "dependency_lock_sha256": _sha(raw_support[1]),
    }
    support_rows = [
        {
            "ordinal": ordinal,
            "role": role,
            "media_type": closure.SUPPORT_ARTIFACT_MEDIA_TYPES_V1[role],
            "sha256": _sha(payload),
            "byte_size": len(payload),
        }
        for ordinal, (role, payload) in enumerate(
            zip(closure.SUPPORT_ARTIFACT_ROLES_V1, raw_support, strict=True)
        )
    ]

    manifests = _static_manifests()
    if static_mutator is not None:
        static_mutator(manifests)
    manifest_commitments = {
        "stdlib_manifest_sha256": _sha(
            closure.canonical_runtime_closure_json_bytes_v1(
                manifests["PYTHON_STDLIB"]
            )
        ),
        "import_manifest_sha256": _sha(
            closure.canonical_runtime_closure_json_bytes_v1(
                manifests["IMPORT_CLOSURE"]
            )
        ),
        "native_loader_manifest_sha256": _sha(
            closure.canonical_runtime_closure_json_bytes_v1(
                manifests["NATIVE_LOADER"]
            )
        ),
        "shared_library_manifest_sha256": _sha(
            closure.canonical_runtime_closure_json_bytes_v1(
                manifests["SHARED_LIBRARIES"]
            )
        ),
    }

    platform = {"os": "linux", "architecture": "amd64", "variant": variant}
    layer_media_type = "application/vnd.oci.image.layer.v1.tar+gzip"
    raw_layers = (b"raw-layer-one-v1", b"raw-layer-two-v1")
    layer_digests = [_oci_sha(payload) for payload in raw_layers]
    diff_ids = [
        _oci_sha(b"uncompressed-one-v1"),
        _oci_sha(b"uncompressed-two-v1"),
    ]
    config: dict[str, object] = {
        "architecture": platform["architecture"],
        "os": platform["os"],
        "rootfs": {"type": "layers", "diff_ids": diff_ids},
    }
    if variant is not None:
        config["variant"] = variant
    if config_mutator is not None:
        config_mutator(config)
    raw_config = _json_bytes(config)
    config_media_type = "application/vnd.oci.image.config.v1+json"
    manifest: dict[str, object] = {
        "schemaVersion": 2,
        "mediaType": closure.OCI_MANIFEST_MEDIA_TYPE_V1,
        "config": {
            "mediaType": config_media_type,
            "digest": _oci_sha(raw_config),
            "size": len(raw_config),
        },
        "layers": [
            {
                "mediaType": layer_media_type,
                "digest": digest,
                "size": len(payload),
            }
            for digest, payload in zip(layer_digests, raw_layers, strict=True)
        ],
    }
    if manifest_mutator is not None:
        manifest_mutator(manifest)
    raw_manifest = _json_bytes(manifest)
    raw_index_platform = {
        "os": platform["os"],
        "architecture": platform["architecture"],
    }
    if variant is not None:
        raw_index_platform["variant"] = variant
    index: dict[str, object] = {
        "schemaVersion": 2,
        "mediaType": closure.OCI_INDEX_MEDIA_TYPE_V1,
        "manifests": [
            {
                "mediaType": closure.OCI_MANIFEST_MEDIA_TYPE_V1,
                "digest": _oci_sha(raw_manifest),
                "size": len(raw_manifest),
                "platform": raw_index_platform,
            }
        ],
    }
    if index_mutator is not None:
        index_mutator(index)
    raw_index = _json_bytes(index)

    uid, gid = authority.SERVICE_UID_GID_V2[service_name]
    policy_body: dict[str, object] = {
        "schema_version": closure.RUNTIME_CLOSURE_POLICY_SCHEMA_V1,
        "status": "STATIC_POLICY_CANDIDATE_NOT_OBSERVED_LIVE",
        "claim_level": (
            "OFFLINE_STATIC_BYTE_POLICY_ONLY_NOT_IMAGE_EXISTENCE_NOT_LIVE_"
            "RUNTIME_CLOSURE_NOT_AUTHORITY_NOT_READINESS"
        ),
        "canonicalization": "RFC8785_EXACT_BYTES_NO_TRAILING_LF",
        "service_name": service_name,
        "service_type": authority.SERVICE_TYPES_V2[service_name],
        "role": service_role,
        "control_domain": authority.ROLE_CONTROL_DOMAINS_V2[service_role],
        "uid": uid,
        "gid": gid,
        "role_key_id": service_key["key_id"],
        "key_epoch": 1,
        "oci_platform": platform,
        "oci_media_policy": {
            "index_media_type": closure.OCI_INDEX_MEDIA_TYPE_V1,
            "manifest_media_type": closure.OCI_MANIFEST_MEDIA_TYPE_V1,
            "config_media_type": config_media_type,
            "allowed_layer_media_types_in_exact_order": [layer_media_type],
        },
        "runtime_identity": dict(closure._EXACT_RUNTIME_IDENTITY_V1),
        "bound_artifacts_in_exact_order": support_rows,
        "static_manifest_commitments": manifest_commitments,
        "resource_bounds": {
            "max_oci_metadata_bytes": 1_000_000,
            "max_blob_bytes": 2_000_000,
            "max_total_layer_bytes": 4_000_000,
            "max_layers": 8,
            "max_manifest_entries": 100,
            "max_path_utf8_bytes": 4096,
            "max_module_name_utf8_bytes": 512,
            "max_chunk_bytes": 1024,
        },
        "static_verification_requirements": dict(
            closure._EXACT_STATIC_REQUIREMENTS_V1
        ),
        "permanent_nonclaims": dict(closure._EXACT_PERMANENT_NONCLAIMS_V1),
    }
    if policy_mutator is not None:
        policy_mutator(policy_body)
    policy_id = closure.phase2b_step5_runtime_closure_policy_id_v1(policy_body)
    policy_payload = closure.canonical_runtime_closure_json_bytes_v1(
        {"body": policy_body, "id": policy_id}
    )

    service_bindings: list[dict[str, object]] = []
    for ordinal, current_service in enumerate(authority.SERVICE_ORDER_V2, start=1):
        role = authority.SERVICE_ROLE_V2[current_service]
        role_key = role_keys[authority.ROLE_ORDER_V2.index(role)]
        current_uid, current_gid = authority.SERVICE_UID_GID_V2[current_service]
        if current_service == service_name:
            image_digests = (
                _oci_sha(raw_index),
                _oci_sha(raw_manifest),
                _oci_sha(raw_config),
            )
            current_layers = layer_digests
            policy_sha = _sha(policy_payload)
        else:
            image_digests = tuple(
                _oci_sha(f"{current_service}:{kind}".encode("ascii"))
                for kind in ("index", "manifest", "config")
            )
            current_layers = [
                _oci_sha(f"{current_service}:layer:{index}".encode("ascii"))
                for index in range(2)
            ]
            policy_sha = _sha(f"{current_service}:policy".encode("ascii"))
        binding: dict[str, object] = {
            "schema_version": authority.SERVICE_STATIC_BINDING_SCHEMA_V2,
            "service_name": current_service,
            "service_type": authority.SERVICE_TYPES_V2[current_service],
            "binding_status": "STATIC_REQUESTED_BINDING_NOT_OBSERVED_LIVE",
            "role": role,
            "control_domain": authority.ROLE_CONTROL_DOMAINS_V2[role],
            "uid": current_uid,
            "gid": current_gid,
            "role_key_id": role_key["key_id"],
            "key_epoch": 1,
            "authorized_signature_purposes": list(
                authority.ROLE_SIGNATURE_PURPOSES_V2[role]
            ),
            "oci_index_digest": image_digests[0],
            "oci_manifest_digest": image_digests[1],
            "oci_config_digest": image_digests[2],
            "ordered_layer_digests": current_layers,
            "static_runtime_closure_policy_sha256": policy_sha,
        }
        binding["binding_id"] = (
            authority.phase2b_step5_service_static_binding_id_v2(binding)
        )
        service_bindings.append(binding)

    profile_body: dict[str, object] = {
        "schema_version": authority.AUTHORITY_PROFILE_SCHEMA_V2,
        "status": "CANDIDATE_STATIC_SERVICE_AUTHORITY_PROFILE_NOT_ADMITTED",
        "claim_level": (
            "STATIC_CANDIDATE_FOUNDATION_NOT_IMAGE_EXISTENCE_NOT_LIVE_"
            "RUNTIME_CLOSURE_NOT_AUTHORITY_NOT_READINESS"
        ),
        "canonicalization": "RFC8785_EXACT_BYTES_NO_TRAILING_LF",
        "profile_epoch": 2,
        "owner_authorization_binding": dict(authority._EXACT_OWNER_BINDING_V2),
        "readiness_dag_binding": {
            "schema_version": (
                "hegel-machine-phase2b-actual-unsealed-960-readiness-dag/1"
            ),
            "readiness_dag_id": authority.READINESS_DAG_ID_V1,
        },
        "readiness_state": _all_false_readiness(),
        "qv2_binding": dict(authority.EXACT_QV2_BINDING_V1),
        "source_binding": source_binding,
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
    profile_id = authority.phase2b_step5_content_id_v2(
        "authority_profile", profile_body
    )
    profile_payload = authority.canonical_step5_json_bytes_v2(
        {"body": profile_body, "id": profile_id}
    )

    registry_body: dict[str, object] = {
        "schema_version": authority.SIGNER_REGISTRY_SCHEMA_V2,
        "authority_profile_id": profile_id,
        "registry_epoch": 2,
        "status": "CANDIDATE_NOT_ADMITTED",
        "claim_level": (
            "SIGNED_STATIC_KEY_REGISTRY_CANDIDATE_NOT_AUTHORITY_NOT_READINESS"
        ),
        "revocation_epoch": 0,
        "role_keys_in_exact_order": role_keys,
        "readiness_state": _all_false_readiness(),
    }
    registry_id = authority.phase2b_step5_content_id_v2(
        "signer_registry", registry_body
    )
    signature_preimage = authority.phase2b_step5_signature_preimage_v2(
        artifact_kind="signer_registry",
        body=registry_body,
        role="GOVERNANCE_ANCHOR",
        purpose="SIGN_SIGNER_REGISTRY",
        signer_key_id=role_keys[0]["key_id"],  # type: ignore[arg-type]
    )
    registry_payload = authority.canonical_step5_json_bytes_v2(
        {
            "body": registry_body,
            "id": registry_id,
            "signatures": [
                {
                    "schema_version": authority.DETACHED_SIGNATURE_SCHEMA_V2,
                    "algorithm": "Ed25519",
                    "role": "GOVERNANCE_ANCHOR",
                    "purpose": "SIGN_SIGNER_REGISTRY",
                    "key_epoch": 1,
                    "signer_key_id": role_keys[0]["key_id"],
                    "signature_hex": private_keys[0].sign(
                        signature_preimage
                    ).hex(),
                }
            ],
        }
    )
    lock_body: dict[str, object] = {
        "schema_version": authority.STAGE_B_TRUST_LOCK_SCHEMA_V2,
        "status": "CANDIDATE_OUT_OF_BAND_STAGE_B_TRUST_LOCK",
        "claim_level": "CANDIDATE_GRAPH_PIN_ONLY_NOT_AUTHORITY_NOT_READINESS",
        "canonicalization": "RFC8785_EXACT_BYTES_NO_TRAILING_LF",
        "authority_profile_schema": authority.AUTHORITY_PROFILE_SCHEMA_V2,
        "authority_profile_id": profile_id,
        "authority_profile_payload_sha256": _sha(profile_payload),
        "signer_registry_schema": authority.SIGNER_REGISTRY_SCHEMA_V2,
        "signer_registry_id": registry_id,
        "signer_registry_payload_sha256": _sha(registry_payload),
        "qv2_binding": dict(authority.EXACT_QV2_BINDING_V1),
    }
    lock_id = authority.phase2b_step5_content_id_v2(
        "stage_b_trust_lock", lock_body
    )
    lock_payload = authority.canonical_step5_json_bytes_v2(
        {"body": lock_body, "id": lock_id}
    )

    runner_binding = service_bindings[authority.SERVICE_ORDER_V2.index(service_name)]
    evidence_body: dict[str, object] = {
        "schema_version": closure.IMAGE_CLOSURE_EVIDENCE_SCHEMA_V1,
        "status": "OFFLINE_STATIC_IMAGE_CLOSURE_MECHANICS_CANDIDATE",
        "claim_level": (
            "RAW_STATIC_BYTES_MATCHED_ONLY_NOT_IMAGE_EXISTENCE_NOT_EXECUTION_"
            "NOT_LIVE_RUNTIME_CLOSURE_NOT_AUTHORITY_NOT_READINESS"
        ),
        "canonicalization": "RFC8785_EXACT_BYTES_NO_TRAILING_LF",
        "stage_b_trust_lock_id": lock_id,
        "authority_profile_id": profile_id,
        "signer_registry_id": registry_id,
        "service_name": service_name,
        "service_binding_id": runner_binding["binding_id"],
        "closure_policy_id": policy_id,
        "closure_policy_payload_sha256": _sha(policy_payload),
        "oci_image": {
            "index_media_type": closure.OCI_INDEX_MEDIA_TYPE_V1,
            "index_digest": _oci_sha(raw_index),
            "index_byte_size": len(raw_index),
            "manifest_media_type": closure.OCI_MANIFEST_MEDIA_TYPE_V1,
            "manifest_digest": _oci_sha(raw_manifest),
            "manifest_byte_size": len(raw_manifest),
            "config_media_type": config_media_type,
            "config_digest": _oci_sha(raw_config),
            "config_byte_size": len(raw_config),
            "platform": platform,
            "rootfs_diff_ids_in_exact_order": diff_ids,
            "ordered_layer_descriptors": [
                {
                    "ordinal": ordinal,
                    "media_type": layer_media_type,
                    "digest": digest,
                    "byte_size": len(payload),
                }
                for ordinal, (digest, payload) in enumerate(
                    zip(layer_digests, raw_layers, strict=True)
                )
            ],
        },
        "runtime_identity": dict(closure._EXACT_RUNTIME_IDENTITY_V1),
        "bound_artifacts_in_exact_order": copy.deepcopy(support_rows),
        "stdlib_manifest": manifests["PYTHON_STDLIB"],
        "import_manifest": manifests["IMPORT_CLOSURE"],
        "native_loader_manifest": manifests["NATIVE_LOADER"],
        "shared_library_manifest": manifests["SHARED_LIBRARIES"],
        "verification_state": dict(closure._EXACT_VERIFICATION_STATE_V1),
    }
    if evidence_mutator is not None:
        evidence_mutator(evidence_body)
    evidence_id = closure.phase2b_step5_image_closure_evidence_id_v1(
        evidence_body
    )
    evidence_payload = closure.canonical_runtime_closure_json_bytes_v1(
        {"body": evidence_body, "id": evidence_id}
    )
    arguments: dict[str, object] = {
        "stage_b_trust_lock_payload": lock_payload,
        "authority_profile_payload": profile_payload,
        "signer_registry_payload": registry_payload,
        "out_of_band_expected_lock_id_bytes": lock_id.encode("ascii"),
        "out_of_band_expected_lock_payload_sha256_digest": hashlib.sha256(
            lock_payload
        ).digest(),
        "service_name": service_name,
        "runtime_closure_policy_payload": policy_payload,
        "image_closure_evidence_payload": evidence_payload,
        "raw_oci_index_payload": raw_index,
        "raw_oci_manifest_payload": raw_manifest,
        "raw_oci_config_payload": raw_config,
        "ordered_raw_layer_chunk_tuples": tuple(
            ((payload[:3], payload[3:]) for payload in raw_layers)
        ),
        "ordered_support_artifact_chunk_tuples": tuple(
            ((payload[:4], payload[4:]) for payload in raw_support)
        ),
    }
    return _RuntimeGraph(
        verify_arguments=arguments,
        policy_body=policy_body,
        policy_payload=policy_payload,
        evidence_body=evidence_body,
        evidence_payload=evidence_payload,
        service_binding=runner_binding,
        profile_id=profile_id,
        registry_id=registry_id,
        lock_id=lock_id,
    )


def _verify(graph: _RuntimeGraph):
    return closure.verify_offline_step5_image_closure_mechanics_v1(
        **graph.verify_arguments  # type: ignore[arg-type]
    )


def _assert_failure(graph: _RuntimeGraph, *codes: str) -> None:
    with pytest.raises(closure.Phase2BStep5RuntimeClosureErrorV1) as caught:
        _verify(graph)
    assert caught.value.code in codes


def _replace_evidence(
    graph: _RuntimeGraph,
    mutator: EvidenceMutator,
) -> _RuntimeGraph:
    body = copy.deepcopy(graph.evidence_body)
    mutator(body)
    evidence_id = closure.phase2b_step5_image_closure_evidence_id_v1(body)
    payload = closure.canonical_runtime_closure_json_bytes_v1(
        {"body": body, "id": evidence_id}
    )
    arguments = dict(graph.verify_arguments)
    arguments["image_closure_evidence_payload"] = payload
    return _RuntimeGraph(
        verify_arguments=arguments,
        policy_body=graph.policy_body,
        policy_payload=graph.policy_payload,
        evidence_body=body,
        evidence_payload=payload,
        service_binding=graph.service_binding,
        profile_id=graph.profile_id,
        registry_id=graph.registry_id,
        lock_id=graph.lock_id,
    )


def _readdress_resign_and_repin_alternate_authority_graph(
    graph: _RuntimeGraph,
    *,
    profile_schema: str,
    registry_schema: str,
    lock_schema: str,
    signature_schema: str,
    qv2_binding: dict[str, object],
) -> _RuntimeGraph:
    """Rebuild every authority/evidence address, signature, digest, and OOB pin."""

    decode = authority.decode_canonical_step5_json_v2
    canonical = authority.canonical_step5_json_bytes_v2
    content_id = authority.phase2b_step5_content_id_v2
    signature_preimage = authority.phase2b_step5_signature_preimage_v2

    profile_envelope = decode(
        graph.verify_arguments["authority_profile_payload"]  # type: ignore[arg-type]
    )
    profile_body = copy.deepcopy(profile_envelope["body"])
    profile_body["schema_version"] = profile_schema
    profile_body["qv2_binding"] = copy.deepcopy(qv2_binding)
    profile_id = content_id("authority_profile", profile_body)
    profile_payload = canonical({"body": profile_body, "id": profile_id})

    registry_envelope = decode(
        graph.verify_arguments["signer_registry_payload"]  # type: ignore[arg-type]
    )
    registry_body = copy.deepcopy(registry_envelope["body"])
    registry_body["schema_version"] = registry_schema
    registry_body["authority_profile_id"] = profile_id
    registry_id = content_id("signer_registry", registry_body)
    governance_key = registry_body["role_keys_in_exact_order"][0]
    governance_private_key = Ed25519PrivateKey.from_private_bytes(b"\x01" * 32)
    registry_payload = canonical(
        {
            "body": registry_body,
            "id": registry_id,
            "signatures": [
                {
                    "schema_version": signature_schema,
                    "algorithm": "Ed25519",
                    "role": "GOVERNANCE_ANCHOR",
                    "purpose": "SIGN_SIGNER_REGISTRY",
                    "key_epoch": 1,
                    "signer_key_id": governance_key["key_id"],
                    "signature_hex": governance_private_key.sign(
                        signature_preimage(
                            artifact_kind="signer_registry",
                            body=registry_body,
                            role="GOVERNANCE_ANCHOR",
                            purpose="SIGN_SIGNER_REGISTRY",
                            signer_key_id=governance_key["key_id"],
                        )
                    ).hex(),
                }
            ],
        }
    )

    lock_envelope = decode(
        graph.verify_arguments["stage_b_trust_lock_payload"]  # type: ignore[arg-type]
    )
    lock_body = copy.deepcopy(lock_envelope["body"])
    lock_body["schema_version"] = lock_schema
    lock_body["authority_profile_schema"] = profile_schema
    lock_body["authority_profile_id"] = profile_id
    lock_body["authority_profile_payload_sha256"] = _sha(profile_payload)
    lock_body["signer_registry_schema"] = registry_schema
    lock_body["signer_registry_id"] = registry_id
    lock_body["signer_registry_payload_sha256"] = _sha(registry_payload)
    lock_body["qv2_binding"] = copy.deepcopy(qv2_binding)
    lock_id = content_id("stage_b_trust_lock", lock_body)
    lock_payload = canonical({"body": lock_body, "id": lock_id})

    evidence_body = copy.deepcopy(graph.evidence_body)
    evidence_body["stage_b_trust_lock_id"] = lock_id
    evidence_body["authority_profile_id"] = profile_id
    evidence_body["signer_registry_id"] = registry_id
    evidence_id = closure.phase2b_step5_image_closure_evidence_id_v1(
        evidence_body
    )
    evidence_payload = closure.canonical_runtime_closure_json_bytes_v1(
        {"body": evidence_body, "id": evidence_id}
    )

    arguments = dict(graph.verify_arguments)
    arguments.update(
        {
            "stage_b_trust_lock_payload": lock_payload,
            "authority_profile_payload": profile_payload,
            "signer_registry_payload": registry_payload,
            "out_of_band_expected_lock_id_bytes": lock_id.encode("ascii"),
            "out_of_band_expected_lock_payload_sha256_digest": hashlib.sha256(
                lock_payload
            ).digest(),
            "image_closure_evidence_payload": evidence_payload,
        }
    )
    return _RuntimeGraph(
        verify_arguments=arguments,
        policy_body=graph.policy_body,
        policy_payload=graph.policy_payload,
        evidence_body=evidence_body,
        evidence_payload=evidence_payload,
        service_binding=graph.service_binding,
        profile_id=profile_id,
        registry_id=registry_id,
        lock_id=lock_id,
    )


@pytest.mark.parametrize("variant", [None, "v8"])
def test_valid_cycle_free_exact_raw_graph_closes(variant: str | None) -> None:
    graph = _build_runtime_graph(variant=variant)
    result = _verify(graph)
    assert closure.require_verified_offline_step5_image_closure_mechanics_v1(
        result
    ) is result
    assert result.closure_policy_payload_sha256 == _sha(graph.policy_payload)
    assert result.ordered_layer_digests == tuple(
        graph.service_binding["ordered_layer_digests"]
    )
    assert result.offline_static_byte_graph_mechanics_verified is True
    assert result.mechanics_only is True
    for field in closure._PERMANENT_NONCLAIMS_FIELDS_V1:
        assert getattr(result, field) is False


def test_policy_is_pre_profile_and_evidence_supplies_later_lineage() -> None:
    graph = _build_runtime_graph()
    for later_identifier in (
        graph.service_binding["binding_id"],
        graph.profile_id,
        graph.registry_id,
        graph.lock_id,
    ):
        assert str(later_identifier).encode("ascii") not in graph.policy_payload
        assert str(later_identifier).encode("ascii") in graph.evidence_payload
    assert _sha(graph.policy_payload) == graph.service_binding[
        "static_runtime_closure_policy_sha256"
    ]
    assert graph.policy_body["role_key_id"] in graph.policy_payload.decode("utf-8")


def test_public_verifier_has_exact_keyword_only_surface_without_var_kwargs() -> None:
    signature = inspect.signature(
        closure.verify_offline_step5_image_closure_mechanics_v1
    )
    assert tuple(signature.parameters) == (
        "stage_b_trust_lock_payload",
        "authority_profile_payload",
        "signer_registry_payload",
        "out_of_band_expected_lock_id_bytes",
        "out_of_band_expected_lock_payload_sha256_digest",
        "service_name",
        "runtime_closure_policy_payload",
        "image_closure_evidence_payload",
        "raw_oci_index_payload",
        "raw_oci_manifest_payload",
        "raw_oci_config_payload",
        "ordered_raw_layer_chunk_tuples",
        "ordered_support_artifact_chunk_tuples",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    assert not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("stage_b_trust_lock_payload", bytearray(b"x")),
        ("authority_profile_payload", memoryview(b"x")),
        ("signer_registry_payload", "not-bytes"),
        ("out_of_band_expected_lock_id_bytes", bytearray(b"x")),
        (
            "out_of_band_expected_lock_payload_sha256_digest",
            b"x" * 31,
        ),
        ("service_name", _EqAll()),
        ("runtime_closure_policy_payload", bytearray(b"x")),
        ("image_closure_evidence_payload", memoryview(b"x")),
        ("raw_oci_index_payload", bytearray(b"x")),
        ("raw_oci_manifest_payload", memoryview(b"x")),
        ("raw_oci_config_payload", "not-bytes"),
        ("ordered_raw_layer_chunk_tuples", []),
        ("ordered_support_artifact_chunk_tuples", []),
    ],
)
def test_top_level_transport_types_are_exact(field: str, replacement: object) -> None:
    graph = _build_runtime_graph()
    graph.verify_arguments[field] = replacement
    _assert_failure(graph, closure.FAIL_INPUT_V1, closure.FAIL_TRUST_ANCHOR_V1)


@pytest.mark.parametrize(
    "replacement",
    [
        ((bytearray(b"a"),), (b"b",)),
        ((memoryview(b"a"),), (b"b",)),
        ((b"",), (b"b",)),
        ((b"a",), [b"b"]),
    ],
)
def test_layer_chunk_types_are_exact(replacement: object) -> None:
    graph = _build_runtime_graph()
    graph.verify_arguments["ordered_raw_layer_chunk_tuples"] = replacement
    _assert_failure(graph, closure.FAIL_INPUT_V1)


def test_support_chunk_matrix_is_fully_preflighted_before_first_hash() -> None:
    graph = _build_runtime_graph()
    graph.verify_arguments["ordered_support_artifact_chunk_tuples"] = (
        (b"wrong-first-digest",),
        [b"late-list-is-not-a-tuple"],
        (b"x",),
        (b"x",),
        (b"x",),
    )
    _assert_failure(graph, closure.FAIL_INPUT_V1)


def test_support_chunk_bytes_are_exact_even_after_an_earlier_bad_digest() -> None:
    graph = _build_runtime_graph()
    graph.verify_arguments["ordered_support_artifact_chunk_tuples"] = (
        (b"wrong-first-digest",),
        (b"x",),
        (b"x",),
        (b"x",),
        (bytearray(b"late-non-bytes"),),
    )
    _assert_failure(graph, closure.FAIL_INPUT_V1)


@pytest.mark.parametrize(
    "replacement",
    [
        tuple(),
        ((b"raw-layer-two-v1",), (b"raw-layer-one-v1",)),
        ((b"raw-layer-one-v1",), (b"raw-layer-one-v1",)),
        ((b"raw-layer-one-v1!",), (b"raw-layer-two-v1",)),
        (
            (b"raw-layer-one-v1",),
            (b"raw-layer-two-v1",),
            (b"extra",),
        ),
    ],
)
def test_layer_missing_extra_reorder_duplicate_or_hash_change_fails(
    replacement: object,
) -> None:
    graph = _build_runtime_graph()
    graph.verify_arguments["ordered_raw_layer_chunk_tuples"] = replacement
    _assert_failure(graph, closure.FAIL_INPUT_V1, closure.FAIL_DIGEST_V1)


@pytest.mark.parametrize(
    "replacement",
    [
        tuple(),
        tuple((b"x",) for _ in range(4)),
        tuple((b"x",) for _ in range(6)),
        (
            (b"locked-dependencies-v1",),
            (b"hegel-source-archive-v1",),
            (b"spdx-sbom-v1",),
            (b"in-toto-provenance-v1",),
            (b"cpython-3.10.12-executable-v1",),
        ),
        tuple((b"same",) for _ in range(5)),
    ],
)
def test_support_missing_extra_reorder_duplicate_or_hash_change_fails(
    replacement: object,
) -> None:
    graph = _build_runtime_graph()
    graph.verify_arguments["ordered_support_artifact_chunk_tuples"] = replacement
    _assert_failure(graph, closure.FAIL_INPUT_V1, closure.FAIL_DIGEST_V1)


def test_support_byte_size_commitment_is_checked() -> None:
    def mutate(body: dict[str, object]) -> None:
        rows = body["bound_artifacts_in_exact_order"]
        assert isinstance(rows, list)
        rows[2]["byte_size"] += 1  # type: ignore[index,operator]

    graph = _build_runtime_graph(policy_mutator=mutate)
    _assert_failure(graph, closure.FAIL_DIGEST_V1)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda index: index.__setitem__("extra", False),
        lambda index: index.__setitem__("schemaVersion", True),
        lambda index: index["manifests"].append(  # type: ignore[union-attr]
            copy.deepcopy(index["manifests"][0])  # type: ignore[index]
        ),
        lambda index: index["manifests"][0].__setitem__(  # type: ignore[index,union-attr]
            "size", 1
        ),
        lambda index: index["manifests"][0]["platform"].__setitem__(  # type: ignore[index,union-attr]
            "extra", "x"
        ),
    ],
)
def test_oci_index_exact_descriptor_graph_rejects_malformed(mutator: JsonMutator) -> None:
    graph = _build_runtime_graph(index_mutator=mutator)
    _assert_failure(graph, closure.FAIL_OCI_GRAPH_V1, closure.FAIL_SCHEMA_V1)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda manifest: manifest.__setitem__("extra", False),
        lambda manifest: manifest.__setitem__("schemaVersion", True),
        lambda manifest: manifest["config"].__setitem__("extra", "x"),  # type: ignore[union-attr]
        lambda manifest: manifest["config"].__setitem__("size", 1),  # type: ignore[union-attr]
        lambda manifest: manifest.__setitem__(
            "layers", list(reversed(manifest["layers"]))  # type: ignore[arg-type]
        ),
        lambda manifest: manifest["layers"].__setitem__(  # type: ignore[union-attr]
            1, copy.deepcopy(manifest["layers"][0])  # type: ignore[index]
        ),
    ],
)
def test_oci_manifest_exact_descriptor_graph_rejects_malformed(
    mutator: JsonMutator,
) -> None:
    graph = _build_runtime_graph(manifest_mutator=mutator)
    _assert_failure(graph, closure.FAIL_OCI_GRAPH_V1, closure.FAIL_SCHEMA_V1)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda config: config.__setitem__("architecture", "arm64"),
        lambda config: config.__setitem__("variant", "unexpected"),
        lambda config: config["rootfs"].__setitem__("extra", False),  # type: ignore[union-attr]
        lambda config: config["rootfs"].__setitem__("type", "not-layers"),  # type: ignore[union-attr]
        lambda config: config["rootfs"].__setitem__(  # type: ignore[union-attr]
            "diff_ids",
            [config["rootfs"]["diff_ids"][0]] * 2,  # type: ignore[index]
        ),
    ],
)
def test_oci_config_platform_rootfs_graph_rejects_malformed(
    mutator: JsonMutator,
) -> None:
    graph = _build_runtime_graph(config_mutator=mutator)
    _assert_failure(graph, closure.FAIL_OCI_GRAPH_V1, closure.FAIL_SCHEMA_V1)


def test_oci_json_rejects_duplicate_keys_float_and_unsafe_integer() -> None:
    graph = _build_runtime_graph()
    for raw in (
        b'{"schemaVersion":2,"schemaVersion":2,"mediaType":"x","manifests":[]}',
        b'{"schemaVersion":2.0,"mediaType":"x","manifests":[]}',
        b'{"schemaVersion":9007199254740992,"mediaType":"x","manifests":[]}',
    ):
        changed = copy.deepcopy(graph)
        changed.verify_arguments["raw_oci_index_payload"] = raw
        _assert_failure(changed, closure.FAIL_DIGEST_V1, closure.FAIL_OCI_GRAPH_V1)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda manifests: manifests["PYTHON_STDLIB"].__setitem__("extra", False),
        lambda manifests: manifests["PYTHON_STDLIB"][
            "entries_in_exact_order"
        ][0].pop("sha256"),  # type: ignore[index,union-attr]
        lambda manifests: manifests["PYTHON_STDLIB"][
            "entries_in_exact_order"
        ].__setitem__(  # type: ignore[union-attr]
            1,
            copy.deepcopy(
                manifests["PYTHON_STDLIB"]["entries_in_exact_order"][0]  # type: ignore[index]
            ),
        ),
        lambda manifests: manifests["NATIVE_LOADER"].__setitem__(
            "entries_in_exact_order", []
        ),
        lambda manifests: manifests["SHARED_LIBRARIES"][
            "entries_in_exact_order"
        ][1].__setitem__(  # type: ignore[index,union-attr]
            "path",
            manifests["SHARED_LIBRARIES"]["entries_in_exact_order"][0][  # type: ignore[index]
                "path"
            ],
        ),
    ],
)
def test_static_manifest_missing_extra_duplicate_and_cardinality_fail(
    mutator: StaticMutator,
) -> None:
    graph = _build_runtime_graph(static_mutator=mutator)
    _assert_failure(graph, closure.FAIL_SCHEMA_V1)


def test_static_manifest_reorder_fails_against_policy_commitment() -> None:
    graph = _build_runtime_graph()

    def mutate(body: dict[str, object]) -> None:
        manifest = body["stdlib_manifest"]
        assert isinstance(manifest, dict)
        manifest["entries_in_exact_order"] = list(
            reversed(manifest["entries_in_exact_order"])  # type: ignore[arg-type]
        )
        for ordinal, row in enumerate(manifest["entries_in_exact_order"]):  # type: ignore[arg-type]
            row["ordinal"] = ordinal

    _assert_failure(_replace_evidence(graph, mutate), closure.FAIL_DIGEST_V1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("implementation", "PyPy"),
        ("implementation_version", "3.10.13"),
        ("language_version", "3.11"),
        ("abi_tag", "cp311"),
        ("cache_tag", "cpython-311"),
    ],
)
def test_cpython_31012_cp310_identity_is_exact(field: str, value: object) -> None:
    def mutate(body: dict[str, object]) -> None:
        identity = body["runtime_identity"]
        assert isinstance(identity, dict)
        identity[field] = value

    graph = _build_runtime_graph(policy_mutator=mutate)
    _assert_failure(graph, closure.FAIL_BINDING_V1)


def test_policy_and_evidence_canonical_transport_is_exact() -> None:
    graph = _build_runtime_graph()
    for field in (
        "runtime_closure_policy_payload",
        "image_closure_evidence_payload",
    ):
        changed = copy.deepcopy(graph)
        changed.verify_arguments[field] = graph.verify_arguments[field] + b"\n"
        _assert_failure(changed, closure.FAIL_INPUT_V1)


def test_oob_lock_pin_and_hybrid_graph_fail_closed() -> None:
    first = _build_runtime_graph(variant=None)
    second = _build_runtime_graph(variant="v8")
    hybrid = copy.deepcopy(first)
    hybrid.verify_arguments["authority_profile_payload"] = (
        second.verify_arguments["authority_profile_payload"]
    )
    hybrid.verify_arguments["signer_registry_payload"] = (
        second.verify_arguments["signer_registry_payload"]
    )
    _assert_failure(hybrid, closure.FAIL_TRUST_ANCHOR_V1)
    first.verify_arguments[
        "out_of_band_expected_lock_payload_sha256_digest"
    ] = b"\x01" * 32
    _assert_failure(first, closure.FAIL_TRUST_ANCHOR_V1)


def test_evidence_missing_extra_reorder_duplicate_hash_and_size_fail() -> None:
    graph = _build_runtime_graph()
    mutations: tuple[EvidenceMutator, ...] = (
        lambda body: body.__setitem__("extra", False),
        lambda body: body.__setitem__("authority_profile_id", graph.lock_id),
        lambda body: body["bound_artifacts_in_exact_order"].__setitem__(  # type: ignore[union-attr]
            1, copy.deepcopy(body["bound_artifacts_in_exact_order"][0])  # type: ignore[index]
        ),
        lambda body: body["oci_image"]["ordered_layer_descriptors"].reverse(),  # type: ignore[index,union-attr]
        lambda body: body["oci_image"].__setitem__("config_byte_size", 1),  # type: ignore[union-attr]
        lambda body: body["oci_image"].__setitem__(  # type: ignore[union-attr]
            "config_digest", "sha256:" + "1" * 64
        ),
    )
    for mutate in mutations:
        _assert_failure(
            _replace_evidence(graph, mutate),
            closure.FAIL_SCHEMA_V1,
            closure.FAIL_TRUST_ANCHOR_V1,
            closure.FAIL_BINDING_V1,
            closure.FAIL_OCI_GRAPH_V1,
        )


def test_exact_structural_types_reject_eqall_float_and_bool() -> None:
    assert closure._equal_exact(_EqAll(), 1) is False
    assert closure._equal_exact(True, 1) is False
    for value in (_EqAll(), 1.0, True):
        with pytest.raises(closure.Phase2BStep5RuntimeClosureErrorV1):
            closure._integer(value, "integer", minimum=1, maximum=1)
    with pytest.raises(closure.Phase2BStep5RuntimeClosureErrorV1):
        closure._object_exact(_EqAll(), frozenset(), "object")


def test_all_image_live_authority_and_readiness_claims_are_false() -> None:
    graph = _build_runtime_graph()
    result = _verify(graph)
    for field in (
        "service_image_exists",
        "service_image_pulled",
        "service_image_executed",
        "live_runtime_closure_observed",
        "entry_exit_closure_equal",
        "live_uid_gid_verified",
        "live_control_domain_verified",
        "key_custody_verified",
        "runtime_closure_signature_verified",
        "remote_attestation_verified",
        "external_cas_verified",
        "authority_evidence",
        "readiness_evidence",
        "r01_satisfied",
        "execution_authorized",
    ):
        assert getattr(result, field) is False


def test_result_constructor_copy_and_unissued_object_are_rejected() -> None:
    graph = _build_runtime_graph()
    result = _verify(graph)
    with pytest.raises(TypeError):
        closure.VerifiedOfflineImageClosureMechanicsV1()
    forged = object.__new__(closure.VerifiedOfflineImageClosureMechanicsV1)
    for field in result.__dataclass_fields__:
        object.__setattr__(forged, field, getattr(result, field))
    with pytest.raises(closure.Phase2BStep5RuntimeClosureErrorV1) as caught:
        closure.require_verified_offline_step5_image_closure_mechanics_v1(forged)
    assert caught.value.code == closure.FAIL_INPUT_V1
    try:
        copied = copy.copy(result)
    except TypeError:
        # Constructor-level rejection is stronger than producing an unissued copy.
        pass
    else:
        if copied is not result:
            with pytest.raises(closure.Phase2BStep5RuntimeClosureErrorV1):
                closure.require_verified_offline_step5_image_closure_mechanics_v1(copied)


def test_result_mutation_is_detected() -> None:
    result = _verify(_build_runtime_graph())
    object.__setattr__(result, "authority_evidence", True)
    with pytest.raises(closure.Phase2BStep5RuntimeClosureErrorV1) as caught:
        closure.require_verified_offline_step5_image_closure_mechanics_v1(result)
    assert caught.value.code in (
        closure.FAIL_INPUT_V1,
        closure.FAIL_CLAIM_BOUNDARY_V1,
    )


def test_import_time_authority_loader_decoder_and_hash_bindings_survive_rebind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _build_runtime_graph()
    monkeypatch.setattr(closure, "_authority_load_lock_v2", lambda *a, **k: None)
    monkeypatch.setattr(
        closure, "_authority_decode_canonical_json_v2", lambda *a, **k: {}
    )
    monkeypatch.setattr(
        closure,
        "hashlib",
        SimpleNamespace(sha256=lambda *a, **k: SimpleNamespace(hexdigest=lambda: "0" * 64)),
    )
    result = _verify(graph)
    assert result.offline_static_byte_graph_mechanics_verified is True
    graph.verify_arguments["raw_oci_config_payload"] = b"changed"
    _assert_failure(graph, closure.FAIL_DIGEST_V1)


def test_fresh_import_runtime_closure_survives_transitive_authority_export_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    importlib.reload(authority)
    importlib.reload(closure)
    graph = _build_runtime_graph()
    assert _verify(graph).offline_static_byte_graph_mechanics_verified is True

    scalar_replacements: dict[str, object] = {
        "AUTHORITY_PROFILE_SCHEMA_V2": "evil-authority-profile/999",
        "STAGE_B_TRUST_LOCK_SCHEMA_V2": "evil-stage-b-lock/999",
        "SIGNER_REGISTRY_SCHEMA_V2": "evil-registry/999",
        "DETACHED_SIGNATURE_SCHEMA_V2": "evil-signature/999",
        "SERVICE_STATIC_BINDING_SCHEMA_V2": "evil-service/999",
        "READINESS_CANDIDATE_SNAPSHOT_SCHEMA_V2": "evil-readiness/999",
        "ROLE_KEY_SCHEMA_V1": "evil-role-key/999",
        "AUTHORITY_PROFILE_PREFIX_V2": "evil_profile_",
        "STAGE_B_TRUST_LOCK_PREFIX_V2": "evil_lock_",
        "SIGNER_REGISTRY_PREFIX_V2": "evil_registry_",
        "SERVICE_STATIC_BINDING_PREFIX_V2": "evil_service_",
        "READINESS_CANDIDATE_PREFIX_V2": "evil_readiness_",
        "ROLE_KEY_PREFIX_V1": "evil_role_",
        "QV2_PROFILE_ID_V1": "evil_qv2_" + "f" * 64,
        "QV2_PROFILE_PAYLOAD_SHA256_V1": "f" * 64,
        "QV2_SOURCE_SHA256_V1": "e" * 64,
        "QV2_DEPENDENCY_LOCK_SHA256_V1": "d" * 64,
        "_MAX_CANONICAL_BYTES": 1,
        "_MAX_DEPTH": 0,
        "_MAX_NODES": 1,
        "_MAX_STRING_UTF8_BYTES": 1,
        "_SAFE_INTEGER": 0,
        "_PUBLIC_KEY_HEX_LENGTH": 1,
        "_SIGNATURE_HEX_LENGTH": 1,
        "_MAX_LAYERS": 0,
    }
    for name, replacement in scalar_replacements.items():
        monkeypatch.setattr(authority, name, replacement)

    monkeypatch.setattr(authority, "EXACT_QV2_BINDING_V1", {"evil": True})
    monkeypatch.setattr(authority, "ROLE_ORDER_V2", ("EVIL_ROLE",))
    monkeypatch.setattr(
        authority, "ROLE_CONTROL_DOMAINS_V2", {"EVIL_ROLE": "EVIL"}
    )
    monkeypatch.setattr(
        authority,
        "ROLE_SIGNATURE_PURPOSES_V2",
        {"EVIL_ROLE": ("SIGN_EVIL",)},
    )
    monkeypatch.setattr(authority, "SERVICE_ORDER_V2", ("evil_service",))
    monkeypatch.setattr(
        authority, "SERVICE_ROLE_V2", {"evil_service": "EVIL_ROLE"}
    )
    monkeypatch.setattr(
        authority, "SERVICE_TYPES_V2", {"evil_service": "EVIL"}
    )
    monkeypatch.setattr(
        authority, "SERVICE_UID_GID_V2", {"evil_service": (1, 1)}
    )
    monkeypatch.setattr(authority, "READINESS_NODE_ORDER_V1", ("EVIL_NODE",))
    monkeypatch.setattr(authority, "READINESS_DEPENDENCIES_V1", ((),))
    monkeypatch.setattr(authority, "_DOMAINS", {"authority_profile": b"EVIL\x00"})
    monkeypatch.setattr(authority, "_PREFIXES", {"authority_profile": "evil_"})
    monkeypatch.setattr(authority, "_SIGNATURE_DOMAINS", {"SIGN_EVIL": b"EVIL\x00"})
    monkeypatch.setattr(authority, "_HEX64_RE", re.compile(r".+"))
    monkeypatch.setattr(authority, "_COMMIT_RE", re.compile(r".+"))
    monkeypatch.setattr(authority, "_OCI_DIGEST_RE", re.compile(r".+"))
    monkeypatch.setattr(authority, "_ID_RE", re.compile(r".+"))
    monkeypatch.setattr(authority, "canonicalize_rfc8785_v1", lambda value: b"{}")
    monkeypatch.setattr(
        authority,
        "decode_and_require_canonical_rfc8785_v1",
        lambda payload: {},
    )
    monkeypatch.setattr(authority, "_Ed25519PublicKey", object)
    monkeypatch.setattr(authority, "_InvalidSignature", RuntimeError)
    for name, value in tuple(vars(authority).items()):
        if inspect.isfunction(value) and value.__module__ == authority.__name__:
            monkeypatch.setattr(authority, name, lambda *args, **kwargs: None)

    result = _verify(graph)
    assert result.offline_static_byte_graph_mechanics_verified is True
    assert result.mechanics_only is True
    assert all(
        getattr(result, field) is False
        for field in closure._PERMANENT_NONCLAIMS_FIELDS_V1
    )


def test_fully_readdressed_resigned_and_oob_repinned_alternate_authority_graph_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _build_runtime_graph()
    evil_profile_schema = "evil-authority-profile/999"
    evil_registry_schema = "evil-registry/999"
    evil_lock_schema = "evil-stage-b-lock/999"
    evil_signature_schema = "evil-signature/999"
    evil_qv2: dict[str, object] = {
        "schema_version": "evil-qv2/999",
        "profile_id": "evil_qv2_" + "a" * 64,
        "profile_payload_sha256": "b" * 64,
        "source_sha256": "c" * 64,
        "dependency_lock_sha256": "d" * 64,
    }
    evil = _readdress_resign_and_repin_alternate_authority_graph(
        graph,
        profile_schema=evil_profile_schema,
        registry_schema=evil_registry_schema,
        lock_schema=evil_lock_schema,
        signature_schema=evil_signature_schema,
        qv2_binding=evil_qv2,
    )

    decode = authority.decode_canonical_step5_json_v2
    profile_envelope = decode(evil.verify_arguments["authority_profile_payload"])
    registry_envelope = decode(evil.verify_arguments["signer_registry_payload"])
    lock_envelope = decode(evil.verify_arguments["stage_b_trust_lock_payload"])
    evidence_envelope = decode(
        evil.verify_arguments["image_closure_evidence_payload"]
    )
    assert profile_envelope["id"] == evil.profile_id
    assert registry_envelope["id"] == evil.registry_id
    assert lock_envelope["id"] == evil.lock_id
    assert profile_envelope["body"]["qv2_binding"] == evil_qv2
    assert lock_envelope["body"]["qv2_binding"] == evil_qv2
    assert evil.verify_arguments["out_of_band_expected_lock_id_bytes"] == (
        evil.lock_id.encode("ascii")
    )
    assert evil.verify_arguments[
        "out_of_band_expected_lock_payload_sha256_digest"
    ] == hashlib.sha256(
        evil.verify_arguments["stage_b_trust_lock_payload"]
    ).digest()
    assert evidence_envelope["body"]["stage_b_trust_lock_id"] == evil.lock_id
    assert evidence_envelope["body"]["authority_profile_id"] == evil.profile_id
    assert evidence_envelope["body"]["signer_registry_id"] == evil.registry_id
    signature = registry_envelope["signatures"][0]
    governance_key = registry_envelope["body"]["role_keys_in_exact_order"][0]
    Ed25519PrivateKey.from_private_bytes(b"\x01" * 32).public_key().verify(
        bytes.fromhex(signature["signature_hex"]),
        authority.phase2b_step5_signature_preimage_v2(
            artifact_kind="signer_registry",
            body=registry_envelope["body"],
            role="GOVERNANCE_ANCHOR",
            purpose="SIGN_SIGNER_REGISTRY",
            signer_key_id=governance_key["key_id"],
        ),
    )

    monkeypatch.setattr(
        authority, "AUTHORITY_PROFILE_SCHEMA_V2", evil_profile_schema
    )
    monkeypatch.setattr(
        authority, "SIGNER_REGISTRY_SCHEMA_V2", evil_registry_schema
    )
    monkeypatch.setattr(
        authority, "STAGE_B_TRUST_LOCK_SCHEMA_V2", evil_lock_schema
    )
    monkeypatch.setattr(
        authority, "DETACHED_SIGNATURE_SCHEMA_V2", evil_signature_schema
    )
    monkeypatch.setattr(authority, "EXACT_QV2_BINDING_V1", evil_qv2)

    _assert_failure(evil, closure.FAIL_TRUST_ANCHOR_V1)


def test_import_time_validator_binding_survives_global_helper_rebind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _build_runtime_graph(
        policy_mutator=lambda body: body.__setitem__("schema_version", "EVIL")
    )
    monkeypatch.setattr(closure, "_validate_runtime_closure_policy", lambda *a, **k: ({}, {}))
    _assert_failure(graph, closure.FAIL_SCHEMA_V1)


def test_import_time_schema_constant_cannot_be_rebound_to_admit_invalid_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _build_runtime_graph(
        policy_mutator=lambda body: body.__setitem__("schema_version", "EVIL")
    )
    monkeypatch.setattr(closure, "RUNTIME_CLOSURE_POLICY_SCHEMA_V1", "EVIL")
    _assert_failure(graph, closure.FAIL_SCHEMA_V1)


def test_import_time_evidence_schema_constant_cannot_admit_invalid_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _build_runtime_graph(
        evidence_mutator=lambda body: body.__setitem__("schema_version", "EVIL")
    )
    monkeypatch.setattr(closure, "IMAGE_CLOSURE_EVIDENCE_SCHEMA_V1", "EVIL")
    _assert_failure(graph, closure.FAIL_SCHEMA_V1)


def test_import_time_manifest_schema_constant_cannot_admit_invalid_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mutate(manifests: dict[str, dict[str, object]]) -> None:
        manifests["PYTHON_STDLIB"]["schema_version"] = "EVIL"

    graph = _build_runtime_graph(static_mutator=mutate)
    monkeypatch.setattr(closure, "STATIC_RUNTIME_MANIFEST_SCHEMA_V1", "EVIL")
    _assert_failure(graph, closure.FAIL_CLAIM_BOUNDARY_V1)


def test_import_time_hard_bounds_cannot_be_rebound_to_admit_oversized_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _build_runtime_graph(
        policy_mutator=lambda body: body["resource_bounds"].__setitem__(  # type: ignore[union-attr]
            "max_layers", 129
        )
    )
    monkeypatch.setattr(closure, "_HARD_MAX_LAYERS", 1000)
    _assert_failure(graph, closure.FAIL_SCHEMA_V1)


def test_import_time_regex_binding_cannot_be_rebound_to_admit_invalid_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mutate(body: dict[str, object]) -> None:
        body["oci_platform"]["architecture"] = "AMD64!"  # type: ignore[index]

    graph = _build_runtime_graph(policy_mutator=mutate)
    monkeypatch.setattr(
        closure,
        "re",
        SimpleNamespace(fullmatch=lambda *a, **k: object()),
    )
    _assert_failure(graph, closure.FAIL_SCHEMA_V1)
