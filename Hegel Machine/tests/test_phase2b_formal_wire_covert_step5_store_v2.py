from __future__ import annotations

from array import array
import copy
from dataclasses import dataclass, replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import sys

import pytest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hegel_machine import phase2b_formal_wire_covert_step5_authority_v2 as authority
from hegel_machine import phase2b_formal_wire_covert_step5_store_v2 as store


EXPECTED_STORE_SOURCE_SHA256 = (
    "6c545144dd584244d9405c92120e09c8ecb1d115c90e964a75fb3239b0b670b1"
)
SEED = bytes(range(32))
OTHER_SEED = bytes(reversed(range(32)))
ATTEMPT_ID = "attempt-context-store-v2-0001"
OTHER_ATTEMPT_ID = "attempt-context-store-v2-0002"
SESSION_ID = "retention-session-store-v2-0001"
OTHER_SESSION_ID = "retention-session-store-v2-0002"
CHAIN_ID = "phase2b-step5-cas-store-v2-chain-0001"
OTHER_CHAIN_ID = "phase2b-step5-cas-store-v2-chain-0002"
PLAN_KIND = "arbitrary-retention-plan-kind-v2"
DEFAULT_ROWS = (
    ("alpha-arbitrary-object", 1_024),
    ("beta-arbitrary-object", 2_048),
    ("gamma-arbitrary-object", 4_096),
)


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical(value: object) -> bytes:
    return authority.canonical_step5_json_bytes_v2(value)


def _qv2_binding() -> dict[str, object]:
    return dict(authority.EXACT_QV2_BINDING_V1)


def _owner_binding() -> dict[str, object]:
    return dict(authority._EXACT_OWNER_BINDING_V2)


def _threat_model() -> dict[str, object]:
    return dict(authority._THREAT_MODEL_V2)


def _permanent_nonclaims() -> dict[str, object]:
    return dict(authority._PERMANENT_NONCLAIMS_V2)


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


def _role_key(role: str, private_key: Ed25519PrivateKey) -> dict[str, object]:
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
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        .hex(),
    }
    row["key_id"] = authority.phase2b_step5_role_key_id_v2(row)
    return row


def _governance_signature(
    body: dict[str, object],
    governance_row: dict[str, object],
    governance_key: Ed25519PrivateKey,
) -> dict[str, object]:
    preimage = authority.phase2b_step5_signature_preimage_v2(
        artifact_kind="signer_registry",
        body=body,
        role="GOVERNANCE_ANCHOR",
        purpose="SIGN_SIGNER_REGISTRY",
        signer_key_id=governance_row["key_id"],  # type: ignore[arg-type]
        key_epoch=1,
    )
    return {
        "schema_version": authority.DETACHED_SIGNATURE_SCHEMA_V2,
        "algorithm": "Ed25519",
        "role": "GOVERNANCE_ANCHOR",
        "purpose": "SIGN_SIGNER_REGISTRY",
        "key_epoch": 1,
        "signer_key_id": governance_row["key_id"],
        "signature_hex": governance_key.sign(preimage).hex(),
    }


@dataclass(frozen=True)
class _AuthorityGraph:
    stage_b_lock: dict[str, object]
    stage_b_lock_payload: bytes
    authority_profile: dict[str, object]
    authority_profile_payload: bytes
    signer_registry: dict[str, object]
    signer_registry_payload: bytes
    out_of_band_expected_lock_id_bytes: bytes
    out_of_band_expected_lock_payload_sha256_digest: bytes
    cas_row: dict[str, object]
    role_rows: tuple[dict[str, object], ...]
    service_bindings: tuple[dict[str, object], ...]
    governance_private_seed: bytes


def _refresh_service_binding_id(binding: dict[str, object]) -> None:
    binding.pop("binding_id", None)
    binding["binding_id"] = authority.phase2b_step5_service_static_binding_id_v2(
        binding
    )


def _authority_graph(
    cas_seed: bytes = SEED,
    *,
    variant: str = "default",
) -> _AuthorityGraph:
    suffix = "default" if variant == "default" else variant
    governance_seed = hashlib.sha256(
        f"step5-store-v2-authority-v2-{suffix}-role-0".encode()
    ).digest()
    private_keys = tuple(
        Ed25519PrivateKey.from_private_bytes(
            cas_seed
            if role == "CAS_STORE"
            else (
                governance_seed
                if index == 0
                else hashlib.sha256(
                    f"step5-store-v2-authority-v2-{suffix}-role-{index}".encode()
                ).digest()
            )
        )
        for index, role in enumerate(authority.ROLE_ORDER_V2)
    )
    rows = tuple(
        _role_key(role, private_keys[index])
        for index, role in enumerate(authority.ROLE_ORDER_V2)
    )
    service_bindings: list[dict[str, object]] = []
    for index, service in enumerate(authority.SERVICE_ORDER_V2):
        role = authority.SERVICE_ROLE_V2[service]
        role_row = rows[authority.ROLE_ORDER_V2.index(role)]
        uid, gid = authority.SERVICE_UID_GID_V2[service]

        def oci(part: str) -> str:
            return "sha256:" + _h(
                f"store-v2-authority-v2-{suffix}-{service}-{part}"
            )

        binding: dict[str, object] = {
            "schema_version": authority.SERVICE_STATIC_BINDING_SCHEMA_V2,
            "service_name": service,
            "service_type": authority.SERVICE_TYPES_V2[service],
            "binding_status": "STATIC_REQUESTED_BINDING_NOT_OBSERVED_LIVE",
            "role": role,
            "control_domain": authority.ROLE_CONTROL_DOMAINS_V2[role],
            "uid": uid,
            "gid": gid,
            "role_key_id": role_row["key_id"],
            "key_epoch": 1,
            "authorized_signature_purposes": list(
                authority.ROLE_SIGNATURE_PURPOSES_V2[role]
            ),
            "oci_index_digest": oci("index"),
            "oci_manifest_digest": oci("manifest"),
            "oci_config_digest": oci("config"),
            "ordered_layer_digests": [oci("layer-0"), oci("layer-1")],
            "static_runtime_closure_policy_sha256": _h(
                f"store-v2-authority-v2-{suffix}-{service}-closure-policy"
            ),
        }
        _refresh_service_binding_id(binding)
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
        "owner_authorization_binding": _owner_binding(),
        "readiness_dag_binding": {
            "schema_version": "hegel-machine-phase2b-actual-unsealed-960-readiness-dag/1",
            "readiness_dag_id": authority.READINESS_DAG_ID_V1,
        },
        "readiness_state": _all_false_readiness(),
        "qv2_binding": _qv2_binding(),
        "source_binding": {
            "source_commit_sha": _h("store-v2-commit-" + suffix)[:40],
            "source_archive_sha256": _h("store-v2-source-" + suffix),
            "dependency_lock_sha256": _h("store-v2-lock-" + suffix),
        },
        "service_static_bindings_in_exact_order": copy.deepcopy(service_bindings),
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
        "governance_root": copy.deepcopy(rows[0]),
        "threat_model": _threat_model(),
        "permanent_nonclaims": _permanent_nonclaims(),
    }
    profile: dict[str, object] = {
        "body": profile_body,
        "id": authority.phase2b_step5_content_id_v2("authority_profile", profile_body),
    }
    profile_payload = _canonical(profile)
    registry_body: dict[str, object] = {
        "schema_version": authority.SIGNER_REGISTRY_SCHEMA_V2,
        "authority_profile_id": profile["id"],
        "registry_epoch": 2,
        "status": "CANDIDATE_NOT_ADMITTED",
        "claim_level": (
            "SIGNED_STATIC_KEY_REGISTRY_CANDIDATE_NOT_AUTHORITY_NOT_READINESS"
        ),
        "revocation_epoch": 0,
        "role_keys_in_exact_order": [copy.deepcopy(row) for row in rows],
        "readiness_state": _all_false_readiness(),
    }
    registry: dict[str, object] = {
        "body": registry_body,
        "id": authority.phase2b_step5_content_id_v2("signer_registry", registry_body),
        "signatures": [_governance_signature(registry_body, rows[0], private_keys[0])],
    }
    registry_payload = _canonical(registry)
    lock_body: dict[str, object] = {
        "schema_version": authority.STAGE_B_TRUST_LOCK_SCHEMA_V2,
        "status": "CANDIDATE_OUT_OF_BAND_STAGE_B_TRUST_LOCK",
        "claim_level": "CANDIDATE_GRAPH_PIN_ONLY_NOT_AUTHORITY_NOT_READINESS",
        "canonicalization": "RFC8785_EXACT_BYTES_NO_TRAILING_LF",
        "authority_profile_schema": authority.AUTHORITY_PROFILE_SCHEMA_V2,
        "authority_profile_id": profile["id"],
        "authority_profile_payload_sha256": hashlib.sha256(profile_payload).hexdigest(),
        "signer_registry_schema": authority.SIGNER_REGISTRY_SCHEMA_V2,
        "signer_registry_id": registry["id"],
        "signer_registry_payload_sha256": hashlib.sha256(registry_payload).hexdigest(),
        "qv2_binding": _qv2_binding(),
    }
    lock: dict[str, object] = {
        "body": lock_body,
        "id": authority.phase2b_step5_content_id_v2("stage_b_trust_lock", lock_body),
    }
    lock_payload = _canonical(lock)
    cas_row = rows[authority.ROLE_ORDER_V2.index("CAS_STORE")]
    return _AuthorityGraph(
        stage_b_lock=lock,
        stage_b_lock_payload=lock_payload,
        authority_profile=profile,
        authority_profile_payload=profile_payload,
        signer_registry=registry,
        signer_registry_payload=registry_payload,
        out_of_band_expected_lock_id_bytes=lock["id"].encode("ascii"),  # type: ignore[union-attr]
        out_of_band_expected_lock_payload_sha256_digest=hashlib.sha256(
            lock_payload
        ).digest(),
        cas_row=copy.deepcopy(cas_row),
        role_rows=tuple(copy.deepcopy(row) for row in rows),
        service_bindings=tuple(copy.deepcopy(row) for row in service_bindings),
        governance_private_seed=governance_seed,
    )


DEFAULT_GRAPH = _authority_graph()


def _repin_registry_graph(
    graph: _AuthorityGraph,
    mutate_cas_row,
) -> _AuthorityGraph:
    profile = copy.deepcopy(graph.authority_profile)
    registry_body = copy.deepcopy(graph.signer_registry["body"])
    assert type(registry_body) is dict
    rows = registry_body["role_keys_in_exact_order"]
    assert type(rows) is list
    cas_row = rows[authority.ROLE_ORDER_V2.index("CAS_STORE")]
    assert type(cas_row) is dict
    mutate_cas_row(cas_row)
    registry: dict[str, object] = {
        "body": registry_body,
        "id": authority.phase2b_step5_content_id_v2("signer_registry", registry_body),
    }
    governance_row = rows[0]
    assert type(governance_row) is dict
    governance_key = Ed25519PrivateKey.from_private_bytes(
        graph.governance_private_seed
    )
    registry["signatures"] = [
        _governance_signature(registry_body, governance_row, governance_key)
    ]
    profile_payload = _canonical(profile)
    registry_payload = _canonical(registry)
    lock_body: dict[str, object] = {
        "schema_version": authority.STAGE_B_TRUST_LOCK_SCHEMA_V2,
        "status": "CANDIDATE_OUT_OF_BAND_STAGE_B_TRUST_LOCK",
        "claim_level": "CANDIDATE_GRAPH_PIN_ONLY_NOT_AUTHORITY_NOT_READINESS",
        "canonicalization": "RFC8785_EXACT_BYTES_NO_TRAILING_LF",
        "authority_profile_schema": authority.AUTHORITY_PROFILE_SCHEMA_V2,
        "authority_profile_id": profile["id"],
        "authority_profile_payload_sha256": hashlib.sha256(profile_payload).hexdigest(),
        "signer_registry_schema": authority.SIGNER_REGISTRY_SCHEMA_V2,
        "signer_registry_id": registry["id"],
        "signer_registry_payload_sha256": hashlib.sha256(registry_payload).hexdigest(),
        "qv2_binding": _qv2_binding(),
    }
    lock: dict[str, object] = {
        "body": lock_body,
        "id": authority.phase2b_step5_content_id_v2("stage_b_trust_lock", lock_body),
    }
    lock_payload = _canonical(lock)
    return _AuthorityGraph(
        stage_b_lock=lock,
        stage_b_lock_payload=lock_payload,
        authority_profile=profile,
        authority_profile_payload=profile_payload,
        signer_registry=registry,
        signer_registry_payload=registry_payload,
        out_of_band_expected_lock_id_bytes=lock["id"].encode("ascii"),  # type: ignore[union-attr]
        out_of_band_expected_lock_payload_sha256_digest=hashlib.sha256(
            lock_payload
        ).digest(),
        cas_row=copy.deepcopy(cas_row),
        role_rows=tuple(copy.deepcopy(row) for row in rows),
        service_bindings=tuple(
            copy.deepcopy(row)
            for row in profile["body"]["service_static_bindings_in_exact_order"]  # type: ignore[index]
        ),
        governance_private_seed=graph.governance_private_seed,
    )


def _legacy_v1_raw_graph() -> tuple[bytes, bytes, bytes, bytes, bytes]:
    """Build one real valid V1 triple solely to prove V2 rejects it."""

    from hegel_machine import (  # Lazy: clean Store V2 import must not load V1.
        phase2b_formal_wire_covert_step5_authority_v1 as legacy,
    )

    private_keys = tuple(
        Ed25519PrivateKey.from_private_bytes(
            SEED
            if role == "CAS_STORE"
            else hashlib.sha256(f"legacy-store-v1-role-{index}".encode()).digest()
        )
        for index, role in enumerate(legacy.ROLE_ORDER_V1)
    )
    rows: list[dict[str, object]] = []
    for index, role in enumerate(legacy.ROLE_ORDER_V1):
        key = private_keys[index]
        row: dict[str, object] = {
            "schema_version": "hegel-machine-phase2b-step5-role-key/1",
            "role": role,
            "control_domain": legacy.ROLE_CONTROL_DOMAINS_V1[role],
            "authorized_signature_purposes": list(
                legacy.ROLE_SIGNATURE_PURPOSES_V1[role]
            ),
            "algorithm": "Ed25519",
            "key_epoch": 1,
            "public_key_hex": key.public_key()
            .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
            .hex(),
        }
        row["key_id"] = legacy.phase2b_step5_role_key_id_v1(row)
        rows.append(row)
    profile_body: dict[str, object] = {
        "schema_version": legacy.AUTHORITY_PROFILE_SCHEMA_V1,
        "status": "FROZEN_OWNER_ACCEPTED_TECHNICAL_AUTHORITY_PROFILE",
        "claim_level": (
            "SAME_ADMIN_TECHNICAL_AUTHORITY_NOT_EXTERNAL_OR_ORGANIZATIONAL_INDEPENDENCE"
        ),
        "canonicalization": "RFC8785_EXACT_BYTES_NO_TRAILING_LF",
        "profile_epoch": 1,
        "owner_authorization_binding": _owner_binding(),
        "readiness_dag_binding": {
            "schema_version": "hegel-machine-phase2b-actual-unsealed-960-readiness-dag/1",
            "readiness_dag_id": legacy.READINESS_DAG_ID_V1,
        },
        "qv2_binding": _qv2_binding(),
        "source_binding": {
            "source_commit_sha": "a" * 40,
            "source_archive_sha256": _h("legacy-v1-source"),
            "dependency_lock_sha256": _h("legacy-v1-lock"),
        },
        "runtime_image_bindings": {
            "runner": "sha256:" + _h("legacy-v1-runner"),
            "pure_replayer": "sha256:" + _h("legacy-v1-replayer"),
            "cas_store": "sha256:" + _h("legacy-v1-cas"),
            "logical_time": "sha256:" + _h("legacy-v1-time"),
        },
        "role_policy": {
            "roles_in_exact_order": list(legacy.ROLE_ORDER_V1),
            "role_count": 8,
            "distinct_public_keys": True,
            "distinct_key_ids": True,
            "distinct_control_domains": True,
            "key_epoch": 1,
        },
        "governance_root": copy.deepcopy(rows[0]),
        "threat_model": copy.deepcopy(legacy._THREAT_MODEL),
        "permanent_nonclaims": copy.deepcopy(legacy._PERMANENT_NONCLAIMS),
    }
    profile_id = legacy.phase2b_step5_content_id_v1(
        "authority_profile", profile_body
    )
    profile_payload = legacy.canonical_step5_json_bytes_v1(
        {"body": profile_body, "id": profile_id}
    )
    registry_body: dict[str, object] = {
        "schema_version": legacy.SIGNER_REGISTRY_SCHEMA_V1,
        "authority_profile_id": profile_id,
        "registry_epoch": 1,
        "status": "ACTIVE",
        "revocation_epoch": 0,
        "role_keys_in_exact_order": copy.deepcopy(rows),
    }
    registry_id = legacy.phase2b_step5_content_id_v1(
        "signer_registry", registry_body
    )
    preimage = legacy.phase2b_step5_signature_preimage_v1(
        artifact_kind="signer_registry",
        body=registry_body,
        role="GOVERNANCE_ANCHOR",
        purpose="SIGN_SIGNER_REGISTRY",
        signer_key_id=rows[0]["key_id"],  # type: ignore[arg-type]
        key_epoch=1,
    )
    registry_payload = legacy.canonical_step5_json_bytes_v1(
        {
            "body": registry_body,
            "id": registry_id,
            "signatures": [
                {
                    "schema_version": legacy.DETACHED_SIGNATURE_SCHEMA_V1,
                    "algorithm": "Ed25519",
                    "role": "GOVERNANCE_ANCHOR",
                    "purpose": "SIGN_SIGNER_REGISTRY",
                    "key_epoch": 1,
                    "signer_key_id": rows[0]["key_id"],
                    "signature_hex": private_keys[0].sign(preimage).hex(),
                }
            ],
        }
    )
    lock_body: dict[str, object] = {
        "schema_version": legacy.STAGE_B_TRUST_LOCK_SCHEMA_V1,
        "status": "FROZEN_OUT_OF_BAND_STAGE_B_TRUST_LOCK",
        "canonicalization": "RFC8785_EXACT_BYTES_NO_TRAILING_LF",
        "authority_profile_id": profile_id,
        "authority_profile_payload_sha256": hashlib.sha256(profile_payload).hexdigest(),
        "signer_registry_id": registry_id,
        "signer_registry_payload_sha256": hashlib.sha256(registry_payload).hexdigest(),
        "qv2_binding": _qv2_binding(),
    }
    lock_id = legacy.phase2b_step5_content_id_v1(
        "stage_b_trust_lock", lock_body
    )
    lock_payload = legacy.canonical_step5_json_bytes_v1(
        {"body": lock_body, "id": lock_id}
    )
    return (
        lock_payload,
        profile_payload,
        registry_payload,
        lock_id.encode("ascii"),
        hashlib.sha256(lock_payload).digest(),
    )


def _plan(
    graph: _AuthorityGraph = DEFAULT_GRAPH,
    *,
    rows: tuple[tuple[str, int], ...] = DEFAULT_ROWS,
    attempt: str = ATTEMPT_ID,
    session: str = SESSION_ID,
    chain: str = CHAIN_ID,
    kind: str = PLAN_KIND,
) -> bytes:
    return store.build_retention_plan_v1(
        authority_profile_id=graph.authority_profile["id"],  # type: ignore[arg-type]
        signer_registry_id=graph.signer_registry["id"],  # type: ignore[arg-type]
        attempt_context_id=attempt,
        retention_session_id=session,
        chain_id=chain,
        plan_kind=kind,
        objects_in_exact_order=rows,
    )


def _new_store(
    root: Path,
    *,
    plan_payload: bytes | None = None,
    graph: _AuthorityGraph = DEFAULT_GRAPH,
    seed: bytes = SEED,
    create: bool = True,
) -> store.CasStoreV2:
    return store.CasStoreV2(
        root_directory=str(root),
        retention_plan_payload=_plan(graph) if plan_payload is None else plan_payload,
        private_key_seed=seed,
        allowed_peer_uids=frozenset({os.geteuid()}),
        stage_b_lock_payload=graph.stage_b_lock_payload,
        authority_profile_payload=graph.authority_profile_payload,
        signer_registry_payload=graph.signer_registry_payload,
        out_of_band_expected_lock_id_bytes=(
            graph.out_of_band_expected_lock_id_bytes
        ),
        out_of_band_expected_lock_payload_sha256_digest=(
            graph.out_of_band_expected_lock_payload_sha256_digest
        ),
        create=create,
    )


def _peer(*, pid: int | None = None) -> store.PeerCredentialsV2:
    return store.PeerCredentialsV2(
        pid=os.getpid() if pid is None else pid,
        uid=os.geteuid(),
        gid=os.getegid(),
    )


def _request_bytes(plan_payload: bytes, ordinal: int, payload: bytes) -> bytes:
    return store.build_generic_store_request_v1(
        retention_plan_payload=plan_payload,
        ordinal=ordinal,
        bound_artifact_id=f"artifact-v2-{ordinal:04d}-{_h(str(ordinal))}",
        bound_artifact_sha256=_sha256(payload),
        reported_size_bytes=len(payload),
    )


def _open_payload(tmp_path: Path, label: str, payload: bytes) -> tuple[Path, int]:
    path = tmp_path / label
    path.write_bytes(payload)
    return path, os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)


@dataclass(frozen=True)
class _Evidence:
    plan: bytes
    requests: tuple[bytes, ...]
    objects: tuple[bytes, ...]
    receipts: tuple[bytes, ...]
    checkpoints: tuple[bytes, ...]
    heads: tuple[bytes, ...]
    responses: tuple[bytes, ...]


def _complete(
    tmp_path: Path,
    *,
    graph: _AuthorityGraph = DEFAULT_GRAPH,
    rows: tuple[tuple[str, int], ...] = DEFAULT_ROWS,
    attempt: str = ATTEMPT_ID,
    session: str = SESSION_ID,
    chain: str = CHAIN_ID,
    payloads: tuple[bytes, ...] | None = None,
) -> _Evidence:
    plan_payload = _plan(
        graph,
        rows=rows,
        attempt=attempt,
        session=session,
        chain=chain,
    )
    actual_payloads = (
        tuple(f"store-v2-object-{index}".encode() for index in range(len(rows)))
        if payloads is None
        else payloads
    )
    assert len(actual_payloads) == len(rows)
    requests: list[bytes] = []
    receipts: list[bytes] = []
    checkpoints: list[bytes] = []
    heads: list[bytes] = []
    responses: list[bytes] = []
    root = tmp_path / f"cas-{attempt}-{session}-{chain}"
    with _new_store(root, plan_payload=plan_payload, graph=graph) as cas:
        for ordinal, payload in enumerate(actual_payloads):
            request_payload = _request_bytes(plan_payload, ordinal, payload)
            request = store.load_generic_store_request_v1(request_payload)
            _, fd = _open_payload(tmp_path, f"input-{chain}-{ordinal}.bin", payload)
            try:
                response = cas.ingest_open_fd(request, fd, _peer())
            finally:
                os.close(fd)
            requests.append(request_payload)
            receipts.append(
                response.checkpoint_record.signed_receipt.canonical_bytes()
            )
            checkpoints.append(response.checkpoint_record.canonical_bytes())
            heads.append(response.head_record.canonical_bytes())
            responses.append(response.canonical_bytes())
        assert len(cas.audit_complete_session()) == len(rows)
    return _Evidence(
        plan=plan_payload,
        requests=tuple(requests),
        objects=actual_payloads,
        receipts=tuple(receipts),
        checkpoints=tuple(checkpoints),
        heads=tuple(heads),
        responses=tuple(responses),
    )


def _verify(evidence: _Evidence, graph: _AuthorityGraph = DEFAULT_GRAPH):
    return store.verify_cas_store_complete_session_v2(
        retention_plan_payload=evidence.plan,
        request_payloads_in_exact_order=evidence.requests,
        object_payloads_in_exact_order=evidence.objects,
        signed_receipt_payloads_in_exact_order=evidence.receipts,
        checkpoint_payloads_in_exact_order=evidence.checkpoints,
        head_payloads_in_exact_order=evidence.heads,
        response_payloads_in_exact_order=evidence.responses,
        stage_b_lock_payload=graph.stage_b_lock_payload,
        authority_profile_payload=graph.authority_profile_payload,
        signer_registry_payload=graph.signer_registry_payload,
        out_of_band_expected_lock_id_bytes=(
            graph.out_of_band_expected_lock_id_bytes
        ),
        out_of_band_expected_lock_payload_sha256_digest=(
            graph.out_of_band_expected_lock_payload_sha256_digest
        ),
    )


class _EqAll:
    def __eq__(self, other: object) -> bool:
        return True

    def __hash__(self) -> int:
        return hash("CREATED_IF_ABSENT")


class _StrSubclass(str):
    pass


def test_exact_source_pin_public_api_and_nonauthority_boundaries() -> None:
    source = Path(store.__file__).read_bytes()
    assert hashlib.sha256(source).hexdigest() == EXPECTED_STORE_SOURCE_SHA256
    assert store.SCHEMA_VERSION.endswith("/2")
    assert store.STORE_REQUEST_SCHEMA_VERSION.endswith("/1")
    assert store.INSPECTED_AUTHORITY_V2_SOURCE_SHA256 == (
        "03d689327ca422f93e5a32db0a7f813fe84b1099f1bac1b31395c975d76d8e86"
    )
    assert not hasattr(store, "INSPECTED_AUTHORITY_V1_SOURCE_SHA256")
    assert store.CAS_STORE_ROLE == "CAS_STORE"
    assert store.CAS_STORE_CONTROL_DOMAIN == "STEP5_CAS_STORE_CONTROL"
    assert store.CAS_STORE_RECEIPT_PURPOSE == "SIGN_CAS_IF_ABSENT_RECEIPT"
    assert "MECHANICS_ONLY" in store.AUTHORITY_CLAIM_BOUNDARY
    assert "NOT_AUTHORITY" in store.AUTHORITY_CLAIM_BOUNDARY
    assert "STATIC_AUTHORITY_V2_SERVICE_BINDINGS_NOT_LIVE" in (
        store.AUTHORITY_CLAIM_BOUNDARY
    )
    assert "NOT_ANTI_ROLLBACK_OR_ANTI_CLONE" in store.AUTHORITY_CLAIM_BOUNDARY
    assert "EXTERNAL_ANCHOR" in store.FAIL_CLOSED_ROLLBACK_DISCLOSURE
    assert store.REQUIRED_EXTERNAL_CONTROLS == (
        "PINNED_CAS_STORE_SOURCE_IMAGE_AND_RUNTIME_CLOSURE",
        "DISTINCT_LIVE_CAS_STORE_UID_AND_CONTROL_DOMAIN",
        "CAS_STORE_PRIVATE_KEY_CUSTODY_OUTSIDE_PRODUCER",
        "INDEPENDENT_DURABLE_STORAGE_AND_EXTERNAL_COMPLETE_HEAD_ANCHOR",
        "FRESH_ATTEMPT_CONTEXT_AND_RETENTION_PLAN_AUTHORIZATION",
        "PRODUCER_FREE_PURE_REPLAY_IN_A_PINNED_DISTINCT_RUNTIME",
    )

    parameters = inspect.signature(
        store.verify_cas_store_complete_session_v2
    ).parameters
    assert tuple(parameters) == (
        "retention_plan_payload",
        "request_payloads_in_exact_order",
        "object_payloads_in_exact_order",
        "signed_receipt_payloads_in_exact_order",
        "checkpoint_payloads_in_exact_order",
        "head_payloads_in_exact_order",
        "response_payloads_in_exact_order",
        "stage_b_lock_payload",
        "authority_profile_payload",
        "signer_registry_payload",
        "out_of_band_expected_lock_id_bytes",
        "out_of_band_expected_lock_payload_sha256_digest",
    )
    assert not any("callback" in name or "verifier" in name for name in parameters)
    assert tuple(
        inspect.signature(
            store.require_verified_cas_store_complete_session_v2
        ).parameters
    ) == ("candidate",)
    assert "require_verified_cas_store_complete_session_v2" in store.__all__
    assert "replacing its provisional authority/profile V1 dependency" in (
        inspect.getdoc(store) or ""
    )


def test_clean_import_has_no_producer_qv2_runtime_or_v1_store_imports() -> None:
    code = r'''
import json, sys
from hegel_machine import phase2b_formal_wire_covert_step5_store_v2
print(json.dumps(sorted(sys.modules)))
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(store.__file__).parents[1])},
    )
    modules = json.loads(completed.stdout)
    forbidden = (
        "phase2b_formal_wire_covert_step5_runtime_v1",
        "phase2b_formal_wire_covert_step5_store_v1",
        "phase2b_formal_wire_covert_step5_owner_status_v1",
        "phase2b_formal_wire_covert_step5_authority_v1",
        "phase2b_formal_wire_covert_qualification_v2",
        "phase2b_formal_wire_covert_auditor",
    )
    assert not any(any(token in name for token in forbidden) for name in modules)


def test_retention_plan_is_canonical_content_addressed_and_generic() -> None:
    payload = _plan()
    plan = store.load_retention_plan_v1(payload)
    assert plan.canonical_bytes() == payload
    assert plan.expected_object_count == 3
    assert tuple(
        (row.ordinal, row.artifact_kind, row.maximum_bytes)
        for row in plan.objects_in_exact_order
    ) == (
        (0, "alpha-arbitrary-object", 1_024),
        (1, "beta-arbitrary-object", 2_048),
        (2, "gamma-arbitrary-object", 4_096),
    )
    assert plan.complete_head_required is True
    assert plan.plan_id.startswith(store.PLAN_ID_PREFIX)
    assert len(plan.plan_id) == len(store.PLAN_ID_PREFIX) + 64
    assert _plan(rows=(("one-object-kind", 7),)) != payload
    assert _plan(attempt=OTHER_ATTEMPT_ID) != payload
    assert _plan(session=OTHER_SESSION_ID) != payload
    assert _plan(chain=OTHER_CHAIN_ID) != payload
    assert _plan(kind="different-arbitrary-plan-kind") != payload
    maximum_payload = _plan(
        rows=(("maximum-sized-plan-entry", store.MAXIMUM_PLANNED_OBJECT_BYTES),)
    )
    assert (
        store.load_retention_plan_v1(maximum_payload).entry(0).maximum_bytes
        == store.MAXIMUM_PLANNED_OBJECT_BYTES
    )
    with pytest.raises(store.CasStoreV2Error):
        store.load_retention_plan_v1(payload + b"\n")
    with pytest.raises(store.CasStoreV2Error):
        store.load_retention_plan_v1(bytearray(payload))  # type: ignore[arg-type]
    with pytest.raises(store.CasStoreV2Error):
        store.build_retention_plan_v1(
            authority_profile_id=plan.authority_profile_id,
            signer_registry_id=plan.signer_registry_id,
            attempt_context_id=ATTEMPT_ID,
            retention_session_id=SESSION_ID,
            chain_id=CHAIN_ID,
            plan_kind=PLAN_KIND,
            objects_in_exact_order=(),
        )
    with pytest.raises(store.CasStoreV2Error):
        store.build_retention_plan_v1(
            authority_profile_id=plan.authority_profile_id,
            signer_registry_id=plan.signer_registry_id,
            attempt_context_id=ATTEMPT_ID,
            retention_session_id=SESSION_ID,
            chain_id=CHAIN_ID,
            plan_kind=PLAN_KIND,
            objects_in_exact_order=(("too-large-object", 1 << 41),),
        )
    duplicate_key = b'{"schema_version":"duplicate",' + payload[1:]
    with pytest.raises(store.CasStoreV2Error):
        store.load_retention_plan_v1(duplicate_key)


def test_plan_id_rejects_rehashed_content_drift_and_noncanonical_numbers() -> None:
    payload = _plan(rows=(("single-object-kind", 1_024),))
    mapping = json.loads(payload)
    mapping["objects_in_exact_order"][0]["maximum_bytes"] = 1_025
    with pytest.raises(store.CasStoreV2Error):
        store.load_retention_plan_v1(_canonical(mapping))
    # Even when a malicious caller recomputes a syntactically shaped ID with
    # the wrong domain, the exact frozen plan-ID derivation rejects it.
    mapping["plan_id"] = store.PLAN_ID_PREFIX + _h("wrong-plan-domain")
    with pytest.raises(store.CasStoreV2Error):
        store.load_retention_plan_v1(_canonical(mapping))
    noncanonical = payload.replace(b'"maximum_bytes":1024', b'"maximum_bytes":1024.0')
    assert noncanonical != payload
    with pytest.raises(store.CasStoreV2Error):
        store.load_retention_plan_v1(noncanonical)


@pytest.mark.parametrize("bad", [0.0, 1.0, True, False, "0", _EqAll()])
def test_planned_object_integer_types_are_exact(bad: object) -> None:
    with pytest.raises(store.CasStoreV2Error):
        store.PlannedObjectV1(
            ordinal=bad,  # type: ignore[arg-type]
            artifact_kind="planned-object-kind",
            maximum_bytes=1,
        )
    with pytest.raises(store.CasStoreV2Error):
        store.PlannedObjectV1(
            ordinal=0,
            artifact_kind="planned-object-kind",
            maximum_bytes=bad,  # type: ignore[arg-type]
        )


def test_string_bool_float_and_eqall_do_not_cross_direct_dto_boundaries(
    tmp_path: Path,
) -> None:
    plan_payload = _plan(rows=(("single-object-kind", 1_024),))
    plan = store.load_retention_plan_v1(plan_payload)
    payload = b"typed-object"
    request = store.load_generic_store_request_v1(_request_bytes(plan_payload, 0, payload))
    with pytest.raises(store.CasStoreV2Error):
        replace(request, ordinal=0.0)  # type: ignore[arg-type]
    with pytest.raises(store.CasStoreV2Error):
        replace(request, artifact_kind=_StrSubclass(request.artifact_kind))
    with pytest.raises(store.CasStoreV2Error):
        replace(request, fd_count=True)
    with pytest.raises(store.CasStoreV2Error):
        replace(plan, expected_object_count=1.0)  # type: ignore[arg-type]
    with pytest.raises(store.CasStoreV2Error):
        replace(plan, complete_head_required=1)  # type: ignore[arg-type]
    with pytest.raises(store.CasStoreV2Error):
        replace(plan, plan_kind=_StrSubclass(plan.plan_kind))
    with pytest.raises(store.CasStoreV2Error):
        store.PeerCredentialsV2(pid=1.0, uid=0, gid=0)  # type: ignore[arg-type]
    with pytest.raises(store.CasStoreV2Error):
        store.PeerCredentialsV2(pid=1, uid=_EqAll(), gid=0)  # type: ignore[arg-type]

    root = tmp_path / "typed-store"
    with _new_store(root, plan_payload=plan_payload) as cas:
        assert str(cas.authority_profile_id).startswith(
            authority.AUTHORITY_PROFILE_PREFIX_V2
        )
        assert str(cas.signer_registry_id).startswith(
            authority.SIGNER_REGISTRY_PREFIX_V2
        )
        _, fd = _open_payload(tmp_path, "typed.bin", payload)
        try:
            response = cas.ingest_open_fd(request, fd, _peer())
        finally:
            os.close(fd)
    binding = response.binding
    body = response.checkpoint_record.signed_receipt.signed_body
    head = response.head_record
    with pytest.raises(store.CasStoreV2Error):
        replace(binding, observed_object_count=1.0)  # type: ignore[arg-type]
    with pytest.raises(store.CasStoreV2Error):
        replace(binding, complete=1)  # type: ignore[arg-type]
    with pytest.raises(store.CasStoreV2Error):
        replace(body, peer_pid=1.0)  # type: ignore[arg-type]
    with pytest.raises(store.CasStoreV2Error):
        replace(body, object_disposition=_EqAll())  # type: ignore[arg-type]
    with pytest.raises(store.CasStoreV2Error):
        replace(head, purpose=_StrSubclass(store.CAS_STORE_HEAD_PURPOSE))
    assert plan.entry(0).maximum_bytes == 1_024


def test_raw_oob_authority_bootstrap_binds_exact_cas_role_key_and_purpose(
    tmp_path: Path,
) -> None:
    plan_payload = _plan(rows=(("single-object-kind", 1_024),))
    root = tmp_path / "authority-store"
    with _new_store(root, plan_payload=plan_payload) as cas:
        assert cas.authority_profile_id == DEFAULT_GRAPH.authority_profile["id"]
        assert cas.signer_registry_id == DEFAULT_GRAPH.signer_registry["id"]
        assert cas.signer_key_id == DEFAULT_GRAPH.cas_row["key_id"]
        assert cas.signer_public_key.hex() == DEFAULT_GRAPH.cas_row["public_key_hex"]
        assert cas.key_epoch == 1
        assert DEFAULT_GRAPH.cas_row["role"] == store.CAS_STORE_ROLE
        assert (
            DEFAULT_GRAPH.cas_row["control_domain"]
            == store.CAS_STORE_CONTROL_DOMAIN
        )
        assert DEFAULT_GRAPH.cas_row["authorized_signature_purposes"] == [
            store.CAS_STORE_RECEIPT_PURPOSE
        ]
        for role in (
            "GOVERNANCE_ANCHOR",
            "SOURCE_RUNTIME_CUSTODIAN",
            "PURE_REPLAYER",
        ):
            row = DEFAULT_GRAPH.role_rows[authority.ROLE_ORDER_V2.index(role)]
            assert tuple(row["authorized_signature_purposes"]) == (
                authority.ROLE_SIGNATURE_PURPOSES_V2[role]
            )
        cas_service = DEFAULT_GRAPH.service_bindings[
            authority.SERVICE_ORDER_V2.index("cas_store")
        ]
        assert cas_service["binding_status"] == (
            "STATIC_REQUESTED_BINDING_NOT_OBSERVED_LIVE"
        )
        assert cas_service["role_key_id"] == DEFAULT_GRAPH.cas_row["key_id"]
        assert cas_service["authorized_signature_purposes"] == [
            store.CAS_STORE_RECEIPT_PURPOSE
        ]
        assert DEFAULT_GRAPH.authority_profile["body"]["readiness_state"][  # type: ignore[index]
            "all_nodes_satisfied"
        ] is False

    base = dict(
        root_directory=str(tmp_path / "bad-oob-store"),
        retention_plan_payload=plan_payload,
        private_key_seed=SEED,
        allowed_peer_uids=frozenset({os.geteuid()}),
        stage_b_lock_payload=DEFAULT_GRAPH.stage_b_lock_payload,
        authority_profile_payload=DEFAULT_GRAPH.authority_profile_payload,
        signer_registry_payload=DEFAULT_GRAPH.signer_registry_payload,
        out_of_band_expected_lock_id_bytes=(
            DEFAULT_GRAPH.out_of_band_expected_lock_id_bytes
        ),
        out_of_band_expected_lock_payload_sha256_digest=(
            DEFAULT_GRAPH.out_of_band_expected_lock_payload_sha256_digest
        ),
        create=True,
    )
    with pytest.raises(store.CasStoreV2Error) as wrong_id:
        store.CasStoreV2(
            **{**base, "out_of_band_expected_lock_id_bytes": "not-raw-bytes"}
        )
    assert wrong_id.value.code == store.FAIL_AUTHORITY
    with pytest.raises(store.CasStoreV2Error) as wrong_hash:
        store.CasStoreV2(
            **{
                **base,
                "out_of_band_expected_lock_payload_sha256_digest": "0" * 64,
            }
        )
    assert wrong_hash.value.code == store.FAIL_AUTHORITY
    with pytest.raises(store.CasStoreV2Error):
        store.CasStoreV2(**{**base, "stage_b_lock_payload": bytearray(base["stage_b_lock_payload"])})  # type: ignore[arg-type]


def test_real_authority_v1_raw_graph_is_rejected_by_store_v2(tmp_path: Path) -> None:
    (
        lock_payload,
        profile_payload,
        registry_payload,
        lock_id_bytes,
        lock_digest,
    ) = _legacy_v1_raw_graph()
    with pytest.raises(store.CasStoreV2Error):
        store.build_retention_plan_v1(
            authority_profile_id="phase2b_step5_authority_profile_v1_" + ("a" * 64),
            signer_registry_id="phase2b_step5_signer_registry_v1_" + ("b" * 64),
            attempt_context_id=ATTEMPT_ID,
            retention_session_id=SESSION_ID,
            chain_id=CHAIN_ID,
            plan_kind=PLAN_KIND,
            objects_in_exact_order=(("single-object-kind", 1_024),),
        )
    with pytest.raises(store.CasStoreV2Error) as rejected:
        store.CasStoreV2(
            root_directory=str(tmp_path / "legacy-v1-rejected"),
            retention_plan_payload=_plan(rows=(("single-object-kind", 1_024),)),
            private_key_seed=SEED,
            allowed_peer_uids=frozenset({os.geteuid()}),
            stage_b_lock_payload=lock_payload,
            authority_profile_payload=profile_payload,
            signer_registry_payload=registry_payload,
            out_of_band_expected_lock_id_bytes=lock_id_bytes,
            out_of_band_expected_lock_payload_sha256_digest=lock_digest,
            create=True,
        )
    assert rejected.value.code == store.FAIL_AUTHORITY


def test_store_captures_frozen_authority_v2_entrypoints_at_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rebound(*args: object, **kwargs: object) -> object:
        raise AssertionError("rebound authority module entrypoint was reached")

    for name in (
        "load_step5_stage_b_trust_lock_v2",
        "load_step5_authority_profile_v2",
        "load_step5_signer_registry_v2",
        "require_locked_step5_authority_graph_v2",
        "phase2b_step5_role_key_id_v2",
    ):
        monkeypatch.setattr(authority, name, rebound)
    plan_payload = _plan(rows=(("single-object-kind", 1_024),))
    with _new_store(
        tmp_path / "captured-authority-v2-entrypoints",
        plan_payload=plan_payload,
    ) as cas:
        assert cas.authority_profile_id == DEFAULT_GRAPH.authority_profile["id"]
        assert cas.signer_registry_id == DEFAULT_GRAPH.signer_registry["id"]


def test_store_bootstrap_uses_authority_definition_time_contract_after_export_rebinding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A consumer cannot launder a new authority contract through module exports."""

    def rebound(*args: object, **kwargs: object) -> object:
        raise AssertionError("rebound authority helper was reached")

    scalar_exports = {
        "AUTHORITY_PROFILE_SCHEMA_V2": "evil-authority-profile/999",
        "STAGE_B_TRUST_LOCK_SCHEMA_V2": "evil-stage-b-lock/999",
        "SIGNER_REGISTRY_SCHEMA_V2": "evil-signer-registry/999",
        "DETACHED_SIGNATURE_SCHEMA_V2": "evil-detached-signature/999",
        "SERVICE_STATIC_BINDING_SCHEMA_V2": "evil-service-binding/999",
        "ROLE_KEY_SCHEMA_V1": "evil-role-key/999",
        "AUTHORITY_PROFILE_PREFIX_V2": "evil_authority_profile_",
        "STAGE_B_TRUST_LOCK_PREFIX_V2": "evil_stage_b_lock_",
        "SIGNER_REGISTRY_PREFIX_V2": "evil_signer_registry_",
        "ROLE_KEY_PREFIX_V1": "evil_role_key_",
        "QV2_PROFILE_ID_V1": "evil-qv2-profile",
        "QV2_PROFILE_PAYLOAD_SHA256_V1": "e" * 64,
        "QV2_SOURCE_SHA256_V1": "e" * 64,
        "QV2_DEPENDENCY_LOCK_SHA256_V1": "e" * 64,
    }
    for name, value in scalar_exports.items():
        monkeypatch.setattr(authority, name, value)

    monkeypatch.setattr(authority, "EXACT_QV2_BINDING_V1", {"evil": True})
    monkeypatch.setattr(authority, "ROLE_ORDER_V2", ("EVIL_ROLE",))
    monkeypatch.setattr(authority, "ROLE_CONTROL_DOMAINS_V2", {"EVIL_ROLE": "EVIL"})
    monkeypatch.setattr(
        authority,
        "ROLE_SIGNATURE_PURPOSES_V2",
        {"EVIL_ROLE": ("SIGN_EVIL",)},
    )
    monkeypatch.setattr(authority, "SERVICE_ORDER_V2", ("evil_service",))
    monkeypatch.setattr(authority, "SERVICE_ROLE_V2", {"evil_service": "EVIL_ROLE"})
    monkeypatch.setattr(authority, "SERVICE_TYPES_V2", {"evil_service": "EVIL"})
    monkeypatch.setattr(authority, "SERVICE_UID_GID_V2", {"evil_service": (1, 1)})
    monkeypatch.setattr(authority, "READINESS_NODE_ORDER_V1", ("EVIL_NODE",))

    for name in (
        "phase2b_step5_content_id_v2",
        "phase2b_step5_role_key_id_v2",
        "phase2b_step5_service_static_binding_id_v2",
        "phase2b_step5_signature_preimage_v2",
        "load_step5_stage_b_trust_lock_v2",
        "load_step5_authority_profile_v2",
        "load_step5_signer_registry_v2",
        "require_locked_step5_authority_graph_v2",
    ):
        monkeypatch.setattr(authority, name, rebound)

    plan_payload = _plan(rows=(("single-object-kind", 1_024),))
    with _new_store(
        tmp_path / "definition-time-authority-contract",
        plan_payload=plan_payload,
    ) as cas:
        assert cas.authority_profile_id == DEFAULT_GRAPH.authority_profile["id"]
        assert cas.signer_registry_id == DEFAULT_GRAPH.signer_registry["id"]
        assert cas._authority_binding.signer_role == store.CAS_STORE_ROLE
        assert (
            cas._authority_binding.signature_purpose
            == store.CAS_STORE_RECEIPT_PURPOSE
        )


def test_authority_cross_graph_key_and_purpose_exchanges_fail_closed(
    tmp_path: Path,
) -> None:
    other = _authority_graph(OTHER_SEED, variant="other")
    plan_payload = _plan(rows=(("single-object-kind", 1_024),))
    with pytest.raises(store.CasStoreV2Error):
        _new_store(
            tmp_path / "cross-graph",
            plan_payload=plan_payload,
            graph=other,
            seed=OTHER_SEED,
        )
    with pytest.raises(store.CasStoreV2Error):
        _new_store(
            tmp_path / "wrong-seed",
            plan_payload=plan_payload,
            seed=OTHER_SEED,
        )

    def wrong_purpose(row: dict[str, object]) -> None:
        row["authorized_signature_purposes"] = ["SIGN_SOMETHING_ELSE"]
        content = {key: value for key, value in row.items() if key != "key_id"}
        row["key_id"] = authority.phase2b_step5_role_key_id_v2(content)

    purpose_drift = _repin_registry_graph(DEFAULT_GRAPH, wrong_purpose)
    purpose_plan = _plan(purpose_drift, rows=(("single-object-kind", 1_024),))
    with pytest.raises(store.CasStoreV2Error) as rejected:
        _new_store(
            tmp_path / "purpose-drift",
            plan_payload=purpose_plan,
            graph=purpose_drift,
        )
    assert rejected.value.code == store.FAIL_AUTHORITY


@pytest.mark.parametrize("hybrid", ("profile", "registry", "lock"))
def test_stage_b_profile_registry_hybrid_exchange_fails_closed(
    tmp_path: Path,
    hybrid: str,
) -> None:
    other = _authority_graph(OTHER_SEED, variant="hybrid")
    plan_payload = _plan(rows=(("single-object-kind", 1_024),))
    kwargs: dict[str, object] = {
        "root_directory": str(tmp_path / f"hybrid-{hybrid}"),
        "retention_plan_payload": plan_payload,
        "private_key_seed": SEED,
        "allowed_peer_uids": frozenset({os.geteuid()}),
        "stage_b_lock_payload": DEFAULT_GRAPH.stage_b_lock_payload,
        "authority_profile_payload": DEFAULT_GRAPH.authority_profile_payload,
        "signer_registry_payload": DEFAULT_GRAPH.signer_registry_payload,
        "out_of_band_expected_lock_id_bytes": (
            DEFAULT_GRAPH.out_of_band_expected_lock_id_bytes
        ),
        "out_of_band_expected_lock_payload_sha256_digest": (
            DEFAULT_GRAPH.out_of_band_expected_lock_payload_sha256_digest
        ),
        "create": True,
    }
    if hybrid == "profile":
        kwargs["authority_profile_payload"] = other.authority_profile_payload
    elif hybrid == "registry":
        kwargs["signer_registry_payload"] = other.signer_registry_payload
    elif hybrid == "lock":
        kwargs["stage_b_lock_payload"] = other.stage_b_lock_payload
    with pytest.raises(store.CasStoreV2Error) as rejected:
        store.CasStoreV2(**kwargs)  # type: ignore[arg-type]
    assert rejected.value.code == store.FAIL_AUTHORITY


def test_attempt_session_chain_and_plan_exchange_rejected_before_append(
    tmp_path: Path,
) -> None:
    payload = b"exchange-object"
    plans = (
        _plan(rows=(("single-object-kind", 1_024),)),
        _plan(rows=(("single-object-kind", 1_024),), attempt=OTHER_ATTEMPT_ID),
        _plan(rows=(("single-object-kind", 1_024),), session=OTHER_SESSION_ID),
        _plan(rows=(("single-object-kind", 1_024),), chain=OTHER_CHAIN_ID),
    )
    request_from_first = store.load_generic_store_request_v1(
        _request_bytes(plans[0], 0, payload)
    )
    for index, plan_payload in enumerate(plans[1:], start=1):
        with _new_store(
            tmp_path / f"exchange-store-{index}",
            plan_payload=plan_payload,
        ) as cas:
            _, fd = _open_payload(tmp_path, f"exchange-{index}.bin", payload)
            try:
                with pytest.raises(store.CasStoreV2Error) as rejected:
                    cas.ingest_open_fd(request_from_first, fd, _peer())
            finally:
                os.close(fd)
            assert rejected.value.code == store.FAIL_BINDING
            assert cas.audit_chain() == ()


def test_request_id_and_transition_bind_every_plan_and_object_field(
    tmp_path: Path,
) -> None:
    plan_payload = _plan(rows=(("single-object-kind", 1_024),))
    payload = b"request-bound-object"
    request_payload = _request_bytes(plan_payload, 0, payload)
    request = store.load_generic_store_request_v1(request_payload)
    assert request.request_id.startswith(store.REQUEST_ID_PREFIX)
    assert request.canonical_bytes() == request_payload
    mapping = request.to_mapping()
    mapping["bound_artifact_id"] = "other-bound-artifact-id-0001"
    with pytest.raises(store.CasStoreV2Error):
        store.load_generic_store_request_v1(_canonical(mapping))
    mapping = request.to_mapping()
    mapping["reported_size_bytes"] = len(payload) + 1
    with pytest.raises(store.CasStoreV2Error):
        store.load_generic_store_request_v1(_canonical(mapping))

    with _new_store(tmp_path / "binding-store", plan_payload=plan_payload) as cas:
        _, fd = _open_payload(tmp_path, "binding-object.bin", payload)
        try:
            response = cas.ingest_open_fd(request, fd, _peer())
        finally:
            os.close(fd)
    binding = response.binding
    plan = store.load_retention_plan_v1(plan_payload)
    assert binding.to_request() == request
    assert binding.authority_profile_id == plan.authority_profile_id
    assert binding.signer_registry_id == plan.signer_registry_id
    assert binding.attempt_context_id == plan.attempt_context_id
    assert binding.retention_session_id == plan.retention_session_id
    assert binding.chain_id == plan.chain_id
    assert binding.plan_id == plan.plan_id
    assert binding.plan_kind == plan.plan_kind
    assert binding.request_id == request.request_id
    assert binding.object_id == binding.object_sha256 == request.bound_artifact_sha256
    assert binding.previous_checkpoint_sha256 == store.GENESIS_CHECKPOINT_SHA256
    assert binding.expected_object_count == binding.observed_object_count == 1
    assert binding.complete is True
    assert binding.head_state_id.startswith(store.HEAD_STATE_ID_PREFIX)
    for field, value in (
        ("request_id", store.REQUEST_ID_PREFIX + _h("drift")),
        ("previous_checkpoint_sha256", "sha256:" + _h("drift")),
        ("object_size_bytes", binding.object_size_bytes + 1),
        ("observed_object_count", 2),
        ("head_state_id", store.HEAD_STATE_ID_PREFIX + _h("drift")),
    ):
        with pytest.raises(store.CasStoreV2Error):
            replace(binding, **{field: value})


def test_seqpacket_scm_rights_binds_one_fd_and_kernel_peer(tmp_path: Path) -> None:
    plan_payload = _plan(rows=(("single-object-kind", 1_024),))
    payload = b"one-scm-rights-object"
    request = store.load_generic_store_request_v1(
        _request_bytes(plan_payload, 0, payload)
    )
    _, fd = _open_payload(tmp_path, "scm-object.bin", payload)
    sender, receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        store.send_store_request_fd_v2(sender, request, fd)
        received = store.recv_store_request_v2(
            receiver, allowed_peer_uids=frozenset({os.geteuid()})
        )
        with received:
            assert received.request == request
            assert received.peer == _peer()
            assert os.get_inheritable(received.fd) is False
            assert os.pread(received.fd, len(payload), 0) == payload
        with pytest.raises(OSError):
            os.fstat(received.fd)
    finally:
        os.close(fd)
        sender.close()
        receiver.close()


def test_scm_rejects_wrong_socket_uid_zero_or_multiple_descriptors(
    tmp_path: Path,
) -> None:
    plan_payload = _plan(rows=(("single-object-kind", 1_024),))
    payload = b"scm-negative-object"
    request = store.load_generic_store_request_v1(
        _request_bytes(plan_payload, 0, payload)
    )
    _, fd = _open_payload(tmp_path, "scm-negative.bin", payload)
    stream_left, stream_right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        with pytest.raises(store.CasStoreV2Error):
            store.send_store_request_fd_v2(stream_left, request, fd)
    finally:
        stream_left.close()
        stream_right.close()

    sender, receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        sender.send(request.canonical_bytes())
        with pytest.raises(store.CasStoreV2Error):
            store.recv_store_request_v2(
                receiver, allowed_peer_uids=frozenset({os.geteuid()})
            )
    finally:
        sender.close()
        receiver.close()

    sender, receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        rights = array("i", [fd, fd]).tobytes()
        sender.sendmsg(
            [request.canonical_bytes()],
            [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights)],
        )
        with pytest.raises(store.CasStoreV2Error):
            store.recv_store_request_v2(
                receiver, allowed_peer_uids=frozenset({os.geteuid()})
            )
    finally:
        sender.close()
        receiver.close()

    sender, receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        with pytest.raises(PermissionError):
            store.recv_store_request_v2(
                receiver, allowed_peer_uids=frozenset({os.geteuid() + 1})
            )
    finally:
        os.close(fd)
        sender.close()
        receiver.close()


def test_ingest_rehashes_readonly_regular_single_link_and_exact_size(
    tmp_path: Path,
) -> None:
    plan_payload = _plan(rows=(("single-object-kind", 1_024),))
    payload = b"independently-rehashed-object"
    valid = store.load_generic_store_request_v1(
        _request_bytes(plan_payload, 0, payload)
    )
    root = tmp_path / "fd-contract-store"
    with _new_store(root, plan_payload=plan_payload) as cas:
        writable_path = tmp_path / "writable.bin"
        writable_path.write_bytes(payload)
        writable_fd = os.open(writable_path, os.O_RDWR | os.O_CLOEXEC)
        try:
            with pytest.raises(store.CasStoreV2Error) as writable:
                cas.ingest_open_fd(valid, writable_fd, _peer())
        finally:
            os.close(writable_fd)
        assert writable.value.code == store.FAIL_INPUT

        linked_path = tmp_path / "linked.bin"
        linked_path.write_bytes(payload)
        second_link = tmp_path / "second-link.bin"
        os.link(linked_path, second_link)
        linked_fd = os.open(linked_path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            with pytest.raises(store.CasStoreV2Error) as linked:
                cas.ingest_open_fd(valid, linked_fd, _peer())
        finally:
            os.close(linked_fd)
        assert linked.value.code == store.FAIL_INPUT

        wrong_size_payload = store.build_generic_store_request_v1(
            retention_plan_payload=plan_payload,
            ordinal=0,
            bound_artifact_id="wrong-size-artifact-id-0001",
            bound_artifact_sha256=_sha256(payload),
            reported_size_bytes=len(payload) + 1,
        )
        wrong_size = store.load_generic_store_request_v1(wrong_size_payload)
        _, fd = _open_payload(tmp_path, "wrong-size.bin", payload)
        try:
            with pytest.raises(store.CasStoreV2Error) as mismatch:
                cas.ingest_open_fd(wrong_size, fd, _peer())
        finally:
            os.close(fd)
        assert mismatch.value.code == store.FAIL_BINDING

        wrong_hash_payload = store.build_generic_store_request_v1(
            retention_plan_payload=plan_payload,
            ordinal=0,
            bound_artifact_id="wrong-hash-artifact-id-0001",
            bound_artifact_sha256="sha256:" + _h("wrong-object"),
            reported_size_bytes=len(payload),
        )
        wrong_hash = store.load_generic_store_request_v1(wrong_hash_payload)
        _, fd = _open_payload(tmp_path, "wrong-hash.bin", payload)
        try:
            with pytest.raises(store.CasStoreV2Error) as mismatch:
                cas.ingest_open_fd(wrong_hash, fd, _peer())
        finally:
            os.close(fd)
        assert mismatch.value.code == store.FAIL_BINDING
        assert cas.audit_chain() == ()


def test_if_absent_deduplicates_content_and_fsyncs_before_verified_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = (("first-kind", 1_024), ("second-kind", 1_024))
    plan_payload = _plan(rows=rows)
    payload = b"same-content-at-two-ordinals"
    original_fsync = os.fsync
    fsynced: list[int] = []

    def observed_fsync(fd: int) -> None:
        fsynced.append(fd)
        original_fsync(fd)

    monkeypatch.setattr(store.os, "fsync", observed_fsync)
    root = tmp_path / "if-absent-store"
    with _new_store(root, plan_payload=plan_payload) as cas:
        responses: list[store.StoreResponseV2] = []
        for ordinal in range(2):
            request = store.load_generic_store_request_v1(
                _request_bytes(plan_payload, ordinal, payload)
            )
            _, fd = _open_payload(tmp_path, f"same-{ordinal}.bin", payload)
            try:
                responses.append(cas.ingest_open_fd(request, fd, _peer()))
            finally:
                os.close(fd)
        assert responses[0].checkpoint_record.signed_receipt.signed_body.object_disposition == (
            "CREATED_IF_ABSENT"
        )
        assert responses[1].checkpoint_record.signed_receipt.signed_body.object_disposition == (
            "PRESENT_VERIFIED"
        )
        assert responses[0].durable_before_return is True
        assert responses[1].readback_verified is True
        assert len(cas.audit_complete_session()) == 2
        object_names = tuple((root / "objects").iterdir())
        assert len(object_names) == 1
        observed = object_names[0].stat()
        assert stat.S_ISREG(observed.st_mode)
        assert observed.st_nlink == 1
        assert observed.st_mode & 0o077 == 0
        assert object_names[0].read_bytes() == payload
        assert object_names[0].name == _sha256(payload)[7:]
    assert len(fsynced) >= 12
    with _new_store(root, plan_payload=plan_payload, create=False) as reopened:
        assert len(reopened.audit_complete_session()) == 2


def test_object_symlink_hardlink_hash_and_namespace_tampering_fail_closed(
    tmp_path: Path,
) -> None:
    rows = (("single-object-kind", 1_024),)
    payload = b"tamper-detection-object"
    evidence = _complete(tmp_path, rows=rows, payloads=(payload,))
    root = tmp_path / f"cas-{ATTEMPT_ID}-{SESSION_ID}-{CHAIN_ID}"
    object_path = root / "objects" / _sha256(payload)[7:]

    extra_link = tmp_path / "external-hard-link.bin"
    os.link(object_path, extra_link)
    with pytest.raises(store.CasStoreV2Error):
        _new_store(root, plan_payload=evidence.plan, create=False)
    extra_link.unlink()

    original = object_path.read_bytes()
    object_path.write_bytes(b"same-length-object-drift!!"[: len(original)])
    with pytest.raises(store.CasStoreV2Error):
        _new_store(root, plan_payload=evidence.plan, create=False)
    object_path.write_bytes(original)

    outside = tmp_path / "outside-object.bin"
    outside.write_bytes(original)
    object_path.unlink()
    object_path.symlink_to(outside)
    with pytest.raises((store.CasStoreV2Error, OSError)):
        _new_store(root, plan_payload=evidence.plan, create=False)


def test_ordinal_predecessor_counts_and_only_terminal_complete_head(
    tmp_path: Path,
) -> None:
    evidence = _complete(tmp_path)
    checkpoints = tuple(
        store.load_store_checkpoint_v2(payload) for payload in evidence.checkpoints
    )
    heads = tuple(store.load_store_head_v2(payload) for payload in evidence.heads)
    previous = store.GENESIS_CHECKPOINT_SHA256
    for index, (checkpoint, head) in enumerate(zip(checkpoints, heads, strict=True)):
        binding = checkpoint.binding
        assert binding == head.binding
        assert binding.ordinal == index
        assert binding.previous_checkpoint_sha256 == previous
        assert binding.expected_object_count == 3
        assert binding.observed_object_count == index + 1
        assert binding.complete is (index == 2)
        previous = checkpoint.checkpoint_sha256
    root = tmp_path / f"cas-{ATTEMPT_ID}-{SESSION_ID}-{CHAIN_ID}"
    assert (root / "state" / "current-head.json").read_bytes() == evidence.heads[-1]
    assert tuple(path.name for path in sorted((root / "checkpoints").iterdir())) == (
        "00000000000000000000.json",
        "00000000000000000001.json",
        "00000000000000000002.json",
    )


def test_out_of_order_gap_completed_append_and_exact_duplicate_semantics(
    tmp_path: Path,
) -> None:
    rows = (("first-kind", 1_024), ("second-kind", 1_024))
    plan_payload = _plan(rows=rows)
    payload0 = b"duplicate-object-zero"
    payload1 = b"object-one"
    request0_payload = _request_bytes(plan_payload, 0, payload0)
    request0 = store.load_generic_store_request_v1(request0_payload)
    request1 = store.load_generic_store_request_v1(
        _request_bytes(plan_payload, 1, payload1)
    )
    root = tmp_path / "duplicate-store"
    with _new_store(root, plan_payload=plan_payload) as cas:
        _, fd = _open_payload(tmp_path, "gap-one.bin", payload1)
        try:
            with pytest.raises(store.CasStoreV2Error) as gap:
                cas.ingest_open_fd(request1, fd, _peer())
        finally:
            os.close(fd)
        assert gap.value.code == store.FAIL_CHAIN

        _, fd = _open_payload(tmp_path, "duplicate-first.bin", payload0)
        try:
            first = cas.ingest_open_fd(request0, fd, _peer())
        finally:
            os.close(fd)
        _, fd = _open_payload(tmp_path, "duplicate-exact.bin", payload0)
        try:
            replay = cas.ingest_open_fd(request0, fd, _peer())
        finally:
            os.close(fd)
        assert replay.canonical_bytes() == first.canonical_bytes()
        assert len(cas.audit_chain()) == 1

        _, fd = _open_payload(tmp_path, "duplicate-peer-drift.bin", payload0)
        try:
            with pytest.raises(store.CasStoreV2Error) as peer_drift:
                cas.ingest_open_fd(request0, fd, _peer(pid=os.getpid() + 1))
        finally:
            os.close(fd)
        assert peer_drift.value.code == store.FAIL_CHAIN

        _, fd = _open_payload(tmp_path, "duplicate-fd-drift.bin", b"different-content")
        try:
            with pytest.raises(store.CasStoreV2Error):
                cas.ingest_open_fd(request0, fd, _peer())
        finally:
            os.close(fd)

        _, fd = _open_payload(tmp_path, "second.bin", payload1)
        try:
            cas.ingest_open_fd(request1, fd, _peer())
        finally:
            os.close(fd)
        assert len(cas.audit_complete_session()) == 2

        drift_payload = b"other-zero-after-complete"
        drift_request = store.load_generic_store_request_v1(
            _request_bytes(plan_payload, 0, drift_payload)
        )
        _, fd = _open_payload(tmp_path, "post-complete-drift.bin", drift_payload)
        try:
            with pytest.raises(store.CasStoreV2Error) as completed:
                cas.ingest_open_fd(drift_request, fd, _peer())
        finally:
            os.close(fd)
        assert completed.value.code == store.FAIL_CHAIN


def test_crash_between_immutable_records_leaves_fail_closed_gap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_payload = _plan(rows=(("single-object-kind", 1_024),))
    payload = b"crash-gap-object"
    request = store.load_generic_store_request_v1(
        _request_bytes(plan_payload, 0, payload)
    )
    root = tmp_path / "crash-gap-store"
    cas = _new_store(root, plan_payload=plan_payload)
    original_write = store._write_immutable_at

    def fail_response_record(
        directory_fd: int,
        basename: str,
        record_payload: bytes,
        maximum: int,
    ) -> None:
        if directory_fd == cas._responses_fd:
            raise OSError("injected response append crash")
        original_write(directory_fd, basename, record_payload, maximum)

    monkeypatch.setattr(store, "_write_immutable_at", fail_response_record)
    _, fd = _open_payload(tmp_path, "crash-gap.bin", payload)
    try:
        with pytest.raises(OSError, match="injected response append crash"):
            cas.ingest_open_fd(request, fd, _peer())
    finally:
        os.close(fd)
        cas.close()
    assert len(tuple((root / "objects").iterdir())) == 1
    assert len(tuple((root / "checkpoints").iterdir())) == 1
    assert len(tuple((root / "heads").iterdir())) == 1
    assert tuple((root / "responses").iterdir()) == ()
    assert not (root / "state" / "current-head.json").exists()
    monkeypatch.setattr(store, "_write_immutable_at", original_write)
    with pytest.raises(store.CasStoreV2Error) as no_repair:
        _new_store(root, plan_payload=plan_payload, create=False)
    assert no_repair.value.code == store.FAIL_CHAIN
    assert not hasattr(store.CasStoreV2, "repair_chain")


@pytest.mark.parametrize(
    "case",
    ("pending", "orphan", "missing-head", "stale-head", "record-gap", "fork"),
)
def test_pending_orphan_head_and_record_forks_fail_closed(
    tmp_path: Path,
    case: str,
) -> None:
    attempt = f"attempt-v2-{case}-0001"
    session = f"session-v2-{case}-0001"
    chain = f"chain-v2-{case}-0001"
    evidence = _complete(
        tmp_path,
        rows=(("first-kind", 1_024), ("second-kind", 1_024)),
        attempt=attempt,
        session=session,
        chain=chain,
    )
    root = tmp_path / f"cas-{attempt}-{session}-{chain}"
    if case == "pending":
        (root / "staging" / "pending-crash-artifact").write_bytes(b"partial")
    elif case == "orphan":
        orphan = root / "objects" / _h("orphan-object")
        orphan.write_bytes(b"orphan")
        orphan.chmod(0o600)
    elif case == "missing-head":
        (root / "state" / "current-head.json").unlink()
    elif case == "stale-head":
        (root / "state" / "current-head.json").write_bytes(evidence.heads[0])
    elif case == "record-gap":
        (root / "responses" / "00000000000000000001.json").unlink()
    elif case == "fork":
        fork = root / "heads" / "99999999999999999999.json"
        fork.write_bytes(evidence.heads[-1])
        fork.chmod(0o600)
    else:  # pragma: no cover - frozen parametrization.
        raise AssertionError(case)
    with pytest.raises((store.CasStoreV2Error, OSError)):
        _new_store(root, plan_payload=evidence.plan, create=False)


def test_consistent_rollback_clone_is_disclosed_not_overclaimed(tmp_path: Path) -> None:
    rows = (("first-kind", 1_024), ("second-kind", 1_024))
    plan_payload = _plan(rows=rows)
    root = tmp_path / "rollback-original"
    clone = tmp_path / "rollback-clone"
    first_payload = b"rollback-first"
    second_payload = b"rollback-second"
    first_request = store.load_generic_store_request_v1(
        _request_bytes(plan_payload, 0, first_payload)
    )
    with _new_store(root, plan_payload=plan_payload) as cas:
        _, fd = _open_payload(tmp_path, "rollback-first.bin", first_payload)
        try:
            cas.ingest_open_fd(first_request, fd, _peer())
        finally:
            os.close(fd)
    shutil.copytree(root, clone)

    with _new_store(root, plan_payload=plan_payload, create=False) as cas:
        second_request = store.load_generic_store_request_v1(
            _request_bytes(plan_payload, 1, second_payload)
        )
        _, fd = _open_payload(tmp_path, "rollback-second.bin", second_payload)
        try:
            cas.ingest_open_fd(second_request, fd, _peer())
        finally:
            os.close(fd)
        assert len(cas.audit_complete_session()) == 2

    # A self-consistent earlier clone is locally valid but incomplete.  V2
    # explicitly requires an external complete-head anchor to distinguish it.
    with _new_store(clone, plan_payload=plan_payload, create=False) as rolled_back:
        assert len(rolled_back.audit_chain()) == 1
        with pytest.raises(store.CasStoreV2Error) as incomplete:
            rolled_back.audit_complete_session()
        assert incomplete.value.code == store.FAIL_INCOMPLETE
    assert "WHOLE_TREE_ROLLBACK_OR_CLONE" in store.FAIL_CLOSED_ROLLBACK_DISCLOSURE


def test_signed_receipt_uses_real_registry_ed25519_and_exact_purpose(
    tmp_path: Path,
) -> None:
    evidence = _complete(
        tmp_path,
        rows=(("single-object-kind", 1_024),),
        payloads=(b"signed-receipt-object",),
    )
    body = store.verify_signed_store_receipt_v2(
        signed_receipt_payload=evidence.receipts[0],
        stage_b_lock_payload=DEFAULT_GRAPH.stage_b_lock_payload,
        authority_profile_payload=DEFAULT_GRAPH.authority_profile_payload,
        signer_registry_payload=DEFAULT_GRAPH.signer_registry_payload,
        out_of_band_expected_lock_id_bytes=(
            DEFAULT_GRAPH.out_of_band_expected_lock_id_bytes
        ),
        out_of_band_expected_lock_payload_sha256_digest=(
            DEFAULT_GRAPH.out_of_band_expected_lock_payload_sha256_digest
        ),
    )
    assert body.purpose == store.CAS_STORE_RECEIPT_PURPOSE
    assert body.signature_domain == store.CAS_STORE_SIGNATURE_DOMAIN
    assert body.binding.signer_role == store.CAS_STORE_ROLE
    assert body.binding.signer_control_domain == store.CAS_STORE_CONTROL_DOMAIN
    assert body.binding.signer_key_id == DEFAULT_GRAPH.cas_row["key_id"]
    assert body.binding.authority_profile_id == DEFAULT_GRAPH.authority_profile["id"]
    assert body.binding.signer_registry_id == DEFAULT_GRAPH.signer_registry["id"]
    assert body.durable_before_return is True
    assert body.readback_verified is True

    tampered = json.loads(evidence.receipts[0])
    signature = tampered["ed25519_signature_hex"]
    tampered["ed25519_signature_hex"] = (
        ("0" if signature[0] != "0" else "1") + signature[1:]
    )
    with pytest.raises(store.CasStoreV2Error) as bad_signature:
        store.verify_signed_store_receipt_v2(
            signed_receipt_payload=_canonical(tampered),
            stage_b_lock_payload=DEFAULT_GRAPH.stage_b_lock_payload,
            authority_profile_payload=DEFAULT_GRAPH.authority_profile_payload,
            signer_registry_payload=DEFAULT_GRAPH.signer_registry_payload,
            out_of_band_expected_lock_id_bytes=(
                DEFAULT_GRAPH.out_of_band_expected_lock_id_bytes
            ),
            out_of_band_expected_lock_payload_sha256_digest=(
                DEFAULT_GRAPH.out_of_band_expected_lock_payload_sha256_digest
            ),
        )
    assert bad_signature.value.code == store.FAIL_SIGNATURE
    assert "verifier" not in inspect.signature(
        store.verify_signed_store_receipt_v2
    ).parameters


def test_complete_bytes_only_replay_consumes_every_raw_artifact(tmp_path: Path) -> None:
    evidence = _complete(tmp_path)
    result = _verify(evidence)
    plan = store.load_retention_plan_v1(evidence.plan)
    assert isinstance(result, store.VerifiedCasStoreCompleteSessionV2)
    assert result.verification_status.endswith("MECHANICS_ONLY_NOT_AUTHORITY")
    assert result.authority_profile_id == plan.authority_profile_id
    assert result.signer_registry_id == plan.signer_registry_id
    assert result.signer_role == store.CAS_STORE_ROLE
    assert result.signer_key_id == DEFAULT_GRAPH.cas_row["key_id"]
    assert result.key_epoch == 1
    assert result.attempt_context_id == ATTEMPT_ID
    assert result.retention_session_id == SESSION_ID
    assert result.chain_id == CHAIN_ID
    assert result.plan_id == plan.plan_id
    assert result.plan_kind == PLAN_KIND
    assert result.expected_object_count == 3
    assert len(result.object_bindings_in_exact_order) == 3
    assert result.complete_head_sha256 == _sha256(evidence.heads[-1])
    assert result.complete_checkpoint_sha256 == _sha256(evidence.checkpoints[-1])
    assert result.complete_response_sha256 == _sha256(evidence.responses[-1])
    assert result.claim_boundary == store.AUTHORITY_CLAIM_BOUNDARY
    assert result.rollback_disclosure == store.FAIL_CLOSED_ROLLBACK_DISCLOSURE

    for index, row in enumerate(result.object_bindings_in_exact_order):
        assert row[0] == index
        assert row[1] == DEFAULT_ROWS[index][0]
        assert row[3] == _sha256(evidence.objects[index])
        assert row[4] == row[3]
        assert row[5] == len(evidence.objects[index])
        assert row[6] == _sha256(evidence.receipts[index])
        assert row[7] == _sha256(evidence.checkpoints[index])
        assert row[8] == _sha256(evidence.heads[index])
        assert row[9] == _sha256(evidence.responses[index])


@pytest.mark.parametrize(
    "field",
    ("requests", "objects", "receipts", "checkpoints", "heads", "responses"),
)
def test_complete_replay_requires_exact_tuple_and_exact_bytes_items(
    tmp_path: Path,
    field: str,
) -> None:
    evidence = _complete(tmp_path)
    sequence = getattr(evidence, field)
    with pytest.raises(store.CasStoreV2Error) as list_container:
        _verify(replace(evidence, **{field: list(sequence)}))  # type: ignore[arg-type]
    assert list_container.value.code == store.FAIL_INPUT
    changed = list(sequence)
    changed[0] = bytearray(changed[0])
    with pytest.raises(store.CasStoreV2Error) as bytearray_item:
        _verify(replace(evidence, **{field: tuple(changed)}))
    assert bytearray_item.value.code == store.FAIL_INPUT
    with pytest.raises(store.CasStoreV2Error) as incomplete:
        _verify(replace(evidence, **{field: sequence[:-1]}))
    assert incomplete.value.code == store.FAIL_INCOMPLETE


@pytest.mark.parametrize(
    "field",
    ("requests", "receipts", "checkpoints", "heads", "responses"),
)
def test_complete_replay_rejects_swapped_raw_wire_records(
    tmp_path: Path,
    field: str,
) -> None:
    evidence = _complete(tmp_path)
    sequence = getattr(evidence, field)
    swapped = (sequence[1], sequence[0], *sequence[2:])
    with pytest.raises(store.CasStoreV2Error):
        _verify(replace(evidence, **{field: swapped}))


def test_complete_replay_rejects_object_request_receipt_and_graph_exchange(
    tmp_path: Path,
) -> None:
    evidence = _complete(tmp_path)
    objects = list(evidence.objects)
    objects[1] = b"same-size-object-drift"[: len(objects[1])]
    with pytest.raises(store.CasStoreV2Error):
        _verify(replace(evidence, objects=tuple(objects)))

    other_attempt = _complete(
        tmp_path,
        attempt=OTHER_ATTEMPT_ID,
        session=OTHER_SESSION_ID,
        chain=OTHER_CHAIN_ID,
    )
    for field in ("requests", "receipts", "checkpoints", "heads", "responses"):
        mixed = list(getattr(evidence, field))
        mixed[0] = getattr(other_attempt, field)[0]
        with pytest.raises(store.CasStoreV2Error):
            _verify(replace(evidence, **{field: tuple(mixed)}))

    other_graph = _authority_graph(OTHER_SEED, variant="complete-other")
    with pytest.raises(store.CasStoreV2Error) as graph_exchange:
        _verify(evidence, other_graph)
    assert graph_exchange.value.code == store.FAIL_AUTHORITY


def test_complete_replay_accepts_same_cas_object_at_distinct_ordinals(
    tmp_path: Path,
) -> None:
    payload = b"duplicate-cas-content"
    evidence = _complete(
        tmp_path,
        rows=(("first-kind", 1_024), ("second-kind", 1_024)),
        payloads=(payload, payload),
    )
    result = _verify(evidence)
    assert result.expected_object_count == 2
    assert result.object_bindings_in_exact_order[0][4] == result.object_bindings_in_exact_order[1][4]
    first_receipt = store.SignedStoreReceiptV2.from_canonical_bytes(evidence.receipts[0])
    second_receipt = store.SignedStoreReceiptV2.from_canonical_bytes(evidence.receipts[1])
    assert first_receipt.signed_body.object_disposition == "CREATED_IF_ABSENT"
    assert second_receipt.signed_body.object_disposition == "PRESENT_VERIFIED"


def test_fully_readdressed_invalid_signature_survives_no_backend_rebinding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _complete(
        tmp_path,
        rows=(("single-object-kind", 1_024),),
        payloads=(b"fully-readdressed-invalid-signature",),
    )
    receipt = json.loads(evidence.receipts[0])
    receipt["ed25519_signature_hex"] = "0" * 128
    receipt_payload = _canonical(receipt)

    checkpoint = json.loads(evidence.checkpoints[0])
    checkpoint["signed_receipt"] = receipt
    checkpoint["signed_receipt_sha256"] = _sha256(receipt_payload)
    checkpoint_payload = _canonical(checkpoint)

    head = json.loads(evidence.heads[0])
    head["checkpoint_sha256"] = _sha256(checkpoint_payload)
    head["signed_receipt_sha256"] = _sha256(receipt_payload)
    head_payload = _canonical(head)

    response = json.loads(evidence.responses[0])
    response["checkpoint_sha256"] = _sha256(checkpoint_payload)
    response["checkpoint_record"] = checkpoint
    response["head_sha256"] = _sha256(head_payload)
    response["head_record"] = head
    response_payload = _canonical(response)
    invalid = replace(
        evidence,
        receipts=(receipt_payload,),
        checkpoints=(checkpoint_payload,),
        heads=(head_payload,),
        responses=(response_payload,),
    )

    class AcceptEverySignature:
        @classmethod
        def from_public_bytes(cls, payload: bytes) -> "AcceptEverySignature":
            return cls()

        def verify(self, signature: bytes, preimage: bytes) -> None:
            return None

    monkeypatch.setattr(store, "_Ed25519PublicKey", AcceptEverySignature)
    monkeypatch.setattr(store, "_InvalidSignature", Exception)
    monkeypatch.setattr(
        store,
        "_receipt_signature_preimage",
        lambda body: b"globally-rebound-preimage",
    )
    with pytest.raises(store.CasStoreV2Error) as rejected:
        _verify(invalid)
    assert rejected.value.code == store.FAIL_SIGNATURE


def test_require_verified_result_enforces_issued_identity_and_immutable_fields(
    tmp_path: Path,
) -> None:
    genuine = _verify(_complete(tmp_path))
    assert store.require_verified_cas_store_complete_session_v2(genuine) is genuine

    forged = object.__new__(store.VerifiedCasStoreCompleteSessionV2)
    for field_name in genuine.__dataclass_fields__:
        object.__setattr__(forged, field_name, getattr(genuine, field_name))
    with pytest.raises(store.CasStoreV2Error) as unissued:
        store.require_verified_cas_store_complete_session_v2(forged)
    assert unissued.value.code == store.FAIL_INPUT

    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(genuine)
    with pytest.raises(TypeError, match="cannot be deep-copied"):
        copy.deepcopy(genuine)

    mutated = _verify(
        _complete(
            tmp_path,
            attempt=OTHER_ATTEMPT_ID,
            session=OTHER_SESSION_ID,
            chain=OTHER_CHAIN_ID,
        )
    )
    object.__setattr__(
        mutated,
        "expected_object_count",
        mutated.expected_object_count + 1,
    )
    with pytest.raises(store.CasStoreV2Error) as changed:
        store.require_verified_cas_store_complete_session_v2(mutated)
    assert changed.value.code == store.FAIL_BINDING


def test_private_result_cannot_be_directly_issued_or_global_helper_rebound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert not hasattr(store, "_COMPLETE_RESULT_TOKEN")
    assert not hasattr(store, "_issue_complete_result")
    with pytest.raises(TypeError):
        store.VerifiedCasStoreCompleteSessionV2()
    with pytest.raises(TypeError):
        store.VerifiedCasStoreCompleteSessionV2(object())

    evidence = _complete(tmp_path)

    def rebound(*args: object, **kwargs: object) -> object:
        raise AssertionError("module-global helper rebinding reached verifier")

    monkeypatch.setattr(store, "_COMPLETE_RESULT_TOKEN", object(), raising=False)
    monkeypatch.setattr(store, "_issue_complete_result", rebound, raising=False)
    monkeypatch.setattr(store, "_verify_record_triplet", rebound)
    result = _verify(evidence)
    assert isinstance(result, store.VerifiedCasStoreCompleteSessionV2)
    with pytest.raises((AttributeError, TypeError)):
        result.expected_object_count = 99  # type: ignore[misc]
