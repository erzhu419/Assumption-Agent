from __future__ import annotations

from array import array
import copy
from dataclasses import dataclass, replace
import hashlib
import inspect
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading

import pytest

from hegel_machine import phase2b_formal_wire_covert_step5_runtime_v1 as runtime
from hegel_machine import phase2b_formal_wire_covert_step5_authority_v1 as authority
from hegel_machine import phase2b_formal_wire_covert_step5_store_v1 as store

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


EXPECTED_STORE_SOURCE_SHA256 = (
    "d22f003771cb7fa378299b53382e556a4d7b723a678659962db1577b2517f5f0"
)
ATTEMPT_ID = "attempt-context-store-0001"
OTHER_ATTEMPT_ID = "attempt-context-store-0002"
SESSION_ID = "retention-session-store-0001"
OTHER_SESSION_ID = "retention-session-store-0002"
CHAIN_ID = "phase2b-step5-cas-store-chain-0001"
SEED = bytes(range(32))
OTHER_SEED = bytes(reversed(range(32)))


def _raw_public_key(seed: bytes) -> bytes:
    return Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical(value: object) -> bytes:
    return authority.canonical_step5_json_bytes_v1(value)


def _qv2_binding() -> dict[str, object]:
    return {
        "schema_version": (
            "hegel-machine-phase2b-formal-wire-covert-qualification/9"
        ),
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
    role: str,
    private_key: Ed25519PrivateKey,
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
        "public_key_hex": private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        .hex(),
    }
    row["key_id"] = authority.phase2b_step5_role_key_id_v1(row)
    return row


def _governance_signature(
    body: dict[str, object],
    governance_row: dict[str, object],
    governance_key: Ed25519PrivateKey,
) -> dict[str, object]:
    purpose = "SIGN_SIGNER_REGISTRY"
    preimage = authority.phase2b_step5_signature_preimage_v1(
        artifact_kind="signer_registry",
        body=body,
        role="GOVERNANCE_ANCHOR",
        purpose=purpose,
        signer_key_id=governance_row["key_id"],  # type: ignore[arg-type]
        key_epoch=1,
    )
    return {
        "schema_version": authority.DETACHED_SIGNATURE_SCHEMA_V1,
        "algorithm": "Ed25519",
        "role": "GOVERNANCE_ANCHOR",
        "purpose": purpose,
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
    out_of_band_expected_lock_id: str
    out_of_band_expected_lock_payload_sha256: str
    cas_row: dict[str, object]


def _authority_graph(
    cas_seed: bytes = SEED,
    *,
    variant: str = "default",
) -> _AuthorityGraph:
    if type(variant) is not str or not variant:
        raise ValueError("test authority graph variant must be one exact label")
    variant_suffix = "" if variant == "default" else "-" + variant
    private_keys = tuple(
        Ed25519PrivateKey.from_private_bytes(
            cas_seed
            if role == "CAS_STORE"
            else hashlib.sha256(f"step5-store-role-{index}".encode()).digest()
        )
        for index, role in enumerate(authority.ROLE_ORDER_V1)
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
            "source_commit_sha": (
                "a" * 40 if variant == "default" else "b" * 40
            ),
            "source_archive_sha256": _h(
                "step5-store-source-archive" + variant_suffix
            ),
            "dependency_lock_sha256": _h(
                "step5-store-dependency-lock" + variant_suffix
            ),
        },
        "runtime_image_bindings": {
            "runner": "sha256:"
            + _h("step5-store-runner-image" + variant_suffix),
            "pure_replayer": "sha256:"
            + _h("step5-store-replayer-image" + variant_suffix),
            "cas_store": "sha256:"
            + _h("step5-store-cas-image" + variant_suffix),
            "logical_time": "sha256:"
            + _h("step5-store-time-image" + variant_suffix),
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
    profile: dict[str, object] = {
        "body": profile_body,
        "id": authority.phase2b_step5_content_id_v1(
            "authority_profile", profile_body
        ),
    }
    profile_payload = _canonical(profile)
    registry_body: dict[str, object] = {
        "schema_version": authority.SIGNER_REGISTRY_SCHEMA_V1,
        "authority_profile_id": profile["id"],
        "registry_epoch": 1,
        "status": "ACTIVE",
        "revocation_epoch": 0,
        "role_keys_in_exact_order": [copy.deepcopy(row) for row in rows],
    }
    registry: dict[str, object] = {
        "body": registry_body,
        "id": authority.phase2b_step5_content_id_v1(
            "signer_registry", registry_body
        ),
        "signatures": [
            _governance_signature(registry_body, rows[0], private_keys[0])
        ],
    }
    registry_payload = _canonical(registry)
    lock_body: dict[str, object] = {
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
    lock: dict[str, object] = {
        "body": lock_body,
        "id": authority.phase2b_step5_content_id_v1(
            "stage_b_trust_lock", lock_body
        ),
    }
    lock_payload = _canonical(lock)
    return _AuthorityGraph(
        stage_b_lock=lock,
        stage_b_lock_payload=lock_payload,
        authority_profile=profile,
        authority_profile_payload=profile_payload,
        signer_registry=registry,
        signer_registry_payload=registry_payload,
        out_of_band_expected_lock_id=lock["id"],  # type: ignore[arg-type]
        out_of_band_expected_lock_payload_sha256=hashlib.sha256(
            lock_payload
        ).hexdigest(),
        cas_row=copy.deepcopy(
            rows[authority.ROLE_ORDER_V1.index("CAS_STORE")]
        ),
    )


def _repin_authority_graph(
    profile: dict[str, object],
    registry: dict[str, object],
) -> _AuthorityGraph:
    profile_payload = _canonical(profile)
    registry_payload = _canonical(registry)
    lock_body: dict[str, object] = {
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
    lock: dict[str, object] = {
        "body": lock_body,
        "id": authority.phase2b_step5_content_id_v1(
            "stage_b_trust_lock", lock_body
        ),
    }
    lock_payload = _canonical(lock)
    registry_body = registry["body"]
    assert type(registry_body) is dict
    rows = registry_body["role_keys_in_exact_order"]
    assert type(rows) is list
    cas_row = rows[authority.ROLE_ORDER_V1.index("CAS_STORE")]
    assert type(cas_row) is dict
    return _AuthorityGraph(
        stage_b_lock=copy.deepcopy(lock),
        stage_b_lock_payload=lock_payload,
        authority_profile=copy.deepcopy(profile),
        authority_profile_payload=profile_payload,
        signer_registry=copy.deepcopy(registry),
        signer_registry_payload=registry_payload,
        out_of_band_expected_lock_id=lock["id"],  # type: ignore[arg-type]
        out_of_band_expected_lock_payload_sha256=hashlib.sha256(
            lock_payload
        ).hexdigest(),
        cas_row=copy.deepcopy(cas_row),
    )


def _resign_registry(registry_body: dict[str, object]) -> dict[str, object]:
    rows = registry_body["role_keys_in_exact_order"]
    assert type(rows) is list and type(rows[0]) is dict
    governance_key = Ed25519PrivateKey.from_private_bytes(
        hashlib.sha256(b"step5-store-role-0").digest()
    )
    return {
        "body": copy.deepcopy(registry_body),
        "id": authority.phase2b_step5_content_id_v1(
            "signer_registry", registry_body
        ),
        "signatures": [
            _governance_signature(registry_body, rows[0], governance_key)
        ],
    }


DEFAULT_AUTHORITY_GRAPH = _authority_graph()
AUTHORITY_PROFILE_ID = DEFAULT_AUTHORITY_GRAPH.authority_profile["id"]
SIGNER_REGISTRY_ID = DEFAULT_AUTHORITY_GRAPH.signer_registry["id"]
TRUST_LOCK_ID = DEFAULT_AUTHORITY_GRAPH.stage_b_lock["id"]
assert type(AUTHORITY_PROFILE_ID) is str
assert type(SIGNER_REGISTRY_ID) is str
assert type(TRUST_LOCK_ID) is str


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _request(
    payload: bytes,
    *,
    ordinal: int = 0,
    attempt_context_id: str = ATTEMPT_ID,
    retention_session_id: str = SESSION_ID,
    reported_sha256: str | None = None,
    reported_size_bytes: int | None = None,
) -> store.StoreRequestV1:
    mode, phase, target = store.EXPECTED_OBJECT_ORDER[ordinal]
    basename = (
        store.INVARIANT_BASENAME
        if target is None
        else store.STATISTICS_BASENAMES[target]
    )
    unsigned: dict[str, object] = {
        "schema_version": store.STORE_REQUEST_SCHEMA_VERSION,
        "attempt_context_id": attempt_context_id,
        "retention_session_id": retention_session_id,
        "ordinal": ordinal,
        "mode": mode,
        "phase": phase,
        "target_index_or_null": target,
        "source_basename": basename,
        "auditor_reported_sha256": (
            _sha256(payload) if reported_sha256 is None else reported_sha256
        ),
        "auditor_reported_size_bytes": (
            len(payload) if reported_size_bytes is None else reported_size_bytes
        ),
        "fd_count": 1,
    }
    request_id = _sha256(store.canonicalize_rfc8785_v1(unsigned))
    return store.StoreRequestV1.from_mapping(
        {**unsigned, "request_id": request_id}
    )


def _peer() -> store.PeerCredentialsV1:
    return store.PeerCredentialsV1(
        pid=os.getpid(),
        uid=os.geteuid(),
        gid=os.getegid(),
    )


def _new_store(
    root: Path,
    *,
    create: bool,
    seed: bytes = SEED,
    attempt_context_id: str = ATTEMPT_ID,
    retention_session_id: str = SESSION_ID,
    authority_graph: _AuthorityGraph | None = None,
    stage_b_lock_payload: bytes | None = None,
    authority_profile_payload: bytes | None = None,
    signer_registry_payload: bytes | None = None,
    out_of_band_expected_lock_id: str | None = None,
    out_of_band_expected_lock_payload_sha256: str | None = None,
) -> store.CasStoreV1:
    graph = (
        _authority_graph(seed) if authority_graph is None else authority_graph
    )
    return store.CasStoreV1(
        root_directory=str(root),
        chain_id=CHAIN_ID,
        attempt_context_id=attempt_context_id,
        retention_session_id=retention_session_id,
        private_key_seed=seed,
        allowed_peer_uids=frozenset({os.geteuid()}),
        stage_b_lock_payload=(
            graph.stage_b_lock_payload
            if stage_b_lock_payload is None
            else stage_b_lock_payload
        ),
        authority_profile_payload=(
            graph.authority_profile_payload
            if authority_profile_payload is None
            else authority_profile_payload
        ),
        signer_registry_payload=(
            graph.signer_registry_payload
            if signer_registry_payload is None
            else signer_registry_payload
        ),
        out_of_band_expected_lock_id=(
            graph.out_of_band_expected_lock_id
            if out_of_band_expected_lock_id is None
            else out_of_band_expected_lock_id
        ),
        out_of_band_expected_lock_payload_sha256=(
            graph.out_of_band_expected_lock_payload_sha256
            if out_of_band_expected_lock_payload_sha256 is None
            else out_of_band_expected_lock_payload_sha256
        ),
        create=create,
    )


class _OpenArtifact:
    def __init__(self, path: Path, payload: bytes) -> None:
        path.write_bytes(payload)
        self.path = path
        self.fd = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )

    def close(self) -> None:
        os.close(self.fd)

    def __enter__(self) -> "_OpenArtifact":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.close()
        return False


def _ingest(
    cas: store.CasStoreV1,
    tmp_path: Path,
    payload: bytes,
    *,
    ordinal: int = 0,
    attempt_context_id: str = ATTEMPT_ID,
    retention_session_id: str = SESSION_ID,
) -> store.StoreResponseV1:
    request = _request(
        payload,
        ordinal=ordinal,
        attempt_context_id=attempt_context_id,
        retention_session_id=retention_session_id,
    )
    artifact_path = tmp_path / (
        f"artifact-{attempt_context_id}-{retention_session_id}-{ordinal}.bin"
    )
    with _OpenArtifact(artifact_path, payload) as artifact:
        return cas.ingest_open_fd(request, artifact.fd, _peer())


def _rewrite_canonical(path: Path, mapping: dict[str, object]) -> None:
    path.write_bytes(store.canonicalize_rfc8785_v1(mapping))


def _verify_receipt(
    cas: store.CasStoreV1,
    receipt: store.SignedStoreReceiptV1,
) -> store.StoreReceiptBodyV1:
    return store.verify_signed_store_receipt_v1(
        receipt,
        expected_public_key=cas.signer_public_key,
        expected_signer_key_id=cas.signer_key_id,
        expected_authority_profile_id=cas.authority_profile_id,
        expected_signer_registry_id=cas.signer_registry_id,
    )


def test_exact_store_source_hash_and_non_authority_boundary() -> None:
    source = Path(store.__file__).read_bytes()
    assert hashlib.sha256(source).hexdigest() == EXPECTED_STORE_SOURCE_SHA256
    assert store.STORE_REQUEST_SCHEMA_VERSION.endswith("/1")
    assert store.STORE_REQUEST_SCHEMA_VERSION == runtime.RETENTION_REQUEST_SCHEMA_VERSION
    assert store.RETENTION_ACK_SCHEMA_VERSION == runtime.RETENTION_ACK_SCHEMA_VERSION
    assert store.EXPECTED_OBJECT_ORDER == runtime.EXPECTED_OBJECT_ORDER
    assert store.EXPECTED_RETAINED_OBJECTS == 12
    assert store.EXPECTED_CLEANUP_CALLS == 4
    assert store.INVARIANCE_WORKER_ARTIFACT_BYTES_BOUND == 1_048_576
    assert store.MAXIMUM_STATISTICS_ARTIFACT_BYTES == 17_825_792
    assert store.CAS_STORE_RECEIPT_PURPOSE == "SIGN_CAS_IF_ABSENT_RECEIPT"
    assert store.CAS_STORE_RECEIPT_PURPOSE in (
        authority.ROLE_SIGNATURE_PURPOSES_V1["CAS_STORE"]
    )
    assert store.RUNTIME_SOURCE_SHA256_PIN == (
        "35315054bf360a6c7148e79b018f7657a16b19d5f833563693461830422b4979"
    )
    assert hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest() == (
        store.RUNTIME_SOURCE_SHA256_PIN
    )
    assert hashlib.sha256(Path(authority.__file__).read_bytes()).hexdigest() == (
        store.AUTHORITY_SOURCE_SHA256_PIN
    )
    assert hashlib.sha256(Path(runtime.auditor.__file__).read_bytes()).hexdigest() == (
        store.AUDITOR_SOURCE_SHA256_PIN
    )
    assert hashlib.sha256(Path(runtime.qv2.__file__).read_bytes()).hexdigest() == (
        store.QV2_SOURCE_SHA256_PIN
    )
    assert "SAME_PROCESS" in store.AUTHORITY_CLAIM_BOUNDARY
    assert "SAME_UID" in store.AUTHORITY_CLAIM_BOUNDARY
    assert "LOCAL_FILESYSTEM" in store.AUTHORITY_CLAIM_BOUNDARY
    assert "INDEPENDENT_UID_CONTROL_DOMAIN" in store.AUTHORITY_CLAIM_BOUNDARY
    assert "LIVE_RUNTIME_CLOSURE" in store.AUTHORITY_CLAIM_BOUNDARY


def test_server_import_path_does_not_load_producer_modules() -> None:
    source_root = Path(store.__file__).resolve().parents[1]
    forbidden = (
        "hegel_machine.phase2b_formal_covert_auditor_v1",
        "hegel_machine.phase2b_formal_wire_covert_qualification_v2",
        "hegel_machine.phase2b_formal_wire_covert_step5_runtime_v1",
    )
    program = "\n".join(
        (
            "import sys",
            f"sys.path.insert(0, {str(source_root)!r})",
            "import hegel_machine.phase2b_formal_wire_covert_step5_store_v1",
            f"forbidden = {forbidden!r}",
            "loaded = [name for name in forbidden if name in sys.modules]",
            "assert loaded == [], loaded",
        )
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root)
    completed = subprocess.run(
        [sys.executable, "-I", "-c", program],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_store_bootstrap_accepts_only_bytes_and_rejects_private_token_laundering(
    tmp_path: Path,
) -> None:
    parameters = inspect.signature(store.CasStoreV1.__init__).parameters
    assert {
        "stage_b_lock_payload",
        "authority_profile_payload",
        "signer_registry_payload",
        "out_of_band_expected_lock_id",
        "out_of_band_expected_lock_payload_sha256",
    } <= set(parameters)
    assert {"trust_lock", "authority_profile", "signer_registry"}.isdisjoint(
        parameters
    )

    fake_lock = authority._issue("stage_b_trust_lock", TRUST_LOCK_ID, {})
    graph = DEFAULT_AUTHORITY_GRAPH
    with pytest.raises(TypeError, match="unexpected keyword argument 'trust_lock'"):
        store.CasStoreV1(
            root_directory=str(tmp_path / "cas-token-laundering"),
            chain_id=CHAIN_ID,
            attempt_context_id=ATTEMPT_ID,
            retention_session_id=SESSION_ID,
            private_key_seed=SEED,
            allowed_peer_uids=frozenset({os.geteuid()}),
            stage_b_lock_payload=graph.stage_b_lock_payload,
            authority_profile_payload=graph.authority_profile_payload,
            signer_registry_payload=graph.signer_registry_payload,
            out_of_band_expected_lock_id=graph.out_of_band_expected_lock_id,
            out_of_band_expected_lock_payload_sha256=(
                graph.out_of_band_expected_lock_payload_sha256
            ),
            trust_lock=fake_lock,  # type: ignore[call-arg]
            create=True,
        )


def test_store_bootstrap_rejects_wrong_oob_pins_and_noncanonical_or_typed_bytes(
    tmp_path: Path,
) -> None:
    graph = DEFAULT_AUTHORITY_GRAPH
    with pytest.raises(authority.Phase2BStep5AuthorityErrorV1):
        _new_store(
            tmp_path / "cas-wrong-oob-id",
            create=True,
            out_of_band_expected_lock_id=(
                authority.STAGE_B_TRUST_LOCK_PREFIX_V1 + "f" * 64
            ),
        )
    with pytest.raises(authority.Phase2BStep5AuthorityErrorV1):
        _new_store(
            tmp_path / "cas-wrong-oob-sha",
            create=True,
            out_of_band_expected_lock_payload_sha256="f" * 64,
        )
    duplicate_lock = b'{"body":null,' + graph.stage_b_lock_payload[1:]
    with pytest.raises(authority.Phase2BStep5AuthorityErrorV1):
        _new_store(
            tmp_path / "cas-duplicate-lock-key",
            create=True,
            stage_b_lock_payload=duplicate_lock,
        )
    with pytest.raises(authority.Phase2BStep5AuthorityErrorV1):
        _new_store(
            tmp_path / "cas-noncanonical-profile",
            create=True,
            authority_profile_payload=graph.authority_profile_payload + b" ",
        )
    with pytest.raises(TypeError, match="exact bytes"):
        _new_store(
            tmp_path / "cas-bytearray-lock",
            create=True,
            stage_b_lock_payload=bytearray(  # type: ignore[arg-type]
                graph.stage_b_lock_payload
            ),
        )


def test_store_bootstrap_rejects_spliced_unsigned_and_cross_graph_artifacts(
    tmp_path: Path,
) -> None:
    graph = DEFAULT_AUTHORITY_GRAPH
    other = _authority_graph(variant="other")

    with pytest.raises(authority.Phase2BStep5AuthorityErrorV1):
        _new_store(
            tmp_path / "cas-cross-profile",
            create=True,
            authority_profile_payload=other.authority_profile_payload,
        )
    with pytest.raises(authority.Phase2BStep5AuthorityErrorV1):
        _new_store(
            tmp_path / "cas-cross-registry",
            create=True,
            signer_registry_payload=other.signer_registry_payload,
        )

    spliced_profile = copy.deepcopy(graph.authority_profile)
    spliced_profile_body = spliced_profile["body"]
    assert type(spliced_profile_body) is dict
    source_binding = spliced_profile_body["source_binding"]
    assert type(source_binding) is dict
    source_binding["source_commit_sha"] = "c" * 40
    spliced_profile_graph = _repin_authority_graph(
        spliced_profile,
        copy.deepcopy(graph.signer_registry),
    )
    with pytest.raises(authority.Phase2BStep5AuthorityErrorV1):
        _new_store(
            tmp_path / "cas-spliced-profile",
            create=True,
            authority_graph=spliced_profile_graph,
        )

    unsigned_registry = copy.deepcopy(graph.signer_registry)
    unsigned_registry["signatures"] = []
    unsigned_graph = _repin_authority_graph(
        copy.deepcopy(graph.authority_profile),
        unsigned_registry,
    )
    with pytest.raises(authority.Phase2BStep5AuthorityErrorV1):
        _new_store(
            tmp_path / "cas-unsigned-registry",
            create=True,
            authority_graph=unsigned_graph,
        )

    bad_signature_registry = copy.deepcopy(graph.signer_registry)
    signatures = bad_signature_registry["signatures"]
    assert type(signatures) is list and type(signatures[0]) is dict
    signature_hex = signatures[0]["signature_hex"]
    assert type(signature_hex) is str
    signatures[0]["signature_hex"] = (
        ("0" if signature_hex[0] != "0" else "1") + signature_hex[1:]
    )
    bad_signature_graph = _repin_authority_graph(
        copy.deepcopy(graph.authority_profile),
        bad_signature_registry,
    )
    with pytest.raises(authority.Phase2BStep5AuthorityErrorV1):
        _new_store(
            tmp_path / "cas-bad-registry-signature",
            create=True,
            authority_graph=bad_signature_graph,
        )


def test_store_bootstrap_rejects_fully_repinned_role_policy_and_key_drift(
    tmp_path: Path,
) -> None:
    graph = DEFAULT_AUTHORITY_GRAPH

    bad_profile = copy.deepcopy(graph.authority_profile)
    bad_profile_body = bad_profile["body"]
    assert type(bad_profile_body) is dict
    role_policy = bad_profile_body["role_policy"]
    assert type(role_policy) is dict
    role_policy["role_count"] = 7
    bad_profile["id"] = authority.phase2b_step5_content_id_v1(
        "authority_profile", bad_profile_body
    )
    relinked_registry_body = copy.deepcopy(graph.signer_registry["body"])
    assert type(relinked_registry_body) is dict
    relinked_registry_body["authority_profile_id"] = bad_profile["id"]
    bad_profile_graph = _repin_authority_graph(
        bad_profile,
        _resign_registry(relinked_registry_body),
    )
    with pytest.raises(authority.Phase2BStep5AuthorityErrorV1):
        _new_store(
            tmp_path / "cas-role-policy-drift",
            create=True,
            authority_graph=bad_profile_graph,
        )

    for name, changes in (
        ("purpose", {"authorized_signature_purposes": ["OTHER_PURPOSE"]}),
        ("key-id", {"key_id": authority.ROLE_KEY_PREFIX_V1 + "f" * 64}),
    ):
        registry_body = copy.deepcopy(graph.signer_registry["body"])
        assert type(registry_body) is dict
        rows = registry_body["role_keys_in_exact_order"]
        assert type(rows) is list
        cas_row = rows[authority.ROLE_ORDER_V1.index("CAS_STORE")]
        assert type(cas_row) is dict
        cas_row.update(changes)
        mutated_graph = _repin_authority_graph(
            copy.deepcopy(graph.authority_profile),
            _resign_registry(registry_body),
        )
        with pytest.raises(authority.Phase2BStep5AuthorityErrorV1):
            _new_store(
                tmp_path / f"cas-role-{name}-drift",
                create=True,
                authority_graph=mutated_graph,
            )


def test_store_requires_exact_verified_cas_role_policy_key_and_private_seed(
    tmp_path: Path,
) -> None:
    cas_row = DEFAULT_AUTHORITY_GRAPH.cas_row
    root = tmp_path / "cas-valid-authority"
    with _new_store(root, create=True) as cas:
        assert cas.authority_profile_id == AUTHORITY_PROFILE_ID
        assert cas.signer_registry_id == SIGNER_REGISTRY_ID
        assert cas.key_epoch == 1
        assert cas.signer_key_id == cas_row["key_id"]
        assert cas.signer_key_id.startswith(authority.ROLE_KEY_PREFIX_V1)
        assert cas.signer_key_id != (
            authority.ROLE_KEY_PREFIX_V1
            + hashlib.sha256(cas.signer_public_key).hexdigest()
        )

    with pytest.raises(ValueError, match="private seed/public registry key"):
        _new_store(
            tmp_path / "cas-private-key-drift",
            create=True,
            seed=OTHER_SEED,
            authority_graph=_authority_graph(SEED),
        )


def test_request_canonical_roundtrip_duplicate_trailing_unknown_and_type() -> None:
    request = _request(b"canonical-worker")
    payload = request.canonical_bytes()
    assert store.StoreRequestV1.from_canonical_bytes(payload) == request

    text = payload.decode("utf-8")
    duplicate = (
        '{"schema_version":"duplicate",' + text.removeprefix("{")
    ).encode("utf-8")
    with pytest.raises(ValueError):
        store.StoreRequestV1.from_canonical_bytes(duplicate)
    with pytest.raises(ValueError):
        store.StoreRequestV1.from_canonical_bytes(payload + b" ")
    with pytest.raises(ValueError):
        store.StoreRequestV1.from_canonical_bytes(b"{}")
    with pytest.raises(ValueError):
        store.StoreRequestV1.from_canonical_bytes(bytearray(payload))  # type: ignore[arg-type]

    body = request.to_mapping()
    body["unknown"] = True
    with pytest.raises(ValueError, match="field set"):
        store.StoreRequestV1.from_mapping(body)
    with pytest.raises(ValueError, match="field set"):
        store.StoreRequestV1.from_mapping({})


def test_request_id_and_ordinal_bind_every_runtime_identity() -> None:
    request = _request(b"request-bind", ordinal=1)
    unsigned = request.to_mapping()
    observed_request_id = unsigned.pop("request_id")
    assert observed_request_id == _sha256(store.canonicalize_rfc8785_v1(unsigned))

    with pytest.raises(ValueError, match="request ID"):
        replace(request, request_id="sha256:" + "f" * 64)
    with pytest.raises(ValueError, match="ordinal/mode/phase/target"):
        replace(request, mode="KNOWN_BINDING_LEAK")
    with pytest.raises(ValueError, match="ordinal/mode/phase/target"):
        replace(request, phase="INVARIANCE")
    with pytest.raises(ValueError, match="ordinal/mode/phase/target"):
        replace(request, target_index_or_null=False)
    with pytest.raises(ValueError, match="basename"):
        replace(request, source_basename="target-1.json")
    with pytest.raises(ValueError, match="exactly one descriptor"):
        replace(request, fd_count=2)

    body = request.to_mapping()
    body["mode"] = "KNOWN_BINDING_LEAK"
    unsigned = dict(body)
    unsigned.pop("request_id")
    body["request_id"] = _sha256(store.canonicalize_rfc8785_v1(unsigned))
    with pytest.raises(ValueError, match="ordinal/mode/phase/target"):
        store.StoreRequestV1.from_mapping(body)


def test_seqpacket_scm_rights_exact_one_and_so_peercred() -> None:
    payload = b"scm-one"
    request = _request(payload)
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    artifact_path = Path("/tmp") / f"step5-store-scm-{os.getpid()}.bin"
    try:
        with _OpenArtifact(artifact_path, payload) as artifact:
            store.send_store_request_fd_v1(left, request, artifact.fd)
            with store.recv_store_request_v1(
                right,
                allowed_peer_uids=frozenset({os.geteuid()}),
            ) as received:
                assert received.request == request
                assert received.peer == _peer()
                assert os.pread(received.fd, len(payload), 0) == payload
                assert os.get_inheritable(received.fd) is False
    finally:
        left.close()
        right.close()
        artifact_path.unlink(missing_ok=True)


def test_scm_rights_zero_extra_and_control_truncation_close_received_fds(
    tmp_path: Path,
) -> None:
    request = _request(b"scm-cardinality")

    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        left.send(request.canonical_bytes())
        with pytest.raises(ValueError, match="exactly one"):
            store.recv_store_request_v1(
                right,
                allowed_peer_uids=frozenset({os.geteuid()}),
            )
    finally:
        left.close()
        right.close()

    paths = [tmp_path / f"extra-{index}.bin" for index in range(12)]
    descriptors: list[int] = []
    try:
        for path in paths:
            path.write_bytes(b"x")
            descriptors.append(os.open(path, os.O_RDONLY | os.O_CLOEXEC))

        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            before = len(os.listdir("/proc/self/fd"))
            rights = array("i", descriptors[:2])
            left.sendmsg(
                [request.canonical_bytes()],
                [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights.tobytes())],
            )
            with pytest.raises(ValueError, match="exactly one"):
                store.recv_store_request_v1(
                    right,
                    allowed_peer_uids=frozenset({os.geteuid()}),
                )
            assert len(os.listdir("/proc/self/fd")) == before
        finally:
            left.close()
            right.close()

        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            before = len(os.listdir("/proc/self/fd"))
            rights = array("i", descriptors)
            left.sendmsg(
                [request.canonical_bytes()],
                [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights.tobytes())],
            )
            with pytest.raises(ValueError, match="truncated|exactly one"):
                store.recv_store_request_v1(
                    right,
                    allowed_peer_uids=frozenset({os.geteuid()}),
                )
            assert len(os.listdir("/proc/self/fd")) == before
        finally:
            left.close()
            right.close()
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def test_scm_payload_truncation_extra_credentials_uid_and_socket_type(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "scm-trunc.bin"
    with _OpenArtifact(artifact_path, b"x") as artifact:
        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            rights = array("i", [artifact.fd])
            left.sendmsg(
                [b"x" * (store.MAXIMUM_REQUEST_BYTES + 128)],
                [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights.tobytes())],
            )
            with pytest.raises(ValueError, match="truncated"):
                store.recv_store_request_v1(
                    right,
                    allowed_peer_uids=frozenset({os.geteuid()}),
                )
        finally:
            left.close()
            right.close()

        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            right.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
            store.send_store_request_fd_v1(left, _request(b"x"), artifact.fd)
            with pytest.raises(ValueError, match="unexpected control"):
                store.recv_store_request_v1(
                    right,
                    allowed_peer_uids=frozenset({os.geteuid()}),
                )
        finally:
            left.close()
            right.close()

        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            with pytest.raises(PermissionError, match="UID"):
                store.recv_store_request_v1(
                    right,
                    allowed_peer_uids=frozenset({os.geteuid() + 100_000}),
                )
        finally:
            left.close()
            right.close()

    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        with pytest.raises(ValueError, match="SOCK_SEQPACKET"):
            store.recv_store_request_v1(
                right,
                allowed_peer_uids=frozenset({os.geteuid()}),
            )
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize(
    ("reported_sha256", "reported_size"),
    (
        ("sha256:" + "f" * 64, None),
        (None, 1),
    ),
)
def test_server_recomputes_hash_and_size_instead_of_trusting_request(
    tmp_path: Path,
    reported_sha256: str | None,
    reported_size: int | None,
) -> None:
    payload = b"server-independent-hash"
    request = _request(
        payload,
        reported_sha256=reported_sha256,
        reported_size_bytes=reported_size,
    )
    root = tmp_path / "cas"
    with _new_store(root, create=True) as cas:
        with _OpenArtifact(tmp_path / "worker.bin", payload) as artifact:
            with pytest.raises(ValueError, match="differs from runtime request"):
                cas.ingest_open_fd(request, artifact.fd, _peer())
        assert cas.audit_chain() == ()


def test_store_freezes_session_enforces_ordinal_order_and_audits_exact_twelve(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cas-complete-session"
    with _new_store(root, create=True) as cas:
        with pytest.raises(RuntimeError, match="exact complete 12"):
            cas.audit_complete_session()

        with _OpenArtifact(tmp_path / "out-of-order.bin", b"out-of-order") as artifact:
            with pytest.raises(ValueError, match="ordinal order"):
                cas.ingest_open_fd(
                    _request(b"out-of-order", ordinal=1),
                    artifact.fd,
                    _peer(),
                )
        with _OpenArtifact(tmp_path / "cross-attempt.bin", b"cross-attempt") as artifact:
            with pytest.raises(ValueError, match="frozen CAS store session"):
                cas.ingest_open_fd(
                    _request(
                        b"cross-attempt",
                        attempt_context_id=OTHER_ATTEMPT_ID,
                    ),
                    artifact.fd,
                    _peer(),
                )
        with _OpenArtifact(tmp_path / "cross-session.bin", b"cross-session") as artifact:
            with pytest.raises(ValueError, match="frozen CAS store session"):
                cas.ingest_open_fd(
                    _request(
                        b"cross-session",
                        retention_session_id=OTHER_SESSION_ID,
                    ),
                    artifact.fd,
                    _peer(),
                )
        assert cas.audit_chain() == ()

        responses = tuple(
            _ingest(cas, tmp_path, f"complete-{ordinal}".encode(), ordinal=ordinal)
            for ordinal in range(store.EXPECTED_RETAINED_OBJECTS)
        )
        records = cas.audit_complete_session()
        assert len(records) == len(responses) == 12
        assert tuple(record.checkpoint_index for record in records) == tuple(range(12))
        assert tuple(
            (
                record.signed_receipt.signed_body.mode,
                record.signed_receipt.signed_body.phase,
                record.signed_receipt.signed_body.target_index_or_null,
            )
            for record in records
        ) == store.EXPECTED_OBJECT_ORDER


def test_committed_response_loss_retry_is_idempotent_and_drift_fails_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cas-response-loss"
    payload = b"response-loss-retry"
    request = _request(payload)
    artifact_path = tmp_path / "response-loss.bin"
    with _new_store(root, create=True) as cas:
        # First connection commits, but the simulated server never sends the
        # returned response.  The client therefore retries the exact packet.
        first_left, first_right = socket.socketpair(
            socket.AF_UNIX, socket.SOCK_SEQPACKET
        )
        try:
            with _OpenArtifact(artifact_path, payload) as artifact:
                store.send_store_request_fd_v1(first_left, request, artifact.fd)
                lost_response = cas.receive_and_ingest_once(first_right)
        finally:
            first_left.close()
            first_right.close()

        retry_left, retry_right = socket.socketpair(
            socket.AF_UNIX, socket.SOCK_SEQPACKET
        )
        try:
            with _OpenArtifact(artifact_path, payload) as artifact:
                store.send_store_request_fd_v1(retry_left, request, artifact.fd)
                retry_response = cas.receive_and_ingest_once(retry_right)
        finally:
            retry_left.close()
            retry_right.close()
        assert retry_response == lost_response
        assert len(cas.audit_chain()) == 1

        with _OpenArtifact(tmp_path / "wrong-peer.bin", payload) as artifact:
            wrong_peer = replace(_peer(), pid=os.getpid() + 1)
            with pytest.raises(ValueError, match="peer drift"):
                cas.ingest_open_fd(request, artifact.fd, wrong_peer)
        with _OpenArtifact(tmp_path / "wrong-fd.bin", b"different-retry") as artifact:
            with pytest.raises(ValueError, match="differs from runtime request"):
                cas.ingest_open_fd(request, artifact.fd, _peer())
        assert len(cas.audit_chain()) == 1


def test_source_hardlink_same_inode_mutation_and_oversize_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cas"
    with _new_store(root, create=True) as cas:
        payload = b"hardlink-source"
        path = tmp_path / "hardlink.bin"
        path.write_bytes(payload)
        os.link(path, tmp_path / "hardlink-alias.bin")
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            with pytest.raises(ValueError, match="regular link"):
                cas.ingest_open_fd(_request(payload), fd, _peer())
        finally:
            os.close(fd)

        mutation_payload = b"mutation-source"
        mutation_path = tmp_path / "mutation.bin"
        original_hash = store.CasStoreV1._hash_received_fd

        def mutate_after_hash(
            self: store.CasStoreV1,
            request: store.StoreRequestV1,
            source_fd: int,
        ) -> tuple[os.stat_result, str, int]:
            result = original_hash(self, request, source_fd)
            write_fd = os.open(mutation_path, os.O_WRONLY | os.O_CLOEXEC)
            try:
                os.pwrite(write_fd, b"Z" * len(mutation_payload), 0)
                os.fsync(write_fd)
            finally:
                os.close(write_fd)
            return result

        monkeypatch.setattr(
            store.CasStoreV1,
            "_hash_received_fd",
            mutate_after_hash,
        )
        with _OpenArtifact(mutation_path, mutation_payload) as artifact:
            with pytest.raises(ValueError, match="changed during CAS copy"):
                cas.ingest_open_fd(_request(mutation_payload), artifact.fd, _peer())
        monkeypatch.setattr(
            store.CasStoreV1,
            "_hash_received_fd",
            original_hash,
        )

        oversize_path = tmp_path / "oversize.bin"
        with oversize_path.open("wb") as stream:
            stream.truncate(store.INVARIANCE_WORKER_ARTIFACT_BYTES_BOUND + 1)
        oversize_fd = os.open(
            oversize_path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        try:
            request = _request(
                b"x",
                reported_sha256=_sha256(b"x"),
                reported_size_bytes=1,
            )
            with pytest.raises(ValueError, match="bounded regular link"):
                cas.ingest_open_fd(request, oversize_fd, _peer())
        finally:
            os.close(oversize_fd)
        assert cas.audit_chain() == ()


@pytest.mark.parametrize("namespace", ("objects", "checkpoints", "state"))
def test_store_reopen_rejects_symlinked_persistent_entries(
    tmp_path: Path,
    namespace: str,
) -> None:
    root = tmp_path / f"cas-{namespace}"
    with _new_store(root, create=True) as cas:
        response = _ingest(cas, tmp_path, b"symlink-namespace")
    outside = tmp_path / f"outside-{namespace}.json"
    if namespace == "objects":
        victim = root / "objects" / response.acknowledgement.sha256[7:]
    elif namespace == "checkpoints":
        victim = root / "checkpoints" / "00000000000000000000.json"
    else:
        victim = root / "state" / "head.json"
    outside.write_bytes(victim.read_bytes())
    victim.unlink()
    victim.symlink_to(outside)
    with pytest.raises((OSError, ValueError)):
        _new_store(root, create=False)


def test_cas_if_absent_new_same_bytes_and_conflicting_existing_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cas"
    payload = b"if-absent-object"
    with _new_store(root, create=True) as cas:
        created = _ingest(cas, tmp_path, payload)
        assert created.checkpoint_record.signed_receipt.signed_body.object_disposition == (
            "CREATED"
        )
        present = _ingest(
            cas,
            tmp_path,
            payload,
            ordinal=1,
        )
        assert present.checkpoint_record.signed_receipt.signed_body.object_disposition == (
            "PRESENT_VERIFIED"
        )
        assert created.acknowledgement.content_address == (
            present.acknowledgement.content_address
        )
        object_path = root / "objects" / created.acknowledgement.sha256[7:]
        assert object_path.read_bytes() == payload
        assert object_path.stat().st_nlink == 1

        duplicate_request = _request(payload)
        with _OpenArtifact(tmp_path / "duplicate.bin", payload) as artifact:
            replay = cas.ingest_open_fd(duplicate_request, artifact.fd, _peer())
        assert replay == created
        assert len(cas.audit_chain()) == 2

    conflict_root = tmp_path / "cas-conflict"
    with _new_store(conflict_root, create=True) as cas:
        digest = _sha256(payload)
        (conflict_root / "objects" / digest[7:]).write_bytes(b"different-bytes")
        with _OpenArtifact(tmp_path / "conflict.bin", payload) as artifact:
            with pytest.raises(ValueError, match="readback|bounded"):
                cas.ingest_open_fd(_request(payload), artifact.fd, _peer())
        assert tuple((conflict_root / "checkpoints").iterdir()) == ()
        with pytest.raises(ValueError, match="content-address binding"):
            cas.audit_chain()


def test_fsync_readback_checkpoint_head_and_reopen_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cas"
    cas = _new_store(root, create=True)
    calls: list[int] = []
    original_fsync = os.fsync

    def recording_fsync(fd: int) -> None:
        calls.append(fd)
        original_fsync(fd)

    monkeypatch.setattr(store.os, "fsync", recording_fsync)
    response = _ingest(cas, tmp_path, b"durability-chain")
    cas.close()
    assert len(calls) >= 8
    assert tuple((root / "staging").iterdir()) == ()
    object_path = root / "objects" / response.acknowledgement.sha256[7:]
    assert object_path.read_bytes() == b"durability-chain"

    checkpoint_path = root / "checkpoints" / "00000000000000000000.json"
    checkpoint_payload = checkpoint_path.read_bytes()
    checkpoint = store.StoreCheckpointV1.from_canonical_bytes(checkpoint_payload)
    assert checkpoint == response.checkpoint_record
    assert checkpoint.previous_checkpoint_sha256 == store.GENESIS_CHECKPOINT_SHA256
    assert checkpoint.checkpoint_sha256 == response.checkpoint_sha256

    head = store.StoreHeadV1.from_canonical_bytes(
        (root / "state" / "head.json").read_bytes()
    )
    assert head.checkpoint_count == 1
    assert head.last_checkpoint_sha256 == checkpoint.checkpoint_sha256

    with _new_store(root, create=False) as reopened:
        assert reopened.audit_chain() == (checkpoint,)


def test_receipt_checkpoint_head_and_response_bind_authority_graph(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cas-authority-bindings"
    with _new_store(root, create=True) as cas:
        response = _ingest(cas, tmp_path, b"authority-binding")
        checkpoint = response.checkpoint_record
        receipt = checkpoint.signed_receipt
        body = _verify_receipt(cas, receipt)
        for value in (body, receipt, checkpoint, response):
            assert value.authority_profile_id == AUTHORITY_PROFILE_ID
            assert value.signer_registry_id == SIGNER_REGISTRY_ID
            assert value.key_epoch == 1
        assert body.signer_key_id == cas.signer_key_id
        assert receipt.signer_key_id == cas.signer_key_id
        assert body.purpose == "SIGN_CAS_IF_ABSENT_RECEIPT"
        assert receipt.purpose == "SIGN_CAS_IF_ABSENT_RECEIPT"
        parsed = store.StoreResponseV1.from_canonical_bytes(
            response.canonical_bytes(),
            expected_public_key=cas.signer_public_key,
            expected_signer_key_id=cas.signer_key_id,
            expected_authority_profile_id=AUTHORITY_PROFILE_ID,
            expected_signer_registry_id=SIGNER_REGISTRY_ID,
        )
        assert parsed == response

        with pytest.raises(ValueError, match="pinned authority binding"):
            store.StoreResponseV1.from_canonical_bytes(
                response.canonical_bytes(),
                expected_public_key=cas.signer_public_key,
                expected_signer_key_id=cas.signer_key_id,
                expected_authority_profile_id=(
                    authority.AUTHORITY_PROFILE_PREFIX_V1 + "d" * 64
                ),
                expected_signer_registry_id=SIGNER_REGISTRY_ID,
            )

    head = store.StoreHeadV1.from_canonical_bytes(
        (root / "state" / "head.json").read_bytes()
    )
    assert head.authority_profile_id == AUTHORITY_PROFILE_ID
    assert head.signer_registry_id == SIGNER_REGISTRY_ID
    assert head.key_epoch == 1

    spliced_head = replace(
        head,
        authority_profile_id=authority.AUTHORITY_PROFILE_PREFIX_V1 + "e" * 64,
    )
    (root / "state" / "head.json").write_bytes(spliced_head.canonical_bytes())
    with pytest.raises(ValueError, match="head chain/authority binding"):
        _new_store(root, create=False)


def _two_checkpoint_store(
    tmp_path: Path,
    name: str,
) -> tuple[Path, store.StoreResponseV1, store.StoreResponseV1]:
    root = tmp_path / name
    with _new_store(root, create=True) as cas:
        first = _ingest(cas, tmp_path, f"{name}-zero".encode())
        second = _ingest(
            cas,
            tmp_path,
            f"{name}-one".encode(),
            ordinal=1,
        )
    return root, first, second


def test_checkpoint_gap_and_missing_tail_with_ahead_head_reject(
    tmp_path: Path,
) -> None:
    root, _, _ = _two_checkpoint_store(tmp_path, "gap")
    checkpoints = root / "checkpoints"
    (checkpoints / "00000000000000000001.json").rename(
        checkpoints / "00000000000000000002.json"
    )
    with pytest.raises(ValueError, match="contiguous"):
        _new_store(root, create=False)

    root, _, _ = _two_checkpoint_store(tmp_path, "missing-tail")
    (root / "checkpoints" / "00000000000000000001.json").unlink()
    with pytest.raises(ValueError, match="ahead"):
        _new_store(root, create=False)


def test_validly_signed_checkpoint_fork_is_rejected(tmp_path: Path) -> None:
    root, _, second = _two_checkpoint_store(tmp_path, "fork")
    original = second.checkpoint_record
    fork_body = replace(
        original.signed_receipt.signed_body,
        previous_checkpoint_sha256=store.GENESIS_CHECKPOINT_SHA256,
    )
    cas_row = DEFAULT_AUTHORITY_GRAPH.cas_row
    signer = store.Ed25519CasStoreSignerV1(
        SEED,
        signer_key_id=cas_row["key_id"],
        expected_public_key=bytes.fromhex(cas_row["public_key_hex"]),
    )
    fork_receipt = signer.sign(fork_body)
    fork = store.StoreCheckpointV1(
        schema_version=store.STORE_CHECKPOINT_SCHEMA_VERSION,
        purpose=store.CAS_STORE_CHECKPOINT_PURPOSE,
        chain_id=CHAIN_ID,
        authority_profile_id=AUTHORITY_PROFILE_ID,
        signer_registry_id=SIGNER_REGISTRY_ID,
        key_epoch=1,
        checkpoint_index=1,
        previous_checkpoint_sha256=store.GENESIS_CHECKPOINT_SHA256,
        request_id=original.request_id,
        content_address=original.content_address,
        size_bytes=original.size_bytes,
        signed_receipt_sha256=fork_receipt.receipt_sha256,
        signed_receipt=fork_receipt,
    )
    (root / "checkpoints" / "00000000000000000001.json").write_bytes(
        fork.canonical_bytes()
    )
    with pytest.raises(ValueError, match="continuity"):
        _new_store(root, create=False)


def test_head_prefix_crash_repair_and_nonprefix_rollback_reject(
    tmp_path: Path,
) -> None:
    root, first, second = _two_checkpoint_store(tmp_path, "head-repair")
    prefix = store.StoreHeadV1(
        schema_version=store.STORE_HEAD_SCHEMA_VERSION,
        purpose=store.CAS_STORE_HEAD_PURPOSE,
        chain_id=CHAIN_ID,
        authority_profile_id=AUTHORITY_PROFILE_ID,
        signer_registry_id=SIGNER_REGISTRY_ID,
        key_epoch=1,
        checkpoint_count=1,
        last_checkpoint_sha256=first.checkpoint_sha256,
    )
    (root / "state" / "head.json").write_bytes(prefix.canonical_bytes())
    with _new_store(root, create=False) as reopened:
        assert len(reopened.audit_chain()) == 2
    repaired = store.StoreHeadV1.from_canonical_bytes(
        (root / "state" / "head.json").read_bytes()
    )
    assert repaired.checkpoint_count == 2
    assert repaired.last_checkpoint_sha256 == second.checkpoint_sha256

    invalid = replace(repaired, last_checkpoint_sha256=store.GENESIS_CHECKPOINT_SHA256)
    (root / "state" / "head.json").write_bytes(invalid.canonical_bytes())
    with pytest.raises(ValueError, match="exact prefix"):
        _new_store(root, create=False)


def _splice_signed_body(
    receipt: store.SignedStoreReceiptV1,
    **changes: object,
) -> store.SignedStoreReceiptV1:
    body = replace(receipt.signed_body, **changes)
    return replace(
        receipt,
        signed_body=body,
        signed_body_sha256=_sha256(body.canonical_bytes()),
    )


@pytest.mark.parametrize(
    "changes",
    (
        {"previous_checkpoint_sha256": "sha256:" + "f" * 64},
        {"request_id": "sha256:" + "e" * 64},
        {"peer_pid": os.getpid() + 1},
        {
            "content_address": "sha256:" + "d" * 64,
            "sha256": "sha256:" + "d" * 64,
        },
    ),
    ids=("previous", "request", "peer", "object"),
)
def test_real_ed25519_receipt_body_splices_reject(
    tmp_path: Path,
    changes: dict[str, object],
) -> None:
    root = tmp_path / "cas"
    with _new_store(root, create=True) as cas:
        response = _ingest(cas, tmp_path, b"signed-receipt")
        receipt = response.checkpoint_record.signed_receipt
        body = _verify_receipt(cas, receipt)
        assert body.request_id == response.request_id
        assert body.previous_checkpoint_sha256 == store.GENESIS_CHECKPOINT_SHA256
        assert body.peer_pid == os.getpid()
        assert body.peer_uid == os.geteuid()
        assert body.content_address == response.acknowledgement.content_address
        assert body.purpose == store.CAS_STORE_RECEIPT_PURPOSE
        assert body.signature_domain == store.CAS_STORE_SIGNATURE_DOMAIN
        spliced = _splice_signed_body(receipt, **changes)
        with pytest.raises(ValueError, match="signature"):
            _verify_receipt(cas, spliced)


def test_receipt_purpose_domain_signature_and_key_splices_reject(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cas"
    with _new_store(root, create=True) as cas:
        response = _ingest(cas, tmp_path, b"signed-key")
        receipt = response.checkpoint_record.signed_receipt
        with pytest.raises(ValueError, match="schema/purpose/domain"):
            replace(receipt.signed_body, purpose="OTHER_PURPOSE")
        with pytest.raises(ValueError, match="schema/purpose/domain"):
            replace(receipt, signature_domain="OTHER_DOMAIN")

        signature = bytearray.fromhex(receipt.ed25519_signature_hex)
        signature[0] ^= 1
        bad_signature = replace(
            receipt,
            ed25519_signature_hex=bytes(signature).hex(),
        )
        with pytest.raises(ValueError, match="signature"):
            _verify_receipt(cas, bad_signature)

        other_row = _authority_graph(OTHER_SEED).cas_row
        other_signer = store.Ed25519CasStoreSignerV1(
            OTHER_SEED,
            signer_key_id=other_row["key_id"],
            expected_public_key=bytes.fromhex(other_row["public_key_hex"]),
        )
        other_body = replace(
            receipt.signed_body,
            signer_key_id=other_signer.key_id,
        )
        key_splice = replace(
            receipt,
            signed_body=other_body,
            signed_body_sha256=_sha256(other_body.canonical_bytes()),
            signer_public_key_hex=other_signer.public_key_bytes.hex(),
            signer_key_id=other_signer.key_id,
        )
        with pytest.raises(ValueError, match="signature"):
            store.verify_signed_store_receipt_v1(
                key_splice,
                expected_public_key=other_signer.public_key_bytes,
                expected_signer_key_id=other_signer.key_id,
                expected_authority_profile_id=AUTHORITY_PROFILE_ID,
                expected_signer_registry_id=SIGNER_REGISTRY_ID,
            )
        with pytest.raises(ValueError, match="pinned authority binding"):
            store.verify_signed_store_receipt_v1(
                receipt,
                expected_public_key=other_signer.public_key_bytes,
                expected_signer_key_id=other_signer.key_id,
                expected_authority_profile_id=AUTHORITY_PROFILE_ID,
                expected_signer_registry_id=SIGNER_REGISTRY_ID,
            )


def test_client_sink_returns_exact_runtime_ack_from_short_lived_socketpair(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cas"
    payload = b"client-ack"
    request = _request(payload)
    with _new_store(root, create=True) as cas:
        client_socket, server_socket = socket.socketpair(
            socket.AF_UNIX,
            socket.SOCK_SEQPACKET,
        )
        errors: list[BaseException] = []
        responses: list[store.StoreResponseV1] = []

        def server_once() -> None:
            try:
                responses.append(cas.handle_one_connection(server_socket))
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=server_once, name="step5-store-kat-server")
        worker.start()
        try:
            sink = store.ScmRightsRetentionSinkV1(
                connection=client_socket,
                expected_server_uid=os.geteuid(),
                expected_public_key=cas.signer_public_key,
                expected_signer_key_id=cas.signer_key_id,
                expected_authority_profile_id=cas.authority_profile_id,
                expected_signer_registry_id=cas.signer_registry_id,
            )
            metadata = request.to_mapping()
            frozen_metadata = dict(metadata)
            with _OpenArtifact(tmp_path / "client.bin", payload) as artifact:
                ack = sink.ingest_open_fd(
                    metadata,
                    artifact.fd,
                    request.auditor_reported_sha256,
                    request.auditor_reported_size_bytes,
                )
            assert metadata == frozen_metadata
            assert type(ack) is runtime.RetentionIngestAckV1
            ack.__post_init__()
            assert ack.request_id == request.request_id
            assert ack.sha256 == _sha256(payload)
            assert ack.durable_before_return is True
            assert ack.readback_verified is True
        finally:
            worker.join(timeout=10)
            client_socket.close()
            server_socket.close()
        assert not worker.is_alive()
        assert errors == []
        assert len(responses) == 1
        assert (
            runtime.RetentionIngestAckV1(
                **responses[0].acknowledgement.to_mapping()
            )
            == ack
        )


def test_checkpoint_written_but_head_failure_returns_no_ack_then_repairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cas"
    cas = _new_store(root, create=True)
    original_write_head = store.CasStoreV1._write_head

    def injected_head_failure(
        self: store.CasStoreV1,
        head: store.StoreHeadV1,
    ) -> None:
        del self, head
        raise OSError("injected head persistence failure")

    monkeypatch.setattr(store.CasStoreV1, "_write_head", injected_head_failure)
    with pytest.raises(OSError, match="injected head persistence failure"):
        _ingest(cas, tmp_path, b"partial-head")
    cas.close()
    monkeypatch.setattr(store.CasStoreV1, "_write_head", original_write_head)

    assert len(tuple((root / "checkpoints").iterdir())) == 1
    with _new_store(root, create=False) as recovered:
        records = recovered.audit_chain()
        assert len(records) == 1
        assert records[0].signed_receipt.signed_body.sha256 == _sha256(b"partial-head")


@pytest.mark.parametrize("partial_namespace", ("staging", "checkpoints", "state"))
def test_crash_pending_files_fail_closed(
    tmp_path: Path,
    partial_namespace: str,
) -> None:
    root = tmp_path / f"cas-partial-{partial_namespace}"
    with _new_store(root, create=True):
        pass
    if partial_namespace == "staging":
        pending = root / "staging" / "pending-crash"
    elif partial_namespace == "checkpoints":
        pending = root / "checkpoints" / ".00000000000000000000.json.pending-crash"
    else:
        pending = root / "state" / ".head.json.pending-crash"
    pending.write_bytes(b"partial")
    with pytest.raises((RuntimeError, ValueError), match="nonempty|unknown"):
        _new_store(root, create=False)


def test_object_readback_corruption_injection_fails_before_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cas"
    cas = _new_store(root, create=True)
    original_read = store._read_regular_at

    def corrupt_object_readback(
        directory_fd: int,
        basename: str,
        maximum_bytes: int,
    ) -> bytes:
        payload = original_read(directory_fd, basename, maximum_bytes)
        if directory_fd == cas._objects_fd and len(payload) > 0:
            return bytes([payload[0] ^ 1]) + payload[1:]
        return payload

    monkeypatch.setattr(store, "_read_regular_at", corrupt_object_readback)
    with pytest.raises(ValueError, match="readback"):
        _ingest(cas, tmp_path, b"readback-corruption")
    assert tuple((root / "checkpoints").iterdir()) == ()
    cas.close()
