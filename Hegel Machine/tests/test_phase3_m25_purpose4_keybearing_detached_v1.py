from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hegel_machine.phase3_m25_parent_absence_audit_v1 import (
    CONTENT_PREDICATE_PROFILE_ID,
    PARENT_DSL_VERSION,
    PARENT_FREEZE_VERSION,
    ParentAbsenceAuditEvidence,
    _CONTENT_ABSENCE_PREDICATES,
    _blob_inventory_digest,
    _digest_set_sha256,
    _legacy_source_rows,
    _path_name_receipt,
    build_parent_absence_attestation_fields_v2,
    parent_absence_public_receipt_v1,
)
from hegel_machine.phase3_m25_purpose4_detached_audit_v1 import (
    RUNTIME_SOURCE_BINDING_DOMAIN,
    SNAPSHOT_SCHEMA,
)
from hegel_machine.phase3_m25_purpose4_keybearing_detached_v1 import (
    FAIL_HOST_ORACLE,
    FAIL_KEY,
    FAIL_PROBE,
    FAIL_REPLAY,
    FAIL_SIGNATURE,
    OPERATION_ID,
    OPERATION_PROBE_SCHEMA,
    REQUEST_SCHEMA,
    RESPONSE_SCHEMA,
    Purpose4KeyBearingError,
    build_purpose4_keybearing_request_v1,
    canonical_json_v1,
    validate_purpose4_keybearing_request_v1,
    validate_purpose4_keybearing_response_v1,
)
from hegel_machine.phase3_m25_wire_v1 import (
    AUDITED_PARENT_COMMIT_SHA1,
    LEGACY_PARENT_SOURCE_IDS,
    OBJECT_TAGS,
    candidate_content_root,
    candidate_record_tree_root,
    encode_formal_object,
    external_signature_preimage_v1,
    git_sha1_commit_id,
    id_digest_v1,
)


BASIS = "12" * 20
IMAGE = "hegel-purpose4-test@sha256:" + "34" * 32


def _runtime_bindings() -> tuple[dict[str, object], dict[str, object]]:
    inventory = {
        "files": [],
        "file_count": 0,
        "inventory_sha256": "45" * 32,
    }
    source_body: dict[str, object] = {
        "schema": "hegel-gate17-purpose4-runtime-source-bindings/1",
        "basis_commit_sha1": BASIS,
        "committed_source_files": [],
        "external_git_dependency": {
            "container_path": "/runtime/bin/git",
            "byte_length": 1,
            "sha256": "56" * 32,
        },
    }
    source_body["binding_sha256"] = hashlib.sha256(
        RUNTIME_SOURCE_BINDING_DOMAIN + canonical_json_v1(source_body)
    ).hexdigest()
    return inventory, source_body


def _snapshot_manifest() -> dict[str, object]:
    body: dict[str, object] = {
        "schema": SNAPSHOT_SCHEMA,
        "basis_commit_sha1": BASIS,
        "audited_parent_commit_sha1": AUDITED_PARENT_COMMIT_SHA1.hex(),
        "git_runtime_binding": {
            "container_path": "/runtime/bin/git",
            "byte_length": 1,
            "sha256": "56" * 32,
        },
    }
    body["manifest_sha256"] = hashlib.sha256(canonical_json_v1(body)).hexdigest()
    return body


def _request(key_id: bytes) -> dict[str, object]:
    inventory, sources = _runtime_bindings()
    return build_purpose4_keybearing_request_v1(
        basis_commit=BASIS,
        actor_image_ref=IMAGE,
        snapshot_manifest=_snapshot_manifest(),
        runtime_inventory=inventory,
        runtime_source_bindings=sources,
        audited_at_unix_seconds=1_800_000_000,
        expected_local_key_id=key_id,
    )


def _synthetic_evidence() -> ParentAbsenceAuditEvidence:
    path_row = {
        "repository_path_alias_id_digest": id_digest_v1("repo-path:gate17-test"),
        "raw_repository_path_utf8_bytes": b"Hegel Machine/legacy/source.json",
        "git_object_algorithm_id": 1,
        "git_blob_digest": bytes.fromhex("11" * 20),
        "file_mode": 0o100644,
        "byte_length": 7,
    }
    touched_root = candidate_record_tree_root("AuditedPathBlobRecordV1", [path_row])
    history_row = {
        "commit_generation": 0,
        "repository_commit_id": git_sha1_commit_id(AUDITED_PARENT_COMMIT_SHA1),
        "ordered_parent_commit_ids": (),
        "touched_path_set_root": touched_root,
    }
    legacy_rows = _legacy_source_rows()
    bundle = {
        "audited_parent_repository_commit_id": git_sha1_commit_id(
            AUDITED_PARENT_COMMIT_SHA1
        ),
        "audited_path_tree_root": touched_root,
        "audited_history_tree_root": candidate_record_tree_root(
            "AuditedHistoryRowV1", [history_row]
        ),
        "legacy_source_tree_root": candidate_record_tree_root(
            "LegacyParentSourceRowV1", legacy_rows
        ),
        "audited_path_count": 1,
        "audited_history_row_count": 1,
        "legacy_source_count": 2,
    }
    audit_root = candidate_content_root("ParentAbsenceAuditBundleV1", bundle)
    static = {
        "parent_dsl_version_digest": id_digest_v1(PARENT_DSL_VERSION),
        "parent_freeze_version_digest": id_digest_v1(PARENT_FREEZE_VERSION),
        "parent_repository_commit_id": git_sha1_commit_id(AUDITED_PARENT_COMMIT_SHA1),
        "audit_bundle_root": audit_root,
        "absence_reason_bitmask": 0b1111,
    }
    content_audit = {
        "content_predicate_profile_id": CONTENT_PREDICATE_PROFILE_ID,
        "inspected_path_blob_row_count": 1,
        "inspected_unique_blob_count": 1,
        "inspected_total_byte_length": 7,
        "inspected_blob_inventory_sha256": _blob_inventory_digest(
            {path_row["git_blob_digest"]: 7}
        ),
        "git_blob_object_id_and_size_verified": True,
        "structured_candidate_unique_blob_count": 1,
        "unscannable_relevant_structured_blob_count": 0,
        "content_absence_predicates": [
            {
                "predicate_id": predicate_id,
                "exact_signatures_ascii": [
                    signature.decode("ascii") for signature in signatures
                ],
                "match_occurrence_count": 0,
                "matching_unique_blob_count": 0,
                "matching_path_blob_row_count": 0,
                "matching_blob_digest_set_sha256": _digest_set_sha256(set()),
                "absent": True,
            }
            for predicate_id, signatures in _CONTENT_ABSENCE_PREDICATES.items()
        ],
        "legacy_source_presence": [
            {
                "legacy_parent_payload_source_id": source_id,
                "match_occurrence_count": 1,
                "matching_unique_blob_count": 1,
                "matching_path_blob_row_count": 1,
                "matching_blob_digest_set_sha256": _digest_set_sha256(
                    {path_row["git_blob_digest"]}
                ),
                "present": True,
            }
            for source_id in LEGACY_PARENT_SOURCE_IDS
        ],
        "all_content_absence_predicates_absent": True,
        "all_legacy_sources_present": True,
    }
    return ParentAbsenceAuditEvidence(
        top_level_path_rows=(path_row,),
        history_rows=(history_row,),
        touched_path_rows_by_history_row=((path_row,),),
        legacy_source_rows=legacy_rows,
        audit_bundle_fields=bundle,
        audit_bundle_root=audit_root,
        attestation_static_fields=static,
        path_name_receipt=_path_name_receipt(
            [path_row],
            audit_bundle_root=audit_root,
            audited_path_tree_root=touched_root,
            content_blob_audit=content_audit,
        ),
    )


def _operation_probe(request: dict[str, object]) -> dict[str, object]:
    environment = {
        "HEGEL_ACTOR_IMAGE_REF": IMAGE,
        "HEGEL_ACTOR_PROFILE_ID": (
            "hegel-owner-accepted-container-technical-actors-v1"
        ),
        "HEGEL_BASIS_COMMIT": BASIS,
        "HEGEL_DAEMON_RECEIPT_SHA256": "61" * 32,
        "HEGEL_HOST_REPOSITORY_PATH_SHA256": "60" * 32,
        "HEGEL_PROFILE_SHA256": "62" * 32,
        "HEGEL_PURPOSE_ID": "4",
        "HEGEL_RUN_ID": "63" * 16,
        "LANG": "C",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "HEGEL_OPERATION_ID": OPERATION_ID,
        "HEGEL_OPERATION_NONCE": "64" * 16,
        "HEGEL_OPERATION_REQUEST_SHA256": request["request_sha256"],
        "HEGEL_OPERATION_SEQUENCE": "4",
        "HEGEL_PROBE_INPUT_WRITE_PATH": "/input/.hegel-write-probe",
    }
    required_checks = {
        name: True
        for name in (
            "same_worker_process",
            "identity_nonroot",
            "capabilities_zero",
            "no_new_privileges",
            "seccomp_mode",
            "network_loopback_only",
            "blocked_syscalls_eperm",
            "root_input_read_only",
            "output_state_writable",
            "custody_scope_exact",
            "forbidden_paths_absent",
            "cross_purpose_paths_absent",
            "mount_destinations_exact",
            "operation_environment_exact",
            "pid1_environment_exact",
            "worker_fds_exact",
            "pid1_fds_exact",
            "memory_limit_exact",
            "memory_swap_zero",
            "pids_limit_exact",
        )
    }
    body: dict[str, object] = {
        "schema": OPERATION_PROBE_SCHEMA,
        "implementation": "python-ctypes-in-process-v1",
        "operation_id": OPERATION_ID,
        "operation_sequence": 4,
        "operation_nonce_hex": "64" * 16,
        "operation_request_sha256": request["request_sha256"],
        "purpose_id": 4,
        "identity": {"uid": 65534, "gid": 65534, "pid": 2, "ppid": 1},
        "proc_status": {
            "CapInh": "0000000000000000",
            "CapPrm": "0000000000000000",
            "CapEff": "0000000000000000",
            "CapBnd": "0000000000000000",
            "CapAmb": "0000000000000000",
            "NoNewPrivs": 1,
            "Seccomp": 2,
        },
        "namespaces": {
            "pid": "pid:[1]",
            "mnt": "mnt:[2]",
            "net": "net:[3]",
            "ipc": "ipc:[4]",
            "uts": "uts:[5]",
        },
        "network_interfaces": ["lo"],
        "syscall_probes": [
            {"probe_id": probe_id, "return_value": -1, "errno": 1}
            for probe_id in (
                "socket(AF_INET, SOCK_STREAM)",
                "socket(AF_INET6, SOCK_STREAM)",
                "mount",
                "ptrace(PTRACE_TRACEME)",
                "bpf(BPF_MAP_CREATE)",
                "perf_event_open",
            )
        ],
        "filesystem_probes": {
            "root_write": {"denied": True, "errno": 30},
            "input_write": {"denied": True, "errno": 30},
            "output_write": {"succeeded": True, "errno": 0},
            "state_write": {"succeeded": True, "errno": 0},
            "custody_present": False,
            "forbidden_paths_present": [],
            "cross_purpose_paths_present": [],
            "mount_destinations": ["/input", "/output", "/state", "/tmp"],
            "custody_write_or_null": None,
        },
        "operation_environment": environment,
        "pid1_environment": {
            key: environment[key]
            for key in (
                "HEGEL_ACTOR_IMAGE_REF",
                "HEGEL_ACTOR_PROFILE_ID",
                "HEGEL_BASIS_COMMIT",
                "HEGEL_DAEMON_RECEIPT_SHA256",
                "HEGEL_HOST_REPOSITORY_PATH_SHA256",
                "HEGEL_PROFILE_SHA256",
                "HEGEL_PURPOSE_ID",
                "HEGEL_RUN_ID",
                "LANG",
                "LC_ALL",
                "PATH",
                "PYTHONDONTWRITEBYTECODE",
                "PYTHONHASHSEED",
            )
        },
        "worker_open_fds": [0, 1, 2],
        "pid1_open_fds": [0, 1, 2],
        "cgroup_limits": {
            "memory_max": str(512 * 1024 * 1024),
            "memory_swap_max": "0",
            "pids_max": "64",
        },
        "required_checks": required_checks,
        "all_required_checks_passed": True,
    }
    body["receipt_sha256"] = hashlib.sha256(canonical_json_v1(body)).hexdigest()
    return body


def _fixture():
    cryptography = pytest.importorskip(
        "cryptography.hazmat.primitives.asymmetric.ed25519"
    )
    serialization = pytest.importorskip("cryptography.hazmat.primitives.serialization")
    private = cryptography.Ed25519PrivateKey.from_private_bytes(
        hashlib.sha256(b"non-authority-purpose4-keybearing-test-key").digest()
    )
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    key_id = hashlib.sha256(public).digest()[:16]
    request = _request(key_id)
    evidence = _synthetic_evidence()
    audit_cbor = encode_formal_object(
        "ParentAbsenceAuditBundleV1", evidence.audit_bundle_fields
    )
    fields = build_parent_absence_attestation_fields_v2(
        evidence,
        auditor_key_id=key_id,
        audited_at_unix_seconds=request["audited_at_unix_seconds"],
    )
    attestation_cbor = encode_formal_object(
        "ParentManifestAbsenceAttestationV2", fields
    )
    attestation_root = candidate_content_root(
        "ParentManifestAbsenceAttestationV2", fields
    )
    preimage = external_signature_preimage_v1(
        OBJECT_TAGS["ParentManifestAbsenceAttestationV2"], attestation_root, 4, 0
    )
    response: dict[str, object] = {
        "schema": RESPONSE_SCHEMA,
        "purpose_id": 4,
        "basis_commit_sha1": BASIS,
        "actor_image_ref": IMAGE,
        "request_sha256": request["request_sha256"],
        "snapshot_manifest_sha256": request["snapshot_manifest"]["manifest_sha256"],
        "runtime_inventory_sha256": request["runtime_bindings"]["runtime_inventory"][
            "inventory_sha256"
        ],
        "runtime_source_binding_sha256": request["runtime_bindings"][
            "runtime_source_bindings"
        ]["binding_sha256"],
        "operation_probe_receipt": _operation_probe(request),
        "parent_absence_public_receipt": parent_absence_public_receipt_v1(evidence),
        "audit_bundle_cbor_hex": audit_cbor.hex(),
        "audit_bundle_root_hex": evidence.audit_bundle_root.hex(),
        "attestation_cbor_hex": attestation_cbor.hex(),
        "attestation_root_hex": attestation_root.hex(),
        "signer_public_key_32_hex": public.hex(),
        "signer_key_id_hex": key_id.hex(),
        "signer_key_epoch": 0,
        "signature_hex": private.sign(preimage).hex(),
        "signature_verified_inside_actor": True,
        "audit_rows_received_from_host": False,
        "attestation_received_from_host": False,
        "signing_preimage_received_from_host": False,
        "private_key_exported": False,
        "raw_split_seed_accessed": False,
        "network_access_performed": False,
    }
    response["response_sha256"] = hashlib.sha256(
        canonical_json_v1(response)
    ).hexdigest()

    def verify(candidate_public: bytes, signature: bytes, message: bytes) -> None:
        cryptography.Ed25519PublicKey.from_public_bytes(candidate_public).verify(
            signature, message
        )

    return request, response, verify


def _redigest(response: dict[str, object]) -> None:
    response.pop("response_sha256", None)
    response["response_sha256"] = hashlib.sha256(
        canonical_json_v1(response)
    ).hexdigest()


def test_request_is_identity_only_and_forbids_host_oracle_material() -> None:
    request = _request(bytes.fromhex("78" * 16))
    validated = validate_purpose4_keybearing_request_v1(request)
    assert validated["schema"] == REQUEST_SCHEMA
    assert validated["host_supplied_audit_rows"] is False
    assert validated["host_supplied_attestation"] is False
    assert validated["host_supplied_signing_preimage"] is False
    assert "audit_rows" not in request
    assert "attestation_cbor" not in request
    assert "signing_preimage" not in request

    for field, value in (
        ("audit_rows", []),
        ("attestation_cbor_hex", "00"),
        ("signing_preimage_hex", "00"),
        ("private_key_hex", "00"),
    ):
        attacked = copy.deepcopy(request)
        attacked[field] = value
        _redigest(attacked)
        with pytest.raises(Purpose4KeyBearingError) as caught:
            validate_purpose4_keybearing_request_v1(attacked)
        assert caught.value.code == FAIL_HOST_ORACLE

    attacked = copy.deepcopy(request)
    attacked["host_supplied_signing_preimage"] = True
    _redigest(attacked)
    with pytest.raises(Purpose4KeyBearingError) as caught:
        validate_purpose4_keybearing_request_v1(attacked)
    assert caught.value.code == FAIL_HOST_ORACLE


def test_signed_actor_response_replays_but_remains_non_authoritative() -> None:
    request, response, verifier = _fixture()
    result = validate_purpose4_keybearing_response_v1(
        response, request=request, signature_verifier=verifier
    )
    assert result.attestation_root.hex() == response["attestation_root_hex"]
    assert result.signature.hex() == response["signature_hex"]
    assert result.authoritative is False


@pytest.mark.parametrize(
    "attack", ["signature", "key", "audit_root", "receipt", "probe"]
)
def test_response_rejects_signature_key_and_replay_splices(attack: str) -> None:
    request, response, verifier = _fixture()
    attacked = copy.deepcopy(response)
    if attack == "signature":
        attacked["signature_hex"] = "00" * 64
        expected = FAIL_SIGNATURE
    elif attack == "key":
        attacked["signer_public_key_32_hex"] = "99" * 32
        expected = FAIL_KEY
    elif attack == "audit_root":
        attacked["audit_bundle_root_hex"] = "88" * 32
        expected = FAIL_REPLAY
    elif attack == "receipt":
        attacked["parent_absence_public_receipt"]["all_predicates_absent"] = False
        receipt = attacked["parent_absence_public_receipt"]
        path_keys = {
            "schema_id",
            "audited_parent_commit_sha1",
            "parent_dsl_version",
            "parent_freeze_version",
            "touched_path_rule_id",
            "path_alias_rule_id",
            "path_name_predicate_profile_id",
            "audited_path_tree_root",
            "audit_bundle_root",
            "predicates",
            "all_predicates_absent",
            "content_blob_audit",
            "authority_claim",
            "purpose_4_signature_present",
        }
        diagnostic = {key: receipt[key] for key in path_keys}
        receipt["diagnostic_receipt_sha256"] = hashlib.sha256(
            json.dumps(
                diagnostic, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("ascii")
        ).hexdigest()
        expected = FAIL_REPLAY
    else:
        probe = attacked["operation_probe_receipt"]
        probe["filesystem_probes"]["input_write"] = {
            "denied": False,
            "errno": 0,
        }
        probe.pop("receipt_sha256")
        probe["receipt_sha256"] = hashlib.sha256(
            canonical_json_v1(probe)
        ).hexdigest()
        expected = FAIL_PROBE
    _redigest(attacked)
    with pytest.raises(Purpose4KeyBearingError) as caught:
        validate_purpose4_keybearing_response_v1(
            attacked, request=request, signature_verifier=verifier
        )
    assert caught.value.code == expected


def test_worker_signs_only_its_in_memory_preimage_with_existing_test_key() -> None:
    cryptography = pytest.importorskip(
        "cryptography.hazmat.primitives.asymmetric.ed25519"
    )
    serialization = pytest.importorskip("cryptography.hazmat.primitives.serialization")
    worker_path = (
        Path(__file__).resolve().parents[1]
        / "tools/phase3_m25_purpose4_keybearing_detached_worker_v1.py"
    )
    spec = importlib.util.spec_from_file_location("purpose4_keybearing_worker", worker_path)
    assert spec is not None and spec.loader is not None
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)

    with tempfile.TemporaryDirectory(
        prefix="hegel-purpose4-test-", dir="/tmp"
    ) as raw_root:
        root = Path(raw_root)
        state = root / "state"
        temporary = root / "temporary"
        state.mkdir(mode=0o700)
        temporary.mkdir(mode=0o700)
        os.chmod(state, 0o700)
        os.chmod(temporary, 0o700)
        private = cryptography.Ed25519PrivateKey.from_private_bytes(
            hashlib.sha256(b"non-authority-local-worker-test-key").digest()
        )
        key_path = state / "ed25519-private.pem"
        key_path.write_bytes(
            private.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        os.chmod(key_path, 0o600)
        public = private.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        key_id = hashlib.sha256(public).digest()[:16]
        preimage = b"non-authority internally-derived test preimage"
        actual_public, signature = worker._sign_and_verify_local_v1(
            preimage,
            key_id,
            state_root=state,
            temporary_root=temporary,
        )
        assert actual_public == public
        cryptography.Ed25519PublicKey.from_public_bytes(public).verify(signature, preimage)
        assert list(temporary.iterdir()) == []
        assert key_path.read_bytes().startswith(b"-----BEGIN PRIVATE KEY-----")


def test_worker_source_has_no_legacy_host_signature_oracle_path() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "tools/phase3_m25_purpose4_keybearing_detached_worker_v1.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "/input/signing-preimage.bin",
        "parent-audit-replay.json",
        "top_level_path_cbor",
        "history_cbor",
        "touched_path_cbor_by_history_row",
        "legacy_source_cbor",
        "genpkey",
    ):
        assert forbidden not in source
