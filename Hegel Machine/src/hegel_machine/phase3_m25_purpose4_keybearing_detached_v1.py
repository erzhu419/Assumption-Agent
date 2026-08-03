"""Host protocol for the key-bearing detached purpose-4 actor.

The request carries only Commit-A, runtime, snapshot, timestamp and expected
local-key bindings.  It cannot carry formal audit rows, an attestation, an
attestation root, or a signing preimage.  The long-lived purpose-4 worker must
derive all of those from its read-only detached Git snapshot and sign with the
private key already held in its purpose-private ``/state`` volume.

This module validates public requests and responses.  It never reads a private
key, signs, starts a container, advances a gate, or removes the existing
authoritative blocker.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from types import MappingProxyType
from typing import Callable, Final, Mapping, NoReturn

from .phase3_m25_parent_absence_audit_v1 import (
    AUDITED_PARENT_COMMIT_SHA1,
    LEGACY_PARENT_SOURCE_IDS,
    PARENT_DSL_VERSION,
    PARENT_FREEZE_VERSION,
    PUBLIC_RECEIPT_SCHEMA_ID,
)
from .phase3_m25_purpose4_detached_audit_v1 import (
    SNAPSHOT_SCHEMA,
    Purpose4DetachedAuditError,
    _validate_complete_receipt,
)
from .phase3_m25_wire_v1 import (
    M25WireError,
    OBJECT_TAGS,
    candidate_content_root,
    decode_formal_object,
    encode_formal_object,
    external_signature_preimage_v1,
)


REQUEST_SCHEMA: Final = "hegel-gate17-purpose4-keybearing-detached-request/1"
RESPONSE_SCHEMA: Final = "hegel-gate17-purpose4-keybearing-detached-response/1"
OPERATION_PROBE_SCHEMA: Final = "hegel-phase3-m25-operation-bound-live-probe/1"
OPERATION_ID: Final = "purpose4-parent-sign"
PURPOSE_ID: Final = 4
SIGNER_EPOCH: Final = 0
MAX_RESPONSE_BYTES: Final = 4 * 1024 * 1024

FAIL_REQUEST: Final = "FAIL_GATE17_PURPOSE4_KEYBEARING_REQUEST"
FAIL_HOST_ORACLE: Final = "FAIL_GATE17_PURPOSE4_HOST_SIGNATURE_ORACLE"
FAIL_RESPONSE: Final = "FAIL_GATE17_PURPOSE4_KEYBEARING_RESPONSE"
FAIL_PROBE: Final = "FAIL_GATE17_PURPOSE4_KEYBEARING_OPERATION_PROBE"
FAIL_REPLAY: Final = "FAIL_GATE17_PURPOSE4_KEYBEARING_REPLAY"
FAIL_KEY: Final = "FAIL_GATE17_PURPOSE4_KEYBEARING_KEY_BINDING"
FAIL_SIGNATURE: Final = "FAIL_GATE17_PURPOSE4_KEYBEARING_SIGNATURE"

_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IMAGE = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}")
_PROBE_BASE_ENV_KEYS: Final = frozenset(
    {
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
    }
)
_PROBE_OPERATION_ENV_KEYS: Final = frozenset(
    {
        "HEGEL_OPERATION_ID",
        "HEGEL_OPERATION_NONCE",
        "HEGEL_OPERATION_REQUEST_SHA256",
        "HEGEL_OPERATION_SEQUENCE",
        "HEGEL_PROBE_INPUT_WRITE_PATH",
    }
)
_PROBE_REQUIRED_CHECK_KEYS: Final = frozenset(
    {
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
    }
)
_PROBE_SYSCALL_IDS: Final = (
    "socket(AF_INET, SOCK_STREAM)",
    "socket(AF_INET6, SOCK_STREAM)",
    "mount",
    "ptrace(PTRACE_TRACEME)",
    "bpf(BPF_MAP_CREATE)",
    "perf_event_open",
)
_FORBIDDEN_REQUEST_KEYS: Final = frozenset(
    {
        "audit_rows",
        "top_level_path_cbor",
        "history_cbor",
        "touched_path_cbor_by_history_row",
        "legacy_source_cbor",
        "audit_bundle_cbor",
        "attestation_cbor",
        "attestation_root",
        "expected_attestation_root",
        "signing_preimage",
        "signature_preimage",
        "private_key",
        "private_key_seed",
    }
)
_HOST_PROVENANCE_MARKERS: Final = frozenset(
    {
        "host_supplied_audit_rows",
        "host_supplied_attestation",
        "host_supplied_signing_preimage",
    }
)


class Purpose4KeyBearingError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Purpose4KeyBearingError(code, detail)


def canonical_json_v1(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def _digest_body(value: Mapping[str, object], digest_field: str) -> str:
    body = dict(value)
    body.pop(digest_field, None)
    return hashlib.sha256(canonical_json_v1(body)).hexdigest()


def _reject_forbidden_request_tree(value: object, *, path: str = "request") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).lower()
            if key in _HOST_PROVENANCE_MARKERS:
                if item is not False:
                    _fail(
                        FAIL_HOST_ORACLE,
                        f"host provenance marker must be false: {path}.{key}",
                    )
                continue
            normalized = key.removesuffix("_hex").removesuffix("_or_null")
            if normalized in _FORBIDDEN_REQUEST_KEYS or any(
                token in normalized
                for token in (
                    "audit_rows",
                    "attestation_cbor",
                    "attestation_root",
                    "signing_preimage",
                    "signature_preimage",
                    "private_key",
                )
            ):
                _fail(FAIL_HOST_ORACLE, f"host request contains forbidden field {path}.{key}")
            _reject_forbidden_request_tree(item, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_forbidden_request_tree(item, path=f"{path}[{index}]")


def _require_snapshot_manifest(value: object, basis_commit: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        _fail(FAIL_REQUEST, "snapshot manifest is absent")
    supplied = dict(value)
    claimed = supplied.pop("manifest_sha256", None)
    if (
        supplied.get("schema") != SNAPSHOT_SCHEMA
        or supplied.get("basis_commit_sha1") != basis_commit
        or type(claimed) is not str
        or _SHA256.fullmatch(claimed) is None
        or hashlib.sha256(canonical_json_v1(supplied)).hexdigest() != claimed
    ):
        _fail(FAIL_REQUEST, "snapshot manifest binding differs")
    return {**supplied, "manifest_sha256": claimed}


def _require_runtime_bindings(value: object, basis_commit: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "runtime_inventory",
        "runtime_source_bindings",
    }:
        _fail(FAIL_REQUEST, "runtime binding field set differs")
    inventory = value["runtime_inventory"]
    sources = value["runtime_source_bindings"]
    if not isinstance(inventory, Mapping) or not isinstance(sources, Mapping):
        _fail(FAIL_REQUEST, "runtime binding body is absent")
    if (
        type(inventory.get("inventory_sha256")) is not str
        or _SHA256.fullmatch(str(inventory["inventory_sha256"])) is None
        or sources.get("schema")
        != "hegel-gate17-purpose4-runtime-source-bindings/1"
        or sources.get("basis_commit_sha1") != basis_commit
        or type(sources.get("binding_sha256")) is not str
        or _SHA256.fullmatch(str(sources["binding_sha256"])) is None
    ):
        _fail(FAIL_REQUEST, "runtime inventory/source binding differs")
    source_body = dict(sources)
    claimed = source_body.pop("binding_sha256")
    domain = b"HEGEL/GATE17/PURPOSE4_RUNTIME_SOURCE_BINDINGS/V1\x00"
    if hashlib.sha256(domain + canonical_json_v1(source_body)).hexdigest() != claimed:
        _fail(FAIL_REQUEST, "runtime source-binding digest differs")
    return {
        "runtime_inventory": dict(inventory),
        "runtime_source_bindings": dict(sources),
    }


def build_purpose4_keybearing_request_v1(
    *,
    basis_commit: str,
    actor_image_ref: str,
    snapshot_manifest: Mapping[str, object],
    runtime_inventory: Mapping[str, object],
    runtime_source_bindings: Mapping[str, object],
    audited_at_unix_seconds: int,
    expected_local_key_id: bytes,
) -> dict[str, object]:
    if type(basis_commit) is not str or _SHA1.fullmatch(basis_commit) is None:
        _fail(FAIL_REQUEST, "basis commit must be lowercase SHA-1")
    if type(actor_image_ref) is not str or _IMAGE.fullmatch(actor_image_ref) is None:
        _fail(FAIL_REQUEST, "actor image is not digest pinned")
    if type(audited_at_unix_seconds) is not int or audited_at_unix_seconds < 0:
        _fail(FAIL_REQUEST, "audit timestamp must be a nonnegative integer")
    if type(expected_local_key_id) is not bytes or len(expected_local_key_id) != 16:
        _fail(FAIL_KEY, "expected local purpose-4 key ID must be 16 bytes")
    snapshot = _require_snapshot_manifest(snapshot_manifest, basis_commit)
    runtime = _require_runtime_bindings(
        {
            "runtime_inventory": runtime_inventory,
            "runtime_source_bindings": runtime_source_bindings,
        },
        basis_commit,
    )
    request: dict[str, object] = {
        "schema": REQUEST_SCHEMA,
        "purpose_id": PURPOSE_ID,
        "basis_commit_sha1": basis_commit,
        "actor_image_ref": actor_image_ref,
        "snapshot_manifest": snapshot,
        "runtime_bindings": runtime,
        "audited_at_unix_seconds": audited_at_unix_seconds,
        "expected_local_key_id_hex": expected_local_key_id.hex(),
        "host_supplied_audit_rows": False,
        "host_supplied_attestation": False,
        "host_supplied_signing_preimage": False,
    }
    _reject_forbidden_request_tree(request)
    request["request_sha256"] = hashlib.sha256(canonical_json_v1(request)).hexdigest()
    return request


def validate_purpose4_keybearing_request_v1(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(FAIL_REQUEST, "request is not an object")
    _reject_forbidden_request_tree(
        {key: item for key, item in value.items() if key != "request_sha256"}
    )
    expected = {
        "schema",
        "purpose_id",
        "basis_commit_sha1",
        "actor_image_ref",
        "snapshot_manifest",
        "runtime_bindings",
        "audited_at_unix_seconds",
        "expected_local_key_id_hex",
        "host_supplied_audit_rows",
        "host_supplied_attestation",
        "host_supplied_signing_preimage",
        "request_sha256",
    }
    if set(value) != expected:
        _fail(FAIL_REQUEST, "request field set differs")
    basis = value["basis_commit_sha1"]
    if type(basis) is not str or _SHA1.fullmatch(basis) is None:
        _fail(FAIL_REQUEST, "basis commit differs")
    if (
        value["schema"] != REQUEST_SCHEMA
        or value["purpose_id"] != PURPOSE_ID
        or type(value["actor_image_ref"]) is not str
        or _IMAGE.fullmatch(str(value["actor_image_ref"])) is None
        or type(value["audited_at_unix_seconds"]) is not int
        or value["audited_at_unix_seconds"] < 0
        or type(value["expected_local_key_id_hex"]) is not str
        or re.fullmatch(r"[0-9a-f]{32}", str(value["expected_local_key_id_hex"])) is None
        or any(
            value[field] is not False
            for field in (
                "host_supplied_audit_rows",
                "host_supplied_attestation",
                "host_supplied_signing_preimage",
            )
        )
        or type(value["request_sha256"]) is not str
        or _digest_body(value, "request_sha256") != value["request_sha256"]
    ):
        _fail(FAIL_REQUEST, "request policy/digest differs")
    _require_snapshot_manifest(value["snapshot_manifest"], basis)
    _require_runtime_bindings(value["runtime_bindings"], basis)
    return MappingProxyType(dict(value))


def _validate_operation_probe(value: object, request: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(FAIL_PROBE, "operation-bound probe is absent")
    body = dict(value)
    claimed = body.pop("receipt_sha256", None)
    exact_body_keys = {
        "schema",
        "implementation",
        "operation_id",
        "operation_sequence",
        "operation_nonce_hex",
        "operation_request_sha256",
        "purpose_id",
        "identity",
        "proc_status",
        "namespaces",
        "network_interfaces",
        "syscall_probes",
        "filesystem_probes",
        "operation_environment",
        "pid1_environment",
        "worker_open_fds",
        "pid1_open_fds",
        "cgroup_limits",
        "required_checks",
        "all_required_checks_passed",
    }
    if (
        type(claimed) is not str
        or _SHA256.fullmatch(claimed) is None
        or hashlib.sha256(canonical_json_v1(body)).hexdigest() != claimed
        or set(body) != exact_body_keys
        or body.get("schema") != OPERATION_PROBE_SCHEMA
        or body.get("implementation") != "python-ctypes-in-process-v1"
        or body.get("operation_id") != OPERATION_ID
        or body.get("purpose_id") != PURPOSE_ID
        or type(body.get("operation_sequence")) is not int
        or body["operation_sequence"] <= 0
        or type(body.get("operation_nonce_hex")) is not str
        or re.fullmatch(r"[0-9a-f]{32}", str(body["operation_nonce_hex"])) is None
        or body.get("operation_request_sha256") != request["request_sha256"]
        or body.get("all_required_checks_passed") is not True
        or not isinstance(body.get("required_checks"), Mapping)
        or set(body["required_checks"]) != _PROBE_REQUIRED_CHECK_KEYS
        or not all(item is True for item in body["required_checks"].values())
    ):
        _fail(FAIL_PROBE, "operation-bound probe policy/digest differs")
    environment = body.get("operation_environment")
    if (
        not isinstance(environment, Mapping)
        or set(environment) != _PROBE_BASE_ENV_KEYS | _PROBE_OPERATION_ENV_KEYS
        or environment.get("HEGEL_OPERATION_REQUEST_SHA256")
        != request["request_sha256"]
        or environment.get("HEGEL_BASIS_COMMIT") != request["basis_commit_sha1"]
        or environment.get("HEGEL_PURPOSE_ID") != "4"
        or environment.get("HEGEL_ACTOR_IMAGE_REF") != request["actor_image_ref"]
        or environment.get("HEGEL_ACTOR_PROFILE_ID")
        != "hegel-owner-accepted-container-technical-actors-v1"
        or environment.get("HEGEL_OPERATION_ID") != OPERATION_ID
        or environment.get("HEGEL_OPERATION_NONCE")
        != body["operation_nonce_hex"]
        or environment.get("HEGEL_OPERATION_SEQUENCE")
        != str(body["operation_sequence"])
        or environment.get("HEGEL_PROBE_INPUT_WRITE_PATH")
        != "/input/.hegel-write-probe"
        or environment.get("LANG") != "C"
        or environment.get("LC_ALL") != "C.UTF-8"
        or environment.get("PATH") != "/usr/local/bin:/usr/bin:/bin"
        or environment.get("PYTHONDONTWRITEBYTECODE") != "1"
        or environment.get("PYTHONHASHSEED") != "0"
        or re.fullmatch(
            r"[0-9a-f]{64}", str(environment.get("HEGEL_DAEMON_RECEIPT_SHA256"))
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(environment.get("HEGEL_PROFILE_SHA256"))
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(environment.get("HEGEL_HOST_REPOSITORY_PATH_SHA256")),
        )
        is None
        or re.fullmatch(r"[0-9a-f]{32}", str(environment.get("HEGEL_RUN_ID")))
        is None
    ):
        _fail(FAIL_PROBE, "operation-bound probe request/runtime binding differs")
    identity = body["identity"]
    status = body["proc_status"]
    namespaces = body["namespaces"]
    syscalls = body["syscall_probes"]
    filesystem = body["filesystem_probes"]
    pid1_environment = body["pid1_environment"]
    cgroup = body["cgroup_limits"]
    if (
        not isinstance(identity, Mapping)
        or set(identity) != {"uid", "gid", "pid", "ppid"}
        or identity.get("uid") != 65534
        or identity.get("gid") != 65534
        or type(identity.get("pid")) is not int
        or identity["pid"] <= 1
        or type(identity.get("ppid")) is not int
        or identity["ppid"] <= 0
        or not isinstance(status, Mapping)
        or set(status)
        != {"CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb", "NoNewPrivs", "Seccomp"}
        or any(
            status.get(name) != "0000000000000000"
            for name in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
        )
        or status.get("NoNewPrivs") != 1
        or status.get("Seccomp") != 2
        or not isinstance(namespaces, Mapping)
        or set(namespaces) != {"pid", "mnt", "net", "ipc", "uts"}
        or any(
            type(item) is not str
            or re.fullmatch(r"[a-z]+:\[[1-9][0-9]*\]", item) is None
            for item in namespaces.values()
        )
        or body["network_interfaces"] != ["lo"]
        or type(syscalls) is not list
        or len(syscalls) != len(_PROBE_SYSCALL_IDS)
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"probe_id", "return_value", "errno"}
            or row.get("probe_id") != probe_id
            or row.get("return_value") != -1
            or row.get("errno") != 1
            for row, probe_id in zip(syscalls, _PROBE_SYSCALL_IDS, strict=True)
        )
    ):
        _fail(FAIL_PROBE, "operation-bound process/isolation evidence differs")
    if (
        not isinstance(filesystem, Mapping)
        or set(filesystem)
        != {
            "root_write",
            "input_write",
            "output_write",
            "state_write",
            "custody_present",
            "forbidden_paths_present",
            "cross_purpose_paths_present",
            "mount_destinations",
            "custody_write_or_null",
        }
        or any(
            not isinstance(filesystem.get(name), Mapping)
            or filesystem[name].get("denied") is not True
            or filesystem[name].get("errno") not in {1, 13, 30}
            for name in ("root_write", "input_write")
        )
        or filesystem.get("output_write") != {"succeeded": True, "errno": 0}
        or filesystem.get("state_write") != {"succeeded": True, "errno": 0}
        or filesystem.get("custody_present") is not False
        or filesystem.get("custody_write_or_null") is not None
        or filesystem.get("forbidden_paths_present") != []
        or filesystem.get("cross_purpose_paths_present") != []
        or filesystem.get("mount_destinations")
        != ["/input", "/output", "/state", "/tmp"]
        or body["worker_open_fds"] != [0, 1, 2]
        or body["pid1_open_fds"] != [0, 1, 2]
        or not isinstance(pid1_environment, Mapping)
        or dict(pid1_environment)
        != {key: environment[key] for key in _PROBE_BASE_ENV_KEYS}
        or not isinstance(cgroup, Mapping)
        or dict(cgroup)
        != {
            "memory_max": str(512 * 1024 * 1024),
            "memory_swap_max": "0",
            "pids_max": "64",
        }
    ):
        _fail(FAIL_PROBE, "operation-bound filesystem/resource evidence differs")
    return MappingProxyType({**body, "receipt_sha256": claimed})


def _validate_public_receipt(receipt: object, audit_root: bytes) -> None:
    if not isinstance(receipt, Mapping):
        _fail(FAIL_REPLAY, "parent-absence public receipt is absent")
    path_receipt_keys = {
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
        "diagnostic_receipt_sha256",
    }
    public_extension_keys = {
        "audited_history_tree_root",
        "legacy_source_tree_root",
        "audited_path_count",
        "audited_history_row_count",
        "legacy_source_count",
        "root_commit_count",
        "merge_commit_count",
        "legacy_parent_source_ids",
        "attestation_static_fields",
        "replay_requires_git_objects",
    }
    if set(receipt) != path_receipt_keys | public_extension_keys:
        _fail(FAIL_REPLAY, "parent-absence public receipt field set differs")
    try:
        _validate_complete_receipt(receipt, audit_root.hex())
    except Purpose4DetachedAuditError as exc:
        _fail(FAIL_REPLAY, f"strict no-key receipt replay failed: {exc.code}")
    path_body = {key: receipt[key] for key in path_receipt_keys}
    claimed_diagnostic = path_body.pop("diagnostic_receipt_sha256")
    diagnostic_payload = json.dumps(
        path_body, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    if (
        type(claimed_diagnostic) is not str
        or _SHA256.fullmatch(claimed_diagnostic) is None
        or hashlib.sha256(diagnostic_payload).hexdigest() != claimed_diagnostic
        or receipt["schema_id"] != PUBLIC_RECEIPT_SCHEMA_ID
        or receipt["audited_parent_commit_sha1"]
        != AUDITED_PARENT_COMMIT_SHA1.hex()
        or receipt["parent_dsl_version"] != PARENT_DSL_VERSION
        or receipt["parent_freeze_version"] != PARENT_FREEZE_VERSION
        or receipt["audit_bundle_root"] != audit_root.hex()
        or receipt["all_predicates_absent"] is not True
        or receipt["authority_claim"] is not False
        or receipt["purpose_4_signature_present"] is not False
        or receipt["replay_requires_git_objects"] is not True
    ):
        _fail(FAIL_REPLAY, "parent-absence receipt identity or replay policy differs")
    predicates = receipt["predicates"]
    if (
        not isinstance(predicates, list)
        or not predicates
        or any(
            not isinstance(row, Mapping) or row.get("absent") is not True
            for row in predicates
        )
    ):
        _fail(FAIL_REPLAY, "path-name absence predicate receipt differs")
    content = receipt["content_blob_audit"]
    if (
        not isinstance(content, Mapping)
        or content.get("all_content_absence_predicates_absent") is not True
        or content.get("all_legacy_sources_present") is not True
        or content.get("git_blob_object_id_and_size_verified") is not True
        or content.get("unscannable_relevant_structured_blob_count") != 0
    ):
        _fail(FAIL_REPLAY, "content/legacy replay receipt differs")
    if (
        type(receipt["audited_path_count"]) is not int
        or receipt["audited_path_count"] <= 0
        or type(receipt["audited_history_row_count"]) is not int
        or receipt["audited_history_row_count"] <= 0
        or receipt["legacy_source_count"] != len(LEGACY_PARENT_SOURCE_IDS)
        or receipt["legacy_parent_source_ids"] != list(LEGACY_PARENT_SOURCE_IDS)
        or type(receipt["root_commit_count"]) is not int
        or receipt["root_commit_count"] <= 0
        or type(receipt["merge_commit_count"]) is not int
        or receipt["merge_commit_count"] < 0
    ):
        _fail(FAIL_REPLAY, "parent-absence replay counts differ")
    static = receipt["attestation_static_fields"]
    if (
        not isinstance(static, Mapping)
        or static.get("audit_bundle_root") != audit_root.hex()
        or static.get("parent_repository_commit_sha1")
        != AUDITED_PARENT_COMMIT_SHA1.hex()
        or static.get("absence_reason_bitmask") != 0b1111
    ):
        _fail(FAIL_REPLAY, "public receipt attestation binding differs")


Ed25519VerifierV1 = Callable[[bytes, bytes, bytes], None]


@dataclass(frozen=True, slots=True)
class Purpose4KeyBearingResultV1:
    attestation_root: bytes
    audit_bundle_root: bytes
    signer_public_key: bytes
    signer_key_id: bytes
    signature: bytes
    response_sha256: str
    authoritative: bool = False


def validate_purpose4_keybearing_response_v1(
    response: object,
    *,
    request: Mapping[str, object],
    signature_verifier: Ed25519VerifierV1,
) -> Purpose4KeyBearingResultV1:
    validated_request = validate_purpose4_keybearing_request_v1(request)
    if not isinstance(response, Mapping):
        _fail(FAIL_RESPONSE, "response is not an object")
    expected = {
        "schema",
        "purpose_id",
        "basis_commit_sha1",
        "actor_image_ref",
        "request_sha256",
        "snapshot_manifest_sha256",
        "runtime_inventory_sha256",
        "runtime_source_binding_sha256",
        "operation_probe_receipt",
        "parent_absence_public_receipt",
        "audit_bundle_cbor_hex",
        "audit_bundle_root_hex",
        "attestation_cbor_hex",
        "attestation_root_hex",
        "signer_public_key_32_hex",
        "signer_key_id_hex",
        "signer_key_epoch",
        "signature_hex",
        "signature_verified_inside_actor",
        "audit_rows_received_from_host",
        "attestation_received_from_host",
        "signing_preimage_received_from_host",
        "private_key_exported",
        "raw_split_seed_accessed",
        "network_access_performed",
        "response_sha256",
    }
    if set(response) != expected:
        _fail(FAIL_RESPONSE, "response field set differs")
    claimed = response["response_sha256"]
    if (
        type(claimed) is not str
        or _SHA256.fullmatch(claimed) is None
        or _digest_body(response, "response_sha256") != claimed
        or response["schema"] != RESPONSE_SCHEMA
        or response["purpose_id"] != PURPOSE_ID
        or response["basis_commit_sha1"] != validated_request["basis_commit_sha1"]
        or response["actor_image_ref"] != validated_request["actor_image_ref"]
        or response["request_sha256"] != validated_request["request_sha256"]
        or response["snapshot_manifest_sha256"]
        != validated_request["snapshot_manifest"]["manifest_sha256"]
        or response["runtime_inventory_sha256"]
        != validated_request["runtime_bindings"]["runtime_inventory"]["inventory_sha256"]
        or response["runtime_source_binding_sha256"]
        != validated_request["runtime_bindings"]["runtime_source_bindings"]["binding_sha256"]
        or response["signer_key_epoch"] != SIGNER_EPOCH
        or any(
            response[field] is not expected_value
            for field, expected_value in (
                ("signature_verified_inside_actor", True),
                ("audit_rows_received_from_host", False),
                ("attestation_received_from_host", False),
                ("signing_preimage_received_from_host", False),
                ("private_key_exported", False),
                ("raw_split_seed_accessed", False),
                ("network_access_performed", False),
            )
        )
    ):
        _fail(FAIL_RESPONSE, "response policy/digest binding differs")
    _validate_operation_probe(response["operation_probe_receipt"], validated_request)
    try:
        audit_cbor = bytes.fromhex(str(response["audit_bundle_cbor_hex"]))
        audit_root = bytes.fromhex(str(response["audit_bundle_root_hex"]))
        attestation_cbor = bytes.fromhex(str(response["attestation_cbor_hex"]))
        attestation_root = bytes.fromhex(str(response["attestation_root_hex"]))
        public_key = bytes.fromhex(str(response["signer_public_key_32_hex"]))
        signer_key_id = bytes.fromhex(str(response["signer_key_id_hex"]))
        signature = bytes.fromhex(str(response["signature_hex"]))
    except ValueError:
        _fail(FAIL_RESPONSE, "response formal/signature hex is malformed")
    if any(len(value) != length for value, length in (
        (audit_root, 32), (attestation_root, 32), (public_key, 32),
        (signer_key_id, 16), (signature, 64),
    )):
        _fail(FAIL_RESPONSE, "response formal/signature byte length differs")
    try:
        audit = decode_formal_object(
            audit_cbor, expected_name="ParentAbsenceAuditBundleV1"
        )
        if (
            encode_formal_object("ParentAbsenceAuditBundleV1", audit.fields)
            != audit_cbor
            or candidate_content_root("ParentAbsenceAuditBundleV1", audit.fields)
            != audit_root
        ):
            _fail(FAIL_REPLAY, "audit bundle is not canonical/root-bound")
        attestation = decode_formal_object(
            attestation_cbor, expected_name="ParentManifestAbsenceAttestationV2"
        )
        if (
            encode_formal_object(
                "ParentManifestAbsenceAttestationV2", attestation.fields
            )
            != attestation_cbor
            or candidate_content_root(
                "ParentManifestAbsenceAttestationV2", attestation.fields
            )
            != attestation_root
            or attestation.fields["audit_bundle_root"] != audit_root
            or attestation.fields["auditor_key_id"] != signer_key_id
            or attestation.fields["audited_at_unix_seconds"]
            != validated_request["audited_at_unix_seconds"]
        ):
            _fail(FAIL_REPLAY, "attestation is not canonical or exact audit-bound")
    except M25WireError as exc:
        _fail(FAIL_REPLAY, f"formal CBOR replay failed: {exc.code}")
    expected_key_id = bytes.fromhex(str(validated_request["expected_local_key_id_hex"]))
    if signer_key_id != expected_key_id or hashlib.sha256(public_key).digest()[:16] != signer_key_id:
        _fail(FAIL_KEY, "response public key does not match the expected local key ID")
    _validate_public_receipt(response["parent_absence_public_receipt"], audit_root)
    preimage = external_signature_preimage_v1(
        OBJECT_TAGS["ParentManifestAbsenceAttestationV2"],
        attestation_root,
        PURPOSE_ID,
        SIGNER_EPOCH,
    )
    try:
        signature_verifier(public_key, signature, preimage)
    except Exception as exc:
        _fail(FAIL_SIGNATURE, f"purpose-4 Ed25519 verification failed: {exc}")
    return Purpose4KeyBearingResultV1(
        attestation_root,
        audit_root,
        public_key,
        signer_key_id,
        signature,
        claimed,
    )


__all__ = [
    "Ed25519VerifierV1",
    "FAIL_HOST_ORACLE",
    "FAIL_KEY",
    "FAIL_PROBE",
    "FAIL_REPLAY",
    "FAIL_REQUEST",
    "FAIL_RESPONSE",
    "FAIL_SIGNATURE",
    "MAX_RESPONSE_BYTES",
    "OPERATION_ID",
    "Purpose4KeyBearingError",
    "Purpose4KeyBearingResultV1",
    "REQUEST_SCHEMA",
    "RESPONSE_SCHEMA",
    "build_purpose4_keybearing_request_v1",
    "canonical_json_v1",
    "validate_purpose4_keybearing_request_v1",
    "validate_purpose4_keybearing_response_v1",
]
