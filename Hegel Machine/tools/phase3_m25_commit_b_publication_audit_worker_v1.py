#!/usr/bin/env python3
"""Standard-library-only purpose-4 worker for Commit-B staged-byte audit."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Mapping


FAIL = "FAIL_COMMIT_B_PUBLICATION_AUDIT_WORKER"
MANIFEST_SCHEMA = "hegel-phase3-m25-commit-b-index-manifest/1"
REQUEST_SCHEMA = "hegel-phase3-m25-commit-b-publication-audit-request/1"
RECEIPT_SCHEMA = "hegel-phase3-m25-commit-b-publication-audit-receipt/1"
POLICY_ID = "hegel-phase3-m25-commit-b-publication-policy-v1"
INVENTORY_DOMAIN = b"HEGEL/PHASE3/M25/COMMIT_B/INDEX_INVENTORY/V1\x00"
RUNTIME_DOMAIN = b"HEGEL/PHASE3/M25/COMMIT_B/AUDIT_RUNTIME/V1\x00"
AUDIT_RECEIPT_PATH = (
    "Hegel Machine/artifacts/phase3_m25_external/"
    "phase3_m25_commit_b_publication_audit_receipt_v1.json"
)
ALLOWED_PUBLIC_PREFIXES = [
    "Hegel Machine/artifacts/phase3_m25_external",
    "Hegel Machine/docs/phase3_m25_external_status.md",
]
EXECUTABLE_PREFIXES = [
    "Hegel Machine/src", "Hegel Machine/rust", "Hegel Machine/tests",
    "Hegel Machine/tools",
]
FORMAL_EVIDENCE_PATH = (
    "Hegel Machine/artifacts/phase3_m25_external/phase3_m25_formal_gate_evidence_v1.json"
)
FORMAL_PROMOTION_PATH = (
    "Hegel Machine/artifacts/phase3_m25_external/phase3_m25_gate_promotion_v1.json"
)
FORMAL_TRANSACTION_RECEIPT_PATH = FORMAL_PROMOTION_PATH + ".publication-receipt.json"
CANONICAL_JSON_REQUIRED_PATHS = {
    FORMAL_EVIDENCE_PATH, FORMAL_PROMOTION_PATH,
    FORMAL_TRANSACTION_RECEIPT_PATH, AUDIT_RECEIPT_PATH,
}
PUBLICATION_ROLE_REGISTRY = {
    "Hegel Machine/artifacts/phase3_m25_external/phase3_m25_actor_qualification_v1.json": "ACTOR_ELIGIBILITY",
    "Hegel Machine/artifacts/phase3_m25_external/phase3_m25_errata_qualification_v1.json": "ERRATA_QUALIFICATION",
    "Hegel Machine/artifacts/phase3_m25_external/phase3_m3_implementation_qualification_v1.json": "M3_IMPLEMENTATION_QUALIFICATION",
    "Hegel Machine/artifacts/phase3_m25_external/phase3_m25_bridge_dag_rust_binary_qualification_v1.json": "BRIDGE_BINARY_QUALIFICATION",
    "Hegel Machine/artifacts/phase3_m25_external/phase3_m25_live_actor_protocol_qualification_v1.json": "LIVE_ACTOR_PROTOCOL_QUALIFICATION",
    "Hegel Machine/artifacts/phase3_m25_external/phase3_m25_pre_genesis_execution_status_v1.json": "PRE_GENESIS_EXECUTION_STATUS",
    "Hegel Machine/artifacts/phase3_m25_external/phase3_m25_pre_genesis_readiness_v1.json": "PRE_GENESIS_READINESS",
    FORMAL_EVIDENCE_PATH: "FORMAL_GATE_EVIDENCE",
    FORMAL_PROMOTION_PATH: "FORMAL_GATE_PROMOTION",
    FORMAL_TRANSACTION_RECEIPT_PATH: "FORMAL_TRANSACTION_PUBLICATION_RECEIPT",
    "Hegel Machine/docs/phase3_m25_external_status.md": "EXTERNAL_STATUS_DOCUMENT",
    AUDIT_RECEIPT_PATH: "COMMIT_B_PREPARE_AUDIT_RECEIPT",
}
AUTHORITY_DISCLOSURE = {
    "same_admin_controller": True,
    "organizational_independence": False,
    "independent_human_actors": False,
    "technical_role_independence": True,
    "owner_accepted_threat_model": True,
    "remote_attestation": False,
    "hardware_key_nonexportability": False,
}
FORBIDDEN_JSON_KEYS = frozenset(
    {
        "assignment_rows", "auditor_private_key", "custodian_private_key",
        "derived_role_key", "ed25519_private_key", "ed25519_private_key_seed",
        "k_role", "k_split_master", "master_seed_hex", "pre_final_match_set",
        "pre_final_output_archive", "private_key", "private_key_base64",
        "private_key_bytes", "private_key_der", "private_key_pem",
        "private_key_seed", "python_attester_private_key", "raw_k_split_master",
        "raw_private_key", "raw_seed", "raw_split_seed",
        "rust_attester_private_key", "sealed_membership",
        "sealed_prediction_membership", "split_master_seed", "split_seed",
        "split_seed_base64", "split_seed_bytes", "split_seed_hex",
        "validation_membership",
    }
)
_PEM_BEGIN = b"-----BEGIN "
_PEM_END = b" PRIVATE KEY-----"
PRIVATE_HEADERS = (
    b"AGE-SECRET-" + b"KEY-1",
    _PEM_BEGIN + b"DSA" + _PEM_END,
    _PEM_BEGIN + b"EC" + _PEM_END,
    _PEM_BEGIN + b"ENCRYPTED" + _PEM_END,
    _PEM_BEGIN + b"PGP" + b" PRIVATE KEY BLOCK-----",
    b"openssh-key-v1" + bytes((0,)),
    _PEM_BEGIN + b"OPENSSH" + _PEM_END,
    _PEM_BEGIN + b"PRIVATE KEY-----",
    b"PuTTY-User-" + b"Key-File-",
    _PEM_BEGIN + b"RSA" + _PEM_END,
)
PRIVATE_FOOTERS = {
    _PEM_BEGIN + b"DSA" + _PEM_END: b"-----END " + b"DSA PRIVATE KEY-----",
    _PEM_BEGIN + b"EC" + _PEM_END: b"-----END " + b"EC PRIVATE KEY-----",
    _PEM_BEGIN + b"ENCRYPTED" + _PEM_END: (
        b"-----END " + b"ENCRYPTED PRIVATE KEY-----"
    ),
    _PEM_BEGIN + b"PGP" + b" PRIVATE KEY BLOCK-----": (
        b"-----END " + b"PGP PRIVATE KEY BLOCK-----"
    ),
    _PEM_BEGIN + b"OPENSSH" + _PEM_END: (
        b"-----END " + b"OPENSSH PRIVATE KEY-----"
    ),
    _PEM_BEGIN + b"PRIVATE KEY-----": b"-----END " + b"PRIVATE KEY-----",
    _PEM_BEGIN + b"RSA" + _PEM_END: b"-----END " + b"RSA PRIVATE KEY-----",
}
PRIVATE_ANY_OFFSET_HEADERS = {
    b"AGE-SECRET-" + b"KEY-1",
    b"openssh-key-v1" + bytes((0,)),
    b"PuTTY-User-" + b"Key-File-",
}
RAW_PATH_PATTERN = re.compile(
    rb"(?i)(?:/home/[a-z0-9._-]+/|/users/[a-z0-9._ -]+/|"
    rb"/mnt/[a-z]/users/[a-z0-9._ -]+/|[a-z]:\\\\users\\\\|"
    rb"\\\\\\\\wsl(?:\.localhost)?\\\\)"
)


class Pairs(tuple):
    pass


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def strict_json(payload: bytes) -> object:
    def hook(pairs: list[tuple[str, object]]) -> Pairs:
        keys = [key for key, _value in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError
        return Pairs(pairs)

    return json.loads(
        payload.decode("utf-8", "strict"),
        object_pairs_hook=hook,
        parse_constant=lambda _token: (_ for _ in ()).throw(ValueError()),
    )


def plain(value: object) -> object:
    if isinstance(value, Pairs):
        return {key: plain(child) for key, child in value}
    if isinstance(value, list):
        return [plain(child) for child in value]
    return value


def normalized_key(key: str) -> str:
    broken = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    return re.sub(r"[^a-z0-9]+", "_", broken.casefold()).strip("_")


def has_forbidden_key(value: object) -> bool:
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, Pairs):
            for key, child in current:
                if normalized_key(key) in FORBIDDEN_JSON_KEYS:
                    return True
                stack.append(child)
        elif isinstance(current, list):
            stack.extend(current)
    return False


def has_private_key_magic(payload: bytes) -> bool:
    for header in PRIVATE_HEADERS:
        offset = payload.find(header)
        while offset >= 0:
            start = max(payload.rfind(b"\n", 0, offset), payload.rfind(b"\r", 0, offset)) + 1
            if all(byte in {0x09, 0x20} for byte in payload[start:offset]):
                return True
            if header in PRIVATE_ANY_OFFSET_HEADERS:
                return True
            footer = PRIVATE_FOOTERS.get(header)
            if footer is not None and payload.find(footer, offset + len(header)) >= 0:
                return True
            offset = payload.find(header, offset + 1)
    return False


def validate_role_headers(payloads: Mapping[str, bytes], basis_commit: str) -> None:
    objects = {
        PUBLICATION_ROLE_REGISTRY[path]: plain(strict_json(raw))
        for path, raw in payloads.items()
        if path.endswith(".json") and path != AUDIT_RECEIPT_PATH
    }
    if any(type(value) is not dict for value in objects.values()):
        raise ValueError
    expected = {
        "ACTOR_ELIGIBILITY": ("schema", "hegel-phase3-container-actor-qualification/1", "basis_commit"),
        "ERRATA_QUALIFICATION": ("schema_version", "hegel-phase3-m25-exact-wire-errata-qualification/2", "implementation_basis_commit"),
        "M3_IMPLEMENTATION_QUALIFICATION": ("schema_version", "hegel-m3-implementation-qualification/1", "basis_commit"),
        "BRIDGE_BINARY_QUALIFICATION": ("schema_version", "hegel-phase3-m25-bridge-dag-rust-binary-qualification/1", "implementation_basis_commit"),
        "LIVE_ACTOR_PROTOCOL_QUALIFICATION": ("schema_version", "hegel-phase3-m25-live-actor-protocol-qualification/2", "basis_commit"),
        "PRE_GENESIS_EXECUTION_STATUS": ("schema", "hegel-phase3-m25-execution-status/2", "basis_commit"),
        "PRE_GENESIS_READINESS": ("schema", "hegel-phase3-m25-formal-container-readiness/2", "basis_commit"),
        "FORMAL_GATE_EVIDENCE": ("schema", "hegel-phase3-m25-public-gate-evidence-replay/1", None),
        "FORMAL_GATE_PROMOTION": ("schema", "hegel-phase3-m25-container-ceremony/1", "basis_commit"),
        "FORMAL_TRANSACTION_PUBLICATION_RECEIPT": ("schema", "hegel-phase3-m25-publication-receipt/1", "basis_commit"),
    }
    if set(objects) != set(expected):
        raise ValueError
    for role, (schema_key, schema_value, basis_key) in expected.items():
        value = objects[role]
        if value.get(schema_key) != schema_value:
            raise ValueError
        if basis_key is not None and value.get(basis_key) != basis_commit:
            raise ValueError
    evidence = objects["FORMAL_GATE_EVIDENCE"]
    if evidence.get("artifact_kind") != "FORMAL_GATE_EVIDENCE_INPUTS_PUBLIC_REPLAY":
        raise ValueError


def render_external_status(basis_commit: str, payloads: Mapping[str, bytes]) -> bytes:
    digest_roles = (
        ("actor_qualification", "ACTOR_ELIGIBILITY"),
        ("errata_qualification", "ERRATA_QUALIFICATION"),
        ("m3_implementation_qualification", "M3_IMPLEMENTATION_QUALIFICATION"),
        ("bridge_qualification", "BRIDGE_BINARY_QUALIFICATION"),
        ("live_protocol_qualification", "LIVE_ACTOR_PROTOCOL_QUALIFICATION"),
        ("pre_genesis_execution_status", "PRE_GENESIS_EXECUTION_STATUS"),
        ("pre_genesis_readiness", "PRE_GENESIS_READINESS"),
        ("formal_gate_evidence", "FORMAL_GATE_EVIDENCE"),
        ("formal_gate_promotion", "FORMAL_GATE_PROMOTION"),
        ("formal_transaction_receipt", "FORMAL_TRANSACTION_PUBLICATION_RECEIPT"),
    )
    path_by_role = {role: path for path, role in PUBLICATION_ROLE_REGISTRY.items()}
    lines = [
        "# Phase-3A M2.5 external genesis status", "",
        f"- basis_commit_sha1: `{basis_commit}`", "- formal_gates: `24/24`",
        "- child_state: `NOT_RUN`", "- m3_run_started: `false`",
        "- publication_state: `COMMIT_B_CANDIDATE`",
        "- audit_receipt_scope: `EXCLUDED_SELF_OUTPUT_UNTIL_PREPARE`",
    ]
    lines.extend(
        f"- {label}_sha256: `{hashlib.sha256(payloads[path_by_role[role]]).hexdigest()}`"
        for label, role in digest_roles
    )
    lines.extend(
        [
            "- formal_gate_delta_from_publication_audit: `0`",
            "- phase3_m3_start_required_separately: `true`",
            "- authority_effect: `NONE`", "",
        ]
    )
    return "\n".join(lines).encode("ascii")


def inventory_sha256(rows: list[dict[str, object]]) -> str:
    digest = hashlib.sha256(INVENTORY_DOMAIN)
    for row in rows:
        digest.update(canonical(row))
    return digest.hexdigest()


def load_request(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    value = plain(strict_json(raw))
    if type(value) is not dict or canonical(value) != raw:
        raise ValueError
    claimed = value.get("request_sha256")
    body = dict(value)
    body.pop("request_sha256", None)
    exact = {
        "schema", "purpose_id", "basis_commit_sha1", "actor_image_ref",
        "audit_phase", "candidate_manifest", "runtime_inventory",
        "private_forbidden_raw_path_tokens", "signature_generation_requested",
        "key_seed_or_marker_access_requested", "formal_gate_or_m3_transition_requested",
    }
    if (
        set(body) != exact
        or body["schema"] != REQUEST_SCHEMA
        or body["purpose_id"] != 4
        or body["audit_phase"] not in {"PREPARE_EXCLUDING_RECEIPT", "FINALIZE_INCLUDING_RECEIPT"}
        or type(body["basis_commit_sha1"]) is not str
        or re.fullmatch(r"[0-9a-f]{40}", body["basis_commit_sha1"]) is None
        or type(body["actor_image_ref"]) is not str
        or re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", body["actor_image_ref"]) is None
        or body["signature_generation_requested"] is not False
        or body["key_seed_or_marker_access_requested"] is not False
        or body["formal_gate_or_m3_transition_requested"] is not False
        or type(claimed) is not str
        or hashlib.sha256(canonical(body)).hexdigest() != claimed
    ):
        raise ValueError
    return value


def verify_runtime(root: Path, expected: object) -> str:
    if not isinstance(expected, Mapping):
        raise ValueError
    rows = expected.get("files")
    if type(rows) is not list:
        raise ValueError
    actual: list[dict[str, object]] = []
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError
        raw = path.read_bytes()
        actual.append(
            {
                "runtime_path": path.relative_to(root).as_posix(),
                "repository_path": next(
                    row["repository_path"] for row in rows
                    if isinstance(row, Mapping) and row.get("runtime_path") == path.relative_to(root).as_posix()
                ),
                "basis_tree_mode": next(
                    row["basis_tree_mode"] for row in rows
                    if isinstance(row, Mapping) and row.get("runtime_path") == path.relative_to(root).as_posix()
                ),
                "basis_tree_blob_sha1": next(
                    row["basis_tree_blob_sha1"] for row in rows
                    if isinstance(row, Mapping) and row.get("runtime_path") == path.relative_to(root).as_posix()
                ),
                "byte_length": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    digest = hashlib.sha256(RUNTIME_DOMAIN)
    for row in actual:
        digest.update(canonical(row))
    if (
        actual != rows
        or expected.get("schema") != "hegel-phase3-m25-commit-b-audit-runtime/1"
        or expected.get("file_count") != len(actual)
        or expected.get("runtime_inventory_sha256") != digest.hexdigest()
    ):
        raise ValueError
    return digest.hexdigest()


def verify_candidate(root: Path, manifest_value: object, tokens: object, phase: str) -> str:
    if not isinstance(manifest_value, Mapping) or type(tokens) is not list or any(type(x) is not str for x in tokens):
        raise ValueError
    manifest = dict(manifest_value)
    claimed = manifest.pop("manifest_sha256", None)
    exact_manifest_keys = {
        "schema", "policy_id", "audit_phase", "basis_commit_sha1",
        "changed_path_scope", "allowed_public_prefixes", "executable_prefixes",
        "excluded_self_output_repository_path",
        "excluded_self_output_present_in_candidate", "path_role_registry",
        "role_cardinalities", "candidate_files",
        "candidate_file_count", "candidate_total_byte_length",
        "candidate_inventory_sha256", "authority_boundary",
    }
    expected_receipt_present = phase == "FINALIZE_INCLUDING_RECEIPT"
    required_paths = sorted(
        path for path in PUBLICATION_ROLE_REGISTRY
        if expected_receipt_present or path != AUDIT_RECEIPT_PATH
    )
    expected_registry = [
        {
            "path": path,
            "role_id": PUBLICATION_ROLE_REGISTRY[path],
            "required_cardinality": 1,
        }
        for path in required_paths
    ]
    expected_cardinalities = {
        PUBLICATION_ROLE_REGISTRY[path]: 1 for path in required_paths
    }
    if (
        set(manifest) != exact_manifest_keys
        or manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("policy_id") != POLICY_ID
        or manifest.get("audit_phase") != phase
        or manifest.get("changed_path_scope") != "EXACT_GIT_INDEX_DIFF_FROM_BASIS_COMMIT"
        or manifest.get("allowed_public_prefixes") != ALLOWED_PUBLIC_PREFIXES
        or manifest.get("executable_prefixes") != EXECUTABLE_PREFIXES
        or manifest.get("excluded_self_output_repository_path") != AUDIT_RECEIPT_PATH
        or manifest.get("excluded_self_output_present_in_candidate") is not expected_receipt_present
        or manifest.get("path_role_registry") != expected_registry
        or manifest.get("role_cardinalities") != expected_cardinalities
        or manifest.get("authority_boundary") != {
            "diagnostic_publication_control_only": True,
            "formal_gate_delta": 0,
            "creates_seed_key_signature_marker_or_formal_root": False,
            "m3_start_or_state_transition": False,
        }
        or type(claimed) is not str
        or hashlib.sha256(canonical(manifest)).hexdigest() != claimed
    ):
        raise ValueError
    disk_manifest = plain(strict_json((root / "manifest.json").read_bytes()))
    if disk_manifest != dict(manifest_value) or canonical(disk_manifest) != (root / "manifest.json").read_bytes():
        raise ValueError
    rows = manifest.get("candidate_files")
    if type(rows) is not list or not rows:
        raise ValueError
    expected_paths = [row.get("path") for row in rows if isinstance(row, Mapping)]
    actual_paths = sorted(
        path.relative_to(root / "files").as_posix()
        for path in (root / "files").rglob("*") if path.is_file()
    )
    if (
        expected_paths != actual_paths
        or actual_paths != required_paths
        or len(expected_paths) != len(rows)
    ):
        raise ValueError
    receipt_present = AUDIT_RECEIPT_PATH in actual_paths
    if receipt_present != expected_receipt_present:
        raise ValueError
    actual_rows: list[dict[str, object]] = []
    payloads: dict[str, bytes] = {}
    raw_tokens = [token.encode("utf-8") for token in tokens if len(token) >= 4]
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "path", "role_id", "git_mode", "index_blob_sha1", "byte_length", "sha256"
        }:
            raise ValueError
        path_text = row["path"]
        if type(path_text) is not str or PurePosixPath(path_text).as_posix() != path_text or ".." in PurePosixPath(path_text).parts:
            raise ValueError
        if not (
            path_text == ALLOWED_PUBLIC_PREFIXES[1]
            or path_text.startswith(ALLOWED_PUBLIC_PREFIXES[0] + "/")
        ) or any(
            path_text == prefix or path_text.startswith(prefix + "/")
            for prefix in EXECUTABLE_PREFIXES
        ):
            raise ValueError
        path = root / "files" / Path(*PurePosixPath(path_text).parts)
        if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode) or row["git_mode"] != "100644":
            raise ValueError
        raw = path.read_bytes()
        blob_sha1 = hashlib.sha1(
            b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
        ).hexdigest()
        actual = {
            "path": path_text,
            "role_id": PUBLICATION_ROLE_REGISTRY.get(path_text),
            "git_mode": "100644",
            "index_blob_sha1": row["index_blob_sha1"],
            "byte_length": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        if (
            actual != dict(row)
            or row["index_blob_sha1"] != blob_sha1
            or has_private_key_magic(raw)
            or RAW_PATH_PATTERN.search(raw) is not None
            or any(token in raw for token in raw_tokens)
        ):
            raise ValueError
        if path_text.endswith(".json"):
            decoded = strict_json(raw)
            if has_forbidden_key(decoded) or (
                path_text in CANONICAL_JSON_REQUIRED_PATHS
                and canonical(plain(decoded)) != raw
            ):
                raise ValueError
        elif path_text != "Hegel Machine/docs/phase3_m25_external_status.md":
            raise ValueError
        actual_rows.append(actual)
        payloads[path_text] = raw
    digest = inventory_sha256(actual_rows)
    if (
        digest != manifest.get("candidate_inventory_sha256")
        or len(actual_rows) != manifest.get("candidate_file_count")
        or sum(int(row["byte_length"]) for row in actual_rows) != manifest.get("candidate_total_byte_length")
    ):
        raise ValueError
    validate_role_headers(payloads, str(manifest["basis_commit_sha1"]))
    status_path = "Hegel Machine/docs/phase3_m25_external_status.md"
    if payloads[status_path] != render_external_status(
        str(manifest["basis_commit_sha1"]), payloads
    ):
        raise ValueError
    return digest


def write_probe(path: str) -> dict[str, object]:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    except OSError as exc:
        return {"denied": True, "errno": int(exc.errno or 0)}
    os.close(descriptor)
    try:
        os.unlink(path)
    except OSError:
        pass
    return {"denied": False, "errno": 0}


def live_isolation(image: str, probe) -> dict[str, object]:
    status = probe._proc_status()
    syscalls = probe._syscall_rows()
    mounts = {
        name: write_probe(path)
        for name, path in {
            "root": "/hegel-commit-b-probe", "candidate": "/candidate/hegel-probe",
            "runtime": "/runtime/hegel-probe", "request": "/request.json",
            "tmp": "/tmp/hegel-commit-b-probe",
        }.items()
    }
    environment = dict(sorted(os.environ.items()))
    expected = {
        "HEGEL_ACTOR_IMAGE_REF": image,
        "HEGEL_ACTOR_PROFILE_ID": "hegel-owner-accepted-container-technical-actors-v1",
        "HEGEL_PURPOSE_ID": "4",
        "LANG": "C", "LC_ALL": "C", "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
    }
    cgroup = {
        "memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
        "memory_swap_max": Path("/sys/fs/cgroup/memory.swap.max").read_text().strip(),
        "pids_max": Path("/sys/fs/cgroup/pids.max").read_text().strip(),
    }
    checks = {
        "nonroot_exact": os.getuid() == 65534 and os.getgid() == 65534,
        "capability_sets_zero": all(status.get(key) == "0000000000000000" for key in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")),
        "no_new_privileges": status.get("NoNewPrivs") == 1,
        "seccomp_filter": status.get("Seccomp") == 2,
        "network_loopback_only": sorted(os.listdir("/sys/class/net")) == ["lo"],
        "six_syscalls_blocked_eperm": len(syscalls) == 6 and all(row.get("return_value") == -1 and row.get("errno") == 1 for row in syscalls),
        "immutable_mounts_read_only": all(mounts[name]["denied"] is True and mounts[name]["errno"] in {1, 13, 30} for name in ("root", "candidate", "runtime", "request")),
        "tmp_private_writable": mounts["tmp"] == {"denied": False, "errno": 0},
        "environment_exact": environment == expected,
        "inherited_fds_exact": probe._open_fds() == [0, 1, 2],
        "cgroup_limits_exact": cgroup == {"memory_max": str(512 * 1024 * 1024), "memory_swap_max": "0", "pids_max": "64"},
    }
    if not all(checks.values()):
        raise ValueError
    body: dict[str, object] = {
        "schema": "hegel-phase3-m25-commit-b-purpose4-live-isolation/1",
        "purpose_id": 4, "actor_image_ref": image, "uid": os.getuid(), "gid": os.getgid(),
        "required_checks": checks, "all_required_checks_passed": True,
    }
    body["receipt_sha256"] = hashlib.sha256(canonical(body)).hexdigest()
    return body


def main() -> int:
    try:
        if len(sys.argv) != 3 or os.getuid() != 65534 or os.getgid() != 65534:
            raise ValueError
        candidate, request_path = Path(sys.argv[1]), Path(sys.argv[2])
        request = load_request(request_path)
        if os.environ.get("HEGEL_PURPOSE_ID") != "4" or os.environ.get("HEGEL_ACTOR_IMAGE_REF") != request["actor_image_ref"]:
            raise ValueError
        runtime_digest = verify_runtime(Path("/runtime"), request["runtime_inventory"])
        phase = str(request["audit_phase"])
        inventory = verify_candidate(
            candidate, request["candidate_manifest"],
            request["private_forbidden_raw_path_tokens"], phase,
        )
        sys.path.insert(0, "/runtime")
        import probe
        isolation = live_isolation(str(request["actor_image_ref"]), probe)
        token_hashes = sorted(
            hashlib.sha256(token.encode("utf-8")).hexdigest()
            for token in request["private_forbidden_raw_path_tokens"]
        )
        checks = {
            "exact_manifest_and_file_set": True,
            "path_mode_size_sha256_bound": True,
            "nonallowlisted_and_executable_paths_absent": True,
            "json_strict_duplicate_free_and_required_bit_exact": True,
            "forbidden_secret_field_names_absent": True,
            "private_key_magic_and_complete_blocks_absent": True,
            "raw_author_or_host_paths_absent": True,
            "receipt_scope_exact_for_audit_phase": True,
            "no_key_seed_signature_marker_or_formal_action": True,
        }
        response: dict[str, object] = {
            "schema": RECEIPT_SCHEMA,
            "artifact_kind": "DIAGNOSTIC_PUBLICATION_CONTROL",
            "policy_id": POLICY_ID,
            "purpose_id": 4,
            "audit_phase": phase,
            "basis_commit_sha1": request["basis_commit_sha1"],
            "actor_image_ref": request["actor_image_ref"],
            "request_sha256": request["request_sha256"],
            "candidate_manifest": request["candidate_manifest"],
            "actor_recomputed_inventory_sha256": inventory,
            "runtime_inventory_sha256": runtime_digest,
            "private_forbidden_raw_path_token_sha256s": token_hashes,
            "isolation_live_receipt": isolation,
            "required_checks": checks,
            "all_required_checks_passed": True,
            "authority_disclosure": AUTHORITY_DISCLOSURE,
            "authority_boundary": {
                "diagnostic_publication_control_only": True,
                "formal_gate_delta": 0,
                "creates_seed_key_signature_marker_or_formal_root": False,
                "m3_start_or_state_transition": False,
            },
        }
        response["receipt_sha256"] = hashlib.sha256(canonical(response)).hexdigest()
        sys.stdout.buffer.write(canonical(response))
        return 0
    except BaseException:
        sys.stderr.write(FAIL + "\n")
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
