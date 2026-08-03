#!/usr/bin/env python3
"""No-key purpose-4 worker for detached Gate-17 Git-object replay."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Mapping


FAIL_CODE = "FAIL_GATE17_PURPOSE4_DETACHED_WORKER"


def fail() -> "None":
    try:
        sys.stderr.write(FAIL_CODE + "\n")
    finally:
        raise SystemExit(70)


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def require_request(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError
    expected = {
        "schema",
        "purpose_id",
        "basis_commit_sha1",
        "actor_image_ref",
        "snapshot_manifest",
        "runtime_inventory",
        "runtime_source_bindings",
        "auditor_key_id_hex",
        "audited_at_unix_seconds",
        "signature_generation_requested",
        "key_seed_or_marker_access_requested",
        "request_sha256",
    }
    if set(value) != expected:
        raise ValueError
    body = dict(value)
    claimed = body.pop("request_sha256")
    if (
        value["schema"] != "hegel-gate17-purpose4-detached-request/1"
        or value["purpose_id"] != 4
        or type(value["basis_commit_sha1"]) is not str
        or re.fullmatch(r"[0-9a-f]{40}", value["basis_commit_sha1"]) is None
        or type(value["actor_image_ref"]) is not str
        or re.fullmatch(
            r"[^@\s]+@sha256:[0-9a-f]{64}", value["actor_image_ref"]
        )
        is None
        or value["signature_generation_requested"] is not False
        or value["key_seed_or_marker_access_requested"] is not False
        or type(claimed) is not str
        or hashlib.sha256(canonical_json(body)).hexdigest() != claimed
    ):
        raise ValueError
    return value


def _read_cgroup_value(path: str) -> str:
    return Path(path).read_text(encoding="ascii").strip()


def _write_probe(path: str) -> dict[str, object]:
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


def purpose4_live_probe(probe_module) -> dict[str, object]:
    status = probe_module._proc_status()
    capability_fields = ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
    syscall_rows = probe_module._syscall_rows()
    filesystem = {
        "root_write": _write_probe("/hegel-purpose4-root-write-probe"),
        "snapshot_write": _write_probe("/snapshot/hegel-purpose4-write-probe"),
        "runtime_write": _write_probe("/runtime/hegel-purpose4-write-probe"),
        "request_write": _write_probe("/request.json"),
        "tmp_write": _write_probe("/tmp/hegel-purpose4-write-probe"),
        "forbidden_paths_present": [
            path for path in probe_module.FORBIDDEN_PATHS if Path(path).exists()
        ],
        "cross_purpose_paths_present": [
            path for path in probe_module.CROSS_PURPOSE_PATHS if Path(path).exists()
        ],
    }
    environment = dict(sorted(os.environ.items()))
    expected_environment = {
        "HEGEL_ACTOR_IMAGE_REF": environment.get("HEGEL_ACTOR_IMAGE_REF"),
        "HEGEL_ACTOR_PROFILE_ID": "hegel-owner-accepted-container-technical-actors-v1",
        "HEGEL_PURPOSE_ID": "4",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/runtime/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }
    blocked = all(
        isinstance(row, Mapping)
        and row.get("return_value") == -1
        and row.get("errno") == 1
        for row in syscall_rows
    )
    denials = all(
        isinstance(filesystem[name], Mapping)
        and filesystem[name].get("denied") is True
        and filesystem[name].get("errno") in {1, 13, 30}
        for name in ("root_write", "snapshot_write", "runtime_write", "request_write")
    )
    cgroup = {
        "memory_max": _read_cgroup_value("/sys/fs/cgroup/memory.max"),
        "memory_swap_max": _read_cgroup_value("/sys/fs/cgroup/memory.swap.max"),
        "pids_max": _read_cgroup_value("/sys/fs/cgroup/pids.max"),
    }
    checks = {
        "identity_nonroot_exact": os.getuid() == 65534 and os.getgid() == 65534,
        "capability_sets_all_zero": all(
            status.get(name) == "0000000000000000" for name in capability_fields
        ),
        "no_new_privileges_exact": status.get("NoNewPrivs") == 1,
        "seccomp_filter_exact": status.get("Seccomp") == 2,
        "network_interfaces_exactly_lo": sorted(os.listdir("/sys/class/net")) == ["lo"],
        "six_syscalls_blocked_eperm": blocked and len(syscall_rows) == 6,
        "immutable_mount_writes_denied": denials,
        "tmp_write_succeeded": filesystem["tmp_write"] == {"denied": False, "errno": 0},
        "forbidden_paths_absent": not filesystem["forbidden_paths_present"],
        "cross_purpose_paths_absent": not filesystem["cross_purpose_paths_present"],
        "environment_exact": environment == expected_environment,
        "inherited_fds_exact": probe_module._open_fds() == [0, 1, 2],
        "memory_512m_exact": cgroup["memory_max"] == str(512 * 1024 * 1024),
        "memory_swap_zero_exact": cgroup["memory_swap_max"] == "0",
        "pids_limit_64_exact": cgroup["pids_max"] == "64",
    }
    if not all(checks.values()):
        raise ValueError
    body: dict[str, object] = {
        "schema": "hegel-gate17-purpose4-live-probe/1",
        "profile_id": "hegel-owner-accepted-container-technical-actors-v1",
        "purpose_id": 4,
        "implementation": "python-ctypes-v1",
        "actor_image_ref": environment["HEGEL_ACTOR_IMAGE_REF"],
        "identity": {"uid": os.getuid(), "gid": os.getgid(), "pid": os.getpid()},
        "proc_status": status,
        "namespaces": probe_module._namespace_rows(),
        "network_interfaces": sorted(os.listdir("/sys/class/net")),
        "syscall_probes": syscall_rows,
        "filesystem_probes": filesystem,
        "environment": environment,
        "open_fds": [0, 1, 2],
        "cgroup_limits": cgroup,
        "required_checks": checks,
        "all_required_checks_passed": True,
    }
    body["receipt_sha256"] = hashlib.sha256(canonical_json(body)).hexdigest()
    return body


def runtime_inventory(runtime: Path) -> dict[str, object]:
    domain = b"HEGEL/GATE17/PURPOSE4_RUNTIME_INVENTORY/V1\x00"
    digest = hashlib.sha256(domain)
    rows: list[dict[str, object]] = []
    for path in sorted(
        (item for item in runtime.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(runtime).as_posix(),
    ):
        if path.is_symlink():
            raise ValueError
        relative = path.relative_to(runtime).as_posix()
        raw_path = relative.encode("utf-8")
        raw = path.read_bytes()
        file_digest = hashlib.sha256(raw).hexdigest()
        rows.append(
            {"path": relative, "byte_length": len(raw), "sha256": file_digest}
        )
        digest.update(len(raw_path).to_bytes(4, "big"))
        digest.update(raw_path)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(bytes.fromhex(file_digest))
    return {
        "files": rows,
        "file_count": len(rows),
        "inventory_sha256": digest.hexdigest(),
    }


def main() -> int:
    if len(sys.argv) != 3:
        fail()
    if (
        os.getuid() != 65534
        or os.getgid() != 65534
        or os.environ.get("HEGEL_ACTOR_PROFILE_ID")
        != "hegel-owner-accepted-container-technical-actors-v1"
        or os.environ.get("HEGEL_PURPOSE_ID") != "4"
    ):
        fail()
    snapshot = Path(sys.argv[1])
    request_path = Path(sys.argv[2])
    runtime = Path("/runtime")
    try:
        request = require_request(json.loads(request_path.read_bytes()))
        actual_runtime = runtime_inventory(runtime)
        if actual_runtime != request["runtime_inventory"]:
            raise ValueError
        sys.path.insert(0, str(runtime))
        from hegel_machine.phase3_m25_parent_absence_audit_v1 import (
            build_parent_absence_attestation_fields_v2,
            generate_parent_absence_audit_v1,
            parent_absence_public_receipt_v1,
            replay_parent_absence_audit_v1,
        )
        from hegel_machine.phase3_m25_purpose4_detached_audit_v1 import (
            _validate_runtime_source_bindings_v1,
            validate_detached_parent_snapshot_v1,
        )
        from hegel_machine.phase3_m25_wire_v1 import (
            OBJECT_TAGS,
            candidate_content_root,
            encode_formal_object,
            external_signature_preimage_v1,
        )
        import probe as probe_module

        snapshot_manifest = request["snapshot_manifest"]
        if not isinstance(snapshot_manifest, Mapping):
            raise ValueError
        git_binding = snapshot_manifest.get("git_runtime_binding")
        if not isinstance(git_binding, Mapping):
            raise ValueError
        basis_commit = request["basis_commit_sha1"]
        if (
            snapshot_manifest.get("basis_commit_sha1") != basis_commit
            or os.environ.get("HEGEL_ACTOR_IMAGE_REF") != request["actor_image_ref"]
        ):
            raise ValueError
        source_bindings = request["runtime_source_bindings"]
        if not isinstance(source_bindings, Mapping):
            raise ValueError
        _validate_runtime_source_bindings_v1(
            source_bindings,
            basis_commit=str(basis_commit),
            runtime_inventory=actual_runtime,
            git_binding=git_binding,
        )
        git_binary = runtime / "bin/git"
        if (
            git_binding.get("container_path") != "/runtime/bin/git"
            or git_binding.get("byte_length") != git_binary.stat().st_size
            or git_binding.get("sha256")
            != hashlib.sha256(git_binary.read_bytes()).hexdigest()
        ):
            raise ValueError
        validate_detached_parent_snapshot_v1(
            snapshot,
            snapshot_manifest,
            git_executable=git_binary,
            require_frozen_parent=True,
            expected_basis_commit=str(basis_commit),
        )
        live_probe = purpose4_live_probe(probe_module)

        # The host inventory is not an audit result.  The actor starts from
        # the object database and independently regenerates all formal rows,
        # blob identity checks, path predicates, and content predicates.
        evidence = generate_parent_absence_audit_v1(
            snapshot,
            git_executable=git_binary,
        )
        replay_parent_absence_audit_v1(
            evidence,
            git_executable=git_binary,
        )
        receipt = parent_absence_public_receipt_v1(evidence)
        auditor_key_id = bytes.fromhex(str(request["auditor_key_id_hex"]))
        fields = build_parent_absence_attestation_fields_v2(
            evidence,
            auditor_key_id=auditor_key_id,
            audited_at_unix_seconds=request["audited_at_unix_seconds"],
        )
        cbor = encode_formal_object("ParentManifestAbsenceAttestationV2", fields)
        root = candidate_content_root("ParentManifestAbsenceAttestationV2", fields)
        preimage = external_signature_preimage_v1(
            OBJECT_TAGS["ParentManifestAbsenceAttestationV2"], root, 4, 0
        )
        response: dict[str, object] = {
            "schema": "hegel-gate17-purpose4-detached-response/1",
            "purpose_id": 4,
            "basis_commit_sha1": basis_commit,
            "actor_image_ref": request["actor_image_ref"],
            "request_sha256": request["request_sha256"],
            "snapshot_manifest_sha256": snapshot_manifest["manifest_sha256"],
            "runtime_inventory_sha256": actual_runtime["inventory_sha256"],
            "runtime_source_binding_sha256": source_bindings["binding_sha256"],
            "git_runtime_binding": dict(git_binding),
            "isolation_live_probe_receipt": live_probe,
            "parent_absence_public_receipt": receipt,
            "attestation_cbor_hex": cbor.hex(),
            "attestation_root_hex": root.hex(),
            "signature_preimage_hex": preimage.hex(),
            "signer_purpose_id": 4,
            "signer_key_epoch": 0,
            "signature_present": False,
            "private_key_seed_marker_accessed": False,
            "network_access_performed": False,
        }
        response["response_sha256"] = hashlib.sha256(canonical_json(response)).hexdigest()
        sys.stdout.buffer.write(canonical_json(response))
        return 0
    except BaseException:
        fail()
    return 70


if __name__ == "__main__":
    raise SystemExit(main())
