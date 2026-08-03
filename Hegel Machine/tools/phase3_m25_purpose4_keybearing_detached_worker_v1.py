#!/usr/bin/env python3
"""Key-bearing purpose-4 worker over a frozen detached Git snapshot.

The host supplies only a public request that binds Commit-A, the immutable
runtime, the detached snapshot, a timestamp, and the already-created local
purpose-4 key ID.  This process regenerates the parent-absence evidence from
Git objects, replays it, constructs the formal attestation, derives the exact
signature preimage in memory, and signs with ``/state/ed25519-private.pem``.

It never accepts audit rows, an attestation, a signing preimage, a private key,
or a seed from the host, and it never creates a key.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Mapping, NoReturn


FAIL_CODE = "FAIL_GATE17_PURPOSE4_KEYBEARING_DETACHED_WORKER"
PROFILE_ID = "hegel-owner-accepted-container-technical-actors-v1"
OPERATION_ID = "purpose4-parent-sign"
REQUEST = Path("/input/purpose4-keybearing-request.json")
RUNTIME = Path("/input/runtime")
SNAPSHOT = Path("/input/detached-parent-snapshot")
STATE = Path("/state")
OUTPUT = Path("/output")
TEMPORARY = Path("/tmp")
PRIVATE_KEY = STATE / "ed25519-private.pem"
RESPONSE = OUTPUT / "purpose4-keybearing-detached-response.json"
OPENSSL = Path("/usr/bin/openssl")
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
ED25519_SPKI_PREFIX = bytes.fromhex("302a300506032b6570032100")


class WorkerFailure(RuntimeError):
    pass


def _fail() -> NoReturn:
    try:
        sys.stderr.write(FAIL_CODE + "\n")
    finally:
        raise SystemExit(70)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def _reject_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise WorkerFailure
        result[key] = value
    return result


def _read_request(path: Path) -> object:
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size > MAX_REQUEST_BYTES
    ):
        raise WorkerFailure
    payload = path.read_bytes()
    if len(payload) != metadata.st_size or not payload.endswith(b"\n"):
        raise WorkerFailure
    return json.loads(payload, object_pairs_hook=_reject_duplicate_object)


def _require_owned_directory(path: Path, mode: int) -> None:
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != mode
        or metadata.st_uid != os.getuid()
        or metadata.st_gid != os.getgid()
    ):
        raise WorkerFailure


def _require_private_key(path: Path) -> None:
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_uid != os.getuid()
        or metadata.st_gid != os.getgid()
        or metadata.st_nlink != 1
        or metadata.st_size <= 0
        or metadata.st_size > 16 * 1024
    ):
        raise WorkerFailure


def _exclusive_bytes(path: Path, payload: bytes, mode: int) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        os.fchmod(descriptor, mode)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise WorkerFailure
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _run_openssl(arguments: list[str], *, working_directory: Path) -> bytes:
    completed = subprocess.run(
        [str(OPENSSL), *arguments],
        cwd=working_directory,
        env={"LANG": "C", "LC_ALL": "C.UTF-8", "PATH": "/usr/bin:/bin"},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0:
        raise WorkerFailure
    return completed.stdout


def _sign_and_verify_local_v1(
    preimage: bytes,
    expected_key_id: bytes,
    *,
    state_root: Path = STATE,
    temporary_root: Path = TEMPORARY,
) -> tuple[bytes, bytes]:
    """Sign an internally built preimage with an existing purpose-4 key.

    This helper deliberately has no key-generation branch.  Explicit roots are
    accepted only so non-authoritative unit tests can exercise the same file
    and OpenSSL boundary in a disposable directory.
    """

    if (
        type(preimage) is not bytes
        or not preimage
        or len(preimage) > 4096
        or type(expected_key_id) is not bytes
        or len(expected_key_id) != 16
    ):
        raise WorkerFailure
    _require_owned_directory(state_root, 0o700)
    _require_owned_directory(temporary_root, 0o700)
    private_key = state_root / "ed25519-private.pem"
    _require_private_key(private_key)
    if not OPENSSL.is_file() or OPENSSL.is_symlink():
        raise WorkerFailure

    workspace = Path(
        tempfile.mkdtemp(prefix="hegel-purpose4-sign-", dir=temporary_root)
    )
    try:
        _require_owned_directory(workspace, 0o700)
        public_der = _run_openssl(
            ["pkey", "-in", str(private_key), "-pubout", "-outform", "DER"],
            working_directory=workspace,
        )
        if (
            len(public_der) != len(ED25519_SPKI_PREFIX) + 32
            or not public_der.startswith(ED25519_SPKI_PREFIX)
        ):
            raise WorkerFailure
        public_key = public_der[len(ED25519_SPKI_PREFIX) :]
        if hashlib.sha256(public_key).digest()[:16] != expected_key_id:
            raise WorkerFailure

        preimage_path = workspace / "internally-derived-preimage.bin"
        public_path = workspace / "derived-public-key.der"
        signature_path = workspace / "signature.bin"
        _exclusive_bytes(preimage_path, preimage, 0o600)
        _exclusive_bytes(public_path, public_der, 0o600)
        signature = _run_openssl(
            [
                "pkeyutl",
                "-sign",
                "-rawin",
                "-inkey",
                str(private_key),
                "-in",
                str(preimage_path),
            ],
            working_directory=workspace,
        )
        if len(signature) != 64:
            raise WorkerFailure
        _exclusive_bytes(signature_path, signature, 0o600)
        _run_openssl(
            [
                "pkeyutl",
                "-verify",
                "-rawin",
                "-pubin",
                "-keyform",
                "DER",
                "-inkey",
                str(public_path),
                "-sigfile",
                str(signature_path),
                "-in",
                str(preimage_path),
            ],
            working_directory=workspace,
        )
        return public_key, signature
    finally:
        shutil.rmtree(workspace)
        _fsync_directory(temporary_root)


def _require_runtime_tree(runtime: Path) -> None:
    metadata = runtime.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise WorkerFailure
    for path in runtime.rglob("*"):
        item = path.lstat()
        if stat.S_ISLNK(item.st_mode) or not (
            stat.S_ISDIR(item.st_mode) or stat.S_ISREG(item.st_mode)
        ):
            raise WorkerFailure


def _write_response(response: Mapping[str, object]) -> None:
    payload = _canonical_json(response)
    if len(payload) > MAX_RESPONSE_BYTES or RESPONSE.exists() or RESPONSE.is_symlink():
        raise WorkerFailure
    _exclusive_bytes(RESPONSE, payload, 0o644)
    _fsync_directory(OUTPUT)


def _run_keybearing_replay() -> None:
    if (
        os.getuid() != 65534
        or os.getgid() != 65534
        or os.environ.get("HEGEL_ACTOR_PROFILE_ID") != PROFILE_ID
        or os.environ.get("HEGEL_PURPOSE_ID") != "4"
    ):
        raise WorkerFailure
    _require_runtime_tree(RUNTIME)
    sys.path.insert(0, str(RUNTIME))
    sys.path.insert(0, str(RUNTIME / "tools"))

    from hegel_machine.phase3_m25_parent_absence_audit_v1 import (
        build_parent_absence_attestation_fields_v2,
        generate_parent_absence_audit_v1,
        parent_absence_public_receipt_v1,
        replay_parent_absence_audit_v1,
    )
    from hegel_machine.phase3_m25_purpose4_detached_audit_v1 import (
        _runtime_inventory,
        _validate_runtime_source_bindings_v1,
        validate_detached_parent_snapshot_v1,
    )
    from hegel_machine.phase3_m25_purpose4_keybearing_detached_v1 import (
        RESPONSE_SCHEMA,
        validate_purpose4_keybearing_request_v1,
    )
    from hegel_machine.phase3_m25_wire_v1 import (
        OBJECT_TAGS,
        candidate_content_root,
        encode_formal_object,
        external_signature_preimage_v1,
    )
    import phase3_m25_actor_operation_probe_v1 as operation_probe

    request = validate_purpose4_keybearing_request_v1(_read_request(REQUEST))
    if (
        os.environ.get("HEGEL_OPERATION_ID") != OPERATION_ID
        or os.environ.get("HEGEL_OPERATION_REQUEST_SHA256")
        != request["request_sha256"]
        or os.environ.get("HEGEL_BASIS_COMMIT") != request["basis_commit_sha1"]
        or os.environ.get("HEGEL_ACTOR_IMAGE_REF") != request["actor_image_ref"]
    ):
        raise WorkerFailure
    operation_receipt = operation_probe.qualify_operation_v1(OPERATION_ID)

    actual_runtime = _runtime_inventory(RUNTIME)
    runtime_bindings = request["runtime_bindings"]
    if not isinstance(runtime_bindings, Mapping):
        raise WorkerFailure
    expected_inventory = runtime_bindings["runtime_inventory"]
    source_bindings = runtime_bindings["runtime_source_bindings"]
    if actual_runtime != expected_inventory or not isinstance(source_bindings, Mapping):
        raise WorkerFailure
    snapshot_manifest = request["snapshot_manifest"]
    if not isinstance(snapshot_manifest, Mapping):
        raise WorkerFailure
    git_binding = snapshot_manifest.get("git_runtime_binding")
    if not isinstance(git_binding, Mapping):
        raise WorkerFailure
    _validate_runtime_source_bindings_v1(
        source_bindings,
        basis_commit=str(request["basis_commit_sha1"]),
        runtime_inventory=actual_runtime,
        git_binding=git_binding,
    )
    git_binary = RUNTIME / "bin/git"
    if (
        git_binary.is_symlink()
        or not git_binary.is_file()
        # Preserve the no-key adapter's frozen logical runtime binding.  The
        # executor integration nests those same immutable bytes below its
        # already-read-only /input mount; the digest/length are rechecked here.
        or git_binding.get("container_path") != "/runtime/bin/git"
        or git_binding.get("byte_length") != git_binary.stat().st_size
        or git_binding.get("sha256")
        != hashlib.sha256(git_binary.read_bytes()).hexdigest()
    ):
        raise WorkerFailure
    validate_detached_parent_snapshot_v1(
        SNAPSHOT,
        snapshot_manifest,
        git_executable=git_binary,
        require_frozen_parent=True,
        expected_basis_commit=str(request["basis_commit_sha1"]),
    )

    # The audit module invokes ``git`` by name.  Limit that resolution to the
    # immutable runtime copy for both generation and independent regeneration.
    original_path = os.environ.get("PATH")
    os.environ["PATH"] = f"{RUNTIME / 'bin'}:/usr/bin:/bin"
    try:
        evidence = generate_parent_absence_audit_v1(SNAPSHOT)
        replay_parent_absence_audit_v1(evidence, repository=SNAPSHOT)
    finally:
        if original_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = original_path

    audit_cbor = encode_formal_object(
        "ParentAbsenceAuditBundleV1", evidence.audit_bundle_fields
    )
    audit_root = candidate_content_root(
        "ParentAbsenceAuditBundleV1", evidence.audit_bundle_fields
    )
    if audit_root != evidence.audit_bundle_root:
        raise WorkerFailure
    expected_key_id = bytes.fromhex(str(request["expected_local_key_id_hex"]))
    attestation_fields = build_parent_absence_attestation_fields_v2(
        evidence,
        auditor_key_id=expected_key_id,
        audited_at_unix_seconds=int(request["audited_at_unix_seconds"]),
    )
    attestation_cbor = encode_formal_object(
        "ParentManifestAbsenceAttestationV2", attestation_fields
    )
    attestation_root = candidate_content_root(
        "ParentManifestAbsenceAttestationV2", attestation_fields
    )
    preimage = external_signature_preimage_v1(
        OBJECT_TAGS["ParentManifestAbsenceAttestationV2"],
        attestation_root,
        4,
        0,
    )
    public_key, signature = _sign_and_verify_local_v1(
        preimage,
        expected_key_id,
    )
    response: dict[str, object] = {
        "schema": RESPONSE_SCHEMA,
        "purpose_id": 4,
        "basis_commit_sha1": request["basis_commit_sha1"],
        "actor_image_ref": request["actor_image_ref"],
        "request_sha256": request["request_sha256"],
        "snapshot_manifest_sha256": snapshot_manifest["manifest_sha256"],
        "runtime_inventory_sha256": actual_runtime["inventory_sha256"],
        "runtime_source_binding_sha256": source_bindings["binding_sha256"],
        "operation_probe_receipt": dict(operation_receipt),
        "parent_absence_public_receipt": parent_absence_public_receipt_v1(evidence),
        "audit_bundle_cbor_hex": audit_cbor.hex(),
        "audit_bundle_root_hex": audit_root.hex(),
        "attestation_cbor_hex": attestation_cbor.hex(),
        "attestation_root_hex": attestation_root.hex(),
        "signer_public_key_32_hex": public_key.hex(),
        "signer_key_id_hex": expected_key_id.hex(),
        "signer_key_epoch": 0,
        "signature_hex": signature.hex(),
        "signature_verified_inside_actor": True,
        "audit_rows_received_from_host": False,
        "attestation_received_from_host": False,
        "signing_preimage_received_from_host": False,
        "private_key_exported": False,
        "raw_split_seed_accessed": False,
        "network_access_performed": False,
    }
    response["response_sha256"] = hashlib.sha256(
        _canonical_json(response)
    ).hexdigest()
    _write_response(response)


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] != OPERATION_ID:
        _fail()
    try:
        _run_keybearing_replay()
        return 0
    except BaseException:
        _fail()
    return 70


if __name__ == "__main__":
    raise SystemExit(main())
