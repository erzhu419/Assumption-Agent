#!/usr/bin/env python3
"""Purpose-4-only parent-absence replay and Ed25519 attestation worker.

This bounded worker currently replays the supplied formal row bundle.  It does
not yet traverse a detached Git object store; the host executor therefore
keeps the dedicated purpose-4 formal blocker active.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import types
from typing import Mapping

import phase3_m25_actor_operation_probe_v1 as operation_probe


STATE = Path("/state")
INPUT = Path("/input")
OUTPUT = Path("/output")
PRIVATE_KEY = STATE / "ed25519-private.pem"
PUBLIC_DER = OUTPUT / "ed25519-public.der"
SIGNATURE = OUTPUT / "ed25519-signature.bin"
MAX_PUBLIC_REQUEST_BYTES = 32 * 1024 * 1024


class WorkerFailure(RuntimeError):
    pass


def fail(code: str) -> "None":
    try:
        sys.stderr.write(code + "\n")
    finally:
        raise SystemExit(70)


def require_profile() -> None:
    if os.environ.get("HEGEL_ACTOR_PROFILE_ID") != (
        "hegel-owner-accepted-container-technical-actors-v1"
    ):
        fail("FAIL_M25_PARENT_AUDITOR_PROFILE")
    if os.environ.get("HEGEL_PURPOSE_ID") != "4":
        fail("FAIL_M25_PARENT_AUDITOR_PURPOSE")
    if os.getuid() != 65534 or os.getgid() != 65534:
        fail("FAIL_M25_PARENT_AUDITOR_IDENTITY")


def exclusive_bytes(path: Path, payload: bytes, mode: int) -> None:
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


def run_quiet(command: list[str]) -> bytes:
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=120,
        env={"LC_ALL": "C.UTF-8", "PATH": "/usr/local/bin:/usr/bin:/bin"},
    )
    if completed.returncode != 0:
        raise WorkerFailure
    return completed.stdout


def keygen_or_resume(*, recovery: bool) -> None:
    require_profile()
    if PUBLIC_DER.exists():
        raise WorkerFailure
    os.chmod(STATE, 0o700)
    if PRIVATE_KEY.exists():
        metadata = PRIVATE_KEY.stat()
        if (
            PRIVATE_KEY.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise WorkerFailure
    else:
        if recovery:
            fail("FAIL_M25_PARENT_AUDITOR_RECOVERY_KEY_ABSENT")
        run_quiet([
            "/usr/bin/openssl", "genpkey", "-algorithm", "ED25519",
            "-out", str(PRIVATE_KEY),
        ])
        os.chmod(PRIVATE_KEY, 0o600)
    der = run_quiet([
        "/usr/bin/openssl", "pkey", "-in", str(PRIVATE_KEY),
        "-pubout", "-outform", "DER",
    ])
    if len(der) != 44:
        raise WorkerFailure
    exclusive_bytes(PUBLIC_DER, der, 0o644)


def transport_decode(value: object) -> object:
    if type(value) is dict:
        if set(value) == {"bytes_hex"} and type(value["bytes_hex"]) is str:
            return bytes.fromhex(value["bytes_hex"])
        return {str(key): transport_decode(item) for key, item in value.items()}
    if type(value) is list:
        return tuple(transport_decode(item) for item in value)
    if value is None or type(value) in {bool, int, str}:
        return value
    raise WorkerFailure


def read_request() -> Mapping[str, object]:
    raw = (INPUT / "parent-audit-replay.json").read_bytes()
    if len(raw) > MAX_PUBLIC_REQUEST_BYTES:
        raise WorkerFailure
    value = transport_decode(json.loads(raw))
    if not isinstance(value, Mapping):
        raise WorkerFailure
    return value


def load_wire():
    package = types.ModuleType("hegel_machine")
    package.__path__ = ["/input/src/hegel_machine"]
    package.__package__ = "hegel_machine"
    sys.modules["hegel_machine"] = package
    return importlib.import_module("hegel_machine.phase3_m25_wire_v1")


def parent_audit_sign() -> None:
    require_profile()
    if not PRIVATE_KEY.is_file() or SIGNATURE.exists():
        raise WorkerFailure
    wire = load_wire()
    request = read_request()

    def decoded_rows(key: str, schema: str):
        raw_rows = request[key]
        if not isinstance(raw_rows, tuple):
            raise WorkerFailure
        return tuple(
            wire.decode_formal_object(row, expected_name=schema).fields
            for row in raw_rows
        )

    top = decoded_rows("top_level_path_cbor", "AuditedPathBlobRecordV1")
    history = decoded_rows("history_cbor", "AuditedHistoryRowV1")
    legacy = decoded_rows("legacy_source_cbor", "LegacyParentSourceRowV1")
    touched_raw = request["touched_path_cbor_by_history_row"]
    if not isinstance(touched_raw, tuple):
        raise WorkerFailure
    touched = tuple(
        tuple(
            wire.decode_formal_object(row, expected_name="AuditedPathBlobRecordV1").fields
            for row in group
        )
        for group in touched_raw
    )
    audit_decoded = wire.decode_formal_object(
        request["audit_bundle_cbor"], expected_name="ParentAbsenceAuditBundleV1"
    )
    audit_root = wire.validate_parent_absence_audit_bundle_v1(
        top, history, touched, legacy, audit_decoded.fields
    )
    attestation_decoded = wire.decode_formal_object(
        request["attestation_cbor"], expected_name="ParentManifestAbsenceAttestationV2"
    )
    if attestation_decoded.fields["audit_bundle_root"] != audit_root:
        raise WorkerFailure
    root = wire.candidate_content_root(
        "ParentManifestAbsenceAttestationV2", attestation_decoded.fields
    )
    if root != request["expected_attestation_root"]:
        raise WorkerFailure
    preimage = wire.external_signature_preimage_v1(
        wire.OBJECT_TAGS["ParentManifestAbsenceAttestationV2"], root, 4, 0
    )
    if (INPUT / "signing-preimage.bin").read_bytes() != preimage:
        raise WorkerFailure
    signature = run_quiet([
        "/usr/bin/openssl", "pkeyutl", "-sign", "-rawin",
        "-inkey", str(PRIVATE_KEY), "-in", str(INPUT / "signing-preimage.bin"),
    ])
    if len(signature) != 64:
        raise WorkerFailure
    exclusive_bytes(SIGNATURE, signature, 0o644)


def main() -> int:
    if len(sys.argv) != 2:
        fail("FAIL_M25_PARENT_AUDITOR_ARGUMENTS")
    operation = sys.argv[1]
    try:
        operation_probe.qualify_operation_v1(operation)
        if operation == "qualify-only":
            pass
        elif operation == "keygen":
            keygen_or_resume(recovery=False)
        elif operation == "keygen-resume":
            keygen_or_resume(recovery=True)
        elif operation == "purpose4-parent-sign":
            parent_audit_sign()
        else:
            raise WorkerFailure
        return 0
    except (Exception, SystemExit):
        fail("FAIL_M25_PARENT_AUDITOR_OPERATION")
    return 70


if __name__ == "__main__":
    raise SystemExit(main())
