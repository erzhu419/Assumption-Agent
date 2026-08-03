#!/usr/bin/env python3
"""Purpose-2-only Python bridge replay and Ed25519 attestation worker."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import types

import phase3_m25_actor_operation_probe_v1 as operation_probe


STATE = Path("/state")
INPUT = Path("/input")
OUTPUT = Path("/output")
PRIVATE_KEY = STATE / "ed25519-private.pem"
PUBLIC_DER = OUTPUT / "ed25519-public.der"
SIGNATURE = OUTPUT / "ed25519-signature.bin"
BRIDGE_DAG_PACKAGE = INPUT / "bridge-dag-package.cbor"
BRIDGE_REPLAY_RECEIPT = OUTPUT / "bridge-dag-replay-receipt.json"
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
        fail("FAIL_M25_PYTHON_BRIDGE_PROFILE")
    if os.environ.get("HEGEL_PURPOSE_ID") != "2":
        fail("FAIL_M25_PYTHON_BRIDGE_PURPOSE")
    if os.getuid() != 65534 or os.getgid() != 65534:
        fail("FAIL_M25_PYTHON_BRIDGE_IDENTITY")


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


def run_quiet(command: list[str], *, stdin: bytes | None = None) -> bytes:
    completed = subprocess.run(
        command,
        input=stdin,
        stdin=subprocess.DEVNULL if stdin is None else None,
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
            fail("FAIL_M25_PYTHON_BRIDGE_RECOVERY_KEY_ABSENT")
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


def load_wire():
    package = sys.modules.get("hegel_machine")
    if package is None:
        package = types.ModuleType("hegel_machine")
        package.__path__ = ["/input/src/hegel_machine"]
        package.__package__ = "hegel_machine"
        sys.modules["hegel_machine"] = package
    return importlib.import_module("hegel_machine.phase3_m25_wire_v1")


def load_bridge_replay():
    load_wire()
    return importlib.import_module(
        "hegel_machine.phase3_m25_bridge_full_dag_replay_v1"
    )


def _bridge_verifier_directory() -> Path:
    nonce = os.environ.get("HEGEL_OPERATION_NONCE", "")
    if re.fullmatch(r"[0-9a-f]{32}", nonce) is None:
        raise WorkerFailure
    return Path("/tmp") / f"hegel-m25-bridge-verifier-p2-{nonce}"


def _openssl_sign_internal(preimage: bytes, private_directory: Path) -> bytes:
    if (
        not PRIVATE_KEY.is_file()
        or PRIVATE_KEY.is_symlink()
        or SIGNATURE.exists()
        or len(preimage) > 4096
    ):
        raise WorkerFailure
    signing_input = private_directory / "signing-preimage.bin"
    exclusive_bytes(signing_input, preimage, 0o600)
    try:
        signature = run_quiet(
            [
                "/usr/bin/openssl",
                "pkeyutl",
                "-sign",
                "-rawin",
                "-inkey",
                str(PRIVATE_KEY),
                "-in",
                str(signing_input),
            ]
        )
    finally:
        try:
            signing_input.unlink()
        except FileNotFoundError:
            pass
    if len(signature) != 64:
        raise WorkerFailure
    exclusive_bytes(SIGNATURE, signature, 0o644)
    return signature


def bridge_replay_sign() -> None:
    require_profile()
    if not PRIVATE_KEY.is_file() or SIGNATURE.exists():
        raise WorkerFailure
    wire = load_wire()
    replay = load_bridge_replay()
    payload = BRIDGE_DAG_PACKAGE.read_bytes()
    if not payload or len(payload) > MAX_PUBLIC_REQUEST_BYTES:
        raise WorkerFailure
    verifier_directory = _bridge_verifier_directory()
    os.mkdir(verifier_directory, 0o700)
    try:
        verifier = replay.make_openssl_ed25519_verifier_v1(verifier_directory)
        result = replay.replay_bridge_dag_package_v1(
            payload,
            allow_authoritative=True,
            signature_verifier=verifier,
        )
        if (
            result.purpose_id != 2
            or not result.authoritative
            or not result.purpose1_signature_verified
            or not result.eligible_to_sign_bridge_statement
            or result.split_membership_recomputed
        ):
            raise WorkerFailure
        preimage = wire.bridge_attestation_signature_preimage_v1(
            result.bridge_statement_root, 2, 0
        )
        signature = _openssl_sign_internal(preimage, verifier_directory)
        exclusive_bytes(
            BRIDGE_REPLAY_RECEIPT,
            replay.build_bridge_actor_replay_receipt_v1(
                result, implementation="python-full-dag-replay-v1"
            ),
            0o644,
        )
    finally:
        try:
            verifier_directory.rmdir()
        except FileNotFoundError:
            pass


def main() -> int:
    if len(sys.argv) != 2:
        fail("FAIL_M25_PYTHON_BRIDGE_ARGUMENTS")
    operation = sys.argv[1]
    try:
        operation_probe.qualify_operation_v1(operation)
        if operation == "qualify-only":
            pass
        elif operation == "keygen":
            keygen_or_resume(recovery=False)
        elif operation == "keygen-resume":
            keygen_or_resume(recovery=True)
        elif operation == "bridge-replay-sign-python":
            bridge_replay_sign()
        else:
            raise WorkerFailure
        return 0
    except (Exception, SystemExit):
        fail("FAIL_M25_PYTHON_BRIDGE_OPERATION")
    return 70


if __name__ == "__main__":
    raise SystemExit(main())
