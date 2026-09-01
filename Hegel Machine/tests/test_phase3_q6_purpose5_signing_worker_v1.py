from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterator

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = REPOSITORY_ROOT / "tools/phase3_q6_purpose5_signing_worker_v1.py"
SPEC = importlib.util.spec_from_file_location("phase3_q6_signing_worker", WORKER_PATH)
assert SPEC is not None and SPEC.loader is not None
worker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worker
SPEC.loader.exec_module(worker)

PUBLIC_KEY = bytes(range(32))
KEY_ID = hashlib.sha256(PUBLIC_KEY).digest()[:16]
MESSAGE = b"HEGEL/Q6/PURPOSE5/FORMAL-ISSUANCE-MESSAGE/V1\x00sealed-body"
MESSAGE_SHA256 = "sha256:" + hashlib.sha256(MESSAGE).hexdigest()
SIGNATURE = bytes(range(64))


@pytest.fixture
def native_tmp_path() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="hegel-q6-worker-", dir="/tmp") as raw:
        yield Path(raw)


def _common_arguments(
    ordinal: int = 1,
    role: str = "FINAL_CERTIFICATE_CUSTODIAN",
    *,
    key_id: bytes = KEY_ID,
    message_sha256: str = MESSAGE_SHA256,
) -> list[str]:
    return [
        "--signer-ordinal",
        str(ordinal),
        "--signer-role",
        role,
        "--expected-uid",
        str(os.getuid()),
        "--expected-gid",
        str(os.getgid()),
        "--expected-public-key-hex",
        PUBLIC_KEY.hex(),
        "--expected-key-id-hex",
        key_id.hex(),
        "--expected-message-raw-sha256",
        message_sha256,
        "--expected-message-byte-count",
        str(len(MESSAGE)),
    ]


def _smoke_arguments(**kwargs: object) -> list[str]:
    return ["--startup-smoke", *_common_arguments(**kwargs)]


def _signing_layout(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    custody = tmp_path / "custody"
    signature_parent = tmp_path / "signatures"
    receipt_parent = tmp_path / "receipts"
    custody.mkdir(mode=0o700)
    custody.chmod(0o700)
    signature_parent.mkdir(mode=0o700)
    receipt_parent.mkdir(mode=0o700)
    private_key = custody / worker.PRIVATE_KEY_NAME
    private_key.write_bytes(b"not-a-real-private-key; metadata-only-test")
    private_key.chmod(0o600)
    message = tmp_path / "message.bin"
    message.write_bytes(MESSAGE)
    message.chmod(0o644)
    return {
        "custody": custody,
        "private_key": private_key,
        "message": message,
        "signature": signature_parent / "signature.bin",
        "receipt": receipt_parent / "receipt.json",
    }


def _sign_arguments(layout: dict[str, Path], **kwargs: object) -> list[str]:
    return [
        "sign",
        *_common_arguments(**kwargs),
        "--custody-root",
        str(layout["custody"]),
        "--message-path",
        str(layout["message"]),
        "--signature-path",
        str(layout["signature"]),
        "--receipt-path",
        str(layout["receipt"]),
    ]


def test_startup_smoke_is_input_read_free_and_never_starts_openssl(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("startup smoke touched the signing path")

    monkeypatch.setattr(worker.subprocess, "run", forbidden)
    monkeypatch.setattr(worker, "_read_regular_v1", forbidden)
    monkeypatch.setattr(worker, "_require_private_key_metadata_v1", forbidden)
    monkeypatch.setattr(worker, "_require_output_paths_v1", forbidden)
    monkeypatch.setattr(worker, "sign_v1", forbidden)

    assert worker.main(_smoke_arguments()) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    receipt = json.loads(captured.out)
    assert captured.out.encode("ascii") == worker._canonical_json(receipt)
    assert receipt == {
        "expected_gid": os.getgid(),
        "expected_key_id_hex": KEY_ID.hex(),
        "expected_message_byte_count": len(MESSAGE),
        "expected_message_raw_sha256": MESSAGE_SHA256,
        "expected_public_key_hex": PUBLIC_KEY.hex(),
        "expected_uid": os.getuid(),
        "key_generation_count": 0,
        "key_import_count": 0,
        "openssl_path": "/usr/bin/openssl",
        "openssl_popen_count": 0,
        "private_key_access_count": 0,
        "purpose_id": 5,
        "schema": "hegel-phase3-q6-purpose5-signing-startup-smoke-receipt/1",
        "signature_count": 0,
        "signer_ordinal": 1,
        "signer_role": "FINAL_CERTIFICATE_CUSTODIAN",
        "status": "Q6_PURPOSE5_SIGNING_STARTUP_SMOKE_PASS",
    }


def test_startup_smoke_rejects_mismatched_public_key_id_without_subprocess(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("startup smoke called subprocess")
        ),
    )
    assert worker.main(_smoke_arguments(key_id=b"\x00" * 16)) == 70
    assert capsys.readouterr().err == worker.FAIL_PUBLIC_BINDING + "\n"


@pytest.mark.parametrize(
    ("ordinal", "role"),
    (
        (0, "FINAL_CERTIFICATE_CUSTODIAN"),
        (1, "FINAL_CERTIFICATE_PYTHON_REPLAY_ATTESTER"),
        (2, "FINAL_CERTIFICATE_CUSTODIAN"),
        (3, "FINAL_CERTIFICATE_PYTHON_REPLAY_ATTESTER"),
        (4, "FINAL_CERTIFICATE_RUST_REPLAY_ATTESTER"),
    ),
)
def test_signer_ordinal_role_mapping_is_exact(ordinal: int, role: str) -> None:
    with pytest.raises(worker.WorkerFailure) as caught:
        worker.parse_arguments_v1(_smoke_arguments(ordinal=ordinal, role=role))
    assert caught.value.code == worker.FAIL_ARGUMENT_SCHEMA


def test_argument_shapes_and_smoke_path_boundary(tmp_path: Path) -> None:
    malformed = _smoke_arguments()
    malformed[malformed.index("--expected-public-key-hex") + 1] = "AA" * 32
    with pytest.raises(worker.WorkerFailure) as public_error:
        worker.parse_arguments_v1(malformed)
    assert public_error.value.code == worker.FAIL_ARGUMENT_SCHEMA

    with_path = [*_smoke_arguments(), "--custody-root", str(tmp_path / "custody")]
    with pytest.raises(worker.WorkerFailure) as path_error:
        worker.parse_arguments_v1(with_path)
    assert path_error.value.code == worker.FAIL_ARGUMENT_SCHEMA

    missing_actual_paths = ["sign", *_common_arguments()]
    with pytest.raises(worker.WorkerFailure) as actual_error:
        worker.parse_arguments_v1(missing_actual_paths)
    assert actual_error.value.code == worker.FAIL_ARGUMENT_SCHEMA


def test_sign_uses_one_private_key_openssl_call_and_publishes_exact_outputs(
    native_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _signing_layout(native_tmp_path)
    private_before = layout["private_key"].read_bytes()
    calls: list[tuple[list[str], dict[str, object]]] = []

    def simulated_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, SIGNATURE, b"")

    monkeypatch.setattr(worker, "_require_openssl_v1", lambda: worker.OPENSSL)
    monkeypatch.setattr(worker.subprocess, "run", simulated_run)

    assert worker.main(_sign_arguments(layout)) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    receipt = json.loads(captured.out)
    assert len(calls) == 1
    command, options = calls[0]
    assert command == [
        "/usr/bin/openssl",
        "pkeyutl",
        "-sign",
        "-rawin",
        "-inkey",
        str(layout["private_key"]),
        "-in",
        str(layout["message"]),
    ]
    assert options["stdin"] is subprocess.DEVNULL
    assert options["stdout"] is subprocess.PIPE
    assert options["stderr"] is subprocess.DEVNULL
    assert options["check"] is False
    assert options["timeout"] == worker.OPENSSL_TIMEOUT_SECONDS
    assert "-verify" not in command
    assert command.count(str(layout["private_key"])) == 1

    assert layout["signature"].read_bytes() == SIGNATURE
    assert stat.S_IMODE(layout["signature"].lstat().st_mode) == 0o644
    assert not layout["signature"].with_name(
        layout["signature"].name + ".q6-tmp"
    ).exists()
    assert layout["receipt"].read_bytes() == worker._canonical_json(receipt)
    assert stat.S_IMODE(layout["receipt"].lstat().st_mode) == 0o644
    assert layout["private_key"].read_bytes() == private_before
    assert stat.S_IMODE(layout["private_key"].lstat().st_mode) == 0o600
    assert layout["message"].read_bytes() == MESSAGE

    assert receipt == {
        "key_epoch": 1,
        "key_generation_count": 0,
        "key_id_hex": KEY_ID.hex(),
        "key_import_count": 0,
        "message_byte_count": len(MESSAGE),
        "message_raw_sha256": MESSAGE_SHA256,
        "openssl_path": "/usr/bin/openssl",
        "openssl_popen_count": 1,
        "private_key_copy_count": 0,
        "private_key_exported": False,
        "private_key_openssl_use_count": 1,
        "public_key_hex": PUBLIC_KEY.hex(),
        "purpose_id": 5,
        "schema": "hegel-phase3-q6-purpose5-signing-public-receipt/1",
        "signature_byte_count": 64,
        "signature_count": 1,
        "signature_raw_sha256": "sha256:" + hashlib.sha256(SIGNATURE).hexdigest(),
        "signature_verification_count": 0,
        "signer_ordinal": 1,
        "signer_role": "FINAL_CERTIFICATE_CUSTODIAN",
        "status": "PURPOSE5_SIGNATURE_COMPLETE_PENDING_PRODUCER_FREE_VERIFICATION",
    }


def test_sign_failure_is_one_shot_and_publishes_nothing(
    native_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _signing_layout(native_tmp_path)
    calls = 0

    def failed_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        nonlocal calls
        del kwargs
        calls += 1
        return subprocess.CompletedProcess(command, 1, b"", b"failure")

    monkeypatch.setattr(worker, "_require_openssl_v1", lambda: worker.OPENSSL)
    monkeypatch.setattr(worker.subprocess, "run", failed_run)
    assert worker.main(_sign_arguments(layout)) == 70
    assert capsys.readouterr().err == worker.FAIL_OPENSSL_SIGN + "\n"
    assert calls == 1
    assert not layout["signature"].exists()
    assert not layout["receipt"].exists()


def test_non_64_byte_signature_is_rejected_before_publication(
    native_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _signing_layout(native_tmp_path)

    def short_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        return subprocess.CompletedProcess(command, 0, bytes(63), b"")

    monkeypatch.setattr(worker, "_require_openssl_v1", lambda: worker.OPENSSL)
    monkeypatch.setattr(worker.subprocess, "run", short_run)
    assert worker.main(_sign_arguments(layout)) == 70
    assert capsys.readouterr().err == worker.FAIL_SIGNATURE_SHAPE + "\n"
    assert not layout["signature"].exists()
    assert not layout["receipt"].exists()


def test_bound_message_and_fresh_output_are_checked_before_private_load(
    native_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _signing_layout(native_tmp_path)

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("invalid admission reached private/sign operation")

    monkeypatch.setattr(worker, "_require_openssl_v1", lambda: worker.OPENSSL)
    monkeypatch.setattr(worker, "_require_private_key_metadata_v1", forbidden)
    monkeypatch.setattr(worker.subprocess, "run", forbidden)
    assert worker.main(
        _sign_arguments(layout, message_sha256="sha256:" + "0" * 64)
    ) == 70
    assert capsys.readouterr().err == worker.FAIL_MESSAGE + "\n"

    layout = _signing_layout(native_tmp_path / "second")
    layout["signature"].write_bytes(b"already exists")
    assert worker.main(_sign_arguments(layout)) == 70
    assert capsys.readouterr().err == worker.FAIL_OUTPUT_NOT_FRESH + "\n"


def test_custody_metadata_is_checked_without_reading_private_key(
    native_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _signing_layout(native_tmp_path)
    layout["private_key"].chmod(0o644)
    monkeypatch.setattr(worker, "_require_openssl_v1", lambda: worker.OPENSSL)
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid custody reached OpenSSL")
        ),
    )
    assert worker.main(_sign_arguments(layout)) == 70
    assert capsys.readouterr().err == worker.FAIL_PRIVATE_KEY + "\n"


def test_worker_source_has_no_keygen_verify_target_or_network_operation() -> None:
    source = WORKER_PATH.read_text(encoding="utf-8")
    assert "genpkey" not in source
    assert '"-verify"' not in source
    assert "socket" not in source
    assert "urllib" not in source
    assert "requests" not in source
    assert "target_open" not in source
