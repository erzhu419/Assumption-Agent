from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = REPOSITORY_ROOT / "tools/phase3_q5a_purpose5_keygen_worker_v1.py"
SPEC = importlib.util.spec_from_file_location("phase3_q5a_keygen_worker", WORKER_PATH)
assert SPEC is not None and SPEC.loader is not None
worker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worker
SPEC.loader.exec_module(worker)


def _smoke_arguments(
    ordinal: int = 1, role: str = "FINAL_CERTIFICATE_CUSTODIAN"
) -> list[str]:
    return [
        "--startup-smoke",
        "--signer-ordinal",
        str(ordinal),
        "--signer-role",
        role,
        "--expected-uid",
        str(os.getuid()),
        "--expected-gid",
        str(os.getgid()),
    ]


def test_startup_smoke_is_write_free_and_never_starts_openssl(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden_popen(*args: object, **kwargs: object) -> object:
        raise AssertionError("startup smoke called subprocess")

    monkeypatch.setattr(worker.subprocess, "run", forbidden_popen)
    monkeypatch.setattr(worker, "_create_fresh_roots_v1", forbidden_popen)
    monkeypatch.setattr(worker, "generate_v1", forbidden_popen)
    assert worker.main(_smoke_arguments()) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    receipt = json.loads(captured.out)
    assert captured.out.encode("ascii") == worker._canonical_json(receipt)
    assert receipt == {
        "expected_gid": os.getgid(),
        "expected_uid": os.getuid(),
        "key_generation_count": 0,
        "openssl_path": "/usr/bin/openssl",
        "openssl_popen_count": 0,
        "public_key_derivation_count": 0,
        "purpose_id": 5,
        "schema": "hegel-phase3-q5a-purpose5-keygen-startup-smoke-receipt/1",
        "signature_count": 0,
        "signer_ordinal": 1,
        "signer_role": "FINAL_CERTIFICATE_CUSTODIAN",
        "status": "Q5A_PURPOSE5_KEYGEN_STARTUP_SMOKE_PASS",
    }


def test_startup_smoke_rejects_identity_and_root_parameters(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr(worker.subprocess, "run", lambda *args, **kwargs: None)
    wrong_identity = _smoke_arguments()
    wrong_identity[wrong_identity.index("--expected-uid") + 1] = str(os.getuid() + 1)
    assert worker.main(wrong_identity) == 70
    assert capsys.readouterr().err == worker.FAIL_IDENTITY + "\n"

    with_root = [*_smoke_arguments(), "--state-root", str(tmp_path / "state")]
    assert worker.main(with_root) == 70
    assert capsys.readouterr().err == worker.FAIL_ARGUMENT_SCHEMA + "\n"


def test_spki_parser_and_public_receipt_are_byte_exact() -> None:
    public_key = bytes(range(32))
    der = worker.ED25519_SPKI_PREFIX + public_key
    decoded, key_id = worker.parse_ed25519_spki_der_v1(der)
    assert decoded == public_key
    assert key_id == hashlib.sha256(public_key).digest()[:16]

    receipt = worker.build_public_receipt_v1(
        signer_ordinal=2,
        signer_role="FINAL_CERTIFICATE_PYTHON_REPLAY_ATTESTER",
        public_key=public_key,
    )
    assert receipt == {
        "key_epoch": 1,
        "key_generation_count": 1,
        "key_id_hex": hashlib.sha256(public_key).hexdigest()[:32],
        "openssl_popen_count": 2,
        "private_key_exported": False,
        "public_key_derivation_count": 1,
        "public_key_hex": public_key.hex(),
        "purpose_id": 5,
        "schema": "hegel-phase3-q5a-purpose5-keygen-public-receipt/1",
        "signature_count": 0,
        "signer_ordinal": 2,
        "signer_role": "FINAL_CERTIFICATE_PYTHON_REPLAY_ATTESTER",
        "status": "PURPOSE5_KEYGEN_COMPLETE_PRIVATE_KEY_RETAINED",
    }
    assert worker._canonical_json(receipt).endswith(b"\n")
    assert json.loads(worker._canonical_json(receipt)) == receipt


@pytest.mark.parametrize(
    "payload",
    (
        b"",
        bytes(44),
        worker.ED25519_SPKI_PREFIX + bytes(31),
        b"X" * len(worker.ED25519_SPKI_PREFIX) + bytes(32),
    ),
)
def test_spki_parser_rejects_every_non_exact_shape(payload: bytes) -> None:
    with pytest.raises(worker.WorkerFailure) as caught:
        worker.parse_ed25519_spki_der_v1(payload)
    assert caught.value.code == worker.FAIL_PUBLIC_DER


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
        worker.parse_arguments_v1(_smoke_arguments(ordinal, role))
    assert caught.value.code == worker.FAIL_ARGUMENT_SCHEMA


def test_generate_arguments_require_canonical_distinct_fresh_roots() -> None:
    base = [
        "generate",
        "--signer-ordinal",
        "3",
        "--signer-role",
        "FINAL_CERTIFICATE_RUST_REPLAY_ATTESTER",
        "--expected-uid",
        str(os.getuid()),
        "--expected-gid",
        str(os.getgid()),
    ]
    with tempfile.TemporaryDirectory(prefix="hegel-q5a-roots-", dir="/tmp") as raw:
        temporary_root = Path(raw)
        roots = (
            temporary_root / "state",
            temporary_root / "receipt",
            temporary_root / "output",
        )
        arguments = worker.parse_arguments_v1(
            [
                *base,
                "--state-root",
                str(roots[0]),
                "--receipt-root",
                str(roots[1]),
                "--output-root",
                str(roots[2]),
            ]
        )
        assert arguments.operation == "generate"
        assert worker.validate_generate_roots_v1(*roots) == roots

        with pytest.raises(worker.WorkerFailure) as overlap:
            worker.validate_generate_roots_v1(
                roots[0], roots[0] / "receipt", roots[2]
            )
        assert overlap.value.code == worker.FAIL_ROOT_OVERLAP

        roots[0].mkdir()
        with pytest.raises(worker.WorkerFailure) as existing:
            worker.validate_generate_roots_v1(*roots)
        assert existing.value.code == worker.FAIL_ROOT_NOT_FRESH

        noncanonical = str(
            temporary_root / "missing" / ".." / "state-two"
        )
        with pytest.raises(worker.WorkerFailure) as path_error:
            worker.parse_arguments_v1(
                [
                    *base,
                    "--state-root",
                    noncanonical,
                    "--receipt-root",
                    str(temporary_root / "receipt-two"),
                    "--output-root",
                    str(temporary_root / "output-two"),
                ]
            )
        assert path_error.value.code == worker.FAIL_ROOT_PATH


def test_publication_helpers_create_only_public_modes() -> None:
    with tempfile.TemporaryDirectory(prefix="hegel-q5a-publish-", dir="/tmp") as raw:
        temporary_root = Path(raw)
        roots = (
            temporary_root / "state",
            temporary_root / "receipt",
            temporary_root / "output",
        )
        worker._create_fresh_roots_v1(roots)
        assert all(stat.S_IMODE(path.lstat().st_mode) == 0o700 for path in roots)

        public = roots[2] / worker.PUBLIC_DER_NAME
        receipt = roots[1] / worker.PUBLIC_RECEIPT_NAME
        worker._exclusive_bytes(
            public, worker.ED25519_SPKI_PREFIX + bytes(32), 0o644
        )
        worker._exclusive_bytes(
            receipt, worker._canonical_json({"public": True}), 0o644
        )
        assert stat.S_IMODE(public.lstat().st_mode) == 0o644
        assert stat.S_IMODE(receipt.lstat().st_mode) == 0o644
        assert list(roots[0].iterdir()) == []


def test_worker_source_has_no_signing_or_network_operation() -> None:
    source = WORKER_PATH.read_text(encoding="utf-8")
    assert "pkeyutl" not in source
    assert "socket" not in source
    assert '"-sign"' not in source
