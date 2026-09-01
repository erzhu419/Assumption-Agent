from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine import phase3_q6_completion_replay_v1 as state  # noqa: E402


SPEC = importlib.util.spec_from_file_location(
    "phase3_q6_controller_v1",
    ROOT / "tools/phase3_q6_formal_quotient_certificate_issuance_v1.py",
)
assert SPEC is not None and SPEC.loader is not None
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


def _contract(tmp_path: Path) -> Path:
    value = json.loads((ROOT / "config/phase3_q6_formal_quotient_certificate_issuance_actual_v1.json").read_text())
    fixed = value["fixed_paths"]
    run = tmp_path / "run"
    ledger = tmp_path / "ledger"
    fixed.update({
        "campaign_lock_path": str(tmp_path / "campaign.lock"),
        "run_root": str(run),
        "ledger_root": str(ledger),
        "capsule_path": str(run / "capsule.json"),
        "authorization_path": str(ledger / "authorization.json"),
        "consumption_path": str(ledger / "consumption.json"),
        "pending_path": str(run / "latest.pending.json"),
        "pending_history_path": str(run / "pending-history.jsonl"),
        "checkpoint_path": str(ledger / "checkpoint.json"),
        "terminal_path": str(ledger / "terminal.json"),
        "signature_preimage_path": str(run / "signature-preimage.cbor"),
        "signing_message_path": str(run / "signing-message.bin"),
        "envelope_path": str(tmp_path / "artifact" / "envelope.cbor"),
        "artifact_path": str(tmp_path / "artifact" / "issuance.cbor"),
        "trajectory_path": str(run / "trajectory.jsonl"),
    })
    for ordinal in state.SIGNER_ORDINALS:
        fixed.update({
            f"signer_{ordinal}_intent_path": str(ledger / f"signer-{ordinal}.intent.json"),
            f"signer_{ordinal}_verification_intent_path": str(ledger / f"signer-{ordinal}.verification-intent.json"),
            f"signer_{ordinal}_signature_path": str(run / f"signer-{ordinal}.signature.bin"),
            f"signer_{ordinal}_signature_temp_path": str(run / f"signer-{ordinal}.signature.bin.q6-tmp"),
            f"signer_{ordinal}_worker_receipt_path": str(ledger / f"signer-{ordinal}.worker.json"),
            f"signer_{ordinal}_verified_checkpoint_path": str(ledger / f"signer-{ordinal}.verified.json"),
            f"signer_{ordinal}_custody_root": str(tmp_path / f"custody-{ordinal}"),
        })
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="ascii")
    return path


def _arg(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _fake_worker(command: list[str]) -> subprocess.CompletedProcess[bytes]:
    ordinal = int(_arg(command, "--signer-ordinal"))
    role = _arg(command, "--signer-role")
    public_key = _arg(command, "--expected-public-key-hex")
    key_id = _arg(command, "--expected-key-id-hex")
    message_hash = _arg(command, "--expected-message-raw-sha256")
    message_bytes = int(_arg(command, "--expected-message-byte-count"))
    uid = int(_arg(command, "--expected-uid"))
    gid = int(_arg(command, "--expected-gid"))
    if "--startup-smoke" in command:
        value = {
            "expected_gid": gid,
            "expected_key_id_hex": key_id,
            "expected_message_byte_count": message_bytes,
            "expected_message_raw_sha256": message_hash,
            "expected_public_key_hex": public_key,
            "expected_uid": uid,
            "key_generation_count": 0,
            "key_import_count": 0,
            "openssl_path": "/usr/bin/openssl",
            "openssl_popen_count": 0,
            "private_key_access_count": 0,
            "purpose_id": 5,
            "schema": state.WORKER_SMOKE_SCHEMA,
            "signature_count": 0,
            "signer_ordinal": ordinal,
            "signer_role": role,
            "status": state.WORKER_SMOKE_STATUS,
        }
    else:
        signature = bytes([ordinal]) * 64
        signature_path = Path(_arg(command, "--signature-path"))
        receipt_path = Path(_arg(command, "--receipt-path"))
        signature_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        signature_path.write_bytes(signature)
        os.chmod(signature_path, 0o644)
        value = {
            "key_epoch": 1,
            "key_generation_count": 0,
            "key_id_hex": key_id,
            "key_import_count": 0,
            "message_byte_count": message_bytes,
            "message_raw_sha256": message_hash,
            "openssl_path": "/usr/bin/openssl",
            "openssl_popen_count": 1,
            "private_key_copy_count": 0,
            "private_key_exported": False,
            "private_key_openssl_use_count": 1,
            "public_key_hex": public_key,
            "purpose_id": 5,
            "schema": state.WORKER_RECEIPT_SCHEMA,
            "signature_byte_count": 64,
            "signature_count": 1,
            "signature_raw_sha256": state.raw_sha256(signature),
            "signature_verification_count": 0,
            "signer_ordinal": ordinal,
            "signer_role": role,
            "status": state.WORKER_SUCCESS_STATUS,
        }
        receipt_path.write_bytes(state.canonical_json_bytes(value))
        os.chmod(receipt_path, 0o644)
    return subprocess.CompletedProcess(command, 0, state.canonical_json_bytes(value), b"")


def _prepare_and_authorize(contract_path: Path, monkeypatch) -> dict:
    monkeypatch.setattr(controller, "_invoke_worker", _fake_worker)
    capsule = controller.prepare_capsule(
        contract_path, "c" * 40, run_id="d" * 32,
        created_at_unix_seconds=1_800_000_000,
        certificate_created_at_unix_seconds=1_800_000_001,
    )
    controller.record_authorization(contract_path, capsule["capsule_digest"])
    return capsule


def test_postconsumption_pending_history_is_append_only_and_cli_reports_clean_prefix(tmp_path, monkeypatch, capsys):
    contract_path = _contract(tmp_path)
    capsule = _prepare_and_authorize(contract_path, monkeypatch)
    for expected_occurrence in (1, 2):
        with pytest.raises(controller.ControllerPending) as caught:
            controller.run_v1(contract_path, capsule["capsule_digest"])
        pending = caught.value.pending_record
        assert pending is not None
        assert pending["status"] == state.PENDING_CLEAN_CHECKPOINT
        assert pending["authorization_consumption_count"] == 1
        assert pending["completed_prefix"] == []
        history = Path(json.loads(contract_path.read_text())["fixed_paths"]["pending_history_path"])
        assert len(history.read_bytes().splitlines()) == expected_occurrence

    exit_code = controller.main([
        "--contract", str(contract_path), "run",
        "--capsule-digest", capsule["capsule_digest"],
    ])
    output = json.loads(capsys.readouterr().out)
    assert exit_code == controller.PENDING_EXIT_CODE
    assert output["status"] == state.PENDING_CLEAN_CHECKPOINT
    assert output["logical_campaign_count"] == 1
    assert len(Path(json.loads(contract_path.read_text())["fixed_paths"]["pending_history_path"]).read_bytes().splitlines()) == 3


def test_three_signers_verify_exactly_once_and_terminal_replay_does_not_repeat(tmp_path, monkeypatch):
    linux_tmp = Path(tempfile.mkdtemp(prefix="q6-controller-test-", dir="/tmp"))
    contract_path = _contract(linux_tmp)
    value = json.loads(contract_path.read_text())
    for ordinal in state.SIGNER_ORDINALS:
        custody = Path(value["fixed_paths"][f"signer_{ordinal}_custody_root"])
        custody.mkdir(mode=0o700)
        os.chmod(custody, 0o700)
        private = custody / "ed25519-private.pem"
        private.write_bytes(b"mock-private-metadata-only\n")
        os.chmod(private, 0o600)
    capsule = _prepare_and_authorize(contract_path, monkeypatch)
    calls: list[bytes] = []

    def verify(public_key: bytes, signature: bytes, message: bytes) -> None:
        calls.append(signature)

    monkeypatch.setattr(controller.wire, "verify_ed25519_v1", verify)
    terminal = controller.run_v1(contract_path, capsule["capsule_digest"])
    assert terminal["status"] == controller.wire.Q6_SUCCESS_STATUS
    assert calls == [b"\x01" * 64, b"\x02" * 64, b"\x03" * 64]
    terminal_again = controller.run_v1(contract_path, capsule["capsule_digest"])
    assert terminal_again == terminal
    assert len(calls) == 3
