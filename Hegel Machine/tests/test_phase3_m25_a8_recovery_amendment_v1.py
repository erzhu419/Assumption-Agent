from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from hegel_machine import phase3_m25_a8_recovery_amendment_v1 as amendment
from hegel_machine.phase3_m25_formal_container_executor_v1 import (
    FormalContainerExecutorError,
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["/usr/bin/git", *args],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": "/nonexistent",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_AUTHOR_NAME": "R1 Test",
            "GIT_AUTHOR_EMAIL": "r1@example.invalid",
            "GIT_COMMITTER_NAME": "R1 Test",
            "GIT_COMMITTER_EMAIL": "r1@example.invalid",
        },
    )
    return completed.stdout.decode("ascii").strip()


def test_source_preflight_requires_clean_direct_child_and_exact_blob_set(
    tmp_path: Path, monkeypatch,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init", "-q")
    source = repository / "source.py"
    source.write_bytes(b"old\n")
    _git(repository, "add", "source.py")
    _git(repository, "commit", "-q", "-m", "base")
    base = _git(repository, "rev-parse", "HEAD")
    monkeypatch.setattr(amendment, "A8_BASIS_COMMIT", base)

    source.write_bytes(b"new\n")
    manifest_path = repository / "manifest.json"
    manifest = {
        "schema": amendment.MANIFEST_SCHEMA,
        "source_commit_selector": "HEAD",
        "sole_parent_commit": base,
        "formal_repository_commit": base,
        "fixed_run_id_hex": amendment.FIXED_RUN_ID_HEX,
        "fixed_ledger_id_hex": amendment.FIXED_LEDGER_ID_HEX,
        "exact_changed_paths": [
            {"status": "A", "path": "manifest.json"},
            {"status": "M", "path": "source.py"},
        ],
        "source_bindings": [{
            "path": "source.py",
            "a8_sha256_or_null": hashlib.sha256(b"old\n").hexdigest(),
            "r1_sha256": hashlib.sha256(b"new\n").hexdigest(),
        }],
        "complete_seed_resume_only": True,
        "formal_identity_entropy_draw_count": 0,
        "ephemeral_container_nonce_allowed": True,
        "ordinary_execute_allowed": False,
        "ordinary_recovery_cross_basis_allowed": False,
        "fixed_audit_directory": amendment.FIXED_AUDIT_DIRECTORY.as_posix(),
    }
    manifest_path.write_bytes(_canonical(manifest))
    _git(repository, "add", "source.py", "manifest.json")
    _git(repository, "commit", "-q", "-m", "r1")

    report = amendment.inspect_r1_source_preflight_v1(
        repository_root=repository, manifest_path=manifest_path
    )
    assert report["sole_parent_commit"] == base
    assert report["formal_repository_commit"] == base
    assert report["formal_identity_entropy_draw_count"] == 0
    source.write_bytes(b"dirty\n")
    with pytest.raises(amendment.A8RecoveryAmendmentError):
        amendment.inspect_r1_source_preflight_v1(
            repository_root=repository, manifest_path=manifest_path
        )


def test_complete_seed_preflight_never_opens_or_hashes_raw_seed(
    tmp_path: Path, monkeypatch,
) -> None:
    custody = tmp_path / "custody"
    custody.mkdir(mode=0o700)
    for name, payload in (
        ("split_seed_generation.intent", b"intent\n"),
        ("split_master_seed.bin", b"s" * 32),
        ("split_seed_generation.complete", b"complete\n"),
    ):
        path = custody / name
        path.write_bytes(payload)
        path.chmod(0o600)
    if (custody / "split_master_seed.bin").stat().st_mode & 0o777 != 0o600:
        pytest.skip("test temporary filesystem does not enforce POSIX mode 0600")
    original_open = amendment.os.open

    def guarded_open(path, *args, **kwargs):
        if Path(path).name == "split_master_seed.bin":
            raise AssertionError("raw seed inode was opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(amendment.os, "open", guarded_open)
    rows = amendment._require_complete_seed_metadata(custody)
    seed = next(row for row in rows if row["name"] == "split_master_seed.bin")
    assert seed["size_bytes"] == 32
    assert seed["raw_bytes_read"] is False
    assert seed["sha256_computed"] is False
    assert "sha256" not in seed
    assert all("st_dev" in row and "st_ino" in row for row in rows)


def test_authorization_is_a_separate_o_excl_owner_action(
    tmp_path: Path, monkeypatch,
) -> None:
    audit = tmp_path / "audit"
    audit.mkdir(mode=0o700)
    audit.chmod(0o700)
    if audit.stat().st_mode & 0o777 != 0o700:
        pytest.skip("test temporary filesystem does not enforce POSIX mode 0700")
    preflight = {
        "schema": f"{amendment.AUDIT_SCHEMA_PREFIX}-preflight/1",
        "amendment_commit": "12" * 20,
        "sole_parent_commit": amendment.A8_BASIS_COMMIT,
        "formal_repository_commit": amendment.A8_BASIS_COMMIT,
        "run_id_hex": amendment.FIXED_RUN_ID_HEX,
        "ledger_id_hex": amendment.FIXED_LEDGER_ID_HEX,
        "manifest_sha256": "34" * 32,
        "source_bindings": [],
        "repository_clean": True,
        "exact_changed_paths_verified": True,
        "formal_identity_entropy_draw_count": 0,
        "ephemeral_container_nonce_allowed": True,
        "m3_start_allowed": False,
    }
    monkeypatch.setattr(
        amendment, "inspect_r1_source_preflight_v1", lambda **_kwargs: preflight
    )
    monkeypatch.setattr(amendment, "FIXED_AUDIT_DIRECTORY", audit)
    amendment.prepare_fixed_a8_r1_authorization_v1(audit_directory=audit)
    assert not (audit / "authorization.json").exists()
    with pytest.raises(amendment.A8RecoveryAmendmentError):
        amendment.write_fixed_a8_r1_owner_authorization_v1(
            audit_directory=audit, owner_confirmation="wrong"
        )
    authorization = amendment.write_fixed_a8_r1_owner_authorization_v1(
        audit_directory=audit,
        owner_confirmation="AUTHORIZE_A8_R1_COMPLETE_ONLY_REAL_PENDING_RESUME",
    )
    assert authorization["owner_authorized_fixed_transaction_only"] is True
    assert (audit / "authorization.json").stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        amendment.write_fixed_a8_r1_owner_authorization_v1(
            audit_directory=audit,
            owner_confirmation="AUTHORIZE_A8_R1_COMPLETE_ONLY_REAL_PENDING_RESUME",
        )


def test_recovery_bridge_binding_accepts_explicit_a8_artifacts_without_default_path(
    tmp_path: Path, monkeypatch,
) -> None:
    custody = tmp_path / "custody"
    custody.mkdir(mode=0o700)
    binary = tmp_path / "bridge"
    binary.write_bytes(b"qualified-a8-bridge")
    binary.chmod(0o755)
    report_path = tmp_path / "report.json"
    source_payload = b"a8-source"
    report = {
        "source": {
            "bindings": {
                "Hegel Machine/source.rs": "sha256:"
                + hashlib.sha256(source_payload).hexdigest()
            }
        },
        "diagnostic_report_sha256": "sha256:" + "56" * 32,
    }
    report_path.write_bytes(_canonical(report))
    report_path.chmod(0o644)
    if (
        binary.stat().st_mode & 0o777 != 0o755
        or report_path.stat().st_mode & 0o777 != 0o644
    ):
        pytest.skip("test temporary filesystem does not enforce POSIX modes")
    monkeypatch.setattr(
        amendment,
        "validate_rust_bridge_dag_binary_qualification_report_v1",
        lambda *_args, **_kwargs: "sha256:" + hashlib.sha256(binary.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        amendment,
        "_git",
        lambda _root, _arguments: source_payload,
    )
    actors = amendment.A8R1RecoveryDockerActorsV1(
        basis_commit=amendment.A8_BASIS_COMMIT,
        custody_directory=custody,
        rust_formal_replay_binary=tmp_path / "formal",
        rust_bridge_dag_replay_binary=binary,
        rust_bridge_dag_qualification_report=report_path,
        timestamp=0,
    )
    assert actors.validate_rust_bridge_dag_binding() == hashlib.sha256(
        binary.read_bytes()
    ).digest()
    assert actors.bridge_qualification_report_id_v1() == bytes.fromhex("56" * 32)


def test_wrapper_closes_actor_backend_when_acquire_fails(
    tmp_path: Path, monkeypatch,
) -> None:
    audit = tmp_path / "audit"
    custody = tmp_path / "custody"
    public = tmp_path / "public"
    for directory in (audit, custody, public):
        directory.mkdir(mode=0o700)
        directory.chmod(0o700)
    if audit.stat().st_mode & 0o777 != 0o700:
        pytest.skip("test temporary filesystem does not enforce POSIX mode 0700")
    monkeypatch.setattr(amendment, "FIXED_AUDIT_DIRECTORY", audit)
    preflight = {
        "schema": f"{amendment.AUDIT_SCHEMA_PREFIX}-preflight/1",
        "amendment_commit": "12" * 20,
        "sole_parent_commit": amendment.A8_BASIS_COMMIT,
        "formal_repository_commit": amendment.A8_BASIS_COMMIT,
        "run_id_hex": amendment.FIXED_RUN_ID_HEX,
        "ledger_id_hex": amendment.FIXED_LEDGER_ID_HEX,
        "manifest_sha256": "34" * 32,
        "source_bindings": [],
        "repository_clean": True,
        "exact_changed_paths_verified": True,
        "formal_identity_entropy_draw_count": 0,
        "ephemeral_container_nonce_allowed": True,
        "m3_start_allowed": False,
        "fixed_audit_directory": audit.as_posix(),
    }
    monkeypatch.setattr(
        amendment, "inspect_r1_source_preflight_v1", lambda **_kwargs: preflight
    )
    amendment.prepare_fixed_a8_r1_authorization_v1(audit_directory=audit)
    amendment.write_fixed_a8_r1_owner_authorization_v1(
        audit_directory=audit,
        owner_confirmation="AUTHORIZE_A8_R1_COMPLETE_ONLY_REAL_PENDING_RESUME",
    )
    close_calls = 0

    class FakeActors:
        def __init__(self, **_kwargs):
            pass

        def close(self):
            nonlocal close_calls
            close_calls += 1

    monkeypatch.setattr(amendment, "A8R1RecoveryDockerActorsV1", FakeActors)
    monkeypatch.setattr(
        amendment,
        "acquire_pending_ceremony_recovery_v1",
        lambda **_kwargs: (_ for _ in ()).throw(
            FormalContainerExecutorError("FAIL_TEST_ACQUIRE", "stop before custody")
        ),
    )
    with pytest.raises(FormalContainerExecutorError):
        amendment.execute_fixed_a8_r1_recovery_v1(
            custody_directory=custody,
            rust_formal_replay_binary=tmp_path / "formal",
            rust_bridge_dag_replay_binary=tmp_path / "bridge",
            rust_bridge_dag_qualification_report=tmp_path / "report",
            public_evidence_path=public / "evidence.json",
            public_promotion_path=public / "promotion.json",
            audit_directory=audit,
        )
    assert close_calls == 1
    assert (audit / "failure.json").exists()


@pytest.mark.parametrize("terminal_name", ("failure.json", "finalize.json"))
def test_r1_terminal_audit_cannot_be_retried_before_actor_construction(
    tmp_path: Path, monkeypatch, terminal_name: str,
) -> None:
    audit = tmp_path / "audit"
    audit.mkdir(mode=0o700)
    audit.chmod(0o700)
    if audit.stat().st_mode & 0o777 != 0o700:
        pytest.skip("test temporary filesystem does not enforce POSIX mode 0700")
    (audit / terminal_name).write_bytes(b"terminal\n")
    (audit / terminal_name).chmod(0o600)
    monkeypatch.setattr(amendment, "FIXED_AUDIT_DIRECTORY", audit)

    def actor_constructor_must_not_run(**_kwargs):
        raise AssertionError("actor backend was constructed for a terminal R1 audit")

    monkeypatch.setattr(
        amendment, "A8R1RecoveryDockerActorsV1", actor_constructor_must_not_run
    )
    with pytest.raises(amendment.A8RecoveryAmendmentError, match="terminal"):
        amendment.execute_fixed_a8_r1_recovery_v1(
            custody_directory=tmp_path / "missing-custody",
            rust_formal_replay_binary=tmp_path / "formal",
            rust_bridge_dag_replay_binary=tmp_path / "bridge",
            rust_bridge_dag_qualification_report=tmp_path / "report",
            public_evidence_path=tmp_path / "evidence.json",
            public_promotion_path=tmp_path / "promotion.json",
            audit_directory=audit,
        )


def test_r1_dangling_terminal_receipt_cannot_be_retried(
    tmp_path: Path, monkeypatch,
) -> None:
    audit = tmp_path / "audit"
    audit.mkdir(mode=0o700)
    audit.chmod(0o700)
    if audit.stat().st_mode & 0o777 != 0o700:
        pytest.skip("test temporary filesystem does not enforce POSIX mode 0700")
    (audit / "failure.json").symlink_to(audit / "absent-target")
    monkeypatch.setattr(amendment, "FIXED_AUDIT_DIRECTORY", audit)
    with pytest.raises(amendment.A8RecoveryAmendmentError, match="terminal"):
        amendment.execute_fixed_a8_r1_recovery_v1(
            custody_directory=tmp_path / "missing-custody",
            rust_formal_replay_binary=tmp_path / "formal",
            rust_bridge_dag_replay_binary=tmp_path / "bridge",
            rust_bridge_dag_qualification_report=tmp_path / "report",
            public_evidence_path=tmp_path / "evidence.json",
            public_promotion_path=tmp_path / "promotion.json",
            audit_directory=audit,
        )
