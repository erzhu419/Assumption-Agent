from __future__ import annotations

import copy
import fcntl
import hashlib
import hmac
import importlib.util
import inspect
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import tempfile
import threading
from types import MappingProxyType, SimpleNamespace

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

import hegel_machine.phase3_m25_actor_protocol_qualification_v1 as qualification
import hegel_machine.phase3_container_actor_runtime_v1 as actor_runtime
import hegel_machine.phase3_m25_container_ceremony_v1 as container_ceremony
import hegel_machine.phase3_m25_errata_qualification_v1 as errata_qualification
import hegel_machine.phase3_m25_parent_absence_audit_v1 as parent_absence
import hegel_machine.phase3_m25_purpose4_detached_audit_v1 as purpose4_detached
import hegel_machine.phase3_m25_secret_absence_v1 as secret_absence
import hegel_machine.phase3_m3_shadow_admission_v1 as shadow_admission


COMMIT = "12" * 20


def _digest(byte: str) -> str:
    return "sha256:" + byte * 64


@pytest.fixture
def linux_tmp_path() -> Path:
    root = Path(tempfile.mkdtemp(prefix="hegel-m25-qualification-test-", dir="/tmp"))
    try:
        yield root
    finally:
        shutil.rmtree(root)


def _stub_replayed() -> qualification.ReplayedActorProtocolQualificationV1:
    return qualification.ReplayedActorProtocolQualificationV1(
        basis_commit=COMMIT,
        bundle_content_id=b"b" * 32,
        qualification_key_ids=MappingProxyType(
            {purpose: bytes([purpose]) * 16 for purpose in (1, 2, 3, 4)}
        ),
        report=MappingProxyType(
            {
                "basis_commit": COMMIT,
                "bundle_content_id": _digest("b"),
                "diagnostic_only": True,
            }
        ),
    )


def _install_archive_stubs(monkeypatch: pytest.MonkeyPatch) -> tuple[
    dict[str, object],
    dict[int, tuple[bytes, bytes, Mapping[str, object]]],
]:
    evidence_id = _digest("a")
    evidence = {"basis_commit": COMMIT, "stub": True}
    keys = {
        purpose: (
            bytes([purpose]) * 32,
            bytes([purpose]) * 16,
            {"manifest_content_id": _digest(str(purpose))},
        )
        for purpose in (1, 2, 3, 4)
    }
    plan = {"destruction_plan_content_id": _digest("d")}
    cleanup = {"cleanup_receipt_content_id": _digest("c")}
    statements = [{"purpose_id": purpose} for purpose in (1, 2, 3, 4)]
    monkeypatch.setattr(
        qualification,
        "_commit_source_set_digest_from_git_v1",
        lambda commit: (_digest("2"), 10)
        if commit == COMMIT
        else pytest.fail("unexpected source-set commit"),
    )
    monkeypatch.setattr(
        qualification,
        "_validate_evidence_v1",
        lambda value, *, basis_commit: evidence_id
        if value == evidence and basis_commit == COMMIT
        else pytest.fail("unexpected evidence replay"),
    )
    monkeypatch.setattr(
        qualification,
        "_validate_key_manifests_v1",
        lambda rows, *, basis_commit, evidence: keys,
    )
    monkeypatch.setattr(
        qualification, "_validate_purpose4_and_bridge_v1", lambda *_args: None
    )
    monkeypatch.setattr(
        qualification,
        "_validate_implementation_bindings_v1",
        lambda value, *, basis_commit: None,
    )
    monkeypatch.setattr(
        qualification,
        "_validate_destruction_plan_v1",
        lambda value, *, evidence: plan,
    )
    monkeypatch.setattr(
        qualification, "_validate_statements_v1", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        qualification,
        "_validate_cleanup_receipt_v1",
        lambda value, *, evidence, plan: cleanup,
    )
    body: dict[str, object] = {
        "schema_version": qualification.SCHEMA_VERSION,
        "artifact_kind": qualification.ARTIFACT_KIND,
        "status": qualification.STATUS,
        "claim_level": qualification.CLAIM_LEVEL,
        "basis_commit": COMMIT,
        "commit_a_source_set_sha256": _digest("2"),
        "commit_a_source_file_count": 10,
        "authority_boundary": dict(qualification.AUTHORITY_BOUNDARY),
        "independence_disclosure": dict(qualification.INDEPENDENCE_DISCLOSURE),
        "implementation_bindings": {
            "formal_rust_replay_binary_sha256": _digest("3"),
            "bridge_rust_replay_binary_sha256": _digest("4"),
            "bridge_rust_qualification_report_sha256": _digest("5"),
            "m3_implementation_qualification_receipt_sha256": _digest("6"),
            "m3_implementation_qualification_receipt": {"stub": True},
        },
        "qualification_key_manifests": [{"purpose_id": value} for value in range(1, 5)],
        "evidence_bundle": evidence,
        "destruction_plan": plan,
        "cleanup_absence_receipt": cleanup,
        "qualification_statements": statements,
    }
    authority_preimage = {
        "basis_commit": COMMIT,
        "evidence_content_id": evidence_id,
        "destruction_plan_content_id": plan["destruction_plan_content_id"],
        "cleanup_receipt_content_id": cleanup["cleanup_receipt_content_id"],
        "qualification_key_manifest_content_ids": [
            keys[purpose][2]["manifest_content_id"] for purpose in (1, 2, 3, 4)
        ],
        "qualification_statement_sha256": [
            qualification._sha256(qualification._canonical_json(row))
            for row in statements
        ],
    }
    body["bundle_content_id"] = qualification._content_id(
        qualification.BUNDLE_AUTHORITY_HASH_DOMAIN, authority_preimage
    )
    body["diagnostic_report_sha256"] = qualification._report_hash(body)
    return body, keys


def test_authority_and_independence_boundaries_are_exact() -> None:
    boundary = qualification.AUTHORITY_BOUNDARY
    assert boundary["authoritative_formal_roots_generated"] is False
    assert boundary["synthetic_formal_shaped_roots_computed_in_memory"] is True
    assert boundary["ephemeral_private_keys_published"] is False
    assert boundary["ephemeral_public_keys_published"] is True
    assert boundary["ephemeral_signatures_published"] is True
    assert boundary["m3_gates_after"] == 14
    assert boundary["m3_state"] == "NOT_RUN"
    assert qualification.INDEPENDENCE_DISCLOSURE == {
        "same_admin_controller": True,
        "organizational_independence": False,
        "independent_human_actors": False,
        "technical_role_independence": True,
        "owner_accepted_threat_model": True,
        "remote_attestation": False,
        "hardware_key_nonexportability": False,
    }


def test_public_synthetic_split_fixture_is_fixed_and_contains_no_assignments() -> None:
    frame = qualification.PUBLIC_SYNTHETIC_SPLIT_FRAME
    assert qualification.PUBLIC_SYNTHETIC_SPLIT_FRAME_SHA256 == (
        "sha256:" + hashlib.sha256(frame).hexdigest()
    )
    assert b"assignment" not in frame.lower()


class _FakeDelegate:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.seed_calls = 0
        for purpose in (1, 2, 3, 4):
            (root / f"p{purpose}" / "input").mkdir(parents=True)
            (root / f"p{purpose}" / "output").mkdir()

    def _actor_dirs(self, purpose: int):
        return self.root / f"p{purpose}" / "input", self.root / f"p{purpose}" / "output"

    def keygen(self, purpose: int) -> bytes:
        return bytes([purpose]) * 32

    def sign_object(self, _name, _fields) -> bytes:
        return b"o" * 64

    def sign_parent(self, _evidence, _fields) -> bytes:
        input_directory, output = self._actor_dirs(4)
        (input_directory / "purpose4-keybearing-request.json").write_bytes(
            b'{"request":"public-synthetic"}\n'
        )
        (output / "purpose4-keybearing-detached-response.json").write_bytes(
            b'{"response":"public-synthetic"}\n'
        )
        return b"p" * 64

    def sign_bridge(self, purpose, _fields, _replay_package) -> bytes:
        _input, output = self._actor_dirs(purpose)
        (output / "bridge-dag-replay-receipt.json").write_bytes(
            json.dumps({"purpose": purpose}, sort_keys=True).encode() + b"\n"
        )
        return bytes([purpose]) * 64

    def seed_split(self):
        self.seed_calls += 1
        raise AssertionError("delegate seed_split must never be called")

    def complete_marker(self, _root):
        self.seed_calls += 1
        raise AssertionError("delegate complete_marker must never be called")


def test_wrapper_delegates_only_public_synthetic_signing_protocol(tmp_path: Path) -> None:
    delegate = _FakeDelegate(tmp_path)
    wrapper = qualification.PublicSyntheticProtocolActorV1(delegate)  # type: ignore[arg-type]
    for purpose in (1, 2, 3, 4):
        assert wrapper.keygen(purpose) == bytes([purpose]) * 32
    assert wrapper.seed_split() == (wrapper.split_frame, wrapper.split_frame)
    assert wrapper.prospective_complete_marker(b"m" * 32).state == "COMPLETE"
    for name in (
        "SplitSeedCommitmentManifestV1",
        "CustodianBindingManifestV1",
        "SeedContinuityManifestV1",
        "HiddenAccessLedgerRecordV1",
    ):
        wrapper.sign_object(name, {})
    wrapper.sign_parent(object(), {})
    for purpose in (1, 2, 3):
        wrapper.sign_bridge(purpose, {}, b"package")
    summary = qualification._wrapper_summary_v1(wrapper)
    assert summary["docker_seed_or_marker_method_call_count"] == 0
    assert delegate.seed_calls == 0


def test_wrapper_seed_state_operations_are_fail_closed(tmp_path: Path) -> None:
    wrapper = qualification.PublicSyntheticProtocolActorV1(_FakeDelegate(tmp_path))  # type: ignore[arg-type]
    with pytest.raises(qualification.ActorProtocolQualificationError):
        wrapper.complete_marker(b"x" * 32)
    assert wrapper.docker_seed_method_call_count == 1


def _stub_custody_validator(monkeypatch: pytest.MonkeyPatch, custody: Path) -> None:
    monkeypatch.setattr(
        qualification,
        "validate_linux_local_durable_custody_location_v1",
        lambda path, **_kwargs: {
            "schema": "test-location",
            "requested_path": os.fspath(path),
            "resolved_path": str(custody.resolve()),
            "owner_uid": os.geteuid(),
            "mode_octal": "0700",
        },
    )


def test_reservation_cleanup_requires_actor_absence_and_is_exact(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custody = linux_tmp_path / "durable"
    custody.mkdir(mode=0o700)
    _stub_custody_validator(monkeypatch, custody)
    reservation = qualification.QualificationCustodyReservationV1(
        custody, COMMIT, b"r" * 16, b"l" * 16
    )
    reservation.reserve()
    reservation.mark_actors_started()
    with pytest.raises(qualification.ActorProtocolQualificationError):
        reservation.cleanup_after_actor_absence()
    assert len(list(custody.iterdir())) == 3

    class FakeRecoveryBackend:
        def __init__(self, **_kwargs) -> None:
            pass

        def recover_preseed_private_state_and_verify_absent(self, value: bytes):
            return {
                "run_id_hex": value.hex(),
                "actor_containers_absent": True,
                "actor_key_volumes_absent": True,
                "seed_continuity_state_absent": True,
            }

    monkeypatch.setattr(qualification, "DockerCeremonyActorsV1", FakeRecoveryBackend)
    assert qualification._recover_orphaned_qualification_reservation_v1(
        custody_directory=custody,
        basis_commit=COMMIT,
        rust_formal_replay_binary=linux_tmp_path / "unused-rust",
    )
    assert list(custody.iterdir()) == []


def test_reservation_partial_fsync_failure_does_not_leave_wedge(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = linux_tmp_path / "partial.reserved"
    monkeypatch.setattr(qualification.os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError("fault")))
    with pytest.raises(qualification.ActorProtocolQualificationError):
        qualification._exclusive_file(path, b"partial")
    assert not path.exists()


def test_cleanup_durably_removes_opaque_files_before_lock(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custody = linux_tmp_path / "durable"
    custody.mkdir(mode=0o700)
    _stub_custody_validator(monkeypatch, custody)
    reservation = qualification.QualificationCustodyReservationV1(
        custody, COMMIT, b"r" * 16, b"l" * 16
    )
    reservation.reserve()
    original_fsync_directory = qualification._fsync_directory
    calls = 0

    def fail_first_barrier(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("crash-order proxy")
        original_fsync_directory(path)

    monkeypatch.setattr(qualification, "_fsync_directory", fail_first_barrier)
    with pytest.raises(OSError):
        reservation.cleanup_before_actor_start()
    assert {path.name for path in custody.iterdir()} == {
        "phase3_m25_ceremony.lock"
    }
    monkeypatch.setattr(
        qualification, "_fsync_directory", original_fsync_directory
    )
    with pytest.raises(qualification.ActorProtocolQualificationError):
        reservation.cleanup_before_actor_start()
    assert {path.name for path in custody.iterdir()} == {
        "phase3_m25_ceremony.lock"
    }


def test_cleanup_first_fsync_failure_releases_locks_for_exact_recovery(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custody = linux_tmp_path / "durable"
    custody.mkdir(mode=0o700)
    _stub_custody_validator(monkeypatch, custody)
    run_id = b"r" * 16
    reservation = qualification.QualificationCustodyReservationV1(
        custody, COMMIT, run_id, b"l" * 16
    )
    reservation.reserve()
    contender_lock = os.open(reservation.paths[0], os.O_RDONLY)
    contender_directory = os.open(custody, os.O_RDONLY | os.O_DIRECTORY)
    original_fsync_directory = qualification._fsync_directory
    calls = 0

    def fail_first(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("first barrier fault")
        original_fsync_directory(path)

    monkeypatch.setattr(qualification, "_fsync_directory", fail_first)
    with pytest.raises(OSError):
        reservation.cleanup_before_actor_start()
    fcntl.flock(contender_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(contender_directory, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(contender_lock, fcntl.LOCK_UN)
    fcntl.flock(contender_directory, fcntl.LOCK_UN)
    os.close(contender_lock)
    os.close(contender_directory)
    monkeypatch.setattr(qualification, "_fsync_directory", original_fsync_directory)

    class FakeRecoveryBackend:
        def __init__(self, **_kwargs) -> None:
            pass

        def recover_preseed_private_state_and_verify_absent(self, value: bytes):
            return {
                "run_id_hex": value.hex(),
                "actor_containers_absent": True,
                "actor_key_volumes_absent": True,
                "seed_continuity_state_absent": True,
            }

    monkeypatch.setattr(qualification, "DockerCeremonyActorsV1", FakeRecoveryBackend)
    assert qualification._recover_orphaned_qualification_reservation_v1(
        custody_directory=custody,
        basis_commit=COMMIT,
        rust_formal_replay_binary=linux_tmp_path / "unused-rust",
    )
    assert list(custody.iterdir()) == []


def test_cleanup_final_fsync_failure_releases_unlinked_inode_and_directory_locks(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custody = linux_tmp_path / "durable"
    custody.mkdir(mode=0o700)
    _stub_custody_validator(monkeypatch, custody)
    reservation = qualification.QualificationCustodyReservationV1(
        custody, COMMIT, b"r" * 16, b"l" * 16
    )
    reservation.reserve()
    contender_lock = os.open(reservation.paths[0], os.O_RDONLY)
    contender_directory = os.open(custody, os.O_RDONLY | os.O_DIRECTORY)
    original_fsync_directory = qualification._fsync_directory
    calls = 0

    def fail_second(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("final barrier fault")
        original_fsync_directory(path)

    monkeypatch.setattr(qualification, "_fsync_directory", fail_second)
    with pytest.raises(OSError):
        reservation.cleanup_before_actor_start()
    assert list(custody.iterdir()) == []
    fcntl.flock(contender_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(contender_directory, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.close(contender_lock)
    os.close(contender_directory)


def test_cleanup_fingerprint_failure_releases_locks_and_recovery_fails_closed(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custody = linux_tmp_path / "durable"
    custody.mkdir(mode=0o700)
    _stub_custody_validator(monkeypatch, custody)
    reservation = qualification.QualificationCustodyReservationV1(
        custody, COMMIT, b"r" * 16, b"l" * 16
    )
    reservation.reserve()
    reservation.paths[2].write_bytes(
        qualification._canonical_json({"externally": "changed"})
    )
    reservation.paths[2].chmod(0o600)
    contender_lock = os.open(reservation.paths[0], os.O_RDONLY)
    contender_directory = os.open(custody, os.O_RDONLY | os.O_DIRECTORY)
    with pytest.raises(qualification.ActorProtocolQualificationError):
        reservation.cleanup_before_actor_start()
    fcntl.flock(contender_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(contender_directory, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(contender_lock, fcntl.LOCK_UN)
    fcntl.flock(contender_directory, fcntl.LOCK_UN)
    os.close(contender_lock)
    os.close(contender_directory)
    with pytest.raises(qualification.ActorProtocolQualificationError):
        qualification._recover_orphaned_qualification_reservation_v1(
            custody_directory=custody,
            basis_commit=COMMIT,
            rust_formal_replay_binary=linux_tmp_path / "unused-rust",
        )
    probe_directory = os.open(custody, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(probe_directory, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(probe_directory)


def test_cleanup_holds_file_and_directory_flocks_through_final_dir_fsync(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custody = linux_tmp_path / "durable"
    custody.mkdir(mode=0o700)
    _stub_custody_validator(monkeypatch, custody)
    reservation = qualification.QualificationCustodyReservationV1(
        custody, COMMIT, b"r" * 16, b"l" * 16
    )
    reservation.reserve()
    contender_lock = os.open(reservation.paths[0], os.O_RDONLY)
    contender_directory = os.open(custody, os.O_RDONLY | os.O_DIRECTORY)
    original_fsync_directory = qualification._fsync_directory
    barriers = 0

    def assert_locked_at_barriers(path: Path) -> None:
        nonlocal barriers
        barriers += 1
        if barriers == 2:
            assert not reservation.paths[0].exists()
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender_directory, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original_fsync_directory(path)

    monkeypatch.setattr(qualification, "_fsync_directory", assert_locked_at_barriers)
    try:
        reservation.cleanup_before_actor_start()
        assert barriers == 2
        fcntl.flock(contender_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(contender_directory, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(contender_lock)
        os.close(contender_directory)


def test_reserve_directory_mutex_precedes_first_visible_reservation(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custody = linux_tmp_path / "durable"
    custody.mkdir(mode=0o700)
    _stub_custody_validator(monkeypatch, custody)
    reservation = qualification.QualificationCustodyReservationV1(
        custody, COMMIT, b"r" * 16, b"l" * 16
    )
    entered = threading.Event()
    proceed = threading.Event()
    failures: list[BaseException] = []
    original_exclusive_file = qualification._exclusive_file

    def paused_exclusive_file(path: Path, payload: bytes):
        if path == reservation.paths[0]:
            entered.set()
            if not proceed.wait(timeout=5):
                raise RuntimeError("test barrier timed out")
        return original_exclusive_file(path, payload)

    monkeypatch.setattr(qualification, "_exclusive_file", paused_exclusive_file)
    monkeypatch.setattr(
        qualification,
        "DockerCeremonyActorsV1",
        lambda **_kwargs: pytest.fail("live reservation invoked orphan Docker cleanup"),
    )

    def run_reserve() -> None:
        try:
            reservation.reserve()
        except BaseException as exc:  # pragma: no cover - assertion below reports it
            failures.append(exc)

    worker = threading.Thread(target=run_reserve)
    worker.start()
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(qualification.ActorProtocolQualificationError):
            qualification._recover_orphaned_qualification_reservation_v1(
                custody_directory=custody,
                basis_commit=COMMIT,
                rust_formal_replay_binary=linux_tmp_path / "unused-rust",
            )
        assert list(custody.iterdir()) == []
    finally:
        proceed.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert failures == []
    reservation.cleanup_before_actor_start()


def test_reservation_rejects_symlink_alias_before_validation(
    linux_tmp_path: Path,
) -> None:
    real = linux_tmp_path / "real"
    real.mkdir(mode=0o700)
    alias = linux_tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(qualification.ActorProtocolQualificationError):
        qualification.QualificationCustodyReservationV1(
            alias, COMMIT, b"r" * 16, b"l" * 16
        )


def test_unlocked_reservation_object_cannot_delete_recovery_evidence(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custody = linux_tmp_path / "durable"
    custody.mkdir(mode=0o700)
    _stub_custody_validator(monkeypatch, custody)
    reservation = qualification.QualificationCustodyReservationV1(
        custody, COMMIT, b"r" * 16, b"l" * 16
    )
    reservation.reserve()
    reservation.release_lock_without_cleanup()
    before = {
        path.name: (path.stat().st_ino, path.read_bytes())
        for path in custody.iterdir()
    }
    with pytest.raises(qualification.ActorProtocolQualificationError):
        reservation.cleanup_before_actor_start()
    after = {
        path.name: (path.stat().st_ino, path.read_bytes())
        for path in custody.iterdir()
    }
    assert after == before


def test_mid_reserve_crash_state_still_proves_docker_absence(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custody = linux_tmp_path / "durable"
    custody.mkdir(mode=0o700)
    _stub_custody_validator(monkeypatch, custody)
    reservation = qualification.QualificationCustodyReservationV1(
        custody, COMMIT, b"r" * 16, b"l" * 16
    )
    reservation.reserve()
    reservation.release_lock_without_cleanup()
    # reserve() writes lock, run, ledger in that order.  This exact prefix is
    # the only possible mid-reserve process-kill state.
    reservation.paths[1].unlink()
    reservation.paths[2].unlink()
    observed: list[bytes] = []

    class FakeRecoveryBackend:
        def __init__(self, **_kwargs) -> None:
            pass

        def recover_preseed_private_state_and_verify_absent(self, value: bytes):
            observed.append(value)
            return {
                "run_id_hex": value.hex(),
                "actor_containers_absent": True,
                "actor_key_volumes_absent": True,
                "seed_continuity_state_absent": True,
            }

    monkeypatch.setattr(qualification, "DockerCeremonyActorsV1", FakeRecoveryBackend)
    assert qualification._recover_orphaned_qualification_reservation_v1(
        custody_directory=custody,
        basis_commit=COMMIT,
        rust_formal_replay_binary=linux_tmp_path / "unused-rust",
    )
    assert observed == [b"r" * 16]
    assert list(custody.iterdir()) == []


def test_post_start_crash_state_uses_exact_preseed_backend_recovery(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custody = linux_tmp_path / "durable"
    custody.mkdir(mode=0o700)
    _stub_custody_validator(monkeypatch, custody)
    run_id = b"r" * 16
    reservation = qualification.QualificationCustodyReservationV1(
        custody, COMMIT, run_id, b"l" * 16
    )
    reservation.reserve()
    reservation.release_lock_without_cleanup()
    observed: list[bytes] = []

    class FakeRecoveryBackend:
        def __init__(self, **_kwargs) -> None:
            pass

        def recover_preseed_private_state_and_verify_absent(self, value: bytes):
            observed.append(value)
            return {
                "run_id_hex": value.hex(),
                "actor_containers_absent": True,
                "actor_key_volumes_absent": True,
                "seed_continuity_state_absent": True,
            }

    monkeypatch.setattr(
        qualification, "DockerCeremonyActorsV1", FakeRecoveryBackend
    )
    assert qualification._recover_orphaned_qualification_reservation_v1(
        custody_directory=custody,
        basis_commit=COMMIT,
        rust_formal_replay_binary=linux_tmp_path / "unused-rust",
    )
    assert observed == [run_id]
    assert list(custody.iterdir()) == []


def test_sigkill_partial_opaque_body_uses_lock_bound_preseed_recovery(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custody = linux_tmp_path / "durable"
    custody.mkdir(mode=0o700)
    _stub_custody_validator(monkeypatch, custody)
    run_id = b"r" * 16
    reservation = qualification.QualificationCustodyReservationV1(
        custody, COMMIT, run_id, b"l" * 16
    )
    reservation.reserve()
    reservation.release_lock_without_cleanup()
    reservation.paths[2].write_bytes(b'{"opaque_id_hex":"truncated')
    reservation.paths[2].chmod(0o600)
    observed: list[bytes] = []

    class FakeRecoveryBackend:
        def __init__(self, **_kwargs) -> None:
            pass

        def recover_preseed_private_state_and_verify_absent(self, value: bytes):
            observed.append(value)
            return {
                "run_id_hex": value.hex(),
                "actor_containers_absent": True,
                "actor_key_volumes_absent": True,
                "seed_continuity_state_absent": True,
            }

    monkeypatch.setattr(qualification, "DockerCeremonyActorsV1", FakeRecoveryBackend)
    qualification._recover_orphaned_qualification_reservation_v1(
        custody_directory=custody,
        basis_commit=COMMIT,
        rust_formal_replay_binary=linux_tmp_path / "unused-rust",
    )
    assert observed == [run_id]
    assert list(custody.iterdir()) == []


def test_orphan_lock_read_error_releases_file_and_directory_flocks(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custody = linux_tmp_path / "durable"
    custody.mkdir(mode=0o700)
    _stub_custody_validator(monkeypatch, custody)
    reservation = qualification.QualificationCustodyReservationV1(
        custody, COMMIT, b"r" * 16, b"l" * 16
    )
    reservation.reserve()
    reservation.release_lock_without_cleanup()
    lock_path = reservation.paths[0]
    original_read_bytes = Path.read_bytes

    def fail_lock_read(path: Path) -> bytes:
        if path == lock_path:
            raise OSError("injected lock read fault")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_lock_read)
    with pytest.raises(qualification.ActorProtocolQualificationError):
        qualification._recover_orphaned_qualification_reservation_v1(
            custody_directory=custody,
            basis_commit=COMMIT,
            rust_formal_replay_binary=linux_tmp_path / "unused-rust",
        )
    probe_lock = os.open(lock_path, os.O_RDONLY)
    probe_directory = os.open(custody, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(probe_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(probe_directory, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(probe_lock)
        os.close(probe_directory)


def test_orphan_recovery_refuses_live_lock_and_unknown_path(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custody = linux_tmp_path / "durable"
    custody.mkdir(mode=0o700)
    _stub_custody_validator(monkeypatch, custody)
    reservation = qualification.QualificationCustodyReservationV1(
        custody, COMMIT, b"r" * 16, b"l" * 16
    )
    reservation.reserve()
    with pytest.raises(qualification.ActorProtocolQualificationError):
        qualification._recover_orphaned_qualification_reservation_v1(
            custody_directory=custody,
            basis_commit=COMMIT,
            rust_formal_replay_binary=linux_tmp_path / "unused-rust",
        )
    reservation.cleanup_before_actor_start()
    (custody / "foreign").write_text("preserve", encoding="ascii")
    with pytest.raises(qualification.ActorProtocolQualificationError):
        qualification._recover_orphaned_qualification_reservation_v1(
            custody_directory=custody,
            basis_commit=COMMIT,
            rust_formal_replay_binary=linux_tmp_path / "unused-rust",
        )
    assert (custody / "foreign").read_text(encoding="ascii") == "preserve"


def test_sigkill_partial_lock_is_exactly_removed_without_docker(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custody = linux_tmp_path / "durable"
    custody.mkdir(mode=0o700)
    _stub_custody_validator(monkeypatch, custody)
    lock = custody / "phase3_m25_ceremony.lock"
    lock.write_bytes(b'{"schema":"truncated')
    lock.chmod(0o600)
    monkeypatch.setattr(
        qualification,
        "DockerCeremonyActorsV1",
        lambda **_kwargs: pytest.fail("partial lock invoked Docker cleanup"),
    )
    assert qualification._recover_orphaned_qualification_reservation_v1(
        custody_directory=custody,
        basis_commit=COMMIT,
        rust_formal_replay_binary=linux_tmp_path / "unused-rust",
    )
    assert list(custody.iterdir()) == []


def test_orphan_recovery_rejects_caller_symlink_before_any_cleanup(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = linux_tmp_path / "real"
    real.mkdir(mode=0o700)
    sentinel = real / "phase3_m25_ceremony.lock"
    sentinel.write_text("do not follow", encoding="ascii")
    alias = linux_tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(
        qualification,
        "DockerCeremonyActorsV1",
        lambda **_kwargs: pytest.fail("symlink alias invoked Docker cleanup"),
    )
    with pytest.raises(qualification.ActorProtocolQualificationError):
        qualification._recover_orphaned_qualification_reservation_v1(
            custody_directory=alias,
            basis_commit=COMMIT,
            rust_formal_replay_binary=linux_tmp_path / "unused-rust",
        )
    assert sentinel.read_text(encoding="ascii") == "do not follow"


def test_archive_round_trip_returns_replay_not_live_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, _keys = _install_archive_stubs(monkeypatch)
    replayed = qualification.validate_actor_protocol_qualification_report_v1(report)
    assert type(replayed) is qualification.ReplayedActorProtocolQualificationV1
    assert type(replayed) is not qualification.LiveActorProtocolAdmissionV1
    payload = qualification.canonical_actor_protocol_qualification_report_bytes_v1(report)
    assert qualification._strict_json_object(payload) == report


def test_commit_reads_ignore_replace_refs_and_hostile_git_environment(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = linux_tmp_path / "git-identity"
    repository.mkdir()

    def git(*arguments: str, environment: dict[str, str] | None = None) -> bytes:
        completed = subprocess.run(
            ["/usr/bin/git", *arguments],
            cwd=repository,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=environment,
        )
        assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
        return completed.stdout

    git("init", "-q")
    git("config", "user.name", "Qualification Test")
    git("config", "user.email", "qualification@example.invalid")
    bound = repository / "bound.txt"
    bound.write_text("honest\n", encoding="ascii")
    git("add", "bound.txt")
    git("commit", "-q", "-m", "honest")
    honest_commit = git("rev-parse", "HEAD").decode("ascii").strip()
    bound.write_text("replacement\n", encoding="ascii")
    git("commit", "-qam", "replacement")
    replacement_commit = git("rev-parse", "HEAD").decode("ascii").strip()
    git("replace", honest_commit, replacement_commit)
    assert git("show", f"{honest_commit}:bound.txt") == b"replacement\n"

    hostile_global = repository / "hostile.gitconfig"
    hostile_global.write_text("[core]\n\tquotepath = false\n", encoding="ascii")
    for name, value in {
        "GIT_CONFIG_NOSYSTEM": "0",
        "GIT_CONFIG_GLOBAL": str(hostile_global),
        "GIT_DIR": str(repository / "not-a-git-dir"),
        "GIT_WORK_TREE": str(linux_tmp_path),
        "GIT_OBJECT_DIRECTORY": str(repository / "hostile-objects"),
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(repository / ".git/objects"),
        "GIT_REPLACE_REF_BASE": "refs/replace",
        "GIT_NO_REPLACE_OBJECTS": "0",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(qualification, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(actor_runtime, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(container_ceremony, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(shadow_admission, "REPOSITORY_ROOT", repository)

    real_subprocess_run = subprocess.run
    git_calls: list[tuple[object, dict[str, str]]] = []

    def capture_git_environment(*args, **kwargs):
        command = args[0] if args else kwargs.get("args")
        environment = kwargs.get("env")
        if isinstance(command, (tuple, list)) and command and str(command[0]).endswith("git"):
            git_calls.append((command, dict(environment or {})))
        return real_subprocess_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", capture_git_environment)

    payloads = (
        qualification._git(("show", f"{honest_commit}:bound.txt")),
        actor_runtime._git_run(("show", f"{honest_commit}:bound.txt")).stdout,
        container_ceremony._run_git(
            ("show", f"{honest_commit}:bound.txt"), binary=True
        ),
        errata_qualification._git_completed(
            repository, ["show", f"{honest_commit}:bound.txt"]
        ).stdout,
        secret_absence._git_bytes(
            repository, ["show", f"{honest_commit}:bound.txt"]
        ),
        parent_absence._run_git(
            repository, ("show", f"{honest_commit}:bound.txt")
        ),
        purpose4_detached._run_git(
            Path("/usr/bin/git"),
            repository,
            ("show", f"{honest_commit}:bound.txt"),
        ),
        shadow_admission._git("show", f"{honest_commit}:bound.txt"),
    )
    assert set(payloads) == {b"honest\n"}
    assert len(git_calls) == len(payloads)
    for command, environment in git_calls:
        assert str(command[0]) == "/usr/bin/git"
        assert environment["PATH"] in {
            "/usr/bin:/bin",
            "/usr/bin:/usr/bin:/bin",
        }
        assert environment["HOME"] == "/nonexistent"
        assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
        assert environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
        assert environment["GIT_CONFIG_SYSTEM"] == "/dev/null"
        assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
        assert environment["GIT_NO_LAZY_FETCH"] == "1"
        assert environment["GIT_PROTOCOL_FROM_USER"] == "0"
        assert environment["GIT_SSH_COMMAND"] == "false"
        assert "GIT_DIR" not in environment
        assert "GIT_OBJECT_DIRECTORY" not in environment
        assert "GIT_ALTERNATE_OBJECT_DIRECTORIES" not in environment


def test_implementation_binding_requires_full_m3_receipt_and_local_binaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = {"basis_commit": COMMIT, "full_receipt": True}
    bindings = {
        "formal_rust_replay_binary_sha256": _digest("3"),
        "bridge_rust_replay_binary_sha256": _digest("4"),
        "bridge_rust_qualification_report_sha256": _digest("5"),
        "m3_implementation_qualification_receipt_sha256": qualification._sha256(
            qualification._canonical_json(receipt)
        ),
        "m3_implementation_qualification_receipt": receipt,
    }
    monkeypatch.setattr(
        qualification,
        "load_committed_dual_golden_v1",
        lambda repository, commit: ({"expected": {}}, b"g", b"r" * 32),
    )
    observed: list[object] = []
    monkeypatch.setattr(
        qualification,
        "validate_qualification_receipt_v1",
        lambda value, *, golden, basis_commit: observed.append(
            (value, golden, basis_commit)
        ),
    )
    monkeypatch.setattr(
        qualification,
        "load_qualified_rust_bridge_dag_binary_binding_v1",
        lambda **_kwargs: ({"diagnostic_report_sha256": _digest("5")}, _digest("4")),
    )
    monkeypatch.setattr(
        qualification,
        "_hash_regular_file",
        lambda *_args, **_kwargs: _digest("3"),
    )
    qualification._validate_implementation_bindings_v1(
        bindings, basis_commit=COMMIT
    )
    assert observed and observed[0][0] == receipt
    changed = copy.deepcopy(bindings)
    changed["m3_implementation_qualification_receipt"]["full_receipt"] = False
    with pytest.raises(qualification.ActorProtocolQualificationError):
        qualification._validate_implementation_bindings_v1(
            changed, basis_commit=COMMIT
        )


def _destruction_and_cleanup_fixture():
    run_hex = "34" * 16
    evidence = {
        "basis_commit": COMMIT,
        "qualification_run_id_hex": run_hex,
        "profile": {"daemon_receipt_sha256": _digest("7")},
        "actor_runtime_rows": [
            {"container_id": str(purpose) * 64}
            for purpose in (1, 2, 3, 4)
        ],
    }
    plan_body = {
        "schema": qualification.DESTRUCTION_PLAN_SCHEMA,
        "basis_commit": COMMIT,
        "qualification_run_id_hex": run_hex,
        "daemon_receipt_sha256": _digest("7"),
        "actor_rows": [
            {
                "purpose_id": purpose,
                "container_id": str(purpose) * 64,
                "actor_key_volume_name": (
                    f"hegel-m25-state-{run_hex}-p{purpose}"
                ),
                "must_remove_container": True,
                "must_remove_actor_key_volume": True,
            }
            for purpose in (1, 2, 3, 4)
        ],
        "required_cleanup_order": [
            "CONTAINERS_REMOVED_AND_VERIFIED_ABSENT",
            "KEY_VOLUMES_REMOVED_AND_VERIFIED_ABSENT",
            "EXACT_RESERVATIONS_REMOVED",
        ],
        "seed_or_marker_artifacts_must_remain_absent": True,
        "qualification_keys_must_be_destroyed": True,
        "formal_genesis_reuse_forbidden": True,
    }
    plan = {
        **plan_body,
        "destruction_plan_content_id": qualification._content_id(
            qualification.DESTRUCTION_PLAN_HASH_DOMAIN, plan_body
        ),
    }
    cleanup_body = {
        "schema": qualification.CLEANUP_RECEIPT_SCHEMA,
        "basis_commit": COMMIT,
        "qualification_run_id_hex": run_hex,
        "daemon_receipt_sha256": _digest("7"),
        "destruction_plan_content_id": plan["destruction_plan_content_id"],
        "actor_rows": [
            {
                "purpose_id": purpose,
                "container_id": str(purpose) * 64,
                "actor_key_volume_name": f"hegel-m25-state-{run_hex}-p{purpose}",
                "container_inspect_returncode": 1,
                "container_inspect_stdout_sha256": _digest("8"),
                "container_list_returncode": 0,
                "container_list_stdout_sha256": qualification._sha256(b""),
                "volume_inspect_returncode": 1,
                "volume_inspect_stdout_sha256": _digest("9"),
                "volume_list_returncode": 0,
                "volume_list_stdout_sha256": qualification._sha256(b""),
                "container_verified_absent": True,
                "actor_key_volume_verified_absent": True,
            }
            for purpose in (1, 2, 3, 4)
        ],
        "seed_marker_absent": True,
        "seed_intent_absent": True,
        "seed_completion_absent": True,
        "raw_seed_absent": True,
        "exact_reservations_removed": True,
        "custody_directory_empty": True,
        "custody_directory_removed": False,
    }
    cleanup = {
        **cleanup_body,
        "cleanup_receipt_content_id": qualification._content_id(
            qualification.CLEANUP_RECEIPT_HASH_DOMAIN, cleanup_body
        ),
    }
    return evidence, plan, cleanup


def test_destruction_volume_identity_and_cleanup_returncodes_are_exact() -> None:
    evidence, plan, cleanup = _destruction_and_cleanup_fixture()
    assert qualification._validate_destruction_plan_v1(
        plan, evidence=evidence
    ) == plan
    assert qualification._validate_cleanup_receipt_v1(
        cleanup, evidence=evidence, plan=plan
    ) == cleanup

    swapped = copy.deepcopy(plan)
    swapped["actor_rows"][0]["actor_key_volume_name"] = swapped["actor_rows"][1][
        "actor_key_volume_name"
    ]
    swapped_body = dict(swapped)
    swapped_body.pop("destruction_plan_content_id")
    swapped["destruction_plan_content_id"] = qualification._content_id(
        qualification.DESTRUCTION_PLAN_HASH_DOMAIN, swapped_body
    )
    with pytest.raises(qualification.ActorProtocolQualificationError):
        qualification._validate_destruction_plan_v1(swapped, evidence=evidence)

    boolean_returncode = copy.deepcopy(cleanup)
    boolean_returncode["actor_rows"][0]["container_list_returncode"] = False
    cleanup_body = dict(boolean_returncode)
    cleanup_body.pop("cleanup_receipt_content_id")
    boolean_returncode["cleanup_receipt_content_id"] = qualification._content_id(
        qualification.CLEANUP_RECEIPT_HASH_DOMAIN, cleanup_body
    )
    with pytest.raises(qualification.ActorProtocolQualificationError):
        qualification._validate_cleanup_receipt_v1(
            boolean_returncode, evidence=evidence, plan=plan
        )


def test_bridge_signature_hex_requires_unique_lowercase_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = {"request": "p4"}
    probe = {"probe": "p4"}
    response = {"operation_probe_receipt": probe}
    keys = {
        purpose: (bytes([purpose]) * 32, bytes([purpose]) * 16, {})
        for purpose in (1, 2, 3, 4)
    }
    evidence = {
        "purpose4_evidence": {"request": request, "response": response},
        "operation_rows": [
            {
                "purpose_id": 4,
                "operation_id": "purpose4-parent-sign",
                "actor_receipt": probe,
                "request_binding": {"body": request},
            }
        ],
        "bridge_evidence_rows": [],
    }
    for purpose in (1, 2, 3):
        package = bytes([purpose])
        evidence["bridge_evidence_rows"].append(
            {
                "purpose_id": purpose,
                "package_base64": qualification._raw_base64(package),
                "package_size": len(package),
                "package_sha256": qualification._sha256(package),
                "replay_receipt": {},
                "bridge_signature_hex": "ab" * 64,
            }
        )
    monkeypatch.setattr(
        qualification,
        "validate_purpose4_keybearing_response_v1",
        lambda *_args, **_kwargs: SimpleNamespace(
            signer_public_key=keys[4][0], signer_key_id=keys[4][1]
        ),
    )
    monkeypatch.setattr(
        qualification,
        "replay_bridge_dag_package_v1",
        lambda package, **_kwargs: SimpleNamespace(
            purpose_id=package[0],
            authoritative=True,
            eligible_to_sign_bridge_statement=True,
            purpose1_signature_verified=package[0] != 1,
            bridge_statement_root=b"r" * 32,
        ),
    )
    monkeypatch.setattr(
        qualification, "validate_bridge_actor_replay_receipt_v1", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(qualification, "_verify_ed25519", lambda *_args: None)
    qualification._validate_purpose4_and_bridge_v1(evidence, keys)
    evidence["bridge_evidence_rows"][0]["bridge_signature_hex"] = "AB" * 64
    with pytest.raises(qualification.ActorProtocolQualificationError):
        qualification._validate_purpose4_and_bridge_v1(evidence, keys)


def _strict_evidence_fixture(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    named_images = {
        "custodian": "actor-1@sha256:" + "1" * 64,
        "python_attester": "actor-2@sha256:" + "2" * 64,
        "rust_attester": "actor-3@sha256:" + "3" * 64,
        "policy_auditor": "actor-4@sha256:" + "4" * 64,
    }
    profile_blob = json.dumps({"images": named_images}, sort_keys=True).encode("ascii")
    monkeypatch.setattr(qualification, "_git", lambda *_args, **_kwargs: profile_blob)
    monkeypatch.setattr(
        qualification,
        "local_docker_daemon_receipt_binding_v1",
        lambda _receipt: bytes.fromhex("78" * 32),
    )
    monkeypatch.setattr(
        qualification, "_validate_operation_evidence_v1", lambda _evidence: None
    )
    checks = {
        name: True
        for name in {
            "container_id_exact", "container_name_exact", "running",
            "image_reference_exact", "image_id_digest_exact",
            "user_nonroot_exact", "pid1_env_i_command_exact", "entrypoint_exact",
            "network_none", "read_only_root", "not_privileged",
            "capabilities_exact", "security_options_exact", "runtime_seccomp_exact",
            "resource_limits_exact", "nofile_exact", "ipc_private",
            "tmpfs_private_exact", "mount_set_exact",
        }
    }
    volume_checks = {
        name: True
        for name in {
            "name_exact", "driver_local", "scope_local", "options_empty",
            "labels_exact", "daemon_managed_mountpoint_exact",
            "not_bind_nfs_or_plugin",
        }
    }
    run_hex = "34" * 16
    runtime_rows = []
    ordered_images = [
        named_images["custodian"], named_images["python_attester"],
        named_images["rust_attester"], named_images["policy_auditor"],
    ]
    for purpose, image in enumerate(ordered_images, start=1):
        inspection = {
            "checks": checks,
            "container_id": str(purpose) * 64,
            "container_name": f"hegel-m25-formal-p{purpose}-" + "a" * 16,
            "host_pid": 1000 + purpose,
            "image_ref": image,
            "mount_destinations": sorted(
                ["/input", "/output", "/state"]
                + (["/custody"] if purpose == 1 else [])
            ),
            "purpose_id": purpose,
        }
        inspection["inspection_sha256"] = hashlib.sha256(
            qualification._canonical_json(inspection)
        ).hexdigest()
        volume_name = f"hegel-m25-state-{run_hex}-p{purpose}"
        volume = {
            "schema": "hegel-phase3-m25-private-volume-initialization-receipt/1",
            "basis_commit": COMMIT,
            "run_id_hex": run_hex,
            "purpose_id": purpose,
            "volume_name_sha256": hashlib.sha256(
                volume_name.encode("ascii")
            ).hexdigest(),
            "image_sha256": image.rsplit(":", 1)[-1],
            "profile_sha256": hashlib.sha256(profile_blob).hexdigest(),
            "initializer_network_none": True,
            "initializer_capabilities": ["CHOWN"],
            "nonroot_live_write_stat_probe_passed": True,
            "resulting_uid": 65534,
            "resulting_gid": 65534,
            "resulting_mode_octal": "0700",
            "volume_identity": {
                "driver": "local",
                "scope": "local",
                "options_empty": True,
                "daemon_managed_mountpoint_sha256": "9" * 64,
                "checks": volume_checks,
            },
            "daemon_receipt_sha256": "78" * 32,
        }
        volume["receipt_sha256"] = hashlib.sha256(
            qualification._canonical_json(volume)
        ).hexdigest()
        runtime_rows.append(
            {
                "purpose_id": purpose,
                "container_id": str(purpose) * 64,
                "container_inspection": inspection,
                "volume_initialization_receipt": volume,
            }
        )
    return {
        "schema": qualification.EVIDENCE_SCHEMA,
        "basis_commit": COMMIT,
        "qualification_run_id_hex": run_hex,
        "profile": {
            "profile_sha256": hashlib.sha256(profile_blob).hexdigest(),
            "images": {
                str(purpose): image
                for purpose, image in enumerate(ordered_images, start=1)
            },
            "daemon_receipt": {"stub": True},
            "daemon_receipt_sha256": "78" * 32,
            "host_repository_path_sha256": "79" * 32,
        },
        "actor_runtime_rows": runtime_rows,
        "operation_rows": [],
        "purpose4_evidence": {},
        "bridge_evidence_rows": [],
        "public_synthetic_fixture": {
            "split_frame_base64": qualification._raw_base64(
                qualification.PUBLIC_SYNTHETIC_SPLIT_FRAME
            ),
            "split_frame_size": len(qualification.PUBLIC_SYNTHETIC_SPLIT_FRAME),
            "split_frame_sha256": qualification.PUBLIC_SYNTHETIC_SPLIT_FRAME_SHA256,
            "contains_assignments": False,
            "contains_real_seed": False,
        },
        "protocol_call_graph": {
            "keygen_order": [1, 2, 3, 4],
            "purpose1_authorized_object_names": [
                "SplitSeedCommitmentManifestV1", "CustodianBindingManifestV1",
                "SeedContinuityManifestV1", "HiddenAccessLedgerRecordV1",
            ],
            "purpose4_detached_sign_count": 1,
            "bridge_replay_order": [1, 2, 3],
            "docker_seed_or_marker_method_call_count": 0,
        },
        "formal_track_claims": {
            "formal_authority": False,
            "authoritative_formal_roots_generated": False,
            "synthetic_formal_shaped_roots_computed_in_memory": True,
            "formal_roots_published": False,
            "gate_evidence_published": False,
            "formal_gates_before": 14,
            "formal_gates_after": 14,
            "m3_state": "NOT_RUN",
            "m3_started": False,
            "real_seed_generated": False,
            "real_seed_accessed": False,
        },
    }


def test_evidence_replay_requires_exact_runtime_and_volume_check_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _strict_evidence_fixture(monkeypatch)
    qualification._validate_evidence_v1(evidence, basis_commit=COMMIT)
    mutations = []
    changed = copy.deepcopy(evidence)
    changed["actor_runtime_rows"][0]["container_inspection"]["checks"]["invented"] = True
    mutations.append(changed)
    changed = copy.deepcopy(evidence)
    changed["actor_runtime_rows"][1]["volume_initialization_receipt"][
        "initializer_network_none"
    ] = False
    mutations.append(changed)
    changed = copy.deepcopy(evidence)
    changed["profile"]["images"]["3"] = "other@sha256:" + "3" * 64
    mutations.append(changed)
    for changed in mutations:
        with pytest.raises(qualification.ActorProtocolQualificationError):
            qualification._validate_evidence_v1(changed, basis_commit=COMMIT)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda report: report.update({"unexpected": True}),
        lambda report: report["authority_boundary"].update({"m3_gates_after": 24}),
        lambda report: report["independence_disclosure"].update(
            {"organizational_independence": True}
        ),
        lambda report: report.update({"commit_a_source_set_sha256": _digest("e")}),
        lambda report: report.update({"bundle_content_id": _digest("f")}),
    ),
)
def test_archive_self_hash_recomputation_does_not_bypass_policy(
    monkeypatch: pytest.MonkeyPatch, mutation
) -> None:
    report, _keys = _install_archive_stubs(monkeypatch)
    mutation(report)
    report.pop("diagnostic_report_sha256", None)
    report["diagnostic_report_sha256"] = qualification._report_hash(report)
    with pytest.raises(qualification.ActorProtocolQualificationError):
        qualification.validate_actor_protocol_qualification_report_v1(report)


def _reset_admission_globals() -> None:
    qualification._PROCESS_ADMISSION_SECRET = None
    qualification._CONSUMED_ADMISSION_MACS.clear()


def _issue_stub_token(monkeypatch: pytest.MonkeyPatch):
    _reset_admission_globals()
    replayed = _stub_replayed()
    monkeypatch.setattr(
        qualification,
        "validate_actor_protocol_qualification_report_v1",
        lambda report, *, expected_basis_commit=None: replayed
        if report == dict(replayed.report) and expected_basis_commit == COMMIT
        else pytest.fail("unexpected token archive replay"),
    )
    canonical_bundle_bytes = qualification._canonical_json(dict(replayed.report))
    live_run_nonce = b"n" * 16
    issuer_pid = os.getpid()
    qualification._PROCESS_ADMISSION_SECRET = b"s" * 32
    preimage = qualification._admission_mac_preimage_v1(
        basis_commit=replayed.basis_commit,
        bundle_content_id=replayed.bundle_content_id,
        qualification_key_ids=replayed.qualification_key_ids,
        daemon_receipt_binding=b"d" * 32,
        canonical_bundle_bytes=canonical_bundle_bytes,
        live_run_nonce=live_run_nonce,
        issuer_pid=issuer_pid,
    )
    token = qualification.LiveActorProtocolAdmissionV1(
        basis_commit=replayed.basis_commit,
        bundle_content_id=replayed.bundle_content_id,
        qualification_key_ids=replayed.qualification_key_ids,
        daemon_receipt_binding=b"d" * 32,
        canonical_bundle_bytes=canonical_bundle_bytes,
        live_run_nonce=live_run_nonce,
        issuer_pid=issuer_pid,
        token_mac=hmac.digest(
            qualification._PROCESS_ADMISSION_SECRET, preimage, "sha256"
        ),
        _seal=qualification._LIVE_ADMISSION_CONSTRUCTOR_SEAL,
    )
    return token, replayed


def test_live_token_is_opaque_hmac_authenticated_and_one_shot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token, replayed = _issue_stub_token(monkeypatch)
    assert not hasattr(token, "report")
    assert not hasattr(token, "bundle_content_id")
    assert "opaque" in repr(token)
    with pytest.raises(TypeError):
        pickle.dumps(token)
    consumed = qualification.consume_live_actor_protocol_admission_v1(
        token, expected_basis_commit=COMMIT
    )
    assert consumed.bundle_content_id == replayed.bundle_content_id
    assert consumed.canonical_bundle_bytes == qualification._canonical_json(
        dict(replayed.report)
    )
    with pytest.raises(qualification.ActorProtocolQualificationError):
        qualification.consume_live_actor_protocol_admission_v1(
            token, expected_basis_commit=COMMIT
        )


def test_live_token_rejects_field_copy_tamper_and_cross_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token, _replayed = _issue_stub_token(monkeypatch)
    with pytest.raises(qualification.ActorProtocolQualificationError):
        qualification.consume_live_actor_protocol_admission_v1(
            token, expected_basis_commit="34" * 20
        )
    object.__setattr__(token, "_bundle_content_id", b"x" * 32)
    with pytest.raises(qualification.ActorProtocolQualificationError):
        qualification.consume_live_actor_protocol_admission_v1(
            token, expected_basis_commit=COMMIT
        )


def test_live_token_constructor_and_archive_loader_cannot_mint_capability(
    monkeypatch: pytest.MonkeyPatch, linux_tmp_path: Path
) -> None:
    replayed = _stub_replayed()
    with pytest.raises(TypeError):
        qualification.LiveActorProtocolAdmissionV1(
            basis_commit=COMMIT,
            bundle_content_id=b"b" * 32,
            qualification_key_ids=replayed.qualification_key_ids,
            daemon_receipt_binding=b"d" * 32,
            canonical_bundle_bytes=qualification._canonical_json(dict(replayed.report)),
            live_run_nonce=b"n" * 16,
            issuer_pid=os.getpid(),
            token_mac=b"m" * 32,
            _seal=object(),
        )
    path = linux_tmp_path / "archive.json"
    path.write_bytes(qualification._canonical_json(dict(replayed.report)))
    _reset_admission_globals()
    monkeypatch.setattr(
        qualification,
        "validate_actor_protocol_qualification_report_v1",
        lambda *_args, **_kwargs: replayed,
    )
    monkeypatch.setattr(
        qualification.secrets,
        "token_bytes",
        lambda _size: pytest.fail("archive loading requested entropy"),
    )
    loaded = qualification.load_actor_protocol_qualification_report_v1(path)
    assert type(loaded) is qualification.ReplayedActorProtocolQualificationV1
    assert qualification._PROCESS_ADMISSION_SECRET is None
    assert not hasattr(qualification, "_issue_live_actor_protocol_admission_v1")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_live_token_is_rejected_in_fork_child(monkeypatch: pytest.MonkeyPatch) -> None:
    token, _replayed = _issue_stub_token(monkeypatch)
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            qualification.consume_live_actor_protocol_admission_v1(
                token, expected_basis_commit=COMMIT
            )
        except qualification.ActorProtocolQualificationError:
            os.write(write_fd, b"rejected")
        else:
            os.write(write_fd, b"accepted")
        finally:
            os.close(write_fd)
        os._exit(0)
    os.close(write_fd)
    result = os.read(read_fd, 32)
    os.close(read_fd)
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status)
    assert result == b"rejected"
    qualification.consume_live_actor_protocol_admission_v1(
        token, expected_basis_commit=COMMIT
    )


def _statement_fixture(monkeypatch: pytest.MonkeyPatch):
    evidence = {
        "basis_commit": COMMIT,
        "qualification_run_id_hex": "34" * 16,
        "profile": {
            "profile_sha256": "56" * 32,
            "daemon_receipt_sha256": "78" * 32,
            "host_repository_path_sha256": "79" * 32,
            "images": {
                str(purpose): f"actor-{purpose}@sha256:" + str(purpose) * 64
                for purpose in (1, 2, 3, 4)
            },
        },
        "actor_runtime_rows": [],
        "operation_rows": [],
    }
    for index, (purpose, sequence, operation) in enumerate(
        qualification.EXPECTED_OPERATION_SEQUENCE, start=1
    ):
        evidence["operation_rows"].append(
            {
                "purpose_id": purpose,
                "operation_sequence": sequence,
                "operation_id": operation,
                "operation_nonce_hex": f"{index:032x}",
            }
        )
    private_keys: dict[int, Ed25519PrivateKey] = {}
    keys: dict[int, tuple[bytes, bytes, Mapping[str, object]]] = {}
    for purpose in (1, 2, 3, 4):
        private = Ed25519PrivateKey.generate()
        public = private.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        key_id = hashlib.sha256(public).digest()[:16]
        private_keys[purpose] = private
        keys[purpose] = (
            public,
            key_id,
            {"manifest_content_id": _digest(str(purpose))},
        )
        inspection = {
            "purpose_id": purpose,
            "container_id": str(purpose) * 64,
            "checks": {"frozen": True},
        }
        evidence["actor_runtime_rows"].append(
            {
                "container_id": str(purpose) * 64,
                "container_inspection": inspection,
            }
        )
    monkeypatch.setattr(
        qualification.DockerCeremonyActorsV1,
        "_validate_common_probe_fields",
        staticmethod(lambda *_args, **_kwargs: {}),
    )
    evidence_id = _digest("a")
    plan_id = _digest("d")
    envelopes = []
    prior = {purpose: 0 for purpose in (1, 2, 3, 4)}
    for purpose, sequence, _operation in qualification.EXPECTED_OPERATION_SEQUENCE:
        prior[purpose] = max(prior[purpose], sequence)
    for purpose in (1, 2, 3, 4):
        inspection = evidence["actor_runtime_rows"][purpose - 1]["container_inspection"]
        nonce = f"{100 + purpose:032x}"
        base = {
            "schema": qualification.STATEMENT_SCHEMA,
            "operation_id": "qualification-finalize",
            "purpose_id": purpose,
            "basis_commit": COMMIT,
            "qualification_run_id_hex": evidence["qualification_run_id_hex"],
            "qualification_evidence_content_id": evidence_id,
            "destruction_plan_content_id": plan_id,
            "qualification_key_manifest_content_id": keys[purpose][2]["manifest_content_id"],
            "profile_sha256": evidence["profile"]["profile_sha256"],
            "image_ref": evidence["profile"]["images"][str(purpose)],
            "daemon_receipt_sha256": evidence["profile"]["daemon_receipt_sha256"],
            "container_id": str(purpose) * 64,
            "key_id_16_hex": keys[purpose][1].hex(),
            "operation_sequence": prior[purpose] + 1,
            "operation_nonce_hex": nonce,
            "container_inspection_sha256": qualification._sha256(
                qualification._canonical_json(inspection)
            ),
            "formal_authority": False,
            "formal_gates_before": 14,
            "formal_gates_after": 14,
            "m3_state": "NOT_RUN",
            "m3_started": False,
            "real_seed_generated": False,
            "real_seed_accessed": False,
            "formal_output_published": False,
            "qualification_identity_usage": "LIVE_PROTOCOL_QUALIFICATION_ONLY",
            "eligible_for_formal_actor_trust": False,
            "formal_genesis_reuse_forbidden": True,
            "must_destroy_after_qualification": True,
            "independence_disclosure": dict(qualification.INDEPENDENCE_DISCLOSURE),
        }
        statement = {
            **base,
            "qualification_finalize_request_sha256": qualification._content_id(
                qualification.STATEMENT_REQUEST_HASH_DOMAIN, base
            ),
        }
        preimage = qualification._qualification_statement_preimage_v1(
            purpose, statement
        )
        request = {
            "schema": "hegel-phase3-m25-protocol-qualification-finalize-request/1",
            "purpose_id": purpose,
            "statement": statement,
            "preimage_sha256": qualification._sha256(preimage),
        }
        request_hash = hashlib.sha256(qualification._canonical_json(request)).hexdigest()
        signature = private_keys[purpose].sign(preimage)
        live_probe = {
            "environment": {},
            "filesystem_probes": {},
            "identity": {},
            "implementation": "rust-ffi-v1" if purpose == 3 else "python-ctypes-v1",
            "namespaces": {},
            "network_interfaces": [],
            "open_fds": [],
            "proc_status": {},
            "profile_id": "hegel-owner-accepted-container-technical-actors-v1",
            "purpose_id": purpose,
            "schema": "hegel-container-actor-live-probe/1",
            "syscall_probes": [],
        }
        probe = {
            "live_probe_sha256": hashlib.sha256(
                qualification._canonical_json(live_probe)
            ).hexdigest(),
            "operation_id": "qualification-finalize",
            "operation_nonce_hex": nonce,
            "operation_request_sha256": request_hash,
            "operation_sequence": prior[purpose] + 1,
            "preimage_sha256": hashlib.sha256(preimage).hexdigest(),
            "purpose_id": purpose,
            "schema": "hegel-phase3-m25-protocol-qualification-finalize-probe/1",
            "signature_sha256": hashlib.sha256(signature).hexdigest(),
        }
        envelopes.append(
            {
                "schema": qualification.SIGNATURE_ENVELOPE_SCHEMA,
                "purpose_id": purpose,
                "statement": statement,
                "container_inspection": inspection,
                "finalize_request": request,
                "finalize_probe_receipt": probe,
                "finalize_live_probe_receipt": live_probe,
                "signature_hex": signature.hex(),
                "signature_verified_before_actor_destruction": True,
            }
        )
    return evidence, evidence_id, plan_id, keys, envelopes


def test_finalize_statement_worker_evidence_replays_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence, evidence_id, plan_id, keys, envelopes = _statement_fixture(monkeypatch)
    qualification._validate_statements_v1(
        envelopes,
        evidence=evidence,
        evidence_content_id=evidence_id,
        destruction_plan_content_id=plan_id,
        keys=keys,
    )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda rows: rows[0].pop("finalize_request"),
        lambda rows: rows[0]["finalize_request"].update({"purpose_id": 2}),
        lambda rows: rows[0]["finalize_probe_receipt"].update(
            {"preimage_sha256": "0" * 64}
        ),
        lambda rows: rows[0].update({"signature_hex": rows[1]["signature_hex"]}),
        lambda rows: rows[0].update(
            {"signature_hex": rows[0]["signature_hex"].upper()}
        ),
        lambda rows: rows[0]["statement"].update(
            {"qualification_evidence_content_id": _digest("f")}
        ),
    ),
)
def test_finalize_statement_rejects_missing_or_swapped_evidence(
    monkeypatch: pytest.MonkeyPatch, mutation
) -> None:
    evidence, evidence_id, plan_id, keys, envelopes = _statement_fixture(monkeypatch)
    mutation(envelopes)
    with pytest.raises(qualification.ActorProtocolQualificationError):
        qualification._validate_statements_v1(
            envelopes,
            evidence=evidence,
            evidence_content_id=evidence_id,
            destruction_plan_content_id=plan_id,
            keys=keys,
        )


def test_finalize_worker_contains_frozen_domain_and_exact_cmp() -> None:
    worker = qualification.PROJECT_ROOT.joinpath(
        "tools/phase3_m25_protocol_qualification_finalize_worker_v1.sh"
    ).read_text(encoding="utf-8")
    assert "PROTOCOL_QUALIFICATION_STATEMENT/V1\\000" in worker
    assert "PROTOCOL_QUALIFICATION_FINALIZE_SIGNATURE/V1\\000" in worker
    assert '/usr/bin/cmp -s "$expected_preimage" "$preimage"' in worker
    assert '/usr/bin/cmp -s "$expected_request" "$request"' in worker
    assert "HEGEL_HOST_REPOSITORY_PATH_SHA256" in worker
    assert '"$(/usr/bin/env | /usr/bin/wc -l)" -eq 20' in worker
    assert "host_repository_path=$HEGEL_HOST_REPOSITORY_PATH" in worker
    assert "unset HEGEL_HOST_REPOSITORY_PATH" in worker
    assert '/usr/bin/printf %s "$host_repository_path"' in worker
    assert worker.index("unset HEGEL_HOST_REPOSITORY_PATH") < worker.index(
        "/usr/bin/openssl"
    )
    assert worker.index("unset host_repository_path") < worker.index(
        "/usr/bin/openssl pkeyutl -sign"
    )
    assert "seed-split" not in worker
    finalize_source = inspect.getsource(
        qualification._finalize_qualification_statements_v1
    )
    assert "backend._actor_environment(" in finalize_source
    assert "backend._actor_launch_environment(" in finalize_source
    assert 'for key, value in launch_environment.items()' in finalize_source


def test_standalone_cli_consumes_token_and_writes_only_immutable_bytes(
    monkeypatch: pytest.MonkeyPatch, linux_tmp_path: Path
) -> None:
    tool_path = qualification.PROJECT_ROOT / "tools/phase3_m25_actor_protocol_qualification_v1.py"
    spec = importlib.util.spec_from_file_location("hegel_qualification_cli_test", tool_path)
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    opaque = object()
    consumed = SimpleNamespace(canonical_bundle_bytes=b'{"archive":true}\n')
    observed: list[object] = []
    monkeypatch.setattr(tool, "qualify_live_actor_protocol_v1", lambda **kwargs: opaque)
    monkeypatch.setattr(
        tool,
        "consume_live_actor_protocol_admission_v1",
        lambda token, *, expected_basis_commit: (
            observed.append((token, expected_basis_commit)) or consumed
        ),
    )
    monkeypatch.setattr(
        tool,
        "_exclusive_write",
        lambda path, payload: observed.append((path, payload)),
    )
    output = linux_tmp_path / "archive.json"
    assert tool.main(
        [
            "--basis-commit", COMMIT,
            "--custody-directory", str(linux_tmp_path),
            "--output", str(output),
        ]
    ) == 0
    assert observed == [
        (opaque, COMMIT),
        (output, consumed.canonical_bundle_bytes),
    ]


def test_strict_json_rejects_duplicate_keys_and_noncanonical_layout() -> None:
    with pytest.raises(qualification.ActorProtocolQualificationError):
        qualification._strict_json_object(b'{"a":1,"a":1}\n')
    with pytest.raises(qualification.ActorProtocolQualificationError):
        qualification._strict_json_object(b'{"a": 1}\n')


def test_json_transport_never_exposes_raw_bytes() -> None:
    assert qualification._json_transport({"value": b"\x01\x02"}) == {
        "value": {"bytes_hex": "0102"}
    }
