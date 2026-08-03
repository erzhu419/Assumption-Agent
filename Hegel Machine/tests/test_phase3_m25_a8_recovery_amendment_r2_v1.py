from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hegel_machine import phase3_m25_a8_recovery_amendment_r2_v1 as amendment
from hegel_machine.phase3_m25_formal_container_executor_v1 import (
    FormalContainerExecutorError,
    _canonical_json as executor_canonical_json,
    _restore,
    _transport,
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def _write_regular(path: Path, payload: bytes, mode: int = 0o600) -> None:
    path.write_bytes(payload)
    path.chmod(mode)


def _write_json(path: Path, value: object, mode: int = 0o600) -> bytes:
    payload = _canonical(value)
    _write_regular(path, payload, mode)
    return payload


def _require_posix_mode(path: Path, expected: int) -> None:
    metadata = path.stat()
    if metadata.st_mode & 0o777 != expected or metadata.st_uid != os.getuid():
        pytest.skip("test temporary filesystem does not enforce POSIX mode/owner")


def _directory_identity_snapshot(
    directory: Path,
) -> dict[str, tuple[bytes, int, int, int, int, int]]:
    snapshot: dict[str, tuple[bytes, int, int, int, int, int]] = {}
    for path in directory.iterdir():
        metadata = path.stat()
        snapshot[path.name] = (
            path.read_bytes(),
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode & 0o777,
            metadata.st_uid,
            metadata.st_gid,
        )
    return snapshot


def _with_receipt(fields: dict[str, object]) -> tuple[dict[str, object], bytes]:
    record = dict(fields)
    record["receipt_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return record, _canonical(record)


def _git(root: Path, *arguments: str) -> str:
    completed = __import__("subprocess").run(
        ["/usr/bin/git", *arguments],
        cwd=root,
        check=True,
        stdout=__import__("subprocess").PIPE,
        stderr=__import__("subprocess").PIPE,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": "/nonexistent",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_AUTHOR_NAME": "R2 Test",
            "GIT_AUTHOR_EMAIL": "r2@example.invalid",
            "GIT_COMMITTER_NAME": "R2 Test",
            "GIT_COMMITTER_EMAIL": "r2@example.invalid",
        },
    )
    return completed.stdout.decode("ascii").strip()


def test_r2_source_preflight_requires_clean_direct_child_and_exact_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init", "-q")
    source = repository / "source.py"
    source.write_bytes(b"r1\n")
    _git(repository, "add", "source.py")
    _git(repository, "commit", "-q", "-m", "r1")
    r1_commit = _git(repository, "rev-parse", "HEAD")

    r1_audit = tmp_path / "r1-audit"
    r2_audit = tmp_path / "r2-audit"
    monkeypatch.setattr(amendment, "R1_AMENDMENT_COMMIT", r1_commit)
    monkeypatch.setattr(amendment, "R1_AUDIT_DIRECTORY", r1_audit)
    monkeypatch.setattr(amendment, "FIXED_R2_AUDIT_DIRECTORY", r2_audit)

    source.write_bytes(b"r2\n")
    manifest_path = repository / "manifest.json"
    manifest = {
        "schema": amendment.MANIFEST_SCHEMA,
        "source_commit_selector": "HEAD",
        "sole_parent_commit": r1_commit,
        "formal_repository_commit": amendment.A8_BASIS_COMMIT,
        "fixed_run_id_hex": amendment.FIXED_RUN_ID_HEX,
        "fixed_ledger_id_hex": amendment.FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 2,
        "exact_changed_paths": [
            {"status": "A", "path": "manifest.json"},
            {"status": "M", "path": "source.py"},
        ],
        "source_bindings": [
            {
                "path": "source.py",
                "r1_sha256_or_null": hashlib.sha256(b"r1\n").hexdigest(),
                "r2_sha256": hashlib.sha256(b"r2\n").hexdigest(),
            }
        ],
        "complete_seed_resume_only": True,
        "formal_identity_entropy_draw_count": 0,
        "ephemeral_container_nonce_allowed": True,
        "ordinary_execute_allowed": False,
        "ordinary_recovery_cross_basis_allowed": False,
        "fixed_r1_audit_directory": r1_audit.as_posix(),
        "fixed_r2_audit_directory": r2_audit.as_posix(),
        "r1_audit_raw_sha256": amendment.R1_AUDIT_RAW_SHA256,
        "r1_failure_receipt_sha256": amendment.R1_FAILURE_RECEIPT_SHA256,
        "expected_live_bundle_sha256": amendment.EXPECTED_LIVE_BUNDLE_SHA256,
        "fixed_continuity_sha256": amendment.FIXED_CONTINUITY_SHA256,
        "continuation_action": "CODE_AMENDMENT_RECOVERY_CONTINUATION",
        "owner_confirmation": amendment.OWNER_CONFIRMATION,
        "fixed_runtime_artifacts": list(amendment.FIXED_RUNTIME_ARTIFACTS),
    }
    manifest_path.write_bytes(_canonical(manifest))
    _git(repository, "add", "source.py", "manifest.json")
    _git(repository, "commit", "-q", "-m", "r2")

    report = amendment.inspect_r2_source_preflight_v1(
        repository_root=repository,
        manifest_path=manifest_path,
    )
    assert report["sole_parent_commit"] == r1_commit
    assert report["formal_repository_commit"] == amendment.A8_BASIS_COMMIT
    assert report["recovery_attempt_ordinal"] == 2
    assert report["formal_identity_entropy_draw_count"] == 0

    source.write_bytes(b"dirty\n")
    with pytest.raises(amendment.A8R2RecoveryAmendmentError):
        amendment.inspect_r2_source_preflight_v1(
            repository_root=repository,
            manifest_path=manifest_path,
        )


def _create_r1_audit_chain(
    directory: Path,
    *,
    wrong_preflight_link: bool = False,
) -> tuple[dict[str, str], str]:
    directory.mkdir(mode=0o700)
    directory.chmod(0o700)
    _require_posix_mode(directory, 0o700)
    common = {
        "formal_repository_commit": amendment.A8_BASIS_COMMIT,
        "run_id_hex": amendment.FIXED_RUN_ID_HEX,
        "ledger_id_hex": amendment.FIXED_LEDGER_ID_HEX,
    }
    preflight, preflight_raw = _with_receipt(
        {
            **common,
            "schema": "hegel-phase3-m25-a8-recovery-audit-preflight/1",
            "amendment_commit": amendment.R1_AMENDMENT_COMMIT,
            "sole_parent_commit": amendment.A8_BASIS_COMMIT,
            "manifest_sha256": (
                "9ddd56e446e4c219840e4f8ba12f4ddc59ed32032a2c9347824b521ce52bd3df"
            ),
            "source_bindings": [{"path": "frozen-r1-source"}],
            "repository_clean": True,
            "exact_changed_paths_verified": True,
            "formal_identity_entropy_draw_count": 0,
            "ephemeral_container_nonce_allowed": True,
            "m3_start_allowed": False,
            "fixed_audit_directory": directory.as_posix(),
        }
    )
    request, request_raw = _with_receipt(
        {
            **common,
            "schema": "hegel-phase3-m25-a8-recovery-audit-authorization-request/1",
            "amendment_commit": amendment.R1_AMENDMENT_COMMIT,
            "preflight_sha256": (
                "00" * 32
                if wrong_preflight_link
                else hashlib.sha256(preflight_raw).hexdigest()
            ),
            "requested_action": "COMPLETE_ONLY_REAL_PENDING_RESUME",
            "ordinary_execute_allowed": False,
            "redraw_allowed": False,
            "abort_allowed": False,
            "poststage_recovery_allowed": False,
            "formal_identity_entropy_draw_count": 0,
        }
    )
    authorization, authorization_raw = _with_receipt(
        {
            **common,
            "schema": "hegel-phase3-m25-a8-recovery-audit-authorization/1",
            "amendment_commit": amendment.R1_AMENDMENT_COMMIT,
            "preflight_sha256": hashlib.sha256(preflight_raw).hexdigest(),
            "authorization_request_sha256": hashlib.sha256(request_raw).hexdigest(),
            "authorization_actor": "PROJECT_OWNER",
            "owner_authorized_fixed_transaction_only": True,
            "ordinary_execute_invoked": False,
            "redraw_allowed": False,
            "abort_allowed": False,
            "poststage_recovery_allowed": False,
            "formal_identity_entropy_draw_count": 0,
        }
    )
    failure, failure_raw = _with_receipt(
        {
            **common,
            "schema": "hegel-phase3-m25-a8-recovery-audit-failure/1",
            "failure_code": "FAIL_M25_FORMAL_CEREMONY_LOCKED_OR_RESERVED",
            "formal_identity_entropy_draw_count": 0,
            "raw_seed_bytes_read_by_amendment_orchestrator": False,
            "raw_seed_sha256_computed": False,
        }
    )
    rows = {
        "preflight.json": (preflight, preflight_raw),
        "authorization-request.json": (request, request_raw),
        "authorization.json": (authorization, authorization_raw),
        "failure.json": (failure, failure_raw),
    }
    for name, (_record, raw) in rows.items():
        _write_regular(directory / name, raw)
    return (
        {name: hashlib.sha256(raw).hexdigest() for name, (_record, raw) in rows.items()},
        str(failure["receipt_sha256"]),
    )


def test_r1_terminal_audit_requires_exact_four_records_and_preserves_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = tmp_path / "r1-audit"
    raw_hashes, failure_receipt = _create_r1_audit_chain(audit)
    monkeypatch.setattr(amendment, "R1_AUDIT_DIRECTORY", audit)
    monkeypatch.setattr(amendment, "R1_AUDIT_RAW_SHA256", raw_hashes)
    monkeypatch.setattr(amendment, "R1_FAILURE_RECEIPT_SHA256", failure_receipt)

    rows = amendment._r1_failure_chain_snapshot_v1()
    assert tuple(row["name"] for row in rows) == (
        "preflight.json",
        "authorization-request.json",
        "authorization.json",
        "failure.json",
    )
    before = _directory_identity_snapshot(audit)
    assert _directory_identity_snapshot(audit) == before
    _write_json(audit / "admission.json", {"forbidden": True})
    with pytest.raises(amendment.A8R2RecoveryAmendmentError):
        amendment._r1_failure_chain_snapshot_v1()
    after = {
        name: identity
        for name, identity in _directory_identity_snapshot(audit).items()
        if name in before
    }
    assert after == before


def test_r1_terminal_audit_rejects_self_consistent_but_wrong_chain_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = tmp_path / "r1-audit"
    raw_hashes, failure_receipt = _create_r1_audit_chain(
        audit,
        wrong_preflight_link=True,
    )
    monkeypatch.setattr(amendment, "R1_AUDIT_DIRECTORY", audit)
    monkeypatch.setattr(amendment, "R1_AUDIT_RAW_SHA256", raw_hashes)
    monkeypatch.setattr(amendment, "R1_FAILURE_RECEIPT_SHA256", failure_receipt)
    with pytest.raises(amendment.A8R2RecoveryAmendmentError):
        amendment._r1_failure_chain_snapshot_v1()


def test_transport_diagnostic_golden_is_sequence_only_and_bytes_hex_roundtrips() -> None:
    live = {"rows": [[index] for index in range(207)]}
    restored = _restore(live)
    diagnostics = amendment._sequence_diagnostics(live, restored)
    assert live != restored
    assert _transport(restored) == live
    assert executor_canonical_json(restored) == executor_canonical_json(live)
    assert diagnostics == {
        "sequence_representation_mismatch_count": 208,
        "mapping_key_mismatch_count": 0,
        "sequence_length_mismatch_count": 0,
        "scalar_value_mismatch_count": 0,
        "other_type_mismatch_count": 0,
    }

    raw_bytes_transport = {
        "nested": [{"bytes_hex": "ab" * 32}],
    }
    restored_bytes = _restore(raw_bytes_transport)
    assert isinstance(restored_bytes["nested"], tuple)
    assert restored_bytes["nested"][0] == bytes.fromhex("ab" * 32)
    assert _transport(restored_bytes) == raw_bytes_transport
    assert executor_canonical_json(restored_bytes) == executor_canonical_json(
        raw_bytes_transport
    )


def _build_incident_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Path]:
    custody = tmp_path / "custody"
    public = tmp_path / "public"
    custody.mkdir(mode=0o700)
    public.mkdir(mode=0o700)
    custody.chmod(0o700)
    public.chmod(0o700)
    _require_posix_mode(custody, 0o700)
    _require_posix_mode(public, 0o700)
    evidence = public / "phase3_m25_formal_gate_evidence_v1.json"
    promotion = public / "phase3_m25_gate_promotion_v1.json"
    stage = public / f".hegel-m25-stage-{amendment.FIXED_RUN_ID_HEX}"
    stage.mkdir(mode=0o700)
    stage.chmod(0o700)

    live = {"rows": [[index] for index in range(207)]}
    live_raw = _write_json(stage / "live-qualification-bundle.json", live)
    live_sha = hashlib.sha256(live_raw).hexdigest()
    prestage = {
        "live_actor_protocol_qualification_bundle": live,
        "live_actor_protocol_qualification_bundle_sha256": {
            "bytes_hex": live_sha,
        },
        "runtime_binding_fields": {
            "formal_rust_replay_binary_path": (
                amendment.FIXED_FORMAL_RUST_BINARY.as_posix()
            ),
            "formal_rust_replay_binary_sha256": {
                "bytes_hex": amendment.FIXED_FORMAL_RUST_BINARY_SHA256,
            },
            "rust_bridge_dag_replay_binary_path": (
                amendment.FIXED_BRIDGE_RUST_BINARY.as_posix()
            ),
            "rust_bridge_dag_replay_binary_sha256": {
                "bytes_hex": amendment.FIXED_BRIDGE_RUST_BINARY_SHA256,
            },
            "rust_bridge_dag_qualification_report_sha256": {
                "bytes_hex": amendment.FIXED_BRIDGE_REPORT_DIAGNOSTIC_SHA256,
            },
        },
    }
    prestage_raw = _write_json(stage / "prestage-intent.json", prestage)
    lock = {
        "schema": "hegel-phase3-m25-persistent-ceremony-lock/4",
        "basis_commit": amendment.A8_BASIS_COMMIT,
        "run_id_hex": amendment.FIXED_RUN_ID_HEX,
        "ledger_id_hex": amendment.FIXED_LEDGER_ID_HEX,
        "custody_directory": custody.resolve().as_posix(),
        "public_evidence_path": evidence.as_posix(),
        "public_promotion_path": promotion.as_posix(),
        "stage_directory_name": stage.name,
        "prestage_intent_sha256_or_null": hashlib.sha256(prestage_raw).hexdigest(),
    }
    lock_raw = _write_json(custody / "phase3_m25_ceremony.lock", lock)

    split_hex = "11" * 32
    key_hex = "22" * 16
    marker = {
        "schema": "hegel-phase3-split-seed-instantiation-marker/1",
        "state": "PENDING",
        "split_version_digest_hex": split_hex,
        "seed_commitment_manifest_root_hex_or_null": None,
        "custodian_key_id_hex": key_hex,
        "created_at_unix_seconds": 7,
    }
    marker_raw = _write_json(custody / "split_seed_instantiation.marker", marker)
    journal = {
        "schema": "hegel-phase3-m25-ceremony-transaction-journal/1",
        "basis_commit": amendment.A8_BASIS_COMMIT,
        "run_id_hex": amendment.FIXED_RUN_ID_HEX,
        "ledger_id_hex": amendment.FIXED_LEDGER_ID_HEX,
        "state": "RESERVED",
        "marker_complete": False,
        "actors_absent": False,
        "public_outputs_complete": False,
    }
    journal_raw = _write_json(stage / "transaction-journal.json", journal)

    additional_raw: dict[str, bytes] = {}
    for name in (
        "actor-trust-checkpoint.json",
        "recovery-anchor.json",
        "recovery-anchor.ready.json",
    ):
        additional_raw[name] = _write_json(stage / name, {"schema": name})

    intent = {
        "schema": "hegel-phase3-m25-seed-generation-intent/1",
        "state": "CSPRNG_CALL_COMMITTED_NO_REDRAW",
    }
    seed_intent_raw = _write_json(custody / "split_seed_generation.intent", intent)
    raw_seed = b"R2-test-seed-must-never-be-opened!"[:32]
    assert len(raw_seed) == 32
    _write_regular(custody / "split_master_seed.bin", raw_seed)
    completion = {
        "attempt": 1,
        "intent_sha256": hashlib.sha256(seed_intent_raw).hexdigest(),
        "schema": "hegel-phase3-m25-seed-generation-complete/1",
        "seed_commitment_hex": "33" * 32,
        "seed_length_bytes": 32,
    }
    seed_complete_raw = _write_json(
        custody / "split_seed_generation.complete",
        completion,
    )
    for name in (
        f"opaque-run-{amendment.FIXED_RUN_ID_HEX}.reserved",
        f"opaque-ledger-{amendment.FIXED_LEDGER_ID_HEX}.reserved",
    ):
        _write_json(custody / name, {"schema": name})

    reservation_raw: dict[str, bytes] = {}
    for output in (
        evidence,
        promotion,
        promotion.with_name(promotion.name + ".publication-receipt.json"),
    ):
        reservation = output.with_name(f".{output.name}.hegel-reserved")
        reservation_raw[reservation.name] = _write_json(
            reservation,
            {"schema": "reservation", "output_path": output.as_posix()},
        )

    continuity = {
        "phase3_m25_ceremony.lock": hashlib.sha256(lock_raw).hexdigest(),
        "split_seed_generation.intent": hashlib.sha256(seed_intent_raw).hexdigest(),
        "split_seed_generation.complete": hashlib.sha256(seed_complete_raw).hexdigest(),
        "split_seed_instantiation.marker": hashlib.sha256(marker_raw).hexdigest(),
        "transaction-journal.json": hashlib.sha256(journal_raw).hexdigest(),
        "prestage-intent.json": hashlib.sha256(prestage_raw).hexdigest(),
        "live-qualification-bundle.json": live_sha,
        **{
            name: hashlib.sha256(raw).hexdigest()
            for name, raw in additional_raw.items()
        },
        **{
            name: hashlib.sha256(raw).hexdigest()
            for name, raw in reservation_raw.items()
        },
    }
    monkeypatch.setattr(amendment, "EXPECTED_LIVE_BUNDLE_SHA256", live_sha)
    monkeypatch.setattr(amendment, "FIXED_CONTINUITY_SHA256", continuity)
    monkeypatch.setattr(amendment, "FIXED_SPLIT_VERSION_DIGEST_HEX", split_hex)
    monkeypatch.setattr(amendment, "FIXED_PENDING_CUSTODIAN_KEY_ID_HEX", key_hex)
    monkeypatch.setattr(
        amendment,
        "_r1_failure_chain_snapshot_v1",
        lambda: ({"name": "failure.json", "raw_sha256": "44" * 32},),
    )
    monkeypatch.setattr(
        amendment,
        "_docker_read_only_state_v1",
        lambda: {
            "fixed_key_volume_names": tuple(f"p{value}" for value in range(1, 5)),
            "fixed_key_volume_count": 4,
            "run_labelled_container_names": (),
            "run_labelled_container_count": 0,
            "probe_read_only": True,
            "network_operation_invoked": False,
        },
    )
    monkeypatch.setattr(amendment, "_probe_formal_lock_available_v1", lambda _path: True)
    return {
        "custody": custody,
        "public": public,
        "evidence": evidence,
        "promotion": promotion,
        "stage": stage,
        "live": stage / "live-qualification-bundle.json",
        "marker": custody / "split_seed_instantiation.marker",
        "raw_seed": custody / "split_master_seed.bin",
    }


def test_incident_golden_reads_only_raw_seed_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_incident_tree(tmp_path, monkeypatch)
    raw_seed_path = paths["raw_seed"]
    forbidden_seed = raw_seed_path.read_bytes()
    real_os_open = amendment.os.open
    real_read_bytes = Path.read_bytes
    real_open = Path.open
    real_sha256 = amendment.hashlib.sha256

    def guarded_os_open(path, *args, **kwargs):
        if isinstance(path, (str, os.PathLike)) and Path(path) == raw_seed_path:
            raise AssertionError("raw seed inode was opened")
        return real_os_open(path, *args, **kwargs)

    def guarded_read_bytes(path: Path) -> bytes:
        if path == raw_seed_path:
            raise AssertionError("raw seed bytes were read")
        return real_read_bytes(path)

    def guarded_path_open(path: Path, *args, **kwargs):
        if path == raw_seed_path:
            raise AssertionError("raw seed Path.open was invoked")
        return real_open(path, *args, **kwargs)

    def guarded_sha256(data: bytes = b"", *args, **kwargs):
        if data == forbidden_seed:
            raise AssertionError("raw seed bytes were hashed")
        return real_sha256(data, *args, **kwargs)

    monkeypatch.setattr(amendment.os, "open", guarded_os_open)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(amendment.hashlib, "sha256", guarded_sha256)
    diagnostic = amendment._build_incident_diagnostic_v1(
        custody_directory=paths["custody"],
        public_evidence_path=paths["evidence"],
        public_promotion_path=paths["promotion"],
    )
    seed_row = next(
        row
        for row in diagnostic["seed_prefix_metadata"]
        if row["name"] == "split_master_seed.bin"
    )
    assert seed_row["raw_seed"] is True
    assert seed_row["raw_bytes_read"] is False
    assert seed_row["sha256_computed"] is False
    assert "sha256" not in seed_row
    assert diagnostic["marker_state"] == "PENDING"
    assert diagnostic["journal_state"] == "RESERVED"
    assert diagnostic["transport_mismatch_diagnostics"][
        "sequence_representation_mismatch_count"
    ] == 208


@pytest.mark.parametrize(
    "tamper",
    (
        "scalar",
        "drop",
        "reorder",
        "extra_stage",
        "wrong_mode",
        "noncanonical",
        "symlink",
        "marker_state",
        "bytes_nibble",
        "wrong_sha",
    ),
)
def test_incident_drift_fails_closed_before_any_actor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    paths = _build_incident_tree(tmp_path, monkeypatch)
    live_path = paths["live"]
    if tamper in {"scalar", "drop", "reorder"}:
        value = json.loads(live_path.read_bytes())
        if tamper == "scalar":
            value["rows"][0][0] = 999
        elif tamper == "drop":
            value["rows"].pop()
        else:
            value["rows"].reverse()
        _write_json(live_path, value)
    elif tamper == "extra_stage":
        _write_json(paths["stage"] / "unexpected.json", {"unexpected": True})
    elif tamper == "wrong_mode":
        live_path.chmod(0o644)
    elif tamper == "noncanonical":
        value = json.loads(live_path.read_bytes())
        _write_regular(live_path, (json.dumps(value, indent=2) + "\n").encode("ascii"))
    elif tamper == "symlink":
        live_path.unlink()
        live_path.symlink_to(paths["stage"] / "prestage-intent.json")
    elif tamper == "marker_state":
        marker = json.loads(paths["marker"].read_bytes())
        marker["state"] = "COMPLETE"
        marker["seed_commitment_manifest_root_hex_or_null"] = "55" * 32
        _write_json(paths["marker"], marker)
    elif tamper == "bytes_nibble":
        prestage_path = paths["stage"] / "prestage-intent.json"
        prestage = json.loads(prestage_path.read_bytes())
        digest = prestage["live_actor_protocol_qualification_bundle_sha256"]
        replacement = "0" if digest["bytes_hex"][0] != "0" else "1"
        digest["bytes_hex"] = replacement + digest["bytes_hex"][1:]
        _write_json(prestage_path, prestage)
    elif tamper == "wrong_sha":
        monkeypatch.setattr(amendment, "EXPECTED_LIVE_BUNDLE_SHA256", "00" * 32)
    else:  # pragma: no cover - parameter list is closed above
        raise AssertionError(tamper)

    with pytest.raises(
        (amendment.A8R2RecoveryAmendmentError, FormalContainerExecutorError)
    ):
        amendment._build_incident_diagnostic_v1(
            custody_directory=paths["custody"],
            public_evidence_path=paths["evidence"],
            public_promotion_path=paths["promotion"],
        )


def _prepare_authorized_stub_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path, dict[str, object], dict[str, object]]:
    repository = tmp_path / "repository"
    custody = tmp_path / "custody"
    public = tmp_path / "public"
    r1_audit = tmp_path / "r1-audit"
    for directory in (repository, custody, public):
        directory.mkdir(mode=0o700)
        directory.chmod(0o700)
        _require_posix_mode(directory, 0o700)
    r1_raw_hashes, r1_failure_receipt = _create_r1_audit_chain(r1_audit)
    monkeypatch.setattr(amendment, "R1_AUDIT_RAW_SHA256", r1_raw_hashes)
    monkeypatch.setattr(
        amendment,
        "R1_FAILURE_RECEIPT_SHA256",
        r1_failure_receipt,
    )
    monkeypatch.setattr(amendment, "R1_AUDIT_DIRECTORY", r1_audit)
    r1_before = _directory_identity_snapshot(r1_audit)
    audit = tmp_path / "r2-audit"
    evidence = public / "evidence.json"
    promotion = public / "promotion.json"
    preflight = {
        "schema": f"{amendment.AUDIT_SCHEMA_PREFIX}-preflight/1",
        "amendment_commit": "66" * 20,
        "formal_repository_commit": amendment.A8_BASIS_COMMIT,
        "run_id_hex": amendment.FIXED_RUN_ID_HEX,
        "ledger_id_hex": amendment.FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 2,
    }
    r1_rows = amendment._r1_failure_chain_snapshot_v1()
    incident = {
        "schema": f"{amendment.AUDIT_SCHEMA_PREFIX}-incident-diagnostic/1",
        "formal_repository_commit": amendment.A8_BASIS_COMMIT,
        "run_id_hex": amendment.FIXED_RUN_ID_HEX,
        "ledger_id_hex": amendment.FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 2,
        "r1_failure_chain": r1_rows,
        "runtime_artifact_bindings": {
            "formal_rust_replay_binary_path": (
                amendment.FIXED_FORMAL_RUST_BINARY.as_posix()
            ),
            "formal_rust_replay_binary_sha256": (
                amendment.FIXED_FORMAL_RUST_BINARY_SHA256
            ),
            "rust_bridge_dag_replay_binary_path": (
                amendment.FIXED_BRIDGE_RUST_BINARY.as_posix()
            ),
            "rust_bridge_dag_replay_binary_sha256": (
                amendment.FIXED_BRIDGE_RUST_BINARY_SHA256
            ),
            "rust_bridge_dag_qualification_report_path": (
                amendment.FIXED_BRIDGE_REPORT.as_posix()
            ),
            "rust_bridge_dag_qualification_report_raw_sha256": (
                amendment.FIXED_BRIDGE_REPORT_RAW_SHA256
            ),
            "rust_bridge_dag_qualification_report_diagnostic_sha256": (
                amendment.FIXED_BRIDGE_REPORT_DIAGNOSTIC_SHA256
            ),
        },
    }
    monkeypatch.setattr(amendment, "FIXED_R2_AUDIT_DIRECTORY", audit)
    monkeypatch.setattr(
        amendment,
        "inspect_r2_source_preflight_v1",
        lambda **_kwargs: dict(preflight),
    )
    monkeypatch.setattr(
        amendment,
        "_build_incident_diagnostic_v1",
        lambda **_kwargs: dict(incident),
    )
    monkeypatch.setattr(
        amendment,
        "_validate_runtime_artifacts_before_attempt_v1",
        lambda **_kwargs: (),
    )
    amendment.prepare_fixed_a8_r2_authorization_v1(
        audit_directory=audit,
        custody_directory=custody,
        public_evidence_path=evidence,
        public_promotion_path=promotion,
        repository_root=repository,
        manifest_path=tmp_path / "unused-manifest.json",
    )
    amendment.write_fixed_a8_r2_owner_authorization_v1(
        audit_directory=audit,
        owner_confirmation=amendment.OWNER_CONFIRMATION,
        repository_root=repository,
    )
    assert _directory_identity_snapshot(r1_audit) == r1_before
    return audit, repository, custody, public, preflight, incident


def test_prepare_creates_fresh_audit_and_owner_authorization_is_o_excl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit, repository, custody, public, _preflight, _incident = (
        _prepare_authorized_stub_audit(tmp_path, monkeypatch)
    )
    assert audit.stat().st_mode & 0o777 == 0o700
    assert {path.name for path in audit.iterdir()} == {
        "preflight.json",
        "incident-diagnostic.json",
        "authorization-request.json",
        "authorization.json",
    }
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in audit.iterdir())
    before = {
        path.name: (path.read_bytes(), path.stat().st_ino)
        for path in audit.iterdir()
    }
    with pytest.raises(amendment.A8R2RecoveryAmendmentError):
        amendment.prepare_fixed_a8_r2_authorization_v1(
            audit_directory=audit,
            custody_directory=custody,
            public_evidence_path=public / "evidence.json",
            public_promotion_path=public / "promotion.json",
            repository_root=repository,
            manifest_path=tmp_path / "unused-manifest.json",
        )
    assert {
        path.name: (path.read_bytes(), path.stat().st_ino)
        for path in audit.iterdir()
    } == before
    with pytest.raises((FileExistsError, amendment.A8R2RecoveryAmendmentError)):
        amendment.write_fixed_a8_r2_owner_authorization_v1(
            audit_directory=audit,
            owner_confirmation=amendment.OWNER_CONFIRMATION,
            repository_root=repository,
        )


def _execute_arguments(
    *,
    audit: Path,
    repository: Path,
    custody: Path,
    public: Path,
    tmp_path: Path,
) -> dict[str, object]:
    return {
        "custody_directory": custody,
        "rust_formal_replay_binary": tmp_path / "formal-rust",
        "rust_bridge_dag_replay_binary": tmp_path / "bridge-rust",
        "rust_bridge_dag_qualification_report": tmp_path / "bridge-report.json",
        "public_evidence_path": public / "evidence.json",
        "public_promotion_path": public / "promotion.json",
        "audit_directory": audit,
        "repository_root": repository,
        "manifest_path": tmp_path / "unused-manifest.json",
    }


def test_attempt_start_precedes_actor_construction_and_consumes_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit, repository, custody, public, _preflight, _incident = (
        _prepare_authorized_stub_audit(tmp_path, monkeypatch)
    )
    constructor_calls = 0
    acquire_calls = 0

    class FailAtConstruction:
        def __init__(self, **_kwargs):
            nonlocal constructor_calls
            constructor_calls += 1
            assert (audit / "attempt-start.json").is_file()
            raise RuntimeError("injected actor construction stop")

    def should_not_acquire(**_kwargs):
        nonlocal acquire_calls
        acquire_calls += 1
        raise AssertionError("acquire must not run")

    monkeypatch.setattr(amendment, "A8R1RecoveryDockerActorsV1", FailAtConstruction)
    monkeypatch.setattr(amendment, "acquire_pending_ceremony_recovery_v1", should_not_acquire)
    arguments = _execute_arguments(
        audit=audit,
        repository=repository,
        custody=custody,
        public=public,
        tmp_path=tmp_path,
    )
    with pytest.raises(RuntimeError, match="construction"):
        amendment.execute_fixed_a8_r2_recovery_v1(**arguments)
    assert constructor_calls == 1
    assert acquire_calls == 0
    failure, _raw = amendment._read_canonical_audit(audit / "failure.json")
    assert failure["failure_phase"] == "ACTOR_CONSTRUCTION"
    assert failure["recovery_attempt_ordinal"] == 2
    assert failure["admission_sha256_or_null"] is None

    with pytest.raises(amendment.A8R2RecoveryAmendmentError):
        amendment.execute_fixed_a8_r2_recovery_v1(**arguments)
    assert constructor_calls == 1
    assert acquire_calls == 0


def test_runtime_precheck_failure_precedes_attempt_start_and_actor_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit, repository, custody, public, _preflight, _incident = (
        _prepare_authorized_stub_audit(tmp_path, monkeypatch)
    )
    r1_audit = tmp_path / "r1-audit"
    r1_before = _directory_identity_snapshot(r1_audit)
    actor_calls = 0

    def fail_runtime_precheck(**_kwargs):
        assert not (audit / "attempt-start.json").exists()
        raise amendment.A8R2RecoveryAmendmentError(
            amendment.FAIL_AMENDMENT,
            "injected runtime artifact SHA mismatch",
        )

    class MustNotConstruct:
        def __init__(self, **_kwargs):
            nonlocal actor_calls
            actor_calls += 1
            raise AssertionError("actor construction must follow runtime precheck")

    monkeypatch.setattr(
        amendment,
        "_validate_runtime_artifacts_before_attempt_v1",
        fail_runtime_precheck,
    )
    monkeypatch.setattr(amendment, "A8R1RecoveryDockerActorsV1", MustNotConstruct)
    arguments = _execute_arguments(
        audit=audit,
        repository=repository,
        custody=custody,
        public=public,
        tmp_path=tmp_path,
    )
    with pytest.raises(amendment.A8R2RecoveryAmendmentError, match="runtime artifact"):
        amendment.execute_fixed_a8_r2_recovery_v1(**arguments)
    assert actor_calls == 0
    assert {path.name for path in audit.iterdir()} == {
        "preflight.json",
        "incident-diagnostic.json",
        "authorization-request.json",
        "authorization.json",
    }
    assert _directory_identity_snapshot(r1_audit) == r1_before


def test_acquire_failure_closes_actor_and_second_call_never_reacquires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit, repository, custody, public, _preflight, _incident = (
        _prepare_authorized_stub_audit(tmp_path, monkeypatch)
    )
    acquire_calls = 0
    close_calls = 0

    class FakeActors:
        def __init__(self, **_kwargs):
            assert (audit / "attempt-start.json").is_file()

        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1

    def fail_acquire(**_kwargs):
        nonlocal acquire_calls
        acquire_calls += 1
        raise FormalContainerExecutorError("FAIL_TEST_ACQUIRE", "injected stop")

    monkeypatch.setattr(amendment, "A8R1RecoveryDockerActorsV1", FakeActors)
    monkeypatch.setattr(amendment, "acquire_pending_ceremony_recovery_v1", fail_acquire)
    arguments = _execute_arguments(
        audit=audit,
        repository=repository,
        custody=custody,
        public=public,
        tmp_path=tmp_path,
    )
    with pytest.raises(FormalContainerExecutorError):
        amendment.execute_fixed_a8_r2_recovery_v1(**arguments)
    assert acquire_calls == 1
    assert close_calls == 1
    failure, _raw = amendment._read_canonical_audit(audit / "failure.json")
    assert failure["failure_phase"] == "FORMAL_RECOVERY_ACQUIRE"

    with pytest.raises(amendment.A8R2RecoveryAmendmentError):
        amendment.execute_fixed_a8_r2_recovery_v1(**arguments)
    assert acquire_calls == 1
    assert close_calls == 1


def test_success_finalize_binds_attempt_and_r1_failure_then_is_one_shot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit, repository, custody, public, _preflight, incident = (
        _prepare_authorized_stub_audit(tmp_path, monkeypatch)
    )
    stage = public / "stage"
    stage.mkdir()
    r1_audit = tmp_path / "r1-audit"
    r1_before = _directory_identity_snapshot(r1_audit)
    acquire_calls = 0
    close_calls = 0

    class FakeActors:
        def __init__(self, **_kwargs):
            self.timestamp = 0
            assert (audit / "attempt-start.json").is_file()

        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1

    marker = SimpleNamespace(state="PENDING", created_at_unix_seconds=7)
    recovery = SimpleNamespace(
        basis_commit=amendment.A8_BASIS_COMMIT,
        run_id=amendment.FIXED_RUN_ID,
        ledger_id=amendment.FIXED_LEDGER_ID,
        marker_snapshot=marker,
        journal_state="RESERVED",
        custody_directory=custody,
        stage_directory=stage,
        prestage_intent_fields={
            "actor_qualification_report": {},
            "errata_qualification_report": {
                "implementation_basis_commit": amendment.A8_BASIS_COMMIT,
            },
        },
    )

    def acquire(**_kwargs):
        nonlocal acquire_calls
        acquire_calls += 1
        return nullcontext(recovery)

    payload = {"evidence": True}
    promotion = {"promotion": True}

    def core(**kwargs):
        admission = kwargs["source_admission_guard"](recovery)
        assert admission["recovery_attempt_ordinal"] == 2
        assert admission["r1_failure_raw_sha256"] == amendment.R1_AUDIT_RAW_SHA256[
            "failure.json"
        ]
        return payload, promotion

    monkeypatch.setattr(amendment, "A8R1RecoveryDockerActorsV1", FakeActors)
    monkeypatch.setattr(amendment, "acquire_pending_ceremony_recovery_v1", acquire)
    monkeypatch.setattr(amendment, "_seed_prefix_stat_only_snapshot", lambda _path: ())
    monkeypatch.setattr(amendment, "REQUIRED_COMMIT_A_INPUTS", ())
    monkeypatch.setattr(
        amendment,
        "validate_ceremony_admission_v1",
        lambda **_kwargs: {"input_sha256": {"basis": "88" * 32}},
    )
    monkeypatch.setattr(
        amendment,
        "validate_qualification_report",
        lambda _report: {
            "basis_commit": amendment.A8_BASIS_COMMIT,
            "technical_actor_eligible": True,
        },
    )
    monkeypatch.setattr(
        amendment,
        "validate_dual_errata_qualification_report",
        lambda _report: None,
    )
    monkeypatch.setattr(
        amendment,
        "_continue_pre_stage_pending_recovery_core_v1",
        core,
    )
    monkeypatch.setattr(
        amendment,
        "_validate_final_publication_v1",
        lambda **_kwargs: {
            "public_evidence_sha256": "91" * 32,
            "public_promotion_sha256": "92" * 32,
            "publication_receipt_sha256": "93" * 32,
            "seed_custody_verification_receipt_sha256": "94" * 32,
            "complete_marker_seed_commitment_manifest_root_hex": "95" * 32,
            "complete_marker_custodian_key_id_hex": "96" * 16,
        },
    )
    arguments = _execute_arguments(
        audit=audit,
        repository=repository,
        custody=custody,
        public=public,
        tmp_path=tmp_path,
    )
    observed_payload, observed_promotion = amendment.execute_fixed_a8_r2_recovery_v1(
        **arguments
    )
    assert observed_payload == payload
    assert observed_promotion == promotion
    assert acquire_calls == 1
    assert close_calls == 1
    finalize, _raw = amendment._read_canonical_audit(audit / "finalize.json")
    attempt_raw = (audit / "attempt-start.json").read_bytes()
    assert finalize["recovery_attempt_ordinal"] == 2
    assert finalize["r1_failure_raw_sha256"] == amendment.R1_AUDIT_RAW_SHA256[
        "failure.json"
    ]
    assert finalize["r1_failure_receipt_sha256"] == amendment.R1_FAILURE_RECEIPT_SHA256
    assert finalize["attempt_start_sha256"] == hashlib.sha256(attempt_raw).hexdigest()
    assert finalize["formal_gates_after"] == 24
    assert finalize["child_state"] == "NOT_RUN"
    assert finalize["m3_start_invoked"] is False
    assert _directory_identity_snapshot(r1_audit) == r1_before

    with pytest.raises(amendment.A8R2RecoveryAmendmentError):
        amendment.execute_fixed_a8_r2_recovery_v1(**arguments)
    assert acquire_calls == 1
    assert close_calls == 1
    assert _directory_identity_snapshot(r1_audit) == r1_before


def test_final_publication_requires_24_not_run_and_all_15_outputs_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    custody = tmp_path / "custody"
    stage = tmp_path / "stage"
    public = tmp_path / "public"
    for directory in (custody, stage, public):
        directory.mkdir()
    evidence_path = public / "evidence.json"
    promotion_path = public / "promotion.json"
    payload = {"evidence": True}
    promotion = {
        "gate_report": {
            "gates_after": 24,
            "all_gates_15_24_passed": True,
            "child_state": "NOT_RUN",
            "m3_run_started": False,
        }
    }
    evidence_raw = executor_canonical_json(payload)
    promotion_raw = executor_canonical_json(promotion)
    evidence_path.write_bytes(evidence_raw)
    promotion_path.write_bytes(promotion_raw)
    seed_verification_raw = _canonical({"seed_verification": True})
    (stage / "seed-custody-verification.json").write_bytes(seed_verification_raw)
    receipt = {
        "schema": "hegel-phase3-m25-publication-receipt/1",
        "basis_commit": amendment.A8_BASIS_COMMIT,
        "run_id_hex": amendment.FIXED_RUN_ID_HEX,
        "ledger_id_hex": amendment.FIXED_LEDGER_ID_HEX,
        "public_evidence_sha256": hashlib.sha256(evidence_raw).hexdigest(),
        "public_promotion_sha256": hashlib.sha256(promotion_raw).hexdigest(),
        "seed_custody_verification_receipt_sha256_or_null": hashlib.sha256(
            seed_verification_raw
        ).hexdigest(),
        "prospective_public_replay_passed": True,
        "marker_was_complete_during_staging": False,
        "actor_cleanup_required_before_publication": True,
        "authority_disclosure": dict(amendment.TECHNICAL_ACTOR_DISCLOSURE_V1),
        "contains_private_key": False,
        "contains_raw_split_seed": False,
        "contains_split_assignment_rows": False,
    }
    receipt_path = promotion_path.with_name(
        promotion_path.name + ".publication-receipt.json"
    )
    receipt_path.write_bytes(_canonical(receipt))

    marker = SimpleNamespace(
        state="COMPLETE",
        seed_commitment_manifest_root=b"r" * 32,
        custodian_key_id=b"k" * 16,
    )
    null_roots = {
        f"{name}_or_null": None for name in amendment.M3_RUN_OUTPUT_ROOTS
    }
    inputs = SimpleNamespace(
        run_genesis_fields=null_roots,
        marker_snapshot=marker,
    )
    monkeypatch.setattr(
        amendment,
        "replay_public_gate_evidence_v1",
        lambda _payload: promotion,
    )
    monkeypatch.setattr(
        amendment,
        "load_gate_evidence_inputs_v1",
        lambda _payload: inputs,
    )
    monkeypatch.setattr(amendment, "read_marker_snapshot_v1", lambda _path: marker)
    result = amendment._validate_final_publication_v1(
        payload=payload,
        promotion=promotion,
        custody_directory=custody,
        stage_directory=stage,
        public_evidence_path=evidence_path,
        public_promotion_path=promotion_path,
    )
    assert result["public_evidence_sha256"] == hashlib.sha256(evidence_raw).hexdigest()

    inputs.run_genesis_fields = dict(null_roots)
    first = next(iter(inputs.run_genesis_fields))
    inputs.run_genesis_fields[first] = b"non-null"
    with pytest.raises(amendment.A8R2RecoveryAmendmentError):
        amendment._validate_final_publication_v1(
            payload=payload,
            promotion=promotion,
            custody_directory=custody,
            stage_directory=stage,
            public_evidence_path=evidence_path,
            public_promotion_path=promotion_path,
        )
