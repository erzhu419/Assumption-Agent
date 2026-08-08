from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from hegel_machine import phase3_m25_a8_recovery_amendment_r4_v1 as amendment
from hegel_machine import phase3_m25_formal_container_executor_v1 as executor


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def _runtime_identity_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Install synthetic runtime artifacts and one immutable R3.1 attempt."""

    runtime = tmp_path / "runtime"
    audit = tmp_path / "r31-audit"
    runtime.mkdir(mode=0o700)
    audit.mkdir(mode=0o700)
    formal = runtime / "hegel-formal-bridge-m25"
    bridge = runtime / "hegel-m25-bridge-dag-replay"
    report = runtime / "phase3_m25_bridge_report.json"
    formal_payload = b"formal-runtime-payload-v1"
    bridge_payload = b"bridge-runtime-payload-v1"
    formal.write_bytes(formal_payload)
    bridge.write_bytes(bridge_payload)
    formal.chmod(0o755)
    bridge.chmod(0o755)
    diagnostic_sha256 = "a7" * 32
    report_payload = _canonical(
        {"diagnostic_report_sha256": "sha256:" + diagnostic_sha256}
    )
    report.write_bytes(report_payload)
    report.chmod(0o644)

    formal_sha256 = hashlib.sha256(formal_payload).hexdigest()
    bridge_sha256 = hashlib.sha256(bridge_payload).hexdigest()
    report_sha256 = hashlib.sha256(report_payload).hexdigest()

    def binary_row(path: Path, digest: str) -> dict[str, object]:
        metadata = path.stat()
        return {
            "path": path.as_posix(),
            "sha256": digest,
            "mode_octal": "0755",
            "size_bytes": metadata.st_size,
            "st_dev": metadata.st_dev,
            "st_ino": metadata.st_ino,
        }

    report_metadata = report.stat()
    historical_rows = [
        binary_row(formal, formal_sha256),
        binary_row(bridge, bridge_sha256),
        {
            "name": report.name,
            "path": report.as_posix(),
            "mode_octal": "0644",
            "size_bytes": report_metadata.st_size,
            "st_dev": report_metadata.st_dev,
            "st_ino": report_metadata.st_ino,
            "uid": report_metadata.st_uid,
            "gid": report_metadata.st_gid,
            "raw_seed": False,
            "raw_bytes_read": True,
            "sha256_computed": True,
            "sha256": report_sha256,
        },
    ]
    attempt = amendment._r31._r2._with_receipt_sha256(
        {
            "schema": "synthetic-r31-attempt-start/1",
            "runtime_artifact_metadata": historical_rows,
        }
    )
    attempt_raw = _canonical(attempt)
    attempt_path = audit / "attempt-start.json"
    attempt_path.write_bytes(attempt_raw)
    attempt_path.chmod(0o600)

    monkeypatch.setattr(amendment, "R31_TERMINAL_AUDIT_DIRECTORY", audit)
    historical_hashes = dict(amendment.R31_TERMINAL_AUDIT_RAW_SHA256)
    historical_hashes["attempt-start.json"] = hashlib.sha256(
        attempt_raw
    ).hexdigest()
    monkeypatch.setattr(
        amendment, "R31_TERMINAL_AUDIT_RAW_SHA256", historical_hashes
    )
    monkeypatch.setattr(amendment._r31._r2, "FIXED_FORMAL_RUST_BINARY", formal)
    monkeypatch.setattr(
        amendment._r31._r2,
        "FIXED_FORMAL_RUST_BINARY_SHA256",
        formal_sha256,
    )
    monkeypatch.setattr(amendment._r31._r2, "FIXED_BRIDGE_RUST_BINARY", bridge)
    monkeypatch.setattr(
        amendment._r31._r2,
        "FIXED_BRIDGE_RUST_BINARY_SHA256",
        bridge_sha256,
    )
    monkeypatch.setattr(amendment._r31._r2, "FIXED_BRIDGE_REPORT", report)
    monkeypatch.setattr(
        amendment._r31._r2,
        "FIXED_BRIDGE_REPORT_RAW_SHA256",
        report_sha256,
    )
    monkeypatch.setattr(
        amendment._r31._r2,
        "FIXED_BRIDGE_REPORT_DIAGNOSTIC_SHA256",
        diagnostic_sha256,
    )
    monkeypatch.setattr(
        amendment._r31,
        "FIXED_RUNTIME_ARTIFACTS",
        ({"path": formal.as_posix()}, {"path": bridge.as_posix()}, {
            "path": report.as_posix()
        }),
    )
    bindings = {
        "formal_rust_replay_binary_path": formal.as_posix(),
        "formal_rust_replay_binary_sha256": formal_sha256,
        "rust_bridge_dag_replay_binary_path": bridge.as_posix(),
        "rust_bridge_dag_replay_binary_sha256": bridge_sha256,
        "rust_bridge_dag_qualification_report_path": report.as_posix(),
        "rust_bridge_dag_qualification_report_raw_sha256": report_sha256,
        "rust_bridge_dag_qualification_report_diagnostic_sha256": (
            diagnostic_sha256
        ),
    }
    return SimpleNamespace(
        audit=audit,
        formal=formal,
        bridge=bridge,
        report=report,
        formal_payload=formal_payload,
        bindings=bindings,
        historical_rows=historical_rows,
        attempt=attempt,
        attempt_raw=attempt_raw,
    )


def _validate_runtime_fixture(fixture: SimpleNamespace):
    return amendment._validate_runtime_artifacts_before_attempt_v1(
        rust_formal_replay_binary=fixture.formal,
        rust_bridge_dag_replay_binary=fixture.bridge,
        rust_bridge_dag_qualification_report=fixture.report,
        expected_bindings=fixture.bindings,
    )


def _abandoned_prefix_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Install a synthetic exact five-record abandoned-R4 prefix."""

    audit = tmp_path / "abandoned-r4-audit"
    audit.mkdir(mode=0o700)
    amendment_commit = "ab" * 20
    common = {
        "formal_repository_commit": amendment.A8_BASIS_COMMIT,
        "run_id_hex": amendment.FIXED_RUN_ID_HEX,
        "ledger_id_hex": amendment.FIXED_LEDGER_ID_HEX,
        "formal_identity_entropy_draw_count": 0,
    }
    bodies = {
        "preflight.json": {
            **common,
            "schema": "hegel-phase3-m25-a8-r4-recovery-audit-preflight/1",
            "amendment_commit": amendment_commit,
            "sole_parent_commit": amendment.R31_AMENDMENT_COMMIT,
            "parent_amendment_commit": amendment.R31_AMENDMENT_COMMIT,
            "recovery_attempt_ordinal": 4,
            "authorization_revision_id": "R4_CANONICAL_AUDIT_INSTALLER_V1",
            "fixed_audit_directory": audit.as_posix(),
            "r31_terminal_chain_root_sha256": (
                amendment.R31_TERMINAL_CHAIN_ROOT_SHA256
            ),
        },
        "incident-diagnostic.json": {
            **common,
            "schema": (
                "hegel-phase3-m25-a8-r4-recovery-audit-"
                "incident-diagnostic/1"
            ),
            "recovery_attempt_ordinal": 4,
            "authorization_revision_id": "R4_CANONICAL_AUDIT_INSTALLER_V1",
            "continuation_action": (
                "POST_R31_TERMINAL_CANONICAL_INSTALLER_"
                "RECOVERY_CONTINUATION"
            ),
            "r31_terminal_chain_root_sha256": (
                amendment.R31_TERMINAL_CHAIN_ROOT_SHA256
            ),
            "raw_seed_bytes_read_by_r4_orchestrator": False,
            "raw_seed_sha256_computed": False,
        },
        "a8-validation-receipt.json": {
            **common,
            "schema": "hegel-phase3-m25-a8-r3-a8-validation-receipt/1",
            "raw_seed_bytes_read": False,
            "raw_seed_sha256_computed": False,
            "m3_start_invoked": False,
        },
        "authorization-request.json": {
            **common,
            "schema": (
                "hegel-phase3-m25-a8-r4-recovery-audit-"
                "authorization-request/1"
            ),
            "amendment_commit": amendment_commit,
            "preflight_sha256": "",
            "incident_diagnostic_sha256": "",
            "a8_validation_receipt_sha256": "",
            "ordinary_execute_allowed": False,
            "redraw_allowed": False,
            "m3_start_allowed": False,
        },
        "authorization.json": {
            **common,
            "schema": "hegel-phase3-m25-a8-r4-recovery-audit-authorization/1",
            "amendment_commit": amendment_commit,
            "preflight_sha256": "",
            "incident_diagnostic_sha256": "",
            "a8_validation_receipt_sha256": "",
            "authorization_request_sha256": "",
            "authorization_actor": "PROJECT_OWNER",
            "owner_authorized_fixed_transaction_only": True,
            "ordinary_execute_invoked": False,
            "redraw_allowed": False,
            "m3_start_allowed": False,
        },
    }
    values: dict[str, dict[str, object]] = {}
    raws: dict[str, bytes] = {}
    for name in (
        "preflight.json",
        "incident-diagnostic.json",
        "a8-validation-receipt.json",
    ):
        values[name] = amendment._r31._r2._with_receipt_sha256(bodies[name])
        raws[name] = _canonical(values[name])
    bodies["authorization-request.json"].update(
        {
            "preflight_sha256": hashlib.sha256(raws["preflight.json"]).hexdigest(),
            "incident_diagnostic_sha256": hashlib.sha256(
                raws["incident-diagnostic.json"]
            ).hexdigest(),
            "a8_validation_receipt_sha256": hashlib.sha256(
                raws["a8-validation-receipt.json"]
            ).hexdigest(),
        }
    )
    values["authorization-request.json"] = (
        amendment._r31._r2._with_receipt_sha256(
            bodies["authorization-request.json"]
        )
    )
    raws["authorization-request.json"] = _canonical(
        values["authorization-request.json"]
    )
    bodies["authorization.json"].update(
        {
            "preflight_sha256": hashlib.sha256(raws["preflight.json"]).hexdigest(),
            "incident_diagnostic_sha256": hashlib.sha256(
                raws["incident-diagnostic.json"]
            ).hexdigest(),
            "a8_validation_receipt_sha256": hashlib.sha256(
                raws["a8-validation-receipt.json"]
            ).hexdigest(),
            "authorization_request_sha256": hashlib.sha256(
                raws["authorization-request.json"]
            ).hexdigest(),
        }
    )
    values["authorization.json"] = amendment._r31._r2._with_receipt_sha256(
        bodies["authorization.json"]
    )
    raws["authorization.json"] = _canonical(values["authorization.json"])

    order = tuple(bodies)
    rows: list[dict[str, object]] = []
    for name in order:
        path = audit / name
        path.write_bytes(raws[name])
        path.chmod(0o600)
        rows.append(
            {
                "name": name,
                "raw_sha256": hashlib.sha256(raws[name]).hexdigest(),
                "receipt_sha256": values[name]["receipt_sha256"],
                "size_bytes": len(raws[name]),
                "mode_octal": "0600",
            }
        )
    raw_hashes = {
        name: hashlib.sha256(raws[name]).hexdigest() for name in order
    }
    receipt_hashes = {
        name: str(values[name]["receipt_sha256"]) for name in order
    }
    sizes = {name: len(raws[name]) for name in order}
    prefix_root = hashlib.sha256(amendment._canonical_json(rows)).hexdigest()
    monkeypatch.setattr(amendment, "ABANDONED_R4_AUDIT_DIRECTORY", audit)
    monkeypatch.setattr(
        amendment, "ABANDONED_R4_AMENDMENT_COMMIT", amendment_commit
    )
    monkeypatch.setattr(
        amendment, "ABANDONED_R4_PREATTEMPT_AUDIT_RAW_SHA256", raw_hashes
    )
    monkeypatch.setattr(
        amendment,
        "ABANDONED_R4_PREATTEMPT_AUDIT_RECEIPT_SHA256",
        receipt_hashes,
    )
    monkeypatch.setattr(
        amendment, "ABANDONED_R4_PREATTEMPT_AUDIT_SIZE_BYTES", sizes
    )
    monkeypatch.setattr(
        amendment, "ABANDONED_R4_PREATTEMPT_PREFIX_ROOT_SHA256", prefix_root
    )
    monkeypatch.setattr(
        amendment,
        "ABANDONED_R4_AUTHORIZATION_RAW_SHA256",
        raw_hashes["authorization.json"],
    )
    return SimpleNamespace(
        audit=audit,
        order=order,
        rows=tuple(rows),
        values=values,
        raws=raws,
        prefix_root=prefix_root,
        amendment_commit=amendment_commit,
    )


def test_r4_frozen_parent_attempt_and_authorization_identity() -> None:
    assert amendment.R31_AMENDMENT_COMMIT == (
        "6c1b73064d292d57d5a9c35fd83c75caff57c300"
    )
    assert amendment.R31_TERMINAL_CHAIN_ROOT_SHA256 == (
        "d4bb2c5984405d127537bde1e973f175b630a16bcaa8ec4fe15617e665400093"
    )
    assert amendment.R31_TERMINAL_AUDIT_RAW_SHA256["attempt-start.json"] == (
        "09bbc99ad2b33930a043b0178bc5c1ebc3f71dfb09b025a412fbb00224493312"
    )
    assert amendment.R31_TERMINAL_AUDIT_RAW_SHA256["failure.json"] == (
        "90c176985d83780440007d2111577c0dc5ffbae5430eae523919653b7b6b0153"
    )
    assert amendment.OWNER_CONFIRMATION == (
        "AUTHORIZE_A8_R4_ATTEMPT_4_REVISION_2_STABLE_RUNTIME_IDENTITY_"
        "COMPLETE_ONLY_REAL_PENDING_RESUME"
    )
    assert amendment.AUTHORIZATION_REVISION_ID == (
        "R4_CANONICAL_AUDIT_INSTALLER_AND_STABLE_RUNTIME_IDENTITY_V2"
    )
    assert amendment.MANIFEST_SCHEMA == (
        "hegel-phase3-m25-a8-recovery-amendment-r4-revision-2/1"
    )
    assert amendment.ABANDONED_R4_AMENDMENT_COMMIT == (
        "6b4c0ed974e8a22663b96afc1817bbcdddc9f0a4"
    )
    assert amendment.ABANDONED_R4_PREATTEMPT_PREFIX_ROOT_SHA256 == (
        "50199f6dd2703cb5615726ed337e00ca18f0428614fa10d7a393c6dbe2f5a147"
    )
    assert amendment.FIXED_R4_AUDIT_DIRECTORY != (
        amendment.R31_TERMINAL_AUDIT_DIRECTORY
    )
    assert amendment.FIXED_R4_AUDIT_DIRECTORY != (
        amendment.ABANDONED_R4_AUDIT_DIRECTORY
    )
    assert amendment.FIXED_R4_AUDIT_DIRECTORY.name.endswith("revision-2")


def test_live_r31_terminal_chain_is_exact_and_attempt3_consumed() -> None:
    rows = amendment._r31_terminal_chain_snapshot_v1()
    assert [row["name"] for row in rows] == [
        "preflight.json",
        "incident-diagnostic.json",
        "a8-validation-receipt.json",
        "authorization-request.json",
        "authorization.json",
        "attempt-start.json",
        "failure.json",
    ]
    assert hashlib.sha256(amendment._canonical_json(rows)).hexdigest() == (
        amendment.R31_TERMINAL_CHAIN_ROOT_SHA256
    )
    assert not (
        amendment.R31_TERMINAL_AUDIT_DIRECTORY / "admission.json"
    ).exists()
    assert not (
        amendment.R31_TERMINAL_AUDIT_DIRECTORY / "finalize.json"
    ).exists()
    assert not any(
        path.name.endswith(".next")
        for path in amendment.R31_TERMINAL_AUDIT_DIRECTORY.iterdir()
    )


def test_real_r31_attempt_false_negative_is_only_runtime_metadata_shape() -> None:
    audit = amendment.R31_TERMINAL_AUDIT_DIRECTORY
    attempt_path = audit / "attempt-start.json"
    if not attempt_path.is_file():
        pytest.skip("fixed R3.1 terminal audit is not present")
    stored = json.loads(attempt_path.read_bytes())
    historical_rows = stored["runtime_artifact_metadata"]
    runtime_rows = tuple(
        dict(row) for row in historical_rows
    )
    rebuilt = dict(stored)
    rebuilt["runtime_artifact_metadata"] = runtime_rows
    assert type(historical_rows) is list
    assert type(runtime_rows) is tuple
    assert stored != rebuilt
    assert _canonical(stored) == _canonical(rebuilt)
    assert [key for key in stored if stored[key] != rebuilt[key]] == [
        "runtime_artifact_metadata"
    ]


def test_r4_revision2_incident_is_inode_independent_and_history_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runtime_identity_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        amendment._r31,
        "_build_incident_diagnostic_v1",
        lambda **_kwargs: {
            "runtime_artifact_bindings": dict(fixture.bindings),
            "raw_seed_bytes_read_by_r3_orchestrator": False,
        },
    )
    monkeypatch.setattr(
        amendment,
        "_r31_terminal_chain_snapshot_v1",
        lambda: ({"name": "frozen-r31-chain"},),
    )
    monkeypatch.setattr(
        amendment,
        "_abandoned_r4_preattempt_prefix_snapshot_v1",
        lambda: ({"name": "frozen-abandoned-r4-prefix"},),
    )

    first = amendment._build_incident_diagnostic_v1(
        custody_directory=tmp_path / "custody",
        public_evidence_path=tmp_path / "evidence.json",
        public_promotion_path=tmp_path / "promotion.json",
    )
    first_inode = fixture.formal.stat().st_ino
    replacement = fixture.formal.with_name("formal-replacement")
    replacement.write_bytes(fixture.formal_payload)
    replacement.chmod(0o755)
    os.replace(replacement, fixture.formal)
    assert fixture.formal.stat().st_ino != first_inode
    second = amendment._build_incident_diagnostic_v1(
        custody_directory=tmp_path / "custody",
        public_evidence_path=tmp_path / "evidence.json",
        public_promotion_path=tmp_path / "promotion.json",
    )

    assert amendment._canonical_json(first) == amendment._canonical_json(second)
    assert first["r31_historical_defect_proof_uses_live_runtime"] is False
    assert first["r31_attempt_start_representation_mismatch_fields"] == (
        "runtime_artifact_metadata",
    )
    assert first["runtime_long_lived_identity_excludes"] == ["st_dev", "st_ino"]
    assert all(
        "st_dev" not in row and "st_ino" not in row
        for row in first["live_runtime_stable_projection"]
    )


def test_descriptor_bound_runtime_identity_accepts_content_identical_new_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runtime_identity_fixture(tmp_path, monkeypatch)
    first = _validate_runtime_fixture(fixture)
    replacement = fixture.formal.with_name("formal-replacement")
    replacement.write_bytes(fixture.formal_payload)
    replacement.chmod(0o755)
    os.replace(replacement, fixture.formal)
    second = _validate_runtime_fixture(fixture)

    assert first[0]["st_ino"] != second[0]["st_ino"]
    assert amendment._stable_runtime_projection_v1(first) == (
        amendment._stable_runtime_projection_v1(second)
    )
    assert second[0]["st_ino"] == fixture.formal.stat().st_ino
    assert second[0]["sha256"] == fixture.historical_rows[0]["sha256"]
    assert second[0]["uid"] == os.getuid()
    assert second[0]["gid"] == os.getgid()
    assert second[0]["nlink"] == 1


def test_actor_snapshot_rejects_wrong_binary_replaced_after_r4_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale host validation cannot authorize different snapshot bytes."""

    fixture = _runtime_identity_fixture(tmp_path, monkeypatch)
    validated = _validate_runtime_fixture(fixture)
    expected_digest = bytes.fromhex(str(validated[0]["sha256"]))

    actors = amendment.A8R1RecoveryDockerActorsV1(
        basis_commit=amendment.A8_BASIS_COMMIT,
        custody_directory=tmp_path / "custody",
        rust_formal_replay_binary=fixture.formal,
        rust_bridge_dag_replay_binary=fixture.bridge,
        rust_bridge_dag_qualification_report=fixture.report,
        timestamp=0,
    )
    actor_root = tmp_path / "actor-root"
    actor_root.mkdir(mode=0o700)
    actors._root = actor_root
    # Model the already completed actor binding that normally follows the R4
    # host validation.  The private-snapshot digest check must independently
    # reject any later path replacement before a worker can start.
    actors._bound_rust_replay_digest = expected_digest
    actors._bound_rust_bridge_dag_digest = b"b" * 32
    actors._bound_rust_bridge_dag_report_sha256 = b"q" * 32
    monkeypatch.setattr(actors, "_load_committed_profile", lambda: None)
    monkeypatch.setattr(actors, "_git_blob", lambda _relative: b"{}")
    compile_calls: list[bool] = []
    purpose4_calls: list[bool] = []
    monkeypatch.setattr(
        actors, "_compile_rust_split", lambda: compile_calls.append(True)
    )
    monkeypatch.setattr(
        actors,
        "_prepare_purpose4_detached_inputs",
        lambda: purpose4_calls.append(True),
    )

    wrong_payload = b"X" * len(fixture.formal_payload)
    replacement = fixture.formal.with_name("formal-wrong-after-validation")
    replacement.write_bytes(wrong_payload)
    replacement.chmod(0o755)
    os.replace(replacement, fixture.formal)

    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        actors._prepare_inputs()

    assert captured.value.code == executor.FAIL_CONTAINER
    assert "snapshot digest changed during copy" in captured.value.detail
    assert (
        actor_root / "purpose-3/input/rust-formal-replay"
    ).read_bytes() == wrong_payload
    assert compile_calls == []
    assert purpose4_calls == []
    assert actors._containers == {}
    assert actors._actor_start_attempted is False


@pytest.mark.parametrize(
    "mutation",
    ("symlink", "mode", "size", "content", "owner", "nlink"),
)
def test_descriptor_bound_runtime_identity_rejects_unstable_artifact(
    mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runtime_identity_fixture(tmp_path, monkeypatch)
    if mutation == "symlink":
        target = fixture.formal.with_name("formal-symlink-target")
        target.write_bytes(fixture.formal_payload)
        target.chmod(0o755)
        fixture.formal.unlink()
        fixture.formal.symlink_to(target)
    elif mutation == "mode":
        fixture.formal.chmod(0o644)
    elif mutation == "size":
        fixture.formal.write_bytes(fixture.formal_payload + b"-larger")
        fixture.formal.chmod(0o755)
    elif mutation == "content":
        fixture.formal.write_bytes(b"X" * len(fixture.formal_payload))
        fixture.formal.chmod(0o755)
    elif mutation == "owner":
        actual_uid = os.getuid()
        monkeypatch.setattr(amendment.os, "getuid", lambda: actual_uid + 1)
    elif mutation == "nlink":
        os.link(fixture.formal, fixture.formal.with_name("formal-hardlink"))
    else:  # pragma: no cover - the parametrization is closed above.
        raise AssertionError(mutation)

    with pytest.raises(amendment.A8R4RecoveryAmendmentError):
        _validate_runtime_fixture(fixture)


def test_abandoned_r4_exact_five_record_prefix_is_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _abandoned_prefix_fixture(tmp_path, monkeypatch)
    rows = amendment._abandoned_r4_preattempt_prefix_snapshot_v1()
    assert rows == fixture.rows
    assert [row["name"] for row in rows] == list(fixture.order)
    assert hashlib.sha256(amendment._canonical_json(rows)).hexdigest() == (
        fixture.prefix_root
    )
    assert not any(
        (fixture.audit / name).exists()
        for name in ("attempt-start.json", "admission.json", "failure.json", "finalize.json")
    )


@pytest.mark.parametrize(
    "mutation",
    ("content", "root", "attempt-start.json", "admission.json", "failure.json", "finalize.json"),
)
def test_abandoned_r4_prefix_tamper_or_terminal_record_is_rejected(
    mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _abandoned_prefix_fixture(tmp_path, monkeypatch)
    if mutation == "content":
        value = dict(fixture.values["authorization.json"])
        value.pop("receipt_sha256")
        value["tampered"] = True
        tampered = amendment._r31._r2._with_receipt_sha256(value)
        path = fixture.audit / "authorization.json"
        path.write_bytes(_canonical(tampered))
        path.chmod(0o600)
    elif mutation == "root":
        monkeypatch.setattr(
            amendment,
            "ABANDONED_R4_PREATTEMPT_PREFIX_ROOT_SHA256",
            "00" * 32,
        )
    else:
        path = fixture.audit / mutation
        path.write_bytes(b"{}\n")
        path.chmod(0o600)

    with pytest.raises(amendment.A8R4RecoveryAmendmentError):
        amendment._abandoned_r4_preattempt_prefix_snapshot_v1()


def test_request_and_authorization_bind_abandoned_prefix_null_attempt() -> None:
    preflight_raw = b"preflight\n"
    incident_raw = b"incident\n"
    validation_raw = b"validation\n"
    request = amendment._authorization_request_fields(
        amendment_commit="35" * 20,
        preflight_raw=preflight_raw,
        incident_raw=incident_raw,
        validation_raw=validation_raw,
    )
    request_raw = amendment._receipt_record_bytes_v1(request)
    authorization = amendment._expected_authorization_fields(
        amendment_commit="35" * 20,
        preflight_raw=preflight_raw,
        incident_raw=incident_raw,
        validation_raw=validation_raw,
        request_raw=request_raw,
    )
    expected_lineage = {
        "abandoned_r4_amendment_commit": amendment.ABANDONED_R4_AMENDMENT_COMMIT,
        "abandoned_r4_preattempt_prefix_root_sha256": (
            amendment.ABANDONED_R4_PREATTEMPT_PREFIX_ROOT_SHA256
        ),
        "abandoned_r4_authorization_raw_sha256": (
            amendment.ABANDONED_R4_AUTHORIZATION_RAW_SHA256
        ),
        "abandoned_r4_attempt_start_sha256_or_null": None,
        "abandoned_r4_ordinal4_consumed": False,
        "abandoned_r4_superseded_for_execution": True,
        "abandoned_r4_defect_code": amendment.ABANDONED_R4_DEFECT_CODE,
    }
    for record in (request, authorization):
        assert {key: record[key] for key in expected_lineage} == expected_lineage
        assert record["recovery_attempt_ordinal"] == 4
        assert record["authorization_revision_id"] == (
            amendment.AUTHORIZATION_REVISION_ID
        )
        assert record["formal_identity_entropy_draw_count"] == 0
        assert record["m3_start_allowed"] is False


@pytest.mark.parametrize(
    "name",
    ("attempt-start.json", "admission.json", "finalize.json", "failure.json"),
)
def test_r4_all_terminal_record_classes_install_by_canonical_bytes(
    name: str, tmp_path: Path
) -> None:
    audit = tmp_path / "audit"
    audit.mkdir(mode=0o700)
    fields = {
        "schema": f"test-r4-{name}/1",
        "recovery_attempt_ordinal": 4,
        "typed_rows": (
            {"purpose_id": 1, "nested": ("one", "two")},
            {"purpose_id": 2, "nested": ()},
        ),
    }
    expected, raw = amendment._r31._build_exact_audit_record_v1(fields)
    amendment._install_exact_audit_record_v1(audit / name, expected, raw)
    stored = json.loads((audit / name).read_bytes())
    assert stored != expected
    assert (audit / name).read_bytes() == raw
    assert type(stored["typed_rows"]) is list


def test_r4_incident_binds_terminal_chain_and_stays_pre_m3() -> None:
    incident = amendment._build_incident_diagnostic_v1(
        custody_directory=Path(
            "/home/erzhu419/.local/state/hegel-machine/"
            "phase3-m25-0af65964235390ce2bebefea7379eaa9c50eda24/"
            "formal-custody"
        ),
        public_evidence_path=Path(
            "/home/erzhu419/mine_code/Asumption Agent/Hegel Machine/artifacts/"
            "phase3_m25_external/formal_genesis_v2/"
            "phase3_m25_formal_gate_evidence_v1.json"
        ),
        public_promotion_path=Path(
            "/home/erzhu419/mine_code/Asumption Agent/Hegel Machine/artifacts/"
            "phase3_m25_external/formal_genesis_v2/"
            "phase3_m25_gate_promotion_v1.json"
        ),
    )
    assert incident["recovery_attempt_ordinal"] == 4
    assert incident["r31_terminal_chain_root_sha256"] == (
        amendment.R31_TERMINAL_CHAIN_ROOT_SHA256
    )
    assert incident["r31_admission_sha256_or_null"] is None
    assert incident["r31_failure_phase"] == "ATTEMPT_START_DURABILITY"
    assert incident["r31_attempt_start_representation_mismatch_fields"] == (
        "runtime_artifact_metadata",
    )
    assert incident["r31_historical_defect_proof_uses_live_runtime"] is False
    assert incident["abandoned_r4_preattempt_prefix_root_sha256"] == (
        amendment.ABANDONED_R4_PREATTEMPT_PREFIX_ROOT_SHA256
    )
    assert incident["abandoned_r4_authorization_raw_sha256"] == (
        amendment.ABANDONED_R4_AUTHORIZATION_RAW_SHA256
    )
    assert incident["abandoned_r4_attempt_start_sha256_or_null"] is None
    assert incident["abandoned_r4_ordinal4_consumed"] is False
    assert incident["raw_seed_bytes_read_by_r4_orchestrator"] is False
    assert incident["marker_state"] == "PENDING"
    assert incident["journal_state"] == "RESERVED"


def test_r4_source_admission_is_exact_ordinal4_and_enters_executor_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unchanged = {"Hegel Machine/frozen.py": "11" * 32}
    root = hashlib.sha256(amendment._executor_canonical_json(unchanged)).hexdigest()
    monkeypatch.setattr(amendment._r31, "EXPECTED_UNCHANGED_A8_INPUT_COUNT", 1)
    monkeypatch.setattr(amendment._r31, "EXPECTED_UNCHANGED_A8_INPUT_ROOT", root)
    validation = {
        "actor_report_sha256": "22" * 32,
        "errata_report_sha256": "33" * 32,
        "live_bundle_sha256": "44" * 32,
    }
    admission = amendment._build_source_admission_v1(
        amendment_commit="55" * 20,
        incident_raw=b"incident\n",
        validation_raw=b"validation\n",
        validation=validation,
        unchanged_inputs=unchanged,
    )
    assert admission["schema"] == "hegel-phase3-m25-a8-r4-source-admission/1"
    assert admission["recovery_attempt_ordinal"] == 4
    assert admission["r4_amendment_commit"] == "55" * 20
    assert admission["r31_terminal_chain_root_sha256"] == (
        amendment.R31_TERMINAL_CHAIN_ROOT_SHA256
    )
    assert admission["r31_admission_sha256_or_null"] is None
    assert admission["ordinary_execute_allowed"] is False
    assert admission["redraw_allowed"] is False
    assert admission["m3_start_allowed"] is False
    monkeypatch.setattr(executor, "_FIXED_A8_R3_UNCHANGED_INPUT_COUNT", 1)
    monkeypatch.setattr(executor, "_FIXED_A8_R3_UNCHANGED_INPUT_ROOT", root)
    assert executor._validate_recovery_source_admission_v1(
        admission,
        basis_commit=amendment.A8_BASIS_COMMIT,
        run_id=amendment.FIXED_RUN_ID,
        ledger_id=amendment.FIXED_LEDGER_ID,
    ) == admission


def test_r4_runtime_exceptions_preserve_exact_95_input_a8_closure() -> None:
    bindings = amendment._unchanged_a8_input_bindings_v1()
    assert len(bindings) == amendment._r31.EXPECTED_UNCHANGED_A8_INPUT_COUNT
    assert hashlib.sha256(
        amendment._executor_canonical_json(bindings)
    ).hexdigest() == amendment._r31.EXPECTED_UNCHANGED_A8_INPUT_ROOT
    assert all(path not in bindings for path in amendment.R4_RUNTIME_EXCEPTION_PATHS)


def test_prepare_and_authorize_r4_prefix_is_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    audit = tmp_path / "audit-r4"
    monkeypatch.setattr(amendment, "FIXED_R4_AUDIT_DIRECTORY", audit)
    preflight = {
        "schema": f"{amendment.AUDIT_SCHEMA_PREFIX}-preflight/1",
        "amendment_commit": "66" * 20,
        "sole_parent_commit": amendment.R31_AMENDMENT_COMMIT,
        "formal_repository_commit": amendment.A8_BASIS_COMMIT,
        "run_id_hex": amendment.FIXED_RUN_ID_HEX,
        "ledger_id_hex": amendment.FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 4,
    }
    runtime_rows = (
        {
            "path": "/fixed/runtime",
            "sha256": "76" * 32,
            "mode_octal": "0755",
            "size_bytes": 1,
            "uid": os.getuid(),
            "gid": os.getgid(),
            "nlink": 1,
            "st_dev": 2,
            "st_ino": 3,
        },
    )
    incident = {
        "schema": f"{amendment.AUDIT_SCHEMA_PREFIX}-incident-diagnostic/1",
        "stage_directory": "/fixed/stage",
        "runtime_artifact_bindings": {},
        "live_runtime_stable_projection": (
            amendment._stable_runtime_projection_v1(runtime_rows)
        ),
    }
    validation = {
        "schema": "hegel-phase3-m25-a8-r3-a8-validation-receipt/1",
        "receipt_sha256": "77" * 32,
    }
    validation_raw = _canonical(validation)
    monkeypatch.setattr(
        amendment,
        "inspect_r4_source_preflight_v1",
        lambda **_kwargs: dict(preflight),
    )
    monkeypatch.setattr(
        amendment,
        "_build_incident_diagnostic_v1",
        lambda **_kwargs: dict(incident),
    )
    monkeypatch.setattr(
        amendment._r31,
        "_validation_request_from_incident_v1",
        lambda _incident: ({"schema": "request"}, {}, {}, {}),
    )
    monkeypatch.setattr(
        amendment._r31,
        "_run_a8_validator_v1",
        lambda _request: (dict(validation), validation_raw),
    )
    monkeypatch.setattr(
        amendment,
        "_validate_runtime_artifacts_before_attempt_v1",
        lambda **_kwargs: runtime_rows,
    )
    monkeypatch.setattr(
        amendment,
        "_abandoned_r4_preattempt_prefix_snapshot_v1",
        lambda: ({"name": "abandoned-r4-prefix"},),
    )
    kwargs = {
        "audit_directory": audit,
        "custody_directory": tmp_path / "custody",
        "public_evidence_path": tmp_path / "evidence.json",
        "public_promotion_path": tmp_path / "promotion.json",
        "repository_root": repository,
        "manifest_path": tmp_path / "manifest.json",
    }
    amendment.prepare_fixed_a8_r4_authorization_v1(**kwargs)
    expected = {
        "preflight.json",
        "incident-diagnostic.json",
        "a8-validation-receipt.json",
        "authorization-request.json",
    }
    assert {path.name for path in audit.iterdir()} == expected
    amendment.prepare_fixed_a8_r4_authorization_v1(**kwargs)
    with pytest.raises(amendment.A8R4RecoveryAmendmentError):
        amendment.write_fixed_a8_r4_owner_authorization_v1(
            audit_directory=audit,
            owner_confirmation="WRONG",
            repository_root=repository,
        )
    amendment.write_fixed_a8_r4_owner_authorization_v1(
        audit_directory=audit,
        owner_confirmation=amendment.OWNER_CONFIRMATION,
        repository_root=repository,
    )
    assert {path.name for path in audit.iterdir()} == expected | {
        "authorization.json"
    }
    request, _request_raw = amendment._r31._r2._read_canonical_audit(
        audit / "authorization-request.json"
    )
    authorization, _authorization_raw = amendment._r31._r2._read_canonical_audit(
        audit / "authorization.json"
    )
    for record in (request, authorization):
        assert record["abandoned_r4_preattempt_prefix_root_sha256"] == (
            amendment.ABANDONED_R4_PREATTEMPT_PREFIX_ROOT_SHA256
        )
        assert record["abandoned_r4_attempt_start_sha256_or_null"] is None
        assert record["abandoned_r4_ordinal4_consumed"] is False


def test_r4_manifest_and_source_have_no_seed_or_m3_start_entrypoint() -> None:
    manifest, _raw = amendment._load_manifest(amendment.DEFAULT_MANIFEST_PATH)
    assert manifest["recovery_attempt_ordinal"] == 4
    assert manifest["sole_parent_commit"] == amendment.R31_AMENDMENT_COMMIT
    assert manifest["r31_terminal_chain_root_sha256"] == (
        amendment.R31_TERMINAL_CHAIN_ROOT_SHA256
    )
    source = Path(amendment.__file__).read_text(encoding="utf-8")
    assert "phase3-m3-start" not in source
    assert "split_master_seed.bin" not in source
    assert '"raw_seed_bytes_read_by_r4_orchestrator": False' in source
    assert '"m3_start_invoked": False' in source


def test_r4_new_audit_namespace_is_repository_external() -> None:
    repository = amendment.REPOSITORY_ROOT.resolve()
    audit = amendment.FIXED_R4_AUDIT_DIRECTORY
    assert audit != repository
    assert repository not in audit.parents
    assert audit != amendment.R31_TERMINAL_AUDIT_DIRECTORY
    if audit.exists():
        assert stat.S_IMODE(audit.stat().st_mode) == 0o700


_R4_PREFIX_INVENTORY = frozenset(
    {
        "preflight.json",
        "incident-diagnostic.json",
        "a8-validation-receipt.json",
        "authorization-request.json",
        "authorization.json",
    }
)

_R4_EXECUTE_MATRIX = {
    "attempt-start-before-link": {
        "target": "attempt-start.json",
        "timing": "before",
        "inventory": _R4_PREFIX_INVENTORY
        | {".attempt-start.json.next"},
        "failure_phase": None,
        "consumed": False,
        "public_complete": False,
        "returns_success": False,
    },
    "attempt-start-after-link": {
        "target": "attempt-start.json",
        "timing": "after",
        "inventory": _R4_PREFIX_INVENTORY
        | {"attempt-start.json", "failure.json"},
        "failure_phase": "ATTEMPT_START_DURABILITY",
        "consumed": True,
        "public_complete": False,
        "returns_success": False,
    },
    "admission-before-link": {
        "target": "admission.json",
        "timing": "before",
        "inventory": _R4_PREFIX_INVENTORY
        | {"attempt-start.json", "failure.json"},
        "failure_phase": "SOURCE_ADMISSION_DURABILITY",
        "consumed": True,
        "public_complete": False,
        "returns_success": False,
    },
    "admission-after-link": {
        "target": "admission.json",
        "timing": "after",
        "inventory": _R4_PREFIX_INVENTORY
        | {"attempt-start.json", "admission.json", "failure.json"},
        "failure_phase": "SOURCE_ADMISSION_DURABILITY",
        "consumed": True,
        "public_complete": False,
        "returns_success": False,
    },
    "complete-only-core-failure": {
        "target": None,
        "timing": None,
        "inventory": _R4_PREFIX_INVENTORY
        | {"attempt-start.json", "admission.json", "failure.json"},
        "failure_phase": "COMPLETE_ONLY_FORMAL_CORE",
        "consumed": True,
        "public_complete": False,
        "returns_success": False,
    },
    "final-public-replay-failure": {
        "target": None,
        "timing": None,
        "inventory": _R4_PREFIX_INVENTORY
        | {"attempt-start.json", "admission.json", "failure.json"},
        "failure_phase": "FINAL_PUBLIC_REPLAY",
        "consumed": True,
        "public_complete": True,
        "returns_success": False,
    },
    "finalize-before-link": {
        "target": "finalize.json",
        "timing": "before",
        "inventory": _R4_PREFIX_INVENTORY
        | {"attempt-start.json", "admission.json", "failure.json"},
        "failure_phase": "FINALIZE_DURABILITY",
        "consumed": True,
        "public_complete": True,
        "returns_success": False,
    },
    "finalize-after-link": {
        "target": "finalize.json",
        "timing": "after",
        "inventory": _R4_PREFIX_INVENTORY
        | {"attempt-start.json", "admission.json", "finalize.json"},
        "failure_phase": None,
        "consumed": True,
        "public_complete": True,
        "returns_success": True,
    },
    "failure-record-before-link": {
        "target": "failure.json",
        "timing": "before",
        "inventory": _R4_PREFIX_INVENTORY
        | {"attempt-start.json", "admission.json"},
        "failure_phase": None,
        "consumed": True,
        "public_complete": False,
        "returns_success": False,
    },
    "failure-record-after-link": {
        "target": "failure.json",
        "timing": "after",
        "inventory": _R4_PREFIX_INVENTORY
        | {"attempt-start.json", "admission.json", "failure.json"},
        "failure_phase": "COMPLETE_ONLY_FORMAL_CORE",
        "consumed": True,
        "public_complete": False,
        "returns_success": False,
    },
    "success": {
        "target": None,
        "timing": None,
        "inventory": _R4_PREFIX_INVENTORY
        | {"attempt-start.json", "admission.json", "finalize.json"},
        "failure_phase": None,
        "consumed": True,
        "public_complete": True,
        "returns_success": True,
    },
}


class _R4InjectedMatrixFailure(RuntimeError):
    pass


def _prepare_r4_execute_matrix_harness(
    *, scenario: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> SimpleNamespace:
    repository = tmp_path / "repository"
    custody = tmp_path / "custody"
    public = tmp_path / "public"
    stage = tmp_path / "stage"
    for directory in (repository, custody, public, stage):
        directory.mkdir(mode=0o700)
    audit = tmp_path / "audit-r4"
    evidence_path = public / "evidence.json"
    promotion_path = public / "promotion.json"
    actor_report = {"actor_reports": [], "technical_actor_eligible": True}
    errata_report = {
        "implementation_basis_commit": amendment.A8_BASIS_COMMIT,
        "objects": [],
    }
    preflight = {
        "schema": f"{amendment.AUDIT_SCHEMA_PREFIX}-preflight/1",
        "amendment_commit": "66" * 20,
        "sole_parent_commit": amendment.R31_AMENDMENT_COMMIT,
        "formal_repository_commit": amendment.A8_BASIS_COMMIT,
        "run_id_hex": amendment.FIXED_RUN_ID_HEX,
        "ledger_id_hex": amendment.FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 4,
        "formal_identity_entropy_draw_count": 0,
        "m3_start_allowed": False,
    }
    incident = {
        "schema": f"{amendment.AUDIT_SCHEMA_PREFIX}-incident-diagnostic/1",
        "stage_directory": stage.as_posix(),
        "runtime_artifact_bindings": [],
        "raw_seed_bytes_read_by_r4_orchestrator": False,
    }
    validation_body = {
        "schema": "hegel-phase3-m25-a8-r3-a8-validation-receipt/1",
        "actor_report_sha256": "21" * 32,
        "errata_report_sha256": "22" * 32,
        "live_bundle_sha256": "23" * 32,
        "formal_identity_entropy_draw_count": 0,
        "raw_seed_bytes_read": False,
        "raw_seed_sha256_computed": False,
        "m3_start_invoked": False,
    }
    validation = amendment._r31._r2._with_receipt_sha256(validation_body)
    validation_raw = _canonical(validation)
    validation_request = {"schema": "test-r4-validation-request/1"}
    runtime_rows = (
        {
            "diagnostic_sha256_or_null": None,
            "mode_octal": "0755",
            "path": "/fixed/hegel-formal-bridge-m25",
            "sha256": "32" * 32,
            "size_bytes": 1,
            "uid": os.getuid(),
            "gid": os.getgid(),
            "nlink": 1,
            "st_dev": 4,
            "st_ino": 5,
        },
    )
    incident["live_runtime_stable_projection"] = (
        amendment._stable_runtime_projection_v1(runtime_rows)
    )

    monkeypatch.setattr(amendment, "FIXED_R4_AUDIT_DIRECTORY", audit)
    monkeypatch.setattr(
        amendment,
        "inspect_r4_source_preflight_v1",
        lambda **_kwargs: dict(preflight),
    )
    monkeypatch.setattr(
        amendment,
        "_build_incident_diagnostic_v1",
        lambda **_kwargs: dict(incident),
    )
    monkeypatch.setattr(
        amendment,
        "_validation_request_from_incident_v1",
        lambda _incident: (
            dict(validation_request),
            dict(actor_report),
            dict(errata_report),
            {},
        ),
    )
    monkeypatch.setattr(
        amendment,
        "_run_a8_validator_v1",
        lambda _request: (dict(validation), validation_raw),
    )
    monkeypatch.setattr(
        amendment,
        "_validate_runtime_artifacts_before_attempt_v1",
        lambda **_kwargs: runtime_rows,
    )
    monkeypatch.setattr(
        amendment,
        "_abandoned_r4_preattempt_prefix_snapshot_v1",
        lambda: ({"name": "abandoned-r4-prefix"},),
    )
    amendment.prepare_fixed_a8_r4_authorization_v1(
        audit_directory=audit,
        custody_directory=custody,
        public_evidence_path=evidence_path,
        public_promotion_path=promotion_path,
        repository_root=repository,
        manifest_path=tmp_path / "unused-manifest.json",
    )
    amendment.write_fixed_a8_r4_owner_authorization_v1(
        audit_directory=audit,
        owner_confirmation=amendment.OWNER_CONFIRMATION,
        repository_root=repository,
    )

    unchanged_inputs = {"Hegel Machine/frozen.py": "31" * 32}
    unchanged_root = hashlib.sha256(
        amendment._executor_canonical_json(unchanged_inputs)
    ).hexdigest()
    monkeypatch.setattr(
        amendment._r31, "EXPECTED_UNCHANGED_A8_INPUT_COUNT", 1
    )
    monkeypatch.setattr(
        amendment._r31, "EXPECTED_UNCHANGED_A8_INPUT_ROOT", unchanged_root
    )
    monkeypatch.setattr(
        amendment,
        "_unchanged_a8_input_bindings_v1",
        lambda: dict(unchanged_inputs),
    )
    counters = {
        "actor": 0,
        "acquire": 0,
        "close": 0,
        "core": 0,
        "final": 0,
    }

    class FakeActors:
        authoritative = True

        def __init__(self, **_kwargs: object) -> None:
            counters["actor"] += 1
            self.timestamp = 0

        def close(self) -> None:
            counters["close"] += 1

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
            "actor_qualification_report": dict(actor_report),
            "errata_qualification_report": dict(errata_report),
        },
    )

    def acquire(**_kwargs: object):
        counters["acquire"] += 1
        return nullcontext(recovery)

    payload = {
        "schema": "test-r4-public-evidence/1",
        "formal_gates_after": 24,
        "child_state": "NOT_RUN",
        "m3_start_invoked": False,
    }
    promotion = {
        "schema": "test-r4-public-promotion/1",
        "gate_report": {
            "gates_after": 24,
            "child_state": "NOT_RUN",
            "m3_run_started": False,
        },
    }

    def core(**kwargs: object):
        counters["core"] += 1
        assert kwargs["complete_seed_resume_only"] is True
        guard = kwargs["source_admission_guard"]
        source_admission = guard(recovery)
        assert source_admission["recovery_attempt_ordinal"] == 4
        assert source_admission["complete_seed_resume_only"] is True
        assert source_admission["ordinary_execute_allowed"] is False
        assert source_admission["redraw_allowed"] is False
        assert source_admission["m3_start_allowed"] is False
        assert source_admission["formal_identity_entropy_draw_count"] == 0
        if scenario in {
            "complete-only-core-failure",
            "failure-record-before-link",
            "failure-record-after-link",
        }:
            raise _R4InjectedMatrixFailure("injected complete-only core failure")
        return dict(payload), dict(promotion)

    evidence_raw = _canonical(payload)
    promotion_raw = _canonical(promotion)

    def validate_final(**_kwargs: object) -> dict[str, object]:
        counters["final"] += 1
        evidence_path.write_bytes(evidence_raw)
        promotion_path.write_bytes(promotion_raw)
        if scenario == "final-public-replay-failure":
            raise _R4InjectedMatrixFailure(
                "injected final public replay failure"
            )
        return {
            "public_evidence_sha256": hashlib.sha256(evidence_raw).hexdigest(),
            "public_promotion_sha256": hashlib.sha256(promotion_raw).hexdigest(),
            "publication_receipt_sha256": "41" * 32,
            "seed_custody_verification_receipt_sha256": "42" * 32,
            "complete_marker_seed_commitment_manifest_root_hex": "43" * 32,
            "complete_marker_custodian_key_id_hex": "44" * 16,
        }

    monkeypatch.setattr(amendment, "A8R1RecoveryDockerActorsV1", FakeActors)
    monkeypatch.setattr(
        amendment, "acquire_pending_ceremony_recovery_v1", acquire
    )
    monkeypatch.setattr(
        amendment, "_continue_pre_stage_pending_recovery_core_v1", core
    )
    monkeypatch.setattr(amendment, "_validate_final_publication_v1", validate_final)

    arguments = {
        "custody_directory": custody,
        "rust_formal_replay_binary": tmp_path / "formal-rust",
        "rust_bridge_dag_replay_binary": tmp_path / "bridge-rust",
        "rust_bridge_dag_qualification_report": tmp_path / "bridge-report.json",
        "public_evidence_path": evidence_path,
        "public_promotion_path": promotion_path,
        "audit_directory": audit,
        "repository_root": repository,
        "manifest_path": tmp_path / "unused-manifest.json",
    }
    return SimpleNamespace(
        audit=audit,
        arguments=arguments,
        counters=counters,
        evidence_path=evidence_path,
        promotion_path=promotion_path,
        evidence_raw=evidence_raw,
        promotion_raw=promotion_raw,
        payload=payload,
        promotion=promotion,
    )


def _assert_r4_matrix_audit_inventory(
    audit: Path, expected_names: frozenset[str] | set[str]
) -> None:
    observed = {path.name for path in audit.iterdir()}
    assert observed == set(expected_names)
    for path in audit.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        _value, raw = amendment._r31._r2._read_canonical_audit(path)
        assert path.read_bytes() == raw


def _materialize_exact_hidden_next(path: Path, raw: bytes) -> Path:
    temporary = path.with_name("." + path.name + ".next")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            assert written > 0
            offset += written
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    amendment._r31._fsync_directory_v1(path.parent)
    assert temporary.read_bytes() == raw
    assert stat.S_IMODE(temporary.stat().st_mode) == 0o600
    return temporary


def _assert_r4_public_pair(harness: SimpleNamespace, *, complete: bool) -> None:
    evidence_exists = harness.evidence_path.exists()
    promotion_exists = harness.promotion_path.exists()
    assert evidence_exists is promotion_exists
    assert evidence_exists is complete
    if complete:
        assert harness.evidence_path.read_bytes() == harness.evidence_raw
        assert harness.promotion_path.read_bytes() == harness.promotion_raw


@pytest.mark.parametrize("scenario", tuple(_R4_EXECUTE_MATRIX))
def test_r4_execute_attempt4_failure_injection_and_one_shot_matrix(
    scenario: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = _R4_EXECUTE_MATRIX[scenario]
    harness = _prepare_r4_execute_matrix_harness(
        scenario=scenario, tmp_path=tmp_path, monkeypatch=monkeypatch
    )
    real_installer = amendment._install_exact_audit_record_v1
    injection_fired = False
    injected_hidden: tuple[Path, bytes] | None = None

    def inject_at_record_boundary(
        path: Path, expected: object, raw: bytes
    ) -> None:
        nonlocal injection_fired, injected_hidden
        if path.name == policy["target"] and not injection_fired:
            injection_fired = True
            if policy["timing"] == "after":
                real_installer(path, expected, raw)
            else:
                temporary = _materialize_exact_hidden_next(path, raw)
                injected_hidden = (temporary, raw)
            raise _R4InjectedMatrixFailure(
                f"injected {path.name} {policy['timing']} durable link"
            )
        real_installer(path, expected, raw)

    if policy["target"] is not None:
        monkeypatch.setattr(
            amendment, "_install_exact_audit_record_v1", inject_at_record_boundary
        )

    first_result = None
    first_error = None
    try:
        first_result = amendment.execute_fixed_a8_r4_recovery_v1(
            **harness.arguments
        )
    except _R4InjectedMatrixFailure as exc:
        first_error = exc

    if policy["returns_success"]:
        assert first_error is None
        assert first_result == (harness.payload, harness.promotion)
    else:
        assert first_result is None
        assert first_error is not None
    if policy["timing"] == "before":
        assert injection_fired is True
        assert injected_hidden is not None
        hidden_path, hidden_raw = injected_hidden
        if scenario == "attempt-start-before-link":
            assert hidden_path.read_bytes() == hidden_raw
            assert stat.S_IMODE(hidden_path.stat().st_mode) == 0o600
        else:
            assert not hidden_path.exists()
            assert not hidden_path.is_symlink()
    _assert_r4_matrix_audit_inventory(harness.audit, policy["inventory"])
    _assert_r4_public_pair(harness, complete=policy["public_complete"])

    authorization, _authorization_raw = amendment._r31._r2._read_canonical_audit(
        harness.audit / "authorization.json"
    )
    assert authorization["recovery_attempt_ordinal"] == 4
    assert authorization["redraw_allowed"] is False
    assert authorization["m3_start_allowed"] is False
    assert authorization["formal_identity_entropy_draw_count"] == 0

    attempt_path = harness.audit / "attempt-start.json"
    admission_path = harness.audit / "admission.json"
    failure_path = harness.audit / "failure.json"
    finalize_path = harness.audit / "finalize.json"
    if attempt_path.exists():
        attempt, attempt_raw = amendment._r31._r2._read_canonical_audit(
            attempt_path
        )
        assert attempt["recovery_attempt_ordinal"] == 4
        assert attempt["abandoned_r4_amendment_commit"] == (
            amendment.ABANDONED_R4_AMENDMENT_COMMIT
        )
        assert attempt["abandoned_r4_preattempt_prefix_root_sha256"] == (
            amendment.ABANDONED_R4_PREATTEMPT_PREFIX_ROOT_SHA256
        )
        assert attempt["abandoned_r4_authorization_raw_sha256"] == (
            amendment.ABANDONED_R4_AUTHORIZATION_RAW_SHA256
        )
        assert attempt["abandoned_r4_attempt_start_sha256_or_null"] is None
        assert attempt["abandoned_r4_ordinal4_consumed"] is False
        assert attempt["abandoned_r4_superseded_for_execution"] is True
        assert attempt["abandoned_r4_defect_code"] == (
            amendment.ABANDONED_R4_DEFECT_CODE
        )
        assert tuple(attempt["runtime_artifact_stable_projection"]) == (
            amendment._stable_runtime_projection_v1(
                tuple(attempt["runtime_artifact_metadata"])
            )
        )
        assert attempt["formal_identity_entropy_draw_count"] == 0
        assert attempt["ordinary_execute_invoked"] is False
        assert attempt["raw_seed_bytes_read_by_r4_orchestrator"] is False
        assert attempt["raw_seed_sha256_computed"] is False
        assert attempt["m3_start_invoked"] is False
    else:
        attempt_raw = None
    if admission_path.exists():
        admission, admission_raw = amendment._r31._r2._read_canonical_audit(
            admission_path
        )
        source_admission = admission["source_admission"]
        assert admission["recovery_attempt_ordinal"] == 4
        assert admission["raw_seed_bytes_read_by_r4_orchestrator"] is False
        assert admission["raw_seed_sha256_computed"] is False
        assert admission["m3_start_invoked"] is False
        assert source_admission["recovery_attempt_ordinal"] == 4
        assert source_admission["ordinary_execute_allowed"] is False
        assert source_admission["redraw_allowed"] is False
        assert source_admission["m3_start_allowed"] is False
        assert source_admission["formal_identity_entropy_draw_count"] == 0
    else:
        admission_raw = None
    if failure_path.exists():
        failure, _failure_raw = amendment._r31._r2._read_canonical_audit(
            failure_path
        )
        assert failure["recovery_attempt_ordinal"] == 4
        assert failure["failure_phase"] == policy["failure_phase"]
        assert failure["attempt_start_sha256"] == hashlib.sha256(
            attempt_raw
        ).hexdigest()
        assert failure["admission_sha256_or_null"] == (
            None
            if admission_raw is None
            else hashlib.sha256(admission_raw).hexdigest()
        )
        assert failure["formal_identity_entropy_draw_count"] == 0
        assert failure["raw_seed_bytes_read_by_r4_orchestrator"] is False
        assert failure["raw_seed_sha256_computed"] is False
        assert failure["m3_start_invoked"] is False
    else:
        assert policy["failure_phase"] is None
    if finalize_path.exists():
        finalize, _finalize_raw = amendment._r31._r2._read_canonical_audit(
            finalize_path
        )
        assert finalize["recovery_attempt_ordinal"] == 4
        assert finalize["formal_gates_after"] == 24
        assert finalize["child_state"] == "NOT_RUN"
        assert finalize["formal_identity_entropy_draw_count"] == 0
        assert finalize["raw_seed_bytes_read_by_r4_orchestrator"] is False
        assert finalize["raw_seed_sha256_computed"] is False
        assert finalize["m3_start_invoked"] is False

    actor_calls_before_retry = harness.counters["actor"]
    public_snapshot = (
        harness.evidence_path.read_bytes()
        if harness.evidence_path.exists()
        else None,
        harness.promotion_path.read_bytes()
        if harness.promotion_path.exists()
        else None,
    )
    if policy["consumed"]:
        inventory_before_retry = {
            path.name: path.read_bytes() for path in harness.audit.iterdir()
        }
        with pytest.raises(
            amendment.A8R4RecoveryAmendmentError, match="already consumed|terminal"
        ):
            amendment.execute_fixed_a8_r4_recovery_v1(**harness.arguments)
        assert harness.counters["actor"] == actor_calls_before_retry
        assert {
            path.name: path.read_bytes() for path in harness.audit.iterdir()
        } == inventory_before_retry
        assert (
            harness.evidence_path.read_bytes()
            if harness.evidence_path.exists()
            else None,
            harness.promotion_path.read_bytes()
            if harness.promotion_path.exists()
            else None,
        ) == public_snapshot
    else:
        assert scenario == "attempt-start-before-link"
        assert not attempt_path.exists()
        second_result = amendment.execute_fixed_a8_r4_recovery_v1(
            **harness.arguments
        )
        assert second_result == (harness.payload, harness.promotion)
        assert harness.counters["actor"] == actor_calls_before_retry + 1
        _assert_r4_matrix_audit_inventory(
            harness.audit,
            _R4_PREFIX_INVENTORY
            | {"attempt-start.json", "admission.json", "finalize.json"},
        )
        _assert_r4_public_pair(harness, complete=True)
