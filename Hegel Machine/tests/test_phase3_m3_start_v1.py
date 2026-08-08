from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from types import MappingProxyType

import pytest

import hegel_machine.phase3_m3_start_v1 as start_module
import hegel_machine.phase3_m3_local_admission_v1 as local_admission_module
from hegel_machine.phase3_m25_wire_v1 import decode_formal_object
from hegel_machine.phase3_m3_start_v1 import (
    FAIL_ALREADY_STARTED,
    FAIL_JSON,
    M3StartError,
    PreparedM3StartV1,
    canonical_start_publication_receipt_path_v1,
    load_publication_blobs_v1,
    prepare_authoritative_m3_start_v1,
    prepare_m3_start_v1,
    read_start_publication_receipt_v1,
    read_state_file_v1,
    strict_json_loads_v1,
    validate_state_document_v1,
    verify_m3_start_v1,
    write_state_exact_once_v1,
)
from hegel_machine.phase3_m25_commit_b_publication_audit_v1 import canonical_json_v1
from hegel_machine import phase3_m3_start_cli_v1 as start_cli


RECORDED_AT = 1_785_779_400


def test_formal_run_parent_is_not_home_environment_derived() -> None:
    assert start_module.FORMAL_RUN_PARENT == Path(
        "/home/erzhu419/.local/state/hegel-machine"
    )
RUNTIME_COMMIT_C = "c" * 40
APPROVAL_COMMIT_D = "d" * 40


@pytest.fixture
def linux_tmp_path() -> Path:
    with tempfile.TemporaryDirectory(prefix="hegel-m3-start-test-", dir="/tmp") as raw:
        path = Path(raw)
        path.chmod(0o700)
        yield path


@lru_cache(maxsize=1)
def _prepared() -> tuple[
    str, bytes, bytes, PreparedM3StartV1, dict[str, object]
]:
    commit, evidence, promotion = load_publication_blobs_v1()
    prepared = prepare_authoritative_m3_start_v1(
        evidence,
        promotion,
        publication_commit=commit,
        recorded_at_unix_seconds=RECORDED_AT,
    )
    return commit, evidence, promotion, prepared, dict(prepared.document)


def _local_admission(
    report: dict[str, object],
) -> local_admission_module.LocalTwoCommitAdmissionResultV1:
    return local_admission_module.LocalTwoCommitAdmissionResultV1(
        runtime_commit_c=RUNTIME_COMMIT_C,
        approval_commit_d=APPROVAL_COMMIT_D,
        artifact_fields=MappingProxyType(
            {
                "publication_commit_b": report["publication_commit"],
                "basis_commit_a": report["basis_commit"],
            }
        ),
        manifest_fields=MappingProxyType(
            {"schema": "hegel-m3-runtime-source-manifest/1"}
        ),
        receipt_fields=MappingProxyType(
            {
                "runtime_commit_c": RUNTIME_COMMIT_C,
                "approval_commit_d": APPROVAL_COMMIT_D,
                "formal_run_id_hex": report["run_id_hex"],
                "execution_manifest_root_hex": report[
                    "execution_manifest_root_hex"
                ],
            }
        ),
    )


def test_prepare_builds_only_the_exact_index_zero_start_record() -> None:
    commit, _evidence, _promotion, _prepared_start, report = _prepared()
    validate_state_document_v1(report)
    decoded = decode_formal_object(
        bytes.fromhex(report["state_record_cbor_hex"]),
        expected_name="M3RunStateRecordV1",
    )
    assert report["publication_commit"] == commit
    assert report["formal_gate_count"] == 24
    assert report["child_state_before"] == "NOT_RUN"
    assert report["child_state_after"] == "RUNNING"
    assert report["running_phase_after"] == "CANONICAL_ENUMERATION"
    assert report["closure_invoked"] is False
    assert dict(decoded.fields) == {
        "run_id": bytes.fromhex(report["run_id_hex"]),
        "transition_index": 0,
        "previous_state_record_root_or_null": None,
        "from_state_id": 0,
        "from_phase_id": 0,
        "to_state_id": 1,
        "to_phase_id": 1,
        "transition_reason_id": 1,
        "execution_manifest_root": bytes.fromhex(
            report["execution_manifest_root_hex"]
        ),
        "triggering_receipt_root_or_null": None,
        "recorded_at_unix_seconds": RECORDED_AT,
    }


def test_exact_once_writer_is_atomic_idempotent_and_never_overwrites(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _commit, _evidence, _promotion, prepared, report = _prepared()
    state_parent = (linux_tmp_path / "state-parent").resolve()
    state_parent.mkdir(mode=0o700)
    monkeypatch.setattr(start_module, "FORMAL_RUN_PARENT", state_parent)
    monkeypatch.setattr(
        start_module._local_admission,
        "validate_local_admission_receipt_v1",
        lambda *_args, **_kwargs: None,
    )
    local_admission = _local_admission(report)
    state = start_module.canonical_start_state_path_v1(report["run_id_hex"])
    assert (
        write_state_exact_once_v1(
            state,
            prepared,
            local_admission=local_admission,
        )
        == "STARTED_NEW"
    )
    first = state.read_bytes()
    receipt_path = canonical_start_publication_receipt_path_v1(
        report["run_id_hex"]
    )
    receipt = read_start_publication_receipt_v1(state, report)
    assert receipt_path.exists()
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
    assert receipt["action_id"] == "phase3-m3-start"
    assert receipt["state_file_sha256"] == hashlib.sha256(first).hexdigest()
    assert (
        write_state_exact_once_v1(
            state,
            prepared,
            local_admission=local_admission,
        )
        == "ALREADY_STARTED_IDENTICAL"
    )
    assert state.read_bytes() == first == canonical_json_v1(report)
    assert not [
        path for path in state.parent.iterdir() if path.name.endswith(".pending")
    ]

    occupied = state
    occupied.write_bytes(b"different\n")
    with pytest.raises(M3StartError) as captured:
        write_state_exact_once_v1(
            occupied,
            prepared,
            local_admission=local_admission,
        )
    assert captured.value.code == FAIL_ALREADY_STARTED
    assert occupied.read_bytes() == b"different\n"


def test_exact_once_writer_recovers_verified_post_link_crash_hardlinks(
    linux_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _commit, _evidence, _promotion, prepared, report = _prepared()
    state_parent = (linux_tmp_path / "state-parent").resolve()
    state_parent.mkdir(mode=0o700)
    monkeypatch.setattr(start_module, "FORMAL_RUN_PARENT", state_parent)
    monkeypatch.setattr(
        start_module._local_admission,
        "validate_local_admission_receipt_v1",
        lambda *_args, **_kwargs: None,
    )
    local_admission = _local_admission(report)
    state = start_module.canonical_start_state_path_v1(report["run_id_hex"])
    assert (
        write_state_exact_once_v1(
            state,
            prepared,
            local_admission=local_admission,
        )
        == "STARTED_NEW"
    )

    # Inject the exact namespace state left by a crash after linkat(state) but
    # before unlinkat(pending): both names address one verified mode-0600 inode.
    state_pending = state.parent / (
        f".{state.name}.{report['state_record_root_hex']}.pending"
    )
    os.link(state, state_pending)
    assert state.stat().st_nlink == state_pending.stat().st_nlink == 2
    assert (
        write_state_exact_once_v1(
            state,
            prepared,
            local_admission=local_admission,
        )
        == "ALREADY_STARTED_IDENTICAL"
    )
    assert not state_pending.exists()
    assert state.stat().st_nlink == 1

    # The sidecar uses a random per-writer pending suffix, so recovery must
    # discover exactly one matching same-inode alias without touching others.
    receipt = canonical_start_publication_receipt_path_v1(report["run_id_hex"])
    receipt_pending = receipt.parent / (
        f".{receipt.name}.4242.{'a' * 32}.pending"
    )
    os.link(receipt, receipt_pending)
    assert receipt.stat().st_nlink == receipt_pending.stat().st_nlink == 2
    assert (
        write_state_exact_once_v1(
            state,
            prepared,
            local_admission=local_admission,
        )
        == "ALREADY_STARTED_IDENTICAL"
    )
    assert not receipt_pending.exists()
    assert receipt.stat().st_nlink == 1
    assert read_start_publication_receipt_v1(state, report)["action_id"] == (
        "phase3-m3-start"
    )


def test_state_hardlink_recovery_rejects_and_preserves_unknown_files(
    linux_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _commit, _evidence, _promotion, prepared, report = _prepared()
    state_parent = (linux_tmp_path / "state-parent").resolve()
    state_parent.mkdir(mode=0o700)
    monkeypatch.setattr(start_module, "FORMAL_RUN_PARENT", state_parent)
    monkeypatch.setattr(
        start_module._local_admission,
        "validate_local_admission_receipt_v1",
        lambda *_args, **_kwargs: None,
    )
    local_admission = _local_admission(report)
    state = start_module.canonical_start_state_path_v1(report["run_id_hex"])
    assert (
        write_state_exact_once_v1(
            state,
            prepared,
            local_admission=local_admission,
        )
        == "STARTED_NEW"
    )

    unknown_alias = state.parent / "unknown-state-hardlink"
    os.link(state, unknown_alias)
    reserved_pending = state.parent / (
        f".{state.name}.{report['state_record_root_hex']}.pending"
    )
    reserved_pending.write_bytes(state.read_bytes())
    reserved_pending.chmod(0o600)
    with pytest.raises(M3StartError) as captured:
        write_state_exact_once_v1(
            state,
            prepared,
            local_admission=local_admission,
        )
    assert captured.value.code == start_module.FAIL_STATE_IO
    assert unknown_alias.exists()
    assert reserved_pending.exists()
    assert (state.stat().st_dev, state.stat().st_ino) == (
        unknown_alias.stat().st_dev,
        unknown_alias.stat().st_ino,
    )
    assert (state.stat().st_dev, state.stat().st_ino) != (
        reserved_pending.stat().st_dev,
        reserved_pending.stat().st_ino,
    )


def test_receipt_hardlink_recovery_rejects_and_preserves_unknown_files(
    linux_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _commit, _evidence, _promotion, prepared, report = _prepared()
    state_parent = (linux_tmp_path / "state-parent").resolve()
    state_parent.mkdir(mode=0o700)
    monkeypatch.setattr(start_module, "FORMAL_RUN_PARENT", state_parent)
    monkeypatch.setattr(
        start_module._local_admission,
        "validate_local_admission_receipt_v1",
        lambda *_args, **_kwargs: None,
    )
    local_admission = _local_admission(report)
    state = start_module.canonical_start_state_path_v1(report["run_id_hex"])
    assert (
        write_state_exact_once_v1(
            state,
            prepared,
            local_admission=local_admission,
        )
        == "STARTED_NEW"
    )
    receipt = canonical_start_publication_receipt_path_v1(report["run_id_hex"])
    unknown_alias = receipt.parent / "unknown-receipt-hardlink"
    os.link(receipt, unknown_alias)
    reserved_pending = receipt.parent / (
        f".{receipt.name}.4242.{'b' * 32}.pending"
    )
    reserved_pending.write_bytes(receipt.read_bytes())
    reserved_pending.chmod(0o600)

    with pytest.raises(M3StartError) as captured:
        write_state_exact_once_v1(
            state,
            prepared,
            local_admission=local_admission,
        )
    assert captured.value.code == start_module.FAIL_STATE_IO
    assert unknown_alias.exists()
    assert reserved_pending.exists()
    assert (receipt.stat().st_dev, receipt.stat().st_ino) == (
        unknown_alias.stat().st_dev,
        unknown_alias.stat().st_ino,
    )
    assert (receipt.stat().st_dev, receipt.stat().st_ino) != (
        reserved_pending.stat().st_dev,
        reserved_pending.stat().st_ino,
    )


def test_state_reader_rejects_wrong_mode_and_additional_hardlink(
    linux_tmp_path: Path,
) -> None:
    state = linux_tmp_path / "state.json"
    state.write_bytes(b"{}\n")
    state.chmod(0o644)
    with pytest.raises(M3StartError) as captured:
        read_state_file_v1(state)
    assert captured.value.code == start_module.FAIL_STATE_IO

    state.chmod(0o600)
    alias = linux_tmp_path / "state-alias.json"
    os.link(state, alias)
    with pytest.raises(M3StartError) as captured:
        read_state_file_v1(state)
    assert captured.value.code == start_module.FAIL_STATE_IO


def test_writer_rejects_a_second_noncanonical_path(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _commit, _evidence, _promotion, prepared, report = _prepared()
    state_parent = (linux_tmp_path / "state-parent").resolve()
    state_parent.mkdir(mode=0o700)
    monkeypatch.setattr(start_module, "FORMAL_RUN_PARENT", state_parent)
    occupied = (linux_tmp_path / "second-start.json").resolve()
    occupied.write_bytes(b"different\n")
    with pytest.raises(M3StartError) as captured:
        write_state_exact_once_v1(
            occupied,
            prepared,
            local_admission=_local_admission(report),
        )
    assert captured.value.code == start_module.FAIL_STATE_IO
    assert occupied.read_bytes() == b"different\n"


def test_strict_json_and_exact_replay_verifier_fail_closed() -> None:
    with pytest.raises(M3StartError) as captured:
        strict_json_loads_v1(b'{"a":1,"a":2}\n', label="duplicate")
    assert captured.value.code == FAIL_JSON

    commit, evidence, promotion, _prepared_start, report = _prepared()
    verified = verify_m3_start_v1(
        canonical_json_v1(report),
        evidence,
        promotion,
        publication_commit=commit,
    )
    assert verified == report

    tampered = json.loads(canonical_json_v1(report))
    tampered["closure_invoked"] = True
    with pytest.raises(M3StartError):
        validate_state_document_v1(tampered)


def test_cli_defaults_to_prepare_and_requires_explicit_start(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    commit, evidence, promotion, _prepared_start, report = _prepared()
    monkeypatch.setattr(
        start_cli,
        "load_publication_blobs_v1",
        lambda **_kwargs: (commit, evidence, promotion),
    )
    monkeypatch.setattr(start_cli, "prepare_m3_start_v1", lambda *_args, **_kwargs: report)
    assert (
        start_cli.main(
            ["--recorded-at-unix-seconds", str(RECORDED_AT)],
            _launch_capability=start_cli._DIRECT_ENTRYPOINT_SEAL,
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "prepare"
    assert output["status"] == "PREPARED_DIAGNOSTIC_ONLY_NOT_PERSISTABLE"
    assert output["closure_invoked"] is False
    assert "state_record_cbor_hex" not in output
    assert "state_artifact_sha256" not in output


def test_cli_start_requires_and_replays_local_admission_before_write(
    linux_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    commit, evidence, promotion, prepared, report = _prepared()
    local_admission = _local_admission(report)
    state_path = (linux_tmp_path / "canonical-start.json").resolve()
    calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        start_cli,
        "canonical_start_state_path_v1",
        lambda _run_id: state_path,
    )
    monkeypatch.setattr(
        start_cli,
        "validate_live_local_admission_v1",
        lambda revision: (
            calls.append(("admission", revision)),
            local_admission,
        )[1],
    )
    monkeypatch.setattr(
        start_cli,
        "load_publication_blobs_v1",
        lambda **_kwargs: (
            calls.append(("publication",)),
            (commit, evidence, promotion),
        )[1],
    )
    monkeypatch.setattr(
        start_cli,
        "prepare_authoritative_m3_start_v1",
        lambda *_args, **_kwargs: (
            calls.append(("prepare",)),
            prepared,
        )[1],
    )
    monkeypatch.setattr(
        start_cli,
        "write_state_exact_once_v1",
        lambda path, value, *, local_admission: (
            calls.append(("write", path, value, local_admission)),
            "STARTED_NEW",
        )[1],
    )

    assert (
        start_cli.main(
            [
                "--mode",
                "start",
                "--recorded-at-unix-seconds",
                str(RECORDED_AT),
                "--state",
                state_path.as_posix(),
                "--admission-revision",
                APPROVAL_COMMIT_D,
            ],
            _launch_capability=start_cli._DIRECT_ENTRYPOINT_SEAL,
        )
        == 0
    )
    assert [call[0] for call in calls] == [
        "admission",
        "publication",
        "prepare",
        "write",
    ]
    assert calls[0] == ("admission", APPROVAL_COMMIT_D)
    assert calls[-1][1:] == (state_path, prepared, local_admission)
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "STARTED_NEW"
    assert summary["closure_invoked"] is False


def test_cli_start_without_admission_revision_fails_before_any_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        start_cli,
        "validate_live_local_admission_v1",
        lambda *_args, **_kwargs: pytest.fail("local admission was invoked"),
    )
    with pytest.raises(SystemExit):
        start_cli.main(
            [
                "--mode",
                "start",
                "--recorded-at-unix-seconds",
                str(RECORDED_AT),
                "--state",
                "/tmp/noncanonical-start.json",
            ],
            _launch_capability=start_cli._DIRECT_ENTRYPOINT_SEAL,
        )
