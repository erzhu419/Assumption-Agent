from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from hegel_machine import phase3_m25_a8_recovery_amendment_r3_v1 as amendment


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def test_r31_frozen_parent_revision_and_terminal_r2_identity() -> None:
    assert amendment.R2_AMENDMENT_COMMIT == "ec7c04cf62190558c72448639d7e3cd13a5b6903"
    assert amendment.R3_AMENDMENT_COMMIT == (
        "52a4a61934a73c70dc09b919cae377db166eaedf"
    )
    assert amendment.R2_AUDIT_RAW_SHA256["attempt-start.json"] == (
        "b4b817878d84c6506739f30adc4f38689791c37e3ee786e5c855b86df4a4f0e0"
    )
    assert amendment.R2_AUDIT_RAW_SHA256["failure.json"] == (
        "bd64cfa99885dd60750615fcb23abd960aed78ef676a0d2d4d8ed942e5395d56"
    )
    assert amendment.R2_AUDIT_RECEIPT_SHA256["failure.json"] == (
        "87b400cf0070efdb3e2f9d7b37dc09675258c5b0341ce629b7c7b6c5431f3f58"
    )
    assert amendment.OWNER_CONFIRMATION == (
        "AUTHORIZE_A8_R31_ATTEMPT_3_REVISION_1_CANONICAL_BYTES_"
        "COMPLETE_ONLY_REAL_PENDING_RESUME"
    )
    assert amendment.AUTHORIZATION_REVISION_ID == (
        "R31_CANONICAL_INCIDENT_BYTES_V1"
    )
    assert amendment.FIXED_R3_AUDIT_DIRECTORY != (
        amendment.R3_PREATTEMPT_AUDIT_DIRECTORY
    )


def test_live_r2_terminal_chain_is_exact_and_has_no_admission() -> None:
    rows = amendment._r2_terminal_chain_snapshot_v1()
    assert [row["name"] for row in rows] == [
        "preflight.json",
        "incident-diagnostic.json",
        "authorization-request.json",
        "authorization.json",
        "attempt-start.json",
        "failure.json",
    ]
    assert hashlib.sha256(amendment._canonical_json(rows)).hexdigest() == (
        amendment.R2_TERMINAL_CHAIN_ROOT_SHA256
    )
    assert not (amendment.R2_AUDIT_DIRECTORY / "admission.json").exists()
    assert not (amendment.R2_AUDIT_DIRECTORY / "finalize.json").exists()


def test_live_r3_preattempt_prefix_is_exact_and_attempt3_is_unconsumed() -> None:
    rows = amendment._r3_preattempt_prefix_snapshot_v1()
    assert [row["name"] for row in rows] == [
        "preflight.json",
        "incident-diagnostic.json",
        "a8-validation-receipt.json",
        "authorization-request.json",
        "authorization.json",
    ]
    assert hashlib.sha256(amendment._canonical_json(rows)).hexdigest() == (
        amendment.R3_PREATTEMPT_PREFIX_ROOT_SHA256
    )
    assert amendment.R3_PREATTEMPT_PREFIX_ROOT_SHA256 == (
        "9771b20bf63f1095456618d3ccd4c9db0c54c693307314b8aea72afa18249999"
    )
    assert not any(
        (
            amendment.R3_PREATTEMPT_AUDIT_DIRECTORY / name
        ).exists()
        for name in (
            "attempt-start.json",
            "admission.json",
            "failure.json",
            "finalize.json",
        )
    )
    assert not any(
        path.name.endswith(".next")
        for path in amendment.R3_PREATTEMPT_AUDIT_DIRECTORY.iterdir()
    )


def test_r31_incident_authority_is_canonical_bytes_not_python_shapes() -> None:
    rebuilt = {
        "schema": "test-r31-incident/1",
        "r1_failure_chain": ({"name": "one"},),
        "docker_state": {
            "fixed_key_volume_names": ("p1", "p2"),
            "run_labelled_container_names": (),
        },
    }
    stored_raw = amendment._receipt_record_bytes_v1(rebuilt)
    stored = json.loads(stored_raw)
    assert stored != amendment._r2._with_receipt_sha256(rebuilt)
    assert amendment._incident_receipt_bytes_equal_v1(stored_raw, rebuilt)

    changed = dict(rebuilt)
    changed["r1_failure_chain"] = ({"name": "two"},)
    assert not amendment._incident_receipt_bytes_equal_v1(stored_raw, changed)
    assert not amendment._incident_receipt_bytes_equal_v1(
        stored_raw + b" ", rebuilt
    )


def test_r31_source_binding_paths_reject_duplicate_omission_and_reorder() -> None:
    expected = ("Hegel Machine/a.py", "Hegel Machine/b.py")
    first = {"path": expected[0]}
    second = {"path": expected[1]}
    assert amendment._source_binding_paths_are_exact_v1(
        [first, second], expected
    )
    assert not amendment._source_binding_paths_are_exact_v1(
        [first, first], expected
    )
    assert not amendment._source_binding_paths_are_exact_v1(
        [second, first], expected
    )
    assert not amendment._source_binding_paths_are_exact_v1([first], expected)
    assert not amendment._source_binding_paths_are_exact_v1(
        [first, second, {"path": "Hegel Machine/c.py"}], expected
    )
    assert not amendment._source_binding_paths_are_exact_v1(
        [first, "Hegel Machine/b.py"], expected
    )


def test_r31_representation_mismatch_set_is_exactly_the_frozen_nine() -> None:
    stored = {
        "additional_stage_continuity_metadata": [],
        "docker_state": {
            "fixed_key_volume_label_rows": [],
            "fixed_key_volume_names": [],
            "run_labelled_container_names": [],
            "unchanged": True,
        },
        "fixed_stage_inventory": [],
        "public_reservation_metadata": [],
        "r1_failure_chain": [],
        "r2_terminal_chain": [],
        "seed_prefix_metadata": [],
        "unchanged": {"value": True},
    }
    rebuilt = {
        **stored,
        "additional_stage_continuity_metadata": (),
        "docker_state": {
            "fixed_key_volume_label_rows": (),
            "fixed_key_volume_names": (),
            "run_labelled_container_names": (),
            "unchanged": True,
        },
        "fixed_stage_inventory": (),
        "public_reservation_metadata": (),
        "r1_failure_chain": (),
        "r2_terminal_chain": (),
        "seed_prefix_metadata": (),
    }
    assert amendment._r3_preattempt_representation_mismatch_fields_v1(
        stored, rebuilt
    ) == amendment.R3_PREATTEMPT_REPRESENTATION_MISMATCH_FIELDS


def test_real_intent_request_normalizes_only_diagnostic_json() -> None:
    stage = Path(
        "/home/erzhu419/mine_code/Asumption Agent/Hegel Machine/artifacts/"
        "phase3_m25_external/formal_genesis_v2/"
        ".hegel-m25-stage-e4af9f57c38fb298462ec628c4ed8a03"
    )
    if not stage.is_dir():
        pytest.skip("fixed pending A8 stage is not present")
    request, actor, errata, bundle = amendment._validation_request_from_incident_v1(
        {"stage_directory": stage.as_posix()}
    )
    assert request["actor_report_sha256"] == (
        "b2aea587f267e864d3be296d3275f07c8ec4847174dd89adc461693ca629a3d4"
    )
    assert request["errata_report_sha256"] == (
        "82ad308e8162da08b22027d57d7dbc6201b3b315e92f27bc25600e37ef9baf81"
    )
    assert request["live_bundle_sha256"] == amendment._r2.EXPECTED_LIVE_BUNDLE_SHA256
    assert type(actor["actor_reports"]) is list
    assert type(errata["python_report"]["objects"]) is list
    assert any(type(value) is list for value in bundle.values())
    assert request["contains_raw_seed"] is False
    assert request["contains_private_key"] is False
    assert request["m3_start_allowed"] is False


def test_source_admission_is_exact_ordinal3_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unchanged = {"Hegel Machine/frozen.py": "11" * 32}
    root = hashlib.sha256(amendment._executor_canonical_json(unchanged)).hexdigest()
    monkeypatch.setattr(amendment, "EXPECTED_UNCHANGED_A8_INPUT_COUNT", 1)
    monkeypatch.setattr(amendment, "EXPECTED_UNCHANGED_A8_INPUT_ROOT", root)
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
    assert admission["schema"] == "hegel-phase3-m25-a8-r3-source-admission/1"
    assert admission["recovery_attempt_ordinal"] == 3
    assert admission["r2_admission_sha256_or_null"] is None
    assert admission["ordinary_execute_allowed"] is False
    assert admission["redraw_allowed"] is False
    assert admission["m3_start_allowed"] is False
    assert admission["prevalidated_report_basis"] is True
    assert admission["prevalidated_transaction_bundle"] is True
    assert admission["formal_identity_entropy_draw_count"] == 0
    assert admission["continuation_action"] == (
        amendment.SOURCE_ADMISSION_CONTINUATION_ACTION
    )
    assert admission["r2_failure_raw_sha256"] == (
        amendment.R2_AUDIT_RAW_SHA256["failure.json"]
    )


def test_prepare_record_install_is_exact_and_crash_resumable(tmp_path: Path) -> None:
    directory = tmp_path / "audit"
    directory.mkdir(mode=0o700)
    path = directory / "preflight.json"
    payload = b'{"ok":true}\n'
    amendment._install_prepare_record_v1(path, payload)
    assert path.read_bytes() == payload
    assert path.stat().st_mode & 0o777 == 0o600
    amendment._install_prepare_record_v1(path, payload)
    assert path.read_bytes() == payload

    second = directory / "incident-diagnostic.json"
    temporary = directory / ".incident-diagnostic.json.next"
    temporary.write_bytes(b"partial")
    temporary.chmod(0o600)
    amendment._install_prepare_record_v1(second, payload)
    assert second.read_bytes() == payload
    assert not temporary.exists()

    with pytest.raises(amendment.A8R3RecoveryAmendmentError):
        amendment._install_prepare_record_v1(path, b'{"ok":false}\n')


def test_exact_audit_installer_accepts_canonical_runtime_tuple_shape(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "audit"
    audit.mkdir(mode=0o700)
    path = audit / "attempt-start.json"
    runtime_artifact_metadata = (
        {
            "diagnostic_sha256_or_null": None,
            "mode_octal": "0755",
            "path": "/fixed/hegel-formal-bridge-m25",
            "sha256": "11" * 32,
        },
        {
            "diagnostic_sha256_or_null": None,
            "mode_octal": "0755",
            "path": "/fixed/hegel-m25-bridge-dag-replay",
            "sha256": "22" * 32,
        },
        {
            "diagnostic_sha256_or_null": "33" * 32,
            "mode_octal": "0644",
            "path": "/fixed/bridge-qualification.json",
            "sha256": "44" * 32,
        },
    )
    expected, raw = amendment._build_exact_audit_record_v1(
        {
            "schema": "test-attempt-start/1",
            "recovery_attempt_ordinal": 3,
            "runtime_artifact_metadata": runtime_artifact_metadata,
            "nested_tuple_fixture": (
                {"attester_roles": ("custodian", "python", "rust")},
            ),
        }
    )
    decoded = json.loads(raw)
    assert decoded != expected
    assert type(decoded["runtime_artifact_metadata"]) is list
    assert type(decoded["nested_tuple_fixture"][0]["attester_roles"]) is list

    amendment._install_exact_audit_record_v1(path, expected, raw)

    observed, observed_raw = amendment._r2._read_canonical_audit(path)
    assert observed != expected
    assert observed_raw == raw
    assert path.read_bytes() == raw


@pytest.mark.parametrize("mutation", ["content", "extra-byte"])
def test_exact_audit_installer_rejects_nonexact_raw_bytes(
    mutation: str, tmp_path: Path
) -> None:
    audit = tmp_path / "audit"
    audit.mkdir(mode=0o700)
    path = audit / "attempt-start.json"
    expected, raw = amendment._build_exact_audit_record_v1(
        {
            "schema": "test-attempt-start/1",
            "recovery_attempt_ordinal": 3,
            "runtime_artifact_metadata": (
                {
                    "diagnostic_sha256_or_null": None,
                    "mode_octal": "0755",
                    "path": "/fixed/hegel-formal-bridge-m25",
                    "sha256": "11" * 32,
                },
            ),
        }
    )
    if mutation == "content":
        candidate = raw.replace(b'"0755"', b'"0644"', 1)
        assert candidate != raw
    else:
        candidate = raw + b" "

    with pytest.raises(amendment.A8R3RecoveryAmendmentError):
        amendment._install_exact_audit_record_v1(path, expected, candidate)
    assert not path.exists()
    assert not (audit / ".attempt-start.json.next").exists()


def test_exact_audit_installer_rejects_existing_different_record(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "audit"
    audit.mkdir(mode=0o700)
    path = audit / "attempt-start.json"
    existing, existing_raw = amendment._build_exact_audit_record_v1(
        {"schema": "test-attempt-start/1", "record_version": "old"}
    )
    desired, desired_raw = amendment._build_exact_audit_record_v1(
        {"schema": "test-attempt-start/1", "record_version": "new"}
    )
    amendment._install_exact_audit_record_v1(path, existing, existing_raw)

    with pytest.raises(amendment.A8R3RecoveryAmendmentError):
        amendment._install_exact_audit_record_v1(path, desired, desired_raw)
    assert path.read_bytes() == existing_raw
    assert not (audit / ".attempt-start.json.next").exists()


@pytest.mark.parametrize("failure_call", [2, 3])
def test_attempt_record_publication_is_atomic_across_fsync_fault(
    failure_call: int,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = tmp_path / "audit"
    audit.mkdir(mode=0o700)
    path = audit / "attempt-start.json"
    expected, raw = amendment._build_exact_audit_record_v1(
        {"schema": "test-attempt-start/1", "recovery_attempt_ordinal": 3}
    )
    real_fsync_directory = amendment._fsync_directory_v1
    calls = 0

    def fail_once_after_link(directory: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise OSError("injected post-link/unlink directory fsync failure")
        real_fsync_directory(directory)

    monkeypatch.setattr(
        amendment, "_fsync_directory_v1", fail_once_after_link
    )
    with pytest.raises(OSError):
        amendment._install_exact_audit_record_v1(path, expected, raw)
    assert amendment._exact_audit_record_is_visible_v1(path, raw)
    assert not (audit / ".attempt-start.json.next").exists()
    observed, observed_raw = amendment._r2._read_canonical_audit(path)
    assert observed == expected
    assert observed_raw == raw


def test_complete_hidden_record_is_recreated_and_fsynced_before_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = tmp_path / "audit"
    audit.mkdir(mode=0o700)
    path = audit / "attempt-start.json"
    expected, raw = amendment._build_exact_audit_record_v1(
        {"schema": "test-attempt-start/1", "recovery_attempt_ordinal": 3}
    )
    temporary = audit / ".attempt-start.json.next"
    real_fsync = amendment.os.fsync
    real_link = amendment.os.link
    real_discard = amendment._discard_non_authoritative_next_v1
    first_file_fsync = True

    def fail_complete_hidden_file_fsync(descriptor: int) -> None:
        nonlocal first_file_fsync
        if first_file_fsync and stat.S_ISREG(os.fstat(descriptor).st_mode):
            first_file_fsync = False
            raise OSError("injected complete-hidden-file fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(amendment.os, "fsync", fail_complete_hidden_file_fsync)
    with pytest.raises(OSError):
        amendment._install_exact_audit_record_v1(path, expected, raw)
    assert not path.exists()
    assert temporary.read_bytes() == raw

    discarded = False
    file_fsync_completed = False

    def observe_discard(candidate: Path) -> None:
        nonlocal discarded
        real_discard(candidate)
        discarded = True

    def observe_fsync(descriptor: int) -> None:
        nonlocal file_fsync_completed
        metadata = os.fstat(descriptor)
        if stat.S_ISREG(metadata.st_mode):
            file_fsync_completed = True
        real_fsync(descriptor)

    def require_fsync_before_link(
        source: Path, destination: Path, *, follow_symlinks: bool
    ) -> None:
        assert discarded
        assert file_fsync_completed
        real_link(source, destination, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(
        amendment, "_discard_non_authoritative_next_v1", observe_discard
    )
    monkeypatch.setattr(amendment.os, "fsync", observe_fsync)
    monkeypatch.setattr(amendment.os, "link", require_fsync_before_link)
    amendment._install_exact_audit_record_v1(path, expected, raw)
    assert amendment._exact_audit_record_is_visible_v1(path, raw)


@pytest.mark.parametrize(
    "name", ["admission.json", "finalize.json", "failure.json"]
)
def test_non_authoritative_terminal_next_is_discarded_and_fsynced(
    name: str, tmp_path: Path
) -> None:
    audit = tmp_path / "audit"
    audit.mkdir(mode=0o700)
    path = audit / name
    temporary = audit / f".{name}.next"
    temporary.write_bytes(b"partial")
    temporary.chmod(0o600)
    amendment._discard_non_authoritative_next_v1(path)
    assert not path.exists()
    assert not temporary.exists()


def test_attempt_record_write_fault_never_exposes_partial_consumption_edge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = tmp_path / "audit"
    audit.mkdir(mode=0o700)
    path = audit / "attempt-start.json"
    expected, raw = amendment._build_exact_audit_record_v1(
        {"schema": "test-attempt-start/1", "recovery_attempt_ordinal": 3}
    )
    real_write_all = amendment._write_all_v1

    def fail_before_link(descriptor: int, payload: bytes) -> None:
        os.write(descriptor, payload[:7])
        raise OSError("injected hidden-inode write failure")

    monkeypatch.setattr(amendment, "_write_all_v1", fail_before_link)
    with pytest.raises(OSError):
        amendment._install_exact_audit_record_v1(path, expected, raw)
    assert not path.exists()
    assert (audit / ".attempt-start.json.next").is_file()

    monkeypatch.setattr(amendment, "_write_all_v1", real_write_all)
    amendment._install_exact_audit_record_v1(path, expected, raw)
    assert amendment._exact_audit_record_is_visible_v1(path, raw)
    assert not (audit / ".attempt-start.json.next").exists()


def test_changed_worktree_blob_requires_exact_head_bytes_and_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    source = repository / "Hegel Machine/source.py"
    source.parent.mkdir(parents=True)
    committed = b"frozen = True\n"
    source.write_bytes(committed)
    source.chmod(0o644)

    def fake_git(_repository: Path, arguments: tuple[str, ...]) -> bytes:
        if arguments[0] == "ls-tree":
            return (
                b"100644 blob " + b"1" * 40
                + b"\tHegel Machine/source.py\0"
            )
        if arguments[0] == "show":
            return committed
        raise AssertionError(arguments)

    monkeypatch.setattr(amendment, "_git", fake_git)
    amendment._verify_changed_worktree_blob_v1(
        repository_root=repository,
        head="2" * 40,
        relative="Hegel Machine/source.py",
        expected_sha256=hashlib.sha256(committed).hexdigest(),
    )
    source.write_bytes(b"frozen = False\n")
    with pytest.raises(amendment.A8R3RecoveryAmendmentError):
        amendment._verify_changed_worktree_blob_v1(
            repository_root=repository,
            head="2" * 40,
            relative="Hegel Machine/source.py",
            expected_sha256=hashlib.sha256(committed).hexdigest(),
        )


def test_changed_index_flags_reject_assume_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    relative = "Hegel Machine/source.py"
    monkeypatch.setattr(
        amendment,
        "_git",
        lambda _repository, _arguments: f"H {relative}\0".encode("utf-8"),
    )
    amendment._verify_changed_index_flags_v1(repository, {relative})
    monkeypatch.setattr(
        amendment,
        "_git",
        lambda _repository, _arguments: f"h {relative}\0".encode("utf-8"),
    )
    with pytest.raises(amendment.A8R3RecoveryAmendmentError):
        amendment._verify_changed_index_flags_v1(repository, {relative})


def test_prepare_and_authorize_are_exact_prefix_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    audit = tmp_path / "audit-r3"
    monkeypatch.setattr(amendment, "FIXED_R3_AUDIT_DIRECTORY", audit)
    preflight = {
        "schema": f"{amendment.AUDIT_SCHEMA_PREFIX}-preflight/1",
        "amendment_commit": "66" * 20,
        "sole_parent_commit": amendment.R3_AMENDMENT_COMMIT,
        "formal_repository_commit": amendment.A8_BASIS_COMMIT,
        "run_id_hex": amendment.FIXED_RUN_ID_HEX,
        "ledger_id_hex": amendment.FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 3,
    }
    incident = {
        "schema": f"{amendment.AUDIT_SCHEMA_PREFIX}-incident-diagnostic/1",
        "stage_directory": "/fixed/stage",
    }
    validation = {
        "schema": "hegel-phase3-m25-a8-r3-a8-validation-receipt/1",
        "receipt_sha256": "77" * 32,
    }
    validation_raw = _canonical(validation)
    child_request = {"schema": "request"}
    monkeypatch.setattr(
        amendment,
        "inspect_r3_source_preflight_v1",
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
        lambda _incident: (child_request, {}, {}, {}),
    )
    monkeypatch.setattr(
        amendment,
        "_run_a8_validator_v1",
        lambda _request: (dict(validation), validation_raw),
    )
    kwargs = {
        "audit_directory": audit,
        "custody_directory": tmp_path / "custody",
        "public_evidence_path": tmp_path / "evidence.json",
        "public_promotion_path": tmp_path / "promotion.json",
        "repository_root": repository,
        "manifest_path": tmp_path / "manifest.json",
    }
    amendment.prepare_fixed_a8_r3_authorization_v1(**kwargs)
    expected = {
        "preflight.json",
        "incident-diagnostic.json",
        "a8-validation-receipt.json",
        "authorization-request.json",
    }
    assert {path.name for path in audit.iterdir()} == expected
    amendment.prepare_fixed_a8_r3_authorization_v1(**kwargs)
    assert {path.name for path in audit.iterdir()} == expected

    with pytest.raises(amendment.A8R3RecoveryAmendmentError):
        amendment.write_fixed_a8_r3_owner_authorization_v1(
            audit_directory=audit,
            owner_confirmation="WRONG",
            repository_root=repository,
        )
    amendment.write_fixed_a8_r3_owner_authorization_v1(
        audit_directory=audit,
        owner_confirmation=amendment.OWNER_CONFIRMATION,
        repository_root=repository,
    )
    assert (audit / "authorization.json").is_file()
    amendment.write_fixed_a8_r3_owner_authorization_v1(
        audit_directory=audit,
        owner_confirmation=amendment.OWNER_CONFIRMATION,
        repository_root=repository,
    )


def test_current_runtime_closure_has_exact_95_unchanged_a8_inputs() -> None:
    later_r5 = {
        "Hegel Machine/src/hegel_machine/phase3_m25_a8_recovery_amendment_r5_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_m25_a8_recovery_cli_r5_v1.py",
    }
    bindings = amendment._unchanged_a8_input_bindings_v1()
    assert len(bindings) == amendment.EXPECTED_UNCHANGED_A8_INPUT_COUNT
    assert hashlib.sha256(
        amendment._executor_canonical_json(bindings)
    ).hexdigest() == amendment.EXPECTED_UNCHANGED_A8_INPUT_ROOT
    assert later_r5.issubset(amendment.R3_RUNTIME_EXCEPTION_PATHS)


def test_r3_sources_do_not_expose_seed_or_m3_start_entrypoint() -> None:
    module_source = Path(amendment.__file__).read_text(encoding="utf-8")
    tool_source = amendment.A8_VALIDATOR_TOOL.read_text(encoding="utf-8")
    assert "phase3-m3-start" not in module_source
    assert "split_master_seed.bin" not in module_source
    assert "split_master_seed.bin" not in tool_source
    assert "raw_seed_bytes_read_by_r3_orchestrator\": False" in module_source
    assert '"m3_start_invoked": False' in module_source
    assert all(flag in module_source for flag in ('"-I"', '"-S"', '"-B"', '"-X"'))
    assert amendment.FIXED_PYCACHE_PREFIX in module_source
    assert "FIXED_PYTHON_EXECUTABLE_SHA256" in module_source
    assert "_verify_a8_import_closure" in tool_source


def test_r3_isolated_validator_manifest_basis_is_exact() -> None:
    manifest, _raw = amendment._load_manifest(amendment.DEFAULT_MANIFEST_PATH)
    execution = manifest["a8_validator_execution"]
    assert execution["isolated_flags"] == ["-I", "-S", "-B"]
    assert execution["python_pycache_prefix"] == amendment.FIXED_PYCACHE_PREFIX
    assert execution["a8_import_closure_sha256_root"] == (
        amendment.EXPECTED_A8_IMPORT_CLOSURE_SHA256_ROOT
    )
    assert execution["validator_dependency_closure_sha256_root"] == (
        amendment.EXPECTED_A8_VALIDATOR_DEPENDENCY_CLOSURE_SHA256_ROOT
    )
    assert execution["tool_path"] == (
        amendment.R31_HISTORICAL_A8_VALIDATOR_TOOL.as_posix()
    )
    assert execution["tool_path"] != amendment.A8_VALIDATOR_TOOL.as_posix()
    assert manifest["expected_a8_validation_receipt_sha256"] == (
        amendment.EXPECTED_A8_VALIDATION_RECEIPT_RAW_SHA256
    )
    assert manifest["sole_parent_commit"] == amendment.R3_AMENDMENT_COMMIT
    assert manifest["authorization_revision_id"] == (
        amendment.AUTHORIZATION_REVISION_ID
    )
    assert manifest["r3_preattempt_prefix_root_sha256"] == (
        amendment.R3_PREATTEMPT_PREFIX_ROOT_SHA256
    )
    assert manifest["recovery_attempt_ordinal"] == 3


@pytest.mark.skipif(
    os.environ.get("HEGEL_RUN_REAL_R3_A8_VALIDATOR") != "1",
    reason="explicit opt-in: two full A8 report replays take about two minutes",
)
def test_real_isolated_a8_validator_receipt_repeats_exactly() -> None:
    incident = amendment._build_incident_diagnostic_v1(
        custody_directory=Path(
            "/home/erzhu419/.local/state/hegel-machine/"
            "phase3-m25-0af65964235390ce2bebefea7379eaa9c50eda24/formal-custody"
        ),
        public_evidence_path=Path(
            "/home/erzhu419/mine_code/Asumption Agent/Hegel Machine/artifacts/"
            "phase3_m25_external/formal_genesis_v2/"
            "phase3_m25_formal_gate_evidence_v1.json"
        ),
        public_promotion_path=Path(
            "/home/erzhu419/mine_code/Asumption Agent/Hegel Machine/artifacts/"
            "phase3_m25_external/formal_genesis_v2/phase3_m25_gate_promotion_v1.json"
        ),
    )
    request, *_ = amendment._validation_request_from_incident_v1(incident)
    first, first_raw = amendment._run_a8_validator_v1(request)
    second, second_raw = amendment._run_a8_validator_v1(request)
    assert first == second
    assert first_raw == second_raw
    assert first["commit_a_input_count"] == 98
    assert first["python_isolated"] is True
    assert first["python_no_site"] is True
    assert first["python_bytecode_disabled"] is True
    assert first["python_pycache_prefix"] == amendment.FIXED_PYCACHE_PREFIX
    assert first["a8_import_closure_sha256_root"] == (
        amendment.EXPECTED_A8_IMPORT_CLOSURE_SHA256_ROOT
    )
    assert first["a8_validator_dependency_closure_sha256_root"] == (
        amendment.EXPECTED_A8_VALIDATOR_DEPENDENCY_CLOSURE_SHA256_ROOT
    )
    assert hashlib.sha256(first_raw).hexdigest() == (
        amendment.EXPECTED_A8_VALIDATION_RECEIPT_RAW_SHA256
    )
    assert first["raw_seed_bytes_read"] is False
    assert first["m3_start_invoked"] is False
