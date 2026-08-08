from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

import pytest

from hegel_machine import phase3_m25_a8_recovery_amendment_r5_v1 as amendment
from hegel_machine import phase3_m25_a8_recovery_cli_r5_v1 as recovery_cli
from hegel_machine import phase3_m25_formal_container_executor_v1 as executor


def _write_record(path: Path, fields: dict[str, object]) -> bytes:
    raw = amendment._receipt_record_bytes_v1(fields)
    path.write_bytes(raw)
    path.chmod(0o600)
    return raw


def test_r5_freezes_exact_r4r2_terminal_chain_and_full_receipts() -> None:
    rows = amendment._r4_terminal_chain_snapshot_v1()
    assert len(rows) == 8
    assert hashlib.sha256(amendment._canonical_json(rows)).hexdigest() == (
        amendment.R4_TERMINAL_CHAIN_ROOT_SHA256
    )
    assert all(len(row["receipt_sha256"]) == 64 for row in rows)
    assert rows[2]["receipt_sha256"] == (
        "83b1ad690914d9dfd5cd402d5c734a1250a3b450c9e0b3ecf4655cfb97c6ba47"
    )


def test_r5_source_admission_is_exact_executor_43_key_schema() -> None:
    unchanged = amendment._unchanged_a8_input_bindings_v1()
    admission = amendment._build_source_admission_v1(
        amendment_commit="11" * 20,
        incident_raw=b"incident",
        validation_raw=b"validation",
        validation={
            "actor_report_sha256": "aa" * 32,
            "errata_report_sha256": "bb" * 32,
            "live_bundle_sha256": "cc" * 32,
        },
        unchanged_inputs=unchanged,
    )
    assert len(admission) == 43
    assert admission["schema"] == amendment.SOURCE_ADMISSION_SCHEMA
    assert admission["recovery_attempt_ordinal"] == 5
    assert admission["r4_terminal_chain_root_sha256"] == (
        amendment.R4_TERMINAL_CHAIN_ROOT_SHA256
    )
    assert admission["r4_a8_validation_receipt_sha256"] == (
        amendment.R4_TERMINAL_AUDIT_RECEIPT_SHA256["a8-validation-receipt.json"]
    )


def test_r5_all_95_unchanged_inputs_use_descriptor_stable_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = amendment._r4._r31._verify_changed_worktree_blob_v1
    observed: list[str] = []

    def verify(**kwargs: object) -> None:
        observed.append(str(kwargs["relative"]))
        real(**kwargs)

    monkeypatch.setattr(amendment._r4._r31, "_verify_changed_worktree_blob_v1", verify)
    bindings = amendment._unchanged_a8_input_bindings_v1()
    assert len(observed) == 95
    assert set(observed) == set(bindings)


def test_r5_unchanged_input_binding_rejects_symlink_before_follow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory(prefix="hegel-r5-unchanged-", dir="/tmp") as raw:
        repository = Path(raw)
        source = repository / "input.py"
        source.write_text("frozen = True\n", encoding="utf-8")
        source.chmod(0o644)

        def git(*arguments: str) -> None:
            subprocess.run(
                ["git", *arguments], cwd=repository, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )

        git("init", "-q")
        git("-c", "user.name=R5 Test", "-c", "user.email=r5@example.invalid", "add", "input.py")
        git(
            "-c", "user.name=R5 Test", "-c", "user.email=r5@example.invalid",
            "commit", "-qm", "fixture",
        )
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
            stdout=subprocess.PIPE,
        ).stdout.decode("ascii").strip()
        frozen = subprocess.run(
            ["git", "show", f"{head}:input.py"], cwd=repository, check=True,
            stdout=subprocess.PIPE,
        ).stdout
        expected = {"input.py": hashlib.sha256(frozen).hexdigest()}
        monkeypatch.setattr(amendment, "REPOSITORY_ROOT", repository)
        monkeypatch.setattr(amendment, "A8_BASIS_COMMIT", head)
        monkeypatch.setattr(amendment, "R5_RUNTIME_EXCEPTION_PATHS", frozenset())
        monkeypatch.setattr(amendment._r4._r31, "REQUIRED_COMMIT_A_INPUTS", (source,))
        monkeypatch.setattr(amendment._r4._r31, "EXPECTED_UNCHANGED_A8_INPUT_COUNT", 1)
        monkeypatch.setattr(
            amendment._r4._r31,
            "EXPECTED_UNCHANGED_A8_INPUT_ROOT",
            hashlib.sha256(amendment._executor_canonical_json(expected)).hexdigest(),
        )
        target = repository / "untracked-target.py"
        target.write_bytes(frozen)
        source.unlink()
        source.symlink_to(target)
        with pytest.raises(amendment.A8R5RecoveryAmendmentError, match="descriptor binding"):
            amendment._unchanged_a8_input_bindings_v1()


def test_r5_runtime_exception_binding_rejects_hidden_index_and_byte_drift() -> None:
    with tempfile.TemporaryDirectory(prefix="hegel-r5-source-binding-", dir="/tmp") as raw:
        repository = Path(raw)
        source = repository / "tracked.py"
        source.write_text("frozen = True\n", encoding="utf-8")
        source.chmod(0o644)

        def git(*arguments: str) -> None:
            subprocess.run(
                ["git", *arguments], cwd=repository, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )

        git("init", "-q")
        git("-c", "user.name=R5 Test", "-c", "user.email=r5@example.invalid", "add", "tracked.py")
        git(
            "-c", "user.name=R5 Test", "-c", "user.email=r5@example.invalid",
            "commit", "-qm", "fixture",
        )
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
            stdout=subprocess.PIPE,
        ).stdout.decode("ascii").strip()
        rows = amendment._runtime_exception_source_bindings_v1(
            repository_root=repository, head=head, relative_paths=("tracked.py",)
        )
        assert rows[0]["head_blob_sha256"] == rows[0]["worktree_sha256"]

        git("update-index", "--assume-unchanged", "tracked.py")
        with pytest.raises(amendment.A8R5RecoveryAmendmentError, match="non-normal index flag"):
            amendment._runtime_exception_source_bindings_v1(
                repository_root=repository, head=head, relative_paths=("tracked.py",)
            )
        git("update-index", "--no-assume-unchanged", "tracked.py")
        source.write_text("frozen = False\n", encoding="utf-8")
        with pytest.raises(amendment.A8R5RecoveryAmendmentError, match="differs from HEAD"):
            amendment._runtime_exception_source_bindings_v1(
                repository_root=repository, head=head, relative_paths=("tracked.py",)
            )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("marker_state",), "COMPLETE"),
        (("journal_state",), "COMMITTED"),
        (("docker_state", "run_labelled_container_count"), 1),
        (("docker_state", "fixed_key_volume_count"), 3),
        (("docker_state", "network_operation_invoked"), True),
        (("seed_prefix_metadata", 0, "size_bytes"), 31),
        (("seed_prefix_metadata", 0, "mode_octal"), "0644"),
        (("seed_prefix_metadata", 0, "raw_bytes_read"), True),
        (("seed_prefix_metadata", 0, "sha256_computed"), True),
        (("formal_identity_entropy_draw_count",), 1),
        (("m3_start_allowed",), True),
    ],
)
def test_r5_incident_preflight_fault_matrix_fails_closed(
    monkeypatch: pytest.MonkeyPatch, path: tuple[object, ...], value: object
) -> None:
    base: dict[str, object] = {
        "marker_state": "PENDING",
        "journal_state": "RESERVED",
        "docker_state": {
            "run_labelled_container_count": 0,
            "fixed_key_volume_count": 4,
            "network_operation_invoked": False,
        },
        "seed_prefix_metadata": [
            {
                "name": "split_master_seed.bin",
                "size_bytes": 32,
                "mode_octal": "0600",
                "raw_seed": True,
                "raw_bytes_read": False,
                "sha256_computed": False,
            }
        ],
        "formal_identity_entropy_draw_count": 0,
        "m3_start_allowed": False,
    }
    cursor: object = base
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    cursor[path[-1]] = value  # type: ignore[index]
    monkeypatch.setattr(amendment._r4, "_build_incident_diagnostic_v1", lambda **_: base)
    monkeypatch.setattr(amendment, "_r4_terminal_chain_snapshot_v1", lambda: ())
    with pytest.raises(amendment.A8R5RecoveryAmendmentError):
        amendment._build_incident_diagnostic_v1(
            custody_directory=Path("/unused"),
            public_evidence_path=Path("/unused-evidence"),
            public_promotion_path=Path("/unused-promotion"),
        )


def test_r5_incident_accepts_r4_producers_tuple_seed_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = {
        "name": "split_master_seed.bin",
        "size_bytes": 32,
        "mode_octal": "0600",
        "raw_seed": True,
        "raw_bytes_read": False,
        "sha256_computed": False,
    }
    base = {
        "marker_state": "PENDING",
        "journal_state": "RESERVED",
        "docker_state": {
            "run_labelled_container_count": 0,
            "fixed_key_volume_count": 4,
            "network_operation_invoked": False,
        },
        "seed_prefix_metadata": (seed,),
        "formal_identity_entropy_draw_count": 0,
        "m3_start_allowed": False,
    }
    monkeypatch.setattr(
        amendment._r4, "_build_incident_diagnostic_v1", lambda **_: base
    )
    monkeypatch.setattr(amendment, "_r4_terminal_chain_snapshot_v1", lambda: ())
    incident = amendment._build_incident_diagnostic_v1(
        custody_directory=Path("/unused"),
        public_evidence_path=Path("/unused-evidence"),
        public_promotion_path=Path("/unused-promotion"),
    )
    assert incident["marker_state"] == "PENDING"
    assert incident["seed_prefix_metadata"] == (seed,)
    assert incident["raw_seed_bytes_read_by_r5_orchestrator"] is False


def test_r5_incident_discloses_noncanonical_r4_inference_without_overclaim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = {
        "marker_state": "PENDING",
        "journal_state": "RESERVED",
        "docker_state": {
            "run_labelled_container_count": 0,
            "fixed_key_volume_count": 4,
            "network_operation_invoked": False,
        },
        "seed_prefix_metadata": [
            {
                "name": "split_master_seed.bin", "size_bytes": 32,
                "mode_octal": "0600", "raw_seed": True,
                "raw_bytes_read": False, "sha256_computed": False,
            }
        ],
        "formal_identity_entropy_draw_count": 0,
        "m3_start_allowed": False,
        "raw_seed_bytes_read_by_r4_orchestrator": False,
    }
    monkeypatch.setattr(amendment._r4, "_build_incident_diagnostic_v1", lambda **_: base)
    monkeypatch.setattr(amendment, "_r4_terminal_chain_snapshot_v1", lambda: ())
    incident = amendment._build_incident_diagnostic_v1(
        custody_directory=Path("/unused"),
        public_evidence_path=Path("/unused-evidence"),
        public_promotion_path=Path("/unused-promotion"),
    )
    assert incident["r4_legacy_primary_status"] == (
        "UNRECOVERABLE_FROM_CANONICAL_R4_RECORD"
    )
    assert incident["r4_inference_is_canonical_failure_evidence"] is False
    assert incident["r4_synthetic_reproduction"]["custody_shape"] == (
        "EXACT_7_NAMES_AND_MODES_SEED_SIZE_ONLY"
    )
    assert incident["r4_synthetic_reproduction"]["attestation_status"] == (
        "UNATTESTED_NONCANONICAL_DIAGNOSTIC"
    )
    assert incident["raw_seed_bytes_read_by_r5_orchestrator"] is False


def test_r5_failure_evidence_preserves_real_amendment_code_and_nested_cleanup() -> None:
    primary = amendment.A8R5RecoveryAmendmentError(
        amendment.FAIL_AMENDMENT, "primary detail"
    )
    cleanup = executor.FormalContainerExecutorError(
        executor.FAIL_CONTAINER, "cleanup detail"
    )
    composite = executor.combine_formal_failures_v1(
        primary, cleanup, phase="R5_OUTER_FINAL_CLOSE"
    )
    evidence = amendment._failure_evidence_v1(composite)
    assert evidence["primary"]["code"] == amendment.FAIL_AMENDMENT
    assert evidence["primary"]["detail_sha256"] == hashlib.sha256(
        b"primary detail"
    ).hexdigest()
    assert evidence["cleanup"]["code"] == executor.FAIL_CONTAINER
    assert "primary detail" not in str(evidence)


def test_r5_failure_evidence_uses_shared_depth_bound() -> None:
    error: BaseException = executor.FormalContainerExecutorError(
        executor.FAIL_CONTAINER, "root"
    )
    for _ in range(executor._FORMAL_FAILURE_EVIDENCE_MAX_DEPTH + 4):
        error = executor.combine_formal_failures_v1(
            error,
            executor.FormalContainerExecutorError(executor.FAIL_CONTAINER, "cleanup"),
            phase="R5_OUTER_FINAL_CLOSE",
        )
    evidence = amendment._failure_evidence_v1(error)
    assert "MAX_DEPTH" in str(evidence)


def _execute_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *,
    core_error: BaseException | None = None,
    close_error: BaseException | None = None,
    install_fault: tuple[str, str] | None = None,
    failure_visibility_raises: bool = False,
    failure_discard_raises: bool = False,
    preexisting_failure: str | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    audit = tmp_path / "audit"
    audit.mkdir(mode=0o700)
    monkeypatch.setattr(amendment, "FIXED_R5_AUDIT_DIRECTORY", audit)
    monkeypatch.setattr(
        amendment,
        "_require_existing_audit_directory",
        lambda path, repository_root: path,
    )

    def read(path: Path):
        raw = path.read_bytes()
        return json.loads(raw), raw

    def read_regular(path: Path, *, mode: int):
        value, raw = read(path)
        return value, raw, {"mode_octal": f"{mode:04o}"}

    def install(path: Path, expected: dict[str, object], raw: bytes) -> None:
        assert amendment._canonical_json(expected) == raw
        if install_fault == (path.name, "before"):
            raise OSError("injected-install-before")
        path.write_bytes(raw)
        if install_fault == (path.name, "after"):
            raise OSError("injected-install-after")

    def visible(path: Path, raw: bytes) -> bool:
        if path.name == "failure.json" and failure_visibility_raises:
            raise OSError("injected-failure-visibility")
        return path.is_file() and path.read_bytes() == raw

    def discard(path: Path) -> None:
        if path.name == "failure.json" and failure_discard_raises:
            raise OSError("injected-failure-discard")

    monkeypatch.setattr(amendment._r4._r31._r2, "_read_canonical_audit", read)
    monkeypatch.setattr(amendment._r4._r31._r2, "_read_canonical_regular", read_regular)
    monkeypatch.setattr(amendment, "_install_exact_audit_record_v1", install)
    monkeypatch.setattr(amendment, "_install_prepare_record_v1", lambda path, raw: path.write_bytes(raw))
    monkeypatch.setattr(
        amendment,
        "_exact_audit_record_is_visible_v1",
        visible,
    )
    monkeypatch.setattr(amendment, "_discard_non_authoritative_next_v1", discard)
    preflight = {
        "amendment_commit": "12" * 20,
        "runtime_exception_source_bindings": (
            {
                "path": "frozen.py",
                "git_mode": "100644",
                "head_blob_sha256": "ab" * 32,
                "worktree_sha256": "ab" * 32,
                "worktree_mode_octal": "0644",
            },
        ),
    }
    incident = {"runtime_artifact_bindings": {}, "live_runtime_stable_projection": ()}
    validation = {
        "actor_report_sha256": "aa" * 32,
        "errata_report_sha256": "bb" * 32,
        "live_bundle_sha256": "cc" * 32,
    }
    preflight_raw = _write_record(audit / "preflight.json", preflight)
    incident_raw = _write_record(audit / "incident-diagnostic.json", incident)
    validation_raw = _write_record(audit / "a8-validation-receipt.json", validation)
    validation_value = json.loads(validation_raw)
    request_fields = amendment._authorization_request_fields(
        amendment_commit=preflight["amendment_commit"],
        preflight_raw=preflight_raw,
        incident_raw=incident_raw,
        validation_raw=validation_raw,
    )
    request_raw = _write_record(audit / "authorization-request.json", request_fields)
    _write_record(
        audit / "authorization.json",
        amendment._expected_authorization_fields(
            amendment_commit=preflight["amendment_commit"],
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
            validation_raw=validation_raw,
            request_raw=request_raw,
        ),
    )
    monkeypatch.setattr(amendment, "inspect_r5_source_preflight_v1", lambda **_: preflight)
    monkeypatch.setattr(amendment, "_build_incident_diagnostic_v1", lambda **_: incident)
    monkeypatch.setattr(
        amendment, "_validation_request_from_incident_v1",
        lambda _: ({}, {}, {}, {}),
    )
    monkeypatch.setattr(
        amendment,
        "_run_a8_validator_v1",
        lambda _: (validation_value, validation_raw),
    )
    monkeypatch.setattr(amendment, "_validate_runtime_artifacts_before_attempt_v1", lambda **_: ())
    monkeypatch.setattr(amendment, "_r4_terminal_chain_snapshot_v1", lambda: ())
    monkeypatch.setattr(amendment, "_unchanged_a8_input_bindings_v1", lambda: {})
    monkeypatch.setattr(amendment, "_build_source_admission_v1", lambda **_: {"r5": True})
    recovery = SimpleNamespace(
        basis_commit=amendment.A8_BASIS_COMMIT,
        run_id=amendment.FIXED_RUN_ID,
        ledger_id=amendment.FIXED_LEDGER_ID,
        marker_snapshot=SimpleNamespace(state="PENDING", created_at_unix_seconds=0),
        journal_state="RESERVED",
        prestage_intent_fields={
            "actor_qualification_report": {}, "errata_qualification_report": {}
        },
        stage_directory=tmp_path / "stage",
    )
    monkeypatch.setattr(amendment, "acquire_pending_ceremony_recovery_v1", lambda **_: nullcontext(recovery))

    class Actors:
        timestamp = 0

        def __init__(self, **_: object) -> None:
            pass

        def close(self) -> None:
            if close_error is not None:
                raise close_error

    monkeypatch.setattr(amendment, "A8R1RecoveryDockerActorsV1", Actors)

    def core(**kwargs: object):
        kwargs["source_admission_guard"](recovery)  # type: ignore[operator]
        if core_error is not None:
            raise core_error
        return {"payload": True}, {"promotion": True}

    monkeypatch.setattr(amendment, "_continue_pre_stage_pending_recovery_core_v1", core)
    monkeypatch.setattr(amendment, "_validate_final_publication_v1", lambda **_: {"replayed": True})
    if preexisting_failure is not None:
        original_failure_fields = amendment._failure_record_fields_v1

        def install_preexisting(**kwargs: object):
            fields = original_failure_fields(**kwargs)
            if not (audit / "failure.json").exists():
                if preexisting_failure == "exact":
                    _write_record(audit / "failure.json", fields)
                else:
                    _write_record(audit / "failure.json", {"conflict": True})
            return fields

        monkeypatch.setattr(amendment, "_failure_record_fields_v1", install_preexisting)
    return amendment.execute_fixed_a8_r5_recovery_v1(
        custody_directory=tmp_path / "custody",
        rust_formal_replay_binary=tmp_path / "formal",
        rust_bridge_dag_replay_binary=tmp_path / "bridge",
        rust_bridge_dag_qualification_report=tmp_path / "report",
        public_evidence_path=tmp_path / "evidence",
        public_promotion_path=tmp_path / "promotion",
        audit_directory=audit,
    )


def test_r5_success_finalizes_then_rejects_second_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, promotion = _execute_harness(tmp_path, monkeypatch)
    assert payload == {"payload": True}
    assert promotion == {"promotion": True}
    assert (tmp_path / "audit/finalize.json").is_file()
    with pytest.raises(amendment.A8R5RecoveryAmendmentError, match="already consumed"):
        amendment.execute_fixed_a8_r5_recovery_v1(
            custody_directory=tmp_path / "custody",
            rust_formal_replay_binary=tmp_path / "formal",
            rust_bridge_dag_replay_binary=tmp_path / "bridge",
            rust_bridge_dag_qualification_report=tmp_path / "report",
            public_evidence_path=tmp_path / "evidence",
            public_promotion_path=tmp_path / "promotion",
            audit_directory=tmp_path / "audit",
        )


def test_r5_primary_and_final_close_are_both_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = executor.FormalContainerExecutorError(
        executor.FAIL_CONTAINER, "primary"
    )
    cleanup = executor.FormalContainerExecutorError(
        executor.FAIL_CONTAINER, "final-close"
    )
    with pytest.raises(executor.FormalContainerCompositeError):
        _execute_harness(
            tmp_path, monkeypatch, core_error=primary, close_error=cleanup
        )
    failure, _raw = amendment._r4._r31._r2._read_canonical_audit(
        tmp_path / "audit/failure.json"
    )
    assert failure["formal_failure_evidence"]["kind"] == "PRIMARY_AND_CLEANUP"
    assert failure["primary_failure"]["detail_sha256"] == hashlib.sha256(
        b"primary"
    ).hexdigest()
    assert failure["final_close_failure_or_null"]["detail_sha256"] == hashlib.sha256(
        b"final-close"
    ).hexdigest()


def test_r5_sole_final_close_failure_is_explicitly_classified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    close_error = executor.FormalContainerExecutorError(
        executor.FAIL_CONTAINER, "sole-final-close"
    )
    with pytest.raises(executor.FormalContainerExecutorError):
        _execute_harness(tmp_path, monkeypatch, close_error=close_error)
    failure, _raw = amendment._r4._r31._r2._read_canonical_audit(
        tmp_path / "audit/failure.json"
    )
    assert failure["failure_phase"] == "R5_OUTER_FINAL_CLOSE"
    assert failure["primary_failure"]["detail_sha256"] == (
        failure["final_close_failure_or_null"]["detail_sha256"]
    )


@pytest.mark.parametrize(
    ("record_name", "timing", "terminal_name"),
    [
        ("attempt-start.json", "before", None),
        ("attempt-start.json", "after", "failure.json"),
        ("admission.json", "before", "failure.json"),
        ("admission.json", "after", "failure.json"),
        ("finalize.json", "before", "failure.json"),
        ("finalize.json", "after", "finalize.json"),
        ("failure.json", "before", "failure.json"),
        ("failure.json", "after", "failure.json"),
    ],
)
def test_r5_record_install_before_after_link_fault_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_name: str,
    timing: str,
    terminal_name: str | None,
) -> None:
    primary = executor.FormalContainerExecutorError(
        executor.FAIL_CONTAINER, "matrix-primary"
    )
    core_error = primary if record_name == "failure.json" else None
    if record_name == "finalize.json" and timing == "after":
        result = _execute_harness(
            tmp_path,
            monkeypatch,
            install_fault=(record_name, timing),
        )
        assert result == ({"payload": True}, {"promotion": True})
    else:
        with pytest.raises(BaseException) as caught:
            _execute_harness(
                tmp_path,
                monkeypatch,
                core_error=core_error,
                install_fault=(record_name, timing),
            )
        if timing == "after" or record_name != "attempt-start.json":
            evidence = executor.formal_failure_evidence_v1(caught.value)
            assert "injected-install" in str(caught.value) or evidence["kind"] in {
                "PRIMARY_AND_CLEANUP", "SINGLE"
            }
    audit = tmp_path / "audit"
    terminals = [name for name in ("failure.json", "finalize.json") if (audit / name).exists()]
    assert terminals == ([] if terminal_name is None else [terminal_name])
    assert not any(path.name.endswith(".next") for path in audit.iterdir())


@pytest.mark.parametrize("fault", ["visibility", "discard"])
def test_r5_post_primary_resolution_fault_never_masks_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    primary = executor.FormalContainerExecutorError(
        executor.FAIL_CONTAINER, "must-survive"
    )
    with pytest.raises(executor.FormalContainerCompositeError) as caught:
        _execute_harness(
            tmp_path,
            monkeypatch,
            core_error=primary,
            install_fault=("failure.json", "before"),
            failure_visibility_raises=fault == "visibility",
            failure_discard_raises=fault == "discard",
        )
    evidence = executor.formal_failure_evidence_v1(caught.value)
    assert hashlib.sha256(b"must-survive").hexdigest() in str(evidence)


@pytest.mark.parametrize("existing", ["exact", "conflict"])
def test_r5_concurrent_existing_failure_must_be_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: str
) -> None:
    primary = executor.FormalContainerExecutorError(
        executor.FAIL_CONTAINER, "concurrent-primary"
    )
    expected = executor.FormalContainerExecutorError if existing == "exact" else executor.FormalContainerCompositeError
    with pytest.raises(expected):
        _execute_harness(
            tmp_path,
            monkeypatch,
            core_error=primary,
            preexisting_failure=existing,
        )
    failure = json.loads((tmp_path / "audit/failure.json").read_bytes())
    assert ("conflict" in failure) is (existing == "conflict")


def test_later_r5_artifacts_are_excluded_from_historical_runtime_closure() -> None:
    required = {
        "Hegel Machine/src/hegel_machine/phase3_m25_a8_recovery_amendment_r5_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_m25_a8_recovery_cli_r5_v1.py",
    }
    assert required.issubset(amendment.R5_RUNTIME_EXCEPTION_PATHS)
    assert required.issubset(amendment._r4._r31.R3_RUNTIME_EXCEPTION_PATHS)
    assert required.issubset(amendment._r4.R4_RUNTIME_EXCEPTION_PATHS)


def test_r5_cli_preflight_requires_live_transaction_paths() -> None:
    parser = recovery_cli._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["preflight"])
    parsed = parser.parse_args(
        [
            "preflight",
            "--custody-directory", "/custody",
            "--public-evidence-output", "/evidence",
            "--promotion-output", "/promotion",
        ]
    )
    assert parsed.operation == "preflight"


def test_r5_cli_never_eagerly_stringifies_nonformal_exception(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class HostileValueError(ValueError):
        def __str__(self) -> str:
            raise AssertionError("stringifier must not run")

    def reject(**_: object) -> dict[str, object]:
        raise HostileValueError()

    monkeypatch.setattr(recovery_cli, "inspect_fixed_a8_r5_preflight_v1", reject)
    code = recovery_cli.main(
        [
            "preflight",
            "--custody-directory", "/custody",
            "--public-evidence-output", "/evidence",
            "--promotion-output", "/promotion",
        ]
    )
    assert code == 2
    error = json.loads(capsys.readouterr().err)
    assert error["detail"] == "non-formal exception type HostileValueError"
