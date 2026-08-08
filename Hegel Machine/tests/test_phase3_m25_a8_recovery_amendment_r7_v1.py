from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from hegel_machine import phase3_m25_a8_recovery_amendment_r7_v1 as amendment
from hegel_machine import phase3_m25_a8_recovery_cli_r7_v1 as recovery_cli
from hegel_machine import phase3_m25_formal_container_executor_v1 as executor


def test_r7_freezes_exact_r6_terminal_ten_record_chain() -> None:
    rows = amendment._r6_terminal_chain_snapshot_v1()
    assert len(rows) == 10
    assert hashlib.sha256(amendment._canonical_json(rows)).hexdigest() == amendment.R6_TERMINAL_CHAIN_ROOT_SHA256
    assert amendment.R6_TERMINAL_CHAIN_ROOT_SHA256 == "d17b2fc442226b1800f7f4900b52dbca824f5391ca0ab0ec1d4f6fc034711de2"
    assert amendment.R6_FAILURE_CODE == "FAIL_M25_FORMAL_CUSTODY_STATE"
    assert amendment.R6_FAILURE_PHASE == "COMPLETE_ONLY_FORMAL_CORE"


def test_r7_reuses_exact_r6_source_admission_without_rebuild() -> None:
    source = amendment._fixed_r6_source_admission_v1()
    assert source["schema"] == amendment._r6.SOURCE_ADMISSION_SCHEMA
    assert hashlib.sha256(amendment._canonical_json(source)).hexdigest() == amendment.FIXED_R6_SOURCE_ADMISSION_SHA256
    executor._validate_recovery_source_admission_v1(
        source,
        basis_commit=amendment.A8_BASIS_COMMIT,
        run_id=amendment.FIXED_RUN_ID,
        ledger_id=amendment.FIXED_LEDGER_ID,
    )


def test_r7_exact_r6_source_loader_rejects_every_top_level_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = amendment._fixed_r6_source_admission_v1()
    assert len(exact) == 47
    monkeypatch.setattr(amendment, "_r6_terminal_chain_snapshot_v1", lambda: ())
    monkeypatch.setattr(
        amendment._executor,
        "_validate_recovery_source_admission_v1",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("hash-drifted source reached the semantic validator")
        ),
    )
    candidates: list[dict[str, object]] = []
    for field in exact:
        changed = dict(exact)
        changed[field] = "__r7_drift__"
        candidates.append(changed)
        missing = dict(exact)
        del missing[field]
        candidates.append(missing)
    extended = dict(exact)
    extended["unfrozen_extension"] = True
    candidates.append(extended)
    for candidate in candidates:
        monkeypatch.setattr(
            amendment._r6,
            "_read_canonical_audit_v1",
            lambda _path, candidate=candidate: (
                {"source_admission": candidate},
                b"not-used",
            ),
        )
        with pytest.raises(
            amendment.A8R7RecoveryAmendmentError,
            match="fixed R6 source admission differs",
        ):
            amendment._fixed_r6_source_admission_v1()


def test_r7_real_historical_bridge_binds_current_sole_child_and_validator() -> None:
    source = amendment._fixed_r6_source_admission_v1()
    head = amendment._git(
        amendment.REPOSITORY_ROOT,
        ("rev-parse", "--verify", "HEAD^{commit}"),
    ).decode("ascii").strip()
    assert head != source["r6_amendment_commit"]
    committed = executor._validate_fixed_a8_r6_commit_context_v1(
        source,
        historical_direct_child_commit=head,
    )
    live = executor._FIXED_A8_R3_VALIDATOR_PATH.read_bytes()
    assert committed == live


def test_r7_historical_bridge_replays_exact_staged_evidence_without_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if path.name == "split_master_seed.bin":
            raise AssertionError("historical public replay attempted raw-seed access")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    source = amendment._fixed_r6_source_admission_v1()
    intent = amendment._load_fixed_prestage_intent_v1()
    evidence, evidence_raw, _evidence_identity = amendment._read_canonical_regular_v1(
        amendment.FIXED_STAGE_DIRECTORY / "public-evidence.json",
        mode=0o600,
    )
    _promotion, promotion_raw, _promotion_identity = amendment._read_canonical_regular_v1(
        amendment.FIXED_STAGE_DIRECTORY / "promotion.json",
        mode=0o600,
    )
    assert hashlib.sha256(evidence_raw).hexdigest() == (
        "ba2195cf83ca9bd26164a3e64b3b18a367965079af9c5f0e382259d0b115091e"
    )
    assert hashlib.sha256(promotion_raw).hexdigest() == (
        "ad33c293ea1d5e97425ce51403c914ee900fa78c255299556565cb19797cfce4"
    )
    head = amendment._git(
        amendment.REPOSITORY_ROOT,
        ("rev-parse", "--verify", "HEAD^{commit}"),
    ).decode("ascii").strip()
    replayed = executor._replay_public_gate_evidence_with_fixed_a8_r6_direct_child_basis_v1(
        evidence,
        source_admission=source,
        prestage_intent_fields=intent,
        historical_direct_child_commit=head,
    )
    replayed_raw = executor._canonical_json(replayed)
    assert replayed_raw == promotion_raw
    assert hashlib.sha256(replayed_raw).hexdigest() == (
        "ad33c293ea1d5e97425ce51403c914ee900fa78c255299556565cb19797cfce4"
    )
    assert replayed["child_state"] == "NOT_RUN"
    assert replayed["m3_run_started"] is False
    gate_report = replayed["gate_report"]
    assert isinstance(gate_report, dict)
    assert gate_report["gates_before"] == 14
    assert gate_report["gates_after"] == 24
    assert gate_report["all_gates_15_24_passed"] is True
    assert gate_report["all_output_slots_null"] is True


def test_r7_historical_bridge_does_not_relax_ordinary_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reached: list[bool] = []
    monkeypatch.setattr(
        executor,
        "_continue_pre_stage_pending_recovery_core_v1",
        lambda **_kwargs: reached.append(True),
    )
    recovery = SimpleNamespace(basis_commit=amendment.A8_BASIS_COMMIT)
    actors = SimpleNamespace(authoritative=True)
    with pytest.raises(
        executor.FormalContainerExecutorError,
        match="ordinary recovery requires HEAD to equal the formal basis commit",
    ) as captured:
        executor.continue_pre_stage_pending_recovery_v1(
            recovery=recovery,
            actors=actors,
        )
    assert captured.value.code == executor.FAIL_RECOVERY_SOURCE_ADMISSION
    assert reached == []


def test_r7_loads_exact_staged_prestage_intent_without_seed_access() -> None:
    intent = amendment._load_fixed_prestage_intent_v1()
    assert intent["basis_commit"] == amendment.A8_BASIS_COMMIT
    assert intent["run_id_hex"] == amendment.FIXED_RUN_ID_HEX
    assert intent["ledger_id_hex"] == amendment.FIXED_LEDGER_ID_HEX
    source = inspect.getsource(amendment._load_fixed_prestage_intent_v1)
    assert "split_master_seed.bin" not in source


def test_r7_policy_is_poststage_only_and_fail_closed() -> None:
    policy = amendment._manifest_fixed_policy_v1()
    assert policy["recovery_attempt_ordinal"] == 7
    assert policy["sole_parent_commit"] == amendment.R6_AMENDMENT_COMMIT
    assert policy["poststage_only"] is True
    for denied in (
        "ordinary_execute_allowed",
        "prestage_core_allowed",
        "signing_allowed",
        "static_rebuild_allowed",
        "source_rebuild_allowed",
        "redraw_allowed",
        "m3_start_allowed",
    ):
        assert policy[denied] is False


def test_r7_runtime_exception_registry_adds_only_module_and_cli() -> None:
    required = {
        "Hegel Machine/src/hegel_machine/phase3_m25_a8_recovery_amendment_r7_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_m25_a8_recovery_cli_r7_v1.py",
    }
    assert amendment.R7_RUNTIME_EXCEPTION_PATHS == (
        amendment._r6.R6_RUNTIME_EXCEPTION_PATHS | required
    )
    assert len(amendment.R7_RUNTIME_EXCEPTION_PATHS) == 17


def test_r7_fixed_replay_binds_exact_source_and_intent(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, object, object, object]] = []

    def replay(
        candidate: object,
        *,
        source_admission: object,
        prestage_intent_fields: object,
        historical_direct_child_commit: object,
    ) -> dict[str, bool]:
        calls.append(
            (
                candidate,
                source_admission,
                prestage_intent_fields,
                historical_direct_child_commit,
            )
        )
        return {"ok": True}

    monkeypatch.setattr(
        amendment,
        "_replay_public_gate_evidence_with_fixed_a8_r6_direct_child_basis_v1",
        replay,
    )
    source = {"schema": "source"}
    intent = {"schema": "intent"}
    child = "11" * 20
    bound = amendment._fixed_replay_v1(
        source_admission=source,
        prestage_intent=intent,
        amendment_commit=child,
    )
    candidate = {"schema": "candidate"}
    assert bound(candidate) == {"ok": True}
    assert calls == [(candidate, source, intent, child)]


def test_r7_pre_attempt_recheck_rejects_same_bytes_new_inode(tmp_path: Path) -> None:
    audit = tmp_path / "audit"
    audit.mkdir(mode=0o700)
    for name in amendment._R7_PRE_ATTEMPT_AUDIT_NAMES:
        path = audit / name
        path.write_bytes(amendment._receipt_record_bytes_v1({"schema": name}))
        path.chmod(0o600)
    identity, raws, inodes = amendment._read_pre_attempt_audit_snapshot_v1(audit, allow_attempt_next=False)
    target = audit / "authorization.json"
    replacement = audit / "replacement.tmp"
    replacement.write_bytes(target.read_bytes())
    replacement.chmod(0o600)
    replacement.replace(target)
    with pytest.raises(amendment.A8R7RecoveryAmendmentError, match="changed under transaction lock"):
        amendment._recheck_pre_attempt_audit_under_lock_v1(
            audit=audit,
            expected_directory_identity=identity,
            expected_raws=raws,
            expected_inodes=inodes,
        )


def test_r7_locked_linearization_precedes_only_poststage_core() -> None:
    source = inspect.getsource(amendment.execute_fixed_a8_r7_recovery_v1)
    assert source.index("_recheck_pre_attempt_audit_under_lock_v1") < source.index("attempt-start.json")
    assert source.index("attempt-start.json") < source.index("_continue_post_stage_transaction_recovery_core_v1")
    assert "_continue_pre_stage_pending_recovery_core_v1" not in source
    assert "execute_formal_container_ceremony_v1" not in source
    assert "sign_manifest" not in source
    assert "phase3-m3-start" not in source


def test_r7_poststage_qualification_rejects_actor_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = tmp_path / "split_master_seed.bin"
    seed.write_bytes(b"x" * 32)
    seed.chmod(0o600)
    marker = executor.MarkerSnapshot("PENDING", b"s" * 32, None, b"k" * 16, 0)
    transaction = SimpleNamespace(
        basis_commit=amendment.A8_BASIS_COMMIT,
        run_id=amendment.FIXED_RUN_ID,
        ledger_id=amendment.FIXED_LEDGER_ID,
        recovery_phase="STAGED_PENDING",
        _state="STAGED_PROSPECTIVE_REPLAY_PASSED",
        _recovery_marker_snapshot=marker,
        _lock_descriptor=7,
        _stage_directory=amendment.FIXED_STAGE_DIRECTORY,
        _prestage_intent_bytes=b"intent",
        _prestage_intent_fields={"live_actor_protocol_daemon_receipt_binding": b"d" * 32},
        _staged_payloads={"evidence": b"e", "promotion": b"p", "receipt": b"r"},
        custody_directory=tmp_path,
    )
    actors = SimpleNamespace(
        _actor_start_attempted=True,
        _containers={},
        _state_volumes={},
    )
    monkeypatch.setattr(amendment, "FIXED_PRESTAGE_INTENT_SHA256", hashlib.sha256(b"intent").hexdigest())
    with pytest.raises(amendment.A8R7RecoveryAmendmentError, match="exact locked"):
        amendment._qualify_poststage_locked_v1(
            transaction=transaction,
            actors=actors,
            amendment_commit="11" * 20,
            source_admission={"schema": "fixed"},
            runtime_rows=(),
        )


def test_r7_cli_requires_full_runtime_for_prepare_and_recover() -> None:
    parser = recovery_cli._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["prepare-authorization", "--audit-directory", "/tmp/audit"])
    with pytest.raises(SystemExit):
        parser.parse_args(["recover-fixed-poststage", "--audit-directory", "/tmp/audit"])


def test_r7_module_never_reads_or_hashes_raw_seed() -> None:
    source = Path(amendment.__file__).read_text(encoding="utf-8")
    assert 'read_bytes()' not in "\n".join(
        line for line in source.splitlines() if "split_master_seed.bin" in line
    )
    assert "raw_seed_bytes_read_by_r7_orchestrator\": False" in source
    assert "m3_start_invoked\": False" in source


def test_r7_docker_snapshot_uses_frozen_read_only_r2_inspector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "run_labelled_container_count": 0,
        "fixed_key_volume_count": 4,
        "network_operation_invoked": False,
    }
    observed: list[bool] = []

    def inspect_read_only() -> dict[str, object]:
        observed.append(True)
        return expected

    monkeypatch.setattr(
        amendment._r6._r4._r31._r2,
        "_docker_read_only_state_v1",
        inspect_read_only,
    )
    assert amendment._docker_snapshot_v1() == expected
    assert observed == [True]


@pytest.mark.parametrize("visible", (False, True))
def test_candidate_install_resolves_before_and_after_link_faults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, visible: bool,
) -> None:
    path = tmp_path / "attempt-start.json"
    raw = amendment._receipt_record_bytes_v1({"schema": "candidate/1"})
    events: list[str] = []

    def faulty_install(*_args: object, **_kwargs: object) -> None:
        events.append("install")
        raise executor.FormalContainerExecutorError(executor.FAIL_CONTAINER, "fault")

    monkeypatch.setattr(amendment, "_install_exact_audit_record_v1", faulty_install)
    monkeypatch.setattr(amendment, "_exact_audit_record_is_visible_v1", lambda *_a: visible)
    monkeypatch.setattr(amendment, "_install_prepare_record_v1", lambda *_a: events.append("repair"))
    monkeypatch.setattr(amendment, "_discard_non_authoritative_next_v1", lambda *_a: events.append("discard"))
    if visible:
        status, error = amendment._install_candidate_resolving_visibility_v1(
            path=path, expected={"schema": "candidate/1"}, raw=raw, phase="TEST"
        )
        assert status == "VISIBLE_REPAIRED"
        assert isinstance(error, executor.FormalContainerExecutorError)
        assert events == ["install", "repair"]
    else:
        status, error = amendment._install_candidate_resolving_visibility_v1(
            path=path, expected={"schema": "candidate/1"}, raw=raw, phase="TEST"
        )
        assert status == "HIDDEN"
        assert isinstance(error, executor.FormalContainerExecutorError)
        assert events == ["install", "discard"]


def test_execute_assigns_authoritative_raw_only_after_visible_install() -> None:
    source = inspect.getsource(amendment.execute_fixed_a8_r7_recovery_v1)
    attempt_install = source.index('path=audit / "attempt-start.json"')
    attempt_bind = source.index("attempt_start_raw = attempt_candidate_raw")
    admission_install = source.index('path=audit / "admission.json"')
    admission_bind = source.index("admission_raw = admission_candidate_raw")
    assert attempt_install < attempt_bind < admission_install < admission_bind


def test_visible_finalize_is_authoritative_and_precedes_failure_path() -> None:
    source = inspect.getsource(amendment.execute_fixed_a8_r7_recovery_v1)
    finalize_install = source.index('path=audit / "finalize.json"')
    success_return = source.index("return payload, promotion", finalize_install)
    failure_terminal = source.index("_terminalize_failure_v1", success_return)
    assert finalize_install < success_return < failure_terminal


def test_failure_hidden_fault_retries_canonical_terminal_without_losing_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "failure.json"
    fields = {"schema": "failure/1", "code": "PRIMARY"}
    raw = amendment._receipt_record_bytes_v1(fields)
    primary = executor.FormalContainerExecutorError(executor.FAIL_CUSTODY, "primary")
    events: list[str] = []

    def fail_install(*_args: object, **_kwargs: object) -> None:
        events.append("install")
        raise executor.FormalContainerExecutorError(executor.FAIL_CONTAINER, "audit")

    def canonical_retry(candidate: Path, candidate_raw: bytes) -> None:
        events.append("retry")
        candidate.write_bytes(candidate_raw)
        candidate.chmod(0o600)

    monkeypatch.setattr(amendment, "_install_exact_audit_record_v1", fail_install)
    monkeypatch.setattr(amendment, "_exact_audit_record_is_visible_v1", lambda *_a: False)
    monkeypatch.setattr(amendment, "_discard_non_authoritative_next_v1", lambda *_a: events.append("discard"))
    monkeypatch.setattr(amendment, "_install_prepare_record_v1", canonical_retry)
    with pytest.raises(executor.FormalContainerCompositeError) as captured:
        amendment._terminalize_failure_v1(
            path=path, failure=fields, failure_raw=raw, primary=primary
        )
    assert path.read_bytes() == raw
    assert events == ["install", "discard", "retry"]
    evidence = executor.formal_failure_evidence_v1(captured.value)
    assert evidence["kind"] == "PRIMARY_AND_CLEANUP"
    assert evidence["primary"]["code"] == executor.FAIL_CUSTODY


def test_r7_failure_classifier_marks_outer_close_and_sole_close() -> None:
    primary = executor.FormalContainerExecutorError(executor.FAIL_CUSTODY, "primary")
    close = executor.FormalContainerExecutorError(executor.FAIL_CONTAINER, "close")
    combined = executor.combine_formal_failures_v1(
        primary, close, phase="R7_OUTER_FINAL_CLOSE_ACTOR_CLOSE"
    )
    evidence = executor.formal_failure_evidence_v1(combined)
    rows = amendment._leaf_failure_rows_v1(evidence)
    assert rows[0]["role"] == "PRIMARY"
    assert rows[1]["role"] == "FINAL_CLOSE"
    sole = amendment._failure_record_fields_v1(
        amendment_commit="11" * 20,
        qualification_raw=b"qualification",
        attempt_start_raw=b"attempt",
        admission_raw=b"admission",
        failure_phase="R7_OUTER_FINAL_CLOSE",
        exc=close,
    )
    assert sole["final_close_failure_or_null"]["role"] == "FINAL_CLOSE"
    assert sole["failure_phase"] == "R7_OUTER_FINAL_CLOSE"


@pytest.mark.parametrize("mutated", ("source", "intent"))
def test_fixed_replay_rejects_delegate_mutation(
    monkeypatch: pytest.MonkeyPatch, mutated: str,
) -> None:
    source = {"schema": "source"}
    intent = {"schema": "intent"}

    def mutate(_candidate: object, **_kwargs: object) -> dict[str, bool]:
        (source if mutated == "source" else intent)["mutated"] = True
        return {"ok": True}

    monkeypatch.setattr(
        amendment,
        "_replay_public_gate_evidence_with_fixed_a8_r6_direct_child_basis_v1",
        mutate,
    )
    replay = amendment._fixed_replay_v1(
        source_admission=source,
        prestage_intent=intent,
        amendment_commit="11" * 20,
    )
    with pytest.raises(amendment.A8R7RecoveryAmendmentError, match="during delegated replay"):
        replay({"candidate": True})


def test_r7_cli_catches_runtime_error_without_stringifying_hostile_exception(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    class HostileRuntimeError(RuntimeError):
        def __str__(self) -> str:
            raise AssertionError("must not stringify")

    monkeypatch.setattr(
        recovery_cli,
        "inspect_fixed_a8_r7_preflight_v1",
        lambda **_kwargs: (_ for _ in ()).throw(HostileRuntimeError()),
    )
    assert recovery_cli.main(["preflight"]) == 2
    report = __import__("json").loads(capsys.readouterr().err)
    assert report == {
        "detail": "non-formal exception type HostileRuntimeError",
        "error_code": "FAIL_M25_A8_R7_RECOVERY_CLI",
        "ok": False,
    }


def test_r7_cli_does_not_swallow_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        recovery_cli,
        "inspect_fixed_a8_r7_preflight_v1",
        lambda **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        recovery_cli.main(["preflight"])


def test_r7_cli_catches_key_error_without_leaking_key(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        recovery_cli,
        "inspect_fixed_a8_r7_preflight_v1",
        lambda **_kwargs: (_ for _ in ()).throw(KeyError("secret-key-name")),
    )
    assert recovery_cli.main(["preflight"]) == 2
    report = __import__("json").loads(capsys.readouterr().err)
    assert report["detail"] == "non-formal exception type KeyError"
    assert "secret-key-name" not in str(report)


def test_nested_close_graph_classifies_cleanup_and_final_close() -> None:
    primary = executor.FormalContainerExecutorError(executor.FAIL_CUSTODY, "primary")
    cleanup = executor.FormalContainerExecutorError(executor.FAIL_CONTAINER, "cleanup")
    close = executor.FormalContainerExecutorError(executor.FAIL_CONTAINER, "close")
    inner = executor.combine_formal_failures_v1(primary, cleanup, phase="POSTSTAGE_RECOVERY_ACTOR_CLEANUP")
    outer = executor.combine_formal_failures_v1(inner, close, phase="R7_OUTER_FINAL_CLOSE_ACTOR_CLOSE")
    rows = amendment._leaf_failure_rows_v1(executor.formal_failure_evidence_v1(outer))
    assert [row["role"] for row in rows] == ["PRIMARY", "CLEANUP", "FINAL_CLOSE"]


def _authorization_prefix_v1(audit: Path) -> tuple[dict[str, object], dict[str, bytes]]:
    preflight = {"schema": "test-preflight/1", "amendment_commit": "11" * 20}
    incident = {"schema": "test-incident/1"}
    qualification = {"schema": "test-qualification/1"}
    raws = {
        "preflight.json": amendment._receipt_record_bytes_v1(preflight),
        "incident-diagnostic.json": amendment._receipt_record_bytes_v1(incident),
        "poststage-qualification.json": amendment._receipt_record_bytes_v1(qualification),
    }
    raws["authorization-request.json"] = amendment._receipt_record_bytes_v1(
        amendment._authorization_request_fields(
            amendment_commit=preflight["amendment_commit"],
            preflight_raw=raws["preflight.json"],
            incident_raw=raws["incident-diagnostic.json"],
            qualification_raw=raws["poststage-qualification.json"],
        )
    )
    for name, raw in raws.items():
        (audit / name).write_bytes(raw)
        (audit / name).chmod(0o600)
    return preflight, raws


def test_authorization_wrong_tamper_repeat_and_foreign_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = tmp_path / "audit"
    audit.mkdir(mode=0o700)
    _preflight, raws = _authorization_prefix_v1(audit)
    monkeypatch.setattr(amendment, "_require_existing_audit_directory", lambda *_a, **_k: audit)
    monkeypatch.setattr(
        amendment,
        "_install_prepare_record_v1",
        lambda path, raw: (path.write_bytes(raw), path.chmod(0o600)),
    )
    with pytest.raises(amendment.A8R7RecoveryAmendmentError, match="confirmation phrase"):
        amendment.write_fixed_a8_r7_owner_authorization_v1(
            audit_directory=audit, owner_confirmation="WRONG"
        )
    request = __import__("json").loads(raws["authorization-request.json"])
    request["requested_action"] = "TAMPER"
    (audit / "authorization-request.json").write_bytes(amendment._canonical_json(request))
    with pytest.raises(amendment.A8R7RecoveryAmendmentError, match="self-hash differs"):
        amendment.write_fixed_a8_r7_owner_authorization_v1(
            audit_directory=audit, owner_confirmation=amendment.OWNER_CONFIRMATION
        )
    (audit / "authorization-request.json").write_bytes(raws["authorization-request.json"])
    amendment.write_fixed_a8_r7_owner_authorization_v1(
        audit_directory=audit, owner_confirmation=amendment.OWNER_CONFIRMATION
    )
    first = (audit / "authorization.json").read_bytes()
    amendment.write_fixed_a8_r7_owner_authorization_v1(
        audit_directory=audit, owner_confirmation=amendment.OWNER_CONFIRMATION
    )
    assert (audit / "authorization.json").read_bytes() == first
    (audit / ".foreign.next").write_bytes(b"foreign")
    with pytest.raises(amendment.A8R7RecoveryAmendmentError, match="path set differs"):
        amendment.write_fixed_a8_r7_owner_authorization_v1(
            audit_directory=audit, owner_confirmation=amendment.OWNER_CONFIRMATION
        )


@pytest.mark.parametrize(
    "crash_name",
    (
        "preflight.json",
        "incident-diagnostic.json",
        "poststage-qualification.json",
        "authorization-request.json",
    ),
)
def test_prepare_exact_prefix_crash_then_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash_name: str,
) -> None:
    audit = tmp_path / "audit"
    audit.mkdir(mode=0o700)
    preflight = {"schema": "test-preflight/1", "amendment_commit": "11" * 20}
    incident = {"schema": "test-incident/1"}
    qualification = {"schema": "test-qualification/1"}
    monkeypatch.setattr(amendment, "inspect_r7_source_preflight_v1", lambda **_k: preflight)
    monkeypatch.setattr(amendment, "_fixed_r6_source_admission_v1", lambda: {"schema": "source"})
    monkeypatch.setattr(amendment, "_load_fixed_prestage_intent_v1", lambda: {"schema": "intent"})
    monkeypatch.setattr(amendment, "_fixed_replay_v1", lambda **_k: (lambda value: dict(value)))
    monkeypatch.setattr(amendment, "_validate_runtime_artifacts_before_attempt_v1", lambda **_k: ())

    class FakeActors:
        def __init__(self, **_kwargs: object) -> None:
            pass

    monkeypatch.setattr(amendment, "A8R7RecoveryDockerActorsV1", FakeActors)
    monkeypatch.setattr(amendment.FormalCeremonyTransactionV1, "rehydrate_post_stage_v1", lambda **_k: object())
    monkeypatch.setattr(amendment, "_qualify_poststage_locked_v1", lambda **_k: (incident, qualification))
    monkeypatch.setattr(amendment, "_close_transaction_and_actor_v1", lambda *_a, **_k: None)
    monkeypatch.setattr(amendment, "_create_or_resume_prepare_audit_directory", lambda *_a, **_k: audit)
    crashing = True

    def install(path: Path, raw: bytes) -> None:
        nonlocal crashing
        if crashing and path.name == crash_name:
            raise executor.FormalContainerExecutorError(executor.FAIL_CONTAINER, "prepare-crash")
        if path.exists():
            assert path.read_bytes() == raw
            return
        path.write_bytes(raw)
        path.chmod(0o600)

    monkeypatch.setattr(amendment, "_install_prepare_record_v1", install)
    kwargs = {
        "audit_directory": audit,
        "custody_directory": tmp_path / "custody",
        "public_evidence_path": tmp_path / "evidence",
        "public_promotion_path": tmp_path / "promotion",
        "rust_formal_replay_binary": tmp_path / "formal",
        "rust_bridge_dag_replay_binary": tmp_path / "bridge",
        "rust_bridge_dag_qualification_report": tmp_path / "report",
    }
    with pytest.raises(executor.FormalContainerExecutorError, match="prepare-crash"):
        amendment.prepare_fixed_a8_r7_authorization_v1(**kwargs)
    assert not (audit / "attempt-start.json").exists()
    crashing = False
    amendment.prepare_fixed_a8_r7_authorization_v1(**kwargs)
    assert {path.name for path in audit.iterdir()} == {
        "preflight.json",
        "incident-diagnostic.json",
        "poststage-qualification.json",
        "authorization-request.json",
    }


def _execute_fault_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *, fault_name: str,
    fault_timing: str,
    core_error: BaseException | None = None,
    repair_fault_name: str | None = None,
    visibility_fault_name: str | None = None,
) -> tuple[Path, object]:
    audit = tmp_path / "audit"
    audit.mkdir(mode=0o700)
    preflight_fields = {"schema": "test-r7-preflight/1", "amendment_commit": "11" * 20}
    incident_fields = {"schema": "test-r7-incident/1"}
    qualification_fields = {"schema": "test-r7-qualification/1"}
    preflight_raw = amendment._receipt_record_bytes_v1(preflight_fields)
    incident_raw = amendment._receipt_record_bytes_v1(incident_fields)
    qualification_raw = amendment._receipt_record_bytes_v1(qualification_fields)
    request_raw = amendment._receipt_record_bytes_v1(
        amendment._authorization_request_fields(
            amendment_commit="11" * 20,
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
            qualification_raw=qualification_raw,
        )
    )
    authorization_raw = amendment._receipt_record_bytes_v1(
        amendment._expected_authorization_fields(
            amendment_commit="11" * 20,
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
            qualification_raw=qualification_raw,
            request_raw=request_raw,
        )
    )
    raws = {
        "preflight.json": preflight_raw,
        "incident-diagnostic.json": incident_raw,
        "poststage-qualification.json": qualification_raw,
        "authorization-request.json": request_raw,
        "authorization.json": authorization_raw,
    }
    monkeypatch.setattr(amendment, "inspect_r7_source_preflight_v1", lambda **_k: preflight_fields)
    monkeypatch.setattr(amendment, "_fixed_r6_source_admission_v1", lambda: {"schema": "fixed-source"})
    monkeypatch.setattr(amendment, "_load_fixed_prestage_intent_v1", lambda: {"schema": "fixed-intent"})
    monkeypatch.setattr(amendment, "_fixed_replay_v1", lambda **_k: (lambda value: dict(value)))
    monkeypatch.setattr(amendment, "_validate_runtime_artifacts_before_attempt_v1", lambda **_k: ())
    monkeypatch.setattr(amendment, "_require_existing_audit_directory", lambda *_a, **_k: audit)
    monkeypatch.setattr(
        amendment,
        "_read_pre_attempt_audit_snapshot_v1",
        lambda *_a, **_k: ((1, 2, 0o700, 3, 4), raws, {name: (1, index, "0600", len(raw), 3, 4) for index, (name, raw) in enumerate(raws.items())}),
    )
    monkeypatch.setattr(amendment, "_recheck_pre_attempt_audit_under_lock_v1", lambda **_k: None)

    class FakeActors:
        def __init__(self, **_kwargs: object) -> None:
            self.timestamp = 0

    transaction = SimpleNamespace(
        _recovery_marker_snapshot=executor.MarkerSnapshot("PENDING", b"s" * 32, None, b"k" * 16, 7)
    )
    monkeypatch.setattr(amendment, "A8R7RecoveryDockerActorsV1", FakeActors)
    monkeypatch.setattr(
        amendment.FormalCeremonyTransactionV1,
        "rehydrate_post_stage_v1",
        lambda **_k: transaction,
    )
    monkeypatch.setattr(
        amendment,
        "_qualify_poststage_locked_v1",
        lambda **_k: (incident_fields, qualification_fields),
    )
    monkeypatch.setattr(amendment, "_close_transaction_and_actor_v1", lambda _t, _a, primary, **_k: primary)

    payload = {"schema": "payload"}
    promotion = {"schema": "promotion"}

    def core(**_kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
        if core_error is not None:
            raise core_error
        return payload, promotion

    monkeypatch.setattr(amendment, "_continue_post_stage_transaction_recovery_core_v1", core)
    monkeypatch.setattr(amendment._r6, "_validate_final_publication_v1", lambda **_k: {"publication_replay_passed": True})

    def install(path: Path, _expected: object, raw: bytes) -> None:
        if path.name == fault_name:
            if fault_timing == "after":
                path.write_bytes(raw)
                path.chmod(0o600)
            raise executor.FormalContainerExecutorError(executor.FAIL_CONTAINER, f"{fault_name}-{fault_timing}")
        path.write_bytes(raw)
        path.chmod(0o600)

    def repair(path: Path, raw: bytes) -> None:
        if path.name == repair_fault_name:
            raise executor.FormalContainerExecutorError(
                executor.FAIL_CONTAINER, f"{path.name}-repair"
            )
        if path.exists():
            assert path.read_bytes() == raw
            return
        path.write_bytes(raw)
        path.chmod(0o600)

    monkeypatch.setattr(amendment, "_install_exact_audit_record_v1", install)
    monkeypatch.setattr(amendment, "_install_prepare_record_v1", repair)
    def visible(path: Path, raw: bytes) -> bool:
        if path.name == visibility_fault_name:
            raise executor.FormalContainerExecutorError(
                executor.FAIL_CONTAINER, f"{path.name}-visibility"
            )
        return path.exists() and path.read_bytes() == raw

    monkeypatch.setattr(amendment, "_exact_audit_record_is_visible_v1", visible)
    monkeypatch.setattr(amendment, "_discard_non_authoritative_next_v1", lambda _path: None)

    def invoke() -> tuple[dict[str, object], dict[str, object]]:
        return amendment.execute_fixed_a8_r7_recovery_v1(
            custody_directory=tmp_path / "custody",
            rust_formal_replay_binary=tmp_path / "formal",
            rust_bridge_dag_replay_binary=tmp_path / "bridge",
            rust_bridge_dag_qualification_report=tmp_path / "report",
            public_evidence_path=tmp_path / "evidence",
            public_promotion_path=tmp_path / "promotion",
            audit_directory=audit,
        )

    return audit, invoke


def test_execute_attempt_before_link_does_not_consume_or_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit, invoke = _execute_fault_harness(
        tmp_path, monkeypatch, fault_name="attempt-start.json", fault_timing="before"
    )
    with pytest.raises(executor.FormalContainerExecutorError, match="attempt-start.json-before"):
        invoke()
    assert not (audit / "attempt-start.json").exists()
    assert not (audit / "failure.json").exists()


def test_execute_admission_before_link_binds_null_admission_in_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit, invoke = _execute_fault_harness(
        tmp_path, monkeypatch, fault_name="admission.json", fault_timing="before"
    )
    with pytest.raises(executor.FormalContainerExecutorError, match="admission.json-before"):
        invoke()
    assert (audit / "attempt-start.json").is_file()
    failure = __import__("json").loads((audit / "failure.json").read_bytes())
    assert failure["admission_sha256_or_null"] is None


@pytest.mark.parametrize("fault_name", ("attempt-start.json", "admission.json"))
def test_execute_consumption_records_after_link_become_failure_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault_name: str,
) -> None:
    audit, invoke = _execute_fault_harness(
        tmp_path, monkeypatch, fault_name=fault_name, fault_timing="after"
    )
    with pytest.raises(executor.FormalContainerExecutorError):
        invoke()
    assert (audit / "attempt-start.json").is_file()
    assert (audit / "admission.json").is_file() is (fault_name == "admission.json")
    assert not (audit / "finalize.json").exists()
    assert (audit / "failure.json").is_file()


def test_execute_finalize_after_link_is_success_without_second_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit, invoke = _execute_fault_harness(
        tmp_path, monkeypatch, fault_name="finalize.json", fault_timing="after"
    )
    payload, promotion = invoke()
    assert payload == {"schema": "payload"}
    assert promotion == {"schema": "promotion"}
    assert (audit / "finalize.json").is_file()
    assert not (audit / "failure.json").exists()


def test_execute_finalize_before_link_emits_only_failure_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit, invoke = _execute_fault_harness(
        tmp_path, monkeypatch, fault_name="finalize.json", fault_timing="before"
    )
    with pytest.raises(executor.FormalContainerExecutorError, match="finalize.json-before"):
        invoke()
    assert not (audit / "finalize.json").exists()
    assert (audit / "failure.json").is_file()


def test_execute_visible_finalize_with_persistent_repair_failure_never_writes_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit, invoke = _execute_fault_harness(
        tmp_path,
        monkeypatch,
        fault_name="finalize.json",
        fault_timing="after",
        repair_fault_name="finalize.json",
    )
    with pytest.raises(executor.FormalContainerCompositeError):
        invoke()
    assert (audit / "finalize.json").is_file()
    assert not (audit / "failure.json").exists()


@pytest.mark.parametrize("fault_name", ("attempt-start.json", "admission.json"))
def test_execute_visible_consumption_with_persistent_repair_failure_terminalizes_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault_name: str,
) -> None:
    audit, invoke = _execute_fault_harness(
        tmp_path,
        monkeypatch,
        fault_name=fault_name,
        fault_timing="after",
        repair_fault_name=fault_name,
    )
    with pytest.raises(executor.FormalContainerCompositeError):
        invoke()
    assert (audit / "attempt-start.json").is_file()
    failure = __import__("json").loads((audit / "failure.json").read_bytes())
    if fault_name == "admission.json":
        assert failure["admission_sha256_or_null"] is not None
    else:
        assert failure["admission_sha256_or_null"] is None


def test_execute_admission_visibility_unknown_never_writes_guessed_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit, invoke = _execute_fault_harness(
        tmp_path,
        monkeypatch,
        fault_name="admission.json",
        fault_timing="after",
        visibility_fault_name="admission.json",
    )
    with pytest.raises(executor.FormalContainerCompositeError):
        invoke()
    assert (audit / "attempt-start.json").is_file()
    assert (audit / "admission.json").is_file()
    assert not (audit / "failure.json").exists()


@pytest.mark.parametrize("boundary", ("rehydrate", "qualification"))
def test_pre_attempt_rehydrate_or_qualification_failure_closes_without_consumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    audit, invoke = _execute_fault_harness(
        tmp_path, monkeypatch, fault_name="never.json", fault_timing="before"
    )
    closed: list[str] = []

    def fail(**_kwargs: object) -> object:
        raise executor.FormalContainerExecutorError(executor.FAIL_PREFLIGHT, boundary)

    if boundary == "rehydrate":
        monkeypatch.setattr(amendment.FormalCeremonyTransactionV1, "rehydrate_post_stage_v1", fail)
    else:
        monkeypatch.setattr(amendment, "_qualify_poststage_locked_v1", fail)
    monkeypatch.setattr(
        amendment,
        "_close_transaction_and_actor_v1",
        lambda _t, _a, primary, **_k: (closed.append("closed"), primary)[1],
    )
    with pytest.raises(executor.FormalContainerExecutorError, match=boundary):
        invoke()
    assert closed == ["closed"]
    assert not (audit / "attempt-start.json").exists()
    assert not (audit / "failure.json").exists()


def test_second_success_invocation_is_rejected_before_core_increment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit, invoke = _execute_fault_harness(
        tmp_path, monkeypatch, fault_name="never.json", fault_timing="before"
    )
    core_calls: list[int] = []

    def core(**_kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
        core_calls.append(1)
        return {"schema": "payload"}, {"schema": "promotion"}

    monkeypatch.setattr(amendment, "_continue_post_stage_transaction_recovery_core_v1", core)
    invoke()
    assert core_calls == [1]
    monkeypatch.setattr(
        amendment,
        "_read_pre_attempt_audit_snapshot_v1",
        lambda *_a, **_k: (_ for _ in ()).throw(
            amendment.A8R7RecoveryAmendmentError(
                amendment.FAIL_AMENDMENT, "terminal namespace already present"
            )
        ),
    )
    with pytest.raises(amendment.A8R7RecoveryAmendmentError, match="terminal namespace"):
        invoke()
    assert core_calls == [1]


@pytest.mark.parametrize("fault_timing", ("before", "after"))
def test_execute_failure_fault_terminalizes_and_preserves_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault_timing: str,
) -> None:
    primary = executor.FormalContainerExecutorError(executor.FAIL_CUSTODY, "core-primary")
    audit, invoke = _execute_fault_harness(
        tmp_path,
        monkeypatch,
        fault_name="failure.json",
        fault_timing=fault_timing,
        core_error=primary,
    )
    with pytest.raises(executor.FormalContainerCompositeError) as captured:
        invoke()
    failure = __import__("json").loads((audit / "failure.json").read_bytes())
    assert failure["failure_code"] == executor.FAIL_CUSTODY
    evidence = executor.formal_failure_evidence_v1(captured.value)
    assert evidence["primary"]["code"] == executor.FAIL_CUSTODY
