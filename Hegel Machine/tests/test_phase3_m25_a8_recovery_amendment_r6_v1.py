from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from hegel_machine import phase3_m25_a8_recovery_amendment_r6_v1 as amendment
from hegel_machine import phase3_m25_a8_recovery_cli_r6_v1 as recovery_cli
from hegel_machine import phase3_m25_formal_container_executor_v1 as executor


@pytest.fixture(autouse=True)
def _freeze_pre_r7_required_input_view(monkeypatch: pytest.MonkeyPatch) -> None:
    future = {"phase3_m25_a8_recovery_amendment_r7_v1.py", "phase3_m25_a8_recovery_cli_r7_v1.py"}
    r3 = amendment._r4._r31
    monkeypatch.setattr(
        r3,
        "REQUIRED_COMMIT_A_INPUTS",
        tuple(path for path in r3.REQUIRED_COMMIT_A_INPUTS if path.name not in future),
    )


def test_r6_freezes_exact_r5_terminal_chain_and_failure_graph() -> None:
    rows = amendment._r5_terminal_chain_snapshot_v1()
    assert len(rows) == 8
    assert hashlib.sha256(amendment._canonical_json(rows)).hexdigest() == (
        amendment.R5_TERMINAL_CHAIN_ROOT_SHA256
    )
    assert amendment.R5_TERMINAL_CHAIN_ROOT_SHA256 == (
        "bcbe5e09f843b71e7448159307a02f698ace61fdccdff80767f3c826b6fb245b"
    )
    assert amendment.R5_FAILURE_CODE == "FormalStaticBasisError"
    assert amendment.R5_FORMAL_FAILURE_EVIDENCE_SHA256 == (
        "a64aed1283957993fb2fdd8eda72e4beceb29d9f5f90dc9cf5b6c82f4b234c37"
    )
    assert amendment.R5_FINAL_CLOSE_DETAIL_SHA256 == hashlib.sha256(
        b"actor identities remain without their bound Docker control plane"
    ).hexdigest()


def test_r6_source_admission_is_exact_executor_47_key_schema() -> None:
    unchanged = amendment._unchanged_a8_input_bindings_v1()
    admission = amendment._build_source_admission_v1(
        amendment_commit="11" * 20,
        incident_raw=b"incident",
        validation_raw=b"validation",
        static_qualification_raw=b"static",
        validation={
            "actor_report_sha256": "aa" * 32,
            "errata_report_sha256": "bb" * 32,
            "live_bundle_sha256": "cc" * 32,
        },
        unchanged_inputs=unchanged,
    )
    assert len(admission) == 47
    assert admission["schema"] == amendment.SOURCE_ADMISSION_SCHEMA
    assert admission["recovery_attempt_ordinal"] == 6
    assert admission["r5_terminal_chain_root_sha256"] == (
        amendment.R5_TERMINAL_CHAIN_ROOT_SHA256
    )
    executor._validate_recovery_source_admission_v1(
        admission,
        basis_commit=amendment.A8_BASIS_COMMIT,
        run_id=amendment.FIXED_RUN_ID,
        ledger_id=amendment.FIXED_LEDGER_ID,
    )
    malformed = dict(admission)
    malformed["static_preconsumption_qualification_sha256"] = "not-hex"
    with pytest.raises(executor.FormalContainerExecutorError):
        executor._validate_recovery_source_admission_v1(
            malformed,
            basis_commit=amendment.A8_BASIS_COMMIT,
            run_id=amendment.FIXED_RUN_ID,
            ledger_id=amendment.FIXED_LEDGER_ID,
        )
    for field in (
        "r5_formal_failure_evidence_sha256",
        "r5_final_close_failure_code",
        "r5_final_close_failure_detail_sha256",
    ):
        changed = dict(admission)
        changed[field] = "00" * 32
        with pytest.raises(
            executor.FormalContainerExecutorError,
            match="attempt-6 recovery source admission provenance differs",
        ):
            executor._validate_recovery_source_admission_v1(
                changed,
                basis_commit=amendment.A8_BASIS_COMMIT,
                run_id=amendment.FIXED_RUN_ID,
                ledger_id=amendment.FIXED_LEDGER_ID,
            )
    for field, confused in (
        ("recovery_attempt_ordinal", 6.0),
        ("recovery_attempt_ordinal", False),
        ("formal_identity_entropy_draw_count", 0.0),
        ("formal_identity_entropy_draw_count", False),
    ):
        changed = dict(admission)
        changed[field] = confused
        with pytest.raises(
            executor.FormalContainerExecutorError,
            match="attempt-6 recovery source admission provenance differs",
        ):
            executor._validate_recovery_source_admission_v1(
                changed,
                basis_commit=amendment.A8_BASIS_COMMIT,
                run_id=amendment.FIXED_RUN_ID,
                ledger_id=amendment.FIXED_LEDGER_ID,
            )


def test_r6_manifest_fixed_policy_rejects_bool_int_and_float_confusion(
    tmp_path: Path,
) -> None:
    manifest, _raw = amendment._load_manifest(amendment.DEFAULT_MANIFEST_PATH)
    for field, confused in (
        ("recovery_attempt_ordinal", 6.0),
        ("recovery_attempt_ordinal", False),
        ("formal_identity_entropy_draw_count", 0.0),
        ("formal_identity_entropy_draw_count", False),
    ):
        changed = dict(manifest)
        changed[field] = confused
        candidate = tmp_path / f"{field}-{type(confused).__name__}.json"
        candidate.write_bytes(amendment._canonical_json(changed))
        with pytest.raises(
            amendment.A8R6RecoveryAmendmentError,
            match="manifest fixed policy differs",
        ):
            amendment._load_manifest(candidate)


def test_r6_preserves_exact_95_input_and_15_exception_closures() -> None:
    unchanged = amendment._unchanged_a8_input_bindings_v1()
    assert len(unchanged) == 95
    assert hashlib.sha256(
        amendment._executor_canonical_json(unchanged)
    ).hexdigest() == amendment._r4._r31.EXPECTED_UNCHANGED_A8_INPUT_ROOT
    assert len(amendment.R6_RUNTIME_EXCEPTION_PATHS) == 15
    assert (
        amendment._r4._r31.R3_RUNTIME_EXCEPTION_PATHS
        == amendment._r4.R4_RUNTIME_EXCEPTION_PATHS
        == amendment._r5.R5_RUNTIME_EXCEPTION_PATHS
        == amendment.R6_RUNTIME_EXCEPTION_PATHS
    )
    assert {
        "Hegel Machine/src/hegel_machine/phase3_m25_a8_recovery_amendment_r6_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_m25_a8_recovery_cli_r6_v1.py",
    }.issubset(amendment.R6_RUNTIME_EXCEPTION_PATHS)


def test_r6_close_latch_only_follows_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def successful_close(_self: object) -> None:
        calls.append("success")

    monkeypatch.setattr(
        amendment.A8R1RecoveryDockerActorsV1, "close", successful_close
    )
    actors = amendment.A8R6RecoveryDockerActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust",
        timestamp=0,
    )
    actors.close()
    actors.close()
    assert calls == ["success"]

    failed_calls: list[str] = []

    def first_close_fails(_self: object) -> None:
        failed_calls.append("call")
        if len(failed_calls) == 1:
            raise executor.FormalContainerExecutorError(
                executor.FAIL_CONTAINER, "first close failed"
            )

    monkeypatch.setattr(
        amendment.A8R1RecoveryDockerActorsV1, "close", first_close_fails
    )
    failed = amendment.A8R6RecoveryDockerActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust",
        timestamp=0,
    )
    with pytest.raises(executor.FormalContainerExecutorError):
        failed.close()
    failed.close()
    assert failed_calls == ["call", "call"]


def test_static_preconsumption_uses_basis_path_without_actor_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound_binary = tmp_path / "bound-rust"
    bound_binary.write_bytes(b"fixed-rust")
    basis = SimpleNamespace(
        implementation_inputs={
            "rust_binary_path": str(bound_binary),
            "rust_binary_sha256": b"r" * 32,
        },
        gate19_plan=(object(),),
    )
    timeline: list[str] = []
    frozen_binding = b"d" * 32
    intent = {
        "live_actor_protocol_daemon_receipt_binding": frozen_binding,
        "rust_bridge_dag_qualification_report_sha256": "71" * 32,
        "runtime_binding_fields": {"runtime": "frozen"},
    }
    recovery = executor.PendingCeremonyRecoveryV1(
        basis_commit=amendment.A8_BASIS_COMMIT,
        run_id=amendment.FIXED_RUN_ID,
        ledger_id=amendment.FIXED_LEDGER_ID,
        marker_snapshot=executor.MarkerSnapshot(
            "PENDING", b"s" * 32, None, b"k" * 16, 0
        ),
        journal_state="RESERVED",
        stage_directory=tmp_path / "stage",
        custody_directory=tmp_path / "custody",
        public_evidence_path=tmp_path / "evidence",
        public_promotion_path=tmp_path / "promotion",
        prestage_intent_fields=intent,
        prestage_intent_sha256="70" * 32,
        actor_trust_checkpoint_fields={},
        lock_descriptor=41,
    )

    class FakeActors:
        authoritative = True
        _actor_start_attempted = False
        _containers: dict[int, str] = {}
        _state_volumes: dict[int, str] = {}

        def validate_rust_replay_binding(self, _basis: object) -> None:
            timeline.append("rust-binding")

        def validate_rust_bridge_dag_binding(self) -> None:
            timeline.append("bridge-binding")

        def unresolved_formal_blockers(self) -> tuple[()]:
            return ()

        def bridge_qualification_report_id_v1(self) -> bytes:
            return bytes.fromhex("71" * 32)

        def prestage_runtime_binding_fields_v1(self, _roots: object) -> dict[str, str]:
            return {"runtime": "frozen"}

        def validate_frozen_daemon_receipt_binding_v1(self, expected: bytes) -> None:
            assert expected == frozen_binding
            timeline.append("daemon-binding")

        def static_replay_control_plane_v1(self) -> tuple[object, bytes]:
            timeline.append("control-plane")
            return object(), frozen_binding

    actors = FakeActors()
    monkeypatch.setattr(executor, "DockerCeremonyActorsV1", FakeActors)
    monkeypatch.setattr(
        executor, "validate_prestage_intent_fields_v1", lambda *_a, **_k: intent
    )
    monkeypatch.setattr(
        executor,
        "build_qualified_formal_static_basis_v1",
        lambda *_args, **_kwargs: basis,
    )
    monkeypatch.setattr(
        executor, "require_formal_ceremony_ready_v1", lambda _basis: {"r": b"x" * 32}
    )
    monkeypatch.setattr(
        executor,
        "build_python_static_replay_receipt_v1",
        lambda _basis: timeline.append("python") or {"receipt_sha256": "11" * 32},
    )

    def rust_replay(_basis: object, **kwargs: object) -> dict[str, object]:
        timeline.append("rust")
        assert kwargs["rust_binary"] == bound_binary
        return {
            "entries": [{"stable": True}],
            "network_mode_none": True,
            "pull_policy_never": True,
            "seed_key_signature_or_state_created": False,
            "binary_sha256": "72" * 32,
            "container_image_ref_or_null": "image@sha256:" + "22" * 32,
            "receipt_sha256": "33" * 32,
        }

    monkeypatch.setattr(executor, "run_rust_static_replay_receipt_v1", rust_replay)
    roots = {f"root-{index}": bytes([index]) * 32 for index in range(6)}
    monkeypatch.setattr(
        executor,
        "validate_dual_static_replay_receipts_v1",
        lambda *_a: timeline.append("dual") or roots,
    )
    parent = SimpleNamespace(audit_bundle_root=b"p" * 32, audit_bundle_fields={})
    monkeypatch.setattr(
        executor, "generate_parent_absence_audit_v1", lambda _root: parent
    )
    monkeypatch.setattr(
        executor, "replay_parent_absence_audit_v1", lambda *_a, **_k: None
    )
    static_dual = executor._prevalidate_pending_recovery_static_dual_v1(
        recovery=recovery,
        actors=actors,
        static_rust_binary_path=bound_binary,
    )
    monkeypatch.setattr(amendment, "FIXED_FORMAL_RUST_BINARY", bound_binary)
    monkeypatch.setattr(
        amendment, "FIXED_FORMAL_RUST_BINARY_SHA256", (b"r" * 32).hex()
    )
    result = amendment._static_qualification_fields_v1(
        amendment_commit="11" * 20, static_dual=static_dual
    )
    assert timeline.index("rust") < timeline.index("dual")
    assert "start" not in timeline
    assert result["purpose_actor_start_attempted"] is False
    assert result["explicit_fixed_main_rust_path_passed"] is True
    assert result["dual_static_replay_validated"] is True


def test_executor_static_replay_is_before_actor_start_and_never_implicit() -> None:
    source = inspect.getsource(executor._continue_pre_stage_pending_recovery_core_v1)
    assert source.index("run_rust_static_replay_receipt_v1") < source.index(
        "actors.start()"
    )
    assert 'basis.implementation_inputs["rust_binary_path"]' in source
    module_source = Path(executor.__file__).read_text(encoding="utf-8")
    assert module_source.count(
        'rust_binary=Path(str(basis.implementation_inputs["rust_binary_path"]))'
    ) >= 1


def test_exact_receipt_bytes_reject_bool_int_type_confusion() -> None:
    expected = {"schema": "typed-test/1", "allowed": False}
    confused = {"schema": "typed-test/1", "allowed": 0}
    assert expected == confused
    confused_raw = amendment._receipt_record_bytes_v1(confused)
    with pytest.raises(
        amendment.A8R6RecoveryAmendmentError,
        match="canonical bytes differ",
    ):
        amendment._require_exact_receipt_raw_v1(
            confused_raw, expected, label="typed test"
        )


def test_formal_static_error_evidence_preserves_exact_code_and_detail() -> None:
    error = executor.FormalStaticBasisError(
        "FAIL_M25_STATIC_DUAL_RECEIPT", "dual roots differ"
    )
    evidence = executor.formal_failure_evidence_v1(error)
    assert evidence["kind"] == "SINGLE"
    assert evidence["code"] == "FAIL_M25_STATIC_DUAL_RECEIPT"
    assert evidence["detail_sha256"] == hashlib.sha256(
        b"dual roots differ"
    ).hexdigest()


def test_static_prevalidation_maps_structured_error_before_actor_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent = {
        "live_actor_protocol_daemon_receipt_binding": b"d" * 32,
        "rust_bridge_dag_qualification_report_sha256": "71" * 32,
        "runtime_binding_fields": {},
    }
    recovery = executor.PendingCeremonyRecoveryV1(
        basis_commit=amendment.A8_BASIS_COMMIT,
        run_id=amendment.FIXED_RUN_ID,
        ledger_id=amendment.FIXED_LEDGER_ID,
        marker_snapshot=executor.MarkerSnapshot(
            "PENDING", b"s" * 32, None, b"k" * 16, 0
        ),
        journal_state="RESERVED",
        stage_directory=tmp_path / "stage",
        custody_directory=tmp_path / "custody",
        public_evidence_path=tmp_path / "evidence",
        public_promotion_path=tmp_path / "promotion",
        prestage_intent_fields=intent,
        prestage_intent_sha256="70" * 32,
        actor_trust_checkpoint_fields={},
        lock_descriptor=41,
    )

    class FakeActors:
        authoritative = True
        _actor_start_attempted = False
        _containers: dict[int, str] = {}
        _state_volumes: dict[int, str] = {}

    actors = FakeActors()
    monkeypatch.setattr(executor, "DockerCeremonyActorsV1", FakeActors)
    monkeypatch.setattr(
        executor, "validate_prestage_intent_fields_v1", lambda *_a, **_k: intent
    )

    def fail_basis(*_args: object, **_kwargs: object) -> object:
        raise executor.FormalStaticBasisError(
            "FAIL_M25_STATIC_RUST_REPLAY_POLICY", "bound path differs"
        )

    monkeypatch.setattr(
        executor, "build_qualified_formal_static_basis_v1", fail_basis
    )
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        executor._prevalidate_pending_recovery_static_dual_v1(
            recovery=recovery,
            actors=actors,
            static_rust_binary_path=tmp_path / "rust",
        )
    assert captured.value.code == "FAIL_M25_STATIC_RUST_REPLAY_POLICY"
    assert captured.value.detail == "bound path differs"
    assert actors._actor_start_attempted is False


def test_prevalidated_formal_prefix_is_same_process_one_shot(
    tmp_path: Path,
) -> None:
    rust_binary = tmp_path / "rust"
    rust_binary.write_bytes(b"rust")
    recovery = executor.PendingCeremonyRecoveryV1(
        basis_commit=amendment.A8_BASIS_COMMIT,
        run_id=amendment.FIXED_RUN_ID,
        ledger_id=amendment.FIXED_LEDGER_ID,
        marker_snapshot=executor.MarkerSnapshot(
            "PENDING", b"s" * 32, None, b"k" * 16, 0
        ),
        journal_state="RESERVED",
        stage_directory=tmp_path / "stage",
        custody_directory=tmp_path / "custody",
        public_evidence_path=tmp_path / "evidence",
        public_promotion_path=tmp_path / "promotion",
        prestage_intent_fields={},
        prestage_intent_sha256="70" * 32,
        actor_trust_checkpoint_fields={},
        lock_descriptor=41,
    )
    actors = object()
    python_receipt = {"receipt_sha256": "11" * 32}
    rust_receipt = {"receipt_sha256": "22" * 32}
    static_dual = executor._PrevalidatedPendingRecoveryStaticDualV1(
        recovery=recovery,
        actors=actors,
        basis=SimpleNamespace(
            implementation_inputs={"rust_binary_path": str(rust_binary)}
        ),
        implementation_roots={},
        python_receipt=python_receipt,
        python_receipt_bytes=executor._canonical_json(python_receipt),
        rust_receipt=rust_receipt,
        rust_receipt_bytes=executor._canonical_json(rust_receipt),
        static_roots={},
        parent_absence=SimpleNamespace(),
        frozen_daemon_binding=b"d" * 32,
        static_daemon_binding=b"d" * 32,
        prestage_intent_sha256=recovery.prestage_intent_sha256,
        _seal=executor._PREVALIDATED_PENDING_STATIC_DUAL_SEAL,
    )
    fixed_capability = SimpleNamespace(
        _seal=executor._FIXED_A8_R3_PREVALIDATED_SEAL,
        basis_commit=recovery.basis_commit,
        run_id=recovery.run_id,
        ledger_id=recovery.ledger_id,
    )
    admission = {"schema": amendment.SOURCE_ADMISSION_SCHEMA}
    prefix = executor._PrevalidatedPendingRecoveryFormalPrefixV1(
        recovery=recovery,
        actors=actors,
        source_admission_bytes=executor._canonical_json(admission),
        static_dual=static_dual,
        fixed_capability=fixed_capability,
        _seal=executor._PREVALIDATED_PENDING_FORMAL_PREFIX_SEAL,
    )
    consumed_static, consumed_fixed = (
        executor._consume_prevalidated_pending_recovery_formal_prefix_v1(
            recovery=recovery,
            actors=actors,
            source_admission=admission,
            static_rust_binary_path=rust_binary,
            prefix=prefix,
        )
    )
    assert consumed_static is static_dual
    assert consumed_fixed is fixed_capability
    with pytest.raises(
        executor.FormalContainerExecutorError,
        match="already consumed",
    ):
        executor._consume_prevalidated_pending_recovery_formal_prefix_v1(
            recovery=recovery,
            actors=actors,
            source_admission=admission,
            static_rust_binary_path=rust_binary,
            prefix=prefix,
        )


def test_r6_core_uses_sealed_source_bytes_without_callback_or_revalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent = {
        "created_at_unix_seconds": 0,
        "trust_genesis_id_hex": "11" * 16,
        "actor_qualification_report": {},
        "errata_qualification_report": {},
        "rust_bridge_dag_qualification_report_sha256": "22" * 32,
        "live_actor_protocol_qualification_bundle_content_id": b"b" * 32,
        "live_actor_protocol_qualification_bundle": {},
        "live_actor_protocol_daemon_receipt_binding": b"d" * 32,
    }
    recovery = executor.PendingCeremonyRecoveryV1(
        basis_commit=amendment.A8_BASIS_COMMIT,
        run_id=amendment.FIXED_RUN_ID,
        ledger_id=amendment.FIXED_LEDGER_ID,
        marker_snapshot=executor.MarkerSnapshot(
            "PENDING", b"s" * 32, None, b"k" * 16, 0
        ),
        journal_state="RESERVED",
        stage_directory=tmp_path / "stage",
        custody_directory=tmp_path / "custody",
        public_evidence_path=tmp_path / "evidence",
        public_promotion_path=tmp_path / "promotion",
        prestage_intent_fields=intent,
        prestage_intent_sha256="70" * 32,
        actor_trust_checkpoint_fields={},
        lock_descriptor=41,
    )
    actors = SimpleNamespace(authoritative=True)
    admission = {"schema": amendment.SOURCE_ADMISSION_SCHEMA}
    prefix = executor._PrevalidatedPendingRecoveryFormalPrefixV1(
        recovery=recovery,
        actors=actors,
        source_admission_bytes=executor._canonical_json(admission),
        static_dual=SimpleNamespace(),
        fixed_capability=SimpleNamespace(),
        _seal=executor._PREVALIDATED_PENDING_FORMAL_PREFIX_SEAL,
    )
    monkeypatch.setattr(
        executor,
        "_qualification_only_key_ids_from_intent_v1",
        lambda _intent: {},
    )

    def forbidden_validator(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("sealed R6 core repeated source validation")

    monkeypatch.setattr(
        executor, "_validate_recovery_source_admission_v1", forbidden_validator
    )

    class ConsumeReached(RuntimeError):
        pass

    def stop_at_consume(**kwargs: object) -> object:
        assert kwargs["source_admission"] == admission
        raise ConsumeReached

    monkeypatch.setattr(
        executor,
        "_consume_prevalidated_pending_recovery_formal_prefix_v1",
        stop_at_consume,
    )
    with pytest.raises(ConsumeReached):
        executor._continue_pre_stage_pending_recovery_core_v1(
            recovery=recovery,
            actors=actors,
            complete_seed_resume_only=True,
            static_rust_binary_path=tmp_path / "rust",
            prevalidated_formal_prefix=prefix,
        )


def test_under_lock_audit_recheck_rejects_same_byte_inode_replacement(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "audit"
    audit.mkdir(mode=0o700)
    for name in amendment._R6_PRE_ATTEMPT_AUDIT_NAMES:
        path = audit / name
        path.write_bytes(amendment._receipt_record_bytes_v1({"schema": name}))
        path.chmod(0o600)
    directory_identity, raws, inodes = (
        amendment._read_pre_attempt_audit_snapshot_v1(
            audit, allow_attempt_next=False
        )
    )
    target = audit / "authorization.json"
    original = target.read_bytes()
    replacement = audit / "replacement.tmp"
    replacement.write_bytes(original)
    replacement.chmod(0o600)
    replacement.replace(target)
    with pytest.raises(
        amendment.A8R6RecoveryAmendmentError,
        match="changed under recovery lock",
    ):
        amendment._recheck_pre_attempt_audit_under_lock_v1(
            audit=audit,
            expected_directory_identity=directory_identity,
            expected_raws=raws,
            expected_inodes=inodes,
        )


def test_r6_cli_prepare_requires_seed_free_runtime_bindings() -> None:
    parser = recovery_cli._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "prepare-authorization",
                "--audit-directory", "/tmp/audit",
                "--custody-directory", "/tmp/custody",
                "--public-evidence-output", "/tmp/evidence",
                "--promotion-output", "/tmp/promotion",
            ]
        )


def test_r6_policy_never_authorizes_redraw_or_m3() -> None:
    policy = amendment._manifest_fixed_policy_v1()
    assert policy["recovery_attempt_ordinal"] == 6
    assert policy["sole_parent_commit"] == amendment.R5_AMENDMENT_COMMIT
    assert policy["redraw_allowed"] is False
    assert policy["ordinary_execute_allowed"] is False
    assert policy["m3_start_allowed"] is False
