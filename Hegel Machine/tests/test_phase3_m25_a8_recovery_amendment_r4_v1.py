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
        "AUTHORIZE_A8_R4_ATTEMPT_4_CANONICAL_BYTES_"
        "COMPLETE_ONLY_REAL_PENDING_RESUME"
    )
    assert amendment.AUTHORIZATION_REVISION_ID == (
        "R4_CANONICAL_AUDIT_INSTALLER_V1"
    )
    assert amendment.FIXED_R4_AUDIT_DIRECTORY != (
        amendment.R31_TERMINAL_AUDIT_DIRECTORY
    )


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
    incident_path = audit / "incident-diagnostic.json"
    attempt_path = audit / "attempt-start.json"
    if not incident_path.is_file() or not attempt_path.is_file():
        pytest.skip("fixed R3.1 terminal audit is not present")
    incident = json.loads(incident_path.read_bytes())
    stored = json.loads(attempt_path.read_bytes())
    runtime_rows = amendment._r31._r2._validate_runtime_artifacts_before_attempt_v1(
        rust_formal_replay_binary=Path(
            "/home/erzhu419/mine_code/Asumption Agent/Hegel Machine/rust/"
            "formal_bridge_m25/target/debug/hegel-formal-bridge-m25"
        ),
        rust_bridge_dag_replay_binary=Path(
            "/home/erzhu419/mine_code/Asumption Agent/Hegel Machine/rust/"
            "m25_bridge_dag_replay/target/commit_a_qualified/"
            "hegel-m25-bridge-dag-replay"
        ),
        rust_bridge_dag_qualification_report=Path(
            "/home/erzhu419/mine_code/Asumption Agent/Hegel Machine/artifacts/"
            "phase3_m25_external/"
            "phase3_m25_bridge_dag_rust_binary_qualification_v1.json"
        ),
        expected_bindings=incident["runtime_artifact_bindings"],
    )
    rebuilt = dict(stored)
    rebuilt["runtime_artifact_metadata"] = runtime_rows
    assert type(stored["runtime_artifact_metadata"]) is list
    assert type(runtime_rows) is tuple
    assert stored != rebuilt
    assert _canonical(stored) == _canonical(rebuilt)
    assert [key for key in stored if stored[key] != rebuilt[key]] == [
        "runtime_artifact_metadata"
    ]


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
    incident = {
        "schema": f"{amendment.AUDIT_SCHEMA_PREFIX}-incident-diagnostic/1",
        "stage_directory": "/fixed/stage",
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
    monkeypatch.setattr(
        amendment,
        "_validate_runtime_artifacts_before_attempt_v1",
        lambda **_kwargs: (
            {
                "diagnostic_sha256_or_null": None,
                "mode_octal": "0755",
                "path": "/fixed/hegel-formal-bridge-m25",
                "sha256": "32" * 32,
            },
        ),
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
