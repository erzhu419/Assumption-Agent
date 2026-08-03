from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile

import pytest

import hegel_machine.phase3_m3_shadow_runtime_v1 as shadow


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILE = ROOT / "pyproject.toml"
PYTHON_CALCULATOR = ROOT / "tools" / "phase3_split_calculator_fd3_v1.py"
RUST_CALCULATOR_SOURCE = ROOT / "tools" / "phase3_split_calculator_fd3_v1.rs"
PINNED_RUST_IMAGE = (
    "rust@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89"
)


def _assert_code(
    error: pytest.ExceptionInfo[shadow.ShadowRuntimeError], code: str
) -> None:
    assert error.value.code == code


@pytest.fixture
def private_parent() -> Path:
    with tempfile.TemporaryDirectory(
        prefix="hegel-shadow-test-", dir="/tmp"
    ) as value:
        path = Path(value)
        os.chmod(path, 0o700)
        yield path


@pytest.fixture(scope="session")
def calculator_endpoints(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    output_dir = tmp_path_factory.mktemp("shadow-runtime-rust-fd3")
    binary = output_dir / "phase3_split_calculator_fd3_v1"
    docker = shutil.which("docker")
    assert docker is not None
    completed = subprocess.run(
        [
            docker,
            "run",
            "--rm",
            "--pull=never",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            f"--user={os.getuid()}:{os.getgid()}",
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m",
            f"--mount=type=bind,src={RUST_CALCULATOR_SOURCE.parent},dst=/src,readonly",
            f"--mount=type=bind,src={output_dir},dst=/out",
            PINNED_RUST_IMAGE,
            "rustc",
            "--edition=2021",
            "-C",
            "opt-level=2",
            "-C",
            "debuginfo=0",
            "-C",
            "strip=symbols",
            "-C",
            "codegen-units=1",
            "-o",
            "/out/phase3_split_calculator_fd3_v1",
            "/src/phase3_split_calculator_fd3_v1.rs",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    return PYTHON_CALCULATOR.resolve(), binary.resolve()


@pytest.fixture(scope="module")
def completed_ceremony(
    calculator_endpoints: tuple[Path, Path],
) -> tuple[dict[str, object], Path, tempfile.TemporaryDirectory[str]]:
    owner = tempfile.TemporaryDirectory(prefix="hegel-shadow-complete-", dir="/tmp")
    parent = Path(owner.name)
    os.chmod(parent, 0o700)
    state = shadow.create_shadow_state_directory(parent / "state")
    report = shadow.run_internal_shadow_ceremony(
        state_directory=state,
        input_files={"basis": SOURCE_FILE},
        python_calculator_path=calculator_endpoints[0],
        rust_calculator_path=calculator_endpoints[1],
        ceremony_id=bytes.fromhex("42" * 16),
    )
    yield report, state, owner
    owner.cleanup()


def test_side_effect_free_admission_runs_four_strong_probes(
    calculator_endpoints: tuple[Path, Path],
) -> None:
    report = shadow.probe_shadow_admission(
        basis_commit_id="1" * 40,
        shadow_run_id=bytes.fromhex("11" * 16),
        input_files={"basis": SOURCE_FILE},
        python_calculator_path=calculator_endpoints[0],
        rust_calculator_path=calculator_endpoints[1],
    )
    shadow.validate_shadow_admission_report(report)

    assert report["admission_status"] == "INTERNAL_SHADOW_ADMISSION_PASS"
    assert report["authority_class"] == shadow.AUTHORITY_CLASS
    assert report["basis_commit_id_or_null"] == "1" * 40
    assert [item["purpose_id"] for item in report["purpose_probe_receipts"]] == [
        1,
        2,
        3,
        4,
    ]
    assert report["side_effects"] == {
        "key_generated": False,
        "seed_generated": False,
        "marker_written": False,
        "formal_root_issued": False,
    }
    assert report["authority_boundary"]["formal_gates_before"] == 14
    assert report["authority_boundary"]["formal_gates_after"] == 14
    assert report["authority_boundary"]["formal_gate_delta"] == 0
    assert report["authority_boundary"]["formal_state_after"] == "NOT_RUN"
    assert report["isolation_plan_inputs"]["network_fetch_allowed"] is False

    for receipt in report["purpose_probe_receipts"]:
        evidence = receipt["process"]["security_evidence"]
        assert evidence["isolation_level"] == shadow.ISOLATION_LEVEL
        assert evidence["seccomp_mode"] == 2
        assert evidence["no_new_privs"] == 1
        assert set(evidence["capability_status_hex"].values()) == {
            "0000000000000000"
        }
        assert evidence["attack_syscall_probe_count"] == 6
        assert {row["errno"] for row in evidence["attack_syscall_errno_rows"]} == {
            "EPERM"
        }
        assert evidence["repository_mount_read_only_live_probe"] is True
        assert evidence["tmp_mount_type"] == "tmpfs"
        assert evidence["network_interfaces"] == ["lo"]
        assert evidence["network_fetch_allowed"] is False


def test_completed_runtime_is_replayable_and_never_advances_formal_track(
    completed_ceremony: tuple[
        dict[str, object], Path, tempfile.TemporaryDirectory[str]
    ],
) -> None:
    report, state, _owner = completed_ceremony
    shadow.validate_shadow_runtime_report(report)

    assert report["shadow_status"] == "INTERNAL_SHADOW_CEREMONY_COMPLETE"
    assert report["authority_class"] == shadow.AUTHORITY_CLASS
    assert report["fresh_admission_probes"]["admission_status"] == (
        "INTERNAL_SHADOW_ADMISSION_PASS"
    )
    boundary = report["authority_boundary"]
    assert boundary["formal_gates_before"] == 14
    assert boundary["formal_gates_after"] == 14
    assert boundary["formal_gates_total"] == 24
    assert boundary["formal_gate_delta"] == 0
    assert boundary["formal_roots_issued"] is False
    assert boundary["external_actor_evidence"] is False
    assert boundary["formal_state_before"] == "NOT_RUN"
    assert boundary["formal_state_after"] == "NOT_RUN"
    assert boundary["formal_transition"] is None
    assert boundary["formal_m3_start_allowed"] is False
    assert boundary["report_alone_authorizes_formal_execution"] is False

    envelopes = report["envelopes"]
    replayed = [shadow.verify_shadow_envelope(item) for item in envelopes]
    assert [item["purpose_id"] for item in replayed] == [1, 2, 3, 4]
    assert len({item["key_id_hex"] for item in replayed}) == 4
    assert len({item["worker_instance_id_hex"] for item in replayed}) == 4
    assert all(len(item["security_evidence_sha256_hex"]) == 64 for item in replayed)

    calculators = report["calculator_agreement"]
    assert calculators["agreement"] is True
    assert [worker["secret_input_fd"] for worker in calculators["workers"]] == [3, 3]
    assert [worker["seccomp_mode"] for worker in calculators["workers"]] == [2, 2]
    assert len(
        {worker["seed_commitment_sha256_hex"] for worker in calculators["workers"]}
    ) == 1

    state_receipt = shadow.inspect_shadow_state(state)
    assert state_receipt == report["state_evidence"]
    assert state_receipt["status"] == "COMPLETE"
    assert state_receipt["ledger_entry_count"] == 2
    assert state_receipt["contains_raw_seed"] is False
    assert state_receipt["contains_private_key"] is False
    assert {item.name for item in state.iterdir()} == {
        shadow.MARKER_FILE_NAME,
        shadow.LEDGER_FILE_NAME,
    }
    assert stat.S_IMODE((state / shadow.MARKER_FILE_NAME).stat().st_mode) == 0o600
    assert stat.S_IMODE((state / shadow.LEDGER_FILE_NAME).stat().st_mode) == 0o600


def test_runtime_report_contains_no_serialized_private_material(
    completed_ceremony: tuple[
        dict[str, object], Path, tempfile.TemporaryDirectory[str]
    ],
) -> None:
    report, state, _owner = completed_ceremony
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":"))
    assert "BEGIN PRIVATE KEY" not in encoded
    assert "ed25519_private" not in encoded.lower()
    assert "raw_seed_hex" not in encoded
    assert "private_key_hex" not in encoded
    for path in (state / shadow.MARKER_FILE_NAME, state / shadow.LEDGER_FILE_NAME):
        payload = path.read_bytes()
        assert b"BEGIN PRIVATE KEY" not in payload
        assert b"raw_seed" not in payload


def test_completed_state_is_one_shot_and_never_redraws(
    completed_ceremony: tuple[
        dict[str, object], Path, tempfile.TemporaryDirectory[str]
    ],
    calculator_endpoints: tuple[Path, Path],
) -> None:
    _report, state, _owner = completed_ceremony
    before = {
        path.name: path.read_bytes()
        for path in (state / shadow.MARKER_FILE_NAME, state / shadow.LEDGER_FILE_NAME)
    }
    with pytest.raises(shadow.ShadowRuntimeError) as error:
        shadow.run_internal_shadow_ceremony(
            state_directory=state,
            input_files={"basis": SOURCE_FILE},
            python_calculator_path=calculator_endpoints[0],
            rust_calculator_path=calculator_endpoints[1],
            ceremony_id=bytes.fromhex("43" * 16),
        )
    _assert_code(error, shadow.FAIL_SHADOW_STATE_ALREADY_COMPLETE)
    after = {
        path.name: path.read_bytes()
        for path in (state / shadow.MARKER_FILE_NAME, state / shadow.LEDGER_FILE_NAME)
    }
    assert after == before


def test_pending_state_requires_recovery_and_is_not_rolled_back(
    private_parent: Path,
    calculator_endpoints: tuple[Path, Path],
) -> None:
    state = shadow.create_shadow_state_directory(private_parent / "pending")
    ceremony_id = bytes.fromhex("21" * 16)
    snapshot_digest = bytes.fromhex("22" * 32)
    shadow._create_pending_state(state, ceremony_id, snapshot_digest)

    receipt = shadow.inspect_shadow_state(state)
    assert receipt["status"] == "PENDING"
    assert receipt["ledger_entry_count"] == 1
    assert receipt["seed_commitment_sha256_hex_or_null"] is None
    with pytest.raises(shadow.ShadowRuntimeError) as error:
        shadow.run_internal_shadow_ceremony(
            state_directory=state,
            input_files={"basis": SOURCE_FILE},
            python_calculator_path=calculator_endpoints[0],
            rust_calculator_path=calculator_endpoints[1],
            ceremony_id=ceremony_id,
        )
    _assert_code(error, shadow.FAIL_SHADOW_STATE_PENDING_RECOVERY_REQUIRED)
    assert shadow.inspect_shadow_state(state)["status"] == "PENDING"


def test_pending_to_complete_marker_and_ledger_are_exact(
    private_parent: Path,
) -> None:
    state = shadow.create_shadow_state_directory(private_parent / "transition")
    ceremony_id = bytes.fromhex("31" * 16)
    snapshot_digest = bytes.fromhex("32" * 32)
    commitment = bytes.fromhex("33" * 32)
    shadow._create_pending_state(state, ceremony_id, snapshot_digest)
    complete = shadow._complete_private_state(
        state, ceremony_id, snapshot_digest, commitment
    )
    assert complete["status"] == "COMPLETE"
    assert complete["seed_commitment_sha256_hex_or_null"] == commitment.hex()
    assert complete["ledger_entry_count"] == 2
    with pytest.raises(shadow.ShadowRuntimeError) as error:
        shadow._create_pending_state(state, ceremony_id, snapshot_digest)
    _assert_code(error, shadow.FAIL_SHADOW_STATE_ALREADY_COMPLETE)


def test_snapshot_hook_mutation_fails_before_marker(
    private_parent: Path,
    calculator_endpoints: tuple[Path, Path],
) -> None:
    state = shadow.create_shadow_state_directory(private_parent / "snapshot-mutated")

    def mutate(snapshot_dir: Path, _report: object) -> None:
        target = snapshot_dir / "basis"
        os.chmod(snapshot_dir, 0o700)
        os.chmod(target, 0o600)
        target.write_bytes(b"mutated")

    with pytest.raises(shadow.ShadowRuntimeError) as error:
        shadow.run_internal_shadow_ceremony(
            state_directory=state,
            input_files={"basis": SOURCE_FILE},
            python_calculator_path=calculator_endpoints[0],
            rust_calculator_path=calculator_endpoints[1],
            ceremony_id=bytes.fromhex("51" * 16),
            snapshot_hook=mutate,
        )
    _assert_code(error, shadow.FAIL_SHADOW_SNAPSHOT_MUTATED)
    assert not (state / shadow.MARKER_FILE_NAME).exists()
    assert not (state / shadow.LEDGER_FILE_NAME).exists()


def test_snapshot_rejects_symlink_and_invalid_label(private_parent: Path) -> None:
    link = private_parent / "source-link"
    link.symlink_to(SOURCE_FILE)
    with pytest.raises(shadow.ShadowRuntimeError) as error:
        shadow.create_readonly_input_snapshot(
            {"basis": link}, private_parent / "symlink-snapshot"
        )
    _assert_code(error, shadow.FAIL_SHADOW_SNAPSHOT_SOURCE_INVALID)

    with pytest.raises(shadow.ShadowRuntimeError) as error:
        shadow.create_readonly_input_snapshot(
            {"../escape": SOURCE_FILE}, private_parent / "label-snapshot"
        )
    _assert_code(error, shadow.FAIL_SHADOW_SNAPSHOT_SOURCE_INVALID)


def test_crypto_and_bwrap_absence_fail_before_private_state(
    private_parent: Path,
    monkeypatch: pytest.MonkeyPatch,
    calculator_endpoints: tuple[Path, Path],
) -> None:
    crypto_state = shadow.create_shadow_state_directory(private_parent / "no-crypto")
    monkeypatch.setattr(shadow, "_Ed25519PrivateKey", None)
    with pytest.raises(shadow.ShadowRuntimeError) as error:
        shadow.run_internal_shadow_ceremony(
            state_directory=crypto_state,
            input_files={"basis": SOURCE_FILE},
            python_calculator_path=calculator_endpoints[0],
            rust_calculator_path=calculator_endpoints[1],
            ceremony_id=bytes.fromhex("61" * 16),
        )
    _assert_code(error, shadow.FAIL_SHADOW_CRYPTO_BACKEND_UNAVAILABLE)
    assert list(crypto_state.iterdir()) == []
    monkeypatch.undo()

    bwrap_state = shadow.create_shadow_state_directory(private_parent / "no-bwrap")
    monkeypatch.setattr(shadow.shutil, "which", lambda _name: None)
    with pytest.raises(shadow.ShadowRuntimeError) as error:
        shadow.run_internal_shadow_ceremony(
            state_directory=bwrap_state,
            input_files={"basis": SOURCE_FILE},
            python_calculator_path=calculator_endpoints[0],
            rust_calculator_path=calculator_endpoints[1],
            ceremony_id=bytes.fromhex("62" * 16),
        )
    _assert_code(error, shadow.FAIL_SHADOW_BWRAP_UNAVAILABLE)
    assert list(bwrap_state.iterdir()) == []


def test_missing_rust_endpoint_fails_before_seed_or_marker(
    private_parent: Path,
) -> None:
    state = shadow.create_shadow_state_directory(private_parent / "no-rust-endpoint")
    with pytest.raises(shadow.ShadowRuntimeError) as error:
        shadow.run_internal_shadow_ceremony(
            state_directory=state,
            input_files={"basis": SOURCE_FILE},
            python_calculator_path=PYTHON_CALCULATOR.resolve(),
            rust_calculator_path=private_parent / "absent-rust-calculator",
            ceremony_id=bytes.fromhex("63" * 16),
        )
    _assert_code(error, shadow.FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE)
    assert list(state.iterdir()) == []


def test_state_path_and_permissions_fail_closed(private_parent: Path) -> None:
    inside_repo = ROOT / ".forbidden-shadow-state"
    with pytest.raises(shadow.ShadowRuntimeError) as error:
        shadow.create_shadow_state_directory(inside_repo)
    _assert_code(error, shadow.FAIL_SHADOW_STATE_INSIDE_REPOSITORY)
    assert not inside_repo.exists()

    weak = private_parent / "weak"
    weak.mkdir(mode=0o700)
    os.chmod(weak, 0o755)
    with pytest.raises(shadow.ShadowRuntimeError) as error:
        shadow.create_shadow_state_directory(weak)
    _assert_code(error, shadow.FAIL_SHADOW_STATE_PERMISSIONS)


def test_calculator_disagreement_is_stable_and_fail_closed() -> None:
    base = {
        "environment_keys": sorted(shadow.SANITIZED_ENVIRONMENT),
        "working_directory_mode_octal": "0700",
        "secret_input_fd": 3,
        "public_output_fd": 5,
        "unexpected_inherited_fd_count": 0,
        "seccomp_mode": 2,
    }
    workers = [
        {
            **base,
            "calculator_id": "PYTHON_FD3_ENDPOINT_V1",
            "process_id": 10,
            "seed_commitment_sha256_hex": "01" * 32,
        },
        {
            **base,
            "calculator_id": "RUST_FD3_ENDPOINT_V1",
            "process_id": 11,
            "seed_commitment_sha256_hex": "02" * 32,
        },
    ]
    with pytest.raises(shadow.ShadowRuntimeError) as error:
        shadow._build_calculator_agreement(bytes.fromhex("01" * 16), bytes(32), workers)
    _assert_code(error, shadow.FAIL_SHADOW_CALCULATOR_DISAGREEMENT)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda report: report["authority_boundary"].__setitem__(
            "formal_gate_delta", 1
        ),
        lambda report: report["authority_boundary"].__setitem__(
            "formal_state_after", "RUNNING"
        ),
        lambda report: report["authority_boundary"].__setitem__(
            "formal_roots_issued", True
        ),
        lambda report: report["process_isolation"]["required_security"].__setitem__(
            "seccomp_required", False
        ),
        lambda report: report["process_isolation"]["role_processes"][0][
            "security_evidence"
        ].__setitem__("seccomp_mode", 0),
    ),
)
def test_runtime_authority_and_security_tampering_is_rejected(
    completed_ceremony: tuple[
        dict[str, object], Path, tempfile.TemporaryDirectory[str]
    ],
    mutation,
) -> None:
    report = deepcopy(completed_ceremony[0])
    mutation(report)
    with pytest.raises(shadow.ShadowRuntimeError):
        shadow.validate_shadow_runtime_report(report)


def test_envelope_signature_and_field_set_tampering_is_rejected(
    completed_ceremony: tuple[
        dict[str, object], Path, tempfile.TemporaryDirectory[str]
    ],
) -> None:
    envelope = deepcopy(completed_ceremony[0]["envelopes"][0])
    signature = bytearray.fromhex(envelope["signature_hex"])
    signature[-1] ^= 1
    envelope["signature_hex"] = bytes(signature).hex()
    with pytest.raises(shadow.ShadowRuntimeError) as error:
        shadow.verify_shadow_envelope(envelope)
    _assert_code(error, shadow.FAIL_SHADOW_SIGNATURE_INVALID)

    envelope = deepcopy(completed_ceremony[0]["envelopes"][0])
    envelope["override"] = True
    with pytest.raises(shadow.ShadowRuntimeError) as error:
        shadow.verify_shadow_envelope(envelope)
    _assert_code(error, shadow.FAIL_SHADOW_SIGNATURE_INVALID)


def test_admission_tampering_and_type_confusion_are_rejected(
    completed_ceremony: tuple[
        dict[str, object], Path, tempfile.TemporaryDirectory[str]
    ],
) -> None:
    original = completed_ceremony[0]["fresh_admission_probes"]
    report = deepcopy(original)
    report["side_effects"]["key_generated"] = True
    with pytest.raises(shadow.ShadowRuntimeError) as error:
        shadow.validate_shadow_admission_report(report)
    _assert_code(error, shadow.FAIL_SHADOW_AUTHORITY_ESCALATION)

    report = deepcopy(original)
    report["authority_boundary"]["formal_gates_after"] = 14.0
    with pytest.raises(shadow.ShadowRuntimeError) as error:
        shadow.validate_shadow_admission_report(report)
    _assert_code(error, shadow.FAIL_SHADOW_AUTHORITY_ESCALATION)

    report = deepcopy(original)
    report["override"] = True
    with pytest.raises(shadow.ShadowRuntimeError) as error:
        shadow.validate_shadow_admission_report(report)
    _assert_code(error, shadow.FAIL_SHADOW_REPORT_INVALID)


def test_marker_and_ledger_tampering_fail_closed(private_parent: Path) -> None:
    state = shadow.create_shadow_state_directory(private_parent / "tampered")
    shadow._create_pending_state(state, bytes.fromhex("71" * 16), bytes.fromhex("72" * 32))
    marker = state / shadow.MARKER_FILE_NAME
    marker.write_bytes(marker.read_bytes() + b"\x00")
    os.chmod(marker, 0o600)
    with pytest.raises(shadow.ShadowRuntimeError) as error:
        shadow.inspect_shadow_state(state)
    _assert_code(error, shadow.FAIL_SHADOW_MARKER_TAMPERED)


def test_snapshot_report_digest_and_exact_fields_are_replayed(
    private_parent: Path,
) -> None:
    report = shadow.create_readonly_input_snapshot(
        {"basis": SOURCE_FILE}, private_parent / "snapshot"
    )
    shadow.validate_readonly_input_snapshot_report(report)
    forged = deepcopy(report)
    forged["manifest_sha256_hex"] = "00" * 32
    with pytest.raises(shadow.ShadowRuntimeError) as error:
        shadow.validate_readonly_input_snapshot_report(forged)
    _assert_code(error, shadow.FAIL_SHADOW_REPORT_INVALID)

    forged = deepcopy(report)
    forged["extra"] = None
    with pytest.raises(shadow.ShadowRuntimeError) as error:
        shadow.validate_readonly_input_snapshot_report(forged)
    _assert_code(error, shadow.FAIL_SHADOW_REPORT_INVALID)
