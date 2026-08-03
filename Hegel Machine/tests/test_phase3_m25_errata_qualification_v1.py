from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from hegel_machine import phase3_m25_errata_qualification_v1 as qualification
from hegel_machine.cli import main
from hegel_machine.hashing import stable_hash
from hegel_machine.phase3_m25_errata_qualification_v1 import (
    EXPECTED_VECTOR_COUNTS,
    M25ErrataQualificationError,
    dual_errata_qualification_report,
    validate_checked_errata_qualification_report,
    validate_dual_errata_qualification_report,
)
from hegel_machine.phase3_m25_errata_vectors_v1 import (
    generate_errata_vector_report_v1,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKED_IN = ROOT / "artifacts" / "phase3_m25_errata_qualification_v1.json"


@pytest.fixture(scope="session")
def m25_commit_a_ready() -> None:
    try:
        qualification._assert_sources_match_commit(
            qualification.repository_head_commit()
        )
        with qualification.LinuxLocalTemporaryDirectoryV1(
            prefix="hegel-m25-test-preflight-",
            repository_root=qualification.PROJECT_ROOT.parent,
        ) as raw:
            control_plane = qualification.prepare_local_docker_control_plane_v1(
                Path(raw),
                repository_root=qualification.PROJECT_ROOT.parent,
            )
            qualification._approved_rust_toolchain(control_plane)
    except (M25ErrataQualificationError, qualification.Phase3LocalRuntimeError) as exc:
        pytest.skip(f"Commit-A OCI qualification preflight is not ready: {exc}")


@pytest.fixture(scope="session")
def m25_errata_report(m25_commit_a_ready: None) -> dict[str, object]:
    return dual_errata_qualification_report()


def test_python_errata_report_matches_checked_golden_exactly() -> None:
    fixture = json.loads(qualification.GOLDEN_VECTOR_PATH.read_text(encoding="utf-8"))
    python_response = generate_errata_vector_report_v1()
    python_core = {
        field: python_response[field]
        for field in qualification.REPORT_CORE_FIELDS
    }
    assert fixture["report"] == python_core
    report = fixture["report"]
    assert {field: len(report[field]) for field in EXPECTED_VECTOR_COUNTS} == (
        EXPECTED_VECTOR_COUNTS
    )


def test_dual_errata_qualification_matches_golden_and_only_authorizes_external_step(
    m25_errata_report: dict[str, object],
) -> None:
    report = m25_errata_report
    validate_checked_errata_qualification_report(report)
    assert report["status"] == "DUAL_EXACT_WIRE_ERRATA_GOLDEN_PASS"
    assert report["python_report"] == report["rust_report"]
    assert report["cross_language_exact_match"] is True
    assert report["golden_exact_match"] is True
    assert report["dual_golden_start_guard"]["passed_check_count"] == 10
    assert report["dual_golden_start_guard"]["external_genesis_start_allowed"] is True
    assert report["external_genesis_start_authorization"] == {
        "authorization_is_side_effect_free": True,
        "external_genesis_start_allowed": True,
        "m3_gates_satisfied": 14,
        "m3_gates_total": 24,
        "child_state": "NOT_RUN",
        "gate24_qualified": False,
        "m3_entry_allowed": False,
        "m3_run_started": False,
        "phase3_m3_start_authorized": False,
        "checked_artifact_replay_alone_sufficient": False,
        "fresh_dual_replay_required": True,
    }
    assert report["formal_input_roots"] is None
    assert report["formal_roots_generated"] is False
    assert report["m3_execution_manifest_root"] is None
    assert report["authority_boundary"]["m3_gate_delta"] == 0
    assert report["authority_boundary"]["child_state"] == "NOT_RUN"
    assert report["rust_execution"]["binary_source_binding_claim"] is True
    assert report["rust_execution"]["fresh_target_directory"] is True
    assert report["rust_execution"]["working_tree_built"] is False
    assert report["rust_execution"]["inherited_environment_allowed"] is False
    assert report["rust_execution"]["binary_hash_and_exec_same_open_inode"] is False
    assert (
        report["rust_execution"][
            "binary_private_snapshot_digest_stable_during_exec"
        ]
        is True
    )
    assert report["rust_execution"]["persisted_binary_replay_equal"] is True
    assert (
        report["rust_execution"]["host_cargo_registry_mounted_into_build_container"]
        is False
    )
    assert report["rust_execution"]["docker_pull_policy"] == "never"
    assert report["rust_execution"]["docker_network_mode"] == "none"
    assert report["rust_execution"]["cargo_lock_registry_archives_verified"] is True
    assert report["rust_execution"]["cargo_lock_registry_archive_count"] > 0
    assert report["python_execution"]["working_tree_executed"] is False
    assert report["python_execution"]["minimal_module_closure"] is True
    assert report["python_execution"]["package_init_executed"] is False
    assert report["repository_secret_absence_receipt"]["pass"] is True


def test_checked_in_errata_qualification_is_current() -> None:
    if not CHECKED_IN.is_file():
        pytest.skip("qualification artifact is emitted only after Commit A")
    try:
        qualification._assert_sources_match_commit(
            qualification.repository_head_commit()
        )
    except M25ErrataQualificationError as exc:
        pytest.skip(f"new qualification sources are awaiting Commit A: {exc}")
    report = json.loads(CHECKED_IN.read_text(encoding="utf-8"))
    validate_checked_errata_qualification_report(report)


def test_errata_qualification_rejects_authority_escalation(
    m25_errata_report: dict[str, object],
) -> None:
    report = m25_errata_report
    mutated = deepcopy(report)
    mutated["authority_boundary"]["formal_roots_generated"] = True
    with pytest.raises(M25ErrataQualificationError):
        qualification._validate_report_envelope(mutated)

    mutated = deepcopy(report)
    mutated["external_genesis_start_authorization"]["m3_entry_allowed"] = True
    with pytest.raises(M25ErrataQualificationError):
        qualification._validate_report_envelope(mutated)

    mutated = deepcopy(report)
    mutated["rust_execution"]["binary_source_binding_claim"] = False
    with pytest.raises(M25ErrataQualificationError):
        qualification._validate_report_envelope(mutated)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("qualified_vector_counts", "objects"), 21.0),
        (("authority_boundary", "m3_gates_before"), 14.0),
        (("authority_boundary", "external_genesis_start_authorized"), 1),
    ],
)
def test_errata_qualification_rejects_json_numeric_type_confusion(
    m25_errata_report: dict[str, object],
    path: tuple[str, str],
    value: object,
) -> None:
    report = m25_errata_report
    mutated = deepcopy(report)
    parent = mutated[path[0]]
    assert isinstance(parent, dict)
    parent[path[1]] = value
    mutated.pop("diagnostic_report_id")
    mutated["diagnostic_report_id"] = stable_hash(
        mutated, prefix="phase3_m25_errata_qualification_"
    )
    with pytest.raises(M25ErrataQualificationError):
        qualification._validate_report_envelope(mutated)


def test_qualification_rejects_dirty_or_uncommitted_commit_a_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uncommitted = tmp_path / "not-in-commit-a.json"
    uncommitted.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(qualification, "SOURCE_PATHS", (str(uncommitted),))
    with pytest.raises(M25ErrataQualificationError):
        qualification._assert_sources_match_commit(
            qualification.repository_head_commit()
        )


def test_qualification_output_cannot_overwrite_a_bound_input() -> None:
    with pytest.raises(M25ErrataQualificationError):
        qualification.validate_errata_qualification_output_path(
            qualification.GOLDEN_VECTOR_PATH
        )


def test_public_qualification_api_rejects_caller_supplied_toolchain() -> None:
    with pytest.raises(TypeError):
        dual_errata_qualification_report(cargo_executable="/tmp/fake-cargo")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        validate_dual_errata_qualification_report(  # type: ignore[call-arg]
            {},
            cargo_executable="/tmp/fake-cargo",
        )


def _synthetic_crate_archive(path: Path, *, payload: bytes) -> str:
    source = path.parent / "Cargo.toml"
    source.write_bytes(payload)
    with tarfile.open(path, mode="w:gz") as archive:
        archive.add(source, arcname="demo-1.2.3/Cargo.toml")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checksum_exact_vendor_snapshot_uses_only_locked_archives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "cache" / "synthetic-index"
    cache.mkdir(parents=True)
    archive = cache / "demo-1.2.3.crate"
    checksum = _synthetic_crate_archive(
        archive,
        payload=b'[package]\nname = "demo"\nversion = "1.2.3"\n',
    )
    lock = tmp_path / "Cargo.lock"
    lock.write_text(
        "version = 3\n\n"
        "[[package]]\n"
        'name = "demo"\n'
        'version = "1.2.3"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{checksum}"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(qualification, "CARGO_ARCHIVE_CACHE_ROOT", tmp_path / "cache")
    destination = tmp_path / "vendor"
    root, file_count, package_count = qualification._build_cargo_dependency_snapshot(
        lock, destination
    )
    assert root.startswith("sha256:")
    assert file_count == 2
    assert package_count == 1
    assert not (destination / "registry").exists()
    assert (destination / "demo-1.2.3" / "Cargo.toml").is_file()
    assert (destination / "demo-1.2.3" / ".cargo-checksum.json").is_file()


def test_isolated_registry_rejects_archive_checksum_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "cache" / "synthetic-index"
    cache.mkdir(parents=True)
    (cache / "demo-1.2.3.crate").write_bytes(b"tampered")
    lock = tmp_path / "Cargo.lock"
    lock.write_text(
        "version = 3\n\n"
        "[[package]]\n"
        'name = "demo"\n'
        'version = "1.2.3"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{hashlib.sha256(b"expected").hexdigest()}"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(qualification, "CARGO_ARCHIVE_CACHE_ROOT", tmp_path / "cache")
    with pytest.raises(M25ErrataQualificationError):
        qualification._build_cargo_dependency_snapshot(
            lock,
            tmp_path / "vendor",
        )


def test_approved_policy_binds_offline_oci_and_vendor_snapshot() -> None:
    policy = qualification._load_approved_toolchain_policy()
    assert policy["schema_version"] == (
        "hegel-phase3-m25-approved-local-rust-oci-toolchain/2"
    )
    assert policy["image_ref"] == qualification.RUST_IMAGE_REF
    assert policy["host_cargo_cache_mounted_into_container"] is False
    assert policy["dependency_snapshot_domain"] == qualification.CARGO_SNAPSHOT_DOMAIN
    assert policy["cargo_lock_registry_package_count"] == 23
    assert policy["dependency_snapshot_file_count"] == 1088
    assert policy["required_docker_flags"] == [
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
    ]


def test_docker_command_uses_exact_offline_env_i_boundary(tmp_path: Path) -> None:
    control_plane = qualification.LocalDockerControlPlaneV1(
        executable=Path("/usr/bin/docker"),
        socket_path=Path("/var/run/docker.sock"),
        config_directory=tmp_path,
        environment={},
        binding={},
    )
    command = qualification._docker_command(
        control_plane,
        qualification.RUST_IMAGE_REF,
        ("-v", "/private/vendor:/vendor:ro"),
        (qualification.RUST_CARGO_PATH, "--version"),
        seccomp_path=qualification.BUILD_SECCOMP_PATH,
        container_environment=qualification.RUST_BUILD_ENVIRONMENT,
    )
    assert command[:3] == [
        "/usr/bin/docker",
        "--host=unix:///var/run/docker.sock",
        "run",
    ]
    assert "--pull=never" in command
    assert "--network=none" in command
    assert "--entrypoint=/usr/bin/env" in command
    assert command.count("-i") == 1
    assert not any(".cargo/registry" in value for value in command)
    assert not any("CARGO_HOME=/usr/local/cargo" in value for value in command)


def test_validated_binary_is_atomically_persisted_to_default_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crate = tmp_path / "formal_bridge_m25"
    destination = crate / "target" / "debug" / "hegel-formal-bridge-m25"
    monkeypatch.setattr(qualification, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(qualification, "RUST_CRATE_ROOT", crate)
    monkeypatch.setattr(qualification, "DEFAULT_RUST_BINARY", destination)
    payload = b"validated synthetic Rust binary"
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    receipt = qualification._persist_validated_rust_binary(payload, digest)
    assert destination.read_bytes() == payload
    assert destination.stat().st_mode & 0o777 == 0o755
    assert receipt == {
        "default_rust_binary_repository_path": destination.relative_to(
            qualification.PROJECT_ROOT
        ).as_posix(),
        "persisted_binary_sha256": digest,
        "persisted_binary_mode_octal": "0755",
        "persisted_binary_atomic_replace": True,
        "persisted_binary_is_symlink": False,
    }


def test_errata_qualification_cli_emits_non_authoritative_evidence(
    tmp_path: Path,
    m25_commit_a_ready: None,
) -> None:
    output = tmp_path / "errata-qualification.json"
    assert main(
        [
            "phase3-m25-errata-qualify",
            "--output",
            str(output),
        ]
    ) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["artifact_kind"] == "DETERMINISTIC_CANDIDATE_NON_AUTHORITATIVE"
    assert report["dual_golden_start_guard"]["external_genesis_start_allowed"] is True
    assert report["authority_boundary"]["external_genesis_started"] is False
    assert report["authority_boundary"]["m3_run_started"] is False


def test_fresh_dual_validator_replays_commit_a_sources(
    m25_errata_report: dict[str, object],
) -> None:
    validate_dual_errata_qualification_report(m25_errata_report)
