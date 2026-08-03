from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import stat
import subprocess
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hegel_machine import phase3_m25_bridge_dag_binary_qualification_v1 as qualification
from hegel_machine.phase3_m25_bridge_dag_binary_qualification_v1 import (
    AUTHORITY_BOUNDARY,
    ARTIFACT_KIND,
    BUILD_COMMAND,
    BUILD_DOCKER_POLICY_ID,
    CLAIM_LEVEL,
    DEFAULT_RUST_BRIDGE_DAG_BINARY,
    DEFAULT_RUST_BRIDGE_DAG_QUALIFICATION_REPORT,
    FAIL_NODE_SET,
    FAIL_PACKAGE_AUTHORITY,
    FAIL_ROOT_BINDING,
    PERSISTED_BINARY_REPOSITORY_PATH,
    RUNTIME_DOCKER_POLICY_ID,
    SCHEMA_VERSION,
    STATUS,
    TEST_COMMAND,
    BridgeDagBinaryQualificationError,
    load_unsigned_public_replay_fixture_v1,
    validate_rust_bridge_dag_binary_qualification_report_v1,
)


def _sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sample_report(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    policy = qualification._load_approved_toolchain_policy()
    monkeypatch.setattr(
        qualification,
        "local_docker_daemon_receipt_binding_v1",
        lambda _receipt: b"d" * 32,
    )
    source_bindings = {
        path: "sha256:" + "11" * 32
        for path in qualification.QUALIFICATION_SOURCE_PATHS
    }
    source_bindings[
        f"{qualification.CRATE_REPOSITORY_PATH}/Cargo.lock"
    ] = "sha256:" + "88" * 32
    toolchain_receipt = {
        "image_ref": policy["image_ref"],
        "image_id": policy["image_id"],
        "oci_manifest_digest": policy["oci_manifest_digest"],
        "operating_system": policy["operating_system"],
        "architecture": policy["architecture"],
        "cargo_binary_path": policy["cargo_binary_path"],
        "cargo_binary_sha256": policy["cargo_binary_sha256"],
        "cargo_version": policy["cargo_version"],
        "cargo_version_stdout_sha256": policy["cargo_version_stdout_sha256"],
        "rustc_binary_path": policy["rustc_binary_path"],
        "rustc_binary_sha256": policy["rustc_binary_sha256"],
        "rustc_version": policy["rustc_version"],
        "rustc_verbose_version_stdout_sha256": policy[
            "rustc_verbose_version_stdout_sha256"
        ],
        "runtime_environment_sha256": policy["runtime_environment_sha256"],
        "build_environment_sha256": policy["build_environment_sha256"],
        "runtime_seccomp_sha256": policy["runtime_seccomp_sha256"],
        "build_seccomp_sha256": policy["build_seccomp_sha256"],
        "image_config_environment_ignored": True,
        "pull_policy": "never",
        "network_mode": "none",
        "toolchain_receipt_is_external_attestation": False,
    }
    rows = []
    test_ids = [
        "FRESH_PUBLIC_PURPOSE1_REPLAY_PASS",
        "PUBLIC_PREIMAGE_SUBSTITUTION_REJECTED",
        "PUBLIC_NODE_OMISSION_REJECTED",
        "AUTHORITATIVE_FLAG_WITHOUT_RUNTIME_OPT_IN_REJECTED",
        "PERSISTED_PUBLIC_PURPOSE1_REPLAY_PASS",
    ]
    errors = [None, FAIL_ROOT_BINDING, FAIL_NODE_SET, FAIL_PACKAGE_AUTHORITY, None]
    for index, (test_id, error) in enumerate(zip(test_ids, errors, strict=True)):
        returncode = 0 if error is None else 1
        rows.append(
            {
                "test_id": test_id,
                "expected_returncode": returncode,
                "observed_returncode": returncode,
                "expected_error_code_or_null": error,
                "stdout_sha256": "sha256:" + ("22" if index in (0, 4) else "33") * 32,
                "stderr_sha256": "sha256:" + "44" * 32,
            }
        )
    binary_digest = "sha256:" + "55" * 32
    commit = "66" * 20
    report: dict[str, object] = {
        "artifact": "phase3_m25_bridge_dag_rust_binary_qualification_v1",
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": STATUS,
        "claim_level": CLAIM_LEVEL,
        "implementation_basis_commit": commit,
        "source": {
            "archive_domain": qualification.SOURCE_ARCHIVE_DOMAIN,
            "basis_commit": commit,
            "git_archive_exact": True,
            "worktree_bytes_equal_commit": True,
            "snapshot_read_only": True,
            "snapshot_manifest_sha256": "sha256:" + "77" * 32,
            "bindings": source_bindings,
        },
        "dependency": {
            "cargo_lock_repository_path": f"{qualification.CRATE_REPOSITORY_PATH}/Cargo.lock",
            "cargo_lock_sha256": "sha256:" + "88" * 32,
            "snapshot_domain": qualification.CARGO_SNAPSHOT_DOMAIN,
            "snapshot_root": policy["dependency_snapshot_root"],
            "snapshot_file_count": policy["dependency_snapshot_file_count"],
            "registry_package_count": policy["cargo_lock_registry_package_count"],
            "vendor_manifest_sha256": "sha256:" + "99" * 32,
            "locked_archive_checksums_verified": True,
            "host_cargo_cache_mounted_into_container": False,
        },
        "toolchain": {
            "approved_policy_repository_path": qualification.APPROVED_TOOLCHAIN_POLICY_REPOSITORY_PATH,
            "approved_policy_sha256": qualification._sha256_file(
                qualification.REPOSITORY_ROOT
                / qualification.APPROVED_TOOLCHAIN_POLICY_REPOSITORY_PATH
            ),
            "receipt": toolchain_receipt,
        },
        "container": {
            "docker_executable": "/usr/bin/docker",
            "docker_host": "unix:///var/run/docker.sock",
            "control_plane_binding": {"test": "binding"},
            "daemon_identity_receipt": {
                "control_plane_binding": {"test": "binding"}
            },
            "daemon_receipt_binding": (b"d" * 32).hex(),
            "image_ref": qualification.RUST_IMAGE_REF,
            "pull_policy": "never",
            "network_mode": "none",
            "read_only_root": True,
            "inherited_environment_allowed": False,
            "runtime_docker_policy_id": RUNTIME_DOCKER_POLICY_ID,
            "build_docker_policy_id": BUILD_DOCKER_POLICY_ID,
            "runtime_seccomp_sha256": policy["runtime_seccomp_sha256"],
            "build_seccomp_sha256": policy["build_seccomp_sha256"],
        },
        "build": {
            "release_profile": True,
            "cargo_locked": True,
            "cargo_offline": True,
            "fresh_linux_local_target": True,
            "source_mount_read_only": True,
            "vendor_mount_read_only": True,
            "test_command": list(TEST_COMMAND),
            "build_command": list(BUILD_COMMAND),
            "fresh_binary_sha256": binary_digest,
            "persisted_binary": {
                "repository_path": PERSISTED_BINARY_REPOSITORY_PATH,
                "sha256": binary_digest,
                "mode_octal": "0755",
                "atomic_replace": True,
                "is_symlink": False,
            },
        },
        "replay_tests": {
            "fixture_repository_path": qualification.GOLDEN_FIXTURE_REPOSITORY_PATH,
            "fixture_sha256": qualification._sha256_file(
                qualification.GOLDEN_FIXTURE_PATH
            ),
            "package_sha256": _sha(load_unsigned_public_replay_fixture_v1()),
            "contains_private_key": False,
            "contains_signature": False,
            "contains_seed": False,
            "tests": rows,
            "all_passed": True,
        },
        "authority_boundary": dict(AUTHORITY_BOUNDARY),
    }
    report["diagnostic_report_sha256"] = qualification._report_sha256(report)
    return report


def _reself(report: dict[str, object]) -> None:
    report.pop("diagnostic_report_sha256", None)
    report["diagnostic_report_sha256"] = qualification._report_sha256(report)


def test_stable_binary_path_is_inside_ignored_crate_target() -> None:
    assert DEFAULT_RUST_BRIDGE_DAG_BINARY == (
        qualification.PROJECT_ROOT
        / "rust/m25_bridge_dag_replay/target/commit_a_qualified/"
        "hegel-m25-bridge-dag-replay"
    )
    assert DEFAULT_RUST_BRIDGE_DAG_BINARY.relative_to(qualification.CRATE_ROOT)
    assert PERSISTED_BINARY_REPOSITORY_PATH.startswith("rust/m25_bridge_dag_replay/target/")


def test_technical_actor_disclosure_is_complete_and_truthful() -> None:
    assert {
        key: AUTHORITY_BOUNDARY[key]
        for key in (
            "same_admin_controller",
            "organizational_independence",
            "independent_human_actors",
            "technical_role_independence",
            "owner_accepted_threat_model",
            "remote_attestation",
            "hardware_key_nonexportability",
        )
    } == {
        "same_admin_controller": True,
        "organizational_independence": False,
        "independent_human_actors": False,
        "technical_role_independence": True,
        "owner_accepted_threat_model": True,
        "remote_attestation": False,
        "hardware_key_nonexportability": False,
    }
    assert DEFAULT_RUST_BRIDGE_DAG_QUALIFICATION_REPORT == (
        qualification.PROJECT_ROOT
        / "artifacts/phase3_m25_external/"
        "phase3_m25_bridge_dag_rust_binary_qualification_v1.json"
    )


def test_public_unsigned_fixture_has_no_secret_or_signature_and_mutations_are_exact() -> None:
    package = load_unsigned_public_replay_fixture_v1()
    assert _sha(package) == "sha256:9ee405e0eaa08bf4fb55b7f9b5f1a782d255ec00bd58c672e47b219ee8849f6c"
    attacks = qualification._mutated_packages_v1(package)
    assert [row[2] for row in attacks] == [
        FAIL_ROOT_BINDING,
        FAIL_NODE_SET,
        FAIL_PACKAGE_AUTHORITY,
    ]
    for _, payload, code in attacks:
        with pytest.raises(Exception) as caught:
            qualification.replay_bridge_dag_package_v1(payload)
        assert getattr(caught.value, "code") == code


def test_normalized_build_policy_is_locked_offline_and_never_mounts_cargo_home() -> None:
    joined = " ".join(BUILD_COMMAND)
    assert "--release" in BUILD_COMMAND
    assert "--locked" in BUILD_COMMAND
    assert "--offline" in BUILD_COMMAND
    assert "/vendor" in joined
    assert ".cargo" not in joined
    assert "--locked" in TEST_COMMAND and "--offline" in TEST_COMMAND
    source = Path(qualification.__file__).read_text(encoding="utf-8")
    assert '"--pull=never"' not in source  # supplied centrally by the hardened helper
    assert "_docker_command(" in source
    assert "host_cargo_cache_mounted_into_container\": False" in source


def test_commit_a_git_reads_reject_ambient_config_transport_and_lazy_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile_environment = {
        "GIT_CONFIG_GLOBAL": "/tmp/hostile-global-config",
        "GIT_CONFIG_SYSTEM": "/tmp/hostile-system-config",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "protocol.ext.allow",
        "GIT_CONFIG_VALUE_0": "always",
        "GIT_NO_LAZY_FETCH": "0",
        "GIT_PROTOCOL_FROM_USER": "1",
        "GIT_SSH_COMMAND": "/tmp/hostile-ssh-command",
        "GIT_TERMINAL_PROMPT": "1",
    }
    for key, value in hostile_environment.items():
        monkeypatch.setenv(key, value)

    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = list(command)
        observed["environment"] = dict(kwargs["env"])
        return subprocess.CompletedProcess(command, 0, b"local-object\n", b"")

    monkeypatch.setattr(qualification.subprocess, "run", fake_run)
    assert qualification._run_git(("show", "00" * 20 + ":bound/path")) == b"local-object\n"

    assert observed["command"] == [
        "/usr/bin/git",
        "show",
        "00" * 20 + ":bound/path",
    ]
    assert observed["environment"] == {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PROTOCOL_FROM_USER": "0",
        "GIT_SSH_COMMAND": "false",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }
    assert not (set(hostile_environment) - {"GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM"}) & (
        set(observed["environment"])
        - {
            "GIT_NO_LAZY_FETCH",
            "GIT_PROTOCOL_FROM_USER",
            "GIT_SSH_COMMAND",
            "GIT_TERMINAL_PROMPT",
        }
    )


def test_persist_is_atomic_regular_0755_and_validator_returns_bound_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    crate = tmp_path / "crate"
    destination = crate / "target/commit_a_qualified/hegel-m25-bridge-dag-replay"
    monkeypatch.setattr(qualification, "CRATE_ROOT", crate)
    monkeypatch.setattr(qualification, "DEFAULT_RUST_BRIDGE_DAG_BINARY", destination)
    payload = b"ELF synthetic test payload"
    digest = _sha(payload)
    binding = qualification._persist_validated_binary_v1(payload, digest)
    assert binding["sha256"] == digest
    assert destination.read_bytes() == payload
    assert stat.S_IMODE(destination.stat().st_mode) == 0o755

    report = _sample_report(monkeypatch)
    report["build"]["fresh_binary_sha256"] = digest
    report["build"]["persisted_binary"]["sha256"] = digest
    _reself(report)
    assert validate_rust_bridge_dag_binary_qualification_report_v1(
        report,
        expected_basis_commit="66" * 20,
        verify_commit_sources=False,
    ) == digest


def test_report_validator_rejects_authority_escalation_and_commit_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _sample_report(monkeypatch)
    assert validate_rust_bridge_dag_binary_qualification_report_v1(
        report,
        expected_basis_commit="66" * 20,
        verify_commit_sources=False,
        verify_persisted_binary=False,
    ) == "sha256:" + "55" * 32
    attacked = copy.deepcopy(report)
    attacked["authority_boundary"]["formal_roots_generated"] = True
    _reself(attacked)
    with pytest.raises(BridgeDagBinaryQualificationError):
        validate_rust_bridge_dag_binary_qualification_report_v1(
            attacked,
            verify_commit_sources=False,
            verify_persisted_binary=False,
        )
    with pytest.raises(BridgeDagBinaryQualificationError) as caught:
        validate_rust_bridge_dag_binary_qualification_report_v1(
            report,
            expected_basis_commit="77" * 20,
            verify_commit_sources=False,
            verify_persisted_binary=False,
        )
    assert caught.value.code == qualification.FAIL_COMMIT


def test_report_validator_rejects_binary_digest_and_replay_test_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _sample_report(monkeypatch)
    for attack in ("binary", "replay"):
        mutated = copy.deepcopy(report)
        if attack == "binary":
            mutated["build"]["persisted_binary"]["sha256"] = "sha256:" + "aa" * 32
        else:
            mutated["replay_tests"]["tests"][1]["expected_error_code_or_null"] = None
        _reself(mutated)
        with pytest.raises(BridgeDagBinaryQualificationError):
            validate_rust_bridge_dag_binary_qualification_report_v1(
                mutated,
                verify_commit_sources=False,
                verify_persisted_binary=False,
            )
