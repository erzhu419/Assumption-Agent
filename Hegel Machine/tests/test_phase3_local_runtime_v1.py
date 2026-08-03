from __future__ import annotations

import os
from pathlib import Path
import stat
import tempfile

import pytest

import hegel_machine.phase3_local_runtime_v1 as local_runtime
from hegel_machine.phase3_local_runtime_v1 import (
    FAIL_DURABLE_CUSTODY_PATH,
    FAIL_LOCAL_DOCKER_CONTROL_PLANE,
    FAIL_LOCAL_RUNTIME_FILESYSTEM,
    FAIL_LOCAL_RUNTIME_PATH,
    FAIL_LOCAL_RUNTIME_PERMISSIONS,
    LinuxLocalTemporaryDirectoryV1,
    Phase3LocalRuntimeError,
    local_docker_daemon_receipt_binding_v1,
    validate_linux_local_durable_custody_v1,
    validate_linux_local_runtime_parent_v1,
)


PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent


def test_live_tmp_boundary_and_private_directory_are_linux_local_0700() -> None:
    evidence = validate_linux_local_runtime_parent_v1(
        Path("/tmp"),
        repository_root=REPOSITORY,
    )
    assert evidence["resolved_parent"] == "/tmp"
    assert evidence["linux_local"] is True
    assert str(evidence["filesystem_type"]).casefold() not in {
        "9p",
        "cifs",
        "drvfs",
        "fuse",
        "nfs",
        "nfs4",
        "smb3",
        "v9fs",
    }

    with LinuxLocalTemporaryDirectoryV1(
        prefix="hegel-local-runtime-test-",
        repository_root=REPOSITORY,
    ) as value:
        private_root = Path(value)
        assert private_root.parent == Path("/tmp")
        assert private_root.stat().st_uid == os.geteuid()
        assert stat.S_IMODE(private_root.stat().st_mode) == 0o700
    assert not private_root.exists()


def test_mountinfo_parser_selects_longest_mount_and_decodes_paths() -> None:
    rows = local_runtime._parse_mountinfo_v1(
        "1 0 8:1 / / rw - ext4 /dev/root rw\n"
        "2 1 8:2 / /tmp rw - xfs /dev/local rw\n"
        "3 2 8:3 / /tmp/a\\040b rw - tmpfs tmpfs rw\n"
    )
    selected = local_runtime._effective_mount_v1(Path("/tmp/a b/child"), rows)
    assert selected.mount_id == 3
    assert selected.mount_point == Path("/tmp/a b")
    assert selected.filesystem_type == "tmpfs"


@pytest.mark.parametrize(
    "filesystem,mount_source,options",
    [
        ("9p", "C:\\", "aname=drvfs"),
        ("nfs4", "server:/export", "rw"),
        ("cifs", "//server/share", "rw"),
        ("fuse.rclone", "cloud:", "rw"),
        ("ext4", "drvfs", "rw"),
    ],
)
def test_remote_drvfs_and_fuse_mounts_fail_closed(
    filesystem: str,
    mount_source: str,
    options: str,
) -> None:
    row = local_runtime.MountInfoV1(
        mount_id=1,
        device="0:1",
        mount_point=Path("/tmp"),
        mount_options=("rw",),
        filesystem_type=filesystem,
        mount_source=mount_source,
        super_options=tuple(options.split(",")),
    )
    with pytest.raises(Phase3LocalRuntimeError) as captured:
        local_runtime._require_linux_local_mount_v1(row)
    assert captured.value.code == FAIL_LOCAL_RUNTIME_FILESYSTEM


@pytest.mark.parametrize(
    "candidate",
    [
        Path("/mnt/c/Users/example/AppData/Local/Temp"),
        Path.home(),
        REPOSITORY,
        Path("/tmp/OneDrive/ceremony"),
    ],
)
def test_repo_home_mnt_c_and_cloud_sync_locations_fail_closed(candidate: Path) -> None:
    with pytest.raises(Phase3LocalRuntimeError) as captured:
        local_runtime._require_location_policy_v1(
            candidate,
            repository_root=REPOSITORY,
            home_directory=Path.home(),
        )
    assert captured.value.code == FAIL_LOCAL_RUNTIME_PATH


def test_custom_parent_requires_caller_owned_mode_0700() -> None:
    with tempfile.TemporaryDirectory(prefix="hegel-local-parent-", dir="/tmp") as value:
        parent = Path(value)
        parent.chmod(0o755)
        with pytest.raises(Phase3LocalRuntimeError) as captured:
            validate_linux_local_runtime_parent_v1(
                parent,
                repository_root=REPOSITORY,
            )
        assert captured.value.code == FAIL_LOCAL_RUNTIME_PERMISSIONS


def test_symlinked_runtime_parent_fails_closed() -> None:
    with tempfile.TemporaryDirectory(prefix="hegel-local-target-", dir="/tmp") as value:
        target = Path(value)
        link = Path("/tmp") / f"hegel-local-link-{os.getpid()}"
        link.symlink_to(target, target_is_directory=True)
        try:
            with pytest.raises(Phase3LocalRuntimeError) as captured:
                validate_linux_local_runtime_parent_v1(
                    link,
                    repository_root=REPOSITORY,
                )
            assert captured.value.code == FAIL_LOCAL_RUNTIME_PATH
        finally:
            link.unlink()


@pytest.mark.skipif(
    not Path("/usr/bin/docker").is_file() or not Path("/var/run/docker.sock").exists(),
    reason="local Docker CLI/socket are unavailable",
)
def test_local_docker_control_plane_uses_empty_config_and_no_proxy_environment() -> None:
    # This validates only local filesystem preparation; the helper never calls
    # the daemon, so the test does not create network traffic or Docker state.
    with LinuxLocalTemporaryDirectoryV1(
        prefix="hegel-local-docker-test-",
        repository_root=REPOSITORY,
    ) as value:
        boundary = local_runtime.prepare_local_docker_control_plane_v1(
            Path(value),
            repository_root=REPOSITORY,
        )
        assert boundary.executable == Path("/usr/bin/docker")
        assert boundary.socket_path == Path("/var/run/docker.sock").resolve(strict=True)
        assert boundary.environment["DOCKER_HOST"] == "unix:///var/run/docker.sock"
        assert set(boundary.environment) == {
            "DOCKER_CONFIG",
            "DOCKER_HOST",
            "HOME",
            "LANG",
            "LC_ALL",
            "PATH",
        }
        assert (boundary.config_directory / "config.json").read_bytes() == b"{}\n"
        assert stat.S_IMODE(boundary.config_directory.stat().st_mode) == 0o700
        assert boundary.binding["proxy_environment_keys"] == []
        assert boundary.command("version") == [
            "/usr/bin/docker",
            "--host=unix:///var/run/docker.sock",
            "version",
        ]


def test_durable_custody_is_home_local_fsync_qualified_and_probe_free() -> None:
    parent = Path.home() / ".local/state/hegel-machine-unit-custody"
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent.chmod(0o700)
    with tempfile.TemporaryDirectory(prefix="custody-", dir=parent) as value:
        custody = Path(value)
        custody.chmod(0o700)
        receipt = validate_linux_local_durable_custody_v1(
            custody,
            repository_root=REPOSITORY,
        )
        assert receipt["linux_local_durable_filesystem"] is True
        assert receipt["file_fsync_probe_passed"] is True
        assert receipt["atomic_rename_probe_passed"] is True
        assert receipt["directory_fsync_probe_passed"] is True
        assert receipt["probe_artifacts_absent"] is True
        assert list(custody.iterdir()) == []


def test_durable_custody_rejects_tmp_even_when_linux_local() -> None:
    with tempfile.TemporaryDirectory(prefix="hegel-custody-reject-", dir="/tmp") as value:
        custody = Path(value)
        custody.chmod(0o700)
        with pytest.raises(Phase3LocalRuntimeError) as captured:
            validate_linux_local_durable_custody_v1(
                custody,
                repository_root=REPOSITORY,
            )
        assert captured.value.code == FAIL_DURABLE_CUSTODY_PATH


def test_daemon_receipt_binding_rejects_any_tamper(monkeypatch, tmp_path: Path) -> None:
    boundary = local_runtime.LocalDockerControlPlaneV1(
        executable=Path("/usr/bin/docker"),
        socket_path=Path("/var/run/docker.sock"),
        config_directory=tmp_path,
        environment={},
        binding={"schema": local_runtime.LOCAL_DOCKER_CONTROL_PLANE_SCHEMA},
    )
    monkeypatch.setattr(
        local_runtime,
        "validate_linux_local_host_path_v1",
        lambda *_args, **_kwargs: {"mount_id": 7, "filesystem_type": "ext4"},
    )
    receipt = local_runtime.build_local_docker_daemon_identity_receipt_v1(
        boundary,
        version_payload={
            "Client": {"Version": "1", "ApiVersion": "1"},
            "Server": {
                "Version": "1", "ApiVersion": "1", "Os": "linux", "Arch": "amd64"
            },
        },
        info_payload={
            "ID": "daemon-id",
            "Name": "daemon-name",
            "OSType": "linux",
            "Architecture": "x86_64",
            "DockerRootDir": "/var/lib/docker",
            "Driver": "overlayfs",
            "HttpProxy": "",
            "HttpsProxy": "",
        },
        repository_root=REPOSITORY,
    )
    assert len(local_docker_daemon_receipt_binding_v1(receipt)) == 32
    tampered = dict(receipt)
    tampered["storage_driver"] = "other"
    with pytest.raises(Phase3LocalRuntimeError) as captured:
        local_docker_daemon_receipt_binding_v1(tampered)
    assert captured.value.code == FAIL_LOCAL_DOCKER_CONTROL_PLANE
