"""Fail-closed local runtime boundaries for Phase-3 Docker work.

The Docker daemon can only preserve the intended isolation when every host
bind source is first materialized on a Linux-local filesystem.  In WSL,
Python's default temporary directory may resolve to DrvFS under ``/mnt/c``;
callers must therefore opt in to the validated ``/tmp`` boundary here.

This module does not start Docker and does not contact a daemon or registry.
It also contains the frozen shape of the future local Docker control-plane
binding so callers can migrate to one absolute CLI, one Unix socket, one
empty client config, and a sanitized environment without inventing separate
policies.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import tempfile
from types import MappingProxyType
from typing import Final, Mapping, NoReturn


LOCAL_RUNTIME_SCHEMA: Final = "hegel-phase3-linux-local-runtime/1"
LOCAL_DOCKER_CONTROL_PLANE_SCHEMA: Final = (
    "hegel-phase3-local-docker-control-plane/1"
)
LOCAL_DOCKER_DAEMON_IDENTITY_SCHEMA: Final = (
    "hegel-phase3-local-docker-daemon-identity/1"
)
DEFAULT_LINUX_LOCAL_RUNTIME_PARENT: Final = Path("/tmp")
DEFAULT_DOCKER_EXECUTABLE: Final = Path("/usr/bin/docker")
DEFAULT_DOCKER_SOCKET: Final = Path("/var/run/docker.sock")
LOCAL_DOCKER_HOST: Final = "unix:///var/run/docker.sock"

FAIL_LOCAL_RUNTIME_PATH: Final = "FAIL_PHASE3_LOCAL_RUNTIME_PATH"
FAIL_LOCAL_RUNTIME_FILESYSTEM: Final = "FAIL_PHASE3_LOCAL_RUNTIME_FILESYSTEM"
FAIL_LOCAL_RUNTIME_PERMISSIONS: Final = "FAIL_PHASE3_LOCAL_RUNTIME_PERMISSIONS"
FAIL_LOCAL_RUNTIME_MOUNTINFO: Final = "FAIL_PHASE3_LOCAL_RUNTIME_MOUNTINFO"
FAIL_LOCAL_DOCKER_CONTROL_PLANE: Final = "FAIL_PHASE3_LOCAL_DOCKER_CONTROL_PLANE"
FAIL_DURABLE_CUSTODY_PATH: Final = "FAIL_PHASE3_DURABLE_CUSTODY_PATH"

_MOUNTINFO_ESCAPE = re.compile(r"\\([0-7]{3})")
_FORBIDDEN_FILESYSTEMS: Final = frozenset(
    {
        "9p",
        "afs",
        "ceph",
        "cifs",
        "coda",
        "davfs",
        "davfs2",
        "drvfs",
        "fuse",
        "fuseblk",
        "gfs",
        "gfs2",
        "glusterfs",
        "ncpfs",
        "nfs",
        "nfs4",
        "ocfs2",
        "smb3",
        "sshfs",
        "v9fs",
    }
)
_CLOUD_COMPONENTS: Final = frozenset(
    {
        "box",
        "box sync",
        "dropbox",
        "google drive",
        "googledrive",
        "icloud",
        "icloud drive",
        "nextcloud",
        "onedrive",
        "syncthing",
    }
)
_DURABLE_LOCAL_FILESYSTEMS: Final = frozenset(
    {"btrfs", "ext2", "ext3", "ext4", "f2fs", "xfs", "zfs"}
)


class Phase3LocalRuntimeError(RuntimeError):
    """Stable fail-closed error for a host runtime boundary."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Phase3LocalRuntimeError(code, detail)


@dataclass(frozen=True, slots=True)
class MountInfoV1:
    mount_id: int
    device: str
    mount_point: Path
    mount_options: tuple[str, ...]
    filesystem_type: str
    mount_source: str
    super_options: tuple[str, ...]


def _decode_mountinfo_field_v1(value: str) -> str:
    return _MOUNTINFO_ESCAPE.sub(lambda match: chr(int(match.group(1), 8)), value)


def _parse_mountinfo_v1(payload: str) -> tuple[MountInfoV1, ...]:
    rows: list[MountInfoV1] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        left, separator, right = line.partition(" - ")
        left_fields = left.split()
        right_fields = right.split()
        if not separator or len(left_fields) < 6 or len(right_fields) < 3:
            _fail(
                FAIL_LOCAL_RUNTIME_MOUNTINFO,
                f"malformed /proc/self/mountinfo row {line_number}",
            )
        try:
            mount_id = int(left_fields[0], 10)
        except ValueError:
            _fail(
                FAIL_LOCAL_RUNTIME_MOUNTINFO,
                f"non-numeric mount ID at row {line_number}",
            )
        mount_point_text = _decode_mountinfo_field_v1(left_fields[4])
        if not mount_point_text.startswith("/"):
            _fail(
                FAIL_LOCAL_RUNTIME_MOUNTINFO,
                f"non-absolute mount point at row {line_number}",
            )
        rows.append(
            MountInfoV1(
                mount_id=mount_id,
                device=left_fields[2],
                mount_point=Path(mount_point_text),
                mount_options=tuple(left_fields[5].split(",")),
                filesystem_type=right_fields[0],
                mount_source=_decode_mountinfo_field_v1(right_fields[1]),
                super_options=tuple(right_fields[2].split(",")),
            )
        )
    if not rows:
        _fail(FAIL_LOCAL_RUNTIME_MOUNTINFO, "/proc/self/mountinfo is empty")
    return tuple(rows)


def _is_within_v1(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _effective_mount_v1(path: Path, rows: tuple[MountInfoV1, ...]) -> MountInfoV1:
    matches = [row for row in rows if _is_within_v1(path, row.mount_point)]
    if not matches:
        _fail(
            FAIL_LOCAL_RUNTIME_MOUNTINFO,
            f"no effective mount row covers {path}",
        )
    return max(matches, key=lambda row: len(row.mount_point.parts))


def _require_linux_local_mount_v1(row: MountInfoV1) -> None:
    filesystem = row.filesystem_type.casefold()
    source = row.mount_source.casefold()
    options = {value.casefold() for value in (*row.mount_options, *row.super_options)}
    if (
        filesystem in _FORBIDDEN_FILESYSTEMS
        or filesystem.startswith("fuse.")
        or "drvfs" in source
        or any(value == "drvfs" or value.startswith("aname=drvfs") for value in options)
    ):
        _fail(
            FAIL_LOCAL_RUNTIME_FILESYSTEM,
            f"runtime mount is not Linux-local: {row.filesystem_type}",
        )


def _normalized_absolute_v1(path: Path) -> Path:
    if not path.is_absolute():
        _fail(FAIL_LOCAL_RUNTIME_PATH, "runtime parent must be absolute")
    return Path(os.path.abspath(os.fspath(path)))


def _require_location_policy_v1(
    path: Path,
    *,
    repository_root: Path,
    home_directory: Path,
    require_under_tmp: bool = True,
) -> None:
    runtime_root = DEFAULT_LINUX_LOCAL_RUNTIME_PARENT.resolve(strict=True)
    if require_under_tmp and not _is_within_v1(path, runtime_root):
        _fail(FAIL_LOCAL_RUNTIME_PATH, "runtime parent must resolve under /tmp")
    rejected_roots = (
        repository_root.resolve(strict=True),
        home_directory.resolve(strict=True),
        Path("/mnt/c"),
    )
    if any(_is_within_v1(path, root) for root in rejected_roots):
        _fail(
            FAIL_LOCAL_RUNTIME_PATH,
            "runtime parent overlaps repository, HOME, or /mnt/c",
        )
    normalized_parts = tuple(
        " ".join(part.casefold().replace("_", " ").replace("-", " ").split())
        for part in path.parts
    )
    if any(
        component in _CLOUD_COMPONENTS
        or any(
            component.startswith(marker + " ")
            for marker in _CLOUD_COMPONENTS
            if marker != "box"
        )
        for component in normalized_parts
    ):
        _fail(FAIL_LOCAL_RUNTIME_PATH, "runtime parent is inside a cloud-sync path")


def _is_cloud_sync_path_v1(path: Path) -> bool:
    normalized_parts = tuple(
        " ".join(part.casefold().replace("_", " ").replace("-", " ").split())
        for part in path.parts
    )
    return any(
        component in _CLOUD_COMPONENTS
        or any(
            component.startswith(marker + " ")
            for marker in _CLOUD_COMPONENTS
            if marker != "box"
        )
        for component in normalized_parts
    )


def validate_linux_local_durable_custody_location_v1(
    custody: Path,
    *,
    repository_root: Path,
    allowed_owner_uids: frozenset[int],
) -> dict[str, object]:
    """Validate durable custody location/metadata without writing a probe.

    This narrower check exists for crash recovery when the exact directory is
    still owned by the frozen non-root actor UID. It is sufficient only to
    authorize metadata-only ownership reclamation; callers must run the full
    durability validator immediately afterwards.
    """

    requested = _normalized_absolute_v1(custody)
    try:
        metadata = requested.lstat()
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        _fail(FAIL_DURABLE_CUSTODY_PATH, f"custody cannot be resolved: {exc}")
    if (
        stat.S_ISLNK(metadata.st_mode)
        or requested != resolved
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        _fail(FAIL_DURABLE_CUSTODY_PATH, "custody must be a real directory without symlinks")
    if metadata.st_uid not in allowed_owner_uids or stat.S_IMODE(metadata.st_mode) != 0o700:
        _fail(FAIL_DURABLE_CUSTODY_PATH, "custody has an unauthorized owner or is not mode 0700")
    repository = repository_root.resolve(strict=True)
    if (
        _is_within_v1(resolved, repository)
        or _is_within_v1(resolved, DEFAULT_LINUX_LOCAL_RUNTIME_PARENT)
        or _is_within_v1(resolved, Path("/mnt/c"))
        or _is_cloud_sync_path_v1(resolved)
    ):
        _fail(
            FAIL_DURABLE_CUSTODY_PATH,
            "custody must be repo-external, non-/tmp, non-/mnt/c, and non-cloud-sync",
        )
    try:
        payload = Path("/proc/self/mountinfo").read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as exc:
        _fail(FAIL_DURABLE_CUSTODY_PATH, f"cannot read custody mountinfo: {exc}")
    mount = _effective_mount_v1(resolved, _parse_mountinfo_v1(payload))
    _require_linux_local_mount_v1(mount)
    if mount.filesystem_type.casefold() not in _DURABLE_LOCAL_FILESYSTEMS:
        _fail(
            FAIL_DURABLE_CUSTODY_PATH,
            f"custody filesystem is not in the durable local allowlist: {mount.filesystem_type}",
        )
    return {
        "schema": "hegel-phase3-durable-custody-location/1",
        "resolved_path": resolved.as_posix(),
        "resolved_path_sha256": hashlib.sha256(
            resolved.as_posix().encode("utf-8")
        ).hexdigest(),
        "owner_uid": metadata.st_uid,
        "mode_octal": "0700",
        "mount_id": mount.mount_id,
        "mount_device": mount.device,
        "filesystem_type": mount.filesystem_type,
        "linux_local_durable_filesystem": True,
    }
def validate_linux_local_durable_custody_v1(
    custody: Path,
    *,
    repository_root: Path,
) -> dict[str, object]:
    """Validate and durability-probe the persistent raw-seed custody path."""

    location = validate_linux_local_durable_custody_location_v1(
        custody,
        repository_root=repository_root,
        allowed_owner_uids=frozenset({os.geteuid()}),
    )
    resolved = Path(str(location["resolved_path"]))

    token = secrets.token_hex(16)
    source = resolved / f".hegel-durability-probe-{token}.pending"
    target = resolved / f".hegel-durability-probe-{token}.complete"
    descriptor: int | None = None
    directory_descriptor: int | None = None
    try:
        descriptor = os.open(
            source,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        payload_bytes = b"HEGEL_DURABLE_CUSTODY_PROBE_V1\n"
        written = os.write(descriptor, payload_bytes)
        if written != len(payload_bytes):
            raise OSError("short durability probe write")
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(source, target)
        directory_descriptor = os.open(
            resolved,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        os.fsync(directory_descriptor)
        target.unlink()
        os.fsync(directory_descriptor)
    except OSError as exc:
        _fail(FAIL_DURABLE_CUSTODY_PATH, f"custody fsync/rename probe failed: {exc}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        for probe_path in (source, target):
            try:
                probe_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                # The primary validation failure remains authoritative; a
                # surviving uniquely named probe is itself fail-visible.
                pass
    artifacts_absent = not source.exists() and not target.exists()
    if not artifacts_absent:
        _fail(FAIL_DURABLE_CUSTODY_PATH, "custody durability probe artifact remains")
    return {
        "schema": "hegel-phase3-durable-custody/1",
        "resolved_path_sha256": hashlib.sha256(
            resolved.as_posix().encode("utf-8")
        ).hexdigest(),
        "owner_uid": location["owner_uid"],
        "mode_octal": "0700",
        "mount_id": location["mount_id"],
        "mount_device": location["mount_device"],
        "filesystem_type": location["filesystem_type"],
        "linux_local_durable_filesystem": True,
        "file_fsync_probe_passed": True,
        "atomic_rename_probe_passed": True,
        "directory_fsync_probe_passed": True,
        "probe_artifacts_absent": True,
    }


def validate_linux_local_host_path_v1(
    path: Path,
    *,
    repository_root: Path,
    home_directory: Path | None = None,
    require_under_tmp: bool = False,
) -> dict[str, object]:
    """Validate a real directory and its effective local Linux mount."""

    requested = _normalized_absolute_v1(path)
    try:
        requested_metadata = requested.lstat()
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        _fail(FAIL_LOCAL_RUNTIME_PATH, f"host path cannot be resolved: {exc}")
    if stat.S_ISLNK(requested_metadata.st_mode) or requested != resolved:
        _fail(FAIL_LOCAL_RUNTIME_PATH, "host path or one of its components is a symlink")
    if not stat.S_ISDIR(requested_metadata.st_mode):
        _fail(FAIL_LOCAL_RUNTIME_PATH, "host path is not a directory")
    try:
        home = Path.home() if home_directory is None else home_directory
        _require_location_policy_v1(
            resolved,
            repository_root=repository_root,
            home_directory=home,
            require_under_tmp=require_under_tmp,
        )
    except OSError as exc:
        _fail(FAIL_LOCAL_RUNTIME_PATH, f"rejected-root resolution failed: {exc}")
    try:
        mountinfo_payload = Path("/proc/self/mountinfo").read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as exc:
        _fail(FAIL_LOCAL_RUNTIME_MOUNTINFO, f"cannot read mountinfo: {exc}")
    mount = _effective_mount_v1(resolved, _parse_mountinfo_v1(mountinfo_payload))
    _require_linux_local_mount_v1(mount)
    mode = stat.S_IMODE(requested_metadata.st_mode)
    return {
        "schema": LOCAL_RUNTIME_SCHEMA,
        "requested_path": requested.as_posix(),
        "resolved_path": resolved.as_posix(),
        "path_device": requested_metadata.st_dev,
        "path_owner_uid": requested_metadata.st_uid,
        "path_mode_octal": f"{mode:04o}",
        "mount_id": mount.mount_id,
        "mount_device": mount.device,
        "mount_point": mount.mount_point.as_posix(),
        "filesystem_type": mount.filesystem_type,
        "mount_source": mount.mount_source,
        "linux_local": True,
    }


def validate_linux_local_runtime_parent_v1(
    parent: Path = DEFAULT_LINUX_LOCAL_RUNTIME_PARENT,
    *,
    repository_root: Path,
    home_directory: Path | None = None,
    allowed_owner_uids: frozenset[int] | None = None,
) -> dict[str, object]:
    """Validate the exact host parent used for private Phase-3 files."""

    evidence = validate_linux_local_host_path_v1(
        parent,
        repository_root=repository_root,
        home_directory=home_directory,
        require_under_tmp=True,
    )
    requested = Path(str(evidence["requested_path"]))
    resolved = Path(str(evidence["resolved_path"]))
    requested_metadata = requested.lstat()
    mode = stat.S_IMODE(requested_metadata.st_mode)
    if resolved == DEFAULT_LINUX_LOCAL_RUNTIME_PARENT:
        if (mode & 0o022) and not (mode & stat.S_ISVTX):
            _fail(
                FAIL_LOCAL_RUNTIME_PERMISSIONS,
                "shared /tmp must use the sticky bit when group/other writable",
            )
    else:
        permitted_owners = (
            frozenset({os.geteuid()})
            if allowed_owner_uids is None
            else allowed_owner_uids
        )
        if requested_metadata.st_uid not in permitted_owners or mode != 0o700:
            _fail(
                FAIL_LOCAL_RUNTIME_PERMISSIONS,
                "custom runtime parent under /tmp has an unauthorized owner or is not mode 0700",
            )
    return {
        **evidence,
        "requested_parent": requested.as_posix(),
        "resolved_parent": resolved.as_posix(),
        "parent_device": requested_metadata.st_dev,
        "parent_owner_uid": requested_metadata.st_uid,
        "parent_mode_octal": f"{mode:04o}",
    }


class LinuxLocalTemporaryDirectoryV1(AbstractContextManager[str]):
    """A 0700 temporary directory proven to remain on the validated mount."""

    def __init__(
        self,
        *,
        prefix: str,
        repository_root: Path,
        parent: Path = DEFAULT_LINUX_LOCAL_RUNTIME_PARENT,
    ) -> None:
        if not prefix or "/" in prefix or "\\" in prefix:
            _fail(FAIL_LOCAL_RUNTIME_PATH, "temporary prefix is invalid")
        parent_evidence = validate_linux_local_runtime_parent_v1(
            parent,
            repository_root=repository_root,
        )
        owner: tempfile.TemporaryDirectory[str] | None = None
        try:
            owner = tempfile.TemporaryDirectory(
                prefix=prefix,
                dir=str(parent_evidence["resolved_parent"]),
            )
            path = Path(owner.name)
            path.chmod(0o700)
            metadata = path.lstat()
            resolved = path.resolve(strict=True)
            expected_parent = Path(str(parent_evidence["resolved_parent"]))
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISDIR(metadata.st_mode)
                or resolved != path
                or resolved.parent != expected_parent
            ):
                _fail(FAIL_LOCAL_RUNTIME_PATH, "private runtime directory escaped its parent")
            mode = stat.S_IMODE(metadata.st_mode)
            if metadata.st_uid != os.geteuid() or mode != 0o700:
                _fail(
                    FAIL_LOCAL_RUNTIME_PERMISSIONS,
                    "private runtime directory must be caller-owned mode 0700",
                )
            child_evidence = validate_linux_local_runtime_parent_v1(
                path,
                repository_root=repository_root,
            )
            if (
                child_evidence["mount_id"] != parent_evidence["mount_id"]
                or child_evidence["parent_device"] != parent_evidence["parent_device"]
            ):
                _fail(
                    FAIL_LOCAL_RUNTIME_FILESYSTEM,
                    "private runtime directory crossed a mount boundary",
                )
        except BaseException:
            if owner is not None:
                owner.cleanup()
            raise
        self._owner = owner
        self.name = owner.name
        self.evidence = {
            **parent_evidence,
            "private_directory": self.name,
            "private_directory_mode_octal": "0700",
            "private_directory_owner_uid": os.geteuid(),
        }

    def cleanup(self) -> None:
        self._owner.cleanup()

    def __enter__(self) -> str:
        return self.name

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.cleanup()
        return False


@dataclass(frozen=True, slots=True)
class LocalDockerControlPlaneV1:
    """Prepared local-only Docker client boundary; no daemon call is made."""

    executable: Path
    socket_path: Path
    config_directory: Path
    environment: Mapping[str, str]
    binding: Mapping[str, object]

    def command(self, *arguments: str) -> list[str]:
        return [
            self.executable.as_posix(),
            f"--host={LOCAL_DOCKER_HOST}",
            *arguments,
        ]


def prepare_local_docker_control_plane_v1(
    runtime_root: Path,
    *,
    repository_root: Path,
) -> LocalDockerControlPlaneV1:
    """Prepare a sanitized Docker CLI environment for later caller migration.

    The function only validates local files and writes an empty private Docker
    client config.  It deliberately does not execute Docker.
    """

    try:
        executable_lstat = DEFAULT_DOCKER_EXECUTABLE.lstat()
        executable = DEFAULT_DOCKER_EXECUTABLE.resolve(strict=True)
        socket_lstat = DEFAULT_DOCKER_SOCKET.lstat()
        socket_path = DEFAULT_DOCKER_SOCKET.resolve(strict=True)
        runtime_evidence = validate_linux_local_runtime_parent_v1(
            runtime_root,
            repository_root=repository_root,
        )
        runtime = Path(str(runtime_evidence["resolved_parent"]))
    except OSError as exc:
        _fail(FAIL_LOCAL_DOCKER_CONTROL_PLANE, f"local Docker binding is absent: {exc}")
    if (
        stat.S_ISLNK(executable_lstat.st_mode)
        or executable != DEFAULT_DOCKER_EXECUTABLE
        or not stat.S_ISREG(executable_lstat.st_mode)
        or not os.access(executable, os.X_OK)
    ):
        _fail(FAIL_LOCAL_DOCKER_CONTROL_PLANE, "/usr/bin/docker is not the exact executable")
    if (
        stat.S_ISLNK(socket_lstat.st_mode)
        or not stat.S_ISSOCK(socket_lstat.st_mode)
    ):
        _fail(FAIL_LOCAL_DOCKER_CONTROL_PLANE, "Docker endpoint is not the exact local Unix socket")
    if stat.S_IMODE(runtime.stat().st_mode) != 0o700 or runtime.stat().st_uid != os.geteuid():
        _fail(FAIL_LOCAL_DOCKER_CONTROL_PLANE, "Docker runtime root is not private")
    docker_home = runtime / "docker-home"
    docker_config = runtime / "docker-config"
    docker_home.mkdir(mode=0o700)
    docker_config.mkdir(mode=0o700)
    config_path = docker_config / "config.json"
    config_path.write_bytes(b"{}\n")
    config_path.chmod(0o600)
    executable_payload = executable.read_bytes()
    environment = {
        "DOCKER_CONFIG": docker_config.as_posix(),
        "DOCKER_HOST": LOCAL_DOCKER_HOST,
        "HOME": docker_home.as_posix(),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }
    return LocalDockerControlPlaneV1(
        executable=executable,
        socket_path=socket_path,
        config_directory=docker_config,
        environment=MappingProxyType(environment),
        binding=MappingProxyType({
            "schema": LOCAL_DOCKER_CONTROL_PLANE_SCHEMA,
            "docker_executable": executable.as_posix(),
            "docker_executable_sha256": hashlib.sha256(executable_payload).hexdigest(),
            "docker_socket_requested": DEFAULT_DOCKER_SOCKET.as_posix(),
            "docker_socket_resolved": socket_path.as_posix(),
            "docker_host": LOCAL_DOCKER_HOST,
            "docker_config_sha256": hashlib.sha256(b"{}\n").hexdigest(),
            "environment_keys": sorted(environment),
            "proxy_environment_keys": [],
            "network_endpoint_kind": "LOCAL_UNIX_SOCKET",
            "runtime_mount_id": runtime_evidence["mount_id"],
            "runtime_filesystem_type": runtime_evidence["filesystem_type"],
        }),
    )


def build_local_docker_daemon_identity_receipt_v1(
    control_plane: LocalDockerControlPlaneV1,
    *,
    version_payload: Mapping[str, object],
    info_payload: Mapping[str, object],
    repository_root: Path,
) -> dict[str, object]:
    """Validate and bind a live daemon reached through the frozen Unix socket."""

    try:
        client = version_payload["Client"]
        server = version_payload["Server"]
        if not isinstance(client, Mapping) or not isinstance(server, Mapping):
            raise TypeError
        client_version = str(client["Version"])
        client_api = str(client["ApiVersion"])
        server_version = str(server["Version"])
        server_api = str(server["ApiVersion"])
        server_os = str(server["Os"])
        server_arch = str(server["Arch"])
        daemon_id = str(info_payload["ID"])
        daemon_name = str(info_payload["Name"])
        info_os = str(info_payload["OSType"])
        info_arch = str(info_payload["Architecture"])
        docker_root = Path(str(info_payload["DockerRootDir"]))
        storage_driver = str(info_payload["Driver"])
        http_proxy = str(info_payload.get("HttpProxy", ""))
        https_proxy = str(info_payload.get("HttpsProxy", ""))
    except (KeyError, TypeError, ValueError) as exc:
        _fail(FAIL_LOCAL_DOCKER_CONTROL_PLANE, f"invalid Docker daemon identity: {exc}")
    if (
        not client_version
        or not client_api
        or not server_version
        or not server_api
        or server_os != "linux"
        or info_os != "linux"
        or server_arch not in {"amd64", "x86_64"}
        or info_arch not in {"amd64", "x86_64"}
        or not daemon_id
        or not daemon_name
        or not storage_driver
        or http_proxy
        or https_proxy
    ):
        _fail(
            FAIL_LOCAL_DOCKER_CONTROL_PLANE,
            "Docker daemon is not the exact local Linux/no-proxy profile",
        )
    root_evidence = validate_linux_local_host_path_v1(
        docker_root,
        repository_root=repository_root,
    )
    body: dict[str, object] = {
        "schema": LOCAL_DOCKER_DAEMON_IDENTITY_SCHEMA,
        "control_plane_binding": dict(control_plane.binding),
        "explicit_host_argument": f"--host={LOCAL_DOCKER_HOST}",
        "client_version": client_version,
        "client_api_version": client_api,
        "server_version": server_version,
        "server_api_version": server_api,
        "server_os": server_os,
        "server_arch": server_arch,
        "daemon_id_sha256": hashlib.sha256(daemon_id.encode("utf-8")).hexdigest(),
        "daemon_name_sha256": hashlib.sha256(daemon_name.encode("utf-8")).hexdigest(),
        "docker_root_dir_sha256": hashlib.sha256(
            docker_root.as_posix().encode("utf-8")
        ).hexdigest(),
        "docker_root_mount_id": root_evidence["mount_id"],
        "docker_root_filesystem_type": root_evidence["filesystem_type"],
        "storage_driver": storage_driver,
        "daemon_proxy_fields_empty": True,
        "registry_contact_performed": False,
        "local_linux_daemon": True,
    }
    encoded = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    body["receipt_sha256"] = hashlib.sha256(encoded).hexdigest()
    return body


def local_docker_daemon_receipt_binding_v1(
    receipt: Mapping[str, object],
) -> bytes:
    """Return the validated 32-byte diagnostic binding for a daemon receipt."""

    body = dict(receipt)
    claimed = body.pop("receipt_sha256", None)
    if (
        body.get("schema") != LOCAL_DOCKER_DAEMON_IDENTITY_SCHEMA
        or body.get("local_linux_daemon") is not True
        or type(claimed) is not str
        or re.fullmatch(r"[0-9a-f]{64}", claimed) is None
    ):
        _fail(FAIL_LOCAL_DOCKER_CONTROL_PLANE, "daemon receipt binding is malformed")
    encoded = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    actual = hashlib.sha256(encoded).hexdigest()
    if actual != claimed:
        _fail(FAIL_LOCAL_DOCKER_CONTROL_PLANE, "daemon receipt binding differs")
    return bytes.fromhex(claimed)


__all__ = [
    "DEFAULT_DOCKER_EXECUTABLE",
    "DEFAULT_DOCKER_SOCKET",
    "DEFAULT_LINUX_LOCAL_RUNTIME_PARENT",
    "FAIL_LOCAL_DOCKER_CONTROL_PLANE",
    "FAIL_DURABLE_CUSTODY_PATH",
    "FAIL_LOCAL_RUNTIME_FILESYSTEM",
    "FAIL_LOCAL_RUNTIME_MOUNTINFO",
    "FAIL_LOCAL_RUNTIME_PATH",
    "FAIL_LOCAL_RUNTIME_PERMISSIONS",
    "LOCAL_DOCKER_HOST",
    "LinuxLocalTemporaryDirectoryV1",
    "LocalDockerControlPlaneV1",
    "MountInfoV1",
    "Phase3LocalRuntimeError",
    "build_local_docker_daemon_identity_receipt_v1",
    "local_docker_daemon_receipt_binding_v1",
    "prepare_local_docker_control_plane_v1",
    "validate_linux_local_host_path_v1",
    "validate_linux_local_durable_custody_location_v1",
    "validate_linux_local_durable_custody_v1",
    "validate_linux_local_runtime_parent_v1",
]
