"""Offline Docker qualification for owner-accepted Phase-3 technical actors.

This module never pulls an image, builds an image, or enables container
networking.  It qualifies four *technical* actors against the committed
profile with measured in-container evidence.  The resulting eligibility does
not claim different administrators, people, organizations, or protection from
a malicious host/Docker daemon.

No split seed, signing key, formal root, or M3 object is created here.  A
passing report establishes actor eligibility only; the formal ceremony is a
separate operation.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import time
from types import MappingProxyType
from typing import Final, Mapping, NoReturn, Sequence

from .phase3_local_runtime_v1 import (
    LinuxLocalTemporaryDirectoryV1,
    LocalDockerControlPlaneV1,
    Phase3LocalRuntimeError,
    build_local_docker_daemon_identity_receipt_v1,
    local_docker_daemon_receipt_binding_v1,
    prepare_local_docker_control_plane_v1,
)


SCHEMA: Final = "hegel-phase3-container-actor-qualification/1"
PROFILE_ID: Final = "hegel-owner-accepted-container-technical-actors-v1"
AUTHORITY_CLASS: Final = "OWNER_ACCEPTED_CONTAINER_TECHNICAL_ACTORS_V1"
PROBE_SCHEMA: Final = "hegel-container-actor-live-probe/1"
TECHNICAL_ACTOR_DISCLOSURE_V1: Final[Mapping[str, bool]] = MappingProxyType(
    {
        "same_admin_controller": True,
        "organizational_independence": False,
        "independent_human_actors": False,
        "technical_role_independence": True,
        "owner_accepted_threat_model": True,
        "remote_attestation": False,
        "hardware_key_nonexportability": False,
    }
)

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT: Final = PROJECT_ROOT.parent
PROFILE_PATH: Final = PROJECT_ROOT / "config/phase3_container_actor_profile_v1.json"
SECCOMP_PATH: Final = PROJECT_ROOT / "config/phase3_internal_actor_seccomp_v1.json"
PYTHON_PROBE_PATH: Final = PROJECT_ROOT / "tools/phase3_container_actor_probe_v1.py"
RUST_PROBE_PATH: Final = PROJECT_ROOT / "tools/phase3_container_actor_probe_v1.rs"
RUNTIME_PATH: Final = Path(__file__).resolve()

PURPOSE_ROLES: Final = {
    1: "CUSTODIAN",
    2: "PYTHON_ATTESTER",
    3: "RUST_ATTESTER",
    4: "POLICY_AUDITOR",
}
PURPOSE_IMAGES: Final = {
    1: "custodian",
    2: "python_attester",
    3: "rust_attester",
    4: "policy_auditor",
}
EXPECTED_ENV_KEYS: Final = {
    "HEGEL_ACTOR_PROFILE_ID",
    "HEGEL_HOST_REPOSITORY_PATH_SHA256",
    "HEGEL_PURPOSE_ID",
    "HEGEL_PROBE_LINGER_SECONDS",
    "LC_ALL",
    "PATH",
    "PYTHONCOERCECLOCALE",
    "PYTHONUTF8",
}
EXPECTED_CAP_FIELDS: Final = ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
EXPECTED_NAMESPACE_KINDS: Final = ("pid", "mnt", "net", "ipc", "uts")
EXPECTED_PROBE_IDS: Final = (
    "socket(AF_INET, SOCK_STREAM)",
    "socket(AF_INET6, SOCK_STREAM)",
    "mount",
    "ptrace(PTRACE_TRACEME)",
    "bpf(BPF_MAP_CREATE)",
    "perf_event_open",
)
MAX_OUTPUT_BYTES: Final = 64 * 1024
COMMAND_TIMEOUT_SECONDS: Final = 90

FAIL_DOCKER_UNAVAILABLE: Final = "FAIL_CONTAINER_ACTOR_DOCKER_UNAVAILABLE"
FAIL_LOCAL_IMAGE_MISSING: Final = "FAIL_CONTAINER_ACTOR_LOCAL_IMAGE_MISSING"
FAIL_IMAGE_BINDING: Final = "FAIL_CONTAINER_ACTOR_IMAGE_BINDING"
FAIL_INPUT_BINDING: Final = "FAIL_CONTAINER_ACTOR_INPUT_BINDING"
FAIL_OFFLINE_RUST_BUILD: Final = "FAIL_CONTAINER_ACTOR_OFFLINE_RUST_BUILD"
FAIL_CONTAINER_CREATE: Final = "FAIL_CONTAINER_ACTOR_CREATE"
FAIL_CONTAINER_EXECUTION: Final = "FAIL_CONTAINER_ACTOR_EXECUTION"
FAIL_CONTAINER_INSPECT: Final = "FAIL_CONTAINER_ACTOR_INSPECT"
FAIL_CONTAINER_REMOVAL: Final = "FAIL_CONTAINER_ACTOR_REMOVAL"
FAIL_OUTPUT_FRAMING: Final = "FAIL_CONTAINER_ACTOR_OUTPUT_FRAMING"
FAIL_LIVE_PROBE: Final = "FAIL_CONTAINER_ACTOR_LIVE_PROBE"
FAIL_PURPOSE_SEPARATION: Final = "FAIL_CONTAINER_ACTOR_PURPOSE_SEPARATION"
FAIL_IMPLEMENTATION_MISMATCH: Final = "FAIL_CONTAINER_ACTOR_IMPLEMENTATION_MISMATCH"
FAIL_NEGATIVE_CONTROL: Final = "FAIL_CONTAINER_ACTOR_NEGATIVE_CONTROL"
FAIL_REPORT_INVALID: Final = "FAIL_CONTAINER_ACTOR_REPORT_INVALID"
FAIL_LOCAL_RUNTIME: Final = "FAIL_CONTAINER_ACTOR_LOCAL_RUNTIME"


class ContainerActorQualificationError(RuntimeError):
    """Stable fail-closed error from the technical-actor qualifier."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise ContainerActorQualificationError(code, detail)


def _run(
    command: Sequence[str],
    *,
    timeout: int = COMMAND_TIMEOUT_SECONDS,
    check: bool = True,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    if command and command[0] == "docker":
        _fail(FAIL_DOCKER_UNAVAILABLE, "unbound Docker CLI invocation is forbidden")
    try:
        completed = subprocess.run(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            env=(
                {"LC_ALL": "C", "LANG": "C", "PATH": "/usr/bin:/bin"}
                if environment is None
                else dict(environment)
            ),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail(FAIL_DOCKER_UNAVAILABLE, f"command failed to execute: {exc}")
    if check and completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", "replace")[-2000:]
        _fail(FAIL_CONTAINER_EXECUTION, f"command exited {completed.returncode}: {stderr}")
    return completed


def _git_run(
    arguments: Sequence[str],
    *,
    timeout: int = COMMAND_TIMEOUT_SECONDS,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    """Run one Commit-A Git read with no ambient config/ref/object injection."""

    if not arguments or any(
        type(value) is not str or not value or "\0" in value
        for value in arguments
    ):
        _fail(FAIL_INPUT_BINDING, "Git argument vector is malformed")
    try:
        completed = subprocess.run(
            ["/usr/bin/git", *arguments],
            cwd=REPOSITORY_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            env={
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
            },
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail(FAIL_INPUT_BINDING, f"Git read failed to execute: {exc}")
    if check and completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", "replace")[-2000:]
        _fail(
            FAIL_INPUT_BINDING,
            f"Git read exited {completed.returncode}: {stderr}",
        )
    return completed


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _is_exact_technical_actor_disclosure(value: object) -> bool:
    """Require the seven frozen keys, exact booleans, and no extensions."""

    expected = dict(TECHNICAL_ACTOR_DISCLOSURE_V1)
    return (
        type(value) is dict
        and set(value) == set(expected)
        and all(
            type(value[key]) is bool and value[key] is expected[key]
            for key in expected
        )
    )


def _file_binding(path: Path, basis_commit: str) -> dict[str, object]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        _fail(FAIL_INPUT_BINDING, f"cannot read {path}: {exc}")
    blob_preimage = b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
    blob_sha1 = hashlib.sha1(blob_preimage).hexdigest()
    try:
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        _fail(FAIL_INPUT_BINDING, f"input is outside the repository: {path}")
    tree = _git_run(
        ["ls-tree", "-z", basis_commit, "--", relative],
        check=False,
    )
    tree_blob: str | None = None
    if tree.returncode == 0 and tree.stdout:
        header, separator, _tree_path = tree.stdout.rstrip(b"\0").partition(b"\t")
        parts = header.decode("ascii", "strict").split()
        if separator and len(parts) == 3 and parts[1] == "blob":
            tree_blob = parts[2]
    return {
        "repository_path": relative,
        "byte_length": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "git_blob_sha1": blob_sha1,
        "basis_tree_blob_sha1_or_null": tree_blob,
        "basis_commit_matches": tree_blob == blob_sha1,
    }


def _validate_profile_value(value: object) -> dict[str, object]:
    if type(value) is not dict or value.get("profile_id") != PROFILE_ID:
        _fail(FAIL_INPUT_BINDING, "unexpected profile ID or representation")
    if not _is_exact_technical_actor_disclosure(value.get("authority_disclosure")):
        _fail(
            FAIL_INPUT_BINDING,
            "profile does not contain the exact seven-field authority disclosure",
        )
    network = value.get("network_policy")
    if type(network) is not dict or network != {
        "allow_registry_access": False,
        "allow_runtime_network": False,
        "docker_network": "none",
        "pull_policy": "never",
    }:
        _fail(FAIL_INPUT_BINDING, "profile does not freeze the offline network policy")
    return dict(value)


def _load_profile() -> dict[str, object]:
    try:
        value = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(FAIL_INPUT_BINDING, f"profile is unreadable: {exc}")
    return _validate_profile_value(value)


def _docker_version(
    control_plane: LocalDockerControlPlaneV1,
) -> tuple[dict[str, object], dict[str, object]]:
    completed = _run(
        control_plane.command("version", "--format", "{{json .}}"),
        environment=control_plane.environment,
    )
    try:
        value = json.loads(completed.stdout)
        client = value["Client"]
        server = value["Server"]
        summary = {
            "client_version": client["Version"],
            "client_api_version": client["ApiVersion"],
            "server_version": server["Version"],
            "server_api_version": server["ApiVersion"],
            "server_os": server["Os"],
            "server_arch": server["Arch"],
        }
        return summary, value
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        _fail(FAIL_DOCKER_UNAVAILABLE, f"invalid docker version output: {exc}")


def _inspect_local_image(
    image_ref: str,
    control_plane: LocalDockerControlPlaneV1,
) -> dict[str, object]:
    # ``docker image inspect`` is deliberately the only image operation.  It
    # cannot pull or contact a registry when the content is missing.
    completed = _run(
        control_plane.command("image", "inspect", image_ref),
        check=False,
        environment=control_plane.environment,
    )
    if completed.returncode != 0:
        _fail(FAIL_LOCAL_IMAGE_MISSING, f"pinned local image is absent: {image_ref}")
    try:
        rows = json.loads(completed.stdout)
        row = rows[0]
        repo_digests = row["RepoDigests"]
        image_id = row["Id"]
        architecture = row["Architecture"]
        operating_system = row["Os"]
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
        _fail(FAIL_IMAGE_BINDING, f"invalid image inspection for {image_ref}: {exc}")
    if image_ref not in repo_digests or architecture != "amd64" or operating_system != "linux":
        _fail(FAIL_IMAGE_BINDING, f"image digest/platform binding failed: {image_ref}")
    return {
        "requested_digest_ref": image_ref,
        "observed_image_id": image_id,
        "observed_repo_digests": sorted(repo_digests),
        "architecture": architecture,
        "os": operating_system,
        "local_only_inspection": True,
    }


def _remove_container(
    container_id: str,
    control_plane: LocalDockerControlPlaneV1,
) -> dict[str, object]:
    stopped = _run(
        control_plane.command("container", "stop", "--time=1", container_id),
        timeout=10,
        check=False,
        environment=control_plane.environment,
    )
    stopped_inspect = _run(
        control_plane.command("container", "inspect", container_id),
        check=False,
        environment=control_plane.environment,
    )
    stopped_pid_zero = False
    if stopped.returncode == 0 and stopped_inspect.returncode == 0:
        try:
            stopped_pid_zero = json.loads(stopped_inspect.stdout)[0]["State"]["Pid"] == 0
        except (IndexError, KeyError, TypeError, json.JSONDecodeError):
            stopped_pid_zero = False
    removed = _run(
        control_plane.command("container", "rm", "--force", container_id),
        check=False,
        environment=control_plane.environment,
    )
    if removed.returncode != 0:
        _fail(FAIL_CONTAINER_REMOVAL, f"could not remove container {container_id}")
    inspect_after = _run(
        control_plane.command("container", "inspect", container_id),
        check=False,
        environment=control_plane.environment,
    )
    listed_after = _run(
        control_plane.command(
            "container", "ls", "--all", "--quiet", "--no-trunc", "--filter",
            f"id={container_id}",
        ),
        check=False,
        environment=control_plane.environment,
    )
    absent = stopped_pid_zero and inspect_after.returncode != 0 and not listed_after.stdout.strip()
    if not absent:
        _fail(FAIL_CONTAINER_REMOVAL, f"container still exists after removal: {container_id}")
    return {
        "stopped_host_pid_zero": True,
        "container_removed": True,
        "container_and_descendants_absent": True,
    }


def _compile_rust_probe_offline(
    *,
    rust_image: str,
    temporary_root: Path,
    control_plane: LocalDockerControlPlaneV1,
) -> tuple[Path, dict[str, object]]:
    source_dir = temporary_root / "rust_source"
    output_dir = temporary_root / "rust_output"
    source_dir.mkdir(mode=0o755)
    output_dir.mkdir(mode=0o777)
    # ``Path.mkdir(mode=...)`` is still masked by the host umask.  The build
    # container deliberately runs as uid/gid 65534, so make the narrowly
    # scoped output bind writable explicitly after creation.
    output_dir.chmod(0o777)
    source_copy = source_dir / "probe.rs"
    shutil.copyfile(RUST_PROBE_PATH, source_copy)
    source_copy.chmod(0o444)
    binary_path = output_dir / "probe"
    builder_name = f"hegel-m25-rust-probe-builder-{secrets.token_hex(8)}"
    command = [
        *control_plane.command("run"),
        "--rm",
        f"--name={builder_name}",
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges=true",
        "--user=65534:65534",
        "--pids-limit=64",
        "--memory=512m",
        "--memory-swap=512m",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m,uid=65534,gid=65534,mode=0700",
        "--mount",
        f"type=bind,src={source_dir},dst=/source,readonly,bind-propagation=rprivate",
        "--mount",
        f"type=bind,src={output_dir},dst=/build,bind-propagation=rprivate",
        "--entrypoint=/usr/bin/env",
        rust_image,
        "-i",
        "PATH=/usr/local/cargo/bin:/usr/bin:/bin",
        "RUSTUP_HOME=/usr/local/rustup",
        "CARGO_HOME=/usr/local/cargo",
        "TMPDIR=/build",
        "/usr/local/cargo/bin/rustc",
        "--edition=2021",
        "-C",
        "debuginfo=0",
        "-C",
        "strip=symbols",
        "/source/probe.rs",
        "-o",
        "/build/probe",
    ]
    completed = _run(
        command,
        timeout=180,
        check=False,
        environment=control_plane.environment,
    )
    if completed.returncode != 0 or not binary_path.is_file():
        stderr = completed.stderr.decode("utf-8", "replace")[-12000:]
        _fail(FAIL_OFFLINE_RUST_BUILD, f"offline Rust probe compilation failed: {stderr}")
    # The file is created by uid 65534 inside the build container.  Do not
    # attempt a host-side chmod as the unprivileged supervisor is not its
    # owner; the later per-purpose snapshot is host-owned and is set to 0555.
    if not binary_path.is_file() or not binary_path.stat().st_mode & 0o111:
        _fail(FAIL_OFFLINE_RUST_BUILD, "offline Rust probe output is not executable")
    version = _run(
        [
            *control_plane.command("run"),
            "--rm",
            "--pull=never",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges=true",
            "--user=65534:65534",
            "--pids-limit=64",
            "--memory=512m",
            "--memory-swap=512m",
            "--ipc=private",
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m,uid=65534,gid=65534,mode=0700",
            "--entrypoint=/usr/bin/env",
            rust_image,
            "-i",
            "PATH=/usr/local/cargo/bin:/usr/bin:/bin",
            "RUSTUP_HOME=/usr/local/rustup",
            "CARGO_HOME=/usr/local/cargo",
            "/usr/local/cargo/bin/rustc",
            "--version",
            "--verbose",
        ],
        environment=control_plane.environment,
    )
    payload = binary_path.read_bytes()
    return binary_path, {
        "method": "PINNED_RUST_IMAGE_OFFLINE_SOURCE_COMPILE",
        "network": "none",
        "pull_policy": "never",
        "image_ref": rust_image,
        "rustc_version": version.stdout.decode("utf-8", "strict").strip(),
        "source_sha256": hashlib.sha256(RUST_PROBE_PATH.read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(payload).hexdigest(),
        "binary_size": len(payload),
    }


def _snapshot_actor_inputs(
    purpose_id: int,
    root: Path,
    rust_binary: Path,
) -> Path:
    snapshot = root / f"actor_{purpose_id}_input"
    snapshot.mkdir(mode=0o755)
    shutil.copyfile(PROFILE_PATH, snapshot / "profile.json")
    shutil.copyfile(SECCOMP_PATH, snapshot / "seccomp.json")
    if purpose_id == 3:
        shutil.copyfile(RUST_PROBE_PATH, snapshot / "probe.rs")
        shutil.copyfile(rust_binary, snapshot / "probe")
        (snapshot / "probe").chmod(0o555)
    else:
        shutil.copyfile(PYTHON_PROBE_PATH, snapshot / "probe.py")
    for path in snapshot.iterdir():
        if path.name != "probe":
            path.chmod(0o444)
    snapshot.chmod(0o555)
    return snapshot


def _actor_environment(purpose_id: int) -> dict[str, str]:
    host_repository_path = REPOSITORY_ROOT.resolve().as_posix().encode("utf-8")
    return {
        "HEGEL_ACTOR_PROFILE_ID": PROFILE_ID,
        "HEGEL_HOST_REPOSITORY_PATH_SHA256": hashlib.sha256(
            host_repository_path
        ).hexdigest(),
        "HEGEL_PURPOSE_ID": str(purpose_id),
        "HEGEL_PROBE_LINGER_SECONDS": "30",
        "LC_ALL": "C",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONCOERCECLOCALE": "0",
        "PYTHONUTF8": "1",
    }


def _actor_launch_environment(purpose_id: int) -> dict[str, str]:
    """Return the in-container env; the raw host path is never reported."""

    environment = _actor_environment(purpose_id)
    environment["HEGEL_HOST_REPOSITORY_PATH"] = REPOSITORY_ROOT.resolve().as_posix()
    return environment


def _decode_probe_output(stdout: bytes) -> dict[str, object]:
    if len(stdout) > MAX_OUTPUT_BYTES:
        _fail(FAIL_OUTPUT_FRAMING, "probe output exceeded the byte limit")
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        _fail(FAIL_OUTPUT_FRAMING, "probe output must contain exactly one non-empty line")
    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        _fail(FAIL_OUTPUT_FRAMING, f"probe output is not JSON: {exc}")
    if type(value) is not dict:
        _fail(FAIL_OUTPUT_FRAMING, "probe output must be a JSON object")
    canonical = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    if lines[0] != canonical:
        _fail(FAIL_OUTPUT_FRAMING, "probe output is not canonical single-line JSON")
    lowered = lines[0].lower()
    for forbidden in (b"raw_seed", b"private_key", b"secret_key", b"mnemonic"):
        if forbidden in lowered:
            _fail(FAIL_OUTPUT_FRAMING, f"secret lint rejected token {forbidden!r}")
    return value


def _validate_probe(
    probe: Mapping[str, object],
    *,
    purpose_id: int,
    implementation: str,
    require_custom_blocking: bool,
) -> dict[str, object]:
    if probe.get("schema") != PROBE_SCHEMA or probe.get("implementation") != implementation:
        _fail(FAIL_LIVE_PROBE, "probe schema/implementation mismatch")
    if probe.get("profile_id") != PROFILE_ID or probe.get("purpose_id") != purpose_id:
        _fail(FAIL_LIVE_PROBE, "profile or purpose mismatch")
    identity = probe.get("identity")
    if type(identity) is not dict or identity != {"uid": 65534, "gid": 65534, "pid": 1}:
        _fail(FAIL_LIVE_PROBE, "worker is not non-root PID 1")
    status = probe.get("proc_status")
    if type(status) is not dict:
        _fail(FAIL_LIVE_PROBE, "missing /proc status")
    try:
        caps_zero = all(int(str(status[field]), 16) == 0 for field in EXPECTED_CAP_FIELDS)
    except (KeyError, ValueError):
        caps_zero = False
    if not caps_zero or status.get("NoNewPrivs") != 1 or status.get("Seccomp") != 2:
        _fail(FAIL_LIVE_PROBE, "capabilities, NNP, or seccomp live state failed")
    if probe.get("network_interfaces") != ["lo"]:
        _fail(FAIL_LIVE_PROBE, "network namespace exposes an interface other than lo")
    namespaces = probe.get("namespaces")
    if type(namespaces) is not dict or set(namespaces) != set(EXPECTED_NAMESPACE_KINDS):
        _fail(FAIL_LIVE_PROBE, "namespace identity field set is invalid")
    if not all(re.fullmatch(r"[a-z]+:\[[0-9]+\]", str(value)) for value in namespaces.values()):
        _fail(FAIL_LIVE_PROBE, "namespace identity format is invalid")
    rows = probe.get("syscall_probes")
    if type(rows) is not list or [row.get("probe_id") for row in rows if type(row) is dict] != list(EXPECTED_PROBE_IDS):
        _fail(FAIL_LIVE_PROBE, "syscall probe order/set is invalid")
    blocked = []
    allowed = []
    for row in rows:
        assert isinstance(row, dict)
        if row.get("return_value") == -1 and row.get("errno") == 1:
            blocked.append(row["probe_id"])
        elif type(row.get("return_value")) is int and row["return_value"] >= 0 and row.get("errno") == 0:
            allowed.append(row["probe_id"])
    if require_custom_blocking and blocked != list(EXPECTED_PROBE_IDS):
        _fail(FAIL_LIVE_PROBE, "not all six syscalls returned -1/EPERM")
    filesystem = probe.get("filesystem_probes")
    if type(filesystem) is not dict:
        _fail(FAIL_LIVE_PROBE, "filesystem probes are missing")
    for key in ("root_write", "input_write"):
        row = filesystem.get(key)
        if type(row) is not dict or row.get("denied") is not True or row.get("errno") not in {1, 13, 30}:
            _fail(FAIL_LIVE_PROBE, f"{key} was not denied")
    if filesystem.get("forbidden_paths_present") != [] or filesystem.get("cross_purpose_paths_present") != []:
        _fail(FAIL_LIVE_PROBE, "a forbidden or cross-purpose path is visible")
    expected_env = _actor_environment(purpose_id)
    if probe.get("environment") != expected_env:
        _fail(
            FAIL_LIVE_PROBE,
            f"environment is not the exact allowlist: {probe.get('environment')!r}",
        )
    if probe.get("open_fds") != [0, 1, 2]:
        _fail(FAIL_LIVE_PROBE, "inherited FD set is not exactly 0,1,2")
    return {
        "capability_sets_all_zero": True,
        "no_new_privileges": 1,
        "seccomp_mode": 2,
        "network_interfaces_exactly_lo": True,
        "blocked_syscalls": blocked,
        "allowed_syscalls": allowed,
        "root_and_input_writes_denied": True,
        "forbidden_and_cross_purpose_paths_absent": True,
        "exact_environment": True,
        "exact_inherited_fds": True,
    }


def _validate_requested_controls(
    inspect: Mapping[str, object],
    *,
    image_ref: str,
    custom_seccomp: bool,
) -> dict[str, object]:
    config = inspect.get("Config")
    host = inspect.get("HostConfig")
    mounts = inspect.get("Mounts")
    if type(config) is not dict or type(host) is not dict or type(mounts) is not list:
        _fail(FAIL_CONTAINER_INSPECT, "container inspect field set is invalid")
    security = host.get("SecurityOpt")
    if type(security) is not list:
        _fail(FAIL_CONTAINER_INSPECT, "security options are absent")
    seccomp_options = [
        str(item)[len("seccomp=") :]
        for item in security
        if str(item).startswith("seccomp=")
    ]
    custom_seccomp_exact = not custom_seccomp
    if custom_seccomp and len(seccomp_options) == 1:
        try:
            custom_seccomp_exact = json.loads(seccomp_options[0]) == json.loads(
                SECCOMP_PATH.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            custom_seccomp_exact = False
    expected_tmpfs = "rw,noexec,nosuid,nodev,size=64m,uid=65534,gid=65534,mode=0700"
    checks = {
        "image_digest_exact": config.get("Image") == image_ref,
        "user_nonroot_exact": config.get("User") == "65534:65534",
        "network_none": host.get("NetworkMode") == "none",
        "read_only_root": host.get("ReadonlyRootfs") is True,
        "all_capabilities_dropped": host.get("CapDrop") == ["ALL"],
        "no_new_privileges_requested": any(str(item).startswith("no-new-privileges") for item in security),
        "custom_seccomp_exact": custom_seccomp_exact,
        "pids_limit_64": host.get("PidsLimit") == 64,
        "memory_512m": host.get("Memory") == 512 * 1024 * 1024,
        "memory_swap_512m": host.get("MemorySwap") == 512 * 1024 * 1024,
        "purpose_private_tmpfs_exact": host.get("Tmpfs") == {"/tmp": expected_tmpfs},
        "ipc_private": host.get("IpcMode") == "private",
        "nofile_limit_64": host.get("Ulimits") == [
            {"Name": "nofile", "Hard": 64, "Soft": 64}
        ],
        "input_mount_read_only": any(
            mount.get("Destination") == "/actor_input" and mount.get("RW") is False
            for mount in mounts
            if type(mount) is dict
        ),
        "no_docker_socket_mount": not any(
            mount.get("Destination") in {"/var/run/docker.sock", "/run/docker.sock"}
            for mount in mounts
            if type(mount) is dict
        ),
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        _fail(FAIL_CONTAINER_INSPECT, f"requested controls failed: {failed}")
    return checks


def _run_actor(
    *,
    purpose_id: int,
    role: str,
    image_ref: str,
    input_snapshot: Path,
    custom_seccomp: bool,
    control_plane: LocalDockerControlPlaneV1,
) -> dict[str, object]:
    name = f"hegel-m25-actor-{purpose_id}-{secrets.token_hex(8)}"
    command = [
        *control_plane.command("container", "create"),
        f"--name={name}",
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges=true",
        "--user=65534:65534",
        "--pids-limit=64",
        "--memory=512m",
        "--memory-swap=512m",
        "--ipc=private",
        "--ulimit=nofile=64:64",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m,uid=65534,gid=65534,mode=0700",
        "--mount",
        f"type=bind,src={input_snapshot},dst=/actor_input,readonly,bind-propagation=rprivate",
    ]
    if custom_seccomp:
        command.append(f"--security-opt=seccomp={SECCOMP_PATH}")
    command.extend(["--entrypoint=/usr/bin/env", image_ref, "-i"])
    for key, value in _actor_launch_environment(purpose_id).items():
        command.append(f"{key}={value}")
    if purpose_id == 3:
        command.append("/actor_input/probe")
        implementation = "rust-ffi-v1"
    else:
        command.extend(["/usr/local/bin/python3", "-I", "/actor_input/probe.py"])
        implementation = "python-ctypes-v1"

    created = _run(
        command,
        check=False,
        environment=control_plane.environment,
    )
    if created.returncode != 0:
        detail = created.stderr.decode("utf-8", "replace")[-3000:]
        _fail(FAIL_CONTAINER_CREATE, f"container create failed: {detail}")
    container_id = created.stdout.decode("ascii", "strict").strip()
    try:
        started = _run(
            control_plane.command("container", "start", container_id),
            check=False,
            environment=control_plane.environment,
        )
        if started.returncode != 0:
            stderr = started.stderr.decode("utf-8", "replace")[-3000:]
            _fail(FAIL_CONTAINER_EXECUTION, f"actor start failed: {stderr}")
        stdout = b""
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            logs = _run(
                control_plane.command("container", "logs", container_id),
                check=False,
                environment=control_plane.environment,
            )
            stdout = logs.stdout
            if stdout.strip():
                break
            time.sleep(0.05)
        if not stdout.strip():
            _fail(FAIL_CONTAINER_EXECUTION, "actor emitted no live probe before timeout")
        inspected = _run(
            control_plane.command("container", "inspect", container_id),
            environment=control_plane.environment,
        )
        try:
            inspect_row = json.loads(inspected.stdout)[0]
        except (IndexError, TypeError, json.JSONDecodeError) as exc:
            _fail(FAIL_CONTAINER_INSPECT, f"invalid inspect JSON: {exc}")
        state = inspect_row.get("State")
        if type(state) is not dict or state.get("Running") is not True or not isinstance(state.get("Pid"), int) or state["Pid"] <= 0:
            _fail(FAIL_CONTAINER_EXECUTION, "actor was not alive during supervisor inspection")
        host_pid = state["Pid"]
        probe = _decode_probe_output(stdout)
        live_checks = _validate_probe(
            probe,
            purpose_id=purpose_id,
            implementation=implementation,
            require_custom_blocking=custom_seccomp,
        )
        requested = _validate_requested_controls(
            inspect_row,
            image_ref=image_ref,
            custom_seccomp=custom_seccomp,
        )
    except BaseException:
        _remove_container(container_id, control_plane)
        raise
    return {
        "purpose_id": purpose_id,
        "role": role,
        "image_ref": image_ref,
        "container_id": container_id,
        "host_pid_while_running": host_pid,
        "requested_control_checks": requested,
        "live_probe": probe,
        "output_binding": {
            "byte_length": len(stdout),
            "sha256": hashlib.sha256(stdout).hexdigest(),
            "canonical_single_line_json": True,
            "secret_lint_passed": True,
        },
        "live_check_summary": live_checks,
        "cleanup": None,
    }


def _normalized_probe_for_implementation_agreement(probe: Mapping[str, object]) -> dict[str, object]:
    environment = deepcopy(probe["environment"])
    assert isinstance(environment, dict)
    environment["HEGEL_PURPOSE_ID"] = "<PURPOSE>"
    filesystem = probe["filesystem_probes"]
    assert isinstance(filesystem, dict)
    return {
        "identity": {"uid": 65534, "gid": 65534, "pid": 1},
        "proc_status": probe["proc_status"],
        "network_interfaces": probe["network_interfaces"],
        "syscall_probes": probe["syscall_probes"],
        "filesystem_denials": {
            "root": bool(filesystem["root_write"]["denied"]),
            "input": bool(filesystem["input_write"]["denied"]),
            "forbidden": filesystem["forbidden_paths_present"],
            "cross": filesystem["cross_purpose_paths_present"],
        },
        "environment": environment,
        "open_fds": probe["open_fds"],
    }


def _compare_python_rust(
    python_probe: Mapping[str, object],
    rust_probe: Mapping[str, object],
) -> str:
    left = _normalized_probe_for_implementation_agreement(python_probe)
    right = _normalized_probe_for_implementation_agreement(rust_probe)
    if left != right:
        _fail(FAIL_IMPLEMENTATION_MISMATCH, "Python and Rust live results disagree")
    return _canonical_sha256(left)


def _validate_cross_actor(actor_reports: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(actor_reports) != 4 or [row.get("purpose_id") for row in actor_reports] != [1, 2, 3, 4]:
        _fail(FAIL_PURPOSE_SEPARATION, "purpose rows are missing, duplicated, or reordered")
    container_ids = [row.get("container_id") for row in actor_reports]
    if len(set(container_ids)) != 4:
        _fail(FAIL_PURPOSE_SEPARATION, "container IDs are not distinct")
    host_pids = [row.get("host_pid_while_running") for row in actor_reports]
    if len(set(host_pids)) != 4 or not all(type(value) is int and value > 0 for value in host_pids):
        _fail(FAIL_PURPOSE_SEPARATION, "live host PIDs are not distinct")
    namespace_distinct: dict[str, bool] = {}
    for kind in EXPECTED_NAMESPACE_KINDS:
        values = [row["live_probe"]["namespaces"][kind] for row in actor_reports]
        namespace_distinct[kind] = len(set(values)) == 4
    if not all(namespace_distinct.values()):
        _fail(FAIL_PURPOSE_SEPARATION, "one or more actor namespaces were reused")
    agreement = _compare_python_rust(
        actor_reports[1]["live_probe"], actor_reports[2]["live_probe"]
    )
    return {
        "distinct_container_ids": True,
        "distinct_live_host_pids": True,
        "namespace_identity_distinct_by_kind": namespace_distinct,
        "purpose_ids_exact": [1, 2, 3, 4],
        "python_rust_live_result_agreement_sha256": agreement,
        "shared_writable_role_mounts": False,
        "technical_role_independence": True,
    }


def _run_negative_control(
    *,
    python_image: str,
    input_snapshot: Path,
    control_plane: LocalDockerControlPlaneV1,
) -> dict[str, object]:
    report = _run_actor(
        purpose_id=1,
        role="DEFAULT_DOCKER_SECCOMP_NEGATIVE_CONTROL",
        image_ref=python_image,
        input_snapshot=input_snapshot,
        custom_seccomp=False,
        control_plane=control_plane,
    )
    try:
        rows = report["live_probe"]["syscall_probes"]
        allowed = [row["probe_id"] for row in rows if row["return_value"] >= 0]
        required_allowed = {
            "socket(AF_INET, SOCK_STREAM)",
            "socket(AF_INET6, SOCK_STREAM)",
            "ptrace(PTRACE_TRACEME)",
        }
        if not required_allowed.issubset(set(allowed)):
            _fail(FAIL_NEGATIVE_CONTROL, "Docker default profile did not reproduce the frozen negative control")
        rejected_by_custom_validator = False
        try:
            _validate_probe(
                report["live_probe"],
                purpose_id=1,
                implementation="python-ctypes-v1",
                require_custom_blocking=True,
            )
        except ContainerActorQualificationError as exc:
            rejected_by_custom_validator = exc.code == FAIL_LIVE_PROBE
        if not rejected_by_custom_validator:
            _fail(FAIL_NEGATIVE_CONTROL, "default seccomp evidence was not rejected")
    finally:
        cleanup = _remove_container(str(report["container_id"]), control_plane)
    return {
        "default_profile_allowed_probes": allowed,
        "default_profile_is_insufficient": True,
        "default_profile_rejected_by_qualifier": True,
        "container_id": report["container_id"],
        "cleanup": cleanup,
    }


def _fault_injection_checks(actor_reports: Sequence[Mapping[str, object]]) -> dict[str, bool]:
    mismatch = deepcopy(actor_reports[2]["live_probe"])
    mismatch["network_interfaces"] = ["eth0", "lo"]
    mismatch_rejected = False
    try:
        _compare_python_rust(actor_reports[1]["live_probe"], mismatch)
    except ContainerActorQualificationError as exc:
        mismatch_rejected = exc.code == FAIL_IMPLEMENTATION_MISMATCH

    duplicate = deepcopy(list(actor_reports))
    duplicate[3]["purpose_id"] = 3
    duplicate_rejected = False
    try:
        _validate_cross_actor(duplicate)
    except ContainerActorQualificationError as exc:
        duplicate_rejected = exc.code == FAIL_PURPOSE_SEPARATION

    replay = deepcopy(actor_reports[0]["live_probe"])
    replay["purpose_id"] = 2
    purpose_replay_rejected = False
    try:
        _validate_probe(
            replay,
            purpose_id=2,
            implementation="python-ctypes-v1",
            require_custom_blocking=True,
        )
    except ContainerActorQualificationError as exc:
        purpose_replay_rejected = exc.code == FAIL_LIVE_PROBE

    result = {
        "python_rust_mismatch_rejected": mismatch_rejected,
        "duplicate_purpose_rejected": duplicate_rejected,
        "cross_purpose_output_replay_rejected": purpose_replay_rejected,
    }
    if not all(result.values()):
        _fail(FAIL_PURPOSE_SEPARATION, "one or more fault injections were accepted")
    return result


def _require_exact_keys(value: object, expected: set[str], context: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        _fail(FAIL_REPORT_INVALID, f"{context} does not have the exact field set")
    return value


def _is_lower_hex(value: object, length: int) -> bool:
    return type(value) is str and bool(re.fullmatch(rf"[0-9a-f]{{{length}}}", value))


def _validate_probe_shape_strict(value: object, context: str) -> dict[str, object]:
    probe = _require_exact_keys(
        value,
        {
            "schema",
            "implementation",
            "profile_id",
            "purpose_id",
            "identity",
            "proc_status",
            "namespaces",
            "network_interfaces",
            "syscall_probes",
            "filesystem_probes",
            "environment",
            "open_fds",
        },
        context,
    )
    _require_exact_keys(probe["identity"], {"uid", "gid", "pid"}, f"{context} identity")
    _require_exact_keys(
        probe["proc_status"],
        {*EXPECTED_CAP_FIELDS, "NoNewPrivs", "Seccomp"},
        f"{context} proc status",
    )
    _require_exact_keys(
        probe["namespaces"], set(EXPECTED_NAMESPACE_KINDS), f"{context} namespaces"
    )
    filesystem = _require_exact_keys(
        probe["filesystem_probes"],
        {
            "root_write",
            "input_write",
            "forbidden_paths_present",
            "cross_purpose_paths_present",
        },
        f"{context} filesystem probes",
    )
    _require_exact_keys(filesystem["root_write"], {"denied", "errno"}, f"{context} root write")
    _require_exact_keys(filesystem["input_write"], {"denied", "errno"}, f"{context} input write")
    rows = probe["syscall_probes"]
    if type(rows) is not list or len(rows) != 6:
        _fail(FAIL_REPORT_INVALID, f"{context} syscall row count is not six")
    for index, row in enumerate(rows):
        _require_exact_keys(
            row,
            {"probe_id", "return_value", "errno"},
            f"{context} syscall row {index}",
        )
    environment = probe["environment"]
    if type(environment) is not dict or set(environment) != EXPECTED_ENV_KEYS:
        _fail(FAIL_REPORT_INVALID, f"{context} environment field set is invalid")
    return probe


def validate_qualification_report(
    value: object,
    *,
    profile_override: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Strictly validate and return a defensive copy of a qualification report.

    ``profile_override`` is reserved for a caller that already obtained strict
    profile bytes from a specific Git tree.  The default preserves the live
    qualification validator's worktree binding.
    """

    report = _require_exact_keys(
        value,
        {
            "schema",
            "profile_id",
            "authority_class",
            "basis_commit",
            "basis_commit_contains_all_inputs",
            "input_bindings",
            "docker_runtime",
            "docker_control_plane_receipt",
            "network_and_registry_policy",
            "image_bindings",
            "rust_probe_offline_build",
            "actor_reports",
            "cross_actor_checks",
            "default_seccomp_negative_control",
            "fault_injection_checks",
            "authority_disclosure",
            "ceremony_outputs",
            "all_live_checks_passed",
            "technical_actor_eligible",
            "ineligibility_reason_or_null",
            "qualification_payload_sha256",
        },
        "qualification report",
    )
    if (
        report["schema"] != SCHEMA
        or report["profile_id"] != PROFILE_ID
        or report["authority_class"] != AUTHORITY_CLASS
        or not _is_lower_hex(report["basis_commit"], 40)
    ):
        _fail(FAIL_REPORT_INVALID, "qualification header is invalid")

    claimed_digest = report["qualification_payload_sha256"]
    if not _is_lower_hex(claimed_digest, 64):
        _fail(FAIL_REPORT_INVALID, "qualification payload digest is not lowercase SHA-256")
    unhashed = dict(report)
    del unhashed["qualification_payload_sha256"]
    if _canonical_sha256(unhashed) != claimed_digest:
        _fail(FAIL_REPORT_INVALID, "qualification payload SHA-256 mismatch")

    bindings = _require_exact_keys(
        report["input_bindings"],
        {"profile", "seccomp", "python_probe", "rust_probe", "supervisor_runtime"},
        "input bindings",
    )
    binding_matches: list[bool] = []
    for name, raw_binding in bindings.items():
        binding = _require_exact_keys(
            raw_binding,
            {
                "repository_path",
                "byte_length",
                "sha256",
                "git_blob_sha1",
                "basis_tree_blob_sha1_or_null",
                "basis_commit_matches",
            },
            f"input binding {name}",
        )
        if (
            type(binding["repository_path"]) is not str
            or not binding["repository_path"]
            or type(binding["byte_length"]) is not int
            or binding["byte_length"] <= 0
            or not _is_lower_hex(binding["sha256"], 64)
            or not _is_lower_hex(binding["git_blob_sha1"], 40)
            or type(binding["basis_commit_matches"]) is not bool
        ):
            _fail(FAIL_REPORT_INVALID, f"input binding {name} is malformed")
        tree_blob = binding["basis_tree_blob_sha1_or_null"]
        if tree_blob is not None and not _is_lower_hex(tree_blob, 40):
            _fail(FAIL_REPORT_INVALID, f"input binding {name} has an invalid tree blob")
        if binding["basis_commit_matches"] is not (tree_blob == binding["git_blob_sha1"]):
            _fail(FAIL_REPORT_INVALID, f"input binding {name} consistency failed")
        binding_matches.append(binding["basis_commit_matches"])
    basis_bound = all(binding_matches)
    if report["basis_commit_contains_all_inputs"] is not basis_bound:
        _fail(FAIL_REPORT_INVALID, "basis-commit input-binding summary is inconsistent")

    network_policy = _require_exact_keys(
        report["network_and_registry_policy"],
        {
            "registry_access_performed",
            "image_pull_performed",
            "image_build_performed",
            "runtime_network_enabled",
            "pull_policy",
            "runtime_network",
        },
        "network and registry policy",
    )
    if network_policy != {
        "registry_access_performed": False,
        "image_pull_performed": False,
        "image_build_performed": False,
        "runtime_network_enabled": False,
        "pull_policy": "never",
        "runtime_network": "none",
    }:
        _fail(FAIL_REPORT_INVALID, "offline network/pull/build policy is invalid")

    docker_runtime = _require_exact_keys(
        report["docker_runtime"],
        {
            "client_version",
            "client_api_version",
            "server_version",
            "server_api_version",
            "server_os",
            "server_arch",
        },
        "docker runtime",
    )
    if docker_runtime["server_os"] != "linux" or docker_runtime["server_arch"] != "amd64":
        _fail(FAIL_REPORT_INVALID, "Docker runtime platform is not linux/amd64")
    if not all(type(item) is str and item for item in docker_runtime.values()):
        _fail(FAIL_REPORT_INVALID, "Docker runtime fields must be non-empty strings")
    try:
        local_docker_daemon_receipt_binding_v1(
            report["docker_control_plane_receipt"]  # type: ignore[arg-type]
        )
    except Phase3LocalRuntimeError as exc:
        _fail(FAIL_REPORT_INVALID, f"Docker control-plane receipt is invalid: {exc.code}")

    profile = (
        _load_profile()
        if profile_override is None
        else _validate_profile_value(dict(profile_override))
    )
    images = profile["images"]
    assert isinstance(images, dict)
    image_bindings = _require_exact_keys(
        report["image_bindings"], set(images), "image bindings"
    )
    for role, image_ref in images.items():
        binding = _require_exact_keys(
            image_bindings[role],
            {
                "requested_digest_ref",
                "observed_image_id",
                "observed_repo_digests",
                "architecture",
                "os",
                "local_only_inspection",
            },
            f"image binding {role}",
        )
        if not (
            type(binding["observed_image_id"]) is str
            and re.fullmatch(r"sha256:[0-9a-f]{64}", binding["observed_image_id"])
            and binding["requested_digest_ref"] == image_ref
            and binding["architecture"] == "amd64"
            and binding["os"] == "linux"
            and binding["local_only_inspection"] is True
            and type(binding["observed_repo_digests"]) is list
            and image_ref in binding["observed_repo_digests"]
        ):
            _fail(FAIL_REPORT_INVALID, f"image binding {role} is invalid")

    rust_build = _require_exact_keys(
        report["rust_probe_offline_build"],
        {
            "method",
            "network",
            "pull_policy",
            "image_ref",
            "rustc_version",
            "source_sha256",
            "binary_sha256",
            "binary_size",
        },
        "offline Rust build",
    )
    if (
        rust_build["method"] != "PINNED_RUST_IMAGE_OFFLINE_SOURCE_COMPILE"
        or rust_build["network"] != "none"
        or rust_build["pull_policy"] != "never"
        or rust_build["image_ref"] != images["rust_attester"]
        or not _is_lower_hex(rust_build["source_sha256"], 64)
        or not _is_lower_hex(rust_build["binary_sha256"], 64)
        or type(rust_build["binary_size"]) is not int
        or rust_build["binary_size"] <= 0
    ):
        _fail(FAIL_REPORT_INVALID, "offline Rust build binding is invalid")

    actors = report["actor_reports"]
    if type(actors) is not list or len(actors) != 4:
        _fail(FAIL_REPORT_INVALID, "actor report count is not four")
    for purpose_id, actor_raw in enumerate(actors, start=1):
        actor = _require_exact_keys(
            actor_raw,
            {
                "purpose_id",
                "role",
                "image_ref",
                "container_id",
                "host_pid_while_running",
                "requested_control_checks",
                "live_probe",
                "output_binding",
                "live_check_summary",
                "cleanup",
            },
            f"actor {purpose_id}",
        )
        if (
            actor["purpose_id"] != purpose_id
            or actor["role"] != PURPOSE_ROLES[purpose_id]
            or actor["image_ref"] != images[PURPOSE_IMAGES[purpose_id]]
            or not _is_lower_hex(actor["container_id"], 64)
            or type(actor["host_pid_while_running"]) is not int
            or actor["host_pid_while_running"] <= 0
        ):
            _fail(FAIL_REPORT_INVALID, f"actor {purpose_id} identity is invalid")
        requested = _require_exact_keys(
            actor["requested_control_checks"],
            {
                "image_digest_exact",
                "user_nonroot_exact",
                "network_none",
                "read_only_root",
                "all_capabilities_dropped",
                "no_new_privileges_requested",
                "custom_seccomp_exact",
                "pids_limit_64",
                "memory_512m",
                "memory_swap_512m",
                "purpose_private_tmpfs_exact",
                "ipc_private",
                "nofile_limit_64",
                "input_mount_read_only",
                "no_docker_socket_mount",
            },
            f"actor {purpose_id} requested controls",
        )
        if not all(value is True for value in requested.values()):
            _fail(FAIL_REPORT_INVALID, f"actor {purpose_id} requested controls failed")
        _validate_probe_shape_strict(actor["live_probe"], f"actor {purpose_id} live probe")
        output_binding = _require_exact_keys(
            actor["output_binding"],
            {"byte_length", "sha256", "canonical_single_line_json", "secret_lint_passed"},
            f"actor {purpose_id} output binding",
        )
        reconstructed_output = (
            json.dumps(
                actor["live_probe"],
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
        if (
            output_binding["byte_length"] != len(reconstructed_output)
            or output_binding["sha256"] != hashlib.sha256(reconstructed_output).hexdigest()
            or output_binding["canonical_single_line_json"] is not True
            or output_binding["secret_lint_passed"] is not True
        ):
            _fail(FAIL_REPORT_INVALID, f"actor {purpose_id} output binding is invalid")
        live_summary = _validate_probe(
            actor["live_probe"],
            purpose_id=purpose_id,
            implementation="rust-ffi-v1" if purpose_id == 3 else "python-ctypes-v1",
            require_custom_blocking=True,
        )
        summary = _require_exact_keys(
            actor["live_check_summary"],
            {
                "capability_sets_all_zero",
                "no_new_privileges",
                "seccomp_mode",
                "network_interfaces_exactly_lo",
                "blocked_syscalls",
                "allowed_syscalls",
                "root_and_input_writes_denied",
                "forbidden_and_cross_purpose_paths_absent",
                "exact_environment",
                "exact_inherited_fds",
            },
            f"actor {purpose_id} live summary",
        )
        if summary != live_summary:
            _fail(FAIL_REPORT_INVALID, f"actor {purpose_id} live summary mismatch")
        cleanup = actor["cleanup"]
        if cleanup != {
            "stopped_host_pid_zero": True,
            "container_removed": True,
            "container_and_descendants_absent": True,
        }:
            _fail(FAIL_REPORT_INVALID, f"actor {purpose_id} cleanup evidence failed")
    recomputed_cross = _validate_cross_actor(actors)
    if report["cross_actor_checks"] != recomputed_cross:
        _fail(FAIL_REPORT_INVALID, "cross-actor check summary mismatch")

    negative = _require_exact_keys(
        report["default_seccomp_negative_control"],
        {
            "default_profile_allowed_probes",
            "default_profile_is_insufficient",
            "default_profile_rejected_by_qualifier",
            "container_id",
            "cleanup",
        },
        "default seccomp negative control",
    )
    required_default_allowed = {
        "socket(AF_INET, SOCK_STREAM)",
        "socket(AF_INET6, SOCK_STREAM)",
        "ptrace(PTRACE_TRACEME)",
    }
    if (
        type(negative["default_profile_allowed_probes"]) is not list
        or not required_default_allowed.issubset(set(negative["default_profile_allowed_probes"]))
        or negative["default_profile_is_insufficient"] is not True
        or negative["default_profile_rejected_by_qualifier"] is not True
        or not _is_lower_hex(negative["container_id"], 64)
        or negative["cleanup"] != {
            "stopped_host_pid_zero": True,
            "container_removed": True,
            "container_and_descendants_absent": True,
        }
    ):
        _fail(FAIL_REPORT_INVALID, "default seccomp negative control is invalid")

    faults = _require_exact_keys(
        report["fault_injection_checks"],
        {
            "python_rust_mismatch_rejected",
            "duplicate_purpose_rejected",
            "cross_purpose_output_replay_rejected",
        },
        "fault injection checks",
    )
    if not all(value is True for value in faults.values()):
        _fail(FAIL_REPORT_INVALID, "fault-injection controls did not all reject")
    if faults != _fault_injection_checks(actors):
        _fail(FAIL_REPORT_INVALID, "fault-injection summary mismatch")

    if not _is_exact_technical_actor_disclosure(report["authority_disclosure"]):
        _fail(FAIL_REPORT_INVALID, "authority disclosure is invalid")
    if report["ceremony_outputs"] != {
        "split_seed_generated": False,
        "ephemeral_signing_keys_generated": False,
        "formal_roots_generated": False,
        "formal_gate_delta": 0,
        "formal_m3_state": "NOT_RUN",
    }:
        _fail(FAIL_REPORT_INVALID, "qualification improperly contains ceremony outputs")
    if report["all_live_checks_passed"] is not True:
        _fail(FAIL_REPORT_INVALID, "live qualification is not fully passing")
    eligible = report["technical_actor_eligible"]
    if type(eligible) is not bool or eligible is not basis_bound:
        _fail(FAIL_REPORT_INVALID, "technical actor eligibility is inconsistent")
    expected_reason = None if eligible else "EXECUTION_BASIS_COMMIT_DOES_NOT_CONTAIN_ALL_INPUTS"
    if report["ineligibility_reason_or_null"] != expected_reason:
        _fail(FAIL_REPORT_INVALID, "ineligibility reason is inconsistent")
    return deepcopy(report)


def run_live_qualification() -> dict[str, object]:
    """Run the complete offline live technical-actor qualification.

    The report is fail-closed.  ``technical_actor_eligible`` can be true only
    when the exact input files already exist in the reported basis commit.
    This permits the normal two-commit workflow: commit implementation inputs,
    run this function, then commit the generated qualification report.
    """

    try:
        temporary_owner = LinuxLocalTemporaryDirectoryV1(
            prefix="hegel-container-actors-",
            repository_root=REPOSITORY_ROOT,
        )
    except Phase3LocalRuntimeError as exc:
        _fail(FAIL_LOCAL_RUNTIME, f"{exc.code}: {exc.detail}")
    with temporary_owner as temporary:
        temporary_root = Path(temporary)
        try:
            control_plane = prepare_local_docker_control_plane_v1(
                temporary_root,
                repository_root=REPOSITORY_ROOT,
            )
        except Phase3LocalRuntimeError as exc:
            _fail(FAIL_LOCAL_RUNTIME, f"{exc.code}: {exc.detail}")
        profile = _load_profile()
        version, raw_version = _docker_version(control_plane)
        if version["server_os"] != "linux" or version["server_arch"] != "amd64":
            _fail(FAIL_DOCKER_UNAVAILABLE, "Docker server must be linux/amd64")
        raw_info_result = _run(
            control_plane.command("info", "--format", "{{json .}}"),
            environment=control_plane.environment,
        )
        try:
            raw_info = json.loads(raw_info_result.stdout)
            control_plane_receipt = build_local_docker_daemon_identity_receipt_v1(
                control_plane,
                version_payload=raw_version,
                info_payload=raw_info,
                repository_root=REPOSITORY_ROOT,
            )
        except (json.JSONDecodeError, Phase3LocalRuntimeError) as exc:
            _fail(FAIL_DOCKER_UNAVAILABLE, f"Docker daemon identity failed: {exc}")
        basis_commit_result = _git_run(["rev-parse", "HEAD"])
        basis_commit = basis_commit_result.stdout.decode("ascii", "strict").strip()
        inputs = {
            "profile": _file_binding(PROFILE_PATH, basis_commit),
            "seccomp": _file_binding(SECCOMP_PATH, basis_commit),
            "python_probe": _file_binding(PYTHON_PROBE_PATH, basis_commit),
            "rust_probe": _file_binding(RUST_PROBE_PATH, basis_commit),
            "supervisor_runtime": _file_binding(RUNTIME_PATH, basis_commit),
        }
        basis_commit_contains_all_inputs = all(
            bool(row["basis_commit_matches"]) for row in inputs.values()
        )
        images = profile.get("images")
        if type(images) is not dict or set(images) != set(PURPOSE_IMAGES.values()):
            _fail(FAIL_INPUT_BINDING, "profile image role set is invalid")
        image_bindings = {
            role: _inspect_local_image(str(image_ref), control_plane)
            for role, image_ref in sorted(images.items())
        }
        rust_binary, rust_build = _compile_rust_probe_offline(
            rust_image=str(images["rust_attester"]),
            temporary_root=temporary_root,
            control_plane=control_plane,
        )
        snapshots = {
            purpose_id: _snapshot_actor_inputs(purpose_id, temporary_root, rust_binary)
            for purpose_id in PURPOSE_ROLES
        }
        actor_reports: list[dict[str, object]] = []
        try:
            for purpose_id, role in PURPOSE_ROLES.items():
                image_role = PURPOSE_IMAGES[purpose_id]
                actor_reports.append(
                    _run_actor(
                        purpose_id=purpose_id,
                        role=role,
                        image_ref=str(images[image_role]),
                        input_snapshot=snapshots[purpose_id],
                        custom_seccomp=True,
                        control_plane=control_plane,
                    )
                )
            cross_actor = _validate_cross_actor(actor_reports)
            faults = _fault_injection_checks(actor_reports)
        finally:
            for actor in actor_reports:
                if actor.get("cleanup") is None:
                    actor["cleanup"] = _remove_container(
                        str(actor["container_id"]), control_plane
                    )
        negative = _run_negative_control(
            python_image=str(images["python_attester"]),
            input_snapshot=snapshots[2],
            control_plane=control_plane,
        )

    all_live_checks_passed = (
        len(actor_reports) == 4
        and cross_actor["technical_role_independence"] is True
        and negative["default_profile_is_insufficient"] is True
        and all(faults.values())
    )
    eligible = basis_commit_contains_all_inputs and all_live_checks_passed
    report: dict[str, object] = {
        "schema": SCHEMA,
        "profile_id": PROFILE_ID,
        "authority_class": AUTHORITY_CLASS,
        "basis_commit": basis_commit,
        "basis_commit_contains_all_inputs": basis_commit_contains_all_inputs,
        "input_bindings": inputs,
        "docker_runtime": version,
        "docker_control_plane_receipt": control_plane_receipt,
        "network_and_registry_policy": {
            "registry_access_performed": False,
            "image_pull_performed": False,
            "image_build_performed": False,
            "runtime_network_enabled": False,
            "pull_policy": "never",
            "runtime_network": "none",
        },
        "image_bindings": image_bindings,
        "rust_probe_offline_build": rust_build,
        "actor_reports": actor_reports,
        "cross_actor_checks": cross_actor,
        "default_seccomp_negative_control": negative,
        "fault_injection_checks": faults,
        "authority_disclosure": dict(TECHNICAL_ACTOR_DISCLOSURE_V1),
        "ceremony_outputs": {
            "split_seed_generated": False,
            "ephemeral_signing_keys_generated": False,
            "formal_roots_generated": False,
            "formal_gate_delta": 0,
            "formal_m3_state": "NOT_RUN",
        },
        "all_live_checks_passed": all_live_checks_passed,
        "technical_actor_eligible": eligible,
        "ineligibility_reason_or_null": (
            None if eligible else "EXECUTION_BASIS_COMMIT_DOES_NOT_CONTAIN_ALL_INPUTS"
        ),
    }
    report["qualification_payload_sha256"] = _canonical_sha256(report)
    return report


__all__ = [
    "AUTHORITY_CLASS",
    "ContainerActorQualificationError",
    "FAIL_IMPLEMENTATION_MISMATCH",
    "FAIL_LIVE_PROBE",
    "FAIL_PURPOSE_SEPARATION",
    "FAIL_REPORT_INVALID",
    "PROFILE_ID",
    "SCHEMA",
    "run_live_qualification",
    "validate_qualification_report",
]
