#!/usr/bin/env python3
"""Commit-bound, offline dual strict qualification for shrink step 3.

The supervisor is an evidence generator, not a recognizer.  It extracts an
exact Git snapshot, feeds the sealed 36-vector wires to two separately
executed recognizers, and fail-closes unless their normalized outcomes and
the inherited 2,160-source capacity replay agree exactly.  It never creates
formal roots or advances M3 out of ``NOT_RUN``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha1, sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import subprocess
import sys
import tarfile
import tempfile
from types import ModuleType
from typing import Final, Mapping, NoReturn, Sequence


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT: Final = PROJECT_ROOT.parent
MODULE_ROOT: Final = PROJECT_ROOT / "src/hegel_machine"

# Load only the sealed evidence-generator dependency closure.  Do not execute
# the package initializer, which exposes unrelated target and split APIs.
if "hegel_machine" not in sys.modules:
    package = ModuleType("hegel_machine")
    package.__path__ = [str(MODULE_ROOT)]  # type: ignore[attr-defined]
    package.__package__ = "hegel_machine"
    sys.modules["hegel_machine"] = package

from hegel_machine.phase3_shrink3_golden_vectors_v1 import (  # noqa: E402
    ACCEPT_PARENT_IDENTITY,
    STRICT_GOLDEN_VECTORS_V1,
    accepted_outcome_bytes,
    rejected_outcome_bytes,
    strict_golden_manifest_root_v1,
    strict_golden_outcome_root_v1,
)


SCHEMA: Final = "hegel-shrink3-sealed-dual-strict-qualification/1"
CLAIM_LEVEL: Final = "NON_FORMAL_DUAL_STRICT_QUALIFICATION"
STATUS_PASS: Final = "SEALED_DUAL_STRICT_OUTCOME_REPLAY_PASS"
SOURCE_SET_DOMAIN: Final = b"HEGEL/SHRINK3/DUAL_STRICT_SOURCE_SET/V1"
REPORT_DOMAIN: Final = b"HEGEL/SHRINK3/DUAL_STRICT_REPORT/V1"
DAEMON_RECEIPT_DOMAIN: Final = b"HEGEL/SHRINK3/DOCKER_DAEMON_RECEIPT/V1"
AST_HASH_DOMAIN: Final = b"HEGEL/AST/V1"
EXPECTED_MANIFEST_ROOT: Final = (
    "sha256:e091e08f33be8bbfa579b6d333f618326b4ed2ebae6d2830d3adc0df7a6333b5"
)
EXPECTED_OUTCOME_ROOT: Final = (
    "sha256:b37fcb96c78d53f7da3271513e0cae128ab7e2538288b8aa723254a0f98fde74"
)
EXPECTED_CAPACITY_COMMITMENT: Final = (
    "sha256:9045e4ebe6416dcbf699e7972f25468aef45c0f0aec0e58806061b7ce64d790e"
)
PYTHON_IMAGE: Final = (
    "python@sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3"
)
RUST_IMAGE: Final = (
    "rust@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89"
)
RUST_TOOLCHAIN_BIN: Final = (
    "/usr/local/rustup/toolchains/1.88.0-x86_64-unknown-linux-gnu/bin"
)
DEFAULT_CARGO_REGISTRY: Final = Path(
    "/home/erzhu419/.local/state/hegel-machine/rust-cargo-cache/registry"
)
DOCKER_EXECUTABLE: Final = "/usr/bin/docker"
DOCKER_HOST_ARGUMENT: Final = "--host=unix:///var/run/docker.sock"
DOCKER_SOCKET: Final = Path("/var/run/docker.sock")
RUNTIME_SECCOMP_PATH: Final = (
    "Hegel Machine/config/phase3_internal_actor_seccomp_v1.json"
)
BUILD_SECCOMP_PATH: Final = (
    "Hegel Machine/config/phase3_m3_offline_build_seccomp_v1.json"
)
BUILD_PROFILE_PATH: Final = (
    "Hegel Machine/config/phase3_shrink3_offline_build_profile_v1.json"
)
RUNTIME_TMPFS: Final = (
    "/tmp:rw,noexec,nosuid,nodev,size=64m,uid=65534,gid=65534,mode=0700"
)
BUILD_TMPFS: Final = "/tmp:rw,noexec,nosuid,nodev,size=64m,uid=0,gid=0,mode=0700"
_DOCKER_ENV: dict[str, str] | None = None

FAIL_ARGUMENT = "FAIL_SHRINK3_DUAL_STRICT_ARGUMENT"
FAIL_GIT_BINDING = "FAIL_SHRINK3_DUAL_STRICT_GIT_BINDING"
FAIL_ARCHIVE = "FAIL_SHRINK3_DUAL_STRICT_ARCHIVE"
FAIL_DOCKER_POLICY = "FAIL_SHRINK3_DUAL_STRICT_DOCKER_POLICY"
FAIL_BUILD = "FAIL_SHRINK3_DUAL_STRICT_RUST_BUILD"
FAIL_ENDPOINT = "FAIL_SHRINK3_DUAL_STRICT_ENDPOINT"
FAIL_VECTOR = "FAIL_SHRINK3_DUAL_STRICT_VECTOR"
FAIL_CAPACITY = "FAIL_SHRINK3_DUAL_STRICT_CAPACITY"
FAIL_GUARD = "FAIL_SHRINK3_DUAL_STRICT_AUTHORITY_GUARD"
FAIL_CLEANUP = "FAIL_SHRINK3_DUAL_STRICT_CLEANUP"
FAIL_INTERNAL = "FAIL_SHRINK3_DUAL_STRICT_INTERNAL"

PYTHON_SOURCE_PATHS: Final = (
    "Hegel Machine/src/hegel_machine/hashing.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_dsl_core_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_shrink1_core_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_shrink2_core_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_shrink3_core_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_shrink2_capacity_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_shrink3_capacity_entrypoint_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_shrink3_capacity_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_shrink3_golden_vectors_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_shrink3_strict_entrypoint_v1.py",
    "Hegel Machine/src/hegel_machine/strict_ast_shrink1_v1.py",
    "Hegel Machine/src/hegel_machine/strict_ast_shrink2_v1.py",
    "Hegel Machine/src/hegel_machine/strict_ast_shrink3_v1.py",
    "Hegel Machine/src/hegel_machine/strict_ast_v1.py",
    "Hegel Machine/src/hegel_machine/strict_cbor_v1.py",
)
RUST_SOURCE_DIRS: Final = (
    "Hegel Machine/rust/strict_canonicalizer",
    "Hegel Machine/rust/strict_canonicalizer_shrink1",
    "Hegel Machine/rust/strict_canonicalizer_shrink2",
    "Hegel Machine/rust/strict_canonicalizer_shrink3",
)
SUPERVISOR_PATH: Final = (
    "Hegel Machine/tools/phase3_shrink3_dual_strict_qualification_v1.py"
)
PROFILE_PATH: Final = "Hegel Machine/config/phase3_container_actor_profile_v1.json"
ARCHIVE_PATHS: Final = (
    SUPERVISOR_PATH,
    PROFILE_PATH,
    BUILD_PROFILE_PATH,
    RUNTIME_SECCOMP_PATH,
    BUILD_SECCOMP_PATH,
    *PYTHON_SOURCE_PATHS,
    *RUST_SOURCE_DIRS,
)

CAPACITY_EXCLUDED_FIELDS: Final = frozenset(
    {"implementation", "loaded_hegel_modules"}
)
ACCEPT_COMPARISON_FIELDS: Final = (
    "status",
    "canonical_cbor_hex",
    "canonical_ast_hash",
    "root_operator_id",
    "output_sort",
    "depth",
    "node_count",
)
EXPECTED_PYTHON_STRICT_MODULES: Final = [
    "hegel_machine.phase3_m3_dsl_core_v1",
    "hegel_machine.phase3_m3_shrink1_core_v1",
    "hegel_machine.phase3_m3_shrink2_core_v1",
    "hegel_machine.phase3_m3_shrink3_core_v1",
    "hegel_machine.strict_ast_shrink1_v1",
    "hegel_machine.strict_ast_shrink2_v1",
    "hegel_machine.strict_ast_shrink3_v1",
    "hegel_machine.strict_ast_v1",
    "hegel_machine.strict_cbor_v1",
]


class QualificationError(RuntimeError):
    """Stable fail-closed supervisor error."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise QualificationError(code, detail)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _docker_prefix() -> list[str]:
    return [DOCKER_EXECUTABLE, DOCKER_HOST_ARGUMENT]


def _docker_environment() -> dict[str, str]:
    if _DOCKER_ENV is None:
        _fail(FAIL_DOCKER_POLICY, "private Docker control environment is not initialized")
    return _DOCKER_ENV


def _cleanup_timed_out_container(command: Sequence[str]) -> str | None:
    if not command or command[0] != DOCKER_EXECUTABLE or "run" not in command:
        return None
    try:
        marker = command.index("--name")
        name = command[marker + 1]
    except (ValueError, IndexError):
        return "Docker run timeout had no exact container name"
    if re.fullmatch(r"hegel-s3-[0-9a-f]{16}", name) is None:
        return "Docker run timeout carried an invalid cleanup name"
    try:
        cleanup = subprocess.run(
            [*_docker_prefix(), "rm", "-f", name],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_docker_environment(),
            timeout=30,
            check=False,
        )
        inspect = subprocess.run(
            [*_docker_prefix(), "container", "inspect", name],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_docker_environment(),
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"container cleanup transport failed: {type(error).__name__}: {error}"
    if inspect.returncode == 1:
        return None
    return (
        f"container cleanup failed: rm={cleanup.returncode}, "
        f"inspect={inspect.returncode}"
    )


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: int = 120,
    allowed_codes: frozenset[int] = frozenset({0}),
    code: str = FAIL_ENDPOINT,
) -> subprocess.CompletedProcess[bytes]:
    environment = None
    if command and command[0] == DOCKER_EXECUTABLE:
        environment = _docker_environment()
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        cleanup_detail = _cleanup_timed_out_container(command)
        suffix = "" if cleanup_detail is None else f"; {cleanup_detail}"
        _fail(code, f"command timed out after {timeout}s{suffix}")
    except OSError as error:
        _fail(code, f"command transport failed: {error}")
    if result.returncode not in allowed_codes:
        detail = result.stderr.decode("utf-8", "replace").strip()
        _fail(code, f"exit {result.returncode}: {detail[:1000]}")
    return result


def _one_json(result: subprocess.CompletedProcess[bytes], label: str) -> dict[str, object]:
    try:
        value = json.loads(result.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(FAIL_ENDPOINT, f"{label} did not emit one JSON object: {error}")
    if type(value) is not dict:
        _fail(FAIL_ENDPOINT, f"{label} emitted non-object JSON")
    return value


def _git(*arguments: str, binary: bool = False) -> bytes | str:
    result = _run(
        ("git", *arguments),
        cwd=REPOSITORY_ROOT,
        code=FAIL_GIT_BINDING,
    )
    return result.stdout if binary else result.stdout.decode("utf-8").strip()


def _git_blob_oid(payload: bytes) -> str:
    return sha1(b"blob " + str(len(payload)).encode("ascii") + b"\x00" + payload).hexdigest()


def source_file_set_root_v1(rows: Sequence[Mapping[str, object]]) -> str:
    digest = sha256()
    digest.update(SOURCE_SET_DOMAIN + b"\x00")
    previous = ""
    for row in rows:
        path = str(row["path"])
        if path <= previous:
            _fail(FAIL_GIT_BINDING, "source-file rows are not strictly path ordered")
        previous = path
        try:
            fields = (
                path.encode("utf-8"),
                str(row["mode"]).encode("ascii"),
                bytes.fromhex(str(row["git_blob_oid"])),
                bytes.fromhex(str(row["sha256"])),
                int(row["size"]).to_bytes(8, "big"),
            )
        except (ValueError, OverflowError) as error:
            _fail(FAIL_GIT_BINDING, f"invalid source-file row {path}: {error}")
        if len(fields[2]) != 20 or len(fields[3]) != 32:
            _fail(FAIL_GIT_BINDING, f"invalid source-file digest width: {path}")
        for field in fields:
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return "sha256:" + digest.hexdigest()


def _source_rows(basis_commit: str) -> tuple[list[dict[str, object]], str]:
    listing = _git("ls-tree", "-r", basis_commit, "--", *ARCHIVE_PATHS)
    if type(listing) is not str:
        _fail(FAIL_GIT_BINDING, "Git tree listing is not text")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for line in listing.splitlines():
        metadata, path = line.split("\t", 1)
        mode, kind, oid = metadata.split(" ")
        if kind != "blob" or path in seen:
            _fail(FAIL_GIT_BINDING, f"invalid source tree row for {path}")
        payload = _git("show", f"{basis_commit}:{path}", binary=True)
        if type(payload) is not bytes:
            _fail(FAIL_GIT_BINDING, f"Git blob is not bytes: {path}")
        worktree_path = REPOSITORY_ROOT / path
        if not worktree_path.is_file() or worktree_path.read_bytes() != payload:
            _fail(FAIL_GIT_BINDING, f"worktree source differs from {basis_commit}: {path}")
        if _git_blob_oid(payload) != oid:
            _fail(FAIL_GIT_BINDING, f"Git blob identity mismatch: {path}")
        seen.add(path)
        rows.append(
            {
                "path": path,
                "mode": mode,
                "git_blob_oid": oid,
                "sha256": sha256(payload).hexdigest(),
                "size": len(payload),
            }
        )
    required_files = set(PYTHON_SOURCE_PATHS) | {
        SUPERVISOR_PATH,
        PROFILE_PATH,
        BUILD_PROFILE_PATH,
        RUNTIME_SECCOMP_PATH,
        BUILD_SECCOMP_PATH,
    }
    if not required_files.issubset(seen):
        _fail(FAIL_GIT_BINDING, f"missing required source blobs: {sorted(required_files - seen)}")
    rows.sort(key=lambda row: str(row["path"]))
    return rows, source_file_set_root_v1(rows)


def _safe_extract_git_archive(payload: bytes, destination: Path) -> None:
    try:
        with tarfile.open(fileobj=BytesIO(payload), mode="r:") as archive:
            members = archive.getmembers()
            for member in members:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts:
                    _fail(FAIL_ARCHIVE, f"unsafe archive path: {member.name}")
                if not (member.isdir() or member.isfile()):
                    _fail(FAIL_ARCHIVE, f"non-regular archive member: {member.name}")
            archive.extractall(destination)
    except (tarfile.TarError, OSError) as error:
        _fail(FAIL_ARCHIVE, f"archive extraction failed: {error}")


def _validate_snapshot(
    snapshot_root: Path, source_rows: Sequence[Mapping[str, object]]
) -> None:
    for row in source_rows:
        path = snapshot_root / str(row["path"])
        try:
            payload = path.read_bytes()
        except OSError as error:
            _fail(FAIL_ARCHIVE, f"snapshot source unreadable: {path}: {error}")
        if (
            len(payload) != row["size"]
            or sha256(payload).hexdigest() != row["sha256"]
            or _git_blob_oid(payload) != row["git_blob_oid"]
        ):
            _fail(FAIL_ARCHIVE, f"snapshot bytes differ from Git blob: {row['path']}")


def _initialize_docker_environment(control_root: Path) -> dict[str, object]:
    global _DOCKER_ENV
    if Path(DOCKER_EXECUTABLE).resolve() != Path("/usr/bin/docker") or not os.access(
        DOCKER_EXECUTABLE, os.X_OK
    ):
        _fail(FAIL_DOCKER_POLICY, "the exact /usr/bin/docker executable is unavailable")
    try:
        socket_stat = DOCKER_SOCKET.stat()
    except OSError as error:
        _fail(FAIL_DOCKER_POLICY, f"Docker socket is unavailable: {error}")
    if not stat.S_ISSOCK(socket_stat.st_mode):
        _fail(FAIL_DOCKER_POLICY, "Docker control endpoint is not a Unix socket")
    config = control_root / "docker-config"
    home = control_root / "home"
    config.mkdir(mode=0o700)
    home.mkdir(mode=0o700)
    config_file = config / "config.json"
    config_file.write_bytes(b"{}\n")
    config_file.chmod(0o600)
    _DOCKER_ENV = {
        "DOCKER_CONFIG": str(config),
        "DOCKER_HOST": "unix:///var/run/docker.sock",
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
    }
    server = _one_json(
        _run(
            (*_docker_prefix(), "version", "--format", "{{json .Server}}"),
            code=FAIL_DOCKER_POLICY,
        ),
        "Docker server version",
    )
    info_template = (
        "{\"id\":{{json .ID}},\"name\":{{json .Name}},"
        "\"driver\":{{json .Driver}},\"operating_system\":{{json .OperatingSystem}},"
        "\"os_type\":{{json .OSType}},\"architecture\":{{json .Architecture}},"
        "\"docker_root_dir\":{{json .DockerRootDir}},"
        "\"security_options\":{{json .SecurityOptions}}}"
    )
    daemon = _one_json(
        _run(
            (*_docker_prefix(), "info", "--format", info_template),
            code=FAIL_DOCKER_POLICY,
        ),
        "Docker daemon identity",
    )
    if (
        daemon.get("os_type") != "linux"
        or type(daemon.get("id")) is not str
        or not daemon["id"]
        or type(daemon.get("driver")) is not str
        or not daemon["driver"]
    ):
        _fail(FAIL_DOCKER_POLICY, "live Docker daemon is not the required local Linux runtime")
    receipt: dict[str, object] = {
        "schema_version": "hegel-shrink3-local-docker-daemon-receipt/1",
        "docker_executable": DOCKER_EXECUTABLE,
        "explicit_host_argument": DOCKER_HOST_ARGUMENT,
        "socket": str(DOCKER_SOCKET),
        "socket_device": socket_stat.st_dev,
        "socket_inode": socket_stat.st_ino,
        "socket_uid": socket_stat.st_uid,
        "socket_gid": socket_stat.st_gid,
        "private_empty_client_config_sha256": sha256(b"{}\n").hexdigest(),
        "host_environment_keys": sorted(_DOCKER_ENV),
        "server": server,
        "daemon": daemon,
    }
    receipt["diagnostic_receipt_hash"] = "sha256:" + sha256(
        DAEMON_RECEIPT_DOMAIN + b"\x00" + _canonical_json_bytes(receipt)
    ).hexdigest()
    return receipt


def _profile_images(snapshot_project: Path) -> dict[str, object]:
    try:
        actor_payload = (
            snapshot_project / "config/phase3_container_actor_profile_v1.json"
        ).read_bytes()
        build_payload = (
            snapshot_project / "config/phase3_shrink3_offline_build_profile_v1.json"
        ).read_bytes()
        profile = json.loads(actor_payload)
        build_profile = json.loads(build_payload)
        images = profile["images"]
    except (OSError, KeyError, json.JSONDecodeError, TypeError) as error:
        _fail(FAIL_DOCKER_POLICY, f"container profile unreadable: {error}")
    if type(profile) is not dict or type(build_profile) is not dict:
        _fail(FAIL_DOCKER_POLICY, "container/build profile is not an object")
    if type(images) is not dict:
        _fail(FAIL_DOCKER_POLICY, "container profile images field is not an object")
    if images.get("python_attester") != PYTHON_IMAGE or images.get("rust_attester") != RUST_IMAGE:
        _fail(FAIL_DOCKER_POLICY, "pinned image references differ from the committed profile")
    if profile.get("network_policy") != {
        "allow_registry_access": False,
        "allow_runtime_network": False,
        "docker_network": "none",
        "pull_policy": "never",
    }:
        _fail(FAIL_DOCKER_POLICY, "committed network policy differs")
    control = profile.get("docker_control_plane_policy")
    if type(control) is not dict or (
        control.get("executable") != DOCKER_EXECUTABLE
        or control.get("explicit_host_argument") != DOCKER_HOST_ARGUMENT
        or control.get("socket") != str(DOCKER_SOCKET)
        or control.get("client_config") != "empty-private-config-json"
        or control.get("host_environment_keys_exact") != sorted(
            ["DOCKER_CONFIG", "DOCKER_HOST", "HOME", "LANG", "LC_ALL", "PATH"]
        )
        or control.get("ambient_proxy_or_docker_variables_allowed") is not False
        or control.get("live_local_linux_daemon_identity_receipt_required") is not True
    ):
        _fail(FAIL_DOCKER_POLICY, "committed Docker control-plane policy differs")
    if profile.get("resource_limits") != {
        "memory": "512m",
        "pids": 64,
        "tmpfs": RUNTIME_TMPFS,
    }:
        _fail(FAIL_DOCKER_POLICY, "committed actor resource limits differ")
    if profile.get("required_runtime_flags") != [
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--user=65534:65534",
        "--pids-limit=64",
    ]:
        _fail(FAIL_DOCKER_POLICY, "committed required runtime flags differ")
    if (
        profile.get("seccomp_profile")
        != "config/phase3_internal_actor_seccomp_v1.json"
        or profile.get("offline_build_seccomp_profile")
        != "config/phase3_m3_offline_build_seccomp_v1.json"
    ):
        _fail(FAIL_DOCKER_POLICY, "committed seccomp profile paths differ")
    disclosure = profile.get("authority_disclosure")
    if type(disclosure) is not dict or (
        disclosure.get("same_admin_controller") is not True
        or disclosure.get("organizational_independence") is not False
        or disclosure.get("independent_human_actors") is not False
        or disclosure.get("technical_role_independence") is not True
        or disclosure.get("owner_accepted_threat_model") is not True
    ):
        _fail(FAIL_DOCKER_POLICY, "committed authority disclosure differs")
    if build_profile != {
        "profile_id": "hegel-shrink3-rust-offline-build-v1",
        "purpose": "build-only target-free shrink3 strict recognizer",
        "authority": "engineering qualification only; no formal root or state transition",
        "image": RUST_IMAGE,
        "network": "none",
        "pull_policy": "never",
        "root_filesystem_read_only": True,
        "user": "0:0",
        "capabilities_dropped": "ALL",
        "no_new_privileges": True,
        "pids_limit": 64,
        "memory": "512m",
        "memory_swap": "512m",
        "nofile_ulimit": "128:128",
        "tmpfs": BUILD_TMPFS,
        "seccomp_profile": "config/phase3_m3_offline_build_seccomp_v1.json",
        "source_mount": "read-only committed git archive snapshot",
        "cargo_registry_mount": "read-only preconfigured local cache",
        "target_mount": "fresh local-driver docker volume, build rw then runtime ro",
        "cargo_flags": ["--release", "--locked", "--offline"],
        "target_or_split_inputs_allowed": False,
        "seed_key_signature_or_formal_root_access_allowed": False,
    }:
        _fail(FAIL_DOCKER_POLICY, "committed shrink-3 build profile differs")
    return {
        "actor_profile_id": profile.get("profile_id"),
        "actor_profile_sha256": sha256(actor_payload).hexdigest(),
        "build_profile_id": build_profile["profile_id"],
        "build_profile_sha256": sha256(build_payload).hexdigest(),
    }


def _inspect_image(image: str) -> str:
    result = _run(
        (*_docker_prefix(), "image", "inspect", image, "--format", "{{.Id}}"),
        code=FAIL_DOCKER_POLICY,
    )
    observed = result.stdout.decode("ascii", "strict").strip()
    expected = image.split("@", 1)[1]
    if observed != expected:
        _fail(FAIL_DOCKER_POLICY, f"local image identity differs for {image}")
    return observed


def _docker_common(
    image: str,
    *,
    seccomp_path: Path,
    user: str | None = None,
    tmpfs: str = RUNTIME_TMPFS,
) -> list[str]:
    command = [
        *_docker_prefix(),
        "run", "--rm", "--name", f"hegel-s3-{secrets.token_hex(8)}",
        "--pull=never", "--network=none", "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--security-opt", f"seccomp={seccomp_path}",
        "--read-only", "--pids-limit=64", "--memory=512m",
        "--memory-swap=512m", "--ulimit=nofile=128:128",
        "--tmpfs", tmpfs,
        "-e", "HOME=/tmp", "-e", "LANG=C.UTF-8", "-e", "LC_ALL=C.UTF-8",
        "-e", "TZ=UTC",
    ]
    if user is not None:
        command.extend(("--user", user))
    command.append(image)
    return command


def python_runtime_command(snapshot_project: Path, arguments: Sequence[str]) -> list[str]:
    command = _docker_common(
        PYTHON_IMAGE,
        seccomp_path=snapshot_project / "config/phase3_internal_actor_seccomp_v1.json",
        user="65534:65534",
    )
    command[-1:-1] = [
        "-e", "PYTHONHASHSEED=0",
        "-v", f"{snapshot_project}:/workspace:ro",
        "-w", "/workspace",
    ]
    command.extend(
        (
            "/usr/local/bin/python3", "-I", "-S", "-B",
            "/workspace/src/hegel_machine/phase3_shrink3_strict_entrypoint_v1.py",
            *arguments,
        )
    )
    return command


def python_capacity_command(snapshot_project: Path, mode: str) -> list[str]:
    command = _docker_common(
        PYTHON_IMAGE,
        seccomp_path=snapshot_project / "config/phase3_internal_actor_seccomp_v1.json",
        user="65534:65534",
    )
    command[-1:-1] = [
        "-e", "PYTHONHASHSEED=0",
        "-v", f"{snapshot_project}:/workspace:ro",
        "-w", "/workspace",
    ]
    command.extend(
        (
            "/usr/local/bin/python3", "-I", "-S", "-B",
            "/workspace/src/hegel_machine/phase3_shrink3_capacity_entrypoint_v1.py",
            mode,
        )
    )
    return command


def rust_runtime_command(
    snapshot_project: Path, volume: str, arguments: Sequence[str]
) -> list[str]:
    command = _docker_common(
        RUST_IMAGE,
        seccomp_path=snapshot_project / "config/phase3_internal_actor_seccomp_v1.json",
        user="65534:65534",
    )
    command[-1:-1] = ["-v", f"{volume}:/cargo-target:ro"]
    command.extend(("/cargo-target/release/hegel-strict-canonicalizer-shrink3", *arguments))
    return command


def _create_fresh_volume(name: str, basis_commit: str) -> dict[str, object]:
    inspected = _run(
        (*_docker_prefix(), "volume", "inspect", name),
        allowed_codes=frozenset({0, 1}),
        code=FAIL_DOCKER_POLICY,
    )
    if inspected.returncode == 0:
        _fail(FAIL_DOCKER_POLICY, f"dedicated target volume already exists: {name}")
    if inspected.returncode != 1:
        _fail(FAIL_DOCKER_POLICY, "unable to establish target-volume absence")
    created = _run(
        (
            *_docker_prefix(), "volume", "create", "--driver", "local",
            "--label", "hegel.machine.role=shrink3-dual-strict",
            "--label", f"hegel.machine.basis={basis_commit}",
            "--label", "hegel.machine.network=none",
            name,
        ),
        code=FAIL_DOCKER_POLICY,
    ).stdout.decode("utf-8").strip()
    if created != name:
        cleanup = _run(
            (*_docker_prefix(), "volume", "rm", name),
            allowed_codes=frozenset({0, 1}),
            code=FAIL_CLEANUP,
        )
        _fail(
            FAIL_DOCKER_POLICY,
            "Docker returned a different target volume name; "
            f"cleanup_exit={cleanup.returncode}",
        )
    expected_labels = {
        "hegel.machine.role": "shrink3-dual-strict",
        "hegel.machine.basis": basis_commit,
        "hegel.machine.network": "none",
    }
    try:
        detail = _one_json(
            _run(
                (*_docker_prefix(), "volume", "inspect", name, "--format", "{{json .}}"),
                code=FAIL_DOCKER_POLICY,
            ),
            "Docker target volume",
        )
        if (
            detail.get("Name") != name
            or detail.get("Driver") != "local"
            or detail.get("Scope") != "local"
            or detail.get("Options") not in (None, {})
            or detail.get("Labels") != expected_labels
            or type(detail.get("Mountpoint")) is not str
            or not str(detail["Mountpoint"]).startswith("/")
        ):
            _fail(FAIL_DOCKER_POLICY, "fresh target volume is not exact local storage")
    except BaseException as primary:
        try:
            _remove_volume(name)
        except QualificationError as cleanup_error:
            if isinstance(primary, QualificationError):
                raise QualificationError(
                    primary.code,
                    f"{primary.detail}; secondary cleanup failure: "
                    f"{cleanup_error.code}: {cleanup_error.detail}",
                ) from primary
            raise
        raise
    return {
        "name": name,
        "driver": "local",
        "scope": "local",
        "options": None,
        "labels": expected_labels,
        "fresh_before_run": True,
    }


def _remove_volume(name: str) -> None:
    result = _run(
        (*_docker_prefix(), "volume", "rm", name),
        timeout=120,
        code=FAIL_CLEANUP,
    )
    if result.stdout.decode("utf-8").strip() != name:
        _fail(FAIL_CLEANUP, "Docker did not confirm target-volume removal")
    absent = _run(
        (*_docker_prefix(), "volume", "inspect", name),
        allowed_codes=frozenset({1}),
        code=FAIL_CLEANUP,
    )
    if absent.returncode != 1:
        _fail(FAIL_CLEANUP, "target volume still exists after removal")


def _build_rust(snapshot_project: Path, cargo_registry: Path, volume: str, workers: int) -> str:
    if not cargo_registry.is_dir():
        _fail(FAIL_BUILD, f"offline Cargo registry is absent: {cargo_registry}")
    command = _docker_common(
        RUST_IMAGE,
        seccomp_path=snapshot_project / "config/phase3_m3_offline_build_seccomp_v1.json",
        tmpfs=BUILD_TMPFS,
    )
    command[-1:-1] = [
        "-e", "CARGO_HOME=/cargo-home",
        "-e", "CARGO_NET_OFFLINE=true",
        "-e", f"CARGO_BUILD_JOBS={workers}",
        "-e", "CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1",
        "-e", "CARGO_TARGET_DIR=/cargo-target",
        "-e", f"RUSTC={RUST_TOOLCHAIN_BIN}/rustc",
        "-e", f"RUSTDOC={RUST_TOOLCHAIN_BIN}/rustdoc",
        "-v", f"{cargo_registry}:/cargo-home/registry:ro",
        "-v", f"{snapshot_project / 'rust'}:/workspace/rust:ro",
        "-v", f"{volume}:/cargo-target:rw",
        "-w", "/workspace/rust/strict_canonicalizer_shrink3",
    ]
    command.extend(
        (
            f"{RUST_TOOLCHAIN_BIN}/cargo", "build", "--release", "--locked", "--offline"
        )
    )
    _run(command, timeout=900, code=FAIL_BUILD)
    hash_result = _run(
        rust_runtime_command(snapshot_project, volume, ()),
        timeout=60,
        allowed_codes=frozenset({2}),
        code=FAIL_BUILD,
    )
    # The no-argument run proves the binary is executable but intentionally
    # returns its CLI error.  Hash it in a separate immutable-container call.
    hash_command = _docker_common(
        RUST_IMAGE,
        seccomp_path=snapshot_project / "config/phase3_internal_actor_seccomp_v1.json",
        user="65534:65534",
    )
    hash_command[-1:-1] = ["-v", f"{volume}:/cargo-target:ro"]
    hash_command.extend(("/usr/bin/sha256sum", "/cargo-target/release/hegel-strict-canonicalizer-shrink3"))
    digest = _run(hash_command, code=FAIL_BUILD).stdout.decode("ascii").split()[0]
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None or not hash_result.stderr:
        _fail(FAIL_BUILD, "Rust binary executable/hash qualification failed")
    return digest


def normalize_endpoint_report(
    report: Mapping[str, object], *, implementation: str
) -> tuple[bytes, dict[str, object]]:
    if report.get("schema_version") != "hegel-strict-canonicalizer-shrink3-replay/1":
        _fail(FAIL_VECTOR, f"{implementation} replay schema differs")
    if report.get("implementation") != implementation:
        _fail(FAIL_VECTOR, f"{implementation} identity differs")
    if report.get("dsl_version") != "hegel-old-dsl-v1.3.0" or report.get("freeze_version") != "hegel-freeze-p2b-p3-v1.3.0":
        _fail(FAIL_VECTOR, f"{implementation} DSL/freeze binding differs")
    status = report.get("status")
    if status == "REJECTED":
        error_code = report.get("error_code")
        if type(error_code) is not str or re.fullmatch(r"[A-Z0-9_]+", error_code) is None:
            _fail(FAIL_VECTOR, f"{implementation} rejection code is invalid")
        return rejected_outcome_bytes(error_code), {
            "status": "REJECTED", "error_code": error_code
        }
    if status != "ACCEPTED":
        _fail(FAIL_VECTOR, f"{implementation} disposition is invalid")
    try:
        cbor_bytes = bytes.fromhex(str(report["canonical_cbor_hex"]))
        hash_id = str(report["canonical_ast_hash"])
    except (KeyError, ValueError) as error:
        _fail(FAIL_VECTOR, f"{implementation} accepted payload is invalid: {error}")
    computed = sha256(AST_HASH_DOMAIN + b"\x00" + cbor_bytes).digest()
    if hash_id != "sha256:" + computed.hex():
        _fail(FAIL_VECTOR, f"{implementation} AST hash does not bind its CBOR")
    normalized = {field: report.get(field) for field in ACCEPT_COMPARISON_FIELDS}
    if any(normalized[field] is None for field in ACCEPT_COMPARISON_FIELDS):
        _fail(FAIL_VECTOR, f"{implementation} accepted metadata is incomplete")
    return accepted_outcome_bytes(cbor_bytes, computed), normalized


def compare_capacity_reports(
    python_report: Mapping[str, object], rust_report: Mapping[str, object]
) -> dict[str, object]:
    if python_report.get("implementation") != "python" or rust_report.get("implementation") != "rust":
        _fail(FAIL_CAPACITY, "capacity implementation identities differ")
    python_common = {
        key: value for key, value in python_report.items() if key not in CAPACITY_EXCLUDED_FIELDS
    }
    rust_common = {
        key: value for key, value in rust_report.items() if key not in CAPACITY_EXCLUDED_FIELDS
    }
    if python_common != rust_common:
        differing = sorted(
            key for key in set(python_common) | set(rust_common)
            if python_common.get(key) != rust_common.get(key)
        )
        _fail(FAIL_CAPACITY, f"capacity fields differ: {differing}")
    guards = {
        "source_candidate_count": 2160,
        "accepted_source_count": 2160,
        "accepted_unique_count": 2160,
        "parent_identity_match_count": 2160,
        "rejected_count": 0,
        "rewrite_collapsed_count": 0,
        "accepted_set_commitment": EXPECTED_CAPACITY_COMMITMENT,
        "subset_status": "SURVIVOR_SUBSET_ONLY_NOT_COMPLETE",
        "executed_closure_status": "NOT_RUN",
        "complete_closure_enumerated": False,
        "interpreted_as_complete_closure": False,
        "formal_roots": None,
        "target_or_split_modules_loaded": False,
    }
    for field, expected in guards.items():
        if python_common.get(field) != expected:
            _fail(FAIL_CAPACITY, f"capacity guard differs: {field}")
    return {
        "status": "DUAL_SURVIVOR_SUBSET_REPLAY_PASS_NOT_COMPLETE",
        "all_comparable_fields_equal": True,
        **{field: python_common[field] for field in guards},
        "comparable_report_sha256": sha256(_canonical_json_bytes(python_common)).hexdigest(),
    }


@dataclass(frozen=True)
class EndpointPair:
    vector_id: str
    python_exit: int
    python_report: dict[str, object]
    rust_exit: int
    rust_report: dict[str, object]


def _run_vector(snapshot_project: Path, volume: str, vector: object) -> EndpointPair:
    vector_id = str(vector.vector_id)
    if vector.boundary == "SOURCE_JSON":
        payload = vector.input_wire.decode("utf-8")
        python_args = ("--source-json", payload)
        rust_args = ("--ast-json", payload)
    else:
        payload = vector.input_wire.hex()
        python_args = ("--formal-cbor-hex", payload)
        rust_args = ("--decode-cbor-hex", payload)
    python_result = _run(
        python_runtime_command(snapshot_project, python_args),
        timeout=90,
        allowed_codes=frozenset({0}),
    )
    rust_result = _run(
        rust_runtime_command(snapshot_project, volume, rust_args),
        timeout=90,
        allowed_codes=frozenset({0, 1}),
    )
    python_report = _one_json(python_result, f"Python {vector_id}")
    rust_report = _one_json(rust_result, f"Rust {vector_id}")
    if python_report.get("boundary") != vector.boundary:
        _fail(FAIL_VECTOR, f"Python boundary differs at {vector_id}")
    loaded_modules = python_report.get("loaded_hegel_modules")
    if (
        loaded_modules != EXPECTED_PYTHON_STRICT_MODULES
        or python_report.get("target_or_split_modules_loaded") is not False
    ):
        _fail(FAIL_GUARD, f"Python target/split isolation differs at {vector_id}")
    if (
        rust_report.get("boundary") != vector.boundary
        or rust_report.get("target_or_split_modules_loaded") is not False
    ):
        _fail(FAIL_GUARD, f"Rust boundary/target isolation differs at {vector_id}")
    if vector.boundary == "FORMAL_CBOR" and rust_report.get("generic_cbor_parse") is not True:
        _fail(FAIL_VECTOR, f"Rust generic CBOR parse differs at {vector_id}")
    return EndpointPair(
        vector_id,
        python_result.returncode,
        python_report,
        rust_result.returncode,
        rust_report,
    )


def _golden_guard(report: Mapping[str, object], implementation: str) -> None:
    expected_schema = {
        "python": "hegel-strict-canonicalizer-shrink3-golden/2",
        "rust": "hegel-strict-canonicalizer-shrink3-golden/1",
    }[implementation]
    expected = {
        "schema_version": expected_schema,
        "implementation": implementation,
        "vector_count": 36,
        "passed_count": 36,
        "surviving_identity_checks": 8,
        "source_add_rejection_checks": 4,
        "source_priority_checks": 6,
        "formal_add_rejection_checks": 3,
        "formal_priority_checks": 6,
        "formal_shape_priority_checks": 6,
        "formal_alias_or_reserved_checks": 3,
        "execution_state": "NOT_RUN",
        "closure_executed": False,
        "formal_roots_generated": False,
        "formal_roots": None,
        "target_or_split_modules_loaded": False,
    }
    for field, value in expected.items():
        if report.get(field) != value:
            _fail(FAIL_GUARD, f"{implementation} built-in golden guard differs: {field}")
    if implementation == "python":
        if report.get("golden_vector_manifest_root") != EXPECTED_MANIFEST_ROOT or report.get("golden_outcome_root") != EXPECTED_OUTCOME_ROOT:
            _fail(FAIL_GUARD, "Python sealed built-in roots differ")


def _runtime_identity(
    snapshot_project: Path, image: str, program: Sequence[str]
) -> dict[str, object]:
    command = _docker_common(
        image,
        seccomp_path=snapshot_project / "config/phase3_internal_actor_seccomp_v1.json",
        user="65534:65534",
    )
    command.extend(program)
    return _one_json(_run(command, code=FAIL_DOCKER_POLICY), f"{image} runtime identity")


def _runtime_text(snapshot_project: Path, image: str, program: Sequence[str]) -> str:
    command = _docker_common(
        image,
        seccomp_path=snapshot_project / "config/phase3_internal_actor_seccomp_v1.json",
        user="65534:65534",
    )
    command.extend(program)
    result = _run(command, code=FAIL_DOCKER_POLICY)
    try:
        text = result.stdout.decode("utf-8", "strict").strip()
    except UnicodeDecodeError as error:
        _fail(FAIL_DOCKER_POLICY, f"{image} runtime identity is not UTF-8: {error}")
    if not text:
        _fail(FAIL_DOCKER_POLICY, f"{image} runtime identity is empty")
    return text


def _qualify(
    snapshot_project: Path,
    volume: str,
    *,
    repository_binding: dict[str, object],
    workers: int,
    cargo_registry: Path,
    python_image_id: str,
    rust_image_id: str,
    daemon_receipt: dict[str, object],
    volume_receipt: dict[str, object],
    profile_receipt: dict[str, object],
) -> dict[str, object]:
    binary_sha256 = _build_rust(snapshot_project, cargo_registry, volume, workers)

    python_identity = _runtime_identity(
        snapshot_project,
        PYTHON_IMAGE,
        (
            "/usr/local/bin/python3", "-I", "-S", "-B", "-c",
            "import hashlib,json,pathlib,sys;p=pathlib.Path(sys.executable).resolve();print(json.dumps({'executable':str(p),'executable_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'version':sys.version},sort_keys=True,separators=(',',':')))",
        ),
    )
    rust_identity = {
        "rustc_version_verbose": _runtime_text(
            snapshot_project,
            RUST_IMAGE,
            (f"{RUST_TOOLCHAIN_BIN}/rustc", "--version", "--verbose"),
        )
    }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pairs = list(
            pool.map(
                lambda vector: _run_vector(snapshot_project, volume, vector),
                STRICT_GOLDEN_VECTORS_V1,
            )
        )
    pair_by_id = {pair.vector_id: pair for pair in pairs}
    if len(pair_by_id) != 36:
        _fail(FAIL_VECTOR, "dual endpoint result IDs are not unique")

    python_outcomes: dict[str, bytes] = {}
    rust_outcomes: dict[str, bytes] = {}
    vector_rows: list[dict[str, object]] = []
    for vector in STRICT_GOLDEN_VECTORS_V1:
        pair = pair_by_id[vector.vector_id]
        python_bytes, python_normalized = normalize_endpoint_report(
            pair.python_report, implementation="python"
        )
        rust_bytes, rust_normalized = normalize_endpoint_report(
            pair.rust_report, implementation="rust"
        )
        expected = vector.expected_disposition
        observed = (
            ACCEPT_PARENT_IDENTITY
            if python_normalized["status"] == "ACCEPTED"
            else python_normalized["error_code"]
        )
        if observed != expected or python_normalized != rust_normalized or python_bytes != rust_bytes:
            _fail(FAIL_VECTOR, f"dual normalized outcome differs at {vector.vector_id}")
        if pair.rust_exit != (0 if observed == ACCEPT_PARENT_IDENTITY else 1):
            _fail(FAIL_VECTOR, f"Rust exit/disposition differs at {vector.vector_id}")
        python_outcomes[vector.vector_id] = python_bytes
        rust_outcomes[vector.vector_id] = rust_bytes
        vector_rows.append(
            {
                "vector_id": vector.vector_id,
                "category": vector.category,
                "boundary": vector.boundary,
                "input_wire_sha256": sha256(vector.input_wire).hexdigest(),
                "input_wire_size": len(vector.input_wire),
                "expected_disposition": expected,
                "python_exit": pair.python_exit,
                "rust_exit": pair.rust_exit,
                "normalized": python_normalized,
                "normalized_outcome_sha256": sha256(python_bytes).hexdigest(),
                "dual_equal": True,
            }
        )

    python_root = strict_golden_outcome_root_v1(python_outcomes)
    rust_root = strict_golden_outcome_root_v1(rust_outcomes)
    manifest_root = strict_golden_manifest_root_v1()
    if manifest_root != EXPECTED_MANIFEST_ROOT or python_root != EXPECTED_OUTCOME_ROOT or rust_root != EXPECTED_OUTCOME_ROOT:
        _fail(FAIL_VECTOR, "sealed manifest/outcome root differs")

    python_golden_result = _run(
        python_capacity_command(snapshot_project, "--golden-replay"), timeout=180
    )
    rust_golden_result = _run(
        rust_runtime_command(snapshot_project, volume, ("--golden-replay",)),
        timeout=180,
    )
    python_golden = _one_json(python_golden_result, "Python built-in golden")
    rust_golden = _one_json(rust_golden_result, "Rust built-in golden")
    _golden_guard(python_golden, "python")
    _golden_guard(rust_golden, "rust")

    with ThreadPoolExecutor(max_workers=2) as pool:
        python_future = pool.submit(
            _run, python_capacity_command(snapshot_project, "--capacity-replay"), timeout=300
        )
        rust_future = pool.submit(
            _run,
            rust_runtime_command(snapshot_project, volume, ("--capacity-replay",)),
            timeout=300,
        )
        python_capacity = _one_json(python_future.result(), "Python capacity")
        rust_capacity = _one_json(rust_future.result(), "Rust capacity")
    capacity = compare_capacity_reports(python_capacity, rust_capacity)

    report: dict[str, object] = {
        "schema_version": SCHEMA,
        "artifact_kind": "COMMIT_BOUND_ENGINEERING_QUALIFICATION_EVIDENCE",
        "status": STATUS_PASS,
        "claim_level": CLAIM_LEVEL,
        "repository_binding": repository_binding,
        "sealed_basis": {
            "golden_vector_manifest_root": manifest_root,
            "expected_outcome_root": EXPECTED_OUTCOME_ROOT,
            "ordered_vector_ids": [vector.vector_id for vector in STRICT_GOLDEN_VECTORS_V1],
            "vector_count": 36,
        },
        "runtime_isolation": {
            "role_topology": "HOST_SUPERVISOR_PLUS_TWO_DISJOINT_PINNED_CONTAINERS",
            "same_admin_controller": True,
            "organizational_independence": False,
            "independent_human_actors": False,
            "technical_role_independence": True,
            "owner_accepted_threat_model": True,
            "docker_daemon_identity_receipt": daemon_receipt,
            "committed_profile_receipt": profile_receipt,
            "python_image_ref": PYTHON_IMAGE,
            "python_image_id": python_image_id,
            "rust_image_ref": RUST_IMAGE,
            "rust_image_id": rust_image_id,
            "pull_policy": "never",
            "network_mode": "none",
            "capabilities_dropped": "ALL",
            "no_new_privileges": True,
            "container_root_filesystem_read_only": True,
            "source_snapshot_mount_read_only": True,
            "fresh_ephemeral_rust_target_volume": True,
            "rust_target_volume_removed_after_run": True,
            "rust_target_volume_receipt": volume_receipt,
            "cargo_locked": True,
            "cargo_offline": True,
            "rust_build_user": "0:0",
            "rust_build_tmpfs": BUILD_TMPFS,
            "recognizer_runtime_user": "65534:65534",
            "recognizer_runtime_tmpfs": RUNTIME_TMPFS,
            "memory_limit": "512m",
            "memory_swap_limit": "512m",
            "pids_limit": 64,
            "python_flags": ["-I", "-S", "-B"],
            "worker_count": workers,
            "python_runtime": python_identity,
            "rust_runtime": rust_identity,
            "rust_binary_sha256": binary_sha256,
        },
        "dual_vector_replay": {
            "status": STATUS_PASS,
            "vector_count": 36,
            "python_outcome_root": python_root,
            "rust_outcome_root": rust_root,
            "all_normalized_outcomes_equal": True,
            "vectors": vector_rows,
        },
        "built_in_replay_controls": {
            "python_report_sha256": sha256(_canonical_json_bytes(python_golden)).hexdigest(),
            "rust_report_sha256": sha256(_canonical_json_bytes(rust_golden)).hexdigest(),
            "python_passed_count": 36,
            "rust_passed_count": 36,
        },
        "dual_capacity_replay": capacity,
        "authority_guards": {
            "execution_state": "NOT_RUN",
            "closure_executed": False,
            "formal_roots_generated": False,
            "formal_roots": None,
            "certificate_issued": False,
            "signature_generated": False,
            "seed_generated": False,
            "target_roles_evaluated": False,
            "active_governance_changed": False,
            "formal_state_transition_allowed": False,
        },
    }
    report["diagnostic_report_hash"] = "sha256:" + sha256(
        REPORT_DOMAIN + b"\x00" + _canonical_json_bytes(report)
    ).hexdigest()
    return report


def _mapping(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        _fail(FAIL_GUARD, f"{label} is not an object")
    return value


def validate_qualification_report(report_value: object) -> None:
    """Recompute every portable receipt invariant before publication."""

    report = _mapping(report_value, "qualification report")
    if set(report) != {
        "schema_version",
        "artifact_kind",
        "status",
        "claim_level",
        "repository_binding",
        "sealed_basis",
        "runtime_isolation",
        "dual_vector_replay",
        "built_in_replay_controls",
        "dual_capacity_replay",
        "authority_guards",
        "diagnostic_report_hash",
    }:
        _fail(FAIL_GUARD, "qualification report fields differ")
    if (
        report["schema_version"] != SCHEMA
        or report["status"] != STATUS_PASS
        or report["claim_level"] != CLAIM_LEVEL
        or report["artifact_kind"]
        != "COMMIT_BOUND_ENGINEERING_QUALIFICATION_EVIDENCE"
    ):
        _fail(FAIL_GUARD, "qualification report identity differs")
    body = dict(report)
    claimed_hash = body.pop("diagnostic_report_hash")
    expected_hash = "sha256:" + sha256(
        REPORT_DOMAIN + b"\x00" + _canonical_json_bytes(body)
    ).hexdigest()
    if claimed_hash != expected_hash:
        _fail(FAIL_GUARD, "diagnostic report hash differs")

    repository = _mapping(report["repository_binding"], "repository binding")
    commit = repository.get("qualification_basis_commit")
    rows = repository.get("source_files")
    if (
        type(commit) is not str
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or type(rows) is not list
        or not rows
        or repository.get("source_file_count") != len(rows)
        or repository.get("source_file_set_root") != source_file_set_root_v1(rows)
    ):
        _fail(FAIL_GUARD, "repository/source-file binding differs")

    sealed = _mapping(report["sealed_basis"], "sealed basis")
    expected_ids = [vector.vector_id for vector in STRICT_GOLDEN_VECTORS_V1]
    if sealed != {
        "golden_vector_manifest_root": EXPECTED_MANIFEST_ROOT,
        "expected_outcome_root": EXPECTED_OUTCOME_ROOT,
        "ordered_vector_ids": expected_ids,
        "vector_count": 36,
    }:
        _fail(FAIL_GUARD, "sealed basis differs")

    replay = _mapping(report["dual_vector_replay"], "dual vector replay")
    vector_rows = replay.get("vectors")
    if type(vector_rows) is not list or len(vector_rows) != 36:
        _fail(FAIL_GUARD, "dual vector rows differ")
    outcomes: dict[str, bytes] = {}
    for vector, row_value in zip(STRICT_GOLDEN_VECTORS_V1, vector_rows, strict=True):
        row = _mapping(row_value, f"vector {vector.vector_id}")
        if (
            row.get("vector_id") != vector.vector_id
            or row.get("category") != vector.category
            or row.get("boundary") != vector.boundary
            or row.get("input_wire_sha256") != sha256(vector.input_wire).hexdigest()
            or row.get("input_wire_size") != len(vector.input_wire)
            or row.get("expected_disposition") != vector.expected_disposition
            or row.get("python_exit") != 0
            or row.get("dual_equal") is not True
        ):
            _fail(FAIL_GUARD, f"vector binding differs: {vector.vector_id}")
        normalized = _mapping(row.get("normalized"), f"normalized {vector.vector_id}")
        if vector.expected_disposition == ACCEPT_PARENT_IDENTITY:
            if normalized.get("status") != "ACCEPTED" or row.get("rust_exit") != 0:
                _fail(FAIL_GUARD, f"accepted vector disposition differs: {vector.vector_id}")
            try:
                cbor = bytes.fromhex(str(normalized["canonical_cbor_hex"]))
            except (KeyError, ValueError) as error:
                _fail(FAIL_GUARD, f"accepted vector CBOR differs: {vector.vector_id}: {error}")
            digest = sha256(AST_HASH_DOMAIN + b"\x00" + cbor).digest()
            if normalized.get("canonical_ast_hash") != "sha256:" + digest.hex():
                _fail(FAIL_GUARD, f"accepted vector hash differs: {vector.vector_id}")
            outcome = accepted_outcome_bytes(cbor, digest)
        else:
            if (
                normalized
                != {"status": "REJECTED", "error_code": vector.expected_disposition}
                or row.get("rust_exit") != 1
            ):
                _fail(FAIL_GUARD, f"rejected vector disposition differs: {vector.vector_id}")
            outcome = rejected_outcome_bytes(vector.expected_disposition)
        if row.get("normalized_outcome_sha256") != sha256(outcome).hexdigest():
            _fail(FAIL_GUARD, f"normalized outcome hash differs: {vector.vector_id}")
        outcomes[vector.vector_id] = outcome
    outcome_root = strict_golden_outcome_root_v1(outcomes)
    if (
        replay.get("status") != STATUS_PASS
        or replay.get("vector_count") != 36
        or replay.get("python_outcome_root") != outcome_root
        or replay.get("rust_outcome_root") != outcome_root
        or outcome_root != EXPECTED_OUTCOME_ROOT
        or replay.get("all_normalized_outcomes_equal") is not True
    ):
        _fail(FAIL_GUARD, "dual outcome root differs")

    capacity = _mapping(report["dual_capacity_replay"], "dual capacity replay")
    capacity_guards = {
        "status": "DUAL_SURVIVOR_SUBSET_REPLAY_PASS_NOT_COMPLETE",
        "all_comparable_fields_equal": True,
        "source_candidate_count": 2160,
        "accepted_source_count": 2160,
        "accepted_unique_count": 2160,
        "parent_identity_match_count": 2160,
        "rejected_count": 0,
        "rewrite_collapsed_count": 0,
        "accepted_set_commitment": EXPECTED_CAPACITY_COMMITMENT,
        "subset_status": "SURVIVOR_SUBSET_ONLY_NOT_COMPLETE",
        "executed_closure_status": "NOT_RUN",
        "complete_closure_enumerated": False,
        "interpreted_as_complete_closure": False,
        "formal_roots": None,
        "target_or_split_modules_loaded": False,
    }
    if any(capacity.get(field) != expected for field, expected in capacity_guards.items()):
        _fail(FAIL_GUARD, "dual capacity guard differs")

    runtime = _mapping(report["runtime_isolation"], "runtime isolation")
    if (
        runtime.get("role_topology")
        != "HOST_SUPERVISOR_PLUS_TWO_DISJOINT_PINNED_CONTAINERS"
        or runtime.get("python_image_ref") != PYTHON_IMAGE
        or runtime.get("python_image_id") != PYTHON_IMAGE.split("@", 1)[1]
        or runtime.get("rust_image_ref") != RUST_IMAGE
        or runtime.get("rust_image_id") != RUST_IMAGE.split("@", 1)[1]
        or runtime.get("pull_policy") != "never"
        or runtime.get("network_mode") != "none"
        or runtime.get("capabilities_dropped") != "ALL"
        or runtime.get("no_new_privileges") is not True
        or runtime.get("container_root_filesystem_read_only") is not True
        or runtime.get("source_snapshot_mount_read_only") is not True
        or runtime.get("fresh_ephemeral_rust_target_volume") is not True
        or runtime.get("rust_target_volume_removed_after_run") is not True
        or runtime.get("cargo_locked") is not True
        or runtime.get("cargo_offline") is not True
        or runtime.get("rust_build_user") != "0:0"
        or runtime.get("rust_build_tmpfs") != BUILD_TMPFS
        or runtime.get("recognizer_runtime_user") != "65534:65534"
        or runtime.get("recognizer_runtime_tmpfs") != RUNTIME_TMPFS
        or runtime.get("memory_limit") != "512m"
        or runtime.get("memory_swap_limit") != "512m"
        or runtime.get("pids_limit") != 64
        or runtime.get("python_flags") != ["-I", "-S", "-B"]
        or runtime.get("same_admin_controller") is not True
        or runtime.get("organizational_independence") is not False
        or runtime.get("independent_human_actors") is not False
        or runtime.get("technical_role_independence") is not True
    ):
        _fail(FAIL_GUARD, "runtime isolation guard differs")
    daemon = _mapping(
        runtime.get("docker_daemon_identity_receipt"), "Docker daemon receipt"
    )
    daemon_body = dict(daemon)
    daemon_hash = daemon_body.pop("diagnostic_receipt_hash", None)
    if daemon_hash != "sha256:" + sha256(
        DAEMON_RECEIPT_DOMAIN + b"\x00" + _canonical_json_bytes(daemon_body)
    ).hexdigest():
        _fail(FAIL_GUARD, "Docker daemon receipt hash differs")
    profiles = _mapping(runtime.get("committed_profile_receipt"), "profile receipt")
    source_hashes = {str(row["path"]): row["sha256"] for row in rows}
    if profiles != {
        "actor_profile_id": "hegel-owner-accepted-container-technical-actors-v1",
        "actor_profile_sha256": source_hashes[PROFILE_PATH],
        "build_profile_id": "hegel-shrink3-rust-offline-build-v1",
        "build_profile_sha256": source_hashes[BUILD_PROFILE_PATH],
    }:
        _fail(FAIL_GUARD, "committed profile receipt differs")
    volume = _mapping(runtime.get("rust_target_volume_receipt"), "target volume receipt")
    if (
        volume.get("name") != f"hegel-shrink3-sealed-{commit[:12]}"
        or volume.get("driver") != "local"
        or volume.get("scope") != "local"
        or volume.get("options") is not None
        or volume.get("fresh_before_run") is not True
        or volume.get("labels")
        != {
            "hegel.machine.role": "shrink3-dual-strict",
            "hegel.machine.basis": commit,
            "hegel.machine.network": "none",
        }
    ):
        _fail(FAIL_GUARD, "target-volume receipt differs")

    if report["authority_guards"] != {
        "execution_state": "NOT_RUN",
        "closure_executed": False,
        "formal_roots_generated": False,
        "formal_roots": None,
        "certificate_issued": False,
        "signature_generated": False,
        "seed_generated": False,
        "target_roles_evaluated": False,
        "active_governance_changed": False,
        "formal_state_transition_allowed": False,
    }:
        _fail(FAIL_GUARD, "authority guards differ")


def qualify(
    basis_commit: str,
    *,
    workers: int,
    cargo_registry: Path,
) -> dict[str, object]:
    global _DOCKER_ENV
    if re.fullmatch(r"[0-9a-f]{40}", basis_commit) is None:
        _fail(FAIL_ARGUMENT, "basis commit must be a full lowercase SHA-1")
    resolved = _git("rev-parse", "--verify", f"{basis_commit}^{{commit}}")
    if resolved != basis_commit:
        _fail(FAIL_GIT_BINDING, "basis commit does not resolve exactly")
    if not 1 <= workers <= 16:
        _fail(FAIL_ARGUMENT, "workers must be in [1,16]")
    source_rows, source_root = _source_rows(basis_commit)
    project_tree_oid = _git("rev-parse", f"{basis_commit}:Hegel Machine")
    subject = _git("show", "-s", "--format=%s", basis_commit)
    parents = str(_git("show", "-s", "--format=%P", basis_commit)).split()
    archive = _git("archive", "--format=tar", basis_commit, "--", *ARCHIVE_PATHS, binary=True)
    if type(archive) is not bytes:
        _fail(FAIL_ARCHIVE, "Git archive is not bytes")
    archive_sha256 = sha256(archive).hexdigest()
    repository_binding = {
        "qualification_basis_commit": basis_commit,
        "qualification_basis_parent_commits": parents,
        "qualification_basis_subject": subject,
        "project_tree_oid": project_tree_oid,
        "archive_sha256": archive_sha256,
        "source_file_count": len(source_rows),
        "source_file_set_root": source_root,
        "supervisor_source_sha256": next(
            row["sha256"] for row in source_rows if row["path"] == SUPERVISOR_PATH
        ),
        "source_files": source_rows,
    }

    volume = f"hegel-shrink3-sealed-{basis_commit[:12]}"
    with (
        tempfile.TemporaryDirectory(prefix="hegel-shrink3-docker-control-") as control,
        tempfile.TemporaryDirectory(prefix="hegel-shrink3-sealed-snapshot-") as temporary,
    ):
        try:
            daemon_receipt = _initialize_docker_environment(Path(control))
            python_image_id = _inspect_image(PYTHON_IMAGE)
            rust_image_id = _inspect_image(RUST_IMAGE)
            snapshot_root = Path(temporary)
            _safe_extract_git_archive(archive, snapshot_root)
            _validate_snapshot(snapshot_root, source_rows)
            snapshot_project = snapshot_root / "Hegel Machine"
            profile_receipt = _profile_images(snapshot_project)
            volume_receipt = _create_fresh_volume(volume, basis_commit)
            primary: BaseException | None = None
            report: dict[str, object] | None = None
            try:
                report = _qualify(
                    snapshot_project,
                    volume,
                    repository_binding=repository_binding,
                    workers=workers,
                    cargo_registry=cargo_registry.resolve(),
                    python_image_id=python_image_id,
                    rust_image_id=rust_image_id,
                    daemon_receipt=daemon_receipt,
                    volume_receipt=volume_receipt,
                    profile_receipt=profile_receipt,
                )
            except BaseException as error:
                primary = error
            try:
                _remove_volume(volume)
            except QualificationError as cleanup_error:
                if primary is not None:
                    if isinstance(primary, QualificationError):
                        raise QualificationError(
                            primary.code,
                            f"{primary.detail}; secondary cleanup failure: "
                            f"{cleanup_error.code}: {cleanup_error.detail}",
                        ) from primary
                    raise QualificationError(
                        FAIL_CLEANUP,
                        f"primary {type(primary).__name__}; secondary cleanup "
                        f"failure: {cleanup_error.detail}",
                    ) from primary
                raise
            if primary is not None:
                raise primary
            if report is None:
                _fail(FAIL_ENDPOINT, "qualification returned no report")
            validate_qualification_report(report)
        finally:
            _DOCKER_ENV = None
    return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-commit", required=True)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--cargo-registry", type=Path, default=DEFAULT_CARGO_REGISTRY)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    try:
        report = qualify(
            arguments.basis_commit,
            workers=arguments.workers,
            cargo_registry=arguments.cargo_registry,
        )
    except QualificationError as error:
        sys.stderr.write(
            json.dumps(
                {"status": "FAIL_CLOSED", "failure_code": error.code, "detail": error.detail},
                sort_keys=True,
                separators=(",", ":"),
            ) + "\n"
        )
        return 2
    except Exception as error:
        sys.stderr.write(
            json.dumps(
                {
                    "status": "FAIL_CLOSED",
                    "failure_code": FAIL_INTERNAL,
                    "detail": f"{type(error).__name__}: {error}",
                },
                sort_keys=True,
                separators=(",", ":"),
            ) + "\n"
        )
        return 2
    sys.stdout.buffer.write(_canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
