#!/usr/bin/env python3
"""Commit-bound offline supervisor for the shrink-3 dual diagnostic.

This control plane builds the independent Rust enumerator from an exact Git
archive, launches the Python and Rust 50,000-program diagnostics concurrently,
and only then asks the untrusted target-free host replay to compare both public
archives.  Its retained output is engineering evidence only: M3 remains
``NOT_RUN`` and no target, split, seed, key, signature, or formal root is used.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Callable, Final, Mapping, NoReturn, Sequence


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT: Final = PROJECT_ROOT.parent
HARDENED_TOOL_PATH: Final = (
    PROJECT_ROOT / "tools/phase3_shrink3_dual_strict_qualification_v1.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "hegel_shrink3_hardened_docker_v1", HARDENED_TOOL_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("hardened Docker primitive module is unavailable")
hardened = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = hardened
_SPEC.loader.exec_module(hardened)


SCHEMA: Final = "hegel-shrink3-dual-complete-diagnostic-supervisor/1"
STATUS_PASS: Final = "DUAL_COMPLETE_DIAGNOSTIC_HOST_REPLAY_PASS"
CLAIM_LEVEL: Final = "NON_FORMAL_DUAL_CHILD_DIAGNOSTIC"
SOURCE_SET_DOMAIN: Final = b"HEGEL/SHRINK3/DUAL_COMPLETE_SOURCE_SET/V1"
CARGO_SEED_SET_DOMAIN: Final = b"HEGEL/SHRINK3/CARGO_DEPENDENCY_SEED_SET/V1"
SUMMARY_DOMAIN: Final = b"HEGEL/SHRINK3/DUAL_COMPLETE_SUPERVISOR/V1"
PYTHON_IMAGE: Final = hardened.PYTHON_IMAGE
RUST_IMAGE: Final = hardened.RUST_IMAGE
RUST_TOOLCHAIN_BIN: Final = hardened.RUST_TOOLCHAIN_BIN
DEFAULT_CARGO_REGISTRY: Final = hardened.DEFAULT_CARGO_REGISTRY
DEFAULT_TIMEOUT_SECONDS: Final = 12 * 60 * 60
BUILD_PROFILE_PATH: Final = (
    "Hegel Machine/config/phase3_shrink3_enumerator_offline_build_profile_v1.json"
)
SUPERVISOR_PATH: Final = (
    "Hegel Machine/tools/phase3_shrink3_dual_complete_diagnostic_v1.py"
)
HARDENED_SUPERVISOR_PATH: Final = (
    "Hegel Machine/tools/phase3_shrink3_dual_strict_qualification_v1.py"
)
ACTOR_PROFILE_PATH: Final = (
    "Hegel Machine/config/phase3_container_actor_profile_v1.json"
)
STRICT_BUILD_PROFILE_PATH: Final = (
    "Hegel Machine/config/phase3_shrink3_offline_build_profile_v1.json"
)
RUNTIME_SECCOMP_PATH: Final = hardened.RUNTIME_SECCOMP_PATH
BUILD_SECCOMP_PATH: Final = hardened.BUILD_SECCOMP_PATH
PYTHON_ENTRYPOINT: Final = (
    "/workspace/src/hegel_machine/phase3_m3_shrink3_isolated_entrypoint_v1.py"
)
HOST_ENTRYPOINT: Final = (
    "/workspace/src/hegel_machine/phase3_m3_shrink3_dual_diagnostic_entrypoint_v1.py"
)
RUST_CRATE: Final = "m3_closure_enumerator_shrink3"
RUST_BINARY: Final = "hegel-m3-closure-enumerator-shrink3"
RUST_BINARY_PATH: Final = f"/cargo-target/release/{RUST_BINARY}"
CHILD_DSL_SPEC_ROOT: Final = (
    "64aaf01392ca89a1ade3a3766d756b53e9b0e7ec6ab4ca2b4fb74ec658490677"
)
OPERATOR_SEMANTICS_ROOT: Final = (
    "e3337cc67974c8fbbfa6d8f89301184c1658a98b80c0a1fac11251ede9aa15f1"
)
IDENTIFIER_REGISTRY_ROOT: Final = (
    "9dd80c452334db8afd9fbb56f1c74f365f63db61ec4c5667bddbb88e57ec05c8"
)

PYTHON_SOURCE_PATHS: Final = (
    "Hegel Machine/src/hegel_machine/phase3_m3_bounded_enumerator_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_bounded_enumerator_shrink2_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_bounded_enumerator_shrink3_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_dsl_core_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_record_wire_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_shrink1_core_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_shrink2_core_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_shrink3_core_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_shrink3_diagnostic_profile_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_shrink3_dual_diagnostic_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_shrink3_isolated_entrypoint_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_shrink3_dual_diagnostic_entrypoint_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_shrink3_golden_vectors_v1.py",
    "Hegel Machine/src/hegel_machine/strict_ast_v1.py",
    "Hegel Machine/src/hegel_machine/strict_ast_shrink1_v1.py",
    "Hegel Machine/src/hegel_machine/strict_ast_shrink2_v1.py",
    "Hegel Machine/src/hegel_machine/strict_ast_shrink3_v1.py",
    "Hegel Machine/src/hegel_machine/strict_cbor_v1.py",
)
RUST_SOURCE_DIRS: Final = (
    "Hegel Machine/rust/strict_canonicalizer",
    "Hegel Machine/rust/strict_canonicalizer_shrink1",
    "Hegel Machine/rust/strict_canonicalizer_shrink2",
    "Hegel Machine/rust/strict_canonicalizer_shrink3",
    "Hegel Machine/rust/m3_closure_enumerator_shrink3",
)
RUST_REQUIRED_PATHS: Final = (
    "Hegel Machine/rust/strict_canonicalizer/Cargo.toml",
    "Hegel Machine/rust/strict_canonicalizer/src/lib.rs",
    "Hegel Machine/rust/strict_canonicalizer_shrink1/Cargo.toml",
    "Hegel Machine/rust/strict_canonicalizer_shrink1/src/lib.rs",
    "Hegel Machine/rust/strict_canonicalizer_shrink2/Cargo.toml",
    "Hegel Machine/rust/strict_canonicalizer_shrink2/src/lib.rs",
    "Hegel Machine/rust/strict_canonicalizer_shrink3/Cargo.toml",
    "Hegel Machine/rust/strict_canonicalizer_shrink3/src/lib.rs",
    "Hegel Machine/rust/m3_closure_enumerator_shrink3/Cargo.toml",
    "Hegel Machine/rust/m3_closure_enumerator_shrink3/Cargo.lock",
    "Hegel Machine/rust/m3_closure_enumerator_shrink3/src/lib.rs",
    "Hegel Machine/rust/m3_closure_enumerator_shrink3/src/main.rs",
    "Hegel Machine/rust/m3_closure_enumerator_shrink3/src/formal_core.rs",
)
ARCHIVE_PATHS: Final = (
    SUPERVISOR_PATH,
    HARDENED_SUPERVISOR_PATH,
    ACTOR_PROFILE_PATH,
    STRICT_BUILD_PROFILE_PATH,
    BUILD_PROFILE_PATH,
    RUNTIME_SECCOMP_PATH,
    BUILD_SECCOMP_PATH,
    *PYTHON_SOURCE_PATHS,
    *RUST_SOURCE_DIRS,
)
EXPECTED_OUTPUT_FILES: Final = frozenset(
    {
        "report.json",
        "canonical_program_records.cborframed",
        "program_chunk_manifests.cborframed",
        "bucket_accounting_records.cborframed",
    }
)

FAIL_ARGUMENT: Final = "FAIL_SHRINK3_DUAL_COMPLETE_ARGUMENT"
FAIL_GIT: Final = "FAIL_SHRINK3_DUAL_COMPLETE_GIT_BINDING"
FAIL_DOCKER: Final = "FAIL_SHRINK3_DUAL_COMPLETE_DOCKER_POLICY"
FAIL_BUILD: Final = "FAIL_SHRINK3_DUAL_COMPLETE_BUILD"
FAIL_ENDPOINT: Final = "FAIL_SHRINK3_DUAL_COMPLETE_ENDPOINT"
FAIL_HOST: Final = "FAIL_SHRINK3_DUAL_COMPLETE_HOST_REPLAY"
FAIL_OUTPUT: Final = "FAIL_SHRINK3_DUAL_COMPLETE_OUTPUT"
FAIL_AUTHORITY: Final = "FAIL_SHRINK3_DUAL_COMPLETE_AUTHORITY"
FAIL_CLEANUP: Final = "FAIL_SHRINK3_DUAL_COMPLETE_CLEANUP"
FAIL_INTERNAL: Final = "FAIL_SHRINK3_DUAL_COMPLETE_INTERNAL"


class SupervisorError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise SupervisorError(code, detail)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _hardened_call(
    failure_code: str,
    function: Callable[..., object],
    *args: object,
    **kwargs: object,
) -> object:
    try:
        return function(*args, **kwargs)
    except hardened.QualificationError as error:
        _fail(failure_code, f"{error.code}: {error.detail}")


def source_file_set_root_v1(rows: Sequence[Mapping[str, object]]) -> str:
    digest = sha256(SOURCE_SET_DOMAIN + b"\x00")
    previous = ""
    for row in rows:
        path = str(row["path"])
        if path <= previous:
            _fail(FAIL_GIT, "source rows are not strictly path ordered")
        previous = path
        fields = (
            path.encode("utf-8"),
            str(row["mode"]).encode("ascii"),
            bytes.fromhex(str(row["git_blob_oid"])),
            bytes.fromhex(str(row["sha256"])),
            int(row["size"]).to_bytes(8, "big"),
        )
        if len(fields[2]) != 20 or len(fields[3]) != 32:
            _fail(FAIL_GIT, f"invalid digest width for {path}")
        for field in fields:
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return "sha256:" + digest.hexdigest()


def _file_set_root(
    domain: bytes, rows: Sequence[Mapping[str, object]]
) -> str:
    digest = sha256(domain + b"\x00")
    previous = ""
    for row in rows:
        path = str(row["path"])
        if path <= previous:
            _fail(FAIL_BUILD, "dependency rows are not strictly path ordered")
        previous = path
        fields = (
            path.encode("utf-8"),
            int(row["size"]).to_bytes(8, "big"),
            bytes.fromhex(str(row["sha256"])),
        )
        if len(fields[2]) != 32:
            _fail(FAIL_BUILD, f"dependency digest width differs: {path}")
        for field in fields:
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return "sha256:" + digest.hexdigest()


def _locked_registry_crates(lock_path: Path) -> dict[str, str]:
    try:
        lines = lock_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        _fail(FAIL_BUILD, f"Cargo.lock is unreadable: {error}")
    packages: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw_line in lines:
        line = raw_line.strip()
        if line == "[[package]]":
            if current is not None:
                packages.append(current)
            current = {}
            continue
        if current is None or " = " not in line:
            continue
        key, encoded = line.split(" = ", 1)
        if key not in {"name", "version", "source", "checksum"}:
            continue
        try:
            value = json.loads(encoded)
        except json.JSONDecodeError as error:
            _fail(FAIL_BUILD, f"Cargo.lock string is invalid: {error}")
        if type(value) is not str:
            _fail(FAIL_BUILD, f"Cargo.lock {key} is not a string")
        current[key] = value
    if current is not None:
        packages.append(current)

    locked: dict[str, str] = {}
    for package in packages:
        source = package.get("source")
        if source is None:
            continue
        if not source.startswith("registry+"):
            _fail(FAIL_BUILD, f"unsupported Cargo source: {source}")
        try:
            filename = f"{package['name']}-{package['version']}.crate"
            checksum = package["checksum"]
        except KeyError as error:
            _fail(FAIL_BUILD, f"registry package lacks {error.args[0]}")
        if re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
            _fail(FAIL_BUILD, f"invalid locked checksum for {filename}")
        if filename in locked:
            _fail(FAIL_BUILD, f"duplicate locked registry package: {filename}")
        locked[filename] = checksum
    if not locked:
        _fail(FAIL_BUILD, "Cargo.lock has no registry dependencies")
    return locked


def verify_cargo_dependency_seed_v1(
    cargo_registry: Path, lock_path: Path
) -> dict[str, object]:
    """Bind cache/index bytes and verify every locked crate before Docker."""

    root = cargo_registry.resolve()
    cache = root / "cache"
    index = root / "index"
    if not cache.is_dir() or not index.is_dir():
        _fail(FAIL_BUILD, "Cargo dependency seed needs cache and index directories")
    locked = _locked_registry_crates(lock_path)
    crate_paths = sorted(cache.rglob("*.crate"))
    actual: dict[str, Path] = {}
    for path in crate_paths:
        if path.is_symlink() or not path.is_file() or path.name in actual:
            _fail(FAIL_BUILD, f"invalid Cargo crate seed path: {path}")
        actual[path.name] = path
    if set(actual) != set(locked):
        _fail(
            FAIL_BUILD,
            "Cargo crate seed differs from Cargo.lock; "
            f"missing={sorted(set(locked) - set(actual))}; "
            f"extra={sorted(set(actual) - set(locked))}",
        )
    for filename, expected in locked.items():
        if sha256(actual[filename].read_bytes()).hexdigest() != expected:
            _fail(FAIL_BUILD, f"locked crate checksum differs: {filename}")

    rows: list[dict[str, object]] = []
    for subtree in (cache, index):
        for path in sorted(subtree.rglob("*")):
            if path.is_symlink():
                _fail(FAIL_BUILD, f"Cargo dependency seed contains symlink: {path}")
            if path.is_dir():
                continue
            if not path.is_file():
                _fail(FAIL_BUILD, f"Cargo dependency seed is non-regular: {path}")
            payload = path.read_bytes()
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": len(payload),
                    "sha256": sha256(payload).hexdigest(),
                }
            )
    rows.sort(key=lambda row: str(row["path"]))
    return {
        "schema_version": "hegel-shrink3-cargo-dependency-seed/1",
        "file_set_root": _file_set_root(CARGO_SEED_SET_DOMAIN, rows),
        "file_count": len(rows),
        "total_bytes": sum(int(row["size"]) for row in rows),
        "locked_registry_package_count": len(locked),
        "verified_crate_count": len(actual),
        "selected_subtrees": ["cache", "index"],
        "src_subtree_included": False,
        "fresh_tmpfs_cargo_home": True,
        "files": rows,
    }


def create_verified_cargo_seed_snapshot_v1(
    cargo_registry: Path, destination: Path, lock_path: Path
) -> dict[str, object]:
    """Copy then reverify the exact dependency bytes Docker will consume."""

    source_receipt = verify_cargo_dependency_seed_v1(cargo_registry, lock_path)
    if destination.exists() or destination.is_symlink():
        _fail(FAIL_BUILD, "Cargo seed snapshot destination already exists")
    try:
        destination.mkdir(mode=0o700)
        for name in ("cache", "index"):
            shutil.copytree(cargo_registry.resolve() / name, destination / name)
        snapshot_receipt = verify_cargo_dependency_seed_v1(
            destination, lock_path
        )
    except OSError as error:
        _fail(FAIL_BUILD, f"Cargo seed snapshot failed: {error}")
    if snapshot_receipt != source_receipt:
        _fail(FAIL_BUILD, "Cargo seed snapshot bytes differ after copy")
    for path in sorted(destination.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    destination.chmod(0o555)
    return {
        **snapshot_receipt,
        "snapshot_reverified_after_copy": True,
        "docker_bind_mounts_exact": ["cache", "index"],
    }


def _restore_owner_writable_tree(path: Path) -> None:
    path.chmod(0o700)
    for child in path.rglob("*"):
        child.chmod(0o700 if child.is_dir() else 0o600)


def _git(*arguments: str, binary: bool = False) -> bytes | str:
    value = _hardened_call(FAIL_GIT, hardened._git, *arguments, binary=binary)
    if not isinstance(value, (bytes, str)):
        _fail(FAIL_GIT, "Git primitive returned an invalid value")
    return value


def _source_rows(basis_commit: str) -> tuple[list[dict[str, object]], str]:
    listing = _git("ls-tree", "-r", basis_commit, "--", *ARCHIVE_PATHS)
    if type(listing) is not str:
        _fail(FAIL_GIT, "Git tree listing is not text")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for line in listing.splitlines():
        metadata, path = line.split("\t", 1)
        mode, kind, oid = metadata.split(" ")
        if kind != "blob" or path in seen:
            _fail(FAIL_GIT, f"invalid Git tree row for {path}")
        payload = _git("show", f"{basis_commit}:{path}", binary=True)
        if type(payload) is not bytes:
            _fail(FAIL_GIT, f"Git blob is not bytes: {path}")
        worktree = REPOSITORY_ROOT / path
        if not worktree.is_file() or worktree.read_bytes() != payload:
            _fail(FAIL_GIT, f"worktree differs from basis commit: {path}")
        if hardened._git_blob_oid(payload) != oid:
            _fail(FAIL_GIT, f"Git blob identity differs: {path}")
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
    required = set(PYTHON_SOURCE_PATHS) | set(RUST_REQUIRED_PATHS) | {
        SUPERVISOR_PATH,
        HARDENED_SUPERVISOR_PATH,
        ACTOR_PROFILE_PATH,
        STRICT_BUILD_PROFILE_PATH,
        BUILD_PROFILE_PATH,
        RUNTIME_SECCOMP_PATH,
        BUILD_SECCOMP_PATH,
    }
    if not required.issubset(seen):
        _fail(FAIL_GIT, f"required committed files are absent: {sorted(required - seen)}")
    rows.sort(key=lambda row: str(row["path"]))
    return rows, source_file_set_root_v1(rows)


def _validate_external_output(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        _fail(FAIL_ARGUMENT, "output directory must not already exist")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as error:
        _fail(FAIL_ARGUMENT, f"output parent is unavailable: {error}")
    if not parent.is_dir():
        _fail(FAIL_ARGUMENT, "output parent is not a directory")
    candidate = parent / path.name
    repository = REPOSITORY_ROOT.resolve()
    if candidate == repository or repository in candidate.parents:
        _fail(FAIL_ARGUMENT, "retained diagnostic output must be outside the repository")
    return candidate


def _load_build_profile(snapshot_project: Path) -> dict[str, object]:
    path = snapshot_project / "config/phase3_shrink3_enumerator_offline_build_profile_v1.json"
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, json.JSONDecodeError) as error:
        _fail(FAIL_DOCKER, f"enumerator build profile is unreadable: {error}")
    expected = {
        "profile_id": "hegel-shrink3-enumerator-offline-build-v1",
        "purpose": "build-only target-free shrink3 complete-closure diagnostic enumerator",
        "authority": "engineering diagnostic only; M3 remains NOT_RUN",
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
        "tmpfs": hardened.BUILD_TMPFS,
        "seccomp_profile": "config/phase3_m3_offline_build_seccomp_v1.json",
        "source_mount": "read-only committed git archive snapshot",
        "cargo_registry_mount": (
            "two exact read-only reverified snapshot mounts: cache and index; "
            "registry root and preexisting src excluded"
        ),
        "cargo_home": (
            "fresh tmpfs; crates re-unpacked after Cargo.lock checksum verification"
        ),
        "cargo_dependency_seed_file_set_root_required": True,
        "target_mount": "fresh local-driver docker volume, build rw then runtime ro",
        "crate": RUST_CRATE,
        "release_binary": RUST_BINARY_PATH,
        "cargo_flags": ["--release", "--locked", "--offline"],
        "parallel_endpoint_count": 2,
        "target_or_split_inputs_allowed": False,
        "seed_key_signature_or_formal_root_access_allowed": False,
    }
    if value != expected:
        _fail(FAIL_DOCKER, "enumerator offline-build profile differs")
    return {
        "profile_id": value["profile_id"],
        "sha256": sha256(payload).hexdigest(),
    }


def _create_fresh_volume(name: str, basis_commit: str) -> dict[str, object]:
    prefix = hardened._docker_prefix()
    inspect = _hardened_call(
        FAIL_DOCKER,
        hardened._run,
        (*prefix, "volume", "inspect", name),
        allowed_codes=frozenset({0, 1}),
        code=hardened.FAIL_DOCKER_POLICY,
    )
    assert isinstance(inspect, subprocess.CompletedProcess)
    if inspect.returncode == 0:
        _fail(FAIL_DOCKER, f"fresh target volume already exists: {name}")
    labels = {
        "hegel.machine.role": "shrink3-dual-complete-diagnostic",
        "hegel.machine.basis": basis_commit,
        "hegel.machine.network": "none",
    }
    command: list[str] = [*prefix, "volume", "create", "--driver", "local"]
    for key, value in labels.items():
        command.extend(("--label", f"{key}={value}"))
    command.append(name)
    created = _hardened_call(
        FAIL_DOCKER, hardened._run, command, code=hardened.FAIL_DOCKER_POLICY
    )
    assert isinstance(created, subprocess.CompletedProcess)
    if created.stdout.decode().strip() != name:
        _hardened_call(FAIL_CLEANUP, hardened._remove_volume, name)
        _fail(FAIL_DOCKER, "Docker returned a different volume name")
    try:
        detail_result = _hardened_call(
            FAIL_DOCKER,
            hardened._run,
            (*prefix, "volume", "inspect", name, "--format", "{{json .}}"),
            code=hardened.FAIL_DOCKER_POLICY,
        )
        assert isinstance(detail_result, subprocess.CompletedProcess)
        detail = _hardened_call(
            FAIL_DOCKER,
            hardened._one_json,
            detail_result,
            "enumerator target volume",
        )
        assert isinstance(detail, dict)
        if (
            detail.get("Name") != name
            or detail.get("Driver") != "local"
            or detail.get("Scope") != "local"
            or detail.get("Options") not in (None, {})
            or detail.get("Labels") != labels
        ):
            _fail(FAIL_DOCKER, "fresh target volume policy differs")
    except BaseException as primary:
        try:
            _hardened_call(FAIL_CLEANUP, hardened._remove_volume, name)
        except SupervisorError as cleanup:
            if isinstance(primary, SupervisorError):
                raise SupervisorError(
                    primary.code,
                    f"{primary.detail}; cleanup failed: {cleanup.code}: {cleanup.detail}",
                ) from primary
            raise
        raise
    return {"name": name, "labels": labels, "fresh_before_run": True}


def rust_build_command(
    snapshot_project: Path, cargo_seed: Path, volume: str, workers: int
) -> list[str]:
    command = hardened._docker_common(
        RUST_IMAGE,
        seccomp_path=snapshot_project / "config/phase3_m3_offline_build_seccomp_v1.json",
        tmpfs=hardened.BUILD_TMPFS,
    )
    command[-1:-1] = [
        "-e", "CARGO_HOME=/tmp/cargo-home",
        "-e", "CARGO_NET_OFFLINE=true",
        "-e", f"CARGO_BUILD_JOBS={workers}",
        "-e", "CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1",
        "-e", "CARGO_TARGET_DIR=/cargo-target",
        "-e", f"RUSTC={RUST_TOOLCHAIN_BIN}/rustc",
        "-e", f"RUSTDOC={RUST_TOOLCHAIN_BIN}/rustdoc",
        "-v", f"{cargo_seed / 'cache'}:/cargo-seed/cache:ro",
        "-v", f"{cargo_seed / 'index'}:/cargo-seed/index:ro",
        "-v", f"{snapshot_project / 'rust'}:/workspace/rust:ro",
        "-v", f"{volume}:/cargo-target:rw",
        "-w", f"/workspace/rust/{RUST_CRATE}",
    ]
    command.extend(
        (
            "/bin/sh",
            "-ceu",
            (
                "mkdir -p /tmp/cargo-home/registry; "
                "cp -R /cargo-seed/cache /tmp/cargo-home/registry/cache; "
                "cp -R /cargo-seed/index /tmp/cargo-home/registry/index; "
                "test ! -e /tmp/cargo-home/registry/src; "
                "exec \"$@\""
            ),
            "hegel-cargo-fresh-unpack",
            f"{RUST_TOOLCHAIN_BIN}/cargo",
            "build",
            "--release",
            "--locked",
            "--offline",
            "--bin",
            RUST_BINARY,
        )
    )
    return command


def _enumeration_arguments(output: str) -> tuple[str, ...]:
    return (
        "--enumerate-diagnostic",
        "--child-dsl-spec-root", CHILD_DSL_SPEC_ROOT,
        "--operator-semantics-root", OPERATOR_SEMANTICS_ROOT,
        "--identifier-registry-root", IDENTIFIER_REGISTRY_ROOT,
        "--output-directory", output,
    )


def python_endpoint_command(snapshot_project: Path, output_parent: Path) -> list[str]:
    command = hardened._docker_common(
        PYTHON_IMAGE,
        seccomp_path=snapshot_project / "config/phase3_internal_actor_seccomp_v1.json",
        user="65534:65534",
    )
    command[-1:-1] = [
        "-e", "PYTHONHASHSEED=0",
        "-v", f"{snapshot_project}:/workspace:ro",
        "-v", f"{output_parent}:/output:rw",
        "-w", "/workspace",
    ]
    command.extend(
        (
            "/usr/local/bin/python3", "-I", "-S", "-B", PYTHON_ENTRYPOINT,
            *_enumeration_arguments("/output/result"),
        )
    )
    return command


def rust_endpoint_command(
    snapshot_project: Path, output_parent: Path, volume: str
) -> list[str]:
    command = hardened._docker_common(
        RUST_IMAGE,
        seccomp_path=snapshot_project / "config/phase3_internal_actor_seccomp_v1.json",
        user="65534:65534",
    )
    command[-1:-1] = [
        "-v", f"{snapshot_project}:/workspace:ro",
        "-v", f"{volume}:/cargo-target:ro",
        "-v", f"{output_parent}:/output:rw",
    ]
    command.extend((RUST_BINARY_PATH, *_enumeration_arguments("/output/result")))
    return command


def host_replay_command(
    snapshot_project: Path, python_output: Path, rust_output: Path
) -> list[str]:
    command = hardened._docker_common(
        PYTHON_IMAGE,
        seccomp_path=snapshot_project / "config/phase3_internal_actor_seccomp_v1.json",
        user="65534:65534",
    )
    command[-1:-1] = [
        "-e", "PYTHONHASHSEED=0",
        "-v", f"{snapshot_project}:/workspace:ro",
        "-v", f"{python_output}:/evidence/python:ro",
        "-v", f"{rust_output}:/evidence/rust:ro",
        "-w", "/workspace",
    ]
    command.extend(
        (
            "/usr/local/bin/python3", "-I", "-S", "-B", HOST_ENTRYPOINT,
            "--validate-dual",
            "--python-output-directory", "/evidence/python",
            "--rust-output-directory", "/evidence/rust",
        )
    )
    return command


@dataclass(frozen=True, slots=True)
class EndpointResult:
    implementation: str
    stdout: bytes
    report: Mapping[str, object]


def _parse_json(payload: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(FAIL_ENDPOINT, f"{label} did not emit one JSON object: {error}")
    if type(value) is not dict:
        _fail(FAIL_ENDPOINT, f"{label} emitted non-object JSON")
    return value


def _default_endpoint_runner(
    implementation: str, command: Sequence[str], timeout: int
) -> EndpointResult:
    result = _hardened_call(
        FAIL_ENDPOINT,
        hardened._run,
        command,
        timeout=timeout,
        code=hardened.FAIL_ENDPOINT,
    )
    assert isinstance(result, subprocess.CompletedProcess)
    return EndpointResult(
        implementation,
        result.stdout,
        _parse_json(result.stdout, implementation),
    )


def run_endpoints_parallel(
    commands: Mapping[str, Sequence[str]],
    *,
    timeout: int,
    runner: Callable[[str, Sequence[str], int], EndpointResult] = _default_endpoint_runner,
) -> dict[str, EndpointResult]:
    if set(commands) != {"python", "rust"}:
        _fail(FAIL_ENDPOINT, "parallel endpoint set must be exactly Python/Rust")
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="hegel-s3-enum") as pool:
        futures = {
            name: pool.submit(runner, name, commands[name], timeout)
            for name in ("python", "rust")
        }
        return {name: futures[name].result() for name in ("python", "rust")}


def _check_output_directory(path: Path, implementation: str) -> None:
    if not path.is_dir() or path.is_symlink():
        _fail(FAIL_OUTPUT, f"{implementation} output directory is invalid")
    observed = frozenset(item.name for item in path.iterdir())
    if observed != EXPECTED_OUTPUT_FILES or any(
        not item.is_file() or item.is_symlink() for item in path.iterdir()
    ):
        _fail(FAIL_OUTPUT, f"{implementation} output file set differs")


def validate_host_receipt(value: Mapping[str, object]) -> None:
    guards = {
        "schema_version": "hegel-m3-shrink3-dual-diagnostic-validation-receipt/1",
        "claim_level": CLAIM_LEVEL,
        "qualification_level": "DIAGNOSTIC_ONLY",
        "diagnostic_only": True,
        "authoritative_claim_allowed": False,
        "execution_state": "NOT_RUN",
        "formal_roots_generated": False,
        "formal_roots": None,
        "formal_state_transition_allowed": False,
        "dual_reports_equal": True,
        "dual_archive_bytes_equal": True,
        "host_strict_archive_replay_verified": True,
        "host_target_free_isolation_verified": True,
        "host_target_or_split_modules_loaded": False,
        "independence_scope": (
            "INDEPENDENT_OF_ENDPOINT_REPORTED_WITNESS_NOT_A_THIRD_IMPLEMENTATION"
        ),
        "typed_language_boundary_independently_derived": True,
        "archive_prefix_exact": True,
        "program_indices_verified": True,
        "program_binding_roots_verified": True,
        "binary_operator_registry_verified": True,
        "removed_binary_operator_absent_from_archive": True,
        "operator_id_compaction_performed": False,
        "automatic_operator_migration_performed": False,
        "chunk_framing_and_blob_hashes_verified": True,
        "bucket_accounting_verified": True,
        "target_roles_evaluated": False,
        "split_material_accessed": False,
        "secrets_accessed": False,
    }
    for field, expected in guards.items():
        if value.get(field) != expected:
            _fail(FAIL_AUTHORITY, f"host receipt guard differs: {field}")
    closure_status = value.get("closure_status")
    if closure_status not in {"DSL_TOO_LARGE", "COMPLETE"}:
        _fail(FAIL_HOST, "host receipt has no terminal diagnostic status")
    if closure_status == "DSL_TOO_LARGE":
        overflow_guards = {
            "canonical_program_count": 50_000,
            "raw_operator_application_count_scope": (
                "THROUGH_FULLY_CLOSED_BOUNDARY_BUCKET"
            ),
            "witness_adjacency_verified": True,
            "witness_closed_bucket_rank_verified": True,
            "post_witness_traversal_buckets_untouched": True,
        }
        for field, expected in overflow_guards.items():
            if value.get(field) != expected:
                _fail(FAIL_AUTHORITY, f"overflow receipt guard differs: {field}")
    elif value.get("raw_operator_application_count_scope") != (
        "THROUGH_FULLY_CLOSED_FRONTIER"
    ):
        _fail(FAIL_AUTHORITY, "complete receipt raw-count scope differs")


def _deterministic_output_archive(
    output: Path, staging: Path, host_payload: bytes
) -> str:
    host_path = staging / "host_replay_receipt.json"
    host_path.write_bytes(host_payload)
    archive_path = output / "diagnostic_outputs.tar"
    files = [
        *(
            path
            for role in ("python", "rust")
            for path in sorted((staging / role / "result").iterdir())
        ),
        host_path,
    ]
    with tarfile.open(archive_path, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in files:
            relative = path.relative_to(staging).as_posix()
            payload = path.read_bytes()
            info = tarfile.TarInfo(relative)
            info.size = len(payload)
            info.mode = 0o444
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            from io import BytesIO

            archive.addfile(info, BytesIO(payload))
    return sha256(archive_path.read_bytes()).hexdigest()


def _build_rust(
    snapshot_project: Path,
    cargo_seed: Path,
    volume: str,
    workers: int,
) -> str:
    if not cargo_seed.is_dir():
        _fail(FAIL_BUILD, f"verified Cargo seed is absent: {cargo_seed}")
    _hardened_call(
        FAIL_BUILD,
        hardened._run,
        rust_build_command(snapshot_project, cargo_seed, volume, workers),
        timeout=1800,
        code=hardened.FAIL_BUILD,
    )
    command = hardened._docker_common(
        RUST_IMAGE,
        seccomp_path=snapshot_project / "config/phase3_internal_actor_seccomp_v1.json",
        user="65534:65534",
    )
    command[-1:-1] = ["-v", f"{volume}:/cargo-target:ro"]
    command.extend(("/usr/bin/sha256sum", RUST_BINARY_PATH))
    result = _hardened_call(
        FAIL_BUILD, hardened._run, command, code=hardened.FAIL_BUILD
    )
    assert isinstance(result, subprocess.CompletedProcess)
    digest = result.stdout.decode("ascii", "strict").split()[0]
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        _fail(FAIL_BUILD, "Rust release binary digest is invalid")
    return digest


def _fresh_endpoint_mount(staging: Path, name: str) -> Path:
    path = staging / name
    path.mkdir(mode=0o777)
    # mkdir honours the host umask.  The isolated runtime UID 65534 must be
    # able to create its exclusive result directory in this dedicated mount.
    path.chmod(0o777)
    if stat.S_IMODE(path.stat().st_mode) != 0o777:
        _fail(FAIL_OUTPUT, f"{name} mount is not writable by the isolated actor")
    return path


def _run_supervisor(
    snapshot_project: Path,
    staging: Path,
    volume: str,
    *,
    timeout: int,
) -> tuple[dict[str, EndpointResult], bytes, dict[str, object]]:
    python_mount = _fresh_endpoint_mount(staging, "python-endpoint")
    rust_mount = _fresh_endpoint_mount(staging, "rust-endpoint")
    commands = {
        "python": python_endpoint_command(snapshot_project, python_mount),
        "rust": rust_endpoint_command(snapshot_project, rust_mount, volume),
    }
    endpoints = run_endpoints_parallel(commands, timeout=timeout)
    endpoint_outputs = {
        "python": python_mount / "result",
        "rust": rust_mount / "result",
    }
    for implementation, path in endpoint_outputs.items():
        _check_output_directory(path, implementation)
    # Freeze the host-owned bind parents for the entire untrusted replay
    # window.  Endpoint files are owned by isolated UID 65534 and mode 0644;
    # removing parent write permission prevents coordinated replacement.
    python_mount.chmod(0o555)
    rust_mount.chmod(0o555)
    staging.chmod(0o555)
    try:
        host = _hardened_call(
            FAIL_HOST,
            hardened._run,
            host_replay_command(
                snapshot_project,
                endpoint_outputs["python"],
                endpoint_outputs["rust"],
            ),
            timeout=timeout,
            code=hardened.FAIL_ENDPOINT,
        )
    finally:
        staging.chmod(0o700)
        python_mount.chmod(0o700)
        rust_mount.chmod(0o700)
    assert isinstance(host, subprocess.CompletedProcess)
    receipt = _parse_json(host.stdout, "host replay")
    validate_host_receipt(receipt)
    # Move the host-owned outer mounts, not the actor-owned result trees.
    # WSL correctly denies UID 1000 a cross-parent rename of a 0755 directory
    # owned by UID 65534, while the outer directories remain controller-owned.
    python_mount.rename(staging / "python")
    rust_mount.rename(staging / "rust")
    return endpoints, host.stdout, receipt


def _preserve_failed_output(
    requested_output: Path, error: SupervisorError
) -> Path | None:
    """Retain fail-closed material while leaving the requested path reusable."""

    try:
        path = requested_output.absolute()
        if not path.is_dir() or path.is_symlink():
            return None
        repository = REPOSITORY_ROOT.resolve()
        resolved = path.resolve()
        if resolved == repository or repository in resolved.parents:
            return None
        receipt = {
            "schema_version": "hegel-shrink3-dual-complete-failure/1",
            "status": "FAIL_CLOSED",
            "failure_code": error.code,
            "detail": error.detail,
            "execution_state": "NOT_RUN",
            "formal_roots_generated": False,
            "formal_roots": None,
            "formal_state_transition_allowed": False,
            "target_roles_evaluated": False,
            "split_material_accessed": False,
            "seeds_accessed": False,
            "keys_or_signatures_generated": False,
        }
        (path / "failure_receipt.json").write_bytes(
            _canonical_json(receipt) + b"\n"
        )
        retained = path.with_name(
            f"{path.name}.failed-{secrets.token_hex(8)}"
        )
        path.rename(retained)
        return retained
    except OSError:
        return None


def qualify(
    basis_commit: str,
    output_directory: Path,
    *,
    workers: int,
    cargo_registry: Path,
    timeout: int,
) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{40}", basis_commit) is None:
        _fail(FAIL_ARGUMENT, "basis commit must be a full lowercase SHA-1")
    if not 1 <= workers <= 16 or timeout < 60:
        _fail(FAIL_ARGUMENT, "workers must be 1..16 and timeout at least 60 seconds")
    output = _validate_external_output(output_directory)
    if _git("rev-parse", "--verify", f"{basis_commit}^{{commit}}") != basis_commit:
        _fail(FAIL_GIT, "basis commit does not resolve exactly")
    rows, source_root = _source_rows(basis_commit)
    archive = _git(
        "archive", "--format=tar", basis_commit, "--", *ARCHIVE_PATHS, binary=True
    )
    if type(archive) is not bytes:
        _fail(FAIL_GIT, "Git archive is not bytes")
    output.mkdir(mode=0o700)
    source_archive_path = output / "commit_bound_sources.tar"
    source_archive_path.write_bytes(archive)
    repository_binding = {
        "basis_commit": basis_commit,
        "basis_parent_commits": str(
            _git("show", "-s", "--format=%P", basis_commit)
        ).split(),
        "basis_subject": _git("show", "-s", "--format=%s", basis_commit),
        "project_tree_oid": _git("rev-parse", f"{basis_commit}:Hegel Machine"),
        "source_archive_sha256": sha256(archive).hexdigest(),
        "source_file_count": len(rows),
        "source_file_set_root": source_root,
        "source_files": rows,
    }
    (output / "repository_binding.json").write_bytes(
        _canonical_json(repository_binding) + b"\n"
    )
    volume = f"hegel-shrink3-enum-{basis_commit[:12]}"
    staging = output / ".run-work"
    staging.mkdir(mode=0o700)
    staging.chmod(0o777)
    with (
        tempfile.TemporaryDirectory(prefix="hegel-shrink3-enum-control-") as control,
        tempfile.TemporaryDirectory(prefix="hegel-shrink3-enum-snapshot-") as temporary,
    ):
        volume_created = False
        primary_error: BaseException | None = None
        try:
            daemon = _hardened_call(
                FAIL_DOCKER,
                hardened._initialize_docker_environment,
                Path(control),
            )
            python_image_id = _hardened_call(
                FAIL_DOCKER, hardened._inspect_image, PYTHON_IMAGE
            )
            rust_image_id = _hardened_call(
                FAIL_DOCKER, hardened._inspect_image, RUST_IMAGE
            )
            snapshot_root = Path(temporary)
            _hardened_call(
                FAIL_GIT, hardened._safe_extract_git_archive, archive, snapshot_root
            )
            _hardened_call(
                FAIL_GIT, hardened._validate_snapshot, snapshot_root, rows
            )
            snapshot_project = snapshot_root / "Hegel Machine"
            actor_profiles = _hardened_call(
                FAIL_DOCKER, hardened._profile_images, snapshot_project
            )
            build_profile = _load_build_profile(snapshot_project)
            cargo_seed_snapshot = Path(control) / "cargo-seed-snapshot"
            cargo_dependency_seed = create_verified_cargo_seed_snapshot_v1(
                cargo_registry,
                cargo_seed_snapshot,
                snapshot_project
                / "rust/m3_closure_enumerator_shrink3/Cargo.lock",
            )
            volume_receipt = _create_fresh_volume(volume, basis_commit)
            volume_created = True
            try:
                rust_binary_sha256 = _build_rust(
                    snapshot_project, cargo_seed_snapshot, volume, workers
                )
            finally:
                _restore_owner_writable_tree(cargo_seed_snapshot)
            endpoints, host_payload, host_receipt = _run_supervisor(
                snapshot_project, staging, volume, timeout=timeout
            )
            for role in ("python", "rust"):
                (staging / role).rename(output / role)
            (output / "host_replay_receipt.json").write_bytes(host_payload)
            diagnostic_archive_sha256 = _deterministic_output_archive(
                output, output, host_payload
            )
            staging.rmdir()
            summary: dict[str, object] = {
                "schema_version": SCHEMA,
                "status": STATUS_PASS,
                "claim_level": CLAIM_LEVEL,
                "qualification_level": "DIAGNOSTIC_ONLY",
                "diagnostic_only": True,
                "authoritative_claim_allowed": False,
                "execution_state": "NOT_RUN",
                "formal_roots_generated": False,
                "formal_roots": None,
                "formal_state_transition_allowed": False,
                "basis_commit": basis_commit,
                "repository_binding": repository_binding,
                "container_runtime": {
                    "python_image": PYTHON_IMAGE,
                    "python_image_id": python_image_id,
                    "rust_image": RUST_IMAGE,
                    "rust_image_id": rust_image_id,
                    "daemon_receipt": daemon,
                    "actor_profiles": actor_profiles,
                    "enumerator_build_profile": build_profile,
                    "cargo_dependency_seed": cargo_dependency_seed,
                    "target_volume": volume_receipt,
                    "target_volume_fresh": True,
                    "target_volume_removed_after_run": True,
                    "network": "none",
                    "pull_policy": "never",
                },
                "rust_release_binary_sha256": rust_binary_sha256,
                "parallel_endpoint_count": 2,
                "parallel_endpoint_execution": True,
                "python_endpoint_stdout_sha256": sha256(
                    endpoints["python"].stdout
                ).hexdigest(),
                "rust_endpoint_stdout_sha256": sha256(
                    endpoints["rust"].stdout
                ).hexdigest(),
                "host_replay_receipt": host_receipt,
                "retained_local_files": {
                    "commit_bound_sources": "commit_bound_sources.tar",
                    "repository_binding": "repository_binding.json",
                    "diagnostic_outputs_archive": "diagnostic_outputs.tar",
                    "diagnostic_outputs_archive_sha256": diagnostic_archive_sha256,
                    "host_replay_receipt": "host_replay_receipt.json",
                    "python_directory": "python/result",
                    "rust_directory": "rust/result",
                },
                "target_roles_evaluated": False,
                "split_material_accessed": False,
                "seeds_accessed": False,
                "keys_or_signatures_generated": False,
                "active_governance_changed": False,
            }
            summary["diagnostic_summary_hash"] = "sha256:" + sha256(
                SUMMARY_DOMAIN + b"\x00" + _canonical_json(summary)
            ).hexdigest()
        except BaseException as error:
            primary_error = error
            raise
        finally:
            cleanup_error: SupervisorError | None = None
            try:
                if volume_created:
                    _hardened_call(FAIL_CLEANUP, hardened._remove_volume, volume)
            except SupervisorError as error:
                cleanup_error = error
            finally:
                hardened._DOCKER_ENV = None
            if cleanup_error is not None:
                if isinstance(primary_error, SupervisorError):
                    raise SupervisorError(
                        primary_error.code,
                        f"{primary_error.detail}; cleanup failure: "
                        f"{cleanup_error.code}: {cleanup_error.detail}",
                    ) from primary_error
                if primary_error is not None:
                    raise SupervisorError(
                        FAIL_CLEANUP,
                        f"primary {type(primary_error).__name__}: "
                        f"{primary_error}; cleanup failure: "
                        f"{cleanup_error.code}: {cleanup_error.detail}",
                    ) from primary_error
                raise cleanup_error
    (output / "supervisor_summary.json").write_bytes(
        _canonical_json(summary) + b"\n"
    )
    return summary


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-commit", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--cargo-registry", type=Path, default=DEFAULT_CARGO_REGISTRY)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    try:
        summary = qualify(
            arguments.basis_commit,
            arguments.output_directory,
            workers=arguments.workers,
            cargo_registry=arguments.cargo_registry,
            timeout=arguments.timeout_seconds,
        )
    except SupervisorError as error:
        retained = _preserve_failed_output(arguments.output_directory, error)
        detail = error.detail
        if retained is not None:
            detail += f"; retained_failure_directory={retained}"
        sys.stderr.buffer.write(
            _canonical_json(
                {"status": "FAIL_CLOSED", "failure_code": error.code, "detail": detail}
            )
            + b"\n"
        )
        return 2
    except Exception as error:
        internal = SupervisorError(
            FAIL_INTERNAL, f"{type(error).__name__}: {error}"
        )
        retained = _preserve_failed_output(
            arguments.output_directory, internal
        )
        detail = internal.detail
        if retained is not None:
            detail += f"; retained_failure_directory={retained}"
        sys.stderr.buffer.write(
            _canonical_json(
                {
                    "status": "FAIL_CLOSED",
                    "failure_code": FAIL_INTERNAL,
                    "detail": detail,
                }
            )
            + b"\n"
        )
        return 2
    sys.stdout.buffer.write(_canonical_json(summary) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
