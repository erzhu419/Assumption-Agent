"""Dual exact-wire qualification for the Phase-3A M2.5 errata.

The Python and Rust endpoints independently construct the same public,
deterministic vector report.  The checked golden fixture is a third immutable
comparison point.  Passing this module authorizes an *external* custodian to
begin genesis preparation; it neither performs genesis nor advances an M3
gate or state.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
from types import MappingProxyType
from typing import Final

from .hashing import stable_hash
from .phase3_m25_errata_vectors_v1 import generate_errata_vector_report_v1
from .phase3_m25_external_v1 import (
    DualGoldenVerification,
    EXTERNAL_GENESIS_START_GUARD_FIELDS,
    assert_external_genesis_start_allowed,
    external_genesis_start_guard_report,
)
from .phase3_m25_secret_absence_v1 import (
    PASS_STATUS as SECRET_ABSENCE_PASS_STATUS,
    repository_genesis_secret_absence_report,
    validate_repository_genesis_secret_absence_report,
)
from .phase3_local_runtime_v1 import (
    LinuxLocalTemporaryDirectoryV1,
    LocalDockerControlPlaneV1,
    Phase3LocalRuntimeError,
    build_local_docker_daemon_identity_receipt_v1,
    local_docker_daemon_receipt_binding_v1,
    prepare_local_docker_control_plane_v1,
)
from .phase3_m25_replay_v1 import DEFAULT_RUST_BINARY
from .strict_cbor_v1 import content_hash


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
MACHINE_FREEZE_ID: Final = "hegel-freeze-p2b-p3-v1.1.2"
CHILD_DSL_ID: Final = "hegel-old-dsl-v1.1.0"
VECTOR_SCHEMA: Final = "hegel-phase3-m25-exact-wire-errata-vectors/1"
GOLDEN_SCHEMA: Final = "hegel-phase3-m25-exact-wire-errata-golden/1"
REPORT_SCHEMA: Final = "hegel-phase3-m25-exact-wire-errata-qualification/2"
ARTIFACT_KIND: Final = "DETERMINISTIC_CANDIDATE_NON_AUTHORITATIVE"
STATUS: Final = "DUAL_EXACT_WIRE_ERRATA_GOLDEN_PASS"
BINARY_PROVENANCE: Final = "DETACHED_COMMIT_A_PINNED_OFFLINE_OCI_SOURCE_BUILD"
IMPLEMENTATION_COMMIT_ROLE: Final = "DETERMINISTIC_IMPLEMENTATION_BASIS_COMMIT_A"
CLAIM_BOUNDARY: Final = (
    "Python and Rust, with Rust built and executed from a detached Commit-A "
    "source snapshot in a digest-pinned, network-disabled OCI image and a "
    "fresh Linux-local target, reproduce the checked public E1-E12 "
    "exact-wire vectors, candidate roots, record-tree roots, and negative "
    "error codes. The locally recorded toolchain receipt is not an external "
    "build attestation. The checksum-exact dependency snapshot and persisted "
    "default Rust binary are diagnostic implementation inputs only. A "
    "successful fresh replay permits an independently isolated "
    "external custodian to begin the separately governed genesis workflow; a "
    "stored artifact alone does not. No seed, key, signature, instantiation "
    "marker, formal root, Gate 15-24 pass, M3 execution identity, closure "
    "result, or NOT_RUN to RUNNING transition is created."
)

BASE_AMENDMENT_DOCUMENT: Final = (
    PROJECT_ROOT
    / "docs"
    / "Hegel_Machine_Phase3A_M25_Bit_Exact_Wire_Completion_Amendment.md"
)
ERRATA_RESOLUTION_DOCUMENT: Final = (
    PROJECT_ROOT
    / "docs"
    / "Hegel_Machine_Phase3A_M25_Exact_Wire_Errata_Resolution.md"
)
IMPLEMENTATION_ADDENDUM_DOCUMENT: Final = (
    PROJECT_ROOT
    / "docs"
    / "Hegel_Machine_Phase3A_M25_Implementation_Closure_Addendum_v1.md"
)
GOLDEN_VECTOR_PATH: Final = (
    PROJECT_ROOT / "golden_vectors" / "phase3_m25_errata_wire_v1.json"
)
CHECKED_REPORT_PATH: Final = (
    PROJECT_ROOT / "artifacts" / "phase3_m25_errata_qualification_v1.json"
)
RUST_CRATE_ROOT: Final = PROJECT_ROOT / "rust" / "formal_bridge_m25"
APPROVED_TOOLCHAIN_POLICY_PATH: Final = (
    PROJECT_ROOT / "config" / "phase3_m25_approved_local_rust_toolchain_v1.json"
)
RUNTIME_SECCOMP_PATH: Final = (
    PROJECT_ROOT / "config" / "phase3_internal_actor_seccomp_v1.json"
)
BUILD_SECCOMP_PATH: Final = (
    PROJECT_ROOT / "config" / "phase3_m3_offline_build_seccomp_v1.json"
)
RUST_IMAGE_REF: Final = (
    "rust@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89"
)
RUST_TOOLCHAIN_BIN: Final = (
    "/usr/local/rustup/toolchains/1.88.0-x86_64-unknown-linux-gnu/bin"
)
RUST_CARGO_PATH: Final = f"{RUST_TOOLCHAIN_BIN}/cargo"
RUSTC_PATH: Final = f"{RUST_TOOLCHAIN_BIN}/rustc"
CARGO_SNAPSHOT_DOMAIN: Final = "HEGEL/M25/CARGO_DEPENDENCY_SNAPSHOT/V1"
CARGO_ARCHIVE_CACHE_ROOT: Final = Path.home() / ".cargo" / "registry" / "cache"
RUNTIME_DOCKER_POLICY_ID: Final = "hegel-m25-errata-rust-runtime-docker-v1"
BUILD_DOCKER_POLICY_ID: Final = "hegel-m25-errata-rust-offline-build-docker-v1"
RUST_RUNTIME_ENVIRONMENT: Final = MappingProxyType(
    {
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": f"{RUST_TOOLCHAIN_BIN}:/usr/bin:/bin",
        "TZ": "UTC",
    }
)
RUST_BUILD_ENVIRONMENT: Final = MappingProxyType(
    {
        **RUST_RUNTIME_ENVIRONMENT,
        "CARGO_BUILD_JOBS": "1",
        "CARGO_HOME": "/tmp/cargo-home",
        "CARGO_INCREMENTAL": "0",
        "CARGO_NET_OFFLINE": "true",
        "CARGO_PROFILE_RELEASE_CODEGEN_UNITS": "1",
        "CARGO_TARGET_DIR": "/output",
        "RUSTC": RUSTC_PATH,
        "RUSTDOC": f"{RUST_TOOLCHAIN_BIN}/rustdoc",
        "SOURCE_DATE_EPOCH": "0",
    }
)

SOURCE_PATHS: Final = (
    "src/hegel_machine/__init__.py",
    "src/hegel_machine/hashing.py",
    "src/hegel_machine/strict_cbor_v1.py",
    "src/hegel_machine/phase3_m25_wire_v1.py",
    "src/hegel_machine/phase3_m25_errata_vectors_v1.py",
    "src/hegel_machine/phase3_m25_external_v1.py",
    "src/hegel_machine/phase3_m25_readiness_v1.py",
    "src/hegel_machine/phase3_m25_replay_v1.py",
    "src/hegel_machine/phase3_m25_secret_absence_v1.py",
    "src/hegel_machine/phase3_local_runtime_v1.py",
    "src/hegel_machine/phase3_m25_errata_qualification_v1.py",
    "src/hegel_machine/cli.py",
    "config/phase3_m25_approved_local_rust_toolchain_v1.json",
    "config/phase3_internal_actor_seccomp_v1.json",
    "config/phase3_m3_offline_build_seccomp_v1.json",
    "rust/formal_bridge_m25/Cargo.toml",
    "rust/formal_bridge_m25/Cargo.lock",
    "rust/formal_bridge_m25/src/lib.rs",
    "rust/formal_bridge_m25/src/main.rs",
    "golden_vectors/phase3_m25_errata_wire_v1.json",
    "docs/Hegel_Machine_Phase3A_M25_Bit_Exact_Wire_Completion_Amendment.md",
    "docs/Hegel_Machine_Phase3A_M25_Exact_Wire_Errata_Resolution.md",
    "docs/Hegel_Machine_Phase3A_M25_Implementation_Closure_Addendum_v1.md",
    "docs/Hegel_Machine_Phase3A_M25_Offline_OCI_Errata_Qualification_Engineering_v1.md",
    "tests/test_phase3_m25_errata_wire_v1.py",
    "tests/test_phase3_m25_errata_vectors_v1.py",
    "tests/test_phase3_m25_errata_qualification_v1.py",
    "tests/test_phase3_m25_external_v1.py",
    "tests/test_phase3_m25_foundation_v1.py",
    "tests/test_phase3_m25_readiness_v1.py",
    "tests/test_phase3_m25_secret_absence_v1.py",
    "tests/test_phase3_m25_wire_completion_v1.py",
)

REPORT_CORE_FIELDS: Final = frozenset(
    {"machine_freeze_id", "vector_schema", "objects", "record_trees", "guard_errors"}
)
RUST_RESPONSE_FIELDS: Final = REPORT_CORE_FIELDS | {"ok", "op"}
EXPECTED_VECTOR_COUNTS: Final = {"objects": 21, "record_trees": 8, "guard_errors": 15}
EXPECTED_AUTHORITY_BOUNDARY: Final = {
    "external_genesis_start_authorized": True,
    "external_genesis_started": False,
    "authoritative_root_generation": False,
    "formal_roots_generated": False,
    "seed_genesis_performed": False,
    "real_key_generated": False,
    "signature_claim": False,
    "m3_gate_delta": 0,
    "m3_gates_before": 14,
    "m3_gates_after": 14,
    "child_state": "NOT_RUN",
    "m3_entry_qualified": False,
    "m3_start_authorized": False,
    "m3_run_started": False,
    "checked_artifact_replay_alone_sufficient": False,
    "fresh_dual_replay_required_for_external_use": True,
}


class M25ErrataQualificationError(RuntimeError):
    """Fail-closed exact-wire replay or evidence validation error."""


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise M25ErrataQualificationError(f"missing qualification input: {path}")
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise M25ErrataQualificationError(f"{name} must be a string-keyed object")
    return value


def _json_type_strict_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _json_type_strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_type_strict_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def _source_bindings() -> dict[str, str]:
    return {
        relative: _sha256_file(PROJECT_ROOT / relative)
        for relative in SOURCE_PATHS
    }


def _document_bindings() -> dict[str, str]:
    return {
        "BASE_AMENDMENT": _sha256_file(BASE_AMENDMENT_DOCUMENT),
        "ERRATA_RESOLUTION": _sha256_file(ERRATA_RESOLUTION_DOCUMENT),
        "IMPLEMENTATION_CLOSURE_ADDENDUM": _sha256_file(
            IMPLEMENTATION_ADDENDUM_DOCUMENT
        ),
    }


def _require_commit_id(value: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise M25ErrataQualificationError(
            "implementation basis commit must be a lowercase 40-hex Git SHA-1"
        )
    return value


def _git_completed(
    repository_root: Path,
    arguments: list[str],
    *,
    timeout: int = 120,
) -> subprocess.CompletedProcess[bytes]:
    """Run a Git read with no ambient config, object, or replace injection."""

    try:
        return subprocess.run(
            ["/usr/bin/git", *arguments],
            cwd=repository_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
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
        raise M25ErrataQualificationError(
            f"Git read failed to execute: {type(exc).__name__}"
        ) from exc


def repository_head_commit() -> str:
    completed = _git_completed(PROJECT_ROOT.parent, ["rev-parse", "HEAD"], timeout=30)
    if completed.returncode != 0:
        raise M25ErrataQualificationError(
            "cannot resolve deterministic implementation basis commit"
        )
    return _require_commit_id(completed.stdout.decode("ascii", "strict").strip())


def validate_errata_qualification_output_path(path: Path) -> Path:
    """Allow in-repository publication only at the dedicated artifact path."""

    if not isinstance(path, Path):
        raise TypeError("qualification output path must be pathlib.Path")
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return resolved
    if resolved != CHECKED_REPORT_PATH.resolve(strict=False):
        raise M25ErrataQualificationError(
            "in-repository qualification output must use the dedicated artifact path"
        )
    return resolved


def _repository_root() -> Path:
    completed = _git_completed(
        PROJECT_ROOT.parent, ["rev-parse", "--show-toplevel"], timeout=30
    )
    if completed.returncode != 0:
        raise M25ErrataQualificationError("cannot resolve repository root")
    return Path(completed.stdout.decode("utf-8", "strict").strip()).resolve()


def _assert_sources_match_commit(commit_id: str) -> None:
    """Prove Commit A contains every byte claimed by the qualification.

    Merely naming ``HEAD`` would allow dirty worktree bytes to masquerade as
    commit-bound evidence.  Read every bound blob back from Git and compare it
    byte-for-byte with the replay input before setting the Commit-A guard.
    """

    repository_root = _repository_root()
    ancestry = _git_completed(
        repository_root,
        ["merge-base", "--is-ancestor", commit_id, "HEAD"],
        timeout=30,
    )
    if ancestry.returncode != 0:
        raise M25ErrataQualificationError(
            "implementation basis commit is not an ancestor of current HEAD"
        )
    paths = tuple(dict.fromkeys(SOURCE_PATHS))
    for relative in paths:
        worktree_path = PROJECT_ROOT / relative
        try:
            repository_relative = worktree_path.resolve().relative_to(repository_root)
        except ValueError as exc:
            raise M25ErrataQualificationError(
                f"qualification source escapes repository: {relative}"
            ) from exc
        completed = _git_completed(
            repository_root,
            ["show", f"{commit_id}:{repository_relative.as_posix()}"],
            timeout=30,
        )
        if completed.returncode != 0:
            raise M25ErrataQualificationError(
                f"Commit A does not contain qualification source: {relative}"
            )
        if completed.stdout != worktree_path.read_bytes():
            raise M25ErrataQualificationError(
                f"qualification source differs from Commit A: {relative}"
            )


def _tool_version(
    executable: str,
    label: str,
    *,
    environment: Mapping[str, str],
    cwd: Path,
) -> str:
    completed = subprocess.run(
        [executable, "--version"],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
        env=dict(environment),
    )
    if completed.returncode != 0:
        raise M25ErrataQualificationError(f"cannot resolve {label} version")
    version = completed.stdout.strip()
    if not version or "\n" in version:
        raise M25ErrataQualificationError(f"invalid {label} version output")
    return version


def _container_environment_sha256(environment: Mapping[str, str]) -> str:
    if not environment or any(
        type(key) is not str
        or re.fullmatch(r"[A-Z][A-Z0-9_]*", key) is None
        or type(value) is not str
        or "\x00" in value
        for key, value in environment.items()
    ):
        raise M25ErrataQualificationError("container environment is malformed")
    payload = json.dumps(
        dict(environment),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
    ):
        raise M25ErrataQualificationError(f"{label} is not an exact SHA-256")
    return value


def _load_approved_toolchain_policy() -> dict[str, object]:
    policy = dict(
        _mapping(
            json.loads(APPROVED_TOOLCHAIN_POLICY_PATH.read_text(encoding="utf-8")),
            "approved local Rust OCI toolchain policy",
        )
    )
    expected_fields = {
        "schema_version",
        "authority_boundary",
        "image_ref",
        "oci_manifest_digest",
        "image_id",
        "operating_system",
        "architecture",
        "cargo_binary_path",
        "cargo_binary_sha256",
        "cargo_version_stdout_sha256",
        "cargo_version",
        "rustc_binary_path",
        "rustc_binary_sha256",
        "rustc_version",
        "rustc_verbose_version_stdout_sha256",
        "runtime_environment_sha256",
        "build_environment_sha256",
        "runtime_seccomp_sha256",
        "build_seccomp_sha256",
        "cargo_lock_sha256",
        "cargo_lock_registry_package_count",
        "dependency_snapshot_domain",
        "dependency_snapshot_root",
        "dependency_snapshot_file_count",
        "host_cargo_cache_mounted_into_container",
        "required_docker_flags",
    }
    if set(policy) != expected_fields:
        raise M25ErrataQualificationError(
            "approved local Rust OCI toolchain policy field-set drift"
        )
    if (
        policy["schema_version"]
        != "hegel-phase3-m25-approved-local-rust-oci-toolchain/2"
        or policy["authority_boundary"]
        != "LOCAL_DETERMINISTIC_BUILD_POLICY_NOT_EXTERNAL_ATTESTATION"
        or policy["image_ref"] != RUST_IMAGE_REF
        or policy["oci_manifest_digest"] != RUST_IMAGE_REF.rsplit("@", 1)[1]
        or policy["operating_system"] != "linux"
        or policy["architecture"] != "amd64"
        or policy["cargo_binary_path"] != RUST_CARGO_PATH
        or policy["rustc_binary_path"] != RUSTC_PATH
        or policy["dependency_snapshot_domain"] != CARGO_SNAPSHOT_DOMAIN
        or policy["host_cargo_cache_mounted_into_container"] is not False
        or policy["required_docker_flags"]
        != [
            "--pull=never",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
        ]
    ):
        raise M25ErrataQualificationError(
            "approved local Rust OCI toolchain policy identity drift"
        )
    for field in (
        "oci_manifest_digest",
        "image_id",
        "cargo_binary_sha256",
        "cargo_version_stdout_sha256",
        "rustc_binary_sha256",
        "rustc_verbose_version_stdout_sha256",
        "runtime_environment_sha256",
        "build_environment_sha256",
        "runtime_seccomp_sha256",
        "build_seccomp_sha256",
        "cargo_lock_sha256",
        "dependency_snapshot_root",
    ):
        _require_sha256(policy[field], f"approved Rust OCI policy {field}")
    for field in ("cargo_version", "rustc_version"):
        if type(policy[field]) is not str or not policy[field] or "\n" in policy[field]:
            raise M25ErrataQualificationError(
                f"approved local Rust OCI toolchain {field} drift"
            )
    for field in (
        "cargo_lock_registry_package_count",
        "dependency_snapshot_file_count",
    ):
        if type(policy[field]) is not int or int(policy[field]) <= 0:
            raise M25ErrataQualificationError(
                f"approved local Rust OCI toolchain {field} drift"
            )
    if policy["runtime_environment_sha256"] != _container_environment_sha256(
        RUST_RUNTIME_ENVIRONMENT
    ) or policy["build_environment_sha256"] != _container_environment_sha256(
        RUST_BUILD_ENVIRONMENT
    ):
        raise M25ErrataQualificationError("approved container environment drift")
    if policy["runtime_seccomp_sha256"] != _sha256_file(
        RUNTIME_SECCOMP_PATH
    ) or policy["build_seccomp_sha256"] != _sha256_file(BUILD_SECCOMP_PATH):
        raise M25ErrataQualificationError("approved container seccomp policy drift")
    return policy


def _docker_command(
    control_plane: LocalDockerControlPlaneV1,
    image: str,
    options: tuple[str, ...],
    command: tuple[str, ...],
    *,
    seccomp_path: Path,
    container_environment: Mapping[str, str],
    user: str = "65534:65534",
) -> list[str]:
    try:
        user_id, group_id = user.split(":", 1)
        int(user_id)
        int(group_id)
    except (AttributeError, TypeError, ValueError) as exc:
        raise M25ErrataQualificationError("container user must be numeric uid:gid") from exc
    _container_environment_sha256(container_environment)
    exact_environment = tuple(
        f"{key}={container_environment[key]}" for key in sorted(container_environment)
    )
    return control_plane.command(
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        f"--security-opt=seccomp={seccomp_path}",
        "--user=" + user,
        "--pids-limit=64",
        "--memory=512m",
        "--memory-swap=512m",
        "--ulimit=nofile=128:128",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m,"
        f"uid={user_id},gid={group_id},mode=0700",
        *options,
        "--entrypoint=/usr/bin/env",
        image,
        "-i",
        *exact_environment,
        *command,
    )


def _run_bytes(
    command: list[str],
    *,
    environment: Mapping[str, str],
    timeout: int,
    input_payload: bytes | None = None,
    label: str,
) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
            command,
            input=input_payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
            env=dict(environment),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise M25ErrataQualificationError(f"{label} failed: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace")[-2000:]
        raise M25ErrataQualificationError(f"{label} failed: {detail}")
    return completed


def _qualify_local_docker_control_plane(
    control_plane: LocalDockerControlPlaneV1,
) -> tuple[dict[str, object], str]:
    version = _run_bytes(
        control_plane.command("version", "--format", "{{json .}}"),
        environment=control_plane.environment,
        timeout=30,
        label="local Docker version probe",
    )
    info = _run_bytes(
        control_plane.command("info", "--format", "{{json .}}"),
        environment=control_plane.environment,
        timeout=30,
        label="local Docker daemon probe",
    )
    try:
        version_value = _mapping(json.loads(version.stdout), "Docker version identity")
        info_value = _mapping(json.loads(info.stdout), "Docker daemon identity")
        receipt = build_local_docker_daemon_identity_receipt_v1(
            control_plane,
            version_payload=version_value,
            info_payload=info_value,
            repository_root=PROJECT_ROOT.parent,
        )
        binding = local_docker_daemon_receipt_binding_v1(receipt).hex()
    except (json.JSONDecodeError, Phase3LocalRuntimeError) as exc:
        raise M25ErrataQualificationError(
            f"local Docker control-plane qualification failed: {exc}"
        ) from exc
    return receipt, binding


def _approved_rust_toolchain(
    control_plane: LocalDockerControlPlaneV1,
) -> tuple[dict[str, object], dict[str, object]]:
    """Qualify the digest-pinned Rust OCI toolchain without host Rust."""

    policy = _load_approved_toolchain_policy()
    inspect = _run_bytes(
        control_plane.command("image", "inspect", RUST_IMAGE_REF, "--format", "{{json .}}"),
        environment=control_plane.environment,
        timeout=30,
        label="approved Rust OCI image inspection",
    )
    try:
        image = dict(_mapping(json.loads(inspect.stdout), "Rust OCI image inspection"))
    except json.JSONDecodeError as exc:
        raise M25ErrataQualificationError("Rust OCI image inspection is not JSON") from exc
    descriptor = image.get("Descriptor")
    if not isinstance(descriptor, Mapping):
        raise M25ErrataQualificationError("Rust OCI descriptor is absent")
    repo_digests = image.get("RepoDigests")
    if (
        image.get("Id") != policy["image_id"]
        or image.get("Os") != policy["operating_system"]
        or image.get("Architecture") != policy["architecture"]
        or descriptor.get("digest") != policy["oci_manifest_digest"]
        or not isinstance(repo_digests, list)
        or RUST_IMAGE_REF not in repo_digests
    ):
        raise M25ErrataQualificationError("approved Rust OCI image identity drift")

    hashes = _run_bytes(
        _docker_command(
            control_plane,
            RUST_IMAGE_REF,
            (),
            ("/usr/bin/sha256sum", RUST_CARGO_PATH, RUSTC_PATH),
            seccomp_path=RUNTIME_SECCOMP_PATH,
            container_environment=RUST_RUNTIME_ENVIRONMENT,
        ),
        environment=control_plane.environment,
        timeout=60,
        label="Rust OCI binary digest probe",
    ).stdout.decode("ascii")
    digest_rows: dict[str, str] = {}
    for line in hashes.splitlines():
        digest, separator, path = line.partition("  ")
        if not separator or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise M25ErrataQualificationError("Rust OCI binary digest output drift")
        digest_rows[path] = "sha256:" + digest
    if digest_rows != {
        RUST_CARGO_PATH: policy["cargo_binary_sha256"],
        RUSTC_PATH: policy["rustc_binary_sha256"],
    }:
        raise M25ErrataQualificationError("Rust OCI tool binary digest drift")

    cargo_probe = _run_bytes(
        _docker_command(
            control_plane,
            RUST_IMAGE_REF,
            (),
            (RUST_CARGO_PATH, "--version"),
            seccomp_path=RUNTIME_SECCOMP_PATH,
            container_environment=RUST_RUNTIME_ENVIRONMENT,
        ),
        environment=control_plane.environment,
        timeout=60,
        label="Rust OCI cargo version probe",
    ).stdout
    rustc_probe = _run_bytes(
        _docker_command(
            control_plane,
            RUST_IMAGE_REF,
            (),
            (RUSTC_PATH, "-vV"),
            seccomp_path=RUNTIME_SECCOMP_PATH,
            container_environment=RUST_RUNTIME_ENVIRONMENT,
        ),
        environment=control_plane.environment,
        timeout=60,
        label="Rust OCI compiler probe",
    ).stdout
    if (
        cargo_probe.decode("utf-8").strip() != policy["cargo_version"]
        or rustc_probe.decode("utf-8").splitlines()[0] != policy["rustc_version"]
        or "host: x86_64-unknown-linux-gnu\n" not in rustc_probe.decode("utf-8")
        or "release: 1.88.0\n" not in rustc_probe.decode("utf-8")
        or "sha256:" + hashlib.sha256(cargo_probe).hexdigest()
        != policy["cargo_version_stdout_sha256"]
        or "sha256:" + hashlib.sha256(rustc_probe).hexdigest()
        != policy["rustc_verbose_version_stdout_sha256"]
    ):
        raise M25ErrataQualificationError("Rust OCI compiler probe drift")
    receipt = {
        "image_ref": RUST_IMAGE_REF,
        "image_id": image["Id"],
        "oci_manifest_digest": descriptor["digest"],
        "operating_system": image["Os"],
        "architecture": image["Architecture"],
        "cargo_binary_path": RUST_CARGO_PATH,
        "cargo_binary_sha256": digest_rows[RUST_CARGO_PATH],
        "cargo_version": cargo_probe.decode("utf-8").strip(),
        "cargo_version_stdout_sha256": "sha256:" + hashlib.sha256(cargo_probe).hexdigest(),
        "rustc_binary_path": RUSTC_PATH,
        "rustc_binary_sha256": digest_rows[RUSTC_PATH],
        "rustc_version": rustc_probe.decode("utf-8").splitlines()[0],
        "rustc_verbose_version_stdout_sha256": "sha256:" + hashlib.sha256(rustc_probe).hexdigest(),
        "runtime_environment_sha256": _container_environment_sha256(RUST_RUNTIME_ENVIRONMENT),
        "build_environment_sha256": _container_environment_sha256(RUST_BUILD_ENVIRONMENT),
        "runtime_seccomp_sha256": _sha256_file(RUNTIME_SECCOMP_PATH),
        "build_seccomp_sha256": _sha256_file(BUILD_SECCOMP_PATH),
        "image_config_environment_ignored": True,
        "pull_policy": "never",
        "network_mode": "none",
        "toolchain_receipt_is_external_attestation": False,
    }
    return policy, receipt


def _archive_commit_a_sources(commit_id: str, destination: Path) -> Path:
    """Materialize only bound Commit-A blobs in a private detached snapshot."""

    repository_root = _repository_root()
    destination.mkdir(parents=True, exist_ok=False)
    project_relative = PROJECT_ROOT.resolve().relative_to(repository_root)
    repository_paths = [
        (project_relative / relative).as_posix()
        for relative in dict.fromkeys(SOURCE_PATHS)
    ]
    completed = _git_completed(
        repository_root,
        ["archive", "--format=tar", commit_id, "--", *repository_paths],
        timeout=120,
    )
    if completed.returncode != 0:
        raise M25ErrataQualificationError("cannot materialize Commit-A snapshot")
    destination_resolved = destination.resolve()
    with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:") as archive:
        members = archive.getmembers()
        if not members:
            raise M25ErrataQualificationError("Commit-A source archive is empty")
        for member in members:
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise M25ErrataQualificationError("unsafe path in Commit-A archive")
            if not (member.isdir() or member.isfile()):
                raise M25ErrataQualificationError(
                    "Commit-A source archive contains a non-file entry"
                )
            target = (destination / member_path).resolve()
            try:
                target.relative_to(destination_resolved)
            except ValueError as exc:
                raise M25ErrataQualificationError(
                    "Commit-A archive entry escapes destination"
                ) from exc
        archive.extractall(destination)
    snapshot_project = destination / project_relative
    if not snapshot_project.is_dir():
        raise M25ErrataQualificationError("Commit-A snapshot project is missing")
    return snapshot_project


def _cargo_lock_registry_packages(cargo_lock: Path) -> tuple[tuple[str, str, str], ...]:
    """Return the registry package archive identities frozen by Cargo.lock.

    Python 3.10 has no stdlib TOML reader.  Cargo package names and versions
    have a deliberately narrow lexical form, so parsing just the three exact
    string fields needed here is both smaller and stricter than accepting an
    optional ambient TOML dependency.
    """

    text = cargo_lock.read_text(encoding="utf-8")
    packages: list[tuple[str, str, str]] = []
    identities: set[tuple[str, str]] = set()
    for section in text.split("[[package]]")[1:]:
        fields: dict[str, str] = {}
        for field in ("name", "version", "source", "checksum"):
            match = re.search(rf'^\s*{field}\s*=\s*"([^"]+)"\s*$', section, re.MULTILINE)
            if match is not None:
                fields[field] = match.group(1)
        source = fields.get("source")
        if source is None:
            continue
        if source != "registry+https://github.com/rust-lang/crates.io-index":
            raise M25ErrataQualificationError(
                "Cargo.lock contains an unsupported non-registry dependency"
            )
        name = fields.get("name")
        version = fields.get("version")
        checksum = fields.get("checksum")
        if (
            name is None
            or version is None
            or checksum is None
            or re.fullmatch(r"[A-Za-z0-9_-]+", name) is None
            or re.fullmatch(r"[0-9A-Za-z.+-]+", version) is None
            or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
        ):
            raise M25ErrataQualificationError(
                "Cargo.lock registry package identity is not exact"
            )
        identity = (name, version)
        if identity in identities:
            raise M25ErrataQualificationError(
                f"Cargo.lock repeats registry package identity {name} {version}"
            )
        identities.add(identity)
        packages.append((name, version, checksum))
    result = tuple(sorted(packages))
    if not result or len(result) != len(set(result)):
        raise M25ErrataQualificationError(
            "Cargo.lock registry package set is empty or non-unique"
        )
    return result


def _cached_crate_path(name: str, version: str, checksum: str) -> Path:
    """Resolve one byte-exact archive without exposing Cargo home to Docker."""

    cache_root = CARGO_ARCHIVE_CACHE_ROOT
    if not cache_root.is_dir():
        raise M25ErrataQualificationError("offline Cargo crate cache is unavailable")
    candidates = sorted(cache_root.glob(f"*/{name}-{version}.crate"))
    matching = [
        candidate
        for candidate in candidates
        if candidate.is_file()
        and not candidate.is_symlink()
        and hashlib.sha256(candidate.read_bytes()).hexdigest() == checksum
    ]
    if not matching:
        raise M25ErrataQualificationError(
            f"no checksum-exact offline crate for {name} {version}"
        )
    return matching[0]


def _extract_locked_crate(
    archive_path: Path,
    vendor_root: Path,
    *,
    name: str,
    version: str,
    package_checksum: str,
) -> tuple[tuple[object, ...], ...]:
    top = f"{name}-{version}"
    destination_root = vendor_root / top
    destination_root.mkdir(mode=0o700, exist_ok=False)
    rows: list[tuple[object, ...]] = []
    seen: set[str] = set()
    total_size = 0
    try:
        archive = tarfile.open(archive_path, mode="r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise M25ErrataQualificationError(
            f"cannot open cached crate {name} {version}: {exc}"
        ) from exc
    with archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or not path.parts
                or path.parts[0] != top
                or any(part in {"", ".", ".."} for part in path.parts)
            ):
                raise M25ErrataQualificationError(
                    f"unsafe crate archive path: {member.name!r}"
                )
            relative = PurePosixPath(*path.parts[1:])
            relative_text = relative.as_posix()
            if not relative_text or relative_text in seen:
                raise M25ErrataQualificationError(
                    f"duplicate or empty crate archive path in {top}"
                )
            seen.add(relative_text)
            destination = destination_root.joinpath(*relative.parts)
            if member.isdir():
                destination.mkdir(parents=True, mode=0o700, exist_ok=True)
                continue
            if not member.isfile() or member.size < 0 or member.size > 16 * 1024 * 1024:
                raise M25ErrataQualificationError(
                    f"unsupported crate archive member: {member.name!r}"
                )
            total_size += member.size
            if total_size > 128 * 1024 * 1024:
                raise M25ErrataQualificationError(f"cached crate {top} is too large")
            source = archive.extractfile(member)
            if source is None:
                raise M25ErrataQualificationError(
                    f"cannot extract crate archive member: {member.name!r}"
                )
            payload = source.read(member.size + 1)
            if len(payload) != member.size:
                raise M25ErrataQualificationError(
                    f"crate archive member length drift: {member.name!r}"
                )
            destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            executable = bool(member.mode & 0o111)
            destination.chmod(0o555 if executable else 0o444)
            rows.append(
                (
                    name.encode("utf-8"),
                    version.encode("ascii"),
                    bytes.fromhex(package_checksum),
                    relative_text.encode("utf-8"),
                    hashlib.sha256(payload).digest(),
                    len(payload),
                    executable,
                )
            )
    file_checksums = {
        row[3].decode("utf-8"): row[4].hex()
        for row in rows
        if row[3] != b".cargo-checksum.json"
    }
    checksum_payload = json.dumps(
        {"files": file_checksums, "package": package_checksum},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    checksum_path = destination_root / ".cargo-checksum.json"
    if checksum_path.exists():
        checksum_path.chmod(0o600)
        checksum_path.unlink()
        rows = [row for row in rows if row[3] != b".cargo-checksum.json"]
    with checksum_path.open("xb") as handle:
        handle.write(checksum_payload)
        handle.flush()
        os.fsync(handle.fileno())
    checksum_path.chmod(0o444)
    rows.append(
        (
            name.encode("utf-8"),
            version.encode("ascii"),
            bytes.fromhex(package_checksum),
            b".cargo-checksum.json",
            hashlib.sha256(checksum_payload).digest(),
            len(checksum_payload),
            False,
        )
    )
    for directory, _, _ in os.walk(destination_root, topdown=False):
        Path(directory).chmod(0o555)
    return tuple(sorted(rows, key=lambda row: row[3]))


def _build_cargo_dependency_snapshot(
    cargo_lock: Path,
    vendor_root: Path,
) -> tuple[str, int, int]:
    vendor_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    packages = _cargo_lock_registry_packages(cargo_lock)
    rows: list[tuple[object, ...]] = []
    for name, version, checksum in packages:
        rows.extend(
            _extract_locked_crate(
                _cached_crate_path(name, version, checksum),
                vendor_root,
                name=name,
                version=version,
                package_checksum=checksum,
            )
        )
    vendor_root.chmod(0o700)
    typed_packages = tuple(
        (name.encode("utf-8"), version.encode("ascii"), bytes.fromhex(checksum))
        for name, version, checksum in packages
    )
    root = content_hash(CARGO_SNAPSHOT_DOMAIN, (1, typed_packages, tuple(rows)))
    return "sha256:" + root.hex(), len(rows), len(packages)


def _sha256_directory_manifest(root: Path) -> str:
    digest = hashlib.sha256()
    digest.update(b"HEGEL/M25/DIRECTORY_MANIFEST/V1\x00")
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(hashlib.sha256(payload).digest())
    return "sha256:" + digest.hexdigest()


def _make_snapshot_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        path.chmod(0o555)
    root.chmod(0o555)


def _snapshot_python_report(
    snapshot_project: Path,
    *,
    environment: Mapping[str, str],
) -> tuple[dict[str, object], dict[str, object]]:
    python = str(Path(sys.executable).resolve(strict=True))
    script = """
import json
import sys
import types

package = types.ModuleType("hegel_machine")
package.__package__ = "hegel_machine"
package.__path__ = [sys.argv[1] + "/hegel_machine"]
sys.modules["hegel_machine"] = package
from hegel_machine.phase3_m25_errata_vectors_v1 import generate_errata_vector_report_v1

print(json.dumps(generate_errata_vector_report_v1(), sort_keys=True, separators=(",", ":")))
"""
    completed = subprocess.run(
        [python, "-I", "-c", script, str(snapshot_project / "src")],
        cwd=snapshot_project,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
        env=dict(environment),
    )
    if completed.returncode != 0:
        raise M25ErrataQualificationError(
            "detached Commit-A Python replay failed: " + completed.stderr
        )
    try:
        response = dict(
            _mapping(json.loads(completed.stdout), "detached Python response")
        )
    except json.JSONDecodeError as exc:
        raise M25ErrataQualificationError(
            "detached Commit-A Python replay returned invalid JSON"
        ) from exc
    if set(response) != RUST_RESPONSE_FIELDS:
        raise M25ErrataQualificationError(
            "detached Python response field-set drift"
        )
    if response.get("ok") is not True or response.get("op") != "errata_vectors":
        raise M25ErrataQualificationError("detached Python response envelope drift")
    report = _validate_vector_report(
        {field: response[field] for field in REPORT_CORE_FIELDS},
        "detached Python errata report",
    )
    execution = {
        "execution_mode": "DETACHED_COMMIT_A_SOURCE_REPLAY",
        "source_commit": None,
        "python_version": _tool_version(
            python,
            "Python",
            environment=environment,
            cwd=snapshot_project,
        ),
        "python_executable_sha256": _sha256_file(Path(python)),
        "isolated_mode": True,
        "minimal_module_closure": True,
        "package_init_executed": False,
        "source_blobs_from_git_archive": True,
        "working_tree_executed": False,
        "execution_receipt_is_external_attestation": False,
    }
    return report, execution


def _controlled_python_environment(temporary_root: Path) -> dict[str, str]:
    home = temporary_root / "python-home"
    process_temp = temporary_root / "python-tmp"
    home.mkdir(mode=0o700)
    process_temp.mkdir(mode=0o700)
    return {
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "SOURCE_DATE_EPOCH": "0",
        "TMPDIR": str(process_temp),
        "TZ": "UTC",
    }


def _run_rust_report_in_container(
    control_plane: LocalDockerControlPlaneV1,
    rust_binary: Path,
) -> tuple[dict[str, object], str]:
    try:
        metadata = rust_binary.lstat()
    except OSError as exc:
        raise M25ErrataQualificationError(f"Rust replay binary is absent: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise M25ErrataQualificationError("Rust replay binary is not a regular file")
    before = _sha256_file(rust_binary)
    completed = _run_bytes(
        _docker_command(
            control_plane,
            RUST_IMAGE_REF,
            (
                "--interactive",
                "-v",
                f"{rust_binary}:/opt/hegel-formal-bridge-m25:ro",
            ),
            ("/opt/hegel-formal-bridge-m25",),
            seccomp_path=RUNTIME_SECCOMP_PATH,
            container_environment=RUST_RUNTIME_ENVIRONMENT,
        ),
        environment=control_plane.environment,
        timeout=120,
        input_payload=b'{"op":"errata_vectors"}',
        label="isolated Rust errata replay",
    )
    after = _sha256_file(rust_binary)
    if before != after:
        raise M25ErrataQualificationError("Rust replay binary changed during execution")
    return (
        _parse_rust_report(
            stdout=completed.stdout.decode("utf-8"),
            stderr=completed.stderr.decode("utf-8", "replace"),
            returncode=completed.returncode,
        ),
        before,
    )


def _persist_validated_rust_binary(payload: bytes, expected_digest: str) -> dict[str, object]:
    """Atomically publish only already golden-validated binary bytes."""

    actual = "sha256:" + hashlib.sha256(payload).hexdigest()
    if actual != expected_digest:
        raise M25ErrataQualificationError("validated Rust binary payload digest drift")
    destination = DEFAULT_RUST_BINARY.absolute()
    try:
        destination.relative_to(RUST_CRATE_ROOT.absolute())
    except ValueError as exc:
        raise M25ErrataQualificationError("default Rust binary escapes crate root") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.resolve(strict=True) != destination.parent:
        raise M25ErrataQualificationError("default Rust binary parent contains a symlink")
    if destination.is_symlink():
        raise M25ErrataQualificationError("default Rust binary is a symlink")
    descriptor, pending_name = tempfile.mkstemp(
        prefix=".hegel-formal-bridge-m25.pending-",
        dir=destination.parent,
    )
    pending = Path(pending_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        pending.chmod(0o755)
        if _sha256_file(pending) != expected_digest:
            raise M25ErrataQualificationError("pending Rust binary digest drift")
        os.replace(pending, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if pending.exists():
            pending.unlink()
    if destination.is_symlink() or _sha256_file(destination) != expected_digest:
        raise M25ErrataQualificationError("persisted Rust binary digest drift")
    if stat.S_IMODE(destination.stat().st_mode) != 0o755:
        raise M25ErrataQualificationError("persisted Rust binary mode drift")
    return {
        "default_rust_binary_repository_path": destination.relative_to(PROJECT_ROOT).as_posix(),
        "persisted_binary_sha256": expected_digest,
        "persisted_binary_mode_octal": "0755",
        "persisted_binary_atomic_replace": True,
        "persisted_binary_is_symlink": False,
    }


def _fresh_commit_a_rust_replay(
    commit_id: str,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    """Replay both endpoints from one detached Commit-A source snapshot."""

    toolchain_policy_sha256 = _sha256_file(APPROVED_TOOLCHAIN_POLICY_PATH)
    _assert_sources_match_commit(commit_id)
    try:
        temporary_owner = LinuxLocalTemporaryDirectoryV1(
            prefix="hegel-m25-commit-a-build-",
            repository_root=PROJECT_ROOT.parent,
        )
    except Phase3LocalRuntimeError as exc:
        raise M25ErrataQualificationError(str(exc)) from exc
    with temporary_owner as temporary:
        temporary_root = Path(temporary)
        try:
            control_plane = prepare_local_docker_control_plane_v1(
                temporary_root,
                repository_root=PROJECT_ROOT.parent,
            )
        except Phase3LocalRuntimeError as exc:
            raise M25ErrataQualificationError(str(exc)) from exc
        daemon_receipt, daemon_receipt_binding = (
            _qualify_local_docker_control_plane(control_plane)
        )
        toolchain_policy, toolchain_receipt = _approved_rust_toolchain(control_plane)
        snapshot_project = _archive_commit_a_sources(
            commit_id, temporary_root / "snapshot"
        )
        cargo_lock = snapshot_project / "rust" / "formal_bridge_m25" / "Cargo.lock"
        if _sha256_file(cargo_lock) != toolchain_policy["cargo_lock_sha256"]:
            raise M25ErrataQualificationError("Commit-A Cargo.lock policy binding drift")
        vendor = temporary_root / "vendor-snapshot"
        (
            dependency_snapshot_root,
            dependency_snapshot_file_count,
            registry_package_count,
        ) = _build_cargo_dependency_snapshot(cargo_lock, vendor)
        if (
            dependency_snapshot_root != toolchain_policy["dependency_snapshot_root"]
            or dependency_snapshot_file_count
            != toolchain_policy["dependency_snapshot_file_count"]
            or registry_package_count
            != toolchain_policy["cargo_lock_registry_package_count"]
        ):
            raise M25ErrataQualificationError(
                "checksum-exact Cargo dependency snapshot policy drift"
            )
        vendor_manifest_sha256 = _sha256_directory_manifest(vendor)
        python_environment = _controlled_python_environment(temporary_root)
        snapshot_manifest_sha256 = _sha256_directory_manifest(snapshot_project)
        _make_snapshot_read_only(snapshot_project)
        python_report, python_execution = _snapshot_python_report(
            snapshot_project,
            environment=python_environment,
        )
        python_execution["source_commit"] = commit_id
        snapshot_golden = dict(
            _mapping(
                json.loads(
                    (snapshot_project / "golden_vectors" / "phase3_m25_errata_wire_v1.json").read_text(
                        encoding="utf-8"
                    )
                ),
                "detached golden fixture",
            )
        )

        target = temporary_root / "fresh-target"
        target.mkdir(mode=0o700)
        build_options = (
            "-v",
            f"{snapshot_project}:/input:ro",
            "-v",
            f"{vendor}:/vendor:ro",
            "-v",
            f"{target}:/output:rw",
            "-w",
            "/input/rust/formal_bridge_m25",
        )
        build_command = (
            RUST_CARGO_PATH,
            "--config",
            'source.crates-io.replace-with="vendored-sources"',
            "--config",
            'source.vendored-sources.directory="/vendor"',
            "build",
            "--release",
            "--locked",
            "--offline",
            "--jobs=1",
            "--manifest-path",
            "Cargo.toml",
        )
        _run_bytes(
            _docker_command(
                control_plane,
                RUST_IMAGE_REF,
                build_options,
                build_command,
                seccomp_path=BUILD_SECCOMP_PATH,
                container_environment=RUST_BUILD_ENVIRONMENT,
                user=f"{os.getuid()}:{os.getgid()}",
            ),
            environment=control_plane.environment,
            timeout=300,
            label="fresh Commit-A offline Rust OCI build",
        )
        binary = target / "release" / "hegel-formal-bridge-m25"
        rust_report, binary_digest = _run_rust_report_in_container(
            control_plane,
            binary,
        )
        golden_report = _validate_vector_report(
            snapshot_golden["report"], "detached golden report before publication"
        )
        if not _json_type_strict_equal(python_report, golden_report) or not (
            _json_type_strict_equal(rust_report, golden_report)
        ):
            raise M25ErrataQualificationError(
                "fresh detached reports did not validate before binary publication"
            )
        binary_payload = binary.read_bytes()
        persistence = _persist_validated_rust_binary(binary_payload, binary_digest)
        persisted_report, persisted_digest = _run_rust_report_in_container(
            control_plane,
            DEFAULT_RUST_BINARY,
        )
        if (
            persisted_digest != binary_digest
            or not _json_type_strict_equal(persisted_report, rust_report)
        ):
            raise M25ErrataQualificationError(
                "persisted default Rust binary replay differs from fresh replay"
            )
        if _sha256_directory_manifest(snapshot_project) != snapshot_manifest_sha256:
            raise M25ErrataQualificationError(
                "detached Commit-A source snapshot changed during replay"
            )
        if _sha256_directory_manifest(vendor) != vendor_manifest_sha256:
            raise M25ErrataQualificationError(
                "offline Cargo vendor snapshot changed during replay"
            )
        if _sha256_file(RUNTIME_SECCOMP_PATH) != toolchain_policy["runtime_seccomp_sha256"] or (
            _sha256_file(BUILD_SECCOMP_PATH) != toolchain_policy["build_seccomp_sha256"]
        ):
            raise M25ErrataQualificationError("container seccomp policy changed during replay")
        _assert_sources_match_commit(commit_id)
        build_receipt = {
            "binary_sha256": binary_digest,
            "binary_provenance": BINARY_PROVENANCE,
            "binary_source_binding_claim": True,
            "listed_rust_sources_are_build_attestation": False,
            "build_receipt_is_external_attestation": False,
            "source_commit": commit_id,
            "source_blobs_match_commit": True,
            "source_blobs_from_git_archive": True,
            "source_snapshot_manifest_sha256": snapshot_manifest_sha256,
            "source_snapshot_read_only": True,
            "source_snapshot_stable_during_replay": True,
            "working_tree_built": False,
            "fresh_target_directory": True,
            "cargo_locked": True,
            "cargo_offline": True,
            "cargo_incremental": False,
            "cargo_version": toolchain_receipt["cargo_version"],
            "rustc_version": toolchain_receipt["rustc_version"],
            "rust_oci_toolchain": toolchain_receipt,
            "approved_toolchain_policy_sha256": toolchain_policy_sha256,
            "approved_toolchain_policy_bound": True,
            "caller_supplied_toolchain_allowed": False,
            "cargo_lock_sha256": toolchain_policy["cargo_lock_sha256"],
            "cargo_lock_registry_archive_count": registry_package_count,
            "cargo_lock_registry_archives_verified": True,
            "dependency_snapshot_domain": CARGO_SNAPSHOT_DOMAIN,
            "dependency_snapshot_root": dependency_snapshot_root,
            "dependency_snapshot_file_count": dependency_snapshot_file_count,
            "vendor_directory_manifest_sha256": vendor_manifest_sha256,
            "host_cargo_registry_mounted_into_build_container": False,
            "environment_profile": "HEGEL_M25_OFFLINE_OCI_BUILD_ENV_V2",
            "runtime_docker_policy_id": RUNTIME_DOCKER_POLICY_ID,
            "build_docker_policy_id": BUILD_DOCKER_POLICY_ID,
            "inherited_environment_allowed": False,
            "rustc_wrapper_allowed": False,
            "binary_hash_and_exec_same_open_inode": False,
            "binary_private_snapshot_digest_stable_during_exec": True,
            "persisted_binary_replay_equal": True,
            "binary_reproducible_across_ephemeral_build_paths": False,
            "local_docker_control_plane_binding": dict(control_plane.binding),
            "local_docker_daemon_identity_receipt": daemon_receipt,
            "local_docker_daemon_receipt_binding": daemon_receipt_binding,
            "docker_pull_policy": "never",
            "docker_network_mode": "none",
            **persistence,
            "normalized_command": [
                RUST_CARGO_PATH,
                "--config",
                'source.crates-io.replace-with="vendored-sources"',
                "--config",
                'source.vendored-sources.directory="/vendor"',
                "build",
                "--release",
                "--locked",
                "--offline",
                "--jobs=1",
                "--manifest-path",
                "Cargo.toml",
            ],
        }
    return (
        python_report,
        rust_report,
        snapshot_golden,
        python_execution,
        build_receipt,
    )


def _load_golden() -> dict[str, object]:
    payload = dict(
        _mapping(
            json.loads(GOLDEN_VECTOR_PATH.read_text(encoding="utf-8")),
            "errata golden fixture",
        )
    )
    expected_fields = {
        "schema_version",
        "artifact_kind",
        "machine_freeze_id",
        "vector_schema",
        "authority_boundary",
        "report",
    }
    if set(payload) != expected_fields:
        raise M25ErrataQualificationError("errata golden fixture field-set drift")
    if payload.get("schema_version") != GOLDEN_SCHEMA:
        raise M25ErrataQualificationError("errata golden fixture schema drift")
    if payload.get("artifact_kind") != "SYNTHETIC_NON_AUTHORITATIVE":
        raise M25ErrataQualificationError("errata golden fixture authority drift")
    if payload.get("machine_freeze_id") != MACHINE_FREEZE_ID:
        raise M25ErrataQualificationError("errata golden fixture freeze drift")
    if payload.get("vector_schema") != VECTOR_SCHEMA:
        raise M25ErrataQualificationError("errata golden fixture vector schema drift")
    expected_fixture_boundary = {
        "gate_effect": "NONE",
        "m3_gates_before": 14,
        "m3_gates_after": 14,
        "child_state": "NOT_RUN",
        "contains_real_secret_material": False,
        "authoritative_root_generation": False,
        "formal_roots_generated": False,
        "seed_genesis_performed": False,
        "signature_claim": False,
    }
    if not _json_type_strict_equal(
        payload.get("authority_boundary"), expected_fixture_boundary
    ):
        raise M25ErrataQualificationError("errata golden authority boundary drift")
    _validate_vector_report(payload.get("report"), "golden report")
    return payload


def _validate_vector_report(value: object, name: str) -> dict[str, object]:
    report = dict(_mapping(value, name))
    if set(report) != REPORT_CORE_FIELDS:
        raise M25ErrataQualificationError(f"{name} field-set drift")
    if report.get("machine_freeze_id") != MACHINE_FREEZE_ID:
        raise M25ErrataQualificationError(f"{name} machine freeze drift")
    if report.get("vector_schema") != VECTOR_SCHEMA:
        raise M25ErrataQualificationError(f"{name} vector schema drift")
    for field, expected_count in EXPECTED_VECTOR_COUNTS.items():
        entries = report.get(field)
        if not isinstance(entries, list) or len(entries) != expected_count:
            raise M25ErrataQualificationError(
                f"{name} {field} must contain exactly {expected_count} vectors"
            )
        identity_field = "vector_id" if field == "guard_errors" else "name"
        identities: list[str] = []
        for index, entry in enumerate(entries):
            row = _mapping(entry, f"{name}.{field}[{index}]")
            identity = row.get(identity_field)
            if not isinstance(identity, str):
                raise M25ErrataQualificationError(
                    f"{name}.{field}[{index}] has no string identity"
                )
            identities.append(identity)
        if identities != sorted(identities) or len(identities) != len(set(identities)):
            raise M25ErrataQualificationError(f"{name} {field} order/uniqueness drift")
    return report


def _parse_rust_report(
    *,
    stdout: str,
    stderr: str,
    returncode: int,
) -> dict[str, object]:
    try:
        response = dict(_mapping(json.loads(stdout), "Rust errata response"))
    except json.JSONDecodeError as exc:
        raise M25ErrataQualificationError(
            "Rust errata replay returned invalid JSON: " + stderr
        ) from exc
    if returncode != 0 or response.get("ok") is not True:
        raise M25ErrataQualificationError(f"Rust errata replay failed: {response}")
    if set(response) != RUST_RESPONSE_FIELDS:
        raise M25ErrataQualificationError("Rust errata response field-set drift")
    if response.get("op") != "errata_vectors":
        raise M25ErrataQualificationError("Rust errata response operation echo drift")
    return _validate_vector_report(
        {field: response[field] for field in REPORT_CORE_FIELDS},
        "Rust errata report",
    )


def _python_report() -> dict[str, object]:
    response = dict(
        _mapping(generate_errata_vector_report_v1(), "Python errata response")
    )
    if set(response) != RUST_RESPONSE_FIELDS:
        raise M25ErrataQualificationError("Python errata response field-set drift")
    if response.get("ok") is not True or response.get("op") != "errata_vectors":
        raise M25ErrataQualificationError("Python errata response envelope drift")
    return _validate_vector_report(
        {field: response[field] for field in REPORT_CORE_FIELDS},
        "Python errata report",
    )


def _passing_dual_golden_verification() -> DualGoldenVerification:
    """Construct 10/10 only after the caller has produced every proof below."""

    return DualGoldenVerification(
        errata_document_in_commit_A=True,
        python_errata_vectors_pass=True,
        rust_errata_vectors_pass=True,
        python_rust_canonical_bytes_equal=True,
        python_rust_error_codes_equal=True,
        actor_trust_genesis_schema_frozen=True,
        append_only_id_registry_schema_frozen=True,
        parent_audit_bundle_schema_frozen=True,
        bridge_statement_and_execution_v2_schema_frozen=True,
        secrets_absent_from_repository=True,
    )


def dual_errata_qualification_report(
    *,
    implementation_basis_commit: str | None = None,
) -> dict[str, object]:
    """Fresh-build/replay both endpoints and authorize only the external step."""

    commit_id = _require_commit_id(
        repository_head_commit()
        if implementation_basis_commit is None
        else implementation_basis_commit
    )
    _assert_sources_match_commit(commit_id)
    source_bindings = _source_bindings()
    document_bindings = _document_bindings()
    golden_fixture_sha256 = _sha256_file(GOLDEN_VECTOR_PATH)

    golden = _load_golden()
    (
        python_report,
        rust_report,
        snapshot_golden,
        python_execution,
        rust_execution,
    ) = _fresh_commit_a_rust_replay(
        commit_id,
    )
    if not _json_type_strict_equal(snapshot_golden, golden):
        raise M25ErrataQualificationError(
            "detached Commit-A golden differs from the checked worktree fixture"
        )
    golden_report = _validate_vector_report(
        snapshot_golden["report"], "detached golden report"
    )
    if not _json_type_strict_equal(python_report, golden_report):
        raise M25ErrataQualificationError(
            "Python exact-wire report differs from the checked golden fixture"
        )
    if not _json_type_strict_equal(rust_report, golden_report):
        raise M25ErrataQualificationError(
            "Rust exact-wire report differs from the checked golden fixture"
        )

    secret_absence_receipt = repository_genesis_secret_absence_report(commit_id)
    if secret_absence_receipt.get("pass") is not True:
        raise M25ErrataQualificationError(
            "Commit A did not pass the frozen genesis-secret absence audit"
        )

    _assert_sources_match_commit(commit_id)
    if _source_bindings() != source_bindings:
        raise M25ErrataQualificationError("qualification sources changed during replay")
    if _document_bindings() != document_bindings:
        raise M25ErrataQualificationError("normative documents changed during replay")
    if _sha256_file(GOLDEN_VECTOR_PATH) != golden_fixture_sha256:
        raise M25ErrataQualificationError("golden fixture changed during replay")

    verification = _passing_dual_golden_verification()
    start_guard = external_genesis_start_guard_report(verification)
    authorization = assert_external_genesis_start_allowed(verification)
    if (
        start_guard.get("external_genesis_start_allowed") is not True
        or authorization.external_genesis_start_allowed is not True
    ):
        raise M25ErrataQualificationError(
            "dual-golden evidence did not authorize external genesis preparation"
        )

    payload: dict[str, object] = {
        "artifact": "phase3_m25_errata_qualification_v1",
        "schema_version": REPORT_SCHEMA,
        "artifact_kind": ARTIFACT_KIND,
        "machine_freeze_id": MACHINE_FREEZE_ID,
        "child_dsl_id": CHILD_DSL_ID,
        "status": STATUS,
        "implementation_basis_commit": commit_id,
        "implementation_commit_role": IMPLEMENTATION_COMMIT_ROLE,
        "publication_commit_may_substitute": False,
        "normative_document_bindings": document_bindings,
        "golden_fixture_sha256": golden_fixture_sha256,
        "source_bindings": source_bindings,
        "source_snapshot_stable_during_replay": True,
        "python_execution": python_execution,
        "rust_execution": rust_execution,
        "repository_secret_absence_receipt": secret_absence_receipt,
        "python_report": python_report,
        "rust_report": rust_report,
        "cross_language_exact_match": True,
        "golden_exact_match": True,
        "qualified_vector_counts": dict(EXPECTED_VECTOR_COUNTS),
        "dual_golden_start_guard": start_guard,
        "external_genesis_start_authorization": {
            "authorization_is_side_effect_free": (
                authorization.authorization_is_side_effect_free
            ),
            "external_genesis_start_allowed": (
                authorization.external_genesis_start_allowed
            ),
            "m3_gates_satisfied": authorization.m3_gates_satisfied,
            "m3_gates_total": authorization.m3_gates_total,
            "child_state": authorization.child_state,
            "gate24_qualified": authorization.gate24_qualified,
            "m3_entry_allowed": authorization.m3_entry_allowed,
            "m3_run_started": authorization.m3_run_started,
            "phase3_m3_start_authorized": (
                authorization.phase3_m3_start_authorized
            ),
            "checked_artifact_replay_alone_sufficient": False,
            "fresh_dual_replay_required": True,
        },
        "formal_input_roots": None,
        "formal_roots_generated": False,
        "m3_execution_manifest_root": None,
        "authority_boundary": dict(EXPECTED_AUTHORITY_BOUNDARY),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["diagnostic_report_id"] = stable_hash(
        payload,
        prefix="phase3_m25_errata_qualification_",
    )
    return payload


def _validate_report_envelope(report: Mapping[str, object]) -> None:
    expected_fields = {
        "artifact",
        "schema_version",
        "artifact_kind",
        "machine_freeze_id",
        "child_dsl_id",
        "status",
        "implementation_basis_commit",
        "implementation_commit_role",
        "publication_commit_may_substitute",
        "normative_document_bindings",
        "golden_fixture_sha256",
        "source_bindings",
        "source_snapshot_stable_during_replay",
        "python_execution",
        "rust_execution",
        "repository_secret_absence_receipt",
        "python_report",
        "rust_report",
        "cross_language_exact_match",
        "golden_exact_match",
        "qualified_vector_counts",
        "dual_golden_start_guard",
        "external_genesis_start_authorization",
        "formal_input_roots",
        "formal_roots_generated",
        "m3_execution_manifest_root",
        "authority_boundary",
        "claim_boundary",
        "diagnostic_report_id",
    }
    if set(report) != expected_fields:
        raise M25ErrataQualificationError("errata qualification field-set drift")
    exact_scalars = {
        "artifact": "phase3_m25_errata_qualification_v1",
        "schema_version": REPORT_SCHEMA,
        "artifact_kind": ARTIFACT_KIND,
        "machine_freeze_id": MACHINE_FREEZE_ID,
        "child_dsl_id": CHILD_DSL_ID,
        "status": STATUS,
        "implementation_commit_role": IMPLEMENTATION_COMMIT_ROLE,
        "publication_commit_may_substitute": False,
        "source_snapshot_stable_during_replay": True,
        "cross_language_exact_match": True,
        "golden_exact_match": True,
        "formal_input_roots": None,
        "formal_roots_generated": False,
        "m3_execution_manifest_root": None,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    for field, expected in exact_scalars.items():
        if not _json_type_strict_equal(report.get(field), expected):
            raise M25ErrataQualificationError(f"errata qualification {field} drift")
    commit = report.get("implementation_basis_commit")
    if not isinstance(commit, str):
        raise M25ErrataQualificationError("implementation basis commit is not text")
    _require_commit_id(commit)
    secret_receipt = _mapping(
        report.get("repository_secret_absence_receipt"),
        "repository secret absence receipt",
    )
    if (
        secret_receipt.get("audited_commit_id") != commit
        or secret_receipt.get("status") != SECRET_ABSENCE_PASS_STATUS
        or secret_receipt.get("pass") is not True
        or secret_receipt.get("zero_findings") is not True
        or secret_receipt.get("findings") != []
        or secret_receipt.get("immediate_second_replay_equal") is not True
    ):
        raise M25ErrataQualificationError(
            "repository secret-absence receipt boundary drift"
        )
    if not _json_type_strict_equal(
        report.get("qualified_vector_counts"), EXPECTED_VECTOR_COUNTS
    ):
        raise M25ErrataQualificationError("qualified vector counts drift")
    if not _json_type_strict_equal(
        report.get("authority_boundary"), EXPECTED_AUTHORITY_BOUNDARY
    ):
        raise M25ErrataQualificationError("qualification authority boundary drift")

    authorization = _mapping(
        report.get("external_genesis_start_authorization"),
        "external genesis authorization",
    )
    expected_authorization = {
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
    if not _json_type_strict_equal(dict(authorization), expected_authorization):
        raise M25ErrataQualificationError("external authorization boundary drift")
    expected_guard = external_genesis_start_guard_report(
        _passing_dual_golden_verification()
    )
    if not _json_type_strict_equal(report.get("dual_golden_start_guard"), expected_guard):
        raise M25ErrataQualificationError("dual-golden guard evidence drift")

    python_execution = _mapping(report.get("python_execution"), "Python execution")
    expected_python_fields = {
        "execution_mode",
        "source_commit",
        "python_version",
        "python_executable_sha256",
        "isolated_mode",
        "minimal_module_closure",
        "package_init_executed",
        "source_blobs_from_git_archive",
        "working_tree_executed",
        "execution_receipt_is_external_attestation",
    }
    if set(python_execution) != expected_python_fields:
        raise M25ErrataQualificationError("Python execution field-set drift")
    expected_python_values = {
        "execution_mode": "DETACHED_COMMIT_A_SOURCE_REPLAY",
        "source_commit": commit,
        "isolated_mode": True,
        "minimal_module_closure": True,
        "package_init_executed": False,
        "source_blobs_from_git_archive": True,
        "working_tree_executed": False,
        "execution_receipt_is_external_attestation": False,
    }
    for field, expected in expected_python_values.items():
        if not _json_type_strict_equal(python_execution.get(field), expected):
            raise M25ErrataQualificationError(f"Python execution {field} drift")
    for field in ("python_version", "python_executable_sha256"):
        value = python_execution.get(field)
        if not isinstance(value, str) or not value or "\n" in value:
            raise M25ErrataQualificationError(f"Python execution {field} drift")

    rust_execution = _mapping(report.get("rust_execution"), "Rust execution")
    expected_rust_fields = {
        "binary_sha256",
        "binary_provenance",
        "binary_source_binding_claim",
        "listed_rust_sources_are_build_attestation",
        "build_receipt_is_external_attestation",
        "source_commit",
        "source_blobs_match_commit",
        "source_blobs_from_git_archive",
        "source_snapshot_manifest_sha256",
        "source_snapshot_read_only",
        "source_snapshot_stable_during_replay",
        "working_tree_built",
        "fresh_target_directory",
        "cargo_locked",
        "cargo_offline",
        "cargo_incremental",
        "cargo_version",
        "rustc_version",
        "rust_oci_toolchain",
        "approved_toolchain_policy_sha256",
        "approved_toolchain_policy_bound",
        "caller_supplied_toolchain_allowed",
        "cargo_lock_sha256",
        "cargo_lock_registry_archive_count",
        "cargo_lock_registry_archives_verified",
        "dependency_snapshot_domain",
        "dependency_snapshot_root",
        "dependency_snapshot_file_count",
        "vendor_directory_manifest_sha256",
        "host_cargo_registry_mounted_into_build_container",
        "environment_profile",
        "runtime_docker_policy_id",
        "build_docker_policy_id",
        "inherited_environment_allowed",
        "rustc_wrapper_allowed",
        "binary_hash_and_exec_same_open_inode",
        "binary_private_snapshot_digest_stable_during_exec",
        "persisted_binary_replay_equal",
        "binary_reproducible_across_ephemeral_build_paths",
        "local_docker_control_plane_binding",
        "local_docker_daemon_identity_receipt",
        "local_docker_daemon_receipt_binding",
        "docker_pull_policy",
        "docker_network_mode",
        "default_rust_binary_repository_path",
        "persisted_binary_sha256",
        "persisted_binary_mode_octal",
        "persisted_binary_atomic_replace",
        "persisted_binary_is_symlink",
        "normalized_command",
    }
    if set(rust_execution) != expected_rust_fields:
        raise M25ErrataQualificationError("Rust execution field-set drift")
    if rust_execution.get("binary_provenance") != BINARY_PROVENANCE:
        raise M25ErrataQualificationError("Rust executable provenance drift")
    if rust_execution.get("binary_source_binding_claim") is not True:
        raise M25ErrataQualificationError("Rust Commit-A source binding is absent")
    if rust_execution.get("listed_rust_sources_are_build_attestation") is not False:
        raise M25ErrataQualificationError("source hashes are not a build attestation")
    toolchain_policy = _load_approved_toolchain_policy()
    expected_build_values = {
        "build_receipt_is_external_attestation": False,
        "source_commit": commit,
        "source_blobs_match_commit": True,
        "source_blobs_from_git_archive": True,
        "source_snapshot_read_only": True,
        "source_snapshot_stable_during_replay": True,
        "working_tree_built": False,
        "fresh_target_directory": True,
        "cargo_locked": True,
        "cargo_offline": True,
        "cargo_incremental": False,
        "cargo_version": toolchain_policy["cargo_version"],
        "rustc_version": toolchain_policy["rustc_version"],
        "approved_toolchain_policy_sha256": _sha256_file(
            APPROVED_TOOLCHAIN_POLICY_PATH
        ),
        "approved_toolchain_policy_bound": True,
        "caller_supplied_toolchain_allowed": False,
        "cargo_lock_sha256": toolchain_policy["cargo_lock_sha256"],
        "cargo_lock_registry_archive_count": toolchain_policy[
            "cargo_lock_registry_package_count"
        ],
        "dependency_snapshot_domain": CARGO_SNAPSHOT_DOMAIN,
        "dependency_snapshot_root": toolchain_policy["dependency_snapshot_root"],
        "dependency_snapshot_file_count": toolchain_policy[
            "dependency_snapshot_file_count"
        ],
        "host_cargo_registry_mounted_into_build_container": False,
        "environment_profile": "HEGEL_M25_OFFLINE_OCI_BUILD_ENV_V2",
        "runtime_docker_policy_id": RUNTIME_DOCKER_POLICY_ID,
        "build_docker_policy_id": BUILD_DOCKER_POLICY_ID,
        "inherited_environment_allowed": False,
        "rustc_wrapper_allowed": False,
        "cargo_lock_registry_archives_verified": True,
        "binary_hash_and_exec_same_open_inode": False,
        "binary_private_snapshot_digest_stable_during_exec": True,
        "persisted_binary_replay_equal": True,
        "binary_reproducible_across_ephemeral_build_paths": False,
        "docker_pull_policy": "never",
        "docker_network_mode": "none",
        "default_rust_binary_repository_path": DEFAULT_RUST_BINARY.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "persisted_binary_mode_octal": "0755",
        "persisted_binary_atomic_replace": True,
        "persisted_binary_is_symlink": False,
        "normalized_command": [
            RUST_CARGO_PATH,
            "--config",
            'source.crates-io.replace-with="vendored-sources"',
            "--config",
            'source.vendored-sources.directory="/vendor"',
            "build",
            "--release",
            "--locked",
            "--offline",
            "--jobs=1",
            "--manifest-path",
            "Cargo.toml",
        ],
    }
    for field, expected in expected_build_values.items():
        if not _json_type_strict_equal(rust_execution.get(field), expected):
            raise M25ErrataQualificationError(f"Rust build receipt {field} drift")
    for field in (
        "cargo_version",
        "rustc_version",
        "approved_toolchain_policy_sha256",
        "cargo_lock_sha256",
        "dependency_snapshot_root",
        "vendor_directory_manifest_sha256",
        "source_snapshot_manifest_sha256",
    ):
        value = rust_execution.get(field)
        if not isinstance(value, str) or not value or "\n" in value:
            raise M25ErrataQualificationError(f"Rust build receipt {field} drift")
    archive_count = rust_execution.get("cargo_lock_registry_archive_count")
    snapshot_file_count = rust_execution.get("dependency_snapshot_file_count")
    if (
        type(archive_count) is not int
        or archive_count <= 0
        or type(snapshot_file_count) is not int
        or snapshot_file_count <= 0
    ):
        raise M25ErrataQualificationError(
            "Rust build receipt Cargo snapshot count drift"
        )
    binary_digest = rust_execution.get("binary_sha256")
    if (
        not isinstance(binary_digest, str)
        or len(binary_digest) != 71
        or not binary_digest.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in binary_digest[7:])
    ):
        raise M25ErrataQualificationError("Rust binary digest syntax drift")
    if rust_execution.get("persisted_binary_sha256") != binary_digest:
        raise M25ErrataQualificationError("persisted Rust binary digest binding drift")

    expected_toolchain_receipt = {
        "image_ref": toolchain_policy["image_ref"],
        "image_id": toolchain_policy["image_id"],
        "oci_manifest_digest": toolchain_policy["oci_manifest_digest"],
        "operating_system": toolchain_policy["operating_system"],
        "architecture": toolchain_policy["architecture"],
        "cargo_binary_path": toolchain_policy["cargo_binary_path"],
        "cargo_binary_sha256": toolchain_policy["cargo_binary_sha256"],
        "cargo_version": toolchain_policy["cargo_version"],
        "cargo_version_stdout_sha256": toolchain_policy[
            "cargo_version_stdout_sha256"
        ],
        "rustc_binary_path": toolchain_policy["rustc_binary_path"],
        "rustc_binary_sha256": toolchain_policy["rustc_binary_sha256"],
        "rustc_version": toolchain_policy["rustc_version"],
        "rustc_verbose_version_stdout_sha256": toolchain_policy[
            "rustc_verbose_version_stdout_sha256"
        ],
        "runtime_environment_sha256": toolchain_policy[
            "runtime_environment_sha256"
        ],
        "build_environment_sha256": toolchain_policy["build_environment_sha256"],
        "runtime_seccomp_sha256": toolchain_policy["runtime_seccomp_sha256"],
        "build_seccomp_sha256": toolchain_policy["build_seccomp_sha256"],
        "image_config_environment_ignored": True,
        "pull_policy": "never",
        "network_mode": "none",
        "toolchain_receipt_is_external_attestation": False,
    }
    if not _json_type_strict_equal(
        rust_execution.get("rust_oci_toolchain"), expected_toolchain_receipt
    ):
        raise M25ErrataQualificationError("Rust OCI toolchain receipt drift")
    daemon_receipt = _mapping(
        rust_execution.get("local_docker_daemon_identity_receipt"),
        "local Docker daemon receipt",
    )
    try:
        daemon_binding = local_docker_daemon_receipt_binding_v1(daemon_receipt).hex()
    except Phase3LocalRuntimeError as exc:
        raise M25ErrataQualificationError(
            f"local Docker daemon receipt drift: {exc}"
        ) from exc
    if (
        rust_execution.get("local_docker_daemon_receipt_binding") != daemon_binding
        or rust_execution.get("local_docker_control_plane_binding")
        != daemon_receipt.get("control_plane_binding")
    ):
        raise M25ErrataQualificationError("local Docker control-plane binding drift")

    provided_id = report.get("diagnostic_report_id")
    body = dict(report)
    body.pop("diagnostic_report_id", None)
    if provided_id != stable_hash(
        body, prefix="phase3_m25_errata_qualification_"
    ):
        raise M25ErrataQualificationError("errata qualification self-ID mismatch")


def validate_checked_errata_qualification_report(
    report: Mapping[str, object],
) -> None:
    """Validate archival consistency; never use this alone for external admission."""

    _validate_report_envelope(report)
    _assert_sources_match_commit(str(report["implementation_basis_commit"]))
    validate_repository_genesis_secret_absence_report(
        _mapping(
            report.get("repository_secret_absence_receipt"),
            "repository secret absence receipt",
        ),
        expected_commit_id=str(report["implementation_basis_commit"]),
    )
    if report.get("source_bindings") != _source_bindings():
        raise M25ErrataQualificationError("errata qualification sources are stale")
    if report.get("normative_document_bindings") != _document_bindings():
        raise M25ErrataQualificationError("errata qualification documents are stale")
    if report.get("golden_fixture_sha256") != _sha256_file(GOLDEN_VECTOR_PATH):
        raise M25ErrataQualificationError("errata qualification golden is stale")
    golden_report = _validate_vector_report(_load_golden()["report"], "golden report")
    python_report = _python_report()
    stored_rust = _validate_vector_report(report.get("rust_report"), "stored Rust report")
    if not _json_type_strict_equal(report.get("python_report"), python_report):
        raise M25ErrataQualificationError("stored Python errata report is stale")
    if not _json_type_strict_equal(python_report, golden_report):
        raise M25ErrataQualificationError("current Python report differs from golden")
    if not _json_type_strict_equal(stored_rust, golden_report):
        raise M25ErrataQualificationError("stored Rust report differs from golden")


def validate_dual_errata_qualification_report(
    report: Mapping[str, object],
) -> None:
    """Run a fresh dual replay and require exact equality with the report."""

    _validate_report_envelope(report)
    expected = dual_errata_qualification_report(
        implementation_basis_commit=str(report["implementation_basis_commit"]),
    )
    supplied = dict(report)
    expected_rust = dict(_mapping(expected["rust_execution"], "fresh Rust execution"))
    supplied_rust = dict(_mapping(supplied["rust_execution"], "stored Rust execution"))
    for nondeterministic_build_field in (
        "binary_sha256",
        "persisted_binary_sha256",
    ):
        expected_rust.pop(nondeterministic_build_field)
        supplied_rust.pop(nondeterministic_build_field)
    expected["rust_execution"] = expected_rust
    supplied["rust_execution"] = supplied_rust
    expected.pop("diagnostic_report_id")
    supplied.pop("diagnostic_report_id")
    if not _json_type_strict_equal(supplied, expected):
        raise M25ErrataQualificationError(
            "errata qualification report differs from current dual replay"
        )


__all__ = [
    "ARTIFACT_KIND",
    "CHECKED_REPORT_PATH",
    "CLAIM_BOUNDARY",
    "EXPECTED_VECTOR_COUNTS",
    "GOLDEN_VECTOR_PATH",
    "M25ErrataQualificationError",
    "MACHINE_FREEZE_ID",
    "STATUS",
    "dual_errata_qualification_report",
    "repository_head_commit",
    "validate_checked_errata_qualification_report",
    "validate_dual_errata_qualification_report",
    "validate_errata_qualification_output_path",
]
