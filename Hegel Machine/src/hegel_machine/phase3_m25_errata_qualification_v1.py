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
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
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


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
MACHINE_FREEZE_ID: Final = "hegel-freeze-p2b-p3-v1.1.2"
CHILD_DSL_ID: Final = "hegel-old-dsl-v1.1.0"
VECTOR_SCHEMA: Final = "hegel-phase3-m25-exact-wire-errata-vectors/1"
GOLDEN_SCHEMA: Final = "hegel-phase3-m25-exact-wire-errata-golden/1"
REPORT_SCHEMA: Final = "hegel-phase3-m25-exact-wire-errata-qualification/1"
ARTIFACT_KIND: Final = "DETERMINISTIC_CANDIDATE_NON_AUTHORITATIVE"
STATUS: Final = "DUAL_EXACT_WIRE_ERRATA_GOLDEN_PASS"
BINARY_PROVENANCE: Final = "DETACHED_COMMIT_A_FRESH_ISOLATED_SOURCE_BUILD"
IMPLEMENTATION_COMMIT_ROLE: Final = "DETERMINISTIC_IMPLEMENTATION_BASIS_COMMIT_A"
CLAIM_BOUNDARY: Final = (
    "Python and Rust, executed from a detached Commit-A source snapshot with "
    "a fresh isolated Rust target, reproduce the checked public E1-E12 "
    "exact-wire vectors, candidate roots, record-tree roots, and negative "
    "error codes. The locally recorded toolchain receipt is not an external "
    "build attestation. A successful fresh replay permits an independent "
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

SOURCE_PATHS: Final = (
    "src/hegel_machine/__init__.py",
    "src/hegel_machine/hashing.py",
    "src/hegel_machine/strict_cbor_v1.py",
    "src/hegel_machine/phase3_m25_wire_v1.py",
    "src/hegel_machine/phase3_m25_errata_vectors_v1.py",
    "src/hegel_machine/phase3_m25_external_v1.py",
    "src/hegel_machine/phase3_m25_readiness_v1.py",
    "src/hegel_machine/phase3_m25_secret_absence_v1.py",
    "src/hegel_machine/phase3_m25_errata_qualification_v1.py",
    "src/hegel_machine/cli.py",
    "config/phase3_m25_approved_local_rust_toolchain_v1.json",
    "rust/formal_bridge_m25/Cargo.toml",
    "rust/formal_bridge_m25/Cargo.lock",
    "rust/formal_bridge_m25/src/lib.rs",
    "rust/formal_bridge_m25/src/main.rs",
    "golden_vectors/phase3_m25_errata_wire_v1.json",
    "docs/Hegel_Machine_Phase3A_M25_Bit_Exact_Wire_Completion_Amendment.md",
    "docs/Hegel_Machine_Phase3A_M25_Exact_Wire_Errata_Resolution.md",
    "docs/Hegel_Machine_Phase3A_M25_Implementation_Closure_Addendum_v1.md",
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


def repository_head_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        raise M25ErrataQualificationError(
            "cannot resolve deterministic implementation basis commit"
        )
    return _require_commit_id(completed.stdout.strip())


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
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        raise M25ErrataQualificationError("cannot resolve repository root")
    return Path(completed.stdout.strip()).resolve()


def _assert_sources_match_commit(commit_id: str) -> None:
    """Prove Commit A contains every byte claimed by the qualification.

    Merely naming ``HEAD`` would allow dirty worktree bytes to masquerade as
    commit-bound evidence.  Read every bound blob back from Git and compare it
    byte-for-byte with the replay input before setting the Commit-A guard.
    """

    repository_root = _repository_root()
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit_id, "HEAD"],
        cwd=repository_root,
        capture_output=True,
        check=False,
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
        completed = subprocess.run(
            ["git", "show", f"{commit_id}:{repository_relative.as_posix()}"],
            cwd=repository_root,
            capture_output=True,
            check=False,
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


def _resolve_tool(executable: str | Path, label: str) -> str:
    raw = os.fspath(executable)
    resolved = shutil.which(raw)
    if resolved is None:
        raise M25ErrataQualificationError(f"{label} executable is unavailable: {raw}")
    # Preserve the discovered launcher path; its resolved file digest is checked.
    return str(Path(resolved).absolute())


def _sha256_resolved_executable(executable: str) -> str:
    path = Path(executable).resolve(strict=True)
    if not path.is_file():
        raise M25ErrataQualificationError(f"tool target is not a file: {executable}")
    return _sha256_file(path)


def _load_approved_toolchain_policy() -> dict[str, str]:
    policy = dict(
        _mapping(
            json.loads(APPROVED_TOOLCHAIN_POLICY_PATH.read_text(encoding="utf-8")),
            "approved local Rust toolchain policy",
        )
    )
    expected_fields = {
        "schema_version",
        "authority_boundary",
        "rustup_toolchain_id",
        "rustup_launcher_sha256",
        "cargo_binary_sha256",
        "rustc_binary_sha256",
        "cargo_version",
        "rustc_version",
        "toolchain_directory_manifest_sha256",
    }
    if set(policy) != expected_fields or not all(
        isinstance(value, str) for value in policy.values()
    ):
        raise M25ErrataQualificationError(
            "approved local Rust toolchain policy field-set drift"
        )
    if (
        policy["schema_version"]
        != "hegel-phase3-m25-approved-local-rust-toolchain/1"
        or policy["authority_boundary"]
        != "LOCAL_DETERMINISTIC_BUILD_POLICY_NOT_EXTERNAL_ATTESTATION"
        or re.fullmatch(
            r"[A-Za-z0-9_.-]+-[A-Za-z0-9_.-]+",
            policy["rustup_toolchain_id"],
        )
        is None
    ):
        raise M25ErrataQualificationError(
            "approved local Rust toolchain policy identity drift"
        )
    for field in (
        "rustup_launcher_sha256",
        "cargo_binary_sha256",
        "rustc_binary_sha256",
        "toolchain_directory_manifest_sha256",
    ):
        value = policy[field]
        if (
            len(value) != 71
            or not value.startswith("sha256:")
            or re.fullmatch(r"[0-9a-f]{64}", value[7:]) is None
        ):
            raise M25ErrataQualificationError(
                f"approved local Rust toolchain {field} drift"
            )
    for field in ("cargo_version", "rustc_version"):
        if not policy[field] or "\n" in policy[field]:
            raise M25ErrataQualificationError(
                f"approved local Rust toolchain {field} drift"
            )
    return policy


def _rustup_discovery_environment(rustup_home: Path, toolchain_id: str) -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent-hegel-m25-toolchain-home",
        "RUSTUP_HOME": str(rustup_home),
        "RUSTUP_TOOLCHAIN": toolchain_id,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }


def _approved_rust_toolchain() -> tuple[dict[str, str], str, str, Path, str]:
    """Resolve only the Commit-A-pinned local Rust toolchain.

    The public qualification API deliberately accepts no executable path.  A
    rustup launcher, both real toolchain binaries, and the entire toolchain
    directory must match the committed policy before any guard can pass.
    """

    policy = _load_approved_toolchain_policy()
    rustup = _resolve_tool("rustup", "rustup")
    if _sha256_resolved_executable(rustup) != policy["rustup_launcher_sha256"]:
        raise M25ErrataQualificationError(
            "rustup launcher is not the Commit-A-approved local launcher"
        )
    rustup_home = Path(
        os.environ.get("RUSTUP_HOME", str(Path.home() / ".rustup"))
    ).resolve()
    discovery_environment = _rustup_discovery_environment(
        rustup_home,
        policy["rustup_toolchain_id"],
    )
    tools: dict[str, str] = {}
    for name in ("cargo", "rustc"):
        completed = subprocess.run(
            [
                rustup,
                "which",
                "--toolchain",
                policy["rustup_toolchain_id"],
                name,
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
            env=discovery_environment,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            raise M25ErrataQualificationError(
                f"cannot resolve approved Rust toolchain {name}"
            )
        candidate = Path(completed.stdout.strip()).resolve(strict=True)
        if not candidate.is_file() or candidate.name != name:
            raise M25ErrataQualificationError(
                f"approved Rust toolchain {name} path drift"
            )
        tools[name] = str(candidate)
    cargo_path = Path(tools["cargo"])
    rustc_path = Path(tools["rustc"])
    if cargo_path.parent != rustc_path.parent or cargo_path.parent.name != "bin":
        raise M25ErrataQualificationError(
            "approved cargo and rustc do not share one toolchain bin directory"
        )
    toolchain_root = cargo_path.parent.parent
    if _sha256_file(cargo_path) != policy["cargo_binary_sha256"]:
        raise M25ErrataQualificationError("approved cargo binary digest drift")
    if _sha256_file(rustc_path) != policy["rustc_binary_sha256"]:
        raise M25ErrataQualificationError("approved rustc binary digest drift")
    toolchain_manifest = _sha256_directory_manifest(toolchain_root)
    if toolchain_manifest != policy["toolchain_directory_manifest_sha256"]:
        raise M25ErrataQualificationError(
            "approved Rust toolchain directory manifest drift"
        )
    return policy, tools["cargo"], tools["rustc"], toolchain_root, toolchain_manifest


def _archive_commit_a_sources(commit_id: str, destination: Path) -> Path:
    """Materialize only bound Commit-A blobs in a private detached snapshot."""

    repository_root = _repository_root()
    destination.mkdir(parents=True, exist_ok=False)
    project_relative = PROJECT_ROOT.resolve().relative_to(repository_root)
    repository_paths = [
        (project_relative / relative).as_posix()
        for relative in dict.fromkeys(SOURCE_PATHS)
    ]
    completed = subprocess.run(
        ["git", "archive", "--format=tar", commit_id, "--", *repository_paths],
        cwd=repository_root,
        capture_output=True,
        check=False,
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
    for section in text.split("[[package]]")[1:]:
        fields: dict[str, str] = {}
        for field in ("name", "version", "source", "checksum"):
            match = re.search(rf'^\s*{field}\s*=\s*"([^"]+)"\s*$', section, re.MULTILINE)
            if match is not None:
                fields[field] = match.group(1)
        source = fields.get("source")
        if source is None:
            continue
        if not source.startswith("registry+"):
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
        packages.append((name, version, checksum))
    result = tuple(sorted(packages))
    if not result or len(result) != len(set(result)):
        raise M25ErrataQualificationError(
            "Cargo.lock registry package set is empty or non-unique"
        )
    return result


def _cargo_registry_input_manifest(registry: Path) -> str:
    digest = hashlib.sha256()
    digest.update(b"HEGEL/M25/CARGO_REGISTRY_INPUT/V1\x00")
    for name in ("cache", "index"):
        subtree = registry / name
        subtree_digest = _sha256_directory_manifest(subtree).encode("ascii")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name.encode("ascii"))
        digest.update(len(subtree_digest).to_bytes(8, "big"))
        digest.update(subtree_digest)
    return "sha256:" + digest.hexdigest()


def _copy_offline_cargo_registry(
    destination_cargo_home: Path,
    *,
    cargo_lock: Path,
) -> tuple[str, int]:
    """Copy only lock-checksummed crate archives plus the offline index.

    Already-unpacked registry source is intentionally excluded.  Cargo must
    unpack every dependency inside the private replay directory from a
    ``.crate`` archive whose SHA-256 is fixed by Commit A's Cargo.lock.
    """

    source_home = Path(
        os.environ.get("CARGO_HOME", str(Path.home() / ".cargo"))
    ).resolve()
    source_registry = source_home / "registry"
    source_cache = source_registry / "cache"
    source_index = source_registry / "index"
    if not source_cache.is_dir() or not source_index.is_dir():
        raise M25ErrataQualificationError(
            "offline Cargo registry cache is unavailable"
        )
    destination_cargo_home.mkdir(parents=True, exist_ok=False)
    destination_registry = destination_cargo_home / "registry"
    destination_cache = destination_registry / "cache"
    shutil.copytree(source_index, destination_registry / "index")
    archive_count = 0
    locked_packages = _cargo_lock_registry_packages(cargo_lock)
    for name, version, checksum in locked_packages:
        archive_name = f"{name}-{version}.crate"
        candidates = sorted(source_cache.glob(f"*/{archive_name}"))
        if len(candidates) > 1:
            raise M25ErrataQualificationError(
                f"offline Cargo archive is ambiguous: {archive_name}"
            )
        if not candidates:
            # Cargo.lock can retain target-specific packages not selected for
            # this host.  A required missing archive will make the later
            # --offline build fail closed; no unchecked archive is copied.
            continue
        archive = candidates[0]
        actual = hashlib.sha256(archive.read_bytes()).hexdigest()
        if actual != checksum:
            raise M25ErrataQualificationError(
                f"offline Cargo archive differs from Cargo.lock: {archive_name}"
            )
        relative = archive.relative_to(source_cache)
        target = destination_cache / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != checksum:
            raise M25ErrataQualificationError(
                f"copied Cargo archive checksum drift: {archive_name}"
            )
        archive_count += 1
    _make_snapshot_read_only(destination_registry / "cache")
    _make_snapshot_read_only(destination_registry / "index")
    if archive_count <= 0:
        raise M25ErrataQualificationError(
            "offline Cargo registry contains no lock-verified archives"
        )
    return _cargo_registry_input_manifest(destination_registry), archive_count


def _assert_no_cargo_config(cwd: Path, cargo_home: Path) -> None:
    """Reject every Cargo config search location visible to this replay."""

    candidates = [cargo_home / "config", cargo_home / "config.toml"]
    current = cwd.resolve()
    for parent in (current, *current.parents):
        candidates.extend(
            (parent / ".cargo" / "config", parent / ".cargo" / "config.toml")
        )
    if any(path.exists() or path.is_symlink() for path in candidates):
        raise M25ErrataQualificationError(
            "ambient or ancestor Cargo configuration is visible to replay"
        )


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


def _controlled_build_environment(
    temporary_root: Path,
    *,
    cargo: str,
    rustc: str,
    cargo_home: Path,
) -> dict[str, str]:
    home = temporary_root / "isolated-home"
    temp = temporary_root / "process-tmp"
    home.mkdir()
    temp.mkdir()
    path_entries = tuple(
        dict.fromkeys(
            [
                str(Path(cargo).parent),
                str(Path(rustc).parent),
                "/usr/bin",
                "/bin",
            ]
        )
    )
    environment = {
        "PATH": os.pathsep.join(path_entries),
        "HOME": str(home),
        "CARGO_HOME": str(cargo_home),
        "CARGO_INCREMENTAL": "0",
        "CARGO_NET_OFFLINE": "true",
        "RUSTC": rustc,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TMPDIR": str(temp),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "SOURCE_DATE_EPOCH": "0",
    }
    return environment


def _snapshot_python_report(
    snapshot_project: Path,
    *,
    environment: Mapping[str, str],
) -> tuple[dict[str, object], dict[str, object]]:
    python = str(Path(sys.executable).resolve(strict=True))
    script = (
        "import json,sys;"
        "sys.path.insert(0,sys.argv[1]);"
        "from hegel_machine.phase3_m25_errata_vectors_v1 import "
        "generate_errata_vector_report_v1;"
        "print(json.dumps(generate_errata_vector_report_v1(),"
        "sort_keys=True,separators=(',',':')))"
    )
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
        "source_blobs_from_git_archive": True,
        "working_tree_executed": False,
        "execution_receipt_is_external_attestation": False,
    }
    return report, execution


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

    (
        toolchain_policy,
        cargo,
        rustc,
        toolchain_root,
        toolchain_manifest_sha256,
    ) = _approved_rust_toolchain()
    toolchain_policy_sha256 = _sha256_file(APPROVED_TOOLCHAIN_POLICY_PATH)
    _assert_sources_match_commit(commit_id)
    with tempfile.TemporaryDirectory(
        prefix="hegel-m25-commit-a-build-", dir="/tmp"
    ) as temporary:
        temporary_root = Path(temporary)
        snapshot_project = _archive_commit_a_sources(
            commit_id, temporary_root / "snapshot"
        )
        cargo_home = temporary_root / "isolated-cargo-home"
        registry_manifest_sha256, registry_archive_count = (
            _copy_offline_cargo_registry(
                cargo_home,
                cargo_lock=(
                    snapshot_project / "rust" / "formal_bridge_m25" / "Cargo.lock"
                ),
            )
        )
        environment = _controlled_build_environment(
            temporary_root,
            cargo=cargo,
            rustc=rustc,
            cargo_home=cargo_home,
        )
        _assert_no_cargo_config(snapshot_project, cargo_home)
        snapshot_manifest_sha256 = _sha256_directory_manifest(snapshot_project)
        _make_snapshot_read_only(snapshot_project)
        cargo_binary_sha256 = _sha256_file(Path(cargo))
        rustc_binary_sha256 = _sha256_file(Path(rustc))
        cargo_version = _tool_version(
            cargo,
            "cargo",
            environment=environment,
            cwd=snapshot_project,
        )
        rustc_version = _tool_version(
            rustc,
            "rustc",
            environment=environment,
            cwd=snapshot_project,
        )
        if (
            cargo_version != toolchain_policy["cargo_version"]
            or rustc_version != toolchain_policy["rustc_version"]
        ):
            raise M25ErrataQualificationError(
                "approved Rust toolchain version drift"
            )
        python_report, python_execution = _snapshot_python_report(
            snapshot_project,
            environment=environment,
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
        if target.exists():
            raise M25ErrataQualificationError("fresh Rust target unexpectedly exists")
        manifest = snapshot_project / "rust" / "formal_bridge_m25" / "Cargo.toml"
        command = [
            cargo,
            "build",
            "--locked",
            "--offline",
            "--manifest-path",
            str(manifest),
            "--target-dir",
            str(target),
        ]
        completed = subprocess.run(
            command,
            cwd=snapshot_project,
            text=True,
            capture_output=True,
            check=False,
            timeout=300,
            env=environment,
        )
        if completed.returncode != 0:
            raise M25ErrataQualificationError(
                "fresh Commit-A Rust build failed: " + completed.stderr
            )
        binary = target / "debug" / "hegel-formal-bridge-m25"
        rust_report, binary_digest = _rust_report_open_inode(
            binary,
            environment=environment,
        )
        if _sha256_directory_manifest(snapshot_project) != snapshot_manifest_sha256:
            raise M25ErrataQualificationError(
                "detached Commit-A source snapshot changed during replay"
            )
        if (
            _cargo_registry_input_manifest(cargo_home / "registry")
            != registry_manifest_sha256
        ):
            raise M25ErrataQualificationError(
                "offline Cargo registry inputs changed during replay"
            )
        _assert_no_cargo_config(snapshot_project, cargo_home)
        if (
            _sha256_file(Path(cargo)) != cargo_binary_sha256
            or _sha256_file(Path(rustc)) != rustc_binary_sha256
            or _sha256_directory_manifest(toolchain_root)
            != toolchain_manifest_sha256
            or _tool_version(
                cargo,
                "cargo",
                environment=environment,
                cwd=snapshot_project,
            )
            != cargo_version
            or _tool_version(
                rustc,
                "rustc",
                environment=environment,
                cwd=snapshot_project,
            )
            != rustc_version
        ):
            raise M25ErrataQualificationError(
                "approved toolchain binary, directory, or version changed during replay"
            )
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
            "cargo_version": cargo_version,
            "rustc_version": rustc_version,
            "rustup_toolchain_id": toolchain_policy["rustup_toolchain_id"],
            "rustup_launcher_sha256": toolchain_policy[
                "rustup_launcher_sha256"
            ],
            "cargo_toolchain_binary_sha256": cargo_binary_sha256,
            "rustc_toolchain_binary_sha256": rustc_binary_sha256,
            "toolchain_directory_manifest_sha256": (
                toolchain_manifest_sha256
            ),
            "approved_toolchain_policy_sha256": toolchain_policy_sha256,
            "approved_toolchain_policy_bound": True,
            "caller_supplied_toolchain_allowed": False,
            "offline_registry_input_manifest_sha256": registry_manifest_sha256,
            "cargo_lock_registry_archive_count": registry_archive_count,
            "cargo_lock_registry_archives_verified": True,
            "environment_profile": "HEGEL_M25_ISOLATED_BUILD_ENV_V1",
            "inherited_environment_allowed": False,
            "rustc_wrapper_allowed": False,
            "ancestor_cargo_config_loaded": False,
            "ancestor_cargo_config_absence_verified": True,
            "binary_hash_and_exec_same_open_inode": True,
            "binary_reproducible_across_ephemeral_build_paths": False,
            "normalized_command": [
                "cargo",
                "build",
                "--locked",
                "--offline",
                "--manifest-path",
                "rust/formal_bridge_m25/Cargo.toml",
                "--target-dir",
                "<FRESH_EPHEMERAL_TARGET>",
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


def _rust_report_open_inode(
    rust_binary: Path,
    *,
    environment: Mapping[str, str],
) -> tuple[dict[str, object], str]:
    """Hash and execute the same already-open binary inode via procfs."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(rust_binary, flags)
    except OSError as exc:
        raise M25ErrataQualificationError(
            f"cannot open Rust executable without following links: {exc}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise M25ErrataQualificationError("Rust executable is not a regular file")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        os.lseek(descriptor, 0, os.SEEK_SET)
        proc_path = f"/proc/self/fd/{descriptor}"
        completed = subprocess.run(
            [proc_path],
            input='{"op":"errata_vectors"}',
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
            env=dict(environment),
            pass_fds=(descriptor,),
        )
        report = _parse_rust_report(
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )
        return report, "sha256:" + digest.hexdigest()
    finally:
        os.close(descriptor)


def _rust_report(rust_binary: Path) -> dict[str, object]:
    report, _ = _rust_report_open_inode(
        rust_binary,
        environment=os.environ,
    )
    return report


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
        "rustup_toolchain_id",
        "rustup_launcher_sha256",
        "cargo_toolchain_binary_sha256",
        "rustc_toolchain_binary_sha256",
        "toolchain_directory_manifest_sha256",
        "approved_toolchain_policy_sha256",
        "approved_toolchain_policy_bound",
        "caller_supplied_toolchain_allowed",
        "offline_registry_input_manifest_sha256",
        "cargo_lock_registry_archive_count",
        "cargo_lock_registry_archives_verified",
        "environment_profile",
        "inherited_environment_allowed",
        "rustc_wrapper_allowed",
        "ancestor_cargo_config_loaded",
        "ancestor_cargo_config_absence_verified",
        "binary_hash_and_exec_same_open_inode",
        "binary_reproducible_across_ephemeral_build_paths",
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
        "rustup_toolchain_id": toolchain_policy["rustup_toolchain_id"],
        "rustup_launcher_sha256": toolchain_policy["rustup_launcher_sha256"],
        "cargo_toolchain_binary_sha256": toolchain_policy[
            "cargo_binary_sha256"
        ],
        "rustc_toolchain_binary_sha256": toolchain_policy[
            "rustc_binary_sha256"
        ],
        "toolchain_directory_manifest_sha256": toolchain_policy[
            "toolchain_directory_manifest_sha256"
        ],
        "approved_toolchain_policy_sha256": _sha256_file(
            APPROVED_TOOLCHAIN_POLICY_PATH
        ),
        "approved_toolchain_policy_bound": True,
        "caller_supplied_toolchain_allowed": False,
        "environment_profile": "HEGEL_M25_ISOLATED_BUILD_ENV_V1",
        "inherited_environment_allowed": False,
        "rustc_wrapper_allowed": False,
        "ancestor_cargo_config_loaded": False,
        "ancestor_cargo_config_absence_verified": True,
        "cargo_lock_registry_archives_verified": True,
        "binary_hash_and_exec_same_open_inode": True,
        "binary_reproducible_across_ephemeral_build_paths": False,
        "normalized_command": [
            "cargo",
            "build",
            "--locked",
            "--offline",
            "--manifest-path",
            "rust/formal_bridge_m25/Cargo.toml",
            "--target-dir",
            "<FRESH_EPHEMERAL_TARGET>",
        ],
    }
    for field, expected in expected_build_values.items():
        if not _json_type_strict_equal(rust_execution.get(field), expected):
            raise M25ErrataQualificationError(f"Rust build receipt {field} drift")
    for field in (
        "cargo_version",
        "rustc_version",
        "rustup_launcher_sha256",
        "cargo_toolchain_binary_sha256",
        "rustc_toolchain_binary_sha256",
        "toolchain_directory_manifest_sha256",
        "approved_toolchain_policy_sha256",
        "offline_registry_input_manifest_sha256",
        "source_snapshot_manifest_sha256",
    ):
        value = rust_execution.get(field)
        if not isinstance(value, str) or not value or "\n" in value:
            raise M25ErrataQualificationError(f"Rust build receipt {field} drift")
    archive_count = rust_execution.get("cargo_lock_registry_archive_count")
    if type(archive_count) is not int or archive_count <= 0:
        raise M25ErrataQualificationError(
            "Rust build receipt Cargo archive count drift"
        )
    binary_digest = rust_execution.get("binary_sha256")
    if (
        not isinstance(binary_digest, str)
        or len(binary_digest) != 71
        or not binary_digest.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in binary_digest[7:])
    ):
        raise M25ErrataQualificationError("Rust binary digest syntax drift")

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
    for nondeterministic_build_field in ("binary_sha256",):
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
