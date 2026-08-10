#!/usr/bin/env python3
"""Offline dual-isolation supervisor for Phase-3A-Q0 qualification.

The two endpoint processes receive disjoint, read-only source snapshots and
have no network.  This host parses and replays their implementation-neutral
43-field endpoint wires, performs a local target-blind replay, and creates the
host-only 40-field receipt only after every gate agrees.  Q1, Q2, M3 formal
roots, role evaluation, and outside certificates remain absent.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
from types import ModuleType
from typing import Final, Iterable, Mapping, NoReturn, Sequence


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
MODULE_ROOT: Final = PROJECT_ROOT / "src/hegel_machine"
CONFIG_PATH: Final = PROJECT_ROOT / "config/phase3_q0_dual_isolation_v1.json"
DEFAULT_CARGO_CACHE: Final = Path(
    "/home/erzhu419/.local/state/hegel-machine/rust-cargo-cache"
)
LINUX_TEMP_ROOT: Final = Path("/tmp")

# Use a private namespace so the historical package initializer, which
# exports target and split APIs, can never run in the host supervisor.
_HOST_PACKAGE: Final = "_hegel_q0_host"
if _HOST_PACKAGE not in sys.modules:
    package = ModuleType(_HOST_PACKAGE)
    package.__path__ = [str(MODULE_ROOT)]  # type: ignore[attr-defined]
    package.__package__ = _HOST_PACKAGE
    sys.modules[_HOST_PACKAGE] = package

_cbor = importlib.import_module(f"{_HOST_PACKAGE}.strict_cbor_v1")
_contract = importlib.import_module(
    f"{_HOST_PACKAGE}.phase3_q0_quotient_contract_v1"
)


SCHEMA_VERSION: Final = "hegel-phase3a-q0-dual-qualification-evidence/1"
PLAN_SCHEMA_VERSION: Final = "hegel-phase3a-q0-dual-qualification-plan/1"
STATUS_PASS: Final = "DUAL_EXHAUSTIVE_ORACLE_EQUIVALENCE_PASS"
PYTHON_SCHEMA: Final = "hegel-q0-python-micro-oracle/1"
RUST_SCHEMA: Final = "hegel-q0-rust-micro-oracle/1"
PYTHON_IMPLEMENTATION_ID: Final = "hegel-q0-python-oracle-v1"
RUST_IMPLEMENTATION_ID: Final = "hegel-rust-q0-quotient-oracle-v1"
PROGRAM_RECORD_SCHEMA_ID: Final = b"hegel-q0-syntax-program-record/1"
FIXED_POINT_STATE_SCHEMA_ID: Final = b"hegel-q0-fixed-point-state/1"
SYNTAX_PATH_ID: Final = b"hegel-q0-exhaustive-syntax-path/1"
DIRECT_PATH_ID: Final = b"hegel-q0-direct-quotient-path/1"
SYNTAX_STATE_ROOT_DOMAIN: Final = "HEGEL/Q0/SYNTAX_STATE/V1"
DIRECT_STATE_ROOT_DOMAIN: Final = "HEGEL/Q0/DIRECT_QUOTIENT_STATE/V1"
PYTHON_IMAGE: Final = (
    "python@sha256:"
    "e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3"
)
RUST_IMAGE: Final = (
    "rust@sha256:"
    "38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89"
)
PYTHON_MANIFEST_DOMAIN: Final = "HEGEL/Q0/PYTHON_IMPLEMENTATION_MANIFEST/V1"
RUST_MANIFEST_DOMAIN: Final = "HEGEL/Q0/RUST_IMPLEMENTATION_MANIFEST/V1"
HOST_MANIFEST_DOMAIN: Final = "HEGEL/Q0/HOST_REPLAY_IMPLEMENTATION_MANIFEST/V1"
ISOLATION_PROFILE_ID: Final = "hegel-phase3a-q0-dual-isolated-qualification-v1"
PYTHON_RUNTIME_IDENTITY: Final = (
    ISOLATION_PROFILE_ID,
    PYTHON_IMAGE,
    "/usr/local/bin/python3",
    "-I",
    "-S",
    "-B",
    "/workspace/tools/phase3_q0_python_oracle_entrypoint_v1.py",
)
RUST_RUNTIME_IDENTITY: Final = (
    ISOLATION_PROFILE_ID,
    RUST_IMAGE,
    "cargo",
    "run",
    "--locked",
    "--offline",
    "--quiet",
    "-j",
    "1",
    "--target",
    "x86_64-unknown-linux-gnu",
    "CARGO_BUILD_JOBS=1",
    "CARGO_NET_OFFLINE=true",
)
HOST_RUNTIME_IDENTITY: Final = (
    ISOLATION_PROFILE_ID,
    "TRUSTED_HOST_ISSUER_NOT_THIRD_ENDPOINT",
    sys.executable,
    f"python-{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    sys.platform,
)
DOCKER_EXECUTABLE: Final = "/usr/bin/docker"
DOCKER_HOST: Final = "unix:///var/run/docker.sock"
ENDPOINT_TIMEOUT_SECONDS: Final = 600
MAX_ENDPOINT_BYTES: Final = _contract.Q0_MAX_OUTPUT_BYTES
READINESS_GATE_MASK: Final = (1 << _contract.Q0_READINESS_GATE_TOTAL) - 1
READINESS_GATES: Final = tuple(
    (index, name)
    for index, name in enumerate(_contract.Q0_READINESS_GATES, start=1)
)
ISOLATION_PREREQUISITES: Final = (
    (1, "PINNED_LOCAL_IMAGES"),
    (2, "NO_NETWORK_AND_NO_PULL"),
    (3, "DISTINCT_READ_ONLY_SOURCE_SNAPSHOTS"),
    (4, "CAPABILITY_AND_PRIVILEGE_ISOLATION"),
    (5, "RESOURCE_LIMITS_ACTIVE"),
    (6, "COMMIT_TREE_SOURCE_BINDING"),
    (7, "SEALED_CARGO_HOME_IDENTITY"),
)

FAIL_CONFIG: Final = "FAIL_Q0_DUAL_ISOLATION_CONFIG"
FAIL_SOURCE: Final = "FAIL_Q0_SOURCE_BINDING"
FAIL_DOCKER: Final = "FAIL_Q0_DOCKER_ISOLATION"
FAIL_ENDPOINT: Final = "FAIL_Q0_ENDPOINT"
FAIL_WIRE: Final = "FAIL_Q0_ENDPOINT_WIRE"
FAIL_DISAGREEMENT: Final = "FAIL_Q0_IMPLEMENTATION_DISAGREEMENT"
FAIL_HOST_REPLAY: Final = "FAIL_Q0_HOST_REPLAY"
FAIL_ARTIFACT: Final = "FAIL_Q0_ARTIFACT"

FORBIDDEN_SOURCE_TOKENS: Final = (
    "target",
    "truth",
    "split",
    "phase3_dsl_v1",
)

COVERAGE_FIELDS: Final = (
    "operator_code",
    "eligible_raw",
    "strict_admitted",
    "rewrite_collapses",
    "canonical_duplicates",
    "new_canonical",
)
ROUND_FIELDS: Final = (
    "round_index",
    "queued_application_count",
    "new_canonical_program_count",
    "new_behavior_class_count",
    "frontier_mutation_count",
    "cohort_bank_mutation_count",
    "complete_state_changed",
)
COMMON_ENDPOINT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "implementation_id",
        "terminal_status",
        "dsl_version",
        "dsl_freeze_version",
        "closure_semantics_version",
        "q0_freeze_version",
        "projection_id",
        "probe_input_signature_id",
        "probe_canonical_cbor_hex",
        "probe_universe_root",
        "frozen_leaf_count",
        "canonical_syntax_count",
        "syntax_raw_operator_applications",
        "quotient_raw_operator_applications",
        "syntax_strict_admitted_applications",
        "quotient_strict_admitted_applications",
        "syntax_rewrite_collapses",
        "quotient_rewrite_collapses",
        "behavior_class_count",
        "frontier_point_count",
        "maximum_frontier_size",
        "syntax_continuation_bank_point_count",
        "quotient_continuation_bank_point_count",
        "maximum_syntax_bank_points_per_class",
        "maximum_quotient_bank_points_per_class",
        "syntax_saturation_rounds",
        "direct_saturation_rounds",
        "work_queue_empty",
        "zero_delta_full_round",
        "all_typed_operator_frontier_tuples_covered",
        "exhaustive_syntax_oracle_complete",
        "syntax_direct_states_equal",
        "final_class_delta",
        "final_frontier_delta",
        "final_bank_delta",
        "projection_manifest_root",
        "semantic_binding_root",
        "syntax_program_root",
        "syntax_class_archive_root",
        "direct_class_archive_root",
        "syntax_state_root",
        "direct_state_root",
        "syntax_saturation_state_preimage_cbor_hex",
        "direct_saturation_state_preimage_cbor_hex",
        "syntax_coverage_root",
        "direct_coverage_root",
        "syntax_coverage",
        "direct_coverage",
        "direct_rounds",
        "endpoint_state_root",
        "resource_guards_ok",
        "target_truth_accessed",
        "split_accessed",
        "role_evaluation_performed",
        "formal_roots_generated",
        "authority_claimed",
    }
)
PYTHON_ENDPOINT_FIELDS: Final = COMMON_ENDPOINT_FIELDS | {
    "python_source_root",
    "endpoint_state_cbor_hex",
}
RUST_ENDPOINT_FIELDS: Final = COMMON_ENDPOINT_FIELDS | {"rust_source_root"}
IMPLEMENTATION_SPECIFIC_FIELDS: Final = {
    "schema_version",
    "implementation_id",
    "python_source_root",
    "rust_source_root",
    "endpoint_state_cbor_hex",
    "direct_rounds",
}


class SupervisorError(RuntimeError):
    """Stable fail-closed supervisor error."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise SupervisorError(code, detail)


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )


def _exact_dict(value: object, fields: Iterable[str], name: str) -> dict[str, object]:
    expected = set(fields)
    if type(value) is not dict or set(value) != expected:
        observed = set(value) if type(value) is dict else type(value).__name__
        _fail(FAIL_WIRE, f"{name} fields differ: {observed!r}")
    return value


def _strict_json_object_v1(payload: bytes, name: str) -> dict[str, object]:
    try:
        text_value = payload.decode("ascii")
        if len(text_value.splitlines()) != 1 or not text_value.endswith("\n"):
            _fail(FAIL_ENDPOINT, f"{name} stdout is not one JSON line")

        def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
            value: dict[str, object] = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError(f"duplicate JSON key: {key}")
                value[key] = item
            return value

        def reject_constant(value: str) -> NoReturn:
            raise ValueError(f"non-finite JSON constant: {value}")

        decoded = json.loads(
            text_value,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, ValueError) as error:
        _fail(FAIL_ENDPOINT, f"{name} stdout is not strict JSON: {error}")
    if type(decoded) is not dict:
        _fail(FAIL_ENDPOINT, f"{name} endpoint JSON is not an object")
    if _canonical_json_bytes(decoded) != payload:
        _fail(FAIL_ENDPOINT, f"{name} stdout is not canonical JSON bytes")
    return decoded


def _root_bytes(value: object, name: str) -> bytes:
    if type(value) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        _fail(FAIL_WIRE, f"{name} is not a canonical SHA-256 root ID")
    return bytes.fromhex(value[7:])


def _uint(value: object, name: str, maximum: int = (1 << 64) - 1) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        _fail(FAIL_WIRE, f"{name} is outside uint range 0..{maximum}")
    return value


def _bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        _fail(FAIL_WIRE, f"{name} is not a boolean")
    return value


def load_isolation_config(project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    path = project_root / "config/phase3_q0_dual_isolation_v1.json"
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, ValueError) as error:
        _fail(FAIL_CONFIG, f"cannot load isolation config: {error}")
    if type(value) is not dict:
        _fail(FAIL_CONFIG, "isolation config is not an object")
    expected_top = {
        "schema_version",
        "profile_id",
        "claim_scope",
        "images",
        "docker",
        "python",
        "rust",
        "isolation",
        "host_role",
        "isolation_prerequisites",
        "readiness_gates",
        "downstream_state",
    }
    if set(value) != expected_top:
        _fail(FAIL_CONFIG, "isolation config top-level fields differ")
    if (
        value["schema_version"] != "hegel-phase3a-q0-dual-isolation/1"
        or value["profile_id"]
        != "hegel-phase3a-q0-dual-isolated-qualification-v1"
        or value["claim_scope"] != "Q0_DUAL_ENGINEERING_QUALIFICATION_ONLY"
    ):
        _fail(FAIL_CONFIG, "isolation profile identity differs")
    images = _exact_dict(value["images"], {"python", "rust"}, "images")
    if images != {"python": PYTHON_IMAGE, "rust": RUST_IMAGE}:
        _fail(FAIL_CONFIG, "pinned image references differ")
    docker = _exact_dict(
        value["docker"],
        {
            "executable",
            "host",
            "pull_policy",
            "network",
            "root_filesystem_read_only",
            "source_mount_read_only",
            "cap_drop",
            "no_new_privileges",
            "memory",
            "memory_swap",
            "pids_limit",
            "nofile_ulimit",
            "tmpfs",
        },
        "docker",
    )
    expected_docker = {
        "executable": DOCKER_EXECUTABLE,
        "host": DOCKER_HOST,
        "pull_policy": "never",
        "network": "none",
        "root_filesystem_read_only": True,
        "source_mount_read_only": True,
        "cap_drop": "ALL",
        "no_new_privileges": True,
        "memory": "512m",
        "memory_swap": "512m",
        "pids_limit": 64,
        "nofile_ulimit": "128:128",
        "tmpfs": "/tmp:rw,exec,nosuid,nodev,size=512m,mode=1777",
    }
    if docker != expected_docker:
        _fail(FAIL_CONFIG, "Docker isolation profile differs")
    python_profile = _exact_dict(
        value["python"],
        {"flags", "entrypoint", "source_manifest_domain"},
        "python profile",
    )
    if python_profile != {
        "flags": ["-I", "-S", "-B"],
        "entrypoint": "tools/phase3_q0_python_oracle_entrypoint_v1.py",
        "source_manifest_domain": PYTHON_MANIFEST_DOMAIN,
    }:
        _fail(FAIL_CONFIG, "Python isolation profile differs")
    rust_profile = _exact_dict(
        value["rust"],
        {
            "crate",
            "cargo_flags",
            "cargo_build_jobs",
            "target_triple",
            "source_manifest_domain",
            "cargo_cache_access",
            "cargo_cache_is_implementation_identity",
            "locked_registry_dependency_manifest_is_implementation_identity",
        },
        "Rust profile",
    )
    if rust_profile != {
        "crate": "rust/q0_quotient_oracle",
        "cargo_flags": ["run", "--locked", "--offline", "--quiet"],
        "cargo_build_jobs": 1,
        "target_triple": "x86_64-unknown-linux-gnu",
        "source_manifest_domain": RUST_MANIFEST_DOMAIN,
        "cargo_cache_access": "ro-offline-dependency-cache",
        "cargo_cache_is_implementation_identity": True,
        "locked_registry_dependency_manifest_is_implementation_identity": True,
    }:
        _fail(FAIL_CONFIG, "Rust isolation profile differs")
    isolation = _exact_dict(
        value["isolation"],
        {
            "distinct_source_snapshots",
            "endpoint_output_exchange",
            "target_truth_split_sources_present",
            "host_replay_after_both_endpoints",
            "same_admin_controller",
            "organizational_independence",
            "technical_process_and_filesystem_independence",
        },
        "isolation disclosure",
    )
    if isolation != {
        "distinct_source_snapshots": True,
        "endpoint_output_exchange": False,
        "target_truth_split_sources_present": False,
        "host_replay_after_both_endpoints": True,
        "same_admin_controller": True,
        "organizational_independence": False,
        "technical_process_and_filesystem_independence": True,
    }:
        _fail(FAIL_CONFIG, "isolation disclosure differs")
    host_role = _exact_dict(
        value["host_role"],
        {
            "trusted_issuer",
            "third_independent_endpoint",
            "filesystem_hard_isolation",
            "target_blind_import_manifest_required",
        },
        "host role",
    )
    if host_role != {
        "trusted_issuer": True,
        "third_independent_endpoint": False,
        "filesystem_hard_isolation": False,
        "target_blind_import_manifest_required": True,
    }:
        _fail(FAIL_CONFIG, "trusted host issuer role differs")
    if value["isolation_prerequisites"] != [
        list(row) for row in ISOLATION_PREREQUISITES
    ]:
        _fail(FAIL_CONFIG, "isolation prerequisite registry differs")
    gates = value["readiness_gates"]
    if gates != [list(row) for row in READINESS_GATES]:
        _fail(FAIL_CONFIG, "readiness gate registry is not exact 1..14")
    downstream = value["downstream_state"]
    if downstream != {
        "q1_status_id": 0,
        "q1_output_root": None,
        "q2_status_id": 0,
        "role_evaluation_performed": False,
        "m3_formal_roots": None,
        "outside_certificate_issued": False,
    }:
        _fail(FAIL_CONFIG, "downstream NOT_RUN/null state differs")
    return value


@dataclass(frozen=True, slots=True)
class SourceFileV1:
    path: str
    mode: int
    size: int
    digest: bytes

    def canonical_object(self) -> tuple[object, ...]:
        return (self.path.encode("utf-8"), self.mode, self.size, self.digest)

    def json_object(self) -> dict[str, object]:
        return {
            "path": self.path,
            "mode": f"{self.mode:o}",
            "size": self.size,
            "sha256": self.digest.hex(),
        }


@dataclass(frozen=True, slots=True)
class RegistryDependencyV1:
    name: str
    version: str
    source: str
    lock_checksum: bytes
    archive_digest: bytes
    files: tuple[SourceFileV1, ...]

    def canonical_object(self) -> tuple[object, ...]:
        return (
            self.name.encode("utf-8"),
            self.version.encode("ascii"),
            self.source.encode("ascii"),
            self.lock_checksum,
            self.archive_digest,
            tuple(row.canonical_object() for row in self.files),
        )

    def json_object(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "source": self.source,
            "lock_checksum": self.lock_checksum.hex(),
            "archive_sha256": self.archive_digest.hex(),
            "file_count": len(self.files),
            "files": [row.json_object() for row in self.files],
        }


@dataclass(frozen=True, slots=True)
class SourceManifestV1:
    implementation: str
    domain: str
    files: tuple[SourceFileV1, ...]
    registry_dependencies: tuple[RegistryDependencyV1, ...] = ()
    dependency_files: tuple[SourceFileV1, ...] = ()
    target_triple: str | None = None
    runtime_identity: tuple[str, ...] = ()

    @property
    def root(self) -> bytes:
        return _cbor.content_hash(
            self.domain,
            (
                self.implementation.encode("ascii"),
                None if self.target_triple is None else self.target_triple.encode("ascii"),
                tuple(row.canonical_object() for row in self.files),
                tuple(row.canonical_object() for row in self.registry_dependencies),
                tuple(row.canonical_object() for row in self.dependency_files),
                tuple(value.encode("ascii") for value in self.runtime_identity),
            ),
        )

    def json_object(self) -> dict[str, object]:
        return {
            "implementation": self.implementation,
            "domain": self.domain,
            "file_count": len(self.files),
            "registry_dependency_count": len(self.registry_dependencies),
            "registry_dependency_file_count": sum(
                len(row.files) for row in self.registry_dependencies
            ),
            "dependency_cache_file_count": len(self.dependency_files),
            "target_triple": self.target_triple,
            "runtime_identity": list(self.runtime_identity),
            "implementation_root": "sha256:" + self.root.hex(),
            "files": [row.json_object() for row in self.files],
            "registry_dependencies": [
                row.json_object() for row in self.registry_dependencies
            ],
            "dependency_cache_files": [
                row.json_object() for row in self.dependency_files
            ],
        }


def _source_row(project_root: Path, relative: str) -> SourceFileV1:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        _fail(FAIL_SOURCE, f"noncanonical source path: {relative}")
    lowered = relative.lower()
    if any(token in lowered for token in FORBIDDEN_SOURCE_TOKENS):
        _fail(FAIL_SOURCE, f"forbidden target/truth/split source: {relative}")
    path = project_root / relative
    try:
        file_stat = path.stat()
        payload = path.read_bytes()
    except OSError as error:
        _fail(FAIL_SOURCE, f"cannot read source {relative}: {error}")
    if not stat.S_ISREG(file_stat.st_mode) or path.is_symlink():
        _fail(FAIL_SOURCE, f"source is not a regular non-symlink file: {relative}")
    mode = 0o100755 if file_stat.st_mode & stat.S_IXUSR else 0o100644
    return SourceFileV1(relative, mode, len(payload), sha256(payload).digest())


def _python_module_path(project_root: Path, module: str) -> Path | None:
    prefix = "hegel_machine."
    if not module.startswith(prefix):
        return None
    suffix = module[len(prefix) :]
    if not suffix or "." in suffix:
        return None
    candidate = project_root / "src/hegel_machine" / f"{suffix}.py"
    return candidate if candidate.is_file() else None


def _local_imports(project_root: Path, path: Path, module: str | None) -> set[str]:
    try:
        tree = ast.parse(path.read_bytes(), filename=str(path))
    except (OSError, SyntaxError) as error:
        _fail(FAIL_SOURCE, f"cannot parse Python dependency {path}: {error}")
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("hegel_machine."):
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module == "hegel_machine":
                    found.update(f"hegel_machine.{alias.name}" for alias in node.names)
                elif node.module and node.module.startswith("hegel_machine."):
                    found.add(node.module)
                continue
            if module is None:
                continue
            package_parts = module.split(".")[:-1]
            remove = node.level - 1
            if remove > len(package_parts):
                _fail(FAIL_SOURCE, f"relative import escapes package in {path}")
            anchor = package_parts[: len(package_parts) - remove]
            if node.module:
                base = ".".join((*anchor, *node.module.split(".")))
                found.add(base)
            else:
                found.update(".".join((*anchor, alias.name)) for alias in node.names)
    return found


def python_source_manifest_v1(project_root: Path = PROJECT_ROOT) -> SourceManifestV1:
    entry = project_root / "tools/phase3_q0_python_oracle_entrypoint_v1.py"
    pending = _local_imports(project_root, entry, None)
    modules: set[str] = set()
    while pending:
        module = min(pending)
        pending.remove(module)
        if module in modules:
            continue
        path = _python_module_path(project_root, module)
        if path is None:
            continue
        relative = path.relative_to(project_root).as_posix()
        if any(token in relative.lower() for token in FORBIDDEN_SOURCE_TOKENS):
            _fail(FAIL_SOURCE, f"Python import closure reaches forbidden source: {relative}")
        modules.add(module)
        pending.update(_local_imports(project_root, path, module) - modules)
    paths = {
        "tools/phase3_q0_python_oracle_entrypoint_v1.py",
        "src/hegel_machine/phase3_q0_gate_qualification_v1.py",
        "config/phase3_q0_quotient_freeze_v1.json",
    }
    paths.update(
        _python_module_path(project_root, module).relative_to(project_root).as_posix()
        for module in modules
        if _python_module_path(project_root, module) is not None
    )
    if "src/hegel_machine/__init__.py" in paths:
        _fail(FAIL_SOURCE, "historical package initializer entered Q0 source closure")
    rows = tuple(_source_row(project_root, path) for path in sorted(paths))
    return SourceManifestV1(
        "python",
        PYTHON_MANIFEST_DOMAIN,
        rows,
        runtime_identity=PYTHON_RUNTIME_IDENTITY,
    )


def host_replay_source_manifest_v1(
    project_root: Path,
    host: "HostReplayV1",
) -> SourceManifestV1:
    paths = {
        "tools/phase3_q0_dual_qualification_v1.py",
        "config/phase3_q0_dual_isolation_v1.json",
        "config/phase3_q0_quotient_freeze_v1.json",
    }
    for name in host.loaded_modules:
        module = sys.modules.get(name)
        module_file = None if module is None else getattr(module, "__file__", None)
        leaf = name.rsplit(".", 1)[-1]
        expected_file = project_root / f"src/hegel_machine/{leaf}.py"
        if (
            type(module_file) is not str
            or not expected_file.is_file()
            or expected_file.is_symlink()
            or Path(module_file).is_symlink()
            or Path(module_file).resolve() != expected_file.resolve()
        ):
            _fail(FAIL_HOST_REPLAY, f"host local module file differs: {name}")
        paths.add(f"src/hegel_machine/{leaf}.py")
    if not host.loaded_modules:
        _fail(FAIL_HOST_REPLAY, "host replay loaded-module manifest is empty")
    rows = tuple(_source_row(project_root, path) for path in sorted(paths))
    return SourceManifestV1(
        "host",
        HOST_MANIFEST_DOMAIN,
        rows,
        runtime_identity=HOST_RUNTIME_IDENTITY,
    )


_PATH_DEPENDENCY = re.compile(r"\{[^}\n]*\bpath\s*=\s*\"([^\"]+)\"[^}\n]*\}")


_LOCK_PACKAGE_BLOCK = re.compile(
    r"(?ms)^\[\[package\]\]\n(.*?)(?=^\[\[package\]\]\n|\Z)"
)


def _lock_text_field(block: str, field: str) -> str | None:
    match = re.search(rf'(?m)^{re.escape(field)} = "([^"]+)"$', block)
    return None if match is None else match.group(1)


def _crate_archive_material(
    archive: Path, stem: str
) -> tuple[tuple[SourceFileV1, bytes], ...]:
    rows: list[tuple[SourceFileV1, bytes]] = []
    try:
        with tarfile.open(archive, mode="r:gz") as bundle:
            for member in sorted(bundle.getmembers(), key=lambda value: value.name):
                pure = PurePosixPath(member.name)
                if (
                    pure.is_absolute()
                    or any(part in {"", ".", ".."} for part in pure.parts)
                    or not pure.parts
                    or pure.parts[0] != stem
                ):
                    _fail(FAIL_SOURCE, f"unsafe crate archive path: {member.name}")
                if member.isdir():
                    continue
                if not member.isfile() or len(pure.parts) < 2:
                    _fail(FAIL_SOURCE, f"crate archive has non-regular member: {member.name}")
                stream = bundle.extractfile(member)
                if stream is None:
                    _fail(FAIL_SOURCE, f"cannot read crate archive member: {member.name}")
                payload = stream.read()
                relative = PurePosixPath(*pure.parts[1:]).as_posix()
                mode = 0o100755 if member.mode & 0o100 else 0o100644
                rows.append(
                    (
                        SourceFileV1(
                            relative,
                            mode,
                            len(payload),
                            sha256(payload).digest(),
                        ),
                        payload,
                    )
                )
    except (OSError, tarfile.TarError) as error:
        _fail(FAIL_SOURCE, f"cannot audit crate archive {stem}: {error}")
    if not rows or len({row.path for row, _ in rows}) != len(rows):
        _fail(FAIL_SOURCE, f"crate archive files are empty or duplicated: {stem}")
    return tuple(rows)


def _crate_archive_rows(archive: Path, stem: str) -> tuple[SourceFileV1, ...]:
    return tuple(row for row, _ in _crate_archive_material(archive, stem))


def _registry_dependency_rows(
    lock_path: Path, cargo_cache: Path
) -> tuple[RegistryDependencyV1, ...]:
    try:
        lock_text = lock_path.read_text(encoding="utf-8")
    except OSError as error:
        _fail(FAIL_SOURCE, f"cannot read Q0 Cargo.lock: {error}")
    locked: list[tuple[str, str, str, bytes]] = []
    for block in _LOCK_PACKAGE_BLOCK.findall(lock_text):
        source = _lock_text_field(block, "source")
        if source is None:
            continue
        name = _lock_text_field(block, "name")
        version = _lock_text_field(block, "version")
        checksum = _lock_text_field(block, "checksum")
        if (
            name is None
            or version is None
            or checksum is None
            or source != "registry+https://github.com/rust-lang/crates.io-index"
            or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
        ):
            _fail(FAIL_SOURCE, "Cargo.lock registry package identity is incomplete")
        locked.append((name, version, source, bytes.fromhex(checksum)))
    if not locked or len(set(locked)) != len(locked):
        _fail(FAIL_SOURCE, "Cargo.lock registry package set is empty or duplicated")
    rows: list[RegistryDependencyV1] = []
    for name, version, source, checksum in sorted(locked):
        stem = f"{name}-{version}"
        archives = sorted((cargo_cache / "registry/cache").glob(f"*/{stem}.crate"))
        sources = sorted((cargo_cache / "registry/src").glob(f"*/{stem}"))
        if len(archives) != 1 or len(sources) > 1:
            _fail(FAIL_SOURCE, f"locked registry archive/source is not unique: {stem}")
        archive_digest = sha256(archives[0].read_bytes()).digest()
        if archive_digest != checksum:
            _fail(FAIL_SOURCE, f"crate archive checksum differs from Cargo.lock: {stem}")
        archive_rows = _crate_archive_rows(archives[0], stem)
        if sources:
            file_rows: list[SourceFileV1] = []
            for path in sorted(sources[0].rglob("*")):
                if path.is_symlink():
                    _fail(FAIL_SOURCE, f"registry dependency contains symlink: {stem}")
                if not path.is_file():
                    continue
                relative = path.relative_to(sources[0]).as_posix()
                payload = path.read_bytes()
                file_stat = path.stat()
                mode = 0o100755 if file_stat.st_mode & stat.S_IXUSR else 0o100644
                file_rows.append(
                    SourceFileV1(relative, mode, len(payload), sha256(payload).digest())
                )
            file_rows.sort(key=lambda row: row.path)
            unpacked_archive_rows = tuple(
                row for row in file_rows if row.path != ".cargo-ok"
            )
            if unpacked_archive_rows != archive_rows:
                _fail(
                    FAIL_SOURCE,
                    f"unpacked registry source differs from checksum-bound archive: {stem}",
                )
        rows.append(
            RegistryDependencyV1(
                name,
                version,
                source,
                checksum,
                archive_digest,
                archive_rows,
            )
        )
    return tuple(rows)


def _sealed_snapshot_file_rows_v1(
    root: Path,
    label: str,
) -> tuple[SourceFileV1, ...]:
    if not root.is_dir() or root.is_symlink():
        _fail(FAIL_SOURCE, f"{label} snapshot root is absent or noncanonical")
    rows: list[SourceFileV1] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            _fail(FAIL_SOURCE, f"{label} snapshot contains symlink: {path}")
        if not path.is_file():
            continue
        file_stat = path.stat()
        if not stat.S_ISREG(file_stat.st_mode):
            _fail(FAIL_SOURCE, f"{label} snapshot contains non-regular file: {path}")
        mode_bits = stat.S_IMODE(file_stat.st_mode)
        if mode_bits not in {0o444, 0o555}:
            _fail(
                FAIL_SOURCE,
                f"{label} snapshot file mode is not sealed read-only: {path}",
            )
        relative = path.relative_to(root).as_posix()
        payload = path.read_bytes()
        mode = 0o100755 if mode_bits == 0o555 else 0o100644
        rows.append(SourceFileV1(relative, mode, len(payload), sha256(payload).digest()))
    if not rows:
        _fail(FAIL_SOURCE, f"{label} snapshot contains no regular files")
    return tuple(sorted(rows, key=lambda row: row.path))


def _cargo_home_file_rows(cargo_cache: Path) -> tuple[SourceFileV1, ...]:
    return _sealed_snapshot_file_rows_v1(cargo_cache, "Cargo home")


def _sealed_cargo_home_material(
    cargo_cache: Path,
    dependencies: Sequence[RegistryDependencyV1],
) -> tuple[tuple[SourceFileV1, bytes], ...]:
    material: dict[str, tuple[SourceFileV1, bytes]] = {}

    def add(path: str, payload: bytes, mode: int = 0o100644) -> None:
        row = SourceFileV1(path, mode, len(payload), sha256(payload).digest())
        prior = material.get(path)
        if prior is not None and prior != (row, payload):
            _fail(FAIL_SOURCE, f"sealed Cargo home path collision: {path}")
        material[path] = (row, payload)

    registry_ids: set[str] = set()
    for dependency in dependencies:
        stem = f"{dependency.name}-{dependency.version}"
        archives = sorted((cargo_cache / "registry/cache").glob(f"*/{stem}.crate"))
        if len(archives) != 1:
            _fail(FAIL_SOURCE, f"locked crate archive is not unique: {stem}")
        registry_id = archives[0].parent.name
        registry_ids.add(registry_id)
        archive_payload = archives[0].read_bytes()
        if sha256(archive_payload).digest() != dependency.lock_checksum:
            _fail(FAIL_SOURCE, f"sealed crate archive checksum differs: {stem}")
        add(f"registry/cache/{registry_id}/{stem}.crate", archive_payload)
        archive_material = _crate_archive_material(archives[0], stem)
        if tuple(row for row, _ in archive_material) != dependency.files:
            _fail(FAIL_SOURCE, f"sealed crate archive file manifest differs: {stem}")
        for row, payload in archive_material:
            add(
                f"registry/src/{registry_id}/{stem}/{row.path}",
                payload,
                row.mode,
            )
        add(f"registry/src/{registry_id}/{stem}/.cargo-ok", b'{"v":1}')

        index_root = cargo_cache / "registry/index" / registry_id
        candidates = sorted(
            path
            for path in (index_root / ".cache").rglob(dependency.name)
            if path.is_file() and path.name == dependency.name
        )
        if len(candidates) != 1:
            _fail(FAIL_SOURCE, f"registry index cache entry is not unique: {dependency.name}")
        relative_index = candidates[0].relative_to(index_root).as_posix()
        add(
            f"registry/index/{registry_id}/{relative_index}",
            candidates[0].read_bytes(),
        )
    if len(registry_ids) != 1:
        _fail(FAIL_SOURCE, "locked registry packages span multiple registry identities")
    registry_id = next(iter(registry_ids))
    config_path = cargo_cache / "registry/index" / registry_id / "config.json"
    if not config_path.is_file():
        _fail(FAIL_SOURCE, "registry index config.json is absent")
    add(f"registry/index/{registry_id}/config.json", config_path.read_bytes())
    add(".package-cache", b"")
    add(".package-cache-mutate", b"")
    return tuple(material[path] for path in sorted(material))


def rust_source_manifest_v1(
    project_root: Path = PROJECT_ROOT,
    cargo_cache: Path = DEFAULT_CARGO_CACHE,
) -> SourceManifestV1:
    root_crate = project_root / "rust/q0_quotient_oracle"
    pending = [root_crate]
    crates: set[Path] = set()
    while pending:
        crate = pending.pop()
        crate = crate.resolve()
        if crate in crates:
            continue
        try:
            relative_crate = crate.relative_to(project_root.resolve())
        except ValueError:
            _fail(FAIL_SOURCE, f"Rust path dependency escapes project: {crate}")
        manifest = crate / "Cargo.toml"
        try:
            text = manifest.read_text(encoding="utf-8")
        except OSError as error:
            _fail(FAIL_SOURCE, f"cannot read {relative_crate}/Cargo.toml: {error}")
        crates.add(crate)
        for dependency in _PATH_DEPENDENCY.findall(text):
            pending.append((crate / dependency).resolve())
    paths: set[str] = {"rust/q0_quotient_oracle/Cargo.lock"}
    for crate in crates:
        paths.add((crate / "Cargo.toml").relative_to(project_root).as_posix())
        build_script = crate / "build.rs"
        if build_script.is_file():
            paths.add(build_script.relative_to(project_root).as_posix())
        for source in (crate / "src").rglob("*.rs"):
            paths.add(source.relative_to(project_root).as_posix())
    rows = tuple(_source_row(project_root, path) for path in sorted(paths))
    dependencies = _registry_dependency_rows(
        project_root / "rust/q0_quotient_oracle/Cargo.lock", cargo_cache
    )
    cargo_material = _sealed_cargo_home_material(cargo_cache, dependencies)
    return SourceManifestV1(
        "rust",
        RUST_MANIFEST_DOMAIN,
        rows,
        dependencies,
        tuple(row for row, _ in cargo_material),
        "x86_64-unknown-linux-gnu",
        RUST_RUNTIME_IDENTITY,
    )


def materialize_cargo_home_snapshot_v1(
    cargo_cache: Path,
    manifest: SourceManifestV1,
    destination: Path,
) -> None:
    if destination.exists():
        _fail(FAIL_SOURCE, f"Cargo snapshot destination already exists: {destination}")
    destination.mkdir(parents=True, mode=0o755)
    material = _sealed_cargo_home_material(
        cargo_cache, manifest.registry_dependencies
    )
    if tuple(row for row, _ in material) != manifest.dependency_files:
        _fail(FAIL_SOURCE, "sealed Cargo home changed during snapshot")
    for row, payload in material:
        output = destination / row.path
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(payload)
        output.chmod(0o555 if row.mode == 0o100755 else 0o444)
    if _cargo_home_file_rows(destination) != manifest.dependency_files:
        _fail(FAIL_SOURCE, "materialized Cargo home manifest differs")


def materialize_source_snapshot_v1(
    project_root: Path,
    manifest: SourceManifestV1,
    destination: Path,
    source_commit: str | None = None,
) -> None:
    if destination.exists():
        _fail(FAIL_SOURCE, f"snapshot destination already exists: {destination}")
    destination.mkdir(parents=True, mode=0o755)
    for row in manifest.files:
        payload = (
            (project_root / row.path).read_bytes()
            if source_commit is None
            else _git_bytes(project_root, "show", f"{source_commit}:./{row.path}")
        )
        if len(payload) != row.size or sha256(payload).digest() != row.digest:
            _fail(FAIL_SOURCE, f"source changed during snapshot: {row.path}")
        output = destination / row.path
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(payload)
        output.chmod(0o555 if row.mode == 0o100755 else 0o444)
    if _sealed_snapshot_file_rows_v1(destination, "source") != manifest.files:
        _fail(FAIL_SOURCE, "materialized source snapshot manifest differs")
    # Directories remain owner-writable only so the host can delete the
    # ephemeral snapshot.  Docker receives the entire tree through a `:ro`
    # bind mount and a read-only container root filesystem.


def _docker_common(image: str, name: str) -> list[str]:
    return [
        DOCKER_EXECUTABLE,
        f"--host={DOCKER_HOST}",
        "run",
        "--rm",
        "--name",
        name,
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--memory=512m",
        "--memory-swap=512m",
        "--pids-limit=64",
        "--ulimit=nofile=128:128",
        "--tmpfs=/tmp:rw,exec,nosuid,nodev,size=512m,mode=1777",
        f"--user={os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
        "-e",
        "LANG=C.UTF-8",
        "-e",
        "LC_ALL=C.UTF-8",
        "-e",
        "TZ=UTC",
        image,
    ]


def python_endpoint_command(snapshot: Path, name: str = "hegel-q0-python") -> list[str]:
    command = _docker_common(PYTHON_IMAGE, name)
    image = command.pop()
    command.extend(
        [
            "-e",
            "PYTHONHASHSEED=0",
            "-v",
            f"{snapshot.resolve()}:/workspace:ro",
            "-w",
            "/workspace",
            image,
            "/usr/local/bin/python3",
            "-I",
            "-S",
            "-B",
            "/workspace/tools/phase3_q0_python_oracle_entrypoint_v1.py",
        ]
    )
    return command


def rust_endpoint_command(
    snapshot: Path,
    cargo_cache: Path,
    name: str = "hegel-q0-rust",
) -> list[str]:
    command = _docker_common(RUST_IMAGE, name)
    image = command.pop()
    command.extend(
        [
            "-e",
            "CARGO_HOME=/cargo-home",
            "-e",
            "CARGO_TARGET_DIR=/tmp/cargo-target",
            "-e",
            "CARGO_NET_OFFLINE=true",
            "-e",
            "CARGO_BUILD_JOBS=1",
            "-v",
            f"{snapshot.resolve()}:/workspace:ro",
            "-v",
            f"{cargo_cache.resolve()}:/cargo-home:ro",
            "-w",
            "/workspace/rust/q0_quotient_oracle",
            image,
            "cargo",
            "run",
            "--locked",
            "--offline",
            "--quiet",
            "-j",
            "1",
            "--target",
            "x86_64-unknown-linux-gnu",
        ]
    )
    return command


@dataclass(frozen=True, slots=True)
class EndpointRunV1:
    implementation: str
    stdout: bytes
    report: dict[str, object]


def _run_endpoint(
    implementation: str,
    command: Sequence[str],
    environment: Mapping[str, str],
) -> EndpointRunV1:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=ENDPOINT_TIMEOUT_SECONDS,
            env=dict(environment),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        _fail(FAIL_DOCKER, f"{implementation} endpoint could not run: {error}")
    if completed.returncode != 0:
        _fail(
            FAIL_ENDPOINT,
            f"{implementation} exit={completed.returncode}, stderr={completed.stderr[:512]!r}",
        )
    if completed.stderr:
        _fail(FAIL_ENDPOINT, f"{implementation} emitted unexpected stderr")
    if not 0 < len(completed.stdout) <= MAX_ENDPOINT_BYTES:
        _fail(FAIL_ENDPOINT, f"{implementation} stdout size is outside guard")
    value = _strict_json_object_v1(completed.stdout, implementation)
    return EndpointRunV1(implementation, completed.stdout, value)


def run_endpoints_parallel_v1(
    python_command: Sequence[str],
    rust_command: Sequence[str],
    environment: Mapping[str, str],
) -> tuple[EndpointRunV1, EndpointRunV1]:
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="hegel-q0") as pool:
        python_future = pool.submit(
            _run_endpoint, "python", python_command, environment
        )
        rust_future = pool.submit(_run_endpoint, "rust", rust_command, environment)
        return python_future.result(), rust_future.result()


def _coverage_records(report: Mapping[str, object], field: str) -> tuple[tuple[int, ...], ...]:
    value = report[field]
    if type(value) is not list or len(value) != len(_contract.Q0_COVERAGE_CODES):
        _fail(FAIL_WIRE, f"{field} does not contain 27 rows")
    rows: list[tuple[int, ...]] = []
    for index, row_value in enumerate(value):
        row = _exact_dict(row_value, COVERAGE_FIELDS, f"{field}[{index}]")
        material = tuple(_uint(row[name], f"{field}[{index}].{name}") for name in COVERAGE_FIELDS)
        rows.append(material)
    if tuple(row[0] for row in rows) != tuple(_contract.Q0_COVERAGE_CODES):
        _fail(FAIL_WIRE, f"{field} operator registry/order differs")
    return tuple(rows)


def _validate_rounds(report: Mapping[str, object]) -> None:
    value = report["direct_rounds"]
    count = _uint(
        report["direct_saturation_rounds"],
        "direct_saturation_rounds",
        _contract.Q0_MAX_SATURATION_ROUNDS,
    )
    if type(value) is not list or len(value) != count or count == 0:
        _fail(FAIL_WIRE, "direct_rounds length differs from saturation count")
    for index, row_value in enumerate(value, start=1):
        row = _exact_dict(row_value, ROUND_FIELDS, f"direct_rounds[{index}]")
        if _uint(row["round_index"], "round_index") != index:
            _fail(FAIL_WIRE, "direct round indices are not canonical 1..N")
        for field in ROUND_FIELDS[1:-1]:
            _uint(row[field], f"direct_rounds[{index}].{field}")
        _bool(row["complete_state_changed"], "complete_state_changed")
    final = value[-1]
    if any(final[name] != 0 for name in ROUND_FIELDS[1:-1]) or final[ROUND_FIELDS[-1]] is not False:
        _fail(FAIL_WIRE, "terminal direct round is not full zero-delta")


def endpoint_state_object_v1(report: Mapping[str, object]) -> tuple[object, ...]:
    return (
        1,
        _contract.ENDPOINT_STATE_SCHEMA_ID,
        str(report["q0_freeze_version"]).encode("ascii"),
        str(report["dsl_version"]).encode("ascii"),
        str(report["closure_semantics_version"]).encode("ascii"),
        str(report["projection_id"]).encode("ascii"),
        _root_bytes(report["probe_universe_root"], "probe_universe_root"),
        _root_bytes(report["projection_manifest_root"], "projection_manifest_root"),
        _root_bytes(report["semantic_binding_root"], "semantic_binding_root"),
        str(report["terminal_status"]).encode("ascii"),
        _uint(report["syntax_raw_operator_applications"], "syntax_raw"),
        _uint(report["quotient_raw_operator_applications"], "quotient_raw"),
        _uint(report["syntax_strict_admitted_applications"], "syntax_admitted"),
        _uint(report["quotient_strict_admitted_applications"], "quotient_admitted"),
        _uint(report["syntax_rewrite_collapses"], "syntax_rewrites"),
        _uint(report["quotient_rewrite_collapses"], "quotient_rewrites"),
        _uint(report["canonical_syntax_count"], "canonical_syntax"),
        _uint(report["behavior_class_count"], "behavior_classes"),
        _uint(report["frontier_point_count"], "frontier_points"),
        _uint(report["maximum_frontier_size"], "maximum_frontier"),
        _uint(report["syntax_continuation_bank_point_count"], "syntax_bank"),
        _uint(report["quotient_continuation_bank_point_count"], "quotient_bank"),
        _uint(report["maximum_syntax_bank_points_per_class"], "maximum_syntax_bank"),
        _uint(report["maximum_quotient_bank_points_per_class"], "maximum_quotient_bank"),
        _uint(report["direct_saturation_rounds"], "direct_rounds"),
        _bool(report["work_queue_empty"], "work_queue_empty"),
        _bool(report["zero_delta_full_round"], "zero_delta_full_round"),
        _uint(report["final_class_delta"], "final_class_delta"),
        _uint(report["final_frontier_delta"], "final_frontier_delta"),
        _uint(report["final_bank_delta"], "final_bank_delta"),
        _root_bytes(report["syntax_program_root"], "syntax_program_root"),
        _root_bytes(report["syntax_class_archive_root"], "syntax_class_root"),
        _root_bytes(report["direct_class_archive_root"], "direct_class_root"),
        _root_bytes(report["syntax_coverage_root"], "syntax_coverage_root"),
        _root_bytes(report["direct_coverage_root"], "direct_coverage_root"),
        _root_bytes(report["syntax_state_root"], "syntax_state_root"),
        _root_bytes(report["direct_state_root"], "direct_state_root"),
        _bool(report["resource_guards_ok"], "resource_guards_ok"),
        _bool(report["target_truth_accessed"], "target_truth_accessed"),
        _bool(report["split_accessed"], "split_accessed"),
        _bool(report["role_evaluation_performed"], "role_evaluation_performed"),
        _bool(report["formal_roots_generated"], "formal_roots_generated"),
        _bool(report["authority_claimed"], "authority_claimed"),
    )


@dataclass(frozen=True, slots=True)
class ValidatedEndpointV1:
    implementation: str
    report: dict[str, object]
    canonical_state: tuple[object, ...]
    canonical_bytes: bytes
    endpoint_root: bytes
    syntax_preimage: "ValidatedSaturationPreimageV1"
    direct_preimage: "ValidatedSaturationPreimageV1"


@dataclass(frozen=True, slots=True)
class ValidatedSaturationPreimageV1:
    path: str
    canonical_object: tuple[object, ...]
    canonical_bytes: bytes
    program_archive_root: bytes
    class_archive_root: bytes
    coverage_root: bytes
    state_root: bytes
    continuation_bank: tuple[object, ...]
    visible_classes: tuple[object, ...]


def _validate_saturation_preimage_v1(
    report: Mapping[str, object],
    implementation: str,
    path: str,
) -> ValidatedSaturationPreimageV1:
    field = f"{path}_saturation_state_preimage_cbor_hex"
    value = report[field]
    if (
        type(value) is not str
        or not value
        or len(value) % 2
        or re.fullmatch(r"[0-9a-f]+", value) is None
    ):
        _fail(FAIL_WIRE, f"{implementation} {field} is not canonical hex")
    encoded = bytes.fromhex(value)
    if len(encoded) > _contract.Q0_MAX_OUTPUT_BYTES:
        _fail(FAIL_WIRE, f"{implementation} {field} exceeds output guard")
    try:
        decoded = _cbor.canonical_cbor_decode(encoded)
        if _cbor.canonical_cbor_encode(decoded) != encoded:
            _fail(FAIL_WIRE, f"{implementation} {field} is noncanonical CBOR")
    except (TypeError, ValueError, RuntimeError) as error:
        _fail(FAIL_WIRE, f"{implementation} {field} strict decode failed: {error}")
    if type(decoded) is not tuple or len(decoded) != 5:
        _fail(FAIL_WIRE, f"{implementation} {field} is not the exact five-tuple")
    programs, bank, classes, coverage, metadata = decoded
    if any(type(item) is not tuple for item in (programs, bank, classes, coverage)):
        _fail(FAIL_WIRE, f"{implementation} {path} state archives are not arrays")

    oracle = importlib.import_module(f"{_HOST_PACKAGE}.phase3_q0_quotient_oracle_v1")
    strict_ast = importlib.import_module(f"{_HOST_PACKAGE}.strict_ast_shrink6_v1")
    program_keys: list[tuple[object, ...]] = []
    for index, record in enumerate(programs):
        if (
            type(record) is not tuple
            or len(record) != 7
            or record[:3] != (1, PROGRAM_RECORD_SCHEMA_ID, index)
            or type(record[3]) is not bytes
            or type(record[4]) is not bytes
            or len(record[4]) != 32
        ):
            _fail(FAIL_WIRE, f"{implementation} {path} program record differs")
        try:
            ast_value = strict_ast.decode_shrink6_canonical_ast(record[3])
        except (TypeError, ValueError, RuntimeError) as error:
            _fail(
                FAIL_WIRE,
                f"{implementation} {path} program AST replay failed: {error}",
            )
        sort_id = {
            "Bool": 1,
            "Bit": 2,
            "Sign": 3,
            "BoundedInt": 4,
            "RationalValue": 5,
        }.get(ast_value.metrics.output_sort)
        if (
            ast_value.cbor_bytes != record[3]
            or ast_value.digest != record[4]
            or record[5] != sort_id
            or record[6] != oracle.program_mdl_length_q32(ast_value)
        ):
            _fail(FAIL_WIRE, f"{implementation} {path} program identity differs")
        program_keys.append(
            (
                ast_value.metrics.depth,
                ast_value.metrics.node_count,
                sort_id,
                ast_value.root_operator_id,
                ast_value.cbor_bytes,
            )
        )
    if not programs or program_keys != sorted(program_keys):
        _fail(FAIL_WIRE, f"{implementation} {path} program archive order differs")

    class_ids: set[bytes] = set()
    frontier_count = 0
    maximum_frontier = 0
    for index, record in enumerate(classes):
        if (
            type(record) is not tuple
            or len(record) != 9
            or record[:4]
            != (
                1,
                _contract.Q0_QUOTIENT_CLASS_TAG,
                _contract.QUOTIENT_CLASS_SCHEMA_ID,
                index,
            )
            or type(record[5]) is not bytes
            or len(record[5]) != 32
            or type(record[6]) is not int
            or type(record[7]) is not tuple
            or record[6] != len(record[7])
        ):
            _fail(FAIL_WIRE, f"{implementation} {path} class record differs")
        if record[5] in class_ids:
            _fail(FAIL_WIRE, f"{implementation} {path} class ID is duplicated")
        class_ids.add(record[5])
        frontier_count += record[6]
        maximum_frontier = max(maximum_frontier, record[6])
    if len(classes) != report["behavior_class_count"]:
        _fail(FAIL_WIRE, f"{implementation} {path} class count differs")
    if (
        frontier_count != report["frontier_point_count"]
        or maximum_frontier != report["maximum_frontier_size"]
    ):
        _fail(FAIL_WIRE, f"{implementation} {path} frontier accounting differs")

    bank_count = 0
    per_class_bank: dict[bytes, int] = {}
    prior_bank_key: tuple[bytes, bytes, bytes] | None = None
    for row in bank:
        if (
            type(row) is not tuple
            or len(row) != 4
            or type(row[0]) is not bytes
            or len(row[0]) != 32
            or type(row[1]) is not bytes
            or type(row[2]) is not tuple
            or type(row[3]) is not tuple
            or not row[3]
            or row[0] not in class_ids
        ):
            _fail(FAIL_WIRE, f"{implementation} {path} bank row differs")
        key = (row[0], row[1], _cbor.canonical_cbor_encode(row[2]))
        if prior_bank_key is not None and key <= prior_bank_key:
            _fail(FAIL_WIRE, f"{implementation} {path} bank order differs")
        prior_bank_key = key
        for rank, entry in enumerate(row[3]):
            if (
                type(entry) is not tuple
                or len(entry) != 3
                or entry[0] != rank
                or type(entry[1]) is not bytes
                or type(entry[2]) is not bytes
                or len(entry[2]) != 32
            ):
                _fail(FAIL_WIRE, f"{implementation} {path} bank entry differs")
            try:
                ast_value = strict_ast.decode_shrink6_canonical_ast(entry[1])
            except (TypeError, ValueError, RuntimeError) as error:
                _fail(
                    FAIL_WIRE,
                    f"{implementation} {path} bank AST replay failed: {error}",
                )
            if ast_value.digest != entry[2]:
                _fail(FAIL_WIRE, f"{implementation} {path} bank AST hash differs")
        bank_count += len(row[3])
        per_class_bank[row[0]] = per_class_bank.get(row[0], 0) + len(row[3])
    expected_bank_count = report[
        f"{path if path == 'syntax' else 'quotient'}_continuation_bank_point_count"
    ]
    expected_maximum_bank = report[
        f"maximum_{path if path == 'syntax' else 'quotient'}_bank_points_per_class"
    ]
    if (
        bank_count != expected_bank_count
        or max(per_class_bank.values(), default=0) != expected_maximum_bank
    ):
        _fail(FAIL_WIRE, f"{implementation} {path} bank accounting differs")

    expected_coverage = _coverage_records(report, f"{path}_coverage")
    if coverage != expected_coverage:
        _fail(FAIL_WIRE, f"{implementation} {path} coverage preimage differs")
    expected_path_id = SYNTAX_PATH_ID if path == "syntax" else DIRECT_PATH_ID
    if (
        type(metadata) is not tuple
        or len(metadata) != 11
        or metadata[:3] != (1, FIXED_POINT_STATE_SCHEMA_ID, expected_path_id)
        or metadata[3] != report[f"{path}_saturation_rounds"]
        or metadata[4:7] != (True, True, True)
        or metadata[7:] != (0, 0, 0, 0)
    ):
        _fail(FAIL_WIRE, f"{implementation} {path} fixed-point metadata differs")

    program_root = _cbor.rfc6962_root(list(programs))
    class_root = _cbor.rfc6962_root(list(classes))
    coverage_root = _cbor.rfc6962_root(list(coverage))
    state_domain = (
        SYNTAX_STATE_ROOT_DOMAIN if path == "syntax" else DIRECT_STATE_ROOT_DOMAIN
    )
    state_root = _cbor.content_hash(state_domain, decoded)
    if class_root != _root_bytes(
        report[f"{path}_class_archive_root"], f"{path}_class_archive_root"
    ):
        _fail(FAIL_WIRE, f"{implementation} {path} class root replay differs")
    if coverage_root != _root_bytes(
        report[f"{path}_coverage_root"], f"{path}_coverage_root"
    ):
        _fail(FAIL_WIRE, f"{implementation} {path} coverage root replay differs")
    if state_root != _root_bytes(
        report[f"{path}_state_root"], f"{path}_state_root"
    ):
        _fail(FAIL_WIRE, f"{implementation} {path} state root replay differs")
    if path == "syntax":
        if len(programs) != report["canonical_syntax_count"]:
            _fail(FAIL_WIRE, f"{implementation} syntax program count differs")
        if program_root != _root_bytes(
            report["syntax_program_root"], "syntax_program_root"
        ):
            _fail(FAIL_WIRE, f"{implementation} syntax program root replay differs")
    return ValidatedSaturationPreimageV1(
        path,
        decoded,
        encoded,
        program_root,
        class_root,
        coverage_root,
        state_root,
        bank,
        classes,
    )


def validate_endpoint_v1(
    report_value: Mapping[str, object], implementation: str
) -> ValidatedEndpointV1:
    expected_fields = (
        PYTHON_ENDPOINT_FIELDS if implementation == "python" else RUST_ENDPOINT_FIELDS
    )
    report = _exact_dict(report_value, expected_fields, f"{implementation} endpoint")
    expected_schema = PYTHON_SCHEMA if implementation == "python" else RUST_SCHEMA
    expected_id = (
        PYTHON_IMPLEMENTATION_ID if implementation == "python" else RUST_IMPLEMENTATION_ID
    )
    if report["schema_version"] != expected_schema or report["implementation_id"] != expected_id:
        _fail(FAIL_WIRE, f"{implementation} schema/implementation identity differs")
    constants = {
        "terminal_status": _contract.Q0_ENDPOINT_PASS_STATUS,
        "dsl_version": _contract.DSL_VERSION,
        "dsl_freeze_version": _contract.DSL_FREEZE_VERSION,
        "closure_semantics_version": _contract.CLOSURE_SEMANTICS_VERSION,
        "q0_freeze_version": _contract.Q0_FREEZE_VERSION,
        "projection_id": _contract.Q0_PROJECTION_ID,
        "probe_input_signature_id": _contract.Q0_PROBE_INPUT_SIGNATURE_ID,
        "probe_canonical_cbor_hex": _contract.Q0ProbeInputV1().canonical_bytes.hex(),
        "probe_universe_root": "sha256:" + _contract.Q0ProbeInputV1().universe_root.hex(),
        "frozen_leaf_count": len(_contract.Q0_FROZEN_LEAF_CANONICAL_NODES),
    }
    for field, expected in constants.items():
        if report[field] != expected:
            _fail(FAIL_WIRE, f"{implementation} {field} differs")
    if report["syntax_saturation_rounds"] != report["direct_saturation_rounds"]:
        _fail(FAIL_WIRE, f"{implementation} saturation round counts differ")
    required_true = (
        "work_queue_empty",
        "zero_delta_full_round",
        "all_typed_operator_frontier_tuples_covered",
        "exhaustive_syntax_oracle_complete",
        "syntax_direct_states_equal",
        "resource_guards_ok",
    )
    required_false = (
        "target_truth_accessed",
        "split_accessed",
        "role_evaluation_performed",
        "formal_roots_generated",
        "authority_claimed",
    )
    if any(report[name] is not True for name in required_true) or any(
        report[name] is not False for name in required_false
    ):
        _fail(FAIL_WIRE, f"{implementation} PASS/nonauthority flags differ")
    if any(
        report[name] != 0
        for name in (
            "final_class_delta",
            "final_frontier_delta",
            "final_bank_delta",
        )
    ):
        _fail(FAIL_WIRE, f"{implementation} terminal delta is nonzero")
    if report["syntax_class_archive_root"] != report["direct_class_archive_root"]:
        _fail(FAIL_WIRE, f"{implementation} syntax/direct class roots differ")
    syntax_coverage = _coverage_records(report, "syntax_coverage")
    direct_coverage = _coverage_records(report, "direct_coverage")
    bounded = {
        "syntax_raw_operator_applications": _contract.Q0_MAX_RAW_APPLICATIONS,
        "quotient_raw_operator_applications": _contract.Q0_MAX_RAW_APPLICATIONS,
        "canonical_syntax_count": _contract.Q0_MAX_CANONICAL_SYNTAX,
        "behavior_class_count": _contract.Q0_MAX_BEHAVIOR_CLASSES,
        "frontier_point_count": _contract.Q0_MAX_FRONTIER_POINTS,
        "maximum_frontier_size": _contract.Q0_MAX_FRONTIER_POINTS_PER_CLASS,
        "syntax_continuation_bank_point_count": _contract.Q0_MAX_CONTINUATION_BANK_POINTS,
        "quotient_continuation_bank_point_count": _contract.Q0_MAX_CONTINUATION_BANK_POINTS,
        "maximum_syntax_bank_points_per_class": _contract.Q0_MAX_CONTINUATION_BANK_POINTS_PER_CLASS,
        "maximum_quotient_bank_points_per_class": (
            _contract.Q0_MAX_CONTINUATION_BANK_POINTS_PER_CLASS
        ),
        "direct_saturation_rounds": _contract.Q0_MAX_SATURATION_ROUNDS,
    }
    for field, maximum in bounded.items():
        if not 1 <= _uint(report[field], field, maximum) <= maximum:
            _fail(FAIL_WIRE, f"{implementation} {field} is outside PASS bounds")
    expected_syntax_sums = {
        "syntax_raw_operator_applications": 1,
        "syntax_strict_admitted_applications": 2,
        "syntax_rewrite_collapses": 3,
        "canonical_syntax_count": 5,
    }
    expected_direct_sums = {
        "quotient_raw_operator_applications": 1,
        "quotient_strict_admitted_applications": 2,
        "quotient_rewrite_collapses": 3,
    }
    if any(
        report[field] != sum(row[index] for row in syntax_coverage)
        for field, index in expected_syntax_sums.items()
    ) or any(
        report[field] != sum(row[index] for row in direct_coverage)
        for field, index in expected_direct_sums.items()
    ):
        _fail(FAIL_WIRE, f"{implementation} coverage/count accounting differs")
    if _cbor.rfc6962_root(list(syntax_coverage)) != _root_bytes(
        report["syntax_coverage_root"], "syntax_coverage_root"
    ):
        _fail(FAIL_WIRE, f"{implementation} syntax coverage root replay differs")
    if _cbor.rfc6962_root(list(direct_coverage)) != _root_bytes(
        report["direct_coverage_root"], "direct_coverage_root"
    ):
        _fail(FAIL_WIRE, f"{implementation} direct coverage root replay differs")
    syntax_preimage = _validate_saturation_preimage_v1(
        report, implementation, "syntax"
    )
    direct_preimage = _validate_saturation_preimage_v1(
        report, implementation, "direct"
    )
    if (
        syntax_preimage.continuation_bank != direct_preimage.continuation_bank
        or syntax_preimage.visible_classes != direct_preimage.visible_classes
    ):
        _fail(FAIL_WIRE, f"{implementation} syntax/direct bank or classes differ")
    _validate_rounds(report)
    state = endpoint_state_object_v1(report)
    if len(state) != 43:
        _fail(FAIL_WIRE, f"{implementation} endpoint state is not 43 fields")
    reconstructed = _cbor.canonical_cbor_encode(state)
    if implementation == "python":
        value = report["endpoint_state_cbor_hex"]
        if (
            type(value) is not str
            or not value
            or len(value) % 2
            or re.fullmatch(r"[0-9a-f]+", value) is None
        ):
            _fail(FAIL_WIRE, "Python endpoint CBOR hex is not canonical")
        encoded = bytes.fromhex(value)
        if encoded != reconstructed:
            _fail(FAIL_WIRE, "Python endpoint CBOR differs from host reconstruction")
    else:
        encoded = reconstructed
    decoded = _cbor.canonical_cbor_decode(encoded)
    if len(decoded) != 43 or _cbor.canonical_cbor_encode(decoded) != encoded:
        _fail(FAIL_WIRE, f"{implementation} strict endpoint CBOR replay differs")
    endpoint_root = _cbor.content_hash(_contract.ENDPOINT_STATE_ROOT_DOMAIN, decoded)
    if endpoint_root != _root_bytes(report["endpoint_state_root"], "endpoint_state_root"):
        _fail(FAIL_WIRE, f"{implementation} endpoint state root replay differs")
    source_field = "python_source_root" if implementation == "python" else "rust_source_root"
    _root_bytes(report[source_field], source_field)
    return ValidatedEndpointV1(
        implementation,
        report,
        state,
        encoded,
        endpoint_root,
        syntax_preimage,
        direct_preimage,
    )


def compare_endpoints_v1(
    python: ValidatedEndpointV1, rust: ValidatedEndpointV1
) -> None:
    python_common = {
        key: value
        for key, value in python.report.items()
        if key not in IMPLEMENTATION_SPECIFIC_FIELDS
    }
    rust_common = {
        key: value
        for key, value in rust.report.items()
        if key not in IMPLEMENTATION_SPECIFIC_FIELDS
    }
    if python_common != rust_common:
        differing = sorted(
            key
            for key in set(python_common) | set(rust_common)
            if python_common.get(key) != rust_common.get(key)
        )
        _fail(FAIL_DISAGREEMENT, f"shared endpoint fields differ: {differing}")
    if python.canonical_bytes != rust.canonical_bytes or python.endpoint_root != rust.endpoint_root:
        _fail(FAIL_DISAGREEMENT, "43-field endpoint CBOR/root differs")
    if (
        python.syntax_preimage.canonical_bytes
        != rust.syntax_preimage.canonical_bytes
        or python.direct_preimage.canonical_bytes
        != rust.direct_preimage.canonical_bytes
    ):
        _fail(FAIL_DISAGREEMENT, "saturation state preimage bytes differ")


@dataclass(frozen=True, slots=True)
class HostReplayV1:
    canonical_state: tuple[object, ...]
    endpoint_root: bytes
    class_archive_root: bytes
    syntax_preimage_bytes: bytes
    direct_preimage_bytes: bytes
    syntax_state_root: bytes
    direct_state_root: bytes
    loaded_modules: tuple[str, ...]


def host_local_replay_v1() -> HostReplayV1:
    oracle = importlib.import_module(f"{_HOST_PACKAGE}.phase3_q0_quotient_oracle_v1")
    result = oracle.run_q0_python_oracle_v1()
    loaded = tuple(
        sorted(
            name
            for name in sys.modules
            if name.startswith(f"{_HOST_PACKAGE}.")
        )
    )
    if any(any(token in name.lower() for token in FORBIDDEN_SOURCE_TOKENS) for name in loaded):
        _fail(FAIL_HOST_REPLAY, "host replay imported target/truth/split module")
    state = result.canonical_state_object()
    root = _cbor.content_hash(_contract.ENDPOINT_STATE_ROOT_DOMAIN, state)
    if root != result.endpoint_state_root:
        _fail(FAIL_HOST_REPLAY, "host endpoint state root did not replay")
    if result.syntax_class_archive_root != result.direct_class_archive_root:
        _fail(FAIL_HOST_REPLAY, "host syntax/direct class roots differ")
    syntax_preimage = result.syntax_saturation_state_preimage_bytes
    direct_preimage = result.direct_saturation_state_preimage_bytes
    try:
        syntax_object = _cbor.canonical_cbor_decode(syntax_preimage)
        direct_object = _cbor.canonical_cbor_decode(direct_preimage)
    except (TypeError, ValueError, RuntimeError) as error:
        _fail(FAIL_HOST_REPLAY, f"host state-preimage decode failed: {error}")
    if (
        type(syntax_object) is not tuple
        or type(direct_object) is not tuple
        or len(syntax_object) != 5
        or len(direct_object) != 5
        or _cbor.canonical_cbor_encode(syntax_object) != syntax_preimage
        or _cbor.canonical_cbor_encode(direct_object) != direct_preimage
    ):
        _fail(FAIL_HOST_REPLAY, "host state preimage is not strict five-tuple CBOR")
    syntax_state_root = _cbor.content_hash(
        SYNTAX_STATE_ROOT_DOMAIN, syntax_object
    )
    direct_state_root = _cbor.content_hash(DIRECT_STATE_ROOT_DOMAIN, direct_object)
    if (
        syntax_state_root != result.syntax_state_root
        or direct_state_root != result.direct_state_root
        or syntax_object[1] != direct_object[1]
        or syntax_object[2] != direct_object[2]
    ):
        _fail(FAIL_HOST_REPLAY, "host saturation preimage/root replay differs")
    return HostReplayV1(
        state,
        root,
        result.syntax_class_archive_root,
        syntax_preimage,
        direct_preimage,
        syntax_state_root,
        direct_state_root,
        loaded,
    )


def compare_host_replay_v1(
    host: HostReplayV1,
    python: ValidatedEndpointV1,
    rust: ValidatedEndpointV1,
) -> None:
    encoded = _cbor.canonical_cbor_encode(host.canonical_state)
    if encoded != python.canonical_bytes or encoded != rust.canonical_bytes:
        _fail(FAIL_HOST_REPLAY, "host 43-field replay differs from endpoint wires")
    if host.endpoint_root != python.endpoint_root or host.endpoint_root != rust.endpoint_root:
        _fail(FAIL_HOST_REPLAY, "host endpoint root differs")
    expected_class = _root_bytes(
        python.report["syntax_class_archive_root"], "syntax_class_archive_root"
    )
    if host.class_archive_root != expected_class:
        _fail(FAIL_HOST_REPLAY, "host class archive root differs")
    if (
        host.syntax_preimage_bytes != python.syntax_preimage.canonical_bytes
        or host.syntax_preimage_bytes != rust.syntax_preimage.canonical_bytes
        or host.direct_preimage_bytes != python.direct_preimage.canonical_bytes
        or host.direct_preimage_bytes != rust.direct_preimage.canonical_bytes
        or host.syntax_state_root != python.syntax_preimage.state_root
        or host.syntax_state_root != rust.syntax_preimage.state_root
        or host.direct_state_root != python.direct_preimage.state_root
        or host.direct_state_root != rust.direct_preimage.state_root
    ):
        _fail(FAIL_HOST_REPLAY, "host saturation preimages differ from endpoints")


_GATE_ISSUER_TOKEN: Final = object()
_PRE_DUAL_GATE_EVIDENCE_ROOT_DOMAIN: Final = (
    b"HEGEL/Q0/PRE_DUAL_GATE_MODULE_EVIDENCE/V1"
)
_ISOLATION_EVIDENCE_ROOT_DOMAIN: Final = b"HEGEL/Q0/ISOLATION_EVIDENCE/V1"
_GATE_EVIDENCE_ROOT_DOMAIN: Final = b"HEGEL/Q0/HOST_GATE_EVIDENCE/V1"
_PRE_RECEIPT_ROOT_DOMAIN: Final = b"HEGEL/Q0/PRE_RECEIPT_GATE_EVIDENCE/V1"


@dataclass(frozen=True, slots=True)
class ValidatedPreDualGateEvidenceV1:
    canonical_payload: bytes
    evidence_root: bytes
    _issuer_token: object

    def __post_init__(self) -> None:
        if self._issuer_token is not _GATE_ISSUER_TOKEN:
            _fail(FAIL_HOST_REPLAY, "pre-dual token was not issued by live replay")
        expected = sha256(
            _PRE_DUAL_GATE_EVIDENCE_ROOT_DOMAIN + b"\x00" + self.canonical_payload
        ).digest()
        if self.evidence_root != expected:
            _fail(FAIL_HOST_REPLAY, "pre-dual gate evidence root differs")

    @property
    def payload(self) -> dict[str, object]:
        value = json.loads(self.canonical_payload)
        if type(value) is not dict:
            _fail(FAIL_HOST_REPLAY, "pre-dual gate evidence is malformed")
        return value


@dataclass(frozen=True, slots=True)
class ValidatedIsolationEvidenceV1:
    canonical_payload: bytes
    evidence_root: bytes
    _issuer_token: object

    def __post_init__(self) -> None:
        if self._issuer_token is not _GATE_ISSUER_TOKEN:
            _fail(FAIL_DOCKER, "isolation token was not issued by live replay")
        expected = sha256(
            _ISOLATION_EVIDENCE_ROOT_DOMAIN + b"\x00" + self.canonical_payload
        ).digest()
        if self.evidence_root != expected:
            _fail(FAIL_DOCKER, "isolation evidence root differs")
        value = json.loads(self.canonical_payload)
        expected_fields = {
            "schema_version",
            "source_commit",
            "python_implementation_root",
            "rust_implementation_root",
            "python_image_id",
            "rust_image_id",
            "python_command",
            "rust_command",
            "python_snapshot",
            "rust_snapshot",
            "cargo_snapshot",
            "isolation_prerequisites",
        }
        if type(value) is not dict or set(value) != expected_fields:
            _fail(FAIL_DOCKER, "isolation evidence fields differ")
        if (
            value["schema_version"]
            != "hegel-phase3a-q0-isolation-evidence/1"
            or type(value["source_commit"]) is not str
            or re.fullmatch(r"[0-9a-f]{40}", value["source_commit"]) is None
        ):
            _fail(FAIL_DOCKER, "isolation source identity differs")
        _root_bytes(
            value["python_implementation_root"],
            "isolation Python implementation root",
        )
        _root_bytes(
            value["rust_implementation_root"],
            "isolation Rust implementation root",
        )
        for name in ("python_command", "rust_command"):
            if type(value[name]) is not list or any(
                type(item) is not str for item in value[name]
            ):
                _fail(FAIL_DOCKER, f"isolation {name} differs")
        rows = value["isolation_prerequisites"]
        if type(rows) is not list or len(rows) != len(ISOLATION_PREREQUISITES):
            _fail(FAIL_DOCKER, "isolation prerequisite registry differs")
        for expected_row, row_value in zip(
            ISOLATION_PREREQUISITES, rows, strict=True
        ):
            row = _exact_dict(
                row_value,
                {"prerequisite_id", "name", "passed", "predicates"},
                "isolation prerequisite",
            )
            if (row["prerequisite_id"], row["name"]) != expected_row:
                _fail(FAIL_DOCKER, "isolation prerequisite identity differs")
            predicates = row["predicates"]
            if (
                type(predicates) is not dict
                or not predicates
                or any(type(item) is not bool for item in predicates.values())
                or row["passed"] is not True
                or not all(predicates.values())
            ):
                _fail(FAIL_DOCKER, "isolation prerequisite did not pass")

    @property
    def payload(self) -> dict[str, object]:
        value = json.loads(self.canonical_payload)
        if type(value) is not dict:
            _fail(FAIL_DOCKER, "isolation evidence payload is malformed")
        return value

    @property
    def prerequisites(self) -> list[dict[str, object]]:
        value = self.payload.get("isolation_prerequisites")
        if type(value) is not list:
            _fail(FAIL_DOCKER, "isolation prerequisite rows are absent")
        return value


@dataclass(frozen=True, slots=True)
class PreReceiptQualificationV1:
    canonical_payload: bytes
    evidence_root: bytes
    _issuer_token: object

    def __post_init__(self) -> None:
        if self._issuer_token is not _GATE_ISSUER_TOKEN:
            _fail(FAIL_HOST_REPLAY, "pre-receipt token was not issued by live finalizer")
        expected = sha256(
            _PRE_RECEIPT_ROOT_DOMAIN + b"\x00" + self.canonical_payload
        ).digest()
        if self.evidence_root != expected:
            _fail(FAIL_HOST_REPLAY, "pre-receipt gate evidence root differs")

    @property
    def gates(self) -> list[dict[str, object]]:
        value = json.loads(self.canonical_payload)
        if type(value) is not dict or type(value.get("gates")) is not list:
            _fail(FAIL_HOST_REPLAY, "pre-receipt gate payload is malformed")
        return value["gates"]


@dataclass(frozen=True, slots=True)
class QualifiedGateSetV1:
    canonical_payload: bytes
    evidence_root: bytes
    _issuer_token: object

    def __post_init__(self) -> None:
        if self._issuer_token is not _GATE_ISSUER_TOKEN:
            _fail(FAIL_HOST_REPLAY, "qualified gate set was not issued by live finalizer")
        expected = sha256(
            _GATE_EVIDENCE_ROOT_DOMAIN + b"\x00" + self.canonical_payload
        ).digest()
        if self.evidence_root != expected:
            _fail(FAIL_HOST_REPLAY, "qualified gate evidence root differs")

    @property
    def gates(self) -> list[dict[str, object]]:
        value = json.loads(self.canonical_payload)
        if type(value) is not dict or type(value.get("gates")) is not list:
            _fail(FAIL_HOST_REPLAY, "qualified gate payload is malformed")
        return value["gates"]


def _validate_gate_rows_v1(
    value: object,
    *,
    allow_pending_dual: bool,
) -> list[dict[str, object]]:
    if type(value) is not dict or set(value) != {"schema_version", "gates"}:
        _fail(FAIL_HOST_REPLAY, "gate qualification envelope fields differ")
    rows = value["gates"]
    if type(rows) is not list or len(rows) != _contract.Q0_READINESS_GATE_TOTAL:
        _fail(FAIL_HOST_REPLAY, "gate qualification does not contain 14 rows")
    for index, row_value in enumerate(rows, start=1):
        row = _exact_dict(
            row_value,
            {"gate_id", "name", "passed", "predicates", "evidence", "pending_dual"},
            f"gate[{index}]",
        )
        if row["gate_id"] != index or row["name"] != _contract.Q0_READINESS_GATES[index - 1]:
            _fail(FAIL_HOST_REPLAY, f"gate {index} identity differs")
        predicates = row["predicates"]
        if type(predicates) is not dict or not predicates or any(
            type(name) is not str or type(result) is not bool
            for name, result in predicates.items()
        ):
            _fail(FAIL_HOST_REPLAY, f"gate {index} predicates are not exact booleans")
        if type(row["evidence"]) is not dict or type(row["pending_dual"]) is not bool:
            _fail(FAIL_HOST_REPLAY, f"gate {index} evidence/pending type differs")
        expected_pass = all(predicates.values()) and not row["pending_dual"]
        if type(row["passed"]) is not bool or row["passed"] is not expected_pass:
            _fail(FAIL_HOST_REPLAY, f"gate {index} PASS does not equal all predicates")
        if row["pending_dual"] and (not allow_pending_dual or index not in {11, 13, 14}):
            _fail(FAIL_HOST_REPLAY, f"gate {index} has unauthorized pending state")
    return rows


def qualify_pre_dual_gate_evidence_v1(
    project_root: Path,
) -> ValidatedPreDualGateEvidenceV1:
    gate_module = importlib.import_module(
        f"{_HOST_PACKAGE}.phase3_q0_gate_qualification_v1"
    )
    payload = gate_module.qualify_q0_pre_dual_gates_v1(
        project_root=project_root
    )
    gate_module.validate_pre_dual_gate_evidence_v1(payload)
    canonical = gate_module.canonical_gate_json_bytes_v1(payload)
    root = sha256(
        _PRE_DUAL_GATE_EVIDENCE_ROOT_DOMAIN + b"\x00" + canonical
    ).digest()
    return ValidatedPreDualGateEvidenceV1(canonical, root, _GATE_ISSUER_TOKEN)


def isolation_prerequisite_evidence_v1(
    project_root: Path,
    source_commit: str,
    python_command: Sequence[str],
    rust_command: Sequence[str],
    python_image_id: str,
    rust_image_id: str,
    python_snapshot: Path,
    rust_snapshot: Path,
    cargo_snapshot: Path,
    python_manifest: SourceManifestV1,
    rust_manifest: SourceManifestV1,
) -> ValidatedIsolationEvidenceV1:
    commit = verify_source_commit_v1(project_root, source_commit)
    verify_manifest_against_commit_v1(project_root, python_manifest, commit)
    verify_manifest_against_commit_v1(project_root, rust_manifest, commit)
    python_snapshot_rows = _sealed_snapshot_file_rows_v1(
        python_snapshot, "Python source"
    )
    rust_snapshot_rows = _sealed_snapshot_file_rows_v1(
        rust_snapshot, "Rust source"
    )
    cargo_snapshot_rows = _cargo_home_file_rows(cargo_snapshot)
    common_flags = {
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--memory=512m",
        "--memory-swap=512m",
        "--pids-limit=64",
        "--ulimit=nofile=128:128",
    }
    commands = (tuple(python_command), tuple(rust_command))
    predicates = (
        {
            "python_image_id_exact": python_image_id == PYTHON_IMAGE.split("@", 1)[1],
            "rust_image_id_exact": rust_image_id == RUST_IMAGE.split("@", 1)[1],
        },
        {
            "both_pull_never": all("--pull=never" in command for command in commands),
            "both_network_none": all("--network=none" in command for command in commands),
        },
        {
            "snapshots_distinct": len(
                {python_snapshot.resolve(), rust_snapshot.resolve(), cargo_snapshot.resolve()}
            )
            == 3,
            "python_source_ro": f"{python_snapshot.resolve()}:/workspace:ro"
            in python_command,
            "rust_source_ro": f"{rust_snapshot.resolve()}:/workspace:ro" in rust_command,
            "cargo_home_ro": f"{cargo_snapshot.resolve()}:/cargo-home:ro" in rust_command,
        },
        {
            "common_hardening_flags": all(
                common_flags.issubset(set(command)) for command in commands
            ),
        },
        {
            "python_resource_profile": "--memory=512m" in python_command
            and "--pids-limit=64" in python_command,
            "rust_resource_profile": "--memory=512m" in rust_command
            and "--pids-limit=64" in rust_command
            and "CARGO_BUILD_JOBS=1" in rust_command,
        },
        {
            "current_clean_head_is_requested_commit": commit == source_commit,
            "python_manifest_matches_commit_tree": python_snapshot_rows
            == python_manifest.files,
            "rust_manifest_matches_commit_tree": rust_snapshot_rows
            == rust_manifest.files,
        },
        {
            "sealed_cargo_home_nonempty": bool(rust_manifest.dependency_files),
            "sealed_cargo_home_manifest_exact": cargo_snapshot_rows
            == rust_manifest.dependency_files,
            "sealed_cargo_home_is_rust_identity": rust_manifest.target_triple
            == "x86_64-unknown-linux-gnu",
            "cargo_target_exact": rust_command[-2:]
            == ["--target", "x86_64-unknown-linux-gnu"],
        },
    )
    rows: list[dict[str, object]] = []
    for (prerequisite_id, name), checks in zip(
        ISOLATION_PREREQUISITES, predicates, strict=True
    ):
        rows.append(
            {
                "prerequisite_id": prerequisite_id,
                "name": name,
                "passed": all(checks.values()),
                "predicates": checks,
            }
        )
    if not all(row["passed"] is True for row in rows):
        _fail(FAIL_DOCKER, "one or more isolation prerequisites failed")
    payload = {
        "schema_version": "hegel-phase3a-q0-isolation-evidence/1",
        "source_commit": commit,
        "python_implementation_root": "sha256:" + python_manifest.root.hex(),
        "rust_implementation_root": "sha256:" + rust_manifest.root.hex(),
        "python_image_id": python_image_id,
        "rust_image_id": rust_image_id,
        "python_command": list(python_command),
        "rust_command": list(rust_command),
        "python_snapshot": str(python_snapshot.resolve()),
        "rust_snapshot": str(rust_snapshot.resolve()),
        "cargo_snapshot": str(cargo_snapshot.resolve()),
        "isolation_prerequisites": rows,
    }
    canonical = _canonical_json_bytes(payload).rstrip(b"\n")
    root = sha256(
        _ISOLATION_EVIDENCE_ROOT_DOMAIN + b"\x00" + canonical
    ).digest()
    return ValidatedIsolationEvidenceV1(canonical, root, _GATE_ISSUER_TOKEN)


def finalize_gate_evidence_v1(
    pre_dual: ValidatedPreDualGateEvidenceV1,
    python: ValidatedEndpointV1,
    rust: ValidatedEndpointV1,
    host: HostReplayV1,
    python_manifest: SourceManifestV1,
    rust_manifest: SourceManifestV1,
    host_manifest: SourceManifestV1,
    isolation_evidence: ValidatedIsolationEvidenceV1,
    downstream_state: Mapping[str, object],
) -> PreReceiptQualificationV1:
    if not isinstance(pre_dual, ValidatedPreDualGateEvidenceV1):
        _fail(FAIL_HOST_REPLAY, "host finalizer requires live pre-dual evidence")
    pre_dual_payload = pre_dual.payload
    pre_envelope = {
        "schema_version": pre_dual_payload.get("schema_version"),
        "gates": pre_dual_payload.get("gates"),
    }
    rows = json.loads(
        json.dumps(
            _validate_gate_rows_v1(pre_envelope, allow_pending_dual=True)
        )
    )
    for index in tuple(range(1, 11)) + (12,):
        if rows[index - 1]["passed"] is not True:
            _fail(FAIL_HOST_REPLAY, f"pre-dual readiness gate {index} did not pass")
    source_binding = pre_dual_payload.get("source_binding")
    if type(source_binding) is not dict:
        _fail(FAIL_HOST_REPLAY, "pre-dual source binding is absent")
    pre_dual_source_root = source_binding.get("manifest_root")
    if (
        type(pre_dual_source_root) is not str
        or re.fullmatch(r"[0-9a-f]{64}", pre_dual_source_root) is None
    ):
        _fail(FAIL_HOST_REPLAY, "pre-dual source manifest root is malformed")
    compare_endpoints_v1(python, rust)
    compare_host_replay_v1(host, python, rust)
    report = python.report

    gate11_predicates = {
        "dual_43_field_endpoint_bytes_equal": python.canonical_bytes == rust.canonical_bytes,
        "dual_endpoint_roots_equal": python.endpoint_root == rust.endpoint_root,
        "syntax_direct_class_roots_equal": report["syntax_class_archive_root"]
        == report["direct_class_archive_root"],
        "dual_coverage_roots_equal": python.report["syntax_coverage_root"]
        == rust.report["syntax_coverage_root"]
        and python.report["direct_coverage_root"] == rust.report["direct_coverage_root"],
        "dual_state_roots_equal": python.report["syntax_state_root"]
        == rust.report["syntax_state_root"]
        and python.report["direct_state_root"] == rust.report["direct_state_root"],
        "dual_complete_state_preimages_byte_equal": (
            python.syntax_preimage.canonical_bytes
            == rust.syntax_preimage.canonical_bytes
            and python.direct_preimage.canonical_bytes
            == rust.direct_preimage.canonical_bytes
        ),
        "host_complete_state_preimages_byte_equal": (
            host.syntax_preimage_bytes == python.syntax_preimage.canonical_bytes
            and host.syntax_preimage_bytes == rust.syntax_preimage.canonical_bytes
            and host.direct_preimage_bytes == python.direct_preimage.canonical_bytes
            and host.direct_preimage_bytes == rust.direct_preimage.canonical_bytes
        ),
        "host_class_archive_replay_equal": host.class_archive_root
        == _root_bytes(report["syntax_class_archive_root"], "syntax_class_root"),
        "fixed_point_zero_delta_and_queue_empty": report["zero_delta_full_round"] is True
        and report["work_queue_empty"] is True
        and report["final_class_delta"] == 0
        and report["final_frontier_delta"] == 0
        and report["final_bank_delta"] == 0,
    }
    rows[10] = {
        "gate_id": 11,
        "name": _contract.Q0_READINESS_GATES[10],
        "passed": all(gate11_predicates.values()),
        "predicates": gate11_predicates,
        "evidence": {
            "syntax_program_root": report["syntax_program_root"],
            "class_archive_root": report["syntax_class_archive_root"],
            "syntax_state_root": report["syntax_state_root"],
            "direct_state_root": report["direct_state_root"],
            "endpoint_state_root": report["endpoint_state_root"],
            "syntax_state_preimage_byte_length": len(host.syntax_preimage_bytes),
            "direct_state_preimage_byte_length": len(host.direct_preimage_bytes),
            "syntax_state_preimage_sha256": sha256(
                host.syntax_preimage_bytes
            ).hexdigest(),
            "direct_state_preimage_sha256": sha256(
                host.direct_preimage_bytes
            ).hexdigest(),
            "pre_dual_source_manifest_root": pre_dual_source_root,
        },
        "pending_dual": False,
    }

    if not isinstance(isolation_evidence, ValidatedIsolationEvidenceV1):
        _fail(FAIL_HOST_REPLAY, "Gate 13 requires live isolation evidence")
    isolation_payload = isolation_evidence.payload
    source_commit = isolation_payload.get("source_commit")
    if (
        isolation_payload.get("python_implementation_root")
        != "sha256:" + python_manifest.root.hex()
        or isolation_payload.get("rust_implementation_root")
        != "sha256:" + rust_manifest.root.hex()
    ):
        _fail(FAIL_HOST_REPLAY, "isolation implementation roots are stale")
    prerequisite_rows = isolation_evidence.prerequisites
    gate13_predicates = {
        "all_isolation_prerequisites_pass": bool(prerequisite_rows)
        and all(row.get("passed") is True for row in prerequisite_rows),
        "python_source_manifest_target_blind": all(
            not any(token in row.path.lower() for token in FORBIDDEN_SOURCE_TOKENS)
            for row in python_manifest.files
        ),
        "rust_source_manifest_target_blind": all(
            not any(token in row.path.lower() for token in FORBIDDEN_SOURCE_TOKENS)
            for row in rust_manifest.files
        ),
        "both_endpoint_access_flags_false": all(
            endpoint.report["target_truth_accessed"] is False
            and endpoint.report["split_accessed"] is False
            for endpoint in (python, rust)
        ),
        "host_loaded_modules_target_blind": all(
            not any(token in name.lower() for token in FORBIDDEN_SOURCE_TOKENS)
            for name in host.loaded_modules
        ),
        "host_replay_manifest_target_blind": all(
            not any(token in row.path.lower() for token in FORBIDDEN_SOURCE_TOKENS)
            for row in host_manifest.files
        ),
        "host_replay_manifest_covers_all_loaded_modules": all(
            f"src/hegel_machine/{name.rsplit('.', 1)[-1]}.py"
            in {row.path for row in host_manifest.files}
            for name in host.loaded_modules
        ),
        "host_role_is_trusted_issuer_not_third_isolated_endpoint": (
            host_manifest.implementation == "host"
            and host_manifest.domain == HOST_MANIFEST_DOMAIN
            and host_manifest.runtime_identity == HOST_RUNTIME_IDENTITY
        ),
        "implementation_roots_distinct_nonzero": len(
            {python_manifest.root, rust_manifest.root, host_manifest.root}
        )
        == 3
        and all(
            root != b"\x00" * 32
            for root in (
                python_manifest.root,
                rust_manifest.root,
                host_manifest.root,
            )
        ),
        "source_commit_full_identity": re.fullmatch(r"[0-9a-f]{40}", source_commit)
        is not None,
    }
    rows[12] = {
        "gate_id": 13,
        "name": _contract.Q0_READINESS_GATES[12],
        "passed": all(gate13_predicates.values()),
        "predicates": gate13_predicates,
        "evidence": {
            "source_commit": source_commit,
            "python_implementation_root": "sha256:" + python_manifest.root.hex(),
            "rust_implementation_root": "sha256:" + rust_manifest.root.hex(),
            "host_replay_implementation_root": (
                "sha256:" + host_manifest.root.hex()
            ),
            "host_role": {
                "trusted_issuer": True,
                "third_independent_endpoint": False,
                "filesystem_hard_isolation": False,
                "target_blind_import_manifest_required": True,
            },
            "isolation_prerequisites": prerequisite_rows,
            "isolation_evidence_root": (
                "sha256:" + isolation_evidence.evidence_root.hex()
            ),
            "pre_dual_source_manifest_root": pre_dual_source_root,
        },
        "pending_dual": False,
    }

    expected_downstream = {
        "q1_status_id": 0,
        "q1_output_root": None,
        "q2_status_id": 0,
        "role_evaluation_performed": False,
        "m3_formal_roots": None,
        "outside_certificate_issued": False,
    }
    gate14_predicates = {
        "host_dual_endpoint_state_agreement": host.endpoint_root
        == python.endpoint_root
        == rust.endpoint_root,
        "single_endpoints_do_not_claim_dual_authority": all(
            endpoint.report["terminal_status"] == _contract.Q0_ENDPOINT_PASS_STATUS
            and endpoint.report["authority_claimed"] is False
            for endpoint in (python, rust)
        ),
        "downstream_state_exact_not_run_null": dict(downstream_state)
        == expected_downstream,
        "no_role_formal_or_certificate_outputs": all(
            endpoint.report["role_evaluation_performed"] is False
            and endpoint.report["formal_roots_generated"] is False
            for endpoint in (python, rust)
        ),
    }
    rows[13] = {
        "gate_id": 14,
        "name": _contract.Q0_READINESS_GATES[13],
        "passed": False,
        "predicates": gate14_predicates,
        "evidence": {
            "endpoint_state_root": "sha256:" + host.endpoint_root.hex(),
            "downstream_state": expected_downstream,
            "pre_dual_source_manifest_root": pre_dual_source_root,
        },
        "pending_dual": True,
    }
    envelope = {
        "schema_version": "hegel-phase3a-q0-pre-receipt-qualification/1",
        "gates": rows,
    }
    pre_rows = _validate_gate_rows_v1(envelope, allow_pending_dual=True)
    if not all(row["passed"] is True for row in pre_rows[:13]):
        _fail(FAIL_HOST_REPLAY, "not all first 13 Q0 readiness gates passed")
    if pre_rows[13]["pending_dual"] is not True:
        _fail(FAIL_HOST_REPLAY, "Gate 14 is not pending candidate-receipt replay")
    payload = _canonical_json_bytes(envelope).rstrip(b"\n")
    root = sha256(_PRE_RECEIPT_ROOT_DOMAIN + b"\x00" + payload).digest()
    return PreReceiptQualificationV1(payload, root, _GATE_ISSUER_TOKEN)


def build_saturation_receipt_v1(
    python: ValidatedEndpointV1,
    rust: ValidatedEndpointV1,
    host: HostReplayV1,
    python_manifest: SourceManifestV1,
    rust_manifest: SourceManifestV1,
    pre_receipt: PreReceiptQualificationV1 | None,
) -> object:
    if not isinstance(pre_receipt, PreReceiptQualificationV1):
        _fail(FAIL_HOST_REPLAY, "candidate receipt requires live pre-receipt token")
    gate_rows = pre_receipt.gates
    if (
        len(gate_rows) != 14
        or not all(row.get("passed") is True for row in gate_rows[:13])
        or gate_rows[13].get("pending_dual") is not True
    ):
        _fail(FAIL_HOST_REPLAY, "pre-receipt gate set is incomplete")
    expected_python_root = "sha256:" + python_manifest.root.hex()
    expected_rust_root = "sha256:" + rust_manifest.root.hex()
    if (
        gate_rows[12]["evidence"].get("python_implementation_root")
        != expected_python_root
        or gate_rows[12]["evidence"].get("rust_implementation_root")
        != expected_rust_root
    ):
        _fail(FAIL_HOST_REPLAY, "pre-receipt implementation roots are stale")
    compare_endpoints_v1(python, rust)
    compare_host_replay_v1(host, python, rust)
    report = python.report
    receipt = _contract.Q0SaturationReceiptV1(
        terminal_status_id=_contract.Q0TerminalStatusId.DUAL_EXHAUSTIVE_ORACLE_EQUIVALENCE_PASS,
        syntax_raw_operator_application_count=report["syntax_raw_operator_applications"],
        quotient_raw_operator_application_count=report["quotient_raw_operator_applications"],
        canonical_syntax_program_count=report["canonical_syntax_count"],
        behavior_class_count=report["behavior_class_count"],
        frontier_point_count=report["frontier_point_count"],
        maximum_frontier_points_per_class=report["maximum_frontier_size"],
        saturation_round_count=report["direct_saturation_rounds"],
        syntax_program_archive_root=_root_bytes(
            report["syntax_program_root"], "syntax_program_root"
        ),
        syntax_oracle_class_archive_root=_root_bytes(
            report["syntax_class_archive_root"], "syntax_class_root"
        ),
        quotient_engine_class_archive_root=_root_bytes(
            report["direct_class_archive_root"], "direct_class_root"
        ),
        syntax_operator_coverage_root=_root_bytes(
            report["syntax_coverage_root"], "syntax_coverage_root"
        ),
        quotient_operator_coverage_root=_root_bytes(
            report["direct_coverage_root"], "direct_coverage_root"
        ),
        python_implementation_root=python_manifest.root,
        rust_implementation_root=rust_manifest.root,
        python_endpoint_output_root=python.endpoint_root,
        rust_endpoint_output_root=rust.endpoint_root,
        host_replay_class_archive_root=host.class_archive_root,
    )
    return receipt


def finalize_candidate_receipt_v1(
    pre_receipt: PreReceiptQualificationV1,
    receipt: object,
) -> QualifiedGateSetV1:
    if not isinstance(pre_receipt, PreReceiptQualificationV1):
        _fail(FAIL_HOST_REPLAY, "final receipt replay requires pre-receipt token")
    if not isinstance(receipt, _contract.Q0SaturationReceiptV1):
        _fail(FAIL_HOST_REPLAY, "candidate is not Q0SaturationReceiptV1")
    decoded = _cbor.canonical_cbor_decode(receipt.canonical_bytes)
    replayed_root = _cbor.content_hash(
        _contract.SATURATION_RECEIPT_ROOT_DOMAIN, decoded
    )
    rows = json.loads(json.dumps(pre_receipt.gates))
    if (
        decoded[27]
        != _root_bytes(
            rows[12]["evidence"].get("python_implementation_root"),
            "pre-receipt Python implementation root",
        )
        or decoded[28]
        != _root_bytes(
            rows[12]["evidence"].get("rust_implementation_root"),
            "pre-receipt Rust implementation root",
        )
        or decoded[29]
        != _root_bytes(
            rows[13]["evidence"].get("endpoint_state_root"),
            "pre-receipt endpoint state root",
        )
        or decoded[30] != decoded[29]
        or decoded[31]
        != _root_bytes(
            rows[10]["evidence"].get("class_archive_root"),
            "pre-receipt host class archive root",
        )
    ):
        _fail(FAIL_HOST_REPLAY, "candidate receipt bindings differ from pre-receipt")
    preconditions = rows[13]["predicates"]
    gate14_predicates = {
        **preconditions,
        "candidate_receipt_exact_40_fields": len(decoded) == 40,
        "candidate_receipt_dual_status_id": decoded[9]
        == int(_contract.Q0TerminalStatusId.DUAL_EXHAUSTIVE_ORACLE_EQUIVALENCE_PASS),
        "candidate_receipt_gate_total_exact": decoded[32]
        == _contract.Q0_READINESS_GATE_TOTAL,
        "candidate_receipt_gate_mask_exact": decoded[33] == READINESS_GATE_MASK,
        "candidate_receipt_downstream_null_not_run": decoded[34:40]
        == (0, None, 0, False, None, False),
        "candidate_receipt_root_replayed": replayed_root == receipt.receipt_root,
    }
    rows[13] = {
        "gate_id": 14,
        "name": _contract.Q0_READINESS_GATES[13],
        "passed": all(gate14_predicates.values()),
        "predicates": gate14_predicates,
        "evidence": {
            **rows[13]["evidence"],
            "receipt_root": "sha256:" + replayed_root.hex(),
            "receipt_field_count": len(decoded),
            "receipt_gate_mask": decoded[33],
        },
        "pending_dual": False,
    }
    envelope = {
        "schema_version": "hegel-phase3a-q0-final-gate-qualification/1",
        "gates": rows,
    }
    final_rows = _validate_gate_rows_v1(envelope, allow_pending_dual=False)
    if not all(row["passed"] is True for row in final_rows):
        _fail(FAIL_HOST_REPLAY, "candidate receipt did not close all 14 gates")
    payload = _canonical_json_bytes(envelope).rstrip(b"\n")
    root = sha256(_GATE_EVIDENCE_ROOT_DOMAIN + b"\x00" + payload).digest()
    return QualifiedGateSetV1(payload, root, _GATE_ISSUER_TOKEN)


def _git(project_root: Path, *arguments: str) -> str:
    return _git_bytes(project_root, *arguments).decode("utf-8", "strict").strip()


def _git_bytes(project_root: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        _fail(FAIL_SOURCE, f"Git source binding failed: {error}")
    return result.stdout


def verify_manifest_against_commit_v1(
    project_root: Path,
    manifest: SourceManifestV1,
    commit: str,
) -> None:
    for row in manifest.files:
        listing = _git(project_root, "ls-tree", commit, "--", row.path)
        match = re.fullmatch(
            r"(100644|100755) blob ([0-9a-f]{40})\t(.+)", listing
        )
        if match is None or match.group(3) != row.path:
            _fail(FAIL_SOURCE, f"commit tree row is absent or non-file: {row.path}")
        expected_mode = 0o100755 if match.group(1) == "100755" else 0o100644
        payload = _git_bytes(project_root, "show", f"{commit}:./{row.path}")
        if (
            expected_mode != row.mode
            or len(payload) != row.size
            or sha256(payload).digest() != row.digest
        ):
            _fail(FAIL_SOURCE, f"manifest differs from commit tree bytes/mode: {row.path}")


def verify_source_commit_v1(project_root: Path, requested: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", requested) is None:
        _fail(FAIL_SOURCE, "--source-commit must be a full lowercase commit ID")
    observed = _git(project_root, "rev-parse", "HEAD")
    if observed != requested:
        _fail(FAIL_SOURCE, "requested source commit is not current HEAD")
    if _git(project_root, "status", "--porcelain=v1", "--untracked-files=all"):
        _fail(FAIL_SOURCE, "actual qualification requires a completely clean worktree")
    return observed


def _inspect_image(image: str, environment: Mapping[str, str]) -> str:
    try:
        result = subprocess.run(
            [
                DOCKER_EXECUTABLE,
                f"--host={DOCKER_HOST}",
                "image",
                "inspect",
                image,
                "--format={{.Id}}",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(environment),
        )
    except (OSError, subprocess.CalledProcessError) as error:
        _fail(FAIL_DOCKER, f"pinned local image is unavailable: {error}")
    observed = result.stdout.decode("ascii", "strict").strip()
    expected = image.split("@", 1)[1]
    if observed != expected:
        _fail(FAIL_DOCKER, f"local image ID differs for {image}")
    return observed


def _docker_environment(control: Path) -> dict[str, str]:
    config = control / "docker-config"
    config.mkdir(mode=0o700)
    (config / "config.json").write_text("{}\n", encoding="ascii")
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(control),
        "DOCKER_CONFIG": str(config),
        "DOCKER_HOST": DOCKER_HOST,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }


def linux_temp_root_v1() -> Path:
    try:
        root = LINUX_TEMP_ROOT.resolve(strict=True)
        root_stat = root.stat()
    except OSError as error:
        _fail(FAIL_SOURCE, f"Linux temporary root is unavailable: {error}")
    if (
        not stat.S_ISDIR(root_stat.st_mode)
        or LINUX_TEMP_ROOT.is_symlink()
        or root == Path("/mnt")
        or Path("/mnt") in root.parents
    ):
        _fail(FAIL_SOURCE, f"temporary root is not a Linux-filesystem path: {root}")
    return root


def _write_artifact(path: Path, value: object) -> None:
    if path.exists():
        _fail(FAIL_ARTIFACT, f"artifact already exists: {path}")
    if not path.parent.is_dir():
        _fail(FAIL_ARTIFACT, f"artifact parent does not exist: {path.parent}")
    payload = _canonical_json_bytes(value)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        _fail(FAIL_ARTIFACT, f"cannot write artifact: {error}")


def dry_run_plan_v1(
    project_root: Path,
    cargo_cache: Path,
    artifact: Path | None,
) -> dict[str, object]:
    config = load_isolation_config(project_root)
    python_manifest = python_source_manifest_v1(project_root)
    rust_manifest = rust_source_manifest_v1(project_root, cargo_cache)
    placeholder = Path("/sealed-source-snapshot")
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "execution": "DRY_RUN",
        "receipt_created": False,
        "artifact_written": False,
        "requested_artifact": None if artifact is None else str(artifact),
        "profile_id": config["profile_id"],
        "host_role": config["host_role"],
        "python_manifest": python_manifest.json_object(),
        "rust_manifest": rust_manifest.json_object(),
        "python_command": python_endpoint_command(placeholder, "hegel-q0-python-dry-run"),
        "rust_command": rust_endpoint_command(
            placeholder,
            Path("/sealed-cargo-home-snapshot"),
            "hegel-q0-rust-dry-run",
        ),
        "cargo_cache": {
            "path": str(cargo_cache),
            "access": "ro-offline-dependency-cache",
            "implementation_identity": True,
            "locked_registry_dependency_manifest_is_identity": True,
        },
        "isolation_prerequisites": [
            {
                "prerequisite_id": prerequisite_id,
                "name": name,
                "passed": False,
            }
            for prerequisite_id, name in ISOLATION_PREREQUISITES
        ],
        "q0_state": "NOT_RUN",
        "readiness_gate_total": _contract.Q0_READINESS_GATE_TOTAL,
        "readiness_gates_passed": 0,
        "readiness_gate_mask": 0,
        "readiness_gates": [
            {
                "gate_id": gate_id,
                "name": name,
                "passed": False,
                "pending_actual_qualification": True,
            }
            for gate_id, name in READINESS_GATES
        ],
        "q1_status_id": 0,
        "q2_status_id": 0,
        "m3_formal_roots": None,
    }


def run_actual_v1(
    project_root: Path,
    source_commit: str,
    cargo_cache: Path,
    artifact: Path,
) -> dict[str, object]:
    config = load_isolation_config(project_root)
    commit = verify_source_commit_v1(project_root, source_commit)
    if not cargo_cache.is_dir():
        _fail(FAIL_DOCKER, f"offline Cargo cache is absent: {cargo_cache}")
    python_manifest = python_source_manifest_v1(project_root)
    rust_manifest = rust_source_manifest_v1(project_root, cargo_cache)
    verify_manifest_against_commit_v1(project_root, python_manifest, commit)
    verify_manifest_against_commit_v1(project_root, rust_manifest, commit)
    pre_dual_gates = qualify_pre_dual_gate_evidence_v1(project_root)
    with tempfile.TemporaryDirectory(
        prefix="hegel-q0-dual-", dir=linux_temp_root_v1()
    ) as temporary:
        control = Path(temporary)
        python_snapshot = control / "python-source"
        rust_snapshot = control / "rust-source"
        cargo_snapshot = control / "cargo-home"
        materialize_source_snapshot_v1(
            project_root, python_manifest, python_snapshot, commit
        )
        materialize_source_snapshot_v1(
            project_root, rust_manifest, rust_snapshot, commit
        )
        materialize_cargo_home_snapshot_v1(
            cargo_cache, rust_manifest, cargo_snapshot
        )
        environment = _docker_environment(control)
        python_image_id = _inspect_image(PYTHON_IMAGE, environment)
        rust_image_id = _inspect_image(RUST_IMAGE, environment)
        suffix = commit[:12]
        python_command = python_endpoint_command(
            python_snapshot, f"hegel-q0-python-{suffix}"
        )
        rust_command = rust_endpoint_command(
            rust_snapshot, cargo_snapshot, f"hegel-q0-rust-{suffix}"
        )
        isolation_evidence = isolation_prerequisite_evidence_v1(
            project_root,
            commit,
            python_command,
            rust_command,
            python_image_id,
            rust_image_id,
            python_snapshot,
            rust_snapshot,
            cargo_snapshot,
            python_manifest,
            rust_manifest,
        )
        python_run, rust_run = run_endpoints_parallel_v1(
            python_command, rust_command, environment
        )
        python_endpoint = validate_endpoint_v1(python_run.report, "python")
        rust_endpoint = validate_endpoint_v1(rust_run.report, "rust")
        compare_endpoints_v1(python_endpoint, rust_endpoint)
        host = host_local_replay_v1()
        host_manifest = host_replay_source_manifest_v1(project_root, host)
        verify_source_commit_v1(project_root, commit)
        verify_manifest_against_commit_v1(project_root, python_manifest, commit)
        verify_manifest_against_commit_v1(project_root, rust_manifest, commit)
        verify_manifest_against_commit_v1(project_root, host_manifest, commit)
        pre_receipt = finalize_gate_evidence_v1(
            pre_dual_gates,
            python_endpoint,
            rust_endpoint,
            host,
            python_manifest,
            rust_manifest,
            host_manifest,
            isolation_evidence,
            config["downstream_state"],
        )
        candidate_receipt = build_saturation_receipt_v1(
            python_endpoint,
            rust_endpoint,
            host,
            python_manifest,
            rust_manifest,
            pre_receipt,
        )
        qualified_gates = finalize_candidate_receipt_v1(
            pre_receipt, candidate_receipt
        )
        receipt = candidate_receipt
        gates = qualified_gates.gates
        evidence = {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS_PASS,
            "source_commit": commit,
            "profile_id": config["profile_id"],
            "claim_scope": config["claim_scope"],
            "host_role": config["host_role"],
            "images": {
                "python": PYTHON_IMAGE,
                "python_local_id": python_image_id,
                "rust": RUST_IMAGE,
                "rust_local_id": rust_image_id,
            },
            "isolation": {
                **config["isolation"],
                "network": "none",
                "pull_policy": "never",
                "source_mounts": "distinct-read-only",
                "cargo_cache_access": "ro-offline-dependency-cache",
                "cargo_cache_is_implementation_identity": True,
                "locked_registry_dependency_manifest_is_implementation_identity": True,
            },
            "isolation_evidence": isolation_evidence.payload,
            "isolation_evidence_root": (
                "sha256:" + isolation_evidence.evidence_root.hex()
            ),
            "python_manifest": python_manifest.json_object(),
            "rust_manifest": rust_manifest.json_object(),
            "host_replay_manifest": host_manifest.json_object(),
            "python_endpoint": python_run.report,
            "rust_endpoint": rust_run.report,
            "endpoint_stdout_sha256": {
                "python": sha256(python_run.stdout).hexdigest(),
                "rust": sha256(rust_run.stdout).hexdigest(),
            },
            "host_replay": {
                "endpoint_state_root": "sha256:" + host.endpoint_root.hex(),
                "class_archive_root": "sha256:" + host.class_archive_root.hex(),
                "loaded_target_blind_modules": list(host.loaded_modules),
                "trusted_issuer": True,
                "third_independent_endpoint": False,
                "filesystem_hard_isolation": False,
            },
            "receipt_cbor_hex": receipt.canonical_bytes.hex(),
            "receipt_root": "sha256:" + receipt.receipt_root.hex(),
            "pre_dual_gate_evidence_root": (
                "sha256:" + pre_dual_gates.evidence_root.hex()
            ),
            "pre_dual_gate_source_binding": pre_dual_gates.payload[
                "source_binding"
            ],
            "pre_receipt_gate_evidence_root": (
                "sha256:" + pre_receipt.evidence_root.hex()
            ),
            "final_gate_evidence_root": (
                "sha256:" + qualified_gates.evidence_root.hex()
            ),
            "q0_state": "QUALIFIED_NOT_Q1_RUN",
            "readiness_gate_total": _contract.Q0_READINESS_GATE_TOTAL,
            "readiness_gates_passed": _contract.Q0_READINESS_GATE_TOTAL,
            "readiness_gate_mask": READINESS_GATE_MASK,
            "gates": gates,
            "q1_status_id": 0,
            "q1_output_root": None,
            "q2_status_id": 0,
            "role_evaluation_performed": False,
            "m3_formal_roots": None,
            "outside_certificate_issued": False,
            "signatures": None,
        }
    verify_source_commit_v1(project_root, commit)
    verify_manifest_against_commit_v1(project_root, python_manifest, commit)
    verify_manifest_against_commit_v1(project_root, rust_manifest, commit)
    verify_manifest_against_commit_v1(project_root, host_manifest, commit)
    _write_artifact(artifact, evidence)
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--source-commit")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--cargo-cache", type=Path, default=DEFAULT_CARGO_CACHE)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    try:
        if options.dry_run:
            value = dry_run_plan_v1(
                options.project_root.resolve(),
                options.cargo_cache.resolve(),
                None if options.artifact is None else options.artifact.resolve(),
            )
        else:
            if options.source_commit is None or options.artifact is None:
                _fail(FAIL_SOURCE, "--run requires --source-commit and --artifact")
            value = run_actual_v1(
                options.project_root.resolve(),
                options.source_commit,
                options.cargo_cache.resolve(),
                options.artifact.resolve(),
            )
    except SupervisorError as error:
        print(
            json.dumps(
                {
                    "schema_version": "hegel-phase3a-q0-dual-qualification-error/1",
                    "status": error.code,
                    "detail": error.detail,
                    "receipt_created": False,
                    "q0_state": "NOT_RUN",
                    "readiness_gate_total": _contract.Q0_READINESS_GATE_TOTAL,
                    "readiness_gates_passed": 0,
                    "readiness_gate_mask": 0,
                    "q1_status_id": 0,
                    "q2_status_id": 0,
                    "m3_formal_roots": None,
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 1
    print(_canonical_json_bytes(value).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
