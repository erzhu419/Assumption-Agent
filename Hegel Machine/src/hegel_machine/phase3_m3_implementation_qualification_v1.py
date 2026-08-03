"""Commit-bound, offline qualification of the two runnable M3 enumerators.

This module upgrades the Phase-3A static basis only as far as runnable
``ImplementationBindingV1`` objects.  It extracts exact Git blobs into two
disjoint source snapshots, proves that both snapshots are target/split free,
builds Rust offline, executes both implementations in digest-pinned OCI
images with networking disabled, and validates their full 50,001-witness
agreement against one committed typed golden vector.

The resulting receipt is deliberately non-authoritative enumeration evidence.
It creates no seed, key, signature, run/ledger ID, formal M3 output root, or
state transition.  A bare enumerator JSON report can never be promoted here.
"""

from __future__ import annotations

import ast
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import stat
import subprocess
import tarfile
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 project runtime.
    import tomli as tomllib  # type: ignore[no-redef]
from types import MappingProxyType
from typing import Final, Mapping, NoReturn, Sequence

from .phase3_m25_formal_static_basis_v1 import (
    CONTAINER_PROFILE_PATH,
    CONTAINER_SECCOMP_PATH,
    PROJECT_ROOT,
    REPOSITORY_ROOT,
    SECCOMP_PATH,
    FormalStaticBasisV1,
    _dependency_lock_rows,
    _git_blob,
    _source_file_rows,
    build_formal_static_basis_v1,
)
from .phase3_local_runtime_v1 import (
    LinuxLocalTemporaryDirectoryV1,
    LocalDockerControlPlaneV1,
    Phase3LocalRuntimeError,
    build_local_docker_daemon_identity_receipt_v1,
    local_docker_daemon_receipt_binding_v1,
    prepare_local_docker_control_plane_v1,
)
from .phase3_m25_wire_v1 import (
    build_formal_object,
    candidate_content_root,
    candidate_record_tree_root,
    decode_formal_object,
    git_sha1_commit_id,
    id_digest_v1,
)
from .strict_ast_shrink1_v1 import decode_shrink1_canonical_ast
from .strict_cbor_v1 import (
    canonical_cbor_decode,
    canonical_cbor_encode,
    content_hash,
    rfc6962_root,
)


SCHEMA: Final = "hegel-m3-implementation-qualification/1"
CLAIM_LEVEL: Final = "IMPLEMENTATION_BINDING_QUALIFICATION_ONLY_NOT_M3_OUTPUT"
GOLDEN_SCHEMA: Final = "hegel-m3-bounded-dual-agreement/1"
GOLDEN_DOMAIN: Final = "HEGEL/M3_ENUMERATOR_DUAL_GOLDEN/V1"
RECEIPT_DOMAIN: Final = "HEGEL/M3_IMPLEMENTATION_QUALIFICATION/V1"
CARGO_SNAPSHOT_DOMAIN: Final = "HEGEL/M3_CARGO_DEPENDENCY_SNAPSHOT/V1"
CHUNK_BLOB_DOMAIN: Final = b"HEGEL/CHUNK_BLOB/V1"
GOLDEN_PATH: Final = (
    "Hegel Machine/golden_vectors/phase3_m3_bounded_dual_agreement_v1.json"
)
INTEGRATION_PATH: Final = (
    "Hegel Machine/src/hegel_machine/phase3_m3_implementation_qualification_v1.py"
)
QUALIFICATION_CLI_PATH: Final = (
    "Hegel Machine/src/hegel_machine/phase3_m3_implementation_qualification_cli_v1.py"
)
LOCAL_RUNTIME_PATH: Final = (
    "Hegel Machine/src/hegel_machine/phase3_local_runtime_v1.py"
)
ENGINEERING_DOCUMENT_PATH: Final = (
    "Hegel Machine/docs/Hegel_Machine_Phase3A_M3_Implementation_Binding_Qualification_Engineering_v1.md"
)
DETERMINISTIC_CARGO_TRANSCRIPT_AMENDMENT_PATH: Final = (
    "Hegel Machine/docs/"
    "Hegel_Machine_Phase3A_M25_Deterministic_Cargo_Transcript_Amendment_v1.md"
)
BOOTSTRAP_RECORD_PATH: Final = (
    "Hegel Machine/artifacts/phase3_m3_cargo_offline_bootstrap_record_v1.json"
)
PYTHON_ENTRYPOINT_ID: Final = "entrypoint:python-m3-isolated-enumerator-v1"
RUST_ENTRYPOINT_ID: Final = "entrypoint:rust-m3-bounded-enumerator-v1"
DOCKER_POLICY_ID: Final = "hegel-m3-dual-offline-qualification-docker-v1"
RUNTIME_DOCKER_POLICY_ID: Final = "hegel-m3-enumerator-runtime-docker-v1"
RUST_BUILD_DOCKER_POLICY_ID: Final = "hegel-m3-rust-offline-build-docker-v1"
EMPTY_BUILD_STREAM_SHA256: Final = hashlib.sha256(b"").digest()
RUST_TOOLCHAIN_BIN: Final = (
    "/usr/local/rustup/toolchains/1.88.0-x86_64-unknown-linux-gnu/bin"
)
PYTHON_RUNTIME_ENVIRONMENT: Final = MappingProxyType(
    {
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "TZ": "UTC",
    }
)
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
        "CARGO_NET_OFFLINE": "true",
        "CARGO_PROFILE_RELEASE_CODEGEN_UNITS": "1",
        "CARGO_TARGET_DIR": "/output",
        "RUSTC": f"{RUST_TOOLCHAIN_BIN}/rustc",
        "RUSTDOC": f"{RUST_TOOLCHAIN_BIN}/rustdoc",
    }
)
BUILD_SECCOMP_PATH: Final = PROJECT_ROOT / "config/phase3_m3_offline_build_seccomp_v1.json"
BUILD_SECCOMP_REPOSITORY_PATH: Final = (
    "Hegel Machine/config/phase3_m3_offline_build_seccomp_v1.json"
)
REQUIRED_QUALIFICATION_BASIS_PATHS: Final = (
    INTEGRATION_PATH,
    QUALIFICATION_CLI_PATH,
    LOCAL_RUNTIME_PATH,
    ENGINEERING_DOCUMENT_PATH,
    DETERMINISTIC_CARGO_TRANSCRIPT_AMENDMENT_PATH,
    GOLDEN_PATH,
    BUILD_SECCOMP_REPOSITORY_PATH,
    BOOTSTRAP_RECORD_PATH,
)

FAIL_COMMIT = "FAIL_M3_IMPLEMENTATION_QUALIFICATION_COMMIT"
FAIL_GOLDEN = "FAIL_M3_IMPLEMENTATION_QUALIFICATION_GOLDEN"
FAIL_SOURCE_CLOSURE = "FAIL_M3_IMPLEMENTATION_SOURCE_CLOSURE"
FAIL_TARGET_VISIBILITY = "FAIL_M3_IMPLEMENTATION_TARGET_VISIBILITY"
FAIL_CONTAINER_POLICY = "FAIL_M3_IMPLEMENTATION_CONTAINER_POLICY"
FAIL_BUILD = "FAIL_M3_IMPLEMENTATION_BUILD"
FAIL_EXECUTION = "FAIL_M3_IMPLEMENTATION_EXECUTION"
FAIL_REPORT = "FAIL_M3_IMPLEMENTATION_REPORT"
FAIL_RECEIPT = "FAIL_M3_IMPLEMENTATION_RECEIPT"
FAIL_BINDING = "FAIL_M3_EXECUTION_IMPLEMENTATION_BINDING"


class M3ImplementationQualificationError(RuntimeError):
    """Stable fail-closed qualification error."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise M3ImplementationQualificationError(code, detail)


PYTHON_SOURCE_PATHS: Final = (
    "Hegel Machine/src/hegel_machine/phase3_m3_isolated_entrypoint_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_bounded_enumerator_cli_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_bounded_enumerator_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_dsl_core_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_shrink1_core_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_record_wire_v1.py",
    "Hegel Machine/src/hegel_machine/strict_ast_v1.py",
    "Hegel Machine/src/hegel_machine/strict_ast_shrink1_v1.py",
    "Hegel Machine/src/hegel_machine/strict_cbor_v1.py",
)
RUST_CRATE_PATHS: Final = (
    "Hegel Machine/rust/m3_closure_enumerator",
    "Hegel Machine/rust/strict_canonicalizer",
    "Hegel Machine/rust/strict_canonicalizer_shrink1",
)
RUST_SOURCE_PATHS: Final = tuple(
    f"{crate}/{suffix}"
    for crate in RUST_CRATE_PATHS
    for suffix in (
        "Cargo.toml",
        "Cargo.lock",
        "src/lib.rs",
        "src/main.rs",
    )
) + ("Hegel Machine/rust/m3_closure_enumerator/src/formal_core.rs",)

FORBIDDEN_SOURCE_PATH_PARTS: Final = (
    "phase3_dsl_v1.py",
    "phase3_shrink1_registry_v1.py",
    "phase3_m25_formal_container_executor_v1.py",
    "formal_bridge_m25",
)
FORBIDDEN_SOURCE_TOKENS: Final = (
    b"ODD_REDUCTION_UNIVERSE",
    b"OMITTED_SINK_UNIVERSE",
    b"TARGET_P3A_GENERIC_ODD_REDUCTION_V1",
    b"CONTROL_P3A_OBSERVED_OMITTED_SINK_V1",
    b"split_seed_commitment",
    b"phase3_m25_container_ceremony",
)

REPORT_FIELDS: Final = frozenset(
    {
        "active_aggregate_map_ids",
        "aliases_excluded_before_count",
        "all_type_buckets_closed",
        "authoritative_claim_allowed",
        "bucket_accounting_root_or_null",
        "bucket_record_count",
        "canonical_program_archive_root_or_null",
        "canonical_program_count",
        "canonicalizer_profile",
        "child_dsl_spec_root",
        "chunk_manifest_count",
        "claim_level",
        "closure_cardinality_or_null",
        "closure_status",
        "closure_status_id",
        "dsl_version",
        "first_out_of_budget_program_cbor_hex_or_null",
        "first_out_of_budget_program_hash_or_null",
        "freeze_version",
        "frontier_exhausted",
        "identifier_registry_root",
        "implementation",
        "implementation_id",
        "implementation_machine_id",
        "maximum_canonical_programs",
        "maximum_raw_operator_applications",
        "mdl_code_table_id",
        "operator_semantics_root",
        "program_chunk_manifest_root_or_null",
        "program_record_count",
        "raw_expansion_limit_hit",
        "raw_operator_application_count",
        "records_per_chunk",
        "schema_version",
        "secrets_accessed",
        "split_material_accessed",
        "target_roles_evaluated",
        "tombstoned_aggregate_map_ids",
        "traversal_prefix_complete",
        "wall_clock_abort_hit",
    }
)

GOLDEN_TOP_FIELDS: Final = frozenset(
    {
        "schema_version",
        "claim_level",
        "authoritative_claim_allowed",
        "formal_root_publication_allowed",
        "formal_m3_output",
        "target_roles_evaluated",
        "split_material_accessed",
        "secrets_accessed",
        "binding_roots",
        "implementations",
        "expected",
    }
)
GOLDEN_BINDING_FIELDS: Final = frozenset(
    {"child_dsl_spec_root", "operator_semantics_root", "identifier_registry_root"}
)
GOLDEN_IMPLEMENTATION_FIELDS: Final = frozenset(
    {"implementation_id", "implementation_machine_id", "language"}
)
GOLDEN_EXPECTED_ORDER: Final = (
    "active_aggregate_map_ids",
    "aliases_excluded_before_count",
    "all_type_buckets_closed",
    "bucket_accounting_root",
    "bucket_record_count",
    "canonical_program_archive_root",
    "canonical_program_count",
    "canonicalizer_profile",
    "chunk_manifest_count",
    "closure_cardinality_or_null",
    "closure_status",
    "closure_status_id",
    "dsl_version",
    "first_out_of_budget_program_cbor_hex",
    "first_out_of_budget_program_hash",
    "freeze_version",
    "frontier_exhausted",
    "maximum_canonical_programs",
    "maximum_raw_operator_applications",
    "mdl_code_table_id",
    "program_chunk_manifest_root",
    "program_record_count",
    "raw_operator_application_count",
    "records_per_chunk",
    "tombstoned_aggregate_map_ids",
    "traversal_prefix_complete",
)
GOLDEN_EXPECTED_FIELDS: Final = frozenset(GOLDEN_EXPECTED_ORDER)

IMPLEMENTATION_RECEIPT_FIELDS: Final = frozenset(
    {
        "implementation_id",
        "implementation_machine_id",
        "source_root",
        "source_file_count",
        "dependency_lock_root",
        "dependency_snapshot_root_or_null",
        "dependency_snapshot_file_count",
        "execution_environment_spec_root",
        "image_ref",
        "bound_executable_locator",
        "binary_digest",
        "compiler_or_interpreter_version_digest",
        "entrypoint_id_digest",
        "implementation_binding_root",
        "canonical_report_sha256",
        "execution_stdout_sha256",
        "runtime_container_environment_sha256",
        "build_container_environment_sha256_or_null",
        "canonical_program_records_stream_sha256",
        "program_chunk_manifests_stream_sha256",
        "bucket_accounting_records_stream_sha256",
        "build_stdout_sha256_or_null",
        "build_stderr_sha256_or_null",
        "input_snapshot_target_free",
        "archive_file_set_verified",
        "host_strict_archive_replay_verified",
        "witness_adjacency_verified",
    }
)
RECEIPT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "claim_level",
        "authoritative_claim_allowed",
        "basis_commit",
        "golden_vector_root",
        "python",
        "rust",
        "agreement",
        "docker_policy_id",
        "runtime_docker_policy_id",
        "rust_build_docker_policy_id",
        "runtime_seccomp_sha256",
        "build_seccomp_sha256",
        "local_docker_daemon_receipt_binding",
        "cargo_offline_bootstrap_record_sha256",
        "pull_policy_never",
        "network_mode_none",
        "independent_source_snapshots",
        "independent_archive_bytes_equal",
        "target_inputs_visible",
        "target_roles_evaluated",
        "split_material_accessed",
        "secrets_accessed",
        "formal_m3_output_roots_generated",
        "m3_state",
        "receipt_cbor_hex",
        "receipt_root",
    }
)


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


def _container_environment_digest(environment: Mapping[str, str]) -> bytes:
    if not isinstance(environment, Mapping) or not environment:
        _fail(FAIL_CONTAINER_POLICY, "container environment is empty")
    checked: dict[str, str] = {}
    for key, value in environment.items():
        if (
            type(key) is not str
            or re.fullmatch(r"[A-Z][A-Z0-9_]*", key) is None
            or type(value) is not str
            or "\x00" in value
        ):
            _fail(FAIL_CONTAINER_POLICY, "container environment entry is invalid")
        checked[key] = value
    payload = json.dumps(
        checked,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).digest()


def _strict_json_load(
    payload: bytes, *, label: str, code: str = FAIL_GOLDEN
) -> object:
    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                _fail(code, f"{label} repeats key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(code, f"{label} is not strict UTF-8 JSON: {error}")


def _exact_fields(value: object, expected: frozenset[str], *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = set(value) if isinstance(value, Mapping) else {type(value).__name__}
        _fail(FAIL_RECEIPT if label.startswith("receipt") else FAIL_GOLDEN, f"{label} fields differ: {sorted(actual ^ expected)}")
    return value


def _hex32(value: object, *, label: str, code: str = FAIL_GOLDEN) -> bytes:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        _fail(code, f"{label} must be lowercase 32-byte hex")
    return bytes.fromhex(value)


def _exact_uint(
    value: object, *, label: str, code: str, minimum: int = 0
) -> int:
    if type(value) is not int or value < minimum:
        _fail(code, f"{label} must be an integer >= {minimum}")
    return value


def _receipt_text(value: object, *, label: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        _fail(FAIL_RECEIPT, f"{label} must be nonempty text without NUL")
    return value


def _formalize_golden_value(key: str, value: object) -> object:
    if key.endswith(("_root", "_hash")):
        return _hex32(value, label=key)
    if key.endswith("_cbor_hex"):
        if type(value) is not str or len(value) % 2 or re.fullmatch(r"[0-9a-f]*", value) is None:
            _fail(FAIL_GOLDEN, f"{key} must be lowercase even-length hex")
        return bytes.fromhex(value)
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is str:
        return value.encode("utf-8")
    if isinstance(value, list):
        return tuple(
            item.encode("utf-8") if type(item) is str else item for item in value
        )
    _fail(FAIL_GOLDEN, f"unsupported typed golden value {key}")


def validate_dual_golden_v1(value: object) -> tuple[Mapping[str, object], bytes, bytes]:
    golden = _exact_fields(value, GOLDEN_TOP_FIELDS, label="golden")
    bindings = _exact_fields(
        golden["binding_roots"], GOLDEN_BINDING_FIELDS, label="golden.binding_roots"
    )
    expected = _exact_fields(
        golden["expected"], GOLDEN_EXPECTED_FIELDS, label="golden.expected"
    )
    implementations = golden["implementations"]
    if not isinstance(implementations, list) or len(implementations) != 2:
        _fail(FAIL_GOLDEN, "golden implementations must contain Python then Rust")
    checked_implementations: list[Mapping[str, object]] = []
    for index, item in enumerate(implementations):
        checked = _exact_fields(
            item,
            GOLDEN_IMPLEMENTATION_FIELDS,
            label=f"golden.implementations[{index}]",
        )
        checked_implementations.append(checked)
    if golden["schema_version"] != GOLDEN_SCHEMA:
        _fail(FAIL_GOLDEN, "golden schema differs")
    if (
        golden["claim_level"] != "FORMAL_PROFILE_CANDIDATE_NOT_AUTHORITY"
        or golden["authoritative_claim_allowed"] is not False
        or golden["formal_root_publication_allowed"] is not False
        or golden["formal_m3_output"] is not False
        or golden["target_roles_evaluated"] is not False
        or golden["split_material_accessed"] is not False
        or golden["secrets_accessed"] is not False
    ):
        _fail(FAIL_GOLDEN, "golden attempts an authority or hidden-input claim")
    expected_implementations = (
        (1, "hegel-python-m3-bounded-closure-enumerator-v1", "python"),
        (2, "hegel-rust-m3-bounded-closure-enumerator-v1", "rust"),
    )
    for checked, expected_row in zip(checked_implementations, expected_implementations, strict=True):
        if tuple(checked[key] for key in ("implementation_id", "implementation_machine_id", "language")) != expected_row:
            _fail(FAIL_GOLDEN, "golden implementation identity differs")
    binding_value = tuple(
        _hex32(bindings[name], label=name)
        for name in (
            "child_dsl_spec_root",
            "operator_semantics_root",
            "identifier_registry_root",
        )
    )
    implementation_value = tuple(
        (
            row["implementation_id"],
            str(row["implementation_machine_id"]).encode("ascii"),
            str(row["language"]).encode("ascii"),
        )
        for row in checked_implementations
    )
    expected_value = tuple(
        _formalize_golden_value(name, expected[name]) for name in GOLDEN_EXPECTED_ORDER
    )
    typed_value = (
        1,
        GOLDEN_SCHEMA.encode("ascii"),
        str(golden["claim_level"]).encode("ascii"),
        False,
        False,
        False,
        binding_value,
        implementation_value,
        expected_value,
        False,
        False,
        False,
    )
    preimage = canonical_cbor_encode(typed_value)
    root = content_hash(GOLDEN_DOMAIN, typed_value)
    return MappingProxyType(dict(golden)), preimage, root


def load_committed_dual_golden_v1(
    repository_root: Path, basis_commit: str
) -> tuple[Mapping[str, object], bytes, bytes]:
    return validate_dual_golden_v1(
        _strict_json_load(
            _git_blob(repository_root, basis_commit, GOLDEN_PATH), label=GOLDEN_PATH
        )
    )


def _module_path(module: str) -> str:
    return "Hegel Machine/src/hegel_machine/" + module.replace(".", "/") + ".py"


def _python_local_imports(path: str, payload: bytes) -> set[str]:
    try:
        tree = ast.parse(payload, filename=path)
    except (SyntaxError, ValueError) as error:
        _fail(FAIL_SOURCE_CLOSURE, f"cannot parse {path}: {error}")
    discovered: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level != 1:
            continue
        if node.module:
            discovered.add(_module_path(node.module))
        else:
            for alias in node.names:
                discovered.add(_module_path(alias.name))
    return discovered


def validate_python_source_closure_v1(
    repository_root: Path, basis_commit: str
) -> Mapping[str, bytes]:
    blobs = {path: _git_blob(repository_root, basis_commit, path) for path in PYTHON_SOURCE_PATHS}
    expected = set(PYTHON_SOURCE_PATHS)
    pending = [PYTHON_SOURCE_PATHS[0]]
    discovered: set[str] = set()
    while pending:
        path = pending.pop()
        if path in discovered:
            continue
        discovered.add(path)
        for imported in _python_local_imports(path, blobs[path]):
            if imported not in expected:
                _fail(FAIL_SOURCE_CLOSURE, f"Python closure omits local import {imported} from {path}")
            pending.append(imported)
    if discovered != expected:
        _fail(FAIL_SOURCE_CLOSURE, f"Python source list has unreachable extras: {sorted(expected - discovered)}")
    _validate_target_free_blobs(blobs, implementation="python")
    return MappingProxyType(blobs)


def _cargo_path_dependencies(manifest: bytes, crate_path: str) -> set[str]:
    try:
        parsed = tomllib.loads(manifest.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        _fail(FAIL_SOURCE_CLOSURE, f"invalid Cargo.toml for {crate_path}: {error}")
    dependencies = parsed.get("dependencies", {})
    if not isinstance(dependencies, dict):
        _fail(FAIL_SOURCE_CLOSURE, f"invalid dependencies table for {crate_path}")
    result: set[str] = set()
    for specification in dependencies.values():
        if not isinstance(specification, dict) or "path" not in specification:
            continue
        relative = specification["path"]
        if type(relative) is not str:
            _fail(FAIL_SOURCE_CLOSURE, f"non-text Cargo path dependency in {crate_path}")
        resolved = PurePosixPath(crate_path).joinpath(relative)
        normalized: list[str] = []
        for part in resolved.parts:
            if part == "..":
                if not normalized:
                    _fail(FAIL_SOURCE_CLOSURE, "Cargo dependency escapes repository")
                normalized.pop()
            elif part not in {"", "."}:
                normalized.append(part)
        result.add("/".join(normalized))
    return result


def validate_rust_source_closure_v1(
    repository_root: Path, basis_commit: str
) -> Mapping[str, bytes]:
    blobs = {path: _git_blob(repository_root, basis_commit, path) for path in RUST_SOURCE_PATHS}
    crate_set = set(RUST_CRATE_PATHS)
    discovered = {RUST_CRATE_PATHS[0]}
    pending = [RUST_CRATE_PATHS[0]]
    while pending:
        crate = pending.pop()
        manifest_path = crate + "/Cargo.toml"
        for dependency in _cargo_path_dependencies(blobs[manifest_path], crate):
            if dependency not in crate_set:
                _fail(FAIL_SOURCE_CLOSURE, f"Rust closure omits path crate {dependency}")
            if dependency not in discovered:
                discovered.add(dependency)
                pending.append(dependency)
    if discovered != crate_set:
        _fail(FAIL_SOURCE_CLOSURE, f"Rust source list has unreachable crates: {sorted(crate_set - discovered)}")
    root_manifest = blobs[RUST_CRATE_PATHS[0] + "/Cargo.toml"]
    if b"formal_bridge_m25" in root_manifest or b"hegel-formal-bridge-m25" in root_manifest:
        _fail(FAIL_TARGET_VISIBILITY, "Rust enumerator still depends on target-aware formal bridge")
    _validate_target_free_blobs(blobs, implementation="rust")
    return MappingProxyType(blobs)


def _validate_target_free_blobs(blobs: Mapping[str, bytes], *, implementation: str) -> None:
    for path, payload in blobs.items():
        if any(part in path for part in FORBIDDEN_SOURCE_PATH_PARTS):
            _fail(FAIL_TARGET_VISIBILITY, f"{implementation} snapshot contains forbidden path {path}")
        for token in FORBIDDEN_SOURCE_TOKENS:
            if token in payload:
                _fail(FAIL_TARGET_VISIBILITY, f"{implementation} snapshot exposes forbidden token {token!r} in {path}")


def _write_snapshot(root: Path, blobs: Mapping[str, bytes], *, strip_prefix: str | None = None) -> None:
    root.mkdir(parents=True, mode=0o755, exist_ok=False)
    for repository_path, payload in blobs.items():
        relative = repository_path
        if strip_prefix is not None:
            if not repository_path.startswith(strip_prefix):
                _fail(FAIL_SOURCE_CLOSURE, f"snapshot path is outside {strip_prefix}")
            relative = repository_path[len(strip_prefix) :]
        destination = root / relative
        destination.parent.mkdir(parents=True, mode=0o755, exist_ok=True)
        destination.write_bytes(payload)
        destination.chmod(0o444)
    for directory, _, _ in os.walk(root):
        Path(directory).chmod(0o755)


def _image_ref(value: object, *, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"[a-z0-9._/-]+@sha256:[0-9a-f]{64}", value) is None:
        _fail(FAIL_CONTAINER_POLICY, f"{label} image is not digest pinned")
    return value


def _load_profile(repository_root: Path, commit: str) -> Mapping[str, object]:
    raw = _strict_json_load(
        _git_blob(repository_root, commit, CONTAINER_PROFILE_PATH),
        label=CONTAINER_PROFILE_PATH,
        code=FAIL_CONTAINER_POLICY,
    )
    if not isinstance(raw, dict):
        _fail(FAIL_CONTAINER_POLICY, "container profile is not an object")
    try:
        network = raw["network_policy"]
        images = raw["images"]
        if (
            not isinstance(network, dict)
            or network.get("allow_registry_access") is not False
            or network.get("allow_runtime_network") is not False
            or network.get("docker_network") != "none"
            or network.get("pull_policy") != "never"
            or not isinstance(images, dict)
        ):
            raise KeyError("offline policy")
        _image_ref(images["python_attester"], label="Python")
        _image_ref(images["rust_attester"], label="Rust")
    except KeyError as error:
        _fail(FAIL_CONTAINER_POLICY, f"container profile lacks {error}")
    return MappingProxyType(raw)


def _docker_base(
    control_plane: LocalDockerControlPlaneV1,
    image: str,
    *,
    seccomp_path: Path,
    user: str = "65534:65534",
) -> list[str]:
    try:
        user_id, group_id = user.split(":", 1)
        int(user_id)
        int(group_id)
    except (AttributeError, TypeError, ValueError):
        _fail(FAIL_CONTAINER_POLICY, "Docker user must be numeric uid:gid")
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
        image,
    )


def _run(
    command: Sequence[str],
    *,
    code: str,
    timeout: int,
    environment: Mapping[str, str],
) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
            env=dict(environment),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        _fail(code, str(error))
    if completed.returncode != 0:
        _fail(code, completed.stderr.decode("utf-8", "replace")[-2000:])
    return completed


def _docker_with_options(
    control_plane: LocalDockerControlPlaneV1,
    image: str,
    options: Sequence[str],
    command: Sequence[str],
    *,
    seccomp_path: Path,
    container_environment: Mapping[str, str],
    user: str = "65534:65534",
) -> list[str]:
    base = _docker_base(
        control_plane, image, seccomp_path=seccomp_path, user=user
    )
    _container_environment_digest(container_environment)
    exact_environment = tuple(
        f"{key}={container_environment[key]}" for key in sorted(container_environment)
    )
    return [
        *base[:-1],
        *options,
        "--entrypoint=/usr/bin/env",
        base[-1],
        "-i",
        *exact_environment,
        *command,
    ]


def _parse_single_json(stdout: bytes, *, label: str) -> Mapping[str, object]:
    value = _strict_json_load(stdout, label=label, code=FAIL_REPORT)
    if not isinstance(value, dict):
        _fail(FAIL_REPORT, f"{label} is not an object")
    return value


def _qualify_local_docker_control_plane_v1(
    control_plane: LocalDockerControlPlaneV1,
    *,
    repository_root: Path,
) -> tuple[Mapping[str, object], bytes]:
    version = _run(
        control_plane.command("version", "--format", "{{json .}}"),
        code=FAIL_CONTAINER_POLICY,
        timeout=30,
        environment=control_plane.environment,
    )
    info = _run(
        control_plane.command("info", "--format", "{{json .}}"),
        code=FAIL_CONTAINER_POLICY,
        timeout=30,
        environment=control_plane.environment,
    )
    version_payload = _strict_json_load(
        version.stdout,
        label="local Docker version identity",
        code=FAIL_CONTAINER_POLICY,
    )
    info_payload = _strict_json_load(
        info.stdout,
        label="local Docker daemon identity",
        code=FAIL_CONTAINER_POLICY,
    )
    if not isinstance(version_payload, Mapping) or not isinstance(
        info_payload, Mapping
    ):
        _fail(FAIL_CONTAINER_POLICY, "local Docker identity payload is not an object")
    try:
        receipt = build_local_docker_daemon_identity_receipt_v1(
            control_plane,
            version_payload=version_payload,
            info_payload=info_payload,
            repository_root=repository_root,
        )
        binding = local_docker_daemon_receipt_binding_v1(receipt)
    except Phase3LocalRuntimeError as error:
        _fail(FAIL_CONTAINER_POLICY, str(error))
    return MappingProxyType(receipt), binding


def _probe_python(
    control_plane: LocalDockerControlPlaneV1,
    image: str,
    *,
    seccomp_path: Path,
) -> tuple[str, bytes, bytes, bytes]:
    script = (
        "import hashlib,json,os,sys;"
        "p=os.path.realpath(sys.executable);"
        "b=open(p,'rb').read();"
        "print(json.dumps({'binary_path':p,'binary_sha256':hashlib.sha256(b).hexdigest(),"
        "'version':sys.version},sort_keys=True,separators=(',',':')))"
    )
    completed = _run(
        _docker_with_options(
            control_plane,
            image,
            (),
            ("python3", "-c", script),
            seccomp_path=seccomp_path,
            container_environment=PYTHON_RUNTIME_ENVIRONMENT,
        ),
        code=FAIL_EXECUTION,
        timeout=60,
        environment=control_plane.environment,
    )
    value = _parse_single_json(completed.stdout, label="Python interpreter probe")
    if set(value) != {"binary_path", "binary_sha256", "version"}:
        _fail(FAIL_EXECUTION, "Python interpreter probe fields differ")
    path = value["binary_path"]
    version = value["version"]
    if type(path) is not str or not path.startswith("/usr/local/bin/python") or type(version) is not str:
        _fail(FAIL_EXECUTION, "Python interpreter identity differs")
    digest = _hex32(value["binary_sha256"], label="Python binary", code=FAIL_EXECUTION)
    return path, digest, hashlib.sha256(version.encode("utf-8")).digest(), completed.stdout


def _probe_rust_version(
    control_plane: LocalDockerControlPlaneV1,
    image: str,
    *,
    seccomp_path: Path,
) -> tuple[bytes, bytes]:
    completed = _run(
        _docker_with_options(
            control_plane,
            image,
            (),
            (f"{RUST_TOOLCHAIN_BIN}/rustc", "-vV"),
            seccomp_path=seccomp_path,
            container_environment=RUST_RUNTIME_ENVIRONMENT,
        ),
        code=FAIL_EXECUTION,
        timeout=60,
        environment=control_plane.environment,
    )
    if not completed.stdout.startswith(b"rustc "):
        _fail(FAIL_EXECUTION, "Rust compiler probe differs")
    return hashlib.sha256(completed.stdout).digest(), completed.stdout


def _cargo_registry_packages(lock_payload: bytes) -> tuple[tuple[str, str, bytes], ...]:
    """Return the exact crates.io packages selected by the committed lockfile."""

    try:
        parsed = tomllib.loads(lock_payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        _fail(FAIL_BUILD, f"committed Cargo.lock is invalid: {error}")
    packages = parsed.get("package")
    if not isinstance(packages, list):
        _fail(FAIL_BUILD, "committed Cargo.lock has no package array")
    selected: list[tuple[str, str, bytes]] = []
    identities: set[tuple[str, str]] = set()
    for row in packages:
        if not isinstance(row, dict):
            _fail(FAIL_BUILD, "committed Cargo.lock package row is not a table")
        source = row.get("source")
        if source is None:
            continue
        if source != "registry+https://github.com/rust-lang/crates.io-index":
            _fail(FAIL_BUILD, f"unsupported locked Cargo source {source!r}")
        name = row.get("name")
        version = row.get("version")
        checksum = row.get("checksum")
        if (
            type(name) is not str
            or re.fullmatch(r"[A-Za-z0-9_-]+", name) is None
            or type(version) is not str
            or re.fullmatch(r"[0-9A-Za-z.+-]+", version) is None
        ):
            _fail(FAIL_BUILD, "locked Cargo package identity is not path-safe")
        checksum_bytes = _hex32(checksum, label="Cargo package checksum", code=FAIL_BUILD)
        identity = (name, version)
        if identity in identities:
            _fail(FAIL_BUILD, f"duplicate locked Cargo package {name} {version}")
        identities.add(identity)
        selected.append((name, version, checksum_bytes))
    if not selected:
        _fail(FAIL_BUILD, "committed Cargo.lock selects no registry packages")
    return tuple(sorted(selected))


def _cached_crate_path(name: str, version: str, checksum: bytes) -> Path:
    cache_root = Path.home() / ".cargo/registry/cache"
    if not cache_root.is_dir():
        _fail(FAIL_BUILD, "offline Cargo crate cache is absent")
    candidates = sorted(cache_root.glob(f"*/{name}-{version}.crate"))
    matching: list[Path] = []
    for candidate in candidates:
        if not candidate.is_file() or candidate.is_symlink():
            continue
        try:
            digest = hashlib.sha256(candidate.read_bytes()).digest()
        except OSError as error:
            _fail(FAIL_BUILD, f"cannot read cached crate {candidate}: {error}")
        if digest == checksum:
            matching.append(candidate)
    if not matching:
        _fail(
            FAIL_BUILD,
            f"no checksum-exact offline crate for {name} {version} ({checksum.hex()})",
        )
    # Duplicate cache trees are harmless only when every selected archive is
    # byte-identical.  The snapshot identity is path-neutral.
    return matching[0]


def _extract_locked_crate(
    archive_path: Path,
    vendor_root: Path,
    *,
    name: str,
    version: str,
    package_checksum: bytes,
) -> tuple[tuple[object, ...], ...]:
    top = f"{name}-{version}"
    destination_root = vendor_root / top
    destination_root.mkdir(mode=0o700, exist_ok=False)
    rows: list[tuple[object, ...]] = []
    seen: set[str] = set()
    total_size = 0
    try:
        archive = tarfile.open(archive_path, mode="r:gz")
    except (OSError, tarfile.TarError) as error:
        _fail(FAIL_BUILD, f"cannot open cached crate {name} {version}: {error}")
    with archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or not path.parts
                or path.parts[0] != top
                or any(part in {"", ".", ".."} for part in path.parts)
            ):
                _fail(FAIL_BUILD, f"unsafe archive path in {name} {version}: {member.name!r}")
            relative = PurePosixPath(*path.parts[1:])
            relative_text = relative.as_posix()
            if not relative_text or relative_text in seen:
                _fail(FAIL_BUILD, f"duplicate/empty archive path in {name} {version}")
            seen.add(relative_text)
            destination = destination_root.joinpath(*relative.parts)
            if member.isdir():
                destination.mkdir(parents=True, mode=0o700, exist_ok=True)
                continue
            if not member.isfile() or member.size < 0 or member.size > 16 * 1024 * 1024:
                _fail(FAIL_BUILD, f"unsupported archive member in {name} {version}: {member.name!r}")
            total_size += member.size
            if total_size > 128 * 1024 * 1024:
                _fail(FAIL_BUILD, f"cached crate {name} {version} exceeds extraction limit")
            destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                _fail(FAIL_BUILD, f"cannot extract {member.name!r}")
            payload = source.read(member.size + 1)
            if len(payload) != member.size:
                _fail(FAIL_BUILD, f"cached crate member length differs: {member.name!r}")
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
                    package_checksum,
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
        {"files": file_checksums, "package": package_checksum.hex()},
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
            package_checksum,
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
    lock_payload: bytes, snapshot_root: Path
) -> tuple[bytes, int]:
    """Create a run-private vendored tree from checksum-exact cached crates."""

    snapshot_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    all_rows: list[tuple[object, ...]] = []
    packages = _cargo_registry_packages(lock_payload)
    for name, version, checksum in packages:
        archive = _cached_crate_path(name, version, checksum)
        all_rows.extend(
            _extract_locked_crate(
                archive,
                snapshot_root,
                name=name,
                version=version,
                package_checksum=checksum,
            )
        )
    snapshot_root.chmod(0o700)
    typed_packages = tuple(
        (name.encode("utf-8"), version.encode("ascii"), checksum)
        for name, version, checksum in packages
    )
    typed_value = (1, typed_packages, tuple(all_rows))
    return content_hash(CARGO_SNAPSHOT_DOMAIN, typed_value), len(all_rows)


def _validate_cargo_bootstrap_record_v1(
    payload: bytes, *, cargo_lock_payload: bytes
) -> tuple[bytes, int, bytes]:
    value = _strict_json_load(
        payload,
        label=BOOTSTRAP_RECORD_PATH,
        code=FAIL_BUILD,
    )
    expected_fields = {
        "schema_version",
        "claim_level",
        "authoritative_claim_allowed",
        "bootstrap_window_utc",
        "cargo_lock",
        "downloaded_crates",
        "successful_exact_command",
        "failed_preflight",
        "post_bootstrap_policy",
        "first_verified_snapshot",
        "generated_seed",
        "generated_key",
        "generated_signature",
        "generated_formal_root",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        _fail(FAIL_BUILD, "Cargo bootstrap audit record fields differ")
    if (
        value["schema_version"] != "hegel-m3-cargo-offline-bootstrap-record/1"
        or value["claim_level"]
        != "LOCAL_DEPENDENCY_BOOTSTRAP_AUDIT_ONLY_NOT_FORMAL_ROOT"
        or value["authoritative_claim_allowed"] is not False
        or any(
            value[name] is not False
            for name in (
                "generated_seed",
                "generated_key",
                "generated_signature",
                "generated_formal_root",
            )
        )
    ):
        _fail(FAIL_BUILD, "Cargo bootstrap audit authority boundary differs")
    lock = value["cargo_lock"]
    packages = _cargo_registry_packages(cargo_lock_payload)
    if not isinstance(lock, dict) or lock != {
        "repository_path": "Hegel Machine/rust/m3_closure_enumerator/Cargo.lock",
        "sha256": hashlib.sha256(cargo_lock_payload).hexdigest(),
        "locked_registry_package_count": len(packages),
    }:
        _fail(FAIL_BUILD, "Cargo bootstrap lock binding differs")
    downloaded = value["downloaded_crates"]
    locked = {(name, version): checksum for name, version, checksum in packages}
    if not isinstance(downloaded, list) or len(downloaded) != 1:
        _fail(FAIL_BUILD, "Cargo bootstrap download set differs")
    downloaded_row = downloaded[0]
    if not isinstance(downloaded_row, dict) or downloaded_row != {
        "name": "libc",
        "version": "0.2.189",
        "locked_source": "registry+https://github.com/rust-lang/crates.io-index",
        "locked_and_observed_archive_sha256": locked.get(
            ("libc", "0.2.189"), b""
        ).hex(),
        "cargo_stdout_event": "Downloaded libc v0.2.189",
    }:
        _fail(FAIL_BUILD, "Cargo bootstrap downloaded-crate evidence differs")
    if value["post_bootstrap_policy"] != {
        "docker_pull": "never",
        "docker_network": "none",
        "cargo_offline": True,
        "host_cargo_registry_mounted_into_build_container": False,
        "build_input": "run-private checksum-exact vendor snapshot",
        "snapshot_root_mode_octal": "0700",
    }:
        _fail(FAIL_BUILD, "Cargo post-bootstrap offline policy differs")
    command = value["successful_exact_command"]
    if (
        type(command) is not str
        or "cargo fetch --locked" not in command
        or "--pull=never" not in command
        or "--network=bridge" not in command
        or "${HEGEL_BOOTSTRAP_CARGO_REGISTRY:" not in command
        or "${HEGEL_BOOTSTRAP_PROJECT_ROOT:" not in command
        or "/home/" in command
    ):
        _fail(
            FAIL_BUILD,
            "Cargo bootstrap command template is absent or machine-path-bound",
        )
    window = value["bootstrap_window_utc"]
    if (
        not isinstance(window, dict)
        or set(window) != {"started_at", "finished_at"}
        or any(
            type(window[name]) is not str
            or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", window[name])
            is None
            for name in window
        )
        or window["started_at"] > window["finished_at"]
    ):
        _fail(FAIL_BUILD, "Cargo bootstrap time window differs")
    failed = value["failed_preflight"]
    if (
        not isinstance(failed, dict)
        or failed.get("occurred") is not True
        or failed.get("download_started") is not False
        or type(failed.get("reason")) is not str
    ):
        _fail(FAIL_BUILD, "Cargo bootstrap failed-preflight disclosure differs")
    snapshot = value["first_verified_snapshot"]
    if (
        not isinstance(snapshot, dict)
        or set(snapshot)
        != {"domain", "root_hex", "package_count", "file_count"}
        or snapshot["domain"] != CARGO_SNAPSHOT_DOMAIN
        or snapshot["package_count"] != len(packages)
    ):
        _fail(FAIL_BUILD, "Cargo bootstrap snapshot identity differs")
    root = _hex32(
        snapshot["root_hex"],
        label="Cargo bootstrap snapshot root",
        code=FAIL_BUILD,
    )
    count = _exact_uint(
        snapshot["file_count"],
        label="Cargo bootstrap snapshot file count",
        code=FAIL_BUILD,
        minimum=1,
    )
    return root, count, hashlib.sha256(payload).digest()


def _lexical_absolute_path_v1(path: Path, *, label: str) -> tuple[Path, tuple[str, ...]]:
    """Return a normalized absolute path without resolving any filesystem link."""

    raw = os.fspath(path)
    if type(raw) is not str or not raw or "\x00" in raw:
        _fail(FAIL_BUILD, f"{label} is not a non-empty filesystem path")
    if not os.path.isabs(raw):
        raw = os.path.join(os.getcwd(), raw)
    raw_parts = raw.split(os.sep)
    if any(part in {".", ".."} for part in raw_parts):
        _fail(FAIL_BUILD, f"{label} contains a dot traversal component")
    components = tuple(part for part in raw_parts if part)
    if not components:
        _fail(FAIL_BUILD, f"{label} cannot be the filesystem root")
    return Path(os.sep).joinpath(*components), components


def _open_qualification_install_directory_v1(
    directory: Path,
    *,
    trusted_base: Path,
    create_missing: bool,
) -> tuple[int, Path]:
    """Open a qualification directory by holding every traversed directory fd.

    The walk starts at the filesystem root and uses only ``openat``/``mkdirat``
    operations relative to the previously opened directory.  Consequently an
    ancestor rename or symlink replacement can neither redirect the live walk
    nor be silently accepted by the final namespace replay.
    """

    _base_absolute, base_components = _lexical_absolute_path_v1(
        trusted_base,
        label="qualification trusted base",
    )
    directory_absolute, directory_components = _lexical_absolute_path_v1(
        directory,
        label="qualification install directory",
    )
    if (
        len(directory_components) < len(base_components)
        or directory_components[: len(base_components)] != base_components
    ):
        _fail(
            FAIL_BUILD,
            "qualification install directory is outside the trusted base",
        )
    relative_components = directory_components[len(base_components) :]
    if any(part in {"", ".", ".."} for part in relative_components):
        _fail(FAIL_BUILD, "qualification install directory traversal differs")

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        current_descriptor = os.open(os.sep, flags)
    except OSError as error:
        _fail(FAIL_BUILD, f"cannot open qualification filesystem root: {error}")
    try:
        for part in base_components:
            try:
                next_descriptor = os.open(
                    part,
                    flags,
                    dir_fd=current_descriptor,
                )
            except OSError as error:
                _fail(
                    FAIL_BUILD,
                    f"qualification trusted-base walk is unsafe: {error}",
                )
            os.close(current_descriptor)
            current_descriptor = next_descriptor

        base_metadata = os.fstat(current_descriptor)
        if (
            not stat.S_ISDIR(base_metadata.st_mode)
            or base_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(base_metadata.st_mode) & 0o022
        ):
            _fail(
                FAIL_BUILD,
                "qualification trusted base is foreign-owned or writable by "
                "another principal",
            )

        for part in relative_components:
            created = False
            try:
                next_descriptor = os.open(
                    part,
                    flags,
                    dir_fd=current_descriptor,
                )
            except FileNotFoundError:
                if not create_missing:
                    _fail(
                        FAIL_BUILD,
                        "qualification install directory disappeared during replay",
                    )
                try:
                    os.mkdir(part, 0o700, dir_fd=current_descriptor)
                    created = True
                except FileExistsError:
                    # A concurrent creator must still pass the no-follow open and
                    # owner/mode validation below.
                    pass
                except OSError as error:
                    _fail(
                        FAIL_BUILD,
                        f"cannot create qualification directory: {error}",
                    )
                if created:
                    os.fsync(current_descriptor)
                try:
                    next_descriptor = os.open(
                        part,
                        flags,
                        dir_fd=current_descriptor,
                    )
                except OSError as error:
                    _fail(
                        FAIL_BUILD,
                        f"qualification directory open after create is unsafe: {error}",
                    )
            except OSError as error:
                _fail(
                    FAIL_BUILD,
                    f"qualification directory walk is unsafe: {error}",
                )
            metadata = os.fstat(next_descriptor)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) & 0o022
            ):
                os.close(next_descriptor)
                _fail(
                    FAIL_BUILD,
                    "qualification directory is foreign-owned or writable by "
                    "another principal",
                )
            os.close(current_descriptor)
            current_descriptor = next_descriptor
        return current_descriptor, directory_absolute
    except BaseException:
        os.close(current_descriptor)
        raise


def _read_exact_descriptor_v1(
    descriptor: int,
    *,
    expected_size: int,
) -> bytes:
    chunks: list[bytes] = []
    remaining = expected_size
    while remaining:
        chunk = os.read(descriptor, min(remaining, 1024 * 1024))
        if not chunk:
            _fail(FAIL_BUILD, "qualified Rust binary read was short")
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        _fail(FAIL_BUILD, "qualified Rust binary grew during replay")
    return b"".join(chunks)


def _install_qualified_rust_binary_v1(
    built: Path,
    destination: Path,
    *,
    trusted_base: Path,
) -> Path:
    """Install once under a nonblocking per-commit lock and never overwrite."""

    try:
        built_metadata = built.lstat()
        built_payload = built.read_bytes()
    except OSError as error:
        _fail(FAIL_BUILD, f"built Rust binary is unreadable: {error}")
    if (
        stat.S_ISLNK(built_metadata.st_mode)
        or not stat.S_ISREG(built_metadata.st_mode)
        or not built_payload
        or len(built_payload) != built_metadata.st_size
        or len(built_payload) > 128 * 1024 * 1024
    ):
        _fail(FAIL_BUILD, "built Rust binary is not a bounded regular file")
    expected_digest = hashlib.sha256(built_payload).digest()
    destination_absolute, destination_components = _lexical_absolute_path_v1(
        destination,
        label="qualified Rust destination",
    )
    trusted_absolute, trusted_components = _lexical_absolute_path_v1(
        trusted_base,
        label="qualification trusted base",
    )
    if (
        len(destination_components) <= len(trusted_components)
        or destination_components[: len(trusted_components)] != trusted_components
        or destination_absolute.name in {"", ".", ".."}
    ):
        _fail(FAIL_BUILD, "qualified Rust destination is outside the trusted base")
    directory_descriptor, parent = _open_qualification_install_directory_v1(
        destination_absolute.parent,
        trusted_base=trusted_absolute,
        create_missing=True,
    )
    lock_descriptor: int | None = None
    temporary_descriptor: int | None = None
    temporary_name: str | None = None
    try:
        try:
            lock_descriptor = os.open(
                ".hegel-m3-qualification.lock",
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_descriptor,
            )
            lock_metadata = os.fstat(lock_descriptor)
            if (
                not stat.S_ISREG(lock_metadata.st_mode)
                or lock_metadata.st_uid != os.geteuid()
                or stat.S_IMODE(lock_metadata.st_mode) != 0o600
            ):
                _fail(FAIL_BUILD, "qualification lock identity or mode differs")
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as error:
            _fail(FAIL_BUILD, f"qualification install lock is unavailable: {error}")

        try:
            existing_descriptor = os.open(
                destination_absolute.name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptor,
            )
        except FileNotFoundError:
            existing_descriptor = None
        except OSError as error:
            _fail(FAIL_BUILD, f"qualified Rust destination is unsafe: {error}")
        if existing_descriptor is not None:
            try:
                metadata = os.fstat(existing_descriptor)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or metadata.st_size != len(built_payload)
                ):
                    _fail(FAIL_BUILD, "existing qualified Rust binary identity differs")
                existing_payload = _read_exact_descriptor_v1(
                    existing_descriptor,
                    expected_size=metadata.st_size,
                )
            finally:
                os.close(existing_descriptor)
            if hashlib.sha256(existing_payload).digest() != expected_digest:
                _fail(
                    FAIL_BUILD,
                    "existing commit-bound Rust binary differs; refusing overwrite",
                )
        else:
            temporary_name = (
                f".hegel-m3-enumerator-{secrets.token_hex(16)}.pending"
            )
            temporary_descriptor = os.open(
                temporary_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o555,
                dir_fd=directory_descriptor,
            )
            os.fchmod(temporary_descriptor, 0o555)
            view = memoryview(built_payload)
            offset = 0
            while offset < len(view):
                count = os.write(temporary_descriptor, view[offset:])
                if count <= 0:
                    _fail(FAIL_BUILD, "qualified Rust binary install write was short")
                offset += count
            os.fsync(temporary_descriptor)
            temporary_metadata = os.fstat(temporary_descriptor)
            if (
                not stat.S_ISREG(temporary_metadata.st_mode)
                or temporary_metadata.st_size != len(built_payload)
            ):
                _fail(FAIL_BUILD, "qualified Rust temporary identity differs")
            os.close(temporary_descriptor)
            temporary_descriptor = None
            try:
                os.link(
                    temporary_name,
                    destination_absolute.name,
                    src_dir_fd=directory_descriptor,
                    dst_dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError:
                _fail(FAIL_BUILD, "qualified Rust destination appeared concurrently")
            os.fsync(directory_descriptor)
            os.unlink(temporary_name, dir_fd=directory_descriptor)
            temporary_name = None
            os.fsync(directory_descriptor)

        installed_descriptor = os.open(
            destination_absolute.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
        try:
            installed_metadata = os.fstat(installed_descriptor)
            installed_payload = _read_exact_descriptor_v1(
                installed_descriptor,
                expected_size=installed_metadata.st_size,
            )
            if (
                not stat.S_ISREG(installed_metadata.st_mode)
                or installed_metadata.st_uid != os.geteuid()
                or stat.S_IMODE(installed_metadata.st_mode) != 0o555
                or hashlib.sha256(installed_payload).digest() != expected_digest
            ):
                _fail(FAIL_BUILD, "installed qualified Rust binary replay differs")

            replay_directory_descriptor, replay_parent = (
                _open_qualification_install_directory_v1(
                    destination_absolute.parent,
                    trusted_base=trusted_absolute,
                    create_missing=False,
                )
            )
            try:
                live_parent_metadata = os.fstat(directory_descriptor)
                replay_parent_metadata = os.fstat(replay_directory_descriptor)
                if (
                    replay_parent != parent
                    or (live_parent_metadata.st_dev, live_parent_metadata.st_ino)
                    != (replay_parent_metadata.st_dev, replay_parent_metadata.st_ino)
                ):
                    _fail(
                        FAIL_BUILD,
                        "qualification install directory identity changed",
                    )
                try:
                    path_descriptor = os.open(
                        destination_absolute.name,
                        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                        dir_fd=replay_directory_descriptor,
                    )
                except OSError as error:
                    _fail(
                        FAIL_BUILD,
                        f"installed qualified Rust path disappeared: {error}",
                    )
                try:
                    path_metadata = os.fstat(path_descriptor)
                finally:
                    os.close(path_descriptor)
                if (path_metadata.st_dev, path_metadata.st_ino) != (
                    installed_metadata.st_dev,
                    installed_metadata.st_ino,
                ):
                    _fail(FAIL_BUILD, "installed qualified Rust path identity changed")
            finally:
                os.close(replay_directory_descriptor)
        finally:
            os.close(installed_descriptor)
        return destination_absolute
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
                os.fsync(directory_descriptor)
            except FileNotFoundError:
                pass
        if lock_descriptor is not None:
            try:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            finally:
                os.close(lock_descriptor)
        os.close(directory_descriptor)


def _build_rust(
    control_plane: LocalDockerControlPlaneV1,
    image: str,
    source_root: Path,
    *,
    seccomp_path: Path,
    basis_commit: str,
    repository_root: Path,
) -> tuple[Path, bytes, bytes, bytes, int]:
    # Codex Desktop may export Windows TEMP into WSL; Rust build scripts cannot
    # execute from that drvfs mount.  The qualification target must live on the
    # local Linux filesystem and is never shared with another actor.
    try:
        target_owner = LinuxLocalTemporaryDirectoryV1(
            prefix="hegel-m3-rust-target-",
            repository_root=repository_root,
        )
    except Phase3LocalRuntimeError as error:
        _fail(FAIL_BUILD, str(error))
    with target_owner as raw_target:
        target = Path(raw_target)
        target.chmod(0o700)
        vendor = target / "vendor-snapshot"
        lock_payload = (
            source_root / "Hegel Machine/rust/m3_closure_enumerator/Cargo.lock"
        ).read_bytes()
        dependency_snapshot_root, dependency_snapshot_file_count = (
            _build_cargo_dependency_snapshot(lock_payload, vendor)
        )
        options = (
            "-v",
            f"{source_root}:/input:ro",
            "-v",
            f"{vendor}:/vendor:ro",
            "-v",
            f"{target}:/output:rw",
            "-w",
            "/input/Hegel Machine/rust/m3_closure_enumerator",
        )
        completed = _run(
            _docker_with_options(
                control_plane,
                image,
                options,
                (
                    f"{RUST_TOOLCHAIN_BIN}/cargo",
                    "--config",
                    'source.crates-io.replace-with="vendored-sources"',
                    "--config",
                    'source.vendored-sources.directory="/vendor"',
                    "build",
                    "--quiet",
                    "--release",
                    "--locked",
                    "--offline",
                    "--jobs=1",
                    "--manifest-path",
                    "Cargo.toml",
                ),
                seccomp_path=seccomp_path,
                container_environment=RUST_BUILD_ENVIRONMENT,
                user=f"{os.getuid()}:{os.getgid()}",
            ),
            code=FAIL_BUILD,
            timeout=300,
            environment=control_plane.environment,
        )
        if completed.stdout != b"" or completed.stderr != b"":
            _fail(
                FAIL_BUILD,
                "successful quiet offline Cargo build emitted output",
            )
        built = target / "release/hegel-m3-closure-enumerator"
        if not built.is_file():
            _fail(FAIL_BUILD, "offline Cargo build produced no release binary")
        trusted_base = PROJECT_ROOT / "rust/m3_closure_enumerator"
        destination = (
            trusted_base
            / "target/m3_qualification"
            / basis_commit
            / "hegel-m3-closure-enumerator"
        )
        installed = _install_qualified_rust_binary_v1(
            built,
            destination,
            trusted_base=trusted_base,
        )
        return (
            installed,
            completed.stdout,
            completed.stderr,
            dependency_snapshot_root,
            dependency_snapshot_file_count,
        )


def _run_python_enumerator(
    control_plane: LocalDockerControlPlaneV1,
    image: str,
    source_root: Path,
    output_parent: Path,
    roots: Sequence[bytes],
    *,
    seccomp_path: Path,
) -> subprocess.CompletedProcess[bytes]:
    output_parent.mkdir(mode=0o700, exist_ok=False)
    options = (
        "-v",
        f"{source_root}:/input:ro",
        "-v",
        f"{output_parent}:/output:rw",
        "-w",
        "/input/hegel_machine",
    )
    command = (
        "python3",
        "/input/hegel_machine/phase3_m3_isolated_entrypoint_v1.py",
        "--enumerate-prefix",
        "--child-dsl-spec-root",
        roots[0].hex(),
        "--operator-semantics-root",
        roots[1].hex(),
        "--identifier-registry-root",
        roots[2].hex(),
        "--output-directory",
        "/output/archive",
    )
    return _run(
        _docker_with_options(
            control_plane,
            image,
            options,
            command,
            seccomp_path=seccomp_path,
            container_environment=PYTHON_RUNTIME_ENVIRONMENT,
            user=f"{os.getuid()}:{os.getgid()}",
        ),
        code=FAIL_EXECUTION,
        timeout=300,
        environment=control_plane.environment,
    )


def _run_rust_enumerator(
    control_plane: LocalDockerControlPlaneV1,
    image: str,
    binary_path: Path,
    output_parent: Path,
    roots: Sequence[bytes],
    *,
    seccomp_path: Path,
) -> subprocess.CompletedProcess[bytes]:
    output_parent.mkdir(mode=0o700, exist_ok=False)
    options = (
        "-v",
        f"{binary_path}:/input/enumerator:ro",
        "-v",
        f"{output_parent}:/output:rw",
    )
    command = (
        "/input/enumerator",
        "--enumerate-prefix",
        "--child-dsl-spec-root",
        roots[0].hex(),
        "--operator-semantics-root",
        roots[1].hex(),
        "--identifier-registry-root",
        roots[2].hex(),
        "--output-directory",
        "/output/archive",
    )
    return _run(
        _docker_with_options(
            control_plane,
            image,
            options,
            command,
            seccomp_path=seccomp_path,
            container_environment=RUST_RUNTIME_ENVIRONMENT,
            user=f"{os.getuid()}:{os.getgid()}",
        ),
        code=FAIL_EXECUTION,
        timeout=180,
        environment=control_plane.environment,
    )


_ARCHIVE_FILES: Final = frozenset(
    {
        "report.json",
        "canonical_program_records.cborframed",
        "program_chunk_manifests.cborframed",
        "bucket_accounting_records.cborframed",
    }
)
_OUTPUT_SORT_IDS: Final = {
    "Bool": 1,
    "Bit": 2,
    "Sign": 3,
    "BoundedInt": 4,
    "RationalValue": 5,
}


def _read_regular_file(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        _fail(FAIL_REPORT, f"cannot stat {label}: {error}")
    if path.is_symlink() or not path.is_file() or metadata.st_size > maximum_bytes:
        _fail(FAIL_REPORT, f"{label} is not a bounded regular file")
    try:
        payload = path.read_bytes()
    except OSError as error:
        _fail(FAIL_REPORT, f"cannot read {label}: {error}")
    if len(payload) != metadata.st_size:
        _fail(FAIL_REPORT, f"{label} changed while being read")
    return payload


def _decode_framed_stream(
    payload: bytes, *, expected_count: int, label: str
) -> tuple[bytes, ...]:
    records: list[bytes] = []
    offset = 0
    while offset < len(payload):
        if len(payload) - offset < 4:
            _fail(FAIL_REPORT, f"{label} has a truncated uint32 frame header")
        length = int.from_bytes(payload[offset : offset + 4], "big")
        offset += 4
        if length < 1 or length > 1_048_576 or offset + length > len(payload):
            _fail(FAIL_REPORT, f"{label} has an invalid frame length")
        records.append(payload[offset : offset + length])
        offset += length
        if len(records) > expected_count:
            _fail(FAIL_REPORT, f"{label} has too many records")
    if offset != len(payload) or len(records) != expected_count:
        _fail(
            FAIL_REPORT,
            f"{label} count differs: expected {expected_count}, got {len(records)}",
        )
    return tuple(records)


def _uint_field(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        _fail(FAIL_REPORT, f"{label} must be an unsigned integer")
    return value


def _host_validate_enumerator_archive_v1(
    output_parent: Path,
    *,
    implementation: str,
    stdout_report: Mapping[str, object],
    roots: Sequence[bytes],
) -> Mapping[str, object]:
    """Strictly replay every emitted record outside the enumerator process."""

    directory = output_parent / "archive"
    try:
        entries = tuple(directory.iterdir())
    except OSError as error:
        _fail(FAIL_REPORT, f"cannot inspect {implementation} archive: {error}")
    if directory.is_symlink() or not directory.is_dir() or {item.name for item in entries} != _ARCHIVE_FILES:
        _fail(FAIL_REPORT, f"{implementation} archive file set differs")
    if any(item.is_symlink() or not item.is_file() for item in entries):
        _fail(FAIL_REPORT, f"{implementation} archive contains a non-regular file")

    report_payload = _read_regular_file(
        directory / "report.json",
        maximum_bytes=1_048_576,
        label=f"{implementation} archive report",
    )
    disk_report = _parse_single_json(
        report_payload, label=f"{implementation} archive report"
    )
    if dict(disk_report) != dict(stdout_report):
        _fail(FAIL_REPORT, f"{implementation} disk/stdout reports disagree")

    stream_specs = (
        (
            "canonical_program_records",
            "canonical_program_records.cborframed",
            _uint_field(stdout_report["program_record_count"], label="program_record_count"),
            "CanonicalProgramRecordV2",
        ),
        (
            "program_chunk_manifests",
            "program_chunk_manifests.cborframed",
            _uint_field(stdout_report["chunk_manifest_count"], label="chunk_manifest_count"),
            "ProgramChunkManifestV2",
        ),
        (
            "bucket_accounting_records",
            "bucket_accounting_records.cborframed",
            _uint_field(stdout_report["bucket_record_count"], label="bucket_record_count"),
            "BucketAccountingRecordV1",
        ),
    )
    streams: dict[str, bytes] = {}
    framed: dict[str, tuple[bytes, ...]] = {}
    values: dict[str, tuple[tuple[object, ...], ...]] = {}
    fields: dict[str, tuple[Mapping[str, object], ...]] = {}
    for name, filename, count, schema in stream_specs:
        payload = _read_regular_file(
            directory / filename,
            maximum_bytes=64 * 1024 * 1024,
            label=f"{implementation} {filename}",
        )
        records = _decode_framed_stream(
            payload, expected_count=count, label=f"{implementation} {filename}"
        )
        decoded = []
        for index, record in enumerate(records):
            try:
                item = decode_formal_object(record, expected_name=schema)
            except (TypeError, ValueError) as error:
                _fail(FAIL_REPORT, f"{implementation} {schema}[{index}] is invalid: {error}")
            decoded.append(item)
        streams[name] = payload
        framed[name] = records
        values[name] = tuple(item.value for item in decoded)
        fields[name] = tuple(item.fields for item in decoded)  # type: ignore[assignment]

    programs = fields["canonical_program_records"]
    program_values = values["canonical_program_records"]
    expected_program_count = _uint_field(
        stdout_report["canonical_program_count"], label="canonical_program_count"
    )
    if len(programs) != expected_program_count:
        _fail(FAIL_REPORT, f"{implementation} canonical program count differs")
    binding_names = (
        "child_dsl_spec_root",
        "operator_semantics_root",
        "identifier_registry_root",
    )
    previous_key: tuple[int, int, int, int, bytes] | None = None
    program_structural_keys: list[tuple[int, int, int]] = []
    for index, row in enumerate(programs):
        if row["program_index"] != index:
            _fail(FAIL_REPORT, f"{implementation} program indices are not contiguous")
        ast_payload = row["canonical_ast_cbor_bytes"]
        if type(ast_payload) is not bytes:
            _fail(FAIL_REPORT, f"{implementation} program AST payload is not bytes")
        try:
            ast = decode_shrink1_canonical_ast(ast_payload)
        except (TypeError, ValueError) as error:
            _fail(FAIL_REPORT, f"{implementation} program AST {index} is invalid: {error}")
        try:
            output_sort_id = _OUTPUT_SORT_IDS[ast.metrics.output_sort]
        except KeyError:
            _fail(FAIL_REPORT, f"{implementation} program AST {index} has unknown sort")
        if (
            row["canonical_ast_hash"] != ast.digest
            or row["output_sort_id"] != output_sort_id
            or row["ast_depth"] != ast.metrics.depth
            or row["ast_node_count"] != ast.metrics.node_count
            or row["distinct_bit_slot_count"] != len(ast.metrics.distinct_bit_slots)
        ):
            _fail(FAIL_REPORT, f"{implementation} program AST metadata differs at {index}")
        if any(row[name] != root for name, root in zip(binding_names, roots, strict=True)):
            _fail(FAIL_REPORT, f"{implementation} program binding roots differ at {index}")
        key = (
            ast.metrics.depth,
            ast.metrics.node_count,
            output_sort_id,
            ast.root_operator_id,
            ast.cbor_bytes,
        )
        if previous_key is not None and key <= previous_key:
            _fail(FAIL_REPORT, f"{implementation} program traversal order differs at {index}")
        previous_key = key
        program_structural_keys.append(
            (output_sort_id, ast.metrics.depth, ast.metrics.node_count)
        )
    if rfc6962_root(list(program_values)).hex() != stdout_report["canonical_program_archive_root_or_null"]:
        _fail(FAIL_REPORT, f"{implementation} program archive root does not replay")

    witness_hex = stdout_report["first_out_of_budget_program_cbor_hex_or_null"]
    witness_hash = stdout_report["first_out_of_budget_program_hash_or_null"]
    if type(witness_hex) is not str or type(witness_hash) is not str or previous_key is None:
        _fail(FAIL_REPORT, f"{implementation} DSL_TOO_LARGE witness is absent")
    try:
        witness_payload = bytes.fromhex(witness_hex)
        witness = decode_shrink1_canonical_ast(witness_payload)
    except (ValueError, TypeError) as error:
        _fail(FAIL_REPORT, f"{implementation} witness is invalid: {error}")
    if witness.digest.hex() != witness_hash:
        _fail(FAIL_REPORT, f"{implementation} witness hash differs")
    witness_key = (
        witness.metrics.depth,
        witness.metrics.node_count,
        _OUTPUT_SORT_IDS[witness.metrics.output_sort],
        witness.root_operator_id,
        witness.cbor_bytes,
    )
    if witness_key <= previous_key:
        _fail(FAIL_REPORT, f"{implementation} witness is not after archived index 49,999")

    chunks = fields["program_chunk_manifests"]
    chunk_values = values["program_chunk_manifests"]
    records_per_chunk = _uint_field(
        stdout_report["records_per_chunk"], label="records_per_chunk"
    )
    for chunk_index, row in enumerate(chunks):
        first = chunk_index * records_per_chunk
        subset_payloads = framed["canonical_program_records"][first : first + records_per_chunk]
        subset_values = program_values[first : first + records_per_chunk]
        if not subset_payloads:
            _fail(FAIL_REPORT, f"{implementation} emitted an empty chunk")
        blob = b"".join(len(item).to_bytes(4, "big") + item for item in subset_payloads)
        if (
            row["chunk_index"] != chunk_index
            or row["first_program_index"] != first
            or row["last_program_index"] != first + len(subset_payloads) - 1
            or row["record_count"] != len(subset_payloads)
            or row["canonical_program_record_subtree_root"]
            != rfc6962_root(list(subset_values))
            or row["compressed_program_blob_hash"]
            != hashlib.sha256(CHUNK_BLOB_DOMAIN + b"\x00" + blob).digest()
            or row["uncompressed_program_byte_length"] != len(blob)
        ):
            _fail(FAIL_REPORT, f"{implementation} chunk manifest {chunk_index} does not replay")
    if rfc6962_root(list(chunk_values)).hex() != stdout_report["program_chunk_manifest_root_or_null"]:
        _fail(FAIL_REPORT, f"{implementation} chunk manifest root does not replay")

    buckets = fields["bucket_accounting_records"]
    bucket_values = values["bucket_accounting_records"]
    expected_bucket_keys = tuple(
        (sort_id, depth, nodes)
        for sort_id in range(1, 6)
        for depth in range(5)
        for nodes in range(1, 8)
    )
    if len(buckets) != len(expected_bucket_keys):
        _fail(FAIL_REPORT, f"{implementation} bucket registry width differs")
    indices_by_key: dict[tuple[int, int, int], list[int]] = {
        key: [] for key in expected_bucket_keys
    }
    for index, key in enumerate(program_structural_keys):
        indices_by_key[key].append(index)
    residual_buckets: list[tuple[int, int]] = []
    raw_total = 0
    accepted_total = 0
    for bucket_index, (row, expected_key) in enumerate(
        zip(buckets, expected_bucket_keys, strict=True)
    ):
        if (
            row["bucket_index"] != bucket_index
            or (row["output_sort_id"], row["ast_depth"], row["ast_node_count"])
            != expected_key
        ):
            _fail(FAIL_REPORT, f"{implementation} bucket identity differs at {bucket_index}")
        indices = indices_by_key[expected_key]
        expected_first = indices[0] if indices else None
        expected_last = indices[-1] if indices else None
        if (
            row["accepted_canonical_programs"] != len(indices)
            or row["first_program_index_or_null"] != expected_first
            or row["last_program_index_or_null"] != expected_last
        ):
            _fail(FAIL_REPORT, f"{implementation} bucket archive indices differ at {bucket_index}")
        counters = tuple(
            _uint_field(row[name], label=f"bucket[{bucket_index}].{name}")
            for name in (
                "raw_operator_applications",
                "accepted_canonical_programs",
                "syntactic_duplicates",
                "type_rejections",
                "structural_limit_rejections",
                "rewrite_collapses",
            )
        )
        raw, accepted, duplicate, type_reject, structural, rewrite = counters
        residual = raw - accepted - duplicate - type_reject - structural - rewrite
        if residual < 0:
            _fail(FAIL_REPORT, f"{implementation} bucket counters over-partition raw count")
        if residual:
            residual_buckets.append((bucket_index, residual))
        raw_total += raw
        accepted_total += accepted
    witness_structural_key = (
        _OUTPUT_SORT_IDS[witness.metrics.output_sort],
        witness.metrics.depth,
        witness.metrics.node_count,
    )
    witness_bucket_index = expected_bucket_keys.index(witness_structural_key)
    if len(residual_buckets) != 1 or residual_buckets[0][0] != witness_bucket_index:
        _fail(FAIL_REPORT, f"{implementation} out-of-budget bucket accounting differs")
    if raw_total != stdout_report["raw_operator_application_count"] or accepted_total != len(programs):
        _fail(FAIL_REPORT, f"{implementation} bucket totals do not replay")
    if rfc6962_root(list(bucket_values)).hex() != stdout_report["bucket_accounting_root_or_null"]:
        _fail(FAIL_REPORT, f"{implementation} bucket accounting root does not replay")

    return MappingProxyType(
        {
            "streams": MappingProxyType(streams),
            "stream_digests": MappingProxyType(
                {name: hashlib.sha256(payload).digest() for name, payload in streams.items()}
            ),
            "report_payload": report_payload,
            "witness_adjacency_verified": True,
            "residual_out_of_budget_canonical_programs": residual_buckets[0][1],
        }
    )


def _validate_dual_archive_bytes_equal_v1(
    python_archive: Mapping[str, object], rust_archive: Mapping[str, object]
) -> None:
    python_streams = python_archive.get("streams")
    rust_streams = rust_archive.get("streams")
    if not isinstance(python_streams, Mapping) or not isinstance(rust_streams, Mapping):
        _fail(FAIL_REPORT, "host archive replay result is malformed")
    if set(python_streams) != set(rust_streams) or any(
        python_streams[name] != rust_streams[name] for name in python_streams
    ):
        _fail(FAIL_REPORT, "Python and Rust formal archive bytes disagree")
    if (
        python_archive.get("witness_adjacency_verified") is not True
        or rust_archive.get("witness_adjacency_verified") is not True
    ):
        _fail(FAIL_REPORT, "dual witness adjacency was not verified")


def validate_enumerator_report_v1(
    report: object,
    *,
    implementation: str,
    golden: Mapping[str, object],
) -> Mapping[str, object]:
    if not isinstance(report, dict) or set(report) != REPORT_FIELDS:
        actual = set(report) if isinstance(report, dict) else {type(report).__name__}
        _fail(FAIL_REPORT, f"{implementation} report fields differ: {sorted(actual ^ REPORT_FIELDS)}")
    expected_identity = {
        "python": (
            "hegel-m3-python-closure-enumerator-report/1",
            1,
            "hegel-python-m3-bounded-closure-enumerator-v1",
        ),
        "rust": (
            "hegel-m3-rust-closure-enumerator-report/1",
            2,
            "hegel-rust-m3-bounded-closure-enumerator-v1",
        ),
    }[implementation]
    if (
        report["schema_version"],
        report["implementation_id"],
        report["implementation_machine_id"],
    ) != expected_identity or report["implementation"] != implementation:
        _fail(FAIL_REPORT, f"{implementation} implementation identity differs")
    if report["claim_level"] != "FORMAL_PROFILE_CANDIDATE_NOT_AUTHORITY" or report["authoritative_claim_allowed"] is not False:
        _fail(FAIL_REPORT, f"{implementation} bare report attempts authority")
    for field in (
        "target_roles_evaluated",
        "split_material_accessed",
        "secrets_accessed",
        "raw_expansion_limit_hit",
        "wall_clock_abort_hit",
    ):
        if report[field] is not False:
            _fail(FAIL_REPORT, f"{implementation} report violates {field}=false")
    bindings = golden["binding_roots"]
    assert isinstance(bindings, dict)
    for field in GOLDEN_BINDING_FIELDS:
        if report[field] != bindings[field]:
            _fail(FAIL_REPORT, f"{implementation} report binding {field} differs")
    expected = golden["expected"]
    assert isinstance(expected, dict)
    report_field = {
        "bucket_accounting_root": "bucket_accounting_root_or_null",
        "canonical_program_archive_root": "canonical_program_archive_root_or_null",
        "first_out_of_budget_program_cbor_hex": "first_out_of_budget_program_cbor_hex_or_null",
        "first_out_of_budget_program_hash": "first_out_of_budget_program_hash_or_null",
        "program_chunk_manifest_root": "program_chunk_manifest_root_or_null",
    }
    for key in GOLDEN_EXPECTED_ORDER:
        if report[report_field.get(key, key)] != expected[key]:
            _fail(FAIL_REPORT, f"{implementation} result differs at {key}")
    return MappingProxyType(dict(report))


def _environment_fields(image: str, runtime: str, dependency_lock_root: bytes) -> dict[str, object]:
    return {
        "os_id_digest": id_digest_v1("os:linux-oci"),
        "architecture_id_digest": id_digest_v1("architecture:x86_64"),
        "runtime_id_digest": id_digest_v1("runtime:" + runtime),
        "runtime_version_id_digest": id_digest_v1(
            "oci-manifest:" + image.rsplit(":", 1)[1]
        ),
        "dependency_lock_root": dependency_lock_root,
        "locale_id_digest": id_digest_v1("locale:C.UTF-8"),
        "timezone_id_digest": id_digest_v1("timezone:UTC"),
        # This is the enumerator runtime profile, not a Purpose 1--4 formal
        # actor identity and not the distinct non-authoritative build profile.
        "container_or_host_profile_id_digest": id_digest_v1(
            RUNTIME_DOCKER_POLICY_ID
        ),
        "oci_manifest_digest_or_null": bytes.fromhex(image.rsplit(":", 1)[1]),
    }


def _implementation_binding_fields(
    *,
    implementation_id: int,
    source_root: bytes,
    binary_digest: bytes,
    environment_root: bytes,
    version_digest: bytes,
    dependency_lock_root: bytes,
    entrypoint: str,
    golden_root: bytes,
    commit_wire: tuple[int, bytes],
) -> dict[str, object]:
    runtime = "cpython" if implementation_id == 1 else "rustc"
    fields = {
        "implementation_id": implementation_id,
        "source_root": source_root,
        "binary_digest": binary_digest,
        "execution_environment_spec_root": environment_root,
        "compiler_or_interpreter_id_digest": id_digest_v1("runtime:" + runtime),
        "compiler_or_interpreter_version_digest": version_digest,
        "dependency_lock_root": dependency_lock_root,
        "build_profile_id_digest": id_digest_v1(
            "hegel-m3-python-direct-source-no-build-v1"
            if implementation_id == 1
            else RUST_BUILD_DOCKER_POLICY_ID
        ),
        "entrypoint_id_digest": id_digest_v1(entrypoint),
        "golden_vector_root": golden_root,
        "repository_commit_id": commit_wire,
    }
    build_formal_object("ImplementationBindingV1", fields)
    return fields


def _receipt_implementation_value(value: Mapping[str, object]) -> tuple[object, ...]:
    return (
        value["implementation_id"],
        str(value["implementation_machine_id"]).encode("ascii"),
        _hex32(value["source_root"], label="receipt source root", code=FAIL_RECEIPT),
        value["source_file_count"],
        _hex32(value["dependency_lock_root"], label="receipt lock root", code=FAIL_RECEIPT),
        None
        if value["dependency_snapshot_root_or_null"] is None
        else _hex32(
            value["dependency_snapshot_root_or_null"],
            label="receipt dependency snapshot root",
            code=FAIL_RECEIPT,
        ),
        value["dependency_snapshot_file_count"],
        _hex32(value["execution_environment_spec_root"], label="receipt environment root", code=FAIL_RECEIPT),
        str(value["image_ref"]).encode("ascii"),
        str(value["bound_executable_locator"]).encode("utf-8"),
        _hex32(value["binary_digest"], label="receipt binary digest", code=FAIL_RECEIPT),
        _hex32(value["compiler_or_interpreter_version_digest"], label="receipt version digest", code=FAIL_RECEIPT),
        _hex32(value["entrypoint_id_digest"], label="receipt entrypoint digest", code=FAIL_RECEIPT),
        _hex32(value["implementation_binding_root"], label="receipt binding root", code=FAIL_RECEIPT),
        _hex32(value["canonical_report_sha256"], label="receipt report digest", code=FAIL_RECEIPT),
        _hex32(value["execution_stdout_sha256"], label="receipt stdout digest", code=FAIL_RECEIPT),
        _hex32(
            value["runtime_container_environment_sha256"],
            label="receipt runtime container environment digest",
            code=FAIL_RECEIPT,
        ),
        None
        if value["build_container_environment_sha256_or_null"] is None
        else _hex32(
            value["build_container_environment_sha256_or_null"],
            label="receipt build container environment digest",
            code=FAIL_RECEIPT,
        ),
        _hex32(
            value["canonical_program_records_stream_sha256"],
            label="receipt program stream digest",
            code=FAIL_RECEIPT,
        ),
        _hex32(
            value["program_chunk_manifests_stream_sha256"],
            label="receipt chunk stream digest",
            code=FAIL_RECEIPT,
        ),
        _hex32(
            value["bucket_accounting_records_stream_sha256"],
            label="receipt bucket stream digest",
            code=FAIL_RECEIPT,
        ),
        None if value["build_stdout_sha256_or_null"] is None else _hex32(value["build_stdout_sha256_or_null"], label="receipt build stdout", code=FAIL_RECEIPT),
        None if value["build_stderr_sha256_or_null"] is None else _hex32(value["build_stderr_sha256_or_null"], label="receipt build stderr", code=FAIL_RECEIPT),
        value["input_snapshot_target_free"],
        value["archive_file_set_verified"],
        value["host_strict_archive_replay_verified"],
        value["witness_adjacency_verified"],
    )


def _receipt_value(receipt: Mapping[str, object]) -> tuple[object, ...]:
    agreement = receipt["agreement"]
    assert isinstance(agreement, dict)
    return (
        1,
        SCHEMA.encode("ascii"),
        CLAIM_LEVEL.encode("ascii"),
        False,
        (1, bytes.fromhex(str(receipt["basis_commit"]))),
        _hex32(receipt["golden_vector_root"], label="receipt golden root", code=FAIL_RECEIPT),
        _receipt_implementation_value(receipt["python"]),  # type: ignore[arg-type]
        _receipt_implementation_value(receipt["rust"]),  # type: ignore[arg-type]
        tuple(_formalize_golden_value(name, agreement[name]) for name in GOLDEN_EXPECTED_ORDER),
        str(receipt["docker_policy_id"]).encode("ascii"),
        str(receipt["runtime_docker_policy_id"]).encode("ascii"),
        str(receipt["rust_build_docker_policy_id"]).encode("ascii"),
        _hex32(receipt["runtime_seccomp_sha256"], label="receipt runtime seccomp", code=FAIL_RECEIPT),
        _hex32(receipt["build_seccomp_sha256"], label="receipt build seccomp", code=FAIL_RECEIPT),
        _hex32(
            receipt["local_docker_daemon_receipt_binding"],
            label="receipt local Docker daemon binding",
            code=FAIL_RECEIPT,
        ),
        _hex32(
            receipt["cargo_offline_bootstrap_record_sha256"],
            label="receipt Cargo bootstrap record digest",
            code=FAIL_RECEIPT,
        ),
        receipt["pull_policy_never"],
        receipt["network_mode_none"],
        receipt["independent_source_snapshots"],
        receipt["independent_archive_bytes_equal"],
        receipt["target_inputs_visible"],
        receipt["target_roles_evaluated"],
        receipt["split_material_accessed"],
        receipt["secrets_accessed"],
        receipt["formal_m3_output_roots_generated"],
        str(receipt["m3_state"]).encode("ascii"),
    )


def validate_qualification_receipt_v1(
    receipt: object,
    *,
    golden: Mapping[str, object],
    basis_commit: str,
) -> bytes:
    checked = _exact_fields(receipt, RECEIPT_FIELDS, label="receipt")
    python = _exact_fields(checked["python"], IMPLEMENTATION_RECEIPT_FIELDS, label="receipt.python")
    rust = _exact_fields(checked["rust"], IMPLEMENTATION_RECEIPT_FIELDS, label="receipt.rust")
    agreement = _exact_fields(checked["agreement"], GOLDEN_EXPECTED_FIELDS, label="receipt.agreement")
    if checked["schema_version"] != SCHEMA or checked["claim_level"] != CLAIM_LEVEL:
        _fail(FAIL_RECEIPT, "qualification receipt identity differs")
    if checked["authoritative_claim_allowed"] is not False:
        _fail(FAIL_RECEIPT, "qualification receipt attempts enumeration authority")
    if (
        type(basis_commit) is not str
        or re.fullmatch(r"[0-9a-f]{40}", basis_commit) is None
        or checked["basis_commit"] != basis_commit
    ):
        _fail(FAIL_RECEIPT, "qualification receipt commit differs")
    try:
        _, _, expected_golden_root = validate_dual_golden_v1(dict(golden))
    except (TypeError, ValueError) as error:
        _fail(FAIL_RECEIPT, f"qualification golden is invalid: {error}")
    if _hex32(
        checked["golden_vector_root"],
        label="receipt golden root",
        code=FAIL_RECEIPT,
    ) != expected_golden_root:
        _fail(FAIL_RECEIPT, "qualification golden root differs")
    if dict(agreement) != golden["expected"]:
        _fail(FAIL_RECEIPT, "qualification agreement differs from committed golden")
    expected_implementations = (
        (
            python,
            1,
            "hegel-python-m3-bounded-closure-enumerator-v1",
            len(PYTHON_SOURCE_PATHS),
            PYTHON_ENTRYPOINT_ID,
        ),
        (
            rust,
            2,
            "hegel-rust-m3-bounded-closure-enumerator-v1",
            len(RUST_SOURCE_PATHS),
            RUST_ENTRYPOINT_ID,
        ),
    )
    for implementation, identity, machine_id, source_count, entrypoint in expected_implementations:
        if implementation["implementation_id"] != identity:
            _fail(FAIL_RECEIPT, "qualification implementation IDs differ")
        if implementation["implementation_machine_id"] != machine_id:
            _fail(FAIL_RECEIPT, "qualification implementation machine ID differs")
        if _exact_uint(
            implementation["source_file_count"],
            label="receipt source_file_count",
            code=FAIL_RECEIPT,
        ) != source_count:
            _fail(FAIL_RECEIPT, "qualification source file count differs")
        for name in (
            "source_root",
            "dependency_lock_root",
            "execution_environment_spec_root",
            "binary_digest",
            "compiler_or_interpreter_version_digest",
            "implementation_binding_root",
            "canonical_report_sha256",
            "execution_stdout_sha256",
            "runtime_container_environment_sha256",
            "canonical_program_records_stream_sha256",
            "program_chunk_manifests_stream_sha256",
            "bucket_accounting_records_stream_sha256",
        ):
            _hex32(implementation[name], label=f"receipt {name}", code=FAIL_RECEIPT)
        if _hex32(
            implementation["entrypoint_id_digest"],
            label="receipt entrypoint digest",
            code=FAIL_RECEIPT,
        ) != id_digest_v1(entrypoint):
            _fail(FAIL_RECEIPT, "qualification entrypoint digest differs")
        image = _receipt_text(implementation["image_ref"], label="receipt image_ref")
        if re.fullmatch(r"[a-z0-9._/-]+@sha256:[0-9a-f]{64}", image) is None:
            _fail(FAIL_RECEIPT, "qualification image is not digest-pinned")
        locator = _receipt_text(
            implementation["bound_executable_locator"],
            label="receipt executable locator",
        )
        expected_runtime_environment = (
            PYTHON_RUNTIME_ENVIRONMENT if identity == 1 else RUST_RUNTIME_ENVIRONMENT
        )
        if _hex32(
            implementation["runtime_container_environment_sha256"],
            label="receipt runtime container environment digest",
            code=FAIL_RECEIPT,
        ) != _container_environment_digest(expected_runtime_environment):
            _fail(FAIL_RECEIPT, "qualification runtime container environment differs")
        if identity == 1:
            if (
                implementation["dependency_snapshot_root_or_null"] is not None
                or implementation["dependency_snapshot_file_count"] != 0
                or implementation["build_stdout_sha256_or_null"] is not None
                or implementation["build_stderr_sha256_or_null"] is not None
                or implementation["build_container_environment_sha256_or_null"]
                is not None
                or re.fullmatch(
                    re.escape(f"oci://{image}") + r"/usr/local/bin/python[0-9.]*",
                    locator,
                )
                is None
            ):
                _fail(FAIL_RECEIPT, "Python qualification provenance differs")
        else:
            _hex32(
                implementation["dependency_snapshot_root_or_null"],
                label="receipt Rust dependency snapshot root",
                code=FAIL_RECEIPT,
            )
            _exact_uint(
                implementation["dependency_snapshot_file_count"],
                label="receipt Rust dependency snapshot file count",
                code=FAIL_RECEIPT,
                minimum=1,
            )
            build_stdout_digest = _hex32(
                implementation["build_stdout_sha256_or_null"],
                label="receipt Rust build stdout",
                code=FAIL_RECEIPT,
            )
            build_stderr_digest = _hex32(
                implementation["build_stderr_sha256_or_null"],
                label="receipt Rust build stderr",
                code=FAIL_RECEIPT,
            )
            if (
                build_stdout_digest != EMPTY_BUILD_STREAM_SHA256
                or build_stderr_digest != EMPTY_BUILD_STREAM_SHA256
            ):
                _fail(
                    FAIL_RECEIPT,
                    "Rust quiet-build success streams are not canonically empty",
                )
            if _hex32(
                implementation["build_container_environment_sha256_or_null"],
                label="receipt Rust build container environment digest",
                code=FAIL_RECEIPT,
            ) != _container_environment_digest(RUST_BUILD_ENVIRONMENT):
                _fail(FAIL_RECEIPT, "Rust build container environment differs")
            expected_locator = (
                "generated-target://rust/m3_closure_enumerator/target/"
                f"m3_qualification/{basis_commit}/hegel-m3-closure-enumerator"
            )
            if locator != expected_locator:
                _fail(FAIL_RECEIPT, "Rust qualification executable locator differs")
        for flag in (
            "input_snapshot_target_free",
            "archive_file_set_verified",
            "host_strict_archive_replay_verified",
            "witness_adjacency_verified",
        ):
            if implementation[flag] is not True:
                _fail(FAIL_RECEIPT, f"qualification implementation flag {flag} differs")
    required_flags = {
        "pull_policy_never": True,
        "network_mode_none": True,
        "independent_source_snapshots": True,
        "independent_archive_bytes_equal": True,
        "target_inputs_visible": False,
        "target_roles_evaluated": False,
        "split_material_accessed": False,
        "secrets_accessed": False,
        "formal_m3_output_roots_generated": False,
    }
    if (
        checked["docker_policy_id"] != DOCKER_POLICY_ID
        or checked["runtime_docker_policy_id"] != RUNTIME_DOCKER_POLICY_ID
        or checked["rust_build_docker_policy_id"] != RUST_BUILD_DOCKER_POLICY_ID
        or checked["m3_state"] != "NOT_RUN"
    ):
        _fail(FAIL_RECEIPT, "qualification Docker/state identity differs")
    _hex32(checked["runtime_seccomp_sha256"], label="receipt runtime seccomp", code=FAIL_RECEIPT)
    _hex32(checked["build_seccomp_sha256"], label="receipt build seccomp", code=FAIL_RECEIPT)
    _hex32(
        checked["local_docker_daemon_receipt_binding"],
        label="receipt local Docker daemon binding",
        code=FAIL_RECEIPT,
    )
    _hex32(
        checked["cargo_offline_bootstrap_record_sha256"],
        label="receipt Cargo bootstrap record digest",
        code=FAIL_RECEIPT,
    )
    for field, expected in required_flags.items():
        if checked[field] is not expected:
            _fail(FAIL_RECEIPT, f"qualification receipt flag {field} differs")
    try:
        preimage = canonical_cbor_encode(_receipt_value(checked))
    except (TypeError, ValueError) as error:
        _fail(FAIL_RECEIPT, f"qualification receipt typed value is invalid: {error}")
    if (
        type(checked["receipt_cbor_hex"]) is not str
        or checked["receipt_cbor_hex"] != preimage.hex()
    ):
        _fail(FAIL_RECEIPT, "qualification receipt CBOR differs")
    root = content_hash(RECEIPT_DOMAIN, _receipt_value(checked))
    if type(checked["receipt_root"]) is not str or checked["receipt_root"] != root.hex():
        _fail(FAIL_RECEIPT, "qualification receipt root differs")
    return root


def _build_receipt(
    *,
    basis_commit: str,
    golden: Mapping[str, object],
    golden_root: bytes,
    python_fields: Mapping[str, object],
    rust_fields: Mapping[str, object],
    runtime_seccomp_digest: bytes,
    build_seccomp_digest: bytes,
    docker_daemon_receipt_binding: bytes,
    cargo_bootstrap_record_digest: bytes,
) -> Mapping[str, object]:
    receipt: dict[str, object] = {
        "schema_version": SCHEMA,
        "claim_level": CLAIM_LEVEL,
        "authoritative_claim_allowed": False,
        "basis_commit": basis_commit,
        "golden_vector_root": golden_root.hex(),
        "python": dict(python_fields),
        "rust": dict(rust_fields),
        "agreement": dict(golden["expected"]),  # type: ignore[arg-type]
        "docker_policy_id": DOCKER_POLICY_ID,
        "runtime_docker_policy_id": RUNTIME_DOCKER_POLICY_ID,
        "rust_build_docker_policy_id": RUST_BUILD_DOCKER_POLICY_ID,
        "runtime_seccomp_sha256": runtime_seccomp_digest.hex(),
        "build_seccomp_sha256": build_seccomp_digest.hex(),
        "local_docker_daemon_receipt_binding": docker_daemon_receipt_binding.hex(),
        "cargo_offline_bootstrap_record_sha256": cargo_bootstrap_record_digest.hex(),
        "pull_policy_never": True,
        "network_mode_none": True,
        "independent_source_snapshots": True,
        "independent_archive_bytes_equal": True,
        "target_inputs_visible": False,
        "target_roles_evaluated": False,
        "split_material_accessed": False,
        "secrets_accessed": False,
        "formal_m3_output_roots_generated": False,
        "m3_state": "NOT_RUN",
    }
    preimage = canonical_cbor_encode(_receipt_value(receipt))
    receipt["receipt_cbor_hex"] = preimage.hex()
    receipt["receipt_root"] = content_hash(RECEIPT_DOMAIN, _receipt_value(receipt)).hex()
    validate_qualification_receipt_v1(receipt, golden=golden, basis_commit=basis_commit)
    return MappingProxyType(receipt)


def _qualified_implementation_inputs_v1(
    base: Mapping[str, object],
    *,
    receipt: Mapping[str, object],
    receipt_root: bytes,
    golden_root: bytes,
    runtime_seccomp_digest: bytes,
    build_seccomp_digest: bytes,
    docker_daemon_receipt: Mapping[str, object],
    docker_daemon_receipt_binding: bytes,
    cargo_bootstrap_record_digest: bytes,
    python_source_root: bytes,
    rust_source_root: bytes,
    python_image: str,
    rust_image: str,
    python_binary_digest: bytes,
    rust_binary_path: Path,
    rust_binary_digest: bytes,
    rust_dependency_snapshot_root: bytes,
    rust_dependency_snapshot_file_count: int,
    python_binding_root: bytes,
    rust_binding_root: bytes,
) -> dict[str, object]:
    """Assemble the only input map allowed to flip binding readiness."""

    digests = (
        receipt_root,
        golden_root,
        runtime_seccomp_digest,
        build_seccomp_digest,
        docker_daemon_receipt_binding,
        cargo_bootstrap_record_digest,
        python_source_root,
        rust_source_root,
        python_binary_digest,
        rust_binary_digest,
        rust_dependency_snapshot_root,
        python_binding_root,
        rust_binding_root,
    )
    if any(type(value) is not bytes or len(value) != 32 for value in digests):
        _fail(FAIL_BINDING, "builder input assembly received a non-32-byte digest")
    try:
        replayed_daemon_binding = local_docker_daemon_receipt_binding_v1(
            docker_daemon_receipt
        )
    except Phase3LocalRuntimeError as error:
        _fail(FAIL_BINDING, str(error))
    if (
        replayed_daemon_binding != docker_daemon_receipt_binding
        or receipt.get("local_docker_daemon_receipt_binding")
        != docker_daemon_receipt_binding.hex()
    ):
        _fail(FAIL_BINDING, "builder Docker daemon receipt binding differs")
    _exact_uint(
        rust_dependency_snapshot_file_count,
        label="Rust dependency snapshot file count",
        code=FAIL_BINDING,
        minimum=1,
    )
    python_receipt = receipt.get("python")
    rust_receipt = receipt.get("rust")
    if not isinstance(python_receipt, Mapping) or not isinstance(rust_receipt, Mapping):
        _fail(FAIL_BINDING, "builder input assembly lacks implementation receipt rows")
    result = dict(base)
    result.update(
        {
            "m3_execution_implementation_bindings_ready": True,
            "m3_execution_implementation_binding_roots": MappingProxyType(
                {
                    "python_implementation_binding_root": python_binding_root,
                    "rust_implementation_binding_root": rust_binding_root,
                }
            ),
            "m3_implementation_qualification_receipt": receipt,
            "m3_implementation_qualification_receipt_root": receipt_root,
            "m3_dual_golden_vector_root": golden_root,
            "m3_runtime_seccomp_sha256": runtime_seccomp_digest,
            "m3_build_seccomp_sha256": build_seccomp_digest,
            "m3_local_docker_daemon_identity_receipt": MappingProxyType(
                dict(docker_daemon_receipt)
            ),
            "m3_local_docker_daemon_receipt_binding": docker_daemon_receipt_binding,
            "m3_cargo_offline_bootstrap_record_sha256": cargo_bootstrap_record_digest,
            "m3_runtime_docker_policy_id": RUNTIME_DOCKER_POLICY_ID,
            "m3_rust_build_docker_policy_id": RUST_BUILD_DOCKER_POLICY_ID,
            "python_m3_source_paths": PYTHON_SOURCE_PATHS,
            "rust_m3_source_paths": RUST_SOURCE_PATHS,
            "python_m3_source_root": python_source_root,
            "rust_m3_source_root": rust_source_root,
            "python_m3_enumerator_image_ref": python_image,
            "rust_m3_enumerator_image_ref": rust_image,
            "python_m3_enumerator_executable_locator": python_receipt[
                "bound_executable_locator"
            ],
            "python_m3_enumerator_binary_sha256": python_binary_digest,
            "rust_m3_enumerator_binary_path": str(rust_binary_path),
            "rust_m3_enumerator_executable_locator": rust_receipt[
                "bound_executable_locator"
            ],
            "rust_m3_enumerator_binary_sha256": rust_binary_digest,
            "rust_m3_dependency_snapshot_root": rust_dependency_snapshot_root,
            "rust_m3_dependency_snapshot_file_count": rust_dependency_snapshot_file_count,
            "python_m3_entrypoint_id_digest": id_digest_v1(PYTHON_ENTRYPOINT_ID),
            "rust_m3_entrypoint_id_digest": id_digest_v1(RUST_ENTRYPOINT_ID),
        }
    )
    required = {
        "m3_runtime_seccomp_sha256": runtime_seccomp_digest,
        "m3_build_seccomp_sha256": build_seccomp_digest,
        "m3_local_docker_daemon_receipt_binding": docker_daemon_receipt_binding,
        "m3_cargo_offline_bootstrap_record_sha256": cargo_bootstrap_record_digest,
        "m3_runtime_docker_policy_id": RUNTIME_DOCKER_POLICY_ID,
        "m3_rust_build_docker_policy_id": RUST_BUILD_DOCKER_POLICY_ID,
    }
    if any(result.get(name) != expected for name, expected in required.items()):
        raise AssertionError("ready implementation input assembly lost policy binding")
    return result


def build_qualified_formal_static_basis_v1(
    basis_commit: str,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    static_rust_binary_path: Path | None = None,
) -> FormalStaticBasisV1:
    """Run the full offline dual qualification and return a ready static basis."""

    repository_root = repository_root.resolve()
    basis = build_formal_static_basis_v1(
        basis_commit,
        repository_root=repository_root,
        **(
            {}
            if static_rust_binary_path is None
            else {"rust_binary_path": static_rust_binary_path}
        ),
    )
    for required_path in REQUIRED_QUALIFICATION_BASIS_PATHS:
        _git_blob(repository_root, basis_commit, required_path)
    python_blobs = validate_python_source_closure_v1(repository_root, basis_commit)
    rust_blobs = validate_rust_source_closure_v1(repository_root, basis_commit)
    golden, golden_preimage, golden_root = load_committed_dual_golden_v1(
        repository_root, basis_commit
    )
    profile = _load_profile(repository_root, basis_commit)
    images = profile["images"]
    assert isinstance(images, dict)
    python_image = _image_ref(images["python_attester"], label="Python")
    rust_image = _image_ref(images["rust_attester"], label="Rust")

    committed_seccomp = _git_blob(repository_root, basis_commit, CONTAINER_SECCOMP_PATH)
    if hashlib.sha256(SECCOMP_PATH.read_bytes()).digest() != hashlib.sha256(committed_seccomp).digest():
        _fail(FAIL_CONTAINER_POLICY, "working seccomp profile differs from basis commit")
    committed_build_seccomp = _git_blob(
        repository_root, basis_commit, BUILD_SECCOMP_REPOSITORY_PATH
    )
    if hashlib.sha256(BUILD_SECCOMP_PATH.read_bytes()).digest() != hashlib.sha256(committed_build_seccomp).digest():
        _fail(FAIL_CONTAINER_POLICY, "working offline-build seccomp profile differs from basis commit")
    runtime_seccomp_digest = hashlib.sha256(committed_seccomp).digest()
    build_seccomp_digest = hashlib.sha256(committed_build_seccomp).digest()

    python_source_rows = _source_file_rows(repository_root, basis_commit, PYTHON_SOURCE_PATHS)
    rust_source_rows = _source_file_rows(repository_root, basis_commit, RUST_SOURCE_PATHS)
    cargo_lock_payload = _git_blob(
        repository_root,
        basis_commit,
        "Hegel Machine/rust/m3_closure_enumerator/Cargo.lock",
    )
    rust_lock_rows = _dependency_lock_rows(cargo_lock_payload)
    (
        expected_dependency_snapshot_root,
        expected_dependency_snapshot_file_count,
        cargo_bootstrap_record_digest,
    ) = _validate_cargo_bootstrap_record_v1(
        _git_blob(repository_root, basis_commit, BOOTSTRAP_RECORD_PATH),
        cargo_lock_payload=cargo_lock_payload,
    )
    python_source_root = candidate_record_tree_root("SourceFileRecordV1", python_source_rows)
    rust_source_root = candidate_record_tree_root("SourceFileRecordV1", rust_source_rows)
    python_lock_root = candidate_record_tree_root("DependencyLockRecordV1", ())
    rust_lock_root = candidate_record_tree_root("DependencyLockRecordV1", rust_lock_rows)
    python_environment = _environment_fields(python_image, "python", python_lock_root)
    rust_environment = _environment_fields(rust_image, "rust", rust_lock_root)
    python_environment_root = candidate_content_root("ExecutionEnvironmentSpecV1", python_environment)
    rust_environment_root = candidate_content_root("ExecutionEnvironmentSpecV1", rust_environment)

    try:
        workspace_owner = LinuxLocalTemporaryDirectoryV1(
            prefix="hegel-m3-qualification-",
            repository_root=repository_root,
        )
    except Phase3LocalRuntimeError as error:
        _fail(FAIL_CONTAINER_POLICY, str(error))
    with workspace_owner as raw_workspace:
        workspace = Path(raw_workspace)
        try:
            control_plane = prepare_local_docker_control_plane_v1(
                workspace,
                repository_root=repository_root,
            )
            (
                docker_daemon_receipt,
                docker_daemon_receipt_binding,
            ) = _qualify_local_docker_control_plane_v1(
                control_plane,
                repository_root=repository_root,
            )
        except Phase3LocalRuntimeError as error:
            _fail(FAIL_CONTAINER_POLICY, str(error))
        python_snapshot = workspace / "python"
        rust_snapshot = workspace / "rust"
        _write_snapshot(
            python_snapshot,
            python_blobs,
            strip_prefix="Hegel Machine/src/hegel_machine/",
        )
        # The direct entrypoint expects a hegel_machine directory.
        package = workspace / "python-package/hegel_machine"
        package.parent.mkdir(parents=True, mode=0o755)
        os.replace(python_snapshot, package)
        python_snapshot = package.parent
        _write_snapshot(rust_snapshot, rust_blobs)

        python_path, python_binary_digest, python_version_digest, _ = _probe_python(
            control_plane, python_image, seccomp_path=SECCOMP_PATH
        )
        rust_version_digest, _ = _probe_rust_version(
            control_plane, rust_image, seccomp_path=SECCOMP_PATH
        )
        (
            rust_binary,
            build_stdout,
            build_stderr,
            rust_dependency_snapshot_root,
            rust_dependency_snapshot_file_count,
        ) = _build_rust(
            control_plane,
            rust_image,
            rust_snapshot,
            seccomp_path=BUILD_SECCOMP_PATH,
            basis_commit=basis_commit,
            repository_root=repository_root,
        )
        if (
            rust_dependency_snapshot_root != expected_dependency_snapshot_root
            or rust_dependency_snapshot_file_count
            != expected_dependency_snapshot_file_count
        ):
            _fail(FAIL_BUILD, "current Cargo vendor snapshot differs from bootstrap audit")
        rust_binary_digest = hashlib.sha256(rust_binary.read_bytes()).digest()
        binding_roots = golden["binding_roots"]
        assert isinstance(binding_roots, dict)
        roots = tuple(
            _hex32(binding_roots[name], label=name)
            for name in (
                "child_dsl_spec_root",
                "operator_semantics_root",
                "identifier_registry_root",
            )
        )
        python_execution = _run_python_enumerator(
            control_plane,
            python_image,
            python_snapshot,
            workspace / "python-output",
            roots,
            seccomp_path=SECCOMP_PATH,
        )
        rust_execution = _run_rust_enumerator(
            control_plane,
            rust_image,
            rust_binary,
            workspace / "rust-output",
            roots,
            seccomp_path=SECCOMP_PATH,
        )
        python_report = validate_enumerator_report_v1(
            _parse_single_json(
                python_execution.stdout, label="Python enumerator report"
            ),
            implementation="python",
            golden=golden,
        )
        rust_report = validate_enumerator_report_v1(
            _parse_single_json(rust_execution.stdout, label="Rust enumerator report"),
            implementation="rust",
            golden=golden,
        )
        python_archive = _host_validate_enumerator_archive_v1(
            workspace / "python-output",
            implementation="python",
            stdout_report=python_report,
            roots=roots,
        )
        rust_archive = _host_validate_enumerator_archive_v1(
            workspace / "rust-output",
            implementation="rust",
            stdout_report=rust_report,
            roots=roots,
        )
        _validate_dual_archive_bytes_equal_v1(python_archive, rust_archive)

    common_fields = REPORT_FIELDS - {
        "schema_version",
        "implementation",
        "implementation_id",
        "implementation_machine_id",
    }
    if any(python_report[field] != rust_report[field] for field in common_fields):
        _fail(FAIL_REPORT, "Python and Rust full reports disagree")

    commit_wire = git_sha1_commit_id(bytes.fromhex(basis_commit))
    python_binding = _implementation_binding_fields(
        implementation_id=1,
        source_root=python_source_root,
        binary_digest=python_binary_digest,
        environment_root=python_environment_root,
        version_digest=python_version_digest,
        dependency_lock_root=python_lock_root,
        entrypoint=PYTHON_ENTRYPOINT_ID,
        golden_root=golden_root,
        commit_wire=commit_wire,
    )
    rust_binding = _implementation_binding_fields(
        implementation_id=2,
        source_root=rust_source_root,
        binary_digest=rust_binary_digest,
        environment_root=rust_environment_root,
        version_digest=rust_version_digest,
        dependency_lock_root=rust_lock_root,
        entrypoint=RUST_ENTRYPOINT_ID,
        golden_root=golden_root,
        commit_wire=commit_wire,
    )
    python_binding_root = candidate_content_root("ImplementationBindingV1", python_binding)
    rust_binding_root = candidate_content_root("ImplementationBindingV1", rust_binding)

    python_stream_digests = python_archive["stream_digests"]
    rust_stream_digests = rust_archive["stream_digests"]
    assert isinstance(python_stream_digests, Mapping)
    assert isinstance(rust_stream_digests, Mapping)
    python_receipt_fields = {
        "implementation_id": 1,
        "implementation_machine_id": "hegel-python-m3-bounded-closure-enumerator-v1",
        "source_root": python_source_root.hex(),
        "source_file_count": len(python_source_rows),
        "dependency_lock_root": python_lock_root.hex(),
        "dependency_snapshot_root_or_null": None,
        "dependency_snapshot_file_count": 0,
        "execution_environment_spec_root": python_environment_root.hex(),
        "image_ref": python_image,
        "bound_executable_locator": f"oci://{python_image}{python_path}",
        "binary_digest": python_binary_digest.hex(),
        "compiler_or_interpreter_version_digest": python_version_digest.hex(),
        "entrypoint_id_digest": id_digest_v1(PYTHON_ENTRYPOINT_ID).hex(),
        "implementation_binding_root": python_binding_root.hex(),
        "canonical_report_sha256": hashlib.sha256(_canonical_json_bytes(dict(python_report))).hexdigest(),
        "execution_stdout_sha256": hashlib.sha256(python_execution.stdout).hexdigest(),
        "runtime_container_environment_sha256": _container_environment_digest(
            PYTHON_RUNTIME_ENVIRONMENT
        ).hex(),
        "build_container_environment_sha256_or_null": None,
        "canonical_program_records_stream_sha256": python_stream_digests[
            "canonical_program_records"
        ].hex(),
        "program_chunk_manifests_stream_sha256": python_stream_digests[
            "program_chunk_manifests"
        ].hex(),
        "bucket_accounting_records_stream_sha256": python_stream_digests[
            "bucket_accounting_records"
        ].hex(),
        "build_stdout_sha256_or_null": None,
        "build_stderr_sha256_or_null": None,
        "input_snapshot_target_free": True,
        "archive_file_set_verified": True,
        "host_strict_archive_replay_verified": True,
        "witness_adjacency_verified": True,
    }
    rust_locator = (
        "generated-target://rust/m3_closure_enumerator/target/m3_qualification/"
        + basis_commit
        + "/hegel-m3-closure-enumerator"
    )
    rust_receipt_fields = {
        "implementation_id": 2,
        "implementation_machine_id": "hegel-rust-m3-bounded-closure-enumerator-v1",
        "source_root": rust_source_root.hex(),
        "source_file_count": len(rust_source_rows),
        "dependency_lock_root": rust_lock_root.hex(),
        "dependency_snapshot_root_or_null": rust_dependency_snapshot_root.hex(),
        "dependency_snapshot_file_count": rust_dependency_snapshot_file_count,
        "execution_environment_spec_root": rust_environment_root.hex(),
        "image_ref": rust_image,
        "bound_executable_locator": rust_locator,
        "binary_digest": rust_binary_digest.hex(),
        "compiler_or_interpreter_version_digest": rust_version_digest.hex(),
        "entrypoint_id_digest": id_digest_v1(RUST_ENTRYPOINT_ID).hex(),
        "implementation_binding_root": rust_binding_root.hex(),
        "canonical_report_sha256": hashlib.sha256(_canonical_json_bytes(dict(rust_report))).hexdigest(),
        "execution_stdout_sha256": hashlib.sha256(rust_execution.stdout).hexdigest(),
        "runtime_container_environment_sha256": _container_environment_digest(
            RUST_RUNTIME_ENVIRONMENT
        ).hex(),
        "build_container_environment_sha256_or_null": _container_environment_digest(
            RUST_BUILD_ENVIRONMENT
        ).hex(),
        "canonical_program_records_stream_sha256": rust_stream_digests[
            "canonical_program_records"
        ].hex(),
        "program_chunk_manifests_stream_sha256": rust_stream_digests[
            "program_chunk_manifests"
        ].hex(),
        "bucket_accounting_records_stream_sha256": rust_stream_digests[
            "bucket_accounting_records"
        ].hex(),
        "build_stdout_sha256_or_null": hashlib.sha256(build_stdout).hexdigest(),
        "build_stderr_sha256_or_null": hashlib.sha256(build_stderr).hexdigest(),
        "input_snapshot_target_free": True,
        "archive_file_set_verified": True,
        "host_strict_archive_replay_verified": True,
        "witness_adjacency_verified": True,
    }
    receipt = _build_receipt(
        basis_commit=basis_commit,
        golden=golden,
        golden_root=golden_root,
        python_fields=python_receipt_fields,
        rust_fields=rust_receipt_fields,
        runtime_seccomp_digest=runtime_seccomp_digest,
        build_seccomp_digest=build_seccomp_digest,
        docker_daemon_receipt_binding=docker_daemon_receipt_binding,
        cargo_bootstrap_record_digest=cargo_bootstrap_record_digest,
    )
    receipt_root = validate_qualification_receipt_v1(
        receipt, golden=golden, basis_commit=basis_commit
    )

    objects = dict(basis.objects)
    objects.update(
        {
            "python_m3_execution_environment": MappingProxyType(python_environment),
            "rust_m3_execution_environment": MappingProxyType(rust_environment),
            "python_m3_implementation_binding": MappingProxyType(python_binding),
            "rust_m3_implementation_binding": MappingProxyType(rust_binding),
        }
    )
    record_sets = dict(basis.record_sets)
    record_sets.update(
        {
            "python_m3_implementation_sources": python_source_rows,
            "rust_m3_implementation_sources": rust_source_rows,
            "python_m3_dependency_lock": (),
            "rust_m3_dependency_lock": rust_lock_rows,
        }
    )
    roots_map = dict(basis.roots)
    roots_map.update(
        {
            "m3_dual_golden_vector_root": golden_root,
            "python_m3_source_root": python_source_root,
            "rust_m3_source_root": rust_source_root,
            "python_m3_dependency_lock_root": python_lock_root,
            "rust_m3_dependency_lock_root": rust_lock_root,
            "rust_m3_dependency_snapshot_root": rust_dependency_snapshot_root,
            "m3_local_docker_daemon_receipt_binding": docker_daemon_receipt_binding,
            "m3_cargo_offline_bootstrap_record_sha256": cargo_bootstrap_record_digest,
            "python_m3_execution_environment_root": python_environment_root,
            "rust_m3_execution_environment_root": rust_environment_root,
            "python_implementation_binding_root": python_binding_root,
            "rust_implementation_binding_root": rust_binding_root,
            "m3_implementation_qualification_receipt_root": receipt_root,
        }
    )
    diagnostics = dict(basis.diagnostic_preimages)
    diagnostics.update(
        {
            "m3/dual_golden_typed_cbor": golden_preimage,
            "m3/qualification_receipt_typed_cbor": bytes.fromhex(str(receipt["receipt_cbor_hex"])),
            "m3/python_enumerator_report_json": _canonical_json_bytes(dict(python_report)),
            "m3/rust_enumerator_report_json": _canonical_json_bytes(dict(rust_report)),
            "m3/local_docker_daemon_identity_receipt_json": _canonical_json_bytes(
                dict(docker_daemon_receipt)
            ),
        }
    )
    candidate_fields = dict(basis.m3_candidate_static_fields)
    candidate_fields.update(
        {
            "python_implementation_binding_root": python_binding_root,
            "rust_implementation_binding_root": rust_binding_root,
        }
    )
    implementation_inputs = _qualified_implementation_inputs_v1(
        basis.implementation_inputs,
        receipt=receipt,
        receipt_root=receipt_root,
        golden_root=golden_root,
        runtime_seccomp_digest=runtime_seccomp_digest,
        build_seccomp_digest=build_seccomp_digest,
        docker_daemon_receipt=docker_daemon_receipt,
        docker_daemon_receipt_binding=docker_daemon_receipt_binding,
        cargo_bootstrap_record_digest=cargo_bootstrap_record_digest,
        python_source_root=python_source_root,
        rust_source_root=rust_source_root,
        python_image=python_image,
        rust_image=rust_image,
        python_binary_digest=python_binary_digest,
        rust_binary_path=rust_binary,
        rust_binary_digest=rust_binary_digest,
        rust_dependency_snapshot_root=rust_dependency_snapshot_root,
        rust_dependency_snapshot_file_count=rust_dependency_snapshot_file_count,
        python_binding_root=python_binding_root,
        rust_binding_root=rust_binding_root,
    )
    blockers = tuple(
        item
        for item in basis.blocking_gaps
        if item != "M3_EXECUTION_IMPLEMENTATION_BINDINGS_NOT_READY"
    )
    qualified = replace(
        basis,
        objects=MappingProxyType(objects),
        record_sets=MappingProxyType(record_sets),
        roots=MappingProxyType(roots_map),
        diagnostic_preimages=MappingProxyType(diagnostics),
        m3_candidate_static_fields=MappingProxyType(candidate_fields),
        implementation_inputs=MappingProxyType(implementation_inputs),
        blocking_gaps=blockers,
    )
    validate_m3_execution_implementation_bindings_v1(qualified, live_python_probe=False)
    return qualified


def validate_m3_execution_implementation_bindings_v1(
    basis: FormalStaticBasisV1,
    *,
    live_python_probe: bool = True,
) -> Mapping[str, bytes]:
    """Reject any static-replayer or arbitrary executable substitution."""

    inputs = basis.implementation_inputs
    roots = inputs.get("m3_execution_implementation_binding_roots")
    if inputs.get("m3_execution_implementation_bindings_ready") is not True or not isinstance(roots, Mapping):
        _fail(FAIL_BINDING, "runnable M3 bindings are absent")
    expected_names = {
        "python_implementation_binding_root",
        "rust_implementation_binding_root",
    }
    if set(roots) != expected_names:
        _fail(FAIL_BINDING, "M3 binding root names differ")
    for name in expected_names:
        value = roots[name]
        if type(value) is not bytes or len(value) != 32 or basis.roots.get(name) != value or basis.m3_candidate_static_fields.get(name) != value:
            _fail(FAIL_BINDING, f"M3 binding root {name} is not exact")
    if tuple(inputs.get("python_m3_source_paths", ())) != PYTHON_SOURCE_PATHS or tuple(inputs.get("rust_m3_source_paths", ())) != RUST_SOURCE_PATHS:
        _fail(FAIL_BINDING, "M3 committed source closure path set differs")
    if inputs.get("python_m3_source_root") != basis.roots.get("python_m3_source_root") or inputs.get("rust_m3_source_root") != basis.roots.get("rust_m3_source_root"):
        _fail(FAIL_BINDING, "M3 committed source roots differ")
    crosslink_digest_names = (
        "python_m3_source_root",
        "rust_m3_source_root",
        "python_m3_enumerator_binary_sha256",
        "rust_m3_enumerator_binary_sha256",
    )
    if any(
        type(inputs.get(name)) is not bytes or len(inputs[name]) != 32
        for name in crosslink_digest_names
    ):
        _fail(FAIL_BINDING, "M3 input cross-link digest is not exact")
    expected_rust_path = (
        PROJECT_ROOT
        / "rust/m3_closure_enumerator/target/m3_qualification"
        / basis.basis_commit
        / "hegel-m3-closure-enumerator"
    ).resolve()
    if inputs.get("rust_m3_enumerator_binary_path") != str(expected_rust_path):
        _fail(FAIL_BINDING, "Rust enumerator path substitution detected")
    try:
        rust_payload = expected_rust_path.read_bytes()
    except OSError as error:
        _fail(FAIL_BINDING, f"bound Rust enumerator is unreadable: {error}")
    if hashlib.sha256(rust_payload).digest() != inputs.get("rust_m3_enumerator_binary_sha256"):
        _fail(FAIL_BINDING, "Rust enumerator binary digest substitution detected")
    python_image = inputs.get("python_m3_enumerator_image_ref")
    rust_image = inputs.get("rust_m3_enumerator_image_ref")
    if python_image != inputs.get("python_image_ref") or rust_image != inputs.get("rust_image_ref"):
        _fail(FAIL_BINDING, "M3 image substitution detected")
    if inputs.get("python_m3_entrypoint_id_digest") != id_digest_v1(PYTHON_ENTRYPOINT_ID) or inputs.get("rust_m3_entrypoint_id_digest") != id_digest_v1(RUST_ENTRYPOINT_ID):
        _fail(FAIL_BINDING, "M3 entrypoint substitution detected")
    if (
        inputs.get("m3_runtime_docker_policy_id") != RUNTIME_DOCKER_POLICY_ID
        or inputs.get("m3_rust_build_docker_policy_id")
        != RUST_BUILD_DOCKER_POLICY_ID
    ):
        _fail(FAIL_BINDING, "M3 runtime/build policy identity differs")
    golden, _, golden_root = load_committed_dual_golden_v1(
        basis.repository_root, basis.basis_commit
    )
    if inputs.get("m3_dual_golden_vector_root") != golden_root:
        _fail(FAIL_BINDING, "M3 golden-vector substitution detected")
    receipt = inputs.get("m3_implementation_qualification_receipt")
    receipt_root = validate_qualification_receipt_v1(
        receipt, golden=golden, basis_commit=basis.basis_commit
    )
    if receipt_root != inputs.get("m3_implementation_qualification_receipt_root") or receipt_root != basis.roots.get("m3_implementation_qualification_receipt_root"):
        _fail(FAIL_BINDING, "M3 qualification receipt root differs")
    assert isinstance(receipt, Mapping)
    daemon_receipt = inputs.get("m3_local_docker_daemon_identity_receipt")
    if not isinstance(daemon_receipt, Mapping):
        _fail(FAIL_BINDING, "local Docker daemon identity receipt is absent")
    try:
        daemon_binding = local_docker_daemon_receipt_binding_v1(daemon_receipt)
    except Phase3LocalRuntimeError as error:
        _fail(FAIL_BINDING, str(error))
    if (
        daemon_binding != inputs.get("m3_local_docker_daemon_receipt_binding")
        or daemon_binding
        != basis.roots.get("m3_local_docker_daemon_receipt_binding")
        or daemon_binding
        != _hex32(
            receipt.get("local_docker_daemon_receipt_binding"),
            label="receipt local Docker daemon binding",
            code=FAIL_BINDING,
        )
    ):
        _fail(FAIL_BINDING, "local Docker control-plane binding differs")
    try:
        committed_lock = _git_blob(
            basis.repository_root,
            basis.basis_commit,
            "Hegel Machine/rust/m3_closure_enumerator/Cargo.lock",
        )
        (
            bootstrap_snapshot_root,
            bootstrap_snapshot_file_count,
            bootstrap_record_digest,
        ) = _validate_cargo_bootstrap_record_v1(
            _git_blob(
                basis.repository_root,
                basis.basis_commit,
                BOOTSTRAP_RECORD_PATH,
            ),
            cargo_lock_payload=committed_lock,
        )
    except (OSError, M3ImplementationQualificationError):
        raise
    if (
        bootstrap_record_digest
        != inputs.get("m3_cargo_offline_bootstrap_record_sha256")
        or bootstrap_record_digest
        != basis.roots.get("m3_cargo_offline_bootstrap_record_sha256")
        or bootstrap_record_digest
        != _hex32(
            receipt.get("cargo_offline_bootstrap_record_sha256"),
            label="receipt Cargo bootstrap record digest",
            code=FAIL_BINDING,
        )
        or bootstrap_snapshot_root
        != inputs.get("rust_m3_dependency_snapshot_root")
        or bootstrap_snapshot_file_count
        != inputs.get("rust_m3_dependency_snapshot_file_count")
    ):
        _fail(FAIL_BINDING, "Cargo bootstrap/snapshot binding differs")
    python_receipt = receipt.get("python")
    rust_receipt = receipt.get("rust")
    if not isinstance(python_receipt, Mapping) or not isinstance(rust_receipt, Mapping):
        _fail(FAIL_BINDING, "M3 qualification implementation rows are absent")
    if (
        python_receipt.get("image_ref") != python_image
        or rust_receipt.get("image_ref") != rust_image
        or python_receipt.get("source_root")
        != inputs.get("python_m3_source_root").hex()
        or rust_receipt.get("source_root") != inputs.get("rust_m3_source_root").hex()
        or python_receipt.get("dependency_lock_root")
        != basis.roots.get("python_m3_dependency_lock_root").hex()
        or rust_receipt.get("dependency_lock_root")
        != basis.roots.get("rust_m3_dependency_lock_root").hex()
        or python_receipt.get("execution_environment_spec_root")
        != basis.roots.get("python_m3_execution_environment_root").hex()
        or rust_receipt.get("execution_environment_spec_root")
        != basis.roots.get("rust_m3_execution_environment_root").hex()
        or python_receipt.get("binary_digest")
        != inputs.get("python_m3_enumerator_binary_sha256").hex()
        or rust_receipt.get("binary_digest")
        != inputs.get("rust_m3_enumerator_binary_sha256").hex()
        or python_receipt.get("implementation_binding_root")
        != roots["python_implementation_binding_root"].hex()
        or rust_receipt.get("implementation_binding_root")
        != roots["rust_implementation_binding_root"].hex()
    ):
        _fail(FAIL_BINDING, "M3 receipt/input cross-link differs")
    rust_snapshot_root = _hex32(
        rust_receipt.get("dependency_snapshot_root_or_null"),
        label="receipt Rust dependency snapshot root",
        code=FAIL_BINDING,
    )
    if (
        inputs.get("rust_m3_dependency_snapshot_root") != rust_snapshot_root
        or basis.roots.get("rust_m3_dependency_snapshot_root") != rust_snapshot_root
        or inputs.get("rust_m3_dependency_snapshot_file_count")
        != rust_receipt.get("dependency_snapshot_file_count")
    ):
        _fail(FAIL_BINDING, "Rust dependency snapshot binding differs")

    try:
        source_record_sets = (
            (
                "python_m3_implementation_sources",
                "python_m3_source_root",
                "SourceFileRecordV1",
                python_receipt["source_file_count"],
            ),
            (
                "rust_m3_implementation_sources",
                "rust_m3_source_root",
                "SourceFileRecordV1",
                rust_receipt["source_file_count"],
            ),
            (
                "python_m3_dependency_lock",
                "python_m3_dependency_lock_root",
                "DependencyLockRecordV1",
                None,
            ),
            (
                "rust_m3_dependency_lock",
                "rust_m3_dependency_lock_root",
                "DependencyLockRecordV1",
                None,
            ),
        )
        for record_name, root_name, schema_name, expected_count in source_record_sets:
            rows = basis.record_sets[record_name]
            if expected_count is not None and len(rows) != expected_count:
                _fail(FAIL_BINDING, f"M3 formal record count {record_name} differs")
            replayed = candidate_record_tree_root(
                schema_name, rows
            )
            if basis.roots.get(root_name) != replayed:
                _fail(FAIL_BINDING, f"M3 formal record root {root_name} does not replay")
        content_objects = (
            (
                "python_m3_execution_environment",
                "python_m3_execution_environment_root",
                "ExecutionEnvironmentSpecV1",
            ),
            (
                "rust_m3_execution_environment",
                "rust_m3_execution_environment_root",
                "ExecutionEnvironmentSpecV1",
            ),
            (
                "python_m3_implementation_binding",
                "python_implementation_binding_root",
                "ImplementationBindingV1",
            ),
            (
                "rust_m3_implementation_binding",
                "rust_implementation_binding_root",
                "ImplementationBindingV1",
            ),
        )
        for object_name, root_name, schema_name in content_objects:
            replayed = candidate_content_root(schema_name, basis.objects[object_name])
            if basis.roots.get(root_name) != replayed:
                _fail(FAIL_BINDING, f"M3 formal content root {root_name} does not replay")
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        _fail(FAIL_BINDING, f"M3 formal preimage set is incomplete: {error}")

    binding_crosslinks = (
        (
            basis.objects["python_m3_implementation_binding"],
            python_receipt,
            basis.roots["python_m3_source_root"],
            basis.roots["python_m3_dependency_lock_root"],
            basis.roots["python_m3_execution_environment_root"],
        ),
        (
            basis.objects["rust_m3_implementation_binding"],
            rust_receipt,
            basis.roots["rust_m3_source_root"],
            basis.roots["rust_m3_dependency_lock_root"],
            basis.roots["rust_m3_execution_environment_root"],
        ),
    )
    for binding, implementation_receipt, source_root, lock_root, environment_root in binding_crosslinks:
        if (
            binding.get("source_root") != source_root
            or binding.get("dependency_lock_root") != lock_root
            or binding.get("execution_environment_spec_root") != environment_root
            or binding.get("binary_digest")
            != _hex32(
                implementation_receipt["binary_digest"],
                label="receipt binary digest",
                code=FAIL_BINDING,
            )
            or binding.get("golden_vector_root") != golden_root
            or binding.get("repository_commit_id")
            != git_sha1_commit_id(bytes.fromhex(basis.basis_commit))
        ):
            _fail(FAIL_BINDING, "M3 ImplementationBindingV1 preimage cross-link differs")
    if (
        inputs.get("m3_runtime_seccomp_sha256")
        != _hex32(
            receipt["runtime_seccomp_sha256"],
            label="receipt runtime seccomp",
            code=FAIL_BINDING,
        )
        or inputs.get("m3_build_seccomp_sha256")
        != _hex32(
            receipt["build_seccomp_sha256"],
            label="receipt build seccomp",
            code=FAIL_BINDING,
        )
    ):
        _fail(FAIL_BINDING, "M3 seccomp policy digest substitution detected")
    if live_python_probe:
        if type(python_image) is not str:
            _fail(FAIL_BINDING, "Python M3 image reference is absent")
        try:
            live_owner = LinuxLocalTemporaryDirectoryV1(
                prefix="hegel-m3-binding-live-probe-",
                repository_root=basis.repository_root,
            )
        except Phase3LocalRuntimeError as error:
            _fail(FAIL_BINDING, str(error))
        with live_owner as raw_live_runtime:
            try:
                live_control = prepare_local_docker_control_plane_v1(
                    Path(raw_live_runtime),
                    repository_root=basis.repository_root,
                )
            except Phase3LocalRuntimeError as error:
                _fail(FAIL_BINDING, str(error))
            _, live_daemon_binding = _qualify_local_docker_control_plane_v1(
                live_control,
                repository_root=basis.repository_root,
            )
            if live_daemon_binding != daemon_binding:
                _fail(FAIL_BINDING, "local Docker daemon identity changed")
            _, digest, _, _ = _probe_python(
                live_control,
                python_image,
                seccomp_path=SECCOMP_PATH,
            )
        if digest != inputs.get("python_m3_enumerator_binary_sha256"):
            _fail(FAIL_BINDING, "Python interpreter digest substitution detected")
    return MappingProxyType(dict(roots))


__all__ = [
    "M3ImplementationQualificationError",
    "PYTHON_SOURCE_PATHS",
    "RUST_SOURCE_PATHS",
    "build_qualified_formal_static_basis_v1",
    "load_committed_dual_golden_v1",
    "validate_dual_golden_v1",
    "validate_enumerator_report_v1",
    "validate_m3_execution_implementation_bindings_v1",
    "validate_python_source_closure_v1",
    "validate_qualification_receipt_v1",
    "validate_rust_source_closure_v1",
]
