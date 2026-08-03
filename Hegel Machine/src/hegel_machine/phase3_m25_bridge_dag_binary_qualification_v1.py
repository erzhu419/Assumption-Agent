"""Offline Commit-A qualification for the independent Rust bridge-DAG replay.

This module builds only from exact Git blobs and checksum-verified Cargo
archives inside the already pinned Rust OCI image.  The persisted binary and
report are deterministic implementation evidence, never ceremony evidence:
no seed, private key, signature, authoritative/formal root, gate promotion, or
M3 state transition is produced here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tarfile
import tempfile
from typing import Final, NoReturn
import zlib

from .phase3_local_runtime_v1 import (
    LinuxLocalTemporaryDirectoryV1,
    Phase3LocalRuntimeError,
    local_docker_daemon_receipt_binding_v1,
    prepare_local_docker_control_plane_v1,
)
from .phase3_m25_bridge_full_dag_replay_v1 import (
    BridgeDagReplayError,
    FAIL_NODE_SET,
    FAIL_PACKAGE_AUTHORITY,
    FAIL_ROOT_BINDING,
    replay_bridge_dag_package_v1,
    validate_bridge_actor_replay_receipt_v1,
)
from .phase3_m25_errata_qualification_v1 import (
    BUILD_SECCOMP_PATH,
    CARGO_SNAPSHOT_DOMAIN,
    RUNTIME_SECCOMP_PATH,
    RUST_BUILD_ENVIRONMENT,
    RUST_CARGO_PATH,
    RUST_IMAGE_REF,
    RUST_RUNTIME_ENVIRONMENT,
    SOURCE_PATHS as ERRATA_SOURCE_PATHS,
    _approved_rust_toolchain,
    _build_cargo_dependency_snapshot,
    _cargo_lock_registry_packages,
    _docker_command,
    _load_approved_toolchain_policy,
    _make_snapshot_read_only,
    _qualify_local_docker_control_plane,
    _sha256_directory_manifest,
)
from .strict_cbor_v1 import canonical_cbor_decode, canonical_cbor_encode


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT: Final = PROJECT_ROOT.parent
GIT_EXECUTABLE: Final = "/usr/bin/git"
CRATE_REPOSITORY_PATH: Final = "Hegel Machine/rust/m25_bridge_dag_replay"
CRATE_ROOT: Final = PROJECT_ROOT / "rust/m25_bridge_dag_replay"
DEFAULT_RUST_BRIDGE_DAG_BINARY: Final = (
    CRATE_ROOT
    / "target/commit_a_qualified/hegel-m25-bridge-dag-replay"
)
DEFAULT_RUST_BRIDGE_DAG_QUALIFICATION_REPORT: Final = (
    PROJECT_ROOT
    / "artifacts/phase3_m25_external/"
    "phase3_m25_bridge_dag_rust_binary_qualification_v1.json"
)
GOLDEN_FIXTURE_REPOSITORY_PATH: Final = (
    "Hegel Machine/golden_vectors/phase3_m25_bridge_dag_purpose1_replay_v1.json"
)
GOLDEN_FIXTURE_PATH: Final = REPOSITORY_ROOT / GOLDEN_FIXTURE_REPOSITORY_PATH
APPROVED_TOOLCHAIN_POLICY_REPOSITORY_PATH: Final = (
    "Hegel Machine/config/phase3_m25_approved_local_rust_toolchain_v1.json"
)
ENGINEERING_DOCUMENT_REPOSITORY_PATH: Final = (
    "Hegel Machine/docs/"
    "Hegel_Machine_Phase3A_M25_Bridge_Rust_Offline_Binary_Qualification_v1.md"
)
QUALIFICATION_MODULE_REPOSITORY_PATH: Final = (
    "Hegel Machine/src/hegel_machine/"
    "phase3_m25_bridge_dag_binary_qualification_v1.py"
)
QUALIFICATION_TOOL_REPOSITORY_PATH: Final = (
    "Hegel Machine/tools/phase3_m25_bridge_dag_binary_qualification_v1.py"
)
QUALIFICATION_TEST_REPOSITORY_PATH: Final = (
    "Hegel Machine/tests/test_phase3_m25_bridge_dag_binary_qualification_v1.py"
)

SCHEMA_VERSION: Final = "hegel-phase3-m25-bridge-dag-rust-binary-qualification/1"
ARTIFACT_KIND: Final = "DETERMINISTIC_CANDIDATE_NON_AUTHORITATIVE"
STATUS: Final = "OFFLINE_COMMIT_A_RUST_BRIDGE_DAG_BINARY_PASS"
CLAIM_LEVEL: Final = "IMPLEMENTATION_BINARY_QUALIFICATION_ONLY_NOT_FORMAL_EVIDENCE"
SOURCE_ARCHIVE_DOMAIN: Final = "HEGEL/M25/BRIDGE_DAG_SOURCE_ARCHIVE/V1"
RUNTIME_DOCKER_POLICY_ID: Final = "hegel-m25-bridge-dag-rust-runtime-docker-v1"
BUILD_DOCKER_POLICY_ID: Final = "hegel-m25-bridge-dag-rust-offline-build-docker-v1"
PERSISTED_BINARY_REPOSITORY_PATH: Final = (
    "rust/m25_bridge_dag_replay/target/commit_a_qualified/"
    "hegel-m25-bridge-dag-replay"
)
REPORT_SHA256_PREFIX: Final = b"HEGEL/M25/BRIDGE_DAG_BINARY_QUALIFICATION_REPORT/V1\x00"

FAIL_COMMIT: Final = "FAIL_M25_BRIDGE_BINARY_QUALIFICATION_COMMIT"
FAIL_SOURCE: Final = "FAIL_M25_BRIDGE_BINARY_QUALIFICATION_SOURCE"
FAIL_DEPENDENCY: Final = "FAIL_M25_BRIDGE_BINARY_QUALIFICATION_DEPENDENCY"
FAIL_CONTAINER: Final = "FAIL_M25_BRIDGE_BINARY_QUALIFICATION_CONTAINER"
FAIL_BUILD: Final = "FAIL_M25_BRIDGE_BINARY_QUALIFICATION_BUILD"
FAIL_REPLAY: Final = "FAIL_M25_BRIDGE_BINARY_QUALIFICATION_REPLAY"
FAIL_PERSIST: Final = "FAIL_M25_BRIDGE_BINARY_QUALIFICATION_PERSIST"
FAIL_REPORT: Final = "FAIL_M25_BRIDGE_BINARY_QUALIFICATION_REPORT"

RUST_BUILD_SOURCE_PATHS: Final = (
    f"{CRATE_REPOSITORY_PATH}/Cargo.toml",
    f"{CRATE_REPOSITORY_PATH}/Cargo.lock",
    f"{CRATE_REPOSITORY_PATH}/src/lib.rs",
    f"{CRATE_REPOSITORY_PATH}/src/main.rs",
    "Hegel Machine/rust/formal_bridge_m25/Cargo.toml",
    "Hegel Machine/rust/formal_bridge_m25/Cargo.lock",
    "Hegel Machine/rust/formal_bridge_m25/src/lib.rs",
    "Hegel Machine/rust/formal_bridge_m25/src/main.rs",
)
QUALIFICATION_SOURCE_PATHS: Final = tuple(
    dict.fromkeys(
        (
            *RUST_BUILD_SOURCE_PATHS,
            *(
                f"Hegel Machine/{path}"
                for path in ERRATA_SOURCE_PATHS
            ),
            "Hegel Machine/src/hegel_machine/phase3_m25_bridge_full_dag_replay_v1.py",
            QUALIFICATION_MODULE_REPOSITORY_PATH,
            QUALIFICATION_TOOL_REPOSITORY_PATH,
            "Hegel Machine/tests/test_phase3_m25_bridge_full_dag_replay_v1.py",
            QUALIFICATION_TEST_REPOSITORY_PATH,
            GOLDEN_FIXTURE_REPOSITORY_PATH,
            ENGINEERING_DOCUMENT_REPOSITORY_PATH,
        )
    )
)

AUTHORITY_BOUNDARY: Final = {
    "authoritative_claim_allowed": False,
    "formal_roots_generated": False,
    "seed_generated": False,
    "private_key_generated": False,
    "signature_generated": False,
    "real_secret_material_accessed": False,
    "m3_gate_delta": 0,
    "m3_gates_before": 14,
    "m3_gates_after": 14,
    "child_state": "NOT_RUN",
    "m3_run_started": False,
    "same_admin_controller": True,
    "organizational_independence": False,
    "independent_human_actors": False,
    "technical_role_independence": True,
    "owner_accepted_threat_model": True,
    "remote_attestation": False,
    "hardware_key_nonexportability": False,
    "synthetic_public_unsigned_replay_fixture_only": True,
}

BUILD_COMMAND: Final = (
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
TEST_COMMAND: Final = (
    RUST_CARGO_PATH,
    "--config",
    'source.crates-io.replace-with="vendored-sources"',
    "--config",
    'source.vendored-sources.directory="/vendor"',
    "test",
    "--release",
    "--locked",
    "--offline",
    "--jobs=1",
    "--manifest-path",
    "Cargo.toml",
    "--",
    "--test-threads=1",
)


class BridgeDagBinaryQualificationError(RuntimeError):
    """Stable fail-closed qualification error."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise BridgeDagBinaryQualificationError(code, detail)


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, *, code: str = FAIL_SOURCE) -> str:
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            _fail(code, f"not a regular non-symlink file: {path}")
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        _fail(code, f"cannot read {path}: {exc}")


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        _fail(FAIL_REPORT, f"{label} is not an exact SHA-256")
    return value


def _require_commit(value: object) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        _fail(FAIL_COMMIT, "basis commit must be lowercase 40-hex Git SHA-1")
    return value


def _git_environment() -> dict[str, str]:
    """Return the complete, non-inheriting environment for local Git reads.

    ``git show`` and ``git archive`` may otherwise lazy-fetch missing objects
    from a promisor remote in a partial clone.  Qualification must instead
    fail closed when an object is not already present in the local object
    database, and no caller Git configuration or transport setting may repair
    that absence.
    """

    return {
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


def _run_git(arguments: Sequence[str], *, timeout: int = 120) -> bytes:
    try:
        completed = subprocess.run(
            [GIT_EXECUTABLE, *arguments],
            cwd=REPOSITORY_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
            env=_git_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail(FAIL_COMMIT, f"Git operation failed: {exc}")
    if completed.returncode != 0:
        _fail(
            FAIL_COMMIT,
            "Git operation failed: "
            + completed.stderr.decode("utf-8", "replace")[-1000:],
        )
    return completed.stdout


def repository_head_commit_v1() -> str:
    return _require_commit(_run_git(("rev-parse", "HEAD"), timeout=30).decode("ascii").strip())


def _commit_source_bindings_v1(
    commit: str,
    *,
    compare_worktree: bool,
) -> dict[str, str]:
    commit = _require_commit(commit)
    bindings: dict[str, str] = {}
    for repository_path in QUALIFICATION_SOURCE_PATHS:
        payload = _run_git(("show", f"{commit}:{repository_path}"))
        if compare_worktree:
            path = REPOSITORY_ROOT / repository_path
            try:
                worktree = path.read_bytes()
            except OSError as exc:
                _fail(FAIL_SOURCE, f"cannot read qualification source {repository_path}: {exc}")
            if worktree != payload:
                _fail(
                    FAIL_SOURCE,
                    f"qualification source differs from Commit A: {repository_path}",
                )
        bindings[repository_path] = _sha256_bytes(payload)
    return bindings


def _archive_commit_a_sources_v1(commit: str, destination: Path) -> Path:
    """Extract only the exact bound Git archive and verify every blob again."""

    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    archive_payload = _run_git(
        ("archive", "--format=tar", _require_commit(commit), "--", *QUALIFICATION_SOURCE_PATHS),
        timeout=180,
    )
    destination_resolved = destination.resolve(strict=True)
    try:
        archive = tarfile.open(fileobj=io.BytesIO(archive_payload), mode="r:")
    except tarfile.TarError as exc:
        _fail(FAIL_SOURCE, f"Git archive is invalid: {exc}")
    with archive:
        members = archive.getmembers()
        if not members:
            _fail(FAIL_SOURCE, "Git archive is empty")
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts:
                _fail(FAIL_SOURCE, "Git archive contains an unsafe path")
            if not (member.isdir() or member.isfile()):
                _fail(FAIL_SOURCE, "Git archive contains a non-file entry")
            try:
                (destination / path).resolve().relative_to(destination_resolved)
            except ValueError:
                _fail(FAIL_SOURCE, "Git archive entry escapes destination")
        archive.extractall(destination)
    for repository_path in QUALIFICATION_SOURCE_PATHS:
        extracted = destination / repository_path
        committed = _run_git(("show", f"{commit}:{repository_path}"))
        try:
            observed = extracted.read_bytes()
        except OSError as exc:
            _fail(FAIL_SOURCE, f"archive omitted {repository_path}: {exc}")
        if observed != committed:
            _fail(FAIL_SOURCE, f"Git archive blob differs: {repository_path}")
    return destination


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        _fail(FAIL_REPORT, f"report is not canonical ASCII JSON: {exc}")


def _report_sha256(body: Mapping[str, object]) -> str:
    return _sha256_bytes(REPORT_SHA256_PREFIX + _canonical_json_bytes(dict(body)))


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        def reject_float(_value: str) -> NoReturn:
            _fail(FAIL_REPORT, f"{label} contains a non-integer number")

        def reject_constant(_value: str) -> NoReturn:
            _fail(FAIL_REPORT, f"{label} contains a non-JSON constant")

        def exact_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    _fail(FAIL_REPORT, f"{label} contains duplicate key {key}")
                result[key] = value
            return result

        value = json.loads(
            payload.decode("ascii", "strict"),
            parse_float=reject_float,
            parse_constant=reject_constant,
            object_pairs_hook=exact_object,
        )
    except BridgeDagBinaryQualificationError:
        raise
    except Exception as exc:
        _fail(FAIL_REPORT, f"{label} is not strict JSON: {exc}")
    if type(value) is not dict or _canonical_json_bytes(value) != payload:
        _fail(FAIL_REPORT, f"{label} is not one canonical JSON line")
    return value


def load_unsigned_public_replay_fixture_v1(path: Path = GOLDEN_FIXTURE_PATH) -> bytes:
    value = _strict_json_object(path.read_bytes(), label="bridge replay fixture")
    expected_fields = {
        "artifact_kind",
        "authority",
        "compression",
        "contains_formal_commitment",
        "contains_private_key",
        "contains_seed",
        "contains_signature",
        "expected",
        "package_sha256",
        "package_uncompressed_size",
        "package_zlib_base64",
        "purpose_id",
        "schema_version",
    }
    if set(value) != expected_fields:
        _fail(FAIL_REPLAY, "bridge replay fixture field set differs")
    if (
        value["schema_version"] != "hegel-m25-bridge-dag-purpose1-replay-fixture/1"
        or value["artifact_kind"] != "SYNTHETIC_PUBLIC_NON_AUTHORITATIVE"
        or value["authority"] is not False
        or value["purpose_id"] != 1
        or value["compression"] != "zlib-level-9"
        or any(
            value[field] is not False
            for field in (
                "contains_formal_commitment",
                "contains_private_key",
                "contains_seed",
                "contains_signature",
            )
        )
    ):
        _fail(FAIL_REPLAY, "bridge replay fixture authority boundary differs")
    encoded = value["package_zlib_base64"]
    if type(encoded) is not str:
        _fail(FAIL_REPLAY, "bridge replay fixture payload is not text")
    try:
        compressed = base64.b64decode(encoded, validate=True)
        package = zlib.decompress(compressed)
    except (ValueError, zlib.error) as exc:
        _fail(FAIL_REPLAY, f"bridge replay fixture cannot be decoded: {exc}")
    if (
        type(value["package_uncompressed_size"]) is not int
        or value["package_uncompressed_size"] != len(package)
        or value["package_sha256"] != _sha256_bytes(package)
    ):
        _fail(FAIL_REPLAY, "bridge replay fixture payload binding differs")
    try:
        replay = replay_bridge_dag_package_v1(package)
    except BridgeDagReplayError as exc:
        _fail(FAIL_REPLAY, f"Python replay rejected public fixture: {exc}")
    expected = value["expected"]
    if type(expected) is not dict or expected != {
        "bridge_statement_root_hex": replay.bridge_statement_root.hex(),
        "candidate_root_hex": replay.candidate_root.hex(),
        "package_digest_hex": replay.package_digest.hex(),
        "purpose1_signature_verified": False,
        "split_membership_recomputed": False,
    }:
        _fail(FAIL_REPLAY, "bridge replay fixture expected result differs")
    return package


def _mutated_packages_v1(package: bytes) -> tuple[tuple[str, bytes, str], ...]:
    def mutate(function) -> bytes:
        value = list(canonical_cbor_decode(package))
        function(value)
        return canonical_cbor_encode(tuple(value))

    def substitute(value: list[object]) -> None:
        nodes = list(value[7])
        node = list(nodes[0])
        rows = list(node[5])
        attacked = bytearray(rows[0])
        attacked[-1] ^= 1
        rows[0] = bytes(attacked)
        node[5] = tuple(rows)
        nodes[0] = tuple(node)
        value[7] = tuple(nodes)

    return (
        ("PUBLIC_PREIMAGE_SUBSTITUTION_REJECTED", mutate(substitute), FAIL_ROOT_BINDING),
        (
            "PUBLIC_NODE_OMISSION_REJECTED",
            mutate(lambda value: value.__setitem__(7, tuple(value[7][:-1]))),
            FAIL_NODE_SET,
        ),
        (
            "AUTHORITATIVE_FLAG_WITHOUT_RUNTIME_OPT_IN_REJECTED",
            mutate(lambda value: value.__setitem__(3, True)),
            FAIL_PACKAGE_AUTHORITY,
        ),
    )


def _run_process(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    timeout: int,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
            env=dict(environment),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail(FAIL_CONTAINER, f"isolated process failed: {exc}")


def _require_success(
    completed: subprocess.CompletedProcess[bytes], *, label: str, code: str
) -> None:
    if completed.returncode != 0:
        _fail(code, f"{label} failed: " + completed.stderr.decode("utf-8", "replace")[-2000:])


def _run_rust_replay_v1(
    control_plane,
    binary: Path,
    package_path: Path,
) -> subprocess.CompletedProcess[bytes]:
    command = _docker_command(
        control_plane,
        RUST_IMAGE_REF,
        (
            "-v",
            f"{binary.resolve(strict=True)}:/opt/hegel-m25-bridge-dag-replay:ro",
            "-v",
            f"{package_path.resolve(strict=True)}:/input/package.cbor:ro",
        ),
        ("/opt/hegel-m25-bridge-dag-replay", "/input/package.cbor", "/tmp"),
        seccomp_path=RUNTIME_SECCOMP_PATH,
        container_environment=RUST_RUNTIME_ENVIRONMENT,
    )
    return _run_process(
        command,
        environment=control_plane.environment,
        timeout=120,
    )


def _replay_test_row(
    test_id: str,
    completed: subprocess.CompletedProcess[bytes],
    *,
    expected_returncode: int,
    expected_error_code: str | None,
) -> dict[str, object]:
    if completed.returncode != expected_returncode:
        _fail(FAIL_REPLAY, f"{test_id} returned {completed.returncode}")
    stderr = completed.stderr.decode("utf-8", "replace")
    if expected_error_code is not None and expected_error_code not in stderr:
        _fail(FAIL_REPLAY, f"{test_id} omitted exact failure code {expected_error_code}")
    return {
        "test_id": test_id,
        "expected_returncode": expected_returncode,
        "observed_returncode": completed.returncode,
        "expected_error_code_or_null": expected_error_code,
        "stdout_sha256": _sha256_bytes(completed.stdout),
        "stderr_sha256": _sha256_bytes(completed.stderr),
    }


def _run_replay_tests_v1(
    control_plane,
    binary: Path,
    fixture_path: Path,
    package: bytes,
    *,
    include_persisted_id: bool,
) -> tuple[dict[str, object], ...]:
    expected = replay_bridge_dag_package_v1(package)
    positive = _run_rust_replay_v1(control_plane, binary, fixture_path)
    _require_success(positive, label="public purpose-1 Rust replay", code=FAIL_REPLAY)
    try:
        validate_bridge_actor_replay_receipt_v1(
            positive.stdout,
            expected_result=expected,
            expected_implementation="rust-full-dag-replay-v1",
            require_authoritative=False,
        )
    except BridgeDagReplayError as exc:
        _fail(FAIL_REPLAY, f"Rust replay receipt differs: {exc}")
    rows = [
        _replay_test_row(
            "PERSISTED_PUBLIC_PURPOSE1_REPLAY_PASS"
            if include_persisted_id
            else "FRESH_PUBLIC_PURPOSE1_REPLAY_PASS",
            positive,
            expected_returncode=0,
            expected_error_code=None,
        )
    ]
    if include_persisted_id:
        return tuple(rows)
    for index, (test_id, attacked, error_code) in enumerate(_mutated_packages_v1(package)):
        path = fixture_path.parent / f"attack-{index}.cbor"
        path.write_bytes(attacked)
        path.chmod(0o444)
        completed = _run_rust_replay_v1(control_plane, binary, path)
        rows.append(
            _replay_test_row(
                test_id,
                completed,
                expected_returncode=1,
                expected_error_code=error_code,
            )
        )
    return tuple(rows)


def _persist_validated_binary_v1(payload: bytes, expected_sha256: str) -> dict[str, object]:
    if _sha256_bytes(payload) != expected_sha256:
        _fail(FAIL_PERSIST, "validated binary payload digest differs")
    destination = DEFAULT_RUST_BRIDGE_DAG_BINARY.absolute()
    try:
        destination.relative_to(CRATE_ROOT.absolute())
    except ValueError:
        _fail(FAIL_PERSIST, "persisted bridge binary escapes its crate")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.resolve(strict=True) != destination.parent:
        _fail(FAIL_PERSIST, "persisted bridge binary parent contains a symlink")
    if destination.is_symlink():
        _fail(FAIL_PERSIST, "persisted bridge binary is a symlink")
    descriptor, name = tempfile.mkstemp(prefix=".bridge-dag.pending-", dir=destination.parent)
    pending = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        pending.chmod(0o755)
        if _sha256_file(pending, code=FAIL_PERSIST) != expected_sha256:
            _fail(FAIL_PERSIST, "pending bridge binary digest differs")
        os.replace(pending, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if pending.exists():
            pending.unlink()
    if (
        destination.is_symlink()
        or stat.S_IMODE(destination.stat().st_mode) != 0o755
        or _sha256_file(destination, code=FAIL_PERSIST) != expected_sha256
    ):
        _fail(FAIL_PERSIST, "persisted bridge binary validation differs")
    return {
        "repository_path": PERSISTED_BINARY_REPOSITORY_PATH,
        "sha256": expected_sha256,
        "mode_octal": "0755",
        "atomic_replace": True,
        "is_symlink": False,
    }


def _fresh_qualification_v1(commit: str) -> dict[str, object]:
    source_bindings = _commit_source_bindings_v1(commit, compare_worktree=True)
    try:
        temporary_owner = LinuxLocalTemporaryDirectoryV1(
            prefix="hegel-m25-bridge-dag-build-",
            repository_root=REPOSITORY_ROOT,
        )
    except Phase3LocalRuntimeError as exc:
        _fail(FAIL_CONTAINER, str(exc))
    with temporary_owner as temporary:
        root = Path(temporary)
        try:
            control_plane = prepare_local_docker_control_plane_v1(
                root, repository_root=REPOSITORY_ROOT
            )
        except Phase3LocalRuntimeError as exc:
            _fail(FAIL_CONTAINER, str(exc))
        daemon_receipt, daemon_binding = _qualify_local_docker_control_plane(control_plane)
        toolchain_policy, toolchain_receipt = _approved_rust_toolchain(control_plane)
        snapshot = _archive_commit_a_sources_v1(commit, root / "source")
        snapshot_manifest = _sha256_directory_manifest(snapshot)
        snapshot_bindings = {
            path: _sha256_file(snapshot / path) for path in QUALIFICATION_SOURCE_PATHS
        }
        if snapshot_bindings != source_bindings:
            _fail(FAIL_SOURCE, "Git archive source bindings differ from Commit A")

        lock_path = snapshot / f"{CRATE_REPOSITORY_PATH}/Cargo.lock"
        lock_sha256 = _sha256_file(lock_path)
        packages = _cargo_lock_registry_packages(lock_path)
        vendor = root / "vendor"
        dependency_root, dependency_file_count, dependency_package_count = (
            _build_cargo_dependency_snapshot(lock_path, vendor)
        )
        if (
            dependency_root != toolchain_policy["dependency_snapshot_root"]
            or dependency_file_count != toolchain_policy["dependency_snapshot_file_count"]
            or dependency_package_count != len(packages)
            or dependency_package_count
            != toolchain_policy["cargo_lock_registry_package_count"]
        ):
            _fail(FAIL_DEPENDENCY, "checksum-exact vendor snapshot policy differs")
        vendor_manifest = _sha256_directory_manifest(vendor)
        _make_snapshot_read_only(snapshot)

        target = root / "target"
        target.mkdir(mode=0o700)
        build_options = (
            "-v",
            f"{snapshot}:/input:ro",
            "-v",
            f"{vendor}:/vendor:ro",
            "-v",
            f"{target}:/output:rw",
            "-w",
            f"/input/{CRATE_REPOSITORY_PATH}",
        )
        for label, command in (("offline Cargo test", TEST_COMMAND), ("offline Cargo build", BUILD_COMMAND)):
            completed = _run_process(
                _docker_command(
                    control_plane,
                    RUST_IMAGE_REF,
                    build_options,
                    command,
                    seccomp_path=BUILD_SECCOMP_PATH,
                    container_environment=RUST_BUILD_ENVIRONMENT,
                    user=f"{os.getuid()}:{os.getgid()}",
                ),
                environment=control_plane.environment,
                timeout=600,
            )
            _require_success(completed, label=label, code=FAIL_BUILD)

        built = target / "release/hegel-m25-bridge-dag-replay"
        built_sha256 = _sha256_file(built, code=FAIL_BUILD)
        fixture = load_unsigned_public_replay_fixture_v1(
            snapshot / GOLDEN_FIXTURE_REPOSITORY_PATH
        )
        replay_directory = root / "replay"
        replay_directory.mkdir(mode=0o700)
        fixture_path = replay_directory / "purpose1-public.cbor"
        fixture_path.write_bytes(fixture)
        fixture_path.chmod(0o444)
        fresh_tests = _run_replay_tests_v1(
            control_plane, built, fixture_path, fixture, include_persisted_id=False
        )
        persistence = _persist_validated_binary_v1(built.read_bytes(), built_sha256)
        persisted_tests = _run_replay_tests_v1(
            control_plane,
            DEFAULT_RUST_BRIDGE_DAG_BINARY,
            fixture_path,
            fixture,
            include_persisted_id=True,
        )
        if fresh_tests[0]["stdout_sha256"] != persisted_tests[0]["stdout_sha256"]:
            _fail(FAIL_PERSIST, "persisted bridge binary replay differs")

        if (
            _sha256_directory_manifest(snapshot) != snapshot_manifest
            or _sha256_directory_manifest(vendor) != vendor_manifest
            or _commit_source_bindings_v1(commit, compare_worktree=True) != source_bindings
        ):
            _fail(FAIL_SOURCE, "qualification inputs changed during build/replay")
        if _sha256_file(RUNTIME_SECCOMP_PATH) != toolchain_policy["runtime_seccomp_sha256"]:
            _fail(FAIL_CONTAINER, "runtime seccomp changed during qualification")
        if _sha256_file(BUILD_SECCOMP_PATH) != toolchain_policy["build_seccomp_sha256"]:
            _fail(FAIL_CONTAINER, "build seccomp changed during qualification")
        return {
            "source": {
                "archive_domain": SOURCE_ARCHIVE_DOMAIN,
                "basis_commit": commit,
                "git_archive_exact": True,
                "worktree_bytes_equal_commit": True,
                "snapshot_read_only": True,
                "snapshot_manifest_sha256": snapshot_manifest,
                "bindings": source_bindings,
            },
            "dependency": {
                "cargo_lock_repository_path": f"{CRATE_REPOSITORY_PATH}/Cargo.lock",
                "cargo_lock_sha256": lock_sha256,
                "snapshot_domain": CARGO_SNAPSHOT_DOMAIN,
                "snapshot_root": dependency_root,
                "snapshot_file_count": dependency_file_count,
                "registry_package_count": dependency_package_count,
                "vendor_manifest_sha256": vendor_manifest,
                "locked_archive_checksums_verified": True,
                "host_cargo_cache_mounted_into_container": False,
            },
            "toolchain": {
                "approved_policy_repository_path": APPROVED_TOOLCHAIN_POLICY_REPOSITORY_PATH,
                "approved_policy_sha256": _sha256_file(
                    REPOSITORY_ROOT / APPROVED_TOOLCHAIN_POLICY_REPOSITORY_PATH
                ),
                "receipt": toolchain_receipt,
            },
            "container": {
                "docker_executable": "/usr/bin/docker",
                "docker_host": "unix:///var/run/docker.sock",
                "control_plane_binding": dict(control_plane.binding),
                "daemon_identity_receipt": daemon_receipt,
                "daemon_receipt_binding": daemon_binding,
                "image_ref": RUST_IMAGE_REF,
                "pull_policy": "never",
                "network_mode": "none",
                "read_only_root": True,
                "inherited_environment_allowed": False,
                "runtime_docker_policy_id": RUNTIME_DOCKER_POLICY_ID,
                "build_docker_policy_id": BUILD_DOCKER_POLICY_ID,
                "runtime_seccomp_sha256": _sha256_file(RUNTIME_SECCOMP_PATH),
                "build_seccomp_sha256": _sha256_file(BUILD_SECCOMP_PATH),
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
                "fresh_binary_sha256": built_sha256,
                "persisted_binary": persistence,
            },
            "replay_tests": {
                "fixture_repository_path": GOLDEN_FIXTURE_REPOSITORY_PATH,
                "fixture_sha256": _sha256_file(snapshot / GOLDEN_FIXTURE_REPOSITORY_PATH),
                "package_sha256": _sha256_bytes(fixture),
                "contains_private_key": False,
                "contains_signature": False,
                "contains_seed": False,
                "tests": [*fresh_tests, *persisted_tests],
                "all_passed": True,
            },
        }


def qualify_rust_bridge_dag_binary_v1(
    *, implementation_basis_commit: str | None = None
) -> dict[str, object]:
    """Build, replay-test, persist, and report one exact Commit-A binary."""

    commit = _require_commit(
        repository_head_commit_v1()
        if implementation_basis_commit is None
        else implementation_basis_commit
    )
    evidence = _fresh_qualification_v1(commit)
    report: dict[str, object] = {
        "artifact": "phase3_m25_bridge_dag_rust_binary_qualification_v1",
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": STATUS,
        "claim_level": CLAIM_LEVEL,
        "implementation_basis_commit": commit,
        **evidence,
        "authority_boundary": dict(AUTHORITY_BOUNDARY),
    }
    report["diagnostic_report_sha256"] = _report_sha256(report)
    validate_rust_bridge_dag_binary_qualification_report_v1(
        report, expected_basis_commit=commit
    )
    return report


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if type(value) is not dict or not all(type(key) is str for key in value):
        _fail(FAIL_REPORT, f"{label} is not a string-keyed object")
    return value


def validate_rust_bridge_dag_binary_qualification_report_v1(
    report: Mapping[str, object],
    *,
    expected_basis_commit: str | None = None,
    verify_commit_sources: bool = True,
    verify_persisted_binary: bool = True,
) -> str:
    """Validate a report and return its Commit-A-bound binary SHA-256."""

    value = dict(_mapping(report, "qualification report"))
    expected_fields = {
        "artifact",
        "schema_version",
        "artifact_kind",
        "status",
        "claim_level",
        "implementation_basis_commit",
        "source",
        "dependency",
        "toolchain",
        "container",
        "build",
        "replay_tests",
        "authority_boundary",
        "diagnostic_report_sha256",
    }
    if set(value) != expected_fields:
        _fail(FAIL_REPORT, "qualification report field set differs")
    exact = {
        "artifact": "phase3_m25_bridge_dag_rust_binary_qualification_v1",
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": STATUS,
        "claim_level": CLAIM_LEVEL,
        "authority_boundary": AUTHORITY_BOUNDARY,
    }
    for field, expected in exact.items():
        if value[field] != expected or type(value[field]) is not type(expected):
            _fail(FAIL_REPORT, f"qualification report {field} differs")
    commit = _require_commit(value["implementation_basis_commit"])
    if expected_basis_commit is not None and commit != _require_commit(expected_basis_commit):
        _fail(FAIL_COMMIT, "qualification report basis commit differs")
    supplied_id = _require_sha256(value["diagnostic_report_sha256"], "report self-ID")
    body = dict(value)
    del body["diagnostic_report_sha256"]
    if supplied_id != _report_sha256(body):
        _fail(FAIL_REPORT, "qualification report self-ID differs")

    source = _mapping(value["source"], "source evidence")
    if set(source) != {
        "archive_domain", "basis_commit", "git_archive_exact",
        "worktree_bytes_equal_commit", "snapshot_read_only",
        "snapshot_manifest_sha256", "bindings",
    }:
        _fail(FAIL_REPORT, "source evidence field set differs")
    if (
        source["archive_domain"] != SOURCE_ARCHIVE_DOMAIN
        or source["basis_commit"] != commit
        or any(
            source[field] is not True
            for field in ("git_archive_exact", "worktree_bytes_equal_commit", "snapshot_read_only")
        )
    ):
        _fail(FAIL_REPORT, "source evidence value differs")
    _require_sha256(source["snapshot_manifest_sha256"], "source snapshot manifest")
    bindings = _mapping(source["bindings"], "source bindings")
    if set(bindings) != set(QUALIFICATION_SOURCE_PATHS):
        _fail(FAIL_REPORT, "source binding path set differs")
    for path, digest in bindings.items():
        _require_sha256(digest, f"source binding {path}")
    if verify_commit_sources and dict(bindings) != _commit_source_bindings_v1(
        commit, compare_worktree=True
    ):
        _fail(FAIL_SOURCE, "report source bindings differ from Commit A")

    policy = _load_approved_toolchain_policy()
    dependency = _mapping(value["dependency"], "dependency evidence")
    if set(dependency) != {
        "cargo_lock_repository_path", "cargo_lock_sha256", "snapshot_domain",
        "snapshot_root", "snapshot_file_count", "registry_package_count",
        "vendor_manifest_sha256", "locked_archive_checksums_verified",
        "host_cargo_cache_mounted_into_container",
    }:
        _fail(FAIL_REPORT, "dependency evidence field set differs")
    if (
        dependency["cargo_lock_repository_path"] != f"{CRATE_REPOSITORY_PATH}/Cargo.lock"
        or dependency["cargo_lock_sha256"]
        != bindings[f"{CRATE_REPOSITORY_PATH}/Cargo.lock"]
        or dependency["snapshot_domain"] != CARGO_SNAPSHOT_DOMAIN
        or dependency["snapshot_root"] != policy["dependency_snapshot_root"]
        or dependency["snapshot_file_count"] != policy["dependency_snapshot_file_count"]
        or dependency["registry_package_count"] != policy["cargo_lock_registry_package_count"]
        or dependency["locked_archive_checksums_verified"] is not True
        or dependency["host_cargo_cache_mounted_into_container"] is not False
    ):
        _fail(FAIL_REPORT, "dependency evidence value differs")
    for field in ("cargo_lock_sha256", "snapshot_root", "vendor_manifest_sha256"):
        _require_sha256(dependency[field], f"dependency {field}")

    toolchain = _mapping(value["toolchain"], "toolchain evidence")
    if set(toolchain) != {
        "approved_policy_repository_path", "approved_policy_sha256", "receipt"
    } or toolchain["approved_policy_repository_path"] != APPROVED_TOOLCHAIN_POLICY_REPOSITORY_PATH:
        _fail(FAIL_REPORT, "toolchain evidence field set/value differs")
    if toolchain["approved_policy_sha256"] != _sha256_file(
        REPOSITORY_ROOT / APPROVED_TOOLCHAIN_POLICY_REPOSITORY_PATH
    ):
        _fail(FAIL_REPORT, "approved toolchain policy digest differs")
    receipt = _mapping(toolchain["receipt"], "toolchain receipt")
    expected_receipt = {
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
        "rustc_verbose_version_stdout_sha256": policy["rustc_verbose_version_stdout_sha256"],
        "runtime_environment_sha256": policy["runtime_environment_sha256"],
        "build_environment_sha256": policy["build_environment_sha256"],
        "runtime_seccomp_sha256": policy["runtime_seccomp_sha256"],
        "build_seccomp_sha256": policy["build_seccomp_sha256"],
        "image_config_environment_ignored": True,
        "pull_policy": "never",
        "network_mode": "none",
        "toolchain_receipt_is_external_attestation": False,
    }
    if dict(receipt) != expected_receipt:
        _fail(FAIL_REPORT, "toolchain receipt differs")

    container = _mapping(value["container"], "container evidence")
    if set(container) != {
        "docker_executable", "docker_host", "control_plane_binding",
        "daemon_identity_receipt", "daemon_receipt_binding", "image_ref",
        "pull_policy", "network_mode", "read_only_root",
        "inherited_environment_allowed", "runtime_docker_policy_id",
        "build_docker_policy_id", "runtime_seccomp_sha256", "build_seccomp_sha256",
    }:
        _fail(FAIL_REPORT, "container evidence field set differs")
    if (
        container["docker_executable"] != "/usr/bin/docker"
        or container["docker_host"] != "unix:///var/run/docker.sock"
        or container["image_ref"] != RUST_IMAGE_REF
        or container["pull_policy"] != "never"
        or container["network_mode"] != "none"
        or container["read_only_root"] is not True
        or container["inherited_environment_allowed"] is not False
        or container["runtime_docker_policy_id"] != RUNTIME_DOCKER_POLICY_ID
        or container["build_docker_policy_id"] != BUILD_DOCKER_POLICY_ID
        or container["runtime_seccomp_sha256"] != policy["runtime_seccomp_sha256"]
        or container["build_seccomp_sha256"] != policy["build_seccomp_sha256"]
    ):
        _fail(FAIL_REPORT, "container evidence value differs")
    daemon = _mapping(container["daemon_identity_receipt"], "daemon receipt")
    try:
        daemon_binding = local_docker_daemon_receipt_binding_v1(daemon).hex()
    except Phase3LocalRuntimeError as exc:
        _fail(FAIL_REPORT, f"daemon receipt differs: {exc}")
    if (
        container["daemon_receipt_binding"] != daemon_binding
        or container["control_plane_binding"] != daemon.get("control_plane_binding")
    ):
        _fail(FAIL_REPORT, "Docker control-plane binding differs")

    build = _mapping(value["build"], "build evidence")
    if set(build) != {
        "release_profile", "cargo_locked", "cargo_offline", "fresh_linux_local_target",
        "source_mount_read_only", "vendor_mount_read_only", "test_command",
        "build_command", "fresh_binary_sha256", "persisted_binary",
    } or any(
        build[field] is not True
        for field in (
            "release_profile", "cargo_locked", "cargo_offline", "fresh_linux_local_target",
            "source_mount_read_only", "vendor_mount_read_only",
        )
    ):
        _fail(FAIL_REPORT, "build evidence field set/value differs")
    if build["test_command"] != list(TEST_COMMAND) or build["build_command"] != list(BUILD_COMMAND):
        _fail(FAIL_REPORT, "normalized Cargo command differs")
    fresh_digest = _require_sha256(build["fresh_binary_sha256"], "fresh binary")
    persisted = _mapping(build["persisted_binary"], "persisted binary")
    if set(persisted) != {"repository_path", "sha256", "mode_octal", "atomic_replace", "is_symlink"}:
        _fail(FAIL_REPORT, "persisted binary field set differs")
    if persisted != {
        "repository_path": PERSISTED_BINARY_REPOSITORY_PATH,
        "sha256": fresh_digest,
        "mode_octal": "0755",
        "atomic_replace": True,
        "is_symlink": False,
    }:
        _fail(FAIL_REPORT, "persisted binary binding differs")

    replays = _mapping(value["replay_tests"], "replay evidence")
    if set(replays) != {
        "fixture_repository_path", "fixture_sha256", "package_sha256",
        "contains_private_key", "contains_signature", "contains_seed", "tests", "all_passed",
    }:
        _fail(FAIL_REPORT, "replay evidence field set differs")
    if (
        replays["fixture_repository_path"] != GOLDEN_FIXTURE_REPOSITORY_PATH
        or replays["fixture_sha256"] != _sha256_file(GOLDEN_FIXTURE_PATH)
        or replays["package_sha256"] != _sha256_bytes(load_unsigned_public_replay_fixture_v1())
        or replays["contains_private_key"] is not False
        or replays["contains_signature"] is not False
        or replays["contains_seed"] is not False
        or replays["all_passed"] is not True
    ):
        _fail(FAIL_REPORT, "replay evidence value differs")
    tests = replays["tests"]
    expected_test_ids = [
        "FRESH_PUBLIC_PURPOSE1_REPLAY_PASS",
        "PUBLIC_PREIMAGE_SUBSTITUTION_REJECTED",
        "PUBLIC_NODE_OMISSION_REJECTED",
        "AUTHORITATIVE_FLAG_WITHOUT_RUNTIME_OPT_IN_REJECTED",
        "PERSISTED_PUBLIC_PURPOSE1_REPLAY_PASS",
    ]
    if not isinstance(tests, list) or [row.get("test_id") if type(row) is dict else None for row in tests] != expected_test_ids:
        _fail(FAIL_REPORT, "replay test identity/order differs")
    expected_errors = [None, FAIL_ROOT_BINDING, FAIL_NODE_SET, FAIL_PACKAGE_AUTHORITY, None]
    for index, (row_value, error) in enumerate(zip(tests, expected_errors, strict=True)):
        row = _mapping(row_value, f"replay test {index}")
        if set(row) != {
            "test_id", "expected_returncode", "observed_returncode",
            "expected_error_code_or_null", "stdout_sha256", "stderr_sha256",
        } or row["expected_error_code_or_null"] != error:
            _fail(FAIL_REPORT, f"replay test {index} structure differs")
        expected_return = 0 if error is None else 1
        if row["expected_returncode"] != expected_return or row["observed_returncode"] != expected_return:
            _fail(FAIL_REPORT, f"replay test {index} return code differs")
        _require_sha256(row["stdout_sha256"], f"replay test {index} stdout")
        _require_sha256(row["stderr_sha256"], f"replay test {index} stderr")
    if tests[0]["stdout_sha256"] != tests[-1]["stdout_sha256"]:
        _fail(FAIL_REPORT, "fresh/persisted replay receipt differs")

    if verify_persisted_binary:
        path = DEFAULT_RUST_BRIDGE_DAG_BINARY
        try:
            metadata = path.lstat()
            exact_parent = path.parent.resolve(strict=True) == path.parent
        except OSError as exc:
            _fail(FAIL_PERSIST, f"persisted bridge binary is absent: {exc}")
        if (
            not exact_parent
            or stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o755
            or _sha256_file(path, code=FAIL_PERSIST) != fresh_digest
        ):
            _fail(FAIL_PERSIST, "persisted bridge binary differs from report")
    return fresh_digest


def canonical_qualification_report_bytes_v1(report: Mapping[str, object]) -> bytes:
    """Return the validated report as one canonical ASCII JSON line."""

    validate_rust_bridge_dag_binary_qualification_report_v1(report)
    return _canonical_json_bytes(dict(report))


def load_qualified_rust_bridge_dag_binary_binding_v1(
    *,
    expected_basis_commit: str,
    report_path: Path = DEFAULT_RUST_BRIDGE_DAG_QUALIFICATION_REPORT,
) -> tuple[Mapping[str, object], str]:
    """Load canonical evidence and return ``(report, binary_sha256)``.

    This is the narrow executor-facing API.  It rechecks Commit-A source
    bytes and the stable persisted binary; callers should then copy that exact
    binary into their private runtime snapshot and bind both returned values.
    """

    try:
        if report_path.resolve(strict=False) != DEFAULT_RUST_BRIDGE_DAG_QUALIFICATION_REPORT.resolve(strict=False):
            _fail(FAIL_REPORT, "executor loader accepts only the stable qualification report path")
        metadata = report_path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            _fail(FAIL_REPORT, "qualification report is not a regular file")
        payload = report_path.read_bytes()
    except OSError as exc:
        _fail(FAIL_REPORT, f"cannot read qualification report: {exc}")
    report = _strict_json_object(payload, label="bridge binary qualification report")
    digest = validate_rust_bridge_dag_binary_qualification_report_v1(
        report,
        expected_basis_commit=expected_basis_commit,
    )
    return report, digest


__all__ = [
    "ARTIFACT_KIND",
    "BridgeDagBinaryQualificationError",
    "CLAIM_LEVEL",
    "DEFAULT_RUST_BRIDGE_DAG_BINARY",
    "DEFAULT_RUST_BRIDGE_DAG_QUALIFICATION_REPORT",
    "GOLDEN_FIXTURE_PATH",
    "PERSISTED_BINARY_REPOSITORY_PATH",
    "QUALIFICATION_SOURCE_PATHS",
    "SCHEMA_VERSION",
    "STATUS",
    "canonical_qualification_report_bytes_v1",
    "load_unsigned_public_replay_fixture_v1",
    "load_qualified_rust_bridge_dag_binary_binding_v1",
    "qualify_rust_bridge_dag_binary_v1",
    "repository_head_commit_v1",
    "validate_rust_bridge_dag_binary_qualification_report_v1",
]
