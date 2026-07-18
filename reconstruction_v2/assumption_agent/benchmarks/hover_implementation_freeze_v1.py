"""Verify the committed implementation closure for the HoVer study.

The verifier is read-only.  It binds the manifest and every declared role to
both the working-tree bytes and the same Git ``HEAD``, and exposes a separate
check that controllers can use to close imported Python modules to those
verified role paths.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import hmac
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from types import ModuleType
from typing import Any


SCHEMA = "hover_joint_graph_implementation_freeze_v1"
VERSION = "hover_implementation_freeze_v1"
MANIFEST_RELATIVE_PATH = "manifests/hover_joint_graph_implementation_freeze_v1.json"
HASH_FIELD = "implementation_freeze_sha256"

# Keep the final role registry in one place.  Callers/tests may supply an exact
# alternate mapping while a new manifest is being assembled.
DEFAULT_EXPECTED_ROLE_PATHS: dict[str, str] = {
    "assumption_agent_package": "assumption_agent/__init__.py",
    "assumption_agent_models": "assumption_agent/models.py",
    "benchmarks_package": "assumption_agent/benchmarks/__init__.py",
    "acquisition": "assumption_agent/benchmarks/hover_direct_acquisition_v1.py",
    "controller": "assumption_agent/benchmarks/hover_joint_graph_formal_controller_v1.py",
    "design": "manifests/hover_joint_graph_evaluator_design_v1.json",
    "formal_runner": "assumption_agent/benchmarks/hover_joint_graph_formal_runner_v1.py",
    "implementation_freeze_verifier": "assumption_agent/benchmarks/hover_implementation_freeze_v1.py",
    "isolated_bootstrap": "assumption_agent/benchmarks/hover_isolated_bootstrap_v1.py",
    "lifecycle_store": "assumption_agent/benchmarks/hover_lifecycle_store_v1.py",
    "local_runtime": "assumption_agent/benchmarks/hover_local_runtime_v1.py",
    "global_hipporag_adapter": "replication_runtime/multihoprag_official_hipporag_v1/adapter.py",
    "global_hipporag_contract": "replication_runtime/multihoprag_official_hipporag_v1/contract.py",
    "global_hipporag_package": "replication_runtime/multihoprag_official_hipporag_v1/__init__.py",
    "global_hipporag_qualification": "manifests/multihoprag_official_hipporag_global_qualification_v1.json",
    "global_hipporag_worker": "replication_runtime/multihoprag_official_hipporag_v1/worker.py",
    "hipporag_adapter_v1": "replication_runtime/musique_official_hipporag_v1/adapter.py",
    "hipporag_adapter_v2": "replication_runtime/musique_official_hipporag_v1/adapter_v2.py",
    "hipporag_adapter_v3": "replication_runtime/musique_official_hipporag_v1/adapter_v3.py",
    "hipporag_binding_v1": "replication_runtime/musique_official_hipporag_v1/binding.py",
    "hipporag_contract_v1": "replication_runtime/musique_official_hipporag_v1/contract.py",
    "hipporag_package": "replication_runtime/musique_official_hipporag_v1/__init__.py",
    "hipporag_runtime_attestation_v2_code": "replication_runtime/musique_official_hipporag_v1/runtime_attestation_v2.py",
    "hipporag_runtime_attestation_v3_code": "replication_runtime/musique_official_hipporag_v1/runtime_attestation_v3.py",
    "hipporag_runtime_attestation": "manifests/musique_official_hipporag_runtime_attestation_v3.json",
    "hipporag_runtime_attestation_v2": "manifests/musique_official_hipporag_runtime_attestation_v2.json",
    "hipporag_runtime_binding_v1": "manifests/musique_official_hipporag_retrieve_only_binding_v1.json",
    "hipporag_runtime_binding_v2_code": "assumption_agent/benchmarks/musique_formal_runtime_binding_v2.py",
    "hipporag_runtime_binding_v3_code": "assumption_agent/benchmarks/musique_formal_runtime_binding_v3.py",
    "hipporag_runtime_qualification": "manifests/official_hipporag_runtime_adapter_qualification_v1.json",
    "hipporag_worker_v1": "replication_runtime/musique_official_hipporag_v1/worker.py",
    "minilm_asset_manifest": "manifests/qasper_minilm_runtime_asset_v1.json",
    "minilm_base_runtime_binding": "replication_runtime/qasper_minilm_v1/binding.py",
    "minilm_runtime_binding": "replication_runtime/multihoprag_minilm_v1/adapter.py",
    "minilm_base_runtime_package": "replication_runtime/qasper_minilm_v1/__init__.py",
    "minilm_runtime_package": "replication_runtime/multihoprag_minilm_v1/__init__.py",
    "ner_asset_manifest": "manifests/multihoprag_ner_runtime_asset_v1.json",
    "ner_binding": "replication_runtime/multihoprag_ner_v1/binding.py",
    "ner_contract": "replication_runtime/multihoprag_ner_v1/contract.py",
    "ner_package": "replication_runtime/multihoprag_ner_v1/__init__.py",
    "ner_worker": "replication_runtime/multihoprag_ner_v1/worker.py",
    "qualifier": "assumption_agent/benchmarks/hover_source_qualification_v1.py",
    "qualification_receipt": "manifests/hover_source_qualification_v1.json",
    "replication_runtime_package": "replication_runtime/__init__.py",
    "typed_core": "assumption_agent/benchmarks/multihoprag_typed_operator_v2.py",
}
DEFAULT_EXPECTED_ROLES = frozenset(DEFAULT_EXPECTED_ROLE_PATHS)

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class HoVerImplementationFreezeError(RuntimeError):
    """The implementation manifest, Git closure, or module origin drifted."""


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise HoVerImplementationFreezeError("value is not canonical JSON") from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def git_blob_sha1(raw: bytes) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {len(raw)}\0".encode("ascii"))
    digest.update(raw)
    return digest.hexdigest()


def _strict_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise HoVerImplementationFreezeError(
                    f"{label} contains a duplicate key"
                )
            output[key] = value
        return output

    def reject_constant(value: str) -> None:
        raise HoVerImplementationFreezeError(
            f"{label} contains non-finite {value}"
        )

    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except HoVerImplementationFreezeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise HoVerImplementationFreezeError(f"{label} is not strict JSON") from exc
    if not isinstance(payload, dict):
        raise HoVerImplementationFreezeError(f"{label} root is not an object")
    return payload


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise HoVerImplementationFreezeError("freeze path is unsafe")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise HoVerImplementationFreezeError("freeze path is unsafe")
    return value


def _canonical_project(value: str | Path) -> Path:
    supplied = Path(value)
    if supplied.is_symlink():
        raise HoVerImplementationFreezeError("project root is a symlink")
    try:
        root = supplied.resolve(strict=True)
    except OSError as exc:
        raise HoVerImplementationFreezeError("project root is unavailable") from exc
    if not root.is_dir():
        raise HoVerImplementationFreezeError("project root is not a directory")
    return root


def _repository_root(project: Path) -> Path:
    try:
        result = subprocess.run(
            ["git", "-C", str(project), "rev-parse", "--show-toplevel"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise HoVerImplementationFreezeError("Git repository is unavailable") from exc
    try:
        return Path(result.stdout.decode("utf-8", errors="strict").strip()).resolve(
            strict=True
        )
    except (OSError, UnicodeDecodeError) as exc:
        raise HoVerImplementationFreezeError("Git repository root is invalid") from exc


def _git_head(repository: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "--verify", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise HoVerImplementationFreezeError("Git HEAD is unavailable") from exc
    try:
        value = result.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise HoVerImplementationFreezeError("Git HEAD is malformed") from exc
    if _HEX40.fullmatch(value) is None:
        raise HoVerImplementationFreezeError("Git HEAD is malformed")
    return value


def _read_regular(path: Path, *, label: str) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise HoVerImplementationFreezeError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise HoVerImplementationFreezeError(f"{label} is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise HoVerImplementationFreezeError(f"{label} cannot be read") from exc
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or before.st_size != sum(len(chunk) for chunk in chunks)
    ):
        raise HoVerImplementationFreezeError(f"{label} changed while reading")
    return b"".join(chunks)


def _project_file_path(root: Path, relative: str, *, label: str) -> Path:
    """Reject symlinks in every path component before a frozen file read."""

    cursor = root
    parts = PurePosixPath(_safe_relative(relative)).parts
    for offset, part in enumerate(parts):
        cursor = cursor / part
        try:
            metadata = cursor.lstat()
        except OSError as exc:
            raise HoVerImplementationFreezeError(f"{label} is unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise HoVerImplementationFreezeError(
                f"{label} contains a symlink component"
            )
        if offset < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise HoVerImplementationFreezeError(
                f"{label} parent is not a directory"
            )
    return cursor


def _head_blobs(repository: Path, head: str, paths: tuple[str, ...]) -> dict[str, str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "ls-tree", "-r", "-z", head, "--", *paths],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise HoVerImplementationFreezeError("Git tree verification failed") from exc
    observed: dict[str, str] = {}
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        fields = metadata.split(b" ")
        if not separator or len(fields) != 3:
            raise HoVerImplementationFreezeError("Git tree output is malformed")
        mode, kind, raw_oid = fields
        try:
            path = raw_path.decode("utf-8", errors="strict")
            oid = raw_oid.decode("ascii", errors="strict")
        except UnicodeDecodeError as exc:
            raise HoVerImplementationFreezeError("Git tree output is malformed") from exc
        if (
            kind != b"blob"
            or mode != b"100644"
            or _HEX40.fullmatch(oid) is None
            or path in observed
        ):
            raise HoVerImplementationFreezeError("Git tree entry is unsafe")
        observed[path] = oid
    if set(observed) != set(paths):
        raise HoVerImplementationFreezeError("freeze files are not exactly present at HEAD")
    return observed


def verify_committed_implementation_freeze(
    project: str | Path,
    *,
    expected_role_paths: Mapping[str, str] = DEFAULT_EXPECTED_ROLE_PATHS,
    manifest_relative_path: str = MANIFEST_RELATIVE_PATH,
) -> dict[str, Any]:
    """Verify self-hash, exact roles, working bytes, HEAD blobs, and stable HEAD."""

    root = _canonical_project(project)
    repository = _repository_root(root)
    try:
        project_prefix = root.relative_to(repository)
    except ValueError as exc:
        raise HoVerImplementationFreezeError("project escaped its Git repository") from exc
    manifest_relative = _safe_relative(manifest_relative_path)
    expected = {role: _safe_relative(path) for role, path in expected_role_paths.items()}
    if (
        not expected
        or any(not isinstance(role, str) or not role for role in expected)
        or len(set(expected.values())) != len(expected)
        or manifest_relative in set(expected.values())
    ):
        raise HoVerImplementationFreezeError("expected role registry is invalid")

    head_before = _git_head(repository)
    manifest_path = _project_file_path(
        root, manifest_relative, label="implementation freeze"
    )
    manifest_raw = _read_regular(manifest_path, label="implementation freeze")
    manifest = _strict_json_object(manifest_raw, label="implementation freeze")
    if (
        set(manifest) != {"schema", "version", "bindings", HASH_FIELD}
        or manifest.get("schema") != SCHEMA
        or manifest.get("version") != "v1"
    ):
        raise HoVerImplementationFreezeError("implementation freeze schema drifted")
    declared_hash = manifest.get(HASH_FIELD)
    if not isinstance(declared_hash, str) or _HEX64.fullmatch(declared_hash) is None:
        raise HoVerImplementationFreezeError("implementation freeze self-hash is invalid")
    body = dict(manifest)
    del body[HASH_FIELD]
    if not hmac.compare_digest(declared_hash, stable_hash(body)):
        raise HoVerImplementationFreezeError("implementation freeze self-hash drifted")
    bindings = manifest.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(expected):
        raise HoVerImplementationFreezeError("implementation freeze role set drifted")

    working: dict[str, tuple[str, str]] = {}
    for role in sorted(expected):
        entry = bindings.get(role)
        if not isinstance(entry, Mapping) or set(entry) != {
            "relative_path",
            "file_sha256",
            "git_blob_sha1",
        }:
            raise HoVerImplementationFreezeError(f"{role} binding width drifted")
        relative = _safe_relative(entry.get("relative_path"))
        if relative != expected[role]:
            raise HoVerImplementationFreezeError(f"{role} path drifted")
        file_sha = entry.get("file_sha256")
        git_oid = entry.get("git_blob_sha1")
        if (
            not isinstance(file_sha, str)
            or _HEX64.fullmatch(file_sha) is None
            or not isinstance(git_oid, str)
            or _HEX40.fullmatch(git_oid) is None
        ):
            raise HoVerImplementationFreezeError(f"{role} hash binding is invalid")
        role_path = _project_file_path(
            root, relative, label=f"{role} implementation"
        )
        raw = _read_regular(role_path, label=f"{role} implementation")
        observed_sha = hashlib.sha256(raw).hexdigest()
        observed_oid = git_blob_sha1(raw)
        if not hmac.compare_digest(observed_sha, file_sha) or not hmac.compare_digest(
            observed_oid, git_oid
        ):
            raise HoVerImplementationFreezeError(f"{role} working bytes drifted")
        working[role] = (observed_sha, observed_oid)

    project_paths = (manifest_relative, *[expected[role] for role in sorted(expected)])
    repository_paths = tuple(
        (PurePosixPath(project_prefix.as_posix()) / path).as_posix()
        for path in project_paths
    )
    head_oids = _head_blobs(repository, head_before, repository_paths)
    if git_blob_sha1(manifest_raw) != head_oids[repository_paths[0]]:
        raise HoVerImplementationFreezeError("implementation freeze does not match HEAD")
    for offset, role in enumerate(sorted(expected), start=1):
        if working[role][1] != head_oids[repository_paths[offset]]:
            raise HoVerImplementationFreezeError(f"{role} does not byte-match HEAD")
    head_after = _git_head(repository)
    if head_after != head_before:
        raise HoVerImplementationFreezeError("Git HEAD drifted during verification")

    return {
        "schema": SCHEMA,
        HASH_FIELD: declared_hash,
        "implementation_freeze_file_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "implementation_freeze_git_blob_sha1": git_blob_sha1(manifest_raw),
        "verified_git_head": head_before,
        "required_role_count": len(expected),
        "required_role_set_sha256": stable_hash(sorted(expected)),
        "role_path_mapping_sha256": stable_hash(dict(sorted(expected.items()))),
        "role_paths": dict(sorted(expected.items())),
        "python_role_set_sha256": stable_hash(
            sorted(role for role, path in expected.items() if path.endswith(".py"))
        ),
        "all_bindings_byte_match_committed_HEAD": True,
        "git_HEAD_stable_during_verification": True,
    }


def verify_loaded_module_origins(
    *,
    project: str | Path,
    implementation_receipt: Mapping[str, Any],
    loaded_modules_by_role: Mapping[str, ModuleType | object],
    expected_roles: Sequence[str],
) -> dict[str, str]:
    """Close an exact controller-declared module set to frozen role paths."""

    root = _canonical_project(project)
    role_paths = implementation_receipt.get("role_paths")
    if (
        implementation_receipt.get("all_bindings_byte_match_committed_HEAD") is not True
        or implementation_receipt.get("git_HEAD_stable_during_verification") is not True
        or not isinstance(role_paths, Mapping)
        or implementation_receipt.get("required_role_set_sha256")
        != stable_hash(sorted(role_paths))
        or implementation_receipt.get("role_path_mapping_sha256")
        != stable_hash(dict(sorted(role_paths.items())))
    ):
        raise HoVerImplementationFreezeError("implementation receipt is not closed")
    expected = tuple(expected_roles)
    frozen_python_roles = {
        role for role, path in role_paths.items() if str(path).endswith(".py")
    }
    if (
        not expected
        or len(set(expected)) != len(expected)
        or set(loaded_modules_by_role) != set(expected)
        or not set(expected) <= frozen_python_roles
        or implementation_receipt.get("python_role_set_sha256")
        != stable_hash(sorted(frozen_python_roles))
    ):
        raise HoVerImplementationFreezeError("loaded module role set drifted")
    observed: dict[str, str] = {}
    for role, module in loaded_modules_by_role.items():
        relative = role_paths.get(role)
        if not isinstance(relative, str) or not relative.endswith(".py"):
            raise HoVerImplementationFreezeError(f"{role} is not a frozen Python role")
        expected = (root / _safe_relative(relative)).resolve(strict=True)
        module_file = getattr(module, "__file__", None)
        spec = getattr(module, "__spec__", None)
        origin = None if spec is None else getattr(spec, "origin", None)
        if not isinstance(module_file, str) or not isinstance(origin, str):
            raise HoVerImplementationFreezeError(f"{role} module origin is unavailable")
        if Path(module_file).resolve(strict=True) != expected or Path(origin).resolve(
            strict=True
        ) != expected:
            raise HoVerImplementationFreezeError(f"{role} module origin drifted")
        observed[role] = str(expected)
    return dict(sorted(observed.items()))


def verify_expected_git_head(
    *, project: str | Path, expected_git_head: str
) -> str:
    """Fail closed if the repository moved after prerequisite verification."""

    if _HEX40.fullmatch(expected_git_head) is None:
        raise HoVerImplementationFreezeError("expected Git HEAD is malformed")
    root = _canonical_project(project)
    observed = _git_head(_repository_root(root))
    if not hmac.compare_digest(observed, expected_git_head):
        raise HoVerImplementationFreezeError("Git HEAD moved after implementation freeze")
    return observed


def verify_no_unfrozen_project_modules(
    *, project: str | Path, implementation_receipt: Mapping[str, Any]
) -> dict[str, str]:
    """Reject any already-loaded project-local module outside the freeze."""

    root = _canonical_project(project)
    role_paths = implementation_receipt.get("role_paths")
    if not isinstance(role_paths, Mapping):
        raise HoVerImplementationFreezeError("implementation receipt has no role paths")
    allowed: set[Path] = set()
    for role, value in role_paths.items():
        if not isinstance(value, str) or not value.endswith(".py"):
            continue
        allowed.add(
            _project_file_path(
                root,
                value,
                label=f"{role} loaded-module implementation",
            ).resolve(strict=True)
        )
    if not allowed:
        raise HoVerImplementationFreezeError("frozen Python role set is empty")

    observed: dict[str, str] = {}
    for module_name, module in tuple(sys.modules.items()):
        if module is None:
            continue
        module_file = getattr(module, "__file__", None)
        spec = getattr(module, "__spec__", None)
        spec_origin = None if spec is None else getattr(spec, "origin", None)
        origins = {
            value
            for value in (module_file, spec_origin)
            if isinstance(value, str)
            and value not in {"built-in", "frozen"}
            and not value.startswith("<")
        }
        local_origins: set[Path] = set()
        for value in origins:
            try:
                path = Path(value).resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            try:
                path.relative_to(root)
            except ValueError:
                continue
            if path.suffix == ".pyc" and path.parent.name == "__pycache__":
                stem = path.name.split(".", 1)[0]
                source = path.parent.parent / f"{stem}.py"
                try:
                    path = source.resolve(strict=True)
                except (OSError, RuntimeError):
                    pass
            local_origins.add(path)
        if any(path not in allowed for path in local_origins):
            raise HoVerImplementationFreezeError(
                f"{module_name} loaded from an unfrozen project path"
            )
        if local_origins:
            if len(local_origins) != 1:
                raise HoVerImplementationFreezeError(
                    f"{module_name} module origins disagree"
                )
            observed[module_name] = str(next(iter(local_origins)))
    return dict(sorted(observed.items()))


def import_and_verify_frozen_python_roles(
    *, project: str | Path, implementation_receipt: Mapping[str, Any]
) -> dict[str, str]:
    """Import every frozen Python role and close its actual module origin."""

    role_paths = implementation_receipt.get("role_paths")
    if not isinstance(role_paths, Mapping):
        raise HoVerImplementationFreezeError("implementation receipt has no role paths")
    modules: dict[str, ModuleType] = {}
    for role, value in sorted(role_paths.items()):
        if not isinstance(value, str) or not value.endswith(".py"):
            continue
        pure = PurePosixPath(_safe_relative(value))
        parts = list(pure.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][:-3]
        if not parts or any(not part.isidentifier() for part in parts):
            raise HoVerImplementationFreezeError(f"{role} module path is invalid")
        module_name = ".".join(parts)
        try:
            modules[role] = importlib.import_module(module_name)
        except Exception as exc:
            raise HoVerImplementationFreezeError(
                f"{role} frozen module import failed"
            ) from exc
    verified = verify_loaded_module_origins(
        project=project,
        implementation_receipt=implementation_receipt,
        loaded_modules_by_role=modules,
        expected_roles=tuple(sorted(modules)),
    )
    verify_no_unfrozen_project_modules(
        project=project,
        implementation_receipt=implementation_receipt,
    )
    return verified


__all__ = [
    "DEFAULT_EXPECTED_ROLES",
    "DEFAULT_EXPECTED_ROLE_PATHS",
    "HASH_FIELD",
    "HoVerImplementationFreezeError",
    "MANIFEST_RELATIVE_PATH",
    "SCHEMA",
    "VERSION",
    "canonical_json_bytes",
    "git_blob_sha1",
    "import_and_verify_frozen_python_roles",
    "stable_hash",
    "verify_committed_implementation_freeze",
    "verify_expected_git_head",
    "verify_loaded_module_origins",
    "verify_no_unfrozen_project_modules",
]
