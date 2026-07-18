"""Formation and verification of the FEVEROUS P6/E2 implementation freeze."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from assumption_agent.benchmarks import (
    feverous_p6_e2_identity_compiler_qualification_v1 as identity_qualification,
)


VERSION = "feverous_p6_e2_implementation_freeze_v1"
SCHEMA = f"{VERSION}_manifest"
MANIFEST_RELATIVE = Path("manifests/feverous_p6_e2_implementation_freeze_v1.json")
DESIGN_RELATIVE = Path("manifests/feverous_p6_e2_evaluator_design_v1.json")
DESIGN_SHA256 = "6193646baca9e35820a5d157bc248012fbd478c89a45db7d879295c4d64f0181"
QUALIFICATION_RELATIVE = Path(
    "manifests/feverous_p6_e2_identity_compiler_qualification_v1.json"
)

BOUND_PATHS: dict[str, str] = {
    "wikipedia_source_qualification": (
        "assumption_agent/benchmarks/"
        "feverous_wikipedia_source_qualification_v1.py"
    ),
    "atomic_corpus": "assumption_agent/benchmarks/feverous_atomic_corpus_v1.py",
    "source_adapter": "assumption_agent/benchmarks/feverous_p6_e2_source_adapter_v1.py",
    "formal_source": "assumption_agent/benchmarks/feverous_p6_e2_formal_source_v1.py",
    "acquisition_core": "assumption_agent/benchmarks/feverous_p6_e2_acquisition_v1.py",
    "parallel_identity_selection": (
        "assumption_agent/benchmarks/"
        "feverous_p6_e2_parallel_identity_selection_v1.py"
    ),
    "formal_acquisition": "assumption_agent/benchmarks/feverous_p6_e2_formal_acquisition_v1.py",
    "formal_acquisition_entrypoint": (
        "assumption_agent/benchmarks/"
        "feverous_p6_e2_formal_acquisition_entrypoint_v1.py"
    ),
    "operator": "assumption_agent/benchmarks/feverous_p6_query_anchored_operator_v1.py",
    "semantic_tensor": "assumption_agent/benchmarks/feverous_offline_semantic_tensor_v1.py",
    "feature_producer": "assumption_agent/benchmarks/feverous_e2_feature_producer_v1.py",
    "evaluator": "assumption_agent/benchmarks/feverous_e2_evaluator_v1.py",
    "nli_runtime": "assumption_agent/benchmarks/feverous_nli_runtime_v1.py",
    "local_runtime": "assumption_agent/benchmarks/feverous_local_runtime_v1.py",
    "formal_runner": "assumption_agent/benchmarks/feverous_p6_e2_formal_runner_v1.py",
    "formal_controller": "assumption_agent/benchmarks/feverous_p6_e2_formal_controller_v1.py",
    "implementation_freeze_verifier": "assumption_agent/benchmarks/feverous_p6_e2_implementation_freeze_v1.py",
    "hipporag_contract": "replication_runtime/feverous_official_hipporag_v1/contract.py",
    "hipporag_adapter": "replication_runtime/feverous_official_hipporag_v1/adapter.py",
    "hipporag_worker": "replication_runtime/feverous_official_hipporag_v1/worker.py",
    "hipporag_package": "replication_runtime/feverous_official_hipporag_v1/__init__.py",
    "test_atomic_corpus": "tests/test_feverous_atomic_corpus_v1.py",
    "test_source_adapter": "tests/test_feverous_p6_e2_source_adapter_v1.py",
    "test_formal_source": "tests/test_feverous_p6_e2_formal_source_v1.py",
    "test_two_phase_source": "tests/test_feverous_p6_e2_two_phase_source_v1.py",
    "test_acquisition_core": "tests/test_feverous_p6_e2_acquisition_v1.py",
    "test_parallel_identity_selection": (
        "tests/test_feverous_p6_e2_parallel_identity_selection_v1.py"
    ),
    "test_formal_acquisition": "tests/test_feverous_p6_e2_formal_acquisition_v1.py",
    "test_formal_acquisition_entrypoint": (
        "tests/test_feverous_p6_e2_formal_acquisition_entrypoint_v1.py"
    ),
    "test_operator": "tests/test_feverous_p6_query_anchored_operator_v1.py",
    "test_semantic_tensor": "tests/test_feverous_offline_semantic_tensor_v1.py",
    "test_feature_producer": "tests/test_feverous_e2_feature_producer_v1.py",
    "test_evaluator": "tests/test_feverous_e2_evaluator_v1.py",
    "test_nli_runtime": "tests/test_feverous_nli_runtime_v1.py",
    "test_local_runtime": "tests/test_feverous_local_runtime_v1.py",
    "test_formal_runner": "tests/test_feverous_p6_e2_formal_runner_v1.py",
    "test_formal_controller": "tests/test_feverous_p6_e2_formal_controller_v1.py",
    "test_hipporag": "tests/test_feverous_official_hipporag_v1.py",
    "identity_compiler_qualification": QUALIFICATION_RELATIVE.as_posix(),
    "identity_compiler_qualification_verifier": (
        "assumption_agent/benchmarks/"
        "feverous_p6_e2_identity_compiler_qualification_v1.py"
    ),
    "test_identity_compiler_qualification": (
        "tests/test_feverous_p6_e2_identity_compiler_qualification_v1.py"
    ),
    "identity_performance_diagnostic": (
        "assumption_agent/benchmarks/"
        "feverous_p6_e2_identity_performance_diagnostic_v1.py"
    ),
    "test_identity_performance_diagnostic": (
        "tests/test_feverous_p6_e2_identity_performance_diagnostic_v1.py"
    ),
    "identity_performance_diagnostic_receipt": (
        "manifests/feverous_p6_e2_identity_performance_diagnostic_v1.json"
    ),
    "identity_parallel_performance_diagnostic": (
        "assumption_agent/benchmarks/"
        "feverous_p6_e2_identity_parallel_performance_diagnostic_v1.py"
    ),
    "test_identity_parallel_performance_diagnostic": (
        "tests/test_feverous_p6_e2_identity_parallel_performance_diagnostic_v1.py"
    ),
    "identity_parallel_performance_diagnostic_receipt": (
        "manifests/"
        "feverous_p6_e2_identity_parallel_performance_diagnostic_v1.json"
    ),
    "test_implementation_freeze": (
        "tests/test_feverous_p6_e2_implementation_freeze_v1.py"
    ),
}

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA1 = re.compile(r"[0-9a-f]{40}\Z")


class FeverousImplementationFreezeError(RuntimeError):
    """The committed implementation or its qualification binding drifted."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise FeverousImplementationFreezeError("value is not canonical JSON") from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise FeverousImplementationFreezeError("bound file cannot be hashed") from exc
    return digest.hexdigest()


def _git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise FeverousImplementationFreezeError(f"{field} is not a SHA-256")
    return value


def _valid_test_receipt(value: object) -> bool:
    """Return whether a receipt proves a nonempty passing implementation suite."""

    return (
        isinstance(value, Mapping)
        and value.get("status") == "passed"
        and type(value.get("test_count")) is int
        and value["test_count"] > 0
    )


def _project(project: str | Path) -> Path:
    try:
        root = Path(project).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FeverousImplementationFreezeError("project root is unavailable") from exc
    if not root.is_dir():
        raise FeverousImplementationFreezeError("project root is not a directory")
    return root


def _git(project: Path, *arguments: str, binary: bool = False) -> str | bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FeverousImplementationFreezeError("Git binding check failed") from exc
    if binary:
        return completed.stdout
    try:
        return completed.stdout.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise FeverousImplementationFreezeError("Git output is not ASCII") from exc


def _git_is_ancestor(project: Path, ancestor: str, descendant: str) -> bool:
    try:
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=project,
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise FeverousImplementationFreezeError("Git binding check failed") from exc
    if completed.returncode == 0:
        return True
    if completed.returncode == 1:
        return False
    raise FeverousImplementationFreezeError("Git binding check failed")


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FeverousImplementationFreezeError("bound manifest is unavailable")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeverousImplementationFreezeError("bound manifest is invalid") from exc
    if not isinstance(value, dict):
        raise FeverousImplementationFreezeError("bound manifest is not an object")
    return value


def _verify_design(project: Path) -> str:
    design = _load_json(project / DESIGN_RELATIVE)
    body = dict(design)
    declared = body.pop("design_sha256", None)
    if declared != DESIGN_SHA256 or stable_hash(body) != DESIGN_SHA256:
        raise FeverousImplementationFreezeError("FEVEROUS design binding drifted")
    return DESIGN_SHA256


def _qualification_sha256(project: Path) -> str:
    try:
        qualification = identity_qualification.verify_identity_compiler_qualification(
            project
        )
    except identity_qualification.FeverousIdentityCompilerQualificationError as exc:
        raise FeverousImplementationFreezeError(
            "identity/compiler qualification binding drifted"
        ) from exc
    declared = qualification.get("qualification_sha256")
    return _require_sha256(declared, "identity/compiler qualification")


def _bound_file_rows(project: Path, implementation_commit: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for role, relative_text in BOUND_PATHS.items():
        relative = Path(relative_text)
        path = project / relative
        if path.is_symlink() or not path.is_file():
            raise FeverousImplementationFreezeError(f"bound role is unavailable: {role}")
        raw = path.read_bytes()
        committed = _git(
            project,
            "show",
            f"{implementation_commit}:{relative.as_posix()}",
            binary=True,
        )
        assert isinstance(committed, bytes)
        if committed != raw:
            raise FeverousImplementationFreezeError(
                f"bound role differs from implementation commit: {role}"
            )
        rows.append(
            {
                "role": role,
                "relative_path": relative.as_posix(),
                "size_bytes": len(raw),
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "git_blob_sha1": _git_blob_sha1(raw),
            }
        )
    return rows


def form_implementation_freeze(
    *,
    project: str | Path,
    test_receipt: Mapping[str, Any],
    runtime_preflight_sha256: str,
) -> dict[str, Any]:
    """Form (but do not write) the manifest from one clean implementation commit."""

    root = _project(project)
    _verify_design(root)
    qualification_sha = _qualification_sha256(root)
    preflight_sha = _require_sha256(runtime_preflight_sha256, "runtime preflight")
    head = _git(root, "rev-parse", "HEAD")
    if not isinstance(head, str) or _GIT_SHA1.fullmatch(head) is None:
        raise FeverousImplementationFreezeError("implementation Git HEAD is invalid")
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "--",
        *(Path(value).as_posix() for value in BOUND_PATHS.values()),
    )
    if status:
        raise FeverousImplementationFreezeError("bound implementation paths are dirty")
    if not _valid_test_receipt(test_receipt):
        raise FeverousImplementationFreezeError("implementation test receipt is invalid")
    files = _bound_file_rows(root, head)
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "implementation_frozen_before_formal_secret_or_train_acquisition",
        "design_sha256": DESIGN_SHA256,
        "identity_compiler_qualification_sha256": qualification_sha,
        "runtime_preflight_sha256": preflight_sha,
        "implementation_git_commit": head,
        "bound_files": files,
        "bound_role_set_sha256": stable_hash(sorted(BOUND_PATHS)),
        "bound_file_set_sha256": stable_hash(files),
        "test_receipt": dict(test_receipt),
        "formal_selection_secret_generated": False,
        "formal_train_acquisition_started": False,
        "development_or_test_source_accessed": False,
        "online_evaluator_calls": 0,
    }
    return {**body, "implementation_freeze_sha256": stable_hash(body)}


def verify_committed_implementation_freeze(
    project: str | Path,
) -> Mapping[str, Any]:
    """Verify manifest semantics, implementation commit blobs, and clean paths."""

    root = _project(project)
    _verify_design(root)
    manifest = _load_json(root / MANIFEST_RELATIVE)
    body = dict(manifest)
    declared = _require_sha256(
        body.pop("implementation_freeze_sha256", None),
        "implementation freeze",
    )
    implementation_commit = manifest.get("implementation_git_commit")
    files = manifest.get("bound_files")
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("version") != VERSION
        or manifest.get("status")
        != "implementation_frozen_before_formal_secret_or_train_acquisition"
        or stable_hash(body) != declared
        or manifest.get("design_sha256") != DESIGN_SHA256
        or manifest.get("identity_compiler_qualification_sha256")
        != _qualification_sha256(root)
        or not _valid_test_receipt(manifest.get("test_receipt"))
        or not isinstance(manifest.get("runtime_preflight_sha256"), str)
        or _SHA256.fullmatch(manifest["runtime_preflight_sha256"]) is None
        or not isinstance(implementation_commit, str)
        or _GIT_SHA1.fullmatch(implementation_commit) is None
        or not isinstance(files, list)
        or manifest.get("bound_role_set_sha256") != stable_hash(sorted(BOUND_PATHS))
        or manifest.get("bound_file_set_sha256") != stable_hash(files)
        or manifest.get("formal_selection_secret_generated") is not False
        or manifest.get("formal_train_acquisition_started") is not False
        or manifest.get("development_or_test_source_accessed") is not False
        or manifest.get("online_evaluator_calls") != 0
    ):
        raise FeverousImplementationFreezeError("implementation freeze semantics drifted")
    roles = {row.get("role") for row in files if isinstance(row, Mapping)}
    if roles != set(BOUND_PATHS) or len(files) != len(BOUND_PATHS):
        raise FeverousImplementationFreezeError("implementation freeze role set drifted")
    expected = _bound_file_rows(root, implementation_commit)
    if files != expected:
        raise FeverousImplementationFreezeError("bound implementation content drifted")
    if not _git_is_ancestor(root, implementation_commit, "HEAD"):
        raise FeverousImplementationFreezeError("implementation commit is not an ancestor")
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "--",
        *(Path(value).as_posix() for value in BOUND_PATHS.values()),
        MANIFEST_RELATIVE.as_posix(),
    )
    if status:
        raise FeverousImplementationFreezeError("formal implementation paths are dirty")
    tracked = _git(root, "ls-files", "--error-unmatch", MANIFEST_RELATIVE.as_posix())
    if tracked != MANIFEST_RELATIVE.as_posix():
        raise FeverousImplementationFreezeError("freeze manifest is not committed")
    return manifest


__all__ = [
    "BOUND_PATHS",
    "DESIGN_SHA256",
    "FeverousImplementationFreezeError",
    "MANIFEST_RELATIVE",
    "QUALIFICATION_RELATIVE",
    "SCHEMA",
    "VERSION",
    "form_implementation_freeze",
    "stable_hash",
    "verify_committed_implementation_freeze",
]
