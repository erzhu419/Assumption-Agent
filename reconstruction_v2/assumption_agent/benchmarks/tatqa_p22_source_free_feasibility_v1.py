"""One-shot source-free feasibility check for a possible P22 study.

This diagnostic is deliberately neither P21/P22 qualification nor efficacy
evidence.  It reuses the already-frozen P21 fingerprint, Qwen and official
HippoRAG systemd capabilities, but substitutes the independently named
portable MiniLM startup binding.  Only the fixed public synthetic canary is
representable; no TAT-QA source path is accepted by this entry point.  A P22
study may be frozen only after this independent one-shot feasibility passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

from assumption_agent.benchmarks import tatqa_p21_public_canary_v1 as canary
from replication_runtime.qasper_minilm_portable_v2.binding import (
    PortableOfflineMiniLMEncoder,
)
from replication_runtime.qasper_minilm_v1 import binding as frozen_minilm
from replication_runtime.tatqa_p21_v1 import formal_runtime


VERSION = "tatqa_p22_source_free_feasibility_v1"
MARKER_FILENAME = "feasibility.one_shot_marker.json"
SUCCESS_FILENAME = "feasibility.terminal_success.json"
FAILURE_FILENAME = "feasibility.terminal_failure.json"
PORTABLE_CAPABILITY_SCHEMA = (
    "tatqa_p22_portable_minilm_capability_receipt_snapshot_v1"
)
P21_IMPLEMENTATION_REGISTRY_SHA256 = (
    "c11193a43eb2c59b91d0ea9f91e47d0b665e2d8f8936d34d35ad36cb76f5fa00"
)
P22_ADDITIONAL_IMPLEMENTATION_PATHS = frozenset(
    {
        "assumption_agent/benchmarks/tatqa_p22_source_free_feasibility_v1.py",
        "replication_runtime/qasper_minilm_portable_v2/__init__.py",
        "replication_runtime/qasper_minilm_portable_v2/binding.py",
        "tests/test_qasper_minilm_portable_v2.py",
        "tests/test_tatqa_p22_source_free_feasibility_v1.py",
    }
)
REQUIRED_IMPLEMENTATION_PATHS = frozenset(
    canary.acquisition.REQUIRED_IMPLEMENTATION_PATHS
) | P22_ADDITIONAL_IMPLEMENTATION_PATHS
REQUIRED_EVIDENCE_PATHS = frozenset(
    {
        "manifests/tatqa_p19_hipporag_runtime_attestation_v1.json",
        "manifests/tatqa_p21_composite_runtime_fingerprint_v1.json",
    }
)
SOURCE_ISOLATION_ROOTS = (
    "artifacts/tatqa_p21_official_source_v1",
    "artifacts/tatqa_p21_formal_v1",
    "artifacts/tatqa_p22_official_source_v1",
    "artifacts/tatqa_p22_formal_v1",
)
SOURCE_ISOLATION_SENTINEL_NAME = ".p22-source-free-isolation-sentinel"
SOURCE_ISOLATION_SENTINEL_BYTES = b"P22 SOURCE-FREE ISOLATION SENTINEL V1\n"
SOURCE_ISOLATION_SENTINEL_PATHS = frozenset(
    f"{root}/{SOURCE_ISOLATION_SENTINEL_NAME}"
    for root in SOURCE_ISOLATION_ROOTS
)
REQUIRED_SNAPSHOT_PATHS = (
    REQUIRED_IMPLEMENTATION_PATHS
    | REQUIRED_EVIDENCE_PATHS
    | SOURCE_ISOLATION_SENTINEL_PATHS
)
REQUIRED_ENTRY_MODULE_NAME = (
    "assumption_agent.benchmarks.tatqa_p22_source_free_feasibility_v1"
)
EXPECTED_NORMALIZER_CALLABLE_NAME: str | None = None
EXPECTED_HOST_ROOT = Path("/home/erzhu419/p22_source_free_feasibility_20260723")
EXPECTED_PROJECT_ROOT = EXPECTED_HOST_ROOT / "runtime/reconstruction_v2"
EXPECTED_FEASIBILITY_ROOT = EXPECTED_HOST_ROOT / "attempt"
EXPECTED_WORK_ROOT = EXPECTED_FEASIBILITY_ROOT / "work"
EXPECTED_PUBLIC_CANARY_OUTPUT = EXPECTED_FEASIBILITY_ROOT / "public-canary.json"
EXPECTED_OUTER_UNIT = "p22-source-free-feasibility-c1-v1.service"
GIT_EXECUTABLE = Path(
    "/home/erzhu419/p22_runtime_tools_20260723/git/bin/git"
)
GIT_EXECUTABLE_SHA256 = (
    "2a8c18fbf43da9f692d75474c72bea9dfd796c260b0f3dfe456376abc3bbd668"
)
GIT_VERSION_STDOUT = b"git version 2.43.0\n"
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class TatqaP22SourceFreeFeasibilityError(RuntimeError):
    """The independent source-free feasibility attempt failed closed."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise TatqaP22SourceFreeFeasibilityError(
            "feasibility receipt is not canonical JSON"
        ) from exc


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _json_copy(value: object) -> object:
    try:
        return json.loads(_canonical_bytes(value).decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:  # pragma: no cover
        raise TatqaP22SourceFreeFeasibilityError(
            "capability evidence is not JSON-safe"
        ) from exc


def _validated_auxiliary_receipt(value: object) -> dict[str, object]:
    """Copy and verify an optional source-free phase receipt."""

    copied = _json_copy(value)
    if not isinstance(copied, dict):
        raise TatqaP22SourceFreeFeasibilityError(
            "post-MiniLM environment normalization receipt is absent"
        )
    body = dict(copied)
    declared = body.pop("self_sha256", None)
    if (
        not isinstance(declared, str)
        or _HEX64.fullmatch(declared) is None
        or _semantic_hash(body) != declared
    ):
        raise TatqaP22SourceFreeFeasibilityError(
            "post-MiniLM environment normalization receipt drifted"
        )
    return copied


def _post_minilm_normalizer_binding(
    project: Path,
    normalizer: Callable[[], Mapping[str, object]] | None,
) -> dict[str, object] | None:
    """Bind the only optional callback to the frozen entry module object."""

    if EXPECTED_NORMALIZER_CALLABLE_NAME is None:
        if normalizer is not None:
            raise TatqaP22SourceFreeFeasibilityError(
                "post-MiniLM normalizer was not preregistered"
            )
        return None
    module = sys.modules.get(REQUIRED_ENTRY_MODULE_NAME)
    if (
        normalizer is None
        or module is None
        or getattr(normalizer, "__module__", None) != REQUIRED_ENTRY_MODULE_NAME
        or getattr(normalizer, "__name__", None)
        != EXPECTED_NORMALIZER_CALLABLE_NAME
        or getattr(normalizer, "__qualname__", None)
        != EXPECTED_NORMALIZER_CALLABLE_NAME
        or getattr(module, EXPECTED_NORMALIZER_CALLABLE_NAME, None) is not normalizer
    ):
        raise TatqaP22SourceFreeFeasibilityError(
            "post-MiniLM normalizer callable identity drifted"
        )
    raw_module_path = getattr(module, "__file__", None)
    if not isinstance(raw_module_path, str):
        raise TatqaP22SourceFreeFeasibilityError(
            "post-MiniLM normalizer module path is absent"
        )
    try:
        module_path = Path(raw_module_path).resolve(strict=True)
        relative = module_path.relative_to(project).as_posix()
    except (OSError, ValueError) as exc:
        raise TatqaP22SourceFreeFeasibilityError(
            "post-MiniLM normalizer module is outside the snapshot"
        ) from exc
    if relative not in REQUIRED_IMPLEMENTATION_PATHS or module_path.suffix != ".py":
        raise TatqaP22SourceFreeFeasibilityError(
            "post-MiniLM normalizer module is outside the implementation registry"
        )
    body: dict[str, object] = {
        "schema": f"{VERSION}_post_minilm_normalizer_binding_v1",
        "module": REQUIRED_ENTRY_MODULE_NAME,
        "callable": EXPECTED_NORMALIZER_CALLABLE_NAME,
        "relative_module_path": relative,
        "module_file_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
    }
    return {**body, "self_sha256": _semantic_hash(body)}


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    raw = _canonical_bytes(value)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise TatqaP22SourceFreeFeasibilityError(
            "exclusive feasibility receipt already exists"
        ) from exc
    finally:
        if "descriptor" in locals() and descriptor >= 0:
            os.close(descriptor)
    if (
        path.is_symlink()
        or path.read_bytes() != raw
        or stat.S_IMODE(path.stat().st_mode) != 0o600
    ):
        raise TatqaP22SourceFreeFeasibilityError(
            "feasibility receipt reopen or mode drifted"
        )
    return hashlib.sha256(raw).hexdigest()


def _git(project: Path, arguments: Sequence[str]) -> bytes:
    if GIT_EXECUTABLE.is_symlink() or not GIT_EXECUTABLE.is_file():
        raise TatqaP22SourceFreeFeasibilityError("exact Git executable is unavailable")
    environment = {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_EXEC_PATH": "/nonexistent-p22-source-free-git-exec-path",
        "HOME": "/nonexistent-p22-source-free-git-home",
        "LANG": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
    }
    try:
        completed = subprocess.run(
            [
                str(GIT_EXECUTABLE),
                "--no-replace-objects",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.untrackedCache=false",
                *arguments,
            ],
            cwd=project,
            check=False,
            capture_output=True,
            env=environment,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TatqaP22SourceFreeFeasibilityError(
            "diagnostic snapshot Git verification failed"
        ) from exc
    if completed.returncode != 0 or completed.stderr:
        raise TatqaP22SourceFreeFeasibilityError(
            "diagnostic snapshot Git verification failed"
        )
    return completed.stdout


def _git_tool_binding() -> dict[str, object]:
    if GIT_EXECUTABLE.is_symlink() or not GIT_EXECUTABLE.is_file():
        raise TatqaP22SourceFreeFeasibilityError("exact Git executable is unavailable")
    raw = GIT_EXECUTABLE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != GIT_EXECUTABLE_SHA256:
        raise TatqaP22SourceFreeFeasibilityError("exact Git executable drifted")
    try:
        completed = subprocess.run(
            [str(GIT_EXECUTABLE), "--version"],
            check=False,
            capture_output=True,
            env={
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_EXEC_PATH": "/nonexistent-p22-source-free-git-exec-path",
                "GIT_NO_REPLACE_OBJECTS": "1",
                "HOME": "/nonexistent-p22-source-free-git-home",
                "LANG": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
            },
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TatqaP22SourceFreeFeasibilityError(
            "exact Git executable version check failed"
        ) from exc
    if (
        completed.returncode != 0
        or completed.stdout != GIT_VERSION_STDOUT
        or completed.stderr
    ):
        raise TatqaP22SourceFreeFeasibilityError(
            "exact Git executable version drifted"
        )
    body: dict[str, object] = {
        "schema": f"{VERSION}_git_tool_binding_v1",
        "executable_file_sha256": GIT_EXECUTABLE_SHA256,
        "version_stdout_sha256": hashlib.sha256(GIT_VERSION_STDOUT).hexdigest(),
        "external_exec_path_disabled": True,
    }
    return {**body, "self_sha256": _semantic_hash(body)}


def _filesystem_snapshot_closure(project: Path) -> dict[str, object]:
    """Reject ignored/untracked injection by enumerating the real worktree."""

    expected_directories: set[str] = set()
    for relative in REQUIRED_SNAPSHOT_PATHS:
        parent = Path(relative).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for current, directories, files in os.walk(project, followlinks=False):
        base = Path(current)
        if base == project:
            directories[:] = [name for name in directories if name != ".git"]
        for name in list(directories):
            path = base / name
            if path.is_symlink() or not path.is_dir():
                raise TatqaP22SourceFreeFeasibilityError(
                    "diagnostic snapshot contains a symlink or non-directory"
                )
            observed_directories.add(path.relative_to(project).as_posix())
        for name in files:
            path = base / name
            if path.is_symlink() or not path.is_file():
                raise TatqaP22SourceFreeFeasibilityError(
                    "diagnostic snapshot contains a symlink or non-file"
                )
            observed_files.add(path.relative_to(project).as_posix())
    if (
        observed_files != REQUIRED_SNAPSHOT_PATHS
        or observed_directories != expected_directories
    ):
        raise TatqaP22SourceFreeFeasibilityError(
            "diagnostic snapshot filesystem closure drifted"
        )
    body: dict[str, object] = {
        "schema": f"{VERSION}_filesystem_snapshot_closure_v1",
        "directory_count": len(observed_directories),
        "file_count": len(observed_files),
        "relative_directory_set_sha256": _semantic_hash(
            sorted(observed_directories)
        ),
        "relative_file_set_sha256": _semantic_hash(sorted(observed_files)),
        "symlink_or_nonregular_count": 0,
    }
    return {**body, "self_sha256": _semantic_hash(body)}


def _loaded_project_module_binding(project: Path) -> dict[str, object]:
    expected_harness = (
        project / "assumption_agent/benchmarks/tatqa_p22_source_free_feasibility_v1.py"
    ).resolve(strict=True)
    if Path(__file__).resolve(strict=True) != expected_harness:
        raise TatqaP22SourceFreeFeasibilityError(
            "source-free harness was not loaded from the snapshot project"
        )
    modules: dict[str, dict[str, object]] = {}
    for name, module in sorted(sys.modules.items()):
        if not (
            name == "assumption_agent"
            or name.startswith("assumption_agent.")
            or name == "replication_runtime"
            or name.startswith("replication_runtime.")
        ):
            continue
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str):
            raise TatqaP22SourceFreeFeasibilityError(
                "loaded project module lacks a concrete source file"
            )
        try:
            path = Path(raw_path).resolve(strict=True)
            relative = path.relative_to(project).as_posix()
        except (OSError, ValueError) as exc:
            raise TatqaP22SourceFreeFeasibilityError(
                "project module was loaded outside the snapshot project"
            ) from exc
        if relative not in REQUIRED_IMPLEMENTATION_PATHS or path.suffix != ".py":
            raise TatqaP22SourceFreeFeasibilityError(
                "loaded project module is outside the frozen implementation registry"
            )
        raw = path.read_bytes()
        modules[name] = {
            "relative_path": relative,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
        }
    required_loaded = {
        "assumption_agent",
        "assumption_agent.benchmarks",
        "assumption_agent.benchmarks.tatqa_p21_acquisition_v1",
        "assumption_agent.benchmarks.tatqa_p21_label_free_runtime_v1",
        "assumption_agent.benchmarks.tatqa_p21_public_canary_v1",
        "assumption_agent.benchmarks.tatqa_p21_typed_evaluator_core_v1",
        "replication_runtime",
        "replication_runtime.qasper_minilm_portable_v2",
        "replication_runtime.qasper_minilm_portable_v2.binding",
        "replication_runtime.qasper_minilm_v1",
        "replication_runtime.qasper_minilm_v1.binding",
        "replication_runtime.tatqa_p21_v1",
        "replication_runtime.tatqa_p21_v1.formal_runtime",
        "replication_runtime.tatqa_p21_v1.hipporag_contract",
        "replication_runtime.tatqa_p21_v1.runtime_attestation_v1",
        "replication_runtime.tatqa_p21_v1.typed_plan_contract",
    }
    required_loaded.add(REQUIRED_ENTRY_MODULE_NAME)
    if not required_loaded.issubset(modules):
        raise TatqaP22SourceFreeFeasibilityError(
            "required project module was not loaded from the snapshot"
        )
    body: dict[str, object] = {
        "schema": f"{VERSION}_loaded_project_modules_v1",
        "module_count": len(modules),
        "modules": modules,
    }
    return {**body, "self_sha256": _semantic_hash(body)}


def _diagnostic_snapshot_binding(
    project: Path, snapshot_commit: str
) -> dict[str, object]:
    """Prove a real clean minimal Git commit and every executable project file."""

    git_tool = _git_tool_binding()
    if _HEX40.fullmatch(snapshot_commit) is None:
        raise TatqaP22SourceFreeFeasibilityError(
            "diagnostic snapshot commit is not an exact Git object id"
        )
    if _semantic_hash(sorted(canary.acquisition.REQUIRED_IMPLEMENTATION_PATHS)) != (
        P21_IMPLEMENTATION_REGISTRY_SHA256
    ):
        raise TatqaP22SourceFreeFeasibilityError(
            "inherited P21 implementation registry drifted"
        )
    git_root = project / ".git"
    if git_root.is_symlink() or not git_root.is_dir():
        raise TatqaP22SourceFreeFeasibilityError(
            "diagnostic snapshot Git object store is unavailable"
        )
    filesystem_closure = _filesystem_snapshot_closure(project)
    repository = Path(
        _git(project, ["rev-parse", "--show-toplevel"]).decode("utf-8").strip()
    ).resolve(strict=True)
    if repository != project:
        raise TatqaP22SourceFreeFeasibilityError(
            "diagnostic snapshot repository root drifted"
        )
    resolved = _git(
        project, ["rev-parse", "--verify", f"{snapshot_commit}^{{commit}}"]
    ).decode("ascii").strip()
    head = _git(project, ["rev-parse", "--verify", "HEAD^{commit}"]).decode(
        "ascii"
    ).strip()
    if resolved != snapshot_commit or head != snapshot_commit:
        raise TatqaP22SourceFreeFeasibilityError(
            "diagnostic snapshot HEAD is not the frozen commit"
        )
    tracked = {
        value.decode("utf-8")
        for value in _git(project, ["ls-files", "-z"]).split(b"\0")
        if value
    }
    if tracked != REQUIRED_SNAPSHOT_PATHS:
        raise TatqaP22SourceFreeFeasibilityError(
            "diagnostic snapshot tracked file registry drifted"
        )
    if _git(project, ["status", "--porcelain=v1", "--untracked-files=all", "-z"]):
        raise TatqaP22SourceFreeFeasibilityError(
            "diagnostic snapshot worktree is not exactly clean"
        )
    files: list[dict[str, object]] = []
    for relative in sorted(REQUIRED_SNAPSHOT_PATHS):
        path = project / relative
        if path.is_symlink() or not path.is_file():
            raise TatqaP22SourceFreeFeasibilityError(
                "diagnostic snapshot member is unavailable"
            )
        raw = path.read_bytes()
        committed = _git(project, ["show", f"{snapshot_commit}:{relative}"])
        if raw != committed:
            raise TatqaP22SourceFreeFeasibilityError(
                "diagnostic snapshot member differs from the frozen commit"
            )
        files.append(
            {
                "relative_path": relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
        )
    tree = _git(project, ["show", "-s", "--format=%T", snapshot_commit]).decode(
        "ascii"
    ).strip()
    if _HEX40.fullmatch(tree) is None:
        raise TatqaP22SourceFreeFeasibilityError(
            "diagnostic snapshot tree object drifted"
        )
    loaded = _loaded_project_module_binding(project)
    body: dict[str, object] = {
        "schema": f"{VERSION}_implementation_snapshot_binding_v1",
        "diagnostic_snapshot_commit": snapshot_commit,
        "diagnostic_snapshot_tree": tree,
        "file_count": len(files),
        "files": files,
        "filesystem_snapshot_closure": filesystem_closure,
        "git_tool_binding": git_tool,
        "loaded_project_modules": loaded,
        "minimal_tracked_registry_exact": True,
        "worktree_clean": True,
    }
    return {**body, "self_sha256": _semantic_hash(body)}


def _source_isolation_receipt(project: Path) -> dict[str, object]:
    roots: list[dict[str, object]] = []
    for relative in SOURCE_ISOLATION_ROOTS:
        root = project / relative
        sentinel = root / SOURCE_ISOLATION_SENTINEL_NAME
        if root.is_symlink() or not root.is_dir():
            raise TatqaP22SourceFreeFeasibilityError(
                "source-isolation root is unavailable"
            )
        entries = sorted(
            child.relative_to(root).as_posix() for child in root.rglob("*")
        )
        if entries != [SOURCE_ISOLATION_SENTINEL_NAME]:
            raise TatqaP22SourceFreeFeasibilityError(
                "formal TAT-QA source/run root is not an empty isolation sentinel"
            )
        if (
            sentinel.is_symlink()
            or not sentinel.is_file()
            or sentinel.read_bytes() != SOURCE_ISOLATION_SENTINEL_BYTES
        ):
            raise TatqaP22SourceFreeFeasibilityError(
                "source-isolation sentinel drifted"
            )
        roots.append(
            {
                "relative_root": relative,
                "sentinel_file_sha256": hashlib.sha256(
                    SOURCE_ISOLATION_SENTINEL_BYTES
                ).hexdigest(),
                "source_or_formal_payload_file_count": 0,
            }
        )
    body: dict[str, object] = {
        "schema": f"{VERSION}_source_isolation_v1",
        "roots": roots,
        "formal_TAT_QA_source_or_rows_present": False,
    }
    return {**body, "self_sha256": _semantic_hash(body)}


def _outer_unit_receipt(
    cgroup_path: Path = Path("/proc/self/cgroup"),
) -> dict[str, object]:
    try:
        raw = cgroup_path.read_bytes()
        lines = raw.decode("utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise TatqaP22SourceFreeFeasibilityError(
            "outer one-shot systemd cgroup is unavailable"
        ) from exc
    matches = [
        line for line in lines if line.rsplit("/", 1)[-1] == EXPECTED_OUTER_UNIT
    ]
    if len(matches) != 1:
        raise TatqaP22SourceFreeFeasibilityError(
            "outer one-shot systemd unit identity drifted"
        )
    body: dict[str, object] = {
        "schema": f"{VERSION}_outer_unit_v1",
        "unit_name": EXPECTED_OUTER_UNIT,
        "cgroup_file_sha256": hashlib.sha256(raw).hexdigest(),
        "matched_line_sha256": hashlib.sha256(matches[0].encode("utf-8")).hexdigest(),
    }
    return {**body, "self_sha256": _semantic_hash(body)}


def _install_source_free_worker_isolation(
    project: Path,
) -> tuple[object, dict[str, object]]:
    original = formal_runtime._worker_inaccessible_paths  # type: ignore[attr-defined]
    expected_paths = tuple(project / relative for relative in SOURCE_ISOLATION_ROOTS)
    calls: list[tuple[str, ...]] = []

    def source_free_paths(paths: formal_runtime.RuntimePaths) -> tuple[Path, ...]:
        if (
            type(paths) is not formal_runtime.RuntimePaths
            or paths.project_root.expanduser().absolute() != project
        ):
            raise TatqaP22SourceFreeFeasibilityError(
                "worker source-isolation project root drifted"
            )
        values = tuple(path.absolute() for path in expected_paths)
        calls.append(tuple(str(path) for path in values))
        return values

    formal_runtime._worker_inaccessible_paths = source_free_paths  # type: ignore[attr-defined]
    state: dict[str, object] = {
        "original": original,
        "replacement": source_free_paths,
        "calls": calls,
        "relative_inaccessible_roots": list(SOURCE_ISOLATION_ROOTS),
    }
    return original, state


def _restore_source_free_worker_isolation(
    original: object, state: Mapping[str, object]
) -> dict[str, object]:
    replacement = state.get("replacement")
    if formal_runtime._worker_inaccessible_paths is not replacement:  # type: ignore[attr-defined]
        raise TatqaP22SourceFreeFeasibilityError(
            "source-free worker isolation hook drifted"
        )
    formal_runtime._worker_inaccessible_paths = original  # type: ignore[attr-defined]
    calls = state.get("calls")
    if not isinstance(calls, list) or len(calls) != 3:
        raise TatqaP22SourceFreeFeasibilityError(
            "source-free worker isolation was not applied to all three launches"
        )
    expected = tuple(
        str(EXPECTED_PROJECT_ROOT / relative) for relative in SOURCE_ISOLATION_ROOTS
    )
    if any(tuple(row) != expected for row in calls):
        raise TatqaP22SourceFreeFeasibilityError(
            "source-free worker isolation path set drifted"
        )
    body: dict[str, object] = {
        "schema": f"{VERSION}_worker_source_isolation_v1",
        "launch_count": 3,
        "relative_inaccessible_roots": list(SOURCE_ISOLATION_ROOTS),
        "hook_restored_exact": formal_runtime._worker_inaccessible_paths is original,  # type: ignore[attr-defined]
    }
    return {**body, "self_sha256": _semantic_hash(body)}


def _portable_capability_receipt(
    encoder: PortableOfflineMiniLMEncoder,
    *,
    expected_asset_manifest: Path,
    expected_model_root: Path,
) -> dict[str, object]:
    """Sanitize absolute paths while preserving the portable startup proof."""

    runtime = getattr(encoder, "runtime_receipt", None)
    startup = getattr(encoder, "canary_receipt", None)
    if not isinstance(runtime, Mapping) or not isinstance(startup, Mapping):
        raise TatqaP22SourceFreeFeasibilityError(
            "portable MiniLM capability receipts are absent"
        )
    if (
        set(runtime)
        != {
            "asset_file_sha256",
            "asset_manifest_path",
            "asset_sha256",
            "embedding_dimension",
            "maximum_sequence_length",
            "model_root",
            "model_tree_sha256",
            "runtime_versions",
            "status",
            "weights_sha256",
        }
        or runtime.get("asset_file_sha256") != frozen_minilm.ASSET_FILE_SHA256
        or runtime.get("asset_sha256") != frozen_minilm.ASSET_SELF_SHA256
        or runtime.get("embedding_dimension") != frozen_minilm.EMBEDDING_DIMENSION
        or runtime.get("maximum_sequence_length")
        != frozen_minilm.MAXIMUM_SEQUENCE_LENGTH
        or runtime.get("model_tree_sha256") != frozen_minilm.MODEL_TREE_SHA256
        or runtime.get("runtime_versions") != frozen_minilm.EXPECTED_RUNTIME_VERSIONS
        or runtime.get("status")
        != "verified_offline_immutable_qasper_minilm_runtime"
        or runtime.get("weights_sha256") != frozen_minilm.WEIGHTS_SHA256
        or not isinstance(runtime.get("asset_manifest_path"), str)
        or not isinstance(runtime.get("model_root"), str)
    ):
        raise TatqaP22SourceFreeFeasibilityError(
            "portable MiniLM immutable runtime receipt drifted"
        )
    try:
        runtime_manifest = Path(str(runtime["asset_manifest_path"])).resolve(
            strict=True
        )
        runtime_model = Path(str(runtime["model_root"])).resolve(strict=True)
    except OSError as exc:
        raise TatqaP22SourceFreeFeasibilityError(
            "portable MiniLM immutable runtime path drifted"
        ) from exc
    if (
        runtime_manifest != expected_asset_manifest.resolve(strict=True)
        or runtime_model != expected_model_root.resolve(strict=True)
    ):
        raise TatqaP22SourceFreeFeasibilityError(
            "portable MiniLM immutable runtime path drifted"
        )
    sanitized_runtime = {
        key: value
        for key, value in runtime.items()
        if key not in {"asset_manifest_path", "model_root"}
    }
    observed_hashes = startup.get("observed_output_hashes")
    maximum_norm_error = startup.get("maximum_observed_row_l2_norm_error")
    if (
        set(startup)
        != {
            "all_values_finite",
            "at_least_two_distinct_vectors",
            "embedding_dtype",
            "embedding_shape",
            "external_network_calls",
            "formal_QASPER_source_or_rows_accessed",
            "formal_TAT_QA_source_or_rows_accessed",
            "maximum_observed_row_l2_norm_error",
            "observed_output_hashes",
            "per_row_l2_norm_maximum_error",
            "public_text_vector_identity_exact",
            "public_text_vector_sha256",
            "qasper_rows_or_archives_accessed_by_canary",
            "repeat_byte_exact",
            "repeat_count",
            "repeat_elementwise_exact",
            "schema",
            "sentence_count",
            "status",
            "tatqa_rows_or_archives_accessed_by_canary",
        }
        or startup.get("schema") != "qasper_minilm_portable_startup_canary_v2"
        or startup.get("status")
        != "passed_portable_public_synthetic_structural_canary"
        or startup.get("repeat_count") != 2
        or startup.get("repeat_byte_exact") is not True
        or startup.get("repeat_elementwise_exact") is not True
        or startup.get("embedding_shape") != [256, 384]
        or startup.get("embedding_dtype") != "float32"
        or startup.get("all_values_finite") is not True
        or startup.get("at_least_two_distinct_vectors") is not True
        or startup.get("formal_QASPER_source_or_rows_accessed") is not False
        or startup.get("formal_TAT_QA_source_or_rows_accessed") is not False
        or startup.get("qasper_rows_or_archives_accessed_by_canary") is not False
        or startup.get("tatqa_rows_or_archives_accessed_by_canary") is not False
        or startup.get("external_network_calls") != 0
        or isinstance(maximum_norm_error, bool)
        or not isinstance(maximum_norm_error, (int, float))
        or not math.isfinite(float(maximum_norm_error))
        or not 0.0 <= float(maximum_norm_error) <= 1e-5
        or startup.get("per_row_l2_norm_maximum_error") != 1e-5
        or startup.get("public_text_vector_identity_exact") is not True
        or startup.get("public_text_vector_sha256")
        != "c122a1e09d2f84ad00a4c0b30abb979e13facdb8c1a5b3b15cb952b51b173249"
        or startup.get("sentence_count") != 256
        or not isinstance(observed_hashes, Mapping)
        or set(observed_hashes)
        != {
            "compared_to_expected_or_allowlist",
            "float32_little_endian_c_order_sha256",
            "normative_acceptance",
            "quantized_embedding_matrix_sha256",
        }
        or observed_hashes.get("normative_acceptance") is not False
        or observed_hashes.get("compared_to_expected_or_allowlist") is not False
        or not isinstance(
            observed_hashes.get("float32_little_endian_c_order_sha256"), str
        )
        or _HEX64.fullmatch(
            str(observed_hashes.get("float32_little_endian_c_order_sha256"))
        )
        is None
        or not isinstance(
            observed_hashes.get("quantized_embedding_matrix_sha256"), str
        )
        or _HEX64.fullmatch(
            str(observed_hashes.get("quantized_embedding_matrix_sha256"))
        )
        is None
    ):
        raise TatqaP22SourceFreeFeasibilityError(
            "portable MiniLM structural canary drifted"
        )
    body: dict[str, object] = {
        "schema": PORTABLE_CAPABILITY_SCHEMA,
        "capability_class": "PortableOfflineMiniLMEncoder",
        "execution": {
            "device": "cpu",
            "dtype": "float32",
            "in_process": True,
            "torch_threads": 1,
        },
        "omitted_absolute_path_fields": ["asset_manifest_path", "model_root"],
        "runtime_receipt": _json_copy(sanitized_runtime),
        "portable_startup_canary_receipt": _json_copy(startup),
        "formal_TAT_QA_source_or_rows_accessed": False,
        "normative_output_hash_acceptance": False,
    }
    return {**body, "self_sha256": _semantic_hash(body)}


def _terminal_failure(root: Path, stage: str, exc: BaseException) -> None:
    body = {
        "schema": f"{VERSION}_terminal_failure_v1",
        "status": "terminal_no_retry_nonqualification_non_efficacy",
        "failure_stage": stage,
        "failure_type_sha256": hashlib.sha256(
            f"{type(exc).__module__}.{type(exc).__qualname__}".encode("utf-8")
        ).hexdigest(),
        "P21_qualification_claimed": False,
        "efficacy_claimed": False,
        "formal_TAT_QA_source_opened": False,
        "external_network_calls": 0,
        "api_or_online_evaluator_calls": 0,
        "retry_replay_resample_provider_switch": 0,
    }
    try:
        _write_exclusive(
            root / FAILURE_FILENAME,
            {**body, "self_sha256": _semantic_hash(body)},
        )
    except BaseException:
        pass


def run_source_free_feasibility(
    *,
    project_root: str | Path,
    typed_runtime_python: str | Path,
    hippo_runtime_python: str | Path,
    qwen_model: str | Path,
    minilm_asset_manifest: str | Path,
    minilm_model: str | Path,
    hippo_llm_model: str | Path,
    hippo_embedding_model: str | Path,
    hipporag_source: str | Path,
    hippo_attestation: str | Path,
    p21_runtime_fingerprint: str | Path,
    diagnostic_snapshot_commit: str,
    _post_minilm_environment_normalizer: (
        Callable[[], Mapping[str, object]] | None
    ) = None,
) -> dict[str, object]:
    """Consume one independent public-synthetic feasibility attempt."""

    project = Path(project_root).expanduser().resolve(strict=True)
    if project != EXPECTED_PROJECT_ROOT:
        raise TatqaP22SourceFreeFeasibilityError(
            "source-free project is not the one preregistered host path"
        )
    root = EXPECTED_FEASIBILITY_ROOT
    work = EXPECTED_WORK_ROOT
    canary_path = EXPECTED_PUBLIC_CANARY_OUTPUT
    if root.exists() or root.is_symlink():
        raise TatqaP22SourceFreeFeasibilityError(
            "source-free feasibility root is already consumed"
        )
    if work.exists() or work.is_symlink() or canary_path.exists() or canary_path.is_symlink():
        raise TatqaP22SourceFreeFeasibilityError(
            "source-free feasibility work/output path is already consumed"
        )
    normalizer_binding = _post_minilm_normalizer_binding(
        project, _post_minilm_environment_normalizer
    )
    implementation = _diagnostic_snapshot_binding(
        project, diagnostic_snapshot_commit
    )
    source_isolation = _source_isolation_receipt(project)
    outer_unit = _outer_unit_receipt()
    root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.mkdir(mode=0o700)
    marker_body = {
        "schema": f"{VERSION}_one_shot_marker_v1",
        "status": "started_source_free_nonqualification_non_efficacy",
        "P21_qualification_claimed": False,
        "efficacy_claimed": False,
        "formal_TAT_QA_source_opened": False,
        "retry_replay_resample_provider_switch": 0,
        "diagnostic_snapshot_commit": diagnostic_snapshot_commit,
        "diagnostic_snapshot_binding_self_sha256": implementation[
            "self_sha256"
        ],
        "source_isolation_self_sha256": source_isolation["self_sha256"],
        "outer_unit_self_sha256": outer_unit["self_sha256"],
        "post_minilm_environment_normalizer_binding": normalizer_binding,
    }
    _write_exclusive(
        root / MARKER_FILENAME,
        {**marker_body, "self_sha256": _semantic_hash(marker_body)},
    )

    typed_python = Path(typed_runtime_python).expanduser().absolute()
    hippo_python = Path(hippo_runtime_python).expanduser().absolute()
    qwen = Path(qwen_model).expanduser().absolute()
    minilm_manifest = Path(minilm_asset_manifest).expanduser().absolute()
    minilm = Path(minilm_model).expanduser().absolute()
    hippo_llm = Path(hippo_llm_model).expanduser().absolute()
    hippo_embedding = Path(hippo_embedding_model).expanduser().absolute()
    hippo_source = Path(hipporag_source).expanduser().absolute()
    hippo_attestation_path = Path(hippo_attestation).expanduser().absolute()
    fingerprint_path = Path(p21_runtime_fingerprint).expanduser().absolute()
    typed_runner: formal_runtime.SystemdTypedPlanBatchRunner | None = None
    hippo_runner: formal_runtime.SystemdHippoByteRunner | None = None
    isolation_original: object | None = None
    isolation_state: dict[str, object] | None = None
    normalization_receipt: dict[str, object] | None = None
    stage = "p21_entry_launch_envelope"
    try:
        entry_phase = formal_runtime.user_systemd_launcher_phase_receipt(
            phase="entry"
        )
        stage = "p21_runtime_inventory"
        inventory = formal_runtime.runtime_inventory_snapshot(
            typed_runtime_python=typed_python,
            hippo_runtime_python=hippo_python,
            qwen_model=qwen,
            minilm_manifest=minilm_manifest,
            hippo_attestation=hippo_attestation_path,
        )
        stage = "p21_post_runtime_inventory_launch_envelope"
        post_inventory_phase = formal_runtime.user_systemd_launcher_phase_receipt(
            phase="post_runtime_inventory"
        )
        stage = "p21_systemd_network_preflight"
        network_preflight = formal_runtime.systemd_network_preflight()
        paths = formal_runtime.RuntimePaths(
            project_root=project,
            typed_runtime_python=typed_python,
            hippo_runtime_python=hippo_python,
            qwen_model=qwen,
            minilm_asset_manifest=minilm_manifest,
            minilm_model=minilm,
            hippo_llm_model=hippo_llm,
            hippo_embedding_model=hippo_embedding,
            hipporag_source=hippo_source,
            hippo_attestation=hippo_attestation_path,
            fingerprint_manifest=fingerprint_path,
            work_root=work,
        )
        stage = "p21_runtime_fingerprint_reverification"
        fingerprint = formal_runtime.verify_runtime_fingerprint(paths)
        try:
            fingerprint_raw = fingerprint_path.read_bytes()
        except OSError as exc:
            raise TatqaP22SourceFreeFeasibilityError(
                "P21 runtime fingerprint disappeared after verification"
            ) from exc
        if fingerprint_raw != _canonical_bytes(fingerprint):
            raise TatqaP22SourceFreeFeasibilityError(
                "verified P21 runtime fingerprint file binding drifted"
            )
        launcher = fingerprint.get("safe_user_systemd_launch_envelope")
        phase_receipts = (
            launcher.get("phase_receipts")
            if isinstance(launcher, Mapping)
            else None
        )
        if (
            fingerprint.get("runtime_inventory") != inventory
            or fingerprint.get("systemd_network_preflight") != network_preflight
            or not isinstance(phase_receipts, Mapping)
            or phase_receipts.get("entry") != entry_phase
            or phase_receipts.get("post_runtime_inventory") != post_inventory_phase
        ):
            raise TatqaP22SourceFreeFeasibilityError(
                "live P21 inventory or launch phases differ from fingerprint"
            )
        stage = "portable_minilm_initialization"
        encoder = PortableOfflineMiniLMEncoder(
            asset_manifest_path=minilm_manifest,
            model_root=minilm,
            run_canary=True,
        )
        portable_receipt = _portable_capability_receipt(
            encoder,
            expected_asset_manifest=minilm_manifest,
            expected_model_root=minilm,
        )
        if _post_minilm_environment_normalizer is not None:
            stage = "post_minilm_environment_normalization"
            if (
                _post_minilm_normalizer_binding(
                    project, _post_minilm_environment_normalizer
                )
                != normalizer_binding
            ):
                raise TatqaP22SourceFreeFeasibilityError(
                    "post-MiniLM normalizer binding changed before invocation"
                )
            normalization_receipt = _validated_auxiliary_receipt(
                _post_minilm_environment_normalizer()
            )
            if (
                _post_minilm_normalizer_binding(
                    project, _post_minilm_environment_normalizer
                )
                != normalizer_binding
            ):
                raise TatqaP22SourceFreeFeasibilityError(
                    "post-MiniLM normalizer binding changed after invocation"
                )
        stage = "p21_post_minilm_launch_envelope"
        post_minilm_phase = formal_runtime.user_systemd_launcher_phase_receipt(
            phase="post_minilm"
        )
        work.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        work.mkdir(mode=0o700)
        stage = "p22_source_free_worker_isolation_install"
        isolation_original, isolation_state = _install_source_free_worker_isolation(
            project
        )
        stage = "p21_public_synthetic_production_path"
        typed_runner = formal_runtime.SystemdTypedPlanBatchRunner(paths)
        hippo_runner = formal_runtime.SystemdHippoByteRunner(paths)
        if (
            type(typed_runner) is not formal_runtime.SystemdTypedPlanBatchRunner
            or type(hippo_runner) is not formal_runtime.SystemdHippoByteRunner
            or type(encoder) is not PortableOfflineMiniLMEncoder
        ):
            raise TatqaP22SourceFreeFeasibilityError(
                "feasibility capability class drifted"
            )
        public_receipt = canary.run_public_production_canary(
            runtime_fingerprint_path=fingerprint_path,
            output_path=canary_path,
            typed_plan_runner=typed_runner,
            encoder=encoder,
            hippo_runner=hippo_runner,
            post_minilm_launcher_phase_receipt=post_minilm_phase,
            minilm_worker_receipt=portable_receipt,
        )
        if (
            public_receipt.get("hippo_canary_ran") is not True
            or public_receipt.get("typed_plan_worker_receipt_source")
            != "capability_receipt_snapshot"
            or public_receipt.get("minilm_worker_receipt_source")
            != "explicit_formal_receipt"
            or public_receipt.get("hippo_worker_receipt_source")
            != "capability_receipt_snapshot"
            or public_receipt.get("formal_source_opened") is not False
            or public_receipt.get("external_network_calls") != 0
            or public_receipt.get("api_or_online_evaluator_calls") != 0
        ):
            raise TatqaP22SourceFreeFeasibilityError(
                "public canary lacked actual non-fallback offline capabilities"
            )
        if public_receipt.get("minilm_worker_receipt_snapshot") != portable_receipt:
            raise TatqaP22SourceFreeFeasibilityError(
                "portable MiniLM evidence was not bound by the public canary"
            )
        stage = "worker_closure_verification"
        typed_aborted = typed_runner.abort_all_workers()
        hippo_aborted = hippo_runner.abort_all_workers()
        typed_closed = typed_runner.verify_all_workers_closed()
        hippo_closed = hippo_runner.verify_all_workers_closed()
        if (
            typed_aborted != typed_closed
            or hippo_aborted != hippo_closed
            or len(typed_closed) != 2
            or len(hippo_closed) != 1
        ):
            raise TatqaP22SourceFreeFeasibilityError(
                "Qwen/Hippo systemd worker closure drifted"
            )
        stage = "p22_source_free_worker_isolation_restore"
        worker_isolation = _restore_source_free_worker_isolation(
            isolation_original, isolation_state
        )
        isolation_original = None
        isolation_state = None
        try:
            canary_raw = canary_path.read_bytes()
        except OSError as exc:
            raise TatqaP22SourceFreeFeasibilityError(
                "public canary file is unavailable"
            ) from exc
        if canary_raw != _canonical_bytes(public_receipt):
            raise TatqaP22SourceFreeFeasibilityError(
                "public canary file binding drifted"
            )
        if fingerprint_path.read_bytes() != fingerprint_raw:
            raise TatqaP22SourceFreeFeasibilityError(
                "P21 runtime fingerprint changed during feasibility execution"
            )
        stage = "terminal_integrity_reverification"
        implementation_end = _diagnostic_snapshot_binding(
            project, diagnostic_snapshot_commit
        )
        source_isolation_end = _source_isolation_receipt(project)
        outer_unit_end = _outer_unit_receipt()
        if (
            implementation_end != implementation
            or source_isolation_end != source_isolation
            or outer_unit_end != outer_unit
        ):
            raise TatqaP22SourceFreeFeasibilityError(
                "source-free terminal integrity binding drifted"
            )
        closure = {
            "typed_unit_count": 2,
            "hipporag_unit_count": 1,
            "abort_and_reverify_receipts_exact": True,
            "typed_closure_set_sha256": _semantic_hash(list(typed_closed)),
            "hipporag_closure_set_sha256": _semantic_hash(list(hippo_closed)),
        }
        terminal_body: dict[str, object] = {
            "schema": f"{VERSION}_terminal_success_v1",
            "status": "passed_source_free_feasibility_only",
            "P21_qualification_claimed": False,
            "efficacy_claimed": False,
            "p21_runtime_fingerprint_self_sha256": fingerprint["self_sha256"],
            "p21_runtime_fingerprint_file_sha256": hashlib.sha256(
                fingerprint_raw
            ).hexdigest(),
            "p21_runtime_inventory_sha256": _semantic_hash(inventory),
            "p21_systemd_network_preflight": network_preflight,
            "p21_launcher_phase_self_sha256s": {
                "entry": entry_phase["self_sha256"],
                "post_runtime_inventory": post_inventory_phase["self_sha256"],
                "post_minilm": post_minilm_phase["self_sha256"],
            },
            "portable_minilm_capability_receipt": portable_receipt,
            "portable_minilm_capability_receipt_self_sha256": portable_receipt[
                "self_sha256"
            ],
            "post_minilm_environment_normalization_receipt": (
                normalization_receipt
            ),
            "post_minilm_environment_normalizer_binding": normalizer_binding,
            "public_canary_self_sha256": public_receipt["self_sha256"],
            "public_canary_file_sha256": hashlib.sha256(canary_raw).hexdigest(),
            "diagnostic_snapshot_binding": implementation,
            "diagnostic_snapshot_binding_self_sha256": implementation[
                "self_sha256"
            ],
            "source_isolation": source_isolation,
            "source_free_worker_isolation": worker_isolation,
            "outer_one_shot_unit": outer_unit,
            "worker_closure": closure,
            "formal_TAT_QA_source_opened": False,
            "external_network_calls": 0,
            "api_or_online_evaluator_calls": 0,
            "retry_replay_resample_provider_switch": 0,
        }
        terminal = {
            **terminal_body,
            "self_sha256": _semantic_hash(terminal_body),
        }
        _write_exclusive(root / SUCCESS_FILENAME, terminal)
        return terminal
    except BaseException as exc:
        closure_error: BaseException | None = None
        for runner in (typed_runner, hippo_runner):
            if runner is None:
                continue
            try:
                runner.abort_all_workers()
                runner.verify_all_workers_closed()
            except BaseException as candidate:
                if closure_error is None:
                    closure_error = candidate
        if isolation_original is not None and isolation_state is not None:
            try:
                formal_runtime._worker_inaccessible_paths = isolation_original  # type: ignore[attr-defined]
            except BaseException as candidate:
                if closure_error is None:
                    closure_error = candidate
        if closure_error is not None:
            stage = "terminal_worker_closure_unproved"
            exc = closure_error
        try:
            if (
                _diagnostic_snapshot_binding(project, diagnostic_snapshot_commit)
                != implementation
                or _source_isolation_receipt(project) != source_isolation
                or _outer_unit_receipt() != outer_unit
            ):
                raise TatqaP22SourceFreeFeasibilityError(
                    "terminal integrity binding drifted"
                )
        except BaseException as candidate:
            stage = "terminal_integrity_reverification_failed"
            exc = candidate
        _terminal_failure(root, stage, exc)
        raise TatqaP22SourceFreeFeasibilityError(
            "source-free feasibility failed terminally"
        ) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--typed-runtime-python", required=True, type=Path)
    parser.add_argument("--hippo-runtime-python", required=True, type=Path)
    parser.add_argument("--qwen-model", required=True, type=Path)
    parser.add_argument("--minilm-asset-manifest", required=True, type=Path)
    parser.add_argument("--minilm-model", required=True, type=Path)
    parser.add_argument("--hippo-llm-model", required=True, type=Path)
    parser.add_argument("--hippo-embedding-model", required=True, type=Path)
    parser.add_argument("--hipporag-source", required=True, type=Path)
    parser.add_argument("--hippo-attestation", required=True, type=Path)
    parser.add_argument("--p21-runtime-fingerprint", required=True, type=Path)
    parser.add_argument("--diagnostic-snapshot-commit", required=True)
    return parser


def _main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_source_free_feasibility(
        project_root=args.project_root,
        typed_runtime_python=args.typed_runtime_python,
        hippo_runtime_python=args.hippo_runtime_python,
        qwen_model=args.qwen_model,
        minilm_asset_manifest=args.minilm_asset_manifest,
        minilm_model=args.minilm_model,
        hippo_llm_model=args.hippo_llm_model,
        hippo_embedding_model=args.hippo_embedding_model,
        hipporag_source=args.hipporag_source,
        hippo_attestation=args.hippo_attestation,
        p21_runtime_fingerprint=args.p21_runtime_fingerprint,
        diagnostic_snapshot_commit=args.diagnostic_snapshot_commit,
    )
    print(_canonical_bytes(result).decode("ascii"), end="")
    return 0


def main() -> int:
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FAILURE_FILENAME",
    "MARKER_FILENAME",
    "PORTABLE_CAPABILITY_SCHEMA",
    "SUCCESS_FILENAME",
    "TatqaP22SourceFreeFeasibilityError",
    "VERSION",
    "main",
    "run_source_free_feasibility",
]
