"""Concrete offline worker launchers for the frozen TAT-QA P18 study.

The lifecycle adapter owns custody and sequencing.  This module owns only
label-free model execution.  Every Qwen and HippoRAG worker is launched in a
fresh user-systemd transient unit with IP networking denied and with an exact,
environment-cleared allowlist.  It never accepts answers, families, mappings,
gold units, source paths, or provider/API credentials.

MiniLM runs in the controller process through the already-attested QASPER
binding.  Qwen is one deterministic batch worker per block.  HippoRAG is one
fresh item-local subprocess and index per item, so the controller can safely
schedule at most eight of them through its dedicated thread executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import site
import socket
import stat
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks import tatqa_p18_label_free_runtime_v1 as features
from assumption_agent.benchmarks import tatqa_p18_typed_evaluator_core_v1 as core
from replication_runtime.qasper_minilm_v1.binding import (
    ASSET_FILE_SHA256 as MINILM_ASSET_FILE_SHA256,
    ASSET_SELF_SHA256 as MINILM_ASSET_SELF_SHA256,
    MODEL_TREE_SHA256 as MINILM_MODEL_TREE_SHA256,
    OfflineMiniLMEncoder,
)

from . import hipporag_contract, typed_plan_contract


VERSION = "tatqa_p18_formal_runtime_v1"
STUDY_DESIGN_SELF_SHA256 = (
    "48bb31c7f906676703fa8f1eff8ee9dd91100d2026dc2cbb977c752791179307"
)
QWEN_MODEL_TREE_SHA256 = (
    "199326617a7dfca1a87357add5b6ad7478ef0681fa3cd42a01256a50042571c3"
)
MINILM_EXPECTED_ASSET_FILE_SHA256 = (
    "62b85c7752f2e46932fb9fb13ae2f3aac9eb750a33c8f07102739040feb6cc75"
)
MINILM_EXPECTED_ASSET_SELF_SHA256 = (
    "921d9b1945581130e03c53f448092c3de3b30714431c6cac9b3b32c2ec10abad"
)
# This is the generic ``[{path,sha256,size_bytes}]`` tree receipt used by the
# remote fingerprint.  The stronger QASPER manifest independently binds its
# normative ``[{path,sha256,size}]`` tree to ``MINILM_MODEL_TREE_SHA256``.
MINILM_GENERIC_TREE_SHA256 = (
    "42d8d798e4f01e68d9bb10634b9c712de00f7f8495271636fd6311b2db58e506"
)
HIPPORAG_ATTESTATION_FILE_SHA256 = (
    "0f047699560fbc10a2d39f9de6380004cd2887915c3e2f6ef0fcc806a6670c0e"
)
HIPPORAG_SOURCE_TREE_SHA256 = (
    "a644ab2811db2739db3cfbdc051561e2cfdf2ed87286f8ebd00a5971d189cdd5"
)
HIPPORAG_LLM_TREE_SHA256 = (
    "d626d755c99c006761d5e069aa85a73fe8b011c6c0f5d0323a6f8de85246bcb5"
)
HIPPORAG_EMBEDDING_TREE_SHA256 = MINILM_GENERIC_TREE_SHA256

HIPPORAG_ATTESTATION_SCHEMA = "musique_official_hipporag_filesystem_attestation_v3"
HIPPORAG_ATTESTATION_RECEIPT_SHA256 = (
    "23996f9f41f494e2fd032b285039ec9420f6a893c24081e59c1ec79f229c2c60"
)
HIPPORAG_ATTESTATION_LLM_REPO_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
HIPPORAG_ATTESTATION_LLM_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
HIPPORAG_ATTESTATION_LLM_ASSET_SHA256 = (
    "378d593e91d13e42da36365ab2e2092c50feec7aea76a3fe228cd0a50310f9f4"
)
HIPPORAG_ATTESTATION_LLM_TOPOLOGY_SHA256 = (
    "e57d1cd3a05f7c7ce8600d8b5789366ba0a2e2394bb5d2a789a0708586d7451e"
)
HIPPORAG_ATTESTATION_EMBEDDING_ASSET_SHA256 = (
    "92cb157b9fff6a2d81ff1713aa36db90d5ccae5a6576c87b643a19be4675a9ef"
)
HIPPORAG_ATTESTATION_EMBEDDING_TOPOLOGY_SHA256 = (
    "b615f83014631e65f16638a0a24894a00e3f87dba9ff69e92230233da7d8c2be"
)
HIPPORAG_ATTESTATION_OFFICIAL_SOURCE_TREE_SHA256 = (
    "30941a14e8dc48f7a41f8679ce6cba0bac9e3cdd99ed919560b45872e1058700"
)
HIPPORAG_ATTESTATION_OFFICIAL_SOURCE_FILE_COUNT = 52
HIPPORAG_ATTESTATION_OFFICIAL_SOURCE_ROOT_ROLE = "overlay"

SYSTEMD_RUN = Path("/usr/bin/systemd-run")
SYSTEMCTL = Path("/usr/bin/systemctl")
ENV_EXECUTABLE = Path("/usr/bin/env")
PREFLIGHT_PYTHON = Path("/usr/bin/python3")
SYSTEMD_FLAGS = ("--user", "--wait", "--pipe", "--collect", "--quiet")
SYSTEMD_NETWORK_PROPERTIES = (
    "IPAddressDeny=any",
    "RestrictAddressFamilies=AF_UNIX",
)
FILESYSTEM_ISOLATION = (
    "systemd_InaccessiblePaths_official_source_and_acquisition_v1"
)
QWEN_BATCH_SIZE = 4
QWEN_TIMEOUT_SECONDS = 3_600
HIPPORAG_TIMEOUT_SECONDS = 14_400
HIPPORAG_CPU_THREADS = 2
HIPPORAG_CONFIGURED_TORCH_INTRAOP_THREADS = 1
HIPPORAG_CONFIGURED_TORCH_INTEROP_THREADS = 1
HIPPORAG_SYSTEMD_TASKS_MAX = 3
HIPPORAG_THREAD_MONITOR_PROCESS_RESERVATION = 1
HIPPORAG_MAXIMUM_WORKER_PROCESS_THREADS = 2
QWEN_PHYSICAL_GPU = "1"
SYSTEMD_WORKER_KILL_MODE = "control-group"
SYSTEMD_UNIT_CLOSURE_SCHEMA = f"{VERSION}_systemd_unit_closure_v1"
SYSTEMD_START_POLICY_SCHEMA = f"{VERSION}_systemd_start_policy_v1"
SYSTEMD_CLOSE_POLL_SECONDS = 0.05
SYSTEMD_STOP_POLL_ATTEMPTS = 20
SYSTEMD_KILL_POLL_ATTEMPTS = 200
SYSTEMD_CONTROL_TIMEOUT_SECONDS = 5
SYSTEMD_CLIENT_REAP_TIMEOUT_SECONDS = 10
CGROUP_ROOT = Path("/sys/fs/cgroup")

RUNTIME_DISTRIBUTIONS = (
    "huggingface-hub",
    "numpy",
    "safetensors",
    "sentence-transformers",
    "tokenizers",
    "torch",
    "transformers",
)

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9_.-]+\Z")


class TatqaP18FormalRuntimeError(RuntimeError):
    """A frozen asset, transport, worker, or receipt failed closed."""


def canonical_json_bytes(value: object, *, newline: bool = True) -> bytes:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise TatqaP18FormalRuntimeError("value is not canonical JSON") from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value, newline=False)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise TatqaP18FormalRuntimeError("bound file is unreadable") from exc
    return digest.hexdigest()


def tree_receipt(root: Path) -> dict[str, object]:
    """Return the exact P17-compatible immutable tree receipt."""

    if root.is_symlink() or not root.is_dir():
        raise TatqaP18FormalRuntimeError("frozen tree is unavailable")
    rows: list[dict[str, object]] = []
    total = 0
    for current, directories, files in os.walk(root, followlinks=False):
        base = Path(current)
        for name in directories:
            if (base / name).is_symlink():
                raise TatqaP18FormalRuntimeError("frozen tree contains a symlink")
        for name in files:
            path = base / name
            if path.is_symlink() or not path.is_file():
                raise TatqaP18FormalRuntimeError("frozen tree contains a non-file")
            size = path.stat().st_size
            total += size
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                    "size_bytes": size,
                }
            )
    rows.sort(key=lambda row: str(row["path"]))
    return {
        "file_count": len(rows),
        "size_bytes": total,
        "tree_sha256": stable_hash(rows),
    }


def _self_hash(value: Mapping[str, Any], field: str = "self_sha256") -> str:
    body = dict(value)
    declared = body.pop(field, None)
    if not isinstance(declared, str) or _HEX64.fullmatch(declared) is None:
        raise TatqaP18FormalRuntimeError("receipt self hash is absent")
    if stable_hash(body) != declared:
        raise TatqaP18FormalRuntimeError("receipt self hash drifted")
    return declared


def _load_canonical_object(path: Path, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise TatqaP18FormalRuntimeError(f"{field} is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TatqaP18FormalRuntimeError(f"{field} is invalid") from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise TatqaP18FormalRuntimeError(f"{field} is not canonical JSON")
    return value


def _safe_root(path: Path, field: str, *, must_exist: bool = True) -> Path:
    absolute = path.expanduser().absolute()
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise TatqaP18FormalRuntimeError(f"{field} contains a symlink")
    if must_exist and not absolute.exists():
        raise TatqaP18FormalRuntimeError(f"{field} is unavailable")
    return absolute


def _safe_runtime_executable(path: Path) -> Path:
    """Allow only a leaf venv symlink to one verified regular executable."""

    absolute = path.expanduser().absolute()
    _safe_root(absolute.parent, "runtime Python parent")
    try:
        metadata = absolute.lstat()
    except OSError as exc:
        raise TatqaP18FormalRuntimeError("runtime Python is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode):
        target = absolute.resolve(strict=True)
        _safe_root(target, "runtime Python target")
    elif not stat.S_ISREG(metadata.st_mode):
        raise TatqaP18FormalRuntimeError("runtime Python is not a regular file")
    if not absolute.is_file() or not os.access(absolute, os.X_OK):
        raise TatqaP18FormalRuntimeError("runtime Python is not executable")
    return absolute


def _safe_leaf(value: str, field: str) -> str:
    if not isinstance(value, str) or _SAFE_COMPONENT.fullmatch(value) is None:
        raise TatqaP18FormalRuntimeError(f"{field} is unsafe")
    return value


def _write_exclusive(path: Path, raw: bytes, *, mode: int = 0o600) -> str:
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            mode,
        )
    except OSError as exc:
        raise TatqaP18FormalRuntimeError("worker artifact already exists") from exc
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class RuntimePaths:
    """Exact local assets; no source or label path is representable here."""

    project_root: Path
    runtime_python: Path
    qwen_model: Path
    minilm_asset_manifest: Path
    minilm_model: Path
    hippo_llm_model: Path
    hippo_embedding_model: Path
    hipporag_source: Path
    hippo_attestation: Path
    fingerprint_manifest: Path
    work_root: Path

    def checked(self) -> "RuntimePaths":
        project = _safe_root(self.project_root, "project root")
        python = _safe_runtime_executable(self.runtime_python)
        qwen = _safe_root(self.qwen_model, "Qwen model")
        minilm_manifest = _safe_root(self.minilm_asset_manifest, "MiniLM manifest")
        minilm = _safe_root(self.minilm_model, "MiniLM model")
        hippo_llm = _safe_root(self.hippo_llm_model, "HippoRAG LLM")
        hippo_embedding = _safe_root(
            self.hippo_embedding_model, "HippoRAG embedding model"
        )
        hipporag_source = _safe_root(self.hipporag_source, "HippoRAG source")
        attestation = _safe_root(self.hippo_attestation, "HippoRAG attestation")
        fingerprint = _safe_root(self.fingerprint_manifest, "runtime fingerprint")
        work = _safe_root(self.work_root, "runtime work root", must_exist=False)
        for path, field in (
            (qwen, "Qwen model"),
            (minilm, "MiniLM model"),
            (hippo_llm, "HippoRAG LLM"),
            (hippo_embedding, "HippoRAG embedding model"),
            (hipporag_source, "HippoRAG source"),
        ):
            if not path.is_dir():
                raise TatqaP18FormalRuntimeError(f"{field} is not a directory")
        for path, field in (
            (minilm_manifest, "MiniLM manifest"),
            (attestation, "HippoRAG attestation"),
            (fingerprint, "runtime fingerprint"),
        ):
            if not path.is_file():
                raise TatqaP18FormalRuntimeError(f"{field} is not a file")
        return RuntimePaths(
            project_root=project,
            runtime_python=python,
            qwen_model=qwen,
            minilm_asset_manifest=minilm_manifest,
            minilm_model=minilm,
            hippo_llm_model=hippo_llm,
            hippo_embedding_model=hippo_embedding,
            hipporag_source=hipporag_source,
            hippo_attestation=attestation,
            fingerprint_manifest=fingerprint,
            work_root=work,
        )


def _executable_binding(path: Path, *, field: str) -> dict[str, object]:
    lexical = path.expanduser().absolute()
    try:
        target = lexical.resolve(strict=True)
    except OSError as exc:
        raise TatqaP18FormalRuntimeError(f"{field} target is unavailable") from exc
    if not target.is_file() or not os.access(lexical, os.X_OK):
        raise TatqaP18FormalRuntimeError(f"{field} is not executable")
    return {
        "lexical_path_sha256": hashlib.sha256(
            str(lexical).encode("utf-8")
        ).hexdigest(),
        "resolved_path_sha256": hashlib.sha256(
            str(target).encode("utf-8")
        ).hexdigest(),
        "resolved_file_sha256": file_sha256(target),
    }


def _public_file_binding(path: Path, *, field: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise TatqaP18FormalRuntimeError(f"{field} is unavailable")
    raw = path.read_bytes()
    return {
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _qwen_config_binding(root: Path) -> dict[str, object]:
    path = root / "config.json"
    binding = _public_file_binding(path, field="Qwen config")
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TatqaP18FormalRuntimeError("Qwen config is invalid") from exc
    context = value.get("max_position_embeddings") if isinstance(value, dict) else None
    if (
        isinstance(context, bool)
        or not isinstance(context, int)
        or context < 16_640
    ):
        raise TatqaP18FormalRuntimeError("Qwen context window is too short")
    return {**binding, "max_position_embeddings": context}


def _canonical_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _metadata_identity(path: Path) -> tuple[str, str]:
    if path.is_symlink() or not path.is_file():
        raise TatqaP18FormalRuntimeError("dependency METADATA is unavailable")
    name: str | None = None
    version: str | None = None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if name is None and line.startswith("Name: "):
                name = line[6:].strip()
            elif version is None and line.startswith("Version: "):
                version = line[9:].strip()
            if name is not None and version is not None:
                break
    except (OSError, UnicodeDecodeError) as exc:
        raise TatqaP18FormalRuntimeError("dependency METADATA is invalid") from exc
    if not name or not version:
        raise TatqaP18FormalRuntimeError("dependency METADATA identity is absent")
    return _canonical_distribution_name(name), version


def _attestation_tree_binding(path: Path) -> tuple[int, str]:
    if path.is_symlink() or not path.is_dir():
        raise TatqaP18FormalRuntimeError("dependency dist-info tree is unavailable")
    entries = sorted(
        (entry for entry in path.rglob("*") if entry.is_file()),
        key=lambda entry: entry.relative_to(path).as_posix(),
    )
    if not entries or any(entry.is_symlink() for entry in path.rglob("*")):
        raise TatqaP18FormalRuntimeError("dependency dist-info tree drifted")
    rows = [
        {
            "path": entry.relative_to(path).as_posix(),
            "sha256": file_sha256(entry),
        }
        for entry in entries
    ]
    return len(rows), stable_hash(rows)


def _parse_pyvenv(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise TatqaP18FormalRuntimeError("runtime is not a lexical venv")
    values: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if " = " in line:
                key, value = line.split(" = ", 1)
                values[key.strip()] = value.strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise TatqaP18FormalRuntimeError("pyvenv.cfg is invalid") from exc
    if (
        not values.get("home")
        or re.fullmatch(r"\d+\.\d+(?:\.\d+)?", values.get("version", ""))
        is None
        or values.get("include-system-site-packages") not in {"true", "false"}
    ):
        raise TatqaP18FormalRuntimeError("pyvenv.cfg identity drifted")
    return values


def _runtime_search_roots(
    runtime_python: Path,
) -> tuple[Path, list[tuple[str, Path]]]:
    lexical = runtime_python.expanduser().absolute()
    if lexical.name != "python" or lexical.parent.name != "bin":
        raise TatqaP18FormalRuntimeError(
            "runtime Python must retain the lexical venv/bin/python path"
        )
    venv_root = lexical.parent.parent
    if venv_root.is_symlink() or not venv_root.is_dir():
        raise TatqaP18FormalRuntimeError("lexical venv root is unavailable")
    pyvenv = venv_root / "pyvenv.cfg"
    values = _parse_pyvenv(pyvenv)
    major_minor = ".".join(values["version"].split(".")[:2])
    overlay = venv_root / "lib" / f"python{major_minor}" / "site-packages"
    if overlay.is_symlink() or not overlay.is_dir():
        raise TatqaP18FormalRuntimeError("overlay site-packages is unavailable")
    roots: list[tuple[str, Path]] = [("overlay", overlay)]
    if values["include-system-site-packages"] == "true":
        home = Path(values["home"])
        if not home.is_absolute():
            raise TatqaP18FormalRuntimeError("pyvenv.cfg home is not absolute")
        base = home.parent / "lib" / f"python{major_minor}" / "site-packages"
        if base.is_symlink() or not base.is_dir():
            raise TatqaP18FormalRuntimeError("base site-packages is unavailable")
        if base.resolve(strict=True) != overlay.resolve(strict=True):
            roots.append(("base", base))
    return pyvenv, roots


def _attested_dependency_rows(
    *, runtime_python: Path, hippo_attestation: Path
) -> tuple[list[dict[str, object]], Path, list[tuple[str, Path]]]:
    try:
        attestation = json.loads(hippo_attestation.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TatqaP18FormalRuntimeError("HippoRAG attestation is invalid") from exc
    filesystem = (
        attestation.get("runtime_filesystem_binding")
        if isinstance(attestation, Mapping)
        else None
    )
    expected_rows = (
        filesystem.get("dependency_metadata_rows")
        if isinstance(filesystem, Mapping)
        else None
    )
    if not isinstance(expected_rows, list) or not expected_rows:
        raise TatqaP18FormalRuntimeError(
            "HippoRAG dependency attestation is absent"
        )
    pyvenv, roots = _runtime_search_roots(runtime_python)
    observed_rows: list[dict[str, object]] = []
    for expected_value in expected_rows:
        if not isinstance(expected_value, Mapping):
            raise TatqaP18FormalRuntimeError(
                "HippoRAG dependency attestation row drifted"
            )
        name = expected_value.get("name")
        version = expected_value.get("version")
        if not isinstance(name, str) or (
            version is not None and not isinstance(version, str)
        ):
            raise TatqaP18FormalRuntimeError(
                "HippoRAG dependency attestation identity drifted"
            )
        all_matches: list[tuple[str, Path, str]] = []
        chosen: list[tuple[str, Path, str]] = []
        for role, root in roots:
            role_matches: list[tuple[str, Path, str]] = []
            for dist_info in sorted(root.glob("*.dist-info"), key=lambda row: row.name):
                metadata_path = dist_info / "METADATA"
                if not metadata_path.is_file():
                    continue
                observed_name, observed_version = _metadata_identity(metadata_path)
                if observed_name != _canonical_distribution_name(name):
                    continue
                match = (role, dist_info, observed_version)
                all_matches.append(match)
                if observed_version == version:
                    role_matches.append(match)
            if version is not None and role_matches:
                if len(role_matches) != 1 or len(
                    [row for row in all_matches if row[0] == role]
                ) != 1:
                    raise TatqaP18FormalRuntimeError(
                        "HippoRAG dependency metadata is ambiguous"
                    )
                chosen.extend(role_matches)
                break
        if version is None:
            if all_matches:
                raise TatqaP18FormalRuntimeError(
                    "attested-absent HippoRAG dependency is installed"
                )
            observed = {
                "dist_info_file_count": 0,
                "dist_info_name": None,
                "dist_info_tree_sha256": None,
                "name": name,
                "root_role": None,
                "version": None,
            }
        else:
            if len(chosen) != 1:
                raise TatqaP18FormalRuntimeError(
                    "attested HippoRAG dependency metadata is not unique"
                )
            role, dist_info, observed_version = chosen[0]
            count, tree_hash = _attestation_tree_binding(dist_info)
            observed = {
                "dist_info_file_count": count,
                "dist_info_name": dist_info.name,
                "dist_info_tree_sha256": tree_hash,
                "name": name,
                "root_role": role,
                "version": observed_version,
            }
        if dict(expected_value) != observed:
            raise TatqaP18FormalRuntimeError(
                "live HippoRAG dependency metadata drifted"
            )
        observed_rows.append(observed)
    expected_set = filesystem.get("dependency_metadata_set_sha256")
    if expected_set != stable_hash(observed_rows):
        raise TatqaP18FormalRuntimeError(
            "HippoRAG dependency metadata set drifted"
        )
    if filesystem.get("pyvenv_cfg_sha256") != file_sha256(pyvenv):
        raise TatqaP18FormalRuntimeError("HippoRAG pyvenv binding drifted")
    return observed_rows, pyvenv, roots


def runtime_inventory_snapshot(
    *,
    runtime_python: Path,
    qwen_model: Path,
    minilm_manifest: Path,
    hippo_attestation: Path,
) -> dict[str, object]:
    """Reproducible content-free inventory bound to the active interpreter.

    MiniLM is in-process, so accepting a different configured interpreter from
    ``sys.executable`` would make the fingerprint describe a runtime that was
    never used.  The same-file assertion is therefore part of both source-free
    qualification and every formal fingerprint re-verification.
    """

    configured = _safe_runtime_executable(runtime_python)
    active_lexical = Path(sys.executable).expanduser().absolute()
    if configured != active_lexical:
        raise TatqaP18FormalRuntimeError(
            "configured runtime Python lexical path differs from sys.executable"
        )
    try:
        active = active_lexical.resolve(strict=True)
        same_interpreter = os.path.samefile(configured, active)
    except OSError as exc:
        raise TatqaP18FormalRuntimeError(
            "active runtime Python cannot be bound"
        ) from exc
    if not same_interpreter:
        raise TatqaP18FormalRuntimeError(
            "configured runtime Python differs from the active interpreter"
        )
    venv_root = configured.parent.parent
    if Path(sys.prefix).expanduser().absolute() != venv_root:
        raise TatqaP18FormalRuntimeError(
            "active sys.prefix differs from the configured lexical venv"
        )
    try:
        versions = {name: metadata.version(name) for name in RUNTIME_DISTRIBUTIONS}
        import torch
    except (metadata.PackageNotFoundError, ImportError) as exc:
        raise TatqaP18FormalRuntimeError(
            "required offline runtime distribution is absent"
        ) from exc

    gpu_rows: list[dict[str, object]] = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        gpu_rows.append(
            {
                "index": index,
                "name": str(properties.name),
                "total_memory_bytes": int(properties.total_memory),
            }
        )
    dependency_rows, pyvenv, search_roots = _attested_dependency_rows(
        runtime_python=configured,
        hippo_attestation=hippo_attestation,
    )
    runtime_binding = _executable_binding(configured, field="runtime Python")
    runtime_binding.update(
        {
            "active_sys_executable_lexical_path_sha256": hashlib.sha256(
                str(active_lexical).encode("utf-8")
            ).hexdigest(),
            "active_sys_executable_resolved_file_sha256": file_sha256(active),
            "samefile_with_active_sys_executable": True,
            "lexical_path_equals_active_sys_executable": True,
            "sys_prefix_path_sha256": hashlib.sha256(
                str(Path(sys.prefix).expanduser().absolute()).encode("utf-8")
            ).hexdigest(),
            "sys_base_prefix_path_sha256": hashlib.sha256(
                str(Path(sys.base_prefix).expanduser().absolute()).encode("utf-8")
            ).hexdigest(),
            "pyvenv_cfg": {
                **_public_file_binding(pyvenv, field="pyvenv.cfg"),
                "lexical_path_sha256": hashlib.sha256(
                    str(pyvenv.absolute()).encode("utf-8")
                ).hexdigest(),
            },
        }
    )
    active_site_packages = sorted(
        {
            str(Path(row).expanduser().absolute())
            for row in site.getsitepackages()
        }
    )
    return {
        "hostname": socket.gethostname(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "executable_bindings": {
            "runtime_python": runtime_binding,
            "systemd_run": _executable_binding(
                SYSTEMD_RUN, field="systemd-run executable"
            ),
            "systemctl": _executable_binding(
                SYSTEMCTL, field="systemctl executable"
            ),
            "environment_clearer": _executable_binding(
                ENV_EXECUTABLE, field="environment clearer"
            ),
            "network_preflight_python": _executable_binding(
                PREFLIGHT_PYTHON, field="network preflight Python"
            ),
        },
        "distribution_versions": dict(sorted(versions.items())),
        "site_packages_roots": [
            {
                "role": role,
                "lexical_path_sha256": hashlib.sha256(
                    str(root.absolute()).encode("utf-8")
                ).hexdigest(),
                "resolved_path_sha256": hashlib.sha256(
                    str(root.resolve(strict=True)).encode("utf-8")
                ).hexdigest(),
            }
            for role, root in search_roots
        ],
        "active_site_packages_path_sha256s": [
            hashlib.sha256(row.encode("utf-8")).hexdigest()
            for row in active_site_packages
        ],
        "hipporag_dependency_metadata_rows": dependency_rows,
        "hipporag_dependency_metadata_set_sha256": stable_hash(dependency_rows),
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_count": len(gpu_rows),
        "cuda_devices": gpu_rows,
        "Qwen_config": _qwen_config_binding(qwen_model),
        "public_file_bindings": {
            "MiniLM_asset_manifest": _public_file_binding(
                minilm_manifest, field="MiniLM asset manifest"
            ),
            "HippoRAG_attestation": _public_file_binding(
                hippo_attestation, field="HippoRAG attestation"
            ),
        },
        "environment_variable_names_or_values_recorded": False,
    }


def _verify_hipporag_attestation_identity(path: Path) -> None:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TatqaP18FormalRuntimeError("HippoRAG attestation is invalid") from exc
    if not isinstance(value, Mapping):
        raise TatqaP18FormalRuntimeError("HippoRAG attestation root drifted")
    normalized = value.get("normalized_llm_asset_binding")
    filesystem = value.get("runtime_filesystem_binding")
    if not isinstance(normalized, Mapping) or not isinstance(filesystem, Mapping):
        raise TatqaP18FormalRuntimeError("HippoRAG attestation identity is absent")
    if (
        value.get("schema") != HIPPORAG_ATTESTATION_SCHEMA
        or value.get("receipt_sha256") != HIPPORAG_ATTESTATION_RECEIPT_SHA256
        or normalized.get("repo_id") != HIPPORAG_ATTESTATION_LLM_REPO_ID
        or normalized.get("model_revision")
        != HIPPORAG_ATTESTATION_LLM_REVISION
        or normalized.get("normalized_asset_sha256")
        != HIPPORAG_ATTESTATION_LLM_ASSET_SHA256
        or normalized.get("normalized_topology_sha256")
        != HIPPORAG_ATTESTATION_LLM_TOPOLOGY_SHA256
        or filesystem.get("local_llm_asset_sha256")
        != HIPPORAG_ATTESTATION_LLM_ASSET_SHA256
        or filesystem.get("local_llm_topology_sha256")
        != HIPPORAG_ATTESTATION_LLM_TOPOLOGY_SHA256
        or filesystem.get("local_embedding_asset_sha256")
        != HIPPORAG_ATTESTATION_EMBEDDING_ASSET_SHA256
        or filesystem.get("local_embedding_topology_sha256")
        != HIPPORAG_ATTESTATION_EMBEDDING_TOPOLOGY_SHA256
        or filesystem.get("official_source_tree_sha256")
        != HIPPORAG_ATTESTATION_OFFICIAL_SOURCE_TREE_SHA256
        or filesystem.get("official_source_file_count")
        != HIPPORAG_ATTESTATION_OFFICIAL_SOURCE_FILE_COUNT
        or filesystem.get("official_source_root_role")
        != HIPPORAG_ATTESTATION_OFFICIAL_SOURCE_ROOT_ROLE
    ):
        raise TatqaP18FormalRuntimeError("HippoRAG attestation identity drifted")


def verify_runtime_fingerprint(paths: RuntimePaths) -> dict[str, Any]:
    checked = paths.checked()
    value = _load_canonical_object(checked.fingerprint_manifest, "runtime fingerprint")
    if (
        value.get("schema") != "tatqa_p18_remote_runtime_fingerprint_v1"
        or value.get("status") != "verified_before_formal_source_open"
        or value.get("study_design_self_sha256") != STUDY_DESIGN_SELF_SHA256
    ):
        raise TatqaP18FormalRuntimeError("runtime fingerprint contract drifted")
    _self_hash(value)
    bindings = value.get("asset_bindings")
    if not isinstance(bindings, Mapping):
        raise TatqaP18FormalRuntimeError("runtime asset bindings are absent")
    expected = {
        "Qwen": QWEN_MODEL_TREE_SHA256,
        "MiniLM": MINILM_GENERIC_TREE_SHA256,
        "HippoRAG_LLM": HIPPORAG_LLM_TREE_SHA256,
        "HippoRAG_embedding": HIPPORAG_EMBEDDING_TREE_SHA256,
        "HippoRAG_source": HIPPORAG_SOURCE_TREE_SHA256,
    }
    observed_paths = {
        "Qwen": checked.qwen_model,
        "MiniLM": checked.minilm_model,
        "HippoRAG_LLM": checked.hippo_llm_model,
        "HippoRAG_embedding": checked.hippo_embedding_model,
        "HippoRAG_source": checked.hipporag_source,
    }
    for name, path in observed_paths.items():
        row = bindings.get(name)
        if not isinstance(row, Mapping):
            raise TatqaP18FormalRuntimeError("runtime asset binding row is absent")
        receipt = tree_receipt(path)
        if dict(row) != receipt:
            raise TatqaP18FormalRuntimeError("runtime asset tree drifted")
        if name in expected and receipt["tree_sha256"] != expected[name]:
            raise TatqaP18FormalRuntimeError("design-bound runtime asset drifted")
    if (
        file_sha256(checked.minilm_asset_manifest)
        != MINILM_EXPECTED_ASSET_FILE_SHA256
        or MINILM_ASSET_FILE_SHA256 != MINILM_EXPECTED_ASSET_FILE_SHA256
        or MINILM_ASSET_SELF_SHA256 != MINILM_EXPECTED_ASSET_SELF_SHA256
        or file_sha256(checked.hippo_attestation)
        != HIPPORAG_ATTESTATION_FILE_SHA256
    ):
        raise TatqaP18FormalRuntimeError("design-bound asset manifest drifted")
    _verify_hipporag_attestation_identity(checked.hippo_attestation)
    expected_inventory = runtime_inventory_snapshot(
        runtime_python=checked.runtime_python,
        qwen_model=checked.qwen_model,
        minilm_manifest=checked.minilm_asset_manifest,
        hippo_attestation=checked.hippo_attestation,
    )
    if value.get("runtime_inventory") != expected_inventory:
        raise TatqaP18FormalRuntimeError("live runtime inventory drifted")
    return value


def _launcher_environment() -> dict[str, str]:
    result = {
        "PATH": "/usr/bin:/bin",
        "HOME": os.environ.get("HOME", "/"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
    }
    for key in ("DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR"):
        value = os.environ.get(key)
        if value:
            result[key] = value
    return result


def _clean_environment_prefix(environment: Mapping[str, str]) -> list[str]:
    if not ENV_EXECUTABLE.is_file():
        raise TatqaP18FormalRuntimeError("environment-clearing executable is absent")
    if any(
        not isinstance(key, str)
        or not key
        or "=" in key
        or "\x00" in key
        or not isinstance(value, str)
        or "\x00" in value
        or "\n" in value
        for key, value in environment.items()
    ):
        raise TatqaP18FormalRuntimeError("worker environment is malformed")
    return [
        str(ENV_EXECUTABLE),
        "--ignore-environment",
        *(f"{key}={environment[key]}" for key in sorted(environment)),
    ]


def _systemd_prefix(
    *,
    inaccessible_paths: Sequence[Path] = (),
    unit_name: str | None = None,
    tasks_max: int | None = None,
) -> list[str]:
    if not SYSTEMD_RUN.is_file():
        raise TatqaP18FormalRuntimeError("systemd-run is unavailable")
    command = [str(SYSTEMD_RUN), *SYSTEMD_FLAGS]
    if unit_name is not None:
        _validate_worker_unit_name(unit_name)
        command.append(f"--unit={unit_name}")
        command.extend(("--property", f"KillMode={SYSTEMD_WORKER_KILL_MODE}"))
    if tasks_max is not None:
        if unit_name is None or tasks_max != HIPPORAG_SYSTEMD_TASKS_MAX:
            raise TatqaP18FormalRuntimeError("systemd TasksMax binding drifted")
        command.extend(("--property", f"TasksMax={tasks_max}"))
    for value in SYSTEMD_NETWORK_PROPERTIES:
        command.extend(("--property", value))
    for path in inaccessible_paths:
        absolute = path.expanduser().absolute()
        if "\x00" in str(absolute) or "\n" in str(absolute):
            raise TatqaP18FormalRuntimeError("inaccessible path is malformed")
        command.extend(("--property", f"InaccessiblePaths=-{absolute}"))
    command.append("--")
    return command


def _worker_inaccessible_paths(paths: RuntimePaths) -> tuple[Path, Path]:
    return (
        paths.project_root / "artifacts/tatqa_p18_official_source_v1",
        paths.project_root / "artifacts/tatqa_p18_formal_v1/acquisition",
    )


def systemd_network_preflight(*, runner=subprocess.run) -> dict[str, object]:
    script = (
        "import os,socket;"
        "assert set(os.environ)=={'LANG'} and os.environ['LANG']=='C.UTF-8';"
        "u=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);u.close();"
        "\nfor f in (socket.AF_INET,socket.AF_INET6):\n"
        " try:s=socket.socket(f,socket.SOCK_STREAM)\n"
        " except OSError:continue\n"
        " s.close();raise SystemExit(41)\n"
    )
    command = (
        _systemd_prefix()
        + _clean_environment_prefix({"LANG": "C.UTF-8"})
        + [str(PREFLIGHT_PYTHON), "-I", "-c", script]
    )
    try:
        completed = runner(
            command,
            check=False,
            capture_output=True,
            env=_launcher_environment(),
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TatqaP18FormalRuntimeError("network isolation preflight failed") from exc
    if completed.returncode != 0:
        raise TatqaP18FormalRuntimeError("network isolation preflight failed")
    return {
        "network_properties": list(SYSTEMD_NETWORK_PROPERTIES),
        "returncode": 0,
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
    }


def _worker_environment(
    paths: RuntimePaths, writable_root: Path, *, role: str
) -> dict[str, str]:
    if role not in {"Qwen", "HippoRAG"}:
        raise TatqaP18FormalRuntimeError("worker role drifted")
    common = {
        "PATH": f"{paths.runtime_python.parent}:/usr/bin:/bin",
        "HOME": str(writable_root / "home"),
        "LANG": "C.UTF-8",
        "HF_HOME": str(writable_root / "hf"),
        "TMPDIR": str(writable_root / "tmp"),
        "TMP": str(writable_root / "tmp"),
        "TEMP": str(writable_root / "tmp"),
        "PYTHONPATH": str(paths.project_root),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    }
    if role == "Qwen":
        common.update(
            {
                "CUDA_VISIBLE_DEVICES": QWEN_PHYSICAL_GPU,
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
    else:
        common.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": str(HIPPORAG_CONFIGURED_TORCH_INTRAOP_THREADS),
                "MKL_NUM_THREADS": str(HIPPORAG_CONFIGURED_TORCH_INTRAOP_THREADS),
                "OPENBLAS_NUM_THREADS": str(HIPPORAG_CONFIGURED_TORCH_INTRAOP_THREADS),
                "NUMEXPR_NUM_THREADS": str(HIPPORAG_CONFIGURED_TORCH_INTRAOP_THREADS),
                "VECLIB_MAXIMUM_THREADS": str(HIPPORAG_CONFIGURED_TORCH_INTRAOP_THREADS),
            }
        )
    return common


def _fresh_stage_root(root: Path, *parts: str) -> Path:
    path = root
    for part in parts:
        path = path / _safe_leaf(part, "stage coordinate")
    try:
        path.mkdir(mode=0o700, parents=False)
    except OSError as exc:
        raise TatqaP18FormalRuntimeError("worker stage root is already consumed") from exc
    for name in ("home", "hf", "tmp"):
        (path / name).mkdir(mode=0o700)
    return path


def _validate_worker_unit_name(value: object) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"tatqa-p18-(?:qwen|hippo)-[0-9a-f]{40}\.service", value)
        is None
    ):
        raise TatqaP18FormalRuntimeError("worker systemd unit name drifted")
    return value


def _worker_unit_name(*, role: str, identity: Mapping[str, object]) -> str:
    if role not in {"Qwen", "HippoRAG"}:
        raise TatqaP18FormalRuntimeError("worker unit role drifted")
    digest = stable_hash(
        {
            "identity": dict(identity),
            "role": role,
            "schema": f"{VERSION}_worker_unit_identity_v1",
        }
    )
    tag = "qwen" if role == "Qwen" else "hippo"
    return _validate_worker_unit_name(f"tatqa-p18-{tag}-{digest[:40]}.service")


def _unit_name_sha256(unit_name: str) -> str:
    return hashlib.sha256(_validate_worker_unit_name(unit_name).encode("ascii")).hexdigest()


def _systemctl_environment() -> dict[str, str]:
    value = _launcher_environment()
    value["LANG"] = "C.UTF-8"
    return value


def _cgroup_task_counts(control_group: str) -> tuple[int, int]:
    """Count all processes and threads below a systemd cgroup recursively."""

    if not isinstance(control_group, str) or "\x00" in control_group or "\n" in control_group:
        raise TatqaP18FormalRuntimeError("systemd control group drifted")
    if not control_group:
        return 0, 0
    pure = PurePosixPath(control_group)
    if not pure.is_absolute() or ".." in pure.parts:
        raise TatqaP18FormalRuntimeError("systemd control group drifted")
    root = CGROUP_ROOT.joinpath(*pure.parts[1:])
    if root.is_symlink():
        raise TatqaP18FormalRuntimeError("systemd control group is a symlink")
    if not root.exists():
        return 0, 0
    if not root.is_dir():
        raise TatqaP18FormalRuntimeError("systemd control group is not a directory")
    processes: set[int] = set()
    threads: set[int] = set()
    try:
        for current, directories, _files in os.walk(root, followlinks=False):
            base = Path(current)
            for name in directories:
                if (base / name).is_symlink():
                    raise TatqaP18FormalRuntimeError(
                        "systemd control group contains a symlink"
                    )
            for filename, target in (
                ("cgroup.procs", processes),
                ("cgroup.threads", threads),
            ):
                receipt = base / filename
                if receipt.is_file():
                    for line in receipt.read_text(encoding="ascii").splitlines():
                        if not line.isdecimal() or int(line) <= 1:
                            raise TatqaP18FormalRuntimeError(
                                "systemd control group task receipt drifted"
                            )
                        target.add(int(line))
    except OSError as exc:
        raise TatqaP18FormalRuntimeError(
            "systemd control group tasks are unreadable"
        ) from exc
    return len(processes), len(threads)


@dataclass(frozen=True)
class _SystemdUnitState:
    load_state: str
    active_state: str
    sub_state: str
    main_pid: int
    control_group: str
    control_group_process_count: int
    control_group_thread_count: int
    show_returncode: int
    show_stdout_sha256: str
    show_stderr_sha256: str


@dataclass
class _TrackedSystemdUnit:
    name: str
    name_sha256: str
    state_lock: threading.RLock = field(default_factory=threading.RLock)
    close_lock: threading.Lock = field(default_factory=threading.Lock)
    process: Any | None = None
    abort_requested: bool = False
    last_control_group: str = ""
    start_policy: dict[str, Any] | None = None
    closure: dict[str, Any] | None = None


def _validate_unit_closure_receipt(
    value: object, *, expected_unit_name_sha256: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TatqaP18FormalRuntimeError("systemd unit closure receipt drifted")
    receipt = dict(value)
    expected_keys = {
        "active_state",
        "control_group_process_count",
        "control_group_sha256",
        "control_group_thread_count",
        "load_state",
        "main_pid",
        "schema",
        "sub_state",
        "systemctl_reset_failed_returncode",
        "systemctl_reset_failed_stderr_sha256",
        "systemctl_reset_failed_stdout_sha256",
        "systemctl_show_returncode",
        "systemctl_show_stderr_sha256",
        "systemctl_show_stdout_sha256",
        "unit_name_sha256",
    }
    hash_fields = {
        "control_group_sha256",
        "systemctl_reset_failed_stderr_sha256",
        "systemctl_reset_failed_stdout_sha256",
        "systemctl_show_stderr_sha256",
        "systemctl_show_stdout_sha256",
        "unit_name_sha256",
    }
    if (
        set(receipt) != expected_keys
        or receipt.get("schema") != SYSTEMD_UNIT_CLOSURE_SCHEMA
        or receipt.get("unit_name_sha256") != expected_unit_name_sha256
        or any(
            not isinstance(receipt.get(key), str)
            or _HEX64.fullmatch(receipt[key]) is None
            for key in hash_fields
        )
        or receipt.get("load_state") != "not-found"
        or receipt.get("active_state") != "inactive"
        or receipt.get("sub_state") != "dead"
        or type(receipt.get("main_pid")) is not int
        or receipt.get("main_pid") != 0
        or type(receipt.get("control_group_process_count")) is not int
        or receipt.get("control_group_process_count") != 0
        or type(receipt.get("control_group_thread_count")) is not int
        or receipt.get("control_group_thread_count") != 0
        or type(receipt.get("systemctl_show_returncode")) is not int
        or receipt.get("systemctl_show_returncode") != 0
        or type(receipt.get("systemctl_reset_failed_returncode")) is not int
        or receipt.get("systemctl_reset_failed_returncode") not in {0, 1}
    ):
        raise TatqaP18FormalRuntimeError("systemd unit closure receipt drifted")
    return receipt


def _validate_start_policy_receipt(
    value: object, *, expected_unit_name_sha256: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TatqaP18FormalRuntimeError("systemd start-policy receipt drifted")
    receipt = dict(value)
    expected_keys = {
        "active_state",
        "control_group_sha256",
        "kill_mode",
        "load_state",
        "main_pid",
        "schema",
        "sub_state",
        "systemctl_show_returncode",
        "systemctl_show_stderr_sha256",
        "systemctl_show_stdout_sha256",
        "tasks_max",
        "unit_name_sha256",
    }
    if (
        set(receipt) != expected_keys
        or receipt.get("schema") != SYSTEMD_START_POLICY_SCHEMA
        or receipt.get("unit_name_sha256") != expected_unit_name_sha256
        or receipt.get("load_state") != "loaded"
        or receipt.get("active_state") != "active"
        or receipt.get("sub_state") != "running"
        or type(receipt.get("main_pid")) is not int
        or receipt["main_pid"] <= 1
        or type(receipt.get("tasks_max")) is not int
        or receipt.get("tasks_max") != HIPPORAG_SYSTEMD_TASKS_MAX
        or receipt.get("kill_mode") != SYSTEMD_WORKER_KILL_MODE
        or type(receipt.get("systemctl_show_returncode")) is not int
        or receipt.get("systemctl_show_returncode") != 0
        or any(
            not isinstance(receipt.get(key), str)
            or _HEX64.fullmatch(receipt[key]) is None
            for key in (
                "control_group_sha256",
                "systemctl_show_stderr_sha256",
                "systemctl_show_stdout_sha256",
                "unit_name_sha256",
            )
        )
    ):
        raise TatqaP18FormalRuntimeError("systemd start-policy receipt drifted")
    return receipt


class _SystemdWorkerSupervisor:
    """Own named transient units and prove control-group-wide termination."""

    def __init__(
        self,
        *,
        popen_factory=subprocess.Popen,
        systemctl_runner=subprocess.run,
        sleeper=time.sleep,
        cgroup_counter=_cgroup_task_counts,
    ) -> None:
        self._popen_factory = popen_factory
        self._systemctl_runner = systemctl_runner
        self._sleeper = sleeper
        self._cgroup_counter = cgroup_counter
        self._registry_lock = threading.RLock()
        self._records: dict[str, _TrackedSystemdUnit] = {}
        self._sealed = False

    def _systemctl(self, arguments: Sequence[str], *, timeout: int):
        if not SYSTEMCTL.is_file():
            raise TatqaP18FormalRuntimeError("systemctl is unavailable")
        command = [str(SYSTEMCTL), "--user", *arguments]
        try:
            completed = self._systemctl_runner(
                command,
                check=False,
                capture_output=True,
                env=_systemctl_environment(),
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TatqaP18FormalRuntimeError("systemctl worker control failed") from exc
        if (
            isinstance(completed.returncode, bool)
            or not isinstance(completed.returncode, int)
            or not isinstance(completed.stdout, bytes)
            or not isinstance(completed.stderr, bytes)
        ):
            raise TatqaP18FormalRuntimeError("systemctl result drifted")
        return completed

    def _query(self, record: _TrackedSystemdUnit) -> _SystemdUnitState:
        completed = self._systemctl(
            [
                "show",
                record.name,
                "--no-pager",
                "--property=LoadState",
                "--property=ActiveState",
                "--property=SubState",
                "--property=MainPID",
                "--property=ControlGroup",
            ],
            timeout=SYSTEMD_CONTROL_TIMEOUT_SECONDS,
        )
        if completed.returncode != 0:
            raise TatqaP18FormalRuntimeError("systemctl show failed")
        try:
            text = completed.stdout.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TatqaP18FormalRuntimeError("systemctl show output drifted") from exc
        rows: dict[str, str] = {}
        for line in text.splitlines():
            if "=" not in line:
                raise TatqaP18FormalRuntimeError("systemctl show output drifted")
            key, value = line.split("=", 1)
            if key in rows:
                raise TatqaP18FormalRuntimeError("systemctl show output duplicated")
            rows[key] = value
        if set(rows) != {
            "ActiveState",
            "ControlGroup",
            "LoadState",
            "MainPID",
            "SubState",
        }:
            raise TatqaP18FormalRuntimeError("systemctl show property registry drifted")
        try:
            main_pid = int(rows["MainPID"])
        except ValueError as exc:
            raise TatqaP18FormalRuntimeError("systemd MainPID drifted") from exc
        if (
            main_pid < 0
            or not rows["LoadState"]
            or not rows["ActiveState"]
            or not rows["SubState"]
            or any(
                "\x00" in rows[key] or "\n" in rows[key]
                for key in ("LoadState", "ActiveState", "SubState", "ControlGroup")
            )
        ):
            raise TatqaP18FormalRuntimeError("systemd unit state drifted")
        processes, threads = self._cgroup_counter(rows["ControlGroup"])
        if (
            isinstance(processes, bool)
            or not isinstance(processes, int)
            or processes < 0
            or isinstance(threads, bool)
            or not isinstance(threads, int)
            or threads < 0
        ):
            raise TatqaP18FormalRuntimeError("systemd cgroup count drifted")
        if rows["ControlGroup"]:
            with record.state_lock:
                record.last_control_group = rows["ControlGroup"]
        return _SystemdUnitState(
            load_state=rows["LoadState"],
            active_state=rows["ActiveState"],
            sub_state=rows["SubState"],
            main_pid=main_pid,
            control_group=rows["ControlGroup"],
            control_group_process_count=processes,
            control_group_thread_count=threads,
            show_returncode=completed.returncode,
            show_stdout_sha256=hashlib.sha256(completed.stdout).hexdigest(),
            show_stderr_sha256=hashlib.sha256(completed.stderr).hexdigest(),
        )

    def _observe_start_policy(
        self, record: _TrackedSystemdUnit
    ) -> dict[str, Any]:
        for _ in range(SYSTEMD_KILL_POLL_ATTEMPTS):
            completed = self._systemctl(
                [
                    "show",
                    record.name,
                    "--no-pager",
                    "--property=LoadState",
                    "--property=ActiveState",
                    "--property=SubState",
                    "--property=MainPID",
                    "--property=ControlGroup",
                    "--property=TasksMax",
                    "--property=KillMode",
                ],
                timeout=SYSTEMD_CONTROL_TIMEOUT_SECONDS,
            )
            if completed.returncode != 0:
                raise TatqaP18FormalRuntimeError(
                    "systemctl start-policy show failed"
                )
            try:
                text = completed.stdout.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise TatqaP18FormalRuntimeError(
                    "systemctl start-policy output drifted"
                ) from exc
            rows: dict[str, str] = {}
            for line in text.splitlines():
                if "=" not in line:
                    raise TatqaP18FormalRuntimeError(
                        "systemctl start-policy output drifted"
                    )
                key, raw = line.split("=", 1)
                if key in rows:
                    raise TatqaP18FormalRuntimeError(
                        "systemctl start-policy output duplicated"
                    )
                rows[key] = raw
            if set(rows) != {
                "ActiveState",
                "ControlGroup",
                "KillMode",
                "LoadState",
                "MainPID",
                "SubState",
                "TasksMax",
            }:
                raise TatqaP18FormalRuntimeError(
                    "systemctl start-policy registry drifted"
                )
            try:
                main_pid = int(rows["MainPID"])
                tasks_max = int(rows["TasksMax"])
            except ValueError as exc:
                raise TatqaP18FormalRuntimeError(
                    "systemd start-policy numeric field drifted"
                ) from exc
            if (
                rows["LoadState"] == "loaded"
                and rows["ActiveState"] == "active"
                and rows["SubState"] == "running"
                and main_pid > 1
                and rows["ControlGroup"].startswith("/")
            ):
                # Validate the live cgroup path while it exists and retain only
                # its hash; the closure proof later reopens this exact path and
                # requires it to contain no processes or threads.
                self._cgroup_counter(rows["ControlGroup"])
                with record.state_lock:
                    record.last_control_group = rows["ControlGroup"]
                receipt = {
                    "schema": SYSTEMD_START_POLICY_SCHEMA,
                    "unit_name_sha256": record.name_sha256,
                    "load_state": rows["LoadState"],
                    "active_state": rows["ActiveState"],
                    "sub_state": rows["SubState"],
                    "main_pid": main_pid,
                    "control_group_sha256": hashlib.sha256(
                        rows["ControlGroup"].encode("utf-8")
                    ).hexdigest(),
                    "tasks_max": tasks_max,
                    "kill_mode": rows["KillMode"],
                    "systemctl_show_returncode": completed.returncode,
                    "systemctl_show_stdout_sha256": hashlib.sha256(
                        completed.stdout
                    ).hexdigest(),
                    "systemctl_show_stderr_sha256": hashlib.sha256(
                        completed.stderr
                    ).hexdigest(),
                }
                record.start_policy = _validate_start_policy_receipt(
                    receipt, expected_unit_name_sha256=record.name_sha256
                )
                return dict(record.start_policy)
            if rows["ActiveState"] == "failed":
                raise TatqaP18FormalRuntimeError(
                    "HippoRAG unit terminated before start-policy attestation"
                )
            if rows["LoadState"] == "not-found":
                with record.state_lock:
                    process = record.process
                if process is None or process.poll() is not None:
                    raise TatqaP18FormalRuntimeError(
                        "HippoRAG unit terminated before start-policy attestation"
                    )
            self._sleeper(SYSTEMD_CLOSE_POLL_SECONDS)
        raise TatqaP18FormalRuntimeError(
            "HippoRAG unit never exposed its active kernel policy"
        )

    @staticmethod
    def _terminal_without_tasks(state: _SystemdUnitState) -> bool:
        if (
            state.main_pid != 0
            or state.control_group_process_count != 0
            or state.control_group_thread_count != 0
        ):
            return False
        if state.load_state == "not-found":
            return (
                state.active_state == "inactive"
                and state.sub_state == "dead"
                and state.control_group == ""
            )
        return state.load_state == "loaded" and state.active_state in {
            "inactive",
            "failed",
        }

    def _best_effort_control(self, arguments: Sequence[str]) -> None:
        try:
            self._systemctl(arguments, timeout=SYSTEMD_CONTROL_TIMEOUT_SECONDS)
        except TatqaP18FormalRuntimeError:
            # Only the subsequently observed MainPID/cgroup state can authorize
            # closure; a failed client-side control request never does.
            pass

    def _poll_terminal(
        self, record: _TrackedSystemdUnit, *, attempts: int
    ) -> _SystemdUnitState | None:
        for _ in range(attempts):
            state = self._query(record)
            if self._terminal_without_tasks(state):
                return state
            self._sleeper(SYSTEMD_CLOSE_POLL_SECONDS)
        return None

    def _finalize(self, record: _TrackedSystemdUnit) -> dict[str, Any]:
        with record.close_lock:
            if record.closure is not None:
                return _validate_unit_closure_receipt(
                    record.closure,
                    expected_unit_name_sha256=record.name_sha256,
                )
            state = self._query(record)
            if not self._terminal_without_tasks(state):
                self._best_effort_control(["stop", record.name])
                state = self._poll_terminal(
                    record, attempts=SYSTEMD_STOP_POLL_ATTEMPTS
                )
            if state is None or not self._terminal_without_tasks(state):
                self._best_effort_control(
                    ["kill", "--kill-whom=all", "--signal=SIGKILL", record.name]
                )
                state = self._poll_terminal(
                    record, attempts=SYSTEMD_KILL_POLL_ATTEMPTS
                )
            if state is None or not self._terminal_without_tasks(state):
                raise TatqaP18FormalRuntimeError(
                    "systemd worker unit retained MainPID or cgroup tasks"
                )

            reset = self._systemctl(
                ["reset-failed", record.name],
                timeout=SYSTEMD_CONTROL_TIMEOUT_SECONDS,
            )
            if reset.returncode not in {0, 1}:
                raise TatqaP18FormalRuntimeError("systemctl reset-failed drifted")
            absent_reset_stderr = (
                "Failed to reset failed state of unit "
                f"{record.name}: Unit {record.name} not loaded.\n"
            ).encode("utf-8")
            if (
                reset.stdout != b""
                or (
                    reset.returncode == 0
                    and reset.stderr != b""
                )
                or (
                    reset.returncode == 1
                    and reset.stderr != absent_reset_stderr
                )
            ):
                raise TatqaP18FormalRuntimeError(
                    "systemctl reset-failed normalization drifted"
                )
            final: _SystemdUnitState | None = None
            for _ in range(SYSTEMD_KILL_POLL_ATTEMPTS):
                observed = self._query(record)
                if (
                    observed.load_state == "not-found"
                    and self._terminal_without_tasks(observed)
                ):
                    final = observed
                    break
                self._sleeper(SYSTEMD_CLOSE_POLL_SECONDS)
            if final is None:
                raise TatqaP18FormalRuntimeError(
                    "systemd worker unit was not garbage-collected"
                )
            if reset.returncode == 1 and final.load_state != "not-found":
                raise TatqaP18FormalRuntimeError(
                    "absent-unit reset normalization drifted"
                )
            with record.state_lock:
                last_control_group = record.last_control_group
            final_processes, final_threads = self._cgroup_counter(last_control_group)
            if (
                type(final_processes) is not int
                or type(final_threads) is not int
                or final_processes != 0
                or final_threads != 0
            ):
                raise TatqaP18FormalRuntimeError(
                    "closed systemd control group retained tasks"
                )
            closure = {
                "schema": SYSTEMD_UNIT_CLOSURE_SCHEMA,
                "unit_name_sha256": record.name_sha256,
                "load_state": final.load_state,
                "active_state": final.active_state,
                "sub_state": final.sub_state,
                "main_pid": final.main_pid,
                "control_group_sha256": hashlib.sha256(
                    last_control_group.encode("utf-8")
                ).hexdigest(),
                "control_group_process_count": final_processes,
                "control_group_thread_count": final_threads,
                "systemctl_show_returncode": final.show_returncode,
                "systemctl_show_stdout_sha256": final.show_stdout_sha256,
                "systemctl_show_stderr_sha256": final.show_stderr_sha256,
                "systemctl_reset_failed_returncode": reset.returncode,
                "systemctl_reset_failed_stdout_sha256": hashlib.sha256(
                    reset.stdout
                ).hexdigest(),
                "systemctl_reset_failed_stderr_sha256": hashlib.sha256(
                    reset.stderr
                ).hexdigest(),
            }
            record.closure = _validate_unit_closure_receipt(
                closure, expected_unit_name_sha256=record.name_sha256
            )
            return dict(record.closure)

    def _reserve(self, unit_name: str) -> _TrackedSystemdUnit:
        unit_name = _validate_worker_unit_name(unit_name)
        with self._registry_lock:
            if self._sealed:
                raise TatqaP18FormalRuntimeError(
                    "worker supervisor is terminally sealed"
                )
            if unit_name in self._records:
                raise TatqaP18FormalRuntimeError("worker systemd unit replayed")
            record = _TrackedSystemdUnit(
                name=unit_name,
                name_sha256=_unit_name_sha256(unit_name),
            )
            initial = self._query(record)
            if not (
                initial.load_state == "not-found"
                and self._terminal_without_tasks(initial)
            ):
                raise TatqaP18FormalRuntimeError(
                    "worker systemd unit name was already consumed"
                )
            self._records[unit_name] = record
            return record

    @staticmethod
    def _kill_client(process: Any) -> None:
        try:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=SYSTEMD_CLIENT_REAP_TIMEOUT_SECONDS)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TatqaP18FormalRuntimeError(
                "systemd-run client could not be terminated"
            ) from exc

    def run(
        self,
        *,
        full_command: Sequence[str],
        timeout: int,
        unit_name: str,
        role: str,
    ) -> tuple[bytes, bytes, dict[str, Any] | None, dict[str, Any]]:
        if role not in {"Qwen", "HippoRAG"}:
            raise TatqaP18FormalRuntimeError("worker role drifted")
        record = self._reserve(unit_name)
        process: Any | None = None
        pending: BaseException | None = None
        stdout = b""
        stderr = b""
        try:
            with record.state_lock:
                if record.abort_requested:
                    raise TatqaP18FormalRuntimeError(
                        "worker launch was cancelled before systemd-run"
                    )
                process = self._popen_factory(
                    list(full_command),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=_launcher_environment(),
                    start_new_session=True,
                )
                record.process = process
            if role == "HippoRAG":
                self._observe_start_policy(record)
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                pending = TatqaP18FormalRuntimeError(
                    "offline worker systemd-run client timed out"
                )
                self._kill_client(process)
                try:
                    stdout, stderr = process.communicate(timeout=1)
                except BaseException:
                    stdout, stderr = b"", b""
                pending.__cause__ = exc
            except BaseException as exc:
                pending = exc
                self._kill_client(process)
        except BaseException as exc:
            pending = exc
            if process is not None:
                self._kill_client(process)

        try:
            closure = self._finalize(record)
        except BaseException:
            # A broken status query must not prevent the control action itself.
            # The caller will still fail closed because no closure receipt can
            # be emitted, while terminal abort can later aggregate/reverify all
            # known units.
            self._best_effort_control(["stop", record.name])
            self._best_effort_control(
                ["kill", "--kill-whom=all", "--signal=SIGKILL", record.name]
            )
            if process is not None:
                try:
                    self._kill_client(process)
                except TatqaP18FormalRuntimeError:
                    pass
            raise
        with record.state_lock:
            aborted = record.abort_requested
        if aborted and pending is None:
            pending = TatqaP18FormalRuntimeError(
                "offline worker was aborted after named-unit launch"
            )
        if pending is not None:
            raise pending
        if (
            not isinstance(stdout, bytes)
            or not isinstance(stderr, bytes)
            or isinstance(process.returncode, bool)
            or not isinstance(process.returncode, int)
        ):
            raise TatqaP18FormalRuntimeError("systemd-run client result drifted")
        if process.returncode != 0:
            raise TatqaP18FormalRuntimeError(
                "offline worker failed; "
                f"returncode={process.returncode};"
                f"stdout_sha256={hashlib.sha256(stdout).hexdigest()};"
                f"stderr_sha256={hashlib.sha256(stderr).hexdigest()}"
            )
        start_policy = (
            _validate_start_policy_receipt(
                record.start_policy,
                expected_unit_name_sha256=record.name_sha256,
            )
            if role == "HippoRAG"
            else None
        )
        return stdout, stderr, start_policy, closure

    def abort_all_workers(self) -> tuple[dict[str, Any], ...]:
        with self._registry_lock:
            # This is a terminal operation.  Sealing and snapshotting under the
            # same lock linearizes abort against _reserve: a reservation either
            # becomes part of this snapshot or is rejected before launch.
            self._sealed = True
            records = tuple(self._records.values())
        errors: list[BaseException] = []
        for record in records:
            with record.state_lock:
                record.abort_requested = True
                process = record.process
            self._best_effort_control(["stop", record.name])
            self._best_effort_control(
                ["kill", "--kill-whom=all", "--signal=SIGKILL", record.name]
            )
            if process is not None:
                try:
                    self._kill_client(process)
                except BaseException as exc:
                    errors.append(exc)
        closures: list[dict[str, Any]] = []
        for record in records:
            try:
                closures.append(self._finalize(record))
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise TatqaP18FormalRuntimeError(
                "terminal worker abort encountered cleanup failures after attempting all units"
            ) from errors[0]
        return tuple(closures)

    def verify_all_workers_closed(self) -> tuple[dict[str, Any], ...]:
        with self._registry_lock:
            records = tuple(self._records.values())
        closures: list[dict[str, Any]] = []
        for record in records:
            with record.state_lock:
                process = record.process
                closure = record.closure
                last_control_group = record.last_control_group
            if process is not None and process.poll() is None:
                raise TatqaP18FormalRuntimeError(
                    "systemd-run client remained active after worker terminal"
                )
            state = self._query(record)
            if not (
                state.load_state == "not-found"
                and self._terminal_without_tasks(state)
            ):
                raise TatqaP18FormalRuntimeError(
                    "systemd worker unit reopened or remained active"
                )
            if closure is None:
                raise TatqaP18FormalRuntimeError(
                    "systemd worker closure receipt is absent"
                )
            processes, threads = self._cgroup_counter(last_control_group)
            if (
                type(processes) is not int
                or type(threads) is not int
                or processes != 0
                or threads != 0
            ):
                raise TatqaP18FormalRuntimeError(
                    "closed systemd control group regained tasks"
                )
            checked = _validate_unit_closure_receipt(
                closure, expected_unit_name_sha256=record.name_sha256
            )
            if checked["control_group_sha256"] != hashlib.sha256(
                last_control_group.encode("utf-8")
            ).hexdigest():
                raise TatqaP18FormalRuntimeError(
                    "systemd closure control-group binding drifted"
                )
            closures.append(checked)
        return tuple(closures)


class _SupervisedWorkerCapability:
    def _initialize_worker_supervisor(
        self,
        *,
        popen_factory=subprocess.Popen,
        systemctl_runner=subprocess.run,
    ) -> None:
        self._worker_supervisor = _SystemdWorkerSupervisor(
            popen_factory=popen_factory,
            systemctl_runner=systemctl_runner,
        )

    def abort_all_workers(self) -> tuple[dict[str, Any], ...]:
        return self._worker_supervisor.abort_all_workers()

    def verify_all_workers_closed(self) -> tuple[dict[str, Any], ...]:
        return self._worker_supervisor.verify_all_workers_closed()


def _run_worker(
    *,
    command: Sequence[str],
    environment: Mapping[str, str],
    inaccessible_paths: Sequence[Path],
    timeout: int,
    role: str,
    unit_name: str,
    supervisor: _SystemdWorkerSupervisor,
) -> tuple[bytes, bytes, dict[str, Any] | None, dict[str, Any]]:
    if role not in {"Qwen", "HippoRAG"}:
        raise TatqaP18FormalRuntimeError("worker role drifted")
    tasks_max = HIPPORAG_SYSTEMD_TASKS_MAX if role == "HippoRAG" else None
    full = (
        _systemd_prefix(
            inaccessible_paths=inaccessible_paths,
            unit_name=unit_name,
            tasks_max=tasks_max,
        )
        + _clean_environment_prefix(environment)
        + list(command)
    )
    return supervisor.run(
        full_command=full,
        timeout=timeout,
        unit_name=unit_name,
        role=role,
    )


def _terminal_status(stdout: bytes, *, role: str, item_count: int) -> dict[str, Any]:
    try:
        value = json.loads(stdout.decode("utf-8").strip().splitlines()[-1])
    except (IndexError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TatqaP18FormalRuntimeError("worker terminal receipt is absent") from exc
    if not isinstance(value, dict) or value.get("status") != "passed":
        raise TatqaP18FormalRuntimeError("worker terminal receipt drifted")
    pid = value.get("worker_pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        raise TatqaP18FormalRuntimeError("worker PID receipt drifted")
    started = value.get("model_execution_started_monotonic_ns")
    finished = value.get("model_execution_finished_monotonic_ns")
    if (
        isinstance(started, bool)
        or isinstance(finished, bool)
        or not isinstance(started, int)
        or not isinstance(finished, int)
        or started < 0
        or finished <= started
    ):
        raise TatqaP18FormalRuntimeError("worker model execution interval drifted")
    if role == "Qwen":
        if (
            set(value)
            != {
                "generation_valid_count",
                "item_count",
                "model_execution_finished_monotonic_ns",
                "model_execution_started_monotonic_ns",
                "model_context_tokens",
                "status",
                "worker_pid",
            }
            or value.get("item_count") != item_count
            or isinstance(value.get("generation_valid_count"), bool)
            or not isinstance(value.get("generation_valid_count"), int)
            or not 0 <= value["generation_valid_count"] <= item_count
            or isinstance(value.get("model_context_tokens"), bool)
            or not isinstance(value.get("model_context_tokens"), int)
            or value["model_context_tokens"] < 16_640
        ):
            raise TatqaP18FormalRuntimeError("Qwen terminal receipt drifted")
    elif role == "HippoRAG":
        if (
            set(value)
            != {
                "graph_edge_count",
                "graph_node_count",
                "configured_torch_interop_threads",
                "configured_torch_intraop_threads",
                "model_execution_finished_monotonic_ns",
                "model_execution_started_monotonic_ns",
                "observed_process_thread_peak",
                "status",
                "unit_count",
                "worker_pid",
            }
            or value.get("unit_count") != item_count
            or value.get("configured_torch_intraop_threads")
            != HIPPORAG_CONFIGURED_TORCH_INTRAOP_THREADS
            or value.get("configured_torch_interop_threads")
            != HIPPORAG_CONFIGURED_TORCH_INTEROP_THREADS
            or isinstance(value.get("observed_process_thread_peak"), bool)
            or not isinstance(value.get("observed_process_thread_peak"), int)
            or not 1
            <= value["observed_process_thread_peak"]
            <= HIPPORAG_CPU_THREADS
            or any(
                isinstance(value.get(key), bool)
                or not isinstance(value.get(key), int)
                or value[key] < 0
                for key in ("graph_edge_count", "graph_node_count")
            )
        ):
            raise TatqaP18FormalRuntimeError("HippoRAG terminal receipt drifted")
    else:
        raise TatqaP18FormalRuntimeError("worker role drifted")
    return value


class BoundMiniLMEncoder:
    """Exact CPU float32 MiniLM binding used by the production adapter."""

    def __init__(self, paths: RuntimePaths) -> None:
        self.paths = paths.checked()
        self.fingerprint = verify_runtime_fingerprint(self.paths)
        self._encoder = OfflineMiniLMEncoder(
            asset_manifest_path=self.paths.minilm_asset_manifest,
            model_root=self.paths.minilm_model,
            run_canary=True,
        )
        self.runtime_receipt = dict(self._encoder.runtime_receipt)
        self.canary_receipt = dict(self._encoder.canary_receipt)

    def encode(self, texts: Sequence[str]):
        return self._encoder.encode(texts)


class SystemdTypedPlanBatchRunner(_SupervisedWorkerCapability):
    """Canonical-byte Qwen boundary consumed by the custody adapter."""

    def __init__(
        self,
        paths: RuntimePaths,
        *,
        popen_factory=subprocess.Popen,
        systemctl_runner=subprocess.run,
    ) -> None:
        self.paths = paths.checked()
        self._initialize_worker_supervisor(
            popen_factory=popen_factory, systemctl_runner=systemctl_runner
        )
        self.receipts: dict[str, dict[str, Any]] = {}

    def __call__(self, block: str, canonical_input: bytes) -> bytes:
        block = _safe_leaf(block, "block")
        if block in self.receipts:
            raise TatqaP18FormalRuntimeError("typed-plan block replay is forbidden")
        if not isinstance(canonical_input, bytes):
            raise TatqaP18FormalRuntimeError("typed-plan input is not bytes")
        projected = typed_plan_contract.parse_input(canonical_input)
        root = _fresh_stage_root(self.paths.work_root, f"qwen_{block}")
        input_path = root / "typed_plan.input.json"
        output_path = root / "typed_plan.output.json"
        input_sha = _write_exclusive(input_path, canonical_input)
        command = [
            str(self.paths.runtime_python),
            "-B",
            "-m",
            "replication_runtime.tatqa_p18_v1.typed_plan_worker",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--model",
            str(self.paths.qwen_model),
            "--batch-size",
            str(QWEN_BATCH_SIZE),
        ]
        unit_name = _worker_unit_name(
            role="Qwen",
            identity={
                "block": block,
                "input_sha256": input_sha,
                "stage_root_sha256": hashlib.sha256(
                    str(root.absolute()).encode("utf-8")
                ).hexdigest(),
            },
        )
        stdout, stderr, start_policy, closure = _run_worker(
            command=command,
            environment=_worker_environment(self.paths, root, role="Qwen"),
            inaccessible_paths=_worker_inaccessible_paths(self.paths),
            timeout=QWEN_TIMEOUT_SECONDS,
            role="Qwen",
            unit_name=unit_name,
            supervisor=self._worker_supervisor,
        )
        if start_policy is not None:
            raise TatqaP18FormalRuntimeError("Qwen received a Hippo start policy")
        if output_path.is_symlink() or not output_path.is_file():
            raise TatqaP18FormalRuntimeError("typed-plan output is absent")
        raw_output = output_path.read_bytes()
        output = typed_plan_contract.parse_output(raw_output)
        if len(output["items"]) != len(projected):
            raise TatqaP18FormalRuntimeError("typed-plan output count drifted")
        terminal = _terminal_status(stdout, role="Qwen", item_count=len(projected))
        self.receipts[block] = {
            "schema": f"{VERSION}_typed_plan_transport_receipt_v1",
            "block": block,
            "item_count": len(projected),
            "input_sha256": input_sha,
            "output_sha256": hashlib.sha256(raw_output).hexdigest(),
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            "batch_size": QWEN_BATCH_SIZE,
            "physical_GPU": QWEN_PHYSICAL_GPU,
            "worker_pid": terminal["worker_pid"],
            "model_context_tokens": terminal["model_context_tokens"],
            "model_execution_started_monotonic_ns": terminal[
                "model_execution_started_monotonic_ns"
            ],
            "model_execution_finished_monotonic_ns": terminal[
                "model_execution_finished_monotonic_ns"
            ],
            "systemd_unit_name_sha256": _unit_name_sha256(unit_name),
            "systemd_unit_closure": _validate_unit_closure_receipt(
                closure,
                expected_unit_name_sha256=_unit_name_sha256(unit_name),
            ),
            "filesystem_isolation": FILESYSTEM_ISOLATION,
            "network_properties": list(SYSTEMD_NETWORK_PROPERTIES),
        }
        return raw_output

    def transport_receipt(self, block: str) -> Mapping[str, Any]:
        block = _safe_leaf(block, "block")
        try:
            return dict(self.receipts[block])
        except KeyError as exc:
            raise TatqaP18FormalRuntimeError(
                "typed-plan transport receipt is unavailable"
            ) from exc


class SystemdHippoByteRunner(_SupervisedWorkerCapability):
    """Canonical-byte item-local HippoRAG boundary for a thread scheduler."""

    def __init__(
        self,
        paths: RuntimePaths,
        *,
        popen_factory=subprocess.Popen,
        systemctl_runner=subprocess.run,
    ) -> None:
        self.paths = paths.checked()
        self._initialize_worker_supervisor(
            popen_factory=popen_factory, systemctl_runner=systemctl_runner
        )
        self.receipts: list[dict[str, Any]] = []

    def __call__(
        self,
        block: str,
        item_commitment_sha256: str,
        canonical_input: bytes,
    ) -> bytes:
        block = _safe_leaf(block, "block")
        if (
            not isinstance(item_commitment_sha256, str)
            or _HEX64.fullmatch(item_commitment_sha256) is None
        ):
            raise TatqaP18FormalRuntimeError("item commitment drifted")
        if not isinstance(canonical_input, bytes):
            raise TatqaP18FormalRuntimeError("HippoRAG input is not bytes")
        query, units = hipporag_contract.parse_input(canonical_input)
        input_semantic_sha = hipporag_contract.input_binding_sha256(query, units)
        if any(
            row.get("block") == block
            and row.get("item_commitment_sha256") == item_commitment_sha256
            for row in self.receipts
        ):
            raise TatqaP18FormalRuntimeError("HippoRAG item replay is forbidden")
        root = _fresh_stage_root(
            self.paths.work_root, f"hippo_{block}_{item_commitment_sha256}"
        )
        input_path = root / "hippo.input.json"
        output_path = root / "hippo.output.json"
        index_root = root / "index"
        input_sha = _write_exclusive(input_path, canonical_input)
        command = [
            str(self.paths.runtime_python),
            "-B",
            "-m",
            "replication_runtime.tatqa_p18_v1.hipporag_worker",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--index-root",
            str(index_root),
            "--llm-model",
            str(self.paths.hippo_llm_model),
            "--embedding-model",
            str(self.paths.hippo_embedding_model),
        ]
        unit_name = _worker_unit_name(
            role="HippoRAG",
            identity={
                "block": block,
                "input_sha256": input_sha,
                "item_commitment_sha256": item_commitment_sha256,
                "stage_root_sha256": hashlib.sha256(
                    str(root.absolute()).encode("utf-8")
                ).hexdigest(),
            },
        )
        stdout, stderr, start_policy, closure = _run_worker(
            command=command,
            environment=_worker_environment(self.paths, root, role="HippoRAG"),
            inaccessible_paths=_worker_inaccessible_paths(self.paths),
            timeout=HIPPORAG_TIMEOUT_SECONDS,
            role="HippoRAG",
            unit_name=unit_name,
            supervisor=self._worker_supervisor,
        )
        checked_start_policy = _validate_start_policy_receipt(
            start_policy,
            expected_unit_name_sha256=_unit_name_sha256(unit_name),
        )
        checked_closure = _validate_unit_closure_receipt(
            closure,
            expected_unit_name_sha256=_unit_name_sha256(unit_name),
        )
        if (
            checked_start_policy["control_group_sha256"]
            != checked_closure["control_group_sha256"]
        ):
            raise TatqaP18FormalRuntimeError(
                "HippoRAG active and closed control-group bindings diverged"
            )
        if output_path.is_symlink() or not output_path.is_file():
            raise TatqaP18FormalRuntimeError("HippoRAG output is absent")
        raw_output = output_path.read_bytes()
        output = hipporag_contract.parse_output(raw_output)
        if (
            output["input_sha256"] != input_semantic_sha
            or output["unit_count"] != len(units)
            or not set(output["top_unit_ids"]).issubset(
                {row.unit_id for row in units}
            )
        ):
            raise TatqaP18FormalRuntimeError("HippoRAG output binding drifted")
        terminal = _terminal_status(stdout, role="HippoRAG", item_count=len(units))
        self.receipts.append(
            {
                "schema": f"{VERSION}_hippo_transport_receipt_v1",
                "block": block,
                "item_commitment_sha256": item_commitment_sha256,
                "input_file_sha256": input_sha,
                "input_semantic_sha256": input_semantic_sha,
                "output_file_sha256": hashlib.sha256(raw_output).hexdigest(),
                "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
                "CPU_threads": HIPPORAG_CPU_THREADS,
                "configured_torch_intraop_threads": terminal[
                    "configured_torch_intraop_threads"
                ],
                "configured_torch_interop_threads": terminal[
                    "configured_torch_interop_threads"
                ],
                "observed_process_thread_peak": terminal[
                    "observed_process_thread_peak"
                ],
                "model_execution_started_monotonic_ns": terminal[
                    "model_execution_started_monotonic_ns"
                ],
                "model_execution_finished_monotonic_ns": terminal[
                    "model_execution_finished_monotonic_ns"
                ],
                "systemd_tasks_max": HIPPORAG_SYSTEMD_TASKS_MAX,
                "thread_monitor_process_reservation": (
                    HIPPORAG_THREAD_MONITOR_PROCESS_RESERVATION
                ),
                "maximum_worker_process_threads": (
                    HIPPORAG_MAXIMUM_WORKER_PROCESS_THREADS
                ),
                "systemd_start_policy": checked_start_policy,
                "systemd_start_policy_sha256": stable_hash(checked_start_policy),
                "systemd_unit_name_sha256": _unit_name_sha256(unit_name),
                "systemd_unit_closure": checked_closure,
                "worker_pid": terminal["worker_pid"],
                "filesystem_isolation": FILESYSTEM_ISOLATION,
                "visible_GPU": "",
                "network_properties": list(SYSTEMD_NETWORK_PROPERTIES),
            }
        )
        return raw_output

    def transport_receipt(
        self, block: str, item_commitment_sha256: str
    ) -> Mapping[str, Any]:
        block = _safe_leaf(block, "block")
        if (
            not isinstance(item_commitment_sha256, str)
            or _HEX64.fullmatch(item_commitment_sha256) is None
        ):
            raise TatqaP18FormalRuntimeError("item commitment drifted")
        matches = [
            row
            for row in self.receipts
            if row.get("block") == block
            and row.get("item_commitment_sha256") == item_commitment_sha256
        ]
        if len(matches) != 1:
            raise TatqaP18FormalRuntimeError(
                "HippoRAG transport receipt is unavailable or duplicated"
            )
        return dict(matches[0])


class SystemdTypedPlanBatcher(_SupervisedWorkerCapability):
    """One strict, label-free Qwen batch for each lifecycle block."""

    def __init__(
        self,
        paths: RuntimePaths,
        *,
        popen_factory=subprocess.Popen,
        systemctl_runner=subprocess.run,
    ) -> None:
        self.paths = paths.checked()
        self._initialize_worker_supervisor(
            popen_factory=popen_factory, systemctl_runner=systemctl_runner
        )
        self.receipts: dict[str, dict[str, Any]] = {}

    def __call__(
        self, block: str, items: Sequence[features.LabelFreeRuntimeItem]
    ) -> tuple[core.TypedPlan, ...]:
        if block in self.receipts:
            raise TatqaP18FormalRuntimeError("typed-plan block replay is forbidden")
        rows = tuple(items)
        if not rows or len(rows) > typed_plan_contract.MAXIMUM_ITEM_COUNT:
            raise TatqaP18FormalRuntimeError("typed-plan block size drifted")
        projected = tuple(
            typed_plan_contract.project_item(item, ordinal)
            for ordinal, item in enumerate(rows)
        )
        payload = typed_plan_contract.input_payload(projected)
        raw_input = typed_plan_contract.canonical_json_bytes(payload)
        root = _fresh_stage_root(self.paths.work_root, f"qwen_{_safe_leaf(block, 'block')}")
        input_path = root / "typed_plan.input.json"
        output_path = root / "typed_plan.output.json"
        input_sha = _write_exclusive(input_path, raw_input)
        command = [
            str(self.paths.runtime_python),
            "-B",
            "-m",
            "replication_runtime.tatqa_p18_v1.typed_plan_worker",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--model",
            str(self.paths.qwen_model),
            "--batch-size",
            str(QWEN_BATCH_SIZE),
        ]
        unit_name = _worker_unit_name(
            role="Qwen",
            identity={
                "block": block,
                "input_sha256": input_sha,
                "stage_root_sha256": hashlib.sha256(
                    str(root.absolute()).encode("utf-8")
                ).hexdigest(),
            },
        )
        stdout, stderr, start_policy, closure = _run_worker(
            command=command,
            environment=_worker_environment(self.paths, root, role="Qwen"),
            inaccessible_paths=_worker_inaccessible_paths(self.paths),
            timeout=QWEN_TIMEOUT_SECONDS,
            role="Qwen",
            unit_name=unit_name,
            supervisor=self._worker_supervisor,
        )
        if start_policy is not None:
            raise TatqaP18FormalRuntimeError("Qwen received a Hippo start policy")
        if output_path.is_symlink() or not output_path.is_file():
            raise TatqaP18FormalRuntimeError("typed-plan output is absent")
        raw_output = output_path.read_bytes()
        output = typed_plan_contract.parse_output(raw_output)
        if len(output["items"]) != len(rows):
            raise TatqaP18FormalRuntimeError("typed-plan output count drifted")
        plans = tuple(
            core.validate_typed_plan(row["plan"]) for row in output["items"]
        )
        terminal = _terminal_status(stdout, role="Qwen", item_count=len(rows))
        self.receipts[block] = {
            "schema": f"{VERSION}_typed_plan_batch_receipt_v1",
            "block": block,
            "item_count": len(rows),
            "input_sha256": input_sha,
            "output_sha256": hashlib.sha256(raw_output).hexdigest(),
            "output_semantic_sha256": stable_hash(output),
            "generation_valid_count": sum(
                bool(row["generation_valid"]) for row in output["items"]
            ),
            "prompt_projection_sha256s": [
                row["prompt_projection_sha256"] for row in output["items"]
            ],
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            "batch_size": QWEN_BATCH_SIZE,
            "physical_GPU": QWEN_PHYSICAL_GPU,
            "worker_pid": terminal["worker_pid"],
            "model_context_tokens": terminal["model_context_tokens"],
            "model_execution_started_monotonic_ns": terminal[
                "model_execution_started_monotonic_ns"
            ],
            "model_execution_finished_monotonic_ns": terminal[
                "model_execution_finished_monotonic_ns"
            ],
            "systemd_unit_name_sha256": _unit_name_sha256(unit_name),
            "systemd_unit_closure": _validate_unit_closure_receipt(
                closure,
                expected_unit_name_sha256=_unit_name_sha256(unit_name),
            ),
            "filesystem_isolation": FILESYSTEM_ISOLATION,
            "network_properties": list(SYSTEMD_NETWORK_PROPERTIES),
        }
        return plans


class SystemdHippoRunner(_SupervisedWorkerCapability):
    """Fresh official item-local HippoRAG subprocess per invocation."""

    def __init__(
        self,
        paths: RuntimePaths,
        *,
        popen_factory=subprocess.Popen,
        systemctl_runner=subprocess.run,
    ) -> None:
        self.paths = paths.checked()
        self._initialize_worker_supervisor(
            popen_factory=popen_factory, systemctl_runner=systemctl_runner
        )
        self.receipts: list[dict[str, Any]] = []

    def __call__(
        self, block: str, item: features.LabelFreeRuntimeItem
    ) -> tuple[str, ...]:
        if not isinstance(item, features.LabelFreeRuntimeItem):
            raise TatqaP18FormalRuntimeError("HippoRAG item type drifted")
        item_commitment = item.item_id
        if any(
            row.get("block") == block
            and row.get("item_commitment_sha256") == item_commitment
            for row in self.receipts
        ):
            raise TatqaP18FormalRuntimeError("HippoRAG item replay is forbidden")
        units = [
            {"ordinal": ordinal, "text": row.text, "unit_id": row.unit_id}
            for ordinal, row in enumerate(item.units)
        ]
        payload = hipporag_contract.input_payload(query=item.question, units=units)
        raw_input = hipporag_contract.canonical_json_bytes(payload)
        root = _fresh_stage_root(
            self.paths.work_root,
            f"hippo_{_safe_leaf(block, 'block')}_{item_commitment}",
        )
        input_path = root / "hippo.input.json"
        output_path = root / "hippo.output.json"
        index_root = root / "index"
        input_sha = _write_exclusive(input_path, raw_input)
        command = [
            str(self.paths.runtime_python),
            "-B",
            "-m",
            "replication_runtime.tatqa_p18_v1.hipporag_worker",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--index-root",
            str(index_root),
            "--llm-model",
            str(self.paths.hippo_llm_model),
            "--embedding-model",
            str(self.paths.hippo_embedding_model),
        ]
        unit_name = _worker_unit_name(
            role="HippoRAG",
            identity={
                "block": block,
                "input_sha256": input_sha,
                "item_commitment_sha256": item_commitment,
                "stage_root_sha256": hashlib.sha256(
                    str(root.absolute()).encode("utf-8")
                ).hexdigest(),
            },
        )
        stdout, stderr, start_policy, closure = _run_worker(
            command=command,
            environment=_worker_environment(self.paths, root, role="HippoRAG"),
            inaccessible_paths=_worker_inaccessible_paths(self.paths),
            timeout=HIPPORAG_TIMEOUT_SECONDS,
            role="HippoRAG",
            unit_name=unit_name,
            supervisor=self._worker_supervisor,
        )
        checked_start_policy = _validate_start_policy_receipt(
            start_policy,
            expected_unit_name_sha256=_unit_name_sha256(unit_name),
        )
        checked_closure = _validate_unit_closure_receipt(
            closure,
            expected_unit_name_sha256=_unit_name_sha256(unit_name),
        )
        if (
            checked_start_policy["control_group_sha256"]
            != checked_closure["control_group_sha256"]
        ):
            raise TatqaP18FormalRuntimeError(
                "HippoRAG active and closed control-group bindings diverged"
            )
        if output_path.is_symlink() or not output_path.is_file():
            raise TatqaP18FormalRuntimeError("HippoRAG output is absent")
        raw_output = output_path.read_bytes()
        output = hipporag_contract.parse_output(raw_output)
        if (
            output["input_sha256"] != payload["input_sha256"]
            or output["unit_count"] != len(item.units)
            or not set(output["top_unit_ids"]).issubset(
                {row.unit_id for row in item.units}
            )
        ):
            raise TatqaP18FormalRuntimeError("HippoRAG output binding drifted")
        terminal = _terminal_status(
            stdout, role="HippoRAG", item_count=len(item.units)
        )
        self.receipts.append(
            {
                "schema": f"{VERSION}_hippo_item_receipt_v1",
                "block": block,
                "item_commitment_sha256": item_commitment,
                "input_file_sha256": input_sha,
                "input_semantic_sha256": payload["input_sha256"],
                "output_file_sha256": hashlib.sha256(raw_output).hexdigest(),
                "top5_behavior_sha256": core.canonical_behavior_hash(
                    output["top_unit_ids"]
                ),
                "unit_count": len(item.units),
                "graph_node_count": output["graph_node_count"],
                "graph_edge_count": output["graph_edge_count"],
                "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
                "CPU_threads": HIPPORAG_CPU_THREADS,
                "configured_torch_intraop_threads": terminal[
                    "configured_torch_intraop_threads"
                ],
                "configured_torch_interop_threads": terminal[
                    "configured_torch_interop_threads"
                ],
                "observed_process_thread_peak": terminal[
                    "observed_process_thread_peak"
                ],
                "model_execution_started_monotonic_ns": terminal[
                    "model_execution_started_monotonic_ns"
                ],
                "model_execution_finished_monotonic_ns": terminal[
                    "model_execution_finished_monotonic_ns"
                ],
                "systemd_tasks_max": HIPPORAG_SYSTEMD_TASKS_MAX,
                "thread_monitor_process_reservation": (
                    HIPPORAG_THREAD_MONITOR_PROCESS_RESERVATION
                ),
                "maximum_worker_process_threads": (
                    HIPPORAG_MAXIMUM_WORKER_PROCESS_THREADS
                ),
                "systemd_start_policy": checked_start_policy,
                "systemd_start_policy_sha256": stable_hash(checked_start_policy),
                "systemd_unit_name_sha256": _unit_name_sha256(unit_name),
                "systemd_unit_closure": checked_closure,
                "worker_pid": terminal["worker_pid"],
                "filesystem_isolation": FILESYSTEM_ISOLATION,
                "visible_GPU": "",
                "network_properties": list(SYSTEMD_NETWORK_PROPERTIES),
            }
        )
        return tuple(output["top_unit_ids"])


__all__ = [
    "BoundMiniLMEncoder",
    "FILESYSTEM_ISOLATION",
    "HIPPORAG_CONFIGURED_TORCH_INTEROP_THREADS",
    "HIPPORAG_CONFIGURED_TORCH_INTRAOP_THREADS",
    "HIPPORAG_CPU_THREADS",
    "HIPPORAG_MAXIMUM_WORKER_PROCESS_THREADS",
    "HIPPORAG_SYSTEMD_TASKS_MAX",
    "HIPPORAG_THREAD_MONITOR_PROCESS_RESERVATION",
    "QWEN_BATCH_SIZE",
    "RuntimePaths",
    "SYSTEMD_NETWORK_PROPERTIES",
    "SYSTEMD_START_POLICY_SCHEMA",
    "SYSTEMD_UNIT_CLOSURE_SCHEMA",
    "SystemdHippoRunner",
    "SystemdHippoByteRunner",
    "SystemdTypedPlanBatchRunner",
    "SystemdTypedPlanBatcher",
    "TatqaP18FormalRuntimeError",
    "canonical_json_bytes",
    "file_sha256",
    "runtime_inventory_snapshot",
    "stable_hash",
    "systemd_network_preflight",
    "tree_receipt",
    "verify_runtime_fingerprint",
]
