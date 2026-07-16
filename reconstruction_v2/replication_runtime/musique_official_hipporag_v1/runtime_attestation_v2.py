"""Prospective filesystem attestation for the frozen official HippoRAG runtime.

Version 1 intentionally remains byte-for-byte frozen because its source hashes
are part of a closed formal cohort.  This version is a prospective trust root:
before a *fresh* freeze it performs exactly one synthetic executable
qualification and exactly one executable identity probe.  Formal entry then
recomputes only deterministic filesystem bindings; it never launches an
identity-probe subprocess and it has no retry path.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
from typing import Any, Mapping

from assumption_agent.models import stable_hash

from .binding import (
    DEPENDENCY_NAMES,
    SYNTHETIC_QUALIFICATION_KEYS,
    _asset_hash,
    _runtime_binding_from_probe,
    _runtime_probe,
    current_implementation_binding as current_v1_implementation_binding,
    validate_binding_receipt,
)
from .adapter import _launch_worker, _write_private_input
from .contract import MuSiQueOfficialHippoRAGError, parse_idx_only_output


ATTESTATION_SCHEMA = "musique_official_hipporag_filesystem_attestation_v2"
ATTESTATION_DECISION = (
    "one_shot_pre_freeze_executable_qualification_then_formal_filesystem_attestation"
)
IMPLEMENTATION_SCHEMA = "musique_official_hipporag_attestation_implementation_v2"
IMPLEMENTATION_RELATIVE_FILES = (
    "assumption_agent/benchmarks/musique_formal_runtime_binding_v2.py",
    "replication_runtime/musique_official_hipporag_v1/runtime_attestation_v2.py",
    "replication_runtime/musique_official_hipporag_v1/adapter_v2.py",
)
FORMAL_ENTRY_POLICY: dict[str, Any] = {
    "attestation_mode": "deterministic_filesystem_only",
    "executable_identity_probe_calls": 0,
    "runtime_python_path_policy": "lexical_venv_bin_python_not_resolved_before_attestation",
    "subprocess_calls": 0,
    "retry_count": 0,
    "requalification_after_freeze": False,
}
_TOP_LEVEL_KEYS = frozenset(
    {
        "base_binding",
        "decision",
        "formal_entry_policy",
        "implementation_binding",
        "pre_freeze_executable_qualification",
        "receipt_sha256",
        "runtime_filesystem_binding",
        "schema",
    }
)
_BASE_BINDING_KEYS = frozenset(
    {
        "file_sha256",
        "qualification_sha256",
        "receipt_sha256",
        "schema",
    }
)
_QUALIFICATION_KEYS = frozenset(
    {
        "benchmark_rows_read",
        "qualified_before_freeze",
        "retry_count",
        "runtime_identity_probe_calls",
        "synthetic_executable_qualification_calls",
        "synthetic_receipt",
        "synthetic_receipt_sha256",
    }
)
_IMPLEMENTATION_KEYS = frozenset({"schema", "files", "set_sha256"})
_IMPLEMENTATION_FILE_KEYS = frozenset({"path", "sha256"})
_SNAPSHOT_KEYS = frozenset(
    {
        "dependency_metadata_rows",
        "dependency_metadata_set_sha256",
        "local_embedding_asset_sha256",
        "local_embedding_topology_sha256",
        "local_llm_asset_sha256",
        "local_llm_topology_sha256",
        "official_source_file_count",
        "official_source_root_role",
        "official_source_tree_sha256",
        "pyvenv_cfg_sha256",
        "runtime_python_is_symlink",
        "runtime_python_relative_path",
        "runtime_python_symlink_target_sha256",
        "runtime_python_target_sha256",
        "search_root_roles",
    }
)
_DEPENDENCY_ROW_KEYS = frozenset(
    {
        "dist_info_file_count",
        "dist_info_name",
        "dist_info_tree_sha256",
        "name",
        "root_role",
        "version",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CANONICAL_NAME_RE = re.compile(r"[-_.]+")
_ATTESTATION_CACHE_LOCK = threading.Lock()
_ATTESTATION_CACHE: dict[tuple[str, ...], dict[str, Any]] = {}
_FIXED_SYNTHETIC_QUESTION = "Which synthetic paragraph completes the local chain?"
_FIXED_SYNTHETIC_PARAGRAPHS: tuple[dict[str, object], ...] = (
    {
        "idx": 0,
        "title": "Synthetic Alpha",
        "paragraph_text": "Synthetic Alpha links to Synthetic Beta.",
    },
    {
        "idx": 1,
        "title": "Synthetic Noise One",
        "paragraph_text": "A locally generated copper circle is unrelated.",
    },
    {
        "idx": 2,
        "title": "Synthetic Beta",
        "paragraph_text": "Synthetic Beta completes the entirely local chain.",
    },
    {
        "idx": 3,
        "title": "Synthetic Noise Two",
        "paragraph_text": "A locally generated silver square is unrelated.",
    },
    {
        "idx": 4,
        "title": "Synthetic Noise Three",
        "paragraph_text": "A locally generated green triangle is unrelated.",
    },
)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MuSiQueOfficialHippoRAGError(f"{field} must be lowercase sha256")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], field: str
) -> None:
    if set(value) != expected:
        raise MuSiQueOfficialHippoRAGError(f"{field} key set mismatch")


def _load_json_object(path: Path, field: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise MuSiQueOfficialHippoRAGError(f"{field} is unavailable")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MuSiQueOfficialHippoRAGError(f"{field} is invalid") from exc
    if not isinstance(payload, dict):
        raise MuSiQueOfficialHippoRAGError(f"{field} must be an object")
    return payload, raw


def _canonical_distribution_name(value: str) -> str:
    return _CANONICAL_NAME_RE.sub("-", value).casefold()


def _metadata_identity(path: Path) -> tuple[str, str]:
    if path.is_symlink() or not path.is_file():
        raise MuSiQueOfficialHippoRAGError("dependency METADATA is unavailable")
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
        raise MuSiQueOfficialHippoRAGError("dependency METADATA is invalid") from exc
    if not name or not version:
        raise MuSiQueOfficialHippoRAGError("dependency METADATA lacks name or version")
    return _canonical_distribution_name(name), version


def _tree_binding(path: Path, *, python_only: bool = False) -> tuple[int, str]:
    if path.is_symlink() or not path.is_dir():
        raise MuSiQueOfficialHippoRAGError("attested source tree is unavailable")
    if any(entry.is_symlink() for entry in path.rglob("*")):
        raise MuSiQueOfficialHippoRAGError("attested source tree contains a symlink")
    entries = sorted(
        (
            entry
            for entry in path.rglob("*")
            if entry.is_file() and (not python_only or entry.suffix == ".py")
        ),
        key=lambda entry: entry.relative_to(path).as_posix(),
    )
    if not entries:
        raise MuSiQueOfficialHippoRAGError("attested source tree is empty")
    rows = []
    for entry in entries:
        rows.append(
            {
                "path": entry.relative_to(path).as_posix(),
                "sha256": _sha256_file(entry),
            }
        )
    return len(rows), stable_hash(rows)


def _asset_topology_hash(path: Path) -> str:
    root = path.resolve(strict=True)
    if not root.is_dir():
        raise MuSiQueOfficialHippoRAGError("local model asset is not a directory")
    entries = sorted(
        (
            entry
            for entry in root.rglob("*")
            if entry.is_file() or entry.is_symlink()
        ),
        key=lambda entry: entry.relative_to(root).as_posix(),
    )
    if not entries:
        raise MuSiQueOfficialHippoRAGError("local model asset is empty")
    rows = []
    for entry in entries:
        link_target_hash = None
        if entry.is_symlink():
            link_target_hash = _sha256_bytes(os.readlink(entry).encode("utf-8"))
            if not entry.is_file():
                raise MuSiQueOfficialHippoRAGError(
                    "local model asset contains a dangling or directory symlink"
                )
        rows.append(
            {
                "content_sha256": _sha256_file(entry),
                "is_symlink": entry.is_symlink(),
                "link_target_sha256": link_target_hash,
                "path": entry.relative_to(root).as_posix(),
            }
        )
    return stable_hash(rows)


def _parse_pyvenv(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise MuSiQueOfficialHippoRAGError("runtime is not a lexical overlay venv")
    values: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if " = " in line:
                key, value = line.split(" = ", 1)
                values[key.strip()] = value.strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise MuSiQueOfficialHippoRAGError("pyvenv.cfg is invalid") from exc
    if not values.get("home") or not re.fullmatch(
        r"\d+\.\d+(?:\.\d+)?", values.get("version", "")
    ):
        raise MuSiQueOfficialHippoRAGError("pyvenv.cfg lacks runtime identity")
    if values.get("include-system-site-packages") not in {"true", "false"}:
        raise MuSiQueOfficialHippoRAGError("pyvenv.cfg system-site-packages flag is invalid")
    return values


def _runtime_search_roots(runtime_python: Path) -> tuple[Path, list[tuple[str, Path]]]:
    runtime_python = runtime_python.absolute()
    if runtime_python.name != "python" or runtime_python.parent.name != "bin":
        raise MuSiQueOfficialHippoRAGError(
            "runtime Python must retain the lexical venv/bin/python path"
        )
    venv_root = runtime_python.parent.parent
    if venv_root.is_symlink() or not venv_root.is_dir():
        raise MuSiQueOfficialHippoRAGError("lexical venv root is unavailable")
    pyvenv = venv_root / "pyvenv.cfg"
    values = _parse_pyvenv(pyvenv)
    major_minor = ".".join(values["version"].split(".")[:2])
    overlay = venv_root / "lib" / f"python{major_minor}" / "site-packages"
    if overlay.is_symlink() or not overlay.is_dir():
        raise MuSiQueOfficialHippoRAGError("overlay site-packages is unavailable")
    roots: list[tuple[str, Path]] = [("overlay", overlay)]
    if values["include-system-site-packages"] == "true":
        home = Path(values["home"])
        if not home.is_absolute():
            raise MuSiQueOfficialHippoRAGError("pyvenv.cfg home must be absolute")
        base = home.parent / "lib" / f"python{major_minor}" / "site-packages"
        if base.is_symlink() or not base.is_dir():
            raise MuSiQueOfficialHippoRAGError("base site-packages is unavailable")
        if base.resolve(strict=True) != overlay.resolve(strict=True):
            roots.append(("base", base))
    return pyvenv, roots


def _dependency_rows(
    roots: list[tuple[str, Path]], expected_versions: Mapping[str, object]
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    rows: list[dict[str, Any]] = []
    selected_roots: dict[str, Path] = {}
    for name in DEPENDENCY_NAMES:
        expected = expected_versions.get(name)
        matches: list[tuple[str, Path, str]] = []
        all_name_matches: list[tuple[str, Path, str]] = []
        for role, root in roots:
            role_matches: list[tuple[str, Path, str]] = []
            for dist_info in sorted(root.glob("*.dist-info"), key=lambda value: value.name):
                metadata = dist_info / "METADATA"
                if not metadata.is_file():
                    continue
                observed_name, observed_version = _metadata_identity(metadata)
                if observed_name != _canonical_distribution_name(name):
                    continue
                all_name_matches.append((role, dist_info, observed_version))
                if observed_version == expected:
                    role_matches.append((role, dist_info, observed_version))
            if expected is not None and role_matches:
                role_name_matches = [
                    row for row in all_name_matches if row[0] == role
                ]
                if len(role_name_matches) != 1 or len(role_matches) != 1:
                    raise MuSiQueOfficialHippoRAGError(
                        f"qualified dependency metadata is ambiguous: {name}"
                    )
                matches.extend(role_matches)
                break
        if expected is None:
            if all_name_matches:
                raise MuSiQueOfficialHippoRAGError(
                    f"qualified-absent dependency is now installed: {name}"
                )
            rows.append(
                {
                    "dist_info_file_count": 0,
                    "dist_info_name": None,
                    "dist_info_tree_sha256": None,
                    "name": name,
                    "root_role": None,
                    "version": None,
                }
            )
            continue
        if len(matches) != 1:
            raise MuSiQueOfficialHippoRAGError(
                f"qualified dependency metadata is not unique: {name}"
            )
        role, dist_info, observed_version = matches[0]
        count, tree_hash = _tree_binding(dist_info)
        rows.append(
            {
                "dist_info_file_count": count,
                "dist_info_name": dist_info.name,
                "dist_info_tree_sha256": tree_hash,
                "name": name,
                "root_role": role,
                "version": observed_version,
            }
        )
        selected_roots[name] = dist_info.parent
    return rows, selected_roots


def _filesystem_snapshot(
    *,
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
    expected_versions: Mapping[str, object],
) -> dict[str, Any]:
    runtime_python = runtime_python.absolute()
    if not runtime_python.is_file() or not os.access(runtime_python, os.X_OK):
        raise MuSiQueOfficialHippoRAGError("runtime Python is unavailable")
    pyvenv, roots = _runtime_search_roots(runtime_python)
    dependency_rows, selected_roots = _dependency_rows(roots, expected_versions)
    hipporag_root = selected_roots.get("hipporag")
    if hipporag_root is None:
        raise MuSiQueOfficialHippoRAGError("official HippoRAG source root is unavailable")
    source_count, source_hash = _tree_binding(hipporag_root / "hipporag", python_only=True)
    target = runtime_python.resolve(strict=True)
    if not target.is_file():
        raise MuSiQueOfficialHippoRAGError("runtime Python target is unavailable")
    symlink_target_hash = (
        _sha256_bytes(os.readlink(runtime_python).encode("utf-8"))
        if runtime_python.is_symlink()
        else None
    )
    return {
        "dependency_metadata_rows": dependency_rows,
        "dependency_metadata_set_sha256": stable_hash(dependency_rows),
        "local_embedding_asset_sha256": _asset_hash(
            local_embedding_model, row_kind="tuple"
        ),
        "local_embedding_topology_sha256": _asset_topology_hash(local_embedding_model),
        "local_llm_asset_sha256": _asset_hash(local_llm_model, row_kind="dict"),
        "local_llm_topology_sha256": _asset_topology_hash(local_llm_model),
        "official_source_file_count": source_count,
        "official_source_root_role": next(
            role for role, root in roots if root == hipporag_root
        ),
        "official_source_tree_sha256": source_hash,
        "pyvenv_cfg_sha256": _sha256_file(pyvenv),
        "runtime_python_is_symlink": runtime_python.is_symlink(),
        "runtime_python_relative_path": "bin/python",
        "runtime_python_symlink_target_sha256": symlink_target_hash,
        "runtime_python_target_sha256": _sha256_file(target),
        "search_root_roles": [role for role, _root in roots],
    }


def current_v2_implementation_binding(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    rows = []
    for relative in IMPLEMENTATION_RELATIVE_FILES:
        path = project_root / relative
        if path.is_symlink() or not path.is_file():
            raise MuSiQueOfficialHippoRAGError(f"v2 implementation file missing: {relative}")
        rows.append({"path": relative, "sha256": _sha256_file(path)})
    return {
        "schema": IMPLEMENTATION_SCHEMA,
        "files": rows,
        "set_sha256": stable_hash(rows),
    }


def _base_binding(
    path: Path, *, project_root: Path
) -> tuple[dict[str, Any], str]:
    payload, raw = _load_json_object(path, "v1 base binding receipt")
    validated = validate_binding_receipt(
        payload, project_root=project_root, verify_implementation=True
    )
    if validated.get("implementation_binding") != current_v1_implementation_binding(
        project_root
    ):
        raise MuSiQueOfficialHippoRAGError("v1 base implementation drifted")
    return validated, _sha256_bytes(raw)


def _validate_synthetic_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(value, SYNTHETIC_QUALIFICATION_KEYS, "synthetic qualification")
    if (
        value.get("status") != "passed_non_scoring_synthetic_local_retrieve_only"
        or value.get("official_core_index_called") is not True
        or value.get("official_core_retrieve_called") is not True
        or value.get("network_namespace_isolated") is not True
        or value.get("external_network_transport_possible") is not False
        or value.get("benchmark_rows_read") != 0
        or value.get("scores_computed") != 0
        or value.get("candidate_count") != 5
        or value.get("output_idx_count") != 5
    ):
        raise MuSiQueOfficialHippoRAGError("synthetic executable qualification failed")
    _require_sha256(value.get("fixture_sha256"), "synthetic fixture hash")
    _require_sha256(value.get("output_idx_sha256"), "synthetic output hash")
    return dict(value)


def _run_fixed_synthetic_qualification_once(
    *,
    project_root: Path,
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
) -> dict[str, Any]:
    """Launch the fixed network-isolated synthetic worker exactly once."""

    fixture = {
        "paragraphs": [dict(row) for row in _FIXED_SYNTHETIC_PARAGRAPHS],
        "question": _FIXED_SYNTHETIC_QUESTION,
    }
    work_root = Path(tempfile.mkdtemp(prefix="musique-hipporag-v2-qualification-"))
    try:
        for name in ("home", "cache", "tmp"):
            (work_root / name).mkdir(mode=0o700)
        input_path = work_root / "fixed_synthetic.input.json"
        output_path = work_root / "fixed_synthetic.idx.json"
        index_root = work_root / "fixed_synthetic.index"
        _write_private_input(
            input_path,
            question=_FIXED_SYNTHETIC_QUESTION,
            paragraphs=_FIXED_SYNTHETIC_PARAGRAPHS,
        )
        # This is intentionally a single call with no retry wrapper.
        _launch_worker(
            project_root=project_root,
            runtime_python=runtime_python.absolute(),
            local_llm_model=local_llm_model.resolve(strict=True),
            local_embedding_model=local_embedding_model.resolve(strict=True),
            input_path=input_path,
            output_path=output_path,
            index_root=index_root,
            writable_root=work_root,
            timeout_seconds=900,
        )
        if output_path.is_symlink() or not output_path.is_file():
            raise MuSiQueOfficialHippoRAGError(
                "fixed synthetic qualification emitted no idx-only output"
            )
        indices = parse_idx_only_output(
            output_path.read_bytes(), candidate_count=len(_FIXED_SYNTHETIC_PARAGRAPHS)
        )
        return {
            "benchmark_rows_read": 0,
            "candidate_count": len(_FIXED_SYNTHETIC_PARAGRAPHS),
            "external_network_transport_possible": False,
            "fixture_sha256": stable_hash(fixture),
            "network_namespace_isolated": True,
            "official_core_index_called": True,
            "official_core_retrieve_called": True,
            "output_idx_count": len(indices),
            "output_idx_sha256": stable_hash(list(indices)),
            "scores_computed": 0,
            "status": "passed_non_scoring_synthetic_local_retrieve_only",
        }
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


def _validate_snapshot_shape(value: Mapping[str, Any]) -> None:
    _require_exact_keys(value, _SNAPSHOT_KEYS, "filesystem snapshot")
    for field in (
        "dependency_metadata_set_sha256",
        "local_embedding_asset_sha256",
        "local_embedding_topology_sha256",
        "local_llm_asset_sha256",
        "local_llm_topology_sha256",
        "official_source_tree_sha256",
        "pyvenv_cfg_sha256",
        "runtime_python_target_sha256",
    ):
        _require_sha256(value.get(field), field)
    link_hash = value.get("runtime_python_symlink_target_sha256")
    if link_hash is not None:
        _require_sha256(link_hash, "runtime Python symlink target hash")
    if (
        value.get("runtime_python_relative_path") != "bin/python"
        or not isinstance(value.get("runtime_python_is_symlink"), bool)
        or value.get("official_source_root_role") not in {"overlay", "base"}
        or value.get("search_root_roles") not in (["overlay"], ["overlay", "base"])
        or isinstance(value.get("official_source_file_count"), bool)
        or not isinstance(value.get("official_source_file_count"), int)
        or value.get("official_source_file_count", 0) <= 0
    ):
        raise MuSiQueOfficialHippoRAGError("filesystem snapshot runtime shape drifted")
    rows = value.get("dependency_metadata_rows")
    if not isinstance(rows, list) or len(rows) != len(DEPENDENCY_NAMES):
        raise MuSiQueOfficialHippoRAGError("dependency metadata row set mismatch")
    for expected_name, row in zip(DEPENDENCY_NAMES, rows):
        if not isinstance(row, Mapping):
            raise MuSiQueOfficialHippoRAGError("dependency metadata row is malformed")
        _require_exact_keys(row, _DEPENDENCY_ROW_KEYS, "dependency metadata row")
        if row.get("name") != expected_name:
            raise MuSiQueOfficialHippoRAGError("dependency metadata order drifted")
        if row.get("version") is None:
            if any(
                row.get(field) is not None
                for field in (
                    "dist_info_name",
                    "dist_info_tree_sha256",
                    "root_role",
                )
            ) or row.get("dist_info_file_count") != 0:
                raise MuSiQueOfficialHippoRAGError("absent dependency row drifted")
        else:
            if (
                not isinstance(row.get("version"), str)
                or not row.get("version")
                or row.get("root_role") not in {"overlay", "base"}
                or not isinstance(row.get("dist_info_name"), str)
                or "/" in row.get("dist_info_name", "")
                or isinstance(row.get("dist_info_file_count"), bool)
                or not isinstance(row.get("dist_info_file_count"), int)
                or row.get("dist_info_file_count", 0) <= 0
            ):
                raise MuSiQueOfficialHippoRAGError("dependency metadata row drifted")
            _require_sha256(
                row.get("dist_info_tree_sha256"), "dependency metadata tree hash"
            )
    if stable_hash(rows) != value.get("dependency_metadata_set_sha256"):
        raise MuSiQueOfficialHippoRAGError("dependency metadata set hash mismatch")


def _validate_snapshot_against_base(
    snapshot: Mapping[str, Any], base: Mapping[str, Any]
) -> None:
    runtime = base.get("runtime_binding")
    assets = base.get("asset_binding")
    source = base.get("official_source_binding")
    if not isinstance(runtime, Mapping) or not isinstance(assets, Mapping) or not isinstance(
        source, Mapping
    ):
        raise MuSiQueOfficialHippoRAGError("v1 base binding is incomplete")
    rows = snapshot.get("dependency_metadata_rows")
    if not isinstance(rows, list):
        raise MuSiQueOfficialHippoRAGError("dependency metadata rows are unavailable")
    observed_versions = {row["name"]: row["version"] for row in rows}
    if (
        snapshot.get("runtime_python_target_sha256")
        != runtime.get("runtime_python_target_sha256")
        or snapshot.get("pyvenv_cfg_sha256") != runtime.get("pyvenv_cfg_sha256")
        or observed_versions != runtime.get("dependency_versions")
        or snapshot.get("official_source_file_count")
        != source.get("python_source_file_count")
        or snapshot.get("official_source_tree_sha256")
        != source.get("python_source_tree_sha256")
        or snapshot.get("local_llm_asset_sha256")
        != assets.get("local_llm_asset_sha256")
        or snapshot.get("local_embedding_asset_sha256")
        != assets.get("local_embedding_asset_sha256")
    ):
        raise MuSiQueOfficialHippoRAGError(
            "filesystem runtime differs from pre-freeze qualification"
        )


def qualify_and_build_attestation_v2(
    *,
    project_root: Path,
    base_binding_receipt_path: Path,
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
) -> dict[str, Any]:
    """Run each executable qualification once, then freeze filesystem evidence.

    There is deliberately no retry loop.  Any fixed-worker/probe failure aborts
    the prospective freeze and must not be converted into a second attempt.
    """

    project_root = project_root.resolve(strict=True)
    base, base_file_hash = _base_binding(
        base_binding_receipt_path.absolute(), project_root=project_root
    )
    synthetic = _validate_synthetic_receipt(
        _run_fixed_synthetic_qualification_once(
            project_root=project_root,
            runtime_python=runtime_python,
            local_llm_model=local_llm_model,
            local_embedding_model=local_embedding_model,
        )
    )
    probe = _runtime_probe(runtime_python.absolute())
    qualification_payload, _raw = _load_json_object(
        project_root / base["qualification_binding"]["path"],
        "qualification receipt",
    )
    runtime_from_probe = _runtime_binding_from_probe(probe, qualification_payload)
    if runtime_from_probe != base.get("runtime_binding"):
        raise MuSiQueOfficialHippoRAGError(
            "one-shot executable identity differs from frozen base qualification"
        )
    expected_versions = base["runtime_binding"]["dependency_versions"]
    snapshot = _filesystem_snapshot(
        runtime_python=runtime_python,
        local_llm_model=local_llm_model,
        local_embedding_model=local_embedding_model,
        expected_versions=expected_versions,
    )
    _validate_snapshot_shape(snapshot)
    _validate_snapshot_against_base(snapshot, base)
    base_binding = {
        "file_sha256": base_file_hash,
        "qualification_sha256": base["qualification_binding"]["qualification_sha256"],
        "receipt_sha256": base["receipt_sha256"],
        "schema": base["schema"],
    }
    qualification = {
        "benchmark_rows_read": 0,
        "qualified_before_freeze": True,
        "retry_count": 0,
        "runtime_identity_probe_calls": 1,
        "synthetic_executable_qualification_calls": 1,
        "synthetic_receipt": synthetic,
        "synthetic_receipt_sha256": stable_hash(synthetic),
    }
    receipt: dict[str, Any] = {
        "base_binding": base_binding,
        "decision": ATTESTATION_DECISION,
        "formal_entry_policy": dict(FORMAL_ENTRY_POLICY),
        "implementation_binding": current_v2_implementation_binding(project_root),
        "pre_freeze_executable_qualification": qualification,
        "runtime_filesystem_binding": snapshot,
        "schema": ATTESTATION_SCHEMA,
    }
    receipt["receipt_sha256"] = stable_hash(receipt)
    return receipt


def _validate_receipt_structure(payload: Mapping[str, Any]) -> None:
    _require_exact_keys(payload, _TOP_LEVEL_KEYS, "v2 attestation receipt")
    body = dict(payload)
    declared = _require_sha256(body.pop("receipt_sha256", None), "v2 receipt hash")
    if payload.get("schema") != ATTESTATION_SCHEMA or stable_hash(body) != declared:
        raise MuSiQueOfficialHippoRAGError("v2 attestation receipt self-hash mismatch")
    if payload.get("decision") != ATTESTATION_DECISION:
        raise MuSiQueOfficialHippoRAGError("v2 attestation decision mismatch")
    if payload.get("formal_entry_policy") != FORMAL_ENTRY_POLICY:
        raise MuSiQueOfficialHippoRAGError("v2 formal-entry policy drifted")
    base = payload.get("base_binding")
    qualification = payload.get("pre_freeze_executable_qualification")
    implementation = payload.get("implementation_binding")
    snapshot = payload.get("runtime_filesystem_binding")
    if not all(
        isinstance(value, Mapping)
        for value in (base, qualification, implementation, snapshot)
    ):
        raise MuSiQueOfficialHippoRAGError("v2 attestation sections are incomplete")
    _require_exact_keys(base, _BASE_BINDING_KEYS, "v2 base binding")
    for field in ("file_sha256", "qualification_sha256", "receipt_sha256"):
        _require_sha256(base.get(field), f"base {field}")
    if not isinstance(base.get("schema"), str) or not base.get("schema"):
        raise MuSiQueOfficialHippoRAGError("base binding schema is malformed")
    _require_exact_keys(qualification, _QUALIFICATION_KEYS, "pre-freeze qualification")
    synthetic = qualification.get("synthetic_receipt")
    if not isinstance(synthetic, Mapping):
        raise MuSiQueOfficialHippoRAGError("synthetic qualification receipt is malformed")
    validated_synthetic = _validate_synthetic_receipt(synthetic)
    if (
        qualification.get("benchmark_rows_read") != 0
        or qualification.get("qualified_before_freeze") is not True
        or qualification.get("retry_count") != 0
        or qualification.get("runtime_identity_probe_calls") != 1
        or qualification.get("synthetic_executable_qualification_calls") != 1
        or qualification.get("synthetic_receipt_sha256")
        != stable_hash(validated_synthetic)
    ):
        raise MuSiQueOfficialHippoRAGError("pre-freeze one-shot qualification drifted")
    _require_exact_keys(implementation, _IMPLEMENTATION_KEYS, "v2 implementation")
    if implementation.get("schema") != IMPLEMENTATION_SCHEMA:
        raise MuSiQueOfficialHippoRAGError("v2 implementation schema drifted")
    files = implementation.get("files")
    if not isinstance(files, list) or len(files) != len(IMPLEMENTATION_RELATIVE_FILES):
        raise MuSiQueOfficialHippoRAGError("v2 implementation file set mismatch")
    for expected, row in zip(IMPLEMENTATION_RELATIVE_FILES, files):
        if not isinstance(row, Mapping):
            raise MuSiQueOfficialHippoRAGError("v2 implementation row is malformed")
        _require_exact_keys(row, _IMPLEMENTATION_FILE_KEYS, "v2 implementation row")
        if row.get("path") != expected:
            raise MuSiQueOfficialHippoRAGError("v2 implementation path drifted")
        _require_sha256(row.get("sha256"), "v2 implementation file hash")
    if implementation.get("set_sha256") != stable_hash(files):
        raise MuSiQueOfficialHippoRAGError("v2 implementation set hash mismatch")
    _validate_snapshot_shape(snapshot)


def verify_formal_runtime_attestation_v2(
    *,
    project_root: Path,
    attestation_receipt_path: Path,
    base_binding_receipt_path: Path,
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
    bypass_cache: bool = False,
) -> dict[str, Any]:
    """Verify filesystem evidence, optionally forcing a fresh postflight read."""

    if not isinstance(bypass_cache, bool):
        raise MuSiQueOfficialHippoRAGError("bypass_cache must be boolean")

    project_root = project_root.resolve(strict=True)
    attestation_path = attestation_receipt_path.absolute()
    base_path = base_binding_receipt_path.absolute()
    payload, attestation_raw = _load_json_object(attestation_path, "v2 attestation receipt")
    _validate_receipt_structure(payload)
    base, base_file_hash = _base_binding(base_path, project_root=project_root)
    expected_base = {
        "file_sha256": base_file_hash,
        "qualification_sha256": base["qualification_binding"]["qualification_sha256"],
        "receipt_sha256": base["receipt_sha256"],
        "schema": base["schema"],
    }
    if payload.get("base_binding") != expected_base:
        raise MuSiQueOfficialHippoRAGError("v2 base binding drifted")
    if payload.get("implementation_binding") != current_v2_implementation_binding(
        project_root
    ):
        raise MuSiQueOfficialHippoRAGError("live v2 attestation implementation drifted")
    cache_key = (
        str(attestation_path),
        _sha256_bytes(attestation_raw),
        str(base_path),
        base_file_hash,
        str(runtime_python.absolute()),
        str(local_llm_model.resolve(strict=True)),
        str(local_embedding_model.resolve(strict=True)),
    )
    with _ATTESTATION_CACHE_LOCK:
        cached = _ATTESTATION_CACHE.get(cache_key)
        if cached is not None and not bypass_cache:
            return dict(cached)
        if bypass_cache:
            # A failed fresh postflight must not leave a stale success reusable.
            _ATTESTATION_CACHE.pop(cache_key, None)
        expected_versions = base["runtime_binding"]["dependency_versions"]
        observed = _filesystem_snapshot(
            runtime_python=runtime_python,
            local_llm_model=local_llm_model,
            local_embedding_model=local_embedding_model,
            expected_versions=expected_versions,
        )
        _validate_snapshot_shape(observed)
        _validate_snapshot_against_base(observed, base)
        if observed != payload.get("runtime_filesystem_binding"):
            raise MuSiQueOfficialHippoRAGError(
                "formal filesystem runtime attestation drifted"
            )
        result = {
            "attestation_receipt_sha256": payload["receipt_sha256"],
            "base_binding_receipt_sha256": base["receipt_sha256"],
            "formal_entry_executable_identity_probe_calls": 0,
            "implementation_set_sha256": payload["implementation_binding"][
                "set_sha256"
            ],
            "runtime_filesystem_binding_sha256": stable_hash(observed),
        }
        _ATTESTATION_CACHE[cache_key] = result
        return dict(result)


__all__ = [
    "ATTESTATION_SCHEMA",
    "FORMAL_ENTRY_POLICY",
    "current_v2_implementation_binding",
    "qualify_and_build_attestation_v2",
    "verify_formal_runtime_attestation_v2",
]
