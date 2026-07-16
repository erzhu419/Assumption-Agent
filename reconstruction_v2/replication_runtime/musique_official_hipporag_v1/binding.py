"""Frozen binding and live verification for the official retrieve-only runtime."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
from typing import Any, Mapping, Sequence

from assumption_agent.models import stable_hash

from .contract import FROZEN_CORE_CONFIG, MuSiQueOfficialHippoRAGError


BINDING_SCHEMA = "musique_official_hipporag_retrieve_only_binding_v1"
IMPLEMENTATION_SCHEMA = "musique_official_hipporag_retrieve_only_implementation_v1"
QUALIFICATION_SCHEMA = "official-hipporag-runtime-adapter-qualification-v1"
QUALIFICATION_RELATIVE_PATH = (
    "manifests/official_hipporag_runtime_adapter_qualification_v1.json"
)
OFFICIAL_COMMIT = "ef2f14c4f254f11ac29f9395f262466ad1bb4d10"
OFFICIAL_REMOTE = "https://github.com/OSU-NLP-Group/HippoRAG.git"
OFFICIAL_SOURCE_FILE_COUNT = 52
OFFICIAL_SOURCE_TREE_SHA256 = (
    "30941a14e8dc48f7a41f8679ce6cba0bac9e3cdd99ed919560b45872e1058700"
)
QUALIFICATION_SHA256 = "c2a6b540e4b91347a23bbe918b495caebcc35a23fbacee9754cd1b7661fda4e4"
OFFICIAL_OPENAI_PIN = "1.91.1"
QUALIFIED_RUNTIME_OPENAI_VERSION = "1.91.0"
EXPECTED_RUNTIME_KIND = "dedicated_overlay_venv"
DEPENDENCY_NAMES = (
    "gritlm",
    "hipporag",
    "litellm",
    "networkx",
    "openai",
    "pydantic",
    "python-igraph",
    "sentence-transformers",
    "tenacity",
    "tiktoken",
    "torch",
    "transformers",
    "vllm",
)
IMPLEMENTATION_RELATIVE_FILES = (
    "assumption_agent/__init__.py",
    "assumption_agent/models.py",
    "replication_runtime/__init__.py",
    "replication_runtime/musique_official_hipporag_v1/__init__.py",
    "replication_runtime/musique_official_hipporag_v1/adapter.py",
    "replication_runtime/musique_official_hipporag_v1/binding.py",
    "replication_runtime/musique_official_hipporag_v1/contract.py",
    "replication_runtime/musique_official_hipporag_v1/worker.py",
)
BINDING_TOP_LEVEL_KEYS = frozenset(
    {
        "asset_binding",
        "config_binding",
        "decision",
        "implementation_binding",
        "limitations",
        "official_source_binding",
        "qualification_binding",
        "receipt_sha256",
        "runtime_binding",
        "schema",
        "scope",
        "synthetic_local_qualification",
    }
)
QUALIFICATION_BINDING_KEYS = frozenset({"path", "file_sha256", "qualification_sha256"})
OFFICIAL_SOURCE_BINDING_KEYS = frozenset(
    {"remote", "commit", "python_source_file_count", "python_source_tree_sha256"}
)
RUNTIME_BINDING_KEYS = frozenset(
    {
        "runtime_kind",
        "python_version",
        "python_implementation",
        "runtime_python_target_sha256",
        "pyvenv_cfg_sha256",
        "dependency_versions",
        "dependency_set_sha256",
        "venv_identity_sha256",
        "official_openai_pin",
        "runtime_openai_version",
        "openai_pin_satisfied",
        "openai_1_91_0_deviation_explicitly_bound",
        "installed_source_file_count",
        "installed_source_tree_sha256",
    }
)
ASSET_BINDING_KEYS = frozenset(
    {
        "local_llm_asset_sha256",
        "local_embedding_asset_sha256",
        "asset_paths_persisted_publicly",
        "assets_are_offline_local_only",
    }
)
CONFIG_BINDING_KEYS = frozenset({"payload", "config_sha256"})
IMPLEMENTATION_BINDING_KEYS = frozenset({"schema", "files", "set_sha256"})
IMPLEMENTATION_FILE_KEYS = frozenset({"path", "sha256"})
SCOPE_KEYS = frozenset(
    {
        "official_core_calls",
        "answer_generation_calls",
        "online_evaluator_calls",
        "benchmark_rows_read_while_binding",
        "one_item_one_independent_index",
        "result_payload",
        "stable_tie_break",
    }
)
SYNTHETIC_QUALIFICATION_KEYS = frozenset(
    {
        "benchmark_rows_read",
        "candidate_count",
        "external_network_transport_possible",
        "fixture_sha256",
        "network_namespace_isolated",
        "official_core_index_called",
        "official_core_retrieve_called",
        "output_idx_count",
        "output_idx_sha256",
        "scores_computed",
        "status",
    }
)
FROZEN_SCOPE: dict[str, Any] = {
    "official_core_calls": ["index", "retrieve"],
    "answer_generation_calls": 0,
    "online_evaluator_calls": 0,
    "benchmark_rows_read_while_binding": 0,
    "one_item_one_independent_index": True,
    "result_payload": "exact_json_array_of_five_paragraph_idx_only",
    "stable_tie_break": "paragraph_idx_ascending",
}
FROZEN_LIMITATIONS = (
    "The frozen local causal model is an infrastructure asset; this binding does not claim answer-generation quality.",
    "The official openai==1.91.1 declaration remains unsatisfied; the qualified overlay deviation is exactly openai==1.91.0.",
    "This receipt qualifies and freezes retrieval infrastructure only; it contains no MuSiQue score or benchmark row.",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CACHE_LOCK = threading.Lock()
_LIVE_CACHE: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise MuSiQueOfficialHippoRAGError(f"{field} must be lowercase sha256")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], field: str) -> None:
    if set(value) != expected:
        raise MuSiQueOfficialHippoRAGError(f"{field} key set mismatch")


def _validated_implementation_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(value, IMPLEMENTATION_BINDING_KEYS, "implementation binding")
    if value.get("schema") != IMPLEMENTATION_SCHEMA:
        raise MuSiQueOfficialHippoRAGError("implementation binding schema mismatch")
    files = value.get("files")
    if not isinstance(files, list) or len(files) != len(IMPLEMENTATION_RELATIVE_FILES):
        raise MuSiQueOfficialHippoRAGError("implementation file set mismatch")
    rows: list[dict[str, str]] = []
    for expected_path, row in zip(IMPLEMENTATION_RELATIVE_FILES, files):
        if not isinstance(row, Mapping):
            raise MuSiQueOfficialHippoRAGError("implementation file row is malformed")
        _require_exact_keys(row, IMPLEMENTATION_FILE_KEYS, "implementation file row")
        if row.get("path") != expected_path:
            raise MuSiQueOfficialHippoRAGError("implementation file order or path mismatch")
        rows.append(
            {
                "path": expected_path,
                "sha256": _require_sha256(row.get("sha256"), "implementation file hash"),
            }
        )
    set_sha256 = _require_sha256(value.get("set_sha256"), "implementation set hash")
    if stable_hash(rows) != set_sha256:
        raise MuSiQueOfficialHippoRAGError("implementation set hash mismatch")
    return {"schema": IMPLEMENTATION_SCHEMA, "files": rows, "set_sha256": set_sha256}


def _load_json_object(path: Path, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MuSiQueOfficialHippoRAGError(f"{field} is unavailable")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MuSiQueOfficialHippoRAGError(f"{field} is invalid") from exc
    if not isinstance(value, dict):
        raise MuSiQueOfficialHippoRAGError(f"{field} must be an object")
    return value


def _project_root_from_binding_path(binding_receipt_path: Path) -> Path:
    path = binding_receipt_path.absolute()
    if path.parent.name != "manifests":
        raise MuSiQueOfficialHippoRAGError("binding receipt must be under manifests")
    root = path.parent.parent
    if not root.is_dir():
        raise MuSiQueOfficialHippoRAGError("project root is unavailable")
    return root


def current_implementation_binding(project_root: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    for relative in IMPLEMENTATION_RELATIVE_FILES:
        path = project_root / relative
        if path.is_symlink() or not path.is_file():
            raise MuSiQueOfficialHippoRAGError(f"implementation file missing: {relative}")
        rows.append({"path": relative, "sha256": _sha256_file(path)})
    return {
        "schema": IMPLEMENTATION_SCHEMA,
        "files": rows,
        "set_sha256": stable_hash(rows),
    }


def _qualified_receipt(project_root: Path) -> tuple[dict[str, Any], str]:
    path = project_root / QUALIFICATION_RELATIVE_PATH
    payload = _load_json_object(path, "qualification receipt")
    declared = payload.get("qualification_sha256")
    without_hash = dict(payload)
    without_hash.pop("qualification_sha256", None)
    if payload.get("schema") != QUALIFICATION_SCHEMA or stable_hash(without_hash) != declared:
        raise MuSiQueOfficialHippoRAGError("qualification receipt self-hash mismatch")
    source = payload.get("source_binding")
    dependency = payload.get("dependency_boundary")
    core = payload.get("official_core")
    if not isinstance(source, Mapping) or not isinstance(dependency, Mapping) or not isinstance(
        core, Mapping
    ):
        raise MuSiQueOfficialHippoRAGError("qualification receipt is incomplete")
    if (
        payload.get("qualified") is not True
        or declared != QUALIFICATION_SHA256
        or source.get("commit") != OFFICIAL_COMMIT
        or source.get("remote") != OFFICIAL_REMOTE
        or source.get("python_source_file_count") != OFFICIAL_SOURCE_FILE_COUNT
        or source.get("python_source_tree_sha256") != OFFICIAL_SOURCE_TREE_SHA256
        or dependency.get("runtime_kind") != EXPECTED_RUNTIME_KIND
        or dependency.get("official_declared_openai_pin") != OFFICIAL_OPENAI_PIN
        or dependency.get("runtime_openai_version") != QUALIFIED_RUNTIME_OPENAI_VERSION
        or dependency.get("declared_openai_pin_satisfied") is not False
        or core.get("status") != "passed"
        or core.get("installed_python_source_tree_matches_commit") is not True
    ):
        raise MuSiQueOfficialHippoRAGError("qualification binding drifted")
    return payload, _sha256_file(path)


def _asset_hash(path: Path, *, row_kind: str) -> str:
    path = path.resolve(strict=True)
    if not path.is_dir():
        raise MuSiQueOfficialHippoRAGError("local model asset is not a directory")
    file_paths = sorted(
        (entry for entry in path.rglob("*") if entry.is_file()),
        key=lambda entry: entry.relative_to(path).as_posix(),
    )
    if not file_paths:
        raise MuSiQueOfficialHippoRAGError("local model asset is empty")
    if row_kind == "dict":
        rows: object = [
            {"path": entry.relative_to(path).as_posix(), "sha256": _sha256_file(entry)}
            for entry in file_paths
        ]
    elif row_kind == "tuple":
        rows = [
            (entry.relative_to(path).as_posix(), _sha256_file(entry))
            for entry in file_paths
        ]
    else:
        raise AssertionError("unknown asset binding row kind")
    return stable_hash(rows)


_RUNTIME_PROBE_PROGRAM = r"""
import hashlib
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
import json
from pathlib import Path
import platform

names = json.loads(__import__('sys').argv[1])
spec = find_spec('hipporag')
assert spec is not None and spec.submodule_search_locations
package_root = Path(next(iter(spec.submodule_search_locations))).resolve()
python_rows = [
    {
        'path': path.relative_to(package_root).as_posix(),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    for path in sorted(package_root.rglob('*.py'))
]
tree_hash = hashlib.sha256(
    json.dumps(python_rows, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode('utf-8')
).hexdigest()
versions = {}
for name in names:
    try:
        versions[name] = version(name)
    except PackageNotFoundError:
        versions[name] = None
print(json.dumps({
    'python_version': platform.python_version(),
    'python_implementation': platform.python_implementation(),
    'dependency_versions': versions,
    'installed_source_file_count': len(python_rows),
    'installed_source_tree_sha256': tree_hash,
}, sort_keys=True, separators=(',', ':')))
""".strip()


def _runtime_probe(runtime_python: Path) -> dict[str, Any]:
    runtime_python = runtime_python.absolute()
    if not runtime_python.is_file() or not os.access(runtime_python, os.X_OK):
        raise MuSiQueOfficialHippoRAGError("runtime Python is unavailable")
    environment = {
        "PATH": f"{runtime_python.parent}:/usr/bin:/bin",
        "HOME": "/nonexistent",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    completed = subprocess.run(
        [
            str(runtime_python),
            "-I",
            "-c",
            _RUNTIME_PROBE_PROGRAM,
            json.dumps(list(DEPENDENCY_NAMES), separators=(",", ":")),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )
    if completed.returncode != 0:
        raise MuSiQueOfficialHippoRAGError(
            "runtime identity probe failed; "
            f"stderr_sha256={_sha256_bytes(completed.stderr.encode('utf-8'))}"
        )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise MuSiQueOfficialHippoRAGError("runtime identity probe output is invalid") from exc
    if not isinstance(payload, dict):
        raise MuSiQueOfficialHippoRAGError("runtime identity probe output is malformed")
    pyvenv_path = runtime_python.parent.parent / "pyvenv.cfg"
    if pyvenv_path.is_symlink() or not pyvenv_path.is_file():
        raise MuSiQueOfficialHippoRAGError("runtime is not the frozen overlay venv")
    payload["runtime_python_target_sha256"] = _sha256_file(runtime_python.resolve(strict=True))
    payload["pyvenv_cfg_sha256"] = _sha256_file(pyvenv_path)
    dependency_versions = payload.get("dependency_versions")
    if not isinstance(dependency_versions, Mapping) or set(dependency_versions) != set(
        DEPENDENCY_NAMES
    ):
        raise MuSiQueOfficialHippoRAGError("runtime dependency set is malformed")
    rows = [
        {"name": name, "version": dependency_versions[name]}
        for name in DEPENDENCY_NAMES
    ]
    payload["dependency_set_sha256"] = stable_hash(rows)
    payload["venv_identity_sha256"] = stable_hash(
        {
            "python_version": payload.get("python_version"),
            "python_implementation": payload.get("python_implementation"),
            "runtime_python_target_sha256": payload["runtime_python_target_sha256"],
            "pyvenv_cfg_sha256": payload["pyvenv_cfg_sha256"],
            "dependency_set_sha256": payload["dependency_set_sha256"],
        }
    )
    return payload


def _runtime_binding_from_probe(
    probe: Mapping[str, Any], qualification: Mapping[str, Any]
) -> dict[str, Any]:
    versions = probe.get("dependency_versions")
    qualified_versions = qualification.get("official_core", {}).get("runtime_versions")
    if not isinstance(versions, Mapping) or not isinstance(qualified_versions, Mapping):
        raise MuSiQueOfficialHippoRAGError("runtime versions are unavailable")
    expected_versions = {name: qualified_versions.get(name) for name in DEPENDENCY_NAMES}
    if dict(versions) != expected_versions:
        raise MuSiQueOfficialHippoRAGError("runtime dependencies drifted from qualification")
    if (
        probe.get("installed_source_file_count") != OFFICIAL_SOURCE_FILE_COUNT
        or probe.get("installed_source_tree_sha256") != OFFICIAL_SOURCE_TREE_SHA256
    ):
        raise MuSiQueOfficialHippoRAGError("installed official source tree drifted")
    return {
        "runtime_kind": EXPECTED_RUNTIME_KIND,
        "python_version": probe.get("python_version"),
        "python_implementation": probe.get("python_implementation"),
        "runtime_python_target_sha256": _require_sha256(
            probe.get("runtime_python_target_sha256"), "runtime Python target hash"
        ),
        "pyvenv_cfg_sha256": _require_sha256(
            probe.get("pyvenv_cfg_sha256"), "pyvenv hash"
        ),
        "dependency_versions": expected_versions,
        "dependency_set_sha256": _require_sha256(
            probe.get("dependency_set_sha256"), "dependency set hash"
        ),
        "venv_identity_sha256": _require_sha256(
            probe.get("venv_identity_sha256"), "venv identity hash"
        ),
        "official_openai_pin": OFFICIAL_OPENAI_PIN,
        "runtime_openai_version": QUALIFIED_RUNTIME_OPENAI_VERSION,
        "openai_pin_satisfied": False,
        "openai_1_91_0_deviation_explicitly_bound": True,
        "installed_source_file_count": OFFICIAL_SOURCE_FILE_COUNT,
        "installed_source_tree_sha256": OFFICIAL_SOURCE_TREE_SHA256,
    }


def build_binding_receipt(
    *,
    project_root: Path,
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
    synthetic_local_qualification: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a safe binding without opening or indexing any benchmark data."""

    project_root = project_root.resolve(strict=True)
    qualification, qualification_file_hash = _qualified_receipt(project_root)
    runtime = _runtime_binding_from_probe(_runtime_probe(runtime_python), qualification)
    dependency = qualification["dependency_boundary"]
    llm_hash = _asset_hash(local_llm_model, row_kind="dict")
    embedding_hash = _asset_hash(local_embedding_model, row_kind="tuple")
    if (
        llm_hash != dependency.get("local_llm_asset_sha256")
        or embedding_hash != dependency.get("local_embedding_asset_sha256")
    ):
        raise MuSiQueOfficialHippoRAGError("local model assets drifted from qualification")
    config_payload = dict(FROZEN_CORE_CONFIG)
    receipt: dict[str, Any] = {
        "schema": BINDING_SCHEMA,
        "decision": "frozen_official_core_retrieve_only_adapter",
        "qualification_binding": {
            "path": QUALIFICATION_RELATIVE_PATH,
            "file_sha256": qualification_file_hash,
            "qualification_sha256": QUALIFICATION_SHA256,
        },
        "official_source_binding": {
            "remote": OFFICIAL_REMOTE,
            "commit": OFFICIAL_COMMIT,
            "python_source_file_count": OFFICIAL_SOURCE_FILE_COUNT,
            "python_source_tree_sha256": OFFICIAL_SOURCE_TREE_SHA256,
        },
        "runtime_binding": runtime,
        "asset_binding": {
            "local_llm_asset_sha256": llm_hash,
            "local_embedding_asset_sha256": embedding_hash,
            "asset_paths_persisted_publicly": False,
            "assets_are_offline_local_only": True,
        },
        "config_binding": {
            "payload": config_payload,
            "config_sha256": stable_hash(config_payload),
        },
        "implementation_binding": current_implementation_binding(project_root),
        "synthetic_local_qualification": dict(synthetic_local_qualification),
        "scope": dict(FROZEN_SCOPE),
        "limitations": list(FROZEN_LIMITATIONS),
    }
    receipt["receipt_sha256"] = stable_hash(receipt)
    return receipt


def validate_binding_receipt(
    payload: Mapping[str, Any], *, project_root: Path, verify_implementation: bool = True
) -> dict[str, Any]:
    _require_exact_keys(payload, BINDING_TOP_LEVEL_KEYS, "binding receipt")
    normalized = dict(payload)
    declared = _require_sha256(normalized.pop("receipt_sha256", None), "receipt hash")
    if payload.get("schema") != BINDING_SCHEMA or stable_hash(normalized) != declared:
        raise MuSiQueOfficialHippoRAGError("binding receipt self-hash mismatch")
    if payload.get("decision") != "frozen_official_core_retrieve_only_adapter":
        raise MuSiQueOfficialHippoRAGError("binding decision mismatch")
    if payload.get("limitations") != list(FROZEN_LIMITATIONS):
        raise MuSiQueOfficialHippoRAGError("binding limitations mismatch")
    qualification, qualification_file_hash = _qualified_receipt(project_root)
    qualification_binding = payload.get("qualification_binding")
    source = payload.get("official_source_binding")
    runtime = payload.get("runtime_binding")
    assets = payload.get("asset_binding")
    config = payload.get("config_binding")
    scope = payload.get("scope")
    implementation = payload.get("implementation_binding")
    synthetic = payload.get("synthetic_local_qualification")
    if not all(
        isinstance(value, Mapping)
        for value in (
            qualification_binding,
            source,
            runtime,
            assets,
            config,
            scope,
            implementation,
            synthetic,
        )
    ):
        raise MuSiQueOfficialHippoRAGError("binding receipt sections are incomplete")
    _require_exact_keys(
        qualification_binding, QUALIFICATION_BINDING_KEYS, "qualification binding"
    )
    _require_exact_keys(source, OFFICIAL_SOURCE_BINDING_KEYS, "official source binding")
    _require_exact_keys(runtime, RUNTIME_BINDING_KEYS, "runtime binding")
    _require_exact_keys(assets, ASSET_BINDING_KEYS, "asset binding")
    _require_exact_keys(config, CONFIG_BINDING_KEYS, "config binding")
    _require_exact_keys(scope, SCOPE_KEYS, "scope")
    _require_exact_keys(
        synthetic, SYNTHETIC_QUALIFICATION_KEYS, "synthetic local qualification"
    )
    frozen_implementation = _validated_implementation_binding(implementation)
    if qualification_binding != {
        "path": QUALIFICATION_RELATIVE_PATH,
        "file_sha256": qualification_file_hash,
        "qualification_sha256": QUALIFICATION_SHA256,
    }:
        raise MuSiQueOfficialHippoRAGError("qualification binding mismatch")
    if source != {
        "remote": OFFICIAL_REMOTE,
        "commit": OFFICIAL_COMMIT,
        "python_source_file_count": OFFICIAL_SOURCE_FILE_COUNT,
        "python_source_tree_sha256": OFFICIAL_SOURCE_TREE_SHA256,
    }:
        raise MuSiQueOfficialHippoRAGError("official source binding mismatch")
    qualified_dependency = qualification["dependency_boundary"]
    dependency_versions = runtime.get("dependency_versions")
    if not isinstance(dependency_versions, Mapping):
        raise MuSiQueOfficialHippoRAGError("runtime dependency versions are malformed")
    _require_exact_keys(
        dependency_versions, frozenset(DEPENDENCY_NAMES), "runtime dependency versions"
    )
    expected_dependency_versions = {
        name: qualification["official_core"]["runtime_versions"].get(name)
        for name in DEPENDENCY_NAMES
    }
    dependency_rows = [
        {"name": name, "version": dependency_versions[name]} for name in DEPENDENCY_NAMES
    ]
    expected_venv_identity = stable_hash(
        {
            "python_version": runtime.get("python_version"),
            "python_implementation": runtime.get("python_implementation"),
            "runtime_python_target_sha256": runtime.get("runtime_python_target_sha256"),
            "pyvenv_cfg_sha256": runtime.get("pyvenv_cfg_sha256"),
            "dependency_set_sha256": runtime.get("dependency_set_sha256"),
        }
    )
    if (
        runtime.get("runtime_kind") != EXPECTED_RUNTIME_KIND
        or not isinstance(runtime.get("python_version"), str)
        or not runtime.get("python_version")
        or runtime.get("python_implementation") != "CPython"
        or _SHA256.fullmatch(str(runtime.get("runtime_python_target_sha256"))) is None
        or _SHA256.fullmatch(str(runtime.get("pyvenv_cfg_sha256"))) is None
        or dependency_versions != expected_dependency_versions
        or runtime.get("dependency_set_sha256") != stable_hash(dependency_rows)
        or runtime.get("venv_identity_sha256") != expected_venv_identity
        or runtime.get("runtime_openai_version") != QUALIFIED_RUNTIME_OPENAI_VERSION
        or runtime.get("official_openai_pin") != OFFICIAL_OPENAI_PIN
        or runtime.get("openai_pin_satisfied") is not False
        or runtime.get("openai_1_91_0_deviation_explicitly_bound") is not True
        or runtime.get("installed_source_file_count") != OFFICIAL_SOURCE_FILE_COUNT
        or runtime.get("installed_source_tree_sha256") != OFFICIAL_SOURCE_TREE_SHA256
        or assets.get("local_llm_asset_sha256")
        != qualified_dependency.get("local_llm_asset_sha256")
        or assets.get("local_embedding_asset_sha256")
        != qualified_dependency.get("local_embedding_asset_sha256")
        or assets.get("asset_paths_persisted_publicly") is not False
        or assets.get("assets_are_offline_local_only") is not True
        or config != {
            "payload": FROZEN_CORE_CONFIG,
            "config_sha256": stable_hash(FROZEN_CORE_CONFIG),
        }
        or scope != FROZEN_SCOPE
        or synthetic.get("status")
        != "passed_non_scoring_synthetic_local_retrieve_only"
        or synthetic.get("official_core_index_called") is not True
        or synthetic.get("official_core_retrieve_called") is not True
        or synthetic.get("network_namespace_isolated") is not True
        or synthetic.get("external_network_transport_possible") is not False
        or synthetic.get("benchmark_rows_read") != 0
        or synthetic.get("scores_computed") != 0
        or synthetic.get("candidate_count") != 5
        or synthetic.get("output_idx_count") != 5
        or _SHA256.fullmatch(str(synthetic.get("fixture_sha256"))) is None
        or _SHA256.fullmatch(str(synthetic.get("output_idx_sha256"))) is None
    ):
        raise MuSiQueOfficialHippoRAGError("frozen runtime or adapter contract drifted")
    if verify_implementation and frozen_implementation != current_implementation_binding(
        project_root
    ):
        raise MuSiQueOfficialHippoRAGError("live adapter implementation drifted")
    return dict(payload)


def verify_live_binding(
    *,
    binding_receipt_path: Path,
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
) -> dict[str, Any]:
    """Verify code, official source, venv, dependencies, and both local assets."""

    project_root = _project_root_from_binding_path(binding_receipt_path)
    payload = _load_json_object(binding_receipt_path, "binding receipt")
    validate_binding_receipt(payload, project_root=project_root, verify_implementation=True)
    key = (
        str(binding_receipt_path.absolute()),
        _sha256_file(binding_receipt_path),
        str(runtime_python.absolute()),
        str(local_llm_model.resolve(strict=True)),
        str(local_embedding_model.resolve(strict=True)),
    )
    with _CACHE_LOCK:
        cached = _LIVE_CACHE.get(key)
        if cached is not None:
            return dict(cached)
        qualification, _ = _qualified_receipt(project_root)
        runtime = _runtime_binding_from_probe(_runtime_probe(runtime_python), qualification)
        if runtime != payload.get("runtime_binding"):
            raise MuSiQueOfficialHippoRAGError("live venv identity drifted")
        assets = payload["asset_binding"]
        if (
            _asset_hash(local_llm_model, row_kind="dict")
            != assets.get("local_llm_asset_sha256")
            or _asset_hash(local_embedding_model, row_kind="tuple")
            != assets.get("local_embedding_asset_sha256")
        ):
            raise MuSiQueOfficialHippoRAGError("live local model asset drifted")
        result = {
            "receipt_sha256": payload["receipt_sha256"],
            "venv_identity_sha256": runtime["venv_identity_sha256"],
            "config_sha256": payload["config_binding"]["config_sha256"],
            "implementation_set_sha256": payload["implementation_binding"]["set_sha256"],
        }
        _LIVE_CACHE[key] = result
        return dict(result)
