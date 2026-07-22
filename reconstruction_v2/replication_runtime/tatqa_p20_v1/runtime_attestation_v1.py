"""Source-free attestation for the exact P19 HippoRAG runtime on jtl311linux.

This is intentionally not an extension of the MuSiQue v3 attestation.  That
receipt described a different source overlay and a different dependency
topology.  P19 instead binds the complete, hardened P17 source tree, the exact
SmolLM and MiniLM trees, the QASPER MiniLM manifest, and the metadata selected
by the *active* HippoRAG interpreter.

The builder contains no TAT-QA loader, performs no network operation, and does
not import a model package.  ``find_spec`` plus distribution metadata are used
to prove which source and packages the active interpreter would resolve.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import platform
import re
import site
import stat
import sys
from typing import Any, Mapping, Sequence


SCHEMA = "tatqa_p19_hipporag_runtime_attestation_v1"
IMPLEMENTATION_SCHEMA = "tatqa_p19_hipporag_runtime_attestation_builder_v1"

EXPECTED_SOURCE_TREE = {
    "file_count": 96,
    "size_bytes": 527_994,
    "tree_sha256": "a644ab2811db2739db3cfbdc051561e2cfdf2ed87286f8ebd00a5971d189cdd5",
}
EXPECTED_SMOLLM_TREE = {
    "file_count": 23,
    "size_bytes": 272_031_142,
    "tree_sha256": "d626d755c99c006761d5e069aa85a73fe8b011c6c0f5d0323a6f8de85246bcb5",
}
EXPECTED_MINILM_GENERIC_TREE = {
    "file_count": 11,
    "size_bytes": 91_578_415,
    "tree_sha256": "42d8d798e4f01e68d9bb10634b9c712de00f7f8495271636fd6311b2db58e506",
}
EXPECTED_MINILM_NORMATIVE_TREE_SHA256 = (
    "1514beb65d2d3a2824a93f133a300cc60d5b437ccd6ea1e622eb4cd9881dcfdb"
)
EXPECTED_MINILM_WEIGHTS_SHA256 = (
    "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
)
EXPECTED_MINILM_MANIFEST_FILE_SHA256 = (
    "62b85c7752f2e46932fb9fb13ae2f3aac9eb750a33c8f07102739040feb6cc75"
)
EXPECTED_MINILM_MANIFEST_SELF_SHA256 = (
    "921d9b1945581130e03c53f448092c3de3b30714431c6cac9b3b32c2ec10abad"
)
EXPECTED_RUNTIME_PYTHON_TARGET_SHA256 = (
    "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"
)
EXPECTED_PYVENV_CFG_SHA256 = (
    "973ff55fad570c3922d91779ff66db497b7fdf69c55ec102ecfd9f3b6b711e45"
)

SMOLLM_REPO_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
SMOLLM_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
MINILM_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

# These versions are the metadata actually selected by the reusable P17
# HippoRAG interpreter.  They are deliberately distinct from the QASPER
# controller contract below; the receipt must expose, not conceal, that fact.
EXPECTED_ACTIVE_DISTRIBUTION_VERSIONS: dict[str, str | None] = {
    "gritlm": "1.0.2",
    "hipporag": "2.0.0a4",
    "huggingface-hub": "0.25.2",
    "litellm": "1.73.1",
    "networkx": "3.3",
    "numpy": "2.1.3",
    "openai": "1.91.0",
    "pydantic": "2.10.4",
    "python-igraph": "0.11.8",
    "safetensors": "0.4.5",
    "scikit-learn": "1.5.2",
    "scipy": "1.14.1",
    "sentence-transformers": "3.1.1",
    "tenacity": "8.5.0",
    "tiktoken": "0.7.0",
    "tokenizers": "0.20.3",
    "torch": "2.4.1+cu118",
    "transformers": "4.45.2",
    "vllm": None,
}

QASPER_CONTROLLER_REQUIRED_VERSIONS = {
    "huggingface-hub": "1.11.0",
    "numpy": "2.2.6",
    "safetensors": "0.7.0",
    "sentence-transformers": "5.5.1",
    "tokenizers": "0.22.2",
    "torch": "2.8.0+cu128",
    "transformers": "5.10.1",
}

ACTIVE_MODULES = {
    "hipporag": "hipporag",
    "networkx": "networkx",
    "numpy": "numpy",
    "openai": "openai",
    "python-igraph": "igraph",
    "sentence-transformers": "sentence_transformers",
    "torch": "torch",
    "transformers": "transformers",
}

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_CANONICAL_NAME = re.compile(r"[-_.]+")


class TatqaP19RuntimeAttestationError(RuntimeError):
    """The public runtime or an attestation receipt failed closed."""


def canonical_json_bytes(value: object, *, newline: bool = True) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise TatqaP19RuntimeAttestationError(
            "value is not canonical JSON"
        ) from exc
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
        raise TatqaP19RuntimeAttestationError("bound file is unreadable") from exc
    return digest.hexdigest()


def _absolute_without_symlink(path: Path, field: str) -> Path:
    absolute = path.expanduser().absolute()
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise TatqaP19RuntimeAttestationError(f"{field} contains a symlink")
    return absolute


def _tree_rows(root: Path, field: str) -> list[dict[str, object]]:
    root = _absolute_without_symlink(root, field)
    if not root.is_dir():
        raise TatqaP19RuntimeAttestationError(f"{field} is unavailable")
    rows: list[dict[str, object]] = []
    for current, directories, files in os.walk(root, followlinks=False):
        base = Path(current)
        for name in directories:
            if (base / name).is_symlink():
                raise TatqaP19RuntimeAttestationError(
                    f"{field} contains a directory symlink"
                )
        for name in files:
            path = base / name
            if path.is_symlink() or not path.is_file():
                raise TatqaP19RuntimeAttestationError(
                    f"{field} contains a non-regular file"
                )
            size = path.stat().st_size
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                    "size_bytes": size,
                }
            )
    rows.sort(key=lambda row: str(row["path"]))
    return rows


def tree_receipt(root: Path, field: str) -> dict[str, object]:
    rows = _tree_rows(root, field)
    return {
        "file_count": len(rows),
        "size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "tree_sha256": stable_hash(rows),
    }


def _require_tree(
    root: Path, field: str, expected: Mapping[str, object]
) -> dict[str, object]:
    observed = tree_receipt(root, field)
    if observed != dict(expected):
        raise TatqaP19RuntimeAttestationError(f"{field} tree drifted")
    return observed


def _canonical_distribution_name(value: str) -> str:
    return _CANONICAL_NAME.sub("-", value).casefold()


def _dist_info_tree(path: Path) -> dict[str, object]:
    rows = _tree_rows(path, "distribution metadata")
    return {
        "file_count": len(rows),
        "tree_sha256": stable_hash(rows),
    }


def _distribution_row(name: str, expected_version: str | None) -> dict[str, object]:
    try:
        distribution = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        if expected_version is not None:
            raise TatqaP19RuntimeAttestationError(
                f"required active distribution is absent: {name}"
            )
        return {
            "dist_info_file_count": 0,
            "dist_info_name": None,
            "dist_info_tree_sha256": None,
            "name": name,
            "root_path": None,
            "version": None,
        }
    actual_name = distribution.metadata.get("Name")
    version = distribution.version
    dist_info = getattr(distribution, "_path", None)
    if (
        not isinstance(actual_name, str)
        or _canonical_distribution_name(actual_name)
        != _canonical_distribution_name(name)
        or version != expected_version
        or not isinstance(dist_info, Path)
        or dist_info.is_symlink()
        or not dist_info.is_dir()
    ):
        raise TatqaP19RuntimeAttestationError(
            f"active distribution identity drifted: {name}"
        )
    binding = _dist_info_tree(dist_info)
    return {
        "dist_info_file_count": binding["file_count"],
        "dist_info_name": dist_info.name,
        "dist_info_tree_sha256": binding["tree_sha256"],
        "name": name,
        "root_path": str(dist_info.parent.absolute()),
        "version": version,
    }


def _active_distribution_binding() -> dict[str, object]:
    rows = [
        _distribution_row(name, version)
        for name, version in EXPECTED_ACTIVE_DISTRIBUTION_VERSIONS.items()
    ]
    return {"rows": rows, "set_sha256": stable_hash(rows)}


def _module_row(distribution_name: str, module_name: str) -> dict[str, object]:
    try:
        specification = util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError) as exc:
        raise TatqaP19RuntimeAttestationError(
            f"active module cannot be resolved: {module_name}"
        ) from exc
    origin = None if specification is None else specification.origin
    if not isinstance(origin, str) or origin in {"built-in", "frozen"}:
        raise TatqaP19RuntimeAttestationError(
            f"active module origin is unavailable: {module_name}"
        )
    path = Path(origin).expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise TatqaP19RuntimeAttestationError(
            f"active module origin drifted: {module_name}"
        )
    return {
        "distribution_name": distribution_name,
        "module_name": module_name,
        "origin_file_sha256": file_sha256(path),
        "origin_path": str(path),
    }


def _active_module_binding(source_root: Path) -> dict[str, object]:
    rows = [
        _module_row(distribution_name, module_name)
        for distribution_name, module_name in ACTIVE_MODULES.items()
    ]
    rows.sort(key=lambda row: str(row["distribution_name"]))
    hippo = next(row for row in rows if row["distribution_name"] == "hipporag")
    source = _absolute_without_symlink(source_root, "HippoRAG source")
    try:
        Path(str(hippo["origin_path"])).relative_to(source)
    except ValueError as exc:
        raise TatqaP19RuntimeAttestationError(
            "active hipporag module is not loaded from the hardened source tree"
        ) from exc
    return {"rows": rows, "set_sha256": stable_hash(rows)}


def _runtime_python_binding(runtime_python: Path) -> dict[str, object]:
    lexical = runtime_python.expanduser().absolute()
    active = Path(sys.executable).expanduser().absolute()
    if lexical != active:
        raise TatqaP19RuntimeAttestationError(
            "builder is not running under the configured HippoRAG interpreter"
        )
    parent = _absolute_without_symlink(lexical.parent, "runtime Python parent")
    try:
        target = lexical.resolve(strict=True)
    except OSError as exc:
        raise TatqaP19RuntimeAttestationError(
            "runtime Python target is unavailable"
        ) from exc
    if not target.is_file() or not os.access(lexical, os.X_OK):
        raise TatqaP19RuntimeAttestationError("runtime Python is not executable")
    target_hash = file_sha256(target)
    if target_hash != EXPECTED_RUNTIME_PYTHON_TARGET_SHA256:
        raise TatqaP19RuntimeAttestationError("runtime Python target drifted")
    venv_root = parent.parent
    if Path(sys.prefix).expanduser().absolute() != venv_root:
        raise TatqaP19RuntimeAttestationError("active sys.prefix is not the lexical venv")
    pyvenv = venv_root / "pyvenv.cfg"
    if pyvenv.is_symlink() or not pyvenv.is_file():
        raise TatqaP19RuntimeAttestationError("pyvenv.cfg is unavailable")
    pyvenv_hash = file_sha256(pyvenv)
    if pyvenv_hash != EXPECTED_PYVENV_CFG_SHA256:
        raise TatqaP19RuntimeAttestationError("P17 pyvenv.cfg drifted")
    overlay = venv_root / "lib" / "python3.10" / "site-packages"
    if overlay.is_symlink() or not overlay.is_dir():
        raise TatqaP19RuntimeAttestationError("runtime overlay is unavailable")
    pth_rows = []
    for path in sorted(overlay.glob("*.pth"), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file():
            raise TatqaP19RuntimeAttestationError("runtime .pth topology drifted")
        pth_rows.append(
            {
                "name": path.name,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    if not pth_rows:
        raise TatqaP19RuntimeAttestationError("runtime .pth binding is absent")
    existing_site_roots = sorted(
        {
            str(Path(value).expanduser().absolute())
            for value in site.getsitepackages()
            if Path(value).expanduser().absolute().is_dir()
        }
    )
    return {
        "active_sys_executable": str(active),
        "active_sys_prefix": str(Path(sys.prefix).expanduser().absolute()),
        "active_sys_base_prefix": str(
            Path(sys.base_prefix).expanduser().absolute()
        ),
        "existing_site_package_roots": existing_site_roots,
        "implementation": platform.python_implementation(),
        "lexical_path": str(lexical),
        "lexical_path_sha256": hashlib.sha256(
            str(lexical).encode("utf-8")
        ).hexdigest(),
        "pth_rows": pth_rows,
        "pth_set_sha256": stable_hash(pth_rows),
        "python_version": platform.python_version(),
        "pyvenv_cfg_sha256": pyvenv_hash,
        "resolved_path": str(target),
        "resolved_target_sha256": target_hash,
        "resolved_target_size_bytes": target.stat().st_size,
        "samefile_with_active_sys_executable": os.path.samefile(lexical, active),
    }


def _load_qasper_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    path = _absolute_without_symlink(path, "QASPER MiniLM manifest")
    if not path.is_file() or path.stat().st_size > 256 * 1024:
        raise TatqaP19RuntimeAttestationError(
            "QASPER MiniLM manifest is unavailable"
        )
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_MINILM_MANIFEST_FILE_SHA256:
        raise TatqaP19RuntimeAttestationError("QASPER MiniLM manifest file drifted")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TatqaP19RuntimeAttestationError(
            "QASPER MiniLM manifest is invalid"
        ) from exc
    if not isinstance(value, dict):
        raise TatqaP19RuntimeAttestationError(
            "QASPER MiniLM manifest root drifted"
        )
    body = dict(value)
    declared = body.pop("asset_sha256", None)
    if (
        declared != EXPECTED_MINILM_MANIFEST_SELF_SHA256
        or stable_hash(body) != declared
    ):
        raise TatqaP19RuntimeAttestationError(
            "QASPER MiniLM manifest self hash drifted"
        )
    return value, raw


def _qasper_minilm_binding(manifest_path: Path, model_root: Path) -> dict[str, object]:
    manifest, raw = _load_qasper_manifest(manifest_path)
    local = manifest.get("local_binding")
    model = manifest.get("model")
    if not isinstance(local, Mapping) or not isinstance(model, Mapping):
        raise TatqaP19RuntimeAttestationError(
            "QASPER MiniLM manifest binding is incomplete"
        )
    expected_rows = local.get("snapshot_files")
    if not isinstance(expected_rows, list) or not expected_rows:
        raise TatqaP19RuntimeAttestationError(
            "QASPER MiniLM file binding is absent"
        )
    observed_generic = _tree_rows(model_root, "MiniLM model")
    observed_normative = [
        {
            "path": row["path"],
            "sha256": row["sha256"],
            "size": row["size_bytes"],
        }
        for row in observed_generic
    ]
    if (
        observed_normative != expected_rows
        or stable_hash(observed_normative)
        != EXPECTED_MINILM_NORMATIVE_TREE_SHA256
        or local.get("snapshot_tree_sha256")
        != EXPECTED_MINILM_NORMATIVE_TREE_SHA256
        or model.get("weights_sha256") != EXPECTED_MINILM_WEIGHTS_SHA256
        or model.get("model_id") != MINILM_REPO_ID
        or model.get("snapshot_revision") != MINILM_REVISION
    ):
        raise TatqaP19RuntimeAttestationError(
            "QASPER MiniLM normative model tree drifted"
        )
    generic = {
        "file_count": len(observed_generic),
        "size_bytes": sum(int(row["size_bytes"]) for row in observed_generic),
        "tree_sha256": stable_hash(observed_generic),
    }
    if generic != EXPECTED_MINILM_GENERIC_TREE:
        raise TatqaP19RuntimeAttestationError("MiniLM generic tree drifted")
    return {
        "asset_manifest_file_sha256": hashlib.sha256(raw).hexdigest(),
        "asset_manifest_self_sha256": EXPECTED_MINILM_MANIFEST_SELF_SHA256,
        "generic_tree": generic,
        "model_id": MINILM_REPO_ID,
        "normative_tree_sha256": EXPECTED_MINILM_NORMATIVE_TREE_SHA256,
        "revision": MINILM_REVISION,
        "role": "HippoRAG_embedding_asset_and_QASPER_controller_model_asset",
        "weights_sha256": EXPECTED_MINILM_WEIGHTS_SHA256,
    }


def _smollm_binding(root: Path) -> dict[str, object]:
    tree = _require_tree(root, "SmolLM model", EXPECTED_SMOLLM_TREE)
    config = _absolute_without_symlink(root, "SmolLM model") / "config.json"
    try:
        value = json.loads(config.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TatqaP19RuntimeAttestationError("SmolLM config is invalid") from exc
    if (
        not isinstance(value, Mapping)
        or value.get("architectures") != ["LlamaForCausalLM"]
        or value.get("model_type") != "llama"
        or value.get("max_position_embeddings") != 8192
        or value.get("hidden_size") != 576
        or value.get("num_hidden_layers") != 30
        or value.get("vocab_size") != 49_152
    ):
        raise TatqaP19RuntimeAttestationError("SmolLM config identity drifted")
    return {
        "config_file_sha256": file_sha256(config),
        "model_id": SMOLLM_REPO_ID,
        "revision": SMOLLM_REVISION,
        "role": "HippoRAG_local_openie_llm",
        "tree": tree,
    }


def _source_binding(root: Path) -> dict[str, object]:
    tree = _require_tree(root, "HippoRAG source", EXPECTED_SOURCE_TREE)
    rows = _tree_rows(root, "HippoRAG source")
    pyc_count = sum(str(row["path"]).endswith(".pyc") for row in rows)
    if pyc_count < 1:
        raise TatqaP19RuntimeAttestationError(
            "the exact P17 source tree no longer includes its bound bytecode"
        )
    return {
        "contains_same_host_nonportable_bytecode": True,
        "nonportable_bytecode_file_count": pyc_count,
        "root_role": "direct_hardened_P17_runtime_source",
        "tree": tree,
    }


def _implementation_binding() -> dict[str, object]:
    path = Path(__file__).absolute()
    if path.is_symlink() or not path.is_file():
        raise TatqaP19RuntimeAttestationError(
            "attestation builder implementation is unavailable"
        )
    return {
        "module_file_sha256": file_sha256(path),
        "module_size_bytes": path.stat().st_size,
        "schema": IMPLEMENTATION_SCHEMA,
    }


def _compatibility_decision() -> dict[str, object]:
    active = {
        name: EXPECTED_ACTIVE_DISTRIBUTION_VERSIONS[name]
        for name in QASPER_CONTROLLER_REQUIRED_VERSIONS
    }
    mismatch_rows = [
        {
            "active_version": active[name],
            "name": name,
            "required_version": QASPER_CONTROLLER_REQUIRED_VERSIONS[name],
        }
        for name in QASPER_CONTROLLER_REQUIRED_VERSIONS
        if active[name] != QASPER_CONTROLLER_REQUIRED_VERSIONS[name]
    ]
    if not mismatch_rows:
        raise TatqaP19RuntimeAttestationError(
            "P17 HippoRAG runtime unexpectedly became the QASPER controller runtime"
        )
    return {
        "P17_HippoRAG_interpreter_is_exact_QASPER_controller_runtime": False,
        "decision": "reuse_for_HippoRAG_only_and_require_a_separate_exact_QASPER_controller_interpreter",
        "mismatch_rows": mismatch_rows,
        "qasper_required_versions": dict(QASPER_CONTROLLER_REQUIRED_VERSIONS),
    }


def build_runtime_attestation(
    *,
    runtime_python: str | Path,
    hipporag_source: str | Path,
    smollm_model: str | Path,
    minilm_model: str | Path,
    minilm_manifest: str | Path,
) -> dict[str, Any]:
    """Build one exact, source-free receipt under the active interpreter."""

    source = Path(hipporag_source).expanduser().absolute()
    body: dict[str, Any] = {
        "active_distribution_binding": _active_distribution_binding(),
        "active_module_binding": _active_module_binding(source),
        "asset_bindings": {
            "HippoRAG_source": _source_binding(source),
            "MiniLM": _qasper_minilm_binding(
                Path(minilm_manifest), Path(minilm_model)
            ),
            "SmolLM": _smollm_binding(Path(smollm_model)),
        },
        "compatibility_decision": _compatibility_decision(),
        "implementation_binding": _implementation_binding(),
        "runtime_python_binding": _runtime_python_binding(Path(runtime_python)),
        "schema": SCHEMA,
        "source_free_scope": {
            "api_or_online_evaluator_calls": 0,
            "environment_variable_names_or_values_recorded": False,
            "external_network_calls": 0,
            "formal_TAT_QA_source_or_rows_accessed": False,
            "model_inference_calls": 0,
            "provider_or_API_credentials_read": False,
        },
        "status": "verified_P17_HippoRAG_runtime_assets_before_P19_formal_source_open",
        "topology_decision": {
            "inherits_or_claims_MuSiQue_v3_topology": False,
            "old_MuSiQue_v3_official_source_tree_used": False,
            "source_identity": "complete_a644_hardened_P17_tree",
        },
    }
    return {**body, "receipt_sha256": stable_hash(body)}


_TOP_LEVEL_KEYS = frozenset(
    {
        "active_distribution_binding",
        "active_module_binding",
        "asset_bindings",
        "compatibility_decision",
        "implementation_binding",
        "receipt_sha256",
        "runtime_python_binding",
        "schema",
        "source_free_scope",
        "status",
        "topology_decision",
    }
)


def validate_receipt_structure(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != _TOP_LEVEL_KEYS or value.get("schema") != SCHEMA:
        raise TatqaP19RuntimeAttestationError("attestation key set drifted")
    body = dict(value)
    declared = body.pop("receipt_sha256", None)
    if (
        not isinstance(declared, str)
        or _HEX64.fullmatch(declared) is None
        or stable_hash(body) != declared
    ):
        raise TatqaP19RuntimeAttestationError("attestation self hash drifted")
    scope = value.get("source_free_scope")
    topology = value.get("topology_decision")
    compatibility = value.get("compatibility_decision")
    assets = value.get("asset_bindings")
    if (
        not isinstance(scope, Mapping)
        or scope.get("formal_TAT_QA_source_or_rows_accessed") is not False
        or scope.get("external_network_calls") != 0
        or scope.get("api_or_online_evaluator_calls") != 0
        or not isinstance(topology, Mapping)
        or topology.get("inherits_or_claims_MuSiQue_v3_topology") is not False
        or topology.get("source_identity")
        != "complete_a644_hardened_P17_tree"
        or not isinstance(compatibility, Mapping)
        or compatibility.get(
            "P17_HippoRAG_interpreter_is_exact_QASPER_controller_runtime"
        )
        is not False
        or not isinstance(assets, Mapping)
        or set(assets) != {"HippoRAG_source", "MiniLM", "SmolLM"}
    ):
        raise TatqaP19RuntimeAttestationError(
            "attestation normative decision drifted"
        )
    return dict(value)


def load_attestation(path: str | Path) -> dict[str, Any]:
    receipt_path = _absolute_without_symlink(Path(path), "attestation receipt")
    if not receipt_path.is_file() or receipt_path.stat().st_size > 2 * 1024 * 1024:
        raise TatqaP19RuntimeAttestationError("attestation receipt is unavailable")
    raw = receipt_path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TatqaP19RuntimeAttestationError("attestation receipt is invalid") from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise TatqaP19RuntimeAttestationError(
            "attestation receipt is not canonical JSON"
        )
    return validate_receipt_structure(value)


def verify_live_attestation(
    *,
    receipt_path: str | Path,
    runtime_python: str | Path,
    hipporag_source: str | Path,
    smollm_model: str | Path,
    minilm_model: str | Path,
    minilm_manifest: str | Path,
) -> dict[str, Any]:
    persisted = load_attestation(receipt_path)
    live = build_runtime_attestation(
        runtime_python=runtime_python,
        hipporag_source=hipporag_source,
        smollm_model=smollm_model,
        minilm_model=minilm_model,
        minilm_manifest=minilm_manifest,
    )
    if persisted != live:
        raise TatqaP19RuntimeAttestationError("live P19 runtime drifted")
    return persisted


def write_attestation_exclusive(path: str | Path, value: Mapping[str, Any]) -> str:
    validate_receipt_structure(value)
    output = Path(path).expanduser().absolute()
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    raw = canonical_json_bytes(dict(value))
    try:
        descriptor = os.open(
            output,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise TatqaP19RuntimeAttestationError(
            "exclusive attestation output is already consumed"
        ) from exc
    if (
        output.is_symlink()
        or output.read_bytes() != raw
        or stat.S_IMODE(output.stat().st_mode) != 0o600
    ):
        raise TatqaP19RuntimeAttestationError(
            "attestation output reopen or mode drifted"
        )
    return hashlib.sha256(raw).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--hipporag-source", type=Path, required=True)
    parser.add_argument("--smollm-model", type=Path, required=True)
    parser.add_argument("--minilm-model", type=Path, required=True)
    parser.add_argument("--minilm-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    receipt = build_runtime_attestation(
        runtime_python=arguments.runtime_python,
        hipporag_source=arguments.hipporag_source,
        smollm_model=arguments.smollm_model,
        minilm_model=arguments.minilm_model,
        minilm_manifest=arguments.minilm_manifest,
    )
    file_hash = write_attestation_exclusive(arguments.output, receipt)
    print(
        json.dumps(
            {
                "file_sha256": file_hash,
                "receipt_sha256": receipt["receipt_sha256"],
                "schema": SCHEMA,
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_ACTIVE_DISTRIBUTION_VERSIONS",
    "EXPECTED_MINILM_GENERIC_TREE",
    "EXPECTED_SMOLLM_TREE",
    "EXPECTED_SOURCE_TREE",
    "QASPER_CONTROLLER_REQUIRED_VERSIONS",
    "SCHEMA",
    "TatqaP19RuntimeAttestationError",
    "build_runtime_attestation",
    "canonical_json_bytes",
    "file_sha256",
    "load_attestation",
    "main",
    "stable_hash",
    "tree_receipt",
    "validate_receipt_structure",
    "verify_live_attestation",
    "write_attestation_exclusive",
]
