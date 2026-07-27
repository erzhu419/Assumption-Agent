"""P17 reused-closure binding for the DSTC9 official HippoRAG arm.

This binding intentionally does not inherit the MuSiQue v3 topology.  It
binds the corrected P17 fingerprint, its exact interpreter target, the
P17-local MiniLM and SmolLM trees, and the repaired hardened HippoRAG source.
It does not reuse the old P17 host driver or kernel claim.  A separate
current-study hardware receipt is captured immediately before source-free
canary and rechecked by the worker.  The worker separately recomputes the P17
dependency inventory and base ``sys.path`` after removing the one exact
formal-code prefix, before importing HippoRAG or any model package.
"""

from __future__ import annotations

from importlib import metadata, util
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


SCHEMA = "dstc9_p17_reused_closure_plus_current_hardware_binding_v3"
CURRENT_HARDWARE_SCHEMA = "dstc9_current_study_hardware_binding_v1"
CURRENT_HARDWARE_STATUS = (
    "captured_immediately_before_source_free_canary_before_formal_source_open"
)
FINGERPRINT_SCHEMA = "bright_p17_remote_runtime_fingerprint_v1"
FINGERPRINT_STATUS = (
    "corrected_after_P17_item_identity_staging_before_any_model_or_"
    "comparator_action"
)

P17_REMOTE_ROOT = Path("/home/erzhu419/p17_all_remote_20260722")
P17_PROJECT_ROOT = P17_REMOTE_ROOT / "runtime/reconstruction_v2"
P17_FINGERPRINT_PATH = (
    P17_PROJECT_ROOT
    / "manifests/bright_p17_remote_runtime_fingerprint_v1.json"
)
P17_RUNTIME_PYTHON = (
    P17_PROJECT_ROOT
    / "artifacts/bright_reasoning_retrieval_runtime_v1/"
    "hipporag_venv/bin/python"
)
P17_VENV_ROOT = P17_RUNTIME_PYTHON.parent.parent
P17_VENV_SITE_PACKAGES = (
    P17_VENV_ROOT / "lib/python3.10/site-packages"
)
P17_SMOLLM_ROOT = (
    P17_PROJECT_ROOT
    / "artifacts/bright_reasoning_retrieval_runtime_v1/"
    "smollm2_135m_instruct_exact"
)
P17_MINILM_ROOT = (
    P17_PROJECT_ROOT / "artifacts/qasper_minilm_runtime_v1/model"
)
P17_MINILM_MANIFEST = (
    P17_PROJECT_ROOT / "manifests/qasper_minilm_runtime_asset_v1.json"
)
P17_P16_SITE = P17_REMOTE_ROOT / "runtime/p16_site"
P17_HIPPORAG_SOURCE = (
    P17_PROJECT_ROOT
    / "reference/self_evo_continual_20260707/repos/HippoRAG/src"
)
P17_HIPPORAG_INIT = P17_HIPPORAG_SOURCE / "hipporag/__init__.py"
P17_REPAIRED_SOURCE_FILE = P17_HIPPORAG_SOURCE / "hipporag/HippoRAG.py"

FINGERPRINT_FILE_SHA256 = (
    "00fb1dff9cac33caeea289b92f29b0f5035814d3a555be5933d9150d6fc500d2"
)
FINGERPRINT_SELF_SHA256 = (
    "cab5f0fed49b9a68ef9caec36dadf9c2b4e5eee00506685798e77cf0cfc9fb1e"
)
RUNTIME_PYTHON_TARGET_SHA256 = (
    "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"
)
RUNTIME_PYTHON_TARGET_SIZE = 5_917_224
REPAIRED_SOURCE_FILE_SHA256 = (
    "960561b080531fe4d668bde635e81f8e65620ce50bdacdd9a25531e856fa3e05"
)
HIPPORAG_INIT_FILE_SHA256 = (
    "f6d25d8b36bf5eb8a9dac0a52de3262d160dce008e62c13a474040835eb1ca94"
)
MINILM_MANIFEST_FILE_SHA256 = (
    "62b85c7752f2e46932fb9fb13ae2f3aac9eb750a33c8f07102739040feb6cc75"
)
MINILM_MANIFEST_SELF_SHA256 = (
    "921d9b1945581130e03c53f448092c3de3b30714431c6cac9b3b32c2ec10abad"
)
MINILM_NORMATIVE_TREE_SHA256 = (
    "1514beb65d2d3a2824a93f133a300cc60d5b437ccd6ea1e622eb4cd9881dcfdb"
)
MINILM_WEIGHTS_SHA256 = (
    "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
)

EXPECTED_ASSET_TREES: dict[str, dict[str, object]] = {
    "HippoRAG_LLM": {
        "file_count": 23,
        "size_bytes": 272_031_142,
        "tree_sha256": (
            "d626d755c99c006761d5e069aa85a73fe8b011c6c0f5d0323a6f8de85246bcb5"
        ),
    },
    "HippoRAG_source": {
        "file_count": 96,
        "size_bytes": 527_994,
        "tree_sha256": (
            "a644ab2811db2739db3cfbdc051561e2cfdf2ed87286f8ebd00a5971d189cdd5"
        ),
    },
    "MiniLM": {
        "file_count": 11,
        "size_bytes": 91_578_415,
        "tree_sha256": (
            "42d8d798e4f01e68d9bb10634b9c712de00f7f8495271636fd6311b2db58e506"
        ),
    },
}
HIPPORAG_SOURCE_NORMATIVE_TREE: dict[str, object] = {
    "file_count": 60,
    "size_bytes": 332_110,
    # Canonical order is the relative POSIX-path string order emitted by
    # _tree_rows(), not pathlib's component-wise Path ordering.
    "tree_sha256": (
        "342505c3aaa8dc5e57718e8ac695ac28f60aa66837ba717f52d6f7b536527b1f"
    ),
}
EXPECTED_RUNTIME_INVENTORY: dict[str, object] = {
    "distribution_count": 169,
    "inventory_sha256": (
        "b17aaaa29c46ff5dd8d2c8e8174f13730cd3ff8268122117ba12fa283ce925c6"
    ),
    "python_executable": str(P17_RUNTIME_PYTHON),
    "python_version": "3.10.12",
    "sys_path_sha256": (
        "0ba3189fd5968f8aa8b1df6f4a5b2d7815ee6e30698d555f011390cb65148f42"
    ),
}
EXPECTED_ACTIVE_DISTRIBUTIONS: dict[str, str | None] = {
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
EXPECTED_MINILM_RUNTIME_VERSIONS = {
    "huggingface_hub": "0.25.2",
    "numpy": "2.1.3",
    "python": "3.10.12",
    "safetensors": "0.4.5",
    "sentence_transformers": "3.1.1",
    "tokenizers": "0.20.3",
    "torch": "2.4.1+cu118",
    "transformers": "4.45.2",
}
EXPECTED_PTH_ROWS = [
    {
        "name": "__editable__.hipporag-2.0.0a4.pth",
        "sha256": (
            "ddaf3bc7516b670a2acf0c643c5160dc956fccec4ba165f3cf2242f10c308998"
        ),
        "size_bytes": 122,
    },
    {
        "name": "_virtualenv.pth",
        "sha256": (
            "69ac3d8f27e679c81b94ab30b3b56e9cd138219b1ba94a1fa3606d5a76a1433d"
        ),
        "size_bytes": 18,
    },
    {
        "name": "bright_project_runtime.pth",
        "sha256": (
            "f4b77c4a91e8a13cad136a1dcc8431fdac0fb5056937dc5ab1e3490df12a80ab"
        ),
        "size_bytes": 112,
    },
    {
        "name": "bright_user_overlay.pth",
        "sha256": (
            "d81126255164784da574dcb7fcb5cd5f27660e3b29e7c8fcdac172eb7e7e09a7"
        ),
        "size_bytes": 56,
    },
    {
        "name": "distutils-precedence.pth",
        "sha256": (
            "2638ce9e2500e572a5e0de7faed6661eb569d1b696fcba07b0dd223da5f5d224"
        ),
        "size_bytes": 151,
    },
]
EXPECTED_PTH_SET_SHA256 = (
    "702748767895a8315d00959f3c55443a51a8c37f0e9e0c61ccf3fdb436ca68ef"
)
WORKER_CODE_RELATIVE_FILES = (
    "replication_runtime/__init__.py",
    "replication_runtime/dstc9_official_hipporag_v1/__init__.py",
    "replication_runtime/dstc9_official_hipporag_v1/adapter.py",
    "replication_runtime/dstc9_official_hipporag_v1/contract.py",
    "replication_runtime/dstc9_official_hipporag_v1/runtime_binding.py",
    "replication_runtime/dstc9_official_hipporag_v1/worker.py",
)
EXPECTED_GPU_ROWS = [
    {
        "UUID": "GPU-32d6e292-70cd-50a0-405b-e344d2da8d39",
        "index": 0,
        "memory_total_MiB": 8192,
        "name": "NVIDIA GeForce RTX 2080",
    },
    {
        "UUID": "GPU-db2137c8-0f6b-b790-a698-6bfbbd5dc9eb",
        "index": 1,
        "memory_total_MiB": 8192,
        "name": "NVIDIA GeForce RTX 2080",
    },
]

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class Dstc9P17RuntimeBindingError(RuntimeError):
    """The P17 closure/current hardware binding failed closed."""


def canonical_json_bytes(
    value: object,
    *,
    newline: bool = True,
) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise Dstc9P17RuntimeBindingError(
            "runtime binding value is not canonical JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(
        canonical_json_bytes(value, newline=False)
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise Dstc9P17RuntimeBindingError(
            "bound runtime file is unreadable"
        ) from exc
    return digest.hexdigest()


def _absolute_without_symlink(
    path: Path,
    field_name: str,
    *,
    allow_final_symlink: bool = False,
) -> Path:
    absolute = path.expanduser().absolute()
    cursor = Path(absolute.anchor)
    for index, component in enumerate(absolute.parts[1:]):
        cursor = cursor / component
        final = index == len(absolute.parts[1:]) - 1
        if cursor.is_symlink() and not (allow_final_symlink and final):
            raise Dstc9P17RuntimeBindingError(
                f"{field_name} contains a symlink component"
            )
    return absolute


def _load_canonical_json(
    path: Path,
    field_name: str,
) -> tuple[dict[str, Any], bytes]:
    path = _absolute_without_symlink(path, field_name)
    if path.is_symlink() or not path.is_file():
        raise Dstc9P17RuntimeBindingError(
            f"{field_name} is unavailable"
        )
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Dstc9P17RuntimeBindingError(
            f"{field_name} is invalid"
        ) from exc
    if (
        not isinstance(value, dict)
        or raw != canonical_json_bytes(value)
    ):
        raise Dstc9P17RuntimeBindingError(
            f"{field_name} is not canonical JSON"
        )
    return value, raw


def _verify_self_hash(
    value: Mapping[str, Any],
    *,
    field_name: str,
    self_field: str,
) -> str:
    body = dict(value)
    declared = body.pop(self_field, None)
    if (
        not isinstance(declared, str)
        or _HEX64.fullmatch(declared) is None
        or not hmac.compare_digest(stable_hash(body), declared)
    ):
        raise Dstc9P17RuntimeBindingError(
            f"{field_name} self hash drifted"
        )
    return declared


def _tree_rows(root: Path, field_name: str) -> list[dict[str, object]]:
    root = _absolute_without_symlink(root, field_name)
    if not root.is_dir():
        raise Dstc9P17RuntimeBindingError(
            f"{field_name} is unavailable"
        )
    rows: list[dict[str, object]] = []
    for current, directories, files in os.walk(root, followlinks=False):
        base = Path(current)
        for name in directories:
            if (base / name).is_symlink():
                raise Dstc9P17RuntimeBindingError(
                    f"{field_name} contains a directory symlink"
                )
        for name in files:
            path = base / name
            if path.is_symlink() or not path.is_file():
                raise Dstc9P17RuntimeBindingError(
                    f"{field_name} contains a non-regular file"
                )
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    rows.sort(key=lambda row: str(row["path"]))
    return rows


def tree_receipt(root: Path, field_name: str) -> dict[str, object]:
    rows = _tree_rows(root, field_name)
    return {
        "file_count": len(rows),
        "size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "tree_sha256": stable_hash(rows),
    }


def _hipporag_source_tree_receipt() -> dict[str, object]:
    """Bind source files while treating local bytecode as diagnostics only."""

    rows = _tree_rows(
        P17_HIPPORAG_SOURCE,
        "repaired P17 HippoRAG source",
    )
    normative_rows: list[dict[str, object]] = []
    bytecode_rows: list[dict[str, object]] = []
    for row in rows:
        relative = Path(str(row["path"]))
        in_pycache = "__pycache__" in relative.parts
        is_pyc = relative.suffix == ".pyc"
        if not in_pycache and not is_pyc:
            normative_rows.append(row)
        elif in_pycache and is_pyc:
            bytecode_rows.append(row)
        else:
            raise Dstc9P17RuntimeBindingError(
                "P17 HippoRAG source contains non-normative "
                "non-cache content"
            )
    normative = {
        "file_count": len(normative_rows),
        "size_bytes": sum(
            int(row["size_bytes"]) for row in normative_rows
        ),
        "tree_sha256": stable_hash(normative_rows),
    }
    if normative != HIPPORAG_SOURCE_NORMATIVE_TREE:
        raise Dstc9P17RuntimeBindingError(
            "repaired P17 HippoRAG normative source tree drifted"
        )
    diagnostic = {
        "acceptance_role": "diagnostic_only_not_live_identity",
        "allowed_shape": "only___pycache___descendant_dot_pyc_v1",
        "file_count": len(bytecode_rows),
        "size_bytes": sum(
            int(row["size_bytes"]) for row in bytecode_rows
        ),
        "tree_sha256": stable_hash(bytecode_rows),
    }
    return {
        "historical_P17_aggregate_lineage": {
            "acceptance_role": "lineage_only_not_live_identity",
            **EXPECTED_ASSET_TREES["HippoRAG_source"],
        },
        "normative_tree": normative,
        "source_local_bytecode_diagnostic": diagnostic,
    }


def runtime_binding_acceptance_identity(
    value: Mapping[str, Any],
) -> str:
    """Return identity with the source-local bytecode diagnostic removed."""

    if not isinstance(value, Mapping):
        raise Dstc9P17RuntimeBindingError(
            "runtime binding receipt is invalid"
        )
    declared = _verify_self_hash(
        value,
        field_name="runtime binding receipt",
        self_field="self_sha256",
    )
    if value.get("schema") != SCHEMA or declared != value.get("self_sha256"):
        raise Dstc9P17RuntimeBindingError(
            "runtime binding receipt identity drifted"
        )
    body = dict(value)
    body.pop("self_sha256")
    assets = body.get("assets")
    if not isinstance(assets, Mapping):
        raise Dstc9P17RuntimeBindingError(
            "runtime binding assets are invalid"
        )
    normalized_assets = dict(assets)
    source = normalized_assets.get("HippoRAG_source")
    if not isinstance(source, Mapping):
        raise Dstc9P17RuntimeBindingError(
            "runtime binding HippoRAG source is invalid"
        )
    normalized_source = dict(source)
    diagnostic = normalized_source.pop(
        "source_local_bytecode_diagnostic",
        None,
    )
    if (
        not isinstance(diagnostic, Mapping)
        or set(diagnostic)
        != {
            "acceptance_role",
            "allowed_shape",
            "file_count",
            "size_bytes",
            "tree_sha256",
        }
        or diagnostic.get("acceptance_role")
        != "diagnostic_only_not_live_identity"
        or diagnostic.get("allowed_shape")
        != "only___pycache___descendant_dot_pyc_v1"
        or type(diagnostic.get("file_count")) is not int
        or int(diagnostic["file_count"]) < 0
        or type(diagnostic.get("size_bytes")) is not int
        or int(diagnostic["size_bytes"]) < 0
        or not isinstance(diagnostic.get("tree_sha256"), str)
        or _HEX64.fullmatch(str(diagnostic["tree_sha256"])) is None
    ):
        raise Dstc9P17RuntimeBindingError(
            "runtime binding bytecode diagnostic is invalid"
        )
    normalized_assets["HippoRAG_source"] = normalized_source
    body["assets"] = normalized_assets
    return stable_hash(body)


def _load_exact_fingerprint(path: Path) -> dict[str, Any]:
    expected_path = _absolute_without_symlink(
        P17_FINGERPRINT_PATH,
        "P17 fingerprint path",
    )
    actual_path = _absolute_without_symlink(
        path,
        "runtime fingerprint",
    )
    if actual_path != expected_path:
        raise Dstc9P17RuntimeBindingError(
            "runtime fingerprint path drifted"
        )
    value, raw = _load_canonical_json(actual_path, "runtime fingerprint")
    if hashlib.sha256(raw).hexdigest() != FINGERPRINT_FILE_SHA256:
        raise Dstc9P17RuntimeBindingError(
            "runtime fingerprint file hash drifted"
        )
    declared = _verify_self_hash(
        value,
        field_name="runtime fingerprint",
        self_field="self_sha256",
    )
    if (
        declared != FINGERPRINT_SELF_SHA256
        or value.get("schema") != FINGERPRINT_SCHEMA
        or value.get("status") != FINGERPRINT_STATUS
        or value.get("remote_root") != str(P17_REMOTE_ROOT)
    ):
        raise Dstc9P17RuntimeBindingError(
            "runtime fingerprint identity drifted"
        )
    return value


def make_current_study_hardware_binding(
    *,
    study_id: str,
    capture_id: str,
    gpus: Sequence[Mapping[str, object]],
    nvidia_driver_version: str,
    kernel_release: str,
) -> dict[str, Any]:
    """Create the hardware receipt immediately before source-free canary."""

    normalized_gpus = [dict(row) for row in gpus]
    if normalized_gpus != EXPECTED_GPU_ROWS:
        raise Dstc9P17RuntimeBindingError(
            "current study GPU identity drifted"
        )
    if (
        not isinstance(nvidia_driver_version, str)
        or not re.fullmatch(r"[0-9][0-9.]{1,31}", nvidia_driver_version)
        or not isinstance(kernel_release, str)
        or not 1 <= len(kernel_release) <= 256
        or any(character in kernel_release for character in "\x00\r\n")
        or not isinstance(study_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,255}", study_id)
        is None
        or not isinstance(capture_id, str)
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,255}", capture_id
        )
        is None
    ):
        raise Dstc9P17RuntimeBindingError(
            "current study hardware receipt input is invalid"
        )
    body: dict[str, Any] = {
        "capture_id": capture_id,
        "hardware": {
            "GPUs": normalized_gpus,
            "NVIDIA_driver_version": nvidia_driver_version,
            "kernel_release": kernel_release,
        },
        "schema": CURRENT_HARDWARE_SCHEMA,
        "source_free_boundary": {
            "capture_scope": (
                "hardware_only_no_model_source_or_evaluator_action_v1"
            ),
            "external_network_call_count": 0,
            "formal_source_open_count": 0,
            "old_P17_driver_or_kernel_used_as_requirement": False,
        },
        "status": CURRENT_HARDWARE_STATUS,
        "study_id": study_id,
    }
    return {**body, "self_sha256": stable_hash(body)}


def capture_current_study_hardware_binding(
    *,
    study_id: str,
    capture_id: str,
) -> dict[str, Any]:
    """Source-free hook run immediately before the canary."""

    hardware = _probe_current_hardware()
    gpus = hardware["GPUs"]
    if not isinstance(gpus, list):
        raise Dstc9P17RuntimeBindingError(
            "current hardware capture is malformed"
        )
    return make_current_study_hardware_binding(
        study_id=study_id,
        capture_id=capture_id,
        gpus=gpus,
        nvidia_driver_version=str(
            hardware["NVIDIA_driver_version"]
        ),
        kernel_release=str(hardware["kernel_release"]),
    )


def _load_current_hardware_binding(
    *,
    path: Path,
    worker_project_root: Path,
    expected_study_id: str,
) -> dict[str, object]:
    actual = _absolute_without_symlink(
        path,
        "current study hardware binding",
    )
    deployment_root = _absolute_without_symlink(
        worker_project_root,
        "DSTC9 worker project root",
    ).parent
    if (
        not actual.is_relative_to(deployment_root)
        or actual.is_relative_to(P17_REMOTE_ROOT)
    ):
        raise Dstc9P17RuntimeBindingError(
            "current study hardware binding path drifted"
        )
    value, raw = _load_canonical_json(
        actual,
        "current study hardware binding",
    )
    declared = _verify_self_hash(
        value,
        field_name="current study hardware binding",
        self_field="self_sha256",
    )
    boundary = value.get("source_free_boundary")
    hardware = value.get("hardware")
    if (
        set(value)
        != {
            "capture_id",
            "hardware",
            "schema",
            "self_sha256",
            "source_free_boundary",
            "status",
            "study_id",
        }
        or value.get("schema") != CURRENT_HARDWARE_SCHEMA
        or value.get("status") != CURRENT_HARDWARE_STATUS
        or not isinstance(boundary, Mapping)
        or set(boundary)
        != {
            "capture_scope",
            "external_network_call_count",
            "formal_source_open_count",
            "old_P17_driver_or_kernel_used_as_requirement",
        }
        or boundary.get("capture_scope")
        != "hardware_only_no_model_source_or_evaluator_action_v1"
        or boundary.get("external_network_call_count") != 0
        or boundary.get("formal_source_open_count") != 0
        or boundary.get(
            "old_P17_driver_or_kernel_used_as_requirement"
        )
        is not False
        or not isinstance(value.get("study_id"), str)
        or value.get("study_id") != expected_study_id
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,255}",
            str(value.get("study_id")),
        )
        is None
        or not isinstance(value.get("capture_id"), str)
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,255}",
            str(value.get("capture_id")),
        )
        is None
        or not isinstance(hardware, Mapping)
        or set(hardware)
        != {"GPUs", "NVIDIA_driver_version", "kernel_release"}
        or hardware.get("GPUs") != EXPECTED_GPU_ROWS
        or not isinstance(hardware.get("NVIDIA_driver_version"), str)
        or re.fullmatch(
            r"[0-9][0-9.]{1,31}",
            str(hardware.get("NVIDIA_driver_version")),
        )
        is None
        or not isinstance(hardware.get("kernel_release"), str)
        or not 1 <= len(str(hardware.get("kernel_release"))) <= 256
        or any(
            character in str(hardware.get("kernel_release"))
            for character in "\x00\r\n"
        )
    ):
        raise Dstc9P17RuntimeBindingError(
            "current study hardware binding drifted"
        )
    return {
        "capture_id": value["capture_id"],
        "hardware": dict(hardware),
        "receipt_file_sha256": hashlib.sha256(raw).hexdigest(),
        "receipt_relative_path": str(actual.relative_to(deployment_root)),
        "receipt_self_sha256": declared,
        "study_id": value["study_id"],
    }


def _probe_current_hardware() -> dict[str, object]:
    nvidia_smi = Path("/usr/bin/nvidia-smi")
    if not nvidia_smi.is_file():
        raise Dstc9P17RuntimeBindingError(
            "current NVIDIA hardware probe is unavailable"
        )
    try:
        completed = subprocess.run(
            [
                str(nvidia_smi),
                (
                    "--query-gpu=index,name,uuid,memory.total,"
                    "driver_version"
                ),
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            env={"LANG": "C.UTF-8", "PATH": "/usr/bin:/bin"},
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Dstc9P17RuntimeBindingError(
            "current NVIDIA hardware probe failed"
        ) from exc
    if completed.returncode != 0:
        raise Dstc9P17RuntimeBindingError(
            "current NVIDIA hardware probe failed"
        )
    rows: list[dict[str, object]] = []
    driver: str | None = None
    try:
        for line in completed.stdout.splitlines():
            index, name, uuid, memory, observed_driver = [
                part.strip() for part in line.split(",")
            ]
            if driver is None:
                driver = observed_driver
            elif driver != observed_driver:
                raise ValueError("GPU driver versions disagree")
            rows.append(
                {
                    "UUID": uuid,
                    "index": int(index),
                    "memory_total_MiB": int(memory),
                    "name": name,
                }
            )
    except (TypeError, ValueError) as exc:
        raise Dstc9P17RuntimeBindingError(
            "current NVIDIA hardware probe is malformed"
        ) from exc
    rows.sort(key=lambda row: int(row["index"]))
    if rows != EXPECTED_GPU_ROWS or driver is None:
        raise Dstc9P17RuntimeBindingError(
            "current GPU identity drifted"
        )
    return {
        "GPUs": rows,
        "NVIDIA_driver_version": driver,
        "kernel_release": os.uname().release,
    }


def verify_current_study_hardware_binding(
    *,
    path: Path,
    worker_project_root: Path,
    expected_study_id: str,
) -> dict[str, object]:
    """Validate the pre-canary receipt against current source-free hardware."""

    binding = _load_current_hardware_binding(
        path=path,
        worker_project_root=worker_project_root,
        expected_study_id=expected_study_id,
    )
    if binding.get("hardware") != _probe_current_hardware():
        raise Dstc9P17RuntimeBindingError(
            "current-study hardware no longer matches its pre-canary capture"
        )
    return binding


def _verify_fingerprint_contract(value: Mapping[str, Any]) -> None:
    python = value.get("python_executable")
    inventory = value.get("runtime_inventory_receipt")
    assets = value.get("frozen_asset_receipts")
    minilm = value.get("minilm_runtime_receipt")
    repair = value.get("prelaunch_runtime_repair")
    canaries = value.get("worker_canaries")
    if not all(
        isinstance(row, Mapping)
        for row in (python, inventory, assets, minilm, repair, canaries)
    ):
        raise Dstc9P17RuntimeBindingError(
            "runtime fingerprint contract is incomplete"
        )
    assert isinstance(python, Mapping)
    assert isinstance(inventory, Mapping)
    assert isinstance(assets, Mapping)
    assert isinstance(minilm, Mapping)
    assert isinstance(repair, Mapping)
    assert isinstance(canaries, Mapping)
    hippo_canary = canaries.get("HippoRAG_CPU")
    if (
        python
        != {
            "path": str(P17_RUNTIME_PYTHON),
            "resolved_target_sha256": RUNTIME_PYTHON_TARGET_SHA256,
            "resolved_target_size_bytes": RUNTIME_PYTHON_TARGET_SIZE,
        }
        or dict(inventory) != EXPECTED_RUNTIME_INVENTORY
        or any(
            assets.get(name) != expected
            for name, expected in EXPECTED_ASSET_TREES.items()
        )
        or minilm.get("asset_file_sha256")
        != MINILM_MANIFEST_FILE_SHA256
        or minilm.get("asset_sha256")
        != MINILM_MANIFEST_SELF_SHA256
        or minilm.get("model_root") != str(P17_MINILM_ROOT)
        or minilm.get("model_tree_sha256")
        != MINILM_NORMATIVE_TREE_SHA256
        or minilm.get("weights_sha256") != MINILM_WEIGHTS_SHA256
        or minilm.get("runtime_versions")
        != EXPECTED_MINILM_RUNTIME_VERSIONS
        or repair.get("repaired_source_file_sha256")
        != REPAIRED_SOURCE_FILE_SHA256
        or repair.get("repair_choice_count") != 1
        or not isinstance(hippo_canary, Mapping)
        or hippo_canary.get("origin")
        != "P17_outcome_blind_prelaunch_deterministic_source_repair_canary"
        or hippo_canary.get("service_result") != "success"
        or hippo_canary.get("terminal") is not True
        or not isinstance(hippo_canary.get("network_audit"), Mapping)
        or hippo_canary["network_audit"].get(
            "external_network_call_count"
        )
        != 0
    ):
        raise Dstc9P17RuntimeBindingError(
            "runtime fingerprint reused-closure contract drifted"
        )


def _verify_runtime_python(runtime_python: Path) -> dict[str, object]:
    lexical = _absolute_without_symlink(
        runtime_python,
        "runtime Python",
        allow_final_symlink=True,
    )
    if (
        lexical != P17_RUNTIME_PYTHON
        or not lexical.is_file()
        or not os.access(lexical, os.X_OK)
    ):
        raise Dstc9P17RuntimeBindingError(
            "runtime Python lexical path drifted"
        )
    try:
        resolved = lexical.resolve(strict=True)
        metadata_row = resolved.stat()
    except OSError as exc:
        raise Dstc9P17RuntimeBindingError(
            "runtime Python target is unavailable"
        ) from exc
    if (
        not stat.S_ISREG(metadata_row.st_mode)
        or metadata_row.st_size != RUNTIME_PYTHON_TARGET_SIZE
        or file_sha256(resolved) != RUNTIME_PYTHON_TARGET_SHA256
    ):
        raise Dstc9P17RuntimeBindingError(
            "runtime Python target drifted"
        )
    return {
        "lexical_path": str(lexical),
        "resolved_path": str(resolved),
        "resolved_target_sha256": RUNTIME_PYTHON_TARGET_SHA256,
        "resolved_target_size_bytes": RUNTIME_PYTHON_TARGET_SIZE,
    }


def _verify_pth_topology() -> list[dict[str, object]]:
    overlay = _absolute_without_symlink(
        P17_VENV_SITE_PACKAGES,
        "P17 runtime site-packages",
    )
    if not overlay.is_dir():
        raise Dstc9P17RuntimeBindingError(
            "P17 runtime site-packages is unavailable"
        )
    rows: list[dict[str, object]] = []
    for path in sorted(overlay.glob("*.pth"), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file():
            raise Dstc9P17RuntimeBindingError(
                "P17 runtime .pth topology drifted"
            )
        rows.append(
            {
                "name": path.name,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    if (
        rows != EXPECTED_PTH_ROWS
        or stable_hash(rows) != EXPECTED_PTH_SET_SHA256
    ):
        raise Dstc9P17RuntimeBindingError(
            "P17 runtime .pth binding drifted"
        )
    return rows


def _verify_worker_project_root(
    worker_project_root: Path,
) -> dict[str, object]:
    actual = _absolute_without_symlink(
        worker_project_root,
        "DSTC9 worker project root",
    )
    module_path = _absolute_without_symlink(
        Path(__file__),
        "DSTC9 runtime binding module",
    )
    expected = module_path.parents[2]
    if actual != expected or actual.is_relative_to(P17_REMOTE_ROOT):
        raise Dstc9P17RuntimeBindingError(
            "DSTC9 worker project root drifted"
        )
    rows: list[dict[str, object]] = []
    for relative in WORKER_CODE_RELATIVE_FILES:
        path = actual / relative
        if path.is_symlink() or not path.is_file():
            raise Dstc9P17RuntimeBindingError(
                "DSTC9 worker code binding is incomplete"
            )
        rows.append(
            {
                "path": relative,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "bootstrap_sys_path_policy": (
            "exact_single_formal_worker_root_prefix_then_"
            "fingerprinted_P17_base_sys_path_v1"
        ),
        "files": rows,
        "project_root": str(actual),
        "set_sha256": stable_hash(rows),
    }


def _verify_minilm_manifest_and_tree() -> dict[str, object]:
    manifest_path = _absolute_without_symlink(
        P17_MINILM_MANIFEST,
        "MiniLM manifest",
    )
    if not manifest_path.is_file():
        raise Dstc9P17RuntimeBindingError(
            "MiniLM manifest is unavailable"
        )
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Dstc9P17RuntimeBindingError(
            "MiniLM manifest is invalid"
        ) from exc
    if not isinstance(manifest, dict):
        raise Dstc9P17RuntimeBindingError(
            "MiniLM manifest is invalid"
        )
    if hashlib.sha256(raw).hexdigest() != MINILM_MANIFEST_FILE_SHA256:
        raise Dstc9P17RuntimeBindingError(
            "MiniLM manifest file hash drifted"
        )
    body = dict(manifest)
    declared = body.pop("asset_sha256", None)
    local = manifest.get("local_binding")
    model = manifest.get("model")
    if (
        declared != MINILM_MANIFEST_SELF_SHA256
        or stable_hash(body) != declared
        or not isinstance(local, Mapping)
        or not isinstance(model, Mapping)
        or local.get("snapshot_tree_sha256")
        != MINILM_NORMATIVE_TREE_SHA256
        or model.get("weights_sha256") != MINILM_WEIGHTS_SHA256
    ):
        raise Dstc9P17RuntimeBindingError(
            "MiniLM normative manifest drifted"
        )
    rows = _tree_rows(P17_MINILM_ROOT, "P17 MiniLM tree")
    generic = {
        "file_count": len(rows),
        "size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "tree_sha256": stable_hash(rows),
    }
    if generic != EXPECTED_ASSET_TREES["MiniLM"]:
        raise Dstc9P17RuntimeBindingError(
            "P17 MiniLM generic tree drifted"
        )
    declared_rows = local.get("snapshot_files")
    normative_rows = [
        {
            "path": row["path"],
            "sha256": row["sha256"],
            "size": row["size_bytes"],
        }
        for row in rows
    ]
    if (
        declared_rows != normative_rows
        or stable_hash(normative_rows) != MINILM_NORMATIVE_TREE_SHA256
    ):
        raise Dstc9P17RuntimeBindingError(
            "P17 MiniLM normative tree drifted"
        )
    return {
        "asset_manifest_file_sha256": MINILM_MANIFEST_FILE_SHA256,
        "asset_manifest_self_sha256": MINILM_MANIFEST_SELF_SHA256,
        "generic_tree": generic,
        "model_root": str(P17_MINILM_ROOT),
        "normative_tree_sha256": MINILM_NORMATIVE_TREE_SHA256,
        "weights_sha256": MINILM_WEIGHTS_SHA256,
    }


def verify_p17_reused_closure_binding(
    *,
    expected_study_id: str,
    worker_project_root: Path,
    current_hardware_binding_path: Path,
    runtime_fingerprint_path: Path,
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
) -> dict[str, Any]:
    """Bind the reused P17 dependency closure and current-study hardware."""

    fingerprint = _load_exact_fingerprint(runtime_fingerprint_path)
    _verify_fingerprint_contract(fingerprint)
    worker_code_binding = _verify_worker_project_root(
        worker_project_root
    )
    current_hardware_binding = _load_current_hardware_binding(
        path=current_hardware_binding_path,
        worker_project_root=worker_project_root,
        expected_study_id=expected_study_id,
    )
    project_root = _absolute_without_symlink(
        P17_PROJECT_ROOT,
        "P17 project root",
    )
    if not project_root.is_dir():
        raise Dstc9P17RuntimeBindingError(
            "P17 project root is unavailable"
        )
    llm = _absolute_without_symlink(
        local_llm_model,
        "P17 SmolLM root",
    )
    embedding = _absolute_without_symlink(
        local_embedding_model,
        "P17 MiniLM root",
    )
    if llm != P17_SMOLLM_ROOT or embedding != P17_MINILM_ROOT:
        raise Dstc9P17RuntimeBindingError(
            "P17 model root path drifted"
        )
    python_binding = _verify_runtime_python(runtime_python)
    pth_rows = _verify_pth_topology()
    smollm_tree = tree_receipt(llm, "P17 SmolLM tree")
    if smollm_tree != EXPECTED_ASSET_TREES["HippoRAG_LLM"]:
        raise Dstc9P17RuntimeBindingError(
            "P17 SmolLM tree drifted"
        )
    minilm_binding = _verify_minilm_manifest_and_tree()
    source_tree = _hipporag_source_tree_receipt()
    if (
        file_sha256(P17_REPAIRED_SOURCE_FILE)
        != REPAIRED_SOURCE_FILE_SHA256
    ):
        raise Dstc9P17RuntimeBindingError(
            "repaired P17 HippoRAG source drifted"
        )
    body: dict[str, Any] = {
        "assets": {
            "HippoRAG_LLM": {
                "model_root": str(llm),
                **smollm_tree,
            },
            "HippoRAG_source": {
                "repaired_file_sha256": (
                    REPAIRED_SOURCE_FILE_SHA256
                ),
                "source_root": str(P17_HIPPORAG_SOURCE),
                **source_tree,
            },
            "MiniLM": minilm_binding,
        },
        "current_hardware_binding": current_hardware_binding,
        "fingerprint_file_sha256": FINGERPRINT_FILE_SHA256,
        "fingerprint_self_sha256": FINGERPRINT_SELF_SHA256,
        "inherited_musique_v3_attestation_called": False,
        "project_root": str(project_root),
        "runtime_inventory": dict(EXPECTED_RUNTIME_INVENTORY),
        "runtime_python": python_binding,
        "runtime_reuse_scope": {
            "asserted_from_P17_fingerprint": [
                "frozen_asset_receipts",
                "minilm_runtime_receipt",
                "prelaunch_runtime_repair",
                "python_executable",
                "runtime_inventory_receipt",
                "worker_canaries.HippoRAG_CPU",
            ],
            "current_host_hardware_from_separate_receipt": True,
            "P17_hardware_driver_or_kernel_asserted_current": False,
        },
        "schema": SCHEMA,
        "status": (
            "verified_P17_reused_dependency_closure_with_"
            "separate_current_hardware_binding"
        ),
        "worker_code_binding": worker_code_binding,
        "worker_provenance_contract": {
            "active_distribution_versions": (
                EXPECTED_ACTIVE_DISTRIBUTIONS
            ),
            "hipporag_import_origin": str(P17_HIPPORAG_INIT),
            "hipporag_import_origin_sha256": (
                HIPPORAG_INIT_FILE_SHA256
            ),
            "pth_rows": pth_rows,
            "pth_set_sha256": EXPECTED_PTH_SET_SHA256,
            "required_P17_base_sys_path_roots": [
                str(P17_PROJECT_ROOT),
                str(P17_P16_SITE),
                str(P17_HIPPORAG_SOURCE),
            ],
            "P17_base_runtime_inventory_sha256": (
                EXPECTED_RUNTIME_INVENTORY["inventory_sha256"]
            ),
            "P17_base_sys_path_sha256": (
                EXPECTED_RUNTIME_INVENTORY["sys_path_sha256"]
            ),
        },
    }
    return {**body, "self_sha256": stable_hash(body)}


def _distribution_rows(
    sys_path_values: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    discovery_arguments: dict[str, object] = {}
    if sys_path_values is not None:
        discovery_arguments["path"] = list(sys_path_values)
    rows: list[dict[str, str]] = []
    for distribution in metadata.distributions(**discovery_arguments):
        name = distribution.metadata.get("Name")
        version = distribution.version
        location = str(distribution.locate_file(""))
        if (
            not isinstance(name, str)
            or not name
            or not version
            or not location
        ):
            raise Dstc9P17RuntimeBindingError(
                "worker distribution inventory is malformed"
            )
        rows.append(
            {"location": location, "name": name, "version": version}
        )
    rows.sort(
        key=lambda row: (
            row["name"].casefold(),
            row["version"],
            row["location"],
        )
    )
    return rows


def _runtime_inventory_receipt(
    sys_path_values: Sequence[str] | None = None,
) -> dict[str, object]:
    rows = _distribution_rows(sys_path_values)
    effective_sys_path = list(
        sys.path if sys_path_values is None else sys_path_values
    )
    body = {
        "distributions": rows,
        "python_executable": sys.executable,
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "sys_path": effective_sys_path,
    }
    return {
        "distribution_count": len(rows),
        "inventory_sha256": stable_hash(body),
        "python_executable": sys.executable,
        "python_version": body["python_version"],
        "sys_path_sha256": stable_hash(body["sys_path"]),
    }


def _active_distribution_version(
    name: str,
    base_sys_path: Sequence[str],
) -> str | None:
    try:
        distribution = next(
            metadata.distributions(path=list(base_sys_path), name=name)
        )
    except StopIteration:
        return None
    version = distribution.version
    if not isinstance(version, str) or not version:
        raise Dstc9P17RuntimeBindingError(
            f"worker active distribution is malformed: {name}"
        )
    return version


def verify_worker_runtime_provenance(
    *,
    binding_receipt_path: Path,
    binding_receipt_file_sha256: str,
    p17_project_root: Path,
    worker_project_root: Path,
    current_hardware_binding_path: Path,
    expected_study_id: str,
    runtime_fingerprint_path: Path,
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
    effective_sys_path: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Recompute the binding inside the clean worker before model imports."""

    if _absolute_without_symlink(
        p17_project_root,
        "P17 dependency project root",
    ) != P17_PROJECT_ROOT:
        raise Dstc9P17RuntimeBindingError(
            "P17 dependency project root drifted"
        )
    if (
        not isinstance(binding_receipt_file_sha256, str)
        or _HEX64.fullmatch(binding_receipt_file_sha256) is None
    ):
        raise Dstc9P17RuntimeBindingError(
            "runtime binding receipt hash is invalid"
        )
    persisted, raw = _load_canonical_json(
        binding_receipt_path,
        "runtime binding receipt",
    )
    if hashlib.sha256(raw).hexdigest() != binding_receipt_file_sha256:
        raise Dstc9P17RuntimeBindingError(
            "runtime binding receipt file drifted"
        )
    live = verify_p17_reused_closure_binding(
        expected_study_id=expected_study_id,
        worker_project_root=worker_project_root,
        current_hardware_binding_path=current_hardware_binding_path,
        runtime_fingerprint_path=runtime_fingerprint_path,
        runtime_python=runtime_python,
        local_llm_model=local_llm_model,
        local_embedding_model=local_embedding_model,
    )
    if (
        runtime_binding_acceptance_identity(persisted)
        != runtime_binding_acceptance_identity(live)
    ):
        raise Dstc9P17RuntimeBindingError(
            "worker live runtime differs from binding receipt"
        )
    if Path(sys.executable).absolute() != P17_RUNTIME_PYTHON:
        raise Dstc9P17RuntimeBindingError(
            "worker active Python path drifted"
        )
    active_sys_path = tuple(
        sys.path if effective_sys_path is None else effective_sys_path
    )
    formal_root = _absolute_without_symlink(
        worker_project_root,
        "DSTC9 worker project root",
    )
    formal_root_text = str(formal_root)
    if (
        not active_sys_path
        or any(not isinstance(value, str) for value in active_sys_path)
        or active_sys_path[0] != formal_root_text
        or sum(
            str(Path(value).absolute()) == formal_root_text
            for value in active_sys_path
        )
        != 1
    ):
        raise Dstc9P17RuntimeBindingError(
            "worker formal-code sys.path prefix drifted"
        )
    base_sys_path = active_sys_path[1:]
    required_roots = (
        P17_PROJECT_ROOT,
        P17_P16_SITE,
        P17_HIPPORAG_SOURCE,
    )
    normalized_paths = {
        str(Path(value).absolute())
        for value in base_sys_path
        if value
    }
    if any(str(root) not in normalized_paths for root in required_roots):
        raise Dstc9P17RuntimeBindingError(
            "worker required P17 base sys.path provenance is absent"
        )
    inventory = _runtime_inventory_receipt(base_sys_path)
    if inventory != EXPECTED_RUNTIME_INVENTORY:
        raise Dstc9P17RuntimeBindingError(
            "worker P17 base runtime inventory or sys.path drifted"
        )
    current_hardware = live.get("current_hardware_binding")
    if (
        not isinstance(current_hardware, Mapping)
        or current_hardware.get("hardware")
        != _probe_current_hardware()
    ):
        raise Dstc9P17RuntimeBindingError(
            "worker current-study hardware drifted"
        )
    for name, expected_version in EXPECTED_ACTIVE_DISTRIBUTIONS.items():
        observed = _active_distribution_version(name, base_sys_path)
        if observed != expected_version:
            raise Dstc9P17RuntimeBindingError(
                f"worker active distribution drifted: {name}"
            )
    hippo_spec = util.find_spec("hipporag")
    origin = getattr(hippo_spec, "origin", None)
    if (
        not isinstance(origin, str)
        or Path(origin).absolute() != P17_HIPPORAG_INIT
        or file_sha256(P17_HIPPORAG_INIT)
        != HIPPORAG_INIT_FILE_SHA256
    ):
        raise Dstc9P17RuntimeBindingError(
            "worker HippoRAG import provenance drifted"
        )
    return live


__all__ = [
    "Dstc9P17RuntimeBindingError",
    "FINGERPRINT_FILE_SHA256",
    "FINGERPRINT_SELF_SHA256",
    "P17_FINGERPRINT_PATH",
    "P17_HIPPORAG_SOURCE",
    "P17_MINILM_ROOT",
    "P17_PROJECT_ROOT",
    "P17_RUNTIME_PYTHON",
    "P17_SMOLLM_ROOT",
    "SCHEMA",
    "capture_current_study_hardware_binding",
    "canonical_json_bytes",
    "file_sha256",
    "make_current_study_hardware_binding",
    "runtime_binding_acceptance_identity",
    "stable_hash",
    "tree_receipt",
    "verify_current_study_hardware_binding",
    "verify_p17_reused_closure_binding",
    "verify_worker_runtime_provenance",
]
