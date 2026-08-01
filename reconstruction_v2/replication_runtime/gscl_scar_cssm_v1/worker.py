"""Two-shard, label-blind SCAR CSSM action worker.

Each process binds one visible GPU, one exact local Qwen snapshot, and one
exact CPU MiniLM snapshot.  It reads only the frozen action pack and appends
private item actions to an ext4 JSONL file.  Labels and scoring are absent
from the interface and imports.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import socket
import stat
import sys
import time
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks import gscl_scar_cssm_action_v1 as action
from assumption_agent.benchmarks import gscl_scar_cssm_source_v1 as source
from replication_runtime.gscl_narrative_extractor_v1 import worker as qwen_assets
from replication_runtime.gscl_narrative_extractor_v2 import (
    document_envelope,
    fixed_document_envelope_qualification as document_qualification,
    fixed_public_qualification as leaf_qualification,
)
from replication_runtime.qasper_minilm_portable_v2.binding import (
    PortableOfflineMiniLMEncoder,
)


VERSION = "gscl_scar_cssm_worker_v1"
SHARD_TERMINAL_SCHEMA = f"{VERSION}.shard.safe_terminal.v1"
RUNTIME_RECEIPT_SCHEMA = f"{VERSION}.runtime.safe_receipt.v1"
SANDBOX_RECEIPT_SCHEMA = "gscl_scar_cssm_sandbox_freeze_v1"
ACTION_RELEASE_SCHEMA = f"{VERSION}.two_shard_action_release.v1"
SHARD_COUNT = 2
MAXIMUM_ACTION_PACK_BYTES = 16 * 1024 * 1024
MAXIMUM_ITEM_LINE_BYTES = 16 * 1024 * 1024
MAXIMUM_SANDBOX_RECEIPT_BYTES = 1 * 1024 * 1024
ACTION_RELEASE_TIMEOUT_SECONDS = 3_600

IMPLEMENTATION_RELATIVE_PATHS = (
    "assumption_agent/__init__.py",
    "assumption_agent/archive.py",
    "assumption_agent/benchmarks/__init__.py",
    "assumption_agent/benchmarks/codex_action_budget.py",
    "assumption_agent/benchmarks/codex_execution_policy.py",
    "assumption_agent/benchmarks/docker_egress.py",
    "assumption_agent/benchmarks/gscl_scar_cssm_action_v1.py",
    "assumption_agent/benchmarks/gscl_scar_cssm_source_v1.py",
    "assumption_agent/benchmarks/offline_verifier.py",
    "assumption_agent/benchmarks/runtime_profile_injection.py",
    "assumption_agent/benchmarks/skilllearn_compiler.py",
    "assumption_agent/benchmarks/skilllearn_lifecycle.py",
    "assumption_agent/benchmarks/skilllearnbench.py",
    "assumption_agent/benchmarks/task_input_closure.py",
    "assumption_agent/benchmarks/typed_task_capability.py",
    "assumption_agent/evaluation.py",
    "assumption_agent/events.py",
    "assumption_agent/evolution.py",
    "assumption_agent/generalized_structural_correspondence_v1.py",
    "assumption_agent/gscl_narrative_correspondence_v1.py",
    "assumption_agent/gscl_slot_graph_binder_v1.py",
    "assumption_agent/gscl_slot_set_mapping_v1.py",
    "assumption_agent/gscl_unit_mapping_v2.py",
    "assumption_agent/meta_assumption.py",
    "assumption_agent/models.py",
    "assumption_agent/proposer.py",
    "assumption_agent/runtime.py",
    "assumption_agent/secure_env.py",
    "assumption_agent/splits.py",
    "assumption_agent/structural_law_residuals_v1.py",
    "assumption_agent/typed_operator_grammar.py",
    "assumption_agent/universal_assumption_ontology_v1.py",
    "assumption_agent/validation.py",
    "replication_runtime/__init__.py",
    "replication_runtime/gscl_narrative_extractor_v1/__init__.py",
    "replication_runtime/gscl_narrative_extractor_v1/closed_choice_worker.py",
    "replication_runtime/gscl_narrative_extractor_v1/contract.py",
    "replication_runtime/gscl_narrative_extractor_v1/worker.py",
    "replication_runtime/gscl_narrative_extractor_v2/__init__.py",
    "replication_runtime/gscl_narrative_extractor_v2/bounded_set_consumer.py",
    "replication_runtime/gscl_narrative_extractor_v2/closed_choice.py",
    "replication_runtime/gscl_narrative_extractor_v2/contract.py",
    "replication_runtime/gscl_narrative_extractor_v2/document_envelope.py",
    "replication_runtime/gscl_narrative_extractor_v2/fixed_document_envelope_qualification.py",
    "replication_runtime/gscl_narrative_extractor_v2/fixed_public_qualification.py",
    "replication_runtime/gscl_narrative_extractor_v2/memory_safe_qwen.py",
    "replication_runtime/gscl_scar_cssm_v1/__init__.py",
    "replication_runtime/gscl_scar_cssm_v1/worker.py",
    "replication_runtime/qasper_minilm_portable_v2/__init__.py",
    "replication_runtime/qasper_minilm_portable_v2/binding.py",
    "replication_runtime/qasper_minilm_v1/__init__.py",
    "replication_runtime/qasper_minilm_v1/binding.py",
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ScarCssmWorkerError(RuntimeError):
    def __init__(self, issue_id: str) -> None:
        self.issue_id = issue_id
        super().__init__(issue_id)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ScarCssmWorkerError("WORKER_CANONICAL_JSON_INVALID") from exc


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _regular_private_file(
    path: Path, *, maximum_bytes: int
) -> tuple[bytes, Mapping[str, object]]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ScarCssmWorkerError("WORKER_INPUT_UNAVAILABLE") from exc
    if (
        not path.is_absolute()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
        or metadata.st_size > maximum_bytes
    ):
        raise ScarCssmWorkerError("WORKER_INPUT_INVALID")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ScarCssmWorkerError("WORKER_INPUT_UNAVAILABLE") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != metadata.st_dev
            or opened.st_ino != metadata.st_ino
            or opened.st_uid != metadata.st_uid
            or opened.st_nlink != metadata.st_nlink
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_size != metadata.st_size
            or opened.st_mtime_ns != metadata.st_mtime_ns
        ):
            raise ScarCssmWorkerError("WORKER_INPUT_CHANGED")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1 << 20, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise ScarCssmWorkerError("WORKER_INPUT_INVALID")
        after = os.fstat(descriptor)
        if (
            after.st_dev != opened.st_dev
            or after.st_ino != opened.st_ino
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or after.st_nlink != opened.st_nlink
            or total != opened.st_size
        ):
            raise ScarCssmWorkerError("WORKER_INPUT_CHANGED")
        raw = b"".join(chunks)
        return raw, MappingProxyType(
            {
                "device": opened.st_dev,
                "inode": opened.st_ino,
                "mode_octal": "0600",
                "mtime_ns": opened.st_mtime_ns,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
        )
    except OSError as exc:
        raise ScarCssmWorkerError("WORKER_INPUT_UNAVAILABLE") from exc
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_action_pack(
    path: Path,
    *,
    study_id: str,
    expected_file_sha256: str,
    expected_action_commitment_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, object]]:
    raw, file_receipt = _regular_private_file(
        path, maximum_bytes=MAXIMUM_ACTION_PACK_BYTES
    )
    if file_receipt["sha256"] != expected_file_sha256:
        raise ScarCssmWorkerError("WORKER_ACTION_PACK_BINDING_DRIFTED")
    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ScarCssmWorkerError("WORKER_ACTION_PACK_INVALID") from exc
    if type(value) is not dict or _canonical_bytes(value) != raw:
        raise ScarCssmWorkerError("WORKER_ACTION_PACK_INVALID")
    try:
        source.validate_scar_cssm_action_pack_v1(value, study_id=study_id)
    except Exception as exc:
        raise ScarCssmWorkerError("WORKER_ACTION_PACK_INVALID") from exc
    if value.get("action_commitment_sha256") != expected_action_commitment_sha256:
        raise ScarCssmWorkerError("WORKER_ACTION_PACK_BINDING_DRIFTED")
    return MappingProxyType(value), file_receipt


def _publish_once(path: Path, raw: bytes) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ScarCssmWorkerError("WORKER_OUTPUT_ALREADY_EXISTS")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _require_sha256(value: object, *, issue_id: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ScarCssmWorkerError(issue_id)
    return value


def _stable_code_file(path: Path) -> dict[str, object]:
    if not path.is_absolute():
        raise ScarCssmWorkerError("WORKER_IMPLEMENTATION_CLOSURE_INVALID")
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_size > 32 * 1024 * 1024
        ):
            raise ScarCssmWorkerError(
                "WORKER_IMPLEMENTATION_CLOSURE_INVALID"
            )
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            opened = os.fstat(descriptor)
            digest = hashlib.sha256()
            total = 0
            while True:
                chunk = os.read(descriptor, 1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                total += len(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ScarCssmWorkerError(
            "WORKER_IMPLEMENTATION_CLOSURE_INVALID"
        ) from exc
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if (
        identity
        != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        or identity
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or total != opened.st_size
    ):
        raise ScarCssmWorkerError("WORKER_IMPLEMENTATION_CLOSURE_CHANGED")
    return {"sha256": digest.hexdigest(), "size_bytes": total}


def _implementation_closure() -> Mapping[str, object]:
    code_root = Path(os.path.abspath(os.fspath(__file__))).parents[2]
    files: dict[str, object] = {}
    for relative in IMPLEMENTATION_RELATIVE_PATHS:
        files[relative] = _stable_code_file(code_root / relative)
    body = {
        "files": files,
        "relative_paths": list(IMPLEMENTATION_RELATIVE_PATHS),
        "version": VERSION,
    }
    return MappingProxyType(
        {**body, "self_sha256": _content_hash(body)}
    )


def _decode_canonical_receipt(
    raw: bytes, *, expected_file_sha256: str, study_id: str
) -> Mapping[str, object]:
    if hashlib.sha256(raw).hexdigest() != expected_file_sha256:
        raise ScarCssmWorkerError("WORKER_SANDBOX_RECEIPT_BINDING_DRIFTED")
    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ScarCssmWorkerError("WORKER_SANDBOX_RECEIPT_INVALID") from exc
    if type(value) is not dict or _canonical_bytes(value) != raw:
        raise ScarCssmWorkerError("WORKER_SANDBOX_RECEIPT_INVALID")
    body = dict(value)
    self_hash = body.pop("self_sha256", None)
    if (
        value.get("schema") != SANDBOX_RECEIPT_SCHEMA
        or value.get("study_id") != study_id
        or value.get("status") != "frozen"
        or self_hash != _content_hash(body)
        or value.get("ip_address_deny") != "any"
        or value.get("restrict_address_families") != "AF_UNIX"
        or value.get("action_label_path_denied") is not True
        or value.get("action_external_network_denied") is not True
    ):
        raise ScarCssmWorkerError("WORKER_SANDBOX_RECEIPT_INVALID")
    return MappingProxyType(value)


def _load_sandbox_receipt(
    path: Path, *, expected_file_sha256: str, study_id: str
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    raw, file_receipt = _regular_private_file(
        path, maximum_bytes=MAXIMUM_SANDBOX_RECEIPT_BYTES
    )
    return (
        _decode_canonical_receipt(
            raw,
            expected_file_sha256=expected_file_sha256,
            study_id=study_id,
        ),
        file_receipt,
    )


def _wait_for_action_release(
    path: Path,
    *,
    study_id: str,
    shard_index: int,
    own_runtime_file_sha256: str,
    expected_action_file_sha256: str,
    expected_action_commitment_sha256: str,
    expected_execution_freeze_sha256: str,
    expected_gpu_uuid: str,
    expected_peer_gpu_uuid: str,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    if (
        not path.is_absolute()
        or expected_gpu_uuid == expected_peer_gpu_uuid
        or not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", expected_peer_gpu_uuid)
    ):
        raise ScarCssmWorkerError("WORKER_ACTION_RELEASE_BINDING_INVALID")
    deadline = time.monotonic() + ACTION_RELEASE_TIMEOUT_SECONDS
    while not path.exists():
        if time.monotonic() >= deadline:
            raise ScarCssmWorkerError("WORKER_ACTION_RELEASE_TIMEOUT")
        time.sleep(0.1)
    raw, file_receipt = _regular_private_file(
        path, maximum_bytes=MAXIMUM_SANDBOX_RECEIPT_BYTES
    )
    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ScarCssmWorkerError("WORKER_ACTION_RELEASE_INVALID") from exc
    if type(value) is not dict or _canonical_bytes(value) != raw:
        raise ScarCssmWorkerError("WORKER_ACTION_RELEASE_INVALID")
    body = dict(value)
    self_hash = body.pop("self_sha256", None)
    expected_gpus = {
        str(shard_index): expected_gpu_uuid,
        str(1 - shard_index): expected_peer_gpu_uuid,
    }
    runtime_files = value.get("runtime_receipt_file_sha256_by_shard")
    if (
        value.get("schema") != ACTION_RELEASE_SCHEMA
        or value.get("status") != "release_both_shards_to_action_pack"
        or value.get("study_id") != study_id
        or value.get("shard_count") != SHARD_COUNT
        or value.get("action_file_sha256") != expected_action_file_sha256
        or value.get("action_commitment_sha256")
        != expected_action_commitment_sha256
        or value.get("execution_freeze_sha256")
        != expected_execution_freeze_sha256
        or value.get("gpu_uuid_by_shard") != expected_gpus
        or type(runtime_files) is not dict
        or set(runtime_files) != {"0", "1"}
        or any(
            not isinstance(child, str) or _SHA256.fullmatch(child) is None
            for child in runtime_files.values()
        )
        or runtime_files[str(shard_index)] != own_runtime_file_sha256
        or self_hash != _content_hash(body)
    ):
        raise ScarCssmWorkerError("WORKER_ACTION_RELEASE_BINDING_INVALID")
    return MappingProxyType(value), file_receipt


def _process_status() -> Mapping[str, object]:
    try:
        rows = {}
        for line in Path("/proc/self/status").read_text(
            encoding="ascii", errors="strict"
        ).splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                rows[key] = value.strip()
    except OSError as exc:
        raise ScarCssmWorkerError("WORKER_SANDBOX_RUNTIME_INVALID") from exc
    if rows.get("NoNewPrivs") != "1":
        raise ScarCssmWorkerError("WORKER_SANDBOX_RUNTIME_INVALID")
    return MappingProxyType(
        {
            "no_new_privileges": True,
            "seccomp_mode": rows.get("Seccomp"),
            "seccomp_filter_count": rows.get("Seccomp_filters"),
        }
    )


def _network_family_negative_canary() -> Mapping[str, object]:
    evidence: dict[str, object] = {}
    for name, family in (("AF_INET", socket.AF_INET), ("AF_INET6", socket.AF_INET6)):
        try:
            handle = socket.socket(family, socket.SOCK_STREAM)
        except OSError as exc:
            evidence[name] = {
                "creation_denied": True,
                "errno": exc.errno,
            }
        else:
            handle.close()
            raise ScarCssmWorkerError("WORKER_NETWORK_SANDBOX_INACTIVE")
    evidence["external_connect_attempt_count"] = 0
    return MappingProxyType(evidence)


def _forbidden_file_negative_canary(path: Path) -> Mapping[str, object]:
    if not path.is_absolute():
        raise ScarCssmWorkerError("WORKER_FORBIDDEN_PROBE_INVALID")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except PermissionError as exc:
        return MappingProxyType(
            {"open_denied": True, "errno": exc.errno, "read_count": 0}
        )
    except OSError as exc:
        # ENOENT does not prove an active filesystem sandbox.
        raise ScarCssmWorkerError("WORKER_FORBIDDEN_PROBE_INVALID") from exc
    else:
        os.close(descriptor)
        raise ScarCssmWorkerError("WORKER_LABEL_SANDBOX_INACTIVE")


def _python_runtime_receipt() -> Mapping[str, object]:
    executable = Path(os.path.abspath(os.fspath(sys.executable)))
    executable_receipt = _stable_code_file(executable)
    versions: dict[str, str] = {}
    for distribution in (
        "huggingface-hub",
        "numpy",
        "safetensors",
        "sentence-transformers",
        "tokenizers",
        "torch",
        "transformers",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ScarCssmWorkerError("WORKER_RUNTIME_DISTRIBUTION_MISSING") from exc
    return MappingProxyType(
        {
            "distributions": versions,
            "executable_path": str(executable),
            "executable_sha256": executable_receipt["sha256"],
            "executable_size_bytes": executable_receipt["size_bytes"],
            "implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "sys_version": sys.version,
        }
    )


def _encoder_binding(encoder: PortableOfflineMiniLMEncoder) -> str:
    runtime = getattr(encoder, "runtime_receipt", None)
    canary = getattr(encoder, "canary_receipt", None)
    if not isinstance(runtime, Mapping) or not isinstance(canary, Mapping):
        raise ScarCssmWorkerError("WORKER_MINILM_BINDING_INVALID")
    return _content_hash(
        {
            "encoder_exact_type": (
                f"{type(encoder).__module__}.{type(encoder).__qualname__}"
            ),
            "portable_canary_receipt": dict(canary),
            "runtime_receipt": dict(runtime),
        }
    )


def _observed_qwen_runtime_requirements(runtime: object) -> dict[str, object]:
    try:
        import torch
        import transformers

        model = runtime._model
        torch_origin = getattr(torch, "__file__", None)
        transformers_origin = getattr(transformers, "__file__", None)
        cudnn_version = torch.backends.cudnn.version()
        capability = torch.cuda.get_device_capability(0)
        if (
            not isinstance(torch_origin, str)
            or not isinstance(transformers_origin, str)
            or not isinstance(cudnn_version, int)
        ):
            raise ValueError("runtime origin unavailable")
        return {
            "attention_implementation": str(
                getattr(model.config, "_attn_implementation", None)
            ),
            "cuda_version": str(torch.version.cuda),
            "cudnn_version": cudnn_version,
            "gpu_compute_capability": [
                int(capability[0]),
                int(capability[1]),
            ],
            "gpu_name": str(torch.cuda.get_device_name(0)),
            "python_executable_sha256": (
                qwen_assets._hash_runtime_executable()
            ),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "torch_distribution_sha256": (
                qwen_assets._distribution_closure_sha256(
                    "torch", required_module_origins=(Path(torch_origin),)
                )
            ),
            "transformers_version": str(transformers.__version__),
            "transformers_distribution_sha256": (
                qwen_assets._distribution_closure_sha256(
                    "transformers",
                    required_module_origins=(Path(transformers_origin),),
                )
            ),
        }
    except Exception as exc:
        raise ScarCssmWorkerError("WORKER_QWEN_RUNTIME_INVALID") from exc


def _load_qwen_runtime(*, model_root: Path, manifest_path: Path):
    try:
        manifest = qwen_assets.load_model_asset_manifest(
            manifest_path=manifest_path, model_root=model_root
        )
        document_qualification._verify_model_binding(
            model_root=model_root, manifest=manifest
        )
        runtime = document_qualification._load_exact_runtime(
            model_root=model_root, manifest=manifest
        )
        runtime_commitment = document_qualification._validate_exact_runtime(
            runtime=runtime, manifest=manifest
        )
        canary = dict(leaf_qualification._run_fixed_teacher_forced_canary(runtime))
        document_qualification._validate_success_canary(canary)
        runtime._validate_binding()
        observed_runtime = _observed_qwen_runtime_requirements(runtime)
        loaded_declarations = runtime._loaded_declarations()
        if (
            observed_runtime != dict(manifest.runtime_requirements)
            or loaded_declarations != dict(manifest.declarations)
        ):
            raise ScarCssmWorkerError("WORKER_QWEN_RUNTIME_DRIFTED")
    except Exception as exc:
        if isinstance(exc, ScarCssmWorkerError):
            raise
        raise ScarCssmWorkerError("WORKER_QWEN_RUNTIME_INVALID") from exc
    parameter_layout = [
        {
            "device": str(parameter.device),
            "dtype": str(parameter.dtype),
            "name": name,
            "shape": list(parameter.shape),
        }
        for name, parameter in runtime._model.named_parameters()
    ]
    evidence = {
        "manifest": {
            "declarations": dict(manifest.declarations),
            "files": list(manifest.files),
            "manifest_file_sha256": manifest.manifest_file_sha256,
            "runtime_requirements": dict(manifest.runtime_requirements),
            "self_sha256": manifest.self_sha256,
            "tree_sha256": manifest.tree_sha256,
        },
        "loaded_declarations": loaded_declarations,
        "observed_runtime_requirements": observed_runtime,
        "parameter_layout_sha256": _content_hash(parameter_layout),
        "qualification_canary": canary,
        "runtime_commitment": runtime_commitment,
        "runtime_source_sha256": runtime._source_sha256_value,
        "strategy": runtime.strategy,
    }
    return runtime, MappingProxyType(evidence)


def _require_one_visible_cuda_device(
    *, expected_gpu_uuid: str, runtime: object | None = None
) -> Mapping[str, object]:
    try:
        import torch

        count = torch.cuda.device_count()
        available = torch.cuda.is_available()
        current = torch.cuda.current_device() if available else -1
        properties = torch.cuda.get_device_properties(0) if available else None
    except Exception as exc:
        raise ScarCssmWorkerError("WORKER_CUDA_VISIBILITY_INVALID") from exc
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    raw_uuid = None if properties is None else str(getattr(properties, "uuid", ""))
    normalized_observed = (
        None
        if not raw_uuid
        else raw_uuid if raw_uuid.startswith("GPU-") else f"GPU-{raw_uuid}"
    )
    if (
        available is not True
        or count != 1
        or current != 0
        or visible != expected_gpu_uuid
        or normalized_observed != expected_gpu_uuid
        or properties is None
    ):
        raise ScarCssmWorkerError("WORKER_CUDA_VISIBILITY_INVALID")
    parameter_devices: set[str] = set()
    if runtime is not None:
        model = getattr(runtime, "_model", None)
        parameters = tuple(model.parameters()) if model is not None else ()
        if not parameters:
            raise ScarCssmWorkerError("WORKER_QWEN_PARAMETER_DEVICE_INVALID")
        parameter_devices = {str(row.device) for row in parameters}
        if parameter_devices != {"cuda:0"}:
            raise ScarCssmWorkerError("WORKER_QWEN_PARAMETER_DEVICE_INVALID")
    return MappingProxyType(
        {
            "compute_capability": [properties.major, properties.minor],
            "cuda_visible_devices": visible,
            "logical_current_device": current,
            "name": properties.name,
            "parameter_devices": sorted(parameter_devices),
            "physical_uuid": normalized_observed,
            "total_memory_bytes": properties.total_memory,
            "visible_device_count": count,
        }
    )


def _load_minilm(
    *, model_root: Path, asset_manifest_path: Path
) -> tuple[PortableOfflineMiniLMEncoder, str, Mapping[str, object]]:
    try:
        encoder = PortableOfflineMiniLMEncoder(
            asset_manifest_path=asset_manifest_path,
            model_root=model_root,
            run_canary=True,
        )
        binding = _encoder_binding(encoder)
    except Exception as exc:
        raise ScarCssmWorkerError("WORKER_MINILM_RUNTIME_INVALID") from exc
    return (
        encoder,
        binding,
        MappingProxyType(
            {
                "canary_receipt": dict(encoder.canary_receipt),
                "encoder_binding_sha256": binding,
                "encoder_exact_type": (
                    f"{type(encoder).__module__}.{type(encoder).__qualname__}"
                ),
                "runtime_receipt": dict(encoder.runtime_receipt),
            }
        ),
    )


def _selected_items(
    action_pack: Mapping[str, Any], *, shard_index: int
) -> tuple[Mapping[str, Any], ...]:
    items = action_pack.get("items")
    if type(items) is not list:
        raise ScarCssmWorkerError("WORKER_ACTION_PACK_INVALID")
    return tuple(
        item
        for ordinal, item in enumerate(items)
        if ordinal % SHARD_COUNT == shard_index
    )


def _runtime_receipt(
    *,
    study_id: str,
    shard_index: int,
    implementation_closure: Mapping[str, object],
    qwen: Mapping[str, object],
    minilm: Mapping[str, object],
    gpu: Mapping[str, object],
    sandbox: Mapping[str, object],
    sandbox_file_receipt: Mapping[str, object],
    process_status: Mapping[str, object],
    network_canary: Mapping[str, object],
    label_canary: Mapping[str, object],
    execution_freeze_sha256: str,
) -> Mapping[str, object]:
    try:
        import torch

        execution = {
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_build_version": torch.version.cuda,
            "cuda_runtime_available": torch.cuda.is_available(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "python": dict(_python_runtime_receipt()),
            "tokenizers_parallelism": os.environ.get(
                "TOKENIZERS_PARALLELISM"
            ),
            "torch_version": torch.__version__,
            "transformers_offline": os.environ.get(
                "TRANSFORMERS_OFFLINE"
            ),
        }
    except Exception as exc:
        if isinstance(exc, ScarCssmWorkerError):
            raise
        raise ScarCssmWorkerError("WORKER_RUNTIME_RECEIPT_INVALID") from exc
    if (
        execution["hf_hub_offline"] != "1"
        or execution["transformers_offline"] != "1"
        or execution["tokenizers_parallelism"] not in {"false", "False"}
        or execution["deterministic_algorithms"] is not True
        or execution["matmul_tf32"] is not False
        or execution["cudnn_tf32"] is not False
        or execution["cudnn_benchmark"] is not False
    ):
        raise ScarCssmWorkerError("WORKER_RUNTIME_ENVIRONMENT_INVALID")
    body = {
        "execution": execution,
        "execution_freeze_sha256": execution_freeze_sha256,
        "gpu": dict(gpu),
        "implementation_closure": dict(implementation_closure),
        "minilm": dict(minilm),
        "network_negative_canary": dict(network_canary),
        "process_sandbox": dict(process_status),
        "qwen": dict(qwen),
        "sandbox_freeze": dict(sandbox),
        "sandbox_freeze_file": dict(sandbox_file_receipt),
        "schema": RUNTIME_RECEIPT_SCHEMA,
        "shard_count": SHARD_COUNT,
        "shard_index": shard_index,
        "status": "qualified_before_action_pack_open",
        "study_id": study_id,
        "forbidden_label_negative_canary": dict(label_canary),
        "version": VERSION,
    }
    return MappingProxyType(
        {**body, "self_sha256": _content_hash(body)}
    )


def _mechanism_resource_counts(
    evidence: Mapping[str, object],
) -> Mapping[str, int]:
    if evidence.get("availability") == "PREMODEL_TYPED_FAILURE":
        return MappingProxyType(
            {
                "binder_receipt_count": 0,
                "bounded_set_receipt_count": 0,
                "candidate_count": 0,
                "document_envelope_receipt_count": 0,
                "forward_batch_count": 0,
                "graph_receipt_count": 0,
                "leaf_call_count": 0,
                "leaf_receipt_count": 0,
                "mapping_receipt_count": 0,
                "semantic_matrix_receipt_count": 0,
            }
        )
    try:
        sides = evidence["sides"]
        variants = evidence["variants"]
        resources = [
            sides[name]["document_envelope"]["receipt"]["receipt"][
                "resource_summary"
            ]
            for name in ("left", "right")
        ]
        leaf_receipts = sum(
            len(sides[name]["document_envelope"]["leaf_records"])
            for name in ("left", "right")
        )
        mapping_receipts = sum(
            len(variants[name]) for name in ("base", "system_swap")
        )
    except (KeyError, TypeError) as exc:
        raise ScarCssmWorkerError("WORKER_MECHANISM_EVIDENCE_INVALID") from exc
    return MappingProxyType(
        {
            "binder_receipt_count": 2,
            "bounded_set_receipt_count": 2,
            "candidate_count": sum(
                int(row["reported_success_candidate_count"])
                for row in resources
            ),
            "document_envelope_receipt_count": 2,
            "forward_batch_count": sum(
                int(row["reported_success_forward_batch_count"])
                for row in resources
            ),
            "graph_receipt_count": 2,
            "leaf_call_count": sum(
                int(row["leaf_call_count"]) for row in resources
            ),
            "leaf_receipt_count": leaf_receipts,
            "mapping_receipt_count": mapping_receipts,
            "semantic_matrix_receipt_count": 1,
        }
    )


def _mount_filesystem_type(path: Path) -> str:
    try:
        resolved = Path(os.path.abspath(os.fspath(path)))
        rows = Path("/proc/self/mountinfo").read_text(
            encoding="utf-8", errors="strict"
        ).splitlines()
    except OSError as exc:
        raise ScarCssmWorkerError("WORKER_OUTPUT_ROOT_INVALID") from exc
    matches: list[tuple[int, str]] = []
    for line in rows:
        if " - " not in line:
            continue
        left, right = line.split(" - ", 1)
        fields = left.split()
        suffix = right.split()
        if len(fields) < 5 or not suffix:
            continue
        mount = fields[4].replace("\\040", " ").replace("\\134", "\\")
        mount_path = Path(mount)
        try:
            resolved.relative_to(mount_path)
        except ValueError:
            continue
        matches.append((len(mount_path.parts), suffix[0]))
    if not matches:
        raise ScarCssmWorkerError("WORKER_OUTPUT_ROOT_INVALID")
    return max(matches)[1]


def _require_output_root(path: Path) -> Mapping[str, object]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ScarCssmWorkerError("WORKER_OUTPUT_ROOT_INVALID") from exc
    filesystem = _mount_filesystem_type(path)
    if (
        not path.is_absolute()
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or filesystem != "ext4"
    ):
        raise ScarCssmWorkerError("WORKER_OUTPUT_ROOT_INVALID")
    return MappingProxyType(
        {
            "device": metadata.st_dev,
            "filesystem_type": filesystem,
            "inode": metadata.st_ino,
            "mode_octal": "0700",
        }
    )


def run_shard(
    *,
    action_pack_path: Path,
    output_root: Path,
    study_id: str,
    shard_index: int,
    qwen_model_root: Path,
    qwen_manifest_path: Path,
    minilm_model_root: Path,
    minilm_manifest_path: Path,
    sandbox_receipt_path: Path,
    action_release_path: Path,
    forbidden_label_probe_path: Path,
    expected_action_file_sha256: str,
    expected_action_commitment_sha256: str,
    expected_implementation_closure_sha256: str,
    expected_sandbox_receipt_sha256: str,
    expected_execution_freeze_sha256: str,
    expected_gpu_uuid: str,
    expected_peer_gpu_uuid: str,
) -> Mapping[str, Any]:
    if (
        isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or shard_index not in {0, 1}
        or not output_root.is_absolute()
    ):
        raise ScarCssmWorkerError("WORKER_COORDINATE_INVALID")
    for value, issue_id in (
        (expected_action_file_sha256, "WORKER_ACTION_PACK_BINDING_INVALID"),
        (
            expected_action_commitment_sha256,
            "WORKER_ACTION_PACK_BINDING_INVALID",
        ),
        (
            expected_implementation_closure_sha256,
            "WORKER_IMPLEMENTATION_CLOSURE_INVALID",
        ),
        (
            expected_sandbox_receipt_sha256,
            "WORKER_SANDBOX_RECEIPT_BINDING_INVALID",
        ),
        (
            expected_execution_freeze_sha256,
            "WORKER_EXECUTION_FREEZE_BINDING_INVALID",
        ),
    ):
        _require_sha256(value, issue_id=issue_id)
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", expected_gpu_uuid):
        raise ScarCssmWorkerError("WORKER_GPU_BINDING_INVALID")
    output_root_receipt = _require_output_root(output_root)
    private_path = output_root / f"shard{shard_index}.records.private.jsonl"
    terminal_path = output_root / f"shard{shard_index}.terminal.safe.json"
    runtime_path = output_root / f"shard{shard_index}.runtime.safe.json"
    sentinel_path = output_root / f"shard{shard_index}.attempt.sentinel"

    # All model, implementation, GPU and sandbox checks precede both the
    # attempt claim and the first action-pack open.  A failure here is a
    # source-free infrastructure qualification failure, never an effect row.
    implementation_closure = _implementation_closure()
    if (
        implementation_closure["self_sha256"]
        != expected_implementation_closure_sha256
    ):
        raise ScarCssmWorkerError("WORKER_IMPLEMENTATION_CLOSURE_DRIFTED")
    sandbox_receipt, sandbox_file_receipt = _load_sandbox_receipt(
        sandbox_receipt_path,
        expected_file_sha256=expected_sandbox_receipt_sha256,
        study_id=study_id,
    )
    process_status = _process_status()
    network_canary = _network_family_negative_canary()
    label_canary = _forbidden_file_negative_canary(
        forbidden_label_probe_path
    )
    _require_one_visible_cuda_device(
        expected_gpu_uuid=expected_gpu_uuid
    )
    runtime, qwen_evidence = _load_qwen_runtime(
        model_root=qwen_model_root, manifest_path=qwen_manifest_path
    )
    gpu_evidence = _require_one_visible_cuda_device(
        expected_gpu_uuid=expected_gpu_uuid, runtime=runtime
    )
    encoder, encoder_binding, minilm_evidence = _load_minilm(
        model_root=minilm_model_root,
        asset_manifest_path=minilm_manifest_path,
    )
    runtime_receipt = _runtime_receipt(
        study_id=study_id,
        shard_index=shard_index,
        implementation_closure=implementation_closure,
        qwen=qwen_evidence,
        minilm=minilm_evidence,
        gpu=gpu_evidence,
        sandbox=sandbox_receipt,
        sandbox_file_receipt=sandbox_file_receipt,
        process_status=process_status,
        network_canary=network_canary,
        label_canary=label_canary,
        execution_freeze_sha256=expected_execution_freeze_sha256,
    )
    runtime_raw = _canonical_bytes(runtime_receipt)
    runtime_file_sha256 = hashlib.sha256(runtime_raw).hexdigest()
    _publish_once(runtime_path, runtime_raw)
    _publish_once(
        sentinel_path,
        _canonical_bytes(
            {
                "expected_action_commitment_sha256": (
                    expected_action_commitment_sha256
                ),
                "expected_action_file_sha256": expected_action_file_sha256,
                "expected_execution_freeze_sha256": (
                    expected_execution_freeze_sha256
                ),
                "runtime_receipt_sha256": hashlib.sha256(
                    runtime_raw
                ).hexdigest(),
                "shard_count": SHARD_COUNT,
                "shard_index": shard_index,
                "study_id": study_id,
                "version": VERSION,
            }
        ),
    )
    action_release, action_release_file_receipt = _wait_for_action_release(
        action_release_path,
        study_id=study_id,
        shard_index=shard_index,
        own_runtime_file_sha256=runtime_file_sha256,
        expected_action_file_sha256=expected_action_file_sha256,
        expected_action_commitment_sha256=(
            expected_action_commitment_sha256
        ),
        expected_execution_freeze_sha256=(
            expected_execution_freeze_sha256
        ),
        expected_gpu_uuid=expected_gpu_uuid,
        expected_peer_gpu_uuid=expected_peer_gpu_uuid,
    )
    action_pack, action_file_receipt = _load_action_pack(
        action_pack_path,
        study_id=study_id,
        expected_file_sha256=expected_action_file_sha256,
        expected_action_commitment_sha256=(
            expected_action_commitment_sha256
        ),
    )
    items = _selected_items(action_pack, shard_index=shard_index)

    descriptor = os.open(
        private_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    completed = 0
    document_calls = 0
    structural_failures = 0
    resource_totals = {
        "binder_receipt_count": 0,
        "bounded_set_receipt_count": 0,
        "candidate_count": 0,
        "document_envelope_receipt_count": 0,
        "forward_batch_count": 0,
        "graph_receipt_count": 0,
        "leaf_call_count": 0,
        "leaf_receipt_count": 0,
        "mapping_receipt_count": 0,
        "semantic_matrix_receipt_count": 0,
    }
    error_counts: dict[str, int] = {}
    digest = hashlib.sha256()
    private_size = 0
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            for ordinal, item in enumerate(items):
                formed = action.form_scar_cssm_item_action_v1(
                    item,
                    document_selector=lambda text: (
                        document_envelope.select_document_runtime_only(
                            text, runtime=runtime
                        )
                    ),
                    encoder=encoder,
                    encoder_binding_sha256=encoder_binding,
                )
                evidence = formed.pop("private_mechanism_receipts")
                record_body = {
                    "evidence": evidence,
                    "item_token": formed["item_token"],
                    "ordinal_within_shard": ordinal,
                    "prediction": formed,
                }
                record = {
                    **record_body,
                    "self_sha256": _content_hash(record_body),
                }
                raw = _canonical_bytes(record) + b"\n"
                if len(raw) > MAXIMUM_ITEM_LINE_BYTES:
                    raise ScarCssmWorkerError("WORKER_ITEM_OUTPUT_TOO_LARGE")
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
                digest.update(raw)
                private_size += len(raw)
                completed += 1
                execution = formed["execution"]
                document_calls += int(execution["document_call_count"])
                structural_failures += (
                    execution["structural_status"] == "TYPED_FAILURE"
                )
                error_code = execution["error_code"]
                if error_code is not None:
                    error_counts[error_code] = error_counts.get(error_code, 0) + 1
                counts = _mechanism_resource_counts(evidence)
                for key, value in counts.items():
                    resource_totals[key] += value
        _fsync_directory(private_path.parent)
    except Exception:
        raise

    body = {
        "action_commitment_sha256": action_pack[
            "action_commitment_sha256"
        ],
        "action_pack_file_receipt": dict(action_file_receipt),
        "action_release_file_receipt": dict(action_release_file_receipt),
        "action_release_self_sha256": action_release["self_sha256"],
        "arm_ids": list(action.ARM_IDS),
        "document_call_count": document_calls,
        "encoder_binding_sha256": encoder_binding,
        "external_network_call_count": 0,
        "formal_label_pack_access_count": 0,
        "formal_scorer_access_count": 0,
        "item_count": completed,
        "mechanism_resource_totals": resource_totals,
        "output_root_receipt": dict(output_root_receipt),
        "private_records_file_sha256": digest.hexdigest(),
        "private_records_file_size_bytes": private_size,
        "runtime_receipt_self_sha256": runtime_receipt["self_sha256"],
        "runtime_receipt_file_sha256": hashlib.sha256(
            runtime_raw
        ).hexdigest(),
        "schema": SHARD_TERMINAL_SCHEMA,
        "shard_count": SHARD_COUNT,
        "shard_index": shard_index,
        "status": "complete",
        "structural_error_code_counts": error_counts,
        "structural_typed_failure_count": structural_failures,
        "study_id": study_id,
        "variant_names": list(action.VARIANT_NAMES),
        "version": VERSION,
    }
    terminal = {**body, "self_sha256": _content_hash(body)}
    _publish_once(terminal_path, _canonical_bytes(terminal))
    return MappingProxyType(terminal)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one SCAR CSSM action shard")
    parser.add_argument("--action-pack", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--shard-index", required=True, type=int, choices=(0, 1))
    parser.add_argument("--qwen-model-root", required=True, type=Path)
    parser.add_argument("--qwen-manifest", required=True, type=Path)
    parser.add_argument("--minilm-model-root", required=True, type=Path)
    parser.add_argument("--minilm-manifest", required=True, type=Path)
    parser.add_argument("--sandbox-receipt", required=True, type=Path)
    parser.add_argument("--action-release", required=True, type=Path)
    parser.add_argument("--forbidden-label-probe", required=True, type=Path)
    parser.add_argument("--expected-action-file-sha256", required=True)
    parser.add_argument("--expected-action-commitment-sha256", required=True)
    parser.add_argument(
        "--expected-implementation-closure-sha256", required=True
    )
    parser.add_argument("--expected-sandbox-receipt-sha256", required=True)
    parser.add_argument("--expected-execution-freeze-sha256", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-peer-gpu-uuid", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    terminal = run_shard(
        action_pack_path=arguments.action_pack,
        output_root=arguments.output_root,
        study_id=arguments.study_id,
        shard_index=arguments.shard_index,
        qwen_model_root=arguments.qwen_model_root,
        qwen_manifest_path=arguments.qwen_manifest,
        minilm_model_root=arguments.minilm_model_root,
        minilm_manifest_path=arguments.minilm_manifest,
        sandbox_receipt_path=arguments.sandbox_receipt,
        action_release_path=arguments.action_release,
        forbidden_label_probe_path=arguments.forbidden_label_probe,
        expected_action_file_sha256=arguments.expected_action_file_sha256,
        expected_action_commitment_sha256=(
            arguments.expected_action_commitment_sha256
        ),
        expected_implementation_closure_sha256=(
            arguments.expected_implementation_closure_sha256
        ),
        expected_sandbox_receipt_sha256=(
            arguments.expected_sandbox_receipt_sha256
        ),
        expected_execution_freeze_sha256=(
            arguments.expected_execution_freeze_sha256
        ),
        expected_gpu_uuid=arguments.expected_gpu_uuid,
        expected_peer_gpu_uuid=arguments.expected_peer_gpu_uuid,
    )
    print(terminal["self_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SHARD_COUNT",
    "SHARD_TERMINAL_SCHEMA",
    "RUNTIME_RECEIPT_SCHEMA",
    "SANDBOX_RECEIPT_SCHEMA",
    "ACTION_RELEASE_SCHEMA",
    "ScarCssmWorkerError",
    "VERSION",
    "run_shard",
]
