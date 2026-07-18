"""Runner-side binding for normalized MuSiQue official HippoRAG v3.

Preparation performs a fresh, subprocess-free v3 filesystem read and returns
an in-memory retrieve handle.  Only the path-free, self-hashed ``safe_binding``
may be serialized.  Postflight forces another fresh read and must reproduce the
same binding exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from ..models import stable_hash
from replication_runtime.musique_official_hipporag_v1.contract import (
    MuSiQueOfficialHippoRAGError,
)
from replication_runtime.musique_official_hipporag_v1.runtime_attestation_v3 import (
    ATTESTATION_SCHEMA,
    FORMAL_ENTRY_POLICY,
    verify_formal_runtime_attestation_v3,
)


BINDING_SCHEMA = "musique_formal_runtime_binding_v3"
ADAPTER_ID = "official_hipporag_retrieve_only_filesystem_attested_v3"
ADAPTER_MODULE = "replication_runtime.musique_official_hipporag_v1.adapter_v3"
ADAPTER_FUNCTION = "run_official_hipporag_retrieve_only_v3"
ADAPTER_PARAMETERS = (
    "question",
    "paragraphs",
    "runtime_python",
    "local_llm_model",
    "local_embedding_model",
    "base_binding_receipt_path",
    "attestation_receipt_path",
    "work_root",
    "timeout_seconds",
)
_BINDING_KEYS = frozenset(
    {
        "adapter_function",
        "adapter_id",
        "adapter_interface_sha256",
        "adapter_module",
        "adapter_source_sha256",
        "attestation_receipt_file_sha256",
        "attestation_receipt_sha256",
        "base_binding_receipt_file_sha256",
        "base_binding_receipt_sha256",
        "binding_sha256",
        "formal_entry_executable_identity_probe_calls",
        "formal_entry_policy_sha256",
        "formal_entry_subprocess_calls",
        "implementation_set_sha256",
        "normalized_llm_asset_binding_sha256",
        "predecessor_v2_attestation_receipt_sha256",
        "runtime_asset_paths_persisted",
        "runtime_filesystem_binding_sha256",
        "schema",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MuSiQueOfficialHippoRAGError(f"{field_name} must be lowercase sha256")
    return value


def _assert_no_symlink_components(path: Path, field_name: str) -> None:
    absolute = path.absolute()
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink():
            raise MuSiQueOfficialHippoRAGError(
                f"{field_name} contains a symbolic-link component"
            )


def _read_regular_bytes(path: Path, field_name: str) -> bytes:
    _assert_no_symlink_components(path, field_name)
    if not path.is_file():
        raise MuSiQueOfficialHippoRAGError(f"{field_name} is unavailable")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise MuSiQueOfficialHippoRAGError(f"{field_name} cannot be read") from exc


def _load_adapter_v3(project_root: Path) -> tuple[Callable[..., tuple[int, ...]], str, str]:
    module = importlib.import_module(ADAPTER_MODULE)
    function = getattr(module, ADAPTER_FUNCTION, None)
    if not callable(function):
        raise MuSiQueOfficialHippoRAGError("prospective v3 adapter is unavailable")
    expected_source = (
        project_root
        / "replication_runtime/musique_official_hipporag_v1/adapter_v3.py"
    ).resolve(strict=True)
    observed_source_raw = inspect.getsourcefile(function)
    if (
        function.__module__ != ADAPTER_MODULE
        or observed_source_raw is None
        or Path(observed_source_raw).resolve(strict=True) != expected_source
    ):
        raise MuSiQueOfficialHippoRAGError("prospective v3 adapter source drifted")
    signature = inspect.signature(function)
    if tuple(signature.parameters) != ADAPTER_PARAMETERS or any(
        parameter.kind is not inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    ):
        raise MuSiQueOfficialHippoRAGError("prospective v3 adapter interface drifted")
    interface = [
        {
            "default": (
                None if parameter.default is inspect.Parameter.empty else parameter.default
            ),
            "kind": parameter.kind.name,
            "name": name,
            "required": parameter.default is inspect.Parameter.empty,
        }
        for name, parameter in signature.parameters.items()
    ]
    return function, stable_hash(interface), _sha256_file(expected_source)


def validate_formal_runtime_binding_v3(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact path-free v3 runner binding and self-hash."""

    if set(payload) != _BINDING_KEYS:
        raise MuSiQueOfficialHippoRAGError("v3 formal runtime binding key set mismatch")
    body = dict(payload)
    declared = _require_sha256(body.pop("binding_sha256", None), "runtime binding hash")
    if payload.get("schema") != BINDING_SCHEMA or stable_hash(body) != declared:
        raise MuSiQueOfficialHippoRAGError("v3 formal runtime binding self-hash mismatch")
    if (
        payload.get("adapter_id") != ADAPTER_ID
        or payload.get("adapter_module") != ADAPTER_MODULE
        or payload.get("adapter_function") != ADAPTER_FUNCTION
        or payload.get("formal_entry_executable_identity_probe_calls") != 0
        or payload.get("formal_entry_subprocess_calls") != 0
        or payload.get("runtime_asset_paths_persisted") is not False
        or payload.get("formal_entry_policy_sha256") != stable_hash(FORMAL_ENTRY_POLICY)
    ):
        raise MuSiQueOfficialHippoRAGError("v3 formal runtime policy drifted")
    for key, value in payload.items():
        if key.endswith("sha256"):
            _require_sha256(value, key)
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    if "/tmp/" in raw or "/home/" in raw or "\\" in raw:
        raise MuSiQueOfficialHippoRAGError("v3 formal runtime binding contains a host path")
    return dict(payload)


def _safe_binding_from_verified(
    *,
    project_root: Path,
    attestation_path: Path,
    base_path: Path,
    verified: Mapping[str, Any],
) -> tuple[Callable[..., tuple[int, ...]], dict[str, Any]]:
    attestation_raw = _read_regular_bytes(attestation_path, "v3 attestation receipt")
    base_raw = _read_regular_bytes(base_path, "v1 base binding receipt")
    try:
        attestation_payload = json.loads(attestation_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MuSiQueOfficialHippoRAGError("v3 attestation receipt is invalid") from exc
    if (
        not isinstance(attestation_payload, Mapping)
        or attestation_payload.get("schema") != ATTESTATION_SCHEMA
    ):
        raise MuSiQueOfficialHippoRAGError("v3 attestation receipt schema mismatch")
    if verified.get("formal_entry_executable_identity_probe_calls") != 0:
        raise MuSiQueOfficialHippoRAGError("formal runtime verification attempted a probe")
    adapter, interface_hash, adapter_source_hash = _load_adapter_v3(project_root)
    body: dict[str, Any] = {
        "adapter_function": ADAPTER_FUNCTION,
        "adapter_id": ADAPTER_ID,
        "adapter_interface_sha256": interface_hash,
        "adapter_module": ADAPTER_MODULE,
        "adapter_source_sha256": adapter_source_hash,
        "attestation_receipt_file_sha256": _sha256_bytes(attestation_raw),
        "attestation_receipt_sha256": verified["attestation_receipt_sha256"],
        "base_binding_receipt_file_sha256": _sha256_bytes(base_raw),
        "base_binding_receipt_sha256": verified["base_binding_receipt_sha256"],
        "formal_entry_executable_identity_probe_calls": 0,
        "formal_entry_policy_sha256": stable_hash(FORMAL_ENTRY_POLICY),
        "formal_entry_subprocess_calls": 0,
        "implementation_set_sha256": verified["implementation_set_sha256"],
        "normalized_llm_asset_binding_sha256": verified[
            "normalized_llm_asset_binding_sha256"
        ],
        "predecessor_v2_attestation_receipt_sha256": verified[
            "predecessor_v2_attestation_receipt_sha256"
        ],
        "runtime_asset_paths_persisted": False,
        "runtime_filesystem_binding_sha256": verified[
            "runtime_filesystem_binding_sha256"
        ],
        "schema": BINDING_SCHEMA,
    }
    safe_binding = validate_formal_runtime_binding_v3(
        {**body, "binding_sha256": stable_hash(body)}
    )
    return adapter, safe_binding


@dataclass(frozen=True)
class PreparedFormalRuntimeV3:
    """In-memory v3 runtime handle; only ``safe_binding`` is serializable."""

    _safe_binding: Mapping[str, Any] = field(repr=False)
    _adapter: Callable[..., tuple[int, ...]] = field(repr=False)
    _project_root: Path = field(repr=False)
    _runtime_python: Path = field(repr=False)
    _local_llm_model: Path = field(repr=False)
    _local_embedding_model: Path = field(repr=False)
    _base_binding_receipt_path: Path = field(repr=False)
    _attestation_receipt_path: Path = field(repr=False)

    @property
    def safe_binding(self) -> dict[str, Any]:
        """Return a copy of the path-free pre-run frozen binding."""

        return dict(self._safe_binding)

    def retrieve(
        self,
        *,
        question: str,
        paragraphs: Sequence[Mapping[str, object]],
        work_root: Path,
    ) -> tuple[int, ...]:
        """Call only the source-verified v3 retrieve adapter."""

        result = self._adapter(
            question=question,
            paragraphs=paragraphs,
            runtime_python=self._runtime_python,
            local_llm_model=self._local_llm_model,
            local_embedding_model=self._local_embedding_model,
            base_binding_receipt_path=self._base_binding_receipt_path,
            attestation_receipt_path=self._attestation_receipt_path,
            work_root=work_root,
            timeout_seconds=900,
        )
        return tuple(result)

    def fresh_reverify(self) -> dict[str, Any]:
        """Re-attest post-run bytes without cache, retry, or score decisions."""

        verified = verify_formal_runtime_attestation_v3(
            project_root=self._project_root,
            attestation_receipt_path=self._attestation_receipt_path,
            base_binding_receipt_path=self._base_binding_receipt_path,
            runtime_python=self._runtime_python,
            local_llm_model=self._local_llm_model,
            local_embedding_model=self._local_embedding_model,
            bypass_cache=True,
        )
        _adapter, fresh_binding = _safe_binding_from_verified(
            project_root=self._project_root,
            attestation_path=self._attestation_receipt_path,
            base_path=self._base_binding_receipt_path,
            verified=verified,
        )
        if fresh_binding != dict(self._safe_binding):
            raise MuSiQueOfficialHippoRAGError(
                "fresh postflight binding differs from frozen pre-run binding"
            )
        return dict(fresh_binding)


def prepare_formal_runtime_v3(
    *,
    project_root: Path,
    attestation_receipt_path: Path,
    base_binding_receipt_path: Path,
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
) -> PreparedFormalRuntimeV3:
    """Verify v3 formal assets and return an explicit v3-only call path."""

    project_root = project_root.resolve(strict=True)
    attestation_path = attestation_receipt_path.absolute()
    base_path = base_binding_receipt_path.absolute()
    runtime_python = runtime_python.absolute()  # never resolve the venv symlink
    llm = local_llm_model.absolute()  # v3 rejects symlink components itself
    embedding = local_embedding_model.resolve(strict=True)
    verified = verify_formal_runtime_attestation_v3(
        project_root=project_root,
        attestation_receipt_path=attestation_path,
        base_binding_receipt_path=base_path,
        runtime_python=runtime_python,
        local_llm_model=llm,
        local_embedding_model=embedding,
        bypass_cache=True,
    )
    adapter, safe_binding = _safe_binding_from_verified(
        project_root=project_root,
        attestation_path=attestation_path,
        base_path=base_path,
        verified=verified,
    )
    return PreparedFormalRuntimeV3(
        _safe_binding=safe_binding,
        _adapter=adapter,
        _project_root=project_root,
        _runtime_python=runtime_python,
        _local_llm_model=llm,
        _local_embedding_model=embedding,
        _base_binding_receipt_path=base_path,
        _attestation_receipt_path=attestation_path,
    )


__all__ = [
    "ADAPTER_FUNCTION",
    "ADAPTER_ID",
    "ADAPTER_MODULE",
    "BINDING_SCHEMA",
    "PreparedFormalRuntimeV3",
    "prepare_formal_runtime_v3",
    "validate_formal_runtime_binding_v3",
]
