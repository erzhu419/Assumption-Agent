"""Normalized filesystem attestation for the official HippoRAG runtime.

The immutable v2 receipt accidentally made Hugging Face ``local_dir`` download
timestamps part of runtime identity.  This prospective version reuses every
v2 qualification and every non-LLM-cache filesystem field verbatim.  It binds
all eleven model payloads and each metadata commit/etag exactly, validates the
timestamp as a finite nonnegative float, and deliberately excludes only that
timestamp value from identity.  Formal verification performs filesystem reads
only and has no executable probe, subprocess, retry, or benchmark-row access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
from typing import Any, Mapping

from assumption_agent.models import stable_hash

from .contract import MuSiQueOfficialHippoRAGError
from . import runtime_attestation_v2 as v2


ATTESTATION_SCHEMA = "musique_official_hipporag_filesystem_attestation_v3"
IMPLEMENTATION_SCHEMA = "musique_official_hipporag_attestation_implementation_v3"
IMPLEMENTATION_RELATIVE_FILES = (
    "assumption_agent/benchmarks/musique_formal_runtime_binding_v3.py",
    "replication_runtime/musique_official_hipporag_v1/runtime_attestation_v3.py",
    "replication_runtime/musique_official_hipporag_v1/adapter_v3.py",
)
V2_ATTESTATION_FILENAME = "musique_official_hipporag_runtime_attestation_v2.json"
ATTESTATION_DECISION = v2.ATTESTATION_DECISION
FORMAL_ENTRY_POLICY: dict[str, Any] = dict(v2.FORMAL_ENTRY_POLICY)

MODEL_REPO_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
EXPECTED_NORMALIZED_ASSET_SHA256 = (
    "378d593e91d13e42da36365ab2e2092c50feec7aea76a3fe228cd0a50310f9f4"
)
EXPECTED_NORMALIZED_TOPOLOGY_SHA256 = (
    "e57d1cd3a05f7c7ce8600d8b5789366ba0a2e2394bb5d2a789a0708586d7451e"
)
EXPECTED_TOTAL_PAYLOAD_BYTES = 272_030_008
EXPECTED_CACHE_CONTROL_FILE: dict[str, Any] = {
    "path": ".cache/huggingface/.gitignore",
    "sha256": "684888c0ebb17f374298b65ee2807526c066094c701bcc7ebbe1c1095f494fc1",
    "size": 1,
}
EXPECTED_PAYLOAD_ROWS: tuple[dict[str, Any], ...] = (
    {
        "etag": "85c006f9a5e464072db2465b66ac20be695963d0",
        "path": "all_results.json",
        "sha256": "1f68a2e167194d764c414643b54f0960945dc3f46bc47ec53b02475df527f376",
        "size": 783,
    },
    {
        "etag": "36293b6099200eb8aeb55ae2c01bca2ba46d80d0",
        "path": "config.json",
        "sha256": "8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a",
        "size": 861,
    },
    {
        "etag": "b6c91c24d0db60c08b7a8be9692d83b5c071d5c7",
        "path": "eval_results.json",
        "sha256": "6c7ecd0c2c0e8f6af8cbb34065fb2d5eb4c504febf2e266a7c7f948f13c70b7c",
        "size": 586,
    },
    {
        "etag": "da6c4d71a43aa7e6f785bdbb28ea5025438a73fa",
        "path": "generation_config.json",
        "sha256": "87b916edaaab66b3899b9d0dd0752727dff6666686da0504d89ae0a6e055a013",
        "size": 132,
    },
    {
        "etag": "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c",
        "path": "model.safetensors",
        "sha256": "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c",
        "size": 269_060_552,
    },
    {
        "etag": "44719d2e365acac0637fd25a3acf46494ca45940",
        "path": "special_tokens_map.json",
        "sha256": "2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52",
        "size": 655,
    },
    {
        "etag": "f922b1797f0c88e71addc8393787831f2477a4bd",
        "path": "tokenizer.json",
        "sha256": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
        "size": 2_104_556,
    },
    {
        "etag": "8c7b22013909450429303ed10be4398bd63f5457",
        "path": "tokenizer_config.json",
        "sha256": "4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df",
        "size": 3_764,
    },
    {
        "etag": "246b9556e2d21f927f43ff1a0ff9447c5906b46f",
        "path": "train_results.json",
        "sha256": "713996ce10b9d99a440fec73c0f3bdb2f3c4db6adba3d027bdaa40796f059b2d",
        "size": 232,
    },
    {
        "etag": "6870fd88eb2bd4aa2a3c7d3ecb9214bfdddf64f1",
        "path": "trainer_state.json",
        "sha256": "9664b2f12a8a29cad27a12da7b0d3a1081c8ba1392bcec71fa3bfd4fafc11284",
        "size": 57_225,
    },
    {
        "etag": "0ad5ecc2035b7031b88afb544ee95e2d49baa484",
        "path": "vocab.json",
        "sha256": "82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79",
        "size": 800_662,
    },
)

_TOP_LEVEL_KEYS = frozenset(
    {
        "base_binding",
        "decision",
        "formal_entry_policy",
        "implementation_binding",
        "normalized_llm_asset_binding",
        "pre_freeze_executable_qualification",
        "predecessor_v2_attestation_binding",
        "receipt_sha256",
        "runtime_filesystem_binding",
        "schema",
    }
)
_PREDECESSOR_KEYS = frozenset(
    {
        "file_sha256",
        "implementation_set_sha256",
        "receipt_sha256",
        "schema",
    }
)
_IMPLEMENTATION_KEYS = frozenset({"schema", "files", "set_sha256"})
_IMPLEMENTATION_FILE_KEYS = frozenset({"path", "sha256"})
_NORMALIZED_KEYS = frozenset(
    {
        "cache_control_file",
        "download_timestamp_fields_persisted",
        "download_timestamps_validated",
        "metadata_file_count",
        "metadata_identity_rows",
        "metadata_identity_set_sha256",
        "model_revision",
        "normalized_asset_sha256",
        "normalized_topology_sha256",
        "payload_file_count",
        "payload_identity_set_sha256",
        "payload_rows",
        "repo_id",
        "total_payload_bytes",
        "unclassified_regular_file_count",
    }
)
_PAYLOAD_ROW_KEYS = frozenset({"etag", "path", "sha256", "size"})
_METADATA_ROW_KEYS = frozenset({"commit_hash", "etag", "path", "payload_path"})
_NON_LLM_SNAPSHOT_KEYS = frozenset(v2._SNAPSHOT_KEYS) - {
    "local_llm_asset_sha256",
    "local_llm_topology_sha256",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TIMESTAMP_RE = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_CACHE_LOCK = threading.Lock()
_CACHE: dict[tuple[str, ...], dict[str, Any]] = {}


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
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MuSiQueOfficialHippoRAGError(f"{field} is invalid") from exc
    if not isinstance(value, dict):
        raise MuSiQueOfficialHippoRAGError(f"{field} must be an object")
    return value, raw


def _assert_no_symlink_components(path: Path, field: str) -> None:
    absolute = path.absolute()
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink():
            raise MuSiQueOfficialHippoRAGError(f"{field} contains a symlink")


def _metadata_rows() -> list[dict[str, str]]:
    return [
        {
            "commit_hash": MODEL_REVISION,
            "etag": row["etag"],
            "path": f'.cache/huggingface/download/{row["path"]}.metadata',
            "payload_path": row["path"],
        }
        for row in EXPECTED_PAYLOAD_ROWS
    ]


def _payload_asset_rows() -> list[dict[str, str]]:
    return [
        {"path": row["path"], "sha256": row["sha256"]}
        for row in EXPECTED_PAYLOAD_ROWS
    ]


def _payload_topology_rows() -> list[dict[str, Any]]:
    return [
        {
            "content_sha256": row["sha256"],
            "is_symlink": False,
            "link_target_sha256": None,
            "path": row["path"],
        }
        for row in EXPECTED_PAYLOAD_ROWS
    ]


def _validate_static_contract() -> None:
    rows = list(EXPECTED_PAYLOAD_ROWS)
    if len(rows) != 11 or [row["path"] for row in rows] != sorted(
        row["path"] for row in rows
    ):
        raise MuSiQueOfficialHippoRAGError("normalized payload contract is malformed")
    for row in rows:
        _require_exact_keys(row, _PAYLOAD_ROW_KEYS, "normalized payload contract row")
        if (
            not isinstance(row.get("path"), str)
            or not row["path"]
            or Path(row["path"]).name != row["path"]
            or isinstance(row.get("size"), bool)
            or not isinstance(row.get("size"), int)
            or row["size"] <= 0
            or not isinstance(row.get("etag"), str)
            or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", row["etag"])
            is None
        ):
            raise MuSiQueOfficialHippoRAGError("normalized payload contract row drifted")
        _require_sha256(row.get("sha256"), "normalized payload hash")
    if (
        sum(row["size"] for row in rows) != EXPECTED_TOTAL_PAYLOAD_BYTES
        or stable_hash(_payload_asset_rows()) != EXPECTED_NORMALIZED_ASSET_SHA256
        or stable_hash(_payload_topology_rows())
        != EXPECTED_NORMALIZED_TOPOLOGY_SHA256
    ):
        raise MuSiQueOfficialHippoRAGError("normalized payload contract hash drifted")
    if EXPECTED_CACHE_CONTROL_FILE != {
        "path": ".cache/huggingface/.gitignore",
        "sha256": "684888c0ebb17f374298b65ee2807526c066094c701bcc7ebbe1c1095f494fc1",
        "size": 1,
    }:
        raise MuSiQueOfficialHippoRAGError("cache control contract drifted")


def _parse_metadata(path: Path, expected_etag: str) -> None:
    try:
        raw = path.read_bytes()
        text = raw.decode("ascii")
    except (OSError, UnicodeDecodeError) as exc:
        raise MuSiQueOfficialHippoRAGError("Hugging Face metadata is invalid") from exc
    if "\r" in text or not text.endswith("\n") or text.count("\n") != 3:
        raise MuSiQueOfficialHippoRAGError("Hugging Face metadata shape drifted")
    commit_hash, etag, timestamp_text = text[:-1].split("\n")
    if commit_hash != MODEL_REVISION or etag != expected_etag:
        raise MuSiQueOfficialHippoRAGError("Hugging Face metadata identity drifted")
    if _TIMESTAMP_RE.fullmatch(timestamp_text) is None:
        raise MuSiQueOfficialHippoRAGError("Hugging Face metadata timestamp is invalid")
    timestamp = float(timestamp_text)
    if not math.isfinite(timestamp) or timestamp < 0:
        raise MuSiQueOfficialHippoRAGError("Hugging Face metadata timestamp is invalid")


def _inspect_normalized_llm(local_llm_model: Path) -> dict[str, Any]:
    """Return a timestamp-free identity after an exact fail-closed tree read."""

    _validate_static_contract()
    root = local_llm_model.absolute()
    _assert_no_symlink_components(root, "local LLM model")
    if not root.is_dir():
        raise MuSiQueOfficialHippoRAGError("local LLM model is unavailable")
    entries = sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix())
    for entry in entries:
        if entry.is_symlink():
            raise MuSiQueOfficialHippoRAGError("local LLM model contains a symlink")
        if not entry.is_file() and not entry.is_dir():
            raise MuSiQueOfficialHippoRAGError(
                "local LLM model contains an unclassified entry"
            )
    observed_directories = {
        entry.relative_to(root).as_posix() for entry in entries if entry.is_dir()
    }
    expected_directories = {
        ".cache",
        ".cache/huggingface",
        ".cache/huggingface/download",
    }
    if observed_directories != expected_directories:
        raise MuSiQueOfficialHippoRAGError("local LLM directory topology drifted")
    observed_files = {
        entry.relative_to(root).as_posix() for entry in entries if entry.is_file()
    }
    expected_metadata = {row["path"] for row in _metadata_rows()}
    expected_payload = {row["path"] for row in EXPECTED_PAYLOAD_ROWS}
    expected_files = expected_payload | expected_metadata | {
        EXPECTED_CACHE_CONTROL_FILE["path"]
    }
    if observed_files != expected_files:
        raise MuSiQueOfficialHippoRAGError(
            "local LLM contains an extra, missing, lock, temp, or unclassified file"
        )
    for row in EXPECTED_PAYLOAD_ROWS:
        path = root / row["path"]
        if path.stat().st_size != row["size"] or _sha256_file(path) != row["sha256"]:
            raise MuSiQueOfficialHippoRAGError("local LLM payload identity drifted")
        _parse_metadata(
            root / f'.cache/huggingface/download/{row["path"]}.metadata',
            row["etag"],
        )
    cache_control = root / EXPECTED_CACHE_CONTROL_FILE["path"]
    if (
        cache_control.stat().st_size != EXPECTED_CACHE_CONTROL_FILE["size"]
        or _sha256_file(cache_control) != EXPECTED_CACHE_CONTROL_FILE["sha256"]
    ):
        raise MuSiQueOfficialHippoRAGError("Hugging Face cache control file drifted")
    metadata_rows = _metadata_rows()
    payload_rows = [dict(row) for row in EXPECTED_PAYLOAD_ROWS]
    return {
        "cache_control_file": dict(EXPECTED_CACHE_CONTROL_FILE),
        "download_timestamp_fields_persisted": False,
        "download_timestamps_validated": len(metadata_rows),
        "metadata_file_count": len(metadata_rows),
        "metadata_identity_rows": metadata_rows,
        "metadata_identity_set_sha256": stable_hash(metadata_rows),
        "model_revision": MODEL_REVISION,
        "normalized_asset_sha256": EXPECTED_NORMALIZED_ASSET_SHA256,
        "normalized_topology_sha256": EXPECTED_NORMALIZED_TOPOLOGY_SHA256,
        "payload_file_count": len(payload_rows),
        "payload_identity_set_sha256": stable_hash(payload_rows),
        "payload_rows": payload_rows,
        "repo_id": MODEL_REPO_ID,
        "total_payload_bytes": EXPECTED_TOTAL_PAYLOAD_BYTES,
        "unclassified_regular_file_count": 0,
    }


def current_v3_implementation_binding(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    rows = []
    for relative in IMPLEMENTATION_RELATIVE_FILES:
        path = project_root / relative
        if path.is_symlink() or not path.is_file():
            raise MuSiQueOfficialHippoRAGError(f"v3 implementation file missing: {relative}")
        rows.append({"path": relative, "sha256": _sha256_file(path)})
    return {
        "schema": IMPLEMENTATION_SCHEMA,
        "files": rows,
        "set_sha256": stable_hash(rows),
    }


def _load_v2_predecessor(
    path: Path, *, project_root: Path
) -> tuple[dict[str, Any], bytes]:
    payload, raw = _load_json_object(path, "v2 attestation receipt")
    v2._validate_receipt_structure(payload)
    if payload.get("implementation_binding") != v2.current_v2_implementation_binding(
        project_root
    ):
        raise MuSiQueOfficialHippoRAGError("immutable v2 implementation drifted")
    return payload, raw


def _predecessor_binding(payload: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    implementation = payload.get("implementation_binding")
    if not isinstance(implementation, Mapping):
        raise MuSiQueOfficialHippoRAGError("v2 implementation binding is unavailable")
    return {
        "file_sha256": _sha256_bytes(raw),
        "implementation_set_sha256": implementation["set_sha256"],
        "receipt_sha256": payload["receipt_sha256"],
        "schema": payload["schema"],
    }


def _live_snapshot(
    *,
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
    expected_versions: Mapping[str, object],
    predecessor_snapshot: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    observed = v2._filesystem_snapshot(
        runtime_python=runtime_python,
        local_llm_model=local_llm_model,
        local_embedding_model=local_embedding_model,
        expected_versions=expected_versions,
    )
    v2._validate_snapshot_shape(observed)
    for key in _NON_LLM_SNAPSHOT_KEYS:
        if observed.get(key) != predecessor_snapshot.get(key):
            raise MuSiQueOfficialHippoRAGError(
                f"v3 non-LLM-cache filesystem field differs from v2: {key}"
            )
    normalized = _inspect_normalized_llm(local_llm_model)
    observed["local_llm_asset_sha256"] = normalized["normalized_asset_sha256"]
    observed["local_llm_topology_sha256"] = normalized[
        "normalized_topology_sha256"
    ]
    v2._validate_snapshot_shape(observed)
    return observed, normalized


def _validate_normalized_binding(value: Mapping[str, Any]) -> None:
    _require_exact_keys(value, _NORMALIZED_KEYS, "normalized LLM binding")
    if (
        value.get("cache_control_file") != EXPECTED_CACHE_CONTROL_FILE
        or value.get("download_timestamp_fields_persisted") is not False
        or value.get("download_timestamps_validated") != 11
        or value.get("metadata_file_count") != 11
        or value.get("model_revision") != MODEL_REVISION
        or value.get("normalized_asset_sha256")
        != EXPECTED_NORMALIZED_ASSET_SHA256
        or value.get("normalized_topology_sha256")
        != EXPECTED_NORMALIZED_TOPOLOGY_SHA256
        or value.get("payload_file_count") != 11
        or value.get("repo_id") != MODEL_REPO_ID
        or value.get("total_payload_bytes") != EXPECTED_TOTAL_PAYLOAD_BYTES
        or value.get("unclassified_regular_file_count") != 0
    ):
        raise MuSiQueOfficialHippoRAGError("normalized LLM policy drifted")
    payload_rows = value.get("payload_rows")
    metadata_rows = value.get("metadata_identity_rows")
    if payload_rows != [dict(row) for row in EXPECTED_PAYLOAD_ROWS]:
        raise MuSiQueOfficialHippoRAGError("normalized payload rows drifted")
    if metadata_rows != _metadata_rows():
        raise MuSiQueOfficialHippoRAGError("normalized metadata identity rows drifted")
    if value.get("payload_identity_set_sha256") != stable_hash(payload_rows):
        raise MuSiQueOfficialHippoRAGError("normalized payload set hash mismatch")
    if value.get("metadata_identity_set_sha256") != stable_hash(metadata_rows):
        raise MuSiQueOfficialHippoRAGError("normalized metadata set hash mismatch")
    _require_sha256(value.get("payload_identity_set_sha256"), "payload set hash")
    _require_sha256(value.get("metadata_identity_set_sha256"), "metadata set hash")


def _validate_receipt_structure(payload: Mapping[str, Any]) -> None:
    _require_exact_keys(payload, _TOP_LEVEL_KEYS, "v3 attestation receipt")
    body = dict(payload)
    declared = _require_sha256(body.pop("receipt_sha256", None), "v3 receipt hash")
    if payload.get("schema") != ATTESTATION_SCHEMA or stable_hash(body) != declared:
        raise MuSiQueOfficialHippoRAGError("v3 attestation receipt self-hash mismatch")
    if payload.get("decision") != ATTESTATION_DECISION:
        raise MuSiQueOfficialHippoRAGError("v3 attestation decision mismatch")
    if payload.get("formal_entry_policy") != FORMAL_ENTRY_POLICY:
        raise MuSiQueOfficialHippoRAGError("v3 formal-entry policy drifted")
    implementation = payload.get("implementation_binding")
    predecessor = payload.get("predecessor_v2_attestation_binding")
    normalized = payload.get("normalized_llm_asset_binding")
    snapshot = payload.get("runtime_filesystem_binding")
    if not all(
        isinstance(value, Mapping)
        for value in (implementation, predecessor, normalized, snapshot)
    ):
        raise MuSiQueOfficialHippoRAGError("v3 attestation sections are incomplete")
    _require_exact_keys(implementation, _IMPLEMENTATION_KEYS, "v3 implementation")
    if implementation.get("schema") != IMPLEMENTATION_SCHEMA:
        raise MuSiQueOfficialHippoRAGError("v3 implementation schema drifted")
    files = implementation.get("files")
    if not isinstance(files, list) or len(files) != len(IMPLEMENTATION_RELATIVE_FILES):
        raise MuSiQueOfficialHippoRAGError("v3 implementation file set mismatch")
    for expected, row in zip(IMPLEMENTATION_RELATIVE_FILES, files):
        if not isinstance(row, Mapping):
            raise MuSiQueOfficialHippoRAGError("v3 implementation row is malformed")
        _require_exact_keys(row, _IMPLEMENTATION_FILE_KEYS, "v3 implementation row")
        if row.get("path") != expected:
            raise MuSiQueOfficialHippoRAGError("v3 implementation path drifted")
        _require_sha256(row.get("sha256"), "v3 implementation file hash")
    if implementation.get("set_sha256") != stable_hash(files):
        raise MuSiQueOfficialHippoRAGError("v3 implementation set hash mismatch")
    _require_exact_keys(predecessor, _PREDECESSOR_KEYS, "v2 predecessor binding")
    for field in ("file_sha256", "implementation_set_sha256", "receipt_sha256"):
        _require_sha256(predecessor.get(field), f"v2 predecessor {field}")
    if predecessor.get("schema") != v2.ATTESTATION_SCHEMA:
        raise MuSiQueOfficialHippoRAGError("v2 predecessor schema drifted")
    base = payload.get("base_binding")
    qualification = payload.get("pre_freeze_executable_qualification")
    if not isinstance(base, Mapping) or not isinstance(qualification, Mapping):
        raise MuSiQueOfficialHippoRAGError("inherited v2 evidence is unavailable")
    _validate_normalized_binding(normalized)
    v2._validate_snapshot_shape(snapshot)
    if (
        snapshot.get("local_llm_asset_sha256")
        != normalized.get("normalized_asset_sha256")
        or snapshot.get("local_llm_topology_sha256")
        != normalized.get("normalized_topology_sha256")
    ):
        raise MuSiQueOfficialHippoRAGError("normalized snapshot hashes drifted")


def build_runtime_attestation_v3(
    *,
    project_root: Path,
    v2_attestation_receipt_path: Path,
    base_binding_receipt_path: Path,
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
) -> dict[str, Any]:
    """Build v3 evidence without executing a probe, worker, or qualification."""

    project_root = project_root.resolve(strict=True)
    predecessor_path = v2_attestation_receipt_path.absolute()
    expected_predecessor_path = project_root / "manifests" / V2_ATTESTATION_FILENAME
    if predecessor_path != expected_predecessor_path:
        raise MuSiQueOfficialHippoRAGError("v2 predecessor path drifted")
    predecessor, predecessor_raw = _load_v2_predecessor(
        predecessor_path, project_root=project_root
    )
    base, base_file_hash = v2._base_binding(
        base_binding_receipt_path.absolute(), project_root=project_root
    )
    expected_base = {
        "file_sha256": base_file_hash,
        "qualification_sha256": base["qualification_binding"]["qualification_sha256"],
        "receipt_sha256": base["receipt_sha256"],
        "schema": base["schema"],
    }
    if predecessor.get("base_binding") != expected_base:
        raise MuSiQueOfficialHippoRAGError("v2 base binding drifted")
    predecessor_snapshot = predecessor.get("runtime_filesystem_binding")
    if not isinstance(predecessor_snapshot, Mapping):
        raise MuSiQueOfficialHippoRAGError("v2 filesystem snapshot is unavailable")
    snapshot, normalized = _live_snapshot(
        runtime_python=runtime_python.absolute(),
        local_llm_model=local_llm_model,
        local_embedding_model=local_embedding_model,
        expected_versions=base["runtime_binding"]["dependency_versions"],
        predecessor_snapshot=predecessor_snapshot,
    )
    receipt: dict[str, Any] = {
        "base_binding": dict(predecessor["base_binding"]),
        "decision": predecessor["decision"],
        "formal_entry_policy": dict(predecessor["formal_entry_policy"]),
        "implementation_binding": current_v3_implementation_binding(project_root),
        "normalized_llm_asset_binding": normalized,
        "pre_freeze_executable_qualification": dict(
            predecessor["pre_freeze_executable_qualification"]
        ),
        "predecessor_v2_attestation_binding": _predecessor_binding(
            predecessor, predecessor_raw
        ),
        "runtime_filesystem_binding": snapshot,
        "schema": ATTESTATION_SCHEMA,
    }
    receipt["receipt_sha256"] = stable_hash(receipt)
    _validate_receipt_structure(receipt)
    return receipt


def write_attestation_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist one validated public v3 receipt without an overwrite path."""

    _validate_receipt_structure(payload)
    destination = path.absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def verify_formal_runtime_attestation_v3(
    *,
    project_root: Path,
    attestation_receipt_path: Path,
    base_binding_receipt_path: Path,
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
    bypass_cache: bool = False,
) -> dict[str, Any]:
    """Verify normalized filesystem evidence, forcing a fresh read on request."""

    if not isinstance(bypass_cache, bool):
        raise MuSiQueOfficialHippoRAGError("bypass_cache must be boolean")
    project_root = project_root.resolve(strict=True)
    attestation_path = attestation_receipt_path.absolute()
    base_path = base_binding_receipt_path.absolute()
    predecessor_path = project_root / "manifests" / V2_ATTESTATION_FILENAME
    payload, attestation_raw = _load_json_object(attestation_path, "v3 attestation receipt")
    _validate_receipt_structure(payload)
    predecessor, predecessor_raw = _load_v2_predecessor(
        predecessor_path, project_root=project_root
    )
    if payload.get("predecessor_v2_attestation_binding") != _predecessor_binding(
        predecessor, predecessor_raw
    ):
        raise MuSiQueOfficialHippoRAGError("v2 predecessor binding drifted")
    if payload.get("base_binding") != predecessor.get("base_binding"):
        raise MuSiQueOfficialHippoRAGError("inherited v2 base binding drifted")
    if payload.get("decision") != predecessor.get("decision") or payload.get(
        "formal_entry_policy"
    ) != predecessor.get("formal_entry_policy"):
        raise MuSiQueOfficialHippoRAGError("inherited v2 formal policy drifted")
    if payload.get("pre_freeze_executable_qualification") != predecessor.get(
        "pre_freeze_executable_qualification"
    ):
        raise MuSiQueOfficialHippoRAGError("inherited v2 qualification drifted")
    base, base_file_hash = v2._base_binding(base_path, project_root=project_root)
    expected_base = {
        "file_sha256": base_file_hash,
        "qualification_sha256": base["qualification_binding"]["qualification_sha256"],
        "receipt_sha256": base["receipt_sha256"],
        "schema": base["schema"],
    }
    if payload.get("base_binding") != expected_base:
        raise MuSiQueOfficialHippoRAGError("v3 base binding drifted")
    if payload.get("implementation_binding") != current_v3_implementation_binding(
        project_root
    ):
        raise MuSiQueOfficialHippoRAGError("live v3 attestation implementation drifted")
    cache_key = (
        str(attestation_path),
        _sha256_bytes(attestation_raw),
        str(predecessor_path),
        _sha256_bytes(predecessor_raw),
        str(base_path),
        base_file_hash,
        str(runtime_python.absolute()),
        str(local_llm_model.absolute()),
        str(local_embedding_model.resolve(strict=True)),
    )
    with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
        if cached is not None and not bypass_cache:
            return dict(cached)
        if bypass_cache:
            _CACHE.pop(cache_key, None)
        predecessor_snapshot = predecessor.get("runtime_filesystem_binding")
        if not isinstance(predecessor_snapshot, Mapping):
            raise MuSiQueOfficialHippoRAGError("v2 filesystem snapshot is unavailable")
        observed, normalized = _live_snapshot(
            runtime_python=runtime_python.absolute(),
            local_llm_model=local_llm_model,
            local_embedding_model=local_embedding_model,
            expected_versions=base["runtime_binding"]["dependency_versions"],
            predecessor_snapshot=predecessor_snapshot,
        )
        if observed != payload.get("runtime_filesystem_binding"):
            raise MuSiQueOfficialHippoRAGError(
                "formal normalized filesystem runtime attestation drifted"
            )
        if normalized != payload.get("normalized_llm_asset_binding"):
            raise MuSiQueOfficialHippoRAGError("formal normalized LLM binding drifted")
        result = {
            "attestation_receipt_sha256": payload["receipt_sha256"],
            "base_binding_receipt_sha256": base["receipt_sha256"],
            "formal_entry_executable_identity_probe_calls": 0,
            "implementation_set_sha256": payload["implementation_binding"][
                "set_sha256"
            ],
            "normalized_llm_asset_binding_sha256": stable_hash(normalized),
            "predecessor_v2_attestation_receipt_sha256": predecessor[
                "receipt_sha256"
            ],
            "runtime_filesystem_binding_sha256": stable_hash(observed),
        }
        _CACHE[cache_key] = result
        return dict(result)


__all__ = [
    "ATTESTATION_SCHEMA",
    "FORMAL_ENTRY_POLICY",
    "build_runtime_attestation_v3",
    "current_v3_implementation_binding",
    "verify_formal_runtime_attestation_v3",
    "write_attestation_exclusive",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--v2-attestation-receipt", type=Path, required=True)
    parser.add_argument("--base-binding-receipt", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--local-llm-model", type=Path, required=True)
    parser.add_argument("--local-embedding-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    receipt = build_runtime_attestation_v3(
        project_root=arguments.project_root,
        v2_attestation_receipt_path=arguments.v2_attestation_receipt,
        base_binding_receipt_path=arguments.base_binding_receipt,
        runtime_python=arguments.runtime_python,
        local_llm_model=arguments.local_llm_model,
        local_embedding_model=arguments.local_embedding_model,
    )
    write_attestation_exclusive(arguments.output, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
