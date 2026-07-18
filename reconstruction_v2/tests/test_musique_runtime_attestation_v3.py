from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from assumption_agent.models import stable_hash
from assumption_agent.benchmarks.musique_formal_runtime_binding_v3 import (
    ADAPTER_ID,
    prepare_formal_runtime_v3,
    validate_formal_runtime_binding_v3,
)
import replication_runtime.musique_official_hipporag_v1.runtime_attestation_v3 as v3
from replication_runtime.musique_official_hipporag_v1.contract import (
    MuSiQueOfficialHippoRAGError,
)


PROJECT = Path(__file__).parents[1]
V2_RECEIPT = PROJECT / "manifests/musique_official_hipporag_runtime_attestation_v2.json"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _install_small_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, timestamp: str = "1.25"
) -> tuple[Path, list[dict[str, Any]]]:
    rows = []
    root = tmp_path / "llm"
    metadata_root = root / ".cache" / "huggingface" / "download"
    metadata_root.mkdir(parents=True)
    (root / ".cache" / "huggingface" / ".gitignore").write_bytes(b"*")
    for index in range(11):
        name = f"payload-{index:02d}.bin"
        raw = bytes([65 + index]) * (index + 1)
        etag = f"{index + 1:040x}"
        (root / name).write_bytes(raw)
        (metadata_root / f"{name}.metadata").write_text(
            f"{v3.MODEL_REVISION}\n{etag}\n{timestamp}\n", encoding="ascii"
        )
        rows.append(
            {"etag": etag, "path": name, "sha256": _sha256(raw), "size": len(raw)}
        )
    monkeypatch.setattr(v3, "EXPECTED_PAYLOAD_ROWS", tuple(rows))
    monkeypatch.setattr(v3, "EXPECTED_TOTAL_PAYLOAD_BYTES", sum(r["size"] for r in rows))
    monkeypatch.setattr(
        v3,
        "EXPECTED_NORMALIZED_ASSET_SHA256",
        stable_hash([{"path": r["path"], "sha256": r["sha256"]} for r in rows]),
    )
    monkeypatch.setattr(
        v3,
        "EXPECTED_NORMALIZED_TOPOLOGY_SHA256",
        stable_hash(
            [
                {
                    "content_sha256": r["sha256"],
                    "is_symlink": False,
                    "link_target_sha256": None,
                    "path": r["path"],
                }
                for r in rows
            ]
        ),
    )
    return root, rows


def _metadata_path(root: Path, payload_path: str) -> Path:
    return root / ".cache" / "huggingface" / "download" / f"{payload_path}.metadata"


def test_frozen_static_contract_is_self_consistent() -> None:
    v3._validate_static_contract()
    assert len(v3.EXPECTED_PAYLOAD_ROWS) == 11
    assert sum(row["size"] for row in v3.EXPECTED_PAYLOAD_ROWS) == 272_030_008
    assert stable_hash(v3._payload_asset_rows()) == (
        "378d593e91d13e42da36365ab2e2092c50feec7aea76a3fe228cd0a50310f9f4"
    )
    assert stable_hash(v3._payload_topology_rows()) == (
        "e57d1cd3a05f7c7ce8600d8b5789366ba0a2e2394bb5d2a789a0708586d7451e"
    )


def test_timestamp_change_preserves_normalized_identity_and_is_not_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, rows = _install_small_contract(tmp_path, monkeypatch, timestamp="1.25")
    first = v3._inspect_normalized_llm(root)
    for index, row in enumerate(rows):
        _metadata_path(root, row["path"]).write_text(
            f"{v3.MODEL_REVISION}\n{row['etag']}\n{5000.5 + index}\n",
            encoding="ascii",
        )
    second = v3._inspect_normalized_llm(root)
    assert first == second
    assert first["download_timestamp_fields_persisted"] is False
    raw = json.dumps(first, sort_keys=True)
    assert "1.25" not in raw
    assert "5000.5" not in raw


@pytest.mark.parametrize(
    "tamper",
    [
        "commit",
        "etag",
        "timestamp_nan",
        "timestamp_negative",
        "payload_hash",
        "payload_size",
        "payload_path",
        "payload_symlink",
        "metadata_symlink",
        "extra_file",
        "lock_file",
        "temp_file",
        "extra_directory",
        "cache_control",
    ],
)
def test_normalized_tree_rejects_every_identity_or_topology_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    root, rows = _install_small_contract(tmp_path, monkeypatch)
    first = rows[0]
    metadata = _metadata_path(root, first["path"])
    if tamper == "commit":
        metadata.write_text(f"{'f' * 40}\n{first['etag']}\n1.25\n", encoding="ascii")
    elif tamper == "etag":
        metadata.write_text(
            f"{v3.MODEL_REVISION}\n{'e' * 40}\n1.25\n", encoding="ascii"
        )
    elif tamper == "timestamp_nan":
        metadata.write_text(
            f"{v3.MODEL_REVISION}\n{first['etag']}\nnan\n", encoding="ascii"
        )
    elif tamper == "timestamp_negative":
        metadata.write_text(
            f"{v3.MODEL_REVISION}\n{first['etag']}\n-1\n", encoding="ascii"
        )
    elif tamper == "payload_hash":
        (root / first["path"]).write_bytes(b"Z")
    elif tamper == "payload_size":
        (root / first["path"]).write_bytes(b"ZZ")
    elif tamper == "payload_path":
        (root / first["path"]).rename(root / "renamed.bin")
    elif tamper == "payload_symlink":
        payload = root / first["path"]
        payload.unlink()
        payload.symlink_to(root / rows[1]["path"])
    elif tamper == "metadata_symlink":
        metadata.unlink()
        metadata.symlink_to(_metadata_path(root, rows[1]["path"]))
    elif tamper == "extra_file":
        (root / "extra.bin").write_bytes(b"extra")
    elif tamper == "lock_file":
        metadata.with_suffix(".lock").write_bytes(b"")
    elif tamper == "temp_file":
        (metadata.parent / "download.incomplete").write_bytes(b"")
    elif tamper == "extra_directory":
        (root / "unexpected-empty-directory").mkdir()
    else:
        (root / ".cache" / "huggingface" / ".gitignore").write_bytes(b"!*")
    with pytest.raises(MuSiQueOfficialHippoRAGError):
        v3._inspect_normalized_llm(root)


def test_v3_adapter_is_ast_mechanically_equivalent_to_v2() -> None:
    v2_source = (
        PROJECT / "replication_runtime/musique_official_hipporag_v1/adapter_v2.py"
    ).read_text(encoding="utf-8")
    v3_source = (
        PROJECT / "replication_runtime/musique_official_hipporag_v1/adapter_v3.py"
    ).read_text(encoding="utf-8")
    normalized_v2 = ast.dump(ast.parse(v2_source.replace("v2", "vX")), include_attributes=False)
    normalized_v3 = ast.dump(ast.parse(v3_source.replace("v3", "vX")), include_attributes=False)
    assert normalized_v3 == normalized_v2


def test_immutable_v2_implementation_files_still_match_frozen_receipt() -> None:
    receipt = json.loads(V2_RECEIPT.read_text(encoding="utf-8"))
    for row in receipt["implementation_binding"]["files"]:
        assert _sha256((PROJECT / row["path"]).read_bytes()) == row["sha256"]


def _install_receipt_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    llm, _rows = _install_small_contract(tmp_path, monkeypatch)
    predecessor = json.loads(V2_RECEIPT.read_text(encoding="utf-8"))
    predecessor_raw = V2_RECEIPT.read_bytes()
    predecessor_snapshot = predecessor["runtime_filesystem_binding"]
    observed = json.loads(json.dumps(predecessor_snapshot))
    observed["local_llm_asset_sha256"] = "a" * 64
    observed["local_llm_topology_sha256"] = "b" * 64
    versions = {
        row["name"]: row["version"] for row in predecessor_snapshot["dependency_metadata_rows"]
    }
    base = {
        "qualification_binding": {
            "qualification_sha256": predecessor["base_binding"]["qualification_sha256"]
        },
        "receipt_sha256": predecessor["base_binding"]["receipt_sha256"],
        "runtime_binding": {"dependency_versions": versions},
        "schema": predecessor["base_binding"]["schema"],
    }
    monkeypatch.setattr(
        v3,
        "_load_v2_predecessor",
        lambda _path, *, project_root: (predecessor, predecessor_raw),
    )
    monkeypatch.setattr(
        v3.v2,
        "_base_binding",
        lambda _path, *, project_root: (base, predecessor["base_binding"]["file_sha256"]),
    )
    monkeypatch.setattr(
        v3.v2,
        "_filesystem_snapshot",
        lambda **_kwargs: json.loads(json.dumps(observed)),
    )
    runtime = tmp_path / "venv" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"python")
    os.chmod(runtime, 0o755)
    embedding = tmp_path / "embedding"
    embedding.mkdir()
    (embedding / "weights.bin").write_bytes(b"embedding")
    base_path = tmp_path / "manifests" / "base.json"
    base_path.parent.mkdir()
    base_path.write_text("{}\n", encoding="utf-8")
    receipt = v3.build_runtime_attestation_v3(
        project_root=PROJECT,
        v2_attestation_receipt_path=V2_RECEIPT,
        base_binding_receipt_path=base_path,
        runtime_python=runtime,
        local_llm_model=llm,
        local_embedding_model=embedding,
    )
    receipt_path = tmp_path / "attestation.v3.json"
    v3.write_attestation_exclusive(receipt_path, receipt)
    return runtime, llm, embedding, base_path, {"path": receipt_path, "payload": receipt}


def test_v3_receipt_preserves_every_non_llm_v2_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _runtime, _llm, _embedding, _base, receipt = _install_receipt_harness(
        tmp_path, monkeypatch
    )
    predecessor = json.loads(V2_RECEIPT.read_text(encoding="utf-8"))
    payload = receipt["payload"]
    for field in (
        "base_binding",
        "decision",
        "formal_entry_policy",
        "pre_freeze_executable_qualification",
    ):
        assert payload[field] == predecessor[field]
    for field in v3._NON_LLM_SNAPSHOT_KEYS:
        assert payload["runtime_filesystem_binding"][field] == predecessor[
            "runtime_filesystem_binding"
        ][field]


def test_non_llm_drift_fails_before_receipt_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm, _rows = _install_small_contract(tmp_path, monkeypatch)
    predecessor = json.loads(V2_RECEIPT.read_text(encoding="utf-8"))
    predecessor_snapshot = predecessor["runtime_filesystem_binding"]
    observed = json.loads(json.dumps(predecessor_snapshot))
    observed["official_source_tree_sha256"] = "f" * 64
    monkeypatch.setattr(v3.v2, "_filesystem_snapshot", lambda **_kwargs: observed)
    with pytest.raises(MuSiQueOfficialHippoRAGError, match="non-LLM-cache"):
        v3._live_snapshot(
            runtime_python=tmp_path / "python",
            local_llm_model=llm,
            local_embedding_model=tmp_path / "embedding",
            expected_versions={},
            predecessor_snapshot=predecessor_snapshot,
        )


def test_prepared_runtime_is_path_free_and_fresh_postflight_is_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, llm, embedding, base, receipt = _install_receipt_harness(
        tmp_path, monkeypatch
    )
    v3._CACHE.clear()
    prepared = prepare_formal_runtime_v3(
        project_root=PROJECT,
        attestation_receipt_path=receipt["path"],
        base_binding_receipt_path=base,
        runtime_python=runtime,
        local_llm_model=llm,
        local_embedding_model=embedding,
    )
    assert prepared.safe_binding["adapter_id"] == ADAPTER_ID
    assert prepared.safe_binding["formal_entry_executable_identity_probe_calls"] == 0
    assert prepared.safe_binding["formal_entry_subprocess_calls"] == 0
    assert str(tmp_path) not in json.dumps(prepared.safe_binding, sort_keys=True)
    assert validate_formal_runtime_binding_v3(prepared.safe_binding) == prepared.safe_binding
    assert prepared.fresh_reverify() == prepared.safe_binding


def test_fresh_postflight_rejects_metadata_identity_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, llm, embedding, base, receipt = _install_receipt_harness(
        tmp_path, monkeypatch
    )
    v3._CACHE.clear()
    prepared = prepare_formal_runtime_v3(
        project_root=PROJECT,
        attestation_receipt_path=receipt["path"],
        base_binding_receipt_path=base,
        runtime_python=runtime,
        local_llm_model=llm,
        local_embedding_model=embedding,
    )
    first = v3.EXPECTED_PAYLOAD_ROWS[0]
    _metadata_path(llm, first["path"]).write_text(
        f"{'f' * 40}\n{first['etag']}\n99.5\n", encoding="ascii"
    )
    with pytest.raises(MuSiQueOfficialHippoRAGError, match="metadata identity"):
        prepared.fresh_reverify()

