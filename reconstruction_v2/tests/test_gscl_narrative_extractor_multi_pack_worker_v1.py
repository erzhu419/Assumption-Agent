from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import pytest

from replication_runtime.gscl_narrative_extractor_v1 import contract
from replication_runtime.gscl_narrative_extractor_v1 import (
    multi_pack_worker as multi,
)


@pytest.fixture
def secure_tmp_path() -> Path:
    root = Path(
        tempfile.mkdtemp(prefix="gscl-multi-pack-", dir="/var/tmp")
    )
    root.chmod(0o700)
    try:
        yield root
    finally:
        shutil.rmtree(root)


def _manifest(root: Path) -> tuple[Path, bytes]:
    batch = root / "batch"
    batch.mkdir(mode=0o700)
    input_path = batch / "input.json"
    input_raw = b'{"synthetic":"source-free"}\n'
    input_path.write_bytes(input_raw)
    input_path.chmod(0o600)
    body = {
        "batches": [
            {
                "input_file_sha256": hashlib.sha256(
                    input_raw
                ).hexdigest(),
                "input_path": str(input_path),
                "output_path": str(batch / "output.json"),
                "sequence": 0,
            }
        ],
        "lineage": "source_free_qualification",
        "schema": multi.INPUT_MANIFEST_SCHEMA,
        "work_root": str(root),
    }
    envelope = {
        **body,
        "self_sha256": hashlib.sha256(
            contract.canonical_json_bytes(body, newline=False)
        ).hexdigest(),
    }
    raw = contract.canonical_json_bytes(envelope)
    path = root / "manifest.json"
    path.write_bytes(raw)
    path.chmod(0o600)
    return path, raw


def test_manifest_binds_exact_secure_paths(
    secure_tmp_path: Path,
) -> None:
    path, raw = _manifest(secure_tmp_path)
    root, batches, file_hash, lineage = multi._decode_manifest(path)
    assert root == secure_tmp_path
    assert len(batches) == 1
    assert batches[0]["sequence"] == 0
    assert file_hash == hashlib.sha256(raw).hexdigest()
    assert lineage == "source_free_qualification"


def test_manifest_rejects_output_escape(
    secure_tmp_path: Path,
) -> None:
    path, _ = _manifest(secure_tmp_path)
    value = json.loads(path.read_text(encoding="ascii"))
    body = {
        key: item for key, item in value.items() if key != "self_sha256"
    }
    body["batches"][0]["output_path"] = str(
        secure_tmp_path.parent / "escape.json"
    )
    value = {
        **body,
        "self_sha256": hashlib.sha256(
            contract.canonical_json_bytes(body, newline=False)
        ).hexdigest(),
    }
    path.write_bytes(contract.canonical_json_bytes(value))
    path.chmod(0o600)
    with pytest.raises(
        multi.MultiPackWorkerError,
        match="multi_pack_path_outside_work",
    ):
        multi._decode_manifest(path)


def test_safe_receipt_writer_is_same_directory_exclusive(
    secure_tmp_path: Path,
) -> None:
    path = secure_tmp_path / "safe.json"
    raw = b'{"safe":true}\n'
    assert multi._write_once(  # noqa: SLF001
        path, raw, work_root=secure_tmp_path
    ) == hashlib.sha256(raw).hexdigest()
    assert path.read_bytes() == raw
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(
        multi.MultiPackWorkerError,
        match="multi_pack_output_already_exists",
    ):
        multi._write_once(  # noqa: SLF001
            path, raw, work_root=secure_tmp_path
        )


def test_loaded_distribution_closure_includes_fixed_critical_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Distribution:
        metadata = {"Name": "fixture"}
        version = "1.0"

    monkeypatch.setattr(
        multi, "_CRITICAL_DISTRIBUTIONS", frozenset({"fixture"})
    )
    monkeypatch.setattr(
        multi.importlib.metadata,
        "packages_distributions",
        lambda: {},
    )
    monkeypatch.setattr(
        multi.importlib.metadata,
        "distribution",
        lambda _: Distribution(),
    )
    monkeypatch.setattr(
        multi.worker,
        "_distribution_closure_sha256",
        lambda _, **__: hashlib.sha256(b"fixture").hexdigest(),
    )
    assert multi._loaded_distribution_rows() == [
        {
            "closure_sha256": hashlib.sha256(
                b"fixture"
            ).hexdigest(),
            "critical": True,
            "distribution": "fixture",
            "loaded_top_level_modules": [],
            "version": "1.0",
        }
    ]


def test_loaded_distribution_closure_binds_actual_module_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Distribution:
        metadata = {"Name": "fixture"}
        version = "1.0"

    origin = Path("/tmp/fixture-shadow.py")
    observed: list[tuple[Path, ...]] = []
    monkeypatch.setattr(
        multi, "_CRITICAL_DISTRIBUTIONS", frozenset({"fixture"})
    )
    monkeypatch.setitem(
        multi.sys.modules,
        "fixture_module",
        SimpleNamespace(__file__=str(origin)),
    )
    monkeypatch.setattr(
        multi.importlib.metadata,
        "packages_distributions",
        lambda: {"fixture_module": ["fixture"]},
    )
    monkeypatch.setattr(
        multi.importlib.metadata,
        "distribution",
        lambda _: Distribution(),
    )

    def closure(
        _: str, *, required_module_origins: tuple[Path, ...]
    ) -> str:
        observed.append(required_module_origins)
        return hashlib.sha256(b"fixture").hexdigest()

    monkeypatch.setattr(
        multi.worker, "_distribution_closure_sha256", closure
    )
    rows = multi._loaded_distribution_rows()
    assert rows[0]["loaded_top_level_modules"] == ["fixture_module"]
    assert observed == [(origin,)]


def test_formal_entry_has_one_model_construction_and_no_injection() -> None:
    parameters = set(
        inspect.signature(multi.run_formal_multi_pack).parameters
    )
    assert parameters == {
        "input_manifest_path",
        "model_root",
        "model_manifest_path",
        "safe_receipt_path",
    }
    source = inspect.getsource(multi.run_formal_multi_pack)
    assert source.count("worker.LocalQwenRuntime(") == 1
    assert "worker.process_trusted_pack(" in source
    assert "runtime=" not in parameters
    assert "parser=" not in parameters
    assert "predictions=" not in parameters
