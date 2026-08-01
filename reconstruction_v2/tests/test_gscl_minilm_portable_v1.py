from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pytest

from replication_runtime.gscl_minilm_portable_v1 import (
    GSCL_MINILM_TARGET_SCHEMA,
    GSCLMiniLMPortableError,
    GSCLPortableOfflineMiniLMEncoder,
    build_target_manifest_qualification_only,
    write_target_manifest_qualification_only,
)
from replication_runtime.gscl_minilm_portable_v1 import binding
from replication_runtime.qasper_minilm_v1 import binding as frozen_v1


_FLOAT_HASH = hashlib.sha256(b"target float").hexdigest()
_QUANT_HASH = hashlib.sha256(b"target quant").hexdigest()


class _Tokenizer:
    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(text) + 2))}


class _Model:
    tokenizer = _Tokenizer()


class _Portable:
    instances: list["_Portable"] = []
    observed_float = _FLOAT_HASH

    def __init__(
        self,
        *,
        asset_manifest_path: object,
        model_root: object,
        run_canary: bool,
    ) -> None:
        assert asset_manifest_path == "/public/asset.json"
        assert model_root == "/public/model"
        assert run_canary is True
        self._model = _Model()
        self.runtime_receipt = {
            "asset_file_sha256": frozen_v1.ASSET_FILE_SHA256,
            "model_tree_sha256": frozen_v1.MODEL_TREE_SHA256,
            "runtime_versions": dict(frozen_v1.EXPECTED_RUNTIME_VERSIONS),
            "status": "verified_offline_immutable_qasper_minilm_runtime",
        }
        self.canary_receipt = {
            "schema": "qasper_minilm_portable_startup_canary_v2",
            "repeat_count": 2,
            "repeat_byte_exact": True,
            "repeat_elementwise_exact": True,
            "public_text_vector_sha256": (
                frozen_v1.CANARY_TEXT_VECTOR_SHA256
            ),
            "maximum_observed_row_l2_norm_error": 0.000000119,
            "observed_output_hashes": {
                "compared_to_expected_or_allowlist": False,
                "float32_little_endian_c_order_sha256": (
                    type(self).observed_float
                ),
                "normative_acceptance": False,
                "quantized_embedding_matrix_sha256": _QUANT_HASH,
            },
        }
        type(self).instances.append(self)

    def encode(self, texts: object) -> np.ndarray:
        matrix = np.zeros((len(tuple(texts)), 384), dtype=np.float32)
        matrix[:, 0] = 1.0
        return matrix

    def query_paragraph_similarities(
        self, query: str, paragraphs: object
    ) -> tuple[int, ...]:
        return tuple(1 for _ in paragraphs)


def _runtime_closure() -> dict[str, object]:
    distributions = [
        {
            "closure_algorithm": "all_declared_files",
            "content_closure_sha256": hashlib.sha256(
                b"distributions"
            ).hexdigest(),
            "distribution": "torch",
            "version": frozen_v1.EXPECTED_RUNTIME_VERSIONS["torch"],
        }
    ]
    return {
        "critical_distributions": distributions,
        "critical_distribution_content_closure_sha256": (
            hashlib.sha256(b"closure").hexdigest()
        ),
        "environment_allowlist": {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "interpreter": {
            "version": "test",
            "sha256": hashlib.sha256(b"python").hexdigest(),
        },
        "platform": {"machine": "test"},
        "cpu": {"architecture": "test"},
        "torch_build": {"version": "test"},
    }


@pytest.fixture(autouse=True)
def _fake_target_runtime(monkeypatch: pytest.MonkeyPatch):
    _Portable.instances.clear()
    _Portable.observed_float = _FLOAT_HASH
    monkeypatch.setattr(
        binding, "PortableOfflineMiniLMEncoder", _Portable
    )
    monkeypatch.setattr(binding, "_runtime_closure", _runtime_closure)


@pytest.fixture
def secure_tmp_path() -> Path:
    path = Path(tempfile.mkdtemp(prefix="gscl-minilm-", dir="/var/tmp"))
    path.chmod(0o700)
    try:
        yield path
    finally:
        shutil.rmtree(path)


def _write(path: Path, raw: bytes, *, mode: int = 0o600) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode
    )
    try:
        os.write(descriptor, raw)
    finally:
        os.close(descriptor)
    os.chmod(path, mode)


def test_qualification_manifest_is_canonical_target_local_and_row_free() -> None:
    raw = build_target_manifest_qualification_only(
        asset_manifest_path="/public/asset.json",
        model_root="/public/model",
    )
    value = json.loads(raw.decode("ascii"))
    assert raw == (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    assert value["schema"] == GSCL_MINILM_TARGET_SCHEMA
    assert value["formal_source_or_rows_accessed"] is False
    assert value["labels_accessed"] is False
    assert value["network_calls"] == 0
    canary = value["public_synthetic_canary"]
    assert canary["target_observed_float32_sha256"] == _FLOAT_HASH
    assert canary["target_observed_quantized_sha256"] == _QUANT_HASH
    assert canary["target_repeat_byte_exact"] is True
    assert canary["cross_hardware_byte_identity_claimed"] is False
    assert canary["legacy_hashes_are_acceptance_oracle"] is False
    assert (
        canary["legacy_v1_float32_sha256_reference_only"]
        == frozen_v1.CANARY_FLOAT32_BYTES_SHA256
    )
    assert json.loads(value["portable_runtime_receipt_json"])[
        "status"
    ].startswith("verified_offline")
    assert json.loads(value["portable_canary_receipt_json"])[
        "repeat_count"
    ] == 2


def test_formal_encoder_recomputes_manifest_and_exposes_compatible_interface(
    secure_tmp_path: Path,
) -> None:
    path = secure_tmp_path / "target.json"
    receipt = write_target_manifest_qualification_only(
        target_manifest_path=path,
        asset_manifest_path="/public/asset.json",
        model_root="/public/model",
    )
    assert receipt["source_or_rows_accessed"] is False
    assert path.stat().st_mode & 0o777 == 0o600
    encoder = GSCLPortableOfflineMiniLMEncoder(
        asset_manifest_path="/public/asset.json",
        model_root="/public/model",
        target_manifest_path=path,
    )
    assert len(_Portable.instances) == 2
    assert encoder.runtime_receipt["status"].startswith(
        "verified_exact_gscl"
    )
    assert encoder.canary_receipt["repeat_count"] == 2
    assert encoder.canary_receipt[
        "cross_hardware_byte_identity_claimed"
    ] is False
    assert encoder.encode(("first", "second")).shape == (2, 384)
    assert encoder.query_paragraph_similarities(
        "query", ("one", "two")
    ) == (1_000_000, 1_000_000)


def test_canary_cannot_be_skipped_before_any_model_construction(
    secure_tmp_path: Path,
) -> None:
    with pytest.raises(
        GSCLMiniLMPortableError, match="cannot_be_skipped"
    ):
        GSCLPortableOfflineMiniLMEncoder(
            asset_manifest_path="/public/asset.json",
            model_root="/public/model",
            target_manifest_path=secure_tmp_path / "absent.json",
            run_canary=False,
        )
    assert not _Portable.instances


def test_formal_rejects_target_observation_or_runtime_drift(
    secure_tmp_path: Path,
) -> None:
    raw = build_target_manifest_qualification_only(
        asset_manifest_path="/public/asset.json",
        model_root="/public/model",
    )
    path = secure_tmp_path / "target.json"
    _write(path, raw)
    _Portable.observed_float = hashlib.sha256(b"drift").hexdigest()
    with pytest.raises(
        GSCLMiniLMPortableError,
        match="target_runtime_or_canary_drifted",
    ):
        GSCLPortableOfflineMiniLMEncoder(
            asset_manifest_path="/public/asset.json",
            model_root="/public/model",
            target_manifest_path=path,
        )


def test_formal_rejects_tamper_noncanonical_or_insecure_mode(
    secure_tmp_path: Path,
) -> None:
    raw = build_target_manifest_qualification_only(
        asset_manifest_path="/public/asset.json",
        model_root="/public/model",
    )
    insecure = secure_tmp_path / "insecure.json"
    _write(insecure, raw, mode=0o644)
    with pytest.raises(
        GSCLMiniLMPortableError, match="custody_invalid"
    ):
        GSCLPortableOfflineMiniLMEncoder(
            asset_manifest_path="/public/asset.json",
            model_root="/public/model",
            target_manifest_path=insecure,
        )

    value = json.loads(raw.decode("ascii"))
    value["labels_accessed"] = True
    tampered = secure_tmp_path / "tampered.json"
    _write(
        tampered,
        json.dumps(
            value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("ascii")
        + b"\n",
    )
    with pytest.raises(
        GSCLMiniLMPortableError, match="self_hash_invalid"
    ):
        GSCLPortableOfflineMiniLMEncoder(
            asset_manifest_path="/public/asset.json",
            model_root="/public/model",
            target_manifest_path=tampered,
        )

    noncanonical_value = json.loads(raw.decode("ascii"))
    noncanonical = secure_tmp_path / "noncanonical.json"
    _write(
        noncanonical,
        json.dumps(
            noncanonical_value, ensure_ascii=True, indent=2
        ).encode("ascii"),
    )
    with pytest.raises(
        GSCLMiniLMPortableError, match="not_canonical"
    ):
        GSCLPortableOfflineMiniLMEncoder(
            asset_manifest_path="/public/asset.json",
            model_root="/public/model",
            target_manifest_path=noncanonical,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repeat_byte_exact", False),
        ("repeat_elementwise_exact", False),
        ("repeat_count", 1),
    ],
)
def test_builder_rejects_nonexact_portable_canary(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    class InvalidPortable(_Portable):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self.canary_receipt[field] = value

    monkeypatch.setattr(
        binding, "PortableOfflineMiniLMEncoder", InvalidPortable
    )
    with pytest.raises(
        GSCLMiniLMPortableError,
        match="portable_canary_receipt_invalid",
    ):
        build_target_manifest_qualification_only(
            asset_manifest_path="/public/asset.json",
            model_root="/public/model",
        )


def test_declared_distribution_hasher_detects_content_and_tamper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = tmp_path / "first.py"
    second = tmp_path / "RECORD"
    first.write_bytes(b"alpha")
    second.write_bytes(b"beta")

    class Distribution:
        files = ("first.py", "RECORD")

        @staticmethod
        def locate_file(value: object) -> Path:
            return tmp_path / str(value)

    monkeypatch.setattr(
        binding.metadata,
        "distribution",
        lambda name: Distribution(),
    )
    original = binding._distribution_content_closure("synthetic")
    first.write_bytes(b"changed")
    changed = binding._distribution_content_closure("synthetic")
    assert original != changed


def test_declared_distribution_exact_duplicate_is_read_once_per_pass_and_retained(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    declared = tmp_path / "declared.py"
    record = tmp_path / "RECORD"
    declared.write_bytes(b"declared")
    record.write_bytes(b"record")

    class Distribution:
        files = ("declared.py", "declared.py", "RECORD")

        @staticmethod
        def locate_file(value: object) -> Path:
            return tmp_path / str(value)

    monkeypatch.setattr(
        binding.metadata,
        "distribution",
        lambda name: Distribution(),
    )
    original_observation = (
        binding._stable_distribution_file_observation
    )
    calls: dict[str, int] = {}

    def counted(
        path: Path, *, declared_text: str
    ) -> tuple[
        tuple[int, int, int, int, int],
        dict[str, object],
    ]:
        calls[declared_text] = calls.get(declared_text, 0) + 1
        return original_observation(
            path, declared_text=declared_text
        )

    monkeypatch.setattr(
        binding,
        "_stable_distribution_file_observation",
        counted,
    )
    observed = binding._distribution_content_closure("synthetic")
    declared_row = {
        "declared_path": "declared.py",
        "sha256": hashlib.sha256(b"declared").hexdigest(),
        "size": len(b"declared"),
    }
    record_row = {
        "declared_path": "RECORD",
        "sha256": hashlib.sha256(b"record").hexdigest(),
        "size": len(b"record"),
    }
    expected = hashlib.sha256(
        binding._canonical_bytes(
            [record_row, declared_row, declared_row]
        )
    ).hexdigest()
    assert observed == expected
    assert calls == {"RECORD": 2, "declared.py": 2}


def test_declared_distribution_rejects_different_names_for_one_inode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = tmp_path / "first.py"
    alias = tmp_path / "alias.py"
    first.write_bytes(b"one inode")
    os.link(first, alias)

    class Distribution:
        files = ("first.py", "alias.py")

        @staticmethod
        def locate_file(value: object) -> Path:
            return tmp_path / str(value)

    monkeypatch.setattr(
        binding.metadata,
        "distribution",
        lambda name: Distribution(),
    )
    with pytest.raises(
        GSCLMiniLMPortableError,
        match="critical_distribution_path_alias",
    ):
        binding._distribution_content_closure("synthetic")


def test_declared_distribution_rejects_cross_pass_content_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    declared = tmp_path / "declared.py"
    record = tmp_path / "RECORD"
    declared.write_bytes(b"before")
    record.write_bytes(b"record")

    class Distribution:
        files = ("declared.py", "RECORD")

        @staticmethod
        def locate_file(value: object) -> Path:
            return tmp_path / str(value)

    monkeypatch.setattr(
        binding.metadata,
        "distribution",
        lambda name: Distribution(),
    )
    original_observation = (
        binding._stable_distribution_file_observation
    )
    calls: dict[str, int] = {}

    def drift(
        path: Path, *, declared_text: str
    ) -> tuple[
        tuple[int, int, int, int, int],
        dict[str, object],
    ]:
        calls[declared_text] = calls.get(declared_text, 0) + 1
        if declared_text == "declared.py" and calls[declared_text] == 2:
            path.write_bytes(b"after")
        return original_observation(
            path, declared_text=declared_text
        )

    monkeypatch.setattr(
        binding,
        "_stable_distribution_file_observation",
        drift,
    )
    with pytest.raises(
        GSCLMiniLMPortableError,
        match="critical_distribution_file_changed_across_closure",
    ):
        binding._distribution_content_closure("synthetic")


def test_declared_distribution_hasher_rejects_unbound_import_origin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    declared = tmp_path / "declared.py"
    record = tmp_path / "RECORD"
    undeclared = tmp_path / "shadow.py"
    declared.write_bytes(b"declared")
    record.write_bytes(b"record")
    undeclared.write_bytes(b"shadow")

    class Distribution:
        files = ("declared.py", "RECORD")

        @staticmethod
        def locate_file(value: object) -> Path:
            return tmp_path / str(value)

    monkeypatch.setattr(
        binding.metadata,
        "distribution",
        lambda name: Distribution(),
    )
    with pytest.raises(
        GSCLMiniLMPortableError,
        match="critical_distribution_module_origin_unbound",
    ):
        binding._distribution_content_closure_with_origin(
            "synthetic", required_module_origin=undeclared
        )


def test_target_manifest_publish_is_write_once_and_same_inode_hardlink(
    monkeypatch: pytest.MonkeyPatch,
    secure_tmp_path: Path,
) -> None:
    target = secure_tmp_path / "target.json"
    original_link = binding.os.link
    links: list[tuple[Path, Path]] = []

    def observed_link(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        assert src_dir_fd is None
        assert dst_dir_fd is None
        links.append((Path(source), Path(destination)))
        original_link(
            source,
            destination,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(binding.os, "link", observed_link)
    write_target_manifest_qualification_only(
        target_manifest_path=target,
        asset_manifest_path="/public/asset.json",
        model_root="/public/model",
    )
    assert len(_Portable.instances) == 1
    original = target.read_bytes()
    assert len(links) == 1
    assert links[0][0].parent == target.parent
    assert links[0][0].name.startswith(".target.json.pending-")
    assert links[0][1] == target
    assert target.stat().st_nlink == 1
    with pytest.raises(
        GSCLMiniLMPortableError,
        match="target_manifest_parent_invalid",
    ):
        write_target_manifest_qualification_only(
            target_manifest_path=target,
            asset_manifest_path="/public/asset.json",
            model_root="/public/model",
        )
    assert len(_Portable.instances) == 1
    assert target.read_bytes() == original
    assert not tuple(secure_tmp_path.glob(".target.json.pending-*"))


def test_target_manifest_validation_failure_leaves_no_final_or_pending(
    monkeypatch: pytest.MonkeyPatch,
    secure_tmp_path: Path,
) -> None:
    target = secure_tmp_path / "target.json"

    def reject_pending(raw: bytes) -> dict[str, object]:
        raise GSCLMiniLMPortableError(
            "synthetic_pending_validation_failure"
        )

    monkeypatch.setattr(
        binding, "_decode_target_manifest", reject_pending
    )
    with pytest.raises(
        GSCLMiniLMPortableError,
        match="synthetic_pending_validation_failure",
    ):
        write_target_manifest_qualification_only(
            target_manifest_path=target,
            asset_manifest_path="/public/asset.json",
            model_root="/public/model",
        )
    assert not target.exists()
    assert not tuple(secure_tmp_path.glob(".target.json.pending-*"))


def test_formal_encoder_detects_nested_receipt_mutation(
    secure_tmp_path: Path,
) -> None:
    path = secure_tmp_path / "target.json"
    write_target_manifest_qualification_only(
        target_manifest_path=path,
        asset_manifest_path="/public/asset.json",
        model_root="/public/model",
    )
    encoder = GSCLPortableOfflineMiniLMEncoder(
        asset_manifest_path="/public/asset.json",
        model_root="/public/model",
        target_manifest_path=path,
    )
    closure = encoder.runtime_receipt["target_runtime_closure"]
    assert isinstance(closure, dict)
    closure["cpu"] = {"architecture": "tampered"}
    with pytest.raises(
        GSCLMiniLMPortableError, match="formal_encoder_binding_changed"
    ):
        encoder.validate_internal()
