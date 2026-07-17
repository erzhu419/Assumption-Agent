from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from replication_runtime.qasper_minilm_v1 import (
    OfflineMiniLMEncoder,
    QasperMiniLMError,
    quantize_embeddings,
    quantized_cosine_similarity,
    query_paragraph_similarities,
    synthetic_canary_texts,
    verify_runtime_asset,
    verify_runtime_binding,
)
from replication_runtime.qasper_minilm_v1.binding import (
    ASSET_FILE_SHA256,
    ASSET_SELF_SHA256,
    CANARY_QUANTIZED_EMBEDDING_SHA256,
    CANARY_TEXT_VECTOR_SHA256,
    EMBEDDING_DIMENSION,
    MODEL_TREE_SHA256,
    _canonical_hash,
    _verify_model_tree,
)


PROJECT = Path(__file__).parents[1]
ASSET = PROJECT / "manifests/qasper_minilm_runtime_asset_v1.json"
MODEL = PROJECT / "artifacts/qasper_minilm_runtime_v1/model"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _small_tree(root: Path) -> dict[str, object]:
    (root / "1_Pooling").mkdir(parents=True)
    (root / "1_Pooling/config.json").write_bytes(b"{}\n")
    (root / "model.safetensors").write_bytes(b"safe-test-weight\n")
    rows = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        raw = path.read_bytes()
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(raw),
                "size": len(raw),
            }
        )
    return {
        "local_binding": {
            "snapshot_directories": ["1_Pooling"],
            "snapshot_file_count": len(rows),
            "snapshot_files": rows,
            "snapshot_size_bytes": sum(int(row["size"]) for row in rows),
            "snapshot_tree_sha256": _canonical_hash(rows),
        }
    }


def test_committed_manifest_self_hash_file_hash_and_row_free_scope() -> None:
    raw = ASSET.read_bytes()
    assert _sha256(raw) == ASSET_FILE_SHA256
    payload = json.loads(raw)
    declared = payload.pop("asset_sha256")
    assert declared == ASSET_SELF_SHA256 == _canonical_hash(payload)
    assert payload["scope"] == {
        "asset_freeze_only": True,
        "item_outcomes_or_performance_observed": False,
        "qasper_archives_accessed_by_asset_freeze": False,
        "qasper_rows_accessed_by_asset_freeze": False,
    }
    assert payload["local_binding"]["snapshot_tree_sha256"] == MODEL_TREE_SHA256
    assert payload["model"]["weight_serialization"] == "safetensors"


def test_complete_snapshot_verifier_rejects_extra_missing_and_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model"
    asset = _small_tree(root)
    # This unit fixture uses a synthetic weight, so bind the module constant to
    # that fixture while testing the complete-tree mechanics.
    import replication_runtime.qasper_minilm_v1.binding as binding

    expected_weight = asset["local_binding"]["snapshot_files"][1]["sha256"]
    original = binding.WEIGHTS_SHA256
    binding.WEIGHTS_SHA256 = expected_weight
    try:
        assert _verify_model_tree(asset, root) == root.absolute()
        extra = root / "extra.json"
        extra.write_bytes(b"{}\n")
        with pytest.raises(QasperMiniLMError, match="file set drifted"):
            _verify_model_tree(asset, root)
        extra.unlink()
        missing = root / "1_Pooling/config.json"
        missing.unlink()
        with pytest.raises(QasperMiniLMError, match="content drifted"):
            _verify_model_tree(asset, root)
        missing.write_bytes(b"{}\n")
        link = root / "unbound-link"
        link.symlink_to(root / "model.safetensors")
        with pytest.raises(QasperMiniLMError, match="symlink"):
            _verify_model_tree(asset, root)
    finally:
        binding.WEIGHTS_SHA256 = original


def test_synthetic_256_sentence_canary_preimage_and_integer_quantizer() -> None:
    texts = synthetic_canary_texts()
    assert len(texts) == 256
    assert len(set(texts)) == 256
    assert _canonical_hash(list(texts)) == CANARY_TEXT_VECTOR_SHA256
    matrix = np.zeros((1, EMBEDDING_DIMENSION), dtype=np.float32)
    matrix[0, :4] = np.asarray([0.125, -0.125, 0.5, -0.5], dtype=np.float32)
    quantized = quantize_embeddings(matrix)
    assert quantized[0][:4] == (125000, -125000, 500000, -500000)
    assert len(CANARY_QUANTIZED_EMBEDDING_SHA256) == 64
    with pytest.raises(QasperMiniLMError, match="shape"):
        quantize_embeddings(np.zeros((1, 3), dtype=np.float32))


def test_query_paragraph_similarity_is_ordered_quantized_cosine() -> None:
    query = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    same = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    opposite = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    orthogonal = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    query[0] = same[0] = 1.0
    opposite[0] = -1.0
    orthogonal[1] = 1.0
    assert quantized_cosine_similarity(query, same) == 1_000_000
    assert quantized_cosine_similarity(query, opposite) == -1_000_000
    assert quantized_cosine_similarity(query, orthogonal) == 0

    class FakeEncoder:
        def encode(self, texts: object) -> np.ndarray:
            assert texts == ("query", "same", "opposite", "orthogonal")
            return np.stack((query, same, opposite, orthogonal))

    assert query_paragraph_similarities(
        FakeEncoder(), "query", ["same", "opposite", "orthogonal"]
    ) == (1_000_000, -1_000_000, 0)


def test_bound_model_path_rejects_a_symlink_component(tmp_path: Path) -> None:
    if not MODEL.is_dir():
        pytest.skip("ignored pinned MiniLM snapshot is not materialized")
    alias = tmp_path / "model-alias"
    alias.symlink_to(MODEL, target_is_directory=True)
    with pytest.raises(QasperMiniLMError, match="symlink component"):
        verify_runtime_binding(asset_manifest_path=ASSET, model_root=alias)


def test_real_offline_asset_binding_and_repeated_canary() -> None:
    if not MODEL.is_dir():
        pytest.skip("ignored pinned MiniLM snapshot is not materialized")
    receipt = verify_runtime_asset(PROJECT)
    assert receipt["model_tree_sha256"] == MODEL_TREE_SHA256
    encoder = OfflineMiniLMEncoder(
        asset_manifest_path=ASSET,
        model_root=MODEL,
    )
    assert encoder.canary_receipt == {
        "float32_bytes_sha256": "e76f373bfc7c2b4f16b12d2841dc8d2ec0e0e93f8fe360c04a79062d628c5746",
        "quantized_embedding_matrix_sha256": CANARY_QUANTIZED_EMBEDDING_SHA256,
        "qasper_rows_or_archives_accessed_by_canary": False,
        "repeat_count": 2,
        "repeat_exact": True,
        "sentence_count": 256,
        "status": "passed_exact_row_free_synthetic_canary",
        "text_vector_sha256": CANARY_TEXT_VECTOR_SHA256,
    }
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    scores = encoder.query_paragraph_similarities(
        "What is a graph?",
        ["A graph has vertices and edges.", "A banana is yellow."],
    )
    assert len(scores) == 2
    assert scores[0] > scores[1]
