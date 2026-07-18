from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import socket
from typing import Any, Sequence

import pytest

from replication_runtime.multihoprag_ner_v1.binding import (
    ASSET_FILE_SHA256,
    ASSET_SELF_SHA256,
    ASSET_VERSION,
    CANARY_OUTPUT_SHA256,
    EXPECTED_AGGREGATION,
    EXPECTED_EXECUTION,
    EXPECTED_LABELS,
    EXPECTED_RUNTIME_VERSIONS,
    EXPECTED_SCOPE,
    MODEL_ARCHITECTURE,
    MODEL_FILES,
    MODEL_ID,
    MODEL_LICENSE,
    MODEL_REVISION,
    MODEL_TREE_SHA256,
    WEIGHTS_SHA256,
    _canonical_hash,
    _verify_model_tree,
    verify_runtime_binding,
)
from replication_runtime.multihoprag_ner_v1.contract import (
    CanonicalText,
    EntitySpan,
    MultiHopRAGNERError,
    canonical_json_line,
    decode_request,
    decode_response,
    encode_request,
    encode_response,
    synthetic_canary_inputs,
)
from replication_runtime.multihoprag_ner_v1.worker import (
    compute_synthetic_canary,
    merge_window_logits,
    network_disabled,
    tokenize_windows,
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _logits(label: int, score: float = 10.0) -> list[float]:
    values = [-10.0] * len(EXPECTED_LABELS)
    values[label] = score
    return values


def _materialize_fake_tree(root: Path) -> list[dict[str, object]]:
    root.mkdir()
    rows: list[dict[str, object]] = []
    for index, name in enumerate(MODEL_FILES):
        raw = f"synthetic asset file {index}: {name}\n".encode()
        (root / name).write_bytes(raw)
        rows.append({"path": name, "sha256": _sha256(raw), "size": len(raw)})
    return rows


def _asset(rows: list[dict[str, object]]) -> dict[str, Any]:
    weights = next(row for row in rows if row["path"] == "model.safetensors")
    payload: dict[str, Any] = {
        "aggregation": copy.deepcopy(EXPECTED_AGGREGATION),
        "asset_version": ASSET_VERSION,
        "deterministic_canary": {
            "generator_version": "multihoprag_ner_synthetic_16_v1",
            "input_count": 16,
            "input_sha256": _canonical_hash(list(synthetic_canary_inputs())),
            "multihoprag_rows_or_archives_accessed": False,
            "output_sha256": "0" * 64,
            "repeat_count": 2,
            "repeat_exact": True,
        },
        "execution": copy.deepcopy(EXPECTED_EXECUTION),
        "license": MODEL_LICENSE,
        "local_binding": {
            "runtime_required_file_count": len(MODEL_FILES),
            "runtime_required_paths": list(MODEL_FILES),
            "snapshot_file_count": len(rows),
            "snapshot_files": rows,
            "snapshot_size_bytes": sum(int(row["size"]) for row in rows),
            "snapshot_tree_sha256": _canonical_hash(rows),
        },
        "model": {
            "architecture": MODEL_ARCHITECTURE,
            "id2label": EXPECTED_LABELS,
            "model_id": MODEL_ID,
            "snapshot_revision": MODEL_REVISION,
            "weight_serialization": "safetensors",
            "weights_sha256": weights["sha256"],
        },
        "runtime_versions": copy.deepcopy(EXPECTED_RUNTIME_VERSIONS),
        "scope": copy.deepcopy(EXPECTED_SCOPE),
    }
    payload["asset_sha256"] = _canonical_hash(payload)
    return payload


def test_exact_wire_contract_canonicalizes_article_and_query_without_labels() -> None:
    values = [
        {"kind": "article", "title": "Title ", "body": " Body"},
        {"kind": "query", "query": "Who is there? "},
    ]
    decoded = decode_request(encode_request(values))
    assert decoded == (
        CanonicalText(kind="article", text="Title \n\n Body"),
        CanonicalText(kind="query", text="Who is there? "),
    )
    for forbidden in ("answer", "evidence", "question_type", "url", "source"):
        contaminated = [{**values[1], forbidden: "must-not-cross"}]
        with pytest.raises(MultiHopRAGNERError, match="exactly article"):
            encode_request(contaminated)
    noncanonical = json.dumps(json.loads(encode_request(values))).encode("ascii") + b"\n"
    with pytest.raises(MultiHopRAGNERError, match="not canonical"):
        decode_request(noncanonical)
    assert decode_request(
        encode_request([{"kind": "article", "title": "Title", "body": ""}])
    ) == (CanonicalText(kind="article", text="Title\n\n"),)


def test_response_is_exact_offset_bound_sorted_and_nonoverlapping() -> None:
    source = "Ada met ACME."
    rows = (
        (
            EntitySpan("PER", 0, 3, "Ada"),
            EntitySpan("ORG", 8, 12, "ACME"),
        ),
    )
    assert decode_response(encode_response(rows), canonical_texts=[source]) == rows
    bad = {
        "entities": [
            [
                {"end": 3, "entity_type": "PER", "start": 0, "text": "Eve"},
            ]
        ],
        "schema": "multihoprag_ner_response_v1",
    }
    with pytest.raises(MultiHopRAGNERError, match="exact offsets"):
        decode_response(canonical_json_line(bad), canonical_texts=[source])


def test_per_character_overlap_pooling_and_deterministic_bio_merge() -> None:
    text = "John Doe met Acme."
    offsets = [
        [[0, 0], [0, 4], [5, 8], [9, 12], [13, 17], [17, 18]],
        [[0, 0], [0, 4], [5, 8], [0, 0], [0, 0], [0, 0]],
    ]
    logits = [
        [
            _logits(0),
            _logits(3, 8.0),
            _logits(4, 8.0),
            _logits(0, 8.0),
            _logits(5, 8.0),
            _logits(0, 8.0),
        ],
        [
            _logits(0),
            _logits(3, 8.0),  # exact tie: earlier window wins
            _logits(6, 7.0),  # lower overlap cannot replace I-PER
            _logits(0),
            _logits(0),
            _logits(0),
        ],
    ]
    assert merge_window_logits(
        text=text, window_offsets=offsets, window_logits=logits
    ) == (
        EntitySpan("PER", 0, 8, "John Doe"),
        EntitySpan("ORG", 13, 17, "Acme"),
    )

    # A higher-logit overlapping O wins only characters 2:4, proving that the
    # aggregation is per character rather than first/last-window selection.
    clipped = merge_window_logits(
        text="John",
        window_offsets=[[[0, 4]], [[2, 4]]],
        window_logits=[[[*_logits(3, 5.0)]], [[*_logits(0, 6.0)]]],
    )
    assert clipped == (EntitySpan("PER", 0, 2, "Jo"),)


def test_orphan_I_starts_and_B_always_splits() -> None:
    orphan = merge_window_logits(
        text="Paris",
        window_offsets=[[[0, 5]]],
        window_logits=[[[_logits(8, 5.0)[index] for index in range(9)]]],
    )
    assert orphan == (EntitySpan("LOC", 0, 5, "Paris"),)
    split = merge_window_logits(
        text="Ann Bob",
        window_offsets=[[[0, 3], [4, 7]]],
        window_logits=[[[_logits(3)[i] for i in range(9)], [_logits(3)[i] for i in range(9)]]],
    )
    assert split == (
        EntitySpan("PER", 0, 3, "Ann"),
        EntitySpan("PER", 4, 7, "Bob"),
    )
    explicit_o_space = merge_window_logits(
        text="Ann Bob",
        window_offsets=[[[0, 3], [3, 4], [4, 7]]],
        window_logits=[[_logits(3), _logits(0), _logits(4)]],
    )
    assert explicit_o_space == (
        EntitySpan("PER", 0, 3, "Ann"),
        EntitySpan("PER", 4, 7, "Bob"),
    )


def test_tokenizer_contract_is_exact_512_with_64_overlap() -> None:
    class FakeTokenizer:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None

        def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
            assert text == "synthetic text"
            self.kwargs = dict(kwargs)
            offsets = [[0, 0]] * 512
            offsets[1] = [0, 9]
            return {
                "attention_mask": [[1, 1] + [0] * 510],
                "input_ids": [[101, 42, 102] + [0] * 509],
                "offset_mapping": [offsets],
                "overflow_to_sample_mapping": [0],
                "token_type_ids": [[0] * 512],
            }

    tokenizer = FakeTokenizer()
    inputs, offsets = tokenize_windows(tokenizer, "synthetic text")
    assert len(inputs["input_ids"][0]) == len(offsets[0]) == 512
    assert tokenizer.kwargs == {
        "add_special_tokens": True,
        "max_length": 512,
        "padding": "max_length",
        "return_attention_mask": True,
        "return_offsets_mapping": True,
        "return_overflowing_tokens": True,
        "stride": 64,
        "truncation": True,
    }


def test_complete_six_file_tree_self_hash_and_drift_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import replication_runtime.multihoprag_ner_v1.binding as binding

    root = tmp_path / "model"
    rows = _materialize_fake_tree(root)
    asset = _asset(rows)
    manifest = tmp_path / "asset.json"
    manifest.write_text(json.dumps(asset, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(binding, "ASSET_FILE_SHA256", _sha256(manifest.read_bytes()))
    monkeypatch.setattr(binding, "ASSET_SELF_SHA256", asset["asset_sha256"])
    monkeypatch.setattr(
        binding, "MODEL_TREE_SHA256", asset["local_binding"]["snapshot_tree_sha256"]
    )
    monkeypatch.setattr(binding, "WEIGHTS_SHA256", asset["model"]["weights_sha256"])
    monkeypatch.setattr(
        binding,
        "CANARY_OUTPUT_SHA256",
        asset["deterministic_canary"]["output_sha256"],
    )
    receipt = verify_runtime_binding(
        asset_manifest_path=manifest,
        model_root=root,
        verify_package_versions=False,
    )
    assert receipt["model_tree_sha256"] == _canonical_hash(rows)
    (root / "extra.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(MultiHopRAGNERError, match="file set drifted"):
        _verify_model_tree(asset, root)
    (root / "extra.json").unlink()
    link = root / "extra-link"
    link.symlink_to(root / "config.json")
    with pytest.raises(MultiHopRAGNERError, match="symlink"):
        _verify_model_tree(asset, root)


def test_manifest_normative_drift_and_manifest_symlink_fail_closed(tmp_path: Path) -> None:
    project = Path(__file__).parents[1]
    root = project / "artifacts/multihoprag_ner_runtime_v1/model"
    asset_path = project / "manifests/multihoprag_ner_runtime_asset_v1.json"
    asset = json.loads(asset_path.read_text(encoding="utf-8"))
    asset["execution"]["stride"] = 63
    # Re-self-hashing must not make a semantically changed manifest acceptable.
    asset.pop("asset_sha256")
    asset["asset_sha256"] = _canonical_hash(asset)
    manifest = tmp_path / "drift.json"
    manifest.write_text(json.dumps(asset, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(MultiHopRAGNERError, match="manifest file drifted"):
        verify_runtime_binding(
            asset_manifest_path=manifest,
            model_root=root,
            verify_package_versions=False,
        )
    alias = tmp_path / "manifest-link.json"
    alias.symlink_to(asset_path)
    with pytest.raises(MultiHopRAGNERError, match="symlink"):
        verify_runtime_binding(
            asset_manifest_path=alias,
            model_root=root,
            verify_package_versions=False,
        )


def test_fixed_trust_root_rejects_self_consistent_replacement(tmp_path: Path) -> None:
    project = Path(__file__).parents[1]
    raw = (project / "manifests/multihoprag_ner_runtime_asset_v1.json").read_bytes()
    assert _sha256(raw) == ASSET_FILE_SHA256
    production = json.loads(raw)
    assert production["asset_sha256"] == ASSET_SELF_SHA256
    assert production["local_binding"]["snapshot_tree_sha256"] == MODEL_TREE_SHA256
    assert production["model"]["weights_sha256"] == WEIGHTS_SHA256
    assert production["deterministic_canary"]["output_sha256"] == CANARY_OUTPUT_SHA256

    root = tmp_path / "replacement-model"
    rows = _materialize_fake_tree(root)
    replacement = _asset(rows)
    manifest = tmp_path / "replacement-asset.json"
    manifest.write_text(json.dumps(replacement, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(MultiHopRAGNERError, match="manifest file drifted"):
        verify_runtime_binding(
            asset_manifest_path=manifest,
            model_root=root,
            verify_package_versions=False,
        )


def test_row_free_canary_is_repeat_exact_and_manifest_ready() -> None:
    class EmptyExtractor:
        def extract_inputs(
            self, rows: Sequence[dict[str, str]]
        ) -> tuple[tuple[EntitySpan, ...], ...]:
            assert tuple(rows) == synthetic_canary_inputs()
            return tuple(() for _ in rows)

    receipt = compute_synthetic_canary(EmptyExtractor())
    assert receipt == {
        "generator_version": "multihoprag_ner_synthetic_16_v1",
        "input_count": 16,
        "input_sha256": _canonical_hash(list(synthetic_canary_inputs())),
        "multihoprag_rows_or_archives_accessed": False,
        "output_sha256": _canonical_hash([[] for _ in range(16)]),
        "repeat_count": 2,
        "repeat_exact": True,
    }


def test_network_guard_fails_closed_and_restores_socket() -> None:
    original = socket.socket
    with network_disabled():
        with pytest.raises(MultiHopRAGNERError, match="network access is forbidden"):
            socket.socket()
    assert socket.socket is original
