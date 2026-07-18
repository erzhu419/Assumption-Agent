"""Immutable local-asset verification for the MultiHopRAG NER runtime."""

from __future__ import annotations

import hashlib
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping

from .contract import MultiHopRAGNERError, synthetic_canary_inputs


ASSET_VERSION = "multihoprag_ner_runtime_asset_v1"
ASSET_RELATIVE_PATH = Path("manifests/multihoprag_ner_runtime_asset_v1.json")
# These values are the trust root, not values learned from the manifest being
# verified.  A replacement manifest therefore cannot authorize replacement
# weights merely by recomputing its own self-hash.
ASSET_FILE_SHA256 = "df197407a0daf8e0a82dfbd2076397fc12b12699c67f42d817e486d133526fe1"
ASSET_SELF_SHA256 = "b70ab3da9d01f0bc61650ddd8f81d27fdf01e434a1d67a0b378e226bd6b3b5c5"
MODEL_TREE_SHA256 = "204acdabd993ad0c3e9c2a4f039b4899c9710dde0f76b75c13cda6b6fd3e94a0"
WEIGHTS_SHA256 = "b04492186cfb45a64908487a17a9f8d6ddec3a403ef39db5bca688f0fa702a34"
CANARY_OUTPUT_SHA256 = "8a8856f1a55e00af8a77d7f6a8d6145aa0c1e99bde7758c6a62e08532c3c163e"
MODEL_ID = "dslim/bert-base-NER"
MODEL_REVISION = "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc"
MODEL_ARCHITECTURE = "BertForTokenClassification"
MODEL_LICENSE = "MIT"
MODEL_FILES = (
    "added_tokens.json",
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.txt",
)
EXPECTED_LABELS = {
    "0": "O",
    "1": "B-MISC",
    "2": "I-MISC",
    "3": "B-PER",
    "4": "I-PER",
    "5": "B-ORG",
    "6": "I-ORG",
    "7": "B-LOC",
    "8": "I-LOC",
}
MAXIMUM_SEQUENCE_LENGTH = 512
WINDOW_OVERLAP = 64
INFERENCE_WINDOW_BATCH_SIZE = 16
EXPECTED_EXECUTION = {
    "backend": "torch",
    "canonical_article_formula": "title + '\\n\\n' + body",
    "canonical_query_formula": "query",
    "device": "cpu",
    "dtype": "float32",
    "environment": {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    },
    "eval_mode": True,
    "inference_window_batch_size": INFERENCE_WINDOW_BATCH_SIZE,
    "local_files_only": True,
    "maximum_sequence_length": MAXIMUM_SEQUENCE_LENGTH,
    "network_calls": 0,
    "offset_mapping": True,
    "padding": "max_length",
    "return_overflowing_tokens": True,
    "stride": WINDOW_OVERLAP,
    "torch_deterministic_algorithms": True,
    "torch_inference_mode": True,
    "torch_interop_threads": 1,
    "torch_manual_seed": 0,
    "torch_num_threads": 1,
    "truncation": True,
    "trust_remote_code": False,
    "use_fast_tokenizer": True,
    "use_safetensors": True,
}
EXPECTED_AGGREGATION = {
    "bio_orphan_I_rule": "start_new_span_of_I_type",
    "bio_rule": "B_always_starts; matching_I_continues_across_only_uncovered_whitespace",
    "character_candidate": "highest_raw_float32_logit_across_all_covering_window_tokens_and_labels",
    "character_tie_break": "smaller_label_id_then_earlier_window_then_earlier_token",
    "entity_types": ["PER", "ORG", "LOC", "MISC"],
    "non_wordpiece_character_rule": "O_unless_whitespace_between_active_span_and_matching_I",
    "output_order": "start_then_end_then_entity_type",
    "special_padding_offset": [0, 0],
}
EXPECTED_RUNTIME_VERSIONS = {
    "huggingface_hub": "1.11.0",
    "python": "3.10.12",
    "safetensors": "0.7.0",
    "tokenizers": "0.22.2",
    "torch": "2.8.0+cu128",
    "transformers": "5.10.1",
}
EXPECTED_SCOPE = {
    "asset_freeze_only": True,
    "item_outcomes_or_performance_observed": False,
    "multihoprag_archives_accessed_by_asset_freeze": False,
    "multihoprag_rows_accessed_by_asset_freeze": False,
}
_PACKAGE_TO_MODULE = {
    "huggingface_hub": ("huggingface-hub", "huggingface_hub"),
    "safetensors": ("safetensors", "safetensors"),
    "tokenizers": ("tokenizers", "tokenizers"),
    "torch": ("torch", "torch"),
    "transformers": ("transformers", "transformers"),
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise MultiHopRAGNERError("value is not canonical-JSON representable") from exc
    return _sha256_bytes(raw)


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise MultiHopRAGNERError(f"{field} must be lowercase sha256")
    return value


def _reject_symlink_components(path: Path, field: str) -> Path:
    absolute = path.expanduser().absolute()
    cursor = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor = cursor / component
        if cursor.is_symlink():
            raise MultiHopRAGNERError(f"{field} contains a symlink component")
    return absolute


def _load_asset_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = _reject_symlink_components(Path(path), "asset manifest path")
    if not manifest_path.is_file() or manifest_path.stat().st_size > 256 * 1024:
        raise MultiHopRAGNERError("asset manifest is unavailable or oversized")
    raw = manifest_path.read_bytes()
    if _sha256_bytes(raw) != ASSET_FILE_SHA256:
        raise MultiHopRAGNERError("committed asset manifest file drifted")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MultiHopRAGNERError("asset manifest is invalid") from exc
    if not isinstance(value, dict) or value.get("asset_version") != ASSET_VERSION:
        raise MultiHopRAGNERError("asset manifest version mismatch")
    body = dict(value)
    declared = body.pop("asset_sha256", None)
    if (
        _require_sha256(declared, "asset_sha256") != ASSET_SELF_SHA256
        or _canonical_hash(body) != ASSET_SELF_SHA256
    ):
        raise MultiHopRAGNERError("asset manifest self-hash mismatch")
    return manifest_path, value


def _verify_manifest_contract(asset: Mapping[str, Any]) -> None:
    model = asset.get("model")
    local = asset.get("local_binding")
    canary = asset.get("deterministic_canary")
    if not all(isinstance(value, Mapping) for value in (model, local, canary)):
        raise MultiHopRAGNERError("asset manifest binding is incomplete")
    assert isinstance(model, Mapping)
    assert isinstance(local, Mapping)
    assert isinstance(canary, Mapping)
    expected_canary_inputs = list(synthetic_canary_inputs())
    if (
        model.get("model_id") != MODEL_ID
        or model.get("snapshot_revision") != MODEL_REVISION
        or model.get("architecture") != MODEL_ARCHITECTURE
        or model.get("weight_serialization") != "safetensors"
        or model.get("id2label") != EXPECTED_LABELS
        or asset.get("license") != MODEL_LICENSE
        or asset.get("execution") != EXPECTED_EXECUTION
        or asset.get("aggregation") != EXPECTED_AGGREGATION
        or asset.get("runtime_versions") != EXPECTED_RUNTIME_VERSIONS
        or asset.get("scope") != EXPECTED_SCOPE
        or local.get("runtime_required_paths") != list(MODEL_FILES)
        or local.get("runtime_required_file_count") != len(MODEL_FILES)
        or canary.get("generator_version") != "multihoprag_ner_synthetic_16_v1"
        or canary.get("input_count") != len(expected_canary_inputs)
        or canary.get("input_sha256") != _canonical_hash(expected_canary_inputs)
        or canary.get("repeat_count") != 2
        or canary.get("repeat_exact") is not True
        or canary.get("multihoprag_rows_or_archives_accessed") is not False
        or model.get("weights_sha256") != WEIGHTS_SHA256
        or local.get("snapshot_tree_sha256") != MODEL_TREE_SHA256
        or canary.get("output_sha256") != CANARY_OUTPUT_SHA256
    ):
        raise MultiHopRAGNERError("asset manifest normative contract drifted")
    _require_sha256(model.get("weights_sha256"), "model weights hash")
    _require_sha256(local.get("snapshot_tree_sha256"), "snapshot tree hash")
    _require_sha256(canary.get("output_sha256"), "canary output hash")


def _verify_model_tree(asset: Mapping[str, Any], model_root: str | Path) -> Path:
    root = _reject_symlink_components(Path(model_root), "model root")
    if not root.is_dir():
        raise MultiHopRAGNERError("model root is unavailable")
    entries = list(root.iterdir())
    if any(entry.is_symlink() for entry in entries):
        raise MultiHopRAGNERError("model tree contains a symlink")
    if any(not entry.is_file() for entry in entries):
        raise MultiHopRAGNERError("model tree must contain zero directories")
    if sorted(entry.name for entry in entries) != sorted(MODEL_FILES):
        raise MultiHopRAGNERError("model tree file set drifted")

    local = asset.get("local_binding")
    model = asset.get("model")
    if not isinstance(local, Mapping) or not isinstance(model, Mapping):
        raise MultiHopRAGNERError("model tree binding is absent")
    rows = local.get("snapshot_files")
    if not isinstance(rows, list) or len(rows) != len(MODEL_FILES):
        raise MultiHopRAGNERError("snapshot file manifest is malformed")
    verified: list[dict[str, object]] = []
    total_size = 0
    for expected_name, row in zip(MODEL_FILES, rows):
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256", "size"}:
            raise MultiHopRAGNERError("snapshot file row is malformed")
        if row.get("path") != expected_name:
            raise MultiHopRAGNERError("snapshot file order or path drifted")
        size = row.get("size")
        digest = _require_sha256(row.get("sha256"), "snapshot file hash")
        path = root / expected_name
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or path.stat().st_size != size
            or _sha256_file(path) != digest
        ):
            raise MultiHopRAGNERError("snapshot file content drifted")
        verified.append({"path": expected_name, "sha256": digest, "size": size})
        total_size += size
    if (
        local.get("snapshot_file_count") != len(MODEL_FILES)
        or local.get("snapshot_size_bytes") != total_size
        or local.get("snapshot_tree_sha256") != MODEL_TREE_SHA256
        or _canonical_hash(verified) != MODEL_TREE_SHA256
        or local.get("runtime_required_paths") != list(MODEL_FILES)
        or local.get("runtime_required_file_count") != len(MODEL_FILES)
    ):
        raise MultiHopRAGNERError("snapshot aggregate binding drifted")
    weights = next(row for row in verified if row["path"] == "model.safetensors")
    if weights["sha256"] != WEIGHTS_SHA256 or model.get("weights_sha256") != WEIGHTS_SHA256:
        raise MultiHopRAGNERError("safetensors weight binding drifted")
    return root


def _verify_package_versions() -> dict[str, str]:
    actual: dict[str, str] = {"python": ".".join(map(str, sys.version_info[:3]))}
    for key, (distribution, module_name) in _PACKAGE_TO_MODULE.items():
        try:
            distribution_version = metadata.version(distribution)
            module_version = str(getattr(import_module(module_name), "__version__"))
        except (ImportError, AttributeError, metadata.PackageNotFoundError) as exc:
            raise MultiHopRAGNERError(
                f"required runtime package is missing: {distribution}"
            ) from exc
        if key != "torch" and distribution_version != module_version:
            raise MultiHopRAGNERError("runtime module and distribution versions disagree")
        actual[key] = module_version
    if actual != EXPECTED_RUNTIME_VERSIONS:
        raise MultiHopRAGNERError("installed runtime package versions drifted")
    return actual


def configure_offline_environment() -> None:
    os.environ.update(EXPECTED_EXECUTION["environment"])
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def verify_runtime_binding(
    *,
    asset_manifest_path: str | Path,
    model_root: str | Path,
    verify_package_versions: bool = True,
) -> dict[str, object]:
    """Recompute the manifest, exact six-file tree, and runtime identity."""

    manifest_path, asset = _load_asset_manifest(asset_manifest_path)
    _verify_manifest_contract(asset)
    verified_root = _verify_model_tree(asset, model_root)
    versions = _verify_package_versions() if verify_package_versions else None
    return {
        "asset_file_sha256": ASSET_FILE_SHA256,
        "asset_manifest_path": str(manifest_path),
        "asset_sha256": ASSET_SELF_SHA256,
        "canary_output_sha256": CANARY_OUTPUT_SHA256,
        "model_root": str(verified_root),
        "model_tree_sha256": MODEL_TREE_SHA256,
        "model_revision": MODEL_REVISION,
        "runtime_versions": versions,
        "status": "verified_exact_six_file_offline_ner_runtime",
        "weights_sha256": WEIGHTS_SHA256,
    }


def verify_runtime_asset(
    project_root: str | Path,
    model_root: str | Path | None = None,
) -> dict[str, object]:
    root = _reject_symlink_components(Path(project_root), "project root")
    if not root.is_dir():
        raise MultiHopRAGNERError("project root is unavailable")
    model = (
        root / "artifacts/multihoprag_ner_runtime_v1/model"
        if model_root is None
        else Path(model_root)
    )
    return verify_runtime_binding(
        asset_manifest_path=root / ASSET_RELATIVE_PATH,
        model_root=model,
    )
