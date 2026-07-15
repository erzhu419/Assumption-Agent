from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..models import stable_hash


SEMANTIC_ASSIGNMENT_OPERATOR_VERSION = (
    "minilm_consumed_train_closed_target_ovr_v1"
)
SEMANTIC_ASSIGNMENT_ASSET_VERSION = "semantic_assignment_operator_asset_v1"
SEMANTIC_ASSIGNMENT_RECEIPT_VERSION = "semantic_assignment_operator_receipt_v1"
RUNTIME_ASSET_VERSION = "semantic_assignment_minilm_runtime_asset_v1"
TRAIN_PACK_VERSION = "semantic_assignment_consumed_train_pack_v1"

TARGET_DESTINATIONS = (
    "LLM",
    "trapped_ion_and_qc",
    "black_hole",
    "DNA",
)
PUBLIC_DEFAULT_DESTINATION = "music_history"
ALL_DESTINATIONS = (*TARGET_DESTINATIONS, PUBLIC_DEFAULT_DESTINATION)

EMBEDDING_DIMENSION = 384
PDF_PAGES = 2
MAXIMUM_NORMALIZED_CHARACTERS = 4096
MAXIMUM_EVIDENCE_BYTES = 2 * 1024 * 1024
MAXIMUM_FILES = 512
DEFAULT_SCORE_THRESHOLD = 0.0
FIT_CONFIGURATION: Mapping[str, Any] = {
    "algorithm": "four_independent_binary_logistic_regressions",
    "class_weight": "balanced",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 3000,
    "tol": 1e-8,
    "positive_classes": list(TARGET_DESTINATIONS),
    "negative_class_policy": "all_other_consumed_train_labels",
    "default_rule": "all_target_decision_scores_below_zero",
    "default_score_threshold": DEFAULT_SCORE_THRESHOLD,
    "tie_break": "TARGET_DESTINATIONS_order",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SemanticAssignmentError(RuntimeError):
    pass


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload_hash(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise SemanticAssignmentError(f"{label} is not a sha256 digest")
    return value


def _read_json(path: str | Path, *, maximum: int = 8 * 1024 * 1024) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if resolved.stat().st_size > maximum:
        raise SemanticAssignmentError("JSON input exceeds its byte bound")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SemanticAssignmentError("JSON input is unreadable") from error
    if not isinstance(value, dict):
        raise SemanticAssignmentError("JSON input must be an object")
    return value


def _verify_self_hash(payload: Mapping[str, Any], *, label: str) -> str:
    declared = _require_sha256(payload.get("manifest_hash"), f"{label} manifest hash")
    body = dict(payload)
    del body["manifest_hash"]
    if stable_hash(body) != declared:
        raise SemanticAssignmentError(f"{label} manifest hash mismatch")
    return declared


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode(
        "utf-8"
    ) + b"\n"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def verify_runtime_asset(
    runtime_asset: Mapping[str, Any], *, snapshot_root: str | Path
) -> dict[str, Any]:
    if runtime_asset.get("asset_version") != RUNTIME_ASSET_VERSION:
        raise SemanticAssignmentError("runtime asset version mismatch")
    manifest_hash = _verify_self_hash(runtime_asset, label="runtime asset")
    if runtime_asset.get("embedding_dimension") != EMBEDDING_DIMENSION:
        raise SemanticAssignmentError("runtime embedding dimension mismatch")
    snapshot = Path(snapshot_root).expanduser().resolve(strict=True)
    if snapshot.name != runtime_asset.get("snapshot_revision"):
        raise SemanticAssignmentError("runtime snapshot revision mismatch")
    rows = runtime_asset.get("runtime_required_files")
    if not isinstance(rows, list) or len(rows) != runtime_asset.get(
        "runtime_required_file_count"
    ):
        raise SemanticAssignmentError("runtime file manifest is malformed")
    if stable_hash(rows) != runtime_asset.get("runtime_required_file_set_hash"):
        raise SemanticAssignmentError("runtime file-set hash mismatch")
    total = 0
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "relative_path",
            "sha256",
            "size_bytes",
        }:
            raise SemanticAssignmentError("runtime file row is malformed")
        relative = Path(str(row["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise SemanticAssignmentError("runtime file path is unsafe")
        # Hugging Face snapshots intentionally use symlinks into the adjacent
        # content-addressed blob store.  The manifest-bound relative path is
        # lexical and traversal-free; exact size/SHA validation below binds the
        # resolved blob without pretending it lives beneath the snapshot.
        path = snapshot / relative
        if not path.is_file():
            raise SemanticAssignmentError("runtime file is missing")
        size = row["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise SemanticAssignmentError("runtime file size is invalid")
        if path.stat().st_size != size or _sha256_file(path) != row["sha256"]:
            raise SemanticAssignmentError("runtime file content drifted")
        total += size
    if total != runtime_asset.get("runtime_required_size_bytes"):
        raise SemanticAssignmentError("runtime asset size mismatch")
    weight_rows = [
        row for row in rows if row["relative_path"] == "model.safetensors"
    ]
    if (
        len(weight_rows) != 1
        or weight_rows[0]["sha256"] != runtime_asset.get("weights_sha256")
    ):
        raise SemanticAssignmentError("runtime weights binding mismatch")
    return {
        "runtime_asset_manifest_hash": manifest_hash,
        "runtime_required_file_set_hash": runtime_asset[
            "runtime_required_file_set_hash"
        ],
        "snapshot_revision": runtime_asset["snapshot_revision"],
        "weights_sha256": runtime_asset["weights_sha256"],
    }


def load_operator_asset(path: str | Path) -> dict[str, Any]:
    asset = _read_json(path)
    if asset.get("asset_version") != SEMANTIC_ASSIGNMENT_ASSET_VERSION:
        raise SemanticAssignmentError("operator asset version mismatch")
    _verify_self_hash(asset, label="operator asset")
    if asset.get("operator_version") != SEMANTIC_ASSIGNMENT_OPERATOR_VERSION:
        raise SemanticAssignmentError("operator version mismatch")
    if tuple(asset.get("target_destinations") or ()) != TARGET_DESTINATIONS:
        raise SemanticAssignmentError("operator target order mismatch")
    if asset.get("public_default_destination") != PUBLIC_DEFAULT_DESTINATION:
        raise SemanticAssignmentError("operator public default mismatch")
    if asset.get("fit_configuration") != dict(FIT_CONFIGURATION):
        raise SemanticAssignmentError("operator fit configuration mismatch")
    coefficients = np.asarray(asset.get("coefficients"), dtype=np.float64)
    intercepts = np.asarray(asset.get("intercepts"), dtype=np.float64)
    if coefficients.shape != (len(TARGET_DESTINATIONS), EMBEDDING_DIMENSION):
        raise SemanticAssignmentError("operator coefficient shape mismatch")
    if intercepts.shape != (len(TARGET_DESTINATIONS),):
        raise SemanticAssignmentError("operator intercept shape mismatch")
    if not np.isfinite(coefficients).all() or not np.isfinite(intercepts).all():
        raise SemanticAssignmentError("operator parameters are not finite")
    expected_parameter_hash = _sha256_bytes(
        coefficients.astype("<f8", copy=False).tobytes(order="C")
        + intercepts.astype("<f8", copy=False).tobytes(order="C")
    )
    if expected_parameter_hash != asset.get("parameter_bytes_sha256"):
        raise SemanticAssignmentError("operator parameter hash mismatch")
    expected_candidate_id = stable_hash(
        {
            "operator_version": SEMANTIC_ASSIGNMENT_OPERATOR_VERSION,
            "parameter_bytes_sha256": expected_parameter_hash,
            "train_pack_manifest_hash": asset.get("train_pack_manifest_hash"),
            "runtime_asset_manifest_hash": asset.get(
                "runtime_asset_manifest_hash"
            ),
            "fit_configuration": dict(FIT_CONFIGURATION),
        }
    )
    if asset.get("candidate_id") != expected_candidate_id:
        raise SemanticAssignmentError("operator candidate identity mismatch")
    return asset


class OfflineMiniLMEncoder:
    def __init__(
        self,
        *,
        runtime_asset_path: str | Path,
        snapshot_root: str | Path,
    ) -> None:
        runtime_asset = _read_json(runtime_asset_path)
        self.runtime_receipt = verify_runtime_asset(
            runtime_asset, snapshot_root=snapshot_root
        )
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        try:
            import torch
            import transformers
            import sentence_transformers
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise SemanticAssignmentError("offline embedding runtime is missing") from error
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        declared_versions = runtime_asset.get("runtime_versions")
        actual_versions = {
            "numpy": np.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }
        if not isinstance(declared_versions, dict) or any(
            declared_versions.get(key) != value
            for key, value in actual_versions.items()
        ):
            raise SemanticAssignmentError("runtime dependency version drifted")
        self._model = SentenceTransformer(
            str(Path(snapshot_root).expanduser().resolve(strict=True)),
            local_files_only=True,
            device="cpu",
        )
        canary = runtime_asset.get("deterministic_canary")
        if not isinstance(canary, dict) or not isinstance(canary.get("texts"), list):
            raise SemanticAssignmentError("runtime deterministic canary is missing")
        values = self._model.encode(
            list(canary["texts"]),
            batch_size=len(canary["texts"]),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        matrix = np.asarray(values, dtype="<f4")
        if list(matrix.shape) != canary.get("normalized_embedding_shape") or (
            _sha256_bytes(matrix.tobytes(order="C"))
            != canary.get("little_endian_float32_c_order_sha256")
        ):
            raise SemanticAssignmentError("runtime deterministic canary drifted")

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        values = self._model.encode(
            list(texts),
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.shape != (len(texts), EMBEDDING_DIMENSION):
            raise SemanticAssignmentError("offline encoder returned wrong shape")
        return matrix


def _normalize_extracted_text(raw: str) -> str:
    cleaned = "".join(
        character if character in "\n\t" or ord(character) >= 32 else " "
        for character in raw
    )
    return re.sub(r"\s+", " ", cleaned).strip()[:MAXIMUM_NORMALIZED_CHARACTERS]


def _extract_pdf(path: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="semantic-assignment-fit-") as folder:
        output = Path(folder) / "first-pages.txt"
        try:
            completed = subprocess.run(
                [
                    "pdftotext",
                    "-f",
                    "1",
                    "-l",
                    str(PDF_PAGES),
                    "-nopgbrk",
                    str(path),
                    str(output),
                ],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise SemanticAssignmentError("TRAIN PDF extraction failed") from error
        if completed.returncode != 0 or not output.is_file():
            raise SemanticAssignmentError("TRAIN PDF extraction failed")
        text = _normalize_extracted_text(output.read_text(encoding="utf-8", errors="replace"))
        if not text:
            raise SemanticAssignmentError("TRAIN PDF extraction was empty")
        return text


def fit_operator_asset(
    *,
    train_pack_path: str | Path,
    runtime_asset_path: str | Path,
    snapshot_root: str | Path,
    object_cache: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
    encoder: Callable[[Sequence[str]], np.ndarray] | None = None,
) -> dict[str, Any]:
    train_pack = _read_json(train_pack_path)
    if train_pack.get("manifest_version") != TRAIN_PACK_VERSION:
        raise SemanticAssignmentError("TRAIN pack version mismatch")
    train_pack_hash = _verify_self_hash(train_pack, label="TRAIN pack")
    if train_pack.get("allowed_design_use") != (
        "consumed_train_fit_and_internal_diagnostics_only"
    ) or train_pack.get("prospective_claim_authorized") is not False:
        raise SemanticAssignmentError("TRAIN pack boundary is invalid")
    excluded = train_pack.get("excluded_split_access")
    if not isinstance(excluded, dict) or any(excluded.values()):
        raise SemanticAssignmentError("TRAIN pack accessed an excluded split")
    records = train_pack.get("records")
    if not isinstance(records, list) or len(records) != train_pack.get("record_count"):
        raise SemanticAssignmentError("TRAIN records are malformed")
    if stable_hash(records) != train_pack.get("records_hash"):
        raise SemanticAssignmentError("TRAIN record hash mismatch")
    runtime_asset = _read_json(runtime_asset_path)
    runtime_receipt = verify_runtime_asset(
        runtime_asset, snapshot_root=snapshot_root
    )
    cache = Path(object_cache).expanduser().resolve(strict=True)
    texts: list[str] = []
    labels: list[str] = []
    source_hashes: list[str] = []
    for row in records:
        if not isinstance(row, dict) or row.get("media_type") != "pdf":
            raise SemanticAssignmentError("TRAIN record is not a PDF")
        label = str(row.get("label") or "")
        if label not in ALL_DESTINATIONS:
            raise SemanticAssignmentError("TRAIN label is outside the public set")
        digest = _require_sha256(row.get("content_sha256"), "TRAIN content hash")
        source = (cache / digest).resolve(strict=True)
        if cache not in source.parents:
            raise SemanticAssignmentError("TRAIN object escaped cache")
        size = row.get("size_bytes")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or source.stat().st_size != size
            or _sha256_file(source) != digest
        ):
            raise SemanticAssignmentError("TRAIN object content drifted")
        texts.append(_extract_pdf(source))
        labels.append(label)
        source_hashes.append(digest)
    actual_encoder = encoder or OfflineMiniLMEncoder(
        runtime_asset_path=runtime_asset_path,
        snapshot_root=snapshot_root,
    )
    embeddings = np.asarray(actual_encoder(texts), dtype=np.float32)
    if embeddings.shape != (len(records), EMBEDDING_DIMENSION):
        raise SemanticAssignmentError("TRAIN embedding matrix shape mismatch")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.isfinite(embeddings).all() or not np.allclose(norms, 1.0, atol=1e-5):
        raise SemanticAssignmentError("TRAIN embeddings are not normalized")
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as error:
        raise SemanticAssignmentError("scikit-learn fit runtime is missing") from error
    coefficients: list[list[float]] = []
    intercepts: list[float] = []
    fit_iterations: list[int] = []
    label_array = np.asarray(labels)
    for target in TARGET_DESTINATIONS:
        binary = label_array == target
        classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=3000,
            tol=1e-8,
        ).fit(embeddings, binary)
        coefficients.append(
            np.asarray(classifier.coef_[0], dtype=np.float64).tolist()
        )
        intercepts.append(float(classifier.intercept_[0]))
        fit_iterations.append(int(classifier.n_iter_[0]))
    coefficient_array = np.asarray(coefficients, dtype=np.float64)
    intercept_array = np.asarray(intercepts, dtype=np.float64)
    parameter_hash = _sha256_bytes(
        coefficient_array.astype("<f8", copy=False).tobytes(order="C")
        + intercept_array.astype("<f8", copy=False).tobytes(order="C")
    )
    scores = embeddings.astype(np.float64) @ coefficient_array.T + intercept_array
    best = scores.argmax(axis=1)
    predicted = [
        TARGET_DESTINATIONS[index]
        if float(scores[row, index]) >= DEFAULT_SCORE_THRESHOLD
        else PUBLIC_DEFAULT_DESTINATION
        for row, index in enumerate(best)
    ]
    correct = sum(left == right for left, right in zip(labels, predicted))
    asset: dict[str, Any] = {
        "asset_version": SEMANTIC_ASSIGNMENT_ASSET_VERSION,
        "operator_version": SEMANTIC_ASSIGNMENT_OPERATOR_VERSION,
        "target_destinations": list(TARGET_DESTINATIONS),
        "public_default_destination": PUBLIC_DEFAULT_DESTINATION,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "fit_configuration": dict(FIT_CONFIGURATION),
        "coefficients": coefficient_array.tolist(),
        "intercepts": intercept_array.tolist(),
        "parameter_bytes_sha256": parameter_hash,
        "train_pack_manifest_hash": train_pack_hash,
        "train_records_hash": train_pack["records_hash"],
        "runtime_asset_manifest_hash": runtime_receipt[
            "runtime_asset_manifest_hash"
        ],
        "runtime_required_file_set_hash": runtime_receipt[
            "runtime_required_file_set_hash"
        ],
        "fit_source_object_set_hash": stable_hash(sorted(source_hashes)),
        "fit_record_count": len(records),
        "fit_iterations": fit_iterations,
        "consumed_train_resubstitution_correct": correct,
        "consumed_train_resubstitution_total": len(labels),
        "prospective_claim_authorized": False,
        "raw_extracted_text_persisted": False,
    }
    asset["candidate_id"] = stable_hash(
        {
            "operator_version": SEMANTIC_ASSIGNMENT_OPERATOR_VERSION,
            "parameter_bytes_sha256": parameter_hash,
            "train_pack_manifest_hash": train_pack_hash,
            "runtime_asset_manifest_hash": runtime_receipt[
                "runtime_asset_manifest_hash"
            ],
            "fit_configuration": dict(FIT_CONFIGURATION),
        }
    )
    asset["manifest_hash"] = stable_hash(asset)
    _atomic_write_json(output_path, asset)
    if report_path is not None:
        report = {
            "report_version": "semantic_assignment_operator_fit_report_v1",
            "candidate_id": asset["candidate_id"],
            "operator_asset_manifest_hash": asset["manifest_hash"],
            "train_pack_manifest_hash": train_pack_hash,
            "fit_record_count": len(records),
            "category_counts": train_pack["category_counts"],
            "resubstitution_correct": correct,
            "resubstitution_total": len(labels),
            "fit_iterations": fit_iterations,
            "operator_created_extracted_text_artifact": False,
            "operator_logged_raw_text": False,
            "online_calls": 0,
            "validation_split_accessed": False,
            "sealed_split_accessed": False,
        }
        report["report_hash"] = stable_hash(report)
        _atomic_write_json(report_path, report)
    return asset


def _validated_evidence_rows(
    evidence_payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str], list[int]]:
    if tuple(evidence_payload.get("destinations") or ()) != ALL_DESTINATIONS:
        raise SemanticAssignmentError("evidence destination set mismatch")
    if evidence_payload.get("public_default") != PUBLIC_DEFAULT_DESTINATION:
        raise SemanticAssignmentError("evidence public default mismatch")
    body = dict(evidence_payload)
    declared_set_hash = _require_sha256(
        body.pop("evidence_set_hash", None), "evidence set hash"
    )
    if _payload_hash(body) != declared_set_hash:
        raise SemanticAssignmentError("evidence set hash mismatch")
    contract_hash = _require_sha256(
        evidence_payload.get("contract_hash"), "evidence contract hash"
    )
    files = evidence_payload.get("files")
    if not isinstance(files, list) or not files or len(files) > MAXIMUM_FILES:
        raise SemanticAssignmentError("evidence file count is invalid")
    texts: list[str] = []
    extracted_indices: list[int] = []
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(files):
        if not isinstance(row, dict):
            raise SemanticAssignmentError("evidence file row is malformed")
        file_id = _require_sha256(row.get("file_id"), "evidence file id")
        if file_id in seen:
            raise SemanticAssignmentError("evidence file id is duplicated")
        seen.add(file_id)
        status = row.get("extraction_status")
        items = row.get("evidence")
        if not isinstance(items, list):
            raise SemanticAssignmentError("evidence fragments are malformed")
        normalized.append(dict(row))
        if status == "ok":
            if len(items) != 1 or not isinstance(items[0], dict):
                raise SemanticAssignmentError("extracted file needs one fragment")
            fragment = items[0]
            text = fragment.get("text")
            if not isinstance(text, str) or not text.strip():
                raise SemanticAssignmentError("evidence text is empty")
            raw = text.encode("utf-8")
            if len(raw) > 64 * 1024:
                raise SemanticAssignmentError("evidence text exceeds bound")
            if _sha256_bytes(raw) != fragment.get("text_sha256"):
                raise SemanticAssignmentError("evidence text hash mismatch")
            _require_sha256(fragment.get("evidence_id"), "evidence id")
            texts.append(text)
            extracted_indices.append(index)
        elif status == "unavailable":
            if items:
                raise SemanticAssignmentError("unavailable file has evidence")
        else:
            raise SemanticAssignmentError("evidence extraction status is invalid")
    return normalized, texts, extracted_indices


def build_semantic_assignment_plan(
    *,
    evidence_payload: Mapping[str, Any],
    operator_asset: Mapping[str, Any],
    encoder: Callable[[Sequence[str]], np.ndarray],
    runtime_receipt: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if operator_asset.get("asset_version") != SEMANTIC_ASSIGNMENT_ASSET_VERSION:
        raise SemanticAssignmentError("operator asset is not loaded")
    rows, texts, extracted_indices = _validated_evidence_rows(evidence_payload)
    coefficients = np.asarray(operator_asset["coefficients"], dtype=np.float64)
    intercepts = np.asarray(operator_asset["intercepts"], dtype=np.float64)
    predictions: dict[int, tuple[str, np.ndarray]] = {}
    if texts:
        embeddings = np.asarray(encoder(texts), dtype=np.float32)
        if embeddings.shape != (len(texts), EMBEDDING_DIMENSION):
            raise SemanticAssignmentError("inference embedding shape mismatch")
        if not np.isfinite(embeddings).all():
            raise SemanticAssignmentError("inference embedding is not finite")
        scores = embeddings.astype(np.float64) @ coefficients.T + intercepts
        for local_index, file_index in enumerate(extracted_indices):
            best = int(np.argmax(scores[local_index]))
            destination = (
                TARGET_DESTINATIONS[best]
                if float(scores[local_index, best]) >= DEFAULT_SCORE_THRESHOLD
                else PUBLIC_DEFAULT_DESTINATION
            )
            predictions[file_index] = (destination, scores[local_index])
    assignments: list[dict[str, Any]] = []
    distribution = {destination: 0 for destination in ALL_DESTINATIONS}
    score_rows: list[np.ndarray] = []
    for index, row in enumerate(rows):
        fragments = row["evidence"]
        if index in predictions:
            destination, score_row = predictions[index]
            evidence_ids = [fragments[0]["evidence_id"]]
            basis = "positive_content_evidence"
            score_rows.append(np.asarray(score_row, dtype="<f8"))
        else:
            destination = PUBLIC_DEFAULT_DESTINATION
            evidence_ids = []
            basis = "public_default"
        distribution[destination] += 1
        assignments.append(
            {
                "file_id": row["file_id"],
                "destination": destination,
                "basis": basis,
                "evidence_ids": evidence_ids,
            }
        )
    plan = {
        "contract_hash": evidence_payload["contract_hash"],
        "evidence_set_hash": evidence_payload["evidence_set_hash"],
        "assignments": assignments,
    }
    score_bytes = (
        np.stack(score_rows).astype("<f8", copy=False).tobytes(order="C")
        if score_rows
        else b""
    )
    receipt: dict[str, Any] = {
        "receipt_version": SEMANTIC_ASSIGNMENT_RECEIPT_VERSION,
        "operator_version": SEMANTIC_ASSIGNMENT_OPERATOR_VERSION,
        "candidate_id": operator_asset["candidate_id"],
        "operator_asset_manifest_hash": operator_asset["manifest_hash"],
        "contract_hash": evidence_payload["contract_hash"],
        "evidence_set_hash": evidence_payload["evidence_set_hash"],
        "file_count": len(rows),
        "extracted_file_count": len(extracted_indices),
        "public_default_unavailable_count": len(rows) - len(extracted_indices),
        "destination_distribution": distribution,
        "decision_score_matrix_sha256": _sha256_bytes(score_bytes),
        "plan_hash": stable_hash(plan),
        "agent_plan_used": False,
        "agent_trajectory_received_content": False,
        "operator_created_extracted_text_artifact": False,
        "extracted_text_transport": "bounded_memory_only",
        "operator_logged_raw_text": False,
        "operator_output_contains_raw_text": False,
        "online_calls": 0,
        "verifier_materialized_or_accessed": False,
    }
    if runtime_receipt is not None:
        receipt.update(
            {
                "runtime_asset_manifest_hash": runtime_receipt.get(
                    "runtime_asset_manifest_hash"
                ),
                "runtime_required_file_set_hash": runtime_receipt.get(
                    "runtime_required_file_set_hash"
                ),
            }
        )
    receipt["receipt_hash"] = stable_hash(receipt)
    return plan, receipt


def _command_fit(args: argparse.Namespace) -> int:
    fit_operator_asset(
        train_pack_path=args.train_pack,
        runtime_asset_path=args.runtime_asset,
        snapshot_root=args.snapshot_root,
        object_cache=args.object_cache,
        output_path=args.output,
        report_path=args.report,
    )
    return 0


def _command_classify(args: argparse.Namespace) -> int:
    evidence = _read_json(args.evidence, maximum=MAXIMUM_EVIDENCE_BYTES)
    asset = load_operator_asset(args.operator_asset)
    encoder = OfflineMiniLMEncoder(
        runtime_asset_path=args.runtime_asset,
        snapshot_root=args.snapshot_root,
    )
    plan, receipt = build_semantic_assignment_plan(
        evidence_payload=evidence,
        operator_asset=asset,
        encoder=encoder,
        runtime_receipt=encoder.runtime_receipt,
    )
    _atomic_write_json(args.plan, plan)
    _atomic_write_json(args.receipt, receipt)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit")
    fit.add_argument("--train-pack", required=True)
    fit.add_argument("--runtime-asset", required=True)
    fit.add_argument("--snapshot-root", required=True)
    fit.add_argument("--object-cache", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--report")
    fit.set_defaults(function=_command_fit)
    classify = subparsers.add_parser("classify")
    classify.add_argument("--evidence", required=True)
    classify.add_argument("--operator-asset", required=True)
    classify.add_argument("--runtime-asset", required=True)
    classify.add_argument("--snapshot-root", required=True)
    classify.add_argument("--plan", required=True)
    classify.add_argument("--receipt", required=True)
    classify.set_defaults(function=_command_classify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
